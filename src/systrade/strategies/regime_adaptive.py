"""
Regime-Adaptive "Rubber Band Snap" Strategy

Three regimes detected and traded differently each day:

1. OPENING RANGE BREAKOUT (9:30–10:00)
   Build a price range from the first N bars, then trade the breakout
   direction with full conviction.  ORB captures the day's biggest
   directional move — most daily range is established in the first hour.

2. VWAP MEAN REVERSION (10:00–15:00)
   Standard rubber-band: enter on extreme z-scores, exit on reversion.

3. MOMENTUM FLIP (any time z > breakout_z and accelerating)
   When a mean-reversion setup FAILS — the rubber band snaps — that IS
   the signal.  Close the MR position and ride the breakout with a
   trailing stop.  This single change captures the fat tails that pure
   MR strategies leave on the table.

Sizing philosophy: this is a paper-money competition.  Alpaca gives 4×
intraday buying power.  We concentrate in 1–2 positions instead of
spreading thin across 5.  80% of buying power per trade.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from typing import override
from zoneinfo import ZoneInfo

from systrade.data import Bar, BarData, ExecutionReport
from systrade.strategy import Strategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Time boundaries ──────────────────────────────────────────────────
MARKET_OPEN = time(9, 30)
ORB_END = time(10, 0)
ENTRY_CLOSE = time(15, 0)
FLATTEN_TIME = time(15, 45)

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")
DEFAULT_ORB_BARS = 5           # 5-min opening range
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.3
DEFAULT_BREAKOUT_Z = 3.5       # z threshold to flip from MR → momentum
DEFAULT_TRAILING_STOP_PCT = 0.005  # 0.5% trailing stop for momentum
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_MIN_BARS = 10
DEFAULT_POSITION_FRAC = 0.80   # 80% of capital per trade (aggressive)
DEFAULT_MAX_POSITIONS = 2      # concentrate in 1-2 names
DEFAULT_LEVERAGE = 4.0         # Alpaca paper gives 4x intraday


class Regime(Enum):
    ORB = auto()
    MEAN_REVERSION = auto()
    MOMENTUM = auto()


@dataclass
class SymbolState:
    """Per-symbol intraday state."""
    # VWAP
    cumulative_volume: float = 0.0
    cumulative_pv: float = 0.0
    vwap: float = 0.0
    deviations: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_WINDOW))
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    bar_count: int = 0

    # ORB
    orb_high: float = 0.0
    orb_low: float = float("inf")
    orb_bars_seen: int = 0
    orb_range_set: bool = False
    orb_traded: bool = False

    # Momentum flip
    regime: Regime = Regime.ORB
    prev_z: float = 0.0
    trailing_stop: float = 0.0
    entry_price: float | None = None
    entry_side: str = ""  # "long" or "short"

    # Tracking
    signal_count: int = 0
    win_count: int = 0


class RegimeAdaptiveStrategy(Strategy):
    """
    Three-regime intraday strategy: ORB → VWAP MR → Momentum flip.

    Designed for maximum returns in a paper-money competition.
    Uses 4x intraday leverage via aggressive position sizing.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        orb_bars: int = DEFAULT_ORB_BARS,
        entry_z: float = DEFAULT_ENTRY_Z,
        exit_z: float = DEFAULT_EXIT_Z,
        breakout_z: float = DEFAULT_BREAKOUT_Z,
        trailing_stop_pct: float = DEFAULT_TRAILING_STOP_PCT,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        min_bars: int = DEFAULT_MIN_BARS,
        position_frac: float = DEFAULT_POSITION_FRAC,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        leverage: float = DEFAULT_LEVERAGE,
    ) -> None:
        super().__init__()
        self._symbols = symbols
        self._orb_bars = orb_bars
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._breakout_z = breakout_z
        self._trailing_stop_pct = trailing_stop_pct
        self._rolling_window = rolling_window
        self._min_bars = min_bars
        self._position_frac = position_frac
        self._max_positions = max_positions
        self._leverage = leverage

        self._states: dict[str, SymbolState] = {}
        self._open_position_count = 0
        self._last_reset_date: datetime | None = None
        self._trading_records: list[dict] = []

        logger.info(
            "RegimeAdaptive initialized | symbols=%s orb_bars=%d entry_z=%.1f breakout_z=%.1f",
            symbols, orb_bars, entry_z, breakout_z,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────

    @override
    def on_start(self) -> None:
        for sym in self._symbols:
            self.subscribe(sym)
            self._states[sym] = SymbolState(
                deviations=deque(maxlen=self._rolling_window),
            )

    @override
    def on_data(self, data: BarData) -> None:
        self.current_time = data.as_of
        now_et = data.as_of.astimezone(ET) if data.as_of.tzinfo else data.as_of
        now_time = now_et.time()

        # Daily reset
        if self._should_reset(now_et):
            self._daily_reset(now_et)

        # EOD flatten
        if now_time >= FLATTEN_TIME:
            self._flatten_all("EOD flatten")
            return

        for sym in self._symbols:
            bar = data.get(sym)
            if bar is None:
                continue
            self._update_vwap(sym, bar)
            self._dispatch(sym, bar, now_time)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        sym = report.order.symbol
        state = self._states.get(sym)
        logger.info(
            "FILL %s %+.0f @ %.2f", sym, report.last_quantity, report.last_price,
        )
        if state and state.entry_price is not None:
            pnl = (report.last_price - state.entry_price) * report.order.quantity
            if pnl > 0:
                state.win_count += 1
        self._record_trade(report)

    # ── Dispatcher ────────────────────────────────────────────────────

    def _dispatch(self, sym: str, bar: Bar, now_time: time) -> None:
        state = self._states[sym]

        # Phase 1: Build ORB range
        if not state.orb_range_set:
            self._build_orb(sym, bar)
            return

        # Phase 2: ORB breakout (9:30 – 10:00)
        if now_time < ORB_END and not state.orb_traded:
            self._check_orb_breakout(sym, bar)
            return

        # Transition to MR regime after ORB window
        if state.regime == Regime.ORB and now_time >= ORB_END:
            state.regime = Regime.MEAN_REVERSION

        # Check momentum flip for existing positions
        if state.regime == Regime.MOMENTUM:
            self._manage_momentum(sym, bar)
            return

        # Phase 3: VWAP Mean Reversion
        if state.regime == Regime.MEAN_REVERSION:
            self._vwap_mean_reversion(sym, bar, now_time)

    # ── ORB ───────────────────────────────────────────────────────────

    def _build_orb(self, sym: str, bar: Bar) -> None:
        state = self._states[sym]
        state.orb_high = max(state.orb_high, bar.high)
        state.orb_low = min(state.orb_low, bar.low)
        state.orb_bars_seen += 1
        if state.orb_bars_seen >= self._orb_bars:
            state.orb_range_set = True
            logger.info(
                "ORB RANGE %s high=%.2f low=%.2f range=%.2f",
                sym, state.orb_high, state.orb_low,
                state.orb_high - state.orb_low,
            )

    def _check_orb_breakout(self, sym: str, bar: Bar) -> None:
        state = self._states[sym]
        if self._open_position_count >= self._max_positions:
            return

        # Pick the symbol with the widest ORB range (strongest signal)
        # Only enter if this breakout is clean (close beyond range)
        if bar.close > state.orb_high:
            qty = self._aggressive_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=qty)
                state.orb_traded = True
                state.entry_price = bar.close
                state.entry_side = "long"
                state.trailing_stop = bar.close * (1 - self._trailing_stop_pct * 2)
                state.regime = Regime.MOMENTUM  # ORB breakouts are momentum
                self._open_position_count += 1
                state.signal_count += 1
                logger.info(
                    "ORB LONG %s @ %.2f (range %.2f–%.2f)",
                    sym, bar.close, state.orb_low, state.orb_high,
                )

        elif bar.close < state.orb_low:
            qty = self._aggressive_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=-qty)
                state.orb_traded = True
                state.entry_price = bar.close
                state.entry_side = "short"
                state.trailing_stop = bar.close * (1 + self._trailing_stop_pct * 2)
                state.regime = Regime.MOMENTUM
                self._open_position_count += 1
                state.signal_count += 1
                logger.info(
                    "ORB SHORT %s @ %.2f (range %.2f–%.2f)",
                    sym, bar.close, state.orb_low, state.orb_high,
                )

    # ── VWAP Mean Reversion ──────────────────────────────────────────

    def _vwap_mean_reversion(self, sym: str, bar: Bar, now_time: time) -> None:
        state = self._states[sym]
        price = bar.close

        if state.bar_count < self._min_bars:
            return

        std = _std(state.deviations)
        if std < 1e-9:
            return
        dev = price - state.vwap
        z = dev / std

        holding = self.portfolio.is_invested_in(sym)

        # ── Momentum flip detection ───────────────────────────────────
        # If z is extreme AND accelerating, the rubber band snapped
        if holding and abs(z) > self._breakout_z:
            z_accel = abs(z) - abs(state.prev_z)
            if z_accel > 0.3:  # z is increasing, not mean-reverting
                self._flip_to_momentum(sym, bar, z)
                state.prev_z = z
                return

        state.prev_z = z

        # ── Exit ──────────────────────────────────────────────────────
        if holding:
            pos = self.portfolio.position(sym)
            if pos.qty > 0 and z > -self._exit_z:
                self._close_position(sym, "MR EXIT LONG z=%.2f" % z)
                return
            if pos.qty < 0 and z < self._exit_z:
                self._close_position(sym, "MR EXIT SHORT z=%.2f" % z)
                return
            # Hard stop
            if abs(z) > 5.0:
                self._close_position(sym, "HARD STOP z=%.2f" % z)
                return
            return

        # ── Entry ─────────────────────────────────────────────────────
        if now_time >= ENTRY_CLOSE:
            return
        if self._open_position_count >= self._max_positions:
            return

        if z < -self._entry_z:
            qty = self._aggressive_size(price)
            if qty > 0:
                self.post_market_order(sym, quantity=qty)
                state.entry_price = price
                state.entry_side = "long"
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("MR LONG %s qty=%d z=%.2f", sym, qty, z)

        elif z > self._entry_z:
            qty = self._aggressive_size(price)
            if qty > 0:
                self.post_market_order(sym, quantity=-qty)
                state.entry_price = price
                state.entry_side = "short"
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("MR SHORT %s qty=%d z=%.2f", sym, qty, z)

    # ── Momentum flip ─────────────────────────────────────────────────

    def _flip_to_momentum(self, sym: str, bar: Bar, z: float) -> None:
        """Close MR position and re-enter in breakout direction."""
        state = self._states[sym]
        pos = self.portfolio.position(sym)

        # Close existing MR position
        self.post_market_order(sym, quantity=-pos.qty)
        logger.info("FLIP %s closing MR pos %+.0f, z=%.2f", sym, pos.qty, z)

        # Re-enter in the breakout direction (same direction z is moving)
        qty = self._aggressive_size(bar.close)
        if qty > 0:
            if z > 0:  # price above VWAP and accelerating up
                self.post_market_order(sym, quantity=qty)
                state.entry_side = "long"
                state.trailing_stop = bar.close * (1 - self._trailing_stop_pct)
            else:  # price below VWAP and accelerating down
                self.post_market_order(sym, quantity=-qty)
                state.entry_side = "short"
                state.trailing_stop = bar.close * (1 + self._trailing_stop_pct)
            state.entry_price = bar.close
            state.regime = Regime.MOMENTUM
            state.signal_count += 1
            logger.info(
                "MOMENTUM %s %s @ %.2f z=%.2f",
                sym, state.entry_side.upper(), bar.close, z,
            )

    def _manage_momentum(self, sym: str, bar: Bar) -> None:
        """Trail stop on momentum positions."""
        state = self._states[sym]
        if not self.portfolio.is_invested_in(sym):
            state.regime = Regime.MEAN_REVERSION
            return

        price = bar.close

        # Update trailing stop
        if state.entry_side == "long":
            new_stop = price * (1 - self._trailing_stop_pct)
            state.trailing_stop = max(state.trailing_stop, new_stop)
            if price <= state.trailing_stop:
                self._close_position(sym, "TRAIL STOP LONG @ %.2f" % price)
                state.regime = Regime.MEAN_REVERSION
        elif state.entry_side == "short":
            new_stop = price * (1 + self._trailing_stop_pct)
            state.trailing_stop = min(state.trailing_stop, new_stop)
            if price >= state.trailing_stop:
                self._close_position(sym, "TRAIL STOP SHORT @ %.2f" % price)
                state.regime = Regime.MEAN_REVERSION

    # ── Helpers ───────────────────────────────────────────────────────

    def _update_vwap(self, sym: str, bar: Bar) -> None:
        state = self._states[sym]
        price = bar.close
        volume = bar.volume

        state.cumulative_pv += price * volume
        state.cumulative_volume += volume
        state.vwap = (
            state.cumulative_pv / state.cumulative_volume
            if state.cumulative_volume > 0 else price
        )

        dev = price - state.vwap
        state.deviations.append(dev)
        state.prices.append(price)
        state.volumes.append(volume)
        state.bar_count += 1

    def _aggressive_size(self, price: float) -> int:
        """Size using available capital — buying_power in live, simulated in backtest."""
        try:
            capital = self.portfolio.buying_power()
        except (AttributeError, NotImplementedError):
            capital = self.portfolio.value() * self._leverage
        raw = (capital * self._position_frac) / price
        return max(int(math.floor(raw)), 0)

    def _close_position(self, sym: str, reason: str) -> None:
        if not self.portfolio.is_invested_in(sym):
            return
        pos = self.portfolio.position(sym)
        self.post_market_order(sym, quantity=-pos.qty)
        self._open_position_count = max(self._open_position_count - 1, 0)
        state = self._states[sym]
        state.entry_price = None
        state.entry_side = ""
        logger.info("CLOSE %s qty=%+.0f reason=%s", sym, -pos.qty, reason)

    def _flatten_all(self, reason: str) -> None:
        for sym in self._symbols:
            self._close_position(sym, reason)
        self._open_position_count = 0

    def _should_reset(self, now_et: datetime) -> bool:
        if self._last_reset_date is None:
            return True
        return now_et.date() != self._last_reset_date.date()

    def _daily_reset(self, now_et: datetime) -> None:
        logger.info("=== DAILY RESET %s ===", now_et.date())
        for sym in self._symbols:
            self._states[sym] = SymbolState(
                deviations=deque(maxlen=self._rolling_window),
            )
        self._open_position_count = 0
        self._last_reset_date = now_et

    def _record_trade(self, report: ExecutionReport) -> None:
        record = {
            "timestamp": report.fill_timestamp.isoformat() if report.fill_timestamp else "",
            "symbol": report.order.symbol,
            "side": "BUY" if report.order.quantity > 0 else "SELL",
            "quantity": abs(report.order.quantity),
            "price": report.last_price,
        }
        self._trading_records.append(record)
        with open("trading_results.json", "a") as f:
            f.write(json.dumps(record) + "\n")


def _std(values) -> float:
    vals = list(values)
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    return math.sqrt(variance)
