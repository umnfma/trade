"""
VWAP Mean Reversion ("Rubber Band") Strategy

Intraday strategy that trades deviations from VWAP on liquid large-caps.
Entries on extreme z-scores, exits on mean reversion.  All positions
flattened before close to eliminate overnight risk.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import override
from zoneinfo import ZoneInfo

from systrade.data import Bar, BarData, ExecutionReport
from systrade.strategy import Strategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Time boundaries ──────────────────────────────────────────────────
MARKET_OPEN = time(9, 30)
ENTRY_OPEN = time(10, 0)       # no new entries before 10:00
ENTRY_CLOSE = time(15, 30)     # no new entries after 15:30
FLATTEN_TIME = time(15, 45)    # flatten everything at 15:45

# ── Default parameters ───────────────────────────────────────────────
DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.3
DEFAULT_STOP_Z = 4.5
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_MIN_BARS = 10
DEFAULT_POSITION_FRAC = 0.20   # 20 % of portfolio per position
DEFAULT_MAX_POSITIONS = 4
DEFAULT_DAILY_LOSS_LIMIT = 0.015   # 1.5 %
DEFAULT_PEAK_DD_LIMIT = 0.04       # 4 %
DEFAULT_VOLUME_MULT = 1.5          # volume confirmation threshold
DEFAULT_EMA_PERIOD = 30            # momentum filter EMA bars
DEFAULT_SCALE_FIRST_PCT = 0.60     # first tranche = 60 %


@dataclass
class SymbolState:
    """Per-symbol intraday state — reset each morning."""
    cumulative_volume: float = 0.0
    cumulative_pv: float = 0.0
    vwap: float = 0.0
    deviations: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_ROLLING_WINDOW))
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    bar_count: int = 0
    ema: float = 0.0
    ema_initialized: bool = False
    # Adaptive tracking
    signal_count: int = 0
    win_count: int = 0
    entry_price: float | None = None
    scaled_in: bool = False  # whether we added the second tranche


@dataclass
class RiskState:
    """Portfolio-level risk tracking — reset each morning."""
    start_of_day_equity: float = 0.0
    high_water_mark: float = 0.0
    daily_loss_breaker_tripped: bool = False
    peak_dd_reduction_active: bool = False


class VWAPMeanReversionStrategy(Strategy):
    """
    Intraday VWAP mean-reversion on liquid large-caps.

    Parameters
    ----------
    symbols : tuple of str
        Ticker universe.
    entry_z / exit_z / stop_z : float
        Z-score thresholds for entry, exit, emergency stop.
    rolling_window : int
        Deviation window for std calculation.
    position_frac : float
        Fraction of portfolio per position.
    max_positions : int
        Simultaneous position cap across all symbols.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        entry_z: float = DEFAULT_ENTRY_Z,
        exit_z: float = DEFAULT_EXIT_Z,
        stop_z: float = DEFAULT_STOP_Z,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        min_bars: int = DEFAULT_MIN_BARS,
        position_frac: float = DEFAULT_POSITION_FRAC,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
        peak_dd_limit: float = DEFAULT_PEAK_DD_LIMIT,
        volume_mult: float = DEFAULT_VOLUME_MULT,
        ema_period: int = DEFAULT_EMA_PERIOD,
        scale_first_pct: float = DEFAULT_SCALE_FIRST_PCT,
    ) -> None:
        super().__init__()
        self._symbols = symbols
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._stop_z = stop_z
        self._rolling_window = rolling_window
        self._min_bars = min_bars
        self._position_frac = position_frac
        self._max_positions = max_positions
        self._daily_loss_limit = daily_loss_limit
        self._peak_dd_limit = peak_dd_limit
        self._volume_mult = volume_mult
        self._ema_period = ema_period
        self._scale_first_pct = scale_first_pct

        self._states: dict[str, SymbolState] = {}
        self._risk = RiskState()
        self._open_position_count = 0
        self._last_reset_date: datetime | None = None
        self._trading_records: list[dict] = []

        logger.info(
            "VWAPMeanReversion initialized | symbols=%s entry_z=%.1f exit_z=%.1f",
            symbols, entry_z, exit_z,
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    @override
    def on_start(self) -> None:
        for sym in self._symbols:
            self.subscribe(sym)
            self._states[sym] = SymbolState(
                deviations=deque(maxlen=self._rolling_window)
            )

    @override
    def on_data(self, data: BarData) -> None:
        self.current_time = data.as_of
        now_et = data.as_of.astimezone(ET) if data.as_of.tzinfo else data.as_of
        now_time = now_et.time()

        # ── Daily reset at market open ────────────────────────────────
        if self._should_reset(now_et):
            self._daily_reset(now_et)

        # ── Flatten at 15:45 ET ───────────────────────────────────────
        if now_time >= FLATTEN_TIME:
            self._flatten_all("EOD flatten")
            return

        # ── Portfolio-level risk check ────────────────────────────────
        self._check_risk()
        if self._risk.daily_loss_breaker_tripped:
            return

        # ── Process each symbol ───────────────────────────────────────
        for sym in self._symbols:
            bar = data.get(sym)
            if bar is None:
                continue
            self._process_symbol(sym, bar, now_time)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        sym = report.order.symbol
        state = self._states.get(sym)
        logger.info(
            "FILL %s %+.0f @ %.2f",
            sym, report.last_quantity, report.last_price,
        )
        # Track wins for adaptive thresholds
        if state and state.entry_price is not None:
            if report.order.quantity < 0 and state.entry_price < report.last_price:
                state.win_count += 1  # long closed at profit
            elif report.order.quantity > 0 and state.entry_price > report.last_price:
                state.win_count += 1  # short closed at profit
        self._record_trade(report)

    # ── core logic ────────────────────────────────────────────────────

    def _process_symbol(self, sym: str, bar: Bar, now_time: time) -> None:
        state = self._states[sym]
        price = bar.close
        volume = bar.volume

        # Accumulate VWAP
        state.cumulative_pv += price * volume
        state.cumulative_volume += volume
        state.vwap = (
            state.cumulative_pv / state.cumulative_volume
            if state.cumulative_volume > 0
            else price
        )

        # Update EMA
        state = self._update_ema(state, price)

        # Track deviation
        dev = price - state.vwap
        state.deviations.append(dev)
        state.prices.append(price)
        state.volumes.append(volume)
        state.bar_count += 1

        # Need minimum bars before trading
        if state.bar_count < self._min_bars:
            return

        std = self._std(state.deviations)
        if std < 1e-9:
            return
        z = dev / std

        holding = self.portfolio.is_invested_in(sym)

        # ── Emergency stop ────────────────────────────────────────────
        if holding and abs(z) > self._stop_z:
            self._close_position(sym, "STOP z=%.2f" % z)
            return

        # ── Exit signals ──────────────────────────────────────────────
        if holding:
            pos = self.portfolio.position(sym)
            if pos.qty > 0 and z > -self._exit_z:
                self._close_position(sym, "EXIT LONG z=%.2f" % z)
                return
            if pos.qty < 0 and z < self._exit_z:
                self._close_position(sym, "EXIT SHORT z=%.2f" % z)
                return

            # Scaled entry: add second tranche if deviation deepens
            if not state.scaled_in and state.entry_price is not None:
                if pos.qty > 0 and z < -(self._entry_z + 0.5):
                    self._add_tranche(sym, price, z, side="long")
                elif pos.qty < 0 and z > (self._entry_z + 0.5):
                    self._add_tranche(sym, price, z, side="short")
            return

        # ── Entry guards ──────────────────────────────────────────────
        if not self._can_enter(sym, bar, now_time, z):
            return

        # ── Entry signals ─────────────────────────────────────────────
        entry_z = self._adaptive_entry_z(state)

        if z < -entry_z:
            qty = self._compute_size(price, z, is_first_tranche=True)
            if qty > 0:
                self.post_market_order(sym, quantity=qty)
                state.entry_price = price
                state.scaled_in = False
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("LONG %s qty=%d z=%.2f vwap=%.2f", sym, qty, z, state.vwap)

        elif z > entry_z:
            qty = self._compute_size(price, z, is_first_tranche=True)
            if qty > 0:
                self.post_market_order(sym, quantity=-qty)
                state.entry_price = price
                state.scaled_in = False
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("SHORT %s qty=%d z=%.2f vwap=%.2f", sym, qty, z, state.vwap)

    # ── helpers ───────────────────────────────────────────────────────

    def _can_enter(self, sym: str, bar: Bar, now_time: time, z: float) -> bool:
        """Check all entry guards."""
        # Time window
        if now_time < ENTRY_OPEN or now_time >= ENTRY_CLOSE:
            return False

        # Daily loss breaker
        if self._risk.daily_loss_breaker_tripped:
            return False

        # Max positions
        if self._open_position_count >= self._max_positions:
            return False

        # Volume confirmation
        state = self._states[sym]
        if len(state.volumes) >= 2:
            avg_vol = sum(state.volumes[:-1]) / len(state.volumes[:-1])
            if avg_vol > 0 and bar.volume < self._volume_mult * avg_vol:
                return False

        # Momentum filter: only long in uptrend, short in downtrend
        if state.ema_initialized:
            if z < 0 and bar.close < state.ema:
                return False  # don't buy dips in downtrend
            if z > 0 and bar.close > state.ema:
                return False  # don't short rips in uptrend

        # Correlation guard: skip if 3+ symbols signaling same direction
        same_dir_count = 0
        for s, st in self._states.items():
            if s == sym or st.bar_count < self._min_bars:
                continue
            std_s = self._std(st.deviations)
            if std_s < 1e-9:
                continue
            z_s = (st.prices[-1] - st.vwap) / std_s if st.prices else 0
            if (z < 0 and z_s < -1.5) or (z > 0 and z_s > 1.5):
                same_dir_count += 1
        if same_dir_count >= 3:
            logger.info("SKIP %s — correlation guard (%d peers same dir)", sym, same_dir_count)
            return False

        return True

    def _adaptive_entry_z(self, state: SymbolState) -> float:
        """Tighten or widen entry threshold based on win rate."""
        if state.signal_count < 5:
            return self._entry_z
        win_rate = state.win_count / state.signal_count
        if win_rate > 0.60:
            return max(self._entry_z - 0.5, 1.5)
        if win_rate < 0.40:
            return self._entry_z + 0.5
        return self._entry_z

    def _compute_size(self, price: float, z: float, *, is_first_tranche: bool) -> int:
        """Position size in whole shares."""
        portfolio_value = self.portfolio.value()
        base = (portfolio_value * self._position_frac) / price
        confidence = min(abs(z) / 2.0, 1.5)
        size = base * confidence

        # Peak drawdown reduction
        if self._risk.peak_dd_reduction_active:
            size *= 0.5

        # Scale first tranche to leave room for adding
        if is_first_tranche:
            size *= self._scale_first_pct

        return max(int(math.floor(size)), 0)

    def _add_tranche(self, sym: str, price: float, z: float, *, side: str) -> None:
        """Add the second tranche to deepen a position."""
        state = self._states[sym]
        remaining_pct = 1.0 - self._scale_first_pct
        portfolio_value = self.portfolio.value()
        base = (portfolio_value * self._position_frac) / price
        confidence = min(abs(z) / 2.0, 1.5)
        qty = max(int(math.floor(base * confidence * remaining_pct)), 1)

        if side == "long":
            self.post_market_order(sym, quantity=qty)
        else:
            self.post_market_order(sym, quantity=-qty)
        state.scaled_in = True
        logger.info("SCALE-IN %s %s qty=%d z=%.2f", sym, side.upper(), qty, z)

    def _close_position(self, sym: str, reason: str) -> None:
        """Flatten position in sym."""
        if not self.portfolio.is_invested_in(sym):
            return
        pos = self.portfolio.position(sym)
        self.post_market_order(sym, quantity=-pos.qty)
        self._open_position_count = max(self._open_position_count - 1, 0)
        state = self._states[sym]
        state.entry_price = None
        state.scaled_in = False
        logger.info("CLOSE %s qty=%+.0f reason=%s", sym, -pos.qty, reason)

    def _flatten_all(self, reason: str) -> None:
        """Close every open position."""
        for sym in self._symbols:
            self._close_position(sym, reason)
        self._open_position_count = 0

    # ── Risk checks (called each bar via _process_symbol path) ────────

    def _check_risk(self) -> None:
        """Update risk state — called once per bar cycle."""
        equity = self.portfolio.value()

        # Daily drawdown breaker
        if self._risk.start_of_day_equity > 0:
            daily_dd = (self._risk.start_of_day_equity - equity) / self._risk.start_of_day_equity
            if daily_dd >= self._daily_loss_limit and not self._risk.daily_loss_breaker_tripped:
                self._risk.daily_loss_breaker_tripped = True
                logger.warning("RISK: daily loss limit hit (%.2f%%) — flattening", daily_dd * 100)
                self._flatten_all("daily loss breaker")

        # Peak drawdown reduction
        if equity > self._risk.high_water_mark:
            self._risk.high_water_mark = equity
        if self._risk.high_water_mark > 0:
            peak_dd = (self._risk.high_water_mark - equity) / self._risk.high_water_mark
            self._risk.peak_dd_reduction_active = peak_dd >= self._peak_dd_limit

    # ── Daily reset ───────────────────────────────────────────────────

    def _should_reset(self, now_et: datetime) -> bool:
        if self._last_reset_date is None:
            return True
        return now_et.date() != self._last_reset_date.date()

    def _daily_reset(self, now_et: datetime) -> None:
        """Reset all intraday state for a new trading day."""
        logger.info("=== DAILY RESET %s ===", now_et.date())
        for sym in self._symbols:
            self._states[sym] = SymbolState(
                deviations=deque(maxlen=self._rolling_window)
            )
        equity = self.portfolio.value()
        self._risk = RiskState(
            start_of_day_equity=equity,
            high_water_mark=max(equity, self._risk.high_water_mark) if self._risk.high_water_mark > 0 else equity,
        )
        self._open_position_count = 0
        self._last_reset_date = now_et

    # ── EMA helper ────────────────────────────────────────────────────

    def _update_ema(self, state: SymbolState, price: float) -> SymbolState:
        """Update exponential moving average."""
        k = 2.0 / (self._ema_period + 1)
        if not state.ema_initialized:
            if state.bar_count == 0:
                state.ema = price
            else:
                state.ema = state.ema + (price - state.ema) / (state.bar_count + 1)
            if state.bar_count + 1 >= self._ema_period:
                state.ema_initialized = True
        else:
            state.ema = price * k + state.ema * (1 - k)
        return state

    @staticmethod
    def _std(values) -> float:
        """Population std of an iterable."""
        vals = list(values)
        n = len(vals)
        if n < 2:
            return 0.0
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        return math.sqrt(variance)

    # ── Trade logging ─────────────────────────────────────────────────

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
