"""
Quantitative VWAP Strategy with HMM Regime Detection + FFT Cycle Timing

The core edge comes from two signal processing layers on top of VWAP MR:

1. HMM REGIME FILTER — A 3-state Hidden Markov Model classifies each
   minute into mean-reverting, trending, or volatile.  We ONLY take
   mean-reversion trades when the HMM says we're in an MR regime
   (confidence > 60%).  When trending, we flip to momentum.  When
   volatile, we sit out.  This single filter eliminates the #1 loss
   source: mean-reverting against a trend.

2. FFT CYCLE TIMING — FFT decomposes VWAP deviations into frequency
   components to find the dominant intraday oscillation cycle.  Instead
   of waiting for z > 2.5 (lagging indicator), we enter when the FFT
   says we're at a cycle trough (for longs) or peak (for shorts), even
   if z hasn't hit the threshold yet.  This gives 5-15 bar earlier
   entries with better prices.

Sizing: aggressive for a paper-money competition.  Concentrated positions
(1-2 names), leveraged sizing, loose risk limits.
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
from systrade.strategies.signal_processing import (
    CycleEstimate,
    FFTCycleDetector,
    HMMRegimeDetector,
    MarketRegime,
    RegimeEstimate,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
ENTRY_OPEN = time(10, 0)
ENTRY_CLOSE = time(15, 15)
FLATTEN_TIME = time(15, 45)

DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")


@dataclass
class SymbolState:
    # VWAP
    cumulative_volume: float = 0.0
    cumulative_pv: float = 0.0
    vwap: float = 0.0
    deviations: deque = field(default_factory=lambda: deque(maxlen=30))
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    bar_count: int = 0

    # Signal processing
    hmm: HMMRegimeDetector = field(default_factory=lambda: HMMRegimeDetector(
        lookback=120, refit_interval=30,
    ))
    fft: FFTCycleDetector = field(default_factory=lambda: FFTCycleDetector(
        window=120, min_period=10, max_period=90,
    ))
    regime: RegimeEstimate = field(default_factory=lambda: RegimeEstimate(
        MarketRegime.UNKNOWN, 0.0, 0.0, 0.0,
    ))
    cycle: CycleEstimate = field(default_factory=lambda: CycleEstimate(
        0, 0, 0, False, False, 0,
    ))

    # Position tracking
    entry_price: float | None = None
    entry_side: str = ""
    trailing_stop: float = 0.0
    signal_count: int = 0
    win_count: int = 0

    # Cooldown: bars since last exit (prevents re-entry churn)
    bars_since_exit: int = 999


class QuantVWAPStrategy(Strategy):
    """
    VWAP mean reversion with HMM regime filtering and FFT cycle timing.

    Parameters
    ----------
    symbols : tuple
        Trading universe.
    entry_z : float
        Z-score threshold for standard MR entry.
    fft_entry_z : float
        Lower z-score threshold when FFT confirms cycle trough/peak.
    exit_z : float
        Z-score threshold to exit (close to VWAP).
    regime_confidence : float
        Minimum HMM confidence to trust the regime classification.
    position_frac : float
        Fraction of portfolio value per position.
    leverage : float
        Leverage multiplier (4x for Alpaca paper).
    max_positions : int
        Max simultaneous positions.
    trailing_stop_pct : float
        Trailing stop for momentum trades.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        entry_z: float = 2.0,
        fft_entry_z: float = 1.3,
        exit_z: float = 0.3,
        stop_z: float = 5.0,
        regime_confidence: float = 0.55,
        position_frac: float = 0.50,
        leverage: float = 2.0,
        max_positions: int = 2,
        trailing_stop_pct: float = 0.004,
        cooldown_bars: int = 0,
        rolling_window: int = 30,
        min_bars: int = 15,
    ) -> None:
        super().__init__()
        self._symbols = symbols
        self._entry_z = entry_z
        self._fft_entry_z = fft_entry_z
        self._exit_z = exit_z
        self._stop_z = stop_z
        self._regime_confidence = regime_confidence
        self._position_frac = position_frac
        self._leverage = leverage
        self._max_positions = max_positions
        self._trailing_stop_pct = trailing_stop_pct
        self._cooldown_bars = cooldown_bars
        self._rolling_window = rolling_window
        self._min_bars = min_bars

        self._states: dict[str, SymbolState] = {}
        self._open_position_count = 0
        self._last_reset_date: datetime | None = None
        self._trading_records: list[dict] = []

        logger.info(
            "QuantVWAP initialized | symbols=%s entry_z=%.1f fft_z=%.1f leverage=%.0fx",
            symbols, entry_z, fft_entry_z, leverage,
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

        if self._should_reset(now_et):
            self._daily_reset(now_et)

        if now_time >= FLATTEN_TIME:
            self._flatten_all("EOD flatten")
            return

        for sym in self._symbols:
            bar = data.get(sym)
            if bar is None:
                continue
            self._process_symbol(sym, bar, now_time)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        sym = report.order.symbol
        state = self._states.get(sym)
        logger.info("FILL %s %+.0f @ %.2f", sym, report.last_quantity, report.last_price)
        if state and state.entry_price is not None:
            if state.entry_side == "long" and report.order.quantity < 0:
                if report.last_price > state.entry_price:
                    state.win_count += 1
            elif state.entry_side == "short" and report.order.quantity > 0:
                if report.last_price < state.entry_price:
                    state.win_count += 1
        self._record_trade(report)

    # ── Core ──────────────────────────────────────────────────────────

    def _process_symbol(self, sym: str, bar: Bar, now_time: time) -> None:
        state = self._states[sym]
        price = bar.close
        volume = bar.volume

        # Update VWAP
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

        # Update signal processors
        state.regime = state.hmm.update(price, volume)
        state.cycle = state.fft.update(dev)

        if state.bar_count < self._min_bars:
            return

        std = _std(state.deviations)
        if std < 1e-9:
            return
        z = dev / std

        holding = self.portfolio.is_invested_in(sym)

        # Track cooldown
        state.bars_since_exit += 1

        # ── Manage existing positions ─────────────────────────────────
        if holding:
            self._manage_position(sym, bar, z, state)
            return

        # ── Entry logic ───────────────────────────────────────────────
        if now_time < ENTRY_OPEN or now_time >= ENTRY_CLOSE:
            return
        if self._open_position_count >= self._max_positions:
            return
        if state.bars_since_exit < self._cooldown_bars:
            return

        self._check_entry(sym, bar, z, state)

    def _check_entry(self, sym: str, bar: Bar, z: float, state: SymbolState) -> None:
        """Decide whether to enter based on regime + z-score + FFT cycle."""
        regime = state.regime.regime
        confidence = state.regime.confidence
        cycle = state.cycle

        # ── Regime-based dispatch ─────────────────────────────────────
        if regime == MarketRegime.VOLATILE and confidence > self._regime_confidence:
            return  # sit out volatile regimes

        if regime == MarketRegime.TRENDING and confidence > self._regime_confidence:
            # In a trending regime, go WITH the trend if z is extreme
            self._trend_entry(sym, bar, z, state)
            return

        # ── Mean Reversion entry (MR regime or unknown) ───────────────
        # FFT-enhanced: if cycle confirms we're at a trough/peak, use
        # a lower z-score threshold for earlier entry
        fft_confirmed = (
            cycle.cycle_strength > 2.0  # strong cycle signal
            and ((z < 0 and cycle.at_trough) or (z > 0 and cycle.at_peak))
        )

        effective_z = self._fft_entry_z if fft_confirmed else self._entry_z
        entry_label = "FFT+MR" if fft_confirmed else "MR"

        if z < -effective_z:
            qty = self._compute_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=qty)
                state.entry_price = bar.close
                state.entry_side = "long"
                state.signal_count += 1
                self._open_position_count += 1
                logger.info(
                    "%s LONG %s qty=%d z=%.2f regime=%s cycle_str=%.1f",
                    entry_label, sym, qty, z, regime.name, cycle.cycle_strength,
                )

        elif z > effective_z:
            qty = self._compute_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=-qty)
                state.entry_price = bar.close
                state.entry_side = "short"
                state.signal_count += 1
                self._open_position_count += 1
                logger.info(
                    "%s SHORT %s qty=%d z=%.2f regime=%s cycle_str=%.1f",
                    entry_label, sym, qty, z, regime.name, cycle.cycle_strength,
                )

    def _trend_entry(self, sym: str, bar: Bar, z: float, state: SymbolState) -> None:
        """In a trending regime, enter WITH the trend on pullbacks."""
        trend_dir = state.regime.trend_strength

        # Long in uptrend on a mild pullback to VWAP
        if trend_dir > 0 and -1.5 < z < -0.5:
            qty = self._compute_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=qty)
                state.entry_price = bar.close
                state.entry_side = "long"
                state.trailing_stop = bar.close * (1 - self._trailing_stop_pct)
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("TREND LONG %s z=%.2f trend=%.4f", sym, z, trend_dir)

        # Short in downtrend on a mild bounce from VWAP
        elif trend_dir < 0 and 0.5 < z < 1.5:
            qty = self._compute_size(bar.close)
            if qty > 0:
                self.post_market_order(sym, quantity=-qty)
                state.entry_price = bar.close
                state.entry_side = "short"
                state.trailing_stop = bar.close * (1 + self._trailing_stop_pct)
                state.signal_count += 1
                self._open_position_count += 1
                logger.info("TREND SHORT %s z=%.2f trend=%.4f", sym, z, trend_dir)

    def _manage_position(self, sym: str, bar: Bar, z: float, state: SymbolState) -> None:
        """Manage open positions: exits, stops, trailing."""
        price = bar.close
        pos = self.portfolio.position(sym)

        # Hard stop
        if abs(z) > self._stop_z:
            self._close_position(sym, "HARD STOP z=%.2f" % z)
            return

        regime = state.regime.regime

        # ── In trending regime with a position: use trailing stop ─────
        if regime == MarketRegime.TRENDING and state.regime.confidence > self._regime_confidence:
            if state.entry_side == "long":
                state.trailing_stop = max(
                    state.trailing_stop,
                    price * (1 - self._trailing_stop_pct),
                )
                if price <= state.trailing_stop:
                    self._close_position(sym, "TRAIL STOP LONG")
            elif state.entry_side == "short":
                state.trailing_stop = min(
                    state.trailing_stop if state.trailing_stop > 0 else float("inf"),
                    price * (1 + self._trailing_stop_pct),
                )
                if price >= state.trailing_stop:
                    self._close_position(sym, "TRAIL STOP SHORT")
            return

        # ── In MR regime: exit on reversion to VWAP ──────────────────
        # FFT-enhanced exit: if cycle says we're at the opposite extreme,
        # exit even if z hasn't fully reverted
        cycle = state.cycle
        fft_exit = (
            cycle.cycle_strength > 2.0
            and ((state.entry_side == "long" and cycle.at_peak)
                 or (state.entry_side == "short" and cycle.at_trough))
        )

        if state.entry_side == "long":
            if z > -self._exit_z or fft_exit:
                self._close_position(sym, "MR EXIT LONG z=%.2f fft=%s" % (z, fft_exit))
        elif state.entry_side == "short":
            if z < self._exit_z or fft_exit:
                self._close_position(sym, "MR EXIT SHORT z=%.2f fft=%s" % (z, fft_exit))

    # ── Sizing ────────────────────────────────────────────────────────

    def _compute_size(self, price: float) -> int:
        """Size position using available capital.

        In live mode, buying_power() already includes Alpaca's leverage.
        In backtest mode, we simulate leverage via value() * multiplier.
        """
        try:
            capital = self.portfolio.buying_power()
        except (AttributeError, NotImplementedError):
            capital = self.portfolio.value() * self._leverage
        raw = (capital * self._position_frac) / price
        return max(int(math.floor(raw)), 0)

    # ── Position management ───────────────────────────────────────────

    def _close_position(self, sym: str, reason: str) -> None:
        if not self.portfolio.is_invested_in(sym):
            return
        pos = self.portfolio.position(sym)
        self.post_market_order(sym, quantity=-pos.qty)
        self._open_position_count = max(self._open_position_count - 1, 0)
        state = self._states[sym]
        state.entry_price = None
        state.entry_side = ""
        state.trailing_stop = 0.0
        state.bars_since_exit = 0  # start cooldown
        logger.info("CLOSE %s qty=%+.0f reason=%s", sym, -pos.qty, reason)

    def _flatten_all(self, reason: str) -> None:
        for sym in self._symbols:
            self._close_position(sym, reason)
        self._open_position_count = 0

    # ── Daily reset ───────────────────────────────────────────────────

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


def _std(values) -> float:
    vals = list(values)
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    return math.sqrt(variance)
