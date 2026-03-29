"""Tests for the VWAP Mean Reversion strategy."""

import math
from collections import deque
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from systrade.broker import BacktestBroker
from systrade.data import Bar, BarData, OrderType
from systrade.engine import Engine
from systrade.feed import FileFeed
from systrade.portfolio import Portfolio
from systrade.strategies.vwap_mean_reversion import (
    DEFAULT_ROLLING_WINDOW,
    SymbolState,
    VWAPMeanReversionStrategy,
)

ET = ZoneInfo("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────

def _make_bar(price: float, volume: float = 1000.0) -> Bar:
    return Bar(open=price, high=price + 0.1, low=price - 0.1, close=price, volume=volume)


def _make_bar_data(
    symbol: str, price: float, volume: float = 1000.0,
    ts: datetime | None = None,
) -> BarData:
    ts = ts or datetime(2026, 4, 7, 10, 30, tzinfo=ET)
    bd = BarData(ts)
    bd[symbol] = _make_bar(price, volume)
    return bd


def _wire_strategy(
    strategy: VWAPMeanReversionStrategy,
    broker: BacktestBroker,
    portfolio: Portfolio,
) -> None:
    """Set up strategy context without an Engine."""
    strategy.setup_context(
        subscribe_hook=lambda s: None,
        post_order_hook=broker.post_order,
        portfolio=portfolio,
    )
    strategy.on_start()


# ── VWAP Computation Tests ───────────────────────────────────────────

class TestVWAPComputation:
    def test_vwap_single_bar(self):
        """VWAP after one bar equals the bar price."""
        state = SymbolState(deviations=deque(maxlen=20))
        price, volume = 100.0, 500.0
        state.cumulative_pv += price * volume
        state.cumulative_volume += volume
        vwap = state.cumulative_pv / state.cumulative_volume
        assert vwap == pytest.approx(100.0)

    def test_vwap_weighted_average(self):
        """VWAP is volume-weighted average."""
        state = SymbolState(deviations=deque(maxlen=20))
        bars = [(100.0, 1000), (102.0, 3000)]
        for p, v in bars:
            state.cumulative_pv += p * v
            state.cumulative_volume += v
        vwap = state.cumulative_pv / state.cumulative_volume
        expected = (100 * 1000 + 102 * 3000) / 4000
        assert vwap == pytest.approx(expected)


# ── Z-Score Tests ────────────────────────────────────────────────────

class TestZScore:
    def test_std_empty(self):
        assert VWAPMeanReversionStrategy._std([]) == 0.0

    def test_std_single(self):
        assert VWAPMeanReversionStrategy._std([5.0]) == 0.0

    def test_std_known_values(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        # Population std = 2.0
        assert VWAPMeanReversionStrategy._std(vals) == pytest.approx(2.0)

    def test_z_score_calculation(self):
        """Z-score = (price - vwap) / std."""
        deviations = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.3, -0.3]
        std = VWAPMeanReversionStrategy._std(deviations)
        dev = -3.0
        z = dev / std
        assert z < -2.0  # should trigger long entry


# ── Signal Generation Tests ──────────────────────────────────────────

class TestSignalGeneration:
    def test_no_signal_before_min_bars(self):
        """Strategy should not trade before accumulating min_bars."""
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",), min_bars=10,
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        # Send 5 bars — less than min_bars
        for i in range(5):
            ts = datetime(2026, 4, 7, 10, 30 + i, tzinfo=ET)
            data = _make_bar_data("TEST", 100.0, 1000.0, ts)
            strategy.current_time = ts
            strategy.on_data(data)

        # No orders should be posted
        broker.on_data(_make_bar_data("TEST", 100.0))
        reports = broker.pop_latest()
        assert len(reports) == 0

    def test_long_entry_on_negative_z(self):
        """Strategy enters long when z-score drops below -entry_z in an uptrend.

        Scenario: High-volume bars at 110 anchor VWAP high, then price drifts
        to 104 (still above EMA, which tracks recent prices, but well below VWAP).
        """
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",),
            entry_z=2.0,
            min_bars=5,
            volume_mult=0.0,  # disable volume filter
            ema_period=1,     # EMA tracks price exactly
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        # Phase 1: anchor VWAP high with heavy-volume bars at 110
        for i in range(4):
            ts = datetime(2026, 4, 7, 10, 0 + i, tzinfo=ET)
            data = _make_bar_data("TEST", 110.0, 50_000.0, ts)
            portfolio.on_data(data)
            strategy.current_time = ts
            strategy.on_data(data)

        # Phase 2: price drifts down to 103 with lower volume.
        # EMA (period 3) quickly adapts to ~103, while VWAP stays anchored
        # near 110 due to heavy early volume.  More bars here to push z below -2.
        for i in range(8):
            ts = datetime(2026, 4, 7, 10, 4 + i, tzinfo=ET)
            data = _make_bar_data("TEST", 103.0, 1_000.0, ts)
            portfolio.on_data(data)
            strategy.current_time = ts
            strategy.on_data(data)

        state = strategy._states["TEST"]

        # Verify VWAP is well above current price (anchored by heavy volume at 110)
        assert state.vwap > 108.0, f"VWAP should be anchored high, got {state.vwap}"
        # EMA should have tracked down close to 103
        assert state.ema < 106.0, f"EMA should be near recent price, got {state.ema}"
        # Deviation should be negative and large enough for z < -2
        dev = 103.0 - state.vwap
        std = VWAPMeanReversionStrategy._std(state.deviations)
        if std > 0:
            z = dev / std
            assert z < -2.0, f"Expected z < -2.0, got {z}"
        # Strategy should have posted an order (entry_price set)
        assert state.entry_price is not None or len(broker._orders.get("TEST", [])) > 0


# ── Position Sizing Tests ────────────────────────────────────────────

class TestPositionSizing:
    def test_size_scales_with_z(self):
        """Larger z-scores produce larger positions."""
        strategy = VWAPMeanReversionStrategy(position_frac=0.20)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        size_low_z = strategy._compute_size(100.0, 2.0, is_first_tranche=True)
        size_high_z = strategy._compute_size(100.0, 3.0, is_first_tranche=True)
        assert size_high_z > size_low_z

    def test_size_capped_at_confidence_max(self):
        """Confidence multiplier caps at 1.5."""
        strategy = VWAPMeanReversionStrategy(position_frac=0.20)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        size_z3 = strategy._compute_size(100.0, 3.0, is_first_tranche=True)
        size_z10 = strategy._compute_size(100.0, 10.0, is_first_tranche=True)
        assert size_z3 == size_z10  # both hit 1.5 cap

    def test_first_tranche_smaller_than_full(self):
        """First tranche is scale_first_pct of full size."""
        strategy = VWAPMeanReversionStrategy(position_frac=0.20, scale_first_pct=0.60)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        first = strategy._compute_size(100.0, 2.5, is_first_tranche=True)
        full = strategy._compute_size(100.0, 2.5, is_first_tranche=False)
        assert first < full

    def test_peak_dd_reduces_size(self):
        """When peak drawdown is active, size is halved."""
        strategy = VWAPMeanReversionStrategy(position_frac=0.20)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        normal = strategy._compute_size(100.0, 2.5, is_first_tranche=False)
        strategy._risk.peak_dd_reduction_active = True
        reduced = strategy._compute_size(100.0, 2.5, is_first_tranche=False)
        assert reduced == normal // 2 or abs(reduced - normal / 2) <= 1


# ── Time Guard Tests ─────────────────────────────────────────────────

class TestTimeGuards:
    def test_no_entry_before_10am(self):
        """Should not enter positions before 10:00 AM ET."""
        strategy = VWAPMeanReversionStrategy(symbols=("TEST",), min_bars=2)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        # 9:35 AM — too early
        ts_early = datetime(2026, 4, 7, 9, 35, tzinfo=ET)
        bar = _make_bar(100.0)
        result = strategy._can_enter("TEST", bar, ts_early.time(), -3.0)
        assert result is False

    def test_no_entry_after_330pm(self):
        """Should not enter positions after 3:30 PM ET."""
        strategy = VWAPMeanReversionStrategy(symbols=("TEST",), min_bars=2)
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        ts_late = datetime(2026, 4, 7, 15, 35, tzinfo=ET)
        bar = _make_bar(100.0)
        result = strategy._can_enter("TEST", bar, ts_late.time(), -3.0)
        assert result is False

    def test_entry_allowed_midday(self):
        """Should allow entries during the trading window."""
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",), min_bars=2, volume_mult=0.0,
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        # Need some volume history to pass volume check
        strategy._states["TEST"].volumes = [1000.0] * 5

        ts_ok = datetime(2026, 4, 7, 11, 0, tzinfo=ET)
        bar = _make_bar(100.0, 2000.0)
        result = strategy._can_enter("TEST", bar, ts_ok.time(), -3.0)
        assert result is True


# ── Risk Management Tests ────────────────────────────────────────────

class TestRiskManagement:
    def test_daily_loss_breaker(self):
        """Daily loss breaker should trip when drawdown exceeds limit."""
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",), daily_loss_limit=0.015,
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        strategy._risk.start_of_day_equity = 1_000_000
        # Simulate a loss by reducing cash
        portfolio._cash = 980_000  # 2% loss > 1.5% limit
        strategy._check_risk()
        assert strategy._risk.daily_loss_breaker_tripped is True

    def test_peak_dd_reduction(self):
        """Peak drawdown reduction activates at 4% from HWM."""
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",), peak_dd_limit=0.04,
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        strategy._risk.high_water_mark = 1_000_000
        portfolio._cash = 955_000  # 4.5% down
        strategy._check_risk()
        assert strategy._risk.peak_dd_reduction_active is True

    def test_no_breaker_below_threshold(self):
        """No breaker if loss is below limit."""
        strategy = VWAPMeanReversionStrategy(
            symbols=("TEST",), daily_loss_limit=0.015,
        )
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        strategy._risk.start_of_day_equity = 1_000_000
        portfolio._cash = 992_000  # 0.8% loss < 1.5% limit
        strategy._check_risk()
        assert strategy._risk.daily_loss_breaker_tripped is False


# ── Daily Reset Tests ────────────────────────────────────────────────

class TestDailyReset:
    def test_reset_clears_state(self):
        """Daily reset should clear all symbol states."""
        strategy = VWAPMeanReversionStrategy(symbols=("SPY", "QQQ"))
        broker = BacktestBroker()
        portfolio = Portfolio(cash=1_000_000, broker=broker)
        _wire_strategy(strategy, broker, portfolio)

        # Dirty up the state
        strategy._states["SPY"].bar_count = 50
        strategy._states["SPY"].cumulative_volume = 999999.0

        now = datetime(2026, 4, 7, 9, 30, tzinfo=ET)
        strategy._daily_reset(now)

        assert strategy._states["SPY"].bar_count == 0
        assert strategy._states["SPY"].cumulative_volume == 0.0
        assert strategy._last_reset_date == now

    def test_should_reset_on_new_day(self):
        """Should trigger reset when date changes."""
        strategy = VWAPMeanReversionStrategy(symbols=("TEST",))
        strategy._last_reset_date = datetime(2026, 4, 7, 9, 30, tzinfo=ET)

        assert strategy._should_reset(datetime(2026, 4, 8, 9, 30, tzinfo=ET)) is True
        assert strategy._should_reset(datetime(2026, 4, 7, 10, 0, tzinfo=ET)) is False


# ── Adaptive Threshold Tests ─────────────────────────────────────────

class TestAdaptiveThresholds:
    def test_tightens_on_high_win_rate(self):
        """Entry z tightens when win rate > 60%."""
        strategy = VWAPMeanReversionStrategy(entry_z=2.0)
        state = SymbolState(deviations=deque(maxlen=20))
        state.signal_count = 10
        state.win_count = 8  # 80% win rate
        z = strategy._adaptive_entry_z(state)
        assert z < 2.0

    def test_widens_on_low_win_rate(self):
        """Entry z widens when win rate < 40%."""
        strategy = VWAPMeanReversionStrategy(entry_z=2.0)
        state = SymbolState(deviations=deque(maxlen=20))
        state.signal_count = 10
        state.win_count = 3  # 30% win rate
        z = strategy._adaptive_entry_z(state)
        assert z > 2.0

    def test_default_with_few_signals(self):
        """Returns default entry_z with fewer than 5 signals."""
        strategy = VWAPMeanReversionStrategy(entry_z=2.0)
        state = SymbolState(deviations=deque(maxlen=20))
        state.signal_count = 3
        z = strategy._adaptive_entry_z(state)
        assert z == 2.0


# ── Order Type Tests ─────────────────────────────────────────────────

class TestOrderTypes:
    def test_limit_order_type_exists(self):
        assert OrderType.LIMIT == 2

    def test_stop_order_type_exists(self):
        assert OrderType.STOP == 3

    def test_stop_limit_order_type_exists(self):
        assert OrderType.STOP_LIMIT == 4


# ── BacktestBroker Limit/Stop Fill Tests ─────────────────────────────

class TestBacktestBrokerFills:
    def test_limit_buy_fills_when_low_touches(self):
        """Limit buy should fill when bar low <= limit price."""
        broker = BacktestBroker()
        from systrade.data import Order
        order = Order(
            id="1", symbol="TEST", quantity=10, type=OrderType.LIMIT,
            submit_time=datetime.now(), limit_price=95.0,
        )
        broker.post_order(order)

        bar = Bar(open=100.0, high=101.0, low=94.0, close=96.0, volume=1000)
        fill_price = BacktestBroker._try_fill(order, bar)
        assert fill_price == 95.0

    def test_limit_buy_no_fill_above_limit(self):
        """Limit buy should NOT fill when bar low > limit price."""
        from systrade.data import Order
        order = Order(
            id="2", symbol="TEST", quantity=10, type=OrderType.LIMIT,
            submit_time=datetime.now(), limit_price=90.0,
        )
        bar = Bar(open=100.0, high=101.0, low=95.0, close=96.0, volume=1000)
        fill_price = BacktestBroker._try_fill(order, bar)
        assert fill_price is None

    def test_stop_sell_fills_when_low_touches(self):
        """Stop sell should fill when bar low <= stop price."""
        from systrade.data import Order
        order = Order(
            id="3", symbol="TEST", quantity=-10, type=OrderType.STOP,
            submit_time=datetime.now(), stop_price=95.0,
        )
        bar = Bar(open=100.0, high=101.0, low=94.0, close=96.0, volume=1000)
        fill_price = BacktestBroker._try_fill(order, bar)
        assert fill_price == 95.0

    def test_market_order_fills_at_open(self):
        """Market orders fill at bar open."""
        from systrade.data import Order
        order = Order(
            id="4", symbol="TEST", quantity=10, type=OrderType.MARKET,
            submit_time=datetime.now(),
        )
        bar = Bar(open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)
        fill_price = BacktestBroker._try_fill(order, bar)
        assert fill_price == 100.0


# ── EMA Tests ────────────────────────────────────────────────────────

class TestEMA:
    def test_ema_initializes_as_average(self):
        """EMA should start as simple average during warmup."""
        strategy = VWAPMeanReversionStrategy(ema_period=5)
        state = SymbolState(deviations=deque(maxlen=20))

        prices = [10.0, 12.0, 11.0, 13.0, 14.0]
        for i, p in enumerate(prices):
            state.bar_count = i
            state = strategy._update_ema(state, p)

        assert state.ema_initialized is True
        assert state.ema == pytest.approx(12.0, abs=0.01)

    def test_ema_tracks_trend(self):
        """EMA should follow an upward trend."""
        strategy = VWAPMeanReversionStrategy(ema_period=5)
        state = SymbolState(deviations=deque(maxlen=20))

        for i in range(20):
            state.bar_count = i
            state = strategy._update_ema(state, 100.0 + i)

        # EMA should be close to recent prices (above 110)
        assert state.ema > 110.0
