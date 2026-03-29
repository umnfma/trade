#!/usr/bin/env python3
"""
Backtest strategies on historical 1-min data.

Usage:
    python scripts/backtest.py --data data/history_1min.csv
    python scripts/backtest.py --data data/history_1min.csv --strategy regime
    python scripts/backtest.py --data data/history_1min.csv --sweep
"""

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd

from systrade.broker import BacktestBroker
from systrade.engine import Engine
from systrade.feed import HistoricalFeed
from systrade.history import FileHistoryProvider
from systrade.portfolio import Portfolio
from systrade.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from systrade.strategies.regime_adaptive import RegimeAdaptiveStrategy

STARTING_CASH = 1_000_000


def _compute_metrics(portfolio: Portfolio, strategy, label: dict) -> dict:
    """Extract performance metrics from a completed backtest."""
    activity = portfolio.activity()
    df = activity.df()

    if df.empty:
        return {**label, "total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trades": 0}

    total_return = activity.total_return()
    equity = activity.equity_curve()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    returns = equity.pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(390 * 252)

    trades = len(strategy._trading_records)

    return {
        **label,
        "total_return": round(total_return * 100, 3),
        "max_drawdown": round(max_dd * 100, 3),
        "sharpe": round(sharpe, 3),
        "trades": trades,
    }


def run_vwap(data_path: str, entry_z=2.5, exit_z=0.3, rolling_window=20) -> dict:
    """Run VWAP mean reversion backtest."""
    provider = FileHistoryProvider(path=data_path)
    feed = HistoricalFeed(provider=provider)
    broker = BacktestBroker()
    strategy = VWAPMeanReversionStrategy(
        entry_z=entry_z, exit_z=exit_z, rolling_window=rolling_window,
    )
    portfolio = Portfolio(cash=STARTING_CASH, broker=broker)
    engine = Engine(feed=feed, broker=broker, strategy=strategy,
                    cash=STARTING_CASH, portfolio=portfolio)
    engine.run()
    return _compute_metrics(
        portfolio, strategy,
        {"strategy": "vwap", "entry_z": entry_z, "exit_z": exit_z, "window": rolling_window},
    )


def run_regime(
    data_path: str,
    orb_bars: int = 5,
    entry_z: float = 2.0,
    breakout_z: float = 3.5,
    trailing_stop_pct: float = 0.005,
    position_frac: float = 0.80,
) -> dict:
    """Run regime-adaptive backtest."""
    provider = FileHistoryProvider(path=data_path)
    feed = HistoricalFeed(provider=provider)
    broker = BacktestBroker()
    strategy = RegimeAdaptiveStrategy(
        orb_bars=orb_bars,
        entry_z=entry_z,
        breakout_z=breakout_z,
        trailing_stop_pct=trailing_stop_pct,
        position_frac=position_frac,
    )
    portfolio = Portfolio(cash=STARTING_CASH, broker=broker)
    engine = Engine(feed=feed, broker=broker, strategy=strategy,
                    cash=STARTING_CASH, portfolio=portfolio)
    engine.run()
    return _compute_metrics(
        portfolio, strategy,
        {"strategy": "regime", "orb_bars": orb_bars, "entry_z": entry_z,
         "breakout_z": breakout_z, "trail_stop": trailing_stop_pct,
         "pos_frac": position_frac},
    )


def sweep_vwap(data_path: str) -> None:
    """VWAP parameter sweep."""
    results = []
    for ez, xz, w in itertools.product([1.5, 2.0, 2.5], [0.3, 0.5, 0.7], [15, 20, 30]):
        print(f"  vwap entry_z={ez} exit_z={xz} window={w} ... ", end="", flush=True)
        m = run_vwap(data_path, entry_z=ez, exit_z=xz, rolling_window=w)
        results.append(m)
        print(f"return={m['total_return']:.2f}% sharpe={m['sharpe']:.2f}")
    return results


def sweep_regime(data_path: str) -> None:
    """Regime-adaptive parameter sweep."""
    results = []
    combos = list(itertools.product(
        [3, 5, 10],           # orb_bars
        [1.5, 2.0, 2.5],     # entry_z
        [3.0, 3.5, 4.0],     # breakout_z
        [0.003, 0.005, 0.01], # trailing_stop_pct
        [0.50, 0.80],         # position_frac
    ))
    print(f"Running {len(combos)} regime combinations ...\n")
    for ob, ez, bz, ts, pf in combos:
        print(f"  regime orb={ob} ez={ez} bz={bz} ts={ts} pf={pf} ... ", end="", flush=True)
        m = run_regime(data_path, orb_bars=ob, entry_z=ez, breakout_z=bz,
                       trailing_stop_pct=ts, position_frac=pf)
        results.append(m)
        print(f"return={m['total_return']:.2f}% dd={m['max_drawdown']:.2f}% sharpe={m['sharpe']:.2f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest strategies")
    parser.add_argument("--data", required=True, help="Path to 1-min CSV")
    parser.add_argument("--strategy", default="both", choices=["vwap", "regime", "both"],
                        help="Which strategy to run")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: data file not found: {args.data}")
        return

    if args.sweep:
        all_results = []
        if args.strategy in ("vwap", "both"):
            print("=== VWAP PARAMETER SWEEP ===")
            all_results.extend(sweep_vwap(args.data))
        if args.strategy in ("regime", "both"):
            print("\n=== REGIME PARAMETER SWEEP ===")
            all_results.extend(sweep_regime(args.data))

        df = pd.DataFrame(all_results).sort_values("total_return", ascending=False)
        print("\n=== ALL RESULTS (sorted by return) ===")
        print(df.head(20).to_string(index=False))
    else:
        if args.strategy in ("vwap", "both"):
            print("Running VWAP backtest ...")
            m = run_vwap(args.data)
            print(f"  VWAP: return={m['total_return']:.3f}% dd={m['max_drawdown']:.3f}% sharpe={m['sharpe']:.2f} trades={m['trades']}")

        if args.strategy in ("regime", "both"):
            print("Running Regime-Adaptive backtest ...")
            m = run_regime(args.data)
            print(f"  REGIME: return={m['total_return']:.3f}% dd={m['max_drawdown']:.3f}% sharpe={m['sharpe']:.2f} trades={m['trades']}")


if __name__ == "__main__":
    main()
