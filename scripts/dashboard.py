#!/usr/bin/env python3
"""
Live dashboard — shows current positions, P&L, and strategy state.

Usage:
    python scripts/dashboard.py              # one-shot
    python scripts/dashboard.py --watch      # refresh every 10s
    python scripts/dashboard.py --trades     # show today's trades
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

ET = ZoneInfo("America/New_York")


def get_client() -> TradingClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not secret:
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET")
        sys.exit(1)
    return TradingClient(api_key, secret, paper=True)


def show_account(client: TradingClient) -> None:
    acct = client.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    buying_power = float(acct.buying_power)
    pnl = float(acct.equity) - float(acct.last_equity)
    pnl_pct = (pnl / float(acct.last_equity)) * 100 if float(acct.last_equity) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  ACCOUNT  {datetime.now(ET).strftime('%H:%M:%S ET')}")
    print(f"{'=' * 60}")
    print(f"  Equity:        ${equity:>12,.2f}")
    print(f"  Cash:          ${cash:>12,.2f}")
    print(f"  Buying Power:  ${buying_power:>12,.2f}")
    print(f"  Today P&L:     ${pnl:>+12,.2f}  ({pnl_pct:+.2f}%)")
    print()


def show_positions(client: TradingClient) -> None:
    positions = client.get_all_positions()
    if not positions:
        print("  No open positions.\n")
        return

    print(f"  {'Symbol':<8} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
    print(f"  {'-'*58}")
    total_pnl = 0.0
    for pos in positions:
        sym = pos.symbol
        qty = float(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        total_pnl += pnl
        side = "LONG" if qty > 0 else "SHORT"
        print(f"  {sym:<8} {qty:>+8.0f} ${entry:>9.2f} ${current:>9.2f} ${pnl:>+11.2f} {pnl_pct:>+7.2f}%")
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<8} {'':>8} {'':>10} {'':>10} ${total_pnl:>+11.2f}")
    print()


def show_strategy_state() -> None:
    path = Path("strategy_state.json")
    if not path.exists():
        path = Path("/app/data/strategy_state.json")
    if not path.exists():
        print("  No strategy checkpoint found.\n")
        return

    with open(path) as f:
        state = json.load(f)

    print(f"  STRATEGY STATE (checkpoint)")
    print(f"  {'─'*40}")
    print(f"  Active symbols: {state.get('active_symbols', '?')}")
    print(f"  Open positions: {state.get('open_position_count', '?')}")

    for sym, data in state.get("symbols", {}).items():
        bar_count = data.get("bar_count", 0)
        if bar_count == 0:
            continue
        vwap = data.get("vwap", 0)
        entry = data.get("entry_price")
        side = data.get("entry_side", "")
        gap = data.get("gap_pct", 0)
        cooldown = data.get("bars_since_exit", 999)

        status = f"{side.upper()} @ ${entry:.2f}" if entry else "FLAT"
        cd_str = f"cd={cooldown}" if cooldown < 999 else "ready"
        print(f"  {sym:<6} bars={bar_count:>4}  vwap=${vwap:>8.2f}  gap={gap:>+5.2f}%  {status:<20} {cd_str}")
    print()


def show_trades() -> None:
    path = Path("trading_results.json")
    if not path.exists():
        print("  No trades recorded yet.\n")
        return

    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))

    today = datetime.now(ET).date().isoformat()
    today_trades = [t for t in trades if t.get("timestamp", "").startswith(today)]

    if not today_trades:
        print(f"  No trades today ({today}). Total all-time: {len(trades)}\n")
        return

    print(f"  TODAY'S TRADES ({len(today_trades)})")
    print(f"  {'Time':<12} {'Symbol':<8} {'Side':<6} {'Qty':>8} {'Price':>10}")
    print(f"  {'-'*48}")
    for t in today_trades[-20:]:  # last 20
        ts = t.get("timestamp", "")
        time_str = ts[11:19] if len(ts) > 19 else ts
        print(f"  {time_str:<12} {t['symbol']:<8} {t['side']:<6} {t['quantity']:>8.0f} ${t['price']:>9.2f}")
    if len(today_trades) > 20:
        print(f"  ... and {len(today_trades) - 20} more")
    print()


def show_orders(client: TradingClient) -> None:
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=20))
    if not orders:
        print("  No open orders.\n")
        return

    print(f"  OPEN ORDERS ({len(orders)})")
    print(f"  {'Symbol':<8} {'Side':<6} {'Type':<8} {'Qty':>8} {'Limit':>10} {'Status':<10}")
    print(f"  {'-'*54}")
    for o in orders:
        limit = f"${float(o.limit_price):.2f}" if o.limit_price else "MKT"
        print(f"  {o.symbol:<8} {o.side:<6} {o.type:<8} {o.qty:>8} {limit:>10} {o.status:<10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Trading dashboard")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10s")
    parser.add_argument("--trades", action="store_true", help="Show today's trades")
    args = parser.parse_args()

    client = get_client()

    while True:
        if not args.watch:
            show_account(client)
            show_positions(client)
            show_orders(client)
            show_strategy_state()
            if args.trades:
                show_trades()
            break
        else:
            os.system("clear" if os.name != "nt" else "cls")
            show_account(client)
            show_positions(client)
            show_orders(client)
            show_strategy_state()
            show_trades()
            time.sleep(10)


if __name__ == "__main__":
    main()
