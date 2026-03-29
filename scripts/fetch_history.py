#!/usr/bin/env python3
"""
Fetch 1-minute historical bars from Alpaca for backtesting.

Usage:
    python scripts/fetch_history.py --start 2026-03-01 --end 2026-03-14 --output data/history_1min.csv
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
ET = ZoneInfo("America/New_York")


def fetch_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Normalize column names to match systrade convention
    df = df.rename(columns={
        "symbol": "Symbol",
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df = df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]

    # Filter to regular trading hours only (9:30 AM - 4:00 PM ET)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(ET)
    df = df[
        (df["Date"].dt.time >= pd.Timestamp("09:30").time())
        & (df["Date"].dt.time < pd.Timestamp("16:00").time())
    ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Alpaca 1-min bars")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data/history_1min.csv", help="Output CSV path")
    args = parser.parse_args()

    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_API_SECRET"]

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=ET)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=ET)

    print(f"Fetching 1-min bars for {SYMBOLS} from {args.start} to {args.end} ...")
    df = fetch_bars(SYMBOLS, start_dt, end_dt, api_key, secret_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} bars to {output_path}")


if __name__ == "__main__":
    main()
