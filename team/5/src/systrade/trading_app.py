"""Systematic Trading Application"""

import os
import json
import logging
from datetime import datetime
from typing import override
from zoneinfo import ZoneInfo

from systrade.feed import AlpacaLiveStockFeed
from systrade.broker import AlpacaBroker
from systrade.strategy import Strategy
from systrade.data import BarData, ExecutionReport
from systrade.portfolio import PortfolioView, Position
from systrade.engine import Engine

import math
# ---------------------------
# --------- LOGGING ---------
# ----- logging imports -----
import logging.config
import logging.handlers
import json
import pathlib
# instantiate logger
logger = logging.getLogger(__name__)

# --- LOGGER CONFIG ---
# Verbose dictionary-type config
#+for custom logger.
# Config file found in:
# /.config/logger/config.json
# (source: youtube.com/mCoding)
def setup_logging():
    config_file = pathlib.Path(".config/logger/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)
# ---------------------


# --- Adapter Class to bridge Broker API with PortfolioView interface ---
class LivePortfolioView(PortfolioView):
    def __init__(self, broker: AlpacaBroker):
        self._broker = broker
        self._last_prices: dict[str, float] = {}

    def update_prices(self, data: BarData):
        """Called by the main loop when new data arrives."""
        for sym, bar in data.items():
            self._last_prices[sym] = bar.close
        self._as_of_time = data.as_of

    @override
    def cash(self) -> float:
        acct = self._broker.get_account_details()
        return float(acct['cash'])

    @override
    def asset_value(self) -> float:
        acct = self._broker.get_account_details()
        return float(acct['equity']) - self.cash()

    @override
    def asset_value_of(self, symbol: str) -> float:
        pos = self._broker.trading_client.get_position(symbol)
        return float(pos.market_value) if pos else 0.0

    @override
    def value(self) -> float:
        acct = self._broker.get_account_details()
        return float(acct['equity'])

    @override
    def as_of(self) -> datetime:
        return getattr(self, '_as_of_time', datetime.now(ZoneInfo("UTC")))

    @override
    def is_invested(self) -> bool:
        positions = self._broker.trading_client.get_all_positions()
        return bool(positions)

    @override
    def is_invested_in(self, symbol: str) -> bool:
        try:
            self._broker.trading_client.get_position(symbol)
            return True
        except Exception:
            return False

    @override
    def position(self, symbol) -> Position:
        pos = self._broker.trading_client.get_position(symbol)
        if not pos:
            raise ValueError(f"Not invested in {symbol}")
        return Position(symbol, float(pos.qty))

    @override
    def activity(self) -> None:
        raise NotImplementedError("Activity tracking is done by the broker in live mode.")


# -------  Momentum strategy -----------
class MyMomentumStrategy(Strategy):
    """
    A live trading adaptation of the Lab 5 Momentum Strategy.
    """
    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.symbol = symbol
        self.history = []
        self.trading_records = []
        logger.info(f"Momentum Strategy initialized for {self.symbol}")

    @override
    def on_start(self) -> None:
        """Called before first event is every received"""
        self.subscribe(self.symbol)

    @override
    def on_data(self, data: BarData) -> None:
        """Processes incoming 1-minute bars live."""
        self.current_time = data.as_of

        if self.symbol in data.symbols():
            bar = data[self.symbol]
            price = bar.close

            logger.info(f"Processing bar for {self.symbol} at {data.as_of}: Close={price}")

            if len(self.history) >= 2:
                buy_signal = price > self.history[-1] > self.history[-2]
                sell_signal = price < self.history[-1] < self.history[-2]

                if buy_signal and not self.portfolio.is_invested_in(self.symbol):
                    # TODO =============================
                    # set quantity to most you can afford
                    qty = math.floor(self.portfolio.cash() / (price * 0.95))
                    logger.info(f"Buy signal! Posting market order for {qty} shares of {self.symbol}")
                    self.post_market_order(self.symbol, quantity=qty)
                    self.order_pending = True
                    self._record_trade("BUY", qty, price)

                elif sell_signal and self.portfolio.is_invested_in(self.symbol):
                    pos = self.portfolio.position(self.symbol)
                    if pos.qty > 0:
                        logger.info(f"Sell signal! Closing position of {pos.qty} shares of {self.symbol}")
                        self.post_market_order(self.symbol, quantity=-pos.qty)
                        self.order_pending = True
                        self._record_trade("SELL", pos.qty, price)

            self.history.append(price)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        """Called on an order update"""
        log_report = report.__dict__.copy()
        log_report['fill_timestamp_iso'] = report.fill_timestamp.isoformat()
        logger.info(f"Notified of execution: {log_report}")
        self.trading_records.append(log_report)

    def _record_trade(self, side, qty, price):
        """Helper to save a simple record locally."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': side,
            'quantity': qty,
            'price': price
        }
        with open("trading_results.json", "a") as f:
            f.write(json.dumps(record) + "\n")

def main():
    setup_logging()
    logger.info("Starting Systrade Live Trading Application...")
    if not os.getenv("ALPACA_API_KEY"):
        logger.error("API keys not set. Exiting.")
        return

    feed = AlpacaLiveStockFeed()
    broker = AlpacaBroker()
    strategy = MyMomentumStrategy(symbol="SPY")

    starting_cash = 100000.0
    engine = Engine(feed=feed, broker=broker, strategy=strategy, cash=starting_cash)

    logger.info("Engine initialized. Starting run...")

    try:
        engine.run()
        logger.info("Engine run completed successfully.")

    except KeyboardInterrupt:
        logger.info("Trading interrupted by user. Stopping engine.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    logger.info("Application stopped.")


if __name__ == "__main__":
    main()

