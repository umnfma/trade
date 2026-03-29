from abc import ABC, abstractmethod
from collections import defaultdict
import os
from typing import override, Optional, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide as AlpacaOrderSideEnum, TimeInForce, QueryOrderStatus
from alpaca.trading.models import Order as AlpacaOrderModel

from systrade.data import BarData, ExecutionReport, Order, OrderType


class Broker(ABC):
    @abstractmethod
    def on_data(self, data: BarData) -> None:
        """Handle data updates"""

    @abstractmethod
    def post_order(self, order: Order) -> None:
        """Post an order to the broker"""

    @abstractmethod
    def pop_latest(self) -> list[ExecutionReport]:
        """Pop latest execution reports, will return an empty list if none"""

class BacktestBroker(Broker):
    """A test broker to simulate order communication.

    Parameters
    ----------
    slippage_bps : float
        Simulated slippage in basis points per trade.  Buys fill higher,
        sells fill lower.  Models spread, market impact, and PFOF.
        0 = no slippage (default for backward compat).
        2-5 bps is realistic for liquid large-caps.
        5-10 bps for mid-caps / volatile names.
    """

    def __init__(self, slippage_bps: float = 0.0) -> None:
        self._orders = defaultdict[str, list[Order]](lambda: [])
        self._exec_reports = list[ExecutionReport]()
        self._last_data = BarData()
        self._slippage_pct = slippage_bps / 10_000

    @override
    def on_data(self, data: BarData) -> None:
        self._last_data = data
        for symbol, bar in data.bars():
            open_orders = self._orders.get(symbol)
            if not open_orders:
                continue
            remaining: list[Order] = []
            for order in open_orders:
                fill_price = self._try_fill(order, bar)
                if fill_price is not None:
                    # Apply slippage: buys fill higher, sells fill lower
                    if order.quantity > 0:
                        fill_price *= (1 + self._slippage_pct)
                    else:
                        fill_price *= (1 - self._slippage_pct)
                    fill = ExecutionReport(
                        order=order,
                        last_price=fill_price,
                        last_quantity=order.quantity,
                        cum_quantity=order.quantity,
                        rem_quantity=0.0,
                        fill_timestamp=data.as_of,
                    )
                    self._exec_reports.append(fill)
                else:
                    remaining.append(order)
            self._orders[symbol] = remaining

    @staticmethod
    def _try_fill(order: Order, bar) -> float | None:
        """Return a fill price if the order would execute on this bar, else None."""
        if order.type == OrderType.MARKET:
            return bar.open
        if order.type == OrderType.LIMIT and order.limit_price is not None:
            # Buy limit fills if price dips to limit; sell limit fills if price rises
            if order.quantity > 0 and bar.low <= order.limit_price:
                return order.limit_price
            if order.quantity < 0 and bar.high >= order.limit_price:
                return order.limit_price
        if order.type == OrderType.STOP and order.stop_price is not None:
            if order.quantity < 0 and bar.low <= order.stop_price:
                return order.stop_price
            if order.quantity > 0 and bar.high >= order.stop_price:
                return order.stop_price
        return None

    @override
    def post_order(self, order: Order) -> None:
        self._orders[order.symbol].append(order)

    @override
    def pop_latest(self) -> list[ExecutionReport]:
        reports = self._exec_reports.copy()
        self._exec_reports.clear()
        return reports


class AlpacaBroker(Broker):
    """A broker to communicate with Alpaca API for live/paper trading."""

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_API_SECRET")
        paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true" 

        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set.")

        self.trading_client = TradingClient(api_key, secret_key, paper=paper_trading)
        
        self._pending_orders: dict[str, Order] = {}
        self._exec_reports: list[ExecutionReport] = []

    def get_account_details(self):
        """Helper method for LivePortfolioView to fetch account details."""
        return self.trading_client.get_account()

    @override
    def on_data(self, data: BarData) -> None:
        """Poll Alpaca for order updates when new data arrives."""
        if not self._pending_orders:
            return

        request_params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=100,
            nested=True
        )
        closed_orders: List[AlpacaOrderModel] = self.trading_client.get_orders(request_params)

        for alpaca_order in closed_orders:
            client_order_id = alpaca_order.client_order_id
            if client_order_id in self._pending_orders:
                original_systrade_order = self._pending_orders.pop(client_order_id)
                
                if alpaca_order.filled_avg_price is not None and alpaca_order.filled_qty is not None:
                    fill = ExecutionReport(
                        order=original_systrade_order,
                        last_price=float(alpaca_order.filled_avg_price),
                        last_quantity=float(alpaca_order.filled_qty),
                        cum_quantity=float(alpaca_order.filled_qty),
                        rem_quantity=0.0,
                        fill_timestamp=alpaca_order.updated_at or alpaca_order.created_at,
                    )
                    self._exec_reports.append(fill)

    @override
    def post_order(self, order: Order) -> None:
        """Post an order to the Alpaca API."""

        if order.quantity > 0:
            alpaca_side = AlpacaOrderSideEnum.BUY
            qty_magnitude = order.quantity
        elif order.quantity < 0:
            alpaca_side = AlpacaOrderSideEnum.SELL
            qty_magnitude = abs(order.quantity)
        else:
            print(f"Order quantity is zero, skipping: {order}")
            return

        tif = TimeInForce.DAY if order.type != OrderType.MARKET else TimeInForce.GTC

        try:
            request = self._build_order_request(order, alpaca_side, qty_magnitude, tif)
            self.trading_client.submit_order(request)
            self._pending_orders[order.id] = order
        except Exception as e:
            print(f"Error submitting order for {order.symbol}: {e}")

    @staticmethod
    def _build_order_request(order: Order, side, qty, tif):
        """Build the appropriate Alpaca order request based on OrderType."""
        common = dict(symbol=order.symbol, qty=qty, side=side,
                      time_in_force=tif, client_order_id=order.id)

        if order.type == OrderType.LIMIT:
            return LimitOrderRequest(**common, limit_price=order.limit_price)
        if order.type == OrderType.STOP:
            return StopOrderRequest(**common, stop_price=order.stop_price)
        if order.type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                **common,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        return MarketOrderRequest(**common)

    @override
    def pop_latest(self) -> list[ExecutionReport]:
        reports = self._exec_reports.copy()
        self._exec_reports.clear()
        return reports
