from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, override

from systrade.data import BarData, ExecutionReport, Order, OrderType
from systrade.portfolio import PortfolioView

import uuid

SubscribeHook = Callable[[str], None]
PostOrderHook = Callable[[Order], None]


class Strategy(ABC):
    def __init__(self) -> None:
        self._subscribe_hook: Callable[[str], None] = lambda _: None
        self._post_order_hook: Callable[[Order], None] = lambda _: None
        self._portfolio: PortfolioView | None = None
        self._current_order_id = 1
        self._current_time: datetime

    def setup_context(
        self,
        subscribe_hook: SubscribeHook,
        post_order_hook: PostOrderHook,
        portfolio: PortfolioView,
    ) -> None:
        """Sets up interface for working with external dependencies. This should
        be called by clients orchestrating a strategy.

        Parameters
        ----------
        subscribe_hook
            Callback to trigger a subscription
        post_order_hook
            Callback to submit an order
        portfolio
            A portfolio view to access positions, etc.
        """
        self._subscribe_hook = subscribe_hook
        self._post_order_hook = post_order_hook
        self._portfolio = portfolio

    @property
    def portfolio(self) -> PortfolioView:
        """Get a view of the portfolio"""
        if self._portfolio is None:
            raise ValueError("No portfolio!")
        return self._portfolio

    @property
    def current_time(self) -> datetime:
        """Get the current time"""
        return self._current_time

    @current_time.setter
    def current_time(self, value) -> None:
        """Set the current time"""
        self._current_time = value

    def subscribe(self, symbol: str) -> None:
        """Subscribe to symbol"""
        self._subscribe_hook(symbol)

    def post_market_order(self, symbol: str, quantity: float) -> None:
        """Post a market order to broker"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            quantity=quantity,
            type=OrderType.MARKET,
            submit_time=self.current_time,
        )
        self._current_order_id += 1
        self._post_order_hook(order)

    def post_limit_order(self, symbol: str, quantity: float, limit_price: float) -> None:
        """Post a limit order to broker"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            quantity=quantity,
            type=OrderType.LIMIT,
            submit_time=self.current_time,
            limit_price=limit_price,
        )
        self._current_order_id += 1
        self._post_order_hook(order)

    def post_stop_order(self, symbol: str, quantity: float, stop_price: float) -> None:
        """Post a stop order to broker"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            quantity=quantity,
            type=OrderType.STOP,
            submit_time=self.current_time,
            stop_price=stop_price,
        )
        self._current_order_id += 1
        self._post_order_hook(order)

    @abstractmethod
    def on_start(self) -> None:
        """Called before first event is every received"""

    @abstractmethod
    def on_data(self, data: BarData) -> None:
        """Called on each market data event"""

    @abstractmethod
    def on_execution(self, report: ExecutionReport) -> None:
        """Called on an order update"""
