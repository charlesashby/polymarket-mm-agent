"""
Minimal test strategy to debug order submission.
"""

from __future__ import annotations

from typing import Any, Optional

from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.enums import OrderSide

from mm.src.mm.strategies.base_strategy import BaseStrategy


class TestOrdersStrategy(BaseStrategy):
    """
    Minimal strategy that aggressively crosses the market on first depth update.
    Purpose: Test if order submission and fills work at all.
    """

    def __init__(self, quote_notional: Optional[float] = None, **kwargs: Any) -> None:
        def active_pair_factory():
            return {"quoted": False}

        def event_state_factory(**kw):
            return {}

        super().__init__(
            active_pair_factory=active_pair_factory,
            event_state_factory=event_state_factory,
            log_label="TestOrders",
            **kwargs
        )

        if quote_notional is None:
            quote_notional = self._max_order_notional
        self.quote_notional = float(quote_notional)
        self._quoted = False

    def on_start(self) -> None:
        super()._on_start()
        print("[TestOrders] Strategy started")

    def _handle_order_book_depth10(self, book: OrderBookDepth10) -> None:
        """Submit ONE aggressive order on first depth update."""
        if self._quoted:
            return

        inst_id = book.instrument_id

        # Extract best ask (we'll buy above it to cross)
        if not book.asks or len(book.asks) == 0:
            return

        best_ask = float(book.asks[0].price)

        # Check if instrument is in cache
        inst = self.cache.instrument(inst_id)
        if inst is None:
            print(f"[TestOrders] Instrument {inst_id} not in cache yet")
            return

        # Aggressively cross: buy at best_ask + 0.05
        buy_price = min(0.99, best_ask + 0.05)
        buy_qty = self.qty_for_notional(buy_price, self.quote_notional)

        print(f"[TestOrders] Creating BUY order: {buy_qty}@{buy_price:.2f} (best_ask={best_ask:.2f})")

        buy_order = self._make_buy_order(inst_id, buy_price, buy_qty)
        if buy_order:
            print(f"[TestOrders] Order created successfully: {buy_order.client_order_id}")
            self.log_order_submit(buy_order, note="aggressive_cross")
            self.submit_order(buy_order)
            self._quoted = True
            print(f"[TestOrders] Order submitted!")
        else:
            print(f"[TestOrders] Order creation returned None")

    def _handle_quote_tick(self, tick) -> None:
        pass

    def _handle_trade_tick(self, tick) -> None:
        pass

    def _handle_order_flow_bucket(self, data) -> None:
        pass

    def _handle_chainlink_data(self, data) -> None:
        pass

    def on_order_filled(self, event) -> None:
        super().on_order_filled(event)
        print(f"[TestOrders] FILL! {event.last_qty}@{event.last_px}")

    def _on_stop(self) -> None:
        print(f"[TestOrders] Strategy stopped")
