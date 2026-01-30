"""
Simple market-maker strategy for Polymarket binary options.

This strategy implements basic market-making logic:
1. Calculate mid price from best bid/offer
2. Place symmetric quotes at mid ± spread
3. Enforce price bounds [0.01, 0.99]
4. Cancel and replace quotes on price updates
5. Track fills and position

This is milestone #2 - basic quoting logic without inventory management.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from collections import defaultdict

from nautilus_trader.model.data import QuoteTick, TradeTick, OrderBookDepth10
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import LimitOrder

from mm.src.mm.strategies.base_strategy import BaseStrategy
from mm.src.mm.types import ChainlinkCustomData, OrderFlowBucketDepth10CustomData

import sys


class SimpleMarketMaker(BaseStrategy):
    """
    Simple market-maker that quotes symmetric spreads around mid price.

    Parameters
    ----------
    spread : float
        Half-spread in decimal (e.g., 0.02 for 2 cents on each side)
    quote_notional : float
        Target notional per quote
    verbose : bool
        Enable verbose logging
    """

    def __init__(
        self,
        spread: float = 0.02,
        quote_notional: Optional[float] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the simple market-maker strategy.

        Parameters
        ----------
        spread : float, default 0.02
            Half-spread to quote around mid price
        quote_notional : float, default None
            Target notional per quote (defaults to MAX_ORDER_NOTIONAL)
        verbose : bool, default False
            If True, log trading activity
        **kwargs
            Additional keyword arguments passed to BaseStrategy
        """
        # Factory functions for BaseStrategy
        def active_pair_factory():
            return {
                "last_mid": None,
                "last_bid": None,
                "last_ask": None,
                "quote_updates": 0,
            }

        def event_state_factory(**kw):
            return {
                "start_dt": kw.get("start_dt"),
                "end_dt": kw.get("end_dt"),
                "ref": kw.get("ref"),
            }

        super().__init__(
            active_pair_factory=active_pair_factory,
            event_state_factory=event_state_factory,
            log_label="SimpleMarketMaker",
            metrics_enabled=False,
            **kwargs
        )

        self.spread = float(spread)
        if quote_notional is None:
            quote_notional = self._max_order_notional
        self.quote_notional = float(quote_notional)
        self.verbose = bool(verbose)

        # Track open orders per instrument
        self._open_orders: Dict[InstrumentId, Dict[OrderSide, LimitOrder]] = defaultdict(dict)

        # Track BBO per instrument
        self._bbo: Dict[InstrumentId, Dict[str, float]] = defaultdict(dict)

        # Statistics
        self._fills = 0
        self._quotes_sent = 0
        self._data_processed = 0

    def on_start(self) -> None:
        """Called when strategy starts."""
        super()._on_start()

        print(f"[SimpleMM] Strategy started with spread={self.spread}, notional={self.quote_notional}")
        print(f"[SimpleMM] Verbose mode: {self.verbose}")

    def _handle_quote_tick(self, tick: QuoteTick) -> None:
        """
        Handle quote tick updates.

        Updates BBO and triggers quoting logic.
        """
        inst_id = tick.instrument_id

        # Update BBO
        self._bbo[inst_id]["bid"] = float(tick.bid_price)
        self._bbo[inst_id]["ask"] = float(tick.ask_price)
        self._bbo[inst_id]["bid_size"] = float(tick.bid_size)
        self._bbo[inst_id]["ask_size"] = float(tick.ask_size)

        # Update quotes
        self._update_quotes(inst_id)

    def _handle_trade_tick(self, tick: TradeTick) -> None:
        """Handle trade tick updates (informational only)."""
        pass

    def _handle_order_book_depth10(self, book: OrderBookDepth10) -> None:
        """
        Handle order book depth updates.

        Use depth snapshots to update BBO and trigger quoting.
        """
        self._data_processed += 1

        if self._data_processed <= 3:
            print(f"[SimpleMM] Depth update #{self._data_processed} for {book.instrument_id}")

        inst_id = book.instrument_id

        # Extract best bid/ask from depth
        if book.bids and len(book.bids) > 0:
            best_bid_price = float(book.bids[0].price)
            best_bid_size = float(book.bids[0].size)
            self._bbo[inst_id]["bid"] = best_bid_price
            self._bbo[inst_id]["bid_size"] = best_bid_size

        if book.asks and len(book.asks) > 0:
            best_ask_price = float(book.asks[0].price)
            best_ask_size = float(book.asks[0].size)
            self._bbo[inst_id]["ask"] = best_ask_price
            self._bbo[inst_id]["ask_size"] = best_ask_size

        # Update quotes
        self._update_quotes(inst_id)

    def _handle_order_flow_bucket(self, data: OrderFlowBucketDepth10CustomData) -> None:
        """Handle order flow bucket updates (not used in simple strategy)."""
        pass

    def _handle_chainlink_data(self, data: ChainlinkCustomData) -> None:
        """Handle Chainlink oracle updates (not used in simple strategy)."""
        pass

    def _update_quotes(self, inst_id: InstrumentId) -> None:
        """
        Update quotes for an instrument based on current BBO.

        Logic:
        1. Calculate mid price from BBO
        2. Calculate quote prices at mid ± spread
        3. Enforce price bounds [0.01, 0.99]
        4. Cancel stale orders
        5. Submit new orders
        """
        bbo = self._bbo.get(inst_id)
        if not bbo or "bid" not in bbo or "ask" not in bbo:
            return

        bid = bbo["bid"]
        ask = bbo["ask"]

        # Calculate mid price
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2.0

        # Calculate our quote prices
        our_bid = mid - self.spread
        our_ask = mid + self.spread

        # Enforce price bounds [0.01, 0.99]
        our_bid = max(self.px_floor, min(our_bid, self.px_ceil))
        our_ask = max(self.px_floor, min(our_ask, self.px_ceil))

        # Don't cross the spread
        if our_bid >= our_ask:
            return

        # Check if we need to update orders
        open_orders = self._open_orders.get(inst_id, {})

        # Cancel existing orders
        for side, order in list(open_orders.items()):
            if order and order.is_open:
                self.cancel_order(order)

        # Clear tracking
        self._open_orders[inst_id] = {}

        bid_qty = self.qty_for_notional(our_bid, self.quote_notional)
        ask_qty = self.qty_for_notional(our_ask, self.quote_notional)

        # Submit new buy order
        buy_order = self._make_buy_order(inst_id, our_bid, bid_qty)
        if buy_order:
            self.log_order_submit(buy_order, note=f"mid={mid:.3f}")
            self.submit_order(buy_order)
            self._open_orders[inst_id][OrderSide.BUY] = buy_order
            self._quotes_sent += 1

        # Submit new sell order
        sell_order = self._make_sell_order(inst_id, our_ask, ask_qty)
        if sell_order:
            self.log_order_submit(sell_order, note=f"mid={mid:.3f}")
            self.submit_order(sell_order)
            self._open_orders[inst_id][OrderSide.SELL] = sell_order
            self._quotes_sent += 1

        if self.verbose and self._quotes_sent % 100 == 0:
            print(f"[SimpleMM] Sent {self._quotes_sent} quotes, {self._fills} fills")

    def on_order_filled(self, event) -> None:
        """
        Handle order fill events.

        Track fills and update position.
        """
        super().on_order_filled(event)

        self._fills += 1

        inst_id = event.instrument_id
        client_order_id = event.client_order_id

        # Remove filled order from tracking
        if inst_id in self._open_orders:
            for side, tracked_order in list(self._open_orders[inst_id].items()):
                if tracked_order and tracked_order.client_order_id == client_order_id:
                    del self._open_orders[inst_id][side]
                    break

        if self.verbose:
            side_str = "BUY" if event.order_side == OrderSide.BUY else "SELL"
            print(f"[SimpleMM] FILL: {inst_id} {side_str} {event.last_qty}@{event.last_px} (fill #{self._fills})")

    def _on_stop(self) -> None:
        """Called when strategy stops."""
        # Cancel all open orders
        for inst_id, orders in self._open_orders.items():
            for order in orders.values():
                if order and order.is_open:
                    self.cancel_order(order)

        print(f"[SimpleMM] Strategy stopped")
        print(f"[SimpleMM] Statistics:")
        print(f"[SimpleMM]   Data processed: {self._data_processed}")
        print(f"[SimpleMM]   Quotes sent: {self._quotes_sent}")
        print(f"[SimpleMM]   Fills: {self._fills}")
