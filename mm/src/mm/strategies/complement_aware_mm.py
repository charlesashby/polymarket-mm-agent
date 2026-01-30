"""
Complement-aware market-maker strategy for Polymarket binary options.

This strategy implements complement-aware pricing (Milestone #3):
1. Track YES/NO instrument pairs by slug
2. Track complement BBO for each instrument
3. Calculate implied BBO from complement prices
4. Use effective BBO: max(direct, implied) for bid, min(direct, implied) for ask
5. Quote around mid of effective BBO with spread adjustment

Key improvement over SimpleMM:
- SimpleMM uses direct BBO only
- ComplementAwareMM uses effective BBO (direct + implied from complement)
- This accounts for complement-implied liquidity and should improve fill quality
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


class ComplementAwareMM(BaseStrategy):
    """
    Complement-aware market-maker that quotes using effective BBO.

    Parameters
    ----------
    spread : float
        Half-spread in decimal (e.g., 0.02 for 2 cents on each side)
    quote_size : int
        Fixed quote size in contracts
    verbose : bool
        Enable verbose logging
    """

    def __init__(
        self,
        spread: float = 0.02,
        quote_size: int = 100,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the complement-aware market-maker strategy.

        Parameters
        ----------
        spread : float, default 0.02
            Half-spread to quote around effective mid price
        quote_size : int, default 100
            Fixed size for each quote
        verbose : bool, default False
            If True, log trading activity
        **kwargs
            Additional keyword arguments passed to BaseStrategy
        """
        # Factory functions for BaseStrategy
        def active_pair_factory():
            return {
                "last_eff_mid": None,
                "last_eff_bid": None,
                "last_eff_ask": None,
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
            log_label="ComplementAwareMM",
            metrics_enabled=False,
            **kwargs
        )

        self.spread = float(spread)
        self.quote_size = int(quote_size)
        self.verbose = bool(verbose)

        # Track open orders per instrument
        self._open_orders: Dict[InstrumentId, Dict[OrderSide, LimitOrder]] = defaultdict(dict)

        # Track direct BBO per instrument
        self._direct_bbo: Dict[InstrumentId, Dict[str, float]] = defaultdict(dict)

        # Statistics
        self._fills = 0
        self._quotes_sent = 0
        self._data_processed = 0
        self._effective_used = 0

    def on_start(self) -> None:
        """Called when strategy starts."""
        super()._on_start()

        print(f"[ComplementAwareMM] Strategy started with spread={self.spread}, size={self.quote_size}")
        print(f"[ComplementAwareMM] Verbose mode: {self.verbose}")
        print(f"[ComplementAwareMM] Tracking {len(self._slugs)} slug pairs")

    def _handle_quote_tick(self, tick: QuoteTick) -> None:
        """
        Handle quote tick updates.

        Updates direct BBO and triggers quoting logic.
        """
        inst_id = tick.instrument_id

        # Update direct BBO
        self._direct_bbo[inst_id]["bid"] = float(tick.bid_price)
        self._direct_bbo[inst_id]["ask"] = float(tick.ask_price)
        self._direct_bbo[inst_id]["bid_size"] = float(tick.bid_size)
        self._direct_bbo[inst_id]["ask_size"] = float(tick.ask_size)

        # Update quotes for this instrument
        self._update_quotes(inst_id)

    def _handle_trade_tick(self, tick: TradeTick) -> None:
        """Handle trade tick updates (informational only)."""
        pass

    def _handle_order_book_depth10(self, book: OrderBookDepth10) -> None:
        """
        Handle order book depth updates.

        Use depth snapshots to update direct BBO and trigger quoting.
        """
        self._data_processed += 1

        if self._data_processed <= 3:
            print(f"[ComplementAwareMM] Depth update #{self._data_processed} for {book.instrument_id}")

        inst_id = book.instrument_id

        # Extract best bid/ask from depth
        if book.bids and len(book.bids) > 0:
            best_bid_price = float(book.bids[0].price)
            best_bid_size = float(book.bids[0].size)
            self._direct_bbo[inst_id]["bid"] = best_bid_price
            self._direct_bbo[inst_id]["bid_size"] = best_bid_size

        if book.asks and len(book.asks) > 0:
            best_ask_price = float(book.asks[0].price)
            best_ask_size = float(book.asks[0].size)
            self._direct_bbo[inst_id]["ask"] = best_ask_price
            self._direct_bbo[inst_id]["ask_size"] = best_ask_size

        # Update quotes for this instrument
        self._update_quotes(inst_id)

    def _handle_order_flow_bucket(self, data: OrderFlowBucketDepth10CustomData) -> None:
        """Handle order flow bucket updates (not used in this strategy)."""
        pass

    def _handle_chainlink_data(self, data: ChainlinkCustomData) -> None:
        """Handle Chainlink oracle updates (not used in this strategy)."""
        pass

    def _get_complement_id(self, inst_id: InstrumentId) -> Optional[InstrumentId]:
        """
        Get the complement instrument ID for a given instrument.

        For YES outcome, returns NO. For NO outcome, returns YES.
        Returns None if complement cannot be found.
        """
        # Find slug for this instrument
        meta = self._meta_by_id.get(inst_id)
        if not meta:
            return None

        slug = meta.get("slug")
        if not slug:
            return None

        # Check if this is YES or NO, return the opposite
        yes_id = self._yes_by_slug.get(slug)
        no_id = self._no_by_slug.get(slug)

        if inst_id == yes_id:
            return no_id
        elif inst_id == no_id:
            return yes_id
        else:
            return None

    def _compute_effective_bbo(self, inst_id: InstrumentId) -> Optional[Dict[str, float]]:
        """
        Compute effective BBO using direct and implied (from complement) prices.

        Returns dict with keys: 'bid', 'ask', 'mid'
        Returns None if insufficient data.

        Algorithm:
        1. Get direct BBO for this instrument
        2. Get direct BBO for complement instrument
        3. Calculate implied BBO:
           - implied_bid = 1 - complement_ask
           - implied_ask = 1 - complement_bid
        4. Compute effective BBO:
           - eff_bid = max(direct_bid, implied_bid)
           - eff_ask = min(direct_ask, implied_ask)
        5. Calculate mid from effective BBO
        """
        # Get direct BBO
        direct_bbo = self._direct_bbo.get(inst_id)
        if not direct_bbo or "bid" not in direct_bbo or "ask" not in direct_bbo:
            return None

        direct_bid = direct_bbo["bid"]
        direct_ask = direct_bbo["ask"]

        if direct_bid <= 0 or direct_ask <= 0:
            return None

        # Get complement instrument
        comp_id = self._get_complement_id(inst_id)
        if not comp_id:
            # No complement, use direct BBO only
            return {
                "bid": direct_bid,
                "ask": direct_ask,
                "mid": (direct_bid + direct_ask) / 2.0,
            }

        # Get complement BBO
        comp_bbo = self._direct_bbo.get(comp_id)
        if not comp_bbo or "bid" not in comp_bbo or "ask" not in comp_bbo:
            # No complement BBO, use direct only
            return {
                "bid": direct_bid,
                "ask": direct_ask,
                "mid": (direct_bid + direct_ask) / 2.0,
            }

        comp_bid = comp_bbo["bid"]
        comp_ask = comp_bbo["ask"]

        if comp_bid <= 0 or comp_ask <= 0:
            # Invalid complement BBO, use direct only
            return {
                "bid": direct_bid,
                "ask": direct_ask,
                "mid": (direct_bid + direct_ask) / 2.0,
            }

        # Calculate implied BBO from complement
        # If complement asks at comp_ask, we can buy complement at comp_ask
        # Then we effectively sell this instrument at (1 - comp_ask)
        # So implied bid for this instrument is (1 - comp_ask)
        implied_bid = 1.0 - comp_ask

        # If complement bids at comp_bid, we can sell complement at comp_bid
        # Then we effectively buy this instrument at (1 - comp_bid)
        # So implied ask for this instrument is (1 - comp_bid)
        implied_ask = 1.0 - comp_bid

        # Compute effective BBO
        eff_bid = max(direct_bid, implied_bid)
        eff_ask = min(direct_ask, implied_ask)

        # Sanity check
        if eff_bid >= eff_ask:
            # Cross detected - use direct BBO to avoid issues
            return {
                "bid": direct_bid,
                "ask": direct_ask,
                "mid": (direct_bid + direct_ask) / 2.0,
            }

        # Enforce price bounds
        eff_bid = max(self.px_floor, min(eff_bid, self.px_ceil))
        eff_ask = max(self.px_floor, min(eff_ask, self.px_ceil))

        # Track if we used implied prices (for statistics)
        if eff_bid > direct_bid or eff_ask < direct_ask:
            self._effective_used += 1

        return {
            "bid": eff_bid,
            "ask": eff_ask,
            "mid": (eff_bid + eff_ask) / 2.0,
        }

    def _update_quotes(self, inst_id: InstrumentId) -> None:
        """
        Update quotes for an instrument based on effective BBO.

        Logic:
        1. Compute effective BBO (direct + implied from complement)
        2. Calculate mid price from effective BBO
        3. Calculate quote prices at mid ± spread
        4. Enforce price bounds [0.01, 0.99]
        5. Cancel stale orders
        6. Submit new orders
        """
        # Compute effective BBO
        eff_bbo = self._compute_effective_bbo(inst_id)
        if not eff_bbo:
            return

        eff_mid = eff_bbo["mid"]

        # Calculate our quote prices around effective mid
        our_bid = eff_mid - self.spread
        our_ask = eff_mid + self.spread

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

        # Submit new buy order
        buy_order = self._make_buy_order(inst_id, our_bid, self.quote_size)
        if buy_order:
            self.log_order_submit(buy_order, note=f"eff_mid={eff_mid:.3f}")
            self.submit_order(buy_order)
            self._open_orders[inst_id][OrderSide.BUY] = buy_order
            self._quotes_sent += 1

        # Submit new sell order
        sell_order = self._make_sell_order(inst_id, our_ask, self.quote_size)
        if sell_order:
            self.log_order_submit(sell_order, note=f"eff_mid={eff_mid:.3f}")
            self.submit_order(sell_order)
            self._open_orders[inst_id][OrderSide.SELL] = sell_order
            self._quotes_sent += 1

        if self.verbose and self._quotes_sent % 100 == 0:
            print(f"[ComplementAwareMM] Sent {self._quotes_sent} quotes, {self._fills} fills, effective_used={self._effective_used}")

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
            print(f"[ComplementAwareMM] FILL: {inst_id} {side_str} {event.last_qty}@{event.last_px} (fill #{self._fills})")

    def _on_stop(self) -> None:
        """Called when strategy stops."""
        # Cancel all open orders
        for inst_id, orders in self._open_orders.items():
            for order in orders.values():
                if order and order.is_open:
                    self.cancel_order(order)

        print(f"[ComplementAwareMM] Strategy stopped")
        print(f"[ComplementAwareMM] Statistics:")
        print(f"[ComplementAwareMM]   Data processed: {self._data_processed}")
        print(f"[ComplementAwareMM]   Quotes sent: {self._quotes_sent}")
        print(f"[ComplementAwareMM]   Fills: {self._fills}")
        print(f"[ComplementAwareMM]   Effective BBO used: {self._effective_used} times")
        if self._quotes_sent > 0:
            print(f"[ComplementAwareMM]   Effective usage %: {100.0 * self._effective_used / self._quotes_sent:.2f}%")
