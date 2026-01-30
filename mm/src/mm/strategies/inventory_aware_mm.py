"""
Inventory-aware market-maker strategy for Polymarket binary options.

This strategy implements inventory-aware position management (Milestone #4):
1. Track net position per instrument from fills
2. Track average cost basis for each position
3. Implement position limit thresholds (long/short)
4. Add inventory skew to quote pricing
5. Widen spread as position approaches limits
6. Stop quoting one side when at position limit

Key improvements over ComplementAwareMM:
- ComplementAwareMM quotes symmetrically around effective mid
- InventoryAwareMM adjusts quotes based on current inventory position
- Skews quotes to reduce adverse selection and manage risk
- Enforces position limits to prevent runaway exposure
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


class InventoryAwareMM(BaseStrategy):
    """
    Inventory-aware market-maker that manages position risk.

    Parameters
    ----------
    spread : float
        Base half-spread in decimal (e.g., 0.01 for 1 cent on each side, optimized)
    quote_notional : float
        Target notional per quote
    position_limit : int
        Maximum absolute position per instrument (e.g., 5000)
    skew_coefficient : float
        Coefficient for inventory skew adjustment (0.0 to 1.0)
        Higher values = more aggressive position unwinding
    spread_widening : float
        Extra spread added per % of position limit used (e.g., 0.01)
    verbose : bool
        Enable verbose logging
    """

    def __init__(
        self,
        spread: float = 0.01,
        quote_notional: Optional[float] = None,
        position_limit: int = 5000,
        skew_coefficient: float = 0.3,
        spread_widening: float = 0.01,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the inventory-aware market-maker strategy.

        Parameters
        ----------
        spread : float, default 0.01
            Base half-spread to quote around effective mid price (optimized from 0.02)
        quote_notional : float, default None
            Target notional per quote (defaults to MAX_ORDER_NOTIONAL)
        position_limit : int, default 5000
            Maximum absolute position per instrument
        skew_coefficient : float, default 0.3
            Inventory skew coefficient (0.0 = no skew, 1.0 = max skew)
        spread_widening : float, default 0.01
            Extra spread per % of limit used
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
            log_label="InventoryAwareMM",
            metrics_enabled=False,
            **kwargs
        )

        self.spread = float(spread)
        if quote_notional is None:
            quote_notional = self._max_order_notional
        self.quote_notional = float(quote_notional)
        self.position_limit = int(position_limit)
        self.skew_coefficient = float(skew_coefficient)
        self.spread_widening = float(spread_widening)
        self.verbose = bool(verbose)

        # Track open orders per instrument
        self._open_orders: Dict[InstrumentId, Dict[OrderSide, LimitOrder]] = defaultdict(dict)

        # Track direct BBO per instrument
        self._direct_bbo: Dict[InstrumentId, Dict[str, float]] = defaultdict(dict)

        # Track position and cost basis per instrument
        self._positions: Dict[InstrumentId, float] = defaultdict(float)
        self._cost_basis: Dict[InstrumentId, float] = defaultdict(float)
        self._position_value: Dict[InstrumentId, float] = defaultdict(float)

        # Statistics
        self._fills = 0
        self._quotes_sent = 0
        self._data_processed = 0
        self._effective_used = 0
        self._position_limit_hits = 0
        self._skew_adjustments = 0

    def on_start(self) -> None:
        """Called when strategy starts."""
        super()._on_start()

        print(f"[InventoryAwareMM] Strategy started")
        print(f"[InventoryAwareMM]   Base spread: {self.spread}")
        print(f"[InventoryAwareMM]   Quote notional: {self.quote_notional}")
        print(f"[InventoryAwareMM]   Position limit: {self.position_limit}")
        print(f"[InventoryAwareMM]   Skew coefficient: {self.skew_coefficient}")
        print(f"[InventoryAwareMM]   Spread widening: {self.spread_widening}")
        print(f"[InventoryAwareMM]   Verbose: {self.verbose}")
        print(f"[InventoryAwareMM]   Tracking {len(self._slugs)} slug pairs")

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
            print(f"[InventoryAwareMM] Depth update #{self._data_processed} for {book.instrument_id}")

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
        implied_bid = 1.0 - comp_ask
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

    def _get_position(self, inst_id: InstrumentId) -> float:
        """Get current net position for instrument."""
        return self._positions.get(inst_id, 0.0)

    def _update_position(self, inst_id: InstrumentId, side: OrderSide, qty: float, price: float) -> None:
        """
        Update position and cost basis for an instrument after a fill.

        Parameters
        ----------
        inst_id : InstrumentId
            Instrument that was filled
        side : OrderSide
            BUY or SELL
        qty : float
            Quantity filled
        price : float
            Fill price
        """
        current_position = self._positions[inst_id]
        current_cost_basis = self._cost_basis[inst_id]
        current_value = self._position_value[inst_id]

        # Update position
        if side == OrderSide.BUY:
            new_position = current_position + qty
            new_value = current_value + (qty * price)
        else:  # SELL
            new_position = current_position - qty
            new_value = current_value - (qty * price)

        self._positions[inst_id] = new_position

        # Update cost basis (volume-weighted average)
        if abs(new_position) > 0.001:
            self._position_value[inst_id] = new_value
            self._cost_basis[inst_id] = new_value / new_position if new_position != 0 else 0.0
        else:
            # Position closed, reset
            self._position_value[inst_id] = 0.0
            self._cost_basis[inst_id] = 0.0

    def _calculate_inventory_skew(self, inst_id: InstrumentId) -> float:
        """
        Calculate inventory skew adjustment for quote prices.

        Returns a value in range approximately [-spread, +spread] that should be
        added to quote prices to encourage position reduction.

        Positive skew = shift quotes higher (encourage selling)
        Negative skew = shift quotes lower (encourage buying)

        Algorithm:
        1. Calculate position as % of limit: position_pct = position / limit
        2. Apply skew coefficient: skew = skew_coeff * position_pct * base_spread
        3. This creates a price adjustment proportional to inventory level
        """
        position = self._get_position(inst_id)

        if abs(position) < 0.001:
            return 0.0

        # Calculate position as fraction of limit
        position_pct = position / self.position_limit

        # Apply skew coefficient
        # When long (position > 0), skew > 0, which shifts quotes up (encourages selling)
        # When short (position < 0), skew < 0, which shifts quotes down (encourages buying)
        skew = self.skew_coefficient * position_pct * self.spread

        return skew

    def _calculate_spread_adjustment(self, inst_id: InstrumentId) -> float:
        """
        Calculate spread widening based on position size.

        Returns additional spread to add to base spread.
        Spread widens as position approaches limit.

        Algorithm:
        1. Calculate absolute position as % of limit
        2. Multiply by spread_widening parameter
        3. Add this to base spread for both bid and ask
        """
        position = self._get_position(inst_id)
        position_pct = abs(position) / self.position_limit

        # Widen spread proportionally to position size
        additional_spread = position_pct * self.spread_widening

        return additional_spread

    def _check_position_limit(self, inst_id: InstrumentId, side: OrderSide, qty: float) -> bool:
        """
        Check if submitting an order would exceed position limits.

        Parameters
        ----------
        inst_id : InstrumentId
            Instrument to check
        side : OrderSide
            BUY or SELL
        qty : float
            Order quantity

        Returns
        -------
        bool
            True if order is allowed, False if it would exceed limits
        """
        current_position = self._get_position(inst_id)

        # Calculate what position would be after fill
        if side == OrderSide.BUY:
            projected_position = current_position + qty
        else:
            projected_position = current_position - qty

        # Check if projected position exceeds limit
        if abs(projected_position) > self.position_limit:
            self._position_limit_hits += 1
            return False

        return True

    def _update_quotes(self, inst_id: InstrumentId) -> None:
        """
        Update quotes for an instrument with inventory-aware adjustments.

        Logic:
        1. Compute effective BBO (direct + implied from complement)
        2. Calculate mid price from effective BBO
        3. Calculate inventory skew adjustment
        4. Calculate spread widening from position size
        5. Calculate quote prices with skew and widened spread
        6. Check position limits before quoting each side
        7. Enforce price bounds [0.01, 0.99]
        8. Cancel stale orders
        9. Submit new orders
        """
        # Compute effective BBO
        eff_bbo = self._compute_effective_bbo(inst_id)
        if not eff_bbo:
            return

        eff_mid = eff_bbo["mid"]

        # Get current position
        position = self._get_position(inst_id)

        # Calculate inventory skew (shifts quotes to reduce position)
        inventory_skew = self._calculate_inventory_skew(inst_id)

        # Calculate spread adjustment (widen spread as position grows)
        spread_adjustment = self._calculate_spread_adjustment(inst_id)
        adjusted_spread = self.spread + spread_adjustment

        # Calculate our quote prices around effective mid with skew
        # Skew shifts both quotes in same direction to encourage unwinding
        our_bid = eff_mid - adjusted_spread + inventory_skew
        our_ask = eff_mid + adjusted_spread + inventory_skew

        # Enforce price bounds [0.01, 0.99]
        our_bid = max(self.px_floor, min(our_bid, self.px_ceil))
        our_ask = max(self.px_floor, min(our_ask, self.px_ceil))

        # Don't cross the spread
        if our_bid >= our_ask:
            return

        # Track if we made skew adjustments
        if abs(inventory_skew) > 0.001:
            self._skew_adjustments += 1

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

        # Check position limits before quoting
        can_buy = self._check_position_limit(inst_id, OrderSide.BUY, bid_qty)
        can_sell = self._check_position_limit(inst_id, OrderSide.SELL, ask_qty)

        # Submit new buy order (if within limits)
        if can_buy:
            buy_order = self._make_buy_order(inst_id, our_bid, bid_qty)
            if buy_order:
                note = f"pos={position:.0f} skew={inventory_skew:.4f} spread={adjusted_spread:.4f}"
                self.log_order_submit(buy_order, note=note)
                self.submit_order(buy_order)
                self._open_orders[inst_id][OrderSide.BUY] = buy_order
                self._quotes_sent += 1

        # Submit new sell order (if within limits)
        if can_sell:
            sell_order = self._make_sell_order(inst_id, our_ask, ask_qty)
            if sell_order:
                note = f"pos={position:.0f} skew={inventory_skew:.4f} spread={adjusted_spread:.4f}"
                self.log_order_submit(sell_order, note=note)
                self.submit_order(sell_order)
                self._open_orders[inst_id][OrderSide.SELL] = sell_order
                self._quotes_sent += 1

        # Verbose logging
        if self.verbose and self._quotes_sent % 100 == 0:
            print(f"[InventoryAwareMM] Quotes={self._quotes_sent} fills={self._fills} " +
                  f"limit_hits={self._position_limit_hits} skew_adj={self._skew_adjustments}")

    def on_order_filled(self, event) -> None:
        """
        Handle order fill events.

        Track fills and update position.
        """
        super().on_order_filled(event)

        self._fills += 1

        inst_id = event.instrument_id
        client_order_id = event.client_order_id
        side = event.order_side
        qty = float(event.last_qty)
        price = float(event.last_px)

        # Update position and cost basis
        self._update_position(inst_id, side, qty, price)

        # Remove filled order from tracking
        if inst_id in self._open_orders:
            for tracked_side, tracked_order in list(self._open_orders[inst_id].items()):
                if tracked_order and tracked_order.client_order_id == client_order_id:
                    del self._open_orders[inst_id][tracked_side]
                    break

        if self.verbose:
            position = self._get_position(inst_id)
            cost_basis = self._cost_basis[inst_id]
            side_str = "BUY" if side == OrderSide.BUY else "SELL"
            print(f"[InventoryAwareMM] FILL #{self._fills}: {inst_id} {side_str} " +
                  f"{qty}@{price:.3f} -> pos={position:.0f} basis={cost_basis:.3f}")

    def _on_stop(self) -> None:
        """Called when strategy stops."""
        # Cancel all open orders
        for inst_id, orders in self._open_orders.items():
            for order in orders.values():
                if order and order.is_open:
                    self.cancel_order(order)

        print(f"[InventoryAwareMM] Strategy stopped")
        print(f"[InventoryAwareMM] Statistics:")
        print(f"[InventoryAwareMM]   Data processed: {self._data_processed}")
        print(f"[InventoryAwareMM]   Quotes sent: {self._quotes_sent}")
        print(f"[InventoryAwareMM]   Fills: {self._fills}")
        print(f"[InventoryAwareMM]   Position limit hits: {self._position_limit_hits}")
        print(f"[InventoryAwareMM]   Skew adjustments: {self._skew_adjustments}")
        print(f"[InventoryAwareMM]   Effective BBO used: {self._effective_used} times")

        # Print final positions
        print(f"[InventoryAwareMM] Final positions:")
        for inst_id in sorted(self._positions.keys()):
            position = self._positions[inst_id]
            if abs(position) > 0.001:
                cost_basis = self._cost_basis[inst_id]
                print(f"[InventoryAwareMM]   {inst_id}: pos={position:.0f} basis={cost_basis:.3f}")
