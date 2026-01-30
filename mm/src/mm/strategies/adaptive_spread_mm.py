"""
Adaptive spread market-maker strategy for Polymarket binary options.

This strategy implements adaptive spread calculation (Milestone #5):
1. Base spread as configurable parameter
2. Volatility estimate from recent price moves
3. Scale spread with volatility metric
4. Optional adverse selection adjustment from order flow
5. Minimum spread floor for safety
6. Test spread adapts to market conditions

Key improvements over InventoryAwareMM:
- InventoryAwareMM uses fixed base spread with position-based widening
- AdaptiveSpreadMM dynamically adjusts spread based on market volatility
- Reduces adverse selection in high volatility periods
- Tightens spread in calm markets to capture more flow
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from collections import defaultdict, deque

from nautilus_trader.model.data import QuoteTick, TradeTick, OrderBookDepth10
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import LimitOrder

from mm.src.mm.strategies.base_strategy import BaseStrategy
from mm.src.mm.types import ChainlinkCustomData, OrderFlowBucketDepth10CustomData


class AdaptiveSpreadMM(BaseStrategy):
    """
    Adaptive spread market-maker with volatility-based spread adjustments.

    Parameters
    ----------
    base_spread : float
        Base half-spread in decimal (e.g., 0.015 for 1.5 cents on each side)
    min_spread : float
        Minimum half-spread floor (e.g., 0.005 for 0.5 cents)
    max_spread : float
        Maximum half-spread ceiling (e.g., 0.05 for 5 cents)
    vol_lookback : int
        Number of mid price updates to use for volatility calculation
    vol_multiplier : float
        Multiplier for volatility-based spread adjustment (e.g., 2.0)
    quote_size : int
        Fixed quote size in contracts
    position_limit : int
        Maximum absolute position per instrument
    skew_coefficient : float
        Coefficient for inventory skew adjustment (0.0 to 1.0)
    use_order_flow : bool
        Whether to use order flow imbalance for adverse selection adjustment
    verbose : bool
        Enable verbose logging
    """

    def __init__(
        self,
        base_spread: float = 0.015,
        min_spread: float = 0.005,
        max_spread: float = 0.05,
        vol_lookback: int = 20,
        vol_multiplier: float = 2.0,
        quote_size: int = 100,
        position_limit: int = 5000,
        skew_coefficient: float = 0.3,
        use_order_flow: bool = False,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the adaptive spread market-maker strategy.

        Parameters
        ----------
        base_spread : float, default 0.015
            Base half-spread before volatility adjustment
        min_spread : float, default 0.005
            Minimum half-spread floor (safety)
        max_spread : float, default 0.05
            Maximum half-spread ceiling
        vol_lookback : int, default 20
            Number of price updates for volatility window
        vol_multiplier : float, default 2.0
            How much to scale spread with volatility
        quote_size : int, default 100
            Fixed size for each quote
        position_limit : int, default 5000
            Maximum absolute position per instrument
        skew_coefficient : float, default 0.3
            Inventory skew coefficient (0.0 = no skew, 1.0 = max skew)
        use_order_flow : bool, default False
            Use order flow imbalance for spread adjustment
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
                "mid_prices": deque(maxlen=vol_lookback),  # Recent mid prices for volatility
                "last_spread": base_spread,  # Track last computed spread
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
            log_label="AdaptiveSpreadMM",
            metrics_enabled=False,
            **kwargs
        )

        self.base_spread = float(base_spread)
        self.min_spread = float(min_spread)
        self.max_spread = float(max_spread)
        self.vol_lookback = int(vol_lookback)
        self.vol_multiplier = float(vol_multiplier)
        self.quote_size = int(quote_size)
        self.position_limit = int(position_limit)
        self.skew_coefficient = float(skew_coefficient)
        self.use_order_flow = bool(use_order_flow)
        self.verbose = bool(verbose)

        # Track open orders per instrument
        self._open_orders: Dict[InstrumentId, Dict[OrderSide, LimitOrder]] = defaultdict(dict)

        # Track direct BBO per instrument
        self._direct_bbo: Dict[InstrumentId, Dict[str, float]] = defaultdict(dict)

        # Track position and cost basis per instrument
        self._positions: Dict[InstrumentId, float] = defaultdict(float)
        self._cost_basis: Dict[InstrumentId, float] = defaultdict(float)
        self._position_value: Dict[InstrumentId, float] = defaultdict(float)

        # Track order flow imbalance per instrument
        self._order_flow_imbalance: Dict[InstrumentId, float] = defaultdict(float)

        # Statistics
        self._fills = 0
        self._quotes_sent = 0
        self._data_processed = 0
        self._effective_used = 0
        self._position_limit_hits = 0
        self._skew_adjustments = 0
        self._spread_adjustments = 0
        self._total_volatility = 0.0
        self._vol_samples = 0

    def on_start(self) -> None:
        """Called when strategy starts."""
        super()._on_start()

        print(f"[AdaptiveSpreadMM] Strategy started")
        print(f"[AdaptiveSpreadMM]   Base spread: {self.base_spread}")
        print(f"[AdaptiveSpreadMM]   Min spread: {self.min_spread}")
        print(f"[AdaptiveSpreadMM]   Max spread: {self.max_spread}")
        print(f"[AdaptiveSpreadMM]   Vol lookback: {self.vol_lookback}")
        print(f"[AdaptiveSpreadMM]   Vol multiplier: {self.vol_multiplier}")
        print(f"[AdaptiveSpreadMM]   Quote size: {self.quote_size}")
        print(f"[AdaptiveSpreadMM]   Position limit: {self.position_limit}")
        print(f"[AdaptiveSpreadMM]   Skew coefficient: {self.skew_coefficient}")
        print(f"[AdaptiveSpreadMM]   Use order flow: {self.use_order_flow}")
        print(f"[AdaptiveSpreadMM]   Verbose: {self.verbose}")
        print(f"[AdaptiveSpreadMM]   Tracking {len(self._slugs)} slug pairs")

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
            print(f"[AdaptiveSpreadMM] Depth update #{self._data_processed} for {book.instrument_id}")

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
        """
        Handle order flow bucket updates.

        Calculate order flow imbalance for adverse selection detection.
        """
        if not self.use_order_flow:
            return

        inst_id = data.instrument_id

        # Calculate net order flow (adds - cancels) on each side
        # Positive = aggressive buying pressure, negative = aggressive selling pressure
        bid_flow = 0.0
        ask_flow = 0.0

        # Sum across all depth levels
        for i in range(10):
            if i < len(data.bid_add_counts):
                bid_flow += data.bid_add_counts[i]
            if i < len(data.bid_cancel_counts):
                bid_flow -= data.bid_cancel_counts[i]
            if i < len(data.ask_add_counts):
                ask_flow += data.ask_add_counts[i]
            if i < len(data.ask_cancel_counts):
                ask_flow -= data.ask_cancel_counts[i]

        # Calculate imbalance: positive = buy pressure, negative = sell pressure
        total_flow = abs(bid_flow) + abs(ask_flow)
        if total_flow > 0:
            imbalance = (bid_flow - ask_flow) / total_flow
        else:
            imbalance = 0.0

        # Store imbalance (exponential moving average)
        alpha = 0.3  # Smoothing factor
        self._order_flow_imbalance[inst_id] = (
            alpha * imbalance + (1 - alpha) * self._order_flow_imbalance.get(inst_id, 0.0)
        )

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

    def _calculate_volatility(self, inst_id: InstrumentId) -> float:
        """
        Calculate recent volatility from mid price moves.

        Uses standard deviation of log returns over the lookback window.

        Returns
        -------
        float
            Volatility estimate (standard deviation of returns)
        """
        # Get active pair state
        slug = self._meta_by_id.get(inst_id, {}).get("slug")
        if not slug or slug not in self._active_by_slug:
            return 0.0

        pair_state = self._active_by_slug[slug]
        mid_prices = pair_state.get("mid_prices")

        if not mid_prices or len(mid_prices) < 2:
            return 0.0

        # Calculate log returns
        returns = []
        for i in range(1, len(mid_prices)):
            if mid_prices[i] > 0 and mid_prices[i-1] > 0:
                log_return = abs((mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1])
                returns.append(log_return)

        if not returns:
            return 0.0

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5

        return volatility

    def _calculate_adaptive_spread(self, inst_id: InstrumentId) -> float:
        """
        Calculate adaptive spread based on volatility and order flow.

        Returns
        -------
        float
            Adaptive half-spread to use for quoting
        """
        # Start with base spread
        spread = self.base_spread

        # Add volatility-based adjustment
        volatility = self._calculate_volatility(inst_id)
        if volatility > 0:
            vol_adjustment = self.vol_multiplier * volatility
            spread += vol_adjustment

            # Track statistics
            self._total_volatility += volatility
            self._vol_samples += 1
            self._spread_adjustments += 1

        # Add order flow imbalance adjustment (if enabled)
        if self.use_order_flow:
            imbalance = self._order_flow_imbalance.get(inst_id, 0.0)
            # High imbalance (either direction) = increase spread to avoid adverse selection
            flow_adjustment = abs(imbalance) * self.base_spread * 0.5
            spread += flow_adjustment

        # Enforce min/max bounds
        spread = max(self.min_spread, min(spread, self.max_spread))

        return spread

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

    def _calculate_inventory_skew(self, inst_id: InstrumentId, spread: float) -> float:
        """
        Calculate inventory skew adjustment for quote prices.

        Returns a value that should be added to quote prices to encourage position reduction.

        Positive skew = shift quotes higher (encourage selling)
        Negative skew = shift quotes lower (encourage buying)

        Parameters
        ----------
        inst_id : InstrumentId
            Instrument to calculate skew for
        spread : float
            Current spread (used for scaling)

        Returns
        -------
        float
            Skew adjustment in price units
        """
        position = self._get_position(inst_id)

        if abs(position) < 0.001:
            return 0.0

        # Calculate position as fraction of limit
        position_pct = position / self.position_limit

        # Apply skew coefficient
        # When long (position > 0), skew > 0, which shifts quotes up (encourages selling)
        # When short (position < 0), skew < 0, which shifts quotes down (encourages buying)
        skew = self.skew_coefficient * position_pct * spread

        return skew

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
        Update quotes for an instrument with adaptive spread.

        Logic:
        1. Compute effective BBO (direct + implied from complement)
        2. Calculate mid price from effective BBO
        3. Store mid price for volatility calculation
        4. Calculate adaptive spread based on volatility and order flow
        5. Calculate inventory skew adjustment
        6. Calculate quote prices with adaptive spread and skew
        7. Check position limits before quoting each side
        8. Enforce price bounds [0.01, 0.99]
        9. Cancel stale orders
        10. Submit new orders
        """
        # Compute effective BBO
        eff_bbo = self._compute_effective_bbo(inst_id)
        if not eff_bbo:
            return

        eff_mid = eff_bbo["mid"]

        # Store mid price for volatility calculation
        slug = self._meta_by_id.get(inst_id, {}).get("slug")
        if slug and slug in self._active_by_slug:
            pair_state = self._active_by_slug[slug]
            mid_prices = pair_state.get("mid_prices")
            if mid_prices is not None:
                mid_prices.append(eff_mid)

        # Calculate adaptive spread
        adaptive_spread = self._calculate_adaptive_spread(inst_id)

        # Get current position
        position = self._get_position(inst_id)

        # Calculate inventory skew (shifts quotes to reduce position)
        inventory_skew = self._calculate_inventory_skew(inst_id, adaptive_spread)

        # Calculate our quote prices around effective mid with skew
        # Skew shifts both quotes in same direction to encourage unwinding
        our_bid = eff_mid - adaptive_spread + inventory_skew
        our_ask = eff_mid + adaptive_spread + inventory_skew

        # Enforce price bounds [0.01, 0.99]
        our_bid = max(self.px_floor, min(our_bid, self.px_ceil))
        our_ask = max(self.px_floor, min(our_ask, self.px_ceil))

        # Don't cross the spread
        if our_bid >= our_ask:
            return

        # Track if we made skew adjustments
        if abs(inventory_skew) > 0.001:
            self._skew_adjustments += 1

        # Store last spread for reporting
        if slug and slug in self._active_by_slug:
            self._active_by_slug[slug]["last_spread"] = adaptive_spread

        # Check if we need to update orders
        open_orders = self._open_orders.get(inst_id, {})

        # Cancel existing orders
        for side, order in list(open_orders.items()):
            if order and order.is_open:
                self.cancel_order(order)

        # Clear tracking
        self._open_orders[inst_id] = {}

        # Check position limits before quoting
        can_buy = self._check_position_limit(inst_id, OrderSide.BUY, self.quote_size)
        can_sell = self._check_position_limit(inst_id, OrderSide.SELL, self.quote_size)

        # Submit new buy order (if within limits)
        if can_buy:
            buy_order = self._make_buy_order(inst_id, our_bid, self.quote_size)
            if buy_order:
                note = f"pos={position:.0f} spread={adaptive_spread:.4f} skew={inventory_skew:.4f}"
                self.log_order_submit(buy_order, note=note)
                self.submit_order(buy_order)
                self._open_orders[inst_id][OrderSide.BUY] = buy_order
                self._quotes_sent += 1

        # Submit new sell order (if within limits)
        if can_sell:
            sell_order = self._make_sell_order(inst_id, our_ask, self.quote_size)
            if sell_order:
                note = f"pos={position:.0f} spread={adaptive_spread:.4f} skew={inventory_skew:.4f}"
                self.log_order_submit(sell_order, note=note)
                self.submit_order(sell_order)
                self._open_orders[inst_id][OrderSide.SELL] = sell_order
                self._quotes_sent += 1

        # Verbose logging
        if self.verbose and self._quotes_sent % 100 == 0:
            print(f"[AdaptiveSpreadMM] Quotes={self._quotes_sent} fills={self._fills} " +
                  f"limit_hits={self._position_limit_hits} spread_adj={self._spread_adjustments}")

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
            print(f"[AdaptiveSpreadMM] FILL #{self._fills}: {inst_id} {side_str} " +
                  f"{qty}@{price:.3f} -> pos={position:.0f} basis={cost_basis:.3f}")

    def _on_stop(self) -> None:
        """Called when strategy stops."""
        # Cancel all open orders
        for inst_id, orders in self._open_orders.items():
            for order in orders.values():
                if order and order.is_open:
                    self.cancel_order(order)

        avg_vol = self._total_volatility / self._vol_samples if self._vol_samples > 0 else 0.0

        print(f"[AdaptiveSpreadMM] Strategy stopped")
        print(f"[AdaptiveSpreadMM] Statistics:")
        print(f"[AdaptiveSpreadMM]   Data processed: {self._data_processed}")
        print(f"[AdaptiveSpreadMM]   Quotes sent: {self._quotes_sent}")
        print(f"[AdaptiveSpreadMM]   Fills: {self._fills}")
        print(f"[AdaptiveSpreadMM]   Position limit hits: {self._position_limit_hits}")
        print(f"[AdaptiveSpreadMM]   Skew adjustments: {self._skew_adjustments}")
        print(f"[AdaptiveSpreadMM]   Spread adjustments: {self._spread_adjustments}")
        print(f"[AdaptiveSpreadMM]   Average volatility: {avg_vol:.6f}")
        print(f"[AdaptiveSpreadMM]   Effective BBO used: {self._effective_used} times")

        # Print final positions
        print(f"[AdaptiveSpreadMM] Final positions:")
        for inst_id in sorted(self._positions.keys()):
            position = self._positions[inst_id]
            if abs(position) > 0.001:
                cost_basis = self._cost_basis[inst_id]
                print(f"[AdaptiveSpreadMM]   {inst_id}: pos={position:.0f} basis={cost_basis:.3f}")
