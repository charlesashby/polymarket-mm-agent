"""
Order flow aware market-maker strategy for Polymarket binary options.

This strategy implements order flow bucket signal extraction (Milestone #9):
1. Subscribe to OrderFlowBucketDepth10CustomData
2. Parse add/cancel/trade counts per level
3. Calculate net flow (adds - cancels) per side
4. Detect aggressive vs passive flow patterns
5. Adjust quotes based on flow imbalance

Key improvements over ComplementAwareMM:
- ComplementAwareMM quotes symmetrically around effective mid
- OrderFlowMM adjusts spread and skew based on order flow signals
- Flow imbalance indicates directional pressure
- Should reduce adverse selection and improve PnL
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


class OrderFlowMM(BaseStrategy):
    """
    Order flow aware market-maker that adjusts quotes based on flow signals.

    Parameters
    ----------
    spread : float
        Base half-spread in decimal (e.g., 0.02 for 2 cents on each side)
    quote_notional : float
        Target notional per quote
    flow_lookback : int
        Number of flow buckets to analyze for signals (default 5)
    flow_skew_coeff : float
        Coefficient for flow-based skew adjustment (0.0 to 1.0, default 0.2)
    min_flow_threshold : float
        Minimum flow imbalance to trigger adjustment (default 100.0)
    verbose : bool
        Enable verbose logging
    """

    def __init__(
        self,
        spread: float = 0.02,
        quote_notional: Optional[float] = None,
        flow_lookback: int = 5,
        flow_skew_coeff: float = 0.2,
        min_flow_threshold: float = 100.0,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the order flow aware market-maker strategy.

        Parameters
        ----------
        spread : float, default 0.02
            Base half-spread to quote around effective mid price
        quote_notional : float, default None
            Target notional per quote (defaults to MAX_ORDER_NOTIONAL)
        flow_lookback : int, default 5
            Number of recent flow buckets to analyze
        flow_skew_coeff : float, default 0.2
            How much to adjust quotes based on flow (0.0 = no adjustment, 1.0 = max)
        min_flow_threshold : float, default 100.0
            Minimum absolute flow imbalance to trigger skew
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
            log_label="OrderFlowMM",
            **kwargs
        )

        self.spread = float(spread)
        if quote_notional is None:
            quote_notional = self._max_order_notional
        self.quote_notional = float(quote_notional)
        self.flow_lookback = int(flow_lookback)
        self.flow_skew_coeff = float(flow_skew_coeff)
        self.min_flow_threshold = float(min_flow_threshold)
        self.verbose = bool(verbose)

        # Track direct BBO for each instrument
        self._direct_bbo: Dict[InstrumentId, Dict[str, float]] = defaultdict(dict)

        # Track open orders per instrument
        self._open_orders: Dict[InstrumentId, Dict[OrderSide, Optional[LimitOrder]]] = defaultdict(dict)

        # Order flow tracking
        self._flow_history: Dict[InstrumentId, deque] = defaultdict(lambda: deque(maxlen=self.flow_lookback))
        self._last_flow_signal: Dict[InstrumentId, float] = {}

        # Statistics
        self._quotes_sent = 0
        self._fills = 0
        self._flow_updates = 0
        self._flow_skew_applied = 0
        self._effective_used = 0  # Count times we used effective BBO over direct

    def on_start(self) -> None:
        """Called when strategy starts - subscribe to market data."""
        self._on_start()
        if self.verbose:
            print(f"[{self._log_label}] Strategy started with spread={self.spread:.4f}, "
                  f"flow_lookback={self.flow_lookback}, flow_coeff={self.flow_skew_coeff:.4f}")

    def on_stop(self) -> None:
        """Called when strategy stops - print summary statistics."""
        if self.verbose:
            print(f"[{self._log_label}] Strategy stopped. Stats:")
            print(f"  Quotes sent: {self._quotes_sent}")
            print(f"  Fills: {self._fills}")
            print(f"  Flow updates: {self._flow_updates}")
            print(f"  Flow skew applied: {self._flow_skew_applied}")
            print(f"  Effective BBO used: {self._effective_used}")

    def _handle_quote_tick(self, tick: QuoteTick) -> None:
        """Handle quote tick updates."""
        # Update direct BBO
        self._direct_bbo[tick.instrument_id] = {
            "bid": float(tick.bid_price),
            "ask": float(tick.ask_price),
            "bid_size": float(tick.bid_size),
            "ask_size": float(tick.ask_size),
        }

    def _handle_trade_tick(self, tick: TradeTick) -> None:
        """Handle trade tick updates."""
        pass  # Not used for quoting logic

    def _handle_order_book_depth10(self, depth: OrderBookDepth10) -> None:
        """Handle depth updates - triggers quote updates."""
        inst_id = depth.instrument_id

        # Update direct BBO from depth
        if depth.bids and depth.asks:
            best_bid = depth.bids[0]
            best_ask = depth.asks[0]

            self._direct_bbo[inst_id] = {
                "bid": float(best_bid.price),
                "ask": float(best_ask.price),
                "bid_size": float(best_bid.size),
                "ask_size": float(best_ask.size),
            }

            # Update quotes based on new market state
            self._update_quotes(inst_id)

    def _handle_order_flow_data(self, data: OrderFlowBucketDepth10CustomData) -> None:
        """
        Handle order flow bucket updates - extract flow signals.

        Algorithm:
        1. Calculate net flow per side: net = adds - cancels (trade is separate)
        2. Calculate flow imbalance: bid_net - ask_net
        3. Positive imbalance = more buying pressure (bid adds > ask adds)
        4. Negative imbalance = more selling pressure (ask adds > bid adds)
        5. Store in history and calculate rolling average
        """
        inst_id = data.instrument_id
        self._flow_updates += 1

        # Calculate net flow per side (adds - cancels)
        # Sum across all 10 levels
        bid_adds = sum(data.bid_add_qty)
        bid_cancels = sum(data.bid_cancel_qty)
        ask_adds = sum(data.ask_add_qty)
        ask_cancels = sum(data.ask_cancel_qty)

        bid_net_flow = bid_adds - bid_cancels
        ask_net_flow = ask_adds - ask_cancels

        # Flow imbalance: positive = buying pressure, negative = selling pressure
        # When bid adds > ask adds, market participants are adding liquidity on bid
        # This often precedes price movement UP (they want to buy)
        flow_imbalance = bid_net_flow - ask_net_flow

        # Store in history
        self._flow_history[inst_id].append({
            "ts": data.ts_event,
            "bid_net": bid_net_flow,
            "ask_net": ask_net_flow,
            "imbalance": flow_imbalance,
        })

        # Calculate signal from recent history
        if len(self._flow_history[inst_id]) >= 2:
            # Average imbalance over lookback period
            avg_imbalance = sum(f["imbalance"] for f in self._flow_history[inst_id]) / len(self._flow_history[inst_id])
            self._last_flow_signal[inst_id] = avg_imbalance

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

    def _calculate_flow_skew(self, inst_id: InstrumentId) -> float:
        """
        Calculate price skew adjustment based on order flow signals.

        Returns a price adjustment in the same units as quotes:
        - Positive skew = shift quotes UP (buying pressure detected)
        - Negative skew = shift quotes DOWN (selling pressure detected)

        Algorithm:
        1. Get recent flow signal (average imbalance)
        2. If absolute imbalance < threshold, return 0 (no adjustment)
        3. Normalize imbalance and apply coefficient
        4. Return skew in price units (fraction of spread)
        """
        flow_signal = self._last_flow_signal.get(inst_id, 0.0)

        # Check if signal is significant enough
        if abs(flow_signal) < self.min_flow_threshold:
            return 0.0

        # Normalize and apply coefficient
        # Divide by threshold to get a [-1, 1] ish range, then multiply by coefficient and spread
        # Positive flow = buying pressure = raise quotes to avoid adverse selection
        # Negative flow = selling pressure = lower quotes to capture edge
        normalized_flow = flow_signal / (self.min_flow_threshold * 5.0)  # Scale down
        normalized_flow = max(-1.0, min(1.0, normalized_flow))  # Clamp to [-1, 1]

        skew = self.flow_skew_coeff * normalized_flow * self.spread

        if abs(skew) > 0.001:
            self._flow_skew_applied += 1

        return skew

    def _update_quotes(self, inst_id: InstrumentId) -> None:
        """
        Update quotes for an instrument with flow-aware adjustments.

        Logic:
        1. Compute effective BBO (direct + implied from complement)
        2. Calculate mid price from effective BBO
        3. Calculate flow-based skew adjustment
        4. Calculate quote prices with skew
        5. Enforce price bounds [0.01, 0.99]
        6. Cancel stale orders
        7. Submit new orders
        """
        # Compute effective BBO
        eff_bbo = self._compute_effective_bbo(inst_id)
        if not eff_bbo:
            return

        eff_mid = eff_bbo["mid"]

        # Calculate flow-based skew
        flow_skew = self._calculate_flow_skew(inst_id)

        # Calculate our quote prices around effective mid with flow skew
        # Positive skew shifts both quotes up (avoid buying into buying pressure)
        # Negative skew shifts both quotes down (aggressively capture selling pressure)
        our_bid = eff_mid - self.spread + flow_skew
        our_ask = eff_mid + self.spread + flow_skew

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
            note = f"eff_mid={eff_mid:.4f} flow_skew={flow_skew:.4f}"
            self.log_order_submit(buy_order, note=note)
            self.submit_order(buy_order)
            self._open_orders[inst_id][OrderSide.BUY] = buy_order
            self._quotes_sent += 1

        # Submit new sell order (if ALLOW_SELL_ORDERS is True)
        sell_order = self._make_sell_order(inst_id, our_ask, ask_qty)
        if sell_order:
            note = f"eff_mid={eff_mid:.4f} flow_skew={flow_skew:.4f}"
            self.log_order_submit(sell_order, note=note)
            self.submit_order(sell_order)
            self._open_orders[inst_id][OrderSide.SELL] = sell_order
            self._quotes_sent += 1

    def on_order_filled(self, event) -> None:
        """Handle order fills."""
        self._fills += 1
        inst_id = event.instrument_id

        # Clear filled order from tracking
        if inst_id in self._open_orders:
            for side, order in list(self._open_orders[inst_id].items()):
                if order and order.client_order_id == event.client_order_id:
                    self._open_orders[inst_id][side] = None

        if self.verbose:
            print(f"[{self._log_label}] Fill #{self._fills}: {inst_id.symbol} "
                  f"side={event.order_side} qty={event.last_qty} px={event.last_px}")

    def on_order_rejected(self, event) -> None:
        """Handle order rejections."""
        inst_id = event.instrument_id

        # Clear rejected order from tracking
        if inst_id in self._open_orders:
            for side, order in list(self._open_orders[inst_id].items()):
                if order and order.client_order_id == event.client_order_id:
                    self._open_orders[inst_id][side] = None

        if self.verbose:
            print(f"[{self._log_label}] Order rejected: {event.reason}")

    def on_order_canceled(self, event) -> None:
        """Handle order cancellations."""
        inst_id = event.instrument_id

        # Clear canceled order from tracking
        if inst_id in self._open_orders:
            for side, order in list(self._open_orders[inst_id].items()):
                if order and order.client_order_id == event.client_order_id:
                    self._open_orders[inst_id][side] = None
