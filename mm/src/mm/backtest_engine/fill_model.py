from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Tuple

from nautilus_trader.backtest.models.fill import FillModel
from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.core.rust.model import BookType, OrderSide
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDepth10, QuoteTick, BookOrder, TradeTick
from nautilus_trader.model.objects import Quantity, Price


# ---------------------------------------------------------------------
# Debug reasons for why we considered an order fillable
# ---------------------------------------------------------------------

_FILL_REASONS: dict[str, str] = {}
_FILL_PAYLOADS: dict[str, dict] = {}


def reset_fill_reasons() -> None:
    _FILL_REASONS.clear()
    _FILL_PAYLOADS.clear()


def get_fill_reasons() -> dict[str, str]:
    return dict(_FILL_REASONS)


def get_fill_payloads() -> dict[str, dict]:
    return {k: dict(v) for k, v in _FILL_PAYLOADS.items()}


def _record_reason(order, reason: str) -> None:
    client_id = getattr(order, "client_order_id", None)
    if client_id is None:
        return
    _FILL_REASONS[str(client_id)] = reason


def _record_payload(order, payload: dict) -> None:
    client_id = getattr(order, "client_order_id", None)
    if client_id is None:
        return
    _FILL_PAYLOADS[str(client_id)] = dict(payload)


# ---------------------------------------------------------------------
# Fill model
# ---------------------------------------------------------------------


class PolymarketBinaryFillModel(FillModel):
    """
    Fill model for Polymarket-style binary markets.

    Modeled behaviors:

      1) Effective BBO cross using direct + implied opposite-outcome BBO.
         We compute implied ladder/BBO from OPP depth (preferred) or OPP flow/quote:
           implied_bid_best ~= max(1 - opp_ask_level_price)
           implied_ask_best ~= min(1 - opp_bid_level_price)
         Then:
           eff_bid = max(direct_bid, implied_bid_best)
           eff_ask = min(direct_ask, implied_ask_best)

      2) Complementary CROSS matching (BUY/SELL across outcomes):
         - BUY(inst @ p) can match SELL(opp @ q) when q <= 1 - p
         - SELL(inst @ p) can match BUY(opp @ q) when q >= 1 - p

      3) Minting matching (BUY+BUY and SELL+SELL) using inequality (not exact complement):
         - BUY(inst @ p) can match BUY(opp @ q) when p + q >= 1  <=> q >= 1 - p
         - SELL(inst @ p) can match SELL(opp @ q) when p + q <= 1 <=> q <= 1 - p

      4) Passive/tape-based fill plausibility:
         If we see a TradeTick on this instrument at (approximately) the order price
         within a time window around the last depth snapshot timestamp, treat the
         order as fillable even if BBO/depth didn't show an instantaneous cross
         (4 Hz snapshots).

    Not modeled yet:
      - queue priority, time ordering, partial fill mechanics, multi-level consumption.

    Important:
      This model *does not* attempt to perfectly consume depth. It only injects
      a single synthetic contra order at the order price to allow Nautilus to fill.
    """

    def __init__(
        self,
        opposite_map: Mapping[str, str],
        bbo_cache: MutableMapping[str, Tuple[float, float]],
        depth_cache: MutableMapping[str, OrderBookDepth10],
        flow_cache: MutableMapping[str, object],
        trade_cache: MutableMapping[str, Tuple[int, float, Optional[int]]],
        passive_window_ns: int = 800_000_000,  # 0.8s; adjust for 4 Hz snapshots
        trade_price_tol: float = 0.011,  # binary ticks are often 0.01; tolerate float noise
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._opposite_map = dict(opposite_map)
        self._bbo_cache = bbo_cache
        self._depth_cache = depth_cache
        self._flow_cache = flow_cache
        self._trade_cache = trade_cache
        self._passive_window_ns = int(passive_window_ns)
        self._trade_price_tol = float(trade_price_tol)

    @staticmethod
    def _clip01(x: float) -> float:
        return min(0.999999, max(0.000001, float(x)))

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        """
        Return a synthetic OrderBook:
          - Apply our direct depth snapshot (Depth10) as the base book.
          - If we detect a Polymarket fill path, inject a single contra order at our
            order price with some size to force a cross.

        Nautilus uses the returned OrderBook for fill simulation.
        """
        inst_key = str(instrument.id)

        # --- Guards ---
        if order.side not in (OrderSide.BUY, OrderSide.SELL):
            return None
        if order.price is None:
            return None

        # Keep your prior "only do anything if we have depth" constraint
        if inst_key not in self._depth_cache:
            return None

        opp_key = self._opposite_map.get(inst_key)
        if not opp_key:
            return None

        order_px = float(order.price.as_double())
        # if abs(order_px - 0.01) <= 1e-12:
        #     _record_reason(order, "skip_fill_at_min_price")
        #     return None
        comp_px = self._clip01(1.0 - order_px)

        # --- Base book from our depth-10 snapshot ---
        depth = self._depth_cache.get(inst_key)
        book = OrderBook(instrument_id=instrument.id, book_type=BookType.L2_MBP)
        if depth is not None:
            book.apply_depth(depth)
        depth_ts_ns: Optional[int] = None
        if depth is not None and hasattr(depth, "ts_event"):
            try:
                depth_ts_ns = int(depth.ts_event)
            except Exception:
                depth_ts_ns = None

        # Helper: inject a contra order at *our* order price to force a cross
        rule_hit = False
        empty_book = OrderBook(instrument_id=instrument.id, book_type=BookType.L2_MBP)

        def _log_rule(reason: str, fields: dict) -> None:
            client_id = getattr(order, "client_order_id", "")
            side = getattr(order, "side", "")
            payload = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[fill_rule] order={client_id} side={side} reason={reason} {payload}")
            _record_payload(order, fields)

        def _inject_contra(size: float, reason: str, fields: Optional[dict] = None) -> None:
            nonlocal rule_hit
            contra = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
            _record_reason(order, reason)
            rule_hit = True
            if fields is not None:
                _log_rule(reason, fields)
            book.add(
                BookOrder(
                    side=contra,
                    price=_make_price(instrument, order_px),
                    size=_make_qty(instrument, float(max(0.0, size))),
                    order_id=1_000_003,
                ),
                0,
                0,
            )

        def _default_order_qty() -> float:
            order_qty = _qty_to_float(getattr(order, "quantity", None))
            if order_qty is None or order_qty <= 0.0:
                return 1_000_000.0
            return float(order_qty)

        # ==========================================================
        # Pull direct BBO from depth (prefer depth10 over quotes)
        # ==========================================================
        direct_bid_f: Optional[float] = None
        direct_ask_f: Optional[float] = None
        if depth is not None:
            for lvl in depth.bids:
                px, _sz = _level_px_size(lvl)
                if px is None or px <= 0.0:
                    continue
                direct_bid_f = px if direct_bid_f is None else max(direct_bid_f, px)
            for lvl in depth.asks:
                px, _sz = _level_px_size(lvl)
                if px is None or px <= 0.0:
                    continue
                direct_ask_f = px if direct_ask_f is None else min(direct_ask_f, px)

        direct_bid = _make_price(instrument, direct_bid_f) if direct_bid_f is not None else best_bid
        direct_ask = _make_price(instrument, direct_ask_f) if direct_ask_f is not None else best_ask
        depth_best_bid = float(direct_bid_f) if direct_bid_f is not None else None
        depth_best_ask = float(direct_ask_f) if direct_ask_f is not None else None

        # ==========================================================
        # Build implied BBO + implied-depth evidence from opposite
        # Prefer opposite DEPTH10 (better than L1 for your mismatch).
        # ==========================================================
        opp_depth = self._depth_cache.get(opp_key)

        implied_bid_best_f: Optional[float] = None  # max implied bid from opp asks: 1 - opp_ask
        implied_ask_best_f: Optional[float] = None  # min implied ask from opp bids: 1 - opp_bid

        if opp_depth is not None:
            # implied_bid_best = max(1 - ask_q)
            # implied_ask_best = min(1 - bid_q)
            for lvl in opp_depth.asks:
                px, _sz = _level_px_size(lvl)
                if px is None:
                    continue
                v = self._clip01(1.0 - float(px))
                if implied_bid_best_f is None or v > implied_bid_best_f:
                    implied_bid_best_f = v
            for lvl in opp_depth.bids:
                px, _sz = _level_px_size(lvl)
                if px is None:
                    continue
                v = self._clip01(1.0 - float(px))
                if implied_ask_best_f is None or v < implied_ask_best_f:
                    implied_ask_best_f = v

        # If we don't have opp_depth, fall back to opp flow/quote L1
        if implied_bid_best_f is None or implied_ask_best_f is None:
            opp_bid_f, opp_ask_f = self._bbo_cache.get(opp_key, (None, None))
            opp_flow = self._flow_cache.get(opp_key)
            if opp_flow is not None and opp_flow.bid_prices and opp_flow.ask_prices:
                opp_bid_f = float(opp_flow.bid_prices[0])
                opp_ask_f = float(opp_flow.ask_prices[0])
            if opp_bid_f is None or opp_ask_f is None:
                # Without opposite, we can't do polymarket-specific matching
                _record_reason(order, "missing_opp_bbo")
                return empty_book

            implied_bid_best_f = self._clip01(1.0 - float(opp_ask_f))
            implied_ask_best_f = self._clip01(1.0 - float(opp_bid_f))

        implied_bid = _make_price(instrument, float(implied_bid_best_f))
        implied_ask = _make_price(instrument, float(implied_ask_best_f))

        # ==========================================================
        # 1) Effective BBO cross (direct OR implied) -> immediate fill
        # ==========================================================
        eff_bid = direct_bid
        if eff_bid is None or float(implied_bid.as_double()) > float(eff_bid.as_double()):
            eff_bid = implied_bid

        eff_ask = direct_ask
        if eff_ask is None or float(implied_ask.as_double()) < float(eff_ask.as_double()):
            eff_ask = implied_ask

        if order.side == OrderSide.BUY:
            if eff_ask is not None and float(eff_ask.as_double()) <= order_px:
                _inject_contra(
                    _default_order_qty(),
                    "eff_bbo_cross",
                    {
                        "order_px": order_px,
                        "direct_ask": float(direct_ask.as_double()) if direct_ask else None,
                        "implied_ask": float(implied_ask.as_double()),
                        "eff_ask": float(eff_ask.as_double()),
                        "depth_ts_ns": depth_ts_ns,
                        "depth_best_ask": depth_best_ask,
                    },
                )
                return book
        else:  # SELL
            if eff_bid is not None and float(eff_bid.as_double()) >= order_px:
                _inject_contra(
                    _default_order_qty(),
                    "eff_bbo_cross",
                    {
                        "order_px": order_px,
                        "direct_bid": float(direct_bid.as_double()) if direct_bid else None,
                        "implied_bid": float(implied_bid.as_double()),
                        "eff_bid": float(eff_bid.as_double()),
                        "depth_ts_ns": depth_ts_ns,
                        "depth_best_bid": depth_best_bid,
                    },
                )
                return book

        # ==========================================================
        # 2) Complementary CROSS matching using implied depth ladder
        #
        # BUY(inst@p) matches SELL(opp@q) when q <= 1-p  (uses opp ASKS)
        # SELL(inst@p) matches BUY(opp@q) when q >= 1-p (uses opp BIDS)
        # ==========================================================
        if opp_depth is not None:
            if order.side == OrderSide.BUY:
                # Look at opp ASKS with price <= comp_px
                avail = _sum_side_size_le(opp_depth.asks, comp_px)
                if avail > 0.0:
                    _inject_contra(avail, "complement_cross_buy_vs_opp_ask")
                    return book
            else:
                # SELL: look at opp BIDS with price >= comp_px
                avail = _sum_side_size_ge(opp_depth.bids, comp_px)
                if avail > 0.0:
                    _inject_contra(avail, "complement_cross_sell_vs_opp_bid")
                    return book

        # ==========================================================
        # 3) Minting inequality (BUY+BUY, SELL+SELL)
        #
        # BUY+BUY: BUY(inst@p) matches BUY(opp@q) when q >= 1-p  (opp BIDS)
        # SELL+SELL: SELL(inst@p) matches SELL(opp@q) when q <= 1-p (opp ASKS)
        #
        # NOTE: These conditions may overlap with complementary-cross checks.
        # We keep separate reasons to debug which path triggers.
        # ==========================================================
        if opp_depth is not None:
            if order.side == OrderSide.BUY:
                avail = _sum_side_size_ge(opp_depth.bids, comp_px)
                if avail > 0.0:
                    _inject_contra(
                        avail,
                        "mint_buy_buy_ineq",
                        {"order_px": order_px, "comp_px": comp_px, "avail": avail},
                    )
                    return book
            else:
                avail = _sum_side_size_le(opp_depth.asks, comp_px)
                if avail > 0.0:
                    _inject_contra(
                        avail,
                        "mint_sell_sell_ineq",
                        {"order_px": order_px, "comp_px": comp_px, "avail": avail},
                    )
                    return book

        # ==========================================================
        # 4) Passive/tape-based fill plausibility (4 Hz snapshots)
        #
        # If we saw a trade at (approximately) our price close to the last
        # depth snapshot time, treat as fillable.
        #
        # We intentionally keep this simple: we don't model size/priority.
        # ==========================================================
        # Anchor "now" to our depth snapshot timestamp (best available in this function)
        now_ts_ns: Optional[int] = None
        try:
            if depth is not None and hasattr(depth, "ts_event"):
                now_ts_ns = int(depth.ts_event)
        except Exception:
            now_ts_ns = None

        t = self._trade_cache.get(inst_key)
        if now_ts_ns is not None and t is not None:
            trade_ts_ns, trade_px, _aggr = t
            if abs(int(trade_ts_ns) - int(now_ts_ns)) <= self._passive_window_ns and abs(float(trade_px) - float(order_px)) <= self._trade_price_tol:
                _inject_contra(
                    _default_order_qty(),
                    "passive_trade_at_price",
                    {
                        "order_px": order_px,
                        "trade_px": trade_px,
                        "trade_ts_ns": trade_ts_ns,
                        "depth_ts_ns": now_ts_ns,
                    },
                )
                return book

        return empty_book


# ---------------------------------------------------------------------
# Cache modules (included as requested)
# ---------------------------------------------------------------------


class QuoteBboCacheModule(SimulationModule):
    """
    Keep a best-bid/ask cache from QuoteTick updates for fill model lookups.
    """

    def __init__(self, bbo_cache: MutableMapping[str, Tuple[float, float]]) -> None:
        super().__init__(SimulationModuleConfig())
        self._bbo_cache = bbo_cache

    def pre_process(self, data) -> None:
        if not isinstance(data, QuoteTick):
            return
        bid = _price_to_float(data.bid_price)
        ask = _price_to_float(data.ask_price)
        if bid is None or ask is None:
            return
        self._bbo_cache[str(data.instrument_id)] = (bid, ask)

    def process(self, ts_now) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        self._bbo_cache.clear()


class Depth10CacheModule(SimulationModule):
    """
    Cache the latest OrderBookDepth10 per instrument.
    """

    def __init__(self, depth_cache: MutableMapping[str, OrderBookDepth10]) -> None:
        super().__init__(SimulationModuleConfig())
        self._depth_cache = depth_cache

    def pre_process(self, data) -> None:
        if isinstance(data, OrderBookDepth10):
            self._depth_cache[str(data.instrument_id)] = data

    def process(self, ts_now) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        self._depth_cache.clear()


class OrderFlowBucketCacheModule(SimulationModule):
    """
    Cache the latest order-flow bucket snapshot per instrument.
    """

    def __init__(self, flow_cache: MutableMapping[str, object]) -> None:
        super().__init__(SimulationModuleConfig())
        self._flow_cache = flow_cache

    def pre_process(self, data) -> None:
        payload = getattr(data, "data", data)
        inst_id = getattr(payload, "instrument_id", None)
        if inst_id is None:
            return
        self._flow_cache[str(inst_id)] = payload

    def process(self, ts_now) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        self._flow_cache.clear()


class TradeTickCacheModule(SimulationModule):
    """
    Cache the latest TradeTick per instrument for tape-based passive-fill plausibility.

    We store: (ts_event_ns, price_float, aggressor_side_int|None)
    """

    def __init__(self, trade_cache: MutableMapping[str, Tuple[int, float, Optional[int]]]) -> None:
        super().__init__(SimulationModuleConfig())
        self._trade_cache = trade_cache

    def pre_process(self, data) -> None:
        if not isinstance(data, TradeTick):
            return
        px = _price_to_float(data.price)
        if px is None:
            return
        ts_ns = int(getattr(data, "ts_event", 0))
        aggr = None
        try:
            aggr = int(getattr(data, "aggressor_side"))
        except Exception:
            aggr = None
        self._trade_cache[str(data.instrument_id)] = (ts_ns, float(px), aggr)

    def process(self, ts_now) -> None:
        return

    def log_diagnostics(self, logger) -> None:
        return

    def reset(self) -> None:
        self._trade_cache.clear()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _price_to_float(px) -> Optional[float]:
    if px is None:
        return None
    try:
        return float(px.as_double())
    except Exception:
        try:
            return float(px)
        except Exception:
            return None


def _make_price(instrument, value: float) -> Price:
    if hasattr(instrument, "make_price"):
        return instrument.make_price(float(value))
    return Price(float(value), instrument.price_precision)


def _make_qty(instrument, value: float) -> Quantity:
    return Quantity(float(value), instrument.size_precision)


def _qty_to_float(qty) -> Optional[float]:
    if qty is None:
        return None
    try:
        return float(qty.as_double())
    except Exception:
        try:
            return float(qty)
        except Exception:
            return None


def _level_px_size(lvl) -> tuple[Optional[float], Optional[float]]:
    try:
        px = float(lvl.price.as_double())
    except Exception:
        try:
            px = float(lvl.price)
        except Exception:
            return None, None
    try:
        sz = float(lvl.size.as_double())
    except Exception:
        try:
            sz = float(lvl.size)
        except Exception:
            return px, None
    return px, sz

def _sum_side_size_ge(levels, threshold_px: float) -> float:
    """
    Sum sizes for levels with price >= threshold.
    Used for opposite BIDS in:
      - SELL(inst) complementary cross vs opp bids
      - BUY+BUY minting inequality
    """
    tot = 0.0
    thr = float(threshold_px)
    for lvl in levels:
        px, sz = _level_px_size(lvl)
        if px is None or sz is None:
            continue
        if px + 1e-12 >= thr:
            tot += max(0.0, sz)
    return tot


def _sum_side_size_le(levels, threshold_px: float) -> float:
    """
    Sum sizes for levels with price <= threshold.
    Used for opposite ASKS in:
      - BUY(inst) complementary cross vs opp asks
      - SELL+SELL minting inequality
    """
    tot = 0.0
    thr = float(threshold_px)
    for lvl in levels:
        px, sz = _level_px_size(lvl)
        if px is None or sz is None:
            continue
        if px <= thr + 1e-12:
            tot += max(0.0, sz)
    return tot
