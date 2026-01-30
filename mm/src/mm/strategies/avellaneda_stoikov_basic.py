from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.identifiers import InstrumentId

from mm.src.mm.strategies.base_strategy import BaseStrategy


@dataclass
class EventState:
    start_dt: datetime
    end_dt: datetime
    ref: float


@dataclass
class ActiveLeg:
    bid_coid: Optional[str] = None
    ask_coid: Optional[str] = None
    last_bid_px: Optional[float] = None
    last_ask_px: Optional[float] = None


@dataclass
class ActivePair:
    yes: ActiveLeg = field(default_factory=ActiveLeg)
    no: ActiveLeg = field(default_factory=ActiveLeg)


@dataclass
class BBO:
    bid: Optional[float] = None
    ask: Optional[float] = None
    ts_event: int = 0


class AvellanedaStoikovBasicStrategy(BaseStrategy):
    """
    Basic market-maker inspired by Avellaneda-Stoikov:
    - Maintains simple fair values from YES/NO mid-prices.
    - Quotes around fair with a fixed edge, skewed by inventory.
    - Uses BaseStrategy helpers for min qty / price bounds.
    """

    def __init__(
        self,
        *,
        edge: float = 0.02,
        max_position: int = 200,
        quote_notional: Optional[float] = None,
        min_quote_interval_secs: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(
            active_pair_factory=ActivePair,
            event_state_factory=EventState,
            log_label="ASBasic",
            **kwargs,
        )
        self.edge = float(edge)
        self.max_position = int(max_position)
        if quote_notional is None:
            quote_notional = self._max_order_notional
        self.quote_notional = float(quote_notional)
        self.min_quote_interval_secs = float(min_quote_interval_secs)

        self._bbo_by_id: Dict[InstrumentId, BBO] = {}
        self._last_quote_ts_by_slug: Dict[str, int] = {}

    def on_start(self) -> None:
        # Ensure BaseStrategy subscriptions and meta handling are wired.
        super()._on_start()

    # ------------------------------------------------------------------
    # Order book handling
    # ------------------------------------------------------------------
    def _handle_order_book_depth10(self, book: OrderBookDepth10) -> None:
        inst_id = book.instrument_id
        bid = None
        ask = None
        if book.bids and len(book.bids) > 0:
            bid = self._px_to_float(book.bids[0].price)
        if book.asks and len(book.asks) > 0:
            ask = self._px_to_float(book.asks[0].price)
        if bid is None or ask is None:
            return

        ts_event = int(getattr(book, "ts_event", 0) or 0)
        if ts_event <= 0:
            ts_event = int(getattr(book, "ts_init", 0) or 0)
        if ts_event <= 0:
            return

        self._bbo_by_id[inst_id] = BBO(bid=bid, ask=ask, ts_event=ts_event)

        meta = self._meta_by_id.get(inst_id)
        if not meta:
            return
        start_dt = meta.get("start_dt")
        end_dt = meta.get("end_dt")
        if isinstance(start_dt, datetime) and isinstance(end_dt, datetime):
            now_dt = datetime.fromtimestamp(ts_event / 1_000_000_000, tz=timezone.utc)
            if now_dt < start_dt or now_dt > end_dt:
                return
        slug = meta.get("slug")
        if not isinstance(slug, str):
            return

        # Only quote if both legs have BBOs
        yes_id = self._yes_by_slug.get(slug)
        no_id = self._no_by_slug.get(slug)
        if yes_id is None or no_id is None:
            return
        if yes_id not in self._bbo_by_id or no_id not in self._bbo_by_id:
            return

        now_ts = ts_event
        last_ts = self._last_quote_ts_by_slug.get(slug, 0)
        if now_ts - last_ts < int(self.min_quote_interval_secs * 1_000_000_000):
            return

        self._last_quote_ts_by_slug[slug] = now_ts

        yes_bbo = self._bbo_by_id[yes_id]
        no_bbo = self._bbo_by_id[no_id]
        fair_yes = self._fair_yes_price(yes_bbo, no_bbo)
        fair_no = 1.0 - fair_yes

        self._quote_leg(yes_id, fair_yes)
        self._quote_leg(no_id, fair_no)

    def _fair_yes_price(self, yes_bbo: BBO, no_bbo: BBO) -> float:
        yes_mid = 0.5 * (yes_bbo.bid + yes_bbo.ask)
        no_mid = 0.5 * (no_bbo.bid + no_bbo.ask)
        implied_from_no = 1.0 - no_mid
        fair = 0.5 * (yes_mid + implied_from_no)
        return float(min(self.px_ceil, max(self.px_floor, fair)))

    @staticmethod
    def _px_to_float(px) -> Optional[float]:
        if px is None:
            return None
        try:
            return float(px.as_double())
        except Exception:
            try:
                return float(px)
            except Exception:
                return None

    def _quote_leg(self, inst_id: InstrumentId, fair: float) -> None:
        # Inventory skew: push price away from fair when long.
        pos, _notional = self.net_exposure(inst_id)
        skew = 0.0
        if self.max_position > 0:
            skew = float(np.clip(pos / self.max_position, -1.0, 1.0)) * self.edge

        bid_px = float(min(self.px_ceil, max(self.px_floor, fair - self.edge - skew)))
        ask_px = float(min(self.px_ceil, max(self.px_floor, fair + self.edge - skew)))

        bid_qty = self.qty_for_notional(bid_px, self.quote_notional)
        ask_qty = self.qty_for_notional(ask_px, self.quote_notional)

        # Cancel outstanding orders for this instrument before re-quoting.
        for order in self.cache.orders():
            if order.instrument_id != inst_id:
                continue
            if order.status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
                continue
            self.cancel_order(order)

        bid_order = self._make_buy_order(inst_id, bid_px, bid_qty)
        # ask_order = self._make_sell_order(inst_id, ask_px, ask_qty)

        if bid_order is not None:
            self.log_order_submit(bid_order, note=f"fair={fair:.3f}")
            self.submit_order(bid_order)
        # if ask_order is not None:
        #     self.log_order_submit(ask_order, note=f"fair={fair:.3f}")
        #     self.submit_order(ask_order)
