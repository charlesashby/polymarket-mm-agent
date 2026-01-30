from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import time
import math
import numpy as np
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union, Callable, Tuple

from nautilus_trader.core import Data
from nautilus_trader.model.data import QuoteTick, TradeTick, OrderBookDepth10, CustomData
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.enums import OrderSide, OrderStatus, TimeInForce

from mm.src.mm.types import CHAINLINK_DATATYPE, ORDER_FLOW_BUCKET_DEPTH10_DATATYPE, POLYMARKET_META_DATATYPE, UNIVERSE_TOPIC, ChainlinkCustomData, OrderFlowBucketDepth10CustomData, PolymarketMetaCustomData
from mm.src.mm.utils import get_open_exposure


MAX_ORDER_NOTIONAL = 5.0
ALLOW_SELL_ORDERS = False


class BaseStrategy(Strategy):
    def __init__(
        self,
        *,
        active_pair_factory: Callable[[], object],
        event_state_factory: Callable[..., object],
        log_label: str = "BaseStrategy",
        metrics_enabled: bool = True,
        px_floor: float = 0.01,
        px_ceil: float = 0.99,
        min_qty: int = 1,
        min_notional: float = 1.0,
    ) -> None:
        super().__init__()
        if active_pair_factory is None or event_state_factory is None:
            raise ValueError("BaseStrategy requires active_pair_factory and event_state_factory.")

        self.min_qty = int(min_qty)
        self.min_notional = float(max(0.0, min_notional))

        self.px_floor = float(px_floor)
        self.px_ceil = float(px_ceil)

        self._log_label = str(log_label)
        self._active_pair_factory = active_pair_factory
        self._event_state_factory = event_state_factory
        self._metrics_enabled = bool(metrics_enabled)

        # DataRecorder-style instrument tracking
        self._target_ids: Set[InstrumentId] = set()
        self._subscribed_ids: Set[InstrumentId] = set()
        self._pending_requests: Set[InstrumentId] = set()

        # Polymarket meta bookkeeping
        self._meta_by_id: Dict[InstrumentId, Dict[str, object]] = {}
        self._state_by_id: Dict[InstrumentId, object] = {}
        self._metas_by_start_second: Dict[datetime, List[InstrumentId]] = {}
        self._yes_by_slug: Dict[str, InstrumentId] = {}
        self._no_by_slug: Dict[str, InstrumentId] = {}
        self._slugs: List[str] = []
        self._active_by_slug: Dict[str, object] = {}
        self._inv_cost: Dict[InstrumentId, Tuple[int, float]] = {}
        self._max_order_notional = float(MAX_ORDER_NOTIONAL)

    def _request_instrument_if_needed(self, inst_id: InstrumentId) -> None:
        if inst_id in self._subscribed_ids or inst_id in self._pending_requests:
            return
        self._pending_requests.add(inst_id)
        self.request_instrument(inst_id)

    def _subscribe_instrument(self, inst_id: InstrumentId) -> None:
        if inst_id in self._subscribed_ids:
            return
        self.subscribe_quote_ticks(inst_id)
        self.subscribe_trade_ticks(inst_id)
        self.subscribe_order_book_depth(inst_id)
        self._subscribed_ids.add(inst_id)

    def _on_start(self) -> None:
        self.subscribe_data(CHAINLINK_DATATYPE)

        self._msgbus.subscribe(UNIVERSE_TOPIC, self._on_polymarket_meta)
        self.subscribe_data(POLYMARKET_META_DATATYPE)
        self.subscribe_data(ORDER_FLOW_BUCKET_DEPTH10_DATATYPE)

        for inst_id in list(self._target_ids):
            self._request_instrument_if_needed(inst_id)
        
        print(
            f"[{self._log_label}] Started. Targets={len(self._target_ids)} "
            f"Subscribed={len(self._subscribed_ids)} Pending={len(self._pending_requests)} "
        )

    def min_qty_for_notional(self, px: float) -> int:
        if self.min_notional <= 0.0:
            return self.min_qty
        if px <= 0.0 or not np.isfinite(px):
            return self.min_qty
        return int(max(self.min_qty, math.ceil(self.min_notional / float(px))))

    def qty_for_notional(self, px: float, target_notional: Optional[float]) -> int:
        if target_notional is None:
            return 0
        if target_notional <= 0.0 or px <= 0.0 or not np.isfinite(px):
            return 0
        notional = min(float(target_notional), float(self._max_order_notional))
        qty = int(notional / float(px))
        qty = max(qty, self.min_qty_for_notional(px))
        return int(qty)

    def add_instruments(self, new_ids: List[InstrumentId]) -> None:
        actually_new = 0
        for inst_id in new_ids:
            if inst_id in self._target_ids:
                continue
            self._target_ids.add(inst_id)
            actually_new += 1
            self._request_instrument_if_needed(inst_id)

        if actually_new:
            print(
                f"[{self._log_label}] Added {actually_new} new instrument targets; "
                f"Targets={len(self._target_ids)} Pending={len(self._pending_requests)}"
            )

    def on_instrument(self, instrument: Instrument) -> None:
        inst_id = instrument.id

        if inst_id in self._pending_requests:
            self._pending_requests.remove(inst_id)

        if inst_id not in self._target_ids:
            return

        self._subscribe_instrument(inst_id)

    # ------------------------------------------------------------------
    # Meta ingestion from publisher (PolymarketMetaCustomData)
    # ------------------------------------------------------------------
    @staticmethod
    def _dt_from_ns(ts_ns: int) -> datetime:
        ts_s = int(ts_ns // 1_000_000_000)
        return datetime.fromtimestamp(ts_s, tz=timezone.utc).replace(microsecond=0)

    def _rebuild_slugs(self) -> None:
        slugs = set(self._yes_by_slug.keys()) | set(self._no_by_slug.keys())
        self._slugs = sorted(s for s in slugs if s in self._yes_by_slug and s in self._no_by_slug)
        for slug in self._slugs:
            self._active_by_slug.setdefault(slug, self._active_pair_factory())

    def normalize_polymarket_meta(self, meta: Any) -> dict[str, Any]:
        if isinstance(meta, Mapping):
            d: dict[str, Any] = dict(meta)
        elif hasattr(meta, "to_dict"):
            d = meta.to_dict()
        else:
            raise TypeError(f"Unsupported meta type: {type(meta)}")

        iid = d.get("instrument_id")
        if isinstance(iid, str):
            d["instrument_id"] = InstrumentId.from_str(iid)

        for k in ("start_date", "end_date", "ts_event", "ts_init"):
            v = d.get(k)
            if isinstance(v, int):
                d[k] = self._dt_from_ns(v)
            elif isinstance(v, datetime):
                if v.tzinfo is None:
                    v = v.replace(tzinfo=timezone.utc)
                d[k] = v.astimezone(timezone.utc).replace(microsecond=0)

        return d

    def _on_polymarket_meta(
        self,
        meta_or_metas: Union[PolymarketMetaCustomData, List[PolymarketMetaCustomData]],
    ) -> None:
        metas: Iterable[PolymarketMetaCustomData] = (
            meta_or_metas if isinstance(meta_or_metas, list) else [meta_or_metas]
        )

        for meta in metas:
            meta = self.normalize_polymarket_meta(meta)

            inst_id = (
                meta["instrument_id"]
                if isinstance(meta["instrument_id"], InstrumentId)
                else InstrumentId.from_str(str(meta["instrument_id"]))
            )
            slug = str(meta["slug"])
            outcome = str(meta["outcome"])

            start_dt = meta["start_date"]
            end_dt = meta["end_date"]
            ref = float(meta["reference_price"])
            question = str(
                meta.get("question")
                or (meta.get("market_info") or {}).get("question")
                or ""
            )
            prev_meta = self._meta_by_id.get(inst_id, {})
            if not question:
                question = str(prev_meta.get("question") or "")

            self._meta_by_id[inst_id] = {
                "slug": slug,
                "question": question,
                "outcome": outcome,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "reference_price": ref,
            }
            self._state_by_id[inst_id] = self._event_state_factory(
                start_dt=start_dt,
                end_dt=end_dt,
                ref=ref,
            )

            if outcome == "Up" or outcome == "YES":
                self._yes_by_slug[slug] = inst_id
            elif outcome == "Down" or outcome == "NO":
                self._no_by_slug[slug] = inst_id

            self._metas_by_start_second.setdefault(start_dt, []).append(inst_id)

            if inst_id not in self._target_ids:
                self._target_ids.add(inst_id)
                self._request_instrument_if_needed(inst_id)

        self._rebuild_slugs()

    # ------------------------------------------------------------------
    # Template methods for data handlers
    # ------------------------------------------------------------------
    def on_quote_tick(self, tick: QuoteTick) -> None:
        self._handle_quote_tick(tick)

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._handle_trade_tick(tick)

    def on_order_book_depth(self, book: OrderBookDepth10) -> None:
        self._process_depth10(book)

    def on_order_book_depth10(self, book: OrderBookDepth10) -> None:
        self._process_depth10(book)

    def on_data(self, data: Data) -> None:
        super().on_data(data)

        if isinstance(data, CustomData):
            inner = data.data
            if isinstance(inner, OrderFlowBucketDepth10CustomData):
                self._handle_order_flow_bucket(inner)
                return
            if isinstance(inner, ChainlinkCustomData):
                self._handle_chainlink_data(inner)
                return

        if isinstance(data, OrderFlowBucketDepth10CustomData):
            self._handle_order_flow_bucket(data)
            return

        if isinstance(data, ChainlinkCustomData):
            self._handle_chainlink_data(data)
            return

        if isinstance(data, PolymarketMetaCustomData):
            self._on_polymarket_meta(data)
            return

        self._handle_data(data)

    def _handle_quote_tick(self, _tick: QuoteTick) -> None:
        return

    def _handle_trade_tick(self, _tick: TradeTick) -> None:
        return

    def _handle_order_book_depth10(self, _book: OrderBookDepth10) -> None:
        return

    def _handle_chainlink_data(self, _data: ChainlinkCustomData) -> None:
        return

    def _handle_order_flow_bucket(self, _data: OrderFlowBucketDepth10CustomData) -> None:
        return

    def _handle_data(self, _data: Data) -> None:
        return

    def on_order_filled(self, event) -> None:
        try:
            inst_id = event.instrument_id
            if getattr(event, "last_qty", None) is not None:
                fill_qty = int(event.last_qty)
            elif getattr(event, "quantity", None) is not None:
                fill_qty = int(event.quantity)
            else:
                return

            if getattr(event, "last_px", None) is not None:
                fill_px = float(event.last_px)
            elif getattr(event, "price", None) is not None:
                fill_px = float(event.price)
            else:
                return

            if event.order_side != OrderSide.BUY:
                return

            self._update_inv_cost(inst_id, fill_qty, fill_px)
        except Exception:
            return

    def _process_depth10(self, book: OrderBookDepth10) -> None:
        self._handle_order_book_depth10(book)

    # ------------------------------------------------------------------
    # Exposure + inventory cost helpers
    # ------------------------------------------------------------------
    def net_exposure(self, inst_id: InstrumentId) -> Tuple[int, float]:
        pos = self.portfolio.net_position(inst_id)
        net_exposure = self.portfolio.net_exposure(inst_id)
        return int(pos), float(net_exposure)

    def open_exposure(self, inst_id: InstrumentId) -> Tuple[int, float]:
        pos_qty_int, pos_notional = self._net_exposure(inst_id)
        all_orders = self.cache.orders()
        order_qty_dec, order_notional = get_open_exposure(inst_id, all_orders)
        order_qty_int = int(order_qty_dec)
        net_qty = pos_qty_int + order_qty_int
        total_notional = float(order_notional) + pos_notional
        return net_qty, total_notional

    def _update_inv_cost(self, inst_id: InstrumentId, fill_qty: int, fill_px: float) -> None:
        old_qty, old_avg = self._inv_cost.get(inst_id, (0, 0.0))
        new_qty = old_qty + fill_qty
        if new_qty <= 0:
            self._inv_cost[inst_id] = (0, 0.0)
            return
        new_avg = (old_qty * old_avg + fill_qty * fill_px) / new_qty
        self._inv_cost[inst_id] = (new_qty, float(new_avg))

    def get_inv_cost(self, inst_id: InstrumentId) -> Tuple[int, float]:
        return self._inv_cost.get(inst_id, (0, float("nan")))

    def log_order_submit(self, order: object, note: str = "") -> None:
        # try:
        #     inst_id = getattr(order, "instrument_id", "")
        #     side = getattr(order, "order_side", "")
        #     qty = getattr(order, "quantity", None)
        #     price = getattr(order, "price", None)
        #     coid = getattr(order, "client_order_id", None)
        #     parts = [f"inst={inst_id}", f"side={side}"]
        #     if qty is not None:
        #         parts.append(f"qty={qty}")
        #     if price is not None:
        #         parts.append(f"px={price}")
        #     if coid is not None:
        #         parts.append(f"coid={coid}")
        #     if note:
        #         parts.append(note)
        #     print("[order] " + " ".join(str(p) for p in parts))
        # except Exception:
        #     return
        return
        

    # ------------------------------------------------------------------
    # Coverage / latency metrics (DataRecorder-style)
    # ------------------------------------------------------------------
    def _maybe_record_coverage_second(
        self,
        inst_id: InstrumentId,
        kind: str,
        event_ns: int,
    ) -> None:
        if kind == "cl":
            now_sec = int(time.time())
            if inst_id not in self._coverage_first_seen_sec:
                self._coverage_first_seen_sec[inst_id] = now_sec
            start_sec = self._coverage_first_seen_sec.get(inst_id)
            event_sec = now_sec
        else:
            start_sec = self._event_start_sec(inst_id)
            event_sec = int(event_ns // 1_000_000_000)
        if start_sec is None:
            return
        if event_sec < start_sec:
            return
        last_sec = self._coverage_last_second[inst_id][kind]
        if event_sec == last_sec:
            return
        self._coverage_last_second[inst_id][kind] = event_sec
        self._coverage_seconds[inst_id][kind] += 1

    def _event_start_sec(self, inst_id: InstrumentId) -> Optional[int]:
        meta = self._meta_by_id.get(inst_id, {})
        start_dt = meta.get("start_dt")
        if not isinstance(start_dt, datetime):
            return None
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        return int(start_dt.timestamp())

    def _coverage_percent(self, inst_id: InstrumentId, kind: str) -> Optional[float]:
        if kind == "cl":
            start_sec = self._coverage_first_seen_sec.get(inst_id)
        else:
            start_sec = self._event_start_sec(inst_id)
        if start_sec is None:
            return None
        now_sec = int(time.time())
        if now_sec < start_sec:
            return None
        total_secs = (now_sec - start_sec) + 1
        if total_secs <= 0:
            return None
        covered = self._coverage_seconds.get(inst_id, {}).get(kind, 0)
        return (covered / total_secs) * 100.0

    def _latency_ms(self, inst_id: InstrumentId, kind: str) -> float:
        total = self._latency_totals_ns.get(inst_id, {}).get(kind, 0)
        count = self._coverage_counts.get(inst_id, {}).get(kind, 0)
        if count == 0:
            return 0.0
        return total / count / 1_000_000.0

    def _ingest_latency_ms(self, inst_id: InstrumentId, kind: str) -> float:
        total = self._ingest_totals_ns.get(inst_id, {}).get(kind, 0)
        count = self._coverage_counts.get(inst_id, {}).get(kind, 0)
        if count == 0:
            return 0.0
        return total / count / 1_000_000.0

    def _instrument_label(self, inst_id: InstrumentId) -> str:
        meta = self._meta_by_id.get(inst_id)
        if isinstance(meta, dict):
            question = str(meta.get("question", "")).strip()
            if question:
                return question
            slug = str(meta.get("slug", "")).strip()
            outcome = str(meta.get("outcome", "")).strip()
            label = " ".join(x for x in (slug, outcome) if x)
            if label:
                return label
        inst_str = str(inst_id)
        if inst_str.endswith(".CHAINLINK"):
            return inst_str.replace(".CHAINLINK", "")
        return inst_str

    def _make_buy_order(self, inst_id: InstrumentId, px: float, qty: int):
        inst = self.cache.instrument(inst_id)
        if inst is None:
            return None
        qty = self._cap_qty_for_notional(px, qty)
        if qty <= 0:
            return None

        return self.order_factory.limit(
            instrument_id=inst_id,
            order_side=OrderSide.BUY,
            quantity=Quantity(Decimal(int(qty)), inst.size_precision),
            price=Price(Decimal(str(px)), inst.price_precision),
            time_in_force=TimeInForce.GTC,
        )

    def _make_sell_order(self, inst_id: InstrumentId, px: float, qty: int):
        if not ALLOW_SELL_ORDERS:
            return None
        inst = self.cache.instrument(inst_id)
        if inst is None:
            return None
        qty = self._cap_qty_for_notional(px, qty)
        if qty <= 0:
            return None

        return self.order_factory.limit(
            instrument_id=inst_id,
            order_side=OrderSide.SELL,
            quantity=Quantity(Decimal(int(qty)), inst.size_precision),
            price=Price(Decimal(str(px)), inst.price_precision),
            time_in_force=TimeInForce.GTC,
        )

    def _cap_qty_for_notional(self, px: float, qty: int) -> int:
        if qty <= 0 or px <= 0.0 or not np.isfinite(px):
            return 0
        max_qty = int(self._max_order_notional // float(px))
        if max_qty <= 0:
            return 0
        capped = min(int(qty), max_qty)
        if capped < self.min_qty:
            return 0
        return capped
