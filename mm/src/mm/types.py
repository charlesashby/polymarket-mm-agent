from __future__ import annotations

import msgspec
import pyarrow as pa

from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

from nautilus_trader.core import Data
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model import DataType
from nautilus_trader.core.datetime import unix_nanos_to_iso8601
from nautilus_trader.model.custom import customdataclass

from nautilus_trader.serialization.arrow.serializer import register_arrow

class PolymarketInstrumentMetas(TypedDict):
    instrument_id: InstrumentId
    slug: str
    start_date: datetime
    end_date: datetime
    reference_price: float
    outcome: str                 # "Up"/"Down" etc (same as PolymarketInstrumentMeta)
    token_id: str
    market_info: dict[str, Any]
    bar_type: NotRequired[object]  # keep optional, avoids circular BarType import


@customdataclass
class PolymarketMetaCustomData(Data):
    """
    Custom data representing polymarket meta for a single leg.
    Stored in catalog (Feather/Parquet) and sent over msgbus.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        slug: str,
        reference_price: float,
        start_date: int,
        end_date: int,
        outcome: str,
        ts_event: int,
        ts_init: int,
    ) -> None:
        self.instrument_id = instrument_id
        self.slug = slug
        self.reference_price = reference_price
        self.start_date = start_date  # unix ns
        self.end_date = end_date      # unix ns
        self.outcome = outcome

        self._ts_event = ts_event
        self._ts_init = ts_init

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    def __repr__(self) -> str:
        return (
            f"PolymarketMetaCustomData("
            f"ts_init={unix_nanos_to_iso8601(self._ts_init)}, "
            f"instrument_id={self.instrument_id}, "
            f"slug={self.slug!r}, "
            f"ref={self.reference_price:.4f}, "
            f"outcome={self.outcome}, "
            f"start={self.start_date}, end={self.end_date}"
            ")"
        )

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id.value,
            "slug": self.slug,
            "reference_price": float(self.reference_price),
            "start_date": int(self.start_date),
            "end_date": int(self.end_date),
            "outcome": self.outcome,
            "ts_event": int(self._ts_event),
            "ts_init": int(self._ts_init),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolymarketMetaCustomData":
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            slug=data["slug"],
            reference_price=float(data["reference_price"]),
            start_date=int(data["start_date"]),
            end_date=int(data["end_date"]),
            outcome=data["outcome"],
            ts_event=int(data["ts_event"]),
            ts_init=int(data["ts_init"]),
        )

    def to_bytes(self) -> bytes:
        return msgspec.msgpack.encode(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "PolymarketMetaCustomData":
        return cls.from_dict(msgspec.msgpack.decode(data))

    @classmethod
    def schema(cls) -> pa.Schema:
        return pa.schema(
            {
                "instrument_id": pa.string(),
                "slug": pa.string(),
                "reference_price": pa.float64(),
                "start_date": pa.int64(),
                "end_date": pa.int64(),
                "outcome": pa.string(),
                "ts_event": pa.int64(),
                "ts_init": pa.int64(),
            }
        )

    def to_catalog(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pylist([self.to_dict()], schema=self.schema())

    @classmethod
    def from_catalog(cls, table: pa.Table) -> list["PolymarketMetaCustomData"]:
        return [cls.from_dict(d) for d in table.to_pylist()]


register_arrow(
    PolymarketMetaCustomData,
    PolymarketMetaCustomData.schema(),
    PolymarketMetaCustomData.to_catalog,
    PolymarketMetaCustomData.from_catalog,
)

# DataType to use in publish_data / subscribe_data
POLYMARKET_META_DATATYPE = DataType(PolymarketMetaCustomData)

@customdataclass
class ChainlinkCustomData(Data):
    """
    Custom data representing a Chainlink BTC/USD oracle tick.

    Not an instrument price for trading, just an oracle value
    (but we still attach an InstrumentId for convenience).
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        symbol: str,
        price: float,
        ts_event: int,
        ts_init: int,
    ) -> None:
        self.instrument_id = instrument_id
        self.symbol = symbol  # e.g. "btc/usd"
        self.price = float(price)

        self._ts_event = int(ts_event)
        self._ts_init = int(ts_init)

    # --- required by Data ---

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    def __repr__(self) -> str:
        return (
            "ChainlinkCustomData("
            f"ts_init={unix_nanos_to_iso8601(self._ts_init)}, "
            f"instrument_id={self.instrument_id}, "
            f"symbol={self.symbol!r}, "
            f"price={self.price:.4f}"
            ")"
        )

    # --- serialization helpers for msgbus + catalog ---

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id.value,
            "symbol": self.symbol,
            "price": float(self.price),
            "ts_event": int(self._ts_event),
            "ts_init": int(self._ts_init),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChainlinkCustomData":
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            symbol=data["symbol"],
            price=float(data["price"]),
            ts_event=int(data["ts_event"]),
            ts_init=int(data["ts_init"]),
        )

    def to_bytes(self) -> bytes:
        return msgspec.msgpack.encode(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChainlinkCustomData":
        return cls.from_dict(msgspec.msgpack.decode(data))

    # --- Arrow / Parquet integration ---

    @classmethod
    def schema(cls) -> pa.Schema:
        return pa.schema(
            {
                "instrument_id": pa.string(),
                "symbol": pa.string(),
                "price": pa.float64(),
                "ts_event": pa.int64(),
                "ts_init": pa.int64(),
            }
        )

    def to_catalog(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pylist([self.to_dict()], schema=self.schema())

    @classmethod
    def from_catalog(cls, table: pa.Table) -> list["ChainlinkCustomData"]:
        return [cls.from_dict(d) for d in table.to_pylist()]


register_arrow(
    ChainlinkCustomData,
    ChainlinkCustomData.schema(),
    ChainlinkCustomData.to_catalog,
    ChainlinkCustomData.from_catalog,
)

CHAINLINK_DATATYPE = DataType(ChainlinkCustomData)
    
print("PolymarketMetaCustomData Arrow serializer registered")


@customdataclass
class OrderFlowBucketDepth10CustomData(Data):
    """
    Bucketed per-level order flow aggregated within depth-10.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        bucket_ts: int,
        bucket_ms: int,
        bid_prices: list[float],
        ask_prices: list[float],
        bid_add_count: list[int],
        bid_add_qty: list[float],
        bid_cancel_count: list[int],
        bid_cancel_qty: list[float],
        bid_trade_count: list[int],
        bid_trade_qty: list[float],
        ask_add_count: list[int],
        ask_add_qty: list[float],
        ask_cancel_count: list[int],
        ask_cancel_qty: list[float],
        ask_trade_count: list[int],
        ask_trade_qty: list[float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        self.instrument_id = instrument_id
        self.bucket_ts = int(bucket_ts)
        self.bucket_ms = int(bucket_ms)
        self.bid_prices = [float(p) for p in bid_prices]
        self.ask_prices = [float(p) for p in ask_prices]
        self.bid_add_count = [int(v) for v in bid_add_count]
        self.bid_add_qty = [float(v) for v in bid_add_qty]
        self.bid_cancel_count = [int(v) for v in bid_cancel_count]
        self.bid_cancel_qty = [float(v) for v in bid_cancel_qty]
        self.bid_trade_count = [int(v) for v in bid_trade_count]
        self.bid_trade_qty = [float(v) for v in bid_trade_qty]
        self.ask_add_count = [int(v) for v in ask_add_count]
        self.ask_add_qty = [float(v) for v in ask_add_qty]
        self.ask_cancel_count = [int(v) for v in ask_cancel_count]
        self.ask_cancel_qty = [float(v) for v in ask_cancel_qty]
        self.ask_trade_count = [int(v) for v in ask_trade_count]
        self.ask_trade_qty = [float(v) for v in ask_trade_qty]

        self._ts_event = int(ts_event)
        self._ts_init = int(ts_init)

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    def __repr__(self) -> str:
        return (
            "OrderFlowBucketDepth10CustomData("
            f"ts_init={unix_nanos_to_iso8601(self._ts_init)}, "
            f"instrument_id={self.instrument_id}, "
            f"bucket_ts={self.bucket_ts}, "
            f"bucket_ms={self.bucket_ms}, "
            f"bid_levels={len(self.bid_prices)}, "
            f"ask_levels={len(self.ask_prices)}"
            ")"
        )

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id.value,
            "bucket_ts": int(self.bucket_ts),
            "bucket_ms": int(self.bucket_ms),
            "bid_prices": [float(p) for p in self.bid_prices],
            "ask_prices": [float(p) for p in self.ask_prices],
            "bid_add_count": [int(v) for v in self.bid_add_count],
            "bid_add_qty": [float(v) for v in self.bid_add_qty],
            "bid_cancel_count": [int(v) for v in self.bid_cancel_count],
            "bid_cancel_qty": [float(v) for v in self.bid_cancel_qty],
            "bid_trade_count": [int(v) for v in self.bid_trade_count],
            "bid_trade_qty": [float(v) for v in self.bid_trade_qty],
            "ask_add_count": [int(v) for v in self.ask_add_count],
            "ask_add_qty": [float(v) for v in self.ask_add_qty],
            "ask_cancel_count": [int(v) for v in self.ask_cancel_count],
            "ask_cancel_qty": [float(v) for v in self.ask_cancel_qty],
            "ask_trade_count": [int(v) for v in self.ask_trade_count],
            "ask_trade_qty": [float(v) for v in self.ask_trade_qty],
            "ts_event": int(self._ts_event),
            "ts_init": int(self._ts_init),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrderFlowBucketDepth10CustomData":
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            bucket_ts=int(data["bucket_ts"]),
            bucket_ms=int(data["bucket_ms"]),
            bid_prices=list(data["bid_prices"]),
            ask_prices=list(data["ask_prices"]),
            bid_add_count=list(data["bid_add_count"]),
            bid_add_qty=list(data["bid_add_qty"]),
            bid_cancel_count=list(data["bid_cancel_count"]),
            bid_cancel_qty=list(data["bid_cancel_qty"]),
            bid_trade_count=list(data["bid_trade_count"]),
            bid_trade_qty=list(data["bid_trade_qty"]),
            ask_add_count=list(data["ask_add_count"]),
            ask_add_qty=list(data["ask_add_qty"]),
            ask_cancel_count=list(data["ask_cancel_count"]),
            ask_cancel_qty=list(data["ask_cancel_qty"]),
            ask_trade_count=list(data["ask_trade_count"]),
            ask_trade_qty=list(data["ask_trade_qty"]),
            ts_event=int(data["ts_event"]),
            ts_init=int(data["ts_init"]),
        )

    def to_bytes(self) -> bytes:
        return msgspec.msgpack.encode(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "OrderFlowBucketDepth10CustomData":
        return cls.from_dict(msgspec.msgpack.decode(data))

    @classmethod
    def schema(cls) -> pa.Schema:
        return pa.schema(
            {
                "instrument_id": pa.string(),
                "bucket_ts": pa.int64(),
                "bucket_ms": pa.int64(),
                "bid_prices": pa.list_(pa.float64()),
                "ask_prices": pa.list_(pa.float64()),
                "bid_add_count": pa.list_(pa.int64()),
                "bid_add_qty": pa.list_(pa.float64()),
                "bid_cancel_count": pa.list_(pa.int64()),
                "bid_cancel_qty": pa.list_(pa.float64()),
                "bid_trade_count": pa.list_(pa.int64()),
                "bid_trade_qty": pa.list_(pa.float64()),
                "ask_add_count": pa.list_(pa.int64()),
                "ask_add_qty": pa.list_(pa.float64()),
                "ask_cancel_count": pa.list_(pa.int64()),
                "ask_cancel_qty": pa.list_(pa.float64()),
                "ask_trade_count": pa.list_(pa.int64()),
                "ask_trade_qty": pa.list_(pa.float64()),
                "ts_event": pa.int64(),
                "ts_init": pa.int64(),
            }
        )

    def to_catalog(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pylist([self.to_dict()], schema=self.schema())

    @classmethod
    def from_catalog(cls, table: pa.Table) -> list["OrderFlowBucketDepth10CustomData"]:
        return [cls.from_dict(d) for d in table.to_pylist()]


register_arrow(
    OrderFlowBucketDepth10CustomData,
    OrderFlowBucketDepth10CustomData.schema(),
    OrderFlowBucketDepth10CustomData.to_catalog,
    OrderFlowBucketDepth10CustomData.from_catalog,
)

ORDER_FLOW_BUCKET_DEPTH10_DATATYPE = DataType(OrderFlowBucketDepth10CustomData)
UNIVERSE_TOPIC = "polymarket_universe_updates"


CHAINLINK_VENUE = Venue("CHAINLINK")
