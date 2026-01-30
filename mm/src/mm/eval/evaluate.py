from __future__ import annotations

import argparse
import bisect
import importlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nautilus_trader.adapters.polymarket import (
    POLYMARKET_VENUE,
    get_polymarket_http_client,
    get_polymarket_instrument_provider,
)
from nautilus_trader.common.component import LiveClock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from nautilus_trader.model.instruments import BinaryOption, Instrument
from nautilus_trader.model.objects import Currency
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from mm.src.mm.backtest_engine.engine import build_engine
from mm.src.mm.strategies.base_strategy import BaseStrategy
from mm.src.mm.types import ChainlinkCustomData, OrderFlowBucketDepth10CustomData, PolymarketMetaCustomData

DEFAULT_CATALOG_ROOT = Path("/home/ashbyc/data/polymarket/1.0.5/nautilus-catalog-prod")
CLIENT_ID_DATA = ClientId("PARQUET-CATALOG")
FORCED_PRICE_PRECISION = 2
HARDCODED_SLUGS = [
    'sol-updown-15m-1768770900', 
    'sol-updown-15m-1768779900', 
    'sol-updown-15m-1768780800', 
    'sol-updown-15m-1768781700', 
    'sol-updown-15m-1768782600', 
    'sol-updown-15m-1768783500', 
    'sol-updown-15m-1768784400', 
    'sol-updown-15m-1768785300', 
    'sol-updown-15m-1768786200'
]

YES_OUTCOMES = {"YES", "UP", "TRUE"}
NO_OUTCOMES = {"NO", "DOWN", "FALSE"}


@dataclass(frozen=True)
class MetaRow:
    inst_id: str
    slug: str
    start_ns: int
    end_ns: int
    outcome: str


def _extract_meta_row(cd: PolymarketMetaCustomData) -> Optional[MetaRow]:
    data = getattr(cd, "data", cd)
    try:
        return MetaRow(
            inst_id=str(data.instrument_id),
            slug=str(data.slug),
            start_ns=int(data.start_date),
            end_ns=int(data.end_date),
            outcome=str(data.outcome),
        )
    except Exception:
        return None


def build_pairs(
    meta_all: Sequence[PolymarketMetaCustomData],
    *,
    slug_whitelist: Sequence[str],
) -> Tuple[Dict[str, Tuple[InstrumentId, InstrumentId]], Dict[str, PolymarketMetaCustomData]]:
    yes: Dict[str, InstrumentId] = {}
    no: Dict[str, InstrumentId] = {}
    by_inst: Dict[str, PolymarketMetaCustomData] = {}
    whitelist = {s.strip() for s in slug_whitelist if s.strip()}

    for cd in meta_all:
        row = _extract_meta_row(cd)
        if row is None:
            continue
        if whitelist and row.slug not in whitelist:
            continue
        inst_id = InstrumentId.from_str(row.inst_id)
        by_inst[row.inst_id] = cd
        outcome = row.outcome.upper()
        if outcome in YES_OUTCOMES:
            yes[row.slug] = inst_id
        elif outcome in NO_OUTCOMES:
            no[row.slug] = inst_id

    pairs = {slug: (yes[slug], no[slug]) for slug in yes if slug in no}
    return pairs, by_inst


def normalize_binary_option_precision(inst: Instrument) -> Instrument:
    if not isinstance(inst, BinaryOption):
        return inst
    data = BinaryOption.to_dict(inst)
    data["price_precision"] = FORCED_PRICE_PRECISION
    data["price_increment"] = str(10 ** -FORCED_PRICE_PRECISION)
    data["maker_fee"] = str(Decimal("0"))
    data["taker_fee"] = str(Decimal("0.03"))
    return BinaryOption.from_dict(data)


def load_instruments(engine, inst_ids: Sequence[InstrumentId]) -> None:
    provider = get_polymarket_instrument_provider(
        client=get_polymarket_http_client(),
        config=InstrumentProviderConfig(load_ids=frozenset(inst_ids)),
        clock=LiveClock(),
    )

    delay = 10.0
    max_delay = 120.0
    attempts = 0
    while True:
        attempts += 1
        try:
            provider.load_ids(inst_ids)
            break
        except Exception as exc:
            if attempts >= 5:
                raise
            wait_s = min(delay, max_delay)
            print(f"[load_instruments] attempt {attempts} failed ({exc}); retrying in {wait_s:.1f}s")
            time.sleep(wait_s)
            delay *= 2

    for inst in provider._instruments.values():
        engine.add_instrument(normalize_binary_option_precision(inst))


def _precision_is_ok(x: object) -> bool:
    if hasattr(x, "bid_price") and getattr(x, "bid_price", None) is not None:
        bp = getattr(x, "bid_price")
        ap = getattr(x, "ask_price", None)
        if bp.precision != FORCED_PRICE_PRECISION:
            return False
        if ap is not None and ap.precision != FORCED_PRICE_PRECISION:
            return False
        return True
    if hasattr(x, "price") and getattr(x, "price", None) is not None:
        return getattr(x, "price").precision == FORCED_PRICE_PRECISION
    return True


def _ts_event_of(data: object) -> int:
    inner = getattr(data, "data", data)
    ts = getattr(inner, "ts_event", None)
    if ts is None:
        ts = getattr(inner, "ts_init", None)
    if ts is None:
        return 0
    return int(ts)


def _safe_trade_volume(fills_df: pd.DataFrame) -> float:
    if fills_df.empty:
        return 0.0
    qty_col = "filled_qty" if "filled_qty" in fills_df.columns else "quantity"
    price_col = "avg_px" if "avg_px" in fills_df.columns else "price"
    if qty_col not in fills_df.columns or price_col not in fills_df.columns:
        return 0.0
    qty = pd.to_numeric(fills_df[qty_col], errors="coerce").abs()
    price = pd.to_numeric(fills_df[price_col], errors="coerce")
    return float((qty * price).sum(skipna=True))


def _mark_price_for_instrument(inst_id: str, meta_by_inst: Mapping[str, PolymarketMetaCustomData]) -> Optional[float]:
    # Deprecated: marks now computed via chainlink vs reference price.
    _ = meta_by_inst
    return None


def _compute_pnl_from_fills(
    fills_df: pd.DataFrame,
    mark_by_inst: Mapping[str, float],
) -> float:
    if fills_df.empty:
        return 0.0
    qty_col = "filled_qty" if "filled_qty" in fills_df.columns else "quantity"
    price_col = "avg_px" if "avg_px" in fills_df.columns else "price"
    if qty_col not in fills_df.columns or price_col not in fills_df.columns:
        return 0.0

    qty = pd.to_numeric(fills_df[qty_col], errors="coerce")
    price = pd.to_numeric(fills_df[price_col], errors="coerce")
    side = (
        fills_df["side"].astype(str).str.upper().str.strip()
        if "side" in fills_df.columns
        else pd.Series([""] * len(fills_df), dtype=object)
    )

    inst_series = (
        fills_df["instrument_id"].astype(str)
        if "instrument_id" in fills_df.columns
        else pd.Series([""] * len(fills_df), dtype=object)
    )

    sign = pd.Series(np.nan, index=fills_df.index)
    sign[side == "BUY"] = 1.0
    sign[side == "SELL"] = -1.0

    mark_prices = inst_series.map(lambda inst: mark_by_inst.get(inst))
    mask = (
        qty.notna()
        & price.notna()
        & sign.notna()
        & mark_prices.notna()
        & (mark_prices != float("nan"))
    )
    if not mask.any():
        return 0.0

    pnl = (sign[mask] * qty[mask] * (mark_prices[mask] - price[mask])).sum()
    return float(pnl)


def _winning_fill_percentage(
    fills_df: pd.DataFrame,
    mark_by_inst: Mapping[str, float],
) -> float:
    if fills_df.empty:
        return 0.0
    qty_col = "filled_qty" if "filled_qty" in fills_df.columns else "quantity"
    price_col = "avg_px" if "avg_px" in fills_df.columns else "price"
    if qty_col not in fills_df.columns or price_col not in fills_df.columns:
        return 0.0

    qty = pd.to_numeric(fills_df[qty_col], errors="coerce")
    price = pd.to_numeric(fills_df[price_col], errors="coerce")
    side = (
        fills_df["side"].astype(str).str.upper().str.strip()
        if "side" in fills_df.columns
        else pd.Series([""] * len(fills_df), dtype=object)
    )

    inst_series = (
        fills_df["instrument_id"].astype(str)
        if "instrument_id" in fills_df.columns
        else pd.Series([""] * len(fills_df), dtype=object)
    )

    sign = pd.Series(np.nan, index=fills_df.index)
    sign[side == "BUY"] = 1.0
    sign[side == "SELL"] = -1.0

    mark_prices = inst_series.map(lambda inst: mark_by_inst.get(inst))
    mask = qty.notna() & price.notna() & sign.notna() & mark_prices.notna()
    if not mask.any():
        return 0.0

    pnl = sign[mask] * qty[mask] * (mark_prices[mask] - price[mask])
    wins = int((pnl > 0).sum())
    total = int(len(pnl))
    if total <= 0:
        return 0.0
    return 100.0 * (wins / total)


def _chainlink_instrument_for_slug(slug: str) -> Optional[str]:
    slug = slug.strip().lower()
    if slug.startswith("sol-"):
        return "SOLUSD.CHAINLINK"
    if slug.startswith("btc-"):
        return "BTCUSD.CHAINLINK"
    return None


def _last_price_before(ts_list: list[int], px_list: list[float], ts_ns: int) -> Optional[float]:
    if not ts_list:
        return None
    idx = bisect.bisect_right(ts_list, ts_ns) - 1
    if idx < 0:
        return None
    return float(px_list[idx])


def _compute_mark_prices(
    meta_by_inst: Mapping[str, PolymarketMetaCustomData],
    chainlink_by_inst: Mapping[str, tuple[list[int], list[float]]],
) -> Dict[str, float]:
    """
    Resolve marks using Chainlink price at end_dt vs reference_price at start.
    YES/UP legs: 1 if P_T > P_0 else 0. NO/DOWN legs: complement.
    """
    marks: Dict[str, float] = {}
    for inst_id, meta in meta_by_inst.items():
        data = getattr(meta, "data", meta)
        slug = str(getattr(data, "slug", "")).strip()
        if not slug:
            continue
        chain_inst = _chainlink_instrument_for_slug(slug)
        if chain_inst is None:
            continue
        series = chainlink_by_inst.get(chain_inst)
        if not series:
            continue
        ts_list, px_list = series

        start_ns = int(getattr(data, "start_date", 0) or 0)
        end_ns = int(getattr(data, "end_date", 0) or 0)
        if end_ns <= 0:
            continue

        ref_px = float(getattr(data, "reference_price", float("nan")))
        ref_from_chain = _last_price_before(ts_list, px_list, start_ns)
        if ref_from_chain is not None and np.isfinite(ref_from_chain):
            ref_px = float(ref_from_chain)
        if not np.isfinite(ref_px):
            continue

        end_px = _last_price_before(ts_list, px_list, end_ns)
        if end_px is None or not np.isfinite(end_px):
            continue

        resolved_yes = 1.0 if float(end_px) > float(ref_px) else 0.0
        outcome = str(getattr(data, "outcome", "")).upper()
        if outcome in YES_OUTCOMES:
            mark = resolved_yes
        elif outcome in NO_OUTCOMES:
            mark = 1.0 - resolved_yes
        else:
            mark = resolved_yes
        marks[str(inst_id)] = float(mark)

    return marks


def _import_strategy(strategy_path: str, strategy_kwargs: Mapping[str, Any]) -> BaseStrategy:
    module_name, _, class_name = strategy_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Strategy path must include module: {strategy_path}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseStrategy):
        raise ValueError(f"{strategy_path} does not inherit from BaseStrategy")
    return cls(**strategy_kwargs)


def evaluate_slugs(
    *,
    catalog_root: Path,
    slugs: Sequence[str],
    strategy_path: str,
    strategy_kwargs: Mapping[str, Any],
    balance: Decimal,
    base_ccy: Currency,
    run_tag: str,
) -> Dict[str, float]:
    if not slugs:
        raise ValueError("At least one slug must be provided.")

    catalog = ParquetDataCatalog(catalog_root)
    meta_all = list(catalog.custom_data(PolymarketMetaCustomData))
    pairs, meta_by_inst = build_pairs(meta_all, slug_whitelist=slugs)
    missing = [slug for slug in slugs if slug not in pairs]
    if missing:
        raise ValueError(f"No instruments found for slugs: {missing}")

    inst_ids: List[InstrumentId] = []
    for slug in slugs:
        yes_id, no_id = pairs[slug]
        inst_ids.extend([yes_id, no_id])

    instrument_pairs: Dict[str, str] = {}
    for yes_id, no_id in pairs.values():
        instrument_pairs[str(yes_id)] = str(no_id)
        instrument_pairs[str(no_id)] = str(yes_id)

    engine = build_engine(
        run_tag=run_tag,
        balance=balance,
        base_ccy=base_ccy,
        instrument_pairs=instrument_pairs,
    )

    load_instruments(engine, inst_ids)

    start_ns = []
    end_ns = []
    metas = []
    for inst_id in inst_ids:
        meta = meta_by_inst.get(str(inst_id))
        if meta is None:
            continue
        metas.append(meta)
        start_ns.append(int(getattr(getattr(meta, "data", meta), "start_date", 0)))
        end_ns.append(int(getattr(getattr(meta, "data", meta), "end_date", 0)))

    if not start_ns or not end_ns:
        raise ValueError("Missing metadata timestamps for requested slugs.")
    w0, w1 = min(start_ns), max(end_ns)
    if w1 <= w0:
        raise ValueError(f"Invalid evaluation window: start={w0} end={w1}")

    data = list(metas)
    inst_set = [str(iid) for iid in inst_ids]

    for qt in catalog.quote_ticks(inst_ids, start=w0, end=w1 + 1):
        if _precision_is_ok(qt):
            data.append(qt)
    for tt in catalog.trade_ticks(inst_ids, start=w0, end=w1 + 1):
        if _precision_is_ok(tt):
            data.append(tt)
    for depth in catalog.order_book_depth10(inst_ids, start=w0, end=w1 + 1):
        if _precision_is_ok(depth):
            data.append(depth)

    for of in catalog.query(
        data_cls=OrderFlowBucketDepth10CustomData,
        identifiers=inst_set,
        start=w0,
        end=w1 + 1,
    ):
        data.append(of)

    chain_inst_ids = sorted(
        {cid for cid in (_chainlink_instrument_for_slug(slug) for slug in slugs) if cid}
    )
    chainlink_by_inst: Dict[str, tuple[list[int], list[float]]] = {}
    if chain_inst_ids:
        for cd in catalog.query(
            data_cls=ChainlinkCustomData,
            identifiers=chain_inst_ids,
            start=w0,
            end=w1 + 1,
        ):
            payload = getattr(cd, "data", cd)
            inst = str(getattr(payload, "instrument_id", ""))
            if not inst:
                continue
            ts = int(getattr(payload, "ts_event", getattr(payload, "ts_init", 0)) or 0)
            px = getattr(payload, "price", None)
            if px is None:
                continue
            try:
                px_f = float(px)
            except Exception:
                continue
            chainlink_by_inst.setdefault(inst, ([], []))
            chainlink_by_inst[inst][0].append(ts)
            chainlink_by_inst[inst][1].append(px_f)
            data.append(cd)

        for inst, (ts_list, px_list) in chainlink_by_inst.items():
            if not ts_list:
                continue
            pairs = sorted(zip(ts_list, px_list), key=lambda x: x[0])
            chainlink_by_inst[inst] = ([p[0] for p in pairs], [p[1] for p in pairs])

    data.sort(key=_ts_event_of)

    engine.add_data(data, client_id=CLIENT_ID_DATA)

    strategy = _import_strategy(strategy_path, strategy_kwargs)
    engine.add_strategy(strategy)
    run_exc: Optional[Exception] = None
    try:
        engine.run()
    except Exception as exc:
        run_exc = exc

    fills_report = engine.trader.generate_order_fills_report()
    fills_df = pd.DataFrame(fills_report)
    mark_by_inst = _compute_mark_prices(meta_by_inst, chainlink_by_inst)
    pnl = _compute_pnl_from_fills(fills_df, mark_by_inst)
    win_pct = _winning_fill_percentage(fills_df, mark_by_inst)
    volume = _safe_trade_volume(fills_df)
    fills_count = float(len(fills_df) if not fills_df.empty else 0)
    avg_notional = (volume / fills_count) if fills_count > 0 else 0.0

    try:
        engine.reset()
        engine.clear_data()
        engine.clear_exec_algorithms()
        engine.clear_actors()
    except Exception:
        pass

    if run_exc is not None:
        raise run_exc

    return {
        "pnl": pnl,
        "volume": volume,
        "fills": fills_count,
        "win_pct": win_pct,
        "avg_notional": avg_notional,
    }


def parse_strategy_kwargs(data: Optional[str]) -> Dict[str, Any]:
    if not data:
        return {}
    return json.loads(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Barebone Polymarket backtest evaluator.")
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Full import path to a strategy class that inherits from mm.src.mm.strategies.base_strategy.BaseStrategy (e.g. module.Class).",
    )
    parser.add_argument(
        "--strategy-kwargs",
        type=str,
        default=None,
        help="JSON object containing keyword arguments for the strategy constructor.",
    )
    parser.add_argument("--balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--base-ccy", type=str, default="USDC.e")
    parser.add_argument("--run-tag", type=str, default="mm-eval")

    args = parser.parse_args()

    strategy_kwargs = parse_strategy_kwargs(args.strategy_kwargs)
    base_ccy = Currency.from_str(args.base_ccy)

    result = evaluate_slugs(
        catalog_root=args.catalog_root,
        slugs=HARDCODED_SLUGS,
        strategy_path=args.strategy,
        strategy_kwargs=strategy_kwargs,
        balance=args.balance,
        base_ccy=base_ccy,
        run_tag=args.run_tag,
    )

    print(
        "PNL={pnl:.4f} volume={vol:.4f} fills={fills:.0f} win%={win:.2f} avg_notional={avg:.4f}".format(
            pnl=result["pnl"],
            vol=result["volume"],
            fills=result["fills"],
            win=result["win_pct"],
            avg=result["avg_notional"],
        )
    )


if __name__ == "__main__":
    main()
