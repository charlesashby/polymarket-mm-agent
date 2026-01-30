from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Dict, MutableMapping, Optional

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import CacheConfig, LoggingConfig
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Currency, Money

from mm.src.mm.backtest_engine.fill_model import (
    Depth10CacheModule,
    OrderFlowBucketCacheModule,
    PolymarketBinaryFillModel,
    QuoteBboCacheModule,
    TradeTickCacheModule,
)
from mm.src.mm.types import CHAINLINK_VENUE


def build_engine(
    *,
    run_tag: str,
    balance: Decimal,
    base_ccy: Currency,
    instrument_pairs: Optional[Mapping[str, str]] = None,
) -> BacktestEngine:
    """
    Create a Nautilus backtest engine wired with the polymarket binary fill model.

    The cross fill model and associated caching modules are only registered for the
    Polymarket venue; Chainlink remains as a plain venue for price data.
    """
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId(f"MM-BACKTEST-{run_tag}"),
            logging=LoggingConfig(log_level="ERROR"),
            cache=CacheConfig(),
        )
    )

    fill_model = None
    modules = None
    if instrument_pairs:
        bbo_cache: Dict[str, tuple[float, float]] = {}
        depth_cache: Dict[str, "OrderBookDepth10"] = {}
        flow_cache: Dict[str, object] = {}
        trade_cache: Dict[str, tuple[int, float, Optional[int]]] = {}
        fill_model = PolymarketBinaryFillModel(
            instrument_pairs,
            bbo_cache,
            depth_cache,
            flow_cache,
            trade_cache,
        )
        modules = [
            QuoteBboCacheModule(bbo_cache),
            Depth10CacheModule(depth_cache),
            OrderFlowBucketCacheModule(flow_cache),
            TradeTickCacheModule(trade_cache),
        ]

    for venue in (POLYMARKET_VENUE, CHAINLINK_VENUE):
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            starting_balances=[Money(balance, base_ccy)],
            base_currency=base_ccy,
            fill_model=fill_model if venue == POLYMARKET_VENUE else None,
            modules=modules if venue == POLYMARKET_VENUE else None,
        )

    return engine
