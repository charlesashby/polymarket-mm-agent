"""
Baseline no-op strategy for Polymarket market-making.

This strategy inherits from BaseStrategy and runs end-to-end without placing
any orders. It serves as a sanity check to ensure the evaluation pipeline works
and as a starting point for implementing actual market-making logic.

The strategy subscribes to all required data feeds but does not submit orders,
resulting in zero fills and zero P&L. This is expected behavior.
"""

from __future__ import annotations

from typing import Any

from nautilus_trader.model.data import QuoteTick, TradeTick, OrderBookDepth10

from mm.src.mm.strategies.base_strategy import BaseStrategy
from mm.src.mm.types import ChainlinkCustomData, OrderFlowBucketDepth10CustomData


class BaselineStrategy(BaseStrategy):
    """
    Baseline no-op strategy that subscribes to data but does not place orders.

    This strategy is used for:
    1. Verifying the evaluation pipeline works end-to-end
    2. Benchmarking the cost of data processing without trading logic
    3. Serving as a template for implementing actual strategies

    Expected results:
    - PNL: 0.0 (no fills)
    - Volume: 0.0 (no fills)
    - Fills: 0 (no orders submitted)
    """

    def __init__(
        self,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the baseline no-op strategy.

        Parameters
        ----------
        verbose : bool, default False
            If True, log data updates for debugging.
        **kwargs
            Additional keyword arguments passed to BaseStrategy.
        """
        # Simple factory functions for BaseStrategy requirements
        def active_pair_factory():
            return {
                "quotes_seen": 0,
                "trades_seen": 0,
                "depths_seen": 0,
            }

        def event_state_factory(**kw):
            return {
                "start_dt": kw.get("start_dt"),
                "end_dt": kw.get("end_dt"),
                "ref": kw.get("ref"),
                "data_count": 0,
            }

        super().__init__(
            active_pair_factory=active_pair_factory,
            event_state_factory=event_state_factory,
            log_label="BaselineStrategy",
            metrics_enabled=False,  # Disable metrics for cleaner output
            **kwargs
        )

        self.verbose = bool(verbose)
        self._total_quotes = 0
        self._total_trades = 0
        self._total_depths = 0
        self._total_flows = 0
        self._total_chainlink = 0

    def _on_start(self) -> None:
        """
        Called when the strategy starts.

        Subscribes to all data feeds but does not place any orders.
        """
        super()._on_start()

        if self.verbose:
            print(f"[{self._log_label}] Strategy started in no-op mode (no orders will be placed)")
            print(f"[{self._log_label}] Subscribed to {len(self._target_ids)} instruments")

    def _handle_quote_tick(self, tick: QuoteTick) -> None:
        """
        Handle quote tick updates (no-op).

        Parameters
        ----------
        tick : QuoteTick
            The quote tick to process.
        """
        self._total_quotes += 1

        if self.verbose and self._total_quotes % 1000 == 0:
            print(f"[{self._log_label}] Processed {self._total_quotes} quotes")

    def _handle_trade_tick(self, tick: TradeTick) -> None:
        """
        Handle trade tick updates (no-op).

        Parameters
        ----------
        tick : TradeTick
            The trade tick to process.
        """
        self._total_trades += 1

        if self.verbose and self._total_trades % 100 == 0:
            print(f"[{self._log_label}] Processed {self._total_trades} trades")

    def _handle_order_book_depth10(self, book: OrderBookDepth10) -> None:
        """
        Handle order book depth updates (no-op).

        Parameters
        ----------
        book : OrderBookDepth10
            The depth snapshot to process.
        """
        self._total_depths += 1

        if self.verbose and self._total_depths % 1000 == 0:
            print(f"[{self._log_label}] Processed {self._total_depths} depth snapshots")

    def _handle_order_flow_bucket(self, data: OrderFlowBucketDepth10CustomData) -> None:
        """
        Handle order flow bucket updates (no-op).

        Parameters
        ----------
        data : OrderFlowBucketDepth10CustomData
            The order flow bucket data to process.
        """
        self._total_flows += 1

        if self.verbose and self._total_flows % 100 == 0:
            print(f"[{self._log_label}] Processed {self._total_flows} flow buckets")

    def _handle_chainlink_data(self, data: ChainlinkCustomData) -> None:
        """
        Handle Chainlink oracle updates (no-op).

        Parameters
        ----------
        data : ChainlinkCustomData
            The Chainlink price data to process.
        """
        self._total_chainlink += 1

        if self.verbose and self._total_chainlink % 100 == 0:
            print(f"[{self._log_label}] Processed {self._total_chainlink} Chainlink ticks")

    def _on_stop(self) -> None:
        """
        Called when the strategy stops.

        Logs summary statistics about data processed.
        """
        print(f"[{self._log_label}] Strategy stopped")
        print(f"[{self._log_label}] Data processed:")
        print(f"[{self._log_label}]   Quotes: {self._total_quotes}")
        print(f"[{self._log_label}]   Trades: {self._total_trades}")
        print(f"[{self._log_label}]   Depths: {self._total_depths}")
        print(f"[{self._log_label}]   Flows:  {self._total_flows}")
        print(f"[{self._log_label}]   Oracle: {self._total_chainlink}")
        print(f"[{self._log_label}] No orders placed (as expected for baseline)")
