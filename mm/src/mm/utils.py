from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Tuple

from decimal import Decimal
from typing import Iterable, Tuple


def get_open_exposure(instrument_id, orders: Iterable) -> Tuple[Decimal, Decimal]:
    """
    Calculate open exposure for a given instrument_id from a collection of orders.

    Includes:
      - local active orders    (order.is_active_local)
      - in-flight orders       (order.is_inflight)
      - open venue orders      (order.is_open)

    Uses leaves_qty (remaining quantity).
    BUY  = +exposure
    SELL = -exposure

    Returns:
        (net_quantity, net_notional) as Decimals
    """
    total_qty = Decimal("0")
    total_notional = Decimal("0")

    for order in orders:
        # Only this instrument
        if order.instrument_id != instrument_id:
            continue

        # Only orders that still matter for exposure
        if not (order.is_active_local or order.is_inflight or order.is_open):
            continue

        leaves = order.leaves_qty  # Quantity

        # Quantity.zero() exists, but easiest is to check underlying raw int
        if leaves.raw == 0:
            continue

        # Quantity as Decimal (respects precision)
        qty_dec = leaves.as_decimal()

        # BUY adds, SELL subtracts
        sign = Decimal(1) if order.is_buy else Decimal(-1)
        total_qty += sign * qty_dec

        # Notional = qty * price (if price known)
        if order.has_price:
            px_dec = order.price.as_decimal()
            total_notional += sign * (qty_dec * px_dec)

    return total_qty, total_notional

