"""OE (Output Error): y = B(L)/F(L) · u + e  (ruído branco — sem componente MA).

Aproximação: SARIMAX(p, d, 0) + exógenas.
q=0 forçado → resíduo branco (não colorido).
"""
from __future__ import annotations
from ._sarimax_base import _SARIMAXBase


class OEWrapper(_SARIMAXBase):
    """OE: q=0 forçado; grid AIC sobre p ∈ [1, p_max]."""

    def __init__(
        self,
        d: int = 0,
        p_max: int = 3,
        mode: str = "raw",
        fixed_order: tuple | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(
            d=d,
            p_max=p_max,
            q_max=0,
            mode=mode,
            fixed_order=fixed_order,
            q_forced=0,     # OE: sem MA
            p_forced=None,  # p livre no grid
            boxcox_lambda=boxcox_lambda,
        )

    def get_params_str(self) -> str:
        p, d, q = self._order
        return f"OE({p},{d},{q})"
