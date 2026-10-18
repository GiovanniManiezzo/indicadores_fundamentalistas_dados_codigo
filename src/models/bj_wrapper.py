"""BJ (Box-Jenkins): y = B(L)/F(L)·u + C(L)/D(L)·e

Aproximação: SARIMAX(0, d, q) + exógenas — AR do endog modelado via MA de ruído.
p=0 forçado; grid AIC sobre q ∈ [1, q_max].
"""
from __future__ import annotations
from ._sarimax_base import _SARIMAXBase


class BJWrapper(_SARIMAXBase):
    """BJ: p=0 forçado; grid AIC sobre q ∈ [1, q_max]."""

    def __init__(
        self,
        d: int = 1,
        q_max: int = 3,
        mode: str = "raw",
        fixed_order: tuple | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(
            d=d,
            p_max=0,
            q_max=q_max,
            mode=mode,
            fixed_order=fixed_order,
            p_forced=0,    # BJ: sem AR no endog
            q_forced=None, # q livre no grid
            boxcox_lambda=boxcox_lambda,
        )

    def get_params_str(self) -> str:
        p, d, q = self._order
        return f"BJ({p},{d},{q})"
