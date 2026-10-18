"""ARMAX: ARIMA(p, 0, q) com exógenas (sem integração da saída — d=0).

A(L) y_t = B(L) u_t + C(L) e_t
"""
from __future__ import annotations
from ._sarimax_base import _SARIMAXBase


class ARMAXWrapper(_SARIMAXBase):
    """ARMAX: d=0 forçado; grid AIC sobre p,q ∈ [0, p_max/q_max]."""

    def __init__(
        self,
        p_max: int = 2,
        q_max: int = 2,
        mode: str = "raw",
        fixed_order: tuple | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(
            d=0,          # ARMAX: sem integração da saída
            p_max=p_max,
            q_max=q_max,
            mode=mode,
            fixed_order=fixed_order,
            boxcox_lambda=boxcox_lambda,
        )

    def get_params_str(self) -> str:
        p, d, q = self._order
        return f"ARMAX({p},{d},{q})"
