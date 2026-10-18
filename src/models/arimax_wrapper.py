"""ARIMAX: ARIMA(p, d, q) com exógenas (d determinado pelos testes de raiz unitária).

∇^d A(L) y_t = B(L) u_t + C(L) e_t
"""
from __future__ import annotations
from ._sarimax_base import _SARIMAXBase


class ARIMAXWrapper(_SARIMAXBase):
    """ARIMAX: d recebe valor dos testes ADF/KPSS/PP; grid AIC sobre p,q."""

    def __init__(
        self,
        d: int = 1,
        p_max: int = 2,
        q_max: int = 2,
        mode: str = "raw",
        fixed_order: tuple | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(
            d=d,
            p_max=p_max,
            q_max=q_max,
            mode=mode,
            fixed_order=fixed_order,
            boxcox_lambda=boxcox_lambda,
        )

    def get_params_str(self) -> str:
        p, d, q = self._order
        return f"ARIMAX({p},{d},{q})"
