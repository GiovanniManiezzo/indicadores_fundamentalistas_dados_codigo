"""SARIMA: Seasonal ARIMA sem exógenas (via statsmodels SARIMAX).

s=4 para dados trimestrais.
Grid de seleção: p,q ∈ [0, p_max]; P,Q ∈ [0,1]; D ∈ {0,1}.
d fixado pelos testes de raiz unitária (igual ao ARIMAWrapper).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseModel


class SARIMAWrapper(BaseModel):
    """SARIMA(p,d,q)(P,D,Q)[s] — sem exógenas."""

    def __init__(
        self,
        d: int = 1,
        p_max: int = 1,
        q_max: int = 1,
        s: int = 4,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        fixed_seasonal: tuple[int, int, int, int] | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        self.d = d
        self.p_max = p_max
        self.q_max = q_max
        self.s = s
        self.mode = mode
        self._fixed_order = fixed_order
        self._fixed_seasonal = fixed_seasonal
        self.boxcox_lambda = boxcox_lambda
        self._fitted = None
        self._order: tuple[int, int, int] = fixed_order or (0, d, 1)
        self._seasonal: tuple[int, int, int, int] = fixed_seasonal or (0, 0, 0, s)
        self._y_original: pd.Series | None = None

    def _transform(self, y: pd.Series) -> pd.Series:
        mode = self.mode
        if mode == "raw":
            return y
        if mode == "log":
            if (y <= 0).any():
                raise ValueError(f"log requer valores positivos; mín={y.min():.4g}")
            return np.log(y)
        if mode == "diff":
            return y.diff().dropna()
        if mode == "log+diff":
            if (y <= 0).any():
                raise ValueError(f"log+diff requer valores positivos; mín={y.min():.4g}")
            return np.log(y).diff().dropna()
        if mode == "boxcox":
            if (y <= 0).any():
                raise ValueError(f"boxcox requer valores positivos; mín={y.min():.4g}")
            if self.boxcox_lambda is None:
                raise ValueError("boxcox requer boxcox_lambda congelado")
            lam = self.boxcox_lambda
            if abs(lam) < 1e-6:
                return np.log(y)
            return (np.power(y, lam) - 1.0) / lam
        raise ValueError(f"Modo desconhecido: {mode!r}")

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "SARIMAWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values

        if self._fixed_order is not None and self._fixed_seasonal is not None:
            p, _, q = self._fixed_order
            P, D, Q, s = self._fixed_seasonal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted = SARIMAX(
                    yv, order=(p, self.d, q),
                    seasonal_order=(P, D, Q, s),
                ).fit(disp=False)
            return self

        best_aic = np.inf
        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                for P in range(2):       # P ∈ {0,1}
                    for Q in range(2):   # Q ∈ {0,1}
                        for D in range(2):  # D ∈ {0,1}
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    m = SARIMAX(
                                        yv,
                                        order=(p, self.d, q),
                                        seasonal_order=(P, D, Q, self.s),
                                    ).fit(disp=False)
                                if np.isfinite(m.aic) and m.aic < best_aic:
                                    best_aic = m.aic
                                    self._order = (p, self.d, q)
                                    self._seasonal = (P, D, Q, self.s)
                                    self._fitted = m
                            except Exception:
                                pass

        if self._fitted is None:
            raise RuntimeError(f"SARIMA falhou [mode={self.mode}, d={self.d}]")
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = np.asarray(self._fitted.forecast(steps=steps), dtype=float)

        mode = self.mode
        if mode == "raw":
            return raw
        if mode == "log":
            return np.exp(raw)
        if mode == "diff":
            last = float(self._y_original.iloc[-1])
            return np.cumsum(raw) + last
        if mode == "log+diff":
            last = float(self._y_original.iloc[-1])
            log_last = np.log(last) if last > 0 else 0.0
            return np.exp(np.cumsum(raw) + log_last)
        if mode == "boxcox":
            if self.boxcox_lambda is None:
                raise ValueError("boxcox requer boxcox_lambda congelado")
            lam = self.boxcox_lambda
            if abs(lam) < 1e-6:
                return np.exp(raw)
            base = np.clip(lam * raw + 1.0, 1e-12, None)
            return np.power(base, 1.0 / lam)
        raise ValueError(f"Modo desconhecido: {mode!r}")

    def get_params_str(self) -> str:
        p, d, q = self._order
        P, D, Q, s = self._seasonal
        return f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self._seasonal
