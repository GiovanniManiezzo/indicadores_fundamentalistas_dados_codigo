"""Wrapper ARIMA sobre statsmodels.

Regras:
- `d` fixado pelos testes de raiz unitária (preprocess.unit_root_tests); nunca
  sobrescrito por grid search.
- `p` e `q` selecionados por AIC com p,q ∈ [0, p_max] (default 4).
- `mode` controla a transformação aplicada antes do ajuste; inverse_transform
  é feito no predict() para retornar previsões na escala original.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .base_model import BaseModel


class ARIMAWrapper(BaseModel):

    def __init__(
        self,
        d: int = 1,
        p_max: int = 4,
        q_max: int = 4,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        boxcox_lambda: float | None = None,
    ):
        self.d     = d
        self.p_max = p_max
        self.q_max = q_max
        self.mode  = mode
        self._fixed_order = fixed_order
        self.boxcox_lambda = boxcox_lambda
        self._fitted  = None
        self._order: tuple[int, int, int] = fixed_order if fixed_order is not None else (1, d, 0)
        self._y_original: pd.Series | None = None

    # ------------------------------------------------------------------
    # Transformação interna
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "ARIMAWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)

        if self._fixed_order is not None:
            # Usar ordem pré-selecionada — sem grid search (performance do walk-forward CV)
            p, _, q = self._fixed_order
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted = ARIMA(y_t, order=(p, self.d, q)).fit()
            return self

        best_aic = np.inf
        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = ARIMA(y_t, order=(p, self.d, q)).fit()
                    if m.aic < best_aic:
                        best_aic  = m.aic
                        self._order   = (p, self.d, q)
                        self._fitted  = m
                except Exception:
                    pass

        if self._fitted is None:
            raise RuntimeError(
                f"ARIMA falhou para todos (p,q) [mode={self.mode}, d={self.d}]"
            )
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        """Retorna previsão na escala original após inverse_transform."""
        raw = np.asarray(self._fitted.forecast(steps=steps), dtype=float)
        mode = self.mode

        # statsmodels ARIMA(p,d,q).forecast() devolve previsão na escala
        # da série de entrada (nível, não diferença), independente de d.
        if mode == "raw":
            return raw
        if mode == "log":
            return np.exp(raw)
        if mode == "diff":
            # raw = Δŷ_{T+1..T+h}; cumsum + último valor original
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
        return f"ARIMA{self._order}"

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order
