"""ETS (Error-Trend-Seasonal, state-space): Hyndman, Koehler, Snyder & Grose (2002).

Univariado — sem suporte a exógenas (statsmodels.ETSModel não aceita regressores).
Sem componente sazonal: N=64 trimestral cobre ~16 ciclos anuais, insuficiente para
estimar sazonalidade trimestral com robustez (literature synthesis, Petropoulos et al. 2022,
§2.3.1, mesma ressalva já aplicada ao SS local-level em protocolo metodológico).

Specs candidatos (erro aditivo, sem sazonalidade):
    ETS(A,N,N)  — nível local, sem tendência
    ETS(A,A,N)  — tendência aditiva
    ETS(A,Ad,N) — tendência aditiva amortecida (damped trend, Gardner & McKenzie 1985)

Seleção por menor AICc entre os specs que convergem (Hyndman et al. 2002 recomendam AICc
para seleção automática dentro da família ETS; `_sarimax_base.py` usa AIC simples porque
opera em grids maiores — aqui, com apenas 3 candidatos e MIN_TRAIN=20, AICc é a escolha
mais robusta, conforme também recomendado por Petropoulos et al. 2022 para amostras
pequenas).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from .base_model import BaseModel

_ETS_SPECS = [
    {"trend": None, "damped_trend": False},   # ETS(A,N,N)
    {"trend": "add", "damped_trend": False},  # ETS(A,A,N)
    {"trend": "add", "damped_trend": True},   # ETS(A,Ad,N)
]
_ETS_SPEC_NAMES = ["A,N,N", "A,A,N", "A,Ad,N"]


class ETSWrapper(BaseModel):
    """Exponential Smoothing em espaço de estados (statsmodels ETSModel), erro aditivo.

    `fixed_spec`: índice em `_ETS_SPECS` a congelar entre folds do walk-forward CV —
    a seleção deve ocorrer uma única vez na janela inicial `MIN_TRAIN` (anti-leakage),
    mesmo padrão de `fixed_order` no `ARIMAWrapper` / `fixed_spec` no `SSWrapper`.
    """

    def __init__(
        self,
        mode: str = "raw",
        fixed_spec: int | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        self.mode = mode
        self.fixed_spec = fixed_spec
        self.fixed_spec_: int | None = None
        self.boxcox_lambda = boxcox_lambda
        self._fitted = None
        self._params_str = "ETS(A,N,N)"
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

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "ETSWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values

        indices = [self.fixed_spec] if self.fixed_spec is not None else range(len(_ETS_SPECS))

        best_aicc = np.inf
        best_m = None
        best_idx = None
        for idx in indices:
            spec = _ETS_SPECS[idx]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ETSModel(
                        yv, error="add", initialization_method="estimated", **spec
                    ).fit(disp=False)
                if np.isfinite(m.aicc) and m.aicc < best_aicc:
                    best_aicc = m.aicc
                    best_m = m
                    best_idx = idx
            except Exception:
                continue

        if best_m is None:
            raise RuntimeError(f"ETS falhou para todos os specs [mode={self.mode}]")

        self._fitted = best_m
        self._params_str = f"ETS({_ETS_SPEC_NAMES[best_idx]})"
        self.fixed_spec_ = best_idx
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
        return self._params_str
