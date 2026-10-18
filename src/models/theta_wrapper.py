"""Theta method: Assimakopoulos & Nikolopoulos (2000).

Univariado — sem suporte a exógenas. Sem dessazonalização: N=64 trimestral cobre ~16
ciclos anuais, insuficiente para estimar sazonalidade trimestral com robustez (mesma
ressalva já aplicada em `ss_wrapper.py`/`ets_wrapper.py`; ver literature synthesis,
Petropoulos et al. 2022, §2.3.1).

Implementação via `statsmodels.tsa.forecasting.theta.ThetaModel` (mesma formulação de
decomposição em duas theta-lines do artigo original). Sem hiperparâmetro de
ordem/estrutura a congelar entre folds do walk-forward CV — `deseasonalize=False` é uma
escolha estática do projeto (não estimada por fold), então não há risco de leakage
estrutural aqui equivalente ao `fixed_order`/`fixed_spec` de outros wrappers.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel
from .base_model import BaseModel


class ThetaWrapper(BaseModel):
    """Método Theta clássico (statsmodels ThetaModel), sem dessazonalização."""

    def __init__(self, mode: str = "raw", boxcox_lambda: float | None = None) -> None:
        self.mode = mode
        self.boxcox_lambda = boxcox_lambda
        self._fitted = None
        self._params_str = "Theta"
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

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "ThetaWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ThetaModel(yv, deseasonalize=False).fit(disp=False)

        self._fitted = m
        self._params_str = "Theta"
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
