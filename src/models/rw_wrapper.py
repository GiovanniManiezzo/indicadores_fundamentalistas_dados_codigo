"""Benchmarks naïve: Random Walk e Random Walk com Drift.

Denominadores do Theil U₂ — todo modelo deve ter U₂ < 1 para justificar complexidade.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base_model import BaseModel


class RandomWalkModel(BaseModel):
    """RW: ŷ_{t+h} = y_t  (sem drift)."""

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "RandomWalkModel":
        self._last = float(y.iloc[-1])
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(steps, self._last)

    def get_params_str(self) -> str:
        return "RW"


class RandomWalkDriftModel(BaseModel):
    """RW com drift: ŷ_{t+h} = y_t + h × mean(Δy_treino)."""

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "RandomWalkDriftModel":
        self._last  = float(y.iloc[-1])
        self._drift = float(y.diff().dropna().mean())
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        return np.array([self._last + i * self._drift for i in range(1, steps + 1)])

    def get_params_str(self) -> str:
        return f"RW_Drift(drift={self._drift:.4g})"
