"""NARX: Nonlinear ARX via regressão polinomial (sklearn Ridge).

Características:
  - p lags de y como entradas (y_{t-1}, ..., y_{t-p})
  - exógenas u_t como entradas adicionais (se disponíveis)
  - PolynomialFeatures(degree) + Ridge(alpha) como modelo não-linear
  - d=0 implícito: work on series as-is (mode controla pré-processamento)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from .base_model import BaseModel


class NARXWrapper(BaseModel):
    """NARX(p, degree): regressão polinomial sobre p lags de y + u atual."""

    def __init__(
        self,
        p: int = 3,
        degree: int = 2,
        alpha: float = 1.0,
        mode: str = "raw",
        boxcox_lambda: float | None = None,
    ) -> None:
        self.p = p
        self.degree = degree
        self.alpha = alpha
        self.mode = mode
        self.boxcox_lambda = boxcox_lambda
        self._pipeline: Pipeline | None = None
        self._last_p_values: np.ndarray | None = None  # y_t[-p:] para predict
        self._n_u_cols: int = 0
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

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "NARXWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        n = len(y_t)
        yv = y_t.values

        u_aligned: pd.DataFrame | None = None
        if u is not None:
            u_aligned = u.iloc[-n:].ffill().bfill().fillna(0.0)
            self._n_u_cols = u_aligned.shape[1]
        else:
            self._n_u_cols = 0

        X_rows, Y_vals = [], []
        for t in range(self.p, n):
            row = yv[t - self.p:t][::-1].tolist()   # [y_{t-1}, ..., y_{t-p}]
            if u_aligned is not None:
                row.extend(u_aligned.values[t].tolist())   # u_t
            X_rows.append(row)
            Y_vals.append(float(yv[t]))

        if len(X_rows) < 3:
            raise RuntimeError(
                f"NARX: amostras insuficientes (n={len(X_rows)}) para p={self.p}"
            )

        X = np.array(X_rows, dtype=float)
        Y = np.array(Y_vals, dtype=float)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=self.degree, include_bias=False)),
            # solver='svd' usa decomposição SVD: estável para matrizes ill-conditioned
            # (n_amostras << n_features polinomiais com dados financeiros em escala 1e9)
            ("ridge",  Ridge(alpha=self.alpha, solver="svd")),
        ])
        self._pipeline.fit(X, Y)

        # Guarda últimos p valores transformados para predict()
        self._last_p_values = yv[-self.p:].copy()
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        # Constrói vetor de features para o próximo passo
        row = self._last_p_values[::-1].tolist()   # [y_{last}, ..., y_{last-p+1}]

        if self._n_u_cols > 0:
            if u_future is not None:
                vals = u_future.ffill().bfill().fillna(0.0).values[0, :self._n_u_cols].tolist()
            else:
                vals = [0.0] * self._n_u_cols
            row.extend(vals)

        X_pred = np.array([row], dtype=float)
        raw = self._pipeline.predict(X_pred).flatten()[:steps]

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
        return f"NARX(p={self.p},deg={self.degree})"
