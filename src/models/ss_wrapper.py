"""SS (State Space): UnobservedComponents — nível local ou tendência linear local.

Usa filtro de Kalman (statsmodels). Exógenas entram como regressores no nível.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .base_model import BaseModel

_SS_SPECS = [
    {"level": True, "stochastic_level": True},                              # local level
    {"level": True, "stochastic_level": True, "trend": True,
     "stochastic_trend": False},                                             # local level + slope fixo
    {"level": True, "stochastic_level": True, "trend": True,
     "stochastic_trend": True},                                              # local linear trend
]
_SS_SPEC_NAMES = ["local level", "local level + slope fixo", "local linear trend"]


class SSWrapper(BaseModel):
    """State Space via Kalman filter (UnobservedComponents statsmodels).

    Seleção de spec por menor AIC entre os candidatos que convergem (mesmo critério
    usado pela família ARIMA em `_sarimax_base.py`) — antes desta correção, o loop
    retornava no primeiro spec que não lançasse exceção, o que na prática significava
    sempre "local level" (o spec mais simples quase sempre converge), deixando os specs
    de tendência inalcançáveis em toda a história do projeto (ver science_log Round 12).

    `fixed_spec`: índice em `_SS_SPECS` a congelar entre folds do walk-forward CV, no
    mesmo padrão de `fixed_order` do `ARIMAWrapper` — a seleção do spec deve ocorrer uma
    única vez na janela inicial `MIN_TRAIN`, nunca por fold (anti-leakage).
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
        self._params_str = "SS(local level)"
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

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "SSWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values

        u_vals: np.ndarray | None = None
        if u is not None:
            u_a = u.iloc[-len(y_t):].ffill().bfill().fillna(0.0)
            u_vals = u_a.values.astype(float)

        indices = [self.fixed_spec] if self.fixed_spec is not None else range(len(_SS_SPECS))

        best_aic = np.inf
        best_m = None
        best_idx = None
        for idx in indices:
            spec, name = _SS_SPECS[idx], _SS_SPEC_NAMES[idx]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kwargs = {**spec, "endog": yv}
                    if u_vals is not None:
                        kwargs["exog"] = u_vals
                    m = UnobservedComponents(**kwargs).fit(
                        disp=False, method="lbfgs", maxiter=200
                    )
                if np.isfinite(m.aic) and m.aic < best_aic:
                    best_aic = m.aic
                    best_m = m
                    best_idx = idx
            except Exception:
                continue

        if best_m is None:
            raise RuntimeError(f"SS falhou para todos os specs [mode={self.mode}]")

        self._fitted = best_m
        self._params_str = f"SS({_SS_SPEC_NAMES[best_idx]})"
        self.fixed_spec_ = best_idx
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        u_fut_vals: np.ndarray | None = None
        if u_future is not None:
            u_f = u_future.ffill().bfill().fillna(0.0)
            u_fut_vals = u_f.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = np.asarray(
                    self._fitted.forecast(steps=steps, exog=u_fut_vals), dtype=float
                )
            except Exception:
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
