"""Damped-trend structural state-space model — Gardner & McKenzie (1985).

Dedicated wrapper requested after Round 13 (science_log "Próximo" (c)): TAEE4×resultado_bruto's
best-FIT candidate is the undamped local-linear-trend `SSWrapper` spec (FIT=51.3%, U₂=1.041 —
fails the strict gate, Theil U2 < 1 rule), because an unbounded linear trend overshoots at
h=1 out-of-sample. `ETSWrapper` already offers ETS(A,Ad,N) (damped trend) as an AICc-selected
candidate, but as a **single-source-of-error** (SSOE) innovations-form model (Hyndman et al. 2002)
it was tried for this combo in Round 12 and did not resolve the gate. This wrapper implements
damped trend in the **multi-source-of-error** (MSOE) *structural* form instead — the same Kalman-
filter/`UnobservedComponents` framework `SSWrapper` already uses (Harvey, 1989) — extended with a
damping coefficient phi on the slope's transition equation, the mechanism Gardner & McKenzie (1985)
show reduces multi-step forecast variance for noisy trended series relative to an undamped linear
trend. `statsmodels.tsa.statespace.structural.UnobservedComponents` has no `damped_trend` option
(only `damped_cycle`), so this MSOE damped-trend spec does not exist as a built-in and is
implemented here directly via `statsmodels.tsa.statespace.mlemodel.MLEModel`.

State-space form (local linear trend, additive, damped slope):
    level_t = level_{t-1} + slope_{t-1} + eta_level_t,   eta_level_t ~ N(0, sigma2_level)
    slope_t = phi * slope_{t-1} + eta_slope_t,            eta_slope_t ~ N(0, sigma2_slope)
    y_t      = level_t + eps_t,                            eps_t ~ N(0, sigma2_eps)

`phi` is constrained to (0.80, 0.98) — Gardner & McKenzie's own recommended practical range: phi
near 1 degenerates to the undamped case already covered by `SSWrapper`, phi near 0 collapses the
slope's memory to noise, and both ends are poorly identified at MIN_TRAIN=20 (see o protocolo do painel
"Guard for small N").

Univariate only — no exogenous support, matching the precedent set by `ETSWrapper`/`ThetaWrapper`
(consistency across the univariate state-space family; TAEE4×resultado_bruto's one exogenous
candidate, IPA-M, was already tried in linear ARX form — ARMAX, Round 02 v2 — and underperformed
the champion, FIT=42.0%/U₂=0.964 vs ARIMA-diff's 44.0%/0.931). Adding a second, harder-to-verify
degree of freedom (a Kalman-filter regression state) to a first version of a new wrapper is not
justified by that prior evidence.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from .base_model import BaseModel

_PHI_LO, _PHI_HI = 0.80, 0.98


class _DampedTrendSS(MLEModel):
    """Local linear trend, MSOE, slope damped by `phi` in (0.80, 0.98)."""

    def __init__(self, endog: np.ndarray) -> None:
        super().__init__(endog, k_states=2, k_posdef=2,
                          initialization="approximate_diffuse")
        self["design", 0, 0] = 1.0
        self["transition", 0, 0] = 1.0
        self["transition", 0, 1] = 1.0
        self["selection", 0, 0] = 1.0
        self["selection", 1, 1] = 1.0

    @property
    def param_names(self) -> list[str]:
        return ["sigma2.eps", "sigma2.level", "sigma2.slope", "phi"]

    @property
    def start_params(self) -> np.ndarray:
        var_y = float(np.nanvar(self.endog)) or 1.0
        return np.array([var_y * 0.5, var_y * 0.1, var_y * 0.01, 0.0])

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        constrained = unconstrained.copy()
        constrained[:3] = unconstrained[:3] ** 2
        constrained[3] = _PHI_LO + (_PHI_HI - _PHI_LO) / (1.0 + np.exp(-unconstrained[3]))
        return constrained

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        unconstrained = constrained.copy()
        unconstrained[:3] = np.sqrt(np.maximum(constrained[:3], 1e-10))
        p = np.clip((constrained[3] - _PHI_LO) / (_PHI_HI - _PHI_LO), 1e-6, 1 - 1e-6)
        unconstrained[3] = np.log(p / (1 - p))
        return unconstrained

    def update(self, params: np.ndarray, **kwargs) -> np.ndarray:
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov", 0, 0] = params[1]
        self["state_cov", 1, 1] = params[2]
        self["transition", 1, 1] = params[3]
        return params


class DampedTrendWrapper(BaseModel):
    """Damped local linear trend, MSOE structural state space (Gardner & McKenzie, 1985)."""

    def __init__(self, mode: str = "raw", boxcox_lambda: float | None = None) -> None:
        self.mode = mode
        self.boxcox_lambda = boxcox_lambda
        self._fitted = None
        self._params_str = "DAMPEDTREND"
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

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "DampedTrendWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _DampedTrendSS(yv).fit(disp=False, method="lbfgs", maxiter=200)

        self._fitted = m
        phi = float(m.params[3])
        self._params_str = f"DAMPEDTREND(phi={phi:.3f})"
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
