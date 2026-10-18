"""Hammerstein-Wiener: f(y_lags) → Bloco Linear ARIMA(p,d,q) → g(w) → ŷ.

Estrutura completa:
    z_i(t)   = f( y(t-i) )          [bloco de entrada não-linear]
    w(t)     = ARIMA(p,d,q)(z)      [bloco linear dinâmico]
    y_hat(t) = g( w(t) )            [bloco de saída não-linear]

Otimização via ALS 3-blocos:
    (A) Fixa g, f → ajusta ARIMA: minimiza ||g⁻¹(y) - ARIMA(f(lags))||²
    (B) Fixa ARIMA, f → ajusta g: fit(w_hat, y)
    (C) Fixa ARIMA, g → ajusta f: fit(y_lags, ARIMA⁻¹(g⁻¹(y)))
    Repete até convergência (||Δŷ||/||ŷ|| < ALS_TOL) ou max_iter

Warm start: ARIMA puro na iteração 0 (identidade em f e g).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

import config as cfg
from ._sarimax_base import _SARIMAXBase
from ._nonlinear_blocks import _NonlinearBlock, make_block
from .base_model import BaseModel
from .hammerstein_wrapper import _merge_exog


class HWWrapper(BaseModel):
    """Hammerstein-Wiener (f → ARIMA → g) com ALS 3-blocos.

    Parâmetros
    ----------
    input_nl    : bloco de entrada ('sigmoid','piecewise','bspline','halfspace')
    output_nl   : bloco de saída  ('sigmoid','piecewise','bspline','halfspace')
    n_knots_in  : nós para BSpline na entrada
    n_knots_out : nós para BSpline na saída
    n_lags      : número de lags y como entradas do bloco f
    d           : ordem de integração do bloco linear
    mode        : pré-processamento
    fixed_order : (p,d,q) congelado para walk-forward CV
    nl_lambda   : regularização L2 nos dois blocos NL
    als_max_iter: máximo de iterações ALS
    als_tol     : tolerância de convergência relativa
    """

    def __init__(
        self,
        input_nl: str = "sigmoid",
        output_nl: str = "piecewise",
        n_knots_in: int | None = None,
        n_knots_out: int | None = None,
        n_lags: int | None = None,
        d: int = 1,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        nl_lambda: float = 0.0,
        als_max_iter: int | None = None,
        als_tol: float | None = None,
    ) -> None:
        self.input_nl = input_nl
        self.output_nl = output_nl
        self.n_knots_in = n_knots_in
        self.n_knots_out = n_knots_out
        self.n_lags = n_lags if n_lags is not None else cfg.NL_P_MAX
        self.d = d
        self.mode = mode
        self._fixed_order = fixed_order
        self.nl_lambda = nl_lambda
        self.als_max_iter = als_max_iter if als_max_iter is not None else cfg.ALS_MAX_ITER
        self.als_tol = als_tol if als_tol is not None else cfg.ALS_TOL

        self._linear: _SARIMAXBase | None = None
        self._f: _NonlinearBlock | None = None
        self._g: _NonlinearBlock | None = None
        self._als_iters: int = 0
        self._als_converged: bool = False
        self._y_original: pd.Series | None = None
        self._last_z_lags: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _make_z_exog(
        self, y_arr: np.ndarray, idx: pd.Index, f: _NonlinearBlock
    ) -> pd.DataFrame:
        cols = {}
        for lag in range(1, self.n_lags + 1):
            shifted = np.full_like(y_arr, np.nan)
            shifted[lag:] = f.transform(y_arr[:-lag])
            cols[f"z_lag{lag}"] = shifted
        return pd.DataFrame(cols, index=idx).ffill().bfill().fillna(0.0)

    def _make_z_exog_identity(
        self, y_arr: np.ndarray, idx: pd.Index
    ) -> pd.DataFrame:
        cols = {}
        for lag in range(1, self.n_lags + 1):
            shifted = np.full_like(y_arr, np.nan)
            shifted[lag:] = y_arr[:-lag]
            cols[f"z_lag{lag}"] = shifted
        return pd.DataFrame(cols, index=idx).ffill().bfill().fillna(0.0)

    def _fit_linear(
        self, y_target: pd.Series, z_exog: pd.DataFrame, u: pd.DataFrame | None
    ) -> _SARIMAXBase:
        u_full = _merge_exog(z_exog, u)
        lin = _SARIMAXBase(
            d=self.d,
            p_max=cfg.NL_P_MAX,
            q_max=cfg.NL_Q_MAX,
            mode=self.mode,
            fixed_order=getattr(self, "_order", self._fixed_order),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin.fit(y_target, u_full)
        return lin

    def _fitted_values(self, lin: _SARIMAXBase) -> np.ndarray:
        return np.asarray(lin._fitted.fittedvalues, dtype=float)

    # ------------------------------------------------------------------
    # ALS fit
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "HWWrapper":
        self._y_original = y.copy()
        y_arr = y.values.astype(float)

        # Blocos NL
        f = make_block(self.input_nl, self.n_knots_in)
        g = make_block(self.output_nl, self.n_knots_out)
        f.set_lambda(self.nl_lambda)
        g.set_lambda(self.nl_lambda)

        # Warm start: ARIMA puro (f=identidade, g=identidade)
        z_id = self._make_z_exog_identity(y_arr, y.index)
        lin = self._fit_linear(y, z_id, u)
        self._order = lin.order

        prev_pred = np.full_like(y_arr, np.nan)

        for iteration in range(self.als_max_iter):

            # --- Passo B: fixa lin e f, ajusta g ---
            try:
                w_hat = self._fitted_values(lin)
            except Exception:
                break
            offset = len(y_arr) - len(w_hat)
            y_aligned = y_arr[offset:]
            if len(w_hat) < 3:
                break

            g.fit(w_hat, y_aligned)

            # --- Passo C: fixa lin e g, ajusta f ---
            # Alvo para f: y_tilde = g⁻¹(y)
            try:
                y_tilde_vals = g.inverse_transform(y_arr)
                bad = ~np.isfinite(y_tilde_vals)
                y_tilde_vals[bad] = y_arr[bad]
            except Exception:
                y_tilde_vals = y_arr.copy()

            # Empilha pares alinhados (y(t-i), y_tilde(t)) para todos os lags i
            all_x = np.concatenate([
                y_arr[: len(y_arr) - lag]
                for lag in range(1, self.n_lags + 1)
            ])
            all_y_f = np.concatenate([
                y_tilde_vals[lag:]
                for lag in range(1, self.n_lags + 1)
            ])
            f.fit(all_x, all_y_f)

            # --- Passo A: fixa g e f, ajusta lin ---
            y_target_lin = pd.Series(y_tilde_vals, index=y.index, name=y.name)
            z_df = self._make_z_exog(y_arr, y.index, f)
            lin = self._fit_linear(y_target_lin, z_df, u)

            # Predição atual: ŷ = g(ARIMA(f(lags)))
            try:
                w_new = self._fitted_values(lin)
                current_pred = np.full_like(y_arr, y_arr.mean())
                off_new = len(y_arr) - len(w_new)
                current_pred[off_new:] = g.transform(w_new)
            except Exception:
                break

            # Convergência
            if not np.all(np.isnan(prev_pred)):
                delta = np.linalg.norm(current_pred - prev_pred)
                ref = np.linalg.norm(prev_pred)
                if ref > 0 and delta / ref < self.als_tol:
                    self._als_converged = True
                    break

            prev_pred = current_pred.copy()

        self._linear = lin
        self._f = f
        self._g = g
        self._als_iters = iteration + 1

        # Guardar últimos lags transformados para predict()
        self._last_z_lags = f.transform(y_arr[-self.n_lags:])
        self._u_last: pd.DataFrame | None = (
            u.iloc[[-1]] if u is not None else None
        )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        z_future = pd.DataFrame(
            {f"z_lag{i+1}": [float(self._last_z_lags[-(i+1)])]
             for i in range(self.n_lags)},
        )
        u_fut = _merge_exog(z_future, u_future)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._linear.predict(steps=steps, u_future=u_fut)

        if self._g is not None:
            try:
                return self._g.transform(w.astype(float))
            except Exception:
                return w
        return w

    def get_params_str(self) -> str:
        if self._linear is not None:
            p, d, q = self._linear.order
        else:
            p, d, q = 0, self.d, 0
        return f"HW(p={p},d={d},q={q},f={self.input_nl},g={self.output_nl})"

    @property
    def als_info(self) -> dict:
        return {"converged": self._als_converged, "iters": self._als_iters}
