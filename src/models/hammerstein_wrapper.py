"""Hammerstein: Não-linearidade estática na entrada f(y_lags) → Bloco Linear ARIMA(p,d,q).

Estrutura:
    z_i(t) = f( y(t-i) )   para i = 1..p   (lags transformados)
    y_hat(t) = ARIMA(p,d,q) ajustado com z_i como regressores exógenos

A função f é aplicada *elemento a elemento* a cada lag de y.
O bloco linear recebe [z_1, ..., z_p] como colunas exógenas via SARIMAX (padrão ARMAX).

Otimização via ALS (Alternating Least Squares):
    1. Init: f = identidade → [y_{t-1}, ..., y_{t-p}] como exog
    2. Ajusta ARIMA(z_lags) → obtém resíduos e_t = y - ARIMA(z)
    3. Re-ajusta f: fit(y_lags, y_compensated) onde y_compensated remove componente linear
    Repete até convergência ou max_iter

Bloco linear: ARIMA(p,d,q) via _SARIMAXBase
  - d via ADF/KPSS/PP; p,q por grid AIC com p_max=NL_P_MAX, q_max=NL_Q_MAX
  - lags p separado do order (p,d,q): 'p' aqui é o número de lags de y, não o AR order

Sub-round ARX: exógenas u adicionais concatenadas após z_lags no SARIMAX.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

import config as cfg
from ._sarimax_base import _SARIMAXBase
from ._nonlinear_blocks import _NonlinearBlock, make_block
from .base_model import BaseModel


class HammersteinWrapper(BaseModel):
    """Hammerstein(input NL f + ARIMA bloco linear) com ALS.

    Parâmetros
    ----------
    input_nl    : nome do bloco de entrada ('sigmoid','piecewise','bspline','halfspace')
    n_knots     : nós para BSplineBlock
    n_lags      : número de lags y a transformar (padrão: NL_P_MAX)
    d           : ordem de integração do bloco linear
    mode        : transformação de pré-processamento
    fixed_order : (p,d,q) do ARIMA interno pré-selecionado
    nl_lambda   : regularização L2 no bloco NL
    als_max_iter: máximo de iterações ALS
    als_tol     : tolerância de convergência relativa
    """

    def __init__(
        self,
        input_nl: str = "sigmoid",
        n_knots: int | None = None,
        n_lags: int | None = None,
        d: int = 1,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        nl_lambda: float = 0.0,
        als_max_iter: int | None = None,
        als_tol: float | None = None,
        nl_fixed_tau: float | None = None,
    ) -> None:
        self.input_nl = input_nl
        self.n_knots = n_knots
        # Breakpoint do bloco piecewise congelado entre dobras (Round 36); None = reescolhido
        # por grade a cada fit, o comportamento de todos os rounds anteriores.
        self.nl_fixed_tau = nl_fixed_tau
        self.n_lags = n_lags if n_lags is not None else cfg.NL_P_MAX
        self.d = d
        self.mode = mode
        self._fixed_order = fixed_order
        self.nl_lambda = nl_lambda
        self.als_max_iter = als_max_iter if als_max_iter is not None else cfg.ALS_MAX_ITER
        self.als_tol = als_tol if als_tol is not None else cfg.ALS_TOL

        self._linear: _SARIMAXBase | None = None
        self._f: _NonlinearBlock | None = None
        self._als_iters: int = 0
        self._als_converged: bool = False
        self._y_original: pd.Series | None = None
        self._last_z_lags: np.ndarray | None = None  # últimos lags transformados para predict()

    # ------------------------------------------------------------------
    # Utilitário: constrói DataFrame de lags transformados
    # ------------------------------------------------------------------

    def _make_z_exog(
        self, y: pd.Series, f: _NonlinearBlock
    ) -> pd.DataFrame:
        """Constrói DataFrame com z_i = f(y_{t-i}) para i=1..n_lags.

        As primeiras n_lags linhas terão NaN (sem histórico suficiente).
        """
        yv = y.values.astype(float)
        cols = {}
        for lag in range(1, self.n_lags + 1):
            shifted = np.full_like(yv, np.nan)
            shifted[lag:] = f.transform(yv[:-lag])
            cols[f"z_lag{lag}"] = shifted

        df = pd.DataFrame(cols, index=y.index)
        return df.ffill().bfill().fillna(0.0)

    # ------------------------------------------------------------------
    # ALS fit
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "HammersteinWrapper":
        self._y_original = y.copy()
        y_arr = y.values.astype(float)

        # --- Init: f = identidade ---
        f = make_block(self.input_nl, self.n_knots)
        if self.nl_fixed_tau is not None:
            if not hasattr(f, "fixed_tau"):
                raise ValueError("nl_fixed_tau só se aplica ao bloco piecewise")
            f.fixed_tau = self.nl_fixed_tau
        f.set_lambda(self.nl_lambda)

        # Primeiro ALS: sem transformação real, apenas lags diretos como exog
        identity_lags = self._make_z_exog_raw(y)
        z_exog = _merge_exog(identity_lags, u)

        lin = _SARIMAXBase(
            d=self.d,
            p_max=cfg.NL_P_MAX,
            q_max=cfg.NL_Q_MAX,
            mode=self.mode,
            fixed_order=self._fixed_order,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin.fit(y, z_exog)

        self._order = lin.order
        prev_pred = np.full_like(y_arr, np.nan)

        for iteration in range(self.als_max_iter):
            # Passo 2: pegar fitted values do ARIMA atual
            try:
                w_hat = np.asarray(lin._fitted.fittedvalues, dtype=float)
            except Exception:
                break

            offset = len(y_arr) - len(w_hat)
            y_aligned = y_arr[offset:]

            if len(w_hat) < 3:
                break

            # Ajusta f: empilha pares (y(t-i), y(t)) para todos os lags i
            # Garante alinhamento: x_i = y[0:n-i], y_i = y[i:n]
            all_lags_x = np.concatenate([
                y_arr[: len(y_arr) - lag]
                for lag in range(1, self.n_lags + 1)
            ])
            all_lags_y = np.concatenate([
                y_arr[lag:]
                for lag in range(1, self.n_lags + 1)
            ])
            f.fit(all_lags_x, all_lags_y)

            # Constrói z_exog com f treinado
            z_df = self._make_z_exog(y, f)
            z_full = _merge_exog(z_df, u)

            # Passo 3: re-ajusta ARIMA com z transformado
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lin2 = _SARIMAXBase(
                    d=self.d,
                    p_max=cfg.NL_P_MAX,
                    q_max=cfg.NL_Q_MAX,
                    mode=self.mode,
                    fixed_order=self._order,
                )
                lin2.fit(y, z_full)

            # Predição atual para verificar convergência
            try:
                current_w = np.asarray(lin2._fitted.fittedvalues, dtype=float)
                current_pred = np.full_like(y_arr, y_arr.mean())
                current_pred[len(y_arr) - len(current_w):] = current_w
            except Exception:
                break

            if not np.all(np.isnan(prev_pred)):
                delta = np.linalg.norm(current_pred - prev_pred)
                ref = np.linalg.norm(prev_pred)
                if ref > 0 and delta / ref < self.als_tol:
                    self._als_converged = True
                    lin = lin2
                    break

            prev_pred = current_pred.copy()
            lin = lin2

        self._linear = lin
        self._f = f
        self._als_iters = iteration + 1

        # Guardar últimos lags transformados para predict()
        last_vals = y_arr[-self.n_lags:]
        self._last_z_lags = f.transform(last_vals)

        # Guardar exog u para predict()
        self._u_last: pd.DataFrame | None = (
            u.iloc[[-1]] if u is not None else None
        )
        return self

    def _make_z_exog_raw(self, y: pd.Series) -> pd.DataFrame:
        """Lags sem transformação (identidade — para inicialização ALS)."""
        yv = y.values.astype(float)
        cols = {}
        for lag in range(1, self.n_lags + 1):
            shifted = np.full_like(yv, np.nan)
            shifted[lag:] = yv[:-lag]
            cols[f"z_lag{lag}"] = shifted
        df = pd.DataFrame(cols, index=y.index)
        return df.ffill().bfill().fillna(0.0)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        # Os lags transformados já estão no ARIMA como exog congelado
        # O predict usa o z_lag do último passo conhecido
        z_future = pd.DataFrame(
            {f"z_lag{i+1}": [float(self._last_z_lags[-(i+1)])]
             for i in range(self.n_lags)},
        )
        u_fut = _merge_exog(z_future, u_future)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._linear.predict(steps=steps, u_future=u_fut)

    def get_params_str(self) -> str:
        if self._linear is not None:
            p, d, q = self._linear.order
        else:
            p, d, q = 0, self.d, 0
        return f"Hammerstein(p={p},d={d},q={q},f={self.input_nl})"

    @property
    def als_info(self) -> dict:
        return {"converged": self._als_converged, "iters": self._als_iters}


# ---------------------------------------------------------------------------
# Utilitário local
# ---------------------------------------------------------------------------

def _merge_exog(
    z: pd.DataFrame, u: pd.DataFrame | None
) -> pd.DataFrame:
    """Concatena lags transformados z com exógenas u (se presentes)."""
    if u is None:
        return z
    # Alinhar pelo índice
    u_aligned = u.reindex(z.index).ffill().bfill().fillna(0.0)
    return pd.concat([z, u_aligned], axis=1)
