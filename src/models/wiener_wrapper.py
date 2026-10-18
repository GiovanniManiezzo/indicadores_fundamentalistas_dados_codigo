"""Wiener: Bloco Linear ARIMA(p,d,q) → Não-linearidade estática na saída g(w).

Estrutura:
    y(t) = g( ARIMA(p,d,q) prediction w(t) )

Otimização via ALS (Alternating Least Squares):
    1. Ajusta ARIMA(p,d,q) em y → obtém predições internas w_hat
    2. Ajusta g: fit(w_hat, y)
    3. Computa y_tilde = g⁻¹(y)  (inversa analítica/numérica)
    4. Re-ajusta ARIMA em y_tilde
    Repete até convergência (||Δpred||/||pred|| < ALS_TOL) ou max_iter

Bloco linear: ARIMA(p,d,q) via _SARIMAXBase
  - d fixado pelos testes ADF/KPSS/PP (protocolo da pipeline existente)
  - p,q por grid AIC; p_max=NL_P_MAX, q_max=NL_Q_MAX  (< ARIMA full para compensar NL params)
  - fixed_order congela (p,d,q) no walk-forward CV (evita re-grid em cada fold)

Sub-round ARX: exógenas u passadas como regressores adicionais no SARIMAX.
"""
from __future__ import annotations
import copy
import warnings
import numpy as np
import pandas as pd

import config as cfg
from ._sarimax_base import _SARIMAXBase
from ._nonlinear_blocks import _NonlinearBlock, make_block
from .base_model import BaseModel


class WienerWrapper(BaseModel):
    """Wiener(ARIMA + output NL g) com ALS.

    Parâmetros
    ----------
    output_nl   : nome do bloco de saída ('sigmoid','piecewise','bspline','halfspace')
    n_knots     : número de nós para BSplineBlock (ignorado pelos demais)
    d           : ordem de integração do bloco linear (ADF/KPSS/PP consenso)
    mode        : transformação de pré-processamento ('raw','log','diff','log+diff')
    fixed_order : (p,d,q) pré-selecionado — congela para walk-forward CV
    nl_lambda   : regularização L2 no bloco NL (0 = sem regularização)
    als_max_iter: máximo de iterações ALS
    als_tol     : tolerância de convergência relativa
    """

    def __init__(
        self,
        output_nl: str = "sigmoid",
        n_knots: int | None = None,
        d: int = 1,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        nl_lambda: float = 0.0,
        als_max_iter: int | None = None,
        als_tol: float | None = None,
        nl_fixed_tau: float | None = None,
    ) -> None:
        self.output_nl = output_nl
        self.n_knots = n_knots
        # Breakpoint do bloco piecewise congelado entre dobras (Round 36); None = reescolhido
        # por grade a cada fit, o comportamento de todos os rounds anteriores.
        self.nl_fixed_tau = nl_fixed_tau
        self.d = d
        self.mode = mode
        self._fixed_order = fixed_order
        self.nl_lambda = nl_lambda
        self.als_max_iter = als_max_iter if als_max_iter is not None else cfg.ALS_MAX_ITER
        self.als_tol = als_tol if als_tol is not None else cfg.ALS_TOL

        self._linear: _SARIMAXBase | None = None
        self._g: _NonlinearBlock | None = None
        self._als_iters: int = 0
        self._als_converged: bool = False
        self._y_original: pd.Series | None = None

    # ------------------------------------------------------------------
    # ALS fit
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "WienerWrapper":
        self._y_original = y.copy()
        y_arr = y.values.astype(float)

        # --- Inicialização: ARIMA sem NL (identidade em g) ---
        lin = _SARIMAXBase(
            d=self.d,
            p_max=cfg.NL_P_MAX,
            q_max=cfg.NL_Q_MAX,
            mode=self.mode,
            fixed_order=self._fixed_order,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin.fit(y, u)

        # Guardar ordem para predict() e future closures
        self._order = lin.order

        # --- ALS ---
        g = make_block(self.output_nl, self.n_knots)
        if self.nl_fixed_tau is not None:
            if not hasattr(g, "fixed_tau"):
                raise ValueError("nl_fixed_tau só se aplica ao bloco piecewise")
            g.fixed_tau = self.nl_fixed_tau
        g.set_lambda(self.nl_lambda)

        y_target = y.copy()   # alvo para o ARIMA; começa como y original
        cur_tgt = None        # bloco cujo inverso gerou y_target (None = identidade)
        lin_tgt = None        # idem, para o bloco linear final `lin`
        prev_pred = np.full_like(y_arr, np.nan)

        for iteration in range(self.als_max_iter):
            # Passo 1: ajusta bloco linear no alvo atual
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lin2 = _SARIMAXBase(
                    d=self.d,
                    p_max=cfg.NL_P_MAX,
                    q_max=cfg.NL_Q_MAX,
                    mode=self.mode,
                    fixed_order=self._order,  # fixo após 1ª iteração
                )
                lin2.fit(y_target, u)

            # Predições internas w_hat = ARIMA(y_target)
            # Usamos fitted values do modelo (in-sample)
            try:
                w_hat = lin2._fitted.fittedvalues
                if hasattr(w_hat, "values"):
                    w_hat = w_hat.values
                w_hat = np.asarray(w_hat, dtype=float)
            except Exception:
                break  # ARIMA não convergiu

            # Alinhar w_hat com y_arr (diff reduz comprimento em 1)
            offset = len(y_arr) - len(w_hat)
            y_aligned = y_arr[offset:]

            if len(w_hat) < 3:
                break

            # Passo 2: ajusta g(w_hat → y_aligned)
            g.fit(w_hat, y_aligned)

            # Predição atual: ŷ = g(w_hat)
            current_pred = np.full_like(y_arr, y_arr.mean())
            current_pred[offset:] = g.transform(w_hat)

            # Verificar convergência
            if not np.all(np.isnan(prev_pred)):
                delta = np.linalg.norm(current_pred - prev_pred)
                ref = np.linalg.norm(prev_pred)
                if ref > 0 and delta / ref < self.als_tol:
                    self._als_converged = True
                    lin = lin2
                    lin_tgt = cur_tgt
                    break

            prev_pred = current_pred.copy()

            # Passo 3: y_tilde = g⁻¹(y) para re-treinar o ARIMA
            try:
                y_tilde_vals = g.inverse_transform(y_arr)
                # Substituir nan/inf por y original
                bad = ~np.isfinite(y_tilde_vals)
                y_tilde_vals[bad] = y_arr[bad]
                y_target = pd.Series(y_tilde_vals, index=y.index, name=y.name)
            except Exception:
                break

            lin = lin2
            lin_tgt = cur_tgt
            cur_tgt = copy.deepcopy(g)

        self._linear = lin
        self._g = g
        # Bloco cujo inverso gerou o alvo do `lin` final — o ALS termina com o bloco linear
        # ajustado sobre g⁻¹ da iteração anterior, não do `g` final. Só registrado (o predict
        # não o usa): a estimação recursiva (src/recursive.py) precisa do preditor exato.
        self._g_target = lin_tgt
        self._als_iters = iteration + 1
        return self

    # ------------------------------------------------------------------
    # predict (h=1 passo à frente)
    # ------------------------------------------------------------------

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        # w = ARIMA forecast (bloco linear)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self._linear.predict(steps=steps, u_future=u_future)

        # ŷ = g(w) — na escala original (ARIMA já fez inverse_transform)
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
        return f"Wiener(p={p},d={d},q={q},g={self.output_nl})"

    @property
    def als_info(self) -> dict:
        return {"converged": self._als_converged, "iters": self._als_iters}
