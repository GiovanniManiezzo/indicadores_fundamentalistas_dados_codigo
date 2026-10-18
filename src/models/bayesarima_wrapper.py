"""Bayesian ARIMA via MAP (aproximação de Laplace ao posterior), prior Minnesota
sobre os coeficientes AR/MA (Litterman 1986; Doan, Litterman & Sims 1984).

MAP, não posterior completa: sem MCMC (pymc/arviz não instalados no ambiente) — a
posteriori é aproximada por uma normal centrada no MLE com precisão dada pela
informação de Fisher observada (`cov_params()`), combinada em forma fechada com um
prior Normal(0, 1/λ_l) por coeficiente AR/MA de lag l, λ_l = λ0·l² (shrinkage cresce
com a ordem do lag). Ver protocolo metodológico §22 para a derivação completa e a limitação
declarada (MAP ≠ posterior sampling).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ._sarimax_base import _SARIMAXBase


class BayesArimaWrapper(_SARIMAXBase):
    """Bayesian ARIMA(X): MLE + shrinkage Minnesota (MAP) sobre AR/MA apenas.

    d        : ordem de integração (ADF/KPSS/PP consensus) — nunca de grid search.
    p_max,q_max : grid AIC para a ordem (idêntico ao ARIMAXWrapper — a ordem é
                  escolhida por MLE não-penalizado, para ficar comparável entre modelos).
    fixed_order   : (p,d,q) congelado por fold (anti-leakage, mesmo padrão da base).
    fixed_lambda0 : tightness global do prior, congelado por fold junto com fixed_order.
                    Quando None e fixed_order é None, ambos são selecionados uma única
                    vez na janela inicial (ordem por AIC; λ0 por holdout interno).
    """

    LAMBDA0_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)

    def __init__(
        self,
        d: int = 1,
        p_max: int = 2,
        q_max: int = 2,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        fixed_lambda0: float | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(
            d=d, p_max=p_max, q_max=q_max, mode=mode, fixed_order=fixed_order,
            boxcox_lambda=boxcox_lambda,
        )
        self._fixed_lambda0 = fixed_lambda0
        self._lambda0: float = fixed_lambda0 if fixed_lambda0 is not None else self.LAMBDA0_GRID[0]

    @property
    def lambda0(self) -> float:
        return self._lambda0

    # ------------------------------------------------------------------
    # MAP core: MLE -> Laplace shrinkage do bloco AR/MA apenas
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_map(y_vals: np.ndarray, u_vals: np.ndarray | None, order: tuple[int, int, int], lambda0: float):
        """Ajusta SARIMAX(order) via MLE e desloca os coeficientes AR/MA para o MAP
        sob prior Minnesota, via aproximação normal (Laplace) à posteriori. trend,
        exógenas e sigma2 permanecem no valor de MLE (não penalizados). Lança
        exceção se o MLE não convergir ou a covariância do bloco AR/MA não for utilizável.
        """
        p, d, q = order
        mod = SARIMAX(y_vals, exog=u_vals, order=(p, d, q))
        mle_res = mod.fit(disp=False)

        if not mle_res.mle_retvals.get("converged", True):
            raise RuntimeError("MLE não convergiu")

        theta_mle = np.asarray(mle_res.params, dtype=float)
        if not np.all(np.isfinite(theta_mle)):
            raise RuntimeError("MLE retornou parâmetros não finitos")

        k_ar, k_ma = mod.k_ar, mod.k_ma
        ar_start = mod.k_trend + mod.k_exog
        ma_start = ar_start + k_ar
        idx = list(range(ar_start, ma_start + k_ma))

        theta_map = theta_mle.copy()
        if idx:  # há termos AR e/ou MA para encolher
            cov_full = np.asarray(mle_res.cov_params(), dtype=float)
            cov_sub = cov_full[np.ix_(idx, idx)]
            if not np.all(np.isfinite(cov_sub)):
                raise RuntimeError("cov_params() não finita no bloco AR/MA")

            lags = [i + 1 for i in range(k_ar)] + [i + 1 for i in range(k_ma)]
            prior_prec = np.diag([lambda0 * (l ** 2) for l in lags])
            lik_prec = np.linalg.pinv(cov_sub, rcond=1e-10)
            post_cov = np.linalg.pinv(lik_prec + prior_prec, rcond=1e-10)
            theta_map[idx] = post_cov @ lik_prec @ theta_mle[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_map = mod.smooth(theta_map)

        sigma2 = theta_map[-1]
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise RuntimeError("MAP degenerou sigma2 <= 0")

        return mod, fitted_map

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "BayesArimaWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)

        u_vals: np.ndarray | None = None
        if u is not None:
            u_a = u.iloc[-len(y_t):].ffill().bfill().fillna(0.0)
            u_vals = u_a.values.astype(float)

        if self._fixed_order is not None:
            if self._fixed_lambda0 is None:
                raise ValueError("fixed_lambda0 é obrigatório quando fixed_order é passado")
            p, _, q = self._fixed_order
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, fitted = self._fit_map(y_t.values, u_vals, self._fixed_order, self._fixed_lambda0)
                except Exception:
                    # fallback: sem exógenas (mesmo padrão de _SARIMAXBase.fit)
                    _, fitted = self._fit_map(y_t.values, None, self._fixed_order, self._fixed_lambda0)
            self._fitted = fitted
            self._order = self._fixed_order
            self._lambda0 = self._fixed_lambda0
            return self

        # ── Estágio 1: ordem (p,q) por AIC não-penalizado — idêntico a _SARIMAXBase,
        #    mantém a ordem comparável entre modelos (ARIMA/ARIMAX usam o mesmo critério).
        best_aic = np.inf
        order = None
        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(y_t.values, exog=u_vals, order=(p, self.d, q)).fit(disp=False)
                    if np.isfinite(m.aic) and m.aic < best_aic:
                        best_aic = m.aic
                        order = (p, self.d, q)
                except Exception:
                    pass
        if order is None:
            raise RuntimeError(f"BayesArimaWrapper: nenhuma ordem convergiu [mode={self.mode}, d={self.d}]")

        # ── Estágio 2: λ0 por holdout interno na janela inicial (mesmo espírito do
        #    INVMSE de CombinationWrapper — CV interno, congelado depois; AIC não serve
        #    aqui porque penaliza só k, não a shrinkage em si — sempre favoreceria λ0→0).
        n = len(y_t)
        n_val = max(4, n // 5)
        n_val = min(n_val, n - 8)  # garante treino mínimo utilizável
        best_lambda0 = self.LAMBDA0_GRID[0]
        if n_val >= 2:
            y_tr, y_val = y_t.values[:-n_val], y_t.values[-n_val:]
            u_tr = u_vals[:-n_val] if u_vals is not None else None
            u_val = u_vals[-n_val:] if u_vals is not None else None
            best_mse = np.inf
            for lambda0 in self.LAMBDA0_GRID:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, fitted_tr = self._fit_map(y_tr, u_tr, order, lambda0)
                        fc = np.asarray(fitted_tr.forecast(steps=n_val, exog=u_val), dtype=float)
                    mse = float(np.mean((fc - y_val) ** 2))
                    if np.isfinite(mse) and mse < best_mse:
                        best_mse = mse
                        best_lambda0 = lambda0
                except Exception:
                    pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, fitted = self._fit_map(y_t.values, u_vals, order, best_lambda0)
            except Exception:
                _, fitted = self._fit_map(y_t.values, None, order, best_lambda0)

        self._order = order
        self._lambda0 = best_lambda0
        self._fitted = fitted
        return self

    def get_params_str(self) -> str:
        p, d, q = self._order
        return f"BAYESARIMA({p},{d},{q}) lambda0={self._lambda0:g}"
