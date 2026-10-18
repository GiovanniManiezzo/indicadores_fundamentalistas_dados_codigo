"""Base interna para modelos SARIMAX (ARMAX, ARIMAX, OE, BJ).

Não exposto diretamente; subclasses adicionam restrições por modelo.
Convenções idênticas ao ARIMAWrapper:
  - `d` fixado pelos testes de raiz unitária (sobre a série transformada).
  - `mode` controla pré-processamento; inverse_transform no predict().
  - `fixed_order` congela (p,d,q) para o walk-forward CV.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseModel


class _SARIMAXBase(BaseModel):
    """Base SARIMAX com exógenas opcionais.

    Parâmetros
    ----------
    d          : ordem de integração (determinada por unit_root_tests).
    p_max      : máximo p no grid AIC.
    q_max      : máximo q no grid AIC.
    mode       : transformação antes do ajuste ('raw','log','diff','log+diff').
    fixed_order: (p,d,q) pré-selecionado — evita re-grid no walk-forward CV.
    p_forced   : força p fixo no grid (ex.: BJ usa p=0).
    q_forced   : força q fixo no grid (ex.: OE usa q=0).
    """

    def __init__(
        self,
        d: int = 1,
        p_max: int = 3,
        q_max: int = 3,
        mode: str = "raw",
        fixed_order: tuple[int, int, int] | None = None,
        p_forced: int | None = None,
        q_forced: int | None = None,
        boxcox_lambda: float | None = None,
    ) -> None:
        self.d = d
        self.p_max = p_max
        self.q_max = q_max
        self.mode = mode
        self._fixed_order = fixed_order
        self._p_forced = p_forced
        self._q_forced = q_forced
        self.boxcox_lambda = boxcox_lambda
        self._fitted = None
        self._order: tuple[int, int, int] = (
            fixed_order if fixed_order is not None else (1, d, 0)
        )
        self._y_original: pd.Series | None = None

    # ------------------------------------------------------------------
    # Transformação interna
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "_SARIMAXBase":
        self._y_original = y.copy()
        y_t = self._transform(y)

        # Alinhar exógenas com y transformado (se diff, descarta 1ª linha de y)
        u_vals: np.ndarray | None = None
        if u is not None:
            u_a = u.iloc[-len(y_t):].ffill().bfill().fillna(0.0)
            u_vals = u_a.values.astype(float)

        if self._fixed_order is not None:
            p, _, q = self._fixed_order
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._fitted = SARIMAX(
                        y_t.values, exog=u_vals, order=(p, self.d, q)
                    ).fit(disp=False)
                except Exception:
                    # Fallback: sem exógenas
                    self._fitted = SARIMAX(
                        y_t.values, order=(p, self.d, q)
                    ).fit(disp=False)
            return self

        p_range = (
            [self._p_forced] if self._p_forced is not None
            else list(range(self.p_max + 1))
        )
        q_range = (
            [self._q_forced] if self._q_forced is not None
            else list(range(self.q_max + 1))
        )

        best_aic = np.inf
        for p in p_range:
            for q in q_range:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(
                            y_t.values, exog=u_vals, order=(p, self.d, q)
                        ).fit(disp=False)
                    if np.isfinite(m.aic) and m.aic < best_aic:
                        best_aic = m.aic
                        self._order = (p, self.d, q)
                        self._fitted = m
                except Exception:
                    pass

        if self._fitted is None:
            raise RuntimeError(
                f"{self.__class__.__name__} falhou para todos (p,q) "
                f"[mode={self.mode}, d={self.d}]"
            )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        u_fut_vals: np.ndarray | None = None
        if u_future is not None:
            u_f = u_future.ffill().bfill().fillna(0.0)
            u_fut_vals = u_f.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = np.asarray(
                    self._fitted.forecast(steps=steps, exog=u_fut_vals),
                    dtype=float,
                )
            except Exception:
                raw = np.asarray(
                    self._fitted.forecast(steps=steps),
                    dtype=float,
                )

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
        p, d, q = self._order
        return f"{self.__class__.__name__}({p},{d},{q})"

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order
