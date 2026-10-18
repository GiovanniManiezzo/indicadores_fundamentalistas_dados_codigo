"""SETAR: Self-Exciting Threshold Autoregressive (Tong, 1990).

Dois regimes com dinâmica AR própria (mesma ordem p nos dois regimes, coeficientes
distintos — parcimônia deliberada dado N pequeno), separados por uma variável de
threshold self-exciting (y_{t-delay}, delay a determinar) e um valor de corte
também a determinar. Busca em grid por SSR mínima com trimming nas extremidades da
distribuição da variável de threshold (Andrews 1993 recomenda 15%; ver Hansen, B.E.
(2000), "Sample Splitting and Threshold Estimation", Econometrica 68(3), 575-603,
para a lógica de inferência de threshold via minimização de SSR).

Delay, threshold e ordem são selecionados 1x na janela inicial de treino (mesmo
padrão fixed_order/fixed_spec_ dos demais wrappers do projeto) e congelados via
`fixed_config` para todos os folds do walk-forward — nunca re-selecionados por fold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base_model import BaseModel


class SETARWrapper(BaseModel):
    """SETAR(2; p, p) self-exciting: dois regimes AR(p) separados por y_{t-delay} vs threshold."""

    def __init__(
        self,
        p_max: int = 2,
        delay_max: int = 2,
        trim: float = 0.15,
        mode: str = "raw",
        fixed_config: dict | None = None,
    ) -> None:
        self.p_max = p_max
        self.delay_max = delay_max
        self.trim = trim
        self.mode = mode
        self._fixed_config = fixed_config

        self.delay_: int | None = None
        self.threshold_: float | None = None
        self.order_: tuple[int, int] | None = None
        self.fixed_config_: dict | None = None

        self._coef: dict[str, np.ndarray] = {}
        self._y_original: pd.Series | None = None
        self._last_values: np.ndarray | None = None

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
        raise ValueError(f"Modo desconhecido: {mode!r}")

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_regime(yv: np.ndarray, p: int, idx: np.ndarray) -> tuple[np.ndarray | None, float]:
        """OLS (lstsq) do regime sobre os índices `idx`. Retorna (coef, SSR)."""
        if len(idx) < p + 2:
            return None, np.inf
        X = np.array([[1.0] + yv[t - p:t][::-1].tolist() for t in idx], dtype=float)
        Y = np.array([yv[t] for t in idx], dtype=float)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ coef
        return coef, float(resid @ resid)

    def _search(self, yv: np.ndarray) -> tuple[int, float, int] | None:
        """Grid search (delay, threshold, p) por SSR total mínima. Retorna None se inviável."""
        n = len(yv)
        best: tuple[float, int, float, int] | None = None  # (ssr, delay, threshold, p)

        for delay in range(1, self.delay_max + 1):
            for p in range(1, self.p_max + 1):
                t0 = max(p, delay)
                t_idx = np.arange(t0, n)
                if len(t_idx) < 2 * (p + 2):
                    continue
                z = yv[t_idx - delay]
                order_z = np.sort(np.unique(z))
                lo_i = int(np.ceil(self.trim * len(order_z)))
                hi_i = int(np.floor((1 - self.trim) * len(order_z)))
                candidates = order_z[lo_i:hi_i] if hi_i > lo_i else order_z
                for c in candidates:
                    idx_lo = t_idx[z <= c]
                    idx_hi = t_idx[z > c]
                    if len(idx_lo) < p + 2 or len(idx_hi) < p + 2:
                        continue
                    _, ssr1 = self._fit_regime(yv, p, idx_lo)
                    _, ssr2 = self._fit_regime(yv, p, idx_hi)
                    ssr = ssr1 + ssr2
                    if best is None or ssr < best[0]:
                        best = (ssr, delay, float(c), p)

        if best is None:
            return None
        _, delay, threshold, p = best
        return delay, threshold, p

    # ------------------------------------------------------------------
    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "SETARWrapper":
        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values.astype(float)
        n = len(yv)

        if self._fixed_config is not None:
            self.delay_ = int(self._fixed_config["delay"])
            self.threshold_ = float(self._fixed_config["threshold"])
            self.order_ = tuple(self._fixed_config["order"])
        else:
            found = self._search(yv)
            if found is None:
                raise RuntimeError(f"SETAR: nenhuma configuração viável (n={n})")
            delay, threshold, p = found
            self.delay_ = delay
            self.threshold_ = threshold
            self.order_ = (p, p)

        p1, p2 = self.order_
        p = max(p1, p2)
        hist_len = max(p, self.delay_)
        t0 = max(p, self.delay_)
        t_idx = np.arange(t0, n)
        if len(t_idx) < p1 + p2 + 4:
            raise RuntimeError(f"SETAR: amostra insuficiente no fold atual (n={n})")

        z = yv[t_idx - self.delay_]
        idx_lo = t_idx[z <= self.threshold_]
        idx_hi = t_idx[z > self.threshold_]
        coef_lo, _ = self._fit_regime(yv, p1, idx_lo)
        coef_hi, _ = self._fit_regime(yv, p2, idx_hi)
        if coef_lo is None or coef_hi is None:
            raise RuntimeError("SETAR: regime com observações insuficientes no fold atual")

        self._coef = {"lo": coef_lo, "hi": coef_hi}
        self._last_values = yv[-hist_len:].copy()
        self.fixed_config_ = {"delay": self.delay_, "threshold": self.threshold_, "order": self.order_}
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        p1, p2 = self.order_
        hist = list(self._last_values)
        preds: list[float] = []

        for _ in range(steps):
            z = hist[-self.delay_]
            regime = "lo" if z <= self.threshold_ else "hi"
            p = p1 if regime == "lo" else p2
            coef = self._coef[regime]
            lags = np.array(hist[-p:][::-1], dtype=float) if p > 0 else np.array([])
            x = np.concatenate([[1.0], lags])
            yhat = float(x @ coef)
            preds.append(yhat)
            hist.append(yhat)

        raw = np.array(preds)
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
        raise ValueError(f"Modo desconhecido: {mode!r}")

    def get_params_str(self) -> str:
        p1, p2 = self.order_
        return f"SETAR(delay={self.delay_},thr={self.threshold_:.4g},p={p1})"
