"""Forecast combination — Bates & Granger (1969); Petropoulos et al. (2022) §2.6.

Rounds 12-15 exhausted the univariate structural/smoothing search (ETS, Theta, damped-trend
SS/MSOE and SSOE) without a single-model improvement on 11 of 12 combos (`science_log.md`,
Round 15 "Próximo"). Petropoulos et al. (2022, §2.6) note the "forecast combination puzzle":
a simple average of a handful of individually reasonable forecasts tends to beat picking the
single best one in-sample, because it diversifies away model-specific misspecification risk
without adding estimated parameters — attractive precisely at N=64 where a genuinely new model
class risks overfitting (`protocolo metodológico` §12's LSTM caveat makes the same point from the other
direction).

This wrapper combines a **fixed** pool of five sub-models — ARIMA, ETS, THETA, SS, DAMPEDTREND —
identical across all 12 combos. The pool is not curated per combo (e.g. "this combo's top-3
candidates") deliberately: choosing the pool by looking at which models already do well on a
given combo would be a data-mined design decision disguised as a modelling choice, at a sample
size (N=64) too small to validate that choice independently. The five members are exactly the
parsimonious univariate family already validated panel-wide in Rounds 02/12/14 (`protocolo metodológico`
§§17-20) — no member has more than 4 estimated parameters, matching Petropoulos et al.'s
recommendation (§2.3.1/§2.3.4) to prefer small, well-understood statistical benchmarks in a
combination over hand-picked or complex ones.

Three fixed combination schemes (never chosen adaptively at predict time — the scheme is part of
the model's identity, exactly like `nl_type` is for the Wiener/Hammerstein wrappers):
  - MEAN:   unweighted average — Bates & Granger (1969); the "puzzle" baseline, hardest to beat.
  - MEDIAN: robust to one sub-model's degenerate forecast on a given fold.
  - INVMSE: weights ∝ 1/MSE, MSE estimated once via an inner walk-forward pass over the initial
            `MIN_TRAIN` window (see `_run_all_models_for_combo`'s `_combination()` in `main.py`)
            and then frozen for every subsequent fold — the same anti-leakage pattern as
            `fixed_order`/`fixed_spec_` elsewhere in this codebase.

Univariate only — no exogenous support, matching every member of the pool (ETS/Theta/DampedTrend
are univariate by construction; SS and ARIMA are used here in their univariate form too, so the
combination is comparable across all 12 combos including the ones with no validated exogenous
set). `u` is accepted for `BaseModel` interface compatibility and ignored.
"""
from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from .base_model import BaseModel

SCHEMES: tuple[str, ...] = ("MEAN", "MEDIAN", "INVMSE")


class CombinationWrapper(BaseModel):
    """Combines a fixed pool of sub-model forecasts under a frozen scheme."""

    def __init__(
        self,
        sub_fns: dict[str, Callable[[pd.Series, "pd.DataFrame | None"], BaseModel]],
        scheme: str = "MEAN",
        weights: dict[str, float] | None = None,
    ) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"scheme desconhecido: {scheme!r} (esperado um de {SCHEMES})")
        self.sub_fns = sub_fns
        self.scheme = scheme
        self.weights = weights or {}
        self._fitted_subs: dict[str, BaseModel] = {}
        self._active: list[str] = []
        self._params_str = ""

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "CombinationWrapper":
        self._fitted_subs = {}
        for name, fn in self.sub_fns.items():
            try:
                self._fitted_subs[name] = fn(y, None)
            except Exception as exc:
                warnings.warn(f"CombinationWrapper: sub-modelo {name} falhou no fit: {exc}")
        self._active = list(self._fitted_subs.keys())
        if not self._active:
            raise RuntimeError("CombinationWrapper: todos os sub-modelos falharam no fit")
        self._params_str = f"COMBINATION-{self.scheme}[{'+'.join(self._active)}]"
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        preds: dict[str, np.ndarray] = {}
        for name in self._active:
            try:
                raw = self._fitted_subs[name].predict(steps=steps, u_future=None)
                preds[name] = np.asarray(raw, dtype=float)
            except Exception as exc:
                warnings.warn(f"CombinationWrapper: sub-modelo {name} falhou no predict: {exc}")
        if not preds:
            raise RuntimeError("CombinationWrapper: nenhum sub-modelo produziu previsão")

        names = list(preds.keys())
        stacked = np.vstack([preds[n] for n in names])  # (n_active, steps)

        if self.scheme == "MEAN":
            return stacked.mean(axis=0)
        if self.scheme == "MEDIAN":
            return np.median(stacked, axis=0)
        # INVMSE
        w = np.array([self.weights.get(n, 0.0) for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names))  # pesos indisponíveis (fold inicial) → equal-weight fallback
        w = w / w.sum()
        return (stacked * w[:, None]).sum(axis=0)

    def get_params_str(self) -> str:
        return self._params_str
