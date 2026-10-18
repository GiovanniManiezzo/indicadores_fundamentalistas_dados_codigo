"""Estimação recursiva dos coeficientes no walk-forward — RPEM (Round 35; protocolo metodológico §28).

`evaluation.walk_forward_cv` reajusta o modelo do zero em cada origem (estimação em lote). Este
módulo troca só o estimador: a ordem/spec, o pré-processamento, as exógenas e os pesos de
combinação continuam os do campeão, congelados na janela inicial `MIN_TRAIN`; os coeficientes
θ são estimados em lote **uma única vez**, nessa janela, e daí em diante atualizados a cada nova
observação pelo Recursive Prediction Error Method de Gauss-Newton com fator de esquecimento λ
(Ljung & Söderström 1983; Ljung 1999, §11.2-11.4):

    ε_t = y_t − ŷ_t(θ_{t−1})           ψ_t = ∂ŷ_t/∂θ |θ_{t−1}
    R_t = λ R_{t−1} + ψ_t ψ_tᵀ          θ_t = θ_{t−1} + R_t⁺ ψ_t ε_t

Para um preditor linear nos parâmetros (ARX) isto é exatamente o RLS com esquecimento. O
gradiente ψ_t é obtido por diferenças finitas centrais do próprio preditor de um passo de cada
família, avaliado em θ_{t−1} sobre todo o histórico (gradiente exato, não a aproximação
recursiva de Ljung eq. 11.47 — viável a N=64). θ vive nas coordenadas naturais de cada modelo;
um passo que sai da região admissível D_M é reduzido à metade até voltar a ela e só é rejeitado
(θ_t = θ_{t−1}) se nem 2^−10 do passo servir — a projeção de Ljung & Söderström (1983) e Ljung
(1999, §11.7). R_t⁺ é calculada com escalonamento diagonal, para que a pseudo-inversa não
dependa das unidades de θ.

Cada família entra por um adaptador que expõe `theta0`, `path(θ, y, u)` (previsões de um passo
dentro da amostra, na escala do modelo) e `forecast(θ, y, u, u_next)` (previsão h=1 na escala
original). O parâmetro de escala (primeira variância) fica fixo: a média da previsão não depende
dele e incluí-lo deixaria R_t singular. Em θ_0 todo adaptador reproduz o preditor do lote, então
a 1ª previsão fora da amostra é idêntica à do walk-forward em lote (checagem de implementação).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg
from src.evaluation import fit_percent, mae, mape, rmse, theil_u2
from src.models._sarimax_base import _SARIMAXBase
from src.models.arima_wrapper import ARIMAWrapper
from src.models.bayesarima_wrapper import BayesArimaWrapper
from src.models.combination_wrapper import CombinationWrapper
from src.models.dampedtrend_wrapper import DampedTrendWrapper
from src.models.ets_wrapper import ETSWrapper, _ETS_SPECS
from src.models.hammerstein_wrapper import HammersteinWrapper, _merge_exog
from src.models.ss_wrapper import SSWrapper
from src.models.theta_wrapper import ThetaWrapper
from src.models.wiener_wrapper import WienerWrapper
from src.models._nonlinear_blocks import PiecewiseLinearBlock


# Constantes de suavização (Theta/ETS) são estimadas nas coordenadas naturais e projetadas na
# caixa admissível. Em logit o gradiente some na fronteira (α ≈ 0 é estimativa comum em lote) e o
# primeiro passo de Gauss-Newton salta para o outro extremo, onde também satura.
_ALPHA_MIN = 1e-4


class _FixedForecast:
    """Substitui `wrapper._fitted` para reaproveitar o inverse_transform de `wrapper.predict`."""

    def __init__(self, raw: float) -> None:
        self._raw = raw

    def forecast(self, steps: int = 1, exog=None) -> np.ndarray:
        return np.full(steps, self._raw, dtype=float)


# ---------------------------------------------------------------------------
# Adaptadores por família
# ---------------------------------------------------------------------------

class _StatespaceAdapter:
    """ARIMA, ARIMAX/_SARIMAXBase, BayesARIMA, SS (UnobservedComponents) e DampedTrend.

    θ vive nas coordenadas naturais (parâmetros restritos do modelo). No espaço irrestrito do
    statsmodels a transformação satura na fronteira (MA ≈ −1, variância ≈ 0): o gradiente some,
    um passo de Gauss-Newton salta para perto da fronteira e ali trava. A região admissível D_M
    (estacionariedade, invertibilidade, variâncias ≥ 0, limites de φ) é imposta por rejeição
    (`admissible`, Ljung 1999 §11.7): θ é admissível se for ponto fixo da ida e volta
    `transform_params(untransform_params(θ))` do próprio modelo.
    """

    def __init__(self, wrapper) -> None:
        self.w = wrapper
        res = wrapper._fitted
        self.mod0 = res.model
        self.p_con = np.asarray(res.params, dtype=float).copy()
        names = list(self.mod0.param_names)
        scale = next((i for i, n in enumerate(names) if n.startswith("sigma2")), None)
        self.free = [i for i in range(len(names)) if i != scale]
        self.names = [names[i] for i in self.free]
        self.theta0 = self.p_con[self.free].copy()
        self.use_exog = getattr(self.mod0, "k_exog", 0) > 0

    def _full(self, theta: np.ndarray) -> np.ndarray:
        p = self.p_con.copy()
        p[self.free] = theta
        return p

    def admissible(self, theta: np.ndarray) -> bool:
        p = self._full(theta)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rt = np.asarray(self.mod0.transform_params(self.mod0.untransform_params(p)))
        except Exception:
            return False
        return bool(np.all(np.isfinite(rt)) and np.allclose(rt, p, rtol=1e-6, atol=1e-10))

    def _filter(self, theta: np.ndarray, y_hist: pd.Series, u_hist: pd.DataFrame | None):
        y_t = self.w._transform(y_hist)
        exog = None
        if self.use_exog:
            exog = u_hist.iloc[-len(y_t):].ffill().bfill().fillna(0.0).values.astype(float)
        endog = np.asarray(y_t, dtype=float)
        try:
            mod = self.mod0.clone(endog, exog=exog) if exog is not None else self.mod0.clone(endog)
        except NotImplementedError:
            mod = type(self.mod0)(endog)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mod, mod.filter(self._full(theta), transformed=True)

    def path(self, theta, y_hist, u_hist):
        mod, r = self._filter(theta, y_hist, u_hist)
        pred = np.asarray(r.fittedvalues, dtype=float)
        target = np.asarray(mod.endog, dtype=float).ravel()
        return pred, target, int(r.loglikelihood_burn)

    def forecast(self, theta, y_hist, u_hist, u_next) -> float:
        _, r = self._filter(theta, y_hist, u_hist)
        self.w._fitted = r
        self.w._y_original = y_hist
        return float(np.asarray(self.w.predict(steps=1, u_future=u_next), dtype=float)[0])

    def prior_info(self) -> np.ndarray | None:
        """BayesARIMA: precisão do prior Minnesota (σ²·Λ sobre AR/MA), nas coordenadas de θ.

        Entra só em R_0, como no RLS bayesiano (θ_0 = MAP, P_0 = covariância a posteriori).
        """
        if not isinstance(self.w, BayesArimaWrapper):
            return None
        mod = self.mod0
        k_ar, k_ma = mod.k_ar, mod.k_ma
        ar_start = mod.k_trend + mod.k_exog
        idx = list(range(ar_start, ar_start + k_ar + k_ma))
        if not idx:
            return None
        lags = [i + 1 for i in range(k_ar)] + [i + 1 for i in range(k_ma)]
        info = np.zeros((len(self.theta0), len(self.theta0)))
        for i, lag in zip(idx, lags):
            j = self.free.index(i)
            info[j, j] = self.w.lambda0 * lag ** 2
        sigma2 = float(np.asarray(self.w._fitted.params)[-1])
        return sigma2 * info


class _ThetaAdapter:
    """Theta clássico (statsmodels ThetaModel, theta=2): SES com nível inicial y_0 + meia
    deriva b0.

    No lote os dois parâmetros vêm de critérios diferentes: α do SES (erro de previsão) e b0 do
    OLS de y sobre (1, t). A versão recursiva preserva cada critério: α entra em θ (RPEM) e b0 é
    atualizado por RLS com o mesmo esquecimento λ sobre a regressão de tendência (`side_update`).
    """

    def __init__(self, wrapper: ThetaWrapper, lam: float, y0: pd.Series) -> None:
        self.w = wrapper
        r = wrapper._fitted
        self.lam = lam
        self.names = ["alpha"]
        self.theta0 = np.array([float(r.params["alpha"])])
        self.lo, self.hi = np.array([_ALPHA_MIN]), np.array([1.0])
        yv = np.asarray(wrapper._transform(y0), dtype=float)
        X = np.column_stack([np.ones(len(yv)), np.arange(len(yv), dtype=float)])
        self.ab = np.linalg.lstsq(X, yv, rcond=None)[0]   # = OLS do ThetaModel (b0 idêntico)
        w = lam ** (len(yv) - 1 - np.arange(len(yv)))
        self.Rab = (X.T * w) @ X

    def project(self, theta):
        return np.clip(theta, self.lo, self.hi)

    @property
    def b0(self) -> float:
        return float(self.ab[1])

    def reset(self, rho: float) -> None:
        self.Rab = rho * self.Rab

    def side_update(self, y_hist: pd.Series) -> None:
        yv = np.asarray(self.w._transform(y_hist), dtype=float)
        x = np.array([1.0, float(len(yv) - 1)])
        self.Rab = self.lam * self.Rab + np.outer(x, x)
        self.ab = self.ab + np.linalg.solve(self.Rab, x) * (yv[-1] - x @ self.ab)

    def _run(self, theta, yv):
        alpha = float(np.clip(theta[0], _ALPHA_MIN, 1.0))
        b0 = self.b0
        n = len(yv)
        preds = np.empty(n + 1)
        level = yv[0]
        preds[0] = yv[0]
        for k in range(1, n + 1):
            # 1/α − (1−α)^k/α = (1 − (1−α)^k)/α, estável perto de α = 0 (limite k)
            drift = 0.5 * b0 * (-np.expm1(k * np.log1p(-alpha)) / alpha if alpha < 1.0 else 1.0)
            preds[k] = level + drift
            if k < n:
                level = alpha * yv[k] + (1.0 - alpha) * level
        return preds

    def path(self, theta, y_hist, u_hist):
        yv = np.asarray(self.w._transform(y_hist), dtype=float)
        return self._run(theta, yv)[:-1], yv, 1

    def forecast(self, theta, y_hist, u_hist, u_next) -> float:
        yv = np.asarray(self.w._transform(y_hist), dtype=float)
        self.w._fitted = _FixedForecast(float(self._run(theta, yv)[-1]))
        self.w._y_original = y_hist
        return float(self.w.predict(steps=1)[0])


class _ETSAdapter:
    """ETS(A,·,·) do statsmodels: θ = constantes de suavização nas coordenadas naturais,
    projetadas nos limites do ETSModel (α∈(0,1), β∈(0,α], φ∈[0,8;0,98]); estados iniciais
    fixos no lote."""

    _PHI_LO, _PHI_HI = 0.8, 0.98   # limites padrão do ETSModel

    def __init__(self, wrapper: ETSWrapper) -> None:
        self.w = wrapper
        r = wrapper._fitted
        self.spec = _ETS_SPECS[wrapper.fixed_spec_]
        self.p0 = np.asarray(r.params, dtype=float).copy()
        names = list(r.model.param_names)
        self.i_a = names.index("smoothing_level")
        self.i_b = names.index("smoothing_trend") if "smoothing_trend" in names else None
        self.i_p = names.index("damping_trend") if "damping_trend" in names else None
        self.idx = [i for i in (self.i_a, self.i_b, self.i_p) if i is not None]
        self.names = [names[i] for i in self.idx]
        self.theta0 = self.p0[self.idx].copy()
        bounds = {self.i_a: (1e-4, 1 - 1e-4), self.i_b: (1e-4, 1 - 1e-4),
                  self.i_p: (self._PHI_LO, self._PHI_HI)}
        self.lo = np.array([bounds[i][0] for i in self.idx])
        self.hi = np.array([bounds[i][1] for i in self.idx])

    def project(self, theta):
        th = np.clip(theta, self.lo, self.hi)
        if self.i_b is not None:          # β ≤ α (limite do ETSModel)
            th[1] = min(th[1], th[0])
        return th

    def _params(self, theta):
        p = self.p0.copy()
        p[self.idx] = theta
        return p

    def _smooth(self, theta, y_hist):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        yv = np.asarray(self.w._transform(y_hist), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = ETSModel(yv, error="add", initialization_method="estimated", **self.spec)
            return yv, mod.smooth(self._params(theta))

    def path(self, theta, y_hist, u_hist):
        yv, r = self._smooth(theta, y_hist)
        return np.asarray(r.fittedvalues, dtype=float), yv, 1

    def forecast(self, theta, y_hist, u_hist, u_next) -> float:
        _, r = self._smooth(theta, y_hist)
        self.w._fitted = r
        self.w._y_original = y_hist
        return float(self.w.predict(steps=1)[0])


def _piecewise_check(block) -> None:
    if not isinstance(block, PiecewiseLinearBlock):
        raise NotImplementedError(
            f"RPEM block-oriented: só o bloco piecewise está implementado (recebido {block.name()})")


class _HammersteinAdapter:
    """Hammerstein f(y_lags) → SARIMAX. θ = (θ do bloco linear, a2, b) do bloco piecewise.

    τ (estrutura, escolhido por grade) e a1 (normalização do ganho — f e os coeficientes do
    bloco linear só são identificáveis a menos de uma escala) ficam fixos no valor do lote.
    """

    def __init__(self, wrapper: HammersteinWrapper) -> None:
        _piecewise_check(wrapper._f)
        self.w = wrapper
        self.lin = _StatespaceAdapter(wrapper._linear)
        self.k_lin = len(self.lin.theta0)
        f = wrapper._f
        self.names = self.lin.names + ["f.a2", "f.b"]
        self.theta0 = np.concatenate([self.lin.theta0, [f._a2, f._b]])

    def _f(self, theta) -> PiecewiseLinearBlock:
        f0 = self.w._f
        f = PiecewiseLinearBlock()
        f._tau, f._a1, f._a2, f._b, f._fitted = f0._tau, f0._a1, theta[-2], theta[-1], True
        return f

    def _z(self, theta, y_hist, u_hist):
        return _merge_exog(self.w._make_z_exog(y_hist, self._f(theta)), u_hist)

    def admissible(self, theta) -> bool:
        return self.lin.admissible(theta[:self.k_lin])

    def path(self, theta, y_hist, u_hist):
        return self.lin.path(theta[:self.k_lin], y_hist, self._z(theta, y_hist, u_hist))

    def forecast(self, theta, y_hist, u_hist, u_next) -> float:
        f = self._f(theta)
        last_z = f.transform(np.asarray(y_hist, dtype=float)[-self.w.n_lags:])
        z_future = pd.DataFrame({f"z_lag{i + 1}": [float(last_z[-(i + 1)])]
                                 for i in range(self.w.n_lags)})
        u_fut = _merge_exog(z_future, u_next)
        return self.lin.forecast(theta[:self.k_lin], y_hist, self._z(theta, y_hist, u_hist), u_fut)


class _WienerAdapter:
    """Wiener SARIMAX → g(w), com o preditor exato do lote: ŷ = g(ARIMA_θ(g_alvo⁻¹(y))).

    O ALS em lote termina com o bloco linear ajustado sobre o inverso do bloco da iteração
    anterior (`wrapper._g_target`, None = identidade), não do `g` final. Esse bloco-alvo é
    estrutura congelada, como τ; θ = (θ do bloco linear, a2, b do `g` de saída), com τ e a1 de
    `g` fixos (normalização do ganho, como no Hammerstein).
    """

    def __init__(self, wrapper: WienerWrapper) -> None:
        _piecewise_check(wrapper._g)
        self.w = wrapper
        self.g_tgt = wrapper._g_target
        self.lin = _StatespaceAdapter(wrapper._linear)
        self.k_lin = len(self.lin.theta0)
        g = wrapper._g
        self.names = self.lin.names + ["g.a2", "g.b"]
        self.theta0 = np.concatenate([self.lin.theta0, [g._a2, g._b]])

    def _g(self, theta) -> PiecewiseLinearBlock:
        g0 = self.w._g
        g = PiecewiseLinearBlock()
        g._tau, g._a1, g._a2, g._b, g._fitted = g0._tau, g0._a1, theta[-2], theta[-1], True
        return g

    def _target(self, y_hist: pd.Series) -> pd.Series:
        """Mesma regra do ALS: g_alvo⁻¹(y), com y onde a inversa não é finita."""
        if self.g_tgt is None:
            return y_hist
        y_arr = np.asarray(y_hist, dtype=float)
        yt = self.g_tgt.inverse_transform(y_arr)
        bad = ~np.isfinite(yt)
        yt[bad] = y_arr[bad]
        return pd.Series(yt, index=y_hist.index, name=y_hist.name)

    def admissible(self, theta) -> bool:
        return self.lin.admissible(theta[:self.k_lin])

    def path(self, theta, y_hist, u_hist):
        y_tgt = self._target(y_hist)
        pred_t, _, burn = self.lin.path(theta[:self.k_lin], y_tgt, u_hist)
        w_level = _insample_inverse(self.lin.w, pred_t, y_tgt)
        y_arr = np.asarray(y_hist, dtype=float)[-len(w_level):]
        return self._g(theta).transform(w_level), y_arr, burn

    def forecast(self, theta, y_hist, u_hist, u_next) -> float:
        w = self.lin.forecast(theta[:self.k_lin], self._target(y_hist), u_hist, u_next)
        return float(self._g(theta).transform(np.array([w]))[0])


def _insample_inverse(wrapper, pred_t: np.ndarray, y_level: pd.Series) -> np.ndarray:
    """Leva previsões de um passo na escala transformada de volta ao nível de `y_level`."""
    mode = wrapper.mode
    yl = np.asarray(y_level, dtype=float)
    if mode == "raw":
        return pred_t
    if mode == "log":
        return np.exp(pred_t)
    if mode == "diff":
        return yl[:-1][-len(pred_t):] + pred_t
    if mode == "log+diff":
        return np.exp(np.log(yl[:-1][-len(pred_t):]) + pred_t)
    raise NotImplementedError(f"Wiener recursivo: modo {mode!r} não suportado")


def make_adapter(wrapper, lam: float, y0: pd.Series):
    if isinstance(wrapper, HammersteinWrapper):
        return _HammersteinAdapter(wrapper)
    if isinstance(wrapper, WienerWrapper):
        return _WienerAdapter(wrapper)
    if isinstance(wrapper, ThetaWrapper):
        return _ThetaAdapter(wrapper, lam, y0)
    if isinstance(wrapper, ETSWrapper):
        return _ETSAdapter(wrapper)
    if isinstance(wrapper, (ARIMAWrapper, _SARIMAXBase, SSWrapper, DampedTrendWrapper)):
        return _StatespaceAdapter(wrapper)
    raise NotImplementedError(f"RPEM: sem adaptador para {type(wrapper).__name__}")


# ---------------------------------------------------------------------------
# Laço RPEM
# ---------------------------------------------------------------------------

def _scaled_pinv(R: np.ndarray) -> np.ndarray:
    """R⁺ com escalonamento diagonal (Jacobi): D R D, D = diag(R)^−½.

    Coeficientes de exógenas em R$ (~10⁶) convivem com AR/MA (~10⁻²) em θ; sem escalar, o rcond
    relativo da pseudo-inversa descarta direções bem identificadas só por causa da unidade. Uma
    direção com diagonal nula (regressor ainda não excitado, ex.: dummy de nível antes do
    evento) fica fora do passo.
    """
    dg = np.diag(R).copy()
    ok = dg > 0
    d = np.zeros_like(dg)
    d[ok] = 1.0 / np.sqrt(dg[ok])
    Dm = np.diag(d)
    return Dm @ np.linalg.pinv(Dm @ R @ Dm, rcond=cfg.RPEM_PINV_RCOND) @ Dm


class RPEMState:
    """θ_t e R_t de um modelo; inicializados pelo ajuste em lote na janela inicial."""

    def __init__(self, wrapper, y0: pd.Series, u0: pd.DataFrame | None, lam: float) -> None:
        self.ad = make_adapter(wrapper, lam, y0)
        self.lam = lam
        self.theta = self.ad.theta0.copy()
        self.n_rejected = 0
        self.n_shrunk = 0
        self.n_resets = 0
        self.n_updates = 0
        pred, target, burn = self.ad.path(self.theta, y0, u0)
        psi = self._jacobian(self.theta, y0, u0, last_only=False)
        n = len(pred)
        keep = np.arange(min(burn, n - 1), n)
        w = lam ** (n - 1 - keep)
        R = (psi[keep].T * w) @ psi[keep]
        prior = self.ad.prior_info() if hasattr(self.ad, "prior_info") else None
        if prior is not None:
            R = R + prior
        self.R = R

    def _jacobian(self, theta, y, u, last_only: bool) -> np.ndarray:
        if len(theta) == 0:
            n = 1 if last_only else len(self.ad.path(theta, y, u)[0])
            return np.zeros((n, 0))
        p0 = None
        cols = []
        for j in range(len(theta)):
            h = cfg.RPEM_FD_REL_STEP * max(abs(theta[j]), 1.0)
            pts = []
            for sgn in (+1, -1):
                t = theta.copy()
                t[j] += sgn * h
                try:
                    pt = self.ad.path(t, y, u)[0]
                    if not np.all(np.isfinite(pt[-1:] if last_only else pt)):
                        raise FloatingPointError("previsão não finita")
                    pts.append((sgn * h, pt))
                except Exception:
                    # Na fronteira de D_M o ponto θ±h pode ser inviável (exceção ou NaN, ex.: ETS
                    # com α = 0): diferença unilateral.
                    if p0 is None:
                        p0 = self.ad.path(theta, y, u)[0]
                    pts.append((0.0, p0))
            (hp, pp), (hm, pm) = pts
            d = (pp - pm) / (hp - hm)
            cols.append(d[-1:] if last_only else d)
        return np.column_stack(cols)

    def forecast(self, y_hist, u_hist, u_next) -> float:
        return self.ad.forecast(self.theta, y_hist, u_hist, u_next)

    def reset(self, rho: float) -> None:
        """Reinicialização da covariância (Goodwin & Sin 1984): R ← ρ·R. Com menos informação
        acumulada, o próximo passo de Gauss-Newton dá mais peso à observação nova."""
        self.R = rho * self.R
        self.n_resets += 1
        if hasattr(self.ad, "reset"):
            self.ad.reset(rho)

    def update(self, y_hist, u_hist) -> None:
        """Incorpora a última observação de `y_hist` (um passo de Gauss-Newton)."""
        if len(self.theta) == 0:      # ex.: nível local, cujo único parâmetro é a escala
            return
        pred, target, _ = self.ad.path(self.theta, y_hist, u_hist)
        eps = float(target[-1] - pred[-1])
        psi = self._jacobian(self.theta, y_hist, u_hist, last_only=True).ravel()
        if not (np.isfinite(eps) and np.all(np.isfinite(psi))):
            self.n_updates += 1          # observação inutilizável: R e θ ficam como estão
            self.n_rejected += 1
            if hasattr(self.ad, "side_update"):
                self.ad.side_update(y_hist)
            return
        self.R = self.lam * self.R + np.outer(psi, psi)
        step = _scaled_pinv(self.R) @ psi * eps
        self.n_updates += 1
        # Projeção em D_M (Ljung & Söderström 1983; Ljung 1999 §11.7): o passo é reduzido à
        # metade até cair na região admissível; só é rejeitado se nem μ = 2^−RPEM_MAX_HALVINGS
        # servir. Rejeitar o passo inteiro congelaria o parâmetro sempre que o Gauss-Newton
        # ultrapassasse a fronteira (MA ≈ −1, variância ≈ 0).
        mu, accepted = 1.0, False
        for _ in range(cfg.RPEM_MAX_HALVINGS + 1):
            cand = self.theta + mu * step
            if hasattr(self.ad, "project"):   # caixa das constantes de suavização
                cand = self.ad.project(cand)
            try:
                ok = (np.all(np.isfinite(cand))
                      and (not hasattr(self.ad, "admissible") or self.ad.admissible(cand))
                      and np.isfinite(self.ad.path(cand, y_hist, u_hist)[0][-1]))
            except Exception:
                ok = False
            if ok:
                self.theta, accepted = cand, True
                break
            mu *= 0.5
        if not accepted:
            self.n_rejected += 1
        elif mu < 1.0:
            self.n_shrunk += 1
        if hasattr(self.ad, "side_update"):
            self.ad.side_update(y_hist)


class _CombinationState:
    """COMBINATION: cada sub-modelo segue seu próprio RPEM; esquema e pesos INVMSE congelados."""

    def __init__(self, wrapper: CombinationWrapper, y0, u0, lam) -> None:
        self.w = wrapper
        self.subs: dict[str, RPEMState] = {}
        for name in wrapper._active:
            try:
                self.subs[name] = RPEMState(wrapper._fitted_subs[name], y0, None, lam)
            except Exception as exc:
                warnings.warn(f"RPEM COMBINATION: sub-modelo {name} fora da combinação: {exc}")

    @property
    def n_rejected(self) -> int:
        return sum(s.n_rejected for s in self.subs.values())

    @property
    def n_updates(self) -> int:
        return sum(s.n_updates for s in self.subs.values())

    @property
    def n_shrunk(self) -> int:
        return sum(s.n_shrunk for s in self.subs.values())

    def forecast(self, y_hist, u_hist, u_next) -> float:
        preds = {}
        for name, s in self.subs.items():
            try:
                p = s.forecast(y_hist, None, None)
                if np.isfinite(p):
                    preds[name] = p
            except Exception as exc:
                warnings.warn(f"RPEM COMBINATION: sub-modelo {name} falhou no predict: {exc}")
        names = list(preds)
        x = np.array([preds[n] for n in names])
        if self.w.scheme == "MEAN":
            return float(x.mean())
        if self.w.scheme == "MEDIAN":
            return float(np.median(x))
        wts = np.array([self.w.weights.get(n, 0.0) for n in names], dtype=float)
        if wts.sum() <= 0:
            wts = np.ones(len(names))
        return float((x * wts / wts.sum()).sum())

    @property
    def n_resets(self) -> int:
        return max((s.n_resets for s in self.subs.values()), default=0)

    def reset(self, rho: float) -> None:
        for s in self.subs.values():
            s.reset(rho)

    def update(self, y_hist, u_hist) -> None:
        for s in self.subs.values():
            s.update(y_hist, None)


def make_state(wrapper, y0, u0, lam):
    if isinstance(wrapper, CombinationWrapper):
        return _CombinationState(wrapper, y0, u0, lam)
    return RPEMState(wrapper, y0, u0, lam)


def recursive_walk_forward(
    model_fn,
    y: pd.Series,
    u_exog: pd.DataFrame | None = None,
    min_train: int = 20,
    lam: float | None = None,
    reset_dates: "list[str] | None" = None,
    rho: float | None = None,
) -> dict:
    """Walk-forward h=1 com estimação recursiva — mesmo contrato de saída de `walk_forward_cv`.

    `model_fn` é a factory do campeão (ordem/spec congeladas): chamada uma única vez, na janela
    inicial, para dar θ_0. Em cada origem t a previsão usa θ_{t−1} (só dados até t−1); a
    observação y_t entra depois, na atualização — sem vazamento. `reset_dates` (Round 37):
    trimestres de evento regulatório conhecido; em cada um, R ← ρ·R depois da previsão do
    trimestre e antes de incorporá-lo, de modo que a previsão do próprio trimestre do evento não
    usa a informação de que ele ocorreu.
    """
    lam = cfg.RPEM_FORGETTING if lam is None else lam
    rho = cfg.RPEM_RESET_RHO if rho is None else rho
    resets = {pd.Timestamp(d) for d in (reset_dates or [])}
    y0 = y.iloc[:min_train]
    u0 = u_exog.iloc[:min_train] if u_exog is not None else None
    state = make_state(model_fn(y0, u0), y0, u0, lam)

    preds, actuals, rw_preds = [], [], []
    for t in range(min_train, len(y)):
        y_hist = y.iloc[:t]
        u_hist = u_exog.iloc[:t] if u_exog is not None else None
        u_next = u_exog.iloc[[t]] if u_exog is not None else None
        preds.append(state.forecast(y_hist, u_hist, u_next))
        actuals.append(float(y.iloc[t]))
        rw_preds.append(float(y_hist.iloc[-1]))
        if y.index[t] in resets:   # evento no trimestre t: reinicializa antes de incorporá-lo
            state.reset(rho)
        state.update(y.iloc[:t + 1], u_exog.iloc[:t + 1] if u_exog is not None else None)

    y_t, y_p, y_rw = np.array(actuals), np.array(preds), np.array(rw_preds)
    res = y_t - y_p
    return {
        "fit_percent": fit_percent(y_t, y_p),
        "rmse": rmse(y_t, y_p),
        "mae": mae(y_t, y_p),
        "mape": mape(y_t, y_p),
        "mse": float(np.mean(res ** 2)),
        "variance": float(np.var(res)),
        "theil_u2": theil_u2(y_t, y_p, y_rw),
        "residuals": res,
        "y_true": y_t,
        "y_pred": y_p,
        "y_rw": y_rw,
        "n_updates": int(state.n_updates),
        "n_rejected": int(state.n_rejected),
        "n_shrunk": int(state.n_shrunk),
        "n_resets": int(state.n_resets),
    }
