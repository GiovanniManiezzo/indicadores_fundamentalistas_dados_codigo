"""Blocos não-lineares estáticos para modelos Wiener, Hammerstein e HW.

Cada bloco implementa:
  fit(x, y)              — ajusta parâmetros minimizando ||y - f(x)||²
  transform(x)           — aplica f(x)
  inverse_transform(y)   — aplica f⁻¹(y) onde analítico, numérico caso contrário
  n_params               — número de parâmetros livres (propriedade)
  set_lambda(lam)        — define força de regularização L2

Blocos disponíveis:
  SigmoidBlock           — f(x) = 1 / (1 + exp(-k*(x - x0)))
  PiecewiseLinearBlock   — 1 breakpoint τ, 2 segmentos lineares contínuos
  BSplineBlock(n_knots)  — B-Spline com nós em quantis uniformes
  HalfSpaceLinearBlock   — inclinações distintas para x>0 e x≤0

Convenção: x e y são np.ndarray 1D na escala transformada (após mode).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import LSQUnivariateSpline


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _NonlinearBlock:
    """Interface base para blocos não-lineares."""

    _lam: float = 0.0  # regularização L2

    def set_lambda(self, lam: float) -> "_NonlinearBlock":
        self._lam = float(lam)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NonlinearBlock":
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# SigmoidBlock
# ---------------------------------------------------------------------------

class SigmoidBlock(_NonlinearBlock):
    """f(x) = 1 / (1 + exp(-k*(x - x0))).

    Parâmetros: k (inclinação), x0 (ponto de inflexão).
    Regularização L2 em k para evitar sigmoid degenerada (k→∞ = degrau).
    Inversa analítica via logit: f⁻¹(y) = x0 - log(1/y - 1) / k.
    """

    def __init__(self) -> None:
        self._k: float = 1.0
        self._x0: float = 0.0
        self._fitted: bool = False

    @staticmethod
    def _sigmoid(x: np.ndarray, k: float, x0: float) -> np.ndarray:
        # Clamp argument to avoid overflow in exp
        z = np.clip(-k * (x - x0), -500, 500)
        return 1.0 / (1.0 + np.exp(z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SigmoidBlock":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # Normalização interna para estabilidade numérica
        x_std = x.std() if x.std() > 0 else 1.0
        x_norm = (x - x.mean()) / x_std

        # Normalizar y para [0, 1] (sigmoid output range)
        y_min, y_range = y.min(), y.max() - y.min()
        y_norm = (y - y_min) / y_range if y_range > 0 else np.zeros_like(y)

        def loss(params: np.ndarray) -> float:
            k, x0_n = params
            pred = self._sigmoid(x_norm, k, x0_n)
            residuals = y_norm - pred
            reg = self._lam * k ** 2
            return float(np.mean(residuals ** 2) + reg)

        x0_init = np.median(x_norm)
        res = minimize(
            loss,
            x0=[1.0, x0_init],
            method="L-BFGS-B",
            bounds=[(0.001, 500.0), (-10.0, 10.0)],
        )

        k_norm, x0_norm = float(res.x[0]), float(res.x[1])
        # Desnormalizar: x_norm = (x - x_mean) / x_std → x0_real = x0_norm * x_std + x_mean
        self._k = k_norm / x_std
        self._x0 = x0_norm * x_std + x.mean()
        # Guardar escala de y para transform/inverse na escala original
        self._y_min = y_min
        self._y_range = y_range
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        s = self._sigmoid(x, self._k, self._x0)
        # Desnormalizar para escala de y
        return s * self._y_range + self._y_min

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inversa analítica via logit, com clamp para evitar log(0)."""
        y = np.asarray(y, dtype=float).ravel()
        y_norm = np.clip((y - self._y_min) / self._y_range if self._y_range > 0 else np.zeros_like(y), 1e-6, 1 - 1e-6)
        # f⁻¹(y_norm) = x0 - log(1/y_norm - 1) / k
        return self._x0 - np.log(1.0 / y_norm - 1.0) / self._k

    @property
    def n_params(self) -> int:
        return 2  # k, x0

    def name(self) -> str:
        return "sigmoid"


# ---------------------------------------------------------------------------
# PiecewiseLinearBlock
# ---------------------------------------------------------------------------

class PiecewiseLinearBlock(_NonlinearBlock):
    """1 breakpoint τ com 2 segmentos lineares contínuos.

    f(x) = a1*x + b                    se x ≤ τ
           a2*x + (b + (a1-a2)*τ)      se x > τ   (continuidade em τ)

    τ selecionado por grid search (Q10..Q90 da série x de treino).
    Slopes a1, a2 resolvidos por OLS por segmento dado τ fixo.
    Regularização L2 penaliza |a1|² + |a2|².
    Inversa analítica por segmento.
    """

    def __init__(self) -> None:
        self._tau: float = 0.0
        self._a1: float = 1.0
        self._a2: float = 1.0
        self._b: float = 0.0
        self._fitted: bool = False
        # τ congelado (Round 36, protocolo metodológico §28): None = grade Q10..Q90 a cada fit.
        self.fixed_tau: float | None = None

    def _fit_given_tau(
        self, x: np.ndarray, y: np.ndarray, tau: float
    ) -> tuple[float, float, float, float]:
        """OLS para a1, a2, b dado τ fixo, com restrição de continuidade."""
        mask1 = x <= tau
        mask2 = x > tau

        if mask1.sum() < 2 or mask2.sum() < 2:
            return 1.0, 1.0, 0.0, np.inf  # τ inválido → custo alto

        # Segmento 1: y = a1*x + b → [x, 1] @ [a1, b]
        X1 = np.column_stack([x[mask1], np.ones(mask1.sum())])
        y1 = y[mask1]

        # Segmento 2: y = a2*x + (b + (a1-a2)*τ) = a2*(x - τ) + (a1*τ + b)
        # Substitui b: y2 = a2*(x-τ) + offset  onde offset = a1*τ + b
        # Primeiro estima seg1, depois usa offset para seg2
        try:
            lam = self._lam
            # Tikhonov OLS: (X'X + lam*I) @ params = X'y
            reg1 = lam * np.eye(2)
            reg1[1, 1] = 0.0  # não regulariza bias
            sol1 = np.linalg.lstsq(X1.T @ X1 + reg1, X1.T @ y1, rcond=None)[0]
            a1, b = float(sol1[0]), float(sol1[1])

            offset = a1 * tau + b
            X2 = (x[mask2] - tau).reshape(-1, 1)
            y2_centered = y[mask2] - offset
            reg2 = lam * np.eye(1)
            sol2 = np.linalg.lstsq(X2.T @ X2 + reg2, X2.T @ y2_centered, rcond=None)[0]
            a2 = float(sol2[0])

            cost = float(np.mean((y1 - (a1 * x[mask1] + b)) ** 2) +
                         np.mean((y[mask2] - (a2 * (x[mask2] - tau) + offset)) ** 2) +
                         lam * (a1 ** 2 + a2 ** 2))
            return a1, a2, b, cost
        except Exception:
            return 1.0, 1.0, 0.0, np.inf

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PiecewiseLinearBlock":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # Grid de τ: percentis 10..90 em passos de 5
        quantiles = np.percentile(x, np.arange(10, 91, 5))
        if self.fixed_tau is not None:
            quantiles = np.array([self.fixed_tau])
        best_cost = np.inf
        best_tau = float(np.median(x)) if self.fixed_tau is None else float(self.fixed_tau)
        best_params = (1.0, 1.0, 0.0)

        for tau in np.unique(quantiles):
            a1, a2, b, cost = self._fit_given_tau(x, y, tau)
            if cost < best_cost:
                best_cost = cost
                best_tau = tau
                best_params = (a1, a2, b)

        self._tau = best_tau
        self._a1, self._a2, self._b = best_params
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        offset = self._a1 * self._tau + self._b
        out = np.where(
            x <= self._tau,
            self._a1 * x + self._b,
            self._a2 * (x - self._tau) + offset,
        )
        return out.astype(float)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inversa analítica por segmento. f é contínua em τ → partição y por f(τ)."""
        y = np.asarray(y, dtype=float).ravel()
        f_tau = self._a1 * self._tau + self._b
        offset = f_tau  # = a1*τ + b

        out = np.empty_like(y)
        seg1 = y <= f_tau
        seg2 = ~seg1

        # Seg1: y = a1*x + b → x = (y - b) / a1
        if self._a1 != 0:
            out[seg1] = (y[seg1] - self._b) / self._a1
        else:
            out[seg1] = self._tau  # degenerate: flat segment

        # Seg2: y = a2*(x - τ) + offset → x = (y - offset)/a2 + τ
        if self._a2 != 0:
            out[seg2] = (y[seg2] - offset) / self._a2 + self._tau
        else:
            out[seg2] = self._tau

        return out

    @property
    def n_params(self) -> int:
        return 3  # τ, a1, a2 (b é determinado pela continuidade + regressão seg1)

    def name(self) -> str:
        return "piecewise"


# ---------------------------------------------------------------------------
# BSplineBlock
# ---------------------------------------------------------------------------

class BSplineBlock(_NonlinearBlock):
    """B-Spline com n_knots nós interiores em quantis uniformes de x.

    Ajuste via LSQUnivariateSpline (scipy) com suavização lambda opcional.
    Inversa numérica via interpolação de grade (scipy interp1d).
    """

    def __init__(self, n_knots: int = 3) -> None:
        self._n_knots = max(1, int(n_knots))
        self._spline: LSQUnivariateSpline | None = None
        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._fitted: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BSplineBlock":
        from scipy.interpolate import interp1d
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # Ordenar por x (LSQUnivariateSpline exige x crescente)
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]

        # Remover duplicatas exatas em x (interpolação quebra)
        _, uniq = np.unique(xs, return_index=True)
        xs, ys = xs[uniq], ys[uniq]

        self._x_min, self._x_max = float(xs[0]), float(xs[-1])

        # Posicionar nós interiores em quantis uniformes (excluir extremos)
        q_positions = np.linspace(0, 100, self._n_knots + 2)[1:-1]
        knots = np.percentile(xs, q_positions)
        # Garantir nós estritamente dentro do domínio
        knots = np.clip(knots, xs[1], xs[-2]) if len(xs) > 2 else np.array([xs.mean()])
        knots = np.unique(knots)

        # Pesos de regularização: w=1/σ² implícito via lam
        w = np.ones(len(xs))

        try:
            self._spline = LSQUnivariateSpline(xs, ys, t=knots, w=w, k=3)
        except Exception:
            # Fallback: 1 nó central
            knots_fb = np.array([xs.mean()])
            try:
                self._spline = LSQUnivariateSpline(xs, ys, t=knots_fb, w=w, k=3)
            except Exception:
                # Fallback máximo: linear
                self._spline = LSQUnivariateSpline(xs, ys, t=[], w=w, k=1)

        # Grid para inversa numérica (1000 pontos)
        x_grid = np.linspace(self._x_min, self._x_max, 1000)
        y_grid = self._spline(x_grid)
        self._inv_interp = interp1d(
            y_grid, x_grid,
            kind="linear",
            bounds_error=False,
            fill_value=(x_grid[0], x_grid[-1]),
        )
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        x_clipped = np.clip(x, self._x_min, self._x_max)
        return self._spline(x_clipped).astype(float)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inversa numérica via interpolação de grade pré-computada."""
        y = np.asarray(y, dtype=float).ravel()
        return self._inv_interp(y).astype(float)

    @property
    def n_params(self) -> int:
        # n_knots + grau + 1 coeficientes típicos do spline
        return self._n_knots + 4

    def name(self) -> str:
        return f"bspline{self._n_knots}"


# ---------------------------------------------------------------------------
# HalfSpaceLinearBlock
# ---------------------------------------------------------------------------

class HalfSpaceLinearBlock(_NonlinearBlock):
    """Inclinações distintas para x>0 e x≤0, intercepto compartilhado em 0.

    f(x) = a_pos * x   se x > 0
           a_neg * x   se x ≤ 0

    Garante f(0) = 0 (sem intercepto livre).
    Modela assimetria fiscal de empresas reguladas:
      - a_pos: resposta a ganhos acima do cap regulatório
      - a_neg: resposta a perdas / revisão tarifária

    Inversa analítica: f⁻¹(y) = y/a_pos se y>0 else y/a_neg
    (pressupõe a_pos > 0, a_neg > 0 — slopes negativos tornam f não monotônica)
    """

    def __init__(self) -> None:
        self._a_pos: float = 1.0
        self._a_neg: float = 1.0
        self._fitted: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HalfSpaceLinearBlock":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        mask_pos = x > 0
        mask_neg = ~mask_pos

        lam = self._lam

        def _ols_no_intercept(xv: np.ndarray, yv: np.ndarray) -> float:
            if len(xv) < 1:
                return 1.0
            # (X'X + lam)⁻¹ X'y com X = xv[:,None]
            xtx = float(np.dot(xv, xv)) + lam
            xty = float(np.dot(xv, yv))
            return xty / xtx if xtx != 0 else 1.0

        self._a_pos = _ols_no_intercept(x[mask_pos], y[mask_pos]) if mask_pos.sum() >= 1 else 1.0
        self._a_neg = _ols_no_intercept(x[mask_neg], y[mask_neg]) if mask_neg.sum() >= 1 else 1.0
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        return np.where(x > 0, self._a_pos * x, self._a_neg * x).astype(float)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inversa analítica: divide por a_pos ou a_neg conforme sinal de y."""
        y = np.asarray(y, dtype=float).ravel()
        a_pos = self._a_pos if self._a_pos != 0 else 1e-8
        a_neg = self._a_neg if self._a_neg != 0 else 1e-8
        return np.where(y > 0, y / a_pos, y / a_neg).astype(float)

    @property
    def n_params(self) -> int:
        return 2  # a_pos, a_neg

    def name(self) -> str:
        return "halfspace"


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------

def make_block(nl_type: str, n_knots: int | None = None) -> _NonlinearBlock:
    """Instancia bloco não-linear pelo nome.

    Args:
        nl_type : "sigmoid" | "piecewise" | "bspline" | "halfspace"
        n_knots : número de nós para BSplineBlock (ignorado pelos demais)
    """
    if nl_type == "sigmoid":
        return SigmoidBlock()
    if nl_type == "piecewise":
        return PiecewiseLinearBlock()
    if nl_type == "bspline":
        return BSplineBlock(n_knots=n_knots or 3)
    if nl_type == "halfspace":
        return HalfSpaceLinearBlock()
    raise ValueError(f"Bloco não-linear desconhecido: {nl_type!r}. "
                     f"Use 'sigmoid', 'piecewise', 'bspline' ou 'halfspace'.")
