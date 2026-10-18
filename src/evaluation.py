"""Métricas, walk-forward CV, Diebold-Mariano, MCS e diagnóstico de resíduos.

Todas as métricas são calculadas na escala original (após inverse_transform).
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro


# ---------------------------------------------------------------------------
# Métricas escalares
# ---------------------------------------------------------------------------

def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FIT% — convenção MATLAB/SITB: (1 - ||e|| / ||y - mean(y)||) × 100."""
    denom = np.linalg.norm(y_true - np.mean(y_true))
    if denom == 0:
        return 0.0
    return float((1 - np.linalg.norm(y_true - y_pred) / denom) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def theil_u2(y_true: np.ndarray, y_pred: np.ndarray, y_rw: np.ndarray) -> float:
    """RMSE_model / RMSE_RandomWalk. Valor > 1 indica pior que naïve."""
    rw = rmse(y_true, y_rw)
    return float("inf") if rw == 0 else float(rmse(y_true, y_pred) / rw)


# ---------------------------------------------------------------------------
# Walk-forward CV (expanding window, h=1)
# ---------------------------------------------------------------------------

# Round 41 (protocolo metodológico §32): máscara de pontuação padrão para uma busca inteira. Os executores
# de busca chamam `walk_forward_cv` em dezenas de pontos sem `score_exclude`; dentro deste contexto
# todos herdam a mesma máscara, sem mudar a assinatura de cada candidato. Fora dele, nada muda.
_SCORE_EXCLUDE_DEFAULT: ContextVar = ContextVar("score_exclude_default", default=None)


@contextmanager
def score_exclusion(dates: "list[pd.Timestamp] | None"):
    """Alvos em `dates` saem da pontuação de todo `walk_forward_cv` chamado sem `score_exclude`."""
    token = _SCORE_EXCLUDE_DEFAULT.set(list(dates) if dates else None)
    try:
        yield
    finally:
        _SCORE_EXCLUDE_DEFAULT.reset(token)


def walk_forward_cv(
    model_fn,
    y: pd.Series,
    u_exog: pd.DataFrame | None = None,
    min_train: int = 20,
    y_score: pd.Series | None = None,
    score_exclude: "list[pd.Timestamp] | None" = None,
) -> dict:
    """Expanding-window walk-forward CV com horizonte h=1.

    model_fn(y_train, u_train) → fitted model com predict(steps, u_future).
    y deve estar na escala original; model_fn é responsável por retornar
    previsões também na escala original (inverse_transform interno).

    `y_score` e `score_exclude` (Round 40, protocolo metodológico §31) separam estimar de pontuar:
    o modelo e o passeio aleatório veem sempre `y` (a série tratada), mas o alvo pode ser lido de
    `y_score` (valores registrados) e os alvos em `score_exclude` saem de TODAS as métricas e
    arrays devolvidos, de modo que a máscara é a mesma para qualquer competidor da combinação.
    Sem os dois, o comportamento é o de sempre.

    Retorna dict com métricas agregadas + arrays residuals/y_true/y_pred (só os alvos pontuados)
    e `score_mask` (bool, um por dobra) com `n_excluded`.
    """
    n = len(y)
    preds: list[float] = []
    actuals: list[float] = []
    rw_preds: list[float] = []
    if score_exclude is None:
        score_exclude = _SCORE_EXCLUDE_DEFAULT.get()
    target = y if y_score is None else y_score.reindex(y.index)
    excl = {pd.Timestamp(t) for t in (score_exclude or [])}

    for t in range(min_train, n):
        y_train = y.iloc[:t]
        u_train = u_exog.iloc[:t] if u_exog is not None else None
        u_fut   = u_exog.iloc[[t]] if u_exog is not None else None

        model = model_fn(y_train, u_train)
        raw = model.predict(steps=1, u_future=u_fut)
        pred = float(raw[0] if hasattr(raw, "__len__") else raw)

        preds.append(pred)
        actuals.append(float(target.iloc[t]))
        rw_preds.append(float(y_train.iloc[-1]))  # RW naïve inline

    keep = np.array([y.index[t] not in excl for t in range(min_train, n)], dtype=bool)
    y_t = np.array(actuals)[keep]
    y_p = np.array(preds)[keep]
    y_rw = np.array(rw_preds)[keep]
    res = y_t - y_p

    return {
        "fit_percent": fit_percent(y_t, y_p),
        "rmse":        rmse(y_t, y_p),
        "mae":         mae(y_t, y_p),
        "mape":        mape(y_t, y_p),
        "mse":         float(np.mean(res ** 2)),
        "variance":    float(np.var(res)),
        "theil_u2":    theil_u2(y_t, y_p, y_rw),
        "residuals":   res,
        "y_true":      y_t,
        "y_pred":      y_p,
        "y_rw":        y_rw,
        "score_mask":  keep,
        "n_excluded":  int((~keep).sum()),
    }


def restrict_to_eval_range(cv: dict, min_train: int, eval_start: int) -> dict:
    """Recalcula as métricas de `walk_forward_cv` restritas a um intervalo OOS fixo.

    Usado para comparar diferentes valores de `min_train` de forma justa (Round 24;
    protocolo metodológico §26): sem isso, uma janela de treino maior automaticamente produz menos
    folds OOS e o FIT%/U₂ deixa de ser comparável entre janelas — não porque o modelo prevê
    melhor, só porque a média é tirada sobre um conjunto de pontos diferente. Recortando todas
    as janelas ao mesmo intervalo comum (`eval_start` até o fim da série) isola o efeito do
    tamanho de treino do efeito de "quantos folds entraram na média".
    """
    if eval_start < min_train:
        raise ValueError(f"eval_start ({eval_start}) deve ser >= min_train ({min_train})")
    offset = eval_start - min_train
    y_t  = np.asarray(cv["y_true"])[offset:]
    y_p  = np.asarray(cv["y_pred"])[offset:]
    y_rw = np.asarray(cv["y_rw"])[offset:]
    res  = y_t - y_p
    return {
        "fit_percent": fit_percent(y_t, y_p),
        "rmse":        rmse(y_t, y_p),
        "mae":         mae(y_t, y_p),
        "mape":        mape(y_t, y_p),
        "mse":         float(np.mean(res ** 2)),
        "variance":    float(np.var(res)),
        "theil_u2":    theil_u2(y_t, y_p, y_rw),
        "residuals":   res,
        "n_folds":     int(len(y_t)),
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano (Harvey, Leybourne & Newbold 1997)
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Teste DM modificado (HLN 1997) para igualdade de acurácia preditiva.

    H0: E[L(e1) - L(e2)] = 0  (perda MSE).
    Retorna (stat, pvalue). Distribuição t(T-1) sob H0.
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    d_bar = float(np.mean(d))

    # Estimador Newey-West com h-1 lags
    gamma0 = float(np.mean((d - d_bar) ** 2))
    gamma_sum = sum(
        float(np.mean((d[k:] - d_bar) * (d[:T - k] - d_bar)))
        for k in range(1, h)
    ) if h > 1 else 0.0
    f_hat = gamma0 + 2 * gamma_sum

    if f_hat <= 0:
        return 0.0, 1.0

    dm = d_bar / np.sqrt(f_hat / T)
    # Correção de amostra finita (HLN eq. 14)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    mdm = float(correction * dm)
    pvalue = float(2 * stats.t.sf(abs(mdm), df=T - 1))
    return mdm, pvalue


# ---------------------------------------------------------------------------
# Clark-West (2007) — comparação de modelos aninhados (ARMAX/ARIMAX vs ARIMA)
# ---------------------------------------------------------------------------

def clark_west_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Teste CW (Clark & West 2007) para acurácia preditiva igual em modelos aninhados.

    Usar no lugar do DM sempre que o modelo 2 contém o modelo 1 como caso particular
    (ex.: ARMAX/ARIMAX vs a ARIMA da mesma ordem) — sob H0 o DM tradicional é enviesado
    contra o modelo maior pelo ruído dos parâmetros extras estimados sob a nula.

    e1: resíduos do modelo MENOR (aninhado, referência). e2: resíduos do modelo MAIOR.
    H0: acurácia igual. H1 (unilateral): o modelo maior é melhor.
    Retorna (stat, pvalue), referido à normal padrão unilateral (cauda superior).
    """
    f = e1 ** 2 - (e2 ** 2 - (e1 - e2) ** 2)
    T = len(f)
    f_bar = float(np.mean(f))

    gamma0 = float(np.mean((f - f_bar) ** 2))
    gamma_sum = sum(
        float(np.mean((f[k:] - f_bar) * (f[:T - k] - f_bar)))
        for k in range(1, h)
    ) if h > 1 else 0.0
    f_hat = gamma0 + 2 * gamma_sum

    if f_hat <= 0:
        return 0.0, 1.0

    cw = float(f_bar / np.sqrt(f_hat / T))
    pvalue = float(stats.norm.sf(cw))
    return cw, pvalue


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011 — eliminação sequencial)
# ---------------------------------------------------------------------------

def model_confidence_set(
    errors_dict: dict[str, np.ndarray],
    alpha: float = 0.10,
) -> list[str]:
    """MCS simplificado: elimina iterativamente o modelo com pior perda via DM.

    errors_dict: {nome_modelo: array_de_residuos}.
    Retorna lista de modelos não significativamente inferiores ao nível alpha.
    """
    active = list(errors_dict.keys())
    while len(active) > 1:
        losses = {m: float(np.mean(errors_dict[m] ** 2)) for m in active}
        worst = max(losses, key=losses.__getitem__)
        best  = min(losses, key=losses.__getitem__)
        _, pval = diebold_mariano_test(errors_dict[worst], errors_dict[best])
        if pval > alpha:
            break  # diferença não significativa — todos os restantes no MCS
        active.remove(worst)
    return active


def spa_test(
    bench_errors: np.ndarray,
    model_errors: np.ndarray,
    block_size: int = 4,
    reps: int = 2000,
    seed: int = 42,
) -> dict:
    """SPA de Hansen (2005) com o Reality Check de White (2000) — Round 41, protocolo metodológico §32.

    bench_errors: (T,) erros do benchmark; model_errors: (T, M) erros dos M candidatos da busca.
    Perda quadrática; bootstrap estacionário (Politis & Romano 1994) de bloco médio `block_size`.
    H0: nenhum candidato tem perda esperada menor que a do benchmark. Devolve os três p-valores do
    `arch` — `consistent` é o de Hansen (leitura principal), `upper` é o Reality Check de White.
    """
    from arch.bootstrap import SPA

    bench = np.asarray(bench_errors, dtype=float) ** 2
    models = np.atleast_2d(np.asarray(model_errors, dtype=float).T).T ** 2
    spa = SPA(bench, models, block_size=block_size, reps=reps,
              bootstrap="stationary", studentize=True, nested=False, seed=seed)
    spa.compute()
    pv = spa.pvalues
    return {"p_consistent": float(pv["consistent"]), "p_lower": float(pv["lower"]),
            "p_white": float(pv["upper"]), "n_models": int(models.shape[1]),
            "n_obs": int(models.shape[0]), "block_size": block_size, "reps": reps}


# ---------------------------------------------------------------------------
# Diagnóstico de resíduos
# ---------------------------------------------------------------------------

def residual_diagnostics(residuals: np.ndarray) -> dict:
    """Ljung-Box Q(4), ARCH-LM(4), Jarque-Bera e Shapiro-Wilk sobre resíduos.

    Retorna dict compatível com schema_resultados.json.
    """
    r = residuals[~np.isnan(residuals)]

    # Ljung-Box Q(4)
    lb = acorr_ljungbox(r, lags=[4], return_df=True)
    lb_stat  = float(lb["lb_stat"].iloc[0])
    lb_pval  = float(lb["lb_pvalue"].iloc[0])

    # ARCH-LM(4)
    arch_stat, arch_pval, _, _ = het_arch(r, nlags=4)

    # Jarque-Bera
    jb_stat, jb_pval, _, _ = jarque_bera(r)

    # Shapiro-Wilk (preferível para N ≤ 64)
    sw_stat, sw_pval = shapiro(r)

    return {
        "ljung_box_q4_stat":   lb_stat,
        "ljung_box_q4_pvalue": lb_pval,
        "arch_lm_stat":        float(arch_stat),
        "arch_lm_pvalue":      float(arch_pval),
        "jarque_bera_stat":    float(jb_stat),
        "jarque_bera_pvalue":  float(jb_pval),
        "shapiro_wilk_stat":   float(sw_stat),
        "shapiro_wilk_pvalue": float(sw_pval),
    }
