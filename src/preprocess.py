"""Transformações de séries e testes de raiz unitária (ADF, KPSS, PP).

Convenções:
- d determinado pelo consenso ADF/KPSS/PP sobre a série *transformada*.
- Conflitos entre testes devem ser registrados em rounds/science_log.md.
- inverse_transform requer a série original como âncora para modos com diff.
- detect_and_impute_spikes aplica imputação ANTES de qualquer transformação;
  spikes conhecidos (SPIKE_KNOWN em config.py) e detecção automática via z-score.
- boxcox: lambda selecionado por método de Guerrero (1993), congelado por combo
  antes do walk-forward (ver `guerrero_lambda`; protocolo metodológico secao 25).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from statsmodels.tsa.stattools import adfuller, kpss

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config as cfg  # noqa: E402

_BOXCOX_LAM_TOL = 1e-6  # |lambda| abaixo disso -> tratado como log (mesma convencao do scipy)


def guerrero_lambda(
    series: pd.Series,
    period: int = None,  # type: ignore[assignment]
) -> float:
    """Seleciona lambda de Box-Cox pelo método de Guerrero (1993).

    Particiona a série em sub-séries não sobrepostas de tamanho `period`
    (default cfg.BOXCOX_GUERRERO_PERIOD, agrupamento trimestre->ano). Para cada
    sub-série i, calcula média x̄_i e desvio padrão s_i; para um lambda candidato,
    a dispersão reescalada é s_i(λ) = s_i / x̄_i^(1-λ). λ* minimiza o coeficiente
    de variação de {s_i(λ)} entre sub-séries — busca por scipy.optimize.minimize_scalar
    (bounded) em cfg.BOXCOX_LAMBDA_BOUNDS.

    Requer valores estritamente positivos (mesma exigência do log/log+diff).
    """
    if period is None:
        period = cfg.BOXCOX_GUERRERO_PERIOD
    s = series.dropna()
    if (s <= 0).any():
        raise ValueError(f"boxcox requer valores positivos; mín={s.min():.4g}")

    n_blocks = len(s) // period
    if n_blocks < 2:
        raise ValueError(
            f"série curta demais para Guerrero (period={period}, n={len(s)})"
        )
    trimmed = s.values[-n_blocks * period:].reshape(n_blocks, period)
    means = trimmed.mean(axis=1)
    stds = trimmed.std(axis=1, ddof=1)

    def _cv(lam: float) -> float:
        rescaled = stds / np.power(means, 1.0 - lam)
        m = rescaled.mean()
        if m <= 0 or not np.isfinite(m):
            return np.inf
        return float(rescaled.std(ddof=1) / m)

    res = minimize_scalar(_cv, bounds=cfg.BOXCOX_LAMBDA_BOUNDS, method="bounded")
    return float(res.x)


# ---------------------------------------------------------------------------
# Detecção e imputação de spikes
# ---------------------------------------------------------------------------

def detect_and_impute_spikes(
    series: pd.Series,
    z_thresh: float | None = None,
    method: str = "linear_interp",
    known_spikes: "list[pd.Timestamp] | None" = None,
    exclude: "list[pd.Timestamp] | None" = None,
) -> "tuple[pd.Series, list[dict]]":
    """Detecta e imputa spikes pontuais (eventos não-recorrentes).

    Detecção: via |z-score| > z_thresh e/ou lista de timestamps conhecidos. Datas em `exclude`
    nunca são imputadas, nem pela lista nem pela detecção automática.
    Imputação:
      - "linear_interp": substitui por (t-1 + t+1) / 2.
        Se o spike está na primeira/última posição, usa o único vizinho disponível.
      - "previous": substitui por t-1 (causal; não usa t+1). Na primeira posição, usa t+1.

    Retorna:
        (série corrigida, lista de registros com timestamp/original/imputed/z_score)

    Importante para anti-leakage: a substituição usa t+1 — isso é aceitável quando
    a imputação é feita ANTES do walk-forward CV e documentada em protocolo metodológico
    como correção pontual de eventos contábeis não-recorrentes identificados a priori.
    """
    s = series.copy()
    mu, sigma = float(series.mean()), float(series.std())

    to_impute: set[pd.Timestamp] = set()

    # Detecção automática via z-score
    if z_thresh is not None and sigma > 0:
        z = (s - mu) / sigma
        for ts in s.index[z.abs() > z_thresh]:
            to_impute.add(ts)

    # Spikes conhecidos (forçados independentemente do z-score)
    if known_spikes:
        for ks in known_spikes:
            ks_ts = pd.Timestamp(ks)
            # Aceita correspondência exata ou por data (ignora hora)
            matches = [t for t in s.index if t == ks_ts or t.date() == ks_ts.date()]
            to_impute.update(matches)

    if exclude:
        excl = {pd.Timestamp(e).date() for e in exclude}
        to_impute = {t for t in to_impute if t.date() not in excl}

    records: list[dict] = []
    for ts in sorted(to_impute):
        loc = s.index.get_loc(ts)
        original = float(s.iloc[loc])

        if method == "linear_interp":
            prev_val = float(s.iloc[loc - 1]) if loc > 0 else float(s.iloc[loc + 1])
            next_val = float(s.iloc[loc + 1]) if loc < len(s) - 1 else float(s.iloc[loc - 1])
            imputed = (prev_val + next_val) / 2.0
        elif method == "previous":
            imputed = float(s.iloc[loc - 1]) if loc > 0 else float(s.iloc[loc + 1])
        else:
            raise ValueError(f"Método de imputação desconhecido: {method!r}")

        s.loc[ts] = imputed
        records.append({
            "timestamp": ts,
            "original": original,
            "imputed": imputed,
            "method": method,
            "z_score": round((original - mu) / sigma, 3) if sigma > 0 else None,
        })

    return s, records


# ---------------------------------------------------------------------------
# Diagnóstico expandido de outliers — Passos 1 e 2 (protocolo metodológico §11.2/11.3)
# ---------------------------------------------------------------------------

def detect_by_residuals(
    series: pd.Series,
    arima_order: "tuple[int, int, int] | None" = None,
    p_max: int = 3,
    q_max: int = 3,
    z_thresh: float = 3.0,
) -> "tuple[pd.Series, list[dict]]":
    """Passo 1: ajusta ARIMA e flagga pontos com |resíduo padronizado| > z_thresh.

    Se arima_order=None, seleciona (p,d,q) via grid AIC com p,q ∈ [0,p_max]; d via consenso.
    Retorna: (residuals Series, lista de {timestamp, residual, z_residual, arima_order}).
    """
    import warnings as _w
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    s = series.dropna()

    if arima_order is None:
        ur = unit_root_tests(s)
        d = ur["consensus_d"]
        best_aic = np.inf
        best_order: tuple[int, int, int] = (1, d, 0)
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        m = _ARIMA(s, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                except Exception:
                    pass
        arima_order = best_order

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fitted = _ARIMA(s, order=arima_order).fit()

    residuals = fitted.resid
    sigma = float(residuals.std())
    flagged: list[dict] = []
    if sigma > 0:
        z_res = residuals / sigma
        for ts in residuals.index[z_res.abs() > z_thresh]:
            flagged.append({
                "timestamp": ts,
                "residual": float(residuals[ts]),
                "z_residual": float(z_res[ts]),
                "arima_order": arima_order,
            })
    return residuals, flagged


def classify_outlier_type(
    residuals: pd.Series,
    ts: "pd.Timestamp",
    sigma: "float | None" = None,
    level_shift_window: int = 8,
    tc_decay_window: int = 3,
    ls_threshold_factor: float = 0.5,
) -> str:
    """Passo 2: classifica outlier em ts como 'AO', 'LS' ou 'TC'.

    AO: spike isolado — resíduos antes e depois ao mesmo nível médio.
    LS: mudança permanente na média dos resíduos após ts (level shift).
    TC: efeito presente em tc_decay_window períodos seguintes com decaimento monotônico.
    IO (Innovation Outlier): indistinguível de AO via resíduos padronizados; tratado como AO.

    Parâmetros
    ----------
    level_shift_window : janela (períodos) usada para estimar μ_pré e μ_pós.
    ls_threshold_factor: |μ_pós - μ_pré| > factor×σ → candidato a LS/TC (não AO).
    """
    if sigma is None:
        sigma = float(residuals.std())
    if sigma == 0:
        return "AO"

    loc = residuals.index.get_loc(ts)
    n = len(residuals)

    pre_slice = residuals.iloc[max(0, loc - level_shift_window): loc]
    post_slice = residuals.iloc[loc + 1: min(n, loc + 1 + level_shift_window)]

    if len(pre_slice) == 0 or len(post_slice) == 0:
        return "AO"  # borda da série — insuficiente para classificar

    shift = abs(float(post_slice.mean()) - float(pre_slice.mean()))
    if shift < ls_threshold_factor * sigma:
        return "AO"

    # Candidato a LS ou TC — inspecionar janela de decaimento pós t*
    decay_slice = residuals.iloc[loc + 1: min(n, loc + 1 + tc_decay_window)]
    if len(decay_slice) >= 2:
        abs_vals = decay_slice.abs().values
        is_monotone = all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))
        far_start = loc + 1 + tc_decay_window
        if far_start < n:
            far_mean = abs(float(residuals.iloc[far_start:].mean()))
            if is_monotone and far_mean < 0.5 * sigma:
                return "TC"

    return "LS"


def build_ls_dummies(
    index: "pd.DatetimeIndex",
    ls_dates: "list[str]",
) -> pd.DataFrame:
    """Constrói DataFrame de dummies binárias para Level Shifts detectados.

    Para cada breakpoint em ls_dates cria coluna 'LS_<YYYYMMDD>' com:
        1.0 se t >= breakpoint, 0.0 caso contrário.

    Dummies são determinísticas — sem leakage no walk-forward CV, pois o
    valor em t+1 é conhecido no momento da estimação no fold [0, t].
    """
    dummies: dict[str, list[float]] = {}
    for ls_str in ls_dates:
        ts = pd.Timestamp(ls_str)
        col = f"LS_{ts.strftime('%Y%m%d')}"
        dummies[col] = [(1.0 if idx >= ts else 0.0) for idx in index]
    if dummies:
        return pd.DataFrame(dummies, index=index)
    return pd.DataFrame(index=index)


# ---------------------------------------------------------------------------
# Transformações
# ---------------------------------------------------------------------------

def transform(series: pd.Series, mode: str, lam: float | None = None) -> pd.Series:
    """Aplica transformação; levanta ValueError se log/boxcox de valor não-positivo.

    lam: lambda de Box-Cox (só usado se mode=="boxcox"). Se None, é calculado
    via guerrero_lambda(series) — mas o padrão do pipeline é calcular uma vez
    por combo e congelar (ver main.py), não recomputar aqui.
    """
    if mode == "raw":
        return series.copy()
    if mode == "log":
        if (series <= 0).any():
            raise ValueError(
                f"log requer valores positivos; mín={series.min():.4g}"
            )
        return np.log(series)
    if mode == "diff":
        return series.diff().dropna()
    if mode == "log+diff":
        if (series <= 0).any():
            raise ValueError(
                f"log+diff requer valores positivos; mín={series.min():.4g}"
            )
        return np.log(series).diff().dropna()
    if mode == "boxcox":
        if (series <= 0).any():
            raise ValueError(
                f"boxcox requer valores positivos; mín={series.min():.4g}"
            )
        if lam is None:
            lam = guerrero_lambda(series)
        if abs(lam) < _BOXCOX_LAM_TOL:
            return np.log(series)
        return (np.power(series, lam) - 1.0) / lam
    raise ValueError(f"Modo desconhecido: {mode!r}")


def inverse_transform(
    pred: np.ndarray, mode: str, ref: pd.Series, lam: float | None = None
) -> np.ndarray:
    """Desfaz transformação de volta à escala original.

    ref: série original de treinamento (usada como âncora para modos com diff).
    lam: lambda de Box-Cox congelado (obrigatório se mode=="boxcox").
    Para h>1, pred contém os incrementos/log-incrementos de cada passo.
    """
    pred = np.asarray(pred, dtype=float)
    if mode == "raw":
        return pred
    if mode == "log":
        return np.exp(pred)
    if mode == "diff":
        last = float(ref.iloc[-1])
        return np.cumsum(pred) + last
    if mode == "log+diff":
        last = float(ref.iloc[-1])
        log_last = np.log(last) if last > 0 else 0.0
        return np.exp(np.cumsum(pred) + log_last)
    if mode == "boxcox":
        if lam is None:
            raise ValueError("boxcox requer lambda congelado para inverse_transform")
        if abs(lam) < _BOXCOX_LAM_TOL:
            return np.exp(pred)
        base = lam * pred + 1.0
        base = np.clip(base, 1e-12, None)  # guarda numérica: base**（1/λ) exige base>0
        return np.power(base, 1.0 / lam)
    raise ValueError(f"Modo desconhecido: {mode!r}")


# ---------------------------------------------------------------------------
# Testes de raiz unitária
# ---------------------------------------------------------------------------

def unit_root_tests(series: pd.Series) -> dict:
    """Executa ADF, KPSS e PP (se arch disponível); retorna dict com resultados e consensus_d.

    H0 ADF / PP : série tem raiz unitária → rejeitar H0 → stationary
    H0 KPSS     : série é estacionária → rejeitar H0 → unit_root
    consensus_d : 0 se ≥2 testes dizem stationary, caso contrário 1.
    Conflitos (1 a 1) são registrados pela camada chamadora no science_log.md.
    """
    s = series.dropna()
    result: dict = {}

    # ADF (H0: raiz unitária)
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    result["ADF"] = {
        "stat": float(adf_stat),
        "p_value": float(adf_p),
        "conclusion": "stationary" if adf_p < 0.05 else "unit_root",
    }

    # KPSS (H0: estacionária) — p truncado em [0.01, 0.10] no statsmodels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
    result["KPSS"] = {
        "stat": float(kpss_stat),
        "p_value": float(kpss_p),
        "conclusion": "unit_root" if kpss_p < 0.05 else "stationary",
    }

    # PP (H0: raiz unitária) — via arch se disponível
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(s)
        result["PP"] = {
            "stat": float(pp.stat),
            "p_value": float(pp.pvalue),
            "conclusion": "stationary" if pp.pvalue < 0.05 else "unit_root",
        }
    except ImportError:
        result["PP"] = {"stat": None, "p_value": None, "conclusion": "not_available"}

    n_stationary = sum(
        1 for v in result.values()
        if isinstance(v, dict) and v.get("conclusion") == "stationary"
    )
    result["consensus_d"] = 0 if n_stationary >= 2 else 1

    return result


# ---------------------------------------------------------------------------
# CLI de diagnóstico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.loaders import load_indicador

    header = f"{'Série':<12} {'Modo':<10} {'ADF':>12} {'KPSS':>12} {'PP':>12} {'d':>3}"
    print(header)
    print("-" * len(header))

    for nome in ["resultado_bruto", "lucro_consolidado", "despesas_operacionais"]:
        df = load_indicador(nome)
        for col in df.columns:
            series = df[col].dropna()
            for mode in ["raw", "diff"]:
                try:
                    y_t = transform(series, mode)
                    r = unit_root_tests(y_t)
                    adf_c = r["ADF"]["conclusion"][:4]
                    kpss_c = r["KPSS"]["conclusion"][:4]
                    pp_c = r["PP"]["conclusion"][:4]
                    d = r["consensus_d"]
                    print(f"{col:<12} {mode:<10} {adf_c:>12} {kpss_c:>12} {pp_c:>12} {d:>3}")
                except ValueError as e:
                    print(f"{col:<12} {mode:<10} {'SKIP':>12} ({e})")
