"""Seleção de variáveis exógenas: VIF → Granger → AIC stepwise.

Fluxo por combinação empresa×indicador:
  1. VIF iterativo: remove colunas com VIF > threshold até max(VIF) ≤ threshold.
  2. Granger: mantém colunas que Granger-causam a série (p < alpha em algum lag).
  3. AIC stepwise forward: adiciona colunas enquanto AIC de SARIMAX(1,d,1) reduz.

Saída: outputs/results/exog_selection.json  →  {emp_code: {ind: [col1, col2, ...]}}

Executável direto: python -m src.feature_selection (ou via main.py --round full)
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Adiciona raiz ao sys.path para uso como script direto
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.loaders import load_all  # noqa: E402
from src.preprocess import unit_root_tests, transform, detect_and_impute_spikes  # noqa: E402
import config as cfg  # noqa: E402
import paths  # noqa: E402

_RESULTS_DIR = paths.RESULTS_DIR
_EXOG_PATH   = _RESULTS_DIR / "exog_selection.json"


# ---------------------------------------------------------------------------
# 1. Filtro VIF
# ---------------------------------------------------------------------------

def vif_filter(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Remove iterativamente a coluna com maior VIF enquanto max(VIF) > threshold."""
    df = df.dropna(axis=0).copy()
    if df.shape[1] <= 1:
        return df
    while df.shape[1] >= 2:
        vifs = np.array([
            variance_inflation_factor(df.values.astype(float), i)
            for i in range(df.shape[1])
        ])
        if np.nanmax(vifs) <= threshold:
            break
        worst = int(np.nanargmax(vifs))
        df = df.drop(columns=df.columns[worst])
    return df


# ---------------------------------------------------------------------------
# 2. Filtro Granger
# ---------------------------------------------------------------------------

def granger_filter(
    series: pd.Series,
    candidates: pd.DataFrame,
    max_lag: int = 4,
    alpha: float = 0.10,
) -> list[str]:
    """Mantém colunas de `candidates` que Granger-causam `series` (p < alpha)."""
    selected: list[str] = []
    s = series.dropna()

    for col in candidates.columns:
        x = candidates[col].reindex(s.index)
        data = pd.concat([s, x], axis=1).dropna()
        if len(data) < max_lag + 5:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
            for _lag, test_result in res.items():
                if test_result[0]["ssr_ftest"][1] < alpha:
                    selected.append(col)
                    break
        except Exception:
            pass
    return selected


# ---------------------------------------------------------------------------
# 3. AIC stepwise forward
# ---------------------------------------------------------------------------

def aic_stepwise_forward(
    series: pd.Series,
    candidates: list[str],
    inputs: pd.DataFrame,
    d: int,
    mode: str,
    max_exog: int = 5,
) -> list[str]:
    """Adiciona variáveis (forward) enquanto AIC de SARIMAX(1,d,1) reduz."""

    y_t = transform(series, mode)

    def _aic(exog_cols: list[str]) -> float:
        if not exog_cols:
            u_vals = None
        else:
            u = inputs[exog_cols].reindex(y_t.index).ffill().bfill().fillna(0.0)
            u_vals = u.iloc[-len(y_t):].values.astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(y_t.values, exog=u_vals, order=(1, d, 1)).fit(disp=False)
            return float(m.aic) if np.isfinite(m.aic) else np.inf
        except Exception:
            return np.inf

    selected: list[str] = []
    current_aic = _aic([])

    for _ in range(max_exog):
        best_aic = current_aic
        best_col: str | None = None
        for col in candidates:
            if col in selected:
                continue
            aic = _aic(selected + [col])
            if aic < best_aic:
                best_aic = aic
                best_col = col
        if best_col is None:
            break
        selected.append(best_col)
        current_aic = best_aic

    return selected


# ---------------------------------------------------------------------------
# Pipeline completo por combinação
# ---------------------------------------------------------------------------

def select_exog(
    series: pd.Series,
    inputs: pd.DataFrame,
    d: int,
    mode: str,
    vif_threshold: float = 10.0,
    granger_alpha: float = 0.10,
    granger_max_lag: int = 4,
    max_exog: int = 5,
) -> list[str]:
    """VIF → Granger → AIC stepwise.  Retorna lista de colunas selecionadas."""
    # Alinhar inputs com a série
    inp = inputs.reindex(series.index).ffill().bfill().dropna(axis=0)
    if inp.empty or inp.shape[1] == 0:
        return []

    # 1. VIF
    inp_vif = vif_filter(inp, threshold=vif_threshold)
    if inp_vif.empty:
        return []

    # 2. Granger
    gr_cols = granger_filter(series, inp_vif, max_lag=granger_max_lag, alpha=granger_alpha)
    if not gr_cols:
        return []

    # 3. AIC stepwise
    return aic_stepwise_forward(series, gr_cols, inputs, d=d, mode=mode, max_exog=max_exog)


# ---------------------------------------------------------------------------
# Execução para todas as 12 combinações
# ---------------------------------------------------------------------------

def run_feature_selection(
    granger_alpha_override: "dict[int, float] | None" = None,
    output_path: "Path | None" = None,
    manuscript: bool = False,
) -> None:
    """Executa VIF+Granger+AIC para todos os combos. Salva exog_selection.json.

    Args:
        granger_alpha_override: mapa opcional {empresa_code: alpha} para relaxar o
            threshold de Granger por empresa. Empresas não mapeadas usam α=0.10 global.
        output_path: caminho de saída do JSON. Se None, usa outputs/results/exog_selection.json.
        manuscript: Round 41 (protocolo metodológico §32) — o funil lê a série do tratamento do manuscrito
            (`src.treatment.treat_series(manuscript=True)`) em vez da série da dissertação.
    """
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    granger_override = granger_alpha_override or {}
    out_path = output_path or _EXOG_PATH

    results: dict = {}
    header = f"{'Empresa':<8} {'Indicador':<22} {'α':>5} {'VIF':>4} {'Gr':>4} {'Sel':>4}  Colunas selecionadas"
    print("=" * 85)
    print(header)
    print("-" * 85)

    for emp_code, ticker in cfg.EMPRESAS.items():
        results[str(emp_code)] = {}
        alpha = granger_override.get(emp_code, 0.10)
        for ind in cfg.INDICADORES:
            series = datasets[ind][str(emp_code)].dropna()
            if manuscript:
                from src.treatment import treat_series
                series, _ = treat_series(series.astype(float), emp_code, ind, manuscript=True)
            elif cfg.EXOG_SELECTION_ON_TREATED:
                # Round 31: mesma série tratada que o resto do pipeline enxerga (METHODOLOGY §10).
                known = [pd.Timestamp(d) for d in cfg.known_spikes_for(emp_code, ind)]
                series, _ = detect_and_impute_spikes(
                    series.astype(float), z_thresh=cfg.SPIKE_Z_THRESH,
                    method=cfg.SPIKE_IMPUTE_METHOD, known_spikes=known,
                )

            # Determinar d (sobre série raw — conservador)
            try:
                urt = unit_root_tests(series)
                d = urt["consensus_d"]
            except Exception:
                d = 1
            mode = "raw"

            inp = inputs.reindex(series.index).ffill().bfill().dropna(axis=0)

            # Contar VIF survivors e Granger survivors para o relatório
            try:
                inp_vif = vif_filter(inp, threshold=10.0)
                n_vif = inp_vif.shape[1]
            except Exception:
                inp_vif = inp
                n_vif = inp.shape[1]

            try:
                gr_cols = granger_filter(series, inp_vif,
                                         max_lag=4, alpha=alpha)
                n_gr = len(gr_cols)
            except Exception:
                gr_cols = []
                n_gr = 0

            selected = select_exog(series, inp, d=d, mode=mode, granger_alpha=alpha)
            results[str(emp_code)][ind] = selected

            short = [c[:25].encode('ascii', 'replace').decode('ascii') for c in selected[:3]]
            suffix = "..." if len(selected) > 3 else ""
            print(
                f"{ticker:<8} {ind:<22} {alpha:>5.2f} {n_vif:>4} {n_gr:>4} {len(selected):>4}"
                f"  {short}{suffix}"
            )

    print("=" * 85)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[feature_selection] Salvo: {out_path}")


if __name__ == "__main__":
    # Uso: python -m src.feature_selection [saida.json] — sem argumento grava exog_selection.json.
    _out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_feature_selection(granger_alpha_override=cfg.GRANGER_ALPHA_OVERRIDE, output_path=_out)
