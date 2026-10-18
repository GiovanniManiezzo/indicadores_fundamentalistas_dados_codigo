"""Diagnóstico expandido de outliers — Passos 1 e 2 (protocolo metodológico §11.2/11.3).

Passo 1: Ajusta ARIMA (pós-imputação Round 04) e flagga resíduos |z| > OUTLIER_DIAG_Z_THRESH.
Passo 2: Classifica cada outlier detectado como AO, LS, TC ou IO.

Uso:
    python scripts/check_outliers.py [--save] [--log]

Opções:
    --save   Salva outputs/results/outlier_diag_<timestamp>.json
    --log    Appenda entrada de Round 05 em rounds/science_log.md
"""
from __future__ import annotations
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import paths
from src.loaders import load_all
from src.preprocess import (
    detect_and_impute_spikes,
    transform,
    detect_by_residuals,
    classify_outlier_type,
)

# Melhor pré-processamento por combinação (Round 04)
_BEST_PREPROC: dict[tuple[int, str], str] = {
    (2437,  "resultado_bruto"):       "raw",
    (2437,  "lucro_consolidado"):     "raw",
    (2437,  "despesas_operacionais"): "raw",
    (18376, "resultado_bruto"):       "log+diff",
    (18376, "lucro_consolidado"):     "raw",
    (18376, "despesas_operacionais"): "raw",
    (20257, "resultado_bruto"):       "diff",
    (20257, "lucro_consolidado"):     "diff",
    (20257, "despesas_operacionais"): "raw",
    (21490, "resultado_bruto"):       "raw",
    (21490, "lucro_consolidado"):     "raw",
    (21490, "despesas_operacionais"): "raw",
}

_IND_SHORT = {
    "resultado_bruto": "res_bruto",
    "lucro_consolidado": "luc_consol",
    "despesas_operacionais": "desp_oper",
}


def _impute(series: pd.Series, emp_code: int, ind: str) -> tuple[pd.Series, list[dict]]:
    """Aplica imputação Round 04: z-thresh automático + SPIKE_KNOWN."""
    known = config.SPIKE_KNOWN.get((emp_code, ind), [])
    return detect_and_impute_spikes(
        series,
        z_thresh=config.SPIKE_Z_THRESH,
        method=config.SPIKE_IMPUTE_METHOD,
        known_spikes=known,
    )


def _recommend(classified: list[dict]) -> str:
    new_types = {c["type"] for c in classified if not c["already_imputed"]}
    if not new_types:
        return "none"
    if "LS" in new_types:
        return "add_ls_dummy"
    if "TC" in new_types:
        return "add_tc_dummy_or_impute"
    return "impute_ao"


def run_diagnostic(*, save: bool = False, log: bool = False) -> dict:
    """Executa diagnóstico expandido para todas as 12 combinações.

    Retorna o dict completo de resultados (também salvo em JSON se save=True).
    """
    config.apply_global_seed()
    rb, lc, do, _ = load_all()
    ind_frames = {
        "resultado_bruto": rb,
        "lucro_consolidado": lc,
        "despesas_operacionais": do,
    }

    output: dict = {
        "meta": {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "protocol": "residual_based_outlier_detection",
            "parameters": {
                "z_thresh_series":    config.SPIKE_Z_THRESH,
                "z_thresh_residuals": config.OUTLIER_DIAG_Z_THRESH,
                "level_shift_window": config.OUTLIER_LS_WINDOW,
                "tc_decay_window":    config.OUTLIER_TC_WINDOW,
                "ls_threshold_factor": config.OUTLIER_LS_THRESH_FACTOR,
            },
        },
        "combinations": {},
    }

    hdr = f"{'Combinação':<32} {'ARIMA':>14} {'z>3':>5} {'Novo':>5}  Novos outliers"
    print(hdr)
    print("─" * 95)

    new_ls_global:    list[str] = []
    new_other_global: list[str] = []

    for emp_code, ticker in config.EMPRESAS.items():
        for ind in config.INDICADORES:
            df = ind_frames[ind]
            col = str(emp_code)
            if col not in df.columns:
                continue

            series_raw = df[col].dropna()

            # --- Pré-processamento: imputação Round 04 ---
            series_clean, imputed_records = _impute(series_raw, emp_code, ind)
            already_imputed: set[str] = {
                str(pd.Timestamp(r["timestamp"]).date()) for r in imputed_records
            }

            # --- Transformação ---
            preproc = _BEST_PREPROC.get((emp_code, ind), "raw")
            try:
                series_tr = transform(series_clean, preproc)
            except ValueError:
                preproc = "raw"
                series_tr = series_clean.copy()

            # --- Passo 1: detecção por resíduos ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                residuals, flagged = detect_by_residuals(
                    series_tr,
                    z_thresh=config.OUTLIER_DIAG_Z_THRESH,
                    p_max=3,
                    q_max=3,
                )

            sigma_e = float(residuals.std())

            # --- Passo 2: classificação ---
            classified: list[dict] = []
            new_outliers: list[str] = []
            for flag in flagged:
                ts = flag["timestamp"]
                ts_str = str(pd.Timestamp(ts).date())
                otype = classify_outlier_type(
                    residuals, ts,
                    sigma=sigma_e,
                    level_shift_window=config.OUTLIER_LS_WINDOW,
                    tc_decay_window=config.OUTLIER_TC_WINDOW,
                    ls_threshold_factor=config.OUTLIER_LS_THRESH_FACTOR,
                )
                is_new = ts_str not in already_imputed
                classified.append({
                    "timestamp": ts_str,
                    "z_residual": round(flag["z_residual"], 3),
                    "type": otype,
                    "already_imputed": not is_new,
                })
                if is_new:
                    label = f"{ts_str}({otype})"
                    new_outliers.append(label)
                    if otype == "LS":
                        new_ls_global.append(f"{ticker}×{_IND_SHORT[ind]}: {ts_str}")
                    else:
                        new_other_global.append(f"{ticker}×{_IND_SHORT[ind]}: {label}")

            arima_order = flagged[0]["arima_order"] if flagged else None
            order_str = f"({','.join(map(str, arima_order))})" if arima_order else "N/A"
            combo_label = f"{ticker}×{_IND_SHORT[ind]}"
            new_str = ", ".join(new_outliers) if new_outliers else "—"
            print(f"{combo_label:<32} {order_str:>14} {len(flagged):>5} {len(new_outliers):>5}  {new_str}")

            output["combinations"][f"{emp_code}_{ind}"] = {
                "empresa": ticker,
                "indicador": ind,
                "preproc": preproc,
                "arima_order": list(arima_order) if arima_order else None,
                "already_imputed": sorted(already_imputed),
                "n_flagged_step1": len(flagged),
                "step2_classified": classified,
                "new_outliers": new_outliers,
                "action_recommended": _recommend(classified),
            }

    print("─" * 95)
    print()

    if new_ls_global:
        print("Level Shifts novos (candidatos a dummy binária LS):")
        for s in new_ls_global:
            print(f"  → {s}")
    else:
        print("Nenhum Level Shift novo detectado além dos já imputados.")

    if new_other_global:
        print("\nAO/TC novos (candidatos a imputação adicional ou dummy TC):")
        for s in new_other_global:
            print(f"  → {s}")

    if save:
        ts_now = datetime.now().strftime("%Y%m%d%H%M")
        paths.ensure_output_dirs()
        out_path = paths.RESULTS_DIR / f"outlier_diag_{ts_now}.json"
        out_path.write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSalvo em: {out_path}")

    if log:
        _append_science_log(output, new_ls_global, new_other_global)

    return output


def _append_science_log(
    output: dict,
    new_ls: list[str],
    new_other: list[str],
) -> None:
    log_path = paths.SCIENCE_LOG
    today = datetime.now().strftime("%Y-%m-%d")
    params = output["meta"]["parameters"]
    combos = output["combinations"]

    n_new_total = sum(len(v["new_outliers"]) for v in combos.values())
    n_ls = len(new_ls)
    n_ao_tc = len(new_other)

    # Monta tabela apenas com outliers novos
    rows = []
    for v in combos.values():
        label = f"{v['empresa']}×{_IND_SHORT[v['indicador']]}"
        for c in v["step2_classified"]:
            if not c["already_imputed"]:
                rows.append(
                    f"| {label:<28} | {c['timestamp']} "
                    f"| {c['z_residual']:>7.3f} | {c['type']:>4} "
                    f"| {v['action_recommended']} |"
                )

    if rows:
        table = (
            "| Combinação                   | Timestamp  |  z_res  | Tipo | Ação |\n"
            "|---|---|---|---|---|\n"
            + "\n".join(rows)
        )
    else:
        table = "Nenhum outlier novo detectado."

    ls_summary    = "; ".join(new_ls)    if new_ls    else "Nenhum"
    other_summary = "; ".join(new_other) if new_other else "Nenhum"
    ts_file = datetime.now().strftime("%Y%m%d%H%M")

    entry = (
        f"\n## Round 05 — Diagnóstico Expandido de Outliers — {today}\n"
        f"**Hipótese:** Outliers não capturados pelo z-score bruto (thresh={params['z_thresh_series']}) "
        f"são detectáveis via resíduos ARIMA (thresh={params['z_thresh_residuals']}) "
        f"e classificáveis por tipologia AO/LS/TC para tratamento diferenciado.\n"
        f"**Protocolo:** Passo 1 — resíduos ARIMA grid(p,q∈[0,3]); "
        f"Passo 2 — AO/LS/TC (LS_win={params['level_shift_window']}, "
        f"TC_win={params['tc_decay_window']}, α={params['ls_threshold_factor']}).\n"
        f"{table}\n"
        f"**Level Shifts novos:** {ls_summary}\n"
        f"**AO/TC novos:** {other_summary}\n"
        f"**Decisões:** {n_new_total} outlier(s) novo(s) — {n_ls} LS, {n_ao_tc} AO/TC. "
        f"Ação: incorporar tratamentos recomendados e re-executar round completo.\n"
        f"**Próximo:** Round 06 — aplicar dummies LS/TC ou imputação e avaliar impacto em FIT% e U₂.\n"
        f"*JSON: outlier_diag_{ts_file}.json*\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Entrada appendada em {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true", help="Salvar JSON de resultados")
    ap.add_argument("--log",  action="store_true", help="Appendar entrada em science_log.md")
    args = ap.parse_args()
    run_diagnostic(save=args.save, log=args.log)
