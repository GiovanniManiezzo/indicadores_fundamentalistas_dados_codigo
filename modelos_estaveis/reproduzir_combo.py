#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduz o campeao de UMA combinacao empresa x indicador e salva o grafico.

Nao duplica a logica de reconstrucao de modelo: reusa `plot_best_results.py`, que por sua vez
le a tabela de campeoes de `scripts/_common.compute_champions()`. Antes do Round 25 este arquivo
tinha sua propria copia de `BEST_CONFIGS`, que divergia tanto do painel quanto do script de plot
— tres tabelas de campeao para manter em sincronia manualmente.

Uso:
    python modelos_estaveis/reproduzir_combo.py --empresa ISAE4 --indicador resultado_bruto
    python modelos_estaveis/reproduzir_combo.py --todos
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config as cfg
import plot_best_results as pbr
from _common import compute_champions
from src.loaders import load_all
from src.evaluation import rmse

warnings.filterwarnings("ignore")
cfg.apply_global_seed()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TICKER_TO_CODE = {v: k for k, v in cfg.EMPRESAS.items()}


def reproduce(emp_code: int, ind: str, datasets: dict, inputs, plots_dir: str) -> bool:
    """Reexecuta o campeao, imprime metricas e salva o grafico. True se reproduziu."""
    ticker = cfg.EMPRESAS[emp_code]
    champ = compute_champions()[(emp_code, ind)]
    if champ is None:
        print(f"{ticker} x {ind}: sem resultado registrado.")
        return False

    res = pbr.run_combo(emp_code, ind, champ, datasets, inputs)
    delta = res["fit_actual"] - res["fit_stored"]
    ok = abs(delta) <= pbr.FIT_TOLERANCE_PP

    print(f"\n{'=' * 72}")
    print(f"{ticker} x {ind}")
    print(f"{'=' * 72}")
    print(f"  Modelo          : {res['model']}  —  {res['desc']}")
    print(f"  Pre-processo    : {res['mode']}")
    print(f"  Exogenas        : {', '.join(res['exog']) if res['exog'] else '(nenhuma)'}")
    print(f"  Fonte           : {res['file']}  (era de tratamento: "
          f"R{cfg.TREATMENT_ERA[(emp_code, ind)]})")
    revert = cfg.SPIKE_REVERTED.get((emp_code, ind))
    if revert:
        print(f"  Tratamento revertido: {', '.join(revert)}  (protocolo metodológico sec.27)")
    print(f"  Spikes tratados : "
          f"{', '.join(cfg.known_spikes_for(emp_code, ind)) or '(so deteccao automatica z>5)'}")
    print(f"  ---")
    print(f"  FIT% gravado    : {res['fit_stored']:.2f}")
    print(f"  FIT% recalculado: {res['fit_actual']:.2f}   (delta {delta:+.2f} p.p.) "
          f"{'OK' if ok else '<-- DIVERGENCIA'}")
    print(f"  Theil U2        : {res['u2']:.3f}  ({'passa' if res['u2'] < 1 else 'FALHA'} o gate)")
    dm = res["dm_p"]
    print(f"  DM-p vs ARIMA   : {dm:.3f}" if isinstance(dm, (int, float)) else "  DM-p vs ARIMA   : -")
    print(f"  RMSE            : {rmse(res['y_true'], res['y_pred']):.4g}")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    pbr._draw(ax, res, ticker, ind, compact=False)
    ax.legend(handles=[
        mpatches.Patch(color="#999999", label="Real (serie completa)"),
        mpatches.Patch(color="#222222", label="Real (out-of-sample)"),
        mpatches.Patch(color=pbr.MODEL_COLORS.get(res["family"], "#555"),
                       label=f"Previsto — {res['model']}"),
        mpatches.Patch(color="#cccccc", label=f"Treino inicial ({cfg.MIN_TRAIN} obs)"),
    ], fontsize=8, loc="best", framealpha=0.9)
    out = os.path.join(plots_dir, f"{ticker}_{ind}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico         : {out}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--empresa", choices=sorted(TICKER_TO_CODE))
    ap.add_argument("--indicador", choices=cfg.INDICADORES)
    ap.add_argument("--todos", action="store_true", help="reproduz as 12 combinacoes")
    args = ap.parse_args()

    if not args.todos and not (args.empresa and args.indicador):
        ap.error("informe --empresa e --indicador, ou --todos")

    plots_dir = str(paths.PLOTS_DIR)
    os.makedirs(plots_dir, exist_ok=True)

    print("Carregando dados...")
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc,
                "despesas_operacionais": do}

    targets = (cfg.COMBOS if args.todos
               else [(TICKER_TO_CODE[args.empresa], args.indicador)])
    failures = [f"{cfg.EMPRESAS[e]}x{i}" for e, i in targets
                if not reproduce(e, i, datasets, inputs, plots_dir)]

    if failures:
        print(f"\n[!] Nao reproduziram o valor gravado: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
