"""Referências sem parâmetro estimado nas doze combinações, na série e na máscara do manuscrito.

Motivo (2026-10-17): o painel não reportava as referências
obrigatórias para séries contábeis trimestrais — o sazonal ingênuo y(k-4) e a média histórica
expansível —, nem o FIT% do próprio passeio aleatório, que reproduz a ordenação por rho_1 sem
estimar nada. Também pede rho_1 sem a imputação por y(k-1), para mostrar que ela não fabricou a
persistência: rho_1 com as mesmas células interpoladas linearmente (a imputação que não copia o
vizinho anterior) e rho_1 de postos (Spearman, defasagem 1) na série registrada, robusto aos picos
que a imputação remove — o rho_1 de Pearson da série registrada é dominado por um único pico (a
indenização RBSE de 2016 na ISA Energy, p.ex.) e é reportado só para completude. Nada aqui é
ajustado: as quatro referências só leem o passado.

Série e pontuação do tratamento do manuscrito (protocolo metodológico §31): série de
`src.treatment.treat_series(manuscript=True)`, alvos imputados fora de todas as métricas, a
mesma máscara do painel. U2 contra o passeio aleatório sobre a série tratada, como no painel.
O FIT% do campeão vem dos campeões reeleitos do Round 41 (`_common.manuscript_champions`).

    python scripts/simple_benchmarks.py            # tabela em stdout + CSV
    python scripts/simple_benchmarks.py --check    # não escreve; sai != 0 se o CSV derivou

Extensão de 2026-10-18: a metade de regimes diferencia as doze séries, inclusive as
quatro com d = 0. `rho1_diff` (autocorrelação de defasagem 1 da primeira diferença da série
tratada) e `razao_var_diff` = Var(Dy)/Var(y) quantificam a sobrediferenciação: para ruído branco
diferenciado, rho1 = -0,5 e a razão é 2. `d` é o consenso de `preprocess.unit_root_tests`.

Saída: paths.JOURNAL_SIMPLE_BENCHMARKS_CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as cfg  # noqa: E402
import paths  # noqa: E402
from _common import manuscript_champions  # noqa: E402
from src.loaders import load_all  # noqa: E402
from src.preprocess import detect_and_impute_spikes, unit_root_tests  # noqa: E402
from src.treatment import treat_series  # noqa: E402


def _fit(y: np.ndarray, p: np.ndarray) -> float:
    den = np.linalg.norm(y - y.mean())
    return float((1 - np.linalg.norm(y - p) / den) * 100) if den else 0.0


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def build_table() -> pd.DataFrame:
    rb, lc, do, _ = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    try:
        champions = manuscript_champions()
    except FileNotFoundError:
        champions = {}

    linhas = []
    for code, empresa in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            bruto = datasets[ind][str(code)].dropna().astype(float)
            registrado = bruto.copy()
            for date, value in cfg.Q4_REBASE_OVERRIDES.get((code, ind), {}).items():
                registrado.loc[pd.Timestamp(date)] = value
            serie, recs = treat_series(bruto, code, ind, manuscript=True)
            imputados = {pd.Timestamp(r["timestamp"]) for r in recs}

            y = serie.to_numpy()
            k0, n = cfg.MIN_TRAIN, len(y)
            alvos = [t for t in range(k0, n) if serie.index[t] not in imputados]
            yt = y[alvos]
            prev = {
                "rw": np.array([y[t - 1] for t in alvos]),
                "rw_drift": np.array([y[t - 1] + (y[t - 1] - y[0]) / (t - 1) for t in alvos]),
                "sazonal": np.array([y[t - 4] for t in alvos]),
                "media": np.array([y[:t].mean() for t in alvos]),
            }
            rmse_rw = _rmse(yt, prev["rw"])
            champ = champions.get((code, ind))
            conhecidos = [pd.Timestamp(d) for d in
                          cfg.known_spikes_for(code, ind, include_reverted=True)]
            brutas = (list(cfg.FIXED_RULE_RAW_CELLS.get((code, ind), []))
                      + list(cfg.Q4_REBASE_OVERRIDES.get((code, ind), {})))
            interp, _ = detect_and_impute_spikes(
                registrado, z_thresh=cfg.SPIKE_Z_THRESH, method="linear_interp",
                known_spikes=conhecidos or None, exclude=brutas or None)
            linha = {"empresa": empresa, "indicador": ind, "n_pontuados": len(alvos),
                     "rho1_tratada": float(serie.autocorr(1)),
                     "rho1_interpolada": float(interp.autocorr(1)),
                     "rho1_postos_registrada": float(registrado.rank().autocorr(1)),
                     "rho1_registrada": float(registrado.autocorr(1)),
                     "d": int(unit_root_tests(serie)["consensus_d"]),
                     "rho1_diff": float(serie.diff().dropna().autocorr(1)),
                     "razao_var_diff": float(serie.diff().var() / serie.var())}
            for nome, p in prev.items():
                linha[f"fit_{nome}"] = _fit(yt, p)
                linha[f"u2_{nome}"] = _rmse(yt, p) / rmse_rw if rmse_rw else float("nan")
            linha["fit_campeao"] = None if champ is None else float(champ["fit"])
            linha["u2_campeao"] = None if champ is None else float(champ["u2"])
            linhas.append(linha)
    return pd.DataFrame(linhas)


def as_markdown(df: pd.DataFrame) -> str:
    out = ["| Combinação | rho1 trat. | rho1 interp. | rho1 postos reg. | FIT% RW | FIT% RW+drift "
           "| FIT% sazonal (U2) | FIT% média (U2) | FIT% campeão |",
           "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for r in df.itertuples():
        champ = "—" if pd.isna(r.fit_campeao) else f"{r.fit_campeao:.1f}"
        out.append(
            f"| {r.empresa}x{r.indicador} | {r.rho1_tratada:.2f} | {r.rho1_interpolada:.2f} | "
            f"{r.rho1_postos_registrada:.2f} | "
            f"{r.fit_rw:.1f} | {r.fit_rw_drift:.1f} | {r.fit_sazonal:.1f} ({r.u2_sazonal:.3f}) | "
            f"{r.fit_media:.1f} ({r.u2_media:.3f}) | {champ} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="não escreve; compara com o CSV em disco e sai != 0 na divergência")
    args = ap.parse_args()

    df = build_table()
    print(as_markdown(df))
    destino = paths.JOURNAL_SIMPLE_BENCHMARKS_CSV
    texto = df.to_csv(index=False, lineterminator="\n")
    if args.check:
        if not destino.exists() or destino.read_text(encoding="utf-8") != texto:
            print(f"\nFALHA: {destino.name} ausente ou divergiu do recomputado.", file=sys.stderr)
            sys.exit(1)
        print(f"\nOK: {destino.name} reproduz.")
        return
    destino.write_text(texto, encoding="utf-8")
    print(f"\nGravado: {destino.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
