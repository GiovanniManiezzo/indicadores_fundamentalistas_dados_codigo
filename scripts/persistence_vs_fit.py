"""Reproduz a tabela persistência x previsibilidade que sustenta a sintese do artigo.

Motivo: a tabela — e com ela a correlacao FIT% x rho_1 — era montada a mao; este script a
regenera e a verifica com --check.

Definicoes, iguais as usadas para montar a tabela original:
  rho_1        `pandas.Series.autocorr(1)` sobre a serie **em nivel**, ja tratada
               (imputacao de spikes do pipeline: `cfg.known_spikes_for()` + z > SPIKE_Z_THRESH).
  trocas_sinal numero de trocas de sinal da serie tratada em nivel.
  FIT%, U2     do campeao vigente de cada combinacao (`_common.compute_champions()`,
               gate estrito U2<1, restrito a era de tratamento corrente — Invariante 10).

    python scripts/persistence_vs_fit.py            # tabela markdown em stdout + CSV
    python scripts/persistence_vs_fit.py --check    # nao escreve; sai !=0 se o CSV derivou

Saida: paths.JOURNAL_PERSISTENCE_CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import paths  # noqa: E402
from _common import compute_champions  # noqa: E402
from export_series_tratadas import treated_frame  # noqa: E402


def sign_changes(serie: pd.Series) -> int:
    """Trocas de sinal da serie em nivel (zeros nao contam como troca)."""
    s = np.sign(serie.to_numpy(dtype=float))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def fit_ar1(rho1: float) -> float:
    """FIT% implicado por um AR(1) de razão rho1, sob previsão um-passo-a-frente.

    Sob um AR(1) estacionário y_t = rho1 * y_{t-1} + e_t, o previsor ótimo tem
    variância de erro sigma_e^2 e a variância da própria série é sigma_e^2 / (1 - rho1^2),
    de modo que a razão de normas da Eq. (4) do artigo (FIT%) se reduz a
    sqrt(1 - rho1^2) por construção — não é uma descoberta empírica, é a álgebra do AR(1).
    """
    return float(100.0 * (1.0 - np.sqrt(max(0.0, 1.0 - rho1 ** 2))))


def _manuscript_champions() -> dict:
    """FIT%, U2 e modelo do campeão reeleito sob o tratamento do manuscrito (Round 41,
    protocolo metodológico §32), lidos de `ar1_walkforward.csv` — gerado por `ar1_walkforward.py`, que
    reproduz o campeão com a mesma máscara de alvos imputados. Rode-o antes."""
    ar1 = pd.read_csv(paths.JOURNAL_AR1_WALKFORWARD_CSV)
    inv = {v: k for k, v in cfg.EMPRESAS.items()}
    return {
        (inv[r.empresa], r.indicador): {
            "fit": float(r.fit_campeao), "u2": float(r.u2_campeao), "model": r.modelo,
            "gate_passed": bool(r.u2_campeao < 1.0),
        }
        for r in ar1.itertuples()
    }


def build_table(manuscript: bool = True) -> pd.DataFrame:
    """`manuscript=True`: série tratada e campeão do Round 40; `False`: tratamento vigente."""
    champions = _manuscript_champions() if manuscript else compute_champions()
    linhas: list[dict] = []

    for indicador in cfg.INDICADORES:
        tratada, _ = treated_frame(indicador, manuscript=manuscript)
        for code, empresa in cfg.EMPRESAS.items():
            serie = tratada[str(code)].astype(float).dropna()
            champ = champions.get((code, indicador))
            rho1 = float(serie.autocorr(1))
            fit_pct = None if champ is None else float(champ["fit"])
            f_ar1 = fit_ar1(rho1)
            linhas.append({
                "empresa": empresa,
                "indicador": indicador,
                "rho1": rho1,
                "trocas_sinal": sign_changes(serie),
                "fit_pct": fit_pct,
                "fit_ar1": f_ar1,
                "delta": None if fit_pct is None else float(fit_pct - f_ar1),
                "theil_u2": None if champ is None else float(champ["u2"]),
                "modelo": None if champ is None else champ["model"],
                "gate_u2": None if champ is None else bool(champ["gate_passed"]),
            })

    df = pd.DataFrame(linhas)
    return df.sort_values("fit_pct", ascending=False, ignore_index=True)


# Distância contábil (ordinal, 0/1/2) à receita regulada — ver Section 4.3 do artigo.
# Rótulo, não medida: só sustenta Spearman, n=12.
DISTANCIA_INDICADOR = {"resultado_bruto": 0, "lucro_consolidado": 1, "despesas_operacionais": 2}


def correlations(df: pd.DataFrame) -> dict[str, float]:
    ok = df.dropna(subset=["fit_pct", "rho1"])
    okd = ok.dropna(subset=["delta"])
    distancia = okd["indicador"].map(DISTANCIA_INDICADOR)
    return {
        "pearson_fit_rho1": float(stats.pearsonr(ok["fit_pct"], ok["rho1"])[0]),
        "spearman_fit_rho1": float(stats.spearmanr(ok["fit_pct"], ok["rho1"])[0]),
        "kendall_fit_rho1": float(stats.kendalltau(ok["fit_pct"], ok["rho1"])[0]),
        "pares_discordantes": int(sum(
            1 for i in range(len(ok)) for j in range(i + 1, len(ok))
            if (ok["fit_pct"].iloc[i] - ok["fit_pct"].iloc[j])
            * (ok["rho1"].iloc[i] - ok["rho1"].iloc[j]) < 0)),
        "pares_total": len(ok) * (len(ok) - 1) // 2,
        "pearson_fit_trocas": float(stats.pearsonr(ok["fit_pct"], ok["trocas_sinal"])[0]),
        # P0.3: o mesmo painel, lido pelo benchmark que o artigo já adota como gate (U2 < 1,
        # Random Walk) em vez do benchmark implícito do FIT% (a média amostral).
        "pearson_rho1_u2": float(stats.pearsonr(ok["rho1"], ok["theil_u2"])[0]),
        "spearman_fit_u2": float(stats.spearmanr(ok["fit_pct"], ok["theil_u2"])[0]),
        # P0.2: o excesso sobre o teto implicado pelo AR(1) — onde a modelagem paga.
        "spearman_delta_distancia": float(stats.spearmanr(okd["delta"], distancia)[0]),
        "n": int(len(ok)),
    }


def as_markdown(df: pd.DataFrame, corr: dict[str, float]) -> str:
    linhas = [
        "| Combinação | rho_1 | Trocas de sinal | FIT% | FIT%_AR(1) | Delta | U2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples():
        fit = "—" if pd.isna(r.fit_pct) else f"{r.fit_pct:.1f}"
        f_ar1 = f"{r.fit_ar1:.1f}"
        delta = "—" if pd.isna(r.delta) else f"{r.delta:+.1f}"
        u2 = "—" if pd.isna(r.theil_u2) else f"{r.theil_u2:.3f}"
        linhas.append(
            f"| {r.empresa}x{r.indicador} | {r.rho1:.2f} | {r.trocas_sinal} | {fit} | "
            f"{f_ar1} | {delta} | {u2} |"
        )
    linhas.append("")
    linhas.append(
        f"FIT% x rho_1: Pearson {corr['pearson_fit_rho1']:.3f}, "
        f"Spearman {corr['spearman_fit_rho1']:.3f} (n={corr['n']}). "
        f"Kendall {corr['kendall_fit_rho1']:.3f} "
        f"({corr['pares_total'] - corr['pares_discordantes']} pares concordantes de "
        f"{corr['pares_total']}). "
        f"FIT% x trocas de sinal: Pearson {corr['pearson_fit_trocas']:.3f}."
    )
    linhas.append(
        f"rho_1 x U2: Pearson {corr['pearson_rho1_u2']:.3f}. "
        f"FIT% x U2: Spearman {corr['spearman_fit_u2']:.3f} "
        "(sinal oposto ao de FIT% x rho_1 — a ordenação por FIT% não sobrevive ao "
        "benchmark do passeio aleatório)."
    )
    linhas.append(
        f"Delta (FIT% observado - FIT%_AR(1)) x distância contábil (0/1/2, ordinal): "
        f"Spearman {corr['spearman_delta_distancia']:.3f}."
    )
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="nao escreve; compara com o CSV em disco e sai !=0 na divergencia")
    ap.add_argument("--tratamento", choices=("manuscrito", "vigente"), default="manuscrito",
                    help="'vigente' reproduz o tratamento anterior ao Round 40 e não escreve")
    args = ap.parse_args()

    df = build_table(manuscript=args.tratamento == "manuscrito")
    corr = correlations(df)
    print(as_markdown(df, corr))
    if args.tratamento == "vigente":
        return

    destino = paths.JOURNAL_PERSISTENCE_CSV
    if args.check:
        if not destino.exists():
            print(f"\nFALHA: {destino.name} nao existe.", file=sys.stderr)
            sys.exit(1)
        if destino.read_text(encoding="utf-8") != df.to_csv(index=False, lineterminator="\n"):
            print(f"\nFALHA: {destino.name} divergiu do recomputado.", file=sys.stderr)
            sys.exit(1)
        print(f"\nOK: {destino.name} reproduz.")
        return

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(df.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    print(f"\nGravado: {destino.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
