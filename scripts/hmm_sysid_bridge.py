"""Testa se alguma quantidade da metade HMM prediz a previsibilidade da metade sysid.

Motivo: testar a hipotese de que os regimes detectados pelo HMM explicam *por que* a
previsibilidade varia entre as 12 combinacoes. Este script cruza, combinacao a
combinacao, os descritores de regime produzidos em N=64 com FIT%/rho_1 do painel.

Descritores testados (todos vindos de `results/hmm_n64/`):
  diag_medio      media da diagonal da matriz de transicao (persistencia do regime)
  duracao_media   duracao esperada de regime, 1/(1 - diag_medio)
  trocas_viterbi  numero de trocas de estado na sequencia Viterbi
  ARI_sementes    concordancia par-a-par media entre 10 inicializacoes do EM
  particoes_id    %% de pares de sementes que chegam a mesma particao
  espurias        numero de celulas (variante x N) descartadas como maximo espurio

Alvos: fit_pct, rho1, trocas_sinal (de `scripts/persistence_vs_fit.py`).

    python scripts/hmm_sysid_bridge.py

Saida: paths.JOURNAL_BRIDGE_CSV, uma linha por (descritor, variante, alvo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402

KEY = ["empresa", "indicador"]
ALVOS = ["fit_pct", "rho1", "trocas_sinal"]


def _hmm(nome: str) -> pd.DataFrame:
    return pd.read_csv(paths.JOURNAL_HMM_DIR / f"{nome}.csv")


def descritores() -> dict[tuple[str, str], pd.Series]:
    """(descritor, variante) -> serie indexada por (empresa, indicador)."""
    tm, vit = _hmm("matrizes_transicao"), _hmm("estados_viterbi")
    sem, conv = _hmm("robustez_semente"), _hmm("convergencia")
    out: dict[tuple[str, str], pd.Series] = {}

    for var in sorted(tm.variante.unique()):
        t = tm[tm.variante == var]
        diag = t[t.de_estado == t.para_estado].groupby(KEY).prob.mean()
        out[("diag_medio", var)] = diag
        out[("duracao_media", var)] = 1.0 / (1.0 - diag.clip(upper=0.999999))
        out[("trocas_viterbi", var)] = (
            vit[vit.variante == var].sort_values("data").groupby(KEY).estado_canonico
            .apply(lambda s: int((s.to_numpy()[1:] != s.to_numpy()[:-1]).sum()))
        )

    for var in sorted(sem.variante.unique()):
        s = sem[sem.variante == var].set_index(KEY)
        out[("ARI_sementes", var)] = s.ARI_par_a_par_medio
        out[("particoes_id", var)] = s.particoes_identicas_pct
    out[("ARI_sementes", "media")] = sem.groupby(KEY).ARI_par_a_par_medio.mean()
    out[("particoes_id", "media")] = sem.groupby(KEY).particoes_identicas_pct.mean()

    out[("espurias", "todas")] = conv.groupby(KEY).espurio.sum().astype(float)
    return out


def main() -> None:
    painel = pd.read_csv(paths.JOURNAL_PERSISTENCE_CSV).set_index(KEY)
    linhas: list[dict] = []

    for (descritor, variante), serie in descritores().items():
        m = painel.join(serie.rename("x")).dropna(subset=["x"])
        if m["x"].nunique() < 3:
            continue
        for alvo in ALVOS:
            pr, pp = stats.pearsonr(m["x"].astype(float), m[alvo].astype(float))
            sr, sp = stats.spearmanr(m["x"].astype(float), m[alvo].astype(float))
            linhas.append({
                "descritor": descritor, "variante": variante, "alvo": alvo, "n": len(m),
                "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
                "significativo_5pct": bool(pp < 0.05 or sp < 0.05),
            })

    df = pd.DataFrame(linhas).sort_values(["alvo", "descritor", "variante"], ignore_index=True)
    paths.JOURNAL_BRIDGE_CSV.write_text(
        df.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    fit = df[df.alvo == "fit_pct"]
    print(fit.to_string(index=False,
          columns=["descritor", "variante", "n", "pearson_r", "pearson_p",
                   "spearman_rho", "spearman_p"], float_format=lambda v: f"{v:+.3f}"))
    n_sig = int(df.significativo_5pct.sum())
    print(f"\n{n_sig} de {len(df)} testes significativos a 5%.")
    print(f"Gravado: {paths.JOURNAL_BRIDGE_CSV.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
