"""Inferência sobre a taxa de sincronização (Eq. 5): nulo por permutação e faixa entre sementes.

Motivo (revisão do autor, 2026-09-27, itens 2 e 3). O nível de acaso 100*J^(1-P) assume
estados equiprováveis e o texto já o chama de "lower reference" -- comparação fraca. Este
script substitui isso por dois complementos:

  (a) um nulo EMPÍRICO por permutação, para a taxa observada (a de
      sincronizacao_cross_company.csv / sincronizacao_intra_company.csv, lida diretamente de
      estados_viterbi.csv -- não reajustada, para casar exatamente com a tabela principal);
  (b) a mesma taxa recomputada sob as 10 sementes do estudo de robustez
      (hmm_n64_robustness.py), para reportar uma FAIXA em vez de um único ponto.

Escolha de permutação: deslocamento circular independente de cada sequência decodificada
(`np.roll` por um deslocamento uniforme em [1, K-1]), não blocos embaralhados. As duas
quebram o alinhamento temporal entre séries; a circular preserva a sequência de transições de
cada série *exatamente* (nenhuma fronteira de bloco introduz uma transição artificial), o que
a embaralhagem em blocos só preserva aproximadamente. Sob H0 (sequências independentes) as
duas têm a mesma distribuição assintótica; a circular é a mais defensável das duas para K=63.

Só a variante categórica entra (SYNC_VARIANTS de hmm_n64.py), a mesma restrição da tabela
principal -- é a única cujo estado é comensurável entre séries de escalas diferentes.

    python hmm_models/Model/hmm_n64_sync_inference.py [--n-states-main N] [--n-perm N]

Pré-requisito: rodar `hmm_n64.py --stage base` (e, para a faixa de sementes, também
`--stage robustness`, embora este script refaça o ajuste por semente por conta própria) no
mesmo `--out-dir` primeiro, para que estados_viterbi.csv reflita a topologia corrente.

Saída (out_dir): sincronizacao_permutacao.csv, sincronizacao_por_semente.csv

`--shift-step 4` (2026-10-12, revisão mínima viável): deslocamentos só em múltiplos de 4
trimestres, o que preserva a fase sazonal de cada série sob o nulo -- se a sincronização
sobrevive, ela não é sazonalidade comum. Grava sincronizacao_permutacao_passo4.csv e não refaz a
faixa de sementes. Com K=63 a fase só é preservada exatamente fora do ponto de emenda circular.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_n64 import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    N_STATES_MAIN,
    PAPER_LABELS,
    canonical_map,
    discretize,
    fit_variant,
    load_combos,
    relabel,
)

SEEDS = tuple(range(10))       # mesmas 10 sementes de hmm_n64_robustness.py
N_PERM_DEFAULT = 5000
RNG_SEED = 20260927             # reprodutibilidade do teste de permutação em si (não é dado)


def draw_shift(rng: np.random.Generator, K: int, step: int = 1) -> int:
    """Deslocamento circular em [1, K-1], restrito a múltiplos de `step` (step=1: qualquer)."""
    return step * int(rng.integers(1, (K - 1) // step + 1))


def load_main_states(out_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    """Estados canônicos da especificação principal (categórica), lidos de estados_viterbi.csv
    -- não reajustados, para casar exatamente com sincronizacao_cross_company.csv/intra."""
    df = pd.read_csv(out_dir / "estados_viterbi.csv")
    df = df[df["variante"] == "categorical"].sort_values("data")
    estados: dict[tuple[str, str], np.ndarray] = {}
    for (empresa, indicador), g in df.groupby(["empresa", "indicador"]):
        estados[(empresa, indicador)] = g["estado_canonico"].to_numpy()
    return estados


def fit_states_at_seed_flagged(combos, n_states: int, seed: int):
    """Decodifica a variante categórica para as 12 combinações sob UMA semente do EM (não a
    melhor de 22 reinicializações) -- o que o estudo de sensibilidade a semente varia.

    Devolve também, por combinação, a marca `espurio` do ajuste (estado quase vazio): uma
    inicialização isolada pode colapsar quase toda a série em um estado, e a sincronização
    medida sobre esse colapso é coincidência por construção, não regime compartilhado.
    Regra fixada antes de ver o resultado: grupo/semente com qualquer membro espúrio sai da
    faixa e do nulo do máximo (a taxa continua gravada, com a marca)."""
    estados: dict[tuple[str, str], np.ndarray] = {}
    espurios: dict[tuple[str, str], bool] = {}
    for c in combos:
        codes, _ = discretize(c.x)
        m, _, diag = fit_variant("categorical", codes, n_states, seeds=[seed])
        if m is None:
            continue
        decoded = m.decode(codes)[1]
        mapa = canonical_map(m.emissionprob_ @ np.arange(m.emissionprob_.shape[1]))
        estados[(c.empresa, c.indicador)] = relabel(decoded, mapa)
        espurios[(c.empresa, c.indicador)] = bool(diag["espurio"])
    return estados, espurios


def fit_states_at_seed(combos, n_states: int, seed: int) -> dict[tuple[str, str], np.ndarray]:
    return fit_states_at_seed_flagged(combos, n_states, seed)[0]


def group_has_spurious(nome: str, membros: dict, espurios: dict[tuple[str, str], bool]) -> bool:
    """Verdadeiro se algum membro do grupo veio de um ajuste espúrio nesta semente."""
    grupo_tipo, rotulo = nome.split("::", 1)
    if grupo_tipo == "cross":
        chaves = [(e, rotulo) for e in membros]
    else:
        chaves = [(rotulo, i) for i in membros]
    return any(espurios.get(k, False) for k in chaves)


def build_groups(combos, estados: dict[tuple[str, str], np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Monta os grupos cross-company (por indicador, P=4) e intra-company (por empresa, P=3)."""
    empresas = sorted({c.empresa for c in combos})
    indicadores = sorted({c.indicador for c in combos}, key=lambda i: list(PAPER_LABELS).index(i))
    grupos: dict[str, dict[str, np.ndarray]] = {}
    for indicador in indicadores:
        membros = {e: estados[(e, indicador)] for e in empresas if (e, indicador) in estados}
        if len(membros) >= 2:
            grupos[f"cross::{indicador}"] = membros
    for empresa in empresas:
        membros = {i: estados[(empresa, i)] for i in indicadores if (empresa, i) in estados}
        if len(membros) >= 2:
            grupos[f"intra::{empresa}"] = membros
    return grupos


def sync_rate_arrays(membros: dict[str, np.ndarray]) -> float:
    """Taxa de sincronização a partir de arrays já alinhados (mesmo K para todos os membros)."""
    tamanhos = {len(a) for a in membros.values()}
    if len(tamanhos) != 1:
        raise ValueError(f"séries de tamanhos diferentes no grupo: {tamanhos}")
    m = np.stack(list(membros.values()), axis=1)          # (K, P)
    coinc = np.all(m == m[:, [0]], axis=1).sum()
    return float(coinc) / m.shape[0]


def permutation_test(
    grupos: dict[str, dict[str, np.ndarray]], n_perm: int, rng: np.random.Generator,
    step: int = 1,
) -> pd.DataFrame:
    """Nulo por deslocamento circular independente de cada série (ver docstring do módulo)."""
    linhas = []
    for nome, membros in grupos.items():
        arrays = list(membros.values())
        K = len(arrays[0])
        observado = sync_rate_arrays(membros)

        nulos = np.empty(n_perm)
        for p in range(n_perm):
            deslocados = [np.roll(a, draw_shift(rng, K, step)) for a in arrays]
            m = np.stack(deslocados, axis=1)
            nulos[p] = np.all(m == m[:, [0]], axis=1).sum() / K

        p_valor = (1 + int(np.sum(nulos >= observado))) / (n_perm + 1)
        grupo_tipo, rotulo = nome.split("::", 1)
        linhas.append(
            {
                "grupo": grupo_tipo, "rotulo": rotulo, "P": len(arrays), "K": K,
                "taxa_observada_pct": round(100 * observado, 3),
                "nulo_media_pct": round(100 * float(nulos.mean()), 3),
                "nulo_p95_pct": round(100 * float(np.percentile(nulos, 95)), 3),
                "percentil_observado": round(100 * float((nulos < observado).mean()), 1),
                "p_valor": round(p_valor, 4), "n_permutacoes": n_perm,
                **({"passo_deslocamento": step} if step > 1 else {}),
            }
        )
    return pd.DataFrame(linhas)


def sync_by_seed(combos, n_states: int) -> pd.DataFrame:
    """Taxa de sincronização recomputada sob as 10 sementes do estudo de robustez -- reporta
    a faixa de variação em vez do único ponto estimado na melhor de 22 reinicializações.
    A coluna `espurio` marca grupo/semente com algum membro colapsado; a faixa reportada
    usa só as linhas com espurio=False."""
    linhas = []
    for seed in SEEDS:
        estados, espurios = fit_states_at_seed_flagged(combos, n_states, seed)
        for nome, membros in build_groups(combos, estados).items():
            grupo_tipo, rotulo = nome.split("::", 1)
            taxa = sync_rate_arrays(membros)
            linhas.append(
                {"semente": seed, "grupo": grupo_tipo, "rotulo": rotulo,
                 "P": len(membros), "taxa_pct": round(100 * taxa, 3),
                 "espurio": group_has_spurious(nome, membros, espurios)}
            )
        print(f"  semente {seed} ok")
    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-states-main", type=int, default=N_STATES_MAIN)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--shift-step", type=int, default=1)
    args = ap.parse_args()

    combos, _ = load_combos(args.data_dir)
    rng = np.random.default_rng(RNG_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[permutação] especificação principal (lida de estados_viterbi.csv, "
          f"N={args.n_states_main}), {args.n_perm} permutações por grupo")
    estados_principal = load_main_states(args.out_dir)
    grupos = build_groups(combos, estados_principal)
    perm = permutation_test(grupos, args.n_perm, rng, args.shift_step)
    sufixo = f"_passo{args.shift_step}" if args.shift_step > 1 else ""
    perm.to_csv(args.out_dir / f"sincronizacao_permutacao{sufixo}.csv", index=False)
    print(perm.to_string(index=False))
    if args.shift_step > 1:
        return 0

    print(f"\n[sementes] taxa de sincronização sob {len(SEEDS)} sementes do EM")
    por_semente = sync_by_seed(combos, args.n_states_main)
    por_semente.to_csv(args.out_dir / "sincronizacao_por_semente.csv", index=False)

    validas = por_semente[~por_semente["espurio"]]
    resumo = (validas.groupby(["grupo", "rotulo"])["taxa_pct"]
              .agg(["count", "mean", "min", "max", "std"]).round(3).reset_index())
    print(resumo.to_string(index=False))

    print(f"\nartefatos em {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
