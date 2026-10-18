"""Nulo do MAXIMO da taxa de sincronizacao sobre as sementes do EM.

Pergunta (autor, 2026-10-11): se cada grupo pudesse escolher a melhor das 10 sementes, a taxa
maxima obtida excede o que o acaso produz ao escolher a melhor de 10? Escolher a semente pela
sincronizacao e selecao sobre o desfecho; o p-valor de hmm_n64_sync_inference.py, calculado
para UM ajuste, nao vale para o maximo. Este script da o p-valor correto para o maximo.

Estatistica: M_g = max_s taxa_s(g), s sobre as 10 sementes de SEEDS (uma inicializacao por
semente, como em hmm_n64_sync_inference.sync_by_seed).

Sementes cujo ajuste e espurio (estado quase vazio) em algum membro do grupo sao excluidas
do maximo e do nulo -- regra fixada antes de ver o resultado, depois que a semente 5 mostrou
colapsar as tres series da Taesa em um unico estado (87,3% de "sincronizacao" por construcao).

Nulo: deslocamento circular independente de cada SERIE, o MESMO deslocamento para todas as
sementes. As particoes de sementes diferentes sobre a mesma serie sao correlacionadas; deslocar
a serie desloca todas elas juntas, e e isso que a permutacao deve preservar. Deslocar cada
semente por conta propria trataria dez particoes parecidas como dez sorteios independentes e
inflaria o nulo do maximo.

    python hmm_models/Model/hmm_n64_sync_maxnull.py --n-states-main 3 --out-dir <pasta>

Saida (out_dir): sincronizacao_nulo_maximo.csv (ou sincronizacao_nulo_maximo_passo4.csv com
`--shift-step 4`: deslocamentos so em multiplos de 4 trimestres, preservando a fase sazonal)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_n64 import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, N_STATES_MAIN, load_combos
from hmm_n64_sync_inference import (
    N_PERM_DEFAULT,
    RNG_SEED,
    SEEDS,
    build_groups,
    draw_shift,
    fit_states_at_seed_flagged,
    group_has_spurious,
    sync_rate_arrays,
)


def max_null_test(grupos_por_semente: list[dict], espurios_por_semente: list[dict],
                  n_perm: int, rng: np.random.Generator, step: int = 1) -> pd.DataFrame:
    linhas = []
    for nome in grupos_por_semente[0]:
        # só sementes cujo ajuste é legítimo para TODOS os membros do grupo (regra fixada antes)
        por_semente = [g[nome] for g, e in zip(grupos_por_semente, espurios_por_semente)
                       if not group_has_spurious(nome, g[nome], e)]
        if not por_semente:
            continue
        membros = list(por_semente[0])
        K = len(por_semente[0][membros[0]])
        taxas = np.array([sync_rate_arrays(m) for m in por_semente])
        observado = float(taxas.max())

        nulos = np.empty(n_perm)
        for p in range(n_perm):
            desloc = {mb: draw_shift(rng, K, step) for mb in membros}  # um deslocamento por serie
            melhor = 0.0
            for m in por_semente:
                arr = np.stack([np.roll(m[mb], desloc[mb]) for mb in membros], axis=1)
                melhor = max(melhor, np.all(arr == arr[:, [0]], axis=1).sum() / K)
            nulos[p] = melhor

        grupo_tipo, rotulo = nome.split("::", 1)
        linhas.append(
            {
                "grupo": grupo_tipo, "rotulo": rotulo, "P": len(membros), "K": K,
                "n_sementes": len(por_semente),
                "max_observado_pct": round(100 * observado, 3),
                "media_observada_pct": round(100 * float(taxas.mean()), 3),
                "nulo_max_media_pct": round(100 * float(nulos.mean()), 3),
                "nulo_max_p95_pct": round(100 * float(np.percentile(nulos, 95)), 3),
                "p_valor_max": round((1 + int(np.sum(nulos >= observado))) / (n_perm + 1), 4),
                "n_permutacoes": n_perm,
            }
        )
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

    grupos, espurios = [], []
    for seed in SEEDS:
        est, esp = fit_states_at_seed_flagged(combos, args.n_states_main, seed)
        grupos.append(build_groups(combos, est))
        espurios.append(esp)
        print(f"  semente {seed} ok", flush=True)

    res = max_null_test(grupos, espurios, args.n_perm, rng, args.shift_step)
    sufixo = f"_passo{args.shift_step}" if args.shift_step > 1 else ""
    res.to_csv(args.out_dir / f"sincronizacao_nulo_maximo{sufixo}.csv", index=False)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
