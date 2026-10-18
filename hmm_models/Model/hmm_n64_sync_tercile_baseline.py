"""Sincronização dos rótulos de tercil, sem HMM — o controle da sincronização decodificada.

Motivo: com emissão categórica em três
bins equiprováveis e estados que duram cerca de um trimestre, o caminho de Viterbi pode apenas
reproduzir o bin observado, e a taxa de sincronização da Eq. 5 passaria a medir a coocorrência
de tercis das primeiras diferenças, pergunta que não precisa de HMM. Este script calcula a mesma
taxa, com o mesmo nulo por deslocamento circular (passo livre e passo 4), sobre os próprios
rótulos de tercil (`discretize`, a entrada da variante categórica), e a põe ao lado da taxa
decodificada de `sincronizacao_permutacao*.csv`. Se as duas coincidem, o modelo latente não
acrescenta nada à sincronização.

    python hmm_models/Model/hmm_n64_sync_tercile_baseline.py [--out-dir DIR] [--n-perm N]

Saída (out_dir): sincronizacao_tercis_sem_hmm.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_n64 import DEFAULT_DATA_DIR, discretize, load_combos
from hmm_n64_sync_inference import N_PERM_DEFAULT, RNG_SEED, build_groups, permutation_test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="diretório da sincronização decodificada (p.ex. hmm_n64_k3_rerun)")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    args = ap.parse_args()

    combos, _ = load_combos(args.data_dir)
    rotulos = {(c.empresa, c.indicador): discretize(c.x)[0].ravel() for c in combos}
    grupos = build_groups(combos, rotulos)

    partes = []
    for passo in (1, 4):
        rng = np.random.default_rng(RNG_SEED)
        t = permutation_test(grupos, args.n_perm, rng, step=passo)
        t["passo_deslocamento"] = passo
        partes.append(t)
    tercis = pd.concat(partes, ignore_index=True)

    decod = []
    for passo, nome in ((1, "sincronizacao_permutacao.csv"),
                        (4, "sincronizacao_permutacao_passo4.csv")):
        arq = args.out_dir / nome
        if arq.exists():
            d = pd.read_csv(arq)[["grupo", "rotulo", "taxa_observada_pct", "p_valor"]]
            d = d.rename(columns={"taxa_observada_pct": "taxa_hmm_pct", "p_valor": "p_hmm"})
            d["passo_deslocamento"] = passo
            decod.append(d)
    if decod:
        tercis = tercis.merge(pd.concat(decod), on=["grupo", "rotulo", "passo_deslocamento"],
                              how="left")

    destino = args.out_dir / "sincronizacao_tercis_sem_hmm.csv"
    tercis.to_csv(destino, index=False, lineterminator="\n")
    cols = [c for c in ("grupo", "rotulo", "passo_deslocamento", "taxa_observada_pct", "p_valor",
                        "taxa_hmm_pct", "p_hmm") if c in tercis.columns]
    print(tercis[cols].to_string(index=False))
    print(f"\nGravado: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
