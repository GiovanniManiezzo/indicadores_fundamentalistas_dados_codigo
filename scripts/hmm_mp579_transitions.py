"""Transicoes decodificadas perto da MP 579/2012, na especificacao atual (dois estados, N=64).

Motivo (2026-09-28): o resultado de maior destaque do artigo do congresso foi a transicao em
Q4/2012, temporalmente consistente com a MP 579. Este script verifica se ela se mantem na
janela N=64.

Leitura: `paths.JOURNAL_HMM_DIR/estados_viterbi.csv` (quatro variantes x doze series, dois
estados, canonicos). Uma transicao e' datada pelo primeiro trimestre no estado novo.

So entram as particoes que se comportam como regime -- no maximo `MAX_TRANSICOES` trocas em 62
passos. Uma particao que troca de estado quase todo trimestre esta descrevendo a oscilacao da
primeira diferenca, nao um regime, e cairia em qualquer janela por construcao. O corte e'
reportado junto com a contagem, e a contagem de particoes excluidas tambem.

Esperado sob uniformidade: total de transicoes x (passos dentro da janela / passos totais).

    python scripts/hmm_mp579_transitions.py            # tabela + grava o CSV
    python scripts/hmm_mp579_transitions.py --check    # nao escreve; sai !=0 se derivou

    python scripts/hmm_mp579_transitions.py --mp579-bruto   # sensibilidade

Saida: paths.JOURNAL_MP579_CSV

Desde o Round 40 (protocolo metodológico §31) a série principal da metade HMM já tem as células da MP 579 no
valor registrado, de modo que o teste principal acima É o teste sem imputação. `--mp579-bruto` é
anterior a isso, compara com as séries do tratamento da dissertação (imputadas) e não alimenta mais
o manuscrito.

`--mp579-bruto` repete o teste sobre as decodificacoes de `paths.journal_mp579_hmm_dir(metodo)`,
series em que as celulas da MP 579 (`cfg.HMM_MP579_RAW_CELLS`) nao foram imputadas, e grava
`paths.JOURNAL_MP579_RAW_CSV` com a especificacao vigente ao lado, mais as transicoes de cada
serie afetada dentro da janela larga.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402

MAX_TRANSICOES = 12
# Estreita: o trimestre da MP (2012-Q3) e os dois seguintes. Larga: ate o fim de 2013, cobrindo
# a conversao em lei (Lei 12.783, jan/2013) e o primeiro ano sob as concessoes renovadas.
JANELAS = {"2012Q3-2013Q1": ("2012-07-01", "2013-03-31"),
           "2012Q3-2013Q4": ("2012-07-01", "2013-12-31")}
# Os outros tres eventos datados do setor, listados a priori, em
# janelas de seis trimestres como a larga acima, so para dar a MP 579 vizinhos comparaveis:
# RBSE (pagamentos pela RAP a partir de jul/2017), IFRS 15/CPC 47 (jan/2018) e a capitalizacao
# da Eletrobras (jun/2022).
EVENTOS_2_1 = {"RBSE 2017Q3-2018Q4": ("2017-07-01", "2018-12-31"),
               "IFRS15 2018Q1-2019Q2": ("2018-01-01", "2019-06-30"),
               "Capitalizacao 2022Q2-2023Q3": ("2022-04-01", "2023-09-30")}
JANELAS = {**JANELAS, **EVENTOS_2_1}


def _transicoes(hmm_dir: Path | None = None) -> tuple[list[dict], np.ndarray]:
    hmm_dir = hmm_dir or paths.JOURNAL_HMM_DIR
    d = pd.read_csv(hmm_dir / "estados_viterbi.csv", parse_dates=["data"])
    particoes = []
    datas_passos = None
    for (emp, ind, var), g in d.groupby(["empresa", "indicador", "variante"]):
        g = g.sort_values("data")
        s = g["estado_canonico"].to_numpy()
        passos = g["data"].to_numpy()[1:]
        trocas = passos[s[1:] != s[:-1]]
        particoes.append({"empresa": emp, "indicador": ind, "variante": var,
                          "n_passos": len(passos), "trocas": trocas})
        if datas_passos is None or len(passos) > len(datas_passos):
            datas_passos = passos
    return particoes, datas_passos


def build_table(hmm_dir: Path | None = None) -> pd.DataFrame:
    particoes, passos = _transicoes(hmm_dir)
    regime = [p for p in particoes if len(p["trocas"]) <= MAX_TRANSICOES]
    todas = np.concatenate([p["trocas"] for p in regime])
    linhas = []
    for nome, (lo, hi) in JANELAS.items():
        lo, hi = np.datetime64(lo), np.datetime64(hi)
        dentro = int(((todas >= lo) & (todas <= hi)).sum())
        fracao = float(((passos >= lo) & (passos <= hi)).mean())
        # Referencia empirica ao lado da uniforme: fracao das janelas moveis de mesmo numero de
        # passos que contem pelo menos tantas transicoes.
        w = int(((passos >= lo) & (passos <= hi)).sum())
        moveis = np.array([np.isin(todas, passos[i:i + w]).sum() for i in range(len(passos) - w + 1)])
        linhas.append({"janela": nome, "particoes_total": len(particoes),
                       "particoes_regime": len(regime), "max_transicoes": MAX_TRANSICOES,
                       "transicoes": len(todas), "dentro": dentro,
                       "esperado_uniforme": round(len(todas) * fracao, 1),
                       "fracao_janelas_ge": round(float((moveis >= dentro).mean()), 3)})
    return pd.DataFrame(linhas)


AFETADAS = {("ELET6", "lucro_consolidado"), ("ELET6", "despesas_operacionais"),
            ("ISAE4", "resultado_bruto")}


def _afetadas(hmm_dir: Path) -> pd.DataFrame:
    """Transicoes de cada particao das series cujas celulas da MP 579 deixaram de ser imputadas."""
    particoes, _ = _transicoes(hmm_dir)
    lo, hi = (np.datetime64(x) for x in JANELAS["2012Q3-2013Q4"])
    linhas = []
    for p in particoes:
        if (p["empresa"], p["indicador"]) not in AFETADAS:
            continue
        dentro = [str(np.datetime_as_string(x, unit="D")) for x in p["trocas"] if lo <= x <= hi]
        linhas.append({"empresa": p["empresa"], "indicador": p["indicador"],
                       "variante": p["variante"], "trocas_total": len(p["trocas"]),
                       "particao_regime": len(p["trocas"]) <= MAX_TRANSICOES,
                       "trocas_2012Q3_2013Q4": ";".join(dentro)})
    return pd.DataFrame(linhas)


def build_sensibilidade() -> tuple[pd.DataFrame, pd.DataFrame]:
    import config as cfg

    tabs = [build_table().assign(series="vigente (MP 579 imputada)")]
    afet = [_afetadas(paths.JOURNAL_HMM_DIR).assign(series="vigente (MP 579 imputada)")]
    for metodo in cfg.HMM_MP579_IMPUTE_METHODS:
        rotulo = f"MP 579 bruta, demais por {metodo}"
        tabs.append(build_table(paths.journal_mp579_hmm_dir(metodo)).assign(series=rotulo))
        afet.append(_afetadas(paths.journal_mp579_hmm_dir(metodo)).assign(series=rotulo))
    return pd.concat(tabs, ignore_index=True), pd.concat(afet, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mp579-bruto", action="store_true")
    args = ap.parse_args()

    if args.mp579_bruto:
        tab, afet = build_sensibilidade()
        with pd.option_context("display.width", 200):
            print(tab.to_string(index=False))
            print()
            print(afet.to_string(index=False))
        tab.to_csv(paths.JOURNAL_MP579_RAW_CSV, index=False)
        afet.to_csv(paths.JOURNAL_MP579_RAW_CSV.with_name(
            paths.JOURNAL_MP579_RAW_CSV.stem + "_afetadas.csv"), index=False)
        print("gravado:", paths.JOURNAL_MP579_RAW_CSV)
        return 0

    tab = build_table()
    print(tab.to_string(index=False))
    destino = paths.JOURNAL_MP579_CSV
    if args.check:
        antigo = pd.read_csv(destino)
        if not antigo.astype(str).equals(tab.astype(str)):
            print("DERIVA contra", destino)
            return 1
        print("OK:", destino)
        return 0
    tab.to_csv(destino, index=False)
    print("gravado:", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
