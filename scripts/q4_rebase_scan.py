"""Quartos trimestres em que a base do anual difere da base dos trimestres — receita (conta 3.01).

Motivo (Round 40, protocolo metodológico §31): o quarto trimestre da série é o anual menos os
trimestres arquivados. Quando o anual do ano já vem reapresentado —
por exemplo para excluir operações descontinuadas — e as ITRs do mesmo ano não, a subtração mistura
duas bases e o T4 sai deslocado (Axia Energy 2018-Q4: R$ -4,84 bi). Este script varre os quatro
grupos e marca os anos em que isso ocorre pelo critério fixado em `config.py`:

  * os nove meses do ano, reapresentados nas comparativas da ITR do ano seguinte, diferem dos
    nove meses arquivados em mais de `Q4_REBASE_MIN_9M_CHANGE`; e
  * o anual arquivado difere da sua reapresentação (comparativa do DFP do ano seguinte) em menos
    de `Q4_REBASE_MAX_ANNUAL_CHANGE` — isto é, o anual já está na base nova.

Nos anos marcados o T4 de base consistente é `anual arquivado - nove meses reapresentados`. Os
valores a usar saem de `config.Q4_REBASE_OVERRIDES`; `--check` confirma que coincidem com o recomputo.

    python scripts/q4_rebase_scan.py            # tabela em stdout + CSV
    python scripts/q4_rebase_scan.py --check    # não escreve; sai != 0 se o CSV ou o config derivou

Fontes: paths.CVM_ANNUAL_CSV, paths.CVM_QUARTERLY_CSV (fora do repositório). Saída:
paths.JOURNAL_Q4_REBASE_CSV
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import paths  # noqa: E402
from src.loaders import load_indicador  # noqa: E402

CONTA = "3.01"
GRUPO = "DF Consolidado - Demonstra"


def _load(path: Path) -> dict[tuple, float]:
    """(cd_cvm, dt_refer, ordem, dt_ini, dt_fim) -> valor em R$, na última versão arquivada."""
    codes = {str(c) for c in cfg.EMPRESAS}
    best: dict[tuple, tuple[float, int]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        ix = {k: i for i, k in enumerate(header)}
        for row in reader:
            if len(row) < len(header) or row[ix["CD_CVM"]] not in codes:
                continue
            if row[ix["CD_CONTA"]] != CONTA or not row[ix["GRUPO_DFP"]].startswith(GRUPO):
                continue
            escala = 1000.0 if row[ix["ESCALA_MOEDA"]] == "MIL" else 1.0
            key = (row[ix["CD_CVM"]], row[ix["DT_REFER"]], row[ix["ORDEM_EXERC"]],
                   row[ix["DT_INI_EXERC"]], row[ix["DT_FIM_EXERC"]])
            val, ver = float(row[ix["VL_CONTA"]]) * escala, int(row[ix["VERSAO"]])
            if key not in best or ver >= best[key][1]:
                best[key] = (val, ver)
    return {k: v[0] for k, v in best.items()}


def build_table() -> pd.DataFrame:
    dfp = _load(paths.CVM_ANNUAL_CSV)
    itr = _load(paths.CVM_QUARTERLY_CSV)
    serie = load_indicador("resultado_bruto")
    linhas: list[dict] = []
    for code, empresa in cfg.EMPRESAS.items():
        c = str(code)
        for y in range(2011, 2025):
            anual = dfp.get((c, f"{y}-12-31", "ÚLTIMO", f"{y}-01-01", f"{y}-12-31"))
            anual_re = dfp.get((c, f"{y+1}-12-31", "PENÚLTIMO", f"{y}-01-01", f"{y}-12-31"))
            n9 = itr.get((c, f"{y}-09-30", "ÚLTIMO", f"{y}-01-01", f"{y}-09-30"))
            n9_re = itr.get((c, f"{y+1}-09-30", "PENÚLTIMO", f"{y}-01-01", f"{y}-09-30"))
            if None in (anual, anual_re, n9, n9_re):
                continue
            var9 = n9_re / n9 - 1.0
            var_an = anual_re / anual - 1.0
            q4_serie = float(serie.loc[f"{y}-12-31", c])
            linhas.append({
                "empresa": empresa, "ano": y,
                "anual_arquivado": anual, "anual_reapresentado": anual_re, "var_anual": var_an,
                "nove_arquivado": n9, "nove_reapresentado": n9_re, "var_nove": var9,
                "q4_serie": q4_serie, "q4_base_consistente": anual - n9_re,
                "aplica": bool(abs(var9) > cfg.Q4_REBASE_MIN_9M_CHANGE
                               and abs(var_an) < cfg.Q4_REBASE_MAX_ANNUAL_CHANGE),
            })
    df = pd.DataFrame(linhas)
    return df[(df["aplica"]) | (df["var_nove"].abs() > 0.03)].reset_index(drop=True)


def check_config(df: pd.DataFrame) -> list[str]:
    """Diferenças entre as células marcadas e `config.Q4_REBASE_OVERRIDES`."""
    esperado = {(cod, "resultado_bruto", f"{r.ano}-12-31"): r.q4_base_consistente
                for r in df[df["aplica"]].itertuples()
                for cod, emp in cfg.EMPRESAS.items() if emp == r.empresa}
    dado = {(cod, ind, d): v for (cod, ind), cells in cfg.Q4_REBASE_OVERRIDES.items()
            for d, v in cells.items()}
    erros = [f"fora do config: {k}" for k in esperado if k not in dado]
    erros += [f"fora do recomputo: {k}" for k in dado if k not in esperado]
    erros += [f"valor diverge em {k}: {dado[k]:.0f} != {esperado[k]:.0f}"
              for k in dado if k in esperado and abs(dado[k] - esperado[k]) > 0.5]
    return erros


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="não escreve; sai != 0 se o CSV ou o config divergirem do recomputo")
    args = ap.parse_args()

    df = build_table()
    show = df.assign(**{c: df[c] / 1e6 for c in
                        ("anual_arquivado", "anual_reapresentado", "nove_arquivado",
                         "nove_reapresentado", "q4_serie", "q4_base_consistente")})
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    erros = check_config(df)
    texto = df.to_csv(index=False, lineterminator="\n")
    destino = paths.JOURNAL_Q4_REBASE_CSV
    if args.check:
        if not destino.exists() or destino.read_text(encoding="utf-8") != texto:
            erros.append(f"{destino.name} ausente ou divergiu do recomputado")
        for e in erros:
            print(f"FALHA: {e}", file=sys.stderr)
        sys.exit(1 if erros else 0)
    for e in erros:
        print(f"AVISO: {e}", file=sys.stderr)
    destino.write_text(texto, encoding="utf-8")
    print(f"\nGravado: {destino.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
