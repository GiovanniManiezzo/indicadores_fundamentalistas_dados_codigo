"""Correlacao entre Net Revenue (3.01) e Net Income (3.11), por empresa, primeira diferenca.

Motivo (revisao de nomenclatura de 2026-09-27): a Secao "Regulated Revenue and Accounting
Distance" do artigo distingue
linhas *cumulativas* (ainda definidas como receita liquida de algo, entao carregam receita como
termo aditivo) de linhas *isoladas* (um grupo unico do DRE, sem receita como termo). A distincao
so vira evidencia, e nao apenas retorica, com uma correlacao medida: Net Income deve correlacionar
com Net Revenue (carrega o termo, diluido); Operational Expenses, por ser isolada, nao deve.

Calculado em primeira diferenca, nunca em nivel: as tres series sao nao-estacionarias por
construcao, e a correlacao em nivel e' inflada por tendencia comum
(confirmado abaixo, correlacao_nivel >> correlacao_diff para as tres transmissoras puras).
Fonte dos dados: paths.JOURNAL_DATA_DIR/{resultado_bruto,lucro_consolidado,despesas_operacionais}.csv
-- a serie ja tratada (imputacao de spikes do Round 28-30), gravada por
`scripts/export_series_tratadas.py`, nao os dados brutos.

    python scripts/revenue_netincome_correlation.py            # tabela + grava o CSV
    python scripts/revenue_netincome_correlation.py --check    # nao escreve; sai !=0 se derivou

Saida: paths.JOURNAL_REVENUE_LINK_CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import paths  # noqa: E402


def _load(indicador: str) -> pd.DataFrame:
    caminho = paths.JOURNAL_DATA_DIR / f"{indicador}.csv"
    return pd.read_csv(caminho, parse_dates=["DT_FIM_EXERC"]).set_index("DT_FIM_EXERC").sort_index()


def build_table() -> pd.DataFrame:
    receita = _load("resultado_bruto")
    lucro = _load("lucro_consolidado")
    despesa = _load("despesas_operacionais")

    linhas: list[dict] = []
    for code, empresa in cfg.EMPRESAS.items():
        r = receita[str(code)].astype(float)
        l = lucro[str(code)].astype(float)
        d = despesa[str(code)].astype(float)

        r_diff, l_diff, d_diff = r.diff().dropna(), l.diff().dropna(), d.diff().dropna()

        rho_nivel = float(r.corr(l))
        rho_rl, p_rl = (float(v) for v in stats.pearsonr(r_diff, l_diff))
        rho_rd, p_rd = (float(v) for v in stats.pearsonr(r_diff, d_diff))

        linhas.append({
            "empresa": empresa,
            "n": int(len(r_diff)),
            "pearson_receita_lucro_nivel": rho_nivel,
            "pearson_receita_lucro_diff": rho_rl,
            "p_receita_lucro_diff": p_rl,
            "pearson_receita_despesa_diff": rho_rd,
            "p_receita_despesa_diff": p_rd,
        })

    return pd.DataFrame(linhas)


def as_markdown(df: pd.DataFrame) -> str:
    linhas = [
        "| Empresa | N | Receita x Net Income (diff) | Receita x Op.Exp (diff) |",
        "|---|---:|---:|---:|",
    ]
    for r in df.itertuples():
        linhas.append(
            f"| {r.empresa} | {r.n} | rho={r.pearson_receita_lucro_diff:.3f} "
            f"(p={r.p_receita_lucro_diff:.4f}) | rho={r.pearson_receita_despesa_diff:.3f} "
            f"(p={r.p_receita_despesa_diff:.4f}) |"
        )
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="nao escreve; compara com o CSV em disco e sai !=0 na divergencia")
    args = ap.parse_args()

    df = build_table()
    print(as_markdown(df))

    destino = paths.JOURNAL_REVENUE_LINK_CSV
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
