"""Especificação completa dos doze campeões do manuscrito.

Motivo (2026-10-18): a Tabela 3 do artigo leva só o essencial; ordens, número de parâmetros,
nós das não linearidades e regressores retidos ficam no CSV gerado por este script.

Para cada campeão reeleito no Round 41 (`_common.manuscript_champions`), a fábrica do painel
(`plot_best_results._make_model_fn`, a mesma que reproduz o campeão) é ajustada na última dobra do
walk-forward (as 63 primeiras observações), sob a grade de hiperparâmetros em que o campeão venceu.
A ordem do bloco linear é a congelada na janela inicial; o número de parâmetros é o da última
dobra: coeficientes do modelo statsmodels (variância da inovação incluída) e, nas estruturas de
blocos, os parâmetros livres da não linearidade estática. Nada aqui é escolhido de novo.

    python scripts/champion_specs.py            # tabela em stdout + CSV
    python scripts/champion_specs.py --check    # não escreve; sai != 0 se o CSV derivou

Saída: paths.JOURNAL_CHAMPION_SPECS_CSV
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore")

import config as cfg  # noqa: E402
import paths  # noqa: E402
import plot_best_results as pbr  # noqa: E402
from _common import manuscript_champions  # noqa: E402
from src.loaders import load_all  # noqa: E402

cfg.apply_global_seed()


def _n_stat_params(fitted) -> "int | None":
    try:
        return int(len(fitted.params))
    except Exception:
        return None


def _describe(model) -> dict:
    out: dict = {"ordem": None, "n_param_linear": None, "n_param_nl": None, "nl_detalhe": None}
    lin = getattr(model, "_linear", None)
    if lin is not None:  # Hammerstein / Wiener: bloco linear + não linearidade estática
        out["ordem"] = str(tuple(lin.order))
        out["n_param_linear"] = _n_stat_params(getattr(lin, "_fitted", None))
        nl = getattr(model, "_f", None) or getattr(model, "_g", None)
        if nl is not None:
            out["n_param_nl"] = int(nl.n_params)
            out["nl_detalhe"] = type(nl).__name__
        out["als_iteracoes"] = getattr(model, "_als_iters", None)
        out["als_convergiu"] = getattr(model, "_als_converged", None)
        return out
    order = getattr(model, "_order", None) or getattr(model, "order", None)
    if order is not None and not callable(order):
        out["ordem"] = str(tuple(order))
    out["n_param_linear"] = _n_stat_params(getattr(model, "_fitted", None))
    spec = getattr(model, "fixed_spec", None) or getattr(model, "fixed_spec_", None)
    if spec is not None:
        out["nl_detalhe"] = str(spec)
    return out


def build_table() -> pd.DataFrame:
    champions = manuscript_champions()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    linhas = []
    for code, empresa in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            champ = champions[(code, ind)]
            rec = champ["record"]
            series, _ = pbr._prepare_series_log(code, ind, datasets, manuscript=True)
            u = pbr._build_exog(series, rec.get("exogenas_utilizadas", []), inputs)
            with pbr.hyperparams_applied(rec.get("hyperparams"), label="grade da busca"):
                fn = pbr._make_model_fn(rec, series, u)
                model = fn(series.iloc[:-1], None if u is None else u.iloc[:-1])
            info = _describe(model)
            n_lin, n_nl = info["n_param_linear"], info["n_param_nl"]
            linhas.append({
                "empresa": empresa, "indicador": ind, "modelo": rec["modelo_utilizado"],
                "descricao": rec["descricao_parametros"], "preprocessamento": rec["preprocessing"],
                "boxcox_lambda": rec.get("boxcox_lambda"), "d": rec.get("unit_root_d"),
                "grade": rec.get("grade_hiperparametros"),
                "exogenas": "; ".join(rec.get("exogenas_utilizadas", [])),
                "n_exogenas": len(rec.get("exogenas_utilizadas", [])),
                "ordem": info["ordem"], "n_param_linear": n_lin, "n_param_nl": n_nl,
                "n_param_total": None if n_lin is None else n_lin + (n_nl or 0),
                "nl_detalhe": info["nl_detalhe"],
                "als_iteracoes": info.get("als_iteracoes"),
                "als_convergiu": info.get("als_convergiu"),
                "params_str": model.get_params_str(),
            })
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="não escreve; compara com o CSV em disco e sai != 0 na divergência")
    args = ap.parse_args()
    df = build_table()
    print(df.drop(columns=["exogenas"]).to_string())
    destino = paths.JOURNAL_CHAMPION_SPECS_CSV
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
