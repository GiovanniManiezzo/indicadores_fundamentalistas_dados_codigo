"""Recomputa o painel de significância (Clark-West/DM + Holm/BH + MCS) do artigo.

Motivo (2026-09-20): o painel publicado testava onze hipóteses de uma vez sem correção de multiplicidade, e comparava
quatro pares ANINHADOS (ARMAX/ARIMAX contra a ARIMA de mesma ordem, que cada um contém como caso
particular) pelo teste de Diebold-Mariano, inválido sob aninhamento — sob H0 os parâmetros extras
do modelo maior enviesam a diferença de perdas contra ele mesmo. Este script:

  1. Reproduz o campeão de cada uma das onze combinações não-ARIMA pela mesma maquinaria de
     `modelos_estaveis/reproduzir_combo.py` (`plot_best_results.run_combo`), garantindo que os
     resíduos usados no teste são os do campeão efetivamente reprodutível sob o `config.py`
     vigente (regra de reprodutibilidade do painel).
  2. Ajusta um baseline ARIMA fresco por combinação — ordem por AIC na janela `MIN_TRAIN` inicial
     sobre a série bruta, consenso de raiz unitária para `d` — pela mesma rotina que `main.py`
     usa para selecionar ordem (`_select_arima_order`), nunca por grid search fora da janela
     inicial (Invariante 4).
  3. Para os pares aninhados (família ARMAX ou ARIMAX), usa `src.evaluation.clark_west_test` no
     lugar do Diebold-Mariano; para os demais, mantém o Diebold-Mariano.
  4. Aplica Holm e Benjamini-Hochberg (alfa = 0.10) ao painel resultante de onze p-valores
     (`statsmodels.stats.multitest.multipletests`).
  4b. (2026-10-12, revisão mínima viável) Grava também o Diebold-Mariano nas onze comparações
     (`dm_p_todos`, com Holm/BH próprios) e marca o aninhamento DE FATO (`aninhado`): família
     ARIMAX/BAYESARIMA, mesmo pré-processamento do baseline (raw), mesmo `d` e ordem (p, q) do
     campeão cobrindo a do baseline. A coluna `teste`/`pvalue` passa a usar Clark-West só onde
     `aninhado` é verdadeiro -- antes bastava a família, o que punha sob Clark-West dois pares em
     log contra um baseline em nível, que não estão aninhados.
  5. Roda o Model Confidence Set (`src.evaluation.model_confidence_set`) sobre {RW, ARIMA,
     Campeão} em cada combinação, como checagem de robustez adicional — a eliminação interna do
     MCS resolve cada par pelo Diebold-Mariano mesmo quando o passo 3 usou Clark-West, então o
     resultado do MCS para as quatro combinações aninhadas é indicativo, não uma reafirmação do
     teste formal acima.

Não recomputa nada da busca de 30 rodadas — parte do campeão já eleito (`compute_champions()`) e
só refaz o baseline ARIMA e os testes de significância sobre ele. Import pesado (statsmodels via
`plot_best_results`): por isso este script vive fora de `_common.py`/`round_status.py` e nunca é
importado por eles, mesmo padrão de `modelos_estaveis/reproduzir_combo.py`.

    python scripts/dm_multiplicity_panel.py            # tabela markdown em stdout + CSV
    python scripts/dm_multiplicity_panel.py --check    # não escreve; sai != 0 se o CSV derivou

  6. (2026-10-18) Efeito mínimo detectável do DM-HLN bilateral a 10% com
     poder de 80%: a diferença média de perdas quadráticas que o teste detectaria com o número de
     previsões pontuadas e a variância observada do diferencial, expressa como fração do erro
     quadrático médio do ARIMA de referência (`mde_mse_rel`). Grava também FIT% e U2 do ARIMA de
     referência na mesma máscara (`fit_arima`, `u2_arima`), para a Tabela 3.

Saída: paths.JOURNAL_SIGNIFICANCE_CSV
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore")

import config as cfg  # noqa: E402
import paths  # noqa: E402
import plot_best_results as pbr  # noqa: E402
from _common import compute_champions, manuscript_champions  # noqa: E402
from src.loaders import load_all  # noqa: E402
from src.preprocess import unit_root_tests  # noqa: E402
from src.evaluation import (  # noqa: E402
    walk_forward_cv, diebold_mariano_test, clark_west_test, model_confidence_set,
)
from src.models.arima_wrapper import ARIMAWrapper  # noqa: E402

cfg.apply_global_seed()

NESTED_FAMILIES = {"ARMAX", "ARIMAX", "BAYESARIMA"}  # contem a ARIMA de mesma ordem como caso
# particular (gamma = 0); BAYESARIMA adicionada em 2026-09-27 -- o unico modelo desta familia no painel (Taesa/lucro_consolidado) retem
# duas exogenas sob um prior de encolhimento, mas e' ainda um ARIMA(p,d,q) com gamma != 0 quando
# a media a posteriori dos coeficientes exogenos difere de zero, entao o mesmo vies do
# Diebold-Mariano sob H0 se aplica.
ALPHA = 0.10
POWER = 0.80


def _mde_rel(e_champ: np.ndarray, e_base: np.ndarray) -> float:
    """Menor |E[d_k]| detectável (DM-HLN, h = 1, bilateral a ALPHA, poder POWER) / MSE base."""
    d = e_champ ** 2 - e_base ** 2
    n = len(d)
    se = np.sqrt(np.mean((d - d.mean()) ** 2) / n) / np.sqrt((n - 1) / n)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1) + stats.t.ppf(POWER, n - 1)
    return float(t * se / np.mean(e_base ** 2))


def _family(model_name: str) -> str:
    return model_name.split("-")[0].upper()


def _baseline_order(series: pd.Series) -> tuple[int, int, int]:
    d = unit_root_tests(series)["consensus_d"]
    return ARIMAWrapper(d=d, mode="raw").fit(series.iloc[:cfg.MIN_TRAIN]).order


def _truly_nested(fam: str, mode: str, champ_order, base_order) -> bool:
    """Aninhamento de fato: o campeão contém o baseline como caso particular (gamma = 0)."""
    if fam not in NESTED_FAMILIES or mode != "raw" or champ_order is None:
        return False
    p, d, q = champ_order
    bp, bd, bq = base_order
    return d == bd and p >= bp and q >= bq


def _fresh_arima_baseline(series: pd.Series,
                          exclude: "list[pd.Timestamp] | None" = None) -> np.ndarray:
    """Baseline ARIMA(p,d,q) — d por consenso de raiz unitária, (p,q) por AIC em MIN_TRAIN.

    Mesma rotina de `main.py::_select_arima_order`: ordem selecionada uma única vez na janela
    inicial e congelada (`fixed_order`) para todos os folds — anti-leakage, Invariante 4.
    """
    d = unit_root_tests(series)["consensus_d"]
    order = ARIMAWrapper(d=d, mode="raw").fit(series.iloc[:cfg.MIN_TRAIN]).order
    fn = lambda yt, ut: ARIMAWrapper(d=d, mode="raw", fixed_order=order).fit(yt)
    cv = walk_forward_cv(fn, series, None, min_train=cfg.MIN_TRAIN, score_exclude=exclude)
    return cv["y_pred"]


def build_table(manuscript: bool = True) -> pd.DataFrame:
    """`manuscript=True`: tratamento do manuscrito (Round 40, protocolo metodológico §31) — mesma máscara
    de alvos imputados para o campeão, o ARIMA, o passeio aleatório e o MCS; campeões reeleitos
    no Round 41 (§32)."""
    champions = manuscript_champions() if manuscript else compute_champions()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}

    linhas: list[dict] = []
    for code, empresa in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            champ = champions.get((code, ind))
            if champ is None or not champ.get("gate_passed", True):
                continue
            fam = _family(champ["model"])
            if fam == "ARIMA":
                continue  # é a própria referência — sem teste de significância

            res = pbr.run_combo(code, ind, champ, datasets, inputs, manuscript=manuscript)
            y_true = np.asarray(res["y_true"], dtype=float)
            y_pred_champ = np.asarray(res["y_pred"], dtype=float)
            y_rw = np.asarray(res["y_rw"], dtype=float)

            series, imputed = pbr._prepare_series_log(code, ind, datasets, manuscript=manuscript)
            y_pred_arima = _fresh_arima_baseline(series, imputed if manuscript else None)

            e_champ = y_true - y_pred_champ
            e_arima = y_true - y_pred_arima
            e_rw = y_true - y_rw

            base_order = _baseline_order(series)
            champ_order = (pbr._order_from(res["desc"], (0, 0, 0))
                           if fam in NESTED_FAMILIES else None)
            aninhado = _truly_nested(fam, res["mode"], champ_order, base_order)
            dm_stat, dm_p = diebold_mariano_test(e_champ, e_arima)
            if aninhado:
                stat, p = clark_west_test(e_arima, e_champ)
                teste = "Clark-West"
            else:
                stat, p = dm_stat, dm_p
                teste = "Diebold-Mariano"

            mcs = model_confidence_set(
                {"RW": e_rw, "ARIMA": e_arima, "Campeao": e_champ}, alpha=ALPHA
            )

            linhas.append({
                "empresa": empresa,
                "indicador": ind,
                "modelo": champ["model"],
                "familia": fam,
                "modo": res["mode"],
                "ordem_campeao": "" if champ_order is None else str(tuple(champ_order)),
                "ordem_baseline": str(tuple(base_order)),
                "aninhado": aninhado,
                "dm_stat_todos": float(dm_stat),
                "dm_p_todos": float(dm_p),
                "teste": teste,
                "stat": float(stat),
                "pvalue": float(p),
                "alvos_excluidos": int(res["n_excluded"]),
                "dm_p_original": champ.get("dm_p"),
                "mcs_sobreviventes": ",".join(mcs),
                "n_pontuados": int(len(e_champ)),
                "reducao_mse_obs": float(1 - np.mean(e_champ ** 2) / np.mean(e_arima ** 2)),
                "mde_mse_rel": _mde_rel(e_champ, e_arima),
                "fit_arima": float((1 - np.linalg.norm(e_arima)
                                    / np.linalg.norm(y_true - y_true.mean())) * 100),
                "u2_arima": float(np.sqrt(np.mean(e_arima ** 2) / np.mean(e_rw ** 2))),
            })

    df = pd.DataFrame(linhas).sort_values("pvalue", ignore_index=True)
    holm_rej, holm_p, _, _ = multipletests(df["pvalue"], alpha=ALPHA, method="holm")
    bh_rej, bh_p, _, _ = multipletests(df["pvalue"], alpha=ALPHA, method="fdr_bh")
    df["holm_p"] = holm_p
    df["holm_rejeita"] = holm_rej
    df["bh_p"] = bh_p
    df["bh_rejeita"] = bh_rej
    holm_rej, holm_p, _, _ = multipletests(df["dm_p_todos"], alpha=ALPHA, method="holm")
    bh_rej, bh_p, _, _ = multipletests(df["dm_p_todos"], alpha=ALPHA, method="fdr_bh")
    df["dm_todos_holm_p"] = holm_p
    df["dm_todos_holm_rejeita"] = holm_rej
    df["dm_todos_bh_rejeita"] = bh_rej
    return df


def as_markdown(df: pd.DataFrame) -> str:
    linhas = [
        "| Combinação | Modelo | Teste | stat | p | Holm-p | BH-p | Rejeita (Holm/BH) | MCS |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in df.itertuples():
        rej = f"{'sim' if r.holm_rejeita else 'não'}/{'sim' if r.bh_rejeita else 'não'}"
        linhas.append(
            f"| {r.empresa}x{r.indicador} | {r.modelo} | {r.teste} | {r.stat:.3f} | "
            f"{r.pvalue:.3f} | {r.holm_p:.3f} | {r.bh_p:.3f} | {rej} | {r.mcs_sobreviventes} |"
        )
    n_holm = int(df["holm_rejeita"].sum())
    n_bh = int(df["bh_rejeita"].sum())
    linhas.append("")
    linhas.append(
        f"Sobreviventes a alfa=0.10: {n_holm} de {len(df)} (Holm), {n_bh} de {len(df)} (BH)."
    )
    linhas.append(
        f"Diebold-Mariano nas {len(df)}: {int(df['dm_todos_holm_rejeita'].sum())} (Holm), "
        f"{int(df['dm_todos_bh_rejeita'].sum())} (BH); "
        f"aninhados de fato: {int(df['aninhado'].sum())}."
    )
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="não escreve; compara com o CSV em disco e sai != 0 na divergência")
    ap.add_argument("--tratamento", choices=("manuscrito", "vigente"), default="manuscrito",
                    help="'vigente' reproduz o tratamento anterior ao Round 40 e não escreve")
    args = ap.parse_args()

    df = build_table(manuscript=args.tratamento == "manuscrito")
    print(as_markdown(df))
    if args.tratamento == "vigente":
        return

    destino = paths.JOURNAL_SIGNIFICANCE_CSV
    if args.check:
        if not destino.exists():
            print(f"\nFALHA: {destino.name} não existe.", file=sys.stderr)
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
