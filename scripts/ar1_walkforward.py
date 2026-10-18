"""AR(1) fixo em walk-forward nas doze combinações — o benchmark de persistência medido.

Motivo (2026-10-12): o benchmark de persistência do artigo era a fórmula populacional
FIT%_AR(1)(rho_1) = (1 - sqrt(1 - rho_1^2)) x 100, com rho_1 estimado na amostra inteira. Aqui o mesmo AR(1) é RODADO: ARIMA(1,0,0)
com constante, sobre a série tratada em nível (a mesma do campeão, `plot_best_results.
_prepare_series`), janela expansiva a partir de MIN_TRAIN, um passo à frente, coeficientes
reestimados a cada passo. Ordem fixa, sem busca e sem pré-processamento: nada é escolhido na
janela de avaliação, então a distância campeão - AR(1) não depende de seleção sobre o desfecho.

    python scripts/ar1_walkforward.py            # tabela markdown em stdout + CSV
    python scripts/ar1_walkforward.py --check    # não escreve; sai != 0 se o CSV derivou

Extensões de 2026-10-18:
- o AR(1) em nível é mal especificado nas oito séries I(1); a referência de persistência do
  manuscrito passa a ser ARIMA(1,d,0) em nível, com d do consenso ADF/KPSS/PP
  (`preprocess.unit_root_tests`, o mesmo d do ARIMA de referência uniforme). As colunas do AR(1)
  em nível ficam no CSV (o modo Round 40 de `main.py` as lê).
- a Eq. do FIT% populacional só vale para série estacionária. Nas séries com d = 1 ela é
  aplicada à série diferenciada: rho_1 do incremento e o FIT% do campeão medido na mesma escala.
  O erro de previsão do incremento é o mesmo do nível (y - yhat = Dy - Dyhat, com
  Dyhat = yhat - y(k-1)); muda só o denominador da Eq. do FIT%, que passa a ser a variabilidade
  do incremento.
- FIT% e U2 do campeão pontuados nos 528 alvos contra o valor registrado (`score="recorded"`).
- MASE e RMSSE com a escala do sazonal ingênuo dentro da amostra, recalculada a cada origem
  sobre a janela de treino, para campeão, passeio aleatório, ARIMA(1,d,0) e sazonal ingênuo.
- correlação cruzada entre o erro fora da amostra e cada regressor retido (defasagens 0 a 4,
  regressor em nível e em primeira diferença), nos campeões com exógena — CSV à parte.

Saída: paths.JOURNAL_AR1_WALKFORWARD_CSV, paths.JOURNAL_RESID_INPUT_CCF_CSV
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
from src.evaluation import walk_forward_cv, fit_percent, diebold_mariano_test  # noqa: E402
from src.models.arima_wrapper import ARIMAWrapper  # noqa: E402
from src.preprocess import unit_root_tests  # noqa: E402

cfg.apply_global_seed()

AR1_ORDER = (1, 0, 0)
SEASON = 4
CCF_LAGS = range(0, 5)


def _ar1_forecasts(series: pd.Series, exclude: "list[pd.Timestamp] | None" = None) -> dict:
    fn = lambda yt, ut: ARIMAWrapper(d=0, mode="raw", fixed_order=AR1_ORDER).fit(yt)
    return walk_forward_cv(fn, series, None, min_train=cfg.MIN_TRAIN, score_exclude=exclude)


def _ref_forecasts(series: pd.Series, d: int,
                   exclude: "list[pd.Timestamp] | None" = None) -> dict:
    """ARIMA(1,d,0) em nível, ordem fixa — a referência de persistência (B2)."""
    fn = lambda yt, ut: ARIMAWrapper(d=d, mode="raw", fixed_order=(1, d, 0)).fit(yt)
    return walk_forward_cv(fn, series, None, min_train=cfg.MIN_TRAIN, score_exclude=exclude)


def fit_pop(rho1: float) -> float:
    return float(100.0 * (1.0 - np.sqrt(max(0.0, 1.0 - rho1 ** 2))))


def _scaled_errors(series: pd.Series, pos: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """MASE e RMSSE: cada erro dividido pela escala do sazonal ingênuo na janela de treino da
    sua origem (y_0..y_{k-1}), como na M5, aplicada origem a origem."""
    y = series.to_numpy(dtype=float)
    a, q = [], []
    for k, e in zip(pos, err):
        dif = y[SEASON:k] - y[:k - SEASON]
        a.append(abs(e) / np.mean(np.abs(dif)))
        q.append(e ** 2 / np.mean(dif ** 2))
    return float(np.mean(a)), float(np.sqrt(np.mean(q)))


def _resid_input_ccf(series: pd.Series, idx: pd.Index, err: np.ndarray, exog: list[str],
                     inputs: pd.DataFrame) -> list[dict]:
    u = pbr._build_exog(series, exog, inputs)
    if u is None:
        return []
    banda = 1.96 / np.sqrt(len(err))
    out = []
    for col in u.columns:
        for forma, x in (("nivel", u[col]), ("diferenca", u[col].diff())):
            for lag in CCF_LAGS:
                xv = x.shift(lag).reindex(idx).to_numpy(dtype=float)
                ok = ~np.isnan(xv)
                r = float(np.corrcoef(err[ok], xv[ok])[0, 1])
                out.append({"regressor": col, "forma": forma, "defasagem": lag, "r": r,
                            "banda95": banda, "fora_da_banda": bool(abs(r) > banda)})
    return out


_CCF_CACHE: dict[str, pd.DataFrame] = {}


def build_table(manuscript: bool = True) -> pd.DataFrame:
    """`manuscript=True`: tratamento do manuscrito (Round 40, protocolo metodológico §31) — série com os
    quartos trimestres corrigidos, alvos imputados fora da pontuação (mesma máscara para o AR(1),
    o campeão e o passeio aleatório) e o campeão reeleito no Round 41 (§32), reproduzido."""
    champions = manuscript_champions() if manuscript else compute_champions()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}

    linhas: list[dict] = []
    ccf_rows: list[dict] = []
    for code, empresa in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            champ = champions[(code, ind)]
            series, imputed = pbr._prepare_series_log(code, ind, datasets, manuscript=manuscript)
            excl = imputed if manuscript else None
            cv = _ar1_forecasts(series, excl)
            y_true = np.asarray(cv["y_true"], dtype=float)
            y_ar1 = np.asarray(cv["y_pred"], dtype=float)
            y_rw = np.asarray(cv["y_rw"], dtype=float)

            d = int(unit_root_tests(series)["consensus_d"])
            y_ref = np.asarray(_ref_forecasts(series, d, excl)["y_pred"], dtype=float)

            res = pbr.run_combo(code, ind, champ, datasets, inputs, manuscript=manuscript)
            fit_champ = res["fit_actual"] if manuscript else float(champ["fit"])
            u2_champ = res["u2_actual"] if manuscript else float(champ["u2"])
            y_champ = np.asarray(res["y_pred"], dtype=float)
            dm_stat, dm_p = diebold_mariano_test(y_true - y_champ, y_true - y_ar1)
            dm_stat_ref, dm_p_ref = diebold_mariano_test(y_true - y_champ, y_true - y_ref)
            rmse_rw = np.sqrt(np.mean((y_true - y_rw) ** 2))

            # B1: escala estacionária. Dy(k) = y(k) - y(k-1), e y(k-1) é a previsão do RW.
            if d == 0:
                rho1_estac, fit_champ_estac = float(series.autocorr(1)), fit_champ
            else:
                rho1_estac = float(series.diff().dropna().autocorr(1))
                fit_champ_estac = fit_percent(y_true - y_rw, y_champ - y_rw)
            fpop = fit_pop(rho1_estac)

            pos = np.array([series.index.get_loc(t) for t in res["index_oos"]])
            y_saz = series.to_numpy(dtype=float)[pos - SEASON]
            escalados = {nome: _scaled_errors(series, pos, y_true - p) for nome, p in
                         (("campeao", y_champ), ("rw", y_rw), ("ref", y_ref), ("sazonal", y_saz))}

            linha = {
                "empresa": empresa,
                "indicador": ind,
                "rho1": float(series.autocorr(1)),
                "fit_ar1_wf": fit_percent(y_true, y_ar1),
                "u2_ar1_wf": float(np.sqrt(np.mean((y_true - y_ar1) ** 2)) / rmse_rw),
                "fit_campeao": fit_champ,
                "fit_campeao_recalc": fit_percent(y_true, y_champ),
                "u2_campeao": u2_champ,
                "alvos_excluidos": int(cv["n_excluded"]),
                "excesso_wf": fit_champ - fit_percent(y_true, y_ar1),
                "dm_stat_campeao_vs_ar1": float(dm_stat),
                "dm_p_campeao_vs_ar1": float(dm_p),
                "exogena": bool(champ["exog"]),
                "modelo": champ["model"],
                "d": d,
                "fit_ref_wf": fit_percent(y_true, y_ref),
                "u2_ref_wf": float(np.sqrt(np.mean((y_true - y_ref) ** 2)) / rmse_rw),
                "excesso_ref_wf": fit_champ - fit_percent(y_true, y_ref),
                "dm_stat_campeao_vs_ref": float(dm_stat_ref),
                "dm_p_campeao_vs_ref": float(dm_p_ref),
                "rho1_estac": rho1_estac,
                "fit_campeao_estac": fit_champ_estac,
                "fit_pop_estac": fpop,
                "excesso_pop_estac": fit_champ_estac - fpop,
            }
            for nome, (mase, rmsse) in escalados.items():
                linha[f"mase_{nome}"] = mase
                linha[f"rmsse_{nome}"] = rmsse
            if manuscript:
                reg = pbr.run_combo(code, ind, champ, datasets, inputs, manuscript=True,
                                    score="recorded")
                linha["n_registrado"] = int(len(reg["y_true"]))
                linha["fit_campeao_registrado"] = float(reg["fit_actual"])
                linha["u2_campeao_registrado"] = float(reg["u2_actual"])
            linhas.append(linha)

            for r in _resid_input_ccf(series, res["index_oos"], y_true - y_champ,
                                      res["exog"], inputs):
                ccf_rows.append({"empresa": empresa, "indicador": ind,
                                 "modelo": champ["model"], **r})
    _CCF_CACHE["df"] = pd.DataFrame(ccf_rows)
    return pd.DataFrame(linhas).sort_values("fit_campeao", ascending=False, ignore_index=True)


def summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "pearson_fit_campeao_ar1": float(stats.pearsonr(df["fit_campeao"], df["fit_ar1_wf"])[0]),
        "spearman_fit_campeao_ar1": float(stats.spearmanr(df["fit_campeao"], df["fit_ar1_wf"])[0]),
        "excesso_medio": float(df["excesso_wf"].mean()),
        "excesso_medio_exog": float(df.loc[df["exogena"], "excesso_wf"].mean()),
        "excesso_medio_univ": float(df.loc[~df["exogena"], "excesso_wf"].mean()),
        "ar1_passa_gate": int((df["u2_ar1_wf"] < 1).sum()),
        "dm_rejeita_10pct": int((df["dm_p_campeao_vs_ar1"] < 0.10).sum()),
        "pearson_fit_campeao_ref": float(stats.pearsonr(df["fit_campeao"], df["fit_ref_wf"])[0]),
        "spearman_fit_campeao_ref": float(stats.spearmanr(df["fit_campeao"], df["fit_ref_wf"])[0]),
        "excesso_ref_medio": float(df["excesso_ref_wf"].mean()),
        "excesso_ref_medio_exog": float(df.loc[df["exogena"], "excesso_ref_wf"].mean()),
        "excesso_ref_medio_univ": float(df.loc[~df["exogena"], "excesso_ref_wf"].mean()),
        "ref_passa_gate": int((df["u2_ref_wf"] < 1).sum()),
        "dm_ref_rejeita_10pct": int((df["dm_p_campeao_vs_ref"] < 0.10).sum()),
        "excesso_pop_medio_exog": float(df.loc[df["exogena"], "excesso_pop_estac"].mean()),
        "excesso_pop_medio_univ": float(df.loc[~df["exogena"], "excesso_pop_estac"].mean()),
    }


def as_markdown(df: pd.DataFrame, s: dict[str, float]) -> str:
    linhas = [
        "| Combinação | rho_1 | FIT% AR(1) wf | U2 AR(1) | FIT% campeão | excesso | DM p | exóg. |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in df.itertuples():
        linhas.append(
            f"| {r.empresa}x{r.indicador} | {r.rho1:.2f} | {r.fit_ar1_wf:.1f} | {r.u2_ar1_wf:.3f} | "
            f"{r.fit_campeao:.1f} | {r.excesso_wf:+.1f} | {r.dm_p_campeao_vs_ar1:.3f} | "
            f"{'sim' if r.exogena else ''} |"
        )
    linhas.append("")
    linhas += ["", "| Combinação | d | FIT% ARIMA(1,d,0) | U2 | excesso | DM p | rho1 estac. | "
               "FIT% camp. estac. | FIT%_pop | exc. pop | MASE camp/RW/ref/saz | FIT% reg. (U2) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in df.itertuples():
        reg = (f"{r.fit_campeao_registrado:.1f} ({r.u2_campeao_registrado:.3f})"
               if hasattr(r, "fit_campeao_registrado") else "")
        linhas.append(
            f"| {r.empresa}x{r.indicador} | {r.d} | {r.fit_ref_wf:.1f} | {r.u2_ref_wf:.3f} | "
            f"{r.excesso_ref_wf:+.1f} | {r.dm_p_campeao_vs_ref:.3f} | {r.rho1_estac:.2f} | "
            f"{r.fit_campeao_estac:.1f} | {r.fit_pop_estac:.1f} | {r.excesso_pop_estac:+.1f} | "
            f"{r.mase_campeao:.2f}/{r.mase_rw:.2f}/{r.mase_ref:.2f}/{r.mase_sazonal:.2f} | {reg} |")
    linhas.append("")
    linhas.append(
        f"ARIMA(1,d,0): Pearson {s['pearson_fit_campeao_ref']:.3f}, Spearman "
        f"{s['spearman_fit_campeao_ref']:.3f}; excesso médio {s['excesso_ref_medio']:+.1f} "
        f"(exógena {s['excesso_ref_medio_exog']:+.1f}, univariado "
        f"{s['excesso_ref_medio_univ']:+.1f}); U2<1: {s['ref_passa_gate']}/12; DM p<0.10: "
        f"{s['dm_ref_rejeita_10pct']}/12. Excesso populacional estacionário: exógena "
        f"{s['excesso_pop_medio_exog']:+.1f}, univariado {s['excesso_pop_medio_univ']:+.1f}.")
    linhas.append(
        f"FIT% campeão x AR(1): Pearson {s['pearson_fit_campeao_ar1']:.3f}, Spearman "
        f"{s['spearman_fit_campeao_ar1']:.3f}. Excesso médio {s['excesso_medio']:+.1f} "
        f"(exógena {s['excesso_medio_exog']:+.1f}, univariado {s['excesso_medio_univ']:+.1f}). "
        f"AR(1) com U2<1: {s['ar1_passa_gate']}/12. DM campeão x AR(1) p<0.10: "
        f"{s['dm_rejeita_10pct']}/12."
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
    print(as_markdown(df, summary(df)))

    if args.tratamento == "vigente":
        return

    ccf = _CCF_CACHE["df"]
    fora = ccf.groupby(["empresa", "indicador", "forma"])["fora_da_banda"].agg(["sum", "count"])
    print("\nCorrelação cruzada erro x regressor (defasagens 0-4), células fora da banda de 95%:")
    print(fora.to_string())

    saidas = ((paths.JOURNAL_AR1_WALKFORWARD_CSV, df), (paths.JOURNAL_RESID_INPUT_CCF_CSV, ccf))
    falhou = False
    for destino, tabela in saidas:
        texto = tabela.to_csv(index=False, lineterminator="\n")
        if args.check:
            if not destino.exists() or destino.read_text(encoding="utf-8") != texto:
                print(f"\nFALHA: {destino.name} ausente ou divergiu do recomputado.",
                      file=sys.stderr)
                falhou = True
            else:
                print(f"\nOK: {destino.name} reproduz.")
            continue
        destino.write_text(texto, encoding="utf-8")
        print(f"\nGravado: {destino.relative_to(paths.ROOT)}")
    if falhou:
        sys.exit(1)

if __name__ == "__main__":
    main()
