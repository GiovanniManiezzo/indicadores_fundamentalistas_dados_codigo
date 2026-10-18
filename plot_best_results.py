#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real vs Previsto (walk-forward, h=1) do campeao de cada combinacao empresa x indicador.

A tabela de campeoes NAO e escrita a mao aqui: vem de `scripts/_common.compute_champions()`,
a mesma funcao que gera a tabela de campeoes — logo o grafico nao pode divergir do painel (era o caso
antes do Round 25, quando este arquivo ainda plotava um SS com U2=1.041 para TAEE4xres_bruto).

Cada campeao e reproduzido sob a configuracao vigente de `config.py`: `known_spikes_for()`
(SPIKE_KNOWN menos SPIKE_REVERTED, protocolo metodológico sec.27) + deteccao automatica z>5. Nao ha
mais "era de spike" a replicar — `cfg.TREATMENT_ERA` garante que todo campeao eleito ja
pertence a era vigente.

Saida (paths.PLOTS_DIR):
  outputs/plots/real_vs_previsto_melhor_modelo.png   painel 4x3
  outputs/plots/<TICKER>_<indicador>.png             um por combinacao

Uso:
  python plot_best_results.py            # painel + os 12 individuais
  python plot_best_results.py --panel    # so o painel
"""
from __future__ import annotations
import sys, os, re, warnings
from contextlib import contextmanager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import paths

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config as cfg
from _common import compute_champions, load_json, RESULTS_DIR
from src.loaders import load_all
from src.preprocess import build_ls_dummies, transform, unit_root_tests
from src.treatment import treat_series
from src.evaluation import walk_forward_cv, fit_percent
from src.models.arima_wrapper import ARIMAWrapper
from src.models.armax_wrapper import ARMAXWrapper
from src.models.arimax_wrapper import ARIMAXWrapper
from src.models.bayesarima_wrapper import BayesArimaWrapper
from src.models.ss_wrapper import SSWrapper
from src.models.oe_wrapper import OEWrapper
from src.models.bj_wrapper import BJWrapper
from src.models.wiener_wrapper import WienerWrapper
from src.models.hammerstein_wrapper import HammersteinWrapper
from src.models.theta_wrapper import ThetaWrapper
from src.models.ets_wrapper import ETSWrapper
from src.models.dampedtrend_wrapper import DampedTrendWrapper
from src.models.combination_wrapper import CombinationWrapper

warnings.filterwarnings("ignore")
cfg.apply_global_seed()

IND_SHORT = {
    "resultado_bruto":       "Resultado Bruto",
    "lucro_consolidado":     "Lucro Consolidado",
    "despesas_operacionais": "Despesas Operacionais",
}

MODEL_COLORS = {
    "ARIMA": "#2166ac", "ARMAX": "#4dac26", "ARIMAX": "#1b7837", "BAYESARIMA": "#b35806",
    "SS": "#762a83", "WIENER": "#d6604d", "HAMMERSTEIN": "#f4a582", "THETA": "#8073ac",
    "ETS": "#5aae61", "DAMPEDTREND": "#e08214", "COMBINATION": "#35978f",
}

# FIT% recalculado pode divergir do gravado por ruido numerico de otimizador; acima disto e bug.
FIT_TOLERANCE_PP = 0.5


SPECS_DIR = str(paths.SPECS_DIR)


@contextmanager
def round_hyperparams(result_file: str):
    """Reaplica os `hyperparams:` da spec que produziu `result_file`, e restaura ao sair.

    Necessario porque os wrappers NL leem `cfg.NL_P_MAX` / `cfg.ALS_*` em tempo de `fit`:
    reproduzir o campeao do Round 26 (Wiener-W3-BSpline3) com os defaults do config da um
    FIT 12 p.p. menor que o gravado. Sem isto o grafico nao mostra o modelo que venceu.
    """
    import yaml

    m = re.search(r"_v(\d+)", result_file)
    spec_path = os.path.join(SPECS_DIR, f"round_{int(m.group(1))}.yaml") if m else ""
    hyper = {}
    if spec_path and os.path.exists(spec_path):
        with open(spec_path, encoding="utf-8") as fh:
            hyper = yaml.safe_load(fh).get("hyperparams") or {}
    with hyperparams_applied(hyper, label="hyperparams da spec"):
        yield


@contextmanager
def hyperparams_applied(hyper: "dict | None", label: str = "hyperparams"):
    """Aplica `hyper` (chaves de cfg.HYPERPARAM_WHITELIST) ao config e restaura ao sair."""
    hyper = hyper or {}
    saved = {}
    try:
        for key, value in hyper.items():
            attr = cfg.HYPERPARAM_WHITELIST[key]
            saved[attr] = getattr(cfg, attr)
            setattr(cfg, attr, value)
        if hyper:
            print(f"    [{label}] {hyper}")
        yield
    finally:
        for attr, value in saved.items():
            setattr(cfg, attr, value)


def _family(model_name: str) -> str:
    """'WIENER-W3-BSpline3' -> 'WIENER'; 'COMBINATION-INVMSE' -> 'COMBINATION'.

    main.py grava o Hammerstein como 'HAMMER-H2-Piecewise'; normalizado para a familia
    (Round 34 elegeu o primeiro Hammerstein campeao, ISAE4xresultado_bruto).
    """
    fam = model_name.split("-")[0].upper()
    return "HAMMERSTEIN" if fam == "HAMMER" else fam


def _order_from(desc: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    m = re.search(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", desc)
    return (int(m[1]), int(m[2]), int(m[3])) if m else default


def _prepare_series_log(
    emp_code: int, ind: str, datasets: dict, fixed_rule: bool = False, manuscript: bool = False
) -> "tuple[pd.Series, list[pd.Timestamp]]":
    """Serie tratada + as datas imputadas (a mascara de pontuacao do Round 40).

    `fixed_rule=True`: regra pre-fixada do Round 39 (protocolo metodológico sec.30) — lista inteira, sem
    reversoes, imputacao por cfg.FIXED_RULE_IMPUTE_METHOD, celulas de cfg.FIXED_RULE_RAW_CELLS brutas.
    `manuscript=True`: Round 40 (sec.31) — a regra anterior mais as correcoes de dado de
    cfg.Q4_REBASE_OVERRIDES, aplicadas antes da imputacao e nunca imputadas.
    """
    series = datasets[ind][str(emp_code)].dropna()
    series, records = treat_series(series, emp_code, ind, fixed_rule=fixed_rule,
                                   manuscript=manuscript)
    return series, [pd.Timestamp(r["timestamp"]) for r in records]


def _prepare_series(
    emp_code: int, ind: str, datasets: dict, fixed_rule: bool = False, manuscript: bool = False
) -> pd.Series:
    """Serie tratada exatamente como `_run_spec_full` a trata (config vigente)."""
    return _prepare_series_log(emp_code, ind, datasets, fixed_rule, manuscript)[0]


def _build_exog(series: pd.Series, exog_cols: list[str], inputs: pd.DataFrame):
    """Reconstroi u_exog a partir das colunas que o modelo campeao de fato consumiu.

    `exogenas_utilizadas` no JSON e a lista exata usada na rodada — inclusive dummies LS
    (`LS_YYYYMMDD`), que sao regeneradas aqui em vez de lidas de um arquivo.
    """
    if not exog_cols:
        return None
    parts = []
    macro = [c for c in exog_cols if c in inputs.columns]
    if macro:
        parts.append(inputs[macro].reindex(series.index).ffill().bfill())
    ls_dates = [
        f"{c[3:7]}-{c[7:9]}-{c[9:11]}" for c in exog_cols if re.fullmatch(r"LS_\d{8}", c)
    ]
    if ls_dates:
        parts.append(build_ls_dummies(series.index, ls_dates))
    unknown = [
        c for c in exog_cols
        if c not in inputs.columns and not re.fullmatch(r"LS_\d{8}", c)
    ]
    if unknown:
        print(f"    [AVISO] exogenas nao encontradas nos inputs: {unknown}")
    return pd.concat(parts, axis=1) if parts else None


def _make_model_fn(rec: dict, series: pd.Series, u_exog, nl_fixed_tau: float | None = None):
    """Factory do campeao, com ordem/spec congelada na janela inicial (anti-leakage).

    `nl_fixed_tau` (Round 36): congela o breakpoint do bloco piecewise de Wiener/Hammerstein;
    None mantem o comportamento do campeao (tau reescolhido por grade a cada dobra).
    """
    fam = _family(rec["modelo_utilizado"])
    mode = rec["preprocessing"]
    desc = rec["descricao_parametros"]
    lam = rec.get("boxcox_lambda") if mode == "boxcox" else None
    d = rec.get("unit_root_d")
    if d is None:
        d = unit_root_tests(transform(series, mode, lam))["consensus_d"]

    init = series.iloc[:cfg.MIN_TRAIN]
    u0 = u_exog.iloc[:cfg.MIN_TRAIN] if u_exog is not None else None

    if fam == "ARIMA":
        fo = _order_from(desc, (1, d, 1))
        return lambda yt, ut: ARIMAWrapper(
            d=d, mode=mode, fixed_order=fo, boxcox_lambda=lam).fit(yt)

    if fam == "ARMAX":
        fo = _order_from(desc, (1, 0, 0))
        return lambda yt, ut: ARMAXWrapper(
            mode=mode, fixed_order=fo, boxcox_lambda=lam).fit(yt, ut)

    if fam == "ARIMAX":
        fo = _order_from(desc, (0, d, 1))
        return lambda yt, ut: ARIMAXWrapper(
            d=d, mode=mode, fixed_order=fo, boxcox_lambda=lam).fit(yt, ut)

    if fam == "BAYESARIMA":
        fo = _order_from(desc, (0, d, 1))
        m = re.search(r"lambda0=([\d.]+)", desc)
        lam0 = float(m[1]) if m else 1.0
        return lambda yt, ut: BayesArimaWrapper(
            d=d, mode=mode, fixed_order=fo, fixed_lambda0=lam0, boxcox_lambda=lam).fit(yt, ut)

    if fam == "SS":
        spec = SSWrapper(mode=mode, boxcox_lambda=lam).fit(init, u0).fixed_spec_
        return lambda yt, ut: SSWrapper(mode=mode, fixed_spec=spec, boxcox_lambda=lam).fit(yt, ut)

    # OE/BJ: espelham main.py::_make_oe_fn/_make_bj_fn (ordem gravada na descricao; Round 31
    # elegeu o primeiro BJ campeao, ALUP4xdespesas_operacionais).
    if fam == "OE":
        fo = _order_from(desc, (1, d, 0))
        return lambda yt, ut: OEWrapper(
            d=d, mode=mode, fixed_order=fo, boxcox_lambda=lam).fit(yt, ut)

    if fam == "BJ":
        fo = _order_from(desc, (0, d, 1))
        return lambda yt, ut: BJWrapper(
            d=d, mode=mode, fixed_order=fo, boxcox_lambda=lam).fit(yt, ut)

    if fam == "ETS":
        spec = ETSWrapper(mode=mode).fit(init).fixed_spec_
        return lambda yt, ut: ETSWrapper(mode=mode, fixed_spec=spec).fit(yt)

    if fam == "THETA":
        return lambda yt, ut: ThetaWrapper(mode=mode).fit(yt)

    if fam == "DAMPEDTREND":
        return lambda yt, ut: DampedTrendWrapper(mode=mode).fit(yt)

    if fam == "COMBINATION":
        # Espelha main.py::_combination — pool fixo, specs e pesos INVMSE congelados na
        # janela inicial (mesmo padrao anti-leakage de fixed_order/fixed_spec_).
        name = rec["modelo_utilizado"]
        scheme = name.split("-", 1)[1] if "-" in name else "MEAN"
        arima_fo = ARIMAWrapper(d=d, mode=mode).fit(init).order
        ets_spec = ETSWrapper(mode=mode).fit(init).fixed_spec_
        ss_spec = SSWrapper(mode=mode).fit(init, u0).fixed_spec_

        def _arima(yt, ut):
            return ARIMAWrapper(d=d, mode=mode, fixed_order=arima_fo).fit(yt, ut)

        def _ets(yt, ut):
            return ETSWrapper(mode=mode, fixed_spec=ets_spec).fit(yt)

        def _theta(yt, ut):
            return ThetaWrapper(mode=mode).fit(yt)

        def _ss(yt, ut):
            return SSWrapper(mode=mode, fixed_spec=ss_spec).fit(yt, ut)

        def _damped(yt, ut):
            return DampedTrendWrapper(mode=mode).fit(yt)

        sub_fns = {"ARIMA": _arima, "ETS": _ets, "THETA": _theta,
                   "SS": _ss, "DAMPEDTREND": _damped}
        weights = None
        if scheme == "INVMSE":
            weights, inner_min = {}, max(cfg.MIN_TRAIN - 5, 10)
            for sub_name, sfn in sub_fns.items():
                try:
                    mse = walk_forward_cv(sfn, init, None, min_train=inner_min)["rmse"] ** 2
                    weights[sub_name] = (1.0 / mse) if mse > 0 else 0.0
                except Exception:
                    weights[sub_name] = 0.0
        return lambda yt, ut: CombinationWrapper(
            sub_fns, scheme=scheme, weights=weights).fit(yt, ut)

    if fam in ("WIENER", "HAMMERSTEIN"):
        # 'WIENER-W3-BSpline3' -> config 'W3-BSpline3' em cfg.WIENER_CONFIGS
        cfg_name = rec["modelo_utilizado"].split("-", 1)[1]
        pool = cfg.WIENER_CONFIGS if fam == "WIENER" else cfg.HAMMERSTEIN_CONFIGS
        entry = next(c for c in pool if c["name"] == cfg_name)
        kw = {"output_nl": entry["output_nl"]} if fam == "WIENER" else {"input_nl": entry["input_nl"]}
        kw["n_knots"] = entry["n_knots"]
        Wrapper = WienerWrapper if fam == "WIENER" else HammersteinWrapper
        try:  # _select_nl_order: ordem do bloco linear vem do ajuste SEM exogenas
            fo = Wrapper(d=d, mode=mode, **kw).fit(init)._linear.order
        except Exception:
            fo = (1, d, 0)
        if nl_fixed_tau is not None:  # so nas dobras: a ordem continua a do campeao
            kw["nl_fixed_tau"] = nl_fixed_tau
        return lambda yt, ut: Wrapper(d=d, mode=mode, fixed_order=fo, **kw).fit(yt, ut)

    raise ValueError(f"Familia de modelo sem factory no plot: {fam}")


def run_combo(emp_code: int, ind: str, champ: dict, datasets: dict,
              inputs: pd.DataFrame, manuscript: bool = False, score: str = "exclude") -> dict:
    """Reproduz o campeao congelado de uma combinacao.

    `manuscript=True` (Round 40, sec.31): serie do tratamento do manuscrito. `score` define a
    pontuacao nesse modo — "exclude" tira os alvos imputados das metricas; "recorded" pontua todos
    os alvos contra o valor registrado (sensibilidade); "imputed" e' o gabarito imputado.
    """
    # Round 41 (sec.32): o campeao reeleito vem com o registro embutido e a grade de
    # hiperparametros em que venceu; os demais sao lidos do JSON da rodada que os produziu.
    inline = "record" in champ
    rec = champ["record"] if inline else load_json(RESULTS_DIR / champ["file"])[str(emp_code)][ind]
    series, imputed = _prepare_series_log(emp_code, ind, datasets, manuscript=manuscript)
    u_exog = _build_exog(series, rec.get("exogenas_utilizadas", []), inputs)
    kw: dict = {}
    if manuscript and score == "exclude":
        kw["score_exclude"] = imputed
    elif manuscript and score == "recorded":
        recorded = datasets[ind][str(emp_code)].dropna().copy()
        for date, value in cfg.Q4_REBASE_OVERRIDES.get((emp_code, ind), {}).items():
            recorded.loc[pd.Timestamp(date)] = value
        kw["y_score"] = recorded
    hp_ctx = (hyperparams_applied(rec.get("hyperparams"), label="grade da busca") if inline
              else round_hyperparams(champ["file"]))
    with hp_ctx:
        cv = walk_forward_cv(_make_model_fn(rec, series, u_exog), series, u_exog,
                             min_train=cfg.MIN_TRAIN, **kw)
    mask = cv.get("score_mask")
    oos = series.index[cfg.MIN_TRAIN:]
    return {
        "index_full": series.index, "y_full": series.values,
        "index_oos": oos if mask is None else oos[mask],
        "y_true": cv["y_true"], "y_pred": cv["y_pred"], "y_rw": cv["y_rw"],
        "u2_actual": cv["theil_u2"], "n_excluded": cv.get("n_excluded", 0),
        "imputed": imputed,
        "fit_stored": rec["percentual_fit"],
        "fit_actual": fit_percent(cv["y_true"], cv["y_pred"]),
        "u2": rec["theil_u2"], "dm_p": (rec.get("dm_vs_arima") or {}).get("pvalue"),
        "model": rec["modelo_utilizado"], "family": _family(rec["modelo_utilizado"]),
        "mode": rec["preprocessing"], "exog": rec.get("exogenas_utilizadas", []),
        "desc": rec["descricao_parametros"], "file": champ["file"],
    }


def _label(res: dict) -> str:
    lbl = res["desc"]
    if res["mode"] != "raw":
        lbl += f" [{res['mode']}]"
    if res["exog"]:
        lbl += f" + {len(res['exog'])} exog."
    return lbl


def _draw(ax, res: dict, ticker: str, ind: str, compact: bool) -> None:
    scale = 1e9
    dates_full, dates_oos = res["index_full"], res["index_oos"]
    color = MODEL_COLORS.get(res["family"], "#555555")
    n_train = cfg.MIN_TRAIN

    ax.axvspan(dates_full[0], dates_full[n_train - 1], alpha=0.08, color="gray")
    ax.plot(dates_full, res["y_full"] / scale, color="#999999", linewidth=1.0, zorder=2)
    ax.plot(dates_oos, res["y_true"] / scale, color="#222222", linewidth=1.8, zorder=3)
    ax.plot(dates_oos, res["y_pred"] / scale, color=color, linewidth=1.6,
            linestyle="--", zorder=4)
    ax.axvline(x=dates_full[n_train], color="#555555", linewidth=0.8,
               linestyle=":", alpha=0.7)

    dm = f"{res['dm_p']:.3f}" if isinstance(res["dm_p"], (int, float)) else "-"
    stats = f"FIT {res['fit_actual']:.1f}%  |  U2 {res['u2']:.3f}  |  DM-p {dm}"
    fs = 8.5 if compact else 11
    ax.set_title(f"{ticker} — {IND_SHORT[ind]}\n{_label(res)}\n{stats}",
                 fontsize=fs, fontweight="bold", linespacing=1.4)
    ax.set_ylabel("R$ Bilhoes", fontsize=7.5 if compact else 10)
    ax.tick_params(axis="x", labelsize=7 if compact else 9, rotation=30)
    ax.tick_params(axis="y", labelsize=7.5 if compact else 9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))


def main() -> None:
    panel_only = "--panel" in sys.argv
    plots_dir = str(paths.PLOTS_DIR)
    os.makedirs(plots_dir, exist_ok=True)

    print("Carregando dados...")
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc,
                "despesas_operacionais": do}
    champions = compute_champions()

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 20), constrained_layout=True)
    fig.suptitle(
        "Real vs Previsto — Campeao por Combinacao (Walk-Forward CV, h=1, MIN_TRAIN=20)\n"
        "Area cinza = janela de treino inicial | Linha tracejada = previsao out-of-sample",
        fontsize=14, fontweight="bold", y=1.01,
    )

    mismatches, idx = [], 0
    for row, emp_code in enumerate(cfg.EMPRESAS):
        ticker = cfg.EMPRESAS[emp_code]
        for col, ind in enumerate(cfg.INDICADORES):
            idx += 1
            ax = axes[row, col]
            champ = champions[(emp_code, ind)]
            print(f"[{idx:2d}/12] {ticker} x {ind} — {champ['model']} ...", flush=True)
            try:
                res = run_combo(emp_code, ind, champ, datasets, inputs)
            except Exception as exc:
                ax.text(0.5, 0.5, f"ERRO:\n{exc}", transform=ax.transAxes,
                        ha="center", va="center", color="red", fontsize=8)
                ax.set_title(f"{ticker} — {IND_SHORT[ind]}", fontsize=9)
                mismatches.append((ticker, ind, f"ERRO: {exc}"))
                continue

            delta = res["fit_actual"] - res["fit_stored"]
            if abs(delta) > FIT_TOLERANCE_PP:
                mismatches.append((ticker, ind,
                                   f"FIT gravado {res['fit_stored']:.2f} vs recalculado "
                                   f"{res['fit_actual']:.2f} (delta={delta:+.2f} p.p.)"))
            _draw(ax, res, ticker, ind, compact=True)

            if not panel_only:
                f1, a1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
                _draw(a1, res, ticker, ind, compact=False)
                a1.legend(handles=[
                    mpatches.Patch(color="#999999", label="Real (serie completa)"),
                    mpatches.Patch(color="#222222", label="Real (out-of-sample)"),
                    mpatches.Patch(color=MODEL_COLORS.get(res["family"], "#555"),
                                   label=f"Previsto — {res['model']}"),
                    mpatches.Patch(color="#cccccc",
                                   label=f"Treino inicial ({cfg.MIN_TRAIN} obs)"),
                ], fontsize=8, loc="best", framealpha=0.9)
                f1.savefig(os.path.join(plots_dir, f"{ticker}_{ind}.png"),
                           dpi=150, bbox_inches="tight")
                plt.close(f1)

    fig.legend(handles=[
        mpatches.Patch(color="#999999", label="Serie real completa"),
        mpatches.Patch(color="#222222", label="Real OOS (walk-forward)"),
        mpatches.Patch(color="#aaaaaa", alpha=0.3,
                       label=f"Janela de treino ({cfg.MIN_TRAIN} obs)"),
    ] + [mpatches.Patch(color=c, label=n) for n, c in MODEL_COLORS.items()],
        loc="lower center", ncol=5, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    out = str(paths.PANEL_PLOT)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPainel salvo em: {out}")
    if not panel_only:
        print(f"12 graficos individuais em: {plots_dir}")

    if mismatches:
        print("\n[!] Campeoes que NAO reproduziram o valor gravado:")
        for t, i, msg in mismatches:
            print(f"    {t}x{i}: {msg}")
        sys.exit(1)
    print(f"\nOK — os 12 campeoes reproduziram o FIT% gravado dentro de "
          f"{FIT_TOLERANCE_PP} p.p.")


if __name__ == "__main__":
    main()
