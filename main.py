"""Orquestrador do pipeline de identificação de sistemas.

Modos:
  python main.py --round spec rounds/specs/round_NN.yaml   → rodada declarada em YAML (padrão)
  python main.py --round baseline      → Fase 2: ARIMA + RW baseline
  python main.py --round full          → Fase 3: todos os modelos + exógenas
  python main.py --compare v1.json v2.json
  python main.py --report vN.json [--output relatorio.md]

O fluxo normal é o spec-driven: uma rodada é um YAML em rounds/specs/.
Os modos legados (wiener[-arx], hammerstein[-arx], hw[-arx], ls-dummies,
hammerstein-ipam) são mantidos para reprodutibilidade exata dos Rounds 01-09.
"""
from __future__ import annotations
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import paths
from src.loaders import load_all
from src.preprocess import (
    transform, unit_root_tests, detect_and_impute_spikes, build_ls_dummies, guerrero_lambda,
)
from src.evaluation import (
    walk_forward_cv,
    residual_diagnostics,
    diebold_mariano_test,
    restrict_to_eval_range,
)
from src.models.arima_wrapper import ARIMAWrapper
from src.models.rw_wrapper import RandomWalkModel, RandomWalkDriftModel
from src.models.sarima_wrapper import SARIMAWrapper
from src.models.armax_wrapper import ARMAXWrapper
from src.models.arimax_wrapper import ARIMAXWrapper
from src.models.oe_wrapper import OEWrapper
from src.models.bj_wrapper import BJWrapper
from src.models.ss_wrapper import SSWrapper
from src.models.narx_wrapper import NARXWrapper
from src.models.narimax_wrapper import NARIMAXWrapper
from src.models.lstm_wrapper import LSTMWrapper
from src.models.ets_wrapper import ETSWrapper
from src.models.theta_wrapper import ThetaWrapper
from src.models.dampedtrend_wrapper import DampedTrendWrapper
from src.models.combination_wrapper import CombinationWrapper
from src.models.bayesarima_wrapper import BayesArimaWrapper
from src.models.wiener_wrapper import WienerWrapper
from src.models.hammerstein_wrapper import HammersteinWrapper
from src.models.hw_wrapper import HWWrapper
from src.models.setar_wrapper import SETARWrapper
from src.save_results import save_results

_ROUNDS_DIR = Path("rounds")
_IND_SHORT  = {
    "resultado_bruto":       "res_bruto",
    "lucro_consolidado":     "luc_consol",
    "despesas_operacionais": "desp_oper",
}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _append_science_log(text: str) -> None:
    path = _ROUNDS_DIR / "science_log.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _make_arima_fn(d: int, mode: str, fixed_order: tuple | None = None, lam: float | None = None):
    """Closure com ordem pré-selecionada para evitar re-grid-search em cada fold."""
    def fn(y_train: "pd.Series", u_train=None) -> ARIMAWrapper:  # type: ignore[name-defined]
        return ARIMAWrapper(
            d=d, mode=mode, fixed_order=fixed_order, boxcox_lambda=lam
        ).fit(y_train, u_train)
    return fn


def _select_arima_order(
    series: "pd.Series", d: int, mode: str, min_train: int, lam: float | None = None  # type: ignore[name-defined]
) -> tuple[int, int, int]:
    """Seleciona (p,d,q) via AIC na janela inicial — ordem congelada para todos os folds."""
    selector = ARIMAWrapper(d=d, mode=mode, boxcox_lambda=lam).fit(series.iloc[:min_train])
    return selector.order


# ---------------------------------------------------------------------------
# Fase 2 — Baseline ARIMA + Random Walk
# ---------------------------------------------------------------------------

def run_baseline() -> None:
    cfg.apply_global_seed()
    rb, lc, do, _ = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    results: dict = {}
    col_w = 75
    print("=" * col_w)
    print(f"{'Empresa':<8} {'Indicador':<22} {'Modo':<8} {'FIT%':>6} {'RMSE':>12} {'U₂':>6} {'LB-p':>5} {'Ordem':<16} {'RW%':>6}")
    print("-" * col_w)

    for emp_code, ticker in cfg.EMPRESAS.items():
        results[str(emp_code)] = {}
        for ind in cfg.INDICADORES:
            series = datasets[ind][str(emp_code)].dropna()

            # --- RW e RW+Drift (escala original, modo-independente) ---
            rw_cv  = walk_forward_cv(lambda y, u: RandomWalkModel().fit(y, u),      series, min_train=cfg.MIN_TRAIN)
            rwd_cv = walk_forward_cv(lambda y, u: RandomWalkDriftModel().fit(y, u), series, min_train=cfg.MIN_TRAIN)

            # --- ARIMA: testar os 4 modos de pré-processamento ---
            best_mode: str | None = None
            best_fit  = -np.inf
            best_cv: dict | None  = None
            best_d    = 1
            best_order_pre: tuple | None = None
            urt_conflict: list[str] = []

            for mode in cfg.PREPROC_MODES:
                try:
                    y_t = transform(series, mode)
                    urt = unit_root_tests(y_t)
                    d   = urt["consensus_d"]

                    # Registrar conflito ADF×KPSS×PP se necessário
                    conclusions = {k: urt[k]["conclusion"] for k in ("ADF", "KPSS", "PP")}
                    unique = set(c for c in conclusions.values() if c != "not_available")
                    if len(unique) > 1:
                        urt_conflict.append(
                            f"{ticker}×{_IND_SHORT[ind]} mode={mode}: "
                            + " / ".join(f"{k}={v}" for k, v in conclusions.items())
                        )

                    # Pré-selecionar (p,q) na janela inicial → congela para todos os folds
                    fixed_order = _select_arima_order(series, d, mode, cfg.MIN_TRAIN)
                    cv = walk_forward_cv(
                        _make_arima_fn(d, mode, fixed_order),
                        series,
                        min_train=cfg.MIN_TRAIN,
                    )
                    if cv["fit_percent"] > best_fit:
                        best_fit  = cv["fit_percent"]
                        best_mode = mode
                        best_cv   = cv
                        best_d    = d
                        best_order_pre = fixed_order
                except Exception as exc:
                    warnings.warn(f"[{ticker}×{ind}] mode={mode}: {exc}")

            if best_cv is None:
                warnings.warn(f"[{ticker}×{ind}] Todos os modos falharam; combinação ignorada.")
                continue

            # Fit final na série completa para order string (reutiliza ordem pré-selecionada)
            try:
                final = ARIMAWrapper(d=best_d, mode=best_mode, fixed_order=best_order_pre).fit(series)
                order_str = final.get_params_str()
            except Exception:
                order_str = f"ARIMA(?,{best_d},?)"

            diag = residual_diagnostics(best_cv["residuals"])

            results[str(emp_code)][ind] = {
                "modelo_utilizado":    "ARIMA",
                "descricao_parametros": order_str,
                "preprocessing":       best_mode,
                "exogenas_utilizadas": [],
                "percentual_fit":      float(best_cv["fit_percent"]),
                "variancia":           float(best_cv["variance"]),
                "mse":                 float(best_cv["mse"]),
                "rmse":                float(best_cv["rmse"]),
                "mae":                 float(best_cv["mae"]),
                "mape":                float(best_cv["mape"]),
                "theil_u2":            float(best_cv["theil_u2"]),
                "unit_root_d":         int(best_d),
                "residual_diagnostics": diag,
                # ARIMA vs ARIMA: DM trivialmente zero
                "dm_vs_arima":          {"stat": 0.0, "pvalue": 1.0},
                "in_model_confidence_set": True,
            }

            print(
                f"{ticker:<8} {ind:<22} {best_mode:<8} "
                f"{best_cv['fit_percent']:>6.1f} {best_cv['rmse']:>12.2e} "
                f"{best_cv['theil_u2']:>6.3f} {diag['ljung_box_q4_pvalue']:>5.3f} "
                f"{order_str:<16} {rw_cv['fit_percent']:>6.1f}"
            )

            if urt_conflict:
                for msg in urt_conflict:
                    print(f"  [CONFLITO URT] {msg}")

    print("=" * col_w)

    json_path = save_results(results, version=cfg.ROUND_VERSION)

    # --- Append no science log ---
    ts = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            r = results.get(str(emp_code), {}).get(ind)
            if r:
                rows.append(
                    f"| {ticker}×{_IND_SHORT[ind]:<14} "
                    f"| {r['percentual_fit']:>5.1f} "
                    f"| {r['theil_u2']:>8.3f} "
                    f"| {r['residual_diagnostics']['ljung_box_q4_pvalue']:>4.2f} "
                    f"| — "
                    f"| {r['descricao_parametros']:<16} "
                    f"| {r['preprocessing']} |"
                )

    log_entry = (
        f"\n## Round 01 — {ts}\n"
        f"**Hipótese:** ARIMA como baseline; melhor pré-proc selecionado por FIT% walk-forward.\n"
        f"| Combinação | FIT% | Theil U₂ | LB p | DM vs ARIMA p | Modelo | Pré-proc |\n"
        f"|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
        f"**Decisões:** d via ADF/KPSS/PP; grid p,q∈[0,4] por AIC; U₂<1 requerido.\n"
        f"**Próximo:** Fase 3 — seleção de exógenas (VIF+Granger+AIC) e modelos completos.\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nScience log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


# ---------------------------------------------------------------------------
# Fase 3 — Helpers
# ---------------------------------------------------------------------------

def _find_latest_result() -> Path | None:
    """Retorna o JSON de resultado com maior número de versão em outputs/results/.

    Ordena pelo número extraído de '_vN.json'. Em empate de versão, usa mtime.
    Exclui arquivos de exog_selection.
    """
    import re as _re
    results_dir = paths.RESULTS_DIR
    jsons = [j for j in results_dir.glob("*.json") if "exog" not in j.name]
    if not jsons:
        return None

    def _version(p: Path) -> int:
        m = _re.search(r"_v(\d+)\.json$", p.name)
        return int(m.group(1)) if m else 0

    jsons.sort(key=lambda p: (_version(p), p.stat().st_mtime))
    return jsons[-1]


def _make_result_dict(
    model_name: str,
    params_str: str,
    mode: str,
    exog_cols: list,
    d: int,
    cv: dict,
    boxcox_lambda: float | None = None,
) -> dict:
    """Constrói dict compatível com schema_resultados.json (+ _residuals interno)."""
    diag = residual_diagnostics(cv["residuals"])
    r = {
        "modelo_utilizado":     model_name,
        "descricao_parametros": params_str,
        "preprocessing":        mode,
        "exogenas_utilizadas":  list(exog_cols),
        "percentual_fit":       float(cv["fit_percent"]),
        "variancia":            float(cv["variance"]),
        "mse":                  float(cv["mse"]),
        "rmse":                 float(cv["rmse"]),
        "mae":                  float(cv["mae"]),
        "mape":                 float(cv["mape"]),
        "theil_u2":             float(cv["theil_u2"]),
        "unit_root_d":          int(d),
        "residual_diagnostics": diag,
        "dm_vs_arima":          {"stat": 0.0, "pvalue": 1.0},
        "in_model_confidence_set": True,
        "_residuals":           cv["residuals"],   # removido antes de salvar
        "_y_true":              cv.get("y_true"),  # idem — usado só por return_cv=True
        "_y_pred":              cv.get("y_pred"),
        "_y_rw":                cv.get("y_rw"),
    }
    if mode == "boxcox" and boxcox_lambda is not None:
        r["boxcox_lambda"] = float(boxcox_lambda)
    return r


# --- Factory functions (closures explícitas para evitar captura tardia) ---

def _make_sarima_fn(d_v, mode_v, fo_v, fs_v, lam_v=None):
    def fn(y_train, u_train):
        return SARIMAWrapper(
            d=d_v, mode=mode_v, fixed_order=fo_v, fixed_seasonal=fs_v, boxcox_lambda=lam_v
        ).fit(y_train)
    return fn


def _make_armax_fn(mode_v, fo_v, lam_v=None):
    def fn(y_train, u_train):
        return ARMAXWrapper(mode=mode_v, fixed_order=fo_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_arimax_fn(d_v, mode_v, fo_v, lam_v=None):
    def fn(y_train, u_train):
        return ARIMAXWrapper(
            d=d_v, mode=mode_v, fixed_order=fo_v, boxcox_lambda=lam_v
        ).fit(y_train, u_train)
    return fn


def _make_bayesarima_fn(d_v, mode_v, fo_v, lam_v, boxcox_lam_v=None):
    def fn(y_train, u_train):
        return BayesArimaWrapper(
            d=d_v, mode=mode_v, fixed_order=fo_v, fixed_lambda0=lam_v, boxcox_lambda=boxcox_lam_v
        ).fit(y_train, u_train)
    return fn


def _make_oe_fn(d_v, mode_v, fo_v, lam_v=None):
    def fn(y_train, u_train):
        return OEWrapper(d=d_v, mode=mode_v, fixed_order=fo_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_bj_fn(d_v, mode_v, fo_v, lam_v=None):
    def fn(y_train, u_train):
        return BJWrapper(d=d_v, mode=mode_v, fixed_order=fo_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_ss_fn(mode_v, fixed_spec_v=None, lam_v=None):
    def fn(y_train, u_train):
        return SSWrapper(mode=mode_v, fixed_spec=fixed_spec_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_narx_fn(mode_v, lam_v=None):
    def fn(y_train, u_train):
        return NARXWrapper(p=3, degree=2, mode=mode_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_narimax_fn(nm_v, lam_v=None):
    def fn(y_train, u_train):
        return NARIMAXWrapper(p=3, degree=2, mode=nm_v, boxcox_lambda=lam_v).fit(y_train, u_train)
    return fn


def _make_lstm_fn(mode_v):
    def fn(y_train, u_train):
        return LSTMWrapper(lookback=4, units=8, mode=mode_v).fit(y_train)
    return fn


def _make_ets_fn(mode_v, fixed_spec_v=None, lam_v=None):
    def fn(y_train, u_train):
        return ETSWrapper(mode=mode_v, fixed_spec=fixed_spec_v, boxcox_lambda=lam_v).fit(y_train)
    return fn


def _make_theta_fn(mode_v, lam_v=None):
    def fn(y_train, u_train):
        return ThetaWrapper(mode=mode_v, boxcox_lambda=lam_v).fit(y_train)
    return fn


def _make_dampedtrend_fn(mode_v, lam_v=None):
    def fn(y_train, u_train):
        return DampedTrendWrapper(mode=mode_v, boxcox_lambda=lam_v).fit(y_train)
    return fn


def _make_combination_fn(sub_fns_v: dict, scheme_v: str, weights_v: dict | None):
    def fn(y_train, u_train):
        return CombinationWrapper(sub_fns_v, scheme=scheme_v, weights=weights_v).fit(y_train, u_train)
    return fn


def _make_wiener_fn(output_nl_v, n_knots_v, d_v, mode_v, fo_v):
    def fn(y_train, u_train):
        return WienerWrapper(
            output_nl=output_nl_v, n_knots=n_knots_v,
            d=d_v, mode=mode_v, fixed_order=fo_v,
        ).fit(y_train, u_train)
    return fn


def _make_hammerstein_fn(input_nl_v, n_knots_v, d_v, mode_v, fo_v):
    def fn(y_train, u_train):
        return HammersteinWrapper(
            input_nl=input_nl_v, n_knots=n_knots_v,
            d=d_v, mode=mode_v, fixed_order=fo_v,
        ).fit(y_train, u_train)
    return fn


def _make_hw_fn(input_nl_v, output_nl_v, nki_v, nko_v, d_v, mode_v, fo_v):
    def fn(y_train, u_train):
        return HWWrapper(
            input_nl=input_nl_v, output_nl=output_nl_v,
            n_knots_in=nki_v, n_knots_out=nko_v,
            d=d_v, mode=mode_v, fixed_order=fo_v,
        ).fit(y_train, u_train)
    return fn


def _make_setar_fn(mode_v, fixed_config_v):
    def fn(y_train, u_train):
        return SETARWrapper(mode=mode_v, fixed_config=fixed_config_v).fit(y_train)
    return fn


def _run_all_models_for_combo(
    series: "pd.Series",  # type: ignore[name-defined]
    u_exog: "pd.DataFrame | None",  # type: ignore[name-defined]
    mode: str,
    d: int,
    exog_cols: list,
    ticker: str,
    ind: str,
    baseline_res: dict | None,
    boxcox_lambda: float | None = None,
    min_train_v: int | None = None,
    return_cv: bool = False,
    candidates_out: dict | None = None,
) -> dict:
    """Testa todos os modelos e retorna o melhor (U₂<1 prioritário, FIT% máximo).

    `candidates_out`: se dado, recebe todos os candidatos avaliados (com os arrays internos), não
    só o vencedor — usado pela reeleição do Round 41 (protocolo metodológico §32) para o SPA.

    `min_train_v`: janela de treino do walk-forward (default `cfg.MIN_TRAIN`). Só usado por
    `_run_window_sweep_round` (Round 24; protocolo metodológico §26) para testar janelas alternativas —
    o painel principal nunca passa este argumento, então seu comportamento é idêntico ao de
    antes por padrão.
    `return_cv`: se True, retorna `(best_r, cv_dict)` com os arrays brutos (y_true/y_pred/y_rw/
    residuals) do modelo vencedor, em vez de só `best_r` — usado para recalcular métricas sob
    um intervalo OOS fixo (`evaluation.restrict_to_eval_range`).
    """
    min_train_v = min_train_v if min_train_v is not None else cfg.MIN_TRAIN
    has_exog = u_exog is not None and not u_exog.empty

    candidates: dict[str, dict] = {}

    def _try(name: str, fn):
        try:
            candidates[name] = fn()
        except Exception as exc:
            warnings.warn(f"[{ticker}×{ind}] {name} ignorado: {exc}")

    # ── SARIMA ──────────────────────────────────────────────────────────
    def _sarima():
        init = SARIMAWrapper(d=d, p_max=1, q_max=1, mode=mode, boxcox_lambda=boxcox_lambda)
        init.fit(series.iloc[:min_train_v])
        mfn = _make_sarima_fn(d, mode, init.order, init.seasonal_order, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
        return _make_result_dict("SARIMA", init.get_params_str(), mode, [], d, cv, boxcox_lambda)
    _try("SARIMA", _sarima)

    # ── ETS ─────────────────────────────────────────────────────────────
    def _ets():
        init = ETSWrapper(mode=mode, boxcox_lambda=boxcox_lambda)
        init.fit(series.iloc[:min_train_v])
        mfn = _make_ets_fn(mode, init.fixed_spec_, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
        return _make_result_dict("ETS", init.get_params_str(), mode, [], d, cv, boxcox_lambda)
    _try("ETS", _ets)

    # ── THETA ───────────────────────────────────────────────────────────
    def _theta():
        mfn = _make_theta_fn(mode, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
        return _make_result_dict("THETA", "Theta", mode, [], d, cv, boxcox_lambda)
    _try("THETA", _theta)

    # ── DAMPEDTREND (Gardner & McKenzie 1985; MSOE structural, ver protocolo metodológico) ──
    def _dampedtrend():
        mfn = _make_dampedtrend_fn(mode, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
        return _make_result_dict("DAMPEDTREND", "DampedTrend", mode, [], d, cv, boxcox_lambda)
    _try("DAMPEDTREND", _dampedtrend)

    # ── COMBINATION (Bates & Granger 1969; Petropoulos et al. 2022 §2.6; ver protocolo metodológico) ──
    # Pool fixo ARIMA+ETS+THETA+SS+DAMPEDTREND, univariado, idêntico em todos os 12 combos.
    def _combination(scheme_name: str):
        def _run():
            y0 = series.iloc[:min_train_v]
            arima_init = ARIMAWrapper(d=d, mode=mode, boxcox_lambda=boxcox_lambda).fit(y0)
            ets_init   = ETSWrapper(mode=mode, boxcox_lambda=boxcox_lambda).fit(y0)
            ss_init    = SSWrapper(mode=mode, boxcox_lambda=boxcox_lambda).fit(y0)
            sub_fns = {
                "ARIMA":       _make_arima_fn(d, mode, arima_init.order, boxcox_lambda),
                "ETS":         _make_ets_fn(mode, ets_init.fixed_spec_, boxcox_lambda),
                "THETA":       _make_theta_fn(mode, boxcox_lambda),
                "SS":          _make_ss_fn(mode, ss_init.fixed_spec_, boxcox_lambda),
                "DAMPEDTREND": _make_dampedtrend_fn(mode, boxcox_lambda),
            }
            weights = None
            if scheme_name == "INVMSE":
                # MSE estimado uma única vez via CV interno na janela inicial, depois congelado
                # para todos os folds do walk-forward real — mesmo padrão anti-leakage do
                # fixed_order/fixed_spec_ .
                weights = {}
                inner_min_train = max(min_train_v - 5, 10)
                for name, fn in sub_fns.items():
                    try:
                        inner_cv = walk_forward_cv(fn, y0, None, min_train=inner_min_train)
                        mse = inner_cv["rmse"] ** 2
                        weights[name] = (1.0 / mse) if mse > 0 else 0.0
                    except Exception as exc:
                        warnings.warn(
                            f"[{ticker}×{ind}] COMBINATION-INVMSE: peso de {name} indisponível: {exc}"
                        )
                        weights[name] = 0.0
            combo_init = CombinationWrapper(sub_fns, scheme=scheme_name, weights=weights).fit(y0, None)
            mfn = _make_combination_fn(sub_fns, scheme_name, weights)
            cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
            return _make_result_dict(
                f"COMBINATION-{scheme_name}", combo_init.get_params_str(), mode, [], d, cv, boxcox_lambda
            )
        return _run
    for _scheme in cfg.COMBINATION_SCHEMES:
        _try(f"COMBINATION-{_scheme}", _combination(_scheme))

    if has_exog:
        u0 = u_exog.iloc[:min_train_v]

        # ── ARMAX ────────────────────────────────────────────────────────
        def _armax():
            init = ARMAXWrapper(
                p_max=cfg.SARIMAX_FAMILY_P_MAX, q_max=cfg.SARIMAX_FAMILY_Q_MAX, mode=mode,
                boxcox_lambda=boxcox_lambda,
            )
            init.fit(series.iloc[:min_train_v], u0)
            mfn = _make_armax_fn(mode, init.order, boxcox_lambda)
            cv = walk_forward_cv(mfn, series, u_exog, min_train=min_train_v)
            return _make_result_dict("ARMAX", init.get_params_str(), mode, exog_cols, 0, cv, boxcox_lambda)
        _try("ARMAX", _armax)

        # ── ARIMAX ───────────────────────────────────────────────────────
        def _arimax():
            init = ARIMAXWrapper(
                d=d, p_max=cfg.SARIMAX_FAMILY_P_MAX, q_max=cfg.SARIMAX_FAMILY_Q_MAX, mode=mode,
                boxcox_lambda=boxcox_lambda,
            )
            init.fit(series.iloc[:min_train_v], u0)
            mfn = _make_arimax_fn(d, mode, init.order, boxcox_lambda)
            cv = walk_forward_cv(mfn, series, u_exog, min_train=min_train_v)
            return _make_result_dict("ARIMAX", init.get_params_str(), mode, exog_cols, d, cv, boxcox_lambda)
        _try("ARIMAX", _arimax)

        # ── OE ───────────────────────────────────────────────────────────
        def _oe():
            init = OEWrapper(d=d, p_max=3, mode=mode, boxcox_lambda=boxcox_lambda)
            init.fit(series.iloc[:min_train_v], u0)
            mfn = _make_oe_fn(d, mode, init.order, boxcox_lambda)
            cv = walk_forward_cv(mfn, series, u_exog, min_train=min_train_v)
            return _make_result_dict("OE", init.get_params_str(), mode, exog_cols, d, cv, boxcox_lambda)
        _try("OE", _oe)

        # ── BJ ───────────────────────────────────────────────────────────
        def _bj():
            init = BJWrapper(d=d, q_max=3, mode=mode, boxcox_lambda=boxcox_lambda)
            init.fit(series.iloc[:min_train_v], u0)
            mfn = _make_bj_fn(d, mode, init.order, boxcox_lambda)
            cv = walk_forward_cv(mfn, series, u_exog, min_train=min_train_v)
            return _make_result_dict("BJ", init.get_params_str(), mode, exog_cols, d, cv, boxcox_lambda)
        _try("BJ", _bj)

    # ── SS ───────────────────────────────────────────────────────────────
    def _ss():
        u_ss = u_exog if has_exog else None
        u0_ss = u_ss.iloc[:min_train_v] if u_ss is not None else None
        init = SSWrapper(mode=mode, boxcox_lambda=boxcox_lambda)
        init.fit(series.iloc[:min_train_v], u0_ss)
        mfn = _make_ss_fn(mode, init.fixed_spec_, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, u_ss, min_train=min_train_v)
        return _make_result_dict(
            "SS", init.get_params_str(), mode, exog_cols if has_exog else [], d, cv, boxcox_lambda
        )
    _try("SS", _ss)

    # ── BAYESARIMA (MAP via prior Minnesota; Litterman 1986; ver protocolo metodológico §22) ──
    def _bayesarima():
        u_b = u_exog if has_exog else None
        u0_b = u_b.iloc[:min_train_v] if u_b is not None else None
        init = BayesArimaWrapper(
            d=d, p_max=cfg.SARIMAX_FAMILY_P_MAX, q_max=cfg.SARIMAX_FAMILY_Q_MAX, mode=mode,
            boxcox_lambda=boxcox_lambda,
        )
        init.fit(series.iloc[:min_train_v], u0_b)
        mfn = _make_bayesarima_fn(d, mode, init.order, init.lambda0, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, u_b, min_train=min_train_v)
        return _make_result_dict(
            "BAYESARIMA", init.get_params_str(), mode, exog_cols if has_exog else [], d, cv, boxcox_lambda
        )
    _try("BAYESARIMA", _bayesarima)

    # ── NARX ─────────────────────────────────────────────────────────────
    def _narx():
        u_n = u_exog if has_exog else None
        mfn = _make_narx_fn(mode, boxcox_lambda)
        cv = walk_forward_cv(mfn, series, u_n, min_train=min_train_v)
        return _make_result_dict(
            "NARX", "NARX(p=3,deg=2)", mode, exog_cols if has_exog else [], d, cv, boxcox_lambda
        )
    _try("NARX", _narx)

    # ── NARIMAX ──────────────────────────────────────────────────────────
    def _narimax():
        nm = "diff" if d >= 1 and mode == "raw" else mode
        nm_lam = boxcox_lambda if nm == mode else None
        u_n = u_exog if has_exog else None
        mfn = _make_narimax_fn(nm, nm_lam)
        cv = walk_forward_cv(mfn, series, u_n, min_train=min_train_v)
        return _make_result_dict(
            "NARIMAX", "NARIMAX(p=3,deg=2)", nm, exog_cols if has_exog else [], d, cv, nm_lam
        )
    _try("NARIMAX", _narimax)

    # ── LSTM (experimental — desabilitado no --round full; use --round lstm se necessario) ──
    # LSTM requer muitos epochs × 44 folds × 12 combos = horas de CPU.
    # Ativar apenas quando explicitamente solicitado.
    if cfg.LSTM_EXPERIMENTAL and False:  # desabilitado por padrão no full
        def _lstm():
            mfn = _make_lstm_fn(mode)
            cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
            return _make_result_dict("LSTM", "LSTM(lb=4,units=8)", mode, [], d, cv)
        _try("LSTM", _lstm)

    # ── ARIMA baseline (recomputa para obter resíduos para DM) ───────────
    arima_residuals = None
    if baseline_res:
        try:
            m = re.match(r"ARIMA\((\d+),\s*(\d+),\s*(\d+)\)",
                         baseline_res.get("descricao_parametros", ""))
            if m:
                bp, bd, bq = int(m[1]), int(m[2]), int(m[3])
            else:
                bp, bd, bq = 1, d, 1
            bl_mode = baseline_res.get("preprocessing", mode)
            bl_lam = baseline_res.get("boxcox_lambda") if bl_mode == "boxcox" else None
            afn = lambda yt, ut: ARIMAWrapper(
                d=bd, mode=bl_mode, fixed_order=(bp, bd, bq), boxcox_lambda=bl_lam
            ).fit(yt, ut)
            acv = walk_forward_cv(afn, series, None, min_train=min_train_v)
            arima_residuals = acv["residuals"]
            if "ARIMA" not in candidates:
                candidates["ARIMA"] = _make_result_dict(
                    "ARIMA", f"ARIMA({bp}, {bd}, {bq})",
                    bl_mode, [], bd, acv, bl_lam,
                )
        except Exception as exc:
            warnings.warn(f"[{ticker}×{ind}] ARIMA baseline recompute: {exc}")

    # ── Seleção do melhor modelo ──────────────────────────────────────────
    if candidates_out is not None:
        candidates_out.update(candidates)
    if not candidates:
        # Fallback total: usar resultado baseline diretamente
        if baseline_res:
            r = baseline_res.copy()
            r.setdefault("dm_vs_arima", {"stat": 0.0, "pvalue": 1.0})
            r.setdefault("in_model_confidence_set", True)
            return (r, None) if return_cv else r
        raise RuntimeError(f"[{ticker}×{ind}] Nenhum modelo executado com sucesso")

    valid = {n: r for n, r in candidates.items() if r["theil_u2"] < 1.0}
    pool  = valid if valid else candidates
    best_name = max(pool, key=lambda n: pool[n]["percentual_fit"])
    best_r    = {k: v for k, v in pool[best_name].items() if not k.startswith("_")}
    best_res  = pool[best_name].get("_residuals")

    # ── DM vs ARIMA ──────────────────────────────────────────────────────
    if arima_residuals is not None and best_name != "ARIMA" and best_res is not None:
        dm_stat, dm_p = diebold_mariano_test(best_res, arima_residuals)
    else:
        dm_stat, dm_p = 0.0, 1.0

    best_r["dm_vs_arima"] = {"stat": float(dm_stat), "pvalue": float(dm_p)}
    best_r["in_model_confidence_set"] = True
    if return_cv:
        winner = pool[best_name]
        cv_out = {
            "y_true":    winner.get("_y_true"),
            "y_pred":    winner.get("_y_pred"),
            "y_rw":      winner.get("_y_rw"),
            "residuals": winner.get("_residuals"),
        }
        return best_r, cv_out
    return best_r


def _append_science_log_round(
    round_num: int,
    results: dict,
    json_path: Path,
    hypothesis: str,
    decisions: str,
    next_round: str,
    baseline_results: dict | None = None,
) -> None:
    """Appenda entrada genérica de rodada ao science_log."""
    ts = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            r = results.get(str(emp_code), {}).get(ind)
            if not r:
                continue
            # Delta FIT% vs rodada anterior (se disponível)
            delta_str = ""
            if baseline_results:
                prev = baseline_results.get(str(emp_code), {}).get(ind)
                if prev:
                    delta = r["percentual_fit"] - prev["percentual_fit"]
                    delta_str = f" ({delta:+.1f})"
            rows.append(
                f"| {ticker}×{_IND_SHORT[ind]:<14} "
                f"| {r['percentual_fit']:>5.1f}{delta_str} "
                f"| {r['theil_u2']:>8.3f} "
                f"| {r['residual_diagnostics']['ljung_box_q4_pvalue']:>4.2f} "
                f"| {r['dm_vs_arima']['pvalue']:>4.2f} "
                f"| {r['modelo_utilizado']:<9} "
                f"| {r['preprocessing']} |"
            )

    # Detectar casos ainda problemáticos (U₂≥1)
    bad_u2 = [
        f"{cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[ind]} (U₂={results[ec][ind]['theil_u2']:.3f})"
        for ec in results
        for ind in cfg.INDICADORES
        if ind in results[ec] and results[ec][ind]["theil_u2"] >= 1.0
    ]
    priority_note = f"**Prioridade:** {'; '.join(bad_u2)}.\n" if bad_u2 else "**Prioridade:** Todos U₂<1.\n"

    col_header = "| Combinação | FIT% (Δ) | Theil U₂ | LB p | DM vs ARIMA p | Modelo | Pré-proc |" if baseline_results \
        else "| Combinação | FIT% | Theil U₂ | LB p | DM vs ARIMA p | Modelo | Pré-proc |"

    log_entry = (
        f"\n## Round {round_num:02d} — {ts}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"{col_header}\n"
        f"|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
        f"**Decisões:** {decisions}\n"
        + priority_note
        + f"**Próximo:** {next_round}\n"
        + f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)


# ---------------------------------------------------------------------------
# Rounds 05-07 — Wiener / Hammerstein / Hammerstein-Wiener
# ---------------------------------------------------------------------------

def _select_nl_order(
    series: "pd.Series",
    wrapper_cls,
    kwargs: dict,
    d: int,
    mode: str,
) -> tuple[int, int, int]:
    """Ajusta wrapper NL na janela inicial e retorna a ordem (p,d,q) do bloco linear."""
    try:
        m = wrapper_cls(**kwargs, d=d, mode=mode).fit(series.iloc[:cfg.MIN_TRAIN])
        if hasattr(m, '_linear') and m._linear is not None:
            return m._linear.order
    except Exception:
        pass
    return (1, d, 0)  # fallback


def _run_nl_round(
    round_num: int,
    model_family: str,   # 'WIENER' | 'HAMMERSTEIN' | 'HW'
    configs: list[dict],
    make_fn,             # _make_wiener_fn | _make_hammerstein_fn | _make_hw_fn
    version: int,
    hypothesis: str,
    next_round: str,
    use_exog: bool = False,
    exog_path: "Path | None" = None,
    combos: "list[tuple[int, str]] | None" = None,
    preproc_modes: "list[str] | None" = None,
    baseline_path: "Path | None" = None,
) -> None:
    """Orquestrador genérico para rounds Wiener/Hammerstein/HW.

    Para cada combo (empresa × indicador) em `combos` (padrão: todas as 12):
      - Testa os modos de pré-processamento em `preproc_modes` (padrão: os 4 modos)
      - Para cada modo: testa todas as configs NL de `configs`
      - Seleciona melhor por FIT% entre candidatos com U₂<1
      - Calcula DM vs ARIMA baseline
      - Salva JSON e appenda ao science_log

    `combos`, `preproc_modes` e `baseline_path` são usados por `run_spec()` para restringir
    um round a um subconjunto de combinações e/ou fixar o modo (sem busca) — mantêm o
    comportamento padrão (todas as combinações, todos os modos, baseline mais recente)
    quando omitidos, preservando `run_wiener_round`/`run_hammerstein_round` inalterados.
    """
    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }
    combo_list = combos if combos is not None else list(cfg.COMBOS)
    modes_list = preproc_modes if preproc_modes is not None else cfg.PREPROC_MODES

    # Exógenas (sub-round ARX)
    exog_sel: dict = {}
    if use_exog and exog_path is not None and exog_path.exists():
        with open(exog_path, encoding="utf-8") as _f:
            exog_sel = json.load(_f)
        print(f"[{model_family}] ARX sub-round com exog de '{exog_path.name}'")

    # Baseline para delta FIT% e DM
    baseline_results: dict = {}
    latest = baseline_path if baseline_path is not None else _find_latest_result()
    if latest:
        with open(latest, encoding="utf-8") as _f:
            baseline_results = json.load(_f)
        print(f"[{model_family}] Baseline: {latest.name}")

    # Imputação de spikes (mesmo protocolo run_full)
    series_clean: dict[str, dict] = {}
    spike_log: list[dict] = []
    for ind in cfg.INDICADORES:
        ds = datasets[ind]
        for emp_code, ticker in cfg.EMPRESAS.items():
            raw_s = ds[str(emp_code)].dropna()
            known = cfg.known_spikes_for(emp_code, ind)
            known_ts = [pd.Timestamp(t) for t in known]
            cleaned, recs = detect_and_impute_spikes(
                raw_s,
                z_thresh=cfg.SPIKE_Z_THRESH,
                method=cfg.SPIKE_IMPUTE_METHOD,
                known_spikes=known_ts,
            )
            series_clean.setdefault(ind, {})[str(emp_code)] = cleaned
            for rec in recs:
                rec.update({"empresa": ticker, "indicador": ind})
                spike_log.append(rec)

    # Semeia `results` com o baseline para combos fora de `combo_list` (rounds focados
    # em um subconjunto ainda precisam produzir um JSON schema-válido com as 12 combinações;
    # ver run_round09 para o mesmo padrão feito manualmente).
    import copy as _copy
    results: dict = _copy.deepcopy(baseline_results) if baseline_results else {}
    for _emp in cfg.EMPRESAS:
        results.setdefault(str(_emp), {})

    col_w = 100
    print("=" * col_w)
    print(f"{model_family} Round {round_num:02d}{'  [ARX]' if use_exog else '  [AR puro]'}")
    print("-" * col_w)

    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        results.setdefault(str(emp_code), {})
        series = series_clean[ind][str(emp_code)]

        # Exógenas para este combo — mesmo formato de run_full(): {emp_code: {ind: [cols]}}
        u_exog: "pd.DataFrame | None" = None
        exog_cols: list[str] = []
        if use_exog:
            all_cols = exog_sel.get(str(emp_code), {}).get(ind, [])
            if all_cols:
                try:
                    u_exog = inputs[all_cols].reindex(series.index).ffill().bfill().fillna(0.0)
                    exog_cols = all_cols
                except Exception as exc:
                    warnings.warn(f"[{ticker}×{ind}] Exog load: {exc}")

        # RW para Theil U₂
        rw_cv = walk_forward_cv(
            lambda yt, ut: RandomWalkModel().fit(yt, ut), series, min_train=cfg.MIN_TRAIN
        )

        # ARIMA baseline para DM test
        arima_residuals = None
        baseline_res = baseline_results.get(str(emp_code), {}).get(ind)
        if baseline_res:
            try:
                m_re = re.match(
                    r"ARIMA\((\d+),\s*(\d+),\s*(\d+)\)",
                    baseline_res.get("descricao_parametros", ""),
                )
                bp, bd, bq = (
                    (int(m_re[1]), int(m_re[2]), int(m_re[3])) if m_re else (1, 1, 0)
                )
                bl_mode = baseline_res.get("preprocessing", "raw")
                afn = lambda yt, ut: ARIMAWrapper(
                    d=bd, mode=bl_mode, fixed_order=(bp, bd, bq)
                ).fit(yt, ut)
                acv = walk_forward_cv(afn, series, None, min_train=cfg.MIN_TRAIN)
                arima_residuals = acv["residuals"]
            except Exception as exc:
                warnings.warn(f"[{ticker}×{ind}] ARIMA baseline: {exc}")

        best_fit   = -np.inf
        best_entry: dict | None = None
        best_res_arr = None

        for mode in modes_list:
            try:
                y_t = transform(series, mode)
                urt = unit_root_tests(y_t)
                d   = urt["consensus_d"]
            except Exception:
                continue

            for nl_cfg in configs:
                try:
                    # Seleciona ordem na janela inicial (congela para walk-forward)
                    if model_family == "WIENER":
                        kw = {"output_nl": nl_cfg["output_nl"],
                              "n_knots": nl_cfg.get("n_knots")}
                        fo = _select_nl_order(series, WienerWrapper, kw, d, mode)
                        mfn = _make_wiener_fn(
                            nl_cfg["output_nl"], nl_cfg.get("n_knots"), d, mode, fo
                        )
                        model_name = f"WIENER-{nl_cfg['name']}"
                    elif model_family == "HAMMERSTEIN":
                        kw = {"input_nl": nl_cfg["input_nl"],
                              "n_knots": nl_cfg.get("n_knots")}
                        fo = _select_nl_order(series, HammersteinWrapper, kw, d, mode)
                        mfn = _make_hammerstein_fn(
                            nl_cfg["input_nl"], nl_cfg.get("n_knots"), d, mode, fo
                        )
                        model_name = f"HAMMER-{nl_cfg['name']}"
                    else:  # HW
                        kw = {"input_nl": nl_cfg["input_nl"],
                              "output_nl": nl_cfg["output_nl"],
                              "n_knots_in": nl_cfg.get("n_knots_in"),
                              "n_knots_out": nl_cfg.get("n_knots_out")}
                        fo = _select_nl_order(series, HWWrapper, kw, d, mode)
                        mfn = _make_hw_fn(
                            nl_cfg["input_nl"], nl_cfg["output_nl"],
                            nl_cfg.get("n_knots_in"), nl_cfg.get("n_knots_out"),
                            d, mode, fo
                        )
                        model_name = f"HW-{nl_cfg['name']}"

                    cv = walk_forward_cv(
                        mfn, series, u_exog, min_train=cfg.MIN_TRAIN
                    )
                    entry = _make_result_dict(
                        model_name, f"{model_family}({nl_cfg['name']},{mode})",
                        mode, exog_cols, d, cv
                    )
                    if cv["fit_percent"] > best_fit and cv["theil_u2"] < 1.0:
                        best_fit   = cv["fit_percent"]
                        best_entry = entry
                        best_res_arr = cv["residuals"]
                    elif best_entry is None:  # guardar mesmo com U₂≥1 como fallback
                        best_fit   = cv["fit_percent"]
                        best_entry = entry
                        best_res_arr = cv["residuals"]
                except Exception as exc:
                    warnings.warn(f"[{ticker}×{ind}] {model_family}/{nl_cfg['name']}/{mode}: {exc}")

        if best_entry is None:
            # Fallback: usar resultado baseline
            if baseline_res:
                best_entry = {k: v for k, v in baseline_res.items() if not k.startswith("_")}
                best_entry.setdefault("dm_vs_arima", {"stat": 0.0, "pvalue": 1.0})
            else:
                warnings.warn(f"[{ticker}×{ind}] Nenhum modelo NL executou; combo ignorado.")
                continue

        # DM vs ARIMA
        dm_stat, dm_p = 0.0, 1.0
        if arima_residuals is not None and best_res_arr is not None:
            try:
                dm_stat, dm_p = diebold_mariano_test(best_res_arr, arima_residuals)
            except Exception:
                pass

        best_r = {k: v for k, v in best_entry.items() if not k.startswith("_")}
        best_r["dm_vs_arima"] = {"stat": float(dm_stat), "pvalue": float(dm_p)}
        best_r["in_model_confidence_set"] = True
        results[str(emp_code)][ind] = best_r

        print(
            f"{ticker:<8} {ind:<22} {best_r['preprocessing']:<8} "
            f"{best_r['percentual_fit']:>6.1f} {best_r['rmse']:>12.2e} "
            f"{best_r['theil_u2']:>6.3f} "
            f"{best_r['residual_diagnostics']['ljung_box_q4_pvalue']:>5.3f} "
            f"{best_r['modelo_utilizado']}"
        )

    # Salvar e appendar science_log
    json_path = save_results(results, version=version)

    arx_note = " + exógenas (ARX sub-round)" if use_exog else " — AR puro (sem exógenas)"
    _append_science_log_round(
        round_num=round_num,
        results=results,
        json_path=json_path,
        hypothesis=hypothesis + arx_note,
        decisions=(
            f"Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA baseline. "
            f"ALS max_iter={cfg.ALS_MAX_ITER}, tol={cfg.ALS_TOL}; "
            f"blocos NL testados: {', '.join(c['name'] for c in configs)}."
        ),
        next_round=next_round,
        baseline_results=baseline_results if baseline_results else None,
    )
    print(f"\nScience log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


def _fresh_arima_baseline(series: "pd.Series", min_train_v: int) -> "np.ndarray | None":
    """Busca (mode, ordem) do ARIMA por FIT%/gate U₂<1, ajustado sob `min_train_v`.

    Usado no lugar de reaproveitar um baseline_json antigo quando a rodada roda sob uma
    janela de treino diferente de cfg.MIN_TRAIN — residuals de tamanhos/janelas diferentes
    não podem alimentar o mesmo teste DM (ver protocolo metodológico §24, experimento isolado).
    """
    best = _best_arima_baseline(series, min_train_v)
    return best["cv"]["residuals"] if best else None


def _best_arima_baseline(series: "pd.Series", min_train_v: int) -> "dict | None":
    """Núcleo de `_fresh_arima_baseline`: devolve também (mode, d, ordem) do ARIMA escolhido,
    para que o Round 35 rode o mesmo baseline sob estimação recursiva."""
    best, best_fit, best_u2 = None, -np.inf, np.inf
    for mode in cfg.PREPROC_MODES:
        try:
            y_t = transform(series, mode)
            d = unit_root_tests(y_t)["consensus_d"]
            init = ARIMAWrapper(d=d, mode=mode).fit(series.iloc[:min_train_v])
            afn = _make_arima_fn(d, mode, init.order)
            cv = walk_forward_cv(afn, series, None, min_train=min_train_v)
        except Exception:
            continue
        better_gate = cv["theil_u2"] < 1.0 and best_u2 >= 1.0
        better_fit = cv["theil_u2"] < 1.0 and best_u2 < 1.0 and cv["fit_percent"] > best_fit
        if best is None or better_gate or better_fit:
            best = {"mode": mode, "d": d, "order": init.order, "fn": afn, "cv": cv}
            best_fit, best_u2 = cv["fit_percent"], cv["theil_u2"]
    return best


def _run_setar_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combos: "list[tuple[int, str]] | None" = None,
    preproc_modes: "list[str] | None" = None,
    baseline_path: "Path | None" = None,
    min_train: "int | None" = None,
    output_dir: "Path | None" = None,
    comparable: bool = True,
) -> None:
    """Orquestrador do round SETAR (Round 22 — ver protocolo metodológico §24).

    Estrutura própria (não reusa `_run_nl_round`): SETAR não compõe blocos NL via ALS —
    delay/threshold/ordem são buscados dentro do próprio `SETARWrapper.fit()` (grid por SSR),
    não por `_select_nl_order`. O baseline ARIMA para o teste DM é sempre recalculado do zero
    sob o `min_train` efetivo desta rodada (`_fresh_arima_baseline`) em vez de reaproveitar um
    baseline_json antigo — necessário sempre que `min_train` difere de `cfg.MIN_TRAIN`.

    `min_train`/`output_dir`/`comparable`: suportam o "experimental_window" opcional da spec
    (rounds/specs/) — quando presentes, o resultado é salvo fora de `outputs/results/` (não
    aparece no glob não-recursivo de `scripts/_common.py`, logo nunca é promovido a campeão
    por engano) e a rodada é marcada como não-comparável ao painel vigente.
    """
    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }
    combo_list = combos if combos is not None else list(cfg.COMBOS)
    modes_list = preproc_modes if preproc_modes is not None else cfg.PREPROC_MODES
    min_train_v = min_train if min_train is not None else cfg.MIN_TRAIN

    baseline_results: dict = {}
    latest = baseline_path if baseline_path is not None else _find_latest_result()
    if latest:
        with open(latest, encoding="utf-8") as f:
            baseline_results = json.load(f)
        print(f"[SETAR] Baseline (referência apenas): {latest.name}")

    import copy as _copy
    results: dict = _copy.deepcopy(baseline_results) if baseline_results else {}
    for emp in cfg.EMPRESAS:
        results.setdefault(str(emp), {})

    spike_log: list[dict] = []
    col_w = 100
    print("=" * col_w)
    print(f"SETAR Round {round_num:02d}  min_train={min_train_v}"
          f"{'  [NÃO-COMPARÁVEL AO PAINEL]' if not comparable else ''}")
    print("-" * col_w)

    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        results.setdefault(str(emp_code), {})
        raw_s = datasets[ind][str(emp_code)].dropna()
        known_ts = [pd.Timestamp(t) for t in cfg.known_spikes_for(emp_code, ind)]
        series, recs = detect_and_impute_spikes(
            raw_s, z_thresh=cfg.SPIKE_Z_THRESH, method=cfg.SPIKE_IMPUTE_METHOD,
            known_spikes=known_ts,
        )
        for rec in recs:
            spike_log.append({"empresa": ticker, "indicador": ind, **rec})

        arima_residuals = _fresh_arima_baseline(series, min_train_v)

        best_fit = -np.inf
        best_entry: dict | None = None
        best_res_arr = None

        for mode in modes_list:
            try:
                y_t = transform(series, mode)
                d = unit_root_tests(y_t)["consensus_d"]
                init = SETARWrapper(
                    p_max=cfg.SETAR_P_MAX, delay_max=cfg.SETAR_DELAY_MAX,
                    trim=cfg.SETAR_TRIM, mode=mode,
                ).fit(series.iloc[:min_train_v])
                mfn = _make_setar_fn(mode, init.fixed_config_)
                cv = walk_forward_cv(mfn, series, None, min_train=min_train_v)
                entry = _make_result_dict("SETAR", init.get_params_str(), mode, [], d, cv)
                if cv["fit_percent"] > best_fit and cv["theil_u2"] < 1.0:
                    best_fit, best_entry, best_res_arr = cv["fit_percent"], entry, cv["residuals"]
                elif best_entry is None:
                    best_fit, best_entry, best_res_arr = cv["fit_percent"], entry, cv["residuals"]
            except Exception as exc:
                warnings.warn(f"[{ticker}×{ind}] SETAR/{mode}: {exc}")

        if best_entry is None:
            warnings.warn(f"[{ticker}×{ind}] SETAR não executou em nenhum modo; combo ignorado.")
            continue

        dm_stat, dm_p = 0.0, 1.0
        if arima_residuals is not None and best_res_arr is not None:
            try:
                dm_stat, dm_p = diebold_mariano_test(best_res_arr, arima_residuals)
            except Exception:
                pass

        best_r = {k: v for k, v in best_entry.items() if not k.startswith("_")}
        best_r["dm_vs_arima"] = {"stat": float(dm_stat), "pvalue": float(dm_p)}
        best_r["in_model_confidence_set"] = True
        results[str(emp_code)][ind] = best_r

        print(
            f"{ticker:<8} {ind:<22} {best_r['preprocessing']:<8} "
            f"{best_r['percentual_fit']:>6.1f} {best_r['theil_u2']:>6.3f} "
            f"dm_p={dm_p:.3f}  {best_r['modelo_utilizado']} {best_r['descricao_parametros']}"
        )

    save_dir = output_dir if output_dir is not None else None
    json_path = save_results(results, version=version, directory=save_dir)

    spike_summary = ""
    if spike_log:
        lines = ["**Imputações de spike (linear_interp):**"]
        for rec in spike_log:
            lines.append(
                f"  - {rec['empresa']}×{_IND_SHORT[rec['indicador']]}: "
                f"{rec['timestamp'].strftime('%Y-%m-%d')} "
                f"{rec['original']/1e9:.3f}B → {rec['imputed']/1e9:.3f}B (z={rec['z_score']})"
            )
        spike_summary = "\n".join(lines) + "\n"

    comparability_note = (
        "" if comparable else
        f"**NÃO-COMPARÁVEL ao painel vigente** — min_train={min_train_v} "
        f"(painel usa cfg.MIN_TRAIN={cfg.MIN_TRAIN}), baseline ARIMA recalculado do zero sob "
        f"esta janela (`_fresh_arima_baseline`), resultado salvo em '{json_path}' "
        f"(fora de outputs/results/, não entra em compute_champions()). "
    )
    _append_science_log_round(
        round_num=round_num,
        results=results,
        json_path=json_path,
        hypothesis=hypothesis,
        decisions=(
            comparability_note + spike_summary
            + "Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA "
              "(recalculado sob a mesma janela)."
        ),
        next_round=next_round,
        baseline_results=None,  # Δ FIT% vs baseline seria enganoso sob janela diferente
    )
    print(f"\nJSON salvo: {json_path}")
    print(f"Science log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


def _run_window_sweep_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    window_values: "list[int]",
    eval_start: int,
    combo_list: "list[tuple[int, str]]",
    use_exog: bool,
    exog_path: "Path | None",
    fixed_mode: "str | None",
) -> None:
    """Round 24 — learning curve de MIN_TRAIN com avaliação OOS fixa (protocolo metodológico §26).

    Reexecuta a busca completa de modelo (`_run_all_models_for_combo`, mesma lógica de
    `models: full`) sob cada valor de `window_values`, mas mede FIT%/U₂/DM-p só nos folds
    comuns a todas as janelas (`eval_start` até o fim da série) — isola o efeito do tamanho
    de treino do efeito de "quantos folds entraram na média" (Round 22, §24, já documentou
    essa armadilha para o caso de um único min_train isolado; este round generaliza para uma
    curva sobre vários valores).

    Diagnóstico apenas — nunca escreve em `outputs/results/` nem entra em `compute_champions()`.
    Promoção de um MIN_TRAIN diferente de `cfg.MIN_TRAIN` exige round dedicado, não-exploratório,
    sob `experimental_window` (ou uma mudança de `cfg.MIN_TRAIN` em si, com round de todo o
    painel), passando pelo mesmo gate U₂<1 e DM-p<0.10 de qualquer outro candidato.
    """
    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    if eval_start <= max(window_values):
        raise ValueError(
            f"window_sweep: eval_start={eval_start} deve ser > max(values)={max(window_values)} "
            "para garantir pelo menos 1 fold OOS na janela mais larga."
        )

    exog_sel: dict = {}
    if use_exog and exog_path is not None and exog_path.exists():
        with open(exog_path, encoding="utf-8") as f:
            exog_sel = json.load(f)

    sweep: dict = {str(w): {} for w in window_values}
    modes_to_try = [fixed_mode] if fixed_mode else cfg.PREPROC_MODES

    print(f"[window_sweep] MIN_TRAIN testados: {window_values}  eval_start={eval_start}")
    print(f"{'Empresa':<8} {'Indicador':<22} {'win':>4} {'Modelo':<14} "
          f"{'FIT%_eval':>10} {'U2_eval':>8} {'dm_p_eval':>10}")
    print("-" * 90)

    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        raw_s = datasets[ind][str(emp_code)].dropna()
        known_ts = [pd.Timestamp(t) for t in cfg.known_spikes_for(emp_code, ind)]
        series, _recs = detect_and_impute_spikes(
            raw_s, z_thresh=cfg.SPIKE_Z_THRESH, method=cfg.SPIKE_IMPUTE_METHOD,
            known_spikes=known_ts,
        )
        sel_cols = exog_sel.get(str(emp_code), {}).get(ind, []) if use_exog else []
        u_exog = inputs[sel_cols].reindex(series.index).ffill().bfill() if sel_cols else None

        for win in window_values:
            arima_res_full = _fresh_arima_baseline(series, win)

            best_r, best_cv = None, None
            for mode in modes_to_try:
                try:
                    lam = guerrero_lambda(series) if mode == "boxcox" else None
                    y_t = transform(series, mode, lam)
                    d = unit_root_tests(y_t)["consensus_d"]
                    r, cv = _run_all_models_for_combo(
                        series, u_exog, mode, d, sel_cols, ticker, ind, None,
                        boxcox_lambda=lam, min_train_v=win, return_cv=True,
                    )
                except Exception as exc:
                    warnings.warn(f"[{ticker}×{ind}] win={win} mode={mode}: {exc}")
                    continue
                if best_r is None or (
                    r["theil_u2"] < 1.0 and (best_r["theil_u2"] >= 1.0 or r["percentual_fit"] > best_r["percentual_fit"])
                ):
                    best_r, best_cv = r, cv

            if best_r is None or best_cv is None or best_cv.get("y_true") is None:
                warnings.warn(f"[{ticker}×{ind}] win={win}: nenhum modo produziu resultado; ponto ignorado.")
                continue

            eval_metrics = restrict_to_eval_range(best_cv, min_train=win, eval_start=eval_start)

            dm_p_eval = 1.0
            if arima_res_full is not None:
                offset = eval_start - win
                arima_res_eval = np.asarray(arima_res_full)[offset:]
                model_res_eval = eval_metrics["residuals"]
                try:
                    _, dm_p_eval = diebold_mariano_test(model_res_eval, arima_res_eval)
                except Exception:
                    pass

            sweep[str(win)].setdefault(str(emp_code), {})[ind] = {
                "modelo_utilizado":      best_r["modelo_utilizado"],
                "descricao_parametros":  best_r["descricao_parametros"],
                "preprocessing":         best_r["preprocessing"],
                "fit_percent_full_range": float(best_r["percentual_fit"]),
                "theil_u2_full_range":    float(best_r["theil_u2"]),
                "fit_percent_eval":       float(eval_metrics["fit_percent"]),
                "theil_u2_eval":          float(eval_metrics["theil_u2"]),
                "rmse_eval":              float(eval_metrics["rmse"]),
                "n_eval_folds":           int(eval_metrics["n_folds"]),
                "dm_p_eval":              float(dm_p_eval),
            }
            print(f"{ticker:<8} {ind:<22} {win:>4} {best_r['modelo_utilizado']:<14} "
                  f"{eval_metrics['fit_percent']:>10.1f} {eval_metrics['theil_u2']:>8.3f} "
                  f"{dm_p_eval:>10.3f}")

    out_dir = paths.RESULTS_EXPERIMENTAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_path = out_dir / f"{ts}_round{round_num}_window_sweep_v{version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "round": round_num,
                "window_values": window_values,
                "eval_start": eval_start,
                "comparable": False,
                "sweep": sweep,
            },
            f, ensure_ascii=False, indent=2, default=str,
        )

    header_cols = " | ".join(f"win={w}" for w in window_values)
    rows = [f"| Combinação | {header_cols} |", "|---" + "|---" * len(window_values) + "|"]
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            cells = []
            for w in window_values:
                cell = sweep[str(w)].get(str(emp_code), {}).get(ind)
                cells.append(f"{cell['fit_percent_eval']:.1f}" if cell else "—")
            rows.append(f"| {ticker}×{_IND_SHORT[ind]} | " + " | ".join(cells) + " |")

    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Learning curve — FIT%_eval (folds t={eval_start}..63, comum a todas as janelas):**\n"
        + "\n".join(rows) + "\n"
        f"**Decisões:** window_sweep diagnóstico, NÃO-COMPARÁVEL ao painel vigente "
        f"(comparable=false, MIN_TRAIN testados={window_values}, eval_start={eval_start}). "
        f"Busca completa de modelo (`_run_all_models_for_combo`) refeita a cada janela — "
        f"não fixa o mesmo modelo campeão entre janelas. Resultado em '{json_path}', "
        f"fora de outputs/results/, não entra em compute_champions(). Não promove campeão "
        f"automaticamente — ver protocolo metodológico §26, condição de promoção.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")
    print(f"Science log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


def _run_recursive_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combo_list: "list[tuple[int, str]]",
    forgetting: "float | None" = None,
    frozen_tau_batch: bool = False,
    resets: bool = False,
) -> None:
    """Round 35+ — campeão de cada combinação sob estimação recursiva (RPEM; protocolo metodológico §28).

    `forgetting` sobrescreve `cfg.RPEM_FORGETTING` nesta rodada (bloco `recursive:` da spec).
    `frozen_tau_batch` (Round 36) roda também o lote com o breakpoint piecewise congelado no
    valor da janela inicial, nos campeões Wiener/Hammerstein — separa o efeito da recursão do
    efeito de τ deixar de ser reescolhido a cada dobra.
    `resets` (Round 37) reinicializa R nos eventos de `cfg.RPEM_RESET_EVENTS` (campeão e ARIMA) e
    roda também o recursivo sem reinicialização, para separar o efeito do evento do de λ.

    Reconstrói o campeão com as factories de `plot_best_results.py` (as mesmas do gate de
    reprodução, Invariante 10) e roda duas vezes: em lote (`walk_forward_cv`, o protocolo
    vigente) e recursivo (`src.recursive.recursive_walk_forward`). O ARIMA de referência
    (`_best_arima_baseline`) passa pelo mesmo par. Exploratório — grava em
    `outputs/results/experimental/`, fora de `compute_champions()`.
    """
    import plot_best_results as pbr
    from src.recursive import recursive_walk_forward

    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    champions = pbr.compute_champions()
    mt = cfg.MIN_TRAIN
    lam = cfg.RPEM_FORGETTING if forgetting is None else float(forgetting)

    def _dm(e1, e2) -> dict:
        stat, p = diebold_mariano_test(np.asarray(e1), np.asarray(e2))
        return {"stat": float(stat), "pvalue": float(p)}

    out: dict = {}
    print(f"[recursive] RPEM, lambda={lam}")
    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        champ = champions[(emp_code, ind)]
        rec = pbr.load_json(pbr.RESULTS_DIR / champ["file"])[str(emp_code)][ind]
        series = pbr._prepare_series(emp_code, ind, datasets)
        u_exog = pbr._build_exog(series, rec.get("exogenas_utilizadas", []), inputs)
        with pbr.round_hyperparams(champ["file"]):
            fn = pbr._make_model_fn(rec, series, u_exog)
            cv_b = walk_forward_cv(fn, series, u_exog, min_train=mt)
            ev = [d for d, _ in cfg.RPEM_RESET_EVENTS.get(emp_code, [])] if resets else []
            cv_r = recursive_walk_forward(fn, series, u_exog, min_train=mt, lam=lam,
                                          reset_dates=ev)
            cv_rn = (recursive_walk_forward(fn, series, u_exog, min_train=mt, lam=lam)
                     if resets else None)
            cv_bt, tau0 = None, None
            if frozen_tau_batch:
                w0 = fn(series.iloc[:mt], u_exog.iloc[:mt] if u_exog is not None else None)
                if isinstance(w0, (WienerWrapper, HammersteinWrapper)):
                    tau0 = float((w0._g if isinstance(w0, WienerWrapper) else w0._f)._tau)
                    fn_tau = pbr._make_model_fn(rec, series, u_exog, nl_fixed_tau=tau0)
                    cv_bt = walk_forward_cv(fn_tau, series, u_exog, min_train=mt)
        arima = _best_arima_baseline(series, mt)
        ar_b = arima["cv"]
        ar_r = recursive_walk_forward(arima["fn"], series, None, min_train=mt, lam=lam,
                                      reset_dates=ev)

        entry = {
            "modelo_utilizado": rec["modelo_utilizado"],
            "descricao_parametros": rec["descricao_parametros"],
            "preprocessing": rec["preprocessing"],
            "exogenas_utilizadas": rec.get("exogenas_utilizadas", []),
            "source_json": champ["file"],
            "fit_stored": float(rec["percentual_fit"]),
            "batch": {"fit": cv_b["fit_percent"], "u2": cv_b["theil_u2"], "rmse": cv_b["rmse"]},
            "recursive": {"fit": cv_r["fit_percent"], "u2": cv_r["theil_u2"], "rmse": cv_r["rmse"],
                          "n_updates": cv_r["n_updates"], "n_rejected": cv_r["n_rejected"],
                          "n_shrunk": cv_r["n_shrunk"]},
            "delta_fit_pp": cv_r["fit_percent"] - cv_b["fit_percent"],
            "first_pred_abs_diff": float(abs(cv_r["y_pred"][0] - cv_b["y_pred"][0])),
            "dm_recursive_vs_batch": _dm(cv_r["residuals"], cv_b["residuals"]),
            "arima": {
                "mode": arima["mode"], "order": list(arima["order"]),
                "batch": {"fit": ar_b["fit_percent"], "u2": ar_b["theil_u2"]},
                "recursive": {"fit": ar_r["fit_percent"], "u2": ar_r["theil_u2"],
                              "n_rejected": ar_r["n_rejected"]},
            },
            "dm_batch_vs_arima_batch": _dm(cv_b["residuals"], ar_b["residuals"]),
            "dm_recursive_vs_arima_recursive": _dm(cv_r["residuals"], ar_r["residuals"]),
            "y_true": cv_b["y_true"].tolist(),
            "y_pred_batch": cv_b["y_pred"].tolist(),
            "y_pred_recursive": cv_r["y_pred"].tolist(),
        }
        if cv_rn is not None:
            entry["reset_events"] = cfg.RPEM_RESET_EVENTS.get(emp_code, [])
            entry["recursive"]["n_resets"] = cv_r["n_resets"]
            entry["recursive_no_reset"] = {
                "fit": cv_rn["fit_percent"], "u2": cv_rn["theil_u2"], "rmse": cv_rn["rmse"],
                "dm_vs_batch": _dm(cv_rn["residuals"], cv_b["residuals"]),
                "dm_reset_vs_no_reset": _dm(cv_r["residuals"], cv_rn["residuals"]),
                "y_pred": cv_rn["y_pred"].tolist(),
            }
        if cv_bt is not None:
            entry["batch_frozen_tau"] = {
                "tau": tau0, "fit": cv_bt["fit_percent"], "u2": cv_bt["theil_u2"],
                "rmse": cv_bt["rmse"],
                "dm_vs_batch": _dm(cv_bt["residuals"], cv_b["residuals"]),
                "dm_recursive_vs_this": _dm(cv_r["residuals"], cv_bt["residuals"]),
                "y_pred": cv_bt["y_pred"].tolist(),
            }
        out.setdefault(str(emp_code), {})[ind] = entry
        print(f"{ticker:<6} {ind:<22} {rec['modelo_utilizado']:<22} "
              f"lote {cv_b['fit_percent']:6.1f}/{cv_b['theil_u2']:.3f}  "
              f"rec {cv_r['fit_percent']:6.1f}/{cv_r['theil_u2']:.3f}  "
              f"d{entry['delta_fit_pp']:+6.1f}  DM-p {entry['dm_recursive_vs_batch']['pvalue']:.3f}  "
              f"rej {cv_r['n_rejected']}/{cv_r['n_updates']}", flush=True)

    out_dir = paths.RESULTS_EXPERIMENTAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_path = out_dir / f"{ts}_round{round_num}_recursive_v{version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"round": round_num, "estimator": "RPEM",
                   "forgetting": lam, "resets": resets,
                   "reset_rho": cfg.RPEM_RESET_RHO if resets else None,
                   "min_train": mt, "combos": out},
                  f, ensure_ascii=False, indent=2, default=str)

    rows = ["| Combinação | Modelo | FIT% lote | FIT% rec | Δ p.p. | U₂ lote | U₂ rec | "
            "DM-p rec×lote | ARIMA FIT% lote→rec | rejeições |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            e = out.get(str(emp_code), {}).get(ind)
            if e is None:
                continue
            rows.append(
                f"| {ticker}×{_IND_SHORT[ind]} | {e['modelo_utilizado']} | {e['batch']['fit']:.1f} | "
                f"{e['recursive']['fit']:.1f} | {e['delta_fit_pp']:+.1f} | {e['batch']['u2']:.3f} | "
                f"{e['recursive']['u2']:.3f} | {e['dm_recursive_vs_batch']['pvalue']:.3f} | "
                f"{e['arima']['batch']['fit']:.1f}→{e['arima']['recursive']['fit']:.1f} | "
                f"{e['recursive']['n_rejected']}/{e['recursive']['n_updates']} |")

    tau_rows = [
        f"| {cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[i]} | {e['modelo_utilizado']} | "
        f"{e['batch_frozen_tau']['tau']:.4g} | {e['batch']['fit']:.1f} | "
        f"{e['batch_frozen_tau']['fit']:.1f} | {e['recursive']['fit']:.1f} | "
        f"{e['batch_frozen_tau']['dm_vs_batch']['pvalue']:.3f} | "
        f"{e['batch_frozen_tau']['dm_recursive_vs_this']['pvalue']:.3f} |"
        for ec, inds in out.items() for i, e in inds.items() if "batch_frozen_tau" in e
    ]
    reset_rows = [
        f"| {cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[i]} | {e['batch']['fit']:.1f} | "
        f"{e['recursive_no_reset']['fit']:.1f} | {e['recursive']['fit']:.1f} | "
        f"{e['recursive']['n_resets']} | "
        f"{e['recursive_no_reset']['dm_vs_batch']['pvalue']:.3f} | "
        f"{e['recursive_no_reset']['dm_reset_vs_no_reset']['pvalue']:.3f} |"
        for ec, inds in out.items() for i, e in inds.items() if "recursive_no_reset" in e
    ]
    if reset_rows:
        rows += [
            "",
            f"**Separação λ × reinicialização (ρ={cfg.RPEM_RESET_RHO}):**",
            "| Combinação | FIT% lote | FIT% rec só λ | FIT% rec λ+reset | resets | "
            "DM-p só λ×lote | DM-p reset×só λ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ] + reset_rows
    tau_block = ""
    if tau_rows:
        tau_block = (
            "**Lote com τ congelado (block-oriented piecewise):**\n"
            "| Combinação | Modelo | τ₀ | FIT% lote | FIT% lote τ fixo | FIT% rec | "
            "DM-p τfixo×lote | DM-p rec×τfixo |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(tau_rows) + "\n")
    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Campeão de cada combinação, estimação em lote vs recursiva (RPEM, λ={lam}):**\n"
        + "\n".join(rows) + "\n" + tau_block +
        f"**Decisões:** round exploratório (protocolo metodológico §28). Mesma família, ordem, "
        f"pré-processamento, exógenas e pesos do campeão; só o estimador muda. Resultado em "
        f"'{json_path.name}', em outputs/results/experimental/, fora de compute_champions(); "
        f"os campeões em lote seguem oficiais.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")


def _lag_exog(u_exog: "pd.DataFrame | None", lag: int) -> "pd.DataFrame | None":
    """Desloca `lag` trimestres as exógenas macro/mercado (x_t -> x_{t-lag}); LS_* não mexem.

    As linhas iniciais que ficam sem valor são preenchidas com `bfill`, o mesmo tratamento do
    alinhamento original; caem dentro de MIN_TRAIN e não entram em nenhuma previsão.
    """
    if u_exog is None or lag == 0:
        return u_exog
    macro = [c for c in u_exog.columns if not re.fullmatch(r"LS_\d{8}", c)]
    shifted = u_exog.copy()
    shifted[macro] = u_exog[macro].shift(lag).bfill()
    return shifted


def _holm_adjust(pvalues: "list[float]") -> "list[float]":
    """p ajustados de Holm (step-down), na ordem original."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def _run_lagged_exog_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combo_list: "list[tuple[int, str]]",
    lag: "int | None" = None,
) -> None:
    """Round 38 — campeões com exógena sob x_{t-L} em vez de x_t (protocolo metodológico §29).

    Reconstrói o campeão com as factories de `plot_best_results.py` (as mesmas do gate de
    reprodução, Invariante 10) e roda o lote duas vezes: com as exógenas contemporâneas (deve
    reproduzir o FIT% armazenado) e deslocadas `lag` trimestres. Família, ordem, pré-processamento
    e conjunto exógeno ficam congelados no campeão contemporâneo, então o resultado é um piso do
    que uma busca refeita sobre x_{t-L} atingiria. Exploratório — grava em
    `outputs/results/experimental/`, fora de `compute_champions()`.
    """
    import plot_best_results as pbr

    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    champions = pbr.compute_champions()
    mt = cfg.MIN_TRAIN
    lag = cfg.EXOG_LAG_QUARTERS if lag is None else int(lag)

    def _dm(e1, e2) -> dict:
        stat, p = diebold_mariano_test(np.asarray(e1), np.asarray(e2))
        return {"stat": float(stat), "pvalue": float(p)}

    out: dict = {}
    print(f"[lagged-exog] x(t-{lag}), campeão fixo")
    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        champ = champions[(emp_code, ind)]
        rec = pbr.load_json(pbr.RESULTS_DIR / champ["file"])[str(emp_code)][ind]
        cols = rec.get("exogenas_utilizadas", [])
        if not cols:
            raise ValueError(f"Round {round_num}: {ticker}×{ind} não tem exógena no campeão")
        series = pbr._prepare_series(emp_code, ind, datasets)
        u_c = pbr._build_exog(series, cols, inputs)
        u_l = _lag_exog(u_c, lag)
        with pbr.round_hyperparams(champ["file"]):
            cv_c = walk_forward_cv(pbr._make_model_fn(rec, series, u_c), series, u_c, min_train=mt)
            cv_l = walk_forward_cv(pbr._make_model_fn(rec, series, u_l), series, u_l, min_train=mt)
        arima = _best_arima_baseline(series, mt)
        ar = arima["cv"]

        rho1 = float(series.autocorr(1))
        bench = 100.0 * (1.0 - float(np.sqrt(max(0.0, 1.0 - rho1 ** 2))))
        fit_stored = float(rec["percentual_fit"])
        if abs(cv_c["fit_percent"] - fit_stored) > 0.05:
            print(f"    [AVISO] {ticker}×{ind}: lote contemporâneo {cv_c['fit_percent']:.2f} "
                  f"!= FIT armazenado {fit_stored:.2f}")
        out.setdefault(str(emp_code), {})[ind] = {
            "modelo_utilizado": rec["modelo_utilizado"],
            "descricao_parametros": rec["descricao_parametros"],
            "preprocessing": rec["preprocessing"],
            "exogenas_utilizadas": cols,
            "source_json": champ["file"],
            "fit_stored": fit_stored,
            "contemporaneous": {"fit": cv_c["fit_percent"], "u2": cv_c["theil_u2"],
                                "rmse": cv_c["rmse"]},
            "lagged": {"fit": cv_l["fit_percent"], "u2": cv_l["theil_u2"], "rmse": cv_l["rmse"]},
            "delta_fit_pp": cv_l["fit_percent"] - cv_c["fit_percent"],
            "arima": {"mode": arima["mode"], "order": list(arima["order"]),
                      "fit": ar["fit_percent"], "u2": ar["theil_u2"]},
            "rho1": rho1, "fit_ar1_benchmark": bench,
            "excess_contemporaneous": cv_c["fit_percent"] - bench,
            "excess_lagged": cv_l["fit_percent"] - bench,
            "dm_lagged_vs_contemporaneous": _dm(cv_l["residuals"], cv_c["residuals"]),
            "dm_lagged_vs_arima": _dm(cv_l["residuals"], ar["residuals"]),
            "dm_contemporaneous_vs_arima": _dm(cv_c["residuals"], ar["residuals"]),
            "y_true": cv_c["y_true"].tolist(),
            "y_pred_contemporaneous": cv_c["y_pred"].tolist(),
            "y_pred_lagged": cv_l["y_pred"].tolist(),
        }
        e = out[str(emp_code)][ind]
        print(f"{ticker:<6} {ind:<22} {rec['modelo_utilizado']:<12} "
              f"cont {cv_c['fit_percent']:6.1f}/{cv_c['theil_u2']:.3f}  "
              f"lag {cv_l['fit_percent']:6.1f}/{cv_l['theil_u2']:.3f}  "
              f"d{e['delta_fit_pp']:+6.1f}  DM-p(lag vs arima) "
              f"{e['dm_lagged_vs_arima']['pvalue']:.3f}", flush=True)

    entries = [(ec, i, e) for ec, inds in out.items() for i, e in inds.items()]
    holm_arima = _holm_adjust([e["dm_lagged_vs_arima"]["pvalue"] for _, _, e in entries])
    holm_cont = _holm_adjust([e["dm_lagged_vs_contemporaneous"]["pvalue"] for _, _, e in entries])
    for (_, _, e), pa, pc in zip(entries, holm_arima, holm_cont):
        e["dm_lagged_vs_arima"]["pvalue_holm"] = pa
        e["dm_lagged_vs_contemporaneous"]["pvalue_holm"] = pc
    n = len(entries)
    summary = {
        "n_combos": n,
        "mean_excess_contemporaneous": float(np.mean([e["excess_contemporaneous"] for *_, e in entries])),
        "mean_excess_lagged": float(np.mean([e["excess_lagged"] for *_, e in entries])),
        "mean_delta_fit_pp": float(np.mean([e["delta_fit_pp"] for *_, e in entries])),
        "n_gate_lost": int(sum(e["lagged"]["u2"] >= 1.0 for *_, e in entries)),
    }

    out_dir = paths.RESULTS_EXPERIMENTAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_path = out_dir / f"{ts}_round{round_num}_lagged_exog_v{version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"round": round_num, "exog_lag_quarters": lag, "min_train": mt,
                   "summary": summary, "combos": out},
                  f, ensure_ascii=False, indent=2, default=str)

    rows = ["| Combinação | Modelo | FIT% cont | FIT% defas | Δ p.p. | U₂ cont | U₂ defas | "
            "ρ₁ | FIT% AR(1) | excesso cont | excesso defas | ARIMA FIT% | "
            "DM-p defas×cont (Holm) | DM-p defas×ARIMA (Holm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for ec, i, e in entries:
        dc, da = e["dm_lagged_vs_contemporaneous"], e["dm_lagged_vs_arima"]
        rows.append(
            f"| {cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[i]} | {e['modelo_utilizado']} | "
            f"{e['contemporaneous']['fit']:.1f} | {e['lagged']['fit']:.1f} | "
            f"{e['delta_fit_pp']:+.1f} | {e['contemporaneous']['u2']:.3f} | "
            f"{e['lagged']['u2']:.3f} | {e['rho1']:.2f} | {e['fit_ar1_benchmark']:.1f} | "
            f"{e['excess_contemporaneous']:+.1f} | {e['excess_lagged']:+.1f} | "
            f"{e['arima']['fit']:.1f} | {dc['pvalue']:.3f} ({dc['pvalue_holm']:.3f}) | "
            f"{da['pvalue']:.3f} ({da['pvalue_holm']:.3f}) |")
    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Campeões com exógena, x(t) contra x(t−{lag}) (protocolo metodológico §29):**\n"
        + "\n".join(rows) + "\n"
        f"**Média do excesso sobre o AR(1):** contemporâneo "
        f"{summary['mean_excess_contemporaneous']:+.1f}; defasado "
        f"{summary['mean_excess_lagged']:+.1f} p.p.; perdas do gate U₂: {summary['n_gate_lost']}/{n}.\n"
        f"**Decisões:** round exploratório (protocolo metodológico §29). Mesma família, ordem, "
        f"pré-processamento e conjunto exógeno do campeão; só o alinhamento das exógenas muda. "
        f"Resultado em '{json_path.name}', em outputs/results/experimental/, fora de "
        f"compute_champions(); os campeões contemporâneos seguem oficiais.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")


def _run_fixed_rule_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combo_list: "list[tuple[int, str]]",
) -> None:
    """Round 39 — campeões congelados sob o tratamento de outliers por regra pré-fixada (§30).

    Reconstrói cada campeão com as factories de `plot_best_results.py` e roda o lote sobre a série
    vigente (deve reproduzir o FIT% armazenado) e sobre a série da regra pré-fixada: lista inteira
    sem reversões, imputação causal y(k−1), células da MP 579 brutas. Sem busca nova. Exploratório
    — grava em `outputs/results/experimental/`, fora de `compute_champions()`.
    """
    import plot_best_results as pbr

    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    champions = pbr.compute_champions()
    mt = cfg.MIN_TRAIN

    out: dict = {}
    print(f"[fixed-rule] imputação {cfg.FIXED_RULE_IMPUTE_METHOD}, sem reversões, "
          f"MP 579 bruta; campeão fixo")
    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        champ = champions[(emp_code, ind)]
        rec = pbr.load_json(pbr.RESULTS_DIR / champ["file"])[str(emp_code)][ind]
        cols = rec.get("exogenas_utilizadas", [])
        s_cur = pbr._prepare_series(emp_code, ind, datasets)
        s_new = pbr._prepare_series(emp_code, ind, datasets, fixed_rule=True)
        u_cur = pbr._build_exog(s_cur, cols, inputs)
        u_new = pbr._build_exog(s_new, cols, inputs)
        with pbr.round_hyperparams(champ["file"]):
            cv_c = walk_forward_cv(pbr._make_model_fn(rec, s_cur, u_cur), s_cur, u_cur, min_train=mt)
            cv_n = walk_forward_cv(pbr._make_model_fn(rec, s_new, u_new), s_new, u_new, min_train=mt)
        arima = _best_arima_baseline(s_new, mt)
        ar = arima["cv"]

        changed = [i for i in range(len(s_cur)) if not np.isclose(s_cur.iloc[i], s_new.iloc[i])]
        rho_c, rho_n = float(s_cur.autocorr(1)), float(s_new.autocorr(1))
        bench_c = 100.0 * (1.0 - float(np.sqrt(max(0.0, 1.0 - rho_c ** 2))))
        bench_n = 100.0 * (1.0 - float(np.sqrt(max(0.0, 1.0 - rho_n ** 2))))
        fit_stored = float(rec["percentual_fit"])
        if abs(cv_c["fit_percent"] - fit_stored) > 0.05:
            print(f"    [AVISO] {ticker}×{ind}: lote vigente {cv_c['fit_percent']:.2f} "
                  f"!= FIT armazenado {fit_stored:.2f}")
        is_arima = pbr._family(rec["modelo_utilizado"]) == "ARIMA"
        stat, p = diebold_mariano_test(np.asarray(cv_n["residuals"]), np.asarray(ar["residuals"]))
        out.setdefault(str(emp_code), {})[ind] = {
            "modelo_utilizado": rec["modelo_utilizado"],
            "descricao_parametros": rec["descricao_parametros"],
            "preprocessing": rec["preprocessing"],
            "exogenas_utilizadas": cols,
            "source_json": champ["file"],
            "fit_stored": fit_stored,
            "cells_changed": [str(s_cur.index[i].date()) for i in changed],
            "cells_changed_targets": int(sum(i >= mt for i in changed)),
            "current": {"fit": cv_c["fit_percent"], "u2": cv_c["theil_u2"], "rmse": cv_c["rmse"]},
            "fixed_rule": {"fit": cv_n["fit_percent"], "u2": cv_n["theil_u2"], "rmse": cv_n["rmse"]},
            "delta_fit_pp": cv_n["fit_percent"] - cv_c["fit_percent"],
            "arima": {"mode": arima["mode"], "order": list(arima["order"]),
                      "fit": ar["fit_percent"], "u2": ar["theil_u2"]},
            "rho1_current": rho_c, "rho1_fixed_rule": rho_n,
            "fit_ar1_benchmark_current": bench_c, "fit_ar1_benchmark_fixed_rule": bench_n,
            "excess_current": cv_c["fit_percent"] - bench_c,
            "excess_fixed_rule": cv_n["fit_percent"] - bench_n,
            "dm_fixed_rule_vs_arima": (None if is_arima
                                       else {"stat": float(stat), "pvalue": float(p)}),
            "y_true_fixed_rule": cv_n["y_true"].tolist(),
            "y_pred_fixed_rule": cv_n["y_pred"].tolist(),
        }
        e = out[str(emp_code)][ind]
        print(f"{ticker:<6} {ind:<22} {rec['modelo_utilizado']:<20} "
              f"vig {cv_c['fit_percent']:6.1f}/{cv_c['theil_u2']:.3f}  "
              f"fixa {cv_n['fit_percent']:6.1f}/{cv_n['theil_u2']:.3f}  "
              f"d{e['delta_fit_pp']:+6.1f}  células {len(changed)} "
              f"({e['cells_changed_targets']} alvos)", flush=True)

    entries = [(ec, i, e) for ec, inds in out.items() for i, e in inds.items()]
    tested = [e for *_, e in entries if e["dm_fixed_rule_vs_arima"] is not None]
    for e, ph in zip(tested, _holm_adjust([e["dm_fixed_rule_vs_arima"]["pvalue"] for e in tested])):
        e["dm_fixed_rule_vs_arima"]["pvalue_holm"] = ph
    n = len(entries)
    summary = {
        "n_combos": n,
        "mean_fit_current": float(np.mean([e["current"]["fit"] for *_, e in entries])),
        "mean_fit_fixed_rule": float(np.mean([e["fixed_rule"]["fit"] for *_, e in entries])),
        "mean_delta_fit_pp": float(np.mean([e["delta_fit_pp"] for *_, e in entries])),
        "mean_excess_current": float(np.mean([e["excess_current"] for *_, e in entries])),
        "mean_excess_fixed_rule": float(np.mean([e["excess_fixed_rule"] for *_, e in entries])),
        "n_gate_lost": int(sum(e["fixed_rule"]["u2"] >= 1.0 for *_, e in entries)),
        "n_holm_rejections": int(sum(e["dm_fixed_rule_vs_arima"]["pvalue_holm"] < 0.10
                                     for e in tested)),
        "n_tested": len(tested),
    }

    out_dir = paths.RESULTS_EXPERIMENTAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_path = out_dir / f"{ts}_round{round_num}_fixed_rule_v{version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"round": round_num, "impute_method": cfg.FIXED_RULE_IMPUTE_METHOD,
                   "raw_cells": {f"{k[0]}:{k[1]}": v for k, v in cfg.FIXED_RULE_RAW_CELLS.items()},
                   "min_train": mt, "summary": summary, "combos": out},
                  f, ensure_ascii=False, indent=2, default=str)

    rows = ["| Combinação | Modelo | Células (alvos) | FIT% vig | FIT% fixa | Δ p.p. | U₂ vig | "
            "U₂ fixa | ρ₁ vig | ρ₁ fixa | excesso vig | excesso fixa | ARIMA FIT%/U₂ | "
            "DM-p fixa×ARIMA (Holm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for ec, i, e in entries:
        dm = e["dm_fixed_rule_vs_arima"]
        dm_txt = "—" if dm is None else f"{dm['pvalue']:.3f} ({dm['pvalue_holm']:.3f})"
        rows.append(
            f"| {cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[i]} | {e['modelo_utilizado']} | "
            f"{len(e['cells_changed'])} ({e['cells_changed_targets']}) | "
            f"{e['current']['fit']:.1f} | {e['fixed_rule']['fit']:.1f} | "
            f"{e['delta_fit_pp']:+.1f} | {e['current']['u2']:.3f} | {e['fixed_rule']['u2']:.3f} | "
            f"{e['rho1_current']:.2f} | {e['rho1_fixed_rule']:.2f} | "
            f"{e['excess_current']:+.1f} | {e['excess_fixed_rule']:+.1f} | "
            f"{e['arima']['fit']:.1f}/{e['arima']['u2']:.3f} | {dm_txt} |")
    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Campeões congelados, tratamento vigente contra regra pré-fixada (protocolo metodológico §30):**\n"
        + "\n".join(rows) + "\n"
        f"**Médias:** FIT% {summary['mean_fit_current']:.1f} → {summary['mean_fit_fixed_rule']:.1f} "
        f"(Δ {summary['mean_delta_fit_pp']:+.1f} p.p.); excesso sobre o AR(1) "
        f"{summary['mean_excess_current']:+.1f} → {summary['mean_excess_fixed_rule']:+.1f}; "
        f"perdas do gate U₂: {summary['n_gate_lost']}/{n}; DM×ARIMA após Holm: "
        f"{summary['n_holm_rejections']}/{summary['n_tested']}.\n"
        f"**Decisões:** round exploratório, só para o manuscrito (decisão do autor, 2026-10-12). "
        f"Família, ordem, modo, λ, d e exógenas do campeão congelados; só a série tratada muda. "
        f"Resultado em '{json_path.name}', em outputs/results/experimental/, fora de "
        f"compute_champions(); campeões e TREATMENT_ERA da dissertação inalterados.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")


def _run_manuscript_treatment_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combo_list: "list[tuple[int, str]]",
) -> None:
    """Round 40 — campeões congelados sob o tratamento do manuscrito (protocolo metodológico §31).

    Série do manuscrito: regra pré-fixada do §30 mais os quartos trimestres corrigidos de
    `cfg.Q4_REBASE_OVERRIDES`; pontuação sem os alvos imputados (mesma máscara para campeão,
    ARIMA, AR(1) e passeio aleatório). Reavalia o campeão de cada combinação sob quatro leituras
    — vigente, manuscrito sem os alvos imputados (a principal), contra o valor registrado
    (sensibilidade) e contra o gabarito imputado — e anexa o AR(1) em walk-forward e o painel
    Clark-West/DM com Holm sobre a mesma máscara. Exploratório: grava em
    `outputs/results/experimental/`, fora de `compute_champions()`.
    """
    import plot_best_results as pbr
    sys.path.insert(0, str(paths.SCRIPTS_DIR))
    import ar1_walkforward as ar1_script
    import dm_multiplicity_panel as dm_script

    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    champions = pbr.compute_champions()

    out: dict = {}
    print("[manuscript-rule] campeão fixo; série do manuscrito; pontuação sem alvos imputados")
    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        champ = champions[(emp_code, ind)]
        cur = pbr.run_combo(emp_code, ind, champ, datasets, inputs)
        ms = pbr.run_combo(emp_code, ind, champ, datasets, inputs, manuscript=True)
        rec = pbr.run_combo(emp_code, ind, champ, datasets, inputs, manuscript=True,
                            score="recorded")
        imp = pbr.run_combo(emp_code, ind, champ, datasets, inputs, manuscript=True,
                            score="imputed")
        s_cur = pbr._prepare_series(emp_code, ind, datasets)
        s_ms, imputed = pbr._prepare_series_log(emp_code, ind, datasets, manuscript=True)
        oos = set(s_ms.index[cfg.MIN_TRAIN:])
        fit_stored = float(champ["fit"])
        if abs(cur["fit_actual"] - fit_stored) > 0.05:
            print(f"    [AVISO] {ticker}×{ind}: lote vigente {cur['fit_actual']:.2f} "
                  f"!= FIT armazenado {fit_stored:.2f}")
        changed = [str(s_cur.index[i].date()) for i in range(len(s_cur))
                   if not np.isclose(s_cur.iloc[i], s_ms.iloc[i])]
        out.setdefault(str(emp_code), {})[ind] = {
            "modelo_utilizado": cur["model"], "descricao_parametros": cur["desc"],
            "preprocessing": cur["mode"], "exogenas_utilizadas": cur["exog"],
            "source_json": champ["file"], "fit_stored": fit_stored,
            "cells_changed": changed,
            "cells_imputed": [str(t.date()) for t in imputed],
            "targets_excluded": ms["n_excluded"],
            "targets_excluded_dates": [str(t.date()) for t in imputed if t in oos],
            "current": {"fit": cur["fit_actual"], "u2": cur["u2_actual"]},
            "manuscript": {"fit": ms["fit_actual"], "u2": ms["u2_actual"]},
            "recorded": {"fit": rec["fit_actual"], "u2": rec["u2_actual"]},
            "imputed_scored": {"fit": imp["fit_actual"], "u2": imp["u2_actual"]},
            "delta_fit_pp": ms["fit_actual"] - cur["fit_actual"],
            "rho1_current": float(s_cur.autocorr(1)), "rho1_manuscript": float(s_ms.autocorr(1)),
        }
        e = out[str(emp_code)][ind]
        print(f"{ticker:<6} {ind:<22} {cur['model']:<20} "
              f"vig {cur['fit_actual']:6.1f}/{cur['u2_actual']:.3f}  "
              f"ms {ms['fit_actual']:6.1f}/{ms['u2_actual']:.3f}  "
              f"reg {rec['fit_actual']:6.1f}/{rec['u2_actual']:.3f}  "
              f"gab {imp['fit_actual']:6.1f}/{imp['u2_actual']:.3f}  "
              f"células {len(changed)}  alvos excl. {e['targets_excluded']}", flush=True)

    print("[manuscript-rule] AR(1) walk-forward e painel de significância na mesma máscara")
    ar1_df = ar1_script.build_table(manuscript=True)
    sig_df = dm_script.build_table(manuscript=True)
    for r in ar1_df.itertuples():
        ec = next(c for c, t in cfg.EMPRESAS.items() if t == r.empresa)
        out[str(ec)][r.indicador]["ar1_wf"] = {
            "fit": float(r.fit_ar1_wf), "u2": float(r.u2_ar1_wf),
            "excess": float(r.excesso_wf), "dm_p_champion_vs_ar1": float(r.dm_p_campeao_vs_ar1),
            "exog": bool(r.exogena),
        }
    for r in sig_df.itertuples():
        ec = next(c for c, t in cfg.EMPRESAS.items() if t == r.empresa)
        out[str(ec)][r.indicador]["significance"] = {
            "test": r.teste, "pvalue": float(r.pvalue), "holm_p": float(r.holm_p),
            "holm_rejects": bool(r.holm_rejeita), "nested": bool(r.aninhado),
            "dm_p_all": float(r.dm_p_todos), "mcs": r.mcs_sobreviventes,
        }

    entries = [(ec, i, e) for ec, inds in out.items() for i, e in inds.items()]
    n = len(entries)
    summary = {
        "n_combos": n,
        "mean_fit_current": float(np.mean([e["current"]["fit"] for *_, e in entries])),
        "mean_fit_manuscript": float(np.mean([e["manuscript"]["fit"] for *_, e in entries])),
        "mean_fit_recorded": float(np.mean([e["recorded"]["fit"] for *_, e in entries])),
        "mean_excess_ar1_wf": float(np.mean([e["ar1_wf"]["excess"] for *_, e in entries])),
        "mean_excess_ar1_wf_exog": float(np.mean(
            [e["ar1_wf"]["excess"] for *_, e in entries if e["ar1_wf"]["exog"]])),
        "mean_excess_ar1_wf_univ": float(np.mean(
            [e["ar1_wf"]["excess"] for *_, e in entries if not e["ar1_wf"]["exog"]])),
        "n_gate_lost_manuscript": int(sum(e["manuscript"]["u2"] >= 1.0 for *_, e in entries)),
        "n_gate_lost_recorded": int(sum(e["recorded"]["u2"] >= 1.0 for *_, e in entries)),
        "n_ar1_passes_gate": int(sum(e["ar1_wf"]["u2"] < 1.0 for *_, e in entries)),
        "n_targets_excluded": int(sum(e["targets_excluded"] for *_, e in entries)),
        "n_holm_rejections": int(sig_df["holm_rejeita"].sum()),
        "n_sig_tested": int(len(sig_df)),
    }

    out_dir = paths.RESULTS_EXPERIMENTAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_path = out_dir / f"{ts}_round{round_num}_manuscript_rule_v{version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"round": round_num, "impute_method": cfg.FIXED_RULE_IMPUTE_METHOD,
                   "raw_cells": {f"{k[0]}:{k[1]}": v for k, v in cfg.FIXED_RULE_RAW_CELLS.items()},
                   "q4_rebase": {f"{k[0]}:{k[1]}": v for k, v in cfg.Q4_REBASE_OVERRIDES.items()},
                   "min_train": cfg.MIN_TRAIN, "summary": summary, "combos": out},
                  f, ensure_ascii=False, indent=2, default=str)

    rows = ["| Combinação | Modelo | Alvos excl. | FIT% vig | FIT% ms | Δ p.p. | U₂ vig | U₂ ms | "
            "FIT% reg. | U₂ reg. | ρ₁ ms | FIT% AR(1) | excesso | teste: p (Holm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for ec, i, e in entries:
        sg = e.get("significance")
        sg_txt = "—" if sg is None else f"{sg['test'][:2]}: {sg['pvalue']:.3f} ({sg['holm_p']:.3f})"
        rows.append(
            f"| {cfg.EMPRESAS[int(ec)]}×{_IND_SHORT[i]} | {e['modelo_utilizado']} | "
            f"{e['targets_excluded']} | {e['current']['fit']:.1f} | {e['manuscript']['fit']:.1f} | "
            f"{e['delta_fit_pp']:+.1f} | {e['current']['u2']:.3f} | {e['manuscript']['u2']:.3f} | "
            f"{e['recorded']['fit']:.1f} | {e['recorded']['u2']:.3f} | "
            f"{e['rho1_manuscript']:.2f} | {e['ar1_wf']['fit']:.1f} | "
            f"{e['ar1_wf']['excess']:+.1f} | {sg_txt} |")
    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Campeões congelados, tratamento vigente contra tratamento do manuscrito "
        f"(protocolo metodológico §31):**\n"
        + "\n".join(rows) + "\n"
        f"**Médias:** FIT% {summary['mean_fit_current']:.1f} → {summary['mean_fit_manuscript']:.1f} "
        f"(contra o registrado: {summary['mean_fit_recorded']:.1f}); excesso sobre o AR(1) em "
        f"walk-forward {summary['mean_excess_ar1_wf']:+.1f} (exógena "
        f"{summary['mean_excess_ar1_wf_exog']:+.1f}, univariado "
        f"{summary['mean_excess_ar1_wf_univ']:+.1f}); perdas do gate U₂: "
        f"{summary['n_gate_lost_manuscript']}/{n} (contra o registrado "
        f"{summary['n_gate_lost_recorded']}/{n}); alvos excluídos: {summary['n_targets_excluded']} "
        f"de {n * (64 - cfg.MIN_TRAIN)}; AR(1) com U₂<1: {summary['n_ar1_passes_gate']}/{n}; "
        f"sobreviventes a Holm: {summary['n_holm_rejections']}/{summary['n_sig_tested']}.\n"
        f"**Decisões:** round exploratório, só para o manuscrito (decisão do autor, 2026-10-12). "
        f"Família, ordem, modo, λ, d e exógenas do campeão congelados; só a série e a pontuação "
        f"mudam. Resultado em '{json_path.name}', em outputs/results/experimental/, fora de "
        f"compute_champions(); campeões e TREATMENT_ERA da dissertação inalterados.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_path.name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")


# Famílias bloco-orientadas da reeleição do Round 41: (prefixo do nome, configs, wrapper).
_MS_NL_FAMILIES = (
    ("WIENER", "WIENER_CONFIGS", WienerWrapper),
    ("HAMMER", "HAMMERSTEIN_CONFIGS", HammersteinWrapper),
    ("HW", "HW_CONFIGS", HWWrapper),
)


def _ms_tie_key(cand: dict) -> tuple[int, int, int]:
    """Desempate da reeleição (protocolo metodológico §32): família na ordem de `cfg.MODELOS`, depois
    modo na ordem de `cfg.PREPROC_MODES`, depois a grade padrão antes da estendida."""
    rec = cand["record"]
    fam = rec["modelo_utilizado"].split("-")[0].upper()
    fam = "HAMMERSTEIN" if fam == "HAMMER" else fam
    ordem = [m["name"] for m in cfg.MODELOS]
    modos = list(cfg.PREPROC_MODES)
    grades = list(cfg.MANUSCRIPT_SEARCH_GRIDS)
    return (ordem.index(fam) if fam in ordem else len(ordem),
            modos.index(rec["preprocessing"]) if rec["preprocessing"] in modos else len(modos),
            grades.index(rec.get("grade_hiperparametros", grades[0])))


def _ms_nl_candidate(prefix: str, nl_cfg: dict, wrapper_cls, series, d: int, mode: str):
    """Um candidato bloco-orientado AR puro — mesma construção de `_run_nl_round`."""
    if prefix == "WIENER":
        kw = {"output_nl": nl_cfg["output_nl"], "n_knots": nl_cfg.get("n_knots")}
        fo = _select_nl_order(series, wrapper_cls, kw, d, mode)
        mfn = _make_wiener_fn(nl_cfg["output_nl"], nl_cfg.get("n_knots"), d, mode, fo)
        family = "WIENER"
    elif prefix == "HAMMER":
        kw = {"input_nl": nl_cfg["input_nl"], "n_knots": nl_cfg.get("n_knots")}
        fo = _select_nl_order(series, wrapper_cls, kw, d, mode)
        mfn = _make_hammerstein_fn(nl_cfg["input_nl"], nl_cfg.get("n_knots"), d, mode, fo)
        family = "HAMMERSTEIN"
    else:
        kw = {"input_nl": nl_cfg["input_nl"], "output_nl": nl_cfg["output_nl"],
              "n_knots_in": nl_cfg.get("n_knots_in"), "n_knots_out": nl_cfg.get("n_knots_out")}
        fo = _select_nl_order(series, wrapper_cls, kw, d, mode)
        mfn = _make_hw_fn(nl_cfg["input_nl"], nl_cfg["output_nl"], nl_cfg.get("n_knots_in"),
                          nl_cfg.get("n_knots_out"), d, mode, fo)
        family = "HW"
    cv = walk_forward_cv(mfn, series, None, min_train=cfg.MIN_TRAIN)
    return _make_result_dict(f"{prefix}-{nl_cfg['name']}", f"{family}({nl_cfg['name']},{mode})",
                             mode, [], d, cv)


def _manuscript_search_combo(emp_code: int, ind: str, exog_cols: list[str]) -> dict:
    """Round 41 (protocolo metodológico §32): a busca inteira de uma combinação sob o tratamento do
    manuscrito. Roda num processo próprio; devolve todos os candidatos com seus arrays fora da
    amostra (alvos imputados já fora, a mesma máscara para todos).

    Espaço: `_run_all_models_for_combo` (com as exógenas do funil refeito), o ARIMA com ordem da
    janela inicial, as famílias bloco-orientadas AR puro, SETAR e LSTM, nos cinco modos e nas duas
    grades de `cfg.MANUSCRIPT_SEARCH_GRIDS` (a estendida só para as famílias que ela altera).
    """
    import plot_best_results as pbr
    from src.evaluation import score_exclusion

    ticker = cfg.EMPRESAS[emp_code]
    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    series, imputed = pbr._prepare_series_log(emp_code, ind, datasets, manuscript=True)
    oos = set(series.index[cfg.MIN_TRAIN:])
    mask_dates = [t for t in imputed if t in oos]
    u_exog = pbr._build_exog(series, exog_cols, inputs)

    candidates: dict[str, dict] = {}

    def _keep(key: str, r: dict, grid: str, hyper: dict) -> None:
        r["grade_hiperparametros"] = grid
        r["hyperparams"] = dict(hyper)
        candidates[key] = r

    with score_exclusion(mask_dates):
        for grid, hyper in cfg.MANUSCRIPT_SEARCH_GRIDS.items():
            extended = bool(hyper)
            with pbr.hyperparams_applied(hyper, label=f"grade {grid}"):
                cfg.apply_global_seed()
                for mode in cfg.PREPROC_MODES:
                    try:
                        lam = guerrero_lambda(series) if mode == "boxcox" else None
                        d = unit_root_tests(transform(series, mode, lam))["consensus_d"]
                    except Exception as exc:
                        warnings.warn(f"[{ticker}×{ind}] modo {mode}: {exc}")
                        continue

                    lin: dict[str, dict] = {}
                    try:
                        _run_all_models_for_combo(
                            series, u_exog, mode, d, list(exog_cols), ticker, ind, None,
                            boxcox_lambda=lam, candidates_out=lin)
                    except Exception as exc:
                        warnings.warn(f"[{ticker}×{ind}] busca linear/{mode}: {exc}")
                    for name, r in lin.items():
                        if extended and not name.startswith(cfg.MANUSCRIPT_GRID_SENSITIVE):
                            continue
                        _keep(f"{name}|{mode}|{grid}", r, grid, hyper)

                    if not extended:
                        try:
                            fo = _select_arima_order(series, d, mode, cfg.MIN_TRAIN, lam)
                            cv = walk_forward_cv(_make_arima_fn(d, mode, fo, lam), series, None,
                                                 min_train=cfg.MIN_TRAIN)
                            _keep(f"ARIMA|{mode}|{grid}", _make_result_dict(
                                "ARIMA", f"ARIMA{tuple(fo)}", mode, [], d, cv, lam), grid, hyper)
                        except Exception as exc:
                            warnings.warn(f"[{ticker}×{ind}] ARIMA/{mode}: {exc}")
                        try:
                            init = SETARWrapper(p_max=cfg.SETAR_P_MAX, delay_max=cfg.SETAR_DELAY_MAX,
                                                trim=cfg.SETAR_TRIM, mode=mode
                                                ).fit(series.iloc[:cfg.MIN_TRAIN])
                            cv = walk_forward_cv(_make_setar_fn(mode, init.fixed_config_), series,
                                                 None, min_train=cfg.MIN_TRAIN)
                            _keep(f"SETAR|{mode}|{grid}", _make_result_dict(
                                "SETAR", init.get_params_str(), mode, [], d, cv), grid, hyper)
                        except Exception as exc:
                            warnings.warn(f"[{ticker}×{ind}] SETAR/{mode}: {exc}")
                        try:
                            cfg.apply_global_seed()
                            cv = walk_forward_cv(_make_lstm_fn(mode), series, None,
                                                 min_train=cfg.MIN_TRAIN)
                            _keep(f"LSTM|{mode}|{grid}", _make_result_dict(
                                "LSTM", "LSTM(lb=4,units=8)", mode, [], d, cv), grid, hyper)
                        except Exception as exc:
                            warnings.warn(f"[{ticker}×{ind}] LSTM/{mode}: {exc}")

                    # Bloco-orientados: mesma transformação de `_run_nl_round` (sem λ explícito).
                    try:
                        d_nl = unit_root_tests(transform(series, mode))["consensus_d"]
                    except Exception:
                        continue
                    for prefix, cfg_attr, wrapper_cls in _MS_NL_FAMILIES:
                        for nl_cfg in getattr(cfg, cfg_attr):
                            try:
                                r = _ms_nl_candidate(prefix, nl_cfg, wrapper_cls, series, d_nl, mode)
                                _keep(f"{r['modelo_utilizado']}|{mode}|{grid}", r, grid, hyper)
                            except Exception as exc:
                                warnings.warn(f"[{ticker}×{ind}] {prefix}/{nl_cfg['name']}/"
                                              f"{mode}: {exc}")

    y_true = None
    out_cands = []
    for key, r in candidates.items():
        yt = np.asarray(r.get("_y_true"), dtype=float)
        if y_true is None:
            y_true = yt
        out_cands.append({
            "key": key,
            "record": {k: v for k, v in r.items() if not k.startswith("_")},
            "y_pred": np.asarray(r.get("_y_pred"), dtype=float).tolist(),
            "y_rw": np.asarray(r.get("_y_rw"), dtype=float).tolist(),
            "same_targets": bool(np.allclose(yt, y_true, equal_nan=False)),
        })
    return {
        "emp_code": emp_code, "indicador": ind, "exog": list(exog_cols),
        "imputed": [str(t.date()) for t in imputed],
        "targets_excluded": [str(t.date()) for t in mask_dates],
        "oos_index": [str(t.date()) for t in series.index[cfg.MIN_TRAIN:] if t not in set(mask_dates)],
        "y_true": [] if y_true is None else y_true.tolist(),
        "candidates": out_cands,
    }


def _run_manuscript_search_round(
    round_num: int,
    version: int,
    hypothesis: str,
    next_round: str,
    combo_list: "list[tuple[int, str]]",
) -> None:
    """Round 41 — reeleição dos campeões sob o tratamento do manuscrito (protocolo metodológico §32).

    Refaz o funil de exógenas sobre a série do manuscrito, roda a busca inteira por combinação em
    paralelo (`_manuscript_search_combo`), elege pelo maior FIT% com U₂ < 1 (sem candidato válido,
    a combinação fica sem campeão), confere a reprodução do eleito por `plot_best_results`, e aplica
    o SPA de Hansen a todos os candidatos contra o ARIMA de referência. Exploratório: grava em
    `outputs/results/experimental/`, fora de `compute_champions()`.
    """
    from concurrent.futures import ProcessPoolExecutor
    import plot_best_results as pbr
    from src.evaluation import spa_test
    from src.feature_selection import run_feature_selection
    sys.path.insert(0, str(paths.SCRIPTS_DIR))
    import dm_multiplicity_panel as dm_script

    cfg.apply_global_seed()
    paths.RESULTS_EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    print("[manuscript-search] funil de exógenas sobre a série do manuscrito")
    run_feature_selection(granger_alpha_override=cfg.GRANGER_ALPHA_OVERRIDE,
                          output_path=paths.EXOG_SELECTION_MANUSCRIPT, manuscript=True)
    with open(paths.EXOG_SELECTION_MANUSCRIPT, encoding="utf-8") as f:
        exog_sel = json.load(f)

    r40_files = sorted(paths.RESULTS_EXPERIMENTAL_DIR.glob("*_round40_manuscript_rule_v*.json"))
    r40 = json.load(open(r40_files[-1], encoding="utf-8"))["combos"] if r40_files else {}

    print(f"[manuscript-search] busca em {cfg.MANUSCRIPT_SEARCH_WORKERS} processos")
    jobs = [(ec, ind, exog_sel.get(str(ec), {}).get(ind, [])) for ec, ind in combo_list]
    with ProcessPoolExecutor(max_workers=cfg.MANUSCRIPT_SEARCH_WORKERS) as pool:
        searched = list(pool.map(_manuscript_search_combo, *zip(*jobs)))

    rb, lc, do, inputs = load_all()
    datasets = {"resultado_bruto": rb, "lucro_consolidado": lc, "despesas_operacionais": do}
    ts = datetime.now().strftime("%Y%m%d%H%M")
    json_name = f"{ts}_round{round_num}_manuscript_search_v{version}.json"

    out: dict = {}
    for s in searched:
        ec, ind = s["emp_code"], s["indicador"]
        ticker = cfg.EMPRESAS[ec]
        cands = [c for c in s["candidates"] if c["same_targets"]
                 and np.all(np.isfinite(c["y_pred"]))]
        valid = [c for c in cands if c["record"]["theil_u2"] < 1.0]
        pool_ = valid if valid else cands
        top = max(c["record"]["percentual_fit"] for c in pool_)
        tied = [c for c in pool_ if top - c["record"]["percentual_fit"] < 1e-6]
        best = min(tied, key=_ms_tie_key)  # §32: empate exato → ordem de cfg.MODELOS
        champ_rec = best["record"]

        # Reprodução do eleito pelo caminho do painel (plot_best_results).
        try:
            rep = pbr.run_combo(ec, ind, {"record": champ_rec, "file": json_name}, datasets,
                                inputs, manuscript=True)
            repro = {"fit": float(rep["fit_actual"]), "u2": float(rep["u2_actual"]),
                     "delta_fit": float(rep["fit_actual"] - champ_rec["percentual_fit"])}
        except Exception as exc:
            repro = {"error": str(exc)}

        # SPA: todos os candidatos contra o ARIMA de referência, na mesma máscara.
        series, imputed = pbr._prepare_series_log(ec, ind, datasets, manuscript=True)
        y_true = np.asarray(s["y_true"], dtype=float)
        y_ref = np.asarray(dm_script._fresh_arima_baseline(series, imputed), dtype=float)
        e_models = np.column_stack([y_true - np.asarray(c["y_pred"]) for c in cands])
        spa = spa_test(y_true - y_ref, e_models, block_size=cfg.SPA_BLOCK_SIZE,
                       reps=cfg.SPA_REPS, seed=cfg.RANDOM_SEED)

        prev = r40.get(str(ec), {}).get(ind, {})
        out.setdefault(str(ec), {})[ind] = {
            "champion": champ_rec if valid else None,
            "best_failing": None if valid else champ_rec,
            "gate_passed": bool(valid),
            "n_candidates": len(cands), "n_valid": len(valid),
            "n_dropped": len(s["candidates"]) - len(cands),
            "exog_funnel": s["exog"],
            "exog_round40": prev.get("exogenas_utilizadas"),
            "round40": {"model": prev.get("modelo_utilizado"),
                        "mode": prev.get("preprocessing"),
                        "fit": (prev.get("manuscript") or {}).get("fit"),
                        "u2": (prev.get("manuscript") or {}).get("u2")},
            "reproduction": repro,
            "spa_vs_arima_ref": spa,
            "targets_excluded": s["targets_excluded"],
            "oos_index": s["oos_index"],
            "y_true": s["y_true"],
            "y_pred_arima_ref": y_ref.tolist(),
            "candidates": [{"key": c["key"], "model": c["record"]["modelo_utilizado"],
                            "mode": c["record"]["preprocessing"],
                            "grid": c["record"]["grade_hiperparametros"],
                            "fit": c["record"]["percentual_fit"], "u2": c["record"]["theil_u2"],
                            "y_pred": c["y_pred"]} for c in cands],
        }
        tag = "" if valid else "  [sem campeão: nenhum U₂<1]"
        print(f"{ticker:<6} {_IND_SHORT[ind]:<6} {champ_rec['modelo_utilizado']:<22} "
              f"{champ_rec['preprocessing']:<9} FIT {champ_rec['percentual_fit']:6.1f} "
              f"U₂ {champ_rec['theil_u2']:.3f}  SPA p {spa['p_consistent']:.3f}  "
              f"repro {repro.get('delta_fit', repro.get('error'))}{tag}")

    json_path = paths.RESULTS_EXPERIMENTAL_DIR / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"round": round_num, "grids": cfg.MANUSCRIPT_SEARCH_GRIDS,
                   "min_train": cfg.MIN_TRAIN, "spa": {"block_size": cfg.SPA_BLOCK_SIZE,
                                                        "reps": cfg.SPA_REPS},
                   "exog_selection": paths.EXOG_SELECTION_MANUSCRIPT.name,
                   "combos": out}, f, ensure_ascii=False, default=float)

    rows = ["| Combinação | Eleito (Round 41) | Modo | Exóg. | FIT% | U₂ | Round 40 | FIT% R40 | "
            "U₂ R40 | Candidatos (U₂<1) | SPA p (White) | Repro Δ |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for ec, ind in combo_list:
        e = out[str(ec)][ind]
        rec = e["champion"] or e["best_failing"]
        name = rec["modelo_utilizado"] + ("" if e["gate_passed"] else " (reprovado)")
        r4 = e["round40"]
        rp = e["reproduction"].get("delta_fit")
        rows.append(
            f"| {cfg.EMPRESAS[ec]}×{_IND_SHORT[ind]} | {name} | {rec['preprocessing']} | "
            f"{len(rec.get('exogenas_utilizadas', []))} | {rec['percentual_fit']:.1f} | "
            f"{rec['theil_u2']:.3f} | {r4['model']} | "
            f"{(r4['fit'] if r4['fit'] is not None else float('nan')):.1f} | "
            f"{(r4['u2'] if r4['u2'] is not None else float('nan')):.3f} | "
            f"{e['n_candidates']} ({e['n_valid']}) | {e['spa_vs_arima_ref']['p_consistent']:.3f} "
            f"({e['spa_vs_arima_ref']['p_white']:.3f}) | "
            f"{'—' if rp is None else f'{rp:+.2f}'} |")
    n_gate = sum(out[str(ec)][ind]["gate_passed"] for ec, ind in combo_list)
    log_entry = (
        f"\n## Round {round_num:02d} — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"**Hipótese:** {hypothesis}\n"
        f"**Reeleição sob o tratamento do manuscrito (protocolo metodológico §32):**\n"
        + "\n".join(rows) + "\n"
        f"**Gate U₂ < 1:** {n_gate}/{len(combo_list)}. Funil de exógenas refeito em "
        f"'{paths.EXOG_SELECTION_MANUSCRIPT.name}'. SPA de Hansen contra o ARIMA de referência, "
        f"bootstrap estacionário (bloco {cfg.SPA_BLOCK_SIZE}, {cfg.SPA_REPS} réplicas).\n"
        f"**Decisões:** round exploratório, só para o manuscrito (decisão do autor, 2026-10-17). "
        f"Resultado em '{json_name}', em outputs/results/experimental/, fora de "
        f"compute_champions(); campeões e TREATMENT_ERA da dissertação inalterados.\n"
        f"**Próximo:** {next_round}\n"
        f"*JSON: {json_name}*\n"
    )
    _append_science_log(log_entry)
    print(f"\nJSON salvo: {json_path}")


def run_wiener_round(use_exog: bool = False) -> None:
    """Round 05 — Wiener Puro: ARIMA(p,d,q) → g(w).

    Sub-round 05a (padrão): AR puro, u=None.
    Sub-round 05b (--round wiener-arx): inclui exog de exog_selection_v4.json.
    """
    exog_path = paths.RESULTS_DIR / "exog_selection_v4.json"
    _run_nl_round(
        round_num=5,
        model_family="WIENER",
        configs=cfg.WIENER_CONFIGS,
        make_fn=_make_wiener_fn,
        version=5,
        hypothesis=(
            "Round 05 — Wiener Puro: bloco linear ARIMA(p,d,q) + não-linearidade estática "
            "na saída g(w). Hipótese: regime change na precificação capturado por g."
        ),
        next_round=(
            "Round 06 — Hammerstein Puro: não-linearidade na entrada f(y_lags) + ARIMA. "
            "Compara fonte de NL: entrada vs saída."
        ),
        use_exog=use_exog,
        exog_path=exog_path,
    )


def run_hammerstein_round(use_exog: bool = False) -> None:
    """Round 06 — Hammerstein Puro: f(y_lags) → ARIMA(p,d,q).

    Sub-round 06a (padrão): AR puro, u=None.
    Sub-round 06b (--round hammerstein-arx): inclui exog.
    """
    exog_path = paths.RESULTS_DIR / "exog_selection_v4.json"
    _run_nl_round(
        round_num=6,
        model_family="HAMMERSTEIN",
        configs=cfg.HAMMERSTEIN_CONFIGS,
        make_fn=_make_hammerstein_fn,
        version=6,
        hypothesis=(
            "Round 06 — Hammerstein Puro: não-linearidade estática na entrada f(y_lags) "
            "+ bloco linear ARIMA(p,d,q). Hipótese: saturação dos indicadores contábeis "
            "capturada por f antes do bloco dinâmico."
        ),
        next_round=(
            "Round 07 — Hammerstein-Wiener: matriz completa de 5 tentativas (f→ARIMA→g). "
            "Identifica a combinação ótima de NL entrada + NL saída."
        ),
        use_exog=use_exog,
        exog_path=exog_path,
    )


# ---------------------------------------------------------------------------
# Round 08 — LS dummies + imputação expandida de AO/TC
# ---------------------------------------------------------------------------

def run_round08() -> None:
    """Round 08 — imputação expandida de AO/TC + dummies binárias de Level Shift.

    Usa:
    - SPIKE_KNOWN atualizado (Round 08): todos os AO/TC detectados em Round 05-diag.
    - LEVEL_SHIFT_DUMMIES: colunas LS_<YYYYMMDD> concatenadas ao u_exog de cada combo.
    - Macro exog de exog_selection_v4.json (sem re-executar feature selection).
    """
    cfg.apply_global_seed()

    exog_path = paths.RESULTS_DIR / "exog_selection_v4.json"
    if not exog_path.exists():
        raise FileNotFoundError(f"Exog selection não encontrado: {exog_path}")
    with open(exog_path, encoding="utf-8") as _f:
        exog_sel = json.load(_f)

    latest = _find_latest_result()
    baseline_results: dict = {}
    if latest:
        with open(latest, encoding="utf-8") as _f:
            baseline_results = json.load(_f)
        print(f"[Round08] Baseline: {latest.name}")

    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    results: dict = {}
    spike_log: list[dict] = []
    col_w = 100
    print("=" * col_w)
    print(
        f"{'Empresa':<8} {'Indicador':<22} {'Melhor':<12} {'FIT%':>6} "
        f"{'U2':>6} {'DM-p':>5}  Exog"
    )
    print("-" * col_w)

    for emp_code, ticker in cfg.EMPRESAS.items():
        results[str(emp_code)] = {}
        for ind in cfg.INDICADORES:
            series = datasets[ind][str(emp_code)].dropna()

            # --- Imputação expandida (AO/TC do Round 05-diag via SPIKE_KNOWN atualizado) ---
            _known_ts = [pd.Timestamp(t) for t in cfg.known_spikes_for(emp_code, ind)]
            if _known_ts or cfg.SPIKE_Z_THRESH:
                series, _recs = detect_and_impute_spikes(
                    series,
                    z_thresh=cfg.SPIKE_Z_THRESH,
                    method=cfg.SPIKE_IMPUTE_METHOD,
                    known_spikes=_known_ts,
                )
                for rec in _recs:
                    spike_log.append({"empresa": ticker, "indicador": ind, **rec})

            # --- Exógenas macro (base Round 04) ---
            sel_cols: list[str] = exog_sel.get(str(emp_code), {}).get(ind, [])
            u_exog: pd.DataFrame | None = None
            if sel_cols:
                u_exog = inputs[sel_cols].reindex(series.index).ffill().bfill()

            # --- Level Shift dummies (concatenar ao u_exog) ---
            ls_dates = cfg.LEVEL_SHIFT_DUMMIES.get((emp_code, ind), [])
            exog_cols: list[str] = list(sel_cols)
            if ls_dates:
                ls_df = build_ls_dummies(series.index, ls_dates)
                u_exog = pd.concat([u_exog, ls_df], axis=1) if u_exog is not None else ls_df
                exog_cols = exog_cols + list(ls_df.columns)

            bl = baseline_results.get(str(emp_code), {}).get(ind, {})
            bl_mode = bl.get("preprocessing", "raw")
            bl_d    = int(bl.get("unit_root_d", 1))

            best_r = _run_all_models_for_combo(
                series, u_exog, bl_mode, bl_d, exog_cols,
                ticker, ind, bl if bl else None,
            )
            results[str(emp_code)][ind] = best_r

            dm_p = best_r.get("dm_vs_arima", {}).get("pvalue", 1.0)
            ls_tag = f"+{len(ls_dates)}LS" if ls_dates else ""
            print(
                f"{ticker:<8} {ind:<22} {best_r['modelo_utilizado']:<12} "
                f"{best_r['percentual_fit']:>6.1f} {best_r['theil_u2']:>6.3f} "
                f"{dm_p:>5.3f}  {', '.join(best_r['exogenas_utilizadas'][:2])}{ls_tag}"
            )

    print("=" * col_w)

    # Sumário de imputações para o log
    spike_lines = []
    for rec in spike_log:
        ts_str = rec["timestamp"].strftime("%Y-%m-%d")
        spike_lines.append(
            f"  - {rec['empresa']}×{_IND_SHORT[rec['indicador']]}: "
            f"{ts_str} {rec['original']/1e9:.3f}B→{rec['imputed']/1e9:.3f}B (z={rec['z_score']})"
        )
    spike_summary = "**Imputações expandidas (AO/TC):**\n" + "\n".join(spike_lines) + "\n" if spike_lines else ""

    ls_summary_lines = [
        f"  - {cfg.EMPRESAS[ec]}×{_IND_SHORT[ind]}: {', '.join(dates)}"
        for (ec, ind), dates in cfg.LEVEL_SHIFT_DUMMIES.items()
    ]
    ls_summary = "**Dummies LS adicionadas:**\n" + "\n".join(ls_summary_lines) + "\n"

    json_path = save_results(results, version=8)
    _append_science_log_round(
        round_num=8,
        results=results,
        json_path=json_path,
        hypothesis=(
            "Round 08 — tratamento diferenciado por tipologia (Round 05-diag): "
            "imputação linear de AO/TC (SPIKE_KNOWN expandido) + dummies binárias LS "
            "(LEVEL_SHIFT_DUMMIES) concatenadas ao u_exog de 5 combinações afetadas."
        ),
        decisions=(
            spike_summary
            + ls_summary
            + "Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA baseline. "
            f"Macro exog de '{exog_path.name}'."
        ),
        next_round=(
            "Round 09 — análise de sensibilidade: avaliar combinações com FIT%<0 persistente; "
            "decidir aceitação formal de não-preditibilidade ou expansão de features."
        ),
        baseline_results=baseline_results if baseline_results else None,
    )
    print(f"\nScience log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


def run_hw_round(use_exog: bool = False) -> None:
    """Round 07 — Hammerstein-Wiener: f(y_lags) → ARIMA(p,d,q) → g(w).

    Sub-round 07a (padrão): AR puro — isola estrutura NL.
    Sub-round 07b (--round hw-arx): inclui exog — quantifica ganho marginal.
    """
    exog_path = paths.RESULTS_DIR / "exog_selection_v4.json"
    _run_nl_round(
        round_num=7,
        model_family="HW",
        configs=cfg.HW_CONFIGS,
        make_fn=_make_hw_fn,
        version=7,
        hypothesis=(
            "Round 07 — Hammerstein-Wiener: matriz de 5 tentativas "
            "(T1-GatilhoMercado, T2-QuebraEstrutural, T3-BenchmarkDD, "
            "T4-MapeamentoFlex, T6-AssimetriaReg). "
            "Hipótese: combinação ótima de NL entrada+saída supera Wiener e Hammerstein puros."
        ),
        next_round=(
            "Round 08 — Sensibilidade: threshold fixo vs otimizado (Piecewise), "
            "n_knots ∈ {3,5,7} (BSpline), regularização L1 vs L2."
        ),
        use_exog=use_exog,
        exog_path=exog_path,
    )


# ---------------------------------------------------------------------------
# Round 09 — ISAE4×resultado_bruto: Hammerstein-Sigmoid + IPA-M (diff)
# ---------------------------------------------------------------------------

def run_round09() -> None:
    """Round 09 — ISAE4×resultado_bruto: Hammerstein-H1-Sigmoid + IPA-M + diff.

    Experimento focado: replica exatamente a configuração vencedora do R06a
    (Hammerstein-Sigmoid, diff) acrescentando IPA-M como variável exógena.
    Hipótese: IPA-M captura pressão inflacionária nos insumos regulados da ISAE4,
    melhorando o FIT% além dos 52.2% obtidos no R06a (sem exógenas).
    """
    cfg.apply_global_seed()

    emp_code = 18376   # ISAE4
    ticker   = "ISAE4"
    ind      = "resultado_bruto"
    mode_fix = "diff"
    ipam_col = "IPA-M - Fechamento mensal (% mês)  (1478707)"

    exog_path = paths.RESULTS_DIR / "exog_selection_v9.json"
    if not exog_path.exists():
        raise FileNotFoundError(f"Exog selection não encontrado: {exog_path}")

    rb, _lc, _do, inputs = load_all()
    series = rb[str(emp_code)].dropna()

    # Spike treatment (mesmo protocolo rounds anteriores)
    known_ts = [pd.Timestamp(t) for t in cfg.known_spikes_for(emp_code, ind)]
    series, _ = detect_and_impute_spikes(
        series,
        z_thresh=cfg.SPIKE_Z_THRESH,
        method=cfg.SPIKE_IMPUTE_METHOD,
        known_spikes=known_ts,
    )

    # Exógena IPA-M
    u_exog = inputs[[ipam_col]].reindex(series.index).ffill().bfill().fillna(0.0)
    exog_cols = [ipam_col]

    # d por testes de raiz unitária na série transformada
    from src.preprocess import transform, unit_root_tests
    y_t = transform(series, mode_fix)
    urt = unit_root_tests(y_t)
    d   = urt["consensus_d"]

    # Baseline para DM: ARIMA do resultado mais recente
    baseline_results: dict = {}
    latest = _find_latest_result()
    if latest:
        with open(latest, encoding="utf-8") as _f:
            baseline_results = json.load(_f)
        print(f"[Round09] Baseline: {latest.name}")

    baseline_res = baseline_results.get(str(emp_code), {}).get(ind)
    arima_residuals = None
    if baseline_res:
        m_re = re.match(
            r"ARIMA\((\d+),\s*(\d+),\s*(\d+)\)",
            baseline_res.get("descricao_parametros", ""),
        )
        bp, bd, bq = (
            (int(m_re[1]), int(m_re[2]), int(m_re[3])) if m_re else (1, 1, 0)
        )
        bl_mode = baseline_res.get("preprocessing", "raw")
        try:
            afn = lambda yt, ut: ARIMAWrapper(
                d=bd, mode=bl_mode, fixed_order=(bp, bd, bq)
            ).fit(yt, ut)
            acv = walk_forward_cv(afn, series, None, min_train=cfg.MIN_TRAIN)
            arima_residuals = acv["residuals"]
        except Exception as exc:
            warnings.warn(f"[Round09] ARIMA baseline: {exc}")

    # Hammerstein-H1-Sigmoid com IPA-M + diff (replicar R06a + exog)
    fo = _select_nl_order(
        series, HammersteinWrapper,
        {"input_nl": "sigmoid", "n_knots": None},
        d, mode_fix,
    )
    mfn = _make_hammerstein_fn("sigmoid", None, d, mode_fix, fo)

    rw_cv = walk_forward_cv(
        lambda yt, ut: RandomWalkModel().fit(yt, ut), series, min_train=cfg.MIN_TRAIN
    )

    cv = walk_forward_cv(mfn, series, u_exog, min_train=cfg.MIN_TRAIN)
    entry = _make_result_dict(
        "HAMMER-H1-Sigmoid",
        f"HAMMERSTEIN(H1-Sigmoid,{mode_fix})+IPA-M",
        mode_fix, exog_cols, d, cv,
    )

    dm_stat, dm_p = 0.0, 1.0
    if arima_residuals is not None:
        try:
            dm_stat, dm_p = diebold_mariano_test(cv["residuals"], arima_residuals)
        except Exception:
            pass

    best_r = {k: v for k, v in entry.items() if not k.startswith("_")}
    best_r["dm_vs_arima"] = {"stat": float(dm_stat), "pvalue": float(dm_p)}
    best_r["in_model_confidence_set"] = True

    # Popula results com todos 12 combos do baseline; sobrescreve apenas ISAE4×res_bruto
    import copy
    results = copy.deepcopy(baseline_results) if baseline_results else {}
    results.setdefault(str(emp_code), {})[ind] = best_r

    print(
        f"\n{'='*80}\nRound 09 — ISAE4×resultado_bruto  Hammerstein-Sigmoid + IPA-M  [diff]\n{'='*80}"
    )
    print(
        f"FIT%={best_r['percentual_fit']:.1f}  U₂={best_r['theil_u2']:.3f}  "
        f"DM-p={dm_p:.3f}  LB-p={best_r['residual_diagnostics']['ljung_box_q4_pvalue']:.3f}"
    )
    delta_str = ""
    if baseline_res:
        delta = best_r["percentual_fit"] - baseline_res.get("percentual_fit", 0.0)
        delta_str = f"  Δ vs R08-ARIMA: {delta:+.1f} p.p."
    print(delta_str)

    json_path = save_results(results, version=9)

    _append_science_log_round(
        round_num=9,
        results=results,
        json_path=json_path,
        hypothesis=(
            "Round 09 — ISAE4×resultado_bruto: Hammerstein-H1-Sigmoid (f=sigmoid, diff) "
            "+ IPA-M como exógena. Replicação exata de R06a com adição de IPA-M. "
            "Hipótese: IPA-M captura pressão inflacionária nos insumos regulados da ISAE4, "
            "melhorando FIT% além dos 52.2% (R06a sem exog)."
        ),
        decisions=(
            f"Experimento focado: único combo ISAE4×res_bruto. Pré-proc fixo='diff'; "
            f"NL fixo=H1-Sigmoid; exog=[IPA-M]. Ordem ARIMA congelada na janela inicial. "
            f"Exog de '{exog_path.name}'. DM bilateral vs ARIMA R08 baseline."
        ),
        next_round=(
            "Round 10 — se IPA-M não trouxer ganho significativo, avaliar IPCA ou "
            "declarar ISAE4×res_bruto como teto alcançado com H6a (52.2%, DM-p=0.012*)."
        ),
        baseline_results=baseline_results if baseline_results else None,
    )
    print(f"\nScience log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


def _append_science_log_round02(results: dict, json_path: Path) -> None:
    _append_science_log_round(
        round_num=2,
        results=results,
        json_path=json_path,
        hypothesis=(
            "Modelos completos (SARIMA, ARMAX, ARIMAX, OE, BJ, SS, NARX, NARIMAX) "
            "+ exógenas selecionadas (VIF+Granger+AIC)."
        ),
        decisions="Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA baseline.",
        next_round=(
            "Round 03 — para 4 combos com ΔFIT=0 (ELET6×res_bruto, ELET6×desp_oper, "
            "ISAE4×desp_oper, ALUP4×desp_oper): relaxar Granger α=0.20 para ELET6 e "
            "expandir grid ARMA p,q∈[0,6]."
        ),
    )


# ---------------------------------------------------------------------------
# Fase 3 — Full
# ---------------------------------------------------------------------------

def run_full() -> None:
    """Fase 3/4: Feature selection + todos os modelos candidatos."""
    cfg.apply_global_seed()

    round_num = cfg.ROUND_VERSION + 1  # 2 → Round 03, etc.
    granger_override = getattr(cfg, "GRANGER_ALPHA_OVERRIDE", {})

    # ── Feature selection ─────────────────────────────────────────────────
    # Versionar o exog_selection por rodada quando há override de Granger
    if granger_override and cfg.ROUND_VERSION >= 2:
        exog_path = paths.RESULTS_DIR / f"exog_selection_v{round_num}.json"
    else:
        exog_path = paths.RESULTS_DIR / "exog_selection.json"

    if not exog_path.exists():
        print(f"[run_full] Executando feature selection (α-override: {granger_override})...")
        from src.feature_selection import run_feature_selection
        run_feature_selection(
            granger_alpha_override=granger_override,
            output_path=exog_path,
        )

    with open(exog_path, encoding="utf-8") as f:
        exog_sel = json.load(f)

    # ── Baseline results ──────────────────────────────────────────────────
    baseline_results: dict = {}
    latest = _find_latest_result()
    if latest:
        with open(latest, encoding="utf-8") as f:
            baseline_results = json.load(f)
        print(f"[run_full] Baseline: {latest.name}")
    else:
        print("[run_full] Aviso: nenhum resultado baseline encontrado; DM vs ARIMA omitido.")

    # ── Dados ─────────────────────────────────────────────────────────────
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    results: dict = {}
    spike_log: list[dict] = []
    col_w = 95
    print("=" * col_w)
    print(
        f"{'Empresa':<8} {'Indicador':<22} {'Melhor':<9} {'FIT%':>6} "
        f"{'U2':>6} {'DM-p':>5}  Parametros"
    )
    print("-" * col_w)

    for emp_code, ticker in cfg.EMPRESAS.items():
        results[str(emp_code)] = {}
        for ind in cfg.INDICADORES:
            series = datasets[ind][str(emp_code)].dropna()

            # ── Imputação de spikes (antes de qualquer transformação) ────
            _spike_known_ts = [
                pd.Timestamp(ts)
                for ts in cfg.known_spikes_for(emp_code, ind)
            ]
            _spike_z = getattr(cfg, "SPIKE_Z_THRESH", None)
            if _spike_known_ts or _spike_z:
                series, _recs = detect_and_impute_spikes(
                    series,
                    z_thresh=_spike_z,
                    method=getattr(cfg, "SPIKE_IMPUTE_METHOD", "linear_interp"),
                    known_spikes=_spike_known_ts,
                )
                for rec in _recs:
                    spike_log.append({
                        "empresa": ticker,
                        "indicador": ind,
                        **rec,
                    })

            bl = baseline_results.get(str(emp_code), {}).get(ind, {})
            bl_mode = bl.get("preprocessing", "raw")
            bl_d    = int(bl.get("unit_root_d", 1))

            # Exógenas selecionadas
            sel_cols = exog_sel.get(str(emp_code), {}).get(ind, [])
            u_exog: "pd.DataFrame | None" = None  # type: ignore[name-defined]
            if sel_cols:
                u_exog = inputs[sel_cols].reindex(series.index).ffill().bfill()

            best_r = _run_all_models_for_combo(
                series, u_exog, bl_mode, bl_d, sel_cols,
                ticker, ind, bl if bl else None,
            )
            results[str(emp_code)][ind] = best_r

            dm_p = best_r.get("dm_vs_arima", {}).get("pvalue", 1.0)
            print(
                f"{ticker:<8} {ind:<22} {best_r['modelo_utilizado']:<9} "
                f"{best_r['percentual_fit']:>6.1f} {best_r['theil_u2']:>6.3f} "
                f"{dm_p:>5.3f}  {best_r['descricao_parametros']}"
            )

    print("=" * col_w)

    # Formatar sumário de imputações para o science_log
    spike_summary = ""
    if spike_log:
        lines = ["**Imputações de spike (linear_interp):**"]
        for rec in spike_log:
            ts_str = rec["timestamp"].strftime("%Y-%m-%d")
            lines.append(
                f"  - {rec['empresa']}×{_IND_SHORT[rec['indicador']]}: "
                f"{ts_str} {rec['original']/1e9:.3f}B → {rec['imputed']/1e9:.3f}B "
                f"(z={rec['z_score']})"
            )
        spike_summary = "\n".join(lines) + "\n"

    json_path = save_results(results, version=cfg.ROUND_VERSION + 1)
    _append_science_log_round(
        round_num=round_num,
        results=results,
        json_path=json_path,
        hypothesis=(
            f"Round {round_num:02d} — imputação de spikes contábeis não-recorrentes "
            f"(ISAE4×luc_consol 2016-Q3 z=6.60; ELET6×luc_consol 2016-Q2 z=3.01) "
            f"+ re-execução completa dos modelos."
        ),
        decisions=(
            spike_summary
            + "Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA baseline. "
            f"Exógenas de '{exog_path.name}'."
        ),
        next_round=(
            f"Round {round_num + 1:02d} — avaliar impacto da imputação; "
            "documentar aceitação formal de não-preditibilidade para combos com FIT%<0 persistente."
        ),
        baseline_results=baseline_results if baseline_results else None,
    )
    print(f"\nScience log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


# ---------------------------------------------------------------------------
# Spec-driven rounds — `python main.py --round spec rounds/specs/round_NN.yaml`
# ---------------------------------------------------------------------------

_NL_FAMILY_CONFIGS = {
    "WIENER": "WIENER_CONFIGS",
    "HAMMERSTEIN": "HAMMERSTEIN_CONFIGS",
    "HW": "HW_CONFIGS",
}
_NL_FAMILY_MAKE_FN = {
    "WIENER": "_make_wiener_fn",
    "HAMMERSTEIN": "_make_hammerstein_fn",
    "HW": "_make_hw_fn",
}


def _resolve_spec_combos(spec: dict) -> list[tuple[int, str]]:
    raw = spec.get("combos", "all")
    if raw == "all":
        return list(cfg.COMBOS)
    combos: list[tuple[int, str]] = []
    for emp_code, ind in raw:
        emp_code = int(emp_code)
        if emp_code not in cfg.EMPRESAS:
            raise ValueError(f"Spec: código de empresa desconhecido: {emp_code}")
        if ind not in cfg.INDICADORES:
            raise ValueError(f"Spec: indicador desconhecido: {ind!r}")
        combos.append((emp_code, ind))
    return combos


def _resolve_spec_exog(spec: dict, combo_list: list[tuple[int, str]], round_num: int):
    """Retorna (use_exog, exog_path) a partir do campo `exog:` da spec.

    Formatos aceitos:
      exog: none                                  → sem exógenas
      exog: {source: outputs/results/exog_selection_vN.json}  → arquivo já existente
      exog: [col1, col2, ...]                      → materializa um JSON temporário
             restrito às combinações da spec (mesmo formato de exog_selection*.json)
    """
    raw = spec.get("exog")
    if not raw or raw == "none":
        return False, None
    if isinstance(raw, dict) and "source" in raw:
        path = paths.resolve(raw["source"])
        if not path.exists():
            raise FileNotFoundError(f"Spec: exog.source não encontrado: {path}")
        return True, path
    if isinstance(raw, list):
        materialized = {
            str(emp): {ind: list(raw) if ind == c_ind else []}
            for emp, c_ind in combo_list
            for ind in [c_ind]
        }
        paths.ensure_output_dirs()
        out_path = paths.RESULTS_DIR / f"exog_selection_spec_round{round_num:02d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(materialized, f, ensure_ascii=False, indent=1)
        return True, out_path
    raise ValueError(f"Spec: formato de 'exog' não reconhecido: {raw!r}")


def _run_spec_full(
    round_num: int,
    combo_list: list[tuple[int, str]],
    hypothesis: str,
    next_round: str,
    version: int,
    use_exog: bool,
    exog_path: "Path | None",
    fixed_mode: "str | None",
    baseline_path: "Path | None",
) -> None:
    """Branch `models: full` — testa todos os cfg.MODELOS candidatos por combo."""
    import copy as _copy

    cfg.apply_global_seed()
    rb, lc, do, inputs = load_all()
    datasets = {
        "resultado_bruto":       rb,
        "lucro_consolidado":     lc,
        "despesas_operacionais": do,
    }

    exog_sel: dict = {}
    if use_exog and exog_path is not None and exog_path.exists():
        with open(exog_path, encoding="utf-8") as f:
            exog_sel = json.load(f)

    baseline_results: dict = {}
    latest = baseline_path if baseline_path is not None else _find_latest_result()
    if latest:
        with open(latest, encoding="utf-8") as f:
            baseline_results = json.load(f)
        print(f"[spec/full] Baseline: {latest.name}")
    elif len(combo_list) < len(cfg.COMBOS):
        raise ValueError(
            "Spec: 'combos' é um subconjunto mas nenhum baseline_json foi encontrado — "
            "não é possível produzir um outputs/results/*.json schema-válido (todas as 12 "
            "combinações são obrigatórias). Informe 'baseline_json' na spec."
        )

    results: dict = _copy.deepcopy(baseline_results) if baseline_results else {}
    for emp in cfg.EMPRESAS:
        results.setdefault(str(emp), {})

    spike_log: list[dict] = []
    print(f"{'Empresa':<8} {'Indicador':<22} {'Melhor':<9} {'FIT%':>6} {'U2':>6}  Parametros")
    print("-" * 95)

    for emp_code, ind in combo_list:
        ticker = cfg.EMPRESAS[emp_code]
        results.setdefault(str(emp_code), {})
        series = datasets[ind][str(emp_code)].dropna()

        known_ts = [pd.Timestamp(t) for t in cfg.known_spikes_for(emp_code, ind)]
        if known_ts or cfg.SPIKE_Z_THRESH:
            series, recs = detect_and_impute_spikes(
                series, z_thresh=cfg.SPIKE_Z_THRESH,
                method=cfg.SPIKE_IMPUTE_METHOD, known_spikes=known_ts,
            )
            for rec in recs:
                spike_log.append({"empresa": ticker, "indicador": ind, **rec})

        sel_cols = exog_sel.get(str(emp_code), {}).get(ind, []) if use_exog else []
        u_exog = inputs[sel_cols].reindex(series.index).ffill().bfill() if sel_cols else None
        bl = baseline_results.get(str(emp_code), {}).get(ind, {})

        modes_to_try = [fixed_mode] if fixed_mode else cfg.PREPROC_MODES
        best_r = None
        for mode in modes_to_try:
            try:
                lam = guerrero_lambda(series) if mode == "boxcox" else None
                y_t = transform(series, mode, lam)
                d = unit_root_tests(y_t)["consensus_d"]
                candidate = _run_all_models_for_combo(
                    series, u_exog, mode, d, sel_cols, ticker, ind, bl if bl else None,
                    boxcox_lambda=lam,
                )
            except Exception as exc:
                warnings.warn(f"[{ticker}×{ind}] mode={mode}: {exc}")
                continue
            if best_r is None:
                best_r = candidate
            elif candidate["theil_u2"] < 1.0 and (
                best_r["theil_u2"] >= 1.0 or candidate["percentual_fit"] > best_r["percentual_fit"]
            ):
                best_r = candidate

        if best_r is None:
            warnings.warn(f"[{ticker}×{ind}] nenhum modo produziu resultado; combo ignorado.")
            continue

        results[str(emp_code)][ind] = best_r
        dm_p = best_r.get("dm_vs_arima", {}).get("pvalue", 1.0)
        print(f"{ticker:<8} {ind:<22} {best_r['modelo_utilizado']:<9} "
              f"{best_r['percentual_fit']:>6.1f} {best_r['theil_u2']:>6.3f}  "
              f"{best_r['descricao_parametros']}")

    spike_summary = ""
    if spike_log:
        lines = ["**Imputações de spike (linear_interp):**"]
        for rec in spike_log:
            lines.append(
                f"  - {rec['empresa']}×{_IND_SHORT[rec['indicador']]}: "
                f"{rec['timestamp'].strftime('%Y-%m-%d')} "
                f"{rec['original']/1e9:.3f}B → {rec['imputed']/1e9:.3f}B (z={rec['z_score']})"
            )
        spike_summary = "\n".join(lines) + "\n"

    json_path = save_results(results, version=version)
    _append_science_log_round(
        round_num=round_num,
        results=results,
        json_path=json_path,
        hypothesis=hypothesis,
        decisions=(
            spike_summary
            + f"Combos: {', '.join(f'{cfg.EMPRESAS[e]}×{i}' for e, i in combo_list)}. "
            + ("Exógenas: " + str(exog_path.name if exog_path else None) + ". " if use_exog else "")
            + "Seleção por FIT% entre candidatos com U₂<1; DM bilateral vs ARIMA baseline."
        ),
        next_round=next_round,
        baseline_results=baseline_results if baseline_results else None,
    )
    print(f"\nJSON salvo: {json_path}")
    print(f"Science log atualizado: {_ROUNDS_DIR / 'science_log.md'}")


# Whitelist de constantes de config.py que uma spec pode sobrescrever via `hyperparams:`.
# Mantido explícito (em vez de setattr genérico) para que uma spec nunca possa mudar um
# invariante do painel por engano (ex.: MIN_TRAIN, RANDOM_SEED) — só os hiperparâmetros de
# grid/convergência dentro da ordem já validada por combo (Round 18; protocolo metodológico §23).
# Definido em config.py (fonte única de verdade) — `plot_best_results.py` reaplica os mesmos
# hiperparâmetros ao reproduzir o campeão de um round.
_HYPERPARAM_WHITELIST = cfg.HYPERPARAM_WHITELIST


def _apply_spec_hyperparams(hyperparams: dict | None) -> None:
    if not hyperparams:
        return
    unknown = set(hyperparams) - set(_HYPERPARAM_WHITELIST)
    if unknown:
        raise ValueError(
            f"Spec: hyperparams desconhecidos: {unknown} "
            f"(aceitos: {sorted(_HYPERPARAM_WHITELIST)})"
        )
    for key, value in hyperparams.items():
        attr = _HYPERPARAM_WHITELIST[key]
        old = getattr(cfg, attr)
        setattr(cfg, attr, value)
        print(f"[spec/hyperparams] cfg.{attr}: {old} -> {value}")


# `spike_revert:` — Round 25 (protocolo metodológico §27). Reverte, só para esta rodada, o tratamento
# de outlier de datas específicas por combinação, escrevendo em `cfg.SPIKE_REVERTED` (lido por
# `cfg.known_spikes_for()`, o único ponto de leitura de SPIKE_KNOWN no pipeline). Chave própria
# no topo da spec, fora de `_HYPERPARAM_WHITELIST`, pelo mesmo motivo de `experimental_window:`:
# mexer no tratamento dos dados é uma decisão de protocolo, nunca um ajuste acidental de grid.
# A data precisa estar de fato em SPIKE_KNOWN — reverter algo não tratado é erro de spec, não
# no-op silencioso.
def _apply_spike_revert(spec: dict) -> None:
    raw = spec.get("spike_revert")
    if not raw:
        return
    for entry in raw:
        emp_code, ind, dates = int(entry["empresa"]), entry["indicador"], list(entry["dates"])
        if (emp_code, ind) not in cfg.COMBOS:
            raise ValueError(f"Spec/spike_revert: combo inexistente: ({emp_code}, {ind})")
        current = cfg.SPIKE_KNOWN.get((emp_code, ind), [])
        missing = [d for d in dates if d not in current]
        if missing:
            raise ValueError(
                f"Spec/spike_revert: {cfg.EMPRESAS[emp_code]}×{ind} — datas ausentes de "
                f"SPIKE_KNOWN: {missing} (tratadas hoje: {current})"
            )
        cfg.SPIKE_REVERTED[(emp_code, ind)] = dates
        print(
            f"[spec/spike_revert] {cfg.EMPRESAS[emp_code]}×{ind}: revertendo {dates} "
            f"-> SPIKE_KNOWN efetivo {cfg.known_spikes_for(emp_code, ind)}"
        )


# `experimental_window:` — escotilha DELIBERADAMENTE separada de `hyperparams:`/
# `_HYPERPARAM_WHITELIST` (Round 22; ver rounds/specs/ e protocolo metodológico §24).
# `hyperparams:` protege contra mudar `MIN_TRAIN` por engano; isso aqui é o oposto — uma
# mudança de MIN_TRAIN sempre deliberada, sempre visível (chave própria no topo da spec, não
# escondida num dict genérico) e sempre presa a `comparable: false`, que a rodada é obrigada a
# declarar explicitamente. Nunca sobrescreve `cfg.MIN_TRAIN` (nada de setattr aqui) — passa
# `min_train` como parâmetro explícito para o orquestrador, que grava o resultado fora de
# `outputs/results/` (invisível para `compute_champions()`).
def _apply_experimental_window(spec: dict) -> tuple[int | None, "Path | None", bool]:
    raw = spec.get("experimental_window")
    if not raw:
        return None, None, True
    if "min_train" not in raw:
        raise ValueError("Spec: experimental_window requer 'min_train'")
    if raw.get("comparable") is not False:
        raise ValueError(
            "Spec: experimental_window requer 'comparable: false' explícito — "
            "esta escotilha existe só para rodadas deliberadamente não-comparáveis ao "
            "painel (min_train != cfg.MIN_TRAIN). Ver protocolo metodológico §24."
        )
    min_train_v = int(raw["min_train"])
    print(
        f"[spec/experimental_window] min_train={min_train_v} "
        f"(painel usa cfg.MIN_TRAIN={cfg.MIN_TRAIN}) — rodada NÃO-COMPARÁVEL, "
        f"resultado vai para outputs/results/experimental/"
    )
    return min_train_v, paths.RESULTS_EXPERIMENTAL_DIR, False


# `window_sweep:` — Round 24; generaliza `experimental_window:` de um único MIN_TRAIN
# isolado para uma curva sobre vários valores, avaliada num intervalo OOS comum
# (ver protocolo metodológico §26). Mesma exigência de `comparable: false` explícito e mesmo
# isolamento físico em `results/experimental/` — nunca `cfg.MIN_TRAIN` via setattr.
def _apply_window_sweep(spec: dict) -> "tuple[list[int], int] | None":
    raw = spec.get("window_sweep")
    if not raw:
        return None
    if "values" not in raw or "eval_start" not in raw:
        raise ValueError("Spec: window_sweep requer 'values' e 'eval_start'")
    if raw.get("comparable") is not False:
        raise ValueError(
            "Spec: window_sweep requer 'comparable: false' explícito — esta escotilha existe "
            "só para rodadas diagnósticas, não-comparáveis ao painel. Ver protocolo metodológico §26."
        )
    values = sorted(int(v) for v in raw["values"])
    eval_start = int(raw["eval_start"])
    print(
        f"[spec/window_sweep] values={values} eval_start={eval_start} "
        f"(painel usa cfg.MIN_TRAIN={cfg.MIN_TRAIN}) — rodada NÃO-COMPARÁVEL, "
        f"resultado vai para outputs/results/experimental/"
    )
    return values, eval_start


def run_spec(spec_path: str) -> None:
    """Executa um round a partir de uma spec YAML — ver rounds/specs/.

    Reusa os orquestradores existentes (`_run_nl_round`, `_run_all_models_for_combo`,
    `_append_science_log_round`) em vez de duplicar a lógica: uma nova rodada passa a
    ser um arquivo YAML de ~25 linhas, não uma nova função Python de ~150 linhas.

    Suporta exatamente uma família de modelo por rodada (`full` ou uma de
    WIENER/HAMMERSTEIN/HW), espelhando a estrutura dos Rounds 01-09. Rodadas que
    combinam múltiplas famílias num único round ainda exigem código bespoke.

    `hyperparams:` (opcional, Round 18+) sobrescreve um conjunto fixo de constantes de
    `config.py` para a duração deste processo — cada round é um processo `python main.py`
    isolado, então não há necessidade de restaurar o valor original depois. Ver
    o formato das specs em rounds/specs/ para a whitelist de chaves aceitas.
    """
    import yaml

    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    _apply_spec_hyperparams(spec.get("hyperparams"))
    _apply_spike_revert(spec)
    min_train_v, experimental_dir, comparable = _apply_experimental_window(spec)

    round_num = int(spec["round"])
    hypothesis = spec["hypothesis"].strip()
    justification = spec.get("justification")
    if justification:
        hypothesis = f"{hypothesis}\n**Fonte/citação:** {justification}"
    next_round = spec.get("next_round", "")
    version = int(spec.get("version", round_num))

    combo_list = _resolve_spec_combos(spec)
    baseline_path = paths.resolve(spec["baseline_json"]) if spec.get("baseline_json") else None
    if baseline_path and not baseline_path.exists():
        raise FileNotFoundError(f"Spec: baseline_json não encontrado: {baseline_path}")

    preproc = spec.get("preproc", "search")
    fixed_mode = None if preproc == "search" else preproc

    use_exog, exog_path = _resolve_spec_exog(spec, combo_list, round_num)

    models = spec["models"]
    if isinstance(models, str):
        models = [models]

    if models == ["full"]:
        _run_spec_full(
            round_num=round_num, combo_list=combo_list, hypothesis=hypothesis,
            next_round=next_round, version=version, use_exog=use_exog,
            exog_path=exog_path, fixed_mode=fixed_mode, baseline_path=baseline_path,
        )
        return

    if models == ["SETAR"]:
        preproc_modes = [fixed_mode] if fixed_mode else None
        _run_setar_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combos=combo_list, preproc_modes=preproc_modes,
            baseline_path=baseline_path, min_train=min_train_v,
            output_dir=experimental_dir, comparable=comparable,
        )
        return

    if models == ["WINDOW_SWEEP"]:
        window_sweep = _apply_window_sweep(spec)
        if window_sweep is None:
            raise ValueError("Spec: models=[WINDOW_SWEEP] requer o bloco 'window_sweep:'")
        window_values, eval_start = window_sweep
        _run_window_sweep_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, window_values=window_values, eval_start=eval_start,
            combo_list=combo_list, use_exog=use_exog, exog_path=exog_path,
            fixed_mode=fixed_mode,
        )
        return

    if models == ["RECURSIVE"]:
        rcfg = spec.get("recursive") or {}
        unknown = set(rcfg) - {"forgetting", "frozen_tau_batch", "resets"}
        if unknown:
            raise ValueError(f"Spec: chaves desconhecidas em 'recursive:': {unknown}")
        _run_recursive_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combo_list=combo_list,
            forgetting=rcfg.get("forgetting"),
            frozen_tau_batch=bool(rcfg.get("frozen_tau_batch", False)),
            resets=bool(rcfg.get("resets", False)),
        )
        return

    if models == ["LAGGED_EXOG"]:
        lcfg = spec.get("lagged_exog") or {}
        unknown = set(lcfg) - {"lag"}
        if unknown:
            raise ValueError(f"Spec: chaves desconhecidas em 'lagged_exog:': {unknown}")
        _run_lagged_exog_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combo_list=combo_list, lag=lcfg.get("lag"),
        )
        return

    if models == ["MANUSCRIPT_TREATMENT"]:
        _run_manuscript_treatment_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combo_list=combo_list,
        )
        return

    if models == ["MANUSCRIPT_SEARCH"]:
        _run_manuscript_search_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combo_list=combo_list,
        )
        return

    if models == ["FIXED_RULE_TREATMENT"]:
        _run_fixed_rule_round(
            round_num=round_num, version=version, hypothesis=hypothesis,
            next_round=next_round, combo_list=combo_list,
        )
        return

    if len(models) == 1 and models[0] in _NL_FAMILY_CONFIGS:
        family = models[0]
        all_configs = getattr(cfg, _NL_FAMILY_CONFIGS[family])
        nl_names = spec.get("nl_configs")
        configs = (
            [c for c in all_configs if c["name"] in nl_names] if nl_names else all_configs
        )
        if nl_names and len(configs) != len(nl_names):
            missing = set(nl_names) - {c["name"] for c in configs}
            raise ValueError(f"Spec: nl_configs desconhecidos para {family}: {missing}")
        make_fn = globals()[_NL_FAMILY_MAKE_FN[family]]
        preproc_modes = [fixed_mode] if fixed_mode else None
        _run_nl_round(
            round_num=round_num, model_family=family, configs=configs, make_fn=make_fn,
            version=version, hypothesis=hypothesis, next_round=next_round,
            use_exog=use_exog, exog_path=exog_path, combos=combo_list,
            preproc_modes=preproc_modes, baseline_path=baseline_path,
        )
        return

    raise NotImplementedError(
        f"Spec: 'models' deve ser 'full', 'SETAR' ou uma família NL única "
        f"({list(_NL_FAMILY_CONFIGS)}) — recebido: {models}"
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare_runs(path1: str, path2: str) -> None:
    with open(path1, encoding="utf-8") as f:
        r1 = json.load(f)
    with open(path2, encoding="utf-8") as f:
        r2 = json.load(f)

    hdr = f"{'Combinação':<32} {'FIT_v1':>8} {'FIT_v2':>8} {'ΔFIT':>8} {'U₂_v1':>7} {'U₂_v2':>7}"
    print(hdr)
    print("-" * len(hdr))
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            k = str(emp_code)
            if k in r1 and k in r2 and ind in r1[k] and ind in r2[k]:
                v1, v2 = r1[k][ind], r2[k][ind]
                delta = v2["percentual_fit"] - v1["percentual_fit"]
                combo = f"{ticker}×{ind[:12]}"
                print(
                    f"{combo:<32} {v1['percentual_fit']:>8.1f} {v2['percentual_fit']:>8.1f} "
                    f"{delta:>+8.1f} {v1['theil_u2']:>7.3f} {v2['theil_u2']:>7.3f}"
                )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(json_path: str, output: str | None = None) -> None:
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    log_text = ""
    log_path = _ROUNDS_DIR / "science_log.md"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        log_text = "\n".join(lines[-50:] if len(lines) > 50 else lines)

    rows = [
        "| Empresa | Indicador | Modelo | Pré-proc | FIT% | RMSE | U₂ | LB-p | DM-p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for emp_code, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            r = results.get(str(emp_code), {}).get(ind, {})
            if r:
                dm_p = r.get("dm_vs_arima", {}).get("pvalue", "N/A")
                rows.append(
                    f"| {ticker} | {ind} | {r['modelo_utilizado']} | {r['preprocessing']} | "
                    f"{r['percentual_fit']:.1f} | {r['rmse']:.2e} | {r['theil_u2']:.3f} | "
                    f"{r['residual_diagnostics']['ljung_box_q4_pvalue']:.3f} | "
                    f"{dm_p if isinstance(dm_p, str) else f'{dm_p:.3f}'} |"
                )

    report = "\n".join([
        "# Relatório de Identificação de Sistemas",
        f"**Fonte:** `{json_path}`",
        f"**Gerado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Resultados",
        "",
        *rows,
        "",
        "## Science Log (extrato)",
        "",
        log_text,
    ])

    if output:
        Path(output).write_text(report, encoding="utf-8")
        print(f"Relatório salvo: {output}")
    else:
        print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == "--round" and len(args) > 1 and args[1] == "spec":
        if len(args) < 3:
            print("Uso: python main.py --round spec rounds/specs/round_NN.yaml")
            return
        run_spec(args[2])
    elif cmd == "--round":
        mode = args[1] if len(args) > 1 else "baseline"
        dispatch = {
            "baseline":        run_baseline,
            "full":            run_full,
            "wiener":          lambda: run_wiener_round(use_exog=False),
            "wiener-arx":      lambda: run_wiener_round(use_exog=True),
            "hammerstein":     lambda: run_hammerstein_round(use_exog=False),
            "hammerstein-arx": lambda: run_hammerstein_round(use_exog=True),
            "hw":              lambda: run_hw_round(use_exog=False),
            "hw-arx":          lambda: run_hw_round(use_exog=True),
            "ls-dummies":      run_round08,
            "hammerstein-ipam": run_round09,
        }
        dispatch.get(mode, lambda: print(f"Modo desconhecido: {mode}"))()
    elif cmd == "--compare" and len(args) >= 3:
        compare_runs(args[1], args[2])
    elif cmd == "--report" and len(args) >= 2:
        out = args[args.index("--output") + 1] if "--output" in args else None
        generate_report(args[1], out)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
