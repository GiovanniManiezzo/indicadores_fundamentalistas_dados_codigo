"""Robustez da metade HMM: tamanho de janela e sensibilidade à inicialização do EM.

Análises de robustez que não couberam no artigo do congresso (limite de 8 páginas), onde
aparecem só como frase qualitativa na seção de limitações.

Diferença em relação à versão original: lá as duas propostas eram medidas por NRMSE. Sob a
divisão de papéis travada, previsão é da metade de identificação de sistemas, então aqui elas são
medidas pelo que pertence ao HMM — **convergência** e **estabilidade da partição de estados**:

- janela: em quantas janelas cada variante ajusta e converge, e quanto a partição estimada na
  janela concorda com a partição da amostra completa (Rand ajustado);
- semente: dispersão da log-verossimilhança e concordância par-a-par entre as 10 partições.

O AR-HMM é o caso de interesse: com 35 trimestres de treino ele só convergia para a ISA Energy;
com K maior a conclusão pode mudar, e a mudança é resultado reportável.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from hmm_n64 import (
    N_STATES_MAIN,
    Combo,
    canonical_map,
    discretize,
    fit_arhmm,
    fit_categorical,
    fit_gaussian,
    fit_gmm,
    fit_variant,
    relabel,
)

JANELAS = (25, 30, 35, 40)
SEEDS = tuple(range(10))


def _padroniza(x: pd.Series) -> np.ndarray:
    return ((x - x.mean()) / x.std(ddof=0)).values.reshape(-1, 1)


def _estados_referencia(c: Combo, n_states_main: int) -> dict[str, np.ndarray]:
    """Partição de estados sobre a amostra completa — a referência da comparação."""
    ref: dict[str, np.ndarray] = {}
    codes, _ = discretize(c.x)
    z = _padroniza(c.x)

    m, _, _ = fit_categorical(codes, n_states_main)
    if m is not None:
        ref["categorical"] = relabel(
            m.decode(codes)[1], canonical_map(m.emissionprob_ @ np.arange(m.emissionprob_.shape[1]))
        )
    m, _, _ = fit_gaussian(z, n_states_main)
    if m is not None:
        ref["gaussian"] = relabel(m.decode(z)[1], canonical_map(m.means_.flatten()))
    m, _, _ = fit_gmm(z, n_states_main)
    if m is not None:
        medias = (m.means_[:, :, 0] * m.weights_).sum(axis=1)
        ref["gmm"] = relabel(m.decode(z)[1], canonical_map(medias))
    return ref


def _janela(combos: list[Combo], n_states_main: int) -> pd.DataFrame:
    linhas = []
    for c in combos:
        ref = _estados_referencia(c, n_states_main)
        for W in JANELAS:
            if W >= len(c.x):
                continue
            xw = c.x.iloc[:W]
            codes, _ = discretize(xw)
            zw = _padroniza(xw)

            for variante, (modelo, _, diag) in (
                ("categorical", fit_categorical(codes, n_states_main)),
                ("gaussian", fit_gaussian(zw, n_states_main)),
                ("gmm", fit_gmm(zw, n_states_main)),
            ):
                X = codes if variante == "categorical" else zw
                ok = modelo is not None
                ari = np.nan
                if ok and variante in ref:
                    ari = float(adjusted_rand_score(ref[variante][:W], modelo.decode(X)[1]))
                linhas.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "variante": variante,
                     "janela": W, "K_disponivel": len(c.x), "ajustou": ok,
                     "convergiu": bool(modelo.monitor_.converged) if ok else False,
                     "espurio": bool(diag["espurio"]) if ok else None,
                     "loglik": float(modelo.score(X)) if ok else np.nan,
                     "ARI_vs_amostra_completa": ari}
                )

            for n_estados in (2, 3):
                try:
                    res = fit_arhmm(_padroniza(xw).ravel(), n_estados)
                    linhas.append(
                        {"empresa": c.empresa, "indicador": c.indicador,
                         "variante": f"arhmm_N{n_estados}", "janela": W,
                         "K_disponivel": len(c.x), "ajustou": True, "convergiu": True,
                         "loglik": float(res.llf), "ARI_vs_amostra_completa": np.nan}
                    )
                except Exception:
                    linhas.append(
                        {"empresa": c.empresa, "indicador": c.indicador,
                         "variante": f"arhmm_N{n_estados}", "janela": W,
                         "K_disponivel": len(c.x), "ajustou": False, "convergiu": False,
                         "loglik": np.nan, "ARI_vs_amostra_completa": np.nan}
                    )
        print(f"  janela  {c.rotulo}")
    return pd.DataFrame(linhas)


def _sementes(combos: list[Combo], n_states_main: int) -> pd.DataFrame:
    linhas = []
    for c in combos:
        codes, _ = discretize(c.x)
        z = _padroniza(c.x)

        for variante in ("categorical", "gaussian", "gmm"):
            X = codes if variante == "categorical" else z
            particoes, logliks = [], []
            for seed in SEEDS:
                m, ll, _ = fit_variant(variante, X, n_states_main, seeds=[seed])
                if m is None:
                    continue
                particoes.append(m.decode(X)[1])
                logliks.append(ll)

            if len(particoes) < 2:
                continue
            aris = [adjusted_rand_score(a, b) for a, b in combinations(particoes, 2)]
            linhas.append(
                {"empresa": c.empresa, "indicador": c.indicador, "variante": variante,
                 "n_sementes_ok": len(particoes),
                 "loglik_media": float(np.mean(logliks)), "loglik_dp": float(np.std(logliks)),
                 "loglik_min": float(np.min(logliks)), "loglik_max": float(np.max(logliks)),
                 "ARI_par_a_par_medio": float(np.mean(aris)),
                 "ARI_par_a_par_min": float(np.min(aris)),
                 "particoes_identicas_pct": round(100 * float(np.mean(np.isclose(aris, 1.0))), 1)}
            )
        print(f"  semente {c.rotulo}")
    return pd.DataFrame(linhas)


def stage_robustness(combos: list[Combo], out_dir: Path, n_states_main: int = N_STATES_MAIN) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _janela(combos, n_states_main).to_csv(out_dir / "robustez_janela.csv", index=False)
    _sementes(combos, n_states_main).to_csv(out_dir / "robustez_semente.csv", index=False)
