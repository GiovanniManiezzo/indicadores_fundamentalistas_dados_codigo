"""Validação embutida: reproduz a janela do congresso com a receita publicada.

Recalcula na janela antiga (até 2024-Q3) e compara com as tabelas impressas no artigo do
congresso. O objetivo é duplo:

1. **Confirmar a atribuição empresa ↔ código CVM.** O artigo publicado imprimiu Alupar como
   CVM 2437 e Axia como 21490; o `config.py` da dissertação diz o contrário. Os notebooks operam
   por código e não têm dicionário código→nome — o rótulo foi aplicado na escrita. Se as taxas de
   sincronização calculadas por código baterem com as linhas publicadas, a atribuição fica
   decidida por evidência, não por memória.
2. **Identificar qual versão do dado o congresso usou** (a ALUP4 tinha `0.0` em 2010-06-30 e
   2011-06-30 na versão da dissertação; o congresso usou a série interpolada — Round 28).

Aqui a receita é a **publicada**, não o protocolo novo: `pct_change` sobre o nível, binning por
quantis com M=3, `CategoricalHMM(n_components=3, n_iter=100, random_state=1)`, ajuste único (sem
reinicializações) e comparação por rótulo **bruto** de estado — exatamente como as células 30, 31
e 36 de `2_2_Modelo_HHM.ipynb`. Reproduzir exige repetir as escolhas, inclusive as que o protocolo
novo abandona.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parents[2]
JANELA_CONGRESSO = "2024-09-30"

# Valores impressos no artigo (Tabelas 3 e 4). São o alvo da comparação, não entrada de cálculo.
# Chaves renomeadas 2026-09-27 (d): "Gross Income"/"Gross Profit" -> "Net Revenue"/"Net Income",
# em conjunto com LABELS abaixo (usado como chave em PUBLICADO_CROSS[LABELS[ind]] mais adiante) --
# o artigo do congresso continua usando os termos antigos, isto é só o rótulo interno de lookup.
PUBLICADO_CROSS = {"Net Revenue": 6.900, "Net Income": 10.345, "Operational Expenses": 13.793}
PUBLICADO_INTRA = {"Alupar": 22.414, "ISA Energy": 41.379, "Taesa": 27.586, "Axia Energy": 27.586}
# Tabela 6 — PMF do Categorical HMM de 3 estados, Net Revenue da Taesa.
PUBLICADO_PMF_TAESA = np.array([[0.000, 0.544, 0.456], [0.000, 0.122, 0.878], [0.886, 0.114, 0.000]])
# Tabela 5 — AR(1)-HMM de 3 estados, Net Revenue da ISA Energy.
PUBLICADO_AR_ISA = {"beta0": [0.167, -0.090, 0.022], "sigma2": [0.057, 0.023, 2.427],
                    "phi1": [0.223, -0.111, -0.748]}
# Bordas de bin citadas no texto da Seção "Categorical HMM results", atribuídas à ISA Energy.
PUBLICADO_BORDAS_ISA = [-0.505, -0.055, 0.076, 1.842]

INDICADORES = ["resultado_bruto", "lucro_consolidado", "despesas_operacionais"]
LABELS = {"resultado_bruto": "Net Revenue", "lucro_consolidado": "Net Income",
          "despesas_operacionais": "Operational Expenses"}


# --------------------------------------------------------------------------------------
# Receita publicada (célula 30 / 36 do notebook 2_2)
# --------------------------------------------------------------------------------------
def _estados_publicados(serie: pd.Series, n_estados: int = 3, n_bins: int = 3,
                        transform: str = "pct_change"):
    """Estados pela receita publicada. `transform` existe porque o artigo não usa a mesma nos
    dois lugares: as tabelas de sincronização vêm de `pct_change` (célula 30/36) e a legenda da
    PMF diz explicitamente "log-return series"."""
    limpa = serie.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if transform == "log_return":
        ret = np.log(limpa / limpa.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        ret = limpa.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
    codes = est.fit_transform(ret.values.reshape(-1, 1)).astype(int)

    model = hmm.CategoricalHMM(n_components=n_estados, n_iter=100, random_state=1)
    model.fit(codes)
    _, estados = model.decode(codes)
    return pd.Series(estados, index=ret.index), model, est.bin_edges_[0]


def _taxa(estados: dict[str, pd.Series]) -> tuple[float, int]:
    chaves = list(estados)
    idx = estados[chaves[0]].index
    for k in chaves[1:]:
        idx = idx.intersection(estados[k].index)
    if len(idx) == 0:
        return float("nan"), 0, 0
    m = pd.DataFrame({k: estados[k].loc[idx] for k in chaves})
    coinc = int(m.nunique(axis=1).eq(1).sum())
    return coinc / len(idx), int(len(idx)), coinc



def _sweep_pmf(serie: pd.Series, alvo: np.ndarray) -> dict:
    """Procura a configuração que gera a PMF publicada (Tabela 6).

    As tabelas de sincronização reproduzem exatamente com a receita das células 30/36, mas a PMF
    impressa não sai dessa configuração. A varredura existe para transformar "não reproduz" numa
    afirmação verificável: registra o espaço percorrido e o resultado, em vez de afirmar de
    memória. A comparação é feita a menos de permutação de regimes, porque o rótulo do estado é
    arbitrário no Baum-Welch.
    """
    from itertools import permutations

    espaco = {"transform": ("pct_change", "log_return"), "janela": (35, 40, 45, 50, 58),
              "n_bins": (2, 3, 4, 5), "strategy": ("quantile", "kmeans"), "seeds": (0, 40)}
    achados, avaliados = [], 0

    for transform in espaco["transform"]:
        limpa = serie.sort_index().astype(float).dropna()
        ret = (np.log(limpa / limpa.shift(1)) if transform == "log_return" else limpa.pct_change())
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        for W in espaco["janela"]:
            for n_bins in espaco["n_bins"]:
                for strategy in espaco["strategy"]:
                    try:
                        codes = KBinsDiscretizer(
                            n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None
                        ).fit_transform(ret.values[:W].reshape(-1, 1)).astype(int)
                    except Exception:
                        continue
                    for seed in range(espaco["seeds"][0], espaco["seeds"][1] + 1):
                        model = hmm.CategoricalHMM(n_components=3, n_iter=100, random_state=seed)
                        try:
                            model.fit(codes)
                        except Exception:
                            continue
                        avaliados += 1
                        E = model.emissionprob_
                        if E.shape != alvo.shape:
                            continue
                        for perm in permutations(range(alvo.shape[0])):
                            if np.allclose(E[list(perm)], alvo, atol=0.0015):
                                achados.append({"transform": transform, "janela": W,
                                                "n_bins": n_bins, "strategy": strategy,
                                                "seed": seed, "permutacao": list(perm)})
                                break
    return {"espaco_varrido": espaco, "ajustes_avaliados": avaliados, "reproduz_em": achados}



def _arhmm_convergencia(serie: pd.Series, janelas=(35, 45), estados=(2, 3)) -> list[dict]:
    """Testa a afirmação publicada de que o AR-HMM não converge com 35 trimestres de treino.

    O artigo reporta que, com janela de 35, só a ISA Energy convergiu, e que as demais exigiriam
    ao menos 45 trimestres. Um ajuste único sobre os mesmos 35 trimestres converge para as quatro
    empresas — o que desloca a explicação da escassez de dados para a fragilidade da
    **reestimação repetida**: o esquema rolling do artigo reajusta o modelo a cada passo, e basta
    um passo cair em `LinAlgError` para a execução inteira falhar.

    Mede-se, então, as duas coisas: o ajuste único na janela e a taxa de sucesso dos reajustes
    ao longo dos passos seguintes. Nenhuma previsão é avaliada — o que se conta é convergência.
    """
    limpa = serie.sort_index().astype(float).dropna()
    lr = np.log(limpa / limpa.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

    linhas = []
    for W in janelas:
        if W >= len(lr):
            continue
        for n_estados in estados:
            def _ajusta(x) -> bool:
                # `search_reps` sorteia os valores iniciais do RNG global do numpy; sem fixar a
                # semente, a contagem de falhas muda a cada execução e a afirmação deixa de ser
                # verificável.
                np.random.seed(42)
                try:
                    MarkovAutoregression(
                        x, k_regimes=n_estados, order=1, trend="c", switching_trend=True,
                        switching_ar=True, switching_variance=True,
                    ).fit(em_iter=500, search_reps=20, disp=False)
                    return True
                except Exception:
                    return False

            unico = _ajusta(lr.values[:W])
            passos = [_ajusta(lr.values[:t]) for t in range(W, len(lr))]
            linhas.append(
                {"janela": W, "n_estados": n_estados, "ajuste_unico_converge": unico,
                 "reajustes": len(passos), "reajustes_ok": int(sum(passos)),
                 "taxa_reajuste_ok": round(sum(passos) / len(passos), 3) if passos else None,
                 "rolling_completo_sem_falha": bool(unico and all(passos))}
            )
    return linhas


# --------------------------------------------------------------------------------------
# Variantes de dado
# --------------------------------------------------------------------------------------
def _carrega(indicador: str, variante: str, data_dir: Path) -> pd.DataFrame:
    if variante == "tratada":
        df = pd.read_csv(data_dir / f"{indicador}.csv", parse_dates=["DT_FIM_EXERC"])
    else:
        df = pd.read_csv(_ROOT / f"{indicador}.csv", parse_dates=["DT_FIM_EXERC"])
    df = df.set_index("DT_FIM_EXERC").sort_index()
    df.columns = [str(c) for c in df.columns]

    if variante == "bruta_alup4_interp":
        # O congresso rodou sobre a série interpolada; a dissertação tinha os zeros. Só a ALUP4
        # (21490) é afetada, e só nos dois trimestres de dado faltante do Round 28.
        col = df["21490"].replace(0.0, np.nan)
        df["21490"] = col.interpolate(method="linear")

    return df.loc[:JANELA_CONGRESSO]


def _mapa_empresas(data_dir: Path) -> dict[str, str]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["empresas"]


# --------------------------------------------------------------------------------------
# Estágio
# --------------------------------------------------------------------------------------
def stage_validation(data_dir: Path, out_dir: Path) -> None:
    empresas = _mapa_empresas(data_dir)
    codigos = list(empresas)
    linhas_cross, linhas_intra = [], []

    for variante in ("bruta", "bruta_alup4_interp", "tratada"):
        dados = {ind: _carrega(ind, variante, data_dir) for ind in INDICADORES}
        n_obs = len(next(iter(dados.values())))
        estados: dict[tuple[str, str], pd.Series] = {}

        for ind, df in dados.items():
            for cod in codigos:
                estados[(cod, ind)], _, _ = _estados_publicados(df[cod])

        for ind in INDICADORES:
            taxa, K, coinc = _taxa({c: estados[(c, ind)] for c in codigos})
            alvo = PUBLICADO_CROSS[LABELS[ind]]
            linhas_cross.append(
                {"variante_dado": variante, "n_obs_nivel": n_obs, "indicador": ind,
                 "indicator_label": LABELS[ind], "K": K, "coincidencias": coinc,
                 "taxa_pct": round(100 * taxa, 3), "publicado_pct": alvo,
                 # o artigo imprime 6.900 para 4/58 = 6.897: tolerância cobre o arredondamento
                 "bate": abs(100 * taxa - alvo) < 0.005}
            )

        for cod in codigos:
            taxa, K, coinc = _taxa({i: estados[(cod, i)] for i in INDICADORES})
            pct = round(100 * taxa, 3)
            casa_com = [nome for nome, v in PUBLICADO_INTRA.items() if abs(pct - v) < 0.005]
            linhas_intra.append(
                {"variante_dado": variante, "codigo_cvm": cod, "empresa_config": empresas[cod],
                 "K": K, "coincidencias": coinc, "taxa_pct": pct,
                 "publicado_casa_com": "|".join(casa_com) or "nenhuma"}
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    df_cross = pd.DataFrame(linhas_cross)
    df_intra = pd.DataFrame(linhas_intra)
    df_cross.to_csv(out_dir / "validacao_sync_cross.csv", index=False)
    df_intra.to_csv(out_dir / "validacao_sync_intra.csv", index=False)

    # --- Tabela 6: PMF do Gross Income da Taesa (20257) ----------------------------------
    # A legenda publicada diz "log-return series", mas as tabelas de sincronização vieram de
    # pct_change: as duas são calculadas para identificar qual gerou a tabela impressa.
    df = _carrega("resultado_bruto", "bruta_alup4_interp", data_dir)
    pmfs = {}
    for t in ("pct_change", "log_return"):
        _, modelo, _ = _estados_publicados(df["20257"], transform=t)
        pmfs[t] = modelo.emissionprob_
        pd.DataFrame(np.round(modelo.emissionprob_, 3), columns=["Low", "Stable", "High"]).to_csv(
            out_dir / f"validacao_pmf_taesa_{t}.csv", index_label="regime"
        )

    # O texto do artigo atribui as bordas [-0.505, -0.055, 0.076, 1.842] ao Gross Income da ISA
    # Energy. Em vez de conferir só a ISA, procura-se a qual código elas de fato pertencem: se
    # caírem noutro código, a atribuição impressa está errada — e isso precisa ser sabido antes
    # de a versão estendida repetir a frase.
    bordas = {}
    for codigo in empresas:
        for t in ("pct_change", "log_return"):
            bordas[f"{codigo}:{t}"] = _estados_publicados(df[codigo], transform=t)[2].round(3).tolist()
    dono_das_bordas = [k for k, v in bordas.items()
                       if np.allclose(v, PUBLICADO_BORDAS_ISA, atol=0.002)]

    sweep = _sweep_pmf(df["20257"], PUBLICADO_PMF_TAESA)

    # Afirmação publicada: AR-HMM só converge para a ISA Energy com 35 trimestres de treino.
    conv = []
    for cod, nome in empresas.items():
        for linha in _arhmm_convergencia(df[cod]):
            conv.append({"codigo_cvm": cod, "empresa_config": nome, **linha})
    df_conv = pd.DataFrame(conv)
    df_conv.to_csv(out_dir / "validacao_arhmm_convergencia.csv", index=False)

    # --- Tabela 5: AR(1)-HMM do Gross Income da ISA Energy (18376), log-retorno ----------
    serie = df["18376"].astype(float)
    lr = np.log(serie / serie.shift(1)).dropna()
    ar = {"convergiu": False}
    try:
        res = MarkovAutoregression(
            lr.values, k_regimes=3, order=1, trend="c",
            switching_trend=True, switching_ar=True, switching_variance=True,
        ).fit(em_iter=500, search_reps=20, disp=False)
        nomes = list(res.model.param_names)
        pega = lambda pre, r: float(res.params[nomes.index(f"{pre}[{r}]")])  # noqa: E731
        ar = {"convergiu": True,
              "beta0": [round(pega("const", r), 3) for r in range(3)],
              "sigma2": [round(pega("sigma2", r), 3) for r in range(3)],
              "phi1": [round(pega("ar.L1", r), 3) for r in range(3)]}
    except Exception as exc:
        ar["erro"] = f"{type(exc).__name__}: {exc}"

    (out_dir / "validacao_ar_isa.json").write_text(
        json.dumps({"reproduzido": ar, "publicado": PUBLICADO_AR_ISA,
                    "bordas_bin_por_codigo": bordas,
                    "bordas_bin_publicadas": PUBLICADO_BORDAS_ISA,
                    "bordas_bin_pertencem_a": dono_das_bordas,
                    "pmf_taesa_sweep": sweep},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(df_cross.to_string(index=False))
    print()
    print(df_intra.to_string(index=False))
    print()
    print(df_conv.to_string(index=False))
    for transform, pmf in pmfs.items():
        print(f"PMF Taesa reproduzida ({transform}):")
        print(np.round(pmf, 3))
    print("PMF Taesa publicada:")
    print(PUBLICADO_PMF_TAESA)
    print(f"Varredura da PMF: {sweep['ajustes_avaliados']} ajustes avaliados, "
          f"{len(sweep['reproduz_em'])} reproduzem a Tabela 6")
    print(f"Bordas publicadas {PUBLICADO_BORDAS_ISA} pertencem a: "
          f"{dono_das_bordas or 'nenhum codigo'}")
    print("AR-HMM ISA reproduzido:", ar)
    print("AR-HMM ISA publicado:  ", PUBLICADO_AR_ISA)


if __name__ == "__main__":
    stage_validation(_ROOT / "results" / "data", _ROOT / "results" / "hmm_n64")
