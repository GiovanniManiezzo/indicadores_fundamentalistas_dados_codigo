"""HMM na janela N=64 (2010-Q1..2025-Q4) — metade HMM do artigo.

Executa as quatro variantes do artigo DATA 2026 sobre as séries **tratadas** exportadas por
`scripts/export_series_tratadas.py`, e produz só o que cabe à metade HMM segundo a divisão de
papéis travada em o protocolo do painel: estados decodificados, matrizes de transição,
PMFs/parâmetros de emissão, taxas de sincronização e curvas AIC/BIC. **Não computa NRMSE, FIT%
nem U₂** — previsão é da metade de identificação de sistemas.

Protocolo de transformação (revisto com o autor em 2026-09-12, ver `TRANSFORM_MODE`):

    primeira diferença  x_t - x_{t-1}  nas 12 séries, sem exceção

O protocolo anterior (2026-08-30) escolhia a transformação por série pelo consenso ADF/KPSS/PP
— nível, log-retorno ou primeira diferença conforme `d` e o sinal da série. Ele é correto para
modelar uma série isolada e impróprio para comparar séries entre si: o grupo cross-company de um
indicador confrontava estados estimados sobre transformações diferentes, e as taxas de
sincronização caíam ao nível do acaso. O congresso usava transformação única; log-retorno é
indisponível nesta amostra (7 das 12 séries têm observação não-positiva), e a primeira diferença
é a transformação única que sempre existe. `unit_root_consensus` continua rodando e é reportado
em protocolo.csv, agora como documentação de quantas séries são sobre-diferenciadas, não como
critério de escolha.

As variantes contínuas (GHMM, GMM-HMM e AR-HMM) são ajustadas sobre a série padronizada
(z-score). Isso é exigência numérica — nível e diferença estão em R$ 10^8..10^10, e uma variância
de emissão da ordem de 10^18 destrói o condicionamento da covariância e faz o EM do AR-HMM
divergir — e não altera estados de Viterbi, matriz de transição nem o argmin de AIC/BIC, pela
invariância afim da emissão gaussiana. Muda apenas a unidade dos parâmetros de emissão
reportados: β₀ e σ² saem em unidades de σ da série modelada, e φ₁ é invariante à escala.

Independência do diretório: `hmm_models/` não importa `config.py`, `paths.py` nem `src/`
(regra de o protocolo do painel). O mapeamento código CVM -> empresa vem do `manifest.json`
escrito pelo exportador, não de um dicionário duplicado aqui.

Uso:
    python hmm_models/Model/hmm_n64.py --stage base
    python hmm_models/Model/hmm_n64.py --stage robustness
    python hmm_models/Model/hmm_n64.py --stage validation
    python hmm_models/Model/hmm_n64.py --stage all
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.stattools import adfuller, kpss

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _ROOT / "results" / "data"
DEFAULT_OUT_DIR = _ROOT / "results" / "hmm_n64"

# Nomes dos indicadores no artigo. Renomeado 2026-09-27 (d):
# "Gross Income"/"Gross Profit" (termos do artigo do congresso) -> "Net Revenue"/"Net Income",
# que descrevem as contas CVM 3.01 e 3.11. Os CSV/MD já gravados em
# results/hmm_n64/ ainda carregam o rótulo antigo até a próxima execução do pipeline.
PAPER_LABELS = {
    "resultado_bruto": "Net Revenue",
    "lucro_consolidado": "Net Income",
    "despesas_operacionais": "Operational Expenses",
}

# Configuração do congresso, reusada para comparabilidade (Seção "Data Pre-processing").
N_BINS = 3
N_STATES_GRID = (2, 3, 4, 5)
# Renomeado 2026-09-27 (e): especificacao principal passa de 3 para 2 estados.
# O BIC prefere 2 estados em 46 das 47 unidades -- reportar 3 "por
# continuidade" com o estudo de 58 trimestres contradizia o proprio criterio de selecao do
# artigo. A topologia de 3 estados NAO desaparece: stage_base aceita `n_states_main=3` (ou
# `--n-states-main 3` na CLI) para gerar os artefatos do apendice de comparacao com o estudo
# anterior, numa pasta de saida separada.
N_STATES_MAIN = 2
EM_TOL = 0.01
EM_MAX_ITER = 100
N_RESTARTS = 22
BASE_SEED = 42
AR_ORDER = 1
GMM_N_MIX = 2

# === Transformação e escopo da sincronização (decisão do autor, 2026-09-12) ============
# A transformação por consenso de raiz unitária ("unit_root_consensus") era correta para
# modelar cada série isoladamente e errada para compará-las: o grupo cross-company de um
# indicador passava a confrontar estados estimados sobre níveis, log-retornos e primeiras
# diferenças ao mesmo tempo (Gross Profit chegava a misturar as três). Um "estado alto" em
# nível significa despesa grande; em primeira diferença, despesa que cresceu — regimes
# incomensuráveis. Medido: sob transformação heterogênea as três taxas cross-company caem ao
# nível do acaso 100·J^(1-P) = 3,704%, e o ranking publicado se inverte por artefato.
# O congresso usava transformação única (log-retorno). Log-retorno é indisponível aqui — 7 das
# 12 séries têm observação não-positiva —, então a transformação única possível é a primeira
# diferença, sempre definida. Ela vale para toda a metade de regimes: AIC/BIC, transições,
# emissões e sincronização saem dos mesmos ajustes.
TRANSFORM_MODE = "uniform_first_diff"      # ou "unit_root_consensus" (protocolo até 2026-09-12)

# Sincronização restrita ao HMM categórico, como no artigo do congresso: é a única variante
# cujo estado tem o mesmo significado entre séries de escalas diferentes, porque a emissão é
# definida sobre bins do próprio suporte. Verificado: nenhuma das 12 células categóricas muda
# de partição com ou sem o filtro de espuriedade, então a Tabela de sincronização é insensível
# ao filtro. O AR-HMM não entra por falhar (LinAlgError) sob a mesma transformação em que as
# demais variantes convergem — falha registrada em convergencia.csv e em sincronizacao_excluidas.csv.
SYNC_VARIANTS = ("categorical",)

# === Máximos espúrios (McLachlan & Peel 2000, §3.10; Day 1969) =========================
# A verossimilhança de modelos de mistura é ilimitada: uma componente que colapsa sobre um
# ponto leva a variância a zero e o log-likelihood a +infinito. Verificado neste dataset —
# o GMM-HMM atinge variâncias de 1e-233 e log-verossimilhança +226 com N=5, e o GHMM dedica
# um estado a UMA observação. Sem tratar isso, o AIC escolhe o número de estados pela
# singularidade, não pelo ajuste. Soluções assim são marcadas e evitadas na seleção — não
# são apagadas: quantas células caem nesse regime é resultado a reportar.
MIN_OCUPACAO = 2      # um "regime" sustentado por um único trimestre é captura de outlier
VAR_MIN = 1e-3        # variância de emissão abaixo disto é colapso numérico, não regime quieto

warnings.filterwarnings("ignore")
# hmmlearn reporta não-convergência por `logging`, não por `warnings`; com 12 combinações x
# 4 variantes x 4 topologias x 22 reinicializações o ruído esconde a saída útil. A convergência
# de cada ajuste é registrada em convergencia.csv, que é a fonte a ser lida.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# --------------------------------------------------------------------------------------
# Dados e protocolo de transformação
# --------------------------------------------------------------------------------------
@dataclass
class Combo:
    """Uma combinação empresa × indicador, já com a transformação do protocolo aplicada."""

    empresa: str
    codigo: str
    indicador: str
    nivel: pd.Series
    x: pd.Series                      # série modelada (nível, log-retorno ou diferença)
    transform: str
    d: int
    testes: dict = field(default_factory=dict)

    @property
    def rotulo(self) -> str:
        return f"{self.empresa}×{self.indicador}"

    @property
    def z(self) -> pd.Series:
        return (self.x - self.x.mean()) / self.x.std(ddof=0)


def unit_root_consensus(s: pd.Series) -> tuple[int, dict]:
    """ADF + KPSS + PP; d=0 se >=2 testes dizem estacionária. Mesma bateria do painel."""
    out: dict = {}
    adf_p = float(adfuller(s.dropna(), autolag="AIC")[1])
    out["ADF"] = {"p_value": adf_p, "conclusion": "stationary" if adf_p < 0.05 else "unit_root"}

    kpss_p = float(kpss(s.dropna(), regression="c", nlags="auto")[1])
    out["KPSS"] = {"p_value": kpss_p, "conclusion": "unit_root" if kpss_p < 0.05 else "stationary"}

    try:
        from arch.unitroot import PhillipsPerron

        pp_p = float(PhillipsPerron(s.dropna()).pvalue)
        out["PP"] = {"p_value": pp_p, "conclusion": "stationary" if pp_p < 0.05 else "unit_root"}
    except ImportError:  # pragma: no cover - depende do ambiente
        out["PP"] = {"p_value": None, "conclusion": "not_available"}

    votos = sum(1 for t in out.values() if t["conclusion"] == "stationary")
    return (0 if votos >= 2 else 1), out


def apply_protocol(s: pd.Series) -> tuple[pd.Series, str, int, dict]:
    """Aplica a transformação da série conforme `TRANSFORM_MODE`.

    Os testes de raiz unitária rodam nos dois modos: no modo uniforme eles não escolhem a
    transformação, mas continuam documentados em protocolo.csv, que é onde se lê quantas
    séries a primeira diferença sobre-diferencia.
    """
    d, testes = unit_root_consensus(s)
    if TRANSFORM_MODE == "uniform_first_diff":
        return s.diff().dropna(), "first_diff", d, testes
    if d == 0:
        return s.copy(), "level", d, testes
    if (s > 0).all():
        return np.log(s).diff().dropna(), "log_return", d, testes
    return s.diff().dropna(), "first_diff", d, testes


def load_combos(data_dir: Path) -> tuple[list[Combo], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    empresas: dict[str, str] = manifest["empresas"]

    combos: list[Combo] = []
    for indicador in manifest["indicadores"]:
        df = pd.read_csv(data_dir / f"{indicador}.csv", parse_dates=["DT_FIM_EXERC"])
        df = df.set_index("DT_FIM_EXERC").sort_index()
        df.columns = [str(c) for c in df.columns]
        for codigo, empresa in empresas.items():
            serie = df[codigo].astype(float)
            x, transform, d, testes = apply_protocol(serie)
            combos.append(
                Combo(empresa, codigo, indicador, serie, x, transform, d, testes)
            )
    return combos, manifest


# --------------------------------------------------------------------------------------
# Ajuste das variantes
# --------------------------------------------------------------------------------------
def discretize(x: pd.Series, bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Binning por quantis (equifrequência) — Eq. 2 do artigo."""
    est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile", subsample=None)
    codes = est.fit_transform(x.values.reshape(-1, 1)).astype(int)
    return codes, est.bin_edges_[0]


def build_model(variant: str, n_states: int, seed: int, n_mix: int = GMM_N_MIX):
    """Instancia uma variante de hmmlearn com a configuração do congresso."""
    if variant == "categorical":
        return hmm.CategoricalHMM(
            n_components=n_states, n_iter=EM_MAX_ITER, tol=EM_TOL, random_state=seed
        )
    if variant == "gaussian":
        return hmm.GaussianHMM(
            n_components=n_states, covariance_type="full",
            n_iter=EM_MAX_ITER, tol=EM_TOL, random_state=seed,
        )
    if variant == "gmm":
        return hmm.GMMHMM(
            n_components=n_states, n_mix=n_mix, covariance_type="full",
            n_iter=EM_MAX_ITER, tol=EM_TOL, random_state=seed,
        )
    raise ValueError(variant)


def diagnostico_ajuste(model, X, variant: str) -> dict:
    """Ocupação mínima de estado e variância mínima de emissão — base do teste de espuriedade."""
    estados = model.decode(X)[1]
    ocupacao = int(np.bincount(estados, minlength=model.n_components).min())
    if variant == "categorical":
        var_min = float("inf")
    else:
        var_min = float(np.asarray(model.covars_).ravel().min())
    return {"ocupacao_min": ocupacao, "var_min": var_min,
            "espurio": bool(ocupacao < MIN_OCUPACAO or var_min < VAR_MIN)}


def fit_variant(variant: str, X, n_states: int, seeds=None, n_restarts: int = N_RESTARTS):
    """Baum-Welch com múltiplas inicializações.

    Devolve o ajuste de maior log-verossimilhança **entre os não espúrios**; se todas as
    inicializações forem espúrias, devolve o melhor global com a marca — porque "não existe
    ótimo legítimo com N estados nesta série" é informação, e apagar a célula esconderia isso.

    `seeds` explícito serve ao estudo de sensibilidade, onde interessa uma inicialização por
    semente — e não o melhor de N reinicializações, que é justamente o que esconde a dispersão.
    """
    if seeds is None:
        seeds = [BASE_SEED + i for i in range(n_restarts)]

    melhor, melhor_ll, melhor_diag = None, -np.inf, None
    fallback, fallback_ll, fallback_diag = None, -np.inf, None

    for seed in seeds:
        # `random_state` não é suficiente: o GMMHMM do hmmlearn inicializa por k-means a partir
        # do RNG **global** do numpy e ignora o seu próprio `random_state`. Medido nesta amostra
        # (ELET6×resultado_bruto, mistura, 3 estados): sem semear o global, quatro reinicializações
        # nominalmente semeadas devolvem log-verossimilhanças diferentes a cada execução do mesmo
        # processo; semeando, reproduzem dígito a dígito. Sem esta linha as 22 reinicializações não
        # são as sementes que o código nomeia, e o estudo de sensibilidade a semente mede
        # aleatoriedade não controlada. Mesma armadilha já documentada para o AR-HMM em `fit_arhmm`.
        np.random.seed(seed)
        model = build_model(variant, n_states, seed)
        try:
            model.fit(X)
            ll = float(model.score(X))
        except Exception:
            continue
        if not np.isfinite(ll):
            continue
        diag = diagnostico_ajuste(model, X, variant)
        if ll > fallback_ll:
            fallback, fallback_ll, fallback_diag = model, ll, diag
        if not diag["espurio"] and ll > melhor_ll:
            melhor, melhor_ll, melhor_diag = model, ll, diag

    if melhor is not None:
        return melhor, melhor_ll, melhor_diag
    return fallback, fallback_ll, fallback_diag


def fit_categorical(codes: np.ndarray, n_states: int, **kw):
    return fit_variant("categorical", codes, n_states, **kw)


def fit_gaussian(z: np.ndarray, n_states: int, **kw):
    return fit_variant("gaussian", z, n_states, **kw)


def fit_gmm(z: np.ndarray, n_states: int, **kw):
    return fit_variant("gmm", z, n_states, **kw)


def fit_arhmm(x: np.ndarray, n_states: int, order: int = AR_ORDER):
    """AR-HMM = Markov Switching Autoregression (statsmodels), como no notebook 2_3.

    `search_reps` sorteia os valores iniciais do RNG **global** do numpy — o `MarkovSwitching`
    não aceita `random_state`. Sem fixar a semente aqui, o mesmo ajuste converge ou levanta
    `LinAlgError` conforme o que rodou antes no processo, e nada disto seria reproduzível.
    """
    np.random.seed(BASE_SEED)
    mod = MarkovAutoregression(
        x, k_regimes=n_states, order=order, trend="c",
        switching_trend=True, switching_ar=True, switching_variance=True,
    )
    res = mod.fit(em_iter=500, search_reps=20, disp=False)
    return res


def n_params(variant: str, n_states: int, n_symbols: int = N_BINS, n_mix: int = GMM_N_MIX) -> int:
    """Parâmetros livres: início (N-1) + transição N(N-1) + emissão."""
    base = (n_states - 1) + n_states * (n_states - 1)
    if variant == "categorical":
        return base + n_states * (n_symbols - 1)
    if variant == "gaussian":
        return base + 2 * n_states                      # média + variância por estado
    if variant == "gmm":
        return base + n_states * (n_mix - 1) + 2 * n_states * n_mix
    raise ValueError(variant)


def aic_bic(loglik: float, p: int, n: int) -> tuple[float, float]:
    return -2 * loglik + 2 * p, -2 * loglik + p * np.log(n)


# --------------------------------------------------------------------------------------
# Rótulos canônicos e sincronização
# --------------------------------------------------------------------------------------
def canonical_map(emission_means: np.ndarray) -> dict[int, int]:
    """Reordena estados por média de emissão crescente.

    Os rótulos que o Baum-Welch devolve são arbitrários (label switching), então comparar
    rótulos brutos entre empresas — como fez o código do congresso — mistura estados que não
    são o mesmo regime. A ordenação canônica torna a comparação entre séries interpretável.
    """
    ordem = np.argsort(emission_means)
    return {int(antigo): int(novo) for novo, antigo in enumerate(ordem)}


def relabel(states: np.ndarray, mapa: dict[int, int]) -> np.ndarray:
    return np.array([mapa[int(s)] for s in states])


def sync_rate(estados: dict[str, pd.Series]) -> dict:
    """Eq. 5: fração de instantes em que TODAS as entidades estão no mesmo estado."""
    chaves = list(estados)
    idx = estados[chaves[0]].index
    for k in chaves[1:]:
        idx = idx.intersection(estados[k].index)
    if len(idx) == 0:
        return {"K": 0, "coincidencias": 0, "taxa": float("nan")}

    m = pd.DataFrame({k: estados[k].loc[idx] for k in chaves})
    coinc = int(m.nunique(axis=1).eq(1).sum())
    return {"K": int(len(idx)), "coincidencias": coinc, "taxa": coinc / len(idx)}


# --------------------------------------------------------------------------------------
# Estágio: pacote base
# --------------------------------------------------------------------------------------
def stage_base(combos: list[Combo], out_dir: Path, n_states_main: int = N_STATES_MAIN) -> None:
    protocolo, estados_rows, trans_rows, emis_cat, emis_gauss, ar_rows = [], [], [], [], [], []
    criterios, convergencia = [], []
    estados_canon: dict[tuple[str, str, str], pd.Series] = {}

    for c in combos:
        protocolo.append(
            {
                "empresa": c.empresa, "codigo_cvm": c.codigo, "indicador": c.indicador,
                "indicator_label": PAPER_LABELS[c.indicador],
                "N_nivel": len(c.nivel), "K_modelada": len(c.x),
                "consensus_d": c.d, "transform": c.transform,
                "ADF_p": c.testes["ADF"]["p_value"], "KPSS_p": c.testes["KPSS"]["p_value"],
                "PP_p": c.testes["PP"]["p_value"],
                "min_nivel": float(c.nivel.min()), "trocas_de_sinal": int(
                    (np.sign(c.nivel).diff().fillna(0) != 0).sum()
                ),
            }
        )

        codes, edges = discretize(c.x)
        z = c.z.values.reshape(-1, 1)

        # --- curvas AIC/BIC sobre N ∈ {2,3,4,5} para as variantes de hmmlearn -------------
        for variant, fitter, X in (
            ("categorical", fit_categorical, codes),
            ("gaussian", fit_gaussian, z),
            ("gmm", fit_gmm, z),
        ):
            for n_states in N_STATES_GRID:
                model, ll, diag = fitter(X, n_states)
                ok = model is not None
                if ok:
                    p = n_params(variant, n_states)
                    a, b = aic_bic(ll, p, len(X))
                    criterios.append(
                        {"empresa": c.empresa, "indicador": c.indicador, "variante": variant,
                         "n_estados": n_states, "loglik": ll, "n_params": p, "AIC": a, "BIC": b,
                         "ocupacao_min": diag["ocupacao_min"], "var_min": diag["var_min"],
                         "espurio": diag["espurio"]}
                    )
                convergencia.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "variante": variant,
                     "n_estados": n_states, "ajustou": ok,
                     "convergiu": bool(model.monitor_.converged) if ok else False,
                     "espurio": bool(diag["espurio"]) if ok else None}
                )

        # --- especificação principal N=3: estados, transições, emissões -------------------
        cat_model, cat_ll, cat_diag = fit_categorical(codes, n_states_main)
        if cat_model is not None:
            estados = cat_model.decode(codes)[1]
            # média de emissão = índice de bin esperado por estado
            mapa = canonical_map(cat_model.emissionprob_ @ np.arange(N_BINS))
            estados_c = relabel(estados, mapa)
            serie_estados = pd.Series(estados_c, index=c.x.index)
            estados_canon[(c.empresa, c.indicador, "categorical")] = serie_estados

            for dt, s_raw, s_can in zip(c.x.index, estados, estados_c):
                estados_rows.append(
                    {"data": dt.strftime("%Y-%m-%d"), "empresa": c.empresa,
                     "indicador": c.indicador, "variante": "categorical",
                     "estado_bruto": int(s_raw), "estado_canonico": int(s_can),
                     "valor_modelado": float(c.x.loc[dt])}
                )
            _append_transitions(trans_rows, c, "categorical", cat_model.transmat_, mapa)
            for antigo, novo in sorted(mapa.items(), key=lambda kv: kv[1]):
                pmf = cat_model.emissionprob_[antigo]
                emis_cat.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "estado": novo,
                     "Low": float(pmf[0]), "Stable": float(pmf[1]), "High": float(pmf[2]),
                     "bin_edges": json.dumps([round(float(e), 6) for e in edges]),
                     "espurio": cat_diag["espurio"]}
                )

        gauss_model, _, gauss_diag = fit_gaussian(z, n_states_main)
        if gauss_model is not None:
            estados = gauss_model.decode(z)[1]
            mapa = canonical_map(gauss_model.means_.flatten())
            estados_g = relabel(estados, mapa)
            estados_canon[(c.empresa, c.indicador, "gaussian")] = pd.Series(estados_g, index=c.x.index)
            for dt, s_raw, s_can in zip(c.x.index, estados, estados_g):
                estados_rows.append(
                    {"data": dt.strftime("%Y-%m-%d"), "empresa": c.empresa,
                     "indicador": c.indicador, "variante": "gaussian",
                     "estado_bruto": int(s_raw), "estado_canonico": int(s_can),
                     "valor_modelado": float(c.x.loc[dt])}
                )
            _append_transitions(trans_rows, c, "gaussian", gauss_model.transmat_, mapa)
            for antigo, novo in sorted(mapa.items(), key=lambda kv: kv[1]):
                emis_gauss.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "variante": "gaussian",
                     "estado": novo, "media_z": float(gauss_model.means_[antigo, 0]),
                     "var_z": float(gauss_model.covars_[antigo].ravel()[0]),
                     "escala_sigma": float(c.x.std(ddof=0)), "escala_mu": float(c.x.mean()),
                     "espurio": gauss_diag["espurio"]}
                )

        gmm_model, _, gmm_diag = fit_gmm(z, n_states_main)
        if gmm_model is not None:
            estados = gmm_model.decode(z)[1]
            medias = (gmm_model.means_[:, :, 0] * gmm_model.weights_).sum(axis=1)
            mapa = canonical_map(medias)
            estados_gm = relabel(estados, mapa)
            estados_canon[(c.empresa, c.indicador, "gmm")] = pd.Series(estados_gm, index=c.x.index)
            for dt, s_raw, s_can in zip(c.x.index, estados, estados_gm):
                estados_rows.append(
                    {"data": dt.strftime("%Y-%m-%d"), "empresa": c.empresa,
                     "indicador": c.indicador, "variante": "gmm",
                     "estado_bruto": int(s_raw), "estado_canonico": int(s_can),
                     "valor_modelado": float(c.x.loc[dt])}
                )
            _append_transitions(trans_rows, c, "gmm", gmm_model.transmat_, mapa)
            for antigo, novo in sorted(mapa.items(), key=lambda kv: kv[1]):
                emis_gauss.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "variante": "gmm",
                     "estado": novo, "media_z": float(medias[antigo]),
                     "var_z": float(
                         (gmm_model.covars_[antigo].ravel() * gmm_model.weights_[antigo]).sum()
                     ),
                     "escala_sigma": float(c.x.std(ddof=0)), "escala_mu": float(c.x.mean()),
                     "espurio": gmm_diag["espurio"]}
                )

        # --- AR-HMM (statsmodels), N ∈ {2,3} como no congresso ---------------------------
        for n_states in (2, 3):
            try:
                res = fit_arhmm(c.z.values, n_states)
                ok = True
            except Exception as exc:
                ar_rows.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "n_estados": n_states,
                     "regime": None, "convergiu": False, "erro": type(exc).__name__,
                     "beta0": None, "phi1": None, "sigma2": None,
                     "loglik": None, "AIC": None, "BIC": None}
                )
                ok = False
            if not ok:
                convergencia.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "variante": "arhmm",
                     "n_estados": n_states, "ajustou": False, "convergiu": False}
                )
                continue

            convergencia.append(
                {"empresa": c.empresa, "indicador": c.indicador, "variante": "arhmm",
                 "n_estados": n_states, "ajustou": True, "convergiu": True}
            )
            criterios.append(
                {"empresa": c.empresa, "indicador": c.indicador, "variante": "arhmm",
                 "n_estados": n_states, "loglik": float(res.llf),
                 "n_params": int(len(res.params)), "AIC": float(res.aic), "BIC": float(res.bic)}
            )
            betas, phis, sigmas = _arhmm_params(res, n_states)
            for regime in range(n_states):
                ar_rows.append(
                    {"empresa": c.empresa, "indicador": c.indicador, "n_estados": n_states,
                     "regime": regime, "convergiu": True, "erro": None,
                     "beta0": betas[regime], "phi1": phis[regime], "sigma2": sigmas[regime],
                     "loglik": float(res.llf), "AIC": float(res.aic), "BIC": float(res.bic)}
                )
            if n_states == n_states_main:
                estados = np.asarray(res.smoothed_marginal_probabilities).argmax(axis=1)
                idx = c.x.index[-len(estados):]
                mapa = canonical_map(np.array(betas))
                estados_ar = relabel(estados, mapa)
                estados_canon[(c.empresa, c.indicador, "arhmm")] = pd.Series(estados_ar, index=idx)
                for dt, s_raw, s_can in zip(idx, estados, estados_ar):
                    estados_rows.append(
                        {"data": dt.strftime("%Y-%m-%d"), "empresa": c.empresa,
                         "indicador": c.indicador, "variante": "arhmm",
                         "estado_bruto": int(s_raw), "estado_canonico": int(s_can),
                         "valor_modelado": float(c.x.loc[dt])}
                    )
                _append_transitions(trans_rows, c, "arhmm", np.asarray(res.regime_transition)[:, :, 0].T, mapa)

        print(f"  ok  {c.rotulo:32s} d={c.d} {c.transform:11s} K={len(c.x)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(protocolo).to_csv(out_dir / "protocolo.csv", index=False)
    pd.DataFrame(estados_rows).to_csv(out_dir / "estados_viterbi.csv", index=False)
    pd.DataFrame(trans_rows).to_csv(out_dir / "matrizes_transicao.csv", index=False)
    pd.DataFrame(emis_cat).to_csv(out_dir / "emissao_pmf_categorical.csv", index=False)
    pd.DataFrame(emis_gauss).to_csv(out_dir / "emissao_gaussiana.csv", index=False)
    pd.DataFrame(ar_rows).to_csv(out_dir / "arhmm_parametros.csv", index=False)
    pd.DataFrame(criterios).to_csv(out_dir / "aic_bic.csv", index=False)
    pd.DataFrame(convergencia).to_csv(out_dir / "convergencia.csv", index=False)

    _write_sync(estados_canon, combos, out_dir)


def _append_transitions(rows: list, c: Combo, variante: str, transmat: np.ndarray, mapa: dict) -> None:
    n = transmat.shape[0]
    inverso = {novo: antigo for antigo, novo in mapa.items()}
    for i in range(n):
        for j in range(n):
            rows.append(
                {"empresa": c.empresa, "indicador": c.indicador, "variante": variante,
                 "de_estado": i, "para_estado": j,
                 "prob": float(transmat[inverso[i], inverso[j]])}
            )


def _arhmm_params(res, n_states: int) -> tuple[list, list, list]:
    """Extrai (β0, φ1, σ²) por regime do MarkovAutoregression."""
    p = res.params
    nomes = list(res.model.param_names)

    def _pega(prefixos: tuple[str, ...], regime: int) -> float:
        for prefixo in prefixos:
            alvo = f"{prefixo}[{regime}]"
            if alvo in nomes:
                return float(p[nomes.index(alvo)])
        return float("nan")

    betas = [_pega(("const",), r) for r in range(n_states)]
    phis = [_pega(("ar.L1", "y.L1"), r) for r in range(n_states)]
    sigmas = [_pega(("sigma2",), r) for r in range(n_states)]
    return betas, phis, sigmas


def _write_sync(estados: dict, combos: list[Combo], out_dir: Path) -> None:
    """Taxas de sincronização (Eq. 5), cross-company (P=4) e intra-company (P=3)."""
    empresas = sorted({c.empresa for c in combos})
    indicadores = sorted({c.indicador for c in combos}, key=lambda i: list(PAPER_LABELS).index(i))
    ajustadas = sorted({k[2] for k in estados})
    variantes = [v for v in SYNC_VARIANTS if v in ajustadas]

    # Registro do que ficou de fora e por quê — uma variante pode estar ausente da tabela por
    # decisão de escopo ou por ter falhado em parte das 12 séries, e as duas razões são
    # diferentes. A segunda é resultado (o AR-HMM não sustenta a transformação uniforme).
    esperadas = {(c.empresa, c.indicador) for c in combos}
    excluidas = []
    for v in sorted(set(ajustadas) | {"arhmm"}):
        faltando = sorted(f"{e}×{i}" for e, i in esperadas
                          if (e, i, v) not in estados)
        excluidas.append(
            {"variante": v, "na_tabela": v in variantes,
             "escopo": "na tabela" if v in variantes
                       else "fora: sincronização restrita ao HMM categórico, como no congresso",
             "ajuste": "ajustou nas 12 séries" if not faltando
                       else "falhou sob a mesma transformação em que as demais variantes ajustaram",
             "series_sem_ajuste": len(faltando), "quais": "; ".join(faltando)}
        )
    pd.DataFrame(excluidas).to_csv(out_dir / "sincronizacao_excluidas.csv", index=False)

    cross, intra = [], []
    for variante in variantes:
        for indicador in indicadores:
            serie = {e: estados[(e, indicador, variante)] for e in empresas
                     if (e, indicador, variante) in estados}
            if len(serie) < 2:
                continue
            r = sync_rate(serie)
            cross.append(
                {"variante": variante, "indicador": indicador,
                 "indicator_label": PAPER_LABELS[indicador], "P": len(serie), **r,
                 "taxa_pct": round(100 * r["taxa"], 3)}
            )
        for empresa in empresas:
            serie = {i: estados[(empresa, i, variante)] for i in indicadores
                     if (empresa, i, variante) in estados}
            if len(serie) < 2:
                continue
            r = sync_rate(serie)
            intra.append(
                {"variante": variante, "empresa": empresa, "P": len(serie), **r,
                 "taxa_pct": round(100 * r["taxa"], 3)}
            )

    pd.DataFrame(cross).to_csv(out_dir / "sincronizacao_cross_company.csv", index=False)
    pd.DataFrame(intra).to_csv(out_dir / "sincronizacao_intra_company.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["base", "robustness", "validation", "all"], default="base")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-states-main", type=int, default=N_STATES_MAIN,
                     help="topologia da especificacao principal (default: a do texto, "
                          "N_STATES_MAIN=2); use 3 com --out-dir separado para o apendice")
    args = ap.parse_args()

    combos, manifest = load_combos(args.data_dir)
    print(f"{len(combos)} combinações, N={manifest['n_observacoes']} "
          f"({manifest['periodo'][0]} .. {manifest['periodo'][1]}), "
          f"N_STATES_MAIN={args.n_states_main}")

    if args.stage in ("base", "all"):
        print("\n[base] estados, transições, emissões, AIC/BIC, sincronização")
        stage_base(combos, args.out_dir, n_states_main=args.n_states_main)

    if args.stage in ("robustness", "all"):
        from hmm_n64_robustness import stage_robustness

        print("\n[robustness] janela de treino e sensibilidade a semente")
        stage_robustness(combos, args.out_dir, n_states_main=args.n_states_main)

    if args.stage in ("validation", "all"):
        from hmm_n64_validation import stage_validation

        print("\n[validation] janela do congresso, receita publicada")
        stage_validation(args.data_dir, args.out_dir)

    print(f"\nartefatos em {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
