"""Ponto único de verdade: parâmetros, versão da rodada, overrides, RANDOM_SEED."""
from __future__ import annotations
import random
import numpy as np

# === Identidade ===
EMPRESAS: dict[int, str] = {2437: "ELET6", 18376: "ISAE4", 20257: "TAEE4", 21490: "ALUP4"}
INDICADORES: list[str] = ["resultado_bruto", "lucro_consolidado", "despesas_operacionais"]
COMBOS: list[tuple[int, str]] = [(emp, ind) for emp in EMPRESAS for ind in INDICADORES]

# === Reprodutibilidade ===
RANDOM_SEED: int = 42

# === Walk-forward CV ===
MIN_TRAIN: int = 20
HORIZONTE: int = 1

# === Pré-processamento ===
PREPROC_MODES: list[str] = ["raw", "log", "diff", "log+diff", "boxcox"]

# Box-Cox: lambda selecionado por metodo de Guerrero (1993), congelado por combo
# (mesmo padrao anti-leakage de `d`/fixed_order; ver protocolo metodológico secao 25).
BOXCOX_GUERRERO_PERIOD: int = 4          # agrupamento trimestre->ano (N=64 -> 16 sub-series)
BOXCOX_LAMBDA_BOUNDS: tuple[float, float] = (-1.0, 2.0)

# === Modelos candidatos ===
MODELOS: list[dict] = [
    {"name": "ARIMA",    "baseline": True,  "experimental": False},
    {"name": "SARIMA",   "baseline": False, "experimental": False},
    {"name": "ARMAX",    "baseline": False, "experimental": False},
    {"name": "ARIMAX",   "baseline": False, "experimental": False},
    {"name": "OE",       "baseline": False, "experimental": False},
    {"name": "BJ",       "baseline": False, "experimental": False},
    {"name": "SS",       "baseline": False, "experimental": False},
    {"name": "NARX",     "baseline": False, "experimental": False},
    {"name": "NARIMAX",  "baseline": False, "experimental": False},
    # ETS/Theta: benchmarks univariados parcimoniosos (Round 12; ver protocolo metodológico)
    {"name": "ETS",      "baseline": False, "experimental": False},
    {"name": "THETA",    "baseline": False, "experimental": False},
    # Damped-trend structural SS (Round 14; MSOE, distinto do ETS(A,Ad,N) SSOE; ver protocolo metodológico)
    {"name": "DAMPEDTREND", "baseline": False, "experimental": False},
    # LSTM: N=64 → risco de overfitting (ver protocolo metodológico)
    {"name": "LSTM",        "baseline": False, "experimental": True},
    {"name": "RW",          "baseline": False, "experimental": False},
    {"name": "RW_DRIFT",    "baseline": False, "experimental": False},
    # Modelos não-lineares Wiener/Hammerstein/HW (Rounds 05-08)
    {"name": "WIENER",      "baseline": False, "experimental": False},
    {"name": "HAMMERSTEIN", "baseline": False, "experimental": False},
    {"name": "HW",          "baseline": False, "experimental": False},
    # SETAR: regime-switching AR completo, self-exciting (Round 22; ver protocolo metodológico §24).
    # experimental=True: cada regime herda uma fração do MIN_TRAIN já pequeno — risco de
    # overfitting real, mesmo precedente do LSTM (protocolo metodológico §12).
    {"name": "SETAR",       "baseline": False, "experimental": True},
    # Combinação de previsões (Round 16; pool fixo ARIMA+ETS+THETA+SS+DAMPEDTREND; ver protocolo metodológico)
    {"name": "COMBINATION", "baseline": False, "experimental": False},
    # Bayesian ARIMA — MAP via prior Minnesota shrinkage em AR/MA (Round 17; ver protocolo metodológico §22)
    {"name": "BAYESARIMA", "baseline": False, "experimental": False},
]
LSTM_EXPERIMENTAL: bool = True  # N=64: arquitetura ultra-parcimonial obrigatória

# === Configurações dos modelos não-lineares (Rounds 05-07) ===
# p_max=2, q_max=2: reduzido vs ARIMA (compensar parâmetros NL com N=64)
# ALS: Alternating Least Squares, max_iter=10, tol=1e-4

WIENER_CONFIGS: list[dict] = [
    {"output_nl": "sigmoid",   "n_knots": None, "name": "W1-Sigmoid"},
    {"output_nl": "piecewise", "n_knots": None, "name": "W2-Piecewise"},
    {"output_nl": "bspline",   "n_knots": 3,    "name": "W3-BSpline3"},
]

HAMMERSTEIN_CONFIGS: list[dict] = [
    {"input_nl": "sigmoid",   "n_knots": None, "name": "H1-Sigmoid"},
    {"input_nl": "piecewise", "n_knots": None, "name": "H2-Piecewise"},
    {"input_nl": "bspline",   "n_knots": 3,    "name": "H3-BSpline3"},
]

# Combinação de previsões (Round 16) — pool fixo idêntico em todos os 12 combos: ARIMA, ETS,
# THETA, SS, DAMPEDTREND. Três esquemas fixos por instância (nunca escolhidos adaptativamente).
COMBINATION_SCHEMES: list[str] = ["MEAN", "MEDIAN", "INVMSE"]

HW_CONFIGS: list[dict] = [
    # T1 — Gatilho de Mercado: saturação contábil → regime change na precificação
    {"input_nl": "sigmoid",   "output_nl": "piecewise", "n_knots_in": None, "n_knots_out": None, "name": "T1-GatilhoMercado"},
    # T2 — Quebra Estrutural e Suavização: quebra na entrada → transição suave na saída
    {"input_nl": "piecewise", "output_nl": "bspline",   "n_knots_in": None, "n_knots_out": 3,    "name": "T2-QuebraEstrutural"},
    # T3 — Benchmark Data-Driven: sigmoid em ambos (inspirado em NN)
    {"input_nl": "sigmoid",   "output_nl": "sigmoid",   "n_knots_in": None, "n_knots_out": None, "name": "T3-BenchmarkDD"},
    # T4 — Mapeamento Flexível: máxima liberdade topológica na entrada, regime claro na saída
    {"input_nl": "bspline",   "output_nl": "piecewise", "n_knots_in": 3,    "n_knots_out": None, "name": "T4-MapeamentoFlex"},
    # T6 — Assimetria Regulatória: slopes assimétricas Δy>0 vs Δy<0 (reguladas ELET6, TAEE4)
    {"input_nl": "halfspace", "output_nl": "bspline",   "n_knots_in": None, "n_knots_out": 5,    "name": "T6-AssimetriaReg"},
]

# Hiperparâmetros ALS comuns a todos os modelos não-lineares
ALS_MAX_ITER: int = 10
ALS_TOL: float = 1e-4
NL_P_MAX: int = 2  # p_max para bloco linear em modelos NL (< ARIMA p_max=4)
NL_Q_MAX: int = 2  # q_max para bloco linear em modelos NL

# === SETAR — regime-switching self-exciting AR (Round 22; ver protocolo metodológico §24) ===
# Grid deliberadamente conservador: cada regime herda só uma fração do treino disponível.
SETAR_P_MAX: int = 2       # ordem AR máxima por regime (mesma ordem nos dois regimes — parcimônia)
SETAR_DELAY_MAX: int = 2   # delay máximo da variável de threshold self-exciting (y_{t-delay})
SETAR_TRIM: float = 0.15   # trimming nas extremidades da grade de threshold (Andrews 1993)

5# Grid AIC (p,q) para a família SARIMAX com exógenas (ARMAX/ARIMAX/BAYESARIMA — main.py
# `_run_all_models_for_combo`). Também reduzido vs ARIMA p_max=4 (mesma lógica do NL_P_MAX/
# NL_Q_MAX acima: compensar o custo de graus de liberdade extra das exógenas em MIN_TRAIN=20).
# Round 18 (protocolo metodológico §23) testa alargar este grid como hiperparâmetro, via `hyperparams:`
# na spec — ver rounds/specs/.
SARIMAX_FAMILY_P_MAX: int = 2
SARIMAX_FAMILY_Q_MAX: int = 2

# === Feature selection ===
GRANGER_ALPHA: float = 0.10
# Round 02: ELET6 (2437) obteve 0/3 exógenas com α=0.10.
# Round 03: relaxar para α=0.20 para permitir candidatos macro/inflação.
# Round 11: relaxar para α=0.30 — res_bruto/desp_oper seguem 0/3 mesmo com universo completo
# de 26 candidatas em α=0.20 (VIF+Granger+AIC stepwise); luc_consolidado ganha 1 candidata
# extra (IPCA) no stepwise a partir de α=0.30.
GRANGER_ALPHA_OVERRIDE: dict[int, float] = {
    2437: 0.30,  # ELET6 — nenhuma exógena sobreviveu ao Granger α=0.10 em Round 02
}
# Round 31: o funil (VIF→Granger→AIC) lê a série TRATADA (mesma imputação de SPIKE_KNOWN/detecção
# automática do pipeline), não a bruta de load_all(). False reproduz exog_selection_v11.json
# (Rounds 11-30). protocolo metodológico §10 e §27 (era 31).
EXOG_SELECTION_ON_TREATED: bool = True

# === Tratamento de outliers (spikes pontuais) ===
# Spikes documentados no registro do round 00: eventos contábeis não-recorrentes.
# Imputação: (t-1 + t+1) / 2  (linear_interp), aplicada antes do walk-forward CV.
# Documentado em protocolo metodológico como correção pontual de eventos a priori conhecidos.
SPIKE_IMPUTE_METHOD: str = "linear_interp"
SPIKE_Z_THRESH: float | None = 5.0   # detecção automática para z > 5; None = desativado
# Round 04: eventos pontuais confirmados por auditoria (z-score bruto)
# Round 08: AO/TC adicionais detectados via resíduos ARIMA (Round 05-diag)
# Round 21: 2ª passada do diagnóstico (check_outliers.py) sobre a série já tratada —
# outliers mascarados pelo tratamento anterior, mesmo protocolo do Round 10 (Chen & Liu 1993).
# Round 28: dado FALTANTE codificado como 0.0 na origem (ALUP4 2010-06-30 e 2011-06-30, nos três
# indicadores) — `criar_dataframe_indicador` do pipeline de extração aplica `fillna(0)` após um
# `join(how="outer")`, de modo que trimestre ausente vira zero contábil. Não é outlier: é missing
# data. Escapava à detecção automática porque um zero numa série de ~2e8 dá z ≈ -1, muito abaixo
# de SPIKE_Z_THRESH=5.0. Imputado pelo mesmo linear_interp das demais entradas.
SPIKE_KNOWN: dict[tuple[int, str], list[str]] = {
    (2437,  "lucro_consolidado"):     ["2016-06-30", "2012-12-31", "2018-12-31", "2025-12-31"],  # ELET6: z=3.01 (R04) + AO/AO (R08) + AO (R21)
    (2437,  "despesas_operacionais"): ["2012-12-31", "2015-12-31", "2018-12-31", "2014-12-31"],  # ELET6: TC+AO+AO (R08+R10) + AO (R21)
    (18376, "resultado_bruto"):       ["2013-03-31", "2013-06-30", "2020-12-31"],   # ISAE4: AO (R08) + AO+AO (R21)
    (18376, "lucro_consolidado"):     ["2016-09-30", "2020-12-31"],                # ISAE4: z=6.60 (R04) + TC (R08)
    (18376, "despesas_operacionais"): ["2013-09-30", "2024-09-30"],                # ISAE4: AO+AO (R08)
    (20257, "resultado_bruto"):       ["2023-12-31"],                               # TAEE4: TC (R08)
    (20257, "lucro_consolidado"):     ["2022-12-31"],                               # TAEE4: AO (R08)
    (20257, "despesas_operacionais"): ["2022-12-31"],                               # TAEE4: AO (R08)
    (21490, "resultado_bruto"):       ["2019-03-31", "2010-06-30", "2011-06-30"],   # ALUP4: AO (R08) + MISSING (R28)
    (21490, "lucro_consolidado"):     ["2010-06-30", "2011-06-30"],                 # ALUP4: MISSING (R28)
    (21490, "despesas_operacionais"): ["2022-06-30", "2010-06-30", "2011-06-30"],   # ALUP4: AO (R08) + MISSING (R28)
}

# === Reversão de tratamento por combinação — Round 25 (protocolo metodológico §27) ===
# Eventos que o diagnóstico sinalizou e que um round anterior TRATOU, mas cuja imputação foi
# empiricamente rejeitada *para aquela combinação* (piorou FIT%/U₂ frente à série não tratada).
# Existe porque a decisão vinha sendo registrada só no log das rodadas enquanto
# `SPIKE_KNOWN` continuava aplicando o tratamento — o que tornava o campeão da combinação
# irreproduzível sob a configuração vigente. Encoda a decisão no único lugar que a governa.
# Chave ausente = nenhuma reversão; a data listada é removida de SPIKE_KNOWN para aquele combo.
SPIKE_REVERTED: dict[tuple[int, str], list[str]] = {
    (20257, "resultado_bruto"):       ["2023-12-31"],  # TAEE4: TC (R08) revertido no R25 — U₂ 1.027 -> 0.905
    (21490, "resultado_bruto"):       ["2019-03-31"],  # ALUP4: AO (R08) revertido no R25 — FIT 44.1 -> 45.1
    (21490, "despesas_operacionais"): ["2022-06-30"],  # ALUP4: AO (R08) revertido no R26 — FIT -0.3 -> 3.8
}


# === Teste de 2012 da metade HMM sem as células da MP 579 — 2026-10-11 ===
# As células abaixo registram o próprio evento da MP 579
# (ISAE4 2013-Q1/Q2 é queda real de receita),
# e a imputação delas apagava a marca do evento que o teste de transição em 2012 procura.
# Usado SÓ pela exportação de sensibilidade da metade HMM (scripts/export_series_tratadas.py
# --variante mp579_bruto); o painel de previsão e o SPIKE_KNOWN não mudam (escopo restrito ao
# teste de 2012).
HMM_MP579_RAW_CELLS: dict[tuple[int, str], list[str]] = {
    (2437,  "lucro_consolidado"):     ["2012-12-31"],
    (2437,  "despesas_operacionais"): ["2012-12-31"],
    (18376, "resultado_bruto"):       ["2013-03-31", "2013-06-30"],
}
# Método das imputações restantes nessa variante: "previous" = y(k-1), causal;
# "linear_interp" roda como controle, para separar o efeito das células do efeito do método.
HMM_MP579_IMPUTE_METHODS: tuple[str, ...] = ("previous", "linear_interp")


# === Tratamento por regra pré-fixada — Round 39 (protocolo metodológico §30) ===
# Variante usada no artigo (2026-10-12): toda célula de SPIKE_KNOWN é imputada
# (SPIKE_REVERTED ignorado — as reversões foram decididas pela métrica), por y(k-1) causal, e as
# células da MP 579 ficam no valor registrado. O painel da dissertação e TREATMENT_ERA não mudam.
FIXED_RULE_IMPUTE_METHOD: str = "previous"
FIXED_RULE_RAW_CELLS: dict[tuple[int, str], list[str]] = HMM_MP579_RAW_CELLS


# === Tratamento do manuscrito — Round 40 (protocolo metodológico §31) ===
# Estende a regra pré-fixada do Round 39 com (i) correção de dado nos quartos trimestres cuja base
# do anual difere da dos trimestres e (ii) pontuação sem os alvos imputados. Variante só do
# manuscrito: o painel da dissertação e TREATMENT_ERA não mudam (2026-10-12).
#
# (i) Critério objetivo, fixado antes da execução (scripts/q4_rebase_scan.py, receita 3.01 apenas):
# os nove meses reapresentados na ITR do ano seguinte diferem dos arquivados em mais de 10% e o
# anual arquivado difere da sua reapresentação em menos de 5% (já está na base nova).
Q4_REBASE_MIN_9M_CHANGE: float = 0.10
Q4_REBASE_MAX_ANNUAL_CHANGE: float = 0.05
# Valor da célula (R$) = anual arquivado - nove meses reapresentados. A célula deixa de ser
# imputada: o valor corrigido é dado, não outlier. Conferido por `q4_rebase_scan.py --check`.
Q4_REBASE_OVERRIDES: dict[tuple[int, str], dict[str, float]] = {
    (2437,  "resultado_bruto"): {"2011-12-31": 9_136_990_000.0,   # série: 11.545 mi
                                 "2018-12-31": 6_337_450_000.0,   # série: -4.841 mi (Axia 2018-Q4)
                                 "2020-12-31": 9_513_703_000.0},  # série: 3.596 mi
    (18376, "resultado_bruto"): {"2020-12-31": 1_281_537_000.0},  # série: 301 mi (em SPIKE_KNOWN)
}
# (ii) Alvos imputados saem da pontuação (FIT%, U₂, DM, Clark-West, MCS e o AR(1)), com a mesma
# máscara para todos os competidores; a imputação protege a estimação, não serve de gabarito.
MANUSCRIPT_SCORE_EXCLUDES_IMPUTED: bool = True


# === Reeleição sob o tratamento do manuscrito — Round 41 (protocolo metodológico §32) ===
# As duas grades de hiperparâmetros que a busca histórica usou (chaves de HYPERPARAM_WHITELIST): a
# padrão deste arquivo e a estendida dos Rounds 18-34. A estendida só muda as famílias de
# MANUSCRIPT_GRID_SENSITIVE; nas demais daria candidatos idênticos aos da padrão.
MANUSCRIPT_SEARCH_GRIDS: dict[str, dict] = {
    "padrao":    {},
    "estendida": {"nl_p_max": 4, "nl_q_max": 4, "als_max_iter": 20, "als_tol": 1.0e-5,
                  "sarimax_family_p_max": 4, "sarimax_family_q_max": 4},
}
MANUSCRIPT_GRID_SENSITIVE: tuple[str, ...] = (
    "ARMAX", "ARIMAX", "BAYESARIMA", "WIENER", "HAMMER", "HW",
)
# Processos paralelos da busca (um por combinação); cada um semeia RANDOM_SEED ao começar.
MANUSCRIPT_SEARCH_WORKERS: int = 6
# SPA de Hansen (2005) sobre todos os candidatos: bloco médio do bootstrap estacionário (trimestres)
# e número de réplicas.
SPA_BLOCK_SIZE: int = 4
SPA_REPS: int = 2000


def known_spikes_for(emp_code: int, indicador: str, include_reverted: bool = False) -> list[str]:
    """SPIKE_KNOWN da combinação, menos as datas revertidas em SPIKE_REVERTED.

    Único ponto de leitura de SPIKE_KNOWN no pipeline — garante que a decisão de reversão
    valha em todo lugar (rounds, plots, reprodução) e não só onde alguém lembrou de aplicá-la.
    `include_reverted=True` devolve a lista inteira (regra pré-fixada do Round 39, §30).
    """
    reverted = set() if include_reverted else set(SPIKE_REVERTED.get((emp_code, indicador), []))
    return [d for d in SPIKE_KNOWN.get((emp_code, indicador), []) if d not in reverted]


# === Level Shift dummies — Round 08 (protocolo metodológico §11.3) ===
# Dummies binárias d_t = 1[t >= t*]; adicionadas como exógenas em u_exog.
# Detectadas via classify_outlier_type() com classificação LS (Round 05-diag).
LEVEL_SHIFT_DUMMIES: dict[tuple[int, str], list[str]] = {
    (2437,  "resultado_bruto"):       ["2018-12-31"],                # ELET6: LS
    (2437,  "lucro_consolidado"):     ["2015-12-31"],                # ELET6: LS (R10 diag)
    (2437,  "despesas_operacionais"): ["2024-09-30"],                # ELET6: LS (R21 diag)
    (20257, "despesas_operacionais"): ["2020-12-31"],                # TAEE4: LS COVID-19
    (21490, "resultado_bruto"):       ["2020-12-31"],                # ALUP4: LS COVID-19
    (21490, "lucro_consolidado"):     ["2019-03-31", "2020-12-31"],  # ALUP4: LS regulatório + COVID
    (21490, "despesas_operacionais"): ["2025-06-30"],                # ALUP4: LS recente
}

# === Era de tratamento vigente por combinação — Round 25 (protocolo metodológico §27) ===
# Round em que o tratamento de outlier DESTA combinação chegou à forma vigente. Resultados de
# rounds anteriores foram produzidos sobre uma série tratada de outra forma e, portanto, não são
# candidatos legítimos a campeão dela — mesmo que o FIT% seja maior. Sem esta tabela,
# `compute_champions()` elege o máximo FIT% varrendo todas as eras, que é seleção sobre
# configurações de pré-processamento (data snooping) e produz campeão irreproduzível.
# Derivada mecanicamente do histórico de SPIKE_KNOWN/LEVEL_SHIFT_DUMMIES/SPIKE_REVERTED em
# config.py: R08 (90d4a7b), R10 (bf28400), R21 (38eb052), R25/R26 (reversões acima).
TREATMENT_ERA: dict[tuple[int, str], int] = {
    (2437,  "resultado_bruto"):        8,   # LS 2018-12-31 (R08); inalterada desde então
    (2437,  "lucro_consolidado"):     31,   # + AO 2025-12-31 (R21) + funil sobre série tratada (R31)
    (2437,  "despesas_operacionais"): 31,   # + AO 2014-12-31 e LS 2024-09-30 (R21) + funil sobre série tratada (R31)
    (18376, "resultado_bruto"):       31,   # + AO 2013-06-30 e 2020-12-31 (R21) + funil sobre série tratada (R31)
    (18376, "lucro_consolidado"):      8,   # AO/TC (R08)
    (18376, "despesas_operacionais"):  8,   # AO/AO (R08)
    (20257, "resultado_bruto"):       25,   # TC 2023-12-31 revertido (R25)
    (20257, "lucro_consolidado"):      8,   # AO 2022-12-31 (R08)
    (20257, "despesas_operacionais"):  31,   # AO 2022-12-31 + LS 2020-12-31 (R08) + funil sobre série tratada (R31)
    (21490, "resultado_bruto"):       31,   # + MISSING 2010-06-30/2011-06-30 imputado (R28) + funil sobre série tratada (R31)
    (21490, "lucro_consolidado"):     28,   # + MISSING 2010-06-30/2011-06-30 imputado (R28)
    (21490, "despesas_operacionais"): 31,   # + MISSING 2010-06-30/2011-06-30 imputado (R28) + funil sobre série tratada (R31)
}

# === Diagnóstico expandido de outliers — Passos 1 e 2 (protocolo metodológico §11.2/11.3) ===
# Passo 1: threshold nos resíduos padronizados do ARIMA (mais sensível que SPIKE_Z_THRESH=5.0)
OUTLIER_DIAG_Z_THRESH: float = 3.0
# Passo 2: janela (trimestres) para estimar μ_pré e μ_pós no teste de level shift
OUTLIER_LS_WINDOW: int = 8
# Passo 2: janela de decaimento para distinguir TC de LS
OUTLIER_TC_WINDOW: int = 3
# Passo 2: fração de σ_e acima da qual shift pré→pós é classificado como LS/TC (não AO)
OUTLIER_LS_THRESH_FACTOR: float = 0.5

# === Estimação recursiva — RPEM (Round 35; ver protocolo metodológico §28) ===
# Fator de esquecimento λ fixo pela literatura, nunca por grid search: memória efetiva
# 1/(1-λ) = 50 trimestres, dentro da faixa 0,98-0,995 de Ljung (1999, §11.2).
RPEM_FORGETTING: float = 0.98
# Passo relativo das diferenças finitas centrais que dão o gradiente ψ do preditor.
RPEM_FD_REL_STEP: float = 1e-4
# rcond da pseudo-inversa de R_t: direções não excitadas (parâmetro não identificável) ficam
# fora do passo de Gauss-Newton em vez de explodirem.
RPEM_PINV_RCOND: float = 1e-10
# Projeção em D_M: máximo de meias-reduções do passo antes de rejeitá-lo (μ ≥ 2^−10).
RPEM_MAX_HALVINGS: int = 10
# Reinicialização da covariância em eventos regulatórios conhecidos (Round 37; Goodwin & Sin
# 1984): R_t ← ρ·R_t antes de incorporar a observação do trimestre do evento. ρ = 0,2 deixa
# a informação de ~4 trimestres (um ano) da memória de ~20 trimestres de λ = 0,95.
RPEM_RESET_RHO: float = 0.2
# Trimestres de mudança persistente de regime, só de 2015-Q1 (fim de MIN_TRAIN) em diante —
# a MP 579/2012 cai na janela inicial, estimada em lote. Fontes: results/
# RBSE, CPC 47/ICPC 01 (vigência 01/01/2018), Lei 14.182/2021.
RPEM_RESET_EVENTS: dict[int, list[tuple[str, str]]] = {
    2437:  [("2017-09-30", "RBSE: início do recebimento (Portaria MME 120/2016)"),
            ("2018-03-31", "CPC 47/ICPC 01: receita de transmissão como ativo de contrato"),
            ("2022-06-30", "Capitalização da Eletrobras (Lei 14.182/2021)")],
    18376: [("2017-09-30", "RBSE: início do recebimento (Portaria MME 120/2016)"),
            ("2018-03-31", "CPC 47/ICPC 01: receita de transmissão como ativo de contrato")],
    20257: [("2018-03-31", "CPC 47/ICPC 01: receita de transmissão como ativo de contrato")],
    21490: [("2018-03-31", "CPC 47/ICPC 01: receita de transmissão como ativo de contrato")],
}

# === Exógenas defasadas (Round 38; ver protocolo metodológico §29) ===
# Defasagem, em trimestres, aplicada às exógenas macro/mercado no round `LAGGED_EXOG`: a previsão
# de y_t usa x_{t-L}, e não x_t. O protocolo oficial do painel usa L = 0 (previsão condicional).
# As dummies LS_* são intervenções determinísticas conhecidas de antemão e nunca são defasadas.
EXOG_LAG_QUARTERS: int = 1

# === Hiperparâmetros sobrescrevíveis por spec (Round 18+; rounds/specs/) ===
# Mapeia chave da spec -> constante deste arquivo. Whitelist deliberadamente explícita (não um
# setattr genérico) para que uma spec nunca sobrescreva um invariante do painel (MIN_TRAIN,
# RANDOM_SEED, o gate U₂). Vive aqui, e não em main.py, porque `plot_best_results.py` também
# precisa dela: reproduzir o campeão de um round exige reaplicar os hiperparâmetros daquele
# round — os wrappers NL leem NL_P_MAX/ALS_* de `cfg` em tempo de fit (Round 25).
HYPERPARAM_WHITELIST: dict[str, str] = {
    "nl_p_max":             "NL_P_MAX",
    "nl_q_max":             "NL_Q_MAX",
    "als_max_iter":         "ALS_MAX_ITER",
    "als_tol":              "ALS_TOL",
    "sarimax_family_p_max": "SARIMAX_FAMILY_P_MAX",
    "sarimax_family_q_max": "SARIMAX_FAMILY_Q_MAX",
}

# === Versão da rodada atual ===
ROUND_VERSION: int = 3

# === Overrides manuais por (empresa_code, indicador) ===
# Ex: OVERRIDES = {(2437, "resultado_bruto"): {"modelo": "SARIMA", "exogenas": ["Selic_meta_pct_aa"]}}
OVERRIDES: dict[tuple[int, str], dict] = {}


def apply_global_seed() -> None:
    """Aplica RANDOM_SEED a numpy, random, tensorflow e torch."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
    except (ImportError, OSError):
        pass
