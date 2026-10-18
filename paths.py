"""Ponto único de verdade dos caminhos do repositório.

Antes deste módulo cada arquivo recalculava `_ROOT = Path(__file__).parent.parent` e
escrevia o nome da subpasta na mão (`Path("results")`, `"modelos_estaveis/plots"`, …),
espalhado por `src/`, `scripts/`, `main.py` e `plot_best_results.py`. Mover uma pasta
exigia caçar todas as ocorrências. Agora move-se aqui.

Convenção: **entrada versionada na raiz**, **código em `src/`/`scripts/`**, **registro
experimental em `rounds/`**, **tudo que é gerado por código em `outputs/`**.

    import paths
    paths.RESULTS_DIR / "202608300945_v30.json"
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Entradas versionadas (dados brutos + contrato de schema) -----------------------
SCHEMA_RESULTADOS = ROOT / "schema_resultados.json"
INPUTS_CSV = ROOT / "dados_input_trimestral.csv"

# --- Fontes externas (CVM), fora do repositório ---------------------------------------
# DFP anual e ITR trimestral consolidados; só `scripts/q4_rebase_scan.py` os lê.
CVM_ANNUAL_CSV = Path("C:/workspace_mestrado/dados_cvm_anuais/base_dados_anual.csv")
CVM_QUARTERLY_CSV = Path("C:/workspace_mestrado/dados_cvm_trimestrais/base_dados_tri.csv")

# --- Código -------------------------------------------------------------------------
SRC_DIR = ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
SCRIPTS_DIR = ROOT / "scripts"

# --- Registro experimental (append-only, versionado) --------------------------------
ROUNDS_DIR = ROOT / "rounds"
SPECS_DIR = ROUNDS_DIR / "specs"
SCIENCE_LOG = ROUNDS_DIR / "science_log.md"

# --- Saídas geradas por código ------------------------------------------------------
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
RESULTS_EXPERIMENTAL_DIR = RESULTS_DIR / "experimental"
# Round 41 (protocolo metodológico §32): funil de exógenas sobre a série do tratamento do manuscrito.
EXOG_SELECTION_MANUSCRIPT = RESULTS_EXPERIMENTAL_DIR / "exog_selection_manuscript_v41.json"
# Campeões reeleitos sob o tratamento do manuscrito (o JSON mais recente deste padrão vale).
MANUSCRIPT_SEARCH_GLOB = "*_round41_manuscript_search_v*.json"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"
PANEL_PLOT = PLOTS_DIR / "real_vs_previsto_melhor_modelo.png"

# --- Tabelas, séries e saídas do HMM usadas no artigo -------------------------------
JOURNAL_DIR = ROOT / "results"
JOURNAL_DATA_DIR = JOURNAL_DIR / "data"
JOURNAL_HMM_DIR = JOURNAL_DIR / "hmm_n64"
# Variante categorica a 3 estados (sincronizacao e figura dos regimes); o resto da metade HMM fica a 2.
JOURNAL_HMM_SYNC_DIR = JOURNAL_DIR / "hmm_n64_k3_rerun"
JOURNAL_PERSISTENCE_CSV = JOURNAL_DIR / "persistencia_vs_fit.csv"
JOURNAL_SIGNIFICANCE_CSV = JOURNAL_DIR / "significancia_painel.csv"
JOURNAL_AR1_WALKFORWARD_CSV = JOURNAL_DIR / "ar1_walkforward.csv"
JOURNAL_RESID_INPUT_CCF_CSV = JOURNAL_DIR / "ccf_residuo_entrada.csv"
JOURNAL_CHAMPION_SPECS_CSV = JOURNAL_DIR / "especificacao_campeoes.csv"
# Round 41: referências sem parâmetro (sazonal ingênuo, média, passeio aleatório) e rho_1 sem imputação.
JOURNAL_SIMPLE_BENCHMARKS_CSV = JOURNAL_DIR / "benchmarks_simples.csv"
JOURNAL_BRIDGE_CSV = JOURNAL_DIR / "bridge_hmm_sysid.csv"
JOURNAL_REVENUE_LINK_CSV = JOURNAL_DIR / "receita_lucro_correlacao.csv"
JOURNAL_HMM_VS_BASELINES_CSV = JOURNAL_DIR / "hmm_vs_baselines_nrmse.csv"
JOURNAL_MP579_CSV = JOURNAL_DIR / "transicoes_mp579.csv"
JOURNAL_Q4_REBASE_CSV = JOURNAL_DIR / "q4_base_reapresentada.csv"
# Sensibilidade do teste de 2012: series sem as celulas da MP 579 imputadas, uma pasta por
# metodo de imputacao das celulas restantes (cfg.HMM_MP579_IMPUTE_METHODS).
JOURNAL_MP579_RAW_DIR = JOURNAL_DIR / "hmm_mp579_bruto"
JOURNAL_MP579_RAW_CSV = JOURNAL_DIR / "transicoes_mp579_bruto.csv"


def journal_mp579_data_dir(method: str) -> Path:
    return JOURNAL_MP579_RAW_DIR / method / "data"


def journal_mp579_hmm_dir(method: str) -> Path:
    return JOURNAL_MP579_RAW_DIR / method / "hmm_n64"
JOURNAL_FIGURES_DIR = JOURNAL_DIR / "figures"


# Prefixos antigos ainda presentes nas 23 specs históricas (`baseline_json: results/...`,
# `exog: {source: results/...}`). As specs são o registro experimental dos Rounds 01-30 e
# devem continuar executáveis exatamente como foram pré-registradas, então `resolve()`
# aceita as duas formas em vez de exigir reescrita retroativa.
_LEGACY_PREFIXES: dict[str, Path] = {
    "results": RESULTS_DIR,
    "modelos_estaveis/plots": PLOTS_DIR,
}


def resolve(path: str | Path) -> Path:
    """Resolve um caminho vindo de spec YAML ou CLI para um caminho absoluto.

    Aceita caminho absoluto, caminho relativo à raiz do repo e as formas legadas
    pré-`outputs/` (`results/x.json` → `outputs/results/x.json`). Se nenhuma existir,
    devolve a interpretação canônica (relativa à raiz) para que o erro apontado ao
    usuário seja o caminho novo, não o legado.
    """
    p = Path(path)
    if p.is_absolute():
        return p

    canonical = ROOT / p
    if canonical.exists():
        return canonical

    posix = p.as_posix()
    for legacy, target in _LEGACY_PREFIXES.items():
        if posix == legacy or posix.startswith(legacy + "/"):
            candidate = target / posix[len(legacy):].lstrip("/")
            if candidate.exists():
                return candidate

    return canonical


def ensure_output_dirs() -> None:
    """Cria a árvore `outputs/` (idempotente). Chamado por quem escreve resultado/plot."""
    for d in (RESULTS_DIR, PLOTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
