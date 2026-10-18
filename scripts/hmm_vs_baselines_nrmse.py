"""GHMM do congresso contra baselines nao-HMM, na metrica e no protocolo do proprio congresso.

Motivo (2026-09-28): o congresso fechou dizendo que nenhuma
afirmacao se fazia sobre HMMs frente a baselines nao-HMM, e listou essa comparacao como trabalho
futuro. A Proposta 1 de `hmm_models/Model/2_5_Revision_Experiments.ipynb` tinha o codigo, mas
nunca foi executada (nenhuma celula com saida).
Este script a executa, com tres cuidados que o notebook nao tinha:

1. **Os quatro modelos sao recomputados aqui, pelo mesmo laco**, em vez de comparar baselines
   novos com o numero impresso do GHMM.
2. **Duas versoes do GHMM.** `GHMM-fixo` e' a especificacao declarada antes de olhar o teste
   (3 estados, semente 42, n_iter=100 -- celula 8 de `2_4_Modelo_Gaussian_HMM.ipynb`).
   `GHMM-grade` reproduz a selecao do congresso: melhor NRMSE de teste sobre estados {3,4} x as
   22 sementes da celula 9 do mesmo notebook. Essa selecao e' feita no conjunto de avaliacao, entao
   e' otimista por construcao; e' reportada para mostrar de onde vem o numero publicado, nao
   como estimativa honesta.
3. **Dois passeios aleatorios.** `RW-nivel` preve log-retorno zero, o passeio aleatorio sobre o
   nivel (o mesmo que o gate U2 do painel usa). `RW-ultimo` repete o ultimo log-retorno (o
   "naive" do notebook).

Serie: Net Revenue (item 3.01), a unica que o congresso pontuou por NRMSE. Transformacao,
janela de treino (35), previsao (media do estado mais provavel em t+1) e NRMSE (RMSE sobre a
amplitude do periodo de avaliacao) sao os do congresso. Dado: a variante `bruta_alup4_interp` que
`hmm_n64_validation.py` mostrou ser a do congresso. Duas janelas: a do congresso (ate 2024-Q3,
58 log-retornos) e a amostra atual (ate 2025-Q4, 63).

    python scripts/hmm_vs_baselines_nrmse.py            # tabela + grava o CSV
    python scripts/hmm_vs_baselines_nrmse.py --check    # nao escreve; sai !=0 se derivou

Saida: paths.JOURNAL_HMM_VS_BASELINES_CSV
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from statsmodels.tsa.arima.model import ARIMA

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import paths  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

JANELAS = {"congresso": "2024-09-30", "n64": "2025-12-31"}
TREINO = 35
N_ITER = 100
SEMENTE_FIXA = 42
ESTADOS_FIXO = 3
# Celula 9 de 2_4_Modelo_Gaussian_HMM.ipynb (duplicatas removidas; nao alteram o maximo).
SEMENTES_GRADE = sorted({1, 42, 6, 99, 7, 150, 33, 25, 21, 12, 200, 2, 3, 4, 5, 100, 350})
ESTADOS_GRADE = (3, 4)


def _serie(janela_fim: str) -> pd.DataFrame:
    df = pd.read_csv(paths.ROOT / "resultado_bruto.csv", parse_dates=["DT_FIM_EXERC"])
    df = df.set_index("DT_FIM_EXERC").sort_index()
    df.columns = [str(c) for c in df.columns]
    # Variante do congresso (hmm_n64_validation.py): os dois zeros da ALUP4 interpolados.
    df["21490"] = df["21490"].replace(0.0, np.nan).interpolate(method="linear")
    return df.loc[:janela_fim]


def _log_retornos(nivel: pd.Series) -> np.ndarray:
    # Receita de preparar_serie_gaussianhmm(usar_log_returns=True): nao-positivos viram NaN e caem.
    limpa = nivel.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(limpa).diff().dropna().to_numpy()


def _nrmse(prev: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prev - real) ** 2)) / (real.max() - real.min()))


def _ghmm(x: np.ndarray, n_estados: int, semente: int) -> np.ndarray:
    prev = []
    for i in range(TREINO, len(x)):
        treino = x[:i].reshape(-1, 1)
        m = hmm.GaussianHMM(n_components=n_estados, covariance_type="full", n_iter=N_ITER,
                            random_state=semente)
        try:
            m.fit(treino)
            _, estados = m.decode(treino)
            prox = int(np.argmax(m.transmat_[estados[-1]]))
            prev.append(float(m.means_[prox][0]))
        except Exception:  # noqa: BLE001 -- o notebook tambem pulava o passo que falhava
            prev.append(np.nan)
    return np.asarray(prev)


def _arima(x: np.ndarray) -> np.ndarray:
    prev = []
    for i in range(TREINO, len(x)):
        prev.append(float(ARIMA(x[:i], order=(1, 0, 1)).fit().forecast(steps=1)[0]))
    return np.asarray(prev)


def _pontua(prev: np.ndarray, real: np.ndarray) -> float:
    ok = ~np.isnan(prev)
    return _nrmse(prev[ok], real[ok])


def build_table() -> pd.DataFrame:
    linhas: list[dict] = []
    for janela, fim in JANELAS.items():
        df = _serie(fim)
        for code, empresa in cfg.EMPRESAS.items():
            x = _log_retornos(df[str(code)])
            real = x[TREINO:]
            np.random.seed(cfg.RANDOM_SEED)

            grade = {(k, s): _pontua(_ghmm(x, k, s), real)
                     for k in ESTADOS_GRADE for s in SEMENTES_GRADE}
            (k_best, s_best), nrmse_grade = min(grade.items(), key=lambda kv: kv[1])

            modelos = {
                "GHMM-fixo": _pontua(_ghmm(x, ESTADOS_FIXO, SEMENTE_FIXA), real),
                "GHMM-grade": nrmse_grade,
                "ARIMA(1,0,1)": _pontua(_arima(x), real),
                "RW-nivel": _nrmse(np.zeros_like(real), real),
                "RW-ultimo": _nrmse(x[TREINO - 1:-1], real),
            }
            for modelo, v in modelos.items():
                linhas.append({"janela": janela, "empresa": empresa, "n_log_retornos": len(x),
                               "n_previsoes": len(real), "modelo": modelo, "nrmse": round(v, 4),
                               "grade_escolha": f"J={k_best},seed={s_best}"
                               if modelo == "GHMM-grade" else ""})
    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    tab = build_table()
    largo = tab.pivot_table(index=["janela", "empresa"], columns="modelo", values="nrmse")
    print(largo.to_string())

    destino = paths.JOURNAL_HMM_VS_BASELINES_CSV
    if args.check:
        antigo = pd.read_csv(destino, keep_default_na=False)
        novo = tab.astype(str).reset_index(drop=True)
        if not antigo.astype(str).equals(novo):
            print("DERIVA contra", destino)
            return 1
        print("OK:", destino)
        return 0
    tab.to_csv(destino, index=False)
    print("gravado:", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
