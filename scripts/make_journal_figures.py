"""Gera as duas figuras da secao de Resultados do artigo (results/).

  persistencia x FIT%   dispersao rho_1 x FIT% das 12 combinacoes, a correlacao-titulo
                        (0.957 Pearson / 0.951 Spearman) virando imagem. Insumo:
                        paths.JOURNAL_PERSISTENCE_CSV, que `scripts/persistence_vs_fit.py`
                        regenera e verifica com --check.
  regimes               estados de Viterbi. Painel superior: duas series em nivel com o
                        estado decodificado sombreado ao fundo, nos dois extremos de
                        persistencia do painel. Painel inferior: as 12 combinacoes como
                        faixas, agrupadas por empresa, onde a concordancia intra-empresa
                        fica visivel contra a cross-empresa. Insumo:
                        paths.JOURNAL_HMM_DIR/estados_viterbi.csv.

A variante categorica e' a unica usada aqui, pela mesma razao que restringe a analise de
sincronizacao a ela (2026-09-12): e' a unica cujas
celulas nao sao afetadas pelo filtro de espuriedade. O numero de estados (J, hoje 2) vem de
`estados_viterbi.csv`, nao e' fixado aqui -- ver 2026-09-27 (e), especificacao principal revista
de 3 para 2 estados por preferencia do BIC.

Nenhum numero e' calculado aqui -- os dois CSV de entrada sao os canonicos do artigo, e a
deriva e' detectada por `scripts/persistence_vs_fit.py --check` e por
`scripts/export_series_tratadas.py --check`, nao por este script.

CORRECAO 2026-09-27: a legenda interna da figura de persistencia dizia "AR(1) ceiling",
mas a curva e' um "population benchmark", nao um teto -- e o dado contradiz a palavra: 6 dos 12 pontos ficam ACIMA da
curva (persistencia_vs_fit.csv, coluna delta > 0). Corrigido para "AR(1) benchmark". A caixa de
correlacoes tambem carregava "Kendall 0.848", numero que nao aparece em lugar algum do texto;
removido, ficando so Pearson/Spearman, que sao os dois citados em prosa. Figura regenerada.

    python scripts/make_journal_figures.py             # escreve os PDF em paths.JOURNAL_FIGURES_DIR
    python scripts/make_journal_figures.py --png       # PNG tambem, para inspecao rapida

Saida: paths.JOURNAL_FIGURES_DIR / {persistence_fit,regimes}.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402

# Nomes de exibicao -- os mesmos do artigo do congresso.
EMPRESA_LABEL = {
    "ELET6": "Axia Energy",
    "ISAE4": "ISA Energy",
    "TAEE4": "Taesa",
    "ALUP4": "Alupar",
}
EMPRESA_SHORT = {"ELET6": "AXI", "ISAE4": "ISA", "TAEE4": "TAE", "ALUP4": "ALU"}
# Renomeado 2026-09-27 (d): "Gross Income"/"Gross Profit" (termos do congresso)
# -> "Net Revenue"/"Net Income", que descrevem as contas CVM 3.01 e 3.11.
INDICADOR_LABEL = {
    "resultado_bruto": "Net Revenue",
    "lucro_consolidado": "Net Income",
    "despesas_operacionais": "Operational Expenses",
}

# Categoricas em ordem fixa, validadas (validate_palette.js: 6/6 PASS, pior par
# adjacente dE 11.0 em deuteranopia). A forma do marcador repete a identidade, entao o
# indicador nunca e' identificado por cor sozinha -- exigencia de impressao em cinza.
COR_INDICADOR = {
    "resultado_bruto": "#0072B2",
    "lucro_consolidado": "#D55E00",
    "despesas_operacionais": "#009E73",
}
MARCADOR_INDICADOR = {
    "resultado_bruto": "o",
    "lucro_consolidado": "s",
    "despesas_operacionais": "^",
}

# Rampa sequencial de um tom para os 3 estados canonicos: monotonica em luminancia, de
# modo que a ordem dos estados sobrevive a impressao em escala de cinza.
COR_ESTADO = ["#deebf7", "#9ecae1", "#3182bd"]

# Deslocamento do rotulo de cada ponto, em pontos tipograficos. Ha tres combinacoes
# empilhadas perto de (0.39, 5) que colidem sob qualquer deslocamento uniforme.
OFFSET_ROTULO = {
    ("ISAE4", "resultado_bruto"): (-6, 6),
    ("ALUP4", "resultado_bruto"): (-20, -10),
    ("TAEE4", "resultado_bruto"): (-19, 3),
    ("TAEE4", "lucro_consolidado"): (6, -3),
    ("ALUP4", "lucro_consolidado"): (-20, -4),
    ("ISAE4", "lucro_consolidado"): (6, -3),
    ("TAEE4", "despesas_operacionais"): (6, -3),
    ("ISAE4", "despesas_operacionais"): (4, 8),
    ("ELET6", "despesas_operacionais"): (-5, 12),
    ("ELET6", "lucro_consolidado"): (-19, 6),
    ("ALUP4", "despesas_operacionais"): (-16, -14),
    ("ELET6", "resultado_bruto"): (7, -5),
}


def _estilo() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#666666",
        "grid.linewidth": 0.4,
        "grid.color": "#d9d9d9",
        "lines.linewidth": 1.0,
        "figure.dpi": 300,
    })


def figura_persistencia(salvar_png: bool) -> Path:
    """Dispersao rho_1 x FIT%, 12 combinacoes rotuladas, com o AR(1) walk-forward de cada uma."""
    df = pd.read_csv(paths.JOURNAL_PERSISTENCE_CSV)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.axhline(0.0, color="#999999", linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)

    # AR(1) fixo em walk-forward (scripts/ar1_walkforward.py, revisao minima viavel de
    # 2026-10-12): substitui a curva populacional FIT%_AR(1)(rho1) e a reta de minimos quadrados.
    # Cada combinacao ganha um marcador vazado no FIT% do AR(1) medido, ligado ao preferido por um
    # segmento vertical cujo comprimento e' o Delta (FIT% do preferido menos o do AR(1)).
    # Desde 2026-10-18 a referencia passa a ser o ARIMA(1,d,0), com d
    # do consenso de raiz unitaria (coluna fit_ref_wf); o AR(1) em nivel fica so no CSV.
    ar1 = pd.read_csv(paths.JOURNAL_AR1_WALKFORWARD_CSV)[["empresa", "indicador", "fit_ref_wf"]]
    ar1 = ar1.rename(columns={"fit_ref_wf": "fit_ar1_wf"})
    df = df.merge(ar1, on=["empresa", "indicador"], how="left", validate="one_to_one")
    for _, linha in df.iterrows():
        ax.plot([linha["rho1"]] * 2, [linha["fit_ar1_wf"], linha["fit_pct"]],
                color="#B2182B", linewidth=0.7, alpha=0.6, zorder=2)
    ax.scatter(df["rho1"], df["fit_ar1_wf"], s=20, marker="o", facecolor="white",
               edgecolor="#B2182B", linewidth=0.8, zorder=3)

    for indicador, grupo in df.groupby("indicador", sort=False):
        ax.scatter(grupo["rho1"], grupo["fit_pct"],
                   s=34, marker=MARCADOR_INDICADOR[indicador],
                   facecolor=COR_INDICADOR[indicador], edgecolor="white",
                   linewidth=0.7, zorder=4)

    for _, linha in df.iterrows():
        chave = (linha["empresa"], linha["indicador"])
        ax.annotate(EMPRESA_SHORT[linha["empresa"]],
                    xy=(linha["rho1"], linha["fit_pct"]),
                    xytext=OFFSET_ROTULO[chave], textcoords="offset points",
                    fontsize=6.2, color="#404040", zorder=5)

    ax.set_xlabel(r"Lag-one autocorrelation $\rho_1$")
    ax.set_ylabel("Out-of-sample FIT%")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(-21, 62)  # o ARIMA(1,d,0) da despesa operacional da ISA (-16,9) cai abaixo de -14

    # Correlacoes lidas do proprio CSV, nunca hard-coded -- script anterior tinha
    # "Pearson 0.957" congelado de uma era de campeoes superada, destoando do 0.952
    # que o manuscrito (e persistence_vs_fit.py) ja citava.
    pearson_r = stats.pearsonr(df["rho1"], df["fit_pct"])[0]
    spearman_r = stats.spearmanr(df["rho1"], df["fit_pct"])[0]
    ax.text(0.055, 0.955,
            f"Pearson {pearson_r:.3f}\nSpearman {spearman_r:.3f}",
            transform=ax.transAxes, fontsize=6.8, va="top", ha="left", color="#404040",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5))

    legenda = [
        Line2D([0], [0], marker="o", color="#B2182B", linewidth=0.7,
               markerfacecolor="white", markeredgecolor="#B2182B", markersize=4.5,
               label="Fixed ARIMA(1,d,0), walk-forward"),
    ] + [Line2D([0], [0], marker=MARCADOR_INDICADOR[k], color="none",
                      markerfacecolor=COR_INDICADOR[k], markeredgecolor="white",
                      markeredgewidth=0.7, markersize=5.6, label=INDICADOR_LABEL[k])
               for k in INDICADOR_LABEL]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4,
              frameon=False, columnspacing=1.0, handletextpad=0.3, fontsize=6.4)

    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    fig.tight_layout(pad=0.4)
    return _salvar(fig, "persistence_fit", salvar_png)


def _sombrear_estados(ax, datas: pd.Series, estados: pd.Series) -> None:
    """Pinta o fundo do eixo por trecho de estado constante."""
    x = matplotlib.dates.date2num(datas)
    e = estados.to_numpy()
    inicio = 0
    for i in range(1, len(e) + 1):
        if i == len(e) or e[i] != e[inicio]:
            borda_esq = x[inicio] if inicio == 0 else (x[inicio - 1] + x[inicio]) / 2
            borda_dir = x[i - 1] if i == len(e) else (x[i - 1] + x[i]) / 2
            ax.axvspan(borda_esq, borda_dir, facecolor=COR_ESTADO[int(e[inicio])],
                       edgecolor="none", zorder=0)
            inicio = i


def figura_regimes(salvar_png: bool) -> Path:
    """Nivel com estado sombreado (duas empresas) + as 12 combinacoes como faixas.

    A linha e' a serie **em nivel**, nao a serie modelada: a transformacao homogenea de
    2026-09-12 diferencia as doze, e um grafico da primeira diferenca rotulado em reais
    seria ilegivel como serie contabil. Os estados sao datados, entao sombrear o nivel
    pelo estado decodificado na serie diferenciada e' a leitura usual de um grafico de
    regime. Nenhum descritor da metade de identificacao entra aqui -- em particular nao
    rho_1, que pertence a outra metade e cujo cruzamento a secao 5.3.4 exclui por decisao
    de projeto.
    """
    vit = pd.read_csv(paths.JOURNAL_HMM_SYNC_DIR / "estados_viterbi.csv")
    vit = vit[vit["variante"] == "categorical"].copy()
    vit["data"] = pd.to_datetime(vit["data"])
    # J vem do artefato, nao de uma constante repetida (mesma regra de hmm_n64_summary.py):
    # e' o numero de estados efetivamente decodificados na variante categorica da sincronizacao,
    # 3 desde 2026-10-11 (ficou 2 entre 2026-09-27 (e) e essa data: a 2 estados 7 das 12
    # decodificacoes degeneravam em alternancia ou numa quebra unica).
    J = int(vit["estado_canonico"].max()) + 1

    # empresa -> codigo CVM vem do artefato gerado, nunca escrito a mao (o codigo em si
    # jamais e' impresso no artigo).
    protocolo = pd.read_csv(paths.JOURNAL_HMM_SYNC_DIR / "protocolo.csv")
    codigo = dict(zip(protocolo["empresa"], protocolo["codigo_cvm"].astype(str)))

    def nivel(empresa: str, indicador: str) -> pd.Series:
        bruto = pd.read_csv(paths.JOURNAL_DATA_DIR / f"{indicador}.csv",
                            index_col=0, parse_dates=True)
        return bruto[codigo[empresa]].astype(float).dropna()

    fig = plt.figure(figsize=(4.8, 3.5))

    # --- painel superior: a transmissora pura contra a holding diversificada -----------
    # Alturas comprimidas em 2026-09-27 (o par de graficos de nivel e a folga entre paineis
    # sobravam espaco vazio que a versao PDF tight-cropped preservava; a faixa inferior, que
    # carrega o dado denso, nao foi tocada). Uma
    # segunda passada mais agressiva (top=0.975/bottom=0.77) colidiu o rotulo do eixo x do
    # painel superior com "Axia Energy" do painel inferior e escondeu os anos do painel
    # direito -- revertida; esta e' a folga minima que nao colide (verificado no PNG).
    topo = fig.add_gridspec(1, 2, left=0.11, right=0.99, top=0.965, bottom=0.735,
                            wspace=0.34)
    destaques = [
        ("ISAE4", "resultado_bruto"),
        ("ELET6", "resultado_bruto"),
    ]
    for coluna, (empresa, indicador) in enumerate(destaques):
        ax = fig.add_subplot(topo[0, coluna])
        estados = vit[(vit["empresa"] == empresa)
                      & (vit["indicador"] == indicador)].sort_values("data")
        serie = nivel(empresa, indicador)
        _sombrear_estados(ax, estados["data"], estados["estado_canonico"])
        ax.plot(serie.index, serie.to_numpy() / 1e9, color="#1a1a1a", linewidth=0.9,
                zorder=3)
        ax.set_title(f"{EMPRESA_LABEL[empresa]}, {INDICADOR_LABEL[indicador]}\n"
                     "(level, R\\$ billion)", fontsize=6.8, pad=3, linespacing=1.3)
        ax.set_xlim(estados["data"].min(), estados["data"].max())
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.tick_params(labelsize=6.2)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)

    # --- painel inferior: as 12 combinacoes como faixas, agrupadas por empresa ---------
    baixo = fig.add_gridspec(1, 1, left=0.225, right=0.99, top=0.655, bottom=0.148)
    ax = fig.add_subplot(baixo[0, 0])
    ordem_empresa = ["ELET6", "ISAE4", "TAEE4", "ALUP4"]
    ordem_indicador = list(INDICADOR_LABEL)

    datas = np.sort(vit["data"].unique())
    x = matplotlib.dates.date2num(pd.to_datetime(datas))
    bordas = np.concatenate([[x[0] - (x[1] - x[0]) / 2],
                             (x[:-1] + x[1:]) / 2,
                             [x[-1] + (x[-1] - x[-2]) / 2]])

    # Cada empresa ocupa um bloco de tres faixas, precedido de uma faixa vazia que leva o
    # seu nome. O nome ficava ilegivel rotacionado a esquerda ("Axia Energy" e "ISA Energy"
    # ocupam mais altura do que as tres linhas do bloco) e tapava trimestres quando posto
    # por cima das faixas; na folga acima do bloco nao encobre dado algum.
    rotulos_y: list[str] = []
    posicoes_y: list[float] = []
    rotulos_empresa: list[tuple[float, str]] = []
    y = 0.0
    for empresa in ordem_empresa:
        y += 0.80
        rotulos_empresa.append((y - 0.40, EMPRESA_LABEL[empresa]))
        for indicador in ordem_indicador:
            estados = vit[(vit["empresa"] == empresa)
                          & (vit["indicador"] == indicador)].sort_values("data")
            for j, estado in enumerate(estados["estado_canonico"].to_numpy()):
                ax.add_patch(plt.Rectangle(
                    (bordas[j], y + 0.10), bordas[j + 1] - bordas[j], 0.80,
                    facecolor=COR_ESTADO[int(estado)], edgecolor="none"))
            rotulos_y.append(INDICADOR_LABEL[indicador])
            posicoes_y.append(y + 0.5)
            y += 1.0

    ax.set_xlim(bordas[0], bordas[-1])
    ax.set_ylim(0, y)
    ax.set_yticks(posicoes_y)
    ax.set_yticklabels(rotulos_y, fontsize=6.0)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.2)
    for lado in ("top", "right", "left", "bottom"):
        ax.spines[lado].set_visible(False)

    mistura = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for posicao, nome in rotulos_empresa:
        ax.text(0.0, posicao, nome, transform=mistura, fontsize=6.4, va="center",
                ha="left", color="#1a1a1a")

    legenda = [Patch(facecolor=COR_ESTADO[i], edgecolor="#bbbbbb", linewidth=0.4,
                     label=f"State {i + 1}") for i in range(J)]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=J, frameon=False, handlelength=1.6, handleheight=0.9,
              columnspacing=1.4, fontsize=6.8)

    return _salvar(fig, "regimes", salvar_png)


def _salvar(fig, nome: str, salvar_png: bool) -> Path:
    paths.JOURNAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destino = paths.JOURNAL_FIGURES_DIR / f"{nome}.pdf"
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.02)
    if salvar_png:
        fig.savefig(paths.JOURNAL_FIGURES_DIR / f"{nome}.png",
                    bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true",
                        help="grava tambem o PNG, para inspecao fora do LaTeX")
    args = parser.parse_args()

    _estilo()
    for destino in (figura_persistencia(args.png), figura_regimes(args.png)):
        print(f"escrito: {destino.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
