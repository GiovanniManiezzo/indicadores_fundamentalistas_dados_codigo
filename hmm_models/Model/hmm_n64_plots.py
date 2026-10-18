"""Figuras da metade HMM: curvas AIC/BIC por número de estados ocultos.

Contrapartida em N=64 da Figura 2 do artigo publicado, que mostrava AIC e BIC contra o número de
estados para Gross Income no Categorical HMM e no GHMM. Aqui as três variantes de emissão entram
como linhas do painel e cada indicador ganha sua própria figura.

Leitura pretendida: onde cada critério atinge o mínimo. O mínimo é marcado e rotulado em cada
painel, porque é a única quantidade que a figura precisa comunicar — a escala absoluta do AIC não
é comparável entre painéis (séries e variantes diferentes), e por isso o eixo y é livre por painel.

Ajustes espúrios (estado com ocupação < 2 ou variância de emissão colapsada) são **excluídos** da
curva: um máximo de verossimilhança produzido por singularidade escolheria o número de estados
pela patologia. Painel sem nenhum ajuste legítimo é rotulado como tal em vez de desenhado.

AIC e BIC são distinguidos por cor **e** por traço/marcador: a figura precisa sobreviver à
impressão em tons de cinza do proceedings.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _ROOT / "results" / "hmm_n64"

# Paleta categórica validada (slots 1 e 2), com codificação secundária por traço e marcador.
SERIES = {
    "AIC": {"color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "BIC": {"color": "#eb6834", "linestyle": "--", "marker": "s"},
}
VARIANTES = {"categorical": "Categorical HMM", "gaussian": "GHMM", "gmm": "GMM-HMM"}
# Renomeado 2026-09-27 (d): "Gross Income"/"Gross Profit" -> "Net Revenue"/"Net Income" (ver
# hmm_n64.py PAPER_LABELS).
LABELS = {"resultado_bruto": "Net Revenue", "lucro_consolidado": "Net Income",
          "despesas_operacionais": "Operational Expenses"}
EMPRESAS_ORDEM = ["ELET6", "ISAE4", "TAEE4", "ALUP4"]

TEXTO, TEXTO_SUAVE, GRADE = "#0b0b0b", "#52514e", "#d8d7d2"


def _estilo() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_SUAVE, "axes.titlecolor": TEXTO,
        "xtick.color": TEXTO_SUAVE, "ytick.color": TEXTO_SUAVE,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_aic_bic(df: pd.DataFrame, indicador: str, out_dir: Path) -> Path:
    variantes = [v for v in VARIANTES if v in set(df["variante"])]
    empresas = [e for e in EMPRESAS_ORDEM if e in set(df["empresa"])]

    fig, axes = plt.subplots(
        len(variantes), len(empresas), figsize=(7.2, 1.7 * len(variantes) + 0.9),
        squeeze=False, sharex=True,
    )

    for i, variante in enumerate(variantes):
        for j, empresa in enumerate(empresas):
            ax = axes[i][j]
            sub = df[(df["variante"] == variante) & (df["empresa"] == empresa)].sort_values("n_estados")
            if "espurio" in sub.columns:
                sub = sub[~sub["espurio"].fillna(False).astype(bool)]
            if sub.empty:
                ax.text(0.5, 0.5, "no non-spurious optimum", ha="center", va="center",
                        fontsize=7, color=TEXTO_SUAVE, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for lado in ("top", "right"):
                    ax.spines[lado].set_visible(False)
                if j == 0:
                    ax.set_ylabel(VARIANTES[variante], fontsize=8, color=TEXTO)
                continue

            for criterio, estilo in SERIES.items():
                ax.plot(sub["n_estados"], sub[criterio], linewidth=1.4, markersize=4,
                        label=criterio, **estilo)
                arg = sub.loc[sub[criterio].idxmin()]
                ax.plot(arg["n_estados"], arg[criterio], marker=estilo["marker"], markersize=7,
                        color=estilo["color"], markeredgecolor="white", markeredgewidth=1.2,
                        zorder=5, linestyle="none")

            melhor_aic = int(sub.loc[sub["AIC"].idxmin(), "n_estados"])
            melhor_bic = int(sub.loc[sub["BIC"].idxmin(), "n_estados"])
            ax.set_title(f"{empresa} · AIC N={melhor_aic}, BIC N={melhor_bic}"
                         if i == 0 else f"AIC N={melhor_aic}, BIC N={melhor_bic}",
                         fontsize=7.5, color=TEXTO, pad=4)
            ax.set_xticks(sorted(sub["n_estados"].unique()))
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            if j == 0:
                ax.set_ylabel(VARIANTES[variante], fontsize=8, color=TEXTO)
            if i == len(variantes) - 1:
                ax.set_xlabel("hidden states", fontsize=8)

    handles, rotulos = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[:2], rotulos[:2], loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{LABELS[indicador]} — information criteria by number of hidden states (N = 64)",
                 fontsize=9, color=TEXTO, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    destino = out_dir / f"aic_bic_{indicador}.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    return destino


def main(out_dir: Path = DEFAULT_OUT_DIR) -> int:
    _estilo()
    df = pd.read_csv(out_dir / "aic_bic.csv")
    df = df[df["variante"].isin(VARIANTES)]
    for indicador in LABELS:
        sub = df[df["indicador"] == indicador]
        if sub.empty:
            continue
        print("escrito ", plot_aic_bic(sub, indicador, out_dir).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
