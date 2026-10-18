"""Exporta as séries tratadas (N=64) que o painel de fato modela, para a metade HMM.

Motivo: os notebooks do HMM leem CSVs de indicador,
mas o tratamento de outlier/missing do painel é aplicado em tempo de execução pelo
pipeline — não está gravado no CSV da raiz. Sem exportar, as duas metades do
artigo falariam de séries diferentes, e a ALUP4 ainda teria os zeros de dado faltante
que o Round 28 corrigiu.

Desde o Round 40 (protocolo metodológico §31) o tratamento exportado é o do manuscrito
(`src/treatment.py`, `manuscript=True`): lista inteira de `cfg.known_spikes_for(...,
include_reverted=True)` mais z > `cfg.SPIKE_Z_THRESH`, imputação causal y(k-1), células da MP 579
no valor registrado e os quartos trimestres de `cfg.Q4_REBASE_OVERRIDES` corrigidos como dado. É o
mesmo da metade de previsão, de modo que as duas metades falam da mesma série. O tratamento da
dissertação continua sendo `treated_frame(indicador)` sem a flag. Nenhuma transformação além
disso — log/Box-Cox/diferenciação são decisões de cada metade, não do dado exportado.

Saída (paths.JOURNAL_DATA_DIR):
    <indicador>.csv        série tratada, índice DT_FIM_EXERC, 4 colunas (códigos CVM)
    manifest.json          proveniência: datas imputadas, valor original e imputado, era

    python scripts/export_series_tratadas.py [--check]
    python scripts/export_series_tratadas.py --variante mp579_bruto

`--variante mp579_bruto` (anterior ao Round 40 e sobre o tratamento da
dissertação, hoje sem uso no manuscrito) grava, para cada método em
`cfg.HMM_MP579_IMPUTE_METHODS`, as séries com as células de `cfg.HMM_MP579_RAW_CELLS` em valor
bruto, em `paths.journal_mp579_data_dir(método)`. Só alimenta o teste de 2012 da metade HMM.

`--check` não escreve: compara o que está em disco com o que seria gerado e sai != 0 na
divergência, para que a série usada pelo HMM não derive silenciosamente do painel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
import paths  # noqa: E402
from src.loaders import load_indicador  # noqa: E402
from src.treatment import treat_series  # noqa: E402


def treated_frame(
    indicador: str, method: str | None = None, mp579_bruto: bool = False, manuscript: bool = False
) -> tuple[pd.DataFrame, list[dict]]:
    """Série tratada das 4 empresas para um indicador + registros de imputação.

    `manuscript=True`: tratamento do Round 40 (protocolo metodológico §31), o mesmo da metade de previsão.
    """
    df = load_indicador(indicador)
    cols: dict[str, pd.Series] = {}
    registros: list[dict] = []

    for code in cfg.EMPRESAS:
        tratada, recs = treat_series(
            df[str(code)].astype(float), code, indicador,
            manuscript=manuscript, method=method, mp579_bruto=mp579_bruto,
        )
        cols[str(code)] = tratada
        for r in recs:
            registros.append(
                {
                    "empresa": cfg.EMPRESAS[code],
                    "codigo_cvm": code,
                    "indicador": indicador,
                    "data": pd.Timestamp(r["timestamp"]).strftime("%Y-%m-%d"),
                    "original": float(r["original"]),
                    "imputado": float(r["imputed"]),
                    "z_score": None if r.get("z_score") is None else float(r["z_score"]),
                    "treatment_era": cfg.TREATMENT_ERA.get((code, indicador)),
                }
            )

    return pd.DataFrame(cols, index=df.index), registros


def build(method: str | None = None, mp579_bruto: bool = False, manuscript: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    frames: dict[str, pd.DataFrame] = {}
    imputacoes: list[dict] = []
    for indicador in cfg.INDICADORES:
        frame, recs = treated_frame(indicador, method, mp579_bruto, manuscript)
        frames[indicador] = frame
        imputacoes.extend(recs)

    qualquer = next(iter(frames.values()))
    manifest = {
        "gerado_por": "scripts/export_series_tratadas.py",
        "n_observacoes": int(len(qualquer)),
        "periodo": [qualquer.index[0].strftime("%Y-%m-%d"), qualquer.index[-1].strftime("%Y-%m-%d")],
        "empresas": {str(k): v for k, v in cfg.EMPRESAS.items()},
        "indicadores": list(cfg.INDICADORES),
        "spike_z_thresh": cfg.SPIKE_Z_THRESH,
        "spike_impute_method": (cfg.FIXED_RULE_IMPUTE_METHOD if manuscript
                                else method or cfg.SPIKE_IMPUTE_METHOD),
        "tratamento_manuscrito": (
            {"q4_corrigidos": {f"{k[0]}:{k[1]}": v for k, v in cfg.Q4_REBASE_OVERRIDES.items()},
             "celulas_brutas": {f"{k[0]}:{k[1]}": v for k, v in cfg.FIXED_RULE_RAW_CELLS.items()}}
            if manuscript else {}
        ),
        "mp579_celulas_brutas": (
            {f"{k[0]}:{k[1]}": v for k, v in cfg.HMM_MP579_RAW_CELLS.items()} if mp579_bruto else {}
        ),
        "treatment_era": {f"{k[0]}:{k[1]}": v for k, v in cfg.TREATMENT_ERA.items()},
        "imputacoes": sorted(imputacoes, key=lambda r: (r["indicador"], r["empresa"], r["data"])),
    }
    return frames, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="não escreve; falha se o disco divergir")
    ap.add_argument("--variante", choices=["manuscrito", "mp579_bruto"], default="manuscrito")
    args = ap.parse_args()

    if args.variante == "mp579_bruto":
        for method in cfg.HMM_MP579_IMPUTE_METHODS:
            frames, manifest = build(method, mp579_bruto=True)
            _write(frames, manifest, paths.journal_mp579_data_dir(method))
        return 0

    frames, manifest = build(manuscript=True)
    destino = paths.JOURNAL_DATA_DIR

    if args.check:
        divergencias = []
        for indicador, frame in frames.items():
            alvo = destino / f"{indicador}.csv"
            if not alvo.exists():
                divergencias.append(f"{alvo.name}: ausente")
                continue
            disco = pd.read_csv(alvo, parse_dates=["DT_FIM_EXERC"]).set_index("DT_FIM_EXERC")
            disco.columns = [str(c) for c in disco.columns]
            if not disco.equals(frame.rename_axis("DT_FIM_EXERC")):
                divergencias.append(f"{alvo.name}: conteúdo difere do tratamento do manuscrito")
        if divergencias:
            print("DIVERGÊNCIA entre results/data/ e o config.py:")
            for d in divergencias:
                print(f"  - {d}")
            return 1
        print(f"OK — {len(frames)} indicadores conferem com o tratamento do manuscrito.")
        return 0

    _write(frames, manifest, destino)
    return 0


def _write(frames: dict[str, pd.DataFrame], manifest: dict, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for indicador, frame in frames.items():
        frame.rename_axis("DT_FIM_EXERC").to_csv(destino / f"{indicador}.csv")
        print(f"escrito  {destino.name}/{indicador}.csv  ({frame.shape[0]}x{frame.shape[1]})")

    (destino / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"escrito  {destino}/manifest.json  ({len(manifest['imputacoes'])} imputações)")


if __name__ == "__main__":
    raise SystemExit(main())
