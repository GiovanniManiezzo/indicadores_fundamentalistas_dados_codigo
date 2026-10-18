"""Carrega CSVs de indicadores e inputs; alinha índices temporais."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402


def load_indicador(nome: str) -> pd.DataFrame:
    """Retorna DataFrame com DatetimeIndex (DT_FIM_EXERC) e 4 colunas numéricas (códigos)."""
    df = pd.read_csv(paths.ROOT / f"{nome}.csv", parse_dates=["DT_FIM_EXERC"])
    df.columns = [str(c) for c in df.columns]
    return df.set_index("DT_FIM_EXERC").sort_index()


def load_inputs() -> pd.DataFrame:
    """Retorna DataFrame com DatetimeIndex (quarter-end) e 26 colunas (22 macro + Selic + 4 preços)."""
    df = pd.read_csv(paths.INPUTS_CSV)
    df["Data"] = pd.to_datetime(df["Data"], format="%m/%Y") + pd.offsets.MonthEnd(0)
    return df.set_index("Data").sort_index()


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (resultado_bruto, lucro_consolidado, despesas_operacionais, inputs)."""
    rb = load_indicador("resultado_bruto")
    lc = load_indicador("lucro_consolidado")
    do = load_indicador("despesas_operacionais")
    inp = load_inputs()
    return rb, lc, do, inp


if __name__ == "__main__":
    rb, lc, do, inp = load_all()
    for name, df in [
        ("resultado_bruto       ", rb),
        ("lucro_consolidado     ", lc),
        ("despesas_operacionais ", do),
        ("inputs                ", inp),
    ]:
        print(f"{name}: {df.shape}  (+ DatetimeIndex)")
    # shapes esperadas: indicadores (64,4), inputs (64,26)  [date como índice]
    assert rb.shape == (64, 4), f"resultado_bruto: expected (64,4), got {rb.shape}"
    assert inp.shape == (64, 26), f"inputs: expected (64,26), got {inp.shape}"
    print("OK — todos os shapes validados.")
