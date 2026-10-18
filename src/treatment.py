"""Tratamento de outliers de uma série — o ponto único que decide o que é imputado, corrigido ou
mantido, para a metade de previsão (`plot_best_results`) e para a exportação da metade HMM
(`scripts/export_series_tratadas.py`).

Três tratamentos, do mais antigo ao mais novo:

  vigente      `cfg.known_spikes_for()` (sem as datas de SPIKE_REVERTED) + z > SPIKE_Z_THRESH,
               imputação por `cfg.SPIKE_IMPUTE_METHOD`. É o da dissertação (protocolo metodológico §27).
  fixed_rule   Round 39 (§30): lista inteira, imputação causal y(k-1), células da MP 579 brutas.
  manuscript   Round 40 (§31): a regra anterior mais os quartos trimestres de receita corrigidos
               como dado (`cfg.Q4_REBASE_OVERRIDES`, aplicados antes da imputação e nunca
               imputados). A pontuação sem os alvos imputados vive em `walk_forward_cv`.

Importa só `config` e `src.preprocess`: pode ser usado por scripts leves.
"""
from __future__ import annotations

import pandas as pd

import config as cfg
from src.preprocess import detect_and_impute_spikes


def treat_series(
    series: pd.Series,
    emp_code: int,
    indicador: str,
    fixed_rule: bool = False,
    manuscript: bool = False,
    method: str | None = None,
    mp579_bruto: bool = False,
) -> "tuple[pd.Series, list[dict]]":
    """Série tratada + registros de imputação (`detect_and_impute_spikes`).

    `fixed_rule` e `manuscript` são as regras dos Rounds 39 e 40; `manuscript` implica `fixed_rule`.
    `method` e `mp579_bruto` são as variantes de sensibilidade da metade HMM (tratamento vigente
    com as células da MP 579 brutas); não se combinam com as regras pré-fixadas.
    """
    fixed_rule = fixed_rule or manuscript
    s = series.copy()
    exclude: list[str] = []
    if fixed_rule:
        exclude += list(cfg.FIXED_RULE_RAW_CELLS.get((emp_code, indicador), []))
    elif mp579_bruto:
        exclude += list(cfg.HMM_MP579_RAW_CELLS.get((emp_code, indicador), []))
    if manuscript:
        for date, value in cfg.Q4_REBASE_OVERRIDES.get((emp_code, indicador), {}).items():
            s.loc[pd.Timestamp(date)] = value
            exclude.append(date)
    known = [pd.Timestamp(t)
             for t in cfg.known_spikes_for(emp_code, indicador, include_reverted=fixed_rule)]
    if not (known or cfg.SPIKE_Z_THRESH):
        return s, []
    return detect_and_impute_spikes(
        s, z_thresh=cfg.SPIKE_Z_THRESH,
        method=cfg.FIXED_RULE_IMPUTE_METHOD if fixed_rule else (method or cfg.SPIKE_IMPUTE_METHOD),
        known_spikes=known or None,
        exclude=exclude or None,
    )
