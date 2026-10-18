"""Shared helpers for round_status.py and round_query.py.

Kept dependency-light on purpose: only stdlib + config.py. Does NOT import main.py
(which pulls in statsmodels/tensorflow/torch) so these tools stay fast for a
session-start hook.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402
import paths  # noqa: E402

RESULTS_DIR = paths.RESULTS_DIR
SCIENCE_LOG = paths.SCIENCE_LOG

# Result files that exist on disk but are not canonical round outputs — excluded from
# champion selection. Document *why* here (mirrors the cfg.OVERRIDES pattern: explicit,
# not a buried special case).
NON_CANONICAL_FILES: dict[str, str] = {
    "202608020946_v2.json": (
        "Round 02b — corrida residual gerada em background com código anterior ao fix "
        "'narx: solver=svd'; LSTM ativo. Documentado em science_log.md como "
        "'não comparável com Round 02'."
    ),
}


def result_files() -> list[Path]:
    """All canonical round-result JSONs, sorted chronologically by filename timestamp."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    out = []
    for f in files:
        name = f.name
        if not name[0].isdigit():
            continue  # skip exog_selection*.json, outlier_diag*.json
        if name in NON_CANONICAL_FILES:
            continue
        out.append(f)
    return out


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_version(path: Path) -> int:
    """Round/version number encoded in a result filename (…_v23.json -> 23; 0 if absent)."""
    m = re.search(r"_v(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def all_combo_records(eligible_only: bool = True) -> dict[tuple[int, str], list[dict]]:
    """(emp_code, indicador) -> list of {fit, u2, model, mode, exog, dm_p, file} across rounds.

    `eligible_only` (default) drops records produced under a data-treatment era that no longer
    holds for that combo — `cfg.TREATMENT_ERA` (protocolo metodológico §27). A champion elected from a
    superseded era is not reproducible under the vigent `config.py`, which is how the panel ended
    up crediting TAEE4×resultado_bruto a U₂<1 it does not have under the current treatment.
    Pass False to inspect the full history (round_query --history does).
    """
    records: dict[tuple[int, str], list[dict]] = {
        (emp, ind): [] for emp in cfg.EMPRESAS for ind in cfg.INDICADORES
    }
    for f in result_files():
        try:
            data = load_json(f)
        except (json.JSONDecodeError, OSError):
            continue
        version = file_version(f)
        for emp in cfg.EMPRESAS:
            k = str(emp)
            if k not in data:
                continue
            for ind in cfg.INDICADORES:
                r = data[k].get(ind)
                if not r or "percentual_fit" not in r:
                    continue
                if eligible_only and version < cfg.TREATMENT_ERA[(emp, ind)]:
                    continue  # era de tratamento superada — ver protocolo metodológico §27
                records[(emp, ind)].append({
                    "fit": r["percentual_fit"],
                    "u2": r.get("theil_u2"),
                    "model": r.get("modelo_utilizado", "?"),
                    "mode": r.get("preprocessing", "?"),
                    "exog": r.get("exogenas_utilizadas", []),
                    "dm_p": (r.get("dm_vs_arima") or {}).get("pvalue"),
                    "file": f.name,
                })
    return records


def manuscript_champions() -> dict[tuple[int, str], dict | None]:
    """Campeões reeleitos sob o tratamento do manuscrito (Round 41, protocolo metodológico §32).

    Mesmo formato de `compute_champions()`, mais `record` — o registro do eleito, com a grade de
    hiperparâmetros em que venceu, de onde `plot_best_results.run_combo` o reproduz. Combinação sem
    candidato com U₂ < 1 traz o melhor reprovado com `gate_passed=False`. Lê o JSON mais recente de
    `paths.MANUSCRIPT_SEARCH_GLOB` em `paths.RESULTS_EXPERIMENTAL_DIR`.
    """
    files = sorted(paths.RESULTS_EXPERIMENTAL_DIR.glob(paths.MANUSCRIPT_SEARCH_GLOB))
    if not files:
        raise FileNotFoundError("Round 41 ainda não rodou: nenhum " + paths.MANUSCRIPT_SEARCH_GLOB)
    data = load_json(files[-1])["combos"]
    out: dict[tuple[int, str], dict | None] = {}
    for emp in cfg.EMPRESAS:
        for ind in cfg.INDICADORES:
            e = data.get(str(emp), {}).get(ind)
            if e is None:
                out[(emp, ind)] = None
                continue
            r = e["champion"] or e["best_failing"]
            out[(emp, ind)] = {
                "fit": r["percentual_fit"], "u2": r["theil_u2"],
                "model": r["modelo_utilizado"], "mode": r["preprocessing"],
                "exog": r.get("exogenas_utilizadas", []), "dm_p": None,
                "file": files[-1].name, "record": r, "gate_passed": bool(e["gate_passed"]),
            }
    return out


def compute_champions() -> dict[tuple[int, str], dict]:
    """Strict-gate champion per combo: max FIT% among U2<1 candidates.

    If no candidate passes the U2<1 gate, the combo has NO champion (gate failure) —
    the caller is expected to surface that in an "open failures" section rather than
    silently picking a model that violates the project's hard invariant.
    """
    result = {}
    for key, recs in all_combo_records().items():
        passing = [r for r in recs if r["u2"] is not None and r["u2"] < 1]
        best_overall = max(recs, key=lambda r: r["fit"]) if recs else None
        if passing:
            result[key] = {**max(passing, key=lambda r: r["fit"]), "gate_passed": True}
        elif best_overall:
            result[key] = {**best_overall, "gate_passed": False}
        else:
            result[key] = None
    return result


def last_next_hypothesis() -> str:
    """Extract the full 'Próximo:' block from the most recent science_log.md entry.

    Reads to the end of the paragraph, not just the first line — the entry is routinely
    wrapped across several lines, and truncating it mid-sentence loses half of it.
    """
    if not SCIENCE_LOG.exists():
        return "(rounds/science_log.md not found)"
    lines = SCIENCE_LOG.read_text(encoding="utf-8").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"\*\*Próximo:\*\*\s*(.*)", lines[i].strip())
        if not m:
            continue
        block = [m.group(1).strip()]
        for nxt in lines[i + 1:]:
            stripped = nxt.strip()
            if not stripped or stripped.startswith(("**", "#", "*JSON:", "---", "|")):
                break
            block.append(stripped)
        return " ".join(p for p in block if p)
    return "(no 'Próximo:' line found)"


def last_round_header() -> str:
    if not SCIENCE_LOG.exists():
        return "?"
    lines = SCIENCE_LOG.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if line.startswith(("## Round", "## Análise", "## Consolidação")):
            return line.lstrip("# ").strip()
    return "?"
