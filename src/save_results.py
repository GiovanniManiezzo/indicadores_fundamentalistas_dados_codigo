"""Valida resultados contra JSON Schema e persiste com nome timestampado."""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import jsonschema

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402

_RESULTS_DIR = paths.RESULTS_DIR
_SCHEMA_PATH = paths.SCHEMA_RESULTADOS


def save_results(results: dict, version: int = 1, directory: Path | None = None) -> Path:
    """Valida `results` contra schema_resultados.json e salva em outputs/results/.

    Nome do arquivo: YYYYMMDDHHMM_vN.json  (N incrementa em colisão no mesmo minuto).
    Levanta jsonschema.ValidationError se inválido (não salva).

    `directory`: subpasta opcional (ex.: paths.RESULTS_EXPERIMENTAL_DIR) para rounds explicitamente
    não-comparáveis ao painel — `scripts/_common.py` faz `RESULTS_DIR.glob("*.json")`
    (não-recursivo), então qualquer subpasta fica automaticamente fora do scan de
    `compute_champions()`/`round_status.py`/`round_query.py`, sem risco de promoção acidental.
    """
    target_dir = directory if directory is not None else _RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=results, schema=schema)

    ts = datetime.now().strftime("%Y%m%d%H%M")
    path = target_dir / f"{ts}_v{version}.json"
    n = 1
    while path.exists():
        n += 1
        path = target_dir / f"{ts}_v{version}_{n}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[save_results] Salvo: {path}")
    return path
