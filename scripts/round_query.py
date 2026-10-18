#!/usr/bin/env python
"""Compact reads over outputs/results/*.json — avoids opening raw JSON files.

Usage:
    python scripts/round_query.py --combo ISAE4:resultado_bruto --history
    python scripts/round_query.py --compare outputs/results/A.json outputs/results/B.json
    python scripts/round_query.py --best
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cfg, compute_champions, load_json, result_files  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TICKER_TO_CODE = {v: k for k, v in cfg.EMPRESAS.items()}


def cmd_history(combo: str) -> None:
    ticker, ind = combo.split(":")
    emp = TICKER_TO_CODE.get(ticker.upper())
    if emp is None or ind not in cfg.INDICADORES:
        print(f"Unknown combo '{combo}'. Tickers: {list(TICKER_TO_CODE)}; "
              f"indicadores: {cfg.INDICADORES}")
        return
    print(f"{'File':<26} {'FIT%':>8} {'U2':>7} {'DM-p':>7}  Model")
    for f in result_files():
        data = load_json(f)
        r = data.get(str(emp), {}).get(ind)
        if not r or "percentual_fit" not in r:
            continue
        dm_p = (r.get("dm_vs_arima") or {}).get("pvalue")
        dm_str = f"{dm_p:.3f}" if isinstance(dm_p, (int, float)) else "—"
        u2 = r.get("theil_u2")
        u2_str = f"{u2:.3f}" if u2 is not None else "?"
        print(f"{f.name:<26} {r['percentual_fit']:>8.1f} {u2_str:>7} {dm_str:>7}  "
              f"{r.get('modelo_utilizado', '?')}")


def cmd_compare(path1: str, path2: str) -> None:
    r1, r2 = load_json(Path(path1)), load_json(Path(path2))
    hdr = f"{'Combo':<32} {'FIT_v1':>8} {'FIT_v2':>8} {'ΔFIT':>8} {'U2_v1':>7} {'U2_v2':>7}"
    print(hdr)
    print("-" * len(hdr))
    for emp, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            k = str(emp)
            a, b = r1.get(k, {}).get(ind), r2.get(k, {}).get(ind)
            if not a or not b:
                continue
            delta = b["percentual_fit"] - a["percentual_fit"]
            combo = f"{ticker}×{ind[:12]}"
            print(f"{combo:<32} {a['percentual_fit']:>8.1f} {b['percentual_fit']:>8.1f} "
                  f"{delta:>+8.1f} {a['theil_u2']:>7.3f} {b['theil_u2']:>7.3f}")


def cmd_best() -> None:
    champions = compute_champions()
    print(f"{'Combo':<32} {'Model':<22} {'FIT%':>7} {'U2':>7}  Source")
    for emp, ticker in cfg.EMPRESAS.items():
        for ind in cfg.INDICADORES:
            c = champions.get((emp, ind))
            combo = f"{ticker}×{ind}"
            if c is None:
                print(f"{combo:<32} (no result)")
                continue
            flag = "" if c["gate_passed"] else " ⚠U2>=1"
            print(f"{combo:<32} {c['model']+flag:<22} {c['fit']:>7.1f} {c['u2']:>7.3f}  {c['file']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combo", help="TICKER:indicador, e.g. ISAE4:resultado_bruto")
    p.add_argument("--history", action="store_true", help="with --combo: FIT%/U2 per round")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two result JSON paths")
    p.add_argument("--best", action="store_true", help="current champion per combo")
    args = p.parse_args()

    if args.combo and args.history:
        cmd_history(args.combo)
    elif args.compare:
        cmd_compare(*args.compare)
    elif args.best:
        cmd_best()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
