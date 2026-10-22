#!/usr/bin/env python3
"""Canonical run for the ``05-duplication-divergence`` study: Duplication–divergence (biological growth).

Runs the duplication_divergence dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='duplication_divergence',
        params={'q': 0.55, 'p': 0.1, 'max_nodes': 60.0},
        n_steps=57,
        study_dir=STUDY_DIR,
        title='Duplication–divergence (biological growth)',
        layout='spring',
        color_by='degree',
        subtitle="a new node copies a neighbour's edges, then sheds each with probability 1−q",
        seed=7,
    )
    checks = [
        ('max_degree ≥ 6 (heavy tail)', bool(m['max_degree']>=6)),
        ('clustering ≥ 0.1', bool((m['clustering'] or 0)>=0.1)),
    ]
    print(f"\n=== 05-duplication-divergence ===")
    print("metrics:", {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in m.items() if k != 'mean_degree_series'})
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("viz:", STUDY_DIR / "viz")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
