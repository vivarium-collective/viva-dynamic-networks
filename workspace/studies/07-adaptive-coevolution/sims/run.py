#!/usr/bin/env python3
"""Canonical run for the ``07-adaptive-coevolution`` study: Adaptive coevolution (opinions + rewiring).

Runs the adaptive_rewire dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='adaptive_rewire',
        params={'w': 0.65, 'n': 44.0, 'k': 4.0},
        n_steps=70,
        study_dir=STUDY_DIR,
        title='Adaptive coevolution (opinions + rewiring)',
        layout='spring',
        color_by='state',
        subtitle='opinions spread while nodes cut ties to the unlike and bond with the alike',
        seed=7,
    )
    checks = [
        ('edges conserved (=88)', bool(m['n_edges']==88)),
        ('nodes conserved (=44)', bool(m['n_nodes']==44)),
    ]
    print(f"\n=== 07-adaptive-coevolution ===")
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
