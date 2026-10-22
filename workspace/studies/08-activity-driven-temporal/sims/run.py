#!/usr/bin/env python3
"""Canonical run for the ``08-activity-driven-temporal`` study: Activity-driven temporal network.

Runs the activity_driven dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='activity_driven',
        params={'m': 1.0, 'n': 44.0},
        n_steps=50,
        study_dir=STUDY_DIR,
        title='Activity-driven temporal network',
        layout='spring',
        color_by='degree',
        subtitle='every step all edges reset; each node fires ∝ its activity, linking to m others',
        seed=7,
    )
    checks = [
        ('nodes conserved (=44)', bool(m['n_nodes']==44)),
        ('sparse snapshot (mean_degree < 2)', bool(m['mean_degree']<2.0)),
    ]
    print(f"\n=== 08-activity-driven-temporal ===")
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
