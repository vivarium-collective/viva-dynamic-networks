#!/usr/bin/env python3
"""Canonical run for the ``04-fitness-condensation`` study: Bianconi–Barabási fitness attachment.

Runs the fitness_attachment dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='fitness_attachment',
        params={'m': 2.0, 'max_nodes': 60.0},
        n_steps=58,
        study_dir=STUDY_DIR,
        title='Bianconi–Barabási fitness attachment',
        layout='spring',
        color_by='degree',
        subtitle='attachment ∝ fitness × degree — a fit latecomer can overtake old hubs',
        seed=7,
    )
    checks = [
        ('max_degree ≥ 10 (hub)', bool(m['max_degree']>=10)),
        ('fully connected', bool(m['largest_component_fraction']>=0.99)),
        ('heavy tail (max ≥ 2.5×mean)', bool(m['max_degree']>=2.5*max(m['mean_degree'],1e-9))),
    ]
    print(f"\n=== 04-fitness-condensation ===")
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
