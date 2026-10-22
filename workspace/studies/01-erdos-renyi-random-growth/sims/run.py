#!/usr/bin/env python3
"""Canonical run for the ``01-erdos-renyi-random-growth`` study: Erdős–Rényi random growth.

Runs the erdos_renyi dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='erdos_renyi',
        params={'p': 0.08, 'max_nodes': 60.0},
        n_steps=60,
        study_dir=STUDY_DIR,
        title='Erdős–Rényi random growth',
        layout='spring',
        color_by='degree',
        subtitle='node arrives each step, links to each existing node w.p. p',
        seed=7,
    )
    checks = [
        ('giant component ≥0.9', bool(m['largest_component_fraction']>=0.9)),
        ('max_degree < 4×mean (no hub)', bool(m['max_degree'] < 4*max(m['mean_degree'],1e-9)+2)),
    ]
    print(f"\n=== 01-erdos-renyi-random-growth ===")
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
