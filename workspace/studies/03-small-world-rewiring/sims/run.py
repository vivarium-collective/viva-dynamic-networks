#!/usr/bin/env python3
"""Canonical run for the ``03-small-world-rewiring`` study: Watts–Strogatz small-world rewiring.

Runs the small_world_rewire dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='small_world_rewire',
        params={'p': 0.14, 'n': 28.0, 'k': 4.0},
        n_steps=45,
        study_dir=STUDY_DIR,
        title='Watts–Strogatz small-world rewiring',
        layout='circular',
        color_by='degree',
        subtitle='a ring lattice dissolves into a small world as random rewires accumulate',
        seed=7,
    )
    checks = [
        ('edges conserved (=56)', bool(m['n_edges']==56)),
        ('nodes conserved (=28)', bool(m['n_nodes']==28)),
        ('still connected', bool(m['largest_component_fraction']>=0.99)),
    ]
    print(f"\n=== 03-small-world-rewiring ===")
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
