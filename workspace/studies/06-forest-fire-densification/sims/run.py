#!/usr/bin/env python3
"""Canonical run for the ``06-forest-fire-densification`` study: Forest-fire growth & densification.

Runs the forest_fire dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='forest_fire',
        params={'p_forward': 0.4, 'max_nodes': 60.0, 'max_links': 12.0},
        n_steps=57,
        study_dir=STUDY_DIR,
        title='Forest-fire growth & densification',
        layout='spring',
        color_by='degree',
        subtitle="a new node 'burns' recursively outward, linking to a sampled subset of neighbours",
        seed=7,
    )
    checks = [
        ('mean_degree rises (densification)', bool(m['mean_degree_series'][-1] > m['mean_degree_series'][0])),
        ('final mean_degree ≥ 3', bool(m['mean_degree']>=3.0)),
        ('connected', bool(m['largest_component_fraction']>=0.99)),
    ]
    print(f"\n=== 06-forest-fire-densification ===")
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
