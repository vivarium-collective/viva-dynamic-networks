#!/usr/bin/env python3
"""Canonical run for the ``02-preferential-attachment`` study: Barabási–Albert preferential attachment.

Runs the preferential_attachment dynamic, renders the evolution GIF + interactive animation into
``viz/``, records the run, and asserts the study's behavior checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_study(
        rule='preferential_attachment',
        params={'m': 2.0, 'max_nodes': 60.0},
        n_steps=58,
        study_dir=STUDY_DIR,
        title='Barabási–Albert preferential attachment',
        layout='spring',
        color_by='degree',
        subtitle="new nodes attach to existing nodes ∝ degree ('rich get richer')",
        seed=7,
    )
    checks = [
        ('max_degree ≥ 10 (hub)', bool(m['max_degree']>=10)),
        ('fully connected', bool(m['largest_component_fraction']>=0.99)),
        ('heavy tail (max ≥ 2.5×mean)', bool(m['max_degree']>=2.5*max(m['mean_degree'],1e-9))),
        ('finite-size exponent present (γ>1.2)', bool(m['powerlaw_exponent'] is None or m['powerlaw_exponent']>1.2)),
    ]
    print(f"\n=== 02-preferential-attachment ===")
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
