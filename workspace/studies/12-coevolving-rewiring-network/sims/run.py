#!/usr/bin/env python3
"""Canonical run for the ``12-coevolving-rewiring-network`` study: Dynamics on a rewiring network — synchrony survives continual rewiring."""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_sbml_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_sbml_study(
        generator='rewiring_repressilator_network',
        params={'D': 0.25, 'p': 0.6, 'n': 12.0, 'seed': 1.0},
        duration=170,
        study_dir=STUDY_DIR,
        title='Dynamics on a rewiring network — synchrony survives continual rewiring',
        species="PX",
        layout='circular',
        subtitle='a ring of Repressilator reactors whose edges are continuously small-world-rewired',
    )
    checks = [
        ('nodes conserved (=12)', bool(m.get('n_nodes_final',0)==12)),
        ('synchronises despite rewiring (late/early spread < 0.3)', bool(m['sync_ratio'] is not None and m['sync_ratio']<0.3)),
        ('oscillation present (amplitude > 50)', bool(m['amplitude']>50)),
    ]
    print(f"\n=== 12-coevolving-rewiring-network ===")
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("viz:", STUDY_DIR / "viz")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
