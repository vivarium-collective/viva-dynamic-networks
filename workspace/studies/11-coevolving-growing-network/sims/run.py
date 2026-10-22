#!/usr/bin/env python3
"""Canonical run for the ``11-coevolving-growing-network`` study: Dynamics on a growing network — reactors entrain as the graph is built."""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_sbml_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_sbml_study(
        generator='growing_repressilator_network',
        params={'D': 0.25, 'max_nodes': 11.0, 'grow_every': 10.0, 'seed': 2.0},
        duration=170,
        study_dir=STUDY_DIR,
        title='Dynamics on a growing network — reactors entrain as the graph is built',
        species="PX",
        layout='spring',
        subtitle='forest-fire growth adds Repressilator nodes under the running reactors; newcomers entrain',
    )
    checks = [
        ('network grew (final nodes >= 9)', bool(m.get('n_nodes_final',0)>=9)),
        ('mostly entrained (late spread < 0.4x amplitude)', bool(m['late_std'] < 0.4*m['amplitude'])),
        ('numerically stable (late spread < 150)', bool(m['late_std'] < 150)),
        ('oscillation present (amplitude > 50)', bool(m['amplitude']>50)),
    ]
    print(f"\n=== 11-coevolving-growing-network ===")
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("viz:", STUDY_DIR / "viz")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
