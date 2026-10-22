#!/usr/bin/env python3
"""Canonical run for the ``09-sbml-nodes-coupled`` study: SBML reactors on a network — diffusive coupling synchronises."""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_sbml_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_sbml_study(
        generator='repressilator_network_coupled',
        params={'D': 0.3, 'n': 9.0, 'seed': 1.0},
        duration=200,
        study_dir=STUDY_DIR,
        title='SBML reactors on a network — diffusive coupling synchronises',
        species="PX",
        layout="circular",
        subtitle='a Repressilator at each node (viva-copasi); proteins diffuse along the ring',
    )
    checks = [
        ('synchronises (late/early across-node spread < 0.3)', bool(m['sync_ratio'] is not None and m['sync_ratio']<0.3)),
        ('oscillation present (amplitude > 50)', bool(m['amplitude']>50)),
        ('spread shrinks (late_std < early_std)', bool(m['late_std'] < m['early_std'])),
    ]
    print(f"\n=== 09-sbml-nodes-coupled ===")
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("viz:", STUDY_DIR / "viz")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
