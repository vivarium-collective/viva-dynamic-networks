#!/usr/bin/env python3
"""Canonical run for the ``10-sbml-nodes-uncoupled`` study: SBML reactors on a network — uncoupled control (no diffusion)."""
from __future__ import annotations

import sys
from pathlib import Path

from viva_dynamic_networks.showcase import run_sbml_study

STUDY_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    m = run_sbml_study(
        generator='repressilator_network_uncoupled',
        params={'n': 9.0, 'seed': 1.0},
        duration=200,
        study_dir=STUDY_DIR,
        title='SBML reactors on a network — uncoupled control (no diffusion)',
        species="PX",
        layout="circular",
        subtitle='identical Repressilator nodes with diffusion off (D=0) — the control',
    )
    checks = [
        ('stays desynchronised (late across-node spread > 10, vs ~0 coupled)', bool(m['late_std']>10)),
        ('does not fully collapse (sync_ratio > 0.25)', bool(m['sync_ratio'] is None or m['sync_ratio']>0.25)),
        ('oscillation present (amplitude > 50)', bool(m['amplitude']>50)),
    ]
    print(f"\n=== 10-sbml-nodes-uncoupled ===")
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("viz:", STUDY_DIR / "viz")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
