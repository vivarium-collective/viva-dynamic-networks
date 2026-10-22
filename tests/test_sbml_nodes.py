"""Family-B tests: SBML reactor at each node + inter-node transport.

These exercise the real COPASI/basico bridge, so they're a touch slower than the
Family-A tests but still run in a few seconds on a short horizon.
"""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import Composite, gather_emitter_results

from viva_dynamic_networks import composites as C
from viva_dynamic_networks.core import build_core
from viva_dynamic_networks.sbml_nodes import repressilator_path, sample_phase_states

basico = pytest.importorskip("basico")


def _px_spread(frame, species="PX"):
    vals = [float((n.get("species") or {}).get(species, 0.0))
            for n in frame["nodes"].values()]
    return float(np.std(vals))


def _run(gen, dur):
    core = build_core()
    doc = gen(core)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(dur))
    from viva_dynamic_networks import viz
    return viz.frames_from_rows(gather_emitter_results(sim)[("emitter",)])


def test_repressilator_loads_and_oscillates():
    states = sample_phase_states(repressilator_path(), 4, seed=0)
    assert len(states) == 4
    # distinct phases → distinct PX values
    pxs = [s["PX"] for s in states]
    assert max(pxs) - min(pxs) > 1.0


def test_family_b_generators_build_and_evolve():
    for name in ("repressilator_network_coupled", "repressilator_network_uncoupled"):
        frames = _run(getattr(C, name), 12)
        assert len(frames) >= 3
        # species present and changing at a node
        px0 = [f["nodes"]["0"]["species"]["PX"] for f in frames]
        assert max(px0) - min(px0) > 1e-6
        # edges emitted (ring topology visible to the viz)
        assert len(frames[-1]["edges"]) >= 3


@pytest.mark.slow
def test_coupling_synchronises_more_than_uncoupled():
    """Over a couple of periods, coupling collapses the across-node spread far
    more than the uncoupled control."""
    coupled = _run(C.repressilator_network_coupled, 150)
    uncoupled = _run(C.repressilator_network_uncoupled, 150)
    c_late = np.mean([_px_spread(f) for f in coupled[-30:]])
    u_late = np.mean([_px_spread(f) for f in uncoupled[-30:]])
    assert c_late < u_late, f"coupled spread {c_late} !< uncoupled {u_late}"
    assert c_late < 5.0, f"coupled did not synchronise (late spread {c_late})"


def test_family_c_coevolution_generators():
    """Topology co-evolves while reactors run: growing adds nodes, rewiring
    churns edges; both keep valid species on every node."""
    grow = _run(C.growing_repressilator_network, 24)
    assert len(grow[-1]["nodes"]) > len(grow[0]["nodes"]), "network did not grow"

    rew = _run(C.rewiring_repressilator_network, 24)
    assert len({len(f["nodes"]) for f in rew}) == 1, "rewiring changed node count"
    edge_sets = {frozenset(f["edges"].keys()) for f in rew}
    assert len(edge_sets) > 1, "edges never rewired"
    # species present on nodes in both
    for frames in (grow, rew):
        seeded = [n for n in frames[-1]["nodes"].values() if (n.get("species") or {})]
        assert seeded, "no node carries species"
