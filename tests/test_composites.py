"""Tests for the Family-A composite generators + the viz layer."""
from __future__ import annotations

import pytest
from process_bigraph import Composite, gather_emitter_results

from viva_dynamic_networks import composites as C
from viva_dynamic_networks.core import build_core

GEN_NAMES = [
    "erdos_renyi", "preferential_attachment", "fitness_attachment",
    "duplication_divergence", "forest_fire", "small_world_rewire",
    "adaptive_rewire", "activity_driven",
]


def test_all_generators_registered():
    from process_bigraph.composite_generator import _REGISTRY
    for name in GEN_NAMES:
        matches = [eid for eid in _REGISTRY if eid.endswith(f".{name}")]
        assert matches, f"generator {name} not registered; have {list(_REGISTRY)[:8]}"


@pytest.mark.parametrize("name", GEN_NAMES)
def test_generator_builds_and_runs(name):
    core = build_core()
    gen = getattr(C, name)
    doc = gen(core)
    sim = Composite({"state": doc}, core=core)
    sim.run(12.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert rows, "no emitter rows"
    # the metrics process populated the metric store
    metrics = sim.state["metrics"]
    assert "mean_degree" in metrics and "n_nodes" in metrics
    assert metrics["n_nodes"] >= 1


def test_viz_products_render():
    """frames → positions → interactive figure + GIF all produce output."""
    import tempfile
    from pathlib import Path

    from viva_dynamic_networks import viz

    core = build_core()
    doc = C.preferential_attachment(core, m=2.0, max_nodes=18.0, seed=1)
    sim = Composite({"state": doc}, core=core)
    sim.run(16.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    frames = viz.frames_from_rows(rows)
    assert len(frames) >= 2

    pos = viz.compute_positions(frames, "spring")
    assert pos and all(len(p) == 2 for p in pos.values())

    html = viz.figure_to_html(viz.evolution_figure(frames, pos, title="t"))
    assert "plotly" in html.lower() and len(html) > 5000

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.gif"
        viz.render_gif(frames, pos, out, title="t", fps=5)
        assert out.is_file() and out.stat().st_size > 1000


def test_live_visualization_renders_in_composite():
    from process_bigraph.visualization import render_results

    core = build_core()
    nodes_edges = C.small_world_rewire(core, p=0.5, n=20.0, k=4.0, seed=2)
    # swap the RAMEmitter for the live NetworkEvolution viz
    nodes_edges["viz"] = {
        "_type": "step", "address": "local:NetworkEvolution",
        "config": {"title": "SW", "layout": "circular", "render_mode": "end"},
        "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
    }
    sim = Composite({"state": nodes_edges}, core=core)
    sim.run(20.0)
    res = render_results(sim)
    assert res, "no visualization rendered"
    html = list(res.values())[0]["html"]
    assert "plotly" in html.lower() and len(html) > 5000
