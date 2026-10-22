"""Unit + end-to-end tests for the reaction library and the two processes."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import Composite, gather_emitter_results

from viva_dynamic_networks import graph_init as gi
from viva_dynamic_networks.core import build_core
from viva_dynamic_networks.metrics import graph_metrics, powerlaw_exponent
from viva_dynamic_networks.reactions import (
    REACTIONS,
    GraphView,
    edge_key,
    get_reaction,
)

ALL_RULES = sorted(REACTIONS)


def test_all_eight_rules_registered():
    assert set(REACTIONS) == {
        "erdos_renyi", "preferential_attachment", "fitness_attachment",
        "duplication_divergence", "forest_fire", "small_world_rewire",
        "adaptive_rewire", "activity_driven",
    }


def test_edge_key_is_order_independent_undirected():
    assert edge_key("3", "7") == edge_key("7", "3")
    assert edge_key("3", "7", directed=True) != edge_key("7", "3", directed=True)


def test_graphview_degree_and_neighbors():
    nodes, edges = gi.ring_lattice(6, 2)  # a 6-cycle
    g = GraphView(nodes, edges)
    assert g.n_nodes == 6
    assert all(g.degree(n) == 2 for n in g.node_ids())
    assert g.next_node_id() == "6"


def test_reaction_returns_graphops_without_mutating_graph():
    nodes, edges = gi.seed_clique(3)
    before_nodes, before_edges = dict(nodes), dict(edges)
    g = GraphView(nodes, edges)
    rng = np.random.default_rng(0)
    ops = get_reaction("preferential_attachment").fn(
        rng, g, {"m": 2.0, "max_nodes": 60.0}, 1.0, 0.0)
    assert ops.node_add and ops.edge_add
    # pure: the input dicts are untouched
    assert nodes == before_nodes and edges == before_edges


@pytest.mark.parametrize("rule", ALL_RULES)
def test_rule_runs_end_to_end(rule):
    """Every rule builds and runs a composite without error and stays valid."""
    core = build_core()
    seeds = {
        "erdos_renyi": gi.seed_clique(2),
        "preferential_attachment": gi.seed_clique(3),
        "fitness_attachment": gi.seed_clique(3),
        "duplication_divergence": gi.seed_clique(3),
        "forest_fire": gi.seed_clique(3),
        "small_world_rewire": gi.ring_lattice(20, 4),
        "adaptive_rewire": gi.labelled_nodes(24, 4, seed=1),
        "activity_driven": gi.activity_nodes(24, seed=1),
    }
    nodes, edges = seeds[rule]
    doc = {
        "net": {"_type": "process", "address": "local:NetworkDynamicsProcess",
                "config": {"rule": rule, "rule_params": {}, "seed": 3},
                "interval": 1.0,
                "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
                "outputs": {"nodes": ["nodes"], "edges": ["edges"]}},
        "nodes": dict(nodes), "edges": dict(edges),
        "emitter": {"_type": "step", "address": "local:RAMEmitter",
                    "config": {"emit": {"nodes": "map[dynamic_node]",
                                        "edges": "map[dynamic_edge]",
                                        "time": "float"}},
                    "inputs": {"nodes": ["nodes"], "edges": ["edges"],
                               "time": ["global_time"]}},
    }
    sim = Composite({"state": doc}, core=core)
    sim.run(15.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 2
    final = rows[-1]
    # every edge references existing nodes (no dangling edges)
    for e in final["edges"].values():
        assert str(e["source"]) in final["nodes"]
        assert str(e["target"]) in final["nodes"]


def test_preferential_attachment_builds_hubs():
    """BA must produce a heavier-tailed degree distribution than ER."""
    core = build_core()

    def final_max_degree(rule):
        nodes, edges = gi.seed_clique(3)
        doc = {
            "net": {"_type": "process", "address": "local:NetworkDynamicsProcess",
                    "config": {"rule": rule,
                               "rule_params": {"m": 2.0, "p": 0.12,
                                               "max_nodes": 60.0}, "seed": 2},
                    "interval": 1.0,
                    "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
                    "outputs": {"nodes": ["nodes"], "edges": ["edges"]}},
            "nodes": dict(nodes), "edges": dict(edges),
            "emitter": {"_type": "step", "address": "local:RAMEmitter",
                        "config": {"emit": {"nodes": "map[dynamic_node]",
                                            "edges": "map[dynamic_edge]"}},
                        "inputs": {"nodes": ["nodes"], "edges": ["edges"]}},
        }
        sim = Composite({"state": doc}, core=core)
        sim.run(57.0)
        rows = gather_emitter_results(sim)[("emitter",)]
        m = graph_metrics(rows[-1]["nodes"], rows[-1]["edges"])
        return m["max_degree"], rows[-1]

    ba_max, ba_final = final_max_degree("preferential_attachment")
    er_max, _ = final_max_degree("erdos_renyi")
    assert ba_max > er_max, f"BA max degree {ba_max} !> ER max degree {er_max}"
    gamma = powerlaw_exponent(ba_final["nodes"], ba_final["edges"])
    assert gamma is None or 1.5 < gamma < 5.0


def test_small_world_rewire_conserves_node_and_edge_count():
    core = build_core()
    nodes, edges = gi.ring_lattice(24, 4)
    n0, e0 = len(nodes), len(edges)
    doc = {
        "net": {"_type": "process", "address": "local:NetworkDynamicsProcess",
                "config": {"rule": "small_world_rewire",
                           "rule_params": {"p": 0.8}, "seed": 5},
                "interval": 1.0,
                "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
                "outputs": {"nodes": ["nodes"], "edges": ["edges"]}},
        "nodes": dict(nodes), "edges": dict(edges),
        "emitter": {"_type": "step", "address": "local:RAMEmitter",
                    "config": {"emit": {"nodes": "map[dynamic_node]",
                                        "edges": "map[dynamic_edge]",
                                        "time": "float"}},
                    "inputs": {"nodes": ["nodes"], "edges": ["edges"],
                               "time": ["global_time"]}},
    }
    sim = Composite({"state": doc}, core=core)
    sim.run(30.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    final = rows[-1]
    assert len(final["nodes"]) == n0
    assert len(final["edges"]) == e0  # rewiring conserves edge count


def test_unknown_rule_raises():
    with pytest.raises(KeyError):
        get_reaction("not_a_rule")


def test_reproducible_with_seed():
    core = build_core()

    def run():
        nodes, edges = gi.seed_clique(3)
        doc = {
            "net": {"_type": "process", "address": "local:NetworkDynamicsProcess",
                    "config": {"rule": "preferential_attachment",
                               "rule_params": {"m": 2.0, "max_nodes": 30.0},
                               "seed": 42},
                    "interval": 1.0,
                    "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
                    "outputs": {"nodes": ["nodes"], "edges": ["edges"]}},
            "nodes": dict(nodes), "edges": dict(edges),
            "emitter": {"_type": "step", "address": "local:RAMEmitter",
                        "config": {"emit": {"edges": "map[dynamic_edge]"}},
                        "inputs": {"edges": ["edges"]}},
        }
        sim = Composite({"state": doc}, core=core)
        sim.run(25.0)
        return set(gather_emitter_results(sim)[("emitter",)][-1]["edges"].keys())

    assert run() == run()
