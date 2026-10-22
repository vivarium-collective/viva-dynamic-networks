"""Family-C composite generators — dynamics ON a *co-evolving* topology.

The synthesis of Family A and Family B: an SBML Repressilator runs at every node
with protein diffusion along the edges (Family B) **while a Family-A reaction
reconfigures the network underneath them** — the topology and the node state
co-evolve.

- ``growing_repressilator_network`` — preferential attachment adds reactor nodes
  over time; each newcomer joins at its own phase and entrains through the edges.
- ``rewiring_repressilator_network`` — a ring of reactors whose edges are
  continuously small-world-rewired while the oscillators run.

The growth/rewire process ticks on its own (slower) interval; the reactors,
transport, and metrics tick every unit. A newcomer node arrives with an empty
``species`` map and :class:`NetworkSBMLReactors` seeds it at a limit-cycle phase.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..core import register_all
from ..graph_init import ring_lattice, seed_clique
from ..sbml_nodes import repressilator_path, sample_phase_states

_PARQUET = [{"address": "local:ParquetEmitter",
             "paths": ["sync/sync_std", "metrics/n_nodes", "metrics/n_edges"]}]
_TRACK = ["PX", "PY", "PZ"]
REACTOR_DT = 1.0


def _seed_phases(nodes, seed):
    phases = sample_phase_states(repressilator_path(), len(nodes), seed=int(seed))
    for i, nid in enumerate(sorted(nodes, key=int)):
        nodes[nid]["species"] = dict(phases[i])
    return nodes


def _reactor_block(D):
    return {
        "reactors": {
            "_type": "process", "address": "local:NetworkSBMLReactors",
            "config": {"model_source": repressilator_path(),
                       "seed_new_nodes": True}, "interval": REACTOR_DT,
            "inputs": {"nodes": ["nodes"]}, "outputs": {"nodes": ["nodes"]},
        },
        "transport": {
            "_type": "process", "address": "local:NetworkTransport",
            "config": {"D": float(D), "species": _TRACK}, "interval": REACTOR_DT,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {"nodes": ["nodes"]},
        },
        "sync_metric": {
            "_type": "process", "address": "local:SpeciesSyncMetric",
            "config": {"species": "PX"}, "interval": REACTOR_DT,
            "inputs": {"nodes": ["nodes"]},
            "outputs": {"sync_std": ["sync", "sync_std"],
                        "mean_conc": ["sync", "mean_conc"]},
        },
        "topology_metric": {
            "_type": "process", "address": "local:NetworkMetrics",
            "config": {"with_nx": False}, "interval": REACTOR_DT,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {
                "n_nodes": ["metrics", "n_nodes"], "n_edges": ["metrics", "n_edges"],
                "mean_degree": ["metrics", "mean_degree"],
                "max_degree": ["metrics", "max_degree"],
                "density": ["metrics", "density"],
                "largest_component_fraction": ["metrics", "largest_component_fraction"],
                "clustering": ["metrics", "clustering"],
            },
        },
        "metrics": {},
        "sync": {},
        "emitter": {
            "_type": "step", "address": "local:RAMEmitter",
            "config": {"emit": {"nodes": "map[dynamic_node]",
                                "edges": "map[dynamic_edge]", "time": "float"}},
            "inputs": {"nodes": ["nodes"], "edges": ["edges"],
                       "time": ["global_time"]},
        },
    }


@composite_generator(
    name="growing_repressilator_network",
    description="Dynamics on a GROWING topology: forest-fire growth adds "
                "Repressilator reactor-nodes over time while protein diffusion "
                "couples them — newcomers join at their own phase and entrain. "
                "(Forest-fire growth gives the dense local coupling that lets the "
                "growing network synchronise at a numerically stable D.)",
    parameters={
        "D": {"type": "float", "default": 0.25,
              "description": "inter-node diffusion coefficient"},
        "max_nodes": {"type": "float", "default": 11.0,
                      "description": "final network size"},
        "grow_every": {"type": "float", "default": 10.0,
                       "description": "model-time units between node additions"},
        "seed": {"type": "float", "default": 1.0, "description": "phase/RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def growing_repressilator_network(core=None, *, D=0.25, max_nodes=11.0,
                                  grow_every=10.0, seed=1.0):
    nodes, edges = seed_clique(3)
    _seed_phases(nodes, seed)
    doc = {
        "growth": {
            "_type": "process", "address": "local:NetworkDynamicsProcess",
            "config": {"rule": "forest_fire",
                       "rule_params": {"p_forward": 0.5, "max_links": 8.0,
                                       "max_nodes": float(max_nodes)},
                       "seed": int(seed)},
            "interval": float(grow_every),
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {"nodes": ["nodes"], "edges": ["edges"]},
        },
        "nodes": nodes, "edges": edges,
        **_reactor_block(D),
    }
    return doc


@composite_generator(
    name="rewiring_repressilator_network",
    description="Dynamics on a REWIRING topology: a ring of Repressilator "
                "reactors whose edges are continuously small-world-rewired while "
                "the coupled oscillators run.",
    parameters={
        "D": {"type": "float", "default": 0.25,
              "description": "inter-node diffusion coefficient"},
        "p": {"type": "float", "default": 0.5,
              "description": "per-step rewiring probability"},
        "n": {"type": "float", "default": 12.0, "description": "number of nodes"},
        "seed": {"type": "float", "default": 1.0, "description": "phase/RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def rewiring_repressilator_network(core=None, *, D=0.25, p=0.5, n=12.0, seed=1.0):
    nodes, edges = ring_lattice(int(n), 4)
    _seed_phases(nodes, seed)
    doc = {
        "rewire": {
            "_type": "process", "address": "local:NetworkDynamicsProcess",
            "config": {"rule": "small_world_rewire",
                       "rule_params": {"p": float(p)}, "seed": int(seed)},
            "interval": REACTOR_DT,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {"nodes": ["nodes"], "edges": ["edges"]},
        },
        "nodes": nodes, "edges": edges,
        **_reactor_block(D),
    }
    return doc
