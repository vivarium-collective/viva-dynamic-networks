"""Family-A composite generators — one per topology-reconfiguring rule.

Every generator returns a document with:

- ``network``       — :class:`NetworkDynamicsProcess` applying the rule's reaction
- ``metrics``       — :class:`NetworkMetrics` deriving scalar observables
- ``nodes`` / ``edges`` — the seed graph the rule starts from
- ``metrics`` store — the scalar series (persisted by the ParquetEmitter)
- ``emitter``       — a RAMEmitter capturing nodes/edges/time for the animation

The declared ``emitters=[ParquetEmitter(paths=…)]`` persists the metric series so
each study's ``behavior_tests`` are evaluable in the dashboard.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..core import register_all
from ..graph_init import activity_nodes, labelled_nodes, ring_lattice, seed_clique

# metric store paths the ParquetEmitter persists (every behavior-test observable)
_METRIC_PATHS = [
    "metrics/n_nodes", "metrics/n_edges", "metrics/mean_degree",
    "metrics/max_degree", "metrics/density",
    "metrics/largest_component_fraction", "metrics/clustering",
]
_PARQUET = [{"address": "local:ParquetEmitter", "paths": list(_METRIC_PATHS)}]


def _build(rule, nodes, edges, rule_params, seed):
    """Assemble the standard dynamic-network composite document."""
    return {
        "network": {
            "_type": "process",
            "address": "local:NetworkDynamicsProcess",
            "config": {"rule": rule, "rule_params": rule_params,
                       "seed": int(seed)},
            "interval": 1.0,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {"nodes": ["nodes"], "edges": ["edges"]},
        },
        "metrics_process": {
            "_type": "process",
            "address": "local:NetworkMetrics",
            "config": {},
            "interval": 1.0,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {
                "n_nodes": ["metrics", "n_nodes"],
                "n_edges": ["metrics", "n_edges"],
                "mean_degree": ["metrics", "mean_degree"],
                "max_degree": ["metrics", "max_degree"],
                "density": ["metrics", "density"],
                "largest_component_fraction":
                    ["metrics", "largest_component_fraction"],
                "clustering": ["metrics", "clustering"],
            },
        },
        "nodes": dict(nodes),
        "edges": dict(edges),
        "metrics": {},
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {"nodes": "map[dynamic_node]",
                                "edges": "map[dynamic_edge]",
                                "time": "float"}},
            "inputs": {"nodes": ["nodes"], "edges": ["edges"],
                       "time": ["global_time"]},
        },
    }


# --------------------------------------------------------------------------- #
# Growth rules
# --------------------------------------------------------------------------- #

@composite_generator(
    name="erdos_renyi",
    description="Random growth (Erdős–Rényi): a node arrives each step and links "
                "to each existing node with probability p. Poisson degrees; the "
                "null baseline every other rule is read against.",
    parameters={
        "p": {"type": "float", "default": 0.08, "description": "edge probability"},
        "max_nodes": {"type": "float", "default": 60.0,
                      "description": "growth cap"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def erdos_renyi(core=None, *, p=0.08, max_nodes=60.0, seed=0.0):
    nodes, edges = seed_clique(2)
    return _build("erdos_renyi", nodes, edges,
                  {"p": p, "max_nodes": max_nodes}, seed)


@composite_generator(
    name="preferential_attachment",
    description="Barabási–Albert: a node arrives each step with m edges to "
                "existing nodes chosen with probability ∝ degree — scale-free "
                "hubs emerge ('rich get richer').",
    parameters={
        "m": {"type": "float", "default": 2.0,
              "description": "edges per new node"},
        "max_nodes": {"type": "float", "default": 60.0,
                      "description": "growth cap"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def preferential_attachment(core=None, *, m=2.0, max_nodes=60.0, seed=0.0):
    nodes, edges = seed_clique(3)
    return _build("preferential_attachment", nodes, edges,
                  {"m": m, "max_nodes": max_nodes}, seed)


@composite_generator(
    name="fitness_attachment",
    description="Bianconi–Barabási: preferential attachment weighted by an "
                "intrinsic per-node fitness η — a fit latecomer can overtake "
                "older hubs.",
    parameters={
        "m": {"type": "float", "default": 2.0,
              "description": "edges per new node"},
        "max_nodes": {"type": "float", "default": 60.0,
                      "description": "growth cap"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def fitness_attachment(core=None, *, m=2.0, max_nodes=60.0, seed=0.0):
    nodes, edges = seed_clique(3)
    return _build("fitness_attachment", nodes, edges,
                  {"m": m, "max_nodes": max_nodes}, seed)


@composite_generator(
    name="duplication_divergence",
    description="Duplication–divergence: a new node copies a random node's edges, "
                "retaining each with probability q (optional parent link with "
                "probability p) — reproduces protein-network topology.",
    parameters={
        "q": {"type": "float", "default": 0.5,
              "description": "edge-retention probability"},
        "p": {"type": "float", "default": 0.1,
              "description": "parent-link probability"},
        "max_nodes": {"type": "float", "default": 60.0,
                      "description": "growth cap"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def duplication_divergence(core=None, *, q=0.5, p=0.1, max_nodes=60.0, seed=0.0):
    nodes, edges = seed_clique(3)
    return _build("duplication_divergence", nodes, edges,
                  {"q": q, "p": p, "max_nodes": max_nodes}, seed)


@composite_generator(
    name="forest_fire",
    description="Forest-fire growth: a new node burns recursively outward from a "
                "random ambassador — densification and shrinking diameter.",
    parameters={
        "p_forward": {"type": "float", "default": 0.37,
                      "description": "forward burn probability"},
        "max_nodes": {"type": "float", "default": 60.0,
                      "description": "growth cap"},
        "max_links": {"type": "float", "default": 12.0,
                      "description": "max links per arrival"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def forest_fire(core=None, *, p_forward=0.37, max_nodes=60.0, max_links=12.0,
                seed=0.0):
    nodes, edges = seed_clique(3)
    return _build("forest_fire", nodes, edges,
                  {"p_forward": p_forward, "max_nodes": max_nodes,
                   "max_links": max_links}, seed)


# --------------------------------------------------------------------------- #
# Rewire / temporal rules (start from a fully-populated graph)
# --------------------------------------------------------------------------- #

@composite_generator(
    name="small_world_rewire",
    description="Watts–Strogatz: on a ring lattice, rewire a random edge with "
                "probability p each step — a tidy ring dissolves into a small "
                "world (high clustering, short paths).",
    parameters={
        "p": {"type": "float", "default": 0.12,
              "description": "per-step rewiring probability"},
        "n": {"type": "float", "default": 28.0, "description": "number of nodes"},
        "k": {"type": "float", "default": 4.0,
              "description": "ring lattice degree"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def small_world_rewire(core=None, *, p=0.12, n=28.0, k=4.0, seed=0.0):
    nodes, edges = ring_lattice(int(n), int(k))
    return _build("small_world_rewire", nodes, edges, {"p": p}, seed)


@composite_generator(
    name="adaptive_rewire",
    description="Gross–Blasius coevolution: opinions spread voter-style while, "
                "with rate w, nodes cut ties to differing neighbours and "
                "reconnect to like-minded ones — topology and state co-adapt.",
    parameters={
        "w": {"type": "float", "default": 0.6, "description": "rewiring rate"},
        "n": {"type": "float", "default": 40.0, "description": "number of nodes"},
        "k": {"type": "float", "default": 4.0,
              "description": "ring lattice degree"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def adaptive_rewire(core=None, *, w=0.6, n=40.0, k=4.0, seed=0.0):
    nodes, edges = labelled_nodes(int(n), int(k), seed=int(seed))
    return _build("adaptive_rewire", nodes, edges, {"w": w}, seed)


@composite_generator(
    name="activity_driven",
    description="Perra activity-driven temporal network: each step all edges "
                "reset, then each node fires with probability ∝ its activity and "
                "links to m random others — a blinking contact sequence.",
    parameters={
        "m": {"type": "float", "default": 1.0,
              "description": "edges per active node"},
        "n": {"type": "float", "default": 40.0, "description": "number of nodes"},
        "seed": {"type": "float", "default": 0.0, "description": "RNG seed"},
    },
    emitters=_PARQUET,
    core_extensions=[register_all],
)
def activity_driven(core=None, *, m=1.0, n=40.0, seed=0.0):
    nodes, edges = activity_nodes(int(n), seed=int(seed))
    return _build("activity_driven", nodes, edges, {"m": m}, seed)
