"""Family-B composite generators — dynamics ON networks (SBML at the nodes).

A fixed ring of reactors, each running the Repressilator SBML model (via
``viva-copasi``/``basico``), coupled by diffusion of the protein species along
the edges. Two generators differ only in the diffusion coefficient ``D``:

- ``repressilator_network_coupled`` (D>0)   — coupling synchronises the oscillators.
- ``repressilator_network_uncoupled`` (D=0) — independent oscillators stay out of phase.

Each node is seeded at a different limit-cycle phase so the synchronisation (or
its absence) is visible from the start.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..core import register_all
from ..graph_init import ring_lattice
from ..sbml_nodes import repressilator_path, sample_phase_states

_SYNC_PARQUET = [{"address": "local:ParquetEmitter",
                  "paths": ["sync/sync_std", "sync/mean_conc"]}]
_TRACK = ["PX", "PY", "PZ"]  # diffuse the three repressilator proteins
DT = 1.0  # tick size (model time units); the Repressilator period is ~100.
# Keep dt small: explicit operator-split diffusion is unstable at larger dt×D.


def _build(*, D: float, n: int, seed: int):
    n = int(n)
    nodes, edges = ring_lattice(n, 2)  # an n-cycle
    phases = sample_phase_states(repressilator_path(), n, seed=int(seed))
    for i, nid in enumerate(sorted(nodes, key=int)):
        nodes[nid]["species"] = dict(phases[i])
    return {
        "reactors": {
            "_type": "process", "address": "local:NetworkSBMLReactors",
            "config": {"model_source": repressilator_path()}, "interval": DT,
            "inputs": {"nodes": ["nodes"]}, "outputs": {"nodes": ["nodes"]},
        },
        "transport": {
            "_type": "process", "address": "local:NetworkTransport",
            "config": {"D": float(D), "species": _TRACK}, "interval": DT,
            "inputs": {"nodes": ["nodes"], "edges": ["edges"]},
            "outputs": {"nodes": ["nodes"]},
        },
        "sync_metric": {
            "_type": "process", "address": "local:SpeciesSyncMetric",
            "config": {"species": "PX"}, "interval": DT,
            "inputs": {"nodes": ["nodes"]},
            "outputs": {"sync_std": ["sync", "sync_std"],
                        "mean_conc": ["sync", "mean_conc"]},
        },
        "nodes": nodes,
        "edges": edges,
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
    name="repressilator_network_coupled",
    description="A ring of Repressilator reactors (SBML at each node, via "
                "viva-copasi) coupled by protein diffusion along the edges — "
                "the oscillators synchronise.",
    parameters={
        "D": {"type": "float", "default": 0.3,
              "description": "inter-node diffusion coefficient"},
        "n": {"type": "float", "default": 9.0, "description": "number of nodes"},
        "seed": {"type": "float", "default": 1.0, "description": "phase seed"},
    },
    emitters=_SYNC_PARQUET,
    core_extensions=[register_all],
)
def repressilator_network_coupled(core=None, *, D=0.3, n=9.0, seed=1.0):
    return _build(D=D, n=n, seed=int(seed))


@composite_generator(
    name="repressilator_network_uncoupled",
    description="The same ring of Repressilator reactors with diffusion off "
                "(D=0) — independent oscillators that stay out of phase.",
    parameters={
        "n": {"type": "float", "default": 9.0, "description": "number of nodes"},
        "seed": {"type": "float", "default": 1.0, "description": "phase seed"},
    },
    emitters=_SYNC_PARQUET,
    core_extensions=[register_all],
)
def repressilator_network_uncoupled(core=None, *, n=9.0, seed=1.0):
    return _build(D=0.0, n=n, seed=int(seed))
