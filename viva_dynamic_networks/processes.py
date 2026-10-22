"""The generalized dynamic-network process.

:class:`NetworkDynamicsProcess` is a single, reusable process-bigraph ``Process``
that reconfigures a graph by applying one *reaction* from
:mod:`viva_dynamic_networks.reactions` per step. The rule is chosen by
``config['rule']``; every rule emits the same structural ``_add`` / ``_remove``
sentinels on the ``nodes`` and ``edges`` maps, so one process covers the whole
catalog and a new dynamic is just a new registered reaction — nothing here
changes.

Ports
-----
inputs  : ``nodes`` (``map[dynamic_node]``), ``edges`` (``map[dynamic_edge]``)
outputs : ``nodes`` (structural + attribute updates),
          ``edges`` (structural updates)
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

from .metrics import graph_metrics
from .reactions import GraphView, get_reaction


def _dedup_entries(entries):
    """Keep the first value for each key in a list of ``(key, value)`` pairs."""
    seen = {}
    for k, v in entries:
        if k not in seen:
            seen[k] = v
    return [[k, v] for k, v in seen.items()]


class NetworkDynamicsProcess(Process):
    """Apply one dynamic-network reaction per step to a bigraph-stored graph."""

    config_schema = {
        "rule": {"_type": "string", "_default": "preferential_attachment"},
        "rule_params": {"_type": "map", "_value": "float"},
        "seed": {"_type": "integer", "_default": 0},
        "directed": {"_type": "boolean", "_default": False},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        # Seeded generator → reproducible trajectories (advances across steps).
        self._rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self._time = 0.0
        # Fail fast on an unknown rule at construction, not mid-run.
        get_reaction(self.config["rule"])

    def inputs(self):
        return {"nodes": "map[dynamic_node]", "edges": "map[dynamic_edge]"}

    def outputs(self):
        return {"nodes": "map[dynamic_node]", "edges": "map[dynamic_edge]"}

    def update(self, state, interval):
        nodes = state.get("nodes", {}) or {}
        edges = state.get("edges", {}) or {}
        graph = GraphView(nodes, edges, directed=bool(self.config["directed"]))

        reaction = get_reaction(self.config["rule"])
        ops = reaction.fn(
            self._rng, graph, dict(self.config.get("rule_params", {}) or {}),
            interval, self._time)
        self._time += interval

        update: dict = {}

        nodes_u: dict = {}
        if ops.node_add:
            nodes_u["_add"] = _dedup_entries(ops.node_add)
        if ops.node_remove:
            nodes_u["_remove"] = list(dict.fromkeys(ops.node_remove))
        for nid, fields in ops.node_set.items():
            nodes_u[nid] = fields
        if nodes_u:
            update["nodes"] = nodes_u

        edges_u: dict = {}
        if ops.edge_add:
            edges_u["_add"] = _dedup_entries(ops.edge_add)
        if ops.edge_remove:
            edges_u["_remove"] = list(dict.fromkeys(ops.edge_remove))
        if edges_u:
            update["edges"] = edges_u

        return update


class NetworkMetrics(Process):
    """Derive scalar graph metrics each step → a persistable observable series.

    Reads the live ``nodes`` / ``edges`` maps and writes absolute scalar
    readings (so the ports are ``overwrite[float]`` — they are sensor values, not
    deltas). A ParquetEmitter on the ``metrics`` store makes a study's
    ``behavior_tests`` evaluable (e.g. final ``mean_degree`` or
    ``largest_component_fraction``). ``with_nx`` adds clustering (networkx).
    """

    config_schema = {
        "directed": {"_type": "boolean", "_default": False},
        "with_nx": {"_type": "boolean", "_default": True},
    }

    def inputs(self):
        return {"nodes": "map[dynamic_node]", "edges": "map[dynamic_edge]"}

    def outputs(self):
        return {
            "n_nodes": "overwrite[float]",
            "n_edges": "overwrite[float]",
            "mean_degree": "overwrite[float]",
            "max_degree": "overwrite[float]",
            "density": "overwrite[float]",
            "largest_component_fraction": "overwrite[float]",
            "clustering": "overwrite[float]",
        }

    def update(self, state, interval):
        m = graph_metrics(state.get("nodes", {}) or {},
                          state.get("edges", {}) or {},
                          with_nx=bool(self.config.get("with_nx", True)))
        return {
            "n_nodes": float(m["n_nodes"]),
            "n_edges": float(m["n_edges"]),
            "mean_degree": float(m["mean_degree"]),
            "max_degree": float(m["max_degree"]),
            "density": float(m["density"]),
            "largest_component_fraction": float(m["largest_component_fraction"]),
            "clustering": float(m["clustering"] or 0.0),
        }
