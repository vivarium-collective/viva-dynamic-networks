"""Custom bigraph-schema types for dynamic networks.

A dynamic network is stored as two maps in the bigraph:

- ``nodes``: ``map[dynamic_node]`` — each node carries a few scalar attributes
  the rewiring rules read (``fitness`` for Bianconi–Barabási, ``activity`` for
  activity-driven, ``state`` for adaptive/voter coevolution) plus ``species``,
  a per-node concentration map reserved for the Family-B SBML-at-nodes phase.
- ``edges``: ``map[dynamic_edge]`` — each entry is an undirected (or directed)
  link keyed by a canonical ``"a#b"`` id, carrying ``source``/``target`` and a
  ``weight`` used as the transport conductance in the reaction–diffusion phase.

Reconfiguration happens through process-bigraph's structural ``_add`` / ``_remove``
sentinels on these two maps (see :mod:`viva_dynamic_networks.reactions`), which is
the native process-bigraph "reaction" mechanism — the same path division uses.
"""
from __future__ import annotations

# Record types declared with string field schemas (the spatio-flux convention),
# registered via the plural ``core.register_types`` so ``map[dynamic_node]`` and
# ``map[dynamic_edge]`` resolve as map value parameters.
DYNAMIC_NODE_TYPE = {
    "fitness": "float{1.0}",
    "birth": "float{0.0}",
    "group": "string",
    "activity": "float{0.0}",
    "state": "float{0.0}",
    # Reserved for the Family-B SBML-at-nodes phase: a well-mixed reactor's
    # species concentrations live here.
    "species": "map[float]",
}

DYNAMIC_EDGE_TYPE = {
    "source": "string",
    "target": "string",
    "weight": "float{1.0}",
}

TYPES_DICT = {
    "dynamic_node": DYNAMIC_NODE_TYPE,
    "dynamic_edge": DYNAMIC_EDGE_TYPE,
}


def register_types(core):
    """Register the ``dynamic_node`` / ``dynamic_edge`` record types.

    ``(core) -> core`` so it can be chained from ``build_core``; idempotent
    (re-registering the same spec is harmless).
    """
    core.register_types(TYPES_DICT)
    return core
