"""Seed-graph builders — the ``nodes`` / ``edges`` maps a composite starts from.

Growth rules (ER, BA, fitness, duplication, forest-fire) start from a tiny seed
(:func:`seed_clique`) and grow. Rewire/temporal rules (small-world, adaptive,
activity-driven) start from a fully-populated graph the rule then reconfigures in
place (:func:`ring_lattice`, :func:`labelled_nodes`, :func:`activity_nodes`).

Each builder returns ``(nodes, edges)`` dicts shaped for ``map[dynamic_node]`` /
``map[dynamic_edge]``.
"""
from __future__ import annotations

import numpy as np

from .reactions import edge_key, make_edge, make_node


def seed_clique(n: int = 3) -> tuple[dict, dict]:
    """A small connected seed (complete graph on ``n`` nodes) for growth rules."""
    nodes = {str(i): make_node(birth=0.0) for i in range(n)}
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            edges[edge_key(str(i), str(j))] = make_edge(str(i), str(j))
    return nodes, edges


def ring_lattice(n: int = 24, k: int = 4) -> tuple[dict, dict]:
    """Watts–Strogatz starting point: ``n`` nodes on a ring, each joined to its
    ``k`` nearest neighbours (``k`` even)."""
    nodes = {str(i): make_node(birth=0.0) for i in range(n)}
    edges = {}
    half = max(1, k // 2)
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges[edge_key(str(i), str(j))] = make_edge(str(i), str(j))
    return nodes, edges


def labelled_nodes(n: int = 40, k: int = 4, p_one: float = 0.5,
                   seed: int = 0) -> tuple[dict, dict]:
    """Adaptive-coevolution start: a ring lattice whose nodes carry a binary
    ``state`` (opinion) and a matching ``group`` label."""
    rng = np.random.default_rng(seed)
    nodes = {}
    for i in range(n):
        s = 1.0 if rng.random() < p_one else 0.0
        nodes[str(i)] = make_node(birth=0.0, state=s, group=str(int(s)))
    _, edges = ring_lattice(n, k)
    return nodes, edges


def activity_nodes(n: int = 40, a_min: float = 0.02, a_max: float = 0.9,
                   gamma: float = 2.1, seed: int = 0) -> tuple[dict, dict]:
    """Activity-driven start: ``n`` edgeless nodes with power-law activities
    ``a_i`` in ``[a_min, a_max]`` (exponent ``gamma``)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    # inverse-transform sample a bounded power law x^-gamma on [a_min, a_max]
    g = 1.0 - gamma
    lo, hi = a_min ** g, a_max ** g
    a = (lo + u * (hi - lo)) ** (1.0 / g)
    nodes = {str(i): make_node(birth=0.0, activity=float(a[i])) for i in range(n)}
    return nodes, {}
