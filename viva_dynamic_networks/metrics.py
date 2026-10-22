"""Graph metrics computed from a ``(nodes, edges)`` snapshot.

Pure-python for the cheap structural quantities (degree, density, connected
components via union-find); networkx is imported lazily only for clustering and
shortest-path summaries, so the package imports fine without it. Study
``behavior_tests`` measure the series these functions produce (e.g. the final
degree-distribution exponent, mean degree, or largest-component fraction).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def degree_sequence(nodes: dict, edges: dict) -> Dict[str, int]:
    deg = {nid: 0 for nid in nodes}
    for e in edges.values():
        u, v = str(e.get("source")), str(e.get("target"))
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def _components(nodes: dict, edges: dict) -> List[set]:
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges.values():
        u, v = str(e.get("source")), str(e.get("target"))
        if u in parent and v in parent:
            union(u, v)
    groups: Dict[str, set] = {}
    for nid in nodes:
        groups.setdefault(find(nid), set()).add(nid)
    return list(groups.values())


def graph_metrics(nodes: dict, edges: dict, *, with_nx: bool = False) -> dict:
    """Return a dict of structural metrics for one graph snapshot.

    ``with_nx=True`` adds clustering + mean shortest path (largest component)
    via networkx; skipped (and the keys set to ``None``) if networkx is absent.
    """
    n = len(nodes)
    m = len(edges)
    deg = degree_sequence(nodes, edges)
    degs = np.array(list(deg.values()), dtype=float) if deg else np.array([0.0])

    comps = _components(nodes, edges) if n else []
    largest = max((len(c) for c in comps), default=0)

    out = {
        "n_nodes": n,
        "n_edges": m,
        "mean_degree": float(degs.mean()) if n else 0.0,
        "max_degree": int(degs.max()) if n else 0,
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "n_components": len(comps),
        "largest_component_fraction": (largest / n) if n else 0.0,
        "clustering": None,
        "mean_shortest_path": None,
    }
    if with_nx:
        try:
            import networkx as nx
            g = to_networkx(nodes, edges)
            out["clustering"] = float(nx.average_clustering(g)) if n else 0.0
            if g.number_of_nodes():
                giant = g.subgraph(max(nx.connected_components(g), key=len))
                if giant.number_of_nodes() > 1:
                    out["mean_shortest_path"] = float(
                        nx.average_shortest_path_length(giant))
        except Exception:
            pass
    return out


def degree_histogram(nodes: dict, edges: dict) -> Counter:
    return Counter(degree_sequence(nodes, edges).values())


def powerlaw_exponent(nodes: dict, edges: dict, kmin: int = 1) -> float | None:
    """Clauset-style MLE exponent of the degree distribution's tail,
    ``gamma = 1 + N / Σ ln(k / (kmin - 0.5))``. ``None`` if too few samples."""
    ks = [d for d in degree_sequence(nodes, edges).values() if d >= kmin]
    if len(ks) < 5:
        return None
    ks = np.array(ks, dtype=float)
    denom = np.sum(np.log(ks / (kmin - 0.5)))
    if denom <= 0:
        return None
    return float(1.0 + len(ks) / denom)


def to_networkx(nodes: dict, edges: dict):
    """Build a networkx ``Graph`` from the snapshot (lazy import)."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(nodes.keys())
    for e in edges.values():
        g.add_edge(str(e.get("source")), str(e.get("target")),
                   weight=float(e.get("weight", 1.0)))
    return g
