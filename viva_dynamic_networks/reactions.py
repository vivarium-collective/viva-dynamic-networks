"""The dynamic-network **reaction library**.

Each *rule* in this library is a process-bigraph **reaction**: given the current
graph it proposes a set of structural changes — nodes/edges to add or remove —
that :class:`~viva_dynamic_networks.processes.NetworkDynamicsProcess` emits as
``_add`` / ``_remove`` sentinels on the ``nodes`` and ``edges`` maps. That is the
native process-bigraph mechanism for reconfiguring a bigraph's structure at
runtime (the same path ``_divide`` uses for cell division).

The library is a **registry**: a rule is a function decorated with
:func:`register_reaction`. Adding a new dynamic is one decorated function — no
change to the process, the composites, or the dashboard. This is the
"generalized, reusable, extensible" core: one process, many pluggable reactions.

Every rule function has the signature::

    rule(rng, graph, params, interval, time) -> GraphOps

- ``rng``    — a ``numpy.random.Generator`` (seeded for reproducibility)
- ``graph``  — a :class:`GraphView` over the current ``nodes`` / ``edges`` maps
- ``params`` — the rule's numeric parameters (from the composite config)
- ``interval``/``time`` — the process interval and current global time

Rules are pure w.r.t. the graph (they read ``graph`` and return ops; they never
mutate it), so they are trivially unit-testable.

References (see the workspace Resources tab / ``structured_resources.yaml``):
ER — ``erdos1960evolution``; BA — ``barabasi1999emergence``; WS —
``watts1998smallworld``; fitness — ``bianconi2001competition``; duplication–
divergence — ``vazquez2003modeling`` / ``ispolatov2005duplication``; forest-fire
— ``leskovec2007graph``; adaptive — ``gross2008adaptive``; activity-driven —
``perra2012activity``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Structural-change payload + graph view
# --------------------------------------------------------------------------- #

NodeEntry = Tuple[str, dict]
EdgeEntry = Tuple[str, dict]


@dataclass
class GraphOps:
    """The changes a reaction proposes for one step.

    ``node_add`` / ``node_remove`` / ``edge_add`` / ``edge_remove`` are the
    *structural* changes (emitted as ``_add`` / ``_remove`` sentinels).
    ``node_set`` carries *attribute* updates to existing nodes (e.g. a voter
    opinion flip) as ``{node_id: {field: delta_or_value}}`` — a normal map write,
    never mixed into the same port as a structural sentinel by any single rule.
    """

    node_add: List[NodeEntry] = field(default_factory=list)
    node_remove: List[str] = field(default_factory=list)
    edge_add: List[EdgeEntry] = field(default_factory=list)
    edge_remove: List[str] = field(default_factory=list)
    node_set: Dict[str, dict] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.node_add or self.node_remove or self.edge_add
                    or self.edge_remove or self.node_set)


def edge_key(u: str, v: str, directed: bool = False) -> str:
    """Canonical id for an edge. Undirected keys are order-independent."""
    if directed:
        return f"{u}>{v}"
    a, b = sorted((str(u), str(v)))
    return f"{a}#{b}"


def make_node(birth: float = 0.0, fitness: float = 1.0, group: str = "0",
              activity: float = 0.0, state: float = 0.0) -> dict:
    """Build a ``dynamic_node`` record with sane defaults."""
    return {
        "fitness": float(fitness),
        "birth": float(birth),
        "group": str(group),
        "activity": float(activity),
        "state": float(state),
        "species": {},
    }


def make_edge(u: str, v: str, weight: float = 1.0) -> dict:
    return {"source": str(u), "target": str(v), "weight": float(weight)}


class GraphView:
    """A read-only view over the ``nodes`` / ``edges`` maps with cached adjacency.

    Rules query this (``degree``, ``neighbors``, ``has_edge``, ``node_ids``)
    instead of re-deriving adjacency themselves.
    """

    def __init__(self, nodes: dict, edges: dict, directed: bool = False):
        self.nodes: dict = nodes or {}
        self.edges: dict = edges or {}
        self.directed = directed
        self._adj: Dict[str, set] | None = None

    @property
    def adj(self) -> Dict[str, set]:
        if self._adj is None:
            adj: Dict[str, set] = {nid: set() for nid in self.nodes}
            for e in self.edges.values():
                u, v = str(e.get("source")), str(e.get("target"))
                adj.setdefault(u, set()).add(v)
                if not self.directed:
                    adj.setdefault(v, set()).add(u)
            self._adj = adj
        return self._adj

    def node_ids(self) -> List[str]:
        return list(self.nodes.keys())

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def degree(self, nid: str) -> int:
        return len(self.adj.get(nid, ()))

    def degrees(self) -> Dict[str, int]:
        return {nid: len(self.adj.get(nid, ())) for nid in self.nodes}

    def neighbors(self, nid: str) -> set:
        return self.adj.get(nid, set())

    def has_edge(self, u: str, v: str) -> bool:
        return edge_key(u, v, self.directed) in self.edges

    def next_node_id(self) -> str:
        """Next integer node id as a string (max existing + 1)."""
        best = -1
        for nid in self.nodes:
            try:
                best = max(best, int(nid))
            except (TypeError, ValueError):
                continue
        return str(best + 1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

RuleFn = Callable[[np.random.Generator, GraphView, dict, float, float], GraphOps]


@dataclass
class Reaction:
    name: str
    fn: RuleFn
    description: str
    params: Dict[str, float]  # name -> default
    family: str               # "growth" | "rewire" | "temporal"


REACTIONS: Dict[str, Reaction] = {}


def register_reaction(name: str, *, description: str = "",
                      params: Dict[str, float] | None = None,
                      family: str = "growth"):
    """Decorator registering a rule function under ``name``."""

    def deco(fn: RuleFn) -> RuleFn:
        REACTIONS[name] = Reaction(name, fn, description, params or {}, family)
        return fn

    return deco


def get_reaction(name: str) -> Reaction:
    if name not in REACTIONS:
        raise KeyError(
            f"unknown dynamic-network rule {name!r}; "
            f"available: {sorted(REACTIONS)}")
    return REACTIONS[name]


def _p(params: dict, key: str, default: float) -> float:
    try:
        v = params.get(key, default)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def _weighted_choice(rng, ids, weights, k):
    """Choose up to ``k`` distinct ids with the given (non-negative) weights."""
    ids = list(ids)
    k = min(int(k), len(ids))
    if k <= 0:
        return []
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones(len(ids))
    w = w / w.sum()
    idx = rng.choice(len(ids), size=k, replace=False, p=w)
    return [ids[i] for i in idx]


# --------------------------------------------------------------------------- #
# Family A — dynamics OF the network (topology reconfiguration)
# --------------------------------------------------------------------------- #

@register_reaction(
    "erdos_renyi",
    description="Random growth: add a node each step, link to each existing "
                "node with probability p (Poisson degrees; null baseline).",
    params={"p": 0.08, "max_nodes": 60.0},
    family="growth",
)
def erdos_renyi(rng, graph, params, interval, time):
    p = _p(params, "p", 0.08)
    max_nodes = int(_p(params, "max_nodes", 60.0))
    ops = GraphOps()
    if graph.n_nodes >= max_nodes:
        return ops
    new_id = graph.next_node_id()
    ops.node_add.append((new_id, make_node(birth=time)))
    for nid in graph.node_ids():
        if rng.random() < p:
            ops.edge_add.append(
                (edge_key(new_id, nid, graph.directed), make_edge(new_id, nid)))
    return ops


@register_reaction(
    "preferential_attachment",
    description="Barabási–Albert: add a node each step with m edges to existing "
                "nodes chosen with probability ∝ degree ('rich get richer').",
    params={"m": 2.0, "max_nodes": 60.0},
    family="growth",
)
def preferential_attachment(rng, graph, params, interval, time):
    m = int(_p(params, "m", 2.0))
    max_nodes = int(_p(params, "max_nodes", 60.0))
    ops = GraphOps()
    if graph.n_nodes >= max_nodes:
        return ops
    new_id = graph.next_node_id()
    ops.node_add.append((new_id, make_node(birth=time)))
    ids = graph.node_ids()
    if ids:
        degs = graph.degrees()
        weights = [degs[i] + 1.0 for i in ids]  # +1 so isolated seeds can attract
        for tgt in _weighted_choice(rng, ids, weights, m):
            ops.edge_add.append(
                (edge_key(new_id, tgt, graph.directed), make_edge(new_id, tgt)))
    return ops


@register_reaction(
    "fitness_attachment",
    description="Bianconi–Barabási: preferential attachment weighted by an "
                "intrinsic per-node fitness η — a fit latecomer can overtake "
                "older hubs.",
    params={"m": 2.0, "max_nodes": 60.0},
    family="growth",
)
def fitness_attachment(rng, graph, params, interval, time):
    m = int(_p(params, "m", 2.0))
    max_nodes = int(_p(params, "max_nodes", 60.0))
    ops = GraphOps()
    if graph.n_nodes >= max_nodes:
        return ops
    new_id = graph.next_node_id()
    eta = float(rng.random())  # fitness ~ U(0,1)
    ops.node_add.append((new_id, make_node(birth=time, fitness=eta)))
    ids = graph.node_ids()
    if ids:
        degs = graph.degrees()
        weights = [graph.nodes[i].get("fitness", 1.0) * (degs[i] + 1.0)
                   for i in ids]
        for tgt in _weighted_choice(rng, ids, weights, m):
            ops.edge_add.append(
                (edge_key(new_id, tgt, graph.directed), make_edge(new_id, tgt)))
    return ops


@register_reaction(
    "duplication_divergence",
    description="Genome-evolution rule: a new node copies a random node's "
                "edges, retaining each with probability q (optional parent link "
                "with probability p). Reproduces protein-network topology.",
    params={"q": 0.5, "p": 0.1, "max_nodes": 60.0},
    family="growth",
)
def duplication_divergence(rng, graph, params, interval, time):
    q = _p(params, "q", 0.5)
    p = _p(params, "p", 0.1)
    max_nodes = int(_p(params, "max_nodes", 60.0))
    ops = GraphOps()
    if graph.n_nodes >= max_nodes:
        return ops
    ids = graph.node_ids()
    new_id = graph.next_node_id()
    ops.node_add.append((new_id, make_node(birth=time)))
    if ids:
        u = str(rng.choice(ids))
        for w in graph.neighbors(u):
            if rng.random() < q:
                ops.edge_add.append(
                    (edge_key(new_id, w, graph.directed), make_edge(new_id, w)))
        if rng.random() < p:
            ops.edge_add.append(
                (edge_key(new_id, u, graph.directed), make_edge(new_id, u)))
    return ops


@register_reaction(
    "forest_fire",
    description="Forest-fire growth: a new node 'burns' recursively outward "
                "from a random ambassador, linking to a sampled subset of "
                "neighbors — yields densification + shrinking diameter.",
    params={"p_forward": 0.37, "max_nodes": 60.0, "max_links": 12.0},
    family="growth",
)
def forest_fire(rng, graph, params, interval, time):
    pf = _p(params, "p_forward", 0.37)
    max_nodes = int(_p(params, "max_nodes", 60.0))
    max_links = int(_p(params, "max_links", 12.0))
    ops = GraphOps()
    if graph.n_nodes >= max_nodes:
        return ops
    ids = graph.node_ids()
    new_id = graph.next_node_id()
    ops.node_add.append((new_id, make_node(birth=time)))
    if not ids:
        return ops
    ambassador = str(rng.choice(ids))
    burned = {ambassador}
    frontier = [ambassador]
    linked = [ambassador]
    while frontier and len(linked) < max_links:
        node = frontier.pop(0)
        nbrs = [n for n in graph.neighbors(node) if n not in burned]
        if not nbrs:
            continue
        # geometric-ish number of forward links
        k = int(rng.geometric(1.0 - pf)) - 1 if pf < 1.0 else len(nbrs)
        k = min(k, len(nbrs), max_links - len(linked))
        if k <= 0:
            continue
        chosen = [nbrs[i] for i in rng.choice(len(nbrs), size=k, replace=False)]
        for c in chosen:
            burned.add(c)
            linked.append(c)
            frontier.append(c)
    for tgt in linked:
        ops.edge_add.append(
            (edge_key(new_id, tgt, graph.directed), make_edge(new_id, tgt)))
    return ops


@register_reaction(
    "small_world_rewire",
    description="Watts–Strogatz: on a seeded ring lattice, rewire a random edge "
                "with probability p each step — a tidy ring dissolves into a "
                "small world.",
    params={"p": 0.12},
    family="rewire",
)
def small_world_rewire(rng, graph, params, interval, time):
    p = _p(params, "p", 0.12)
    ops = GraphOps()
    if graph.n_edges == 0 or graph.n_nodes < 3:
        return ops
    if rng.random() >= p:
        return ops
    eids = list(graph.edges.keys())
    eid = str(rng.choice(eids))
    e = graph.edges[eid]
    u, v = str(e["source"]), str(e["target"])
    # keep u, move v to a random non-neighbor (no self/multi-edge)
    candidates = [n for n in graph.node_ids()
                  if n != u and n not in graph.neighbors(u)]
    if not candidates:
        return ops
    w = str(rng.choice(candidates))
    ops.edge_remove.append(eid)
    ops.edge_add.append((edge_key(u, w, graph.directed), make_edge(u, w)))
    return ops


@register_reaction(
    "adaptive_rewire",
    description="Gross–Blasius coevolution: node opinions spread voter-style, "
                "and with rate w a node cuts a tie to a differing neighbor and "
                "reconnects to a like-minded one — topology and state co-adapt.",
    params={"w": 0.6},
    family="rewire",
)
def adaptive_rewire(rng, graph, params, interval, time):
    w = _p(params, "w", 0.6)
    ops = GraphOps()
    ids = graph.node_ids()
    if len(ids) < 3:
        return ops
    nid = str(rng.choice(ids))
    nbrs = list(graph.neighbors(nid))
    if not nbrs:
        return ops
    my_state = graph.nodes[nid].get("state", 0.0)
    differing = [n for n in nbrs if graph.nodes[n].get("state", 0.0) != my_state]
    if differing and rng.random() < w:
        # rewire: drop a differing tie, reconnect to a like-minded non-neighbor
        drop = str(rng.choice(differing))
        like = [n for n in ids
                if n != nid and n not in graph.neighbors(nid)
                and graph.nodes[n].get("state", 0.0) == my_state]
        if like:
            new = str(rng.choice(like))
            ops.edge_remove.append(edge_key(nid, drop, graph.directed))
            ops.edge_add.append(
                (edge_key(nid, new, graph.directed), make_edge(nid, new)))
    else:
        # voter update: adopt a random neighbor's opinion. `state` is an
        # additive float port, so emit the delta that lands on the new opinion.
        src = str(rng.choice(nbrs))
        new_state = graph.nodes[src].get("state", 0.0)
        if new_state != my_state:
            ops.node_set[nid] = {"state": float(new_state) - float(my_state)}
    return ops


@register_reaction(
    "activity_driven",
    description="Perra activity-driven temporal network: every step all edges "
                "reset, then each node fires with probability ∝ its activity and "
                "connects to m random others — a blinking contact sequence.",
    params={"m": 1.0},
    family="temporal",
)
def activity_driven(rng, graph, params, interval, time):
    m = int(_p(params, "m", 1.0))
    ops = GraphOps()
    ids = graph.node_ids()
    if len(ids) < 2:
        return ops
    ops.edge_remove.extend(graph.edges.keys())  # edges last one window only
    for nid in ids:
        a = graph.nodes[nid].get("activity", 0.0)
        if rng.random() < a:
            others = [n for n in ids if n != nid]
            k = min(m, len(others))
            for tgt in (others[i] for i in
                        rng.choice(len(others), size=k, replace=False)):
                ops.edge_add.append(
                    (edge_key(nid, tgt, graph.directed), make_edge(nid, tgt)))
    return ops
