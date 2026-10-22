"""Family B — dynamics ON networks: an SBML reactor at every node.

Two processes implement reaction–diffusion on a dynamic-networks graph by
operator splitting (reaction delta + diffusion delta summed each tick):

- :class:`NetworkSBMLReactors` places a **real COPASI/SBML model at every node**
  (the Repressilator by default), reusing ``viva-copasi``/``basico`` to load and
  integrate it. Each node's concentrations live in ``node['species']`` (the
  ``dynamic_node`` field reserved for this phase); the process re-syncs each
  model to that authoritative store, integrates one interval, and emits the
  per-species reaction delta.
- :class:`NetworkTransport` diffuses species along edges: a Fickian flux
  ``D · weight · (X_j − X_i)`` per edge, emitted as a per-node delta — so
  connected reactors couple and can synchronise.

Because both emit additive deltas into the same ``node['species']`` map, the
bigraph sums reaction + diffusion each tick — explicit reaction–diffusion on the
network, with the topology itself produced by a Family-A rule (or any fixed graph).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

import numpy as np
from process_bigraph import Process

_MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"


def repressilator_path() -> str:
    """Absolute path to the vendored Repressilator SBML (BIOMD0000000012)."""
    return str(_MODEL_DIR / "repressilator.xml")


@functools.lru_cache(maxsize=8)
def _species_mapping(model_source: str):
    """Return ``(sbml_ids, id2name, name2id, initial)`` for a model (cached)."""
    import basico
    dm = basico.load_model(model_source)
    sp = basico.get_species(model=dm)
    id2name = {row["sbml_id"]: name for name, row in sp.iterrows()}
    name2id = {name: sid for sid, name in id2name.items()}
    sbml_ids = list(id2name.keys())
    initial = {row["sbml_id"]: float(row["concentration"]) for _, row in sp.iterrows()}
    return sbml_ids, id2name, name2id, initial


def sample_phase_states(model_source: str, n: int, *, seed: int = 0,
                        span: float = 120.0) -> List[Dict[str, float]]:
    """Sample ``n`` species states at random phases along the limit cycle.

    Integrates the model once over ``span`` and reads concentrations at ``n``
    random times, so each node starts the network run at a different phase
    (otherwise identical reactors would be trivially synchronised from t=0).
    """
    import basico
    dm = basico.load_model(model_source)
    _, id2name, name2id, _ = _species_mapping(model_source)
    df = basico.run_time_course(start_time=0.0, duration=span,
                                values=list(np.linspace(0, span, 400)), model=dm)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(df) // 3, len(df), size=n)  # skip transient
    states = []
    for i in idx:
        row = df.iloc[int(i)]
        states.append({name2id[c]: float(row[c]) for c in df.columns if c in name2id})
    return states


class NetworkSBMLReactors(Process):
    """A COPASI/SBML reactor at every node; emits per-node reaction deltas."""

    config_schema = {
        "model_source": {"_type": "string", "_default": ""},
        "seed_new_nodes": {"_type": "boolean", "_default": True},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._src = self.config.get("model_source") or repressilator_path()
        self._models: dict = {}
        self._sbml_ids, self._id2name, self._name2id, self._initial = \
            _species_mapping(self._src)
        # A pool of limit-cycle phases so nodes that appear mid-run (e.g. added
        # by a Family-A growth reaction on a co-evolving topology) start at a
        # varied phase rather than all at the model's single default state.
        self._phase_pool = (sample_phase_states(self._src, 32, seed=0)
                            if self.config.get("seed_new_nodes", True) else [])

    def _seed_species(self, nid: str) -> dict:
        if self._phase_pool:
            try:
                return dict(self._phase_pool[int(nid) % len(self._phase_pool)])
            except (TypeError, ValueError):
                return dict(self._phase_pool[hash(nid) % len(self._phase_pool)])
        return dict(self._initial)

    def inputs(self):
        return {"nodes": "map[dynamic_node]"}

    def outputs(self):
        return {"nodes": "map[dynamic_node]"}

    def _model(self, nid: str):
        dm = self._models.get(nid)
        if dm is None:
            import basico
            dm = basico.load_model(self._src)
            self._models[nid] = dm
        return dm

    def update(self, state, interval):
        import basico
        nodes = state.get("nodes", {}) or {}
        out: dict = {}
        for nid, node in nodes.items():
            was_empty = not (node.get("species"))
            cur = dict(node.get("species") or {})
            if not cur:
                cur = self._seed_species(nid)  # newcomer → a varied phase
            dm = self._model(nid)
            for sid, val in cur.items():
                name = self._id2name.get(sid)
                if name is not None:
                    basico.set_species(name, initial_concentration=float(val),
                                       model=dm)
            df = basico.run_time_course(start_time=0.0, duration=float(interval),
                                        values=[float(interval)], model=dm)
            last = df.iloc[-1]
            delta = {}
            for sid in cur:
                name = self._id2name.get(sid)
                if name in df.columns:
                    new_val = float(last[name])
                    # For a newcomer the store holds no baseline yet, so emit the
                    # absolute value (store 0 → new_val); otherwise emit the delta.
                    delta[sid] = new_val if was_empty else new_val - cur[sid]
            out[nid] = {"species": delta}
        return {"nodes": out}


class NetworkTransport(Process):
    """Diffuse node species along edges (Fickian flux D·weight·(X_j − X_i))."""

    config_schema = {
        "D": {"_type": "float", "_default": 0.05},
        "species": {"_type": "list", "_element": "string"},  # empty = all
        "directed": {"_type": "boolean", "_default": False},
    }

    def inputs(self):
        return {"nodes": "map[dynamic_node]", "edges": "map[dynamic_edge]"}

    def outputs(self):
        return {"nodes": "map[dynamic_node]"}

    def update(self, state, interval):
        D = float(self.config.get("D", 0.05))
        if D <= 0:
            return {}
        which = list(self.config.get("species") or [])
        nodes = state.get("nodes", {}) or {}
        edges = state.get("edges", {}) or {}
        deltas: Dict[str, Dict[str, float]] = {}

        def spec(nid):
            return (nodes.get(nid, {}) or {}).get("species") or {}

        for e in edges.values():
            u, v = str(e.get("source")), str(e.get("target"))
            if u not in nodes or v not in nodes:
                continue
            w = float(e.get("weight", 1.0))
            su, sv = spec(u), spec(v)
            keys = which or (set(su) | set(sv))
            for s in keys:
                flux = D * w * (float(sv.get(s, 0.0)) - float(su.get(s, 0.0))) \
                    * float(interval)
                deltas.setdefault(u, {}).setdefault(s, 0.0)
                deltas.setdefault(v, {}).setdefault(s, 0.0)
                deltas[u][s] += flux
                deltas[v][s] -= flux
        if not deltas:
            return {}
        return {"nodes": {nid: {"species": d} for nid, d in deltas.items()}}


class SpeciesSyncMetric(Process):
    """Emit across-node synchrony + amplitude observables for one species.

    ``sync_std`` is the standard deviation of the tracked species' concentration
    across all nodes at the current step — it falls toward zero as coupled
    reactors synchronise. ``mean_conc`` is the across-node mean. Persisting these
    makes the Family-B studies' ``behavior_tests`` evaluable.
    """

    config_schema = {
        "species": {"_type": "string", "_default": "PX"},
    }

    def inputs(self):
        return {"nodes": "map[dynamic_node]"}

    def outputs(self):
        return {"sync_std": "overwrite[float]", "mean_conc": "overwrite[float]"}

    def update(self, state, interval):
        s = self.config.get("species", "PX")
        vals = [float((n.get("species") or {}).get(s, 0.0))
                for n in (state.get("nodes", {}) or {}).values()]
        if not vals:
            return {"sync_std": 0.0, "mean_conc": 0.0}
        arr = np.asarray(vals, dtype=float)
        return {"sync_std": float(arr.std()), "mean_conc": float(arr.mean())}
