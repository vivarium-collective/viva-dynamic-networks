"""Shared canonical-run helper for the dynamic-networks showcase studies.

Each study's ``sims/run.py`` is a thin wrapper around :func:`run_study`: it names
the rule, its composite generator, parameters, and viz style, and this helper
runs the composite, renders the GIF + interactive animation into the study's
``viz/`` dir, computes the metric series the study's ``behavior_tests`` measure,
and records the run in the workspace run log.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

from process_bigraph import Composite, gather_emitter_results

from . import composites as _composites
from . import viz
from .core import build_core
from .metrics import graph_metrics, powerlaw_exponent

INVESTIGATION_SLUG = "dynamic-networks"


def run_study(
    *,
    rule: str,
    params: dict,
    n_steps: int,
    study_dir: Path,
    title: str,
    layout: str = "spring",
    color_by: str = "degree",
    subtitle: str = "",
    seed: int = 7,
) -> dict:
    """Run one showcase study end-to-end; return its computed metrics.

    Produces ``<study_dir>/viz/<rule>.gif`` (looping animation) and
    ``<study_dir>/viz/<rule>.html`` (interactive), records a run event, and
    returns a dict of final + series metrics for behavior-test verification.
    """
    study_dir = Path(study_dir).resolve()
    workspace_root = _find_workspace_root(study_dir)
    spec_id = f"viva_dynamic_networks.composites.family_a.{rule}"
    study_slug = study_dir.name

    run_id = uuid.uuid4().hex
    _record(workspace_root, {
        "run_id": run_id, "event": "started", "spec_id": spec_id,
        "label": study_slug, "started_at": time.time(), "status": "running",
        "n_steps": n_steps, "emitter": "ram", "origin": "canonical_run",
        "study_slug": study_slug, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        gen: Callable = getattr(_composites, rule)
        doc = gen(core, seed=float(seed), **params)
        sim = Composite({"state": doc}, core=core)
        sim.run(float(n_steps))
        rows = gather_emitter_results(sim)[("emitter",)]
        frames = viz.frames_from_rows(rows)

        # --- render the two viz artifacts ---
        viz_dir = study_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        positions = viz.compute_positions(frames, kind=layout)
        viz.render_gif(frames, positions, viz_dir / f"{rule}.gif",
                       title=title, color_by=color_by, fps=5)
        html = viz.figure_to_html(viz.evolution_figure(
            frames, positions, title=title, color_by=color_by, subtitle=subtitle))
        (viz_dir / f"{rule}.html").write_text(html, encoding="utf-8")

        # --- metrics the behavior tests read ---
        final = frames[-1]
        m = graph_metrics(final["nodes"], final["edges"], with_nx=True)
        m["powerlaw_exponent"] = powerlaw_exponent(final["nodes"], final["edges"])
        m["n_frames"] = len(frames)
        m["mean_degree_series"] = [
            graph_metrics(f["nodes"], f["edges"])["mean_degree"] for f in frames]
    except Exception:
        _record(workspace_root, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    _record(workspace_root, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": n_steps, "status": "completed"})
    return m


def _record(workspace_root: Path, event: dict) -> None:
    try:
        from vivarium_workbench.lib.run_log import append_run_event
        append_run_event(workspace_root, event)
    except Exception:
        pass  # run-log is best-effort; never fail a study over it


def _find_workspace_root(start: Path) -> Path:
    node = start
    for _ in range(8):
        if (node / "workspace.yaml").is_file():
            return node
        if node.parent == node:
            break
        node = node.parent
    return start
