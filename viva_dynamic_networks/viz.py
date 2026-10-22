"""Visualization layer for dynamic networks.

Three products, all driven by the same per-step ``(nodes, edges)`` frames:

- :func:`evolution_figure` — an **interactive Plotly** node-link animation
  (play/pause + step slider), the headline interactive artifact.
- :func:`render_gif` — a looping **GIF** of the network evolving (the headline
  static-shareable artifact every study ships).
- :class:`NetworkEvolution` — a process-bigraph ``Visualization`` Step that
  accumulates frames during a run and renders the Plotly animation, so the same
  figure appears **live in the dashboard's Visualizations tab**.

Colors come from the validated dataviz palette: a single-hue blue ramp for the
sequential encodings (degree, birth time — CVD-safe) and blue/orange for the two
coevolution opinions (categorical, well-separated).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from process_bigraph.visualization import Visualization

from .metrics import degree_sequence

# --- validated palette (dataviz skill) ------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
EDGE_COLOR = "rgba(82,81,78,0.28)"
# sequential blue ramp 100->700 (degree / birth)
BLUE_RAMP = [
    [0.0, "#cde2fb"], [0.2, "#9ec5f4"], [0.4, "#5598e7"],
    [0.6, "#2a78d6"], [0.8, "#184f95"], [1.0, "#0d366b"],
]
STATE_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # opinion 0 / 1 (categorical)

Frame = Dict[str, dict]
Pos = Dict[str, Tuple[float, float]]


# --------------------------------------------------------------------------- #
# Frames + layout
# --------------------------------------------------------------------------- #

def frames_from_rows(rows: List[dict]) -> List[Frame]:
    """Convert RAMEmitter rows (each ``{nodes, edges, ...}``) into frames."""
    out = []
    for r in rows:
        nodes = {k: dict(v) for k, v in (r.get("nodes") or {}).items()}
        edges = {k: dict(v) for k, v in (r.get("edges") or {}).items()}
        out.append({"nodes": nodes, "edges": edges})
    return out


def _union(frames: List[Frame]):
    nodes: dict = {}
    edges: dict = {}
    for f in frames:
        for nid, nv in f["nodes"].items():
            nodes.setdefault(nid, nv)
        for eid, ev in f["edges"].items():
            edges.setdefault(eid, ev)
    return nodes, edges


def compute_positions(frames: List[Frame], kind: str = "spring",
                      seed: int = 7) -> Pos:
    """Stable 2-D positions over the *union* of all nodes/edges across frames,
    so node locations don't jump between animation frames.

    ``kind='circular'`` arranges nodes by integer id on a ring (best for the
    small-world rewire study); ``'spring'`` uses a force layout on the aggregate
    graph (best for growth / temporal).
    """
    nodes, edges = _union(frames)
    ids = list(nodes.keys())
    if not ids:
        return {}
    if kind == "circular":
        def _key(x):
            try:
                return (0, int(x))
            except (TypeError, ValueError):
                return (1, x)
        ordered = sorted(ids, key=_key)
        n = len(ordered)
        return {nid: (math.cos(2 * math.pi * i / n),
                      math.sin(2 * math.pi * i / n))
                for i, nid in enumerate(ordered)}
    # spring layout via networkx on the aggregate graph
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(ids)
        for e in edges.values():
            g.add_edge(str(e["source"]), str(e["target"]))
        pos = nx.spring_layout(g, seed=seed, k=1.4 / math.sqrt(max(len(ids), 1)))
        return {nid: (float(p[0]), float(p[1])) for nid, p in pos.items()}
    except Exception:
        # deterministic circular fallback
        n = len(ids)
        return {nid: (math.cos(2 * math.pi * i / n),
                      math.sin(2 * math.pi * i / n))
                for i, nid in enumerate(ids)}


def _node_colors(frame: Frame, color_by: str):
    nodes, edges = frame["nodes"], frame["edges"]
    if color_by == "state":
        return [STATE_COLORS.get(int(round(nodes[n].get("state", 0.0))), "#999")
                for n in nodes], None
    if color_by == "birth":
        return [float(nodes[n].get("birth", 0.0)) for n in nodes], "birth"
    if color_by.startswith("species:"):
        sid = color_by.split(":", 1)[1]
        return [float((nodes[n].get("species") or {}).get(sid, 0.0))
                for n in nodes], sid
    # default: degree
    deg = degree_sequence(nodes, edges)
    return [deg.get(n, 0) for n in nodes], "degree"


# --------------------------------------------------------------------------- #
# Interactive Plotly animation
# --------------------------------------------------------------------------- #

def evolution_figure(frames: List[Frame], positions: Pos, *, title: str = "",
                     color_by: str = "degree", subtitle: str = ""):
    """Build an animated Plotly node-link figure (play/pause + slider)."""
    import plotly.graph_objects as go

    categorical = color_by == "state"

    def edge_trace(frame):
        xs, ys = [], []
        for e in frame["edges"].values():
            u, v = str(e["source"]), str(e["target"])
            if u in positions and v in positions:
                xs += [positions[u][0], positions[v][0], None]
                ys += [positions[u][1], positions[v][1], None]
        return go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                          line=dict(width=1.0, color=EDGE_COLOR),
                          showlegend=False)

    def node_trace(frame):
        nodes = frame["nodes"]
        xs = [positions[n][0] for n in nodes if n in positions]
        ys = [positions[n][1] for n in nodes if n in positions]
        ns = [n for n in nodes if n in positions]
        deg = degree_sequence(nodes, frame["edges"])
        colors, _ = _node_colors(frame, color_by)
        colors = [c for n, c in zip(nodes, colors) if n in positions]
        sizes = [8 + 2.2 * math.sqrt(deg.get(n, 0)) for n in ns]
        text = [f"node {n}<br>degree {deg.get(n,0)}" for n in ns]
        marker = dict(size=sizes, line=dict(width=1.0, color=SURFACE))
        if categorical:
            marker["color"] = colors
        else:
            marker.update(color=colors, colorscale=BLUE_RAMP, showscale=True,
                          colorbar=dict(title=color_by, thickness=12, len=0.6,
                                        tickfont=dict(color=INK_MUTED, size=10),
                                        title_font=dict(color=INK_MUTED, size=11)),
                          cmin=0)
        return go.Scatter(x=xs, y=ys, mode="markers", hoverinfo="text",
                          text=text, marker=marker, showlegend=False)

    if not frames:
        frames = [{"nodes": {}, "edges": {}}]

    plotly_frames = [
        go.Frame(data=[edge_trace(f), node_trace(f)], name=str(i))
        for i, f in enumerate(frames)
    ]
    fig = go.Figure(data=plotly_frames[0].data, frames=plotly_frames)

    steps = [dict(method="animate", label=str(i),
                  args=[[str(i)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for i in range(len(frames))]
    fig.update_layout(
        title=dict(text=title + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
                   font=dict(color=INK, size=18), x=0.02, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=20, r=20, t=70, b=50), height=600,
        xaxis=dict(visible=False, range=[-1.25, 1.25]),
        yaxis=dict(visible=False, range=[-1.25, 1.25],
                   scaleanchor="x", scaleratio=1),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.02, y=-0.04, xanchor="left",
            showactive=False, bgcolor="#ffffff", bordercolor="#d9d8d4",
            font=dict(color=INK, size=12),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=350, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))]),
            ])],
        sliders=[dict(active=0, x=0.14, len=0.84, y=-0.02,
                      currentvalue=dict(prefix="step ", font=dict(color=INK_MUTED)),
                      steps=steps)],
    )
    return fig


def figure_to_html(fig) -> str:
    return fig.to_html(full_html=True, include_plotlyjs="cdn",
                       default_width="100%")


# --------------------------------------------------------------------------- #
# GIF
# --------------------------------------------------------------------------- #

def render_gif(frames: List[Frame], positions: Pos, out_path, *, title: str = "",
               color_by: str = "degree", fps: int = 5, hold: int = 6) -> str:
    """Render a looping GIF of the network evolving to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio.v2 as imageio

    images = []
    for f in frames:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        # edges
        for e in f["edges"].values():
            u, v = str(e["source"]), str(e["target"])
            if u in positions and v in positions:
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color=(0.32, 0.32, 0.31, 0.3), lw=0.8, zorder=1)
        nodes = f["nodes"]
        ns = [n for n in nodes if n in positions]
        if ns:
            deg = degree_sequence(nodes, f["edges"])
            xs = [positions[n][0] for n in ns]
            ys = [positions[n][1] for n in ns]
            sizes = [30 + 22 * math.sqrt(deg.get(n, 0)) for n in ns]
            if color_by == "state":
                cols = [STATE_COLORS.get(int(round(nodes[n].get("state", 0.0))),
                                         "#999999") for n in ns]
                ax.scatter(xs, ys, s=sizes, c=cols, edgecolors=SURFACE,
                           linewidths=0.8, zorder=2)
            else:
                if color_by == "birth":
                    vals = [float(nodes[n].get("birth", 0.0)) for n in ns]
                elif color_by.startswith("species:"):
                    sid = color_by.split(":", 1)[1]
                    vals = [float((nodes[n].get("species") or {}).get(sid, 0.0))
                            for n in ns]
                else:
                    vals = [deg.get(n, 0) for n in ns]
                ax.scatter(xs, ys, s=sizes, c=vals, cmap="Blues",
                           vmin=0, edgecolors=SURFACE, linewidths=0.8, zorder=2)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, color=INK, fontsize=14, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        images.append(buf)
        plt.close(fig)
    if images:
        images += [images[-1]] * hold  # hold the final frame
    # imageio v2 wants per-frame duration in ms (fps is deprecated).
    imageio.mimsave(str(out_path), images, duration=1000.0 / max(fps, 1), loop=0)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Live dashboard Visualization
# --------------------------------------------------------------------------- #

class NetworkEvolution(Visualization):
    """Accumulate ``(nodes, edges)`` each tick; render the Plotly animation.

    Declared per-study in ``study.yaml`` ``visualizations:`` so it renders live
    in the dashboard's Visualizations tab and re-runs with the study.
    """

    config_schema = {
        **Visualization.config_schema,
        "layout": {"_type": "string", "_default": "spring"},
        "color_by": {"_type": "string", "_default": "degree"},
        "subtitle": {"_type": "string", "_default": ""},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._frames: List[Frame] = []

    def inputs(self):
        return {"nodes": "map[dynamic_node]", "edges": "map[dynamic_edge]"}

    def accumulate(self, state):
        nodes = {k: dict(v) for k, v in (state.get("nodes") or {}).items()}
        edges = {k: dict(v) for k, v in (state.get("edges") or {}).items()}
        self._frames.append({"nodes": nodes, "edges": edges})

    def render(self) -> str:
        positions = compute_positions(
            self._frames, kind=self.config.get("layout", "spring"))
        fig = evolution_figure(
            self._frames, positions,
            title=self.config.get("title", "") or "Network evolution",
            color_by=self.config.get("color_by", "degree"),
            subtitle=self.config.get("subtitle", ""))
        return figure_to_html(fig)


def concentration_timeseries_figure(frames: List[Frame], *, species: str = "PX",
                                    title: str = "", subtitle: str = ""):
    """Per-node concentration of ``species`` over time (shows (de)synchronisation).

    Each faint line is one node; the bold line is the across-node mean. When
    coupled reactors synchronise, the faint lines collapse onto the mean.
    """
    import numpy as np
    import plotly.graph_objects as go

    node_ids = sorted({n for f in frames for n in f["nodes"]},
                      key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x))
    t = list(range(len(frames)))
    fig = go.Figure()
    series = {nid: [] for nid in node_ids}
    for f in frames:
        for nid in node_ids:
            sp = (f["nodes"].get(nid, {}) or {}).get("species") or {}
            series[nid].append(float(sp.get(species, float("nan"))))
    for nid in node_ids:
        fig.add_trace(go.Scatter(x=t, y=series[nid], mode="lines",
                                 line=dict(width=1.0, color="rgba(42,120,214,0.33)"),
                                 hoverinfo="skip", showlegend=False))
    mean = np.nanmean(np.array([series[nid] for nid in node_ids]), axis=0)
    fig.add_trace(go.Scatter(x=t, y=mean, mode="lines", name="across-node mean",
                             line=dict(width=2.6, color="#0d366b")))
    fig.update_layout(
        title=dict(text=title + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
                   font=dict(color=INK, size=18), x=0.02, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=420,
        margin=dict(l=60, r=20, t=70, b=50),
        xaxis=dict(title="step", color=INK_MUTED, gridcolor="rgba(82,81,78,0.12)"),
        yaxis=dict(title=f"{species} concentration", color=INK_MUTED,
                   gridcolor="rgba(82,81,78,0.12)"),
        legend=dict(font=dict(color=INK_MUTED), x=0.98, xanchor="right", y=0.98))
    return fig
