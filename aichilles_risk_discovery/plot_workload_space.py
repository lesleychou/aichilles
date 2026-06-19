# aichilles_risk_discovery/plot_workload_space.py
"""
Visualise ADRS bug witnesses in the UMAP-projected workload parameter space.

Dimensionality is adaptive:
  1 numeric dim  → 1-D strip plot (jittered y)
  2 numeric dims → 2-D scatter (native, no projection)
  3+ numeric dims → 3-D scatter (UMAP/PCA to 3 components)

Usage:
  python plot_workload_space.py <results_dir> [--out_png path] [--out_html path]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

_EXCLUDED_PARAMS = frozenset({"seed"})

# Per-cluster styling: Okabe-Ito colorblind-safe palette (vivid against light-grey
# background), paired with distinct marker shapes so cluster identity is encoded
# both in color AND in shape (redundant — works in grayscale and for CVD users).
_CLUSTER_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
    "#999933",  # olive
]
_CLUSTER_MARKERS_MPL    = ["o", "s", "D", "^", "v", "P", "X", "*"]
_CLUSTER_MARKERS_PLY_2D = ["circle", "square", "diamond", "triangle-up",
                            "triangle-down", "cross", "x", "star"]
_CLUSTER_MARKERS_PLY_3D = ["circle", "square", "diamond", "cross", "x",
                            "circle-open", "square-open", "diamond-open"]

# Compact signature labels for legend (full names take too much horizontal space).
_SIG_ABBREV = {
    "correctness": "Correct",
    "scalab_time": "S_time",
    "scalab_mem":  "S_memory",
    "optimality":  "Opt",
}


def _abbrev_sigs(sigs: list[str]) -> str:
    return ", ".join(_SIG_ABBREV.get(s, s) for s in sigs)

# Clean white-background theme for paper publication.
_FIG_BG   = "white"
_PLOT_BG  = "white"
_GRID_COL = "#cccccc"
_TEXT_COL = "#1a1a1a"


def _group_style(idx: int) -> dict:
    """Return color + marker variants for cluster index ``idx``.

    Cluster identity is the primary distinction (one color+marker per cluster);
    the bug-type signature is conveyed in the legend text instead.
    """
    n_c = len(_CLUSTER_COLORS)
    n_m = len(_CLUSTER_MARKERS_MPL)
    return {
        "color":         _CLUSTER_COLORS[idx % n_c],
        "marker_mpl":    _CLUSTER_MARKERS_MPL[idx % n_m],
        "marker_ply_2d": _CLUSTER_MARKERS_PLY_2D[idx % n_m],
        "marker_ply_3d": _CLUSTER_MARKERS_PLY_3D[idx % n_m],
    }


def load_data(results_dir: Path) -> dict:
    """Load matrix_V, clusters, grammar, and config from results_dir.

    Returns dict with keys: matrix_v, clusters, numeric_params, cat_params, app, run_name.
    Exits with error if matrix_V.json or clusters.json is missing.
    """
    matrix_v_path = results_dir / "matrix_V.json"
    clusters_path  = results_dir / "clusters.json"
    config_path    = results_dir / "config.json"
    grammar_path   = results_dir / "grammar.json"

    if not matrix_v_path.exists():
        sys.exit(f"ERROR: {matrix_v_path} not found")
    if not clusters_path.exists():
        sys.exit(f"ERROR: {clusters_path} not found")

    matrix_v = json.loads(matrix_v_path.read_text())
    clusters  = json.loads(clusters_path.read_text())

    app      = "unknown"
    run_name = results_dir.name
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        app = cfg.get("app", "unknown")

    numeric_params: list[str] = []
    cat_params: list[str] = []
    if grammar_path.exists():
        grammar = json.loads(grammar_path.read_text())
        for section in ("grammar_config", "grammar_workload"):
            for p in grammar.get(section, []):
                name = p.get("name", "")
                if name in _EXCLUDED_PARAMS:
                    continue
                if p.get("type") in ("int", "float"):
                    numeric_params.append(name)
                elif p.get("type") == "str":
                    cat_params.append(name)
    elif matrix_v:
        numeric_params, cat_params = _infer_params(matrix_v[0])

    return {
        "matrix_v":       matrix_v,
        "clusters":       clusters,
        "numeric_params": numeric_params,
        "cat_params":     cat_params,
        "app":            app,
        "run_name":       run_name,
    }


def _infer_params(entry: dict) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical params from a single matrix_V entry."""
    combined = {**entry.get("c", {}), **entry.get("w", {})}
    numeric, cat = [], []
    for k, v in combined.items():
        if k in _EXCLUDED_PARAMS:
            continue
        if isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            numeric.append(k)
        elif isinstance(v, str):
            cat.append(k)
    return numeric, cat


def build_feature_matrix(matrix_v: list[dict], numeric_params: list[str]) -> np.ndarray:
    """Build (n_explored × n_dims) matrix. Each row = one matrix_V entry.

    Missing params default to 0.0.
    """
    rows = []
    for entry in matrix_v:
        combined = {**entry.get("c", {}), **entry.get("w", {})}
        rows.append([float(combined.get(p, 0.0)) for p in numeric_params])
    return np.array(rows, dtype=float)


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize each column to [0, 1].

    Returns (X_norm, mins, ranges). Constant columns (range=0) become all-zero
    to avoid division by zero; the corresponding ranges entry is set to 1.0 (sentinel)
    not 0.0.
    """
    mins   = X.min(axis=0)
    maxs   = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0.0] = 1.0
    return (X - mins) / ranges, mins, ranges


def apply_normalization(X: np.ndarray,
                        mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Apply pre-computed min-max normalization to new data.

    Values outside the training range will extrapolate beyond [0, 1].
    """
    return (X - mins) / ranges


def fit_projection(X_norm: np.ndarray,
                   n_components: int | None = None) -> tuple[object, str]:
    """Fit UMAP (or PCA fallback) on X_norm. Returns (fitted_model, method_name).

    n_components defaults to min(3, n_input_dims) — adaptive dimensionality:
      1 dim  → 1D (strip plot)
      2 dims → 2D (native scatter)
      3+ dims → 3D UMAP projection

    Falls back to PCA with a warning if umap-learn is not installed.
    """
    if n_components is None:
        n_components = min(3, X_norm.shape[1])

    try:
        import umap as umap_lib
        n_neighbors = max(2, min(15, len(X_norm) - 1))
        model = umap_lib.UMAP(n_components=n_components, random_state=42,
                               n_neighbors=n_neighbors, init="random")
        model.fit(X_norm)
        return model, "UMAP"
    except ImportError:
        print("WARNING: umap-learn not installed; falling back to PCA", file=sys.stderr)
        from sklearn.decomposition import PCA
        n_comp = min(n_components, X_norm.shape[1])
        model  = PCA(n_components=n_comp)
        model.fit(X_norm)
        return model, "PCA"


def _transform(model, X_norm: np.ndarray) -> np.ndarray:
    """Project X_norm using the fitted model."""
    return model.transform(X_norm)


def compute_axis_labels(bg_xy: np.ndarray, X: np.ndarray,
                        numeric_params: list[str], method: str) -> list[str]:
    """Label each projected axis with the most-correlated workload parameter.

    Each param is assigned to at most one axis (greedy, in axis order) so that
    all axes show different param names. Falls back to ``"<method>-N"`` when
    no eligible param remains (e.g. fewer params than axes, or all columns
    constant).
    """
    n_dims = bg_xy.shape[1]
    labels = []
    used: set[str] = set()
    for d in range(n_dims):
        proj = bg_xy[:, d]
        best_name, best_r = None, -1.0
        for j, name in enumerate(numeric_params):
            if name in used:
                continue
            col = X[:, j]
            if col.std() < 1e-10 or proj.std() < 1e-10:
                continue
            r = float(abs(np.corrcoef(proj, col)[0, 1]))
            if r > best_r:
                best_r, best_name = r, name
        if best_name:
            used.add(best_name)
            labels.append(best_name)
        else:
            labels.append(f"{method}-{d + 1}")
    return labels


def convex_hull_vertices(xy: np.ndarray) -> np.ndarray | None:
    """Return closed hull vertex array (shape (k+1, 2)) for 2-D polygon plotting.

    Returns None if < 3 points or hull computation fails.
    Only meaningful for 2-D inputs.
    """
    if len(xy) < 3:
        return None
    try:
        hull  = ConvexHull(xy)
        verts = xy[hull.vertices]
        return np.vstack([verts, verts[0]])  # close the polygon
    except Exception:
        return None


def project_witnesses(clusters: list[dict],
                      numeric_params: list[str],
                      cat_params: list[str],
                      mins: np.ndarray,
                      ranges: np.ndarray,
                      model,
                      literal_param: str | None = None,
                      lit_mins: np.ndarray | None = None,
                      lit_ranges: np.ndarray | None = None) -> list[dict]:
    """Project bug witnesses into N-D space using the already-fitted model.

    If literal_param is set, that param's (normalised) values are appended as
    the last coordinate column instead of being projected by the model.

    Returns list of group dicts:
      {"label", "signatures", "size", "coords" (n×D), "params" (list[dict]), "cat_values" (list[str])}
    Clusters with no witnesses are silently skipped.
    """
    groups = []
    for cluster in clusters:
        witnesses = cluster.get("witnesses", [])
        if not witnesses:
            continue
        rows, params_list, cat_values = [], [], []
        for w_entry in witnesses:
            combined = {**w_entry.get("c", {}), **w_entry.get("w", {})}
            rows.append([float(combined.get(p, 0.0)) for p in numeric_params])
            params_list.append(combined)
            cat_values.append(combined.get(cat_params[0], "") if cat_params else "")
        X_w_norm = apply_normalization(np.array(rows, dtype=float), mins, ranges)
        coords = _transform(model, X_w_norm)
        if literal_param is not None:
            lit_col = np.array([[float(p.get(literal_param, 0.0))] for p in params_list],
                               dtype=float)
            lit_col_norm = apply_normalization(lit_col, lit_mins, lit_ranges)
            coords = np.hstack([coords, lit_col_norm])
        groups.append({
            "label":      cluster.get("trigger_func", f"cluster_{len(groups)}"),
            "signatures": cluster.get("signatures", []),
            "size":       cluster.get("size", len(witnesses)),
            "coords":     coords,
            "params":     params_list,
            "cat_values": cat_values,
        })
    return groups


def plot_png(bg_xy: np.ndarray,
             witness_groups: list[dict],
             cat_param: str | None,
             method: str,
             app: str,
             run_name: str,
             out_path: Path,
             axis_labels: list[str] | None = None) -> None:
    """Save a static PNG. Layout adapts to the number of projected dimensions (1, 2, or 3)."""
    n_dims = bg_xy.shape[1]
    if axis_labels is None:
        axis_labels = [f"{method}-{i + 1}" for i in range(n_dims)]

    if n_dims == 3:
        _plot_png_3d(bg_xy, witness_groups, cat_param, method, app, run_name, out_path, axis_labels)
    elif n_dims == 1:
        _plot_png_1d(bg_xy, witness_groups, method, app, run_name, out_path, axis_labels)
    else:
        _plot_png_2d(bg_xy, witness_groups, cat_param, method, app, run_name, out_path, axis_labels)


def _plot_png_2d(bg_xy, witness_groups, cat_param, method, app, run_name, out_path, axis_labels):
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=_FIG_BG)
    ax.set_facecolor(_PLOT_BG)
    ax.scatter(bg_xy[:, 0], bg_xy[:, 1],
               c="#a8d5a2", s=110, alpha=0.45, zorder=1,
               label="explored (not-bug)", linewidths=0)
    for i, group in enumerate(witness_groups):
        style  = _group_style(i)
        color  = style["color"]
        coords = group["coords"]
        label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=color, s=200, zorder=3, marker=style["marker_mpl"],
                   edgecolors="white", linewidths=1.2, label=label)
        hull_verts = convex_hull_vertices(coords)
        if hull_verts is not None:
            ax.fill(hull_verts[:, 0], hull_verts[:, 1], alpha=0.10, color=color, zorder=2)
            ax.plot(hull_verts[:, 0], hull_verts[:, 1],
                    color=color, linewidth=1.8, zorder=2, linestyle="--")
    ax.set_xlabel(axis_labels[0], color=_TEXT_COL, fontsize=24, labelpad=8)
    ax.set_ylabel(axis_labels[1], color=_TEXT_COL, fontsize=24, labelpad=8)
    ax.tick_params(colors=_TEXT_COL, labelsize=18)
    ax.grid(True, color=_GRID_COL, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(_GRID_COL)
        s.set_linewidth(0.8)
    ax.legend(loc="best", fontsize=18, framealpha=0.95,
              edgecolor=_GRID_COL, fancybox=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0, facecolor=_FIG_BG)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor=_FIG_BG)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")
    print(f"Saved PDF: {pdf_path}")


def _plot_png_3d(bg_xy, witness_groups, cat_param, method, app, run_name, out_path, axis_labels):
    """3D scatter with tight whitespace.

    matplotlib's 3D axes reserve large "dead corners" around the rotated cube
    and ``bbox_inches="tight"`` does not reliably catch rotated z-axis labels.
    The strategy here is:
      - choose a figure aspect close to what the rendered 3D cube actually
        occupies (wider than tall after labels are drawn),
      - place the axes with a fixed-margin ``set_position`` (no heuristic),
      - scale the 3D cube inside the axes via ``set_box_aspect(zoom=...)``,
      - save without ``bbox_inches="tight"`` so the figsize is the exact crop.
    Margins below were chosen so all labels (incl. the rotated z-label) fit
    inside the figure boundary without leaving extra whitespace.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection
    fig = plt.figure(figsize=(9, 6.2), facecolor=_FIG_BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_PLOT_BG)

    # Reserve just enough room around the axes for the rendered labels.
    # left=4%, bottom=10% (x-label + ticks), right margin = 12% (z-label).
    ax.set_position([0.04, 0.10, 0.84, 0.88])
    try:
        ax.set_box_aspect(None, zoom=1.25)  # matplotlib ≥ 3.6
    except TypeError:
        pass

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(_PLOT_BG)
        pane.set_edgecolor(_GRID_COL)
    ax.xaxis._axinfo["grid"]["color"] = _GRID_COL
    ax.yaxis._axinfo["grid"]["color"] = _GRID_COL
    ax.zaxis._axinfo["grid"]["color"] = _GRID_COL

    ax.scatter(bg_xy[:, 0], bg_xy[:, 1], bg_xy[:, 2],
               c="#a8d5a2", s=100, alpha=0.40,
               label="explored (not-bug)", linewidths=0)
    for i, group in enumerate(witness_groups):
        style  = _group_style(i)
        color  = style["color"]
        coords = group["coords"]
        label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=color, s=170, marker=style["marker_mpl"],
                   edgecolors="white", linewidths=1.0, label=label, depthshade=True)

    # Small labelpad keeps the axis name tight against the axis.
    ax.set_xlabel(axis_labels[0], labelpad=4, color=_TEXT_COL, fontsize=22)
    ax.set_ylabel(axis_labels[1], labelpad=4, color=_TEXT_COL, fontsize=22)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(axis_labels[2], labelpad=8, color=_TEXT_COL, fontsize=22, rotation=90)
    # Negative pad pulls tick labels close to the axis.
    ax.tick_params(axis="x", colors=_TEXT_COL, labelsize=15, pad=-2)
    ax.tick_params(axis="y", colors=_TEXT_COL, labelsize=15, pad=-2)
    ax.tick_params(axis="z", colors=_TEXT_COL, labelsize=15, pad=0)
    # Compact legend so it doesn't add vertical whitespace at the top.
    ax.legend(loc="upper left", fontsize=14, framealpha=0.95,
              edgecolor=_GRID_COL, fancybox=False,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # No bbox_inches="tight" — figsize × set_position determines the exact
    # crop so the rotated z-label always lives inside the figure boundary.
    fig.savefig(out_path, dpi=200, facecolor=_FIG_BG)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, facecolor=_FIG_BG)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")
    print(f"Saved PDF: {pdf_path}")


def _plot_png_1d(bg_xy, witness_groups, method, app, run_name, out_path, axis_labels):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=_FIG_BG)
    ax.set_facecolor(_PLOT_BG)
    jitter = rng.uniform(-0.1, 0.1, len(bg_xy))
    ax.scatter(bg_xy[:, 0], jitter,
               c="#a8d5a2", s=110, alpha=0.45, zorder=1,
               label="explored (not-bug)", linewidths=0)
    for i, group in enumerate(witness_groups):
        style  = _group_style(i)
        color  = style["color"]
        coords = group["coords"]
        label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
        jitter_w = rng.uniform(-0.05, 0.05, len(coords))
        ax.scatter(coords[:, 0], jitter_w,
                   c=color, s=200, zorder=3, marker=style["marker_mpl"],
                   edgecolors="white", linewidths=1.2, label=label)
    ax.set_xlabel(axis_labels[0], color=_TEXT_COL, fontsize=24, labelpad=8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(_GRID_COL)
    ax.tick_params(colors=_TEXT_COL, labelsize=18)
    ax.legend(loc="best", fontsize=18, framealpha=0.95,
              edgecolor=_GRID_COL, fancybox=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0, facecolor=_FIG_BG)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor=_FIG_BG)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")
    print(f"Saved PDF: {pdf_path}")


def plot_html(bg_xy: np.ndarray,
              bg_params: list[dict],
              witness_groups: list[dict],
              cat_param: str | None,
              method: str,
              app: str,
              run_name: str,
              out_path: Path,
              axis_labels: list[str] | None = None) -> None:
    """Save an interactive HTML. Skips with a warning if plotly is not installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("WARNING: plotly not installed; skipping HTML output", file=sys.stderr)
        return

    n_dims = bg_xy.shape[1]
    if axis_labels is None:
        axis_labels = [f"{method}-{i + 1}" for i in range(n_dims)]

    _FONT   = dict(family="Arial, sans-serif", size=14, color=_TEXT_COL)
    _LEGEND = dict(font=dict(size=13), bgcolor="rgba(255,255,255,0.9)",
                   bordercolor="#cccccc", borderwidth=1)

    fig = go.Figure()
    hover_bg = ["<br>".join(f"<b>{k}</b>: {v}" for k, v in p.items()) for p in bg_params]

    if n_dims == 3:
        _html_add_traces_3d(fig, go, bg_xy, hover_bg, witness_groups)
        fig.update_layout(
            template="simple_white",
            paper_bgcolor=_FIG_BG,
            font=_FONT,
            scene=dict(
                bgcolor=_PLOT_BG,
                xaxis=dict(title=dict(text=axis_labels[0], font=dict(size=14)),
                           backgroundcolor=_PLOT_BG,
                           gridcolor=_GRID_COL, linecolor=_GRID_COL),
                yaxis=dict(title=dict(text=axis_labels[1], font=dict(size=14)),
                           backgroundcolor=_PLOT_BG,
                           gridcolor=_GRID_COL, linecolor=_GRID_COL),
                zaxis=dict(title=dict(text=axis_labels[2], font=dict(size=14)),
                           backgroundcolor=_PLOT_BG,
                           gridcolor=_GRID_COL, linecolor=_GRID_COL),
            ),
            legend=_LEGEND,
        )
    elif n_dims == 1:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.1, 0.1, len(bg_xy))
        fig.add_trace(go.Scatter(
            x=bg_xy[:, 0], y=jitter, mode="markers",
            marker=dict(color="#74c476", size=7, opacity=0.4),
            text=hover_bg, hoverinfo="text", name="explored (not-bug)",
        ))
        for i, group in enumerate(witness_groups):
            style  = _group_style(i)
            color  = style["color"]
            coords = group["coords"]
            label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
            hover  = ["<br>".join(f"<b>{k}</b>: {v}" for k, v in p.items())
                      for p in group["params"]]
            jitter_w = rng.uniform(-0.05, 0.05, len(coords))
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=jitter_w, mode="markers",
                marker=dict(color=color, size=11, symbol=style["marker_ply_2d"],
                            line=dict(color="white", width=1)),
                text=hover, hoverinfo="text", name=label, legendgroup=f"cluster_{i}",
            ))
        fig.update_layout(
            template="simple_white",
            paper_bgcolor=_FIG_BG, plot_bgcolor=_PLOT_BG,
            font=_FONT,
            xaxis=dict(title=dict(text=axis_labels[0], font=dict(size=14)),
                       gridcolor=_GRID_COL, linecolor=_GRID_COL),
            yaxis=dict(visible=False),
            legend=_LEGEND,
        )
    else:  # 2D
        _html_add_traces_2d(fig, go, bg_xy, hover_bg, witness_groups)
        fig.update_layout(
            template="simple_white",
            paper_bgcolor=_FIG_BG, plot_bgcolor=_PLOT_BG,
            font=_FONT,
            xaxis=dict(title=dict(text=axis_labels[0], font=dict(size=14)),
                       gridcolor=_GRID_COL, linecolor=_GRID_COL),
            yaxis=dict(title=dict(text=axis_labels[1], font=dict(size=14)),
                       gridcolor=_GRID_COL, linecolor=_GRID_COL),
            legend=_LEGEND,
        )

    fig.write_html(str(out_path))
    print(f"Saved HTML: {out_path}")


def _html_add_traces_2d(fig, go, bg_xy, hover_bg, witness_groups):
    fig.add_trace(go.Scatter(
        x=bg_xy[:, 0], y=bg_xy[:, 1],
        mode="markers",
        marker=dict(color="#74c476", size=5, opacity=0.3),
        text=hover_bg, hoverinfo="text",
        name="explored (not-bug)",
    ))
    for i, group in enumerate(witness_groups):
        style  = _group_style(i)
        color  = style["color"]
        coords = group["coords"]
        label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
        hover  = ["<br>".join(f"<b>{k}</b>: {v}" for k, v in p.items())
                  for p in group["params"]]
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(color=color, size=11, symbol=style["marker_ply_2d"],
                        line=dict(color="white", width=1)),
            text=hover, hoverinfo="text",
            name=label,
            legendgroup=f"cluster_{i}",
        ))
        hull_verts = convex_hull_vertices(coords)
        if hull_verts is not None:
            fig.add_trace(go.Scatter(
                x=hull_verts[:, 0], y=hull_verts[:, 1],
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="toself", fillcolor=color, opacity=0.08,
                showlegend=False,
                legendgroup=f"cluster_{i}",
                hoverinfo="skip",
            ))


def _html_add_traces_3d(fig, go, bg_xy, hover_bg, witness_groups):
    fig.add_trace(go.Scatter3d(
        x=bg_xy[:, 0], y=bg_xy[:, 1], z=bg_xy[:, 2],
        mode="markers",
        marker=dict(color="#74c476", size=3, opacity=0.25),
        text=hover_bg, hoverinfo="text",
        name="explored (not-bug)",
    ))
    for i, group in enumerate(witness_groups):
        style  = _group_style(i)
        color  = style["color"]
        coords = group["coords"]
        label  = f"Bugs ({_abbrev_sigs(group['signatures'])})"
        hover  = ["<br>".join(f"<b>{k}</b>: {v}" for k, v in p.items())
                  for p in group["params"]]
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(color=color, size=8, symbol=style["marker_ply_3d"],
                        line=dict(color="white", width=0.5)),
            text=hover, hoverinfo="text",
            name=label,
            legendgroup=f"cluster_{i}",
        ))


DEFAULT_FIG_DIR = Path(__file__).parent / "plot" / "figs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise bug witnesses in UMAP-projected workload space.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out_png",  type=Path, default=None,
                        help=f"PNG output path (default: {DEFAULT_FIG_DIR}/<run_name>.png; "
                             ".pdf is saved alongside)")
    parser.add_argument("--out_html", type=Path, default=None,
                        help=f"HTML output path (default: {DEFAULT_FIG_DIR}/<run_name>.html)")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Projection dims: 1, 2, or 3 (default: min(3, n_numeric_params))")
    parser.add_argument("--literal_axis", type=str, default=None,
                        help="Use this param as a literal (non-projected) axis; "
                             "UMAP fills the remaining dims from all other params")
    args = parser.parse_args()

    data = load_data(args.results_dir)
    if not data["numeric_params"]:
        sys.exit("ERROR: no numeric params found after excluding seed")

    n_components = args.n_components or min(3, len(data["numeric_params"]))
    n_components = max(1, min(3, n_components))  # clamp to [1, 3]

    # ── literal-axis mode ────────────────────────────────────────────────────
    literal = args.literal_axis
    if literal and literal not in data["numeric_params"]:
        sys.exit(f"ERROR: --literal_axis '{literal}' not found in numeric params "
                 f"{data['numeric_params']}")

    if literal:
        proj_params  = [p for p in data["numeric_params"] if p != literal]
        n_proj       = max(1, n_components - 1)   # one slot taken by literal
        X            = build_feature_matrix(data["matrix_v"], proj_params)
        X_norm, mins, ranges = normalize_features(X)
        model, method = fit_projection(X_norm, n_proj)
        bg_proj      = _transform(model, X_norm)
        X_lit        = build_feature_matrix(data["matrix_v"], [literal])
        X_lit_norm, lit_mins, lit_ranges = normalize_features(X_lit)
        bg_xy        = np.hstack([bg_proj, X_lit_norm])
        proj_labels  = compute_axis_labels(bg_proj, X, proj_params, method)
        axis_labels  = proj_labels + [literal]
        witness_groups = project_witnesses(
            data["clusters"], proj_params, data["cat_params"],
            mins, ranges, model,
            literal_param=literal, lit_mins=lit_mins, lit_ranges=lit_ranges,
        )
    # ── standard projection mode ─────────────────────────────────────────────
    else:
        X                    = build_feature_matrix(data["matrix_v"], data["numeric_params"])
        X_norm, mins, ranges = normalize_features(X)
        model, method        = fit_projection(X_norm, n_components)
        bg_xy                = _transform(model, X_norm)
        axis_labels          = compute_axis_labels(bg_xy, X, data["numeric_params"], method)
        witness_groups       = project_witnesses(
            data["clusters"], data["numeric_params"], data["cat_params"],
            mins, ranges, model,
        )

    print(f"[axes] {' | '.join(axis_labels)}")
    if not literal and any(p not in " ".join(axis_labels) for p in data["numeric_params"]):
        unshown = [p for p in data["numeric_params"] if p not in " ".join(axis_labels)]
        print(f"[axes] tip: --literal_axis {unshown[0]}  to pin that param to an axis")

    bg_params = [{**e.get("c", {}), **e.get("w", {})} for e in data["matrix_v"]]
    cat_param = data["cat_params"][0] if data["cat_params"] else None

    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_stem  = data["run_name"]
    out_png   = args.out_png  or (DEFAULT_FIG_DIR / f"{run_stem}.png")
    out_html  = args.out_html or (DEFAULT_FIG_DIR / f"{run_stem}.html")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    plot_png(bg_xy, witness_groups, cat_param, method,
             data["app"], data["run_name"], out_png, axis_labels)
    plot_html(bg_xy, bg_params, witness_groups, cat_param, method,
              data["app"], data["run_name"], out_html, axis_labels)


if __name__ == "__main__":
    main()
