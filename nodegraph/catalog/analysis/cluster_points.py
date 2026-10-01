"""Cluster Points (``analysis.cluster_points``) — Cluster a Point cloud into groups (GaussianMixture / KMeans, BIC model-order sweep over n_clusters ± relax%) → a per-point cluster-id label column;."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer

# ── Cluster a point cloud → a per-point cluster-id label column ────────────────
#
# Ports the vendored `granule_cluster` kernel (scikit-learn GaussianMixture / KMeans with a
# BIC model-order sweep). GENERAL: clusters ANY 3-D point cloud into groups (beads→granules
# is one use). Adds a per-point integer label column to the Point layer (per (m,t,c) group;
# ids 0..k-1 within a group — the natural key for the downstream `analysis.tessellate`, which
# also groups by (m,t,c)). The granule chain's front half: detect.particles → cluster_points
# → tessellate → rasterize_mesh. Requires scikit-learn (installed 2026-07-26).


def _compute_cluster_points(ctx: EvalContext) -> Dataset:
    """Cluster a Point cloud into groups → a per-point cluster-id label column (ported v1
    ``granule_cluster`` kernel). Fits a full-covariance GaussianMixture (``gmm``, keeps
    anisotropic clusters whole) or spherical ``kmeans``, selecting the model order ``k`` by a
    BIC sweep over ``n_clusters ± relax_pct%`` (both BICs directly comparable). Points are
    scaled to µm (``voxel_size_um``) before fitting so anisotropic Z doesn't bias the model.
    Clusters each ``(m,t,c)`` group independently (ids ``0..k-1`` per group).

    Resolved spec: category analysis; op ``analysis.cluster_points``; reads POINT, adds POINT
    (a label column on the ``source`` layer, default ``cluster``). ``voxel_size_um=(z_step_um,
    pixel_size_um, pixel_size_um)`` via ``ctx.calib`` (memo-fenced). Footprint WHOLE_VOLUME
    (clusters a whole 3-D cloud; no image access → ``kernel_axes`` empty). Kernel:
    :func:`nodegraph.kernels.granule_cluster.cluster_granules` (scikit-learn)."""
    from nodegraph.kernels.granule_cluster import cluster_granules
    ds = ctx.inputs[0]
    # the one Point cloud on the wire, whatever it is called (`_resolve_layer`)
    source, _note = _resolve_layer(
        _point_layers(ds), ctx.layer("source"), node="cluster points", socket="source",
        what="Point table", where="the `data` input",
        remedy="these are the points to group, so wire a detection (detect.particles / "
               "detect.spots) or transform.label_to_points upstream", ctx=ctx)
    name = ctx.params.get("name", "cluster")
    method = ctx.params.get("__modes__", {}).get("method", "gmm")
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)
    cp = {"n_granules": max(1, int(ctx.params.get("n_clusters", 1))),
          "relax_pct": max(0.0, float(ctx.params.get("relax_pct", 0.0))),
          "method": method,
          "n_init": max(1, int(ctx.params.get("n_init", 1)))}
    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:                                      # pragma: no cover - _resolve_layer
        raise ValueError(f"cluster points needs a Point layer {source!r} "
                         "(run detect.particles first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    out = np.zeros(len(z_all), dtype=np.int64)
    for m in sorted(set(m_all.tolist())):
        for t in sorted(set(t_all[m_all == m].tolist())):
            for c in sorted(set(c_all[(m_all == m) & (t_all == t)].tolist())):
                sel = (m_all == m) & (t_all == t) & (c_all == c)
                if not sel.any():
                    continue
                pzyx = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                labels, _info = cluster_granules(pzyx, vox, cp)
                out[sel] = np.asarray(labels, dtype=np.int64)
    # Add the label column to the SAME Point layer (with_layer, not with_structure — leaves
    # the source's z_kind provenance untouched, no clobber).
    return ds.with_layer(Domain.POINT, name, out, layer=source)

def _columns_cluster_points(params, modes, incoming):
    """One column, onto the SOURCE layer. This node writes its cluster id back onto the very
    Point table it read (``with_layer``, so the source's z_kind provenance is untouched)
    rather than emitting a new instance — so the name to declare is the READ socket's, which
    is the ``analysis.measure`` / ``analysis.object_metrics`` shape."""
    return on_layer(Domain.POINT, str((params or {}).get("source") or "particles"),
                    (str((params or {}).get("name") or "cluster"),))

register_node(
    _compute_cluster_points, op_key="analysis.cluster_points", label="Cluster Points",
    adds_columns=_columns_cluster_points,
    category="analysis",
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(),
            InString("source", "Point layer", field=False, default="particles",
                     layer_in=Domain.POINT,
                     description=
                     "Which Point cloud to cluster — a Spot or Particle Detection output. "
                     "Positions are scaled to MICRONS before fitting, so an anisotropic z step "
                     "cannot stretch the clusters along z and bias the result."),
            InString("name", "Label column", field=False, default="cluster",
                     description=
                     "Name of the new COLUMN added to the same Point table — this node does not "
                     "create a layer, it annotates the existing one with each point's cluster "
                     "id. Tessellate reads this column name to mesh one element per cluster. "
                     "Use two different names if you cluster the same cloud twice, or the "
                     "second result overwrites the first."),
            InInt("n_clusters", "Cluster count", unit="", field=True, default=1,
                  description=
                  "How many groups to fit. It is a STARTING POINT, not a hard count: the node "
                  "sweeps a range around it (set by Relax %) and keeps whichever count scores "
                  "best by BIC, so the answer may differ from what you type. Leave Relax at 0 "
                  "to force exactly this many. Asking for more clusters than the data supports "
                  "splits real groups; asking for fewer merges them."),
            InFloat("relax_pct", "Relax %", unit="", field=True, default=0.0,
                    description=
                    "How far either side of Cluster count to search, as a percentage — the "
                    "model-order sweep. 0 pins the count exactly. 20 with a count of 10 tries "
                    "roughly 8 through 12 and selects by BIC, which penalises extra clusters, so "
                    "it will not simply choose the largest. Each extra candidate is a full "
                    "refit, so runtime scales with the width of the sweep."),
            InInt("n_init", "Restarts", unit="", field=False, default=1,
                  description=
                  "How many times to refit each candidate from a different random start, keeping "
                  "the best. Both models are sensitive to initialisation, so 1 can land in a "
                  "poor local optimum and give a different answer on data that only differs "
                  "slightly. MORE restarts make the result stable and reproducible at "
                  "proportional cost — raise this when a clustering looks arbitrary.")],
    outputs=[OutDataset()],
    modes=[Mode("method", ["gmm", "kmeans"], default="gmm", label="Model",
                description=
                "Which clustering model is fitted to the point cloud (in µm, so an "
                "anisotropic z step does not bias it). Both are fitted at several model "
                "orders and scored by BIC, so this changes the SHAPE of cluster each can "
                "represent, not how k is chosen.",
                choice_docs={
                    "gmm":
                        "Full-covariance Gaussian mixture: each cluster may be elongated and "
                        "tilted in any direction, so a stretched or flattened group survives "
                        "as ONE cluster. The better fit for real biological groupings, and "
                        "the default; it has more parameters to estimate, so it is less "
                        "stable on very few points per cluster.",
                    "kmeans":
                        "Spherical clusters of comparable size — it minimizes distance to "
                        "the nearest centre, which implicitly assumes round, equal-sized "
                        "groups. Faster and far more stable on sparse clouds, but it SPLITS "
                        "an elongated group down its middle rather than keeping it whole.",
                })],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Cluster a Point cloud into groups (GaussianMixture / KMeans, BIC model-order "
                "sweep over n_clusters ± relax%) → a per-point cluster-id label column; µm-"
                "scaled so anisotropic Z doesn't bias the fit (ported v1 kernel; scikit-learn).")
