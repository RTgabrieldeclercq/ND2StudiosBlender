"""Object Metrics (``analysis.object_metrics``) — Per-object derived metrics onto the member layer: velocity/speed (µm/s), K-nearest-neighbour distances (µm), local divergence/curl (1/s), and frame- or self-normalized intensity fold change."""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import member_layer, on_layer
from nodegraph.catalog._shared.objects import (
    _InFrameInterval,
    _frame_interval_s,
    _object_table,
    _object_velocity,
)

# ── per-object derived metrics (CT compute_spatial_metrics + self/frame fold) ───

#: ``metrics`` → the column(s) each one writes onto the member layer.
_OBJECT_METRIC_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "velocity": ("vy", "vx"),
    "speed": ("speed",),
    "neighbors": ("neighbor_dist_mean", "neighbor_dist_std"),
    "divergence": ("local_divergence",),
    "curl": ("local_curl",),
    "frame_fold": ("frame_fold",),
    "self_fold": ("self_fold",),
}
#: metrics that need a ``track_id`` column on the member layer (run ``track.objects``
#: or ``track.link`` first) — everything built on per-object velocity, plus self_fold.
_OBJECT_METRICS_TRACKED = frozenset({"velocity", "speed", "divergence", "curl",
                                     "self_fold"})
#: metrics that need an intensity column (``analysis.measure`` first).
_OBJECT_METRICS_INTENSITY = frozenset({"frame_fold", "self_fold"})
def _object_metric_names(raw) -> list:
    """The ``metrics`` selector → an ordered, de-duplicated list of metric names. Same
    comma-separated-string / sequence forms as ``analysis.measure``'s ``stats``, and a
    blank selector falls back to the cheap positional set rather than being a silent
    no-op."""
    names = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in (raw or ())])
    names = [s for s in names if s] or ["velocity", "speed", "neighbors"]
    bad = [s for s in names if s not in _OBJECT_METRIC_COLUMNS]
    if bad:
        raise ValueError(f"unknown object metric(s) {bad} — choose from "
                         f"{list(_OBJECT_METRIC_COLUMNS)} (comma-separated)")
    return list(dict.fromkeys(names))
def _group_key(*cols: np.ndarray) -> np.ndarray:
    """One int64 group id per row from several integer columns — for grouping by
    ``(m,t,z,c)`` without building tuples. Uses ``np.unique(..., axis=0)``'s inverse on
    the stacked columns, which is exact for any ranges involved (a bit-packed radix key
    would overflow on a long series or a large label space)."""
    stack = np.column_stack([np.asarray(c, dtype=np.int64).ravel() for c in cols])
    _uniq, inv = np.unique(stack, axis=0, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()
def _group_fold(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """``value / mean(value within its group)`` — Cell-Tracker's fold change, used for both
    ``frame_fold`` (group = one frame) and ``self_fold`` (group = one track).

    A group whose mean is not finite and positive yields NaN rather than an infinity: a
    fold change against nothing is undefined, and NaN is the value every downstream
    reducer already ignores, whereas ``inf`` poisons a mean."""
    v = np.asarray(values, dtype=float)
    uniq, inv = np.unique(np.asarray(keys), return_inverse=True)
    inv = np.asarray(inv).ravel()
    finite = np.isfinite(v)
    sums = np.bincount(inv, weights=np.where(finite, v, 0.0), minlength=len(uniq))
    cnts = np.bincount(inv, weights=finite.astype(float), minlength=len(uniq))
    means = np.divide(sums, cnts, out=np.full(len(uniq), np.nan, dtype=float),
                      where=cnts > 0)
    denom = means[inv]
    ok = np.isfinite(denom) & (denom > 0)
    return np.divide(v, denom, out=np.full(v.shape, np.nan, dtype=float), where=ok)
def _neighbourhood_metrics(coords: np.ndarray, vy: np.ndarray, vx: np.ndarray,
                           k: int, min_r: float):
    """Cell-Tracker's ``compute_spatial_metrics`` for ONE frame group, vectorized.

    Returns ``(dist_mean, dist_std, divergence, curl)``, each ``(N,)``. ``coords`` is
    ``(N,2)`` ``(y,x)`` in **µm**, ``vy``/``vx`` are µm/s, so the distances come out in µm
    and the two gradient measures in 1/s.

    * neighbour distances — mean and std of the ``k`` nearest neighbours' distances (self
      excluded), the local crowding measure.
    * ``divergence`` — the mean radial component of each neighbour's *relative* velocity
      over its distance, ``⟨(Δr·Δv)/|Δr|²⟩``. Positive = neighbours moving away
      (expansion), negative = converging.
    * ``curl`` — the same with the 2-D cross product, ``⟨(Δy·Δvx - Δx·Δvy)/|Δr|²⟩``: the
      local rotation of the neighbourhood about this object.

    ``min_r`` floors ``|Δr|`` (Cell-Tracker clips at 1 px; here it is one pixel in µm), so
    two objects whose centroids nearly coincide cannot produce an arbitrarily large
    gradient. A neighbour with unknown velocity — a track's first detection — is excluded
    rather than treated as stationary, and an object with fewer than 2 usable neighbours
    gets NaN for the two gradient measures, matching the source. The original looped per
    object in Python; this is array form, which is where the time goes on a dense field.

    **One deliberate loosening.** Cell-Tracker skips a frame entirely below 3 objects
    (``if len(fdf) < 3: continue``), which also blanks the neighbour DISTANCES — yet a
    nearest-neighbour distance is perfectly well defined for two objects, and returning NaN
    for a number that exists is a silent degradation. The floor here is 2, and the gradient
    measures keep their own independent ``>= 2 neighbours`` guard, so they still come back
    NaN exactly where the source left them NaN."""
    n = len(coords)
    nan = np.full(n, np.nan, dtype=float)
    if n < 2 or k < 1:
        return nan.copy(), nan.copy(), nan.copy(), nan.copy()
    from scipy.spatial import cKDTree
    kk = int(min(k, n - 1))
    dists, idx = cKDTree(coords).query(coords, k=kk + 1)
    nb_d = np.atleast_2d(dists)[:, 1:]                      # drop self (column 0)
    nb_i = np.atleast_2d(idx)[:, 1:]
    dist_mean = nb_d.mean(axis=1)
    dist_std = nb_d.std(axis=1)
    # relative positions + relative velocities of every neighbour, (N, kk)
    dy = coords[nb_i, 0] - coords[:, None, 0]
    dx = coords[nb_i, 1] - coords[:, None, 1]
    r2 = np.maximum(dy * dy + dx * dx, float(min_r) ** 2)
    dvy = vy[nb_i] - np.nan_to_num(vy)[:, None]
    dvx = vx[nb_i] - np.nan_to_num(vx)[:, None]
    valid = np.isfinite(dvy) & np.isfinite(dvx)
    count = valid.sum(axis=1)
    radial = np.where(valid, (dy * dvy + dx * dvx) / r2, 0.0).sum(axis=1)
    tangent = np.where(valid, (dy * dvx - dx * dvy) / r2, 0.0).sum(axis=1)
    div, crl = nan.copy(), nan.copy()
    keep = count >= 2
    div[keep] = radial[keep] / count[keep]
    crl[keep] = tangent[keep] / count[keep]
    return dist_mean, dist_std, div, crl
def _layers_object_metrics(params, modes):
    """`analysis.object_metrics` has no output-name socket: it adds columns to the member
    layer its READ socket names, creating that ``(domain, name)`` pair when the upstream
    produced only a Voxel raster. Total by contract (runs on every keystroke)."""
    if (modes or {}).get("target") == "point":
        return ((Domain.POINT, params.get("points") or "spots"),)
    return ((Domain.LABEL, params.get("labels") or "labels"),)
def _compute_object_metrics(ctx: EvalContext) -> Dataset:
    """Per-object derived metrics written back as columns on the member layer — the
    Lagrangian half of Cell-Tracker's spatial analysis
    (``backend/measurement.py::compute_spatial_metrics``, ``backend/fields.py::
    compute_self_fold_change``, and ``scripts/mean_velocity.py``).

    ``metrics`` is a comma-separated selector; each entry writes its own column(s):

    ==================  ==========================================  ================
    metric              column(s)                                   unit
    ==================  ==========================================  ================
    ``velocity``        ``vy``, ``vx``                              µm/s
    ``speed``           ``speed``                                   µm/s
    ``neighbors``       ``neighbor_dist_mean``, ``neighbor_dist_std``  µm
    ``divergence``      ``local_divergence``                        1/s
    ``curl``            ``local_curl``                              1/s
    ``frame_fold``      ``frame_fold``                              —
    ``self_fold``       ``self_fold``                               —
    ==================  ==========================================  ================

    Everything is **physical**: positions convert with ``pixel_size_um`` and the time step
    with ``dt_s``, so a velocity is µm/s rather than Cell-Tracker's px/frame. With neither
    calibration present both degrade to 1:1 (px per frame), the same 1:1 fallback
    :func:`to_pixels_v2` uses everywhere.

    Prerequisites, refused rather than silently skipped: ``velocity``/``speed``/
    ``divergence``/``curl``/``self_fold`` need a ``track_id`` column (run ``track.objects``
    or ``track.link`` first) — without it there is no correspondence between frames and a
    displacement cannot be defined at all. ``frame_fold``/``self_fold`` need an intensity
    column (run ``analysis.measure`` first).

    ``frame_fold`` is intensity over the mean of its own frame group — Cell-Tracker's
    ``<channel>_norm``, the "is this cell brighter than its neighbours right now" measure.
    ``self_fold`` is intensity over the mean over that object's OWN track — "is this cell
    brighter than it usually is", which is independent of what the rest of the field is
    doing and is the one you want for a signalling response.

    Grouping is per ``(m, t, z, c)`` for anything positional: objects in different
    multipoints, z planes or channels are not neighbours of each other, which is the same
    grouping ``track.objects`` links within.

    Footprint ``WHOLE_SERIES`` with no image access (``kernel_axes`` empty) — every input
    is already a structure column, and velocity needs the whole T axis at once."""
    ds = ctx.inputs[0]
    domain, layer, cols, zk = _object_table(ctx, ds, node="object metrics")
    names = _object_metric_names(ctx.params.get("metrics", ""))
    n = len(cols["id"])
    px = ctx.calib("pixel_size_um") or 0.1
    need_track = bool(set(names) & _OBJECT_METRICS_TRACKED)
    need_inten = bool(set(names) & _OBJECT_METRICS_INTENSITY)
    # dt_s only matters once a velocity is involved (R1: don't fence a pull on a key it
    # cannot use) — and when it IS involved an absent interval is refused, not defaulted.
    dt_s = _frame_interval_s(ctx, needed=need_track,
                             wanted=set(names) & _OBJECT_METRICS_TRACKED,
                             node="object metrics")

    track = cols.get("track_id")
    if need_track and track is None:
        raise ValueError(
            f"object metrics: {sorted(set(names) & _OBJECT_METRICS_TRACKED)} are built "
            f"from each object's motion between frames, and layer {layer!r} carries no "
            f"'track_id' column — so there is no correspondence between one frame's "
            f"objects and the next's. Run track.objects (or track.link) upstream, or ask "
            f"only for 'neighbors' / 'frame_fold', which need no tracking.")
    inten_name = ctx.params.get("intensity", "mean_intensity")
    inten = cols.get(inten_name)
    if need_inten and inten is None:
        raise ValueError(
            f"object metrics: {sorted(set(names) & _OBJECT_METRICS_INTENSITY)} need an "
            f"intensity column, but layer {layer!r} has no {inten_name!r} "
            f"(it carries {sorted(cols)}). Run analysis.measure upstream, or point the "
            f"`intensity` socket at the column you want the fold change taken over.")

    tt = np.rint(np.asarray(cols["t"], dtype=float)).astype(np.int64)
    y_um = np.asarray(cols["y"], dtype=float) * px
    x_um = np.asarray(cols["x"], dtype=float) * px
    out: Dict[str, np.ndarray] = {}

    vy = vx = None
    if need_track and {"velocity", "speed", "divergence", "curl"} & set(names):
        vy, vx = _object_velocity(np.asarray(track, dtype=np.int64), tt, y_um, x_um, dt_s)
        if "velocity" in names:
            out["vy"], out["vx"] = vy, vx
        if "speed" in names:
            out["speed"] = np.hypot(vy, vx)

    if {"neighbors", "divergence", "curl"} & set(names):
        k = max(1, int(ctx.params.get("n_neighbors", 6)))
        v_y = vy if vy is not None else np.full(n, np.nan)
        v_x = vx if vx is not None else np.full(n, np.nan)
        dm, dsd = np.full(n, np.nan), np.full(n, np.nan)
        dv, cl = np.full(n, np.nan), np.full(n, np.nan)
        groups = _group_key(cols["m"], tt, np.rint(np.asarray(cols["z"], dtype=float)),
                            cols["c"])
        for g in np.unique(groups):
            sel = np.flatnonzero(groups == g)
            coords = np.column_stack([y_um[sel], x_um[sel]])
            a, b, c_, d = _neighbourhood_metrics(coords, v_y[sel], v_x[sel], k,
                                                 min_r=px)
            dm[sel], dsd[sel], dv[sel], cl[sel] = a, b, c_, d
        if "neighbors" in names:
            out["neighbor_dist_mean"], out["neighbor_dist_std"] = dm, dsd
        if "divergence" in names:
            out["local_divergence"] = dv
        if "curl" in names:
            out["local_curl"] = cl

    if "frame_fold" in names:
        out["frame_fold"] = _group_fold(
            np.asarray(inten, dtype=float),
            _group_key(cols["m"], tt, np.rint(np.asarray(cols["z"], dtype=float)),
                       cols["c"]))
    if "self_fold" in names:
        out["self_fold"] = _group_fold(np.asarray(inten, dtype=float),
                                       np.asarray(track, dtype=np.int64))

    # with_layer (not with_structure): adds columns to the EXISTING member layer without
    # re-emitting `id`, so every sibling column keeps its row order, and the layer's
    # z_kind provenance is left untouched (the cluster_points pattern).
    res = ds
    for name, values in out.items():
        res = res.with_layer(domain, name, np.asarray(values, dtype=float), layer=layer)
    return res

def _columns_object_metrics(params, modes, incoming):
    """The column(s) each selected metric writes onto the member layer (V2.28), read from
    the same ``_OBJECT_METRIC_COLUMNS`` map the compute writes through — so ``velocity``
    correctly offers ``vy``/``vx`` rather than a column called ``velocity`` that no table
    ever carries.

    ``_object_metric_names`` refuses an unknown name; called defensively here for
    ``analysis.measure``'s reason — a selector mid-edit must not blank the envelope."""
    try:
        dom, lyr = member_layer(params, modes)
        try:
            names = _object_metric_names(params.get("metrics", ""))
        except Exception:
            return ()                        # a selector mid-edit offers nothing new
        cols = [c for n in names for c in _OBJECT_METRIC_COLUMNS.get(n, ())]
        return on_layer(dom, lyr, dict.fromkeys(cols))
    except Exception:                        # pragma: no cover - defensive
        return ()

register_node(
    batch_aware(_compute_object_metrics), op_key="analysis.object_metrics", label="Object Metrics",
    category="analysis",
    adds_columns=_columns_object_metrics,
    extra_layers=_layers_object_metrics,
    # ONE of the two, never both — the `analysis.object_field` note applies verbatim:
    # `_object_table` is shared, so the two nodes' requirements are the same function of
    # `target` and the static union over-claimed on both branches (V2.22).
    reads_domains=frozenset(),
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.LABEL, Domain.VOXEL}),
        "point": frozenset({Domain.POINT}),
    }},
    adds_domains=frozenset({Domain.LABEL, Domain.POINT}),
    inputs=[
        InDataset(),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})},
                 description=
                 "Which Label table's objects to measure — the output of Segmentation or "
                 "Connected Components, after Measure and Tracking have added their "
                 "columns. The new metric columns are written back onto THIS layer, so "
                 "downstream nodes and the spreadsheet see them alongside area and "
                 "intensity."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})},
                 description=
                 "Which Point table's objects to measure — the output of Spot Detection or "
                 "Particle Detection. Point tables carry no area or intensity of their own, "
                 "so `frame_fold` and `self_fold` need a measured column pointed at "
                 "explicitly; the positional metrics work as they do for labels."),
        InString("metrics", "Metrics", field=False, default="velocity,speed,neighbors",
                 vocab=tuple(_OBJECT_METRIC_COLUMNS),
                 description=
                 "Which metrics to compute, comma-separated, each writing its own "
                 "column(s): `velocity`->vy,vx (um/s), `speed`->speed (um/s), "
                 "`neighbors`->neighbor_dist_mean,neighbor_dist_std (um), "
                 "`divergence`->local_divergence (1/s, positive = neighbours spreading "
                 "apart), `curl`->local_curl (1/s, local rotation), `frame_fold`->intensity "
                 "over its own frame's mean, `self_fold`->intensity over this track's own "
                 "time-average. Everything except `neighbors` and `frame_fold` needs a "
                 "track_id column; both folds need an intensity column. "
                 "Asking for fewer "
                 "writes fewer columns and does no less work per metric — velocity is "
                 "computed once and shared by speed, divergence and curl.",
                 choice_docs={
                     "velocity":
                         "→ `vy`,`vx` in µm/s: the object's DISPLACEMENT per second along y "
                         "and x, signed. Keep it when direction matters (do cells move up the "
                         "gradient?); it averages toward zero over a round trip, which `speed` "
                         "does not. Needs a track_id column.",
                     "speed":
                         "→ `speed` in µm/s: the magnitude of that velocity, always positive. "
                         "The measure of how ACTIVE an object is regardless of direction — and "
                         "the one that inflates with tracking noise, since every spurious "
                         "jitter adds to it rather than cancelling. Needs a track_id column.",
                     "neighbors":
                         "→ `neighbor_dist_mean`,`neighbor_dist_std` in µm: distance to the N "
                         "nearest neighbours, averaged and spread. The mean is a local density "
                         "measure (small = crowded); the std says whether the neighbourhood is "
                         "evenly spaced or clumped. The only metric here that needs NO "
                         "tracking.",
                     "divergence":
                         "→ `local_divergence` in 1/s: how fast the local neighbourhood is "
                         "spreading apart (positive) or converging (negative) — expansion, "
                         "proliferation and drainage read positive, a wound closing or cells "
                         "packing in read negative. Built from neighbour velocities, so it "
                         "needs tracking and inherits its noise.",
                     "curl":
                         "→ `local_curl` in 1/s: local ROTATION of the neighbourhood, signed by "
                         "handedness. Picks up swirling and vortical collective motion that "
                         "speed alone cannot distinguish from random walk. Needs tracking; "
                         "sensitive to the neighbour count, which sets the scale of rotation "
                         "measured.",
                     "frame_fold":
                         "→ `frame_fold`: the object's intensity divided by the MEAN over its "
                         "own frame. Removes whole-frame illumination and bleaching changes, so "
                         "values are comparable across timepoints, and answers \"is this cell "
                         "brighter than its neighbours right now\". Needs an intensity column; "
                         "no tracking.",
                     "self_fold":
                         "→ `self_fold`: the object's intensity divided by ITS OWN track's "
                         "time-average. Per-object normalization, so it answers \"is this cell "
                         "brighter than it usually is\" and is unaffected by cell-to-cell "
                         "expression differences. Needs both tracking and an intensity column.",
                 }),
        InInt("n_neighbors", "Neighbours", unit="", field=False, default=6,
              description=
              "How many nearest neighbours define each object's local neighbourhood — used "
              "by `neighbors`, `divergence` and `curl`. It sets the SCALE these read at: "
              "FEW neighbours (3-4) measure the immediate contact ring and respond to "
              "single-cell rearrangement, while MANY (10+) average over a wider patch and "
              "report bulk flow, smoothing individual events away. 6 is Cell-Tracker's "
              "default and roughly the contact number of a confluent monolayer. Clamped to "
              "the objects actually present in each frame, so a sparse frame silently uses "
              "fewer. Ignored when none of those three metrics is requested."),
        InString("intensity", "Intensity column", field=False, default="mean_intensity",
                 description=
                 "Which measured column the two fold changes are computed over — "
                 "`mean_intensity` is what analysis.measure writes for its `mean` stat. "
                 "Point it at `median_intensity`, `max_intensity` or any other column you "
                 "have measured to take the fold change over that instead. The choice does "
                 "not change any other metric, and it is not read at all unless "
                 "`frame_fold` or `self_fold` is requested."),
        *_InFrameInterval(),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Members",
                description=
                "Which kind of object the metrics are written onto — the segmented regions of "
                "a Label table or the detections of a Point table. It selects which source "
                "socket is live and, because the two carry different columns, which metrics "
                "are available at all. New columns are added to that table in place.",
                choice_docs={
                    "label":
                        "Measure the regions of a Label table (Segmentation / Connected "
                        "Components). Positions come from region centroids, and because a "
                        "Label table already carries `area` and measured intensities, the two "
                        "fold-change metrics work as soon as Measure has run.",
                    "point":
                        "Measure the detections of a Point table (Spot / Particle Detection). "
                        "The positional metrics behave identically, but a Point table has no "
                        "area and no intensity of its own — so the folds need an intensity "
                        "column to have been transferred onto the points explicitly.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Per-object derived metrics onto the member layer: velocity/speed (µm/s), "
                "K-nearest-neighbour distances (µm), local divergence/curl (1/s), and "
                "frame- or self-normalized intensity fold change. Ports Cell-Tracker's "
                "compute_spatial_metrics + self-fold + mean_velocity; 2D, per (m,t,z,c).")
