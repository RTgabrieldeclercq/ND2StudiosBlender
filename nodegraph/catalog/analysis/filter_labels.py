"""Filter Labels (``analysis.filter_labels``) — Keep or drop WHOLE labels by thresholding one per-label column (mean_intensity, area, eccentricity, …), with the cut chosen by a histogram method over the label population itself → a new Label instance with the surviving ids unchanged."""

from __future__ import annotations

import numpy as np

from typing import Dict, List, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_raster, _resolve_label_instance
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.scope import LATTICE_SCOPES, ScopeMode, scope_row_key

# ── Filter Labels (a cut on the POPULATION of per-label values) ─────────────────
#
# The gap this fills: the only per-object gates in the catalog are the `min_area`/`max_area`
# windows baked into `analysis.segment` and `analysis.histogram_threshold`. They are typed in
# absolute µm² and they only exist inside the node that produced the labels, so there was no
# way to say "drop the dim third of these cells" (a cut on a column those nodes do not
# measure), and no way to filter a segmentation at all once it had left the segmenter.
#
# It reads the TABLE, never the pixels. That is what keeps it composable — the column can be
# anything upstream measured, including one neither this node nor `analysis.measure` knows
# about (a track column, a shape metric, a field sampled onto the labels) — and it is why the
# node costs one pass over the raster rather than a full gather of the enhancement chain.

#: cut methods. The five histogram methods run over the POPULATION of per-label values (one
#: sample per label, not per voxel); ``fixed`` and ``percentile`` are the explicit forms.
_FILTER_METHODS: Tuple[str, ...] = ("otsu", "li", "yen", "triangle", "mean",
                                    "fixed", "percentile")
#: This node's slice of the shared population vocabulary
#: (:mod:`nodegraph.catalog._shared.scope`, V2.27) — the same four words, and the same meanings,
#: as ``analysis.threshold``'s. This node groups TABLE ROWS by their own m/t/z/c columns rather
#: than voxels by their address, but "which population is the cut derived over" is the identical
#: question and must not read differently in two places.
#:
#: The two STRUCTURE scopes are deliberately not offered: this node's members already *are* the
#: objects, so a per-object population would be one row and the cut derived from it would be
#: that row's own value — every label would then survive, whichever side was kept.
_FILTER_SCOPES: Tuple[str, ...] = LATTICE_SCOPES
#: the per-label columns that are common enough to name in the socket's own documentation.
#: NOT a closed set: the socket takes any column on the table, which is the point.
_KNOWN_COLUMNS: Tuple[str, ...] = (
    "mean_intensity", "max_intensity", "min_intensity", "total_intensity",
    "median_intensity", "area", "area_um2", "volume_um3", "eccentricity", "solidity",
    "extent", "perimeter", "axis_major", "axis_minor", "n_sub", "frac_above", "level")


def _population_cut(values: np.ndarray, *, method: str, level: float,
                    percentile: float) -> Optional[float]:
    """The cut for one population of per-label values, or ``None`` when it has none.

    ``values`` is already finite-filtered. A population with **zero spread** has no derivable
    cut — every label carries the same number, so any histogram method's answer would put all
    of them on one side and the choice of side would be an artefact of the tie-break rather
    than a decision about the data. ``fixed`` is exempt: an explicit level is a statement
    about the values, not a question asked of them, so it applies to a degenerate population
    exactly as the user typed it."""
    if method == "fixed":
        return float(level)
    if values.size == 0:
        return None
    if method == "percentile":
        return float(np.percentile(values, percentile))
    if float(np.ptp(values)) <= 0.0:
        return None
    from skimage import filters
    try:
        cut = float(getattr(filters, f"threshold_{method}")(values))
    except (ValueError, RuntimeError):
        return None
    return cut if np.isfinite(cut) else None


def _compute_filter_labels(ctx: EvalContext) -> Dataset:
    """Keep or drop **whole labels** by thresholding one per-label column, with the cut
    derived from the population of label values itself.

    Resolved spec (grilled 2026-08-05): category analysis; op ``analysis.filter_labels``;
    reads ``{VOXEL, LABEL}``, adds ``{VOXEL, LABEL}``; ``WHOLE_VOLUME``; **no 2D/3D lever**
    (it rewrites a raster it does not interpret geometrically, and the rows it cuts were
    measured in whatever dimensionality produced them); backend
    ``skimage.filters.threshold_*`` over the value population, lazily imported. Reads NO
    calibration key: the column is already in whatever unit it was measured in, and this node
    does not convert it — so its memo fences on nothing.

    **The statistic is a COLUMN, not something this node measures.** It reads
    ``column`` off the incoming Label table, so it filters on anything upstream produced —
    ``analysis.measure``'s intensity stats and regionprops geometry, ``analysis.object_metrics``
    physical areas, ``analysis.threshold_per_label``'s ``n_sub``/``frac_above``, a tracked
    column. Re-implementing the Voxel→Label bridge here would have bought one fewer node in
    the graph and permanently limited the cut to the handful of stats it hardcoded.

    **Ids are PRESERVED, never renumbered.** Label 7 stays label 7 in the output raster and
    the output table, with gaps where labels were dropped. Renumbering to a compact 1..K
    would silently break every join against a table already measured upstream or a track
    referencing those ids — a mislabelled result that looks entirely healthy. The output goes
    to a NEW layer (``name``), so the unfiltered instance stays on the wire and both can be
    viewed.

    **NaN is not a value.** A row whose column is non-finite was never measured (a region the
    regionprops walk never saw, a stat that divided by zero), so it is excluded from the
    population that derives the cut AND from the survivors — inventing a side for it would
    either keep an unmeasured object or drop a measured one, and both are decided by an
    accident. The count is reported on the progress rail.

    ``scope`` groups the table's rows by their own ``m``/``t``/``z``/``c`` columns using
    ``analysis.threshold``'s vocabulary; unlike there it changes no read footprint, because
    this node only ever touches the table and one pass of the raster."""
    ds = ctx.inputs[0]
    ax = ds.axes
    layer, lnote = _resolve_label_instance(
        ds, ctx.layer("labels"), node="filter labels", socket="labels",
        remedy="this keeps or drops whole REGIONS, so it needs a label raster AND the table "
               "whose column it cuts on — run analysis.segment / analysis.label upstream, "
               "then analysis.measure to produce the column")
    raster6, zk = _label_raster(ds, layer, node="filter labels")
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.LABEL)
            if a.layer == layer}
    column = str(ctx.params.get("column", "mean_intensity") or "mean_intensity").strip()
    if "id" not in cols:
        raise ValueError(
            f"filter labels: the Label table {layer!r} has no `id` column, so there is no "
            f"way to say which raster ids survive (it carries {sorted(cols)}).")
    if column not in cols:
        raise ValueError(
            f"filter labels: the Label table {layer!r} carries no column {column!r} — it has "
            f"{sorted(k for k in cols if k not in ('id',))}. Measure it first: "
            f"analysis.measure `stats` gives mean/max/min/sum/median intensity and `area`, "
            f"its `shape` selector the regionprops geometry (eccentricity, solidity, …), and "
            f"analysis.object_metrics the physical µm areas.")
    ids = np.asarray(cols["id"], dtype=np.int64)
    if ids.size == 0:
        raise ValueError(
            f"filter labels: the Label table {layer!r} has no rows, so there is no population "
            f"to derive a cut from and nothing to keep. Whatever produced it segmented no "
            f"region — fix that upstream rather than filtering an empty result.")
    values = np.asarray(cols[column], dtype=float)
    if values.shape != ids.shape:
        raise ValueError(
            f"filter labels: column {column!r} has {values.size} entries but the table has "
            f"{ids.size} ids — every label would be cut against another label's value.")

    modes = ctx.params.get("__modes__", {})
    method = modes.get("method", "otsu")
    keep_side = modes.get("keep", "above")
    scope = modes.get("scope", "series")
    if method not in _FILTER_METHODS:
        raise ValueError(f"filter labels: unknown method {method!r} — expected one of "
                         f"{', '.join(_FILTER_METHODS)}.")
    if scope not in _FILTER_SCOPES:
        raise ValueError(f"filter labels: unknown scope {scope!r} — expected one of "
                         f"{', '.join(_FILTER_SCOPES)}.")
    level = float(ctx.params.get("level", 0.0) or 0.0)
    percentile = float(ctx.params.get("percentile", 50.0) or 50.0)
    if method == "percentile" and not 0.0 <= percentile <= 100.0:
        raise ValueError(f"filter labels: percentile={percentile:g} is outside 0–100 — it is "
                         "a rank within the population of label values.")
    need = ("m", "t", "z", "c") if scope == "plane" else \
           ("m", "t", "c") if scope == "volume" else \
           ("m", "c") if scope == "series" else ("c",)
    missing = [k for k in need if k not in cols]
    if missing:
        raise ValueError(
            f"filter labels: scope={scope} groups the labels by {list(need)}, but the table "
            f"{layer!r} is missing {missing}. Use scope=dataset, or re-measure the labels "
            f"with a producer that writes the invariant id,m,t,c,z,y,x schema.")
    if scope == "plane" and zk != "plane_index":
        raise ValueError(
            f"filter labels: scope=plane needs each row's `z` to BE a plane index, but the "
            f"Label table {layer!r} is 3D (z_kind={zk!r}) — its z is a subpixel centroid, so "
            f"grouping on it would give most volumetric regions a population of one and cut "
            f"each against itself. Use scope=volume (per m,t,c), which is the finest grouping "
            f"a volumetric segmentation has.")

    # `z` is a float column (a centroid), so the plane scope needs it rounded to the plane
    # index it actually is; the guard above has already refused the volumetric case where that
    # rounding would be meaningless. Every other axis is already integral.
    key_cols = {k: np.asarray(cols[k], dtype=np.int64) for k in ("m", "t", "c") if k in cols}
    key_cols["z"] = (np.rint(np.asarray(cols["z"], dtype=float)).astype(np.int64)
                     if "z" in cols else np.zeros(ids.size, dtype=np.int64))
    for k in ("m", "t", "c"):                      # a scope this node offers may omit them
        key_cols.setdefault(k, np.zeros(ids.size, dtype=np.int64))
    keys = scope_row_key(scope, key_cols)

    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(
            f"filter labels: every one of the {ids.size} value(s) in column {column!r} is "
            f"non-finite, so no cut can be derived and no label could survive. That normally "
            f"means the column was written for a different set of ids than this raster's — "
            f"check that the measurement and the labels come from the same chain.")
    keep = np.zeros(ids.size, dtype=bool)
    cuts = np.full(ids.size, np.nan, dtype=float)
    no_cut: List[str] = []
    for gk in np.unique(keys):
        rows = np.flatnonzero(keys == gk)
        live = rows[finite[rows]]
        cut = _population_cut(values[live], method=method, level=level,
                              percentile=percentile)
        if cut is None:
            no_cut.append(f"{int(gk)} ({live.size} label(s))")
            continue
        cuts[rows] = cut
        keep[live] = (values[live] >= cut) if keep_side == "above" else \
                     (values[live] <= cut)
    n_skipped = int((~finite).sum())
    if lnote:
        ctx.progress(0, 1, "using " + lnote)
    if n_skipped:
        ctx.progress(0, 1, f"{n_skipped} label(s) skipped: {column} is not finite")
    if not keep.any():
        lo, hi = float(np.min(values[finite])), float(np.max(values[finite]))
        # every group's cut when they agree (the common case: one group, or `fixed`), else a
        # range — never a bare NaN, which is what a group that derived no cut leaves behind.
        live_cuts = cuts[np.isfinite(cuts)]
        shown = ("no derivable cut" if live_cuts.size == 0 else
                 f"cut {float(live_cuts[0]):g}" if float(np.ptp(live_cuts)) == 0.0 else
                 f"cuts spanning {float(live_cuts.min()):g}–{float(live_cuts.max()):g}")
        raise ValueError(
            f"filter labels: not one of the {int(finite.sum())} measured label(s) is "
            f"{keep_side} {shown} on column {column!r}, so the output would be empty. Those "
            f"values span {lo:g}–{hi:g} (median {float(np.median(values[finite])):g}) — "
            f"compare that against the cut, flip `keep` to the other side, or read the "
            f"column in the Spreadsheet and use method=fixed with a number from it.")

    kept_ids = ids[keep]
    out_layer = ctx.layer("name")
    if out_layer == layer:
        raise ValueError(
            f"filter labels: the output layer is also named {out_layer!r}, which would "
            f"overwrite the labels being filtered and leave no way to see what was dropped. "
            f"Give it its own name (default: `labels_kept`).")
    # One pass over the raster, plane by plane, so the progress bar is real and no
    # frame-sized boolean is held longer than the plane it belongs to. The parent's dtype is
    # preserved: an int32 raster stays int32 rather than being widened to int64 for nothing.
    kept_sorted = np.sort(kept_ids)
    out_raster = dense_output(tuple(raster6.shape), raster6.dtype,
                              tag=f"filtered_{ctx.node_id}")
    arr = out_raster.array
    n_units = ax.m * ax.t * ax.z * ax.c
    note = f"filtering labels on {column}"
    ctx.progress(0, n_units, note, frames=ax.t)
    for i, (m, t, z, c) in enumerate(_each_plane(ax)):
        plane = np.asarray(raster6[m, t, z, c])
        arr[m, t, z, c] = np.where(np.isin(plane, kept_sorted), plane, 0)
        ctx.progress(i + 1, n_units, note, frames=ax.t)

    # the surviving rows of EVERY column, plus the cut each was judged against — so the
    # decision is auditable in the Spreadsheet instead of being an invisible act of the node.
    new_cols: Dict[str, np.ndarray] = {k: np.asarray(v)[keep] for k, v in cols.items()}
    new_cols["cut"] = cuts[keep]
    out = ds.with_layer(Domain.VOXEL, out_layer, out_raster.seal())
    if no_cut:
        ctx.progress(0, 1, f"no cut derivable for group(s) {', '.join(no_cut)} — "
                           f"every label in them has the same {column}")
    return out.with_structure(StructureTable(Domain.LABEL, new_cols, layer=out_layer,
                                             z_kind=zk))


register_node(
    _compute_filter_labels, op_key="analysis.filter_labels", label="Filter Labels",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    inputs=[
        InDataset(description=
                  "The Dataset carrying the Label instance to filter — its raster and the "
                  "table whose column the cut is made on. No pixels are read: whatever the "
                  "column was measured from happened upstream."),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 description=
                 "Which Label instance to filter — a raster plus its table, from Segmentation, "
                 "Connected Components or Histogram Threshold. This layer is left UNCHANGED on "
                 "the wire; the survivors go to the output layer below, so both are available "
                 "downstream and the viewer can show what was dropped."),
        InString("column", "Column", field=False, default="mean_intensity",
                 description=
                 "Which per-label column supplies the one number each label is judged by. Any "
                 "column on the table works, which is the point — `mean_intensity` / "
                 "`max_intensity` / `total_intensity` / `area` from Measure's `stats`, "
                 "`eccentricity` / `solidity` / `perimeter` from its `shape` selector, the "
                 "physical µm areas from Object Metrics, or `n_sub` / `frac_above` from "
                 "Threshold Per Label. Naming a column the table does not carry is refused "
                 "with the list it does carry. Deliberately NOT a closed dropdown: a table can "
                 "hold columns no node in this catalog knows about."),
        InString("name", "Output layer", field=False, default="labels_kept",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the filtered result — one name into two domains, like Connected "
                 "Components: a Voxel raster holding only the surviving regions and a Label "
                 "table holding only their rows, with every original column plus `cut` (the "
                 "value each row was judged against). Ids are NOT renumbered, so id 7 still "
                 "means the same cell and any table or track measured upstream still joins on "
                 "it; the ids simply have gaps. Must differ from the input layer."),
        InFloat("level", "Level", unit="", field=False, default=0.0,
                available_in={"method": frozenset({"fixed"})},
                description=
                "The cut, typed in the column's OWN units — raw counts for an intensity "
                "column, µm² for `area_um2`, dimensionless for `solidity`. The one method "
                "that means the same thing on every frame of every experiment, and therefore "
                "the one to use for a hard biological cutoff (\"drop anything under 200 µm²\"). "
                "It is also the only method that applies to a population where every label "
                "carries the same value, since it asks the data nothing. Only shown for the "
                "`fixed` method."),
        InFloat("percentile", "Percentile", unit="", field=False, default=50.0,
                available_in={"method": frozenset({"percentile"})},
                description=
                "The cut as a percentile (0–100) of the label VALUES in each scope group — 50 "
                "keeps half the labels, 90 the top tenth. Note it ranks LABELS, not voxels: "
                "the count of surviving objects is what this controls, which makes it the way "
                "to say \"the brightest 20% of cells\" regardless of how bright they happen to "
                "be. Only shown for the `percentile` method."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("method", list(_FILTER_METHODS), default="otsu",
             description=
             "How the cut on the column is arrived at. The first five DERIVE it from the "
             "population of label values, so they let the objects split themselves when you "
             "do not know the number in advance; the last two are the explicit forms and take "
             "a socket each. Every one of them is applied within each `scope` group "
             "separately. A population in which every label carries the same value has no "
             "derivable cut and is skipped with a note — except under `fixed`, which needs "
             "nothing from the data.",
             choice_docs={
                 "otsu":
                     "Maximizes between-class variance over the label values — the standard "
                     "two-population split, and the right default when you believe there ARE "
                     "two kinds of object here (positive and negative cells, real objects and "
                     "debris). It always returns a split, so on a genuinely single population "
                     "it will still cut it in half.",
                 "li":
                     "Minimizes cross-entropy between the two classes. Sits LOWER than Otsu on "
                     "a right-skewed population — the usual shape when most objects are dim "
                     "and a few are very bright — so it keeps more of the middle. Use it when "
                     "Otsu is discarding objects you can see are real.",
                 "yen":
                     "An entropy criterion that typically lands HIGHER than Otsu, so it keeps "
                     "fewer, more clearly separated objects. The conservative choice when a "
                     "false positive is more costly than a miss.",
                 "triangle":
                     "Geometric: furthest from the line joining the population's peak to its "
                     "far tail. Assumes no bimodality at all, so it is the one to use when the "
                     "objects form a single skewed distribution with a bright tail rather than "
                     "two groups.",
                 "mean":
                     "The mean of the label values. Crude and perfectly predictable; on a "
                     "population with a few extreme outliers it is dragged toward them and "
                     "keeps too little, which is exactly when the four above earn their keep.",
                 "fixed":
                     "The `level` socket verbatim, in the column's own units — no statistics. "
                     "The only choice that means the same thing across experiments, so it is "
                     "how a published or protocol cutoff gets applied, and the only one that "
                     "survives a population where every label is identical.",
                 "percentile":
                     "A rank of the label values (the `percentile` socket): keeps a fixed "
                     "FRACTION of the objects. Predictable in count rather than in value, "
                     "which is what you want for a balanced comparison between conditions — "
                     "and its assumption is exactly that, so it keeps its share even from a "
                     "population where nothing is positive.",
             }),
        Mode("keep", ["above", "below"], default="above",
             description=
             "Which side of the cut SURVIVES. Inclusive at the boundary, so a label sitting "
             "exactly on the cut is kept. This is the whole difference between a bright-object "
             "filter and a debris filter on the same column and the same method.",
             choice_docs={
                 "above":
                     "Keep the labels at or ABOVE the cut, drop the rest — brighter, larger or "
                     "rounder objects depending on the column, and the default. On "
                     "`mean_intensity` this is the positive-cell filter.",
                 "below":
                     "Keep the labels at or BELOW the cut — the way to keep DIM or SMALL "
                     "objects, and the way to use this node as a debris remover on a column "
                     "where the artefacts are the high values (a huge merged blob's `area`, a "
                     "saturated speck's `max_intensity`).",
             }),
        # The shared factory (V2.27) stamps `role="scope"`, which is what makes the card's
        # footprint band edit this Mode. The description and the per-choice prose stay LOCAL:
        # this node's populations are table ROWS and its statistic is a cut on a column, both
        # more specific than the facility's general wording, and its `plane` scope carries a
        # refusal the shared text cannot mention.
        ScopeMode(_FILTER_SCOPES, default="series",
             description=
             "Which labels form the population the cut is derived over — the same four words, "
             "with the same meanings, as Threshold's `scope`, except that here they group "
             "TABLE ROWS by their own m/t/z/c rather than voxels by address. Channel is never "
             "pooled at any scope: two channels are two stains with different dynamic ranges. "
             "Narrower scopes adapt to drift at the cost of a smaller sample per cut; wider "
             "ones are more comparable. It costs nothing either way — the table is already in "
             "memory, and no scope reads more of the image. Ignored by method=fixed, whose cut "
             "does not come from a population.",
             choice_docs={
                 "plane":
                     "One cut per (position, timepoint, z, channel) — the labels in a single "
                     "plane. The most adaptive and the least stable: a plane holding four "
                     "objects derives its cut from four numbers. Needs a 2D-per-plane Label "
                     "table, since only there is `z` a plane index; refused on a volumetric "
                     "one.",
                 "volume":
                     "One cut per (position, timepoint, channel), pooling the labels across z. "
                     "The finest grouping a volumetric segmentation has, and the right one when "
                     "a z stack is one field of view.",
                 "series":
                     "One cut per (position, channel), pooling every timepoint — the default. "
                     "Each position gets its own cut, so differences in density or focus "
                     "between wells do not leak, while a bleaching series is still judged "
                     "against one consistent number instead of drifting frame by frame.",
                 "dataset":
                     "One cut per channel over EVERY label in the run, positions included. The "
                     "most comparable and the most sample-rich, and the only scope under which "
                     "a surviving-object count can be compared across positions directly — at "
                     "the cost of assuming the positions are interchangeable.",
             }),
    ],
    granularity=Granularity.WHOLE_VOLUME,
    description="Keep or drop WHOLE labels by thresholding one per-label column "
                "(mean_intensity, area, eccentricity, n_sub — anything the table carries), "
                "with the cut chosen by a histogram method over the population of label "
                "values, or given explicitly as a fixed level / percentile. `scope` groups "
                "the labels the way analysis.threshold's does (plane/volume/series/dataset, "
                "never pooling channels). Writes a NEW Label instance — raster + table, every "
                "original column plus the `cut` each row was judged against — with the "
                "surviving ids UNCHANGED, so upstream measurements and tracks still join on "
                "them. Non-finite values are excluded from both the population and the "
                "survivors rather than being given a side.",
)
