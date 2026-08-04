"""Transfer Structure (``transform.transfer_structure``) — Move an attribute across the detected-structure spine — Voxel↔Label↔Point↔Track and structure→Frame (15 pairs)."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs, is_structure
from nodegraph.engine import EvalContext
from nodegraph.reducers import reducer_docs
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import COORD_COLUMNS, TrackMembership

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _label_raster

# ── Transfer Structure (the geometric spine: Voxel↔Label↔Point↔Track, +→Frame) ──
#
# The structure half of the domain model. `nodegraph.bridges` has implemented these
# transfers since Phase 2 and `execute_bridge_plan` has chained them since C2, but until
# now NO NODE exposed any of it: `transform.transfer_domain` refuses every structure pair
# by construction, and 9 of the 10 bridge functions had zero callers outside the selftest.
# A user who segmented, detected or tracked could not move a value between those domains
# at all. This node is that surface.
#
# The matrix is exactly what `execute_bridge_plan` can RUN — 15 pairs, verified by
# exhaustive probe: the four-domain geometric spine fully connected (12 ordered pairs) plus
# the three structure→Frame reductions. Nothing else is offered, for the reason
# `transfer_domain` gives for staying lattice-only: a menu entry that always raises is a
# dead control (wire-node-v2 §4b clause 2).

#: source domains — the four the spine's bridges start from.
_XS_FROM: Tuple[str, ...] = ("voxel", "label", "point", "track")
#: target domains — the same four, plus Frame (structure→Frame is a registered bridge).
_XS_TO: Tuple[str, ...] = ("voxel", "label", "point", "track", "frame")
#: the reducers `_group_reduce` implements (the structure vocabulary). Deliberately NOT
#: `_TRANSFER_REDUCERS`: sigma_clip/trimmed_mean are lattice fusion reducers with no
#: group-wise form here, and `first` is not a group reduction.
_XS_REDUCERS: Tuple[str, ...] = ("mean", "sum", "count", "max", "min", "median")
#: the splat (`point_to_voxel`) implements only these three.
_XS_SPLAT_REDUCERS: FrozenSet[str] = frozenset({"sum", "mean", "max"})
#: pairs whose executor consumes a reducer — mirrors `transfer._REDUCING_HOPS`, resolved
#: through the ROUTE (voxel→track reduces at both of its hops; track→frame at its second).
_XS_REDUCING: FrozenSet[Tuple[str, str]] = frozenset({
    ("voxel", "label"), ("voxel", "track"), ("point", "voxel"), ("point", "label"),
    ("label", "track"), ("point", "track"), ("label", "frame"), ("point", "frame"),
    ("track", "frame"),
})
#: the two pairs whose route passes through Label without either endpoint naming it.
_XS_VIA_LABEL: FrozenSet[Tuple[str, str]] = frozenset({("voxel", "track"),
                                                       ("track", "voxel")})
#: pairs whose executor actually reads the label RASTER. Not the same as "Label is an
#: endpoint": label→track and track→label are pure table joins over the membership, and
#: label→frame just reduces the values it is handed — requiring a raster for those would
#: refuse a perfectly valid graph whose Label table outlived its raster.
_XS_NEEDS_RASTER: FrozenSet[Tuple[str, str]] = frozenset({
    ("voxel", "label"), ("label", "voxel"), ("point", "label"), ("label", "point"),
}) | _XS_VIA_LABEL
def _xs_pair(modes: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(modes.get("from_domain") or "voxel"), str(modes.get("to_domain") or "label"))
def _xs_check_pair(src: str, dst: str) -> None:
    """Refuse the 5 combos of the 4×5 grid that are not transfers."""
    if src == dst:
        raise ValueError(
            f"transfer structure: From and To are both {src!r} — there is nothing to "
            f"transfer. This node moves an attribute BETWEEN domains; to copy a column "
            f"under a new name, that is a different operation.")
    if (src, dst) == ("voxel", "frame"):
        raise ValueError(
            "transfer structure: voxel→frame is a LATTICE transfer — use Transfer Domain, "
            "which reduces over the dropped axes properly and also offers the fusion "
            "reducers (sigma_clip / trimmed_mean) this node's group vocabulary lacks.")
def _xs_structure_layer(ds: Dataset, domain: Domain, layer: str, *, node: str,
                        role: str) -> Dict[str, np.ndarray]:
    """The columns of structure instance ``layer`` on ``domain`` — refused if absent."""
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(domain) if a.layer == layer}
    if not cols:
        have = sorted({k[1] for k in ds.attributes if k[0] is domain and k[1]})
        raise ValueError(
            f"{node}: no {domain.value} instance {layer!r} on the input Dataset to use as "
            f"the {role}"
            + (f" (it carries {have})" if have else
               f" (it carries no {domain.value} instances at all)")
            + f". A {domain.value} transfer needs the {domain.value} table itself — it is "
              f"what defines the elements the values belong to.")
    return cols
def _xs_positions(cols: Mapping[str, np.ndarray], *, volumetric: bool,
                  node: str, layer: str) -> np.ndarray:
    """A Point instance's positions as ``(N, ndim)`` in the raster's axis order."""
    missing = [k for k in (("z", "y", "x") if volumetric else ("y", "x")) if k not in cols]
    if missing:
        raise ValueError(
            f"{node}: the point layer {layer!r} is missing the coordinate column(s) "
            f"{missing}, so its points have no position to sample at.")
    axes = ("z", "y", "x") if volumetric else ("y", "x")
    return np.stack([np.asarray(cols[a], dtype=float) for a in axes], axis=1)
def _xs_zkind(ds: Dataset, domain: Domain, layer: str) -> Optional[str]:
    return ds.structure_zkind(domain, layer) if is_structure(domain) else None
def _compute_transfer_structure(ctx: EvalContext) -> Dataset:
    """Move an attribute across the **detected-structure** spine — the node surface for
    :mod:`nodegraph.bridges` / :func:`nodegraph.transfer.execute_bridge_plan`.

    Resolved spec (grilled 2026-07-30):

    * **15 pairs**, ``From ∈ {voxel,label,point,track} × To ∈ {…,frame}``. The four
      self-pairs and ``voxel→frame`` are refused, naming the alternative.
    * **Geometry comes from the named endpoints.** A Label instance stores its raster as a
      Voxel layer of the SAME name as its table's layer (every producer in the catalog does
      this), a Point instance carries its positions in its own ``y/x/z`` columns, and a
      Track instance carries its membership in ``track_id/t/member_id``. So naming the
      source and target instances supplies everything 13 of the 15 pairs need. The two that
      route through Label without naming it — ``voxel→track`` and ``track→voxel`` — take a
      ``via_label`` socket, gated to exactly those two.
    * **Dimensionality is inherited, never levered** (§7b). The structure endpoint's own
      ``z_kind`` decides per-plane vs per-volume; two structure endpoints that disagree are
      refused rather than coerced, because rounding a sub-pixel ``z`` onto a plane index is
      the silent corruption that provenance exists to prevent.
    * **The reducer is refused, not hidden, where it is inert.** ``available_in`` gates a
      rectangle of mode values and "this hop reduces" is not one (it holds for
      ``label→track`` but not ``label→point``), so this follows ``transfer_domain``'s own
      precedent: remedy (b), an explicit refusal naming why.
    * **Footprint ``WHOLE_SERIES``, ``kernel_axes`` empty** — every input is an attribute
      array or a structure table already on the Dataset; the image provider is never read.
      Track membership spans T, hence the series.

    Values land on the target by **id**, scattered into that table's own row order, with
    NaN where the bridge produced nothing for a row — the rule ``analysis.measure`` already
    uses ("0 is a legal eccentricity, so a missing region must not read as a round one").
    """
    from nodegraph.transfer import Carrier, execute_bridge_plan, plan_transfer

    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    src_name, dst_name = _xs_pair(modes)
    _xs_check_pair(src_name, dst_name)
    src, dst = Domain(src_name), Domain(dst_name)
    reducer = str(modes.get("reducer") or "mean")
    if reducer not in _XS_REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r} — one of {list(_XS_REDUCERS)}")

    # ── the reducer must be live, and legal for this hop ──────────────────────
    if reducer != "mean" and (src_name, dst_name) not in _XS_REDUCING:
        raise ValueError(
            f"transfer structure: {src_name}→{dst_name} does not reduce — it "
            f"{'broadcasts one value onto many elements' if src_name == 'track' or dst_name == 'voxel' else 'maps each target element to exactly one source value'}"
            f", so reducer={reducer!r} would be ignored. Set Reduce back to 'mean'. "
            f"(The pairs that DO reduce many source values into one target value are: "
            f"{', '.join(f'{a}→{b}' for a, b in sorted(_XS_REDUCING))}.)")
    if (src_name, dst_name) == ("point", "voxel") and reducer not in _XS_SPLAT_REDUCERS:
        raise ValueError(
            f"transfer structure: point→voxel deposits each point onto its nearest voxel "
            f"and reduces only the COLLISIONS, which supports "
            f"{sorted(_XS_SPLAT_REDUCERS)} — not {reducer!r}. Splat with 'sum' (total per "
            f"voxel) or 'mean', then transfer that Voxel layer onward if you need another "
            f"statistic.")

    # ── endpoints ──────────────────────────────────────────────────────────────
    src_layer = ctx.layer("source_layer") if is_structure(src) else None
    dst_layer = ctx.layer("target_layer") if is_structure(dst) else None
    attr_name = ctx.layer("attr")
    out_name = str(ctx.params.get("name") or "").strip() or attr_name
    if out_name in COORD_COLUMNS:
        raise ValueError(
            f"transfer structure: {out_name!r} is one of the invariant coordinate columns "
            f"{list(COORD_COLUMNS)} that every structure table must carry — writing it "
            f"would overwrite the target's own geometry. Choose another output name.")

    src_cols = (_xs_structure_layer(ds, src, src_layer, node="transfer structure",
                                   role="source") if is_structure(src) else None)
    dst_cols = (_xs_structure_layer(ds, dst, dst_layer, node="transfer structure",
                                   role="target") if is_structure(dst) else None)

    # ── dimensionality: inherited from the structure endpoint(s), never levered ─
    zk_src = _xs_zkind(ds, src, src_layer)
    zk_dst = _xs_zkind(ds, dst, dst_layer)
    if zk_src is not None and zk_dst is not None and zk_src != zk_dst:
        raise ValueError(
            f"transfer structure: {src.value} {src_layer!r} is {zk_src!r} and {dst.value} "
            f"{dst_layer!r} is {zk_dst!r} — they were detected in different geometries, so "
            f"they do not share a coordinate space. A 'plane_index' table's z is which "
            f"PLANE an object is on; a 'subpixel' table's z is a coordinate within the "
            f"volume. Segment and detect both branches with the 2D/3D lever the same way.")
    z_kind = zk_src or zk_dst or ("subpixel" if ds.axes.z > 1 else "plane_index")
    volumetric = (z_kind == "subpixel")

    # ── the label raster, when the route touches Label ─────────────────────────
    raster_layer = None
    if (src_name, dst_name) in _XS_VIA_LABEL:
        raster_layer = str(ctx.layer("via_label") or "").strip()
        if not raster_layer:
            raise ValueError(
                f"transfer structure: {src_name}→{dst_name} has no direct bridge — it "
                f"routes through Label ({'voxel→label→track' if src_name == 'voxel' else 'track→label→voxel'}), "
                f"so it needs to know WHICH Label instance to pass through. Set the "
                f"`via_label` socket, or do it as two nodes "
                f"({'voxel→label then label→track' if src_name == 'voxel' else 'track→label then label→voxel'}), "
                f"which also lets you pick a different reducer for each step.")
    elif (src_name, dst_name) in _XS_NEEDS_RASTER:
        raster_layer = src_layer if src is Domain.LABEL else dst_layer
    raster6 = None
    if raster_layer is not None:
        raster6, zk_r = _label_raster(ds, raster_layer, node="transfer structure")
        if (zk_r == "subpixel") != volumetric:
            raise ValueError(
                f"transfer structure: the label raster {raster_layer!r} is {zk_r!r} but "
                f"this transfer resolved to {z_kind!r} from its endpoints — they must "
                f"agree, or regions would be grouped in a geometry they were not found in.")

    # ── the membership, when the route touches Track ───────────────────────────
    membership = None
    if src is Domain.TRACK or dst is Domain.TRACK:
        tcols = src_cols if src is Domain.TRACK else dst_cols
        tlayer = src_layer if src is Domain.TRACK else dst_layer
        miss = [k for k in ("track_id", "t", "member_id") if k not in tcols]
        if miss:
            raise ValueError(
                f"transfer structure: the track layer {tlayer!r} is missing {miss} — a "
                f"Track domain IS its membership (which member each track occupies at each "
                f"timepoint). Run track.objects or track.link upstream.")
        membership = TrackMembership(
            track_id=tcols["track_id"], t=tcols["t"], member_id=tcols["member_id"],
            member_domain=Domain.LABEL)

    # ── the source values ──────────────────────────────────────────────────────
    if is_structure(src):
        if attr_name not in src_cols:
            raise ValueError(
                f"transfer structure: the {src.value} layer {src_layer!r} has no column "
                f"{attr_name!r} (it carries {sorted(src_cols)}). Point `attr` at a column "
                f"that exists — analysis.measure adds the intensity ones, "
                f"analysis.object_metrics the motion ones.")
        src_vals = np.asarray(src_cols[attr_name], dtype=float)
        if src is Domain.TRACK:
            # A Track table is one row per (track, timepoint) OCCUPANCY, not one per track,
            # and `broadcast_track` wants one value per TRACK keyed by track_id. Collapse to
            # the first row of each track: a genuinely track-level column (track_length, or
            # anything transferred INTO the Track domain) is constant down a track, and
            # taking the first occupancy is what makes the id list unique either way.
            tid_all = np.asarray(src_cols["track_id"], dtype=np.int64)
            src_ids, first = np.unique(tid_all, return_index=True)
            src_vals = src_vals[first]
        else:
            src_ids = np.asarray(src_cols.get("id", np.arange(len(src_vals))),
                                 dtype=np.int64)
    else:
        vattr = ds.get(Domain.VOXEL, attr_name)
        if vattr is None:
            have = sorted({k[2] for k in ds.attributes if k[0] is Domain.VOXEL})
            raise ValueError(
                f"transfer structure: no Voxel layer {attr_name!r} to move"
                + (f" (this Dataset carries {have})" if have else "") + ".")
        src_vals = np.asarray(vattr.values, dtype=float)
        src_ids = None

    ax = ds.axes
    # ── voxel→track: the one route whose two hops need DIFFERENT unit granularity ──
    #
    # `mean-in-mask` is per-frame (each frame has its own raster) but `gather-by-track` is
    # whole-series (a track's identity IS its run across t), and `execute_bridge_plan`
    # chains a route at one granularity. Running it collapsed gathered only frame 0's
    # labels; running it per-frame would gather each frame separately, which is not a
    # track at all. So the first hop is completed here, frame by frame, into a per-label
    # value list — and what continues below is a plain Label→Track gather over the whole
    # series. (track→voxel is NOT symmetric: `broadcast_track` yields every member across
    # every t in one call and `paint-by-label` then only touches the ids present in the
    # frame it is given, so the ordinary per-frame loop is already correct there.)
    if (src_name, dst_name) == ("voxel", "track"):
        from nodegraph.bridges import voxel_to_label
        stage_ids: List[np.ndarray] = []
        stage_vals: List[np.ndarray] = []
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    for z in ([None] if volumetric else range(ax.z)):
                        sub = raster6[m, t, :, c]
                        val = src_vals[m, t, :, c]
                        if z is not None:
                            sub, val = sub[z], val[z]
                        i, v = voxel_to_label(val, sub, reducer)
                        stage_ids.append(i)
                        stage_vals.append(v)
        src = Domain.LABEL
        src_ids = (np.concatenate(stage_ids) if stage_ids
                   else np.array([], dtype=np.int64))
        src_vals = (np.concatenate(stage_vals) if stage_vals
                    else np.array([], dtype=float))
        src_cols = None                      # the values no longer come from a table
        raster6 = None                       # and the second hop needs no raster

    plan = plan_transfer(src, dst, reducer)
    # Per-unit loop: a plane each when the structures are per-plane, a volume each when they
    # are sub-pixel.
    #
    # T is collapsed for a Track TARGET only. Gathering into a track is what needs the whole
    # series at once — a track's identity IS its run across t, so its members must all be in
    # hand for one call. Going the other way does NOT: `broadcast_track` hands back a value
    # per member across every t, and the hop after it is per-frame (paint onto THIS frame's
    # raster, reduce within THIS frame). Collapsing t there wrote frame 0 and left the rest
    # untouched — track→voxel painted only the first frame and track→frame reported one
    # number for the whole series instead of one per frame.
    spans_t = dst is Domain.TRACK
    zs = [None] if volumetric else list(range(ax.z))
    units = [(m, t, z, c)
             for m in range(ax.m)
             for t in ([None] if spans_t else range(ax.t))
             for z in zs
             for c in range(ax.c)]

    def _slice6(arr6: np.ndarray, m, t, z, c) -> np.ndarray:
        """The (Z,Y,X) or (Y,X) block of a 6-D Voxel array for one unit."""
        sub = arr6[m, 0 if t is None else t, :, c]              # (Z, Y, X)
        return sub if z is None else sub[z]

    def _rows_of(cols: Mapping[str, np.ndarray], m, t, z, c) -> np.ndarray:
        """Boolean row selector for the structure rows belonging to one unit."""
        sel = (np.asarray(cols["m"], dtype=np.int64) == m)
        sel &= (np.asarray(cols["c"], dtype=np.int64) == c)
        if t is not None and "t" in cols:
            sel &= (np.asarray(cols["t"], dtype=np.int64) == t)
        if z is not None and "z" in cols:
            sel &= (np.rint(np.asarray(cols["z"], dtype=float)).astype(np.int64) == z)
        return sel

    def _membership_for(t) -> Optional[TrackMembership]:
        """The membership this unit's hop should see.

        Whole, except for **track→frame**: that route broadcasts each track's value onto its
        members and then reduces "in frame", but the chained executor's Frame hop reduces
        every value it is handed into ONE scalar. Handing it the whole membership therefore
        gave the same all-series number to every frame. Restricting the rows to this t makes
        the reduction what its name says — over the tracks active in THIS frame."""
        if membership is None or t is None or dst is not Domain.FRAME:
            return membership
        keep = membership.t == int(t)
        return TrackMembership(track_id=membership.track_id[keep], t=membership.t[keep],
                               member_id=membership.member_id[keep],
                               member_domain=membership.member_domain)

    # accumulators for the two output shapes
    out_vox = (np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
               if dst is Domain.VOXEL else None)
    out_frame = (np.full((ax.m, ax.t), np.nan, dtype=float)
                 if dst is Domain.FRAME else None)
    out_rows = (np.full(len(next(iter(dst_cols.values()))), np.nan, dtype=float)
                if is_structure(dst) else None)
    dst_ids_all = (np.asarray(dst_cols.get("track_id" if dst is Domain.TRACK else "id"),
                              dtype=np.int64) if is_structure(dst) else None)

    ctx.progress(0, max(1, len(units)), "transferring", frames=ax.t)
    for i, (m, t, z, c) in enumerate(units):
        raster = None if raster6 is None else _slice6(raster6, m, t, z, c)
        points = None
        pt_rows = None
        for dom, cols, lay in ((src, src_cols, src_layer), (dst, dst_cols, dst_layer)):
            if dom is Domain.POINT:
                pt_rows = _rows_of(cols, m, t, z, c)
                sub = {k: v[pt_rows] for k, v in cols.items()}
                points = _xs_positions(sub, volumetric=volumetric,
                                       node="transfer structure", layer=lay)
        # the carrier for this unit
        if src is Domain.VOXEL:
            car = Carrier(src, array=_slice6(src_vals, m, t, z, c))
        elif src is Domain.POINT:
            car = Carrier(src, ids=np.arange(int(pt_rows.sum())),
                          values=src_vals[pt_rows])
        else:
            # `src_cols is None` is the staged voxel→track case above: its per-label
            # values are already the whole series, so there is nothing to slice.
            rows = (_rows_of(src_cols, m, t, z, c)
                    if (src is Domain.LABEL and src_cols is not None) else slice(None))
            car = Carrier(src, ids=src_ids[rows], values=src_vals[rows])
        shape = (raster.shape if raster is not None
                 else ((ax.z, ax.y, ax.x) if volumetric else (ax.y, ax.x)))
        mem_here = _membership_for(t)
        if mem_here is not None and mem_here.n == 0:
            ctx.progress(i + 1, len(units), "transferring", frames=ax.t)
            continue                       # no track active in this frame → leave it NaN
        got = execute_bridge_plan(car, plan, axes=ax, label_raster=raster, points=points,
                                  membership=mem_here, shape=shape)
        # ── scatter the result ────────────────────────────────────────────────
        if dst is Domain.VOXEL:
            block = np.asarray(got.array, dtype=float)
            if z is None:
                out_vox[m, 0 if t is None else t, :, c] = block
            else:
                out_vox[m, 0 if t is None else t, z, c] = block
        elif dst is Domain.FRAME:
            out_frame[m, 0 if t is None else t] = float(np.asarray(got.values).reshape(()))
        else:
            rows = (pt_rows if dst is Domain.POINT
                    else (_rows_of(dst_cols, m, t, z, c) if dst is Domain.LABEL
                          else np.ones(len(out_rows), dtype=bool)))
            row_ids = dst_ids_all[rows]
            if dst is Domain.POINT and got.ids is not None and got.ids.size == len(row_ids):
                # Point ids from a bridge are the ROW ORDER within this unit, not table ids
                out_rows[np.flatnonzero(rows)] = np.asarray(got.values, dtype=float)
            else:
                lut = {int(i): float(v)
                       for i, v in zip(np.asarray(got.ids, dtype=np.int64).ravel(),
                                       np.asarray(got.values, dtype=float).ravel())}
                out_rows[np.flatnonzero(rows)] = [lut.get(int(i), np.nan) for i in row_ids]
        ctx.progress(i + 1, len(units), "transferring", frames=ax.t)

    if dst is Domain.VOXEL:
        return ds.with_layer(Domain.VOXEL, out_name, out_vox)
    if dst is Domain.FRAME:
        return ds.with_layer(Domain.FRAME, out_name, out_frame)
    return ds.with_layer(dst, out_name, out_rows, dst_layer)
def _layers_transfer_structure(params: Mapping, modes: Mapping):
    """The layer this node writes — announced for the edit-time catalog (V2.11).

    ``layer_out`` cannot express it: that declaration is per-TYPE and its domains must be a
    subset of ``adds_domains``, which is deliberately EMPTY here because the target domain
    is a per-instance Mode value (the same reason ``transform.transfer_domain`` and
    ``track.link`` leave theirs empty). Must never raise — ``propagate_meta`` calls it on
    every keystroke."""
    try:
        dst = str((modes or {}).get("to_domain") or "label")
        if dst == str((modes or {}).get("from_domain") or "voxel"):
            return ()
        name = str((params or {}).get("name") or "").strip() \
            or str((params or {}).get("attr") or "") or "mask"
        return ((Domain(dst), name),)
    except Exception:                                    # pragma: no cover - defensive
        return ()
register_node(
    _compute_transfer_structure, op_key="transform.transfer_structure",
    label="Transfer Structure", category="transform",
    # reads/adds stay EMPTY on purpose: both are per-INSTANCE here (whatever From/To say)
    # while NodeSpec's declarations are per-TYPE — the same reason transform.transfer_domain
    # and track.link leave theirs empty. The output layer is announced via extra_layers.
    extra_layers=_layers_transfer_structure,
    inputs=[
        InDataset(),
        InString("attr", "Attribute", field=False, default="mask",
                 layer_in_mode="from_domain",
                 description=
                 "Which attribute to move. On a structure source this is a COLUMN of the "
                 "source table (`mean_intensity`, `area`, `speed`); on a voxel source it is "
                 "a Voxel layer name (a mask, a distance field). The picker follows the "
                 "From lever, so changing From changes the names offered. It keeps this "
                 "name on the target unless you set Output name."),
        InString("source_layer", "Source layer", field=False, default="labels",
                 layer_in_mode="from_domain",
                 description=
                 "Which instance of the source domain to read — the segmentation, the "
                 "detection or the tracking, by the name its producer gave it. Not read "
                 "when From is voxel, which is a singleton domain with no instances. This "
                 "is also where the transfer gets its GEOMETRY: a Label instance's raster, "
                 "a Point instance's positions, a Track instance's membership all live "
                 "under this one name."),
        InString("target_layer", "Target layer", field=False, default="labels",
                 layer_in_mode="to_domain",
                 description=
                 "Which instance of the target domain receives the values, as a new column "
                 "on its table. It must already exist — a transfer adds a measurement to "
                 "objects that were already found, it does not create them. Not read when "
                 "To is voxel or frame, which have no instances (those targets get a new "
                 "layer of their own instead)."),
        InString("via_label", "Via label layer", field=False, default="",
                 layer_in=Domain.VOXEL,
                 # {voxel,track} x {voxel,track} is exactly the two routed pairs plus the
                 # two self-pairs, and the self-pairs are refused before anything reads
                 # this. Gating also exempts the socket from the reads_domains
                 # consistency check, which this node cannot satisfy: its domains are
                 # per-INSTANCE (the From/To levers), so reads_domains is empty by design.
                 available_in={"from_domain": frozenset({"voxel", "track"}),
                               "to_domain": frozenset({"voxel", "track"})},
                 description=
                 "Only for voxel→track and track→voxel, which have no direct bridge and "
                 "route through Label — this names WHICH segmentation to pass through, and "
                 "the transfer reduces at both steps (voxel→label→track means mean-in-mask "
                 "then mean-over-track, with your chosen reducer at each). Doing it as two "
                 "separate nodes instead lets you pick a different reducer per step."),
        InString("name", "Output name", field=False, default="",
                 description=
                 "What to call the result on the target. Empty keeps the source "
                 "attribute's name, which is what you usually want. Set it when two "
                 "transfers land on the same target instance — the second would otherwise "
                 "overwrite the first silently. The invariant coordinate columns "
                 "(id, m, t, c, z, y, x) are refused: writing one would destroy the "
                 "target's own geometry."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("from_domain", list(_XS_FROM), default="label", label="From",
                description=
                "Where the attribute comes from, on the detected-structure spine "
                "Voxel↔Label↔Point↔Track. Every pair the bridges can actually run is offered "
                "and nothing else; the Attribute and Source layer pickers both follow this "
                "lever, so set it first.",
                choice_docs=domain_docs(_XS_FROM)),
           Mode("to_domain", list(_XS_TO), default="track", label="To",
                description=
                "Where the attribute lands. Moving toward a COARSER structure (voxel→label, "
                "point→track) groups many values into one and uses Reduce; moving toward a "
                "finer one broadcasts (paint a label's value into its voxels, give every "
                "member of a track the track's value) and Reduce is inert. voxel→track and "
                "track→voxel have no direct bridge and route through Label, which is what "
                "the Via label socket names.",
                choice_docs=domain_docs(_XS_TO)),
           Mode("reducer", list(_XS_REDUCERS), default="mean", label="Reduce",
                description=
                "How a GROUP of source values becomes one target value — the voxels inside a "
                "region, the points inside a cell, a track's samples over time. It is read "
                "only by the pairs that coarsen; on a broadcasting pair it is refused rather "
                "than ignored. The splat to voxels supports only sum, mean and max.",
                choice_docs=reducer_docs(_XS_REDUCERS)),
           # `via_label` is gated by the pair, which is not a rectangle of (From, To) — the
           # compute refuses it instead (see the docstring). The MODE gating below is a
           # rectangle and is honest: `reducer` is dead only for pairs, not for a whole
           # From value, so it stays visible and is refused when inert.
           ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Move an attribute across the detected-structure spine — Voxel↔Label↔"
                "Point↔Track and structure→Frame (15 pairs). Coarsening reduces (mean in "
                "mask, points in region, gather over a track); refining broadcasts (paint "
                "by label, splat, broadcast a track's value onto its members). "
                "Dimensionality is inherited from the structures' own provenance.")
