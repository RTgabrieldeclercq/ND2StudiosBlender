"""Track Objects (``track.objects``) — Frame-to-frame object tracking with five interchangeable linkers (centroid Hungarian / SerialTrack topology PTV / Cell-Tracker topology, fingerprint, mask-overlap IoU) → a Track membership + a…"""

from __future__ import annotations

import numpy as np

from typing import Dict

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
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import (
    _label_tables,
    _point_layers,
    _resolve_layer,
)
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Track Objects (the vendored v1 `track_objects` kernel — five linkers) ───────
#
# The richer sibling of `track.link`. Where `track.link` ships the two dep-free
# pure-numpy linkers in `nodegraph.tracking` (max-IoU overlap / nearest neighbour),
# this wraps the vendored v1 kernel and exposes its FIVE interchangeable linking
# methods. It is a SEPARATE node rather than a mode on `track.link` because the kernel
# imports numba and pandas at module scope, while `nodegraph.tracking` is deliberately
# dep-free — folding it in would make the built-in linker unimportable without numba.
#
# The kernel consumes measurement ROW-DICTS, which v2 Label/Point structure tables
# already carry (`id,m,t,c,z,y,x` + Label `area`, all in pixels — structure.py
# `_label_table`). Both trackers therefore read the same upstream layers and emit the
# same Track membership shape, so they are drop-in alternatives to each other.

#: v2 `method` mode value → the kernel's ``METHOD_*`` attribute name. The kernel's
#: dispatch ends in a bare ``else:`` that silently runs the centroid linker on an
#: unrecognized string, so the node maps + validates rather than passing one through.
_TRACK_OBJECT_METHODS = {
    "centroid": "METHOD_CENTROID",
    "serialtrack": "METHOD_SERIALTRACK",
    "topology": "METHOD_CT_TOPOLOGY",
    "fingerprint": "METHOD_CT_FINGERPRINT",
    "overlap": "METHOD_CT_OVERLAP",
}
def _compute_track_objects(ctx: EvalContext) -> Dataset:
    """Frame-to-frame object tracking via the vendored v1 ``track_objects`` kernel →
    a Track membership + a ``track_id`` column written back onto the member layer.

    **Resolved spec (§0 grill, 2026-07-27).**

    *Kind* analysis; reads a Label **or** Point structure layer (the ``target`` mode,
    mirroring ``track.link``) and adds the Track domain. *Members* keep their upstream
    layer; the tracker never re-derives geometry.

    Two of the five methods are exclusive to one target, and both refuse the other rather
    than running on evidence they cannot use: ``overlap`` intersects label rasters and is
    **Label-only**; ``serialtrack`` is a particle tracker that reads centroids and nothing
    else, so it is **Point-only** (2026-08-03).

    *Dimensionality* (2026-08-04). Four of the five linkers are two-column by
    construction, so this node is 2-D for them: a z-stack is tracked plane by plane, with
    z in the grouping key. ``serialtrack`` is the exception — SerialTrack is a 2-D *and*
    3-D method and the vendored kernel implements both — so it carries its own ``st_dim``
    Mode. At ``3D`` the member layer must have ``z_kind='subpixel'`` (a measured depth, as
    a 3-D detection emits), z leaves the grouping key and becomes a third coordinate, and
    it is rescaled by ``z_step_um / pixel_size_um`` into the lateral unit first, because
    the descriptor is built from Euclidean neighbour distances and anisotropic voxels would
    otherwise distort every one of them. The lever is checked against the layer's z_kind in
    both directions, and the resolved space is stamped as ``track_dim``.

    *Rows.* Each member row becomes one kernel row-dict. v2 Label tables already carry
    every key the kernel needs — ``y``/``x`` → ``centroid_y_px``/``centroid_x_px`` and
    ``area`` → ``area_px``, all in pixels (``structure._label_table``). Point tables have
    **no** ``area`` column, so ``area_px`` is 0.0 there. For the *fingerprint* linker that
    is harmless (area is a weighted cost term, not a gate), but for the *centroid* linker
    ``max_size_diff_frac`` IS a gate and the kernel neutralises it on zero areas — so that
    socket is both hidden on Point members and refused below if explicitly set.

    *Grouping.* The kernel groups by ``(segmentation_channel, m_position)`` and runs each
    group independently off ONE shared track-id counter. This node keys those as
    ``(f"c{c}z{z}", m)`` — a compound channel key — so every ``(m, c, z)`` is tracked
    independently, which is what makes a 2D-only kernel correct on a z-stack: planes are
    never linked to each other. Do **not** pre-split; one call keeps ids unique.

    *2D only.* Every linker builds a two-column ``[y, x]`` array and ``track_overlap``
    hard-raises unless its masks are ``(T,H,W)``. A 3D (``z_kind="subpixel"``) structure
    is refused with a pointer to ``track.link`` point mode, which links in true µm 3D.
    Dimensionality is inherited from the members' ``z_kind`` (§7b) — never a DimMode
    lever, which could silently disagree with the data.

    *Determinism (memo invariant, §10).* Verified empirically: every linker is
    repeat-stable, and row order changes only the track-id NUMBERING, never the induced
    partition. Two guards make the node fully deterministic anyway — rows are emitted in
    a canonical ``lexsort`` order ``(m, c, z, t, id)``, and the result is renumbered
    through ``tracking.build_membership`` (contiguous ``1..K`` by first appearance, rows
    sorted ``(track_id, t, member_id)``) so ids match ``track.link``'s conventions
    exactly. ``st_use_prev_results`` is pinned ``False``: its POD-GPR warm start draws on
    numpy's global RNG with no seed, which would break memo determinism outright.

    *Hard refusals* where the kernel would otherwise degrade in silence: an unknown
    method (dispatch falls through to centroid), ``overlap`` without its label raster
    (falls back to the fingerprint linker with hardcoded weights), ``overlap`` on Point
    members, duplicate member ids (the Cell-Tracker bridge keys a plain dict on
    ``(frame, label_id)`` and both duplicates end up untracked), ragged columns, an
    out-of-range ``t``, an inert ``max_size_diff_frac`` on arealess members, and
    ``ct_max_gap=0`` under ``overlap`` (the kernel floors that path at 1). Sockets a
    method does not consume are hidden from it rather than accepted and discarded —
    notably ``max_distance``, which the ``overlap`` linker never receives.

    *Sockets* are all ``field=False``: this node has no lattice iteration and no
    ``FieldContext``, so a wired per-voxel Field would be silently discarded.
    ``min_circularity``/``max_eccentricity`` are deliberately **omitted** — nothing in
    the v2 catalog emits circularity or eccentricity, so those sockets could only ever
    error; the kernel defaults disable both filters.

    *Footprint* ``WHOLE_SERIES`` over ``{t,z,y,x}`` (linking is global in T, and the
    overlap linker reads the whole label raster).
    """
    from nodegraph.kernels import track_objects as _tk
    from nodegraph.tracking import build_membership

    ds: Dataset = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    target = modes.get("target", "label")
    method_key = modes.get("method", "centroid")
    if method_key not in _TRACK_OBJECT_METHODS:
        raise ValueError(
            f"track objects: unknown method {method_key!r} "
            f"(choose one of {sorted(_TRACK_OBJECT_METHODS)})")
    method = getattr(_tk, _TRACK_OBJECT_METHODS[method_key])

    # SerialTrack is a PARTICLE tracker and is Point-only, the mirror image of `overlap`
    # (Label-only, refused for Points below). `_link_group_serialtrack` builds its rows
    # from `[[_cy(r), _cx(r)]]` and nothing else: no `area_px`, no raster. Its whole
    # method is the topology of neighbouring POSITIONS.
    #
    # A Label table does carry those centroids, so this combination would run — which is
    # exactly why it has to be refused rather than left to work by accident. Under
    # `target='label'` the node declares `reads_domains = {LABEL, VOXEL}` and offers a
    # Voxel-raster picker, because `reads_domains_by_mode` unions each mode's branch
    # independently and so cannot say "VOXEL for the other four methods only". The node
    # would therefore demand a raster this linker never opens, and present the shape/size
    # criteria (the area gate) that come with Label members to the one linker that cannot
    # consume them.
    if method_key == "serialtrack" and target != "point":
        raise ValueError(
            "track objects: the 'serialtrack' method is a particle tracker — it identifies "
            "each object by the ARRANGEMENT of its neighbouring positions and reads nothing "
            "but (y, x) centroids, so it has no use for a label raster or for object shape "
            "and size. Set Members to 'point' (a Spot / Particle Detection layer), or pick "
            "centroid / topology / fingerprint / overlap to track Label regions.")

    member_domain = Domain.LABEL if target == "label" else Domain.POINT
    # The one member table on the wire, whatever it is called (`_resolve_layer`). Candidates
    # are the structure TABLES, not `_label_instances`: every linker but `overlap` reads only
    # the table, and `overlap` raises its own message about the missing raster below.
    src, _note = _resolve_layer(
        _label_tables(ds) if target == "label" else _point_layers(ds),
        ctx.layer("labels") if target == "label" else ctx.layer("points"),
        node="track objects", socket="labels" if target == "label" else "points",
        what=f"{member_domain.value} table", where="the `data` input",
        remedy="these are the objects to link across frames, so run analysis.segment / "
               "analysis.label (Label members) or detect.spots / detect.particles (Point "
               "members) upstream", ctx=ctx)

    # ── pull + validate the member table ──────────────────────────────────────
    need = ("id", "m", "t", "c", "z", "y", "x")
    attrs = {k: ds.get(member_domain, k, layer=src) for k in need}
    if attrs["id"] is None:
        raise ValueError(
            f"track objects: no {member_domain.value} structure {src!r} on the input "
            f"Dataset (run analysis.label / detect.spots upstream, or point the layer "
            f"socket at the right layer)")
    missing = sorted(k for k, v in attrs.items() if v is None)
    if missing:
        raise ValueError(
            f"track objects: the {member_domain.value} layer {src!r} is missing the "
            f"invariant column(s) {missing} needed to build tracking rows")
    vals = {k: np.asarray(v.values) for k, v in attrs.items()}
    n = len(vals["id"])
    ragged = sorted(k for k, v in vals.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"track objects: column(s) {ragged} on layer {src!r} disagree in length with "
            f"'id' ({n}) — the rows would be built from misaligned data")

    # ── dimensionality: the `st_dim` lever, checked against the layer's own z_kind ────
    # SerialTrack is a 2D *and* 3D method (Yang et al. Table 1) and the vendored kernel
    # implements both: `_build_features_3d` / `_match_features_3d`, the 3-D ADMM loop and
    # `remove_outliers`' per-dimension constants all key off `coords.shape[1]`. The other
    # four linkers are two-column by construction (the Hungarian and Cell-Tracker linkers
    # correlate `(_cy, _cx)`; `overlap` needs `(T,H,W)` masks), so 3-D is SerialTrack-only.
    st_dim = modes.get("st_dim", "2D") if method_key == "serialtrack" else "2D"
    want_3d = st_dim == "3D"
    zk = ds.structure_zkind(member_domain, src) or "plane_index"

    if want_3d and zk != "subpixel":
        raise ValueError(
            f"track objects: SerialTrack 3D needs a depth to track in, but the "
            f"{member_domain.value} layer {src!r} has z_kind={zk!r} — its z is a plane "
            f"INDEX, not a measured position, so every object in a plane shares one z and "
            f"the topology descriptor would see a flat sheet. Run a 3D detection upstream "
            f"(detect.particles / detect.spots with the 3D lever, which emit "
            f"z_kind='subpixel'), or set SerialTrack dim to 2D to track each plane "
            f"independently.")
    if zk == "subpixel" and not want_3d:
        if method_key != "serialtrack":
            raise ValueError(
                f"track objects: the {method_key!r} method is 2D-only — the vendored linker "
                f"correlates a two-column (y, x) centroid array (and 'overlap' needs "
                f"(T,H,W) masks), so it cannot use the measured depth on layer {src!r} "
                f"(z_kind='subpixel'). Use the 'serialtrack' method, which tracks in true "
                f"3D, or track.link in point mode.")
        # serialtrack + st_dim=2D on a subpixel layer is PERMITTED and lossy: `zz` below
        # rounds z to a plane and folds it into the group key, so the measured depth is
        # discarded and planes are tracked in isolation. That is the right call when z_step
        # is coarse or z is too noisy to trust, and the wrong one for a bead field that
        # moves through depth — so the choice is recorded as `track_dim` in the output
        # metadata (nothing downstream could otherwise tell) and stated in the lever's
        # own `choice_docs`.

    ids = vals["id"].astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError(
            f"track objects needs globally-unique member ids, but layer {src!r} repeats "
            f"some (n={ids.size}, unique={np.unique(ids).size}). The Cell-Tracker linkers "
            f"key a dict on (frame, label_id), so duplicates silently drop BOTH rows.")
    mm = vals["m"].astype(np.int64)
    cc = vals["c"].astype(np.int64)
    tt = np.rint(vals["t"]).astype(np.int64)
    zz = np.rint(vals["z"]).astype(np.int64)          # 2D ⇒ z is the plane index
    if n and (int(tt.min()) < 0 or int(tt.max()) >= ax.t):
        raise ValueError(
            f"track objects: layer {src!r} has t outside the Dataset's T axis "
            f"(t∈[{int(tt.min())},{int(tt.max())}], T={ax.t})")
    yy = vals["y"].astype(float)
    xx = vals["x"].astype(float)
    area_attr = ds.get(member_domain, "area", layer=src)
    area = (np.asarray(area_attr.values, dtype=float) if area_attr is not None
            and len(area_attr.values) == n else np.zeros(n, dtype=float))

    # Canonical row order: the kernel hands out track ids in first-appearance order, so a
    # stable input order is what makes the raw ids reproducible (we renumber below anyway).
    order = np.lexsort((ids, tt, zz, cc, mm))         # primary key last

    if want_3d:
        # z stops being a GROUP KEY and becomes a COORDINATE. In 2D the kernel's
        # `(segmentation_channel, m_position)` grouping is what keeps a 2D linker honest on
        # a z-stack — planes are tracked in isolation and never linked to each other. In 3D
        # that is exactly wrong: an object must be free to move through depth, so every
        # plane of one channel has to land in ONE group.
        #
        # And z must be in the LATERAL length unit, because the descriptor is built from
        # Euclidean neighbour distances (`_build_features_3d`) — a raw plane index mixed
        # with lateral pixels would distort every radius and angle the match depends on by
        # the anisotropy ratio, which on a typical stack is 5-10x. Upstream never faces
        # this (its 3D examples are isotropic synthetic volumes with xstep=1), so there is
        # no MATLAB behaviour to copy here.
        px_um = ctx.calib("pixel_size_um")
        z_um = ctx.calib("z_step_um")
        if not px_um or not z_um:
            raise ValueError(
                "track objects: SerialTrack 3D needs both 'pixel_size_um' and 'z_step_um' "
                "to put z on the same scale as y/x — its topology descriptor is built from "
                "Euclidean neighbour distances, so an unscaled plane index would distort "
                f"every one of them (have pixel_size_um={px_um!r}, z_step_um={z_um!r}). "
                "Ingest a calibrated file, or set the calibration explicitly.")
        z_scale = float(z_um) / float(px_um)          # plane index → lateral px
        zc = vals["z"].astype(float) * z_scale
        rows = [{"segmentation_channel": f"c{int(cc[i])}",
                 "m_position": int(mm[i]),
                 "frame": int(tt[i]),
                 "label_id": int(ids[i]),
                 "centroid_z_px": float(zc[i]),
                 "centroid_y_px": float(yy[i]),
                 "centroid_x_px": float(xx[i]),
                 "area_px": float(area[i])}
                for i in order.tolist()]
    else:
        rows = [{"segmentation_channel": f"c{int(cc[i])}z{int(zz[i])}",
                 "m_position": int(mm[i]),
                 "frame": int(tt[i]),
                 "label_id": int(ids[i]),
                 "centroid_y_px": float(yy[i]),
                 "centroid_x_px": float(xx[i]),
                 "area_px": float(area[i])}
                for i in order.tolist()]

    # ── the overlap linker's (T,H,W) label masks, keyed like the row groups ───
    label_masks = None
    if method_key == "overlap":
        if member_domain is not Domain.LABEL:
            raise ValueError(
                "the 'overlap' method matches label masks by IoU and is unavailable for "
                "Point members (no raster to intersect) — pick centroid / topology / "
                "fingerprint / serialtrack, or track a Label layer.")
        raster_attr = ds.get(Domain.VOXEL, src)
        if raster_attr is None:
            raise ValueError(
                f"the 'overlap' method needs the Voxel label raster {src!r} that carries "
                f"the member ids (analysis.label emits the raster and the table under one "
                f"layer name). Without it the kernel silently falls back to the "
                f"fingerprint linker with hardcoded weights.")
        raster6 = raster_attr.values                  # (m,t,z,c,y,x); ids match the table
        label_masks = {}
        for m_i in np.unique(mm).tolist():
            for c_i in np.unique(cc).tolist():
                for z_i in np.unique(zz).tolist():
                    label_masks[(f"c{int(c_i)}z{int(z_i)}", int(m_i))] = \
                        raster6[int(m_i), :, int(z_i), int(c_i)]

    # `overlap` matches purely by mask IoU: the kernel's live path takes no distance
    # bound at all (`max_displacement_px` reaches only the fingerprint FALLBACK branch,
    # which the missing-raster refusal above makes unreachable). So the socket is hidden
    # for it — and calibration is not read either, since fencing the memo on a pixel size
    # that cannot change the result would invalidate cached tracks for nothing.
    if method_key == "overlap":
        max_disp_px = 100.0                   # the kernel default; never consulted
        # One socket serves both CT gap methods, but the kernel floors the overlap path at
        # `max(1, ct_max_gap)` while fingerprint honours 0 — so 0 would silently bridge a
        # one-frame hole here. Refuse rather than accept-and-rewrite (the vendored kernel
        # stays byte-verbatim; the deviation is documented in its .md as `>= 0`).
        if int(ctx.params.get("ct_max_gap", 3)) < 1:
            raise ValueError(
                "track objects: the 'overlap' linker floors its frame gap at 1, so "
                "ct_max_gap=0 would still bridge a one-frame hole instead of requiring "
                "consecutive detections. Use ct_max_gap>=1, or the 'fingerprint' method, "
                "which honours 0.")
    else:
        px = ctx.calib("pixel_size_um") or 0.1
        max_disp_px = max(1.0, to_pixels_v2(
            float(ctx.params.get("max_distance", 5.0)), "um", pixel_size_um=px))

    # The centroid linker's size gate divides by max(area); the kernel neutralises itself
    # on all-zero areas (`area_max[area_max <= 0] = 1.0` ⇒ size_diff ≡ 0), so on Point
    # members — which carry no `area` column, ever — an explicit gate would silently do
    # nothing and let the Hungarian solver swap identities it was set to keep apart.
    size_gate = float(ctx.params.get("max_size_diff_frac", 1.0))
    if method_key == "centroid" and size_gate < 1.0 and not area.any():
        raise ValueError(
            "track objects: 'Max size diff' gates the centroid linker on "
            f"|Δarea|/max(area), but layer {src!r} carries no 'area' column (Point tables "
            "never do), so the gate would be silently inert and identities could swap. "
            "Leave it at 1.0, or track a Label layer.")

    # ── link (the kernel mutates `rows` in place and returns the same list) ───
    kp = dict(
        max_displacement_px=max_disp_px,
        min_track_length=int(ctx.params.get("min_track_length", 2)),
        max_size_diff_frac=size_gate,
        max_frame_gap=int(ctx.params.get("max_frame_gap", 0)),
        method=method,
        st_n_neighbors=int(ctx.params.get("st_n_neighbors", 25)),
        st_smoothness=float(ctx.params.get("st_smoothness", 0.01)),
        st_use_prev_results=False,        # pinned — unseeded POD-GPR would break the memo
        st_ndim=3 if want_3d else 2,
        # The remaining st_* knobs have no socket, so the node pins them — and upstream
        # ships a DIFFERENT set per dimensionality, which is not interchangeable (the
        # 2-D values are 2.5x tighter). Pick by the lever, taking each from the shipped
        # example that matches this node's use case (pre-detected coordinates, no image
        # re-detection):
        #
        #                     2-D                          3-D
        #   source   Example_main_2D_hardpar_       Example_main_3D_hardpar_
        #            inc_coords_only.m              inc_coords_only.m
        #   gbSolver 3 (ADMM)                       2 (Regularization)
        #   outlrThres      2                       5
        #   distMissing     2                       5
        #   iterStopThres   1e-2                    1e-3
        #   smoothness      1e-2                    1e-1   (the `st_smoothness` socket)
        #   locSolver / n_neighborsMin / maxIterNum are 1 / 1 / 20 in both.
        #
        # `remove_outliers` picks its OWN neighbour count and fluctuation floor off
        # `coords.shape[1]` (27 / 0.075 px in 3-D, 40 / 0.1 px in 2-D), so those follow
        # `st_ndim` without being passed.
        st_mode="Incremental",
        st_loc_solver="Topology",
        st_solver="Regularization" if want_3d else "ADMM",
        st_n_neighbors_min=1,
        st_outlier_threshold=5.0 if want_3d else 2.0,
        st_max_iter=20,
        st_iter_stop_threshold=1e-3 if want_3d else 1e-2,
        st_dist_missing=5.0 if want_3d else 2.0,
        ct_n_neighbors=int(ctx.params.get("ct_n_neighbors", 5)),
        ct_topo_weight=float(ctx.params.get("ct_topo_weight", 0.3)),
        ct_area_weight=float(ctx.params.get("ct_area_weight", 0.3)),
        ct_max_gap=int(ctx.params.get("ct_max_gap", 3)),
        ct_min_iou=float(ctx.params.get("ct_min_iou", 0.1)),
    )
    if label_masks is not None:
        kp["label_masks"] = label_masks
    _tk.link_objects(rows, **kp)

    # ── renumber through the shared membership builder (track.link conventions) ─
    # Kernel gotcha #8 ("two-frame minimum", track_objects.md §6): EVERY linker
    # early-returns on a (m,c,z) group spanning fewer than 2 distinct frames, leaving its
    # rows track_id=None BEFORE the min_track_length post-pass runs. At min_track_length
    # <= 1 the user has asked for 1-frame tracks, so seed those rows as singletons (what
    # track.link emits) — otherwise a lone detection's fate would depend on whether an
    # UNRELATED object in the same group happens to exist at some other frame. At the
    # default (>= 2) `short_groups` is empty, so this is a no-op on the normal path.
    short_groups: set = set()
    if int(ctx.params.get("min_track_length", 2)) <= 1:
        gframes: Dict[tuple, set] = {}
        for r in rows:
            gframes.setdefault((r["segmentation_channel"], r["m_position"]),
                               set()).add(int(r["frame"]))
        short_groups = {k for k, fs in gframes.items() if len(fs) < 2}
    t_of: Dict[int, int] = {}
    per_track: Dict[int, list] = {}
    for r in rows:
        tid = r.get("track_id")
        if tid is None:                   # excluded, or shorter than min_track_length
            if (r["segmentation_channel"], r["m_position"]) in short_groups:
                t_of[int(r["label_id"])] = int(r["frame"])      # singleton — no links
            continue
        mid = int(r["label_id"])
        t_of[mid] = int(r["frame"])
        per_track.setdefault(int(tid), []).append(mid)
    links: list = []
    for members in per_track.values():    # chain each kernel track into pairwise links
        chain = sorted(members, key=lambda mid2: (t_of[mid2], mid2))
        links.extend(zip(chain, chain[1:]))
    mem = build_membership(t_of, links, member_domain)

    # ── emit: the Track table (+ per-row extras) and the member write-back ────
    if mem.n:
        _uniq, inverse, counts = np.unique(mem.track_id, return_inverse=True,
                                           return_counts=True)
        length_col = counts[inverse].astype(np.int64)
        m_of = dict(zip(ids.tolist(), mm.tolist()))
        c_of = dict(zip(ids.tolist(), cc.tolist()))
        m_col = np.array([m_of[int(i)] for i in mem.member_id.tolist()], dtype=np.int64)
        c_col = np.array([c_of[int(i)] for i in mem.member_id.tolist()], dtype=np.int64)
    else:
        length_col = m_col = c_col = np.array([], dtype=np.int64)
    out_layer = ctx.layer("name")
    # Bypasses TrackMembership.to_table (a hard-coded 3-key literal) to carry the per-row
    # extras. The viewer's gate is a superset test, so they ride along for the spreadsheet
    # and CSV export; the Track bridges read only the three membership arrays.
    # The Track table's own z_kind follows the space the linking happened in: in 3D the
    # tracks span depth, so calling them 'plane_index' would misreport their provenance
    # to §7b consumers exactly as the members' would be.
    track_tbl = StructureTable(Domain.TRACK, {
        "track_id": mem.track_id, "t": mem.t, "member_id": mem.member_id,
        "track_length": length_col, "m": m_col, "c": c_col,
    }, layer=out_layer, z_kind="subpixel" if want_3d else "plane_index")

    # Write-back: a `track_id` column on the MEMBER layer, in that layer's original row
    # order (0 = untracked, matching the bridges' drop_nonpositive background rule). This
    # is what makes the result reachable to the rest of the catalog — nothing in v2
    # consumes Domain.TRACK, so without it the tracking would be viewer/export-only.
    # Only this one column is re-emitted, so the layer's other columns keep their order
    # (with_structure explodes a table into per-column layers — re-emitting `id` in a
    # different order would silently misalign every sibling column).
    tid_of = dict(zip(mem.member_id.tolist(), mem.track_id.tolist()))
    back = np.array([tid_of.get(int(i), 0) for i in ids.tolist()], dtype=np.int64)

    prov_md = {"track_method": method_key,
               "track_member_domain": member_domain.value,
               "track_member_layer": src,
               # Which space the linking ran in — 2D per-plane or true 3D. Stamped because
               # `st_dim=2D` on a subpixel layer is a legitimate but lossy choice (depth
               # discarded, planes isolated) and nothing downstream could otherwise tell.
               "track_dim": "3D" if want_3d else "2D"}
    return (ds.with_structure(track_tbl)
              .with_structure(StructureTable(member_domain, {"track_id": back},
                                             layer=src, z_kind=zk))
              .with_metadata(**prov_md))
register_node(
    _compute_track_objects,
    op_key="track.objects", label="Track Objects", category="analysis",
    adds_domains=frozenset({Domain.TRACK}),
    # reads Label OR Point, per the 'target' mode — stated per branch since V2.22 rather
    # than left empty. The member rows come from the LABEL table (`ds.get(member_domain,
    # …)`), not from the raster, but VOXEL is required alongside it because the `labels`
    # socket is `layer_in=VOXEL`: what it names is a Label INSTANCE, and with no VOXEL
    # catalog on the edge the picker has no names to offer.
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.LABEL, Domain.VOXEL}),
        "point": frozenset({Domain.POINT}),
    }},
    inputs=[
        InDataset(),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 # Label members only, AND not for `serialtrack`, which is Point-only —
                 # see the refusal in the compute. Without the second key this picker
                 # would be offered for a combination that cannot run.
                 available_in={"target": frozenset({"label"}),
                               "method": frozenset({"centroid", "topology",
                                                    "fingerprint", "overlap"})},
                 description=
                 "Which label raster supplies the objects to track — a Connected Components or "
                 "Segmentation output. Tracking a label layer gives every linker access to "
                 "object SHAPE and SIZE as well as position, which is what the area and overlap "
                 "criteria need. A `track_id` column is written back onto this layer, so the "
                 "result is reachable from the object rows themselves. Label target only, and "
                 "not for `serialtrack`, which tracks Point members."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})},
                 description=
                 "Which Point table supplies the objects to track — a Spot or Particle Detection "
                 "output. Only positions are available, so size- and overlap-based criteria "
                 "cannot apply; a `track_id` column is written back onto this layer. Point "
                 "target only, and the only input `serialtrack` accepts."),
        InString("name", "Output layer", field=False, default="tracks",
                 layer_out=(Domain.TRACK,),
                 description=
                 "Name of the Track table this node writes — one row per (track, timepoint) with "
                 "the member id and the track's total length. Tracking runs independently per "
                 "(m,c) and ids are offset so they never collide. A matching `track_id` column "
                 "is also added to the member layer, which is how the rest of the catalog reads "
                 "the result."),
        InFloat("max_distance", "Max distance", unit="um", field=False, default=5.0,
                pick_kind="distance",
                available_in={"method": frozenset({"centroid", "serialtrack",
                                                   "topology", "fingerprint"})},
                description=
                "Largest distance an object may move between frames and still be linked, in "
                "microns — the search gate every position-based linker uses. Set it from the "
                "fastest real motion over one frame interval: TOO SMALL and fast objects break "
                "into several short tracks, TOO LARGE and crowded objects swap identities, which "
                "is worse because the trajectories still look plausible. Physical, so it survives "
                "a change of magnification. The `overlap` method ignores it and uses Min IoU "
                "instead."),
        InInt("min_track_length", "Min track length", unit="", field=False, default=2,
              description=
              "Discard tracks appearing in fewer than this many frames. 2 drops single-frame "
              "detections, which are usually noise rather than objects. Raising it cleans up "
              "fragmentary tracks but also removes genuinely short-lived objects and anything "
              "entering or leaving near the end of the series — so it BIASES any lifetime or "
              "count statistic computed afterwards. Applies to every method."),
        InInt("max_frame_gap", "Max frame gap", unit="", field=False, default=0,
              description=
              "How many consecutive frames an object may vanish for and still be rejoined to the "
              "same track. 0 requires it in every frame, so one missed detection ends the track "
              "and the object restarts with a new id. Raising it bridges detection dropouts and "
              "blinking, at the risk of joining two different objects across the gap. Read by the "
              "`centroid` method here; `fingerprint` and `overlap` have their own Max gap.",
              available_in={"method": frozenset({"centroid"})}),
        InFloat("max_size_diff_frac", "Max size diff", unit="", field=False, default=1.0,
                available_in={"method": frozenset({"centroid"}),
                              "target": frozenset({"label"})},
                description=
                "How much an object's size may change between frames and still be linked, as a "
                "FRACTION of its size — 1.0 allows a doubling or halving and effectively "
                "disables the check. Tighten it to stop a small object being linked to a large "
                "neighbour, which is a common identity swap in dense fields. Too tight and "
                "genuine growth, division, or a partly-cut-off object at the frame edge breaks "
                "the track. Needs object size, so it exists only for the `centroid` method on a "
                "label target."),
        InInt("ct_n_neighbors", "Neighbours", unit="", field=False, default=5,
              available_in={"method": frozenset({"topology"})},
              description=
              "How many nearest neighbours describe each object's local arrangement. The "
              "`topology` linker matches objects by that neighbourhood pattern as well as by "
              "position, which is what lets it follow a group that has moved or rotated as a "
              "whole. MORE neighbours make the descriptor more distinctive but reach further, "
              "so it starts including objects that are not really part of the same local "
              "structure and becomes sensitive to any of them appearing or vanishing; FEWER "
              "make it cheap but ambiguous between similar-looking neighbourhoods. Minimum 1. "
              "Topology method only."),
        InFloat("ct_topo_weight", "Topology weight", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"topology"})},
                description=
                "How much the neighbourhood pattern counts against raw distance when scoring a "
                "candidate link — the two are blended as "
                "(1−w)·distance + w·topology, so it is a straight trade-off in [0,1]. 0 "
                "ignores topology and reduces this to nearest-neighbour matching; 1 ignores "
                "distance and matches purely on arrangement, which can link objects across the "
                "frame. Raise it when objects move together and positions alone confuse them; "
                "lower it when objects move independently. Topology method only."),
        InFloat("ct_area_weight", "Area weight", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"fingerprint"})},
                description=
                "How much object SIZE similarity counts against distance, blended as "
                "(1−w)·distance + w·area-mismatch, in [0,1]. 0 matches on position alone; 1 "
                "matches on size alone. Raise it to stop a small object being linked to a "
                "large neighbour, the common identity swap in crowded fields — but a genuinely "
                "growing or dividing object then breaks its track, and so does one clipped by "
                "the frame edge. Fingerprint method only."),
        InInt("ct_max_gap", "Max gap", unit="", field=False, default=3,
              available_in={"method": frozenset({"fingerprint", "overlap"})},
              description=
              "How many consecutive frames an object may go undetected and still be re-linked "
              "to the same track. 0 requires it in every frame, so a single missed detection "
              "ends the track and the object reappears with a new id. RAISING it bridges "
              "blinking and dropout, at the risk of joining two different objects across the "
              "gap — and any gap-crossing link is an inference, not an observation. Read by "
              "the `fingerprint` and `overlap` methods; `centroid` has its own Max frame gap."),
        InFloat("ct_min_iou", "Min IoU", unit="", field=False, default=0.1,
                available_in={"method": frozenset({"overlap"})},
                description=
                "Minimum mask overlap for two regions in consecutive frames to be declared the "
                "same object, in [0,1]. This linker uses overlap INSTEAD of distance, so it "
                "needs objects that move less than their own size — it is the most reliable "
                "choice for dense, slow-moving cells and it fails outright on anything fast. "
                "LOWER accepts marginal overlaps and can link an object to a neighbour it "
                "merely grazes; HIGHER demands substantial overlap and breaks tracks when an "
                "object moves quickly or changes shape. Overlap method only."),
        InInt("st_n_neighbors", "ST neighbours", unit="", field=False, default=25,
              available_in={"method": frozenset({"serialtrack"})},
              description=
              "Size of the neighbourhood descriptor the SerialTrack particle tracker uses to "
              "recognise a particle by the pattern of those around it — the reason it survives "
              "large, rotating displacements that defeat nearest-neighbour matching. Much "
              "larger than the Cell-Tracker equivalent because dense particle fields need a "
              "wide descriptor to be distinctive. MORE is more robust and slower; too few "
              "becomes ambiguous in a uniform field. Minimum 2. SerialTrack method only."),
        InFloat("st_smoothness", "ST smoothness", unit="", field=False, default=0.01,
                available_in={"method": frozenset({"serialtrack"})},
                description=
                "Regularisation weight on the fitted global displacement field, as a ratio of "
                "smoothness to goodness-of-fit: 1 weights the two equally and visibly smooths, "
                "0.01 gives smoothness 1% of the weight. 0 interpolates the individual matches "
                "exactly, so one bad match distorts the field around it; HIGHER suppresses "
                "outlier matches and fills gaps where the field is sparse — but oversmoothing "
                "flattens genuine local variation, which matters if you go on to differentiate "
                "the field into strain. Scale-free: the same value means the same thing at any "
                "detection density or pixel size. SerialTrack method only."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Members",
                description=
                "What is being tracked: the regions of a Label raster or the detections of a "
                "Point table. Labels carry shape and size, so every linker's criteria are "
                "available; Points carry position only, which rules the mask-overlap linker "
                "out entirely. Two methods are exclusive to one target — `overlap` needs "
                "rasters to intersect and so is Label-only, `serialtrack` is a particle "
                "tracker and so is Point-only — and each refuses the other target rather "
                "than running on evidence it cannot use. Either way a `track_id` column is "
                "written back onto the member layer, which is how the rest of the catalog "
                "reads the result.",
                choice_docs={
                    "label":
                        "Track the regions of a Voxel label raster (Segmentation / Connected "
                        "Components). The richest input — area gates, shape fingerprints and "
                        "mask IoU all become usable — and the only target the `overlap` method "
                        "can run on, since it needs rasters to intersect. Refused by "
                        "`serialtrack`, which reads centroids only.",
                    "point":
                        "Track the rows of a Point table (Spot / Particle Detection). Positions "
                        "only: the area column is absent, so size-based gates neutralize "
                        "themselves and `overlap` is refused rather than silently falling back "
                        "to another linker. The only target `serialtrack` accepts.",
                }),
           # SerialTrack's own dimensionality. NOT the canonical `DimMode()`: a
           # `role="dim_lever"` Mode is drawn from `spec.dim_lever()`, which does not
           # consult `available_in`, so the header lever would show for all five methods
           # while only this one can use it — the dead control the charter forbids. A plain
           # Mode renders through `active_modes(state)`, which does respect the gate.
           Mode("st_dim", ["2D", "3D"], default="2D", label="SerialTrack dim",
                available_in={"method": frozenset({"serialtrack"})},
                description=
                "Whether SerialTrack links in the image plane or in true 3D. It is the one "
                "method here that implements both (the others correlate a two-column "
                "centroid array, and `overlap` needs rasters), and the choice is checked "
                "against the member layer's own z provenance: 3D REQUIRES a layer whose z "
                "is a measured depth (`z_kind='subpixel'`, what a 3D detection emits) and "
                "refuses a plane index, because every object in a plane would share one z "
                "and the descriptor would see a flat sheet. Changing this changes the "
                "tracks, not just their speed, and it is recorded as `track_dim`.",
                choice_docs={
                    "2D":
                        "Link within each z plane, independently — planes are part of the "
                        "grouping key, so an object is never linked across depth. The right "
                        "choice for a 2D acquisition, and a deliberate one on a 3D "
                        "detection when z_step is coarse or z is too noisy to trust: there "
                        "the measured depth is DISCARDED (z is rounded to a plane), which "
                        "will fragment any object that really does move through depth.",
                    "3D":
                        "Link in true 3D, as SerialTrack3D does: z becomes a third "
                        "coordinate instead of a grouping key, so one object is free to "
                        "move through depth, and the topology descriptor is built from 3D "
                        "neighbour geometry (radius + polar + azimuthal angle rather than "
                        "radius + angle). z is rescaled by z_step/pixel_size first so "
                        "anisotropic voxels cannot distort those distances, which means "
                        "both calibrations must be known. Needs a 3D detection upstream.",
                }),
           Mode("method", ["centroid", "serialtrack", "topology", "fingerprint",
                           "overlap"], default="centroid", label="Method",
                description=
                "Which linker decides who is who between frames. All five produce the same "
                "Track membership; they differ in what EVIDENCE they use — position alone, the "
                "local arrangement of neighbours, object shape, or mask overlap — and therefore "
                "in how they fail when objects are dense or move far. Each method's own "
                "parameters appear with it and the others are hidden.",
                choice_docs={
                    "centroid":
                        "Nearest-neighbour matching by centroid distance, solved globally "
                        "(Hungarian) so the whole frame's assignment is optimal rather than "
                        "greedy. The fastest and the right default for sparse objects; it can "
                        "bridge missed frames and gate on area change, but with nothing but "
                        "distance to go on it swaps identities in a crowd.",
                    "serialtrack":
                        "Topology-based particle tracking (a from-scratch port of FranckLab's "
                        "SerialTrack): each object is described by the ARRANGEMENT of its "
                        "neighbours and matched on that descriptor, then the field is "
                        "regularized globally. Built for dense fields of near-identical "
                        "particles — beads in a gel — where distance alone is hopeless. "
                        "POINT MEMBERS ONLY: it reads (y, x) centroids and nothing else, so "
                        "Label members are refused rather than silently ignoring the shape and "
                        "size they carry. The most expensive option, and it wants many "
                        "neighbours (25 by default) to be reliable.",
                    "topology":
                        "Cell-Tracker's topology linker: cost is `(1-w)·distance + w·topology`, "
                        "so neighbourhood pattern is blended with proximity by an explicit "
                        "weight, and the assignment is Hungarian. The middle ground between "
                        "`centroid` and `serialtrack` — better than distance alone in a "
                        "monolayer, far cheaper than full PTV. Vendored from ND2Studios' own "
                        "Cell-Tracker.",
                    "fingerprint":
                        "Cell-Tracker's spatial fingerprint: matches on position AND relative "
                        "SIZE, `(1-w)·distance + w·area`, and can bridge gaps of several "
                        "frames. Use it when objects differ in size and that difference is "
                        "stable — it keeps a big cell from being confused with a small "
                        "neighbour — and avoid it where objects grow or divide during the "
                        "series.",
                    "overlap":
                        "Cell-Tracker's mask-overlap linker: two regions are the same object "
                        "when their masks INTERSECT enough (Min IoU). The most reliable rule "
                        "for cells that move less than their own diameter, because it uses "
                        "the full mask rather than a point, and it fails outright on fast "
                        "motion — no overlap, no link. Label members only; its frame gap is "
                        "floored at 1.",
                })],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Frame-to-frame object tracking with five interchangeable linkers "
                "(centroid Hungarian / SerialTrack topology PTV / Cell-Tracker topology, "
                "fingerprint, mask-overlap IoU) → a Track membership + a track_id column "
                "on the member layer. 2D per (m,c,z); Label or Point members. The richer "
                "alternative to track.link. NOTE: serialtrack is 1–2 orders of magnitude "
                "slower than the others (≈8-18 s on 1k–15k detections, plus a one-time "
                "numba JIT) — the cheap default is centroid.")
