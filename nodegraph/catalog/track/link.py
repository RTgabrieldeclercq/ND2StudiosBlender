"""Track Linking (``track.link``) — Frame-to-frame tracking → a Track membership: label mode links by maximum IoU overlap, point mode by nearest neighbour within max_distance (µm)."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import (
    _point_layers,
    _resolve_layer,
    _voxel_layers,
)

# ── Track Linking (frame-to-frame tracking → a Track membership, C3) ──────────

def _compute_track_link(ctx: EvalContext) -> Dataset:
    """Frame-to-frame tracking → a Track membership attached to the Dataset (C3).

    The ``target`` Mode selects label-vs-point linking. Label mode links the per-t
    Label raster (a Domain.VOXEL layer whose ids are global-unique per (m,t,z,c), as
    ``analysis.label`` emits) by maximum IoU overlap; point mode links the Point
    structure by nearest neighbour within ``max_distance`` (µm → the point coordinate
    space). Tracking runs per (m, c) across the whole T axis; track ids from different
    (m,c) runs are offset so they stay unique. The membership is attached via
    ``ds.with_structure(membership.to_table(layer=...))``."""
    from nodegraph.tracking import link_labels, link_points
    from nodegraph.structure import TrackMembership

    ds: Dataset = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    target = modes.get("target", "label")
    out_layer = ctx.layer("name")
    parts = []                                        # (track_id, t, member_id) triples
    offset = 0

    def _accumulate(mem) -> None:
        nonlocal offset
        if mem.n:
            parts.append((mem.track_id + offset, mem.t, mem.member_id))
            offset += int(mem.track_id.max())         # keep (m,c) runs' track ids unique

    if target == "label":
        # the one raster on the wire, whatever it is called (`_resolve_layer`); this branch
        # groups the raster by id and needs no Label table, so any Voxel layer qualifies
        src, _note = _resolve_layer(
            _voxel_layers(ds), ctx.layer("labels"), node="track.link (label)",
            socket="labels", what="Voxel layer", where="the `data` input",
            remedy="label mode links regions frame to frame by IoU overlap, so run "
                   "analysis.segment / analysis.label upstream — or set Target to `point` "
                   "to link a detection's dots instead", ctx=ctx)
        attr = ds.get(Domain.VOXEL, src)
        if attr is None:                             # pragma: no cover - _resolve_layer
            raise ValueError(f"track.link: no label raster {src!r} on the input Dataset")
        raster6 = attr.values                         # (m,t,z,c,y,x)
        iou = float(ctx.params.get("iou_threshold", 0.0))
        for m in range(ax.m):
            for c in range(ax.c):
                rasters_by_t = {t: raster6[m, t, :, c, :, :] for t in range(ax.t)}
                _accumulate(link_labels(rasters_by_t, iou_threshold=iou))
        member_domain = Domain.LABEL
    else:
        src, _note = _resolve_layer(
            _point_layers(ds), ctx.layer("points"), node="track.link (point)",
            socket="points", what="Point table", where="the `data` input",
            remedy="point mode links dots frame to frame, so wire a detection "
                   "(detect.spots / detect.particles) or transform.label_to_points upstream",
            ctx=ctx)
        cols = {k: ds.get(Domain.POINT, k, layer=src)
                for k in ("id", "m", "t", "c", "z", "y", "x")}
        if cols["id"] is None:                       # a Point layer with no `id` column
            raise ValueError(f"track.link: no Point structure {src!r} on the input Dataset")
        cid, cm, ct, cc, cz, cy, cx = (cols[k].values
                                       for k in ("id", "m", "t", "c", "z", "y", "x"))
        px = ctx.calib("pixel_size_um") or 1.0
        zs = ctx.calib("z_step_um") or 1.0            # µm-scaled coords → max_distance in µm
        coords = np.stack([cz * zs, cy * px, cx * px], axis=1)
        max_d = float(ctx.params.get("max_distance", float("inf")))
        for m in np.unique(cm).tolist():
            for c in np.unique(cc).tolist():
                grp = (cm == m) & (cc == c)
                if not np.any(grp):
                    continue
                positions_by_t = {}
                for t in np.unique(ct[grp]).tolist():
                    sel = grp & (ct == t)
                    positions_by_t[int(t)] = (cid[sel], coords[sel])
                _accumulate(link_points(positions_by_t, max_distance=max_d))
        member_domain = Domain.POINT

    if parts:
        track_id = np.concatenate([p[0] for p in parts])
        t_col = np.concatenate([p[1] for p in parts])
        member_id = np.concatenate([p[2] for p in parts])
    else:
        track_id = t_col = member_id = np.array([], dtype=np.int64)
    membership = TrackMembership(track_id=track_id, t=t_col, member_id=member_id,
                                 member_domain=member_domain)
    return ds.with_structure(membership.to_table(layer=out_layer))
register_node(
    _compute_track_link,
    op_key="track.link", label="Track Linking", category="analysis",
    adds_domains=frozenset({Domain.TRACK}),
    # reads Label OR Point, per the 'target' mode — stated per branch since V2.22. Unlike
    # track.objects the label branch reads the RASTER and nothing else (`attr.values` fed
    # to `link_labels`, which groups by id), so it does not require a Label table and a
    # graph whose raster outlived its table still links.
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.VOXEL}),
        "point": frozenset({Domain.POINT}),
    }},
    inputs=[
        InDataset(),
        InFloat("max_distance", "Max distance", unit="um", field=True, default=2.0,
                pick_kind="distance",
                available_in={"target": frozenset({"point"})},
                description=
                "How far a point may move between consecutive frames and still be linked, in "
                "microns — the gate on nearest-neighbour matching. Set it from the fastest "
                "real motion you expect over one frame interval: TOO SMALL and fast objects "
                "drop their tracks and restart as new ids, TOO LARGE and dense objects swap "
                "identities with each other, which is the more damaging error because it "
                "silently produces plausible-looking wrong trajectories. Physical, so it "
                "survives a change of magnification. Point mode only."),
        InFloat("iou_threshold", "IoU threshold", unit="", field=True, default=0.0,
                available_in={"target": frozenset({"label"})},
                description=
                "Minimum overlap between a region in frame t and one in t+1 for the two to be "
                "declared the same object — matching is greedy, highest overlap first. 0 (the "
                "default) accepts ANY positive overlap, which suits objects that move less "
                "than their own size; raise it to demand substantial overlap and refuse "
                "marginal matches, at the cost of breaking tracks whenever an object moves "
                "quickly or changes size. Above about 0.5 two shapes must share most of their "
                "area. Label mode only."),
        # the member source layer, one per target mode (the compute picks by mode)
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})},
                 description=
                 "Which label raster supplies the objects to link — a Connected Components or "
                 "Segmentation output. Its ids must be unique per frame, which both of those "
                 "guarantee. Label mode only; in point mode the Point layer is used instead."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})},
                 description=
                 "Which Point table supplies the objects to link — a Spot Detection or "
                 "Particle Detection output. Linking uses the point coordinates only, so "
                 "nothing about object shape or size constrains the match. Point mode only."),
        InString("name", "Output layer", field=False, default="tracks",
                 layer_out=(Domain.TRACK,),
                 description=
                 "Name of the Track membership this node writes: which object in each frame "
                 "belongs to which trajectory. Tracking runs independently per (m,c) and the "
                 "resulting track ids are offset so they never collide between them. "
                 "Downstream nodes select the tracks by this name."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("target", ["label", "point"], default="label", label="Track",
                description=
                "What is being linked across frames, which also selects the matching rule and "
                "therefore which parameters are live. Segmented REGIONS are matched by how "
                "much they overlap; detected POINTS by how close they are. Pick the one that "
                "matches what you actually produced upstream — the other mode's source layer "
                "and threshold are hidden.",
                choice_docs={
                    "label":
                        "Link regions of a label raster by maximum IoU overlap, greedily, "
                        "highest overlap first. Uses object SHAPE and size, so it is robust "
                        "for cells that move less than their own diameter — and it fails "
                        "exactly when they do not overlap between frames, however obvious the "
                        "correspondence looks to a human.",
                    "point":
                        "Link Point detections to their nearest neighbour within Max distance "
                        "(in µm). Works for objects of any size and for fast motion the "
                        "overlap rule cannot follow, but it knows nothing about shape, so in "
                        "a dense field two nearby objects can swap identities and produce "
                        "plausible-looking wrong trajectories.",
                })],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="Frame-to-frame tracking → a Track membership: label mode links by "
                "maximum IoU overlap, point mode by nearest neighbour within "
                "max_distance (µm). Runs per (m,c) across the whole T axis.",
)
