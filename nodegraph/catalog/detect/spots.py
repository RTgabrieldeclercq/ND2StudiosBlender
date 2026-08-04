"""Spot Detection (``detect.spots``) — LoG/DoG spot detection → Points (bright or dark polarity);."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import fold_units
from nodegraph.registry import (
    DimMode,
    Granularity,
    InDataset,
    InFloat,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.progress import _parallel_progress

# ── Spot detection (LoG → Points) ───────────────────────────────────────────────

def _compute_spots(ctx: EvalContext) -> Dataset:
    """Blob detection → a Point structure table. ``method`` picks LoG (``blob_log``) or
    DoG (``blob_dog``); ``polarity`` detects bright spots (default) or dark spots (the
    normalized image inverted). The metadata-intelligent radii derive from the
    diffraction limit; radius↔σ uses ``σ = r/√ndim`` (2D vs 3D, anisotropic axial σ from
    ``z_step_um``). 2D detects per plane (``z_kind="plane_index"``); 3D in the volume
    (subpixel z). Ids are global-unique across the whole detection."""
    from skimage.feature import blob_dog, blob_log
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("spot detection needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    blob_fn = blob_dog if modes.get("method") == "dog" else blob_log
    dark = modes.get("polarity") == "dark"
    px = ctx.calib("pixel_size_um") or 0.1
    thr = float(ctx.params.get("threshold", 0.1))
    layer = ctx.layer("name")
    is_3d = ctx.is_volume
    # Anisotropic 3D: the axial σ derives from z_step_um, the lateral from pixel_size_um
    # (the detector takes a per-axis σ sequence; a scalar σ would search Z at the lateral
    # scale). Read z_step_um ONLY in 3D so a 2D pull isn't memo-fenced on it (R1).
    zs = (ctx.calib("z_step_um") or 0.5) if is_3d else None

    # radius r ↔ Gaussian σ (the blob detector's characteristic scale): σ = r/√ndim.
    # Floor at 1.0 voxel: scipy's discretized LoG (gaussian_laplace, used by blob_log)
    # is ill-conditioned below ~1 voxel — a sub-voxel σ makes the second-derivative
    # kernel stop summing near zero and return a large spurious uniform response, which
    # blob_log then reports as thousands of false blobs (you cannot resolve a blob
    # smaller than a voxel anyway). Common anisotropic z-steps drive the axial σ
    # sub-voxel, so the floor is load-bearing, not cosmetic.
    def sig_xy(r_um: float, ndim: int) -> float:
        return max(1.0, (r_um / px) / np.sqrt(ndim))

    def sig_z(r_um: float) -> float:                        # anisotropic axial σ (3D only)
        return max(1.0, (r_um / zs) / np.sqrt(3))           # LoG floor (see sig_xy)

    # Per-channel radii (C8 / H12): min/max_radius DERIVE from THIS channel's emission λ
    # (0.61·λ/NA), so each channel gets its own diffraction-limited scale — unless the user
    # pinned a value (one widget ⇒ it applies to every channel). The 3D axial radius is a
    # user override else the per-channel lateral radius (paired-float pattern → tracks it).
    def channel_sigmas(c: int):
        ch = ctx.channel(c)
        rmin = float(ch.param("min_radius")); rmax = float(ch.param("max_radius"))
        if is_3d:
            rmin_z = float(ctx.params["min_radius_z"]) if "min_radius_z" in ctx.params else rmin
            rmax_z = float(ctx.params["max_radius_z"]) if "max_radius_z" in ctx.params else rmax
            return ((sig_z(rmin_z), sig_xy(rmin, 3), sig_xy(rmin, 3)),
                    (sig_z(rmax_z), sig_xy(rmax, 3), sig_xy(rmax, 3)))
        return (sig_xy(rmin, 2), sig_xy(rmax, 2))

    def prep(a: np.ndarray) -> np.ndarray:
        """Normalize to [0,1]; invert for dark-spot polarity so a dark blob reads as
        a bright peak the detector can find. A flat/blank image has no spots of EITHER
        polarity — guard it BEFORE inverting (else dark would turn all-zeros into a
        constant-1 field and flood the detector with false positives)."""
        mn, mx = float(a.min()), float(a.max())
        if mx <= mn:
            return np.zeros_like(a)
        v = (a - mn) / (mx - mn)
        return 1.0 - v if dark else v

    # Resolve every channel's σ pair HERE, on the calling thread, before any fan-out.
    # `channel_sigmas` goes through `ctx.channel(c).param(...)`, which evaluates the socket
    # `derive` against `ctx.env.metadata` — the engine's RECORDING proxy. Calling it from a
    # pool worker would mutate the ReadContext's plain dict from several threads at once and
    # could land after the compute returns, where the frozen-reads guard raises (V2.02 §8 /
    # V2.04 §6b). There are only `ax.c` channels, so hoisting costs nothing.
    sigmas = {c: channel_sigmas(c) for c in range(ax.c)}

    tables = []
    # Parallel detection → SERIAL ORDERED append (V2.14). The order is load-bearing: the
    # merge below assigns `id` as `arange` over the CONCATENATED rows, so which point gets
    # which id is decided by the order tables land in this list.
    #
    # Note the unit order is (m, t, c, z) — c OUTSIDE z, as the original nesting had it,
    # because `channel_sigmas(c)` was hoisted per channel. Rebuilding it as the more usual
    # (m, t, z, c) would renumber every point.
    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c) for z in range(ax.z)])
    tick = _parallel_progress(ctx, len(units), "detecting", frames=ax.t)

    def _detect_one(unit):
        m, t, z, c = unit
        min_sig, max_sig = sigmas[c]                     # pre-resolved (see above)
        if z is None:
            vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                         0, ax.x).astype(float)
            blobs = blob_fn(prep(vol), min_sigma=min_sig, max_sigma=max_sig,
                            threshold=thr)
            return (point_table(blobs[:, :3], m=m, t=t, c=c, z_kind="subpixel",
                                layer=layer) if len(blobs) else None)
        plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
        blobs = blob_fn(prep(plane), min_sigma=min_sig, max_sigma=max_sig, threshold=thr)
        return (point_table(blobs[:, :2], z=z, m=m, t=t, c=c, z_kind="plane_index",
                            layer=layer) if len(blobs) else None)

    def _collect(_i, _unit, tbl):
        if tbl is not None:
            tables.append(tbl)
        tick()

    fold_units(_detect_one, units, _collect)
    zk = "subpixel" if is_3d else "plane_index"
    if not tables:
        merged = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=layer)
    else:
        cols = {k: np.concatenate([tb.columns[k] for tb in tables])
                for k in tables[0].columns}
        cols["id"] = np.arange(len(cols["id"]), dtype=np.int64)   # global-unique ids
        merged = StructureTable(Domain.POINT, cols, layer=layer, z_kind=zk)
    return ds.with_structure(merged)
register_node(
    _compute_spots, op_key="detect.spots", label="Spot Detection", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InFloat("min_radius", "Min radius", unit="um", field=True, default=0.2,
                pick_kind="radius", pick_peer="max_radius",
                derive="0.5*0.61*(emission_nm or 520)/(na or 1.4)/1000",
                description=
                "Smallest spot radius to look for, in microns — the bottom of the size band "
                "searched. Spots smaller than this are missed; setting it too small wastes "
                "time and starts detecting noise peaks as spots. Auto derives it from HALF "
                "the current optics' diffraction limit (emission λ and NA), i.e. just below "
                "the smallest real point source, which is usually right. Radius is converted "
                "to the filter's σ as r/√ndim, so the same radius means the same physical "
                "spot in 2D and 3D."),
        InFloat("max_radius", "Max radius", unit="um", field=True, default=0.6,
                pick_kind="radius", pick_peer="min_radius",
                derive="1.5*0.61*(emission_nm or 520)/(na or 1.4)/1000",
                description=
                "Largest spot radius to look for, in microns — the top of the size band. "
                "Spots larger than this are missed or split into several detections, and "
                "runtime grows with the width of the band because a separate filter scale is "
                "evaluated across it. Auto uses 1.5× the diffraction limit. Must exceed Min "
                "radius to leave a band to search."),
        # 3D-only axial radii (paired-float pattern) — default to the lateral radius
        InFloat("min_radius_z", "Min radius Z", unit="um_axial", field=True, default=0.2,
                available_in={"dim": frozenset({"3D"})},
                description=
                "Smallest spot EXTENT ALONG Z, in microns — separate from the lateral radius "
                "because a diffraction-limited spot is an ellipsoid roughly 3× taller than "
                "wide, and forcing one radius on both would make the detector look for a "
                "sphere that real optics never produce. 3D only."),
        InFloat("max_radius_z", "Max radius Z", unit="um_axial", field=True, default=0.6,
                available_in={"dim": frozenset({"3D"})},
                description=
                "Largest spot extent along Z, in microns — the axial top of the size band, "
                "the axial counterpart of Max radius. With a coarse z step a spot may span "
                "only one or two planes, and a value below the z step cannot be resolved at "
                "all. 3D only."),
        InFloat("threshold", "Threshold", unit="", field=True, default=0.1,
                pick_kind="level",
                description=
                "Minimum filter response for a peak to be reported — the sensitivity knob "
                "and the main precision/recall control. LOWER finds dimmer spots and more "
                "false positives; HIGHER keeps only strong ones. It is measured on the "
                "NORMALIZED filter response, not on image intensity, so it does not scale "
                "with your data's units and 0.1 is a reasonable starting point regardless of "
                "bit depth. If a run returns nothing, this is the first thing to lower."),
        InString("name", "Output layer", field=False, default="spots",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this node writes: one row per detected spot with "
                 "its id and position (plus a per-plane z index in 2D, a subpixel z in 3D). "
                 "Downstream nodes — Track Linking, Cluster Points, Tessellate — select it "
                 "by this name."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("method", ["log", "dog"], default="log",
                description=
                "Which scale-space blob filter finds the peaks. Both search the radius band "
                "between Min and Max radius and both report the same kind of result; they "
                "trade accuracy against speed, so switching changes how many spots you get "
                "at the margins rather than what a spot means.",
                choice_docs={
                    "log":
                        "Laplacian of Gaussian, evaluated at a series of scales across the "
                        "radius band. The more accurate of the two for both position and "
                        "size, and the better choice for faint or closely spaced spots; it "
                        "is also the slower, since every scale is a separate filter pass.",
                    "dog":
                        "Difference of Gaussians — a cheap approximation of the same "
                        "response, computed from pairs of blurs. Several times faster on "
                        "large stacks and near-identical on well-separated bright spots, but "
                        "it samples scale more coarsely, so radius estimates are rougher and "
                        "marginal detections can appear or vanish.",
                }),
           Mode("polarity", ["bright", "dark"], default="bright",
                description=
                "Whether a spot is brighter or darker than its surroundings. The image is "
                "normalized to [0,1] per unit and, for dark polarity, INVERTED before "
                "detection, so a dark spot reads as a peak the same filter can find. Picking "
                "the wrong polarity finds nothing (or only halos), rather than finding the "
                "same spots with a sign flipped.",
                choice_docs={
                    "bright":
                        "Detect intensity MAXIMA — fluorescent puncta, beads, labelled foci. "
                        "The normal case for fluorescence, and the default.",
                    "dark":
                        "Detect intensity MINIMA by inverting the normalized image first — "
                        "for transmitted-light data where cells or particles are dark on a "
                        "bright field, or for holes and voids. A blank plane is caught before "
                        "inversion, so it yields no spots instead of a flood of false ones.",
                })],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="LoG/DoG spot detection → Points (bright or dark polarity); radii "
                "derive from the diffraction limit (σ = r/√ndim); 2D per-plane vs 3D.")
