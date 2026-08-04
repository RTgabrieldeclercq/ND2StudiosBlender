"""Particle Detection (``detect.particles``) — Particle/small-object detection → Points…"""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    DimMode,
    Granularity,
    InBool,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX
from nodegraph.catalog._shared.units import to_pixels_v2

# ══ Ported v1 analysis kernels (Phase 7 capability gaps, V2.05 §2) ══════════════
#
# Each wraps a vendored pure-compute kernel in `nodegraph.kernels.*` (byte-verbatim
# ND2Studios v1.45), lazily imported inside the compute so the engine core stays
# importable without the heavy per-kernel deps. The kernels act on ONE frame/volume;
# the node owns the m/t/c loop, derives voxel_size from calibration, and attaches the
# result (image / Voxel mask / Point / Label / Track). See each kernel's `.md`.


# ── Particle detection (LoG maxima or component centroids → Points) ────────────

def _compute_particles(ctx: EvalContext) -> Dataset:
    """Particle / small-object detection → a Point structure. A general blob/particle
    detector (beads, puncta, foci, …) ported from the v1 ``bead_detect`` kernel. ``mode``
    picks LoG local-maxima (sub-voxel parabola refinement) or connected-component centroids
    (radial-symmetry refinement in 3D). 2D detects per plane (``z_kind="plane_index"``), 3D
    in the volume (subpixel z). Ids are global-unique across the whole detection.

    Resolved spec: category analysis; op ``detect.particles``; **Point** output; DimMode
    lever (2D WHOLE_PLANE / 3D WHOLE_VOLUME, ``kernel_axes`` per dim). ``min_distance``
    µm→px via ``pixel_size_um`` (doubles as the LoG σ and the NMS radius, in voxels);
    ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` slowest-first. Kernel:
    :func:`nodegraph.kernels.bead_detect.detect_beads` (numpy/scipy/numba)."""
    from nodegraph.kernels.bead_detect import detect_beads
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("particle detection needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)                                  # (dz, dy, dx) slowest-first
    min_dist_px = max(1.0, to_pixels_v2(float(ctx.params.get("min_distance", 0.5)),
                                        "um", pixel_size_um=px))
    kp = {
        "detect_mode": modes.get("mode", "log"),
        "min_distance_px": min_dist_px,
        "threshold": float(ctx.params.get("threshold", 0.0)),
        "min_intensity": float(ctx.params.get("min_intensity", 0.0)),
        "subpixel": bool(ctx.params.get("subpixel", True)),
        "min_size": max(1, int(ctx.params.get("min_size", 1))),
    }
    layer = ctx.layer("name")
    is_3d = ctx.is_volume
    tables = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                kpp = {**kp, "m": m, "frame": t}
                if is_3d:
                    vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                                 0, ax.x).astype(float)
                    pts, _rows = detect_beads(vol, vox, kpp)
                    if len(pts):
                        tables.append(point_table(pts[:, :3], m=m, t=t, c=c,
                                                  z_kind="subpixel", layer=layer))
                else:
                    for z in range(ax.z):
                        plane = prov.get_region(0, m, t, z, c, 0, ax.y,
                                                0, ax.x).astype(float)
                        pts, _rows = detect_beads(plane, vox, kpp)
                        if len(pts):
                            # 2D fallback forces the z-column to 0 → record the true
                            # plane index (take the (y,x) columns, stamp z=z)
                            tables.append(point_table(pts[:, 1:3], z=z, m=m, t=t, c=c,
                                                      z_kind="plane_index", layer=layer))
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
    _compute_particles, op_key="detect.particles", label="Particle Detection",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InFloat("min_distance", "Min distance", unit="um", field=True, default=0.5,
                pick_kind="distance",
                derive="0.61*(emission_nm or 520)/(na or 1.4)/1000",
                description=
                "Closest two particles may be and still be reported separately, in microns — "
                "it doubles as the detection filter's scale, so it sets both the size of "
                "particle looked for AND the suppression radius around each hit. LARGER "
                "merges nearby particles into single detections and undercounts dense fields; "
                "SMALLER splits one particle into several and lets noise peaks through. Auto "
                "derives it from the current optics' diffraction limit (emission λ and NA), "
                "which is the closest two genuine point sources can be and still be "
                "distinguished — usually the right answer. `components` mode ignores it."),
        InFloat("threshold", "Threshold", unit="", field=True, default=0.0,
                pick_kind="level",
                description=
                "Foreground cut, measured on the image after it is divided by its own maximum "
                "— so it lives in [0,1] and is independent of bit depth, NOT in raw counts. "
                "0 does NOT mean \"no threshold\": it means AUTO, an Otsu level computed on "
                "that normalized histogram, which is the usual choice. Set it explicitly to "
                "override Otsu when the automatic level is too greedy or too timid; higher "
                "keeps fewer, brighter candidates. Because the scale is relative to the "
                "brightest voxel present, one very bright artefact can depress the effective "
                "cut for everything else."),
        InFloat("min_intensity", "Min intensity", unit="", field=True, default=0.0,
                pick_kind="level",
                description=
                "Post-filter in RAW intensity units (unlike Threshold, which is normalized): a "
                "detection whose raw value at its own position is below this is discarded. 0 "
                "disables it. This is the knob for \"only count particles brighter than X "
                "counts\" — an absolute, calibration-independent criterion that survives "
                "changes in the brightest object present, which the relative Threshold does "
                "not. Applied after detection, so it only ever REMOVES particles."),
        InInt("min_size", "Min size", unit="", field=False, default=1,
              description=
              "Smallest blob to accept, in VOXELS — area in 2D, volume in 3D. Raise it to "
              "reject single-voxel noise spikes and shot noise that survived thresholding; "
              "each step up also discards genuinely small particles, and in 3D the count grows "
              "as the cube of radius so a threshold that seems small removes a lot. 1 accepts "
              "everything. Applied before centroids are computed, so a rejected blob "
              "contributes nothing to any position."),
        InBool("subpixel", "Subpixel", field=False, default=True,
               description=
               "Refine each detection to a position BETWEEN voxels instead of snapping it to "
               "the voxel it was found in. On, positions carry fractional coordinates — "
               "essential for tracking or for measuring displacements smaller than a pixel, "
               "since integer positions quantize every motion to whole pixels and can hide "
               "real sub-pixel movement entirely. Off, positions are voxel centres. Costs a "
               "little time per particle and nothing else; leave it on unless you specifically "
               "want integer coordinates."),
        InString("name", "Output layer", field=False, default="particles",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this node writes: one row per detected particle with "
                 "its id and position (a per-plane z index in 2D, a subpixel z in 3D). Ids are "
                 "unique across the whole detection. Downstream nodes — Track Linking, Cluster "
                 "Points, Tessellate — select it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("mode", ["log", "components"], default="log", label="Detector",
                description=
                "Whether particles are found as filter PEAKS or as thresholded BLOBS. The "
                "two answer different questions — one localizes point-like objects, the other "
                "measures whatever the threshold connects — so they disagree most on touching "
                "particles and on anything larger than the filter scale. They also read "
                "different params: `log` uses Min distance as its scale, `components` ignores "
                "it and leans on Threshold and Min size.",
                choice_docs={
                    "log":
                        "Laplacian-of-Gaussian local maxima with non-maximum suppression, "
                        "refined to sub-voxel by a parabola fit. Built for point-like "
                        "particles of a known size: two touching beads stay two detections, "
                        "and Min distance sets both the filter scale and the suppression "
                        "radius. It reports POSITIONS only and says nothing about extent.",
                    "components":
                        "Threshold the image, label the connected blobs and take each one's "
                        "centroid (radial-symmetry refined in 3D). Handles particles of "
                        "mixed and unknown size, which the fixed LoG scale does not — but two "
                        "particles that touch become one component with one centroid between "
                        "them, and the result depends entirely on where Threshold lands.",
                })],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes=_DIM_KAX,
    description="Particle/small-object detection → Points (LoG local-maxima or component "
                "centroids, sub-voxel refined; beads, puncta, foci); min distance = LoG σ "
                "+ NMS radius (µm→px); 2D per-plane vs 3D volume (ported v1 kernel).")
