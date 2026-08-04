"""Boundary Band (``analysis.boundary_band``) — Outward boundary band per label region…"""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InBool,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)

from nodegraph.catalog._base import register_node

# ── Boundary bands (Voxel label raster → outward band raster, 3D) ──────────────

def _compute_boundary_band(ctx: EvalContext) -> Dataset:
    """Outward boundary band per label region (general — any Voxel label raster: cells,
    nuclei, granules, …; ported v1 ``granule_boundary`` kernel): for each labelled region,
    the voxels within ``band_voxels`` (or ``band_um``) of its surface whose label is NOT
    that region. Reads a **Voxel label raster** (from CCL / watershed / a mask-volume node)
    and writes a Voxel **band-label** raster — each band voxel painted with its source
    region id. 3D per ``(m,t,c)`` volume; ``method`` = dilation (6-connectivity iterations)
    or EDT (µm threshold).

    voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um) drives the EDT sampling.
    Kernel: :func:`nodegraph.kernels.granule_boundary.extract_boundary_bands`."""
    from nodegraph.kernels.granule_boundary import extract_boundary_bands
    ds = ctx.inputs[0]
    ax = ds.axes
    src = ctx.layer("labels")
    lab_attr = ds.get(Domain.VOXEL, src)
    if lab_attr is None:
        raise ValueError(f"boundary band needs a Voxel label layer {src!r} "
                         f"(run a label / watershed / mask-volume node first)")
    labels6 = np.asarray(lab_attr.values)
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)
    kp = {
        "band_voxels": max(0, int(ctx.params.get("band_voxels", 1))),
        "band_method": ctx.params.get("__modes__", {}).get("method", "dilation"),
        "band_um": float(ctx.params.get("band_um", 0.0)),
        "include_neighbors": bool(ctx.params.get("include_neighbors", True)),
    }
    out = np.zeros_like(labels6)
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                vol = labels6[m, t, :, c]                      # (Z, H, W)
                masks = {int(g): (vol == g) for g in np.unique(vol) if g != 0}
                if not masks:
                    continue
                _bands, combined = extract_boundary_bands(masks, vol, vox, kp)
                out[m, t, :, c] = combined
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out)
register_node(
    _compute_boundary_band, op_key="analysis.boundary_band",
    label="Boundary Band", category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            InString("labels", "Label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which label raster to grow bands around — any Voxel label layer (cells, "
                     "nuclei, granules) from Connected Components, Segmentation or Rasterize "
                     "Mesh. Bands are grown per region and carry the surrounding region's id, "
                     "so this choice fixes both what the bands surround and how they are "
                     "joined back to their objects."),
            # band_voxels is live in BOTH methods — dilation iterations, and the `edt`
            # fallback threshold (N·finest-voxel) when band_um is 0. band_um is read
            # only by the EDT criterion (kernel: `edt <= edt_threshold`), so it stays
            # hidden under dilation, where it has no effect at all.
            InInt("band_voxels", "Band voxels", unit="", field=False, default=1,
                  description=
                  "Band thickness in VOXELS — how far outward from each region's surface the "
                  "band extends. Live in BOTH methods: under `dilation` it is the number of "
                  "dilation steps (and so the only thickness control), and under `edt` it is "
                  "the fallback used when Band width is 0. Because it counts voxels rather "
                  "than microns, the physical thickness changes with magnification and is "
                  "anisotropic on non-cubic voxels — prefer Band width under `edt` when the "
                  "thickness needs to mean a real distance."),
            InFloat("band_um", "Band width", unit="um", field=True, default=0.0,
                    pick_kind="radius",
                    available_in={"method": frozenset({"edt"})},
                    description=
                    "Band thickness in MICRONS — the physically meaningful thickness control, "
                    "measured as a true Euclidean distance from the region surface, so it is "
                    "correct even on anisotropic voxels. 0 falls back to Band voxels. Only the "
                    "`edt` method reads this; under `dilation` it is hidden because dilation "
                    "counts steps and cannot honour a distance."),
            InBool("include_neighbors", "Include neighbors", field=False, default=True,
                   description=
                   "What to do where two regions' bands would overlap. ON, a band may extend "
                   "into territory nearer another region, so bands around crowded objects stay "
                   "their full requested thickness but can double-count the space between "
                   "neighbours. OFF, growth stops at the midline between regions, giving "
                   "thinner but mutually exclusive bands — the right choice when the band is a "
                   "measurement region and each voxel must belong to exactly one object."),
            InString("name", "Output layer", field=False, default="bands",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the band raster this node writes. Each band carries the ID OF THE "
                     "REGION IT SURROUNDS, so it can be joined back to that object's row, and "
                     "the band excludes the region interior — it is a shell, not a dilated "
                     "object. Feed it to Measure to quantify a rim, halo, or the local "
                     "environment just outside each object.")],
    outputs=[OutDataset()],
    modes=[Mode("method", ["dilation", "edt"], default="dilation", label="Band method",
                description=
                "How thickness is measured outward from each region's surface — and therefore "
                "which of the two width sockets is used. The difference matters on an "
                "ANISOTROPIC stack, where a step in z is a different physical distance than a "
                "step in y or x.",
                choice_docs={
                    "dilation":
                        "Grow the region by a whole number of 6-connected voxel steps (Band "
                        "voxels). Fast and exactly reproducible, but it counts STEPS, not "
                        "distance: on a stack whose z spacing differs from the pixel size, "
                        "one step in z covers more µm than one step in x, so the band is "
                        "thicker along z than it looks.",
                    "edt":
                        "Threshold a Euclidean distance transform sampled with the real voxel "
                        "size, so the band is a true physical thickness in µm (Band µm) and is "
                        "isotropic on anisotropic data. The correct choice whenever the width "
                        "is a measurement rather than a nudge; it costs a distance transform "
                        "per volume.",
                })],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Outward boundary band per label region (any Voxel label raster — cells, "
                "nuclei, granules; dilation or EDT, µm-aware) → band-label raster, 3D "
                "(ported v1 kernel).")
