"""Bead Finder (``detect.beads``) — slab-projected 3-D bead centroids → Points."""

from __future__ import annotations

import numpy as np

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
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import POINT_INVARIANT, on_layer
from nodegraph.catalog._shared.units import to_pixels_v2

#: The per-bead columns this node writes beside the invariant Point schema.
BEAD_COLUMNS = ("amplitude", "snr", "sigma_xy_um", "sigma_z_um", "skew_z", "slab",
                "deblended", "stacked")

# The immersion index is not in any file's metadata, so the axial-PSF derive infers it from
# the NA: above 1.0 only oil (1.515) reaches it, 0.8–1.0 is a water dipping lens (1.33),
# below that air. The confocal axial FWHM is then 0.88 λ / (n − sqrt(n² − NA²)) — the
# pinhole ≥ 1 AU formula; NA is clamped under n so the root stays real.
_N_IMM = "(1.515 if (na or 1.4) > 1.0 else (1.33 if (na or 1.4) > 0.8 else 1.0))"
_AXIAL_FWHM_DERIVE = (f"0.88*((emission_nm or 520)/1000)/({_N_IMM} - "
                      f"sqrt({_N_IMM}**2 - min((na or 1.4), 0.98*{_N_IMM})**2))")


def _compute_beads(ctx: EvalContext) -> Dataset:
    """Fluorescent-bead centroids in a Z-stack → a Point structure, by slab projection.

    The volume is cut into Z slabs (thickness ``S``, overlap ``G``), each slab is flattened
    (max / mean; ``min`` for dark beads), beads are found in 2-D on every flattened image
    (scale-normalised LoG at the bead's expected apparent size, peaks above ``min_snr``
    robust noise sigmas), and each hit is then refined on the RAW stack: a matched-filter
    z-profile under the hit, searched near the slab that found it, fitted with a split
    Gaussian (one sigma each side of the peak — the confocal axial profile is skewed); the
    planes around that z averaged and fitted with a 2-D Gaussian for the sub-pixel (y, x)
    and the apparent sigmas, with pixels nearer to a neighbouring hit left out so touching
    beads do not widen each other. A column whose profile holds several prominent peaks
    yields one bead per peak (beads stacked in z), each fitted on its own plane; a compact
    blob too wide for one bead is offered the two-bead model and split only when both halves
    are bead-sized, at least three quarters of a diameter apart and explain the pixels
    markedly better than one bead or one elongated object. What survives is what looks like
    a bead: a blob rather than a ridge (Hessian anisotropy of the projection), a fitted
    lateral sigma inside ``size_tolerance`` of the expected one, a fitted aspect under
    ``max_aspect``, an axial sigma inside twice the tolerance, a peak not on the first or
    last plane. The same bead found in two overlapping slabs is merged within half a
    diameter (rigid spheres cannot be closer than one), brightest wins.

    ``slabs=auto`` derives ``S`` from the bead density: a whole-stack projection counts the
    beads, then ``S`` is set so at most ~10 % of beads share a projected footprint with
    another, between about two axial sigmas and half the stack, re-estimated once or twice.
    ``slabs=manual`` takes ``slab_thickness`` / ``slab_overlap`` in microns along Z.

    Resolved spec: category analysis; op ``detect.beads``; **Point** output with the
    invariant schema plus :data:`BEAD_COLUMNS`; 3-D only (``WHOLE_VOLUME``, no dim lever —
    a slab has nothing to project on one plane, so ``z < 3`` is refused); ``diameter`` µm
    and the axial PSF FWHM (``um_axial``, derived from λ, NA and an inferred immersion
    index) become expected sigmas via :func:`~nodegraph.kernels.bead_slabs.expected_sigmas_px`
    (a least-squares Gaussian on a solid sphere: ``0.272 d`` lateral, ``0.28 d`` axial, PSF
    added in quadrature); ``voxel_size_um = (z_step_um, pixel_size_um, pixel_size_um)``.
    Kernel: :func:`nodegraph.kernels.bead_slabs.find_beads` (numpy/scipy)."""
    from nodegraph.kernels.bead_slabs import expected_sigmas_px, find_beads
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Bead Finder needs an image provider on its input Dataset")
    ax = prov.axes
    if ax.z < 3:
        raise ValueError(
            f"Bead Finder needs a Z-stack of at least 3 planes, got z={ax.z}: a slab "
            f"projection has nothing to project on a single plane. For 2-D data use Spot "
            f"Detection or Particle Detection, which detect per plane.")
    modes = ctx.params.get("__modes__", {})
    px = float(ctx.calib("pixel_size_um") or 0.1)
    zs = float(ctx.calib("z_step_um") or 0.5)
    na = float(ctx.calib("objective_na") or 1.4)
    d_um = float(ctx.params.get("diameter", 1.0))
    layer = ctx.layer("name")
    kp = {
        "projection": str(modes.get("projection", "max")),
        "min_snr": float(ctx.params.get("min_snr", 5.0)),
        # the fallbacks MUST equal the SocketSpec defaults: the engine does not default-fill
        # params, so a headless caller passing `params={}` lands here
        "size_tolerance": float(ctx.params.get("size_tolerance", 0.5)),
        "max_aspect": float(ctx.params.get("max_aspect", 1.5)),
        "slab_px": 0, "overlap_px": -1,
        # rigid spheres: two centres are never closer than a diameter, and anything closer
        # than half of one is the same bead seen twice — the kernel's deblend floor and
        # merge radius, in voxels along each axis
        "diameter_px": d_um / px, "diameter_zpx": d_um / zs,
    }
    if modes.get("slabs", "auto") == "manual":
        kp["slab_px"] = max(1, int(round(to_pixels_v2(
            float(ctx.params.get("slab_thickness", 2.0)), "um_axial", z_step_um=zs))))
        kp["overlap_px"] = max(0, int(round(to_pixels_v2(
            float(ctx.params.get("slab_overlap", 0.8)), "um_axial", z_step_um=zs))))

    units = [(m, t, c) for m in range(ax.m) for t in range(ax.t) for c in range(ax.c)]
    ctx.progress(0, len(units), "finding beads", frames=ax.t)
    tables = []
    for i, (m, t, c) in enumerate(units):
        # per-channel optics (C8/H12): λ and the derived axial FWHM are THIS channel's
        ch = ctx.channel(c)
        lam_nm = ch.emission_nm(None)
        lam_nm = float(lam_nm) if lam_nm else 520.0
        fwhm_z = ch.param("axial_fwhm")
        fwhm_z_um = float(fwhm_z) if fwhm_z is not None else 0.5
        s_xy, s_z = expected_sigmas_px(d_um, pixel_size_um=px, z_step_um=zs,
                                       psf_sigma_xy_um=0.21 * lam_nm / 1000.0 / na,
                                       psf_sigma_z_um=fwhm_z_um / 2.355)
        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
        pts, cols, info = find_beads(vol, (zs, px, px),
                                     {**kp, "sigma_xy_px": s_xy, "sigma_z_px": s_z})
        tb = point_table(pts if len(pts) else np.zeros((0, 3)), m=m, t=t, c=c,
                         z_kind="subpixel", layer=layer)
        merged = dict(tb.columns)
        merged["amplitude"] = np.asarray(cols["amplitude"], dtype=float)
        merged["snr"] = np.asarray(cols["snr"], dtype=float)
        merged["sigma_xy_um"] = np.asarray(cols["sigma_xy_px"], dtype=float) * px
        merged["sigma_z_um"] = np.asarray(cols["sigma_z_px"], dtype=float) * zs
        merged["skew_z"] = np.asarray(cols["skew_z"], dtype=float)
        merged["slab"] = np.asarray(cols["slab"], dtype=np.int64)
        flags = np.asarray(cols["flags"], dtype=np.int64)
        merged["deblended"] = flags & 1           # one of a pair the LoG saw as one blob
        merged["stacked"] = (flags >> 1) & 1      # its column held more than one z peak
        tables.append(merged)
        rej = info["rejected"]
        ctx.progress(i + 1, len(units),
                     f"{len(pts)} beads · slabs {info['slab_px']} planes ×{info['n_slabs']} "
                     f"(overlap {info['overlap_px']}) · refused {sum(rej.values())}: "
                     + ", ".join(f"{k} {v}" for k, v in rej.items() if v),
                     frames=ax.t)
    cols_out = {k: np.concatenate([tb[k] for tb in tables]) for k in tables[0]}
    cols_out["id"] = np.arange(len(cols_out["id"]), dtype=np.int64)   # global-unique ids
    return ds.with_structure(StructureTable(Domain.POINT, cols_out, layer=layer,
                                            z_kind="subpixel"))


def _columns_beads(params, modes, incoming):
    """The invariant Point schema plus the per-bead fit columns. Total by contract (runs on
    every keystroke)."""
    return on_layer(Domain.POINT, str((params or {}).get("name") or "beads"),
                    POINT_INVARIANT + BEAD_COLUMNS)


register_node(
    batch_aware(_compute_beads), op_key="detect.beads", label="Bead Finder",
    category="analysis",
    adds_columns=_columns_beads,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InFloat("diameter", "Bead diameter", unit="um", field=True, default=1.0,
                pick_kind="distance",
                description=
                "The physical diameter of the beads you are looking for, in microns. It "
                "sets the scale of everything downstream: the detection filter is tuned to "
                "the apparent size of a sphere this big seen through the current optics "
                "(the PSF is added in quadrature, so a 0.2 µm bead is searched at the "
                "diffraction limit, not at 0.2 µm), and Size tolerance is a band AROUND "
                "that apparent size. Too SMALL and real beads are refused as too big while "
                "noise peaks pass; too LARGE and beads are refused as too small while "
                "aggregates and debris of that size are accepted. It is also the physical "
                "floor: beads are rigid spheres, so two reported beads are never closer "
                "than three quarters of it, and two fits closer than half of it are merged "
                "as one bead. Set it to the bead supplier's nominal diameter; the fitted "
                "`sigma_xy_um` column tells you what the data actually shows."),
        InFloat("axial_fwhm", "Axial PSF FWHM", unit="um_axial", field=True, default=0.5,
                derive=_AXIAL_FWHM_DERIVE,
                description=
                "How far a point source smears along Z in this microscope, as the full "
                "width at half maximum of the axial PSF, in microns. It only matters for "
                "the EXPECTED axial size of a bead (the sphere's own extent is added in "
                "quadrature) and so for the axial size band and the z-fit window; it does "
                "not move any position. Auto derives the confocal value "
                "0.88·λ/(n − √(n² − NA²)) from the channel's emission λ and the objective "
                "NA, inferring the immersion index from the NA (oil above 1.0, water "
                "0.8–1.0, air below) — about 0.5 µm at 1.4 NA, 4 µm at 0.45 NA. A real "
                "stack with index mismatch is broader than that; if beads are being refused "
                "as `size_z`, raise it toward what the fitted `sigma_z_um` column reports "
                "(×2.355)."),
        InFloat("min_snr", "Min SNR", unit="", field=True, default=5.0,
                description=
                "How far above the noise a bead's detection-filter response has to stand, "
                "in robust standard deviations of that slab's response, to become a "
                "candidate. LOWER finds dimmer beads and admits more noise peaks (which the "
                "size test then has to refuse, so the run gets slower before it gets "
                "wronger); HIGHER keeps only bright beads. 5 is a conservative start — on "
                "Gaussian noise it admits well under one false peak per million pixels. It "
                "is relative to each slab's own noise, so it does not change with bit depth "
                "or exposure; it is the first thing to lower if dim beads are missing and "
                "the per-bead `snr` column shows the found ones far above it."),
        InFloat("size_tolerance", "Size tolerance", unit="", field=False, default=0.5,
                description=
                "How far a fitted bead's lateral width may deviate from the EXPECTED width "
                "for Bead diameter, as a fraction: 0.5 accepts fitted sigmas between half "
                "and one-and-a-half times the expectation. The axial width is allowed twice "
                "this. This is the filter that turns \"bright spots\" into \"beads\": a hot "
                "voxel is far too narrow, an aggregate or a haze far too wide, and both are "
                "refused here. TIGHTER refuses more non-beads but also real beads when the "
                "diameter, the pixel size or the optics are slightly off; LOOSER accepts "
                "more of everything. It changes how many beads are reported, so every "
                "downstream count and track moves with it."),
        InFloat("max_aspect", "Max aspect", unit="", field=False, default=1.5,
                description=
                "The largest ratio of a fitted bead's long to short lateral width that still "
                "counts as round. A bead is a sphere, so its image is round; a fibre, a "
                "scan line, two merged beads or a bead smeared by stage motion is not, and "
                "is refused above this. 1.5 is generous for a clean stack (beads fit at "
                "~1.1); tighten toward 1.3 to refuse marginally merged pairs at the price "
                "of refusing beads at the lateral border, whose fit is one-sided. Ridges "
                "(fibres, lines) are also caught earlier by a curvature test that this "
                "value does not control."),
        InFloat("slab_thickness", "Slab thickness", unit="um_axial", field=False, default=2.0,
                available_in={"slabs": frozenset({"manual"})},
                description=
                "How thick each Z sub-stack is before it is flattened, in microns along Z. "
                "THINNER slabs keep beads at different depths from landing on top of each "
                "other in the projection — the dense-field problem — at the cost of more "
                "projections to search and a dimmer mean projection; THICKER slabs are "
                "fewer and brighter but merge beads that overlap in (x, y). A slab thinner "
                "than a bead's axial extent (about 2× Axial PSF FWHM plus the diameter) "
                "buys nothing. Only read when Slabs is `manual`; `auto` derives it from "
                "the measured bead density."),
        InFloat("slab_overlap", "Slab overlap", unit="um_axial", field=False, default=0.8,
                available_in={"slabs": frozenset({"manual"})},
                description=
                "How much consecutive slabs share, in microns along Z. A bead centred on a "
                "slab boundary is split between two projections; overlap by about a bead's "
                "axial extent and it appears whole in at least one of them (the same bead "
                "found twice is merged afterwards, brightest wins). MORE overlap costs more "
                "projections and more duplicates to merge; zero is fine for a max "
                "projection of bright beads and loses dim ones on the boundaries of a mean "
                "projection. Must be smaller than Slab thickness. Only read when Slabs is "
                "`manual`."),
        InString("name", "Output layer", field=False, default="beads",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this node writes: one row per bead with its "
                 "sub-voxel position and the fit columns `amplitude`, `snr`, `sigma_xy_um`, "
                 "`sigma_z_um`, `skew_z` (how much wider the axial profile is above the "
                 "bead than below, −1…1), `slab` (which projection found it), `deblended` "
                 "(1 when it is one of two touching beads the detector saw as a single blob "
                 "and split with the two-bead model) and `stacked` (1 when its column held "
                 "another bead above or below it) — the last two mark the positions that "
                 "rest on a model choice rather than an isolated peak, so a strict analysis "
                 "can drop them with If / Else. Ids are "
                 "unique across the whole detection. Downstream nodes — Track Linking, "
                 "Track Objects, Cluster Points, Measure — select it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("slabs", ["auto", "manual"], default="auto", label="Slabs",
             description=
             "Where the Z sub-stack thickness comes from. Both cut the stack into "
             "overlapping slabs and flatten each; they differ in who decides how thick, "
             "which is the knob that trades dense-field separation against speed.",
             choice_docs={
                 "auto":
                     "Derive the slab thickness from the data: a whole-stack projection "
                     "counts the beads, and the thickness is set so that at most about one "
                     "bead in ten shares a projected footprint with another — between about "
                     "two axial sigmas and half the stack — then re-estimated once or twice "
                     "as the count firms up. Right for an unknown sample; costs one or two "
                     "extra passes.",
                 "manual":
                     "Use Slab thickness and Slab overlap as given, in microns along Z. "
                     "One pass, reproducible across stacks of different density, and the "
                     "only way to go thinner than the automatic choice when beads stack "
                     "closely in Z or when artefacts occupy many planes.",
             }),
        Mode("projection", ["max", "mean", "min"], default="max", label="Projection",
             description=
             "How each slab is flattened before the 2-D search. The projection is only "
             "used to FIND candidates; every position and size is then refined on the raw "
             "stack, so this changes which beads are found, not where they are placed.",
             choice_docs={
                 "max":
                     "Brightest voxel through the slab. Keeps every bead at its own peak "
                     "brightness however thick the slab, so it finds dim beads; also keeps "
                     "every hot voxel and the brightest plane of every artefact, which the "
                     "size tests then have to refuse. The default for bright beads on a "
                     "dark background.",
                 "mean":
                     "Average through the slab. Suppresses single-voxel noise and makes the "
                     "noise floor the detection threshold measures itself against quieter, "
                     "but dilutes a bead by the ratio of slab thickness to bead extent — so "
                     "use it with thin slabs, or dim beads vanish.",
                 "min":
                     "Darkest voxel through the slab, for DARK beads on a bright background "
                     "(transmitted light, negative staining). The whole volume is inverted "
                     "first, so a dark bead reads as a bright one and every other control "
                     "keeps its meaning; `amplitude` is then the depth of the dip.",
             }),
    ],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Fluorescent-bead centroids in a Z-stack → Points, by slab projection: Z "
                "sub-stacks flattened (max/mean/min), 2-D LoG detection per slab, then a "
                "split-Gaussian z fit and a 2-D Gaussian (y, x) fit on the raw stack; "
                "size, roundness and axial-extent filters keep only beads. 3-D only.")
