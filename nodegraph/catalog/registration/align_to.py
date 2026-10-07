"""Align To (``registration.align_to``) — Measure how far this input's stage log sits from a reference's, per field, by phase-correlating their physical overlap — and RECORD the correction rather than applying it to pixels."""

from __future__ import annotations

import numpy as np

from typing import List

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, OutDataset

from nodegraph.catalog._base import register_node

# ── Align To (registration — correct a stage log against a reference) ──────────

def _align_sample(md, axes, m, box, n_y, n_x, read_plane):
    """One field resampled onto a µm-increasing grid over ``box`` → ``(n_y, n_x)`` float.

    Deliberately sampled in µm order with **no flip**: handedness is a property of how the
    camera is mounted, it is identical for both inputs from the same scope, and the answer
    wanted here is a displacement in microns. Working in µm space means the measurement
    never has to know which way the camera is on, and the number it produces is directly
    the correction :data:`nodegraph.placement.ALIGN_KEY` carries."""
    from nodegraph.placement import axis_map, field_box
    fb = field_box(md, axes, m)
    plane = read_plane(m)
    if fb is None or plane is None:
        return None
    rows = axis_map(n_y, box[0], box[1], plane.shape[0], fb.y0, fb.y1, flip=False)
    cols = axis_map(n_x, box[2], box[3], plane.shape[1], fb.x0, fb.x1, flip=False)
    if np.any(rows < 0) or np.any(cols < 0):
        return None                       # the window is not fully inside this field
    return plane[np.ix_(rows, cols)].astype(float)
def _compute_align_to(ctx: EvalContext) -> Dataset:
    """Measure how far this Dataset's stage log is wrong against a ``reference``, per field.

    Resolved spec (§0 grill, 2026-07-31)
    ------------------------------------
    * **Kind** registration → ``op_key="registration.align_to"``.
    * **Data contract** pixels and axes pass through **untouched**; the node writes only the
      measured correction. So it does NOT stamp sampling provenance (``_sampled``): the index
      grid is unchanged, voxel *(m,z,y,x)* still holds the pixel it held, and an overlay
      downstream must keep working — which it would not if this looked like a resample.
    * **2D/3D** no lever: registration runs on ONE reference plane per field and the
      correction is lateral, exactly as ``align.drift`` and ``util.stitch`` do it.
    * **Footprint** ``MULTI_VIEW`` / ``{m, y, x}`` — it reads several of the reference's
      multipoints to find the ones overlapping each of ours.

    **Why a stamp and not a rewritten ``origin_um``.** The correction is measured FROM
    PIXELS, and a ``meta_transform`` touches none — so an origin this node rewrote could
    never be predicted at edit time, and the envelope would disagree with the payload for
    every downstream ``ctx.calib("origin_um")``. Riding as separate non-calibration
    provenance keeps the calibration prediction exact and still reaches every consumer,
    because :func:`nodegraph.placement.field_box` adds it. This is also literally what the
    engine's registration rule asks for (V2.03 §2): registration STORES its transform.

    A field whose correlation is not believed keeps a zero correction — the stage log
    stands, which is exactly what not running this node would have done. That is the same
    accept-or-drop discipline as ``util.stitch``'s refine, and for the same reason: a
    confident wrong shift is worse than an uncorrected one.
    """
    from skimage.registration import phase_cross_correlation
    from nodegraph.kernels.registration import _hann2d, _highpass, _ncc
    from nodegraph.placement import ALIGN_KEY, field_box, overlap_fraction

    ds = ctx.inputs[0]
    ref = ctx.input("reference")
    if ref is None:
        raise ValueError(
            "Align To needs a `reference` Dataset — the one whose stage coordinates you "
            "trust. This node measures how far THIS input's field is from it and records "
            "the correction; it never moves pixels.")
    prov, rprov = ds.image, getattr(ref, "image", None)
    if prov is None or rprov is None:
        raise ValueError("Align To needs an image on both inputs.")

    max_shift = float(ctx.params.get("max_shift_um", 10.0))
    min_ncc = float(ctx.params.get("min_ncc", 0.3))
    hp_um = float(ctx.params.get("highpass_um", 5.0))
    ref_t = int(ctx.params.get("ref_t", 0))
    ref_c = int(ctx.params.get("ref_c", 0))
    ps = float(ctx.calib("pixel_size_um") or 1.0)
    rps = float(ref.metadata.get("pixel_size_um") or ps)
    grid_ps = max(ps, rps)                 # never invent detail the coarser input lacks

    ax, rax = prov.axes, rprov.axes
    mid_z = lambda a: int(a.z) // 2
    read_mine = lambda m: prov.get_region(0, m, min(ref_t, ax.t - 1), mid_z(ax),
                                          min(ref_c, ax.c - 1), 0, ax.y, 0, ax.x)
    read_ref = lambda m: rprov.get_region(0, m, min(ref_t, rax.t - 1), mid_z(rax),
                                          min(ref_c, rax.c - 1), 0, rax.y, 0, rax.x)

    shifts: List[List[float]] = []
    nccs: List[float] = []
    for m in range(ax.m):
        dy = dx = 0.0
        ncc_val = 0.0
        mine = field_box(ds.metadata, ax, m)
        if mine is not None:
            best, best_frac = None, 0.0
            for j in range(rax.m):
                rb = field_box(ref.metadata, rax, j)
                if rb is None:
                    continue
                frac = overlap_fraction(mine, rb)
                if frac > best_frac:
                    best, best_frac = (j, rb), frac
            # Correlating a sliver measures noise, so a field that does not sit mostly
            # inside ONE reference field is left uncorrected rather than guessed at.
            if best is not None and best_frac >= 0.5:
                j, rb = best
                box = (max(mine.y0, rb.y0), min(mine.y1, rb.y1),
                       max(mine.x0, rb.x0), min(mine.x1, rb.x1))
                n_y = int(max(16, min(512, (box[1] - box[0]) / grid_ps)))
                n_x = int(max(16, min(512, (box[3] - box[2]) / grid_ps)))
                a = _align_sample(ds.metadata, ax, m, box, n_y, n_x, read_mine)
                b = _align_sample(ref.metadata, rax, j, box, n_y, n_x, read_ref)
                if a is not None and b is not None:
                    um_per_sample_y = (box[1] - box[0]) / n_y
                    um_per_sample_x = (box[3] - box[2]) / n_x
                    sig = max(1.0, hp_um / max(grid_ps, 1e-9))
                    win = _hann2d(a.shape)
                    fa, fb_ = _highpass(a, sig) * win, _highpass(b, sig) * win
                    # Plain cross-correlation of the band-passed samples (2026-10-07):
                    # skimage's default phase whitening cancels any filter applied to
                    # both inputs, which had made `highpass_um` a dead control here, and
                    # it is 3-15x less precise on noisy microscopy fields (see
                    # nodegraph.kernels.registration.estimate_translation).
                    sh = phase_cross_correlation(fb_, fa, upsample_factor=10,
                                                 normalization=None)[0]
                    cand_y = float(sh[0]) * um_per_sample_y
                    cand_x = float(sh[1]) * um_per_sample_x
                    from scipy.ndimage import shift as ndi_shift
                    moved = ndi_shift(fa, shift=(float(sh[0]), float(sh[1])), order=1,
                                      mode="constant")
                    ncc_val = float(_ncc(moved, fb_))
                    if (abs(cand_y) <= max_shift and abs(cand_x) <= max_shift
                            and ncc_val >= min_ncc):
                        dy, dx = cand_y, cand_x
                    else:
                        ncc_val = 0.0      # rejected: the stage log stands
        shifts.append([dy, dx])
        nccs.append(ncc_val)
        ctx.progress(m + 1, ax.m, "aligning fields")

    return ds.with_metadata(**{ALIGN_KEY: shifts, "align_to_ncc": nccs})
register_node(
    _compute_align_to, op_key="registration.align_to", label="Align To",
    category="registration",
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference"),
        InFloat("max_shift_um", "Max shift", unit="um", field=False, default=10.0,
                description=
                "The largest correction that will be believed, in microns of stage travel. "
                "An encoded stage repeats to roughly 10 µm, so a measurement much larger "
                "than this is the correlation locking onto the wrong feature rather than a "
                "real error. RAISE it if your two acquisitions genuinely started from "
                "different stage origins; LOWER it to reject more aggressively. A rejected "
                "field keeps its stage position, which is what you would have had without "
                "this node."),
        InFloat("min_ncc", "Min correlation", unit="", field=False, default=0.3,
                description=
                "How well the two fields must agree, AFTER applying the measured shift, for "
                "it to be accepted — 0 is no agreement, 1 identical. LOWER accepts more "
                "fields including ones matched on noise; HIGHER keeps only confident "
                "matches and leaves the rest on the stage log, which is the safe direction "
                "because the stage log is already close. 0.3 passes textured specimen and "
                "rejects empty background, the same value Stitch's refine uses."),
        InFloat("highpass_um", "High-pass", unit="um", field=False, default=5.0,
                description=
                "Scale of the smooth background removed from both fields before "
                "correlating. Uneven illumination is the same shape in every field, so "
                "without this the correlation can lock onto the vignette instead of the "
                "specimen and confidently report zero shift. Set it well BELOW your "
                "features and ABOVE the illumination falloff. Affects only the measured "
                "correction, never a pixel that is stored."),
        InInt("ref_t", "Reference frame", unit="", field=False, default=0,
              description=
              "Which timepoint the alignment is measured on. Measured ONCE and applied to "
              "every frame, like Drift Correction and Stitch — the two files do not move "
              "relative to each other between frames, so one measurement keeps them "
              "consistent instead of letting the overlay jitter frame to frame. Pick a "
              "frame where both inputs have real structure."),
        InInt("ref_c", "Reference channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the correlation runs on, in BOTH inputs. Choose one whose "
              "appearance is stable and structural in each — a fiducial or a "
              "morphology marker — rather than a reporter whose brightness changes for "
              "biological reasons. Only affects the measurement."),
    ],
    outputs=[OutDataset()],
    granularity=Granularity.MULTI_VIEW, kernel_axes=frozenset({"m", "y", "x"}),
    description="Measure how far this input's stage log sits from a reference's, per field, "
                "by phase-correlating their physical overlap — and RECORD the correction "
                "rather than applying it to pixels. Overlay and every other placement "
                "consumer honour it automatically. A field whose match is not believed "
                "keeps its stage position.")
