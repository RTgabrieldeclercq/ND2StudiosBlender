"""PIV (OpenPIV) (``analysis.piv``) — 2D particle image velocimetry (multipass FFT cross-correlation with window deformation) → a Point displacement field per frame pair (window centers, disp µm, signal-to-noise);."""

from __future__ import annotations

import numpy as np

from typing import Any, List, Tuple

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
from nodegraph.structure import StructureTable, point_table

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dvc import _dvc_rows
from nodegraph.catalog._shared.objects import _frame_interval_s
from nodegraph.catalog._shared.progress import _UnitBar

# ── PIV (OpenPIV) — the FLOW sibling of DIC/DVC ─────────────────────────────────
#
# Wires the `piv_field` kernel (nodegraph/kernels/piv_field.py), a thin multipass driver
# around the third-party **openpiv** package (GPLv3), which it imports LAZILY — the node is
# fully wired but its compute raises a friendly ImportError only when actually RUN, until
# `pip install openpiv` (upstream: github.com/OpenPIV/openpiv-python). Where
# `analysis.dic_correlate` solves solid DEFORMATION against an undeformed reference with a
# global FE regularizer, PIV cross-correlates INDEPENDENT interrogation windows per frame
# pair — orders of magnitude faster, per-vector signal-to-noise, the standard tool for flow:
# microfluidics, cytoplasmic streaming, collective cell migration, bead fields (TFM). The
# V3.00 roadmap (W5-P2) notes the two should be cross-validated on the same pair.


class _PIVPlaneSeq:
    """Lazily-indexed ``(n,)`` view of the planes one PIV series consumes.

    ``run_piv_series`` indexes ``images`` on demand and holds at most two planes at a
    time, so handing it this instead of a materialized list keeps peak memory flat no
    matter how long T is. (A private twin of ``dic_correlate._DICPlaneSeq`` — kept
    per-module because catalog nodes may not import from each other's modules.)"""

    __slots__ = ("_plan",)

    def __init__(self, plan: List[Tuple[Any, int, int, int, int]]) -> None:
        #: one (provider, m, t, z, c) per entry, in solve order
        self._plan = plan

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, i: int) -> np.ndarray:
        prov, m, t, z, c = self._plan[i]
        pax = prov.axes
        return prov.get_region(0, m, t, z, c, 0, pax.y, 0, pax.x)


def _compute_piv(ctx: EvalContext) -> Dataset:
    """2D Particle Image Velocimetry (OpenPIV: multipass FFT cross-correlation with
    symmetric window deformation, Westerweel–Scarano universal outlier detection, local-mean
    vector replacement, optional Garcia-2010 smoothn predictor smoothing) → a **Point**
    displacement field per frame pair — the FLOW sibling of ``analysis.dic_correlate``.

    Resolved design (§0 grill, 2026-08-10): analysis → Point; consumes one reference
    channel of the input series plus an optional external ``reference`` Dataset (a separate
    rest-state acquisition, e.g. traction-force microscopy's relaxed-gel bead image —
    always a FIXED reference); loops m/z and hands each (m, z) line the ordered T stack in
    ONE :func:`nodegraph.kernels.piv_field.run_piv_series` call (frames stream lazily
    through :class:`_PIVPlaneSeq`, ~two planes of memory). 2D only — PIV has no volumetric
    form here; a z-stack is analysed per plane (use ``analysis.dvc_field`` for true 3D).
    Footprint WHOLE_SERIES / {t, y, x}: ``previous_frame`` pairing reads t-1 and t. No
    axis or calibration change → no meta_transform. Backend re-verified in-env against
    openpiv 0.25.4 (signatures read from installed source; sign conventions established
    empirically — raw OpenPIV u/v are +x-right/+y-DOWN, i.e. this repo's (dy, dx) with no
    swap; bit-identical parity with ``windef.simple_multipass`` in the kernel bench). The
    hot path is FFT/vectorized numpy inside openpiv → no numba. Window/pass params are in
    px deliberately: interrogation windows are texture-statistics units (features per
    window), not physical lengths, so a µm derive would mislead more than help.

    ``reference_mode`` defaults to ``previous_frame`` (flow: one field per frame step).
    ``fixed_frame`` measures total displacement since a chosen frame, like DIC.
    ``ensemble`` (V3 W5-P2 step 1; Meinhart–Wereley–Santiago 2000) averages the
    CORRELATION PLANES of every consecutive pair before peak-finding and emits ONE
    time-averaged field per (m, z) line, stamped ``t = 0`` — the micro-PIV method for
    seeding too sparse for any single pair (kernel bench: 2.63 → 0.41 px RMSE at ~3
    particles/window over 16 pairs); it assumes the flow is statistically steady, and
    with an external reference wired it averages (ref → t) pairs instead (static
    deformation, e.g. a TFM bead series). ``velocity`` (step 3) divides the µm
    displacement by the frame interval — the ``frame_interval`` socket if set, else the
    file's ``dt_s`` via the shared :func:`_frame_interval_s` resolver, which REFUSES
    when neither exists rather than mislabel µm/frame as µm/s — and emits ``vy``/``vx``/
    ``speed`` columns (µm/s, names matching ``analysis.object_metrics``). Velocity is
    refused for ``fixed_frame`` or an external reference (cumulative displacement has no
    per-row rate). ``uncertainty`` (step 5) emits per-vector ``unc_y``/``unc_x`` (µm) via
    the Sciacchitano-2013 image-matching disparity method — the external ``pivuq``
    package, lazily imported through :func:`disparity_uncertainty`, whose docstring pins
    the probed conventions; refused in ``ensemble`` mode (a per-pair method). Per pair
    the kernel runs a coarse-to-fine window ladder (each pass 2×
    the next, ending at ``window_size``) with symmetric image deformation between passes;
    each pass validates (global limit / global std / median test / signal-to-noise) and
    inpaints failures — mandatory between passes (holes poison the predictor), governed
    by ``replace`` on the final field. Vectors that end non-finite (outside the ROI, or
    failed with replacement off/impossible) are DROPPED, not emitted as NaN rows; the
    count lands in provenance metadata as ``piv_invalid_points``.

    Output: a Point layer, one row per interrogation-window centre — ``disp_y``/``disp_x``/
    ``disp_mag_um`` in µm (grid px × ``pixel_size_um``), ``qfactor`` = the final pass's
    correlation signal-to-noise (the per-vector confidence DIC cannot provide), and
    ``replaced`` (1 = failed validation, value inpainted from neighbours). The grid is
    COARSER than the image (pitch = window × (1 − overlap)) — feed it to
    ``transform.rasterize_field`` for a dense map. Reuses the shared :func:`_dvc_rows`
    flattener, so columns align with the DVC/DIC siblings."""
    from nodegraph.kernels.piv_field import (
        disparity_uncertainty, run_piv_ensemble, run_piv_series)

    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("PIV needs an image provider on its input Dataset")
    ax = prov.axes
    ref_ds = ctx.input("reference")
    ref_prov = ref_ds.image if ref_ds is not None else None
    if ref_ds is not None and ref_prov is None:
        raise ValueError("the PIV 'reference' input Dataset has no image provider")
    modes = ctx.params.get("__modes__", {})
    ref_mode = modes.get("reference_mode", "previous_frame")
    corr = modes.get("correlation", "circular")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    vox: Tuple[float, float] = (px, px)                  # (y, x) µm/px — kernel contract §3
    window = max(8, int(ctx.params.get("window_size", 32)))
    passes = min(4, max(1, int(ctx.params.get("passes", 3))))
    params = {
        # coarse→fine ladder ending at window_size; the kernel drops passes that do not
        # fit the image (diagnostics report the effective ladder)
        "windowsizes": tuple(window * (2 ** (passes - 1 - i)) for i in range(passes)),
        "overlap": min(0.75, max(0.0, float(ctx.params.get("overlap", 0.5)))),
        "correlation_method": corr,
        "subpixel_method": ctx.params.get("subpixel", "gaussian") or "gaussian",
        "max_disp_px": max(0.0, float(ctx.params.get("max_disp", 0.0))),
        "sig2noise_threshold": float(ctx.params.get("s2n_threshold", 1.0)),
        "median_test": ctx.params.get("median_test", "universal") or "universal",
        "median_threshold": max(0.0, float(ctx.params.get("median_threshold", 2.0))),
        "std_threshold": max(0.0, float(ctx.params.get("std_threshold", 10.0))),
        "replace_vectors": bool(ctx.params.get("replace", True)),
        "smoothn": bool(ctx.params.get("smooth", False)),
        "smoothn_p": max(0.0, float(ctx.params.get("smooth_strength", 0.05))),
    }
    # velocity: only per-step pairings have a per-row rate. The shared resolver reads the
    # `frame_interval` socket override, else the file's dt_s (memo-fenced), else REFUSES —
    # never a silent 1.0 that would mislabel µm/frame as µm/s.
    want_vel = bool(ctx.params.get("velocity", False))
    if want_vel and (ref_mode == "fixed_frame" or ref_prov is not None):
        raise ValueError(
            "PIV velocity needs per-step pairing: fixed_frame (or an external reference) "
            "reports CUMULATIVE displacement since the reference, so µm/s is not defined "
            "per row — use previous_frame or ensemble, or divide the disp columns "
            "downstream yourself")
    # uncertainty: Sciacchitano-2013 image-matching disparity (external pivuq, lazy).
    # A per-PAIR method: both frames of one pair are warped by that pair's field and the
    # residual particle disparities are pooled — undefined for the ensemble AVERAGE.
    want_unc = bool(ctx.params.get("uncertainty", False))
    if want_unc and ref_mode == "ensemble":
        raise ValueError(
            "PIV uncertainty is a per-pair method (both frames of a pair are warped by "
            "that pair's own field) — it is not defined for the ensemble-averaged field. "
            "Use previous_frame/fixed_frame, or turn Uncertainty off for ensemble runs")
    try:
        dt_s = _frame_interval_s(ctx, needed=want_vel, wanted=("velocity",),
                                 node="analysis.piv")
    except ValueError:
        raise ValueError(
            "analysis.piv: velocity columns (vy/vx/speed) are µm/s RATES, and this "
            "Dataset declares no frame interval — set the `frame_interval` socket to the "
            "seconds between timepoints, or re-open the source if it should carry timing "
            "(an ND2's per-frame timestamps become `dt_s` at ingest; a plain TIFF has "
            "none). The displacement columns need no interval — turn Velocity off to "
            "proceed without one.") from None
    layer = ctx.layer("name")
    # optional ROI: a Voxel mask layer restricts where vectors are computed; the mask is
    # STATIC per (m, z) line — a time-varying layer contributes its t=0 plane
    roi_name = ctx.layer("roi")
    roi_attr = ds.get(Domain.VOXEL, roi_name) if roi_name else None
    roi6 = roi_attr.values if roi_attr is not None else None

    def _roi_plane(m: int, z: int):
        if roi6 is None:
            return None
        s = roi6.shape
        return roi6[min(m, s[0] - 1), 0, min(z, s[2] - 1), min(c, s[3] - 1)] > 0

    def _plan(m: int, z: int) -> Tuple[List[Tuple[Any, int, int, int, int]], List[int], str]:
        """(plane plan, the t each RESULT belongs to, kernel pairing).

        previous_frame — and ensemble without an external reference — feed the T stack
        in order (consecutive pairs; ensemble averages their correlation planes).
        fixed_frame (or an external reference, always fixed — ensemble included)
        prepends the reference plane so every t correlates against it — including
        t == the reference, which self-pairs to ~0 exactly as the DIC sibling does."""
        if ref_mode in ("previous_frame", "ensemble") and ref_prov is None:
            return ([(prov, m, t, z, c) for t in range(ax.t)],
                    list(range(1, ax.t)), "previous")
        if ref_prov is not None:                          # external reference (fixed)
            rax = ref_prov.axes
            head = (ref_prov, min(m, rax.m - 1), min(max(0, ref_frame), rax.t - 1),
                    min(z, max(0, rax.z - 1)), min(c, max(0, rax.c - 1)))
        else:
            head = (prov, m, min(max(0, ref_frame), ax.t - 1), z, c)
        return ([head] + [(prov, m, t, z, c) for t in range(ax.t)],
                list(range(ax.t)), "fixed_head")

    rows: list = []
    n_dropped = 0

    def _emit(res, m: int, t: int, z: int, unc=None) -> None:
        """One PIVResult → filtered Point rows (+ velocity/uncertainty columns)."""
        nonlocal n_dropped
        row = _dvc_rows(res, vox, m=m, t=t, c=c, z_plane=z)
        row["replaced"] = res.flags.reshape(-1).astype(np.int64)
        if want_vel:                                      # µm / s — one frame step per pair
            row["vy"] = row["disp_y"] / dt_s
            row["vx"] = row["disp_x"] / dt_s
            row["speed"] = row["disp_mag_um"] / dt_s
        if unc is not None:                               # px → µm, [y, x] components
            row["unc_y"] = unc[..., 0].reshape(-1) * vox[0]
            row["unc_x"] = unc[..., 1].reshape(-1) * vox[1]
        keep = np.isfinite(row["disp_y"]) & np.isfinite(row["disp_x"])
        if not keep.all():
            n_dropped += int((~keep).sum())
            row = {k: v[keep] for k, v in row.items()}
        if row["m"].size:
            rows.append(row)

    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=1, note="correlating")
    for m in range(ax.m):
        for z in range(ax.z):
            plan, result_t, pairing = _plan(m, z)
            if len(plan) < 2:                             # a 1-frame series has no pair
                bar.finish_unit(note=f"m={m} z={z}")
                continue
            if ref_mode == "ensemble":
                # ONE time-averaged field per (m, z), stamped t=0
                res = run_piv_ensemble(
                    _PIVPlaneSeq(plan), params, vox, pairing=pairing,
                    roi_mask=_roi_plane(m, z), progress_cb=bar.emit)
                _emit(res, m, 0, z)
            else:
                seq = _PIVPlaneSeq(plan)
                series = run_piv_series(
                    seq, params, vox, pairing=pairing,
                    roi_mask=_roi_plane(m, z), progress_cb=bar.emit)
                for i, t in enumerate(result_t):
                    if i >= len(series):
                        break
                    unc = None
                    if want_unc:                          # re-fetch this result's pair
                        unc = disparity_uncertainty(
                            seq[0 if pairing == "fixed_head" else i], seq[i + 1],
                            series[i], window_size=window)
                    _emit(series[i], m, t, z, unc)
            bar.finish_unit(note=f"m={m} z={z}")

    # Provenance (§7b): PIV-specific keys (NOT the dvc_* keys accumulate_field consumes —
    # a PIV per-step field is Eulerian flow, not an ALDVC volume increment).
    prov_md = {"piv_reference_mode": (ref_mode if ref_prov is None or
                                      ref_mode == "ensemble" else "fixed_frame"),
               "piv_reference_frame": ref_frame, "piv_correlation": corr,
               "piv_invalid_points": n_dropped}
    if want_vel:
        prov_md["piv_dt_s"] = float(dt_s)
    if not rows:
        empty = point_table(np.zeros((0, 2)), z_kind="plane_index", layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(
        StructureTable(Domain.POINT, merged, layer=layer, z_kind="plane_index"))
        .with_metadata(**prov_md))


register_node(
    _compute_piv, op_key="analysis.piv", label="PIV (OpenPIV)",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference",
                  description=
                  "Optional separate rest-state acquisition to measure against — e.g. "
                  "traction-force microscopy's relaxed-gel bead image taken after cell "
                  "release. When wired it is ALWAYS treated as a fixed reference "
                  "(`Reference frame` picks its timepoint) and the reference-mode lever "
                  "is ignored."),
        InInt("channel", "Channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the correlation runs on (0-based). Displacement comes from "
              "image texture, so choose the channel with stable, high-contrast tracers — "
              "beads, particles or persistent cell texture rather than a reporter whose "
              "brightness changes biologically. A channel whose appearance changes between "
              "frames breaks the constant-pattern assumption and yields confident but "
              "wrong vectors."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame",
              description=
              "Which timepoint is the reference (0-based). Read when the reference mode is "
              "`fixed_frame`, or when an external Reference Dataset is wired (it then picks "
              "that Dataset's frame). The default `previous_frame` mode ignores it and "
              "measures per-step motion instead."),
        InInt("window_size", "Window size", unit="px", field=False, default=32,
              description=
              "Edge length, in pixels, of the FINAL interrogation window — the texture "
              "patch matched between the two frames, and THE resolution/reliability "
              "trade-off. SMALLER resolves finer flow structure but each window holds less "
              "texture, so vectors get noisier and validation flags more of them; LARGER "
              "averages real velocity gradients away. Keep several tracers per window; the "
              "classic guidance is that one pass recovers displacements up to ~1/4 of its "
              "window, and the coarser earlier passes extend that reach."),
        InInt("passes", "Passes", unit="", field=False, default=3,
              description=
              "Coarse-to-fine passes; each earlier pass uses a window twice the next (3 "
              "passes at 32 px = 128 → 64 → 32). MORE passes reach larger displacements "
              "and sharpen gradients via window deformation between passes, at roughly "
              "linear cost; 1 = a single plain correlation pass. Passes whose windows do "
              "not fit the image, or whose vector grid is too small to interpolate "
              "(under 4 rows/columns), are dropped automatically — a small crop may "
              "quietly run single-pass."),
        InFloat("overlap", "Overlap", unit="", field=False, default=0.5,
                description=
                "Window overlap as a fraction of the window (0.5 = half-overlapping). Sets "
                "grid DENSITY, not per-vector accuracy: HIGHER gives more measurement "
                "points and smoother-looking fields at superlinear cost without adding "
                "independent information; LOWER is faster and sparser. 0.5 is the field "
                "standard; capped at 0.75."),
        InFloat("max_disp", "Max displacement", unit="px", field=False, default=0.0,
                description=
                "Hard sanity limit on either displacement component, in pixels; vectors "
                "beyond it are flagged. 0 = auto (half the coarsest window — nothing "
                "larger can be a real match). Set it manually when you know the physical "
                "maximum (flow speed × frame interval / pixel size) to catch "
                "confident-looking junk the statistical tests miss."),
        InString("subpixel", "Subpixel fit", field=False, default="gaussian",
                 choices=("gaussian", "parabolic", "centroid"),
                 description=
                 "How the correlation peak is located between integer pixels — the "
                 "difference between ~0.1 px accuracy and pixel-quantized output. Affects "
                 "peak-locking (histograms of displacement clumping at integers), which "
                 "biases every downstream statistic.",
                 choice_docs={
                     "gaussian":
                         "Three-point Gaussian fit through the peak and its neighbours — "
                         "the PIV standard, matching the roughly Gaussian shape of a "
                         "particle-image correlation peak; lowest peak-locking on "
                         "well-resolved tracers. OpenPIV falls back to centroid on its own "
                         "when correlation values around the peak go negative.",
                     "parabolic":
                         "Parabola through the same three points — marginally cheaper, "
                         "slightly more peak-locking on particle-like peaks; occasionally "
                         "steadier when the peak is broad and non-Gaussian (large smooth "
                         "structures rather than discrete particles).",
                     "centroid":
                         "Centre of mass of the 3×3 neighbourhood — defined even when "
                         "values around the peak are negative (the Gaussian's own "
                         "fallback), but with the strongest peak-locking of the three. "
                         "Choose deliberately only when the fits misbehave.",
                 }),
        InFloat("s2n_threshold", "S/N threshold", unit="", field=False, default=1.0,
                description=
                "Minimum signal-to-noise of the correlation peak (first peak against the "
                "correlation-plane mean) for a vector to count as valid. 1.0 accepts "
                "almost everything (OpenPIV's default); 1.1–1.5 starts rejecting windows "
                "with weak or ambiguous texture — RAISE it on noisy fluorescence rather "
                "than smoothing the field afterwards. The same quantity is emitted "
                "per-vector as the `qfactor` column, so you can also filter downstream."),
        InString("median_test", "Median test", field=False, default="universal",
                 choices=("universal", "classic", "off"),
                 description=
                 "The neighbourhood-consistency check — PIV's workhorse outlier detector. "
                 "It compares each vector against the median of its 3×3 neighbours and "
                 "flags the ones that disagree; the flavours differ in what counts as "
                 "disagreement.",
                 choice_docs={
                     "universal":
                         "Westerweel–Scarano universal outlier detection: the deviation "
                         "from the neighbourhood median is normalized by the local "
                         "residual level (ε = 0.2 px), so ONE threshold (~2) works across "
                         "flows, magnifications and velocity ranges. The field standard "
                         "and this node's default.",
                     "classic":
                         "Raw median test: flag when |vector − local median| exceeds the "
                         "threshold in PIXELS. Simple to reason about, but the right "
                         "threshold depends on the flow's own gradients — it under- or "
                         "over-flags exactly where the universal test would not.",
                     "off":
                         "No spatial-consistency check; only the global limit, global std "
                         "and signal-to-noise tests run. For fields expected to be "
                         "genuinely discontinuous (shear bands, interfaces) where the "
                         "median test would flag real physics.",
                 }),
        InFloat("median_threshold", "Median threshold", unit="", field=False, default=2.0,
                description=
                "Threshold for the median test. In `universal` mode it is the normalized "
                "residual r* — the literature default is 2, LARGER tolerates more local "
                "deviation before flagging; in `classic` mode it is an absolute deviation "
                "in PIXELS. Inert when the median test is off."),
        InFloat("std_threshold", "Std threshold", unit="", field=False, default=10.0,
                description=
                "Global filter: flag vectors more than this many standard deviations from "
                "the field mean. The default 10 catches only catastrophic outliers; ~5 is "
                "a reasonable tightening for smooth flows. Beware genuinely bimodal fields "
                "(a jet through still fluid) where a tight value flags the minority "
                "population wholesale."),
        InBool("uncertainty", "Uncertainty", field=False, default=False,
               description=
               "Also emit `unc_y`/`unc_x` columns (µm): the standard uncertainty of each "
               "vector by IMAGE MATCHING (Sciacchitano 2013) — both frames are warped "
               "onto each other by the measured field and the residual per-particle "
               "misalignments pooled per window. A well-measured vector reads at the "
               "actual error scale (~0.03 px-equivalent on the bench); a wrong one reads "
               "its own error. NaN where fewer than 2 particles matched — no honest "
               "estimate exists there, which on sparse or featureless data may be most "
               "rows. Refused in `ensemble` mode (a per-pair method). Costs roughly one "
               "extra pass per pair *(dep: pivuq — install with pip install pivuq "
               "--no-deps)*."),
        InBool("velocity", "Velocity", field=False, default=False,
               description=
               "Also emit `vy`, `vx` and `speed` columns in µm/s — the µm displacement "
               "divided by the frame interval (the socket below if set, else the file's "
               "own `dt_s`; with NEITHER available the node refuses rather than mislabel "
               "µm/frame as µm/s). Column names match Object Metrics, so downstream "
               "tables compare directly. Only defined for per-step pairing: refused under "
               "`fixed_frame` or an external Reference, whose displacement is cumulative."),
        InFloat("frame_interval", "Frame interval", unit="s", field=False, default=0.0,
                description=
                "Seconds between consecutive frames, used by Velocity. 0 = read the "
                "file's own `dt_s` (ND2 timestamps supply it); set it only to OVERRIDE "
                "the file — e.g. a series exported without timing metadata. Only read "
                "when Velocity is on; a wrong value scales every velocity linearly."),
        InBool("replace", "Replace outliers", field=False, default=True,
               description=
               "Fill vectors that failed validation from their valid neighbours "
               "(iterative local mean) instead of dropping them; filled rows carry "
               "`replaced` = 1 so downstream can discount them. OFF drops failed vectors "
               "from the output entirely — honest gaps instead of interpolated guesses. "
               "Between passes replacement always happens regardless: holes would poison "
               "the next pass's deformation predictor."),
        InBool("smooth", "Smooth", field=False, default=False,
               description=
               "Smooth the field between passes with the Garcia-2010 DCT penalized-least-"
               "squares smoother (`smoothn`) to stabilize the deformation predictor on "
               "noisy data. The FINAL field stays raw — except in a single-pass run, where "
               "this smooths the output itself (mirroring OpenPIV). Off by default: it "
               "trades small-scale structure for noise suppression."),
        InFloat("smooth_strength", "Smooth strength", unit="", field=False, default=0.05,
                description=
                "The smoother's s parameter — LARGER flattens the predictor field harder. "
                "Only read when Smooth is on. OpenPIV's default 0.05 is mild; values "
                "around 1 visibly erase real gradients."),
        InString("roi", "ROI mask layer", field=False, default="roi_mask",
                 layer_in=Domain.VOXEL,
                 description=
                 "Optional Voxel mask restricting WHERE vectors are computed — from ROI "
                 "Mask or a Threshold. Windows centred outside it are skipped: both a "
                 "speed win and a quality one, since featureless background windows return "
                 "meaningless vectors that pollute validation statistics and any smoothing "
                 "of neighbouring points. The mask is static per (m, z) line — a "
                 "time-varying layer contributes its t=0 plane. Naming a layer that is not "
                 "present leaves the whole frame in play."),
        InString("name", "Output layer", field=False, default="piv",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point field this node writes: one row per interrogation-"
                 "window centre with displacement (µm), correlation signal-to-noise "
                 "(`qfactor`) and a `replaced` flag. SPARSE, on the window grid — use "
                 "Rasterize Field for a dense per-voxel map. Downstream nodes select it "
                 "by this name."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("reference_mode", ["previous_frame", "fixed_frame", "ensemble"],
                default="previous_frame", label="Reference",
                description=
                "Which image each frame is correlated AGAINST, and whether pairs are "
                "reported individually or pooled. It decides if the field is per-step "
                "motion (flow), total displacement since a chosen state (deformation), "
                "or one time-averaged field for the whole series. With an external "
                "Reference Dataset wired, pairing is always against it (fixed).",
                choice_docs={
                    "previous_frame":
                        "Correlate each frame against the one before it (t=0 emits "
                        "nothing). The natural mode for FLOW: every pair spans one frame "
                        "interval, so displacements stay small and correlation stays "
                        "reliable no matter how far material travels over the series. "
                        "Turn on Velocity for µm/s directly.",
                    "fixed_frame":
                        "Correlate every frame against one chosen frame of the same "
                        "series (the self-pair reports ~0). Displacement is TOTAL motion "
                        "since that state — the right mode for deformation against a rest "
                        "state, e.g. substrate bead fields — until motion grows beyond "
                        "what the window ladder can span.",
                    "ensemble":
                        "Average the CORRELATION PLANES of all consecutive pairs before "
                        "peak-finding and emit ONE time-averaged field per position/plane "
                        "at t=0 — the micro-PIV method for seeding too sparse for any "
                        "single pair (true signal accumulates over pairs, random peaks "
                        "cancel; measured 2.6 → 0.4 px at ~3 particles/window over 16 "
                        "pairs). Assumes statistically STEADY flow — real temporal change "
                        "is averaged away, not detected. With an external Reference it "
                        "averages (reference → t) pairs: a static deformation measured "
                        "many times.",
                }),
           Mode("correlation", ["circular", "linear"],
                default="circular", label="Correlation",
                description=
                "How each window pair is cross-correlated. The choice trades speed "
                "against robustness to background and brightness changes between the two "
                "frames.",
                choice_docs={
                    "circular":
                        "FFT correlation with periodic wraparound — OpenPIV's default and "
                        "the fastest path. Assumes the pattern effectively continues "
                        "across window edges; near strong brightness boundaries the "
                        "wraparound can bias edge vectors. Sound for texture-filled "
                        "frames.",
                    "linear":
                        "Zero-padded correlation on intensity-NORMALIZED windows (each "
                        "window mean-subtracted and variance-scaled), so uneven "
                        "background, vignetting and slow bleaching contribute nothing. "
                        "Roughly 2× slower; the robust choice for fluorescence on a dark "
                        "background.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="2D particle image velocimetry (OpenPIV: multipass FFT correlation with "
                "window deformation, universal outlier detection, per-vector S/N) → a "
                "Point displacement field per frame pair; previous/fixed self-reference, "
                "ENSEMBLE correlation averaging (one time-averaged field — the micro-PIV "
                "method for sparse seeding), or an external reference Dataset + optional "
                "ROI mask; optional vy/vx/speed µm/s via the file's dt_s and optional "
                "per-vector unc_y/unc_x µm (Sciacchitano image-matching, dep: pivuq). "
                "The FLOW sibling of DIC — independent windows, far faster, per-vector "
                "confidence. Wired to the piv_field kernel; needs openpiv (dep-gated: "
                "friendly ImportError at run time until installed).")
