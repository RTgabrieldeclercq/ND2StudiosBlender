"""Registration (``registration.stabilize``) — Register-once (on a reference channel) / apply-to-all-channels stabilization;."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import (DimMode, Granularity, InDataset, InFloat, InInt, InString,
                                Mode, OutDataset)
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.drift_layers import _layers_stabilize
from nodegraph.catalog._shared.sampling import _sampled

# ── Registration / stabilization (register-once on a ref channel, apply-to-all) ─

def _region_roi(raw: Any, metadata, H: int, W: int) -> Optional[Dict[str, Any]]:
    """The ``region`` socket (the Draw tool's JSON shape list, or a list handed in
    headless) → the kernel's ``roi`` dict, or ``None`` for the whole frame.

    Same two accepted forms and the same refusal of malformed JSON as ``analysis.roi_mask``
    — a region that silently degraded to the whole frame would look like a node that ran
    fine and quietly estimate on the deforming specimen the user drew around. The shapes are
    shifted into a troubleshooting window first (:func:`shapes_in_frame`) and rasterized by
    the same replay the other shapes consumers use, so Cut / Invert / brush strokes mean
    what they mean everywhere else."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    shapes = raw
    if isinstance(raw, str):
        import json as _json
        try:
            shapes = _json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                f"Registration: `region` is not valid JSON ({exc}). Draw the region with "
                f"the Pick button, or pass a list of shape objects, e.g. "
                '[{"type": "rect", "op": "add", "vertices": [[5,5],[15,15]]}]') from exc
    if shapes in (None, [], {}):
        return None
    if not isinstance(shapes, list):
        raise ValueError(f"Registration: `region` must be a JSON list, got "
                         f"{type(shapes).__name__}")
    from nodegraph.catalog._shared.regions import shapes_in_frame
    from nodegraph.kernels.dic_mesh_region import build_roi_mask, has_region
    shapes = shapes_in_frame(shapes, metadata)
    if not has_region(shapes):
        return None
    mask = np.asarray(build_roi_mask(shapes, H, W), dtype=bool)
    if not mask.any():
        raise ValueError(
            "Registration: the drawn `region` covers no pixel of this frame — it sits "
            "outside the image (or outside the troubleshooting window). Redraw it on the "
            "part of the field that does not move or deform, or clear it to estimate on "
            "the whole frame.")
    return {"kind": "mask", "mask": mask}


def _compute_stabilize(ctx: EvalContext) -> Dataset:
    """Full registration / drift-stabilization (ported v1 ``registration`` kernel):
    estimate a per-frame transform bundle on a **reference channel** and apply the *same*
    bundle to every channel and z — the register-once / apply-to-all design that preserves
    colocalization. Axis-preserving; the per-frame shift is stored as Frame attributes
    (``drift_y``/``drift_x``, plus ``drift_z`` in 3D) alongside the per-frame
    ``drift_confidence``.

    Richer than :func:`_compute_drift` (``align.drift``): ``model`` ∈ translation
    (cross-correlation) / euclidean·affine (ECC) / feature (ORB+RANSAC) / auto (chosen from
    the landmarks), and ``reference`` ∈ first / previous (cumulative) / mean / template.
    Pixel-space (no calibration — pure geometric registration, like ``align.drift``).
    Kernel: :mod:`nodegraph.kernels.registration` (numpy/scipy/skimage/cv2).

    Resolved spec of the 2026-10-07 refinement (``scripts/registration_synthetic_bench.py``
    is the evidence; ``CodeLog/Updates/worklog/2026-10-07_registration-bench-and-fixes.md``
    the record)
    --------------------------------------------------------------------------------------
    * **2D/3D** ``DimMode``. 2D estimates on the MIDDLE z plane of the reference channel
      and applies the in-plane correction to every plane — what the node always did. 3D
      correlates the whole ``(Z, Y, X)`` volume per timepoint and corrects the axial drift
      too (``drift_z``); the warp models keep their planar warp from the middle plane and
      take only ``dz`` from the volume. The bench's star volume showed the 2D path leaving
      100 % of an axial drift in place while its lateral estimate degraded as the mid-plane
      content changed under it; 3D recovers ``dz`` to ~0.4 plane on a 16-plane stack (the
      z error is set by stack depth vs the axial PSF, not by the algorithm).
    * **Footprint** ``WHOLE_SERIES`` in both dims; ``kernel_axes`` ``{t,y,x}`` / ``{t,z,y,x}``.
      The estimate needs every timepoint of a whole plane (or volume), so a scheduler that
      tiled this node or fed it one frame at a time would still be wrong.
    * **Sockets** ``lowpass_sigma`` (px) — the matched-filter half of the band-pass the
      correlation runs on; ``region`` — a drawn region the ESTIMATE is restricted to
      (applied full-frame), which is the standard remedy when the specimen deforms and the
      dish, beads or a stable neighbour do not; ``landmarks`` — clicked correspondences
      between frame 0 and frame ``t`` that pick the model (``auto``) and seed the warp
      estimate. All three are read on every branch.
    * **Confidence** stored per frame so a gated / weak frame is a row in a table, not a
      judgement by eye.
    * **Kernel defaults that changed under this node**: plain cross-correlation instead of
      phase whitening (3–15× more precise here, and it made ``highpass_sigma`` a live
      control — whitening had cancelled it), the ECC seed sign fixed, ECC seeded from the
      feature match when that fit is trusted (rotation beyond ~10° now registers in
      ``first`` mode), and a failed warp estimate gated rather than applied.

    **The estimate is eager, the apply is LAZY** — the same split ``align.drift`` uses,
    and for the same reason. Estimating is inherently whole-series (a global correlation
    per frame) but reads only ONE plane per (m, t) in 2D (the reference channel at mid-z),
    so it costs m·t plane reads no matter how deep or how many-channelled the stack is; 3D
    reads the reference channel's Z planes per (m, t), which is what an axial estimate
    honestly costs. Applying is separable per plane / per volume
    (:func:`~nodegraph.kernels.registration.apply_frame` /
    :func:`~nodegraph.kernels.registration.apply_volume` — frame t's output depends only on
    frame t's pixels and frame t's transform), so it is deferred to a per-unit streaming
    provider and only the units actually pulled are ever computed.

    This used to allocate ``(M,T,Z,C,Y,X)`` float64 up front and apply to every plane
    eagerly, which made the node scale linearly in Z for a viewer showing ONE plane. On a
    16-position 10-plane 4-channel 2048² stack that is 20.0 GiB and 640 plane-warps
    against 2.0 GiB and 64 for the same stack z-projected — so resetting an upstream
    Z-Project to ``none`` turned a working pull into a ~10× slower, 20 GiB one, and the
    Viewer sat on the previous z==1 frame while it ran. The node that looked broken was
    Z-Project; the node that could not absorb the result was this one (2026-08-04)."""
    from nodegraph.kernels.registration import (apply_frame, apply_volume, estimate_series,
                                                parse_landmarks)
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stabilize needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    # Read the lever itself, not ``ctx.is_volume``: that property means "the resolved
    # footprint is WHOLE_VOLUME", and this node's footprint is WHOLE_SERIES in BOTH dims
    # (the estimate always needs every timepoint). The engine hands over the resolved
    # state (the z>1 ⇒ 3D derive included); a bare ctx without a ``dim`` takes the Mode's
    # declared default, exactly as ``NodeInstance.state`` would, so the eager reference
    # and the lazy path can never disagree about which lever they ran under.
    volumetric = str(modes.get("dim", "2D")) == "3D"
    model = modes.get("model", "translation")
    reference = modes.get("reference", "previous")
    upsample = max(1, int(ctx.params.get("upsample", 20)))
    highpass = float(ctx.params.get("highpass_sigma", 2.0))
    lowpass = float(ctx.params.get("lowpass_sigma", 1.0))
    min_conf = float(ctx.params.get("min_confidence", 0.0))
    ref_c = min(max(0, int(ctx.params.get("ref_channel", 0))), ax.c - 1)
    ref_z = ax.z // 2
    roi = _region_roi(ctx.params.get("region"), ds.metadata, ax.y, ax.x)
    landmarks = parse_landmarks(ctx.params.get("landmarks"))
    click_precision = max(0.05, float(ctx.params.get("click_precision", 1.0)))
    nd = 3 if volumetric else 2

    def series_of(m: int) -> np.ndarray:
        if volumetric:
            return np.stack([np.asarray(prov.get_region_volume(0, m, t, ref_c, 0, ax.z,
                                                               0, ax.y, 0, ax.x), dtype=float)
                             for t in range(ax.t)])
        return np.stack([prov.get_region(0, m, t, ref_z, ref_c, 0, ax.y, 0, ax.x).astype(float)
                         for t in range(ax.t)])

    # ── the estimate: eager, one transform bundle per M, off the reference channel ──
    # Held for the lazy closure below: t shifts (or t 2×3 warps) per M — kilobytes, and
    # the whole reason the apply needs nothing else from the estimate pass.
    tfs: Dict[int, Dict[str, Any]] = {}
    shifts = np.zeros((ax.m, ax.t, nd), dtype=float)
    conf = np.ones((ax.m, ax.t), dtype=float)
    resolved_model = model
    lm_summary: Optional[str] = None
    for m in range(ax.m):
        tf = estimate_series(series_of(m), model=model, reference=reference,
                             upsample=upsample, highpass_sigma=highpass,
                             lowpass_sigma=lowpass, min_confidence=min_conf,
                             roi=roi, landmarks=landmarks, click_sigma=click_precision)
        tfs[m] = tf
        shifts[m] = np.asarray(tf["shifts"], dtype=float)
        conf[m] = np.asarray(tf["confidence"], dtype=float)
        resolved_model = str(tf["model"])
        if tf.get("landmarks") is not None:
            lm_summary = str(tf["landmarks"]["summary"])
        ctx.progress(m + 1, ax.m, "estimating registration")

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        # SAME per-unit kernel call as the lazy path below (apply_frame / apply_volume), so
        # the two paths cannot drift in the last bits — the selftest asserts they are identical.
        out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m in range(ax.m):
            for t in range(ax.t):
                for c in range(ax.c):
                    if volumetric:
                        vol = np.asarray(prov.get_region_volume(0, m, t, c, 0, ax.z,
                                                                0, ax.y, 0, ax.x), dtype=float)
                        out[m, t, :, c] = apply_volume(vol, tfs[m], t)
                    else:
                        for z in range(ax.z):
                            plane = prov.get_region(0, m, t, z, c,
                                                    0, ax.y, 0, ax.x).astype(float)
                            out[m, t, z, c] = apply_frame(plane, tfs[m], t)
        res = ds.with_image(ArrayProvider(out))
    else:
        # WHOLE_PLANE / WHOLE_VOLUME unit: a shift/warp needs the whole unit (the vacated
        # edge is filled with zeros, so a tile would invent a border mid-image). The
        # transforms are baked from the eager estimate and fold into the provider fp via
        # the BASE fingerprint — a base change re-estimates and re-keys, as in align.drift.
        fp = stream_fp("stabilize", ctx.op_key, ctx.params,
                       ctx.reads.declared_reads(), (), prov)
        if volumetric:
            res = ds.with_image(VolumeComputeProvider(
                prov, lambda v, m, t, c: apply_volume(v, tfs[m], t), fp=fp, cache=cache))
        else:
            res = ds.with_image(MapComputeProvider(
                prov, lambda a, m, t, z, c, *_: apply_frame(a, tfs[m], t),
                unit="plane", fp=fp, cache=cache))
    res = _sampled(res, f"registration.stabilize[{resolved_model},{'3D' if volumetric else '2D'}]")
    # Same reason as align.drift: the content has been moved (and for the non-rigid models
    # WARPED) under an unchanged index grid, so the per-M corner no longer describes it.
    # Dropped rather than carried forward as a claim that cannot be honoured.
    res = res.with_metadata(origin_um=None)
    if volumetric:
        res = res.with_layer(Domain.FRAME, "drift_z", shifts[:, :, 0])
    res = res.with_layer(Domain.FRAME, "drift_y", shifts[:, :, -2])
    res = res.with_layer(Domain.FRAME, "drift_x", shifts[:, :, -1])
    res = res.with_layer(Domain.FRAME, "drift_confidence", conf)
    # Non-calibration provenance: which model actually ran (matters under `auto`) and what
    # the landmark fit said, so the inspector / a report can show the user the evidence.
    extra = {"registration_model": resolved_model}
    if lm_summary:
        extra["registration_landmarks"] = lm_summary
    return res.with_metadata(**extra)


_DIM_DOCS = {
    "2D": "Estimate on the MIDDLE z plane of the reference channel and apply the same "
          "in-plane correction to every plane. Right for single-plane data and for a stack "
          "whose focus did not move. On a z-stack it ignores axial drift entirely: a sample "
          "that drifted two planes stays two planes off, and the lateral estimate itself "
          "degrades as the mid-plane's content changes underneath it.",
    "3D": "Correlate the whole (Z, Y, X) volume of the reference channel per timepoint and "
          "correct the axial drift as well as the lateral one, storing it as drift_z. A warp "
          "model (euclidean / affine / feature) still estimates its in-plane warp on the "
          "middle plane and takes only dz from the volume. Costs Z plane reads per timepoint "
          "for the estimate instead of one. The axial precision is set by how many planes "
          "the stack has against the axial spot size (about 0.4 plane on a 16-plane stack "
          "in the bench), so a very thin stack gains little.",
}

register_node(
    batch_aware(_compute_stabilize), op_key="registration.stabilize", label="Registration",
    extra_layers=_layers_stabilize,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.FRAME}),
    category="registration",
    inputs=[
        InDataset(),
        InInt("ref_channel", "Ref channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the shift is MEASURED on (0-based). The measured shift is then "
              "applied to every channel, which keeps them in register with each other — "
              "estimating per-channel would let them drift apart. Pick the channel with stable, "
              "high-contrast structure; a sparse or blinking reporter channel gives a noisy "
              "estimate that is then imposed on all the others, so this choice affects the "
              "whole dataset even though it names one channel."),
        InInt("upsample", "Upsample", unit="", field=False, default=20,
              description=
              "Sub-pixel precision factor for the cross-correlation peak: the shift is resolved "
              "to about 1/this of a pixel, so 20 gives roughly 0.05 px. HIGHER costs a little "
              "more time and buys nothing once you pass the noise floor of the data; LOWER "
              "quantizes the correction and leaves visible residual jitter. 1 disables "
              "sub-pixel refinement entirely."),
        InFloat("lowpass_sigma", "Lowpass σ", unit="px", field=True, default=1.0,
                description=
                "Gaussian smoothing applied BEFORE correlating, in pixels — the matched filter "
                "for point-like features. Set it near the spot radius of the features that "
                "should drive the alignment (beads, nuclei, puncta: 1–1.5 px). It is what "
                "makes a sparse, noisy field registrable: on a field of faint beads at a peak "
                "signal-to-noise of 8 the error drops from about 2 px to 0.1 px. LARGER "
                "suppresses more noise and blurs fine texture that a dense specimen could have "
                "locked onto; 0 disables it. Must stay below Highpass σ or no band is left. "
                "Only the ESTIMATE is filtered — output pixels are never smoothed."),
        InFloat("highpass_sigma", "Highpass σ", unit="px", field=True, default=2.0,
                description=
                "Width of the high-pass filter applied BEFORE correlating, in pixels — it "
                "removes smooth illumination gradients, vignetting and bleaching trends that "
                "would otherwise dominate the correlation peak and pull the estimate toward "
                "zero shift. In PIXELS rather than µm because it acts on the correlation, not "
                "on physical structure. LARGER filters less; 0 disables it, and with it off a "
                "smooth textured specimen under a drifting illumination gradient registers "
                "30× worse. Keep it above Lowpass σ. Only the shift ESTIMATE is filtered — "
                "the output pixels are never high-passed."),
        InFloat("min_confidence", "Min confidence", unit="", field=True, default=0.0,
                description=
                "Reject a frame's measured transform when its confidence — the correlation "
                "after alignment for translation, the ECC coefficient for euclidean/affine, "
                "the RANSAC inlier fraction for feature — is below this, holding the last good "
                "transform (first/mean/template) or skipping the increment (previous) instead. "
                "0 (the default) accepts every ESTIMATE, but an outright failure (a warp that "
                "did not converge or is not a plausible motion) is always rejected — there is "
                "nothing to apply. Raise it when some frames are blank, defocused or bleached: "
                "a spurious shift from a frame with no real structure would otherwise be "
                "applied as a genuine correction. Too high rejects valid shifts and leaves real "
                "drift uncorrected. Every frame's value is stored as drift_confidence, so look "
                "there before choosing."),
        InString("region", "Estimate in region", field=False, default="",
                 pick_kind="shapes",
                 description=
                 "Restrict the ESTIMATE to a drawn region; the correction is still applied to "
                 "the whole frame. This is the remedy when the specimen itself deforms, grows "
                 "or moves while the dish, beads or a neighbouring cell do not: draw around "
                 "what should hold still and the deformation stops being absorbed into (or "
                 "biasing) the drift. Empty = the whole frame. A rectangle keeps sub-pixel "
                 "precision; a freeform region correlates at whole-pixel precision for the "
                 "translation model (ECC and feature are unaffected)."),
        InString("landmarks", "Landmarks", field=False, default="",
                 description=
                 "Point pairs you identified by hand: the same physical places on frame 0 and "
                 "on one later frame, as 't=5; row,col -> row,col; row,col -> row,col' (or the "
                 "equivalent JSON). From them the node fits every transform family and reports "
                 "the residual of each in the output metadata (registration_landmarks): with "
                 "Model = auto the simplest family within about 1 px of click precision is "
                 "chosen, and if even affine cannot fit the clicks the field is flagged "
                 "NON-RIGID — no global transform will register it. For euclidean/affine the "
                 "fitted warp also seeds the estimate of every frame (scaled to that frame's "
                 "share of the motion), which is what lets a large rotation register when the "
                 "automatic seed cannot find it. Five well-spread pairs are enough to tell the "
                 "families apart; three cannot (affine fits any three exactly). Empty = no "
                 "landmarks."),
        InFloat("click_precision", "Click precision", unit="px", field=False, default=1.0,
                description=
                "How far off each landmark click may be, in pixels — the yardstick for the "
                "NON-RIGID verdict only (which family fits best is decided from the clicks' own "
                "scatter and does not depend on this). A pure drift clicked at 2 px precision "
                "but judged at 1 px is flagged non-rigid; set it to what your clicks really "
                "achieve: about 0.3 px when zoomed onto a bead, 1 px by eye, 2 px on soft "
                "structure. Only read when Landmarks is filled."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(choice_docs=_DIM_DOCS),
           Mode("model", ["translation", "euclidean", "affine", "feature", "auto"],
                default="translation", label="Model",
                description=
                "How much freedom the per-frame transform has. More degrees of freedom can "
                "correct more kinds of motion, and can also absorb REAL deformation into the "
                "registration — so use the least that fits your data. The transform is "
                "estimated on the reference channel and applied to every channel and z, so "
                "colocalization is preserved whichever model you pick.",
                choice_docs={
                    "translation":
                        "Sub-pixel shift only, by cross-correlation of the band-passed images. "
                        "Two parameters, so it is the fastest, the most robust, and incapable "
                        "of distorting anything — the right choice for ordinary stage drift, "
                        "and the default. It cannot correct any rotation, and on a specimen that "
                        "deforms it measures the deformation's average pull as drift unless the "
                        "estimate is restricted to a rigid region.",
                    "euclidean":
                        "Shift plus RIGID rotation (ECC, seeded from the feature match when "
                        "that is trusted, else from the correlation shift). Handles a dish that "
                        "was knocked or a sample that rotated, while still preserving distances "
                        "and angles — so nothing you measure afterwards is stretched. Needs "
                        "decent texture to converge; on a sparse field of spots it is no better "
                        "than translation.",
                    "affine":
                        "Shift, rotation, scale and shear (ECC, seeded like euclidean). Corrects "
                        "apparent stretch from focus or thermal changes, and is the first model "
                        "that can silently absorb genuine deformation — if the specimen is what "
                        "deforms, this removes the signal you were measuring. Slowest of the ECC "
                        "models.",
                    "feature":
                        "Detect ORB keypoints in each frame, match them and fit a transform "
                        "with RANSAC. Survives LARGE motion and rotation where the correlation "
                        "models have nothing to lock onto, at about 0.3–0.5 px precision rather "
                        "than the ~0.03 px of ECC — which is why euclidean/affine now use it as "
                        "a seed and then refine. Fails outright on smooth, featureless images "
                        "and on sparse fields of round spots, since it needs distinctive corners.",
                    "auto":
                        "Let the landmarks decide: the simplest family whose fit to the clicked "
                        "pairs is within click precision becomes the model (translation, "
                        "euclidean, or affine for a similarity/affine fit), and the per-family "
                        "residuals are written to the output metadata. Without landmarks this "
                        "is translation.",
                }),
           Mode("reference", ["first", "previous", "mean", "template"],
                default="previous", label="Reference",
                description=
                "What each frame is registered TO. This is the choice that decides whether "
                "errors accumulate over a long series and whether a bleaching or changing "
                "specimen can still be matched to its anchor — it matters more than the model "
                "on a long timelapse.",
                choice_docs={
                    "first":
                        "Register every frame to frame 0. Errors never accumulate and the "
                        "output is anchored to the acquisition's own start, which is what you "
                        "want when positions must stay comparable to the first frame. It fails "
                        "late in a series whose appearance has drifted far from frame 0 "
                        "(bleaching, growth, a cell leaving) and when the cumulative motion "
                        "leaves little overlap with frame 0.",
                    "previous":
                        "Register each frame to the one before it and accumulate the shifts. "
                        "Consecutive frames always look alike, so individual matches are the "
                        "most reliable — but every small error is added to all later frames, "
                        "producing slow cumulative drift over hundreds of timepoints (about "
                        "0.02 px per frame on a dense field, 0.05 px on a sparse one). The "
                        "default.",
                    "mean":
                        "Register every frame to the average of the whole series. The anchor is "
                        "less noisy than any single frame and no error accumulates; its "
                        "weakness is that an UNREGISTERED average is blurred by the very drift "
                        "being corrected, so it is a poor anchor when motion is large.",
                    "template":
                        "Two passes: roughly stabilize with a cheap translation-to-previous "
                        "pass, average the result into a sharp template, then register every "
                        "frame to that. Avoids both cumulative drift and the atypical-anchor "
                        "problem, at roughly double the cost — the recommended choice for long "
                        "or photobleaching series.",
                })],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes={"2D": frozenset({"t", "y", "x"}), "3D": frozenset({"t", "z", "y", "x"})},
    description="Register-once (on a reference channel) / apply-to-all-channels "
                "stabilization; translation/euclidean/affine/feature/auto models, "
                "first/previous/mean/template anchor, 2D or 3D (axial drift), a drawn "
                "estimate region and hand-clicked landmarks; stores the per-frame shift and "
                "confidence as Frame attributes.")
