"""Registration (``registration.stabilize``) — Register-once (on a reference channel) / apply-to-all-channels stabilization;."""

from __future__ import annotations

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.drift_layers import _layers_drift
from nodegraph.catalog._shared.sampling import _sampled

# ── Registration / stabilization (register-once on a ref channel, apply-to-all) ─

def _compute_stabilize(ctx: EvalContext) -> Dataset:
    """Full registration / drift-stabilization (ported v1 ``registration`` kernel):
    estimate a per-frame transform bundle on a **reference channel** (mid-z of the T
    series) and apply the *same* bundle to every channel and z — the register-once/
    apply-to-all design that preserves colocalization. Axis-preserving; the per-frame
    ``(Δy, Δx)`` translation is stored as Frame attributes (``drift_y``/``drift_x``).

    Richer than :func:`_compute_drift` (``align.drift``): ``model`` ∈ translation
    (phase-corr) / euclidean·affine (ECC) / feature (ORB+RANSAC), and ``reference`` ∈
    first / previous (cumulative) / mean / template. WHOLE_SERIES; pixel-space (no
    calibration — pure geometric registration, like ``align.drift``). Kernel:
    :mod:`nodegraph.kernels.registration` (numpy/scipy/skimage/cv2)."""
    from nodegraph.kernels.registration import apply_series, estimate_series
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stabilize needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    model = modes.get("model", "translation")
    reference = modes.get("reference", "previous")
    upsample = max(1, int(ctx.params.get("upsample", 20)))
    highpass = float(ctx.params.get("highpass_sigma", 2.0))
    min_conf = float(ctx.params.get("min_confidence", 0.0))
    ref_c = min(max(0, int(ctx.params.get("ref_channel", 0))), ax.c - 1)
    ref_z = ax.z // 2
    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
    dy = np.zeros((ax.m, ax.t), dtype=float)
    dx = np.zeros((ax.m, ax.t), dtype=float)

    def series_of(m: int, z: int, c: int) -> np.ndarray:
        return np.stack([prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                         for t in range(ax.t)])

    for m in range(ax.m):
        tf = estimate_series(series_of(m, ref_z, ref_c), model=model,
                             reference=reference, upsample=upsample,
                             highpass_sigma=highpass, min_confidence=min_conf)
        shifts = np.asarray(tf["shifts"], dtype=float)
        dy[m, :], dx[m, :] = shifts[:, 0], shifts[:, 1]
        for c in range(ax.c):
            for z in range(ax.z):
                out[m, :, z, c] = apply_series(series_of(m, z, c), tf)
    res = _sampled(ds.with_image(ArrayProvider(out)), f"registration.stabilize[{model}]")
    # Same reason as align.drift: the content has been moved (and for the non-rigid models
    # WARPED) under an unchanged index grid, so the per-M corner no longer describes it.
    # Dropped rather than carried forward as a claim that cannot be honoured.
    res = res.with_metadata(origin_um=None)
    res = res.with_layer(Domain.FRAME, "drift_y", dy)
    return res.with_layer(Domain.FRAME, "drift_x", dx)
register_node(
    _compute_stabilize, op_key="registration.stabilize", label="Registration",
    extra_layers=_layers_drift,
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
        InFloat("highpass_sigma", "Highpass σ", unit="px", field=True, default=2.0,
                description=
                "Width of the high-pass filter applied BEFORE correlating, in pixels — it "
                "removes smooth illumination gradients that would otherwise dominate the "
                "correlation peak and pull the estimate toward zero shift. In PIXELS rather "
                "than µm because it acts on the correlation, not on physical structure. "
                "LARGER filters less; 0 disables it. Only the shift ESTIMATE is filtered — "
                "the output pixels are never high-passed."),
        InFloat("min_confidence", "Min confidence", unit="", field=True, default=0.0,
                description=
                "Reject a frame's measured shift when the correlation peak is weaker than this, "
                "falling back rather than trusting it. 0 (the default) accepts every estimate. "
                "Raise it when some frames are blank, defocused or bleached: a spurious shift "
                "from a frame with no real structure would otherwise be applied as a genuine "
                "correction and physically displace that frame's data. The trade-off is that "
                "too high a value rejects valid shifts and leaves real drift uncorrected."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("model", ["translation", "euclidean", "affine", "feature"],
                default="translation", label="Model",
                description=
                "How much freedom the per-frame transform has. More degrees of freedom can "
                "correct more kinds of motion, and can also absorb REAL deformation into the "
                "registration — so use the least that fits your data. The transform is "
                "estimated on the reference channel and applied to every channel and z, so "
                "colocalization is preserved whichever model you pick.",
                choice_docs={
                    "translation":
                        "Sub-pixel shift only, by phase correlation. Two parameters, so it is "
                        "the fastest, the most robust, and incapable of distorting anything — "
                        "the right choice for ordinary stage drift, and the default. It cannot "
                        "correct any rotation.",
                    "euclidean":
                        "Shift plus RIGID rotation (ECC, seeded from a phase-correlation "
                        "shift). Handles a dish that was knocked or a sample that rotated, "
                        "while still preserving distances and angles — so nothing you measure "
                        "afterwards is stretched. Needs decent contrast to converge.",
                    "affine":
                        "Shift, rotation, scale and shear (ECC). Corrects apparent stretch from "
                        "focus or thermal changes, and is the first model that can silently "
                        "absorb genuine deformation — if the specimen is what deforms, this "
                        "removes the signal you were measuring. Slowest of the ECC models.",
                    "feature":
                        "Detect ORB keypoints in each frame, match them and fit a transform "
                        "with RANSAC. The only option that survives LARGE motion, where the "
                        "correlation-based models have no overlap to lock onto — and the only "
                        "one that can fail outright on smooth, featureless images, since it "
                        "needs distinctive corners to match.",
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
                        "(bleaching, growth, a cell leaving).",
                    "previous":
                        "Register each frame to the one before it and accumulate the shifts. "
                        "Consecutive frames always look alike, so individual matches are the "
                        "most reliable — but every small error is added to all later frames, "
                        "producing slow cumulative drift over hundreds of timepoints. The "
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
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Register-once (on a reference channel) / apply-to-all-channels "
                "stabilization; translation/euclidean/affine/feature models, "
                "first/previous/mean/template anchor; stores per-frame shift (ported "
                "v1 kernel).")
