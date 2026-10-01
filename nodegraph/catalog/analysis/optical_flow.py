"""Optical Flow (``analysis.optical_flow``) — dense per-pixel 2D motion (skimage TV-L1 / iterative Lucas-Kanade) → Voxel displacement layers ``<name>_y``/``<name>_x``/``<name>_mag`` in µm;."""

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
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.progress import _UnitBar

# ── Optical Flow — the DENSE motion sibling of analysis.piv ─────────────────────
#
# Where PIV cross-correlates INTERROGATION WINDOWS (sparse vectors, each backed by
# window-scale texture statistics + a per-vector S/N), optical flow solves a per-PIXEL
# motion field from the brightness-constancy equation. The cell-biology literature is
# explicit about when each wins (Vig et al. 2016; Tran et al. 2024): dense, confluent
# texture — monolayers, cytoplasmic streaming — favours optical flow; sparse tracers
# favour PIV (and its ensemble mode). Both estimators here are scikit-image's, so this
# node adds NO dependency. V3 W5-P2 companion node (roadmap: "covers the low-texture
# regime where the correlation nodes fail; displacement only, densely").


def _layers_optical_flow(params, modes):
    """Announce the derived Voxel layer names (`<name>_y/_x/_mag`) to the edit-time
    layer catalog — the `name` socket alone cannot describe them. Total: never raises."""
    base = params.get("name") or "flow"
    return tuple((Domain.VOXEL, f"{base}_{s}") for s in ("y", "x", "mag"))


def _compute_optical_flow(ctx: EvalContext) -> Dataset:
    """Dense 2D optical flow per frame pair (scikit-image ``optical_flow_tvl1`` /
    ``optical_flow_ilk``) → three **Voxel** layers, ``<name>_y``/``<name>_x``/
    ``<name>_mag``, holding the per-pixel displacement in **µm** (+y down rows) — the
    dense motion sibling of ``analysis.piv``.

    Resolved design (§0 grill, 2026-08-11): analysis → Voxel layers; one channel is
    analysed per (m, z) line and the flow written into THAT channel's slot of the
    output lattice — every other channel, and every timepoint without a pair (t=0 in
    ``previous_frame`` mode), is **NaN**: an honest absence, where 0 would claim "no
    motion". 2D per-plane (skimage's estimators are nD, but volumetric TV-L1 cost is
    unjustified here — a z-stack is analysed plane-by-plane); no DimMode. Footprint
    WHOLE_SERIES / {t, y, x} (``previous_frame`` reads t-1 and t). Layers are float32
    (the lattice is dense: 3 full-size layers per run; µm at float32 is ~1e-7 relative
    precision, far below any flow estimator's error). No meta_transform (axes and
    calibration untouched; the new layers are announced via ``extra_layers``). Backend
    signatures re-verified in-env against scikit-image (probed empirically: both
    estimators return ``(2, H, W)`` with component 0 = +rows-DOWN, component 1 =
    +cols, signed as A→B motion — this repo's convention with no swap). Sign/units:
    px × ``pixel_size_um`` → µm; velocity is deliberately NOT offered here (rates live
    on ``analysis.piv``; divide by ``dt_s`` downstream if needed).

    Method choice mirrors the literature: ``tvl1`` (variational, edge-preserving,
    accurate at motion boundaries, slower) vs ``ilk`` (iterative Lucas–Kanade,
    window-smoothed, faster, robust on noise). Both assume BRIGHTNESS CONSTANCY —
    bleaching or reporter dynamics between frames violate it; correct intensity drift
    upstream (``enhance.temporal_gain``) or prefer PIV's ``linear`` correlation, which
    normalizes it away per window. Displacements beyond a few px per pair strain the
    coarse-to-fine warping — for fast motion, PIV's window ladder is the right tool.
    Reference pairing mirrors PIV minus the external input: ``previous_frame`` (flow)
    or ``fixed_frame`` (cumulative, self-pair ≈ 0)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Optical Flow needs an image provider on its input Dataset")
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    ref_mode = modes.get("reference_mode", "previous_frame")
    method = modes.get("method", "tvl1")
    ref_frame = min(max(0, int(ctx.params.get("reference_frame", 0))), max(0, ax.t - 1))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    base = ctx.layer("name")
    n_warp = int(ctx.params.get("num_warp", 0))
    prefilter = bool(ctx.params.get("prefilter", False))

    def _flow_fn():
        if method == "tvl1":
            from skimage.registration import optical_flow_tvl1

            kw = dict(attachment=float(ctx.params.get("attachment", 15.0)),
                      tightness=float(ctx.params.get("tightness", 0.3)),
                      num_warp=(n_warp if n_warp > 0 else 5),
                      num_iter=max(1, int(ctx.params.get("num_iter", 10))),
                      tol=max(0.0, float(ctx.params.get("tol", 1e-4))),
                      prefilter=prefilter)
            return lambda a, b: optical_flow_tvl1(a, b, **kw)
        from skimage.registration import optical_flow_ilk

        kw = dict(radius=max(1, int(ctx.params.get("radius", 7))),
                  num_warp=(n_warp if n_warp > 0 else 10),
                  gaussian=bool(ctx.params.get("gaussian", False)),
                  prefilter=prefilter)
        return lambda a, b: optical_flow_ilk(a, b, **kw)

    flow = _flow_fn()

    def _plane(m: int, t: int, z: int) -> np.ndarray:
        return prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(np.float64)

    out_y = np.full((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), np.nan, dtype=np.float32)
    out_x = np.full_like(out_y, np.nan)
    pairs_per_line = (ax.t - 1) if ref_mode == "previous_frame" else ax.t
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z),
                   units_per_frame=max(1, pairs_per_line), note="optical flow")
    for m in range(ax.m):
        for z in range(ax.z):
            if ax.t < 2:
                bar.skip(pairs_per_line, note=f"m={m} z={z}")
                continue
            if ref_mode == "previous_frame":
                prev = _plane(m, 0, z)
                for t in range(1, ax.t):
                    cur = _plane(m, t, z)
                    f = flow(prev, cur)
                    out_y[m, t, z, c] = f[0] * px
                    out_x[m, t, z, c] = f[1] * px
                    prev = cur
                    bar.finish_unit(note=f"m={m} z={z} t={t}")
            else:                                          # fixed_frame (self-pair ~0)
                head = _plane(m, ref_frame, z)
                for t in range(ax.t):
                    f = flow(head, _plane(m, t, z))
                    out_y[m, t, z, c] = f[0] * px
                    out_x[m, t, z, c] = f[1] * px
                    bar.finish_unit(note=f"m={m} z={z} t={t}")

    mag = np.sqrt(out_y ** 2 + out_x ** 2)                 # NaN propagates honestly
    return (ds.with_layer(Domain.VOXEL, f"{base}_y", out_y)
              .with_layer(Domain.VOXEL, f"{base}_x", out_x)
              .with_layer(Domain.VOXEL, f"{base}_mag", mag)
              .with_metadata(of_method=method,
                             of_reference_mode=ref_mode,
                             of_reference_frame=ref_frame))


register_node(
    batch_aware(_compute_optical_flow), op_key="analysis.optical_flow", label="Optical Flow",
    category="analysis",
    extra_layers=_layers_optical_flow,
    reads_domains=frozenset(), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[
        InDataset(),
        InInt("channel", "Channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the flow is estimated on (0-based). Optical flow assumes "
              "BRIGHTNESS CONSTANCY — pick a channel whose intensity pattern moves "
              "without changing (structural marker, dense texture), not a reporter whose "
              "brightness is the signal. The flow layers are written into this channel's "
              "slot; other channels stay NaN."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame",
              description=
              "Which timepoint is the reference (0-based). Read only in `fixed_frame` "
              "mode, where every frame's flow is measured against this one and the "
              "self-pair reports ~0. `previous_frame` ignores it."),
        InInt("radius", "Window radius", unit="px", field=False, default=7,
              available_in={"method": frozenset({"ilk"})},
              description=
              "Half-size of the local window each pixel's motion is solved over (iLK "
              "only). LARGER smooths the field and resists noise but blurs motion "
              "boundaries — e.g. a migrating cell's edge against still background; "
              "SMALLER resolves boundaries and gets noisier. Roughly the counterpart of "
              "PIV's window size, at per-pixel output resolution."),
        InFloat("attachment", "Attachment", unit="", field=True, default=15.0,
                available_in={"method": frozenset({"tvl1"})},
                description=
                "TV-L1's data-attachment weight: HIGHER trusts the brightness-constancy "
                "term more, giving a field that follows the images closely but picks up "
                "their noise; LOWER trusts the smoothness prior, flattening noise and "
                "fine structure together. skimage's default 15 suits clean imagery; drop "
                "toward 5-10 on noisy fluorescence."),
        InFloat("tightness", "Tightness", unit="", field=True, default=0.3,
                available_in={"method": frozenset({"tvl1"})},
                description=
                "TV-L1's coupling between the smoothed field and the data-attached "
                "auxiliary field. The literature default 0.3 rarely needs touching; "
                "SMALLER couples them more tightly (slower convergence, slightly "
                "sharper), LARGER relaxes the coupling. Change Attachment first."),
        InInt("num_iter", "Iterations", unit="", field=False, default=10,
              available_in={"method": frozenset({"tvl1"})},
              description=
              "Fixed-point iterations of the TV-L1 solver per warp level. MORE converges "
              "the field further at linear cost; the default 10 with Tolerance below is "
              "skimage's balance. Raise it if the field looks under-converged (patchy "
              "in smooth regions) before touching anything else."),
        InFloat("tol", "Tolerance", unit="", field=True, default=1e-4,
                available_in={"method": frozenset({"tvl1"})},
                description=
                "Early-stop threshold on the TV-L1 update norm: iteration stops once "
                "the field changes less than this. TIGHTER (smaller) means Iterations "
                "becomes the binding limit; LOOSER returns sooner with a slightly "
                "under-converged field."),
        InInt("num_warp", "Warps", unit="", field=False, default=0,
              description=
              "Coarse-to-fine warping steps — how many times the moving image is "
              "re-warped by the current estimate and re-solved, which is what lets a "
              "differential method reach displacements beyond ~1 px. 0 = the method's "
              "own default (5 for TV-L1, 10 for iLK). MORE extends displacement reach "
              "at linear cost; it cannot rescue genuinely large motion — use PIV's "
              "window ladder for that."),
        InBool("gaussian", "Gaussian window", field=False, default=False,
               available_in={"method": frozenset({"ilk"})},
               description=
               "Weight the iLK window with a Gaussian instead of a flat top-hat: "
               "centre pixels dominate, giving slightly sharper motion boundaries at "
               "the same radius. Off matches skimage's default."),
        InBool("prefilter", "Prefilter", field=False, default=False,
               description=
               "Median-filter (3×3) the images at each warp level before solving. "
               "Suppresses shot-noise outliers that the brightness-constancy term "
               "otherwise chases — worth turning ON for low-light fluorescence; it "
               "slightly softens fine texture on clean data. Read by both methods."),
        InString("name", "Output layers", field=False, default="flow",
                 description=
                 "Base name of the three Voxel layers this node writes: `<name>_y`, "
                 "`<name>_x` (µm displacement per pair, +y down) and `<name>_mag`. "
                 "DENSE per-pixel maps on the analysed channel; timepoints without a "
                 "pair (t=0 in previous-frame mode) and other channels are NaN — an "
                 "honest absence, not zero motion. Downstream nodes select the layers "
                 "by these names."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("method", ["tvl1", "ilk"], default="tvl1", label="Method",
                description=
                "Which dense flow estimator solves the per-pixel motion. Both are "
                "scikit-image implementations sharing the coarse-to-fine warping "
                "scheme; they differ in what regularizes the field.",
                choice_docs={
                    "tvl1":
                        "Variational TV-L1 (Zach 2007 / IPOL 2013): total-variation "
                        "smoothness with an L1 data term — edge-PRESERVING, so motion "
                        "discontinuities (a cell edge against still background) stay "
                        "sharp, and single-pixel noise is rejected. The accuracy "
                        "choice; several times slower than iLK.",
                    "ilk":
                        "Iterative Lucas–Kanade: each pixel's motion solved over a "
                        "local window (see Window radius) — fast and robust on noisy "
                        "data, but the window smooths across motion boundaries. The "
                        "speed choice, and often sufficient on smooth collective "
                        "motion.",
                }),
           Mode("reference_mode", ["previous_frame", "fixed_frame"],
                default="previous_frame", label="Reference",
                description=
                "Which image each frame's motion is measured AGAINST — per-step flow "
                "or cumulative displacement since a chosen frame. Differential flow "
                "estimators live on SMALL displacements, which is exactly what "
                "per-step pairing preserves.",
                choice_docs={
                    "previous_frame":
                        "Each frame against the one before it (t=0 stays NaN). Keeps "
                        "per-pair motion small — the regime optical flow is good at — "
                        "no matter how far material travels over the series. The "
                        "default, and almost always the right choice here.",
                    "fixed_frame":
                        "Every frame against one chosen frame (self-pair ≈ 0). "
                        "Cumulative motion grows without bound, and a differential "
                        "method degrades once it exceeds a few pixels beyond what the "
                        "warping can bridge — use PIV fixed_frame for large cumulative "
                        "deformation.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="Dense per-pixel 2D motion between frames (scikit-image TV-L1 or "
                "iterative Lucas-Kanade) → Voxel layers `<name>_y`/`<name>_x`/"
                "`<name>_mag` in µm (+y down; NaN where no pair exists). The dense "
                "sibling of analysis.piv: per-pixel resolution on continuous texture "
                "(monolayers, cytoplasmic streaming) where sparse-tracer correlation "
                "wins instead. 2D per-plane; previous/fixed reference; no extra "
                "dependency.")
