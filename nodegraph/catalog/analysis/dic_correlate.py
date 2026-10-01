"""DIC (pyALDIC) (``analysis.dic_correlate``) — 2D digital image correlation (pyALDIC: IC-GN + ADMM over an adaptive FE mesh) → a Point displacement field (subset centers, disp µm, optional strain);."""

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
from nodegraph.catalog._shared.columns import dvc_field_columns, on_layer
from nodegraph.catalog._shared.dvc import _dvc_rows
from nodegraph.catalog._shared.progress import _UnitBar

# ── DIC (pyALDIC) — 2D digital image correlation, the image sibling of DVC ──────
#
# Wires the vendored `dic_correlate` kernel (nd2studios v1 → `nodegraph.kernels.dic_correlate`).
# The kernel is a thin adapter around the third-party **al-dic (pyALDIC)** solver, which it
# imports LAZILY (inside `run_pyaldic_pair`). So this node is FULLY wired — it owns the m/t/z
# loop, the reference pairing, unit handling, and the Point output — but its compute raises a
# friendly ImportError only when actually RUN, until `pip install al-dic` (upstream:
# https://github.com/zachtong/pyALDIC). This mirrors the DVC port (`analysis.dvc_field`): DIC
# is its 2D form (correlate a ref/def IMAGE pair → a displacement field on a coarse FE grid).


class _DICPlaneSeq:
    """Lazily-indexed ``(n,)`` view of the planes one DIC series solve consumes.

    ``run_pyaldic_series`` indexes its ``images`` argument on demand (al_dic pulls each
    frame exactly once), so handing it this instead of a materialized list keeps peak
    memory at ~two planes no matter how long T is — a 100-frame 2048² stack would
    otherwise be 3.2 GB of float64 before the solver starts."""

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
def _compute_dic_correlate(ctx: EvalContext) -> Dataset:
    """2D Digital Image Correlation (pyALDIC: IC-GN subset matching + ADMM over an adaptive
    quadtree FE mesh) → a **Point** displacement field on the correlation grid — the 2D image
    sibling of ``analysis.dvc_field``. Loops m/z on one reference channel and hands each
    (m, z) line the WHOLE ordered T stack in ONE
    :func:`nodegraph.kernels.dic_correlate.run_pyaldic_series` call, which imports the
    external ``al-dic`` package **lazily** → the node is fully wired but raises a clear
    ImportError at run time until ``pip install al-dic`` (kernel ready).

    **Why one series call and not a pair per frame (V2.18).** al_dic caches its reference
    bundle, subpb1 precompute and 6-DOF IC-GN context keyed on the reference frame index,
    warm-starts each frame from the previous solution instead of re-running the FFT integer
    search, and remembers the search radius that worked for that reference. A
    pair-per-frame loop threw all of that away every frame. A/B against the previous
    implementation, same graph and same point count: **2.3x at 256²/T=9, 2.5x at 512²/T=6,
    3.2x with a disc ROI** — for BIT-IDENTICAL displacements, because the node's two
    reference modes map straight onto al_dic's own ``accumulative`` / ``incremental`` and
    every frame still gets its own coarse search. This is a restructuring, not an
    approximation. Frames are pulled lazily through :class:`_DICPlaneSeq`, so the series
    call costs no more memory than the pair loop did.

    Reference pairing mirrors DVC: a ``reference_mode`` lever {fixed_frame | previous_frame}
    + ``reference_frame``, OR an optional external ``reference`` Dataset (a separate undeformed
    acquisition — always a FIXED reference). An optional ``roi`` Voxel mask layer (e.g. from
    ``analysis.roi_mask``) restricts correlation to a region — and now also tightens the
    correlation grid to the mask's bounding box, so subsets outside the ROI are never solved
    (and the grid samples the ROI more densely for the same cost).

    A ``solver`` Mode picks AL-DIC (local IC-GN reconciled by the global ADMM step) or plain
    Local DIC (~2x faster; upstream's own case9 shows no accuracy cost on smooth fields).
    ``compute_strain`` adds al_dic's FEM nodal displacement gradients as ``strain_*`` columns
    — the same columns the DVC sibling emits, and materially better than differencing the
    coarse grid downstream (0.01984 vs truth 0.02 on a 2% biaxial case, against 0.01928 from
    the raw per-subset gradient).

    Displacements are stored in **µm** (grid px × ``pixel_size_um``, component order (dy, dx));
    the FE grid is COARSER than the image (pitch = the snapped ``winstepsize``) — feed the
    Point field to ``transform.rasterize_field`` for a Voxel heatmap. Nodes the solver could
    not resolve (masked-out subsets) come back non-finite and are DROPPED rather than written
    as NaN rows that would poison downstream statistics. 2D only (DIC has no volumetric form —
    use ``analysis.dvc_field`` for 3D). Footprint WHOLE_SERIES (crosses T); reuses the shared
    :func:`_dvc_rows` Point flattener (``qfactor`` stays absent — al-dic 0.7.2 keeps its
    per-point NCC quality inside the FFT search and never surfaces it on the result)."""
    from nodegraph.kernels.dic_correlate import run_pyaldic_series
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DIC needs an image provider on its (deformed) input Dataset")
    ax = prov.axes
    ref_ds = ctx.input("reference")
    ref_prov = ref_ds.image if ref_ds is not None else None
    if ref_ds is not None and ref_prov is None:
        raise ValueError("the DIC 'reference' input Dataset has no image provider")
    modes = ctx.params.get("__modes__", {})
    ref_mode = modes.get("reference_mode", "fixed_frame")
    solver = modes.get("solver", "aldic")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    vox: Tuple[float, ...] = (px, px)                    # (y, x) µm/px — kernel §3
    want_strain = bool(ctx.params.get("compute_strain", False))
    params = {
        "winsize": max(2, int(ctx.params.get("winsize", 40))),
        "winstepsize": max(2, int(ctx.params.get("winstepsize", 16))),
        "winsize_min": max(2, int(ctx.params.get("winsize_min", 8))),
        # "fft" — every frame gets its OWN coarse integer search, which is what upstream's
        # quickstart recommends for real datasets and what the old pair-per-frame loop did
        # implicitly. NOT "auto": al_dic maps that to "previous", warm-starting each frame
        # from the last solution, and in fixed_frame mode the prepended reference self-pair
        # solves to ~0 and then hands that ~0 forward as the next frame's guess. A series
        # whose very first step is large then has to find it by local refinement alone,
        # which does not converge — measured as a 6.3 px error on an 11 px shift that the
        # FFT path recovers to 0.003 px. The reference-bundle cache, not the warm start, is
        # where the series speedup comes from, so this costs almost nothing.
        "init_guess_mode": "fft",
        "mu": float(ctx.params.get("mu", 1e-3)),
        "tol": float(ctx.params.get("tol", 1e-2)),
        "admm_max_iter": max(1, int(ctx.params.get("admm_max_iter", 3))),
        "icgn_max_iter": max(1, int(ctx.params.get("icgn_max_iter", 100))),
        "disp_smoothness": max(0.0, float(ctx.params.get("disp_smoothness", 5e-4))),
        "strain_smoothness": max(0.0, float(ctx.params.get("strain_smoothness", 1e-5))),
        "size_of_fft_search_region": max(1, int(ctx.params.get("search_range", 20))),
        "use_global_step": solver != "local",
        "compute_strain": want_strain,
    }
    layer = ctx.layer("name")
    # optional ROI: a Voxel mask layer (e.g. analysis.roi_mask) limits correlation; absent → none
    roi_name = ctx.layer("roi")
    roi_attr = ds.get(Domain.VOXEL, roi_name) if roi_name else None
    roi6 = roi_attr.values if roi_attr is not None else None

    def _roi_plane(m: int, t: int, z: int, cc: int):
        if roi6 is None:
            return None
        s = roi6.shape
        return roi6[min(m, s[0] - 1), min(t, s[1] - 1), min(z, s[2] - 1), min(cc, s[3] - 1)]

    def _plan(m: int, z: int) -> Tuple[List[Tuple[Any, int, int, int, int]], List[int], str]:
        """(plane plan, the t each RESULT belongs to, al_dic reference_mode).

        fixed_frame prepends the reference plane so al_dic's ``accumulative`` mode
        correlates every t against it (including t == the reference, which self-pairs to
        ~0 exactly as the pair loop did). previous_frame feeds the stack in order and
        uses ``incremental``, whose per-step ``U`` is the increment t-1 → t."""
        if ref_mode == "previous_frame" and ref_prov is None:
            return ([(prov, m, t, z, c) for t in range(ax.t)],
                    list(range(1, ax.t)), "incremental")
        if ref_prov is not None:                          # external reference file (fixed)
            rax = ref_prov.axes
            head = (ref_prov, min(m, rax.m - 1), min(max(0, ref_frame), rax.t - 1),
                    min(z, max(0, rax.z - 1)), min(c, max(0, rax.c - 1)))
        else:
            head = (prov, m, min(max(0, ref_frame), ax.t - 1), z, c)
        return ([head] + [(prov, m, t, z, c) for t in range(ax.t)],
                list(range(ax.t)), "accumulative")

    rows: list = []
    n_dropped = 0
    # One IC-GN/ADMM series solve per (m, z) line. The solve is long enough that whole-unit
    # counting would freeze the sub bar through it, so the kernel's 0-100 callback drives
    # the sub bar and the frame bar steps once per completed (m, z) series (V2.18 — it
    # counted timepoints while each pair was its own solve).
    bar = _UnitBar(ctx, frames=max(1, ax.m * ax.z), units_per_frame=1, note="correlating")
    for m in range(ax.m):
        for z in range(ax.z):
            plan, result_t, al_mode = _plan(m, z)
            if len(plan) < 2:                             # a 1-frame series has no pair
                bar.finish_unit(note=f"m={m} z={z}")
                continue
            masks = ([_roi_plane(m, t, z, c) for (_p, _m, t, _z, _c) in plan]
                     if roi6 is not None else None)
            series = run_pyaldic_series(
                _DICPlaneSeq(plan), masks, params, vox, reference_mode=al_mode,
                progress_cb=bar.emit, resample="nodes")
            key = "primary" if al_mode == "accumulative" else "increment"
            for i, t in enumerate(result_t):
                if i >= len(series):
                    break
                row = _dvc_rows(series[i][key], vox, m=m, t=t, c=c, z_plane=z)
                keep = np.isfinite(row["disp_y"]) & np.isfinite(row["disp_x"])
                if not keep.all():
                    n_dropped += int((~keep).sum())
                    row = {k: v[keep] for k, v in row.items()}
                if row["m"].size:
                    rows.append(row)
            bar.finish_unit(note=f"m={m} z={z}")

    # Provenance (§7b): DIC-specific keys (NOT the dvc_* keys accumulate_field consumes — a
    # DIC 2D increment is not an ALDVC volume increment, so it must not be fed there).
    prov_md = {"dic_reference_mode": (ref_mode if ref_prov is None else "fixed_frame"),
               "dic_reference_frame": ref_frame, "dic_solver": solver,
               "dic_unsolved_points": n_dropped}
    if not rows:
        empty = point_table(np.zeros((0, 2)), z_kind="plane_index", layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(
        StructureTable(Domain.POINT, merged, layer=layer, z_kind="plane_index"))
        .with_metadata(**prov_md))

def _columns_dic_correlate(params, modes, incoming):
    """The correlation grid this node writes, as ``_shared.dvc._dvc_rows`` builds it.

    Derived from :func:`dvc_field_columns` rather than transcribed, so the declaration and the
    flattener cannot drift on the axis-dependent ``disp_``/``strain_`` names. The strain block
    is declared unconditionally: it is present whenever the solver was asked for it, and the
    only cost of naming it on a strain-free run is a menu entry the compute then refuses by
    name — whereas omitting it would make a real column unpickable, which is the direction
    this catalog must not err in."""
    try:
        return on_layer(Domain.POINT, str((params or {}).get("name") or "dic"),
                        list(dvc_field_columns(is_3d=False)) + ["qfactor"])
    except Exception:                        # pragma: no cover - defensive
        return ()

register_node(
    _compute_dic_correlate, op_key="analysis.dic_correlate", label="DIC (pyALDIC)",
    adds_columns=_columns_dic_correlate,
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference"),
        InInt("channel", "Channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the correlation runs on (0-based). Displacement comes from image "
              "texture, so choose the channel with stable, high-contrast speckle — a fiducial "
              "or bead channel rather than a reporter whose brightness changes biologically. A "
              "channel that changes appearance between frames breaks the constant-brightness "
              "assumption and yields confident but wrong vectors."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame",
              description=
              "Which timepoint is the undeformed reference (0-based). Read only when the "
              "reference mode is `fixed_frame`, where every frame correlates against this one "
              "and displacements are directly comparable across the whole series — until "
              "deformation grows large enough that correlation loses the pattern. The "
              "`previous_frame` mode ignores this and measures increments instead."),
        # AL-DIC solver parameters. Each maps 1:1 onto a field of `al_dic`'s `DICPara`
        # (core/data_structures.py) with the same default; semantics from the AL-DIC /
        # STAQ-DIC method (Yang & Bhattacharya; github.com/zachtong/STAQ-DIC-GUI).
        InInt("winsize", "Subset size", unit="px", field=False, default=40,
              pick_kind="grid", pick_peer="winstepsize",
              description=
              "Edge length of the correlation subset, in pixels — the patch of speckle matched "
              "between reference and deformed images. THE core resolution/noise trade-off: "
              "LARGER contains more pattern so each vector is more reliable, but it averages "
              "deformation over a bigger area and blurs sharp features like crack tips; SMALLER "
              "resolves detail and grows noisy, and below roughly 3 speckles the correlation "
              "stops being unique. Forced even. Independent of Grid step — subsets normally "
              "OVERLAP, and should."),
        InInt("winstepsize", "Grid step", unit="px", field=False, default=16,
              pick_kind="grid", pick_peer="winsize",
              description=
              "Spacing between subset centres, in pixels — this alone sets how many measurement "
              "points you get, and becomes the output grid's step. SMALLER gives a denser field "
              "at superlinear cost; LARGER is faster and can miss localized features entirely. "
              "It does NOT change the accuracy of any single vector (that is Subset size); "
              "neighbouring subsets simply overlap more. Snapped to a power of 2."),
        InInt("search_range", "Search range", unit="px", field=False, default=20,
              description=
              "STARTING radius, in pixels, of the coarse whole-pattern search that hands the "
              "fine solve its first guess — it wants to cover the largest displacement "
              "between a frame and its reference. Getting it wrong is a SPEED cost, not a "
              "wrong answer: when matches pile up against the edge of the window the solver "
              "notices and retries with a bigger one (up to six times, capped at half the "
              "shorter image side), then remembers the radius that worked for the rest of the "
              "series. So too small means repeated wasted searches, too large means every "
              "search is slower than it needed to be. Note it is measured PER PAIR, so "
              "`fixed_frame` must span the whole accumulated drift where `previous_frame` "
              "only spans one step — and on a small crop it is quietly reduced when twice the "
              "range plus Subset size exceeds a quarter of the shorter side."),
        InInt("winsize_min", "Min element", unit="px", field=False, default=8,
              description=
              "Floor on element size during adaptive mesh refinement — how finely the solver may "
              "subdivide near a mask edge, ROI boundary or high-error region. It stops "
              "refinement from producing subsets too small to correlate reliably, so LOWERING it "
              "buys resolution at discontinuities and risks noise there. Snapped to a power of 2 "
              "and clamped to at most Grid step."),
        InInt("icgn_max_iter", "IC-GN iterations", unit="", field=False, default=100,
              description=
              "Iteration cap for the LOCAL solve — the inverse-compositional Gauss–Newton match "
              "of one subset, run independently per point. It normally converges in far fewer "
              "than this, so the cap only bites on subsets that are struggling (poor speckle, "
              "large deformation, edge of the ROI). Raising it rarely rescues those and costs "
              "time on every hard point; lowering it abandons them sooner. Tolerance decides "
              "when convergence is declared."),
        InBool("compute_strain", "Compute strain", field=False, default=False,
               description=
               "Also emit the solver's own nodal displacement gradients as `strain_yy`, "
               "`strain_yx`, `strain_xy`, `strain_xx` columns — the same columns the DVC node "
               "writes. Prefer this to differencing the displacement field downstream: it is a "
               "smoothed finite-element fit rather than a difference of two neighbouring "
               "subsets, which on a 2% biaxial test recovers 0.01984 against a true 0.0200 "
               "where the raw per-subset gradient gives 0.01928. Costs roughly 20% more time. "
               "Off by default so the node stays a pure displacement measurement."),
        InInt("admm_max_iter", "ADMM iterations", unit="", field=False, default=3,
              available_in={"solver": frozenset({"aldic"})},
              description=
              "Iterations of the OUTER augmented-Lagrangian loop that reconciles the independent "
              "per-subset matches with one globally compatible displacement field — this "
              "alternation is what separates AL-DIC from plain local DIC. The method's own "
              "guidance is that 3–5 steps suffice; MORE tightens compatibility and costs a full "
              "pass each, FEWER leaves the field closer to the raw, noisier local result."),
        InFloat("mu", "ADMM μ", unit="", field=True, default=1e-3,
                available_in={"solver": frozenset({"aldic"})},
                description=
                "Augmented-Lagrangian penalty weight — how hard each subset's own best match is "
                "pushed toward agreeing with the global compatible field. HIGHER enforces "
                "compatibility more strictly, giving a smoother field that resists noise but can "
                "suppress genuine sharp gradients; LOWER lets subsets stay independent, closer to "
                "local DIC. ADMM converges over a wide range of μ, so this mostly changes the "
                "convergence RATE and the smoothness of the answer, not whether you get one."),
        InFloat("tol", "Tolerance", unit="", field=True, default=1e-2,
                description=
                "Convergence tolerance for the local IC-GN solve: iteration stops once the "
                "parameter step falls below this. TIGHTER (smaller) buys sub-pixel precision at "
                "more iterations per subset — and past the noise floor of your speckle it buys "
                "nothing real; LOOSER returns sooner and leaves each vector slightly "
                "under-converged. In the solver's own parameter units, not pixels."),
        InFloat("disp_smoothness", "Displacement smoothing", unit="", field=True,
                default=5e-4,
                available_in={"solver": frozenset({"aldic"})},
                description=
                "Regularization weight on the DISPLACEMENT field in the global step — not a "
                "post-hoc filter, but part of the solve. HIGHER suppresses noise and outlier "
                "vectors at the cost of flattening real localized motion; 0 leaves the global "
                "step to fit the local matches as closely as it can. Raise it when the field "
                "looks speckled with isolated bad vectors. The response is very sharp and the "
                "default sits below its onset: 0 and 5e-4 are indistinguishable on synthetic "
                "tests, while 5e-3 cuts noise-case error 3x AND blurs a 2% stretch field 50x "
                "worse. Treat 1e-3 as the floor of the useful range."),
        InFloat("strain_smoothness", "Strain smoothing", unit="", field=True, default=1e-5,
                description=
                "Regularization weight on the STRAIN field. Deliberately ~50× smaller than the "
                "displacement weight by default, because strain is a DERIVATIVE of displacement: "
                "differentiating already amplifies noise, and smoothing the derivative as hard as "
                "the field itself would erase the strain concentrations that are usually the "
                "whole point of the measurement. Raise it for a cleaner-looking strain map, "
                "knowing peak strain is being averaged down as you do."),
        InString("roi", "ROI mask layer", field=False, default="roi_mask",
                 layer_in=Domain.VOXEL,
                 description=
                 "Optional Voxel mask restricting WHERE correlation is attempted — from ROI "
                 "Mask or a Threshold. Its point is both speed and quality: subsets outside the "
                 "region of interest cost time and, over featureless background, return "
                 "meaningless vectors that then pollute any smoothing or strain derivative "
                 "computed from neighbouring points. Naming a layer that is not present leaves "
                 "the whole frame in play."),
        InString("name", "Output layer", field=False, default="dic",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point field this node writes: one row per correlation subset "
                 "centre with displacement, strain and a per-point quality value. SPARSE, on "
                 "the subset grid — use Rasterize Field for a dense per-voxel version. "
                 "Downstream nodes select it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("reference_mode", ["fixed_frame", "previous_frame"],
                default="fixed_frame", label="Reference",
                description=
                "Which image each frame's displacement is measured AGAINST — the undeformed "
                "state. It decides whether the field is total deformation since a chosen frame "
                "or the increment since the last one, and how error behaves over a long series. "
                "Ignored when an external reference Dataset is wired, which is always treated "
                "as a fixed reference.",
                choice_docs={
                    "fixed_frame":
                        "Correlate every frame against one chosen reference frame of the same "
                        "series. Displacements are TOTAL deformation from that state — directly "
                        "comparable across the series, with no accumulated error — as long as "
                        "the correlation still succeeds at the largest deformation reached.",
                    "previous_frame":
                        "Correlate each frame against the one before it (t=0 is skipped). Each "
                        "step is small, so correlation stays reliable however far the specimen "
                        "eventually deforms; errors accumulate frame by frame, and the field "
                        "reports increments rather than totals.",
                }),
           # AL-DIC = local IC-GN matches reconciled by a global FEM/ADMM step (the method
           # the node is named for). "local" runs the local step alone — everything only
           # the global step reads (ADMM iterations, μ, displacement smoothing) is gated
           # off it above, verified against al_dic's `if para.use_global_step:` block.
           Mode("solver", ["aldic", "local"], default="aldic", label="Solver",
                description=
                "Whether the independently-matched subsets are reconciled into one globally "
                "consistent field. This is the difference between the AL-DIC method the node is "
                "named for and plain local DIC; the controls only the global step reads (ADMM "
                "iterations, µ, displacement smoothing) appear only under `aldic`.",
                choice_docs={
                    "aldic":
                        "Local IC-GN matching plus the global FEM/ADMM step that reconciles "
                        "neighbouring subsets — the published AL-DIC method and the default. The "
                        "field comes out smooth and kinematically consistent, so strains derived "
                        "from it are usable, and a subset that matched badly is pulled into line "
                        "by its neighbours. Costs several alternating passes.",
                    "local":
                        "The local IC-GN step alone: every subset is solved independently and "
                        "nothing couples them. Faster, and the honest choice when you want to "
                        "see the RAW per-subset measurements — including their noise and their "
                        "outliers — rather than a regularized field. Derivatives of it are "
                        "correspondingly noisy.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset({"t", "y", "x"}),
    description="2D digital image correlation (pyALDIC: IC-GN + ADMM over an adaptive FE "
                "mesh) → a Point displacement field (subset centers, disp µm, optional "
                "strain); fixed/previous self-reference or an external reference Dataset + "
                "optional ROI mask. Solves the whole T stack in one call per (m, z). Wired "
                "to the vendored kernel; needs al-dic (dep-gated: friendly ImportError at "
                "run time until installed).")
