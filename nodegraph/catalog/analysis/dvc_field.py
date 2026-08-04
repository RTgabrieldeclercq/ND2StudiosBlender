"""DVC (ALDVC) (``analysis.dvc_field``) — DVC/ALDVC displacement + strain field on a ref/def volume pair → a Point field (subset centers with disp µm / strain / qfactor);."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, Tuple

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
from nodegraph.catalog._shared.dvc import _dvc_rows
from nodegraph.catalog._shared.progress import _UnitBar

def _compute_dvc_field(ctx: EvalContext) -> Dataset:
    """DVC / ALDVC displacement + strain field (ported v1 ``aldvc_field`` kernel) → a
    **Point** structure on the correlation subset grid. Loops m/t on a single reference
    channel; each ref/def pair (2D plane or 3D volume) runs the Augmented-Lagrangian
    IC-GN+ADMM solver. Displacements are stored in **µm** (voxels × ``voxel_size_um``);
    strain is dimensionless; ``qfactor`` is the per-subset ZNCC confidence.

    Resolved spec (V2.06): category analysis; op ``analysis.dvc_field``; **Point** output;
    DimMode lever (2D per-plane / 3D per-volume). ``reference_mode`` ∈ fixed_frame (ref =
    ``reference_frame`` of the same series) / previous_frame (ref = t−1, t=0 skipped). An
    optional ``reference`` Dataset input overrides self-reference — ref = the external
    dataset at ``reference_frame`` (clamped m/z/c), so a second file (a separate undeformed
    stack) becomes the reference. Footprint WHOLE_SERIES (crosses T), ``kernel_axes`` per
    dim (adds ``t``). ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` slowest-
    first (2-tuple in 2D); it drives the displacement→µm scale + anisotropic strain rescale.
    subset/spacing/search are voxel extents (unit ``px``, no optical derive — set by the
    speckle pattern). ``n_workers`` is forced to 1 (embedding-safe: no spawn/import hazard,
    kernel gotcha 9). Kernel: :func:`nodegraph.kernels.aldvc_field.run_aldvc`."""
    from nodegraph.kernels.aldvc_field import run_aldvc
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DVC needs an image provider on its (deformed) input Dataset")
    ax = prov.axes
    ref_ds = ctx.input("reference")
    ref_prov = ref_ds.image if ref_ds is not None else None
    if ref_ds is not None and ref_prov is None:
        raise ValueError("the DVC 'reference' input Dataset has no image provider")
    modes = ctx.params.get("__modes__", {})
    is_3d = modes.get("dim") == "3D"
    if is_3d and ax.z < 2:
        raise ValueError("3D DVC needs z>1; use 2D mode for a single-plane series")
    ref_mode = modes.get("reference_mode", "fixed_frame")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox: Tuple[float, ...] = (zs, px, px)
    else:
        vox = (px, px)
    dvc_params = {
        "subset_size": max(4, int(ctx.params.get("subset_size", 16))),
        "subset_spacing": max(1, int(ctx.params.get("subset_spacing", 10))),
        "search_radius": max(0, int(ctx.params.get("search_radius", 0))),
        "seed_levels": max(1, int(ctx.params.get("seed_levels", 3))),
        "correlation": modes.get("correlation", "zncc"),
        "mu": float(ctx.params.get("mu", 1e-3)),
        "admm_iterations": max(0, int(ctx.params.get("admm_iterations", 4))),
        "strain_type": modes.get("strain_type", "infinitesimal"),
        "strain_smooth": max(0.0, float(ctx.params.get("strain_smooth", 0.0))),
        # Off by default, matching the reference run (main_ALDVC's cumulative branch
        # sets qDICOrNot=0): an absolute correlation floor applied to every subset
        # replaces measurements with interpolation. `repair_zncc` is the preferred
        # guard — it only rejects what neighbour propagation could not rescue.
        "cc_thresh": float(ctx.params.get("cc_thresh", 0.0)),
        "repair_zncc": float(ctx.params.get("repair_zncc", 0.6)),
        "n_workers": 1,          # embedding-safe (kernel gotcha 9: spawn/import hazard)
        "use_gpu": False,
    }
    layer = ctx.layer("name")
    # Cross-frame warm-start (ALDVC series workflow): seed each frame's IC-GN from the
    # previous frame's field (u0_seed, skipping the FFT search) — faithful to FranckLab's
    # series correlation + more robust for large motion. `newFFTSearch` forces a fresh
    # FFT seed every frame; the first computed frame of a series always FFT-seeds. The
    # warm-start applies in BOTH reference modes (v1 parity, pipelines_page._run_series_all_m).
    newfft = bool(ctx.params.get("newFFTSearch", False))

    def _resolve_ref(m: int, t: int):
        """(ref_provider, ref_m, ref_t) for this def frame, or None to skip it."""
        if ref_prov is not None:                              # external reference file
            rax = ref_prov.axes
            return (ref_prov, min(m, rax.m - 1),
                    min(max(0, ref_frame), rax.t - 1))
        if ref_mode == "previous_frame":
            return None if t == 0 else (prov, m, t - 1)        # no increment into t0
        return prov, m, min(max(0, ref_frame), ax.t - 1)       # fixed_frame

    def _match(ref_arr: np.ndarray, def_arr: np.ndarray) -> None:
        if ref_arr.shape != def_arr.shape:
            raise ValueError(
                f"DVC reference shape {ref_arr.shape} != deformed {def_arr.shape}; the "
                "reference Dataset must match the primary's Y/X (and Z in 3D)")

    def _seeded(ref_arr: np.ndarray, def_arr: np.ndarray, u0):
        """Run the kernel warm-started from ``u0`` (previous frame's ``(ndim,*grid)``
        field) unless ``u0`` is None or ``newFFTSearch`` — then FFT-seed. Returns
        ``(res, next_u0)`` where ``next_u0`` seeds the following frame."""
        warm = u0 is not None and not newfft
        res = run_aldvc(ref_arr, def_arr, voxel_size_um=vox, params=dvc_params,
                        u0_seed=(u0 if warm else None), use_fft_seed=not warm,
                        progress_cb=bar.emit)
        return res, np.moveaxis(np.asarray(res.displacement_field), -1, 0)

    rows: list = []
    # per-node progress: correlation is the dominant cost in any graph that has it and it
    # is EAGER (one solver call per unit), so the unit count is a real denominator. It is
    # also a COARSE one — a single ALDVC solve runs for tens of seconds — so the kernel's
    # own 0-100 callback drives the sub bar inside each unit and the frame bar steps once
    # per timepoint (V2.17).
    upf = 1 if is_3d else ax.z             # units (solves) per frame
    bar = _UnitBar(ctx, frames=ax.t, units_per_frame=upf, note="correlating")
    for m in range(ax.m):
        prev_u: Dict[int, Any] = {}      # z-plane (or -1 for the whole 3D volume) → u0
        for t in range(ax.t):
            resolved = _resolve_ref(m, t)
            if resolved is None:
                bar.skip(upf, "correlating")            # skipped t0 still advances the bar
                continue
            rprov, rm, rt = resolved
            rax = rprov.axes
            rc = min(c, max(0, rax.c - 1))
            if is_3d:
                dvol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y,
                                              0, ax.x).astype(float)
                rvol = rprov.get_region_volume(0, rm, rt, rc, 0, rax.z, 0, rax.y,
                                               0, rax.x).astype(float)
                _match(rvol, dvol)
                res, prev_u[-1] = _seeded(rvol, dvol, prev_u.get(-1))
                rows.append(_dvc_rows(res, vox, m=m, t=t, c=c, z_plane=None))
                bar.finish_unit(note=f"volume t={t}")
            else:
                for z in range(ax.z):
                    rz = min(z, max(0, rax.z - 1))
                    dpl = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x).astype(float)
                    rpl = rprov.get_region(0, rm, rt, rz, rc, 0, rax.y,
                                           0, rax.x).astype(float)
                    _match(rpl, dpl)
                    res, prev_u[z] = _seeded(rpl, dpl, prev_u.get(z))
                    rows.append(_dvc_rows(res, vox, m=m, t=t, c=c, z_plane=z))
                    bar.finish_unit(note=f"t={t} z={z}")

    # Provenance stamp (metadata intelligence): record HOW this field was correlated so a
    # downstream `analysis.accumulate_field` inherits the reference config instead of the
    # user re-specifying it, and refuses an already-cumulative field. An external reference
    # is always a FIXED reference (the lever governs self-reference only, V2.06), so it
    # stamps `fixed_frame` — its output is not incremental and must not be accumulated.
    eff_mode = ref_mode if ref_prov is None else "fixed_frame"
    # DVC-specific provenance (§7b). Dimensionality is NOT stamped here — it flows generically
    # as the Point table's z_kind (preserved by `with_structure` → `__struct_zkind__`), which
    # `accumulate_field` / `rasterize_field` inherit via `ds.structure_zkind`.
    prov_md = {
        "dvc_reference_mode": eff_mode, "dvc_reference_frame": ref_frame,
        "dvc_strain_type": dvc_params["strain_type"],
        "dvc_strain_smooth": dvc_params["strain_smooth"],
    }
    zk = "subpixel" if is_3d else "plane_index"
    if not rows:
        empty = point_table(np.zeros((0, 3 if is_3d else 2)), z_kind=zk, layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(StructureTable(Domain.POINT, merged, layer=layer, z_kind=zk))
            .with_metadata(**prov_md))
register_node(
    _compute_dvc_field, op_key="analysis.dvc_field", label="DVC (ALDVC)",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InDataset("reference", label="Reference"),
        InInt("channel", "Channel", unit="", field=False, default=0,
              pick_kind="channel",
              description=
              "Which channel the correlation runs on (0-based). Displacement is measured from "
              "image texture, so pick the channel with the most stable, high-contrast speckle — "
              "typically beads or a fiducial marker, not a signalling reporter whose intensity "
              "changes for biological reasons. A channel whose appearance changes between "
              "frames violates the brightness assumption correlation depends on and produces "
              "confident but wrong vectors."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame",
              description=
              "Which timepoint is the undeformed reference (0-based). Only read when the "
              "reference mode is `fixed_frame`, where every frame is correlated against this "
              "one, giving displacements that are cumulative and directly comparable — but "
              "which fail once deformation grows too large for correlation to track. The "
              "`previous_frame` mode ignores this and measures small increments instead, which "
              "you then compose with Accumulate DVC Field."),
        # AL-DVC solver parameters (Bar-Kochba / Yang / Landauer et al.;
        # github.com/FranckLab/ALDVC, Exp. Mech. 10.1007/s11340-020-00607-3), cross-checked
        # against the vendored kernel's own table in `kernels/aldvc_field.md` §4.
        InInt("subset_size", "Subset size", unit="px", field=False, default=16,
              pick_kind="grid", pick_peer="subset_spacing",
              description=
              "Edge length of each correlation subset, in VOXELS — the cube of speckle matched "
              "between reference and deformed volumes. The core resolution/robustness "
              "trade-off, and the paper measures it: RMS error RISES as the subset shrinks "
              "(0.0034 vx at 30³ vs 0.0067 at 10³ for the same field), because a larger subset "
              "is itself a form of spatial filtering. But accuracy is not the whole story — "
              "fidelity DROPS where strain varies rapidly, since a big subset assumes one "
              "affine deformation across its whole volume. So: large where deformation is "
              "smooth, small only where you need to resolve a gradient. Range 4–128, even; cost "
              "grows with the cube."),
        InInt("subset_spacing", "Subset spacing", unit="px", field=False, default=10,
              pick_kind="grid", pick_peer="subset_size",
              description=
              "Spacing between subset centres, in voxels — this alone sets the measurement grid "
              "density and hence how many vectors you get. SMALLER gives a denser field at "
              "cubic cost in 3D, so it is the dominant runtime knob here; LARGER is faster and "
              "can step straight over localized deformation. It does not change any single "
              "vector's accuracy — that is Subset size — and with the default it is smaller "
              "than the subset, so subsets deliberately overlap."),
        InInt("search_radius", "Search radius", unit="px", field=False, default=0,
              description=
              "How far, in voxels per axis, the initial integer FFT search looks for each "
              "subset's match. It must bracket the largest real displacement between frames or "
              "fast-moving regions are seeded wrongly and the refinement converges to the wrong "
              "answer. 0 means AUTO — max(4, subset size) — which is the sensible default; set "
              "it explicitly only when motion exceeds that. LARGER costs search time and admits "
              "more chances of a spurious correlation peak."),
        InInt("seed_levels", "Seed levels", unit="", field=False, default=3,
              description=
              "Levels in the coarse-to-fine pyramid used to find the integer starting guess. "
              "Each level halves resolution, so MORE levels bracket progressively larger motion "
              "— raise it when displacements are big relative to the subset. 1 means "
              "single-scale, which only works for small motion. Too many and the coarsest level "
              "has too little structure left to correlate. This matters more than it looks: the "
              "method's convergence proof assumes the initial guess lands in the local subset's "
              "convergence basin, and when it does not, the local subproblem DIVERGES rather "
              "than merely losing accuracy. A seed that brackets the motion is the difference "
              "between a result and a failure."),
        InInt("admm_iterations", "ADMM iterations", unit="", field=False, default=4,
              description=
              "Iterations of the outer augmented-Lagrangian loop that reconciles the independent "
              "per-subset matches with one globally compatible displacement field — the step "
              "that makes this AL-DVC rather than conventional DVC. The paper's own guidance is "
              "that **3–5 iterations suffice**: its primal and dual residuals both collapse "
              "within the first three and then plateau, so pushing this higher buys a full pass "
              "of compute for almost nothing. 0 is meaningful and cheap: pass 0 only, i.e. plain "
              "local DVC plus a single global solve. Some individual subsets converge poorly "
              "within an iteration; the global step absorbs a small fraction of those, and their "
              "number falls as iterations proceed."),
        InFloat("mu", "ADMM μ", unit="", field=True, default=1e-3,
                description=
                "Augmented-Lagrangian penalty weight on the constraint that each subset's own "
                "displacement equals the globally compatible field. HIGHER enforces that "
                "agreement harder; LOWER lets subsets stay independent, closer to plain local "
                "DVC. It is a PENALTY, not a smoothing filter — ADMM also carries Lagrange "
                "multipliers, which is why the method reaches accuracy without the explicit "
                "smoothness regularizer a global solver needs, and why it does not oversmooth "
                "the way that regularizer does. The paper scales it to the problem rather than "
                "fixing it absolutely (order 1e-3 to 1e-1 times the local Hessian diagonal), so "
                "treat the default as a starting order of magnitude. Convergence holds across a "
                "broad range, so this changes the rate and the stiffness of the coupling far "
                "more than whether you get an answer. Practical range 1e-6 to 1."),
        InFloat("cc_thresh", "Correlation threshold", unit="", field=True, default=0.0,
                description=
                "Minimum correlation confidence for a subset's vector to be trusted, in [-1,1]. "
                "Anything below is DISCARDED and inpainted from its neighbours — so raising it "
                "does not simply remove points, it replaces measurements with interpolation. "
                "**0 (off) is the default and matches the reference implementation**, which "
                "disables correlation-based rejection entirely on a cumulative solve: an "
                "absolute floor is applied before anything has tried to rescue a poor subset, "
                "and at 0.5 it discarded up to 40% of a perfectly valid large-strain field. "
                "Use 'Repair threshold' instead — it rejects only what neighbour propagation "
                "could not fix. Raise this one only to be deliberately stricter than that, "
                "knowing that HIGHER quietly turns more of the field into smooth interpolation "
                "that still looks like data."),
        InFloat("repair_zncc", "Repair threshold", unit="", field=True, default=0.6,
                description=
                "Correlation below which a subset is RETRIED from a converged neighbour's full "
                "affine warp before being judged, in [-1,1]. The integer starting guess is a "
                "translation-only search, so under a large rotation or stretch the reference "
                "subset is itself rotated relative to its match and that search degrades — a "
                "20–25° in-plane rotation leaves SEVERAL VOXELS of error with this off, versus "
                "about a thousandth of a voxel with it on. Retrying from a neighbour that "
                "already converged carries the affine part of the deformation into the guess, "
                "and reliability spreads outward from wherever correlation was good. Subsets "
                "still below the threshold afterwards are discarded and interpolated — by then "
                "that verdict is earned rather than assumed. HIGHER retries and then rejects "
                "more; LOWER trusts marginal matches. -1 disables the repair pass entirely, "
                "which costs nothing on a well-correlated field and is only worth doing to "
                "reproduce an older result."),
        InFloat("strain_smooth", "Strain smoothing", unit="", field=True, default=0.0,
                description=
                "Gaussian σ applied to the displacement field BEFORE it is differentiated into "
                "strain, in voxels. **0 (off) is the principled default here**, unlike in plain "
                "local DVC: AL-DVC already produces a kinematically compatible field, so strain "
                "comes out as an intrinsic output rather than something you have to rescue with "
                "a filter — the paper reports strain RMS error around 1e-3 with no strain "
                "filtering at all, against 4e-2 for the local method. Reach for this only if "
                "the strain map is still unusable, knowing that HIGHER systematically lowers peak "
                "strain by averaging exactly the concentrations you are trying to measure. Any "
                "reported maximum is a function of this value."),
        InBool("newFFTSearch", "Fresh FFT seed each frame", field=False, default=False,
               description=
               "How each frame gets its starting guess. OFF (the default) warm-starts from the "
               "previous frame's solution, which is faster and more stable for smooth, "
               "progressive deformation — but lets an error propagate down the series. ON runs a "
               "fresh integer FFT search on every frame, costing time but making each frame "
               "independent, which is what you want after a jump, a reference switch, or when "
               "one bad frame appears to have corrupted everything after it. A series-level "
               "control: the solver itself never sees it."),
        InString("name", "Output layer", field=False, default="dvc",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point field this node writes: one row per correlation subset "
                 "centre carrying displacement, strain and a per-point correlation quality. It "
                 "is a SPARSE field on the subset grid, not a per-voxel one — use Rasterize "
                 "Field to interpolate it onto voxels, or Accumulate DVC Field to compose an "
                 "increment series. Downstream nodes select it by this name."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("reference_mode", ["fixed_frame", "previous_frame"],
                default="fixed_frame", label="Reference",
                description=
                "Which volume each frame's displacement is measured AGAINST — the undeformed "
                "state. This is what decides whether the field you get is total deformation "
                "since a chosen frame or the increment since the last one, and therefore how "
                "the errors behave over a long series.",
                choice_docs={
                    "fixed_frame":
                        "Correlate every frame against one chosen reference frame of the same "
                        "series. Displacements are TOTAL deformation from that state, directly "
                        "comparable across the whole series and free of accumulation. It "
                        "requires the correlation to still succeed at large deformation, which "
                        "is where the subset size and repair threshold start to matter.",
                    "previous_frame":
                        "Correlate each frame against t−1 (t=0 is skipped). Every step is a "
                        "small increment, so correlation stays easy even under large total "
                        "deformation — at the cost of accumulating error frame by frame, and of "
                        "reporting rates rather than totals. Use it when the specimen deforms "
                        "too far for a fixed reference to track.",
                }),
           Mode("correlation", ["zncc", "phase"], default="zncc", label="Correlation",
                description=
                "Which similarity metric seeds each subset's search — the initial "
                "translation-only guess the iterative solver refines. It affects robustness to "
                "brightness and contrast changes, not the final sub-voxel precision, which "
                "comes from the refinement stage either way.",
                choice_docs={
                    "zncc":
                        "FFT normalized cross-correlation. Invariant to brightness and contrast "
                        "change, so a bleaching series or an exposure change does not bias the "
                        "seed. The default and the robust choice; it is also what the reported "
                        "per-subset quality factor is measured with.",
                    "phase":
                        "Phase cross-correlation, which uses only the Fourier phase. Sharper "
                        "peak and cheaper on large subsets, and it copes well with a smooth "
                        "illumination gradient — but it is more easily fooled by periodic "
                        "texture and by noise in low-contrast volumes.",
                }),
           Mode("strain_type", ["infinitesimal", "green-lagrange", "almansi", "hencky"],
                default="infinitesimal", label="Strain",
                description=
                "Which strain MEASURE is derived from the displacement gradient. All four are "
                "computed from the same measured displacements and agree to within a fraction "
                "of a percent while strains are small; they diverge as deformation grows, so "
                "for anything beyond a few percent the choice must be stated in the methods "
                "and matched to whatever you compare against.",
                choice_docs={
                    "infinitesimal":
                        "The linearized (engineering) strain, the symmetric part of the "
                        "displacement gradient. Correct for SMALL deformation and by far the "
                        "easiest to interpret; it systematically misreports large strain and is "
                        "not invariant to rigid rotation, so a rotating sample picks up "
                        "spurious strain. The default.",
                    "green-lagrange":
                        "The finite-strain measure in the REFERENCE configuration — the "
                        "standard choice for solid mechanics at large deformation, and what "
                        "most constitutive models expect. Exactly zero under rigid rotation, "
                        "unlike the infinitesimal measure, and it reads larger than engineering "
                        "strain in tension.",
                    "almansi":
                        "The Eulerian counterpart: finite strain referred to the DEFORMED "
                        "configuration. Use it when the quantity should be expressed per "
                        "current geometry — flow-like problems, or comparison with measurements "
                        "made on the deformed state. Reads smaller than Green–Lagrange in "
                        "tension.",
                    "hencky":
                        "True/logarithmic strain, the log of the stretch. Additive across "
                        "successive deformation steps, which makes it the natural measure for "
                        "incremental loading and for large strains where engineering strain "
                        "stops being meaningful; it is also the most expensive of the four to "
                        "compute.",
                })],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes={"2D": frozenset({"t", "y", "x"}), "3D": frozenset({"t", "z", "y", "x"})},
    description="DVC/ALDVC displacement + strain field on a ref/def volume pair → a Point "
                "field (subset centers with disp µm / strain / qfactor); fixed-frame or "
                "previous-frame self-reference, or an optional external reference Dataset; "
                "2D per-plane vs 3D volume (ported v1 kernel).")
