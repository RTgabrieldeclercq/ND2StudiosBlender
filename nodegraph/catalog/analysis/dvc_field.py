"""DVC (pyALDVC) (``analysis.dvc_field``) — 3D Augmented-Lagrangian digital volume correlation on a ref/def volume series → a Point field (subset centers with disp µm / strain / qfactor);."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

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

# Params the in-repo ALDVC port read and pyALDVC has no equivalent for, mapped to
# their old defaults. A saved graph that still carries one at its OLD DEFAULT is
# silently fine (the GUI serializes defaults); one carrying a value the user
# actually chose is REFUSED by name, because running it would quietly ignore a
# deliberate setting. See `_refuse_retired`.
_RETIRED: Dict[str, Tuple[Any, str]] = {
    "seed_levels": (3, "the seed pyramid is now `Initial guess` = pyramid plus "
                       "`Pyramid levels` (0 = automatic)"),
    "cc_thresh": (0.0, "pyALDVC has no absolute correlation floor; use "
                       "`Outlier threshold` (the normalized median test)"),
    "repair_zncc": (0.6, "pyALDVC's local pass does its own median-test rejection "
                         "and inpainting; use `Outlier threshold`"),
    "newFFTSearch": (False, "per-frame seeding is now `Initial guess` — `pyramid` "
                            "re-seeds every frame, `previous` warm-starts"),
}
_RETIRED_MODES: Dict[str, Tuple[Any, str]] = {
    "correlation": ("zncc", "the seed metric is no longer a choice; the rigid "
                            "phase-correlation pre-shift is `Global pre-shift`"),
    "dim": ("3D", "pyALDVC is 3D-only — 2D correlation is the `DIC (pyALDIC)` node "
                  "(`analysis.dic_correlate`)"),
}


def _refuse_retired(params: Dict[str, Any], modes: Dict[str, Any]) -> None:
    """Refuse a saved graph that set a param the new solver cannot honour.

    Silently dropping it would run the graph on a DIFFERENT correlation than the one
    the user configured and report it as the same node, which is the one failure mode
    a solver swap must not have. Anything still at its old default is not a choice, so
    it passes.
    """
    for key, (old_default, replacement) in _RETIRED.items():
        if key not in params:
            continue
        val = params[key]
        try:
            same = bool(np.isclose(float(val), float(old_default)))
        except (TypeError, ValueError):
            same = bool(val) == bool(old_default)
        if not same:
            raise ValueError(
                f"`{key}` = {val!r} was a parameter of the in-repo ALDVC port, which "
                f"this node replaced with the official pyALDVC (al-dvc) solver on "
                f"2026-09-25. It has no equivalent — {replacement}. Clear it to run.")
    for key, (old_default, replacement) in _RETIRED_MODES.items():
        if key in modes and str(modes[key]) != str(old_default):
            raise ValueError(
                f"mode `{key}` = {modes[key]!r} is not available: {replacement}.")


def _compute_dvc_field(ctx: EvalContext) -> Dataset:
    """3D Augmented-Lagrangian DVC (official pyALDVC / ``al-dvc``) → a **Point** structure
    on the solver's correlation node grid. One ``run_aldvc`` call per m-position covers the
    whole T stack; each frame pair yields displacement in **µm** (voxels × ``voxel_size_um``),
    a dimensionless strain tensor, and ``qfactor`` = the per-subset ZNCC.

    Resolved spec (V3, 2026-09-25 solver swap): category analysis; op ``analysis.dvc_field``;
    **Point** output; **NO DimMode — 3D only**, because pyALDVC requires an even ``winsize``
    triple ≥ 4 per axis and cannot correlate a single plane (2D is ``analysis.dic_correlate``,
    the pyALDIC sibling). ``reference_mode`` ∈ fixed_frame (ref = ``reference_frame`` of the
    same series) / previous_frame (ref = t−1, t=0 skipped). An optional ``reference`` Dataset
    input overrides self-reference — ref = that dataset at ``reference_frame`` (clamped m/z/c),
    so a separate undeformed stack becomes the reference. Footprint WHOLE_SERIES (crosses T),
    ``kernel_axes`` = {t, z, y, x}.

    ``voxel_size_um = (z_step_um, pixel_size_um, pixel_size_um)`` slowest-first is passed
    THROUGH to ``DVCPara.voxel_size`` (reversed to pyALDVC's ``(x,y,z)``), so strain carries
    the ``voxel_i/voxel_j`` cross-axis rescale that non-cubic confocal voxels require;
    displacement stays in voxels and is scaled to µm by ``_shared.dvc._dvc_rows``.

    Subset/spacing/search are VOXEL extents (unit ``px``, no optical derive — the speckle
    pattern sets them, not the objective). Each has an optional ``_z`` companion for
    anisotropic stacks (0 = same as lateral), which is upstream's documented advice for
    anisotropic voxels. Threading is numba's in-process pool (``n_threads``), not a process
    pool, so there is no spawn/import hazard to force to 1 the way the old port had.

    Kernel: :func:`nodegraph.kernels.aldvc_field.run_aldvc_series`.
    """
    from nodegraph.kernels.aldvc_field import run_aldvc_series

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
    _refuse_retired(ctx.params, modes)

    if ax.z < 2:
        raise ValueError(
            f"3D DVC needs z>1 (got z={ax.z}). pyALDVC is 3D-only: its subset must be an "
            "even cube of >=4 voxels per axis, so a single plane cannot be correlated. Use "
            "the 'DIC (pyALDIC)' node (analysis.dic_correlate) for 2D image correlation.")

    ref_mode = modes.get("reference_mode", "fixed_frame")
    ref_frame = int(ctx.params.get("reference_frame", 0))
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox: Tuple[float, float, float] = (zs, px, px)

    dvc_params = {
        # grid
        "subset_size": max(4, int(ctx.params.get("subset_size", 32))),
        "subset_size_z": max(0, int(ctx.params.get("subset_size_z", 0))),
        "subset_spacing": max(1, int(ctx.params.get("subset_spacing", 16))),
        "subset_spacing_z": max(0, int(ctx.params.get("subset_spacing_z", 0))),
        # initial guess
        "init_guess": modes.get("init_guess", "pyramid"),
        "search_radius": max(0, int(ctx.params.get("search_radius", 0))),
        "global_shift": bool(ctx.params.get("global_shift", True)),
        "init_coarse_factor": max(1, int(ctx.params.get("init_coarse_factor", 1))),
        "prefilter_sigma": max(0.0, float(ctx.params.get("prefilter_sigma", 0.0))),
        # local IC-GN
        "interp_method": modes.get("interp_method", "cubic"),
        "icgn_max_iter": max(1, int(ctx.params.get("icgn_max_iter", 100))),
        "subset_stride": max(1, int(ctx.params.get("subset_stride", 1))),
        # ADMM
        "use_global_step": bool(ctx.params.get("use_global_step", True)),
        "admm_iterations": max(1, int(ctx.params.get("admm_iterations", 4))),
        "mu": float(ctx.params.get("mu", 1e-3)),
        "beta": max(0.0, float(ctx.params.get("beta", 0.0))),
        # smoothing + strain
        "disp_smoothing": max(0.0, float(ctx.params.get("disp_smoothing", 0.0))),
        "strain_smooth": max(0.0, float(ctx.params.get("strain_smooth", 0.0))),
        "strain_method": modes.get("strain_method", "plane_fit"),
        "strain_type": modes.get("strain_type", "infinitesimal"),
        "strain_halfwidth": max(1, int(ctx.params.get("strain_halfwidth", 1))),
        # compute
        "backend": modes.get("backend", "auto"),
        "n_threads": max(0, int(ctx.params.get("n_threads", 0))),
        "tile_local": max(0, int(ctx.params.get("tile_local", 0))),
    }
    layer = ctx.layer("name")

    def _ref_frame_for(m: int) -> Tuple[Any, int, int, int]:
        """``(provider, m, t, c)`` of the volume that is this m-position's reference."""
        if ref_prov is not None:                                # external reference file
            rax = ref_prov.axes
            return (ref_prov, min(m, rax.m - 1),
                    min(max(0, ref_frame), rax.t - 1),
                    min(c, max(0, rax.c - 1)))
        return (prov, m, min(max(0, ref_frame), ax.t - 1), c)

    def _vol(p, m: int, t: int, cc: int) -> np.ndarray:
        pax = p.axes
        return p.get_region_volume(0, m, t, cc, 0, pax.z, 0, pax.y, 0, pax.x).astype(float)

    rows: List[Dict[str, np.ndarray]] = []
    # One solver call per m-position, so the bar's frame axis is m and each unit is one
    # opaque pyALDVC run whose own 0-1 progress fraction drives the sub bar.
    bar = _UnitBar(ctx, frames=max(1, ax.m), units_per_frame=1, note="correlating")

    for m in range(ax.m):
        if ref_mode == "previous_frame" and ref_prov is None:
            # Incremental: frames in series order, each referencing its predecessor.
            # t=0 has nothing to correlate against and yields no row (v1 parity).
            if ax.t < 2:
                bar.finish_unit(note=f"m={m} (skipped: needs t>1)")
                continue
            n_frames = ax.t
            out_t = list(range(1, ax.t))
            ref_indices = tuple(range(ax.t - 1))

            def get_volume(i: int, _m=m) -> np.ndarray:
                return _vol(prov, _m, i, c)
        else:
            # Fixed reference (self or external): the reference volume is prepended as
            # frame 0 and every timepoint references it, which keeps pyALDVC's DAG rule
            # (ref_indices[i] <= i) satisfied for ANY reference_frame, including one
            # later in the series than the frame being correlated.
            rp, rm, rt, rc = _ref_frame_for(m)
            n_frames = ax.t + 1
            out_t = list(range(ax.t))
            ref_indices = tuple(0 for _ in range(ax.t))

            def get_volume(i: int, _m=m, _rp=rp, _rm=rm, _rt=rt, _rc=rc) -> np.ndarray:
                if i == 0:
                    return _vol(_rp, _rm, _rt, _rc)
                return _vol(prov, _m, i - 1, c)

        shape = (ax.z, ax.y, ax.x)
        # Compare the providers' declared AXES, never a materialized volume: fetching
        # frame 0 just to read `.shape` would pull the whole reference stack here and
        # again inside LazyVolumeProvider — 1.2 GB read twice on a 1024x1024x306 scan.
        if ref_prov is not None:          # an external reference always takes this branch
            rax = ref_prov.axes
            if (rax.z, rax.y, rax.x) != shape:
                raise ValueError(
                    f"DVC reference shape {(rax.z, rax.y, rax.x)} != deformed {shape}; the "
                    "reference Dataset must match the primary's Z/Y/X")

        results = run_aldvc_series(
            get_volume, n_frames, shape,
            voxel_size_um=vox, params=dvc_params, ref_indices=ref_indices,
            compute_strain=True,
            progress_cb=lambda frac, msg, _b=bar: _b.emit(int(100 * float(frac)), None),
        )
        for t_out, res in zip(out_t, results):
            rows.append(_dvc_rows(res, vox, m=m, t=t_out, c=c, z_plane=None))
        bar.finish_unit(note=f"m={m}")

    # Provenance stamp (metadata intelligence): record HOW this field was correlated so a
    # downstream `analysis.accumulate_field` inherits the reference config instead of the
    # user re-specifying it, and refuses an already-cumulative field. An external reference
    # is always a FIXED reference (the lever governs self-reference only, V2.06), so it
    # stamps `fixed_frame` — its output is not incremental and must not be accumulated.
    eff_mode = ref_mode if ref_prov is None else "fixed_frame"
    prov_md = {
        "dvc_reference_mode": eff_mode, "dvc_reference_frame": ref_frame,
        "dvc_strain_type": dvc_params["strain_type"],
        "dvc_strain_smooth": dvc_params["strain_smooth"],
    }
    zk = "subpixel"
    if not rows:
        empty = point_table(np.zeros((0, 3)), z_kind=zk, layer=layer)
        return ds.with_structure(empty).with_metadata(**prov_md)
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    merged["id"] = np.arange(len(merged["m"]), dtype=np.int64)               # global ids
    return (ds.with_structure(StructureTable(Domain.POINT, merged, layer=layer, z_kind=zk))
            .with_metadata(**prov_md))


def _columns_dvc_field(params, modes, incoming):
    """The correlation grid this node writes, as ``_shared.dvc._dvc_rows`` builds it.

    Derived from :func:`dvc_field_columns` rather than transcribed, so the declaration and
    the flattener cannot drift on the axis-dependent ``disp_``/``strain_`` names. Always 3D
    now that the dim lever is gone. The strain block is declared unconditionally: this node
    always asks the solver for strain, and the only cost of naming a column a degenerate run
    lacks is a menu entry the compute then refuses by name — whereas omitting a real column
    would make it unpickable, which is the direction this catalog must not err in."""
    try:
        return on_layer(Domain.POINT, str((params or {}).get("name") or "dvc"),
                        list(dvc_field_columns(is_3d=True)) + ["qfactor"])
    except Exception:                        # pragma: no cover - defensive
        return ()


register_node(
    _compute_dvc_field, op_key="analysis.dvc_field", label="DVC (pyALDVC)",
    adds_columns=_columns_dvc_field,
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

        # ── subset grid ───────────────────────────────────────────────────────────────
        InInt("subset_size", "Subset size", unit="px", field=False, default=32,
              pick_kind="grid", pick_peer="subset_spacing",
              description=
              "Edge length of each correlation subset, in VOXELS — the cube of speckle matched "
              "between reference and deformed volumes. pyALDVC's own sizing rule is the one to "
              "follow: at least 4–5× your speckle/pore diameter, so a subset holds 2–3 features "
              "per axis; below that it has too little texture to lock onto and the match "
              "degrades. LARGER lowers noise and is itself a form of spatial filtering; SMALLER "
              "resolves strain gradients but assumes one affine deformation across a window "
              "that no longer has one. Rounded UP to an even number ≥4 — the solver refuses "
              "anything else. Cost grows with the cube."),
        InInt("subset_size_z", "Subset size (Z)", unit="px", field=False, default=0,
              description=
              "Subset edge along Z in voxels, for anisotropic stacks. 0 means SQUARE — use the "
              "lateral Subset size on all three axes. Confocal voxels are usually far coarser "
              "in Z than in XY, so a cubic voxel subset spans several times more physical depth "
              "than width; upstream's explicit advice for anisotropic voxels is to set the "
              "axial extent separately (e.g. 32/32/16). Set this to the number of Z SLICES that "
              "covers roughly the same physical distance as the lateral subset. Rounded up to "
              "an even number ≥4."),
        InInt("subset_spacing", "Subset spacing", unit="px", field=False, default=16,
              pick_kind="grid", pick_peer="subset_size",
              description=
              "Spacing between subset centres, in voxels — this alone sets the measurement grid "
              "density and hence how many vectors you get. SMALLER gives a denser field at "
              "cubic cost in 3D, so it is the dominant runtime knob here; LARGER is faster and "
              "can step straight over localized deformation. It does not change any single "
              "vector's accuracy — that is Subset size. Upstream's rule of thumb: half the "
              "subset for smooth fields, a quarter of it when you are chasing strain gradients."),
        InInt("subset_spacing_z", "Subset spacing (Z)", unit="px", field=False, default=0,
              description=
              "Node spacing along Z in voxels. 0 means use the lateral Subset spacing on all "
              "three axes. Worth setting on a stack with few, widely-spaced Z slices: the "
              "lateral spacing may leave only one or two nodes through the depth, and a grid "
              "that thin makes the through-plane strain terms meaningless. LOWER it to get more "
              "axial nodes out of a shallow stack."),

        # ── initial guess ─────────────────────────────────────────────────────────────
        InInt("search_radius", "Search radius", unit="px", field=False, default=0,
              description=
              "Half-width, in voxels per axis, of the NCC search that finds each subset's "
              "integer starting guess (at the coarsest pyramid level). It must bracket the "
              "largest real displacement or a fast-moving region is seeded wrongly and the "
              "refinement converges confidently to the wrong match. 0 means LEAVE IT TO THE "
              "SOLVER — pyALDVC's own default of 8, which it also expands automatically when it "
              "detects correlation peaks clipped at the search boundary. Upstream's advice is "
              "to raise this only if the run reports many clipped peaks; LARGER costs search "
              "time and admits more spurious peaks."),
        InInt("init_coarse_factor", "Coarse init factor", unit="", field=False, default=1,
              description=
              "Run the initial NCC search and a full 12-DOF solve on only every k-th node per "
              "axis (k³ fewer nodes), then interpolate BOTH displacement and gradient to every "
              "node as the starting guess for the full pass. A pure speed lever on SMOOTH "
              "fields with dense grids — upstream measures a confocal example's initial guess "
              "dropping 50→17 s for the same result. 1 is off. Raise it only when the "
              "deformation varies slowly between neighbouring nodes; on a field with sharp "
              "local features the interpolated guess lands outside the real convergence basin "
              "and you lose more in iterations than you saved. 2 is the useful value."),
        InFloat("prefilter_sigma", "Pre-smoothing σ", unit="px", field=True, default=0.0,
                description=
                "Gaussian σ (voxels) applied to EVERY volume before correlation. 0 is off. This "
                "is the recommended first move on low-SNR scans — upstream names 0.6–1.0 for "
                "SNR below about 5 — because correlation noise enters through the image "
                "gradients, which smoothing tames. The cost is real: it blurs the speckle that "
                "carries the signal, so it lowers spatial resolution and can bias displacement "
                "where the field curves sharply. Reach for it when median ZNCC is poor, not by "
                "default."),
        InBool("global_shift", "Global pre-shift", field=False, default=True,
               description=
               "Estimate one rigid whole-volume translation by phase correlation and remove it "
               "before the per-subset search. ON (the default) is what lets a modest search "
               "radius cope with a series that has drifted bodily — stage drift, a bumped dish "
               "— because the search then only has to cover the deformation, not the drift. "
               "Turn it OFF when the volume genuinely does not translate and you would rather "
               "not risk a spurious global estimate biasing every vector by the same amount."),

        # ── local IC-GN ───────────────────────────────────────────────────────────────
        InInt("icgn_max_iter", "IC-GN iterations", unit="", field=False, default=100,
              description=
              "Hard cap on inverse-compositional Gauss-Newton iterations per subset per pass. "
              "Nodes that hit it are reported with a `max_iter` status rather than silently "
              "accepted, so this is a safety valve, not a tuning knob: the solver normally "
              "converges in well under ten. LOWER it only to bound the runtime of a run you "
              "know contains hopeless subsets; the result at those nodes is then whatever the "
              "iteration reached, which the global step still has to reconcile."),
        InInt("subset_stride", "Subset stride", unit="", field=False, default=1,
              description=
              "Sample only every k-th voxel along each axis inside a subset, giving k³ fewer "
              "voxels per iteration. A large speed lever — upstream measures the local step "
              "about 5× faster at k=2 with a 32-voxel subset, for the same answer on clean "
              "data. The price is noise: a subset with k³ fewer samples carries roughly 3× the "
              "noise-induced error, while still averaging over the full span. 2 is a good "
              "choice for subsets of 32 or more with decent SNR; leave it at 1 on noisy data. "
              "Clamped so at least 5 samples per axis remain."),

        # ── ADMM ──────────────────────────────────────────────────────────────────────
        InBool("use_global_step", "Global compatibility step", field=False, default=True,
               description=
               "Whether the independently-matched subsets are reconciled into one kinematically "
               "compatible field. This IS the difference between AL-DVC and plain local DVC: "
               "with it on, a badly-matched subset is pulled into line by its neighbours and "
               "the strain derived from the field is usable; with it OFF you get the raw "
               "per-subset measurements, including their noise and outliers, and derivatives of "
               "them are correspondingly noisy. Turn it off to SEE the unregularized "
               "measurement, or as a much faster first look. ADMM iterations, μ and β are inert "
               "while it is off."),
        InInt("admm_iterations", "ADMM iterations", unit="", field=False, default=4,
              description=
              "Iterations of the outer augmented-Lagrangian loop that alternates local subset "
              "matching with the global compatibility solve. The published guidance is that 3–5 "
              "suffice — the primal and dual residuals collapse within the first three and then "
              "plateau, so more buys a full pass of compute for very little. Minimum 1, which "
              "means one local pass plus one global solve. Raise it only when the run's reported "
              "update per iteration has not settled. Only read when the global step is on."),
        InFloat("mu", "ADMM μ", unit="", field=True, default=1e-3,
                description=
                "Augmented-Lagrangian penalty weighting the constraint that each subset's own "
                "displacement equals the globally compatible field. HIGHER enforces that "
                "agreement harder; LOWER lets subsets stay independent, closer to plain local "
                "DVC. It is a PENALTY, not a smoothing filter — ADMM also carries Lagrange "
                "multipliers, which is why the method reaches accuracy without the explicit "
                "smoothness regularizer a purely global solver needs, and why it does not "
                "oversmooth the way that regularizer does. Upstream states this rarely needs "
                "changing from 1e-3. Only read when the global step is on."),
        InFloat("beta", "ADMM β", unit="", field=True, default=0.0,
                description=
                "Weight on the compatibility term of the global solve. 0 means AUTO — the "
                "solver sweeps a candidate range and picks the L-curve corner, once per "
                "reference frame, which is the documented default and almost always the right "
                "answer. Set a positive value only to pin a specific β for reproducibility or "
                "when the automatic sweep visibly bottoms out at an end of its range. HIGHER "
                "weights global compatibility over the local matches, which smooths; LOWER "
                "trusts the subsets. Only read when the global step is on."),

        # ── smoothing + strain ────────────────────────────────────────────────────────
        InFloat("disp_smoothing", "Displacement smoothing", unit="", field=True, default=0.0,
                description=
                "Gaussian σ, in NODE units (grid steps, not voxels), applied to the solved "
                "displacement field itself. 0 is off and is the right default — the "
                "displacement is what you measured, and AL-DVC has already reconciled it "
                "globally. Anything above 0 rewrites the reported displacements, so a "
                "downstream track or accumulation inherits the smoothing. Use it only when you "
                "want a visibly cleaner vector field and accept that the peaks are lowered."),
        InFloat("strain_smooth", "Strain smoothing", unit="", field=True, default=0.0,
                description=
                "Gaussian σ, in NODE units, applied when strain is computed. 0 is the "
                "principled default here, unlike in plain local DVC: AL-DVC already produces a "
                "kinematically compatible field, so strain is an intrinsic output rather than "
                "something a filter has to rescue. Upstream lists this as one of two fixes for "
                "SPECKLED strain maps (the other is a wider plane-fit window, which costs less "
                "peak). HIGHER systematically lowers peak strain by averaging exactly the "
                "concentrations you are trying to measure — any reported maximum is a function "
                "of this value."),
        InInt("strain_halfwidth", "Strain window half-width", unit="", field=False, default=1,
              description=
              "Half-width, in NODES, of the local window the plane-fit strain method "
              "differentiates over: 1 means a 3×3×3 node neighbourhood. WIDER is upstream's "
              "first suggestion for a speckled strain map — it fits the plane over more nodes, "
              "so noise averages down — at the cost of smoothing genuine strain gradients and "
              "of flagging more nodes near the volume edge as having an incomplete window. Only "
              "read when the strain method is `plane_fit`."),

        # ── compute ───────────────────────────────────────────────────────────────────
        InInt("n_threads", "Threads", unit="", field=False, default=0,
              description=
              "Numba worker threads for the CPU local solver. 0 means ALL CORES, which is the "
              "default and normally what you want. Lower it to leave headroom when several "
              "nodes pull in parallel or the machine is doing other work — correlation is the "
              "dominant cost in any graph that has it, so an unbounded thread count here can "
              "starve everything else. Inert on the CUDA backend."),
        InInt("tile_local", "Local tile size", unit="px", field=False, default=0,
              description=
              "Solve the local step in blocks of nodes against a box of the volume this many "
              "voxels on an edge, instead of holding the whole scan at once. 0 is OFF (one "
              "whole-volume box) and is right until memory is the problem: the pipeline "
              "otherwise holds the reference, its three gradient volumes and the deformed frame "
              "in float32, about 22 bytes per voxel. Setting a tile bounds the gradients, the "
              "mask and — on the GPU — the whole upload by the box rather than the scan, at the "
              "cost of recomputing the overlap halos. Reach for it when a large stack fails to "
              "allocate; 256 is a reasonable first try."),

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
    modes=[
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
                     "series (upstream's `accumulative` schedule). Displacements are TOTAL "
                     "deformation from that state, directly comparable across the whole series "
                     "and free of accumulation drift. The solver builds that reference's bundle "
                     "once and reuses it for every frame, so this is also the cheaper schedule. "
                     "It requires correlation to still succeed at large deformation.",
                 "previous_frame":
                     "Correlate each frame against t−1 (t=0 is skipped) — upstream's "
                     "`incremental` schedule. Every step is a small increment, so correlation "
                     "stays easy under large total deformation, at the cost of accumulating "
                     "error frame by frame and of reporting increments rather than totals. "
                     "Compose them with Accumulate DVC Field. Every frame is a fresh reference, "
                     "so this costs a reference bundle per frame.",
             }),
        Mode("init_guess", ["pyramid", "ncc", "zero", "previous"],
             default="pyramid", label="Initial guess",
             description=
             "How each subset gets the integer starting displacement that the sub-voxel "
             "refinement then polishes. This matters more than it looks: the method's "
             "convergence assumes the guess lands inside the subset's convergence basin, and "
             "when it does not the local problem DIVERGES rather than merely losing accuracy. "
             "It costs time but does not set the final precision, which comes from the "
             "refinement either way.",
             choice_docs={
                 "pyramid":
                     "Coarse-to-fine NCC over an image pyramid: correlate at reduced "
                     "resolution, then refine level by level. Robust to LARGE motion because "
                     "the coarsest level brackets a displacement many times the search radius, "
                     "and the default for that reason. Costs the pyramid passes, and needs the "
                     "speckle to survive downsampling.",
                 "ncc":
                     "A single full-resolution NCC search of ±Search radius. Cheaper than the "
                     "pyramid and entirely predictable, but it only finds motion that fits "
                     "inside the radius you set. Use it when you know the displacement is small "
                     "and want to spend nothing on seeding.",
                 "zero":
                     "Start every subset from zero displacement. The fastest option and "
                     "correct only for SMALL motion — under about half a subset — where the "
                     "refinement can reach the match unaided. On anything larger it seeds every "
                     "subset into the wrong basin at once.",
                 "previous":
                     "Reuse the previous frame's solved field as this frame's guess, falling "
                     "back to the pyramid on the first frame and after a reference switch. "
                     "Upstream's recommendation for MANY FRAMES WITH SMALL INCREMENTS: it skips "
                     "the search entirely on all but the first frame. It also propagates an "
                     "error down the series, so switch back to `pyramid` when one bad frame "
                     "appears to have corrupted everything after it.",
             }),
        Mode("interp_method", ["cubic", "bspline", "linear"],
             default="cubic", label="Interpolation",
             description=
             "How the deformed volume is sampled at the non-integer positions the warped "
             "subset lands on. This is the numerical floor on sub-voxel accuracy: a "
             "coarser interpolant leaves a periodic bias that peaks near half-voxel "
             "displacements, which shows up as a ripple through an otherwise smooth field.",
             choice_docs={
                 "cubic":
                     "Keys cubic convolution — the same interpolant as MATLAB `ba_interp3`, so "
                     "this is the setting that reproduces the reference implementation. The "
                     "default and the right choice unless you have a specific reason.",
                 "bspline":
                     "Cubic B-spline (prefiltered). Smoother than Keys and slightly more "
                     "accurate on well-sampled, low-noise data, at a higher cost per sample "
                     "and with more spreading of noise between neighbouring voxels.",
                 "linear":
                     "Trilinear. Fastest and lowest memory, and clearly the least accurate — "
                     "it carries a known displacement bias toward integer voxel positions. Use "
                     "it for a quick look or when memory is critical, not for reported numbers.",
             }),
        Mode("strain_method", ["plane_fit", "fem", "fd", "direct"],
             default="plane_fit", label="Strain method",
             description=
             "How the displacement gradient is extracted from the node field before it is "
             "turned into a strain measure. All four differentiate the SAME displacements; "
             "they differ in how much they average, and therefore in how much strain noise "
             "they let through against how much real gradient they blunt.",
             choice_docs={
                 "plane_fit":
                     "Least-squares fit of a plane to the displacements in a local node window "
                     "(Strain window half-width), and take its slope. Averages noise over the "
                     "window, which is why it is the default; it correspondingly smooths real "
                     "strain gradients and flags edge nodes whose window is incomplete.",
                 "fem":
                     "Differentiate the finite-element shape functions of the hexahedral node "
                     "mesh. Consistent with the global compatibility step's own discretisation, "
                     "so it is the natural choice when you care that strain and the solved "
                     "field come from one formulation.",
                 "fd":
                     "Plain finite differences between neighbouring nodes. The least smoothing "
                     "and the most local, so it resolves a sharp gradient best — and passes "
                     "displacement noise straight through, amplified by the node spacing.",
                 "direct":
                     "Use the per-node displacement gradient the IC-GN solve already fitted as "
                     "part of each subset's 12-DOF warp, with no differentiation between nodes "
                     "at all. Genuinely independent of the node spacing and the only option "
                     "that reports a subset's OWN measured stretch; it is also noisier, because "
                     "nothing averages across neighbours.",
             }),
        Mode("strain_type", ["infinitesimal", "green_lagrange", "euler_almansi", "hencky"],
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
                 "green_lagrange":
                     "The finite-strain measure in the REFERENCE configuration — the "
                     "standard choice for solid mechanics at large deformation, and what "
                     "most constitutive models expect. Exactly zero under rigid rotation, "
                     "unlike the infinitesimal measure, and it reads larger than engineering "
                     "strain in tension.",
                 "euler_almansi":
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
             }),
        Mode("backend", ["auto", "numba", "numpy", "cuda"],
             default="auto", label="Backend",
             description=
             "Which compute path runs the local subset solve — the dominant cost of a DVC "
             "run. It changes speed and hardware requirements only; every backend solves "
             "the same problem to the same tolerance.",
             choice_docs={
                 "auto":
                     "Use the NVIDIA GPU when `numba-cuda` and a usable CUDA device are both "
                     "present, and fall back to the numba CPU kernels otherwise. The default, "
                     "and safe on any machine: a box with no GPU simply runs on the CPU.",
                 "numba":
                     "Force the JIT-compiled CPU kernels, threaded across `Threads`. Choose it "
                     "to keep a GPU free for something else, or to get reproducible timings "
                     "that do not depend on what else is using the card. Note the first run of "
                     "a session pays a one-off JIT compile.",
                 "numpy":
                     "Plain NumPy, no JIT. Much slower and present as a fallback and a "
                     "reference: it is what to run when you suspect a numba compilation or "
                     "threading problem and want to check the answer without it.",
                 "cuda":
                     "INSIST on the GPU and fail loudly if it is unusable, rather than quietly "
                     "falling back. Use it in a batch run where a silent drop to the CPU would "
                     "turn a ten-minute job into an overnight one and you would rather be told.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES,
    kernel_axes=frozenset({"t", "z", "y", "x"}),
    description="3D Augmented-Lagrangian digital volume correlation (official pyALDVC: "
                "pyramid-NCC seed + 12-DOF IC-GN + global compatibility/ADMM) → a Point "
                "displacement + strain field on the subset grid (disp µm, dimensionless "
                "strain, per-subset ZNCC). Fixed-frame or previous-frame self-reference, or "
                "an external reference Dataset. Solves the whole T stack in one call per m. "
                "3D ONLY — 2D is the DIC (pyALDIC) node. Needs al-dvc (dep-gated: friendly "
                "error at run time until installed).")
