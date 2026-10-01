"""Track Field (``analysis.track_field``) — SerialTrack's post-processing: a tracked object field → per-particle displacement (µm), velocity (µm/s), deformation gradient / strain and optional linear-elastic stress (Pa), as a Point field ready for transform.rasterize_field."""

from __future__ import annotations

import numpy as np

from typing import Dict, List, Tuple

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
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import member_layer, on_layer
from nodegraph.catalog._shared.objects import (
    _InFrameInterval,
    _frame_interval_s,
    _object_table,
    _object_velocity_nd,
)

# ── Track Field — the half of SerialTrack that lives downstream of the track ids ──
#
# `track.objects` (and `track.link`) answer the correspondence question and stop. Every
# number a particle-tracking experiment is actually for — the displacement field, its
# gradient, the strain, the stress — is derived from that answer and needs nothing else
# from the tracker. So this is a separate node reading a `track_id` column, not a mode on
# the tracker: it works on ANY tracked field, whichever linker produced the ids, and
# re-running it with a different strain measure or a different modulus does not re-run the
# tracking (which, for `serialtrack`, is the expensive half by two orders of magnitude).
#
# The output is a Point field in the SAME schema `analysis.dvc_field` / `analysis.piv` /
# `analysis.dic_correlate` emit (`_shared.dvc._dvc_rows`), which is what makes
# `transform.rasterize_field` turn it into full-resolution Voxel maps with no adapter —
# the "maps" at the end of the pipeline. The maths is in `nodegraph.kernels.track_field`.

#: Axis names by dimensionality, slowest-first — the Dataset's own order.
_AXES: Dict[bool, Tuple[str, ...]] = {True: ("z", "y", "x"), False: ("y", "x")}


def _displacement(track: np.ndarray, tt: np.ndarray, coords: np.ndarray,
                  reference: str) -> np.ndarray:
    """Per-row displacement ``(N, D)`` against the track's own reference observation.

    ``first_frame`` differences every row against the track's FIRST observation — the
    cumulative, Lagrangian field, and the one a traction/strain measurement wants, since
    strain is only meaningful against an undeformed state. ``previous_frame`` differences
    against the immediately preceding observation — the per-interval increment.

    A row with no predecessor is NaN, and the caller drops it: under ``first_frame`` that
    is nothing (a track's first row differences against itself and is exactly zero),
    under ``previous_frame`` it is each track's first row. Untracked rows
    (``track_id <= 0``) are NaN throughout, the same background convention
    :func:`_object_velocity_nd` uses."""
    n, d = coords.shape
    out = np.full((n, d), np.nan, dtype=float)
    live = np.flatnonzero(np.asarray(track) > 0)
    if live.size == 0:
        return out
    order = live[np.lexsort((tt[live], track[live]))]      # by track, then time
    tid = np.asarray(track)[order]
    edge = np.concatenate(([True], tid[1:] != tid[:-1]))   # first row of each track
    pos = coords[order]
    if reference == "previous_frame":
        prev = np.vstack([pos[:1], pos[:-1]])
        out[order] = pos - prev
        out[order[edge]] = np.nan                          # no predecessor to difference
    else:                                                  # first_frame (cumulative)
        # The first row of each run, broadcast forward over that run.
        starts = np.flatnonzero(edge)
        run_of = np.repeat(np.arange(starts.size), np.diff(np.append(starts, len(order))))
        out[order] = pos - pos[starts][run_of]
    return out


def _compute_track_field(ctx: EvalContext) -> Dataset:
    """A tracked object field → SerialTrack's post-processing outputs, as a Point field:
    displacement (µm), velocity (µm/s), deformation gradient / strain, and — with a
    material model stated — isotropic linear-elastic Cauchy stress (Pa).

    **Resolved spec (§0, 2026-09-17).**

    *Kind* analysis; reads a Label **or** Point structure layer carrying a ``track_id``
    column (the ``target`` mode, mirroring ``track.objects`` / ``analysis.object_metrics``)
    and adds the Point domain. It does not touch the image at all — ``kernel_axes`` is
    empty — so its footprint is ``WHOLE_SERIES`` purely because a displacement needs every
    timepoint of a track at once.

    *Why a separate node.* Correspondence and kinematics are different questions. Keeping
    them apart means this runs on any tracker's output (``track.objects``' five linkers and
    ``track.link``'s two), and means changing the strain measure or the modulus does not
    re-run the tracking — which for ``serialtrack`` is 1-2 orders of magnitude of the cost.

    *Dimensionality is INHERITED from the member layer's ``z_kind`` (§7b), never a lever.*
    ``subpixel`` (a 3-D detection, or 3-D ``analysis.label``) means z is a measured depth:
    it becomes a third coordinate, the gradient is a full 3x3, and the strain gauge searches
    a sphere. ``plane_index`` means z is a plane number, so each ``(m,t,c,z)`` plane is its
    own 2-D field and nothing crosses a plane. A lever here could disagree with how the
    field was produced — the exact failure ``transform.rasterize_field`` avoids the same
    way, and the reason the output's own ``z_kind`` is stamped to match so the rasterizer
    inherits it in turn.

    *Units.* Positions are written in PIXELS/plane-index (what ``_dvc_rows`` writes, and
    what ``transform.rasterize_field`` maps onto the voxel grid); every derived quantity is
    physical — displacement µm, velocity µm/s, strain dimensionless, stress in whatever
    ``youngs_modulus`` is given in. The gradient is fitted on coordinates and displacements
    BOTH converted to µm, so it is dimensionless with no anisotropy factor left to forget.
    ``pixel_size_um`` is therefore required, and ``z_step_um`` too in 3-D: labelling µm on a
    number that is really pixels is the one failure this catalog refuses rather than
    degrades.

    *Reference configuration* (the ``reference`` mode). ``first_frame`` differences each
    track against its own first observation — cumulative/Lagrangian, and the only one
    against which strain means anything. ``previous_frame`` gives the per-interval
    increment. Note "first" is per TRACK, not a global frame: a bead first detected at t=5
    references t=5, which is why this is not called ``fixed_frame`` the way
    ``analysis.dvc_field``'s lever is.

    *Rows emitted.* One per member row whose displacement is DEFINED. Under
    ``previous_frame`` that drops each track's first observation (it has no predecessor);
    under ``first_frame`` it drops nothing. A row that survives may still carry NaN strain —
    the MLS gauge refuses a particle with too few measured neighbours or a degenerate
    (collinear / coplanar) neighbourhood rather than returning the arbitrary least-norm fit
    — and NaN velocity at a track's first row, which is ``analysis.object_metrics``'
    convention for "no predecessor" and deliberately not 0.

    *Strain* is the moving-least-squares gauge that IS SerialTrack's own
    ``funCompDefGrad3.m`` (``kernels.track_field.mls_displacement_gradient``, pinned to the
    loop port in ``kernels.track_objects.compute_strain_mls`` by selftest), fitted
    independently within each ``(m,t,c)`` volume or ``(m,t,c,z)`` plane — a scatter pooled
    across timepoints or positions would fit a gradient through points that were never
    neighbours. The four strain MEASURES come from
    ``kernels.field_math.strain_from_gradient``, shared with ``analysis.dvc_field`` so the
    two nodes cannot disagree about what "Green-Lagrange" means.

    *The fit is in the REFERENCE configuration*, at ``x - u``, while the row is REPORTED at
    its current position. That asymmetry is load-bearing: the shared measure code builds
    ``F = I + G``, which is the deformation gradient only for the material gradient
    ``du/dX``. Fitting at the deformed positions instead would leave ``infinitesimal``
    right to first order and make all three finite measures quietly wrong — see the
    comment at the fit. SerialTrack's own ``_compute_strain`` computes both configurations
    and derives the reference one the same way.

    *Stress is not SerialTrack.* SerialTrack computes displacement and strain and stops,
    because stress is a claim about the SPECIMEN rather than about the images. So it is an
    explicit, named ``constitutive`` choice, off by default, and turning it on makes the two
    moduli appear: nobody gets a stress map without having stated what they believe the gel
    is made of. Its ``plane`` companion is read only on a 2-D field, where the out-of-plane
    component has to be assumed; a 3-D field measures it. ``traction`` is deliberately NOT a
    choice here — surface traction is an inverse problem over a half-space, not a local law,
    and belongs in its own node.

    *Determinism (memo invariant, §10).* Every step is a pure function of the member table:
    rows are emitted in a canonical ``lexsort`` order, the gauge is a batched linear solve,
    and nothing draws on an RNG. ``pixel_size_um`` / ``z_step_um`` / ``dt_s`` are read
    through ``ctx.calib`` so the memo fences on them.

    *Hard refusals* where a number would otherwise be silently wrong: no ``track_id``
    column (there is no correspondence, so no displacement exists); a missing
    ``pixel_size_um`` (or ``z_step_um`` in 3-D), which would label pixels as µm; a
    Poisson ratio at or beyond 0.5, where the bulk modulus is infinite and the pressure is
    not determined by the strain at all; a non-positive Young's modulus; and a strain
    radius that resolves to nothing.
    """
    from nodegraph.kernels.field_math import strain_from_gradient
    from nodegraph.kernels.track_field import (
        linear_elastic_stress,
        mls_displacement_gradient,
        stress_invariants,
        strain_invariants,
    )

    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    reference = modes.get("reference", "first_frame")
    strain_type = modes.get("strain_type", "infinitesimal")
    constitutive = modes.get("constitutive", "none")
    plane = modes.get("plane", "strain")
    want_stress = constitutive == "linear_elastic"
    want_vel = bool(ctx.params.get("velocity", False))

    domain, src, cols, zk = _object_table(ctx, ds, node="track field", allow_3d=True)
    is_3d = zk == "subpixel"
    axes = _AXES[is_3d]
    d = len(axes)

    track = cols.get("track_id")
    if track is None:
        raise ValueError(
            f"track field: the {domain.value} layer {src!r} carries no 'track_id' column, "
            f"so there is no correspondence between one frame's objects and the next — and "
            f"a displacement is a statement ABOUT that correspondence, not about the "
            f"positions. Run track.objects (five linkers, `serialtrack` for a bead field) "
            f"or track.link upstream; both write `track_id` back onto the layer they "
            f"tracked.")
    track = np.rint(np.asarray(track, dtype=float)).astype(np.int64)

    # ── calibration: the gradient is fitted in µm, so both scales are mandatory ──
    px_um = ctx.calib("pixel_size_um")
    z_um = ctx.calib("z_step_um") if is_3d else 1.0
    if not px_um or (is_3d and not z_um):
        raise ValueError(
            f"track field: every column this node writes is physical — displacement in µm, "
            f"strain from a gradient of µm per µm — and this Dataset declares "
            f"pixel_size_um={px_um!r}"
            + (f" / z_step_um={z_um!r}" if is_3d else "")
            + ". A placeholder 1.0 would report PIXELS under a µm label, and in 3D would "
              "additionally distort every gradient by the anisotropy ratio (5-10x on a "
              "typical stack). Select the Load card that opened this file and type the "
              "missing value into its box (Parameters, in µm) — a plain TIFF records no "
              "spacing at all, so an absent z_step_um is the normal case for one and "
              "nothing downstream can infer it.")
    px_um = float(px_um)
    z_um = float(z_um)
    step = np.array([z_um, px_um, px_um][3 - d:], dtype=float)   # µm per unit index

    # ── rows, in a canonical order so the output is byte-stable (§10) ──
    mm = np.asarray(cols["m"], dtype=float).round().astype(np.int64)
    cc = np.asarray(cols["c"], dtype=float).round().astype(np.int64)
    tt = np.asarray(cols["t"], dtype=float).round().astype(np.int64)
    zz = np.asarray(cols["z"], dtype=float)
    yy = np.asarray(cols["y"], dtype=float)
    xx = np.asarray(cols["x"], dtype=float)
    n = len(track)

    # Index coordinates (what goes back out), and their µm twin (what the maths uses).
    idx_coords = np.column_stack([zz, yy, xx] if is_3d else [yy, xx])
    coords_um = idx_coords * step

    # ── drop short tracks ───────────────────────────────────────────────────────
    min_len = max(1, int(ctx.params.get("min_track_length", 2)))
    keep = track > 0
    if min_len > 1 and keep.any():
        uniq, counts = np.unique(track[keep], return_counts=True)
        # np.isin (not a Python set membership test): a bead field is hundreds of thousands
        # of rows, and this is the only place the whole table is scanned per track.
        keep &= np.isin(track, uniq[counts >= min_len])

    work_track = np.where(keep, track, 0)

    # ── displacement (µm) and the instantaneous velocity (µm/s) ────────────────
    disp_um = _displacement(work_track, tt, coords_um, reference)
    dt_s = _frame_interval_s(ctx, needed=want_vel, wanted=("velocity",),
                             node="track field")
    vel_um_s = (_object_velocity_nd(work_track, tt, coords_um, dt_s)
                if want_vel else None)

    # A row is EMITTED iff its displacement is defined; strain and velocity may still be
    # NaN on an emitted row (too few neighbours; no predecessor) — see the docstring.
    alive = np.isfinite(disp_um).all(axis=1)
    if not alive.any():
        return _empty_field(ds, ctx, axes, want_vel, want_stress, zk)

    # ── the MLS strain gauge, per independent field ────────────────────────────
    radius_um = float(ctx.params.get("strain_radius", 20.0))
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError(
            f"track field: the strain radius must be a positive distance in µm (got "
            f"{radius_um!r}); it is the neighbourhood the displacement gradient is fitted "
            f"over, and at zero there is no neighbourhood to fit.")
    n_nb = max(d + 1, int(ctx.params.get("strain_neighbors", 20)))

    G = np.full((n, d, d), np.nan)
    # The gradient is fitted in the REFERENCE configuration — at `x - u`, where each
    # particle WAS — not at the deformed positions where it now is. That is not a
    # preference: `strain_from_gradient` forms the deformation gradient as `F = I + G`,
    # which is only true for the material gradient `du/dX`. Handing it the spatial
    # gradient `du/dx` would leave `infinitesimal` right to first order (the two differ by
    # a factor `1/(1+eps)`, 6% at 6% strain) and all three FINITE measures quietly wrong,
    # since `I + du/dx` is the inverse deformation gradient, not F — Green-Lagrange would
    # report Almansi's answer under Green-Lagrange's name. Fitting in the reference config
    # also matches `analysis.dvc_field`, whose subset centres live on the reference image,
    # so the shared measure code means the same thing in both nodes. It matches
    # SerialTrack's own post-processing too: `SerialTracker._compute_strain` computes both
    # configurations and gets the reference one from exactly this `coords_b - disp`.
    #
    # The ROW is still reported at its current position, which is where you want it drawn
    # on frame t; only the fit's geometry is referential. Neighbourhoods are therefore
    # material neighbourhoods, which is also the physically right choice.
    ref_coords_um = coords_um - np.nan_to_num(disp_um, nan=0.0)
    # A gradient may only be fitted among points that were simultaneously present and
    # genuinely adjacent: one (m,t,c) volume in 3D, one (m,t,c,z) plane in 2D.
    group_key = ([mm, tt, cc] if is_3d else [mm, tt, cc, np.rint(zz).astype(np.int64)])
    keys = np.stack(group_key, axis=1)[alive]
    idx_alive = np.flatnonzero(alive)
    order = np.lexsort(tuple(keys[:, k] for k in range(keys.shape[1] - 1, -1, -1)))
    keys, idx_alive = keys[order], idx_alive[order]
    bounds = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    for chunk in np.split(idx_alive, bounds):
        if chunk.size == 0:
            continue
        _, g, ok = mls_displacement_gradient(
            ref_coords_um[chunk], disp_um[chunk], radius=radius_um, n_neighbors=n_nb)
        g[~ok] = np.nan
        G[chunk] = g

    # ── strain, its invariants, and (optionally) stress ────────────────────────
    fit = np.isfinite(G).all(axis=(1, 2))
    strain = np.full((d, d, n), np.nan)
    if fit.any():
        strain[:, :, fit] = strain_from_gradient(
            np.moveaxis(G[fit], 0, -1), strain_type)
    strain_vol, strain_shear = strain_invariants(strain)

    stress = press = vm = tau = None
    if want_stress:
        e = float(ctx.params.get("youngs_modulus", 1000.0))
        nu = float(ctx.params.get("poisson_ratio", 0.45))
        stress = np.full((3, 3, n), np.nan)
        if fit.any():
            stress[:, :, fit] = linear_elastic_stress(
                strain[:, :, fit], youngs_modulus=e, poisson_ratio=nu, plane=plane)
        press, vm, tau = stress_invariants(stress)

    # ── assemble the Point table ───────────────────────────────────────────────
    sel = np.flatnonzero(alive)
    sel = sel[np.lexsort((track[sel], tt[sel], cc[sel], mm[sel]))]   # canonical (§10)
    out: Dict[str, np.ndarray] = {
        "id": np.arange(sel.size, dtype=np.int64),
        "m": mm[sel], "t": tt[sel], "c": cc[sel],
        "z": (zz[sel] if is_3d else np.rint(zz[sel])),
        "y": yy[sel], "x": xx[sel],
        "track_id": track[sel].astype(np.int64),
    }
    for k, a in enumerate(axes):
        out[f"disp_{a}"] = disp_um[sel, k]
    out["disp_mag_um"] = np.sqrt((disp_um[sel] ** 2).sum(axis=1))
    if want_vel:
        for k, a in enumerate(axes):
            out[f"v{a}"] = vel_um_s[sel, k]
        out["speed"] = np.sqrt((vel_um_s[sel] ** 2).sum(axis=1))
    for i, ai in enumerate(axes):
        for j, aj in enumerate(axes):
            out[f"strain_{ai}{aj}"] = strain[i, j, sel]
    out["strain_vol"] = strain_vol[sel]
    out["strain_max_shear"] = strain_shear[sel]
    if want_stress:
        # Always the full 3x3 in (z,y,x): a 2-D field's out-of-plane component is produced
        # by BOTH plane assumptions (as nu*(sxx+syy) or as exactly 0) and von Mises needs
        # it, so hiding it would make the invariant unreproducible from the columns.
        s_axes = ("z", "y", "x")
        for i, ai in enumerate(s_axes):
            for j, aj in enumerate(s_axes):
                if not is_3d and (i == 0 or j == 0) and not (i == 0 and j == 0):
                    continue            # the shear terms onto z are identically zero
                out[f"stress_{ai}{aj}"] = stress[i, j, sel]
        out["pressure"] = press[sel]
        out["von_mises"] = vm[sel]
        out["max_shear"] = tau[sel]

    layer = ctx.layer("name")
    table = StructureTable(Domain.POINT, out, layer=layer, z_kind=zk)
    # Provenance (§7b) so a reader — and a future accumulate/traction node — can tell how
    # this field was referenced and what material was assumed, which no column records.
    prov = {"track_field_reference": reference, "track_field_strain_type": strain_type,
            "track_field_constitutive": constitutive}
    if want_stress:
        prov["track_field_youngs_modulus"] = float(ctx.params.get("youngs_modulus", 1000.0))
        prov["track_field_poisson_ratio"] = float(ctx.params.get("poisson_ratio", 0.45))
        if not is_3d:
            prov["track_field_plane"] = plane
    if want_vel:
        prov["track_field_dt_s"] = float(dt_s)
    return ds.with_structure(table).with_metadata(**prov)


def _empty_field(ds: Dataset, ctx: EvalContext, axes: Tuple[str, ...],
                 want_vel: bool, want_stress: bool, zk: str) -> Dataset:
    """A schema-correct EMPTY Point table, for a graph whose tracks are all too short.

    Emitting the columns with zero rows rather than nothing at all is what keeps a
    downstream column picker populated while the user fixes the upstream tracking — the
    same reason ``analysis.dvc_field`` builds an empty ``point_table`` on a strain-free
    run."""
    cols: Dict[str, np.ndarray] = {k: np.zeros(0, dtype=np.int64)
                                   for k in ("id", "m", "t", "c", "track_id")}
    for k in ("z", "y", "x"):
        cols[k] = np.zeros(0, dtype=float)
    for name in _field_columns(axes, want_vel, want_stress):
        cols.setdefault(name, np.zeros(0, dtype=float))
    return ds.with_structure(
        StructureTable(Domain.POINT, cols, layer=ctx.layer("name"), z_kind=zk))


def _field_columns(axes: Tuple[str, ...], want_vel: bool,
                   want_stress: bool) -> List[str]:
    """The derived columns the compute writes, in its own order.

    Shared by the compute's empty-table path and by :func:`_columns_track_field` so the
    edit-time declaration and the runtime table are derived from ONE rule rather than
    transcribed twice — the ``_shared.columns.dvc_field_columns`` discipline."""
    names = [f"disp_{a}" for a in axes] + ["disp_mag_um"]
    if want_vel:
        names += [f"v{a}" for a in axes] + ["speed"]
    names += [f"strain_{i}{j}" for i in axes for j in axes]
    names += ["strain_vol", "strain_max_shear"]
    if want_stress:
        s = ("z", "y", "x")
        names += [f"stress_{i}{j}" for i in s for j in s
                  if len(axes) == 3 or i == j or (i != "z" and j != "z")]
        names += ["pressure", "von_mises", "max_shear"]
    return names


def _columns_track_field(params, modes, incoming):
    """The Point columns this node writes (V2.28), for the downstream column pickers.

    Dimensionality is inherited from the member layer's ``z_kind`` and so is NOT knowable
    at edit time — there is no Dataset here to ask. Both the 2-D and the 3-D names are
    therefore declared: over-declaring offers a column a 2-D run will not carry (a menu
    entry the compute then refuses by name), while under-declaring would make a real
    column unpickable, which is the direction this catalog must not err in."""
    try:
        _dom, lyr = member_layer(params, modes)
        vel = bool((params or {}).get("velocity", False))
        stress = (modes or {}).get("constitutive", "none") == "linear_elastic"
        names = ["track_id"]
        for ax in (("y", "x"), ("z", "y", "x")):
            names += _field_columns(ax, vel, stress)
        return on_layer(Domain.POINT, str((params or {}).get("name") or "track_field"),
                        dict.fromkeys(names))
    except Exception:                        # pragma: no cover - defensive
        return ()


register_node(
    _compute_track_field, op_key="analysis.track_field", label="Track Field",
    category="analysis",
    adds_columns=_columns_track_field,
    # ONE of the two, never both — `analysis.object_metrics`' note applies verbatim:
    # `_object_table` is shared, so the requirement is the same function of `target` and a
    # static union would over-claim on both branches (V2.22).
    reads_domains=frozenset(),
    reads_domains_by_mode={"target": {
        "label": frozenset({Domain.LABEL, Domain.VOXEL}),
        "point": frozenset({Domain.POINT}),
    }},
    adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InString("labels", "Label layer", field=False, default="labels",
                 layer_in=Domain.VOXEL,
                 available_in={"target": frozenset({"label"})},
                 description=
                 "Which tracked Label table supplies the particles — the output of "
                 "Segmentation or Connected Components after a tracker has written its "
                 "`track_id` column onto it. Only the CENTROIDS are read: a region's shape "
                 "and area play no part in a displacement field, so tracked labels and "
                 "tracked points give the same answer for the same centroids."),
        InString("points", "Point layer", field=False, default="spots",
                 layer_in=Domain.POINT,
                 available_in={"target": frozenset({"point"})},
                 description=
                 "Which tracked Point table supplies the particles — the output of Spot or "
                 "Particle Detection after a tracker has written its `track_id` column onto "
                 "it. This is the usual input for a bead field: Particle Detection with the "
                 "3D lever emits subpixel depths, which is what makes the gradient a full "
                 "3x3 rather than a per-plane 2x2."),
        InString("name", "Output layer", field=False, default="track_field",
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point field this node writes — one row per tracked particle "
                 "per timepoint, carrying its displacement, strain and (if asked) stress. "
                 "Feed it to Rasterize Field to interpolate those columns into "
                 "full-resolution image maps, one Voxel layer per column."),
        InFloat("strain_radius", "Strain radius", unit="um", field=False, default=20.0,
                pick_kind="distance",
                description=
                "Radius of the neighbourhood the displacement GRADIENT is fitted over, in "
                "microns — the gauge length of the strain measurement, and the single knob "
                "that decides what it can resolve. SMALL follows sharp strain "
                "concentrations but is noisy and leaves sparse particles with no valid fit "
                "at all (they come out NaN, not zero); LARGE averages over a wider patch, "
                "which is smooth and stable and systematically flattens exactly the "
                "concentrations a traction measurement is looking for. Set it from the "
                "particle SPACING — a few times the mean bead separation is the usual "
                "starting point, so that every fit has neighbours in every direction. It "
                "changes strain and stress but never displacement or velocity."),
        InInt("strain_neighbors", "Strain neighbours", unit="", field=False, default=20,
              description=
              "How many nearest particles are offered to each gradient fit before the "
              "radius cut applies — a ceiling on cost, not a second radius. Raise it only "
              "if the radius is large enough to hold more particles than this (a dense "
              "field will otherwise be cut off at an arbitrary subset); lowering it below "
              "roughly 10 makes the fit sensitive to individual localisation errors. "
              "Floored at 4 in 3D (3 in 2D), since a gradient tensor has that many "
              "unknowns. Inert whenever the radius is the binding constraint."),
        InInt("min_track_length", "Min track length", unit="", field=False, default=2,
              description=
              "Discard any track observed in fewer than this many frames before computing "
              "anything. RAISE it to drop the short spurious tracks a crowded or noisy "
              "field produces — they carry large, meaningless apparent displacements and, "
              "because the gradient fit averages over neighbours, one of them corrupts "
              "strain for every particle around it too. 2 is the minimum that can define a "
              "displacement at all; 1 keeps single-frame detections, which contribute a "
              "reference row and nothing else."),
        InBool("velocity", "Velocity", field=False, default=False,
               description=
               "Also emit per-particle velocity columns (`vy`/`vx`, plus `vz` in 3D) and "
               "`speed`, in µm/s. These are INSTANTANEOUS — each particle against its own "
               "previous detection, divided by the real frame gap — and so are unaffected "
               "by the reference-configuration choice, which governs displacement only. "
               "Needs a frame interval and REFUSES without one rather than report µm/frame "
               "under a µm/s label. A track's first observation has no predecessor and "
               "comes out NaN, not 0."),
        *_InFrameInterval(),
        InFloat("youngs_modulus", "Young's modulus", unit="Pa", field=False,
                default=1000.0,
                available_in={"constitutive": frozenset({"linear_elastic"})},
                description=
                "The specimen's stiffness E. It scales every stress column LINEARLY and "
                "sets their unit — give it in pascals and the stress maps are in pascals. "
                "This is a measured property of your gel, not a fitting knob: polyacrylamide "
                "TFM substrates are typically hundreds to tens of thousands of Pa, and the "
                "1000 here is a placeholder, not a recommendation. It has no effect on "
                "displacement, velocity or strain, so getting it wrong rescales the stress "
                "maps and nothing else."),
        InFloat("poisson_ratio", "Poisson ratio", unit="", field=False, default=0.45,
                available_in={"constitutive": frozenset({"linear_elastic"})},
                description=
                "How much the material contracts transversely when stretched, 0 to just "
                "under 0.5. It sets how much the HYDROSTATIC part of the strain contributes "
                "to stress, so it moves `pressure` strongly and `von_mises` far less. "
                "Hydrogels are nearly incompressible and are usually taken at 0.45-0.49; "
                "0.5 exactly is refused, because there the bulk modulus is infinite and "
                "pressure stops being determined by strain at all. Like the modulus, it "
                "leaves displacement, velocity and strain untouched."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("target", ["label", "point"], default="point", label="Members",
             description=
             "Which kind of tracked object the field is built from — the regions of a Label "
             "table or the detections of a Point table. It selects which source socket is "
             "live. Only centroids are read either way, so this is a question about where "
             "your particles came from, not about what gets computed. It defaults to "
             "`point`, unlike the tracker's own matching lever, because a displacement "
             "field is almost always measured on fiducial beads.",
             choice_docs={
                 "label":
                     "Build the field from a tracked Label table (Segmentation / Connected "
                     "Components). Use it when the things that move ARE the segmented "
                     "objects — cells in a monolayer, nuclei in a tissue — and you want "
                     "their collective deformation. Region shape, area and intensity are "
                     "ignored; only the centroid trajectory matters.",
                 "point":
                     "Build the field from a tracked Point table (Spot / Particle "
                     "Detection). The bead-field case: fiducial markers embedded in a gel, "
                     "detected per frame and linked by `serialtrack`. Particle Detection "
                     "with the 3D lever emits a measured depth, which is what puts this "
                     "node into full 3D rather than per-plane 2D.",
             }),
        Mode("reference", ["first_frame", "previous_frame"], default="first_frame",
             label="Reference",
             description=
             "Which configuration each displacement is measured AGAINST. It changes the "
             "displacement and everything derived from it (strain, stress) but never the "
             "velocity columns, which are instantaneous by definition. Note the reference "
             "is per TRACK, not a global frame — a particle first detected at t=5 "
             "references t=5.",
             choice_docs={
                 "first_frame":
                     "Cumulative (Lagrangian): every row is differenced against its own "
                     "track's FIRST observation. This is the one strain means something "
                     "against, since strain is deformation relative to an undeformed "
                     "state, and it is what a traction or stiffness measurement wants. "
                     "Drift and slow deformation accumulate visibly, which is the point. "
                     "Every row survives; a track's first row is exactly zero.",
                 "previous_frame":
                     "Incremental: each row is differenced against the immediately "
                     "preceding observation of the same track, giving the per-interval "
                     "step. Use it to watch RATES of deformation or to isolate one "
                     "loading step; the strain it yields is an increment, not a total, and "
                     "increments are not additive for the finite measures. Each track's "
                     "first observation has no predecessor and is dropped.",
             }),
        Mode("strain_type", ["infinitesimal", "green-lagrange", "almansi", "hencky"],
             default="infinitesimal", label="Strain measure",
             description=
             "Which strain MEASURE is derived from the displacement gradient. All four are "
             "computed by the same code `analysis.dvc_field` uses, so the two nodes cannot "
             "disagree about a name. They differ by fractions of a percent while strains "
             "are small and diverge as deformation grows, so the choice only matters once "
             "you are past a few percent.",
             choice_docs={
                 "infinitesimal":
                     "The linearized (engineering) strain, the symmetric part of the "
                     "displacement gradient. Cheapest and easiest to interpret, and the "
                     "right default for a gel under small deformation. It systematically "
                     "misreports large strain and is not rotation-invariant, so a large "
                     "rigid rotation of the field registers as spurious strain.",
                 "green-lagrange":
                     "The finite-strain measure in the REFERENCE configuration — the "
                     "material description, and the one that pairs with the cumulative "
                     "reference above. Rotation-invariant, and reports slightly LARGER "
                     "values than engineering strain in tension.",
                 "almansi":
                     "The Eulerian counterpart: finite strain referred to the DEFORMED "
                     "configuration, which is where these particles actually are. Also "
                     "rotation-invariant; reports slightly SMALLER values than engineering "
                     "strain in tension.",
                 "hencky":
                     "True/logarithmic strain, the log of the stretch. Additive across "
                     "successive increments, which makes it the measure to use with "
                     "incremental loading, and the best-behaved at large strain where "
                     "engineering strain becomes meaningless.",
             }),
        Mode("constitutive", ["none", "linear_elastic"], default="none",
             label="Stress model",
             description=
             "Whether to convert strain into STRESS, and by what material model. This is "
             "the one place the node asserts something about the specimen rather than about "
             "the images, which is why it is off by default and why the moduli only appear "
             "once it is on. Surface TRACTION is deliberately absent: that is an inverse "
             "problem over a half-space, not a local law, and does not belong on this node.",
             choice_docs={
                 "none":
                     "Report kinematics only — displacement, velocity, strain. Exactly what "
                     "SerialTrack itself computes, and the honest output when the material "
                     "is unknown or not linearly elastic. No moduli are asked for and no "
                     "stress columns are written.",
                 "linear_elastic":
                     "Isotropic linear elasticity, sigma = 2*mu*eps + lambda*tr(eps)*I, from "
                     "a Young's modulus and a Poisson ratio you supply. Adds the stress "
                     "tensor plus `pressure`, `von_mises` and `max_shear` in the modulus's "
                     "own unit. Valid for small strains in a homogeneous, isotropic, "
                     "time-independent material — so it describes a polyacrylamide gel "
                     "well and a viscoelastic or fibrous matrix only roughly.",
             }),
        Mode("plane", ["strain", "stress"], default="strain", label="2D assumption",
             available_in={"constitutive": frozenset({"linear_elastic"})},
             description=
             "What to assume about the third dimension when the field is 2D — read ONLY "
             "then, and inert on a 3D (subpixel-depth) field, which measures the "
             "out-of-plane component instead of assuming it. The dimensionality comes from "
             "the data, not from a lever, so this cannot be greyed out for you; the choice "
             "moves `pressure` and `von_mises` substantially and the in-plane shear not at "
             "all.",
             choice_docs={
                 "strain":
                     "PLANE STRAIN: no out-of-plane stretch (eps_zz = 0), so the material "
                     "around the plane holds it. The right assumption for a slice through a "
                     "THICK specimen — which a TFM gel is — and the reason it is the "
                     "default. It produces a non-zero sigma_zz = nu*(sigma_yy + sigma_xx).",
                 "stress":
                     "PLANE STRESS: nothing resists out-of-plane thinning (sigma_zz = 0), "
                     "so the material is free to contract through its thickness. The right "
                     "assumption for a THIN free-standing film or membrane. It reports "
                     "smaller in-plane stresses than plane strain for the same measured "
                     "strain.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="SerialTrack's post-processing half: a TRACKED object field (any linker's "
                "`track_id`, on Label or Point members) → per-particle displacement (µm), "
                "optional velocity (µm/s), the MLS deformation gradient and strain "
                "(4 measures), and optional isotropic linear-elastic stress + von Mises / "
                "pressure (Pa). Cumulative or incremental reference; 2D-per-plane vs full "
                "3D inherited from the members' z_kind. Emits the same Point schema as "
                "analysis.dvc_field, so transform.rasterize_field turns it into maps.")
