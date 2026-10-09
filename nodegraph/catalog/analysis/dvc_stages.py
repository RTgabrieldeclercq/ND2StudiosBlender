"""Staged DVC (pyALDVC) — one node per section of the AL-DVC pipeline, placed together as
the ``bundle:dvc`` node bundle (2026-10-08).

``analysis.dvc_field`` runs pyALDVC's Sections 2-8 as one card. That is right for a batch
and wrong for a user who wants to SEE the seed field before paying for the refinement,
re-run the global step with another β without re-correlating, or re-derive strain under
another measure in a second. These five nodes are those sections, one each, the way
``al_dvc/core/pipeline.py`` itself is organised:

    dvc.grid           §2  the node grid + the reference policy        (no correlation)
    dvc.initial_guess  §3  the integer-plus seed per node (pyramid NCC / FFT shift)
    dvc.local_match    §4  the 12-DOF IC-GN subset match               (the cost centre)
    dvc.global_step    §5-6 the global compatibility step + ADMM iterations
    dvc.strain         §8  strain from the displacement field          (no image read)

**The field between stages is a Point table** on one layer (``dvc`` by default): the grid
node writes the rows (one per node per correlated frame, in pyALDVC's node order), every
later stage reads it back by layer name and ADDS or OVERWRITES columns on those same rows
(``disp_*`` in µm, ``grad_*``, ``qfactor``, ``status``, …, ``strain_*``). So the Viewer draws
the field after any stage, a Spreadsheet shows it, and a stage re-pulled with new values
re-writes only its own columns. Each stage rebuilds what it needs from the volumes (the
reference's gradients, Hessians and global operators are a few seconds); nothing opaque
travels between nodes, which is what keeps every stage memoised, spillable and saveable.

**What is decided once, at the grid, and inherited** (metadata intelligence, §7b): the
subset size/spacing, the correlation channel, the pre-filter and the reference policy
(fixed frame N / previous frame) are stamped as ``dvc_*`` provenance on the grid's output
and READ by every later stage — no stage asks for them again, and a stage fed a table
that lacks them refuses by name. The local stage stamps its interpolation/backend choices
the same way, because the global step must rebuild the identical local context.

**One deliberate narrowing** against the single node: no external ``reference`` Dataset
input. To correlate against a separate undeformed stack, chain it in front of the series
with Timeseries Builder and keep *Reference frame* at 0 — one mechanism, already in the
catalog, instead of a socket on four cards that must all be wired to the same thing.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from nodegraph.bundles import Bundle, BundleEdge, BundleNode, register_bundle
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
from nodegraph.catalog._shared.columns import POINT_INVARIANT, on_layer
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer
from nodegraph.catalog._shared.progress import _UnitBar
from nodegraph.catalog._shared.analysis_units import scoped_frame

_DEFAULT_LAYER = "dvc"
_AX = ("z", "y", "x")
#: ``disp_z / disp_y / disp_x`` — µm on the wire, the same names ``analysis.dvc_field``
#: writes, so Accumulate DVC Field, Rasterize Field and the Viewer's arrows read them.
DISP_COLUMNS = tuple(f"disp_{a}" for a in _AX) + ("disp_mag_um",)
#: ``grad_<i><j>`` = ∂u_i/∂x_j, dimensionless, this repo's (z, y, x) index order.
GRAD_COLUMNS = tuple(f"grad_{i}{j}" for i in _AX for j in _AX)
STRAIN_COLUMNS = tuple(f"strain_{i}{j}" for i in _AX for j in _AX)
STD_COLUMNS = tuple(f"disp_std_{a}" for a in _AX)
SEED_COLUMNS = DISP_COLUMNS + ("seed_cc", "seed_ok")
LOCAL_COLUMNS = DISP_COLUMNS + GRAD_COLUMNS + STD_COLUMNS + ("qfactor", "status", "n_iter",
                                                              "bad", "valid")
GLOBAL_COLUMNS = DISP_COLUMNS + GRAD_COLUMNS + STD_COLUMNS + ("qfactor", "status", "outlier")
STRAIN_OUT_COLUMNS = STRAIN_COLUMNS + ("strain_valid",)

#: Provenance the grid stage stamps and every later stage reads.
_GRID_PROV = ("dvc_subset_size", "dvc_subset_size_z", "dvc_subset_spacing",
              "dvc_subset_spacing_z", "dvc_prefilter_sigma", "dvc_channel",
              "dvc_reference_mode", "dvc_reference_frame")
#: Provenance the local stage stamps so the global step rebuilds the same local context.
_LOCAL_PROV = ("dvc_interp_method", "dvc_backend", "dvc_subset_stride", "dvc_tile_local",
               "dvc_n_threads", "dvc_icgn_max_iter")


# ── shared plumbing ───────────────────────────────────────────────────────────────

def _vox(ctx: EvalContext) -> Tuple[float, float, float]:
    px = float(ctx.calib("pixel_size_um") or 0.1)
    zs = float(ctx.calib("z_step_um") or 0.5)
    return (zs, px, px)


def _need_z(ax, node: str) -> None:
    if ax.z < 2:
        raise ValueError(
            f"{node} needs z>1 (got z={ax.z}): pyALDVC is 3D-only — its subset is a cube of "
            "≥4 voxels per axis, so a single plane cannot be correlated. Use DIC (pyALDIC) "
            "for 2D image correlation.")


def _plan(ax, ref_mode: str, ref_frame: int) -> List[Tuple[int, int]]:
    """``[(t_out, t_ref), …]`` — which frames are correlated against which. Fixed frame:
    every t against ``ref_frame`` (the reference's own row is the zero field, written
    without a solve); previous frame: ``t`` against ``t-1`` for ``t ≥ 1`` (t=0 has no row)."""
    if ref_mode == "previous_frame":
        return [(t, t - 1) for t in range(1, ax.t)]
    r = min(max(0, int(ref_frame)), max(0, ax.t - 1))
    return [(t, r) for t in range(ax.t)]


def _table(ds: Dataset, layer: str, node: str) -> Dict[str, np.ndarray]:
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT) if a.layer == layer}
    for req in ("id", "m", "t", "c", "z", "y", "x"):
        if req not in cols:
            raise ValueError(f"{node}: Point layer {layer!r} lacks the column {req!r} — run DVC "
                             "Grid upstream (it writes the grid rows every stage fills)")
    return cols


def _prov(md, node: str, keys=_GRID_PROV) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k not in md:
            raise ValueError(
                f"{node}: the input carries no {k!r} provenance — this node reads its grid and "
                "reference policy from DVC Grid's output; wire DVC Grid upstream (the "
                "bundle places the chain in order)")
        out[k] = md[k]
    return out


def _grid_params(gp: Dict[str, Any]) -> Dict[str, Any]:
    return {"subset_size": int(gp["dvc_subset_size"]), "subset_size_z": int(gp["dvc_subset_size_z"]),
            "subset_spacing": int(gp["dvc_subset_spacing"]),
            "subset_spacing_z": int(gp["dvc_subset_spacing_z"]),
            "prefilter_sigma": float(gp["dvc_prefilter_sigma"])}


def _local_params(lp: Dict[str, Any]) -> Dict[str, Any]:
    return {"interp_method": str(lp["dvc_interp_method"]), "backend": str(lp["dvc_backend"]),
            "subset_stride": int(lp["dvc_subset_stride"]), "tile_local": int(lp["dvc_tile_local"]),
            "n_threads": int(lp["dvc_n_threads"]), "icgn_max_iter": int(lp["dvc_icgn_max_iter"])}


def _rows(cols: Dict[str, np.ndarray], m: int, t: int, c: int, n_nodes: int, node: str
          ) -> np.ndarray:
    idx = np.flatnonzero((cols["m"] == m) & (cols["t"] == t) & (cols["c"] == c))
    if len(idx) != n_nodes:
        raise ValueError(
            f"{node}: the layer holds {len(idx)} rows for (m={m}, t={t}, c={c}) but the grid "
            f"has {n_nodes} nodes — the table was filtered or re-ordered between stages; "
            "wire the stages directly after DVC Grid")
    return idx


def _vol(prov, m: int, t: int, c: int) -> np.ndarray:
    ax = prov.axes
    return prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x).astype(np.float32)


def _write(ds: Dataset, layer: str, cols: Dict[str, np.ndarray]) -> Dataset:
    """Add / overwrite columns on the EXISTING rows of ``layer`` (`with_layer`, not
    `with_structure`: ``id`` is not re-emitted, every sibling column keeps its order)."""
    for name, vals in cols.items():
        ds = ds.with_layer(Domain.POINT, name, np.asarray(vals), layer=layer)
    return ds


def _source(ctx: EvalContext, ds: Dataset, node: str, want: str) -> str:
    """The grid layer a stage fills: ``want`` is the compute's own ``ctx.layer("source")``
    (read in the compute, so the socket contract sees the read), resolved to the one Point
    table on the wire when it can be."""
    layer, _note = _resolve_layer(
        _point_layers(ds), want, node=node, socket="source",
        what="DVC grid table", where="the `data` input",
        remedy="run DVC Grid upstream — it writes the Point table the stages fill", ctx=ctx)
    return layer


class _RefCache:
    """The reference contexts of ONE m-position, keyed by reference frame — two at most
    (pyALDVC's own `_REF_CACHE_SIZE`): a fixed reference is built once per position, a
    previous-frame schedule rebuilds every pair, as the driver does."""

    def __init__(self, build):
        self._build = build
        self._d: Dict[int, Any] = {}

    def get(self, t_ref: int):
        if t_ref not in self._d:
            while len(self._d) >= 2:
                self._d.pop(next(iter(self._d)))
            self._d[t_ref] = self._build(t_ref)
        return self._d[t_ref]


# ── §2  DVC Grid ──────────────────────────────────────────────────────────────────

def _compute_dvc_grid(ctx: EvalContext) -> Dataset:
    """pyALDVC Section 2 as a node: the correlation node grid for this stack under the
    subset size / spacing, written as a Point table — one row per node per frame to be
    correlated, in pyALDVC's node order — plus the reference policy, stamped as
    ``dvc_*`` provenance for every later stage. No volume is read: this card is instant,
    and it is where the measurement density is judged on the Viewer before any
    correlation is paid for.

    Resolved spec: category analysis; op ``dvc.grid``; reads VOXEL (axes + calibration
    only), adds POINT; 3-D only (z<2 refused); footprint WHOLE_SERIES / {t,z,y,x} — it
    declares the whole series because its rows span every frame of it, though nothing is
    read. ``reference_mode`` fixed_frame writes a row block for every t (the reference's
    own block is the zero field); previous_frame writes t ≥ 1. Grid via
    :func:`nodegraph.kernels.aldvc_field.stage_grid` (al_dvc ``build_grid_axes`` +
    ``mesh_setup``) after :func:`stage_para` validated the subset against the volume
    (every axis ≥ 16, winsize + 11 ≤ extent, ≥ 2 nodes per axis)."""
    from nodegraph.kernels.aldvc_field import stage_grid, stage_para
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("DVC Grid needs an image provider on its input Dataset")
    ax = prov.axes
    _need_z(ax, "DVC Grid")
    modes = ctx.params.get("__modes__", {})
    ref_mode = str(modes.get("reference_mode", "fixed_frame"))
    # under the quick-analysis scope a source frame index is not a payload index; the
    # stamped reference is what the later stages index the payload with
    ref_frame = scoped_frame(ds.metadata, int(ctx.params.get("reference_frame", 0)), ax.t,
                             mode=ref_mode, node="DVC Grid")
    c = min(max(0, int(ctx.params.get("channel", 0))), max(0, ax.c - 1))
    gp = {
        "subset_size": max(4, int(ctx.params.get("subset_size", 32))),
        "subset_size_z": max(0, int(ctx.params.get("subset_size_z", 0))),
        "subset_spacing": max(1, int(ctx.params.get("subset_spacing", 16))),
        "subset_spacing_z": max(0, int(ctx.params.get("subset_spacing_z", 0))),
        "prefilter_sigma": max(0.0, float(ctx.params.get("prefilter_sigma", 0.0))),
    }
    shape = (ax.z, ax.y, ax.x)
    para = stage_para(gp, _vox(ctx), shape)
    mesh, coords = stage_grid(para, shape)
    n = int(mesh.n_nodes)
    plan = _plan(ax, ref_mode, ref_frame)
    layer = ctx.layer("name")
    blocks = []
    for m in range(ax.m):
        for t_out, _t_ref in plan:
            blocks.append((m, t_out))
    total = n * len(blocks)
    cols: Dict[str, np.ndarray] = {
        "id": np.arange(total, dtype=np.int64),
        "m": np.repeat([b[0] for b in blocks], n).astype(np.int64),
        "t": np.repeat([b[1] for b in blocks], n).astype(np.int64),
        "c": np.full(total, c, dtype=np.int64),
        "z": np.tile(coords[:, 0], len(blocks)), "y": np.tile(coords[:, 1], len(blocks)),
        "x": np.tile(coords[:, 2], len(blocks)),
    }
    if not blocks:
        cols = {k: v[:0] for k, v in cols.items()}
    md = {
        "dvc_subset_size": gp["subset_size"], "dvc_subset_size_z": gp["subset_size_z"],
        "dvc_subset_spacing": gp["subset_spacing"],
        "dvc_subset_spacing_z": gp["subset_spacing_z"],
        "dvc_prefilter_sigma": gp["prefilter_sigma"], "dvc_channel": c,
        "dvc_reference_mode": ref_mode, "dvc_reference_frame": ref_frame,
        "dvc_grid_shape": tuple(int(s) for s in mesh.grid_shape), "dvc_stage": "grid",
    }
    ctx.progress(1, 1, f"{n} nodes per frame ({' × '.join(str(s) for s in mesh.grid_shape)}), "
                       f"{len(plan)} frame(s) per position", frames=ax.t)
    return (ds.with_structure(StructureTable(Domain.POINT, cols, layer=layer, z_kind="subpixel"))
            .with_metadata(**md))


def _columns_dvc_grid(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("name") or _DEFAULT_LAYER),
                    POINT_INVARIANT)


# ── §3  DVC Initial Guess ─────────────────────────────────────────────────────────

def _compute_dvc_initial_guess(ctx: EvalContext) -> Dataset:
    """pyALDVC Section 3 as a node: the starting displacement of every grid node —
    an FFT whole-volume shift (``global_shift``), then a pyramid / NCC integer search
    inside ``search_radius`` at the coarsest level, cleaned of outliers and inpainted —
    written as ``disp_*`` (µm) on the grid rows, with the match score (``seed_cc``) and
    whether the search accepted the node (``seed_ok``). Pull THIS before the local match:
    arrows that point the wrong way here mean the radius does not bracket the motion or
    the volumes are not registered, and no refinement recovers from a wrong seed.

    Resolved spec: category analysis; op ``dvc.initial_guess``; reads VOXEL + POINT, adds
    POINT (columns on the grid layer); 3-D only; footprint WHOLE_SERIES / {t,z,y,x} (reads
    the reference and each correlated frame). Grid, channel and reference policy come
    from the grid's ``dvc_*`` provenance. ``init_coarse_factor > 1`` runs the coarse-lattice
    seed (a full solve on every k-th node, interpolated) and also writes its gradient seed
    as ``grad_*`` for the local stage. Against a fixed reference the previous frame's seed
    is handed to the next frame's search, as pyALDVC does. Kernel:
    :func:`nodegraph.kernels.aldvc_field.stage_initial_guess`."""
    from nodegraph.kernels.aldvc_field import (stage_deformed, stage_initial_guess, stage_para,
                                               stage_reference)
    node = "DVC Initial Guess"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    _need_z(ax, node)
    layer = _source(ctx, ds, node, ctx.layer("source"))
    cols = _table(ds, layer, node)
    gp = _prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    p = _grid_params(gp)
    p.update({
        "init_guess": str(modes.get("init_guess", "pyramid")),
        "search_radius": max(0, int(ctx.params.get("search_radius", 0))),
        "global_shift": bool(ctx.params.get("global_shift", True)),
        "init_coarse_factor": max(1, min(8, int(ctx.params.get("init_coarse_factor", 1)))),
    })
    vox = np.asarray(_vox(ctx), dtype=np.float64)
    shape = (ax.z, ax.y, ax.x)
    para = stage_para(p, vox, shape)
    plan = _plan(ax, str(gp["dvc_reference_mode"]), int(gp["dvc_reference_frame"]))
    c = int(gp["dvc_channel"])
    n_rows = len(cols["id"])
    disp = np.full((n_rows, 3), np.nan)
    seed_cc = np.full(n_rows, np.nan)
    seed_ok = np.zeros(n_rows, dtype=np.int64)
    grad: Optional[np.ndarray] = None
    bar = _UnitBar(ctx, frames=max(1, ax.m), units_per_frame=max(1, len(plan)), note="seeding")
    for m in range(ax.m):
        refs = _RefCache(lambda t_ref, _m=m: stage_reference(_vol(prov, _m, t_ref, c), para, shape))
        prev_u, prev_ref = None, None
        for t_out, t_ref in plan:
            if t_out == t_ref:
                ref = refs.get(t_ref)
                idx = _rows(cols, m, t_out, c, ref.n_nodes, node)
                disp[idx] = 0.0
                seed_cc[idx] = 1.0
                seed_ok[idx] = 1
                bar.finish_unit(note=f"m={m} t={t_out}: the reference itself (zero field)")
                continue
            ref = refs.get(t_ref)
            idx = _rows(cols, m, t_out, c, ref.n_nodes, node)
            g_norm, g_prep = stage_deformed(_vol(prov, m, t_out, c), para, shape)
            previous = prev_u if (prev_ref == t_ref and prev_u is not None) else None
            u0, f0, info = stage_initial_guess(ref, g_norm, g_prep, previous_zyx=previous)
            disp[idx] = u0 * vox[None, :]
            cc = info.get("cc")
            ok = info.get("ok")
            seed_cc[idx] = np.asarray(cc, dtype=float).reshape(-1) if cc is not None else np.nan
            seed_ok[idx] = (np.asarray(ok, dtype=bool).reshape(-1).astype(np.int64)
                            if ok is not None else 1)
            if f0 is not None:
                if grad is None:
                    grad = np.full((n_rows, 3, 3), np.nan)
                grad[idx] = f0
            prev_u, prev_ref = u0, t_ref
            bar.finish_unit(note=f"m={m} t={t_out}: {info.get('method', '?')} seed, "
                                 f"{int(info.get('n_bad', 0))} nodes inpainted")
    out = {f"disp_{a}": disp[:, i] for i, a in enumerate(_AX)}
    out["disp_mag_um"] = np.sqrt(np.nansum(disp ** 2, axis=1))
    out["seed_cc"] = seed_cc
    out["seed_ok"] = seed_ok
    if grad is not None:
        for i, a in enumerate(_AX):
            for j, b in enumerate(_AX):
                out[f"grad_{a}{b}"] = grad[:, i, j]
    md = {"dvc_stage": "initial_guess", "dvc_init_guess": p["init_guess"],
          "dvc_search_radius": p["search_radius"], "dvc_global_shift": p["global_shift"],
          "dvc_init_coarse_factor": p["init_coarse_factor"]}
    return _write(ds, layer, out).with_metadata(**md)


def _columns_dvc_initial_guess(params, modes, incoming):
    cols = list(SEED_COLUMNS)
    if int((params or {}).get("init_coarse_factor", 1) or 1) > 1:
        cols += list(GRAD_COLUMNS)
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER), cols)


def _layers_on_source(params, modes):
    """Every stage after the grid writes INTO the layer its read socket names (total)."""
    return ((Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER)),)


# ── §4  DVC Local Match ───────────────────────────────────────────────────────────

def _compute_dvc_local_match(ctx: EvalContext) -> Dataset:
    """pyALDVC Section 4 as a node: the 12-DOF inverse-compositional Gauss-Newton match of
    every subset, seeded by the ``disp_*`` (and, when present, ``grad_*``) on the grid rows —
    the cost centre of the whole pipeline. Overwrites ``disp_*`` with the sub-voxel result,
    writes the deformation gradient ``grad_*``, the per-node ZNCC as ``qfactor``, pyALDVC's
    ``status`` (0 converged, 1 max_iter, 2 out of bounds, 3 invalid subset, 4 singular,
    5 nan, 6 skipped, 7 stalled), ``n_iter``, ``bad`` (rejected by the median test and
    inpainted), ``valid`` (a subset with enough texture to be solved) and the displacement
    uncertainty ``disp_std_*`` (µm).

    Resolved spec: category analysis; op ``dvc.local_match``; reads VOXEL + POINT, adds
    POINT; 3-D only; footprint WHOLE_SERIES / {t,z,y,x}. Inherits the grid / reference
    provenance; stamps ``dvc_interp_method`` / ``dvc_backend`` / ``dvc_subset_stride`` /
    ``dvc_tile_local`` / ``dvc_n_threads`` / ``dvc_icgn_max_iter`` so the global step
    rebuilds the identical local context. Kernel:
    :func:`nodegraph.kernels.aldvc_field.stage_local`."""
    from nodegraph.kernels.aldvc_field import (stage_deformed, stage_local, stage_para,
                                               stage_reference)
    node = "DVC Local Match"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    _need_z(ax, node)
    layer = _source(ctx, ds, node, ctx.layer("source"))
    cols = _table(ds, layer, node)
    for dn in DISP_COLUMNS[:3]:
        if dn not in cols:
            raise ValueError(f"{node}: the grid layer {layer!r} carries no {dn!r} — run DVC "
                             "Initial Guess first (the local match refines a seed)")
    gp = _prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    p = _grid_params(gp)
    p.update({
        "interp_method": str(modes.get("interp_method", "cubic")),
        "backend": str(modes.get("backend", "auto")),
        "icgn_max_iter": max(1, int(ctx.params.get("icgn_max_iter", 100))),
        "subset_stride": max(1, int(ctx.params.get("subset_stride", 1))),
        "tile_local": max(0, int(ctx.params.get("tile_local", 0))),
        "n_threads": max(0, int(ctx.params.get("n_threads", 0))),
        # the seed search does not run here, but the global step is ON by default in
        # DVCPara and the reference context builds the operators only when it is; the
        # local stage never uses them, so leave them off and save the FEM assembly
        "use_global_step": False,
    })
    vox = np.asarray(_vox(ctx), dtype=np.float64)
    shape = (ax.z, ax.y, ax.x)
    para = stage_para(p, vox, shape)
    plan = _plan(ax, str(gp["dvc_reference_mode"]), int(gp["dvc_reference_frame"]))
    c = int(gp["dvc_channel"])
    n_rows = len(cols["id"])
    disp_in = np.stack([np.asarray(cols[f"disp_{a}"], dtype=float) for a in _AX], axis=1)
    have_grad = all(g in cols for g in GRAD_COLUMNS)
    grad_in = (np.stack([np.asarray(cols[g], dtype=float) for g in GRAD_COLUMNS], axis=1)
               .reshape(-1, 3, 3) if have_grad else None)
    disp = np.full((n_rows, 3), np.nan)
    grad = np.full((n_rows, 3, 3), np.nan)
    std = np.full((n_rows, 3), np.nan)
    q = np.full(n_rows, np.nan)
    status = np.full(n_rows, -1, dtype=np.int64)
    n_iter = np.zeros(n_rows, dtype=np.int64)
    bad = np.zeros(n_rows, dtype=np.int64)
    valid = np.zeros(n_rows, dtype=np.int64)
    bar = _UnitBar(ctx, frames=max(1, ax.m), units_per_frame=max(1, len(plan)), note="matching")
    for m in range(ax.m):
        refs = _RefCache(lambda t_ref, _m=m: stage_reference(_vol(prov, _m, t_ref, c), para, shape))
        for t_out, t_ref in plan:
            ref = refs.get(t_ref)
            idx = _rows(cols, m, t_out, c, ref.n_nodes, node)
            valid[idx] = ref.valid.astype(np.int64)
            if t_out == t_ref:
                disp[idx] = 0.0
                grad[idx] = 0.0
                std[idx] = 0.0
                q[idx] = 1.0
                status[idx] = 0
                bar.finish_unit(note=f"m={m} t={t_out}: the reference itself (zero field)")
                continue
            _g_norm, g_prep = stage_deformed(_vol(prov, m, t_out, c), para, shape)
            u0 = np.where(np.isfinite(disp_in[idx]), disp_in[idx], 0.0) / vox[None, :]
            f0 = None
            if grad_in is not None and np.all(np.isfinite(grad_in[idx])):
                f0 = grad_in[idx]
            r = stage_local(ref, g_prep, u0, F0_zyx=f0)
            disp[idx] = r["U"] * vox[None, :]
            grad[idx] = r["F"]
            std[idx] = r["U_std"] * vox[None, :]
            q[idx] = r["zncc"]
            status[idx] = r["status"]
            n_iter[idx] = r["n_iter"]
            bad[idx] = r["bad"].astype(np.int64)
            n_conv = int(np.sum(r["status"] == 0))
            bar.finish_unit(note=f"m={m} t={t_out}: {n_conv}/{ref.n_nodes} converged in "
                                 f"{r['solve_time']:.1f} s, ZNCC median {np.nanmedian(r['zncc']):.3f}")
    out = {f"disp_{a}": disp[:, i] for i, a in enumerate(_AX)}
    out["disp_mag_um"] = np.sqrt(np.nansum(disp ** 2, axis=1))
    for i, a in enumerate(_AX):
        for j, b in enumerate(_AX):
            out[f"grad_{a}{b}"] = grad[:, i, j]
    for i, a in enumerate(_AX):
        out[f"disp_std_{a}"] = std[:, i]
    out.update({"qfactor": q, "status": status, "n_iter": n_iter, "bad": bad, "valid": valid})
    md = {"dvc_stage": "local_match", "dvc_interp_method": p["interp_method"],
          "dvc_backend": p["backend"], "dvc_subset_stride": p["subset_stride"],
          "dvc_tile_local": p["tile_local"], "dvc_n_threads": p["n_threads"],
          "dvc_icgn_max_iter": p["icgn_max_iter"]}
    return _write(ds, layer, out).with_metadata(**md)


def _columns_dvc_local_match(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    LOCAL_COLUMNS)


# ── §5-6  DVC Global Step ─────────────────────────────────────────────────────────

def _compute_dvc_global_step(ctx: EvalContext) -> Dataset:
    """pyALDVC Sections 5-6 as a node: the global compatibility step and the ADMM
    iterations that make AL-DVC more than local DVC. The independently matched subsets
    (``disp_*`` / ``grad_*`` / ``bad`` from the local match) are reconciled into one
    kinematically compatible field — a badly matched subset is pulled toward what its
    neighbours imply — alternating a 3-DOF local re-solve with the global solve for
    ``admm_iterations`` passes. Overwrites ``disp_*`` and ``grad_*``; refreshes ``qfactor``
    / ``status`` from the last local pass; writes ``outlier`` (converged, then rejected by
    the median test and replaced — not a measurement) and ``disp_std_*``. β (``beta``) is
    auto-tuned by the L-curve once per reference and reused across its frames, exactly
    as the single-node driver does; ``mu`` is the AL penalty.

    Resolved spec: category analysis; op ``dvc.global_step``; reads VOXEL + POINT, adds
    POINT; 3-D only; footprint WHOLE_SERIES / {t,z,y,x} (the local re-solves read the
    volumes again). Inherits the grid, reference AND local provenance (the local context
    must be rebuilt identically, so this card has no interpolation/backend controls of its
    own). Kernel: :func:`nodegraph.kernels.aldvc_field.stage_global` — the driver's
    `pipeline.py:282-368` verbatim."""
    from nodegraph.kernels.aldvc_field import (stage_deformed, stage_global, stage_para,
                                               stage_reference)
    node = "DVC Global Step"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset")
    ax = prov.axes
    _need_z(ax, node)
    layer = _source(ctx, ds, node, ctx.layer("source"))
    cols = _table(ds, layer, node)
    for dn in DISP_COLUMNS[:3] + GRAD_COLUMNS + ("bad",):
        if dn not in cols:
            raise ValueError(f"{node}: the grid layer {layer!r} carries no {dn!r} — run DVC "
                             "Local Match first (the global step reconciles its subsets)")
    gp = _prov(ds.metadata, node)
    lp = _prov(ds.metadata, node, _LOCAL_PROV)
    p = _grid_params(gp)
    p.update(_local_params(lp))
    p.update({
        "use_global_step": True,
        "mu": float(ctx.params.get("mu", 1e-3)),
        "beta": max(0.0, float(ctx.params.get("beta", 0.0))),
        "admm_iterations": max(1, int(ctx.params.get("admm_iterations", 4))),
    })
    vox = np.asarray(_vox(ctx), dtype=np.float64)
    shape = (ax.z, ax.y, ax.x)
    para = stage_para(p, vox, shape)
    plan = _plan(ax, str(gp["dvc_reference_mode"]), int(gp["dvc_reference_frame"]))
    c = int(gp["dvc_channel"])
    n_rows = len(cols["id"])
    disp_in = np.stack([np.asarray(cols[f"disp_{a}"], dtype=float) for a in _AX], axis=1)
    grad_in = np.stack([np.asarray(cols[g], dtype=float) for g in GRAD_COLUMNS], axis=1).reshape(-1, 3, 3)
    bad_in = np.asarray(cols["bad"]).astype(bool)
    q_in = np.asarray(cols["qfactor"], dtype=float) if "qfactor" in cols else np.full(n_rows, np.nan)
    st_in = (np.asarray(cols["status"]).astype(np.int64) if "status" in cols
             else np.full(n_rows, -1, dtype=np.int64))
    disp = disp_in.copy()
    grad = grad_in.copy()
    std = np.full((n_rows, 3), np.nan)
    q = q_in.copy()
    status = st_in.copy()
    outlier = np.zeros(n_rows, dtype=np.int64)
    last_beta = None
    bar = _UnitBar(ctx, frames=max(1, ax.m), units_per_frame=max(1, len(plan)), note="ADMM")
    for m in range(ax.m):
        refs = _RefCache(lambda t_ref, _m=m: stage_reference(_vol(prov, _m, t_ref, c), para, shape))
        beta_by_ref: Dict[int, float] = {}
        for t_out, t_ref in plan:
            ref = refs.get(t_ref)
            idx = _rows(cols, m, t_out, c, ref.n_nodes, node)
            if t_out == t_ref:
                std[idx] = 0.0
                bar.finish_unit(note=f"m={m} t={t_out}: the reference itself (zero field)")
                continue
            _g_norm, g_prep = stage_deformed(_vol(prov, m, t_out, c), para, shape)
            u1 = np.where(np.isfinite(disp_in[idx]), disp_in[idx], 0.0) / vox[None, :]
            f1 = np.where(np.isfinite(grad_in[idx]), grad_in[idx], 0.0)
            r = stage_global(ref, g_prep, u1, f1, bad_in[idx], beta=beta_by_ref.get(t_ref))
            beta_by_ref[t_ref] = float(r["beta"])
            last_beta = float(r["beta"])
            disp[idx] = r["U"] * vox[None, :]
            grad[idx] = r["F"]
            std[idx] = r["U_std"] * vox[None, :]
            fin = np.isfinite(r["zncc"])
            q[idx[fin]] = r["zncc"][fin]
            known = r["status"] >= 0
            status[idx[known]] = r["status"][known]
            outlier[idx] = r["outlier"].astype(np.int64)
            bar.finish_unit(note=f"m={m} t={t_out}: β={r['beta']:.3g}, {r['n_steps']} ADMM "
                                 f"step(s), |ΔU| global {r['update_global'][-1] if r['update_global'] else 0:.2e}")
    out = {f"disp_{a}": disp[:, i] for i, a in enumerate(_AX)}
    out["disp_mag_um"] = np.sqrt(np.nansum(disp ** 2, axis=1))
    for i, a in enumerate(_AX):
        for j, b in enumerate(_AX):
            out[f"grad_{a}{b}"] = grad[:, i, j]
    for i, a in enumerate(_AX):
        out[f"disp_std_{a}"] = std[:, i]
    out.update({"qfactor": q, "status": status, "outlier": outlier})
    md = {"dvc_stage": "global_step", "dvc_use_global_step": True, "dvc_mu": p["mu"],
          "dvc_beta": (last_beta if last_beta is not None else p["beta"]),
          "dvc_admm_iterations": p["admm_iterations"]}
    return _write(ds, layer, out).with_metadata(**md)


def _columns_dvc_global_step(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    GLOBAL_COLUMNS)


# ── §8  DVC Strain ────────────────────────────────────────────────────────────────

def _compute_dvc_strain(ctx: EvalContext) -> Dataset:
    """pyALDVC Section 8 as a node: strain from the displacement field on the grid rows —
    the nine ``strain_<i><j>`` columns (tensor shear, this repo's (z, y, x) index order,
    the measure chosen by ``strain_type``) and ``strain_valid``. Reads no volume: the grid
    is rebuilt from the stamped provenance and the stack's axes, so changing the measure,
    the smoothing or the differentiation method re-runs in a second, with the correlation
    untouched. ``direct`` uses the solver's own ``grad_*`` and is only meaningful against a
    fixed reference (an increment field's gradient is not a strain) — it falls back to the
    plane fit otherwise, as the driver does.

    Resolved spec: category analysis; op ``dvc.strain``; reads POINT, adds POINT; footprint
    WHOLE_SERIES with EMPTY kernel_axes — it crosses every frame of the table and touches no
    voxel. Stamps ``dvc_strain_type`` / ``dvc_strain_smooth`` / ``dvc_strain_method`` (what
    Accumulate DVC Field inherits). Kernel: :func:`nodegraph.kernels.aldvc_field.stage_strain`."""
    from nodegraph.kernels.aldvc_field import stage_para, stage_strain
    node = "DVC Strain"
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError(f"{node} needs an image provider on its input Dataset (for its axes)")
    ax = prov.axes
    _need_z(ax, node)
    layer = _source(ctx, ds, node, ctx.layer("source"))
    cols = _table(ds, layer, node)
    for dn in DISP_COLUMNS[:3]:
        if dn not in cols:
            raise ValueError(f"{node}: the grid layer {layer!r} carries no {dn!r} — run DVC "
                             "Local Match (or at least DVC Initial Guess) first")
    gp = _prov(ds.metadata, node)
    modes = ctx.params.get("__modes__", {})
    method = str(modes.get("strain_method", "plane_fit"))
    p = _grid_params(gp)
    p.update({
        "strain_method": method,
        "strain_type": str(modes.get("strain_type", "infinitesimal")),
        "strain_halfwidth": max(1, int(ctx.params.get("strain_halfwidth", 1))),
        "strain_smooth": max(0.0, float(ctx.params.get("strain_smooth", 0.0))),
        "disp_smoothing": max(0.0, float(ctx.params.get("disp_smoothing", 0.0))),
        "use_global_step": False,
    })
    vox = np.asarray(_vox(ctx), dtype=np.float64)
    shape = (ax.z, ax.y, ax.x)
    para = stage_para(p, vox, shape)
    ref_mode = str(gp["dvc_reference_mode"])
    plan = _plan(ax, ref_mode, int(gp["dvc_reference_frame"]))
    c = int(gp["dvc_channel"])
    n_rows = len(cols["id"])
    n_nodes = int(np.prod(np.asarray(ds.metadata.get("dvc_grid_shape", (0,)), dtype=np.int64)))
    disp_in = np.stack([np.asarray(cols[f"disp_{a}"], dtype=float) for a in _AX], axis=1)
    have_grad = all(g in cols for g in GRAD_COLUMNS)
    grad_in = (np.stack([np.asarray(cols[g], dtype=float) for g in GRAD_COLUMNS], axis=1)
               .reshape(-1, 3, 3) if have_grad else None)
    valid_in = np.asarray(cols["valid"]).astype(bool) if "valid" in cols else None
    strain = np.full((n_rows, 3, 3), np.nan)
    svalid = np.zeros(n_rows, dtype=np.int64)
    stype = p["strain_type"]
    bar = _UnitBar(ctx, frames=max(1, ax.m), units_per_frame=max(1, len(plan)), note="strain")
    for m in range(ax.m):
        for t_out, t_ref in plan:
            idx = _rows(cols, m, t_out, c, n_nodes, node) if n_nodes else np.flatnonzero(
                (cols["m"] == m) & (cols["t"] == t_out) & (cols["c"] == c))
            u = np.where(np.isfinite(disp_in[idx]), disp_in[idx], 0.0) / vox[None, :]
            f_direct = None
            if grad_in is not None and ref_mode == "fixed_frame" and np.all(np.isfinite(grad_in[idx])):
                f_direct = grad_in[idx]
            s3, sv, stype = stage_strain(para, shape, u, F_direct_zyx=f_direct,
                                         valid=(valid_in[idx] if valid_in is not None else None))
            strain[idx] = s3
            svalid[idx] = sv.astype(np.int64)
            bar.finish_unit(note=f"m={m} t={t_out}: {int(sv.sum())}/{len(idx)} nodes with strain")
    out: Dict[str, np.ndarray] = {}
    for i, a in enumerate(_AX):
        for j, b in enumerate(_AX):
            out[f"strain_{a}{b}"] = strain[:, i, j]
    out["strain_valid"] = svalid
    md = {"dvc_stage": "strain", "dvc_strain_type": stype, "dvc_strain_smooth": p["strain_smooth"],
          "dvc_strain_method": method}
    return _write(ds, layer, out).with_metadata(**md)


def _columns_dvc_strain(params, modes, incoming):
    return on_layer(Domain.POINT, str((params or {}).get("source") or _DEFAULT_LAYER),
                    STRAIN_OUT_COLUMNS)


# ── registration ─────────────────────────────────────────────────────────────────

_SOURCE = InString(
    "source", "Grid layer", field=False, default=_DEFAULT_LAYER, layer_in=Domain.POINT,
    description=
    "Which Point table holds the DVC grid this stage fills in — the one DVC Grid wrote "
    "(`dvc` unless you renamed it there). The stage reads its rows and the previous stages' "
    "columns from it and writes its own columns BACK ONTO THE SAME ROWS, so the Viewer, the "
    "Spreadsheet and every downstream node see one table that fills up stage by stage. With "
    "a single Point table on the wire the name is resolved for you.")

_STAGE_GRAN = Granularity.WHOLE_SERIES
_STAGE_KAX = frozenset({"t", "z", "y", "x"})

register_node(
    _compute_dvc_grid, op_key="dvc.grid", label="DVC Grid", category="analysis",
    adds_columns=_columns_dvc_grid,
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        InInt("channel", "Channel", unit="", field=False, default=0, pick_kind="channel",
              description=
              "Which channel every stage correlates (0-based) — decided here, once, and "
              "inherited by the stages after. Pick the channel with the most stable, "
              "high-contrast speckle (beads, a fiducial), never a reporter whose intensity "
              "changes for biological reasons: correlation assumes brightness is conserved, "
              "and a channel that violates that produces confident but wrong vectors."),
        InInt("reference_frame", "Reference frame", unit="", field=False, default=0,
              pick_kind="frame", available_in={"reference_mode": frozenset({"fixed_frame"})},
              description=
              "Which timepoint is the undeformed state (0-based) when the reference mode is a "
              "fixed frame: every frame is correlated against it, so the displacements are "
              "cumulative and directly comparable — until the deformation grows beyond what a "
              "subset can follow. Its own row block is the zero field. Inert under "
              "`previous_frame`. To use a SEPARATE undeformed stack, chain it in front of the "
              "series with Timeseries Builder and keep this at 0."),
        InInt("subset_size", "Subset size", unit="px", field=False, default=32,
              pick_kind="grid", pick_peer="subset_spacing",
              description=
              "Edge of each correlation subset in VOXELS — the cube of speckle matched between "
              "reference and deformed volumes. pyALDVC's sizing rule: 4-5× the speckle/pore "
              "diameter, so a subset holds 2-3 features per axis; below that it has too little "
              "texture to lock onto. LARGER lowers noise (and acts as a spatial filter); SMALLER "
              "resolves strain gradients but assumes one affine deformation across a window "
              "that no longer has one. Rounded UP to an even number ≥ 4. Every stage after this "
              "one inherits it."),
        InInt("subset_size_z", "Subset size (Z)", unit="px", field=False, default=0,
              description=
              "Subset edge along Z in voxels, for anisotropic stacks; 0 = the lateral size on "
              "all three axes. A confocal z-step is usually coarser than the pixel, so a cubic "
              "voxel subset spans several times more depth than width — set the number of "
              "planes covering roughly the same PHYSICAL distance as the lateral subset "
              "(e.g. 32/32/16). On a shallow stack this is also what makes a grid possible at "
              "all: the solver needs ≥ 2 nodes per axis with a 5-voxel margin each side. "
              "Rounded up to an even number ≥ 4."),
        InInt("subset_spacing", "Subset spacing", unit="px", field=False, default=16,
              pick_kind="grid", pick_peer="subset_size",
              description=
              "Spacing between node centres in voxels — THE grid density, and the dominant "
              "runtime knob (cost grows with the cube of the node count). SMALLER gives a "
              "denser field; LARGER is faster and can step straight over localised "
              "deformation. It does not change any single vector's accuracy — that is Subset "
              "size. Rule of thumb: half the subset for smooth fields, a quarter when chasing "
              "strain gradients. Pull this card and look: the dots ARE the grid."),
        InInt("subset_spacing_z", "Subset spacing (Z)", unit="px", field=False, default=0,
              description=
              "Node spacing along Z in voxels; 0 = the lateral spacing on all three axes. "
              "Worth setting on a stack with few planes: the lateral spacing may leave only "
              "one or two nodes through the depth, and a grid that thin cannot differentiate "
              "the through-plane strain."),
        InFloat("prefilter_sigma", "Pre-filter σ", unit="px", field=False, default=0.0,
                description=
                "Gaussian σ in voxels applied to EVERY volume before any stage correlates it; "
                "0 is off. The recommended first move on a low-SNR scan — pyALDVC names "
                "0.6-1.0 below an SNR of about 5 — because correlation noise enters through "
                "the image gradients, which smoothing tames at the cost of a little lateral "
                "resolution. Decided here so the seed, the match and the global step all see "
                "the same volume."),
        InString("name", "Grid layer", field=False, default=_DEFAULT_LAYER,
                 layer_out=(Domain.POINT,),
                 description=
                 "Name of the Point table this node writes: one row per grid node per "
                 "correlated frame, carrying only the node positions until the later stages "
                 "fill in `disp_*`, `grad_*`, `qfactor`, `status`, `strain_*`. Every stage "
                 "after this one selects the table by this name."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("reference_mode", ["fixed_frame", "previous_frame"], default="fixed_frame",
             label="Reference",
             description=
             "How frames pair up for correlation — decided once here and inherited by every "
             "stage: against one fixed undeformed frame, or each frame against the one before.",
             choice_docs={
                 "fixed_frame":
                     "Every timepoint is correlated against Reference frame: displacements "
                     "are cumulative from that state and directly comparable across time, "
                     "and the reference's own rows are the zero field. Breaks down once the "
                     "total deformation outgrows what a subset can follow.",
                 "previous_frame":
                     "Each timepoint is correlated against the one before it: small "
                     "increments that track large total deformation, at the price of one "
                     "reference build per pair and no row for t=0. Compose the increments "
                     "with Accumulate DVC Field to get totals.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_STAGE_KAX,
    description="pyALDVC §2 as a node: the correlation node grid (subset size / spacing) as a "
                "Point table + the channel and reference policy every later DVC stage inherits. "
                "Instant — nothing is correlated. First card of the staged DVC bundle.")

register_node(
    _compute_dvc_initial_guess, op_key="dvc.initial_guess", label="DVC Initial Guess",
    category="analysis", adds_columns=_columns_dvc_initial_guess, extra_layers=_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("search_radius", "Search radius", unit="px", field=False, default=0,
              description=
              "Half-width in voxels per axis of the NCC search that finds each node's integer "
              "starting guess at the coarsest pyramid level; 0 leaves pyALDVC's default (8). It "
              "must bracket the largest REAL displacement after the whole-volume shift, or a "
              "fast-moving region is seeded wrongly and the local match then converges "
              "confidently onto the wrong feature. Arrows pointing the wrong way on THIS card "
              "are the symptom; raise it (cost grows with the cube)."),
        InBool("global_shift", "Remove global shift", default=True,
               description=
               "Estimate one rigid whole-volume translation by phase correlation and remove "
               "it before the per-node search. ON is what lets a modest search radius cope "
               "with a series that drifted bodily (stage drift, a bumped dish). OFF when the "
               "volumes are already registered — then it can only add noise."),
        InInt("init_coarse_factor", "Coarse factor", unit="", field=False, default=1,
              description=
              "1 = seed every node. k > 1 runs the search AND a full 12-DOF solve on only "
              "every k-th node per axis, then interpolates displacement and gradient to the "
              "rest as the starting guess (a gradient seed is then written as `grad_*` too). "
              "A pure speed lever for SMOOTH fields on dense grids; a localised deformation "
              "between coarse nodes is missed in the seed and must be recovered by the match."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("init_guess", ["pyramid", "ncc", "zero", "previous"], default="pyramid",
             label="Seed",
             description="Where each node's starting displacement comes from.",
             choice_docs={
                 "pyramid":
                     "A coarse-to-fine NCC search: the radius is searched on a downsampled "
                     "volume and refined level by level — the default, and the one that copes "
                     "with large displacements at an affordable cost.",
                 "ncc":
                     "A single-level NCC search of the full radius at full resolution: exact "
                     "within the radius, but the cost grows with its cube, so only for small "
                     "motions or small grids.",
                 "zero":
                     "Start every node at zero displacement (after the global shift when "
                     "that is on): free, and right only when the motion is under about a "
                     "quarter subset everywhere — otherwise the match never finds the feature.",
                 "previous":
                     "Start from the previous frame's field against the same reference "
                     "(falls back to pyramid on the first frame): the cheap, right choice for "
                     "a smooth time series where each frame moves a little further.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_STAGE_KAX,
    description="pyALDVC §3 as a node: the starting displacement of every grid node (global "
                "shift + pyramid/NCC search) written as disp_* on the grid rows, with seed_cc / "
                "seed_ok. Pull it first: a wrong seed is never recovered by the refinement.")

register_node(
    _compute_dvc_local_match, op_key="dvc.local_match", label="DVC Local Match",
    category="analysis", adds_columns=_columns_dvc_local_match, extra_layers=_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("icgn_max_iter", "Max iterations", unit="", field=False, default=100,
              description=
              "Hard cap on inverse-compositional Gauss-Newton iterations per subset. A node "
              "that hits it is reported with status 1 (max_iter) rather than silently "
              "accepted, so this is a safety valve: raise it only when many nodes report 1 "
              "and their ZNCC is still climbing; lower it to bound the run time on a poor "
              "match."),
        InInt("subset_stride", "Subset stride", unit="", field=False, default=1,
              description=
              "Sample only every k-th voxel along each axis inside a subset — k³ fewer "
              "voxels per iteration, pyALDVC measures about 5× faster at k=2 with a 32-voxel "
              "subset — at the cost of a slightly noisier match. Clamped so at least 5 "
              "samples per axis remain. Inherited by the global step so its local re-solves "
              "match this one."),
        InInt("tile_local", "Tile edge", unit="px", field=False, default=0,
              description=
              "Solve in blocks of nodes against a box of the volume this many voxels on an "
              "edge instead of holding the whole scan at once; 0 is OFF and right until "
              "memory is the problem (the reference's gradient bundle alone is 17 bytes per "
              "voxel)."),
        InInt("n_threads", "Threads", unit="", field=False, default=0,
              description=
              "Numba worker threads for the CPU solver; 0 = ALL cores. Lower it to leave "
              "headroom when several nodes pull in parallel. Inherited by the global step."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("interp_method", ["cubic", "bspline", "linear"], default="cubic", label="Interpolation",
             description="How the deformed volume is sampled at sub-voxel positions inside the match.",
             choice_docs={
                 "cubic":
                     "Cubic convolution: sub-voxel accurate at a moderate cost — the default, "
                     "and what the single DVC node uses.",
                 "bspline":
                     "Cubic B-spline on prefiltered coefficients: the smoothest gradients and "
                     "the lowest bias on a clean scan, at a higher prepare cost per frame.",
                 "linear":
                     "Trilinear: the fastest and the most biased (it under-estimates "
                     "sub-voxel motion near half-voxel offsets); for a quick look, not a "
                     "measurement.",
             }),
        Mode("backend", ["auto", "numba", "numpy", "cuda"], default="auto", label="Backend",
             description="Which implementation runs the per-subset solve.",
             choice_docs={
                 "auto": "CUDA when a usable device exists, else the numba CPU kernels, else "
                         "numpy — the right default everywhere.",
                 "numba": "The compiled multi-threaded CPU kernels (first call in a process "
                          "compiles, ~10-20 s, cached on disk afterwards).",
                 "numpy": "Pure numpy: no compilation, several times slower; for a machine "
                          "where numba cannot compile.",
                 "cuda": "The GPU local solver (needs al-dvc[gpu] and a CUDA device); refused "
                         "with a clear message when none is usable.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=_STAGE_KAX,
    description="pyALDVC §4 as a node: the 12-DOF IC-GN match of every subset from the seed — "
                "the cost centre. Writes sub-voxel disp_*, grad_*, qfactor (ZNCC), status, "
                "n_iter, bad, valid and disp_std_* on the grid rows.")

register_node(
    _compute_dvc_global_step, op_key="dvc.global_step", label="DVC Global Step",
    category="analysis", adds_columns=_columns_dvc_global_step, extra_layers=_layers_on_source,
    reads_domains=frozenset({Domain.VOXEL, Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InFloat("mu", "AL penalty μ", unit="", field=False, default=1e-3,
                description=
                "Augmented-Lagrangian penalty weighting the constraint that each subset's own "
                "displacement equals the globally compatible field. HIGHER enforces agreement "
                "harder (smoother, but a real discontinuity is smeared); LOWER lets subsets stay "
                "independent (the field converges toward plain local DVC). pyALDVC's default "
                "is 1e-3; change it by factors of ten."),
        InFloat("beta", "Compatibility β", unit="", field=False, default=0.0,
                description=
                "Weight of the compatibility term of the global solve. 0 = AUTO: the solver "
                "sweeps candidates and picks the L-curve corner once per reference, reusing it "
                "for every frame against that reference — the documented default. A positive "
                "value pins it, which is how two runs are compared on equal terms."),
        InInt("admm_iterations", "ADMM iterations", unit="", field=False, default=4,
              description=
              "Passes of the outer loop that alternates the 3-DOF local re-solve with the "
              "global solve; the published guidance is that 3-5 suffice, and the loop stops "
              "early once the update falls under pyALDVC's tolerance. 1 runs the global solve "
              "once with no re-solve (no refreshed qfactor/status). MORE costs one local pass "
              "each."),
    ],
    outputs=[OutDataset()],
    granularity=_STAGE_GRAN, kernel_axes=_STAGE_KAX,
    description="pyALDVC §5-6 as a node: the global compatibility step + ADMM iterations that "
                "reconcile the matched subsets into one kinematically compatible field. "
                "Overwrites disp_*/grad_*, writes outlier; β auto-tuned per reference.")

register_node(
    _compute_dvc_strain, op_key="dvc.strain", label="DVC Strain", category="analysis",
    adds_columns=_columns_dvc_strain, extra_layers=_layers_on_source,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        _SOURCE,
        InInt("strain_halfwidth", "Fit half-width", unit="", field=False, default=1,
              description=
              "Half-width in NODES of the window the plane-fit method differentiates over: 1 "
              "is a 3×3×3 neighbourhood. WIDER is pyALDVC's first suggestion for a speckled "
              "strain map — it averages more nodes and so lowers the strain noise, at the cost "
              "of smearing a strain gradient over that many nodes. Only read by `plane_fit`."),
        InFloat("strain_smooth", "Strain smoothing σ", unit="", field=False, default=0.0,
                description=
                "Gaussian σ in NODE units applied when strain is computed; 0 is off and is the "
                "principled default here: the global step already produced a compatible field, "
                "so strain is an intrinsic property of it. Use it only on a plainly local run."),
        InFloat("disp_smoothing", "Displacement smoothing σ", unit="", field=False, default=0.0,
                description=
                "Gaussian σ in NODE units applied to the displacement field BEFORE it is "
                "differentiated — the displacement columns themselves are untouched; 0 is off "
                "and right by default (the displacement is what was measured)."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("strain_method", ["plane_fit", "fem", "fd", "direct"], default="plane_fit",
             label="Method",
             description="How the displacement gradient is obtained from the grid field.",
             choice_docs={
                 "plane_fit":
                     "A least-squares plane through each node's neighbourhood (Fit half-width "
                     "wide): robust to a noisy node, the pyALDVC default.",
                 "fem":
                     "The hex-element FEM gradient of the global step's own operators: "
                     "consistent with the compatibility solve, slightly smoother than the "
                     "plane fit, and the operators are rebuilt here (a few seconds).",
                 "fd":
                     "Central finite differences between neighbouring nodes: the sharpest "
                     "and the noisiest — for a dense, clean grid.",
                 "direct":
                     "The solver's own per-subset deformation gradient (`grad_*`), no "
                     "differentiation across nodes at all: the most local, only meaningful "
                     "against a FIXED reference (an increment's gradient is not a strain); "
                     "falls back to the plane fit otherwise.",
             }),
        Mode("strain_type", ["infinitesimal", "green_lagrange", "euler_almansi", "hencky"],
             default="infinitesimal", label="Measure",
             description="Which strain measure the nine strain_* columns report.",
             choice_docs={
                 "infinitesimal": "½(F + Fᵀ) − I: the small-strain measure, linear in the "
                                  "gradient; right under a few percent of strain.",
                 "green_lagrange": "½(FᵀF − I): finite strain referred to the UNDEFORMED "
                                   "configuration — the Lagrangian choice for large gels.",
                 "euler_almansi": "½(I − F⁻ᵀF⁻¹): finite strain referred to the DEFORMED "
                                  "configuration.",
                 "hencky": "½ ln(FᵀF): the logarithmic (true) strain, additive across "
                           "successive deformations.",
             }),
    ],
    granularity=_STAGE_GRAN, kernel_axes=frozenset(),
    description="pyALDVC §8 as a node: strain from the grid's displacement field — nine strain_* "
                "columns under the chosen measure — without reading a voxel, so a new measure or "
                "smoothing re-runs in a second.")


# ── the bundle: what the palette places ─────────────────────────────────────────

_DX = 270.0
register_bundle(Bundle(
    key="bundle:dvc", label="DVC (pyALDVC) · staged", category="analysis",
    role="motion_and_deformation_fields",
    description=
    "pyALDVC as five cards, wired in order and framed as one region: Grid → Initial Guess → "
    "Local Match → Global Step → Strain. Pull any card to run the pipeline up to that "
    "section and see the field as it stands; change a later card and only that section "
    "re-runs. The same solver as the single DVC (pyALDVC) node, section for section.",
    nodes=(
        BundleNode("dvc_grid", "dvc.grid", 0.0, 0.0),
        BundleNode("dvc_seed", "dvc.initial_guess", _DX, 0.0),
        BundleNode("dvc_local", "dvc.local_match", 2 * _DX, 0.0),
        BundleNode("dvc_global", "dvc.global_step", 3 * _DX, 0.0),
        BundleNode("dvc_strain", "dvc.strain", 4 * _DX, 0.0),
    ),
    edges=(
        BundleEdge("dvc_grid", "out", "dvc_seed", "data"),
        BundleEdge("dvc_seed", "out", "dvc_local", "data"),
        BundleEdge("dvc_local", "out", "dvc_global", "data"),
        BundleEdge("dvc_global", "out", "dvc_strain", "data"),
    ),
    input=("dvc_grid", "data"), output=("dvc_strain", "out"),
    frame_title="DVC (pyALDVC)",
))
