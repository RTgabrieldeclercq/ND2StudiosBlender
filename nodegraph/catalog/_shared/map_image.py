"""map image — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Callable, Optional

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.field import (
    Attr as FieldAttr,
    BinOp as FieldBinOp,
    Const as FieldConst,
    Input as FieldInput,
    UnaryOp as FieldUnaryOp,
    Where as FieldWhere,
)
from nodegraph.parallel import map_units
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

#: the Field IR node classes (``field.Field`` is a typing Union — not isinstance-able).
_FIELD_TYPES = (FieldConst, FieldAttr, FieldInput, FieldBinOp, FieldUnaryOp, FieldWhere)
def _kernel_field_varies(ctx: EvalContext) -> bool:
    """True iff a ``kernel_param`` socket is wired to a **non-Const** Field — a spatially
    varying kernel (the gate defers to `_FIELD_TYPES` defined below, so it is checked
    lazily). A varying kernel breaks a tile's translation-invariance and its halo sizing
    (the halo assumes one radius for the whole plane), so the consumer must stream at the
    WHOLE_PLANE unit, never tiled (V2.04 §6b Fork B — the kernel-param field gate). A
    Const field is spatially uniform, so it stays tileable."""
    spec = ctx.spec
    if spec is None:
        return False
    for s in getattr(spec, "inputs", ()):
        if not getattr(s, "kernel_param", False):
            continue
        payload = ctx.input(s.name)
        if isinstance(payload, _FIELD_TYPES) and not isinstance(payload, FieldConst):
            return True
    return False
def _map_image(ctx: EvalContext, ds: Dataset,
               plane_fn: Callable[[np.ndarray], np.ndarray],
               volume_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               halo: int = 0) -> Dataset:
    """Apply a spatial op across the image — **lazily** (C1 / V2.04): the returned
    Dataset carries a streaming compute provider instead of a realized array.

    In a 2D footprint ``plane_fn`` runs per ``(Y,X)`` unit — per canonical **tile**
    (window+``halo`` read from the base, overlap-recompute) when the resolved
    granularity is ``TILEABLE``, per whole **plane** when it is ``WHOLE_PLANE`` (an op
    with a plane-global statistic/solver must declare WHOLE_PLANE — a tile would see
    the wrong population). In a 3D (``WHOLE_VOLUME``) footprint ``volume_fn`` runs on
    the ``(Z,Y,X)`` volume per ``(m,t,c)``, computed lazily per touched volume. A node
    resolving to ``WHOLE_VOLUME`` MUST supply a ``volume_fn`` — a *stack-of-2D* node
    keeps its 3D granularity at ``WHOLE_PLANE`` so it stays on the plane path (H15).

    ``halo`` is the op's full **influence radius** in px (V2.04 §2: gaussian
    ``int(4σ+0.5)``, median ``w//2``, open/close/tophat ``2·(w//2)``, DoG from the
    larger σ). Falls back to the pre-C1 eager whole-realize when there is no engine
    cache (a bare ``EvalContext``), an unsupported granularity, or an oversize unit
    (> half the cache budget — V2.04 §6b pin/bypass policy).
    """
    prov = ds.image
    if prov is None:
        raise ValueError(f"{ctx.op_key} needs an image provider on its input Dataset")
    ax = prov.axes
    volumetric = ctx.is_volume
    if volumetric and volume_fn is None:
        raise ValueError(f"{ctx.op_key}: resolved to WHOLE_VOLUME but no volume op given")
    cache = ctx.tiles
    # kernel-param field gate (V2.04 §6b Fork B): a non-Const Field on a kernel_param
    # socket varies the kernel spatially → drop TILEABLE to the plane unit (a tile's halo
    # assumes one radius for the whole plane; a varying kernel breaks that + translation
    # invariance). Const/absent ⇒ genuinely tileable.
    tileable = (ctx.granularity is Granularity.TILEABLE
                and not _kernel_field_varies(ctx))
    # NOT promoted to the plane unit when CUDA is live, though it is tempting: a 512² tile
    # is below `nodegraph.gpu`'s dispatch threshold, so a tiled node never reaches the card,
    # and computing whole planes instead would fix that. Measured, it backfires. `gpu.active()`
    # is a PROCESS-wide flag, but dispatch is decided PER OP by the equivalence gate — so a
    # node whose kernel is rejected (or unimplemented in cupyx) would get the plane unit and
    # still run on the CPU, losing the tiling that keeps a 512²+halo window inside this
    # machine's 64 MB L3. Measured on a 4096² top-hat: **0.14×**. Tileable nodes therefore
    # stay tiled on the CPU; the GPU serves the units that are already large (WHOLE_PLANE /
    # WHOLE_VOLUME granularity, EDT), which are the expensive ones anyway.
    # size the oversize-bypass check by the EFFECTIVE lazy unit: a true tile unit is
    # tiny regardless of plane size (review 2026-07-22 — sizing tileable ops by plane
    # bytes forced whole-series eager realization exactly when memory is scarce)
    if volumetric:
        unit_bytes = ax.z * ax.y * ax.x * 8
    elif tileable and 2 * (getattr(prov, "cum_halo", 0) + halo) < prov.tile:
        w = prov.tile + 2 * halo
        unit_bytes = w * w * 8
    else:                                        # plane unit (declared or fence-promoted)
        unit_bytes = ax.y * ax.x * 8
    lazy = (cache is not None
            and ctx.granularity in (Granularity.TILEABLE, Granularity.WHOLE_PLANE,
                                    Granularity.WHOLE_VOLUME)
            and unit_bytes <= cache.budget // 2)
    if not lazy:                                  # pre-C1 eager whole-realize path
        # Fanned out over units (V2.14). Each task owns a disjoint slice of `out`, and the
        # kernels are pure, so this is a pure scheduling change — byte-identical output.
        # Threads, not processes: `plane_fn`/`volume_fn` are closures built by the caller
        # and cannot be pickled, and the scipy filters underneath release the GIL.
        # V3.01: `b` joins the unit tuple rather than the allocation's leading shape,
        # because `shape_for` elides the batch axis at b == 1 — so an ordinary one-member
        # pull allocates and indexes exactly what it always did, and only a real batch
        # grows the array. The batch index is also what makes this fan out K times wider:
        # `map_units` already runs units in parallel, so K files' planes are K× the units
        # to spread over the same cores, with no scheduler change.
        nb = int(getattr(ax, "b", 1))
        batched = nb > 1
        shape = ((nb,) if batched else ()) + (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)
        out = np.zeros(shape, dtype=float)
        if volumetric:
            def do_volume(unit):
                b, m, t, c = unit
                vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x, b=b)
                res = volume_fn(vol.astype(float))
                if batched:
                    out[b, m, t, :, c] = res
                else:
                    out[m, t, :, c] = res

            map_units(do_volume, [(b, m, t, c) for b in range(nb) for m in range(ax.m)
                                  for t in range(ax.t) for c in range(ax.c)])
        else:
            def do_plane(unit):
                b, m, t, z, c = unit
                plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x, b=b)
                res = plane_fn(plane.astype(float))
                if batched:
                    out[b, m, t, z, c] = res
                else:
                    out[m, t, z, c] = res

            map_units(do_plane, [(b, m, t, z, c) for b in range(nb) for m in range(ax.m)
                                 for t in range(ax.t) for z in range(ax.z)
                                 for c in range(ax.c)])
        return ds.with_image(ArrayProvider(out))
    fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(), (), prov)
    if volumetric:
        return ds.with_image(VolumeComputeProvider(
            prov, lambda v, m, t, c: volume_fn(v), fp=fp, cache=cache))
    unit = "tile" if tileable else "plane"
    return ds.with_image(MapComputeProvider(
        prov, lambda a, m, t, z, c, gy0, gy1, gx0, gx1: plane_fn(a),
        halo=halo, unit=unit, fp=fp, cache=cache))
