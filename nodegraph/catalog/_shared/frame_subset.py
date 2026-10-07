"""frame_subset — carrying everything that rides ALONGSIDE an image through an M/T/Z subset.

Narrowing a frame axis is three jobs, not one. The image is the easy one:
:class:`~nodegraph.provider.FrameSubsetProvider` is a pure index remap and no pixel moves.
The other two are where a subset goes quietly wrong, and they are the reason this module
exists rather than a `np.take` at each call site:

* **lattice layers** (a mask, a per-plane statistic) are indexed on the same axes as the
  image, so they must be subset WITH it — otherwise ``reshaped_axes`` drops them for no
  longer fitting and a segmentation silently disappears one node after it was computed;
* **structure rows** (Point / Label / Track / Mesh) carry ``m``/``t``/``z`` ADDRESSES, so
  rows on a dropped frame must go and the survivors must be RENUMBERED. A row still claiming
  ``t=9`` in a 3-frame output is an out-of-range read waiting to happen, and it will happen
  somewhere else entirely.

Both were written for ``util.crop``'s frames mode (V2.27) and lived inside that node until
``util.select_group`` (2026-09-15) needed the identical behaviour for a group selection.
They are here, not there, because a second copy of the structure-renumbering rule is exactly
the kind of thing that gets fixed in one place and not the other — and the failure it would
cause (a Point table addressing the wrong position) is invisible until someone reads a
spreadsheet.

Qt-free; numpy only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, is_lattice

#: The frame axes a subset may narrow, in canonical order.
FRAME_AXES: Tuple[str, ...] = ("m", "t", "z")

#: What each frame axis is called in a message. A user picked "positions", not "m".
AXIS_NOUN: Dict[str, str] = {"m": "position", "t": "timepoint", "z": "plane"}

Picks = Dict[str, Optional[Tuple[int, ...]]]


def subset_lattice_layers(ds: Dataset, new_axes: Any, keep: Picks) -> Dataset:
    """Move ``ds`` onto ``new_axes``, reindexing every lattice layer onto the kept indices.

    Generic over the domain: :meth:`AxisSizes.axis_list` says which axes the layer's array
    has and in what order, so a Voxel mask (m,t,z,c,y,x), a Plane statistic (m,t,z) and a
    Timepoint series (t) are all handled by the same two lines with no per-domain table —
    and a domain added later is handled for free.

    The arrays are all cut BEFORE the axis change and written back after it, which is the
    only order that works: ``with_attribute`` validates a lattice layer's shape against the
    Dataset's CURRENT axes, so a subset written too early is rejected and the original left
    in place is dropped by ``reshaped_axes`` a line later. A layer that already disagrees
    with the input axes is not touched at all — it was stale before this ran, and
    ``reshaped_axes`` is the right place for it to go.

    **An axis the output ELIDES is squeezed out, not left at size 1** (V3.01). The batch
    axis is absent from a layer's shape once ``b == 1`` (see
    :meth:`~nodegraph.dataset.AxisSizes.axis_list`), so narrowing a K-member batch down to
    one member has to remove that axis rather than keep a length-1 stub — otherwise every
    layer is one rank too deep for the axes it is about to be validated against. Expressed
    as a diff between the input's and the output's own axis lists, so it needs no mention
    of ``b`` and covers any axis that gains the same treatment later."""
    subset = []
    for attr in ds.attributes.values():
        if not is_lattice(attr.domain):
            continue
        arr = np.asarray(attr.values)
        if tuple(arr.shape) != ds.axes.shape_for(attr.domain):
            continue
        axes_in = ds.axes.axis_list(attr.domain)
        for pos, axis in enumerate(axes_in):
            idx = keep.get(axis)
            if idx is not None:
                arr = np.take(arr, list(idx), axis=pos)
        dropped = tuple(a for a in axes_in
                        if a not in new_axes.axis_list(attr.domain))
        for axis in dropped:
            pos = axes_in.index(axis)
            if arr.shape[pos] == 1:
                arr = np.squeeze(arr, axis=pos)
                axes_in = axes_in[:pos] + axes_in[pos + 1:]
        subset.append((attr, arr))
    out = ds.reshaped_axes(new_axes)
    for attr, arr in subset:
        out = out.with_layer(attr.domain, attr.name, arr, attr.layer)
    return out


def subset_structure_rows(ds: Dataset, keep: Picks) -> Dataset:
    """Filter and renumber structure rows onto the kept m/t/z indices.

    A structure table lives on the Dataset as one attribute layer per COLUMN, keyed by
    ``(domain, layer)``, so a group is reassembled here, masked as a unit, and written back.
    A row is kept when its address survives on every subset axis, and its address is then
    the POSITION of that index within the picks — the same remap the image view performs
    (:func:`~nodegraph.provider.subset_index` is its inverse). A sub-pixel ``z`` keeps its
    fractional offset within its plane; which plane it is in is decided by rounding, because
    that is the plane the voxel data itself was subset by.

    **Mesh is different and is dropped instead**, unless nothing about it moves. Its three
    buckets (element / vertex / face) are joined by CSR ranges and dense per-frame ids that
    ``nodegraph.mesh`` validates strictly, so filtering the element bucket alone would leave
    every vertex pointing at the wrong element — a corrupted mesh that raises somewhere else
    later. When no element row is dropped and Z is untouched, there is nothing to rebuild and
    only the frame ADDRESSES are remapped, which covers the ordinary "keep the positions I
    care about" case; anything more is a rebuild this has no business doing silently, so the
    mesh layers go and the user re-runs the meshing after the subset."""
    if all(v is None for v in keep.values()):
        return ds
    groups: Dict[Tuple[Domain, Optional[str]], Dict[str, Any]] = {}
    for attr in ds.attributes.values():
        if not is_lattice(attr.domain):
            groups.setdefault((attr.domain, attr.layer), {})[attr.name] = attr
    out = ds
    for (domain, layer), cols in groups.items():
        shapes = {np.asarray(a.values).shape for a in cols.values()}
        if len(shapes) != 1 or len(shapes.copy().pop()) != 1:
            continue           # not one row per element (a mesh CSR bucket, a 2-D column):
                               # there is no row to filter, so leave it exactly as it is
        n = int(shapes.pop()[0])
        mask = np.ones(n, dtype=bool)
        moves = []
        for axis in FRAME_AXES:
            idx, col = keep.get(axis), cols.get(axis)
            if idx is None or col is None:
                continue
            vals = np.asarray(col.values)
            plane = np.rint(vals.astype(float)).astype(np.int64)
            order = {int(v): i for i, v in enumerate(idx)}
            inside = np.array([int(p) in order for p in plane], dtype=bool)
            mask &= inside
            moves.append((axis, col, vals, plane, order))
        if domain is Domain.MESH and (not mask.all() or keep.get("z") is not None):
            for name in cols:
                out = out.without(domain, name, layer)
            continue
        for _axis, col, vals, plane, order in moves:
            moved = (np.array([order[int(p)] for p in plane[mask]], dtype=float)
                     + (vals[mask].astype(float) - plane[mask].astype(float)))
            out = out.with_layer(domain, col.name, moved.astype(vals.dtype), layer)
        readdressed = {col.name for _a, col, _v, _p, _o in moves}
        if not mask.all():
            for name, col in cols.items():
                if name not in readdressed:
                    out = out.with_layer(domain, name,
                                         np.asarray(col.values)[mask], layer)
    return out


__all__ = ["FRAME_AXES", "AXIS_NOUN", "Picks",
           "subset_lattice_layers", "subset_structure_rows"]


def remap_lattice_layers(ds: Dataset, axis: str, sources) -> Dataset:
    """Re-index every lattice layer along ``axis`` onto ``sources`` — output index ``k`` takes
    the layer's slice ``sources[k]``, repeats allowed, ``None`` a ZERO slice — with the axes
    unchanged (2026-10-07, for ``util.time_shift``). The companion of
    :class:`~nodegraph.provider.FrameRemapProvider`: a mask has to move frame for frame with
    the image it was computed on, or a segmentation ends up two frames out of step with its
    pixels. A layer whose shape already disagrees with the Dataset is left alone, as
    :func:`subset_lattice_layers` leaves it."""
    idx = [0 if s is None else int(s) for s in sources]
    blanks = [k for k, s in enumerate(sources) if s is None]
    out = ds
    for attr in list(ds.attributes.values()):
        if not is_lattice(attr.domain):
            continue
        arr = np.asarray(attr.values)
        if tuple(arr.shape) != ds.axes.shape_for(attr.domain):
            continue
        axes_in = ds.axes.axis_list(attr.domain)
        if axis not in axes_in:
            continue
        pos = axes_in.index(axis)
        moved = np.take(arr, idx, axis=pos)
        if blanks:
            sl: list = [slice(None)] * moved.ndim
            sl[pos] = blanks
            moved[tuple(sl)] = 0
        out = out.with_layer(attr.domain, attr.name, moved, attr.layer)
    return out


def shift_structure_rows(ds: Dataset, axis: str, delta: int, size: int) -> Dataset:
    """Move every structure row's ``axis`` address by ``delta`` (2026-10-07, for
    ``util.time_shift``): a row landing outside ``[0, size)`` is DROPPED — the shifted stream
    has no frame for it — and the survivors keep their other columns. Rows are never
    duplicated onto a held edge frame: an object exists once. A mesh that would lose a row
    is dropped whole, for the reason :func:`subset_structure_rows` gives; one that keeps
    every row is only re-addressed. A sub-pixel fraction of the address is kept."""
    if int(delta) == 0:
        return ds
    groups: Dict[Tuple[Domain, Optional[str]], Dict[str, Any]] = {}
    for attr in ds.attributes.values():
        if not is_lattice(attr.domain):
            groups.setdefault((attr.domain, attr.layer), {})[attr.name] = attr
    out = ds
    for (domain, layer), cols in groups.items():
        col = cols.get(axis)
        if col is None:
            continue
        shapes = {np.asarray(a.values).shape for a in cols.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            continue           # not one row per element: nothing to move
        vals = np.asarray(col.values)
        moved = vals.astype(float) + float(delta)
        plane = np.rint(moved).astype(np.int64)
        mask = (plane >= 0) & (plane < int(size))
        if domain is Domain.MESH and not mask.all():
            for name in cols:
                out = out.without(domain, name, layer)
            continue
        out = out.with_layer(domain, col.name, moved[mask].astype(vals.dtype), layer)
        if not mask.all():
            for name, c in cols.items():
                if name != axis:
                    out = out.with_layer(domain, name, np.asarray(c.values)[mask], layer)
    return out
