"""analysis_mesh — the correlation GRID of a DIC / PIV / DVC result as a table the Viewer
draws over the image (2026-10-09).

A correlation field is a Point table of node positions + displacement; the elements that
join those nodes — pyALDIC's (adaptively refined) quadrilaterals, PIV's lattice cells,
pyALDVC's hexahedra — were thrown away in the kernel adapters, so the image could show the
vectors but never the grid they were measured on. This module keeps it, in ONE shape for all
three: every element kind here is an **axis-aligned box in the reference configuration**
(pyALDIC's quadtree only ever halves squares; a PIV cell spans adjacent window centres; an
ALDVC hex spans adjacent grid nodes), so an element row is::

    id, m, t, c, z, y, x          the invariant schema — (y, x) the element's centre, z its
                                  plane (2D) or centre (3D)
    y_lo, y_hi, x_lo, x_hi        its box in image pixels
    z_lo, z_hi                    its z extent (the plane, twice, for a 2-D element)
    size_px                       its edge length
    level                         refinement depth: 0 = the base grid, 1 = halved once, …
    n0 … n7                       (DIC only) the Q8 node indices into the unit's own node
                                  list, ``-1`` = no hanging mid-side node — what a staged
                                  stage needs to rebuild pyALDIC's mesh

``t = -1`` marks a mesh shared by EVERY frame of its unit (the common case: one reference
mesh per position), written once instead of once per frame — a 96-position DVC run would
otherwise carry its ~12 000 hexes 11 times over. The table sits on its own Point layer,
``<field layer>_mesh``, and the ``__analysis_mesh__`` metadata names it so the Viewer draws it
as a grid and the Points overlay leaves it alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: metadata key: ``{mesh_layer: {"field": field_layer, "kind": "quad" | "hex"}}``
MESH_MARK = "__analysis_mesh__"

#: the columns every element row carries beyond the invariant schema
BOX_COLUMNS = ("y_lo", "y_hi", "x_lo", "x_hi", "z_lo", "z_hi", "size_px", "level")
NODE_COLUMNS = tuple(f"n{i}" for i in range(8))


def mesh_layer_name(field_layer: str) -> str:
    return f"{field_layer}_mesh"


def mesh_columns(kind: str) -> tuple:
    """The columns a ``kind`` mesh table carries (for ``adds_columns`` declarations)."""
    base = ("id", "m", "t", "c", "z", "y", "x") + BOX_COLUMNS
    return base + (NODE_COLUMNS if kind == "fem" else ())


def _mesh_levels(size: np.ndarray) -> np.ndarray:
    if size.size == 0:
        return np.zeros(0, dtype=np.int64)
    base = float(np.nanmax(size))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.round(np.log2(base / np.maximum(size, 1e-9)))
    return np.nan_to_num(lv, nan=0.0).astype(np.int64)


def _mesh_box_rows(m: int, t: int, c: int, y_lo, y_hi, x_lo, x_hi, z_lo,
                   z_hi) -> Dict[str, np.ndarray]:
    y_lo, y_hi = np.asarray(y_lo, float), np.asarray(y_hi, float)
    x_lo, x_hi = np.asarray(x_lo, float), np.asarray(x_hi, float)
    z_lo, z_hi = np.asarray(z_lo, float), np.asarray(z_hi, float)
    n = y_lo.size
    size = np.maximum(y_hi - y_lo, x_hi - x_lo)
    return {
        "m": np.full(n, int(m), dtype=np.int64), "t": np.full(n, int(t), dtype=np.int64),
        "c": np.full(n, int(c), dtype=np.int64),
        "z": (z_lo + z_hi) / 2.0, "y": (y_lo + y_hi) / 2.0, "x": (x_lo + x_hi) / 2.0,
        "y_lo": y_lo, "y_hi": y_hi, "x_lo": x_lo, "x_hi": x_hi, "z_lo": z_lo, "z_hi": z_hi,
        "size_px": size, "level": _mesh_levels(size),
    }


def fem_quads(coords_xy: np.ndarray, elements: np.ndarray, *, m: int, t: int, c: int,
              z_plane: int) -> Dict[str, np.ndarray]:
    """pyALDIC elements → rows. ``coords_xy`` ``(N, 2)`` ``[x, y]`` (pyALDIC's order),
    ``elements`` ``(E, 4 | 8)`` 0-based with ``-1`` for an absent mid-side node; the box is
    the bounding box of the four corners (columns 0-3), which is exact for a quadtree."""
    xy = np.asarray(coords_xy, dtype=float).reshape(-1, 2)
    el = np.asarray(elements, dtype=np.int64)
    if el.ndim != 2 or el.shape[0] == 0 or xy.shape[0] == 0:
        return {}
    corners = el[:, :4]
    ok = (corners >= 0).all(axis=1) & (corners < xy.shape[0]).all(axis=1)
    el, corners = el[ok], corners[ok]
    cx, cy = xy[corners, 0], xy[corners, 1]
    z = np.full(len(el), float(z_plane))
    out = _mesh_box_rows(m, t, c, cy.min(axis=1), cy.max(axis=1), cx.min(axis=1),
                         cx.max(axis=1), z, z)
    q8 = np.full((len(el), 8), -1, dtype=np.int64)
    q8[:, :min(8, el.shape[1])] = el[:, :8]
    for i in range(8):
        out[f"n{i}"] = q8[:, i]
    return out


def lattice_quads(ys: np.ndarray, xs: np.ndarray, *, m: int, t: int, c: int, z_plane: int,
                  keep: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The cells of a regular ``(ny, nx)`` lattice of node centres (PIV window centres) —
    one box between each 2 × 2 block of neighbours. ``keep`` ``(ny, nx)`` bool drops cells
    with any corner outside it (excluded windows)."""
    ys, xs = np.asarray(ys, float).reshape(-1), np.asarray(xs, float).reshape(-1)
    if ys.size < 2 or xs.size < 2:
        return {}
    iy, ix = np.meshgrid(np.arange(ys.size - 1), np.arange(xs.size - 1), indexing="ij")
    iy, ix = iy.reshape(-1), ix.reshape(-1)
    if keep is not None:
        k = np.asarray(keep, bool)
        ok = k[iy, ix] & k[iy + 1, ix] & k[iy, ix + 1] & k[iy + 1, ix + 1]
        iy, ix = iy[ok], ix[ok]
    z = np.full(iy.size, float(z_plane))
    return _mesh_box_rows(m, t, c, ys[iy], ys[iy + 1], xs[ix], xs[ix + 1], z, z)


def grid_hexes(grid_shape: Sequence[int], coords_zyx: np.ndarray, *, m: int, t: int, c: int,
               valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The hexahedra of a regular ``(nz, ny, nx)`` node grid (pyALDVC: node
    ``n = iz·ny·nx + iy·nx + ix``) — one box per 2 × 2 × 2 block of neighbours, dropped when
    any corner is not ``valid``."""
    nz, ny, nx = (int(v) for v in grid_shape)
    if min(nz, ny, nx) < 2:
        return {}
    zyx = np.asarray(coords_zyx, dtype=float).reshape(nz, ny, nx, 3)
    iz, iy, ix = np.meshgrid(np.arange(nz - 1), np.arange(ny - 1), np.arange(nx - 1),
                             indexing="ij")
    iz, iy, ix = iz.reshape(-1), iy.reshape(-1), ix.reshape(-1)
    if valid is not None:
        v = np.asarray(valid, bool).reshape(nz, ny, nx)
        ok = np.ones(iz.size, bool)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ok &= v[iz + dz, iy + dy, ix + dx]
        iz, iy, ix = iz[ok], iy[ok], ix[ok]
    lo = zyx[iz, iy, ix]
    hi = zyx[iz + 1, iy + 1, ix + 1]
    return _mesh_box_rows(m, t, c, lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2], lo[:, 0], hi[:, 0])


def concat_rows(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    parts = [p for p in parts if p and len(p.get("m", ())) > 0]
    if not parts:
        return {}
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}


def with_mesh(ds: Any, field_layer: str, rows: Dict[str, np.ndarray], *, kind: str,
              z_kind: str, window: Optional[Sequence[float]] = None) -> Any:
    """``ds`` with the element table on ``<field_layer>_mesh`` and the marker metadata.
    ``window`` is the subset / interrogation window ``(y, x[, z])`` in pixels, which the
    Viewer's *subset boxes* style draws around the field's nodes."""
    from nodegraph.domains import Domain
    from nodegraph.structure import StructureTable
    if not rows:
        return ds
    rows = dict(rows)
    rows["id"] = np.arange(len(rows["m"]), dtype=np.int64)
    layer = mesh_layer_name(field_layer)
    # REPLACE, never merge: a PIV deform pass's finer grid (or a refined DIC mesh) must not
    # leave a coarser grid's columns behind on rows they no longer describe
    for a in list(ds.layers_on(Domain.POINT)):
        if a.layer == layer:
            ds = ds.without(Domain.POINT, a.name, layer)
    marks = dict(ds.metadata.get(MESH_MARK) or {})
    marks[layer] = {"field": field_layer, "kind": kind}
    if window is not None:
        marks[layer]["window"] = [float(w) for w in window]
    return (ds.with_structure(StructureTable(Domain.POINT, rows, layer=layer, z_kind=z_kind))
            .with_metadata(**{MESH_MARK: marks}))


__all__ = ["MESH_MARK", "BOX_COLUMNS", "NODE_COLUMNS", "mesh_layer_name", "mesh_columns",
           "fem_quads", "lattice_quads", "grid_hexes", "concat_rows", "with_mesh"]
