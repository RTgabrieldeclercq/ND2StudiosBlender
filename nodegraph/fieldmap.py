"""fieldmap — a measured field drawn on its image, as plain numpy (Qt-free, testable).

Shared by the Viewer's overlays (``nodelab_v2``) and the ``plot.field`` figure node, so the
screen and an exported figure draw a field the same way; it lives in the engine because a
catalog node may not import the GUI (the seam, CLAUDE.md). Two overlays read a **field layer** (the Point table a DIC / PIV / DVC node, a staged card, Track
Field, Load Analysis or a ``field.*`` card wrote):

* the **Vectors** tab draws its displacement as arrows — :func:`vectors`;
* the **Field** tab colours the image by ANY numeric column of it (``disp_mag_um``,
  ``strain_eq``, ``jacobian``, ``von_mises`` …) — :func:`raster`, :func:`series_limits`.

Both used to live (twice) inside the painters, and both drew a µm displacement at pixel
positions, so every arrow was ``1 / pixel_size_um`` of its true length. Here every geometry
comes out in full-resolution IMAGE PIXELS — the units the rows' ``y`` / ``x`` are in — with
the µm → px conversion done once, from the payload's own ``pixel_size_um``.

Rasterising is cheap on purpose: it runs on the GUI thread once per frame change.

* A field whose nodes form a regular lattice (every correlation grid; holes allowed) is
  placed directly on its own ``(ny, nx)`` grid — no interpolation, microseconds — and drawn
  as one image over the lattice's extent, each node at the centre of its cell.
* A scattered field (tracked particles, a refined DIC mesh) is filled onto at most
  ``max_side`` (128) cells a side, none finer than half a node spacing, by a k-d tree
  nearest-neighbour query, with cells further than 1.5 node spacings from any node left
  empty — about 20 ms for 20 000 nodes. No triangulation runs on the GUI thread.
* A 3-D field on a lattice is interpolated between the two lattice planes that bracket the
  viewed z; a scattered 3-D field shows the nodes within one spacing of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain

#: columns that place or identify a row rather than measure anything
COORD_COLUMNS = frozenset({"id", "m", "t", "c", "z", "y", "x", "b", "track_id", "member_id",
                           "parent_id"})
#: the Field tab's Auto column, most derived first: a stress map beats a strain map beats a
#: volume map beats the raw displacement
AUTO_COLUMNS: Tuple[str, ...] = ("von_mises", "strain_eq", "jacobian", "curl", "curl_mag",
                                 "disp_mag_um", "speed")
#: displacement pairs in preference order, with the unit of their components
_DISP_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("disp_y", "disp_x", "um"), ("u_um", "v_um", "um"), ("u", "v", "px"), ("dy", "dx", "px"))
_VEL_PAIR = ("vy", "vx")
_DT_KEYS = ("piv_dt_s", "track_field_dt_s", "dt_s")
MESH_MARK = "__analysis_mesh__"


# ── reading tables ─────────────────────────────────────────────────────────────

def point_tables(ds: Any) -> Dict[str, Dict[str, np.ndarray]]:
    """Every Point table on ``ds`` except the correlation GRIDS (``<field>_mesh``), as
    ``{layer: {column: values}}``."""
    if ds is None or not hasattr(ds, "attributes"):
        return {}
    marks = (getattr(ds, "metadata", None) or {}).get(MESH_MARK) or {}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in ds.attributes.items():
        if dom is Domain.POINT and layer and layer not in marks:
            out.setdefault(str(layer), {})[str(name)] = attr.values
    return out


def _numeric(a: Any) -> bool:
    try:
        return np.issubdtype(np.asarray(a).dtype, np.number) and not np.issubdtype(
            np.asarray(a).dtype, np.bool_)
    except TypeError:
        return False


def data_columns(cols: Dict[str, np.ndarray]) -> List[str]:
    """The numeric columns that MEASURE something (coordinates and ids excluded), in table
    order."""
    return [k for k, v in cols.items() if k not in COORD_COLUMNS and _numeric(v)]


def layer_names(ds: Any) -> List[str]:
    """The Point tables the Field tab can colour by — at least one numeric data column."""
    return sorted(k for k, cols in point_tables(ds).items()
                  if "y" in cols and "x" in cols and data_columns(cols))


def vector_layer_names(ds: Any) -> List[str]:
    return sorted(k for k, cols in point_tables(ds).items() if _vector_pair(cols) is not None)


def column_names(ds: Any, layer: str) -> List[str]:
    return data_columns(point_tables(ds).get(str(layer or ""), {}))


def auto_pick(ds: Any, layer: str = "", column: str = "") -> Tuple[str, str]:
    """``(layer, column)`` the Field tab draws: the user's picks where they exist on this
    payload, else the most derived quantity available (:data:`AUTO_COLUMNS`), else the
    first data column. ``("", "")`` when there is nothing to draw."""
    tables = point_tables(ds)
    names = layer_names(ds)
    if not names:
        return "", ""
    lays = [layer] if layer in names else names
    for lay in lays:
        cols = data_columns(tables[lay])
        if column and column in cols:
            return lay, column
    for want in AUTO_COLUMNS:
        for lay in lays:
            if want in tables[lay] and _numeric(tables[lay][want]):
                return lay, want
    lay = lays[0]
    cols = data_columns(tables[lay])
    return (lay, cols[0]) if cols else ("", "")


def column_unit(ds: Any, layer: str, column: str) -> str:
    """The column's unit as the producer recorded it, else by the naming convention; ``""``
    when dimensionless or unknown."""
    try:
        from nodegraph.units import format_unit, recorded_unit, unit_of_column
        md = getattr(ds, "metadata", None) or {}
        u = recorded_unit(md, Domain.POINT, column, layer)
        if u is None:
            zk = ds.structure_zkind(Domain.POINT, layer) if hasattr(ds, "structure_zkind") else None
            u = unit_of_column(column, z_kind=zk)
        return format_unit(u, pretty=True) if u else ""
    except Exception:  # noqa: BLE001 — a unit label is advisory
        return ""


def _z_kind(ds: Any, layer: str) -> Optional[str]:
    try:
        return ds.structure_zkind(Domain.POINT, layer)
    except Exception:  # noqa: BLE001
        return None


def _frame_rows(cols: Dict[str, np.ndarray], m: int, t: int, c: int) -> np.ndarray:
    """Rows on the viewed (m, t, c). A field measured on ONE channel is shown whichever
    channel is viewed — the displacement applies to the whole frame, and an overlay that
    vanished on switching channels read as a broken one."""
    n = len(cols["y"])
    keep = np.ones(n, dtype=bool)
    for key, want in (("m", m), ("t", t)):
        if key in cols:
            keep &= np.asarray(cols[key]) == want
    if "c" in cols:
        cc = np.asarray(cols["c"])
        on = keep & (cc == c)
        if not on.any():
            chans = np.unique(cc[keep])
            on = keep & (cc == chans[0]) if chans.size == 1 else on
        keep = on
    return np.flatnonzero(keep)


def _plane_rows(ds: Any, layer: str, cols: Dict[str, np.ndarray], idx: np.ndarray,
                z: int) -> Tuple[np.ndarray, Optional[float]]:
    """Restrict frame rows to the viewed plane: equality for a per-plane 2-D field; for a
    3-D field the rows of the nearest node plane, returning that plane's z too."""
    if idx.size == 0 or "z" not in cols:
        return idx, None
    zz = np.asarray(cols["z"], dtype=float)[idx]
    if _z_kind(ds, layer) != "subpixel":
        return idx[np.rint(zz) == z], None
    planes = np.unique(np.round(zz, 6))
    near = planes[np.argmin(np.abs(planes - z))]
    return idx[np.abs(zz - near) < 1e-6], float(near)


# ── vectors ────────────────────────────────────────────────────────────────────

def _vector_pair(cols: Dict[str, np.ndarray]) -> Optional[Tuple[str, str, str]]:
    for a, b, unit in _DISP_PAIRS:
        if a in cols and b in cols:
            return a, b, unit
    if _VEL_PAIR[0] in cols and _VEL_PAIR[1] in cols:
        return _VEL_PAIR[0], _VEL_PAIR[1], "um/s"
    return None


def _px_um(ds: Any) -> Optional[float]:
    try:
        v = float((getattr(ds, "metadata", None) or {}).get("pixel_size_um") or 0)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _to_px_factor(ds: Any, unit: str) -> float:
    """Multiply a vector component in ``unit`` by this to get image pixels."""
    if unit == "px":
        return 1.0
    px = _px_um(ds) or 1.0
    if unit == "um":
        return 1.0 / px
    md = getattr(ds, "metadata", None) or {}
    dt = next((float(md[k]) for k in _DT_KEYS if md.get(k)), 1.0)
    return dt / px                     # a velocity drawn as the motion over one frame


@dataclass
class VectorField:
    rows: np.ndarray                   # (N, 4) y, x, dy_px, dx_px on the viewed plane
    layer: str
    spacing_px: float                  # node spacing of the whole table (for auto scale)
    mag_hi: float                      # 99th-percentile |d| over the WHOLE series, px
    p95: float                         # 95th-percentile |d| over the series, px


def vectors(ds: Any, m: int, t: int, z: int, c: int, *, layer: str = ""
            ) -> Optional[VectorField]:
    """The arrows of the viewed plane, in image pixels, from ``layer`` (or the first table
    carrying a displacement or velocity)."""
    tables = point_tables(ds)
    order = [layer] if layer in tables else sorted(tables)
    for lay in order:
        cols = tables[lay]
        pair = _vector_pair(cols)
        if pair is None or "y" not in cols or "x" not in cols:
            continue
        k = _to_px_factor(ds, pair[2])
        uy = np.asarray(cols[pair[0]], dtype=float) * k
        ux = np.asarray(cols[pair[1]], dtype=float) * k
        idx = _frame_rows(cols, m, t, c)
        idx, _zp = _plane_rows(ds, lay, cols, idx, z)
        if idx.size == 0:
            continue
        y = np.asarray(cols["y"], dtype=float)
        x = np.asarray(cols["x"], dtype=float)
        mag = np.hypot(uy, ux)
        fin = mag[np.isfinite(mag)]
        hi = float(np.percentile(fin, 99)) if fin.size else 0.0
        p95 = float(np.percentile(fin, 95)) if fin.size else 0.0
        return VectorField(np.stack([y[idx], x[idx], uy[idx], ux[idx]], axis=1), lay,
                           _spacing(np.stack([y[idx], x[idx]], axis=1)), hi, p95)
    return None


#: nodes sampled for the spacing estimate — a median, so a couple of thousand is exact enough
#: and keeps a 20k-node tracked field inside the GUI thread's budget
_SPACING_SAMPLE = 2000


def _spacing(yx: np.ndarray, tree=None) -> float:
    """Median nearest-neighbour distance of ``yx`` (from a uniform sample of the nodes)."""
    yx = yx[np.isfinite(yx).all(axis=1)]
    if len(yx) < 2:
        return float("nan")
    if tree is None:
        from scipy.spatial import cKDTree   # lazy
        tree = cKDTree(yx)
    pick = np.linspace(0, len(yx) - 1, min(len(yx), _SPACING_SAMPLE)).astype(np.intp)
    d, _ = tree.query(yx[pick], k=2)
    d = np.asarray(d)[:, 1]
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float("nan")


def auto_scale(vf: VectorField) -> float:
    """The arrow multiplier that draws the series' 95th-percentile vector one node spacing
    long — long enough to read, short enough not to cross the next arrow."""
    if not (np.isfinite(vf.spacing_px) and vf.p95 > 0):
        return 1.0
    return float(vf.spacing_px / vf.p95)


# ── the Field tab's raster ─────────────────────────────────────────────────────

@dataclass
class FieldRaster:
    grid: np.ndarray                   # (ny, nx) float, NaN = nothing to draw
    extent: Tuple[float, float, float, float]   # y0, y1, x0, x1 in image pixels
    lattice: bool                      # True: one cell per node; False: resampled
    layer: str
    column: str


def _lattice(coords: np.ndarray):
    from nodegraph.kernels.field_mechanics import lattice_of   # lazy: scipy-free import
    return lattice_of(coords)


def raster(ds: Any, layer: str, column: str, m: int, t: int, z: int, c: int, *,
           max_side: int = 128) -> Optional[FieldRaster]:
    """``column`` of ``layer`` on the viewed plane as a small grid plus its image extent."""
    cols = point_tables(ds).get(layer)
    if not cols or column not in cols or "y" not in cols or "x" not in cols:
        return None
    vals_all = np.asarray(cols[column], dtype=float)
    idx = _frame_rows(cols, m, t, c)
    if idx.size == 0:
        return None
    y_all = np.asarray(cols["y"], dtype=float)
    x_all = np.asarray(cols["x"], dtype=float)
    if _z_kind(ds, layer) == "subpixel" and "z" in cols:
        z_all = np.asarray(cols["z"], dtype=float)
        lat = _lattice(np.stack([z_all[idx], y_all[idx], x_all[idx]], axis=1))
        if lat is not None and lat.shape[0] > 1:
            return _slice3d(lat, idx, z_all, y_all, x_all, vals_all, z, layer, column)
        idx = idx[np.abs(z_all[idx] - z) <= max(1.0, 1.5 * _spacing(
            np.stack([y_all[idx], x_all[idx]], axis=1)) / 1.0)]
    elif "z" in cols:
        idx = idx[np.rint(np.asarray(cols["z"], dtype=float)[idx]) == z]
    if idx.size == 0:
        return None
    y, x, v = y_all[idx], x_all[idx], vals_all[idx]
    lat = _lattice(np.stack([y, x], axis=1)) if idx.size >= 2 else None
    if lat is not None and lat.shape[0] > 1 and lat.shape[1] > 1:
        g = np.full(lat.shape, np.nan)
        g[lat.index] = v
        hy, hx = lat.step
        return FieldRaster(g, (y.min() - hy / 2, y.max() + hy / 2,
                               x.min() - hx / 2, x.max() + hx / 2), True, layer, column)
    return _scatter(y, x, v, max_side, layer, column)


def _slice3d(lat, idx, z_all, y_all, x_all, vals, z, layer, column) -> Optional[FieldRaster]:
    nz, ny, nx = lat.shape
    hz, hy, hx = lat.step
    vol = np.full(lat.shape, np.nan)
    vol[lat.index] = vals[idx]
    z0 = float(z_all[idx].min())
    k = (float(z) - z0) / hz
    if k < -0.5 or k > nz - 0.5:
        return None
    k0 = int(np.clip(np.floor(k), 0, nz - 1))
    k1 = min(nz - 1, k0 + 1)
    f = float(np.clip(k - k0, 0.0, 1.0))
    a, b = vol[k0], vol[k1]
    g = np.where(np.isfinite(a) & np.isfinite(b), (1 - f) * a + f * b,
                 np.where(f < 0.5, a, b))
    yy, xx = y_all[idx], x_all[idx]
    if ny < 2 or nx < 2:
        return None
    return FieldRaster(g, (yy.min() - hy / 2, yy.max() + hy / 2,
                           xx.min() - hx / 2, xx.max() + hx / 2), True, layer, column)


def _scatter(y, x, v, max_side, layer, column) -> Optional[FieldRaster]:
    ok = np.isfinite(y) & np.isfinite(x)
    y, x, v = y[ok], x[ok], v[ok]
    if y.size < 3:
        return None
    from scipy.spatial import cKDTree   # lazy
    pts = np.stack([y, x], axis=1)
    tree = cKDTree(pts)
    s = _spacing(pts, tree)
    if not np.isfinite(s):
        return None
    y0, y1 = y.min() - s / 2, y.max() + s / 2
    x0, x1 = x.min() - s / 2, x.max() + s / 2
    span = max(y1 - y0, x1 - x0)
    # no finer than half a node spacing: finer cells only repeat the nearest node's value,
    # and the grid query is the cost (Qt's bilinear scaling does the smoothing)
    cell = max(span / max(8, int(max_side)), s / 2.0)
    ny = max(2, int(np.ceil((y1 - y0) / cell)))
    nx = max(2, int(np.ceil((x1 - x0) / cell)))
    gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    Y, X = np.meshgrid(gy, gx, indexing="ij")
    d, i = tree.query(np.stack([Y.ravel(), X.ravel()], axis=1), k=1,
                      distance_upper_bound=1.5 * s)
    i = np.asarray(i)
    far = ~np.isfinite(np.asarray(d))                 # beyond the bound: i == len(v)
    g = np.asarray(v, dtype=float)[np.where(far, 0, i)]
    g[far] = np.nan
    return FieldRaster(g.reshape(ny, nx), (y0, y1, x0, x1), False, layer, column)


def series_limits(ds: Any, layer: str, column: str) -> Tuple[float, float]:
    """1st / 99th percentile of ``column`` over EVERY frame of the table — the default
    limits, so a colour means the same value on every frame."""
    cols = point_tables(ds).get(layer) or {}
    a = np.asarray(cols.get(column, []), dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return 0.0, 1.0
    lo, hi = np.percentile(a, [1.0, 99.0])
    return float(lo), float(hi if hi > lo else lo + 1e-9)


def value_at(fr: Optional[FieldRaster], y: float, x: float) -> Optional[float]:
    """The raster's value under image pixel ``(y, x)`` (the hover readout)."""
    if fr is None:
        return None
    y0, y1, x0, x1 = fr.extent
    ny, nx = fr.grid.shape
    if not (y0 <= y < y1 and x0 <= x < x1):
        return None
    i = int((y - y0) / (y1 - y0) * ny)
    j = int((x - x0) / (x1 - x0) * nx)
    v = float(fr.grid[min(i, ny - 1), min(j, nx - 1)])
    return v if np.isfinite(v) else None


# ── tensor glyphs and the deformed grid (2026-10-09) ──────────────────────────

@dataclass
class Glyphs:
    rows: np.ndarray                   # (K, 5) y, x, theta (rad, image axes), p1, p2
    scale: float                       # image px per unit of the tensor (half-bar length)


def tensor_glyphs(ds: Any, layer: str, prefix: str, m: int, t: int, z: int, c: int, *,
                  every: int = 1) -> Optional[Glyphs]:
    """Principal axes of the in-plane ``<prefix>_yy/yx/xy/xx`` tensor (a Strain or Stress
    card's columns; the in-plane part of a 3-D one) at every ``every``-th node of the viewed
    plane: the angle of the largest principal axis in IMAGE axes (``atan2(dy, dx)``, y
    down) and both principal values. ``scale`` makes the series' 95th-percentile |value| a
    half-bar of 0.45 node spacings, so bars never reach the next node and mean the same
    length on every frame."""
    cols = point_tables(ds).get(layer) or {}
    names = [f"{prefix}_{a}" for a in ("yy", "yx", "xy", "xx")]
    if not all(n in cols for n in names) or "y" not in cols:
        return None
    T = np.stack([np.asarray(cols[n], dtype=float) for n in names], axis=1).reshape(-1, 2, 2)
    T = 0.5 * (T + np.swapaxes(T, 1, 2))
    ok = np.isfinite(T).all(axis=(1, 2))
    if not ok.any():
        return None
    w_all = np.full((len(T), 2), np.nan)
    V_all = np.full((len(T), 2, 2), np.nan)
    w_all[ok], V_all[ok] = np.linalg.eigh(T[ok])
    mags = np.abs(w_all[ok]).ravel()
    p95 = float(np.percentile(mags, 95)) if mags.size else 0.0
    idx = _frame_rows(cols, m, t, c)
    idx, _zp = _plane_rows(ds, layer, cols, idx, z)
    idx = idx[ok[idx]]
    if idx.size == 0 or not p95 > 0:
        return None
    y = np.asarray(cols["y"], dtype=float)
    x = np.asarray(cols["x"], dtype=float)
    sp = _spacing(np.stack([y[idx], x[idx]], axis=1))
    pick = idx[::max(1, int(every))]
    v1 = V_all[pick, :, 1]                       # (vy, vx) of the LARGEST value
    theta = np.arctan2(v1[:, 0], v1[:, 1])
    rows = np.stack([y[pick], x[pick], theta, w_all[pick, 1], w_all[pick, 0]], axis=1)
    scale = 0.45 * (sp if np.isfinite(sp) else 8.0) / p95
    return Glyphs(rows, float(scale))


def deformed_lines(ds: Any, layer: str, m: int, t: int, z: int, c: int, *,
                   exaggerate: float = 1.0) -> List[np.ndarray]:
    """The field's node LATTICE drawn through the displaced nodes, ``exaggerate`` times the
    displacement: one ``(K, 2)`` y/x polyline (image px) per lattice row and column, broken
    at holes. ``[]`` for a field whose nodes are not a lattice (tracks, a refined DIC mesh)
    — there are no neighbours to join."""
    vf = vectors(ds, m, t, z, c, layer=layer)
    if vf is None or len(vf.rows) < 4:
        return []
    r = vf.rows
    lat = _lattice(r[:, :2])
    if lat is None or min(lat.shape) < 2:
        return []
    k = float(exaggerate)
    py = np.full(lat.shape, np.nan)
    px = np.full(lat.shape, np.nan)
    py[lat.index] = r[:, 0] + k * r[:, 2]
    px[lat.index] = r[:, 1] + k * r[:, 3]
    out: List[np.ndarray] = []
    for gy, gx in ((py, px), (py.T, px.T)):      # rows, then columns
        for a, b in zip(gy, gx):
            good = np.isfinite(a) & np.isfinite(b)
            # split at holes: runs of consecutive finite nodes
            edges = np.flatnonzero(np.diff(np.r_[0, good.astype(np.int8), 0]))
            for s, e in zip(edges[::2], edges[1::2]):
                if e - s >= 2:
                    out.append(np.stack([a[s:e], b[s:e]], axis=1))
    return out


def contour_lines(grid: np.ndarray, levels: Sequence[float]
                  ) -> List[Tuple[float, List[np.ndarray]]]:
    """Iso-lines of ``grid`` at each level as ``(level, [(K, 2) row/col polylines])`` in
    GRID index coordinates (cell centres at integers). NaN cells are filled with a value
    below every level first so a contour closes around a hole instead of crossing it."""
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or min(g.shape) < 2 or not np.isfinite(g).any():
        return []
    from skimage.measure import find_contours   # lazy
    floor = float(np.nanmin(g)) - 1.0
    filled = np.where(np.isfinite(g), g, floor)
    out = []
    for lv in levels:
        if not np.isfinite(lv):
            continue
        out.append((float(lv), [c for c in find_contours(filled, float(lv)) if len(c) >= 2]))
    return out


__all__ = [
    "COORD_COLUMNS", "AUTO_COLUMNS", "point_tables", "data_columns", "layer_names",
    "vector_layer_names", "column_names", "auto_pick", "column_unit", "VectorField",
    "vectors", "auto_scale", "FieldRaster", "raster", "series_limits", "value_at",
    "Glyphs", "tensor_glyphs", "deformed_lines", "contour_lines",
]
