"""scene_viewer — layer geometry for the composable 3-D HTML scene viewer.

A *scene* is a list of layers in one world frame (µm, image convention on input: +x
right, +y DOWN, +z up the stack; the packer flips y for display so the page is right-handed
with z up). Each builder below turns one kind of data into one layer dict whose big
arrays are base64 strings, ``pack`` assembles the scene (bounds, timeline, the display
flip) and ``render_html`` writes it into ``scene_viewer_template.html``.

The builders know nothing about Datasets, calibration or nodes: every input is already in
µm (positions) and seconds or frame index (time). ``nodegraph/kernels/scene_viewer.md`` is
the contract; ``nodegraph/catalog/io/write_scene_viewer.py`` is the one caller.
"""

from __future__ import annotations

import base64
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "scene_viewer_template.html")

#: colormap names in the order the page's shader indexes them
COLORMAPS: Tuple[str, ...] = ("turbo", "viridis", "inferno", "plasma", "magma", "gray", "cool")

_NAMED_COLORS = {
    "white": "#ffffff", "gray": "#9aa3ad", "grey": "#9aa3ad", "red": "#ff6b6b",
    "green": "#69db7c", "blue": "#74c0fc", "cyan": "#31d0c6", "magenta": "#e599f7",
    "yellow": "#ffd43b", "orange": "#ff922b", "sand": "#d9c2a0",
}


# ── small helpers ────────────────────────────────────────────────────────────

def b64(arr: np.ndarray, dtype) -> str:
    """Base64 of ``arr`` as a C-contiguous little-endian ``dtype`` buffer."""
    a = np.ascontiguousarray(np.asarray(arr), dtype=dtype)
    if a.dtype.byteorder == ">":
        a = a.byteswap().newbyteorder()
    return base64.b64encode(a.tobytes()).decode("ascii")


def parse_color(spec: Any, default: str = "#8ab4d8") -> List[float]:
    """``'#rrggbb'`` / ``'#rgb'`` / a named colour / an ``(r, g, b)`` triple (0–1 or
    0–255) → ``[r, g, b]`` in 0–1. Anything unparseable → ``default``."""
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        try:
            v = [float(c) for c in spec]
        except (TypeError, ValueError):
            return parse_color(default)
        if max(v) > 1.0:
            v = [c / 255.0 for c in v]
        return [min(1.0, max(0.0, c)) for c in v]
    s = str(spec or "").strip().lower()
    s = _NAMED_COLORS.get(s, s)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) == 6:
        try:
            return [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        except ValueError:
            pass
    return parse_color(default) if spec != default else [0.54, 0.71, 0.85]


def colormap_index(name: Any) -> int:
    s = str(name or "turbo").strip().lower()
    return COLORMAPS.index(s) if s in COLORMAPS else 0


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    ok = None
    for a in arrays:
        f = np.isfinite(np.asarray(a, dtype=float))
        f = f.all(axis=-1) if f.ndim > 1 else f
        ok = f if ok is None else (ok & f)
    return ok if ok is not None else np.zeros(0, bool)


def _range(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> List[float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0.0, 1.0]
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    if not hi > lo:
        lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        hi = lo + 1.0
    return [lo, hi]


# ── volumes ──────────────────────────────────────────────────────────────────

def downsample_factors(shape: Tuple[int, int, int], spacing_um: Tuple[float, float, float],
                       voxel_budget: int) -> Tuple[int, int, int]:
    """Integer block sizes ``(fz, fy, fx)`` that bring ``shape`` under ``voxel_budget``
    voxels, growing the axis whose pooled spacing is currently the finest in µm — so a
    14 µm plane step is never pooled while the pixels are 0.5 µm, and a 0.1 µm step pools
    before the pixels do."""
    n = [int(max(1, v)) for v in shape]
    sp = [float(max(1e-9, s)) for s in spacing_um]
    budget = max(1000, int(voxel_budget))
    f = [1, 1, 1]
    while math.prod(math.ceil(n[i] / f[i]) for i in range(3)) > budget:
        growable = [i for i in range(3) if f[i] < n[i]]
        if not growable:
            break
        i = min(growable, key=lambda k: sp[k] * f[k])
        f[i] += 1
    return f[0], f[1], f[2]


def block_mean(vol: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """Mean-pool ``vol`` (nz, ny, nx) by integer ``factors``, edge blocks included."""
    v = np.asarray(vol, dtype=np.float32)
    fz, fy, fx = (int(max(1, f)) for f in factors)
    if (fz, fy, fx) == (1, 1, 1):
        return v
    nz, ny, nx = v.shape
    pz, py, px = (-nz) % fz, (-ny) % fy, (-nx) % fx
    if pz or py or px:
        v = np.pad(v, ((0, pz), (0, py), (0, px)), mode="edge")
    nz, ny, nx = v.shape
    return v.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx).mean(axis=(1, 3, 5))


def to_uint8(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float32)
    v = (v - lo) / max(hi - lo, 1e-12)
    return np.clip(np.nan_to_num(v) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def isosurface(vol: np.ndarray, level: float, spacing_um: Tuple[float, float, float]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marching cubes of ``vol`` (nz, ny, nx) at ``level`` → ``(pos (N,3) µm as (x, y, z)
    offsets from the brick origin, normals (N,3), faces (F,3))``; empty arrays when the
    volume is too small or the level is outside its range."""
    v = np.asarray(vol, dtype=np.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    if v.ndim != 3 or min(v.shape) < 2:
        return empty
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if not (vmin < level < vmax):
        return empty
    from skimage.measure import marching_cubes
    verts, faces, normals, _ = marching_cubes(np.nan_to_num(v), level=level,
                                              spacing=tuple(float(s) for s in spacing_um))
    pos = verts[:, ::-1].astype(np.float32)          # (z, y, x) → (x, y, z)
    nrm = normals[:, ::-1].astype(np.float32)
    return pos, nrm, faces.astype(np.uint32)


def volume_layer(frames: Sequence[Tuple[float, np.ndarray]], spacing_um: Tuple[float, float, float],
                 origin_um: Tuple[float, float, float], *, name: str, render: str = "mip",
                 colormap: str = "gray", color: Any = "#ffffff", opacity: float = 0.8,
                 window_pct: Tuple[float, float] = (0.5, 99.8), iso_level_pct: float = 90.0,
                 voxel_budget: int = 2_500_000, units: str = "") -> Dict[str, Any]:
    """One image channel as a volume layer. ``frames`` = ``[(t, vol (nz, ny, nx) float), …]``
    at full resolution; the layer carries each frame as a uint8 brick (``render`` in
    ``mip`` / ``cloud`` / ``slices``) or an iso-surface mesh (``render == 'iso'``) on the
    block-mean grid that fits ``voxel_budget``. The intensity window is one pair of
    percentiles over every frame, so brightness is comparable along the timeline."""
    if not frames:
        raise ValueError("volume_layer: no frames")
    render = str(render or "mip").lower()
    if render not in ("mip", "cloud", "slices", "iso"):
        raise ValueError(f"volume_layer: unknown render {render!r}")
    shape = tuple(int(n) for n in np.asarray(frames[0][1]).shape)
    if len(shape) != 3:
        raise ValueError(f"volume_layer: frames must be (nz, ny, nx), got {shape}")
    fz, fy, fx = downsample_factors(shape, spacing_um, voxel_budget)
    bricks = []
    for t, vol in frames:
        vol = np.asarray(vol)
        if tuple(vol.shape) != shape:
            raise ValueError(f"volume_layer: frame at t={t} has shape {vol.shape}, "
                             f"expected {shape}")
        bricks.append((float(t), block_mean(vol, (fz, fy, fx))))
    sample = np.concatenate([b[1].ravel()[::max(1, b[1].size // 200_000)] for b in bricks])
    lo, hi = _range(sample, window_pct[0], window_pct[1])
    sp = (float(spacing_um[0]) * fz, float(spacing_um[1]) * fy, float(spacing_um[2]) * fx)
    dims = [int(n) for n in bricks[0][1].shape]
    layer: Dict[str, Any] = {
        "kind": "volume", "name": str(name), "render": render,
        "colormap": colormap_index(colormap), "color": parse_color(color, "#ffffff"),
        "opacity": float(min(1.0, max(0.0, opacity))), "units": units,
        "dims": dims, "spacing": list(sp),
        "origin": [float(origin_um[2]), float(origin_um[1]), float(origin_um[0])],  # (x, y, z)
        "window": [lo, hi], "downsample": [fz, fy, fx], "frames": [],
    }
    if render == "iso":
        level = float(np.percentile(sample[np.isfinite(sample)], iso_level_pct)) \
            if np.isfinite(sample).any() else 0.0
        layer["iso_level"] = level
        n_tri = 0
        for t, brick in bricks:
            pos, nrm, faces = isosurface(brick, level, sp)
            n_tri += int(faces.shape[0])
            layer["frames"].append({"t": t, "pos": b64(pos, np.float32), "nrm": b64(nrm, np.float32),
                                    "idx": b64(faces, np.uint32), "n_vertices": int(pos.shape[0]),
                                    "n_triangles": int(faces.shape[0])})
        layer["n_triangles"] = n_tri
    else:
        for t, brick in bricks:
            layer["frames"].append({"t": t, "data": b64(to_uint8(brick, lo, hi), np.uint8)})
    layer["bounds"] = _box_bounds(layer["origin"], dims, sp)
    return layer


def _box_bounds(origin_xyz: Sequence[float], dims_zyx: Sequence[int],
                spacing_zyx: Sequence[float]) -> List[List[float]]:
    ox, oy, oz = (float(v) for v in origin_xyz)
    nz, ny, nx = (int(v) for v in dims_zyx)
    sz, sy, sx = (float(v) for v in spacing_zyx)
    return [[ox, oy, oz], [ox + nx * sx, oy + ny * sy, oz + nz * sz]]


# ── vectors ──────────────────────────────────────────────────────────────────

def grid_pitch(coords: np.ndarray) -> float:
    """Median spacing of the distinct values of one coordinate column (0 when fewer than
    two distinct values)."""
    u = np.unique(np.round(np.asarray(coords, dtype=float), 6))
    if u.size < 2:
        return 0.0
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(np.median(d)) if d.size else 0.0


def streamlines(pos: np.ndarray, vec: np.ndarray, *, pitch: float, n_steps: int = 60,
                seed_stride: int = 3, max_lines: int = 800, step_frac: float = 0.5,
                rng_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streamlines through a vector field sampled on a (near-)regular grid of ``pitch`` µm.
    The rows are binned to a lattice (nearest cell, mean per cell), the field is
    trilinearly interpolated and RK2-integrated forward and backward from every
    ``seed_stride``-th cell with a measurement. Returns ``(points (P,3), speed (P),
    ranges (L,2) start/count)``; empty when the field is degenerate."""
    from scipy.ndimage import map_coordinates
    empty = (np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros((0, 2), np.uint32))
    pos = np.asarray(pos, dtype=float); vec = np.asarray(vec, dtype=float)
    ok = _finite_rows(pos, vec)
    pos, vec = pos[ok], vec[ok]
    if pos.shape[0] < 4 or not pitch > 0:
        return empty
    lo = pos.min(axis=0)
    idx = np.rint((pos - lo) / pitch).astype(int)
    dims = idx.max(axis=0) + 1                                  # (nx, ny, nz)
    if int(np.prod(dims)) > 50_000_000:
        return empty
    nx, ny, nz = (int(d) for d in dims)
    field = np.zeros((3, nz, ny, nx), np.float64)
    count = np.zeros((nz, ny, nx), np.float64)
    flat = (idx[:, 2] * ny + idx[:, 1]) * nx + idx[:, 0]
    for k in range(3):
        np.add.at(field[k].ravel(), flat, vec[:, k])
    np.add.at(count.ravel(), flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(count > 0, field / np.maximum(count, 1e-12), 0.0)
    support = count > 0
    mag = np.sqrt((field ** 2).sum(axis=0))
    vmax = float(np.percentile(mag[support], 95)) if support.any() else 0.0
    if not vmax > 0:
        return empty
    # seeds: every seed_stride-th supported cell, shuffled, capped
    seeds = np.argwhere(support)                                 # (z, y, x)
    seeds = seeds[(seeds[:, 0] % max(1, seed_stride // 2 or 1) == 0) &
                  (seeds[:, 1] % seed_stride == 0) & (seeds[:, 2] % seed_stride == 0)]
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(seeds)
    seeds = seeds[:max_lines].astype(float)
    if seeds.shape[0] == 0:
        return empty
    h = step_frac / vmax                                        # cells per unit speed

    def sample(p):                                               # p: (S,3) as (z,y,x) cells
        c = p.T
        out = np.stack([map_coordinates(field[k], c, order=1, mode="nearest") for k in range(3)], 1)
        s = map_coordinates(support.astype(np.float64), c, order=1, mode="constant", cval=0.0)
        return out, s                                            # (S,3) as (x,y,z), (S,)

    def integrate(direction):
        p = seeds.copy(); alive = np.ones(len(p), bool); traj = [p.copy()]; spd = []
        v, s = sample(p); spd.append(np.sqrt((v ** 2).sum(1)))
        for _ in range(n_steps):
            v1, s1 = sample(p)
            k1 = direction * v1[:, ::-1] * h                     # (x,y,z) → (z,y,x) cell step
            v2, s2 = sample(p + 0.5 * k1)
            k2 = direction * v2[:, ::-1] * h
            step = np.where(alive[:, None], k2, 0.0)
            p = p + step
            inside = (p >= 0).all(1) & (p <= np.array([nz - 1, ny - 1, nx - 1])).all(1)
            moving = np.sqrt((v2 ** 2).sum(1)) > 0.02 * vmax
            alive &= inside & (s2 > 0.25) & moving
            if not alive.any():
                break
            traj.append(np.where(alive[:, None], p, traj[-1]))
            spd.append(np.where(alive, np.sqrt((v2 ** 2).sum(1)), spd[-1]))
        return np.stack(traj, 1), np.stack(spd, 1), alive         # (S, K, 3), (S, K)

    fw_p, fw_s, _ = integrate(+1.0)
    bw_p, bw_s, _ = integrate(-1.0)
    pts, spds, ranges = [], [], []
    start = 0
    for i in range(seeds.shape[0]):
        line = np.concatenate([bw_p[i, ::-1], fw_p[i, 1:]], 0)
        sp = np.concatenate([bw_s[i, ::-1], fw_s[i, 1:]], 0)
        # drop the stalled duplicates at either end
        d = np.r_[True, (np.abs(np.diff(line, axis=0)).sum(1) > 1e-9)]
        line, sp = line[d], sp[d]
        if line.shape[0] < 3:
            continue
        world = lo + line[:, ::-1] * pitch                          # (z,y,x) cells → (x,y,z) µm
        pts.append(world.astype(np.float32)); spds.append(sp.astype(np.float32))
        ranges.append((start, line.shape[0])); start += line.shape[0]
    if not pts:
        return empty
    return (np.concatenate(pts, 0), np.concatenate(spds, 0), np.asarray(ranges, np.uint32))


def vectors_layer(frames: Sequence[Tuple[float, np.ndarray, np.ndarray]], *, name: str,
                  units: str = "", color_by: str = "magnitude", colormap: str = "turbo",
                  color: Any = "#31d0c6", scale: float = 0.0, max_vectors: int = 200_000,
                  with_streamlines: bool = False, max_lines: int = 800, seed_stride: int = 3,
                  rng_seed: int = 0) -> Dict[str, Any]:
    """A vector field (PIV / DVC) as line glyphs, one frame per timepoint. ``frames`` =
    ``[(t, pos (N,3) µm (x,y,z), vec (N,3) in the field's units), …]``. ``scale`` is the
    display length in µm per unit of the field; 0 = auto so the median glyph spans one grid
    pitch. With ``with_streamlines`` the first frame's field is also integrated."""
    if not frames:
        raise ValueError("vectors_layer: no frames")
    color_by = str(color_by or "magnitude").lower()
    if color_by not in ("magnitude", "vertical", "uniform"):
        raise ValueError(f"vectors_layer: unknown color_by {color_by!r}")
    out_frames, mags, pitch = [], [], 0.0
    rng = np.random.default_rng(rng_seed)
    first_pos = first_vec = None
    for t, pos, vec in frames:
        pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        vec = np.asarray(vec, dtype=float).reshape(-1, 3)
        if pos.shape[0] != vec.shape[0]:
            raise ValueError("vectors_layer: pos and vec row counts differ")
        ok = _finite_rows(pos, vec)
        pos, vec = pos[ok], vec[ok]
        if pos.shape[0] > max_vectors > 0:
            keep = np.sort(rng.choice(pos.shape[0], size=max_vectors, replace=False))
            pos, vec = pos[keep], vec[keep]
        mag = np.sqrt((vec ** 2).sum(1))
        mags.append(mag)
        if first_pos is None:
            first_pos, first_vec = pos, vec
            pitch = max(grid_pitch(pos[:, 0]), grid_pitch(pos[:, 1]))
        out_frames.append({"t": float(t), "pos": b64(pos, np.float32), "vec": b64(vec, np.float32),
                           "mag": b64(mag, np.float32), "n": int(pos.shape[0])})
    all_mag = np.concatenate(mags) if mags else np.zeros(0)
    med = float(np.median(all_mag[all_mag > 0])) if (all_mag > 0).any() else 1.0
    auto = (pitch if pitch > 0 else 1.0) / max(med, 1e-12)
    layer = {
        "kind": "vectors", "name": str(name), "units": units, "color_by": color_by,
        "colormap": colormap_index(colormap), "color": parse_color(color, "#31d0c6"),
        "scale": float(scale) if scale and scale > 0 else float(auto), "auto_scale": float(auto),
        "pitch": float(pitch), "mag_range": _range(all_mag, 1.0, 99.0), "frames": out_frames,
        "n_vectors": int(sum(f["n"] for f in out_frames)),
    }
    if with_streamlines and first_pos is not None and pitch > 0:
        pts, spd, rng_ = streamlines(first_pos, first_vec, pitch=pitch, max_lines=max_lines,
                                     seed_stride=seed_stride, rng_seed=rng_seed)
        layer["lines"] = {"pos": b64(pts, np.float32), "spd": b64(spd, np.float32),
                          "ranges": b64(rng_, np.uint32), "n": int(rng_.shape[0])}
    layer["bounds"] = _points_bounds([f for f in frames]) if frames else None
    return layer


def _points_bounds(frames: Sequence[Tuple]) -> Optional[List[List[float]]]:
    lo = hi = None
    for fr in frames:
        pos = np.asarray(fr[1], dtype=float).reshape(-1, 3)
        pos = pos[np.isfinite(pos).all(1)]
        if pos.shape[0] == 0:
            continue
        a, b = pos.min(0), pos.max(0)
        lo = a if lo is None else np.minimum(lo, a)
        hi = b if hi is None else np.maximum(hi, b)
    if lo is None:
        return None
    return [[float(v) for v in lo], [float(v) for v in hi]]


# ── objects ──────────────────────────────────────────────────────────────────

def objects_layer(frames: Sequence[Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]], *,
                  name: str, color_by: str = "uniform", colormap: str = "viridis",
                  color: Any = "#ffd43b", value_label: str = "", size_scale: float = 1.0,
                  max_objects: int = 200_000, rng_seed: int = 0) -> Dict[str, Any]:
    """Objects (cells, beads, particles) as spheres, one frame per timepoint. ``frames`` =
    ``[(t, pos (N,3) µm, radius (N) µm, value (N) or None), …]``; ``value`` colours the
    spheres through ``colormap`` when ``color_by == 'value'``."""
    if not frames:
        raise ValueError("objects_layer: no frames")
    color_by = str(color_by or "uniform").lower()
    if color_by not in ("uniform", "value", "z"):
        raise ValueError(f"objects_layer: unknown color_by {color_by!r}")
    rng = np.random.default_rng(rng_seed)
    out, vals, rads = [], [], []
    for t, pos, rad, val in frames:
        pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        rad = np.asarray(rad, dtype=float).reshape(-1)
        val = (np.asarray(val, dtype=float).reshape(-1) if val is not None
               else np.zeros(pos.shape[0]))
        if not (pos.shape[0] == rad.shape[0] == val.shape[0]):
            raise ValueError("objects_layer: pos / radius / value row counts differ")
        ok = _finite_rows(pos) & np.isfinite(rad) & (rad > 0)
        pos, rad, val = pos[ok], rad[ok], np.nan_to_num(val[ok])
        if pos.shape[0] > max_objects > 0:
            keep = np.sort(rng.choice(pos.shape[0], size=max_objects, replace=False))
            pos, rad, val = pos[keep], rad[keep], val[keep]
        vals.append(val); rads.append(rad)
        out.append({"t": float(t), "pos": b64(pos, np.float32), "rad": b64(rad, np.float32),
                    "val": b64(val, np.float32), "n": int(pos.shape[0])})
    all_val = np.concatenate(vals) if vals else np.zeros(0)
    all_rad = np.concatenate(rads) if rads else np.zeros(0)
    return {
        "kind": "objects", "name": str(name), "color_by": color_by,
        "colormap": colormap_index(colormap), "color": parse_color(color, "#ffd43b"),
        "value_label": value_label, "value_range": _range(all_val, 1.0, 99.0),
        "size_scale": float(size_scale if size_scale > 0 else 1.0),
        "radius_median": float(np.median(all_rad)) if all_rad.size else 1.0,
        "frames": out, "n_objects": int(sum(f["n"] for f in out)),
        "bounds": _points_bounds([(t, p) for t, p, _, _ in frames]),
    }


# ── tracks ───────────────────────────────────────────────────────────────────

def tracks_layer(tracks: Sequence[Tuple[int, np.ndarray, np.ndarray]], *, name: str,
                 color_by: str = "track", colormap: str = "turbo", color: Any = "#e599f7",
                 tail: float = 0.0, max_tracks: int = 5000, min_points: int = 2) -> Dict[str, Any]:
    """Tracks as polylines. ``tracks`` = ``[(track_id, times (K), pos (K,3) µm), …]`` with
    the rows of one track in time order. Speed per vertex is the forward difference in
    µm per time unit. ``tail`` is the comet length in time units (0 = whole track)."""
    color_by = str(color_by or "track").lower()
    if color_by not in ("track", "time", "speed", "uniform"):
        raise ValueError(f"tracks_layer: unknown color_by {color_by!r}")
    pts, tt, spd, ranges, ids = [], [], [], [], []
    start = 0
    kept = 0
    for tid, times, pos in tracks:
        times = np.asarray(times, dtype=float).reshape(-1)
        pos = np.asarray(pos, dtype=float).reshape(-1, 3)
        ok = np.isfinite(times) & _finite_rows(pos)
        times, pos = times[ok], pos[ok]
        order = np.argsort(times, kind="stable")
        times, pos = times[order], pos[order]
        if times.shape[0] < min_points:
            continue
        if kept >= max_tracks > 0:
            break
        d = np.sqrt((np.diff(pos, axis=0) ** 2).sum(1))
        dt = np.diff(times)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(dt > 0, d / dt, 0.0)
        v = np.r_[v, v[-1] if v.size else 0.0]
        pts.append(pos.astype(np.float32)); tt.append(times.astype(np.float32))
        spd.append(v.astype(np.float32)); ranges.append((start, times.shape[0])); ids.append(int(tid))
        start += times.shape[0]; kept += 1
    if not pts:
        P = np.zeros((0, 3), np.float32); T = np.zeros(0, np.float32); S = np.zeros(0, np.float32)
        R = np.zeros((0, 2), np.uint32); I = np.zeros(0, np.uint32)
    else:
        P = np.concatenate(pts, 0); T = np.concatenate(tt); S = np.concatenate(spd)
        R = np.asarray(ranges, np.uint32); I = np.asarray(ids, np.uint32)
    return {
        "kind": "tracks", "name": str(name), "color_by": color_by,
        "colormap": colormap_index(colormap), "color": parse_color(color, "#e599f7"),
        "tail": float(max(0.0, tail)), "n_tracks": int(R.shape[0]), "n_points": int(P.shape[0]),
        "pos": b64(P, np.float32), "t": b64(T, np.float32), "spd": b64(S, np.float32),
        "ranges": b64(R, np.uint32), "ids": b64(I, np.uint32),
        "speed_range": _range(S, 1.0, 99.0),
        "time_range": [float(T.min()), float(T.max())] if T.size else [0.0, 0.0],
        "bounds": _points_bounds([(0.0, P)]) if P.size else None,
    }


# ── series (charts) ──────────────────────────────────────────────────────────

def series_layer(curves: Sequence[Tuple[str, np.ndarray, np.ndarray]], *, name: str,
                 x_label: str = "time", y_label: str = "", x_is_time: bool = True,
                 color: Any = "#31d0c6") -> Dict[str, Any]:
    """A chart panel: ``curves`` = ``[(label, x (K), y (K)), …]``. When ``x_is_time`` the
    chart's cursor follows the scene timeline."""
    out = []
    for label, x, y in curves:
        x = np.asarray(x, dtype=float).reshape(-1); y = np.asarray(y, dtype=float).reshape(-1)
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if x.shape[0] == 0:
            continue
        out.append({"label": str(label), "x": b64(x, np.float32), "y": b64(y, np.float32),
                    "n": int(x.shape[0])})
    return {"kind": "series", "name": str(name), "x_label": x_label, "y_label": y_label,
            "x_is_time": bool(x_is_time), "color": parse_color(color, "#31d0c6"),
            "curves": out, "n_curves": len(out)}


# ── scene assembly ───────────────────────────────────────────────────────────

def _flip_y_buffer(b: str, dtype, stride: int, y0y1: float) -> str:
    a = np.frombuffer(base64.b64decode(b), dtype=dtype).reshape(-1, stride).copy()
    a[:, 1] = y0y1 - a[:, 1]
    return b64(a, dtype)


def _flip_layer_y(layer: Dict[str, Any], y0y1: float) -> Dict[str, Any]:
    """Display frame: ``y' = y0 + y1 - y`` on every position buffer, the brick's y axis
    reversed and its origin moved. Normals: ``ny' = -ny``."""
    L = dict(layer)
    kind = L["kind"]
    if kind == "volume":
        nz, ny, nx = L["dims"]; sy = L["spacing"][1]
        ox, oy, oz = L["origin"]
        L["origin"] = [ox, y0y1 - (oy + ny * sy), oz]
        frames = []
        for f in L["frames"]:
            g = dict(f)
            if "data" in f:
                a = np.frombuffer(base64.b64decode(f["data"]), np.uint8).reshape(nz, ny, nx)
                g["data"] = b64(a[:, ::-1, :], np.uint8)
            else:
                pos = np.frombuffer(base64.b64decode(f["pos"]), np.float32).reshape(-1, 3).copy()
                nrm = np.frombuffer(base64.b64decode(f["nrm"]), np.float32).reshape(-1, 3).copy()
                # mesh vertices are offsets from the brick origin: mirror inside the brick
                pos[:, 1] = ny * sy - pos[:, 1]; nrm[:, 1] = -nrm[:, 1]
                g["pos"] = b64(pos, np.float32); g["nrm"] = b64(nrm, np.float32)
            frames.append(g)
        L["frames"] = frames
    elif kind == "vectors":
        frames = []
        for f in L["frames"]:
            g = dict(f)
            g["pos"] = _flip_y_buffer(f["pos"], np.float32, 3, y0y1)
            v = np.frombuffer(base64.b64decode(f["vec"]), np.float32).reshape(-1, 3).copy()
            v[:, 1] = -v[:, 1]; g["vec"] = b64(v, np.float32)
            frames.append(g)
        L["frames"] = frames
        if L.get("lines"):
            ln = dict(L["lines"]); ln["pos"] = _flip_y_buffer(ln["pos"], np.float32, 3, y0y1)
            L["lines"] = ln
    elif kind == "objects":
        L["frames"] = [dict(f, pos=_flip_y_buffer(f["pos"], np.float32, 3, y0y1)) for f in L["frames"]]
    elif kind == "tracks":
        if L["n_points"]:
            L["pos"] = _flip_y_buffer(L["pos"], np.float32, 3, y0y1)
    if L.get("bounds"):
        (x0, y0, z0), (x1, y1, z1) = L["bounds"]
        L["bounds"] = [[x0, y0y1 - y1, z0], [x1, y0y1 - y0, z1]]
    return L


def pack(layers: Sequence[Dict[str, Any]], *, title: str, subtitle: str = "",
         time_unit: str = "s", pad_fraction: float = 0.03) -> Dict[str, Any]:
    """Assemble the page's ``SCENE``: the layers in the display frame (y flipped about the
    scene's y extent), the world bounds in µm, the union timeline and the counts. Raises
    on an empty scene."""
    layers = [dict(L) for L in layers if L]
    if not layers:
        raise ValueError("pack: a scene needs at least one layer")
    boxes = [L["bounds"] for L in layers if L.get("bounds")]
    if not boxes:
        raise ValueError("pack: no layer has any geometry (every frame was empty)")
    lo = np.min([b[0] for b in boxes], axis=0); hi = np.max([b[1] for b in boxes], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = float(max(span.max() * pad_fraction, 1e-6))
    y0y1 = float(lo[1] + hi[1])
    disp = [_flip_layer_y(L, y0y1) for L in layers]
    times = sorted({float(f["t"]) for L in disp for f in L.get("frames", []) if "t" in f})
    for L in disp:
        if L["kind"] == "tracks" and L["n_points"]:
            times.extend([L["time_range"][0], L["time_range"][1]])
    times = sorted(set(times))
    dynamic = any(len(L.get("frames", [])) > 1 for L in disp) or any(
        L["kind"] == "tracks" and L["n_points"] for L in disp)
    meta = {
        "title": title, "subtitle": subtitle, "units_xyz": "µm",
        "bounds": [[float(lo[0]) - pad, float(lo[1]) - pad, float(lo[2]) - pad],
                   [float(hi[0]) + pad, float(hi[1]) + pad, float(hi[2]) + pad]],
        "times": times, "time_unit": time_unit, "dynamic": bool(dynamic),
        "n_layers": len(disp),
        "layers": [{"kind": L["kind"], "name": L["name"]} for L in disp],
    }
    return {"meta": meta, "layers": disp}


def render_html(scene: Dict[str, Any], *, title: str, subtitle: str = "",
                snapshot_name: str = "scene.png", template_path: str = TEMPLATE_PATH) -> str:
    """Fill the template: ``__SCENE__`` (the JSON), ``__TITLE__``, ``__SUBTITLE__``,
    ``__SNAPNAME__``. Raises ``FileNotFoundError`` without the template."""
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"scene viewer template missing: {template_path}")
    with open(template_path, "r", encoding="utf-8") as fh:
        page = fh.read()
    data = json.dumps(scene, separators=(",", ":"), allow_nan=False)
    data = data.replace("</", "<\\/")
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (page.replace("__SCENE__", data)
                .replace("__TITLE__", esc(title))
                .replace("__SUBTITLE__", esc(subtitle))
                .replace("__SNAPNAME__", json.dumps(snapshot_name)))


def scene_report(scene: Dict[str, Any]) -> Dict[str, Any]:
    """The counts a node puts in its metadata report."""
    rep: Dict[str, Any] = {"n_layers": scene["meta"]["n_layers"], "n_times": len(scene["meta"]["times"]),
                           "dynamic": scene["meta"]["dynamic"], "bounds_um": scene["meta"]["bounds"],
                           "layers": []}
    for L in scene["layers"]:
        k = L["kind"]
        item: Dict[str, Any] = {"kind": k, "name": L["name"]}
        if k == "volume":
            item.update(render=L["render"], dims=L["dims"], spacing_um=L["spacing"],
                        n_frames=len(L["frames"]), downsample=L["downsample"])
            if L["render"] == "iso":
                item["n_triangles"] = L.get("n_triangles", 0)
        elif k == "vectors":
            item.update(n_vectors=L["n_vectors"], n_frames=len(L["frames"]), scale=L["scale"],
                        n_lines=(L.get("lines") or {}).get("n", 0))
        elif k == "objects":
            item.update(n_objects=L["n_objects"], n_frames=len(L["frames"]))
        elif k == "tracks":
            item.update(n_tracks=L["n_tracks"], n_points=L["n_points"])
        elif k == "series":
            item.update(n_curves=L["n_curves"])
        rep["layers"].append(item)
    return rep


__all__ = [
    "TEMPLATE_PATH", "COLORMAPS", "b64", "parse_color", "colormap_index",
    "downsample_factors", "block_mean", "to_uint8", "isosurface", "volume_layer",
    "grid_pitch", "streamlines", "vectors_layer", "objects_layer", "tracks_layer",
    "series_layer", "pack", "render_html", "scene_report",
]
