"""Phantoms — deterministic synthetic microscopy datasets, for showing what a node does.

The node demo window (:mod:`nodelab_v2.demo_window`) runs every node on one of these so a
user can see a *before* and an *after* without opening a file, and ``selftest.test_node_demos``
runs the same recipes headless so the demo of every node is proven to compute. They are the
engine's business for the same reason :class:`nodegraph.provider.SyntheticProvider` is: a
Qt-free test instrument that the GUI merely displays.

**What makes them phantoms rather than noise.** Every one is built from a *layout* — a list of
cells with centres, radii, orientation and brightness drawn from a seeded generator — rendered
with soft Gaussian edges, a brighter rim, an uneven background (a ramp plus a low-frequency
blob) and shot noise, at 12-bit full scale. Two cells are planted *touching* so a segmenter
has something to split and a watershed something to prove; ``puncta=True`` sprinkles σ≈1 px
spots for the detectors. The time-lapse variants re-render the same layout shifted (drift)
or with each cell carrying its own velocity (tracking); the 3-D variant renders ellipsoids as
per-plane cross-sections with axial blur; the speckle pair is a reference frame and a smoothly
warped copy for DIC / PIV / DVC; the plate mosaic cuts one field into overlapping stage tiles.

**Determinism is a contract, not a hope.** Same name, same keywords ⇒ bit-identical array:
the layout comes from ``default_rng(seed)`` and every frame's noise from ``default_rng(seed +
1000 + frame)``, never the wall clock — so a demo screenshot, a selftest assertion and the memo
key all agree. Sizes are deliberately small (≤ 160², z ≤ 7, t ≤ 6) so a slider drag recomputes
a plane-granular node well under the eye's patience.

Qt-free; module-scope imports are numpy only, scipy is imported inside the functions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import math

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import ArrayProvider

__all__ = ["Phantom", "PHANTOMS", "phantom", "intensity_range", "cells2d", "cells3d",
           "timelapse_drift", "moving_cells", "two_channel", "speckle_pair", "plate_mosaic",
           "star_field", "moving_mass", "deforming_mass", "star_volume",
           "BIT_DEPTH", "FULL_SCALE", "PIXEL_SIZE_UM", "Z_STEP_UM", "DT_S"]

#: every phantom is 12-bit, like most ND2s — so a ``bit_depth``-derived threshold lands mid-range
BIT_DEPTH = 12
FULL_SCALE = (1 << BIT_DEPTH) - 1
#: a 60×/1.4 NA objective on a 2048² camera — the calibration every µm-denominated default reads
PIXEL_SIZE_UM = 0.325
Z_STEP_UM = 1.0
DT_S = 30.0

_BASE_META: Dict[str, Any] = {
    "pixel_size_um": PIXEL_SIZE_UM,
    "objective_na": 1.4,
    "objective_magnification": 60,
    "bit_depth": BIT_DEPTH,
    #: declared on every phantom, single-frame ones included: a rate-reporting node refuses a
    #: Dataset with no frame interval rather than report µm/frame under a µm/s label
    "dt_s": DT_S,
    "channel_emission_nm": [461],
    "channel_names": ["DAPI"],
}


class Phantom(NamedTuple):
    """One synthetic dataset: the raw ``(m,t,z,c,y,x)`` uint16 array (the *before*), the same
    pixels wrapped as the Dataset + MetaEnvelope a demo graph is seeded with, and a caption."""
    name: str
    array: np.ndarray
    dataset: Dataset
    envelope: MetaEnvelope
    caption: str
    kw: Tuple[Tuple[str, Any], ...]

    @property
    def axes(self) -> AxisSizes:
        return self.dataset.axes


class _Cell(NamedTuple):
    cy: float
    cx: float
    ry: float
    rx: float
    theta: float
    amp: float


# ── building blocks ────────────────────────────────────────────────────────────

def _layout(rng: np.random.Generator, size: int, n: int, *, r_lo: float = 6.0,
            r_hi: float = 11.0, margin: float = 14.0, pairs: int = 2) -> List[_Cell]:
    """``n`` non-overlapping cells plus ``pairs`` touching partners, inside the margin."""
    cells: List[_Cell] = []
    tries = 0
    while len(cells) < n and tries < 5000:
        tries += 1
        cy, cx = (float(v) for v in rng.uniform(margin, size - margin, 2))
        ry = float(rng.uniform(r_lo, r_hi))
        rx = ry * float(rng.uniform(0.7, 1.0))
        if all(np.hypot(cy - c.cy, cx - c.cx) > (ry + c.ry) * 1.4 for c in cells):
            cells.append(_Cell(cy, cx, ry, rx, float(rng.uniform(0, np.pi)),
                               float(rng.uniform(0.7, 1.0))))
    out = list(cells)
    for k in range(min(pairs, len(cells))):
        c = cells[k]
        ang = float(rng.uniform(0, 2 * np.pi))
        d = 1.55 * c.ry
        py = float(np.clip(c.cy + d * np.sin(ang), margin * 0.5, size - margin * 0.5))
        px = float(np.clip(c.cx + d * np.cos(ang), margin * 0.5, size - margin * 0.5))
        out.append(_Cell(py, px, c.ry * 0.9, c.rx * 0.9, c.theta, c.amp * 0.9))
    return out


def _render(cells: Sequence[_Cell], size: int, *, shift: Tuple[float, float] = (0.0, 0.0),
            amp: float = 1700.0, rim: float = 0.35, scale: float = 1.0,
            edge: float = 0.06) -> np.ndarray:
    """A float plane of soft-edged elliptical cells with a brighter rim, no background."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    img = np.zeros((size, size), dtype=float)
    for c in cells:
        ry, rx = c.ry * scale, c.rx * scale
        if ry < 0.6 or rx < 0.6:
            continue
        dy = yy - (c.cy + shift[0])
        dx = xx - (c.cx + shift[1])
        ct, st = np.cos(c.theta), np.sin(c.theta)
        u = (dy * ct + dx * st) / ry
        v = (-dy * st + dx * ct) / rx
        r = np.sqrt(u * u + v * v)
        body = 1.0 / (1.0 + np.exp(np.clip((r - 1.0) / edge, -60, 60)))
        ring = np.exp(-((r - 0.86) ** 2) / (2 * 0.07 ** 2))
        img += c.amp * amp * body * (1.0 + rim * ring)
    return img


def _background(size: int, *, level: float = 320.0, ramp: float = 260.0,
                blob: float = 220.0) -> np.ndarray:
    """An uneven field: a left-to-right ramp plus one low-frequency bright blob."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    return (level + ramp * xx / size
            + blob * np.exp(-(((yy - 0.35 * size) ** 2 + (xx - 0.65 * size) ** 2)
                              / (2 * (0.4 * size) ** 2))))


def _puncta(rng: np.random.Generator, size: int, *, n: int = 40, amp: float = 900.0,
            sigma: float = 1.1, margin: float = 6.0) -> np.ndarray:
    """``n`` sub-resolution spots — what a spot / particle detector is for."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    img = np.zeros((size, size), dtype=float)
    for _ in range(n):
        cy, cx = rng.uniform(margin, size - margin, 2)
        a = amp * float(rng.uniform(0.6, 1.0))
        img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return img


def _expose(signal: np.ndarray, rng: np.random.Generator, *, read_noise: float = 14.0,
            blur: float = 0.8) -> np.ndarray:
    """Optical blur, shot noise and read noise — the camera's contribution."""
    s = np.asarray(signal, dtype=float)
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        s = gaussian_filter(s, blur)
    s = np.clip(s, 0.0, None)
    return rng.poisson(s).astype(float) + rng.normal(0.0, read_noise, s.shape)


def _finish(name: str, arr: np.ndarray, meta: Dict[str, Any], caption: str,
            kw: Dict[str, Any]) -> Phantom:
    a = np.ascontiguousarray(np.clip(np.rint(np.asarray(arr, dtype=float)), 0, FULL_SCALE)
                             .astype(np.uint16))
    if a.ndim != 6:
        raise ValueError(f"phantom {name!r}: expected a 6-D (m,t,z,c,y,x) array, got {a.shape}")
    m, t, z, c, y, x = a.shape
    ax = AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)
    md = dict(meta)
    ds = Dataset(axes=ax, metadata=dict(md)).with_image(ArrayProvider(a, tile=512))
    env = MetaEnvelope(axes=ax, metadata=dict(md), domains=frozenset({Domain.VOXEL}))
    return Phantom(name, a, ds, env, caption, tuple(sorted(kw.items())))


def _meta(**extra: Any) -> Dict[str, Any]:
    md = dict(_BASE_META)
    md["channel_emission_nm"] = list(md["channel_emission_nm"])
    md["channel_names"] = list(md["channel_names"])
    md.update(extra)
    return md


# ── the catalogue ──────────────────────────────────────────────────────────────

def cells2d(*, seed: int = 0, size: int = 160, n_cells: int = 14, puncta: bool = False,
            shift_px: Tuple[float, float] = (0.0, 0.0)) -> Phantom:
    """One plane of fluorescent nuclei on an uneven background, two of them touching.
    ``puncta`` adds spots for the detectors; ``shift_px`` re-renders the same layout
    shifted (a reference image for registration and overlay nodes)."""
    rng = np.random.default_rng(seed)
    cells = _layout(rng, size, n_cells)
    sig = _render(cells, size, shift=tuple(float(v) for v in shift_px)) + _background(size)
    if puncta:
        sig = sig + _puncta(np.random.default_rng(seed + 7), size)
    plane = _expose(sig, np.random.default_rng(seed + 1000))
    cap = (f"{len(cells)} nuclei ({min(2, n_cells)} touching pairs) on an uneven "
           f"background, shot noise" + (", ~40 puncta" if puncta else ""))
    return _finish("cells2d", plane[None, None, None, None], _meta(), cap,
                   dict(seed=seed, size=size, n_cells=n_cells, puncta=puncta,
                        shift_px=tuple(shift_px)))


def cells3d(*, seed: int = 0, size: int = 128, nz: int = 7, n_cells: int = 10) -> Phantom:
    """A z-stack of ellipsoidal nuclei: each plane is a cross-section, blurred axially."""
    rng = np.random.default_rng(seed)
    cells = _layout(rng, size, n_cells, r_lo=7.0, r_hi=12.0)
    cz = rng.uniform(1.5, nz - 2.5, len(cells))
    rz = rng.uniform(1.6, 3.0, len(cells))
    stack = np.zeros((nz, size, size), dtype=float)
    for z in range(nz):
        for i, c in enumerate(cells):
            s2 = 1.0 - ((z - cz[i]) / rz[i]) ** 2
            if s2 <= 0.02:
                continue
            s = float(np.sqrt(s2))
            stack[z] += _render([c], size, scale=s, amp=1700.0 * (0.55 + 0.45 * s))
    from scipy.ndimage import gaussian_filter1d
    stack = gaussian_filter1d(stack, 0.6, axis=0)
    bg = _background(size)
    out = np.zeros((1, 1, nz, 1, size, size), dtype=float)
    for z in range(nz):
        out[0, 0, z, 0] = _expose(stack[z] + bg, np.random.default_rng(seed + 1000 + z))
    cap = f"{len(cells)} ellipsoidal nuclei over {nz} planes, {Z_STEP_UM} µm apart"
    return _finish("cells3d", out, _meta(z_step_um=Z_STEP_UM), cap,
                   dict(seed=seed, size=size, nz=nz, n_cells=n_cells))


def timelapse_drift(*, seed: int = 0, size: int = 160, nt: int = 6,
                    drift_px: Tuple[float, float] = (1.5, -2.0)) -> Phantom:
    """The same field over ``nt`` frames with a cumulative stage drift — what a drift
    correction undoes. Frame ``t`` is shifted by ``t × drift_px`` plus a small jitter."""
    rng = np.random.default_rng(seed)
    cells = _layout(rng, size, 14)
    jit = np.random.default_rng(seed + 3).normal(0.0, 0.3, (nt, 2))
    bg = _background(size)
    out = np.zeros((1, nt, 1, 1, size, size), dtype=float)
    for t in range(nt):
        sh = (t * float(drift_px[0]) + float(jit[t, 0]), t * float(drift_px[1]) + float(jit[t, 1]))
        out[0, t, 0, 0] = _expose(_render(cells, size, shift=sh) + bg,
                                  np.random.default_rng(seed + 1000 + t))
    cap = (f"{len(cells)} nuclei over {nt} frames drifting ({drift_px[0]:+.1f}, "
           f"{drift_px[1]:+.1f}) px per frame")
    return _finish("timelapse_drift", out, _meta(dt_s=DT_S), cap,
                   dict(seed=seed, size=size, nt=nt, drift_px=tuple(drift_px)))


def moving_cells(*, seed: int = 0, size: int = 160, nt: int = 6, n_cells: int = 10) -> Phantom:
    """``n_cells`` nuclei each moving with its own velocity over ``nt`` frames — what a
    tracker links. Speeds are 1–4 px per frame, so a track is visibly a path."""
    rng = np.random.default_rng(seed)
    cells = _layout(rng, size, n_cells, pairs=0, margin=22.0)
    vel = np.random.default_rng(seed + 5).normal(0.0, 2.2, (len(cells), 2))
    vel = np.clip(vel, -4.0, 4.0)
    bg = _background(size)
    out = np.zeros((1, nt, 1, 1, size, size), dtype=float)
    for t in range(nt):
        moved = [c._replace(cy=float(np.clip(c.cy + vel[i, 0] * t, 8, size - 8)),
                            cx=float(np.clip(c.cx + vel[i, 1] * t, 8, size - 8)),
                            amp=c.amp * (1.0 + 0.04 * np.sin(t + i)))
                 for i, c in enumerate(cells)]
        out[0, t, 0, 0] = _expose(_render(moved, size) + bg,
                                  np.random.default_rng(seed + 1000 + t))
    cap = f"{len(cells)} nuclei moving 1–4 px per frame over {nt} frames"
    return _finish("moving_cells", out, _meta(dt_s=DT_S), cap,
                   dict(seed=seed, size=size, nt=nt, n_cells=n_cells))


def two_channel(*, seed: int = 0, size: int = 160) -> Phantom:
    """Two channels of one field: DAPI nuclei, and a GFP cytoplasm around them with
    puncta — what channel selection, splitting and merging act on."""
    rng = np.random.default_rng(seed)
    cells = _layout(rng, size, 12)
    bg = _background(size)
    nuc = _render(cells, size) + bg
    cyto = (_render(cells, size, scale=1.9, amp=650.0, rim=0.0, edge=0.12)
            + _puncta(np.random.default_rng(seed + 7), size, n=30, amp=700.0)
            + _background(size, level=260.0, ramp=120.0, blob=90.0))
    out = np.zeros((1, 1, 1, 2, size, size), dtype=float)
    out[0, 0, 0, 0] = _expose(nuc, np.random.default_rng(seed + 1000))
    out[0, 0, 0, 1] = _expose(cyto, np.random.default_rng(seed + 1001))
    cap = f"{len(cells)} cells: Ch0 nuclei (DAPI), Ch1 cytoplasm with puncta (GFP)"
    return _finish("two_channel", out,
                   _meta(channel_emission_nm=[461, 520], channel_names=["DAPI", "GFP"]),
                   cap, dict(seed=seed, size=size))


def _speckle(rng: np.random.Generator, shape: Tuple[int, ...], sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(rng.normal(0.0, 1.0, shape), sigma)
    s = (s - s.min()) / max(1e-9, float(s.max() - s.min()))
    return 400.0 + 2800.0 * s


def speckle_pair(*, seed: int = 0, size: int = 128, three_d: bool = False,
                 amplitude_px: float = 2.5) -> Phantom:
    """A speckle texture and a smoothly warped copy of it — the reference / deformed pair
    a DIC, PIV or DVC node correlates. Frame 1 = frame 0 displaced by a sinusoidal field
    of ``amplitude_px``; the 3-D variant is a 5-plane volume warped in all three axes."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(seed)
    nz = 5 if three_d else 1
    if three_d:
        size = min(size, 64)
        ref = _speckle(rng, (nz, size, size), 1.2)
        zz, yy, xx = np.mgrid[0:nz, 0:size, 0:size].astype(float)
        u = amplitude_px * np.sin(2 * np.pi * xx / size) * 0.6
        v = amplitude_px * np.cos(2 * np.pi * yy / size)
        w = 0.4 * np.sin(2 * np.pi * xx / size)
        warped = map_coordinates(ref, [zz - w, yy - u, xx - v], order=1, mode="nearest")
        out = np.zeros((1, 2, nz, 1, size, size), dtype=float)
        for z in range(nz):
            out[0, 0, z, 0] = _expose(ref[z], np.random.default_rng(seed + 1000 + z), blur=0.0)
            out[0, 1, z, 0] = _expose(warped[z], np.random.default_rng(seed + 2000 + z), blur=0.0)
        meta = _meta(dt_s=1.0, z_step_um=Z_STEP_UM)
    else:
        ref = _speckle(rng, (size, size), 1.5)
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        u = amplitude_px * np.sin(2 * np.pi * xx / size)
        v = amplitude_px * np.cos(2 * np.pi * yy / size) * 0.7
        warped = map_coordinates(ref, [yy - u, xx - v], order=1, mode="nearest")
        out = np.zeros((1, 2, 1, 1, size, size), dtype=float)
        out[0, 0, 0, 0] = _expose(ref, np.random.default_rng(seed + 1000), blur=0.0)
        out[0, 1, 0, 0] = _expose(warped, np.random.default_rng(seed + 2000), blur=0.0)
        meta = _meta(dt_s=1.0)
    cap = (f"speckle reference (t=0) and a copy warped by a ±{amplitude_px:g} px sinusoidal "
           f"field (t=1)" + (f", {nz} planes" if three_d else ""))
    return _finish("speckle_pair", out, meta, cap,
                   dict(seed=seed, size=size, three_d=three_d, amplitude_px=amplitude_px))


def plate_mosaic(*, seed: int = 0, tile: int = 96, grid: Tuple[int, int] = (2, 2),
                 overlap: float = 0.12) -> Phantom:
    """One field cut into a ``rows × cols`` raster of overlapping stage tiles (the M axis),
    each carrying its stage position — what stitching and position selection consume."""
    rows, cols = int(grid[0]), int(grid[1])
    step = int(round(tile * (1.0 - overlap)))
    H = step * (rows - 1) + tile
    W = step * (cols - 1) + tile
    rng = np.random.default_rng(seed)
    big = max(H, W)
    cells = [c for c in _layout(rng, big, int(14 * (H * W) / (160.0 * 160.0)) + 6, pairs=1)
             if c.cy < H and c.cx < W]
    field = _expose(_render(cells, big)[:H, :W] + _background(big)[:H, :W],
                    np.random.default_rng(seed + 1000))
    n = rows * cols
    out = np.zeros((n, 1, 1, 1, tile, tile), dtype=float)
    stage: List[List[float]] = []
    names: List[str] = []
    for r in range(rows):
        for c in range(cols):
            m = r * cols + c
            out[m, 0, 0, 0] = field[r * step:r * step + tile, c * step:c * step + tile]
            stage.append([c * step * PIXEL_SIZE_UM, r * step * PIXEL_SIZE_UM])
            names.append(f"{chr(65 + r)}{c + 1}")
    meta = _meta(stage_xy_um=stage, position_name=names, position_index=list(range(n)))
    cap = (f"{rows}×{cols} stage tiles of {tile} px overlapping {int(overlap * 100)}%, "
           f"with their stage positions")
    return _finish("plate_mosaic", out, meta, cap,
                   dict(seed=seed, tile=tile, grid=(rows, cols), overlap=overlap))


# ── the registration worlds (2026-10-07) ───────────────────────────────────────
#
# ``scripts/registration_synthetic_bench.py`` measured the registration kernel against four
# synthetic worlds with an EXACT known motion. These are the same worlds at demo size, so the
# *What does this node do?* window shows Registration on the data its behaviour was
# established on — and the caption states the true motion, so the per-frame shift the node
# reports can be read against it.

def _affine_yx(angle_deg: float, shift: Tuple[float, float], shear: float, scale: float,
               centre: Tuple[float, float]) -> np.ndarray:
    """3×3 forward map in (y, x): rotate about ``centre`` (positive = +x toward +y, clockwise
    on screen), shear x by ``shear``·(y − cy), scale about the centre, then translate."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    L = scale * (np.array([[c, s], [-s, c]]) @ np.array([[1.0, 0.0], [shear, 1.0]]))
    cvec = np.asarray(centre, dtype=float)
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = cvec - L @ cvec + np.asarray(shift, dtype=float)
    return M


def _move_points(M: np.ndarray, pts_yx: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts_yx, np.ones((len(pts_yx), 1))], axis=1)
    return (M @ p.T).T[:, :2]


def _star_catalogue(rng: np.random.Generator, n_in_frame: int, size: int, margin: float,
                    amp_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """``n_in_frame`` beads per frame on average, laid over the frame plus ``margin`` so
    beads enter and leave as the field moves; log-uniform brightness."""
    area = (size + 2 * margin) ** 2 / float(size * size)
    n = int(round(n_in_frame * area))
    pos = np.stack([rng.uniform(-margin, size + margin, n),
                    rng.uniform(-margin, size + margin, n)], axis=1)
    amp = np.exp(rng.uniform(math.log(amp_range[0]), math.log(amp_range[1]), n))
    return pos, amp


def _render_stars(pos_yx: np.ndarray, amp: np.ndarray, size: int, sigma: float = 1.5) -> np.ndarray:
    """Gaussian beads at sub-pixel positions, rendered analytically (no resampling)."""
    img = np.zeros((size, size), dtype=float)
    r = int(math.ceil(4 * sigma))
    for (py, px), a in zip(pos_yx, amp):
        y0, x0 = int(math.floor(py)) - r, int(math.floor(px)) - r
        ya, yb = max(0, y0), min(size, y0 + 2 * r + 1)
        xa, xb = max(0, x0), min(size, x0 + 2 * r + 1)
        if ya >= yb or xa >= xb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        img[ya:yb, xa:xb] += a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
    return img


def star_field(*, seed: int = 0, size: int = 160, nt: int = 6, n_stars: int = 40,
               drift_px: Tuple[float, float] = (1.5, -2.0), rotate_deg: float = 0.0,
               faint: bool = False) -> Phantom:
    """Fluorescent beads over ``nt`` frames under a known motion: frame ``t`` is drifted by
    ``t × drift_px`` and turned by ``t × rotate_deg`` about the frame centre, on an uneven
    background. ``faint`` drops the beads to a peak signal-to-noise of about 7 — the regime a
    matched filter rescues. Sparse (``n_stars`` ≈ 12) or faint fields are where a
    registration earns or loses its sub-pixel claim."""
    rng = np.random.default_rng(seed)
    margin = max(30.0, abs(drift_px[0]) * nt + 12.0, abs(drift_px[1]) * nt + 12.0,
                 0.35 * size if rotate_deg else 0.0)
    pos0, amp = _star_catalogue(rng, n_stars, size, margin,
                                (120.0, 300.0) if faint else (600.0, 3400.0))
    c = ((size - 1) / 2.0, (size - 1) / 2.0)
    bg = _background(size)
    out = np.zeros((1, nt, 1, 1, size, size), dtype=float)
    for t in range(nt):
        M = _affine_yx(rotate_deg * t, (drift_px[0] * t, drift_px[1] * t), 0.0, 1.0, c)
        img = _render_stars(_move_points(M, pos0), amp, size) + bg
        out[0, t, 0, 0] = _expose(img, np.random.default_rng(seed + 1000 + t), blur=0.0,
                                  read_noise=26.0 if faint else 14.0)
    cap = (f"{n_stars} beads over {nt} frames drifting ({drift_px[0]:+.1f}, {drift_px[1]:+.1f}) "
           f"px per frame")
    if rotate_deg:
        cap += f" and turning {rotate_deg:g}° per frame about the centre"
    if faint:
        cap += ", faint (peak SNR ≈ 7)"
    return _finish("star_field", out, _meta(dt_s=DT_S), cap,
                   dict(seed=seed, size=size, nt=nt, n_stars=n_stars, drift_px=tuple(drift_px),
                        rotate_deg=rotate_deg, faint=faint))


def _textured_body(rng: np.random.Generator, size: int, radius: float, *,
                   texture_sigma: float = 2.0, edge: float = 4.0,
                   peak: float = 2200.0) -> np.ndarray:
    """A soft-edged disk filled with speckle texture — a body with enough internal detail for
    ECC to lock onto and no corners for a feature detector to lean on."""
    from scipy.ndimage import gaussian_filter
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cy = cx = (size - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    disk = 1.0 / (1.0 + np.exp(np.clip((r - radius) / edge, -60, 60)))
    tex = gaussian_filter(rng.random((size, size)), texture_sigma)
    tex = (tex - tex.min()) / max(1e-9, float(np.ptp(tex)))
    return peak * disk * (0.35 + 0.65 * tex)


def moving_mass(*, seed: int = 0, size: int = 160, nt: int = 6, rotate_deg: float = 4.0,
                drift_px: Tuple[float, float] = (1.0, -0.5), shear: float = 0.0,
                scale: float = 1.0, radius: float = 46.0) -> Phantom:
    """A textured body under an EXACT affine motion per frame: turned by ``t × rotate_deg``
    about the frame centre, drifted by ``t × drift_px``, sheared by ``t × shear`` (x grows
    with y) and scaled by ``scale ** t``. What the euclidean / affine models are for, and
    what the translation model cannot represent."""
    from scipy.ndimage import affine_transform
    rng = np.random.default_rng(seed)
    base = _textured_body(rng, size, radius)
    bg = _background(size)
    c = ((size - 1) / 2.0, (size - 1) / 2.0)
    out = np.zeros((1, nt, 1, 1, size, size), dtype=float)
    for t in range(nt):
        M = _affine_yx(rotate_deg * t, (drift_px[0] * t, drift_px[1] * t), shear * t,
                       scale ** t, c)
        inv = np.linalg.inv(M)
        moved = base if t == 0 else affine_transform(base, inv[:2, :2], offset=inv[:2, 2],
                                                     order=3, mode="constant", cval=0.0)
        out[0, t, 0, 0] = _expose(np.clip(moved, 0.0, None) + bg,
                                  np.random.default_rng(seed + 1000 + t), blur=0.0)
    bits = []
    if rotate_deg:
        bits.append(f"turning {rotate_deg:g}° per frame")
    if drift_px[0] or drift_px[1]:
        bits.append(f"drifting ({drift_px[0]:+.1f}, {drift_px[1]:+.1f}) px per frame")
    if shear:
        bits.append(f"shearing {shear:g} per frame")
    if scale != 1.0:
        bits.append(f"growing ×{scale:g} per frame")
    cap = f"a textured body over {nt} frames " + (", ".join(bits) if bits else "holding still")
    return _finish("moving_mass", out, _meta(dt_s=DT_S), cap,
                   dict(seed=seed, size=size, nt=nt, rotate_deg=rotate_deg,
                        drift_px=tuple(drift_px), shear=shear, scale=scale, radius=radius))


def deforming_mass(*, seed: int = 0, size: int = 160, nt: int = 6, bulge_px: float = 0.6,
                   drift_px: Tuple[float, float] = (0.6, 0.3), radius: float = 46.0,
                   bulge_sigma: float = 18.0) -> Phantom:
    """A textured body whose right side bulges outward by ``t × bulge_px`` (a smooth,
    NON-affine displacement) while the whole body drifts by ``t × drift_px``. No global
    transform can register the bulge; the question a drift correction has to answer is
    whether it recovers the drift anyway — and the bulge pulls on that estimate unless the
    estimate is restricted to the half that holds still."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(seed)
    base = _textured_body(rng, size, radius)
    bg = _background(size)
    cy = cx = (size - 1) / 2.0
    bc = np.array([cy, cx + 0.55 * radius])
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    g = np.stack([yy.ravel(), xx.ravel()], axis=1)
    out = np.zeros((1, nt, 1, 1, size, size), dtype=float)
    for t in range(nt):
        a, d = bulge_px * t, np.asarray(drift_px, dtype=float) * t

        def fwd(p: np.ndarray) -> np.ndarray:
            r2 = ((p - bc) ** 2).sum(axis=1)
            return (a * np.exp(-r2 / (2 * bulge_sigma ** 2)))[:, None] * np.array([[0.0, 1.0]]) + d
        if t == 0:
            moved = base
        else:
            q = g.copy()
            for _ in range(10):                     # q + D(q) = p, solved by fixed point
                q = g - fwd(q)
            moved = map_coordinates(base, [q[:, 0].reshape(size, size),
                                           q[:, 1].reshape(size, size)],
                                    order=3, mode="constant", cval=0.0)
        out[0, t, 0, 0] = _expose(np.clip(moved, 0.0, None) + bg,
                                  np.random.default_rng(seed + 1000 + t), blur=0.0)
    cap = (f"a textured body over {nt} frames: its right side bulges out {bulge_px:g} px per "
           f"frame while the whole body drifts ({drift_px[0]:+.1f}, {drift_px[1]:+.1f}) px per "
           f"frame")
    return _finish("deforming_mass", out, _meta(dt_s=DT_S), cap,
                   dict(seed=seed, size=size, nt=nt, bulge_px=bulge_px, drift_px=tuple(drift_px),
                        radius=radius, bulge_sigma=bulge_sigma))


def star_volume(*, seed: int = 0, size: int = 96, nz: int = 12, nt: int = 4, n_stars: int = 50,
                drift_px: Tuple[float, float, float] = (0.5, 1.0, -0.6)) -> Phantom:
    """Beads in a ``nz``-plane stack over ``nt`` frames drifting in all three axes by
    ``t × drift_px`` (planes, px, px). A planar registration sees only the lateral part and
    leaves the axial drift in place; the 3D lever recovers it."""
    rng = np.random.default_rng(seed)
    dz, dy, dx = (float(v) for v in drift_px)
    mz = max(3.0, abs(dz) * nt + 2.0)
    mxy = max(16.0, abs(dy) * nt + 8.0, abs(dx) * nt + 8.0)
    n = int(round(n_stars * ((nz + 2 * mz) / nz) * ((size + 2 * mxy) ** 2 / float(size * size))))
    pos0 = np.stack([rng.uniform(-mz, nz + mz, n), rng.uniform(-mxy, size + mxy, n),
                     rng.uniform(-mxy, size + mxy, n)], axis=1)
    amp = np.exp(rng.uniform(math.log(600.0), math.log(3400.0), n))
    sig_z, sig_xy = 1.5, 1.5
    rz, r = int(math.ceil(3 * sig_z)), int(math.ceil(3.5 * sig_xy))
    bg = _background(size, level=260.0, ramp=120.0, blob=90.0)
    out = np.zeros((1, nt, nz, 1, size, size), dtype=float)
    for t in range(nt):
        vol = np.zeros((nz, size, size), dtype=float)
        for (pz, py, px), a in zip(pos0 + np.array([dz, dy, dx]) * t, amp):
            z0, y0, x0 = int(math.floor(pz)) - rz, int(math.floor(py)) - r, int(math.floor(px)) - r
            za, zb = max(0, z0), min(nz, z0 + 2 * rz + 1)
            ya, yb = max(0, y0), min(size, y0 + 2 * r + 1)
            xa, xb = max(0, x0), min(size, x0 + 2 * r + 1)
            if za >= zb or ya >= yb or xa >= xb:
                continue
            zz, yy, xx = np.mgrid[za:zb, ya:yb, xa:xb]
            vol[za:zb, ya:yb, xa:xb] += a * np.exp(
                -((zz - pz) ** 2) / (2 * sig_z ** 2) - ((yy - py) ** 2 + (xx - px) ** 2) / (2 * sig_xy ** 2))
        for z in range(nz):
            out[0, t, z, 0] = _expose(vol[z] + bg, np.random.default_rng(seed + 1000 + 31 * t + z),
                                      blur=0.0)
    cap = (f"{n_stars} beads in a {nz}-plane stack over {nt} frames, drifting {dz:+.1f} planes "
           f"and ({dy:+.1f}, {dx:+.1f}) px per frame")
    return _finish("star_volume", out, _meta(dt_s=DT_S, z_step_um=Z_STEP_UM), cap,
                   dict(seed=seed, size=size, nz=nz, nt=nt, n_stars=n_stars,
                        drift_px=tuple(drift_px)))


PHANTOMS: Dict[str, Callable[..., Phantom]] = {
    "cells2d": cells2d,
    "cells3d": cells3d,
    "timelapse_drift": timelapse_drift,
    "moving_cells": moving_cells,
    "two_channel": two_channel,
    "speckle_pair": speckle_pair,
    "plate_mosaic": plate_mosaic,
    "star_field": star_field,
    "moving_mass": moving_mass,
    "deforming_mass": deforming_mass,
    "star_volume": star_volume,
}


def _freeze(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v


@lru_cache(maxsize=64)
def _cached(name: str, items: Tuple[Tuple[str, Any], ...]) -> Phantom:
    fn = PHANTOMS.get(name)
    if fn is None:
        raise KeyError(f"unknown phantom {name!r}; known: {', '.join(sorted(PHANTOMS))}")
    return fn(**{k: v for k, v in items})


def phantom(name: str, **kw: Any) -> Phantom:
    """The named phantom with these keywords, cached — the same object every time, so a
    demo window and the memo see one identity for one dataset."""
    return _cached(name, tuple(sorted((k, _freeze(v)) for k, v in kw.items())))


def intensity_range(ph: Phantom, *, c: int = 0, t: int = 0,
                    z: Optional[int] = None) -> Tuple[float, float, float, float]:
    """``(min, p1, p99, max)`` of one plane — the span an intensity-level slider covers."""
    a = ph.array
    zz = a.shape[2] // 2 if z is None else int(z)
    plane = np.asarray(a[0, min(t, a.shape[1] - 1), min(zz, a.shape[2] - 1),
                         min(c, a.shape[3] - 1)], dtype=float)
    return (float(plane.min()), float(np.percentile(plane, 1.0)),
            float(np.percentile(plane, 99.0)), float(plane.max()))
