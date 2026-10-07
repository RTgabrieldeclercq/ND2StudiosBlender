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
warped copy for DIC / PIV / DVC; the plate mosaic cuts one field into overlapping stage tiles;
the bead field (``beads3d``) is a confocal z-stack of fluorescent spheres at a chosen density,
imaged through a skewed confocal axial PSF, with non-bead artefacts planted for a detector to
refuse — and it carries its **ground truth** (:attr:`Phantom.truth`) so a validation can score
recall, precision and localisation error against the planted positions.

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

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import ArrayProvider

__all__ = ["Phantom", "PHANTOMS", "phantom", "intensity_range", "cells2d", "cells3d",
           "timelapse_drift", "moving_cells", "two_channel", "speckle_pair", "plate_mosaic",
           "beads3d", "confocal_axial_kernel",
           "BIT_DEPTH", "FULL_SCALE", "PIXEL_SIZE_UM", "Z_STEP_UM", "DT_S",
           "BEAD_Z_STEP_UM", "BEAD_EMISSION_NM", "BEAD_AXIAL_SKEW"]

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
    #: ground truth where the phantom has one (``beads3d``: the planted positions and what
    #: each artefact was) — ``None`` for the phantoms that only need to look right
    truth: Any = None

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
            kw: Dict[str, Any], truth: Any = None) -> Phantom:
    a = np.ascontiguousarray(np.clip(np.rint(np.asarray(arr, dtype=float)), 0, FULL_SCALE)
                             .astype(np.uint16))
    if a.ndim != 6:
        raise ValueError(f"phantom {name!r}: expected a 6-D (m,t,z,c,y,x) array, got {a.shape}")
    m, t, z, c, y, x = a.shape
    ax = AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)
    md = dict(meta)
    ds = Dataset(axes=ax, metadata=dict(md)).with_image(ArrayProvider(a, tile=512))
    env = MetaEnvelope(axes=ax, metadata=dict(md), domains=frozenset({Domain.VOXEL}))
    return Phantom(name, a, ds, env, caption, tuple(sorted(kw.items())), truth)


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


#: the bead stack is sampled finer than the nuclei stack — a 1 µm bead has to span planes
BEAD_Z_STEP_UM = 0.4
#: green beads (FITC-like) under the same 60×/1.4 objective as every other phantom
BEAD_EMISSION_NM = 520
#: how much wider the axial profile is on the far side of focus than on the near side —
#: the spherical-aberration skew a real confocal stack carries (index mismatch)
BEAD_AXIAL_SKEW = 0.5


def confocal_axial_kernel(z_step_um: float, fwhm_um: float, *, skew: float = BEAD_AXIAL_SKEW,
                          pinhole_leak: float = 0.15) -> np.ndarray:
    """The axial PSF of a confocal microscope, sampled at the plane spacing and summing
    to one, with its **peak exactly at the centre sample**.

    Confocal detection multiplies the illumination and the detection responses, so for a
    point emitter the ideal axial profile is ``sinc^4`` (the square of the widefield
    ``sinc^2``), with sidelobes of ~0.2 %. A real pinhole is not infinitesimal, which lets
    some widefield character through: ``pinhole_leak`` mixes that ``sinc^2`` back in. The
    ``sinc^4`` core's FWHM is ``0.638 · zeta`` for ``sinc(z / zeta)``, so ``zeta`` is set from
    ``fwhm_um``. Finally the far side of focus is stretched by ``(1 + skew)``: refractive-
    index mismatch between immersion medium and sample broadens the profile asymmetrically,
    and that asymmetry is exactly what a symmetric Gaussian fit in z gets wrong. Warping
    about zero leaves the peak at zero, so the planted bead centre IS the brightest plane."""
    zeta = max(1e-6, float(fwhm_um)) / 0.638
    half = int(np.ceil(4.0 * float(fwhm_um) * (1.0 + max(0.0, skew)) / float(z_step_um))) + 2
    z = np.arange(-half, half + 1, dtype=float) * float(z_step_um)
    zw = np.where(z > 0, z / (1.0 + max(0.0, skew)), z)
    s = np.sinc(zw / zeta)                       # numpy's sinc is sin(pi x)/(pi x)
    k = (1.0 - pinhole_leak) * s ** 4 + pinhole_leak * s ** 2
    return k / k.sum()


def _ball(stack: np.ndarray, centre_zyx: Tuple[float, float, float],
          radii_um: Tuple[float, float, float], vox_um: Tuple[float, float, float],
          amp: float, *, edge_um: float = 0.08) -> None:
    """Add a soft-edged solid ellipsoid (radii in µm per axis) to ``stack`` in place."""
    nz, ny, nx = stack.shape
    cz, cy, cx = centre_zyx
    rz, ry, rx = radii_um
    dz, dy, dx = vox_um
    z0, z1 = max(0, int(np.floor(cz - rz / dz)) - 2), min(nz, int(np.ceil(cz + rz / dz)) + 3)
    y0, y1 = max(0, int(np.floor(cy - ry / dy)) - 2), min(ny, int(np.ceil(cy + ry / dy)) + 3)
    x0, x1 = max(0, int(np.floor(cx - rx / dx)) - 2), min(nx, int(np.ceil(cx + rx / dx)) + 3)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(float)
    # normalised radius: 1 on the surface, scaled back to µm for a size-independent edge
    rn = np.sqrt(((zz - cz) * dz / rz) ** 2 + ((yy - cy) * dy / ry) ** 2
                 + ((xx - cx) * dx / rx) ** 2)
    r_um = rn * min(rz, ry, rx)
    body = 1.0 / (1.0 + np.exp(np.clip((r_um - min(rz, ry, rx)) / edge_um, -60, 60)))
    stack[z0:z1, y0:y1, x0:x1] += amp * body


def _rod(stack: np.ndarray, centre_zyx: Tuple[float, float, float], length_um: float,
         angle: float, radius_um: float, vox_um: Tuple[float, float, float], amp: float
         ) -> None:
    """Add a fibre: a soft cylinder of ``length_um`` in the plane at ``angle``, in place."""
    nz, ny, nx = stack.shape
    cz, cy, cx = centre_zyx
    dz, dy, dx = vox_um
    uy, ux = np.sin(angle), np.cos(angle)
    half = 0.5 * length_um
    pad = radius_um * 3
    y0 = max(0, int(np.floor(cy - (half * abs(uy) + pad) / dy)))
    y1 = min(ny, int(np.ceil(cy + (half * abs(uy) + pad) / dy)) + 1)
    x0 = max(0, int(np.floor(cx - (half * abs(ux) + pad) / dx)))
    x1 = min(nx, int(np.ceil(cx + (half * abs(ux) + pad) / dx)) + 1)
    z0 = max(0, int(np.floor(cz - pad / dz)))
    z1 = min(nz, int(np.ceil(cz + pad / dz)) + 1)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(float)
    py, px = (yy - cy) * dy, (xx - cx) * dx
    along = np.clip(py * uy + px * ux, -half, half)
    perp2 = (py - along * uy) ** 2 + (px - along * ux) ** 2 + ((zz - cz) * dz) ** 2
    r = np.sqrt(perp2)
    stack[z0:z1, y0:y1, x0:x1] += amp / (1.0 + np.exp(np.clip((r - radius_um) / 0.06, -60, 60)))


def beads3d(*, seed: int = 0, size: int = 128, nz: int = 24, n_beads: int = 60,
            diameter_um: float = 1.0, artifacts: bool = True,
            gradient: bool = False) -> Phantom:
    """A confocal z-stack of ``n_beads`` fluorescent spheres of ``diameter_um`` at random
    positions (and random brightness, 0.4–1.0 of nominal), imaged through a confocal PSF —
    an Airy-like lateral blur and the skewed ``sinc^4`` axial profile of
    :func:`confocal_axial_kernel` — over an uneven background that dims with depth, with
    shot and read noise. ``gradient=True`` places beads with a left-to-right density ramp
    so one field spans sparse and dense.

    ``artifacts=True`` plants the things a bead finder must refuse: three large bright
    aggregates, two fibres, one bright scan line on a single plane, a diffuse
    autofluorescent haze, 25 hot voxels (added after the optics, before the noise — a
    detector event, not an object), and four beads whose centres lie **outside** the stack
    so only an axial tail is visible.

    :attr:`Phantom.truth` records every planted bead's ``(z, y, x)`` voxel centre and
    brightness, the beads outside the stack, the PSF parameters and the artefact counts,
    so a validation can score the finder rather than eyeball it."""
    rng = np.random.default_rng(seed)
    dz, dy, dx = BEAD_Z_STEP_UM, PIXEL_SIZE_UM, PIXEL_SIZE_UM
    vox = (dz, dy, dx)
    na, n_imm, lam_um = 1.4, 1.515, BEAD_EMISSION_NM / 1000.0
    fwhm_z_um = 0.88 * lam_um / (n_imm - np.sqrt(n_imm ** 2 - na ** 2))     # ≈ 0.49 µm
    sigma_xy_um = 0.21 * lam_um / na                                        # ≈ 0.078 µm
    radius_um = 0.5 * float(diameter_um)
    peak = 2200.0                               # nominal bead peak, counts above background

    # layout — the bead body must lie inside the stack, so the truth is detectable
    margin_xy = radius_um / dx + 3.0
    z_lo, z_hi = radius_um / dz + 1.5, nz - 1 - radius_um / dz - 1.5
    n = max(0, int(n_beads))
    cy = rng.uniform(margin_xy, size - margin_xy, n)
    if gradient:
        cx = margin_xy + (size - 2 * margin_xy) * np.sqrt(rng.uniform(0.0, 1.0, n))
    else:
        cx = rng.uniform(margin_xy, size - margin_xy, n)
    cz = rng.uniform(z_lo, z_hi, n) if z_hi > z_lo else np.full(n, 0.5 * (nz - 1))
    amp = rng.uniform(0.4, 1.0, n)
    # render on a Z grid padded above and below the stack: an object just outside the
    # acquired range still throws its axial tail into the first or last planes, and the
    # convolution has to see it to reproduce that (the padding is cut off at the end)
    pad = (int(np.ceil(4.0 * fwhm_z_um * (1.0 + BEAD_AXIAL_SKEW) / dz))
           + int(np.ceil(radius_um / dz)) + 2)
    sig = np.zeros((nz + 2 * pad, size, size), dtype=float)
    for i in range(n):
        _ball(sig, (cz[i] + pad, cy[i], cx[i]), (radius_um,) * 3, vox, peak * amp[i])

    counts: Dict[str, int] = {}
    where: Dict[str, Any] = {}       # where each artefact was put, for a validation to score
    outside = np.zeros((0, 3), dtype=float)
    hot: List[Tuple[int, int, int]] = []
    if artifacts:
        arng = np.random.default_rng(seed + 77)
        # aggregates / debris: big, bright per voxel but not as bright as a bead's core
        aggs = []
        for _ in range(3):
            c = (arng.uniform(3, nz - 4), arng.uniform(12, size - 12),
                 arng.uniform(12, size - 12))
            r = (arng.uniform(0.9, 1.4), arng.uniform(1.4, 2.4), arng.uniform(1.4, 2.4))
            _ball(sig, (c[0] + pad, c[1], c[2]), r, vox, peak * 0.6)
            aggs.append((*c, *r))
        counts["aggregate"] = 3
        where["aggregate"] = np.asarray(aggs, dtype=float)          # (z, y, x, rz, ry, rx µm)
        rods = []
        for _ in range(2):
            c = (arng.uniform(3, nz - 4), arng.uniform(16, size - 16),
                 arng.uniform(16, size - 16))
            length, ang = arng.uniform(6.0, 10.0), arng.uniform(0, np.pi)
            _rod(sig, (c[0] + pad, c[1], c[2]), length, ang, 0.25, vox, peak * 0.9)
            rods.append((*c, length, ang))
        counts["fibre"] = 2
        where["fibre"] = np.asarray(rods, dtype=float)              # (z, y, x, length µm, angle)
        zline, yline = int(arng.integers(2, nz - 2)), int(arng.integers(8, size - 8))
        sig[zline + pad, yline, :] += peak * 0.8
        counts["scan_line"] = 1
        where["scan_line"] = (zline, yline)
        zz, yy, xx = np.mgrid[0:nz, 0:size, 0:size].astype(float)
        hc = (arng.uniform(6, nz - 7), arng.uniform(24, size - 24), arng.uniform(24, size - 24))
        sig[pad:pad + nz] += peak * 0.35 * np.exp(
            -(((zz - hc[0]) * dz) ** 2 / (2 * 2.0 ** 2)
              + ((yy - hc[1]) * dy) ** 2 / (2 * 5.0 ** 2)
              + ((xx - hc[2]) * dx) ** 2 / (2 * 5.0 ** 2)))
        counts["haze"] = 1
        where["haze"] = tuple(float(v) for v in hc)
        out_rows = []
        for k in range(4):
            # centre 0.6 plane beyond the first / last acquired plane: the bead's body
            # reaches into the stack and its axial tail is bright on the end plane, but its
            # brightest plane was never acquired — the finder must not place it inside
            czo = -0.6 if k % 2 == 0 else nz - 1 + 0.6
            c = (czo, arng.uniform(margin_xy, size - margin_xy),
                 arng.uniform(margin_xy, size - margin_xy))
            _ball(sig, (czo + pad, c[1], c[2]), (radius_um,) * 3, vox, peak * 0.9)
            out_rows.append(c)
        outside = np.asarray(out_rows, dtype=float)
        counts["outside_bead"] = 4
        for _ in range(25):
            hot.append((int(arng.integers(0, nz)), int(arng.integers(0, size)),
                        int(arng.integers(0, size))))
        counts["hot_voxel"] = 25
        where["hot_voxel"] = np.asarray(hot, dtype=float)
        where["outside_bead"] = outside

    # the optics: lateral Airy-like blur per plane, then the skewed confocal axial profile
    from scipy.ndimage import convolve1d, gaussian_filter
    sig = gaussian_filter(sig, (0.0, sigma_xy_um / dy, sigma_xy_um / dx))
    kz = confocal_axial_kernel(dz, fwhm_z_um)
    sig = convolve1d(sig, kz, axis=0, mode="constant", cval=0.0)[pad:pad + nz]
    # depth attenuation, an uneven background, detector events, and the camera
    depth = 1.0 - 0.25 * np.arange(nz, dtype=float)[:, None, None] / max(1, nz - 1)
    bg = _background(size, level=180.0, ramp=120.0, blob=90.0)
    out = np.zeros((1, 1, nz, 1, size, size), dtype=float)
    for z in range(nz):
        plane = sig[z] * depth[z, 0, 0] + bg
        for (hz, hy, hx) in hot:
            if hz == z:
                plane = plane.copy()
                plane[hy, hx] += peak * 3.0
        out[0, 0, z, 0] = _expose(plane, np.random.default_rng(seed + 1000 + z),
                                  read_noise=10.0, blur=0.0)
    truth = {
        "beads": np.stack([cz, cy, cx], axis=1) if n else np.zeros((0, 3)),
        "amp": amp, "diameter_um": float(diameter_um), "outside": outside,
        "artifacts": counts, "artifact_positions": where,
        "voxel_size_um": vox, "axial_fwhm_um": float(fwhm_z_um),
        "axial_skew": BEAD_AXIAL_SKEW, "psf_sigma_xy_um": float(sigma_xy_um),
        "n_beads": n,
    }
    vol_um3 = nz * dz * size * dy * size * dx
    cap = (f"{n} fluorescent {diameter_um:g} µm beads ({n / vol_um3 * 1000:.1f} per 1000 µm³"
           f"{', denser to the right' if gradient else ''}) over {nz} planes {dz} µm apart; "
           f"confocal PSF (axial FWHM {fwhm_z_um:.2f} µm, far side {1 + BEAD_AXIAL_SKEW:g}× "
           f"wider), depth dimming, shot noise"
           + (f"; artefacts: {', '.join(f'{v} {k}' for k, v in counts.items())}"
              if counts else ""))
    return _finish("beads3d", out, _meta(z_step_um=dz, channel_emission_nm=[BEAD_EMISSION_NM],
                                          channel_names=["beads"]), cap,
                   dict(seed=seed, size=size, nz=nz, n_beads=n_beads, diameter_um=diameter_um,
                        artifacts=artifacts, gradient=gradient), truth)


PHANTOMS: Dict[str, Callable[..., Phantom]] = {
    "cells2d": cells2d,
    "cells3d": cells3d,
    "timelapse_drift": timelapse_drift,
    "moving_cells": moving_cells,
    "two_channel": two_channel,
    "speckle_pair": speckle_pair,
    "plate_mosaic": plate_mosaic,
    "beads3d": beads3d,
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
