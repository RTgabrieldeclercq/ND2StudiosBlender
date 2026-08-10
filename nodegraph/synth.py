"""Synthetic packed bed of convex bodies with **known per-voxel ownership** — a test fixture.

This exists because the 585 hand labels count granules and are structurally blind to where a
boundary lies. A count metric cannot validate a signed distance, a confidence field, or a
surface-energy term, and the last pass demonstrated what happens when it is used anyway: a
geometric partition scored well on counting while placing its divides at full granule
brightness, inside the bodies. Ground truth is the only instrument that can say *correct*
rather than *better than the last thing*.

**It is a fixture, not a shipped compute path.** ``test_catalog_import_hygiene`` forbids any
module under ``nodegraph/catalog/`` from importing it — a machine-checked guarantee, strictly
stronger than the accidental one that living under ``scripts/`` would give. Module-scope
imports are **numpy only**; scipy is imported inside the functions that need it, with pure-numpy
fallbacks, so this is never the reason a test skips.

## What is planted, and why each one is mandatory

Contact geometry is a property of the *construction*, not a hope about the RNG:

* **A thin-necked pair** — two lobes joined by a narrow bridge. The split-at-neck move **must**
  take this apart. Its neck plane is recorded exactly, so the boundary error is measurable in µm.
* **A broad-contact pair** — two bodies meeting over a wide flat facet. The split move **must
  leave this alone**. Without it, a move that bisects everything would pass by fixing the neck
  case, and a count metric would barely notice; this is the control that makes the asymmetry
  in ``ΔE = γ̃·A_neck − gain`` the thing actually being tested.
* **Tangent pairs at a chosen surface gap**, so ``s₁ = s₂`` on a known plane and the claim
  "the confidence crosses 0.5 at the contact" has somewhere exact to be checked.
* **An isolated body**, far from everything, so the background class ``E_∅`` has a voxel that
  must be assigned to *nothing*.

Deliberately absent in this pass: children/dots, drift, rotation, surface roughness, and the
polydisperse forced-overlap ladder. They belong to pose tracking and relaxation, not to
validating the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Body", "SynthBed", "make_bed", "bed_to_dataset"]

#: class ids for the two intensity populations (a body's `kind`)
KIND_A, KIND_B = 0, 1


@dataclass(frozen=True)
class Body:
    """One body. ``planes`` is ``(F, dims+1)`` as ``[normal | offset]``, interior ≤ 0.

    ``bite`` is an optional second plane set describing a convex region **subtracted** from the
    body, which makes it genuinely non-convex — a dented face. It exists because the real
    granules are not all convex: some faces carry concave stretches. That matters more than it
    sounds, because a concave face lowers ``solidity`` exactly the way a thin neck does, so a
    split move that PROPOSES on low solidity will fire on a perfectly good single granule. The
    ``ΔE`` accept test is supposed to reject it — there is no thin place to cut — and a planted
    concave body is the only way to find out whether it actually does.

    A body with a bite is deliberately NOT described by its own ``planes``: the convex hull
    bridges the dent, so ``s < 0`` inside space the body does not occupy. That is the real
    behaviour, not a defect of the fixture.
    """

    gid: int
    kind: int
    centre_um: np.ndarray
    planes: np.ndarray
    r_eq_um: float
    role: str = "bulk"           # bulk | neck_a | neck_b | broad_a | broad_b | lone | concave
    partner: int = 0                        # the other gid of a planted pair, else 0
    bite: Optional[np.ndarray] = None       # planes of a convex region removed from the body
    aspect: float = 1.0                     # elongation actually applied


@dataclass
class SynthBed:
    """A synthetic field plus everything needed to grade a segmentation of it."""

    image: np.ndarray                        # (M,T,Z,C,Y,X) float32
    owner: np.ndarray                        # (Z,Y,X) int32 — 0 = background, else gid
    bodies: List[Body]
    voxel_um: Tuple[float, ...]
    #: `(gid_a, gid_b, plane)` per planted pair, plane as `[normal | offset]` in µm
    contacts: List[Tuple[int, int, np.ndarray, str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def dims(self) -> int:
        return int(self.owner.ndim)

    def body(self, gid: int) -> Body:
        return next(b for b in self.bodies if b.gid == gid)

    def role_gids(self, role: str) -> List[int]:
        return [b.gid for b in self.bodies if b.role == role]


# ── geometry ─────────────────────────────────────────────────────────────────

def _polygon_planes(centre: np.ndarray, r: float, n_sides: int, phase: float,
                    lobing: float, rng: np.random.Generator) -> np.ndarray:
    """A convex polygon (2D) as ``[normal | offset]`` rows, unit normals, interior ≤ 0.

    A polytope is defined by its **facets**, so it is built facet-first rather than by hulling
    vertices: pick `n_sides` outward directions and push each supporting line to distance
    `d_i` from the centre. Every such intersection is convex by construction, which is what
    makes "these bodies are convex" a fact about the fixture rather than a hope.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False) + phase
    nrm = np.stack([np.sin(th), np.cos(th)], axis=1)                    # (F, 2) as (y, x)
    d = r * (1.0 + lobing * np.cos(3.0 * th + 0.7)
             + 0.04 * rng.standard_normal(n_sides))
    return np.c_[nrm, -(nrm @ centre) - d]


def _prism_planes(centre: np.ndarray, r: float, n_sides: int, phase: float,
                  lobing: float, half_z: float, rng: np.random.Generator) -> np.ndarray:
    """The 3D case: a polygonal prism, i.e. the 2D facets plus two z caps."""
    p2 = _polygon_planes(centre[1:], r, n_sides, phase, lobing, rng)
    A = np.zeros((p2.shape[0] + 2, 4))
    A[:-2, 1:3] = p2[:, :2]
    A[:-2, 3] = p2[:, 2]
    A[-2] = [1.0, 0.0, 0.0, -(centre[0] + half_z)]
    A[-1] = [-1.0, 0.0, 0.0, (centre[0] - half_z)]
    return A


def _scale_planes(planes: np.ndarray, s_vec: Sequence[float]) -> np.ndarray:
    """Anisotropically scale a body about the ORIGIN, giving it an aspect ratio.

    For ``{x : n·x + b <= 0}`` scaled by a diagonal ``S``, the image is
    ``{y : (S⁻¹n)·y + b <= 0}``, so the normals divide by the scale factors and are then
    renormalised (with ``b`` divided by the same norms, or ``|∇s| = 1`` is lost). Real granules
    are not all equant — some are distinctly elongated — and a fixture of near-circular bodies
    would never exercise a boundary between two long thin ones.
    """
    d = planes.shape[1] - 1
    A = planes[:, :d] / np.asarray(s_vec, dtype=float)[None, :]
    b = planes[:, d].copy()
    nrm = np.linalg.norm(A, axis=1)
    return np.c_[A / nrm[:, None], b / nrm]


def _translate(planes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Planes of a body moved by ``t``: normals unchanged, ``b -> b - A·t``."""
    d = planes.shape[1] - 1
    out = planes.copy()
    out[:, d] = planes[:, d] - planes[:, :d] @ np.asarray(t, dtype=float)
    return out


def _cross_interval(planes: np.ndarray, xc: float, *, axis: int,
                    other: int) -> Tuple[float, float]:
    """``(lo, hi)`` of the body's cross-section along ``other`` at ``axis == xc``.

    Closed form, not sampled: on the plane ``axis = xc`` each facet ``A·p + b <= 0`` becomes a
    one-sided bound on the ``other`` coordinate, so the cross-section is the interval between
    the tightest of them. ``(0.0, 0.0)`` when the cut misses the body.
    """
    d = planes.shape[1] - 1
    A, b = planes[:, :d], planes[:, d]
    c = -(A[:, axis] * xc + b)                     # A_other * y <= c
    ao = A[:, other]
    hi, lo = np.inf, -np.inf
    pos, neg = ao > 1e-12, ao < -1e-12
    if np.any(pos):
        hi = float(np.min(c[pos] / ao[pos]))
    if np.any(neg):
        lo = float(np.max(c[neg] / ao[neg]))
    flat = ~(pos | neg)
    if np.any(flat) and np.any(c[flat] < -1e-12):
        return 0.0, 0.0                            # infeasible: the cut misses the body
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return 0.0, 0.0
    return lo, hi


def _cross_half_extent(planes: np.ndarray, xc: float, *, axis: int, other: int) -> float:
    lo, hi = _cross_interval(planes, xc, axis=axis, other=other)
    return 0.5 * (hi - lo)


def _solve_cut_offset(planes0: np.ndarray, target_half: float, r_hint: float,
                      *, axis: int, other: int, sign: float = 1.0) -> float:
    """Distance from the centre at which the cross-section has half-extent ``target_half``.

    ``sign`` selects **which side** of the centre the cut is on, and it is not optional. These
    bodies are lobed polygons with an arbitrary rotation, so the cross-section at ``+xc`` and
    at ``−xc`` are different shapes; a solve that always evaluated ``+xc`` and was then used
    for the body sitting on the other side of the plane delivered 6.39 µm for a 4.74 µm
    request. The body on the right of the cut has its cut at ``−off`` in its own frame.

    Solved by bisection on the closed-form cross-section rather than derived from the radius.
    The disc identity ``sqrt(r² − w²)`` is what a first pass used and it is **wrong for a
    polygon**: `_polygon_planes` puts the facets at distance ~r, so the corners reach
    ``r/cos(pi/n)`` and the chord at that offset came out 10.44 µm for the same request.
    """
    lo, hi = 0.0, r_hint * 3.0
    if _cross_half_extent(planes0, 0.0, axis=axis, other=other) < target_half:
        return lo                                  # even the centre cut is narrower
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _cross_half_extent(planes0, sign * mid, axis=axis, other=other) > target_half:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _shared_facet(pa: np.ndarray, pb: np.ndarray, mid: float, *, axis: int,
                  other: int) -> Tuple[float, float]:
    """``(lo, hi)`` of the facet the two bodies actually SHARE at ``axis == mid``.

    The shared facet is the intersection of the two cross-sections, so it is narrower than
    either whenever they are not concentric. This is measured from the bodies as built and
    becomes the ground truth the split move is graded against — a requested width is a target,
    an achieved width is evidence, and only one of them belongs in the metadata.
    """
    la, ha = _cross_interval(pa, mid, axis=axis, other=other)
    lb, hb = _cross_interval(pb, mid, axis=axis, other=other)
    lo, hi = max(la, lb), min(ha, hb)
    return (lo, hi) if hi > lo else (0.0, 0.0)


def _clip_pair(pa: np.ndarray, pb: np.ndarray, sep_axis: int,
               ca: np.ndarray, cb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut two overlapping bodies along the plane midway between their centres.

    Returns the two clipped plane sets and the cutting plane. This is how a *contact* is
    made exact: the two bodies then share that plane as a common facet, so ``s_a = s_b = 0``
    on it and the analytic answer to "where is the boundary" is a single equation rather than
    a rasterisation artefact.
    """
    d = np.zeros(len(ca))
    d[sep_axis] = 1.0
    mid = 0.5 * (ca[sep_axis] + cb[sep_axis])
    lo, hi = (ca, cb) if ca[sep_axis] < cb[sep_axis] else (cb, ca)
    plane_lo = np.r_[d, -mid]                       # keep  x <= mid
    plane_hi = np.r_[-d, mid]                       # keep  x >= mid
    if ca[sep_axis] < cb[sep_axis]:
        return np.vstack([pa, plane_lo]), np.vstack([pb, plane_hi]), plane_lo
    return np.vstack([pa, plane_hi]), np.vstack([pb, plane_lo]), plane_lo


def _rasterize(planes: np.ndarray, grid_um: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean interior mask of ``A x + b <= 0`` on a coordinate grid, facet by facet."""
    d = len(grid_um)
    A, b = planes[:, :d], planes[:, d]
    nrm = np.linalg.norm(A, axis=1)
    A, b = A / nrm[:, None], b / nrm
    s = None
    for i in range(A.shape[0]):
        pl = np.full(tuple(len(g) for g in grid_um), float(b[i]))
        for j, g in enumerate(grid_um):
            sh = [1] * d
            sh[j] = len(g)
            pl += A[i, j] * g.reshape(sh)
        s = pl if s is None else np.maximum(s, pl)
    return s <= 0.0


# ── the generator ────────────────────────────────────────────────────────────

def make_bed(*, dims: int = 2, shape: Optional[Sequence[int]] = None,
             voxel_um: Optional[Sequence[float]] = None,
             r_eq_um: float = 33.85, n_bulk: int = 40, n_sides: int = 7,
             lobing: float = 0.10, neck_frac: float = 0.14, broad_frac: float = 0.62,
             pack: float = 0.92, seam_um: float = 1.6,
             aspect_range: Tuple[float, float] = (1.0, 2.1),
             bite_depth_frac: float = 0.55,
             bite_width_frac: float = 0.55,
             intensity: Tuple[float, float] = (3700.0, 1500.0),
             background: float = 55.0, psf_sigma_um: float = 0.567,
             body_gain_cv: float = 0.06, texture_cv: float = 0.06,
             halo_amp: float = 1.40, halo_um: float = 3.5,
             saturate_at: float = 4095.0,
             poisson: bool = False, frac_b: float = 0.0, seed: int = 0) -> SynthBed:
    """A packed field of convex bodies with exact ownership and planted contacts.

    ``r_eq_um`` defaults to the **label-measured** M08 single-granule radius (3599 µm² →
    33.85 µm), and ``voxel_um``/``psf_sigma_um`` to this acquisition's real values, so the
    fixture sits at the scale of the data rather than at a convenient one.

    ``neck_frac`` and ``broad_frac`` are the contact half-width as a fraction of ``r_eq``. The
    neck default 0.14 is just under ``w_min/r_eq = 0.15`` — i.e. deliberately *inside* the
    régime the surface energy is calibrated to forbid — and the broad default 0.62 is far
    outside it. If the split move cannot separate the first and preserve the second, the
    energy is not doing what the derivation says.

    ``surface_gap_um`` separates the tangent pairs' surfaces; 0.0 is exact tangency.
    """
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    rng = np.random.default_rng(seed)
    px = 1.7182777601481225
    voxel_um = tuple(voxel_um) if voxel_um is not None else ((px, px) if dims == 2
                                                            else (40.0, px, px))
    shape = tuple(shape) if shape is not None else ((320, 320) if dims == 2 else (5, 320, 320))
    if len(shape) != dims or len(voxel_um) != dims:
        raise ValueError(f"shape/voxel_um must have {dims} entries")
    grid = [np.arange(n, dtype=np.float64) * v for n, v in zip(shape, voxel_um)]
    span = np.array([g[-1] for g in grid])
    lat = slice(1, None) if dims == 3 else slice(0, None)
    half_z = 2.5 * voxel_um[0] if dims == 3 else 0.0

    bodies: List[Body] = []
    contacts: List[Tuple[int, int, np.ndarray, str]] = []
    #: the ACHIEVED shared-facet half-widths, measured from the bodies as built
    widths: Dict[str, float] = {}
    gid = 0

    def _centre(y, x):
        return np.array([span[0] * 0.5, y, x]) if dims == 3 else np.array([y, x])

    def _planes(c, r, ph, ns=None, asp=1.0):
        # 5-7 straight sides per body: the visual QC that settled the convex-polytope model
        # described "five to seven straight sides and sharp corners", and a fixed side count
        # makes every body the same shape, which is a symmetry the real material does not have.
        # Built at the ORIGIN so the aspect scaling is about the body's own centre, then moved.
        ns = int(ns if ns is not None else rng.integers(5, n_sides + 1))
        z0 = _centre(0.0, 0.0)
        p = (_prism_planes(z0, r, ns, ph, lobing, half_z, rng) if dims == 3
             else _polygon_planes(z0, r, ns, ph, lobing, rng))
        if asp != 1.0:
            # Elongate along x and shorten along y by the same factor, so the body's AREA is
            # preserved. Scaling one axis only would make an elongated body proportionally
            # larger, and the solid fraction climbed from 0.474 to 0.555 doing exactly that —
            # aspect ratio is a shape parameter, not a size one.
            sv = np.ones(dims)
            sv[dims - 1] = np.sqrt(asp)
            sv[dims - 2] = 1.0 / np.sqrt(asp)
            p = _scale_planes(p, sv)
        return _translate(p, np.asarray(c, dtype=float) - z0)

    def _add(c, r, ph, role, kind, *, asp=1.0, bite=None):
        nonlocal gid
        gid += 1
        bodies.append(Body(gid=gid, kind=kind, centre_um=c,
                           planes=_planes(c, r, ph, asp=asp), r_eq_um=r, role=role,
                           bite=bite, aspect=float(asp)))
        return gid

    def _pair(y, x, contact_frac, role_a, role_b):
        """Two bodies whose shared facet has half-width exactly ``contact_frac*r_eq``.

        The centre offset is **solved** for the body that is actually built, by bisecting the
        closed-form cross-section, rather than taken from the disc identity
        ``sqrt(r² − w²)``. For a polygon the facets sit at radius `r` while the corners reach
        past it, so the disc formula over-delivers the width by more than 2x — and the planted
        width is precisely the number the split move is graded against.
        """
        nonlocal gid
        w = contact_frac * r_eq_um
        ax, oth = (2, 1) if dims == 3 else (1, 0)          # cut along x, measure along y
        p0a = _planes(_centre(0.0, 0.0), r_eq_um, 0.20)
        p0b = _planes(_centre(0.0, 0.0), r_eq_um, 0.95)
        off_a = _solve_cut_offset(p0a, w, r_eq_um, axis=ax, other=oth, sign=+1.0)
        off_b = _solve_cut_offset(p0b, w, r_eq_um, axis=ax, other=oth, sign=-1.0)
        # The shared facet is the INTERSECTION of the two cross-sections, not either one.
        # The bodies carry different rotations, so their cross-sections at the cut plane sit
        # at different `other` positions and the overlap comes out narrower than requested
        # (measured 15.32 µm against 20.99 for the broad case). Re-centre each cross-section
        # on the pair's own axis first; a translation along `other` shifts an interval without
        # changing its extent, so the solved width survives exactly.
        lo_a, hi_a = _cross_interval(p0a, +off_a, axis=ax, other=oth)
        lo_b, hi_b = _cross_interval(p0b, -off_b, axis=ax, other=oth)
        sh_a = np.zeros(dims)
        sh_b = np.zeros(dims)
        sh_a[oth] = -0.5 * (lo_a + hi_a)
        sh_b[oth] = -0.5 * (lo_b + hi_b)
        ca, cb = _centre(y, x - off_a) + sh_a, _centre(y, x + off_b) + sh_b
        pa = _translate(p0a, ca - _centre(0.0, 0.0))
        pb = _translate(p0b, cb - _centre(0.0, 0.0))
        # the two bodies were solved so that x = mid cuts each at half-width w; put the
        # cutting plane exactly there rather than at the midpoint of unequal offsets
        mid = float(x)
        dvec = np.zeros(dims)
        dvec[ax] = 1.0
        pa = np.vstack([pa, np.r_[dvec, -mid]])            # A keeps  x <= mid
        pb = np.vstack([pb, np.r_[-dvec, mid]])            # B keeps  x >= mid
        plane = np.r_[dvec, -mid]
        gid += 1
        ga = gid
        bodies.append(Body(gid=ga, kind=KIND_A, centre_um=ca, planes=pa, r_eq_um=r_eq_um,
                           role=role_a, partner=ga + 1))
        gid += 1
        gb = gid
        bodies.append(Body(gid=gb, kind=KIND_A, centre_um=cb, planes=pb, r_eq_um=r_eq_um,
                           role=role_b, partner=ga))
        f_lo, f_hi = _shared_facet(pa, pb, mid, axis=ax, other=oth)
        achieved = 0.5 * (f_hi - f_lo)
        contacts.append((ga, gb, plane, role_a.split("_")[0]))
        widths[role_a.split("_")[0]] = achieved
        return ga, gb, achieved

    # planted pairs, well separated from one another and from the bulk
    y0 = span[-2] * 0.22 if dims == 2 else span[-2] * 0.22
    _pair(y0, span[-1] * 0.30, neck_frac, "neck_a", "neck_b")
    _pair(y0, span[-1] * 0.74, broad_frac, "broad_a", "broad_b")
    # an isolated body: nothing near it, so E_∅ must claim the space around it
    _add(_centre(span[-2] * 0.86, span[-1] * 0.14), r_eq_um * 0.8, 0.4, "lone", KIND_A)
    # A SINGLE body with a concave face — the negative control for the split move, added after
    # the user pointed out that real granules are not all convex and "some parts of the granule
    # may even have some concave parts of their faces". That matters more than it sounds: a
    # concave face lowers solidity EXACTLY the way a thin neck does, so a split move that
    # proposes on convexity will fire on a perfectly good single granule.
    #
    # The defaults are chosen so this body is indistinguishable from the necked pair BY
    # SOLIDITY — 0.854 against the pair's 0.846 — while an opening sweep finds no radius that
    # separates it into two granule-sized pieces. So only the presence of a thin cut tells them
    # apart, which is precisely what `ΔE = γ̃·A_neck − gain` is supposed to notice. If the split
    # move splits this body, its proposal is doing the deciding and the energy is decorative.
    cc = _centre(span[-2] * 0.86, span[-1] * 0.52)
    cp = _planes(cc, r_eq_um * 1.05, 1.1, asp=1.0)
    bd = bite_depth_frac * r_eq_um
    bw = bite_width_frac * r_eq_um
    # An axis-aligned box removed from the +x face. Column indices are EXPLICIT: the normal
    # occupies columns 0..dims-1 and the offset is column `dims`, so a negative index lands on
    # the OFFSET and silently produces a zero normal (which then divides by zero on
    # renormalisation). That is exactly what a first version did.
    ix, iy, ioff = dims - 1, dims - 2, dims
    bn = np.zeros((4, dims + 1))
    xc0 = float(cc[ix]) + r_eq_um * 1.05 - bd            # the dent reaches bd into the body
    bn[0, ix], bn[0, ioff] = +1.0, -(xc0 + 3.0 * bd)     # x <= xc0 + 3bd  (well outside)
    bn[1, ix], bn[1, ioff] = -1.0, +xc0                  # x >= xc0
    bn[2, iy], bn[2, ioff] = +1.0, -(float(cc[iy]) + bw)
    bn[3, iy], bn[3, ioff] = -1.0, +(float(cc[iy]) - bw)
    _add(cc, r_eq_um * 1.05, 1.1, "concave", KIND_A, bite=bn)
    bodies[-1] = Body(**{**bodies[-1].__dict__, "planes": cp})

    # ── bulk: dart-thrown CLOSE ENOUGH TO TOUCH, then overlaps clipped ───────
    # A first version kept a circumradius hard core so nothing could overlap, and it produced
    # a fixture with wide dark gaps and only the two planted contacts — visibly unlike the
    # real bed, where granules touch almost everywhere with thin dark seams. A validation set
    # for BOUNDARY placement that contains two boundaries is not a validation set.
    #
    # So bodies are now placed close enough to interpenetrate and every overlapping pair is
    # resolved the same way the planted pairs are: cut both by the plane midway between their
    # centres. That plane becomes a facet of BOTH, so `s_a = s_b = 0` on it, ownership stays
    # single-valued, and each body's `planes` still exactly describe the voxels it owns. A
    # contact is created rather than avoided, which is what the bed actually looks like.
    circ = 1.0 / np.cos(np.pi / n_sides) * (1.0 + lobing)
    keep_clear = [(b.centre_um[lat], b.r_eq_um * circ * 1.55) for b in bodies]
    bulk: List[Tuple[np.ndarray, float]] = []
    tries = 0
    while len(bulk) < n_bulk and tries < 40000:
        tries += 1
        r_i = r_eq_um * rng.uniform(0.86, 1.10)
        c_lat = np.array([rng.uniform(r_i, span[-2] - r_i),
                          rng.uniform(r_i, span[-1] - r_i)])
        # stay off the planted pairs entirely — their facet widths are the ground truth and a
        # bulk body encroaching on one would silently re-cut it
        if any(np.linalg.norm(c_lat - q) < clr + r_i * circ for q, clr in keep_clear):
            continue
        # `pack < 1` lets neighbours overlap; the clip below turns that into a shared facet
        if any(np.linalg.norm(c_lat - q) < pack * (r_i + r_j) for q, r_j in bulk):
            continue
        bulk.append((c_lat, r_i))
    bulk_asp = [float(rng.uniform(*aspect_range)) for _ in bulk]
    bulk_gids = [_add(_centre(*c), r, rng.uniform(0, 2 * np.pi), "bulk",
                      KIND_B if rng.uniform() < frac_b else KIND_A, asp=a)
                 for (c, r), a in zip(bulk, bulk_asp)]

    # resolve every overlapping bulk pair into an exact shared facet
    by_gid = {b.gid: i for i, b in enumerate(bodies)}
    extra: Dict[int, List[np.ndarray]] = {g: [] for g in bulk_gids}
    pending: List[Tuple[int, int, np.ndarray]] = []
    for i, ga in enumerate(bulk_gids):
        ca, ra = bulk[i]
        ra = ra
        for j in range(i + 1, len(bulk_gids)):
            gb = bulk_gids[j]
            cb, rb = bulk[j]
            dv = cb - ca
            dist = float(np.linalg.norm(dv))
            # Clip GENEROUSLY. `circ` is the circumradius factor of an equant polygon and
            # ignores the aspect elongation, so a pair of long bodies could clear this test and
            # still overlap — measured as 1.07e-04 of voxels claimed by two bodies. An extra
            # midplane that misses both bodies is a constraint that never binds, so
            # over-clipping costs nothing while under-clipping breaks the ownership invariant.
            reach = circ * np.sqrt(max(aspect_range[1], 1.0))
            if dist >= (ra + rb) * reach:
                continue
            u = np.zeros(dims)
            u[lat] = dv / dist
            mid = 0.5 * float(u @ (_centre(*ca) + _centre(*cb)))
            extra[ga].append(np.r_[u, -mid])             # A keeps  u·p <= mid
            extra[gb].append(np.r_[-u, mid])             # B keeps  u·p >= mid
            pending.append((ga, gb, np.r_[u, -mid]))
    for g, planes_extra in extra.items():
        if planes_extra:
            k = by_gid[g]
            bodies[k] = Body(**{**bodies[k].__dict__,
                                "planes": np.vstack([bodies[k].planes] + planes_extra)})


    # ── a real fluid SEAM between touching bodies ────────────────────────────
    # Clipping two bodies by their midplane makes them share a facet with ZERO gap, and a
    # zero gap has no intensity signature at all: measured, the seam read 1.0000 of the
    # interior median, i.e. no dip whatever. Real granule surfaces are separated by a thin
    # interstitial layer, and the real seam measures 0.870 of the interior at M08.
    #
    # Eroding every body by seam/2 produces exactly that: a `seam_um`-wide dark gap at every
    # contact, and a sub-pixel change to the free surface elsewhere. The recorded clip plane
    # is still the true divide, and at that plane BOTH bodies now sit at s = +seam/2 — equal,
    # so the confidence still crosses 0.5 there, and the background class gets exercised too.
    # The eroded planes replace the originals so `s = 0` remains the boundary of the region a
    # body actually owns.
    if seam_um > 0:
        bodies = [Body(**{**b.__dict__,
                          "planes": np.c_[b.planes[:, :dims],
                                          b.planes[:, dims]
                                          + 0.5 * seam_um * np.linalg.norm(
                                              b.planes[:, :dims], axis=1)]})
                  for b in bodies]

    # ── ownership, lowest gid wins any (constructed-away) tie ────────────────
    owner = np.zeros(shape, dtype=np.int32)
    for b in bodies:
        m = _rasterize(b.planes, grid)
        if b.bite is not None:
            m &= ~_rasterize(b.bite, grid)          # the concave dent
        owner[m & (owner == 0)] = b.gid

    # ── which clipped pairs are REAL contacts ────────────────────────────────
    # The clip is deliberately generous, so some midplanes never bind and the pair does not
    # actually touch. Confirming that from the planes would need the support function of an
    # OBLIQUE cut, and `_cross_interval` only handles axis-aligned ones — a first attempt used
    # it anyway and confirmed zero contacts out of 37. The ownership map answers it directly and
    # for any orientation: two bodies are in contact iff their painted regions come within the
    # seam of each other.
    if pending:
        from scipy.ndimage import binary_dilation as _bd
        reach_vox = max(2, int(np.ceil(seam_um / min(voxel_um))) + 1)
        for ga, gb, pl in pending:
            ma = owner == ga
            if not ma.any():
                continue
            if np.any(_bd(ma, iterations=reach_vox) & (owner == gb)):
                contacts.append((ga, gb, pl, "bulk"))

    # ── intensity ────────────────────────────────────────────────────────────
    # The default level for class A is ABOVE the 12-bit ceiling on purpose, so its interiors
    # clip to 4095 exactly as the real R-B channel does (2.2% of its pixels are saturated).
    # That is not cosmetic: it means two touching same-class bodies are intensity-identical
    # inside, so the split between them CANNOT come from intensity and must come from
    # geometry — which is the premise of this entire project. A first version gave each body
    # its own brightness (gain CV 0.22), and a side-by-side against real data showed adjacent
    # bodies separable by grey level alone. That would have let the energy's intensity term do
    # work it cannot do on the real bed, and made every result optimistic.
    # `body_gain_cv` is therefore small and within-class only; `texture_cv` adds interior
    # structure without making neighbours distinguishable.
    #
    # The defaults for `intensity`, `halo_amp` and `halo_um` were TUNED AGAINST MEASUREMENTS of
    # the real R-B channel at M08, not chosen for looks, because these three numbers are the
    # difficulty of the problem:
    #     saturated fraction        1.53%   vs real 2.23%
    #     contact seam / interior   0.874   vs real 0.870   <- the one that matters
    #     far void / interior       0.033   vs real ~0.03
    # A shallower seam makes every boundary trivial to find; a deeper one makes the fixture
    # easier than the data and every result optimistic.
    img = np.full(shape, float(background), dtype=np.float64)
    for b in bodies:
        lvl = float(intensity[b.kind]) * float(
            np.clip(1.0 + body_gain_cv * rng.standard_normal(), 0.35, 2.0))
        img[owner == b.gid] = lvl
    if texture_cv > 0:
        fgm = owner > 0
        tex = rng.standard_normal(shape)
        try:
            from scipy.ndimage import gaussian_filter as _gf
            tex = _gf(tex, [max(2.0, 4.0 / v * v) for v in voxel_um])
            tex /= max(float(tex.std()), 1e-9)
        except ImportError:                                        # pragma: no cover
            pass
        img[fgm] *= 1.0 + texture_cv * tex[fgm]
    # ── the interstitial halo — this is what sets the DIFFICULTY ─────────────
    # A seam of pure background reads 0.026 of the interior; the real seam at M08 reads 0.870,
    # only 13% dimmer. The difference is the halo the plan records: "every granule carries a
    # soft bright halo fading into the dark gap, from the PSF and probably dye in the
    # interstitial fluid". That halo is precisely why two touching granules are hard to
    # separate, so a fixture without it is not a fixture for this problem — it makes every
    # boundary trivially findable from intensity.
    if halo_amp > 0 and halo_um > 0:
        try:
            from scipy.ndimage import distance_transform_edt as _edt
            d = _edt(owner == 0, sampling=voxel_um)
            lvl = float(np.median(img[owner > 0])) if np.any(owner > 0) else 0.0
            img = np.where(owner == 0,
                           img + lvl * halo_amp * np.exp(-d / float(halo_um)), img)
        except ImportError:                                        # pragma: no cover
            pass
    if psf_sigma_um > 0:
        try:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, [psf_sigma_um / v for v in voxel_um])
        except ImportError:                                        # pragma: no cover
            pass
    if poisson:
        img = rng.poisson(np.clip(img, 0, None)).astype(np.float64)
    if saturate_at:
        img = np.minimum(img, float(saturate_at))

    img6 = img.reshape((1, 1) + ((1,) if dims == 2 else (shape[0],))
                       + (1,) + tuple(shape[lat])).astype(np.float32)
    meta = {"pixel_size_um": float(voxel_um[-1]), "bit_depth": 12,
            "objective_na": 0.45, "channel_emission_nm": [571.0],
            "psf_sigma_um": float(psf_sigma_um), "r_eq_um": float(r_eq_um),
            # ACHIEVED, measured from the bodies as built — this is the ground truth the
            # split move is graded against. The requested values are kept beside them so a
            # solve that drifts is visible rather than silently redefining the target.
            "neck_half_width_um": float(widths.get("neck", 0.0)),
            "broad_half_width_um": float(widths.get("broad", 0.0)),
            "neck_half_width_requested_um": float(neck_frac * r_eq_um),
            "broad_half_width_requested_um": float(broad_frac * r_eq_um),
            "n_bodies": len(bodies), "seed": int(seed), "dims": dims,
            "n_contacts": len(contacts), "pack": float(pack), "seam_um": float(seam_um), "halo_amp": float(halo_amp), "halo_um": float(halo_um),
            "aspect_range": [float(a) for a in aspect_range],
            "bite_depth_um": float(bite_depth_frac * r_eq_um),
            "solid_fraction": float((owner > 0).mean())}
    if dims == 3:
        meta["z_step_um"] = float(voxel_um[0])
    return SynthBed(image=img6, owner=owner, bodies=bodies, voxel_um=voxel_um,
                    contacts=contacts, metadata=meta)


def bed_to_dataset(bed: SynthBed):
    """``(Dataset, MetaEnvelope)`` so a selftest can wire the fixture in one line.

    Imported here rather than at module scope: this file must stay numpy-only at import time
    so the hygiene clause that forbids the catalog from importing it cannot be defeated by a
    transitive engine import.
    """
    from nodegraph.dataset import AxisSizes, Dataset
    from nodegraph.metadata import MetaEnvelope
    from nodegraph.provider import ArrayProvider
    s = bed.image.shape
    ax = AxisSizes(m=s[0], t=s[1], z=s[2], c=s[3], y=s[4], x=s[5])
    md = dict(bed.metadata)
    ds = Dataset(axes=ax, metadata=md).with_image(ArrayProvider(bed.image.astype(np.float64)))
    return ds, MetaEnvelope(axes=ax, metadata=md)
