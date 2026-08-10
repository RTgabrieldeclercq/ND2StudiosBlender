"""Physical placement — where a Dataset's voxels actually are on the microscope.

Every other module in this engine addresses data by INDEX: voxel ``(m, t, z, c, y, x)``.
That is sufficient right up until you want to show two *files* together, at which point
index space stops meaning anything — the WellA3 640 file's ``m=0`` and the GFP file's
``m=0`` are 5 mm apart on the same plate. This module is the one place that converts an
index to a **position in the microscope's own µm frame**, and answers the questions
``view.overlay`` has to ask before it can draw anything:

* where is multipoint *m*'s field? (:func:`field_box`)
* which of the OTHER file's multipoints cover it, and how much? (:func:`tiles_covering`)
* which timepoints correspond, and how wrong is that pairing? (:func:`pair_timepoints`)
* where in absolute Z does slice *k* sit? (:func:`z_um_of_slice`)

**Everything here is best-effort and says so.** Each function returns ``None`` or an empty
result when the file did not carry what it needs, and :func:`plan_placement` sorts the
failures into the two classes agreed for this node: REFUSALS, where a missing or stale
input would still render a plausible-looking picture that is wrong, and WARNINGS, where the
loss is visible on screen anyway. Nothing here ever substitutes a default for an absent
measurement — that is the whole reason the ``stage_z_um``/``frame_time_jd`` readers drop a
partial list instead of padding it.

Qt-free on purpose, and everything except :func:`axis_map` is pure arithmetic over metadata
dicts — so it is testable headlessly against real files without opening an image.
:func:`axis_map` is the one function here that produces an INDEX array rather than an
answer about geometry; it lives here rather than beside its first caller because both the
Viewer's compositor and ``registration.align_to`` need to resample one field onto another's
grid, and the engine layer cannot import the GUI one.

Coordinate conventions
----------------------
``stage_xy_um[m]`` is the field **CENTRE**; ``stage_z_um[m]`` is that position's **nominal
focus**, which the ``ZStackLoop``'s ``z_home_index`` ties to a specific slice. Image ``+x``
/ ``+y`` are taken to run along stage ``+x``/``+y``, mirrored by ``flip_x``/``flip_y``
exactly as :func:`nodegraph.nodes._stitch_offsets_stage` does — the same defaults, because a
mosaic and an overlay of the same file disagreeing about handedness would be indefensible.

Note that a flip mirrors *sampling direction*, never *extent*: a field centred at ``x`` with
width ``W`` occupies ``[x - W/2, x + W/2]`` whichever way the camera is mounted. So every
overlap/coverage answer in this module is flip-independent, and handedness only enters when
pixels are actually fetched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FieldBox", "SECONDS_PER_DAY", "field_box", "z_um_of_slice", "lateral_extent_um",
    "overlap_fraction", "axis_map", "translate", "ALIGN_KEY", "tiles_covering", "pair_timepoints",
    "PlacementPlan", "plan_placement", "compose_secondary_plane", "paired_t",
    "context_extent", "sub_field_box", "source_window",
    "secondary_z_index", "ZGrid", "merge_z_grid", "Z_GRID_BLOWUP",
]

#: Julian day → seconds. ``frame_time_jd`` is the only clock two files share, and it is in
#: days, so every time difference this module reports goes through here.
SECONDS_PER_DAY = 86400.0

#: Two fields are treated as describing the same field of view when their physical extents
#: agree to within this fraction. Deliberately loose: a resample by a non-integer factor
#: lands the extent a fraction of a pixel off, and refusing over that would be pedantry.
_EXTENT_TOL = 0.02

#: Coverage shortfall below this fraction is not reported as a gap.
#:
#: Montage tiles do not abut to the micron. The WellA3 GFP scan steps **1759.7 µm** between
#: rows while one field is **1759.52 µm** wide, leaving a 0.18 µm seam — 0.06% of the
#: primary's 294 µm field, about a third of one primary pixel. Testing coverage against an
#: exact 1.0 flagged five of twelve fields as "only partly covered (down to 100%)", which is
#: the precise shape of a warning that trains people to ignore warnings. The floor is set
#: well above stage rounding and well below anything you could see.
_COVERAGE_TOL = 0.01


@dataclass(frozen=True)
class FieldBox:
    """One multipoint's physical extent in microscope µm — lateral bounds plus the axial
    span the stack occupies. ``z0 == z1`` for a single plane, and both are ``None`` when
    the file carried no focus log (a box that can be placed laterally but not axially,
    which is a normal and useful state — the WellA3 GFP tiles are exactly that until you
    ask about Z)."""

    y0: float
    y1: float
    x0: float
    x1: float
    z0: Optional[float] = None
    z1: Optional[float] = None

    @property
    def area_um2(self) -> float:
        return max(0.0, self.y1 - self.y0) * max(0.0, self.x1 - self.x0)

    @property
    def centre(self) -> Tuple[float, float]:
        return (0.5 * (self.y0 + self.y1), 0.5 * (self.x0 + self.x1))


def _seq(md: Mapping[str, Any], key: str) -> Sequence:
    v = md.get(key)
    return v if isinstance(v, (list, tuple)) else ()


def lateral_extent_um(md: Mapping[str, Any], axes: Any) -> Optional[Tuple[float, float]]:
    """The ``(height, width)`` of one field in µm, or ``None`` without a pixel size.

    This is what makes two differently-sampled files comparable: the WellA3 pair are both
    1024², but at 0.287 vs 1.718 µm/px they are a 294 µm field and a 1760 µm field — a
    36× area difference that index space cannot see.
    """
    ps = md.get("pixel_size_um")
    try:
        ps = float(ps)
    except (TypeError, ValueError):
        return None
    if not (ps > 0):
        return None
    return (float(axes.y) * ps, float(axes.x) * ps)


def z_um_of_slice(md: Mapping[str, Any], axes: Any, m: int,
                  k: float) -> Optional[float]:
    """Absolute µm focus of z-slice ``k`` at multipoint ``m`` — ``None`` if unknowable.

    ``stagePositionUm.z`` is constant down a stack (it is the position's nominal focus, not
    a per-slice reading), so a stack needs the ``ZStackLoop`` anchoring to place a slice:

        ``z_um(k) = stage_z_um[m] + dir * (k - z_home_index) * z_step_um``

    with ``dir = +1`` when ``z_bottom_to_top`` else ``-1``. A **single-plane** file needs
    none of that — its one slice IS the nominal focus — which is why ``n_z == 1`` returns
    early. That early return is what lets the WellA3 GFP file (no Z loop at all, so
    ``z_home_index is None``) still be placed in Z.

    Returns ``None`` rather than guessing when a multi-slice stack has no anchoring: the
    plausible guess (``home = 0``) is wrong by up to the full stack depth, which for the
    640 file is 60 µm — a whole cell layer.
    """
    zs = _seq(md, "stage_z_um")
    if not (0 <= int(m) < len(zs)):
        return None
    try:
        z_nom = float(zs[int(m)])
    except (TypeError, ValueError):
        return None
    if int(getattr(axes, "z", 1) or 1) <= 1:
        return z_nom
    home = md.get("z_home_index")
    step = md.get("z_step_um")
    try:
        home = int(home)
        step = float(step)
    except (TypeError, ValueError):
        return None
    direction = 1.0 if md.get("z_bottom_to_top", True) else -1.0
    return z_nom + direction * (float(k) - home) * step


#: Namespaced, non-calibration key holding `registration.align_to`'s measured correction —
#: a list of ``[dy_um, dx_um]``, one per multipoint (`wire-node-v2` §7b stamp-and-inherit).
#:
#: NOT a change to ``origin_um``, deliberately. The correction is measured FROM PIXELS, and a
#: ``meta_transform`` touches none — so an origin the node rewrote could never be predicted at
#: edit time, and the envelope would disagree with the payload for every downstream
#: ``ctx.calib`` read. Riding as separate provenance keeps the calibration prediction exact
#: and still reaches every consumer, because `field_box` adds it.
ALIGN_KEY = "__align_um__"


def _align_shift(md: Mapping[str, Any], m: int) -> Tuple[float, float]:
    """``(dy, dx)`` µm correction for multipoint ``m`` — ``(0, 0)`` when none was measured."""
    rows = _seq(md, ALIGN_KEY)
    if not (0 <= int(m) < len(rows)):
        return (0.0, 0.0)
    try:
        return (float(rows[int(m)][0]), float(rows[int(m)][1]))
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)


def field_box(md: Mapping[str, Any], axes: Any, m: int) -> Optional[FieldBox]:
    """Multipoint ``m``'s field as a :class:`FieldBox`, or ``None`` if it cannot be placed.

    Lateral placement needs ``stage_xy_um[m]`` and ``pixel_size_um``; axial placement needs
    the focus log too, and is simply omitted (``z0 = z1 = None``) when that is missing
    rather than failing the whole box — a lateral-only placement is what drives tile
    selection, and it is the common case for a widefield montage.
    """
    ext = lateral_extent_um(md, axes)
    if ext is None:
        return None
    h, w = ext
    # A measured alignment correction, if `registration.align_to` left one. Applied HERE so
    # a single change reaches every placement consumer — the overlay, the tile query, the
    # coverage report — instead of each one having to remember to add it.
    ady, adx = _align_shift(md, m)

    # `origin_um` FIRST: it is the transform-maintained corner, so it is still true after a
    # crop or a stitch, where `stage_xy_um` has silently stopped describing the data. The
    # stage log remains the fallback for a Dataset that never went through the calibration
    # seam (a hand-built fixture, an older saved checkpoint).
    origins = md.get("origin_um")
    if isinstance(origins, (list, tuple)) and 0 <= int(m) < len(origins):
        try:
            oz, oy, ox = (float(v) for v in origins[int(m)])
        except (TypeError, ValueError):
            oz = None                                     # malformed → fall through
        else:
            nz = int(getattr(axes, "z", 1) or 1)
            step = md.get("z_step_um")
            try:
                depth = (nz - 1) * float(step) if (nz > 1 and step) else 0.0
            except (TypeError, ValueError):
                depth = 0.0
            return FieldBox(y0=oy + ady, y1=oy + h + ady,
                            x0=ox + adx, x1=ox + w + adx,
                            z0=oz, z1=oz + depth)

    xy = _seq(md, "stage_xy_um")
    if not (0 <= int(m) < len(xy)):
        return None
    try:
        cx, cy = float(xy[int(m)][0]), float(xy[int(m)][1])
    except (TypeError, ValueError, IndexError):
        return None
    nz = int(getattr(axes, "z", 1) or 1)
    z_lo = z_um_of_slice(md, axes, m, 0)
    z_hi = z_um_of_slice(md, axes, m, nz - 1) if nz > 1 else z_lo
    if z_lo is not None and z_hi is not None and z_hi < z_lo:
        z_lo, z_hi = z_hi, z_lo          # a top-to-bottom stack still spans a range
    return FieldBox(y0=cy - 0.5 * h + ady, y1=cy + 0.5 * h + ady,
                    x0=cx - 0.5 * w + adx, x1=cx + 0.5 * w + adx,
                    z0=z_lo, z1=z_hi)


def overlap_fraction(a: FieldBox, b: FieldBox) -> float:
    """Fraction of ``a``'s lateral area covered by ``b`` (0 when they miss each other).

    Asymmetric on purpose — the question the overlay asks is always "how much of the
    PRIMARY's field does this secondary tile supply", and with a 36× area ratio the
    symmetric answer would be a meaningless 3%.
    """
    denom = a.area_um2
    if denom <= 0:
        return 0.0
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    if dy <= 0 or dx <= 0:
        return 0.0
    return (dy * dx) / denom


def axis_map(n_out: int, out_lo: float, out_hi: float,
             n_src: int, src_lo: float, src_hi: float,
             *, flip: bool) -> np.ndarray:
    """Index of the source sample under each output sample → ``(n_out,)`` int array.

    ``-1`` marks an output sample that falls outside the source's extent, so the caller can
    leave those pixels alone instead of clamping them to the edge — a clamp would smear the
    border tile's last row across the uncovered part of the field and make a gap look like
    data.

    Both extents are in µm; the sample centres are at ``lo + (i + 0.5) * span / n``.
    """
    if n_out <= 0 or n_src <= 0:
        return np.zeros(max(0, n_out), dtype=np.intp)
    out_span = (out_hi - out_lo) / float(n_out)
    src_span = (src_hi - src_lo) / float(n_src)
    if not (src_span > 0):
        return np.full(n_out, -1, dtype=np.intp)
    centres = out_lo + (np.arange(n_out, dtype=float) + 0.5) * out_span
    idx = np.floor((centres - src_lo) / src_span).astype(np.intp)
    inside = (idx >= 0) & (idx < n_src)
    if flip:
        idx = (n_src - 1) - idx
    return np.where(inside, idx, -1)


def sub_field_box(box: FieldBox, region: Tuple[float, float, float, float]) -> FieldBox:
    """The µm box of a fractional sub-rect ``(fy0, fy1, fx0, fx1)`` of ``box``'s IMAGE grid.

    The viewport shows a *rect* of the primary, not the whole field, and a detail patch read
    for that rect has to be composited against the µm extent the rect covers rather than the
    field's. Row fraction 0 is image row 0 — the same direction :func:`axis_map` walks the
    output in — so this is a straight linear interpolation and needs no handedness: a flip
    mirrors which SOURCE sample an output sample takes, never where the output sample is.
    Z is carried through untouched; a lateral zoom does not change focus.
    """
    fy0, fy1, fx0, fx1 = (float(v) for v in region)
    hy, hx = box.y1 - box.y0, box.x1 - box.x0
    return FieldBox(y0=box.y0 + fy0 * hy, y1=box.y0 + fy1 * hy,
                    x0=box.x0 + fx0 * hx, x1=box.x0 + fx1 * hx,
                    z0=box.z0, z1=box.z1)


def source_window(out_box: FieldBox, src_box: FieldBox, *, flip_y: bool = False,
                  flip_x: bool = True) -> Optional[Tuple[float, float, float, float]]:
    """Which fractional part of a source tile's PIXEL grid ``out_box`` needs, or ``None``.

    Returns ``(fy0, fy1, fx0, fx1)`` in ``[0, 1]`` of the tile's rows/columns — deliberately
    in *pixel* fractions rather than µm, because the caller's job is to read pixels and it
    should not have to re-derive the handedness rule to do it. ``None`` means the tile does
    not reach the box at all, which saves the read entirely.

    This is the inverse of :func:`axis_map`, and it exists so that an overlay can fetch **the
    pixels it is about to draw** instead of the whole tile: a stitched secondary is tens of
    thousands of pixels wide, of which a zoomed viewport wants a few hundred. Reading the
    whole plane and decimating it to a display cap is what made a magnified overlay a blur of
    the source's own overview rather than the source's data.
    """
    def part(o_lo: float, o_hi: float, s_lo: float, s_hi: float,
             flip: bool) -> Optional[Tuple[float, float]]:
        span = s_hi - s_lo
        if not (span > 0):
            return None
        lo, hi = max(o_lo, s_lo), min(o_hi, s_hi)
        if not (hi > lo):
            return None
        f0, f1 = (lo - s_lo) / span, (hi - s_lo) / span
        if flip:
            f0, f1 = 1.0 - f1, 1.0 - f0
        return max(0.0, min(1.0, f0)), max(0.0, min(1.0, f1))

    ry = part(out_box.y0, out_box.y1, src_box.y0, src_box.y1, bool(flip_y))
    rx = part(out_box.x0, out_box.x1, src_box.x0, src_box.x1, bool(flip_x))
    if ry is None or rx is None:
        return None
    return (ry[0], ry[1], rx[0], rx[1])


#: Minify by area-averaging once the source is more than this many samples per output sample.
#: Below it, point-sampling is what you want — it keeps the source's own values, and the
#: "blocky is honest" argument for a MAGNIFIED secondary depends on exactly that.
_SHRINK_AT = 1.5


def _area_shrink(plane: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """``plane`` block-averaged down to ``(ny, nx)`` — the minification :func:`axis_map` must
    not do by point-sampling.

    A secondary coarser than the primary is magnified, and nearest-neighbour is right there: it
    keeps real source values and makes the sampling grid visible. A secondary FINER than the
    primary is the opposite case and the same code was doing the same thing — on the WellA3 pair
    the 640 mosaic lands on 6031x1925 output pixels from 36093x11520, so 36 source pixels fall in
    each output pixel and ``axis_map`` kept exactly one of them. That is the defect
    :func:`nodelab_v2.runner._fit_plane` documents at length: it throws away ``1 - 1/36`` of the
    data while keeping the noise at full amplitude, so the result is *noisier* than the image it
    came from. Averaging the block instead is both quieter and more honest — every output pixel
    then reports what the sensor saw over that area.
    """
    h, w = plane.shape[:2]
    ny, nx = max(1, min(int(ny), h)), max(1, min(int(nx), w))
    if ny == h and nx == w:
        return plane
    ys = (np.arange(ny) * h) // ny
    xs = (np.arange(nx) * w) // nx
    acc = np.add.reduceat(np.add.reduceat(np.asarray(plane, dtype=np.float64), ys, axis=0),
                          xs, axis=1)
    counts = (np.diff(np.append(ys, h))[:, None] * np.diff(np.append(xs, w))[None, :])
    return acc / counts


def _window_extent(lo: float, hi: float, covered: Tuple[float, float],
                   flip: bool) -> Tuple[float, float]:
    """The µm span a *windowed* read covers, given the pixel fractions it spans.

    ``covered`` is the ``(f0, f1)`` slice of the tile's pixel grid the reader actually
    returned (snapped out to whole pixels of whatever level it chose). Under a flip the
    pixel grid runs against µm, so the window's µm bounds mirror — getting this backwards
    draws the right pixels in the wrong place, which looks like a placement error rather
    than a sampling one.
    """
    f0, f1 = float(covered[0]), float(covered[1])
    span = hi - lo
    if flip:
        return lo + (1.0 - f1) * span, lo + (1.0 - f0) * span
    return lo + f0 * span, lo + f1 * span


def paired_t(entry: Dict[str, Any], t: int) -> Optional[int]:
    """The secondary timepoint paired with primary ``t``, or ``None`` if unpaired.

    Read from the recipe rather than recomputed, so the picture can never disagree with the
    pairing the node card reports."""
    for row in entry.get("t_pairs") or ():
        if int(row[0]) == int(t):
            return None if row[1] is None else int(row[1])
    return None


def secondary_z_index(sec_md: Dict[str, Any], sec_axes: Any, sec_m: int,
                      z_um: Optional[float], *, dz: float = 0.0) -> int:
    """Which secondary slice to draw for a primary plane sitting at ``z_um``.

    A single-plane secondary has one answer and takes it — that is the WellA3 case, and it
    is why a 2D context view stays visible while you scroll a 210-slice stack rather than
    appearing on one slice and vanishing. A volumetric secondary picks the slice NEAREST in
    absolute µm, clamped into range, so scrolling the primary's Z walks the secondary's too.
    """
    nz = int(getattr(sec_axes, "z", 1) or 1)
    if nz <= 1 or z_um is None:
        return 0
    from nodegraph.placement import z_um_of_slice
    z0 = z_um_of_slice(sec_md, sec_axes, sec_m, 0)
    z1 = z_um_of_slice(sec_md, sec_axes, sec_m, nz - 1)
    if z0 is None or z1 is None or nz < 2 or z1 == z0:
        return 0
    frac = ((z_um - dz) - z0) / ((z1 - z0) / (nz - 1))
    return int(min(max(0, round(frac)), nz - 1))


@dataclass(frozen=True)
class ZGrid:
    """The Z axis a merged Dataset is expressed on: a uniform grid, per multipoint.

    ``z0_um[m]`` is field *m*'s first plane in absolute µm and ``step_um`` the spacing, so
    plane *k* of field *m* sits at ``z0_um[m] + k * step_um``. ``z0_um[m] is None`` means that
    field could not be placed axially at all (no focus log) — the grid then degenerates to the
    primary's own index space, which :meth:`plane_um` reports by returning ``None``.

    **Uniform on purpose, and that is a schema constraint rather than a simplification.** The
    calibration vocabulary describes Z as an origin plus a step plus a count
    (``origin_um``/``z_step_um``, read back by :func:`z_um_of_slice`); there is nowhere to put
    an irregular list of focus positions. So a "union" of two stacks is the finest-step uniform
    grid that SPANS both — which contains every acquired focus to within half a step, and which
    collapses to the finer stack's own grid whenever one range nests inside the other (the
    WellA3 pair: a single GFP plane at 5999.7 µm sits inside the 640 stack's 5972–6032 µm, so
    the union grid IS the 640's 210 planes). Claiming a non-uniform grid in metadata that says
    uniform would be a lie about where the pixels are.
    """

    z0_um: Tuple[Optional[float], ...]
    step_um: float
    n: int

    def plane_um(self, m: int, k: int) -> Optional[float]:
        """Absolute µm focus of plane ``k`` of field ``m`` (``None`` when unplaceable)."""
        z0 = self.z0_um[int(m)] if 0 <= int(m) < len(self.z0_um) else None
        return None if z0 is None else z0 + int(k) * self.step_um


def _z_span(md: Mapping[str, Any], axes: Any, m: int) -> Optional[Tuple[float, float]]:
    """Field ``m``'s axial span in absolute µm (``lo, hi``), or ``None`` without a focus log."""
    nz = int(getattr(axes, "z", 1) or 1)
    lo = z_um_of_slice(md, axes, int(m), 0)
    hi = z_um_of_slice(md, axes, int(m), nz - 1) if nz > 1 else lo
    if lo is None or hi is None:
        return None
    return (min(lo, hi), max(lo, hi))


def _z_step(md: Mapping[str, Any], axes: Any) -> Optional[float]:
    """A file's own Z spacing in µm, or ``None`` for a single plane / no step recorded."""
    if int(getattr(axes, "z", 1) or 1) <= 1:
        return None
    try:
        step = abs(float(md.get("z_step_um")))
    except (TypeError, ValueError):
        return None
    return step if step > 0 else None


#: Refuse a merged Z grid more than this many times either input's own plane count. Two stacks
#: focused far apart (one at 100 µm, one at 500) have a *union* spanning the gap, and at a fine
#: step that is thousands of planes of which almost none carry data from both files. That is not
#: a merge, it is an accident — and a silent one, since every plane still reads.
Z_GRID_BLOWUP = 4


def merge_z_grid(dst_md: Mapping[str, Any], dst_axes: Any,
                 src_md: Mapping[str, Any], src_axes: Any,
                 tiles: Mapping[int, Sequence[Tuple[int, float]]],
                 *, mode: str = "union", dz: float = 0.0
                 ) -> Tuple[ZGrid, List[str], List[str]]:
    """The Z grid a channel merge is expressed on → ``(grid, warnings, refusals)``.

    ``mode="union"`` spans both files' focus ranges at the finer of their two steps, so every
    acquired plane of either is addressable — the point being that which file you happened to
    wire as the primary must not decide whether you can see the other's stack.
    ``mode="primary"`` keeps the primary's own grid exactly, for when downstream measurements
    have to be expressed on it; the secondary's out-of-range planes are then unreachable, and
    that is reported rather than left to be discovered.

    Without a focus log on either side the grid falls back to the primary's index space and
    says so: a Z placement that cannot be computed must not be invented.
    """
    warnings: List[str] = []
    refusals: List[str] = []
    nm = int(getattr(dst_axes, "m", 1) or 1)
    p_step = _z_step(dst_md, dst_axes)
    s_step = _z_step(src_md, src_axes)
    if mode == "primary":
        # The primary's grid EXACTLY — its own step, not the finer of the two. Keeping the
        # primary's range at the secondary's finer step would silently multiply its plane count,
        # which is the one thing this mode promises not to do.
        step = p_step or 1.0
    else:
        steps = [s for s in (p_step, s_step) if s]
        step = min(steps) if steps else 1.0

    spans: List[Optional[Tuple[float, float]]] = []
    for m in range(nm):
        p = _z_span(dst_md, dst_axes, m)
        if p is None:
            spans.append(None)
            continue
        lo, hi = p
        if mode != "primary":
            for j, _frac in (tiles.get(m) or ()):
                s = _z_span(src_md, src_axes, int(j))
                if s is not None:
                    lo, hi = min(lo, s[0] + dz), max(hi, s[1] + dz)
        spans.append((lo, hi))

    if not any(s is not None for s in spans):
        # No axial placement anywhere: keep the primary's own axis and be explicit that each
        # channel is then paired BY INDEX in z, which is the one thing this node otherwise
        # never does.
        warnings.append(
            "no focus log on either input: Z is paired by INDEX, not by absolute µm — the "
            "channels are laterally placed but their planes are only assumed to correspond")
        return (ZGrid(tuple([None] * nm), step, max(1, int(getattr(dst_axes, "z", 1) or 1))),
                warnings, refusals)

    n = 1
    for s in spans:
        if s is not None:
            n = max(n, int(round((s[1] - s[0]) / step)) + 1)
    z0 = tuple(None if s is None else s[0] for s in spans)
    own = max(int(getattr(dst_axes, "z", 1) or 1), int(getattr(src_axes, "z", 1) or 1))
    if n > max(2, Z_GRID_BLOWUP * own):
        refusals.append(
            f"the two inputs' focus ranges are too far apart to merge in Z: a union grid at "
            f"{step:g} µm would be {n} planes, against {own} in the deeper input. They were "
            f"probably not focused on the same specimen — check the Nudge Z, or set the Z grid "
            f"Mode to 'primary' to keep this input's own planes.")
    if mode == "primary":
        missing = [m for m in range(nm)
                   for j, _f in (tiles.get(m) or ())
                   if (_z_span(src_md, src_axes, int(j)) or (0.0, 0.0))[1]
                   > (spans[m] or (0.0, 0.0))[1] + 0.5 * step]
        if missing:
            warnings.append(
                f"Z grid = primary: the secondary reaches deeper than this input does, so its "
                f"planes past {n} are not addressable (fields {sorted(set(missing))[:4]}…). "
                f"Use Z grid = union to walk the whole stack.")
    elif p_step and s_step and abs(p_step - s_step) > 1e-9:
        warnings.append(
            f"Z grid = union at the finer step ({step:g} µm of {max(p_step, s_step):g}), so the "
            f"coarser input repeats a plane across neighbouring steps — it has none of its own "
            f"to show there")
    return ZGrid(z0, step, n), warnings, refusals


def compose_secondary_plane(entry: Dict[str, Any],
                            out_shape: Tuple[int, int],
                            pri_md: Dict[str, Any], pri_axes: Any, pri_m: int,
                            sec_md: Dict[str, Any], sec_axes: Any,
                            read_tile: Any,
                            *, fill: float = 0.0,
                            region: Optional[Tuple[float, float, float, float]] = None
                            ) -> Optional[np.ndarray]:
    """The secondary's pixels for primary field ``pri_m``, on the primary's display grid.

    ``out_shape`` is the output plane's ``(H, W)`` — generally not the primary's native size,
    because the display path decimates to ``MAX_DISPLAY_DIM``; the mapping is done in µm, so
    that difference costs nothing and no caller has to track the decimation factor.

    ``region`` is a fractional ``(fy0, fy1, fx0, fx1)`` sub-rect of the primary's image grid
    that ``out_shape`` covers, or ``None`` for the whole field. It is what lets the Viewer
    compose onto a zoomed-in **detail patch**: the patch is a rect of the primary read at a
    finer pyramid level, and an overlay that was only ever composed for the whole field would
    have to be dropped there — which is exactly how a zoom made the overlay disappear.

    ``read_tile(sec_m, want)`` returns that secondary multipoint's pixels at the already-
    resolved t/z/c, or ``None`` if it cannot be read. ``want`` is the fractional window of
    the tile's pixel grid the composite actually needs (:func:`source_window`), and the
    reader may honour it or ignore it:

    * a 2-D array is taken to be **the whole tile**, whatever ``want`` asked for — what every
      in-memory reader does, and the cheapest correct answer;
    * ``(plane, (fy0, fy1, fx0, fx1))`` says "these pixels are that fractional part of the
      tile", so a reader backed by a pyramid can serve the window at full resolution instead
      of handing back a decimated whole plane. The window it returns need not be the one
      asked for — it will have been snapped out to whole pixels of whichever level it chose.

    ``None`` when the field cannot be placed or no tile covers it, which the caller shows as
    "no overlay here" rather than as a black plane.
    """
    tiles = dict((int(m), v) for m, v in (entry.get("tiles") or ()))
    hits = tiles.get(int(pri_m)) or ()
    if not hits:
        return None

    # An INDEX placement (the user overrode a refusal) has no field boxes to map through —
    # that is exactly what it could not compute. The honest rendering is then a plain
    # resize of the whole source plane onto the whole output: it makes no claim about where
    # anything is, which is the truthful statement in that state. Unit extents give that
    # for free through the same axis_map, so there is no second sampling path to keep in
    # step with this one.
    by_index = str(entry.get("placed_by", "stage")) == "index"
    pri_box = (FieldBox(0.0, 1.0, 0.0, 1.0) if by_index
               else field_box(pri_md, pri_axes, pri_m))
    if pri_box is None:
        return None
    # The µm box the OUTPUT covers — the whole field, or the zoomed rect of it.
    out_box = pri_box if region is None else sub_field_box(pri_box, region)

    dz, dy, dx = (float(v) for v in (entry.get("offset_um") or (0.0, 0.0, 0.0)))
    flip_x = bool(entry.get("flip_x", True))
    flip_y = bool(entry.get("flip_y", False))
    h, w = int(out_shape[0]), int(out_shape[1])
    out = np.full((h, w), float(fill), dtype=np.float32)
    painted = False

    # Smallest contributor first, so the largest tile wins any shared boundary pixel — with
    # tiles that abut to a fraction of a micron, the alternative is a one-pixel seam drawn
    # from whichever tile happened to be painted last.
    for sec_m, _frac in sorted(hits, key=lambda p: p[1]):
        sec_box = (FieldBox(0.0, 1.0, 0.0, 1.0) if by_index
                   else field_box(sec_md, sec_axes, int(sec_m)))
        if sec_box is None:
            continue
        # The nudge moves the SECONDARY, so it is added to the tile's box once, here, and
        # every µm comparison below is against the nudged box.
        s_y0, s_y1 = sec_box.y0 + dy, sec_box.y1 + dy
        s_x0, s_x1 = sec_box.x0 + dx, sec_box.x1 + dx
        want = source_window(out_box, FieldBox(s_y0, s_y1, s_x0, s_x1),
                             flip_y=flip_y, flip_x=flip_x)
        if want is None:
            continue                  # this tile does not reach the output — no read at all
        got = read_tile(int(sec_m), want)
        if got is None:
            continue
        plane, cover = (got, None) if isinstance(got, np.ndarray) else got
        plane = np.asarray(plane)
        if plane.ndim != 2 or plane.size == 0:
            continue
        n_sy, n_sx = plane.shape
        if cover is None:                      # whole tile: its extent IS the tile's box
            r_lo, r_hi, c_lo, c_hi = s_y0, s_y1, s_x0, s_x1
        else:
            r_lo, r_hi = _window_extent(s_y0, s_y1, (cover[0], cover[1]), flip_y)
            c_lo, c_hi = _window_extent(s_x0, s_x1, (cover[2], cover[3]), flip_x)
        rows = axis_map(h, out_box.y0, out_box.y1, n_sy, r_lo, r_hi, flip=flip_y)
        cols = axis_map(w, out_box.x0, out_box.x1, n_sx, c_lo, c_hi, flip=flip_x)
        r_ok = np.flatnonzero(rows >= 0)
        c_ok = np.flatnonzero(cols >= 0)
        if r_ok.size == 0 or c_ok.size == 0:
            continue
        # MINIFICATION: this tile's pixels outnumber the output samples they land in, so
        # point-sampling would keep one in N and discard the rest (see `_area_shrink`). Average
        # the blocks down to the output's own sampling first, then the map is ~1:1.
        if (n_sy > _SHRINK_AT * r_ok.size) or (n_sx > _SHRINK_AT * c_ok.size):
            plane = _area_shrink(plane, r_ok.size, c_ok.size)
            n_sy, n_sx = plane.shape
            rows = axis_map(h, out_box.y0, out_box.y1, n_sy, r_lo, r_hi, flip=flip_y)
            cols = axis_map(w, out_box.x0, out_box.x1, n_sx, c_lo, c_hi, flip=flip_x)
            r_ok = np.flatnonzero(rows >= 0)
            c_ok = np.flatnonzero(cols >= 0)
            if r_ok.size == 0 or c_ok.size == 0:
                continue
        patch = plane[np.ix_(rows[r_ok], cols[c_ok])]
        out[np.ix_(r_ok, c_ok)] = patch.astype(np.float32, copy=False)
        painted = True

    return out if painted else None


def context_extent(dst_md: Mapping[str, Any], dst_axes: Any, m: int,
                   src_md: Mapping[str, Any], src_axes: Any,
                   hits: Sequence[Tuple[int, float]],
                   *, offset_um: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                   ) -> Optional[Tuple[FieldBox, FieldBox]]:
    """``(union, primary)`` boxes for the CONTEXT canvas, or ``None``.

    The primary's field is 294 µm and the GFP tile it sits in is 1760 µm — **36x the
    area** — so cropping the secondary to the primary throws away precisely the context
    that made it worth overlaying. This returns the union of the contributing tiles with
    the primary's own box inside it, which is what the Viewer needs both to widen the view
    and to draw the "you are here" frame on it.

    ``None`` when either side cannot be placed: a context view of an unplaced field would
    be a rectangle drawn around a guess.
    """
    pri = field_box(dst_md, dst_axes, m)
    if pri is None or not hits:
        return None
    dz, dy, dx = (float(v) for v in offset_um)
    boxes = []
    for j, _frac in hits:
        sb = field_box(src_md, src_axes, int(j))
        if sb is not None:
            boxes.append(translate(sb, dz, dy, dx) if (dz or dy or dx) else sb)
    if not boxes:
        return None
    union = FieldBox(
        y0=min([pri.y0] + [b.y0 for b in boxes]), y1=max([pri.y1] + [b.y1 for b in boxes]),
        x0=min([pri.x0] + [b.x0 for b in boxes]), x1=max([pri.x1] + [b.x1 for b in boxes]),
        z0=pri.z0, z1=pri.z1)
    return (union, pri)


def translate(box: FieldBox, dz: float = 0.0, dy: float = 0.0,
              dx: float = 0.0) -> FieldBox:
    """``box`` moved by a µm offset — the manual nudge, applied to the SECONDARY.

    The nudge has to move the box *before* tile selection, not after: shifting a 294 µm
    field by 200 µm can bring an entirely different montage tile into view, and an offset
    applied only at draw time would sample the tiles chosen for the un-nudged position.
    """
    return FieldBox(
        y0=box.y0 + dy, y1=box.y1 + dy, x0=box.x0 + dx, x1=box.x1 + dx,
        z0=None if box.z0 is None else box.z0 + dz,
        z1=None if box.z1 is None else box.z1 + dz)


def tiles_covering(dst_md: Mapping[str, Any], dst_axes: Any, m: int,
                   src_md: Mapping[str, Any], src_axes: Any,
                   *, min_fraction: float = 1e-4,
                   offset_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   ) -> List[Tuple[int, float]]:
    """Which multipoints of ``src`` cover ``dst``'s field ``m`` → ``[(m_src, fraction)]``,
    largest contributor first.

    This is the function that makes cross-file overlay real, and the reason the secondary
    input cannot be reduced to "the matching position". On the WellA3 pair the primary's
    294 µm field lands inside the GFP montage's 1760 µm tiles without respecting their
    seams: ``m=2`` and ``m=3`` sit wholly inside one tile, six positions straddle two, and
    ``m=10`` draws from **four** (85 / 8 / 6 / 1 %). A design that assumed one source tile
    would silently show three-quarters of a field and a black corner.

    ``min_fraction`` drops slivers — a tile contributing 0.001% of the field costs a full
    read to change nothing.
    """
    dst_box = field_box(dst_md, dst_axes, m)
    if dst_box is None:
        return []
    hits: List[Tuple[int, float]] = []
    n_src = int(getattr(src_axes, "m", 1) or 1)
    dz, dy, dx = (float(v) for v in offset_um)
    for j in range(n_src):
        src_box = field_box(src_md, src_axes, j)
        if src_box is None:
            continue
        if dz or dy or dx:
            src_box = translate(src_box, dz, dy, dx)
        frac = overlap_fraction(dst_box, src_box)
        if frac >= min_fraction:
            hits.append((j, frac))
    hits.sort(key=lambda h: -h[1])
    return hits


def pair_timepoints(dst_md: Mapping[str, Any], n_dst_t: int,
                    src_md: Mapping[str, Any], n_src_t: int,
                    *, shift: int = 0) -> List[Tuple[int, Optional[int], Optional[float]]]:
    """Pair each destination timepoint with a source one → ``[(t_dst, t_src, error_s)]``.

    Pairing is by **index** plus an integer ``shift``, which is the decision this node
    locked: index pairing is stable across a re-ingest, survives in a saved graph, and on
    the WellA3 pair it is what the microscope actually did (each cycle ran the 640 stacks
    then the GFP sweep). ``t_src`` is ``None`` where the shift walks off the end.

    ``error_s`` is the honest part: the **absolute** time between the paired frames,
    computed from ``frame_time_jd`` — the shared wall clock — so a wrong pairing is
    reported rather than inferred. It is ``None`` when either file lacks that clock.

    Why not pair BY that clock: on this pair it would choose ``GFP[k] ↔ 640[k+2]``, because
    the two loops run at 1421.9 s and 1417.8 s and drift apart. Nearest-in-time is a
    defensible policy and a terrible default — it changes which frames you are comparing
    based on rounding, and it disagrees with index pairing by two full cycles here. So the
    clock informs the readout and never drives the choice.
    """
    dst_jd = _seq(dst_md, "frame_time_jd")
    src_jd = _seq(src_md, "frame_time_jd")
    out: List[Tuple[int, Optional[int], Optional[float]]] = []
    for t in range(int(n_dst_t)):
        j = t + int(shift)
        if not (0 <= j < int(n_src_t)):
            out.append((t, None, None))
            continue
        err: Optional[float] = None
        if 0 <= t < len(dst_jd) and 0 <= j < len(src_jd):
            try:
                err = (float(src_jd[j]) - float(dst_jd[t])) * SECONDS_PER_DAY
            except (TypeError, ValueError):
                err = None
        out.append((t, j, err))
    return out


@dataclass(frozen=True)
class PlacementPlan:
    """The resolved answer to "where does the secondary go inside the primary's field".

    ``refusals`` is non-empty exactly when the overlay must NOT be drawn: each entry is a
    failure that would otherwise render convincingly and be wrong. ``warnings`` is the
    other tier — states that are visible on screen as soon as you look (a blank corner, no
    overlay at all), so they annotate rather than block.
    """

    #: primary m → [(secondary m, fraction of the primary field it supplies)]
    tiles: Dict[int, List[Tuple[int, float]]]
    #: primary m → total fraction of its field the secondary can supply (0..1)
    coverage: Dict[int, float]
    #: secondary µm/px ÷ primary µm/px — >1 means the secondary is coarser and will be
    #: magnified into the primary's grid (5.985 on the WellA3 pair)
    scale: Optional[float]
    #: [(t_primary, t_secondary | None, absolute seconds between them | None)]
    t_pairs: List[Tuple[int, Optional[int], Optional[float]]]
    #: primary m → the secondary's focus minus the primary's stack centre, in µm
    z_offset_um: Dict[int, Optional[float]]
    refusals: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    #: ``"stage"`` (the real thing) or ``"index"`` (the user overrode a refusal). Carried
    #: rather than inferred so every readout can say WHICH it is — a picture placed by
    #: index looks exactly like one placed by stage, and that is the whole hazard.
    placed_by: str = "stage"

    @property
    def ok(self) -> bool:
        return not self.refusals

    def describe(self, m: int = 0) -> str:
        """The one-line readout for the node card — what actually got matched, so a bad
        placement is something you SEE rather than something you audit."""
        tiles = self.tiles.get(m, [])
        if not tiles:
            return f"m{m}: no secondary tile covers this field"
        names = ", ".join(f"m{j:02d}" for j, _ in tiles)
        parts = [f"m{m} <- {names}"]
        if self.placed_by != "stage":
            parts.append("BY INDEX (override)")
        parts.append(f"{100.0 * self.coverage.get(m, 0.0):.0f}% covered")
        if self.scale:
            parts.append(f"{self.scale:.3g}x px")
        err = next((e for _, _, e in self.t_pairs if e is not None), None)
        if err is not None:
            parts.append(f"{err:+.0f}s")
        dz = self.z_offset_um.get(m)
        if dz is not None:
            parts.append(f"{dz:+.1f}um z")
        return " · ".join(parts)


def plan_placement(dst_md: Mapping[str, Any], dst_axes: Any,
                   src_md: Mapping[str, Any], src_axes: Any,
                   *, t_shift: int = 0,
                   dst_sampling: Sequence[str] = (),
                   src_sampling: Sequence[str] = (),
                   offset_um: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   min_coverage: float = 0.0,
                   on_unplaceable: str = "refuse") -> PlacementPlan:
    """Resolve the full placement of ``src`` into ``dst``'s field, with the tiered
    refuse/degrade verdict attached.

    ``dst_sampling`` / ``src_sampling`` are the two Datasets' ``_sampling_of`` provenance
    tuples. They are compared because **size equality is not geometry equality**: a
    drift-corrected Dataset is shape-preserving and origin-moving, and the guard that
    only compared ``AxisSizes`` is exactly the one that let a 40 px shift through and
    reported ``mean_intensity`` 242.3 against a true 879.8. A stage log is a description of
    where the *camera* was, so it stops describing the *data* the moment a node resamples
    it — and nothing in the pixel values reveals that.
    """
    refusals: List[str] = []
    warnings: List[str] = []

    # ── DEGRADE: the two chains diverged, but both can still be PLACED ───────────
    #
    # This used to be a refusal, and it was wrong — wrong in a way that defeated the whole
    # point of `origin_um`. The sampling-provenance test is borrowed from
    # `_intensity_provider`, whose contract is voxel-for-voxel reading: there, two datasets
    # with different sampling genuinely do not address the same voxels. An overlay does not
    # read voxel-for-voxel. It places by PHYSICAL µm, and maintaining a true µm origin
    # ACROSS crop / resample / z-project / stitch is exactly what `origin_um` was added to
    # do. Refusing whenever the two chains differed meant "overlay after stitch" and
    # "overlay after a Z projection" both refused, having correctly computed the placement.
    #
    # The dangerous case is not divergence — it is placement resting on a stage log that has
    # gone stale, and that is caught precisely by the per-input test below (no maintained
    # origin AND a non-empty sampling history). A node that cannot keep its origin true,
    # like drift correction, DROPS the key, which lands it there.
    #
    # It is still worth saying out loud, because the two pictures came from different
    # geometry and a reader deserves to know which ops each side went through.
    if tuple(dst_sampling) != tuple(src_sampling):
        only_dst = [x for x in dst_sampling if x not in src_sampling]
        only_src = [x for x in src_sampling if x not in dst_sampling]
        warnings.append(
            "the two inputs went through different geometry"
            + (f" — primary also: {only_dst}" if only_dst else "")
            + (f" — secondary also: {only_src}" if only_src else "")
            + "; they are placed by their maintained µm origins, not voxel-for-voxel")

    # ── REFUSE: nothing trustworthy to place with ─────────────────────────────────
    #
    # Two admissible sources of a corner, and the fallback is the delicate one. A Dataset
    # is placeable if it carries a maintained `origin_um` covering every multipoint, OR if
    # it still carries the raw stage log AND has not been resampled since the source. The
    # second clause is not belt-and-braces: `align.drift` DROPS `origin_um` because the
    # content moved under a fixed grid, but `stage_xy_um` is non-calibration provenance and
    # rides the payload untouched — so without the sampling test the placement would
    # quietly fall back to a stage log that has stopped describing these pixels, which is
    # the whole failure class `origin_um` was added to close.
    for label, md, axes, sampling in (("primary", dst_md, dst_axes, tuple(dst_sampling)),
                                      ("secondary", src_md, src_axes, tuple(src_sampling))):
        n_m = int(getattr(axes, "m", 1) or 1)
        origins = md.get("origin_um")
        has_origin = (isinstance(origins, (list, tuple)) and len(origins) >= n_m)
        xy = _seq(md, "stage_xy_um")
        if not has_origin:
            if not xy:
                refusals.append(
                    f"{label} carries no spatial origin and no stage position log "
                    f"(stage_layout_source={md.get('stage_layout_source', 'missing')!r}), "
                    f"so it cannot be placed on the microscope; a TIFF never carries one")
            elif len(xy) < n_m:
                refusals.append(
                    f"{label}'s stage log covers {len(xy)} of {n_m} multipoints; a partial "
                    f"layout places the uncovered fields at some other position's "
                    f"coordinates")
            elif sampling:
                refusals.append(
                    f"{label} has no maintained origin_um and its geometry has changed "
                    f"since the source ({list(sampling)}), so its stage log no longer "
                    f"describes these pixels; place the overlay before that node, or use "
                    f"an input whose origin survived")
        if md.get("pixel_size_um") in (None, 0):
            refusals.append(f"{label} has no pixel_size_um, so its field has no physical size")

    if refusals and on_unplaceable == "refuse":
        return PlacementPlan(tiles={}, coverage={}, scale=None, t_pairs=[],
                             z_offset_um={}, refusals=tuple(refusals))

    # ── the OVERRIDE: the user has taken the placement decision back ──────────────
    #
    # `align_by_index` says "I know these two line up (or I will nudge them myself) —
    # stop refusing and pair the fields by index". It exists because the refusals above
    # are about what the FILES can prove, not about what is true: a TIFF carries no stage
    # log at all, a cropped source has thrown its origin away, and in both cases a person
    # who knows the acquisition can still place them by hand with the µm nudge.
    #
    # Three rules keep it from becoming the silent-wrongness hatch the refusals exist to
    # close. It is never the default. Every refusal it walks past is re-emitted as a
    # warning that says OVERRIDDEN in words, so the reason survives into the node card and
    # the status line rather than being swallowed. And the fallback is INDEX pairing, which
    # is honest about what it is doing — it does not fabricate a stage position and then
    # present the result as physically placed.
    if refusals:
        n_src_m = int(getattr(src_axes, "m", 1) or 1)
        warnings.extend(f"OVERRIDDEN — {r}" for r in refusals)
        warnings.append(
            "placement is by INDEX, not by stage position: field m is paired with the "
            "secondary's field m (clamped to its last). Physical alignment is NOT "
            "verified — check it by eye and correct it with the µm nudge")
        n_dst_m = int(getattr(dst_axes, "m", 1) or 1)
        idx_tiles = {m: [(min(m, n_src_m - 1), 1.0)] for m in range(n_dst_m)}
        try:
            idx_scale = float(src_md["pixel_size_um"]) / float(dst_md["pixel_size_um"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            idx_scale = None
        return PlacementPlan(
            tiles=idx_tiles, coverage={m: 1.0 for m in idx_tiles}, scale=idx_scale,
            t_pairs=pair_timepoints(dst_md, int(getattr(dst_axes, "t", 1) or 1),
                                    src_md, int(getattr(src_axes, "t", 1) or 1),
                                    shift=t_shift),
            z_offset_um={m: None for m in idx_tiles},
            refusals=(), warnings=tuple(warnings), placed_by="index")

    # ── resolve ──────────────────────────────────────────────────────────────────
    dst_ps = float(dst_md["pixel_size_um"])
    src_ps = float(src_md["pixel_size_um"])
    scale = src_ps / dst_ps if dst_ps > 0 else None

    tiles: Dict[int, List[Tuple[int, float]]] = {}
    coverage: Dict[int, float] = {}
    z_offset: Dict[int, Optional[float]] = {}
    n_dst_m = int(getattr(dst_axes, "m", 1) or 1)
    for m in range(n_dst_m):
        hits = tiles_covering(dst_md, dst_axes, m, src_md, src_axes,
                              offset_um=offset_um)
        tiles[m] = hits
        # Tiles of one montage do not overlap each other, so summing their contributions is
        # the field's coverage; clamped because a file with genuinely overlapping tiles
        # (a stitch scan with margin) would otherwise report >100%.
        coverage[m] = min(1.0, sum(f for _, f in hits))
        z_offset[m] = _z_offset(dst_md, dst_axes, m, src_md, src_axes, hits,
                                dz=float(offset_um[0]))

    t_pairs = pair_timepoints(dst_md, int(getattr(dst_axes, "t", 1) or 1),
                              src_md, int(getattr(src_axes, "t", 1) or 1),
                              shift=t_shift)

    # ── DEGRADE: states you can see on screen ────────────────────────────────────
    blank = [m for m, c in coverage.items() if c <= 0.0]
    if blank:
        warnings.append(
            f"no secondary tile covers primary field(s) {_brief(blank)} — nothing will be "
            f"drawn there")
    partial = [m for m, c in coverage.items() if 0.0 < c < 1.0 - _COVERAGE_TOL]
    if partial:
        worst = min(coverage[m] for m in partial)
        warnings.append(
            f"primary field(s) {_brief(partial)} are only partly covered by the secondary "
            f"(down to {100.0 * worst:.1f}%) — the remainder stays blank")
    if min_coverage > 0.0:
        thin = [m for m, c in coverage.items() if c < min_coverage]
        if thin:
            warnings.append(
                f"field(s) {_brief(thin)} fall below the {100.0 * min_coverage:.0f}% "
                f"coverage floor")
    unpaired = [t for t, j, _ in t_pairs if j is None]
    if unpaired:
        warnings.append(
            f"t_shift={t_shift} leaves timepoint(s) {_brief(unpaired)} with no secondary "
            f"frame — the overlay is empty there")
    if dst_md.get("z_collapsed"):
        warnings.append(
            "the primary's Z is collapsed (a projection), so no depth relationship is "
            "reported — X/Y placement is unaffected and the overlay is drawn on the "
            "single remaining plane")
    errs = [abs(e) for _, _, e in t_pairs if e is not None]
    if not errs:
        warnings.append(
            "no shared absolute clock (frame_time_jd absent on at least one input), so the "
            "time distance between paired frames cannot be reported")
    if scale is not None and not (1.0 / (1.0 + _EXTENT_TOL) <= scale <= 1.0 + _EXTENT_TOL):
        warnings.append(
            f"the secondary is sampled {scale:.3g}x the primary's pixel size and will be "
            f"{'magnified' if scale > 1 else 'minified'} into its grid")

    return PlacementPlan(tiles=tiles, coverage=coverage, scale=scale, t_pairs=t_pairs,
                         z_offset_um=z_offset, refusals=(), warnings=tuple(warnings))


def _z_offset(dst_md: Mapping[str, Any], dst_axes: Any, m: int,
              src_md: Mapping[str, Any], src_axes: Any,
              hits: Sequence[Tuple[int, float]], *, dz: float = 0.0) -> Optional[float]:
    """The secondary's focus minus the centre of the primary's stack at field ``m``, µm.

    Measured against the stack CENTRE rather than slice 0 because that is the number a user
    can act on — "the GFP plane sits 1.6 µm below the middle of your 60 µm stack" locates
    it; "it is 27.8 µm above slice 0" does not, unless you also remember the stack depth.
    Uses the largest-contributing tile: the tiles of a montage share a focus only
    approximately (the WellA3 plate tilts ~4 µm per tile), and averaging them would invent
    a focus no tile actually had.
    """
    if not hits:
        return None
    # A Z-PROJECTED primary has no depth left to be offset from. Its origin still carries
    # where the slab STARTED, so `field_box` happily returns z0 == z1 there and a
    # subtraction would produce a confident number describing a plane the data is not on —
    # a max projection is every plane at once. Lateral placement is unaffected and stays
    # exact, which is the whole point: collapsing Z costs you Z, not X and Y.
    if dst_md.get("z_collapsed"):
        return None
    dst_box = field_box(dst_md, dst_axes, m)
    if dst_box is None or dst_box.z0 is None or dst_box.z1 is None:
        return None
    src_z = z_um_of_slice(src_md, src_axes, hits[0][0], 0)
    if src_z is None:
        return None
    return (src_z + dz) - 0.5 * (dst_box.z0 + dst_box.z1)


def _brief(items: Sequence[int], limit: int = 6) -> str:
    """``[0, 1, 2, …]`` clipped — a 49-position file must not put 49 numbers in a warning."""
    vals = list(items)
    if len(vals) <= limit:
        return ", ".join(str(v) for v in vals)
    head = ", ".join(str(v) for v in vals[:limit])
    return f"{head}, … (+{len(vals) - limit} more)"
