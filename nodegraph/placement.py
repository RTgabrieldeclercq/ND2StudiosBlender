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
    "paired_t_frac", "map_t", "secondary_z_weights", "plan_time", "frame_interval_s",
    "snap_rate", "parse_pins", "pins_json", "nudge_delta_um", "RATE_SNAP_TOL",
    "SUB_TICK_CAP",
    "PositionGroup", "GroupPlan", "position_groups", "group_key", "GROUP_GAP_FACTOR",
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


#: Nudges a mapped frame position up past float noise before it is floored, so a map that
#: lands EXACTLY on a frame (every index pairing, every integer rate) cannot round down to the
#: frame before it.
_FRAME_EPS = 1e-6


def paired_t_frac(entry: Dict[str, Any], t: int, k: int = 0, n: int = 1) -> Optional[int]:
    """The secondary timepoint on SUB-TICK ``k`` of ``n`` inside primary frame ``t``.

    "Play all" holds each primary frame for ``n`` ticks so a faster source can show every one
    of its own frames: a source recorded at 4x the primary's rate advances one frame per tick,
    while a source at the primary's rate stays on its paired frame for all four. So the map is
    sampled at the fractional primary time ``t + k/n``.

    Sub-tick 0 (and any entry with no ``t_map``, i.e. every recipe stamped before rates
    existed, and every synthesized ``view_source`` entry) is exactly :func:`paired_t`, so the
    picture at rest never disagrees with the node card's pairing.
    """
    tm = entry.get("t_map")
    if not tm or int(n) <= 1 or int(k) <= 0:
        return paired_t(entry, t)
    n_src = int(tm.get("n_src", 0) or 0)
    if n_src <= 1:
        return paired_t(entry, t)                      # a still is HELD, sub-tick or not
    j = int(np.floor(map_t(tm, float(t) + float(k) / float(n)) + _FRAME_EPS))
    return j if 0 <= j < n_src else None


def _pw_linear(knots: Sequence[Tuple[float, float]], x: float, slope: float) -> float:
    """Evaluate the piecewise-linear map through ``knots`` at ``x``.

    The ONE interpolation rule every pin uses, in T and in Z: straight lines between
    consecutive knots, and a line of gradient ``slope`` beyond the outermost ones. So one pin
    is a pure offset at the natural rate, two pins fix the rate between them, and a third
    lets the rate change part-way (a stage that stalled, a dropped cycle).
    """
    if not knots:
        return float(x)
    x = float(x)
    u0, s0 = knots[0]
    if x <= u0 or len(knots) == 1:
        return float(s0) + float(slope) * (x - float(u0))
    u_last, s_last = knots[-1]
    if x >= u_last:
        return float(s_last) + float(slope) * (x - float(u_last))
    for (ua, sa), (ub, sb) in zip(knots, knots[1:]):
        if ua <= x <= ub:
            if ub == ua:
                return float(sb)
            return float(sa) + (float(sb) - float(sa)) * (x - float(ua)) / (float(ub) - float(ua))
    return float(s_last)


def map_t(t_map: Mapping[str, Any], u: float) -> float:
    """The secondary's (fractional) frame position at primary time ``u`` under ``t_map``."""
    knots = [(float(a), float(b)) for a, b in (t_map.get("knots") or ((0.0, 0.0),))]
    return _pw_linear(knots, u, float(t_map.get("rate", 1.0) or 1.0))


def secondary_z_weights(sec_md: Mapping[str, Any], sec_axes: Any, sec_m: int,
                        z_um: Optional[float], *, dz: float = 0.0, linear: bool = False,
                        z_pins: Sequence[Sequence[Any]] = (),
                        pri_k: Optional[int] = None,
                        pri_step_um: Optional[float] = None) -> List[Tuple[int, float]]:
    """Which secondary slice(s) to draw for a primary plane at ``z_um`` → ``[(k, weight)]``.

    ``linear=False`` (the ``nearest`` Z sampling) returns ONE slice at weight 1 — the slice
    nearest in absolute µm, clamped into range — and is bit-identical to what the overlay has
    always drawn (:func:`secondary_z_index` is now this function's first answer). A channel
    shown that way is always a plane its microscope actually acquired.

    ``linear=True`` blends the two bracketing slices by distance, so two stacks taken at
    different Z steps can be scrolled together without the coarser one jumping a whole step
    at a time. Off either end it clamps to the end plane (weight 1) rather than extrapolate
    a plane that was never imaged.

    ``z_pins`` are the user's "this plane goes with that plane" rows,
    ``(k_pri, k_sec, um_pri | None, um_sec | None)``. When every pin carries its µm and both
    files have a focus log, the pins define a µm → µm map (one pin = an offset, two = the
    axial scale a refractive-index mismatch introduces) — that form survives an upstream Z
    crop, since absolute focus does not re-index. Otherwise they map slice index to slice
    index, with the files' own step ratio (or 1) beyond the outermost pin. Pins REPLACE the
    Nudge Z: they are the more specific statement.

    A single-plane secondary has one answer and takes it, pinned or not — the WellA3 case,
    and the reason a 2D context view stays visible while you scroll a 210-slice stack rather
    than appearing on one slice and vanishing.
    """
    nz = int(getattr(sec_axes, "z", 1) or 1)
    if nz <= 1:
        return [(0, 1.0)]
    frac: Optional[float] = None
    if z_pins:
        frac = _pinned_z_frac(sec_md, sec_axes, sec_m, z_um, z_pins, pri_k, pri_step_um)
    if frac is None:
        if z_um is None:
            return [(0, 1.0)]
        z0 = z_um_of_slice(sec_md, sec_axes, sec_m, 0)
        z1 = z_um_of_slice(sec_md, sec_axes, sec_m, nz - 1)
        if z0 is None or z1 is None or z1 == z0:
            return [(0, 1.0)]
        frac = ((z_um - dz) - z0) / ((z1 - z0) / (nz - 1))
    if not linear:
        return [(int(min(max(0, round(frac)), nz - 1)), 1.0)]
    f = min(max(float(frac), 0.0), float(nz - 1))
    k0 = int(np.floor(f))
    w = f - k0
    if k0 >= nz - 1 or w < _FRAME_EPS:
        return [(min(k0, nz - 1), 1.0)]
    if w > 1.0 - _FRAME_EPS:
        return [(k0 + 1, 1.0)]
    return [(k0, 1.0 - w), (k0 + 1, w)]


def _pinned_z_frac(sec_md: Mapping[str, Any], sec_axes: Any, sec_m: int,
                   z_um: Optional[float], z_pins: Sequence[Sequence[Any]],
                   pri_k: Optional[int], pri_step_um: Optional[float]) -> Optional[float]:
    """The secondary's fractional slice under the user's Z pins, or ``None`` (not applicable).

    µm form first (crop-proof), index form as the fallback — see :func:`secondary_z_weights`.
    """
    nz = int(getattr(sec_axes, "z", 1) or 1)
    rows = [tuple(r) + (None,) * (4 - len(r)) for r in z_pins]
    if z_um is not None and all(r[2] is not None and r[3] is not None for r in rows):
        z0 = z_um_of_slice(sec_md, sec_axes, sec_m, 0)
        z1 = z_um_of_slice(sec_md, sec_axes, sec_m, nz - 1)
        if z0 is not None and z1 is not None and z1 != z0:
            target = _pw_linear([(float(r[2]), float(r[3])) for r in rows], float(z_um), 1.0)
            return (target - z0) / ((z1 - z0) / (nz - 1))
    if pri_k is None:
        return None
    s_step = _z_step(sec_md, sec_axes)
    slope = (float(pri_step_um) / s_step) if (pri_step_um and s_step) else 1.0
    return _pw_linear([(float(r[0]), float(r[1])) for r in rows], float(pri_k), slope)


def secondary_z_index(sec_md: Dict[str, Any], sec_axes: Any, sec_m: int,
                      z_um: Optional[float], *, dz: float = 0.0) -> int:
    """Which secondary slice to draw for a primary plane sitting at ``z_um``.

    A single-plane secondary has one answer and takes it — that is the WellA3 case, and it
    is why a 2D context view stays visible while you scroll a 210-slice stack rather than
    appearing on one slice and vanishing. A volumetric secondary picks the slice NEAREST in
    absolute µm, clamped into range, so scrolling the primary's Z walks the secondary's too.

    The ``nearest`` answer of :func:`secondary_z_weights`, kept by this name because
    ``util.merge``'s channel branch and the resample bake's older callers ask for exactly one
    slice.
    """
    return secondary_z_weights(sec_md, sec_axes, sec_m, z_um, dz=dz)[0][0]


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
    placed_by = str(entry.get("placed_by", "stage"))
    by_index = placed_by == "index"
    # CENTRE placement (`unplaceable=align_centres`): both fields at their true µm size,
    # centre on centre — for files the stage cannot relate (another well, no stage log).
    centred = placed_by == "centre"
    if by_index:
        pri_box = FieldBox(0.0, 1.0, 0.0, 1.0)
    elif centred:
        pri_box = _centred_box(pri_md, pri_axes)
    else:
        pri_box = field_box(pri_md, pri_axes, pri_m)
    if pri_box is None:
        return None
    # The µm box the OUTPUT covers — the whole field, or the zoomed rect of it.
    out_box = pri_box if region is None else sub_field_box(pri_box, region)

    dz, dy, dx = (float(v) for v in (entry.get("offset_um") or (0.0, 0.0, 0.0)))
    if by_index:
        # The index layout is in FIELD fractions, and the nudge is in µm. Added raw, a 10 µm
        # nudge moved the secondary ten whole fields. Scaled by the primary's own extent, a
        # µm means a µm here too — and a two-click nudge (`nudge_delta_um`) lands on target.
        ext = lateral_extent_um(pri_md, pri_axes)
        if ext is not None and ext[0] > 0 and ext[1] > 0:
            dy, dx = dy / ext[0], dx / ext[1]
    flip_x = bool(entry.get("flip_x", True))
    flip_y = bool(entry.get("flip_y", False))
    h, w = int(out_shape[0]), int(out_shape[1])
    out = np.full((h, w), float(fill), dtype=np.float32)
    painted = False

    # Smallest contributor first, so the largest tile wins any shared boundary pixel — with
    # tiles that abut to a fraction of a micron, the alternative is a one-pixel seam drawn
    # from whichever tile happened to be painted last.
    for sec_m, _frac in sorted(hits, key=lambda p: p[1]):
        if by_index:
            sec_box = FieldBox(0.0, 1.0, 0.0, 1.0)
        elif centred:
            sec_box = _centred_box(sec_md, sec_axes)
        else:
            sec_box = field_box(sec_md, sec_axes, int(sec_m))
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
                    *, shift: int = 0, t_map: Optional[Mapping[str, Any]] = None
                    ) -> List[Tuple[int, Optional[int], Optional[float]]]:
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

    **A single-frame secondary is HELD, not unpaired** (reported 2026-09-14: "overlay only
    appears on the first frame"). A still — a reference snapshot, a mask, a brightfield
    context shot — has exactly one answer for every primary timepoint, so it takes it, and
    ``shift`` is inapplicable rather than merely unsatisfiable. This is the T twin of the
    rule :func:`secondary_z_index` already applies on the other axis, for the same reason
    and in the same words: a single-plane secondary stays visible while you scroll a
    210-slice stack "rather than appearing on one slice and vanishing". Index pairing
    without this special case sent ``t=1..N-1`` to ``None`` and the overlay vanished after
    frame 0 — a still being magnified into a timelapse is the commonest overlay there is,
    and it was the one shape of input the pairing could not express. ``error_s`` is still
    computed against that one frame, so the readout says how far the primary has travelled
    from the moment the still was taken.

    ``t_map`` (from :func:`plan_time`) replaces ``t + shift`` with the user's pins and/or a
    frame-RATE ratio: secondary frame ``floor(map(t))`` pairs with primary frame ``t``. Still
    an explicit, reported choice — never a nearest-in-time search.
    """
    dst_jd = _seq(dst_md, "frame_time_jd")
    src_jd = _seq(src_md, "frame_time_jd")
    still = int(n_src_t) <= 1
    out: List[Tuple[int, Optional[int], Optional[float]]] = []
    for t in range(int(n_dst_t)):
        if still:
            j = 0
        elif t_map is not None:
            j = int(np.floor(map_t(t_map, float(t)) + _FRAME_EPS))
        else:
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


#: A measured frame-rate ratio within this fraction of a whole number IS that whole number.
#:
#: Two acquisition loops never run at exactly the rate they were programmed for: the WellA3
#: pair cycles at 1421.9 s and 1417.8 s, a ratio of 1.003. Taken literally that is "the
#: secondary runs 0.3% fast", which drifts the pairing a full frame every ~350 frames — the
#: nearest-in-time policy :func:`pair_timepoints` rejects, re-entering through the side door —
#: and would split every primary frame into two sub-ticks for a source that is not faster at
#: all. What the user programmed was "the same rate" (or "4x"), and that is what is snapped
#: to. Set the Rate by hand to state anything else exactly.
RATE_SNAP_TOL = 0.02

#: The most sub-ticks "Play all" will split one primary frame into. A 100x-faster source
#: would otherwise ask the Viewer for 100 composes per primary frame; past this it plays the
#: fast source by skipping frames instead, which the readout shows.
SUB_TICK_CAP = 16


def frame_interval_s(md: Mapping[str, Any]) -> Optional[float]:
    """A file's frame interval in seconds, or ``None`` when it cannot be read.

    ``dt_s`` first (the calibration key the ingest derives from the frame timestamps), then
    the median step of the per-frame ``frame_time_jd`` clock — median, because one dropped or
    delayed frame must not change what the whole series' rate is taken to be.
    """
    try:
        dt = float(md.get("dt_s"))
        if dt > 0 and np.isfinite(dt):
            return dt
    except (TypeError, ValueError):
        pass
    jd = _seq(md, "frame_time_jd")
    if len(jd) >= 2:
        try:
            steps = np.diff(np.asarray([float(v) for v in jd], dtype=float)) * SECONDS_PER_DAY
        except (TypeError, ValueError):
            return None
        steps = steps[np.isfinite(steps) & (steps > 0)]
        if steps.size:
            return float(np.median(steps))
    return None


def snap_rate(r: float) -> float:
    """``r`` snapped to a whole ratio (``n`` or ``1/n``) when within :data:`RATE_SNAP_TOL`."""
    r = float(r)
    if not (r > 0):
        return r
    if r >= 1.0:
        n = round(r)
        return float(n) if n >= 1 and abs(r - n) <= RATE_SNAP_TOL * n else r
    inv = 1.0 / r
    n = round(inv)
    return 1.0 / n if n >= 1 and abs(inv - n) <= RATE_SNAP_TOL * n else r


def parse_pins(raw: Any, *, axis: str) -> Tuple[Tuple[Any, ...], ...]:
    """A ``t_pins`` / ``z_pins`` value → canonical rows ``(pri, sec, anchor_pri, anchor_sec)``.

    The value is JSON in a STRING socket (the ``roi_mask.shapes`` precedent: structured data
    with no SocketType of its own), a list of ``[primary, secondary]`` frame-index pairs,
    optionally ``[primary, secondary, anchor_primary, anchor_secondary]`` where the anchors
    are the absolute clock (T, Julian days) or focus (Z, µm) the Viewer recorded when the pin
    was made. ``""`` is no pins.

    Canonical = sorted by the primary index, ONE pin per primary index (the later row wins —
    re-pinning a frame replaces its pin), anchors as floats or ``None``. The canonical form is
    what makes the recipe hash repeat-stable (INV-12): two spellings of the same pins must not
    be two memo keys.

    Malformed input is REFUSED with the parse error rather than read as "no pins": silently
    dropping a user's pairing is the one outcome worse than an error.
    """
    if raw is None:
        return ()
    rows: Any = raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ()
        import json as _json
        try:
            rows = _json.loads(s)
        except ValueError as exc:
            raise ValueError(
                f"`{axis}_pins` is not valid JSON ({exc}). Expected a list of "
                f"[primary {axis}, secondary {axis}] pairs, e.g. [[0, 0], [12, 48]].") from None
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f"`{axis}_pins` must be a JSON list of [primary, secondary] pairs, "
                         f"not {type(rows).__name__}")
    by_pri: Dict[int, Tuple[Any, ...]] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) not in (2, 4):
            raise ValueError(
                f"`{axis}_pins` row {i} is {row!r}: each pin is [primary, secondary] or "
                f"[primary, secondary, anchor_primary, anchor_secondary]")
        try:
            a, b = int(row[0]), int(row[1])
            if a != float(row[0]) or b != float(row[1]):
                raise ValueError
            anchors = tuple(None if v is None else float(v) for v in row[2:4]) \
                if len(row) == 4 else (None, None)
        except (TypeError, ValueError):
            raise ValueError(
                f"`{axis}_pins` row {i} is {row!r}: frame indices must be whole numbers and "
                f"anchors numbers or null") from None
        if a < 0 or b < 0:
            raise ValueError(f"`{axis}_pins` row {i} is {row!r}: frame indices start at 0")
        by_pri[a] = (a, b) + anchors
    return tuple(by_pri[k] for k in sorted(by_pri))


def pins_json(rows: Sequence[Sequence[Any]]) -> str:
    """Canonical rows → the canonical JSON string (``""`` for none). Anchors ride only when
    present, so a hand-typed ``[[3, 7]]`` round-trips as itself."""
    if not rows:
        return ""
    import json as _json
    out = []
    for r in parse_pins(list(list(x) for x in rows), axis="pin"):
        out.append(list(r[:2]) if r[2] is None and r[3] is None else list(r))
    return _json.dumps(out, separators=(",", ":"))


def _nearest_frame(clock: Sequence[Any], anchor: float, fallback: int) -> Tuple[int, bool]:
    """``(index, found)``: the frame of ``clock`` at ``anchor``, or ``fallback`` if no frame is
    within half an interval of it (the pinned frame was cropped away, or it is another file)."""
    try:
        vals = np.asarray([float(v) for v in clock], dtype=float)
    except (TypeError, ValueError):
        return int(fallback), False
    if vals.size == 0:
        return int(fallback), False
    d = np.abs(vals - float(anchor))
    i = int(np.argmin(d))
    if vals.size == 1:
        return i, bool(d[i] < 1e-9)
    steps = np.diff(np.sort(vals))
    steps = steps[steps > 0]
    half = 0.5 * float(np.median(steps)) if steps.size else 0.0
    return (i, True) if d[i] <= half else (int(fallback), False)


def plan_time(dst_md: Mapping[str, Any], n_dst_t: int,
              src_md: Mapping[str, Any], n_src_t: int,
              *, shift: int = 0, pairing: str = "index", rate: float = 0.0,
              pins: Sequence[Sequence[Any]] = ()) -> Dict[str, Any]:
    """Resolve the T pairing → ``{t_map, t_pairs, sub_ticks, rate, warnings, refusals}``.

    ``pairing="index"`` with no pins is exactly the historical ``t + shift`` pairing and
    returns ``t_map=None``, so a recipe that uses none of this is byte-identical to one
    stamped before it existed.

    ``pairing="rate"``: the secondary advances ``r = dt_primary / dt_secondary`` frames per
    primary frame — 4 for a 5 s series against a 20 s one, 0.25 the other way round, which
    then HOLDS each secondary frame for four primary ones. ``rate > 0`` states ``r`` exactly;
    ``0`` measures it from the two files' clocks (:func:`frame_interval_s`) and snaps it to a
    whole ratio (:data:`RATE_SNAP_TOL`). No clock on either file refuses: a rate this node
    guessed would be a pairing it invented.

    ``pins`` (canonical, :func:`parse_pins`) are "primary frame ``a`` goes with secondary
    frame ``b``": the map runs straight through them, at the rate beyond the outermost. One
    pin re-anchors, two or more also fix the rate between them. A pin that recorded both
    files' clocks is re-found BY CLOCK, so an upstream T crop — which re-indexes — does not
    silently move it onto different frames. Pins that run backwards are refused: a map that
    goes back in time pairs one secondary frame with two unrelated moments.

    ``sub_ticks`` is how many ticks "Play all" should split each primary frame into for this
    source to show every one of its frames — ``ceil`` of the steepest part of the map, capped
    at :data:`SUB_TICK_CAP`.
    """
    warnings: List[str] = []
    refusals: List[str] = []
    n_dst_t, n_src_t = int(n_dst_t), int(n_src_t)
    still = n_src_t <= 1
    if still or (pairing != "rate" and not pins):
        if still and (pins or pairing == "rate"):
            warnings.append("the secondary is a single frame, so T pins and the rate are "
                            "inert — that frame is held across every primary timepoint")
        return {"t_map": None, "sub_ticks": 1, "rate": 1.0, "warnings": warnings,
                "refusals": refusals,
                "t_pairs": pair_timepoints(dst_md, n_dst_t, src_md, n_src_t, shift=shift)}

    r = 1.0
    if pairing == "rate":
        if rate and float(rate) > 0:
            r = float(rate)
        else:
            dp, ds = frame_interval_s(dst_md), frame_interval_s(src_md)
            if dp is None or ds is None:
                missing = " and ".join(
                    n for n, v in (("primary", dp), ("secondary", ds)) if v is None)
                refusals.append(
                    f"T pairing = rate needs each file's frame interval, and the {missing} "
                    f"carries neither dt_s nor a frame clock — set Rate to the number of "
                    f"secondary frames per primary frame (e.g. 4), or pair by index")
                return {"t_map": None, "sub_ticks": 1, "rate": 1.0, "warnings": warnings,
                        "refusals": refusals, "t_pairs": []}
            raw_r = dp / ds
            r = snap_rate(raw_r)
            warnings.append(
                f"rate pairing: the secondary runs {r:g}x the primary's frame rate "
                f"(primary every {dp:g} s, secondary every {ds:g} s"
                + (f", measured {raw_r:.4g}x and snapped" if r != raw_r else "") + ")")

    knots: List[Tuple[float, float]] = []
    if pins:
        dst_jd, src_jd = _seq(dst_md, "frame_time_jd"), _seq(src_md, "frame_time_jd")
        lost: List[int] = []
        for row in pins:
            a, b = int(row[0]), int(row[1])
            ja = row[2] if len(row) > 2 else None
            jb = row[3] if len(row) > 3 else None
            if ja is not None and dst_jd:
                a2, ok = _nearest_frame(dst_jd, float(ja), a)
                a = a2 if ok else a
                if not ok:
                    lost.append(int(row[0]))
            if jb is not None and src_jd:
                b2, ok = _nearest_frame(src_jd, float(jb), b)
                b = b2 if ok else b
            knots.append((float(a), float(b)))
        if lost:
            warnings.append(
                f"T pin(s) at primary frame(s) {_brief(lost)} no longer find their pinned "
                f"moment on the clock (was that frame cropped away?) — used by index")
        knots.sort()
        for (ua, sa), (ub, sb) in zip(knots, knots[1:]):
            if ub == ua or sb < sa:
                refusals.append(
                    f"T pins cross: primary t={ua:g} -> secondary t={sa:g} but primary "
                    f"t={ub:g} -> secondary t={sb:g}, which runs backwards in time. Remove one "
                    f"of them.")
        out_of_range = [int(u) for u, s in knots
                        if not (0 <= u < n_dst_t) or not (0 <= s < n_src_t)]
        if out_of_range:
            warnings.append(f"T pin(s) at primary frame(s) {_brief(out_of_range)} point "
                            f"outside the data they pair — they still steer the map")
        if shift:
            warnings.append(f"t_shift={shift} is inert: T pins decide the pairing")
    else:
        knots = [(0.0, float(shift))]
    if refusals:
        return {"t_map": None, "sub_ticks": 1, "rate": r, "warnings": warnings,
                "refusals": refusals, "t_pairs": []}

    t_map = {"knots": [[u, s] for u, s in knots], "rate": r, "n_src": n_src_t}
    slopes = [abs(r)] + [abs((sb - sa) / (ub - ua))
                         for (ua, sa), (ub, sb) in zip(knots, knots[1:]) if ub > ua]
    steep = max(slopes)
    sub = int(min(SUB_TICK_CAP, max(1, int(np.ceil(steep * (1.0 - RATE_SNAP_TOL))))))
    if steep > SUB_TICK_CAP:
        warnings.append(f"the secondary runs {steep:.3g}x faster than the primary; Play all "
                        f"shows {SUB_TICK_CAP} of its frames per primary frame and skips the "
                        f"rest")
    return {"t_map": t_map, "sub_ticks": sub, "rate": r, "warnings": warnings,
            "refusals": refusals,
            "t_pairs": pair_timepoints(dst_md, n_dst_t, src_md, n_src_t, t_map=t_map)}


def nudge_delta_um(pri_md: Mapping[str, Any], pri_axes: Any,
                   p_pri: Tuple[float, float], p_sec: Tuple[float, float],
                   *, placed_by: str = "stage") -> Optional[Tuple[float, float]]:
    """The ``(dy, dx)`` µm nudge that moves a feature drawn at ``p_sec`` onto ``p_pri``.

    Both points are ``(row, col)`` in the PRIMARY's native image pixels: where a feature sits
    in the primary, and where the same feature is currently drawn in the overlaid secondary.
    The compositor lays the primary's image out along increasing µm on both axes and moves the
    secondary by adding the nudge to its box, so the answer is the pixel delta times the
    primary's pixel size — and it is **flip-independent**: a flip mirrors which source pixel
    a box samples, never where the box sits, so the sign cannot depend on handedness.

    An INDEX placement lays the fields out on the primary's own extent too
    (:func:`compose_secondary_plane` scales the nudge by it), so the same rule holds there.
    ``None`` without a pixel size — a delta that cannot be expressed in µm.
    """
    del placed_by, pri_axes          # the rule is the same for every placement — see above
    try:
        ps = float(pri_md.get("pixel_size_um"))
    except (TypeError, ValueError):
        return None
    if not (ps > 0):
        return None
    return ((float(p_pri[0]) - float(p_sec[0])) * ps,
            (float(p_pri[1]) - float(p_sec[1])) * ps)


def _centred_box(md: Mapping[str, Any], axes: Any) -> Optional[FieldBox]:
    """The field as a box of its TRUE µm size centred on the origin — CENTRE placement.

    For two files whose stage positions do not overlap (another well, another dish, a file
    with no stage log at all) but which the user wants seen together: centre on centre at each
    file's real pixel size, so a 60x field sits at its true size inside a 20x one rather than
    being stretched onto it as an index placement does."""
    ext = lateral_extent_um(md, axes)
    if ext is None:
        return None
    h, w = ext
    return FieldBox(-0.5 * h, 0.5 * h, -0.5 * w, 0.5 * w)


def _held_still_note(dst_axes: Any, src_axes: Any, t_shift: int) -> List[str]:
    """The "single-frame secondary" note, or ``[]`` — one builder, two callers.

    A still is drawn on every primary frame (:func:`pair_timepoints`), which is what the
    user wants and also looks exactly like an overlay that has stopped updating, so it says
    which it is unprompted. The ``t_shift`` half matters more than it looks: a shift
    silently doing nothing is the state where someone reaches for that spinner to fix the
    alignment and concludes the node is broken."""
    n_src_t = int(getattr(src_axes, "t", 1) or 1)
    n_dst_t = int(getattr(dst_axes, "t", 1) or 1)
    if not (n_src_t <= 1 < n_dst_t):
        return []
    return [f"the secondary has a single timepoint, so that one frame is HELD across all "
            f"{n_dst_t} primary timepoints — the overlay is not frozen, there is nothing "
            f"for it to advance to" +
            (f"; t_shift={t_shift} is inapplicable and was ignored" if t_shift else "")]


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
    #: ``"stage"`` (the real thing), ``"index"`` (the user overrode a refusal) or
    #: ``"centre"`` (centre-on-centre at true size, `align_centres`). Carried rather than
    #: inferred so every readout can say WHICH it is — a picture placed by index looks
    #: exactly like one placed by stage, and that is the whole hazard.
    placed_by: str = "stage"
    #: The T map (:func:`plan_time`) when pins or a rate decide the pairing, else ``None``
    #: (plain ``t + shift``) — and how many "Play all" sub-ticks this source wants per
    #: primary frame.
    t_map: Optional[Dict[str, Any]] = None
    sub_ticks: int = 1

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
        if self.placed_by == "centre":
            parts.append("CENTRE-ON-CENTRE (override)")
        elif self.placed_by != "stage":
            parts.append("BY INDEX (override)")
        if self.t_map is not None and float(self.t_map.get("rate", 1.0)) != 1.0:
            parts.append(f"{float(self.t_map['rate']):g}x rate")
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
                   on_unplaceable: str = "refuse",
                   t_pairing: str = "index", rate: float = 0.0,
                   t_pins: Sequence[Sequence[Any]] = (),
                   z_pins: Sequence[Sequence[Any]] = ()) -> PlacementPlan:
    """Resolve the full placement of ``src`` into ``dst``'s field, with the tiered
    refuse/degrade verdict attached.

    ``t_pairing`` / ``rate`` / ``t_pins`` decide the T pairing (:func:`plan_time`). Their
    refusals — no clock for a measured rate, pins that run backwards — are NOT walked past by
    an ``unplaceable`` override: that override is about where the secondary goes in space,
    and it says nothing about which frame it is.

    ``on_unplaceable="align_centres"`` places centre-on-centre at each file's true pixel size
    — both when the files cannot prove a placement (as ``align_by_index`` does) AND when they
    can but no field overlaps at all (two wells, two dishes), which would otherwise draw
    nothing. Partly-overlapping files keep their stage placement: mixing the two in one
    overlay would put some fields where they are and others where they are not.

    ``z_pins`` only silence the FROZEN-IN-Z warning here (the pins are the user's own answer
    to it); the compositor applies them (:func:`secondary_z_weights`).

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

    # ── TIME first: its refusals stand whatever the spatial override says ─────────
    tp = plan_time(dst_md, int(getattr(dst_axes, "t", 1) or 1),
                   src_md, int(getattr(src_axes, "t", 1) or 1),
                   shift=t_shift, pairing=t_pairing, rate=rate, pins=t_pins)
    if tp["refusals"]:
        return PlacementPlan(tiles={}, coverage={}, scale=None, t_pairs=[],
                             z_offset_um={}, refusals=tuple(tp["refusals"]))
    t_extra = dict(t_map=tp["t_map"], sub_ticks=int(tp["sub_ticks"]))

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
        warnings.extend(f"OVERRIDDEN — {r}" for r in refusals)
        return _override_plan(dst_md, dst_axes, src_md, src_axes, tp, warnings,
                              t_shift=t_shift, on_unplaceable=on_unplaceable, t_extra=t_extra)

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

    t_pairs = tp["t_pairs"]

    # ── no field overlaps at all: the stage cannot relate these two files ─────────
    # (two wells, two dishes). `align_centres` is the user saying "show them together
    # anyway"; everything else keeps the honest blank-plus-warning below.
    if (on_unplaceable == "align_centres" and coverage
            and all(c <= 0.0 for c in coverage.values())):
        warnings.append(
            "no secondary field overlaps any primary field on the stage — the two files were "
            "imaged at different positions")
        return _override_plan(dst_md, dst_axes, src_md, src_axes, tp, warnings,
                              t_shift=t_shift, on_unplaceable=on_unplaceable, t_extra=t_extra)

    # ── DEGRADE: states you can see on screen ────────────────────────────────────
    warnings.extend(tp["warnings"])
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
        why = "the T pins / rate" if tp["t_map"] is not None else f"t_shift={t_shift}"
        warnings.append(
            f"{why} leaves timepoint(s) {_brief(unpaired)} with no secondary "
            f"frame — the overlay is empty there")
    warnings.extend(_held_still_note(dst_axes, src_axes, t_shift))
    if z_pins and float(offset_um[0]):
        warnings.append(f"Nudge Z ({float(offset_um[0]):+g} µm) is inert: Z pins decide "
                        f"which secondary plane goes with each primary plane")
    # ── the FROZEN-IN-Z warning (reported 2026-08-25) ────────────────────────────
    #
    # T is paired by INDEX and Z by absolute µm, so when two stacks do not overlap axially
    # the overlay still tracks T perfectly while Z looks completely dead: every plane of the
    # primary resolves to the same clamped end plane of the secondary. "Updating in T but
    # not in Z" is therefore not a hint that something is broken in the Z code — it is the
    # exact signature of a Z placement that has nothing to place against, and it is
    # invisible unless something says so.
    #
    # It is computable, so it gets computed, including the nudge that fixes it. The sign
    # falls out of `primary centre - secondary centre` because `offset_z` moves the
    # SECONDARY — which is also the sign users get wrong, since it flips when the same two
    # files are wired the other way round.
    for m, hits in tiles.items():
        if not hits or z_pins:
            # Z pins ARE the user's answer to "these stacks do not share a focus": the
            # pinned planes are paired whatever their stage Z says, so the warning (and its
            # Nudge Z advice, now inert) would be describing a state that no longer exists.
            break
        p_span = _z_span(dst_md, dst_axes, m)
        s_span = _z_span(src_md, src_axes, int(hits[0][0]))
        if p_span is None or s_span is None:
            continue
        s_lo, s_hi = s_span[0] + float(offset_um[0]), s_span[1] + float(offset_um[0])
        if s_lo <= p_span[1] and s_hi >= p_span[0]:
            break                       # they overlap somewhere: nothing to warn about
        gap = ((p_span[0] + p_span[1]) - (s_lo + s_hi)) / 2.0
        warnings.append(
            f"the two stacks do not overlap in Z (primary {p_span[0]:.0f}–{p_span[1]:.0f} µm, "
            f"secondary {s_lo:.0f}–{s_hi:.0f} µm), so the overlay is CLAMPED to one end plane "
            f"and will not move as you scroll Z — it still follows T, which is paired by "
            f"index. Set Nudge Z to {float(offset_um[0]) + gap:+.1f} µm to centre them. Two "
            f"objectives rarely focus at the same stage Z, and no file records the difference")
        break

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
                         z_offset_um=z_offset, refusals=(), warnings=tuple(warnings),
                         **t_extra)


def _override_plan(dst_md: Mapping[str, Any], dst_axes: Any,
                   src_md: Mapping[str, Any], src_axes: Any, tp: Dict[str, Any],
                   warnings: List[str], *, t_shift: int, on_unplaceable: str,
                   t_extra: Dict[str, Any]) -> PlacementPlan:
    """The plan when the user has taken the SPATIAL decision back — by index, or centre on
    centre. One builder for both the refusal override and the no-overlap fallback, so the two
    routes into ``align_centres`` cannot disagree about what it means.

    Centre placement needs a pixel size on both sides (it places at TRUE size); without one it
    degrades to index placement and says so, rather than inventing a size."""
    n_src_m = int(getattr(src_axes, "m", 1) or 1)
    n_dst_m = int(getattr(dst_axes, "m", 1) or 1)
    placed_by = "index"
    if on_unplaceable == "align_centres":
        if (lateral_extent_um(dst_md, dst_axes) is not None
                and lateral_extent_um(src_md, src_axes) is not None):
            placed_by = "centre"
            warnings.append(
                "placement is CENTRE-ON-CENTRE at each file's true pixel size, not by stage "
                "position: field m is paired with the secondary's field m (clamped to its "
                "last). Physical alignment is NOT verified — anchor it by eye with the µm "
                "nudge or the two-click Nudge pick")
        else:
            warnings.append("centre placement needs a pixel size on both files — placed by "
                            "INDEX instead")
    if placed_by == "index":
        warnings.append(
            "placement is by INDEX, not by stage position: field m is paired with the "
            "secondary's field m (clamped to its last). Physical alignment is NOT "
            "verified — check it by eye and correct it with the µm nudge")
    warnings.extend(tp["warnings"])
    # A still is the commonest thing to reach the override at all — a snapshot TIFF
    # carries no stage log — so the held-T note belongs on this path too.
    warnings.extend(_held_still_note(dst_axes, src_axes, t_shift))
    idx_tiles = {m: [(min(m, n_src_m - 1), 1.0)] for m in range(n_dst_m)}
    try:
        idx_scale = float(src_md["pixel_size_um"]) / float(dst_md["pixel_size_um"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        idx_scale = None
    if placed_by == "centre" and idx_scale is not None:
        coverage = {m: _centred_coverage(dst_md, dst_axes, src_md, src_axes) for m in idx_tiles}
    else:
        coverage = {m: 1.0 for m in idx_tiles}
    return PlacementPlan(
        tiles=idx_tiles, coverage=coverage, scale=idx_scale, t_pairs=tp["t_pairs"],
        z_offset_um={m: None for m in idx_tiles},
        refusals=(), warnings=tuple(warnings), placed_by=placed_by, **t_extra)


def _centred_coverage(dst_md: Mapping[str, Any], dst_axes: Any,
                      src_md: Mapping[str, Any], src_axes: Any) -> float:
    """Fraction of the primary field a centred secondary covers (a 60x inside a 20x: <1)."""
    a, b = _centred_box(dst_md, dst_axes), _centred_box(src_md, src_axes)
    return overlap_fraction(a, b) if (a is not None and b is not None) else 1.0


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


# ── position GROUPS: which multipoints belong to the same specimen ─────────────
#
# A multipoint acquisition is very often not one flat list of fields. The lab's
# `9.1.26_CRC_Gradient/Channel640_Seq0001.nd2` holds 54 positions that are really SIX 3x3
# mosaics — nine tiles at a 147 µm pitch across a 293 µm field (50 % overlap), and then a
# jump of a millimetre to the next specimen. Every consumer that treats `m` as a flat axis
# gets that wrong in the same way: Stitch fuses all 54 into one canvas with four huge holes
# in it, a `scope="dataset"` statistic pools six unrelated specimens, and a per-position
# table gives you 54 rows when the experiment had six samples.
#
# The grouping is not recorded anywhere in the file as such — NIS writes the point list flat
# — so it has to be RECOVERED from the geometry, which is what this section does.

#: Default gap threshold, as a multiple of the field of view: two fields whose centres are
#: further apart than this start different groups.
#:
#: One FOV is the honest place to put it, and the reason is not tuning. Tiles of one mosaic
#: must OVERLAP (or at worst abut) to be stitchable at all, so their centres are always
#: closer than one field; anything further apart is not a neighbouring tile of the same
#: mosaic, whatever else it is. That makes the threshold a statement about what a mosaic IS
#: rather than a number fitted to one plate — and :attr:`GroupPlan.margin` reports how much
#: room the answer actually had, so a layout where the choice mattered says so instead of
#: quietly returning one of several defensible answers.
GROUP_GAP_FACTOR = 1.0

#: How far two coordinates may differ and still count as the same row/column of a grid,
#: as a fraction of the inferred pitch. A stage repeats a nominal position to well under a
#: micron (the CRC file's nine tiles sit within 0.2 µm of three clean x levels), so this is
#: loose by two orders of magnitude on purpose: it is here to absorb a stage that settles
#: differently per row, not to make a decision.
_GRID_TOL = 0.25


@dataclass(frozen=True)
class PositionGroup:
    """One cluster of multipoints — the fields of a single specimen.

    ``members`` are multipoint indices in ACQUISITION order, which is the order Stitch and
    every table already address them in; the grid description below is derived from their
    coordinates and is advisory. ``rows``/``cols`` are the inferred mosaic shape and
    ``order`` says how the scan walked it, so a caller can present "3x3 serpentine" rather
    than nine numbers. A group whose fields do not lie on a grid at all still has valid
    ``members`` — it reports ``rows=cols=0`` and ``order="irregular"`` instead of forcing a
    shape onto it.
    """

    key: str
    members: Tuple[int, ...]
    #: ``(y, x)`` µm centre of the group's bounding box, or ``None`` when the group was READ
    #: rather than measured (a sidecar names members, not coordinates). ``None`` instead of
    #: ``(0, 0)`` deliberately: an origin is a real place on the stage, and a caller that
    #: drew a group there would be drawing a measurement nobody made.
    center_um: Optional[Tuple[float, float]] = None
    rows: int = 0
    cols: int = 0
    pitch_um: Tuple[float, float] = (0.0, 0.0)   # (y, x); 0 on an axis with one level
    order: str = "irregular"                # serpentine | raster | single | irregular

    @property
    def size(self) -> int:
        return len(self.members)

    def brief(self) -> str:
        """One line for a menu or an error: ``"G2 — 9 positions, 3x3 serpentine"``."""
        shape = (f", {self.rows}x{self.cols} {self.order}"
                 if self.rows and self.cols else "")
        noun = "position" if self.size == 1 else "positions"
        return f"{self.key} — {self.size} {noun}{shape}"


@dataclass(frozen=True)
class GroupPlan:
    """The whole grouping of one Dataset's multipoint axis, plus how sure it is.

    ``placed`` is the honest failure: a Dataset whose fields cannot be located at all (no
    ``origin_um``, no ``stage_xy_um``, no pixel size) gets an EMPTY plan rather than one
    group containing everything. The two are very different answers and a caller must be
    able to tell them apart — "they are all one specimen" is a claim, "I cannot see where
    they are" is not.

    ``margin`` is the ratio between the narrowest gap SEPARATING two groups and the widest
    gap INSIDE any of them (formally: the smallest inter-cluster distance over the largest
    single-linkage edge, i.e. the MST bottleneck). Any threshold between those two numbers
    produces this exact grouping, so the margin says how much the answer depended on
    :data:`GROUP_GAP_FACTOR`:

    * ``margin`` well above 1 — the layout separates cleanly and the threshold is not doing
      the work. The CRC file measures **7.1** (147 µm inside a mosaic, 1039 µm between).
    * ``margin`` near 1 — the gap the grouping turns on is barely wider than the gaps it is
      ignoring, and a different threshold would give a different answer. Say so; do not
      present the result as discovered fact.
    * ``inf`` — one group (or none), so nothing was separated and no threshold mattered.
    """

    groups: Tuple[PositionGroup, ...] = ()
    gap_um: float = 0.0
    margin: float = float("inf")
    placed: bool = False

    def __len__(self) -> int:
        return len(self.groups)

    def of_member(self, m: int) -> Optional[PositionGroup]:
        """The group multipoint ``m`` belongs to, or ``None``."""
        for g in self.groups:
            if int(m) in g.members:
                return g
        return None

    def labels(self) -> Tuple[str, ...]:
        """One group key per multipoint — the per-M list :data:`POSITION_GROUP_KEY` holds.

        Empty when the plan is unplaced, because a list of the wrong length read
        positionally is the failure this module refuses everywhere else.
        """
        if not self.placed or not self.groups:
            return ()
        n = 1 + max(max(g.members) for g in self.groups)
        out: List[str] = [""] * n
        for g in self.groups:
            for m in g.members:
                out[m] = g.key
        return tuple(out)


def _group_centers(md: Mapping[str, Any], axes: Any
                   ) -> Optional[Tuple[np.ndarray, float]]:
    """Every multipoint's field CENTRE in µm as an ``(M, 2)`` ``[y, x]`` array, plus the
    field size to measure gaps against — or ``None`` when any field cannot be placed.

    Routed through :func:`field_box` rather than reading a stage key directly, so this
    inherits the preference order the rest of the module already agreed on: the
    transform-maintained ``origin_um`` first, the raw stage log as the fallback, and any
    correction ``registration.align_to`` measured folded in. A grouping computed off a
    different coordinate than the one Stitch will place the tiles with would be a grouping
    that disagrees with the picture it produces.

    All-or-nothing on purpose (the :func:`~nodegraph.nodes._stitch_stage_xy` rule): a log
    covering some positions would cluster the ones it has and silently drop the rest, and
    the result looks exactly like a complete answer.
    """
    n = int(getattr(axes, "m", 0) or 0)
    if n <= 0:
        return None
    ext = lateral_extent_um(md, axes)
    if ext is None:
        return None
    pts = np.empty((n, 2), dtype=float)
    for m in range(n):
        box = field_box(md, axes, m)
        if box is None:
            return None
        pts[m] = (0.5 * (box.y0 + box.y1), 0.5 * (box.x0 + box.x1))
    return pts, float(max(ext))


def _single_linkage(pts: np.ndarray, gap: float) -> Tuple[List[List[int]], float, float]:
    """Cluster ``pts`` by single linkage at ``gap``; also return the two margin numbers.

    Union-find over the full pair list. That is O(M^2) and deliberately not a KD-tree: M is
    the number of stage positions, which is tens — 54 on the file this was written for,
    1536 on a pathological plate — so the pair matrix is at most a few megabytes and the
    tree would cost more to build than the scan saves. It also keeps this module dependency
    free (numpy only), which is what lets it run inside an edit-time ``meta_transform`` on
    every keystroke.

    Returned margins are ``(bottleneck, separation)``: the widest edge single linkage had to
    ACCEPT to build these clusters, and the narrowest distance between two of them.
    """
    n = len(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    near = np.asarray(d[iu] <= gap).nonzero()[0]
    for e in near:
        i, j = int(iu[0][e]), int(iu[1][e])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    clusters = sorted((sorted(v) for v in buckets.values()), key=lambda v: v[0])

    # The bottleneck is the widest edge of each cluster's minimum spanning tree. Computed
    # as the widest step of a Prim walk, which for a cluster of this size is one small dense
    # scan and avoids pulling in scipy for a number that only ever gets REPORTED.
    bottleneck = 0.0
    for c in clusters:
        if len(c) < 2:
            continue
        idx = np.asarray(c)
        sub = d[np.ix_(idx, idx)]
        seen = [0]
        rest = set(range(1, len(idx)))
        while rest:
            best = min(((sub[a, b], b) for a in seen for b in rest), key=lambda t: t[0])
            bottleneck = max(bottleneck, float(best[0]))
            seen.append(best[1])
            rest.discard(best[1])

    separation = float("inf")
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            ia, ib = np.asarray(clusters[a]), np.asarray(clusters[b])
            separation = min(separation, float(d[np.ix_(ia, ib)].min()))
    return clusters, bottleneck, separation


def _levels(vals: np.ndarray, tol_frac: float = _GRID_TOL) -> Tuple[List[float], float]:
    """Collapse 1-D coordinates onto the distinct grid LEVELS they sample, and the pitch.

    Sorted-and-split rather than rounded to a bin: a fixed bin boundary falling between two
    readings of the same nominal position splits one row in two, and which readings land
    either side of it depends on where the stage happened to settle — the dead-band failure
    that makes a layout parse differently on two files from the same protocol. Splitting at
    the LARGEST gaps instead means the answer depends on the spacing of the data, not on the
    phase of a grid this function chose.
    """
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) <= 1:
        return ([float(v[0])] if len(v) else []), 0.0
    steps = np.diff(v)
    real = steps[steps > 0]
    if not len(real):
        return [float(v[0])], 0.0
    # The pitch is the smallest step that is not stage jitter: take the median of the steps
    # in the upper half, which is the spacing between ADJACENT levels rather than within one.
    pitch = float(np.median(real[real >= 0.5 * float(real.max())]))
    tol = max(tol_frac * pitch, 1e-9)
    levels = [float(v[0])]
    for prev, cur, step in zip(v[:-1], v[1:], steps):
        if step > tol:
            levels.append(float(cur))
    return levels, (pitch if len(levels) > 1 else 0.0)


def _grid_of(pts: np.ndarray, members: Sequence[int]
             ) -> Tuple[int, int, Tuple[float, float], str]:
    """Infer ``(rows, cols, (pitch_y, pitch_x), order)`` for one group's fields."""
    sub = pts[np.asarray(members)]
    ys, py = _levels(sub[:, 0])
    xs, px = _levels(sub[:, 1])
    rows, cols = len(ys), len(xs)
    if rows * cols != len(members):
        # Not a filled rectangle — a partial mosaic, a hand-picked scatter, or a stage that
        # moved diagonally. Report the levels honestly and refuse to name a scan order for a
        # walk that is not on a grid.
        return rows, cols, (py, px), "irregular"
    if rows == 1 and cols == 1:
        return rows, cols, (py, px), "single"

    def level_of(v: float, levels: Sequence[float]) -> int:
        return int(np.argmin([abs(v - L) for L in levels]))

    walk = [(level_of(sub[i, 0], ys), level_of(sub[i, 1], xs))
            for i in range(len(members))]
    if len({w for w in walk}) != len(members):
        return rows, cols, (py, px), "irregular"
    # A raster scan visits every row left-to-right; a serpentine (boustrophedon) one
    # alternates. Read the direction each row was actually walked in rather than assuming:
    # the CRC file's mosaics run RIGHT-to-left first, so "the first row ascends" is not a
    # property a scan order may be tested on.
    per_row: Dict[int, List[int]] = {}
    for r, c in walk:
        per_row.setdefault(r, []).append(c)
    if any(len(v) != cols for v in per_row.values()):
        return rows, cols, (py, px), "irregular"
    dirs = []
    for r in sorted(per_row):
        seq = per_row[r]
        if seq == sorted(seq):
            dirs.append(1)
        elif seq == sorted(seq, reverse=True):
            dirs.append(-1)
        else:
            return rows, cols, (py, px), "irregular"
    if rows == 1 or len(set(dirs)) == 1:
        return rows, cols, (py, px), "raster"
    if all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1)):
        return rows, cols, (py, px), "serpentine"
    return rows, cols, (py, px), "irregular"


def group_key(i: int) -> str:
    """The default key of the ``i``-th group (0-based) — ``G1``, ``G2``, …

    1-based in the TEXT because the thing it names is a specimen the user picked on the
    microscope, and NIS numbers those from 1. The index stays 0-based everywhere in code.
    """
    return f"G{int(i) + 1}"


def position_groups(md: Mapping[str, Any], axes: Any,
                    gap_factor: float = GROUP_GAP_FACTOR) -> GroupPlan:
    """Recover which multipoints of ``md``/``axes`` belong to the same specimen.

    Single-linkage clustering of the field centres, cutting any link longer than
    ``gap_factor`` field widths (:data:`GROUP_GAP_FACTOR`). Single linkage — not k-means,
    not a fixed group size — because the question is genuinely "is there a gap here?", and
    the shape of a mosaic is not known in advance: the CRC file is six 3x3s, but the same
    protocol run with one position skipped is six groups of 8 and 9, and any method told how
    many groups or how big to expect them would return a confident wrong answer for it.

    Deterministic and dependency-free (numpy only), so the edit-time
    ``meta_transform`` behind ``util.select_group`` can call it on every keystroke and get
    the same answer the pull will.

    Returns an unplaced (empty) :class:`GroupPlan` when the fields cannot be located — see
    :attr:`GroupPlan.placed`, and note that this is NOT the same as finding one group.
    """
    got = _group_centers(md, axes)
    if got is None:
        return GroupPlan()
    pts, fov = got
    gap = float(gap_factor) * fov
    clusters, bottleneck, separation = _single_linkage(pts, gap)
    groups = []
    for i, members in enumerate(clusters):
        sub = pts[np.asarray(members)]
        rows, cols, pitch, order = _grid_of(pts, members)
        groups.append(PositionGroup(
            key=group_key(i), members=tuple(int(m) for m in members),
            center_um=(float(0.5 * (sub[:, 0].min() + sub[:, 0].max())),
                       float(0.5 * (sub[:, 1].min() + sub[:, 1].max()))),
            rows=rows, cols=cols, pitch_um=pitch, order=order))
    margin = (separation / bottleneck) if bottleneck > 0 else float("inf")
    return GroupPlan(groups=tuple(groups), gap_um=gap, margin=margin, placed=True)
