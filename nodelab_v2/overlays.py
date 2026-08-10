"""Viewer overlay system — per-domain settings, persistence and the painter.

One place owns *what an overlay looks like* for every attribute domain the Viewer can
draw on top of the image: **Points**, **Labels**, **Tracks**, **Mesh**, and (reserved,
not drawn yet) **Voxels**. The Viewer keeps a single :class:`OverlaySettings` and a
single :class:`OverlayRenderer`; the popup in :mod:`nodelab_v2.overlay_dialog` edits the
settings and the renderer paints them.

Four properties are structural, not incidental:

* **Everything is painted in WIDGET space, in screen pixels.** The renderer is handed a
  ``plane px → widget px`` mapping (:attr:`OverlayFrame.map_pt`) and every size in the
  settings (outline width, glyph spread, vertex radius) is a *screen* size. So an outline
  stays exactly as thick when you zoom into a label as it was zoomed out — which the old
  overlays could not do on the CPU path, because they were baked into the image pixmap
  and magnified with it. Region **fills** are the one thing that scales, because a fill
  *is* the region.
* **The two backends share this code.** Both :class:`~nodelab_v2.glview.GLImageView` and
  the CPU ``_ImageView`` call back with a ``QPainter`` in widget coordinates and expose
  ``plane_to_widget``, so there is exactly one implementation of every overlay look.
* **Every id label goes through one painter.** Any overlay that draws an integer id —
  region ids, track ids, whatever a later domain adds — calls
  :meth:`OverlayRenderer.draw_id` and nothing else. Small text over an arbitrary image
  needs glyph-outline stroking (not a second offset ``drawText``), device-pixel-snapped
  origins, and luminance-lifted ink; reimplementing that per domain is how ids end up as
  unreadable grain. The method's docstring carries the reasoning for each rule.
* **Field specs drive the UI.** :data:`FIELDS` describes each setting (kind, range,
  choices, what it depends on, and its cross-domain *role*). The dialog builds its
  controls from that, and :func:`spread_settings` copies a tab's look onto the other tabs
  *by role* — so "opacity" travels from Points to Labels even though "arm spread" cannot.
* **Per-item colour keys off an object, not an id.** A segmentation re-issues its label
  ids from 1 on every frame, so "a colour per id" makes one cell flash a different colour
  at every T step. Items are therefore coloured by a *palette slot* that belongs to the
  physical object (its track, when one links it), and neighbouring objects are pushed
  apart on the colour wheel — see the identity-palette section below.

Settings persist as JSON (partial dicts merge, so a file written by an older build still
loads). Three layers, later wins: built-in defaults → the project file shipped in this
package (:data:`PROJECT_FILE`, meant to be committed) → this machine's file
(:data:`USER_FILE`). ``NODELAB_OVERLAYS`` points at one explicit file instead.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields as dc_fields
from functools import lru_cache
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF, QTransform)

SCHEMA = 1

#: How many channels the GL composite shader can sample in one pass — its sampler-bank
#: size, and a hard limit on what the composite can show.
#:
#: It lives in this module, which neither backend can avoid importing, because it is needed
#: by BOTH: :mod:`nodelab_v2.glview` builds its fragment shader from it, and
#: :mod:`nodelab_v2.viewer` has to know the number to warn about — while deliberately not
#: importing ``glview`` at module level, since the viewer must keep working with no GL at all
#: (``NODELAB_GL=0``, a headless probe, or a driver failure that falls back to the CPU
#: renderer). Two copies of this number would mean the warning could quote a limit the
#: shader does not enforce, which is worse than no warning.
GL_MAX_CHANNELS = 8

#: Golden-angle hue step — successive integer indices land far apart on the colour wheel,
#: so "each label / point / track a different colour" stays legible for hundreds of items.
GOLDEN_ANGLE = 137.507764

#: Above this many label ids **in view** the ids are not drawn — that many numbers on one
#: screen is noise, not information. Culling happens first, so zooming in brings ids back.
#: (Declared here because the Labels tab tooltip quotes it.)
MAX_ID_LABELS = 400


# ── colour helpers (ONE hue→rgb implementation, shared by outlines and fills) ────
def _hues_to_rgb(hues: np.ndarray, sat: float, val: float) -> np.ndarray:
    """Vectorized HSV→RGB for an array of hues (degrees) at a common ``sat``/``val``
    (both 0..1) → ``(N, 3)`` uint8.

    Both the per-item outline colours and the per-label fill LUT go through this, so a
    label's outline and its fill can never disagree about what colour it is.
    """
    hh = (np.asarray(hues, dtype=float) / 60.0) % 6.0
    i = np.floor(hh).astype(int)
    f = hh - i
    v = np.full(hh.shape, float(val))
    p = v * (1.0 - sat)
    q = v * (1.0 - sat * f)
    t = v * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [v, q, p, p, t, v], default=v)
    g = np.select(conds, [t, v, v, q, p, p], default=v)
    b = np.select(conds, [p, p, t, v, v, q], default=v)
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0, 255).round().astype(np.uint8)


@lru_cache(maxsize=1 << 15)
def _distinct_rgb(index: int, sat: int, val: int) -> Tuple[int, int, int]:
    """``distinct_color`` without the alpha — memoized, because one numpy HSV conversion
    per item per paint is a real cost once there are hundreds of labels or thousands of
    points on screen (it dominated the label-outline frame time)."""
    rgb = _hues_to_rgb(np.array([(int(index) * GOLDEN_ANGLE) % 360.0]),
                       max(0.0, min(1.0, sat / 255.0)),
                       max(0.0, min(1.0, val / 255.0)))[0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def distinct_color(index: int, sat: int = 205, val: int = 255,
                   opacity: int = 100) -> QColor:
    """A deterministic, well-spread colour for item ``index`` (golden-angle hue).

    ``sat``/``val`` are 0..255 (as stored in the settings) and ``opacity`` is a percent.
    Stable across frames — which is what makes a track keep its colour over T.
    """
    r, g, b = _distinct_rgb(int(index), int(sat), int(val))
    col = QColor(r, g, b)
    col.setAlpha(_alpha(opacity))
    return col


def _alpha(opacity_pct: float) -> int:
    return max(0, min(255, int(round(255.0 * float(opacity_pct) / 100.0))))


def qcolor(spec: str, opacity: int = 100) -> QColor:
    """A settings colour string (``#rrggbb``) as a QColor at ``opacity`` percent."""
    col = QColor(str(spec))
    if not col.isValid():
        col = QColor("#ffffff")
    col.setAlpha(_alpha(opacity))
    return col


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Blend ``a`` toward ``b`` by ``t``, keeping ``a``'s alpha."""
    out = QColor(round(a.red() * (1 - t) + b.red() * t),
                 round(a.green() * (1 - t) + b.green() * t),
                 round(a.blue() * (1 - t) + b.blue() * t))
    out.setAlpha(a.alpha())
    return out


# ── identity palette: one colour per PHYSICAL object, neighbours kept apart ─────
#
# ``distinct_color`` answers "a colour per id". That is the wrong question the moment a
# tracker runs: a cell's label id is re-issued from 1 on every frame, so colouring by it
# makes the same cell flash a different colour at every T step. The fix is one level of
# indirection — a **palette slot** per *physical object* (a track, or an untracked region)
# that every frame's id maps into. Two things then have to hold, and the second is why
# this is not a one-liner:
#
#  * one object, one slot, for all time — that is what makes a tracked cell keep its
#    colour while its id churns underneath;
#  * two objects the eye compares SIDE BY SIDE must not land on the same hue. The golden
#    angle spreads *consecutive* indices beautifully and says nothing about distant ones:
#    slots 5 and 39 sit 4.7° apart and are indistinguishable. Harmless on opposite sides
#    of the field; a wrong reading when they are two touching cells.
#
# So slots are assigned greedily against a **neighbour graph**, each object *preferring*
# the slot it would have had anyway — its own id, or (for a track) the id of its first
# member — and moving only when a neighbour already sits too close on the wheel. Untracked
# data therefore keeps exactly the colours it has today, except where two neighbours
# genuinely collided.

#: Degrees of hue two NEIGHBOURING objects must differ by. Around 30° is where two fills
#: at the default saturation stop reading as "the same colour, maybe".
MIN_HUE_SEP = 30.0

#: How many nearest objects count as an object's neighbours when the graph is built.
NEIGHBOR_K = 6

#: Per frame, above this many items the brute-force neighbour search is skipped (scipy's
#: KD-tree has no such limit — this is only the no-scipy fallback's cutoff).
MAX_NEIGHBOR_ITEMS = 20_000

#: Ceiling on how many palette slots the de-conflict pass will search through. The golden
#: angle equidistributes (three-distance theorem), so 512 slots already sample the wheel
#: to under a degree — searching further buys no colour and costs real time.
MAX_SLOT_SEARCH = 512

#: Ceiling on how many distinct neighbour hues constrain one object. Past a couple of dozen
#: the wheel is saturated: every choice is a bad choice and the extra constraints only make
#: the search matrix bigger. Over the cap the hues are sampled evenly *around the circle*,
#: so the picture of which arcs are blocked survives — it just gets coarser.
MAX_CONSTRAINT_HUES = 32


def _hues(slots: np.ndarray) -> np.ndarray:
    """The hue (degrees) each palette slot paints — the rule ``distinct_color`` uses."""
    return (np.asarray(slots, dtype=float) * GOLDEN_ANGLE) % 360.0


def _hue_gaps(hues: np.ndarray, against: np.ndarray) -> np.ndarray:
    """For each hue, its distance (degrees, the short way round the wheel) to the NEAREST
    hue in ``against``."""
    d = np.abs(np.asarray(hues, dtype=float)[:, None] - np.asarray(against, float)[None, :])
    return np.minimum(d, 360.0 - d).min(axis=1)


def neighbor_pairs(pos: np.ndarray, k: int = NEIGHBOR_K) -> np.ndarray:
    """Index pairs joining each row of ``pos`` ``(N, D)`` to its ``k`` nearest rows.

    Returns a deduplicated ``(P, 2)`` int array with ``a < b`` — the "these two are close
    enough that the eye will compare them" graph :func:`deconflict_slots` colours against.
    Uses scipy's KD-tree when it imports (it always should — scipy is a hard dependency);
    the numpy fallback is capped at :data:`MAX_NEIGHBOR_ITEMS` rows, above which building
    the graph costs more than the confusion it prevents.
    """
    p = np.asarray(pos, dtype=np.float32)
    if p.ndim != 2 or len(p) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    n = len(p)
    k = int(min(max(1, int(k)), n - 1))
    try:
        from scipy.spatial import cKDTree
        idx = np.asarray(cKDTree(p).query(p, k=k + 1)[1])[:, 1:]     # col 0 is self
    except Exception:                              # noqa: BLE001 — colour is never fatal
        if n > MAX_NEIGHBOR_ITEMS:
            return np.zeros((0, 2), dtype=np.int64)
        idx = np.empty((n, k), dtype=np.int64)
        for lo in range(0, n, 128):                # chunked: the full N² matrix is too big
            hi = min(n, lo + 128)
            d = ((p[lo:hi, None, :] - p[None, :, :]) ** 2).sum(axis=-1)
            d[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
            idx[lo:hi] = np.argpartition(d, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    pairs = np.sort(np.stack([rows, np.asarray(idx).ravel()], axis=1), axis=1)
    return np.unique(pairs, axis=0)


def deconflict_slots(prefer: Mapping[Any, int], pairs: Iterable[Tuple[Any, Any]],
                     min_sep: float = MIN_HUE_SEP) -> Dict[Any, int]:
    """Assign a palette slot to every object in ``prefer``, keeping neighbours apart.

    ``prefer`` maps an object key to the slot it *would* have had with no de-confliction
    (its own id, or its track's first member id); ``pairs`` are the object pairs that sit
    next to each other. Each object keeps its preferred slot unless an already-placed
    neighbour is within ``min_sep`` degrees of it — only then does it move, to the nearest
    slot *above* its preference that clears every neighbour.

    Searching upward from the object's OWN preference rather than from zero is what keeps
    the palette varied: the golden angle samples the wheel just as finely starting from
    500 as from 0, and because preferences are distinct ids, the alternatives an object
    finds are ones nothing else has claimed. Searching a shared low range instead would
    funnel every displaced object into the same few dozen colours.

    Deterministic: objects are placed in ``(preferred slot, key)`` order and every
    tie-break is an index scan. The required separation shrinks for a crowded object
    (``330/(n+1)``), because 8 neighbours cannot all be 30° from each other *and* from the
    newcomer — better a guaranteed-feasible spread than an arbitrary give-up.
    """
    adj: Dict[Any, set] = {}
    for a, b in pairs:
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    offsets = np.arange(1, MAX_SLOT_SEARCH + 1, dtype=np.int64)
    taken: set = set()
    slot: Dict[Any, int] = {}
    for obj in sorted(prefer, key=lambda o: (int(prefer[o]), str(o))):
        want = int(prefer[obj])
        placed = [slot[nb] for nb in adj.get(obj, ()) if nb in slot]
        if placed:
            hues = np.unique(_hues(np.asarray(placed)))     # sorted around the wheel
            sep = min(float(min_sep), 330.0 / (len(hues) + 1))
            if len(hues) > MAX_CONSTRAINT_HUES:             # even sample, arcs preserved
                hues = hues[np.linspace(0, len(hues) - 1, MAX_CONSTRAINT_HUES).astype(int)]
            if float(_hue_gaps(_hues(np.array([want])), hues)[0]) < sep:
                gaps = _hue_gaps(_hues(want + offsets), hues)
                ok = np.nonzero(gaps >= sep)[0]
                if not ok.size:                             # the wheel is full: least bad
                    want += int(offsets[int(np.argmax(gaps))])
                else:
                    # the nearest clearing slot, skipping ones already spoken for
                    free = [want + int(offsets[j]) for j in ok[:64].tolist()
                            if want + int(offsets[j]) not in taken]
                    want = free[0] if free else want + int(offsets[int(ok[0])])
        slot[obj] = want
        taken.add(want)
    return slot


def slot_lut(ids: np.ndarray, slots: np.ndarray,
             size: Optional[int] = None) -> np.ndarray:
    """``id → palette slot`` as a dense int array, identity where an id has no object.

    A dense LUT rather than a dict because this is consulted per *pixel* (the label fill
    builds its colour table straight out of it) and per paint, not per object.
    """
    ids = np.asarray(ids, dtype=np.int64).ravel()
    slots = np.asarray(slots, dtype=np.int64).ravel()
    n = int(size) if size is not None else int(ids.max() + 1) if ids.size else 1
    lut = np.arange(max(1, n), dtype=np.int64)
    keep = (ids >= 0) & (ids < lut.size)
    lut[ids[keep]] = slots[keep]
    return lut


def slot_of(keys: Optional[np.ndarray], value: int) -> int:
    """One item's palette slot — ``keys[value]`` when a LUT covers it, else the value
    itself (no palette ⇒ exactly the old "a colour per id" behaviour)."""
    v = int(value)
    if keys is not None and 0 <= v < len(keys):
        return int(keys[v])
    return v


# ── settings model ──────────────────────────────────────────────────────────────
@dataclass
class PointsOverlay:
    """Point-domain detections. The default look is the requested **golden star**: a
    bright centre pixel exactly on the detection, with arms stepping ``spread`` pixels
    out in each direction and dimming as they go.

    ``z_project`` governs **every** layer, 2-D and 3-D alike: off, only the detections whose
    ``z`` lands on the viewed plane are drawn. Leaving a 3-D layer projected regardless (a
    2026-08-03 revision did) makes the control inert AND flattens a per-point palette into
    one wash, because every plane's glyphs overlap — see
    :meth:`~nodelab_v2.viewer.ViewerPanel._point_marks`."""
    enabled: bool = True
    #: WHICH Point table to draw (``""`` = all of them, the historical behaviour).
    #:
    #: Drawing every layer is right for one detection but wrong the moment a node emits a
    #: FILTERED view of another's cloud: `analysis.voronoi` publishes the seeds that actually
    #: won a territory as `<name>_seeds`, and with no selector the display showed those *and*
    #: the full input — including the dots the node dropped, which is what the caller asked to
    #: stop seeing (2026-08-04).
    layer: str = ""
    shape: str = "star"
    spread: int = 3
    unit_px: float = 3.0
    thickness: float = 1.6
    gradient: bool = True
    center_boost: bool = True
    color_mode: str = "single"
    color: str = "#ffc83c"          # gold
    sat: int = 205
    val: int = 255
    opacity: int = 100
    z_project: bool = False
    off_opacity: int = 30


@dataclass
class LabelsOverlay:
    """Label rasters (an integer Voxel-domain layer). Defaults deliberately favour
    *visibility*: a distinct colour per region, a 2 px outline and a light fill.

    ``per_label`` colours by the region's **palette slot**, not its raw id — so once a
    tracker has linked the regions, one cell keeps one colour across every T (see the
    identity-palette section above). Untracked regions fall back to their own id.
    """
    enabled: bool = True
    #: WHICH integer Voxel raster to draw (``""`` = the one with the most regions).
    #:
    #: There was no such control until 2026-08-04, and the automatic choice is not a
    #: preference — it is a guess. A Dataset routinely carries several label rasters at once
    #: (``analysis.voronoi`` alone emits its territories, inherits the seeds' branch labels
    #: and copies the areas it clipped to), and "most regions" then draws whichever happens
    #: to be the most fragmented. Reported as "it pulls the wrong label — I select the
    #: segmentation from the Red channel but it still shows the UV labels": there was
    #: nothing to select, and the heuristic was choosing.
    layer: str = ""
    style: str = "both"             # outline | fill | both
    width: float = 2.0
    color_mode: str = "per_label"
    color: str = "#e08a3a"          # Domain.LABEL
    sat: int = 205
    val: int = 255
    opacity: int = 100
    fill_opacity: int = 30
    show_ids: bool = False
    id_px: int = 11


@dataclass
class TracksOverlay:
    """Track trajectories. ``per_track`` colouring goes through the same palette slot the
    track's member regions use — literally the same rule as labels — so a trajectory keeps
    its colour across every T *and* matches the cell it belongs to."""
    enabled: bool = True
    width: float = 1.8
    color_mode: str = "per_track"
    color: str = "#c264a0"          # Domain.TRACK
    sat: int = 205
    val: int = 255
    opacity: int = 100
    vertex_px: float = 1.8
    head_px: float = 5.0
    trail: str = "all"              # all | past | window
    window: int = 5
    fade: bool = False
    show_ids: bool = False
    id_px: int = 10


@dataclass
class VoxelsOverlay:
    """Scalar Voxel layers as a colour-mapped wash — strain, a distance transform, a
    density, a probability, a mask (V2.19: declared since v2.00, drawn at last).

    ``alpha_mode`` is the setting that decides whether this is usable rather than merely
    present: a flat wash over a whole field hides exactly the pixels you are trying to
    judge the field against, so the default lets the low end stay transparent. See
    :func:`nodelab_v2.overlay_render.scalar_rgba`, which owns the arithmetic."""
    enabled: bool = False
    style: str = "heatmap"          # heatmap | mask | contour
    colormap: str = "viridis"
    opacity: int = 45
    threshold: float = 0.0
    width: float = 1.4
    #: flat | ramp | gated — how transparency follows the value
    alpha_mode: str = "ramp"
    #: limits from the data's own percentiles, or the explicit pair below
    clim_auto: bool = True
    clim_lo: float = 0.0
    clim_hi: float = 1.0
    #: symmetric limits about zero, so the midpoint colour lands on zero. Only meaningful
    #: for a DIVERGING colormap, and only honoured for one — a signed field read through a
    #: sequential map cannot show its sign at all.
    center_zero: bool = False


@dataclass
class VectorsOverlay:
    """A displacement field (DIC/DVC ``u``/``v``) as decimated arrows (V2.19).

    Screen-space like every other overlay: an arrow keeps its weight and head size as you
    zoom, so zooming inspects the field rather than magnifying the drawing of it."""
    enabled: bool = True
    every: int = 4                  # draw 1 in N vectors
    scale: float = 1.0              # multiplies BOTH components — never one
    gate: float = 0.0               # hide vectors shorter than this
    color_mode: str = "magnitude"   # magnitude | flat
    colormap: str = "turbo"
    color: str = "#6fe3ff"
    width: float = 1.3
    head_px: float = 5.0
    opacity: int = 90


@dataclass
class DiffOverlay:
    """Two object sets matched against each other and split TP / FP / FN (V2.19).

    The validation view: "did my new segmentation get better". It draws the PRIMARY set's
    objects coloured by whether a partner was found in the comparison set, plus the
    comparison-only objects that were missed — so a screen full of green is agreement and
    every coloured mark is a disagreement worth looking at.

    Colours are green / red / blue rather than a red-green pair on purpose: red-green is
    the commonest colour-vision deficiency, and this overlay exists to be judged by eye."""
    enabled: bool = False
    match: str = "distance"         # distance (points) | iou (labels)
    max_dist: float = 5.0           # µm, for `distance`
    min_iou: float = 0.5            # for `iou`
    radius_px: float = 6.0
    width: float = 1.6
    show_links: bool = True         # an arrow from each matched A object to its partner
    opacity: int = 95


@dataclass
class MeshOverlay:
    """Mesh-domain boundary surfaces (V2.08), drawn as their **cross-section at the viewed
    Z** — the honest 2-D reading of a 3-D surface, and the one that lines up with the
    image underneath. Projecting all the 3-D edges instead would be an unreadable tangle
    that says nothing about the plane you are looking at."""
    enabled: bool = True
    style: str = "wireframe"        # wireframe | surface | points
    width: float = 1.4
    color_mode: str = "per_object"
    color: str = "#b06ad8"          # Domain.MESH
    sat: int = 205
    val: int = 255
    opacity: int = 100
    fill_opacity: int = 25
    vertex_px: float = 1.6
    near_z: float = 1.0             # 'points' style: how many Z planes count as "near"


@dataclass
class OverlaySettings:
    """The whole overlay configuration — one group per tab in the popup."""
    points: PointsOverlay = field(default_factory=PointsOverlay)
    labels: LabelsOverlay = field(default_factory=LabelsOverlay)
    tracks: TracksOverlay = field(default_factory=TracksOverlay)
    voxels: VoxelsOverlay = field(default_factory=VoxelsOverlay)
    vectors: VectorsOverlay = field(default_factory=VectorsOverlay)
    diff: DiffOverlay = field(default_factory=DiffOverlay)
    mesh: MeshOverlay = field(default_factory=MeshOverlay)

    def group(self, tab: str):
        return getattr(self, tab)

    # ── (de)serialization — partial dicts MERGE onto the current values, so a file
    #    from an older build (or a hand-edited one) never wipes unknown settings ──
    def to_dict(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "overlays": {t: asdict(self.group(t)) for t in TABS}}

    def update_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """Merge ``data`` in; returns the ``tab.key`` names actually changed."""
        blob = data.get("overlays", data) if isinstance(data, dict) else {}
        changed: List[str] = []
        for tab in TABS:
            src = blob.get(tab)
            if not isinstance(src, dict):
                continue
            grp = self.group(tab)
            valid = {f.name: f.type for f in dc_fields(grp)}
            for key, raw in src.items():
                if key not in valid:
                    continue                      # unknown key: ignore, never crash
                try:
                    val = _coerce(getattr(grp, key), raw)
                except (TypeError, ValueError):
                    continue
                if val != getattr(grp, key):
                    setattr(grp, key, val)
                    changed.append(f"{tab}.{key}")
        return changed

    def copy(self) -> "OverlaySettings":
        out = OverlaySettings()
        out.update_from_dict(self.to_dict())
        return out


def defaults_for(tab: str) -> Dict[str, Any]:
    """The built-in default values for one tab (what "Reset tab" restores)."""
    return asdict(OverlaySettings().group(tab))


def _coerce(current: Any, raw: Any) -> Any:
    """Cast ``raw`` to the type of the current value (JSON gives us floats for ints)."""
    if isinstance(current, bool):
        return bool(raw)
    if isinstance(current, int):
        return int(round(float(raw)))
    if isinstance(current, float):
        return float(raw)
    return str(raw)


# ── tab metadata ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TabInfo:
    key: str
    title: str
    implemented: bool
    blurb: str


TAB_INFO: Tuple[TabInfo, ...] = (
    TabInfo("points", "Points", True,
            "Point-domain detections on the viewed plane (detect.spots, "
            "detect.particles, DVC/DIC field samples)."),
    TabInfo("labels", "Labels", True,
            "Integer label rasters — the Voxel-domain label layer with the most "
            "regions on the viewed plane (analysis.segment, analysis.label)."),
    TabInfo("tracks", "Tracks", True,
            "Track-domain trajectories: member positions joined across time "
            "(track.link, track.objects)."),
    TabInfo("voxels", "Voxels", True,
            "Scalar Voxel layers as a colour-mapped wash — strain, distance transforms, "
            "densities, probabilities, masks. Transparency follows the value, so the "
            "image shows through where the field has nothing to report."),
    TabInfo("vectors", "Vectors", True,
            "Displacement fields as decimated arrows, optionally coloured by magnitude "
            "(analysis.dvc_field, analysis.dic_correlate)."),
    TabInfo("diff", "Compare", True,
            "Two object sets matched and split into matched / only-here / only-there — "
            "the 'did it get better' view for a segmentation or a detector."),
    TabInfo("mesh", "Mesh", True,
            "Mesh-domain boundary surfaces, drawn as their cross-section at the viewed "
            "Z (analysis.tessellate)."),
)
TABS: Tuple[str, ...] = tuple(t.key for t in TAB_INFO)
TAB_BY_KEY: Dict[str, TabInfo] = {t.key: t for t in TAB_INFO}

#: Per tab, the ``color_mode`` value that means "give every item its own colour".
PER_ITEM_MODE: Dict[str, Optional[str]] = {
    "points": "per_point", "labels": "per_label", "tracks": "per_track",
    "mesh": "per_object", "voxels": None,
}


# ── field specs (drive the dialog AND the spread-to-other-tabs button) ──────────
@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: str                    # bool | int | float | choice | color
    label: str
    tip: str = ""
    lo: float = 0.0
    hi: float = 100.0
    step: float = 1.0
    decimals: int = 1
    choices: Tuple[Tuple[str, str], ...] = ()
    role: Optional[str] = None                        # cross-tab spread role
    enable_if: Optional[Tuple[str, Tuple[Any, ...]]] = None
    unit: str = ""


_SAT = FieldSpec("sat", "int", "Palette saturation", role="sat", lo=40, hi=255,
                 tip="Saturation of the auto-generated per-item colours (0–255)")
_VAL = FieldSpec("val", "int", "Palette brightness", role="val", lo=40, hi=255,
                 tip="Brightness of the auto-generated per-item colours (0–255)")
_OPACITY = FieldSpec("opacity", "int", "Opacity", role="opacity", lo=5, hi=100,
                     unit="%", tip="Overall opacity of this overlay")

#: Colour-ramp choices, taken from the renderer's own table so the dialog can never offer a
#: map :func:`nodelab_v2.overlay_render.colormap_lut` does not have (the Voxels tab used to
#: offer "gray", which is not a key there and silently fell back to viridis).
_CMAP_CHOICES: Tuple[Tuple[str, str], ...] = tuple(
    (k, k) for k in __import__("nodelab_v2.overlay_render", fromlist=["COLORMAPS"]).COLORMAPS)

_POINT_SHAPES = (("star", "Golden star (8 arms, gradient)"),
                 ("cross", "Cross + (4 arms, gradient)"),
                 ("diag", "Diagonal × (4 arms, gradient)"),
                 ("circle", "Filled dot"),
                 ("ring", "Hollow ring"),
                 ("square", "Hollow square"))
_GRADIENT_SHAPES = ("star", "cross", "diag")

FIELDS: Dict[str, Tuple[FieldSpec, ...]] = {
    "points": (
        FieldSpec("layer", "layer", "Which layer",
                  tip="WHICH Point table to draw. Leave it on Auto and EVERY table on the "
                      "payload is drawn, each in its own colour — right for a single "
                      "detection, wrong once a node publishes a filtered view of another's "
                      "cloud: Voronoi Cells emits `<Output layer>_seeds` holding only the dots "
                      "that actually won a territory, and on Auto you see those AND the full "
                      "input, dropped dots included. The list is the Point tables on the "
                      "viewed payload"),
        FieldSpec("shape", "choice", "Marker", choices=_POINT_SHAPES,
                  tip="The glyph stamped on each detection. The gradient shapes put a "
                      "bright pixel on the exact position and step outward."),
        FieldSpec("spread", "int", "Arm spread", lo=1, hi=5, unit=" steps",
                  tip="How many pixel steps the arms reach in each direction (1–5). "
                      "For the ring/dot/square shapes this is the radius."),
        FieldSpec("unit_px", "float", "Step size", lo=1.0, hi=12.0, step=0.5,
                  decimals=1, unit=" px",
                  tip="Size of ONE arm step in SCREEN pixels — so the marker keeps its "
                      "size while you zoom."),
        FieldSpec("thickness", "float", "Line width", role="line_width",
                  lo=0.5, hi=6.0, step=0.1, decimals=1, unit=" px",
                  tip="Stroke width for the ring / square outlines (screen px)"),
        FieldSpec("gradient", "bool", "Brightness gradient",
                  enable_if=("shape", _GRADIENT_SHAPES),
                  tip="Dim each arm step as it moves away from the centre"),
        FieldSpec("center_boost", "bool", "Bright centre pixel",
                  enable_if=("shape", _GRADIENT_SHAPES),
                  tip="Brighten the single pixel sitting on the actual position"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("single", "One colour"),
                           ("per_z", "One colour per Z plane"),
                           ("per_point", "Every point a different colour"),
                           ("per_layer", "One colour for each whole layer")),
                  tip="`One colour per Z plane` colours each marker by the plane it sits on, "
                      "so a 3-D cloud reads as depth instead of a pile — neighbouring planes "
                      "are pushed far apart on the wheel. It only says anything with `Show "
                      "points from every Z` ON: with it off, every marker drawn IS on the "
                      "viewed plane, so they legitimately all share one colour. And note it "
                      "cannot rescue a crowded cloud — a few thousand markers overlap into a "
                      "wash however they are coloured; thin the detection instead. `Every "
                      "point a different colour` tells individual detections apart, and "
                      "follows the TRACK once a tracker has linked them, so a particle keeps "
                      "its colour across T. `One colour for each whole layer` gives every "
                      "point in a layer the SAME colour and differs only between layers — so "
                      "with a single Point layer (one detection node) it looks identical to "
                      "`One colour`, which is correct rather than broken"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The marker colour (default: gold)"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("z_project", "bool", "Show points from every Z",
                  tip="Also draw detections that sit on other Z planes, dimmed. Useful for a "
                      "SPARSE 3-D cloud; on a dense one every plane's markers overlap into a "
                      "wash that hides both the image and the per-point colours, so leave it "
                      "off and step through Z instead — the status line always reports how "
                      "many detections are on other planes"),
        FieldSpec("off_opacity", "int", "Off-plane opacity", lo=5, hi=100, unit="%",
                  enable_if=("z_project", (True,)),
                  tip="Opacity of the points that are NOT on the viewed plane"),
    ),
    "labels": (
        FieldSpec("layer", "layer", "Which layer",
                  tip="WHICH label raster to draw. A Dataset often carries several at once — "
                      "Voronoi Cells alone emits its territories, inherits the seed branch's "
                      "labels and copies the areas it clipped to, and two segmentations both "
                      "default to the name `labels`. On Auto the VIEWED NODE's own output is "
                      "drawn (you opened it to see what it made), falling back to whichever "
                      "raster holds the most regions when the node declares none. The list is "
                      "the label rasters actually on the viewed payload"),
        FieldSpec("style", "choice", "Style",
                  choices=(("outline", "Outline only"), ("fill", "Fill only"),
                           ("both", "Outline + fill")),
                  tip="Region outlines, a translucent fill, or both"),
        FieldSpec("width", "float", "Outline width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("outline", "both")),
                  tip="Contour width in SCREEN pixels — constant at any zoom"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_label", "Every label a different colour"),
                           ("single", "One colour")),
                  tip="A distinct colour per region, or one colour for all. Once a "
                      "tracker has linked the regions the colour follows the TRACK, so a "
                      "cell keeps its colour across T even though its label id is "
                      "re-issued every frame — and neighbouring objects are pushed apart "
                      "on the colour wheel so two close ones never read as one"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The outline / fill colour when not colouring per label"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("fill_opacity", "int", "Fill opacity", lo=0, hi=100, unit="%",
                  enable_if=("style", ("fill", "both")),
                  tip="Opacity of the region fill (the outline uses Opacity)"),
        FieldSpec("show_ids", "bool", "Label the ids", role="show_ids",
                  tip=f"Draw each region's integer id at its centroid. Regions off screen, "
                      f"or too small on screen to hold the number, are skipped; above "
                      f"{MAX_ID_LABELS} ids in view none are drawn — zoom in to read them"),
        FieldSpec("id_px", "int", "Id text size", lo=6, hi=24, unit=" px",
                  enable_if=("show_ids", (True,)),
                  tip="Font size of the id text in screen pixels"),
    ),
    "tracks": (
        FieldSpec("width", "float", "Line width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  tip="Trajectory line width in SCREEN pixels"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_track", "Every track a different colour"),
                           ("single", "One colour")),
                  tip="A distinct colour per track — kept identical across every T frame, "
                      "and the SAME colour its member regions are drawn in — or one "
                      "colour for all"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The trajectory colour when not colouring per track"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("vertex_px", "float", "Vertex dot", lo=0.0, hi=6.0, step=0.1,
                  decimals=1, unit=" px",
                  tip="Radius of the per-timepoint dots (0 hides them)"),
        FieldSpec("head_px", "float", "Current-T marker", lo=0.0, hi=14.0, step=0.5,
                  decimals=1, unit=" px",
                  tip="Radius of the enlarged dot on the vertex at the viewed T"),
        FieldSpec("trail", "choice", "Trail",
                  choices=(("all", "Whole trajectory"),
                           ("past", "Up to the viewed T"),
                           ("window", "A window of frames before T")),
                  tip="How much of each trajectory to draw relative to the viewed T"),
        FieldSpec("window", "int", "Trail length", lo=1, hi=50, unit=" frames",
                  enable_if=("trail", ("window",)),
                  tip="How many timepoints back the trail reaches"),
        FieldSpec("fade", "bool", "Fade older segments",
                  tip="Ramp the line's opacity so the newest segment is brightest"),
        FieldSpec("show_ids", "bool", "Label the track ids", role="show_ids",
                  tip="Draw each track's id next to its current position"),
        FieldSpec("id_px", "int", "Id text size", lo=6, hi=24, unit=" px",
                  enable_if=("show_ids", (True,)),
                  tip="Font size of the id text in screen pixels"),
    ),
    "voxels": (
        FieldSpec("style", "choice", "Style",
                  choices=(("heatmap", "Colour-mapped wash"), ("mask", "Flat mask"),
                           ("contour", "Iso-contours")),
                  tip="How a scalar Voxel layer will be drawn"),
        FieldSpec("colormap", "choice", "Colour map", role="colormap",
                  choices=_CMAP_CHOICES,
                  tip="The colour ramp for the wash. coolwarm is the DIVERGING one — the "
                      "only correct choice for a signed field like strain, where zero must "
                      "read as neutral and the two signs must be tellable apart"),
        FieldSpec("alpha_mode", "choice", "Transparency",
                  choices=(("ramp", "Ramp with value"), ("flat", "Flat"),
                           ("gated", "Hide below threshold")),
                  tip="How transparency follows the value. Ramp keeps the low end "
                      "see-through so the image shows where the field has nothing to "
                      "report; Flat is for a mask, whose value carries no magnitude; "
                      "Gated cuts hard at the threshold"),
        _OPACITY,
        FieldSpec("center_zero", "bool", "Centre on zero",
                  tip="Make the limits symmetric about zero so the midpoint colour lands "
                      "exactly on zero. Only meaningful — and only applied — for a "
                      "diverging colour map"),
        FieldSpec("clim_auto", "bool", "Auto limits",
                  tip="Take the limits from the data's own 1st/99th percentiles, so a few "
                      "extreme samples at a solver's boundary cannot compress every real "
                      "value into the middle of the ramp"),
        FieldSpec("clim_lo", "float", "Limit low", lo=-1e6, hi=1e6, step=0.01, decimals=3,
                  enable_if=("clim_auto", (False,)),
                  tip="Value mapped to the bottom of the colour ramp"),
        FieldSpec("clim_hi", "float", "Limit high", lo=-1e6, hi=1e6, step=0.01, decimals=3,
                  enable_if=("clim_auto", (False,)),
                  tip="Value mapped to the top of the colour ramp"),
        FieldSpec("threshold", "float", "Threshold", lo=-1e6, hi=1e6, step=0.01,
                  decimals=3, enable_if=("alpha_mode", ("gated",)),
                  tip="Value below which a voxel draws nothing at all, in the field's own "
                      "units"),
        FieldSpec("width", "float", "Contour width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("contour",)),
                  tip="Iso-contour line width in screen pixels"),
    ),
    "vectors": (
        FieldSpec("every", "int", "Draw every", lo=1, hi=64,
                  tip="Keep 1 vector in N. A dense field is unreadable at full density "
                      "and slow to pan; this is the first knob to reach for"),
        FieldSpec("scale", "float", "Arrow scale", lo=0.05, hi=100.0, step=0.05,
                  decimals=2, unit="x",
                  tip="Multiplies the displacement to get the drawn arrow length. Applied "
                      "to BOTH components, so an arrow always points the way the field "
                      "does — only its length is exaggerated"),
        FieldSpec("gate", "float", "Hide below", lo=0.0, hi=1e6, step=0.1, decimals=3,
                  tip="Vectors shorter than this are not drawn. Solver noise is mostly "
                      "tiny vectors and it dominates the arrow count, so raising this "
                      "usually makes the real field appear"),
        FieldSpec("color_mode", "choice", "Colour by",
                  choices=(("magnitude", "Magnitude (colour map)"),
                           ("flat", "One colour")),
                  tip="Colour each arrow by how long it is, or draw them all alike"),
        FieldSpec("colormap", "choice", "Colour map", role="colormap",
                  choices=_CMAP_CHOICES, enable_if=("color_mode", ("magnitude",)),
                  tip="The ramp magnitude is mapped through"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("flat",)),
                  tip="The single arrow colour"),
        FieldSpec("width", "float", "Line width", role="line_width", lo=0.4, hi=6.0,
                  step=0.1, decimals=1, unit=" px",
                  tip="Arrow stroke width in SCREEN pixels, so it stays this thick at any "
                      "zoom"),
        FieldSpec("head_px", "float", "Head size", lo=2.0, hi=20.0, step=0.5, decimals=1,
                  unit=" px", tip="Arrow head length in screen pixels"),
        _OPACITY,
    ),
    "diff": (
        FieldSpec("match", "choice", "Match by",
                  choices=(("distance", "Centroid distance"),
                           ("iou", "Region overlap (IoU)")),
                  tip="How an object here is paired with one in the reference. IoU asks "
                      "whether they claim the same EXTENT, which is what a segmentation "
                      "comparison is about — two masks can share a centroid and disagree "
                      "about half their area"),
        FieldSpec("max_dist", "float", "Max distance", lo=0.1, hi=1000.0, step=0.1,
                  decimals=2, unit=" um", enable_if=("match", ("distance",)),
                  tip="Nothing further apart than this is a match, however lonely the two "
                      "objects are"),
        FieldSpec("min_iou", "float", "Min IoU", lo=0.05, hi=1.0, step=0.05, decimals=2,
                  enable_if=("match", ("iou",)),
                  tip="Overlap fraction two regions must share to count as the same "
                      "object. 0.5 is the usual reporting threshold"),
        FieldSpec("radius_px", "float", "Marker size", lo=2.0, hi=30.0, step=0.5,
                  decimals=1, unit=" px", tip="Ring radius in screen pixels"),
        FieldSpec("width", "float", "Line width", role="line_width", lo=0.5, hi=6.0,
                  step=0.1, decimals=1, unit=" px", tip="Ring stroke width"),
        FieldSpec("show_links", "bool", "Link matched pairs",
                  tip="Draw a line from each matched object to its partner — how far the "
                      "two disagree about position, for the ones they agree exist"),
        _OPACITY,
    ),
    "mesh": (
        FieldSpec("style", "choice", "Style",
                  choices=(("wireframe", "Cross-section outline"),
                           ("surface", "Cross-section filled"),
                           ("points", "Vertices near this Z")),
                  tip="A 3-D surface has no single 2-D picture: the outline is where the "
                      "mesh crosses the viewed Z plane, the filled style shades that "
                      "cross-section's interior, and the vertex style stamps the mesh "
                      "vertices that sit near this plane."),
        FieldSpec("width", "float", "Line width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("wireframe", "surface")),
                  tip="Cross-section outline width in SCREEN pixels — constant at any zoom"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_object", "Every object a different colour"),
                           ("single", "One colour")),
                  tip="A distinct colour per mesh element id, or one colour for all"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The mesh colour when not colouring per object"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("fill_opacity", "int", "Fill opacity", lo=0, hi=100, unit="%",
                  enable_if=("style", ("surface",)),
                  tip="Opacity of the cross-section fill (the outline uses Opacity)"),
        FieldSpec("vertex_px", "float", "Vertex dot", lo=0.5, hi=8.0, step=0.1,
                  decimals=1, unit=" px", enable_if=("style", ("points",)),
                  tip="Radius of each vertex dot in screen pixels"),
        FieldSpec("near_z", "float", "Z tolerance", lo=0.0, hi=10.0, step=0.5,
                  decimals=1, unit=" planes", enable_if=("style", ("points",)),
                  tip="How far off the viewed Z a vertex may sit and still be drawn"),
    ),
}

#: What "Spread this tab's look to the others" copies — a *role*, not a key, because the
#: same idea wears a different name per domain (``points.thickness`` ↔ ``labels.width``).
SPREAD_ROLES: Tuple[str, ...] = ("opacity", "color", "color_mode", "line_width",
                                 "sat", "val", "show_ids")

SPREAD_ROLE_LABEL: Dict[str, str] = {
    "opacity": "opacity", "color": "single colour", "color_mode": "colour-by mode",
    "line_width": "line width", "sat": "palette saturation",
    "val": "palette brightness", "show_ids": "id labels",
}


def _spec(tab: str, key: str) -> Optional[FieldSpec]:
    for sp in FIELDS[tab]:
        if sp.key == key:
            return sp
    return None


def _role_key(tab: str, role: str) -> Optional[str]:
    for sp in FIELDS[tab]:
        if sp.role == role:
            return sp.key
    return None


def field_enabled(group: Any, spec: FieldSpec) -> bool:
    """Whether ``spec``'s control is live given the group's other values."""
    if spec.enable_if is None:
        return True
    other, allowed = spec.enable_if
    return getattr(group, other, None) in allowed


def spread_settings(settings: OverlaySettings, source: str) -> List[str]:
    """Copy ``source``'s look onto every other tab **where the role applies**.

    Returns one human-readable line per change (so the dialog can say exactly what it
    did, and say nothing when a role had nowhere to go). ``color_mode`` travels as the
    *notion* "one colour" vs "a colour per item" — each tab keeps its own spelling of the
    per-item value (``per_label`` / ``per_track`` / …).
    """
    src = settings.group(source)
    notes: List[str] = []
    for role in SPREAD_ROLES:
        skey = _role_key(source, role)
        if skey is None:
            continue
        sval = getattr(src, skey)
        for tab in TABS:
            if tab == source:
                continue
            tkey = _role_key(tab, role)
            if tkey is None:
                continue
            grp = settings.group(tab)
            if role == "color_mode":
                per_src, per_dst = PER_ITEM_MODE.get(source), PER_ITEM_MODE.get(tab)
                if per_dst is None:
                    continue
                want = per_dst if sval == per_src else "single"
            else:
                want = _coerce(getattr(grp, tkey), sval)
            if getattr(grp, tkey) != want:
                setattr(grp, tkey, want)
                notes.append(f"{TAB_BY_KEY[tab].title}: {SPREAD_ROLE_LABEL[role]} "
                             f"→ {want}")
    return notes


# ── persistence ─────────────────────────────────────────────────────────────────
FILE_SUFFIX = ".nd2overlay.json"
FILE_FILTER = f"Overlay settings (*{FILE_SUFFIX});;JSON (*.json);;All files (*)"

#: this machine's permanent overlay ("make default") — outside the repo
USER_FILE = Path.home() / ".nd2studios" / "overlays.json"
#: the project default, shipped inside the package so committing it shares the look
PROJECT_FILE = Path(__file__).resolve().parent / "overlay_defaults.json"
#: an explicit override (tests / a shared lab config)
ENV_VAR = "NODELAB_OVERLAYS"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("overlay settings must be a JSON object")
    return data


def write_json(path: Path, settings: OverlaySettings) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def default_sources() -> List[Path]:
    """The layers :func:`load_defaults` merges, in increasing precedence."""
    env = os.environ.get(ENV_VAR)
    if env:
        return [Path(env)]
    return [PROJECT_FILE, USER_FILE]


def load_defaults() -> Tuple[OverlaySettings, List[str]]:
    """Built-in defaults with each existing settings layer merged on top.

    Returns ``(settings, notes)``; ``notes`` names the files that were applied and any
    that failed to parse, so the UI can say where the current look came from instead of
    silently falling back.
    """
    settings = OverlaySettings()
    notes: List[str] = []
    for path in default_sources():
        try:
            if not path.is_file():
                continue
            settings.update_from_dict(read_json(path))
            notes.append(f"loaded {path}")
        except Exception as exc:                     # noqa: BLE001 — never block the GUI
            notes.append(f"ignored {path} ({exc})")
    return settings, notes


# ── the painter ─────────────────────────────────────────────────────────────────
@dataclass
class PointMark:
    """One point ready to draw: position in **displayed-plane** pixels, the id used for
    per-point colouring, its layer index, and whether it sits on the viewed Z.

    ``zplane`` is the point's ``z`` rounded to a plane index — the same rounding
    :attr:`on_plane` is decided by, so a mark that *is* on the viewed plane always carries
    that plane's number. It exists for the ``per_z`` colour mode, which is the useful one for
    a projected 3-D cloud: with every plane's markers drawn at once, colouring by depth is
    what turns an unreadable pile into something you can read the structure of.
    """
    y: float
    x: float
    key: int
    layer: int
    on_plane: bool = True
    zplane: int = 0


@dataclass
class TrackPath:
    """One trajectory: ``track_id``, its member positions in displayed-plane pixels, the
    index of the vertex at the viewed T (or ``None``), and each vertex's timepoint.

    ``color_key`` is the track's palette slot — the same slot its member regions paint
    with, so a trajectory and the cell it belongs to are one colour. ``None`` falls back
    to the track id.
    """
    track_id: int
    path: List[Tuple[float, float]]
    current: Optional[int]
    times: List[int]
    color_key: Optional[int] = None


@dataclass
class MeshSection:
    """One mesh element's intersection with the viewed Z plane, ready to draw.

    ``loops`` are polylines of ``(y, x)`` in full-resolution image pixels — closed where
    the cross-section closed, open where it did not (a mesh that is not watertight, or one
    clipped by the plane's edge). ``verts`` are the element's vertices near the plane, for
    the vertex style. ``object_id`` is the per-object colour key.
    """
    object_id: int
    loops: List[List[Tuple[float, float]]] = field(default_factory=list)
    verts: List[Tuple[float, float]] = field(default_factory=list)
    closed: List[bool] = field(default_factory=list)


#: Above this many crossing triangles one element's cross-section is decimated, so a pan
#: over a marching-cubes surface (which can put 10⁴ triangles on one plane) stays live.
MAX_SECTION_TRIANGLES = 40_000


def mesh_section(verts_zyx: np.ndarray, faces: np.ndarray, z: float
                 ) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
    """Cross-section of a triangle mesh at the plane ``z``, as chained ``(y, x)`` loops.

    Each triangle straddling the plane contributes exactly one segment (the two edges whose
    endpoints fall on opposite sides); the segments are then chained end-to-end into
    polylines. Vertices sitting *exactly* on the plane are nudged to one side, so a
    triangle can never yield zero or three crossings — the degenerate case that would
    otherwise drop or double a segment.

    Returns ``(loops, closed_flags)``.
    """
    v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0 or len(v) == 0:
        return [], []
    tol = 1e-9
    d = v[:, 0] - float(z)
    d = np.where(np.abs(d) < tol, tol, d)          # never let a vertex sit ON the plane
    dd = d[f]
    pos = dd > 0
    n_pos = pos.sum(axis=1)
    hit = np.flatnonzero((n_pos == 1) | (n_pos == 2))
    if hit.size == 0:
        return [], []
    if hit.size > MAX_SECTION_TRIANGLES:
        hit = hit[::int(np.ceil(hit.size / MAX_SECTION_TRIANGLES))]
    fh, ddh = f[hit], dd[hit]
    P = np.zeros((len(fh), 3, 2), dtype=float)
    SC = np.zeros((len(fh), 3), dtype=bool)
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        da, db = ddh[:, i], ddh[:, j]
        sc = (da > 0) != (db > 0)
        tt = np.where(sc, da / np.where(da == db, 1.0, da - db), 0.0)
        a, b = v[fh[:, i]][:, 1:], v[fh[:, j]][:, 1:]      # (y, x) only
        P[:, k, :] = a + tt[:, None] * (b - a)
        SC[:, k] = sc
    keep = SC.sum(axis=1) == 2
    segs = [(tuple(pair[0]), tuple(pair[1]))
            for pair in (P[i][SC[i]] for i in np.flatnonzero(keep))]
    return _chain_segments(segs)


def _chain_segments(segs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
                    quant: float = 1e-6
                    ) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
    """Chain unordered segments into polylines by matching endpoints.

    Endpoints are matched on a quantized key rather than exact equality: the two triangles
    sharing an edge compute the same crossing point from the same two vertices, but not
    necessarily in the same operand order, so the results can differ in the last bit.
    """
    if not segs:
        return [], []

    def key(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(pt[0] / quant)), int(round(pt[1] / quant)))

    ends: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append(i)
        ends.setdefault(key(b), []).append(i)
    used = [False] * len(segs)
    loops: List[List[Tuple[float, float]]] = []
    closed: List[bool] = []
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        a, b = segs[start]
        path = [a, b]
        # walk forward from b, then (if it did not close) backward from a
        for direction in (0, 1):
            if direction == 1:
                if key(path[0]) == key(path[-1]):
                    break
                path.reverse()
            while True:
                nxt = None
                for i in ends.get(key(path[-1]), ()):
                    if not used[i]:
                        nxt = i
                        break
                if nxt is None:
                    break
                used[nxt] = True
                p, q = segs[nxt]
                path.append(q if key(p) == key(path[-1]) else p)
                if key(path[0]) == key(path[-1]):
                    break
        loops.append(path)
        closed.append(key(path[0]) == key(path[-1]) and len(path) > 3)
    return loops, closed


@dataclass
class OverlayFrame:
    """Everything the renderer needs about *this* frame, in one object.

    ``map_pt`` maps **displayed-plane** pixel coordinates to widget pixels (the backend's
    ``plane_to_widget``); ``sy``/``sx`` scale *structure* coordinates (full-resolution
    image pixels, as the axes report them) into displayed-plane pixels, which is how a
    decimated display plane still lands the geometry in the right place.
    """
    map_pt: Callable[[float, float], QPointF]
    plane_wh: Tuple[int, int]                       # displayed plane (w, h)
    sy: float = 1.0
    sx: float = 1.0
    points: Sequence[PointMark] = ()
    tracks: Sequence[TrackPath] = ()
    mesh: Sequence[MeshSection] = ()
    label_plane: Optional[np.ndarray] = None
    #: ``label id → palette slot`` for ``label_plane`` (see :func:`slot_lut`). This is what
    #: makes a tracked region keep one colour while its id is re-issued every frame;
    #: ``None`` colours by the raw id, the behaviour with no tracking in the graph.
    label_keys: Optional[np.ndarray] = None
    #: A Voxel-domain SCALAR field for this plane, at the displayed plane's resolution —
    #: strain, a distance transform, a density, a probability, a mask. ``None`` when the
    #: viewed node carries no scalar layer, which is every node that is not producing one.
    scalar: Optional[np.ndarray] = None
    #: Vectors for this plane, in full-resolution image pixels: ``(y, x, u, v)`` columns.
    #: ``u``/``v`` are the displacement's y/x components in the SAME units as y/x.
    vectors: Optional[np.ndarray] = None
    #: The two object sets the comparison overlay judges — ``(N, 2)`` ``y,x`` each, in
    #: full-resolution image pixels. ``diff_a`` is the viewed node's own set and ``diff_b``
    #: the reference it is being scored against.
    diff_a: Optional[np.ndarray] = None
    diff_b: Optional[np.ndarray] = None
    current_t: int = 0


#: Above this many boundary segments (after collinear runs are merged) the label outline is
#: uniformly decimated so a pan stays interactive. Applied once, when the contour is built
#: for a plane — not per paint. Disclosed in the Labels tab tooltip and warned about once
#: on stderr.
MAX_OUTLINE_SEGMENTS = 250_000

#: Above this many segments the contour is stroked *aliased*. At that density the lines are
#: a pixel apart anyway, so antialiasing buys nothing and costs the whole frame.
OUTLINE_AA_LIMIT = 60_000


def _merge_runs(fixed: np.ndarray, key: np.ndarray,
                run: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse unit boundary segments into maximal straight runs.

    ``fixed`` is the coordinate the run lies on (a column for the vertical segments, a row
    for the horizontal ones), ``key`` the colour key, ``run`` the integer coordinate the
    segment steps along. Returns ``(fixed, key, run_start, run_end)`` per maximal group of
    equal ``(fixed, key)`` with consecutive ``run`` — so a 40 px straight edge becomes one
    segment instead of 40, which is a 2–5× cut on real segmentations and pure profit for
    both the cache build and Qt's stroker.
    """
    if run.size == 0:
        return run, run, run, run
    order = np.lexsort((run, key, fixed))
    fixed, key, run = fixed[order], key[order], run[order]
    brk = np.empty(run.shape, dtype=bool)
    brk[0] = True
    brk[1:] = ((fixed[1:] != fixed[:-1]) | (key[1:] != key[:-1])
               | (run[1:] != run[:-1] + 1))
    starts = np.nonzero(brk)[0]
    ends = np.append(starts[1:], run.size) - 1
    return fixed[starts], key[starts], run[starts], run[ends]

_DIRS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "cross": ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)),
    "diag": ((-0.70710678, -0.70710678), (-0.70710678, 0.70710678),
             (0.70710678, -0.70710678), (0.70710678, 0.70710678)),
}
_DIRS["star"] = _DIRS["cross"] + _DIRS["diag"]


class OverlayRenderer:
    """Paints the overlays in widget space. Owns three caches — stamped point glyphs, the
    label fill image and the label contour paths — all keyed so a pure pan/zoom never
    rebuilds them."""

    def __init__(self) -> None:
        self._glyphs: Dict[tuple, QPixmap] = {}
        self._id_glyphs: Dict[Tuple[str, int], QPainterPath] = {}
        self._id_fonts: Dict[Tuple[str, int], Tuple[QFont, QFontMetricsF]] = {}
        self._fill_key: Optional[tuple] = None
        self._fill_img: Optional[QImage] = None
        self._fill_buf: Optional[np.ndarray] = None      # keeps the QImage's memory alive
        self._fill_lab: Optional[np.ndarray] = None      # pins id(lab) against reuse
        self._fill_keys: Optional[np.ndarray] = None     # pins id(keys) against reuse
        self._out_key: Optional[tuple] = None
        self._out_paths: List[Tuple[int, QPainterPath]] = []
        self._out_segs = 0
        self._out_lab: Optional[np.ndarray] = None       # pins id(lab) against reuse
        self._pen_key: Optional[tuple] = None
        self._out_pens: List[QPen] = []                  # aligned with _out_paths
        self._pen_keys: Optional[np.ndarray] = None      # pins id(keys) against reuse
        self._cent_key: Optional[tuple] = None
        self._cent: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0))
        self._cent_lab: Optional[np.ndarray] = None
        self._warned_truncate = False
        self._warned_ids = False

    def invalidate(self) -> None:
        """Drop the caches (a settings change or a new node)."""
        self._glyphs.clear()
        self._id_glyphs.clear()
        self._id_fonts.clear()
        self._fill_key = None
        self._fill_img = None
        self._fill_buf = None
        self._fill_lab = None
        self._fill_keys = None
        self._out_key = None
        self._out_paths = []
        self._out_lab = None
        self._pen_key = None
        self._out_pens = []
        self._pen_keys = None
        self._cent_key = None
        self._cent_lab = None

    # ── entry point ────────────────────────────────────────────────────────────
    def paint(self, p: QPainter, s: OverlaySettings, frame: OverlayFrame) -> None:
        """Draw every enabled overlay, bottom to top: labels, mesh, points, tracks.

        Mesh sits above the label fill (a cross-section outline is meant to be read
        *against* the region it bounds) and below the point/track markers, which are the
        smallest marks and must never be covered.
        """
        # The scalar wash goes UNDER everything: it is a continuous field covering the
        # frame, so anything drawn beneath it would be tinted by it, and the marks above it
        # (an outline, an arrow, an id) are exactly what you read the field against.
        if s.voxels.enabled and frame.scalar is not None:
            self._paint_scalar(p, s.voxels, frame)
        if s.vectors.enabled and frame.vectors is not None and len(frame.vectors):
            self._paint_vectors(p, s.vectors, frame)
        if s.labels.enabled and frame.label_plane is not None:
            self._paint_labels(p, s.labels, frame)
        if s.mesh.enabled and frame.mesh:
            self._paint_mesh(p, s.mesh, frame)
        if s.points.enabled and frame.points:
            self._paint_points(p, s.points, frame)
        if s.tracks.enabled and frame.tracks:
            self._paint_tracks(p, s.tracks, frame)
        # the comparison verdict goes on TOP: it is the thing being judged, and a ring
        # hidden under a label fill is a verdict you cannot read
        if s.diff.enabled and frame.diff_a is not None:
            self._paint_diff(p, s.diff, frame)

    # ── scalar field (V2.19 — VoxelsOverlay, declared since v2.00 and now drawn) ──
    def _paint_scalar(self, p: QPainter, s: "VoxelsOverlay",
                      frame: OverlayFrame) -> None:
        """Colour-map the plane's scalar layer and draw it over the image.

        Drawn as ONE RGBA image through the same plane→widget mapping the other overlays
        use, rather than per-pixel: the field is already at the displayed plane's
        resolution, so this is a single scaled blit and it stays smooth under a pan.
        Transparency comes from :func:`nodelab_v2.overlay_render.scalar_rgba` — the field's
        own values decide where the image shows through, which is what makes a heatmap
        readable instead of a coloured sheet over the data."""
        from nodelab_v2.overlay_render import scalar_rgba
        arr = np.asarray(frame.scalar)
        if arr.ndim != 2 or arr.size == 0:
            return
        rgba = scalar_rgba(
            arr, cmap=s.colormap,
            lo=(s.clim_lo if s.clim_auto is False else None),
            hi=(s.clim_hi if s.clim_auto is False else None),
            center_zero=bool(s.center_zero),
            alpha_mode=("flat" if s.style == "mask" else s.alpha_mode),
            threshold=float(s.threshold), opacity=max(0.0, min(1.0, s.opacity / 100.0)))
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        # the two corners of the plane map the image into widget space, so the wash pans
        # and zooms locked to the picture underneath it
        tl = frame.map_pt(0.0, 0.0)
        br = frame.map_pt(float(frame.plane_wh[0]), float(frame.plane_wh[1]))
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, s.style != "mask")
        p.drawImage(QRectF(tl, br), img)
        p.restore()

    # ── vector field (V2.19) ──────────────────────────────────────────────────
    def _paint_vectors(self, p: QPainter, s: "VectorsOverlay",
                       frame: OverlayFrame) -> None:
        """Draw a displacement field as decimated arrows in WIDGET space.

        Screen-space like every other overlay here: an arrow keeps its stroke weight and
        head size when you zoom, so zooming inspects the field instead of magnifying the
        drawing of it. Only the tail and head POSITIONS come from the data."""
        from nodelab_v2.overlay_render import colormap_lut, quiver_arrows
        v = np.asarray(frame.vectors, dtype=float)
        if v.ndim != 2 or v.shape[1] < 4:
            return
        tails, heads, mag = quiver_arrows(
            v[:, 0], v[:, 1], v[:, 2], v[:, 3],
            every=max(1, int(s.every)), scale=float(s.scale),
            gate=float(s.gate))
        if not len(tails):
            return
        lut = colormap_lut(s.colormap)
        hi = float(mag.max()) if (s.color_mode == "magnitude" and mag.size) else 0.0
        flat = qcolor(s.color, s.opacity)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        head_px = max(2.0, float(s.head_px))
        for k in range(len(tails)):
            a = frame.map_pt(tails[k][1] * frame.sx, tails[k][0] * frame.sy)
            b = frame.map_pt(heads[k][1] * frame.sx, heads[k][0] * frame.sy)
            if s.color_mode == "magnitude" and hi > 0:
                r, g, bl = lut[int(min(255, max(0, mag[k] / hi * 255)))]
                col = QColor(int(r), int(g), int(bl), _alpha(s.opacity))
            else:
                col = flat
            p.setPen(QPen(col, float(s.width)))
            p.drawLine(a, b)
            # the head is drawn from the SCREEN direction, so a very short arrow still
            # points somewhere legible instead of collapsing into its own barb
            dy, dx = b.y() - a.y(), b.x() - a.x()
            ln = (dy * dy + dx * dx) ** 0.5
            if ln < 1e-6:
                continue
            uy, ux = dy / ln, dx / ln
            for sign in (-1.0, 1.0):
                p.drawLine(b, QPointF(b.x() - head_px * (ux + sign * 0.6 * uy),
                                      b.y() - head_px * (uy - sign * 0.6 * ux)))
        p.restore()

    # ── comparison (V2.19) ────────────────────────────────────────────────────
    def _paint_diff(self, p: QPainter, s: "DiffOverlay", frame: OverlayFrame) -> None:
        """Match the two sets and draw matched / only-A / only-B.

        Only the DISTANCE matcher runs here: an IoU comparison needs two label rasters, and
        the frame carries centroids. Selecting `iou` with only centroids available draws
        nothing rather than quietly scoring a different question than the one asked."""
        from nodelab_v2.overlay_render import DIFF_COLORS, match_points
        a = frame.diff_a
        b = frame.diff_b
        if a is None or b is None or s.match != "distance":
            return
        pairs, only_a, only_b = match_points(a, b, max_dist=float(s.max_dist))
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        r = float(s.radius_px)

        def ring(pt, key):
            col = QColor(*DIFF_COLORS[key], _alpha(s.opacity))
            p.setPen(QPen(col, float(s.width)))
            p.setBrush(Qt.NoBrush)
            w = frame.map_pt(pt[1] * frame.sx, pt[0] * frame.sy)
            p.drawEllipse(w, r, r)
            return w

        for i, j in pairs:
            wa = ring(a[i], "matched")
            if s.show_links:
                wb = frame.map_pt(b[j][1] * frame.sx, b[j][0] * frame.sy)
                p.drawLine(wa, wb)
        for i in only_a:
            ring(a[i], "only_a")
        for j in only_b:
            ring(b[j], "only_b")
        p.restore()

    # ── mesh ──────────────────────────────────────────────────────────────────
    def mesh_color(self, s: MeshOverlay, object_id: int,
                   opacity: Optional[int] = None) -> QColor:
        op = s.opacity if opacity is None else opacity
        if s.color_mode == "per_object":
            return distinct_color(int(object_id), s.sat, s.val, op)
        return qcolor(s.color, op)

    def _paint_mesh(self, p: QPainter, s: MeshOverlay, frame: OverlayFrame) -> None:
        """Draw each element's Z cross-section (outline / filled) or its near-plane
        vertices. Only a CLOSED cross-section is filled — filling an open polyline would
        invent an edge that the mesh does not have."""
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        for sec in frame.mesh:
            col = self.mesh_color(s, sec.object_id)
            if s.style == "points":
                p.setPen(QPen(col, 1.0))
                p.setBrush(col)
                rad = float(s.vertex_px)
                if rad > 0:
                    for y, x in sec.verts:
                        p.drawEllipse(frame.map_pt(x * frame.sx, y * frame.sy), rad, rad)
                continue
            polys = [QPolygonF([frame.map_pt(x * frame.sx, y * frame.sy)
                                for y, x in loop]) for loop in sec.loops]
            if s.style == "surface" and s.fill_opacity > 0:
                fill = self.mesh_color(s, sec.object_id, s.fill_opacity)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                for poly, is_closed in zip(polys, sec.closed):
                    if is_closed:
                        p.drawPolygon(poly)
            if s.width > 0:
                pen = QPen(col, float(s.width))
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                for poly in polys:
                    p.drawPolyline(poly)
        p.restore()

    # ── labels ────────────────────────────────────────────────────────────────
    def _label_color(self, s: LabelsOverlay, value: int, opacity: int,
                     keys: Optional[np.ndarray] = None) -> QColor:
        if s.color_mode == "per_label":
            return distinct_color(slot_of(keys, value), s.sat, s.val, opacity)
        return qcolor(s.color, opacity)

    def _paint_labels(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame) -> None:
        lab = frame.label_plane
        if lab is None or lab.size == 0:
            return
        keys = frame.label_keys
        W, H = frame.plane_wh
        lh, lw = lab.shape[:2]
        lsy, lsx = H / max(1, lh), W / max(1, lw)

        if s.style in ("fill", "both") and s.fill_opacity > 0:
            img = self._fill_image(s, lab, keys)
            if img is not None:
                tl = frame.map_pt(0.0, 0.0)
                br = frame.map_pt(float(W), float(H))
                p.save()
                # nearest-neighbour: a zoomed-in label mask must stay pixel-crisp, and
                # interpolating distinct label colours invents regions that aren't there.
                p.setRenderHint(QPainter.SmoothPixmapTransform, False)
                p.drawImage(QRectF(tl, br), img)
                p.restore()

        if s.style in ("outline", "both") and s.width > 0:
            self._paint_label_outline(p, s, frame, lab, lsy, lsx, keys)

        if s.show_ids:
            self._paint_label_ids(p, s, frame, lab, lsy, lsx, keys)

    def _fill_image(self, s: LabelsOverlay, lab: np.ndarray,
                    keys: Optional[np.ndarray] = None) -> Optional[QImage]:
        """A cached RGBA image of the label plane — one colour per region (per *object*,
        when ``keys`` maps the plane's ids onto palette slots)."""
        # id(lab) is safe as a key only because the cache holds a reference to the array
        # (below): a freed plane's id could otherwise be handed to a different array.
        key = (id(lab), lab.shape, id(keys), s.color_mode, s.color, s.sat, s.val,
               s.fill_opacity)
        if key == self._fill_key and self._fill_img is not None:
            return self._fill_img
        top = int(lab.max()) if lab.size else 0
        if top <= 0 or top > 1 << 20:            # nothing to fill / absurd id space
            return None
        alpha = _alpha(s.fill_opacity)
        lut = np.zeros((top + 1, 4), dtype=np.uint8)
        if s.color_mode == "per_label":
            idx = np.arange(1, top + 1, dtype=np.int64)
            if keys is not None:
                # the ids this plane carries, through the palette; ids past the LUT keep
                # their own slot (a region the palette never saw)
                covered = idx < len(keys)
                idx = np.where(covered, np.asarray(keys)[np.clip(idx, 0, len(keys) - 1)],
                               idx)
            lut[1:, :3] = _hues_to_rgb(_hues(idx),
                                       max(0.0, min(1.0, s.sat / 255.0)),
                                       max(0.0, min(1.0, s.val / 255.0)))
        else:
            col = qcolor(s.color)
            lut[1:, :3] = np.array([col.red(), col.green(), col.blue()], dtype=np.uint8)
        lut[1:, 3] = alpha
        rgba = np.ascontiguousarray(np.take(lut, np.clip(lab, 0, top), axis=0))
        h, w = rgba.shape[:2]
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self._fill_key, self._fill_img, self._fill_buf = key, img, rgba
        self._fill_lab, self._fill_keys = lab, keys   # pin both ids against reuse
        return img

    def _outline_paths(self, s: LabelsOverlay,
                       lab: np.ndarray) -> List[Tuple[int, QPainterPath]]:
        """Cached region boundaries as ``(colour key, path)`` in **label-pixel** space.

        Segments, not pixels: the old overlay stamped one dot per boundary pixel, which
        zoomed in turns into a dotted line with gaps as wide as the magnification. A
        vertical boundary between ``lab[i, j]`` and ``lab[i, j+1]`` is the unit segment
        from ``(j+1, i)`` to ``(j+1, i+1)`` in label-pixel space, so the contour stays a
        continuous line at any zoom.

        The paths live in label-pixel space precisely so that pan and zoom are a *painter
        transform* at draw time (see :meth:`_paint_label_outline`) rather than a rebuild.
        Building them per paint — 2 Python calls into ``map_pt`` and one ``QLineF`` per
        segment, plus one full-array mask scan per label — was what made a drag with
        outlines on feel like treacle while the (cached, single-``drawImage``) fill stayed
        smooth.
        """
        key = (id(lab), lab.shape, s.color_mode)
        if key == self._out_key:
            return self._out_paths
        v_i, v_j = np.nonzero(lab[:, :-1] != lab[:, 1:])
        h_i, h_j = np.nonzero(lab[:-1, :] != lab[1:, :])
        # colour key = the larger of the two sides, so a label/background edge takes the
        # label's colour and a label/label edge is drawn once, deterministically.
        v_key = np.maximum(lab[v_i, v_j], lab[v_i, v_j + 1]).astype(np.int64)
        h_key = np.maximum(lab[h_i, h_j], lab[h_i + 1, h_j]).astype(np.int64)
        v_keep, h_keep = v_key > 0, h_key > 0
        # merge collinear runs: vertical segments share a column, horizontal ones a row
        v_x, v_key, v_y0, v_y1 = _merge_runs(v_j[v_keep], v_key[v_keep], v_i[v_keep])
        h_y, h_key, h_x0, h_x1 = _merge_runs(h_i[h_keep], h_key[h_keep], h_j[h_keep])
        # (x0, y0, x1, y1) in label-pixel space
        v_seg = np.stack([v_x + 1.0, v_y0 * 1.0, v_x + 1.0, v_y1 + 1.0], axis=1)
        h_seg = np.stack([h_x0 * 1.0, h_y + 1.0, h_x1 + 1.0, h_y + 1.0], axis=1)
        segs = np.concatenate([v_seg, h_seg], axis=0)
        keys = np.concatenate([v_key, h_key], axis=0)
        if segs.shape[0] > MAX_OUTLINE_SEGMENTS:
            stride = int(np.ceil(segs.shape[0] / MAX_OUTLINE_SEGMENTS))
            segs, keys = segs[::stride], keys[::stride]
            if not self._warned_truncate:
                self._warned_truncate = True
                print(f"[overlays] label outline decimated 1/{stride} "
                      f"(>{MAX_OUTLINE_SEGMENTS} boundary segments)",
                      file=sys.stderr, flush=True)
        paths: List[Tuple[int, QPainterPath]] = []
        if segs.shape[0]:
            if s.color_mode != "per_label":
                groups = [(0, segs)]
            else:
                order = np.argsort(keys, kind="stable")
                keys, segs = keys[order], segs[order]
                cuts = np.nonzero(np.diff(keys))[0] + 1
                groups = [(int(ks[0]), gs) for ks, gs in
                          zip(np.split(keys, cuts), np.split(segs, cuts))]
            for value, group in groups:
                path = QPainterPath()
                for x0, y0, x1, y1 in group.tolist():
                    path.moveTo(x0, y0)
                    path.lineTo(x1, y1)
                paths.append((value, path))
        self._out_key, self._out_paths, self._out_lab = key, paths, lab
        self._out_segs = int(segs.shape[0])
        return paths

    def _paint_label_outline(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame,
                             lab: np.ndarray, lsy: float, lsx: float,
                             keys: Optional[np.ndarray] = None) -> None:
        """Stroke the cached contours through the pan/zoom transform.

        ``map_pt`` is affine on both backends (a scale plus a translation), so probing it
        at three points recovers the whole mapping and Qt applies it to every segment in
        C++. The pen is **cosmetic**, which is what keeps ``width`` a screen size: a
        cosmetic width is in device pixels and ignores the transform, so it is scaled by
        the device pixel ratio here to stay the same thickness as a plain pen would be.
        """
        paths = self._outline_paths(s, lab)
        if not paths:
            return
        dpr = float(p.device().devicePixelRatio() or 1.0)
        pens = self._outline_pens(s, dpr, keys)
        mp = frame.map_pt
        o, ex, ey = mp(0.0, 0.0), mp(1.0, 0.0), mp(0.0, 1.0)
        plane = QTransform(ex.x() - o.x(), ex.y() - o.y(),
                           ey.x() - o.x(), ey.y() - o.y(), o.x(), o.y())
        full = QTransform(lsx, 0.0, 0.0, lsy, 0.0, 0.0) * plane
        p.save()
        p.setRenderHint(QPainter.Antialiasing, self._out_segs <= OUTLINE_AA_LIMIT)
        p.setTransform(full, True)
        p.setBrush(Qt.NoBrush)
        for (_value, path), pen in zip(paths, pens):
            p.setPen(pen)
            p.drawPath(path)
        p.restore()

    def _outline_pens(self, s: LabelsOverlay, dpr: float,
                      keys: Optional[np.ndarray] = None) -> List[QPen]:
        """One cached cosmetic pen per contour group, aligned with :meth:`_outline_paths`.

        Cached because ``per_label`` colouring needs a colour per group and building those
        per paint — hundreds of QPens, and before the memo a numpy HSV conversion each —
        cost more than the stroking itself.
        """
        key = (self._out_key, id(keys), s.color_mode, s.color, s.sat, s.val, s.opacity,
               round(float(s.width), 3), round(dpr, 3))
        if key == self._pen_key:
            return self._out_pens
        width = max(1.0, float(s.width) * dpr)
        pens: List[QPen] = []
        for value, _path in self._out_paths:
            pen = QPen(self._label_color(s, value, s.opacity, keys), width)
            pen.setCapStyle(Qt.FlatCap)
            pen.setCosmetic(True)             # width in device px → a screen-space size
            pens.append(pen)
        self._pen_key, self._out_pens = key, pens
        self._pen_keys = keys                    # pins id(keys) against reuse
        return pens

    def _label_centroids(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
        """Cached ``(values, cx, cy, area)`` for the regions actually present in ``lab``,
        in label-pixel space. Only the ids that exist — a plane holding one region with id
        9000 yields one centroid, not 9000 empty slots. ``area`` is the region's pixel
        count, which is what tells the painter whether an id can fit inside it."""
        key = (id(lab), lab.shape)
        if key == self._cent_key:
            return self._cent
        top = int(lab.max()) if lab.size else 0
        z = np.zeros(0)
        empty = (np.zeros(0, dtype=np.int64), z, z, z)
        if top <= 0 or top > 1 << 22:           # nothing to label / absurd id space
            self._cent_key, self._cent, self._cent_lab = key, empty, lab
            return empty
        flat = lab.ravel()
        h, w = lab.shape[:2]
        counts = np.bincount(flat, minlength=top + 1)
        ys = np.repeat(np.arange(h, dtype=np.float64), w)
        xs = np.tile(np.arange(w, dtype=np.float64), h)
        sy = np.bincount(flat, weights=ys, minlength=top + 1)
        sx = np.bincount(flat, weights=xs, minlength=top + 1)
        vals = np.nonzero(counts)[0]
        vals = vals[vals > 0]
        n = counts[vals].astype(float)
        out = (vals.astype(np.int64), sx[vals] / n, sy[vals] / n, n)
        self._cent_key, self._cent, self._cent_lab = key, out, lab
        return out

    def _paint_label_ids(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame,
                         lab: np.ndarray, lsy: float, lsx: float,
                         keys: Optional[np.ndarray] = None) -> None:
        """Stamp each region's id at its centroid, centred on it.

        Three rules, all about what is *readable*. It does not cap on the largest id —
        that is what used to make the whole feature silently draw nothing on any real
        segmentation, where ids run into the thousands. It skips regions off screen and
        regions too small on screen to hold their own number, which is what turns a
        zoomed-out plane from a mush of overlapping digits into the handful you can
        actually read. Only then does :data:`MAX_ID_LABELS` apply. Zooming in brings more
        ids in, as their regions grow past the text.

        The rect used for all of that is the **viewport**, never
        ``QPainter.clipBoundingRect()``: the CPU backend repaints in dirty patches, and
        culling to the patch made the ids that got drawn depend on which band of the widget
        Qt happened to be refreshing.
        """
        vals, cxs, cys, areas = self._label_centroids(lab)
        if vals.size == 0:
            return
        mp = frame.map_pt
        o, ex, ey = mp(0.0, 0.0), mp(1.0, 0.0), mp(0.0, 1.0)
        # affine map (see _paint_label_outline), applied to every centroid at once
        wx = o.x() + (cxs * lsx) * (ex.x() - o.x()) + (cys * lsy) * (ey.x() - o.x())
        wy = o.y() + (cxs * lsx) * (ex.y() - o.y()) + (cys * lsy) * (ey.y() - o.y())
        view = QRectF(p.viewport())
        if not view.isValid() or view.isEmpty():
            view = p.clipBoundingRect()
        pad = 2.0 * float(s.id_px)
        vis = ((wx >= view.left() - pad) & (wx <= view.right() + pad)
               & (wy >= view.top() - pad) & (wy <= view.bottom() + pad))
        # a label-pixel covers this much screen area, so sqrt(area) is the region's
        # on-screen size — drop the ones that cannot hold their own digits
        px_w = float(np.hypot(ex.x() - o.x(), ex.y() - o.y())) * lsx
        px_h = float(np.hypot(ey.x() - o.x(), ey.y() - o.y())) * lsy
        vis &= np.sqrt(areas * px_w * px_h) >= float(max(1, int(s.id_px)))
        count = int(vis.sum())
        if count == 0:
            return
        if count > MAX_ID_LABELS:
            if not self._warned_ids:
                self._warned_ids = True
                print(f"[overlays] {count} label ids in view (>{MAX_ID_LABELS}) - "
                      f"not drawn; zoom in to read them", file=sys.stderr, flush=True)
            return
        # centred on the centroid — the id belongs to the region under it
        for value, x, y in zip(vals[vis].tolist(), wx[vis].tolist(), wy[vis].tolist()):
            self.draw_id(p, str(value), QPointF(x, y),
                         self._label_color(s, value, s.opacity, keys), s.id_px)

    # ── id text (ONE implementation, shared by every domain) ──────────────────
    def draw_id(self, p: QPainter, text: str, at: QPointF, col: QColor,
                px_size: int, *, center: bool = True) -> None:
        """Draw one id label at ``at`` in label/track colour ``col``.

        **Every overlay that draws an id must come through here** — labels, tracks, and
        anything added later. There is one implementation because getting small text
        readable over an arbitrary image took three specific decisions, and any overlay
        that reimplements them will get them wrong the same way this code did:

        * **Stroke the glyph outline, never stamp the text twice.** A second ``drawText``
          at a 1 px offset (the classic shadow/halo) floods the counters of 0, 8 and 9 at
          these sizes: a 3-digit id becomes a blob. The outline is stroked with a dark
          round-joined pen and then filled.
        * **Snap the origin to the device-pixel grid.** Anchors are computed from centroids
          and vertex positions, so they are fractional; a glyph on a fractional pixel
          rasterizes soft, and a screenful of them each soft in a different direction is
          what reads as *grain* instead of as numbers. Worst at fractional display scaling
          (125 % / 150 %), which is the common Windows case.
        * **Keep the hue, not the luminance** (:meth:`_id_ink`). A mid-dark label colour at
          11 px is illegible however it is outlined.

        ``center`` centres the text on ``at`` (a region's centroid); pass ``False`` to
        anchor the text's baseline-left there (a track id sitting beside its head).
        """
        size = max(1, int(px_size))
        font, fm = self._id_font(p, size)
        dpr = float(p.device().devicePixelRatio() or 1.0)
        grid = 1.0 / dpr if dpr > 0 else 1.0
        x = at.x() - (fm.horizontalAdvance(text) / 2.0 if center else 0.0)
        y = at.y() + ((fm.ascent() - fm.descent()) / 2.0 if center else 0.0)
        ink, edge = self._id_ink(col)
        path = self._id_path(text, font, size)
        pen = QPen(edge, max(1.5, size / 6.0))
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.translate(round(x / grid) * grid, round(y / grid) * grid)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.NoPen)
        p.setBrush(ink)
        p.drawPath(path)
        p.restore()

    def _id_font(self, p: QPainter, px_size: int) -> Tuple[QFont, QFontMetricsF]:
        """The id font at ``px_size`` plus its metrics, cached. Deliberately **not bold**:
        the stroke supplies the weight, and bold closes the counters at 11 px."""
        base = p.font()
        key = (base.family(), px_size)
        got = self._id_fonts.get(key)
        if got is None:
            font = QFont(base)
            font.setPixelSize(px_size)
            font.setBold(False)
            got = (font, QFontMetricsF(font))
            if len(self._id_fonts) > 64:
                self._id_fonts.clear()
            self._id_fonts[key] = got
        return got

    def _id_path(self, text: str, font: QFont, px_size: int) -> QPainterPath:
        """The glyph outline of ``text`` at the origin, cached per ``(text, size)`` — built
        once per distinct id, so a repaint only translates and fills it."""
        key = (text, px_size)
        path = self._id_glyphs.get(key)
        if path is None:
            if len(self._id_glyphs) > 4096:          # scrubbing a long movie of fresh ids
                self._id_glyphs.clear()
            path = QPainterPath()
            path.addText(0.0, 0.0, font, text)
            self._id_glyphs[key] = path
        return path

    @staticmethod
    def _id_ink(col: QColor) -> Tuple[QColor, QColor]:
        """``(ink, edge)`` for an id drawn in label colour ``col``: the hue lifted to a
        luminance that reads as text, and a dark edge to stroke it with.

        The id keeps its label's hue — that is the association with the region — but not its
        *luminance*: at 11 px a mid-dark hue outlined in white leaves nothing but thin
        coloured strokes, which is illegible on any background. Lifting the ink and stroking
        it dark is the map-label treatment, and it reads over both a bright field and a dark
        one. Both keep the label's alpha, so Opacity still fades the whole mark together.
        """
        lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
        ink = col if lum >= 170.0 else _mix(col, QColor(255, 255, 255),
                                            min(0.75, (170.0 - lum) / 220.0))
        edge = QColor(0, 0, 0)
        edge.setAlpha(max(0, min(255, int(col.alpha() * 0.85))))
        return ink, edge

    # ── points ────────────────────────────────────────────────────────────────
    def _point_color(self, s: PointsOverlay, mark: PointMark) -> QColor:
        opacity = s.opacity if mark.on_plane else min(s.opacity, s.off_opacity)
        if s.color_mode == "per_point":
            return distinct_color(mark.key, s.sat, s.val, opacity)
        if s.color_mode == "per_z":
            # golden-angle on the PLANE INDEX, so consecutive planes land far apart on the
            # wheel — the question this mode answers is "which plane is this marker on",
            # and neighbouring planes are exactly the ones that must not look alike
            return distinct_color(mark.zplane, s.sat, s.val, opacity)
        if s.color_mode == "per_layer":
            return distinct_color(mark.layer, s.sat, s.val, opacity)
        return qcolor(s.color, opacity)

    def _paint_points(self, p: QPainter, s: PointsOverlay,
                      frame: OverlayFrame) -> None:
        dpr = float(p.device().devicePixelRatio() or 1.0)
        for mark in frame.points:
            col = self._point_color(s, mark)
            pm, size = self.glyph(s, col, dpr)
            centre = frame.map_pt(mark.x * frame.sx, mark.y * frame.sy)
            p.drawPixmap(QPointF(centre.x() - size / 2.0, centre.y() - size / 2.0), pm)

    def glyph(self, s: PointsOverlay, col: QColor, dpr: float) -> Tuple[QPixmap, float]:
        """A cached marker pixmap for ``col`` plus its logical size.

        Pre-rendering the glyph once and stamping it is what keeps a 10 000-point overlay
        cheap, and it is also what makes the size zoom-invariant: the pixmap is built in
        screen pixels and never scaled.
        """
        radius = max(1.0, float(s.spread) * float(s.unit_px))
        pad = max(2.0, float(s.thickness) + 1.0)
        size = 2.0 * radius + 2.0 * pad
        key = (s.shape, s.spread, round(s.unit_px, 2), round(s.thickness, 2),
               s.gradient, s.center_boost, col.rgba(), round(dpr, 3))
        cached = self._glyphs.get(key)
        if cached is not None:
            return cached, size
        if len(self._glyphs) > 512:                 # a per-point palette is bounded by
            self._glyphs.clear()                    # 360 hues; this is the safety net
        px = max(1, int(round(size * dpr)))
        pm = QPixmap(px, px)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        gp = QPainter(pm)
        c = size / 2.0
        rays = _DIRS.get(s.shape)
        if rays is not None:
            unit = float(s.unit_px)
            # crisp square "pixels": antialiasing a 3 px block just blurs its edges
            gp.setRenderHint(QPainter.Antialiasing, False)
            gp.setPen(Qt.NoPen)
            for k in range(int(s.spread), 0, -1):
                frac = (1.0 - k / (s.spread + 1.0)) if s.gradient else 1.0
                step = QColor(col)
                step.setAlpha(max(0, min(255, int(round(col.alpha() * frac)))))
                gp.setBrush(step)
                for dy, dx in rays:
                    gp.drawRect(QRectF(c + dx * k * unit - unit / 2.0,
                                       c + dy * k * unit - unit / 2.0, unit, unit))
            centre = _mix(col, QColor(255, 255, 255), 0.5) if s.center_boost else col
            gp.setBrush(centre)
            gp.drawRect(QRectF(c - unit / 2.0, c - unit / 2.0, unit, unit))
        else:
            gp.setRenderHint(QPainter.Antialiasing, True)
            if s.shape == "circle":
                gp.setPen(Qt.NoPen)
                gp.setBrush(col)
                gp.drawEllipse(QPointF(c, c), radius * 0.5, radius * 0.5)
            elif s.shape == "square":
                gp.setPen(QPen(col, float(s.thickness)))
                gp.setBrush(Qt.NoBrush)
                gp.drawRect(QRectF(c - radius, c - radius, 2 * radius, 2 * radius))
            else:                                    # ring
                gp.setPen(QPen(col, float(s.thickness)))
                gp.setBrush(Qt.NoBrush)
                gp.drawEllipse(QPointF(c, c), radius, radius)
        gp.end()
        self._glyphs[key] = pm
        return pm, size

    # ── tracks ────────────────────────────────────────────────────────────────
    def track_color(self, s: TracksOverlay, track_id: int,
                    opacity: Optional[int] = None,
                    slot: Optional[int] = None) -> QColor:
        """The trajectory colour. ``slot`` is the track's palette slot — the one its member
        regions paint with — and defaults to the track id when there is no palette."""
        op = s.opacity if opacity is None else opacity
        if s.color_mode == "per_track":
            return distinct_color(int(track_id if slot is None else slot),
                                  s.sat, s.val, op)
        return qcolor(s.color, op)

    def _paint_tracks(self, p: QPainter, s: TracksOverlay,
                      frame: OverlayFrame) -> None:
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        for tr in frame.tracks:
            keep = self._trail_mask(s, tr, frame.current_t)
            verts = [frame.map_pt(x * frame.sx, y * frame.sy)
                     for i, (y, x) in enumerate(tr.path) if keep[i]]
            if not verts:
                continue
            col = self.track_color(s, tr.track_id, slot=tr.color_key)
            if len(verts) >= 2:
                if s.fade:
                    n = len(verts) - 1
                    for i in range(n):
                        frac = 0.25 + 0.75 * ((i + 1) / n)
                        seg = QColor(col)
                        seg.setAlpha(int(round(col.alpha() * frac)))
                        p.setPen(QPen(seg, float(s.width)))
                        p.drawLine(verts[i], verts[i + 1])
                else:
                    p.setPen(QPen(col, float(s.width)))
                    p.setBrush(Qt.NoBrush)
                    p.drawPolyline(QPolygonF(verts))
            # the current-T vertex keeps its identity in the ORIGINAL path, so recover
            # its index within the drawn subset
            cur_drawn = None
            if tr.current is not None and keep[tr.current]:
                cur_drawn = int(np.count_nonzero(keep[:tr.current]))
            p.setPen(QPen(col, 1.0))
            p.setBrush(col)
            for k, pt in enumerate(verts):
                rad = float(s.head_px) if k == cur_drawn else float(s.vertex_px)
                if rad > 0:
                    p.drawEllipse(pt, rad, rad)
            if s.show_ids:
                anchor = verts[cur_drawn if cur_drawn is not None else -1]
                off = QPointF(float(s.head_px) + 3.0, -float(s.head_px) - 2.0)
                # beside the head, not centred on it — the head marker is the position, and
                # the id must not cover it. Same renderer as the label ids: see draw_id.
                self.draw_id(p, str(tr.track_id), anchor + off, col, s.id_px,
                             center=False)
        p.restore()

    @staticmethod
    def _trail_mask(s: TracksOverlay, tr: TrackPath, current_t: int) -> np.ndarray:
        """Which vertices of ``tr`` the ``trail`` mode draws at the viewed T."""
        times = np.asarray(tr.times if tr.times else [0] * len(tr.path))
        if times.size != len(tr.path):
            return np.ones(len(tr.path), dtype=bool)
        if s.trail == "past":
            return times <= current_t
        if s.trail == "window":
            return (times <= current_t) & (times >= current_t - int(s.window))
        return np.ones(len(tr.path), dtype=bool)


# ── live preview (used by the dialog; proves the renderer, not a second look) ────
def render_preview(p: QPainter, rect: QRectF, s: OverlaySettings, tab: str,
                   renderer: Optional[OverlayRenderer] = None) -> None:
    """Draw a small sample of ``tab``'s overlay inside ``rect`` using the REAL renderer,
    over a synthetic checker so opacity is readable. Reserved tabs get a plain note."""
    p.save()
    p.setClipRect(rect)
    p.fillRect(rect, QColor(18, 21, 26))
    cell = 12.0                              # a checker, so opacity is readable
    shade = QColor(31, 36, 44)
    ny = int(rect.height() // cell) + 1
    nx = int(rect.width() // cell) + 1
    for iy in range(ny):
        for ix in range(nx):
            if (ix + iy) % 2 == 0:
                p.fillRect(QRectF(rect.left() + ix * cell, rect.top() + iy * cell,
                                  cell, cell), shade)
    info = TAB_BY_KEY[tab]
    if not info.implemented:
        p.setPen(QPen(QColor(150, 160, 175), 1.0))
        p.drawText(rect, int(Qt.AlignCenter | Qt.TextWordWrap),
                   f"{info.title} overlays are not drawn yet —\n"
                   "these settings are stored for when they land.")
        p.restore()
        return

    ren = renderer or OverlayRenderer()
    w, h = 40.0, 24.0                       # a tiny synthetic plane
    sx = rect.width() / w
    sy = rect.height() / h

    def map_pt(px: float, py: float) -> QPointF:
        return QPointF(rect.left() + px * sx, rect.top() + py * sy)

    frame = OverlayFrame(map_pt=map_pt, plane_wh=(int(w), int(h)), current_t=2)
    sub = OverlaySettings()
    sub.update_from_dict(s.to_dict())
    for other in TABS:                      # preview exactly one overlay at a time
        sub.group(other).enabled = (other == tab)
    if tab == "points":
        frame.points = [PointMark(6.0, 8.0, 1, 0), PointMark(12.0, 20.0, 2, 0),
                        PointMark(17.0, 32.0, 3, 1), PointMark(9.0, 30.0, 4, 1)]
    elif tab == "labels":
        lab = np.zeros((int(h), int(w)), dtype=np.int32)
        lab[4:12, 5:16] = 1
        lab[6:18, 18:27] = 2
        lab[3:9, 29:38] = 3
        frame.label_plane = lab
    elif tab == "tracks":
        frame.tracks = [
            TrackPath(1, [(6.0, 4.0), (8.0, 11.0), (10.0, 18.0), (13.0, 26.0),
                          (15.0, 34.0)], 2, [0, 1, 2, 3, 4]),
            TrackPath(2, [(18.0, 6.0), (16.0, 13.0), (15.0, 21.0), (12.0, 29.0)],
                      2, [0, 1, 2, 3]),
        ]
    elif tab == "mesh":
        # two closed cross-sections — a hexagon and an L, so the fill / outline / vertex
        # styles all show something honest (the L proves a concave section renders)
        hexa = [(6.0 + 4.0 * np.sin(a), 11.0 + 5.0 * np.cos(a))
                for a in np.linspace(0, 2 * np.pi, 7)]
        ell = [(4.0, 22.0), (4.0, 34.0), (9.0, 34.0), (9.0, 28.0),
               (18.0, 28.0), (18.0, 22.0), (4.0, 22.0)]
        frame.mesh = [MeshSection(1, [hexa], hexa[:-1], [True]),
                      MeshSection(2, [ell], ell[:-1], [True])]
    ren.paint(p, sub, frame)
    p.restore()


__all__ = [
    "SCHEMA", "GOLDEN_ANGLE", "distinct_color", "qcolor",
    "MIN_HUE_SEP", "NEIGHBOR_K", "MAX_NEIGHBOR_ITEMS", "MAX_SLOT_SEARCH",
    "MAX_CONSTRAINT_HUES",
    "neighbor_pairs", "deconflict_slots", "slot_lut", "slot_of",
    "PointsOverlay", "LabelsOverlay", "TracksOverlay", "VoxelsOverlay", "MeshOverlay",
    "OverlaySettings", "TabInfo", "TAB_INFO", "TABS", "TAB_BY_KEY", "PER_ITEM_MODE",
    "FieldSpec", "FIELDS", "SPREAD_ROLES", "SPREAD_ROLE_LABEL", "field_enabled",
    "spread_settings", "FILE_SUFFIX", "FILE_FILTER", "USER_FILE", "PROJECT_FILE",
    "ENV_VAR", "read_json", "write_json", "default_sources", "load_defaults",
    "defaults_for",
    "PointMark", "TrackPath", "MeshSection", "OverlayFrame", "OverlayRenderer",
    "MAX_OUTLINE_SEGMENTS", "OUTLINE_AA_LIMIT", "MAX_ID_LABELS", "MAX_SECTION_TRIANGLES",
    "mesh_section", "render_preview",
]
