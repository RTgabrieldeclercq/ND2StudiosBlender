"""Label colours and label painting for the Viewer AND the movie export, without Qt.

The numpy half of the Viewer's label overlay: the golden-angle hue rule, the identity palette
that keeps neighbouring regions apart on the colour wheel, and the dense ``id -> slot`` LUT.
It lived in :mod:`nodelab_v2.overlays` until 2026-09-30, when ``io.write_movie`` learned to
draw labels with their ids and needed the SAME colours in the engine, which may not import Qt.
``overlays`` re-exports every name under its old spelling, so the Viewer is unchanged.

The painting half (:func:`label_palette`, :func:`paint_labels`, :func:`draw_label_ids`) is
new, and is what the movie export uses. It paints into an ``(H, W, 3)`` uint8 frame with numpy
and Pillow, following the Viewer's ``LabelsOverlay`` defaults (outline + light fill,
sat 205, val 255) so a labelled movie looks like the labelled Viewer.

Qt-free.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Golden-angle hue step — successive integer indices land far apart on the colour wheel,
#: so "each label / point / track a different colour" stays legible for hundreds of items.
GOLDEN_ANGLE = 137.507764

#: Above this many label ids **in view** the ids are not drawn — that many numbers on one
#: screen is noise, not information. Culling happens first, so zooming in brings ids back.
#: (The Labels tab tooltip quotes it, via its re-export in ``nodelab_v2.overlays``; the movie
#: export caps its burned-in ids at the same number.)
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


# ── painting: what the movie export draws labels with (2026-09-30) ─────────────────

#: The Viewer's ``LabelsOverlay`` defaults as 0..1 fractions. The movie starts from the same
#: look, so a labelled export resembles the labelled Viewer it was tuned in.
LABEL_SAT = 205 / 255.0
LABEL_VAL = 1.0
LABEL_FILL_OPACITY = 0.30
LABEL_OUTLINE_PX = 2
#: ``#e08a3a``, the Domain.LABEL colour ``LabelsOverlay.color`` defaults to.
LABEL_SINGLE_RGB = (224, 138, 58)

#: The colour modes a movie panel's labels accept (see :func:`label_palette`).
LABEL_COLOR_MODES = ("per_id", "deconflicted", "single")
#: The Viewer's three label styles.
LABEL_STYLES = ("both", "outline", "fill")


def label_palette(ids: np.ndarray, mode: str = "per_id", *,
                  centroids: Optional[np.ndarray] = None,
                  sat: float = LABEL_SAT, val: float = LABEL_VAL,
                  single: Tuple[int, int, int] = LABEL_SINGLE_RGB) -> np.ndarray:
    """A dense ``(max_id + 1, 3)`` uint8 LUT: row ``i`` is the colour label ``i`` paints.

    ``ids`` are the ids present; zero and negatives are ignored, and row 0 is never painted.

    * ``per_id``: the golden-angle hue of the id itself, which is exactly the Viewer's colour
      for an untracked region (``overlays.distinct_color(id)``). The same id keeps the same
      colour on every frame and at every z, which a z sweep needs: one object stays one
      colour as the stack passes through it.
    * ``deconflicted``: the Viewer's identity palette. Each id prefers its own slot and moves
      only when one of its nearest neighbours already sits within ``MIN_HUE_SEP`` degrees.
      Needs ``centroids``, one row per id in ``ids`` order. Neighbours differ from frame to
      frame, so a region can change colour between frames; that is the price of never
      giving two touching regions one colour.
    * ``single``: every region in ``single``.
    """
    ids = np.asarray(ids, dtype=np.int64).ravel()
    keep = ids > 0
    ids = ids[keep]
    lut = np.zeros((int(ids.max()) + 1 if ids.size else 1, 3), dtype=np.uint8)
    if not ids.size:
        return lut
    if mode == "single":
        lut[ids] = np.asarray(single, dtype=np.uint8)
        return lut
    slots = ids
    if mode == "deconflicted" and centroids is not None and ids.size > 1:
        cent = np.asarray(centroids, dtype=float)
        if cent.ndim == 2 and len(cent) == len(keep):
            cent = cent[keep]
        if cent.ndim == 2 and len(cent) == len(ids):
            pairs = neighbor_pairs(cent)
            got = deconflict_slots({int(i): int(i) for i in ids.tolist()},
                                   [(int(ids[a]), int(ids[b])) for a, b in pairs.tolist()])
            slots = np.array([got[int(i)] for i in ids.tolist()], dtype=np.int64)
    lut[ids] = _hues_to_rgb(_hues(slots), float(sat), float(val))
    return lut


def resample_ids(ids: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """``ids`` ``(H, W)`` point-sampled to ``shape`` ``(h, w)``: nearest neighbour.

    The only honest resize for a label raster. Averaging two ids invents a third that
    belongs to neither region, so the area filter every image goes through is wrong here.
    Each output pixel takes the id under its centre.
    """
    lab = np.asarray(ids)
    h, w = lab.shape[:2]
    oh, ow = int(shape[0]), int(shape[1])
    if (oh, ow) == (h, w):
        return lab
    ry = np.minimum(((np.arange(oh) + 0.5) * (h / float(oh))).astype(np.int64), h - 1)
    rx = np.minimum(((np.arange(ow) + 0.5) * (w / float(ow))).astype(np.int64), w - 1)
    return lab[ry[:, None], rx[None, :]]


#: The four 4-neighbour shifts, as slices into an array padded by one on each side.
_SHIFTS = ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
           (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None)))


def _inner_edge(lab: np.ndarray, width: int) -> np.ndarray:
    """Pixels of each region that lie within ``width`` px of a pixel carrying another id.

    Grown inward along the 4-neighbour grid, one step per pixel of width, and never across
    into a neighbouring region, so two touching regions both keep their own border. The
    frame edge is not a border (edge-replicated padding), which matches the Viewer: a cell
    cut by the field of view is not outlined along the cut.
    """
    pad = np.pad(lab, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    fg = lab > 0
    edge = np.zeros(lab.shape, dtype=bool)
    for sl in _SHIFTS:
        edge |= pad[sl] != c
    edge &= fg
    for _ in range(max(0, int(width) - 1)):
        pe = np.pad(edge, 1)
        grow = np.zeros_like(edge)
        for sl in _SHIFTS:
            grow |= pe[sl] & (pad[sl] == c)
        edge |= grow & fg
    return edge


def paint_labels(rgb: np.ndarray, ids: np.ndarray, lut: np.ndarray, *,
                 style: str = "both", fill_opacity: float = LABEL_FILL_OPACITY,
                 outline_px: int = LABEL_OUTLINE_PX) -> np.ndarray:
    """Paint label raster ``ids`` onto ``rgb`` and return a new ``(H, W, 3)`` uint8 array.

    ``ids`` must already be at ``rgb``'s size (:func:`resample_ids`). Painting happens at
    OUTPUT resolution, after the resize, because an outline drawn at full resolution and
    then area-averaged down fivefold fades to a faint smear, and a downscaled overview is
    where an outline matters most. ``style`` is the Viewer's: ``outline``, ``fill`` or
    ``both``. The fill alpha-blends each region's colour over the picture at
    ``fill_opacity`` (0..1). The outline is opaque, ``outline_px`` wide, on the region's
    inner edge.
    """
    out = np.array(rgb, dtype=np.uint8, copy=True)
    lab = np.asarray(ids)
    if lab.shape != out.shape[:2]:
        raise ValueError(f"label raster {lab.shape} does not match the frame "
                         f"{out.shape[:2]}; resample it first")
    lab = lab.astype(np.int64, copy=False)
    fg = (lab > 0) & (lab < len(lut))
    if not fg.any():
        return out
    colour = lut[np.where(fg, lab, 0)]
    a = float(max(0.0, min(1.0, fill_opacity)))
    if style in ("fill", "both") and a > 0.0:
        mix = out[fg].astype(np.float32) * (1.0 - a) + colour[fg].astype(np.float32) * a
        out[fg] = np.clip(np.rint(mix), 0, 255).astype(np.uint8)
    if style in ("outline", "both") and int(outline_px) > 0:
        edge = _inner_edge(np.where(fg, lab, 0), int(outline_px))
        out[edge] = colour[edge]
    return out


def label_id_marks(ids: np.ndarray, *, scale_y: float, scale_x: float, min_px: float,
                   max_ids: int = MAX_ID_LABELS) -> list:
    """Where each region's id is drawn, in OUTPUT pixels: ``[(x, y, id), ...]``.

    ``ids`` is the FULL-resolution label plane, so centroids are exact however far the
    frame is scaled; ``scale_y``/``scale_x`` map it to the output. A region smaller on
    screen than ``min_px`` (the square root of its scaled area) is skipped, because a number
    wider than its region covers the neighbours instead of naming one. Past ``max_ids`` only
    the largest regions keep their number. That is the Viewer's cap turned into a selection
    rather than an all-or-nothing switch, because a movie cannot zoom in to bring them back.
    """
    from nodegraph.catalog._shared.labels import _label_centroids

    lab = np.asarray(ids)
    present = np.unique(lab)
    present = present[present > 0]
    if not present.size:
        return []
    cent, counts = _label_centroids(lab, present)
    area = counts.astype(float) * float(scale_y) * float(scale_x)
    order = np.nonzero(np.sqrt(area) >= float(min_px))[0]
    if order.size > int(max_ids):
        order = order[np.argsort(-area[order], kind="stable")[:int(max_ids)]]
    return [(float((cent[i, 1] + 0.5) * scale_x), float((cent[i, 0] + 0.5) * scale_y),
             int(present[i])) for i in sorted(order.tolist())]
