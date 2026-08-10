"""Stitch (M→1) (``util.stitch``) — Stitch the multipoint axis into one mosaic (M→1, Y/X grow): tiles placed from the file's stage log (optionally refined by phase correlation) or on an explicit grid, blended feather/max/mean/overwrite."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Tuple

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import is_structure
from nodegraph.engine import EvalContext
from nodegraph.metadata import drop_position_keys, stitch as _meta_stitch
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InBool, InDataset, InFloat, InInt, Mode, OutDataset
from nodegraph.streaming import MultiViewProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane_p
from nodegraph.catalog._shared.sampling import _sampled
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Stitch (axis-changing: M→1 tile mosaic, Y/X grow) ───────────────────────────
#
# The multipoint (M) axis of a plate/mosaic acquisition holds the FIELDS of one
# specimen, and the file records where the stage was for each one. This node is the
# op that turns those fields back into the single image they were sampled from — the
# `MULTI_VIEW` footprint and the `stitch` meta_transform have existed since V2.02/V2.03
# with no node behind them, and this is that node.
#
# The layout maths is ported from v1's `nd2studios/backend/exporters/stitch_exporter.py`
# (`compute_tile_layout`, commit 1aa2641), which ran against the lab's real plate files:
# stage µm → pixel offsets directly, gaps between non-adjacent fields left black, no
# silent grid guess. Two deliberate departures from that port are documented at their
# sockets — `flip_x`/`flip_y` replace v1's undocumented `reversed(offsets)` (§ _stitch_
# offsets_stage), and a missing position log now REFUSES instead of falling back to a
# sqrt grid the user never asked for.

#: Overlap (px) a tile pair must share on BOTH axes before `stage+refine` will try to
#: correlate it. Not a tuning knob and deliberately not a socket: below roughly this the
#: window is too small for an FFT peak to mean anything, so a "refinement" from it is
#: noise dressed as a measurement — the pair is dropped and the stage positions stand.
_STITCH_MIN_OVERLAP_PX = 16
#: Weight of the "stay where the stage said" prior in the refine solve, relative to a
#: correlated pair (weight 1). Small enough that any accepted pair dominates it, non-zero
#: so the system is always full rank: a tile that shares no usable overlap with anything
#: (a lone field on the far side of a well) keeps its stage position instead of being
#: dragged to the origin by a minimum-norm least-squares solution.
_STITCH_PRIOR_W = 1e-3
def _stitch_stage_xy(ds: Dataset, n_m: int) -> list:
    """The per-multipoint stage coordinates ``[(x_um, y_um), …]`` for ``ds``, or ``[]``.

    Read straight off the input **payload's** metadata, which is the sanctioned route for
    non-calibration provenance (`wire-node-v2` §7b): ``stage_xy_um`` is not one of the
    locked :data:`~nodegraph.dataset.CALIBRATION_KEYS` — it is a per-M geometry LIST that
    no ``meta_transform`` could keep true across a crop — so it rides the Dataset rather
    than the envelope (:data:`nodelab_v2.ingest.STAGE_KEYS`), and ``_StrictCalibMetadata``
    passes a non-calibration read through untouched.

    A log that does not cover every multipoint returns ``[]``: v1 silently truncated to the
    positions it had and stitched a partial mosaic, which looks exactly like a complete one."""
    xy = getattr(ds, "metadata", {}).get("stage_xy_um") or []
    if len(xy) < n_m:
        return []
    out = []
    for m in range(n_m):
        try:
            out.append((float(xy[m][0]), float(xy[m][1])))
        except (TypeError, ValueError, IndexError):
            return []
    return out
def _stitch_offsets_stage(stage_xy: list, px_um: float, tile_h: int, tile_w: int,
                          flip_x: bool, flip_y: bool) -> list:
    """Stage µm → integer ``(y, x)`` canvas corner offsets, normalized so the top-left
    tile sits at ``(0, 0)``.

    The base mapping is the naive one — image ``+x``/``+y`` run along stage ``+x``/``+y``
    — with ``flip_x``/``flip_y`` mirroring either axis. **The file does not record the
    handedness** (see :meth:`nodelab_v2.viewer.ImageViewer._hover_text`), so it cannot be
    derived; it is a property of how the camera is mounted on that scope.

    This is where v1's ``reversed(offsets)`` went. That line placed tile *m*'s pixels at
    tile *(n-1-m)*'s slot, which is a 180° rotation of the mosaic **only when the scan
    positions happen to be point-symmetric** — true for a full rectangular raster, false
    for the sparse or ROI scans the same code also served, where it simply scattered the
    tiles. Its net effect on the regular case is exactly ``flip_x=True, flip_y=False``
    against this base mapping, which is why those are the defaults: a plate that stitched
    correctly under v1 still does, and a scope with the other handedness now has a control
    instead of needing the source edited."""
    xs = np.array([p[0] for p in stage_xy], dtype=float)
    ys = np.array([p[1] for p in stage_xy], dtype=float)
    u = (xs.max() - xs) if flip_x else (xs - xs.min())
    v = (ys.max() - ys) if flip_y else (ys - ys.min())
    oy = np.rint(v / px_um).astype(int)
    ox = np.rint(u / px_um).astype(int)
    return _stitch_normalize(list(zip(oy.tolist(), ox.tolist())), tile_h, tile_w)[0]
def _stitch_offsets_grid(n_m: int, tile_h: int, tile_w: int, cols: int) -> list:
    """Row-major contact-sheet offsets — v1's ``grid_fallback``, promoted to something the
    user asks for by name. ``cols <= 0`` picks ``ceil(sqrt(M))``, the near-square v1 used."""
    if cols <= 0:
        cols = max(1, int(np.ceil(np.sqrt(n_m))))
    return [((m // cols) * tile_h, (m % cols) * tile_w) for m in range(n_m)]
def _stitch_normalize(offsets: list, tile_h: int, tile_w: int):
    """Shift ``offsets`` so the minimum corner is ``(0, 0)`` and report the canvas that
    bounds every tile → ``(offsets, canvas_h, canvas_w)``."""
    oy0 = min(o[0] for o in offsets)
    ox0 = min(o[1] for o in offsets)
    norm = [(int(o[0] - oy0), int(o[1] - ox0)) for o in offsets]
    return (norm,
            max(o[0] for o in norm) + tile_h,
            max(o[1] for o in norm) + tile_w)
def _stitch_pairs(offsets: list, tile_h: int, tile_w: int) -> list:
    """Every tile pair whose placed boxes share at least :data:`_STITCH_MIN_OVERLAP_PX`
    on both axes → ``(i, j, y0, y1, x0, x1)`` in canvas coordinates."""
    pairs = []
    n = len(offsets)
    for i in range(n):
        for j in range(i + 1, n):
            y0 = max(offsets[i][0], offsets[j][0])
            y1 = min(offsets[i][0] + tile_h, offsets[j][0] + tile_h)
            x0 = max(offsets[i][1], offsets[j][1])
            x1 = min(offsets[i][1] + tile_w, offsets[j][1] + tile_w)
            if (y1 - y0) >= _STITCH_MIN_OVERLAP_PX and (x1 - x0) >= _STITCH_MIN_OVERLAP_PX:
                pairs.append((i, j, y0, y1, x0, x1))
    return pairs
def _stitch_refine(ctx: EvalContext, prov: Any, offsets: list, tile_h: int, tile_w: int,
                   *, max_shift_px: float, min_ncc: float, hp_sigma_px: float,
                   ref_t: int, ref_z: int, ref_c: int):
    """Refine stage-derived ``offsets`` by phase-correlating each overlapping pair, then
    solving one globally consistent layout. Returns ``(offsets, note)``.

    Register **once, on one reference plane** (``t=0``, mid-Z, ``c=0``) and apply to every
    ``(t, z, c)`` — the same rule ``align.drift`` follows, and for the same reason: the
    fields do not move between channels, so one layout keeps colocalization exact.

    Three stages:

    1. **Per pair** — cut the overlap window out of both tiles, high-pass (kill the
       illumination gradient, which otherwise correlates better than the specimen does)
       and Hann-window them, then ``phase_cross_correlation`` at 10× upsampling. The
       primitives are ``nodegraph.kernels.registration``'s ``_highpass``/``_hann2d``/
       ``_ncc``, vendored verbatim from the v1 **stitcher** — its own docstring records
       them as "the same ones the stitcher uses", so this is the original pairing rather
       than a fourth copy of them.
    2. **Accept or drop** — a residual beyond ``max_shift_px`` on either axis, or a
       post-shift NCC under ``min_ncc``, means the FFT locked onto the wrong peak (flat
       agar, a bubble, an empty corner). Dropped pairs are not errors; the stage position
       simply stands, which is exactly what ``layout=stage`` would have done.
    3. **Global solve** — one weighted least-squares per axis over ``p_j - p_i = d_ij``,
       plus the weak stay-put prior (:data:`_STITCH_PRIOR_W`). Solving globally rather than
       chaining pairwise shifts is what stops error accumulating along a scan row and
       leaving the last tile metres out.

    An accepted pair is weighted by **how much evidence it rests on** —
    ``ncc × sqrt(overlap area)``, scaled so a perfectly-correlated whole-tile overlap
    weighs 1. Both halves are needed and both were measured on a 6-tile fixture: pair
    estimates degrade sharply as the shared band narrows (mean absolute error 0.6 px over
    overlaps wider than 25 px, against 2.0-2.3 px for the 18-21 px corners), and the NCC
    tracks it (0.86-0.98 versus 0.65-0.69). Unweighted, those few thin corners dragged the
    solved layout a whole pixel off a layout the wide bands had pinned exactly."""
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation
    from nodegraph.kernels.registration import _hann2d, _highpass, _ncc

    pairs = _stitch_pairs(offsets, tile_h, tile_w)
    if not pairs:
        return offsets, "refine: no overlapping pairs (positions abut or are disjoint)"

    def window(i: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        oy, ox = offsets[i]
        return np.asarray(prov.get_region(0, i, ref_t, ref_z, ref_c,
                                          y0 - oy, y1 - oy, x0 - ox, x1 - ox), dtype=float)

    accepted = []
    ctx.progress(0, len(pairs), "correlating tile pairs")
    for k, (i, j, y0, y1, x0, x1) in enumerate(pairs):
        han = _hann2d((y1 - y0, x1 - x0))
        a = _highpass(window(i, y0, y1, x0, x1), hp_sigma_px) * han
        b = _highpass(window(j, y0, y1, x0, x1), hp_sigma_px) * han
        sh = phase_cross_correlation(a, b, upsample_factor=10)[0]
        dy, dx = float(sh[0]), float(sh[1])
        if abs(dy) <= max_shift_px and abs(dx) <= max_shift_px:
            crop = int(np.ceil(max(abs(dy), abs(dx)))) + 1
            bs = ndi_shift(b, shift=(dy, dx), order=1, mode="constant")
            if crop * 2 < min(a.shape):
                score = _ncc(a[crop:-crop, crop:-crop], bs[crop:-crop, crop:-crop])
            else:
                score = _ncc(a, bs)
            if score >= min_ncc:
                # a[k] ≈ b[k - sh] ⇒ tile j's true corner is its current one PLUS sh, so
                # the measured relative displacement is (o_j + sh) - o_i
                w = score * np.sqrt((y1 - y0) * (x1 - x0) / float(tile_h * tile_w))
                accepted.append((i, j,
                                 (offsets[j][0] + dy) - offsets[i][0],
                                 (offsets[j][1] + dx) - offsets[i][1], w))
        ctx.progress(k + 1, len(pairs), "correlating tile pairs")

    n = len(offsets)
    rows = len(accepted) + n
    a_mat = np.zeros((rows, n), dtype=float)
    b_y = np.zeros(rows, dtype=float)
    b_x = np.zeros(rows, dtype=float)
    for r, (i, j, dy, dx, w) in enumerate(accepted):
        a_mat[r, j] = w
        a_mat[r, i] = -w
        b_y[r], b_x[r] = w * dy, w * dx
    for i in range(n):                        # the stay-put prior, one row per tile
        r = len(accepted) + i
        a_mat[r, i] = _STITCH_PRIOR_W
        b_y[r] = _STITCH_PRIOR_W * offsets[i][0]
        b_x[r] = _STITCH_PRIOR_W * offsets[i][1]
    py = np.linalg.lstsq(a_mat, b_y, rcond=None)[0]
    px_ = np.linalg.lstsq(a_mat, b_x, rcond=None)[0]
    solved = [(int(round(py[i])), int(round(px_[i]))) for i in range(n)]
    return (_stitch_normalize(solved, tile_h, tile_w)[0],
            f"refine: {len(accepted)}/{len(pairs)} pairs accepted")
def _stitch_feather_weight(tile_h: int, tile_w: int) -> np.ndarray:
    """A tile's blending weight: distance to the nearest tile edge, separable and scaled
    to a ``1.0`` peak. Zero-free (the ramp starts at 1, not 0) so a pixel covered by only
    one tile still has a defined value at that tile's very edge.

    This is linear blending — a pixel in an overlap is the weighted mean of the tiles
    covering it, each weighted by how far from ITS edge the pixel is, so a tile's
    contribution fades out exactly as its neighbour's fades in and the seam disappears."""
    wy = np.minimum(np.arange(tile_h), np.arange(tile_h)[::-1]) + 1.0
    wx = np.minimum(np.arange(tile_w), np.arange(tile_w)[::-1]) + 1.0
    w = np.outer(wy, wx)
    return w / w.max()
def _stitch_plane(read_tile: Callable[[int], np.ndarray], offsets: list,
                  canvas_h: int, canvas_w: int, blend: str,
                  weight: Optional[np.ndarray],
                  tile_h: int, tile_w: int) -> np.ndarray:
    """Paste every tile onto one canvas and return it. ``read_tile(m)`` is pulled per
    tile and dropped again, so the peak is the canvas plus one tile (see
    :class:`~nodegraph.streaming.MultiViewProvider`).

    Offsets may be **negative** and may point past the canvas: that is how a WINDOW of the
    mosaic is stitched (the provider shifts every offset by the window origin, so a tile
    overlapping the window from outside it simply has its leading rows/columns cropped
    here). The clip is computed per tile and applied to the feather weight identically —
    the weight is a property of the tile, not of the canvas, so a cropped tile keeps
    exactly the ramp it would have had in a full-canvas stitch and a window is therefore
    pixel-identical to the same region of the whole.

    ``tile_h``/``tile_w`` are the placed size of one tile, and they are here so a tile that
    cannot touch the canvas is skipped **without being read**. That is the entire cost of a
    windowed stitch: ``read_tile`` decompresses a whole source plane (4 Mpx on the WellA3
    mosaic) and a window touches at most a handful of tiles. Testing only the far edge
    ``oy >= canvas_h`` — as this did until V2.20 — caught a tile below the window but missed
    its mirror image: a tile ABOVE the window has a negative offset, so ``max(0, oy)`` put it
    inside the canvas, it was read in full, and the intersection test below then threw it
    away. A 32×32 window of a 3×3 mosaic read all nine tiles; the far corner of a 49-position
    one read all forty-nine."""
    weighted = blend in ("feather", "mean")
    acc = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    # float64, not float32: the weight sum divides the value sum, so a float32 accumulator
    # puts its own ~1e-7 relative error straight into every blended pixel — measured at
    # 4.5e-5 counts on a 500-count fixture whose overlap should have been EXACT (every
    # tile there carries the same value, so any weighted mean of them is that value). The
    # canvas is already float64; halving one of the two accumulators is not worth making
    # the blended path the only one that cannot reproduce its input.
    wsum = np.zeros((canvas_h, canvas_w), dtype=np.float64) if weighted else None
    painted = np.zeros((canvas_h, canvas_w), dtype=bool) if blend == "max" else None
    for m, (oy, ox) in enumerate(offsets):
        # Intersect the tile's placed box with the canvas FIRST, so a tile that does not
        # overlap it at all is skipped without being read. BOTH sides of each axis: past the
        # far edge (`oy >= canvas_h`) and entirely before the near one (`oy + tile_h <= 0`,
        # the negative-offset case a window produces).
        if (oy >= canvas_h or ox >= canvas_w
                or oy + tile_h <= 0 or ox + tile_w <= 0):
            continue
        ty0, tx0 = max(0, -oy), max(0, -ox)
        cy0, cx0 = max(0, oy), max(0, ox)
        tile = read_tile(m)
        th, tw = tile.shape[0], tile.shape[1]
        cy1 = min(oy + th, canvas_h)
        cx1 = min(ox + tw, canvas_w)
        if cy1 <= cy0 or cx1 <= cx0 or ty0 >= th or tx0 >= tw:
            continue
        sub = tile[ty0:ty0 + (cy1 - cy0), tx0:tx0 + (cx1 - cx0)]
        oy, ox, y1, x1 = cy0, cx0, cy0 + sub.shape[0], cx0 + sub.shape[1]
        if weighted:
            wt = (weight[ty0:ty0 + sub.shape[0], tx0:tx0 + sub.shape[1]]
                  if blend == "feather" else 1.0)
            acc[oy:y1, ox:x1] += sub * wt
            wsum[oy:y1, ox:x1] += wt
        elif blend == "max":
            region = acc[oy:y1, ox:x1]
            seen = painted[oy:y1, ox:x1]
            np.copyto(region, np.where(seen, np.maximum(region, sub), sub))
            seen[...] = True
        else:                                   # overwrite — v1's last-tile-wins
            acc[oy:y1, ox:x1] = sub
    if weighted:
        # In place: a third full-canvas allocation is real money at mosaic sizes (86 MB
        # per copy even at the display level, 1.3 GB at level 0). Safe because `acc` is
        # only ever added to where a tile covers, so the pixels `where` skips are still
        # the 0.0 they were initialized to — which is what an uncovered pixel must read.
        return np.divide(acc, wsum, out=acc, where=wsum > 0)
    return acc
def _stitch_layout(ctx: EvalContext, ds: Dataset, prov: Any, ax: AxisSizes):
    """Resolve the whole layout → ``(offsets, canvas_h, canvas_w, note)``.

    Refuses rather than guesses, three ways, each of which v1 answered with a silent
    sqrt-grid fallback that is indistinguishable from a real mosaic once written to disk:
    no position log, a log that does not cover every M, and a log whose positions are all
    the same point (an M axis that indexes repeat VISITS, not fields)."""
    layout = ctx.params.get("__modes__", {}).get("layout", "stage")
    tile_h, tile_w = ax.y, ax.x
    if layout == "grid":
        cols = int(ctx.params.get("grid_cols", 0) or 0)
        offsets, h, w = _stitch_normalize(
            _stitch_offsets_grid(ax.m, tile_h, tile_w, cols), tile_h, tile_w)
        return offsets, h, w, f"grid[{ax.m} tiles]"

    stage_xy = _stitch_stage_xy(ds, ax.m)
    if not stage_xy:
        have = len(getattr(ds, "metadata", {}).get("stage_xy_um") or [])
        raise ValueError(
            f"layout='{layout}' needs a per-position stage log covering all {ax.m} "
            f"multipoints, and this Dataset carries {have}. TIFFs never have one, and an "
            f"ND2 whose SDK did not fill the XYPosLoop in has none either. Set the Layout "
            f"mode to 'grid' for a contact-sheet montage instead — a guessed placement "
            f"would look exactly like a measured one.")
    px_um = float(ctx.calib("pixel_size_um") or 0.0)
    if px_um <= 0:
        raise ValueError(
            "stitching needs pixel_size_um to turn stage microns into pixels, and this "
            "Dataset declares none. Load the file through the ND2/TIFF reader, or set the "
            "Layout mode to 'grid' (which needs no calibration).")
    flip_x = bool(ctx.params.get("flip_x", True))
    flip_y = bool(ctx.params.get("flip_y", False))
    offsets = _stitch_offsets_stage(stage_xy, px_um, tile_h, tile_w, flip_x, flip_y)
    span_y = max(o[0] for o in offsets)
    span_x = max(o[1] for o in offsets)
    if ax.m > 1 and span_y < 1 and span_x < 1:
        raise ValueError(
            f"all {ax.m} stage positions land within one pixel of each other, so this M "
            f"axis indexes repeat visits to one field, not a tile grid — stitching them "
            f"would stack {ax.m} copies of the same view. Set the Layout mode to 'grid' if "
            f"a contact sheet is what you want.")
    note = f"stage[{ax.m} tiles]"
    if layout == "stage+refine":
        offsets, refined = _stitch_refine(
            ctx, prov, offsets, tile_h, tile_w,
            max_shift_px=to_pixels_v2(float(ctx.params.get("refine_max_shift", 10.0)),
                                      "um", pixel_size_um=px_um),
            min_ncc=float(ctx.params.get("refine_min_ncc", 0.3)),
            hp_sigma_px=to_pixels_v2(float(ctx.params.get("refine_highpass", 5.0)),
                                     "um", pixel_size_um=px_um),
            ref_t=0, ref_z=ax.z // 2, ref_c=0)
        note = f"stage+refine[{ax.m} tiles; {refined}]"
    offsets, h, w = _stitch_normalize(offsets, tile_h, tile_w)
    return offsets, h, w, note
def _compute_stitch(ctx: EvalContext) -> Dataset:
    """Fuse the multipoint axis into one mosaic (M→1, Y/X grow) — the tile-stitch node.

    Resolved spec (`build-node-v2` §0):

    * **Kind** utility, axis-changing → ``op_key="util.stitch"``, ``meta_transform=stitch``
      (M→1; the canvas extent is reported UNKNOWN, never guessed — see below).
    * **Data contract** image → image. Lattice attribute layers that no longer fit the
      canvas are dropped by ``reshaped_axes``, exactly as Crop/Resample drop theirs; a
      **structure** table (Label/Point/Track/Mesh) is REFUSED, because its ``m``/``y``/``x``
      columns index the tile grid and nothing here can honestly re-address them (detections
      duplicated across a seam would need deduplicating, which is a different node's job).
      Stitch first, then detect.
    * **2D/3D** no lever: each ``(t, z, c)`` plane is placed with the same layout, so the
      node is dimension-agnostic. A z-stack stitches to a z-stack.
    * **Footprint** ``MULTI_VIEW`` / ``{"m","y","x"}`` — it reads every M at one ``(t,z,c)``
      and is the first node in the catalog to claim the footprint V2.02 defined for it.
    * **Backend** numpy for the paste; ``skimage.registration.phase_cross_correlation`` +
      ``scipy.ndimage.shift`` only on the ``stage+refine`` path.
    * **Performance** no numba: the hot path is whole-array slice-adds into a canvas, which
      is already one vectorized pass per tile (`wire-node-v2` §12, row 1).

    **Why the header says UNKNOWN.** ``metadata.stitch`` marks the output Y/X unknown
    because the canvas depends on the stage log, and the stage log rides the *payload*, not
    the envelope — the runner deliberately keeps per-M geometry out of the calibration
    schema. So the edit-time pass cannot compute the extent without reading data, and it
    reports "not statically knowable" rather than the input's tile size, which would be a
    confident lie. The payload's axes are the real canvas.

    **The canvas is streamed, never realized 6-D.** One plane of a 49-position mosaic is
    ~268 Mpx; the whole 6-D array would be tens of GB. A
    :class:`~nodegraph.streaming.MultiViewProvider` stitches one ``(t,z,c)`` plane on
    demand instead, pulling one tile at a time.
    """
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("stitch needs an image provider on its input Dataset")
    ax = prov.axes
    structure = sorted({attr.domain.value for attr in ds.attributes.values()
                        if is_structure(attr.domain)})
    if structure:
        raise ValueError(
            f"this Dataset carries {'/'.join(structure)} structure table(s), whose m/y/x "
            f"columns address the per-tile grid — stitching would silently leave every "
            f"object at the wrong place on the canvas, and objects seen twice across a "
            f"seam would stay duplicated. Stitch BEFORE detection/labelling, then run the "
            f"analysis on the mosaic.")
    if int(ax.m) < 2:
        raise ValueError(
            "this input has only ONE multipoint, so there is no tile grid to stitch — the "
            "output would be a copy of the single field on a canvas its own size.\n"
            "The usual cause is the solo-frame troubleshooting scope (F9), which narrows M "
            "to the position the cursor is on: it makes a per-frame node cheap, but this "
            "node's unit of work IS every M at one (t, z, c), so scoping M does not make it "
            "faster — it changes what it produces. Pick every position on the M strip, or "
            "turn the scope off, then stitch.\n"
            "If the file really has one position, this node has nothing to do — delete it.")
    blend = ctx.params.get("__modes__", {}).get("blend", "feather")
    offsets, canvas_h, canvas_w, note = _stitch_layout(ctx, ds, prov, ax)
    new_axes = replace(ax, m=1, y=canvas_h, x=canvas_w)
    # The layout arrives as an argument rather than a closed-over constant because the
    # provider re-derives it per PYRAMID LEVEL (MultiViewProvider): the feather ramp has
    # to match the tile size actually being pasted, so it is built per tile shape and
    # kept — there are at most `levels` of them, and each is one tile in size.
    weights: Dict[Tuple[int, int], np.ndarray] = {}

    def fuse(read_tile: Callable[[int], np.ndarray], offs, ch: int, cw: int,
             th: int, tw: int) -> np.ndarray:
        wt = None
        if blend == "feather":
            wt = weights.get((th, tw))
            if wt is None:
                wt = weights[(th, tw)] = _stitch_feather_weight(th, tw)
        return _stitch_plane(read_tile, offs, ch, cw, blend, wt, th, tw)

    cache = ctx.tiles
    if cache is None:                             # pre-C1 eager fallback (bare ctx)
        out = np.zeros((1, ax.t, ax.z, ax.c, canvas_h, canvas_w), dtype=float)
        for _m, t, z, c in _each_plane_p(ctx, replace(ax, m=1), "stitching"):
            out[0, t, z, c] = fuse(
                lambda mi, _t=t, _z=z, _c=c: np.asarray(
                    prov.get_region(0, mi, _t, _z, _c, 0, ax.y, 0, ax.x), dtype=float),
                offsets, canvas_h, canvas_w, ax.y, ax.x)
        res = ds.with_image(ArrayProvider(out)).reshaped_axes(new_axes)
    else:
        # The RESOLVED layout is folded into the fingerprint, not just the params: the
        # offsets come from `stage_xy_um` on the payload and (under refine) from the
        # reference pixels, neither of which the params capture. Without them two runs
        # that differ only in the position log would share cached canvases.
        fp = stream_fp("stitch", ctx.op_key,
                       {**ctx.params, "__offsets__": tuple(offsets),
                        "__canvas__": (canvas_h, canvas_w)},
                       ctx.reads.declared_reads(), (), prov)
        res = ds.with_image(
            MultiViewProvider(prov, new_axes, fuse, tuple(offsets), fp=fp, cache=cache)
        ).reshaped_axes(new_axes)
    out = _sampled(res, f"stitch[{note}]")
    # M→1 collapses twelve origins into the mosaic's union corner; sync it from the env
    # (post-`_meta_stitch`) so payload and header agree.
    origin = ctx.calib("origin_um")
    if origin is not None:
        out = out.with_metadata(origin_um=origin)
    # ...and retire the OTHER per-M lists in the same breath. They are read positionally
    # (`stage_xy_um[m]`), so leaving a 49-entry log on a 1-position canvas does not read as
    # stale — the Viewer's hover readout computes an absolute stage coordinate from
    # position 0's centre and the CANVAS width, i.e. a confident wrong answer that looks
    # like a handedness bug. Dropping rather than subsetting is the honest move: after a
    # stitch there is no per-position stage coordinate left to keep, only one canvas, and
    # `origin_um` above already carries its corner — which is exactly why
    # `placement.field_box` prefers `origin_um` and keeps the stage log as a fallback only
    # for a Dataset that never crossed the calibration seam.
    #
    # This matters most under a BAKE: `write_checkpoint` copies `ds.metadata` into the
    # manifest, so without this the transient wrong list becomes a permanent one that
    # `checkpoint_envelope` then feeds to every downstream node.
    retire = drop_position_keys(out.metadata)
    return out.with_metadata(**retire) if retire else out
_STITCH_FLIP_DOC = (
    "Mirror the {axis} axis when turning stage microns into canvas pixels. The ND2 does "
    "NOT record which way the camera is mounted relative to the stage, so this cannot be "
    "derived — it is a property of the microscope, and once you have it right for one "
    "file it is right for every file from that scope. Get it wrong and the mosaic comes "
    "out mirrored: the tiles are individually sharp but assembled in the wrong order, "
    "which is easiest to spot along a feature that crosses a seam. {why} No effect on "
    "pixel values, only on where each field is placed.")
register_node(
    _compute_stitch, op_key="util.stitch", label="Stitch (M→1)", category="utility",
    inputs=[
        InDataset(),
        InBool("flip_x", "Flip X", field=False, default=True,
               available_in={"layout": frozenset({"stage", "stage+refine"})},
               description=_STITCH_FLIP_DOC.format(
                   axis="X",
                   why="Default ON, which reproduces what NodeLab v1 produced on this "
                       "lab's plates (v1 mirrored X through an undocumented tile-order "
                       "reversal); turn it off for a scope whose camera X runs with the "
                       "stage.")),
        InBool("flip_y", "Flip Y", field=False, default=False,
               available_in={"layout": frozenset({"stage", "stage+refine"})},
               description=_STITCH_FLIP_DOC.format(
                   axis="Y",
                   why="Default OFF, matching v1's net behaviour on this lab's plates. "
                       "Turn it on for a scope whose stage Y counts the opposite way to "
                       "image rows.")),
        InFloat("refine_max_shift", "Max correction", unit="um", field=False, default=10.0,
                available_in={"layout": frozenset({"stage+refine"})},
                description=
                "How far a correlated tile pair is allowed to disagree with the stage log "
                "before the measurement is thrown away and the stage position stands. This "
                "is a REJECTION cut, not a search radius — phase correlation always "
                "searches the whole overlap, and a large answer means it locked onto the "
                "wrong peak (flat agar, a bubble, an empty corner), not that the stage was "
                "far out. RAISE it if a scope with sloppy encoders is having its real "
                "corrections discarded; LOWER it if refinement is visibly scattering tiles. "
                "Only read when Layout is 'stage+refine'. Compare against your stage's "
                "repeatability spec: 10 µm is generous for a modern encoded stage."),
        InFloat("refine_min_ncc", "Min correlation", unit="", field=False, default=0.3,
                available_in={"layout": frozenset({"stage+refine"})},
                description=
                "The normalized cross-correlation a tile pair must reach, after applying "
                "the measured shift, for that measurement to be believed. 0 is no "
                "agreement and 1 is identical overlap. LOWER accepts more pairs, including "
                "ones matched on noise, which drags neighbours off true; HIGHER keeps only "
                "confident matches and falls back to the stage log everywhere else — the "
                "safe direction, since the stage log is already close. 0.3 passes typical "
                "textured specimen overlap while rejecting empty background. Only read "
                "when Layout is 'stage+refine'."),
        InFloat("refine_highpass", "High-pass", unit="um", field=False, default=5.0,
                available_in={"layout": frozenset({"stage+refine"})},
                description=
                "Scale of the smooth background subtracted from each overlap before "
                "correlating. Uneven illumination is the same shape in every tile, so "
                "without this the correlation can lock onto the vignette instead of the "
                "specimen and confidently report zero shift. Set it well BELOW your "
                "features and ABOVE the illumination falloff: too large leaves the gradient "
                "in, too small erases the features along with it. Only read when Layout is "
                "'stage+refine'; it affects only the estimated positions, never the pixels "
                "that are pasted."),
        InInt("grid_cols", "Columns", unit="", field=False, default=0,
              available_in={"layout": frozenset({"grid"})},
              description=
              "How many tiles per row in the contact-sheet layout. 0 means auto — "
              "ceil(sqrt(M)), the near-square arrangement, which is what you want unless "
              "you know the acquisition's real row length and would rather see it laid out "
              "the way it was scanned. Purely how the fields are arranged on the canvas: "
              "no pixel is resampled and no measurement changes. Only read when Layout is "
              "'grid'."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("layout", ["stage", "stage+refine", "grid"], default="stage", label="Layout",
             description=
             "Where each tile is PLACED on the canvas. The file's stage log is the ground "
             "truth when it exists and covers every field; the other two options exist for "
             "when it does not, or when it is right to within a few microns but not to the "
             "pixel. Nothing here guesses silently — a missing position log refuses rather "
             "than inventing a grid.",
             choice_docs={
                 "stage":
                     "Convert each field's recorded stage position (µm) straight to pixel "
                     "offsets. Exact, instant, and correct as long as the stage is accurate "
                     "— which for most scopes means seams of a few pixels. Refuses when the "
                     "file has no position log, or one that does not cover every multipoint.",
                 "stage+refine":
                     "Start from the stage positions, then refine them by phase-correlating "
                     "each overlapping pair and solving the offsets globally. Removes visible "
                     "seams that pure stage placement leaves; pairs that share too little "
                     "overlap to correlate meaningfully are dropped and keep their stage "
                     "position, so a lone field is never dragged out of place.",
                 "grid":
                     "Place tiles on an explicit rows × columns lattice with a stated overlap "
                     "fraction, in acquisition order. The fallback for files with no stage log "
                     "at all, and the only option that requires you to know the acquisition "
                     "pattern — including its serpentine/flip handedness, which the file does "
                     "not record.",
             }),
        Mode("blend", ["feather", "max", "mean", "overwrite"], default="feather",
             label="Blend",
             description=
             "What happens where tiles OVERLAP. It affects only the overlap regions, so it "
             "cannot fix a placement error — but it decides whether a seam is invisible, "
             "visible, or quantitatively trustworthy, and three of the four alter pixel "
             "values there.",
             choice_docs={
                 "feather":
                     "Weighted mean, with each tile's weight falling off toward its own "
                     "edges. The best-looking result and the default: it hides both seams and "
                     "the illumination roll-off tiles usually have at their edges. Overlap "
                     "pixels are blends, so intensities there are not any single tile's "
                     "measurement.",
                 "max":
                     "Take the brightest contributing tile per pixel. Never dims a real "
                     "signal and needs no weighting, so it is the safe choice when the "
                     "overlap must not be averaged — at the cost of a visible bright seam "
                     "wherever tile edges are vignetted, and of keeping the noisier sample.",
                 "mean":
                     "Plain unweighted average of the overlapping tiles. Slightly better "
                     "noise than a single tile and dead simple to reason about, but the seam "
                     "stays visible as a discontinuity because a tile's dim edge is given the "
                     "same weight as its neighbour's bright centre.",
                 "overwrite":
                     "Last tile wins — no averaging at all, so every pixel comes from exactly "
                     "one field and is bit-identical to what was acquired. The choice when the "
                     "mosaic must remain raw data; the seams are hard edges, and which tile "
                     "wins depends on acquisition order.",
             }),
    ],
    granularity=Granularity.MULTI_VIEW, kernel_axes=frozenset({"m", "y", "x"}),
    meta_transform=_meta_stitch,
    description="Stitch the multipoint axis into one mosaic (M→1, Y/X grow): tiles placed "
                "from the file's stage log (optionally refined by phase correlation) or on "
                "an explicit grid, blended feather/max/mean/overwrite.")
