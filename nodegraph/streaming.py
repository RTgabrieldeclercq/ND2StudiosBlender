"""Per-tile lazy / streaming evaluation (nodegraph v2, C1 — V2.04, LOCKED 2026-07-22).

The C1 execution model is **provider-chaining**: a ``TILEABLE`` / ``WHOLE_PLANE`` /
``WHOLE_VOLUME`` node returns a Dataset whose image is a *computing* lazy provider
instead of a realized :class:`~nodegraph.provider.ArrayProvider`. A
:class:`MapComputeProvider` serves ``read_region`` on demand: it decomposes the request
into **canonical tiles of its own grid**, computes each missing tile by reading the
tile extent **+ halo** from its base provider (itself possibly lazy — the chain is the
dataflow), applying the node's kernel, and cropping; only canonical tiles enter the
shared :class:`TileCache` (V2.04 §6b) so every chain level gets cache reuse. Plane /
volume units cache whole planes (volumes as per-z planar slabs — the keystone verdict).

Correctness pillars (V2.04 §3/§6b):

* **Flat fingerprints** — every streaming provider's identity is a single digest string
  computed **once at construction** from ``(op_key, params, declared calibration reads,
  field expression hashes, base fingerprint)``. Folding the declared reads closes the
  ``reseed_meta`` staleness hole (the node-level reads fence does not protect the tile
  cache); folding field expression hashes (which embed layer *revisions*) closes the
  attribute-layer hole; flatness avoids the nested-tuple ``_canon`` recursion blow-up
  on unrolled zone chains.
* **Halo = overlap-recompute** (V2.04 §2, probe-verified): windows are clipped at the
  *immediate base's* true extents, reproducing scipy ``mode='reflect'`` edge behavior.
* **cum-halo fence** — when ``2·cum_halo ≥ tile`` the accumulated windows make tiling
  pointless; the provider silently switches to the plane unit (V2.04 §6b).
* **Uniform freeze** — every array a streaming provider returns is read-only; an
  in-place kernel fails loudly and deterministically.

Qt-free; numpy + stdlib.
"""
from __future__ import annotations

import sys
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.memo import digest
from nodegraph.parallel import map_units, stream_dtype
from nodegraph.provider import ArrayProvider, TileProvider
from nodegraph.reducers import partial_reducer, reduce as _reduce_whole


#: The float width every streaming provider computes and caches in — ``float64`` unless
#: ``NODEGRAPH_FLOAT32=1`` (:func:`nodegraph.parallel.stream_dtype`). Resolved once at
#: import so one run cannot mix widths between chain levels.
#:
#: **What opting in costs.** float32 is a different computation, not a faster one. In
#: particular the ``_AxisReduceProvider`` promise below — that a tiled reduce is
#: *byte-identical* to the eager ``astype(float)`` reduce — holds for float64 only, since
#: the eager reference path is numpy-default double. So under the flag, tiled and eager
#: results agree to ~7 significant digits rather than exactly. What it buys, measured:
#: 1.52× on gaussian, 1.81× on uniform, 1.01× on median (compute- not bandwidth-bound),
#: plus 2× the effective :class:`TileCache` capacity for the same budget.
_F = stream_dtype()


# ── fingerprint stability guard (V2.04 §6b: no id-bearing reprs in a fp) ────────

_STABLE_SCALARS = (type(None), bool, int, float, str, bytes, np.integer, np.floating,
                   Enum)


def _assert_stable(o: Any, path: str = "value") -> None:
    """Reject values whose canonical encoding would fall into ``_canon``'s repr
    fallback (id-bearing reprs make the fingerprint unstable across re-pulls —
    silently cache-cold + ``changed`` always true)."""
    if isinstance(o, _STABLE_SCALARS) or isinstance(o, np.ndarray):
        return
    if isinstance(o, (tuple, list, set, frozenset)):
        for i, x in enumerate(o):
            _assert_stable(x, f"{path}[{i}]")
        return
    if isinstance(o, dict):
        for k, v in o.items():
            _assert_stable(k, f"{path} key {k!r}")
            _assert_stable(v, f"{path}[{k!r}]")
        return
    raise TypeError(
        f"unstable fingerprint input at {path}: {type(o).__name__!r} would hash via "
        f"repr() (id-bearing) — bake only plain scalars/containers into a streaming "
        f"provider fingerprint (V2.04 §6b)")


def stream_fp(kind: str, op_key: str, params: Mapping[str, Any],
              reads: Tuple[Tuple[str, str], ...], field_hashes: Tuple[str, ...],
              base: TileProvider) -> str:
    """The flat streaming-provider fingerprint digest (V2.04 §3/§6b): op + params +
    **declared calibration reads** (everything the closure could have baked from
    ``ctx.calib``) + **field expression hashes** (embed layer revisions) + the base
    provider's fingerprint. Computed once at construction; O(1) per chain level
    (a streaming base's own fingerprint is already a flat digest)."""
    p = dict(params)
    _assert_stable(p, "params")
    base_fp = base.fingerprint()
    _assert_stable(base_fp, "base fingerprint")
    # the base's tile size is folded in because the TileCache key's (iy, ix) address is
    # only meaningful ON a grid — two same-content providers on different grids must
    # never share tile entries (review 2026-07-22 BLOCKER: cross-grid collisions)
    return digest("stream", kind, op_key, p, tuple(reads), tuple(field_hashes),
                  base_fp, getattr(base, "tile", None))


# ── recursion headroom (deep unrolled chains; V2.04 §6b fence notes) ────────────

def _stack_depth() -> int:
    f = sys._getframe()
    d = 0
    while f is not None:
        d += 1
        f = f.f_back
    return d


@contextmanager
def recursion_headroom(extra_frames: int):
    """Scoped ``sys.setrecursionlimit`` raise — the stopgap for deep unrolled-zone
    chains (a recursive read/pull nests a few frames per chain level; the iterative
    ``_entry`` rewrite is a follow-up). Headroom is granted **relative to the live
    stack depth** — entering from an already-deep stack must not silently grant less
    than requested (review 2026-07-22)."""
    need = _stack_depth() + int(extra_frames) + 200
    old = sys.getrecursionlimit()
    if need <= old:
        yield
        return
    sys.setrecursionlimit(need)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


# ── the shared byte-budget LRU tile/unit cache (V2.04 §3) ───────────────────────

class TileCache:
    """Engine-owned LRU over frozen ndarrays, keyed by namespaced tuples that embed a
    streaming provider's fingerprint digest — the concrete V2.02 §9 "image tile key".
    Field materializations share the budget under a disjoint ``("f", ...)`` namespace.
    An entry larger than the whole budget is served but never stored; eviction is
    plain LRU. Eviction never invalidates handed-out arrays (they are immutable).

    **Thread-safe** (V2.14). Once node unit loops and plane reads run on a worker pool
    (:mod:`nodegraph.parallel`) several threads share one cache, and the bookkeeping here
    is not incidentally safe: ``self.nbytes -= old.nbytes`` is a read-modify-write across
    three bytecodes, and the eviction loop mutates ``_lru`` while another thread may be
    inside ``get``. Both are guarded by ``_lock``.

    The expensive part of :meth:`put` — owning + freezing the buffer, i.e. a memcpy of a
    whole tile — is done OUTSIDE the lock deliberately; holding it there would serialize
    every worker on a copy and give back most of the parallel win.

    Two threads may still *compute* the same missing tile concurrently and both call
    :meth:`put`. That is benign rather than a race to fix: node computes are pure and
    deterministic, so the two results are byte-identical and the second simply replaces
    the first. It costs duplicated work in the rare case, and avoiding it (a per-key
    in-flight latch) would risk deadlock across a nested provider chain, where a tile's
    computation legitimately re-enters the cache for its base's tiles."""

    def __init__(self, budget_bytes: int = 1 << 30) -> None:
        self.budget = int(budget_bytes)
        self._lru: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[np.ndarray]:
        with self._lock:
            a = self._lru.get(key)
            if a is None:
                self.misses += 1
                return None
            self._lru.move_to_end(key)
            self.hits += 1
            return a

    def put(self, key: Any, arr: np.ndarray) -> np.ndarray:
        """Freeze (owning the buffer first — a frozen *view* still reflects base
        writes) and store; returns the stored (read-only) array."""
        a = np.asarray(arr)
        if a.base is not None:
            a = a.copy()
        a.flags.writeable = False
        if a.nbytes > self.budget:               # oversize: serve, never store
            return a
        with self._lock:
            old = self._lru.pop(key, None)
            if old is not None:
                self.nbytes -= old.nbytes
            self._lru[key] = a
            self.nbytes += a.nbytes
            while self.nbytes > self.budget and len(self._lru) > 1:
                _, v = self._lru.popitem(last=False)
                self.nbytes -= v.nbytes
        return a

    def __len__(self) -> int:
        with self._lock:
            return len(self._lru)

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self.nbytes = 0
            self.hits = self.misses = 0


# ── streaming provider base ─────────────────────────────────────────────────────

def _freeze(a: np.ndarray) -> np.ndarray:
    if a.base is not None:
        a = a.copy()
    if a.flags.writeable:
        a.flags.writeable = False
    return a


class _NoCache:
    """Fallback when a provider's (weakly-held) engine cache has been collected:
    every read recomputes, results still come back frozen. Correctness-preserving —
    a memoized lazy Dataset stays readable after its engine is gone (V2.04 §3:
    providers hold a weak/late-bound cache reference; review 2026-07-22)."""

    budget = 0

    def get(self, key: Any) -> Optional[np.ndarray]:
        return None

    def put(self, key: Any, arr: np.ndarray) -> np.ndarray:
        return _freeze(np.asarray(arr))


_NO_CACHE = _NoCache()


class StreamProvider(TileProvider):
    """Base for computing lazy providers: flat cached fingerprint, chain ``depth``,
    accumulated halo, and canonical-tile request decomposition. ``levels = 1``:
    pyramid semantics for *computed* results are a Viewer decision (V2.04 §5-1) — a kernel
    applied to a downsampled input is not the downsample of its result, so a computed
    pyramid would be a lie. Two levels override it, each with its own reason to be allowed
    to: :class:`MultiViewProvider` (a paste of downsampled tiles *is* the downsampled paste)
    and :class:`_AxisReduceProvider` (a z/t fold is orthogonal to an xy mean-pool). Both keep
    level 0 exact, which is what makes it safe — no measurement reads above it.
    The cache reference is **weak** — a memoized Dataset must not pin an orphaned
    engine's whole TileCache across engine rebuilds (V2.04 §3)."""

    levels = 1

    #: Does one *compute* cover a whole ``(Z,Y,X)`` volume? (V2.14 — parallel realize.)
    #: This is the difference between the independent read tasks being ``(m,t,z,c)`` and
    #: ``(m,t,c)``. Touching any z of a volume-unit provider computes the ENTIRE volume
    #: and caches it as per-z slabs, so handing each z to its own worker would compute the
    #: same volume Z times concurrently — a Z-fold *waste*, not a speedup. Consumers that
    #: fan out over units must group by ``(m,t,c)`` when this is True.
    volume_unit = False

    #: Is this level's compute unit a whole **plane**, so that asking it for a small window
    #: costs the same as asking for the whole thing? (V2.20.)
    #:
    #: A tile-unit level is happy to be read one 512² tile at a time — that is the point of
    #: tiling. A plane-unit level is not: it computes (and caches) a whole plane on first
    #: touch, and :class:`MultiViewProvider`'s windowed path does not even cache, so N
    #: window reads of one plane cost N whole-plane computes. A consumer that decomposes its
    #: own output into tiles and pulls a matching window per tile therefore multiplies its
    #: base's work by its own tile count, which is how a Z-Project over a 49-position Stitch
    #: turned a 10-second mosaic into a 5-hour one (676 output tiles × 49 tiles × Z full
    #: source-plane reads). :class:`_AxisReduceProvider` reads this and folds whole planes
    #: instead; any future level that tiles its base must do the same.
    plane_unit = False

    def __init__(self, base: TileProvider, *, fp: str, cache: TileCache) -> None:
        self._base = base
        self._fp = fp
        self._cache_ref = weakref.ref(cache)
        self.axes = base.axes
        self.tile = base.tile
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)

    @property
    def _cache(self):
        c = self._cache_ref()
        return c if c is not None else _NO_CACHE

    def _fanout_ok(self) -> bool:
        """May this level compute its tiles concurrently? (V2.14 — see :meth:`_assemble`.)

        Yes when neighbouring tiles cannot duplicate each other's upstream work: either
        this level reads no halo (``halo == 0`` ⇒ tile windows are disjoint), or there is
        no lazy chain beneath it to duplicate into (``depth <= 1`` ⇒ the base is a real
        store, whose own reads are already internally threaded by blosc2)."""
        return getattr(self, "halo", 0) == 0 or self.depth <= 1

    def fingerprint(self) -> tuple:
        return ("stream", self._fp)              # flat — O(1) at any chain depth

    def level_axes(self, level: int) -> AxisSizes:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        return self.axes

    # ── request assembly ──────────────────────────────────────────────────────
    def _assemble(self, y0: int, y1: int, x0: int, x1: int,
                  tile_of: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """Serve window ``[y0:y1, x0:x1]`` from canonical tiles. A request that IS one
        whole canonical tile returns the cached array itself (no copy); any other
        window is assembled into a fresh (frozen) buffer — copy-on-serve, so a held
        window never pins a whole cached plane (V2.04 §6b)."""
        T = self.tile
        ay, ax_ = self.axes.y, self.axes.x
        y0, y1 = max(0, y0), min(y1, ay)
        x0, x1 = max(0, x0), min(x1, ax_)
        if y1 <= y0 or x1 <= x0:
            return _freeze(np.empty((max(0, y1 - y0), max(0, x1 - x0)), dtype=_F))
        if (y0 % T == 0 and x0 % T == 0
                and y1 == min(y0 + T, ay) and x1 == min(x0 + T, ax_)):
            return tile_of(y0 // T, x0 // T)     # exactly one canonical tile
        addrs = [(iy, ix)
                 for iy in range(y0 // T, (y1 - 1) // T + 1)
                 for ix in range(x0 // T, (x1 - 1) // T + 1)]
        # Compute the window's missing tiles CONCURRENTLY, then stitch serially (V2.14) —
        # but only where that is actually a win. This is the interactive path: a whole-plane
        # read of a 2048² plane on a 512² grid is sixteen independent kernel evaluations,
        # and serializing them was the single-core floor on the Viewer's per-frame latency.
        #
        # `_fanout_ok` is not caution, it is a MEASURED correction. With a halo AND a
        # provider stacked on other lazy providers, neighbouring output tiles read
        # OVERLAPPING base windows. Run serially, tile (0,1) reuses the base tiles tile
        # (0,0) just cached; run concurrently, all sixteen miss at once and each recomputes
        # the shared base tiles — a cache-miss stampede that adds total work. Measured on a
        # 3-deep tophat→gaussian→unsharp chain reading one 2048² plane: **0.74×**, i.e.
        # slower than serial. The same fan-out on a single-level 4096² median — one
        # expensive kernel, no base to stampede — is **7.0×**.
        #
        # So: fan out when there is no overlap to duplicate (`halo == 0`) or nothing
        # underneath to duplicate it in (`depth <= 1`); otherwise stay serial and keep the
        # warm-cache reuse, which is worth more than the concurrency. Either way the tiles
        # are stitched in a fixed address order, so the window is byte-identical.
        if self._fanout_ok() and len(addrs) > 1:
            tiles = map_units(lambda a: tile_of(a[0], a[1]), addrs)
        else:
            tiles = [tile_of(iy, ix) for iy, ix in addrs]
        out: Optional[np.ndarray] = None
        for (iy, ix), tl in zip(addrs, tiles):
            ty0, tx0 = iy * T, ix * T
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + tl.shape[0])
            sx0, sx1 = max(x0, tx0), min(x1, tx0 + tl.shape[1])
            if out is None:
                out = np.empty((y1 - y0, x1 - x0), dtype=tl.dtype)
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                tl[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return _freeze(out)

    @staticmethod
    def _window_of(plane: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """A window of a cached plane — copy-on-serve unless it is the whole plane."""
        y1 = min(y1, plane.shape[0])
        x1 = min(x1, plane.shape[1])
        if y0 == 0 and x0 == 0 and (y1, x1) == plane.shape:
            return plane
        return _freeze(plane[y0:y1, x0:x1].copy())


# ── the map providers (TILEABLE tile+halo / WHOLE_PLANE plane / WHOLE_VOLUME) ───

#: internal kernel contract — receives the float window and its address
#: ``(m, t, z, c, gy0, gy1, gx0, gx1)`` (the actual extent incl. halo), so a
#: field-consuming kernel can evaluate its Field on exactly that window.
MapFn = Callable[..., np.ndarray]


class MapComputeProvider(StreamProvider):
    """The TILEABLE / WHOLE_PLANE lazy unit (V2.04 §1). ``unit="tile"`` computes
    canonical tiles from tile+halo base windows (overlap-recompute, halo clipped at
    the base's true extents); ``unit="plane"`` computes whole planes on first touch.
    The cum-halo fence silently promotes a tile unit to the plane unit when
    ``2·cum_halo ≥ tile`` (window blow-up makes tiling pointless, V2.04 §6b)."""

    def __init__(self, base: TileProvider, fn: MapFn, *, halo: int = 0,
                 unit: str = "tile", fp: str, cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._fn = fn
        self.halo = int(max(0, halo))
        self.cum_halo = getattr(base, "cum_halo", 0) + self.halo
        self._plane_unit = (unit == "plane") or (2 * self.cum_halo >= self.tile)
        self.plane_unit = self._plane_unit    # the tiling consumer's fence (see base class)
        if self._plane_unit:
            # a plane-unit level computes + caches its WHOLE unit — a window-growth
            # cut point: downstream halo accumulation restarts here, so halo-0 chains
            # after a tripped fence stay tile-unit (review 2026-07-22)
            self.cum_halo = 0

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if self._plane_unit:
            return self._window_of(self._plane(m, t, z, c), y0, y1, x0, x1)
        return self._assemble(y0, y1, x0, x1,
                              lambda iy, ix: self._tile(m, t, z, c, iy, ix))

    def _tile(self, m: int, t: int, z: int, c: int, iy: int, ix: int) -> np.ndarray:
        key = ("t", self._fp, m, t, z, c, iy, ix)
        a = self._cache.get(key)
        if a is not None:
            return a
        T, h = self.tile, self.halo
        ay, ax_ = self.axes.y, self.axes.x
        ty0, tx0 = iy * T, ix * T
        ty1, tx1 = min(ty0 + T, ay), min(tx0 + T, ax_)
        gy0, gx0 = max(0, ty0 - h), max(0, tx0 - h)
        gy1, gx1 = min(ay, ty1 + h), min(ax_, tx1 + h)
        win = self._base.get_region(0, m, t, z, c, gy0, gy1, gx0, gx1)
        res = np.asarray(
            self._fn(np.asarray(win, dtype=_F), m, t, z, c, gy0, gy1, gx0, gx1),
            dtype=_F)
        res = res[ty0 - gy0:ty1 - gy0, tx0 - gx0:tx1 - gx0]
        return self._cache.put(key, res)

    def _plane(self, m: int, t: int, z: int, c: int) -> np.ndarray:
        key = ("p", self._fp, m, t, z, c)
        a = self._cache.get(key)
        if a is not None:
            return a
        ay, ax_ = self.axes.y, self.axes.x
        win = self._base.get_region(0, m, t, z, c, 0, ay, 0, ax_)
        res = np.asarray(self._fn(np.asarray(win, dtype=_F), m, t, z, c,
                                  0, ay, 0, ax_), dtype=_F)
        return self._cache.put(key, res)


#: volume kernel contract — ``(vol_float, m, t, c) -> (Z, Y, X)``.
VolumeFn = Callable[[np.ndarray, int, int, int], np.ndarray]


class VolumeComputeProvider(StreamProvider):
    """The WHOLE_VOLUME lazy unit: first touch of any window in ``(m, t, c)`` computes
    the whole ``(Z, Y, X)`` volume once and caches it as **per-z planar slabs** (the
    keystone verdict — never one monolith), then serves windows from the z-plane.
    Strictly better than today's realize-ALL-of-6D: only touched volumes compute."""

    volume_unit = True                   # one compute per (m,t,c) — see StreamProvider
    plane_unit = True                    # a window is no cheaper than the plane it is in

    def __init__(self, base: TileProvider, vfn: VolumeFn, *, fp: str,
                 cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._vfn = vfn
        self.cum_halo = 0        # a realized-unit level is a window-growth cut point

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range [0, {self.axes.z}) — an out-of-range "
                             f"read must not trigger a whole-volume compute")
        key = ("p", self._fp, m, t, z, c)
        plane = self._cache.get(key)
        if plane is None:
            ax = self.axes
            vol = self._base.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
            res = np.asarray(self._vfn(np.asarray(vol, dtype=_F), m, t, c),
                             dtype=_F)
            if res.shape != (ax.z, ax.y, ax.x):
                raise ValueError(
                    f"volume kernel returned {res.shape}, expected {(ax.z, ax.y, ax.x)}")
            for zi in range(ax.z):               # per-z planar slabs (best-effort cache)
                stored = self._cache.put(("p", self._fp, m, t, zi, c), res[zi])
                if zi == z:
                    plane = stored
        return self._window_of(plane, y0, y1, x0, x1)


class _AxisReduceProvider(StreamProvider):
    """Engine-driven tree-reduce over ONE acquisition axis (``_axis`` = ``"z"`` or
    ``"t"``, set by a subclass): output tile ``(iy, ix)`` folds the base's slices along
    that axis **incrementally** via the matching :class:`~nodegraph.reducers.PartialReducer`
    monoid (memory O(window)); a non-monoid reducer (median / sigma_clip / trimmed_mean)
    stacks the reduced-axis column (memory O(window·N)) — both exact, neither realizes a
    whole plane. Tiles are cast to float at lift so the tiled result is byte-identical to
    the eager ``astype(float)`` reduce.

    **Unless the base is plane-unit, in which case the unit here is the plane too** (V2.20).
    Tiling is a bargain only when the base charges per tile. Over a
    :class:`MultiViewProvider` it is the opposite of one: each output tile asked its base for
    a matching window, every one of those windows re-stitched the mosaic from its source
    tiles, and none of them was cached — so Z-Project on a 49-position Stitch cost
    ``n_out_tiles × Z × n_m`` full source-plane reads instead of ``Z × n_m``. On the WellA3
    mosaic (13106² canvas, 512 grid) that is a **676× multiplier**: a ~30-second projection
    became a five-hour one, which is the "it froze" the user sees.

    So when ``base.plane_unit`` is set, fold whole base planes into one whole output plane
    and serve windows out of it. Same arithmetic, same bytes — ``_fold`` is shared by both
    paths — just asked for in the granularity the base can actually deliver. Peak memory is
    the accumulator plus one base plane rather than one tile, which is the honest price of a
    base that has no cheaper unit to offer.

    **This is the second streaming provider WITH a pyramid** (V2.20), and it is what makes a
    projected mosaic *scrub* rather than merely finish. Folding whole planes fixes the
    catastrophe above but leaves the floor: with ``levels = 1`` the Viewer's
    ``render_plane_native`` has to read level 0, so displaying one 13106² projected canvas
    meant Z full-canvas stitches — tens of seconds per frame — to fill a 3.5 Mpx widget.
    Forwarding the base's pyramid serves level 3 instead: the same picture for 1/64th of the
    work.

    ``MultiViewProvider`` earned its pyramid by being a paste (stitching downsampled tiles
    *is* the downsampled stitch). This one earns it differently: **the reduce is orthogonal
    to the pyramid.** A z/t fold is per-``(y,x)``-column arithmetic and the pyramid is an xy
    mean-pool, so the two commute or nearly do:

    * ``sum`` and ``mean`` commute **exactly** — a coarse level is bit-for-bit the
      downsample of level 0 (up to float ordering).
    * ``max``, ``min`` and ``median`` do not. A coarse pixel is the reduce of mean-pooled
      columns rather than the mean-pool of reduced ones, which for ``max`` means
      ``max(mean) ≤ mean(max)``: the coarse projection reads very slightly *smoother*,
      bounded by the xy variation inside one 2**L block.

    Level 0 is untouched and exact in every case, and that is the whole safety argument:
    **no measurement can reach a coarse level.** Every node compute, :func:`realize`, the
    checkpoint writer and every export read ``get_region(0, …)``; the only callers that pass
    ``level > 0`` are the three display reads in :mod:`nodelab_v2.runner`, which then
    stride-decimate the result to a few thousand pixels anyway."""

    _axis = "z"

    def __init__(self, base: TileProvider, reducer: str, *, fp: str,
                 cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self._reducer = reducer
        self.axes = replace(base.axes, **{self._axis: 1})
        self.plane_unit = bool(getattr(base, "plane_unit", False))
        self.levels = max(1, int(getattr(base, "levels", 1)))

    def level_axes(self, level: int) -> AxisSizes:
        if not (0 <= level < self.levels):
            raise ValueError(f"level {level} out of range [0, {self.levels})")
        # off the base's REAL per-level geometry, never an assumed 2**level: `_build_levels`
        # floors odd sizes, so deriving it would drift from the planes actually being folded.
        return replace(self._base.level_axes(level), **{self._axis: 1})

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if not (0 <= level < self.levels):
            raise ValueError(f"level {level} out of range [0, {self.levels})")
        # A coarse level always folds whole planes: it is 1/4**L of level 0, so it caches
        # comfortably (level 3 of a 13106² canvas is 21 MB) and tiling it would only put the
        # plane_unit trap back for anything downstream that reads it a window at a time.
        if level or self.plane_unit:
            return self._window_of(self._plane(level, m, t, z, c), y0, y1, x0, x1)
        return self._assemble(y0, y1, x0, x1,
                              lambda iy, ix: self._tile(m, t, z, c, iy, ix))

    def _fold(self, slab: Callable[[int], np.ndarray]) -> np.ndarray:
        """Reduce ``self._base.axes[_axis]`` slices returned by ``slab(i)`` — the one
        arithmetic both the tile and the plane path use, so the granularity a read arrives
        at can never change the answer.

        A monoid reducer folds incrementally (memory O(one slice)); a non-monoid one
        (median / sigma_clip / trimmed_mean) has to see the whole column at once."""
        n = getattr(self._base.axes, self._axis)
        pr = partial_reducer(self._reducer)
        if pr is not None:
            acc = None
            for i in range(n):
                acc = pr.combine(acc, pr.lift(np.asarray(slab(i), dtype=_F)[None], (0,)))
            return np.asarray(pr.lower(acc), dtype=_F)
        col = np.stack([np.asarray(slab(i), dtype=_F) for i in range(n)], axis=0)
        return np.asarray(_reduce_whole(col, (0,), self._reducer), dtype=_F)

    def _slab_reader(self, level: int, m: int, t: int, z: int, c: int,
                     y0: int, y1: int, x0: int, x1: int) -> Callable[[int], np.ndarray]:
        """``slab(i)`` → window ``[y0:y1, x0:x1]`` of the base at ``level`` and reduced-axis
        index ``i``. The caller's coordinate for the reduced axis is always 0 (it is size-1
        on the output), so it is the one this overwrites."""
        coords = {"m": m, "t": t, "z": z, "c": c}

        def slab(i: int) -> np.ndarray:
            coords[self._axis] = i                # vary only the reduced axis
            return self._base.get_region(level, coords["m"], coords["t"], coords["z"],
                                         coords["c"], y0, y1, x0, x1)

        return slab

    def _tile(self, m: int, t: int, z: int, c: int, iy: int, ix: int) -> np.ndarray:
        # the reduced axis is size-1 in the output, so the caller's coordinate for it is
        # 0 (z-reduce) / 0 (t-reduce); it is included in the key harmlessly and the base
        # reader overwrites it with the real slice index.
        key = ("t", self._fp, m, t, z, c, iy, ix)
        a = self._cache.get(key)
        if a is not None:
            return a
        T = self.tile
        ty0, tx0 = iy * T, ix * T
        ty1, tx1 = min(ty0 + T, self.axes.y), min(tx0 + T, self.axes.x)
        return self._cache.put(
            key, self._fold(self._slab_reader(0, m, t, z, c, ty0, ty1, tx0, tx1)))

    def _plane(self, level: int, m: int, t: int, z: int, c: int) -> np.ndarray:
        key = ("p", self._fp, level, m, t, z, c)
        a = self._cache.get(key)
        if a is not None:
            return a
        ax = self.level_axes(level)
        return self._cache.put(
            key, self._fold(self._slab_reader(level, m, t, z, c, 0, ax.y, 0, ax.x)))


class ZReduceProvider(_AxisReduceProvider):
    """Reduce over Z → a z==1 provider (``util.zproject``; V2.04 §1)."""

    _axis = "z"


class TReduceProvider(_AxisReduceProvider):
    """Reduce over Timepoint → a t==1 provider (``util.stack`` T→1 SNR fusion). The
    tree-reduce that closes the V2.04 §6b `util.stack` sliver — the whole-series
    reduce streams per tile, folding all T incrementally, instead of eagerly stacking
    the T series in memory per (m, z, c)."""

    _axis = "t"


#: 2D per-plane op — ``(plane_float, m, t, z, c) -> (ny, nx)`` (output geometry may differ).
PlaneFn = Callable[..., np.ndarray]


class PlaneRealizeProvider(StreamProvider):
    """A per-unit lazy realize whose OUTPUT geometry may differ from the base — the
    "lazy units" provider for a non-tileable / geometry-changing op (``util.resample``;
    V2.04 §6b). No tiling, no halo: each output UNIT (a whole plane in 2D, a whole
    ``(Z, Y, X)`` volume in 3D) is computed **once on first touch** from the whole input
    unit, cached (volumes as per-z planar slabs), then windows are served from it.
    Strictly better than realizing ALL of 6-D up front — a Viewer scrubbing one
    ``(m, t, z, c)`` computes only that unit."""

    plane_unit = True                    # a window is no cheaper than the plane it is in

    def __init__(self, base: TileProvider, out_axes: AxisSizes, *,
                 plane_fn: Optional[PlaneFn] = None,
                 volume_fn: Optional[VolumeFn] = None, is_volume: bool = False,
                 fp: str, cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self.axes = out_axes
        self._plane_fn = plane_fn
        self._volume_fn = volume_fn
        self._is_volume = bool(is_volume)
        # in 3D the unit IS the whole volume, so parallel consumers must group by
        # (m,t,c) rather than fan out per z (see StreamProvider.volume_unit)
        self.volume_unit = bool(is_volume)
        self.cum_halo = 0        # a realized-unit level is a window-growth cut point
        if self._is_volume and volume_fn is None:
            raise ValueError("PlaneRealizeProvider(is_volume=True) needs a volume_fn")
        if not self._is_volume and plane_fn is None:
            raise ValueError("PlaneRealizeProvider needs a plane_fn")

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level != 0:
            raise ValueError("streaming providers have no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range [0, {self.axes.z}) — an out-of-range "
                             f"read must not trigger a whole-unit compute")
        key = ("p", self._fp, m, t, z, c)
        plane = self._cache.get(key)
        if plane is None:
            plane = self._unit(m, t, z, c)
        return self._window_of(plane, y0, y1, x0, x1)

    def _unit(self, m: int, t: int, z: int, c: int) -> np.ndarray:
        out, bax = self.axes, self._base.axes
        if self._is_volume:
            vin = self._base.get_region_volume(0, m, t, c, 0, bax.z, 0, bax.y, 0, bax.x)
            res = np.asarray(self._volume_fn(np.asarray(vin, dtype=_F), m, t, c),
                             dtype=_F)
            if res.shape != (out.z, out.y, out.x):
                raise ValueError(f"volume op returned {res.shape}, expected "
                                 f"{(out.z, out.y, out.x)}")
            plane = None
            for zi in range(out.z):                # cache per-z planar slabs
                stored = self._cache.put(("p", self._fp, m, t, zi, c), res[zi])
                if zi == z:
                    plane = stored
            return plane
        # 2D: the output plane at (m,t,z,c) is a function of the whole input plane at the
        # SAME (m,t,z,c) — z is unchanged in a 2D op (resample keeps z; normalize/drift too).
        pin = self._base.get_region(0, m, t, z, c, 0, bax.y, 0, bax.x)
        res = np.asarray(self._plane_fn(np.asarray(pin, dtype=_F), m, t, z, c),
                         dtype=_F)
        if res.shape != (out.y, out.x):
            raise ValueError(f"plane op returned {res.shape}, expected {(out.y, out.x)}")
        return self._cache.put(("p", self._fp, m, t, z, c), res)


#: M→1 fusion kernel — ``(read_tile, offsets, canvas_h, canvas_w, tile_h, tile_w) -> (H, W)``.
#: ``read_tile(m)`` returns multipoint ``m``'s whole ``(Y, X)`` plane, as float. The layout
#: is passed in rather than closed over because it is **per pyramid level** (see
#: :class:`MultiViewProvider`).
FuseFn = Callable[..., np.ndarray]


class MultiViewProvider(StreamProvider):
    """The ``MULTI_VIEW`` lazy unit: one output plane is a function of **every**
    multipoint's plane at the same ``(t, z, c)`` — tile stitching / multi-view fusion
    (``util.stitch``). The output has ``m == 1`` and a larger ``(Y, X)`` than the base.

    Why this is not :class:`PlaneRealizeProvider`: that one computes output unit
    ``(m,t,z,c)`` from the base's plane at the **same** ``(m,t,z,c)``, so it can neither
    read across M nor collapse it. This provider reads across M and serves only ``m == 0``.

    **The kernel PULLS its tiles** rather than receiving a list, and that is the whole
    memory argument for the class. A 49-position montage of 2048² tiles is 1.5 GB of
    source planes against a 172 Mpx (1.3 GB float64) canvas; handing the kernel a list
    would hold both at once. With ``read_tile(m)`` the stitcher reads one tile, pastes it,
    and drops it, so the peak is the canvas plus a single tile.

    Caching is per output plane (never per tile), because the unit of work IS the whole
    canvas — no window of it is cheaper to produce than all of it.

    **This is the one streaming provider WITH a pyramid, and it is why a stitched series
    scrubs.** ``StreamProvider.levels = 1`` is right in general (V2.04 §5-1): a kernel
    applied to a downsampled input is not the downsample of its result, so a computed
    pyramid would be a lie. A **paste** is the exception — stitching mean-downsampled
    tiles *is* the mean-downsampled stitch, exactly away from the seams — so this level
    forwards the base's pyramid, scaling the layout by the base's real per-level axis
    ratio (never an assumed ``2**level``, since ``_build_levels`` floors odd sizes).

    That matters because the Viewer never displays the full canvas: ``render_plane_native``
    picks the coarsest level above ~2048 px and stride-decimates. Without a pyramid it had
    to read level 0, so showing one 13106² frame stitched 172 Mpx to display 3.5 Mpx —
    **measured at 0.92 s per frame for overwrite and 2.9 s for feather**, on every frame
    the user scrubbed to. Serving level 3 instead is the same picture for 1/64th of the
    work.

    Two honest limits on the coarse levels, both invisible where they are used and both
    the reason level 0 stays exact:

    * a seam pixel at level L is one tile's own downsample rather than a blend of two,
      and the scaled offsets round to the coarse grid (≤ 2**(L-1) fine px of seam
      movement — at level 3 that is 4 px in 13106, under half a displayed pixel);
    * a coarse level exists only if the BASE has one. A ``B2ndProvider`` straight from
      ingest carries three; a computed intermediate (an enhancement between the load and
      the stitch) carries none, so that chain falls back to level 0 and its full cost.

    Neither can reach a measurement: every node compute reads ``get_region(0, …)``, and
    level > 0 is read only by the Viewer's display path."""

    #: Emphatically so — and this is the *only* level where the flag guards a cliff rather
    #: than a factor of two. The windowed path below is not merely uncached, it re-reads
    #: every source tile that overlaps the window, so a consumer that walks a 26×26 output
    #: grid pays for the whole mosaic 676 times over.
    plane_unit = True

    def __init__(self, base: TileProvider, out_axes: AxisSizes, fuse_fn: FuseFn,
                 offsets: Tuple[Tuple[int, int], ...], *,
                 fp: str, cache: TileCache) -> None:
        super().__init__(base, fp=fp, cache=cache)
        self.axes = out_axes
        self._fuse = fuse_fn
        self.cum_halo = 0        # a realized-unit level is a window-growth cut point
        # Per-level plan: level 0 is EXACTLY what the compute resolved (so the payload
        # matches the node's own layout); coarser levels rescale it off the base's real
        # axis ratio, so a floored odd size cannot drift the tiles apart.
        self.levels = max(1, int(getattr(base, "levels", 1)))
        self._plan: Dict[int, Tuple[AxisSizes, Tuple[Tuple[int, int], ...]]] = {
            0: (out_axes, tuple((int(a), int(b)) for a, b in offsets))}
        b0 = base.level_axes(0)
        for lv in range(1, self.levels):
            bl = base.level_axes(lv)
            sy, sx = bl.y / float(b0.y), bl.x / float(b0.x)
            offs = tuple((int(round(oy * sy)), int(round(ox * sx)))
                         for oy, ox in self._plan[0][1])
            self._plan[lv] = (replace(out_axes,
                                      y=max(o[0] for o in offs) + bl.y,
                                      x=max(o[1] for o in offs) + bl.x), offs)

    def level_axes(self, level: int) -> AxisSizes:
        if level not in self._plan:
            raise ValueError(f"level {level} out of range [0, {self.levels})")
        return self._plan[level][0]

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if level not in self._plan:
            raise ValueError(f"level {level} out of range [0, {self.levels})")
        axes, offs = self._plan[level]
        if not (0 <= z < axes.z):
            raise IndexError(f"z={z} out of range [0, {axes.z}) — an out-of-range "
                             f"read must not trigger a whole-canvas stitch")
        if m != 0:
            raise IndexError(f"m={m} out of range [0, 1) — this provider fused every "
                             f"multipoint into a single view")
        key = ("p", self._fp, level, t, z, c)
        plane = self._cache.get(key)
        if plane is not None:
            return self._window_of(plane, y0, y1, x0, x1)

        bl = self._base.level_axes(level)

        def read_tile_at(offsets):
            def read_tile(mi: int) -> np.ndarray:
                return np.asarray(
                    self._base.get_region(level, mi, t, z, c, 0, bl.y, 0, bl.x), dtype=_F)
            return read_tile, offsets

        y0c, y1c = max(0, y0), min(y1, axes.y)
        x0c, x1c = max(0, x0), min(x1, axes.x)
        whole = (y0c == 0 and x0c == 0 and y1c == axes.y and x1c == axes.x)
        if not whole:
            # ── WINDOWED stitch: only the tiles that touch this window ────────────
            # The Viewer's detail-on-demand asks for the visible rect at a FINE level,
            # and level 0 of a 49-tile mosaic is a 172 Mpx / 1.3 GB canvas. Stitching
            # the whole thing to serve a 4096² window would make zooming in cost more
            # than the overview did. `_stitch_plane` accepts offsets outside its canvas
            # (it clips per tile), so shifting every offset by the window origin gives
            # exactly this window — and, because the feather weight travels with the
            # tile, byte-identical to that region of the full canvas.
            if y1c <= y0c or x1c <= x0c:
                return _freeze(np.empty((max(0, y1c - y0c), max(0, x1c - x0c)), dtype=_F))
            shifted = tuple((oy - y0c, ox - x0c) for oy, ox in offs)
            read_tile, shifted = read_tile_at(shifted)
            res = np.asarray(self._fuse(read_tile, shifted, y1c - y0c, x1c - x0c,
                                        bl.y, bl.x), dtype=_F)
            return _freeze(res)

        read_tile, offs = read_tile_at(offs)
        res = np.asarray(
            self._fuse(read_tile, offs, axes.y, axes.x, bl.y, bl.x), dtype=_F)
        if res.shape != (axes.y, axes.x):
            raise ValueError(f"fusion kernel returned {res.shape}, expected "
                             f"{(axes.y, axes.x)} at level {level}")
        plane = self._cache.put(key, res)
        return self._window_of(plane, y0, y1, x0, x1)


class WindowView(TileProvider):
    """A lazy crop — a pure translated sub-extent view (the ``_ChannelView`` pattern
    with offsets, V2.04 §6b). No compute, no cache; kernel ops downstream clip their
    halos at THIS view's extents, which reproduces the eager reflect-at-crop-edge
    behavior exactly (the eager path also filtered the already-cropped array)."""

    levels = 1

    def __init__(self, base: TileProvider, *, z0: int = 0, y0: int = 0, x0: int = 0,
                 axes: AxisSizes) -> None:
        self._base = base
        self._z0, self._y0, self._x0 = int(z0), int(y0), int(x0)
        self.axes = axes
        self.tile = base.tile
        self.depth = getattr(base, "depth", 0) + 1
        self.cum_halo = getattr(base, "cum_halo", 0)
        # A crop adds no compute of its own, so it inherits its base's cost model whole: a
        # window of a cropped plane-unit provider still costs a whole plane underneath.
        self.plane_unit = bool(getattr(base, "plane_unit", False))
        self.volume_unit = bool(getattr(base, "volume_unit", False))
        self._fp = digest("window", base.fingerprint(), (self._z0, self._y0, self._x0),
                          (axes.m, axes.t, axes.z, axes.c, axes.y, axes.x))

    def fingerprint(self) -> tuple:
        return ("window", self._fp)

    def level_axes(self, level: int) -> AxisSizes:
        if level != 0:
            raise ValueError("WindowView has no pyramid (level 0 only)")
        return self.axes

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        if level != 0:
            raise ValueError("WindowView has no pyramid (level 0 only)")
        if not (0 <= z < self.axes.z):
            raise IndexError(f"z={z} out of range for cropped view [0, {self.axes.z}) "
                             f"— translating it would serve pixels OUTSIDE the crop")
        return self._base.read_region(
            0, m, t, z + self._z0, c,
            y0 + self._y0, y1 + self._y0, x0 + self._x0, x1 + self._x0)


# ── realization (sinks: export, debug-verify, oversize fallback) ────────────────

def realize(payload: Any) -> Any:
    """Force a Dataset's lazy image to bytes → an :class:`ArrayProvider`-backed
    Dataset. Non-Datasets, image-less Datasets, and already-realized images pass
    through. Used by ``assert_zone_pure`` (structural fingerprints are equal by
    construction — purity must compare BYTES, V2.04 §3) and export.

    Reads run on a worker pool (V2.14, :func:`nodegraph.parallel.map_units`). Each task
    writes a **disjoint** slice of the pre-allocated ``out`` buffer, so no lock is needed
    for the fill itself — the shared mutable state is the engine's
    :class:`TileCache`, which is now internally locked.

    Task granularity follows the provider's compute unit, not the array's shape: a
    ``volume_unit`` provider is fanned out over ``(m,t,c)`` with z read serially inside,
    because touching any single z computes the whole volume. Fanning out per-z there
    would run the same volume computation Z times at once."""
    if not isinstance(payload, Dataset) or payload.image is None:
        return payload
    prov = payload.image
    if isinstance(prov, ArrayProvider):
        return payload
    ax = prov.axes
    with recursion_headroom(8 * (getattr(prov, "depth", 0) + 2)):
        probe = prov.get_region(0, 0, 0, 0, 0, 0, min(1, ax.y), 0, min(1, ax.x))
        out = np.empty((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=probe.dtype)
        if getattr(prov, "volume_unit", False):
            def read_volume(unit):
                m, t, c = unit
                for z in range(ax.z):
                    out[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

            map_units(read_volume, [(m, t, c) for m in range(ax.m)
                                    for t in range(ax.t) for c in range(ax.c)])
        else:
            def read_plane(unit):
                m, t, z, c = unit
                out[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

            map_units(read_plane, [(m, t, z, c) for m in range(ax.m)
                                   for t in range(ax.t) for z in range(ax.z)
                                   for c in range(ax.c)])
    return payload.with_image(ArrayProvider(out, tile=prov.tile))


__all__ = [
    "TileCache", "StreamProvider", "MapComputeProvider", "VolumeComputeProvider",
    "ZReduceProvider", "TReduceProvider", "PlaneRealizeProvider", "MultiViewProvider",
    "WindowView", "stream_fp", "recursion_headroom", "realize",
]
