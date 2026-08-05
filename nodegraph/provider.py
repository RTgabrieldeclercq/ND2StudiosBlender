"""Lazy tiled image provider (nodegraph v2, Phase 2a — V2.02 §3 + V2.03 §5 D1).

``Dataset.image`` holds a :class:`TileProvider`: the lazy voxel source the pull
engine reads through. The keystone benchmark (2026-07-21, real 6554² ND2 plane)
settled the read granularity: a 512² block ROI costs **1.5–4.8%** of a whole-plane
read → **TILED**; and for 3D, **planar `(1,512,512)` blocks win** — a z-range
subvolume reads in ~11% of whole-volume with per-z blocks vs ~47% with a fat
z-spanning block (blosc2 decompresses whole blocks), so **block = one 2D tile per
z**, and a subvolume is *gathered* from planar blocks across the z-range.

The contract (V2.03 §5 D1): ``get_tile``/``get_region`` serve TILEABLE / WHOLE_PLANE
pulls at a single z; ``get_subvolume``/``get_region_volume`` serve the WHOLE_VOLUME
(and WHOLE_SERIES z-range) pulls the 2D/3D toggle's 3D mode needs — realized whole
once and memoized as one payload upstream (V2.02 §7b), then sliced per display tile.

Concrete providers implement :meth:`TileProvider.read_region` (a single-z 2D window)
and report ``axes``/``levels``/``tile``; the base derives every other read from it.

* :class:`SyntheticProvider` — a deterministic formula (numpy only; for tests).
* :class:`B2ndProvider` — a Blosc2 b2nd store with planar blocks (``blosc2`` is
  **lazily imported**, so the core and the synthetic provider need no blosc2). ND2
  ingest (ND2 → 6-D numpy) is an app/ingest-layer concern that feeds
  :meth:`B2ndProvider.from_array`; ``nodegraph`` itself stays nd2-free.
* :class:`ArrayProvider` — an already-realized 6-D array (what a realizing node returns).
* :class:`FrameSubsetProvider` — a lazy VIEW of another provider restricted to chosen
  ``m``/``t``/``z`` indices (each axis picks independently, so the view is their cross
  product). It scopes a whole run to those planes — the GUI's frame-selection
  troubleshooting mode — without touching the graph. :class:`FrameSliceProvider` is the
  one-frame case of it.

Qt-free; numpy at the core, blosc2 optional.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes
from nodegraph.parallel import map_units


class TileProvider(ABC):
    """The lazy voxel-source contract. Subclasses set ``axes``/``levels``/``tile``
    and implement :meth:`read_region`; the base derives tile/volume reads."""

    axes: AxisSizes
    levels: int = 1
    tile: int = 512

    # ── the one primitive concrete providers implement ────────────────────────
    @abstractmethod
    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """A single-z 2D window ``[y0:y1, x0:x1]`` at ``level`` for ``(m,t,z,c)``."""

    # ── multiscale geometry ────────────────────────────────────────────────────
    def level_axes(self, level: int) -> AxisSizes:
        """Axis sizes at ``level`` (spatial pyramid — y,x halve per level)."""
        f = 1 << level
        ax = self.axes
        return replace(ax, y=max(1, ax.y // f), x=max(1, ax.x // f))

    def tiles_per_plane(self, level: int = 0) -> tuple:
        ax = self.level_axes(level)
        return (-(-ax.y // self.tile), -(-ax.x // self.tile))   # ceil-div (ny, nx)

    def fingerprint(self) -> tuple:
        """A memo identity for this provider's content, folded into a Dataset's
        output-fingerprint (so datasets differing only by image don't collide/dedup).
        The base is **structural** (type + geometry) — correct for a deterministic
        provider like :class:`SyntheticProvider`; a content-backed provider overrides
        it (see :class:`ArrayProvider`). A distinct on-disk source that is only
        structurally identical is the deferred provider-identity item (review #3)."""
        return ("provider", type(self).__name__,
                (self.axes.m, self.axes.t, self.axes.z, self.axes.c,
                 self.axes.y, self.axes.x), self.levels, self.tile)

    @property
    def version(self) -> Any:
        """The provider's data identity folded into a **source** node's recipe key
        (C5 / review #3): the engine reads ``prov.version`` into ``__provider_version__``
        so two providers with different data on the same source node do not collide on
        one cached payload. Defaults to :meth:`fingerprint` — structural for a
        deterministic provider (:class:`SyntheticProvider`), content for
        :class:`ArrayProvider` (it overrides ``fingerprint``). A disk-backed provider
        (the future on-disk :class:`B2ndProvider`) overrides this to fold in the file's
        ``mtime_ns``/id so an in-place file change invalidates the memo."""
        return self.fingerprint()

    # ── derived reads (V2.03 §5 D1) ────────────────────────────────────────────
    def get_region(self, level: int, m: int, t: int, z: int, c: int,
                   y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        ax = self.level_axes(level)
        return self.read_region(level, m, t, z, c,
                                max(0, y0), min(y1, ax.y), max(0, x0), min(x1, ax.x))

    def get_tile(self, level: int, m: int, t: int, z: int, c: int,
                 iy: int, ix: int) -> np.ndarray:
        """One block at grid position ``(iy, ix)`` — clipped at the plane edge."""
        ax = self.level_axes(level)
        y0, x0 = iy * self.tile, ix * self.tile
        if y0 >= ax.y or x0 >= ax.x or iy < 0 or ix < 0:
            raise IndexError(f"tile ({iy},{ix}) out of range for level {level} "
                             f"{(ax.y, ax.x)} tile={self.tile}")
        return self.read_region(level, m, t, z, c, y0, min(y0 + self.tile, ax.y),
                                x0, min(x0 + self.tile, ax.x))

    def get_subvolume(self, level: int, m: int, t: int, c: int,
                      z0: int, z1: int, iy: int, ix: int) -> np.ndarray:
        """A ``(z1-z0, block_y, block_x)`` brick — planar blocks gathered across the
        z-range (the benchmark-preferred path; not a MIP). An empty range returns a
        ``(0, block_y, block_x)`` array (review #12: ``np.stack([])`` would crash)."""
        if z1 <= z0:
            probe = self.get_tile(level, m, t, 0, c, iy, ix)   # z=0 always in range
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_tile(level, m, t, z, c, iy, ix)
                         for z in range(z0, z1)], axis=0)

    def get_region_volume(self, level: int, m: int, t: int, c: int,
                          z0: int, z1: int, y0: int, y1: int, x0: int, x1: int
                          ) -> np.ndarray:
        """An arbitrary ``(Z,Y,X)`` ROI (composes single-z region reads); an empty
        range returns a ``(0, ...)`` array (review #12)."""
        if z1 <= z0:
            probe = self.get_region(level, m, t, 0, c, y0, y1, x0, x1)
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_region(level, m, t, z, c, y0, y1, x0, x1)
                         for z in range(z0, z1)], axis=0)


# ── synthetic provider (numpy only; deterministic — for tests) ────────────────

class SyntheticProvider(TileProvider):
    """A provider whose voxels are a cheap deterministic function of their address,
    so a window read is exact and O(window) — ideal for testing the read contract.
    Levels are **stride-decimated** (level ``l`` samples every ``2**l``)."""

    def __init__(self, axes: AxisSizes, *, tile: int = 512, levels: int = 1) -> None:
        self.axes = axes
        self.tile = tile
        self.levels = levels

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        s = 1 << level
        yy = (np.arange(y0, y1, dtype=np.int64) * s)[:, None]
        xx = (np.arange(x0, x1, dtype=np.int64) * s)[None, :]
        val = (yy * 7 + xx * 3 + z * 131 + c * 17 + t * 19 + m * 23) & 0xFFF
        return np.broadcast_to(val, (y1 - y0, x1 - x0)).astype(np.uint16)


# ── Blosc2 b2nd provider (planar blocks; blosc2 lazily imported) ──────────────

#: Target bytes per b2nd **chunk** (the on-disk framing unit, not the read unit).
#: Bigger chunks amortize blosc2's per-chunk framing, which is what capped the ingest
#: write at 55 MB/s; the cap keeps the one contiguous slab the progress-reporting fill
#: loop holds bounded, so a >50 GB ingest still never carries a second copy of the
#: volume. 128 MiB sits on the flat part of the measured curve (67 MiB → 517 MB/s,
#: 268 MiB → 630 MB/s).
_CHUNK_TARGET_BYTES = 128 << 20


#: The compression settings **every** pyramid store in this project is written with — the
#: ingest (:meth:`B2ndProvider._build_levels`), the pyramid repair
#: (:meth:`B2ndProvider.ensure_levels`) and the Dock's checkpoint
#: (:func:`nodegraph.checkpoint._write_image`). One definition because it was three
#: identical literals, and the third had already drifted in effect: none of them named
#: ``clevel``, so all three inherited blosc2's dataclass default.
#:
#: **``clevel=1`` is explicit, and the explicitness is the point.** blosc2 4.9.1's
#: ``CParams.clevel`` defaults to **5** (``blosc2/storage.py``) while its own docstring a few
#: lines above claims "Default is 1" — so omitting the key does not get you the cheap setting
#: it looks like it gets you. Measured at this project's write geometry, ZSTD+BITSHUFFLE on
#: 12-bit-in-uint16 image data: clevel 1 ≈ 780 MB/s at ratio 2.07, clevel 5 ≈ 530 MB/s at
#: ratio 2.10, clevel 9 ≈ 91 MB/s at ratio 2.14. So level 5 was costing ~1.5× the write time
#: for ~1% of the ratio, and 9 is a 5.8× regression that must never be offered as a choice.
#:
#: BITSHUFFLE stays: this repo's own granularity sweep
#: (``scripts/_bench_provider_granularity.py``) measured it against SHUFFLE on a real 6554²
#: plane and took the ratio, and the ROI-latency argument for changing it was checked and
#: does not hold at the Viewer's actual read sizes.
PYRAMID_CPARAMS: dict = {"codec": None, "clevel": 1, "filters": None}


def pyramid_cparams() -> dict:
    """A fresh copy of :data:`PYRAMID_CPARAMS` with the blosc2 enums resolved (they need
    the module imported, which the core does lazily). Returned as a NEW dict every call —
    blosc2 stores what it is handed, and a shared mutable would let one store's tweak
    follow every later one."""
    import blosc2
    return {"codec": blosc2.Codec.ZSTD, "clevel": 1,
            "filters": [blosc2.Filter.BITSHUFFLE]}


#: ``vlmeta`` key each pyramid level is stamped with once it is **fully written**
#: (V2.19). A b2nd array declares its full shape at ``blosc2.empty`` time, so an ingest
#: killed partway (crash, cancel, OOM, full disk) leaves a file whose geometry looks right
#: and whose unwritten chunks read as **zeros** — silently wrong pixels, and a torn pyramid
#: that :meth:`B2ndProvider.open` used to accept as a complete store of fewer levels. The
#: marker is what makes "complete" a fact on disk rather than an assumption.
_LEVEL_META = "nodegraph_level"


def _plane_mean_2x(plane: np.ndarray) -> np.ndarray:
    """One 2-D plane mean-pooled 2× — **the** pyramid arithmetic, in one place.

    Every pyramid path routes its per-plane reduction through this function so the
    streamed build (:meth:`B2ndProvider._append_level`), the in-RAM reference
    (:func:`_mean_downsample_2x`) and :mod:`nodegraph.checkpoint` cannot drift apart."""
    y, x = plane.shape[-2], plane.shape[-1]
    hy, hx = y // 2, x // 2
    return plane[..., :hy * 2, :hx * 2].reshape(
        *plane.shape[:-2], hy, 2, hx, 2).mean(axis=(-3, -1)).astype(plane.dtype)


def _mean_downsample_2x(a: np.ndarray) -> np.ndarray:
    """Mean-pool the trailing (Y,X) by 2× (intensity pyramid). Returns ``a`` if a
    spatial axis is < 2 (cannot halve further).

    **The in-RAM reference form.** The pyramid writer no longer calls this — it streams
    level *l* out of level *l-1*'s store one z-slab at a time (:meth:`
    B2ndProvider._append_level`), because materializing a whole downsampled level in RAM
    is what killed the 84.7 GB ingest that motivated V2.19: this function's output for
    level 1 of that series is 21 GB, allocated *while* the 84.7 GB source is still held.
    It stays as the reference the selftest compares the streamed pyramid against, and for
    bare plane/volume callers.

    Computed **per plane on a worker pool** for a 6-D input (V2.14). Two reasons, and the
    memory one matters more than the speed one: ``mean`` over a whole uint16 volume
    promotes to float64, so the one-shot form allocated a float64 intermediate the size of
    the *entire series* (4× the input's bytes for uint16) before casting back. Per-plane
    keeps that intermediate at one plane. Once the b2nd write stopped being the
    bottleneck this pass became ~58% of an ingest, so it is also now worth the fan-out.

    The result is bit-identical to the one-shot form: each output plane is a reduction over
    its own 2×2 neighbourhoods only, so splitting by plane changes no arithmetic."""
    y, x = a.shape[-2], a.shape[-1]
    if y < 2 or x < 2:
        return a
    hy, hx = y // 2, x // 2
    if a.ndim != 6:                       # keep the simple path for a bare plane/volume
        return _plane_mean_2x(a)
    m, t, z, c = a.shape[:4]
    out = np.empty((m, t, z, c, hy, hx), dtype=a.dtype)

    def one(unit):
        im, it, iz, ic = unit
        out[im, it, iz, ic] = _plane_mean_2x(a[im, it, iz, ic])

    map_units(one, [(im, it, iz, ic) for im in range(m) for it in range(t)
                    for iz in range(z) for ic in range(c)])
    return out


class B2ndProvider(TileProvider):
    """A Blosc2 b2nd-backed provider. Each pyramid level is a 6-D ``(M,T,Z,C,Y,X)``
    b2nd array with **planar blocks** ``(1,1,1,1,tile,tile)`` (the benchmark verdict);
    a window read decompresses only the touched planar blocks."""

    def __init__(self, arrays: List, axes: AxisSizes, *, tile: int = 512,
                 urlpath: Optional[str] = None, mtime_ns: Optional[int] = None) -> None:
        self._arrays = arrays
        self.axes = axes
        self.tile = tile
        self.levels = len(arrays)
        self._urlpath = urlpath          # set for a disk-backed store (else in-memory)
        self._mtime_ns = mtime_ns        # store mtime → cheap identity for a disk store

    def level_axes(self, level: int) -> AxisSizes:
        m, t, z, c, y, x = self._arrays[level].shape       # exact stored geometry
        return AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        return np.asarray(self._arrays[level][m, t, z, c, y0:y1, x0:x1])

    def fingerprint(self) -> tuple:
        """Content/data identity (C5 / review). A b2nd store is content-bearing, so the
        base *structural* identity would collide two stores of equal geometry but
        different pixels (a wrong memo hit + blob-dedup aliasing). A **disk** store folds
        in its path + ``mtime_ns`` (cheap — an in-place file edit re-stamps mtime → new
        identity; C4/C5 disk path). An **in-memory** store hashes the level-0 payload
        (O(level0) decompress) — **memoized on the instance** (C1: streaming chains
        re-read the base fingerprint per provider construction; the store is immutable
        for this provider's lifetime, so hash once)."""
        if self._urlpath is not None:
            m, t, z, c, y, x = self._arrays[0].shape
            return ("b2nd-disk", str(self._urlpath), self._mtime_ns,
                    (m, t, z, c, y, x), str(self._arrays[0].dtype))
        fp = getattr(self, "_fp", None)
        if fp is not None:
            return fp
        import hashlib
        a0 = np.ascontiguousarray(self._arrays[0][...])
        h = hashlib.blake2b(a0.tobytes(), digest_size=16).hexdigest()
        self._fp = ("b2nd", a0.shape, str(a0.dtype), self.levels, self.tile, h)
        return self._fp

    # ── ingest / persistence (blosc2 lazily imported; nd2-free) ─────────────────
    @staticmethod
    def _pyramid_shapes(shape: Tuple[int, ...], levels: int) -> List[Tuple[int, ...]]:
        """The 6-D shape of each pyramid level, stopping early when Y or X can no longer
        be halved (mirrors :func:`_mean_downsample_2x`'s guard, so the writer, the byte
        budget and :meth:`ensure_levels` all agree on how tall the pyramid can be)."""
        m, t, z, c, y, x = shape
        out = [(m, t, z, c, y, x)]
        for _ in range(max(1, levels) - 1):
            if y < 2 or x < 2:
                break
            y, x = y // 2, x // 2
            out.append((m, t, z, c, y, x))
        return out

    #: Smallest chunk a write should aim for, when the axis it normally batches over (Z)
    #: cannot supply one. Deliberately far below :data:`_CHUNK_TARGET_BYTES`: this is a
    #: FLOOR that exists to escape per-chunk overhead, not a target, and every byte of chunk
    #: past it costs full-plane read latency for nothing. 32 MiB is where the measured write
    #: curve flattens on this pipeline — a 2048² uint16 plane (8 MiB) batches 4 timepoints
    #: into one chunk and the write goes from 143 to 552 MiB/s, while a full-plane read
    #: slows ~17%; a 7168² plane is already 98 MiB, clears the floor on its own, and is
    #: therefore left framed one plane per chunk exactly as before.
    _CHUNK_FLOOR_BYTES = 32 << 20

    @staticmethod
    def _store_kwargs(shape: Tuple[int, ...], itemsize: int, *, tile: int,
                      cparams: dict, urlpath: Optional[str], level: int,
                      batch_thin_z: bool = False) -> dict:
        """The b2nd geometry for one level. **The two geometries do DIFFERENT jobs**
        (V2.14):

        ``blocks`` is the decompression granularity — the keystone verdict, one 2D tile per
        z, and the whole reason a 512² ROI costs a few percent of a plane. It is the read
        contract and does not move.

        ``chunks`` is only the on-disk framing, and the shipped (2·tile)² was far too
        small: measured on this pipeline, a 1 GiB uint16 series wrote at 55 MB/s with
        ``(1,1,1,1,1024,1024)`` versus 630 MB/s at ``(1,1,32,1,2048,2048)`` — an 11.5×
        ingest speedup for byte-identical compressed output (561.4 vs 561.5 MiB) and a 9%
        tile-read cost. The write was never codec- or disk-bound: every codec landed at
        53–58 MB/s, while the NVMe underneath absorbs 952 MB/s.

        ``batch_thin_z`` extends that verdict to the case it cannot reach. ``cz`` is capped
        by **Z**, so a 2D or thin-Z series pins one plane per chunk no matter how small that
        plane is — and thin Z is not an edge case here: a timelapse, a Z-projection, a
        stitched mosaic and a channel merge are all ``z == 1``, i.e. exactly the transformed
        data a Dock gets pointed at. With the flag on, a chunk still under
        :data:`_CHUNK_FLOOR_BYTES` batches over **T** until it clears the floor.

        Opt-in rather than always-on because the two callers want different things. A Dock
        checkpoint is written once and read whole-plane, so it takes the 3.9× write for a
        ~17% full-plane read. The ingest store's read profile is load-bearing for every live
        chain and has never been measured under T-batching, so it keeps the framing it was
        benchmarked with."""
        import os
        _m, t, z, _c, y, x = shape
        by, bx = min(tile, y), min(tile, x)              # READ granularity — unchanged
        plane_bytes = max(1, y * x * itemsize)
        cz = max(1, min(z, int(_CHUNK_TARGET_BYTES // plane_bytes)))
        ct = 1
        if batch_thin_z:
            slab = max(1, cz * plane_bytes)
            if slab < B2ndProvider._CHUNK_FLOOR_BYTES:
                ct = max(1, min(int(t), int(B2ndProvider._CHUNK_FLOOR_BYTES // slab)))
        kw: dict = {"chunks": (1, ct, cz, 1, y, x), "blocks": (1, 1, 1, 1, by, bx),
                    "cparams": cparams}
        if urlpath is not None:
            kw["urlpath"] = os.path.join(urlpath, f"level_{level}.b2nd")
            kw["mode"] = "w"
        return kw

    @staticmethod
    def _mark(arr: Any, level: int, complete: bool) -> None:
        """Stamp a level's write state (see :data:`_LEVEL_META`).

        Called **twice** per level: ``complete=False`` immediately after ``blosc2.empty``
        declares the geometry, and ``complete=True`` once the last chunk is in. Marking the
        start is what makes a tear detectable at all — stamping only on success would leave
        a half-written level indistinguishable from a pre-V2.19 one, and those are trusted
        (see :meth:`_level_state`), so a crashed ingest would go on serving zeros.

        Best-effort: a blosc2 without writable ``vlmeta`` must not fail an otherwise good
        ingest. There the store simply reads back as legacy, exactly as before V2.19."""
        try:
            arr.vlmeta[_LEVEL_META] = {"complete": bool(complete), "level": int(level),
                                       "shape": [int(v) for v in arr.shape]}
        except Exception:  # noqa: BLE001 — the marker is metadata, never the data
            pass

    @staticmethod
    def _level_state(arr: Any) -> str:
        """``"complete"`` / ``"torn"`` / ``"legacy"`` for one opened level array.

        ``legacy`` — no marker at all — means *written before V2.19* and is **trusted**.
        Those stores are overwhelmingly fine, and the alternative (distrusting them) would
        force a multi-hour re-ingest of every file already on disk to fix a fault we have no
        evidence of. The one case it cannot catch is a pre-V2.19 ingest torn mid-level, and
        the honest reason that is acceptable: the only tell would be a tail of zero planes,
        which a real fluorescence z-stack has too, so acting on it would corrupt good stores
        to protect bad ones. Everything written from V2.19 on is marked before its first
        byte, so a tear there is a fact, not an inference."""
        try:
            meta = arr.vlmeta.get(_LEVEL_META)
        except Exception:  # noqa: BLE001 — no vlmeta support ⇒ indistinguishable from none
            return "legacy"
        if meta is None:
            return "legacy"
        if isinstance(meta, dict) and meta.get("complete") is True \
                and list(meta.get("shape", arr.shape)) == [int(v) for v in arr.shape]:
            return "complete"
        return "torn"

    @staticmethod
    def _blank_tail(arr: Any) -> Optional[Tuple[int, int]]:
        """``(chunks_written, chunks_total)`` when ``arr``'s never-written chunks form a
        **pure trailing run**, else ``None``. This is the marker-free tear test — the one
        that catches a store written *before* :data:`_LEVEL_META` existed (V2.20).

        Why it is needed at all: :meth:`_level_state` trusts an unmarked level, and the
        lab's 84.7 GB series turned out to have a *legacy* ``level_0`` holding only
        7 899 of 40 320 chunks — positions m≥2 read back as solid zeros, and every layer
        above (the pyramid repair included) faithfully reproduced them. "Legacy is fine"
        was an assumption, and this is the evidence that replaced it.

        Cheap: ``iterchunks_info`` reads each chunk's 32-byte header and decompresses
        nothing (2.7 s for those 40 320 chunks). blosc2 stores a chunk that was never
        written as a *special* run-length value rather than as data, so "written" is a
        fact on disk, not an inference from the pixels.

        **A pure tail, and nothing weaker.** blosc2 also files a chunk that was genuinely
        written and happens to be constant as that same special value, so a blank chunk
        alone proves nothing: a segmentation mask is legitimately blank almost everywhere.
        A write that stops partway, though, leaves blanks *only* at the end — so a
        trailing run with a fully-written body is the tear's signature and a scattered
        pattern is real sparse data. The rule errs toward silence: a torn store whose
        written part contains one constant chunk reads as sparse and is not flagged, which
        is the right way round for a test whose verdict costs the user a re-ingest."""
        try:
            blank = [i.special.name != "NOT_SPECIAL"
                     for i in arr.schunk.iterchunks_info()]
        except Exception:  # noqa: BLE001 — no census available ⇒ no verdict, not a fault
            return None
        n = len(blank)
        if not n or not blank[-1]:
            return None                      # the last chunk holds data ⇒ nothing torn
        first = n - 1
        while first > 0 and blank[first - 1]:
            first -= 1
        if any(blank[:first]):
            return None                      # blanks through the body ⇒ sparse, not torn
        return (first, n)

    def level_state(self, level: int = 0) -> str:
        """:meth:`_level_state` for an already-open level — ``"complete"`` / ``"torn"`` /
        ``"legacy"`` — so a caller can ask what a store's marker says without re-opening
        it (the ingest layer gates its density check on ``"legacy"``)."""
        return self._level_state(self._arrays[level])

    def blank_tail(self, level: int = 0) -> Optional[Tuple[int, int]]:
        """:meth:`_blank_tail` for an already-open level: ``(chunks_written,
        chunks_total)`` when this level stops partway through, else ``None``."""
        try:
            return self._blank_tail(self._arrays[level])
        except (IndexError, AttributeError):    # no such level / not a b2nd array
            return None

    @classmethod
    def _append_level(cls, prev: Any, level: int, *, tile: int, cparams: dict,
                      urlpath: Optional[str], bump: Optional[Callable[[int], None]] = None,
                      batch_thin_z: bool = False) -> Optional[Any]:
        """Build level ``level`` by **streaming out of ``prev``** (level ``level-1``'s
        b2nd array) one z-slab at a time, and return it — or ``None`` when ``prev`` can no
        longer be halved.

        This is the V2.19 repair. The old writer downsampled the whole previous level in
        RAM (``_mean_downsample_2x``) and handed the result to the store: for the lab's
        84.7 GB ND2 that is a 21 GB allocation made *while* the 84.7 GB source array is
        still live, and it is why that file's store on disk has a complete ``level_0`` and
        no pyramid at all — the ingest died between levels, leaving no trace that it had.
        Streaming holds one slab of each side instead (bounded by
        :data:`_CHUNK_TARGET_BYTES`), so the pyramid's memory cost no longer scales with
        the series at all, and level *l* can be rebuilt from level *l-1* long after the
        ND2 is gone.

        A slab is chunk-aligned on both sides and never crosses ``(M,T,C)``, so this writes
        exactly the bytes the one-shot form did. The per-plane reduction is
        :func:`_plane_mean_2x` — the same expression, so the result is bit-identical.

        **The slab's planes reduce on the worker pool** (V2.20), exactly as the in-RAM
        reference :func:`_mean_downsample_2x` already did. Streaming was never the reason
        to give that up, but the first cut of this function looped the planes serially and
        it cost more than everything else in an ingest put together: measured on a 1.64 GiB
        slice of the lab's 640 series (24 cores), level 1 alone took **6.0 s** against 2.2 s
        to compress level 0 and 2.1 s to decode the ND2 — a quarter of the data for three
        times the time, because one core was mean-pooling 840 planes while 23 sat idle.
        Order is irrelevant here (each plane writes its own row of ``out``), so this is the
        cheapest possible use of the pool: pure per-plane compute, no fold.

        **The block is ``(t-batch, z-slab)``, not one ``(m,t,c)`` frame's z-slab**, and on a
        thin-Z series that distinction is most of the runtime. With ``z == 1`` the loop used to
        degenerate to one iteration per ``(m,t,c)`` handling a SINGLE plane — so it paid a
        worker-pool dispatch over ``range(1)``, a fan-out of one, and wrote one small chunk,
        160 times over for a 160-plane series. Measured on a 1.25 GiB 2D series, the two coarse
        levels together cost 17.4 s against 3.3 s for all of level 0. Batching over T (via
        ``batch_thin_z``, the same floor :meth:`_store_kwargs` applies) gives the pool a real
        unit count and the store a real chunk. ``ct == 1`` reproduces the old loop exactly,
        which is what keeps the ingest path byte-identical."""
        import blosc2
        m, t, z, c, y, x = prev.shape
        if y < 2 or x < 2:
            return None
        shape = (m, t, z, c, y // 2, x // 2)
        dtype = prev.dtype
        kw = cls._store_kwargs(shape, dtype.itemsize, tile=tile, cparams=cparams,
                               urlpath=urlpath, level=level, batch_thin_z=batch_thin_z)
        dst = blosc2.empty(shape, dtype=dtype, **kw)
        cls._mark(dst, level, False)                 # declared, not yet written
        ct, cz = int(kw["chunks"][1]), int(kw["chunks"][2])
        for im, ic in np.ndindex(m, c):
            for t0 in range(0, t, ct):
                t1 = min(t0 + ct, t)
                for z0 in range(0, z, cz):
                    z1 = min(z0 + cz, z)
                    src = np.asarray(prev[im, t0:t1, z0:z1, ic, :, :])
                    out = np.empty((t1 - t0, z1 - z0, shape[4], shape[5]), dtype=dtype)

                    def one(unit, _s=src, _o=out) -> None:
                        j, k = unit
                        _o[j, k] = _plane_mean_2x(_s[j, k])

                    map_units(one, [(j, k) for j in range(t1 - t0)
                                    for k in range(z1 - z0)])
                    dst[im:im + 1, t0:t1, z0:z1, ic:ic + 1, :, :] = \
                        out.reshape(1, t1 - t0, z1 - z0, 1, shape[4], shape[5])
                    if bump is not None:
                        bump((t1 - t0) * (z1 - z0))
        cls._mark(dst, level, True)
        return dst

    @classmethod
    def _build_levels(cls, vol6d: Any, *, tile: int, levels: int,
                      cparams: Optional[dict], urlpath: Optional[str] = None,
                      progress: Optional[Callable[[float], None]] = None) -> List:
        """Build ``levels`` planar-block b2nd pyramid arrays from a 6-D volume. If
        ``urlpath`` is a directory, each level persists to ``level_<l>.b2nd`` there;
        otherwise the arrays are in-memory.

        ``vol6d`` is any 6-D array-like that reports ``shape``/``dtype`` and supports
        numpy-style slicing — a realized numpy array, or a **lazy** one (an nd2
        ``to_dask()`` view). The level-0 loop below already reads exactly one z-slab per
        write, so handing it a lazy source makes the whole ingest streaming: peak memory
        is one slab rather than the entire series, and an 84.7 GB ND2 no longer needs an
        84.7 GB ``np.empty`` to exist before its first byte is compressed (V2.20).

        Level 0 is written from ``vol6d``; every level above it is streamed out of the
        level below (:meth:`_append_level`) rather than downsampled in RAM, and each is
        stamped complete only once fully written (:data:`_LEVEL_META`).

        ``progress``, when given, is called with a monotonic ``0→1`` fraction of the
        PLANES written so far across the WHOLE pyramid (level *l* carries ~1/4^l of level
        0 by bytes, but one plane per source plane, which is what the loops step), so a
        caller drives one determinate bar instead of one per level."""
        import blosc2
        if vol6d.ndim != 6:
            raise ValueError(f"expected a 6-D (M,T,Z,C,Y,X) array, got {vol6d.shape}")
        cparams = cparams or pyramid_cparams()
        shapes = cls._pyramid_shapes(vol6d.shape, levels)
        # The pyramid's plane budget up front (predicting the halvings is free) so the
        # fraction stays monotonic across levels rather than restarting at 0 on each.
        per_level = int(np.prod(vol6d.shape[:4]))
        total = max(1, per_level * len(shapes))
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += int(n)
            if progress is not None:
                progress(min(1.0, done / total))

        kw = cls._store_kwargs(shapes[0], np.dtype(vol6d.dtype).itemsize, tile=tile,
                               cparams=cparams, urlpath=urlpath, level=0)
        m, t, z, c = vol6d.shape[:4]
        cz = int(kw["chunks"][2])
        # A determinate bar needs sub-level granularity, and a chunk spans `cz` planes and
        # never crosses (M,T,C) — so filling ONE z-slab at a time is exactly chunk-aligned
        # (the same on-disk bytes as `blosc2.asarray`) and reports honestly. It also holds
        # one slab contiguous at a time instead of a whole second copy of the volume, which
        # is what made a >50 GB ingest thrash.
        arr0 = blosc2.empty(vol6d.shape, dtype=vol6d.dtype, **kw)
        cls._mark(arr0, 0, False)                    # declared, not yet written
        for im, it, ic in np.ndindex(m, t, c):
            for z0 in range(0, z, cz):
                z1 = min(z0 + cz, z)
                arr0[im:im + 1, it:it + 1, z0:z1, ic:ic + 1, :, :] = \
                    np.ascontiguousarray(vol6d[im:im + 1, it:it + 1, z0:z1, ic:ic + 1])
                bump(z1 - z0)
        cls._mark(arr0, 0, True)

        arrays = [arr0]
        for lvl in range(1, len(shapes)):
            nxt = cls._append_level(arrays[-1], lvl, tile=tile, cparams=cparams,
                                    urlpath=urlpath, bump=bump)
            if nxt is None:
                break
            arrays.append(nxt)
        if progress is not None:
            progress(1.0)
        return arrays

    @classmethod
    def from_array(cls, vol6d: np.ndarray, *, tile: int = 512, levels: int = 1,
                   cparams: Optional[dict] = None,
                   progress: Optional[Callable[[float], None]] = None) -> "B2ndProvider":
        """Ingest a ``(M,T,Z,C,Y,X)`` numpy volume into an **in-memory** planar-block
        b2nd store with ``levels`` mean-downsampled pyramid levels. (An ND2 → 6-D numpy
        reader is an app/ingest-layer concern; ``nodegraph`` stays nd2-free.)"""
        arrays = cls._build_levels(vol6d, tile=tile, levels=levels, cparams=cparams,
                                   progress=progress)
        m, t, z, c, y, x = vol6d.shape
        return cls(arrays, AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), tile=tile)

    @classmethod
    def write(cls, vol6d: Any, urlpath: str, *, tile: int = 512, levels: int = 1,
              cparams: Optional[dict] = None,
              progress: Optional[Callable[[float], None]] = None) -> "B2ndProvider":
        """Persist a 6-D volume to an **on-disk** b2nd store at directory ``urlpath``
        (one ``level_<l>.b2nd`` per pyramid level) and return a provider opened over it
        (the C4 disk path — ingest once, then a lazy disk-backed provider). ``progress``
        is forwarded to :meth:`_build_levels` (0→1 over the whole pyramid).

        ``vol6d`` may be **lazy** (see :meth:`_build_levels`); it is deliberately NOT
        coerced with ``np.asarray`` here, since that would realize the whole series in RAM
        and defeat the streaming write."""
        import os
        os.makedirs(urlpath, exist_ok=True)
        cls._build_levels(vol6d, tile=tile, levels=levels, cparams=cparams,
                          urlpath=urlpath, progress=progress)
        return cls.open(urlpath)

    @staticmethod
    def _level_files(urlpath: str) -> List[str]:
        """``level_0.b2nd`` … in index order, requiring a **contiguous run from 0**. A gap
        (``level_0`` + ``level_2``) is a broken store, not a two-level one: ``levels`` is a
        count and every reader indexes by position, so accepting the gap would serve
        level 2's pixels to a request for level 1."""
        import glob
        import os
        found = {}
        for p in glob.glob(os.path.join(urlpath, "level_*.b2nd")):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                found[int(stem.split("_")[1])] = p
            except (IndexError, ValueError):
                continue
        out: List[str] = []
        while len(out) in found:
            out.append(found[len(out)])
        return out

    @classmethod
    def open(cls, urlpath: str) -> "B2ndProvider":
        """Open an on-disk b2nd store written by :meth:`write` — lazy (blocks decompress
        on read). ``version``/``fingerprint`` fold in **level 0's** ``mtime_ns`` so an
        in-place file change invalidates the memo (C5).

        Levels are taken as the leading run that is complete-or-legacy
        (:meth:`_level_state`); a **torn** level above 0 is dropped (the store opens
        shorter and :meth:`ensure_levels` can rebuild it), and a torn level 0 is a hard
        error — its pixels are partly zeros, so serving them would be silent corruption,
        and the caller's answer is to re-ingest from the source file."""
        import blosc2
        import os
        files = cls._level_files(urlpath)
        if not files:
            raise FileNotFoundError(f"no level_*.b2nd store in {urlpath!r}")
        arrays: List[Any] = []
        for lvl, f in enumerate(files):
            a = blosc2.open(f)
            state = cls._level_state(a)
            if state == "torn":
                if lvl == 0:
                    raise ValueError(
                        f"torn b2nd store: {f!r} was never finished writing (its "
                        f"unwritten blocks would read as zeros), so it must be re-ingested "
                        f"from the source file — level 0 is the only level no other level "
                        f"can be rebuilt from. Deleting {urlpath!r} forces that.")
                break                      # a repairable pyramid: open what is sound
            arrays.append(a)
        m, t, z, c, y, x = arrays[0].shape
        blocks = getattr(arrays[0], "blocks", None)
        tile = int(blocks[-1]) if blocks else 512
        # LEVEL 0 ONLY (V2.19). Every compute reads level 0; levels above it are a display
        # convenience. Folding the whole store's newest mtime in meant that *completing the
        # pyramid* — which does not touch a single pixel a compute can see — re-keyed the
        # source and threw away every memoized result downstream of it. On the lab's series
        # that is a 4-minute Deconvolve recomputed to add a thumbnail level.
        mtime_ns = os.stat(files[0]).st_mtime_ns
        return cls(arrays, AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), tile=tile,
                   urlpath=urlpath, mtime_ns=mtime_ns)

    @classmethod
    def ensure_levels(cls, urlpath: str, levels: int, *,
                      cparams: Optional[dict] = None,
                      progress: Optional[Callable[[float], None]] = None
                      ) -> "B2ndProvider":
        """Complete an existing store's pyramid **in place**, from the deepest sound level
        it already has, and return a provider over the result (V2.19).

        This is the repair half of the torn-pyramid fix. A store whose ``level_0`` is
        intact needs no source file to regain its pyramid — level *l* is a pure function of
        level *l-1* — so the 84.7 GB ND2 that produced it need not be read, re-decoded, or
        even still exist. Idempotent: a store that already has ``levels`` sound levels is
        opened and returned untouched.

        Raises whatever :meth:`open` raises for a store that cannot be repaired (no
        ``level_0``, or a torn one) — the caller's fallback there is a real re-ingest."""
        import blosc2
        prov = cls.open(urlpath)                     # raises on an unrepairable store
        want = len(cls._pyramid_shapes(prov._arrays[0].shape, levels))
        if prov.levels >= want:
            return prov
        cparams = cparams or pyramid_cparams()
        per_level = int(np.prod(prov._arrays[0].shape[:4]))
        total = max(1, per_level * (want - prov.levels))
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += int(n)
            if progress is not None:
                progress(min(1.0, done / total))

        last = prov._arrays[-1]
        for lvl in range(prov.levels, want):
            nxt = cls._append_level(last, lvl, tile=prov.tile, cparams=cparams,
                                    urlpath=urlpath, bump=bump)
            if nxt is None:
                break
            last = nxt
        if progress is not None:
            progress(1.0)
        return cls.open(urlpath)


def _picked(idx, n: int) -> tuple:
    """``idx`` as a sorted, de-duplicated, in-range index tuple for an axis of length
    ``n`` — never empty (an empty or fully out-of-range pick degrades to index 0).

    Clamping rather than raising is deliberate: the picks come from a GUI cursor bounded
    by whatever was last displayed, and a graph edit can shrink a source underneath it.
    Showing the nearest real plane beats turning a stale cursor into a failed pull."""
    hi = max(0, int(n) - 1)
    out = sorted({min(max(0, int(i)), hi) for i in idx})
    return tuple(out) if out else (0,)


def subset_index(picks: Sequence[int], value: int) -> int:
    """Where ``value`` — an index in the BASE provider's addressing — lands inside a
    subset built from (sorted) ``picks``: the inverse of the remap
    :meth:`FrameSubsetProvider.read_region` performs.

    A value that was **not** picked resolves to the nearest one that was. That is what
    lets a GUI cursor sit outside the run scope and still address a real plane of the
    payload instead of reading past its end. ``picks`` of ``None``/empty means the axis
    was not subset at all, so the value passes through unchanged."""
    n = len(picks) if picks else 0
    if n == 0:
        return int(value)
    j = bisect_left(picks, int(value))
    if j >= n:
        return n - 1
    if j == 0 or picks[j] == value:
        return j
    return j if (picks[j] - value) < (value - picks[j - 1]) else j - 1


class ChannelMergeProvider(TileProvider):
    """Two acquisitions on one channel axis, placed by absolute stage position and focus —
    ``channel.merge``'s image, and the only provider that reads from two sources.

    A channel index below ``pri_c`` is served from the PRIMARY and one at or above it from the
    SECONDARY. Both are addressed the same way: the output plane has an absolute µm focus (the
    node's :class:`~nodegraph.placement.ZGrid`), and each side answers with **its own plane
    nearest that focus**. Nothing is interpolated in Z — a channel shows a plane its microscope
    actually acquired, or the nearest one it has, which is what makes "walk the merged stack"
    mean something for two files with different Z sampling.

    Laterally the secondary IS re-addressed, because the two files rarely share a pixel size
    (the WellA3 pair are 0.287 and 1.718 µm/px). That resampling is nearest-neighbour through
    :func:`~nodegraph.placement.compose_secondary_plane`, so every value written is a real
    sample of the source file rather than a blend of several — and only the WINDOW being read is
    fetched, at whichever pyramid level suits it.

    **Lazy, which is the whole reason it exists.** ``view.overlay``'s ``resample`` mode does the
    same co-registration eagerly: it allocates the entire ``(M,T,Z,C,Y,X)`` result up front,
    which is 6.1 GiB for sixteen frames of a 7168² mosaic and ~77 GiB for two hundred. Here a
    plane costs a plane.

    Uncovered pixels read ``0``: the secondary genuinely has nothing there, and a gap that reads
    as a gap is the honest rendering (the node reports the coverage fraction per field).
    """

    def __init__(self, primary: TileProvider, secondary: TileProvider, *,
                 axes: AxisSizes, entry: Dict[str, Any], grid: Any,
                 pri_md: Dict[str, Any], pri_axes: AxisSizes,
                 sec_md: Dict[str, Any], sec_axes: AxisSizes) -> None:
        self._p, self._s = primary, secondary
        self.axes = axes
        self._entry = dict(entry)
        self._grid = grid
        self._pri_md, self._pri_axes = dict(pri_md), pri_axes
        self._sec_md, self._sec_axes = dict(sec_md), sec_axes
        self._pri_c = int(pri_axes.c)
        # The pyramid is the PRIMARY's: it defines the lateral grid, so its levels are the ones
        # whose geometry the output shares. The secondary picks its own level per read.
        self.levels = max(1, int(getattr(primary, "levels", 1)))
        self.tile = int(getattr(primary, "tile", 512))
        self._dz = float((self._entry.get("offset_um") or (0.0, 0.0, 0.0))[0])
        # A window is no cheaper than the plane it is in if EITHER source says so, and a stitch
        # says so emphatically: its windowed path re-reads every overlapping tile, uncached, so
        # a consumer walking a 26x26 output grid pays for the whole mosaic 676 times. This flag
        # is the fence that stops that, and not forwarding it would put the cliff back one
        # provider further along.
        self.plane_unit = bool(getattr(primary, "plane_unit", False)
                               or getattr(secondary, "plane_unit", False))
        self.volume_unit = bool(getattr(primary, "volume_unit", False))

    def level_axes(self, level: int) -> AxisSizes:
        base = self._p.level_axes(level)
        return replace(self.axes, y=base.y, x=base.x)

    def fingerprint(self) -> tuple:
        # Content identity of BOTH sources plus the placement — two merges of the same pair with
        # different nudges are different images and must not share a memo entry.
        return ("merge", self._p.fingerprint(), self._s.fingerprint(),
                (self.axes.m, self.axes.t, self.axes.z, self.axes.c),
                self._grid.step_um, self._grid.n, tuple(self._grid.z0_um),
                repr(sorted(self._entry.items(), key=lambda kv: kv[0])))

    def _z_of(self, md: Dict[str, Any], axes: AxisSizes, m: int,
              z_um: Optional[float], *, dz: float = 0.0) -> int:
        from nodegraph.placement import secondary_z_index
        return secondary_z_index(md, axes, int(m), z_um, dz=dz)

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        from nodegraph.placement import compose_secondary_plane, paired_t
        lax = self.level_axes(level)
        z_um = self._grid.plane_um(m, z)
        if int(c) < self._pri_c:
            # The primary's own plane nearest this focus. With no focus log (`z_um is None`)
            # `secondary_z_index` answers 0 for a single plane and the grid is the primary's own
            # index space anyway, so pass the index straight through.
            pz = int(z) if z_um is None else self._z_of(self._pri_md, self._pri_axes, m, z_um)
            pz = min(max(0, pz), int(self._pri_axes.z) - 1)
            return np.asarray(self._p.get_region(level, m, t, pz, int(c), y0, y1, x0, x1))

        k = int(c) - self._pri_c
        h, w = max(0, y1 - y0), max(0, x1 - x0)
        out = np.zeros((h, w), dtype=np.float32)
        if h == 0 or w == 0:
            return out
        t_sec = paired_t(self._entry, int(t))
        tiles = dict((int(a), b) for a, b in (self._entry.get("tiles") or ()))
        hits = tiles.get(int(m)) or ()
        if t_sec is None or not hits:
            return out                     # unpaired frame / no covering tile: honestly empty
        def read_tile(j, want):
            """The window of secondary tile ``j`` this output window needs, at the finest of ITS
            levels that window fits — so a coarse read stays coarse and a full-resolution one
            gets full resolution, without either being decided here."""
            budget = max(h, w)
            fy0, fy1, fx0, fx1 = want
            lv, sax = 0, self._s.level_axes(0)
            for cand_lv in range(max(1, int(getattr(self._s, "levels", 1)))):
                cand = self._s.level_axes(cand_lv)
                lv, sax = cand_lv, cand
                if max((fy1 - fy0) * cand.y, (fx1 - fx0) * cand.x) <= budget:
                    break
            ny, nx = max(1, int(sax.y)), max(1, int(sax.x))
            wy0, wy1 = int(np.floor(fy0 * ny)), int(np.ceil(fy1 * ny))
            wx0, wx1 = int(np.floor(fx0 * nx)), int(np.ceil(fx1 * nx))
            wy1, wx1 = min(ny, max(wy0 + 1, wy1)), min(nx, max(wx0 + 1, wx1))
            wy0, wx0 = max(0, min(wy0, wy1 - 1)), max(0, min(wx0, wx1 - 1))
            # THIS tile's own nearest plane, not the first hit's. The secondary's fields are
            # each focused at their own height — that is why a union Z grid over twelve WellA3
            # 640 fields is 297 planes and not 210 — so one plane index shared across every
            # covering tile is the wrong plane for all but one of them.
            z_j = (0 if z_um is None
                   else self._z_of(self._sec_md, self._sec_axes, int(j), z_um, dz=self._dz))
            plane = np.asarray(self._s.get_region(lv, int(j), int(t_sec), int(z_j),
                                                  min(k, int(self._sec_axes.c) - 1),
                                                  wy0, wy1, wx0, wx1))
            return plane, (wy0 / ny, wy1 / ny, wx0 / nx, wx1 / nx)

        # The window as a fraction of the primary's plane at THIS level — which is what makes a
        # tile read and a whole-plane read place the secondary identically.
        region = (y0 / max(1, lax.y), y1 / max(1, lax.y),
                  x0 / max(1, lax.x), x1 / max(1, lax.x))
        got = compose_secondary_plane(
            self._entry, (h, w), self._pri_md, self._pri_axes, int(m),
            self._sec_md, self._sec_axes, read_tile, region=region)
        return out if got is None else got


class FrameSubsetProvider(TileProvider):
    """A lazy **subset** view of another provider: ``m``/``t``/``z`` shrink to the picked
    indices and every read resolves through them into the base's own addressing.

    The three axes pick **independently**, so the view is their cross product: picking two
    positions, three timepoints and five z-planes yields 2·3 frames of 5 planes each. That
    is the honest generalisation of what the axes mean — a *frame* is the Frame domain's
    own address ``(m, t)`` and z lives inside one — and it is what turns a per-frame node
    loop into ``len(ms)·len(ts)`` units instead of M·T of them, each over a shorter volume.
    ``c`` and the spatial extent (and the pyramid) still pass through untouched, so a
    multi-channel node gets every channel of every picked plane; ``zs`` of ``None`` leaves
    z alone, which is the right default because a 3D node needs a volume.

    Picks are **sorted**, so t and z rise monotonically through the subset: a tracker (or
    any other temporal node) walks the chosen frames in acquisition order, and a 3D kernel
    walks the chosen planes bottom-to-top, just with the unpicked ones absent. Calibration
    passes through untouched — ``dt_s`` and ``z_step_um`` still describe the *source's*
    interval and spacing, exactly as :func:`nodegraph.metadata.frame_slice` reasons for
    ``zone.frame``. That is exact for a contiguous pick and a deliberate approximation for
    a strided one: a subset is a *sampling* of the acquisition, not a re-timed or
    re-spaced one.

    Its reason to exist is the GUI's **frame-selection troubleshooting scope**
    (:meth:`nodelab_v2.runner.EngineRunner.set_solo_frame`): the runner seeds the graph
    with a subset source and every node downstream — the eager per-unit computes included
    — then sees a short series without a single node knowing about it. That is why it
    lives here rather than in the node catalog: scoping a *run* is not an edit to the
    user's graph, and a node the user cannot see in their own graph would be a lie about
    what ran.

    The picks fold into :meth:`fingerprint`, hence into ``version``, hence into the source
    node's ``__seed_version__`` and so into every downstream ``recipe_hash`` (see
    :meth:`nodegraph.engine.Engine._entry`). Two consequences, both required: a scoped
    result can never be served for a different selection or for the un-scoped series, and
    coming back to a selection already computed is a plain memo hit.
    """

    #: fingerprint discriminator — the single-frame subclass keeps its own, so the two
    #: never alias in the memo and a one-frame scope keys exactly as it always has.
    _TAG = "frame-subset"

    def __init__(self, base: TileProvider, ms=(0,), ts=(0,), zs=None) -> None:
        self._base = base
        ax = base.axes
        self._ms = _picked(ms, ax.m)
        self._ts = _picked(ts, ax.t)
        # None (not an empty tuple) is "z untouched": the whole volume, no remap, and
        # nothing added to the fingerprint — so a scope that only picks frames keeps the
        # identity it had before z became pickable.
        self._zs = _picked(zs, ax.z) if zs else None
        self.axes = replace(ax, m=len(self._ms), t=len(self._ts),
                            z=ax.z if self._zs is None else len(self._zs))
        self.tile = base.tile
        self.levels = base.levels
        # A pure index remap computes NOTHING, so it is not a level of the lazy chain:
        # inheriting `depth` (rather than +1) keeps a consumer's tile fan-out heuristic
        # (`StreamProvider._fanout_ok`) reading the same as it would without the pick.
        self.depth = getattr(base, "depth", 0)
        self.cum_halo = getattr(base, "cum_halo", 0)

    @property
    def frames(self) -> tuple:
        """The picked ``(ms, ts)`` in the BASE provider's addressing (both sorted)."""
        return (self._ms, self._ts)

    @property
    def planes(self) -> Optional[tuple]:
        """The picked ``zs`` in the BASE provider's addressing, or ``None`` when the whole
        z range passes through."""
        return self._zs

    def level_axes(self, level: int) -> AxisSizes:
        lax = self._base.level_axes(level)
        return replace(lax, m=len(self._ms), t=len(self._ts),
                       z=lax.z if self._zs is None else len(self._zs))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        # m/t/z are CLAMPED into the subset rather than trusted: this view has only the
        # picked indices, a consumer's own clamping already sends an in-range one, and
        # resolving through the pick means a hand-built read cannot escape the scope.
        bm = self._ms[min(max(0, int(m)), len(self._ms) - 1)]
        bt = self._ts[min(max(0, int(t)), len(self._ts) - 1)]
        bz = z if self._zs is None else \
            self._zs[min(max(0, int(z)), len(self._zs) - 1)]
        return self._base.read_region(level, bm, bt, bz, c, y0, y1, x0, x1)

    def fingerprint(self) -> tuple:
        fp = (self._TAG, self._base.fingerprint(), self._ms, self._ts)
        return fp if self._zs is None else fp + (self._zs,)


class FrameSliceProvider(FrameSubsetProvider):
    """The one-frame case of :class:`FrameSubsetProvider`: ``m``/``t`` collapse to 1 and
    every read resolves to the pinned ``(m, t)``. ``zs`` picks planes within it, exactly
    as on the general view, and defaults to the whole volume.

    Kept as its own type — and its own fingerprint tag — because scoping to a single
    frame is the common troubleshooting move: it reads better at the call site, and a
    one-frame scope keeps keying the memo the way it did before subsets existed."""

    _TAG = "frame-slice"

    def __init__(self, base: TileProvider, m: int = 0, t: int = 0, zs=None) -> None:
        super().__init__(base, (m,), (t,), zs)

    @property
    def frame(self) -> tuple:
        """The pinned ``(m, t)`` in the BASE provider's addressing."""
        return (self._ms[0], self._ts[0])


class ArrayProvider(TileProvider):
    """A provider backed by an in-memory ``(M,T,Z,C,Y,X)`` array — how a *realizing*
    node (e.g. deconvolve) wraps its computed volume back into ``Dataset.image``.
    Single-level (no pyramid); numpy only."""

    def __init__(self, array: np.ndarray, *, tile: int = 512) -> None:
        a = np.asarray(array)
        if a.ndim != 6:
            raise ValueError(f"expected a 6-D (M,T,Z,C,Y,X) array, got {a.shape}")
        self._a = a
        m, t, z, c, y, x = a.shape
        self.axes = AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x)
        self.tile = tile
        self.levels = 1

    @property
    def nbytes(self) -> int:
        """The realized raster's in-memory bytes — what a memo eviction of the Dataset
        holding this provider actually frees (Memo GC sizing, :func:`~nodegraph.memo.payload_bytes`)."""
        return int(self._a.nbytes)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1) -> np.ndarray:
        if level != 0:
            raise ValueError("ArrayProvider has no pyramid (level 0 only)")
        return np.asarray(self._a[m, t, z, c, y0:y1, x0:x1])

    def fingerprint(self) -> tuple:
        """Content fingerprint — the realized array's bytes (this provider IS its
        content, so two ArrayProviders differ iff their arrays differ). **Memoized on
        the instance** (C1): the array is frozen by convention once memoized, and
        streaming chains re-read base fingerprints per construction — hash once."""
        fp = getattr(self, "_fp", None)
        if fp is not None:
            return fp
        import hashlib
        digest = hashlib.blake2b(np.ascontiguousarray(self._a).tobytes(),
                                 digest_size=16).hexdigest()
        # `tile` is part of the identity: the streaming TileCache addresses tiles by
        # (iy, ix) ON this provider's grid, so two same-content providers on different
        # grids must never alias (C1 review 2026-07-22).
        self._fp = ("array", self._a.shape, str(self._a.dtype), self.tile, digest)
        return self._fp


__all__ = ["TileProvider", "SyntheticProvider", "B2ndProvider", "ArrayProvider",
           "ChannelMergeProvider", "FrameSubsetProvider", "FrameSliceProvider",
           "subset_index"]
