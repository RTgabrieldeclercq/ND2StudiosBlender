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
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        """A single-z 2D window ``[y0:y1, x0:x1]`` at ``level`` for ``(b,m,t,z,c)``.

        **``b`` is keyword-only with a default (V3.01), deliberately.** The batch axis
        was added to a repo with 206 positional call sites of this contract; threading it
        positionally would have been 206 edits in which a single mis-ordered argument
        reads ``m`` as ``b`` and returns the wrong position's pixels — plausible data, no
        error. Keyword-only makes every existing call keep meaning exactly what it meant
        (member 0 of a one-member batch), and makes every batch-aware read say ``b=`` at
        the call site, where it can be read.
        """

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
                   y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        ax = self.level_axes(level)
        return self.read_region(level, m, t, z, c,
                                max(0, y0), min(y1, ax.y), max(0, x0), min(x1, ax.x), b=b)

    def get_tile(self, level: int, m: int, t: int, z: int, c: int,
                 iy: int, ix: int, *, b: int = 0) -> np.ndarray:
        """One block at grid position ``(iy, ix)`` — clipped at the plane edge."""
        ax = self.level_axes(level)
        y0, x0 = iy * self.tile, ix * self.tile
        if y0 >= ax.y or x0 >= ax.x or iy < 0 or ix < 0:
            raise IndexError(f"tile ({iy},{ix}) out of range for level {level} "
                             f"{(ax.y, ax.x)} tile={self.tile}")
        return self.read_region(level, m, t, z, c, y0, min(y0 + self.tile, ax.y),
                                x0, min(x0 + self.tile, ax.x), b=b)

    def get_subvolume(self, level: int, m: int, t: int, c: int,
                      z0: int, z1: int, iy: int, ix: int, *, b: int = 0) -> np.ndarray:
        """A ``(z1-z0, block_y, block_x)`` brick — planar blocks gathered across the
        z-range (the benchmark-preferred path; not a MIP). An empty range returns a
        ``(0, block_y, block_x)`` array (review #12: ``np.stack([])`` would crash)."""
        if z1 <= z0:
            probe = self.get_tile(level, m, t, 0, c, iy, ix, b=b)   # z=0 always in range
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_tile(level, m, t, z, c, iy, ix, b=b)
                         for z in range(z0, z1)], axis=0)

    def get_region_volume(self, level: int, m: int, t: int, c: int,
                          z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
                          *, b: int = 0) -> np.ndarray:
        """An arbitrary ``(Z,Y,X)`` ROI (composes single-z region reads); an empty
        range returns a ``(0, ...)`` array (review #12)."""
        if z1 <= z0:
            probe = self.get_region(level, m, t, 0, c, y0, y1, x0, x1, b=b)
            return np.empty((0,) + probe.shape, dtype=probe.dtype)
        return np.stack([self.get_region(level, m, t, z, c, y0, y1, x0, x1, b=b)
                         for z in range(z0, z1)], axis=0)


# ── constant provider (a blank backdrop: no pixels held) ──────────────────────

class ConstantProvider(TileProvider):
    """Every voxel is ``value`` — a backdrop that costs nothing however large it is.

    ``view.canvas``'s image: an experiment canvas spanning several files' stage footprint
    can be tens of thousands of pixels on a side (three adjacent 1760 µm grids at
    1.718 µm/px are ~9000 x 13500), and nothing in it is data — the files are drawn onto it
    by the Overlays downstream. Allocating it would be gigabytes of zeros; generating a
    window on demand is O(window), at any pyramid level, exactly like
    :class:`SyntheticProvider`. Structural fingerprint (type + geometry + value): two
    canvases of one geometry ARE the same pixels."""

    def __init__(self, axes: AxisSizes, *, value: float = 0.0, levels: int = 1,
                 tile: int = 512, dtype: Any = np.uint16) -> None:
        self.axes = axes
        self.value = value
        self.levels = max(1, int(levels))
        self.tile = tile
        self.dtype = np.dtype(dtype)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        return np.full((max(0, y1 - y0), max(0, x1 - x0)), self.value, dtype=self.dtype)

    def fingerprint(self) -> tuple:
        return super().fingerprint() + (float(self.value), self.dtype.str)


# ── synthetic provider (numpy only; deterministic — for tests) ────────────────

class SyntheticProvider(TileProvider):
    """A provider whose voxels are a cheap deterministic function of their address,
    so a window read is exact and O(window) — ideal for testing the read contract.
    Levels are **stride-decimated** (level ``l`` samples every ``2**l``)."""

    def __init__(self, axes: AxisSizes, *, tile: int = 512, levels: int = 1) -> None:
        self.axes = axes
        self.tile = tile
        self.levels = levels

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
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

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
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
                               cparams=cparams, urlpath=urlpath, level=0,
                               batch_thin_z=True)
        m, t, z, c = vol6d.shape[:4]
        ct, cz = int(kw["chunks"][1]), int(kw["chunks"][2])
        # A determinate bar needs sub-level granularity, and a chunk spans `ct` timepoints x
        # `cz` planes and never crosses (M,C) — so filling ONE such block at a time is exactly
        # chunk-aligned (the same on-disk bytes as `blosc2.asarray`) and reports honestly. It
        # also holds one block contiguous at a time instead of a whole second copy of the
        # volume, which is what made a >50 GB ingest thrash.
        #
        # `batch_thin_z=True` (V2.26) is what lets `ct` exceed 1, and it was measured before
        # being turned on here rather than after. `cz` is capped by Z, so a 2D or thin-Z
        # acquisition used to pin one 8 MiB chunk per plane and write at 50 MB/s — against
        # 182 MB/s for the same data through the Dock's writer, which got the fix first. What
        # held it back was that this store's READ profile serves every live chain and had
        # never been measured under T-batching. It has now, on a 2048² thin-Z store: full-plane
        # scrub 7.7 -> 7.9 ms and a 1024² window 4.75 -> 4.19 ms going ct=1 -> ct=4, at
        # byte-identical size. Flat, because `blocks` — the decompression granularity a reader
        # actually pays for — is untouched; the framing is a write-side concern only.
        arr0 = blosc2.empty(vol6d.shape, dtype=vol6d.dtype, **kw)
        cls._mark(arr0, 0, False)                    # declared, not yet written
        for im, ic in np.ndindex(m, c):
            for t0 in range(0, t, ct):
                t1 = min(t0 + ct, t)
                for z0 in range(0, z, cz):
                    z1 = min(z0 + cz, z)
                    arr0[im:im + 1, t0:t1, z0:z1, ic:ic + 1, :, :] = \
                        np.ascontiguousarray(
                            vol6d[im:im + 1, t0:t1, z0:z1, ic:ic + 1])
                    bump((t1 - t0) * (z1 - z0))
        cls._mark(arr0, 0, True)

        arrays = [arr0]
        for lvl in range(1, len(shapes)):
            nxt = cls._append_level(arrays[-1], lvl, tile=tile, cparams=cparams,
                                    urlpath=urlpath, bump=bump,
                                    batch_thin_z=True)
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
    ``channel.merge``'s image, and the only provider that reads from two sources and
    RE-ADDRESSES them (:class:`MultiSourceProvider` also spans several files, but by pure
    index arithmetic on ``m``: nothing there is resampled, paired in time, or placed).

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
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        from nodegraph.placement import compose_secondary_plane, paired_t
        lax = self.level_axes(level)
        z_um = self._grid.plane_um(m, z)
        if int(c) < self._pri_c:
            # The primary's own plane nearest this focus. With no focus log (`z_um is None`)
            # `secondary_z_index` answers 0 for a single plane and the grid is the primary's own
            # index space anyway, so pass the index straight through.
            pz = int(z) if z_um is None else self._z_of(self._pri_md, self._pri_axes, m, z_um)
            pz = min(max(0, pz), int(self._pri_axes.z) - 1)
            return np.asarray(
                self._p.get_region(level, m, t, pz, int(c), y0, y1, x0, x1, b=b))

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
        # A pure index remap adds no compute, so it inherits its base's COST model whole —
        # the same rule (and the same words) as :class:`~nodegraph.streaming.WindowView`. A
        # window of a plane-unit provider still costs a whole plane underneath, and dropping
        # the flag here would put back the cliff it exists to fence: ``util.stitch``'s
        # windowed path re-reads every overlapping tile uncached, so a consumer walking a
        # 26x26 grid would pay for the whole mosaic 676 times. It went unnoticed while this
        # view was only ever wrapped around a run-scope SOURCE (a file provider, which is
        # neither plane- nor volume-unit); ``util.crop``'s frames mode puts it downstream of
        # arbitrary nodes, where the flags are real.
        self.plane_unit = bool(getattr(base, "plane_unit", False))
        self.volume_unit = bool(getattr(base, "volume_unit", False))

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

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        # m/t/z are CLAMPED into the subset rather than trusted: this view has only the
        # picked indices, a consumer's own clamping already sends an in-range one, and
        # resolving through the pick means a hand-built read cannot escape the scope.
        bm = self._ms[min(max(0, int(m)), len(self._ms) - 1)]
        bt = self._ts[min(max(0, int(t)), len(self._ts) - 1)]
        bz = z if self._zs is None else \
            self._zs[min(max(0, int(z)), len(self._zs) - 1)]
        return self._base.read_region(level, bm, bt, bz, c, y0, y1, x0, x1, b=b)

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


class FrameRemapProvider(TileProvider):
    """A lazy RE-INDEX of another provider's T axis (2026-10-07): output frame ``k`` reads
    the base's frame ``sources[k]``, which may REPEAT (a held edge frame) or be ``None`` (a
    blank frame, read as zeros of the base's dtype). The output has ``len(sources)`` frames;
    ``m``, ``z``, ``c``, the spatial extent and the pyramid pass through untouched, and the
    cost flags are inherited exactly as :class:`FrameSubsetProvider` inherits them.

    Written for ``util.time_shift``. :class:`FrameSubsetProvider` cannot express it on
    purpose — its picks are sorted and de-duplicated because a subset is a *sampling* of the
    acquisition; this is a *re-timing*, where frame 0 may legitimately appear three times.
    The map folds into :meth:`fingerprint`, so two different shifts of one series never alias
    in the memo, and the same shift is a plain memo hit."""

    _TAG = "frame-remap"

    def __init__(self, base: TileProvider, sources) -> None:
        self._base = base
        ax = base.axes
        hi = max(0, int(ax.t) - 1)
        self._ts = tuple(None if s is None else min(max(0, int(s)), hi) for s in sources)
        if not self._ts:
            raise ValueError("FrameRemapProvider needs at least one output frame")
        self.axes = replace(ax, t=len(self._ts))
        self.tile = base.tile
        self.levels = base.levels
        self.depth = getattr(base, "depth", 0)
        self.cum_halo = getattr(base, "cum_halo", 0)
        self.plane_unit = bool(getattr(base, "plane_unit", False))
        self.volume_unit = bool(getattr(base, "volume_unit", False))

    @property
    def sources(self) -> tuple:
        """Per output frame, the BASE frame it reads — ``None`` for a blank frame."""
        return self._ts

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), t=len(self._ts))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        bt = self._ts[min(max(0, int(t)), len(self._ts) - 1)]
        if bt is None:
            # a blank frame: zeros in the base's own dtype and shape — learned from a real
            # frame's read, because a provider declares no dtype of its own
            real = next((s for s in self._ts if s is not None), 0)
            return np.zeros_like(self._base.read_region(level, m, real, z, c, y0, y1, x0, x1, b=b))
        return self._base.read_region(level, m, bt, z, c, y0, y1, x0, x1, b=b)

    def fingerprint(self) -> tuple:
        return (self._TAG, self._base.fingerprint(), self._ts)


class ArrayProvider(TileProvider):
    """A provider backed by an in-memory ``(B,M,T,Z,C,Y,X)`` array — how a *realizing*
    node (e.g. deconvolve) wraps its computed volume back into ``Dataset.image``.
    Single-level (no pyramid); numpy only.

    **6-D is one batch member (V3.01).** A realizing compute runs per member (a batch is
    a per-member unroll), so the array it hands back is ``(M,T,Z,C,Y,X)`` and ``b`` is
    elided exactly as :meth:`nodegraph.dataset.AxisSizes.axis_list` describes. A 7-D array
    is accepted too, for the ``b > 1`` case at the batch boundary.
    """

    def __init__(self, array: np.ndarray, *, tile: int = 512) -> None:
        a = np.asarray(array)
        if a.ndim == 6:
            b, (m, t, z, c, y, x) = 1, a.shape
        elif a.ndim == 7:
            b, m, t, z, c, y, x = a.shape
            a = a if b > 1 else a[0]       # keep the stored form canonical: b elided at 1
        else:
            raise ValueError(
                f"expected a 6-D (M,T,Z,C,Y,X) array for one batch member, or a 7-D "
                f"(B,M,T,Z,C,Y,X) one for a batch, got {a.shape}")
        self._a = a
        self._batched = b > 1
        self.axes = AxisSizes(b=b, m=m, t=t, z=z, c=c, y=y, x=x)
        self.tile = tile
        self.levels = 1

    @property
    def nbytes(self) -> int:
        """The realized raster's in-memory bytes — what a memo eviction of the Dataset
        holding this provider actually frees (Memo GC sizing, :func:`~nodegraph.memo.payload_bytes`)."""
        return int(self._a.nbytes)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        if level != 0:
            raise ValueError("ArrayProvider has no pyramid (level 0 only)")
        a = self._a[b] if self._batched else self._a
        return np.asarray(a[m, t, z, c, y0:y1, x0:x1])

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


class MultiSourceProvider(TileProvider):
    """K files laid end to end on the multipoint axis — a **file bundle**'s image.

    Position ``m`` of the bundle is position ``m - offset[i]`` of source ``i``. Nothing is
    resampled, blended or copied: this is a pure index remap, so a bundled pull reads
    exactly the bytes an un-bundled one would, from each file's own store.

    **Why ``m`` and not a new axis.** ``m`` already means "one acquisition site" — a well,
    a tile, a stage point — and every domain, every table row (``COORD_COLUMNS`` carries
    ``m``) and the viewer's M strip already address it. Only three nodes in the catalog
    read across ``m`` at all (the ``MULTI_VIEW`` three), so for everything else a position
    is an independent unit of work and a bundle runs one pipeline over N files for free.
    That independence is a *measured* property, not an assumption:
    ``scripts/_probe_m_batch_independence.py`` pins it, and names the one documented
    exception — a histogram ``scope="dataset"`` pools its level over the whole (m,t,z,c)
    population, so under a bundle it pools **across files**. That is the same widening the
    2026-07-30 scope fix made opt-in for positions; a bundle does not change its meaning,
    it enlarges what it reaches.

    **The grid must agree.** Sources are required to match on ``t``/``z``/``c``/``y``/``x``.
    A bundle is a claim that one pipeline is meaningful over all of it, and a chain that
    silently re-addressed a different pixel size or channel count per position would break
    that claim somewhere far downstream — where it reads as a bad result rather than a bad
    grouping. Refusing here is the cheap end of that trade.

    **Identity.** :meth:`version` folds each source's own ``version`` (not its
    ``fingerprint``): ``B2ndProvider`` overrides ``version`` to carry the store's mtime, so
    editing one file of a bundle in place invalidates the bundle — folding fingerprints
    would have silently served the stale pixels.
    """

    def __init__(self, sources: Sequence[TileProvider],
                 labels: Optional[Sequence[str]] = None) -> None:
        srcs = list(sources)
        if not srcs:
            raise ValueError("a file bundle needs at least one source")
        first = srcs[0].axes
        for i, s in enumerate(srcs[1:], start=1):
            a = s.axes
            bad = [n for n in ("t", "z", "c", "y", "x")
                   if int(getattr(a, n)) != int(getattr(first, n))]
            if bad:
                want = tuple(int(getattr(first, n)) for n in ("t", "z", "c", "y", "x"))
                got = tuple(int(getattr(a, n)) for n in ("t", "z", "c", "y", "x"))
                raise ValueError(
                    f"file {i} does not share the bundle's grid on {', '.join(bad)}: "
                    f"(t,z,c,y,x) is {got}, the bundle's is {want}. Group only files with "
                    f"the same channels and frame geometry, or load them separately.")
        self._srcs = tuple(srcs)
        counts = [max(0, int(s.axes.m)) for s in srcs]
        # exclusive prefix sums: source i owns bundle positions [_off[i], _off[i+1])
        off, run = [0], 0
        for n in counts:
            run += n
            off.append(run)
        self._off = tuple(off)
        self.axes = replace(first, m=run)
        self.labels = tuple(labels) if labels is not None else \
            tuple(f"file{i}" for i in range(len(srcs)))
        # The shared pyramid is only as deep as the SHALLOWEST member: a level the bundle
        # advertises must exist in every source, or a read into the short one would fall
        # off its own pyramid at display time rather than at construction.
        self.levels = max(1, min(int(getattr(s, "levels", 1)) for s in srcs))
        self.tile = min(int(getattr(s, "tile", 512)) for s in srcs)
        # A bundle costs what its most expensive member costs: these flags fence real
        # read-amplification cliffs (see FrameSubsetProvider), and taking `any`/`max`
        # keeps the fence up for every position, not just the ones in a cheap file.
        self.depth = max((int(getattr(s, "depth", 0)) for s in srcs), default=0)
        self.cum_halo = max((int(getattr(s, "cum_halo", 0)) for s in srcs), default=0)
        self.plane_unit = any(bool(getattr(s, "plane_unit", False)) for s in srcs)
        self.volume_unit = any(bool(getattr(s, "volume_unit", False)) for s in srcs)

    @property
    def sources(self) -> tuple:
        return self._srcs

    @property
    def spans(self) -> tuple:
        """``(label, m_start, m_count)`` per source — the file identity of every position."""
        return tuple((self.labels[i], self._off[i], self._off[i + 1] - self._off[i])
                     for i in range(len(self._srcs)))

    def locate(self, m: int) -> Tuple[int, int]:
        """Bundle position ``m`` → ``(source index, that source's own m)``."""
        mm = min(max(0, int(m)), max(0, int(self.axes.m) - 1))
        i = bisect_left(self._off, mm + 1) - 1
        i = min(max(0, i), len(self._srcs) - 1)
        return i, mm - self._off[i]

    def level_axes(self, level: int) -> AxisSizes:
        # Delegate the spatial halving to a real member so the bundle's pyramid geometry
        # is the members' own, then restore the concatenated m.
        return replace(self._srcs[0].level_axes(level), m=int(self.axes.m))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        i, local_m = self.locate(m)
        return np.asarray(self._srcs[i].read_region(
            level, local_m, t, z, c, y0, y1, x0, x1, b=b))

    def fingerprint(self) -> tuple:
        return ("bundle", tuple(s.fingerprint() for s in self._srcs), self._off)

    @property
    def version(self) -> Any:
        # `version`, not `fingerprint`: a disk-backed member folds its mtime into version
        # only, so this is what makes an in-place edit of one file invalidate the bundle.
        return ("bundle", tuple(s.version for s in self._srcs), self._off)


class BatchProvider(TileProvider):
    """K acquisitions stacked on the **batch** axis — one file per ``b`` (V3.01).

    Member ``b`` of the batch is source ``b``, read at its own ``(m,t,z,c)`` unchanged.
    Like :class:`MultiSourceProvider` this is a pure re-addressing — nothing is resampled,
    blended or copied — but it grows a *new* axis rather than lengthening ``m``, and that
    difference is the entire point of the class.

    **Why not just use** :class:`MultiSourceProvider`. Laying files end to end on ``m``
    makes them positions of one acquisition, and the engine then cannot tell "position 3 of
    file 1" from "position 3 of the run". That is not a naming quibble: it is exactly why a
    ``scope="dataset"`` statistic pools its level **across files** under a file bundle — the
    caveat :class:`MultiSourceProvider`'s own docstring names, pinned by
    ``scripts/_probe_m_batch_independence.py``. On ``b`` the population a scope pools over
    stops at the file boundary, so a per-file threshold is a per-file threshold.

    **The grid must agree on every other axis.** Sources must match on ``m``/``t``/``z``/
    ``c``/``y``/``x``, and the refusal names the axis. An array axis is rectangular, so a
    ragged batch is not representable at all — files with different frame counts have to be
    run as separate graphs, not silently padded or truncated to the shortest.

    **Identity** folds each member's ``version``, not its ``fingerprint``, for the reason
    :class:`MultiSourceProvider` gives: a disk-backed member carries its store's mtime in
    ``version`` alone, so folding fingerprints would serve a batch whose file changed
    underneath it.
    """

    #: Axes every member must already agree on — everything except ``b`` itself.
    MUST_MATCH: Tuple[str, ...] = ("m", "t", "z", "c", "y", "x")

    def __init__(self, sources: Sequence[TileProvider],
                 labels: Optional[Sequence[str]] = None) -> None:
        srcs = list(sources)
        if not srcs:
            raise ValueError("a batch needs at least one source")
        for i, s in enumerate(srcs):
            if int(getattr(s.axes, "b", 1)) != 1:
                raise ValueError(
                    f"batch member {i} is itself a {s.axes.b}-member batch; nested "
                    f"batches are not representable on one b axis. Unbatch it first.")
        first = srcs[0].axes
        for i, s in enumerate(srcs[1:], start=1):
            a = s.axes
            bad = [n for n in self.MUST_MATCH
                   if int(getattr(a, n)) != int(getattr(first, n))]
            if bad:
                want = tuple(int(getattr(first, n)) for n in self.MUST_MATCH)
                got = tuple(int(getattr(a, n)) for n in self.MUST_MATCH)
                raise ValueError(
                    f"batch member {i} does not share the batch's grid on "
                    f"{', '.join(bad)}: (m,t,z,c,y,x) is {got}, the batch's is {want}. "
                    f"A batch runs ONE pipeline over every member, so the members have to "
                    f"share a grid; run the odd one out as its own graph.")
        self._srcs = tuple(srcs)
        self.axes = replace(first, b=len(srcs))
        self.labels = tuple(labels) if labels is not None else \
            tuple(f"file{i}" for i in range(len(srcs)))
        # Same reasoning as MultiSourceProvider for every one of these: a batch advertises
        # only the shallowest member's pyramid, costs what its most expensive member costs,
        # and keeps a read-amplification fence up for every member rather than the cheap ones.
        self.levels = max(1, min(int(getattr(s, "levels", 1)) for s in srcs))
        self.tile = min(int(getattr(s, "tile", 512)) for s in srcs)
        self.depth = max((int(getattr(s, "depth", 0)) for s in srcs), default=0)
        self.cum_halo = max((int(getattr(s, "cum_halo", 0)) for s in srcs), default=0)
        self.plane_unit = any(bool(getattr(s, "plane_unit", False)) for s in srcs)
        self.volume_unit = any(bool(getattr(s, "volume_unit", False)) for s in srcs)

    @property
    def sources(self) -> tuple:
        return self._srcs

    @property
    def spans(self) -> tuple:
        """``(label, b_index)`` per member — the file identity of every batch index."""
        return tuple((self.labels[i], i) for i in range(len(self._srcs)))

    def member(self, b: int) -> TileProvider:
        """The source backing batch index ``b``, clamped into range."""
        return self._srcs[min(max(0, int(b)), len(self._srcs) - 1)]

    def level_axes(self, level: int) -> AxisSizes:
        # Delegate spatial halving to a real member so the pyramid geometry is the
        # members' own, then restore the batch extent.
        return replace(self._srcs[0].level_axes(level), b=int(self.axes.b))

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        # b addresses the MEMBER here and is consumed, not forwarded: the member is an
        # ordinary single-acquisition provider and reads at its own b=0.
        return np.asarray(self.member(b).read_region(
            level, m, t, z, c, y0, y1, x0, x1))

    def fingerprint(self) -> tuple:
        return ("batch", tuple(s.fingerprint() for s in self._srcs))

    @property
    def version(self) -> Any:
        return ("batch", tuple(s.version for s in self._srcs))


class BatchSliceProvider(TileProvider):
    """One member of a batch, as a ``b == 1`` provider — what ``util.select_batch`` puts on
    the wire, and the exact inverse of :class:`BatchProvider` (V3.01).

    A pure re-address, like :class:`FrameSliceProvider` one axis out: every read is
    forwarded to the base at the pinned member, so member ``k``'s pixels are byte-for-byte
    what an unbatched graph would have read. It wraps ANY provider rather than only a
    :class:`BatchProvider`, which is the point — by the time an unbatch runs, the batch has
    a whole pipeline stacked on it and the thing on the wire is a compute provider, not the
    original stack.

    The unbatch is therefore free: splitting K results apart costs no compute and no copy,
    because each member's chain was always addressed by ``b`` and this only fixes it.
    """

    def __init__(self, base: TileProvider, b: int) -> None:
        nb = int(getattr(base.axes, "b", 1))
        k = int(b)
        if not (0 <= k < nb):
            raise ValueError(
                f"batch member {k} is out of range for a {nb}-member batch [0, {nb}).")
        self._base = base
        self._b = k
        self.axes = replace(base.axes, b=1)
        self.levels = int(getattr(base, "levels", 1))
        self.tile = int(getattr(base, "tile", 512))
        # the member costs exactly what the batch charged for it — inherit every fence
        self.depth = int(getattr(base, "depth", 0))
        self.cum_halo = int(getattr(base, "cum_halo", 0))
        self.plane_unit = bool(getattr(base, "plane_unit", False))
        self.volume_unit = bool(getattr(base, "volume_unit", False))

    @property
    def member_index(self) -> int:
        return self._b

    def level_axes(self, level: int) -> AxisSizes:
        return replace(self._base.level_axes(level), b=1)

    def read_region(self, level, m, t, z, c, y0, y1, x0, x1, *, b: int = 0) -> np.ndarray:
        # The caller's `b` addresses THIS view, which has one member, so it is discarded
        # and the pinned index used instead. Forwarding the caller's would re-index into
        # the base and hand back a different file.
        return np.asarray(self._base.read_region(
            level, m, t, z, c, y0, y1, x0, x1, b=self._b))

    def fingerprint(self) -> tuple:
        return ("batchslice", self._base.fingerprint(), self._b)

    @property
    def version(self) -> Any:
        return ("batchslice", self._base.version, self._b)


#: The axes :class:`AxisConcatProvider` can grow — never ``c``: a channel merge needs
#: :class:`ChannelMergeProvider`'s placement/resampling, not a pure index remap (two files
#: at different pixel sizes cannot be laid end to end on Y/X the way two timepoints can be
#: laid end to end on T).
CONCAT_AXES: Tuple[str, ...] = ("t", "m", "z")


class AxisConcatProvider(TileProvider):
    """N sources laid end to end on ONE axis — ``t``, ``m`` or ``z`` — ``util.merge``'s
    literal-concatenation branch (:data:`CONCAT_AXES`; ``c`` goes through
    :class:`ChannelMergeProvider` instead, see above).

    This is :class:`MultiSourceProvider` generalised from "always ``m``" to "whichever axis
    the node picked" — the exclusive-prefix-sum index remap is identical, just applied to a
    different axis of ``read_region``'s signature. **Nothing is resampled, blended or
    interpolated**: source ``i`` owns concat-axis positions ``[_off[i], _off[i+1])``, and a
    read at index ``k`` of the merged axis is index ``k - _off[i]`` of source ``i``, on
    whichever of its own ``(m,t,z)`` the other two indices already name. A read still costs
    exactly what it would cost un-merged.

    **The grid must agree on every OTHER axis** (mirrors :class:`MultiSourceProvider`'s own
    contract exactly, generalised the same way): growing ``t`` requires every input to
    already match on ``m,z,c,y,x``; growing ``m`` requires ``t,z,c,y,x``; growing ``z``
    requires ``m,t,c,y,x``. A caller with mismatched inputs resamples/crops upstream
    (``util.resample``, ``util.crop``) rather than this provider silently reconciling a
    difference it cannot see the physical meaning of — the same trade
    :class:`MultiSourceProvider`'s docstring argues for M.
    """

    def __init__(self, sources: Sequence[TileProvider], axis: str,
                 labels: Optional[Sequence[str]] = None) -> None:
        axis = str(axis).lower()
        if axis not in CONCAT_AXES:
            raise ValueError(f"AxisConcatProvider: axis must be one of {CONCAT_AXES}, "
                             f"got {axis!r}")
        srcs = list(sources)
        if len(srcs) < 2:
            raise ValueError("a merge needs at least two sources")
        other = [n for n in ("m", "t", "z", "c", "y", "x") if n != axis]
        first = srcs[0].axes
        for i, s in enumerate(srcs[1:], start=1):
            a = s.axes
            bad = [n for n in other if int(getattr(a, n)) != int(getattr(first, n))]
            if bad:
                want = tuple(int(getattr(first, n)) for n in other)
                got = tuple(int(getattr(a, n)) for n in other)
                raise ValueError(
                    f"input {i} does not match input 0 on {', '.join(bad)}: "
                    f"({', '.join(other)}) is {got}, input 0's is {want}. Growing "
                    f"{axis!r} requires every other axis to already agree — resample or "
                    f"crop upstream so they match, or merge onto a different axis.")
        self._srcs = tuple(srcs)
        self._axis = axis
        counts = [max(0, int(getattr(s.axes, axis))) for s in srcs]
        # exclusive prefix sums: source i owns merged-axis positions [_off[i], _off[i+1])
        off, run = [0], 0
        for n in counts:
            run += n
            off.append(run)
        self._off = tuple(off)
        self.axes = replace(first, **{axis: run})
        self.labels = tuple(labels) if labels is not None else \
            tuple(f"input{i}" for i in range(len(srcs)))
        # Same reasoning as MultiSourceProvider: the shared pyramid/tile/cost model is the
        # worst (shallowest/most expensive) of the members, so no read past what a bundle
        # advertises falls off a shorter member's own pyramid.
        self.levels = max(1, min(int(getattr(s, "levels", 1)) for s in srcs))
        self.tile = min(int(getattr(s, "tile", 512)) for s in srcs)
        self.depth = max((int(getattr(s, "depth", 0)) for s in srcs), default=0)
        self.cum_halo = max((int(getattr(s, "cum_halo", 0)) for s in srcs), default=0)
        self.plane_unit = any(bool(getattr(s, "plane_unit", False)) for s in srcs)
        self.volume_unit = any(bool(getattr(s, "volume_unit", False)) for s in srcs)

    @property
    def sources(self) -> tuple:
        return self._srcs

    @property
    def spans(self) -> tuple:
        """``(label, start, count)`` per source along the merged axis — the input
        identity of every index, the same shape as :attr:`MultiSourceProvider.spans`."""
        return tuple((self.labels[i], self._off[i], self._off[i + 1] - self._off[i])
                     for i in range(len(self._srcs)))

    def locate(self, i: int) -> Tuple[int, int]:
        """Merged-axis index ``i`` → ``(source index, that source's own index)``."""
        ii = min(max(0, int(i)), max(0, int(getattr(self.axes, self._axis)) - 1))
        j = bisect_left(self._off, ii + 1) - 1
        j = min(max(0, j), len(self._srcs) - 1)
        return j, ii - self._off[j]

    def level_axes(self, level: int) -> AxisSizes:
        # Delegate the spatial halving to a real member (its pyramid geometry), then
        # restore the concatenated axis to its own full extent.
        base = self._srcs[0].level_axes(level)
        return replace(base, **{self._axis: int(getattr(self.axes, self._axis))})

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        idx = {"m": m, "t": t, "z": z}
        src_i, local = self.locate(idx[self._axis])
        idx[self._axis] = local
        return np.asarray(self._srcs[src_i].read_region(
            level, idx["m"], idx["t"], idx["z"], c, y0, y1, x0, x1, b=b))

    def fingerprint(self) -> tuple:
        return ("axis_concat", self._axis, tuple(s.fingerprint() for s in self._srcs),
                self._off)

    @property
    def version(self) -> Any:
        return ("axis_concat", self._axis, tuple(s.version for s in self._srcs), self._off)

#: Axes a chain's file members can be laid onto (:class:`AxisRespreadProvider`). ``y``/``x``
#: are absent because a file boundary is not a spatial one: laying two files side by side in
#: the field of view is a stitch (``util.stitch``), which needs overlap and blending rather
#: than an index remap. ``m`` IS here — laying members onto positions is an ordinary chain,
#: and for one already-ordered source it is the identity.
RESPREAD_AXES: Tuple[str, ...] = ("t", "z", "c", "m")


class AxisRespreadProvider(TileProvider):
    """K file **members** laid end to end on one axis — ``util.chain``.

    A member is one source file: a provider, the ``m`` offset at which that file's positions
    begin inside it, and how many positions it holds. Two shapes of input collapse to the
    same thing here, which is the point of taking triples rather than either one alone:

    * a **file bundle** (:class:`MultiSourceProvider`) is ONE provider whose ``m`` already
      holds K files end to end, so its members share a provider and differ in ``start``;
    * **separately loaded files** are K providers each holding one file, so its members
      differ in provider and all start at 0.

    Either way member ``i`` owns ``m`` in ``[start_i, start_i + count_i)``, and::

        in :  m = start_i + j        axis = a
        out:  m = j                  axis = off_i + a          (axis != "m")
        out:  m = off_i + j                                    (axis == "m")

    where ``off`` is the exclusive prefix sum of the members' own extents along the chained
    axis. So 120 single-position files each holding one frame become ``t = 0..119``; three
    files holding 50, 30 and 50 frames become ``t = 0..129`` with each file's own frames
    contiguous; and onto ``m`` the members simply line up as positions.

    **Members may differ on the axis being chained — that is the whole point.** A series
    exported in unequal chunks (5 frames, then 3) is still one series, and requiring the
    chunks to match would refuse the ordinary case. Every OTHER axis must agree, because the
    result is rectangular in those; ``m`` is exempt from that check too, since each member is
    read through its own ``[start, start+count)`` window rather than whole.

    **Nothing is resampled, blended or copied** — one output voxel is one input voxel, so a
    read costs exactly what the un-chained read costs. That is why ``util.chain`` declares
    ``TILEABLE`` despite changing two axes.

    Members are in OUTPUT order, so the ordering decision — by the counting field in the
    filenames (:mod:`nodegraph.file_sequence`) rather than by wiring order — is expressed
    here and nowhere else, and folds into :meth:`version`: the same files chained the other
    way round can never be served from each other's memo entry.
    """

    def __init__(self, members: Sequence[Tuple[TileProvider, int, int]], axis: str,
                 labels: Optional[Sequence[str]] = None) -> None:
        axis = str(axis).lower()
        if axis not in RESPREAD_AXES:
            raise ValueError(f"AxisRespreadProvider: axis must be one of {RESPREAD_AXES}, "
                             f"got {axis!r}")
        mem = [(p, int(s), int(n)) for p, s, n in members]
        if not mem:
            raise ValueError("a chain needs at least one file member")
        first = mem[0][0].axes
        # Every axis but `m` and the chained one. `m` is exempt because each member is read
        # through its own window; the chained axis is exempt because laying files end to end
        # along it is exactly what this provider does.
        others = [a for a in ("b", "t", "z", "c", "y", "x") if a != axis]
        for i, (prov, start, count) in enumerate(mem):
            a = prov.axes
            bad = [nm for nm in others
                   if int(getattr(a, nm)) != int(getattr(first, nm))]
            if bad:
                want = tuple(int(getattr(first, nm)) for nm in others)
                got = tuple(int(getattr(a, nm)) for nm in others)
                raise ValueError(
                    f"member {i} does not match member 0 on {', '.join(bad)}: "
                    f"({', '.join(others)}) is {got}, member 0's is {want}. Chaining lays "
                    f"files end to end on {axis.upper()} alone, so every OTHER axis has to "
                    f"already agree — resample or crop upstream so they match, or chain "
                    f"onto a different axis.")
            if count < 1:
                raise ValueError(
                    f"member {i} holds {count} positions; every file must hold at least one.")
            if start < 0 or start + count > int(a.m):
                raise ValueError(
                    f"member {i} occupies positions [{start}, {start + count}) but its "
                    f"source has only {int(a.m)} — the source_file labels disagree with "
                    f"the image.")
        if axis != "m":
            counts = {n for _, _, n in mem}
            if len(counts) != 1:
                raise ValueError(
                    f"chaining onto {axis.upper()} needs every file to hold the same number "
                    f"of positions (got {sorted(counts)}) — the result has ONE position "
                    f"axis, so a file with more has nowhere to put them. Chain onto M "
                    f"instead to keep them as separate positions.")
        self._mem = tuple(mem)
        self._axis = axis
        self._n = mem[0][2]
        # Each member's own extent along the chained axis — its position count onto `m`,
        # otherwise its provider's own size on that axis. These are what differ.
        self._ext = tuple(n if axis == "m" else max(1, int(getattr(p.axes, axis)))
                          for p, _s, n in mem)
        off, run = [0], 0
        for e in self._ext:
            run += e
            off.append(run)
        self._off = tuple(off)                      # exclusive prefix sums
        self.axes = replace(first, m=run) if axis == "m" else \
            replace(first, m=self._n, **{axis: run})
        self.labels = tuple(labels) if labels is not None else \
            tuple(f"file{i}" for i in range(len(mem)))
        # Geometry and cost model are the members': this is a pure re-address, so it
        # neither deepens the pyramid nor changes what a tile costs. The SHALLOWEST member
        # bounds the shared pyramid (the `MultiSourceProvider` rule) so no read into a
        # level this provider advertises falls off a shorter member's own.
        self.levels = max(1, min(int(getattr(p, "levels", 1)) for p, _s, _n in mem))
        self.tile = min(int(getattr(p, "tile", 512)) for p, _s, _n in mem)
        self.depth = max((int(getattr(p, "depth", 0)) for p, _s, _n in mem), default=0)
        self.cum_halo = max((int(getattr(p, "cum_halo", 0)) for p, _s, _n in mem), default=0)
        self.plane_unit = any(bool(getattr(p, "plane_unit", False)) for p, _s, _n in mem)
        self.volume_unit = any(bool(getattr(p, "volume_unit", False)) for p, _s, _n in mem)

    @property
    def members(self) -> tuple:
        """``(provider, m start, position count)`` per file member, in output order."""
        return self._mem

    @property
    def spans(self) -> tuple:
        """``(label, start, count)`` per file along the chained axis — the same shape as
        :attr:`AxisConcatProvider.spans`, so a caller reporting which file an index came
        from does not care which of the two it is reading through."""
        return tuple((self.labels[i], self._off[i], self._ext[i])
                     for i in range(len(self._mem)))

    def locate(self, i: int) -> Tuple[int, int]:
        """Chained-axis index ``i`` -> ``(member index, that member's own index)``."""
        ii = min(max(0, int(i)), max(0, int(getattr(self.axes, self._axis)) - 1))
        j = bisect_left(self._off, ii + 1) - 1
        j = min(max(0, j), len(self._mem) - 1)
        return j, ii - self._off[j]

    def level_axes(self, level: int) -> AxisSizes:
        # A member owns the spatial pyramid; the re-addressed axes are this provider's own
        # and never halve.
        base = self._mem[0][0].level_axes(level)
        run = int(getattr(self.axes, self._axis))
        return replace(base, m=run) if self._axis == "m" else \
            replace(base, m=self._n, **{self._axis: run})

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        idx = {"t": t, "z": z, "c": c}
        if self._axis == "m":
            member_i, local = self.locate(m)
            m_in = self._mem[member_i][1] + local
        else:
            member_i, local = self.locate(idx[self._axis])
            idx[self._axis] = local
            m_in = self._mem[member_i][1] + int(m)
        return np.asarray(self._mem[member_i][0].read_region(
            level, m_in, idx["t"], idx["z"], idx["c"], y0, y1, x0, x1, b=b))

    def fingerprint(self) -> tuple:
        return ("axis_respread", self._axis,
                tuple((p.fingerprint(), s, n) for p, s, n in self._mem))

    @property
    def version(self) -> Any:
        return ("axis_respread", self._axis,
                tuple((p.version, s, n) for p, s, n in self._mem))


__all__ = ["TileProvider", "SyntheticProvider", "B2ndProvider", "ArrayProvider",
           "ChannelMergeProvider", "FrameSubsetProvider", "FrameSliceProvider",
           "FrameRemapProvider",
           "MultiSourceProvider", "AxisConcatProvider", "AxisRespreadProvider",
           "BatchProvider", "BatchSliceProvider", "CONCAT_AXES", "RESPREAD_AXES",
           "subset_index"]
