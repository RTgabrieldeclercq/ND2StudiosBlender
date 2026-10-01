"""Direct, **no-ingest** ND2 reading — the seam that makes a series too big to copy
usable the moment it is picked.

``nodelab_v2.ingest`` turns an ND2 into a ``.b2nd`` store once and serves every later read
out of that store. That is the right trade for an ordinary series: the store compresses,
carries a display pyramid, and decompresses one planar block per window. It is the wrong
trade the moment the *copy* is the dominant cost. The lab's ``Channel640_Seq0001.nd2`` is
453 GB (54 positions × 74 timepoints × 54 z, 1024², uint16, **uncompressed**); ingesting it
means writing a ~280 GB second copy over many hours before a single pixel can be looked at,
and an ingest that dies partway leaves a torn store that the next open has to start over.

This module is the other option: read the planes **where they already are**. It is worth
having only because of what a modern ND2 actually is —

* ``nd2``'s modern reader resolves a frame to an offset in the file's chunkmap and returns
  ``np.ndarray(buffer=self._mmap, offset=..., strides=...)``. For an **uncompressed** file
  that is a *zero-copy memory-mapped view*: no decode, no allocation. The read happens as
  page faults when the pixels are touched, so slicing a window before copying costs only
  the pages that window covers, and the OS page cache — not a cache of ours — keeps the
  hot planes resident.
* Measured here on that 453 GB file (internal NVMe): **6.2 ms** for a random 1024² plane
  single-threaded, and 8.2 ms/plane effective across 4 threads (255 MB/s aggregate). A
  whole 54-plane z-volume lands in ~60 ms. That is already inside the frame budget the
  Viewer scrubs at, which is why no ingest is needed to make the file *interactive*.

**This IS the default** (``io.load``'s ``access`` mode, :data:`~nodelab_v2.ops.ACCESS_DEFAULT`)
— every file opens this way unless a card is told otherwise. It is still not free of
trade-offs, and the mode exists precisely so those trade-offs stay a manual override rather
than a silent policy: the store wins wherever a plane is large enough that the display
pyramid matters (the lab's 3-channel ``p53-GFP`` series has 6247×4506 planes, 56 MB each,
and direct-reading one costs ~275 ms against a stored level-2's ~4 MB), direct reading gives
up compression, and it depends on the file staying put. ``access="auto"`` picks up an
already-ingested file's store for free (direct on its own never looks for one) while still
defaulting new and unfamiliar files to no copy at all; ``access="ingest"`` forces the store
unconditionally.

**Why the fast path proves itself at open.** It bypasses the sanctioned reader
(``ND2File.to_dask()``), so it has to re-derive two things the SDK would have done: which
file coordinate a ``(m,t,z)`` maps to, and where the channel axis sits in a frame. Both are
file-shaped and both are easy to get quietly wrong — while writing this, the obvious guess
for the second (channel *last*, as ``_TIFF_TO_ND`` would suggest) was wrong: the public
``ND2File.read_frame`` ends in ``transpose((2, 0, 1, 3)).squeeze()``, so a 3-channel file
hands back ``(C, Y, X)`` while a true RGB camera — where the components are ``S``, not
``C`` — squeezes to ``(Y, X, 3)``. A shape-matching resolver plus :meth:`verify` against
``to_dask`` on a handful of random planes is what caught that, and it runs on every open
(``verify=3``) precisely because a silent axis swap reads as real data.

Qt-free, and imported lazily by the runner so a build with no ``nd2`` SDK still starts.
"""
from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes
from nodegraph.provider import TileProvider, _plane_mean_2x

#: ND2 experiment-loop type → the canonical nodegraph axis it iterates. Mirrors
#: :data:`nodelab_v2.ingest._ND_TO_CANON` and is deliberately just as strict: a loop this
#: table does not name (``CustomLoop``, and anything a future SDK adds) means we cannot say
#: which axis a coordinate belongs to, so :class:`Nd2DirectProvider` refuses the file rather
#: than guessing and mis-addressing every frame — an ``access="auto"`` card falls back to
#: ingesting it; a card explicitly on ``"direct"`` (the default) surfaces the refusal and
#: needs its Access switched by hand.
_LOOP_AXIS = {
    "TimeLoop": "t",
    "NETimeLoop": "t",
    "XYPosLoop": "m",
    "ZStackLoop": "z",
}

#: Pyramid depth a direct provider reports. There is no stored pyramid — level *l* is
#: mean-pooled from the level-0 pixels on read (:meth:`Nd2DirectProvider.read_region`) —
#: so this is chosen to match :data:`nodelab_v2.ingest.PYRAMID_LEVELS`: the Viewer picks a
#: level from ``levels``, and reporting fewer than a store would makes the same file
#: display at a different resolution depending on how it was opened.
DIRECT_LEVELS = 3

#: Random planes cross-checked against ``to_dask()`` at open. Small because the check is
#: about *layout*, not pixels: an index or axis error is wrong on every plane, so three
#: disagreeing draws is as conclusive as three hundred. The cost is dominated by building
#: the dask graph once (~1.7 s on the 453 GB file, which has 215,784 blocks), not by the
#: planes.
_VERIFY_PLANES = 3

_GIB = 1 << 30


class Nd2DirectError(RuntimeError):
    """This ND2 cannot be served directly — the caller should ingest it instead.

    Always carries *why* in its message, because the fallback is silent and expensive: a
    user who asked for direct access and got a multi-hour ingest deserves the reason on the
    card rather than in a log nobody opens."""


def can_open_direct(path: str) -> Tuple[bool, str]:
    """``(ok, reason)`` — whether :class:`Nd2DirectProvider` can serve ``path``.

    A cheap, read-only probe for the GUI: it opens the file's headers (~3 ms) and never
    touches pixels, so a source card can offer or grey out the direct mode without paying
    for it. ``reason`` is empty when ``ok``; otherwise it is the sentence to show."""
    try:
        prov = Nd2DirectProvider(path, verify=0)
    except Nd2DirectError as exc:
        return False, str(exc)
    except Exception as exc:                       # noqa: BLE001 — unreadable file etc.
        return False, f"{type(exc).__name__}: {exc}"
    prov.close()
    return True, ""


class Nd2DirectProvider(TileProvider):
    """A :class:`~nodegraph.provider.TileProvider` that reads planes straight out of an
    uncompressed ``.nd2``, with no ingest and no store.

    **Handles are per-thread, not locked.** ``nd2`` guards ``ND2File`` with an internal
    lock, and the sanctioned ``to_dask()`` path takes it for every block — correct, but it
    serialises exactly the fan-out the engine relies on (parallel pulls, tiled reads). An
    ``ND2File`` open measured 3 ms here, so a handle per worker thread is cheaper than the
    contention it removes: the 4-thread read above scales to 255 MB/s on per-thread handles
    and would not on a shared one. Every handle opened is tracked so :meth:`close` can shut
    them; the mmap itself is shared by the OS, so N handles do not mean N copies.

    **There is no plane cache here on purpose.** The obvious optimisation — memoise decoded
    planes so a tiled read does not re-read one — buys nothing: the frame *is* an mmap view,
    so the tiles of one plane slice the same already-faulted pages, and the OS page cache
    already holds them. Adding an LRU would duplicate resident pages in the heap and fight
    the engine's own :class:`~nodegraph.streaming.TileCache` for the memory budget."""

    def __init__(self, path: str, *, levels: int = DIRECT_LEVELS,
                 verify: int = _VERIFY_PLANES) -> None:
        if not os.path.isfile(path):
            raise Nd2DirectError(f"No such ND2 file: {path!r}")
        self.path = os.path.abspath(path)
        self._tl = threading.local()
        self._handles: List[Any] = []            # every handle opened, for close()
        self._hlock = threading.Lock()
        self._closed = False

        f = self._handle()
        # Refuse anything whose frames are not a flat mmap. `read_frame` still *works* for
        # a lossless-compressed file, but it zlib-decompresses a whole frame per call, so
        # the property this provider is built on — a window costing only its own pages —
        # is gone and the store is the better answer again.
        comp = getattr(getattr(f, "attributes", None), "compressionType", None)
        if comp is not None:
            raise Nd2DirectError(
                f"{os.path.basename(path)} is {comp}-compressed; direct reading is only "
                f"faster than a store for UNCOMPRESSED ND2s. Load it normally instead.")
        if getattr(f, "is_legacy", False):
            raise Nd2DirectError(
                f"{os.path.basename(path)} is a legacy (JPEG-2000 era) ND2; the legacy "
                f"reader has no memory-mapped frame path. Load it normally instead.")

        s = f.sizes
        if "S" in s and "C" in s:
            # The same duplicate-mapping refusal `ingest._nd2_dims` makes, for the same
            # reason: both would fold onto `c` and one would silently overwrite the other.
            raise Nd2DirectError(
                f"{os.path.basename(path)} carries both C and S (RGB) axes; direct "
                f"reading cannot place both on the channel axis. Load it normally.")
        self.axes = AxisSizes(m=int(s.get("P", 1)), t=int(s.get("T", 1)),
                              z=int(s.get("Z", 1)), c=int(s.get("C", s.get("S", 1))),
                              y=int(s.get("Y", 1)), x=int(s.get("X", 1)))
        self.levels = max(1, int(levels))
        self.tile = 512

        # ── the (m,t,z) → file sequence mapping ────────────────────────────────────
        # `_coord_shape` is the file's own coordinate shape in EXPERIMENT order, which is
        # not the canonical order and is not fixed across files: the 640 series is
        # (TimeLoop, XYPosLoop, ZStackLoop) and reading it as (m,t,z) would address the
        # wrong frame for every coordinate but the diagonal.
        loops = [(str(getattr(lp, "type", "")), int(getattr(lp, "count", 1)))
                 for lp in f.experiment]
        unknown = [name for name, _ in loops if name not in _LOOP_AXIS]
        if unknown:
            raise Nd2DirectError(
                f"{os.path.basename(path)} has unsupported acquisition loop(s) "
                f"{', '.join(sorted(set(unknown)))}; direct reading cannot say which axis "
                f"they iterate. Load it normally instead.")
        self._order: Tuple[str, ...] = tuple(_LOOP_AXIS[name] for name, _ in loops)
        self._cshape: Tuple[int, ...] = tuple(n for _, n in loops)
        if len(set(self._order)) != len(self._order):
            raise Nd2DirectError(
                f"{os.path.basename(path)} iterates one axis twice "
                f"({'+'.join(self._order)}); direct reading cannot flatten that. "
                f"Load it normally instead.")

        # ── where the channel axis sits in a frame ─────────────────────────────────
        probe = f.read_frame(0)
        self._chan_axis = self._resolve_channel_axis(tuple(probe.shape))
        self.dtype = np.dtype(probe.dtype)

        if verify:
            self.verify(int(verify))

    # ── layout resolution ─────────────────────────────────────────────────────────
    def _resolve_channel_axis(self, shape: Tuple[int, ...]) -> Optional[int]:
        """Which axis of a ``read_frame`` result indexes the channel — or ``None`` when
        the frame is a bare ``(Y, X)``.

        Resolved by matching the frame's shape against the geometry we already know rather
        than by assuming a layout, because the layout genuinely differs: ``read_frame``
        transposes to ``(C, Y, X, RGBcomponents)`` and then ``squeeze()``s, so a
        3-channel file arrives ``(C, Y, X)`` and an RGB camera — one channel, three
        components — arrives ``(Y, X, 3)``."""
        y, x, c = self.axes.y, self.axes.x, self.axes.c
        if shape == (y, x):
            if c != 1:
                raise Nd2DirectError(
                    f"frame is {shape} but the file declares {c} channels; direct "
                    f"reading cannot locate the channel axis. Load it normally instead.")
            return None
        if shape == (c, y, x):
            return 0
        if shape == (y, x, c):
            return 2
        raise Nd2DirectError(
            f"unrecognised ND2 frame layout {shape} for a "
            f"{c}-channel {y}x{x} image; direct reading cannot address it. "
            f"Load it normally instead.")

    # ── file handles ──────────────────────────────────────────────────────────────
    def _handle(self) -> Any:
        """This thread's ``ND2File``, opened on first use. See the class docstring for why
        a handle per thread beats one shared handle behind ``nd2``'s own lock."""
        if self._closed:
            raise Nd2DirectError(f"provider for {self.path!r} is closed")
        h = getattr(self._tl, "handle", None)
        if h is None:
            from nodelab_v2.nd2_compat import import_nd2
            h = import_nd2().ND2File(self.path)
            self._tl.handle = h
            with self._hlock:
                self._handles.append(h)
        return h

    def close(self) -> None:
        """Close every ``ND2File`` this provider opened, on whatever thread opened it.

        Worth having rather than leaning on garbage collection: ``nd2`` emits a
        ``UserWarning`` per un-closed handle at interpreter shutdown, and a provider that
        is dropped when its source card is deleted would otherwise keep the file locked on
        Windows — which is how a re-open of the same path fails with a sharing violation."""
        self._closed = True
        with self._hlock:
            handles, self._handles = self._handles, []
        for h in handles:
            try:
                h.close()
            except Exception:                      # noqa: BLE001 — already closed/torn
                pass
        self._tl = threading.local()

    # ── the frame read ────────────────────────────────────────────────────────────
    def _seq(self, m: int, t: int, z: int) -> int:
        """The flat ND2 sequence index for a canonical coordinate."""
        if not self._cshape:
            return 0
        pick = {"m": m, "t": t, "z": z}
        coords = tuple(pick[a] for a in self._order)
        return int(np.ravel_multi_index(coords, self._cshape))

    def _frame(self, m: int, t: int, z: int, c: int) -> np.ndarray:
        """The ``(Y, X)`` **view** for one coordinate — memory-mapped, not copied.

        Returning a view rather than an array is the whole point: the caller slices its
        window out of this and copies only that, so a 512² tile faults 0.5 MB instead of
        the plane's 2 MB."""
        fr = self._handle().read_frame(self._seq(m, t, z))
        if self._chan_axis == 0:
            return fr[c]
        if self._chan_axis == 2:
            return fr[:, :, c]
        return fr

    def read_region(self, level: int, m: int, t: int, z: int, c: int,
                    y0: int, y1: int, x0: int, x1: int, *, b: int = 0) -> np.ndarray:
        """One single-z window. Level 0 slices the mmap and copies just the window; a
        higher level reads the window's level-0 footprint and mean-pools it.

        The reduction routes through :func:`nodegraph.provider._plane_mean_2x` — the same
        function the stored pyramid is built with — so a plane displayed at level *l* is
        bit-identical whether this file was opened direct or ingested. Using a private
        helper is deliberate: a second, independent downsample here is exactly how the two
        paths would drift into showing subtly different images of the same data."""
        f = 1 << max(0, int(level))
        if f == 1:
            return np.array(self._frame(m, t, z, c)[y0:y1, x0:x1])
        # Clamp the footprint to the real plane: `level_axes` floors, so the last level-l
        # row/column can name a level-0 span that runs off the edge of an odd-sized plane.
        ay, ax = self.axes.y, self.axes.x
        win = self._frame(m, t, z, c)[y0 * f:min(y1 * f, ay), x0 * f:min(x1 * f, ax)]
        out = np.array(win)
        for _ in range(int(level)):
            out = _plane_mean_2x(out)
        return out

    # ── identity ──────────────────────────────────────────────────────────────────
    def fingerprint(self) -> tuple:
        """Content identity for the memo: the file itself, by path + size + ``mtime_ns``.

        The same shape of identity :meth:`B2ndProvider.fingerprint` uses for a disk store,
        and for the same reason — the base *structural* fingerprint would collide two
        different files of equal geometry. The tag differs from the store's, so the same
        ND2 read direct and read through a store are distinct memo identities: they are
        pixel-identical, so sharing would be sound, but the two differ in dtype promotion
        at higher levels and a false hit is far more expensive to debug than a cold key."""
        try:
            st = os.stat(self.path)
            ident: Any = (st.st_size, st.st_mtime_ns)
        except OSError:                            # unlinked mid-session
            ident = None
        ax = self.axes
        return ("nd2-direct", self.path, ident,
                (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), str(self.dtype), self.levels)

    # ── the self-check ────────────────────────────────────────────────────────────
    def verify(self, planes: int = _VERIFY_PLANES, *, seed: int = 0) -> None:
        """Cross-check this provider's addressing against ``ND2File.to_dask()`` on
        ``planes`` pseudo-random coordinates, raising :class:`Nd2DirectError` on any
        disagreement.

        This is the guard that makes the fast path safe to ship. It compares against the
        *sanctioned* reader — the one :mod:`nodelab_v2.ingest` uses — so it is testing the
        two things this module re-derives (the sequence index and the channel axis) against
        the SDK's own answer, on the actual file in hand. It found a real bug during
        development: indexing the channel axis last, which is right for a TIFF and wrong
        for a 3-channel ND2.

        Deterministic by default (``seed``) so a failure names coordinates that can be
        re-tested, and cheap by design — see :data:`_VERIFY_PLANES`."""
        import random

        from nodelab_v2.ingest import _to_6d

        f = self._handle()
        lazy = _to_6d(f.to_dask(), list(f.sizes.keys()))
        rnd = random.Random(seed)
        ax = self.axes
        for _ in range(max(1, int(planes))):
            m = rnd.randrange(ax.m)
            t = rnd.randrange(ax.t)
            z = rnd.randrange(ax.z)
            c = rnd.randrange(ax.c)
            want = np.asarray(lazy[m, t, z, c])
            got = np.asarray(self._frame(m, t, z, c))
            if got.shape != want.shape or not np.array_equal(got, want):
                raise Nd2DirectError(
                    f"direct read of {os.path.basename(self.path)} disagrees with the "
                    f"SDK at (m={m}, t={t}, z={z}, c={c}): got {got.shape}, expected "
                    f"{want.shape}. Refusing the direct path — load it normally.")


# ── the `access=auto` decision ──────────────────────────────────────────────────────
#
# `io.load`'s Access mode defaults to "auto" (`nodelab_v2.ops.ACCESS_MODE`). Half of
# what "auto" means — reuse a store that already exists — lives in
# `nodelab_v2.runner.EngineRunner._effective_access`, because that half needs the
# runner's own cache and `open_store`. What lives here is the OTHER half: given that
# there is nothing usable on disk yet, would a fresh ingest fit?

def _estimate_ingest_bytes(axes: AxisSizes, itemsize: int, levels: int = DIRECT_LEVELS
                           ) -> int:
    """A deliberately PESSIMISTIC estimate of what ingesting this geometry would write to
    disk: level 0 at its full RAW size — no compression credited, because blosc2's ratio
    depends on the data and this is a go/no-go check, not a benchmark to defend — plus the
    pyramid's geometric tail. Each level halves Y and X, so it costs 1/4 of the level below
    it; ``levels`` of them sum to a fixed multiplier (1.3125x of level 0 for the shipped 3
    levels).

    Real ingests come in under this: the lab's compressed 3-channel sibling file's WHOLE
    store, pyramid included, was smaller than its raw pixel count (10.15 GB store for
    12.67 GB of pixels). A file this rejects would have had to overshoot an already
    generous number to actually fail."""
    raw = axes.m * axes.t * axes.z * axes.c * axes.y * axes.x * itemsize
    multiplier = sum(4 ** -lvl for lvl in range(max(1, int(levels))))
    return int(raw * multiplier)


def _free_bytes(dir_path: str) -> Optional[int]:
    """Free space on the drive holding ``dir_path``, or ``None`` if it cannot be told.

    ``shutil.disk_usage`` needs a path that already EXISTS, and a store's own directory
    usually does not — this decision runs precisely because nothing has been ingested yet
    — so this walks up to the nearest existing ancestor. The drive is the same either
    way, which is the only thing being asked."""
    import shutil
    probe = os.path.abspath(dir_path)
    seen = set()
    while probe and probe not in seen and not os.path.isdir(probe):
        seen.add(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not probe or not os.path.isdir(probe):
        return None
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _fits_on_disk(needed_bytes: int, free_bytes: int) -> bool:
    """Whether an ingest estimated at ``needed_bytes`` should be attempted given
    ``free_bytes`` free on the destination drive.

    The margin on top — 25%, or 5 GiB, whichever is larger — exists for two reasons at
    once: ``needed_bytes`` is itself an estimate with no compression credited, and the
    drive is not this app's alone to fill. Running it to the literal last byte is how the
    640 series' first ingest died and left a torn store behind; this is the check that
    stops the next one before it starts, not after."""
    margin = max(int(needed_bytes * 0.25), 5 * _GIB)
    return free_bytes >= needed_bytes + margin


def decide_access(path: str, store_dir_path: str) -> Tuple[str, str]:
    """The ``auto`` access decision for one file with **nothing usable on disk yet** —
    ``(access, reason)``, where ``access`` is always a concrete ``"ingest"``/``"direct"``
    (never ``"auto"`` itself, and this function never sees a file that already has a
    valid store — see the module-level note above).

    * If the file cannot be read directly at all — compressed, legacy, a TIFF, an
      unsupported acquisition loop (:class:`Nd2DirectProvider`'s own refusals) — ingest
      is the only option; there is nothing to decide.
    * Otherwise both are genuinely on the table, and the decision is disk space: would a
      fresh ingest fit, with room to spare (:func:`_fits_on_disk`)? This is the fix for
      the actual failure that motivated this module — a 453 GB series whose ingest needs
      roughly 340+ GB, attempted on a drive that had 45 GB free. It ran the disk to zero
      bytes free and died with a torn store.
    * Comfortable room → ``ingest``, unchanged from every file before this mode existed:
      compression and a display pyramid are free when they fit."""
    from nodelab_v2.ingest import _is_tiff
    from nodelab_v2.ops import ACCESS_DIRECT, ACCESS_INGEST
    if _is_tiff(path):
        return ACCESS_INGEST, "TIFF has no direct-read path"
    try:
        prov = Nd2DirectProvider(path, verify=0)
    except Nd2DirectError as exc:
        return ACCESS_INGEST, str(exc)
    except Exception as exc:                      # noqa: BLE001 — any unreadable file
        return ACCESS_INGEST, f"{type(exc).__name__}: {exc}"
    try:
        needed = _estimate_ingest_bytes(prov.axes, prov.dtype.itemsize)
    finally:
        prov.close()
    free = _free_bytes(store_dir_path)
    if free is None:
        return ACCESS_INGEST, "could not check free disk space on the destination drive"
    if _fits_on_disk(needed, free):
        return ACCESS_INGEST, (f"fits comfortably: an ingest needs up to "
                               f"{needed / _GIB:.0f} GB and the drive has "
                               f"{free / _GIB:.0f} GB free")
    return ACCESS_DIRECT, (f"an ingest could need up to {needed / _GIB:.0f} GB (no "
                           f"compression credited — a go/no-go check, not a benchmark) "
                           f"and the drive has only {free / _GIB:.0f} GB free")


__all__ = ["Nd2DirectProvider", "Nd2DirectError", "can_open_direct",
           "decide_access", "DIRECT_LEVELS"]
