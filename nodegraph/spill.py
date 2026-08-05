"""Out-of-core dense outputs — a per-voxel layer too large for RAM (V2.24).

Most of the catalog's cost problems are solved by *laziness*: a filter returns a provider
and only the planes someone looks at are ever computed (V2.04 §6b). That does not work for
a **dense structure output**, and the two cases here are exactly why:

* ``analysis.threshold``'s Voxel ``mask`` is pointwise, but it is a ``Domain.VOXEL``
  attribute **layer**, and a layer is an array — there is no lazy layer.
* ``analysis.segment``'s label raster cannot be lazy even in principle: ids are assigned
  **globally unique** by folding the units in order, so plane *n*'s labels depend on how
  many objects the previous *n-1* planes held.

Both therefore allocate one array over the whole ``(M,T,Z,C,Y,X)`` grid, and on a real
acquisition that array is the whole problem. The lab's 640 series is
12 × 16 × 210 × 1 × 1024 × 1024 = 42.3 **Gvoxel**, which is 39.4 GiB as ``uint8`` and
315 GiB as ``int64`` — so ``threshold`` and ``segment`` both died with
``numpy._core._exceptions._ArrayMemoryError`` the moment an upstream Z-Project was reset
to ``none`` and Z came back at full extent (2026-08-04). With the projection ON the same
graph needs 1/210th of that and runs fine, which is what made the failure read as
Z-Project's fault rather than as a hard ceiling in the nodes downstream.

So: above :func:`spill_budget` a dense output is created as a ``.npy`` on disk and written
through an ``np.memmap``, then handed on as a **read-only** mapping. That is not a new
concept in the codebase — it is the same shape a docked checkpoint already serves its
Voxel layers with (V2.18), and the two invariants that make it safe are already in place
and load-bearing:

* :class:`~nodegraph.dataset.AttributeLayer` does NOT copy a read-only ``np.memmap``
  (``dataset.py`` — copying is what a dock exists to avoid), so sealing one into a layer
  costs nothing.
* :func:`~nodegraph.memo.payload_bytes` counts a memmapped layer as **0** retained bytes,
  because evicting the entry frees nothing — the pages belong to the OS page cache against
  a file, not to this process's heap.
* :func:`~nodegraph.memo.output_fingerprint` hashes a Dataset by its layer **revisions**,
  never by layer content, so a spilled layer is not read back to be hashed. Content-hashing
  a 315 GiB mapping would obviously defeat the entire point.

**Determinism is unaffected.** Where the bytes live is not part of any identity: the recipe
hash is over params/reads, and the output fingerprint is over revisions. A graph that
spills and one that does not produce the same values, so a node must never branch its
*arithmetic* on :func:`spilling` — only its allocation.

**Lifetime** is the session. Files land in one per-process directory (:func:`spill_dir`)
removed at exit, and because a live mapping cannot be deleted on Windows, that removal is
best-effort and a **sweep of older sessions' directories** runs first. The consequence to
be honest about: a hard-killed process can leave a spill directory behind, and the next run
is what clears it.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading
import time

import numpy as np

from typing import Optional, Tuple

from nodegraph.parallel import _GiB, _MiB, _env_int, ram_budget

#: Directory-name prefix, shared by the creator and the stale-sweep so one session can
#: recognise another's leftovers without a registry.
_PREFIX = "nodegraph-spill-"

_dir_lock = threading.Lock()
_session_dir: Optional[str] = None
_counter = 0


def spill_budget() -> int:
    """Dense-output ceiling: above this many bytes the array goes to disk.

    25% of RAM, floored at 256 MiB and capped at 64 GiB — a *share* of the machine for the
    same reason every other budget here is (:func:`~nodegraph.parallel.ram_budget`), since
    what matters is whether the output still leaves room for the caches and the node's own
    working set, not whether it fits once.

    The cap matches :func:`~nodegraph.parallel.tile_cache_bytes`' rather than the memo's
    because spilling is not free and RAM is the faster answer whenever it is available: on
    the lab's 256 GiB box a 32 GiB cap sent Threshold's 39.4 GiB mask to disk with 218 GiB
    of RAM sitting idle, trading a fast allocation for a 39.4 GiB write. At 64 GiB that
    mask stays in memory there and still spills on a 64 GiB machine (16 GiB budget), which
    is the behaviour wanted in both places.

    ``NODEGRAPH_SPILL_BYTES`` overrides it; ``0`` forces every dense output to spill and a
    very large value disables spilling entirely (the pre-V2.24 behaviour), which is how the
    selftest runs the two paths against each other.
    """
    override = _env_int("NODEGRAPH_SPILL_BYTES", None)
    if override is not None:
        return max(0, int(override))
    return ram_budget(0.25, floor=256 * _MiB, cap=64 * _GiB)


#: Only spill directories older than this are swept. The age gate is what makes the sweep
#: safe beside a CONCURRENT second instance of the app: its directory is minutes old, and
#: "delete every directory with our prefix" would otherwise race it. Deletion of a live
#: mapping fails on Windows anyway, but :meth:`DenseOut.seal` briefly closes the writable
#: mapping before re-opening read-only, and a sibling sweeping in that window could take
#: the file. 12 hours is far longer than that window and far shorter than "never".
_STALE_AFTER_S = 12 * 3600


def _sweep_stale(parent: str) -> None:
    """Remove spill directories left by earlier sessions (best-effort).

    This is the only thing that reclaims the disk after a hard kill, since the exit hook
    cannot run then. Every error is swallowed: a directory still held open by a live
    process must not be removed, and failing to remove it is the correct outcome.

    Called before this session's own directory exists (``mkdtemp`` runs after it), so there
    is nothing of ours to exclude."""
    try:
        names = os.listdir(parent)
    except OSError:
        return
    now = time.time()
    for name in names:
        if not name.startswith(_PREFIX):
            continue
        path = os.path.join(parent, name)
        try:
            if now - os.stat(path).st_mtime < _STALE_AFTER_S:
                continue                          # too young to be certainly abandoned
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)


def spill_dir() -> str:
    """This process's spill directory, created on first use.

    ``NODEGRAPH_SPILL_DIR`` names the PARENT to create it under — the point of the
    override being the same as ``NODEGRAPH_STORE_DIR``'s: spill I/O should be able to land
    on fast media, and the system temp directory is not always on it."""
    global _session_dir
    with _dir_lock:
        if _session_dir is not None:
            return _session_dir
        parent = os.environ.get("NODEGRAPH_SPILL_DIR", "").strip().strip('"').strip("'")
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:                      # unwritable override → system temp
                parent = ""
        parent = parent or tempfile.gettempdir()
        _sweep_stale(parent)
        _session_dir = tempfile.mkdtemp(prefix=_PREFIX, dir=parent)
        atexit.register(_cleanup)
        return _session_dir


def _cleanup() -> None:
    """Drop this session's directory at exit. Best-effort: on Windows a file with a live
    mapping cannot be unlinked, and at ``atexit`` time the interpreter has not yet dropped
    the arrays that hold them — so what this reliably cleans is the common case, and
    :func:`_sweep_stale` is what catches the rest on the next run."""
    global _session_dir
    path, _session_dir = _session_dir, None
    if path:
        shutil.rmtree(path, ignore_errors=True)


def spilling(shape: Tuple[int, ...], dtype) -> bool:
    """Would an array of this shape/dtype be spilled? (The budget test, in one place.)"""
    return _nbytes(shape, dtype) > spill_budget()


def _nbytes(shape: Tuple[int, ...], dtype) -> int:
    n = 1
    for s in shape:
        n *= max(0, int(s))
    return n * int(np.dtype(dtype).itemsize)


#: Bytes written between flushes of a spilled output. Small enough that the modified-page
#: set stays bounded, large enough that the flush cost amortizes — see :class:`_SpillView`.
_FLUSH_EVERY = int(_env_int("NODEGRAPH_SPILL_FLUSH_BYTES", None) or (1 * _GiB))


class _SpillView:
    """The writable face of a spilled output: an array-like that **flushes as it fills**.

    Spilling to a mapping does not bound memory on its own, and that was the whole point —
    so this is not an optimization but the load-bearing half of the mechanism. Pages written
    through an ``np.memmap`` enter the process working set and stay *modified* (dirty) until
    something flushes them; the OS cannot reclaim a modified page. Writing a 315 GiB label
    raster therefore climbed to **208 GiB RSS** and left 5.3 GiB of the machine's 256 GiB
    available — at which point ``analysis.label``'s next per-volume allocation failed with
    ``MemoryError: Unable to allocate 1.64 GiB`` (measured 2026-08-04). The node had traded
    one 315 GiB allocation for the same amount of dirty page cache; a different error
    message for the same exhaustion.

    So every ``_FLUSH_EVERY`` bytes assigned, the mapping is flushed: the pages go clean,
    the OS can move them to the standby list, and they count as available again. Only
    ``__setitem__`` is instrumented because that is how every node writes its output
    (``mask[m, t, z, c] = ...``, ``raster[m, t, :, c] = ...``); the rest of the array
    protocol is forwarded so callers cannot tell the difference.

    The counter is locked because ``analysis.threshold`` writes from a thread pool.
    """

    __slots__ = ("_a", "_pending", "_lock")

    def __init__(self, a: np.ndarray) -> None:
        self._a = a
        self._pending = 0
        self._lock = threading.Lock()

    def __setitem__(self, key, value) -> None:
        self._a[key] = value
        n = getattr(value, "nbytes", None)
        if n is None:
            v = np.asarray(value)
            n = v.nbytes or int(self._a.dtype.itemsize)
        with self._lock:
            self._pending += int(n)
            if self._pending < _FLUSH_EVERY:
                return
            self._pending = 0
        # outside the lock: a flush is a syscall over the mapping, and holding the lock
        # across it would serialize the pool's writers on it
        self._a.flush()

    # ── forwarded array protocol ──────────────────────────────────────────────
    def __getitem__(self, key):
        return self._a[key]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._a, dtype=dtype) if dtype is not None else np.asarray(self._a)

    def __len__(self) -> int:
        return len(self._a)

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def ndim(self) -> int:
        return self._a.ndim

    @property
    def size(self) -> int:
        return self._a.size

    @property
    def nbytes(self) -> int:
        return self._a.nbytes

    def flush(self) -> None:
        with self._lock:
            self._pending = 0
        self._a.flush()


class DenseOut:
    """A zero-filled dense output array, in RAM or on disk, plus how to seal it.

    Write into :attr:`array` exactly as into an ``np.zeros`` of the same shape — a memmap
    supports the same indexed assignment, so no caller needs to know which it got. Then
    call :meth:`seal` and put the result in the layer: it returns a read-only array that
    :class:`~nodegraph.dataset.AttributeLayer` will accept without copying.

    Zero-fill: a freshly created file is zero-filled by both POSIX and Windows when it is
    extended to length, and ``open_memmap(mode="w+")`` creates at full length — so pages
    never written read as 0. That is relied upon (a mask's background, a label raster's
    unlabelled voxels are never assigned explicitly).
    """

    __slots__ = ("array", "path", "_mm")

    def __init__(self, shape: Tuple[int, ...], dtype, *, tag: str = "layer") -> None:
        global _counter
        self.path: Optional[str] = None
        self._mm: Optional[np.ndarray] = None
        if not spilling(shape, dtype):
            self.array = np.zeros(shape, dtype=dtype)
            return
        d = spill_dir()
        with _dir_lock:
            _counter += 1
            n = _counter
        # the pid is in the DIRECTORY name (mkdtemp); the counter only has to be unique
        # within this process, and a plain lock-guarded int is that without importing uuid
        self.path = os.path.join(d, f"{n:04d}_{_slug(tag)}.npy")
        self._mm = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=np.dtype(dtype), shape=tuple(int(s) for s in shape))
        # the writable face flushes as it fills — without that, spilling bounds nothing
        self.array = _SpillView(self._mm)

    @property
    def spilled(self) -> bool:
        return self.path is not None

    def seal(self) -> np.ndarray:
        """Finish writing and return the array to hand to ``with_layer``.

        In RAM this is the array itself (``AttributeLayer`` freezes it). Spilled, the
        buffer is flushed and **re-opened read-only**, which is what makes it both safe to
        freeze and free to hold: read-only is the flag ``AttributeLayer`` and the Memo GC
        both key their no-copy / zero-bytes behaviour on."""
        if self.array is None:
            raise RuntimeError("DenseOut already sealed")
        if not self.spilled:
            out = self.array
            self.array = None                     # release our own reference
            return out
        self.array.flush()
        self.array = None                         # drop the writable faces, in order:
        self._mm = None                           # view first, then the mapping itself
        return np.load(self.path, mmap_mode="r")


def _slug(s: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(s or "")]
    return ("".join(keep)[:40] or "layer")


def dense_output(shape: Tuple[int, ...], dtype, *, tag: str = "layer") -> DenseOut:
    """A :class:`DenseOut` for ``shape``/``dtype`` — RAM under :func:`spill_budget`,
    a disk-backed mapping above it."""
    return DenseOut(shape, dtype, tag=tag)


__all__ = ["DenseOut", "dense_output", "spill_budget", "spill_dir", "spilling"]
