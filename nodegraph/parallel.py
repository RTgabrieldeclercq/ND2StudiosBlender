"""Parallel execution policy for the nodegraph engine (V2.14 — machine-aware run).

The v2 engine is a *lazy pull* scheduler: :meth:`nodegraph.engine.Engine.pull` walks
backward and each node either returns a streaming provider (cost deferred to read time)
or realizes its units eagerly. Both shapes are, until this module, strictly
**single-threaded** — one core of however many the machine has.

This module is the one place that decides *how wide* to run, and it exists because the
answer is not "as wide as possible":

* **Determinism is non-negotiable.** Several eager nodes fold a running counter across
  units (``analysis.segment``/``analysis.label`` shift each unit's region ids by a global
  offset), so unit ORDER is part of the result. Every map here is therefore
  **order-preserving**, and the intended shape is *parallel pure per-unit compute →
  serial deterministic fold*. See :func:`map_units`.
* **The GIL decides thread-vs-process, and the crossover is measured, not guessed.**
  ``scipy.ndimage`` filters release it; numpy elementwise work and most ``skimage`` glue
  do not. On 16 units of 512² on this box:

  =========================================  ==============  ==============
  workload                                   threads         processes
  =========================================  ==============  ==============
  cheap — one ``gaussian_filter`` per unit    **5.88×**       2.31×
  heavy — ``denoise_bilateral`` per unit      1.71×           **5.23×**
  =========================================  ==============  ==============

  So: cheap per-unit work goes on :func:`map_units` (threads, nothing pickled), heavy
  per-unit work goes on :func:`map_units_proc` (processes, which need a module-level
  function). Results were byte-identical on all three paths.
* **Nesting must not multiply.** A parallel plane-read that calls a node whose unit loop
  is also parallel would launch ``N×M`` workers and thrash. :func:`map_units` is
  **re-entrant-safe**: inside an active parallel region it runs serial (see
  :data:`_ACTIVE`).
* **The libraries are already threaded.** ``blosc2`` decompresses on 21 threads here and
  OpenBLAS is built ``MAX_THREADS=24``; running 24 workers on top of that oversubscribes
  a 12-core machine. :func:`inner_threads` is the cap workers set on themselves.

Sizing comes from the machine, not a constant — see :func:`cpu_budget` /
:func:`ram_budget`. Every knob has an environment override so a headless run on a
different box (or a bisect) needs no code edit:

===============================  =====================================================
``NODEGRAPH_PARALLEL``           ``auto`` (default) / ``thread`` / ``process`` / ``off``
``NODEGRAPH_WORKERS``            max workers (default ``min(cpu_count-2, 8)``, floor 1)
``NODEGRAPH_PROC_WORKERS``       process-pool size (default ``min(workers, 8)``)
``NODEGRAPH_CACHE_BYTES``        engine TileCache budget (default: RAM-derived)
``NODEGRAPH_MEMO_BYTES``         persistent Memo budget (default: RAM-derived)
``NODEGRAPH_PLANE_CACHE_BYTES``  GUI decoded-plane cache (default: RAM-derived)
``NODEGRAPH_FLOAT32``            ``1`` = stream in float32 (opt-in; changes numerics)
``NODEGRAPH_GPU``                ``off`` (default) / ``auto`` / ``on`` (CuPy dispatch)
``NODEGRAPH_STORE_DIR``          redirect ``.b2nd`` ingest stores off slow media
===============================  =====================================================

Qt-free; numpy + stdlib only.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_GiB = 1 << 30
_MiB = 1 << 20


# ── environment helpers ────────────────────────────────────────────────────────

def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None or v == "" else v.strip().lower()


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    """An int env override. Accepts a plain integer or a ``<n>g``/``<n>m`` byte suffix,
    so a budget can be written the way a human thinks about it (``48g``). An
    unparseable value is IGNORED rather than fatal — a typo in an env var must not stop
    a run that would otherwise work fine on the defaults."""
    v = os.environ.get(key)
    if v is None or not v.strip():
        return default
    s = v.strip().lower()
    mult = 1
    if s.endswith("g"):
        mult, s = _GiB, s[:-1]
    elif s.endswith("m"):
        mult, s = _MiB, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return default


def _env_flag(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── machine sizing ─────────────────────────────────────────────────────────────

#: Ceiling on the default worker count. The unit pool is a THREAD pool in the GUI's own
#: process, and every worker runs Python glue between its numpy calls, so each one is a
#: contender for the GIL against the GUI thread. Measured 2026-10-08 on an 80-core box
#: (Threshold over a 66-position ND2, direct access): 78 workers 53.6 s with the GUI frozen
#: for 31.6 s at a stretch; 16 workers 69 s; 8 workers 49.7 s with the longest GUI stall
#: 0.3 s. Past a handful of threads the GIL is the bottleneck, so more workers only make
#: the convoy worse — the run gets no faster and every click waits behind it.
#: ``NODEGRAPH_WORKERS`` still sets any number, above or below.
DEFAULT_WORKER_CAP = 8


def cpu_budget() -> int:
    """How many workers to run at once.

    ``cpu_count - 2`` rather than ``cpu_count``: the GUI thread and the Qt worker that
    drives the pull both need to stay responsive, and the point of parallelising is a
    smoother run, not a pegged machine — and never more than :data:`DEFAULT_WORKER_CAP`,
    because on a many-core machine the extra threads fight the GUI for the GIL instead of
    shortening the run. Floor 1 (never zero — a zero-worker pool deadlocks).
    ``NODEGRAPH_WORKERS`` overrides, in either direction."""
    n = _env_int("NODEGRAPH_WORKERS", None)
    if n is not None:
        return max(1, int(n))
    return max(1, min((os.cpu_count() or 2) - 2, DEFAULT_WORKER_CAP))


def proc_budget() -> int:
    """Process-pool size. Capped below :func:`cpu_budget` by default because a spawned
    worker re-imports numpy/scipy/skimage (seconds) and each holds its own copy of the
    payload — 8 is enough to saturate a 12-core box on work heavy enough to deserve
    processes at all. ``NODEGRAPH_PROC_WORKERS`` overrides."""
    n = _env_int("NODEGRAPH_PROC_WORKERS", None)
    if n is not None:
        return max(1, int(n))
    return max(1, min(cpu_budget(), 8))


def total_ram_bytes() -> int:
    """Installed physical RAM. Best-effort and deliberately dependency-free: Windows via
    ``GlobalMemoryStatusEx``, POSIX via ``sysconf``. Falls back to 8 GiB — a conservative
    guess that reproduces the old fixed 1 GiB-scale budgets rather than over-committing a
    machine we failed to measure."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        else:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001 — sizing is advisory; never fail a run over it
        pass
    return 8 * _GiB


def ram_budget(share: float, *, floor: int, cap: int) -> int:
    """``share`` of installed RAM, clamped to ``[floor, cap]``.

    Budgets are a FRACTION of the machine rather than a constant because the number that
    matters is not "how much cache is nice" but "is the largest unit I process still
    under ``budget // 2``" — the gate in :func:`nodegraph.nodes._map_image` that decides
    lazy-per-unit versus eager-whole-6-D. On a 256 GB box the old 1 GiB default put that
    ceiling at a 67-Mvoxel volume, so anything past ~16 planes of 2048² silently
    realized the entire series."""
    return max(floor, min(int(total_ram_bytes() * share), cap))


def tile_cache_bytes() -> int:
    """Engine :class:`~nodegraph.streaming.TileCache` budget — the streaming tile/plane
    store, and (via ``budget // 2``) the lazy-vs-eager gate. 20% of RAM, capped at
    64 GiB: measured 81.1 s → 25.8 s on a one-plane view of a 3×80×1024² series purely
    by keeping the node on the lazy path."""
    return _env_int("NODEGRAPH_CACHE_BYTES", None) or \
        ram_budget(0.20, floor=1 * _GiB, cap=64 * _GiB)


def memo_bytes() -> int:
    """Persistent :class:`~nodegraph.memo.Memo` budget (GUI runner). 12% of RAM, capped
    at 32 GiB. Eviction only costs a recompute, so this trades RAM for not re-running
    an unrelated chain after every edit."""
    return _env_int("NODEGRAPH_MEMO_BYTES", None) or \
        ram_budget(0.12, floor=512 * _MiB, cap=32 * _GiB)


def plane_cache_bytes() -> int:
    """GUI decoded-plane LRU budget. 4% of RAM, capped at 8 GiB. A decimated 2048²
    uint16 plane is ~8 MB, so the old 512 MiB held only ~64 of them — a 200-frame
    2-channel series re-decoded continuously while scrubbing. 8 GiB holds ~1000."""
    return _env_int("NODEGRAPH_PLANE_CACHE_BYTES", None) or \
        ram_budget(0.04, floor=256 * _MiB, cap=8 * _GiB)


def stream_dtype():
    """The float width streaming providers compute in.

    ``float64`` by default and that default is load-bearing, not inertia:
    :class:`~nodegraph.streaming._AxisReduceProvider` documents that its tiled reduce is
    **byte-identical** to the eager ``astype(float)`` reduce, and the selftest asserts
    equality against reference numpy. float32 breaks that bit-for-bit promise (it is a
    different computation, not a faster one), so it is opt-in via
    ``NODEGRAPH_FLOAT32=1``: 1.3–1.8× on bandwidth-bound kernels (gaussian 1.52×,
    uniform 1.81×, median 1.01× — median is compute-bound) plus 2× effective cache
    capacity, in exchange for ~7 significant digits instead of ~16."""
    import numpy as np
    return np.float32 if _env_flag("NODEGRAPH_FLOAT32") else np.float64


def store_dir(default_dir: str) -> str:
    """Where a ``.b2nd`` ingest store should live.

    Defaults to beside the source file (``default_dir``), which is the right answer until
    the source sits on slow media — a USB SSD here measured 27 MB/s of store write versus
    56 MB/s on the internal NVMe, and 55 MB/s of raw sequential write versus 952 MB/s.
    ``NODEGRAPH_STORE_DIR`` redirects every store to one fast directory instead."""
    override = os.environ.get("NODEGRAPH_STORE_DIR", "").strip().strip('"').strip("'")
    if not override:
        return default_dir
    try:
        os.makedirs(override, exist_ok=True)
    except OSError:            # unwritable override → stay beside the file, don't fail
        return default_dir
    return override


# ── mode + re-entrancy ─────────────────────────────────────────────────────────

#: Thread-local "am I already inside a parallel region?" flag, so a nested map (a parallel
#: plane read whose node also parallelises its unit loop) runs SERIAL instead of launching
#: ``N×M`` workers.
#:
#: It must be set **on the worker thread**, not on the thread that submitted the map: the
#: submitter is blocked inside ``ex.map`` and it is the *workers* that go on to call a
#: nested map. Setting it only on the submitter is a guard that silently does nothing —
#: exactly what ``test_parallel`` now pins. :func:`_mark_region` is the per-task wrapper
#: that sets it; the serial path deliberately does NOT set it, since a map that ran serial
#: has consumed no workers and its callee is free to fan out.
_ACTIVE = threading.local()

#: Set in a spawned pool process by :func:`_worker_init`. A process worker is inside a
#: parallel region for its whole life, so a nested map there stays serial rather than
#: multiplying an already-forked-out fan-out by another factor of N.
_IS_POOL_WORKER = False


def in_parallel_region() -> bool:
    return _IS_POOL_WORKER or bool(getattr(_ACTIVE, "on", False))


def _mark_region(fn: Callable[[T], R]) -> Callable[[T], R]:
    """Wrap a task so the flag is set on whichever pool thread runs it. Cleared after each
    task because pool threads are reused across maps."""
    def run(x):
        _ACTIVE.on = True
        try:
            return fn(x)
        finally:
            _ACTIVE.on = False
    return run


def mode() -> str:
    """``auto`` / ``thread`` / ``process`` / ``off`` (``NODEGRAPH_PARALLEL``)."""
    m = _env_str("NODEGRAPH_PARALLEL", "auto")
    return m if m in ("auto", "thread", "process", "off") else "auto"


def enabled() -> bool:
    return mode() != "off" and cpu_budget() > 1


def inner_threads(workers: int) -> int:
    """Threads a single worker should allow its own libraries (blosc2, OpenBLAS, TF).

    ``blosc2`` already decompresses on 21 threads and OpenBLAS is built with
    ``MAX_THREADS=24``; ``workers`` of those on a 12-core machine is
    ``workers × inner`` runnable threads. Dividing the core count by the worker count
    keeps the product at roughly one thread per core."""
    return max(1, (os.cpu_count() or 2) // max(1, workers))


# ── the process pool (spawn-safe, created once, reused) ─────────────────────────

_POOL: Any = None
_POOL_N = 0
_POOL_LOCK = threading.Lock()


def _worker_init() -> None:
    """Runs once per spawned worker. Marks the process as a pool worker (so a nested map
    inside it stays serial) and pins the worker's own libraries to a small thread count so
    ``N`` workers do not each try to use the whole machine."""
    global _IS_POOL_WORKER
    _IS_POOL_WORKER = True
    n = str(inner_threads(proc_budget()))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, n)
    try:
        import blosc2
        blosc2.set_nthreads(int(n))
    except Exception:  # noqa: BLE001 — blosc2 is optional in a worker
        pass


def process_pool(workers: Optional[int] = None):
    """The shared spawn-based :class:`~concurrent.futures.ProcessPoolExecutor`, created
    on first use and kept for the session.

    Spawn (the only option on Windows) re-imports numpy/scipy/skimage per worker, which
    is seconds — paying it once per session instead of once per node is the difference
    between processes being worth it and not. Returns ``None`` when a pool cannot be
    created (a frozen build, a sandbox with no process rights, or an already-spawned
    worker), so every caller must handle ``None`` by staying on threads."""
    global _POOL, _POOL_N
    if mode() == "off":
        return None
    n = int(workers or proc_budget())
    if n <= 1:
        return None
    # A worker must never build its own pool (that is how a spawn storm starts).
    import multiprocessing as mp
    if mp.current_process().name != "MainProcess":
        return None
    with _POOL_LOCK:
        if _POOL is not None and _POOL_N >= n:
            return _POOL
        if _POOL is not None:                      # need a wider pool — replace it
            try:
                _POOL.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            _POOL = None
        try:
            from concurrent.futures import ProcessPoolExecutor
            _POOL = ProcessPoolExecutor(
                max_workers=n, mp_context=mp.get_context("spawn"),
                initializer=_worker_init)
            _POOL_N = n
        except Exception:  # noqa: BLE001 — no process rights → caller falls back
            _POOL, _POOL_N = None, 0
        return _POOL


def shutdown(*, threads: bool = False) -> None:
    """Tear down the process pool (tests / app exit). Safe to call repeatedly.
    ``threads=True`` also drops the shared thread pool."""
    global _POOL, _POOL_N, _TPOOL, _TPOOL_N
    with _POOL_LOCK:
        if _POOL is not None:
            try:
                _POOL.shutdown(wait=True)
            except Exception:  # noqa: BLE001
                pass
        _POOL, _POOL_N = None, 0
    if threads:
        with _TPOOL_LOCK:
            tp, _TPOOL, _TPOOL_N = _TPOOL, None, 0
        if tp is not None:
            try:
                tp.shutdown(wait=True)
            except Exception:  # noqa: BLE001
                pass


# ── worker-thread stack headroom ───────────────────────────────────────────────

#: Stack size for pool threads. A read through a deep provider chain recurses a few
#: frames per level, and :func:`nodegraph.streaming.recursion_headroom` raises
#: ``sys.setrecursionlimit`` to allow it. That limit is interpreter-wide, but the actual
#: C stack is PER THREAD, and a pool thread gets the platform default (1 MiB on Windows)
#: rather than the main thread's 8 MiB. Raising the Python limit without raising the real
#: stack turns a clean ``RecursionError`` into a hard interpreter crash, so pool threads
#: are given room to match.
_THREAD_STACK_BYTES = 64 << 20


@contextmanager
def _big_stack_threads():
    """Create pool threads with :data:`_THREAD_STACK_BYTES` of stack, then restore the
    process default. ``threading.stack_size`` applies to threads created *after* the
    call, which is exactly the window this guards."""
    try:
        old = threading.stack_size()
    except (ValueError, RuntimeError):           # platform without the knob
        yield
        return
    try:
        threading.stack_size(_THREAD_STACK_BYTES)
    except (ValueError, RuntimeError):           # size rejected — run on the default
        yield
        return
    try:
        yield
    finally:
        try:
            threading.stack_size(old)
        except (ValueError, RuntimeError):
            pass


# ── the shared thread pool ─────────────────────────────────────────────────────

_TPOOL: Any = None
_TPOOL_N = 0
_TPOOL_LOCK = threading.Lock()


def thread_pool(workers: int):
    """The process-wide worker pool, created once and reused.

    Building a fresh :class:`ThreadPoolExecutor` per map is what made the fan-out a
    *pessimisation* on cheap units: spawning ``N`` threads — each reserving
    :data:`_THREAD_STACK_BYTES` — costs more than a handful of millisecond-scale tile
    kernels. Measured on a 3-deep tileable chain reading one 2048² plane (16 tiles of a few
    ms each): **0.74×** with a per-call pool, versus a speedup once the pool is shared. The
    expensive-kernel case (a 4096² median) was 7× either way, because there the threads
    disappeared into the work.

    Sharing one pool is only safe because :func:`map_units` refuses to fan out inside an
    active region (:func:`in_parallel_region`). Without that, a nested map would submit to
    the pool its own parent is occupying and deadlock on starvation — the classic
    thread-pool re-entrancy trap. The guard is load-bearing, not a nicety."""
    global _TPOOL, _TPOOL_N
    n = max(1, int(workers))
    with _TPOOL_LOCK:
        if _TPOOL is not None and _TPOOL_N >= n:
            return _TPOOL
        old = _TPOOL
        from concurrent.futures import ThreadPoolExecutor
        with _big_stack_threads():                # paid once per pool, not per map
            _TPOOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ng-unit")
        _TPOOL_N = n
    if old is not None:
        old.shutdown(wait=False)                  # outside the lock; its tasks are done
    return _TPOOL


# ── the maps ───────────────────────────────────────────────────────────────────

def map_units(fn: Callable[[T], R], items: Sequence[T], *,
              workers: Optional[int] = None,
              min_items: int = 2) -> List[R]:
    """``[fn(x) for x in items]``, evaluated on a thread pool, **in order**.

    Order-preserving is the whole contract: callers fold the results with a running
    counter (region-id offsets, Label table rows), so a reordered result set is a
    different — silently wrong — output, not just a differently-timed one. Use it as
    *parallel pure compute → serial deterministic fold*:

    .. code-block:: python

        labs = map_units(segment_one, units)        # pure, any order internally
        for unit, lab in zip(units, labs):         # fold IN ORDER, single-threaded
            raster[...] = take(lab, *unit)

    ``fn`` may be any callable including a closure (nothing is pickled). Runs serial
    when parallelism is off, when there are fewer than ``min_items`` items, or when
    already inside a parallel region — so nesting degrades instead of multiplying.

    Exceptions propagate from the first failing item (by submission order), matching
    what the serial loop would have raised."""
    n = len(items)
    if n == 0:
        return []
    if n < max(1, min_items) or not enabled() or in_parallel_region():
        return [fn(x) for x in items]
    w = min(int(workers or cpu_budget()), n)
    if w <= 1:
        return [fn(x) for x in items]
    return list(thread_pool(w).map(_mark_region(fn), items))


def map_units_proc(fn: Callable[[T], R], items: Sequence[T], *,
                   workers: Optional[int] = None,
                   min_items: int = 2,
                   fallback: bool = True) -> List[R]:
    """As :func:`map_units` but on the shared **process** pool, still **in order**.

    For work heavy enough that spawn + pickling disappear into it — per-frame CNN
    inference (StarDist/CellSAM), Richardson–Lucy, bilateral/NLM denoising. The two
    hard requirements the thread map does not have:

    * ``fn`` must be a **module-level** function (pickled by qualified name), not a
      closure or a lambda.
    * every item and every return value must be picklable. A model object is NOT — pass
      the model *identity* (name/path) and let the worker's own module-level singleton
      load it once per process. That is why this is the right tool for StarDist: its
      kernel already pins TensorFlow to one inter/intra-op thread (removing that cap
      measured ~10× slower), so one-thread TF × N processes is exactly how to use a
      12-core machine on a per-frame detector.

    Falls back to :func:`map_units` (threads) when a pool cannot be created or the
    payload turns out not to be picklable, unless ``fallback=False``."""
    n = len(items)
    if n == 0:
        return []
    if n < max(1, min_items) or not enabled() or in_parallel_region():
        return [fn(x) for x in items]
    if mode() == "thread":
        return map_units(fn, items, workers=workers, min_items=min_items)
    pool = process_pool(min(int(workers or proc_budget()), n))
    if pool is None:
        return map_units(fn, items, workers=workers, min_items=min_items) \
            if fallback else [fn(x) for x in items]
    # No _mark_region wrapper here: the wrapper is a closure and would itself be
    # unpicklable. The child marks ITSELF via `_worker_init` instead.
    try:
        return list(pool.map(fn, items))
    except Exception:                      # noqa: BLE001
        # An unpicklable payload / a dead pool must not lose the run. Retry on threads,
        # where nothing is pickled at all; a genuine compute error re-raises there too,
        # so the user still sees the real failure.
        if not fallback:
            raise
        shutdown()
        return map_units(fn, items, workers=workers, min_items=min_items)


def _imap(fn: Callable[[T], R], items: Sequence[T], *,
          workers: Optional[int], proc: bool):
    """:func:`map_units` / :func:`map_units_proc` as a **generator**: results still strictly
    in submission order, but yielded *as they land* instead of after the last one.

    This is what :func:`fold_units` needs and the list-returning maps cannot give it. An
    ``Executor.map`` iterator yields result *i* the moment future *i* is done, so folding
    off it converts a batch into ``batch`` separate folds spread over real time. Wrapping it
    in ``list()`` collapses all of them to the instant the SLOWEST unit of the batch
    finishes — which is invisible to the output but very visible on a progress bar, where a
    per-plane node appears frozen and then jumps a whole batch at once.

    Deliberately NOT reusing the public maps: they promise a ``List`` and are gated on that,
    and the process path's fallback-on-failure has to be resumable here (see
    :func:`fold_units`), which a self-contained list call cannot express.

    Raises whatever ``fn`` raises, from the first failing item in submission order — but
    note that with streaming, earlier results have ALREADY been yielded when it does."""
    n = len(items)
    if n == 0:
        return
    if n < 2 or not enabled() or in_parallel_region():
        for x in items:                     # nesting degrades rather than multiplying
            yield fn(x)
        return
    if proc and mode() != "thread":
        pool = process_pool(min(int(workers or proc_budget()), n))
        if pool is not None:
            yield from pool.map(fn, items)
            return
    w = min(int(workers or cpu_budget()), n)
    if w <= 1:
        for x in items:
            yield fn(x)
        return
    yield from thread_pool(w).map(_mark_region(fn), items)


def fold_units(compute: Callable[[T], R], items: Sequence[T],
               fold: Callable[[int, T, R], None], *,
               prepare: Optional[Callable[[T], Any]] = None,
               proc: bool = False, workers: Optional[int] = None,
               batch: Optional[int] = None) -> None:
    """The canonical eager-node shape: **parallel pure compute → serial ordered fold**.

    ``compute(item)`` runs on the pool; ``fold(i, item, result)`` then runs on the calling
    thread, **strictly in submission order**. That split is what makes parallelising the
    eager catalog nodes safe: several of them thread a running counter through the fold
    (``analysis.segment`` and ``analysis.label`` shift each unit's region ids by a global
    offset, and append Label rows in unit order), so the fold must stay serial and ordered
    while the expensive part does not.

    Work is submitted in **batches** of ``batch`` (default: one per worker) rather than
    all at once, which bounds peak memory to the output buffer plus ``batch`` in-flight
    results. Submitting a whole series would hold a second copy of every unit's raster —
    exactly the eager full-raster hazard the fold exists to avoid. Call sites with large
    units (whole volumes) should pass a smaller ``batch``.

    Within a batch each result is folded **as it lands** (:func:`_imap`), not after the whole
    batch completes. Order and output are identical either way; what changes is *when* the
    folds happen, and since the catalog's eager nodes tick their progress bar from the fold,
    that is the difference between a bar that advances per unit and one that stands still for
    a batch and then jumps.

    ``proc=True`` routes the compute to the process pool, which additionally requires
    ``compute`` to be a module-level function and every item/result to be picklable.

    ``prepare(item) -> payload`` builds each task's argument on the **calling** thread,
    one batch ahead of the workers. The process path needs it: a worker cannot reach the
    engine's providers (a lazy provider holds closures; a b2nd provider holds blosc2
    handles — neither pickles), so the pixels have to be read parent-side and shipped.
    Doing that per batch rather than up front is what keeps the whole series out of RAM."""
    n = len(items)
    if n == 0:
        return
    w = max(1, int(workers or (proc_budget() if proc else cpu_budget())))
    b = max(1, int(batch or w))
    for start in range(0, n, b):
        chunk = list(items[start:start + b])
        payloads = [prepare(x) for x in chunk] if prepare is not None else chunk
        # Fold each result AS IT LANDS (`_imap`, not `map_units`): a fold per unit rather
        # than a burst of `batch` folds when the batch's slowest unit finishes. Nodes tick
        # their progress bar from the fold, so the difference is a bar that moves per plane
        # instead of standing still and then jumping a whole batch.
        done = 0
        use_proc = proc
        while done < len(chunk):
            try:
                for res in _imap(compute, payloads[done:], workers=w, proc=use_proc):
                    fold(start + done, chunk[done], res)
                    done += 1
            except Exception:                      # noqa: BLE001
                # Same contract map_units_proc has: an unpicklable payload or a dead pool
                # must not lose the run, so retry on threads (where nothing is pickled) —
                # and a genuine compute error re-raises there, so the user still sees the
                # real failure. RESUMING at `done` is what makes this safe under streaming:
                # the units already folded must not be folded twice, and `compute` is
                # required pure, so recomputing only the remainder is equivalent.
                if not use_proc:
                    raise
                use_proc = False                   # one fallback, never a retry loop
                shutdown()
                continue
            break


def describe() -> str:
    """One-line summary of the resolved policy — for a startup log / bug report."""
    import numpy as np
    return (f"parallel={mode()} workers={cpu_budget()} proc={proc_budget()} "
            f"ram={total_ram_bytes()/_GiB:.0f}GiB "
            f"tiles={tile_cache_bytes()/_GiB:.1f}GiB "
            f"memo={memo_bytes()/_GiB:.1f}GiB "
            f"planes={plane_cache_bytes()/_GiB:.1f}GiB "
            f"dtype={np.dtype(stream_dtype()).name} "
            # ask the gpu module rather than re-reading the env var, so the two can never
            # disagree about what the default is
            f"gpu={__import__('nodegraph.gpu', fromlist=['mode']).mode()}")


__all__ = [
    "cpu_budget", "proc_budget", "total_ram_bytes", "ram_budget",
    "tile_cache_bytes", "memo_bytes", "plane_cache_bytes", "stream_dtype",
    "store_dir", "mode", "enabled", "inner_threads", "in_parallel_region",
    "process_pool", "shutdown", "map_units", "map_units_proc", "fold_units",
    "describe",
]
