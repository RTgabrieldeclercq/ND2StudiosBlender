"""The resource governor — how many analysis workers this machine can run *right now*
without freezing it (2026-10-09).

:mod:`nodegraph.parallel` sizes its pools once, from the core count and installed RAM. That
is right for cheap per-plane work, and wrong for the expensive analysis nodes (DIC, DVC,
PIV, SerialTrack), whose units are whole positions that each hold gigabytes and run numba
kernels that spin every core they are allowed. Run eight of those at normal priority with
numba on its default thread count and an 80-thread box schedules 640 compute threads, pages
once RAM runs out, and the GUI stops repainting. So this module answers four questions the
pools cannot:

* **What is the machine?** :func:`machine_profile` reads logical AND physical cores (a
  hyper-threaded core gives a numba FP kernel almost nothing), installed and *available*
  RAM, and how busy the whole system is — live, so a second program starting mid-run is
  seen. Dependency-free ``ctypes`` on Windows (the same call :func:`parallel.total_ram_bytes`
  already makes), ``/proc`` on Linux, and :mod:`psutil` when it happens to be installed.
* **How wide?** :func:`plan_lanes` — a pure function of the profile, the unit count and
  one worker's memory: lanes × inner threads never exceed the physical cores minus a
  reserve kept for the GUI and the OS, and lanes × worker memory stays above a free-RAM
  floor.
* **How wide, now?** :class:`Governor` is the live half: it starts narrow, **calibrates**
  worker memory from what the first units actually used (each worker reports its own peak
  working set), widens to the plan, and before every new unit re-reads the machine —
  holding submissions while free RAM is under the floor and dropping a lane while other
  programs saturate the CPU. :func:`nodegraph.parallel.adaptive_fold` consults it.
* **How does a worker behave?** :func:`worker_enter` runs in each worker process before
  its unit: below-normal priority (the OS then always serves the GUI and input first — the
  main freeze guard), numba and BLAS/OpenMP pinned to the planned inner thread count.

Every number has an environment override (``NODEGRAPH_PROC_WORKERS`` still caps lanes;
``NODEGRAPH_RESERVE_CORES``, ``NODEGRAPH_MIN_FREE_RAM``). Qt-free; stdlib only (numba and
threadpoolctl are touched only inside a worker, and only if importable).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

_GiB = 1 << 30
_MiB = 1 << 20

#: The hard ceiling on analysis lanes, whatever the machine. Each lane is a spawned
#: process that re-imports the scientific stack (seconds, ~300 MB) before its first unit,
#: and past ~16 the per-lane inner thread count on even an 80-thread box drops to 2 —
#: the cores are better spent inside one unit's numba kernels than on another process.
MAX_LANES = 16

#: Fraction of the RAM above the floor that in-flight workers may plan to use. The rest
#: absorbs what the estimate cannot see (the parent's own reads, page cache, the GUI).
RAM_SHARE = 0.8

#: Safety factor on a measured worker peak: a later unit can be larger than the first
#: (a denser ROI, a longer track set).
PEAK_MARGIN = 1.25

#: System CPU busy fraction at which a lane is given back, and the level it must fall
#: back below before one is restored — a hysteresis band so the plan does not flap.
CPU_HIGH = 0.97
CPU_LOW = 0.85


def _env_int(key: str) -> Optional[int]:
    v = os.environ.get(key, "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _env_bytes(key: str) -> Optional[int]:
    """``8G`` / ``512M`` / a plain byte count."""
    v = os.environ.get(key, "").strip().upper()
    if not v:
        return None
    mult = 1
    for suffix, m in (("GIB", _GiB), ("GB", _GiB), ("G", _GiB),
                      ("MIB", _MiB), ("MB", _MiB), ("M", _MiB)):
        if v.endswith(suffix):
            mult, v = m, v[: -len(suffix)]
            break
    try:
        return int(float(v) * mult)
    except ValueError:
        return None


# ── the machine ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MachineProfile:
    """One reading of the machine. ``cpu_busy`` is the whole system's busy fraction since
    the previous reading (0..1), ``None`` on the first reading or where it cannot be read."""
    logical: int
    physical: int
    total_ram: int
    avail_ram: int
    commit_avail: int
    cpu_busy: Optional[float] = None


def _win_memory() -> Optional[Tuple[int, int, int]]:
    try:
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
            return int(st.ullTotalPhys), int(st.ullAvailPhys), int(st.ullAvailPageFile)
    except Exception:  # noqa: BLE001 — advisory
        pass
    return None


def _win_physical_cores() -> Optional[int]:
    """Count ``RelationProcessorCore`` records across every processor group — the ``Ex``
    call is the one that sees past 64 logical processors (two groups on an 80-thread box)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        need = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))
        if need.value == 0:
            return None
        buf = ctypes.create_string_buffer(need.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
            return None
        n, off, raw = 0, 0, buf.raw
        while off + 8 <= need.value:
            rel = int.from_bytes(raw[off:off + 4], "little")
            size = int.from_bytes(raw[off + 4:off + 8], "little")
            if size <= 0:
                break
            if rel == 0:                       # RelationProcessorCore
                n += 1
            off += size
        return n or None
    except Exception:  # noqa: BLE001
        return None


def _linux_meminfo() -> Optional[Tuple[int, int, int]]:
    try:
        vals: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    vals[k] = int(parts[0]) * 1024
        total = vals.get("MemTotal", 0)
        avail = vals.get("MemAvailable", vals.get("MemFree", 0))
        commit = max(0, vals.get("CommitLimit", total) - vals.get("Committed_AS", 0))
        return (total, avail, commit) if total else None
    except Exception:  # noqa: BLE001
        return None


def _linux_physical_cores() -> Optional[int]:
    try:
        cores = set()
        phys = core = None
        with open("/proc/cpuinfo", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((phys, core))
                    phys = core = None
        if core is not None:
            cores.add((phys, core))
        return len(cores) or None
    except Exception:  # noqa: BLE001
        return None


_CPU_LOCK = threading.Lock()
_CPU_LAST: Optional[Tuple[float, float]] = None      # (idle, total) at the previous reading
_CPU_CACHE: Tuple[float, Optional[float]] = (0.0, None)   # (monotonic time, busy) last computed

#: A busy fraction over a few microseconds is noise (0 % or 100 % depending on one tick),
#: so readings closer together than this return the previous value.
CPU_SAMPLE_S = 0.5


def _cpu_times() -> Optional[Tuple[float, float]]:
    """``(idle, total)`` cumulative CPU time for the whole system."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return None

            def ft(f) -> float:
                return float((f.dwHighDateTime << 32) | f.dwLowDateTime)
            # kernel time INCLUDES idle time on Windows
            return ft(idle), ft(kern) + ft(user)
        except Exception:  # noqa: BLE001
            return None
    try:
        with open("/proc/stat", "r", encoding="ascii") as fh:
            parts = [float(x) for x in fh.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
        return idle, sum(parts)
    except Exception:  # noqa: BLE001
        return None


def _cpu_busy() -> Optional[float]:
    """System busy fraction since the previous sample (any caller), or ``None`` on the
    first. Samples at most every :data:`CPU_SAMPLE_S`; in between, the last value."""
    global _CPU_LAST, _CPU_CACHE
    t = time.monotonic()
    with _CPU_LOCK:
        if _CPU_LAST is not None and t - _CPU_CACHE[0] < CPU_SAMPLE_S:
            return _CPU_CACHE[1]
    now = _cpu_times()
    if now is None:
        return None
    with _CPU_LOCK:
        last, _CPU_LAST = _CPU_LAST, now
        busy = None
        if last is not None:
            d_idle, d_tot = now[0] - last[0], now[1] - last[1]
            if d_tot > 0:
                busy = max(0.0, min(1.0, 1.0 - d_idle / d_tot))
        _CPU_CACHE = (t, busy)
    return busy


_PHYS_CACHE: Optional[int] = None


def physical_cores() -> int:
    """Physical cores (cached — it cannot change under a running process). Falls back to
    the logical count when the platform will not say, which errs toward the old behaviour
    rather than under-using a machine we failed to measure."""
    global _PHYS_CACHE
    if _PHYS_CACHE is None:
        n = None
        try:
            import psutil  # type: ignore
            n = psutil.cpu_count(logical=False)
        except Exception:  # noqa: BLE001
            n = None
        if not n:
            n = _win_physical_cores() if sys.platform == "win32" else _linux_physical_cores()
        _PHYS_CACHE = int(n or (os.cpu_count() or 2))
    return _PHYS_CACHE


def machine_profile() -> MachineProfile:
    """A live reading. Cheap (a few syscalls) — the governor calls it before every unit."""
    logical = int(os.cpu_count() or 2)
    mem = None
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        mem = (int(vm.total), int(vm.available), int(vm.available + sw.free))
    except Exception:  # noqa: BLE001
        mem = _win_memory() if sys.platform == "win32" else _linux_meminfo()
    if mem is None:
        from nodegraph.parallel import total_ram_bytes
        tot = total_ram_bytes()
        mem = (tot, tot // 2, tot // 2)
    return MachineProfile(logical=logical, physical=max(1, min(physical_cores(), logical)),
                          total_ram=mem[0], avail_ram=mem[1], commit_avail=mem[2],
                          cpu_busy=_cpu_busy())


# ── the plan (pure) ────────────────────────────────────────────────────────────

def reserve_cores(physical: int) -> int:
    """Cores held back for the GUI thread, the pull thread, the parent's own reads and the
    OS: ``max(2, physical // 10)``, never so many that none is left for work.
    ``NODEGRAPH_RESERVE_CORES`` overrides."""
    n = _env_int("NODEGRAPH_RESERVE_CORES")
    if n is None:
        n = max(2, physical // 10)
    return max(0, min(int(n), physical - 1))


def min_free_ram(total_ram: int) -> int:
    """The floor free RAM is never planned below: ``max(4 GiB, 10 %)`` — below it Windows
    starts trimming working sets and the desktop stutters before anything runs out.
    ``NODEGRAPH_MIN_FREE_RAM`` (``6G``, ``800M``) overrides."""
    v = _env_bytes("NODEGRAPH_MIN_FREE_RAM")
    if v is not None:
        return max(0, v)
    return max(4 * _GiB, total_ram // 10)


@dataclass(frozen=True)
class LanePlan:
    """How wide to run: ``lanes`` concurrent units, each allowed ``inner`` threads.
    ``cpu_lanes`` / ``ram_lanes`` are the two bounds it came from."""
    lanes: int
    inner: int
    cpu_lanes: int
    ram_lanes: int
    usable_cores: int
    worker_bytes: int

    def note(self) -> str:
        return (f"{self.lanes} lane{'s' if self.lanes != 1 else ''} × {self.inner} "
                f"thread{'s' if self.inner != 1 else ''}")


def plan_lanes(n_units: int, worker_bytes: int, profile: MachineProfile, *,
               min_inner: int = 1, max_lanes: int = MAX_LANES) -> LanePlan:
    """The widest safe plan for ``n_units`` independent units of ``worker_bytes`` each.

    * CPU: ``usable = physical − reserve``; ``lanes ≤ usable // min_inner`` (every lane
      keeps at least ``min_inner`` threads — a numba kernel on one thread is the slowest
      way to run it) and ``inner = usable // lanes`` (lanes × inner ≤ usable — the product
      is what oversubscribes, not either factor).
    * RAM: ``lanes ≤ (available − floor) × RAM_SHARE // worker_bytes``.
    * ``NODEGRAPH_PROC_WORKERS`` caps lanes when set; :data:`MAX_LANES` always does.

    Floor 1 lane — progress must always be possible; a unit that does not fit at all runs
    alone, which is the best that can be done for it."""
    n_units = max(0, int(n_units))
    usable = max(1, profile.physical - reserve_cores(profile.physical))
    cap = _env_int("NODEGRAPH_PROC_WORKERS")
    hard = max(1, min(int(max_lanes), cap if cap is not None else MAX_LANES))
    cpu_lanes = max(1, min(usable // max(1, int(min_inner)), hard))
    room = max(0, profile.avail_ram - min_free_ram(profile.total_ram))
    wb = max(1, int(worker_bytes))
    ram_lanes = max(1, int(room * RAM_SHARE) // wb)
    lanes = max(1, min(cpu_lanes, ram_lanes, max(1, n_units)))
    inner = max(1, usable // lanes)
    return LanePlan(lanes=lanes, inner=inner, cpu_lanes=cpu_lanes, ram_lanes=ram_lanes,
                    usable_cores=usable, worker_bytes=wb)


# ── the live half ──────────────────────────────────────────────────────────────

@dataclass
class UnitStats:
    """What a worker reports back with its result (:func:`timed_call`)."""
    seconds: float
    peak_bytes: int
    pid: int


class Governor:
    """Live admission control for one run of ``n_units`` units.

    :meth:`allowed` is asked before each submission and answers how many units may be in
    flight now: the warm-up width until :attr:`calibrated`, then the plan, minus a lane per
    CPU-saturation episode, and never more than free RAM allows — but always at least 1
    when nothing is in flight (progress beats caution: one unit that does not fit is still
    the user's job). :meth:`record` feeds each finished unit's time and peak memory back
    in; the first one re-plans on the MEASURED peak. ``profile_fn`` is injectable so the
    policy is testable on a fake machine."""

    def __init__(self, n_units: int, worker_bytes: int, *, min_inner: int = 1,
                 max_lanes: int = MAX_LANES, warmup: int = 2,
                 profile_fn: Optional[Callable[[], MachineProfile]] = None) -> None:
        self.n_units = max(0, int(n_units))
        self.min_inner = max(1, int(min_inner))
        self.max_lanes = max(1, int(max_lanes))
        self.warmup = max(1, int(warmup))
        self.profile_fn = profile_fn or machine_profile
        self.calibrated = False
        self.cpu_drop = 0
        self.held = ""
        self.fallback = ""                   # set by adaptive_fold when the lanes fail
        self.seconds: List[float] = []
        self.peak_bytes = 0
        self._starts: List[float] = []        # monotonic start times of units in flight
        self._clock: Callable[[], float] = time.monotonic
        self._last: Optional[MachineProfile] = self.profile_fn()
        self.plan = plan_lanes(self.n_units, worker_bytes, self._last,
                               min_inner=self.min_inner, max_lanes=self.max_lanes)

    def pool_width(self) -> int:
        """The process-pool width to ask for: the CPU bound, which calibration cannot raise."""
        return max(1, min(self.plan.cpu_lanes, max(1, self.n_units)))

    def inner_threads(self) -> int:
        """Threads a worker may use for the unit it is about to start — sized from the
        TARGET width, not the warm-up width, so the two warm-up units do not each take half
        the machine and then share it with six more."""
        target = max(1, min(self.plan.lanes, max(1, self.n_units)) - self.cpu_drop)
        return max(1, self.plan.usable_cores // target)

    def allowed(self, in_flight: int) -> int:
        """How many units may be in flight after this check (``≥ in_flight`` never forces
        a running unit off; ``in_flight`` means *hold*)."""
        prof = self.profile_fn()
        self._last = prof
        target = self.plan.lanes if self.calibrated else min(self.plan.lanes, self.warmup)
        busy = prof.cpu_busy
        if busy is not None:
            if busy >= CPU_HIGH and in_flight >= 1 and self.cpu_drop < target - 1:
                ours = (in_flight * self.inner_threads()) / max(1, prof.logical)
                if ours < 0.75:                      # someone else is using the rest
                    self.cpu_drop += 1
            elif busy <= CPU_LOW and self.cpu_drop:
                self.cpu_drop -= 1
        target = max(1, target - self.cpu_drop)
        # Free RAM right now does not yet show what a JUST-started worker will grow into
        # (spawn import, reading its unit, the kernel's own arrays), so a unit started
        # recently is counted as still to come — otherwise one burst of submissions reads
        # the same pre-growth figure sixteen times and admits them all.
        floor = min_free_ram(prof.total_ram)
        wb = self.plan.worker_bytes
        if in_flight >= 1 and prof.avail_ram - floor - self._young() * wb < wb:
            self.held = (f"holding at {in_flight} lane{'s' if in_flight != 1 else ''}: "
                         f"{prof.avail_ram / _GiB:.1f} GB free")
            return in_flight
        self.held = ""
        return max(1, target)

    def started(self) -> None:
        """A unit was just submitted (see :meth:`allowed`'s growth allowance)."""
        self._starts.append(self._clock())

    def _young(self) -> int:
        """Units started within the growth horizon — half a measured unit, at least 5 s."""
        horizon = max(5.0, 0.5 * (self.mean_seconds() or 0.0))
        now = self._clock()
        return sum(1 for t in self._starts if now - t < horizon)

    def record(self, stats: Optional[UnitStats]) -> None:
        """A unit finished. The first report re-plans on its measured peak."""
        if self._starts:
            self._starts.pop(0)
        if stats is None:
            return
        self.seconds.append(float(stats.seconds))
        self.peak_bytes = max(self.peak_bytes, int(stats.peak_bytes))
        if not self.calibrated and self.peak_bytes > 0:
            measured = int(self.peak_bytes * PEAK_MARGIN)
            self.plan = plan_lanes(self.n_units, measured, self.profile_fn(),
                                   min_inner=self.min_inner, max_lanes=self.max_lanes)
            self.calibrated = True

    def mean_seconds(self) -> Optional[float]:
        return (sum(self.seconds) / len(self.seconds)) if self.seconds else None

    def note(self) -> str:
        if self.fallback:
            return self.fallback
        if self.held:
            return self.held
        prof = self._last
        free = f" · {prof.avail_ram / _GiB:.0f} GB free" if prof is not None else ""
        drop = f" (−{self.cpu_drop}: CPU busy)" if self.cpu_drop else ""
        return f"{self.plan.note()}{drop}{free}"


def estimate_seconds(unit_seconds: float, n_units: int, plan: LanePlan,
                     *, measured_threads: Optional[int] = None) -> float:
    """Wall time for ``n_units`` at ``unit_seconds`` each on ``plan`` — the full-run
    estimate a quick analysis reports. A unit measured with ``measured_threads`` threads
    (default: every usable core, the in-thread quick run) gets ``plan.inner`` of them at
    full width, so its time is scaled by the thread ratio — damped, because numba kernels
    do not scale linearly (half the threads is not twice the time)."""
    if n_units <= 0:
        return 0.0
    lanes = max(1, plan.lanes)
    had = max(1, int(measured_threads or plan.usable_cores))
    slow = max(1.0, had / max(1, plan.inner)) ** 0.6
    waves = -(-int(n_units) // lanes)
    return float(unit_seconds) * slow * waves


# ── the worker side ────────────────────────────────────────────────────────────

_ENTERED: Dict[str, int] = {}


def _below_normal_priority() -> None:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            below_normal_priority_class = 0x4000
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # HANDLE-typed, or ctypes passes the pseudo-handle (-1) as a 32-bit int and the
            # call fails silently on 64-bit Windows — measured: workers stayed at Normal
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(k32.GetCurrentProcess(), below_normal_priority_class)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001 — a priority we cannot set is not a reason to fail
        pass


def worker_enter(inner: int, *, numba: bool = True) -> None:
    """Called in a worker process before each unit: drop to below-normal priority (once),
    then pin numba and BLAS/OpenMP to ``inner`` threads (re-applied when the plan changes).
    A no-op in the parent process, where the GUI must keep its priority and the in-thread
    serial path keeps the libraries' own defaults."""
    import multiprocessing as mp
    if mp.current_process().name == "MainProcess":
        return
    if not _ENTERED.get("priority"):
        _below_normal_priority()
        _ENTERED["priority"] = 1
    inner = max(1, int(inner))
    if _ENTERED.get("inner") == inner and _ENTERED.get("numba", 0) >= int(numba):
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(limits=inner)
    except Exception:  # noqa: BLE001
        pass
    if numba:
        try:
            import numba as _nb  # type: ignore
            _nb.set_num_threads(max(1, min(inner, int(_nb.config.NUMBA_NUM_THREADS))))
            _ENTERED["numba"] = 1
        except Exception:  # noqa: BLE001
            pass
    _ENTERED["inner"] = inner


def peak_rss_bytes() -> int:
    """This process's peak working set (Windows) / max RSS (POSIX), or 0."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            k32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                                   wintypes.DWORD]
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return int(pmc.PeakWorkingSetSize)
            return 0
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except Exception:  # noqa: BLE001
        return 0


class _FileLock:
    """A machine-wide lock on a file — ``msvcrt.locking`` on Windows, ``fcntl.flock``
    elsewhere — held for a ``with`` block. Waits (never fails) for the holder to finish."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "a+b")
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:          # LK_LOCK gives up after ~10 s; keep waiting
                    continue
        else:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        try:
            if sys.platform == "win32":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


def serialized_import(name: str):
    """Import module ``name`` holding a machine-wide lock, so worker processes starting at
    the same moment import it ONE AT A TIME.

    For a package whose import-time probe races between processes: ``al_dvc``'s
    ``_numba_compat`` writes and deletes a probe file in its own ``__pycache__`` to decide
    whether numba may cache compiled kernels on disk. Eight workers importing together on
    Windows collide on that file (removing a file another process holds open fails), the
    losers silently turn the cache OFF and recompile every kernel — measured as a DVC run
    on 8 lanes taking 8× the serial time. Already-imported modules return at once."""
    import importlib
    if name in sys.modules:
        return sys.modules[name]
    import tempfile
    lock = os.path.join(tempfile.gettempdir(), f"nodegraph-import-{name}.lock")
    try:
        with _FileLock(lock):
            return importlib.import_module(name)
    except OSError:                      # no lock possible: import unguarded
        return importlib.import_module(name)


#: The sentinel tag :func:`timed_call` wraps a compute exception in.
ERROR_TAG = "__unit_error__"


def timed_call(fn, payload, setup: Optional[Dict[str, object]] = None):
    """The module-level task wrapper :func:`nodegraph.parallel.adaptive_fold` submits:
    ``setup`` → :func:`worker_enter`, then ``fn(payload)``, returned as ``(result, stats)``
    with the :class:`UnitStats` the governor calibrates on. A compute error comes back as
    ``((ERROR_TAG, exc, traceback_text), None)`` rather than raised, so the parent can tell
    a genuine compute failure (re-raise it) from a broken pool (fall back)."""
    setup = setup or {}
    for name in setup.get("preimport") or ():
        try:
            serialized_import(str(name))
        except ImportError:
            pass
    worker_enter(int(setup.get("inner", 1) or 1), numba=bool(setup.get("numba", True)))
    t0 = time.perf_counter()
    try:
        res = fn(payload)
    except Exception as exc:  # noqa: BLE001
        import traceback
        return (ERROR_TAG, exc, traceback.format_exc()), None
    return res, UnitStats(seconds=time.perf_counter() - t0, peak_bytes=peak_rss_bytes(),
                          pid=os.getpid())


def describe() -> str:
    p = machine_profile()
    return (f"cores={p.physical}p/{p.logical}l reserve={reserve_cores(p.physical)} "
            f"ram={p.avail_ram / _GiB:.0f}/{p.total_ram / _GiB:.0f}GiB free "
            f"floor={min_free_ram(p.total_ram) / _GiB:.0f}GiB")


__all__ = [
    "MachineProfile", "machine_profile", "physical_cores", "reserve_cores", "min_free_ram",
    "LanePlan", "plan_lanes", "UnitStats", "Governor", "estimate_seconds",
    "worker_enter", "peak_rss_bytes", "timed_call", "describe", "ERROR_TAG",
    "serialized_import",
    "MAX_LANES", "RAM_SHARE", "PEAK_MARGIN",
]
