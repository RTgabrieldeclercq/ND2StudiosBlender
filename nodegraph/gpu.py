"""Optional CUDA dispatch for the ndimage kernels (nodegraph v2, V2.14).

An RTX-class card sits idle in this pipeline: nothing in the engine has ever touched it.
``cupyx.scipy.ndimage`` is a near-drop-in for the ``scipy.ndimage`` calls the enhancement
filters make, so this module routes them to the GPU **when that is provably safe** and to
scipy otherwise.

Design rules, because this is a scientific measurement tool and a silently-wrong filter is
worse than a slow one:

* **Dormant unless usable.** No CuPy, no CUDA device, or ``NODEGRAPH_GPU=off`` → every
  call is plain scipy, with no import cost paid.
* **Each kernel earns its place** (:func:`_verified`). The first time a given function is
  requested, it runs on a small deterministic fixture on BOTH backends and the results are
  compared. Disagreement beyond float32 tolerance permanently disables that function for
  the session. This is what makes the dispatch trustworthy without a human having
  hand-checked every op on every driver/CuPy combination: an op that does not reproduce
  scipy does not get used. Failures are recorded in :data:`REJECTED`.
* **One device, one queue.** GPU work does not compose with the CPU fan-out — twelve
  workers pushing kernels at one card serializes on the card anyway while multiplying
  device memory. :func:`active` is therefore consulted by
  :func:`nodegraph.parallel.map_units`' callers to keep the unit loop narrow.
* **Host arrays in, host arrays out.** A kernel returns numpy, so nothing downstream (the
  TileCache freeze, the memo fingerprint, `_canon`) ever sees a device array.

``NODEGRAPH_GPU``: ``off`` (**the default** — see :func:`mode` for the measurements behind
that choice) / ``auto`` (use the card where it is present and verified) / ``on`` (fail
loudly if unavailable, for a deliberate GPU run).

Qt-free; numpy + stdlib, CuPy strictly optional.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

#: Functions that failed their CPU-equivalence check this session → name: reason.
REJECTED: Dict[str, str] = {}

#: Functions that passed.
VERIFIED: Set[str] = set()

_LOCK = threading.Lock()
#: Serializes device work — the card is one resource (see :func:`ndimage`).
_DEVICE = threading.Lock()
_CP: Any = None                  # the cupy module, or False once known unavailable
_CPX: Any = None                 # cupyx.scipy.ndimage


def mode() -> str:
    """``off`` (the DEFAULT) / ``auto`` / ``on`` — from ``NODEGRAPH_GPU``.

    **Off by default, on measured evidence rather than caution.** The kernels themselves
    are dramatic — verified bit-exact against scipy on an RTX 3090, a 4096² median ran 79×
    and a 64×512² gaussian 94× — but a whole-node pull is a different question, because the
    CPU alternative is not one core, it is twelve. Through the engine:

    * 3D volume filters (gaussian, median): **8–10×** — a genuine, large win.
    * anything tiled: ~1× — a 512² tile is under the dispatch threshold, so it never
      reaches the card (and forcing whole planes to fix that measured **0.14×**, because
      an op the gate rejects then loses the L3-sized tiling as well).
    * 3D EDT: cupyx's 3D implementation is only ~2× a single core, which twelve cores beat
      outright.

    So the card helps a specific, identifiable shape of work — big 3D units — and is a wash
    or a loss elsewhere. A default that silently made some graphs slower would be worse than
    one that asks; `NODEGRAPH_GPU=auto` turns it on for the graphs that want it."""
    m = (os.environ.get("NODEGRAPH_GPU", "") or "off").strip().lower()
    return m if m in ("auto", "off", "on") else "off"


# ── machine detection (no CuPy, no toolkit, no nvidia-smi) ─────────────────────

@dataclass(frozen=True)
class Device:
    name: str
    sm: int                       # compute capability as major*10+minor (sm_86 → 86)
    total_bytes: int


@dataclass(frozen=True)
class Platform:
    """What accelerator this machine actually has, probed from the DRIVER.

    Deliberately independent of CuPy: the whole point is to answer "which CuPy should I
    install?" on a machine that has none yet. It is also independent of ``nvidia-smi``,
    which is frequently absent from ``PATH`` on Windows even when the driver is fine — the
    driver library itself is loaded with :mod:`ctypes` and asked directly."""

    kind: str                     # "cuda" | "rocm" | "none"
    driver_cuda: Tuple[int, int] = ()      # (major, minor) the DRIVER supports
    devices: Tuple[Device, ...] = ()
    note: str = ""                # why, when kind == "none"

    @property
    def min_sm(self) -> int:
        return min((d.sm for d in self.devices), default=0)


#: ``(wheel, minimum driver CUDA, minimum compute capability)``, newest first.
#:
#: Both bounds matter and for different reasons. A wheel's CUDA runtime cannot exceed what
#: the installed driver supports (a CUDA 13 build will not load on a CUDA 12 driver). And
#: each CUDA major drops old architectures — 13.x requires Turing (sm_75+), 12.x dropped
#: Kepler — so on an older card with a brand-new driver the NEWEST wheel is the wrong
#: answer even though the driver would load it. Picking by both is what makes this adapt to
#: the machine rather than to the calendar.
_WHEELS: Tuple[Tuple[str, Tuple[int, int], int], ...] = (
    ("cupy-cuda13x", (13, 0), 75),
    ("cupy-cuda12x", (12, 0), 50),
    ("cupy-cuda11x", (11, 2), 35),
)

#: The ``[ctk]`` extra is NOT optional for this project, so it is baked into every
#: recommendation. CuPy JIT-compiles the ``cupyx.scipy.ndimage`` kernels, which needs the
#: CUDA headers; without them every single op fails at first use. Measured on a clean
#: install here: all seven kernels were rejected by the equivalence gate with "Failed to
#: find CUDA headers", and the engine silently ran everything on the CPU.
_CTK_EXTRA = "[ctk]"


def _driver_lib():
    for name in (["nvcuda.dll"] if sys.platform == "win32"
                 else ["libcuda.dylib"] if sys.platform == "darwin"
                 else ["libcuda.so.1", "libcuda.so"]):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _rocm_present() -> bool:
    for name in (["amdhip64.dll"] if sys.platform == "win32"
                 else ["libamdhip64.so", "libamdhip64.so.5"]):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


_PLATFORM: Optional[Platform] = None


def detect_platform(*, refresh: bool = False) -> Platform:
    """Probe the accelerator via the driver library. Cached; never raises."""
    global _PLATFORM
    if _PLATFORM is not None and not refresh:
        return _PLATFORM
    p = _detect_platform_uncached()
    _PLATFORM = p
    return p


def _detect_platform_uncached() -> Platform:
    if sys.platform == "darwin":
        return Platform("none", note="macOS: NVIDIA CUDA is not available on this platform")
    lib = _driver_lib()
    if lib is None:
        if _rocm_present():
            return Platform("rocm", note="an AMD ROCm runtime is present but no CUDA driver")
        return Platform("none", note="no CUDA driver library found (no NVIDIA driver "
                                     "installed, or no NVIDIA GPU in this machine)")
    try:
        if lib.cuInit(0) != 0:
            return Platform("none", note="the CUDA driver is present but cuInit() failed "
                                         "(no usable GPU, or a driver/permissions problem)")
        raw = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(raw)) != 0 or raw.value <= 0:
            return Platform("none", note="cuDriverGetVersion() failed")
        driver = (raw.value // 1000, (raw.value % 1000) // 10)
        count = ctypes.c_int(0)
        lib.cuDeviceGetCount(ctypes.byref(count))
        devs = []
        for i in range(max(0, count.value)):
            buf = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(buf, 256, i)
            maj, mnr = ctypes.c_int(0), ctypes.c_int(0)
            lib.cuDeviceGetAttribute(ctypes.byref(maj), 75, i)   # CC major
            lib.cuDeviceGetAttribute(ctypes.byref(mnr), 76, i)   # CC minor
            tot = ctypes.c_size_t(0)
            try:
                lib.cuDeviceTotalMem_v2(ctypes.byref(tot), i)
            except AttributeError:                               # pre-v2 driver ABI
                lib.cuDeviceTotalMem(ctypes.byref(tot), i)
            devs.append(Device(buf.value.decode(errors="replace"),
                               maj.value * 10 + mnr.value, int(tot.value)))
        if not devs:
            return Platform("none", driver_cuda=driver,
                            note="the CUDA driver loaded but reports zero devices")
        return Platform("cuda", driver_cuda=driver, devices=tuple(devs))
    except Exception as exc:      # noqa: BLE001 — detection must never break a caller
        return Platform("none", note=f"probing the CUDA driver raised {type(exc).__name__}: "
                                     f"{exc}")


def recommended_package(platform: Optional[Platform] = None) -> Optional[str]:
    """The pip requirement to install for THIS machine, or ``None`` if none fits.

    e.g. ``"cupy-cuda13x[ctk]"``. Chosen by both the driver's CUDA version and the oldest
    installed GPU's compute capability — see :data:`_WHEELS`."""
    p = platform or detect_platform()
    if p.kind == "rocm":
        return "cupy-rocm-5-0"                # no ctk extra on the ROCm builds
    if p.kind != "cuda":
        return None
    for wheel, min_cuda, min_sm in _WHEELS:
        if p.driver_cuda >= min_cuda and p.min_sm >= min_sm:
            return wheel + _CTK_EXTRA
    return None


def install_hint(platform: Optional[Platform] = None) -> str:
    """A machine-specific, copy-pasteable explanation of how to enable the GPU path —
    or of why this machine cannot. Used in the ``NODEGRAPH_GPU=on`` error so the message
    names the right wheel instead of a guess."""
    p = platform or detect_platform()
    if p.kind == "none":
        return (f"No CUDA GPU usable here — {p.note}. The engine runs entirely on the CPU "
                f"(which is the default); nothing needs installing.")
    if p.kind == "rocm":
        return ("An AMD ROCm runtime was detected. CuPy's ROCm builds are source builds and "
                "are not tested by this project:\n    pip install cupy-rocm-5-0\n"
                "The CPU path remains the supported one.")
    devs = ", ".join(f"{d.name} (sm_{d.sm}, {d.total_bytes / (1 << 30):.0f} GiB)"
                     for d in p.devices)
    pkg = recommended_package(p)
    head = f"Detected: {devs}; driver supports CUDA {p.driver_cuda[0]}.{p.driver_cuda[1]}."
    if pkg is None:
        return (f"{head}\nNo CuPy wheel matches this combination — the driver is older than "
                f"CUDA 11.2, or the GPU predates what current CUDA supports. Update the "
                f"NVIDIA driver, or stay on the CPU path (the default).")
    return (f"{head}\nInstall:\n    python scripts/setup_cupy.py\n"
            f"  (or directly:  pip install \"{pkg}\")\n"
            f"The [ctk] extra is required, not cosmetic — CuPy JIT-compiles these kernels "
            f"and rejects every one of them without the CUDA headers.")


def _load() -> bool:
    """Import CuPy once and confirm a device is actually present. An installed CuPy with
    no working driver is the common broken state, so a device count of zero counts as
    unavailable rather than raising later inside a kernel."""
    global _CP, _CPX
    if _CP is not None:
        return bool(_CP)
    with _LOCK:
        if _CP is not None:
            return bool(_CP)
        if mode() == "off":
            _CP = False
            return False
        try:
            import cupy
            import cupyx.scipy.ndimage as cpx
            if int(cupy.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("no CUDA device")
            _CP, _CPX = cupy, cpx
        except Exception as exc:                      # noqa: BLE001
            if mode() == "on":
                # The hint is derived from THIS machine's driver and GPU, not from a
                # hardcoded wheel name — a fixed suggestion is wrong on most other boxes.
                raise RuntimeError(
                    f"NODEGRAPH_GPU=on but the CUDA path is unavailable: {exc}\n\n"
                    f"{install_hint()}\n\nOr set NODEGRAPH_GPU=off."
                ) from exc
            _CP = False
        return bool(_CP)


def available() -> bool:
    """CuPy + a CUDA device are usable (says nothing about a specific kernel)."""
    return _load()


def active() -> bool:
    """GPU dispatch is on AND at least one kernel has been verified — the flag callers
    use to narrow their CPU fan-out, since one device does not want twelve feeders."""
    return _load() and bool(VERIFIED)


def device_name() -> str:
    if not _load():
        return ""
    try:
        props = _CP.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001
        return "CUDA device"


# ── the equivalence gate ───────────────────────────────────────────────────────

def _fixture(ndim: int) -> np.ndarray:
    """A small deterministic test array (no RNG — the gate must be reproducible)."""
    if ndim == 3:
        z, y, x = np.mgrid[0:6, 0:24, 0:28].astype(np.float64)
        return (np.sin(x / 3.0) * np.cos(y / 4.0) + 0.5 * z
                + ((x + y + z) % 5) * 0.25)
    y, x = np.mgrid[0:32, 0:36].astype(np.float64)
    return np.sin(x / 3.0) * np.cos(y / 4.0) + ((x + y) % 5) * 0.25


def _verified(name: str, kwargs_2d: dict, ndim: int) -> bool:
    """Has ``cupyx.scipy.ndimage.<name>`` reproduced ``scipy.ndimage.<name>`` here?

    Checked once per (name, ndim) and cached. The tolerance is float32-ish because CuPy
    may use different intermediate precision or a separable decomposition in a different
    order; what this is really screening for is a signature mismatch, an unimplemented
    boundary mode, or an op CuPy spells differently — all of which show up as gross
    disagreement, not as 1e-6."""
    key = f"{name}/{ndim}d"
    if key in VERIFIED:
        return True
    if key in REJECTED:
        return False
    with _LOCK:
        if key in VERIFIED:
            return True
        if key in REJECTED:
            return False
        try:
            import scipy.ndimage as sndi
            cpu_fn = getattr(sndi, name)
            gpu_fn = getattr(_CPX, name)
            a = _fixture(ndim)
            want = np.asarray(cpu_fn(a, **kwargs_2d), dtype=np.float64)
            got = _CP.asnumpy(gpu_fn(_CP.asarray(a), **kwargs_2d)).astype(np.float64)
            if want.shape != got.shape:
                raise ValueError(f"shape {got.shape} != scipy {want.shape}")
            if not np.allclose(want, got, rtol=1e-4, atol=1e-4):
                raise ValueError(
                    f"max |Δ| = {float(np.max(np.abs(want - got))):.3g} vs scipy")
            VERIFIED.add(key)
            return True
        except Exception as exc:                      # noqa: BLE001
            REJECTED[key] = str(exc)[:200]
            return False


# ── the dispatch entry point ───────────────────────────────────────────────────

def ndimage(name: str, arr: np.ndarray, **kwargs: Any) -> Optional[np.ndarray]:
    """Run ``scipy.ndimage.<name>(arr, **kwargs)`` on the GPU, or return ``None``.

    ``None`` means "not this time" — no CuPy, no device, the op failed its equivalence
    check, the array is too small to pay for the host↔device round trip, or the transfer
    would not fit device memory. The caller then runs its normal scipy path, so a
    ``None`` is never an error and never needs handling beyond the fallback it already
    has.

    Small arrays go to the CPU deliberately: a 2 MB tile costs more in PCIe latency than
    a gaussian costs to compute, so dispatching everything would make the tiled path
    slower. The threshold is per-array-bytes, not per-op."""
    if arr.size < _MIN_ELEMENTS or not _load():
        return None
    if not _verified(name, _probe_kwargs(name, kwargs), arr.ndim):
        return None
    try:
        # ONE DEVICE, ONE QUEUE — and this lock is what makes that true (V2.14).
        #
        # Node unit loops fan out across cores, so without it N workers issue N concurrent
        # transfers + kernels at a single card. That does not run N times faster; it thrashes.
        # Measured on a 3D EDT over 8 volumes: 520 ms per volume when issued alone, but
        # **7.8 s** per volume with 8 threads pushing at once — a 33× regression versus the
        # plain multi-core CPU path. Serializing here turns the card back into what it is: a
        # fast single resource that the workers queue for.
        with _DEVICE:
            # keep a safety margin on device memory: the op needs input + output + scratch
            free, _total = _CP.cuda.runtime.memGetInfo()
            if arr.nbytes * 4 > free:
                return None
            d = _CP.asarray(arr)
            out = getattr(_CPX, name)(d, **kwargs)
            return _CP.asnumpy(out)
    except Exception as exc:                          # noqa: BLE001
        # A runtime failure (out of memory, an unsupported dtype/mode combination that the
        # fixture did not exercise) retires this op for the session rather than raising:
        # the CPU path is always correct, so degrading is strictly better than failing a
        # user's run — but it must not be retried per tile.
        REJECTED[f"{name}/{arr.ndim}d"] = f"runtime: {str(exc)[:180]}"
        VERIFIED.discard(f"{name}/{arr.ndim}d")
        return None


#: Below this element count the PCIe round trip dominates the kernel. ~4 Mpx: a 2048²
#: plane dispatches, a 512² tile does not.
_MIN_ELEMENTS = 4 << 20


def _probe_kwargs(name: str, kwargs: dict) -> dict:
    """The kwargs the equivalence fixture is checked with.

    The *real* call's kwargs cannot be reused verbatim — a footprint/size tuned to a
    2048² plane is meaningless on a 32² fixture, and a `size` larger than the fixture
    would make both backends agree trivially on an all-edge result. So the probe uses a
    small, shape-appropriate stand-in for the size-like arguments and passes everything
    else (mode, cval) through, since those are exactly the semantics worth screening."""
    out = {k: v for k, v in kwargs.items() if k in ("mode", "cval")}
    if "sigma" in kwargs:
        out["sigma"] = 1.5
    elif "size" in kwargs:
        out["size"] = 3
    elif "footprint" in kwargs:
        fp = np.asarray(kwargs["footprint"])
        out["footprint"] = np.ones((3,) * fp.ndim, dtype=bool)
    return out


def describe() -> str:
    if mode() == "off":
        return "gpu=off"
    if not _load():
        p = detect_platform()
        if p.kind == "cuda":                  # hardware is fine — CuPy is what's missing
            pkg = recommended_package(p) or "(no matching wheel)"
            return (f"gpu=idle ({p.devices[0].name}, driver CUDA "
                    f"{p.driver_cuda[0]}.{p.driver_cuda[1]}) — CuPy not installed; "
                    f"`python scripts/setup_cupy.py` installs {pkg}")
        return f"gpu=unavailable ({p.note})"
    bits = [f"gpu={device_name()}"]
    if VERIFIED:
        bits.append("verified=" + ",".join(sorted(VERIFIED)))
    if REJECTED:
        bits.append("rejected=" + ",".join(sorted(REJECTED)))
    return " ".join(bits)


__all__ = ["available", "active", "mode", "device_name", "ndimage", "describe",
           "Device", "Platform", "detect_platform", "recommended_package",
           "install_hint", "VERIFIED", "REJECTED"]
