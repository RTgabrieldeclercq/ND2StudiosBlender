"""One-command CuPy install, matched to THIS machine — run once per machine, optional.

    python scripts/setup_cupy.py                 # detect, install, verify
    python scripts/setup_cupy.py --check         # report only, install nothing
    python scripts/setup_cupy.py --dry-run       # print the exact pip command
    python scripts/setup_cupy.py --force         # reinstall / switch variant
    python scripts/setup_cupy.py --bench         # also time each kernel vs scipy

WHY THIS EXISTS
    CuPy does not ship one wheel. There is a separate build per CUDA major
    (``cupy-cuda11x`` / ``cupy-cuda12x`` / ``cupy-cuda13x``, plus ROCm), and installing
    the wrong one fails at import or at the first kernel — after a ~200 MB download. The
    right one depends on two properties of the machine that no requirements file can know:

      * the **driver's** CUDA version — a wheel's runtime cannot exceed it, and
      * the **GPU's compute capability** — each CUDA major drops old architectures, so on
        an older card with a current driver the newest wheel is the wrong answer even
        though the driver would happily load it.

    :func:`nodegraph.gpu.detect_platform` probes both straight from the driver library with
    ``ctypes`` — no ``nvidia-smi`` (routinely missing from ``PATH`` on Windows even with a
    healthy driver), no toolkit, and no CuPy, since the point is to run before CuPy exists.

    The ``[ctk]`` extra is always included and that is load-bearing, not tidiness: CuPy
    JIT-compiles the ``cupyx.scipy.ndimage`` kernels this project dispatches, so without the
    CUDA headers **every** op fails its first call. Measured on a headerless install here:
    all seven kernels were rejected and the engine silently ran everything on the CPU — the
    worst kind of outcome, since it looks like it worked.

IS IT WORTH INSTALLING?
    Only for some graphs, and the script says which after it verifies. Measured on an
    RTX 3090 against this 12-core CPU, through whole node pulls: 3D volume filters
    (gaussian, median) **9–31×**; tiled 2D filters ~1× (a 512² tile is below the dispatch
    threshold, so it never reaches the card); ``analysis.edt`` ~1× (deliberately never
    dispatched — cupyx's 3D transform is only ~2× a single core, which 12 cores beat).
    So: install it if you do heavy 3D volumetric filtering. The GPU stays **off by default**
    either way; enable it per run with ``NODEGRAPH_GPU=auto``.

AFTERWARDS
    Nothing else to configure. Each kernel must still pass an equivalence check against
    scipy on first use, so an op that cannot reproduce the CPU result is never used on your
    data (see :mod:`nodegraph.gpu`). Verify any time with:
        NODEGRAPH_GPU=auto python scripts/setup_cupy.py --check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: CuPy's own numpy floor/ceiling. Checked BEFORE installing because pip resolves it
#: silently: on a pinned-numpy scientific environment a CuPy install can move numpy
#: underneath every other package, which is exactly the surprise this script exists to
#: prevent. Read from the wheel's metadata after install; this is the pre-flight guess.
_NUMPY_MIN = (2, 0)


def _installed_cupy() -> list:
    """Every installed distribution whose name looks like a CuPy build, as
    ``[(dist_name, version), …]``. More than one is itself a fault — the builds collide on
    the ``cupy`` import name, so whichever lands first on ``sys.path`` wins silently."""
    out = []
    try:
        from importlib.metadata import distributions
        for d in distributions():
            name = (d.metadata["Name"] or "").lower()
            if name == "cupy" or name.startswith("cupy-cuda") or name.startswith("cupy-rocm"):
                out.append((name, d.version))
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(out))


def _best(fn, repeat: int = 3) -> float:
    """Best of ``repeat`` timed runs, after one warm-up.

    The warm-up is not politeness: CuPy JIT-compiles each kernel on its first call, which
    costs seconds and would otherwise be charged to the GPU column and invert the result."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _pip(args: list, dry: bool) -> int:
    cmd = [sys.executable, "-m", "pip", *args]
    printable = " ".join(f'"{c}"' if " " in c or "[" in c else c for c in cmd)
    print(f"   $ {printable}")
    if dry:
        return 0
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Install the CuPy build that matches this machine's GPU + driver.")
    ap.add_argument("--check", action="store_true", help="report state, install nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the pip command only")
    ap.add_argument("--force", action="store_true",
                    help="reinstall, or replace a mismatched CuPy variant")
    ap.add_argument("--bench", action="store_true",
                    help="after verifying, time each kernel against scipy")
    ap.add_argument("--package", default="",
                    help="override the detected wheel (e.g. cupy-cuda12x[ctk])")
    a = ap.parse_args(argv)

    def die(msg: str) -> int:
        print("\n[FAIL] " + msg)
        return 1

    from nodegraph import gpu

    # ── 1. what is in this machine ────────────────────────────────────────────
    print("== 1. detect (driver library via ctypes — no nvidia-smi, no CuPy) ==")
    p = gpu.detect_platform()
    print(f"   platform   : {p.kind}")
    if p.driver_cuda:
        print(f"   driver CUDA: {p.driver_cuda[0]}.{p.driver_cuda[1]}")
    for i, d in enumerate(p.devices):
        print(f"   device {i}   : {d.name}  sm_{d.sm}  "
              f"{d.total_bytes / (1 << 30):.1f} GiB")
    if p.kind != "cuda":
        print(f"   note       : {p.note}")
        print("\n[OK] Nothing to install — this machine has no usable CUDA GPU, and the "
              "engine's CPU path is the default. " + gpu.install_hint())
        return 0

    want = a.package or gpu.recommended_package(p)
    if not want:
        return die("no CuPy wheel matches this driver/GPU combination.\n" +
                   gpu.install_hint(p))
    print(f"   -> matched  : {want}")
    print(f"      (chosen as the newest build whose CUDA runtime this driver can load, and "
          f"whose\n       architecture floor the oldest device here (sm_{p.min_sm}) still "
          f"meets)")

    # ── 2. numpy compatibility, checked BEFORE pip resolves anything ──────────
    print("== 2. numpy compatibility ==")
    import numpy as np
    nv = tuple(int(x) for x in np.__version__.split(".")[:2])
    print(f"   installed numpy {np.__version__}")
    if nv < _NUMPY_MIN:
        return die(f"CuPy needs numpy >= {_NUMPY_MIN[0]}.{_NUMPY_MIN[1]}; this environment "
                   f"has {np.__version__}. Upgrade numpy first, deliberately, so the change "
                   f"is yours rather than a side effect of this install.")
    print("   ok (pip will report if it still wants to move it — read that line)")

    # ── 3. anything already installed? ────────────────────────────────────────
    print("== 3. existing install ==")
    have = _installed_cupy()
    for name, ver in have:
        print(f"   {name} {ver}")
    if not have:
        print("   none")
    base = want.split("[")[0]
    wrong = [n for n, _ in have if n != base]
    if len(have) > 1:
        return die(f"more than one CuPy build is installed ({[n for n, _ in have]}). They "
                   f"share the `cupy` import name, so which one loads is down to sys.path "
                   f"order. Remove them all and re-run:\n"
                   f"    pip uninstall -y {' '.join(n for n, _ in have)}")
    if wrong and not a.force:
        return die(f"{wrong[0]} is installed but this machine wants {base}. Re-run with "
                   f"--force to replace it, or keep it if you know it works.")
    if have and not wrong and not a.force:
        print("   -> already the right build; verifying it rather than reinstalling "
              "(--force to reinstall)")
    elif a.check:
        print("\n[CHECK] not installed. Re-run without --check to install "
              f"{want} (~200-400 MB with headers).")
        return 1
    else:
        # ── 4. install ─────────────────────────────────────────────────────────
        print(f"== 4. install {want} ==")
        if wrong:
            print(f"   removing {wrong[0]} first")
            if _pip(["uninstall", "-y", wrong[0]], a.dry_run) != 0 and not a.dry_run:
                return die(f"could not uninstall {wrong[0]}")
        args = ["install", want]
        if a.force:
            args.insert(1, "--force-reinstall")
        t0 = time.time()
        if _pip(args, a.dry_run) != 0:
            return die("pip failed — see its output above.\n" + gpu.install_hint(p))
        if a.dry_run:
            print("\n[DRY RUN] nothing was installed.")
            return 0
        print(f"   installed in {time.time() - t0:.0f}s")

    # ── 5. verify: import, device, and the per-kernel equivalence gate ────────
    print("== 5. verify (each kernel must reproduce scipy before it is ever used) ==")
    os.environ["NODEGRAPH_GPU"] = "auto"          # this process only
    import importlib
    importlib.reload(gpu)
    if not gpu.available():
        return die("CuPy is installed but reports no usable device.\n" +
                   gpu.install_hint(p) +
                   "\n  A common cause on Windows is a driver older than the wheel's CUDA "
                   "major; --package cupy-cuda12x[ctk] steps down one.")
    print(f"   device: {gpu.device_name()}")

    from scipy import ndimage as sndi
    gy, gx = np.mgrid[0:2048, 0:2048].astype(np.float64)
    plane = np.ascontiguousarray(np.sin(gx / 7.0) + np.cos(gy / 5.0))
    # Both fixtures must clear the dispatch size threshold, or the check "passes" without
    # ever having exercised the card. The 3D one especially: a 32x512² volume is the shape
    # where the measured win actually lives (9-31x), so verifying only 2D would leave the
    # decisive case untested.
    vol = np.ascontiguousarray(
        np.stack([plane[:512, :512] * (1.0 + k / 32.0) for k in range(32)]))
    CASES = [("gaussian_filter", plane, dict(sigma=2.0)),
             ("gaussian_filter", vol, dict(sigma=(1.5, 2.0, 2.0))),
             ("median_filter", plane, dict(size=(5, 5))),
             ("median_filter", vol, dict(size=(3, 3, 3))),
             ("uniform_filter", plane, dict(size=15)),
             ("morphological_gradient", plane, dict(size=(7, 7))),
             ("grey_opening", plane, dict(size=(9, 9)))]

    hdr = f"   {'kernel':24s} {'shape':16s} {'max|Δ| vs scipy':>16s}"
    print(hdr + (f" {'cpu ms':>9s} {'gpu ms':>9s} {'speedup':>8s}" if a.bench else "")
          + "  verdict")
    n_gpu = 0
    for name, arr, kw in CASES:
        shape = "x".join(str(s) for s in arr.shape)
        want_r = np.asarray(getattr(sndi, name)(arr, **kw), dtype=np.float64)
        got = gpu.ndimage(name, arr, **kw)
        if got is None:
            why = gpu.REJECTED.get(f"{name}/{arr.ndim}d", "below the dispatch size threshold")
            print(f"   {name:24s} {shape:16s} {'—':>16s}"
                  + ("".ljust(29) if a.bench else "")
                  + f"  CPU ({why[:44]})")
            continue
        d = float(np.max(np.abs(want_r - np.asarray(got, dtype=np.float64))))
        row = f"   {name:24s} {shape:16s} {d:16.2e}"
        if a.bench:
            tc = _best(lambda: getattr(sndi, name)(arr, **kw))
            tg = _best(lambda: gpu.ndimage(name, arr, **kw))
            row += f" {tc*1e3:9.1f} {tg*1e3:9.1f} {tc/max(tg, 1e-9):7.2f}x"
        print(row + "  GPU")
        n_gpu += 1

    print(f"\n[OK] {gpu.describe()}")
    if gpu.REJECTED:
        print("\n     ops the gate REJECTED (they stay on the CPU — this is the intended")
        print("     behaviour, not a failure; a kernel that cannot reproduce scipy is")
        print("     never used on your data):")
        for k, v in sorted(gpu.REJECTED.items()):
            print(f"       {k}: {v[:110]}")
    if n_gpu == 0:
        return die("no kernel was dispatched. If every line says 'Failed to find CUDA "
                   "headers', the [ctk] extra did not install — re-run with\n"
                   f"    --force --package {base}[ctk]")
    print("\n     The GPU is OFF by default. Enable it per run:")
    print("       NODEGRAPH_GPU=auto  (PowerShell: $env:NODEGRAPH_GPU='auto')")
    print("     Worth it for heavy 3D volumetric filtering (measured 9-31x on 3D")
    print("     gaussian/median); roughly neutral for tiled 2D work. See MANUAL.md §6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
