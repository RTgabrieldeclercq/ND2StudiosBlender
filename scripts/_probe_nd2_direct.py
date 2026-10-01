"""Probe ``io.load``'s ``access=direct``/``access=auto`` paths against REAL ND2s — the
half ``nodegraph.selftest.test_nd2_direct_access`` cannot cover.

The selftest pins the arithmetic (channel-axis resolution, the sequence index, the pyramid
levels, the cache key, the size estimate, the disk-space margin) with no file, which is
what keeps it runnable anywhere. What it cannot do is prove that arithmetic addresses
*these* files correctly, or that the result is fast enough to scrub, or that ``auto``
actually decides the way the lab's own drive says it should. That is this script:

* every plane it reads is cross-checked against ``ND2File.to_dask()`` — the same reader
  :mod:`nodelab_v2.ingest` uses — so a disagreement means the fast path is wrong, not slow;
* the timings are taken through :func:`nodelab_v2.runner.render_plane_native`, the function
  the Viewer itself calls, rather than through the provider's own API, so the number is the
  one the user experiences;
* it asserts that nothing was written beside the source under ``direct``, which is the
  whole promise;
* it runs the actual ``access=auto`` decision (:meth:`~nodelab_v2.runner.EngineRunner.
  _effective_access`) against both of the lab's real files sitting on the SAME drive at
  the SAME time, and checks it picks the two different answers their two different
  situations call for: ``direct`` for the 453 GB series (no store, and nowhere near
  enough free space for one), ``ingest`` for its already-fully-ingested sibling (a
  complete store sitting right there, reused without even asking how much space is free).

Usage (``python`` = ``.venv\\Scripts\\python.exe``)::

    python -u -B scripts/_probe_nd2_direct.py [path/to/file.nd2]

Defaults to the lab's 453 GB 640 series. Exits 0 and prints ``ALL ND2-DIRECT PROBES
PASSED``; skips with 0 (and says so) when the file is not on this machine, so it is safe in
a checkout that does not have the data.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: The file this path was built for: 54 positions x 74 timepoints x 54 z, 1024^2, uint16,
#: uncompressed — 453 GB, which is ~280 GB and many hours to ingest and 4.7 s to open here.
DEFAULT = (r"E:\9.1.26_CRC_Gradient\20260902_210302_312"
           r"\Channel640_Seq0001.nd2")

#: Its sibling in the same acquisition folder — 3-channel, 12.7 GB, already fully
#: ingested (a complete 3-level store sits beside it). Used only by the ``auto`` section
#: below, and only when it is actually there; its absence never fails the probe, since the
#: interesting half of that check (the 453 GB file must go ``direct``) does not need it.
SIBLING = (r"E:\9.1.26_CRC_Gradient\20260902_210302_312"
          r"\Channelp53-GFP,ERK-mRuby2,H2B-iRFP670_Seq0000.nd2")

#: Planes cross-checked against the SDK. Larger than the provider's own open-time
#: `verify=3` because this probe is where the budget for it exists.
CHECK_PLANES = 24


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def main(argv: list) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT
    if not os.path.isfile(path):
        print(f"[skip] no such ND2 on this machine: {path}")
        print("ALL ND2-DIRECT PROBES PASSED (skipped — no file)")
        return 0

    from nodelab_v2.ingest import _to_6d
    from nodelab_v2.nd2_compat import import_nd2
    from nodelab_v2.nd2_direct import Nd2DirectProvider, can_open_direct
    from nodelab_v2.runner import EngineRunner, render_plane_native

    gb = os.path.getsize(path) / 1e9
    print(f"file: {os.path.basename(path)}  ({gb:,.1f} GB)")

    ok, why = can_open_direct(path)
    assert ok, f"can_open_direct refused: {why}"
    _ok(f"can_open_direct: yes")

    t0 = time.perf_counter()
    prov, env = EngineRunner._open_direct(path)
    t_open = time.perf_counter() - t0
    ax = prov.axes
    planes = ax.m * ax.t * ax.z * ax.c
    raw = planes * ax.y * ax.x * np.dtype(prov.dtype).itemsize
    _ok(f"opened in {t_open:.2f} s — {ax.m}x{ax.t}x{ax.z}x{ax.c} of {ax.y}x{ax.x} "
        f"{prov.dtype} = {planes:,} planes / {raw / 1e9:,.1f} GB, levels={prov.levels}")
    assert env.axes == ax
    assert env.metadata.get("pixel_size_um"), "calibration did not come through"

    # ── correctness: every plane against the sanctioned reader ────────────────
    with import_nd2().ND2File(path) as f:
        lazy = _to_6d(f.to_dask(), list(f.sizes.keys()))
        rnd = random.Random(0)
        for i in range(CHECK_PLANES):
            m, t = rnd.randrange(ax.m), rnd.randrange(ax.t)
            z, c = rnd.randrange(ax.z), rnd.randrange(ax.c)
            want = np.asarray(lazy[m, t, z, c])
            got = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
            assert got.shape == want.shape and np.array_equal(got, want), \
                f"plane (m={m},t={t},z={z},c={c}) disagrees with ND2File.to_dask()"
            # a window must equal the matching slice of the whole plane
            w = prov.get_region(0, m, t, z, c, 100, 612, 200, 712)
            assert np.array_equal(w, want[100:612, 200:712]), "window != plane slice"
    _ok(f"{CHECK_PLANES} random planes + windows are pixel-identical to "
        f"ND2File.to_dask()")

    # ── the promise: nothing written beside the source ───────────────────────
    d = os.path.dirname(os.path.abspath(path))
    before = sorted((n, os.path.getsize(os.path.join(d, n)))
                    for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)))
    for _ in range(50):
        prov.get_region(0, random.randrange(ax.m), random.randrange(ax.t),
                        random.randrange(ax.z), 0, 0, 512, 0, 512)
    after = sorted((n, os.path.getsize(os.path.join(d, n)))
                   for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)))
    assert before == after, "a direct read changed files beside the source"
    _ok("50 reads wrote nothing beside the source — no store, no copy")

    # ── speed, through the function the Viewer calls ─────────────────────────
    def scrub(label: str, coords) -> float:
        coords = list(coords)
        t0 = time.perf_counter()
        for m, t, z in coords:
            render_plane_native(prov, m, t, z, 0)
        dt = time.perf_counter() - t0
        per = 1000 * dt / len(coords)
        print(f"     {label:<28} {per:6.1f} ms/plane   {len(coords) / dt:5.0f} fps")
        return per

    print("  scrubbing (render_plane_native — the Viewer's own path):")
    mid_m, mid_t, mid_z = ax.m // 2, ax.t // 2, ax.z // 2
    worst = max(
        scrub("z (within one volume)", ((mid_m, mid_t, z) for z in range(ax.z))),
        scrub("t (across timepoints)", ((mid_m, t, mid_z) for t in range(ax.t))),
        scrub("m (across positions)", ((m, mid_t, mid_z) for m in range(ax.m))),
    )
    # 40 ms is 25 fps — the point below which scrubbing stops feeling attached to the
    # cursor. Deliberately not tighter: this is a floor for "interactive", not a
    # benchmark to defend, and it has ~6x of headroom over what this drive measures.
    assert worst < 40.0, (
        f"slowest scrub was {worst:.1f} ms/plane — direct access is no longer "
        f"interactive on this file; check whether the ND2 is compressed")
    _ok(f"every scrub axis stays interactive (worst {worst:.1f} ms/plane)")

    prov.close()
    _ok("provider closed; handles released")

    # ── the `auto` decision, on the real drive, against the real numbers ──────
    from nodelab_v2.ops import ACCESS_AUTO, ACCESS_DIRECT, ACCESS_INGEST

    # `EngineRunner.__new__` skips `__init__` (no QApplication needed here) — safe
    # because `_effective_access`/`_store_path_for` touch only the two plain dicts
    # seeded below, never Qt state.
    r = EngineRunner.__new__(EngineRunner)
    r._auto_access, r._auto_access_reason = {}, {}
    auto = r._effective_access(path, ACCESS_AUTO)
    reason = r._auto_access_reason[path]
    print(f"  auto({os.path.basename(path)}) -> {auto}  ({reason})")
    assert auto == ACCESS_DIRECT, (
        f"auto picked {auto!r} for a {gb:,.0f} GB file with no store on a drive this "
        f"low on space — either free space changed since this script was written, or "
        f"the auto decision regressed")
    _ok("auto picks direct for this file, on THIS drive, right now")

    if os.path.isfile(SIBLING):
        auto2 = r._effective_access(SIBLING, ACCESS_AUTO)
        reason2 = r._auto_access_reason[SIBLING]
        print(f"  auto({os.path.basename(SIBLING)}) -> {auto2}  ({reason2})")
        assert auto2 == ACCESS_INGEST and "existing store" in reason2, (
            f"auto picked {auto2!r} ({reason2}) for the sibling file's own already-"
            f"complete store — auto should reuse it outright, never re-deciding by "
            f"space it no longer needs")
        _ok("auto reuses the sibling's existing store without re-checking disk space")
    else:
        print(f"[skip] sibling file not present, skipping the existing-store half: "
              f"{SIBLING}")

    # a manual override must be untouched by any of the above — auto's decision for
    # ONE file must never leak into what a DIFFERENT card, or the same card set
    # explicitly, resolves to
    assert r._effective_access(path, ACCESS_INGEST) == ACCESS_INGEST
    assert r._effective_access(path, ACCESS_DIRECT) == ACCESS_DIRECT
    _ok("an explicit Ingest/Direct override is never second-guessed by auto's cache")

    print("ALL ND2-DIRECT PROBES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
