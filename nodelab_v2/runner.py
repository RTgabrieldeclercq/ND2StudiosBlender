"""EngineRunner — canvas → Engine off the UI thread (G7, LOCKED 2026-07-22: one worker
thread + **epoch registry**, no qasync — the engine is synchronous CPU work, so a worker
thread + queued-signal delivery is the whole bridge; stale results are dropped by epoch on
arrival. One pull runs at a time (latest-wins queueing).

The pull runs on ONE persistent, Python-created thread (:class:`_PullThread`), not on a
``QThreadPool``. That is a correctness requirement, not a preference: a pool's recycled
Qt-created thread is re-adopted by CPython as a new ``Dummy-N`` thread per pull, and torch's
per-thread state does not survive that — it killed the process natively on the second
CellSAM pull of a session. See :class:`_PullThread`. The read-only display jobs (decode,
prefetch, viewport detail) stay on the shared pool.

The runner owns the run-side model glue:

* **Snapshot at submit** — the headless :class:`Graph` is built on the GUI thread from
  the :class:`~nodelab_v2.document.GraphDocument` (``to_graph(for_run=True)``: strips
  the ``__locked__`` UI annotation, bypasses muted nodes), so the worker never touches
  live GUI state.
* **A persistent Memo across runs** — engines are rebuilt when the document revision
  changes, the two-hash memo carries over, so an unrelated edit recomputes only the
  invalidated chain (the C1 promise, now user-visible).
* **Source resolution** — an ``io.load`` root resolves its ``path`` param through
  :mod:`nodelab_v2.ingest` (ingested once to an on-disk b2nd store next to the file,
  re-opened lazily after); an empty path falls back to a deterministic
  :class:`~nodegraph.provider.SyntheticProvider` demo source with real calibration so
  the whole GUI runs out of the box. The resolved source envelope is delivered back to
  the document (``set_meta_seed``) — the G8 live widget re-seed. A pull resolves only the
  sources **its own closure reaches** (V2.21), so a canvas full of files costs nothing
  until each one's chain is asked for.
* **A second lane for ingest** (V2.21) — :meth:`EngineRunner.ingest_source` puts ONE
  source file on disk on a separate, multi-slot pool (:func:`ingest_workers`), so several
  ND2s ingest at once and none of them occupies the pull slot. It goes through the same
  ``_resolve_source``, under a per-file lock, so a pull and a background ingest of the
  same file cooperate (the pull waits and takes the result) instead of both writing the
  store. That lock is the only synchronization the design needs; everything else about an
  ingest is per-path and order-free.
* **Plane rendering** — a job may ask for a display plane at ``(m, t, z, c)``; it is
  read in the worker (through the C1 tile cache) and decimated to ``max_dim`` for the
  Viewer, so the GUI thread never blocks on a lazy-chain compute.
* **The solo-frame scope** (:meth:`EngineRunner.set_solo_frame`) — troubleshooting mode:
  every source seed is restricted to the frames the Viewer picked with a
  :class:`~nodegraph.provider.FrameSubsetProvider` (the one-frame
  :class:`~nodegraph.provider.FrameSliceProvider` when that is all that is scoped), so a
  pull computes those frames instead of the series. See that method for why it is done
  here rather than in the graph.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from nodegraph.checkpoint import checkpoint_envelope
from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.engine import Engine
from nodegraph.graph import Graph
from nodegraph.memo import Memo
from nodegraph.metadata import MetaEnvelope, position_subset
from nodegraph.parallel import (
    cpu_budget, memo_bytes, plane_cache_bytes, ram_budget, store_dir, tile_cache_bytes)
from nodegraph.provider import (
    FrameSliceProvider, FrameSubsetProvider, SyntheticProvider, _picked, subset_index)
from nodegraph.streaming import StreamProvider, TileCache
from nodelab_v2.ops import LOAD_OP, dock_seeds, dock_store_of

#: byte-budget LRU cap for the persistent Memo (Memo GC). The persistent memo is the
#: V2.04-flagged hazard: an eager full-raster node in a high-T zone would otherwise
#: retain one raster per iteration forever; eviction only costs a recompute
#: (correctness-safe).
#:
#: Sized from installed RAM (V2.14) instead of the old fixed 1 GiB — see
#: :func:`nodegraph.parallel.memo_bytes`. Override with ``NODEGRAPH_MEMO_BYTES``.
MEMO_BUDGET_BYTES = memo_bytes()


def ingest_workers() -> int:
    """How many source files may ingest **concurrently** (:meth:`EngineRunner.ingest_source`).

    Not one, because the point of the per-card ingest is to start a folder's worth of ND2s
    and walk away. Not unbounded either: an ingest is not idle-waiting on the disk —
    :meth:`~nodegraph.provider.B2ndProvider.write` fans its pyramid reduce across
    ``cpu_budget()`` threads and blosc2 compresses on its own — so past a handful they
    contend for the same cores and each one lands *later* than it would have alone. A small
    cap keeps several genuinely in flight (one file's decode overlaps another's compress and
    IO) without handing the machine over; the rest queue and start as slots free.

    ``NODELAB_INGEST_WORKERS`` overrides."""
    n = os.environ.get("NODELAB_INGEST_WORKERS", "").strip()
    if n:
        try:
            return max(1, int(n))
        except ValueError:
            pass
    return max(1, min(4, cpu_budget() // 4))


def _clean_source_path(path: Any) -> str:
    """Normalize an ``io.load`` ``path`` param: strip whitespace and surrounding quotes
    (Windows "Copy as path" wraps the path in double quotes — passing those to the reader
    yields a cryptic ``OSError [Errno 22]``). Shared so :meth:`EngineRunner.ingest_source`
    and :meth:`EngineRunner._resolve_source` cannot disagree about which file a card names,
    which would let a background ingest and a pull write the same store twice."""
    if not isinstance(path, str):
        return ""
    return path.strip().strip('"').strip("'").strip()


def source_key(path: str) -> Any:
    """The provider-cache key for a source path — the identity a store, a provider and an
    in-flight ingest are all shared under. An empty path is the synthetic demo source."""
    return ("image", os.path.abspath(path)) if path else ("synthetic",)

#: demo calibration for the synthetic fallback source (drives the ƒmd derive pills)
_SYNTH_META = {
    "pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 1.4,
    "objective_magnification": 60.0, "channel_emission_nm": [520.0, 640.0],
}
_SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)


def ensure_gui_ops() -> None:
    """Register the GUI-facing ops (``io.load`` source + ``view.viewer`` pass-through).
    Delegates to the **Qt-free** :mod:`nodelab_v2.ops` so the same registration (incl.
    ``view.viewer``'s compute in ``COMPUTES``) is available to a headless consumer of a
    saved graph — the GUI is not the only path that must run one."""
    from nodelab_v2.ops import ensure_ops
    ensure_ops()


#: Longest edge, in pixels, a display plane is decimated to before it becomes a texture.
#:
#: 4096, not the historical 2048, and the reason is the **stitched mosaic**: a display plane
#: is capped ONCE and zooming is a pure view transform over that texture, so the cap is the
#: only resolution the user will ever get. For a 2048² camera frame that was free — the cap
#: was never reached. For a 13106² mosaic it decided everything: the pyramid bottoms out at
#: 3277², which 2048 then decimated to 1638² — a 1/8 view of the data, which is exactly the
#: "grainy, nothing like the raw resolution" report this constant answers. At 4096 that same
#: level 3277² is shown WHOLE, doubling the linear resolution **for the same read** (the
#: pyramid level chosen does not change), which is why this is a cap change and not a
#: cost/quality trade.
#:
#: The cost is texture and cache bytes: a 4096² uint16 plane is 32 MB against 8 MB, so the
#: RAM-derived :class:`PlaneCache` holds a quarter as many frames. Override with
#: ``NODELAB_MAX_DISPLAY_DIM`` on a machine where that hurts more than the detail helps.
MAX_DISPLAY_DIM = int(os.environ.get("NODELAB_MAX_DISPLAY_DIM", "") or 4096)

#: Share of installed RAM the Viewer may hold in decoded display frames — the budget that
#: decides whether a big frame is shown WHOLE at full resolution or progressively off the
#: pyramid (V2.23). ``NODELAB_DISPLAY_RAM_PCT`` overrides the percentage,
#: ``NODELAB_DISPLAY_RAM_BYTES`` the absolute number.
#:
#: 25% is Nikon's number: NIS-Elements sets ``MaxMemoryImageSize`` to a quarter of RAM at
#: startup and opens an image in "normal mode" when it fits and "progressive mode" — thumbnail
#: first, detail as you zoom, *and much of the processing menu disabled* — when it does not.
#: Borrowing the threshold is deliberate: it is the number a decade of microscopy users have
#: had their expectations set by, and the decision it drives here is the same one.
#:
#: Configurable because the right answer is a property of the MACHINE, and this codebase
#: already runs on both a 256 GiB workstation (a quarter of which holds 160 full-resolution
#: 7168² frames) and a laptop where a quarter is 4 GiB.
DISPLAY_RAM_SHARE = max(0.01, min(0.9,
                                  float(os.environ.get("NODELAB_DISPLAY_RAM_PCT", "")
                                        or 25.0) / 100.0))


def display_ram_bytes() -> int:
    """How many bytes of decoded display frames the Viewer may hold at once."""
    override = os.environ.get("NODELAB_DISPLAY_RAM_BYTES", "").strip()
    if override:
        try:
            return max(64 * 1024 * 1024, int(override))
        except ValueError:
            pass
    return ram_budget(DISPLAY_RAM_SHARE, floor=512 * 1024 * 1024,
                      cap=192 * 1024 * 1024 * 1024)


#: What one texture axis may be before the GPU refuses it. Replaced with the context's real
#: ``GL_MAX_TEXTURE_SIZE`` by :meth:`EngineRunner.set_display_limits` as soon as a surface
#: comes up — 8192 until then, which every GL 3.3 implementation this decade meets and which
#: is the conservative direction to be wrong in: too small only costs sharpness, while too
#: large is a black frame.
DEFAULT_TEXTURE_LIMIT = 8192


#: Bytes of GPU texture the shown channels may occupy at once. The uploader packs each plane
#: into **RGBA8** (:meth:`nodelab_v2.glview.GLImageView._upload` — 16-bit split across R and G),
#: so one frame costs ``4 * y * x`` of VRAM *and* two transient CPU copies of the same size:
#: a 7168² plane is 205 MB three times over, per channel. ``NODELAB_TEXTURE_BYTES`` overrides.
#:
#: This exists because ``GL_MAX_TEXTURE_SIZE`` is the wrong ceiling to trust — this GPU reports
#: 32768, which would permit a 4 GB texture. Uploads are not error-checked (there is no
#: ``glGetError`` on that path), so exceeding VRAM does not raise: it leaves the previous
#: texture's content bound, which is a frame with part of the picture missing.
TEXTURE_BYTES = int(os.environ.get("NODELAB_TEXTURE_BYTES", "") or 512 * 1024 * 1024)


def display_cap(axes: Any, *, texture_limit: int, bytes_per_px: int = 8,
                planes: int = 1, streaming: bool = False) -> int:
    """The display cap for a frame of ``axes``: its own long edge when the whole thing can be
    shown at FULL resolution *affordably*, else :data:`MAX_DISPLAY_DIM`.

    Four ceilings, and a frame has to clear all of them:

    * the **texture limit** in px, because the overview is one texture per channel;
    * the **texture BYTES** those channels cost (:data:`TEXTURE_BYTES`) — the limit that
      actually bites, see its note;
    * the **RAM budget** (:func:`display_ram_bytes`), because these frames land in the
      :class:`PlaneCache` and playback wants a series of them resident, not one;
    * **cost to produce**: ``streaming`` marks a provider that COMPUTES each plane (a stitch, a
      filter chain) rather than reading bytes. Full resolution there is not a texture decision,
      it is a 4x-per-frame decision — measured on the WellA3 mosaic, level 1 is 0.21 s a frame
      and level 0 is 1.0 s — and it buys detail you can only see zoomed in, which the viewport
      detail patch already serves at full resolution where you are actually looking. So a live
      mosaic stays on the pyramid and a **baked** one (Flatten to Large Image → a store, which
      is not streaming) gets the whole frame. That is the same normal-vs-progressive split
      NIS-Elements makes, with cost added to the memory test.

    ``bytes_per_px`` is deliberately pessimistic by default (8 — float64, what a stitch canvas
    serves): budgeting may be conservative, and over-committing is the failure that matters.

    Below the cap nothing changes: a camera frame was never decimated and is unaffected.
    """
    long_edge = max(1, int(getattr(axes, "y", 1)), int(getattr(axes, "x", 1)))
    if long_edge <= MAX_DISPLAY_DIM:
        return MAX_DISPLAY_DIM                # nothing to decide; it fits either way
    if streaming:
        return MAX_DISPLAY_DIM                # every frame is a compute: stay progressive
    px = int(axes.y) * int(axes.x) * max(1, int(planes))
    if (long_edge <= int(texture_limit)
            and px * 4 <= TEXTURE_BYTES
            and px * int(max(1, bytes_per_px)) <= display_ram_bytes()):
        return long_edge
    return MAX_DISPLAY_DIM

#: How many channels ONE overlay source may contribute to the display.
#:
#: The ceiling is the GL sampler bank (``nodelab_v2.glview._MAX_CH`` = 8), shared with the
#: primary's own channels — so this is deliberately small rather than "whatever the file
#: has". A 2-channel primary plus a 2-channel secondary is 4 of the 8; letting a 6-channel
#: secondary in would silently push the primary's own channels out of the shader.
MAX_OVERLAY_CHANNELS = 2

def _gl_channel_cap(default: int = 8) -> int:
    """The shader's sampler-bank size, taken from :mod:`nodelab_v2.glview` itself so the
    two cannot drift. Imported lazily and defensively: the runner must stay usable where
    ``QtOpenGLWidgets`` is unavailable (a headless test box), and the cap is the same number
    either way."""
    try:
        from nodelab_v2.glview import _MAX_CH
        return int(_MAX_CH)
    except Exception:      # noqa: BLE001 — no GL module here; the constant is still right
        return default


#: HARD ceiling on the primary's channels plus every overlay source's. A chain that exceeds
#: it drops whole sources — announced in `overlay_note`, never silently, because a missing
#: layer is indistinguishable from a source that simply had nothing to show.
GL_MAX_CHANNELS = _gl_channel_cap()

#: Cells on the long edge of the checkerboard blend. 8 is the registration-QA convention —
#: coarse enough that each square shows real structure to judge, fine enough that a shift
#: breaks the alignment at several seams at once. Not a socket: it changes nothing about
#: what is being compared, only how densely the comparison is tiled.
OVERLAY_CHECKER_CELLS = 8.0


def _z_um_of(md, axes, m: int, z: int):
    """The absolute µm focus of the displayed plane, for picking the secondary's slice.
    ``None`` whenever the primary cannot be placed in Z, which simply means the secondary
    draws its first slice."""
    try:
        from nodegraph.placement import z_um_of_slice
        return z_um_of_slice(md, axes, int(m), int(z))
    except Exception:      # noqa: BLE001 — a display hint, never a failure
        return None


def _pick_level(provider: Any, max_dim: int):
    """The finest pyramid level whose plane fits ``max_dim`` — or the coarsest there is,
    when even that overflows (every computing provider without a pyramid, and a mosaic
    whose pyramid bottoms out above the cap). Returns ``(level, axes)``."""
    level, ax = 0, provider.level_axes(0)
    for lv in range(getattr(provider, "levels", 1)):
        lax = provider.level_axes(lv)
        level, ax = lv, lax
        if max(lax.y, lax.x) <= max_dim:
            break
    return level, ax


def _pick_window_level(provider: Any, want: Tuple[float, float, float, float],
                       budget: int) -> Tuple[int, Any]:
    """The finest pyramid level at which the fractional window ``want`` fits ``budget`` px —
    or the coarsest there is. Returns ``(level, level_axes)``.

    The window, not the whole plane: that is the whole difference between
    :func:`_pick_level` and this. A 13106² mosaic has no level whose *plane* is small, and
    picking by plane size is what makes a zoomed-in read coarse — while the 1/20th of it that
    is on screen fits level 0 comfortably.
    """
    fy0, fy1, fx0, fx1 = want
    level, ax = 0, provider.level_axes(0)
    for lv in range(max(1, int(getattr(provider, "levels", 1)))):
        cand = provider.level_axes(lv)
        level, ax = lv, cand
        if max((fy1 - fy0) * cand.y, (fx1 - fx0) * cand.x) <= budget:
            break
    return level, ax


def _snap_window(want: Tuple[float, float, float, float],
                 lax: Any) -> Tuple[int, int, int, int]:
    """A fractional window as whole pixel bounds ``(y0, y1, x0, x1)`` of ``lax``, snapped
    OUTWARD and never empty — a rect that is narrower than what was asked for would leave a
    hairline of the coarser picture showing along its edge."""
    fy0, fy1, fx0, fx1 = want
    ny, nx = max(1, int(lax.y)), max(1, int(lax.x))
    y0, y1 = int(np.floor(fy0 * ny)), int(np.ceil(fy1 * ny))
    x0, x1 = int(np.floor(fx0 * nx)), int(np.ceil(fx1 * nx))
    y1, x1 = min(ny, max(y0 + 1, y1)), min(nx, max(x0 + 1, x1))
    y0, x0 = max(0, min(y0, y1 - 1)), max(0, min(x0, x1 - 1))
    return y0, y1, x0, x1


def _window_read(provider: Any, m: int, t: int, z: int, c: int,
                 want: Tuple[float, float, float, float], budget: int
                 ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """``(pixels, covered)`` for a fractional window of one plane — the read behind both the
    viewport detail patch and the overlay's secondary.

    ``covered`` is the window the returned pixels REALLY span, in the same fractional units
    as ``want``, because it was snapped out to whole pixels of whichever level was chosen.
    Area-averaged down to ``budget`` afterwards for the same reason :func:`_fit_plane` exists:
    point-sampling a window that overshoots the budget is noisier than the data it came from.
    """
    level, lax = _pick_window_level(provider, want, budget)
    y0, y1, x0, x1 = _snap_window(want, lax)
    plane = np.asarray(provider.get_region(level, int(m), int(t), int(z), int(c),
                                           y0, y1, x0, x1))
    covered = (y0 / float(max(1, lax.y)), y1 / float(max(1, lax.y)),
               x0 / float(max(1, lax.x)), x1 / float(max(1, lax.x)))
    return _fit_plane(plane, budget), covered


def _display_dtype(ds: Any) -> Any:
    """The dtype this Dataset's DISPLAY planes may be narrowed to, or ``None``.

    ``bit_depth`` is the payload's own statement that its samples are integers of N bits —
    maintained across the catalog and dropped to ``None`` by every node that makes the values
    continuous (``enhance.normalize``, the overlay's ``resample``, any float field). So it is
    exactly the right gate: an integer image narrows, a strain field or a probability does not,
    and neither has to be scanned to find out.

    ``uint16`` for anything up to 16 bits rather than ``uint8`` for the small ones: the GPU
    uploader packs to 16-bit regardless, so a second narrowing would buy bytes in the cache and
    lose a LUT range that a user can legitimately window into.
    """
    try:
        bd = ds.metadata.get("bit_depth")
    except Exception:  # noqa: BLE001 — a display hint, never a failure
        return None
    if isinstance(bd, bool) or not isinstance(bd, (int, float)):
        return None
    return np.uint16 if 0 < int(bd) <= 16 else None


def _narrow(plane: np.ndarray, as_dtype: Any) -> Optional[np.ndarray]:
    """``plane`` as ``as_dtype``, or ``None`` when that would not be lossless enough to do
    silently.

    The one caller is :func:`_fit_plane`, and the point is bytes: ``util.stitch`` fuses in
    float64 (it has to — a feather blend is a weighted mean), so a 7168² mosaic frame reaches
    the display as 392 MB of float64 carrying 12-bit data. At 98 MB instead, four times as
    many frames stay resident, which is the difference between a series that plays from cache
    and one that re-decodes.

    **Display only.** Level 0 of a stitch is what every node compute reads, and rounding a
    feather blend there would move a measured intensity — small, but a measurement change made
    behind the user's back. That is what the Dock's explicit ``precision`` is for. This copy is
    the one handed to the texture uploader and nothing else.

    Refuses rather than wraps when the values do not fit. The min/max that decides costs one
    pass (~8% of a full-resolution mosaic read); a wrapped intensity would look like data.
    """
    if as_dtype is None:
        return None
    dt = np.dtype(as_dtype)
    if plane.dtype == dt or not np.issubdtype(dt, np.integer):
        return None
    info = np.iinfo(dt)
    if plane.size:
        lo, hi = float(np.nanmin(plane)), float(np.nanmax(plane))
        if not (info.min <= lo and hi <= info.max):
            return None
    return np.ascontiguousarray(np.rint(plane).astype(dt))


def _fit_plane(plane: np.ndarray, max_dim: int, *, as_dtype: Any = None) -> np.ndarray:
    """Decimate ``plane`` to ``max_dim`` on its long edge by **area-averaging**.

    This used to be ``plane[::stride, ::stride]``, and point-sampling was the wrong tool
    twice over:

    * it keeps sensor noise at full amplitude while throwing away ``1 - 1/stride²`` of the
      data that would have averaged it down — measured on a Poisson-noise mosaic, local
      roughness 0.0488 stride-decimated versus 0.0288 area-averaged, against 0.0418 for the
      raw full-resolution image. The decimated view was NOISIER than the data it came from,
      which is what "grainy" meant;
    * the stride is an integer, so a 3277-px plane under a 2048 cap became 1638 — half the
      cap thrown away for nothing. Area-averaging hits the cap exactly.

    Integer dtypes are rounded back to themselves, so the GPU uploader still picks the same
    texture format and the LUT still spans the same range. ``_sample`` maps a cursor to a
    texel by RATIO, so nothing downstream depends on the output shape.

    ``as_dtype`` narrows the display copy (see :func:`_narrow`) — free, because both paths
    below already materialize one."""
    h, w = plane.shape[:2]
    big = max(h, w)
    if big <= max_dim:
        narrowed = _narrow(plane, as_dtype)
        return np.ascontiguousarray(plane) if narrowed is None else narrowed
    nh = max(1, int(round(h * max_dim / big)))
    nw = max(1, int(round(w * max_dim / big)))
    ys = (np.arange(nh) * h) // nh
    xs = (np.arange(nw) * w) // nw
    acc = np.add.reduceat(np.add.reduceat(plane.astype(np.float64), ys, axis=0),
                          xs, axis=1)
    counts = (np.diff(np.append(ys, h))[:, None]
              * np.diff(np.append(xs, w))[None, :])
    out = acc / counts
    keep = (plane.dtype if np.issubdtype(plane.dtype, np.integer)
            else as_dtype if as_dtype is not None else None)
    if keep is not None:
        narrowed = _narrow(out, keep)
        out = narrowed if narrowed is not None else out
    return np.ascontiguousarray(out)


def _plane_key(node_id: str, pin: Optional[Any], m: int, t: int, z: int, ch: int,
               cap: int) -> tuple:
    """The one spelling of a :class:`PlaneCache` address.

    It exists because there were two. `_plane_addrs` built a 7-tuple ending in the display
    cap — for the reason its own comment gives, that several readers write these keys and a
    disagreement must be a miss rather than a plane served at the wrong size — while
    `_decode_planes`' two fallback branches built a 6-tuple without it. Those writes landed
    in slots `_cached_planes` never probes, so the plane was re-decoded (or the overlay
    re-composed) on every scrub. Worse, the fallback also omitted `max_dim=cap` on the read,
    so with a cap above :data:`MAX_DISPLAY_DIM` — every docked mosaic that fits the texture
    and RAM budgets — the shape-reference plane came back decimated to 4096 and the overlay
    was composed onto a grid the primary channels do not share."""
    return (node_id, pin, int(m), int(t), int(z), int(ch), int(cap))


def render_plane(provider: Any, m: int, t: int, z: int, c: int,
                 *, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    """A display plane at ``(m,t,z,c)``: picks the finest pyramid level that fits
    ``max_dim`` (a streaming provider usually has no pyramid — V2.04 §5-1, and
    ``MultiViewProvider`` is the one exception), then area-averages down to the cap.
    Returns float64 (Y', X')."""
    level, ax = _pick_level(provider, max_dim)
    plane = np.asarray(provider.get_region(level, m, t, z, c, 0, ax.y, 0, ax.x),
                       dtype=float)
    return _fit_plane(plane, max_dim)


def render_plane_native(provider: Any, m: int, t: int, z: int, c: int,
                        *, max_dim: int = MAX_DISPLAY_DIM,
                        as_dtype: Any = None) -> Tuple[np.ndarray, int]:
    """A display plane at ``(m,t,z,c)`` in its **native dtype** (uint8/uint16/float) —
    the GPU uploader picks the texture internal format from the dtype, and the CPU
    fallback casts to float. Same level-selection + decimation as :func:`render_plane`,
    but without the ``dtype=float`` cast (which is the expensive per-frame work we push
    onto the GPU / do once). Returns ``(plane2d, level)``.

    ``max_dim`` is the caller's display cap — :func:`display_cap` resolves it per provider, so
    a frame that fits the texture limit and the RAM budget comes back at LEVEL 0 rather than
    off the pyramid. ``as_dtype`` narrows the copy (see :func:`_narrow`)."""
    level, ax = _pick_level(provider, max_dim)
    plane = np.asarray(provider.get_region(level, m, t, z, c, 0, ax.y, 0, ax.x))
    return _fit_plane(plane, max_dim, as_dtype=as_dtype), level


#: the run scope: the ``(ms, ts, zs)`` picks every source seed is restricted to, or
#: ``None`` for the whole series. ``ms``/``ts`` are sorted and never empty (they fall back
#: to the display cursor); ``zs`` is ``None`` when the whole volume runs, which is the
#: default because a 3D node needs one.
Pin = Tuple[Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]]

#: PlaneCache namespace for SOURCE (pre-enhancement) planes — see
#: :meth:`EngineRunner.raw_plane`. A sentinel string rather than a node id so it can
#: never collide with one.
_RAW_TAG = "\x00raw"


def _pin_frames(provider: Any, env: MetaEnvelope, pin: Pin) -> Tuple[Any, MetaEnvelope]:
    """Scope one resolved source to the picked frames and planes: a
    :class:`~nodegraph.provider.FrameSubsetProvider` over ``provider`` plus the matching
    shortened envelope (the solo-frame scope, :meth:`EngineRunner.set_solo_frame`).

    A single picked frame gets the :class:`~nodegraph.provider.FrameSliceProvider`
    instead — the same view, but with its own fingerprint tag, so the one-frame scope
    keeps keying the memo exactly as it did before subsets existed.

    The picks are **clamped** to the source's own axes rather than validated (the
    provider's own :func:`~nodegraph.provider._picked` does it): the display cursor is
    bounded by whatever the Viewer last showed, and a graph edit can shrink a source under
    it. Clamping shows the nearest real plane; raising would turn a stale cursor into a
    failed pull.

    Scalar calibration passes through untouched — ``dt_s`` and ``z_step_um`` still describe
    the source's own interval and spacing, exactly as
    :func:`nodegraph.metadata.frame_slice` reasons for ``zone.frame``.

    **Per-M lists are not scalar and do NOT pass through**: they are indexed by multipoint
    (:data:`nodegraph.metadata.PER_POSITION_KEYS`) and read positionally, so narrowing M
    while leaving them full-length hands every consumer the wrong POSITION's coordinate.
    Picking M = (10, 11, 12) and stitching put the tiles at positions 0-2's stage µm, which
    reads as a handedness bug rather than as a stale list. `origin_um` is subset here for
    the same reason and by the same call — the identical discipline
    :func:`nodegraph.metadata.channel_subset` applies on the channel axis.

    The picks are clamped by the provider, so they are re-derived from the VIEW's own axes
    rather than from ``pin`` — a cursor past the end of a shrunken source names a real
    position after clamping, and the subset must use the position actually served.
    """
    ms, ts, zs = pin
    view = (FrameSliceProvider(provider, ms[0], ts[0], zs) if len(ms) == len(ts) == 1
            else FrameSubsetProvider(provider, ms, ts, zs))
    out = env.with_axes(replace(env.axes, m=view.axes.m, t=view.axes.t, z=view.axes.z))
    kept = _picked(ms, int(env.axes.m))
    changes = position_subset(env.metadata, kept)
    return view, (out.with_metadata(**changes) if changes else out)


class PlaneCache:
    """A byte-budgeted LRU of decoded, display-decimated planes (native dtype), keyed by
    ``(node_id, revision, m, t, z, c)``. Separate from the engine ``Memo``/``TileCache``:
    it holds *ready-to-upload* planes so scrubbing/playback and the background prefetcher
    share one warm store. Thread-safe (the prefetch pool + the GUI thread both touch it).
    Decimated planes are small (2048² uint16 ≈ 8 MB), so hundreds of T fit in the budget.

    The default budget is RAM-derived (V2.14, :func:`nodegraph.parallel.plane_cache_bytes`).
    At the old fixed 512 MiB it held only ~64 planes, so a 200-frame 2-channel series
    evicted faster than a scrub could read it and every frame was re-decoded.

    **One budget, not two** (V2.23). This cache holds decoded display frames, which is exactly
    what :func:`display_ram_bytes` is the budget *for* — and the two disagreeing is not a
    tuning question, it is a bug: a preload sized against one and stored in the other evicts
    its own head and re-decodes every lap. So the default is the display budget (25% of RAM,
    configurable), floored by the historical ``plane_cache_bytes`` so no machine gets less than
    it used to. ``NODEGRAPH_PLANE_CACHE_BYTES`` still overrides both."""

    def __init__(self, budget_bytes: Optional[int] = None) -> None:
        if budget_bytes is None:
            budget_bytes = max(plane_cache_bytes(), display_ram_bytes())
        self._budget = int(budget_bytes)
        self._d: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def budget(self) -> int:
        """The byte ceiling — what a preload must size itself against, since this is where the
        frames it decodes actually live."""
        return self._budget

    def get(self, key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._d.get(key)
            if arr is not None:
                self._d.move_to_end(key)
            return arr

    def put(self, key: tuple, arr: np.ndarray) -> None:
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= int(getattr(old, "nbytes", 0))
            self._d[key] = arr
            self._bytes += int(getattr(arr, "nbytes", 0))
            while self._bytes > self._budget and len(self._d) > 1:
                _k, v = self._d.popitem(last=False)
                self._bytes -= int(getattr(v, "nbytes", 0))

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


#: minimum gap between two *fractional* progress deliveries for one node (seconds). A
#: per-plane compute can call ``ctx.progress`` thousands of times a second; the card only
#: needs enough to look alive. Start/finish transitions are never throttled — and neither
#: is the final ``done == total`` update, so a bar always lands full. A **frame boundary**
#: is likewise exempt: the frame bar steps rarely and each step is the one update that
#: carries it, so dropping one to the throttle would leave it reading a frame behind for as
#: long as the next frame takes.
PROGRESS_MIN_INTERVAL_S = 0.05


class _Job:
    __slots__ = ("epoch", "graph", "revision", "node_id", "coords", "channels",
                 "sources", "all_sources", "pin", "bake")

    def __init__(self, epoch: int, graph: Graph, revision: int, node_id: str,
                 coords: Optional[Tuple[int, int, int, int]],
                 channels: Optional[Tuple[int, ...]],
                 sources: Dict[str, Dict[str, Any]],
                 pin: Optional[Pin] = None,
                 bake: Optional[Dict[str, Any]] = None,
                 all_sources: Optional[frozenset] = None) -> None:
        self.epoch = epoch
        self.graph = graph
        self.revision = revision
        self.node_id = node_id
        self.coords = coords
        self.channels = channels      # channels to render into a colour composite
        self.sources = sources        # node_id -> {"path": str}: the io.load roots THIS
        #                               pull's closure actually reaches (V2.21)
        # …and the ids of EVERY io.load in the document. Only the difference between the
        # two is interesting: a node in `all_sources` but not `sources` is a source this
        # pull does not touch, so it keeps whatever seed the engine already holds for it
        # instead of being swept as stale (see :meth:`EngineRunner._ensure_engine`).
        self.all_sources = all_sources if all_sources is not None else frozenset(sources)
        self.pin = pin                # solo-frame scope: the (ms, ts) sources are cut to
        self.bake = bake              # a Dock bake request: {store, precision, sig, …}


class _PullThread:
    """The ONE thread every node compute runs on, for the life of the process.

    A pull used to be a :class:`QRunnable` on ``QThreadPool.globalInstance()``, and that is
    what made the app die — natively, with no traceback — on the **second** CellSAM pull of a
    session (2026-08-03). A pool recycles one OS thread but *Qt* created it, so CPython
    re-adopts it as a fresh ``Dummy-N`` ``threading.Thread`` on every pull. Measured: three
    pulls reported ``Dummy-1``/``Dummy-2``/``Dummy-3`` all with ``ident=9136`` — one OS
    thread wearing three Python thread states. torch leaves per-thread native state attached
    to that OS thread, so the second pull's inference ran against state belonging to a thread
    identity that no longer existed: ``Windows fatal exception: code 0xc0000374``
    (heap corruption) inside ``torch.from_numpy``, and when a model rebuild happened on that
    thread CPython said it outright — ``_PyThreadState_Attach: non-NULL old thread state``.

    A plain ``threading.Thread`` is the fix precisely because **Python** creates it: one real
    ``Thread`` object, one thread state, created once and never re-adopted. That is the
    configuration the isolating repros proved safe — the same 32 tiled CellSAM calls that
    pass on the main thread crash on the second pull through a pool. A ``QThread`` would not
    have done: it is Qt-created too, and only avoids the bug by accident of living long
    enough. Nothing here needs a Qt event loop; results reach the GUI the way they always
    did, through the runner's queued signals, which are safe to emit from any thread.

    It costs nothing in concurrency: a pull was ALREADY single-slot latest-wins
    (``_busy``/``_pending``, resolved on the GUI thread), so exactly one job ran at a time
    anyway. The decode / prefetch / detail jobs stay on the shared pool — they read
    providers rather than owning native per-thread state, and serializing them would undo
    the display fast path.

    Daemon, so a queued pull can never hold the app open at exit.
    """

    def __init__(self) -> None:
        import queue
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="nodelab-pull",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            try:
                job.run()
            except BaseException:  # noqa: BLE001 — the loop must outlive any single job
                # `_Worker.run` already converts every exception into a delivered packet, so
                # reaching here means the delivery itself failed. Swallow it: a dead pull
                # thread would wedge every future pull with no way back short of a restart.
                traceback.print_exc()

    def start(self, job: Any) -> None:
        """Queue a job (duck-typed ``.run()``) — the ``QThreadPool.start`` signature, so
        the call sites read unchanged."""
        self._q.put(job)


class _Worker(QRunnable):
    def __init__(self, runner: "EngineRunner", job: _Job) -> None:
        super().__init__()
        self._r = runner
        self._job = job

    def run(self) -> None:  # worker thread
        r, job = self._r, self._job
        t0 = time.perf_counter()
        try:
            engine = r._ensure_engine(job)
            engine.observer = r._make_observer(job.epoch)
            if job.bake is not None:
                r._run_bake(engine, job)
                r._done.emit((job.epoch, job.node_id, None, None, None,
                              time.perf_counter() - t0, None, job.revision,
                              None, None, job.pin))
                return
            payload = engine.pull(job.node_id)
            # An overlay node's picture needs a SECOND chain evaluated. Resolved here, on
            # the worker, because pulling the secondary is real work (and normally a memo
            # hit); the compose itself happens per displayed plane in `_decode_planes`.
            r._overlay_ctx = r._resolve_overlay(engine, job.graph, job.node_id, payload)
            plane = None                    # dict {channel_index: 2-D native plane}
            axes = None
            if isinstance(payload, Dataset) and payload.image is not None:
                axes = payload.axes
                # Set BEFORE the decode: `_plane_addrs` folds the display cap into every plane
                # key and the cap's byte estimate depends on whether the planes narrow. Same
                # write-on-the-worker discipline as `_overlay_ctx` above.
                r._viewer_dtype = _display_dtype(payload)
                if job.coords is not None:
                    # a lazy chain does its real work HERE, under the viewed node's name —
                    # report it as that node's state so the card isn't idle while the
                    # provider reads tiles (the honest counterpart to ctx.progress).
                    r._progress.emit(("decode", job.node_id,
                                      {"epoch": job.epoch, "op_key": ""}))
                    plane = r._decode_planes(payload.image, job.node_id,
                                             job.coords, job.channels, axes,
                                             pin=job.pin, overlay_all=True)
            dt = time.perf_counter() - t0
            r._done.emit((job.epoch, job.node_id, payload, plane, axes, dt, None,
                          job.revision, job.coords, job.channels, job.pin))
        except Exception:  # noqa: BLE001 — full trace to the GUI, never a dead thread
            r._done.emit((job.epoch, job.node_id, None, None, None,
                          time.perf_counter() - t0, traceback.format_exc(),
                          job.revision, job.coords, job.channels, job.pin))


class _IngestJob(QRunnable):
    """One source file's ingest, off the GUI thread and **outside the pull slot** (V2.21).

    A pull is single-slot and latest-wins for good reasons (one engine, one held viewer
    provider, one epoch), but an ingest is none of those things: it is a pure, idempotent
    function of a path — decode the ND2/TIFF once into its own ``.b2nd`` store beside it —
    whose only output is a directory on disk and an entry in
    :attr:`EngineRunner._providers`, keyed by that path. Two different files share nothing,
    so N of them run at once on their own pool (:func:`ingest_workers`) while the pull slot
    stays free for the rest of the app.

    It reaches the store through the SAME :meth:`EngineRunner._resolve_source` a pull uses,
    rather than calling :func:`~nodelab_v2.ingest.ingest_image` itself. That is what makes
    the two paths cooperate instead of race: the per-key lock in there means a pull that
    wants a file already being ingested WAITS for it and then takes the cached provider,
    where a second writer would have torn the store the first was still filling."""

    def __init__(self, runner: "EngineRunner", node_id: str, path: str, key: Any) -> None:
        super().__init__()
        self._r = runner
        self._node_id = node_id
        self._path = path
        self._key = key

    def run(self) -> None:  # ingest-pool thread
        r = self._r
        t0 = time.perf_counter()
        err = None
        try:
            # An un-epoched observer: an ingest is not part of any pull, so it must not be
            # silenced when the epoch moves on (an edit, or another node being pulled while
            # this runs). Its card reports for as long as it takes.
            r._resolve_source(self._node_id, {"path": self._path},
                              observe=r._make_observer(None))
        except Exception:  # noqa: BLE001 — full trace to the GUI, never a dead thread
            err = traceback.format_exc()
        r._ingest_done.emit((self._node_id, self._key, time.perf_counter() - t0, err))


class _DecodeJob(QRunnable):
    """Decodes the DISPLAYED plane(s) of a coords-only request on a pool thread.

    The fast path used to decode a :class:`PlaneCache` miss inline, on the GUI thread, on
    the premise stated in its own docstring: "a miss decodes one small plane". That holds
    for a store-backed provider — a decompress, single-digit milliseconds — and fails
    completely for a lazy *computing* provider, where the same read RUNS THE NODE. On the
    WellA3 640 series (12×16×210×1024², 3D Deconvolve) one plane read is a whole-volume
    Richardson–Lucy — ~130 s with the V2.19 kernel, and ~230 s when this was found: minutes
    with no repaint, no status, and no way out — a frozen application, reported as "the
    deconvolved viewer was very slow".

    Warm frames still serve inline from the cache (:meth:`EngineRunner._serve_from_cache`),
    so scrubbing already-decoded planes keeps its zero-hop latency; this runs only when the
    pixels are not in hand. One decode is in flight at a time with a single latest-wins
    pending slot — the same shape as :meth:`EngineRunner.pull` — because on a
    ``volume_unit`` provider each concurrent job would carry a whole volume's working set
    (~30 GB measured), and a scrub across cold frames would otherwise start one per pool
    thread."""

    def __init__(self, runner: "EngineRunner", gen: int, epoch: int, provider: Any,
                 node_id: str, coords, channels, axes, pin: Optional[Pin]) -> None:
        super().__init__()
        self._r = runner
        self._gen = gen
        self._epoch = epoch
        self._prov = provider
        self._node_id = node_id
        self._coords = coords
        self._channels = channels
        self._axes = axes
        self._pin = pin

    def run(self) -> None:  # worker thread
        r = self._r
        t0 = time.perf_counter()
        packet = (self._gen, self._epoch, self._node_id, self._coords, self._channels,
                  self._pin)
        planes = axes = err = None
        try:
            if self._gen == r._decode_gen:        # else: superseded before it ever started
                planes = r._decode_planes(self._prov, self._node_id, self._coords,
                                          self._channels, self._axes, pin=self._pin)
                axes = self._axes
        except Exception:  # noqa: BLE001 — full trace to the GUI, never a dead thread
            err = traceback.format_exc()
        # ALWAYS delivered, on every path: the packet is what clears `_decode_busy` and
        # releases the pending slot, so swallowing it would wedge the fast path for good.
        r._planes_done.emit((*packet, planes, axes, time.perf_counter() - t0, err))


class _PrefetchJob(QRunnable):
    """Decodes a list of adjacent-frame planes off the held viewer provider into the
    shared :class:`PlaneCache`, on a pool thread. A stale generation (the cursor moved
    on) short-circuits the remaining reads, so a fast scrub never backs up the pool.

    Never runs under the solo-frame scope: the held provider has ``t == 1`` there, so
    :meth:`EngineRunner.prefetch` returns before building any job — the adjacent frames
    are simply not in this pull's dataset."""

    def __init__(self, runner: "EngineRunner", gen: int,
                 jobs: List[Tuple[tuple, int, int, int, int]]) -> None:
        super().__init__()
        self._r = runner
        self._gen = gen
        self._jobs = jobs

    def run(self) -> None:  # worker thread
        r = self._r
        prov = r._viewer_provider
        if prov is None:
            return
        for (key, m, t, z, ch) in self._jobs:
            if self._gen != r._prefetch_gen:
                return                       # superseded by a newer cursor position
            if r._planes.get(key) is not None:
                continue
            try:
                # No decode lock (V2.14). It used to serialize every provider read because
                # the engine's TileCache was unsynchronized; the cache carries its own lock
                # now, so holding one here only forced the prefetch pool to decode one frame
                # at a time — the opposite of what a prefetcher is for.
                # The cap rides in the key (V2.23) — it must, because this writes planes the
                # displayed-frame path then reads, and a prefetcher decimating to a different
                # size would have it serve frames of the wrong shape.
                arr, _lv = render_plane_native(prov, m, t, z, ch, max_dim=int(key[-1]),
                                               as_dtype=r._viewer_dtype)
                r._planes.put(key, arr)
            except Exception:                # noqa: BLE001 — prefetch is best-effort
                return


class _PreloadJob(QRunnable):
    """Decode a share of the series into the :class:`PlaneCache` so playback can run off it.

    Deliberately NOT a :class:`_PrefetchJob` with a longer list, for two reasons that are the
    whole point of the class:

    * **it has its own cancellation generation.** The prefetcher's is bumped by every cursor
      move, and playback moves the cursor — so a preload sharing it died on the first frame
      advance and playback went back to decoding each frame as it arrived. That is precisely
      the "it loads every time" report (2026-08-05). A preload is cancelled by an edit, a node
      change, a new preload, or stopping playback; never by the cursor it exists to serve.
    * **several run at once.** Measured on the WellA3 mosaic with a cold tile cache, one
      whole-canvas stitch is 1.03 s and four in parallel are 0.51 s each — ~2x, tailing off
      past four (0.43 s at eight), because the source-tile reads parallelize but the paste
      contends. So the fan-out is capped low, which also leaves pool threads for the frame
      being *displayed*: starving that to prefetch the future would be exactly backwards.

    Every plane decoded ticks the runner, so the window can show real progress and start
    playback when the series is actually ready rather than hoping."""

    def __init__(self, runner: "EngineRunner", gen: int,
                 jobs: List[Tuple[tuple, int, int, int, int]]) -> None:
        super().__init__()
        self._r, self._gen, self._jobs = runner, gen, jobs

    def run(self) -> None:  # worker thread
        r = self._r
        prov = r._viewer_provider
        for (key, m, t, z, ch) in self._jobs:
            if self._gen != r._preload_gen or prov is None:
                return                       # cancelled: an edit, a stop, or a newer preload
            if r._planes.get(key) is None:
                try:
                    arr, _lv = render_plane_native(prov, m, t, z, ch,
                                                   max_dim=int(key[-1]),
                                                   as_dtype=r._viewer_dtype)
                    r._planes.put(key, arr)
                except Exception:            # noqa: BLE001 — one unreadable frame must not
                    pass                     # abandon the rest of the series
            try:
                r._preload_tick.emit((self._gen, 1))
            except RuntimeError:
                return    # the runner was destroyed under us (app teardown): stop, quietly


class _DetailJob(QRunnable):
    """Read the visible rect of the viewed node at the finest level that fits the budget,
    off the GUI thread, and hand it back through :attr:`EngineRunner.detail_ready`.

    This is the second half of "the display plane is capped at
    :data:`MAX_DISPLAY_DIM`": the overview answers *where am I*, and this answers *what is
    actually there*. Without it a 13106² mosaic could only ever be seen at 1/4 scale,
    because the cap is applied once and zooming is a pure view transform over that texture.

    Off the GUI thread for the same reason :class:`_DecodeJob` is: on a lazy provider a
    miss RUNS THE NODE. A detail read of a stitched mosaic is a windowed stitch
    (``MultiViewProvider`` stitches only the tiles the window touches), which is hundreds
    of milliseconds — an eternity to block a repaint on.

    A stale generation short-circuits: the user is still zooming, and the rect we were
    asked for is no longer the one on screen."""

    def __init__(self, runner: "EngineRunner", gen: int, node_id: str, prov: Any,
                 coords: tuple, channels: Sequence[int], axes: Any,
                 rect01: Tuple[float, float, float, float], budget: int) -> None:
        super().__init__()
        self._r, self._gen, self._node_id = runner, gen, node_id
        self._prov, self._coords, self._channels = prov, coords, list(channels)
        self._axes, self._rect01, self._budget = axes, rect01, int(budget)

    def run(self) -> None:                                    # worker thread
        r = self._r
        if self._gen != r._detail_gen:
            return
        try:
            planes, rect01 = r.detail_planes(
                self._prov, self._node_id, self._coords, self._channels, self._axes,
                self._rect01, self._budget)
        except Exception:                    # noqa: BLE001 — detail is best-effort; the
            return                           # overview is already on screen and correct
        if self._gen == r._detail_gen and planes:
            r._detail_done.emit((self._gen, self._node_id, planes, rect01))


class EngineRunner(QObject):
    """Submit pulls; receive results on the GUI thread; drop stale epochs."""

    started = Signal(str)                        # node_id
    finished = Signal(str, object, object, object, float)   # id, payload, plane, axes, s
    plane_ready = Signal(str, object, object, float)   # id, planes, axes, s (fast path)
    failed = Signal(str, str)                    # node_id, traceback
    source_resolved = Signal(str, object)        # node_id, MetaEnvelope (G8 re-seed)
    #: the node set this pull may touch (the pulled node's ancestor closure), emitted on
    #: the GUI thread at submit so every participating card can show "queued" before any
    #: work starts: ``(target_node_id, [node_id, …])``.
    plan = Signal(str, object)
    #: per-node run progress, forwarded from the engine observer onto the GUI thread:
    #: ``(event, node_id, info)`` — see :data:`nodegraph.engine.Observer`, plus the
    #: runner-level ``"decode"`` event (the viewed node's planes are being read).
    node_progress = Signal(str, str, object)
    #: a Dock finished baking: ``(node_id, spec)`` where ``spec`` carries the request
    #: plus the resulting ``manifest``/``bytes``. Emitted on the GUI thread and — unlike
    #: a pull result — **never dropped as stale**, because the checkpoint is already on
    #: disk and the document has to record it or the bake is orphaned.
    baked = Signal(str, object)
    #: a viewport detail patch is ready: ``(node_id, {channel: plane}, (x0,y0,x1,y1))``
    #: with the rect in NORMALIZED image coordinates, so it is independent of both the
    #: patch's own resolution and the overview's.
    detail_ready = Signal(str, object, object)
    #: a playback preload advanced: ``(node_id, done, total)``. Emitted on the GUI thread so the
    #: window can show progress and — the point — start the play timer only once the frames are
    #: actually resident, which is what makes playback of a computed chain smooth instead of
    #: stuttering at decode speed.
    preload_progress = Signal(str, int, int)
    #: the preload finished or was cancelled: ``(node_id, completed)``.
    preload_finished = Signal(str, bool)
    #: a source card's own ingest started / ended (V2.21, :meth:`ingest_source`):
    #: ``(node_id)`` and ``(node_id, seconds, traceback-or-None)``. Separate from
    #: ``started``/``finished``, which belong to the single pull slot — several ingests run
    #: at once, none of them is "the run", and none produces a payload to view.
    ingest_started = Signal(str)
    ingest_finished = Signal(str, float, object)

    _done = Signal(object)                       # internal cross-thread delivery
    _detail_done = Signal(object)                # internal: _DetailJob → GUI thread
    _preload_tick = Signal(object)               # internal: _PreloadJob → GUI thread
    _progress = Signal(object)                   # internal cross-thread progress delivery
    _planes_done = Signal(object)                # internal: _DecodeJob → GUI thread
    _ingest_done = Signal(object)                # internal: _IngestJob → GUI thread

    def __init__(self, document) -> None:
        super().__init__()
        self.document = document
        #: the shared pool, for the jobs that only READ providers (display decode, prefetch,
        #: viewport detail). Deliberately NOT the pull: see :class:`_PullThread`.
        self._pool = QThreadPool.globalInstance()
        #: the single Python-owned thread every node compute runs on (:class:`_PullThread`).
        self._pull_thread = _PullThread()
        # Persists across engine rebuilds — so it's the memory hazard V2.04 flagged: an
        # eager full-raster node in a high-T zone would otherwise pin one raster per
        # iteration forever. Cap it with a byte-budget LRU (Memo GC); eviction only costs
        # a recompute (correctness-safe). Budget matches the engine's default tile cache.
        self._memo = Memo(budget_bytes=MEMO_BUDGET_BYTES)
        # Persists across engine rebuilds for the SAME reason the memo does, and it has to:
        # a streaming provider holds its cache by weakref, and the memo hands out lazy
        # Datasets built by an earlier engine. Let each engine own its own TileCache and
        # every memo-hit lazy chain reads through `_NO_CACHE` after the first rebuild —
        # measured on the WellA3 640 series (12×16×210×1024², 3D Deconvolve): four z-plane
        # reads went from 1 whole-volume Richardson–Lucy to 4, i.e. a ~130 s recompute per
        # z-step, forever, because the document revision had bumped once. The tile key is
        # content-addressed (provider fingerprint + grid), so sharing is safe.
        self._tiles = TileCache(tile_cache_bytes())
        self._engine: Optional[Engine] = None
        self._engine_rev = -1
        self._providers: Dict[Any, Tuple[Any, MetaEnvelope]] = {}
        self._channel_display: Dict[Any, Dict[str, Any]] = {}   # source key → Viewer meta
        # ── per-source ingest, concurrent and outside the pull slot (V2.21) ───────
        self._ingest_pool = QThreadPool()
        self._ingest_pool.setMaxThreadCount(ingest_workers())
        #: node_id → source key, for every card whose ingest is running or queued. GUI
        #: thread only. Several nodes can name the SAME file; one job serves them all, and
        #: they all finish together (see :meth:`_deliver_ingest`).
        self._ingesting: Dict[str, Any] = {}
        #: source key → lock. The one piece of mutual exclusion the whole design needs:
        #: exactly one thread may ingest a given file, because two writers on one ``.b2nd``
        #: store produce the torn store `ingest.verify_store` exists to detect. Holding it
        #: also means a *pull* that reaches a file being ingested in the background blocks
        #: until it lands and then takes the finished provider — rather than starting a
        #: duplicate ingest of a file that is already half-written.
        self._src_locks: Dict[Any, threading.Lock] = {}
        self._src_locks_guard = threading.Lock()
        self._epoch = 0
        self._busy = False
        self._pending: Optional[Tuple[str, Any, Any]] = None
        self._announced: Dict[str, Any] = {}     # node_id → last announced source key
        self._node_source_key: Dict[str, Any] = {}
        #: (node_id, document revision) → EngineRunner.raw_source result — the hover
        #: readout asks per mouse move, and the answer costs a run-graph build.
        self._raw_src: Dict[Tuple[str, int], Any] = {}
        # ── decoded-plane fast path (scrub/play without a graph re-pull) ──────────
        self._planes = PlaneCache()              # (node,rev,m,t,z,c) → native plane
        # (V2.14) The former `_decode_lock` is gone: it serialized every provider read
        # because the engine's TileCache was unsynchronized. The cache locks itself now, so
        # the prefetch pool decodes frames concurrently — which is the point of a
        # prefetcher — and a plane read fans its tiles out across cores.
        self._viewer_provider: Any = None        # held image provider of the viewed node
        self._viewer_node: Optional[str] = None
        #: resolved secondary chain for the viewed overlay node (see `_resolve_overlay`).
        #: Written on the worker, read on the worker — same discipline as the held provider.
        self._overlay_ctx: Optional[Dict[str, Any]] = None
        self._viewer_axes: Any = None
        self._viewer_rev: int = -1               # document.revision the provider belongs to
        self._viewer_pin: Optional[Pin] = None   # the scope it was pulled for
        #: dtype the viewed node's display planes may be narrowed to, or ``None`` — set from
        #: the payload's declared bit depth (see :func:`_display_dtype`).
        self._viewer_dtype: Any = None
        #: what one texture axis may be, from the live surface (:meth:`set_display_limits`).
        self._texture_limit: int = DEFAULT_TEXTURE_LIMIT
        self._prefetch_gen = 0
        # ── playback preload (see `preload_series`) — its OWN generation, because the
        #    prefetcher's is bumped by the very cursor moves a preload exists to serve ──
        self._preload_gen = 0
        self._preload_node: Optional[str] = None
        self._preload_total = 0
        self._preload_done = 0
        # The displayed-plane decode is asynchronous when the plane is COLD (see
        # :class:`_DecodeJob`): one job in flight, one latest-wins pending request. A warm
        # plane never reaches this — it is served inline from the PlaneCache.
        self._decode_gen = 0
        self._decode_busy = False
        self._decode_pending: Optional[Tuple[str, Any, Any]] = None
        # ── viewport detail-on-demand (see :class:`_DetailJob`) ───────────────────
        #: bumped on every new request AND on every pull/scrub, so a patch that arrives
        #: for a rect (or a frame) the user has already left is dropped instead of being
        #: painted over the wrong place.
        self._detail_gen = 0
        # ── solo-frame (troubleshooting) scope ────────────────────────────────────
        self._solo = False
        # the Viewer's M/T/Z picks. Empty on M or T → that axis follows the cursor; empty
        # on Z → the whole volume (a 3D node needs one, so z is not cursor-scoped).
        self._sel_m: Tuple[int, ...] = ()
        self._sel_t: Tuple[int, ...] = ()
        self._sel_z: Tuple[int, ...] = ()
        self._seed_axes: Dict[str, AxisSizes] = {}   # what the live engine's meta was built on
        #: every meta seed the live engine has been given, ACCUMULATED across pulls at the
        #: same revision. A pull only resolves the sources its own closure reaches (V2.21),
        #: so `reseed_meta` has to be handed the union — passing just this pull's would
        #: strip the envelopes of every other source the engine already knows.
        self._meta_seeds: Dict[str, MetaEnvelope] = {}
        # ── per-node progress bookkeeping ─────────────────────────────────────────
        #: epoch → the in-flight bake request (see :meth:`bake`), consumed by `_deliver`
        self._bakes: Dict[int, Dict[str, Any]] = {}
        #: node_id → the Dataset a ``held`` dock is serving, and its envelope. Modelled on
        #: :attr:`_providers` (keyed by source path): a payload cache that must OUTLIVE every
        #: engine rebuild, because `_ensure_engine` builds a new Engine per document revision
        #: and re-derives its seeds on every pull — so anything held on the engine would be
        #: dropped by the next edit, which is exactly when a hold has to survive.
        #:
        #: A seed, not a memo entry, and that is load-bearing: `Memo._evict_to_budget` drops
        #: entries on recency alone and is documented as "always correctness-safe" precisely
        #: because dropping one only costs a recompute. For a held dock the upstream edge is
        #: CUT, so an eviction would not cost a recompute — it would lose the data.
        self._held: Dict[str, Dataset] = {}
        self._held_envs: Dict[str, MetaEnvelope] = {}
        #: set by :meth:`request_stop_bake`, polled by the writer at every block boundary.
        #: A plain bool rather than an Event: it is written by the GUI thread and read by the
        #: worker, and a torn read of a bool that only ever goes False→True cannot be wrong
        #: in a way that matters — the worst case is one extra block written before it stops.
        self._stop_bake: bool = False
        self._last_progress: Dict[str, float] = {}   # node_id → last delivery (perf time)
        self._last_frame: Dict[str, int] = {}        # node_id → last delivered frame step
        self._last_sweep: Dict[str, bool] = {}       # node_id → was the sub bar sweeping?
        # Iterate nodes whose EVERY iteration must be minted on the next pull, rather than
        # only the one their preserve mode reads (:meth:`set_sweep_all`). Held on the runner
        # and not in the document because it is a property of the next RUN, not of the
        # graph: it must never enter a saved file, and a re-pull for any other reason has
        # to go back to the cheap single-clone form on its own.
        self._sweep_all: frozenset = frozenset()
        self._done.connect(self._deliver)
        self._progress.connect(self._deliver_progress)
        self._planes_done.connect(self._deliver_planes)
        self._detail_done.connect(self._deliver_detail)
        self._preload_tick.connect(self._deliver_preload_tick)
        self._ingest_done.connect(self._deliver_ingest)
        document.on_change(self._prune)

    # ── per-source ingest (V2.21) ─────────────────────────────────────────────
    def source_state(self, node_id: str) -> str:
        """What :meth:`ingest_source` would do for ``node_id``, without doing it.

        One of ``"ready"`` (the file is already ingested — a pull of this card is
        immediate), ``"running"`` (its ingest is in flight or queued), ``"cold"`` (it would
        start one), ``"synthetic"`` (an empty path — the demo source needs no ingest),
        ``"missing"`` (the path names no file) or ``"not-a-source"``."""
        rec = self.document.nodes.get(node_id)
        if rec is None or rec.op_key != LOAD_OP:
            return "not-a-source"
        if node_id in self._ingesting:
            return "running"
        path = _clean_source_path(rec.params.get("path", ""))
        if not path:
            return "synthetic"
        if source_key(path) in self._providers:
            return "ready"
        return "cold" if os.path.isfile(path) else "missing"

    def ingest_source(self, node_id: str) -> str:
        """Ingest the file an ``io.load`` card names — **now, concurrently, and without
        taking the pull slot**. Returns the same vocabulary as :meth:`source_state`, plus
        ``"started"`` (a job was queued for it) and ``"joined"`` (another card already has
        this same file in flight; this one will finish with it).

        This is what a double-click on a source card does. The alternative — pulling it —
        works, but it runs the ingest *inside* the single latest-wins pull slot, so a second
        file cannot start until the first has finished, the app's one worker is occupied for
        the whole multi-minute decode, and the pull that finally arrives is superseded if
        you touched anything meanwhile. Ingest is the wrong shape for that slot: it is
        per-file, order-free, and its result (a ``.b2nd`` store + a cached provider) is
        worth keeping no matter what the graph does next.

        Idempotent: a file already in :attr:`_providers` reports ``"ready"`` and starts
        nothing, and two cards naming the same path share one job."""
        state = self.source_state(node_id)
        if state != "cold":
            return state
        rec = self.document.nodes[node_id]
        path = _clean_source_path(rec.params.get("path", ""))
        key = source_key(path)
        joined = key in set(self._ingesting.values())
        self._ingesting[node_id] = key
        self.ingest_started.emit(node_id)
        if joined:
            return "joined"
        self._ingest_pool.start(_IngestJob(self, node_id, path, key))
        return "started"

    def ingest_all_sources(self) -> Dict[str, str]:
        """Start (or report) the ingest of every ``io.load`` in the document — ``{node_id:
        state}``, the per-card result of :meth:`ingest_source`. The whole point of a
        multi-file load: drop five ND2s on the canvas and put them all on disk in one go,
        :func:`ingest_workers` at a time."""
        return {rec.id: self.ingest_source(rec.id)
                for rec in self.document.nodes.values() if rec.op_key == LOAD_OP}

    def ingesting(self) -> Tuple[str, ...]:
        """The node ids whose ingest is in flight or queued (GUI thread)."""
        return tuple(self._ingesting)

    @property
    def busy(self) -> bool:
        """Whether the single **pull** slot is occupied. An ingest never occupies it, so
        this stays False while a folder's worth of files is being written to disk."""
        return self._busy

    @property
    def baking(self) -> bool:
        """Whether the job in flight is a **bake that writes** — so a Stop is offered only
        when there is something to stop.

        A ``hold`` is excluded on purpose: it writes nothing and finishes at the speed of the
        pull it needs, so there is no long write for a Stop to interrupt, and offering one
        would imply a partial artifact that cannot exist."""
        return any(not spec.get("hold") for spec in self._bakes.values())

    def _deliver_ingest(self, packet) -> None:       # GUI thread (queued)
        node_id, key, seconds, err = packet
        # Every card naming this same file finishes with it — one job served them all.
        done = [nid for nid, k in self._ingesting.items() if k == key] or [node_id]
        for nid in done:
            self._ingesting.pop(nid, None)
        if err is None and not self._busy:
            # G8 live re-seed, as a pull's `_deliver` does it: the ingest resolved this
            # source's real envelope, so every metadata-intelligent param downstream
            # re-derives from the file rather than from the load-time metadata read.
            #
            # Two guards, both about `set_meta_seed`'s side effect — it bumps the document
            # revision, and the window answers a revision bump with `invalidate()`:
            #
            # * NOT while a pull is in flight. Invalidating there would drop the result of
            #   a pull the user is waiting on, for a display-metadata refresh. Nothing is
            #   lost by waiting: `_deliver` calls `_fresh_envs` too, and it announces every
            #   resolved source, not only the ones that pull touched.
            # * only when the envelope actually CHANGED. In the normal flow the document
            #   was already seeded from this very file at load time (`read_meta_only`), so
            #   the answer is identical and the bump would cost a re-pull for nothing.
            for nid, env in self._fresh_envs():
                if self.document.meta_seeds.get(nid) != env:
                    self.document.set_meta_seed(nid, env)
        for nid in done:
            self.ingest_finished.emit(nid, float(seconds), err)

    # ── display resolution policy (V2.23) ─────────────────────────────────────
    #
    # One question, asked in one place: is this frame shown WHOLE at full resolution, or off
    # the pyramid with a detail patch filling in where you look? Nikon asks it once per image
    # against `MaxMemoryImageSize`; the same decision, against the same 25%-of-RAM default,
    # lives here — see `display_cap`.
    #
    # The answer MUST be a pure function of (axes, channel count) for a given machine, because
    # it decides the SHAPE of a plane that goes into the PlaneCache under a key several other
    # readers write to. Two writers disagreeing about the cap would serve each other's planes
    # at the wrong size. The cap is in the cache key as well, so even a disagreement is a miss
    # rather than a wrong picture.

    def set_display_limits(self, texture_px: int) -> None:
        """Tell the runner what the live surface can actually upload (its
        ``GL_MAX_TEXTURE_SIZE``, or :data:`MAX_DISPLAY_DIM` for the CPU backend, which has no
        texture but does have a QImage the size of the frame).

        Called when a surface comes up, and on a backend switch. Bumps nothing and clears
        nothing: the cap is part of every plane key, so planes decoded under the old limit are
        simply never looked up again."""
        px = max(int(MAX_DISPLAY_DIM), int(texture_px or 0))
        if px != self._texture_limit:
            self._texture_limit = px

    def display_dim(self, axes: Any, *, planes: int = 1, provider: Any = None) -> int:
        """The display cap for the viewed node's frames — full resolution when one frame of
        every shown channel is affordable in texture, in RAM, and to PRODUCE.

        ``provider`` decides the last of those and defaults to the held one; it is passed
        explicitly by :meth:`_decode_planes`, which runs on the worker DURING the pull that
        will later install it — reading the held one there would answer for the node being
        replaced, and the answer is folded into every plane key."""
        if axes is None:
            return MAX_DISPLAY_DIM
        prov = self._viewer_provider if provider is None else provider
        return display_cap(axes, texture_limit=self._texture_limit,
                           bytes_per_px=(2 if self._viewer_dtype is not None else 8),
                           planes=planes,
                           streaming=isinstance(prov, StreamProvider))

    # ── viewport detail-on-demand ─────────────────────────────────────────────
    def request_detail(self, node_id: str, coords, channels, rect01, budget: int) -> bool:
        """Ask for the visible rect of ``node_id`` at full detail. Returns whether a job
        was queued. GUI thread; the read itself runs on the pool (:class:`_DetailJob`).

        Latest-wins by generation rather than by a queue: while the user is still zooming,
        every intermediate rect is dead on arrival, and rendering them in order would just
        put the pool behind the cursor."""
        prov = self._viewer_provider
        if prov is None or node_id != self._viewer_node or self._viewer_axes is None:
            return False
        self._detail_gen += 1
        self._pool.start(_DetailJob(self, self._detail_gen, node_id, prov, coords,
                                    channels, self._viewer_axes, rect01, budget))
        return True

    def invalidate_detail(self) -> None:
        """Drop any in-flight detail patch — the frame, the node or the graph moved, so
        whatever is being read is about to describe something that is no longer shown."""
        self._detail_gen += 1

    def detail_planes(self, prov, node_id, coords, channels, axes, rect01, budget):
        """``({channel: plane}, rect01)`` for a normalized rect of ``node_id``.

        **Never call on the GUI thread** — same contract as :meth:`_decode_planes`, and for
        the same reason. Picks the FINEST pyramid level at which the requested rect fits
        ``budget`` px, reads just that rect, and area-fits the remainder; so the patch is
        the best detail the budget can hold, not the best the pyramid happens to offer.

        The returned rect is the one actually read — snapped out to whole pixels of the
        chosen level — because the caller has to place the patch on screen and a rect that
        disagrees with the pixels by half a texel shows up as a seam against the overview
        underneath it.

        The OVERLAY is composed onto the patch too, against the snapped rect. It has to be:
        the detail quad is drawn opaquely over the overview inside its own rect, so a patch
        that carried only the primary's channels *erased* the overlay wherever you zoomed in —
        the "it disappears when I zoom" report (2026-08-04). Composing it here is also what
        makes zooming show more of the secondary, since the composite is re-sampled from a
        window of the secondary at the patch's own resolution."""
        pin = self._viewer_pin
        # the SAME re-addressing the display planes get, so a detail patch under the
        # solo-frame scope reads the frame the overview is showing, not the global one
        m, t, z, cur_c = self._clamp_coords(self._payload_coords(coords, pin), axes)
        x0, y0, x1, y1 = rect01
        # clamp + require a non-degenerate rect (a fully zoomed-out view asks for none)
        x0, x1 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))
        y0, y1 = max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return {}, rect01
        want = (y0, y1, x0, x1)
        level, lax = _pick_window_level(prov, want, budget)
        ly0, ly1, lx0, lx1 = _snap_window(want, lax)
        out: Dict[int, np.ndarray] = {}
        for ch in (channels if channels else (cur_c,)):
            ch = int(ch)
            if ch >= int(axes.c):
                continue          # a composed OVERLAY channel: it is not in this provider
            ch = min(max(0, ch), axes.c - 1)
            if ch in out:
                continue
            arr = np.asarray(prov.get_region(level, m, t, z, ch, ly0, ly1, lx0, lx1))
            out[ch] = _fit_plane(arr, budget, as_dtype=self._viewer_dtype)
        # the rect the pixels REALLY cover, in normalized coords
        snapped = (lx0 / lax.x, ly0 / lax.y, lx1 / lax.x, ly1 / lax.y)
        if out and self._overlay_ctx is not None:
            ref = next(iter(out.values()))
            composed = self._compose_overlay(
                node_id, ref.shape[:2], m, t,
                _z_um_of(self._overlay_ctx["pri_md"], self._overlay_ctx["pri_axes"], m, z),
                # the SNAPPED rect, in the (fy0, fy1, fx0, fx1) order placement uses: the
                # patch's pixels cover that box and not the one that was requested
                region=(ly0 / lax.y, ly1 / lax.y, lx0 / lax.x, lx1 / lax.x))
            # the caller asked for a channel set; an overlay channel whose toggle is off is
            # not in it, and the patch must not put back what the overview leaves out
            want = {int(ch) for ch in (channels or ()) if int(ch) >= int(axes.c)}
            out.update({ch: pl for ch, pl in composed.items()
                        if not channels or ch in want})
        return out, snapped

    def _deliver_detail(self, packet) -> None:            # GUI thread
        gen, node_id, planes, rect01 = packet
        if gen == self._detail_gen:
            self.detail_ready.emit(node_id, planes, rect01)

    # ── public API (GUI thread) ────────────────────────────────────────────────
    def pull(self, node_id: str,
             coords: Optional[Tuple[int, int, int, int]] = None,
             channels: Optional[Tuple[int, ...]] = None) -> None:
        """Request a full engine pull (+ optional display planes for ``channels``).
        Establishes/refreshes the held viewer provider. Latest-wins while busy."""
        if node_id not in self.document.nodes:
            return
        if self._busy:
            self._pending = (node_id, coords, channels)
            return
        self._submit(node_id, coords, channels)

    def request_plane(self, node_id: str,
                      coords: Optional[Tuple[int, int, int, int]] = None,
                      channels: Optional[Tuple[int, ...]] = None) -> None:
        """Coords-only request. When the viewed node + document revision are unchanged
        (only the M/T/Z cursor or the active-channel set moved), bypass the graph
        snapshot + ``engine.pull`` entirely and serve the plane straight from the
        :class:`PlaneCache` (decoding a miss synchronously — a single decimated read),
        then warm adjacent frames. Otherwise fall back to a full :meth:`pull`, which
        re-establishes the provider handle for this (node, revision).

        Under the solo-frame scope the held payload contains only the scoped frames, so
        moving the M/T cursor is normally a change of *what was computed*, not of what is
        displayed: the scope must move with it, which means a real re-pull. The exception
        is a cursor move *within* a multi-frame selection — the pin is unchanged there, so
        it stays on the fast path. Z and channel moves always do, and between them that is
        most of the interactive scrubbing a troubleshooting session does."""
        if node_id not in self.document.nodes:
            return
        if (coords is not None and node_id == self._viewer_node
                and self._viewer_provider is not None
                and self.document.revision == self._viewer_rev
                and self._pin_for(coords) == self._viewer_pin):
            self._serve_from_cache(node_id, coords, channels)
            return
        self.pull(node_id, coords, channels)

    def invalidate(self) -> None:
        """Drop any in-flight result (edits during a run) and drop the held provider —
        a graph edit may change pixels, so the next request must re-pull through the
        engine and the decoded-plane cache is no longer valid."""
        self._epoch += 1
        self._viewer_rev = -1
        self._viewer_pin = None
        self._prefetch_gen += 1
        # an in-flight display decode is reading pixels the edit may have changed: retire
        # its generation so the result is dropped rather than painted over the new graph
        self._decode_gen += 1
        self._decode_pending = None
        # …and the same for an in-flight viewport detail patch, which is reading through
        # the provider this call is about to drop
        self.invalidate_detail()
        # A preload is reading that provider too, and its planes are about to be wrong.
        self.cancel_preload()
        # the composed overlay belongs to the graph that produced it — an edit can change
        # the placement, the pairing or the secondary chain entirely
        self._overlay_ctx = None
        self._planes.clear()

    # ── the solo-frame (troubleshooting) scope ─────────────────────────────────
    @property
    def solo_frame(self) -> bool:
        return self._solo

    def set_solo_frame(self, on: bool) -> None:
        """Turn the **solo-frame scope** on/off: while on, every pull is evaluated over
        what the Viewer's M/T/Z strips scope to instead of the whole series — the frames
        and planes picked on them, or just the frame the cursor is on when nothing is
        picked (:meth:`set_frame_selection`).

        Implemented by cutting the *source seeds* — each resolved provider is wrapped in a
        :class:`~nodegraph.provider.FrameSubsetProvider` and its seed envelope shortens to
        match (:meth:`_ensure_engine`). Three properties follow, and they are why this is a
        seed concern and not a graph edit:

        * **Nothing in the catalog changes.** Every node keeps looping over "all frames" —
          there are now few, each of them shorter in z. The eager per-unit computes
          (Segmentation, Measure, Detection, Tracking) drop from M·T units to the picked
          count, *and* their output rasters allocate for those frames and planes only,
          which is usually the larger win of the two.
        * **The user's graph is untouched**, so the canvas, the run plan, the per-node
          progress and the saved file are all exactly what they were. A pull under this
          scope reports on the same cards as a full one.
        * **The memo stays honest.** The picks ride in the provider's ``fingerprint`` →
          ``version`` → the source node's ``__seed_version__`` → every downstream
          ``recipe_hash``. A scoped result can therefore never be served for a different
          selection or for the full series, and revisiting an already-computed selection is
          a memo hit — which is what makes flipping between two frames instant.

        The honest cost is proportional to how little is scoped: with one frame picked a
        tracker links nothing and a time reduction reduces one sample. Picking several T
        frames is the answer to exactly that — the tracker then sees a real (if sparse)
        series. Sparse, though, is the caveat, and it applies to z the same way: indices
        the picks skip are simply *absent*, not empty, so linking distances, per-frame
        rates and any 3D measurement are computed across the **picked** neighbours rather
        than the acquired ones (``dt_s``/``z_step_um`` still describe the source, which is
        exact for a contiguous pick and an approximation for a strided one). It is scoped
        for checking parameters, not for producing results.
        """
        on = bool(on)
        if on == self._solo:
            return
        self._solo = on
        # The held provider + decoded planes belong to the other scope. The MEMO does not
        # need clearing: entries are keyed by the pin, so both scopes coexist in it.
        self.invalidate()

    def set_frame_selection(self, ms: Iterable[int] = (), ts: Iterable[int] = (),
                            zs: Iterable[int] = ()) -> None:
        """The indices the Viewer has picked on its M, T and Z strips.

        The three axes are independent, so what runs is their **cross product**: two
        positions, three timepoints and five planes scope the run to 2·3 frames of 5
        planes each. An empty M or T means *nothing picked there*, and :meth:`_pin_for`
        fills it from the display cursor — so untouched strips reproduce the original
        one-frame scope exactly. An empty Z means *the whole volume*, not "the plane the
        cursor is on": z is inside a frame and a 3D node needs all of it unless the user
        deliberately says otherwise.

        Held state (the viewer provider, the pull in flight) belongs to the old selection,
        hence the :meth:`invalidate`; it is skipped while the scope is off, where the picks
        change nothing about what runs."""
        sel = tuple(tuple(sorted({int(i) for i in axis})) for axis in (ms, ts, zs))
        if sel == (self._sel_m, self._sel_t, self._sel_z):
            return
        self._sel_m, self._sel_t, self._sel_z = sel
        if self._solo:
            self.invalidate()

    @property
    def frame_selection(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return (self._sel_m, self._sel_t, self._sel_z)

    def _pin_for(self, coords: Optional[Tuple[int, int, int, int]]) -> Optional[Pin]:
        """The ``(ms, ts, zs)`` a request scopes to — ``None`` when the whole series runs.
        The picks where the user made them; the display cursor (exactly "the frame the
        user is on") on whichever *frame* axis they did not; and the whole volume when z
        is unpicked."""
        if not self._solo or coords is None:
            return None
        return (self._sel_m or (int(coords[0]),), self._sel_t or (int(coords[1]),),
                self._sel_z or None)

    @staticmethod
    def _payload_coords(coords, pin: Optional[Pin]):
        """``coords`` (a GLOBAL display cursor) re-addressed into a scoped payload, which
        holds only the picked frames and planes. Off the scope this is the identity; under
        it a cursor parked on an unpicked index resolves to the nearest picked one
        (:func:`~nodegraph.provider.subset_index`) rather than off the end of the payload,
        and an axis with no picks passes through.

        Called **once** per request, at the boundary where a global cursor first meets a
        payload — re-applying it to an already-mapped index would remap the index."""
        if pin is None or coords is None:
            return coords
        ms, ts, zs = pin
        m, t, z, c = coords
        return (subset_index(ms, m), subset_index(ts, t), subset_index(zs or (), z), c)

    # ── decoded-plane fast path ────────────────────────────────────────────────
    def _clamp_coords(self, coords, axes):
        return tuple(min(max(0, int(v)), s - 1) for v, s in
                     zip(coords, (axes.m, axes.t, axes.z, axes.c)))

    def _plane_addrs(self, node_id, coords, channels, axes,
                     *, pin: Optional[Pin] = None, provider: Any = None
                     ) -> List[Tuple[tuple, int, int, int, int]]:
        """``(key, m, t, z, ch)`` per requested channel — the :class:`PlaneCache` addresses
        one display update needs, de-duplicated. Shared by the cache probe and the decode so
        the two can never disagree about a key.

        The cache key omits the revision — :meth:`invalidate` clears the whole cache on any
        edit, so a stale plane can never be served across a graph change.

        ``pin`` (the solo-frame scope) does two jobs here. It re-addresses the cursor into
        the short payload (:meth:`_payload_coords`), and it IS part of the key: under the
        scope a payload addresses its first frame as ``t == 0``, so ten different scoped
        selections would otherwise all write ``(node, 0, 0, z, ch)`` and serve each
        other's pixels.

        A composed OVERLAY channel (an index at or above the payload's own channel count)
        gets a cache slot of its own — the index cannot collide with a real one, and
        :meth:`_decode_planes` fills it. It used to be *clamped* into the primary's range,
        which meant a warm frame was served without the overlay at all: scrubbing then blinked
        the secondary off on every frame that happened to be decoded already, and back on for
        every frame that was not."""
        m, t, z, c = self._clamp_coords(self._payload_coords(coords, pin), axes)
        ovl = self.overlay_channels(node_id)
        wanted = [int(v) for v in (channels if channels else (c,))]
        # The display CAP is part of the key (V2.23). It decides the plane's shape, several
        # readers write these keys (the decode, the prefetcher, the raw probe), and the limit it
        # comes from can arrive late — a surface coming up raises it. Keying on it turns any
        # disagreement into a cache miss instead of a plane served at the wrong size.
        cap = self.display_dim(axes, planes=max(1, len(set(wanted))), provider=provider)
        out: List[Tuple[tuple, int, int, int, int]] = []
        seen: set = set()
        for ch in wanted:
            if ch >= int(axes.c):
                if ch not in ovl:
                    continue          # a stale index from a node that no longer overlays
            else:
                ch = min(max(0, ch), axes.c - 1)
            if ch in seen:
                continue
            seen.add(ch)
            out.append((_plane_key(node_id, pin, m, t, z, ch, cap), m, t, z, ch))
        return out

    def _cached_planes(self, node_id, coords, channels, axes,
                       *, pin: Optional[Pin] = None) -> Optional[Dict[int, np.ndarray]]:
        """The requested planes if EVERY one of them is already decoded, else ``None`` —
        a read-only probe, so the GUI thread can decide whether serving this frame is free
        before it commits to doing it there (:meth:`_serve_from_cache`)."""
        out: Dict[int, np.ndarray] = {}
        for key, _m, _t, _z, ch in self._plane_addrs(node_id, coords, channels, axes,
                                                    pin=pin):
            arr = self._planes.get(key)
            if arr is None:
                return None
            out[ch] = arr
        return out

    # ── overlay: the second chain, composed onto the display grid (V2.19) ──────
    #
    # `view.overlay` records WHERE the secondary goes and deliberately touches no pixels,
    # so the picture is assembled here, in the DISPLAY path, alongside the decimation to
    # MAX_DISPLAY_DIM. The composed plane is handed to the Viewer as an extra channel
    # index and never reaches a downstream node — the same contract as the viewport
    # detail patch (probe V3): detail is a display artefact.

    @staticmethod
    def overlay_chain(graph, node_id: str) -> List[Tuple[str, str]]:
        """``[(overlay node, its secondary node), …]`` from the base outward.

        N-way overlay is expressed by CHAINING — an Overlay whose primary is another
        Overlay — so drawing three sources means walking back down that spine and
        collecting every `secondary` edge on the way, not just the viewed node's own. The
        walk stops at the first non-Overlay primary, which is the base image.

        Guarded against revisiting a node so a malformed graph cannot spin here; the engine
        rejects cycles, but this runs on GUI-supplied structure during editing."""
        out: List[Tuple[str, str]] = []
        seen: set = set()
        cur: Optional[str] = node_id
        while cur and cur not in seen:
            seen.add(cur)
            node = graph.nodes.get(cur)
            if node is None:
                break
            pri = [e.src for e in graph.preds(cur) if e.dst_socket == "data"]
            if node.op_key == "view.overlay":
                sec = [e.src for e in graph.preds(cur) if e.dst_socket == "secondary"]
                if sec:
                    out.append((cur, sec[0]))
            else:
                # ANY node may declare an auxiliary Dataset input a viewer SOURCE
                # (`SocketSpec.view_source`, 2026-08-04). `analysis.voronoi`'s `areas` is the
                # first: the payload carries only the primary's image, so a graph whose seeds
                # and areas came from two channels could only ever show one of them. Same
                # compositing path as an Overlay from here on — the placement entry is
                # synthesized in `_resolve_overlay`, since these nodes stamp no recipe.
                from nodegraph.registry import NODES as _NODES
                _spec = _NODES.get(node.op_key)
                _vs = {s.name for s in getattr(_spec, "inputs", ())
                       if getattr(s, "view_source", False)} if _spec else set()
                if _vs:
                    for e in graph.preds(cur):
                        if e.dst_socket in _vs:
                            out.append((cur, e.src))
                            break        # one source per node, like `secondary`
            # Walk THROUGH a non-overlay node rather than stopping at it. The recipe rides
            # on `Dataset.metadata`, so it survives every node downstream of the overlay —
            # view an `Overlay -> Stitch` at the Stitch and the overlay is still logically
            # present, and stopping here is why it drew nothing (reported 2026-08-03).
            # The primary edge is the spine either way.
            cur = pri[0] if pri else None
        out.reverse()                      # base-first, the order they composite in
        return out

    def _resolve_overlay(self, engine, graph, node_id: str, payload):
        """Everything :meth:`_compose_overlay` needs, or ``None`` — worker thread.

        Every source in the chain is resolved, each paired with the recipe entry its own
        node stamped (matched by node id, not by position, so a partially-broken chain
        cannot shift one source's placement onto another's pixels).

        Channels are handed out in chain order and stop at the shader's sampler bank: the
        primary's own channels come first and must not be pushed out of the composite by a
        third overlay source. A source that does not fit is dropped whole rather than half
        drawn, and :meth:`overlay_note` says so.
        """
        from nodegraph.nodes import OVERLAY_KEY
        if not isinstance(payload, Dataset):
            return None
        recipe = payload.metadata.get(OVERLAY_KEY) or ()
        chain = self.overlay_chain(graph, node_id)
        if not chain:
            return None
        # A `view.overlay` STAMPS its plan; a `view_source` socket's node does not (and must
        # not — display config has no business in a payload the memo keys on). So the entry is
        # synthesized for those, and the absence of a recipe is no longer a reason to bail.
        by_node = {str(e.get("node")): dict(e) for e in recipe[1:] if isinstance(e, dict)}
        base_c = int(payload.axes.c)
        budget = max(0, GL_MAX_CHANNELS - base_c)
        sources: List[Dict[str, Any]] = []
        dropped = 0
        for ovl_id, sec_id in chain:
            entry = by_node.get(ovl_id)
            synthetic = entry is None
            if synthetic and not self._is_view_source_node(ovl_id):
                continue                   # an Overlay that did not stamp: nothing to place
            try:
                sec = engine.pull(sec_id)
            except Exception:  # noqa: BLE001 — a broken source must not kill the view
                continue
            if not isinstance(sec, Dataset) or sec.image is None:
                continue
            if synthetic:
                entry = self._view_source_entry(ovl_id, payload, sec)
                if entry is None:
                    continue
            # RE-PLAN against the geometry actually being viewed. A stamped plan is a
            # cache keyed on the primary's geometry AT THE OVERLAY NODE, and a Stitch /
            # Crop / Resample after the overlay changes exactly that — the stamped tiles
            # would place the secondary against fields that no longer exist (a stitch
            # collapses twelve of them into one). Re-planning is pure metadata arithmetic
            # and it is what makes the overlay follow its primary through anything that
            # keeps `origin_um` true.
            # ...but a SYNTHETIC entry was already planned against this very payload just
            # above, and `_replan` would re-read blend params off a node that has none —
            # picking up `overlay_entry`'s `flip_x=True` default, which is right for a
            # secondary from another acquisition and wrong for a sibling branch of one file.
            if not synthetic:
                entry = self._replan(ovl_id, entry, payload, sec)
            n = min(int(sec.axes.c), MAX_OVERLAY_CHANNELS)
            if n <= 0 or n > budget:
                dropped += 1
                continue
            sources.append({"entry": entry, "ovl_id": ovl_id,
                            # A `view_source` wire is not an OVERLAY, it is another CHANNEL of
                            # the same acquisition — "I don't want an overlay, I want the
                            # channel to be active" (2026-08-04). So it takes the socket's name
                            # on the channel strip instead of `ovl1:`, and full opacity instead
                            # of an Overlay node's 0.5: at half strength a second channel reads
                            # as a wash over the first rather than as itself.
                            "as_channel": synthetic,
                            "prefix": (self._view_source_socket(graph, ovl_id)
                                       if synthetic else ""),
                            "sec_md": dict(sec.metadata),
                            "sec_axes": sec.axes, "sec_provider": sec.image,
                            "base_c": base_c + (GL_MAX_CHANNELS - base_c - budget),
                            "n": n})
            budget -= n
        if not sources:
            return None
        return {"node": node_id, "sources": sources, "dropped": dropped,
                "pri_md": dict(payload.metadata), "pri_axes": payload.axes}

    def _compose_overlay(self, node_id: str, out_shape, m: int, t: int, z_um,
                         *, region=None) -> Dict[int, np.ndarray]:
        """``{channel_index: composed plane}`` for the overlay on ``node_id``, or ``{}``.

        ``region`` is a fractional ``(fy0, fy1, fx0, fx1)`` sub-rect of the primary's image
        that ``out_shape`` covers — ``None`` for the whole field (the overview), the patch's
        own rect when a zoomed viewport asked for detail. Both go through one function, so the
        overlay cannot be present at one zoom level and absent at another.

        **The secondary is read as a WINDOW, at the finest pyramid level that window fits in**
        (:func:`_window_read`). It used to be read as a whole plane decimated to
        ``MAX_DISPLAY_DIM`` — which is the source's own overview, and when the primary's field
        is a small part of a stitched secondary that is a handful of source pixels magnified
        over the whole display: "not the raw channel data" exactly. A window costs the same
        read or less and scales with the zoom, so magnifying the picture now resolves more of
        the secondary rather than more of the blur.

        Never raises: an overlay that cannot be drawn must cost the user their overlay, not
        their image."""
        from nodelab_v2.overlay_compose import (
            compose_secondary_plane, paired_t, secondary_z_index)
        ctx = self._overlay_ctx
        if not ctx or ctx["node"] != node_id:
            return {}
        # The composite is sampled onto `out_shape`, so reading the window any finer than that
        # would be thrown away by the nearest-neighbour map — and any coarser is the blur.
        budget = max(1, int(max(out_shape[0], out_shape[1])))
        out: Dict[int, np.ndarray] = {}
        for src in ctx.get("sources", ()):
            # Per SOURCE, so one source that cannot be placed at this frame (an unpaired
            # timepoint, an unreadable tile) costs only its own layer — the rest of the
            # chain still draws.
            try:
                entry = src["entry"]
                t_sec = paired_t(entry, t)
                if t_sec is None:
                    continue
                sec_ax = src["sec_axes"]
                tiles = dict((int(a), b) for a, b in (entry.get("tiles") or ()))
                hits = tiles.get(int(m)) or ()
                if not hits:
                    continue
                dz = float((entry.get("offset_um") or (0.0, 0.0, 0.0))[0])
                z_sec = secondary_z_index(src["sec_md"], sec_ax, int(hits[0][0]),
                                          z_um, dz=dz)
                prov = src["sec_provider"]
                for k in range(int(src["n"])):
                    def read_tile(j, want, _k=k, _p=prov, _t=t_sec, _z=z_sec):
                        try:
                            return _window_read(_p, int(j), int(_t), int(_z), _k,
                                                want, budget)
                        except Exception:  # noqa: BLE001 — one bad tile, not a crash
                            return None
                    composed = compose_secondary_plane(
                        entry, out_shape, ctx["pri_md"], ctx["pri_axes"], int(m),
                        src["sec_md"], sec_ax, read_tile, region=region)
                    if composed is not None:
                        out[int(src["base_c"]) + k] = composed
            except Exception:  # noqa: BLE001 — the image must survive a bad overlay
                continue
        return out

    def overlay_flicker_hz(self, node_id: str) -> float:
        """Blink rate for the overlay's flicker mode, or ``0`` when no source uses it.

        The FIRST flickering source decides the rate: two sources blinking out of phase is
        not a comparison of anything."""
        ctx = self._overlay_ctx
        if not ctx or ctx["node"] != node_id:
            return 0.0
        for src in ctx.get("sources", ()):
            if str(src["entry"].get("blend")) == "flicker":
                return max(0.1, min(30.0, self._look(src["ovl_id"], "flicker_hz", 2.0)))
        return 0.0

    def overlay_note(self, node_id: str) -> str:
        """The one-line placement readout for the status bar, or ``""``.

        The FIRST source's note leads (it is the one most people are checking), and a chain
        appends how many more there are — plus, loudly, any source that had to be dropped
        for want of a shader channel, because a silently missing layer is the one thing this
        readout exists to prevent."""
        ctx = self._overlay_ctx
        if not ctx or ctx["node"] != node_id:
            return ""
        srcs = ctx.get("sources", ())
        if not srcs:
            return ""
        note = str(srcs[0]["entry"].get("note") or "")
        if len(srcs) > 1:
            note += f"  (+{len(srcs) - 1} more source{'s' if len(srcs) > 2 else ''})"
        if ctx.get("dropped"):
            note += (f"  ·  {ctx['dropped']} source(s) NOT drawn — only "
                     f"{GL_MAX_CHANNELS} display channels exist")
        return note

    def overlay_channels(self, node_id: str) -> Dict[int, str]:
        """``{channel index: label}`` the overlay contributes, for the channel strip."""
        ctx = self._overlay_ctx
        if not ctx or ctx["node"] != node_id:
            return {}
        out: Dict[int, str] = {}
        for i, src in enumerate(ctx.get("sources", ())):
            names = src["sec_md"].get("channel_names") or []
            for k in range(int(src["n"])):
                label = str(names[k]) if k < len(names) and names[k] else str(k)
                # the source's ordinal is in the label, so three overlaid files are
                # tellable apart on the channel strip without opening the graph. A
                # `view_source` wire is named by its SOCKET instead — it is a channel of this
                # graph, not the nth overlaid file, and `ovl1:` misdescribes it.
                pre = str(src.get("prefix") or "")
                out[int(src["base_c"]) + k] = f"{pre}:{label}" if pre else f"ovl{i + 1}:{label}"
        return out

    #: `blend` Mode value → the shader's mode number. Kept here, next to the only reader,
    #: because it is a mapping between a SAVED graph's vocabulary and a shader detail: the
    #: node's Mode choices are a file contract and the integers are not, so an unknown name
    #: falls back to additive rather than raising — a graph written by a newer build must
    #: still open and draw.
    #: `flicker` maps to ADD deliberately: it is not a compositing rule at all but a
    #: blink comparator, so the shader draws the overlay normally and the Viewer's timer
    #: switches its opacity between full and zero. Keeping it out of `blend_one` means the
    #: shader has no notion of time, which it should not.
    BLEND_MODES = {"add": 0, "over": 1, "difference": 2, "checkerboard": 3,
                   "wipe": 4, "flicker": 0}

    @staticmethod
    def _view_source_socket(graph, node_id: str) -> str:
        """The name of the wired ``view_source`` socket on ``node_id`` (``""`` if none)."""
        try:
            from nodegraph.registry import NODES
            node = graph.nodes.get(node_id)
            spec = NODES.get(node.op_key) if node is not None else None
            names = {s.name for s in getattr(spec, "inputs", ())
                     if getattr(s, "view_source", False)}
            for e in graph.preds(node_id):
                if e.dst_socket in names:
                    return str(e.dst_socket)
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _is_view_source_node(self, node_id: str) -> bool:
        """Whether ``node_id``'s op declares any ``view_source`` Dataset input."""
        try:
            from nodegraph.registry import NODES
            node = self.document.nodes.get(node_id)
            spec = NODES.get(node.op_key) if node is not None else None
            return bool(spec) and any(getattr(s, "view_source", False)
                                      for s in spec.inputs)
        except Exception:  # noqa: BLE001 — a display question must never break a pull
            return False

    def _view_source_entry(self, ovl_id: str, payload, sec) -> Optional[Dict[str, Any]]:
        """A placement entry for a ``view_source`` socket — synthesized, not stamped.

        These nodes record nothing: display configuration in a payload would ride the memo
        key, and `view.overlay` exists precisely so that "how it looks" is a graph decision
        with its own node. So the plan is built here, through the SAME
        :func:`~nodegraph.placement.plan_placement` + ``overlay_entry`` pair a real Overlay
        uses — one builder, so a synthesized source and a real one cannot place differently.

        The settings are identity on purpose (``blend="add"``, no flip, no shift): a
        ``view_source`` wire is a sibling branch of the same acquisition, and the node has
        already refused it unless it addresses the *same voxels* (``_require_same_grid``).
        ``view.overlay``'s ``flip_x=True`` default is for a secondary from a different
        acquisition and would be wrong here.

        **Placement is BY INDEX, not by stage position** (``on_unplaceable="index"``). Stage
        placement is the right default for an Overlay, whose two sources may be different
        acquisitions — and it *refuses* without a stage log, which a sibling branch does not
        need and a TIFF never has. Here field m pairs with field m at scale 1.0, which is not
        a guess: the node already proved the two address the same voxels. The three
        "alignment is NOT verified" warnings that path emits are therefore untrue of this
        one and are replaced, or they would appear on the node card as a problem."""
        try:
            from dataclasses import replace as _dc_replace
            from nodegraph.catalog.view.overlay import overlay_entry
            from nodegraph.placement import plan_placement
            from nodegraph.nodes import SAMPLING_KEY
            if payload is None or sec is None:
                return None
            plan = plan_placement(
                payload.metadata, payload.axes, sec.metadata, sec.axes,
                t_shift=0, offset_um=(0.0, 0.0, 0.0),
                dst_sampling=tuple(payload.metadata.get(SAMPLING_KEY, ())),
                src_sampling=tuple(sec.metadata.get(SAMPLING_KEY, ())),
                on_unplaceable="index")
            if plan is None or not getattr(plan, "ok", True):
                return None
            if plan.placed_by == "index":
                plan = _dc_replace(plan, warnings=(
                    "placed field-for-field: this branch was verified to address the same "
                    "voxels as the primary, so no stage alignment is needed",))
            return overlay_entry(ovl_id, plan,
                                 {"blend": "add", "flip_x": False, "flip_y": False,
                                  "t_shift": 0})
        except Exception:  # noqa: BLE001 — an unplaceable source costs its layer, not the view
            return None

    def _replan(self, ovl_id: str, stamped: Dict[str, Any], payload, sec) -> Dict[str, Any]:
        """The recipe entry re-resolved against the VIEWED payload's geometry.

        Settings come from the document (the same live-truth route the presentation params
        take), placement from :func:`nodegraph.placement.plan_placement`, and the entry from
        the node's own :func:`overlay_entry` — one builder, so the display and the stamped
        record cannot drift. Falls back to the stamped entry if anything is missing, which
        is the pre-existing behaviour and correct whenever nothing downstream moved."""
        try:
            from nodegraph.catalog.view.overlay import overlay_entry
            from nodegraph.placement import plan_placement
            from nodegraph.nodes import SAMPLING_KEY
            node = self.document.nodes.get(ovl_id)
            if node is None or payload is None or sec is None:
                return stamped
            prm, mds = dict(node.params or {}), dict(node.modes or {})
            offset = (float(prm.get("offset_z", 0.0)), float(prm.get("offset_y", 0.0)),
                      float(prm.get("offset_x", 0.0)))
            plan = plan_placement(
                payload.metadata, payload.axes, sec.metadata, sec.axes,
                t_shift=int(prm.get("t_shift", 0)), offset_um=offset,
                dst_sampling=tuple(payload.metadata.get(SAMPLING_KEY, ())),
                src_sampling=tuple(sec.metadata.get(SAMPLING_KEY, ())),
                min_coverage=float(prm.get("min_coverage", 0.0)),
                on_unplaceable=mds.get("unplaceable", "refuse"))
            if not plan.ok:
                return stamped        # keep the record; the pull itself already refused
            # The SAME handedness derivation the compute makes — an already-stitched secondary
            # is in stage coordinates, and a display path that flipped it while the compute did
            # not would draw the overlay 456 px from where a bake put it.
            from nodegraph.catalog._shared.placement_entry import handedness_for
            fx, fy, _w = handedness_for(sec, prm.get("flip_x", True),
                                        prm.get("flip_y", False))
            return overlay_entry(ovl_id, plan, {
                "blend": mds.get("blend", "add"), "flip_x": fx, "flip_y": fy,
                "t_shift": int(prm.get("t_shift", 0)), "offset_um": offset})
        except Exception:      # noqa: BLE001 — a re-plan failure must not cost the image
            return stamped

    def _look(self, ovl_id: str, name: str, default: float) -> float:
        """A PRESENTATION param, read live from the document rather than from the payload.

        This is the other half of `SocketSpec.presentation`. Those params are excluded from
        the recipe hash so moving one cannot invalidate a memo entry — which means the
        payload cannot carry them either, or a hit would serve the value the slider used to
        have. The document is the live truth, and reading it here is what makes dragging an
        overlay's opacity a repaint instead of a re-bake."""
        node = self.document.nodes.get(ovl_id)
        try:
            return float((node.params if node is not None else {}).get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def overlay_style(self, node_id: str) -> Dict[int, Tuple[int, float, float]]:
        """``{channel index: (blend mode, opacity, checker cells)}`` for the overlay's
        channels — what the shader and its CPU mirror need to composite them."""
        ctx = self._overlay_ctx
        if not ctx or ctx["node"] != node_id:
            return {}
        out: Dict[int, Tuple[int, float, float]] = {}
        for src in ctx.get("sources", ()):
            entry = src["entry"]
            # each source carries its OWN blend and opacity — the whole point of chaining
            # is that a context view and a segmentation want different looks
            name = str(entry.get("blend", "add"))
            # the third slot is the MODE's own parameter, so it changes meaning with the
            # mode: checker density for `checkerboard`, divider position for `wipe`
            param = (self._look(src["ovl_id"], "wipe_pos", 0.5) if name == "wipe"
                     else OVERLAY_CHECKER_CELLS)
            # A `view_source` channel composites at FULL strength: it has no opacity socket to
            # read, and an Overlay's 0.5 default would make a second channel of the same
            # acquisition read as a wash laid over the first instead of as a channel.
            opacity = (1.0 if src.get("as_channel")
                       else self._look(src["ovl_id"], "opacity", 0.5))
            style = (self.BLEND_MODES.get(name, 0), opacity, param)
            for k in range(int(src["n"])):
                out[int(src["base_c"]) + k] = style
        return out

    def _decode_planes(self, provider, node_id, coords, channels, axes,
                       *, pin: Optional[Pin] = None, overlay_all: bool = False,
                       as_dtype: Any = None) -> Dict[int, np.ndarray]:
        """Native per-channel planes at ``coords`` (a GLOBAL display cursor), cache-first
        (used by the pull worker and by :class:`_DecodeJob`). Misses decode and are cached.

        Composed overlay planes are cached alongside the read ones, under the same keys
        :meth:`_plane_addrs` hands out, so the warm fast path serves a frame's overlay instead
        of quietly dropping it.

        ``overlay_all`` includes every overlay channel the chain contributes, whether or not
        it was asked for — right for a FULL PULL, which is the moment the Viewer first learns
        those channels exist (it cannot request an index it has never been told about). On the
        cursor fast path the request is authoritative instead, which is what makes an overlay
        channel's toggle button work.

        **Never call this on the GUI thread.** On a lazy provider a miss runs the node — see
        :class:`_DecodeJob` for the minutes-long freeze that taught us so."""
        out: Dict[int, np.ndarray] = {}
        addrs = self._plane_addrs(node_id, coords, channels, axes, pin=pin,
                                  provider=provider)
        if not addrs:
            return out
        nc = int(axes.c)
        cap = addrs[0][0][-1]          # the cap the keys were built with — never re-derived
        dt = self._viewer_dtype if as_dtype is None else as_dtype
        ovl_addrs = [a for a in addrs if a[4] >= nc]
        ref: Optional[np.ndarray] = None
        for key, m, t, z, ch in addrs:
            if ch >= nc:
                continue                       # composed below, not read from the provider
            arr = self._planes.get(key)
            if arr is None:
                arr, _lv = render_plane_native(provider, m, t, z, ch, max_dim=cap,
                                               as_dtype=dt)
                self._planes.put(key, arr)
            out[ch] = arr
            if ref is None:
                ref = arr
        if self._overlay_ctx is None or not (ovl_addrs or overlay_all):
            return out
        m, t, z, cur = self._clamp_coords(self._payload_coords(coords, pin), axes)
        if ref is None:
            # Every primary channel is toggled off and only the overlay is shown. The
            # composite is still expressed on the primary's display grid, so one primary
            # plane is read for its SHAPE — and deliberately not put in `out`, because the
            # user turned it off.
            key = _plane_key(node_id, pin, m, t, z, cur, cap)
            ref = self._planes.get(key)
            if ref is None:
                ref, _lv = render_plane_native(provider, m, t, z, cur, max_dim=cap,
                                               as_dtype=dt)
                self._planes.put(key, ref)
        # The overlay is composed onto the shape of the plane actually being shown, and
        # placed by µm, so the display decimation costs it nothing and no caller has to
        # track a scale factor.
        z_um = _z_um_of(self._overlay_ctx["pri_md"], self._overlay_ctx["pri_axes"], m, z)
        composed = self._compose_overlay(node_id, ref.shape[:2], m, t, z_um)
        keys = {a[4]: a[0] for a in ovl_addrs}
        for ch, plane in composed.items():
            key = keys.get(ch)
            if key is None:
                if not overlay_all:
                    continue                   # not asked for: its toggle is off
                key = _plane_key(node_id, pin, m, t, z, int(ch), cap)
            self._planes.put(key, plane)
            out[int(ch)] = plane
        return out

    # ── raw (pre-enhancement) pixels, for the Viewer's hover readout ───────────
    def raw_source(self, node_id: str):
        """``(source_node_id, provider, envelope)`` of the ONE ``io.load`` feeding
        ``node_id`` **whose geometry ``node_id`` preserves** — or ``None``.

        This is what lets the hover readout say "1301 (raw 1284)": the number a filter
        produced beside the number the microscope recorded at the same pixel. Two gates,
        both refusals rather than guesses:

        * **exactly one** source in the pull's ancestor closure. With two loads merged
          there is no single "the raw data", and picking one would silently attribute a
          value to the wrong file.
        * the node's propagated axes **equal** the source's on all six. A crop, a
          resample, a z-project or a channel-select all re-address what ``(m,t,z,c,y,x)``
          means, so the source pixel at the same index is a *different* pixel — and a
          confidently mislabelled "raw" is worse than no raw at all. Enhancement nodes
          (filters, normalize, background subtraction) preserve axes by construction, and
          they are exactly the case this readout exists for.

        Both envelopes are read UNPINNED, so the answer does not flip when the
        solo-frame scope shortens the axes on both sides at once.

        Memoized per ``(node, document revision)``: the caller is a MOUSE-MOVE handler and
        the answer costs a run-graph build (``planned_nodes``) plus an envelope lookup —
        several milliseconds that would land on every pixel the pointer crosses. Keying on
        the revision is what keeps it honest: any edit that could change the answer (a new
        wire, a different path, a crop inserted upstream) bumps it.
        """
        ck = (node_id, self.document.revision)
        hit = self._raw_src.get(ck)
        if hit is not None:
            return hit
        got = self._raw_source_uncached(node_id)
        if got is None:
            # Deliberately NOT cached. A refusal is often just "the source has not been
            # resolved yet" — the provider is registered by the worker DURING the first
            # pull, at an unchanged document revision — and caching that would leave the
            # readout without raw values until the next edit.
            return None
        self._raw_src[ck] = got
        if len(self._raw_src) > 64:              # bounded: revisions climb forever
            for stale in [k for k in self._raw_src if k[1] != self.document.revision]:
                self._raw_src.pop(stale, None)
        return got

    def _raw_source_uncached(self, node_id: str):
        srcs = [n for n in self.planned_nodes(node_id) if n in self._node_source_key]
        if len(srcs) != 1:
            return None
        entry = self._providers.get(self._node_source_key[srcs[0]])
        if entry is None:
            return None
        prov, env = entry
        try:
            node_env = self.document.env(node_id)
        except Exception:                        # noqa: BLE001 — an un-propagated node
            return None
        a, b = node_env.axes, env.axes
        if ((a.m, a.t, a.z, a.c, a.y, a.x) != (b.m, b.t, b.z, b.c, b.y, b.x)):
            return None
        return srcs[0], prov, env

    def raw_plane(self, node_id: str, m: int, t: int, z: int,
                  c: int) -> Optional[np.ndarray]:
        """The SOURCE plane behind the viewed node at GLOBAL ``(m,t,z,c)``, decoded with
        the same level-pick + decimation as the displayed one (so it indexes identically),
        or ``None`` when :meth:`raw_source` refuses.

        Global coords on purpose: the source provider held here is the **unpinned** one,
        so the solo-frame scope's payload re-addressing must NOT be applied.

        Cached in the same :class:`PlaneCache` as the display planes. When the viewed node
        *is* the load and nothing is pinned, it reuses the display path's own key rather
        than minting a second one — hovering over a raw source then costs no extra read
        and no extra memory."""
        got = self.raw_source(node_id)
        if got is None:
            return None
        src_id, prov, env = got
        ax = env.axes
        m, t, z, c = (min(max(0, int(v)), s - 1) for v, s in
                      zip((m, t, z, c), (ax.m, ax.t, ax.z, ax.c)))
        key = ((node_id, None, m, t, z, c) if src_id == node_id
               and self._viewer_pin is None
               else (_RAW_TAG, self._node_source_key[src_id], m, t, z, c))
        arr = self._planes.get(key)
        if arr is None:
            arr, _lv = render_plane_native(prov, m, t, z, c)
            self._planes.put(key, arr)
        return arr

    def _serve_from_cache(self, node_id, coords, channels) -> None:  # GUI thread
        """Display the plane(s) at ``coords`` without going through the engine.

        A **warm** frame is served right here: the pixels are already decoded, so emitting
        them is a dictionary lookup and scrubbing keeps its zero-hop latency. A **cold** one
        goes to :class:`_DecodeJob` on the pool — decoding it here would run the node on the
        GUI thread and freeze the application for as long as that takes."""
        t0 = time.perf_counter()
        axes = self._viewer_axes
        pin = self._viewer_pin
        warm = self._cached_planes(node_id, coords, channels, axes, pin=pin)
        if warm is not None:
            self.plane_ready.emit(node_id, warm, axes, time.perf_counter() - t0)
            self.prefetch(node_id,
                          self._clamp_coords(self._payload_coords(coords, pin), axes),
                          tuple(channels) if channels else None, pin=pin)
            return
        if self._decode_busy:
            self._decode_pending = (node_id, coords, channels)   # latest-wins
            return
        self._decode_gen += 1
        self._decode_busy = True
        # the card + status bar say "reading planes" for the whole decode — the same
        # runner-level event the full-pull path emits when a lazy chain does its real work
        # at read time (`_Worker.run`), so a cold scrub is visibly working rather than hung
        self._progress.emit(("decode", node_id, {"epoch": self._epoch, "op_key": ""}))
        self._pool.start(_DecodeJob(self, self._decode_gen, self._epoch,
                                    self._viewer_provider, node_id, coords, channels,
                                    axes, pin))

    def _deliver_planes(self, packet) -> None:   # GUI thread (queued)
        """Land a :class:`_DecodeJob`'s planes, then run whatever the cursor did meanwhile.

        Staleness is judged on ``_decode_gen`` (the cursor moved on) and ``_epoch`` (an edit
        or a real pull happened): either way the pixels describe a frame nobody is looking at
        any more, so they are dropped — they are still in the PlaneCache, so nothing is
        wasted if the cursor comes back."""
        (gen, epoch, node_id, coords, channels, pin, planes, axes, dt, err) = packet
        self._decode_busy = False
        pending, self._decode_pending = self._decode_pending, None
        fresh = gen == self._decode_gen and epoch == self._epoch
        if fresh and err is not None:
            self.failed.emit(node_id, err)
        elif fresh and planes:
            self._progress.emit(("done", node_id, {"epoch": epoch, "seconds": dt}))
            self.plane_ready.emit(node_id, planes, axes, dt)
            self.prefetch(node_id,
                          self._clamp_coords(self._payload_coords(coords, pin), axes),
                          tuple(channels) if channels else None, pin=pin)
        if pending is not None:
            # back through request_plane, not straight to the decode: the node, the document
            # revision and the pin all have to be re-checked, and a cursor that wandered
            # back onto a warm plane while this ran should serve inline
            self.request_plane(*pending)

    def prefetch(self, node_id, center, channels, *, span: int = 8,
                 pin: Optional[Pin] = None) -> None:
        """Warm the plane cache for frames around ``center`` — **payload** coords, already
        mapped through the scope — on background pool threads: bidirectional in T (covers
        scrubbing), nearest-first (covers forward play). A new call supersedes older
        prefetch jobs via ``_prefetch_gen``.

        The span is cut to what a neighbouring plane actually COSTS on the held provider
        (:meth:`_prefetch_span`) — reading one ahead is only free when the pixels are
        already bytes somewhere.

        Under a one-frame scope the held payload's ``t`` is 1, so the guard below returns
        and nothing is warmed — correctly, since a neighbouring frame is not in that
        dataset at all and reaching it needs a re-pull. A multi-frame scope does have
        neighbours to warm, and they are the picked ones."""
        prov, axes = self._viewer_provider, self._viewer_axes
        if prov is None or axes is None or getattr(axes, "t", 1) <= 1:
            return
        span = min(span, self._prefetch_span(prov))
        if span <= 0:
            return
        m, t, z, _c = center
        nt = axes.t
        chans = tuple(min(max(0, int(ch)), axes.c - 1)
                      for ch in (channels if channels else
                                 (min(max(0, center[3]), axes.c - 1),)))
        cap = self.display_dim(axes, planes=max(1, len(set(chans))))
        self._prefetch_gen += 1
        gen = self._prefetch_gen
        jobs: List[Tuple[tuple, int, int, int, int]] = []
        for d in range(1, span + 1):
            for tt in ((t + d) % nt, (t - d) % nt):
                for ch in chans:
                    key = (node_id, pin, m, tt, z, ch, cap)
                    if self._planes.get(key) is None:
                        jobs.append((key, m, tt, z, ch))
        if jobs:
            self._pool.start(_PrefetchJob(self, gen, jobs))

    #: How many preload jobs run at once. Four, measured: on the WellA3 mosaic with a cold tile
    #: cache one whole-canvas stitch is 1.03 s, four in parallel are 0.51 s each and eight are
    #: 0.43 s — the source-tile reads parallelize, the paste contends, and the curve is flat past
    #: four. Kept low on purpose: these share the pool with the frame being DISPLAYED, and
    #: starving that to fetch the future is exactly backwards.
    PRELOAD_JOBS = 4

    def _frame_bytes(self, axes: Any, chans: int) -> int:
        """Bytes one displayed frame of ``chans`` channels occupies in the plane cache."""
        cap = self.display_dim(axes, planes=max(1, chans))
        return (min(cap, int(axes.y)) * min(cap, int(axes.x))
                * (2 if self._viewer_dtype is not None else 8) * max(1, chans))

    def preload_series(self, node_id: str, center, channels, *,
                       pin: Optional[Pin] = None) -> int:
        """Decode the whole T range of the viewed frame into the plane cache, in playback
        order, on several pool threads. Returns how many planes were queued.

        This is Nikon's "normal mode" made explicit: an image that fits the memory budget is
        held whole, and then playback is a texture upload per frame rather than a read. The
        ordinary :meth:`prefetch` cannot do this job — it is a *scrub* heuristic, cost-gated to
        ±2 frames on a computing provider precisely so that nudging the cursor cannot queue
        sixteen whole-volume deconvolutions. Pressing play is a different statement: every
        frame is wanted, in order.

        Three properties, each of which was a bug first:

        * **its own cancellation generation.** Sharing ``_prefetch_gen`` meant the first frame
          advance — which calls :meth:`prefetch` — cancelled the preload, so playback went
          straight back to decoding every frame as it arrived (reported 2026-08-05, "it loads
          every time"). Cancelled now only by :meth:`cancel_preload`, an edit, or a newer one.
        * **from the CURSOR, wrapping.** Playback starts where you are, not at t=0, so reading
          from 0 spent the first seconds decoding frames that would be shown last.
        * **bounded by the cache that actually holds it.** The frames land in
          :class:`PlaneCache`, so its budget is the ceiling — sizing against a *different*
          number is how a preload evicts its own head and re-decodes every lap.
        """
        prov, axes = self._viewer_provider, self._viewer_axes
        if prov is None or axes is None or node_id != self._viewer_node:
            return 0
        m, t0, z, _c = center
        chans = tuple(sorted({min(max(0, int(ch)), axes.c - 1)
                              for ch in (channels or (center[3],))}))
        cap = self.display_dim(axes, planes=max(1, len(chans)))
        nt = max(1, int(axes.t))
        room = max(1, min(self._planes.budget, display_ram_bytes())
                   // max(1, self._frame_bytes(axes, len(chans))))
        start = min(max(0, int(t0)), nt - 1)
        want: List[Tuple[tuple, int, int, int, int]] = []
        for i in range(min(nt, int(room))):
            tt = (start + i) % nt
            for ch in chans:
                key = (node_id, pin, m, tt, z, ch, cap)
                if self._planes.get(key) is None:
                    want.append((key, m, tt, z, ch))
        self._preload_gen += 1
        self._preload_node = node_id
        self._preload_total = len(want)
        self._preload_done = 0
        if not want:
            return 0
        # Round-robin rather than contiguous blocks: every job then walks forward through the
        # series roughly together, so the frames nearest the cursor land first whichever job
        # gets a thread.
        n = max(1, min(self.PRELOAD_JOBS, max(1, self._pool.maxThreadCount() - 1)))
        for k in range(n):
            share = want[k::n]
            if share:
                self._pool.start(_PreloadJob(self, self._preload_gen, share))
        return len(want)

    def cancel_preload(self) -> None:
        """Retire any preload in flight (playback stopped, the node changed, an edit landed).
        The planes already decoded stay cached — they are still the right pixels."""
        if self._preload_total:
            self._preload_gen += 1
            node, self._preload_node = self._preload_node, None
            self._preload_total = self._preload_done = 0
            if node:
                self.preload_finished.emit(node, False)

    def _deliver_preload_tick(self, packet) -> None:            # GUI thread
        gen, n = packet
        if gen != self._preload_gen or not self._preload_total:
            return
        self._preload_done += int(n)
        node = self._preload_node or ""
        self.preload_progress.emit(node, self._preload_done, self._preload_total)
        if self._preload_done >= self._preload_total:
            self._preload_total = self._preload_done = 0
            self._preload_node = None
            self.preload_finished.emit(node, True)

    def preloading(self) -> bool:
        return bool(self._preload_total)

    def series_fits(self, axes: Any = None, *, planes: int = 1) -> bool:
        """Whether the whole T range of the viewed frame fits the budget — i.e. whether a
        preload can make playback read-free rather than merely warmer."""
        axes = self._viewer_axes if axes is None else axes
        if axes is None:
            return False
        ceiling = min(self._planes.budget, display_ram_bytes())
        return self._frame_bytes(axes, planes) * max(1, int(axes.t)) <= ceiling

    @staticmethod
    def _prefetch_span(prov: Any) -> int:
        """How many T-neighbours it is worth warming on ``prov`` — a COST decision, not a
        depth-of-lookahead one.

        The prefetcher was written against a store-backed provider, where a neighbouring
        plane is a decompress: read sixteen ahead and scrubbing never waits. On a *computing*
        provider the same read runs the node, and on a ``volume_unit`` one it runs the whole
        ``(Z, Y, X)`` unit. Warming ±8 T on the WellA3 640 series (210×1024² units, 3D
        Deconvolve) therefore queued sixteen whole-volume Richardson–Lucy computes — roughly
        an hour of CPU and ~30 GB of working set per volume — behind a cursor that had moved
        one frame, saturating every core the Viewer's own decode needed and pushing the
        volume being *looked at* down the cache LRU.

        So: full span on real bytes, a short one on a per-plane/per-tile compute (where a
        neighbour is one kernel and prefetching is still the right trade), none across
        frames of a whole-unit compute — there, the useful neighbours are the other z of the
        unit already in hand, and they are cached by the read that displayed it."""
        if not isinstance(prov, StreamProvider):
            return 8                                  # decompress-only: warm freely
        if getattr(prov, "volume_unit", False):
            return 0                                  # a neighbour t IS a whole volume
        return 2

    # ── per-node progress (engine observer → GUI thread) ──────────────────────
    def _make_observer(self, epoch: Optional[int]):
        """An :data:`nodegraph.engine.Observer` that forwards node events to the GUI
        thread through ``_progress`` (a queued signal — the observer is called on the
        worker thread). Fractional ``progress`` events are rate-limited per node; every
        other event, the final ``done == total``, and every frame boundary always gets
        through.

        ``epoch=None`` means **not part of a pull** and switches the staleness gate off, at
        both ends (here and in :meth:`_deliver_progress`). That is what a background ingest
        reports through (:class:`_IngestJob`): it belongs to a file rather than to a run, so
        an edit or another node's pull moving the epoch must not silence its card — the work
        carries on either way, and a bar that stops moving reads as a hang."""
        def observe(event: str, node_id: str, info: Dict[str, Any]) -> None:
            if epoch is not None and epoch != self._epoch:
                return                        # superseded pull — stop reporting for it
            if event == "progress":
                now = time.perf_counter()
                last = self._last_progress.get(node_id, 0.0)
                final = info.get("done") == info.get("total")
                frames_done = info.get("frames_done")
                # a frame step is never dropped: it is the *only* update that moves the
                # frame bar, and the next one may be a whole frame of work away.
                stepped = (frames_done is not None
                           and frames_done != self._last_frame.get(node_id))
                # nor is a switch between a determinate sub bar and a sweeping one. These
                # arrive back-to-back — a unit lands and the next unit's opaque call starts
                # a millisecond later — so the throttle would eat the sweep every time and
                # leave the card frozen at the last percentage for the whole unit, which is
                # precisely the "bar isn't moving" this reports its way out of.
                sweeping = info.get("sub_fraction", False) is None
                flipped = sweeping != self._last_sweep.get(node_id)
                if (not final and not stepped and not flipped
                        and (now - last) < PROGRESS_MIN_INTERVAL_S):
                    return
                self._last_progress[node_id] = now
                self._last_sweep[node_id] = sweeping
                if frames_done is not None:
                    self._last_frame[node_id] = frames_done
            else:
                self._last_progress.pop(node_id, None)
                self._last_frame.pop(node_id, None)
                self._last_sweep.pop(node_id, None)
            self._progress.emit((event, node_id, {**info, "epoch": epoch}))
        return observe

    def _deliver_progress(self, packet) -> None:     # GUI thread (queued)
        event, node_id, info = packet
        epoch = info.get("epoch")
        if epoch is not None and epoch != self._epoch:
            return                            # a stale pull's tail — the cards moved on
        self.node_progress.emit(event, node_id, info)

    def planned_nodes(self, node_id: str, graph: Optional[Graph] = None) -> List[str]:
        """Every node a pull of ``node_id`` may evaluate: itself plus its transitive
        upstream (a memo hit still *participates*, and reports itself ``cached``). Read
        off the run graph so it matches what the engine will actually walk — muted nodes
        are bypassed there, and group bodies are expanded."""
        try:
            graph = graph if graph is not None else self.document.to_graph(
                for_run=True, materialize=True, unroll_iterate=True,
                sweep_all=self._sweep_all)
        except Exception:  # noqa: BLE001 — an unbuildable graph plans as just the target
            return [node_id]
        if node_id not in graph.nodes:
            return [node_id]
        seen, stack = set(), [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(e.src for e in graph.preds(nid) if e.src not in seen)
        return sorted(seen)

    # ── bake (Dock) ───────────────────────────────────────────────────────────
    def start_hold(self, node_id: str, *, signature: str) -> bool:
        """Pull ``node_id``'s input and freeze it in memory. Returns False when a pull is
        already in flight.

        A thin wrapper over :meth:`bake` sharing its job, graph and worker — see
        :meth:`_run_bake`. It passes no store and no precision because a hold writes nothing;
        ``signature`` is still recorded so a held dock reports staleness on an upstream edit
        exactly as a docked one does."""
        return self.bake(node_id, store="", precision="", bake_id="",
                         signature=signature, hold=True)

    def bake(self, node_id: str, *, store: str, precision: str, bake_id: str,
             signature: str, scoped: bool = False, hold: bool = False,
             coords: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Run a Dock node's bake: pull everything upstream of ``node_id`` and write it to
        the checkpoint at ``store``. Returns False when a pull is already in flight.

        The graph is built with this dock forced **live**, so the chain behind it is
        walked even on a *re*-bake — where the node is currently docked and its in-edge
        would otherwise be cut, and the bake would cheerfully write the old checkpoint
        back over itself.

        ``scoped`` bakes only what the Viewer's frame selection covers instead of the
        whole series. That is a troubleshooting shortcut, not a result: the checkpoint
        then contains a *truncated* series and every node downstream runs on it, which is
        why the caller has to ask for it explicitly and the card stays marked."""
        if self._busy:
            return False
        self._stop_bake = False          # a stop never leaks into the next bake
        self._epoch += 1
        self._last_progress.clear()
        self._last_frame.clear()
        self._last_sweep.clear()
        graph = self.document.to_graph(for_run=True, materialize=True,
                                       unroll_iterate=True, sweep_all=self._sweep_all,
                                       live_docks=frozenset({node_id}))
        sources, every = self._sources_for(node_id, graph)
        job = _Job(self._epoch, graph, self.document.revision, node_id, None, None,
                   sources, pin=self._pin_for(coords) if scoped else None,
                   bake={"store": store, "precision": precision, "bake_id": bake_id,
                         "signature": signature, "scoped": bool(scoped),
                         "hold": bool(hold)},
                   all_sources=every)
        self._busy = True
        self._bakes[job.epoch] = job.bake
        self.plan.emit(node_id, self.planned_nodes(node_id, graph))
        self.started.emit(node_id)
        self._pull_thread.start(_Worker(self, job))
        return True

    def _run_bake(self, engine: Engine, job: _Job) -> None:   # worker thread
        """Pull the dock's input and freeze it — to disk (``bake``) or in memory (``hold``).

        Runs inside the worker, reporting through the same observer a compute's
        ``ctx.progress`` uses — so the dock's card shows the bake exactly like any other long
        node, throttle and stale-epoch guard included.

        A **hold** shares this whole path deliberately, and takes the same graph: the dock is
        forced live so the chain behind it is walked rather than cut, which is what makes
        re-holding an already-held node work instead of freezing its own frozen output. All it
        then skips is the write. Sharing the path is also what gives a hold the pull's epoch
        guard, its refusals and its "a pull is already running" interlock for free."""
        from nodegraph.checkpoint import checkpoint_bytes, write_checkpoint
        spec = job.bake or {}
        payload = engine.pull(job.node_id)
        verb = "hold" if spec.get("hold") else "bake"
        if not isinstance(payload, Dataset):
            raise TypeError(
                f"a Dock can only {verb} a Dataset; this one's input produced "
                f"{type(payload).__name__}. Wire the image/analysis chain into it.")
        if spec.get("hold"):
            # No write, no copy — the whole reason this tier is instant. The envelope is
            # captured alongside because a held dock has no manifest to re-derive one from.
            spec["payload"] = payload
            spec["env"] = engine.env(job.node_id)
            return
        observe = self._make_observer(job.epoch)

        def on_progress(fraction: float, note: str) -> None:
            f = max(0.0, min(1.0, float(fraction)))
            observe("progress", job.node_id,
                    {"fraction": f, "note": note,
                     "done": int(round(f * 1000)), "total": 1000})

        env = engine.env(job.node_id)
        man = write_checkpoint(
            payload, spec["store"], precision=spec["precision"],
            bake_id=spec["bake_id"],
            # Prefer the edit-time envelope's own catalog over one derived from the
            # payload: it is what `propagate_meta` computed for this edge, so a docked
            # node describes itself to the GUI exactly as the live one did.
            domains=(env.domains or None), layer_names=(env.layer_names or None),
            progress=on_progress,
            # A bake polls this at every block boundary and returns None having written no
            # manifest, so a stopped bake leaves the directory reading as *absent* — the
            # state this module already treats as correct for an interrupted one.
            should_cancel=lambda: self._stop_bake)
        if man is None:
            # Cancelled. Leave `spec["manifest"]` unset so `_deliver` records no bake and
            # the dock does not start claiming a checkpoint that was never finished.
            spec["cancelled"] = True
            return
        spec["manifest"] = man
        spec["bytes"] = checkpoint_bytes(spec["store"])

    # ── hold (the session tier) ───────────────────────────────────────────────
    @property
    def held(self) -> frozenset:
        """The node ids currently holding a payload in memory — what
        :func:`nodelab_v2.ops.dock_status` needs to tell ``held`` from ``released``."""
        return frozenset(self._held)

    def hold(self, node_id: str, payload: Any, env: Optional[MetaEnvelope] = None) -> None:
        """Pin ``payload`` as ``node_id``'s frozen result. **No copy and no write** — this is
        the whole reason the tier is instant.

        The engine must be rebuilt so the next pull seeds from the registry rather than
        walking the (now cut) chain; ``_engine_rev = -1`` is the narrow form of that, already
        used by :meth:`set_sweep_all` — it drops no memo entry, so everything computed on the
        way to this payload stays a hit.

        Refuses a non-Dataset for the same reason :meth:`_run_bake` does: the alternative is a
        seed the engine cannot use, surfacing later as a confusing type error inside a compute
        rather than here where the user pressed the button."""
        if not isinstance(payload, Dataset):
            raise TypeError(
                f"a Dock can only hold a Dataset; this one's input produced "
                f"{type(payload).__name__}. Wire the image/analysis chain into it.")
        self._held[node_id] = payload
        if env is not None:
            self._held_envs[node_id] = env
        self._engine_rev = -1

    def release(self, node_id: str) -> bool:
        """Drop ``node_id``'s held payload (un-hold). True when something was released.

        Rebuilds the engine for the mirror of :meth:`hold`'s reason: without it the cached
        engine keeps the stale seed and would go on serving the released payload until the
        next document edit happened to invalidate it."""
        had = self._held.pop(node_id, None) is not None
        self._held_envs.pop(node_id, None)
        if had:
            self._engine_rev = -1
        return had

    def request_stop_bake(self) -> None:
        """Ask an in-flight bake to stop at its next block boundary.

        Cleared by :meth:`bake` when the next one starts, so a stop can never leak into a
        later request. The writer's own contract does the rest: no manifest is written, so
        the half-written store reads as un-baked rather than as a shorter valid one."""
        self._stop_bake = True

    # ── the sweep scope (flow.iterate) ────────────────────────────────────────
    @property
    def sweep_all(self) -> frozenset:
        return self._sweep_all

    def set_sweep_all(self, node_ids: Iterable[str]) -> None:
        """Mint EVERY iteration of these Iterate nodes on subsequent pulls.

        A ``preserve=picked`` sweep normally clones only the chosen iteration, so a finished
        graph costs one run instead of N. "Run sweep" turns that off for the named nodes so
        all N results exist to compare and to fill the results table. The extra clones are
        ordinary memoized nodes, so flipping the flag back off does not throw them away —
        re-running the sweep after looking away is a cache hit.

        **The engine must be rebuilt.** ``_ensure_engine`` caches one Engine per *document
        revision* and only refreshes its seeds — the graph it was constructed with is fixed.
        This flag changes the graph without touching the document, so without forcing a
        rebuild the next pull would hand the worker a freshly unrolled 3-clone graph and the
        cached engine would quietly walk the 1-clone one it still held. Clearing
        ``_engine_rev`` is the narrow form of that: no memo is dropped, so every iteration
        already computed is still a hit."""
        ids = frozenset(node_ids)
        if ids != self._sweep_all:
            self._sweep_all = ids
            self._engine_rev = -1

    def unload(self, node_ids: Iterable[str]) -> int:
        """Release everything held for ``node_ids`` — the point of docking.

        Drops their memo entries (the eager full-raster Voxel layers a segmentation or
        threshold chain retains) and empties the shared tile and decoded-plane caches,
        whose contents belong to providers nothing will read again. Returns the number of
        memo entries freed. Correctness-safe: every drop can only cost a recompute.

        The tile cache is cleared WHOLESALE rather than by node: its keys are content
        addresses (a provider fingerprint plus a grid position), with no node in them, so
        there is nothing to select on. That is acceptable here precisely because a dock
        makes most of it dead — the tiles of a chain nothing will read again — and what
        survives is one decompress away."""
        n = self._memo.drop_nodes(list(node_ids))
        self._tiles.clear()
        self._planes.clear()
        self._viewer_provider = None
        self._viewer_node = None
        self._viewer_rev = -1
        self._prefetch_gen += 1
        return n

    # ── internals ─────────────────────────────────────────────────────────────
    def _all_sources(self) -> Dict[str, Dict[str, Any]]:
        return {rec.id: {"path": str(rec.params.get("path", "") or "")}
                for rec in self.document.nodes.values() if rec.op_key == LOAD_OP}

    def _sources_for(self, node_id: str, graph: Graph
                     ) -> Tuple[Dict[str, Dict[str, Any]], frozenset]:
        """``(the sources this pull must resolve, every source in the document)``.

        Only the ``io.load`` roots inside the pulled node's ancestor closure are resolved
        (V2.21). Before this the runner resolved **every** source in the document on every
        pull, which meant that with the multi-file loader — five ND2s dropped on the canvas
        — double-clicking any one node ingested all five, in series, inside the single pull
        slot, before the engine ran a line. Nothing needed the other four; the pull just
        happened to be holding the list.

        The un-needed ones are not forgotten, only left alone: they keep whatever the engine
        already holds for them (see :meth:`_ensure_engine`) and land in ``_providers`` when
        their own chain is pulled or their card is ingested."""
        every = self._all_sources()
        needed = set(self.planned_nodes(node_id, graph))
        return ({nid: cfg for nid, cfg in every.items() if nid in needed},
                frozenset(every))

    def _submit(self, node_id: str, coords, channels=None) -> None:
        self._epoch += 1
        self._last_progress.clear()
        self._last_frame.clear()
        self._last_sweep.clear()
        graph = self.document.to_graph(for_run=True, materialize=True,
                                       unroll_iterate=True, sweep_all=self._sweep_all)
        sources, every = self._sources_for(node_id, graph)
        job = _Job(self._epoch, graph,
                   self.document.revision, node_id, coords, channels, sources,
                   pin=self._pin_for(coords), all_sources=every)
        self._busy = True
        self.plan.emit(node_id, self.planned_nodes(node_id, graph))
        self.started.emit(node_id)
        self._pull_thread.start(_Worker(self, job))

    def _deliver(self, packet) -> None:          # GUI thread (queued)
        (epoch, node_id, payload, plane, axes, dt, err, revision, coords, channels,
         pin) = packet
        self._busy = False
        # A bake is resolved FIRST and outside the staleness rule. Its real result is a
        # directory on disk that already exists by now, so "the user moved on while it
        # ran" is not a reason to forget it — that would leave an orphaned checkpoint and
        # a dock that still thinks it was never baked.
        spec = self._bakes.pop(epoch, None)
        if spec is not None:
            if err is not None:
                self.failed.emit(node_id, err)
            else:
                self.baked.emit(node_id, spec)
            pending, self._pending = self._pending, None
            if pending is not None:
                self._submit(*pending)
            return
        # staleness is judged BEFORE the re-seed side effect below: delivering a
        # resolved source envelope notifies the document → the window calls
        # invalidate() → the epoch bumps — and would drop the very result being
        # delivered. The envelope seeding is display-only (the ENGINE resolved its
        # own meta seeds at run time), so it never stales this result.
        stale = epoch != self._epoch
        # G8 live re-seed: hand newly resolved source envelopes to the document
        for nid, env in self._fresh_envs():
            self.document.set_meta_seed(nid, env)
        pending, self._pending = self._pending, None
        if pending is not None:
            self._submit(*pending)               # latest-wins supersedes this result
            stale = True
        if stale:
            return
        if err is not None:
            self.failed.emit(node_id, err)
            return
        # Hold the image provider so subsequent coords-only requests skip the engine
        # (the fast path). Tie it to the CURRENT document revision — not the job's: the
        # G8 source re-seed just above (``set_meta_seed``) can bump the revision (display
        # metadata only, the graph/pixels are unchanged), and a genuine edit later runs
        # invalidate() → ``_viewer_rev = -1`` anyway, so the next request re-pulls.
        if isinstance(payload, Dataset) and payload.image is not None:
            self._viewer_provider = payload.image
            self._viewer_node = node_id
            self._viewer_axes = payload.axes
            self._viewer_rev = self.document.revision
            self._viewer_pin = pin       # which frame this held payload IS
        self.finished.emit(node_id, payload, plane, axes, dt)
        if (coords is not None and self._viewer_provider is not None
                and self._viewer_axes is not None):
            self.prefetch(node_id,
                          self._clamp_coords(self._payload_coords(coords, pin),
                                             self._viewer_axes),
                          tuple(channels) if channels else None, pin=pin)

    def _fresh_envs(self):
        # re-announce whenever a node's RESOLVED source key changed (a path edit →
        # a new provider/envelope), not only on first resolution — else the G8 pill
        # re-seed would freeze on the synthetic fallback forever (review 2026-07-22).
        out = []
        for nid, key in list(self._node_source_key.items()):
            if self._announced.get(nid) == key:
                continue
            entry = self._providers.get(key)
            if entry is not None:
                self._announced[nid] = key
                out.append((nid, entry[1]))
        return out

    def _prune(self) -> None:
        """Drop bookkeeping for deleted io.load nodes so it doesn't grow unbounded
        (the resolved-provider cache is keyed by source path, shared across nodes, so
        it is left intact — reopening the same file is a hit)."""
        live = set(self.document.nodes)
        for nid in list(self._node_source_key):
            if nid not in live:
                self._node_source_key.pop(nid, None)
                self._announced.pop(nid, None)
        # A card deleted mid-ingest stops being tracked, but the JOB is deliberately left
        # to finish: it is writing a store keyed by the file, not by the card, and killing
        # it halfway is what leaves a torn one behind. Its delivery finds nothing to
        # announce and quietly lands in `_providers` for whoever loads that file next.
        for nid in [n for n in self._ingesting if n not in live]:
            self._ingesting.pop(nid, None)
        for ck in [k for k in self._raw_src if k[0] not in live]:
            self._raw_src.pop(ck, None)
        if self._viewer_node is not None and self._viewer_node not in live:
            self._viewer_node = None
            self._viewer_provider = None
            self._viewer_rev = -1
            self._viewer_pin = None

    def _ensure_engine(self, job: _Job) -> Engine:   # worker thread
        seeds: Dict[str, Any] = {}
        meta_seeds: Dict[str, MetaEnvelope] = {}
        for nid, cfg in job.sources.items():
            prov, env = self._resolve_source(nid, cfg, epoch=job.epoch)
            full_m = int(env.axes.m)
            if job.pin is not None:
                prov, env = _pin_frames(prov, env, job.pin)
            # Display metadata (channel names/emission/colors) rides on the seed
            # Dataset only — NOT the engine meta-seed (kept to the calibration schema).
            disp = self._channel_display.get(self._node_source_key.get(nid), {})
            md = dict(env.metadata); md.update(disp)
            # ...and the display dict carries the per-M STAGE keys too (`ingest.STAGE_KEYS`
            # is documented as riding "alongside the channel display keys"), at FULL length.
            # So this merge lands after `_pin_frames` and would put the un-subset list back
            # — re-apply the subset to the merged result, or the pin's fix is undone one
            # line later.
            if job.pin is not None:
                changes = position_subset(md, _picked(job.pin[0], full_m))
                md.update({k: v for k, v in changes.items() if v is not None})
                for k, v in changes.items():
                    if v is None:
                        md.pop(k, None)
            seeds[nid] = Dataset(axes=env.axes, metadata=md).with_image(prov)
            meta_seeds[nid] = env
        # Docked nodes are sources too (V2.18): the seed both satisfies the engine's
        # "a source needs no wired input" rule and folds the checkpoint's on-disk
        # identity into the node's recipe hash, so a RE-bake invalidates everything
        # downstream by itself. Deliberately NOT frame-pinned — a checkpoint is already
        # a finished result, and cutting one to the solo-frame scope would re-address
        # frames the bake did not have.
        # A HELD dock is seeded from the in-memory registry instead of a store, by the same
        # mechanism and for the same two reasons — it makes the node a source so its cut
        # primary input is not refused, and its provider's `version` folds into the recipe
        # hash so re-holding invalidates everything downstream on its own.
        for nid, ds in dock_seeds(job.graph, held=self._held).items():
            seeds[nid] = ds
            env = self._held_envs.get(nid) if nid in self._held else \
                checkpoint_envelope(dock_store_of(job.graph.nodes[nid]))
            if env is not None:
                meta_seeds[nid] = env
        seed_axes = {nid: ds.axes for nid, ds in seeds.items()}
        if job.bake is not None:
            # A bake runs a DIFFERENT graph at the same document revision — the target
            # dock is forced live so its chain is walked rather than cut — so it gets its
            # own engine rather than clobbering the cached one. The Memo is shared, which
            # is the point: everything the bake computes is available to the pull that
            # follows, and everything already computed is a hit the bake does not repeat.
            from nodegraph.nodes import COMPUTES
            return Engine(job.graph, computes=COMPUTES, memo=self._memo,
                          # the runner's own TileCache, for the same reason the cached
                          # engine gets it: a lazy Dataset the bake leaves in the shared
                          # memo holds its cache by weakref, and a per-bake cache would be
                          # collected the moment this engine is dropped — leaving every
                          # such chain reading through _NO_CACHE afterwards.
                          tiles=self._tiles,
                          seeds=seeds, meta_seeds=meta_seeds)
        if self._engine is None or self._engine_rev != job.revision:
            from nodegraph.nodes import COMPUTES
            self._engine = Engine(job.graph, computes=COMPUTES, memo=self._memo,
                                  tiles=self._tiles,
                                  seeds=seeds, meta_seeds=meta_seeds)
            self._engine_rev = job.revision
            self._seed_axes = dict(seed_axes)
            self._meta_seeds = dict(meta_seeds)
            return self._engine
        # Drop seeds this revision no longer has before adding the new ones: an
        # un-docked node keeps its stale checkpoint seed otherwise, and the engine
        # would still treat it as a source — silently serving the old bake through
        # a node the user just switched back to live.
        #
        # "No longer has" is NOT "not in this pull" (V2.21): a pull resolves only the
        # sources its own closure reaches, so a second ND2 sitting on the canvas is absent
        # from `seeds` while being a perfectly live source. Sweeping it would drop its seed
        # and the next pull of ITS chain would find an unseeded source — so the sweep is
        # against every source the document still has, plus this graph's docks.
        keep = set(seeds) | set(job.all_sources)
        for stale in [k for k in self._engine.seeds if k not in keep]:
            self._engine.seeds.pop(stale, None)
            self._seed_axes.pop(stale, None)
            self._meta_seeds.pop(stale, None)
        self._engine.seeds.update(seeds)
        # An engine kept across pulls carries the envelopes it was BUILT with, and
        # entering/leaving the solo-frame scope changes a source's axes (T→1) without
        # touching the document revision. Re-propagate when the seed geometry moves, so
        # every node's edit-time envelope keeps matching the payload it will be handed
        # (an axes disagreement is what the build-node-v2 §2 gate forbids). Moving the
        # pin WITHIN the scope changes only which frame, never the geometry — so
        # flipping between frames costs no re-propagation.
        #
        # Both sides are the ACCUMULATED maps, for the same reason the sweep is: comparing
        # this pull's sources against the previous pull's would report a change every time
        # the user alternates between two files' chains, and reseeding from the bare
        # `meta_seeds` would strip the other file's envelope off the engine each time.
        merged_axes = {**self._seed_axes, **seed_axes}
        merged_meta = {**self._meta_seeds, **meta_seeds}
        changed = merged_axes != self._seed_axes
        self._seed_axes = merged_axes
        self._meta_seeds = merged_meta
        if changed:
            self._engine.reseed_meta(merged_meta)
        return self._engine

    def _source_lock(self, key: Any) -> threading.Lock:
        """The one lock for ``key`` — created once, shared by every thread that wants that
        file. See :attr:`_src_locks` for why this exists."""
        with self._src_locks_guard:
            lock = self._src_locks.get(key)
            if lock is None:
                lock = self._src_locks[key] = threading.Lock()
            return lock

    def _resolve_source(self, node_id: str, cfg: Dict[str, Any],
                        epoch: Optional[int] = None, observe=None
                        ) -> Tuple[Any, MetaEnvelope]:   # worker thread
        """Resolve an ``io.load`` node's ``path`` to ``(provider, envelope)``, ingesting it
        to its ``.b2nd`` store on the first call and serving the cached provider after.

        Called from the pull worker AND from :class:`_IngestJob` on the ingest pool, so
        everything past the cache probe runs under the file's own lock: one ingest per file,
        and a caller that arrives mid-ingest waits and takes the result rather than starting
        a second writer on the same store.

        ``observe`` overrides the progress sink (the ingest pool passes an un-epoched one);
        by default it reports under ``epoch``, or the current one."""
        path = _clean_source_path(cfg.get("path", ""))
        if path:
            # an .nd3 path may carry a '#image_id' fragment (the one-image
            # escape hatch) — probe the FILE part, or every fragment path
            # would be reported missing.
            from nodelab_v2.ingest import split_fragment
            if not os.path.isfile(split_fragment(path)[0]):
                raise FileNotFoundError(
                    f"No such image file (ND2/ND3/TIFF):\n  {path!r}\n"
                    f"Reload it via File → Load ND2/ND3/TIFF file… (or fix the node's "
                    f"'path' field). Leave it empty for the synthetic demo source.")
        key = source_key(path)
        self._node_source_key[node_id] = key
        hit = self._providers.get(key)
        if hit is not None:
            return hit
        with self._source_lock(key):
            hit = self._providers.get(key)
            if hit is not None:
                return hit      # someone else ingested it while we waited on the lock
            return self._ingest_locked(node_id, path, key, epoch, observe)

    def _ingest_locked(self, node_id: str, path: str, key: Any,
                       epoch: Optional[int], observe
                       ) -> Tuple[Any, MetaEnvelope]:   # worker thread, holding the key lock
        if not path:
            prov = SyntheticProvider(_SYNTH_AXES, tile=128)
            env = MetaEnvelope(axes=_SYNTH_AXES, metadata=dict(_SYNTH_META))
            disp = {"channel_names": [f"Ch{i}" for i in range(_SYNTH_AXES.c)],
                    "channel_emission_nm": list(_SYNTH_META["channel_emission_nm"])}
        else:
            from nodelab_v2.ingest import (
                PYRAMID_LEVELS, ensure_store_levels, ingest_image, open_store,
                read_calibration, read_channel_display)
            # Beside the source file by default. `NODEGRAPH_STORE_DIR` moves every store to
            # one directory instead, which matters when the data lives on slow media: a USB
            # SSD here measured 27 MB/s of store write against 56 MB/s on the internal
            # NVMe, and 55 MB/s of raw sequential write against 952 MB/s. The store is a
            # derived cache, so relocating it costs nothing but has to stay UNIQUE per
            # source — hence the path digest, or two files of the same basename in
            # different folders would fight over one store.
            # An .nd3 '#image_id' fragment must land in its OWN store:
            # splitext("well.nd3#DAPI") strips ".nd3#DAPI", so the fragment
            # load and the whole-file load would otherwise share
            # "well.b2nd_store" — one image's pixels served under the other's
            # metadata. nd3 ids are [A-Za-z0-9_.-]+, filesystem-safe as a tag.
            from nodelab_v2.ingest import split_fragment
            file_part, frag = split_fragment(path)
            base = (os.path.splitext(file_part)[0]
                    + (f".{frag}" if frag else "") + ".b2nd_store")
            target_dir = store_dir(os.path.dirname(base))
            if os.path.abspath(target_dir) == os.path.abspath(os.path.dirname(base)):
                store = base
            else:
                tag = hashlib.blake2b(
                    os.path.abspath(path).lower().encode("utf-8"),
                    digest_size=6).hexdigest()
                store = os.path.join(
                    target_dir,
                    f"{os.path.splitext(os.path.basename(path))[0]}.{tag}.b2nd_store")
            # A store DIRECTORY existing is not proof the store is USABLE: write()
            # mkdir -p's it BEFORE writing any level, so an ingest killed partway
            # (crash, cancel, full disk) leaves an empty or torn store behind. Trusting
            # isdir() there dead-ends every later load on "no level_*.b2nd store" —
            # which also masks whatever actually failed on the first attempt. Probe it,
            # and fall back to a re-ingest: the ND2/TIFF is the source of truth and
            # write() rebuilds the levels with mode="w".
            #
            # `open_store` also runs the V2.20 chunk census on an unmarked level 0
            # (`verify_store`), so a store torn BEFORE the completeness marker existed
            # lands here too rather than silently serving zeros for the frames it never
            # got — which is exactly what the lab's 84.7 GB 640 series was doing.
            prov = None
            if os.path.isdir(store):
                try:
                    prov = open_store(store)
                except Exception:  # noqa: BLE001 — empty dir / torn or short-written
                    prov = None    # level_0 / bad frame ⇒ re-ingest below
            # The FIRST pull of a file pays the whole one-time ingest (read the volume,
            # then compress the pyramid) — minutes for a big series, and before this the
            # card just sat 'queued' with nothing moving. Report it through the SAME
            # observer a compute's ctx.progress uses, so it gets that path's per-node
            # throttle and stale-epoch guard for free. The pyramid repair below borrows the
            # same channel: it is the same kind of one-time cost on the same card.
            if observe is None:
                observe = self._make_observer(self._epoch if epoch is None else epoch)

            def on_ingest(fraction: float, note: str) -> None:
                f = max(0.0, min(1.0, float(fraction)))
                # done/total must be REAL numbers: _make_observer treats done == total as
                # the final update and exempts it from throttling, so leaving them absent
                # (None == None) would defeat the throttle.
                observe("progress", node_id,
                        {"fraction": f, "note": note,
                         "done": int(round(f * 1000)), "total": 1000})

            if prov is not None:
                # A store can be SOUND but SHORT: an ingest that died between pyramid
                # levels leaves a complete level_0 and no pyramid, and nothing about
                # opening it says so — which is how the lab's 84.7 GB series ended up
                # displaying every zoom level off full-res planes. Level l is a pure
                # function of level l-1, so this repairs in place without reading the
                # source file, and it is memo-neutral (identity comes from level 0).
                if prov.levels < PYRAMID_LEVELS:
                    try:
                        prov = ensure_store_levels(
                            store, PYRAMID_LEVELS,
                            progress=lambda f: on_ingest(
                                f, f"completing pyramid for "
                                   f"{os.path.basename(path)}"))
                    except Exception:  # noqa: BLE001 — a short pyramid still DISPLAYS
                        pass           # (stride-decimated); never fail a pull over it
                env = MetaEnvelope(axes=prov.axes, metadata=read_calibration(path))
            else:
                prov, env = ingest_image(path, store_path=store,
                                         levels=PYRAMID_LEVELS, progress=on_ingest)
            disp = read_channel_display(path)
        self._providers[key] = (prov, env)
        self._channel_display[key] = disp
        return prov, env


__all__ = ["EngineRunner", "ensure_gui_ops", "render_plane", "render_plane_native",
           "PlaneCache", "ingest_workers", "source_key"]
