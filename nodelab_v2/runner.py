"""EngineRunner — canvas → Engine off the UI thread (G7, LOCKED 2026-07-22: one worker
thread + **epoch registry**, no qasync — the engine is synchronous CPU work, so a worker
thread + queued-signal delivery is the whole bridge; stale results are dropped by epoch on
arrival. One pull runs at a time; the branches asked for behind it are QUEUED, each with
its own run id, so several can be in flight from the user's point of view and the first to
land is viewable and editable while the rest are still going.

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
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from nodegraph.checkpoint import checkpoint_envelope
from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.engine import Engine, PullCancelled
from nodegraph.graph import Graph
from nodegraph.memo import Memo
from nodegraph.metadata import (
    MetaEnvelope, PER_POSITION_KEYS, SOURCE_FILE_KEY, position_subset,
    stamp_source_file)
from nodegraph.parallel import (
    cpu_budget, memo_bytes, plane_cache_bytes, ram_budget, store_dir, tile_cache_bytes)
from nodegraph.provider import (
    FrameSliceProvider, FrameSubsetProvider, MultiSourceProvider, SyntheticProvider,
    _picked, subset_index)
from nodegraph.streaming import StreamProvider, TileCache
from nodelab_v2.document import BUNDLE_PATHS_KEY
from nodelab_v2.ops import (ACCESS_AUTO, ACCESS_DIRECT, ACCESS_INGEST, ACCESS_MODE,
                            CALIB_OVERRIDE_KEYS, GROUPING_AUTO, GROUPING_DEFAULT,
                            GROUPING_MODE, LOAD_OP,
                            calib_overrides, dock_seeds, dock_store_of,
                            source_access_of)

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


def source_key(path: str, access: str = ACCESS_INGEST) -> Any:
    """The provider-cache key for a source path — the identity a store, a provider and an
    in-flight ingest are all shared under. An empty path is the synthetic demo source.

    ``access`` (``io.load``'s mode) is part of the identity because the two modes build
    genuinely different providers over the same bytes — a
    :class:`~nodegraph.provider.B2ndProvider` over a store versus a
    :class:`~nodelab_v2.nd2_direct.Nd2DirectProvider` over the file — and they carry
    different fingerprints. Sharing one cache slot would serve whichever was resolved
    first and make the mode look like it had done nothing.

    The ingest key is left **byte-identical** to the pre-mode tuple rather than growing a
    uniform ``(…, access)`` tail. Every saved graph and every store on disk means
    ``ingest``, so the default must not become a different key than it was.

    For a REAL path, ``access`` must already be a CONCRETE choice — never
    :data:`ACCESS_AUTO` itself. ``"auto"`` names a decision, not a provider, so keying on
    the literal word would open a third cache slot belonging to neither real provider type
    and sharable by nothing. :meth:`EngineRunner._effective_access` is where ``auto`` gets
    resolved before it ever reaches here; this function's own ``access=ACCESS_INGEST``
    default is an implementation fallback for a caller with no opinion — it is NOT
    :data:`~nodelab_v2.ops.ACCESS_DEFAULT` (``"direct"``, the mode's actual default), which
    a caller with a real ``io.load`` record should read via :func:`source_access_of`
    instead of relying on this function's default to guess it. An EMPTY path is exempt —
    the synthetic source has no file to decide anything about, so ``io.load``'s default
    access (``"direct"``, unresolved) reaching here for the demo source is not a caller's
    mistake."""
    if not path:
        return ("synthetic",)
    assert access != ACCESS_AUTO, "source_key() needs a RESOLVED access, not 'auto'"
    key = ("image", os.path.abspath(path))
    return key if access == ACCESS_INGEST else key + (str(access),)


def bundle_key(paths: Sequence[str], access: str = ACCESS_INGEST) -> Any:
    """The provider-cache key for a file BUNDLE. Order is part of the identity because it
    decides which file each multipoint index addresses, so the same files bundled the other
    way round must not share a cached provider.

    ``access`` joins it for the reason :func:`source_key` gives, and is appended the same
    way — only when it is not the default — so an existing bundle keeps its key.

    Unlike :func:`source_key`, ``access`` here is the card's own RAW setting and MAY be
    ``"auto"`` — this key only has to be stable and distinct for the bundle's own
    :class:`~nodegraph.provider.MultiSourceProvider` cache slot, never mind what each
    member resolves to underneath; each member's own key (built by
    :meth:`EngineRunner._resolve_one`, one call per file) is what actually distinguishes
    ingest from direct, and members can resolve differently from each other."""
    key = ("bundle", tuple(os.path.abspath(p) for p in paths))
    return key if access == ACCESS_INGEST else key + (str(access),)


def _with_card_calib(env: MetaEnvelope, cfg: Mapping[str, Any]) -> MetaEnvelope:
    """``env`` with the source card's own calibration override applied (V2.29).

    Applied HERE — after :meth:`EngineRunner._resolve_one`, never inside it — because that
    cache is keyed on ``(path, access)`` and shared: two cards opening the SAME file with
    different Z steps must not serve each other's calibration, and the provider (the pixels)
    is genuinely identical for both. So the bytes stay shared and only the envelope forks.

    This is also the ONE place the payload and the engine's meta-seed both come from
    (:meth:`EngineRunner._ensure_engine` builds the seed ``Dataset`` from ``env.metadata``
    and passes the same ``env`` as ``meta_seeds``), which is what keeps the pulled result and
    the edit-time header from disagreeing about the spacing — the lockstep rule an
    axis-changing node's ``meta_transform`` follows, applied to a source.
    """
    over = calib_overrides(cfg)
    return env.with_metadata(**over) if over else env


def _with_position_groups(env: MetaEnvelope, path: str, display: Mapping[str, Any],
                          grouping: str = GROUPING_DEFAULT) -> Tuple[MetaEnvelope, str]:
    """``env`` with the per-M grouping stamped on, plus the note the card shows.

    Resolved HERE, once per source, rather than inside the node that selects a group. Three
    reasons, and the last is the one that decides it:

    * the sidecar is a property of the FILE, so the file's reader is where it belongs;
    * every consumer benefits, not just ``util.select_group`` — a measurement table can
      carry which specimen a row came from without anything else knowing how groups are
      found;
    * and this is applied on the same path as :func:`_with_card_calib`, which is the one
      place the payload and the engine's meta-seed both come from. Stamping anywhere else
      would put the grouping in one of them and not the other, and the two disagreeing about
      how many positions a group has is precisely the edit-time/pull-time split an
      axis-changing node's ``meta_transform`` exists to prevent.

    Never fails a load: a file that cannot be grouped simply is not stamped, and
    ``util.select_group`` reports that when (and only when) somebody asks for a group.
    """
    # OFF is the default, and off means NOTHING happens \u2014 no sidecar read, no clustering,
    # no stamp. The lever is checked here rather than inside the detector so that a card
    # nobody has opted in does not pay for, or fail on, work it will not use: this runs on
    # every source of every pull, and a file whose header the SDK struggles with must not
    # be made less openable by a feature that was not asked for.
    if str(grouping or GROUPING_DEFAULT) != GROUPING_AUTO:
        return env, ""
    from nodelab_v2.position_groups import group_metadata, names_agree, resolve_plan
    if int(getattr(env.axes, "m", 0) or 0) <= 0:
        return env, ""
    try:
        plan, note = resolve_plan(path, env.metadata, env.axes)
    except Exception:                    # noqa: BLE001 — grouping is never load-fatal
        return env, ""
    stamped = group_metadata(plan, display.get("position_name"))
    if not stamped:
        return env, note
    agree = names_agree(plan, display.get("position_name"))
    if agree is False:
        note = (note + " — but the acquisition's own point names do NOT restart at these "
                       "boundaries, so the geometry and the point list disagree about where "
                       "one specimen ends. Worth checking before you rely on it.").strip()
    return env.with_metadata(**stamped), note


def _clean_source_paths(cfg: Dict[str, Any]) -> List[str]:
    """A source cfg's member paths, cleaned. Returns 2+ entries only for a real bundle;
    a single-file card (or a bundle someone reduced to one member) returns at most one, so
    every caller can treat ``len(...) >= 2`` as "this is a bundle" without a second flag
    that could disagree with the list."""
    raw = cfg.get(BUNDLE_PATHS_KEY)
    out: List[str] = []
    if isinstance(raw, (list, tuple)):
        out = [p for p in (_clean_source_path(p) for p in raw) if p]
    if len(out) >= 2:
        return out
    fallback = _clean_source_path(cfg.get("path", ""))
    return out or ([fallback] if fallback else [])


def _unique_labels(paths: Sequence[str]) -> List[str]:
    """A short, DISTINCT display name per bundle member — the basename where that is
    already unique, otherwise enough trailing path segments to tell the duplicates apart.

    Two wells exported as ``.../A3/data.nd2`` and ``.../B7/data.nd2`` is the ordinary case,
    not a corner: bare basenames would put the identical name on both files' rows, which is
    worse than a long name because the spreadsheet still looks answerable."""
    names = [os.path.basename(p) or p for p in paths]
    if len(set(names)) == len(names):
        return names
    out: List[str] = []
    for p, name in zip(paths, names):
        if names.count(name) == 1:
            out.append(name)
            continue
        parts = os.path.normpath(os.path.abspath(p)).replace("\\", "/").split("/")
        out.append("/".join(parts[-2:]) if len(parts) >= 2 else name)
    if len(set(out)) == len(out):
        return out
    # still colliding (same parent AND same basename cannot happen, but a UNC/drive edge
    # could) — fall back to the full path, which is always distinct
    return [os.path.abspath(p) for p in paths]


def bundle_envelope(axes: AxisSizes, envs: Sequence[MetaEnvelope],
                    labels: Sequence[str]) -> MetaEnvelope:
    """One envelope for a bundle of K files: the FIRST file's calibration, its multipoint
    lists concatenated across the members, and a ``source_file`` name per position.

    The first file's calibration is the bundle's because the members are required to share
    a grid (:class:`~nodegraph.provider.MultiSourceProvider` refuses otherwise), so pixel
    size, z step and channel optics already agree — taking one is not a choice between
    disagreeing values.

    The per-M lists are the part that can go quietly wrong. ``origin_um`` /
    ``stage_xy_um`` and friends are read POSITIONALLY, so a bundle whose ``m`` runs 0..K·n
    needs a list of that length or every read past the first file lands on another file's
    coordinate. A key that is missing from ANY member, or the wrong length on one, is
    dropped for the whole bundle rather than padded: a partial positional list reports the
    wrong position instead of admitting it does not know (the same rule
    :func:`nodegraph.metadata.position_subset` follows)."""
    md: Dict[str, Any] = dict(envs[0].metadata) if envs else {}
    counts = [int(e.axes.m) for e in envs]
    for key in PER_POSITION_KEYS:
        if key == SOURCE_FILE_KEY:
            continue
        vals: List[Any] = []
        for env, n in zip(envs, counts):
            got = env.metadata.get(key)
            if not isinstance(got, (list, tuple)) or len(got) != n:
                vals = []
                break
            vals.extend(got)
        if vals:
            md[key] = vals
        else:
            md.pop(key, None)
    md[SOURCE_FILE_KEY] = [labels[i] for i, n in enumerate(counts) for _ in range(n)]
    return MetaEnvelope(axes=axes, metadata=md)

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

#: How many FINISHED branch results to keep for instant re-viewing
#: (:attr:`EngineRunner._results`). Four covers the shape this exists for — a handful of
#: per-channel branches off one file, switched between while the slowest is still going —
#: without turning the runner into a second, unbudgeted memo. The payloads are lazy, so the
#: cost is the entry, not the pixels; what the cap really bounds is how long an eager node's
#: realized raster is held against the Memo's own byte budget.
_FINISHED_RESULTS = 4

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


#: Bytes of GPU texture the shown channels may occupy at once. The uploader is **GL_R16**
#: (:meth:`nodelab_v2.glview.GLImageView._upload` — one normalized 16-bit channel per texture;
#: it packed RGBA8 at 4 bytes/px until 2026-08-10, when the repack itself was measured as the
#: playback cost), so one frame costs ``2 * y * x`` of VRAM plus transient CPU copies of the
#: same size (``pack_u16`` + ``tobytes``): a 7168² plane is 103 MB per channel, a 13106² mosaic
#: 344 MB. ``NODELAB_TEXTURE_BYTES`` overrides.
#:
#: This exists because ``GL_MAX_TEXTURE_SIZE`` is the wrong ceiling to trust — this GPU reports
#: 32768, which would permit a 4 GB texture. Uploads are not error-checked (there is no
#: ``glGetError`` on that path), so exceeding VRAM does not raise: it leaves the previous
#: texture's content bound, which is a frame with part of the picture missing.
TEXTURE_BYTES = int(os.environ.get("NODELAB_TEXTURE_BYTES", "") or 512 * 1024 * 1024)


def display_cap(axes: Any, *, texture_limit: int, bytes_per_px: int = 8,
                planes: int = 1) -> int:
    """The display cap for a frame of ``axes``: its own long edge when the whole thing can be
    shown at FULL resolution *affordably*, else :data:`MAX_DISPLAY_DIM`.

    Three ceilings, and a frame has to clear all of them:

    * the **texture limit** in px, because the overview is one texture per channel;
    * the **texture BYTES** those channels cost (:data:`TEXTURE_BYTES`) — the limit that
      actually bites, see its note;
    * the **RAM budget** (:func:`display_ram_bytes`), because these frames land in the
      :class:`PlaneCache` and playback wants a series of them resident, not one.

    Cost to produce is deliberately NOT a ceiling any more (it was V2.23b's fourth, dropped
    2026-08-10 — "I want the image to appear at its native resolution at all times"). A
    ``StreamProvider`` frame is a compute — level 0 of the WellA3 mosaic is 1.0 s against
    level 1's 0.21 s — and that once justified pinning every live mosaic to the pyramid.
    Two things changed underneath it: pressing ▶ now materializes the whole series to RAM
    behind a held, cancellable prepare (so the per-frame cost is paid once, visibly, and
    playback re-uses the resident planes), and playback no longer runs the detail patch
    that was the pyramid's zoomed-in alibi. A cold scrub does pay the level-0 read, on the
    worker, with the card showing "reading planes" — slower and honest, like every other
    lazy read in this app. The affordability ceilings above still refuse what the surface
    or the budget genuinely cannot hold. The budget charges the uploader's REAL texel cost
    — 2 bytes/px since the R16 upload (2026-08-10) — and keeping the retired RGBA8 packing's
    4 here was a live bug: it refused a single-channel 13106² whole-well canvas (687 MB
    budgeted, 343 MB actual — "stitch is still pixelated", 2026-08-25). Two shown channels
    of that canvas DO exceed the default; ``NODELAB_TEXTURE_BYTES`` is the knob, VRAM
    permitting.

    ``bytes_per_px`` is deliberately pessimistic by default (8 — float64, what a stitch canvas
    serves): budgeting may be conservative, and over-committing is the failure that matters.

    Below the cap nothing changes: a camera frame was never decimated and is unaffected.
    """
    long_edge = max(1, int(getattr(axes, "y", 1)), int(getattr(axes, "x", 1)))
    if long_edge <= MAX_DISPLAY_DIM:
        return MAX_DISPLAY_DIM                # nothing to decide; it fits either way
    px = int(axes.y) * int(axes.x) * max(1, int(planes))
    if (long_edge <= int(texture_limit)
            and px * 2 <= TEXTURE_BYTES
            and px * int(max(1, bytes_per_px)) <= display_ram_bytes()):
        return long_edge
    return MAX_DISPLAY_DIM

#: How many channels ONE overlay source may contribute to the display.
#:
#: The ceiling is the GL sampler bank (``nodelab_v2.glview._MAX_CH`` = 8), shared with the
#: primary's own channels — so this is deliberately small rather than "whatever the file
#: has". A 2-channel primary plus a 2-channel secondary is 4 of the 8; letting a 6-channel
#: secondary in would silently push the primary's own channels out of the shader.
#:
#: 3 since 2026-10-01: an Experiment Canvas primary (``view.canvas``) is ONE blank channel,
#: and three overlaid files of a real well (2 + 2 + 3 channels — GFP / R-B / Nile Blue on the
#: third) then fit the bank exactly, where a cap of 2 silently dropped Nile Blue. The
#: primary's channels are still reserved first (`_resolve_overlay`'s budget), so this can
#: crowd out a LATER source — announced, never silent — but never the primary.
MAX_OVERLAY_CHANNELS = 3

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
               cap: int, sub: int = 0, ovr: int = 0) -> tuple:
    """The one spelling of a :class:`PlaneCache` address.

    It exists because there were two. `_plane_addrs` built a 7-tuple ending in the display
    cap — for the reason its own comment gives, that several readers write these keys and a
    disagreement must be a miss rather than a plane served at the wrong size — while
    `_decode_planes`' two fallback branches built a 6-tuple without it. Those writes landed
    in slots `_cached_planes` never probes, so the plane was re-decoded (or the overlay
    re-composed) on every scrub. Worse, the fallback also omitted `max_dim=cap` on the read,
    so with a cap above :data:`MAX_DISPLAY_DIM` — every docked mosaic that fits the texture
    and RAM budgets — the shape-reference plane came back decimated to 4096 and the overlay
    was composed onto a grid the primary channels do not share.

    ``sub`` / ``ovr`` exist only for COMPOSED overlay channels (2026-09-30): the Play-all
    sub-tick inside primary frame ``t`` (a 4x-faster source shows a different frame on each of
    four ticks of one primary frame), and the generation of the Viewer's per-source frame
    override (its ◀▶ steppers — display-only, and so must never be served for the un-overridden
    frame). Both are appended only when non-zero, so every primary key — and every overlay key
    at rest — is the 7-tuple it always was, and the primary plane is shared by every sub-tick."""
    base = (node_id, pin, int(m), int(t), int(z), int(ch), int(cap))
    return base + ((int(sub), int(ovr)),) if (sub or ovr) else base


def self_is_canvas_node(graph, node_id: str) -> bool:
    """Whether ``node_id`` OUTPUTS an experiment canvas — a ``view.canvas``, or an Overlay
    in ``canvas=union`` mode. A union Overlay on top of one grows that canvas rather than
    drawing it as a picture of its own."""
    node = graph.nodes.get(node_id) if graph is not None else None
    if node is None:
        return False
    if node.op_key == "view.canvas":
        return True
    return (node.op_key == "view.overlay"
            and dict(getattr(node, "modes", None) or {}).get("canvas") == "union")


def _key_sub(key: tuple) -> Tuple[int, int]:
    """``(sub, ovr)`` of a :func:`_plane_key` (``(0, 0)`` for a 7-tuple)."""
    return tuple(key[7]) if len(key) > 7 else (0, 0)


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


class _HeldView(NamedTuple):
    """One node's held display state (V2.28) — everything the coords-only fast path
    needs to serve that node's planes without re-pulling, as ONE immutable snapshot.

    A snapshot on purpose: these five values are only correct *together* (the dtype
    decides the display cap that is folded into every plane key, the pin decides how a
    global cursor addresses the payload), and while they were five separate ``_viewer_*``
    attributes a worker writing one of them mid-scrub could pair another node's dtype
    with this node's provider. Kept per NODE (:attr:`EngineRunner._views`) so the Viewer's
    side-by-side compare pane can scrub two results without each cursor move evicting the
    other pane's fast path."""

    provider: Any            # the payload's image provider
    axes: Any                # its AxisSizes
    rev: int                 # document.revision the provider belongs to
    pin: Optional[Pin]       # the solo-frame scope it was pulled under
    dtype: Any               # display narrowing dtype (:func:`_display_dtype`), or None


#: how many nodes' display state is held at once — one per Viewer pane (the viewed node
#: and the compare pane's). Not a cache: an evicted node is re-armed from
#: :attr:`EngineRunner._results` in O(1) (:meth:`EngineRunner._rearm_view`), so this only
#: bounds how many payload providers are pinned against the Memo's budget.
HELD_VIEWS = 2

#: "not passed" marker for ``dtype`` params, where ``None`` is a real value (an
#: un-narrowed display copy) and the default is "read the node's held view".
_UNSET = object()

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


def _doc_id_of(run_id: str) -> str:
    """The document node id a RUN-graph id answers to.

    The run graph renames two kinds of node: an Iterate clone is
    ``{doc_id}#{iterate_id}@{i}`` (and the zone's synthetic advance node
    ``{iterate_id}#adv@{i}`` belongs to the Iterate card itself), and an inlined group
    body node is ``{body_id}%{instance_id}`` — nesting appends further ``%instance``
    segments, and the LAST one is the instance that actually sits on the canvas.
    Everything else passes through unchanged. Used to store run cones in document
    terms, so a delete or edit of a card matches the runs computing its clones."""
    head = run_id.split("#", 1)[0]
    return head.rsplit("%", 1)[-1] if "%" in head else head


class _Job:
    __slots__ = ("epoch", "graph", "revision", "node_id", "pull_id", "coords", "channels",
                 "sources", "all_sources", "pin", "bake", "cancelled")

    def __init__(self, epoch: int, graph: Graph, revision: int, node_id: str,
                 coords: Optional[Tuple[int, int, int, int]],
                 channels: Optional[Tuple[int, ...]],
                 sources: Dict[str, Dict[str, Any]],
                 pin: Optional[Pin] = None,
                 bake: Optional[Dict[str, Any]] = None,
                 all_sources: Optional[frozenset] = None,
                 pull_id: Optional[str] = None) -> None:
        self.epoch = epoch
        self.graph = graph
        self.revision = revision
        self.node_id = node_id
        # The id actually pulled from the run graph, which differs from `node_id` for a
        # node INSIDE an Iterate segment: that node exists only as per-iteration clones, so
        # the card the user clicked is served by one of them (V2.22,
        # `nodegraph.iterate.aliases`). Everything else — progress, delivery, the viewer's
        # provider handle — stays keyed on `node_id`, the card that was clicked.
        self.pull_id = pull_id or node_id
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
        # Set (GUI thread) by `invalidate` when this run is cancelled while ON the worker;
        # read (worker thread) through `Engine.should_stop`, so the pull aborts at its next
        # node boundary or progress tick instead of grinding out a payload nobody will
        # accept. A plain bool under the GIL — no lock needed for a latch that only ever
        # goes False→True. Lives on the JOB, not the runner, so closures a finished pull
        # leaves behind (lazy providers in the memo) can never be tripped by a later
        # cancellation: their flag is this job's, frozen in whatever state it ended.
        self.cancelled = False


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

    It costs nothing in concurrency: one pull runs at a time by design (``_busy``, resolved
    on the GUI thread), and the branches waiting behind it sit in :attr:`EngineRunner._queue`
    rather than on another thread. The decode / prefetch / detail jobs stay on the shared
    pool — they read providers rather than owning native per-thread state, and serializing
    them would undo the display fast path.

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
            # Cooperative cancel: `invalidate` latches `job.cancelled` when it kills this
            # run (a deleted node, an edit in its cone), and the engine then aborts at its
            # next node boundary / progress tick — freeing the single pull slot for the
            # queue instead of computing a payload `_deliver` would drop anyway. Set
            # unconditionally: the cached engine is reused across pulls, so a stale
            # callable from the previous job must never survive into this one. A bake
            # opts out — its cancel story is `_stop_bake`, and its result is a directory
            # on disk that `_deliver` resolves outside the staleness rule.
            engine.should_stop = None if job.bake is not None \
                else (lambda: job.cancelled)
            if job.bake is not None:
                r._run_bake(engine, job)
                r._done.emit((job.epoch, job.node_id, None, None, None,
                              time.perf_counter() - t0, None, job.revision,
                              None, None, job.pin))
                return
            payload = engine.pull(job.pull_id)
            if job.cancelled:
                # cancelled between the last engine poll and here — don't decode planes
                # for a result that is already dead on arrival
                raise PullCancelled(job.node_id)
            # An overlay node's picture needs a SECOND chain evaluated. Resolved here, on
            # the worker, because pulling the secondary is real work (and normally a memo
            # hit); the compose itself happens per displayed plane in `_decode_planes`.
            # A payload-only fetch (`EngineRunner.fetch`) is never displayed, so it skips it.
            if job.epoch not in r._fetches:
                r._overlay_ctxs[job.node_id] = r._resolve_overlay(
                    engine, job.graph, job.pull_id, payload)
            plane = None                    # dict {channel_index: 2-D native plane}
            axes = None
            if isinstance(payload, Dataset) and payload.image is not None:
                axes = payload.axes
                # The dtype travels WITH this decode rather than through runner state:
                # `_plane_addrs` folds the display cap into every plane key and the cap's
                # byte estimate depends on whether the planes narrow, so the value must be
                # this payload's own whatever the held views do meanwhile. `_deliver`
                # installs the same value in the node's _HeldView, which is what keeps the
                # keys written here and the keys probed later identical.
                if job.coords is not None:
                    # a lazy chain does its real work HERE, under the viewed node's name —
                    # report it as that node's state so the card isn't idle while the
                    # provider reads tiles (the honest counterpart to ctx.progress).
                    r._progress.emit(("decode", job.node_id,
                                      {"epoch": job.epoch, "op_key": ""}))
                    plane = r._decode_planes(payload.image, job.node_id,
                                             job.coords, job.channels, axes,
                                             pin=job.pin, overlay_all=True,
                                             as_dtype=_display_dtype(payload))
            dt = time.perf_counter() - t0
            r._done.emit((job.epoch, job.node_id, payload, plane, axes, dt, None,
                          job.revision, job.coords, job.channels, job.pin))
        except PullCancelled:
            # An aborted pull still delivers a packet — `_deliver` is the ONLY place the
            # pull slot is freed and the queue advanced, so a cancel that skipped it would
            # wedge every branch waiting behind this one. No error rides along: the run's
            # `cancelled` signal already fired from `invalidate`, and the liveness test
            # drops this packet without painting anything.
            r._done.emit((job.epoch, job.node_id, None, None, None,
                          time.perf_counter() - t0, None, job.revision,
                          None, None, job.pin))
        except Exception:  # noqa: BLE001 — full trace to the GUI, never a dead thread
            r._done.emit((job.epoch, job.node_id, None, None, None,
                          time.perf_counter() - t0, traceback.format_exc(),
                          job.revision, job.coords, job.channels, job.pin))


class _IngestJob(QRunnable):
    """One source file's ingest, off the GUI thread and **outside the pull slot** (V2.21).

    A pull is single-slot (queued, not latest-wins) for good reasons (one engine, one held viewer
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

    def __init__(self, runner: "EngineRunner", node_id: str,
                 paths: Sequence[str], key: Any,
                 access: str = ACCESS_INGEST) -> None:
        super().__init__()
        self._r = runner
        self._node_id = node_id
        # a LIST, not a path: a file-bundle card names several files and all of them have
        # to reach disk, or its first pull pays for the rest inside the pull slot — the
        # exact cost this job exists to keep out of it
        self._paths = list(paths)
        self._key = key
        # The card's access mode, carried so the cfg this job hands `_resolve_source`
        # matches the one a pull would build. A `direct` card resolves here too — it just
        # finishes in milliseconds instead of minutes, since there is nothing to write.
        self._access = access

    def run(self) -> None:  # ingest-pool thread
        r = self._r
        t0 = time.perf_counter()
        err = None
        try:
            # An un-epoched observer: an ingest is not part of any pull, so it must not be
            # silenced when the epoch moves on (an edit, or another node being pulled while
            # this runs). Its card reports for as long as it takes.
            cfg: Dict[str, Any] = {"path": self._paths[0] if self._paths else "",
                                   ACCESS_MODE: self._access}
            if len(self._paths) >= 2:
                cfg[BUNDLE_PATHS_KEY] = list(self._paths)
            r._resolve_source(self._node_id, cfg, observe=r._make_observer(None))
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
                 node_id: str, coords, channels, axes, pin: Optional[Pin],
                 dtype: Any = None, sub: int = 0) -> None:
        super().__init__()
        self._r = runner
        self._sub = int(sub)       # the Play-all sub-tick, captured with the coords
        self._gen = gen
        self._epoch = epoch
        self._prov = provider
        self._node_id = node_id
        self._coords = coords
        self._channels = channels
        self._axes = axes
        self._pin = pin
        self._dtype = dtype        # captured with the provider — one _HeldView, one node

    def run(self) -> None:  # worker thread
        r = self._r
        t0 = time.perf_counter()
        packet = (self._gen, self._epoch, self._node_id, self._coords, self._channels,
                  self._pin, self._sub)
        planes = axes = err = None
        try:
            if self._gen == r._decode_gen:        # else: superseded before it ever started
                planes = r._decode_planes(self._prov, self._node_id, self._coords,
                                          self._channels, self._axes, pin=self._pin,
                                          as_dtype=self._dtype, sub=self._sub)
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
                 jobs: List[Tuple[tuple, int, int, int, int]],
                 prov: Any, dtype: Any) -> None:
        super().__init__()
        self._r = runner
        self._gen = gen
        self._jobs = jobs
        # captured at CONSTRUCTION, with the keys: reading the runner's held view at run
        # time could pair another pane's provider with this node's keys if the viewed
        # node changed while this job sat in the pool's queue.
        self._prov, self._dtype = prov, dtype

    def run(self) -> None:  # worker thread
        r = self._r
        prov = self._prov
        if prov is None:
            return
        for (key, m, t, z, ch) in self._jobs:
            if self._gen != r._prefetch_gen:
                return                       # superseded by a newer cursor position
            if r._planes.get(key) is not None:
                continue
            try:
                if ch >= int(getattr(getattr(prov, "axes", None), "c", ch + 1)):
                    # An overlay channel is COMPOSED, not read — the provider has no such
                    # index. Warmed through the same call the displayed frame uses, so the
                    # cache entry it writes is byte-identical to the one it saves.
                    r._warm_overlay(key, m, t, z, ch)
                    continue
                # No decode lock (V2.14). It used to serialize every provider read because
                # the engine's TileCache was unsynchronized; the cache carries its own lock
                # now, so holding one here only forced the prefetch pool to decode one frame
                # at a time — the opposite of what a prefetcher is for.
                # The cap rides in the key (V2.23) — it must, because this writes planes the
                # displayed-frame path then reads, and a prefetcher decimating to a different
                # size would have it serve frames of the wrong shape.
                arr, _lv = render_plane_native(prov, m, t, z, ch, max_dim=int(key[6]),
                                               as_dtype=self._dtype)
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
                 jobs: List[Tuple[tuple, int, int, int, int]],
                 prov: Any, dtype: Any) -> None:
        super().__init__()
        self._r, self._gen, self._jobs = runner, gen, jobs
        # captured at construction, same reason as _PrefetchJob: the keys and the
        # provider must describe the same node whatever the panes do meanwhile
        self._prov, self._dtype = prov, dtype

    def run(self) -> None:  # worker thread
        r = self._r
        prov = self._prov
        for (key, m, t, z, ch) in self._jobs:
            if self._gen != r._preload_gen or prov is None:
                return                       # cancelled: an edit, a stop, or a newer preload
            if r._planes.get(key) is None:
                try:
                    if ch >= int(getattr(getattr(prov, "axes", None), "c", ch + 1)):
                        # a COMPOSED overlay plane - the same warm call the prefetcher uses,
                        # so a preloaded plane is byte-identical to a displayed one
                        r._warm_overlay(key, m, t, z, ch)
                    else:
                        arr, _lv = render_plane_native(prov, m, t, z, ch,
                                                       max_dim=int(key[6]),
                                                       as_dtype=self._dtype)
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
                 rect01: Tuple[float, float, float, float], budget: int,
                 *, pin: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self._r, self._gen, self._node_id = runner, gen, node_id
        self._prov, self._coords, self._channels = prov, coords, list(channels)
        self._axes, self._rect01, self._budget = axes, rect01, int(budget)
        # captured WITH the provider (they are one _HeldView), so a pane switch between
        # queue and run cannot pair this node's provider with another node's pin/dtype
        self._pin, self._dtype = pin, dtype

    def run(self) -> None:                                    # worker thread
        r = self._r
        if self._gen != r._detail_gen:
            return
        try:
            planes, rect01 = r.detail_planes(
                self._prov, self._node_id, self._coords, self._channels, self._axes,
                self._rect01, self._budget, pin=self._pin, dtype=self._dtype)
        except Exception:                    # noqa: BLE001 — detail is best-effort; the
            return                           # overview is already on screen and correct
        if self._gen == r._detail_gen and planes:
            # coords ride in the packet so the panel can refuse a patch that outlived its
            # frame — the generation only advances on a NEW request, and during playback
            # (or a fast scrub) the frame moves on without one, so a windowed read that
            # took longer than the frame cadence would land with a CURRENT generation and
            # paint the previous timepoint's pixels over the new frame (2026-08-10,
            # "the frames are going back to previously loaded frames").
            r._detail_done.emit((self._gen, self._node_id, planes, rect01,
                                 tuple(self._coords)))


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
    #: a PAYLOAD-ONLY pull finished (:meth:`fetch`): ``(node_id, payload, seconds)``. Unlike
    #: ``finished`` it never reaches a Viewer pane, holds no view and decodes no planes — it
    #: is how the Movie Editor gets a source's Dataset without retargeting what is on screen.
    fetched = Signal(str, object, float)
    #: a fetch was submitted to the pull slot: ``(node_id)``. The counterpart to ``started``,
    #: kept separate because ``started`` marks every pane showing that node as running, and
    #: a fetch never delivers the ``finished`` that would clear it.
    fetch_started = Signal(str)
    #: a viewport detail patch is ready: ``(node_id, {channel: plane}, (x0,y0,x1,y1))``
    #: with the rect in NORMALIZED image coordinates, so it is independent of both the
    #: patch's own resolution and the overview's.
    detail_ready = Signal(str, object, object, object)   # node, planes, rect01, coords
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
    #: a pull was QUEUED behind one already running (2026-08-06): ``(node_id, depth)``.
    #: The card's own "waiting" state — the counterpart to ``started``/``finished``, and the
    #: reason a second branch is now visibly pending rather than invisibly discarded.
    queued = Signal(str, int)
    #: a queued or running pull was DROPPED without producing a result ``(node_id)`` —
    #: an edit landed inside its cone (:meth:`invalidate`), or the node/graph went away.
    #: A card must be able to leave the running state on this path too, or it spins forever.
    cancelled = Signal(str)

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
        #: path → the concrete access (`"ingest"`/`"direct"`) an `access=auto` card
        #: resolved to, decided once per path and never revisited this session
        #: (:meth:`_effective_access`) — plus the human-readable reason, for
        #: :meth:`auto_access_reason`.
        self._auto_access: Dict[str, str] = {}
        self._auto_access_reason: Dict[str, str] = {}
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
        #: The pull QUEUE — every branch the user asked for, in the order they asked
        #: (2026-08-06). It was one slot, latest-wins: requesting a second branch while the
        #: first ran did not queue it, it REPLACED whatever was waiting. So asking for two
        #: segmentations gave you one result and one silently dropped request, which is what
        #: made two per-channel branches look like they could not both be run.
        #:
        #: De-duplicated by ``node_id`` rather than appended blindly: clicking the same card
        #: twice while it waits means "show me that node", not "compute it twice", so a repeat
        #: request UPDATES the queued entry's coords/channels and keeps its place. That also
        #: keeps the old latest-wins behaviour for the case it was actually right for — a user
        #: scrubbing the cursor over one waiting node.
        self._queue: "OrderedDict[str, Tuple[str, Any, Any]]" = OrderedDict()
        #: run id → the node it is computing, for every pull that is RUNNING or queued.
        #: Replaces "stale means not the newest epoch" with "stale means this run is no longer
        #: live", which is the whole difference between one pull at a time and several
        #: branches in flight: a second branch finishing must not retire the first's result,
        #: and an edit to one branch must not silently discard the other's.
        self._runs: Dict[int, str] = {}
        #: run id → the set of node ids that run computes (its cone), so an edit can cancel
        #: exactly the runs it can affect and leave the rest alone. Without it "an edit"
        #: means "every in-flight pull dies", which is precisely what stops you from
        #: adjusting a finished branch while another one is still going.
        self._run_cones: Dict[int, frozenset] = {}
        #: the :class:`_Job` currently ON the worker thread (None before the first pull).
        #: `invalidate` latches its ``cancelled`` flag when it kills that run, which is what
        #: turns a logical cancellation into an actual abort: the engine polls the flag at
        #: node boundaries and progress ticks and unwinds instead of finishing a result
        #: nobody will accept — a deleted node's half-hour segmentation used to keep the
        #: pull slot to the end, with every queued branch waiting behind it.
        self._active: Optional[_Job] = None
        #: ``(node_id, document revision)`` → ``(payload, axes)`` for the last few branches
        #: that FINISHED — so clicking a completed branch while another one is still computing
        #: shows it at once instead of joining the queue behind a job that may take minutes
        #: (2026-08-06). Without it "you can view the finished one" is only true once
        #: everything else has stopped, which is not the ask.
        #:
        #: Bounded and revision-keyed. The payloads are lazy Datasets whose pixels the Memo
        #: already holds, so this pins little beyond what a re-pull would hit anyway; the cap
        #: is what stops a long session from holding every branch's label raster against the
        #: Memo's own budget. An edit bumps the revision, so a stale entry is never reachable
        #: — it just ages out.
        self._results: "OrderedDict[Tuple[str, int], Tuple[Any, Any]]" = OrderedDict()
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
        #: node_id → its held display state (:class:`_HeldView`), most-recent LAST.
        #: Capacity :data:`HELD_VIEWS` — one per Viewer pane, so the side-by-side compare
        #: can scrub both results on the fast path at once.
        self._views: "OrderedDict[str, _HeldView]" = OrderedDict()
        #: node_id → its resolved secondary chain (see `_resolve_overlay`), or ``None`` for
        #: a node that overlays nothing. Per NODE for the same reason :attr:`_views` is
        #: (V2.28): with two Viewer panes open, a single slot held only the node pulled
        #: LAST, so the other pane's composed overlay channels dropped out of
        #: :meth:`_plane_addrs` on its next scrub — the overlay vanished from the pane
        #: nobody had touched.
        #:
        #: The worker only ever assigns one key (an atomic dict store); the GUI thread does
        #: the bounded trim in :meth:`_deliver`, so no reader can see a half-evicted map.
        self._overlay_ctxs: Dict[str, Optional[Dict[str, Any]]] = {}
        #: source node id → {channel: {"clim", "bit_depth"}} for overlay channels. One
        #: plane read of the SOURCE, cached because it never changes for that source.
        self._src_lut_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        #: overlay node id → ``(dt, dz)``: the Viewer's per-source ◀▶ steppers, a DISPLAY-ONLY
        #: move of that source's frame away from its mapped one, so the user can find the frame
        #: that goes with this one before pinning it. Never written to the document; composed
        #: planes made under it are keyed by ``_ovr_gen`` (:func:`_plane_key`), which advances on
        #: every change, so they can never be served for the un-overridden frame.
        self._src_override: Dict[str, Tuple[int, int]] = {}
        self._ovr_gen: int = 0
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
        self._decode_pending: Optional[Tuple[Any, ...]] = None
        # ── viewport detail-on-demand (see :class:`_DetailJob`) ───────────────────
        #: bumped on every new request AND on every pull/scrub, so a patch that arrives
        #: for a rect (or a frame) the user has already left is dropped instead of being
        #: painted over the wrong place.
        self._detail_gen = 0
        #: per source node, what the grouping resolution decided — read by the card so a
        #: detected split and one a person wrote are not shown as the same claim.
        self._group_note: Dict[str, str] = {}

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
        #: epochs of the in-flight payload-only pulls (:meth:`fetch`), and the node ids
        #: waiting for one. Fetches queue BEHIND every pull the user asked for: they feed an
        #: editor's preview, which must never delay the result someone is waiting to see.
        self._fetches: set = set()
        self._fetch_queue: "OrderedDict[str, None]" = OrderedDict()
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
        paths = _clean_source_paths(dict(rec.params))
        if not paths:
            return "synthetic"
        if self._source_key_of(paths, source_access_of(rec)) in self._providers:
            return "ready"
        # a bundle is only as ready as its coldest member, and only as loadable as its
        # most missing one — reporting "cold" for a bundle with a deleted member would
        # start a job that can only fail
        if any(not os.path.isfile(p) for p in paths):
            return "missing"
        return "cold"

    def _source_key_of(self, paths: Sequence[str], access: str = ACCESS_INGEST) -> Any:
        """The provider-cache key a source card's paths resolve under — one place, so the
        state check, the ingest job and the pull cannot disagree about a card's identity.

        ``access`` has to come along for exactly that reason: it is part of the key
        (:func:`source_key`), so omitting it here would have :meth:`source_state` probing
        the ingest slot for a card set to direct and reporting ``"cold"`` for a file that
        is already open.

        A single file's key is RESOLVED (:meth:`_effective_access`) the same way
        :meth:`_resolve_one` resolves it, for the same reason — the key has to name the
        provider that is actually cached, not the word ``"auto"``. A bundle's key is left
        literal, matching :meth:`_resolve_bundle`: it only has to be stable and distinct,
        and each member resolves independently underneath it anyway."""
        if len(paths) >= 2:
            return bundle_key(paths, access)
        path = paths[0] if paths else ""
        return source_key(path, self._effective_access(path, access) if path else access)

    def ingest_source(self, node_id: str) -> str:
        """Ingest the file an ``io.load`` card names — **now, concurrently, and without
        taking the pull slot**. Returns the same vocabulary as :meth:`source_state`, plus
        ``"started"`` (a job was queued for it) and ``"joined"`` (another card already has
        this same file in flight; this one will finish with it).

        This is what a double-click on a source card does. The alternative — pulling it —
        works, but it runs the ingest *inside* the single pull slot, so a second
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
        paths = _clean_source_paths(dict(rec.params))
        access = source_access_of(rec)
        key = self._source_key_of(paths, access)
        joined = key in set(self._ingesting.values())
        self._ingesting[node_id] = key
        self.ingest_started.emit(node_id)
        if joined:
            return "joined"
        self._ingest_pool.start(_IngestJob(self, node_id, list(paths), key, access))
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

    def display_dim(self, axes: Any, *, planes: int = 1, provider: Any = None,
                    dtype: Any = None) -> int:
        """The display cap for a node's frames — full resolution when one frame of
        every shown channel is affordable in texture and in RAM.

        ``dtype`` decides whether the display copy narrows (half the bytes per pixel);
        it is passed EXPLICITLY — from the node's :class:`_HeldView`, or by
        :meth:`_decode_planes`, which runs on the worker DURING the pull that will later
        install that view — because the answer is folded into every plane key and reading
        another node's held state here would have two panes disagree about a key's shape.

        ``provider`` no longer moves the answer (2026-08-10 — cost-to-produce was dropped
        as a ceiling, see :func:`display_cap`); it stays in the signature because every
        call site and test harness passes it, and it is the hook any future
        provider-shaped cap decision would hang off."""
        if axes is None:
            return MAX_DISPLAY_DIM
        return display_cap(axes, texture_limit=self._texture_limit,
                           bytes_per_px=(2 if dtype is not None else 8),
                           planes=planes)

    # ── viewport detail-on-demand ─────────────────────────────────────────────
    def request_detail(self, node_id: str, coords, channels, rect01, budget: int) -> bool:
        """Ask for the visible rect of ``node_id`` at full detail. Returns whether a job
        was queued. GUI thread; the read itself runs on the pool (:class:`_DetailJob`).

        Latest-wins by generation rather than by a queue: while the user is still zooming,
        every intermediate rect is dead on arrival, and rendering them in order would just
        put the pool behind the cursor."""
        view = self._view_of(node_id) or self._rearm_view(node_id)
        if view is None or view.provider is None or view.axes is None:
            return False
        self._detail_gen += 1
        self._pool.start(_DetailJob(self, self._detail_gen, node_id, view.provider,
                                    coords, channels, view.axes, rect01, budget,
                                    pin=view.pin, dtype=view.dtype))
        return True

    def invalidate_detail(self) -> None:
        """Drop any in-flight detail patch — the frame, the node or the graph moved, so
        whatever is being read is about to describe something that is no longer shown."""
        self._detail_gen += 1

    def detail_planes(self, prov, node_id, coords, channels, axes, rect01, budget,
                      *, pin: Optional[Pin] = None, dtype: Any = None):
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
            out[ch] = _fit_plane(arr, budget, as_dtype=dtype)
        # the rect the pixels REALLY cover, in normalized coords
        snapped = (lx0 / lax.x, ly0 / lax.y, lx1 / lax.x, ly1 / lax.y)
        octx = self._overlay_ctxs.get(node_id)
        if out and octx is not None:
            ref = next(iter(out.values()))
            composed = self._compose_overlay(
                node_id, ref.shape[:2], m, t,
                _z_um_of(octx["pri_md"], octx["pri_axes"], m, z),
                # the SNAPPED rect, in the (fy0, fy1, fx0, fx1) order placement uses: the
                # patch's pixels cover that box and not the one that was requested
                region=(ly0 / lax.y, ly1 / lax.y, lx0 / lax.x, lx1 / lax.x),
                # detail is requested at REST (playback holds it off), so sub-tick 0 — but
                # the stepper override and Z index must match the overview underneath it
                z_index=z, override=dict(self._src_override))
            # the caller asked for a channel set; an overlay channel whose toggle is off is
            # not in it, and the patch must not put back what the overview leaves out
            want = {int(ch) for ch in (channels or ()) if int(ch) >= int(axes.c)}
            out.update({ch: pl for ch, pl in composed.items()
                        if not channels or ch in want})
        return out, snapped

    def _deliver_detail(self, packet) -> None:            # GUI thread
        gen, node_id, planes, rect01, coords = packet
        if gen == self._detail_gen:
            self.detail_ready.emit(node_id, planes, rect01, coords)

    # ── public API (GUI thread) ────────────────────────────────────────────────
    def pull(self, node_id: str,
             coords: Optional[Tuple[int, int, int, int]] = None,
             channels: Optional[Tuple[int, ...]] = None,
             *, queue: bool = True) -> None:
        """Request a full engine pull (+ optional display planes for ``channels``).
        Establishes/refreshes the held viewer provider.

        **Queued, not latest-wins** (2026-08-06). A request made while another pull is
        running joins :attr:`_queue` and is run in turn; it does not replace what was
        waiting. Asking for a second per-channel branch used to drop whichever request was
        already pending, so a two-branch graph could only ever produce one result — the
        "can't both run" half of the two-channel report. A repeat request for a node ALREADY
        queued updates that entry in place rather than adding a second (see :attr:`_queue`).

        Requests are independent of each other from here on: each gets its own run id, each
        card reports its own state, and the first to land is viewable and editable while the
        rest are still going (:meth:`invalidate`).

        ``queue=False`` is for a request the user did not ASK to compute — click-to-preview,
        which fires on every settled selection while the canvas is maximized (2026-08-06).
        Such a request is served if the answer is already in hand and otherwise DROPPED. The
        old single slot was latest-wins, which made previewing free: clicking around while
        something ran only ever replaced one pending entry. A queue turned each of those
        clicks into a committed pull, so idly selecting four cards during a long
        segmentation silently signed the machine up for four more — the opposite of a
        preview. A pull the user explicitly asked for (double-click, F5, Run) still queues."""
        if node_id not in self.document.nodes:
            return
        if self._busy:
            if self._serve_finished(node_id, coords, channels):
                return
            if not queue:
                return          # a preview never commits the machine — see the docstring
            # assigning an existing key keeps its position, which is exactly the
            # "keeps its place" rule — no reordering call is needed or wanted
            self._queue[node_id] = (node_id, coords, channels)
            self.queued.emit(node_id, len(self._queue))
            return
        self._submit(node_id, coords, channels)

    def _serve_finished(self, node_id: str, coords, channels) -> bool:
        """Show an already-FINISHED branch straight from :attr:`_results`, without touching
        the pull slot. ``True`` if it was served.

        This is the half of "run one branch, look at the other" that the queue alone does not
        give you: the result is in hand and the Memo holds its pixels, so making the user wait
        behind a half-hour segmentation to see it again would be waiting on nothing. The
        payload is re-delivered as it stands and the PIXELS come through the ordinary decode
        lane (:meth:`request_plane` → the shared pool), which is the same path a scrub uses
        and is already off the pull thread.

        Only for the CURRENT document revision — an edit bumps it, and a re-pull is then
        genuinely required rather than merely slow."""
        key = (node_id, self.document.revision)
        entry = self._results.get(key)
        if entry is None:
            return False
        payload, axes = entry
        self._results.move_to_end(key)               # keep the branches in active use warm
        if isinstance(payload, Dataset) and payload.image is not None:
            self._hold_view(node_id, payload, pin=None)
        self.finished.emit(node_id, payload, None, axes, 0.0)
        if coords is not None:
            # planes off the decode lane, exactly as a cursor move would fetch them
            self.request_plane(node_id, coords, channels)
        return True

    def queue_depth(self) -> int:
        """How many pulls are waiting behind the running one (0 when nothing is queued)."""
        return len(self._queue)

    def in_flight(self, node_id: str) -> bool:
        """Whether a live run (the one on the worker or one waiting behind it) reads
        ``node_id`` — i.e. whether an edit to it now would cancel work in progress."""
        return (any(node_id in cone for cone in self._run_cones.values())
                or node_id in self._queue)

    def finished_result(self, node_id: str) -> Any:
        """``node_id``'s finished, UNPINNED payload at the current document revision, or
        ``None``. A frame-scoped (solo) payload is never returned: it holds only the frames
        it was scoped to, and handing it out as the node's answer would play a truncated
        series."""
        entry = self._results.get((node_id, self.document.revision))
        return entry[0] if entry is not None else None

    def fetch(self, node_id: str) -> None:
        """Compute ``node_id``'s whole Dataset and deliver it on :attr:`fetched` ONLY.

        The pull a normal :meth:`pull` makes goes to the Viewer: it holds a view, decodes
        planes, retargets the primary pane and evicts the held views of whatever was being
        compared. An editor that needs another node's payload in order to preview it must do
        none of that, so a fetch is its own job kind, modelled on the Dock bake: it shares
        the one pull thread and the engine's memo (an already computed chain is a memo hit),
        and is resolved in :meth:`_deliver` before the view-holding path. Never frame-scoped.

        Served at once when the answer is already in hand; otherwise queued behind every
        user-requested pull, and a repeat request while one is waiting is a no-op."""
        if node_id not in self.document.nodes:
            return
        have = self.finished_result(node_id)
        if have is not None:
            self.fetched.emit(node_id, have, 0.0)
            return
        if self._busy:
            self._fetch_queue[node_id] = None
            return
        self._submit(node_id, None, None, fetch=True)

    def queued_nodes(self) -> Tuple[str, ...]:
        """The node ids waiting to be pulled, in the order they will run."""
        return tuple(self._queue)

    # ── held display views (the coords-only fast path's state) ─────────────────
    def _view_of(self, node_id: str) -> Optional[_HeldView]:
        return self._views.get(node_id)

    def _hold_view(self, node_id: str, payload: Dataset, *,
                   pin: Optional[Pin]) -> None:
        """Snapshot ``payload``'s display state for ``node_id`` and keep it hot.
        Evicts the least-recently held node past :data:`HELD_VIEWS` — eviction is cheap
        to undo (:meth:`_rearm_view`), so the cap only bounds pinned providers."""
        self._views[node_id] = _HeldView(payload.image, payload.axes,
                                         self.document.revision, pin,
                                         _display_dtype(payload))
        self._views.move_to_end(node_id)
        while len(self._views) > HELD_VIEWS:
            self._views.popitem(last=False)

    def _rearm_view(self, node_id: str) -> Optional[_HeldView]:
        """Re-establish ``node_id``'s held view from its remembered FINISHED result —
        the O(1) recovery that lets two Viewer panes outlive the :data:`HELD_VIEWS` cap
        and lets a pane scrub a branch that finished while another was running, without
        re-emitting :attr:`finished` or taking the pull slot.

        Unpinned results only, because :attr:`_results` stores nothing else: a scoped
        payload holds only its picked frames and re-serving it as the node's whole
        answer would show a truncated series."""
        entry = self._results.get((node_id, self.document.revision))
        if entry is None:
            return None
        payload, _axes = entry
        if not (isinstance(payload, Dataset) and payload.image is not None):
            return None
        self._results.move_to_end((node_id, self.document.revision))
        self._hold_view(node_id, payload, pin=None)
        return self._views[node_id]

    def request_plane(self, node_id: str,
                      coords: Optional[Tuple[int, int, int, int]] = None,
                      channels: Optional[Tuple[int, ...]] = None, sub: int = 0) -> None:
        """Coords-only request. When ``node_id`` has a held view at the current document
        revision (only the M/T/Z cursor or the active-channel set moved), bypass the
        graph snapshot + ``engine.pull`` entirely and serve the plane straight from the
        :class:`PlaneCache` (decoding a miss synchronously — a single decimated read),
        then warm adjacent frames. A node whose view was evicted but whose FINISHED
        result is still remembered re-arms in O(1) (:meth:`_rearm_view`) — that is what
        lets the compare pane scrub a second result without re-pulling. Otherwise fall
        back to a full :meth:`pull`, which re-establishes the view for this
        (node, revision).

        Under the solo-frame scope the held payload contains only the scoped frames, so
        moving the M/T cursor is normally a change of *what was computed*, not of what is
        displayed: the scope must move with it, which means a real re-pull. The exception
        is a cursor move *within* a multi-frame selection — the pin is unchanged there, so
        it stays on the fast path. Z and channel moves always do, and between them that is
        most of the interactive scrubbing a troubleshooting session does."""
        if node_id not in self.document.nodes:
            return
        if coords is not None:
            view = self._view_of(node_id) or self._rearm_view(node_id)
            if (view is not None and view.provider is not None
                    and self.document.revision == view.rev
                    and self._pin_for(coords) == view.pin):
                # `sub` is the Play-all sub-tick (overlay planes only). A full pull below
                # draws sub-tick 0: it happens at a document change, never mid-playback.
                self._serve_from_cache(node_id, coords, channels, sub)
                return
        self.pull(node_id, coords, channels)

    def invalidate(self, nodes: Optional[Iterable[str]] = None) -> None:
        """Drop the in-flight results a graph edit could have changed, and drop the held
        provider — the decoded-plane cache describes pixels the edit may have moved.

        ``nodes`` names what the edit TOUCHED. Only runs whose cone contains one of them are
        cancelled; every other branch in flight keeps going and still delivers (2026-08-06).
        That is what makes "adjust the finished branch while the other one is still running"
        possible at all: before this, one global epoch meant any edit anywhere — a threshold
        nudged on branch A, a layer renamed, a card moved onto a different channel — silently
        killed branch B's half-hour segmentation, with no error and no card state to show for
        it.

        ``nodes=None`` keeps the old meaning, "assume everything": structural edits that no
        single node accounts for (a file loaded, a graph replaced, an Iterate rewritten) still
        cancel the lot, because a cone computed against the previous graph cannot be trusted
        to describe the new one. Callers that KNOW what changed should say so — the accuracy
        of this is only ever as good as what they pass.

        A cancelled run emits :attr:`cancelled` so its card can leave the running state; a
        result that arrives for a cancelled id is dropped by :meth:`_deliver`'s liveness test.
        Cancelling the run that is ON the worker additionally latches its job's ``cancelled``
        flag, and the engine polls that at every node boundary and progress tick — so the
        compute actually stops and frees the pull slot for the queue, instead of grinding a
        dead branch to completion with every other request waiting behind it. Queued requests
        whose node has left the document are dropped here too (deleting a queued card must
        not leave a phantom entry holding a place in line).
        """
        if nodes is not None and not nodes:
            # An EMPTY set is a positive statement — "this edit changed nothing a run can
            # see" — sent by ``set_meta_seed`` (the G8 source re-seed and the ingest's late
            # envelope), both fired from INSIDE a delivery. The run/queue half of that
            # contract has held since 2026-08-06; this return makes the display half hold
            # too. Falling through wiped the held views, the decoded-plane cache and any
            # preload in flight for an edit that by its own definition moved no pixels —
            # measured on a 146-frame stitched export (2026-08-10): the source envelope
            # resolving a few seconds into playback killed the preload at 46/129 and every
            # later frame re-pulled through the engine at ~3.4 s instead of serving the
            # planes it had already decoded.
            return
        self._views.clear()
        self.invalidate_detail()
        dropped: List[str] = []          # queued requests this call retires
        if nodes is None:
            dead = list(self._runs)
            # Emitted as `cancelled` below rather than discarded silently: a queued card
            # left on "queued" after the queue was drained is a state the runner will
            # never resolve.
            dropped = list(self._queue)
            self._queue.clear()
        else:
            touched = frozenset(nodes)
            dead = [rid for rid, cone in self._run_cones.items() if cone & touched]
            # A run with no recorded cone is one this registry never saw finish registering;
            # treat it as affected rather than assume it is safe.
            dead += [rid for rid in self._runs if rid not in self._run_cones]
        # A queued request whose node has left the document is dead where it stands.
        # `_start_next` would skip it when its turn came, but until then it holds a place
        # in line, inflates `queue_depth`, and — because its card no longer exists — has
        # nothing left to resolve it.
        for qid in [q for q in self._queue if q not in self.document.nodes]:
            del self._queue[qid]
            dropped.append(qid)
        for rid in dead:
            nid = self._runs.pop(rid, None)
            self._run_cones.pop(rid, None)
            act = self._active
            if act is not None and act.epoch == rid and act.bake is None:
                # the cancelled run is the one on the worker: tell the engine to stop.
                # Latched before `cancelled` is emitted so no handler can observe a run
                # that is cancelled on the cards but still uncancellable on the worker.
                act.cancelled = True
            if nid is not None:
                self.cancelled.emit(nid)
        for qid in dropped:
            self.cancelled.emit(qid)
        # The epoch still advances so anything keyed on "the current run" (progress
        # throttling, the observer's stale check) moves on with the graph.
        self._epoch += 1
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
        self._overlay_ctxs.clear()
        # ...and the per-source LUT with them: it is keyed by SOURCE NODE, so editing that
        # node's path would otherwise leave the previous file's window on the overlay.
        self._src_lut_cache.clear()
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
                     *, pin: Optional[Pin] = None, provider: Any = None,
                     dtype: Any = _UNSET, sub: int = 0) -> List[Tuple[tuple, int, int, int, int]]:
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
        if dtype is _UNSET or provider is None:
            view = self._view_of(node_id)
            if dtype is _UNSET:
                dtype = view.dtype if view is not None else None
            if provider is None and view is not None:
                provider = view.provider
        # The display CAP is part of the key (V2.23). It decides the plane's shape, several
        # readers write these keys (the decode, the prefetcher, the raw probe), and the limit it
        # comes from can arrive late — a surface coming up raises it. Keying on it turns any
        # disagreement into a cache miss instead of a plane served at the wrong size.
        cap = self.display_dim(axes, planes=max(1, len(set(wanted))), provider=provider,
                               dtype=dtype)
        out: List[Tuple[tuple, int, int, int, int]] = []
        seen: set = set()
        ovr = self._ovr_gen if self._src_override else 0
        for ch in wanted:
            is_ovl = ch >= int(axes.c)
            if is_ovl:
                if ch not in ovl:
                    continue          # a stale index from a node that no longer overlays
            else:
                ch = min(max(0, ch), axes.c - 1)
            if ch in seen:
                continue
            seen.add(ch)
            # the Play-all sub-tick and the stepper override address OVERLAY planes only —
            # the primary plane is the same pixels on every sub-tick of its frame
            out.append((_plane_key(node_id, pin, m, t, z, ch, cap,
                                   sub if is_ovl else 0, ovr if is_ovl else 0),
                        m, t, z, ch))
        if not out:
            # Every requested index was stale for THIS payload — the Viewer's active set
            # is state from the PREVIOUSLY viewed node, and a payload with fewer channels
            # (a per-channel `chK` tap materializes to c=1) can invalidate all of it at
            # once. Degrade to the clamped cursor channel, exactly as an empty request
            # does above: a payload that HAS an image must never deliver zero planes and
            # read "no image on this output" (2026-08-15; the Viewer corrects its active
            # set only on delivery, one click too late to save this pull).
            out.append((_plane_key(node_id, pin, m, t, z, c, cap), m, t, z, c))
        return out

    def _cached_planes(self, node_id, coords, channels, axes,
                       *, pin: Optional[Pin] = None, provider: Any = None,
                       dtype: Any = _UNSET, sub: int = 0) -> Optional[Dict[int, np.ndarray]]:
        """The requested planes if EVERY one of them is already decoded, else ``None`` —
        a read-only probe, so the GUI thread can decide whether serving this frame is free
        before it commits to doing it there (:meth:`_serve_from_cache`)."""
        out: Dict[int, np.ndarray] = {}
        for key, _m, _t, _z, ch in self._plane_addrs(node_id, coords, channels, axes,
                                                    pin=pin, provider=provider,
                                                    dtype=dtype, sub=sub):
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
                # canvas=union (2026-10-01): the output is a BLANK canvas, so the primary is a
                # source too — `<node>#primary`, stamped by the compute. Not when the primary
                # is itself a canvas (an earlier union Overlay, or a `view.canvas`): its own
                # sources are further down this spine and are collected there. Appended
                # AFTER the secondary so the reverse below puts it first (drawn underneath).
                if (dict(getattr(node, "modes", None) or {}).get("canvas") == "union"
                        and pri and not self_is_canvas_node(graph, pri[0])):
                    out.append((f"{cur}#primary", pri[0]))
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
            src_lut = self._source_lut(sec_id, sec)
            n = min(int(sec.axes.c), MAX_OVERLAY_CHANNELS)
            if n <= 0 or n > budget:
                dropped += 1
                continue
            sources.append({"entry": entry, "ovl_id": ovl_id, "sec_id": sec_id,
                            "src_lut": src_lut,
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
                         *, region=None, t_sub: int = 0, z_index: Optional[int] = None,
                         override: Optional[Dict[str, Tuple[int, int]]] = None
                         ) -> Dict[int, np.ndarray]:
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
        their image.

        ``t_sub`` is the Play-all sub-tick inside primary frame ``t`` (of
        :meth:`overlay_sub_ticks`); each source reads the frame its OWN map puts there
        (:func:`~nodegraph.placement.paired_t_frac`), so a 4x-faster source advances on every
        tick and a same-rate one holds. ``z_index`` is the primary's plane index, which Z pins
        in index form need. ``override`` is the Viewer's display-only per-source ``(dt, dz)``
        stepper offset. Z is sampled through the node's own :func:`z_pick` — the same slice
        choice and weights the ``resample`` bake uses — so ``linear`` blends here exactly as it
        does there."""
        from nodelab_v2.overlay_compose import compose_secondary_plane
        ctx = self._overlay_ctxs.get(node_id)
        if not ctx or ctx["node"] != node_id:
            return {}
        # The composite is sampled onto `out_shape`, so reading the window any finer than that
        # would be thrown away by the nearest-neighbour map — and any coarser is the blur.
        budget = max(1, int(max(out_shape[0], out_shape[1])))
        n_sub = self.overlay_sub_ticks(node_id)
        pri_k = int(z_index) if z_index is not None else 0
        out: Dict[int, np.ndarray] = {}
        for src in ctx.get("sources", ()):
            # Per SOURCE, so one source that cannot be placed at this frame (an unpaired
            # timepoint, an unreadable tile) costs only its own layer — the rest of the
            # chain still draws.
            try:
                entry = src["entry"]
                sec_ax = src["sec_axes"]
                frame = self._source_frame(src, ctx, t, int(t_sub), n_sub, pri_k, z_um,
                                           m, override)
                if frame is None:
                    continue
                t_sec, z_weights = frame
                tiles = dict((int(a), b) for a, b in (entry.get("tiles") or ()))
                hits = tiles.get(int(m)) or ()
                if not hits:
                    continue
                prov = src["sec_provider"]
                for k in range(int(src["n"])):
                    blended = None
                    for z_sec, w in z_weights:
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
                            part = composed if w == 1.0 else composed * np.float32(w)
                            blended = part if blended is None else blended + part
                    if blended is not None:
                        out[int(src["base_c"]) + k] = blended
            except Exception:  # noqa: BLE001 — the image must survive a bad overlay
                continue
        return out

    def _source_frame(self, src: Dict[str, Any], ctx: Dict[str, Any], t: int, t_sub: int,
                      n_sub: int, pri_k: int, z_um, m: int,
                      override: Optional[Dict[str, Tuple[int, int]]] = None):
        """``(t_sec, [(z_sec, weight)])`` this source shows at primary ``(t, sub, z)``, or
        ``None`` where it has no frame. The ONE resolution the compositor and the Play-all
        readout share, so the numbers under the strip are the frame on screen."""
        from nodegraph.catalog._shared.placement_entry import z_pick
        from nodegraph.placement import paired_t_frac
        entry = src["entry"]
        sec_ax = src["sec_axes"]
        t_sec = paired_t_frac(entry, int(t), int(t_sub), int(n_sub))
        if t_sec is None:
            return None
        tiles = dict((int(a), b) for a, b in (entry.get("tiles") or ()))
        hits = tiles.get(int(m)) or ()
        sec_m = int(hits[0][0]) if hits else 0
        z_weights = z_pick(entry, src["sec_md"], sec_ax, sec_m, ctx["pri_md"],
                           ctx["pri_axes"], pri_k, z_um)
        dt, dz = (override or {}).get(str(src.get("ovl_id")), (0, 0))
        if dt:
            t_sec = min(max(0, int(t_sec) + int(dt)), int(getattr(sec_ax, "t", 1) or 1) - 1)
        if dz:
            nz = int(getattr(sec_ax, "z", 1) or 1)
            # a stepped Z is a real acquired plane: the user is looking for the ONE that goes
            # with this one, which an interpolated blend could not show them
            z0 = int(max(z_weights, key=lambda p: p[1])[0])
            z_weights = [(min(max(0, z0 + int(dz)), nz - 1), 1.0)]
        return int(t_sec), z_weights

    def overlay_flicker_hz(self, node_id: str) -> float:
        """Blink rate for the overlay's flicker mode, or ``0`` when no source uses it.

        The FIRST flickering source decides the rate: two sources blinking out of phase is
        not a comparison of anything."""
        ctx = self._overlay_ctxs.get(node_id)
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
        ctx = self._overlay_ctxs.get(node_id)
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
        ctx = self._overlay_ctxs.get(node_id)
        if not ctx or ctx["node"] != node_id:
            return {}
        out: Dict[int, str] = {}
        for i, src in enumerate(ctx.get("sources", ())):
            names = src["sec_md"].get("channel_names") or []
            for k in range(int(src["n"])):
                label = str(names[k]) if k < len(names) and names[k] else str(k)
                # the SOURCE's name is in the label — the Overlay's own `label`, else the
                # card title or file it came from (`source_label`) — so five overlaid files
                # are tellable apart on the channel strip without opening the graph. A
                # `view_source` wire is named by its SOCKET instead — it is a channel of this
                # graph, not the nth overlaid file.
                out[int(src["base_c"]) + k] = f"{EngineRunner.source_label(self, src, i)}:{label}"
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
            from nodegraph.catalog._shared.placement_entry import (
                handedness_for, overlay_settings, plan_kwargs)
            from nodegraph.catalog.view.overlay import overlay_entry
            from nodegraph.placement import plan_placement
            from nodegraph.nodes import SAMPLING_KEY
            base_id, _sep, role = str(ovl_id).partition("#")
            node = self.document.nodes.get(base_id)
            if node is None or payload is None or sec is None:
                return stamped
            # The node's OWN settings reader, not a list kept here: a list here is how every
            # new Overlay setting used to be honoured by the record and ignored by the picture.
            s = overlay_settings(node, dict(node.modes or {}))
            if role == "primary":
                # a union canvas's PRIMARY source (`<node>#primary`): placed by its own stage
                # position with the node's handedness, and none of the secondary's nudge,
                # pins or rate — exactly what the compute stamped
                s = {**s, "blend": "add", "t_shift": 0, "offset_um": (0.0, 0.0, 0.0),
                     "t_pairing": "index", "rate": 0.0, "t_pins": (), "z_pins": (),
                     "min_coverage": 0.0}
            plan = plan_placement(
                payload.metadata, payload.axes, sec.metadata, sec.axes,
                dst_sampling=tuple(payload.metadata.get(SAMPLING_KEY, ())),
                src_sampling=tuple(sec.metadata.get(SAMPLING_KEY, ())),
                **plan_kwargs(s))
            if not plan.ok:
                return stamped        # keep the record; the pull itself already refused
            # The SAME handedness derivation the compute makes — an already-stitched secondary
            # is in stage coordinates, and a display path that flipped it while the compute did
            # not would draw the overlay 456 px from where a bake put it.
            fx, fy, _w = handedness_for(sec, s["flip_x"], s["flip_y"])
            # `context` is carried over from the stamped entry: the context boxes are the
            # compute's to build (they need the canvas mode against the stamped geometry),
            # and a re-plan that dropped them silently narrowed a context canvas back to the
            # primary's field.
            return overlay_entry(ovl_id, plan, {**s, "flip_x": fx, "flip_y": fy},
                                 context=stamped.get("context") or [])
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

    def _source_lut(self, sec_id: str, sec) -> Dict[int, Dict[str, Any]]:
        """The secondary's OWN display terms, per channel — worker thread, cached per source.

        An overlay channel is a second FILE, not a second view of this one, so its contrast
        has to come from that file: its own significant bit depth for the histogram extent,
        and a window taken from one of ITS OWN whole planes. Two things it deliberately is
        not. Not the composed crop, which is a sliver of a different-sized field and gave a
        (300, 301) window on the WellA3 pair — noise. And not the secondary NODE's live
        clim: reading that would tie the two together, so tuning the overlay would move the
        source view and vice versa, when what the two datasets share is the OVERLAP (a
        placement, which is metadata) and nothing about how either is displayed.
        """
        out = self._src_lut_cache.get(sec_id)
        if out is not None:
            return out
        out = {}
        prov = getattr(sec, "image", None)
        ax = getattr(prov, "axes", None)
        depth = (sec.metadata or {}).get("bit_depth") if hasattr(sec, "metadata") else None
        for k in range(min(int(getattr(ax, "c", 1) or 1), MAX_OVERLAY_CHANNELS)):
            lohi = None
            try:
                plane, _lv = render_plane_native(
                    prov, 0, 0, int(getattr(ax, "z", 1) or 1) // 2, k, max_dim=1024)
                finite = np.asarray(plane, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lo = float(np.percentile(finite, 1.0))
                    hi = float(np.percentile(finite, 99.5))
                    lohi = (lo, hi if hi > lo else lo + 1.0)
            except Exception:      # noqa: BLE001 — no window is better than a wrong one
                lohi = None
            out[k] = {"clim": lohi, "bit_depth": int(depth) if depth else None}
        self._src_lut_cache[sec_id] = out
        return out

    def _warm_overlay(self, key: tuple, m: int, t: int, z: int, ch: int) -> None:
        """Compose ONE overlay plane into the cache ahead of the cursor (worker thread).

        Goes through `_compose_overlay` so a prefetched plane and a displayed one are the
        same pixels; a prefetcher that composed differently would be worse than none.

        The Play-all sub-tick (and stepper generation) come from the KEY itself
        (:func:`_key_sub`), so a job list can warm every sub-tick of a frame without a second
        job shape. One compose yields every overlay channel of the chain, so every one of them
        is cached — keeping only ``ch`` threw the rest away and composed them again for each
        of their own jobs, n_sub times over under Play all."""
        node_id = str(key[0])
        try:
            ctx = self._overlay_ctxs.get(node_id) if hasattr(self, "_overlay_ctxs")                 else self._overlay_ctx
            if not ctx:
                return
            shape = ctx.get("display_shape")
            if not shape:
                return
            sub, ovr = _key_sub(key)
            if ovr and ovr != self._ovr_gen:
                return             # a stepper moved since this was queued: stale on arrival
            z_um = _z_um_of(ctx["pri_md"], ctx["pri_axes"], m, z)
            got = self._compose_overlay(node_id, tuple(shape), m, t, z_um, t_sub=sub,
                                        z_index=z,
                                        override=dict(self._src_override) if ovr else {})
            for c_k, arr in got.items():
                if arr is not None:
                    self._planes.put(key[:5] + (int(c_k),) + key[6:], arr)
        except Exception:      # noqa: BLE001 — a warm miss costs latency, never the frame
            return

    def overlay_sources(self, node_id: str) -> Dict[int, Dict[str, Any]]:
        """``{display channel: (secondary node id, that source's channel)}``.

        What the Viewer needs to give an overlay channel the LUT of the FILE it came from
        rather than one invented from the crop. A composed overlay plane is the part of the
        secondary that happens to fall inside the primary's field — on the WellA3 pair a
        294 µm sliver of a 1760 µm frame — so percentiles taken from it describe a different
        image than the one the user tuned while looking at that file, which is what made the
        overlay "rewrite the LUT" and look nothing like its source."""
        ctx = self._overlay_ctxs.get(node_id) if hasattr(self, "_overlay_ctxs")             else self._overlay_ctx
        if not ctx or (ctx.get("node") not in (None, node_id)):
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        for src in ctx.get("sources", ()):
            sid = str(src.get("sec_id") or src.get("ovl_id") or "")
            lut = src.get("src_lut") or {}
            for k in range(int(src.get("n", 1))):
                info = dict(lut.get(k) or {})
                info.update({"node": sid, "ch": k})
                out[int(src["base_c"]) + k] = info
        return out

    def overlay_sub_ticks(self, node_id: str) -> int:
        """How many ticks Play all splits each primary frame into for ``node_id``'s overlay:
        the most any source asked for (its entry's ``sub_ticks``), else 1. A same-rate chain
        is 1 — playback is exactly what it always was."""
        ctx = self._overlay_ctxs.get(node_id) if hasattr(self, "_overlay_ctxs") else None
        if not ctx or ctx.get("node") != node_id:
            return 1
        return max([1] + [int(s["entry"].get("sub_ticks", 1) or 1)
                          for s in ctx.get("sources", ())])

    def overlay_frame_readout(self, node_id: str, m: int, t: int, sub: int,
                              z: int) -> List[Dict[str, Any]]:
        """Per source, what it is showing at primary ``(m, t, sub, z)`` — the Play-all strip.

        ``{ovl_id, sec_id, label, t, n_t, z, n_z, t_pinned, z_pinned, offset}``; ``t``/``z``
        are ``None`` where the source has no frame (the pairing ran off its end). ``z`` is the
        dominant plane of a linear blend. Resolved through the compositor's own
        :meth:`_source_frame`, so the readout IS the frame on screen."""
        ctx = self._overlay_ctxs.get(node_id)
        if not ctx or ctx.get("node") != node_id:
            return []
        n_sub = self.overlay_sub_ticks(node_id)
        z_um = _z_um_of(ctx["pri_md"], ctx["pri_axes"], int(m), z)
        out: List[Dict[str, Any]] = []
        for i, src in enumerate(ctx.get("sources", ())):
            if src.get("as_channel"):
                continue           # a `view_source` channel of this graph, not an overlaid file
            entry, sax = src["entry"], src["sec_axes"]
            try:
                fr = self._source_frame(src, ctx, t, sub, n_sub, int(z), z_um, int(m),
                                        dict(self._src_override))
            except Exception:  # noqa: BLE001 — a readout must never cost the view
                fr = None
            tz = (None, None) if fr is None else (
                fr[0], int(max(fr[1], key=lambda p: p[1])[0]))
            pins_t = {int(r[0]) for r in (self._pins_of(src["ovl_id"], "t_pins"))}
            pins_z = {int(r[0]) for r in (self._pins_of(src["ovl_id"], "z_pins"))}
            out.append({
                "ovl_id": src["ovl_id"], "sec_id": src["sec_id"],
                "label": self.source_label(src, i),
                "t": tz[0], "n_t": int(getattr(sax, "t", 1) or 1),
                "z": tz[1], "n_z": int(getattr(sax, "z", 1) or 1),
                "t_pinned": int(t) in pins_t, "z_pinned": int(z) in pins_z,
                "offset": tuple(self._src_override.get(str(src["ovl_id"]), (0, 0))),
                "sub_ticks": int(entry.get("sub_ticks", 1) or 1)})
        return out

    def _pins_of(self, ovl_id: str, name: str) -> tuple:
        """The overlay node's live pins (canonical rows), ``()`` when absent or malformed."""
        from nodegraph.placement import parse_pins
        node = self.document.nodes.get(ovl_id)
        try:
            return parse_pins((node.params if node is not None else {}).get(name, ""),
                              axis=name[0])
        except ValueError:
            return ()

    def source_label(self, src: Dict[str, Any], index: int) -> str:
        """The name a source goes by on the channel strip, its LUTs and the Play-all strip.

        The Overlay's own ``label`` (presentation, read live), else the secondary card's
        title, else the base name of the file at the root of the secondary's chain, else
        ``ovl{n}`` — never blank, because five overlaid files must be tellable apart."""
        pre = str(src.get("prefix") or "")
        if pre:
            return pre                          # a `view_source` channel: its socket name
        try:
            return self._source_label_of(src) or f"ovl{index + 1}"
        except Exception:  # noqa: BLE001 — a name is cosmetic; it must never cost the strip
            return f"ovl{index + 1}"

    def _source_label_of(self, src: Dict[str, Any]) -> str:
        from nodelab_v2.document import TITLE_KEY
        from nodelab_v2.ops import batch_member_identity
        doc = self.document
        node = doc.nodes.get(str(src.get("ovl_id")))
        own = str((node.params if node is not None else {}).get("label") or "").strip()
        if own:
            return own
        sid = str(src.get("sec_id") or "")
        sec = doc.nodes.get(sid)
        title = str((sec.params if sec is not None else {}).get(TITLE_KEY) or "").strip()
        if title:
            return title
        # the root of the secondary's primary spine — the file it came from
        cur, seen = sid, set()
        while cur and cur not in seen:
            seen.add(cur)
            preds = [e[0] for e in doc.edges if e[2] == cur and e[3] == "data"]
            if not preds:
                break
            cur = preds[0]
        root = doc.nodes.get(cur)
        if root is not None and root.params.get("path"):
            name = batch_member_identity(root, cur)
            if name and name != cur:
                return name
        return ""

    def overlay_pin_anchors(self, node_id: str, ovl_id: str, axis: str, m: int,
                            pri: int, sec: int) -> Tuple[Optional[float], Optional[float]]:
        """The ABSOLUTE anchors a new pin records beside its two indices, so it keeps
        meaning the same frames after an upstream crop re-numbers them: each file's frame
        clock (``frame_time_jd``) for a T pin, each plane's absolute focus (µm) for a Z pin.
        ``(None, None)`` where either file lacks it — the pin then pairs by index."""
        ctx = self._overlay_ctxs.get(node_id)
        src = next((s for s in (ctx or {}).get("sources", ())
                    if str(s.get("ovl_id")) == str(ovl_id)), None)
        if src is None:
            return (None, None)
        pmd, smd = ctx["pri_md"], src["sec_md"]
        try:
            if axis == "t":
                pj, sj = pmd.get("frame_time_jd") or (), smd.get("frame_time_jd") or ()
                if 0 <= pri < len(pj) and 0 <= sec < len(sj):
                    return (float(pj[pri]), float(sj[sec]))
                return (None, None)
            from nodegraph.placement import z_um_of_slice
            tiles = dict((int(a), b) for a, b in (src["entry"].get("tiles") or ()))
            hits = tiles.get(int(m)) or ()
            sec_m = int(hits[0][0]) if hits else 0
            zp = z_um_of_slice(pmd, ctx["pri_axes"], int(m), int(pri))
            zs = z_um_of_slice(smd, src["sec_axes"], sec_m, int(sec))
            return (None, None) if (zp is None or zs is None) else (float(zp), float(zs))
        except (TypeError, ValueError):
            return (None, None)

    def set_source_override(self, ovl_id: str, dt: int = 0, dz: int = 0) -> None:
        """Move one overlay source's DISPLAYED frame by ``(dt, dz)`` from its mapped frame
        (``(0, 0)`` clears it) — the Viewer's ◀▶ steppers. Display-only: the document is not
        touched, nothing re-runs, and the planes composed under it are keyed apart."""
        key = str(ovl_id)
        now = self._src_override.get(key, (0, 0))
        new = (int(dt), int(dz))
        if new == now:
            return
        if new == (0, 0):
            self._src_override.pop(key, None)
        else:
            self._src_override[key] = new
        self._ovr_gen += 1

    def clear_source_overrides(self) -> None:
        if self._src_override:
            self._src_override.clear()
            self._ovr_gen += 1

    def overlay_style(self, node_id: str) -> Dict[int, Tuple[int, float, float]]:
        """``{channel index: (blend mode, opacity, checker cells)}`` for the overlay's
        channels — what the shader and its CPU mirror need to composite them."""
        ctx = self._overlay_ctxs.get(node_id)
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
            # A union canvas's own PRIMARY (`#primary`) is the base image, drawn onto a blank
            # backdrop — at full strength, or the picture's main file would be a half wash.
            opacity = (1.0 if (src.get("as_channel") or "#" in str(src["ovl_id"]))
                       else self._look(src["ovl_id"], "opacity", 0.5))
            style = (self.BLEND_MODES.get(name, 0), opacity, param)
            for k in range(int(src["n"])):
                out[int(src["base_c"]) + k] = style
        return out

    def _decode_planes(self, provider, node_id, coords, channels, axes,
                       *, pin: Optional[Pin] = None, overlay_all: bool = False,
                       as_dtype: Any = _UNSET, sub: int = 0) -> Dict[int, np.ndarray]:
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
        if as_dtype is _UNSET:
            view = self._view_of(node_id)
            as_dtype = view.dtype if view is not None else None
        addrs = self._plane_addrs(node_id, coords, channels, axes, pin=pin,
                                  provider=provider, dtype=as_dtype, sub=sub)
        if not addrs:
            return out
        nc = int(axes.c)
        cap = addrs[0][0][6]           # the cap the keys were built with — never re-derived
        dt = as_dtype
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
        octx = self._overlay_ctxs.get(node_id)
        if octx is None or not (ovl_addrs or overlay_all):
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
        z_um = _z_um_of(octx["pri_md"], octx["pri_axes"], m, z)
        # Recorded so the PREFETCHER composes at exactly this size. A warmed plane of a
        # different shape would be worse than no warming: the displayed-frame path would
        # find a cache entry it cannot use, or worse, use one of the wrong shape.
        octx["display_shape"] = tuple(ref.shape[:2])
        # The override snapshot is taken ONCE, with the keys: a stepper moved while this runs
        # must not compose one frame's pixels under another generation's key.
        ovr = self._ovr_gen if self._src_override else 0
        composed = self._compose_overlay(node_id, ref.shape[:2], m, t, z_um, t_sub=sub,
                                         z_index=z, override=dict(self._src_override))
        keys = {a[4]: a[0] for a in ovl_addrs}
        for ch, plane in composed.items():
            key = keys.get(ch)
            if key is None:
                if not overlay_all:
                    continue                   # not asked for: its toggle is off
                key = _plane_key(node_id, pin, m, t, z, int(ch), cap, sub, ovr)
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
        view = self._view_of(node_id)
        key = ((node_id, None, m, t, z, c) if src_id == node_id
               and (view is None or view.pin is None)
               else (_RAW_TAG, self._node_source_key[src_id], m, t, z, c))
        arr = self._planes.get(key)
        if arr is None:
            arr, _lv = render_plane_native(prov, m, t, z, c)
            self._planes.put(key, arr)
        return arr

    def _serve_from_cache(self, node_id, coords, channels, sub: int = 0) -> None:  # GUI thread
        """Display the plane(s) at ``coords`` without going through the engine.

        A **warm** frame is served right here: the pixels are already decoded, so emitting
        them is a dictionary lookup and scrubbing keeps its zero-hop latency. A **cold** one
        goes to :class:`_DecodeJob` on the pool — decoding it here would run the node on the
        GUI thread and freeze the application for as long as that takes."""
        t0 = time.perf_counter()
        view = self._view_of(node_id)
        if view is None:                     # dropped between request and here: re-pull
            self.pull(node_id, coords, channels)
            return
        axes, pin = view.axes, view.pin
        warm = self._cached_planes(node_id, coords, channels, axes, pin=pin,
                                   provider=view.provider, dtype=view.dtype, sub=sub)
        if warm is not None:
            self.plane_ready.emit(node_id, warm, axes, time.perf_counter() - t0)
            # Only on sub-tick 0: one prefetch warms every sub-tick of the frames ahead, and
            # each call supersedes the last (`_prefetch_gen`) — calling it on every Play-all
            # tick would cancel the warm after one compose, and it would never get ahead.
            if not sub:
                self.prefetch(node_id,
                              self._clamp_coords(self._payload_coords(coords, pin), axes),
                              tuple(channels) if channels else None, pin=pin)
            return
        if self._decode_busy:
            self._decode_pending = (node_id, coords, channels, sub)   # latest-wins
            return
        self._decode_gen += 1
        self._decode_busy = True
        # the card + status bar say "reading planes" for the whole decode — the same
        # runner-level event the full-pull path emits when a lazy chain does its real work
        # at read time (`_Worker.run`), so a cold scrub is visibly working rather than hung
        self._progress.emit(("decode", node_id, {"epoch": self._epoch, "op_key": ""}))
        self._pool.start(_DecodeJob(self, self._decode_gen, self._epoch,
                                    view.provider, node_id, coords, channels,
                                    axes, pin, view.dtype, sub))

    def _deliver_planes(self, packet) -> None:   # GUI thread (queued)
        """Land a :class:`_DecodeJob`'s planes, then run whatever the cursor did meanwhile.

        Staleness is judged on ``_decode_gen`` (the cursor moved on) and ``_epoch`` (an edit
        or a real pull happened): either way the pixels describe a frame nobody is looking at
        any more, so they are dropped — they are still in the PlaneCache, so nothing is
        wasted if the cursor comes back."""
        (gen, epoch, node_id, coords, channels, pin, sub, planes, axes, dt, err) = packet
        self._decode_busy = False
        pending, self._decode_pending = self._decode_pending, None
        fresh = gen == self._decode_gen and epoch == self._epoch
        if fresh and err is not None:
            self.failed.emit(node_id, err)
        elif fresh and planes:
            self._progress.emit(("done", node_id, {"epoch": epoch, "seconds": dt}))
            self.plane_ready.emit(node_id, planes, axes, dt)
            if not sub:
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
        view = self._view_of(node_id)
        if view is None:
            return
        prov, axes = view.provider, view.axes
        if prov is None or axes is None or getattr(axes, "t", 1) <= 1:
            return
        span = min(span, self._prefetch_span(prov))
        if span <= 0:
            return
        m, t, z, _c = center
        nt = axes.t
        # Overlay channels live ABOVE `axes.c` and must survive the clamp. They used not
        # to, so prefetch warmed only the primary's channels and every frame of a playback
        # paid the overlay compose on the critical path — the "does not play fast" report.
        # The composed planes are cached under the same keys, so warming them is the whole
        # fix; they just have to be asked for.
        ovl_ch = tuple(sorted(self.overlay_channels(node_id)))
        chans = tuple(ch if ch in ovl_ch else min(max(0, int(ch)), axes.c - 1)
                      for ch in (channels if channels else
                                 (min(max(0, center[3]), axes.c - 1),)))
        chans = tuple(dict.fromkeys(chans + ovl_ch))
        cap = self.display_dim(axes, planes=max(1, len(set(chans))),
                               provider=prov, dtype=view.dtype)
        # Play all: every sub-tick of each frame ahead, for OVERLAY channels only - the
        # primary plane is shared by all of them. The frame UNDER the cursor is warmed too,
        # since its later sub-ticks are exactly what the next few ticks show.
        n_sub = self.overlay_sub_ticks(node_id)
        ovr = self._ovr_gen if self._src_override else 0
        self._prefetch_gen += 1
        gen = self._prefetch_gen
        jobs: List[Tuple[tuple, int, int, int, int]] = []
        order = ([t] if n_sub > 1 else []) + [tt for d in range(1, span + 1)
                                              for tt in ((t + d) % nt, (t - d) % nt)]
        for tt in order:
            for ch in chans:
                is_ovl = ch in ovl_ch
                for sb in (range(n_sub) if is_ovl else (0,)):
                    if tt == t and not (is_ovl and sb):
                        continue               # the displayed frame itself is already here
                    key = _plane_key(node_id, pin, m, tt, z, ch, cap,
                                     sb if is_ovl else 0, ovr if is_ovl else 0)
                    if self._planes.get(key) is None:
                        jobs.append((key, m, tt, z, ch))
        if jobs:
            self._pool.start(_PrefetchJob(self, gen, jobs, prov, view.dtype))

    #: How many preload jobs run at once on a series whose frames are affordable to read
    #: ahead — real bytes AND per-plane computes alike. Four, measured on the per-plane case
    #: itself: on the WellA3 mosaic (a computed stitch) with a cold tile cache one
    #: whole-canvas paste is 1.03 s, four in parallel are 0.51 s each and eight are 0.43 s —
    #: the source-tile reads parallelize, the paste contends, and the curve is flat past
    #: four. Kept low on purpose: these share the pool with the frame being DISPLAYED, and
    #: starving that to fetch the future is exactly backwards. The one series that gets NO
    #: width at all is a whole-volume compute — :meth:`_preload_jobs` returns 0 there.
    PRELOAD_JOBS = 4

    def _frame_bytes(self, axes: Any, chans: int, *, provider: Any = None,
                     dtype: Any = None) -> int:
        """Bytes one displayed frame of ``chans`` channels occupies in the plane cache."""
        cap = self.display_dim(axes, planes=max(1, chans), provider=provider, dtype=dtype)
        return (min(cap, int(axes.y)) * min(cap, int(axes.x))
                * (2 if dtype is not None else 8) * max(1, chans))

    def _overlay_frame_bytes(self, node_id: str, axes: Any, *, provider: Any = None,
                             dtype: Any = None) -> int:
        """Extra bytes one frame's COMPOSED overlay planes cost in the plane cache: float32,
        one per overlay channel per Play-all sub-tick. `_frame_bytes` alone assumed every
        plane was a 2-byte read, which under-counts a composed overlay by 2x and a 4x-rate
        source by 8x - the preload would then size itself past the cache and evict its own
        head."""
        n_ovl = len(self.overlay_channels(node_id))
        if not n_ovl:
            return 0
        cap = self.display_dim(axes, planes=1, provider=provider, dtype=dtype)
        return (min(cap, int(axes.y)) * min(cap, int(axes.x)) * 4
                * n_ovl * self.overlay_sub_ticks(node_id))

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
        * **cost-gated on the provider** (:meth:`_preload_jobs`). "Every frame is wanted" is a
          licence to READ them all, never to COMPUTE them all: on a whole-volume chain this
          queued a preload measured in hours, and playback waited for it.
        """
        view = self._view_of(node_id)
        if view is None or view.provider is None or view.axes is None:
            return 0
        prov, axes = view.provider, view.axes
        jobs = self._preload_jobs(prov)
        if not jobs:
            # a frame here IS a whole-volume compute — see `_preload_jobs`. Nothing is queued
            # and nothing is waited for; the frame the cursor lands on decodes on its own, one
            # at a time, with the pool to itself. Any older preload is retired with it, so its
            # progress bar cannot outlive the series it was reading.
            self.cancel_preload()
            return 0
        m, t0, z, _c = center
        # Overlay channels (indices above the primary's own) are KEPT, not clamped into the
        # primary's range: the clamp is why Play never preloaded the overlay at all, so every
        # frame of a playback composed it on the critical path.
        ovl_ch = set(self.overlay_channels(node_id))
        chans = tuple(sorted({int(ch) if int(ch) in ovl_ch else min(max(0, int(ch)), axes.c - 1)
                              for ch in (channels or (center[3],))}))
        prim = [ch for ch in chans if ch not in ovl_ch]
        cap = self.display_dim(axes, planes=max(1, len(chans)),
                               provider=prov, dtype=view.dtype)
        nt = max(1, int(axes.t))
        n_sub = self.overlay_sub_ticks(node_id)
        ovr = self._ovr_gen if self._src_override else 0
        per_frame = (self._frame_bytes(axes, max(1, len(prim)), provider=prov,
                                       dtype=view.dtype)
                     + (self._overlay_frame_bytes(node_id, axes, provider=prov,
                                                  dtype=view.dtype)
                        if ovl_ch & set(chans) else 0))
        room = max(1, min(self._planes.budget, display_ram_bytes()) // max(1, per_frame))
        start = min(max(0, int(t0)), nt - 1)
        want: List[Tuple[tuple, int, int, int, int]] = []
        for i in range(min(nt, int(room))):
            tt = (start + i) % nt
            for ch in chans:
                is_ovl = ch in ovl_ch
                for sb in (range(n_sub) if is_ovl else (0,)):
                    key = _plane_key(node_id, pin, m, tt, z, ch, cap,
                                     sb if is_ovl else 0, ovr if is_ovl else 0)
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
        n = max(1, min(jobs, max(1, self._pool.maxThreadCount() - 1)))
        for k in range(n):
            share = want[k::n]
            if share:
                self._pool.start(_PreloadJob(self, self._preload_gen, share,
                                             prov, view.dtype))
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

    def series_fits(self, axes: Any = None, *, planes: int = 1,
                    node_id: Optional[str] = None) -> bool:
        """Whether the whole T range of a held node's frames fits the budget — i.e.
        whether a preload can make playback read-free rather than merely warmer.
        ``node_id`` names which pane's node; default is the most recently held one."""
        view = (self._view_of(node_id) if node_id is not None
                else next(reversed(self._views.values()), None))
        if axes is None:
            axes = view.axes if view is not None else None
        if axes is None:
            return False
        ceiling = min(self._planes.budget, display_ram_bytes())
        prov, dt = (view.provider, view.dtype) if view else (None, None)
        # composed overlay planes are float32 and one per Play-all sub-tick — counted, or a
        # 4x-rate overlay reports "fits" for a series that is 8x the budget in overlay alone
        n_ovl = len(self.overlay_channels(node_id)) if node_id is not None else 0
        extra = (self._overlay_frame_bytes(node_id, axes, provider=prov, dtype=dt)
                 if n_ovl else 0)
        return ((self._frame_bytes(axes, max(1, planes - n_ovl), provider=prov, dtype=dt)
                 + extra) * max(1, int(axes.t)) <= ceiling)

    def frames_are_reads(self, node_id: str) -> bool:
        """Whether one frame of ``node_id``'s held result is BYTES — a decompress off a store
        — rather than the node running again.

        The one question the Viewer's playback policy turns on, asked once and answered here
        so the gate and the pacing cannot disagree about it. A lazy chain's provider computes
        at read time, which is what makes "play" mean two different things: a stream of
        uploads on an ingested file, and a queue of computes on a deconvolution. Unknown
        nodes (never pulled, no held view) answer False — the cautious direction, since it
        costs a smooth playback and the other costs a frozen one."""
        view = self._view_of(node_id)
        if view is None or view.provider is None:
            return False
        return not isinstance(view.provider, StreamProvider)

    @staticmethod
    def _preload_jobs(prov: Any) -> int:
        """How many whole-series preload decodes may run at once on ``prov`` — the COST
        question :meth:`_prefetch_span` asks about two neighbours, asked about the whole
        T range. Zero means do not preload this series at all.

        Pressing play licenses reading every frame. It does not license *computing* every
        frame. On a ``volume_unit`` chain one "decode" is a whole Richardson–Lucy /
        ZS-DeconvNet volume — ~130 s and ~30 GB of working set — so the byte-backed job count
        put four of them on the pool at once and held playback until they finished. That is
        how pressing play on the 3D-deconvolved series stopped showing frames altogether
        (2026-08-06): a wait measured in hours, with four volumes' working set crowding out
        the frame being *displayed* — the same starvation :class:`_DecodeJob` stays
        single-flight to avoid, re-introduced by the job that runs beside it.

        A PER-PLANE compute (a stitched mosaic, a Z-projection — standalone or over that
        mosaic) is the other side of that line and warms at the full
        :data:`PRELOAD_JOBS` width, which was *measured on it* (see the constant: the 4-way
        numbers are the WellA3 mosaic's own). It warmed one at a time for a while out of
        pool-sharing caution, and that width could never outrun playback consuming one frame
        per frame: pressing play on a stitched series stayed at decode cadence — ~9 fps with
        33–267 ms of jitter, recorded 2026-08-10 — for the whole first lap instead of going
        smooth after a short prepare. The pool keeps a thread back for the displayed frame
        either way (:meth:`preload_series` fans out to ``maxThreadCount - 1`` at most).

        So: warm wide on anything priced per plane — bytes or kernel — and never speculate
        across frames of a whole-unit compute, where the only useful frame is the one being
        looked at."""
        if getattr(prov, "volume_unit", False):
            return 0                               # a neighbour t IS a whole volume
        return EngineRunner.PRELOAD_JOBS

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
            # LIVENESS, the same rule `_deliver_progress` applies on the GUI side: a run
            # reports for as long as it is live. The old `epoch != self._epoch` test muted
            # a SURVIVING run's card the moment any narrowed invalidate advanced the epoch
            # — an edit or a delete on the *other* branch froze this one's bar mid-way.
            # A bake is live by being in `_bakes` (it never registers in `_runs`; its
            # result is resolved outside the staleness rule, so it reports to the end).
            # (GIL-atomic dict probes from the worker thread, like `_Job.cancelled`.)
            if (epoch is not None and epoch not in self._runs
                    and epoch not in self._bakes):
                return                        # a cancelled pull — stop reporting for it
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
        # LIVENESS again, not "is this the newest" (2026-08-06). ``epoch != self._epoch``
        # silenced a run's progress the instant any other pull started — so queueing a second
        # branch froze the first one's bar mid-way and its card sat there looking hung while
        # it was in fact still working. A run reports for as long as it is live; only a
        # CANCELLED run's tail is dropped, which is the case this gate was written for.
        # A bake's liveness lives in `_bakes` — it never registers in `_runs`, and gating
        # it on `_runs` alone muted every bake's card from the first tick.
        if (epoch is not None and epoch not in self._runs
                and epoch not in self._bakes):
            return                            # a cancelled pull's tail — the cards moved on
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
        pull_id = self._pull_id(node_id, graph)
        sources, every = self._sources_for(pull_id, graph)
        job = _Job(self._epoch, graph, self.document.revision, node_id, None, None,
                   sources, pin=self._pin_for(coords) if scoped else None,
                   bake={"store": store, "precision": precision, "bake_id": bake_id,
                         "signature": signature, "scoped": bool(scoped),
                         "hold": bool(hold)},
                   all_sources=every, pull_id=pull_id)
        self._busy = True
        self._active = job          # what the worker is running (never cancel-latched:
        #                             a bake resolves outside the staleness rule)
        self._bakes[job.epoch] = job.bake
        self.plan.emit(node_id, self.planned_nodes(pull_id, graph))
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
        payload = engine.pull(job.pull_id)
        verb = "hold" if spec.get("hold") else "bake"
        if not isinstance(payload, Dataset):
            raise TypeError(
                f"a Dock can only {verb} a Dataset; this one's input produced "
                f"{type(payload).__name__}. Wire the image/analysis chain into it.")
        if spec.get("hold"):
            # No write, no copy — the whole reason this tier is instant. The envelope is
            # captured alongside because a held dock has no manifest to re-derive one from.
            spec["payload"] = payload
            spec["env"] = engine.env(job.pull_id)
            return
        observe = self._make_observer(job.epoch)

        def on_progress(fraction: float, note: str) -> None:
            f = max(0.0, min(1.0, float(fraction)))
            observe("progress", job.node_id,
                    {"fraction": f, "note": note,
                     "done": int(round(f * 1000)), "total": 1000})

        env = engine.env(job.pull_id)
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
        self._views.clear()
        # the remembered finished results of the unloaded nodes go too — a scrub would
        # otherwise re-arm a payload this call exists to release
        gone = set(node_ids)
        for key in [k for k in self._results if k[0] in gone]:
            self._results.pop(key, None)
        self._prefetch_gen += 1
        return n

    # ── internals ─────────────────────────────────────────────────────────────
    def _all_sources(self) -> Dict[str, Dict[str, Any]]:
        # `paths` rides alongside `path` rather than replacing it: a bundle card keeps a
        # usable single `path` (its first member), so every reader that predates bundles —
        # a saved graph opened by an older build, the LabLink protocol — still names a real
        # file instead of an empty one.
        out: Dict[str, Dict[str, Any]] = {}
        for rec in self.document.nodes.values():
            if rec.op_key != LOAD_OP:
                continue
            cfg: Dict[str, Any] = {"path": str(rec.params.get("path", "") or ""),
                                   # the card's access mode rides in the cfg beside its
                                   # path because that cfg is the whole of what the worker
                                   # thread is given about a source (`_resolve_source`);
                                   # reading `rec.modes` down there would mean touching the
                                   # document off the GUI thread.
                                   ACCESS_MODE: source_access_of(rec),
                                   # ...and so does the grouping lever, for the same reason
                                   # and with the same unset-means-default rule the document
                                   # applies (`group_descriptors`). The worker must not reach
                                   # into `rec.modes` off the GUI thread.
                                   GROUPING_MODE: str(
                                       rec.modes.get(GROUPING_MODE, GROUPING_DEFAULT)
                                       or GROUPING_DEFAULT)}
            # The card's own calibration ride along for the same reason the access mode
            # does: this cfg is the WHOLE of what the worker thread is told about a source,
            # and `_resolve_source` applies the override to the envelope that becomes both
            # the payload's metadata and the engine's meta-seed. Leaving them out is not a
            # missing feature but a DRIFT: the edit-time header (document.propagate reads
            # the params directly) would show the typed Z step while the pulled payload
            # still carried None, so the graph refused with a number the card said it had.
            for key in CALIB_OVERRIDE_KEYS:
                if key in rec.params:
                    cfg[key] = rec.params[key]
            members = rec.params.get(BUNDLE_PATHS_KEY)
            if isinstance(members, (list, tuple)) and len(members) >= 2:
                cfg[BUNDLE_PATHS_KEY] = [str(p or "") for p in members]
            out[rec.id] = cfg
        return out

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

    def _submit(self, node_id: str, coords, channels=None, *, fetch: bool = False) -> None:
        self._epoch += 1
        self._last_progress.clear()
        self._last_frame.clear()
        self._last_sweep.clear()
        graph = self.document.to_graph(for_run=True, materialize=True,
                                       unroll_iterate=True, sweep_all=self._sweep_all)
        pull_id = self._pull_id(node_id, graph)
        sources, every = self._sources_for(pull_id, graph)
        job = _Job(self._epoch, graph,
                   self.document.revision, node_id, coords, channels, sources,
                   pin=None if fetch else self._pin_for(coords), all_sources=every,
                   pull_id=pull_id)
        if fetch:
            self._fetches.add(self._epoch)
        self._busy = True
        self._active = job          # the job on the worker — `invalidate` latches its
        #                             cancel flag so the engine can abort mid-run
        # Registered LIVE for the whole run. `_deliver` accepts a result iff its id is still
        # here, and `invalidate` removes the ids it cancels — which is what lets several
        # branches be in flight without any of them retiring the others.
        self._runs[self._epoch] = node_id
        # The cone is stored as DOCUMENT ids: the run graph names iterate clones
        # `n#it@2` and inlined group bodies `b%inst`, and `invalidate` matches this set
        # against the ids the document reports touched — a delete or param edit on the
        # card `n` must hit a run that is computing `n`'s clones.
        self._run_cones[self._epoch] = frozenset(
            _doc_id_of(c) for c in self.planned_nodes(pull_id, graph))
        self.plan.emit(node_id, self.planned_nodes(pull_id, graph))
        (self.fetch_started if fetch else self.started).emit(node_id)
        self._pull_thread.start(_Worker(self, job))

    def _start_next(self) -> None:
        """Free the pull slot and start the next queued branch, if any.

        The single place ``_busy`` is cleared, so no delivery path can leave the queue
        stalled with work in it — a bake, a stale result and an ordinary one all come
        through here."""
        self._busy = False
        while self._queue:
            _nid, req = self._queue.popitem(last=False)
            if req[0] in self.document.nodes:
                self._submit(*req)
                return
            self.cancelled.emit(req[0])      # the node was deleted while it waited
        # only once no pull the user asked for is waiting: an editor's source fetch
        while self._fetch_queue:
            nid, _ = self._fetch_queue.popitem(last=False)
            if nid not in self.document.nodes:
                continue
            have = self.finished_result(nid)
            if have is not None:              # a user pull computed it while it waited
                self.fetched.emit(nid, have, 0.0)
                continue
            self._submit(nid, None, None, fetch=True)
            return

    def _pull_id(self, node_id: str, graph: Graph) -> str:
        """Which id in the RUN graph serves ``node_id``.

        Itself, except for a node inside an Iterate segment: the rewrite replaced it with
        one clone per iteration, so the card the user double-clicked has no id of its own
        any more and is served by the clone for the iteration the strip is on. Without this
        the pull would raise a bare KeyError on the node the user just clicked — the most
        ordinary thing to do while tuning a swept parameter is to look at the node being
        swept."""
        if node_id in graph.nodes:
            return node_id
        alias = self.document.iterate_aliases(sweep_all=self._sweep_all).get(node_id)
        return alias if alias and alias in graph.nodes else node_id

    def _deliver(self, packet) -> None:          # GUI thread (queued)
        (epoch, node_id, payload, plane, axes, dt, err, revision, coords, channels,
         pin) = packet
        self._run_cones.pop(epoch, None)
        # `_busy` is cleared by `_start_next` alone (see there), so every exit path below
        # frees the slot AND starts the next branch, and neither can be forgotten separately.
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
            self._runs.pop(epoch, None)
            self._start_next()
            return
        # staleness is judged BEFORE the re-seed side effect below: delivering a
        # resolved source envelope notifies the document → the window calls
        # invalidate() → and would drop the very result being delivered. The envelope
        # seeding is display-only (the ENGINE resolved its own meta seeds at run time),
        # so it never stales this result.
        #
        # LIVENESS, not "is this the newest" (2026-08-06). The old test was
        # ``epoch != self._epoch``, which meant any later request retired this result — so
        # queueing a second branch threw the first one's payload away the moment it landed,
        # and the user was left with nothing to look at. A run is stale only if something
        # actually cancelled it: an edit inside its own cone, or its node going away. Every
        # other run in flight is a SIBLING, not a successor.
        stale = self._runs.pop(epoch, None) is None
        # G8 live re-seed: hand newly resolved source envelopes to the document
        for nid, env in self._fresh_envs():
            self.document.set_meta_seed(nid, env)
        # A queued request for THIS SAME node does supersede this result — that is a newer
        # view of the node the user is looking at (a moved cursor, a changed channel set),
        # and showing the older one first would be a visible flicker backwards. A queued
        # request for a DIFFERENT node is simply the next branch and retires nothing.
        # A FETCH is exempt from that rule: a queued view of the same node does not make a
        # payload-only answer any less true — only a cancellation (an edit in its cone) does.
        fetch = epoch in self._fetches
        self._fetches.discard(epoch)
        if node_id in self._queue and not fetch:
            stale = True
        self._start_next()
        if stale:
            return
        if err is not None:
            self.failed.emit(node_id, err)
            return
        if fetch:
            # A payload-only pull: remembered like any finished branch (so a later click on
            # the node is served instantly), and handed to whoever asked. No view is held, no
            # plane decoded, no `finished` — so no pane is retargeted and none is left
            # showing a run that never delivers.
            if isinstance(payload, Dataset):
                self._results[(node_id, self.document.revision)] = (payload, axes)
                self._results.move_to_end((node_id, self.document.revision))
                while len(self._results) > _FINISHED_RESULTS:
                    self._results.popitem(last=False)
            self.fetched.emit(node_id, payload, dt)
            return
        # Hold the image provider so subsequent coords-only requests skip the engine
        # (the fast path). Tied to the CURRENT document revision — not the job's: the
        # G8 source re-seed just above (``set_meta_seed``) can bump the revision (display
        # metadata only, the graph/pixels are unchanged), and a genuine edit later runs
        # invalidate() → the held views clear, so the next request re-pulls.
        # Remember the finished branch so switching back to it costs nothing while another
        # branch is still computing (:meth:`_serve_finished`). Unpinned results only: a
        # frame-scoped payload holds ONLY those frames, so re-serving it later as if it were
        # the node's whole answer would quietly show a truncated series.
        if pin is None and err is None:
            self._results[(node_id, self.document.revision)] = (payload, axes)
            self._results.move_to_end((node_id, self.document.revision))
            while len(self._results) > _FINISHED_RESULTS:
                self._results.popitem(last=False)
        if isinstance(payload, Dataset) and payload.image is not None:
            self._hold_view(node_id, payload, pin=pin)   # which frame this held payload IS
        # The worker only ever ASSIGNS one overlay-context key; the bound is applied here,
        # on the GUI thread, so no reader can see a half-evicted map. Kept to the nodes with
        # a held view — those are the panes, and a context for anything else is unreachable.
        if len(self._overlay_ctxs) > HELD_VIEWS:
            for nid in [n for n in self._overlay_ctxs if n not in self._views]:
                self._overlay_ctxs.pop(nid, None)
        self.finished.emit(node_id, payload, plane, axes, dt)
        view = self._view_of(node_id)
        if coords is not None and view is not None and view.axes is not None:
            self.prefetch(node_id,
                          self._clamp_coords(self._payload_coords(coords, pin),
                                             view.axes),
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
        for nid in [n for n in self._views if n not in live]:
            self._views.pop(nid, None)

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

    def _store_path_for(self, path: str) -> str:
        """Where ``path``'s ``.b2nd`` store lives — beside the file, or under
        ``NODEGRAPH_STORE_DIR`` with a path-digest tag when the store is relocated (see
        :func:`~nodegraph.parallel.store_dir`). The one place this is computed, so an
        auto-access decision and an actual ingest cannot disagree about a file's store
        identity."""
        base = os.path.splitext(path)[0] + ".b2nd_store"
        target_dir = store_dir(os.path.dirname(base))
        if os.path.abspath(target_dir) == os.path.abspath(os.path.dirname(base)):
            return base
        tag = hashlib.blake2b(os.path.abspath(path).lower().encode("utf-8"),
                              digest_size=6).hexdigest()
        return os.path.join(
            target_dir, f"{os.path.splitext(os.path.basename(path))[0]}.{tag}.b2nd_store")

    def _effective_access(self, path: str, access: str) -> str:
        """``access`` unchanged unless it is :data:`ACCESS_AUTO`, in which case the
        concrete choice ``path`` resolves to — decided once and cached for the runner's
        lifetime.

        Caching matters here specifically: :meth:`source_state` is polled continuously by
        the canvas to paint each source card's status, and the decision below opens the
        file and stats a drive — cheap once, wasteful on every tick. A later free-space
        change or a store built by another means within the same session will not move an
        already-cached decision; that trade mirrors :attr:`_providers` itself, which is
        just as sticky for the same reason (a session-lifetime resolve, not a live poll).

        **A store that already exists and opens cleanly always wins**, without even
        asking :func:`~nodelab_v2.nd2_direct.decide_access` — a copy already sitting on
        disk has nothing left to protect against by reading around it, and "auto reads
        the file in place because the drive filled up with something unrelated after the
        ingest finished" would silently stop using a store the user already paid for.
        Only when there is nothing usable on disk does the decision turn on whether a
        FRESH ingest would fit."""
        if access != ACCESS_AUTO:
            return access
        cached = self._auto_access.get(path)
        if cached is not None:
            return cached
        from nodelab_v2.ingest import open_store
        store = self._store_path_for(path)
        resolved, reason = ACCESS_INGEST, "an existing store already covers this file"
        valid_store = False
        if os.path.isdir(store):
            try:
                open_store(store)
                valid_store = True
            except Exception:                      # noqa: BLE001 — torn/short store
                valid_store = False
        if not valid_store:
            from nodelab_v2.nd2_direct import decide_access
            resolved, reason = decide_access(path, os.path.dirname(store))
        self._auto_access[path] = resolved
        self._auto_access_reason[path] = reason
        return resolved

    def auto_access_reason(self, node_id: str) -> str:
        """Why this card's ``access=auto`` resolved the way it did — empty when the card
        is not set to auto, names a bundle (each member decides independently; there is
        no single reason for the card), or has not been decided yet (nothing has pulled
        or ingested it this session)."""
        rec = self.document.nodes.get(node_id)
        if rec is None or source_access_of(rec) != ACCESS_AUTO:
            return ""
        paths = _clean_source_paths(dict(rec.params))
        if len(paths) != 1:
            return ""
        return self._auto_access_reason.get(paths[0], "")

    def _resolve_source(self, node_id: str, cfg: Dict[str, Any],
                        epoch: Optional[int] = None, observe=None
                        ) -> Tuple[Any, MetaEnvelope]:   # worker thread
        """Resolve an ``io.load`` node's ``path`` to ``(provider, envelope)``, ingesting it
        to its ``.b2nd`` store on the first call and serving the cached provider after.

        Called from the pull worker AND from :class:`_IngestJob` on the ingest pool, so
        everything past the cache probe runs under the file's own lock: one ingest per file,
        and a caller that arrives mid-ingest waits and takes the result rather than starting
        a second writer on the same store.

        A card carrying 2+ ``paths`` is a **file bundle** and resolves through
        :meth:`_resolve_bundle` instead.

        ``observe`` overrides the progress sink (the ingest pool passes an un-epoched one);
        by default it reports under ``epoch``, or the current one."""
        paths = _clean_source_paths(cfg)
        access = str(cfg.get(ACCESS_MODE) or ACCESS_INGEST)
        if len(paths) >= 2:
            prov, env = self._resolve_bundle(node_id, paths, epoch, observe, access)
            # A bundle has no sidecar of its own — it is K files, and the grouping question
            # is about the concatenated M axis — so it detects from the joined geometry.
            env, note = _with_position_groups(_with_card_calib(env, cfg), "", {},
                                              cfg.get(GROUPING_MODE, GROUPING_DEFAULT))
            self._group_note[node_id] = note
            return prov, env
        path = paths[0] if paths else ""
        key, res = self._resolve_one(node_id, path, epoch, observe, access)
        self._node_source_key[node_id] = key
        prov, env = res
        # A single file names itself too (V3.02), the way a bundle's members always have.
        # Stamped HERE rather than inside `_resolve_one` only for symmetry with the bundle
        # branch above -- it is derived from the path, which that cache is keyed on, so
        # either side would be correct. `util.chain` needs it to order separately loaded
        # files by the counting number in their names; without it the filename lives only
        # in the node's params, where no compute can reach it.
        env = stamp_source_file(env, path)
        env, note = _with_position_groups(_with_card_calib(env, cfg), path,
                                          self._channel_display.get(key, {}),
                                          cfg.get(GROUPING_MODE, GROUPING_DEFAULT))
        self._group_note[node_id] = note
        return prov, env

    def _resolve_one(self, node_id: str, path: str, epoch: Optional[int], observe,
                     access: str = ACCESS_INGEST
                     ) -> Tuple[Any, Tuple[Any, MetaEnvelope]]:   # worker thread
        """``(cache key, (provider, envelope))`` for ONE file — the cache/lock/ingest core.

        Split out of :meth:`_resolve_source` so a bundle's members go through exactly this
        path: each file keeps its own store, its own lock and its own cache entry, so a file
        that appears both on its own card and inside a bundle is ingested once and shared.
        It deliberately does NOT stamp :attr:`_node_source_key` — for a bundle the node's
        key is the bundle's, and stamping per member would leave it naming the last file.

        ``access`` is resolved (:meth:`_effective_access`) BEFORE the key is built — a
        bundle's members can resolve differently from each other (one may already have a
        store, another may not fit the drive), which is exactly why this lives here and
        not in :meth:`_resolve_bundle`: one call per file, one decision per file."""
        if path and not os.path.isfile(path):
            raise FileNotFoundError(
                f"No such ND2/TIFF file:\n  {path!r}\n"
                f"Reload it via File → Load ND2/TIFF file… (or fix the node's 'path' "
                f"field). Leave it empty for the synthetic demo source.")
        # There is nothing for `access` to mean for the synthetic source — no file, no
        # store, no drive to check — so it is normalized to a concrete value rather than
        # resolved: ACCESS_INGEST reads correctly either way, and it keeps the invariant
        # every OTHER caller of `source_key`/`_ingest_locked` relies on (never literal
        # "auto") true here too, instead of adding a second empty-path exemption.
        resolved = self._effective_access(path, access) if path else ACCESS_INGEST
        key = source_key(path, resolved)
        hit = self._providers.get(key)
        if hit is not None:
            return key, hit
        with self._source_lock(key):
            hit = self._providers.get(key)
            if hit is not None:
                return key, hit  # someone else ingested it while we waited on the lock
            return key, self._ingest_locked(node_id, path, key, epoch, observe, resolved)

    def _resolve_bundle(self, node_id: str, paths: Sequence[str],
                        epoch: Optional[int], observe,
                        access: str = ACCESS_INGEST
                        ) -> Tuple[Any, MetaEnvelope]:   # worker thread
        """Resolve a **file bundle** — K files laid end to end on the multipoint axis.

        Every member resolves through :meth:`_resolve_one`, so the bundle costs nothing
        beyond the files themselves and shares their stores with any other card using them.
        The bundle's own cache entry is keyed on the ordered member paths, so re-pulling it
        is a hit and a re-ordered bundle is not.

        The member locks are taken one at a time INSIDE the bundle's own lock; the keys are
        disjoint by construction (``("bundle", …)`` vs ``("image", …)``), so this cannot
        deadlock against a concurrent single-file pull of the same file."""
        key = bundle_key(paths, access)
        self._node_source_key[node_id] = key
        hit = self._providers.get(key)
        if hit is not None:
            return hit
        with self._source_lock(key):
            hit = self._providers.get(key)
            if hit is not None:
                return hit
            provs: List[Any] = []
            envs: List[MetaEnvelope] = []
            member_keys: List[Any] = []
            for p in paths:
                mkey, (mprov, menv) = self._resolve_one(node_id, p, epoch, observe,
                                                        access)
                provs.append(mprov)
                envs.append(menv)
                member_keys.append(mkey)
            labels = _unique_labels(paths)
            prov = MultiSourceProvider(provs, labels=labels)
            env = bundle_envelope(prov.axes, envs, labels)
            self._providers[key] = (prov, env)
            # The members share a grid, so they share their channel display; taking the
            # first is not a choice between disagreeing values.
            self._channel_display[key] = dict(
                self._channel_display.get(member_keys[0], {}))
            return prov, env

    @staticmethod
    def _open_direct(path: str) -> Tuple[Any, MetaEnvelope]:
        """Open ``path`` as a no-ingest :class:`~nodelab_v2.nd2_direct.Nd2DirectProvider`.

        A failure here **raises** rather than falling back to the ingest. ``direct`` is an
        explicit per-card choice, and the thing it is chosen to avoid is a copy that can
        run for hours on a series too big to duplicate — so silently doing that copy
        anyway, because the file turned out to be compressed or oddly shaped, is the one
        outcome the user most needs not to get by surprise. The provider's messages all end
        by naming the way out ("Load it normally instead"), and the card shows them."""
        from nodelab_v2.ingest import _is_tiff, read_calibration
        from nodelab_v2.nd2_direct import Nd2DirectError, Nd2DirectProvider
        if _is_tiff(path):
            raise Nd2DirectError(
                f"{os.path.basename(path)} is a TIFF; direct reading is ND2-only "
                f"(tifffile has no memory-mapped frame path here). Set this card's "
                f"Access back to 'ingest'.")
        prov = Nd2DirectProvider(path)
        env = MetaEnvelope(axes=prov.axes, metadata=read_calibration(path))
        return prov, env

    def _ingest_locked(self, node_id: str, path: str, key: Any,
                       epoch: Optional[int], observe,
                       access: str = ACCESS_INGEST
                       ) -> Tuple[Any, MetaEnvelope]:   # worker thread, holding the key lock
        # `access` must already be resolved — `_resolve_one` is this method's only caller
        # and calls `_effective_access` before building `key`, so an `auto` reaching here
        # would mean two callers disagreeing about what `key` even means.
        assert access != ACCESS_AUTO, "_ingest_locked() needs a RESOLVED access"
        if not path:
            prov = SyntheticProvider(_SYNTH_AXES, tile=128)
            env = MetaEnvelope(axes=_SYNTH_AXES, metadata=dict(_SYNTH_META))
            disp = {"channel_names": [f"Ch{i}" for i in range(_SYNTH_AXES.c)],
                    "channel_emission_nm": list(_SYNTH_META["channel_emission_nm"])}
        else:
            from nodelab_v2.ingest import (
                PYRAMID_LEVELS, ensure_store_levels, ingest_image, open_store,
                read_calibration, read_channel_display)
            if access == ACCESS_DIRECT:
                # No store, no ingest, nothing written beside the file: the provider
                # memory-maps the ND2's frames where they are. Deliberately BEFORE the
                # store probe below — a card switched to direct must not be held up by a
                # torn store left behind by an ingest that was abandoned precisely because
                # the file was too big to copy.
                prov, env = self._open_direct(path)
                self._providers[key] = (prov, env)
                self._channel_display[key] = read_channel_display(path)
                return prov, env
            # Beside the source file by default. `NODEGRAPH_STORE_DIR` moves every store to
            # one directory instead, which matters when the data lives on slow media: a USB
            # SSD here measured 27 MB/s of store write against 56 MB/s on the internal
            # NVMe, and 55 MB/s of raw sequential write against 952 MB/s. The store is a
            # derived cache, so relocating it costs nothing but has to stay UNIQUE per
            # source — hence the path digest, or two files of the same basename in
            # different folders would fight over one store (:meth:`_store_path_for`).
            store = self._store_path_for(path)
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
           "PlaneCache", "ingest_workers", "source_key", "bundle_key",
           "bundle_envelope", "BUNDLE_PATHS_KEY"]
