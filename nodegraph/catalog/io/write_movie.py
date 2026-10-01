"""Export Movie (``io.write_movie``) — render the Dataset on the wire to an MP4, a GIF or a
numbered image sequence, with the acquisition's own clock burned into the corner."""

from __future__ import annotations


import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.metadata import parse_channels
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, InInt, InString,
                                Mode, OutDataset)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.movie_draw import (  # noqa: F401 -- some re-exported
    _TEXT_RGB, _emission_rgb, _even, _micro, _scalebar, _time_text)
from nodegraph.catalog._shared.movie_timeline import (
    MovieSource, MovieStyle, Timeline, flat_spec)

# ── Export Movie (io — the pipeline's *presentation* write end) ────────────────
#
# `io.write_tiff` is the archival export: it writes the PIXELS, losslessly, and hands the
# calibration to Fiji so someone can measure them again. This node is the other half, and it
# answers a different question — "what does this look like?". Its product is lossy, 8-bit and
# RGB, and that is not a compromise, it is the point: it goes in a talk, a figure panel, a
# Slack message to the person who asked what the drug did.
#
# Three consequences run through every decision below.
#
#   1. **A movie is 2-D and 8-bit; a Dataset is 6-D and usually 16-bit.** So this node has to
#      make display decisions that `write_tiff` never does: which axis is time, how the
#      channels composite, where the black and white points sit. It makes them ONCE and holds
#      them fixed for every frame — see `_window`. A per-frame auto-contrast is the single
#      most common way to ruin a timelapse, because it turns "the cells got brighter" into a
#      constant mid-grey and makes dead frames pulse.
#
#   2. **The burn-ins are read off the calibration envelope, not typed by hand.** The frame
#      interval comes from `dt_s`, which `nodelab_v2.ingest` already derives from the ND2's
#      per-frame timestamps; the scale bar comes from `pixel_size_um`; the channel tints come
#      from `channel_emission_nm` through the same Bruton map the Viewer tints with. A number
#      typed into a video caption by hand is a number nobody can check later.
#
#   3. It is a TAP, like `write_tiff` — `Dataset -> the SAME Dataset`, byte-identical. Drop it
#      mid-chain and a Viewer after it still draws what was written, which is the only way to
#      check an export without opening the file.

#: Longest-edge cap, in px, above which ``max_px=0`` ("full size") is still refused for GIF.
#: A GIF is built in RAM as a list of full-size RGB frames before PIL can write it (the
#: format needs a global palette, so there is no streaming encoder), and a 240-frame 6554²
#: series is 3 TB of uint8. MP4 and the image sequence stream and have no such ceiling.
_GIF_FULL_SIZE_LIMIT = 2048

#: ``format`` mode -> (default extension, whether it is one file or a numbered sequence).
_FORMATS = {"mp4": (".mp4", False), "gif": (".gif", False),
            "png": (".png", True), "jpeg": (".jpg", True)}

def _writer_backends():
    """OpenCV writer backends that can be ASKED for H.264, best-first for this platform.

    An explicit allowlist, not ``videoio_registry.getWriterBackends()``, and that is the
    load-bearing part: the registry also offers ``CV_IMAGES`` and ``CV_MJPEG``, which open
    happily for an ``.mp4`` path and then write a PNG sequence or a Motion-JPEG AVI under
    it. A probe loop that trusted the registry would "succeed" into a file that is not the
    format its name claims.

    MSMF first on Windows — see :func:`_open_writer` for why the order is the whole fix.
    """
    import sys
    import cv2
    order = (("CAP_MSMF", "CAP_FFMPEG") if sys.platform == "win32"
             else ("CAP_FFMPEG", "CAP_MSMF"))
    try:
        available = set(cv2.videoio_registry.getWriterBackends())
    except Exception:          # noqa: BLE001 — no registry: try them and let open() decide
        available = None
    out = []
    for name in order:
        api = getattr(cv2, name, None)
        if api is not None and (available is None or api in available):
            out.append((name, api))
    return out


def _open_writer(fmt: str, path: str, fps: float, size: Tuple[int, int], quality: int):
    """An open MP4 writer for ``(width, height)`` ``size``, or raise with what to do instead.

    **H.264 (``avc1``), not ``mp4v``** — and the difference matters more than it looks. Both
    open without complaint here and both produce a playable ``.mp4``, but ``mp4v`` is MPEG-4
    Part 2, which Chrome, Firefox, PowerPoint and Keynote will not play: the file looks fine
    in VLC on the machine that wrote it and is a black rectangle in the talk. So the default
    is the one that plays where the movie is going.

    **The backend is named explicitly rather than left to OpenCV's auto-chain**, and that is
    what keeps the console clean. Measured here 2026-09-25: OpenCV's default order tries
    FFmpeg first, FFmpeg's only H.264 encoder on this build is ``libopenh264``, and
    ``openh264-2.5.0-win64.dll`` is not installed — so every single export printed

        Failed to load OpenH264 library: openh264-2.5.0-win64.dll
        [libopenh264 @ …] Unable to create encoder
        [ERROR:0@…] VIDEOIO/FFMPEG: Failed to initialize VideoWriter

    and then **succeeded anyway**, because OpenCV silently fell through to Media Foundation,
    which encodes H.264 natively on Windows. Asking for ``CAP_MSMF`` first produces a
    BYTE-IDENTICAL file (verified: same 28,401 bytes, same ``avc1``/``avcC`` sample-entry
    boxes, all frames decode) with none of that output. The noise was never a symptom of a
    broken export — it was OpenCV loudly failing at something it did not need to do.

    The order is per-platform because the right answer inverts: MSMF does not exist off
    Windows, where FFmpeg is the one that works (a Linux OpenCV normally links a usable
    x264/openh264), so the LabLink worker takes the same code path to the opposite backend.

    ``mp4v`` remains the last resort. It is reached only when no backend will encode H.264 at
    any size, and it is a real downgrade — hence the ``ValueError`` below rather than a
    silent third fallback when even that fails.

    **There is deliberately no attempt to SUPPRESS the messages**, only to stop provoking
    them. ``cv2.utils.logging.setLogLevel(LOG_LEVEL_SILENT)`` was tried and measured to have
    no effect on them (2026-09-25: byte-identical output with and without, because videoio
    emits these below the level that switch gates), and redirecting fd 2 around the probe
    would also swallow genuine errors from an encoder that is mid-write. So the remaining
    case that still prints — a Windows box where MSMF itself fails and FFmpeg is tried — is
    one where the output is *informative*, because something really is wrong there.
    """
    import cv2
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    for _name, api in _writer_backends():
        writer = cv2.VideoWriter(path, api, fourcc, float(fps), (w, h))
        if writer.isOpened():
            return writer
        writer.release()
    # No backend would take H.264. Fall back to MPEG-4 Part 2, which almost anything
    # encodes — a worse file, but a file, and the docstring above says what it costs.
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if writer.isOpened():
        return writer
    writer.release()
    raise ValueError(
        f"Export Movie could not open any MP4 encoder for a {w}x{h} frame (tried H.264 on "
        f"{', '.join(n for n, _ in _writer_backends()) or 'no available backend'}, then "
        f"MPEG-4). The usual cause is a frame edge too small for the codec — this node "
        f"already pads to even dimensions, so if you are seeing this the frame is probably "
        f"tiny. Raise `Max px`, or set Format to 'gif' or 'png'.")


def _seq_path(path: str, index: int, total: int, ext: str) -> str:
    """``out.png`` -> ``out_0007.png`` — one numbered file per frame.

    Zero-padded to the width of the largest index so the frames sort in playback order in
    every file browser, every glob and every ``ffmpeg -pattern_type glob`` — which is the
    whole reason to write a sequence rather than a video.
    """
    stem = path[: -len(ext)] if path.lower().endswith(ext.lower()) else path
    for other in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".mp4"):
        if stem.lower().endswith(other):
            stem = stem[: -len(other)]
            break
    return f"{stem}_{index:0{len(str(max(0, total - 1)))}d}{ext}"


def _part_path(path: str) -> str:
    """``out.mp4`` -> ``out.part.mp4`` — the temporary name a write lands on before it is
    renamed into place.

    ``.part`` goes BEFORE the extension, not after it, and that is not cosmetic. Unlike
    tifffile, which is told its format explicitly, ``cv2.VideoWriter`` and ``PIL.Image.save``
    infer the container from the filename's EXTENSION — so the obvious ``out.mp4.part``
    presents as a file of type ``.part``, which OpenCV has no muxer for, and the writer
    refuses to open with no useful diagnostic. Keeping the real extension last makes the
    atomic-write pattern and the extension-sniffing encoders compatible.
    """
    stem, ext = os.path.splitext(path)
    return f"{stem}.part{ext}" if ext else f"{path}.part"


#: Extensions this node will REPLACE when it forces the path to match the chosen Format.
#: A closed list rather than "strip whatever follows the last dot", because a path like
#: ``WellA3_t0-240_1.7183um`` has a dot in it and is not carrying an extension at all —
#: splitting it there would write ``WellA3_t0-240_1.mp4`` and lose the calibration from the
#: name the user chose deliberately.
_REPLACEABLE_EXT = (".mp4", ".gif", ".png", ".jpg", ".jpeg", ".avi", ".mov", ".mkv",
                    ".webm", ".m4v", ".tif", ".tiff")


def _with_ext(path: str, ext: str) -> str:
    """Force ``path`` to end in ``ext``, replacing any media extension it already carries.

    **This is the fix for "I chose GIF and got an MP4".** The writers are told their format
    explicitly (``Image.save(..., format="GIF")``), so the BYTES were always right — but the
    filename was whatever the Browse dialog last produced, so a GIF landed as ``movie.mp4``
    and every player on the machine refused it as a corrupt video. The file was a valid GIF
    with a lying name, which is worse than a failure: nothing reported an error.

    Forcing the extension here rather than validating it and refusing, because the Format
    Mode is unambiguous about what is being written and the path is the thing that goes
    stale — you pick a destination once and then change your mind about the format, which is
    exactly the sequence that produced the bug.
    """
    low = path.lower()
    for known in _REPLACEABLE_EXT:
        if low.endswith(known):
            return path[: -len(known)] + ext
    return path if low.endswith(ext.lower()) else path + ext


def _position_path(path: str, m: int, n: int, ext: str) -> str:
    """``out.mp4`` -> ``out_m03.mp4`` — the per-position filename, padded to sort in order.

    The same rule as :func:`nodegraph.catalog.io.write_tiff._position_path`, so a movie export
    and a TIFF export of the same graph produce files whose names line up.
    """
    stem = path[: -len(ext)] if path.lower().endswith(ext.lower()) else path
    return f"{stem}_m{m:0{len(str(max(0, n - 1)))}d}{ext}"


class MoviePlan:
    """Everything needed to turn a frame index into the finished RGB frame the export writes.

    Exists so that the **preview and the export cannot drift**. A preview that re-implements
    the compositing, the window or the burn-ins is a preview of a different movie, and the
    whole value of looking before exporting is that what you looked at is what you get. So
    there is one renderer and both callers go through it: :func:`_compute_write_movie`
    resolves the settings from its ``EvalContext``, and :mod:`nodelab_v2.movie_preview`
    resolves the same names from the node's saved state and the payload's envelope.

    Since 2026-09-30 the renderer itself is the timeline compositor
    (:class:`nodegraph.catalog._shared.movie_timeline.Timeline`). A flat movie is the
    one-clip, one-panel timeline :func:`flat_spec` builds, one per position, so a flat
    export and a timeline export draw through the same code. This class is the thin adapter
    that keeps the flat node's surface (``render(m, i)``, a settable ``max_px``) in front of it.

    It deliberately knows nothing about FILES — no path, no container, no codec, no split.
    Those are the export's business, the preview has no use for them, and keeping them out is
    what lets a preview run before any destination has been chosen.
    """

    __slots__ = ("ds", "ax", "chans", "ch_names", "sweep", "n_frames", "fps", "show",
                 "interval_s", "_max_px", "_sources", "_spec_args", "_style_args", "_tl")

    @property
    def max_px(self) -> int:
        return self._max_px

    @max_px.setter
    def max_px(self, value: int) -> None:
        # the canvas size is decided when a timeline is built, so a new cap rebuilds them
        self._max_px = int(value or 0)
        self._tl = {}

    def position_name(self, m: int) -> str:
        """The multipoint's name, or a positional fallback so the burn-in is never blank."""
        return self._sources["A"].position_name(m)

    def timeline(self, m: int) -> Timeline:
        """The one-clip timeline for position ``m``, built on first use and then kept."""
        tl = self._tl.get(m)
        if tl is None:
            spec = flat_spec(m=m, **self._spec_args)
            style = MovieStyle(max_px=self._max_px, **self._style_args)
            tl = self._tl[m] = Timeline(spec, self._sources, style)
        return tl

    def window(self, m: int, progress=None) -> List[Tuple[float, float]]:
        """The per-channel display window for position ``m``, measured once and CACHED.

        The cache is the point rather than an optimization: the window has to be ONE decision
        for the whole movie (see the module note), so every frame of a position must use the
        same numbers, and the only way to guarantee that is for there to be one set of them.
        ``progress`` is ticked once per sampled frame.
        """
        tl = self.timeline(m)
        tl.measure(progress)
        return tl.first_windows()

    def render(self, m: int, i: int, progress=None) -> np.ndarray:
        """Frame ``i`` of position ``m``, finished: composited, resized and annotated.

        The order is load-bearing, and it is the export's own: resize BEFORE annotate, so the
        text is sized in OUTPUT pixels and does not turn into an illegible smear when a
        6554-px frame is scaled down to a 1200-px movie.
        """
        tl = self.timeline(m)
        if progress is not None:
            tl.measure(progress)
        return tl.render(i)


def plan_movie(ds, *, ch, modes, calib, meta, layer: str = "") -> MoviePlan:
    """Resolve a Dataset plus this node's flat settings into a ready :class:`MoviePlan`.

    ``ch`` is a channel-like READER — anything exposing ``.param(name)``. The engine hands
    over its own ``ctx.channel(0)``, which is what keeps the ``derive`` on ``interval_s``
    firing and the envelope reads memo-fenced; the GUI passes an equivalent shim over the
    node's saved state (:func:`nodelab_v2.movie_preview.resolve_params`). ``calib`` / ``meta``
    stay plain callables because nothing inspects those.

    **It is an object with a ``.param`` method, not a bare callable, and that is not style.**
    ``selftest::_param_key_index`` finds a node's param reads by AST, matching a call whose
    function is an attribute named ``param`` — so ``ch.param("fps")`` is visible and a bare
    ``P("fps")`` is not. Passing the callable directly made all the value sockets read as
    dead controls and failed the socket contract, which is exactly the gate working: the
    keys really had become unfindable, and a later reader would have had no way to tell which
    sockets this node still honours.

    Every refusal raised here is one the preview should make too: a channel list that selects
    nothing, a layer that is not on the wire, a sweep axis that is not a sequence. Far better
    to say so in a dialog than at the end of a long export.
    """
    ax = ds.axes
    sweep = modes.get("sweep", "time")
    if layer:
        got = ds.get(Domain.VOXEL, layer)
        if got is None:
            have = sorted({a.name for a in ds.layers_on(Domain.VOXEL)})
            raise ValueError(
                f"no Voxel layer {layer!r} to render — this Dataset carries {have or 'none'}. "
                f"Leave `layer` empty to render the IMAGE instead.")
        shape = np.shape(got.values)
        if shape != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
            raise ValueError(
                f"Voxel layer {layer!r} has shape {shape}, which does not match the "
                f"Dataset's axes {(ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)}.")
    elif ds.image is None:
        raise ValueError("Export Movie has nothing to render: no image on the input Dataset "
                         "and no `layer` naming a Voxel raster.")

    picked = parse_channels(ch.param("channels"))
    chans = list(range(ax.c)) if picked is None else [c for c in picked if 0 <= c < ax.c]
    if not chans:
        raise ValueError(
            f"Export Movie: `channels` selected nothing in range — this Dataset has "
            f"{ax.c} channel(s), indexed 0..{ax.c - 1}. Leave it empty for all of them.")

    if sweep == "time":
        n_frames = ax.t
        if n_frames < 2:
            raise ValueError(
                "Export Movie: Sweep = 'time' needs more than one timepoint and this Dataset "
                "has T=1. Set Sweep to 'z' to fly through the stack, or set Format to 'png' "
                "for a still.")
    else:
        n_frames = ax.z
        if n_frames < 2:
            raise ValueError(
                "Export Movie: Sweep = 'z' needs more than one Z slice and this Dataset has "
                "Z=1. Set Sweep to 'time' to play the timelapse.")
        if ax.t > 1:
            raise ValueError(
                f"Export Movie: Sweep = 'z' plays the Z axis, but this Dataset also has "
                f"{ax.t} timepoints and a movie has only one axis to play. Pick a timepoint "
                f"first with `util.crop` (Frames mode, e.g. `t0`), set Sweep to 'timeline' "
                f"to interleave them, or set Sweep to 'time'.")

    p = MoviePlan()
    p.ds, p.ax, p.chans, p._tl = ds, ax, chans, {}
    p._sources = {"A": MovieSource("A", ds, calib=calib, meta=meta)}
    p.ch_names = [p._sources["A"].channel_name(c) for c in chans]
    p.sweep, p.n_frames = sweep, n_frames
    p.interval_s = float(ch.param("interval_s") or 0.0)
    p.fps = float(ch.param("fps") or 10.0)
    p.show = {"frame": bool(ch.param("show_frame")),
              "time": bool(ch.param("show_time")) and sweep == "time",
              "scalebar": bool(ch.param("show_scalebar")),
              "channels": bool(ch.param("show_channels")),
              "position": bool(ch.param("show_position"))}
    p._spec_args = dict(
        sweep=sweep, z_reduce=modes.get("z_reduce", "max"), channels=chans, layer=layer,
        contrast=modes.get("contrast", "auto"),
        low_pct=float(ch.param("low_pct") or 0.0),
        high_pct=float(ch.param("high_pct") or 100.0),
        black=float(ch.param("black") or 0.0), white=float(ch.param("white") or 0.0),
        gamma=float(ch.param("gamma") or 1.0),
        brightness=float(ch.param("brightness") or 100.0), show=p.show)
    p._style_args = dict(
        corner=str(ch.param("corner") or "top_left"), font_px=int(ch.param("font_px") or 0),
        text_rgb=_TEXT_RGB.get(str(ch.param("text_color")), (255, 255, 255)),
        interval_s=p.interval_s)
    p.max_px = int(ch.param("max_px") or 0)
    return p


def _compute_write_movie(ctx: EvalContext) -> Dataset:
    """Render the Dataset (or one of its Voxel layers) to a movie, and hand it through.

    Resolved spec (§0 grill, 2026-09-25)
    ------------------------------------
    * **Kind** io / side effect → ``op_key="io.write_movie"``, category ``"io"``. The
      counterpart to ``io.write_tiff``: that one writes the pixels for measuring, this one
      writes a picture of them for showing.

    * **Data contract** ``Dataset -> the SAME Dataset``, byte-identical: same provider, same
      axes, same calibration, same layers. Not axis-changing, so **no** ``meta_transform``,
      and nothing downstream can tell an export happened. A tap, not a terminal.

    * **2D/3D** no lever, and deliberately so. The dimension question this node actually asks
      is not "run the kernel per-plane or per-volume" but "**which axis is playback time**",
      which the ``sweep`` Mode asks directly. A `DimMode` here would be a second control
      saying something adjacent and non-equivalent, and `wire-node-v2` §5 is explicit that a
      lever which cannot disagree with the data should not be asked for.

    * **Footprint** ``{"time": WHOLE_VOLUME, "z": WHOLE_PLANE}`` keyed by ``sweep``
      (``footprint_mode="sweep"``), with ``kernel_axes`` to match. Honest per branch: sweeping
      time, one output frame is a whole ``(Z, Y, X)`` column reduced by ``z_reduce``, so the
      kernel reaches across z; sweeping z, one output frame IS one plane. Note that Z=1 — the
      2-D timelapse, which is most of them — makes the ``time`` branch a plane at identical
      cost, so the claim costs nothing in the common case.

      It is NOT ``WHOLE_SERIES``, even though the contrast pre-pass looks along the whole
      sweep axis: that pass accumulates percentile SAMPLES, one plane at a time through
      ``provider.get_region``, and never holds two frames. Peak memory is one frame, or one
      Z-column, exactly as declared — which is what makes a 49-position 6554² file
      exportable. Same reasoning as ``write_tiff``'s ``WHOLE_PLANE``.

    * **Sockets** ``path`` (``save_file``), ``layer`` (Voxel, empty = the image), plus the
      display and burn-in controls. Modes: ``format``, ``sweep``, ``z_reduce`` (gated to
      ``sweep="time"``), ``contrast``, ``split``, ``existing``.

    * **Timeline (2026-09-30)** ``sweep="timeline"`` plays the JSON spec in the ``timeline``
      socket, written by the Movie Editor (``nodelab_v2.movie_editor``) and rendered by
      :class:`~nodegraph.catalog._shared.movie_timeline.Timeline`. Two more Dataset inputs,
      ``source_b``/``source_c``, declared after ``data`` (INV-09), not gated (a hidden
      socket's wire is invisible and still pulled), and ``passes_domains=False`` because
      they are read and never merged: the output is still ``data``. Every per-clip flat
      control is gated to ``sweep`` in ``{time, z}``; the movie-wide ones (fps, Max px, text,
      file, Format, Existing, Frame interval for A's clock) stay live in both. Footprint
      ``MULTI_VIEW`` over ``{m, t, z, y, x}``: a grid frame gathers several positions, a
      contact sheet several timepoints, a max-Z panel a Z column. The flat branch now renders
      through the same compositor (a one-clip timeline per position), bit-identical to the
      renderer it replaced except that a downscaled frame's scale bar is now measured in
      OUTPUT pixels; it was drawn in source pixels, i.e. too long by the downscale factor.

    * **Backend** OpenCV 5.0.0 for the MP4 (``cv2.VideoWriter``, fourcc ``avc1`` — see
      :func:`_open_writer` for why not ``mp4v``) and for the area-averaged downscale; Pillow
      12.3.0 for the GIF, the PNG/JPEG sequence and the text. Both already dependencies, both
      re-verified in-env on 2026-09-25. ``imageio-ffmpeg`` and ``av`` are NOT installed, which
      is why the MP4 path goes through OpenCV rather than the more obvious ``imageio``.

    * **Numba** no. The hot path is a handful of vectorized numpy ops per frame plus the
      codec's own C encoder; there is no Python loop over small arrays to fuse.

    The display window is resolved ONCE
    -----------------------------------
    Everything about how this node looks is decided before the first frame is written, and
    then held fixed — see :func:`_window`. A per-frame auto-contrast is the classic way to
    destroy a timelapse: it normalizes away the very change the movie exists to show, and it
    makes the background pulse. The cost is one subsampled pre-pass, reported on the progress
    rail as its own phase so a long export does not look stalled.

    Re-running
    ----------
    A memo HIT returns the cached payload without entering this function at all
    (``engine.py``), so an unchanged graph re-pulled in one session writes nothing.

    Unlike ``write_tiff``, ``existing`` has no ``"skip"``: that option is only safe there
    because a TIFF has a private tag to hold an export STAMP, and an MP4 and a GIF have no
    comparable place to put one that survives every player and editor. Offering a ``skip``
    backed by "a file exists with this name" would mean shipping a stale movie under the name
    of a fresh one — precisely the failure ``write_tiff``'s stamp exists to prevent — so the
    honest options are ``overwrite`` (the default) and ``refuse``.

    Each file is written to ``<path>.part`` and renamed on success, so an interrupted export
    cannot leave a truncated movie wearing the real name.
    """
    ds = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {})
    # Every value param is read as `ch.param(name)`, never `ctx.params.get(name, <fallback>)`.
    # `ChannelContext.param` resolves override -> `derive` -> the SocketSpec default, which
    # buys two things an inline fallback does not. (1) The default lives in exactly one
    # place, the socket, instead of a second copy here that drifts. (2) The `derive` is
    # actually evaluated at pull time and its envelope reads are memo-fenced — which is what
    # makes `interval_s` auto-fill from the file's `dt_s` on a headless pull, and makes a
    # graph re-exported after the frame interval changed re-render instead of hitting a
    # stale memo. Read with `ctx.params.get` these sockets looked right and quietly clocked
    # every movie at 0 s.
    #
    # Spelled out at every call site rather than aliased to a short local (`P = ch.param`),
    # which is tempting with nineteen of them. `selftest::_param_key_index` finds param reads
    # by AST, matching a CALL whose func is an attribute named `param` — an alias turns every
    # read into a bare `P(...)` the index cannot see, which silently empties this node's
    # `params_read` and makes the socket contract gate vacuous for it. The gate would stop
    # proving that all 21 sockets are live.
    ch = ctx.channel(0)
    fmt = modes.get("format", "mp4")
    sweep = modes.get("sweep", "time")
    z_reduce = modes.get("z_reduce", "max")
    contrast = modes.get("contrast", "auto")
    split = modes.get("split", "none")
    existing = modes.get("existing", "overwrite")

    path = str(ch.param("path") or "").strip()
    if not path:
        raise ValueError(
            "Export Movie has no output path. Use Browse… on the File socket to pick where "
            "the movie is written — this node puts bytes somewhere permanent and will not "
            "guess a location.")
    ext, is_sequence = _FORMATS[fmt]
    # The Format Mode owns the extension, not the path. Applied ONCE, here, so that every
    # name derived downstream (`_position_path`, `_seq_path`, `_part_path`) starts from a
    # path already agreeing with the encoder — see :func:`_with_ext` for the bug this fixes.
    path = _with_ext(path, ext)

    # Everything about HOW the picture is made is resolved once, here, into the shared
    # renderer. The preview builds the same objects from the node's saved state, so what it
    # shows is produced by this exact code rather than by a second implementation that
    # agrees with it today. Either branch ends as a list of (file, Timeline): a flat movie
    # is one one-clip timeline per position, a timeline movie is the one the editor laid out.
    unused: List[str] = []
    if sweep == "timeline":
        movie = _timeline_movie(ctx, ds, ch)
        targets = [(path, movie)]
        max_px = movie.style.max_px
        fps = float(ch.param("fps") or 10.0)
    else:
        plan = plan_movie(ds, ch=ch, modes=modes, calib=ctx.calib, meta=ctx.meta,
                          layer=ctx.layer("layer"))
        max_px, fps = plan.max_px, plan.fps
        unused = [s for s in ("source_b", "source_c") if ctx.input(s) is not None]
        # ── which positions, and therefore which files ────────────────────────
        if split == "position":
            targets = [(_position_path(path, m, ax.m, ext), plan.timeline(m))
                       for m in range(ax.m)]
        elif ax.m > 1:
            raise ValueError(
                f"Export Movie: this Dataset has {ax.m} positions and a movie holds one. Set "
                f"Split to 'position' to write {ax.m} files, pick one position first with "
                f"`util.crop` (Frames mode, e.g. `m0`), or set Sweep to 'timeline' and tile "
                f"the positions into one grid.")
        else:
            targets = [(path, plan.timeline(0))]
    n_frames = targets[0][1].n_frames

    if existing == "refuse":
        clash = [p for p, _ in targets if os.path.exists(p)]
        if clash:
            raise ValueError(
                "Export Movie refuses to overwrite an existing file (Existing = 'refuse'): "
                + ", ".join(os.path.basename(p) for p in clash[:3])
                + (" …" if len(clash) > 3 else "")
                + ". Change the path, or set Existing to 'overwrite'.")

    if fmt == "gif" and (not max_px or max_px > _GIF_FULL_SIZE_LIMIT):
        raise ValueError(
            f"Export Movie: a GIF needs a global palette, so every frame is held in RAM "
            f"before it can be written — {n_frames} frames at this size would not fit. Set "
            f"`Max px` to {_GIF_FULL_SIZE_LIMIT} or less (1200 is the default), or use "
            f"Format = 'mp4', which streams and has no size ceiling.")

    quality = int(ch.param("quality") or 80)

    # ── phase 1: the display windows, measured once per movie ────────────────
    total_work = sum(tl.measure_steps() + tl.n_frames for _t, tl in targets)
    done = 0

    def tick(note: str):
        """A progress sink for the window pre-pass, which reports its own phase."""
        def _t() -> None:
            nonlocal done
            done += 1
            ctx.progress(done, total_work, note, frames=n_frames)
        return _t

    for _target, tl in targets:
        tl.measure(tick("measuring display range"))

    # ── phase 2: render + encode ──────────────────────────────────────────────
    written: List[str] = []
    for target, tl in targets:
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        note = f"rendering {os.path.basename(target)}"
        render = tl.render

        if is_sequence:
            for i in range(n_frames):
                _write_image(_seq_path(target, i, n_frames, ext), render(i), fmt, quality)
                done += 1
                ctx.progress(done, total_work, note, frames=n_frames)
            written.append(_seq_path(target, 0, n_frames, ext))
        else:
            part = _part_path(target)
            try:
                if fmt == "mp4":
                    done = _write_mp4(ctx, part, render, n_frames, fps, quality,
                                      done, total_work, note)
                else:
                    done = _write_gif(ctx, part, render, n_frames, fps,
                                      done, total_work, note)
                os.replace(part, target)
                written.append(target)
            finally:
                if os.path.exists(part):
                    os.remove(part)      # an interrupted write never keeps the real name

    ctx.progress(total_work, total_work,
                 f"exported {len(written)} {fmt.upper()}"
                 f"{' sequence' if is_sequence else ''}"
                 f"{'s' if len(written) != 1 else ''}, {n_frames} frames each: "
                 f"{os.path.basename(written[0]) if written else path}"
                 + (f" ({', '.join(unused)} wired but only read when Sweep = 'timeline')"
                    if unused else ""))
    return ds


def _timeline_movie(ctx: EvalContext, ds: Dataset, ch) -> Timeline:
    """The Movie Editor's timeline for this pull, bound to sources A, B and C.

    A's calibration is read through ``ctx.calib``/``ctx.meta``, so it is memo-fenced like
    every other read on this node. B's and C's come off their own ``metadata``: this is
    INV-05's documented exception for a SECOND input (``util/merge.py`` reads a secondary's
    ``bit_depth`` the same way). It is safe because the engine's fence is for the primary
    envelope, and each secondary's recipe hash is already in this node's memo key, so any
    change to B's calibration re-keys the movie anyway. ``dict(...)`` first, so the strict
    debug wrapper's per-key guard is not what answers.
    """
    sources = {"A": MovieSource("A", ds, calib=ctx.calib, meta=ctx.meta)}
    for letter, socket in (("B", "source_b"), ("C", "source_c")):
        sec = ctx.input(socket)
        if isinstance(sec, Dataset):
            md = dict(sec.metadata)
            sources[letter] = MovieSource(letter, sec, calib=md.get, meta=md.get)
    style = MovieStyle(
        max_px=int(ch.param("max_px") or 0), corner=str(ch.param("corner") or "top_left"),
        font_px=int(ch.param("font_px") or 0),
        text_rgb=_TEXT_RGB.get(str(ch.param("text_color")), (255, 255, 255)),
        interval_s=float(ch.param("interval_s") or 0.0))
    return Timeline(str(ch.param("timeline") or ""), sources, style)


def _write_mp4(ctx: EvalContext, path: str, render, n_frames: int, fps: float,
               quality: int, done: int, total: int, note: str) -> int:
    """Stream ``n_frames`` rendered frames into an H.264 MP4. Returns the progress count."""
    import cv2
    first = _even(render(0))
    h, w = first.shape[:2]
    writer = _open_writer("mp4", path, fps, (w, h), quality)
    try:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(max(1, min(100, quality))))
    except Exception:      # noqa: BLE001 — not every backend exposes it; the default is fine
        pass
    try:
        # cv2 wants BGR; everything upstream of here is RGB, which is what PIL and the
        # sequence writers want. One flip, at the boundary, rather than a BGR array wandering
        # through the render path waiting to be drawn on with the wrong channel order.
        writer.write(first[:, :, ::-1])
        done += 1
        ctx.progress(done, total, note, frames=n_frames)
        for i in range(1, n_frames):
            frame = _even(render(i))
            if frame.shape[:2] != (h, w):
                raise ValueError(
                    f"Export Movie: frame {i} is {frame.shape[1]}x{frame.shape[0]} but frame "
                    f"0 was {w}x{h}. A video needs one frame size; something upstream is "
                    f"changing the field as the series plays.")
            writer.write(frame[:, :, ::-1])
            done += 1
            ctx.progress(done, total, note, frames=n_frames)
    finally:
        writer.release()
    return done


def _write_gif(ctx: EvalContext, path: str, render, n_frames: int, fps: float,
               done: int, total: int, note: str) -> int:
    """Write an animated GIF. Returns the progress count.

    GIF stores a per-frame delay in HUNDREDTHS of a second, so the frame rate is quantized:
    12 fps asks for 83.3 ms and gets 80 ms, i.e. 12.5 fps. Nothing can be done about that in
    the format, but it is worth knowing before wondering why a GIF and an MP4 of the same
    series do not finish together.
    """
    from PIL import Image
    frames = []
    for i in range(n_frames):
        frames.append(Image.fromarray(render(i)))
        done += 1
        ctx.progress(done, total, note, frames=n_frames)
    delay = max(20, int(round(1000.0 / max(0.1, float(fps)))))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=delay,
                   loop=0, optimize=True, format="GIF")
    return done


def _write_image(path: str, rgb: np.ndarray, fmt: str, quality: int) -> None:
    """Write one frame of a numbered sequence, through a ``.part`` rename like the others."""
    from PIL import Image
    part = _part_path(path)
    try:
        img = Image.fromarray(rgb)
        if fmt == "jpeg":
            img.save(part, format="JPEG", quality=max(1, min(100, int(quality))),
                     subsampling=0)
        else:
            img.save(part, format="PNG", optimize=True)
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


#: The two flat sweeps. Every per-clip control (channels, contrast, gamma, burn-ins,
#: the Voxel layer, Split) is gated to them: under Sweep = 'timeline' each panel of
#: the timeline carries its own, and a node-level value there would be a live-looking
#: control the renderer ignores.
_FLAT = frozenset({"time", "z"})


register_node(
    _compute_write_movie, op_key="io.write_movie", label="Export Movie", category="io",
    inputs=[
        InDataset("data"),
        # Declared AFTER `data` (INV-09), so `data` stays the envelope source whichever wire
        # was drawn first. Not gated on Sweep: a wire into a hidden socket is drawn nowhere,
        # is still pulled, and cannot be selected to delete, so an input that exists only in
        # one mode would strand its wire the moment the mode changed.
        InDataset("source_b", label="Source B", passes_domains=False,
                  description=
                  "A second Dataset for the Movie Editor's timeline — `B` in its source "
                  "list. Wire the node whose picture you want to cut in: a segmentation to "
                  "show labelled z sweeps between the frames of a max projection, another "
                  "condition to play after this one, a different channel's processing. Only "
                  "READ when Sweep = 'timeline'; nothing from it passes through (the output "
                  "is still `data`, unchanged), so its layers are not offered downstream."),
        InDataset("source_c", label="Source C", passes_domains=False,
                  description=
                  "A third Dataset for the timeline — `C` in the Movie Editor's source "
                  "list. Same rules as Source B: only read when Sweep = 'timeline', and "
                  "nothing from it reaches this node's output."),
        InString("timeline", "Timeline", field=False, default="",
                 available_in={"sweep": frozenset({"timeline"})},
                 description=
                 "The movie as the Movie Editor laid it out: which clips play in what "
                 "order, which source and channels each panel shows, grids of positions, "
                 "channels, z slices or timepoints, loops that cut a z sweep in between "
                 "every timepoint, label overlays with ids, and each channel's colour and "
                 "black/white. Edit it in the Movie Editor (it opens when this node is "
                 "selected); underneath it is canonical JSON, and a malformed or "
                 "out-of-range value is refused with the path of the bad entry, e.g. "
                 "`segments[0].panels[1].layer`. Frame rate, size, text and the file are "
                 "still this node's own controls."),
        InString("path", "File", field=False, default="",
                 path_kind="save_file",
                 path_filter=("MP4 video (*.mp4);;Animated GIF (*.gif);;"
                              "PNG sequence (*.png);;JPEG sequence (*.jpg);;All files (*)"),
                 path_hint="Browse… to choose where this writes",
                 description=
                 "Where the movie is written, on the machine that runs the graph. There is "
                 "no default and an empty value is refused rather than guessed. **The "
                 "extension is set by Format, not by what you type here** — pick GIF with "
                 "`movie.mp4` still in this box and the node writes `movie.gif`, because a "
                 "GIF named `.mp4` is a file every player refuses while nothing reports an "
                 "error. A name whose dots are not an extension (`WellA3_1.7183um`) keeps "
                 "all of them and simply gains the right suffix. With "
                 "Format = 'png'/'jpeg' this is the TEMPLATE for a numbered sequence — "
                 "`<name>_0007.png`, zero-padded so the frames sort in playback order — and "
                 "with Split = 'position' it also gains `_m03`."),
        InString("layer", "Layer", field=False, default="", layer_in=Domain.VOXEL,
                 available_in={"sweep": _FLAT},
                 description=
                 "Render a Voxel raster — a mask, a label image, a distance field — instead "
                 "of the image. Leave it EMPTY (the default) to render the image, which is "
                 "what you want most of the time. Pointed at a segmentation this gives you "
                 "the mask movie that goes beside the data movie in a methods figure. The "
                 "raster goes through the same display window as an image would, so a label "
                 "image renders as a grey ramp of ids, not as distinct colours."),
        InString("channels", "Channels", field=False, default="",
                 available_in={"sweep": _FLAT},
                 description=
                 "Which channels to composite, as indices — `0`, `0,2`, `1-3`. EMPTY (the "
                 "default) means all of them. Each channel is tinted by its own emission "
                 "wavelength and the tints add, so a 2-colour merge reads yellow where both "
                 "are bright. Narrowing to ONE channel switches the render to greyscale "
                 "rather than tinting it, since a lone channel in its own deep-blue emission "
                 "colour is harder to see and is not what the Viewer shows either."),
        InFloat("fps", "Frame rate", unit="", field=False, default=10.0,
                description=
                "Playback speed, in frames per second — how fast the movie plays, which has "
                "nothing to do with how fast the experiment ran. 10 is a good default for "
                "cell motion; raise it to compress a long timelapse, lower it to let a "
                "reviewer see each frame. GIF quantizes this to hundredths of a second, so "
                "12 fps becomes 12.5; MP4 honours it exactly."),
        InFloat("interval_s", "Frame interval", unit="s", field=False, default=0.0,
                derive="dt_s or 0.0",
                description=
                "Seconds of EXPERIMENT time between frames, used for the burned-in clock. "
                "Auto-filled from the file's own per-frame timestamps (the median interval), "
                "so on an ND2 you should not have to touch it. It is the median of a "
                "nominally-regular acquisition, not a per-frame clock, so a run that stalled "
                "shows nominal time — override it here if you know better. 0 means unknown, "
                "and the time readout is then omitted rather than showing a made-up number. "
                "Changes only the overlay; nothing about the pixels or the playback speed."),
        InInt("max_px", "Max px", unit="px", field=False, default=1200,
              description=
              "Downscale so the frame's longest edge is at most this many pixels; 0 leaves "
              "it at full size. The default 1200 is a deliberate choice for a movie that is "
              "going into a talk or a message — a 6554² frame makes a file nobody can send "
              "and no projector resolves. Area-averaged on the way down, and it NEVER "
              "upscales. Raise it for a figure panel that will be viewed at full size. GIF "
              f"refuses a value above {_GIF_FULL_SIZE_LIMIT} (or 0), because every GIF frame "
              "must be in RAM at once."),
        InInt("quality", "Quality", unit="", field=False, default=80,
              available_in={"format": frozenset({"mp4", "jpeg"})},
              description=
              "Encoder quality, 1 (smallest file) to 100 (best picture). 80 is close to "
              "visually lossless on microscopy and a small fraction of the size of 100. "
              "Raise it if fine texture — speckle, thin processes, a bead field — is "
              "smearing into blocks. Only read for the lossy formats; PNG is lossless and "
              "GIF's palette is not a quality dial, so this field is hidden there."),
        InFloat("low_pct", "Black point", unit="", field=False, default=0.5,
                available_in={"contrast": frozenset({"auto"}), "sweep": _FLAT},
                description=
                "The percentile of sampled intensity that becomes BLACK. Raise it to crush "
                "more background and get a cleaner-looking movie; lower it toward 0 to keep "
                "dim structure visible. 0.5 clips the darkest half-percent, which on a "
                "fluorescence image is camera offset and read noise. Affects only how the "
                "movie LOOKS — no measurement downstream sees this."),
        InFloat("high_pct", "White point", unit="", field=False, default=99.7,
                available_in={"contrast": frozenset({"auto"}), "sweep": _FLAT},
                description=
                "The percentile of sampled intensity that becomes WHITE. LOWER it to "
                "brighten the movie and let the brightest few percent blow out; raise it "
                "toward 100 to keep every peak distinguishable at the cost of a dimmer "
                "picture. 99.7 is chosen so one hot pixel or a dust speck cannot set the "
                "white point — a true max routinely maps real signal into the bottom of "
                "the ramp."),
        InFloat("black", "Black level", unit="", field=False, default=0.0,
                available_in={"contrast": frozenset({"absolute"}), "sweep": _FLAT},
                description=
                "The intensity that becomes black, in the image's OWN units (raw camera "
                "counts on unprocessed data, 0..1 after a Normalize). The point of typing it "
                "is comparability: two movies of two conditions exported with the same "
                "black/white can be put side by side and the brightness difference is real. "
                "Auto contrast cannot promise that, because it measures each movie "
                "separately."),
        InFloat("white", "White level", unit="", field=False, default=4095.0,
                derive="(2 ** bit_depth - 1) if bit_depth else 4095.0",
                available_in={"contrast": frozenset({"absolute"}), "sweep": _FLAT},
                description=
                "The intensity that becomes white, in the image's own units. Defaults to the "
                "camera's full significant range read from the file — 4095 on the usual "
                "12-bit sensor, not 65535, which is the mistake that makes an export look "
                "sixteen times too dark. After a Normalize the depth is gone from the "
                "envelope and this falls back to 4095, so set it to 1.0 for 0..1 data."),
        InFloat("gamma", "Gamma", unit="", field=False, default=1.0,
                available_in={"sweep": _FLAT},
                description=
                "Midtone curve applied after the window. 1.0 is linear — what the camera "
                "recorded. Above 1 lifts dim structure into view (2.0 is a strong lift, and "
                "the usual way to show faint processes next to bright cell bodies) and "
                "below 1 deepens it. It is a DISPLAY curve only, but be aware it makes "
                "brightness in the movie non-proportional to intensity, so do not let a "
                "reader eyeball a fold-change off a gamma-lifted export."),
        InFloat("brightness", "Brightness", unit="", field=False, default=100.0,
                available_in={"sweep": _FLAT},
                description=
                "A straight LINEAR gain on the finished picture, in percent. 100 is the "
                "image as the contrast window produced it; 200 is twice as bright, 300 is "
                "three times. Unlike Gamma — which bends the midtones and changes the "
                "RATIO between a dim and a bright object — this multiplies everything by "
                "the same number, so relative brightness is preserved and the picture "
                "stays honest about which structure is brighter. Use it when the whole "
                "movie is simply too dark to read on a projector. 300 is the practical "
                "ceiling: by then the brightest structure has clipped to flat white and "
                "raising it further only spreads that flattening into the midtones. "
                "Display only, like every other control here — no measurement downstream "
                "sees it."),
        InBool("show_frame", "Frame counter", field=False, default=True,
               available_in={"sweep": _FLAT},
               description=
               "Burn the frame number and the total into the corner — `12/240`. The thing "
               "that lets someone in a meeting say 'go back to frame 87' and be understood, "
               "and that ties a movie back to the frame indices every table downstream uses."),
        InBool("show_time", "Elapsed time", field=False, default=True,
               available_in={"sweep": _FLAT},
               description=
               "Burn the elapsed EXPERIMENT time into the corner, from `Frame interval`. "
               "The unit is picked once from the run's total duration and then holds, so a "
               "6-hour run reads `01:15:00` and a 90-second one reads `12.5 s` rather than "
               "`00:00:12`. Silently omitted when the interval is 0/unknown, and inert when "
               "Sweep = 'z' — a Z position is not a time."),
        InBool("show_scalebar", "Scale bar", field=False, default=True,
               available_in={"sweep": _FLAT},
               description=
               "Burn a scale bar with its length in microns. Snapped to a round 1-2-5 value "
               "near a fifth of the frame width, so it reads `50 um` and never `73.4 um`. "
               "Omitted when the Dataset carries no `pixel_size_um` — a bar drawn without a "
               "pixel size would be a fabrication, which is worse than no bar."),
        InBool("show_channels", "Channel names", field=False, default=True,
               available_in={"sweep": _FLAT},
               description=
               "Burn each channel's name in its own tint, so a merge says which colour is "
               "which. Automatically omitted for a single-channel movie, where there is "
               "nothing to disambiguate and the label would just cover data."),
        InBool("show_position", "Position name", field=False, default=False,
               available_in={"sweep": _FLAT},
               description=
               "Burn the multipoint's name (`WellA3`, or `position 12` when the file names "
               "none). Off by default because it is redundant for a single-position export, "
               "and close to essential when Split = 'position' writes 49 files that "
               "otherwise differ only in a number buried in the filename."),
        InString("corner", "Corner", field=False, default="top_left",
                 choices=["top_left", "top_right", "bottom_left", "bottom_right"],
                 description=
                 "Which corner the frame counter and clock go in. Only this block is placed "
                 "by hand; the scale bar, the channel names and the position name fill the "
                 "remaining three corners in a fixed order, which is what guarantees no two "
                 "burn-ins can ever land on top of each other. Move it to whichever corner "
                 "of YOUR data is empty.",
                 choice_docs={
                     "top_left":
                         "The default, and the safe one: Western reading order puts the eye "
                         "here first, and it is the corner least likely to hold the "
                         "specimen in a centred acquisition.",
                     "top_right":
                         "For data whose subject sits top-left — a corner-anchored mosaic, "
                         "or a field the stage drifted toward over the run.",
                     "bottom_left":
                         "Keeps the top edge completely clear, which is what you want when "
                         "the movie will be cropped into a figure panel with a caption bar "
                         "across the top.",
                     "bottom_right":
                         "The counter sits where video players usually show a timestamp, "
                         "which reads naturally. Note it displaces the scale bar to the "
                         "other bottom corner.",
                 }),
        InInt("font_px", "Text size", unit="px", field=False, default=0,
              description=
              "Height of the burned-in text, in OUTPUT pixels; 0 auto-sizes it to about 3.5% "
              "of the frame's short edge, which stays legible from a 1200 px movie down to a "
              "thumbnail. Text is drawn AFTER the downscale, so this is the size it really "
              "appears at — it does not shrink when you lower `Max px`. Set it explicitly to "
              "match the text size across a set of movies exported at different sizes."),
        InString("text_color", "Text colour", field=False, default="white",
                 choices=["white", "black", "yellow", "cyan"],
                 description=
                 "Colour of the burned-in text and the scale bar. Every option is drawn with "
                 "a contrasting outline behind it, so all four stay readable over black "
                 "background and over saturated signal — pick on taste, not on legibility. "
                 "The channel names ignore this and use their own tints, which is the whole "
                 "point of that legend.",
                 choice_docs={
                     "white":
                         "The default. Reads on the dark background that most fluorescence "
                         "has, and matches the convention for a scale bar in a figure.",
                     "black":
                         "For brightfield, phase and DIC, where the field is bright grey and "
                         "white text competes with it.",
                     "yellow":
                         "Reads as an annotation rather than as data, which is what you want "
                         "when the movie itself is white-on-black and a white caption could "
                         "be mistaken for signal.",
                     "cyan":
                         "The same idea as yellow, for a movie whose warm channel is already "
                         "using the red/yellow end and a yellow caption would blend in.",
                 }),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("format", ["mp4", "gif", "png", "jpeg"], default="mp4", label="Format",
             description=
             "What lands on disk. All four render IDENTICAL pixels through the same display "
             "window and the same burn-ins; they differ in what will open the result and in "
             "what it costs to store.",
             choice_docs={
                 "mp4":
                     "H.264 video — the one that plays in a browser, in PowerPoint, in "
                     "Keynote and in Slack without anyone installing anything, and by far "
                     "the smallest of the four. Streams frame by frame, so there is no "
                     "length or size ceiling. The default, and the right answer unless you "
                     "need something else. Encoded by whichever backend on this machine "
                     "genuinely has an H.264 encoder — Media Foundation on Windows, since "
                     "this OpenCV build's FFmpeg has none.",
                 "gif":
                     "An animated GIF — loops forever, autoplays inline in GitHub, Slack and "
                     "a wiki, and needs no player at all. In exchange it is limited to 256 "
                     "colours (so a smooth fluorescence ramp bands visibly), it is several "
                     "times larger than the same MP4, its frame delay is quantized to "
                     "hundredths of a second, and every frame must be held in RAM at once, "
                     "which is why `Max px` is capped here.",
                 "png":
                     "A numbered PNG per frame — `<name>_0007.png` — losslessly compressed. "
                     "Not a movie: this is the option for pulling individual frames into a "
                     "figure, for handing to Illustrator or Premiere, or for re-encoding "
                     "with your own ffmpeg settings. On a long series it writes a LOT of "
                     "files.",
                 "jpeg":
                     "A numbered JPEG per frame, lossy and much smaller than the PNG "
                     "sequence. For a contact sheet, a quick visual check of a long run, or "
                     "a browse directory. Do not pick it for a figure panel — JPEG blocking "
                     "on fine texture survives into print.",
             }),
        Mode("sweep", ["time", "z", "timeline"], default="time", label="Sweep",
             description=
             "What the movie plays. 'time' and 'z' play one axis of this one Dataset, with "
             "the controls on this card; 'timeline' plays whatever the Movie Editor laid "
             "out — several clips, several sources, grids and interleaved z sweeps — and "
             "hides the per-clip controls here, because each clip there carries its own.",
             choice_docs={
                 "timeline":
                     "Play the Movie Editor's timeline: clips in sequence, each showing one "
                     "or more panels from `data` (A), `source_b` (B) or `source_c` (C); grids "
                     "that tile positions, channels, z slices or timepoints; loops that cut "
                     "a z sweep in between every timepoint; label rasters drawn with their "
                     "ids; per-channel colours and black/white taken from the Viewer. "
                     "Channels, contrast, gamma, the burn-in toggles, Layer and Split move "
                     "into the editor. Frame rate, Max px, text and the file stay here.",
                 "time":
                     "Play the T axis — the timelapse. The default and the common case, and "
                     "the only one where the burned-in clock means anything. A Z-stack at "
                     "each timepoint is flattened first by `Z projection`.",
                 "z":
                     "Play the Z axis — a flythrough of one stack, which is how you show 3D "
                     "structure in a 2D medium. Needs a single timepoint, and refuses rather "
                     "than guessing when the Dataset has several: crop to one with "
                     "`util.crop` first, or use 'timeline' to play a z sweep at every "
                     "timepoint. The elapsed-time overlay switches off here, since a Z "
                     "position is not a time.",
             }),
        Mode("z_reduce", ["max", "mid", "mean"], default="max", label="Z projection",
             available_in={"sweep": frozenset({"time"})},
             description=
             "How a Z-stack collapses to the one frame a movie can show at each timepoint. "
             "Only read when Sweep = 'time' (when Z is what plays there is nothing to "
             "collapse), and inert on data with a single Z slice.",
             choice_docs={
                 "max":
                     "Maximum along Z — the brightest value at each (y, x). The default and "
                     "the fluorescence convention: it shows every labelled object wherever "
                     "it sat in the stack, which is what makes a moving cell stay visible as "
                     "it changes focus. Exaggerates bright noise, since a single hot voxel "
                     "anywhere in the column wins.",
                 "mid":
                     "The middle slice, untouched. The honest option when the stack is a "
                     "focus sweep rather than a volume, and the one that preserves what the "
                     "raw image actually looked like at the focal plane — no projection "
                     "artefacts, but anything out of that plane is simply gone.",
                 "mean":
                     "The average along Z. Suppresses noise (by roughly the square root of "
                     "the slice count) and gives a smoother, dimmer picture than 'max'. Good "
                     "for a densely labelled or transmitted-light volume where a maximum "
                     "saturates into a white sheet.",
             }),
        Mode("contrast", ["auto", "full", "absolute"], default="auto", label="Contrast",
             available_in={"sweep": _FLAT},
             description=
             "How the image's intensities map onto the 0-255 the movie can show. Whichever "
             "you pick, the window is computed ONCE and held for every frame — a per-frame "
             "auto-contrast normalizes away the very change a timelapse exists to show and "
             "makes the background pulse.",
             choice_docs={
                 "auto":
                     "Measure the window from the data: percentiles of a subsample spread "
                     "evenly across the whole series, per channel. The default, and the one "
                     "that produces a usable movie without being told anything about the "
                     "detector. Because it measures each export separately, brightness is "
                     "NOT comparable between two movies — use 'absolute' when that matters.",
                 "full":
                     "Map the sensor's entire range, read from the file's significant bit "
                     "depth (4095 on the usual 12-bit camera, not 65535). Nothing clips and "
                     "nothing is exaggerated, so it is the faithful option — and it is "
                     "usually very dark, because fluorescence rarely fills a camera's range.",
                 "absolute":
                     "Type the black and white points in the image's own units. The option "
                     "for a figure: two conditions exported with the same window can be put "
                     "side by side and the brightness difference a reader sees is a real "
                     "difference in the data, which neither other option can promise.",
             }),
        Mode("split", ["none", "position"], default="none", label="Split",
             available_in={"sweep": _FLAT},
             description=
             "Whether a multipoint Dataset writes one movie or one per position. There is no "
             "'group' option here, unlike Export TIFF: a group is a mosaic, and the thing to "
             "do with a mosaic is stitch it into one field first, which `util.stitch` already "
             "does upstream of this node.",
             choice_docs={
                 "none":
                     "One movie, which requires exactly one position. Refused with a pointer "
                     "to `util.crop` when the Dataset carries several — a movie cannot hold "
                     "them, and quietly exporting position 0 would be a stale result wearing "
                     "a confident filename.",
                 "position":
                     "One movie per multipoint, named `<name>_m03.<ext>` with the index "
                     "zero-padded so the files sort in position order. On a 49-position "
                     "plate this writes 49 movies — turn on `Position name` so they are "
                     "tellable apart without opening them. Each position gets its OWN auto "
                     "window, so a dim well is not crushed by a bright one; switch Contrast "
                     "to 'absolute' when you need them comparable.",
             }),
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description=
             "What to do when a file is already at the target path. Export TIFF's third "
             "option, 'skip', is deliberately absent here: it is only safe there because a "
             "TIFF has a private tag to hold an export stamp, and neither MP4 nor GIF has a "
             "comparable place to put one, so a 'skip' could only mean 'a file exists with "
             "this name' — which is how a stale movie ships under the name of a fresh one.",
             choice_docs={
                 "overwrite":
                     "Always re-render and replace. The default: rendering a movie is cheap "
                     "next to the analysis upstream of it, and an unchanged graph re-pulled "
                     "in one session never reaches this node at all — the memo answers "
                     "first, which costs nothing.",
                 "refuse":
                     "Raise rather than touch an existing file. For a destination that must "
                     "be written exactly once — a published figure, a shared drive — where "
                     "an accidental overwrite is worse than a failed run. You then move or "
                     "rename the old file by hand.",
             }),
    ],
    # Keyed by `sweep`, not by a dim lever: sweeping time, one output frame is a whole
    # (Z, Y, X) column reduced to a plane, so the kernel reaches across z; sweeping z, one
    # output frame IS one plane. NOT WHOLE_SERIES — the contrast pre-pass accumulates
    # percentile samples one plane at a time and never holds two frames, so peak memory is
    # one frame exactly as declared. A timeline frame can gather several positions (an M
    # grid), several timepoints (a contact sheet) and a Z column (a max-Z panel), so it
    # declares MULTI_VIEW with every axis it may reach; peak memory is still one output
    # frame's worth of planes.
    footprint_mode="sweep",
    granularity={"time": Granularity.WHOLE_VOLUME, "z": Granularity.WHOLE_PLANE,
                 "timeline": Granularity.MULTI_VIEW},
    kernel_axes={"time": frozenset({"z", "y", "x"}), "z": frozenset({"y", "x"}),
                 "timeline": frozenset({"m", "t", "z", "y", "x"})},
    reads_domains=frozenset({Domain.VOXEL}),
    description="Render the Dataset on this wire to an H.264 MP4, an animated GIF or a "
                "numbered PNG/JPEG sequence — the presentation counterpart to Export TIFF. "
                "Channels composite by their emission tints, the display window is measured "
                "once and held so a timelapse does not pulse, and the acquisition's own "
                "calibration is burned into the corners: the frame counter, the elapsed "
                "experiment time interpolated from the file's frame interval, a round-number "
                "scale bar from the pixel size, and the channel names. With Sweep = "
                "'timeline' it plays the Movie Editor's layout instead: clips from up to "
                "three sources in sequence, grids of positions/channels/z/timepoints, z "
                "sweeps cut in between timepoints, and label rasters drawn with their ids. "
                "Hands the Dataset through unchanged, so it can sit mid-chain.")
