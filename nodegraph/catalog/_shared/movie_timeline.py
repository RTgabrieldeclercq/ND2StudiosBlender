"""Export Movie's timeline: several sources, several clips and grids, drawn onto one canvas.

A **timeline spec** is a small JSON document the Movie Editor writes into the node's
``timeline`` param. It says what each output frame shows:

* ``segments`` play in order. A segment is a ``clip`` or a ``loop``.
* A **clip** plays one axis (``t``, ``z``, or ``none`` for a held still) of one source, and
  shows one or more **panels** laid out in a grid. A panel is one picture: a source
  (``A`` = the node's ``data``, ``B``/``C`` = its extra inputs), the channels it composites
  and how they look, which ``m``/``t``/``z`` it reads, and optionally a label raster drawn over
  it with the ids burned in.
* A panel with a ``tile`` expands into one picture per index along an axis: per position
  (``m``), per channel (``c``), per z slice (``z``) or per timepoint (``t``). That is the
  n x n grid.
* A **loop** walks ``t`` over a source and plays its body clips once per step, with every
  panel whose ``t`` is ``"auto"`` bound to the loop's current ``t``. "Source A's max-Z at t,
  then source B's z sweep at t, for every t" is one loop with two clips in its body.

The flat Export Movie (no timeline) is the one-clip, one-panel case: :func:`flat_spec`
builds it from the node's ordinary params, and it renders through this same code, so the
movie a user converts into a timeline starts out pixel-identical to the one they had.

Everything is Qt-free and file-free, and nothing here touches an ``EvalContext``: sources
arrive with their calibration already read (:class:`MovieSource`), because a ``ctx.calib``
read after the compute returns raises, and the preview renders long after that.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain

from nodegraph.catalog._shared.label_paint import (
    LABEL_COLOR_MODES, LABEL_STYLES, MAX_ID_LABELS, label_id_marks, label_palette,
    paint_labels, resample_ids)
from nodegraph.catalog._shared.movie_draw import (
    _annotate, _composite, _draw_text, _emission_rgb, _font, _micro, _reduce_z, _text_px,
    _time_text)

#: The spec format this module reads and writes. Bumped only for a change an older reader
#: would misread; a new key with a default is not such a change.
SCHEMA_VERSION = 1

#: Source letters and the node sockets they name. ``A`` is always the node's own ``data``.
SOURCES = ("A", "B", "C")
SOURCE_SOCKETS = {"A": "data", "B": "source_b", "C": "source_c"}

PLAY_AXES = ("t", "z", "none")
#: ``alternate`` plays up on a loop's even steps and down on its odd ones, so an
#: interleaved z sweep never jumps from the top of the stack back to the bottom.
DIRECTIONS = ("up", "down", "pingpong", "alternate")
RENDERS = ("image", "labels", "labels_over_image")
Z_REDUCERS = ("max", "mean", "mid")
TILE_AXES = ("none", "m", "c", "z", "t")
#: ``fit`` scales every clip to fill the canvas; ``center`` never scales a clip past the
#: movie's own downscale, so a small clip sits at its true size in the middle.
FITS = ("fit", "center")
CONTRAST_MODES = ("auto", "full")

#: Frames sampled per display window in ``contrast`` ``auto``. The window is ONE decision
#: for the whole movie, so it is measured up front from a subsample spread evenly over the
#: frames that window will actually colour. 24 is enough for a stable percentile and few
#: enough that the pre-pass is not a second full read of the series.
_WINDOW_SAMPLES = 24


class TimelineError(ValueError):
    """A timeline that cannot be rendered. The message starts with the spec PATH of the
    offending value (``segments[0].body[1].panels[0].layer``), so the editor can point at
    it and a headless user can find it in the JSON."""


# ── spec normalization ─────────────────────────────────────────────────────────────
#
# One pass that validates every value, fills every default and refuses every unknown key.
# Refusing unknown keys is deliberate: a typo (`"chanels"`) would otherwise be silently
# ignored and the movie would render with the default, which looks like the editor not
# working. Floats are rounded to nine significant digits, so a value re-stamped from the
# Viewer with float64 noise does not churn the canonical string (and so the memo key), while
# every value the float32 compositor can tell apart survives.

def _tl_need(ok: bool, path: str, msg: str) -> None:
    if not ok:
        raise TimelineError(f"{path}: {msg}")


def _tl_int(lo: Optional[int] = None, hi: Optional[int] = None):
    def check(v: Any, path: str) -> int:
        _tl_need(isinstance(v, (int, float)) and not isinstance(v, bool)
                 and float(v).is_integer(), path, f"expected a whole number, got {v!r}")
        v = int(v)
        _tl_need(lo is None or v >= lo, path, f"must be >= {lo}, got {v}")
        _tl_need(hi is None or v <= hi, path, f"must be <= {hi}, got {v}")
        return v
    return check


def _tl_num(lo: Optional[float] = None, hi: Optional[float] = None):
    def check(v: Any, path: str) -> float:
        _tl_need(isinstance(v, (int, float)) and not isinstance(v, bool)
                 and math.isfinite(float(v)), path, f"expected a number, got {v!r}")
        v = float(f"{float(v):.9g}")
        if lo is not None and v < lo:
            raise TimelineError(f"{path}: must be >= {lo:g}, got {v:g}")
        if hi is not None and v > hi:
            raise TimelineError(f"{path}: must be <= {hi:g}, got {v:g}")
        return v
    return check


def _tl_bool(v: Any, path: str) -> bool:
    _tl_need(isinstance(v, bool), path, f"expected true/false, got {v!r}")
    return bool(v)


def _tl_text(v: Any, path: str) -> str:
    _tl_need(isinstance(v, str), path, f"expected text, got {v!r}")
    _tl_need(len(v) <= 500, path, "text longer than 500 characters")
    return v


def _tl_choice(options: Sequence[str]):
    def check(v: Any, path: str) -> str:
        _tl_need(v in options, path, f"must be one of {list(options)}, got {v!r}")
        return str(v)
    return check


def _tl_optional(inner):
    def check(v: Any, path: str) -> Any:
        return None if v is None else inner(v, path)
    return check


def _tl_rgb(v: Any, path: str) -> List[int]:
    _tl_need(isinstance(v, (list, tuple)) and len(v) == 3, path,
             f"expected [r, g, b], got {v!r}")
    return [_tl_int(0, 255)(c, f"{path}[{i}]") for i, c in enumerate(v)]


def _tl_ints(lo: int = 0):
    def check(v: Any, path: str) -> List[int]:
        _tl_need(isinstance(v, (list, tuple)), path, f"expected a list, got {v!r}")
        return [_tl_int(lo)(x, f"{path}[{i}]") for i, x in enumerate(v)]
    return check


def _tl_tval(v: Any, path: str) -> Any:
    if v == "auto":
        return "auto"
    return _tl_int(0)(v, path)


def _tl_zval(v: Any, path: str) -> Any:
    if v in ("auto",) + Z_REDUCERS:
        return str(v)
    return _tl_int(0)(v, path)


def _tl_obj(fields: Dict[str, Tuple[Callable, Any]]):
    def check(v: Any, path: str) -> Dict[str, Any]:
        v = {} if v is None else v
        _tl_need(isinstance(v, dict), path, f"expected an object, got {type(v).__name__}")
        unknown = sorted(set(v) - set(fields))
        _tl_need(not unknown, path, f"unknown key(s) {unknown}; allowed: {sorted(fields)}")
        return {k: chk(v.get(k, default), f"{path}.{k}")
                for k, (chk, default) in fields.items()}
    return check


def _tl_list(item, *, min_len: int = 0):
    def check(v: Any, path: str) -> List[Any]:
        v = [] if v is None else v
        _tl_need(isinstance(v, (list, tuple)), path, f"expected a list, got {v!r}")
        _tl_need(len(v) >= min_len, path, f"needs at least {min_len} entr"
                 f"{'y' if min_len == 1 else 'ies'}")
        return [item(x, f"{path}[{i}]") for i, x in enumerate(v)]
    return check


_CHANNEL_DISPLAY = _tl_obj({
    "rgb": (_tl_optional(_tl_rgb), None),
    "lo": (_tl_optional(_tl_num()), None),
    "hi": (_tl_optional(_tl_num()), None),
    "gamma": (_tl_num(0.05, 20.0), 1.0),
    # A GUI hint only: `viewer` means the Movie Editor keeps these values in step with the
    # Viewer's LUT for this source. The renderer reads the concrete values and nothing else.
    "link": (_tl_choice(("manual", "viewer")), "manual"),
})


def _tl_display(v: Any, path: str) -> Dict[str, Any]:
    v = {} if v is None else v
    _tl_need(isinstance(v, dict), path, "expected an object keyed by channel index")
    out = {}
    for k in sorted(v, key=lambda s: (len(str(s)), str(s))):
        _tl_need(str(k).isdigit(), f"{path}", f"channel key {k!r} is not an index")
        out[str(int(k))] = _CHANNEL_DISPLAY(v[k], f"{path}.{k}")
    return out


_PANEL = _tl_obj({
    "source": (_tl_choice(SOURCES), "A"),
    "layer": (_tl_text, ""),
    "render": (_tl_choice(RENDERS), "image"),
    "channels": (_tl_ints(0), []),
    "t": (_tl_tval, "auto"),
    "z": (_tl_zval, "auto"),
    "m": (_tl_int(0), 0),
    "display": (_tl_display, {}),
    "contrast": (_tl_obj({
        "mode": (_tl_choice(CONTRAST_MODES), "auto"),
        "low_pct": (_tl_num(0.0, 100.0), 0.5),
        "high_pct": (_tl_num(0.0, 100.0), 99.7),
    }), None),
    "brightness": (_tl_num(1.0, 10000.0), 100.0),
    "labels": (_tl_obj({
        # -1 = the panel's first channel, else 0. A label raster carries the C axis like
        # the image does, and most segmentations fill only the channel they segmented.
        "c": (_tl_int(-1), -1),
        "color": (_tl_choice(LABEL_COLOR_MODES), "per_id"),
        "style": (_tl_choice(LABEL_STYLES), "both"),
        "fill_opacity": (_tl_num(0.0, 1.0), 0.3),
        "outline_px": (_tl_int(0, 20), 2),
        "show_ids": (_tl_bool, True),
        "id_px": (_tl_int(4, 200), 11),
    }), None),
    "tile": (_tl_obj({
        "axis": (_tl_choice(TILE_AXES), "none"),
        "indices": (_tl_ints(0), []),
    }), None),
    "caption": (_tl_text, ""),
})

_PLAY = _tl_obj({
    "axis": (_tl_choice(PLAY_AXES), "t"),
    "source": (_tl_choice(("",) + SOURCES), ""),
    "from": (_tl_int(), 0),
    "to": (_tl_int(), -1),
    "step": (_tl_int(1), 1),
    "direction": (_tl_choice(DIRECTIONS), "up"),
})

_ANNOTATIONS = _tl_obj({
    "frame": (_tl_bool, True),
    "time": (_tl_bool, True),
    "z": (_tl_bool, True),
    "scalebar": (_tl_bool, True),
    "channels": (_tl_bool, True),
    "position": (_tl_bool, False),
    "captions": (_tl_bool, True),
    "title": (_tl_text, ""),
})

_CLIP_FIELDS = {
    "kind": (_tl_choice(("clip",)), "clip"),
    "name": (_tl_text, ""),
    "play": (_PLAY, None),
    "hold": (_tl_int(1, 10000), 1),
    "cols": (_tl_int(0, 64), 0),
    "panels": (_tl_list(_PANEL, min_len=1), None),
    "annotations": (_ANNOTATIONS, None),
}
_CLIP = _tl_obj(_CLIP_FIELDS)

_LOOP = _tl_obj({
    "kind": (_tl_choice(("loop",)), "loop"),
    "name": (_tl_text, ""),
    "axis": (_tl_choice(("t",)), "t"),
    "source": (_tl_choice(SOURCES), "A"),
    "from": (_tl_int(), 0),
    "to": (_tl_int(), -1),
    "step": (_tl_int(1), 1),
    "body": (_tl_list(_CLIP, min_len=1), None),
})


def _tl_segment(v: Any, path: str) -> Dict[str, Any]:
    _tl_need(isinstance(v, dict), path, f"expected an object, got {type(v).__name__}")
    kind = v.get("kind", "clip")
    if kind == "loop":
        return _LOOP(v, path)
    _tl_need(kind == "clip", f"{path}.kind", f"must be 'clip' or 'loop', got {kind!r}")
    return _CLIP(v, path)


_SPEC = _tl_obj({
    "v": (_tl_int(1, SCHEMA_VERSION), SCHEMA_VERSION),
    "canvas": (_tl_obj({
        "fit": (_tl_choice(FITS), "fit"),
        "bg": (_tl_rgb, [0, 0, 0]),
        "gap": (_tl_int(0, 400), 4),
    }), None),
    "segments": (_tl_list(_tl_segment, min_len=1), None),
})


def normalize_spec(spec: Any) -> Dict[str, Any]:
    """``spec`` (a dict, or JSON text) validated, defaults filled, unknown keys refused.

    Raises :class:`TimelineError` naming the spec path of the first bad value."""
    if isinstance(spec, (str, bytes)):
        text = spec.decode("utf-8") if isinstance(spec, bytes) else spec
        if not text.strip():
            raise TimelineError("timeline: empty. Open the Movie Editor on this node to lay "
                                "out a timeline, or set Sweep back to 'time' or 'z'.")
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TimelineError(f"timeline: not valid JSON ({exc.msg} at line {exc.lineno}, "
                                f"column {exc.colno})") from None
    return _SPEC(spec, "timeline")


def canonical_json(spec: Any) -> str:
    """The ONE spelling of ``spec``: normalized, keys sorted, no whitespace.

    What the editor writes into the param. The memo hashes the param string, so two spellings
    of the same timeline would be two memo keys, and a Viewer re-stamp that changed nothing
    would still re-render the movie."""
    return json.dumps(normalize_spec(spec), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def try_normalize(spec: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(normalized, "")`` or ``(None, message)``. Total, never raises, for edit-time
    callers (the editor re-validates on every keystroke)."""
    try:
        return normalize_spec(spec), ""
    except TimelineError as exc:
        return None, str(exc)
    except Exception as exc:      # noqa: BLE001 — an edit-time check must never raise
        return None, f"timeline: {exc}"


def blank_clip(source: str = "A", *, axis: str = "t") -> Dict[str, Any]:
    """A default clip on ``source``: every channel, max-Z, auto contrast."""
    return _CLIP({"play": {"axis": axis}, "panels": [{"source": source}]}, "clip")


def flat_spec(*, sweep: str, z_reduce: str, channels: Sequence[int], layer: str, m: int,
              contrast: str, low_pct: float, high_pct: float, black: float, white: float,
              gamma: float, brightness: float, show: Dict[str, bool]) -> Dict[str, Any]:
    """The flat Export Movie's settings as a one-clip, one-panel timeline.

    This is what makes the flat node and the timeline one renderer rather than two, and what
    the editor's *Convert to timeline* starts from."""
    display: Dict[str, Any] = {}
    for c in channels:
        d: Dict[str, Any] = {"gamma": float(gamma)}
        if contrast == "absolute":
            d.update(lo=float(black), hi=float(white))
        display[str(int(c))] = d
    panel = {"source": "A", "layer": str(layer or ""), "render": "image",
             "channels": [int(c) for c in channels], "t": "auto",
             "z": (z_reduce if sweep == "time" else "auto"), "m": int(m),
             "display": display,
             "contrast": {"mode": "full" if contrast == "full" else "auto",
                          "low_pct": float(low_pct), "high_pct": float(high_pct)},
             "brightness": float(brightness)}
    clip = {"kind": "clip", "play": {"axis": "t" if sweep == "time" else "z", "source": "A"},
            "cols": 1, "panels": [panel],
            "annotations": {"frame": bool(show.get("frame")),
                            "time": bool(show.get("time")) and sweep == "time",
                            "z": False, "scalebar": bool(show.get("scalebar")),
                            "channels": bool(show.get("channels")),
                            "position": bool(show.get("position")), "captions": False}}
    return normalize_spec({"canvas": {"gap": 0}, "segments": [clip]})


# ── sources ──────────────────────────────────────────────────────────────────────

class MovieSource:
    """One Dataset a timeline draws from, with its calibration read ONCE, up front.

    ``calib`` / ``meta`` are plain getters. The engine passes ``ctx.calib``/``ctx.meta`` for
    ``A``, so those reads are memo-fenced, and the dataset's own ``metadata.get`` for
    ``B``/``C``, whose recipe hashes already sit in this node's memo key. The preview
    passes ``metadata.get`` for all of them.
    """

    __slots__ = ("letter", "ds", "ax", "prov", "px_um", "z_step", "dt_s", "emis",
                 "bit_depth", "names", "pos_names", "_vox", "picture", "colors")

    def __init__(self, letter: str, ds, *, calib: Callable, meta: Callable) -> None:
        self.letter, self.ds, self.ax = letter, ds, ds.axes
        self.prov = ds.image
        self.px_um = calib("pixel_size_um")
        self.z_step = calib("z_step_um")
        self.dt_s = calib("dt_s")
        self.emis = list(calib("channel_emission_nm") or [])
        # a PICTURE (a plot's figure, V4.00) is RGB already: its channels keep their own
        # colours on a fixed 0-255 window, as the Viewer shows it
        md = getattr(ds, "metadata", None) or {}
        self.picture = bool(md.get("picture"))
        self.colors = list(md.get("channel_colors") or []) if self.picture else []
        self.bit_depth = calib("bit_depth")
        # Names are DISPLAY metadata, and the GUI seeds them onto the source Dataset only,
        # never into the engine's envelope (`runner._channel_display`), so `ctx.meta` reports
        # none on a GUI export while the Viewer shows the file's real names. The Dataset's
        # own metadata is the fallback: not calibration, so no fence is bypassed, and the
        # preview (which reads that same dict) and the export then agree.
        own = dict(getattr(ds, "metadata", None) or {})
        self.names = list(meta("channel_names") or own.get("channel_names") or [])
        self.pos_names = list(meta("position_name") or own.get("position_name") or [])
        self._vox: Dict[str, np.ndarray] = {}

    def channel_name(self, c: int) -> str:
        n = self.names
        return str(n[c]) if c < len(n) and n[c] else f"C{c}"

    def position_name(self, m: int) -> str:
        n = self.pos_names
        return str(n[m]) if m < len(n) and n[m] else f"position {m}"

    def voxel(self, name: str, path: str) -> np.ndarray:
        """The full 6-D ``(m, t, z, c, y, x)`` array of Voxel layer ``name``."""
        got = self._vox.get(name)
        if got is not None:
            return got
        lay = self.ds.get(Domain.VOXEL, name)
        if lay is None:
            have = sorted({a.name for a in self.ds.layers_on(Domain.VOXEL)})
            raise TimelineError(f"{path}: no Voxel layer {name!r} on source {self.letter}; "
                                f"it carries {have or 'none'}")
        arr = np.asarray(lay.values)
        ax = self.ax
        if arr.shape != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
            raise TimelineError(f"{path}: Voxel layer {name!r} has shape {arr.shape}, which "
                                f"does not match source {self.letter}'s axes "
                                f"{(ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)}")
        self._vox[name] = arr
        return arr


class MovieStyle:
    """The movie-wide settings every clip shares: the node's own sockets."""

    __slots__ = ("max_px", "corner", "font_px", "text_rgb", "interval_s")

    def __init__(self, *, max_px: int, corner: str, font_px: int,
                 text_rgb: Tuple[int, int, int], interval_s: float) -> None:
        self.max_px, self.corner, self.font_px = int(max_px or 0), corner, int(font_px or 0)
        self.text_rgb, self.interval_s = tuple(text_rgb), float(interval_s or 0.0)


# ── the compiled timeline ────────────────────────────────────────────────────────

class _View:
    """One picture on the canvas: a panel, or one tile of a tiled panel, resolved."""

    __slots__ = ("path", "src", "layer", "render", "chans", "tints", "gammas", "names",
                 "brightness", "m", "t_fix", "z_fix", "z_red", "lab_c", "lab", "caption",
                 "contrast", "explicit", "wkey", "tile_axis")


class _Clip:
    __slots__ = ("path", "spec", "views", "cols", "rows", "play_axis", "play_src", "ann",
                 "sc", "place")


def _tl_span(n: int, frm: int, to: int, step: int, path: str) -> List[int]:
    """Indices ``from..to`` inclusive, python-style negatives, every ``step``-th."""
    a = frm + n if frm < 0 else frm
    b = to + n if to < 0 else to
    _tl_need(0 <= a < n, f"{path}.from", f"{frm} is outside 0..{n - 1}")
    _tl_need(0 <= b < n, f"{path}.to", f"{to} is outside 0..{n - 1}")
    _tl_need(a <= b, path, f"from ({a}) is after to ({b})")
    return list(range(a, b + 1, max(1, int(step))))


def _tl_directed(seq: List[int], direction: str, iteration: int) -> List[int]:
    if direction == "down" or (direction == "alternate" and iteration % 2 == 1):
        return list(reversed(seq))
    if direction == "pingpong" and len(seq) > 1:
        return seq + list(reversed(seq[:-1]))
    return list(seq)


def _fit_resize(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """``rgb`` resized to ``size`` ``(w, h)``: area-averaged down, nearest-neighbour up.

    Nearest on the way UP because a clip enlarged to fill the canvas should show its pixels
    as pixels. Smoothing an upscale invents detail the camera never recorded."""
    h, w = rgb.shape[:2]
    tw, th = int(size[0]), int(size[1])
    if (tw, th) == (w, h):
        return rgb
    import cv2
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_NEAREST
    return cv2.resize(rgb, (tw, th), interpolation=interp)


class Timeline:
    """A normalized timeline bound to its sources: ``render(k)`` gives output frame ``k``.

    Built once per export (or per preview). Construction validates everything that can be
    validated without reading pixels: every source is wired, every layer exists at the right
    shape, every index is in range for every frame. So a bad timeline fails before the first
    frame is written, not at frame 4,000.
    """

    def __init__(self, spec: Any, sources: Dict[str, MovieSource], style: MovieStyle) -> None:
        self.spec = normalize_spec(spec)
        self.sources = dict(sources)
        self.style = style
        cv = self.spec["canvas"]
        self.bg = tuple(int(c) for c in cv["bg"])
        self.gap = int(cv["gap"])
        self.fit = cv["fit"]
        self._clips: List[_Clip] = []
        #: one entry per output frame: (clip index, t_play, z_play, loop_t)
        self.frames: List[Tuple[int, Optional[int], Optional[int], Optional[int]]] = []
        #: per top-level segment: (first frame, frame count)
        self.segment_spans: List[Tuple[int, int]] = []
        for si, seg in enumerate(self.spec["segments"]):
            start = len(self.frames)
            path = f"timeline.segments[{si}]"
            if seg["kind"] == "loop":
                src = self._source(seg["source"], f"{path}.source")
                steps = _tl_span(src.ax.t, seg["from"], seg["to"], seg["step"], path)
                body = [self._compile(clip, f"{path}.body[{bi}]", in_loop=True)
                        for bi, clip in enumerate(seg["body"])]
                for it, lt in enumerate(steps):
                    for ci in body:
                        self._expand(ci, lt, it)
            else:
                self._expand(self._compile(seg, path, in_loop=False), None, 0)
            self.segment_spans.append((start, len(self.frames) - start))
        self.n_frames = len(self.frames)
        _tl_need(self.n_frames > 0, "timeline", "renders no frames")
        self._layout()
        self._windows: Dict[Tuple[Any, int], Tuple[float, float]] = {}
        self._samples: Dict[Any, set] = {}
        self._validate()
        self._measured = False
        self._last: Tuple[Any, Optional[np.ndarray]] = (None, None)
        self._luts: Dict[Tuple[str, int], np.ndarray] = {}

    # ── compile ──────────────────────────────────────────────────────────────
    def _source(self, letter: str, path: str) -> MovieSource:
        src = self.sources.get(letter)
        if src is None:
            raise TimelineError(
                f"{path}: source {letter} is not wired. Connect a node to this Export Movie's "
                f"'{SOURCE_SOCKETS[letter]}' input, or point the panel at another source.")
        return src

    def _compile(self, clip: Dict[str, Any], path: str, *, in_loop: bool) -> int:
        c = _Clip()
        c.path, c.spec, c.ann = path, clip, clip["annotations"]
        play = clip["play"]
        c.play_axis = play["axis"]
        views: List[_View] = []
        for pi, panel in enumerate(clip["panels"]):
            views.extend(self._panel_views(panel, f"{path}.panels[{pi}]", c.play_axis,
                                           in_loop))
        c.views = views
        n = len(views)
        c.cols = int(clip["cols"]) or int(math.ceil(math.sqrt(n)))
        c.cols = max(1, min(c.cols, n))
        c.rows = int(math.ceil(n / c.cols))
        c.play_src = self._source(play["source"] or clip["panels"][0]["source"],
                                  f"{path}.play.source")
        self._clips.append(c)
        return len(self._clips) - 1

    def _panel_views(self, panel: Dict[str, Any], path: str, play_axis: str,
                     in_loop: bool) -> List[_View]:
        src = self._source(panel["source"], f"{path}.source")
        ax = src.ax
        chans = list(panel["channels"]) or list(range(ax.c))
        for i, ch in enumerate(chans):
            _tl_need(0 <= ch < ax.c, f"{path}.channels[{i}]",
                     f"channel {ch} does not exist on source {src.letter} "
                     f"(C={ax.c}, indexed 0..{ax.c - 1})")
        _tl_need(0 <= panel["m"] < ax.m, f"{path}.m",
                 f"position {panel['m']} does not exist on source {src.letter} (M={ax.m})")
        render, layer = panel["render"], panel["layer"]
        _tl_need(render == "image" or bool(layer), f"{path}.layer",
                 f"render '{render}' draws a label raster; name the Voxel layer to draw")
        if layer:
            src.voxel(layer, f"{path}.layer")
        tile = panel["tile"]
        axis = tile["axis"]
        _tl_need(axis == "none" or axis != play_axis, f"{path}.tile.axis",
                 f"cannot tile on '{axis}' while the clip plays '{axis}'")
        _tl_need(not (axis == "t" and in_loop and panel["t"] == "auto"), f"{path}.tile.axis",
                 "cannot tile on 't' inside a loop, which already binds t")
        size = {"m": ax.m, "c": ax.c, "z": ax.z, "t": ax.t}.get(axis, 1)
        if axis == "none":
            variants: List[Dict[str, int]] = [{}]
        else:
            idx = list(tile["indices"]) or (chans if axis == "c" else list(range(size)))
            for i, k in enumerate(idx):
                _tl_need(0 <= k < size, f"{path}.tile.indices[{i}]",
                         f"{k} is outside 0..{size - 1} on source {src.letter}")
            variants = [{axis: k} for k in idx]
        out = []
        for var in variants:
            v = _View()
            v.path, v.src, v.layer, v.render = path, src, layer, render
            v.tile_axis = axis
            v.chans = [var["c"]] if "c" in var else list(chans)
            disp = panel["display"]
            v.tints, v.gammas, v.explicit = [], [], {}
            for ch in v.chans:
                d = disp.get(str(ch)) or {}
                pic = (src.colors[ch] if getattr(src, "picture", False)
                       and ch < len(src.colors) else None)
                if d.get("rgb") is not None:
                    v.tints.append(tuple(int(x) for x in d["rgb"]))
                elif isinstance(pic, (list, tuple)) and len(pic) == 3:
                    v.tints.append(tuple(int(x) for x in pic))
                elif len(v.chans) == 1:
                    # one channel reads greyscale, not in its emission tint: a lone deep-blue
                    # channel is hard to see, and it is not what the Viewer shows either
                    v.tints.append((255, 255, 255))
                else:
                    v.tints.append(_emission_rgb(src.emis[ch] if ch < len(src.emis) else None))
                v.gammas.append(float(d.get("gamma", 1.0)))
                if d.get("lo") is not None and d.get("hi") is not None:
                    lo, hi = float(d["lo"]), float(d["hi"])
                    v.explicit[ch] = (lo, hi if hi > lo else lo + 1.0)
                elif getattr(src, "picture", False):
                    v.explicit[ch] = (0.0, 255.0)       # a picture is shown as drawn
            v.names = [src.channel_name(ch) for ch in v.chans]
            v.brightness = float(panel["brightness"])
            v.m = var.get("m", panel["m"])
            v.t_fix = var.get("t", panel["t"] if isinstance(panel["t"], int) else None)
            zval = panel["z"]
            v.z_fix = var.get("z", zval if isinstance(zval, int) else None)
            if v.z_fix is not None:
                _tl_need(0 <= v.z_fix < ax.z, f"{path}.z",
                         f"z {v.z_fix} is outside 0..{ax.z - 1} on source {src.letter}")
            v.z_red = zval if zval in Z_REDUCERS else ("max" if ax.z > 1 else None)
            lab = panel["labels"]
            v.lab = lab
            v.lab_c = (lab["c"] if lab["c"] >= 0 else (v.chans[0] if v.chans else 0))
            _tl_need(0 <= v.lab_c < ax.c, f"{path}.labels.c",
                     f"channel {v.lab_c} does not exist on source {src.letter}")
            v.caption = panel["caption"]
            v.contrast = panel["contrast"]
            img_layer = layer if render == "image" else ""
            v.wkey = (src.letter, img_layer, "plane" if v.z_fix is not None else v.z_red,
                      v.contrast["mode"], v.contrast["low_pct"], v.contrast["high_pct"])
            out.append(v)
        return out

    def _expand(self, ci: int, loop_t: Optional[int], iteration: int) -> None:
        c = self._clips[ci]
        play = c.spec["play"]
        hold = int(c.spec["hold"])
        if c.play_axis == "none":
            values: List[Optional[int]] = [None]
        else:
            n = c.play_src.ax.t if c.play_axis == "t" else c.play_src.ax.z
            values = _tl_directed(_tl_span(n, play["from"], play["to"], play["step"],
                                           f"{c.path}.play"), play["direction"], iteration)
        for val in values:
            tp = val if c.play_axis == "t" else None
            zp = val if c.play_axis == "z" else None
            for _ in range(hold):
                self.frames.append((ci, tp, zp, loop_t))

    def _bind(self, v: _View, tp: Optional[int], zp: Optional[int],
              loop_t: Optional[int]) -> Tuple[int, Optional[int], Any]:
        """``(m, t, z)`` for view ``v`` at a frame: ``z`` is an index or a reducer name.
        ``t`` is ``None`` only for a tile on t, which never happens here (it sets t_fix)."""
        t = v.t_fix if v.t_fix is not None else (tp if tp is not None else
                                                 (loop_t if loop_t is not None else 0))
        if v.z_fix is not None:
            z: Any = v.z_fix
        elif zp is not None:
            z = zp
        else:
            z = v.z_red if v.z_red is not None else 0
        return v.m, t, z

    def _validate(self) -> None:
        """Every frame's every view in range, and the window sample sets collected."""
        seen = set()
        for ci, tp, zp, lt in self.frames:
            key = (ci, tp, zp, lt)
            if key in seen:
                continue
            seen.add(key)
            for v in self._clips[ci].views:
                m, t, z = self._bind(v, tp, zp, lt)
                ax = v.src.ax
                if not (0 <= t < ax.t):
                    raise TimelineError(
                        f"{v.path}: needs t={t}, but source {v.src.letter} has T={ax.t}. A "
                        f"loop over a longer source, or a play range past its end, reaches "
                        f"beyond it; narrow the range or pin this panel's t.")
                if isinstance(z, int) and not (0 <= z < ax.z):
                    raise TimelineError(
                        f"{v.path}: needs z={z}, but source {v.src.letter} has Z={ax.z}")
                if v.render != "labels":
                    auto = [ch for ch in v.chans if ch not in v.explicit]
                    if auto and v.contrast["mode"] == "auto":
                        self._samples.setdefault(v.wkey, set()).add((m, t, z))

    def _layout(self) -> None:
        """The canvas size, and where every view of every clip lands on it."""
        nat = []
        for c in self._clips:
            cw = max(v.src.ax.x for v in c.views)
            ch = max(v.src.ax.y for v in c.views)
            nat.append((c.cols * cw, c.rows * ch, cw, ch))
        wn = max(n[0] for n in nat)
        hn = max(n[1] for n in nat)
        mp = self.style.max_px
        f = 1.0 if (not mp or max(wn, hn) <= mp) else float(mp) / float(max(wn, hn))
        self.width = max(1, int(round(wn * f)))
        self.height = max(1, int(round(hn * f)))
        self.scale = f
        for c, (nw, nh, cw, ch) in zip(self._clips, nat):
            g = self.gap if len(c.views) > 1 else 0
            gs = g / f
            fit = min((wn - gs * (c.cols - 1)) / float(nw), (hn - gs * (c.rows - 1)) / float(nh))
            if self.fit == "center":
                fit = min(fit, 1.0)
            sc = f * fit
            c.sc = sc
            cell_w, cell_h = cw * sc, ch * sc
            grid_w = c.cols * cell_w + g * (c.cols - 1)
            grid_h = c.rows * cell_h + g * (c.rows - 1)
            ox, oy = (self.width - grid_w) / 2.0, (self.height - grid_h) / 2.0
            c.place = []
            for i, v in enumerate(c.views):
                r, k = divmod(i, c.cols)
                tw = max(1, int(round(v.src.ax.x * sc)))
                th = max(1, int(round(v.src.ax.y * sc)))
                x0 = int(round(ox + k * (cell_w + g) + (cell_w - tw) / 2.0))
                y0 = int(round(oy + r * (cell_h + g) + (cell_h - th) / 2.0))
                x0 = max(0, min(x0, self.width - tw)) if tw <= self.width else 0
                y0 = max(0, min(y0, self.height - th)) if th <= self.height else 0
                c.place.append((tw, th, x0, y0))

    # ── pixels ───────────────────────────────────────────────────────────────
    def _read(self, v: _View, m: int, t: int, z: int, c: int, *, image: bool) -> np.ndarray:
        """One ``(Y, X)`` plane: the image, or the view's Voxel raster rendered as image."""
        if v.layer and not image:
            return v.src.voxel(v.layer, v.path)[m, t, z, c]
        prov = v.src.prov
        if prov is None:
            raise TimelineError(f"{v.path}: source {v.src.letter} has no image to render")
        ax = v.src.ax
        return prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

    def _plane(self, v: _View, m: int, t: int, z: Any, c: int, *, image: bool) -> np.ndarray:
        if isinstance(z, int):
            return np.asarray(self._read(v, m, t, z, c, image=image))
        nz = v.src.ax.z
        if nz == 1:
            return np.asarray(self._read(v, m, t, 0, c, image=image))
        if z == "mid":
            return np.asarray(self._read(v, m, t, nz // 2, c, image=image))
        return _reduce_z(np.stack([np.asarray(self._read(v, m, t, zz, c, image=image))
                                   for zz in range(nz)]), z)

    def _image_is_raster(self, v: _View) -> bool:
        return bool(v.layer) and v.render == "image"

    def measure_steps(self) -> int:
        """How many progress ticks :meth:`measure` will report."""
        return sum(min(len(s), _WINDOW_SAMPLES) for s in self._samples.values())

    def measure(self, progress: Optional[Callable[[], None]] = None) -> None:
        """Measure every display window once. Called by the first ``render`` if not before.

        ONE window per (source, raster, z mode, contrast settings, channel) for the whole
        movie, shared by every panel and tile that shows it. Sharing across tiles is the
        point: a 7x7 grid of positions with one window per tile would make every well look
        equally bright, which is a picture of the normalization rather than of the plate."""
        if self._measured:
            return
        groups: Dict[Any, Tuple[_View, set]] = {}
        for c in self._clips:
            for v in c.views:
                if v.render == "labels":
                    continue
                g = groups.setdefault(v.wkey, (v, set()))
                g[1].update(ch for ch in v.chans if ch not in v.explicit)
        for key, (v, chans) in groups.items():
            chans_l = sorted(chans)
            if not chans_l:
                continue
            raster = self._image_is_raster(v)
            if v.contrast["mode"] == "full":
                hi = self._full_range(v, raster)
                for ch in chans_l:
                    self._windows[(key, ch)] = (0.0, hi)
                continue
            triples = sorted(self._samples.get(key, ()),
                             key=lambda s: (s[0], s[1], 0 if isinstance(s[2], int) else 1,
                                            s[2] if isinstance(s[2], int) else 0, str(s[2])))
            if not triples:
                for ch in chans_l:
                    self._windows[(key, ch)] = (0.0, 1.0)
                continue
            n = len(triples)
            idx = np.unique(np.linspace(0, n - 1, min(n, _WINDOW_SAMPLES)).astype(int))
            pools: Dict[int, List[np.ndarray]] = {ch: [] for ch in chans_l}
            for i in idx:
                m, t, z = triples[int(i)]
                for ch in chans_l:
                    a = self._plane(v, m, t, z, ch, image=not raster).ravel()
                    # a deterministic stride, not a random subsample, so the window is
                    # reproducible and the memo fence holds
                    pools[ch].append(a[:: max(1, a.size // 200_000)])
                if progress is not None:
                    progress()
            lo_pct, hi_pct = v.contrast["low_pct"], v.contrast["high_pct"]
            for ch in chans_l:
                vals = np.concatenate(pools[ch])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    self._windows[(key, ch)] = (0.0, 1.0)
                    continue
                lo = float(np.percentile(vals, max(0.0, min(100.0, lo_pct))))
                hi = float(np.percentile(vals, max(0.0, min(100.0, hi_pct))))
                self._windows[(key, ch)] = (lo, hi if hi > lo else lo + 1.0)
        self._measured = True

    def _full_range(self, v: _View, raster: bool) -> float:
        """The sensor's range from ``bit_depth``, else the dtype's, else 0..1 for floats."""
        if v.src.bit_depth:
            return float(2 ** int(v.src.bit_depth) - 1)
        probe = self._plane(v, v.m, v.t_fix or 0, v.z_fix or 0, v.chans[0], image=not raster)
        if np.issubdtype(probe.dtype, np.integer):
            return float(np.iinfo(probe.dtype).max)
        return 1.0

    def window(self, v: _View, ch: int) -> Tuple[float, float]:
        if ch in v.explicit:
            return v.explicit[ch]
        self.measure()
        return self._windows.get((v.wkey, ch), (0.0, 1.0))

    def first_windows(self) -> List[Tuple[float, float]]:
        """The ``(lo, hi)`` window of each channel of the first clip's first panel, which
        for a flat movie is THE window."""
        v = self._clips[0].views[0]
        return [self.window(v, ch) for ch in v.chans]

    def clip_count(self) -> int:
        return len(self._clips)

    def _label_plane(self, v: _View, m: int, t: int, z: Any) -> np.ndarray:
        arr = v.src.voxel(v.layer, v.path)
        if isinstance(z, int):
            return np.asarray(arr[m, t, z, v.lab_c])
        nz = arr.shape[2]
        if nz == 1:
            return np.asarray(arr[m, t, 0, v.lab_c])
        if z == "mid":
            return np.asarray(arr[m, t, nz // 2, v.lab_c])
        # a projection of a label raster: the largest id along the column wins. `mean` of ids
        # is not an id, so it projects the same way `max` does.
        return np.asarray(arr[m, t, :, v.lab_c]).max(axis=0)

    def _palette(self, v: _View, ids: np.ndarray) -> np.ndarray:
        mode = v.lab["color"]
        top = int(ids.max()) if ids.size else 0
        if mode == "deconflicted":
            from nodegraph.catalog._shared.labels import _label_centroids
            present = np.unique(ids)
            present = present[present > 0]
            cent, _n = _label_centroids(ids, present)
            return label_palette(present, mode, centroids=cent)
        key = (mode, top)
        lut = self._luts.get(key)
        if lut is None:
            lut = label_palette(np.arange(1, top + 1), mode)
            self._luts[key] = lut
        return lut

    def _view_rgb(self, v: _View, m: int, t: int, z: Any) -> np.ndarray:
        ax = v.src.ax
        if v.render == "labels":
            return np.zeros((ax.y, ax.x, 3), np.uint8)
        raster = self._image_is_raster(v)
        planes = [self._plane(v, m, t, z, ch, image=not raster) for ch in v.chans]
        return _composite(planes, v.tints, [self.window(v, ch) for ch in v.chans],
                          v.gammas, v.brightness)

    def frame_info(self, k: int) -> Dict[str, Any]:
        """What frame ``k`` shows, for the editor's readout: its segment and bindings."""
        ci, tp, zp, lt = self.frames[int(k)]
        seg = next(i for i, (s, n) in enumerate(self.segment_spans) if s <= k < s + n)
        return {"segment": seg, "clip": self._clips[ci].path, "t": tp, "z": zp, "loop_t": lt}

    def render(self, k: int) -> np.ndarray:
        """Output frame ``k``: every view composited, resized, labelled and placed, then the
        burn-ins. ``(height, width, 3)`` uint8, the same size for every ``k``."""
        ci, tp, zp, lt = self.frames[int(k)]
        c = self._clips[ci]
        key = (ci, tp, zp, lt)
        if self._last[0] == key and self._last[1] is not None:
            # a held or ping-ponged repeat: the picture is the same, only the counter moved,
            # so re-burn the text and read nothing
            canvas, bound = self._last[1]
            return self._burn_in(canvas, c, k, bound)
        canvas = np.empty((self.height, self.width, 3), np.uint8)
        canvas[:] = np.asarray(self.bg, np.uint8)
        marks: List[Tuple[float, float, int, int]] = []
        captions: List[Tuple[int, int, str]] = []
        bound = []
        for v, (tw, th, x0, y0) in zip(c.views, c.place):
            m, t, z = self._bind(v, tp, zp, lt)
            bound.append((m, t, z))
            img = _fit_resize(self._view_rgb(v, m, t, z), (tw, th))
            if v.render != "image":
                full = self._label_plane(v, m, t, z)
                small = resample_ids(full, (th, tw))
                img = paint_labels(img, small, self._palette(v, full),
                                   style=v.lab["style"], fill_opacity=v.lab["fill_opacity"],
                                   outline_px=v.lab["outline_px"])
                if v.lab["show_ids"]:
                    fy, fx = th / float(full.shape[0]), tw / float(full.shape[1])
                    if fy < 1.0:
                        got = label_id_marks(small, scale_y=1.0, scale_x=1.0,
                                             min_px=v.lab["id_px"], max_ids=MAX_ID_LABELS)
                    else:
                        got = label_id_marks(full, scale_y=fy, scale_x=fx,
                                             min_px=v.lab["id_px"], max_ids=MAX_ID_LABELS)
                    marks.extend((x0 + x, y0 + y, i, v.lab["id_px"]) for x, y, i in got)
            ch_, cw_ = min(th, self.height - y0), min(tw, self.width - x0)
            canvas[y0:y0 + ch_, x0:x0 + cw_] = img[:ch_, :cw_]
            if c.ann["captions"]:
                text = v.caption or self._tile_caption(v, m, t, z)
                if text:
                    captions.append((x0, y0, tw, th, text))
        if marks or captions:
            canvas = self._draw_extras(canvas, marks, captions,
                                       self._block_rect(self._block_lines(c, k, bound)))
        self._last = (key, (canvas, bound))
        return self._burn_in(canvas, c, k, bound)

    def _tile_caption(self, v: _View, m: int, t: int, z: Any) -> str:
        if v.tile_axis == "m":
            return v.src.position_name(m)
        if v.tile_axis == "c":
            return v.names[0] if v.names else ""
        if v.tile_axis == "z":
            return f"z {int(z) + 1}"
        if v.tile_axis == "t":
            iv = self._interval(v.src)
            return _time_text(t, iv, v.src.ax.t) if iv > 0 else f"t {t + 1}"
        return ""

    def _interval(self, src: MovieSource) -> float:
        """Seconds between timepoints: the node's own ``Frame interval`` for ``A``, which
        already derives from ``dt_s`` and which the user may override; ``dt_s`` for the
        others."""
        if src.letter == "A":
            return self.style.interval_s
        return float(src.dt_s or 0.0)

    def _block_rect(self, lines: Sequence[str]) -> List[Tuple[int, int, int, int]]:
        """``(x0, y0, x1, y1)`` of each LINE of the text block when it sits in a TOP corner
        (empty otherwise). Per line, not one bounding box: the block is a staircase — a short
        counter over a long title — and a box as wide as the title would evict every tile
        caption along the top row for no reason. Mirrors :func:`_annotate`'s own layout."""
        corner = self.style.corner
        if not lines or not corner.startswith("top"):
            return []
        size = _text_px(self.height, self.width, self.style.font_px)
        font = _font(size)
        pad = max(6, size // 2)
        m = max(2, pad // 3)
        out = []
        for i, s in enumerate(str(x) for x in lines if x):
            w = int(font.getlength(s))
            y = pad + i * int(size * 1.35)
            if corner.endswith("left"):
                out.append((pad - m, y - m, pad + w + m, y + size + m))
            else:
                out.append((self.width - pad - w - m, y - m, self.width - pad + m,
                            y + size + m))
        return out

    def _draw_extras(self, canvas: np.ndarray, marks, captions,
                     avoid: Sequence[Tuple[int, int, int, int]] = ()) -> np.ndarray:
        """Label ids and tile captions, drawn in one Pillow pass after placement.

        A tile caption that would land on a line of the movie's own text block (``avoid``)
        moves to another corner of ITS OWN tile — top-right, then bottom-left, then
        bottom-right — and is left out when all four are covered. In a grid that fills the
        frame the first tile's corner IS the frame's corner, and two lines of text on top of
        each other read as neither; pushing the caption straight down instead (the first
        cut) landed it on the next row's caption."""
        from PIL import Image, ImageDraw
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        fonts: Dict[int, Any] = {}
        for x, y, ident, px in marks:
            f = fonts.get(px)
            if f is None:
                f = fonts[px] = _font(int(px))
            # Pillow's own stroke, not a stamp at 1 px offsets: at id sizes a second draw
            # one pixel over floods the counters of 0, 8 and 9 (overlays.draw_id says why)
            draw.text((x, y), str(ident), font=f, fill=(255, 255, 255), anchor="mm",
                      stroke_width=max(1, int(px) // 7), stroke_fill=(0, 0, 0))
        if captions:
            size = max(9, int(round(_text_px(self.height, self.width,
                                             self.style.font_px) * 0.8)))
            f = _font(size)
            pad = max(3, size // 3)
            for x0, y0, tw, th, text in captions:
                w = int(f.getlength(text))

                def hits(cx: int, cy: int) -> bool:
                    return any(cx < r[2] and cx + w > r[0] and cy < r[3] and cy + size > r[1]
                               for r in avoid)

                spot = next(((cx, cy) for cx, cy in (
                    (x0 + pad, y0 + pad), (x0 + tw - pad - w, y0 + pad),
                    (x0 + pad, y0 + th - pad - size), (x0 + tw - pad - w, y0 + th - pad - size))
                    if not hits(cx, cy)), None)
                if spot is not None:
                    _draw_text(draw, spot, text, f, self.style.text_rgb, "la")
        return np.asarray(img)

    def _block_lines(self, c: _Clip, k: int, bound) -> List[str]:
        """The clip's text block at frame ``k``: counter, clock, z readout, title."""
        ann, st = c.ann, self.style
        v0 = c.views[0]
        _m0, t0, z0 = bound[0]
        src = v0.src
        lines: List[str] = []
        if ann["frame"]:
            lines.append(f"{k + 1}/{self.n_frames}")
        if ann["time"] and v0.tile_axis != "t":
            iv = self._interval(src)
            if iv and iv > 0:
                lines.append(_time_text(t0, iv, src.ax.t))
        if ann["z"] and v0.tile_axis != "z" and isinstance(z0, int) and src.ax.z > 1:
            txt = f"z {z0 + 1}/{src.ax.z}"
            if src.z_step:
                micro = _micro(_font(_text_px(self.height, self.width, st.font_px)))
                txt += f"  {z0 * float(src.z_step):.1f} {micro}"
            lines.append(txt)
        if ann["title"]:
            lines.append(ann["title"])
        return lines

    def _burn_in(self, canvas: np.ndarray, c: _Clip, k: int, bound) -> np.ndarray:
        """The clip's text block, scale bar, legend and position name, over the canvas."""
        ann, st = c.ann, self.style
        v0 = c.views[0]
        m0 = bound[0][0]
        src = v0.src
        lines = self._block_lines(c, k, bound)
        tiled_c = v0.tile_axis == "c"
        px = (float(src.px_um) / c.sc) if (src.px_um and c.sc > 0) else None
        return _annotate(canvas, lines=lines, pixel_size_um=px,
                         ch_names=[] if (tiled_c or getattr(src, "picture", False))
                         else v0.names, tints=v0.tints,
                         position=("" if v0.tile_axis == "m" else src.position_name(m0)),
                         show={"scalebar": ann["scalebar"], "channels": ann["channels"],
                               "position": ann["position"]},
                         corner=st.corner, font_px=st.font_px, text_rgb=st.text_rgb)
