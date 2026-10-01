"""The Export Movie preview: build the movie the node WOULD write, before it writes it.

The point is that it is not a mock-up. Every frame the Movie Editor's monitor shows is
produced by :class:`nodegraph.catalog._shared.movie_timeline.Timeline`, the same object
``io.write_movie``'s compute renders through, so the compositing, the fixed display window,
the resize and every burn-in are the export's own, not a second implementation that agrees
with it on the day it was written. If the preview and the file ever disagree, that is a bug in
one renderer rather than a difference between two.

Qt-free on purpose: this is the half of the preview a probe can drive headless. The window
owns the payloads (:meth:`nodelab_v2.runner.EngineRunner.fetch`), the editor owns the
widgets (:mod:`nodelab_v2.movie_editor`), and both meet here.

Two things it deliberately does NOT do:

* **It does not write anything.** No path is needed, no codec is opened, `Existing` is not
  consulted. So it works on a node whose File socket is still empty, which is exactly when
  you want to look.
* **It does not pull the graph.** It is handed payloads that were already computed and only
  renders frames out of them. Scrubbing is therefore as cheap as reading planes.

(The flat-movie ``MoviePreviewDialog`` this module used to hold was retired on 2026-09-30:
the Movie Editor's monitor plays flat movies and timelines alike.)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: Longest edge the preview renders at, whatever the node's own `Max px` says. The monitor is
#: a few hundred pixels on screen; rendering a 4000 px frame to scale it down would make
#: scrubbing unusable for a difference nobody can see. The EXPORT still uses the node's real
#: value — this cap applies to the preview only, and the editor says so.
_PREVIEW_MAX_PX = 900


class _StateChannel:
    """A stand-in for ``EvalContext.channel(0)`` backed by a node's SAVED STATE.

    Exposes the one method :func:`nodegraph.catalog.io.write_movie.plan_movie` asks for, so
    the preview and the engine hand the renderer the same shape. (It is also why the reader
    is an object at all rather than a plain function — see that function's docstring.)
    """

    __slots__ = ("_get",)

    def __init__(self, get) -> None:
        self._get = get

    def param(self, name: str) -> Any:
        return self._get(name)


def resolve_params(spec, state: Dict[str, Any], env) -> Any:
    """A channel-like reader over a node's SAVED STATE, matching what the engine does.

    The engine resolves a param as *override -> ``derive`` -> socket default*
    (:class:`nodegraph.engine.ChannelContext`), and it does it against the node's envelope.
    The preview has no ``EvalContext``, so the same three steps are re-created here against
    the payload's metadata. Getting this wrong is not cosmetic: ``interval_s`` carries
    ``derive="dt_s or 0.0"``, so a resolver that skipped the derive would preview every movie
    with no clock and then export one with a clock.
    """
    from nodegraph.metadata import envelope_symbols, eval_derive

    socks = {s.name: s for s in spec.inputs}
    try:
        symbols = envelope_symbols(env)
    except Exception:      # noqa: BLE001 — a preview must not die on an odd envelope
        symbols = {}

    def param(name: str) -> Any:
        if name in state and state[name] not in (None, ""):
            return state[name]
        sock = socks.get(name)
        if sock is None:
            return None
        if getattr(sock, "derive", ""):
            try:
                return eval_derive(sock.derive, symbols)
            except Exception:      # noqa: BLE001 — bad expr falls back to the static default
                pass
        return sock.default

    return _StateChannel(param)


def _cap_note(export_px: int, capped: bool, cap_px: int) -> str:
    """The "rendered smaller than the export" note, or "" — said only when TRUE.

    ``export_px`` is the longest edge the EXPORT will actually have (its own `Max px` already
    applied); ``capped`` says whether that came from `Max px` rather than the data's full
    size. The note exists to stop the preview quietly differing from the file, so one
    claiming a downscale that did not happen would be the same defect with the opposite
    sign, and would train the reader to ignore the line in the case that matters."""
    if export_px <= cap_px:
        return ""
    return (f"Preview rendered at {cap_px} px for responsiveness; the export writes at "
            + (f"{export_px} px." if capped else "full size."))


def build_plan(payload, spec, params: Dict[str, Any], modes: Dict[str, str], env, *,
               cap_px: int = _PREVIEW_MAX_PX):
    """A flat :class:`MoviePlan` for ``payload`` under a node's saved ``params``/``modes``.

    ``params`` and ``modes`` are the two LIVE dicts a ``NodeRecord`` keeps, passed separately
    because that is how the document stores them — the engine only fuses them (as
    ``params["__modes__"]``) when it builds the run graph, and reaching for that key here
    would find nothing.

    Returns ``(plan, note)``; raises ``ValueError`` with the node's own message when the
    settings cannot produce a movie at all, which the caller shows as-is.
    """
    from nodegraph.catalog.io.write_movie import plan_movie

    md = dict(getattr(payload, "metadata", {}) or {})
    reader = resolve_params(spec, params, env)
    layer = str(params.get("layer", "") or "")

    plan = plan_movie(payload, ch=reader, modes=dict(modes or {}),
                      calib=md.get, meta=md.get, layer=layer)

    note_bits = []
    real_px = plan.max_px
    # What the EXPORT's longest edge will actually be: its own cap, or the data's own size
    # when it has none. The resize never upscales, so a frame already smaller than a cap is
    # untouched by it.
    source_px = max(int(payload.axes.y), int(payload.axes.x))
    export_px = min(source_px, real_px) if real_px else source_px
    note = _cap_note(export_px, bool(real_px) and real_px < source_px, cap_px)
    if note:
        note_bits.append(note)
        plan.max_px = cap_px
    if plan.show.get("time") and not plan.interval_s:
        note_bits.append(
            "No frame interval on this data, so the clock is omitted — type one into "
            "`Frame interval` if you know it.")
    note_bits.append("Nothing is written until you pull the node.")
    return plan, "  ".join(note_bits)


def flat_timeline_spec(spec, params: Dict[str, Any], modes: Dict[str, str],
                       env) -> Dict[str, Any]:
    """The node's CURRENT flat settings (Sweep = time/z) as a one-clip timeline.

    What the editor's *Convert to timeline* starts from, so the first thing a converted movie
    shows is exactly the movie the node was already making. Channels the flat node measured
    with ``auto`` contrast are linked to the Viewer (``link: viewer``): the user asked for the
    movie to wear the LUTs they tuned, and a channel whose source the Viewer has not shown
    keeps the auto window it had. Typed (``absolute``) windows stay manual."""
    from nodegraph.catalog._shared.movie_timeline import flat_spec
    from nodegraph.metadata import parse_channels

    reader = resolve_params(spec, params, env)
    n_c = int(getattr(getattr(env, "axes", None), "c", 1) or 1)
    picked = parse_channels(reader.param("channels"))
    chans = (list(range(n_c)) if picked is None
             else [c for c in picked if 0 <= c < n_c]) or list(range(n_c))
    sweep = modes.get("sweep", "time")
    sweep = "time" if sweep not in ("time", "z") else sweep
    contrast = modes.get("contrast", "auto")
    out = flat_spec(
        sweep=sweep, z_reduce=modes.get("z_reduce", "max"), channels=chans,
        layer=str(reader.param("layer") or ""), m=0, contrast=contrast,
        low_pct=float(reader.param("low_pct") or 0.0),
        high_pct=float(reader.param("high_pct") or 100.0),
        black=float(reader.param("black") or 0.0), white=float(reader.param("white") or 0.0),
        gamma=float(reader.param("gamma") or 1.0),
        brightness=float(reader.param("brightness") or 100.0),
        show={"frame": bool(reader.param("show_frame")),
              "time": bool(reader.param("show_time")),
              "scalebar": bool(reader.param("show_scalebar")),
              "channels": bool(reader.param("show_channels")),
              "position": bool(reader.param("show_position"))})
    if contrast == "auto":
        for d in out["segments"][0]["panels"][0]["display"].values():
            d["link"] = "viewer"
    return out


def preview_style(spec, params: Dict[str, Any], env, *, cap_px: int = _PREVIEW_MAX_PX):
    """``(MovieStyle, export_max_px)`` from a node's movie-wide sockets, with ``Max px``
    capped for the preview (``cap_px=0`` = no cap, which is what a parity check wants)."""
    from nodegraph.catalog._shared.movie_draw import _TEXT_RGB
    from nodegraph.catalog._shared.movie_timeline import MovieStyle

    reader = resolve_params(spec, params, env)
    real = int(reader.param("max_px") or 0)
    shown = real if not cap_px else (min(real, cap_px) if real else cap_px)
    return MovieStyle(
        max_px=shown, corner=str(reader.param("corner") or "top_left"),
        font_px=int(reader.param("font_px") or 0),
        text_rgb=_TEXT_RGB.get(str(reader.param("text_color")), (255, 255, 255)),
        interval_s=float(reader.param("interval_s") or 0.0)), real


def build_timeline(payloads: Dict[str, Any], timeline, spec, params: Dict[str, Any], env, *,
                   cap_px: int = _PREVIEW_MAX_PX) -> Tuple[Any, str]:
    """A :class:`Timeline` over the given source ``payloads`` (``{"A": Dataset, ...}``) for
    ``timeline`` (a spec dict or its JSON), styled by the node's saved ``params``.

    ``env`` is the MOVIE node's envelope, which is source A's (the node is a tap), so the
    ``interval_s`` derive fires against A's frame interval exactly as it does in the engine.
    Every source's calibration is read off its own payload here; the engine reads A's through
    ``ctx.calib`` instead, which returns the same values the payload carries.

    Returns ``(timeline, note)``; raises :class:`TimelineError` (a ``ValueError``) naming the
    spec path when the timeline cannot be rendered with these sources."""
    from nodegraph.catalog._shared.movie_timeline import MovieSource, Timeline

    sources = {}
    for letter, ds in payloads.items():
        if ds is None:
            continue
        md = dict(getattr(ds, "metadata", {}) or {})
        sources[letter] = MovieSource(letter, ds, calib=md.get, meta=md.get)
    style, real = preview_style(spec, params, env, cap_px=cap_px)
    tl = Timeline(timeline, sources, style)
    note = ""
    if cap_px:
        full, _r = preview_style(spec, params, env, cap_px=0)
        if full.max_px != style.max_px:
            big = Timeline(tl.spec, sources, full)    # validation only: reads no pixels
            note = _cap_note(max(big.width, big.height), bool(real), cap_px)
    return tl, note
