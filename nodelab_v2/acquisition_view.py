"""What the Viewer says about a payload's acquisition (2026-10-13) — the channel tooltips and
the Info window. Qt-free, so the selftest can read exactly what a user is shown.

Everything here reads the PAYLOAD's metadata (``Dataset.metadata``), i.e. the data on the
wire being viewed, never the file again: a channel tap shows its own channel's settings
(``channel_acquisition`` is a :data:`~nodegraph.metadata.PER_CHANNEL_KEYS` member), a
Timeseries Builder's series shows its own merged clock, and a frame crop shows the frames
it kept. The file-level record (``acquisition_info``) describes the SOURCE file — the first
file of a chained series — and is labelled that way rather than passed off as the series'.
"""
from __future__ import annotations

import html
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

Rows = List[List[str]]


def _md(md: Any) -> Mapping[str, Any]:
    return md if isinstance(md, Mapping) else {}


def channel_rows(md: Any, c: int) -> Rows:
    """Channel ``c``'s acquisition rows (``[[label, value], …]``), or ``[]``."""
    rows = _md(md).get("channel_acquisition")
    if isinstance(rows, (list, tuple)) and 0 <= int(c) < len(rows):
        got = rows[int(c)] or []
        return [[str(r[0]), str(r[1])] for r in got
                if isinstance(r, (list, tuple)) and len(r) == 2]
    return []


def channel_tooltip(md: Any, c: int, name: str, *, emission_nm: Any = None,
                    hint: str = "") -> str:
    """The rich-text tooltip for channel ``c``'s toggle: its name and emission, then every
    acquisition setting the file recorded for it, then ``hint`` (what clicking does)."""
    head = html.escape(str(name))
    try:
        em = float(emission_nm)
        head += f" · {em:.0f} nm" if math.isfinite(em) else ""
    except (TypeError, ValueError):
        pass
    parts = [f"<b>{head}</b>"]
    rows = channel_rows(md, c)
    if rows:
        cells = "".join(
            f"<tr><td style='padding-right:10px; color:gray'>{html.escape(k)}</td>"
            f"<td>{html.escape(v)}</td></tr>" for k, v in rows)
        parts.append(f"<table cellspacing='0' cellpadding='1'>{cells}</table>")
    if hint:
        parts.append(f"<i>{html.escape(hint)}</i>")
    return "<br>".join(parts)


def _offset_text(off: Any) -> str:
    try:
        o = int(off)
    except (TypeError, ValueError):
        return ""
    sign = "+" if o >= 0 else "-"
    o = abs(o)
    return f"UTC{sign}{o // 3600:02d}:{(o % 3600) // 60:02d}"


def _dur(s: float) -> str:
    s = float(s)
    if abs(s) < 120:
        return f"{s:.1f} s"
    if abs(s) < 7200:
        return f"{s / 60:.2f} min"
    return f"{s / 3600:.2f} h"


def series_time_rows(md: Any, n_t: int) -> Rows:
    """The viewed SERIES' time — its own clock, after every node above it (a Builder's
    interleave, a crop's dropped frame): frame count, first and last frame in local time,
    span, spacing, zone. Empty when the data carries no time axis."""
    from nodegraph.placement import frame_seconds, utc_offset_of, jd_to_datetime_text
    md = _md(md)
    n_t = max(1, int(n_t or 1))
    rows: Rows = [["Frames (T)", str(n_t)]]
    jd = md.get("frame_time_jd")
    secs = frame_seconds(jd, n_t)
    off = utc_offset_of(md)
    zone = _offset_text(off)
    if secs is not None:
        fd = md.get("frame_datetime")
        text = ([str(v) for v in fd] if isinstance(fd, (list, tuple)) and len(fd) == n_t
                and all(fd) else [jd_to_datetime_text(v, off) for v in jd])
        rows.append(["First frame", f"{text[0]}  ({zone})"])
        if n_t > 1:
            rows.append(["Last frame", f"{text[-1]}  ({zone})"])
            rows.append(["Span", _dur(secs[-1] - secs[0])])
            steps = [b - a for a, b in zip(secs, secs[1:])]
            lo, hi = min(steps), max(steps)
            mean = (secs[-1] - secs[0]) / (n_t - 1)
            if hi - lo <= 0.1 * abs(mean):
                rows.append(["Interval", f"{_dur(mean)} (mean; {_dur(lo)} – {_dur(hi)})"])
            else:
                rows.append(["Interval", f"uneven — {_dur(lo)} to {_dur(hi)}, mean "
                                         f"{_dur(mean)}"])
            if lo <= 0:
                rows.append(["Order", "NOT increasing — some frame is not later than the "
                                      "one before it"])
        rows.append(["Clock", "each frame's own acquisition time"])
        src = md.get("utc_offset_source")
        if src:
            rows.append(["Time zone", f"{zone} (from {src})" if src != "file" else
                         f"{zone} (read from the file's own local date)"])
    else:
        from nodegraph.metadata import series_seconds
        # a clockless series an uneven subset (Trim, Crop) left with per-frame seconds
        own = series_seconds(md, n_t) if n_t > 1 else None
        try:
            dt = float(md.get("dt_s") or 0.0)
        except (TypeError, ValueError):
            dt = 0.0
        if own is not None:
            steps = [b - a for a, b in zip(own, own[1:])]
            rows.append(["Span", _dur(own[-1] - own[0])])
            rows.append(["Interval", f"uneven — {_dur(min(steps))} to {_dur(max(steps))}, "
                                     f"mean {_dur((own[-1] - own[0]) / (n_t - 1))}"])
            rows.append(["Clock", "none — each frame's time kept from the source interval "
                                  "(frame_time_s)"])
        elif dt > 0:
            rows.append(["Interval", f"{_dur(dt)} (dt_s — no per-frame clock)"])
        elif n_t > 1:
            rows.append(["Clock", "none — frames are numbered, not timed"])
    return rows


def chain_rows(md: Any) -> Rows:
    """A Timeseries Builder's own account of the series it built (its ``__chain__`` note)."""
    ch = _md(md).get("__chain__")
    if not isinstance(ch, Mapping):
        return []
    rows: Rows = []
    if ch.get("note"):
        rows.append(["Built by", f"Timeseries Builder — {ch['note']}"])
    files = list(ch.get("files") or [])
    per = list(ch.get("per_file") or [])
    first = list(ch.get("first_frame") or [])
    for i, f in enumerate(files):
        bits = []
        if i < len(per):
            bits.append(f"{per[i]} frames")
        if i < len(first) and first[i]:
            bits.append(f"first frame {first[i]}")
        rows.append([f"File {i + 1}", f"{f}" + (f"  ({', '.join(bits)})" if bits else "")])
    if ch.get("order"):
        rows.append(["Ordered by", str(ch["order"])
                     + (" — frames interleaved by their clocks" if ch.get("interleaved")
                        else "")])
    return rows


def info_sections(md: Any, axes: Any) -> List[Tuple[str, Rows]]:
    """Every section the Info window shows, in order: the viewed series' time, how it was
    built, the source file's record, then one section per channel."""
    md = _md(md)
    n_t = int(getattr(axes, "t", 1) or 1)
    n_c = int(getattr(axes, "c", 1) or 1)
    out: List[Tuple[str, Rows]] = [("This series — time", series_time_rows(md, n_t))]
    chain = chain_rows(md)
    if chain:
        out.append(("This series — files", chain))
    src = "Source file" + (" (the series' first file)" if chain else "")
    for sec in md.get("acquisition_info") or []:
        try:
            title, rows = sec
        except (TypeError, ValueError):
            continue
        rows = [[str(r[0]), str(r[1])] for r in rows or []
                if isinstance(r, (list, tuple)) and len(r) == 2]
        if rows:
            out.append((f"{src} — {title}", rows))
    names = list(md.get("channel_names") or [])
    for c in range(n_c):
        rows = channel_rows(md, c)
        if rows:
            nm = str(names[c]) if c < len(names) and names[c] else f"Ch{c + 1}"
            out.append((f"Channel {c + 1}: {nm}", rows))
    return out


def info_text(md: Any, axes: Any) -> str:
    """The Info window as plain text — what "Copy all" puts on the clipboard: every section,
    then the file's raw text blocks verbatim."""
    lines: List[str] = []
    for title, rows in info_sections(md, axes):
        lines.append(title)
        w = max((len(k) for k, _v in rows), default=0)
        lines.extend(f"  {k.ljust(w)}  {v}" for k, v in rows)
        lines.append("")
    raw = _md(md).get("acquisition_text")
    if isinstance(raw, Mapping) and raw:
        lines.append("Raw text blocks recorded in the source file")
        for k, v in raw.items():
            lines.append(f"--- {k} ---")
            lines.append(str(v).rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def has_record(md: Any) -> bool:
    """Whether the payload carries anything beyond the series' own time to show."""
    md = _md(md)
    return bool(md.get("acquisition_info") or md.get("channel_acquisition")
                or md.get("acquisition_text"))


__all__ = ["channel_rows", "channel_tooltip", "series_time_rows", "chain_rows",
           "info_sections", "info_text", "has_record"]
