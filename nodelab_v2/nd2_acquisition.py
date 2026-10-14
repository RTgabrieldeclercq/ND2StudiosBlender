"""ND2 acquisition record — the true per-frame clock, its time zone, and every setting the
file carries (2026-10-13). Qt-free; takes an open ``nd2.ND2File``.

**The clock.** An ND2 holds exactly ONE absolute timestamp: frame 0's ``dTimeAbsolute``
(a Julian day, in UTC), in the ``ImageMetadataSeqLV|0`` record. Every frame's time is
``relativeTimeMs`` (the ``AcqTimesCache``), and its origin is the start of the RUN that
wrote the file — for an NIS JOBS run that writes several files, the job start, shared by
all of them, not the file's own first frame. ``nd2`` (0.11.3, ``_parse.py``
``load_frame_metadata``) builds each frame's ``absoluteJulianDayNumber`` as
``jd0 + relativeTimeMs``, which adds frame 0's own offset twice. On an ordinary file that
offset is ~0 and nothing shows; on the CRC_TF_ELISA ``CRC_1_T1..T5`` JOBS series it is
20-151 min, different per file, and the Timeseries Builder interleaved the five files out
of order (true frames 0,1,2,5,3,6,4,7,10,8,…) — the "twitching" the user reported. On the
WellA3 640/GFP pair it doubled the true 1207.3 s gap to 2414.6 s, which then read as GFP
starting during the 640's NEXT sweep. :func:`frame_jd` is the correct composition:

    jd(frame k) = jd0 + (relativeTimeMs[k] - relativeTimeMs[0]) / 86 400 000

Verified on both data sets: ``jd0 - 7 h`` equals the JOBS folder's own name
(``20261006_194721_215``) plus frame 0's ``relativeTimeMs`` to the millisecond on all five
files, and each file's ``text_info["date"]`` (local time, written when the file closed) is
its corrected LAST frame to the second, while the library's clock misses it by 20-151 min.

**The time zone.** ``dTimeAbsolute`` is UTC — the same comparison proves it (Arizona,
UTC-7). The file never states its zone, but ``text_info["date"]`` is the microscope PC's
LOCAL wall clock at a known frame, so :func:`utc_offset` reads the offset off the file
itself and only falls back to this computer's zone when the date cannot be matched.

**The settings.** Everything the user may need to report — laser line and power %,
detector and gain, offset, pinhole, emission range, dwell time, scanner, objective, the
recorded per-frame data (gain/laser power per channel, PFS, stage), software version — is
gathered into display-ready ``[label, value]`` rows by :func:`acquisition_record`: one list
of sections for the file, one list of rows per channel. Text, not numbers: these are for a
person to read, and every value keeps the unit the microscope wrote with it. The raw
``text_info`` blocks ride along whole, so nothing the SDK exposes is hidden behind a parse.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Milliseconds per day — ``relativeTimeMs`` → Julian days.
MS_PER_DAY = 86_400_000.0

#: The Julian day of the Unix epoch (1970-01-01 00:00:00 UTC).
JD_UNIX_EPOCH = 2440587.5

#: How far a file's local ``date`` may sit from the frame it names and still identify the
#: zone. NIS writes it when the file closes, within a second of the last frame on the lab's
#: files; two minutes is generous for a slow flush and far below the 15-min zone grain.
_ZONE_MATCH_S = 120.0

#: The grain zone offsets come in (India +5:30, Nepal +5:45, Chatham +12:45).
_ZONE_GRAIN_S = 900.0

#: Recorded-data columns of ``ND2File.events`` that index the frame rather than describe
#: the acquisition — shown nowhere, they are what the rows are keyed by.
_EVENT_INDEX_COLS = ("Index", "T Index", "P Index", "Z Index", "C Index", "Position Name",
                     "Time [s]")


def _safe(get, default=None):
    try:
        return get()
    except Exception:
        return default


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ── the clock ────────────────────────────────────────────────────────────────────

def frame0_stamp(f: Any) -> Optional[Tuple[float, float]]:
    """``(jd0, rel0_ms)`` — frame 0's absolute Julian day and its relative time, exactly as
    NIS wrote them — or ``None`` when the file does not carry them.

    Read through the PUBLIC ``unstructured_metadata`` with ``strip_prefix=False`` (the
    stripped keys are ``TimeAbsolute`` / ``TimeMSec``), from ``ImageMetadataSeqLV|0`` on a
    v3+ file and ``ImageMetadataSeq|0`` on an older one."""
    for key in ("ImageMetadataSeqLV|0", "ImageMetadataSeq|0"):
        um = _safe(lambda k=key: f.unstructured_metadata(strip_prefix=False, include={k}))
        rec = um.get(key) if isinstance(um, dict) else None
        if not isinstance(rec, dict):
            continue
        rec = rec.get("SLxPictureMetadata", rec)
        jd, ms = _num(rec.get("dTimeAbsolute")), _num(rec.get("dTimeMSec"))
        if jd is not None and jd > 1.0 and ms is not None:
            return jd, ms
    return None


def frame_jd(stamp: Tuple[float, float], rel_ms: Any) -> Optional[float]:
    """The true Julian day (UTC) of the frame whose ``relativeTimeMs`` is ``rel_ms`` —
    frame 0's stamp plus how long after frame 0 it was taken. ``None`` for a non-number."""
    rel = _num(rel_ms)
    if rel is None:
        return None
    jd0, ms0 = stamp
    return jd0 + (rel - ms0) / MS_PER_DAY


def jd_to_utc(jd: float) -> datetime:
    """A Julian day → a naive UTC ``datetime``."""
    return datetime(1970, 1, 1) + timedelta(seconds=(float(jd) - JD_UNIX_EPOCH) * 86400.0)


# ── the time zone ────────────────────────────────────────────────────────────────

_DATE_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p",       # NIS on an en-US / en-GB PC
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S",
)


def parse_nis_date(text: Any) -> List[datetime]:
    """Every reading of an NIS ``text_info["date"]`` string under the formats NIS writes
    across locales — a list, because ``10/7/2026`` is October 7th in one and July 10th in
    another, and only the clock can say which (:func:`utc_offset`)."""
    s = " ".join(str(text or "").split())
    out: List[datetime] = []
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if d not in out:
            out.append(d)
    return out


def local_offset_s(jd: float) -> int:
    """THIS computer's UTC offset (seconds east) at the instant ``jd`` — DST-correct for that
    date, not for today."""
    unix = (float(jd) - JD_UNIX_EPOCH) * 86400.0
    try:
        return int(time.localtime(unix).tm_gmtoff)
    except (OverflowError, OSError, ValueError):
        return int(-time.timezone)


def utc_offset(date_text: Any, jds: Sequence[float]) -> Tuple[int, str]:
    """``(seconds east of UTC, source)`` for the microscope PC that wrote the file.

    The file's ``date`` is that PC's local wall clock at a frame it also stamped in UTC —
    the last one on every file checked (NIS writes it at close), the first on an export
    that predates that habit. So each reading of the date is compared with each candidate
    instant in ``jds``; a difference that lands on a whole zone grain (15 min) within
    :data:`_ZONE_MATCH_S` IS the zone. ``source`` is ``"file"`` then.

    Otherwise ``"this computer"``: the zone of the machine running the app, for that date.
    Right whenever analysis happens in the lab's own zone, and the card says which it is,
    so a file from elsewhere is not silently mislabelled."""
    cands = [j for j in jds if _num(j) is not None]
    for local in parse_nis_date(date_text):
        for jd in cands:
            diff = (local - jd_to_utc(jd)).total_seconds()
            snapped = round(diff / _ZONE_GRAIN_S) * _ZONE_GRAIN_S
            if abs(diff - snapped) <= _ZONE_MATCH_S and abs(snapped) <= 14 * 3600:
                return int(snapped), "file"
    ref = cands[0] if cands else (time.time() / 86400.0 + JD_UNIX_EPOCH)
    return local_offset_s(ref), "this computer"


def offset_text(offset_s: Any) -> str:
    """``-25200`` → ``"UTC-07:00"``."""
    try:
        o = int(offset_s)
    except (TypeError, ValueError):
        return ""
    sign = "+" if o >= 0 else "-"
    o = abs(o)
    return f"UTC{sign}{o // 3600:02d}:{(o % 3600) // 60:02d}"


# ── the settings: text_info parsed into a tree ───────────────────────────────────

def _indent(line: str) -> int:
    n = 0
    for ch in line:
        if ch == "\t":
            n += 4
        elif ch == " ":
            n += 1
        else:
            break
    return n


def parse_settings_text(text: Any) -> List[Tuple[str, Any]]:
    """An NIS settings block (``text_info["capturing"]``, or a plane's ``Camera Settings``)
    → a tree of ``(key, value)`` pairs, where ``value`` is a string for a ``key: value``
    line and a nested list for a ``key:`` header. A bare line (``Enabled``, a channel list)
    becomes ``(line, "")``. Nesting follows the indentation, tabs counting as four."""
    root: List[Tuple[str, Any]] = []
    stack: List[Tuple[int, List[Tuple[str, Any]]]] = [(-1, root)]
    for raw in str(text or "").splitlines():
        if not raw.strip():
            continue
        ind = _indent(raw)
        line = raw.strip()
        while len(stack) > 1 and ind <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.endswith(":"):
            kids: List[Tuple[str, Any]] = []
            parent.append((line[:-1].strip(), kids))
            stack.append((ind, kids))
        elif ": " in line:
            k, v = line.split(": ", 1)
            parent.append((k.strip(), v.strip()))
        else:
            parent.append((line, ""))
    return root


_DEVICE_FLAGS = ("Enabled", "Disabled", "On", "Off")


def _flatten(tree: List[Tuple[str, Any]], prefix: str = "") -> List[List[str]]:
    """A settings tree → ``[label, value]`` rows. A DEVICE block — leaves only, one of them
    an on/off flag (``Laser 488 nm: Enabled, Power: 3.0, Laser Port: Standard``) — folds
    onto ONE row, because that is how a person reads a laser line; any other header expands
    into one row per entry."""
    rows: List[List[str]] = []
    for key, val in tree:
        label = f"{prefix}{key}"
        if isinstance(val, list):
            if (val and all(not isinstance(v, list) for _k, v in val)
                    and any(k in _DEVICE_FLAGS and v == "" for k, v in val)):
                parts = [k if v == "" else f"{k} {v}" for k, v in val]
                rows.append([label, ", ".join(parts)])
            elif val:
                rows.extend(_flatten(val, f"{label} › "))
        elif val == "":
            if prefix:
                rows.append([prefix.rstrip(" ›"), key])
        else:
            rows.append([label, val])
    return rows


def _find(tree: List[Tuple[str, Any]], key: str) -> Optional[List[Tuple[str, Any]]]:
    for k, v in tree:
        if k == key and isinstance(v, list):
            return v
    return None


def _components(tree: List[Tuple[str, Any]]) -> Dict[str, List[Tuple[str, Any]]]:
    """``Component <name>:`` blocks of a capturing tree, by channel name."""
    out: Dict[str, List[Tuple[str, Any]]] = {}
    comps = _find(tree, "Components") or []
    for k, v in list(comps) + list(tree):
        if k.startswith("Component ") and isinstance(v, list):
            out.setdefault(k[len("Component "):].strip(), v)
    return out


def _planes(description: str) -> List[Tuple[str, str]]:
    """``(channel name, plane text)`` per ``Plane #k:`` section of ``text_info["description"]``.
    A single-channel file has no ``Plane #1:`` header — its one plane is the indented block
    that opens with `` Name:`` and runs up to the loop lines."""
    text = str(description or "")
    parts = re.split(r"(?m)^Plane #\d+:\s*$", text)
    blocks = parts[1:] if len(parts) > 1 else [text]
    out = []
    for b in blocks:
        m = re.search(r"(?m)^\s*Name:\s*(.+?)\s*$", b)
        if m:
            out.append((m.group(1), b))
    return out


def _plane_microscope(plane_text: str) -> List[List[str]]:
    """The ``Microscope Settings:`` lines of one plane (turret, light path, LED intensity…)."""
    m = re.search(r"(?ms)Microscope Settings:(.*?)(?=^\S|\Z)", plane_text)
    if not m:
        return []
    rows = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ": " in ln:
            k, v = ln.split(": ", 1)
            rows.append([k.strip(), v.strip()])
        else:
            rows.append(["Setting", ln])
    return rows


# ── the record ───────────────────────────────────────────────────────────────────

def _fmt(v: Any, unit: str = "", nd: int = 4) -> str:
    x = _num(v)
    if x is None:
        return "" if v is None else str(v)
    s = f"{x:.{nd}g}"
    return f"{s} {unit}".strip()


def _summary(vals: Sequence[Any], unit: str = "") -> str:
    """A recorded-data column → ``"50 %"`` when constant, ``"7636 – 7928 (varies)"`` when
    not. Non-finite entries (a frame never acquired) are left out."""
    xs = [x for x in (_num(v) for v in vals) if x is not None]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    u = f" {unit}" if unit else ""
    if lo == hi:
        return f"{lo:.6g}{u}"
    return f"{lo:.6g} – {hi:.6g}{u} (varies over {len(xs)} frames)"


def _split_unit(col: str) -> Tuple[str, str]:
    m = re.match(r"^(.*?)\s*\[(.*)\]\s*$", col)
    return (m.group(1), m.group(2)) if m else (col, "")


def _loop_rows(f: Any) -> List[List[str]]:
    rows = []
    for lp in _safe(lambda: list(f.experiment), default=[]) or []:
        typ = str(_safe(lambda: lp.type) or "")
        n = _safe(lambda: int(lp.count))
        p = _safe(lambda: lp.parameters)
        if "Time" in typ:
            per = _num(_safe(lambda: p.periodMs))
            avg = _num(_safe(lambda: p.periodDiff.avg))
            dur = _num(_safe(lambda: p.durationMs))
            bits = [f"{n} frames"]
            if per:
                bits.append(f"period {per / 1000:.6g} s")
            if avg:
                bits.append(f"measured mean {avg / 1000:.6g} s")
            if dur and dur > 0:
                bits.append(f"duration {dur / 1000:.6g} s")
            rows.append(["Time loop", ", ".join(bits)])
        elif "XYPos" in typ:
            rows.append(["XY positions", f"{n} points"
                         + (", per-point Z" if _safe(lambda: p.isSettingZ) else "")])
        elif "ZStack" in typ:
            bits = [f"{n} planes"]
            step = _num(_safe(lambda: p.stepUm))
            if step:
                bits.append(f"step {step:.6g} µm")
            home = _safe(lambda: p.homeIndex)
            if home is not None:
                bits.append(f"home plane {home}")
            b2t = _safe(lambda: p.bottomToTop)
            if b2t is not None:
                bits.append("bottom → top" if b2t else "top → bottom")
            dev = _safe(lambda: p.deviceName)
            if dev:
                bits.append(f"device {dev}")
            rows.append(["Z stack", ", ".join(bits)])
        elif typ:
            rows.append([typ, f"{n}"])
    return rows


def acquisition_record(f: Any, *, channel_names: Sequence[str],
                       frame_jd_first: Optional[float], frame_jd_last: Optional[float],
                       utc_offset_s: Optional[int], utc_offset_source: str,
                       stamp: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    """Every acquisition setting the file carries, ready to show.

    Returns ``{"acquisition_info": [[section, [[label, value], …]], …],
    "channel_acquisition": [[[label, value], …] per channel],
    "acquisition_text": {block: text}}``. Missing fields are simply absent — a row is only
    written for something the file actually recorded."""
    names = [str(n) for n in channel_names]
    ti = _safe(lambda: dict(f.text_info), default={}) or {}
    meta = _safe(lambda: f.metadata)
    chans = _safe(lambda: list(meta.channels), default=[]) or []
    events = _safe(lambda: f.events(orient="list"), default={}) or {}
    custom = _safe(lambda: f.custom_data, default={}) or {}
    capturing = parse_settings_text(ti.get("capturing", ""))
    comps = _components(capturing)
    planes = dict(_planes(ti.get("description", "")))

    sections: List[List[Any]] = []

    # ── file ──
    rows: List[List[str]] = []
    app = custom.get("AppInfo_V1_0") or {}
    sw = " ".join(str(app.get(k) or "") for k in ("SWNameString", "VersionString")).strip()
    if sw:
        rows.append(["Software", sw])
    if app.get("GrabberString"):
        rows.append(["Acquisition system", str(app["GrabberString"])])
    sizes = _safe(lambda: dict(f.sizes), default={}) or {}
    if sizes:
        rows.append(["Dimensions", " × ".join(f"{k}={v}" for k, v in sizes.items())])
    attrs = _safe(lambda: f.attributes)
    if attrs is not None:
        bits = _safe(lambda: attrs.bitsPerComponentSignificant)
        rows.append(["Bit depth", f"{bits}-bit significant in {f.dtype}"])
        comp = _safe(lambda: attrs.compressionType)
        if comp:
            rows.append(["Compression", f"{comp} (level {_safe(lambda: attrs.compressionLevel)})"])
    if ti.get("date"):
        rows.append(["File date (local, as written)", " ".join(str(ti["date"]).split())])
    sections.append(["File", rows])

    # ── time ──
    rows = []
    if frame_jd_first is not None and utc_offset_s is not None:
        tz = offset_text(utc_offset_s)
        loc0 = jd_to_utc(frame_jd_first) + timedelta(seconds=utc_offset_s)
        rows.append(["First frame", f"{loc0:%Y-%m-%d %H:%M:%S} ({tz})"])
        if frame_jd_last is not None:
            loc1 = jd_to_utc(frame_jd_last) + timedelta(seconds=utc_offset_s)
            rows.append(["Last frame", f"{loc1:%Y-%m-%d %H:%M:%S} ({tz})"])
            span = (frame_jd_last - frame_jd_first) * 86400.0
            rows.append(["Acquisition span", _duration(span)])
        rows.append(["Time zone", f"{tz} — from "
                     + ("the file's own local date" if utc_offset_source == "file"
                        else "this computer (the file's date could not be matched)")])
    if stamp is not None and stamp[1] > 0 and utc_offset_s is not None:
        start = jd_to_utc(stamp[0] - stamp[1] / MS_PER_DAY) + timedelta(seconds=utc_offset_s)
        rows.append(["Run started", f"{start:%Y-%m-%d %H:%M:%S}  (this file's first frame came "
                     f"{_duration(stamp[1] / 1000.0)} later — a multi-file run)"])
    sections.append(["Time", rows])

    # ── optics ──
    rows = []
    mic = _safe(lambda: chans[0].microscope) if chans else None
    if ti.get("optics"):
        rows.append(["Objective", str(ti["optics"])])
    elif mic is not None and _safe(lambda: mic.objectiveName):
        rows.append(["Objective", str(mic.objectiveName)])
    if mic is not None:
        for label, attr, unit in (("Magnification", "objectiveMagnification", "×"),
                                  ("Numerical aperture", "objectiveNumericalAperture", ""),
                                  ("Immersion refractive index", "immersionRefractiveIndex", ""),
                                  ("Zoom", "zoomMagnification", "×"),
                                  ("Projective magnification", "projectiveMagnification", "×"),
                                  ("Pinhole", "pinholeDiameterUm", "µm")):
            v = _num(_safe(lambda a=attr: getattr(mic, a)))
            if v is not None:
                rows.append([label, f"{v:.6g} {unit}".strip()])
    vol = _safe(lambda: chans[0].volume) if chans else None
    cal = _safe(lambda: tuple(vol.axesCalibration)) if vol is not None else None
    if cal:
        rows.append(["Pixel size", f"{cal[0]:.6g} × {cal[1]:.6g} µm"])
        if len(cal) > 2 and sizes.get("Z", 1) > 1:
            rows.append(["Z step", f"{cal[2]:.6g} µm"])
    desc = str(ti.get("description", ""))
    for label, pat in (("Camera / detector", r"(?m)^Camera Name:\s*(.+)$"),
                       ("Relay lens", r"(?m)^\s*Relay Lens Magnification:\s*(.+)$")):
        m = re.search(pat, desc)
        if m:
            rows.append([label, m.group(1).strip()])
    sections.append(["Optics", rows])

    # ── detector / scan configuration (shared by all channels) ──
    rows = []
    for head in ("Configuration", "Scan Area", "Misc"):
        sub = _find(capturing, head)
        if sub:
            rows.extend(_flatten(sub, f"{head} › "))
    if capturing and capturing[0][1] == "" and not rows:
        rows.append(["Device", capturing[0][0]])
    for k, v in capturing:
        if not isinstance(v, list) and v != "":
            rows.append([k, v])
    if rows:
        sections.append(["Acquisition settings", rows])

    # ── microscope (the first plane's; per-plane differences go to the channel rows) ──
    first_plane = next(iter(planes.values()), "")
    rows = _plane_microscope(first_plane)
    if rows:
        sections.append(["Microscope", rows])

    # ── loops ──
    rows = _loop_rows(f)
    if rows:
        sections.append(["Experiment loops", rows])

    # ── recorded per-frame data that is not a channel's ──
    rows = []
    for col, vals in events.items():
        if col in _EVENT_INDEX_COLS or col == "Z-Series":
            continue
        name, unit = _split_unit(col)
        if any(name.startswith(f"{n} - ") for n in names):
            continue
        s = _summary(vals, unit)
        if s:
            rows.append([name, s])
    if rows:
        sections.append(["Recorded data (per frame)", rows])

    # ── per channel ──
    per_ch: List[List[List[str]]] = []
    for i, nm in enumerate(names):
        rows = []
        ch = chans[i] if i < len(chans) else None
        inner = _safe(lambda: ch.channel) if ch is not None else None
        ex = _num(_safe(lambda: inner.excitationLambdaNm))
        em = _num(_safe(lambda: inner.emissionLambdaNm))
        if ex:
            rows.append(["Excitation", f"{ex:.6g} nm"])
        if em:
            rows.append(["Emission", f"{em:.6g} nm"])
        flags = _safe(lambda: list(ch.microscope.modalityFlags), default=[]) if ch else []
        if flags:
            rows.append(["Modality", ", ".join(str(x) for x in flags)])
        comp = comps.get(nm) or (next(iter(comps.values())) if len(comps) == 1 else None)
        if comp:
            rows.extend(_flatten(comp))
        for col, vals in events.items():
            if col.startswith(f"{nm} - "):
                label, unit = _split_unit(col[len(nm) + 3:])
                s = _summary(vals, unit)
                if s:
                    rows.append([f"{label} (recorded)", s])
        pin = _num(_safe(lambda: ch.microscope.pinholeDiameterUm)) if ch else None
        if pin and not any(r[0].startswith("Pinhole") for r in rows):
            rows.append(["Pinhole", f"{pin:.4g} µm"])
        # A plane's microscope state (turret, LED intensity…) is listed under the channel
        # only where it DIFFERS from the first plane's, which the file section already shows
        # — on a confocal run every plane repeats the same lines.
        plane_rows = _plane_microscope(planes.get(nm, ""))
        if plane_rows and plane_rows != _plane_microscope(first_plane):
            rows.extend([[f"Microscope › {k}", v] for k, v in plane_rows])
        color = _safe(lambda: inner.color)
        if color is not None:
            rgb = _safe(lambda: (int(color.r), int(color.g), int(color.b)))
            if rgb:
                rows.append(["Display colour", "#%02x%02x%02x" % rgb])
        per_ch.append(rows)

    text = {k: str(v) for k, v in ti.items() if str(v or "").strip()}
    return {"acquisition_info": [s for s in sections if s[1]],
            "channel_acquisition": per_ch,
            "acquisition_text": text}


def _duration(s: float) -> str:
    s = float(s)
    if abs(s) < 120:
        return f"{s:.1f} s"
    if abs(s) < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def channel_colors_rgb(f: Any) -> List[Optional[Tuple[int, int, int]]]:
    """Each channel's NIS display colour as ``(r, g, b)`` — what ``colorRGB`` used to be, on
    the SDK that replaced it with a ``Color`` object."""
    meta = _safe(lambda: f.metadata)
    out: List[Optional[Tuple[int, int, int]]] = []
    for ch in _safe(lambda: list(meta.channels), default=[]) or []:
        c = _safe(lambda: ch.channel.color)
        out.append(_safe(lambda: (int(c.r), int(c.g), int(c.b))) if c is not None else None)
    return out


__all__ = ["frame0_stamp", "frame_jd", "jd_to_utc", "utc_offset", "offset_text",
           "local_offset_s", "parse_nis_date", "parse_settings_text", "acquisition_record",
           "channel_colors_rgb", "MS_PER_DAY", "JD_UNIX_EPOCH"]
