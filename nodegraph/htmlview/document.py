"""The view document: what a standalone HTML viewer of a Dataset carries (2026-11-05).

:func:`build_view` turns a pulled :class:`~nodegraph.dataset.Dataset` — plus, when the
caller is the Viewer or the figure editor, the display state it holds — into one JSON-able
dict: the exported planes of every shown channel at every frame (packed PNGs, see
:mod:`nodegraph.htmlview.planes`), the channels with their names, colours, windows and
gammas, the frame times, the positions, and the overlays as DATA — the Point rows, the label
rasters, the trajectories, a scalar wash, a field raster, the vectors — with the look the
Viewer would draw them in. The JavaScript in :mod:`nodegraph.htmlview.page` reads nothing
else, so what the page shows is decided here, where the Dataset is, and the browser only
draws.

What the viewer reproduces of the Viewer, and from where:

* the composite is the Viewer's CPU path (:func:`nodelab_v2.viewer.composite_with_clim`):
  window, gamma, emission colour, additive — ported line for line in the JavaScript;
* Auto is the Viewer's ``_auto_clim`` (1 / 99.5 percentiles on a subsample), the flat-plane
  rule included;
* channel colours are the emission colours Export Movie uses (``movie_draw._emission_rgb``,
  the Qt-free twin of ``theme.emission_qcolor``) unless the Viewer handed over a pick;
* the identity palette, the label raster the Labels overlay draws and the scalar wash come
  from :mod:`nodegraph.catalog._shared.identity_palette`, the same code the Viewer runs;
* a field raster and the vectors come from :mod:`nodegraph.fieldmap`, as in the Viewer.

Budgets: every plane is read at ``max_px`` (the Viewer's display budget by default) and
the embedded bytes are counted as they are made; past ``budget_mb`` the build refuses
with the numbers, rather than writing a page no browser opens.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared import identity_palette as IP
from nodegraph.catalog._shared.label_paint import slot_lut
from nodegraph.catalog._shared.movie_draw import _emission_rgb
from nodegraph.catalog._shared.overlay_math import COLORMAPS, colormap_lut
from nodegraph.domains import Domain
from nodegraph.htmlview.planes import (DISPLAY_MAX_PX, dtype_range, plane_record,
                                       read_plane, record_bytes)

FORMAT = "nd2studios-htmlview/1"

#: the embedded bytes a build may reach before it refuses (the base64 of every plane)
DEFAULT_BUDGET_MB = 250.0
#: past this many Point rows the points overlay is left out (a marker each would bury the
#: page and the JSON); the page says so
MAX_POINT_ROWS = 400_000
#: past this many track vertices the tracks overlay is left out
MAX_TRACK_VERTICES = 2_000_000

#: the Viewer's overlay defaults, the fields the page's controls expose — kept in step with
#: :mod:`nodelab_v2.overlays` by name (``test_htmlview`` checks the names exist there)
LOOK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "points": {"enabled": True, "shape": "star", "spread": 3, "unit_px": 3.0,
               "color_mode": "single", "color": "#ffc83c", "sat": 205, "val": 255,
               "opacity": 100, "z_project": False, "off_opacity": 30},
    "labels": {"enabled": True, "style": "both", "width": 2.0, "color_mode": "per_label",
               "color": "#e08a3a", "sat": 205, "val": 255, "opacity": 100,
               "fill_opacity": 30, "show_ids": False, "id_px": 11},
    "tracks": {"enabled": True, "width": 1.8, "color_mode": "per_track", "color": "#c264a0",
               "sat": 205, "val": 255, "opacity": 100, "vertex_px": 1.8, "head_px": 5.0,
               "trail": "all", "window": 5, "fade": False},
    "voxels": {"enabled": False, "style": "heatmap", "colormap": "viridis", "opacity": 45,
               "alpha_mode": "ramp", "clim_auto": True, "clim_lo": 0.0, "clim_hi": 1.0,
               "center_zero": False},
    "field": {"enabled": True, "style": "smooth", "colormap": "auto", "limits": "series",
              "clim_lo": 0.0, "clim_hi": 1.0, "center_zero": True, "opacity": 55,
              "colorbar": True},
    "vectors": {"enabled": True, "every": 4, "auto_scale": True, "scale": 1.0, "gate": 0.0,
                "color_mode": "magnitude", "colormap": "turbo", "color": "#6fe3ff",
                "width": 1.3, "head_px": 5.0, "opacity": 90},
}

#: the Viewer's additive channel presets (``nodelab_v2.viewer._COLOR_PRESETS``)
COLOR_PRESETS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("Red", (255, 40, 40)), ("Green", (60, 255, 80)), ("Blue", (70, 120, 255)),
    ("Cyan", (60, 230, 240)), ("Magenta", (255, 70, 220)), ("Yellow", (255, 220, 60)),
    ("Orange", (255, 150, 40)), ("Grey", (210, 214, 222)),
)


# ── small helpers ─────────────────────────────────────────────────────────────────────
def parse_frames(spec: Any, n: int) -> List[int]:
    """Frame indices from ``spec`` over an axis of ``n``: blank / ``None`` → every frame;
    ``7`` → one; ``0-99`` → a span; ``0-99:3`` → every third of it; ``2,5,9`` → a list;
    ``last`` → the last. Out-of-range indices are refused by name."""
    if n <= 0:
        return []
    if spec is None:
        return list(range(n))
    if isinstance(spec, (list, tuple)):
        out = [int(v) for v in spec]
    else:
        s = str(spec).strip().lower()
        if not s or s == "all":
            return list(range(n))
        if s == "last":
            return [n - 1]
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            step = 1
            if ":" in part:
                part, st = part.split(":", 1)
                step = max(1, int(st))
            if "-" in part:
                a, b = part.split("-", 1)
                a_i = int(a) if a.strip() else 0
                b_i = int(b) if b.strip() else n - 1
                out.extend(range(a_i, b_i + 1, step))
            else:
                out.append(int(part))
    bad = [v for v in out if not 0 <= v < n]
    if bad:
        raise ValueError(f"frame {bad[0]} is outside 0..{n - 1}")
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def parse_channels(spec: Any, n: int) -> List[int]:
    """Channel indices from ``spec`` (the frames grammar; blank → all)."""
    return parse_frames(spec, n)


def _hex(rgb: Sequence[int]) -> str:
    r, g, b = (max(0, min(255, int(v))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt_clock(md: Mapping[str, Any], n_t: int) -> Optional[List[str]]:
    fd = md.get("frame_datetime")
    if isinstance(fd, (list, tuple)) and len(fd) == n_t:
        return [str(v or "") for v in fd]
    jd = md.get("frame_time_jd")
    if isinstance(jd, (list, tuple)) and len(jd) == n_t:
        try:
            from nodegraph.placement import jd_to_datetime_text, utc_offset_of
            off = utc_offset_of(md)
            return [jd_to_datetime_text(float(v), off) for v in jd]
        except Exception:                      # noqa: BLE001 — a clock is advisory
            return None
    return None


def auto_window(plane: np.ndarray) -> Tuple[float, float]:
    """The Viewer's ``_auto_clim``: 1 / 99.5 percentiles on a cheap subsample, flat-plane
    safe (:func:`nodelab_v2.viewer._flat_window`'s rule)."""
    a = np.asarray(plane)
    step = max(1, int(np.sqrt(max(1, a.size) / 200_000)))
    s = a[::step, ::step] if a.ndim == 2 else a
    s = np.asarray(s, dtype=float)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi > lo:
        return lo, hi
    return (0.0, lo) if lo > 0 else (lo, lo + 1.0)


def _slots_for(lut: Optional[np.ndarray], ids: np.ndarray) -> np.ndarray:
    """The palette slot of every id (``slot_of`` vectorised)."""
    ids = np.asarray(ids).astype(np.int64)
    if lut is None or len(lut) == 0:
        return ids
    inside = (ids >= 0) & (ids < len(lut))
    out = ids.copy()
    out[inside] = np.asarray(lut)[ids[inside]]
    return out


def _z_index(z: Any, nz: int) -> int:
    return 0 if z in ("max", "mean") else min(max(int(z), 0), max(0, nz - 1))


def _z_key(z: Any) -> str:
    return str(z) if z in ("max", "mean") else str(int(z))


class BudgetExceeded(ValueError):
    """The embedded bytes passed the budget."""


# ── the build ─────────────────────────────────────────────────────────────────────────
def build_view(ds: Any, *, title: str = "", max_px: int = DISPLAY_MAX_PX,
               channels: Any = None, t_frames: Any = None, m_frames: Any = None,
               z: Any = "all", overlays: bool = True,
               state: Optional[Mapping[str, Any]] = None,
               budget_mb: float = DEFAULT_BUDGET_MB, scalebar: Any = True,
               timestamp: Any = "elapsed", stage_ok: bool = False,
               provenance: Optional[Mapping[str, Any]] = None,
               own_layers: Optional[Sequence[str]] = None,
               window: Optional[Tuple[float, float, float, float]] = None,
               progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """The view document of ``ds``.

    ``channels`` / ``t_frames`` / ``m_frames`` take the frames grammar (blank = all);
    ``z`` is ``all``, an index, a list, or ``max`` / ``mean`` (a projection, one plane per
    frame). ``state`` is the Viewer's: ``{"channels": {c: {"on", "lo", "hi", "gamma",
    "rgb"}}, "overlays": {tab: {...look fields...}}, "coords": (m, t, z, c)}`` — every key
    optional. ``window`` is the fraction ``(fy0, fy1, fx0, fx1)`` of the plane to export
    (a figure's popout); ``scalebar`` is ``True`` / ``False`` or the Viewer's
    ``{"um", "color"}``; ``timestamp`` a mode (``elapsed`` / ``clock`` / ``frame`` /
    ``elapsed_clock`` / ``off``) or the Viewer's ``{"mode", "color"}``."""
    state = dict(state or {})
    ax = ds.axes
    M, T, Z, C = int(ax.m), int(ax.t), int(ax.z), int(ax.c)
    Y, X = int(ax.y), int(ax.x)
    md = dict(getattr(ds, "metadata", None) or {})
    c_list = parse_channels(channels, C)
    if not c_list:
        raise ValueError("no channel to export")
    m_list = parse_frames(m_frames, M)
    t_list = parse_frames(t_frames, T)
    if z in ("max", "mean"):
        z_list: List[Any] = [z] if Z > 1 else [0]
    elif z in (None, "all"):
        z_list = list(range(Z))
    elif isinstance(z, (list, tuple)):
        z_list = parse_frames(list(z), Z)
    else:
        z_list = [min(max(int(z), 0), Z - 1)]
    budget = float(budget_mb) * 1e6
    n_total = len(m_list) * len(t_list) * len(z_list) * len(c_list)
    tell = progress or (lambda i, n, s: None)

    # ── planes ──────────────────────────────────────────────────────────────────
    frames: Dict[str, Dict[str, Any]] = {}
    used = 0
    first_plane: Dict[int, np.ndarray] = {}
    shape: Optional[Tuple[int, int]] = None
    level_w = X
    stride = 1
    crop = (0, 0)
    done = 0
    for m in m_list:
        for t in t_list:
            for zz in z_list:
                for c in c_list:
                    plane, k = read_plane(ds, m, t, zz, c, max_px=max_px)
                    stride = k
                    if shape is None:
                        level_w = plane.shape[1] * k
                    if window is not None:
                        fy0, fy1, fx0, fx1 = window
                        H, W = plane.shape
                        y0, y1 = max(0, int(math.floor(fy0 * H))), min(H, int(math.ceil(fy1 * H)))
                        x0, x1 = max(0, int(math.floor(fx0 * W))), min(W, int(math.ceil(fx1 * W)))
                        y1, x1 = max(y1, y0 + 1), max(x1, x0 + 1)
                        plane = np.ascontiguousarray(plane[y0:y1, x0:x1])
                        crop = (y0, x0)
                    if shape is None:
                        shape = (int(plane.shape[0]), int(plane.shape[1]))
                    if c not in first_plane:
                        first_plane[c] = plane
                    rec = plane_record(plane)
                    used += record_bytes(rec)
                    if used > budget:
                        raise BudgetExceeded(
                            f"the page would embed more than {budget_mb:g} MB "
                            f"({done + 1} of {n_total} planes read, {shape[1]}×{shape[0]} px "
                            f"each) — pick fewer frames or channels, one Z plane or a "
                            f"projection, a smaller max_px, or raise the budget")
                    frames[f"{m},{t},{_z_key(zz)},{c}"] = rec
                    done += 1
                    if done % 8 == 0 or done == n_total:
                        tell(done, n_total, f"plane {done}/{n_total}")
    if shape is None:
        raise ValueError("nothing to export: no frame selected")
    h, w = shape

    # ── channels ────────────────────────────────────────────────────────────────
    names = list(md.get("channel_names") or [])
    emis = list(md.get("channel_emission_nm") or [])
    chan_state = dict(state.get("channels") or {})
    on_set = state.get("active")
    chans: List[Dict[str, Any]] = []
    for c in c_list:
        st = dict(chan_state.get(c) or chan_state.get(str(c)) or {})
        name = str(names[c]) if c < len(names) and names[c] else f"Ch{c + 1}"
        rgb = st.get("rgb") or _emission_rgb(emis[c] if c < len(emis) else None)
        plane = first_plane[c]
        if st.get("lo") is not None and st.get("hi") is not None:
            lo, hi, auto = float(st["lo"]), float(st["hi"]), False
        else:
            lo, hi = auto_window(plane)
            auto = True
        rmin, rmax = dtype_range(ds, plane)
        rmin, rmax = min(rmin, lo), max(rmax, hi)
        chans.append({"c": c, "name": name, "rgb": [int(v) for v in rgb], "lo": lo, "hi": hi,
                      "gamma": float(st.get("gamma", 1.0) or 1.0), "auto": auto,
                      "on": (bool(st["on"]) if "on" in st else
                             (c in set(on_set) if on_set is not None else True)),
                      "range": [rmin, rmax],
                      "emission_nm": (emis[c] if c < len(emis) else None)})

    # ── calibration, time, positions ────────────────────────────────────────────
    try:
        px_um = float(md.get("pixel_size_um") or 0.0) or None
    except (TypeError, ValueError):
        px_um = None
    sx = level_w / max(1, X) / max(1, stride)             # exported px per full-res px
    sy = sx
    plane_um = (px_um / sx) if px_um else None
    from nodegraph.metadata import frame_times_s
    secs = frame_times_s(md, T)
    try:
        dt_s = float(md.get("dt_s") or 0.0) or None
    except (TypeError, ValueError):
        dt_s = None
    pos_names = [str(v) for v in (md.get("position_name") or [])]
    stage = md.get("stage_xy_um") or []
    try:
        stage = [[float(v[0]), float(v[1])] for v in stage]
    except (TypeError, ValueError, IndexError):
        stage = []
    try:
        z_step = float(md.get("z_step_um") or 0.0) or None
    except (TypeError, ValueError):
        z_step = None

    # ── overlays ────────────────────────────────────────────────────────────────
    ov_state = dict(state.get("overlays") or {})
    looks = {tab: dict(LOOK_DEFAULTS[tab], **{k: v for k, v in dict(ov_state.get(tab) or {}).items()
                                              if k in LOOK_DEFAULTS[tab]})
             for tab in LOOK_DEFAULTS}
    ov: Dict[str, Any] = {"look": looks, "unsupported": [], "notes": []}
    if overlays:
        try:
            _add_overlays(ds, ov, looks, m_list, t_list, z_list, c_list[0], (h, w), sx, sy,
                          crop, stride, max_px, own_layers, ov_state, used, budget, tell)
        except BudgetExceeded:
            raise
        except Exception as exc:              # noqa: BLE001 — an overlay is never worth the page
            ov["notes"].append(f"overlays left out: {type(exc).__name__}: {exc}")

    # ── decor ───────────────────────────────────────────────────────────────────
    if isinstance(scalebar, Mapping):
        sb = {"show": True, "um": float(scalebar.get("um") or 0.0),
              "color": str(scalebar.get("color") or "white")}
    else:
        sb = {"show": bool(scalebar), "um": 0.0, "color": "white"}
    if isinstance(timestamp, Mapping):
        ts = {"mode": str(timestamp.get("mode") or "elapsed"),
              "color": str(timestamp.get("color") or "white")}
    else:
        ts = {"mode": str(timestamp or "off"), "color": "white"}
    coords = state.get("coords")
    start = {"m": m_list[0], "t": t_list[0], "z": z_list[0], "c": c_list[0]}
    if coords:
        try:
            cm, ct, cz, _cc = (int(v) for v in coords)
            start = {"m": cm if cm in m_list else m_list[0], "t": ct if ct in t_list else t_list[0],
                     "z": cz if cz in z_list else z_list[0], "c": c_list[0]}
        except (TypeError, ValueError):
            pass

    doc: Dict[str, Any] = {
        "format": FORMAT, "title": str(title or md.get("source_name") or "ND2Studios view"),
        "written": _dt.datetime.now().isoformat(timespec="seconds"),
        "axes": {"m": M, "t": T, "z": Z, "c": C, "y": Y, "x": X},
        "shown": {"m": m_list, "t": t_list, "z": [_z_key(v) for v in z_list], "c": c_list},
        "start": {"m": start["m"], "t": start["t"], "z": _z_key(start["z"])},
        "plane": {"h": h, "w": w, "sx": sx, "sy": sy, "x0": int(crop[1]), "y0": int(crop[0]),
                  "stride": int(stride)},
        "pixel_size_um": px_um, "plane_um": plane_um, "z_step_um": z_step,
        "channels": chans, "frames": frames,
        "times": {"seconds": secs, "clock": _fmt_clock(md, T), "dt_s": dt_s},
        "positions": {"names": pos_names, "stage_xy_um": stage},
        "stage_ok": bool(stage_ok and stage and window is None),
        "overlays": ov, "scalebar": sb, "timestamp": ts,
        "cmaps": {name: colormap_lut(name).tolist() for name in COLORMAPS},
        "presets": [{"name": n, "rgb": list(rgb)} for n, rgb in COLOR_PRESETS],
        "bytes": used,
        "provenance": dict(provenance or {}),
    }
    return doc


# ── overlays ──────────────────────────────────────────────────────────────────────────
def _add_overlays(ds: Any, ov: Dict[str, Any], looks: Mapping[str, Mapping[str, Any]],
                  m_list: Sequence[int], t_list: Sequence[int], z_list: Sequence[Any],
                  c0: int, shape: Tuple[int, int], sx: float, sy: float,
                  crop: Tuple[int, int], stride: int, max_px: int,
                  own_layers: Optional[Sequence[str]], ov_state: Mapping[str, Any],
                  used: int, budget: float, tell: Callable[[int, int, str], None]) -> None:
    attrs = ds.attributes if hasattr(ds, "attributes") else {}
    if not attrs:
        return
    h, w = shape
    palette = IP.build_palette(ds)
    nz = int(ds.axes.z)

    # points — every row of every Point layer the Viewer would draw; the page filters by frame
    by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
    for (dom, layer, name), attr in attrs.items():
        if dom is Domain.POINT:
            by_layer.setdefault(layer, {})[name] = attr.values
    want = str(dict(ov_state.get("points") or {}).get("layer") or "").strip()
    if want and want in {str(k) for k in by_layer}:
        by_layer = {k: v for k, v in by_layer.items() if str(k) == want}
    else:
        by_layer = IP.drop_synth(ds, by_layer, Domain.POINT)
        grids = (getattr(ds, "metadata", None) or {}).get("__analysis_mesh__") or {}
        by_layer = {k: v for k, v in by_layer.items() if str(k) not in grids}
    cols_out = {k: [] for k in ("m", "t", "z", "x", "y", "id", "layer", "slot")}
    layer_names: List[str] = []
    z_kinds: List[str] = []
    total_rows = 0
    for li, layer in enumerate(sorted(by_layer, key=lambda v: str(v))):
        cols = by_layer[layer]
        if "y" not in cols or "x" not in cols:
            continue
        n = min(len(np.asarray(cols[k])) for k in cols)
        if n == 0:
            continue
        layer_names.append(str(layer))
        try:
            z_kinds.append(str(ds.structure_zkind(Domain.POINT, layer) or "plane_index"))
        except Exception:                      # noqa: BLE001
            z_kinds.append("plane_index")
        ys = np.asarray(cols["y"], dtype=float)[:n]
        xs = np.asarray(cols["x"], dtype=float)[:n]
        zs = np.asarray(cols["z"], dtype=float)[:n] if "z" in cols else np.zeros(n)
        ms = np.asarray(cols["m"]).astype(np.int64)[:n] if "m" in cols else np.zeros(n, np.int64)
        ts = np.asarray(cols["t"]).astype(np.int64)[:n] if "t" in cols else np.zeros(n, np.int64)
        ids = (np.asarray(cols["id"]).astype(np.int64)[:n] if "id" in cols
               else np.arange(1, n + 1, dtype=np.int64))
        keep = np.isin(ms, list(m_list)) & np.isin(ts, list(t_list))
        total_rows += int(keep.sum())
        if total_rows > MAX_POINT_ROWS:
            ov["notes"].append(f"points left out: more than {MAX_POINT_ROWS:,} rows")
            cols_out = {k: [] for k in cols_out}
            layer_names = []
            break
        slots = _slots_for(palette.points.get((Domain.POINT, layer)), ids)
        cols_out["m"].extend(ms[keep].tolist())
        cols_out["t"].extend(ts[keep].tolist())
        cols_out["z"].extend(np.round(zs[keep], 2).tolist())
        cols_out["x"].extend(np.round(xs[keep], 2).tolist())
        cols_out["y"].extend(np.round(ys[keep], 2).tolist())
        cols_out["id"].extend(ids[keep].tolist())
        cols_out["layer"].extend([li] * int(keep.sum()))
        cols_out["slot"].extend(slots[keep].tolist())
    if layer_names:
        ov["points"] = dict(cols_out, layers=layer_names, z_kind=z_kinds)

    # labels — the raster the Labels overlay draws, one plane per shown frame
    want = str(dict(ov_state.get("labels") or {}).get("layer") or "").strip()
    src = IP.label_source(ds, (m_list[0], t_list[0], _z_index(z_list[0], nz), c0), want,
                          own_layers)
    if src is not None:
        name = IP.label_source_name(ds, src)
        lz = int(src.shape[2])
        lc = min(c0, max(0, int(src.shape[3]) - 1))
        lab_frames: Dict[str, Dict[str, Any]] = {}
        skipped = False
        for m in m_list:
            for t in t_list:
                for zz in z_list:
                    if zz in ("max", "mean") and lz > 1:
                        skipped = True
                        continue
                    zi = 0 if lz == 1 else _z_index(zz, lz)
                    try:
                        plane = src[m, t, zi, lc]
                    except IndexError:
                        continue
                    plane = _match_plane(plane, ds, shape, sx, crop)
                    if plane is None:
                        continue
                    if int(plane.max()) > 65535:
                        ov["notes"].append(f"label ids above 65535 in {name!r} are clamped")
                        plane = np.minimum(plane, 65535)
                    rec = plane_record(plane.astype(np.uint16))
                    used += record_bytes(rec)
                    if used > budget:
                        raise BudgetExceeded(
                            f"the page would embed more than {budget / 1e6:g} MB once the "
                            f"label rasters are added — fewer frames, or turn overlays off")
                    lab_frames[f"{m},{t},{_z_key(zz)}"] = rec
        keys = IP.label_keys(src[m_list[0], t_list[0], 0, lc], palette)
        if lab_frames:
            ov["labels"] = {"layer": name, "frames": lab_frames,
                            "keys": (keys.tolist() if keys is not None else None)}
        if skipped:
            ov["notes"].append("labels are not drawn on a Z projection of a 3-D raster")

    # tracks — every trajectory, per position; the page draws the trail per frame
    tl = IP.track_layers(ds)
    if tl:
        members = IP.member_layers(ds)
        tracks: List[Dict[str, Any]] = []
        verts = 0
        tname = ""
        for tlayer, cols in tl.items():
            if not {"track_id", "t", "member_id"} <= set(cols):
                continue
            tname = tname or str(tlayer)
            tid = np.asarray(cols["track_id"]).astype(np.int64)
            tt = np.asarray(cols["t"]).astype(np.int64)
            mem = np.asarray(cols["member_id"]).astype(np.int64)
            mkey = IP.best_member_layer(members, mem)
            if mkey is None:
                continue
            bc = members[mkey]
            lut = palette.tracks.get(tlayer)
            ids = np.asarray(bc["id"]).astype(np.int64)
            ys = np.asarray(bc["y"], dtype=float)
            xs = np.asarray(bc["x"], dtype=float)
            ms = np.asarray(bc["m"]).astype(np.int64) if "m" in bc else None
            id_to_i = {int(v): i for i, v in enumerate(ids.tolist())}
            for track in np.unique(tid).tolist():
                sel = tid == track
                order = np.argsort(tt[sel], kind="stable")
                ts_ord = tt[sel][order].tolist()
                mem_ord = mem[sel][order].tolist()
                by_m: Dict[int, Dict[str, list]] = {}
                for tv, mv in zip(ts_ord, mem_ord):
                    i = id_to_i.get(int(mv))
                    if i is None:
                        continue
                    mm_ = int(ms[i]) if ms is not None else 0
                    d = by_m.setdefault(mm_, {"t": [], "x": [], "y": []})
                    d["t"].append(int(tv))
                    d["x"].append(round(float(xs[i]), 2))
                    d["y"].append(round(float(ys[i]), 2))
                for mm_, d in by_m.items():
                    if mm_ not in m_list or not d["t"]:
                        continue
                    verts += len(d["t"])
                    tracks.append({"id": int(track), "m": mm_,
                                   "slot": int(_slots_for(lut, np.array([track]))[0]),
                                   "t": d["t"], "x": d["x"], "y": d["y"]})
            if verts > MAX_TRACK_VERTICES:
                ov["notes"].append(f"tracks left out: more than {MAX_TRACK_VERTICES:,} vertices")
                tracks = []
                break
        if tracks:
            ov["tracks"] = {"layer": tname, "tracks": tracks}

    # a scalar wash (the Voxels tab)
    vox_frames: Dict[str, Dict[str, Any]] = {}
    lo_hi: List[Tuple[float, float]] = []
    for m in m_list:
        for t in t_list:
            for zz in z_list:
                zi = _z_index(zz, nz)
                plane = IP.scalar_plane(ds, (m, t, zi, c0), (h * stride, w * stride))
                if plane is None:
                    continue
                plane = _match_plane(plane, ds, shape, sx, crop)
                if plane is None:
                    continue
                finite = plane[np.isfinite(plane)]
                if finite.size:
                    lo_hi.append((float(np.percentile(finite, 1)), float(np.percentile(finite, 99))))
                rec = plane_record(plane)
                used += record_bytes(rec)
                if used > budget:
                    raise BudgetExceeded(
                        f"the page would embed more than {budget / 1e6:g} MB once the scalar "
                        f"layer is added — fewer frames, or turn overlays off")
                vox_frames[f"{m},{t},{_z_key(zz)}"] = rec
    if vox_frames:
        ov["voxels"] = {"frames": vox_frames,
                        "limits": [min(v[0] for v in lo_hi), max(v[1] for v in lo_hi)]
                        if lo_hi else [0.0, 1.0]}

    # a measured field's column (the Field tab) and the vectors
    from nodegraph import fieldmap as FV
    fl = looks["field"]
    try:
        layer, col = FV.auto_pick(ds, str(fl.get("layer") or ""), str(fl.get("column") or ""))
    except Exception:                          # noqa: BLE001
        layer, col = "", ""
    if col:
        fr_frames: Dict[str, Dict[str, Any]] = {}
        for m in m_list:
            for t in t_list:
                for zz in z_list:
                    zi = _z_index(zz, nz)
                    fr = FV.raster(ds, layer, col, m, t, zi, c0)
                    if fr is None:
                        continue
                    rec = plane_record(np.asarray(fr.grid, dtype=float), nan0=True)
                    y0, y1, x0, x1 = (float(v) for v in fr.extent)
                    rec["extent"] = [y0, y1, x0, x1]
                    rec["lattice"] = bool(fr.lattice)
                    used += record_bytes(rec)
                    fr_frames[f"{m},{t},{_z_key(zz)}"] = rec
        if fr_frames:
            limits_mode = str(fl.get("limits") or "series")
            if limits_mode == "manual":
                lims = (float(fl.get("clim_lo", 0.0)), float(fl.get("clim_hi", 1.0)))
            else:
                lims = tuple(float(v) for v in FV.series_limits(ds, layer, col))
            cmap = str(fl.get("colormap") or "auto")
            if cmap == "auto":
                cmap = "coolwarm" if lims[0] < 0.0 < lims[1] else "turbo"
            unit = FV.column_unit(ds, layer, col)
            ov["field"] = {"layer": layer, "column": col, "frames": fr_frames,
                           "limits": list(lims), "cmap": cmap,
                           "label": f"{col} ({unit})" if unit else col}
    vl = looks["vectors"]
    vec_frames: Dict[str, List[List[float]]] = {}
    mag_hi, auto = 0.0, 0.0
    for m in m_list:
        for t in t_list:
            for zz in z_list:
                zi = _z_index(zz, nz)
                try:
                    vf = FV.vectors(ds, m, t, zi, c0, layer=str(vl.get("layer") or ""))
                except Exception:              # noqa: BLE001
                    vf = None
                if vf is None or not len(vf.rows):
                    continue
                rows = np.asarray(vf.rows, dtype=float)
                rows = rows[np.isfinite(rows).all(axis=1)]
                vec_frames[f"{m},{t},{_z_key(zz)}"] = np.round(rows, 2).tolist()
                mag_hi = max(mag_hi, float(vf.mag_hi))
                if not auto:
                    auto = float(FV.auto_scale(vf))
    if vec_frames:
        ov["vectors"] = {"frames": vec_frames, "mag_hi": mag_hi, "auto": auto or 1.0}

    # what the page does not draw, so nobody wonders
    doms = {dom for (dom, _l, _n) in attrs}
    if Domain.MESH in doms:
        ov["unsupported"].append("mesh")
    if (getattr(ds, "metadata", None) or {}).get("__analysis_mesh__"):
        ov["unsupported"].append("grid")


def _match_plane(plane: np.ndarray, ds: Any, shape: Tuple[int, int], sx: float,
                 crop: Tuple[int, int]) -> Optional[np.ndarray]:
    """A full-resolution structure plane brought to the exported plane's grid: stepped
    by whole pixels to the exported resolution, then cropped like the image."""
    if plane is None or plane.ndim != 2:
        return None
    h, w = shape
    full_y, full_x = int(ds.axes.y), int(ds.axes.x)
    ty = max(1, int(round(plane.shape[0] / max(1.0, full_y * sx))))
    tx = max(1, int(round(plane.shape[1] / max(1.0, full_x * sx))))
    p = plane[::ty, ::tx]
    y0, x0 = crop
    p = p[y0:y0 + h, x0:x0 + w]
    if p.shape[0] < h or p.shape[1] < w:
        pad = np.zeros((h, w), dtype=p.dtype)
        pad[:p.shape[0], :p.shape[1]] = p
        p = pad
    return np.ascontiguousarray(p)


__all__ = ["FORMAT", "DEFAULT_BUDGET_MB", "LOOK_DEFAULTS", "COLOR_PRESETS", "BudgetExceeded",
           "build_view", "parse_frames", "parse_channels", "auto_window"]
