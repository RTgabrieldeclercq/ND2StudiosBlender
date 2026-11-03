"""The figure document — defaults, validation, canonical JSON, digest (V5.00 step 1).

A figure document is a plain JSON-able dict (schema :data:`SCHEMA`) with three
collections:

* ``sources`` — what the page reads: a named Page Output (``"<page id>:<name>"``), with an
  optional item of a several-item Output and an optional part;
* ``items`` — what is on the page, each with a ``kind`` (:data:`ITEM_KINDS`), a
  ``rect_mm`` ``[x, y, w, h]`` on the page, an optional panel ``letter``, and a block named
  after its kind (``image``, ``chart``, …) holding its own fields;
* ``links`` — how items depend on one another (:data:`LINK_KINDS`): a popout's box on its
  parent (``zoom``), twins sharing region / frame / channels (``twin``), one display range
  across items (``lut_share``), a shared M/T/Z cursor (``cursor``), a chart's hovered row
  lighting objects (``highlight``), a line feeding a profile or kymograph chart.

Three rules carry over from the movie timeline and the plot nodes: **partial dicts merge**
(:func:`normalize` fills every default, so a file from an older build still loads),
**every error names its spec path** (``items[2].image.channels[0].rgb: …``), and the
document has **one spelling** (:func:`canonical_json`) because the render cache and the
memo hash the string.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure.layout import LETTER_STYLES, PAGE_PRESETS

SCHEMA = "nd2studios-figure/1"

GOALS = ("plot", "overlay", "composite", "table", "montage")

ITEM_KINDS = ("image", "montage", "picture", "chart", "colorbar", "histogram", "scale_bar",
              "legend", "table", "text", "arrow", "line", "shape", "measure_line", "letter",
              "ortho", "kymograph", "profile")

#: link kind → the item-reference fields it carries
LINK_KINDS: Dict[str, Tuple[str, ...]] = {
    "zoom": ("parent", "child"),
    "twin": ("a", "b"),
    "lut_share": ("items",),
    "cursor": ("items",),
    "highlight": ("chart", "targets"),
    "profile": ("line", "chart"),
    "kymograph": ("line", "chart"),
}

#: what a twin may share with its other half
TWIN_FIELDS = ("region", "frame", "channels", "layers", "decor")

CHART_KINDS = ("xy", "timeseries", "bar", "distribution", "heatmap", "hexbin", "msd",
               "survival", "rose")

DISTRIBUTION_KINDS = ("histogram", "kde", "box", "violin", "ecdf", "strip", "superplot")

LAYER_KINDS = ("points", "labels", "tracks", "vectors", "field", "voxels", "mesh", "grid")

REGION_KINDS = ("full", "rect_um", "placement", "object")

CHANNEL_MODES = ("auto_pct", "viewer", "manual", "shared")

LIMIT_MODES = ("series", "frame", "manual", "normalized")

STYLES = ("paper", "talk", "poster", "dark")

FONTS = ("sans", "serif")

PALETTE_NAMES = ("colorblind", "tab10", "viridis", "greys")

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOURCE = re.compile(r"^[A-Za-z0-9_]+:[^:]+$")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class FigureError(ValueError):
    """A figure document that cannot be used; ``path`` names the offending value."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


# ── defaults ─────────────────────────────────────────────────────────────────────────
def page_defaults(preset: str = "single_column") -> Dict[str, Any]:
    p = PAGE_PRESETS.get(preset) or PAGE_PRESETS["single_column"]
    return {"preset": preset if preset in PAGE_PRESETS else "custom",
            "width_mm": float(p["width_mm"]), "height_mm": float(p["height_mm"]),
            "dpi": 300, "style": "paper", "font": "sans", "font_pt": 8.0,
            "line_width": 1.0, "palette": "colorblind", "dark": False, "grid_mm": 1.0,
            "letters": "A"}


def channel_defaults() -> Dict[str, Any]:
    return {"c": 0, "rgb": "#ffffff", "mode": "auto_pct", "pct": [0.5, 99.8],
            "lo": None, "hi": None, "gamma": 1.0, "label": ""}


def layer_defaults() -> Dict[str, Any]:
    return {"domain": "label", "layer": "", "kind": "labels", "column": "",
            "look": {}, "cmap": "auto", "opacity": 55,
            "limits": {"mode": "series", "lo": None, "hi": None, "vmax": None}}


def image_defaults() -> Dict[str, Any]:
    return {"source": None, "frame": {"m": 0, "t": 0, "z": "max"},
            "region": {"from": "full"}, "channels": [], "flip_x": False, "flip_y": False,
            "z_mode": "plain", "t_mode": "plain", "layers": [], "decor": {}}


def montage_defaults() -> Dict[str, Any]:
    return {"tile_by": "m", "columns": 0, "gap_mm": 0.5, "labels": "auto",
            "lut": "shared", "count": 0, "start": 0, "step": 1}


def chart_defaults() -> Dict[str, Any]:
    return {"kind": "xy", "source": None, "data": None, "table": "", "domain": "label",
            "x": "", "y": "", "value": "", "group_by": "", "facet_by": "", "color_by": "",
            "sub_kind": "", "bins": 20, "normalize": "count",
            "draw": {"mode": "scatter", "error": "none", "error_style": "bars",
                     "step": False, "stacked": False, "points": True},
            "fit": {"model": "none", "span": None},
            "stats": {"test": "none", "pairs": []},
            "axes": {"title": "", "x_label": "", "y_label": "", "log_x": False,
                     "log_y": False, "x_range": [None, None], "y_range": [None, None]},
            "legend": {"show": True, "loc": "inside"},
            "annotations": []}


ITEM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "image": {"image": image_defaults},
    "montage": {"image": image_defaults, "montage": montage_defaults},
    "picture": {"picture": lambda: {"source": None, "frame": {"m": 0, "t": 0, "z": 0}}},
    "chart": {"chart": chart_defaults},
    "colorbar": {"of": lambda: {"item": None, "layer": None, "channel": None},
                 "label": lambda: "", "orientation": lambda: "horizontal",
                 "ticks": lambda: 5},
    "histogram": {"of": lambda: {"item": None, "channel": 0}, "show_window": lambda: True,
                  "log": lambda: True},
    "scale_bar": {"of": lambda: {"item": None}, "length_um": lambda: 20.0,
                  "label": lambda: True},
    "legend": {"of": lambda: {"item": None}},
    "table": {"table": lambda: {"source": None, "data": None, "columns": [],
                                "group_by": "", "stats": ["n", "mean", "sem"]}},
    "text": {"text": lambda: "", "pt": lambda: None, "bold": lambda: False,
             "align": lambda: "left"},
    "arrow": {"to_mm": lambda: [0.0, 0.0], "width_pt": lambda: 1.0,
              "color": lambda: "#ffffff"},
    "line": {"to_mm": lambda: [0.0, 0.0], "width_pt": lambda: 1.0,
             "color": lambda: "#ffffff", "of": lambda: {"item": None}},
    "shape": {"shape": lambda: "rect", "width_pt": lambda: 0.75, "color": lambda: "#ffffff",
              "fill": lambda: None, "dash": lambda: "solid"},
    "measure_line": {"to_mm": lambda: [0.0, 0.0], "of": lambda: {"item": None},
                     "width_pt": lambda: 1.0, "color": lambda: "#ffffff"},
    "letter": {"text": lambda: "", "pt": lambda: None},
    "ortho": {"of": lambda: {"item": None}, "axis": lambda: "xz",
              "at_mm": lambda: None},
    "kymograph": {"of": lambda: {"item": None, "line": None}, "width_px": lambda: 1,
                  "cmap": lambda: "grey"},
    "profile": {"of": lambda: {"item": None, "line": None}, "width_px": lambda: 1,
                "chart": chart_defaults},
}


def new_document(goal: Optional[str] = None, preset: str = "single_column"
                 ) -> Dict[str, Any]:
    """An empty figure document of ``preset``, ``goal`` unset until the start card answers."""
    return {"schema": SCHEMA, "goal": goal, "page": page_defaults(preset),
            "sources": [], "data": [], "items": [], "links": []}


# ── normalization + validation ───────────────────────────────────────────────────────
def _merge(defaults: Mapping[str, Any], given: Any, path: str) -> Dict[str, Any]:
    """``defaults`` updated by ``given`` (a mapping), nested one level for dict values."""
    if given is None:
        return copy.deepcopy(dict(defaults))
    if not isinstance(given, Mapping):
        raise FigureError(path, f"must be an object, not {type(given).__name__}")
    out = copy.deepcopy(dict(defaults))
    for k, v in given.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v, f"{path}.{k}")
        else:
            out[k] = copy.deepcopy(v)
    return out


def _num(v: Any, path: str, *, lo: Optional[float] = None, hi: Optional[float] = None,
         allow_none: bool = False) -> Optional[float]:
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FigureError(path, f"must be a number, not {v!r}")
    f = float(v)
    if f != f:
        raise FigureError(path, "must not be NaN")
    if lo is not None and f < lo:
        raise FigureError(path, f"must be ≥ {lo}, not {f}")
    if hi is not None and f > hi:
        raise FigureError(path, f"must be ≤ {hi}, not {f}")
    return f


def _choice(v: Any, path: str, choices: Sequence[Any]) -> Any:
    if v not in choices:
        raise FigureError(path, f"must be one of {list(choices)}, not {v!r}")
    return v


def _rect(v: Any, path: str) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise FigureError(path, "must be [x, y, w, h] in mm")
    out = [_num(x, f"{path}[{i}]") for i, x in enumerate(v)]
    if out[2] <= 0 or out[3] <= 0:
        raise FigureError(path, f"width and height must be > 0, not {out[2]} × {out[3]}")
    return [float(x) for x in out]             # type: ignore[arg-type]


def _frame(v: Any, path: str) -> Dict[str, Any]:
    f = _merge({"m": 0, "t": 0, "z": "max"}, v, path)
    for ax in ("m", "t"):
        x = f.get(ax)
        if x != "cursor":
            f[ax] = int(_num(x, f"{path}.{ax}", lo=0))
    z = f.get("z")
    if z not in ("max", "mean", "cursor"):
        f["z"] = int(_num(z, f"{path}.z", lo=0))
    return f


def _region(v: Any, path: str) -> Dict[str, Any]:
    r = _merge({"from": "full"}, v, path)
    _choice(r["from"], f"{path}.from", REGION_KINDS)
    if r["from"] == "rect_um":
        rect = r.get("rect_um")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise FigureError(f"{path}.rect_um", "must be [y0, y1, x0, x1] in µm")
        r["rect_um"] = [_num(x, f"{path}.rect_um[{i}]") for i, x in enumerate(rect)]
        if not (r["rect_um"][1] > r["rect_um"][0] and r["rect_um"][3] > r["rect_um"][2]):
            raise FigureError(f"{path}.rect_um", "must have y1 > y0 and x1 > x0")
    elif r["from"] == "object":
        if r.get("object") is None:
            raise FigureError(f"{path}.object", "names no object id")
        r["object"] = int(_num(r["object"], f"{path}.object", lo=0))
        r["margin_um"] = _num(r.get("margin_um", 5.0), f"{path}.margin_um", lo=0)
    return r


def _channel(v: Any, path: str) -> Dict[str, Any]:
    c = _merge(channel_defaults(), v, path)
    c["c"] = int(_num(c["c"], f"{path}.c", lo=0))
    if not _HEX.match(str(c["rgb"])):
        raise FigureError(f"{path}.rgb", f"must be '#rrggbb', not {c['rgb']!r}")
    _choice(c["mode"], f"{path}.mode", CHANNEL_MODES)
    pct = c.get("pct")
    if not isinstance(pct, (list, tuple)) or len(pct) != 2:
        raise FigureError(f"{path}.pct", "must be [low, high] percentiles")
    c["pct"] = [_num(pct[0], f"{path}.pct[0]", lo=0, hi=100),
                _num(pct[1], f"{path}.pct[1]", lo=0, hi=100)]
    if c["pct"][1] <= c["pct"][0]:
        raise FigureError(f"{path}.pct", "high must be above low")
    c["lo"] = _num(c["lo"], f"{path}.lo", allow_none=True)
    c["hi"] = _num(c["hi"], f"{path}.hi", allow_none=True)
    if c["mode"] == "manual" and (c["lo"] is None or c["hi"] is None):
        raise FigureError(f"{path}.mode", "manual needs lo and hi")
    c["gamma"] = _num(c["gamma"], f"{path}.gamma", lo=0.05, hi=20)
    return c


def _layer(v: Any, path: str) -> Dict[str, Any]:
    L = _merge(layer_defaults(), v, path)
    _choice(L["kind"], f"{path}.kind", LAYER_KINDS)
    L["opacity"] = int(_num(L["opacity"], f"{path}.opacity", lo=0, hi=100))
    lim = L["limits"]
    _choice(lim["mode"], f"{path}.limits.mode", LIMIT_MODES)
    lim["lo"] = _num(lim.get("lo"), f"{path}.limits.lo", allow_none=True)
    lim["hi"] = _num(lim.get("hi"), f"{path}.limits.hi", allow_none=True)
    lim["vmax"] = _num(lim.get("vmax"), f"{path}.limits.vmax", allow_none=True)
    if lim["mode"] == "manual" and (lim["lo"] is None or lim["hi"] is None):
        raise FigureError(f"{path}.limits", "manual limits need lo and hi")
    if lim["mode"] == "normalized" and not (lim["vmax"] and lim["vmax"] > 0):
        raise FigureError(f"{path}.limits.vmax", "normalized limits need vmax > 0")
    if L["kind"] == "field" and not str(L.get("column") or ""):
        pass                                  # blank = the most derived column (bind)
    return L


def _image(v: Any, path: str, sources: Mapping[str, Any], *, twin_ok: bool) -> Dict[str, Any]:
    img = _merge(image_defaults(), v, path)
    src = img.get("source")
    if src is not None and src not in sources:
        raise FigureError(f"{path}.source", f"unknown source {src!r}")
    if src is None and not twin_ok:
        raise FigureError(f"{path}.source", "names no source")
    img["frame"] = _frame(img.get("frame"), f"{path}.frame")
    img["region"] = _region(img.get("region"), f"{path}.region")
    chans = img.get("channels")
    if not isinstance(chans, list):
        raise FigureError(f"{path}.channels", "must be a list")
    img["channels"] = [_channel(c, f"{path}.channels[{i}]") for i, c in enumerate(chans)]
    layers = img.get("layers")
    if not isinstance(layers, list):
        raise FigureError(f"{path}.layers", "must be a list")
    img["layers"] = [_layer(L, f"{path}.layers[{i}]") for i, L in enumerate(layers)]
    _choice(img["z_mode"], f"{path}.z_mode", ("plain", "depth_coded"))
    _choice(img["t_mode"], f"{path}.t_mode", ("plain", "temporal_code"))
    img["flip_x"] = bool(img["flip_x"])
    img["flip_y"] = bool(img["flip_y"])
    if not isinstance(img.get("decor"), Mapping):
        raise FigureError(f"{path}.decor", "must be an object")
    d = dict(img["decor"])
    if "scale_bar_um" in d and d["scale_bar_um"] is not None:
        d["scale_bar_um"] = _num(d["scale_bar_um"], f"{path}.decor.scale_bar_um", lo=0)
    img["decor"] = d
    return img


def _chart(v: Any, path: str, sources: Mapping[str, Any], data_ids: Sequence[str]
           ) -> Dict[str, Any]:
    ch = _merge(chart_defaults(), v, path)
    _choice(ch["kind"], f"{path}.kind", CHART_KINDS)
    if ch.get("source") is not None and ch["source"] not in sources:
        raise FigureError(f"{path}.source", f"unknown source {ch['source']!r}")
    if ch.get("data") is not None and ch["data"] not in data_ids:
        raise FigureError(f"{path}.data", f"unknown data {ch['data']!r}")
    if ch["kind"] == "distribution":
        _choice(ch["sub_kind"] or "histogram", f"{path}.sub_kind", DISTRIBUTION_KINDS)
        ch["sub_kind"] = ch["sub_kind"] or "histogram"
    ch["bins"] = int(_num(ch["bins"], f"{path}.bins", lo=1, hi=10000))
    ax = ch["axes"]
    for key in ("x_range", "y_range"):
        r = ax.get(key)
        if r is None:
            ax[key] = [None, None]
            continue
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise FigureError(f"{path}.axes.{key}", "must be [low, high]")
        ax[key] = [_num(r[0], f"{path}.axes.{key}[0]", allow_none=True),
                   _num(r[1], f"{path}.axes.{key}[1]", allow_none=True)]
        if ax[key][0] is not None and ax[key][1] is not None and ax[key][1] <= ax[key][0]:
            raise FigureError(f"{path}.axes.{key}", "high must be above low")
    ax["log_x"] = bool(ax["log_x"])
    ax["log_y"] = bool(ax["log_y"])
    _choice(ch["legend"]["loc"], f"{path}.legend.loc", ("inside", "right", "bottom", "none"))
    _choice(ch["draw"]["error"], f"{path}.draw.error",
            ("none", "sd", "sem", "ci95", "pooled_sd", "column"))
    _choice(ch["draw"]["error_style"], f"{path}.draw.error_style", ("bars", "band"))
    _choice(ch["draw"]["mode"], f"{path}.draw.mode", ("line", "scatter", "line_markers"))
    _choice(ch["fit"]["model"], f"{path}.fit.model",
            ("none", "linear", "langmuir", "exponential", "power"))
    return ch


def _item(v: Any, path: str, sources: Mapping[str, Any], data_ids: Sequence[str],
          twin_children: Iterable[str]) -> Dict[str, Any]:
    if not isinstance(v, Mapping):
        raise FigureError(path, "must be an object")
    kind = v.get("kind")
    _choice(kind, f"{path}.kind", ITEM_KINDS)
    iid = v.get("id")
    if not isinstance(iid, str) or not _ID.match(iid):
        raise FigureError(f"{path}.id", f"must be an identifier, not {iid!r}")
    it: Dict[str, Any] = {"id": iid, "kind": kind, "rect_mm": _rect(v.get("rect_mm"),
                                                                     f"{path}.rect_mm"),
                          "letter": str(v.get("letter") or ""), "z": int(v.get("z", 0) or 0),
                          "locked": bool(v.get("locked", False)),
                          "group": str(v.get("group") or ""),
                          "name": str(v.get("name") or "")}
    for key, make in ITEM_DEFAULTS.get(kind, {}).items():
        given = v.get(key)
        if key == "image":
            it[key] = _image(given, f"{path}.image", sources, twin_ok=iid in twin_children)
        elif key == "montage":
            m = _merge(make(), given, f"{path}.montage")
            _choice(m["tile_by"], f"{path}.montage.tile_by", ("m", "t", "z", "c"))
            m["columns"] = int(_num(m["columns"], f"{path}.montage.columns", lo=0))
            m["gap_mm"] = _num(m["gap_mm"], f"{path}.montage.gap_mm", lo=0)
            _choice(m["lut"], f"{path}.montage.lut", ("shared", "per_tile"))
            m["count"] = int(_num(m["count"], f"{path}.montage.count", lo=0))
            m["start"] = int(_num(m["start"], f"{path}.montage.start", lo=0))
            m["step"] = int(_num(m["step"], f"{path}.montage.step", lo=1))
            it[key] = m
        elif key == "chart":
            it[key] = _chart(given, f"{path}.chart", sources, data_ids)
        else:
            default = make()
            if isinstance(default, Mapping):
                it[key] = _merge(default, given, f"{path}.{key}")
            else:
                it[key] = copy.deepcopy(default) if given is None else copy.deepcopy(given)
    # kind-specific extras
    if kind == "colorbar":
        _choice(it["orientation"], f"{path}.orientation", ("horizontal", "vertical"))
    if kind in ("text", "letter"):
        it["text"] = str(it.get("text") or "")
    if kind == "ortho":
        _choice(it["axis"], f"{path}.axis", ("xz", "yz"))
    # carry unknown top-level keys through, so a newer build's fields survive a round trip
    for k, val in v.items():
        if k not in it:
            it[k] = copy.deepcopy(val)
    return it


def _link(v: Any, path: str, items: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(v, Mapping):
        raise FigureError(path, "must be an object")
    kind = v.get("kind")
    _choice(kind, f"{path}.kind", tuple(LINK_KINDS))
    ln: Dict[str, Any] = {"kind": kind}

    def ref(key: str, kinds: Optional[Sequence[str]] = None) -> str:
        x = v.get(key)
        if x not in items:
            raise FigureError(f"{path}.{key}", f"unknown item {x!r}")
        if kinds and items[x]["kind"] not in kinds:
            raise FigureError(f"{path}.{key}", f"{x!r} is a {items[x]['kind']}, not one of "
                                               f"{list(kinds)}")
        return str(x)

    def refs(key: str, kinds: Optional[Sequence[str]] = None, *, least: int = 2) -> List[str]:
        xs = v.get(key)
        if not isinstance(xs, (list, tuple)) or len(xs) < least:
            raise FigureError(f"{path}.{key}", f"must list at least {least} items")
        out = []
        for i, x in enumerate(xs):
            if x not in items:
                raise FigureError(f"{path}.{key}[{i}]", f"unknown item {x!r}")
            if kinds and items[x]["kind"] not in kinds:
                raise FigureError(f"{path}.{key}[{i}]", f"{x!r} is a {items[x]['kind']}")
            out.append(str(x))
        if len(set(out)) != len(out):
            raise FigureError(f"{path}.{key}", "lists an item twice")
        return out

    imgs = ("image", "montage")
    if kind == "zoom":
        ln["parent"] = ref("parent", imgs)
        ln["child"] = ref("child", ("image",))
        if ln["parent"] == ln["child"]:
            raise FigureError(f"{path}.child", "cannot zoom into itself")
        ln["box"] = _merge({"style": "dashed", "leaders": True, "color": "#ffffff",
                            "width_pt": 0.75}, v.get("box"), f"{path}.box")
    elif kind == "twin":
        ln["a"] = ref("a", imgs)
        ln["b"] = ref("b", imgs)
        if ln["a"] == ln["b"]:
            raise FigureError(f"{path}.b", "a twin of itself")
        share = v.get("share", list(TWIN_FIELDS))
        if not isinstance(share, (list, tuple)):
            raise FigureError(f"{path}.share", "must be a list")
        for i, s in enumerate(share):
            _choice(s, f"{path}.share[{i}]", TWIN_FIELDS)
        ln["share"] = [str(s) for s in share]
    elif kind == "lut_share":
        ln["items"] = refs("items", imgs)
    elif kind == "cursor":
        ln["items"] = refs("items", imgs + ("chart", "kymograph", "ortho"))
        axes = v.get("axes", ["t"])
        if not isinstance(axes, (list, tuple)) or not axes:
            raise FigureError(f"{path}.axes", "must list m, t and/or z")
        for i, a in enumerate(axes):
            _choice(a, f"{path}.axes[{i}]", ("m", "t", "z"))
        ln["axes"] = [str(a) for a in axes]
    elif kind == "highlight":
        ln["chart"] = ref("chart", ("chart",))
        ln["targets"] = refs("targets", imgs, least=1)
    elif kind in ("profile", "kymograph"):
        ln["line"] = ref("line", ("line", "measure_line"))
        ln["chart"] = ref("chart", ("chart", "profile", "kymograph"))
    for k, val in v.items():
        if k not in ln:
            ln[k] = copy.deepcopy(val)
    return ln


def migrate(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """``doc`` brought to the current :data:`SCHEMA`. Only version 1 exists; a document
    without a schema is read as version 1, a later version is refused by name."""
    d = copy.deepcopy(dict(doc))
    s = d.get("schema")
    if s is None:
        d["schema"] = SCHEMA
    elif s != SCHEMA:
        raise FigureError("schema", f"{s!r} is not {SCHEMA!r}; this build cannot read it")
    return d


def normalize(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """A complete, validated copy of ``doc``: every default filled, every value checked,
    every reference resolved. Raises :class:`FigureError` naming the first bad value."""
    if not isinstance(doc, Mapping):
        raise FigureError("", "a figure document must be an object")
    d = migrate(doc)
    goal = d.get("goal")
    if goal is not None:
        _choice(goal, "goal", GOALS)
    page_in = d.get("page") or {}
    preset = str(page_in.get("preset") or "single_column")
    page = _merge(page_defaults(preset if preset in PAGE_PRESETS else "single_column"),
                  page_in, "page")
    page["preset"] = preset if preset in PAGE_PRESETS else "custom"
    page["width_mm"] = _num(page["width_mm"], "page.width_mm", lo=10, hi=2000)
    page["height_mm"] = _num(page["height_mm"], "page.height_mm", lo=10, hi=2000)
    page["dpi"] = int(_num(page["dpi"], "page.dpi", lo=50, hi=2400))
    _choice(page["style"], "page.style", STYLES)
    _choice(page["font"], "page.font", FONTS)
    page["font_pt"] = _num(page["font_pt"], "page.font_pt", lo=3, hi=72)
    page["line_width"] = _num(page["line_width"], "page.line_width", lo=0.1, hi=20)
    _choice(page["palette"], "page.palette", PALETTE_NAMES)
    page["dark"] = bool(page["dark"])
    page["grid_mm"] = _num(page["grid_mm"], "page.grid_mm", lo=0)
    _choice(page["letters"], "page.letters", LETTER_STYLES)

    sources: Dict[str, Dict[str, Any]] = {}
    src_out: List[Dict[str, Any]] = []
    for i, s in enumerate(d.get("sources") or []):
        path = f"sources[{i}]"
        if not isinstance(s, Mapping):
            raise FigureError(path, "must be an object")
        sid = s.get("id")
        if not isinstance(sid, str) or not _ID.match(sid):
            raise FigureError(f"{path}.id", f"must be an identifier, not {sid!r}")
        if sid in sources:
            raise FigureError(f"{path}.id", f"duplicate source id {sid!r}")
        src = s.get("source")
        if not isinstance(src, str) or not _SOURCE.match(src):
            raise FigureError(f"{path}.source", f"must be '<page id>:<output name>', not {src!r}")
        rec = {"id": sid, "source": src, "item": s.get("item"), "part": s.get("part"),
               "label": str(s.get("label") or src.split(":", 1)[1])}
        sources[sid] = rec
        src_out.append(rec)

    data_out: List[Dict[str, Any]] = []
    data_ids: List[str] = []
    for i, dd in enumerate(d.get("data") or []):
        path = f"data[{i}]"
        if not isinstance(dd, Mapping):
            raise FigureError(path, "must be an object")
        did = dd.get("id")
        if not isinstance(did, str) or not _ID.match(did):
            raise FigureError(f"{path}.id", f"must be an identifier, not {did!r}")
        if did in data_ids or did in sources:
            raise FigureError(f"{path}.id", f"duplicate id {did!r}")
        frm = dd.get("from")
        if not isinstance(frm, (list, tuple)) or not frm:
            raise FigureError(f"{path}.from", "must list at least one source")
        for j, f in enumerate(frm):
            if f not in sources:
                raise FigureError(f"{path}.from[{j}]", f"unknown source {f!r}")
        rec = _merge({"id": did, "from": list(frm), "domain": "label", "table": "",
                      "pool": len(frm) > 1, "aggregate": None, "join": None, "where": []},
                     dd, path)
        data_ids.append(did)
        data_out.append(rec)

    raw_items = d.get("items") or []
    if not isinstance(raw_items, list):
        raise FigureError("items", "must be a list")
    raw_links = d.get("links") or []
    if not isinstance(raw_links, list):
        raise FigureError("links", "must be a list")
    twin_children = {ln.get("b") for ln in raw_links
                     if isinstance(ln, Mapping) and ln.get("kind") == "twin"}
    items: Dict[str, Dict[str, Any]] = {}
    items_out: List[Dict[str, Any]] = []
    for i, it in enumerate(raw_items):
        rec = _item(it, f"items[{i}]", sources, data_ids, twin_children)
        if rec["id"] in items:
            raise FigureError(f"items[{i}].id", f"duplicate item id {rec['id']!r}")
        items[rec["id"]] = rec
        items_out.append(rec)
    links_out = [_link(ln, f"links[{i}]", items) for i, ln in enumerate(raw_links)]

    # twin chains must be acyclic, and a twin's other half must have a source somewhere
    nxt = {ln["b"]: ln["a"] for ln in links_out if ln["kind"] == "twin"}
    for start in nxt:
        seen = {start}
        cur = start
        while cur in nxt:
            cur = nxt[cur]
            if cur in seen:
                raise FigureError("links", f"twin links form a cycle through {start!r}")
            seen.add(cur)
        if items[cur].get("image", {}).get("source") is None:
            raise FigureError("links", f"twin {start!r} inherits from {cur!r}, which has "
                                       f"no source")
    # a popout child must be an image whose region is a part of something
    out = {"schema": SCHEMA, "goal": goal, "page": page, "sources": src_out,
           "data": data_out, "items": items_out, "links": links_out}
    for k, val in d.items():
        if k not in out:
            out[k] = copy.deepcopy(val)
    return out


def check(doc: Mapping[str, Any]) -> List[str]:
    """``[]`` when ``doc`` normalizes, else the first error as one line (a GUI shows it
    beside the field the path names)."""
    try:
        normalize(doc)
    except FigureError as e:
        return [str(e)]
    return []


# ── canonical spelling, digest, lookups ──────────────────────────────────────────────
def _json_safe(v: Any) -> Any:
    try:
        import numpy as np
    except Exception:                      # pragma: no cover — numpy is a core dep
        np = None
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if np is not None and isinstance(v, np.generic):
        return v.item()
    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and v != v:
        return None
    return v


def canonical_json(doc: Mapping[str, Any]) -> str:
    """The one spelling of ``doc``: sorted keys, no spaces, numpy scalars as Python."""
    return json.dumps(_json_safe(doc), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(doc: Mapping[str, Any]) -> str:
    """A 16-hex digest of the canonical spelling — the render cache and memo key."""
    return hashlib.sha1(canonical_json(doc).encode("utf-8")).hexdigest()[:16]


def item(doc: Mapping[str, Any], item_id: str) -> Dict[str, Any]:
    for it in doc.get("items") or []:
        if it.get("id") == item_id:
            return it
    raise KeyError(item_id)


def source(doc: Mapping[str, Any], source_id: str) -> Dict[str, Any]:
    for s in doc.get("sources") or []:
        if s.get("id") == source_id:
            return s
    raise KeyError(source_id)


def links_of(doc: Mapping[str, Any], item_id: str) -> List[Dict[str, Any]]:
    """Every link that names ``item_id`` in any of its reference fields."""
    out = []
    for ln in doc.get("links") or []:
        for key in LINK_KINDS.get(ln.get("kind"), ()):
            v = ln.get(key)
            if v == item_id or (isinstance(v, (list, tuple)) and item_id in v):
                out.append(ln)
                break
    return out


def twin_parent(doc: Mapping[str, Any], item_id: str) -> Optional[Tuple[str, List[str]]]:
    """``(the item this one is a twin OF, the fields shared)`` or ``None``."""
    for ln in doc.get("links") or []:
        if ln.get("kind") == "twin" and ln.get("b") == item_id:
            return str(ln["a"]), list(ln.get("share") or TWIN_FIELDS)
    return None


def effective_image(doc: Mapping[str, Any], item_id: str) -> Dict[str, Any]:
    """The image block of ``item_id`` with its twin inheritance resolved: each shared field
    comes from the twin's other half (recursively), the rest are the item's own. The
    ``source`` follows ``region`` and ``frame`` — a twin shows the same field."""
    it = item(doc, item_id)
    own = copy.deepcopy(it.get("image") or image_defaults())
    tp = twin_parent(doc, item_id)
    if tp is None:
        return own
    parent_id, share = tp
    parent = effective_image(doc, parent_id)
    for f in share:
        if f in parent:
            own[f] = copy.deepcopy(parent[f])
    if own.get("source") is None or "region" in share or "frame" in share:
        own["source"] = parent.get("source")
    return own


def item_digest(doc: Mapping[str, Any], item_id: str) -> str:
    """The digest of what decides one item's picture: the page style, the item (with twin
    inheritance resolved for an image), and the links that touch it."""
    it = item(doc, item_id)
    body = copy.deepcopy(it)
    if "image" in body:
        body["image"] = effective_image(doc, item_id)
    return digest({"page": doc.get("page"), "item": body,
                   "links": links_of(doc, item_id)})
