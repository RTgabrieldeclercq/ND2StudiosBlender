"""Node demos — what a node does, shown on a phantom. The Qt-free half.

Every registered op gets a :class:`DemoRecipe`: which :mod:`nodegraph.phantom` to seed, which
*prelude* chain manufactures the structure it consumes (threshold → label before Measure, a
spot detector before Voronoi, a tracker before Track Field), which of its parameters become
sliders and over what range, and what *kind* of result to show — an image, a mask, labels,
points, tracks, a vector field, a mesh, a table, a plot, or a *guide* for the nodes that do
not transform pixels (writers, page boundaries, zones, the Viewer). :class:`DemoSession`
then runs ``seed → prelude → node`` through the real engine and extracts the planes and
geometry the window paints.

**Where the decisions live.** The broad strokes come from the node's functional role
(:mod:`nodegraph.roles`, :data:`ROLE_DEFAULTS`) so a new node demos itself the day it is
registered; everything the role cannot know — a second input, a fixed column, a shapes
list, a slow mode, the "key features" of a guide — is curated in
``codemap/node_demos.json`` and validated against the live registry by
``selftest.test_node_demos`` (the precedent is ``codemap/node_roles.json``).

**Three engine facts this module is built on.** (1) A param the user has not touched is
*absent* from the node and the engine resolves its ``derive`` / default itself
(``nodegraph/engine.py``, "a param is auto/derived iff it is absent"), so a session sends
only the values the sliders changed; the derived value is computed here purely to START
the slider where the inspector's auto box would. (2) The GUI seeds an ``io.load`` node with
a Dataset + MetaEnvelope rather than a path (``nodelab_v2.runner``), and that is exactly how
a phantom enters the graph — no ``define_node``, nothing touched on the live runner
(INV-03). (3) An engine rebuilt per slider value keeps its ``Memo`` only if it also shares
one ``TileCache`` (``nodegraph/engine.py``, the cache note), so the session owns both and
the prelude never recomputes.

Slider ranges are a heuristic because the registry carries no min/max (``SocketSpec`` has
``unit``, ``default``, ``derive`` and ``pick_kind``, deliberately not bounds): the unit, the
pick gesture and the magnitude of the default choose a span, the phantom's own axes and
histogram bound the index-like and level-like ones, and the JSON overrides the rest.

Qt-free: importable by the selftest and by the GUI alike.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph import roles as R
from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.graph import Graph, NodeInstance
from nodegraph.memo import Memo
from nodegraph.metadata import MetaEnvelope, envelope_symbols, eval_derive, propagate_meta
from nodegraph.phantom import PHANTOMS, Phantom, intensity_range, phantom
from nodegraph.registry import NODES
from nodegraph.sockets import SocketType
from nodegraph.streaming import TileCache, realize
from nodelab_v2.value_steps import value_decimals, value_step

__all__ = [
    "KINDS", "GUIDE", "SliderRange", "PreludeStep", "Scenario", "DemoRecipe", "DemoResult",
    "DemoSession", "DEMOS_PATH", "ROLE_DEFAULTS", "PRELUDES", "load_curation",
    "validate_curation", "recipe_for", "scenario_recipe", "role_default", "all_demo_ops",
    "control_kind", "slider_range", "derived_defaults", "derived_modes", "guide_html",
    "demo_rows",
]

#: the curated overrides, beside the roles file they extend
DEMOS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "codemap", "node_demos.json")

#: what the window shows for the *after*. ``scalar`` is an image with a colour-mapped Voxel
#: field over it (a distance transform, a flow magnitude); ``guide`` runs nothing.
KINDS: Tuple[str, ...] = ("image", "mask", "labels", "scalar", "points", "tracks", "field",
                          "mesh", "table", "plot", "guide")
GUIDE = "guide"

#: ``name → the overlay group`` the window enables for that kind (``None`` = none)
KIND_OVERLAY: Dict[str, Optional[str]] = {
    "image": None, "mask": "voxels", "labels": "labels", "scalar": "voxels",
    "points": "points", "tracks": "tracks", "field": "vectors", "mesh": "mesh",
    "table": "labels", "plot": None, "guide": None,
}


@dataclass(frozen=True)
class SliderRange:
    lo: float
    hi: float
    step: float
    decimals: int


@dataclass(frozen=True)
class PreludeStep:
    op: str
    params: Mapping[str, Any] = field(default_factory=dict)
    modes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    """One synthetic world a demo can be switched to (2026-10-07): a phantom with its
    keywords, the plane to view, values the scenario fixes, and a one-line note on what to
    look for. The first scenario is the demo's default."""
    label: str
    phantom: str
    phantom_kw: Mapping[str, Any] = field(default_factory=dict)
    view: Mapping[str, int] = field(default_factory=dict)
    fixed_params: Mapping[str, Any] = field(default_factory=dict)
    fixed_modes: Mapping[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class DemoRecipe:
    op: str
    kind: str
    phantom: str = ""
    phantom_kw: Mapping[str, Any] = field(default_factory=dict)
    #: alternative synthetic worlds the window offers in a *Synthetic data* dropdown; the
    #: entry-level ``phantom`` may be empty when scenarios exist — the first one is then the
    #: default (see :func:`scenario_recipe`)
    scenarios: Tuple["Scenario", ...] = ()
    #: which scenario this recipe IS (``-1`` = the entry as curated, no scenario applied)
    scenario_index: int = -1
    #: ``seed.image → pre0.data → … → demo``; each step wired ``out → data``
    prelude: Tuple[PreludeStep, ...] = ()
    #: where the demo node's PRIMARY input comes from: ``"tail"`` (the prelude's last node, or
    #: the seed when there is none) or ``"src"`` (the seed, past the prelude)
    data_from: str = "tail"
    #: extra Dataset inputs: ``dst socket → source``, a source being ``{"phantom": name,
    #: "kw": {...}}`` (a second seed) or ``{"pre": i}`` (a prelude node's output)
    extra_inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: params the user cannot edit (layer names, columns, a shapes list)
    fixed_params: Mapping[str, Any] = field(default_factory=dict)
    fixed_modes: Mapping[str, str] = field(default_factory=dict)
    #: curated slider spans, ``param → (lo, hi)``; everything else → :func:`slider_range`
    sliders: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    #: False → a Run button instead of live recomputation (and the gate does not run it)
    live: bool = True
    slow_modes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    slow_reason: str = ""
    #: selftest-only overrides (``iterations=3``) so the gate stays quick
    gate_params: Mapping[str, Any] = field(default_factory=dict)
    #: optional packages the compute imports lazily (``openpiv``); absent → the window says
    #: so instead of running, and the gate skips the op
    requires: Tuple[str, ...] = ()
    #: the ``(t, z, c)`` plane the window shows; missing keys → t=0, z=middle, c=0
    view: Mapping[str, int] = field(default_factory=dict)
    #: curated "key features / how to use" bullets
    features: Tuple[str, ...] = ()
    #: the role the defaults came from (for the window's header)
    role: str = ""
    stage: str = ""

    @property
    def is_guide(self) -> bool:
        return self.kind == GUIDE or not (self.phantom or self.scenarios)


# ── named preludes and role defaults ──────────────────────────────────────────

#: the chains a recipe may name instead of spelling out. Layer names are the nodes' own
#: defaults — threshold writes ``mask``, label reads ``mask`` and writes ``labels``, measure
#: reads ``labels`` — so the chain lines up with nothing configured.
PRELUDES: Dict[str, Tuple[PreludeStep, ...]] = {
    "threshold2d": (PreludeStep("analysis.threshold", {}, {"method": "otsu"}),),
    "labels2d": (PreludeStep("analysis.threshold", {}, {"method": "otsu"}),
                 PreludeStep("analysis.label")),
    "measured2d": (PreludeStep("analysis.threshold", {}, {"method": "otsu"}),
                   PreludeStep("analysis.label"),
                   PreludeStep("analysis.measure")),
    "spots2d": (PreludeStep("detect.spots"),),
    "tracks2d": (PreludeStep("analysis.threshold", {}, {"method": "otsu"}),
                 PreludeStep("analysis.label"),
                 PreludeStep("track.objects")),
    "piv2d": (PreludeStep("analysis.piv"),),
}

#: ``role → (kind, phantom, phantom_kw, prelude name)``. Keyed on the functional role from
#: ``codemap/node_roles.json`` so an unlisted node still gets a sensible demo.
ROLE_DEFAULTS: Dict[str, Tuple[str, str, Dict[str, Any], str]] = {
    "input_output": (GUIDE, "", {}, ""),
    "dataset_and_channel_organization": ("image", "two_channel", {}, ""),
    "image_restoration": ("image", "cells2d", {}, ""),
    "background_and_intensity_correction": ("image", "cells2d", {}, ""),
    "image_filtering": ("image", "cells2d", {}, ""),
    "registration_and_alignment": ("image", "timelapse_drift", {}, ""),
    "geometric_transformation": ("image", "cells2d", {}, ""),
    "projection_and_stitching": ("image", "cells3d", {}, ""),
    "segmentation_and_labeling": ("labels", "cells2d", {}, ""),
    "object_detection": ("points", "cells2d", {"puncta": True}, ""),
    "shape_and_distance_geometry": ("labels", "cells2d", {}, "labels2d"),
    "representation_transfer": ("labels", "cells2d", {}, "labels2d"),
    "measurement_and_statistics": ("table", "cells2d", {}, "labels2d"),
    "object_filtering_and_selection": ("labels", "cells2d", {}, "measured2d"),
    "table_synthesis": ("table", "cells2d", {}, "measured2d"),
    "tracking": ("tracks", "moving_cells", {}, "labels2d"),
    "motion_and_deformation_fields": ("field", "speckle_pair", {}, ""),
    "control_flow": (GUIDE, "", {}, ""),
    "graph_structure": (GUIDE, "", {}, ""),
    "visualization": ("image", "cells2d", {}, ""),
    "plotting": ("plot", "cells2d", {}, "measured2d"),
    "page_boundary": (GUIDE, "", {}, ""),
    "arithmetic": ("image", "cells2d", {}, ""),
}


def _ensure_ops() -> None:
    from nodelab_v2.ops import ensure_ops
    ensure_ops()


#: the module that registers the GUI-layer ops (``io.load``, ``view.viewer``, ``io.dock``,
#: ``page.*``, ``data.part``) — demoed as guides beside the catalog
_GUI_OPS_OWNER = "nodelab_v2.ops"


def all_demo_ops() -> List[str]:
    """Every SHIPPED op — the catalog (:func:`nodegraph.hotreload.is_catalog_op`) plus the
    GUI-layer ops — and nothing else. ``NODES`` is process-global, so a test fixture
    registered by an earlier selftest would otherwise walk into the gate as an op with no
    role and no demo (INV-03's other face)."""
    from nodegraph.hotreload import is_catalog_op
    _ensure_ops()
    return sorted(op for op in NODES.keys()
                  if is_catalog_op(op) or NODES.owner(op) == _GUI_OPS_OWNER)


def _steps_of(value: Any, named: Mapping[str, Tuple[PreludeStep, ...]]) -> Tuple[PreludeStep, ...]:
    """A prelude as the JSON spells it: a name, or a list of names / ``{op, params, modes}``."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        if value not in named:
            raise KeyError(f"unknown prelude {value!r}")
        return tuple(named[value])
    out: List[PreludeStep] = []
    for item in value:
        if isinstance(item, str):
            out.extend(_steps_of(item, named))
        else:
            out.append(PreludeStep(str(item["op"]), dict(item.get("params") or {}),
                                   {k: str(v) for k, v in (item.get("modes") or {}).items()}))
    return tuple(out)


@lru_cache(maxsize=1)
def load_curation() -> Dict[str, Any]:
    """The parsed ``codemap/node_demos.json`` (``{}`` sections when the file is missing)."""
    try:
        with open(DEMOS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data.setdefault("preludes", {})
    data.setdefault("ops", {})
    return data


def reload_curation() -> Dict[str, Any]:
    load_curation.cache_clear()
    recipe_for.cache_clear()
    return load_curation()


def _named_preludes(cur: Mapping[str, Any]) -> Dict[str, Tuple[PreludeStep, ...]]:
    """The built-in preludes plus the file's, resolved in dependency order rather than
    file order so an entry may name one declared below it."""
    named = dict(PRELUDES)
    pending = dict(cur.get("preludes") or {})
    while pending:
        progressed = False
        for name in list(pending):
            try:
                named[name] = _steps_of(pending[name], named)
            except KeyError:
                continue
            del pending[name]
            progressed = True
        if not progressed:
            raise KeyError(f"preludes reference unknown names: {sorted(pending)}")
    return named


def role_default(op_key: str) -> DemoRecipe:
    """The recipe the node's role alone implies."""
    role, stage = R.role_of(op_key)
    kind, ph, kw, pre = ROLE_DEFAULTS.get(role, (GUIDE, "", {}, ""))
    return DemoRecipe(op=op_key, kind=kind, phantom=ph, phantom_kw=dict(kw),
                      prelude=PRELUDES.get(pre, ()), role=role, stage=stage)


@lru_cache(maxsize=256)
def recipe_for(op_key: str) -> DemoRecipe:
    """The node's demo: its role default merged with the curated entry, if any."""
    _ensure_ops()
    base = role_default(op_key)
    if NODES.get(op_key) is None:
        return replace(base, kind=GUIDE, phantom="", prelude=(),
                       features=("This op is not in the registry — the graph may come from "
                                 "a newer build or a plugin that is not loaded.",))
    cur = load_curation()
    entry = (cur.get("ops") or {}).get(op_key)
    if not entry:
        return base
    named = _named_preludes(cur)
    kw: Dict[str, Any] = {}
    if "kind" in entry:
        kw["kind"] = str(entry["kind"])
    if "phantom" in entry:
        kw["phantom"] = str(entry["phantom"] or "")
        kw["phantom_kw"] = {}
    if "phantom_kw" in entry:
        kw["phantom_kw"] = dict(entry["phantom_kw"] or {})
    if "prelude" in entry:
        kw["prelude"] = _steps_of(entry["prelude"], named)
    if "data_from" in entry:
        kw["data_from"] = str(entry["data_from"])
    if "extra_inputs" in entry:
        kw["extra_inputs"] = {str(k): dict(v) for k, v in (entry["extra_inputs"] or {}).items()}
    if "fixed_params" in entry:
        kw["fixed_params"] = dict(entry["fixed_params"] or {})
    if "fixed_modes" in entry:
        kw["fixed_modes"] = {k: str(v) for k, v in (entry["fixed_modes"] or {}).items()}
    if "sliders" in entry:
        kw["sliders"] = {k: (float(v[0]), float(v[1]))
                         for k, v in (entry["sliders"] or {}).items()}
    if "live" in entry:
        kw["live"] = bool(entry["live"])
    if "slow_modes" in entry:
        kw["slow_modes"] = {k: tuple(str(c) for c in v)
                            for k, v in (entry["slow_modes"] or {}).items()}
    if "slow_reason" in entry:
        kw["slow_reason"] = str(entry["slow_reason"])
    if "gate_params" in entry:
        kw["gate_params"] = dict(entry["gate_params"] or {})
    if "requires" in entry:
        kw["requires"] = tuple(str(r) for r in (entry["requires"] or ()))
    if "view" in entry:
        kw["view"] = {k: int(v) for k, v in (entry["view"] or {}).items()}
    if "features" in entry:
        kw["features"] = tuple(str(f) for f in (entry["features"] or ()))
    if "scenarios" in entry:
        kw["scenarios"] = tuple(_scenario_of(sc) for sc in (entry["scenarios"] or ()))
        if "phantom" not in entry:
            kw["phantom"] = ""            # the first scenario is the default world
            kw["phantom_kw"] = {}
    rec = replace(base, **kw)
    if rec.kind == GUIDE:
        rec = replace(rec, phantom="", prelude=(), scenarios=())
    return rec


def _scenario_of(sc: Mapping[str, Any]) -> Scenario:
    return Scenario(label=str(sc.get("label", "")), phantom=str(sc.get("phantom", "")),
                    phantom_kw=dict(sc.get("phantom_kw") or {}),
                    view={k: int(v) for k, v in (sc.get("view") or {}).items()},
                    fixed_params=dict(sc.get("fixed_params") or {}),
                    fixed_modes={k: str(v) for k, v in (sc.get("fixed_modes") or {}).items()},
                    note=str(sc.get("note", "")))


def scenario_recipe(rec: DemoRecipe, index: int) -> DemoRecipe:
    """``rec`` with its ``index``-th scenario applied: that world's phantom, view and fixed
    values laid over the entry's own. A recipe without scenarios is returned unchanged, so
    callers need not branch; an index out of range clamps."""
    if not rec.scenarios:
        return rec
    i = min(max(0, int(index)), len(rec.scenarios) - 1)
    sc = rec.scenarios[i]
    return replace(rec, phantom=sc.phantom, phantom_kw=dict(sc.phantom_kw),
                   view=dict(sc.view) if sc.view else dict(rec.view),
                   fixed_params={**rec.fixed_params, **sc.fixed_params},
                   fixed_modes={**rec.fixed_modes, **sc.fixed_modes},
                   scenario_index=i)


def validate_curation(cur: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Every way the curated file can disagree with the registry, as messages (``[]`` = fine).
    Checked by the selftest so a renamed socket or a dropped op fails loudly."""
    _ensure_ops()
    cur = load_curation() if cur is None else cur
    problems: List[str] = []
    try:
        named = _named_preludes(cur)
    except KeyError as exc:
        return [f"preludes: {exc}"]
    for name, steps in named.items():
        for st in steps:
            spec = NODES.get(st.op)
            if spec is None:
                problems.append(f"prelude {name!r} names unknown op {st.op!r}")
                continue
            for p in st.params:
                if spec.input(p) is None:
                    problems.append(f"prelude {name!r}: {st.op} has no param {p!r}")
            for mname, choice in st.modes.items():
                m = next((x for x in spec.modes if x.name == mname), None)
                if m is None:
                    problems.append(f"prelude {name!r}: {st.op} has no mode {mname!r}")
                elif choice not in m.choices:
                    problems.append(f"prelude {name!r}: {st.op} mode {mname!r} has no "
                                    f"choice {choice!r}")
    for op, entry in (cur.get("ops") or {}).items():
        spec = NODES.get(op)
        if spec is None:
            problems.append(f"ops names unknown op {op!r}")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{op}: entry must be an object")
            continue
        kind = entry.get("kind")
        if kind is not None and kind not in KINDS:
            problems.append(f"{op}: unknown kind {kind!r} (one of {', '.join(KINDS)})")
        ph = entry.get("phantom")
        if ph and ph not in PHANTOMS:
            problems.append(f"{op}: unknown phantom {ph!r}")
        try:
            _steps_of(entry.get("prelude"), named)
        except (KeyError, TypeError) as exc:
            problems.append(f"{op}: prelude — {exc}")
        if entry.get("data_from", "tail") not in ("tail", "src"):
            problems.append(f"{op}: data_from must be 'tail' or 'src'")
        for sock, src in (entry.get("extra_inputs") or {}).items():
            s = spec.input(sock)
            if s is None or s.type is not SocketType.DATASET:
                problems.append(f"{op}: extra_inputs names {sock!r}, not a Dataset input")
            if not isinstance(src, dict) or not ({"phantom", "pre"} & set(src)):
                problems.append(f"{op}: extra_inputs[{sock!r}] must be {{phantom}} or {{pre}}")
            elif "phantom" in src and src["phantom"] not in PHANTOMS:
                problems.append(f"{op}: extra_inputs[{sock!r}] unknown phantom {src['phantom']!r}")
        for p in list(entry.get("fixed_params") or {}) + list(entry.get("gate_params") or {}):
            if spec.input(p) is None:
                problems.append(f"{op}: no param {p!r}")
        for mname, choice in (entry.get("fixed_modes") or {}).items():
            m = next((x for x in spec.modes if x.name == mname), None)
            if m is None:
                problems.append(f"{op}: no mode {mname!r}")
            elif str(choice) not in m.choices:
                problems.append(f"{op}: mode {mname!r} has no choice {choice!r}")
        for mname, choices in (entry.get("slow_modes") or {}).items():
            m = next((x for x in spec.modes if x.name == mname), None)
            if m is None:
                problems.append(f"{op}: slow_modes names no mode {mname!r}")
            else:
                bad = [c for c in choices if c not in m.choices]
                if bad:
                    problems.append(f"{op}: slow_modes[{mname!r}] unknown choice(s) {bad}")
        for p, span in (entry.get("sliders") or {}).items():
            s = spec.input(p)
            if s is None or s.type not in (SocketType.INT, SocketType.FLOAT):
                problems.append(f"{op}: sliders names {p!r}, not a numeric param")
            elif not (isinstance(span, (list, tuple)) and len(span) == 2
                      and float(span[1]) > float(span[0])):
                problems.append(f"{op}: sliders[{p!r}] must be [lo, hi] with hi > lo")
        if entry.get("live") is False and not entry.get("slow_reason"):
            problems.append(f"{op}: live=false needs a slow_reason")
        feats = entry.get("features")
        if feats is not None and not (isinstance(feats, list)
                                      and all(isinstance(f, str) for f in feats)):
            problems.append(f"{op}: features must be a list of strings")
        req = entry.get("requires")
        if req is not None and not (isinstance(req, list)
                                    and all(isinstance(r, str) and r for r in req)):
            problems.append(f"{op}: requires must be a list of module names")
        for k in ("view",):
            v = entry.get(k)
            if v is not None and not (isinstance(v, dict)
                                      and set(v) <= {"t", "z", "c", "m"}):
                problems.append(f"{op}: view must be an object with t/z/c/m keys")
        scs = entry.get("scenarios")
        if scs is not None:
            if not (isinstance(scs, list) and scs and all(isinstance(s, dict) for s in scs)):
                problems.append(f"{op}: scenarios must be a non-empty list of objects")
            else:
                seen: set = set()
                for i, sc in enumerate(scs):
                    label = str(sc.get("label", "")).strip()
                    if not label:
                        problems.append(f"{op}: scenarios[{i}] needs a label")
                    elif label in seen:
                        problems.append(f"{op}: scenarios[{i}] duplicates the label {label!r}")
                    seen.add(label)
                    if sc.get("phantom") not in PHANTOMS:
                        problems.append(f"{op}: scenarios[{i}] unknown phantom "
                                        f"{sc.get('phantom')!r}")
                    for p in (sc.get("fixed_params") or {}):
                        if spec.input(p) is None:
                            problems.append(f"{op}: scenarios[{i}] fixes unknown param {p!r}")
                    for mname, choice in (sc.get("fixed_modes") or {}).items():
                        m = next((x for x in spec.modes if x.name == mname), None)
                        if m is None:
                            problems.append(f"{op}: scenarios[{i}] fixes unknown mode {mname!r}")
                        elif str(choice) not in m.choices:
                            problems.append(f"{op}: scenarios[{i}] mode {mname!r} has no "
                                            f"choice {choice!r}")
                    v = sc.get("view")
                    if v is not None and not (isinstance(v, dict)
                                              and set(v) <= {"t", "z", "c", "m"}):
                        problems.append(f"{op}: scenarios[{i}] view must have t/z/c/m keys")
                    if not isinstance(sc.get("note", ""), str):
                        problems.append(f"{op}: scenarios[{i}] note must be a string")
            if kind == GUIDE:
                problems.append(f"{op}: a guide cannot have scenarios")
    return problems


# ── which controls a socket gets, and over what span ──────────────────────────

def control_kind(s: Any) -> str:
    """``slider`` / ``check`` / ``choice`` / ``readonly`` / ``hidden`` for one input socket.

    Hidden: wired Datasets, presentation-only sockets (the compute never reads them) and
    free text (layer names stay at their defaults so the prelude lines up). Read-only: a
    file path, a drawn shape list, a layer / column picker, a tick list — shown as the
    recipe fixed them."""
    t = getattr(s, "type", None)
    if t is SocketType.DATASET or getattr(s, "presentation", False):
        return "hidden"
    if (getattr(s, "path_kind", "") or getattr(s, "pick_kind", "") == "shapes"
            or getattr(s, "layer_in", None) is not None or getattr(s, "layer_in_mode", "")
            or getattr(s, "column_in", None) is not None or getattr(s, "column_in_mode", "")
            or getattr(s, "layer_out", ()) or getattr(s, "vocab", ())):
        return "readonly"
    if t is SocketType.STRING:
        return "choice" if getattr(s, "choices", ()) else "hidden"
    if t is SocketType.BOOL:
        return "check"
    if t in (SocketType.INT, SocketType.FLOAT):
        return "slider"
    return "hidden"


def _nice_ceil(x: float) -> float:
    """``x`` rounded UP to a 1-2-5 mantissa: 0.5 → 0.5, 3.2 → 5, 47 → 50, 0.15 → 0.2."""
    if not (x > 0) or not math.isfinite(x):
        return 1.0
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    for cand in (1.0, 2.0, 5.0, 10.0):
        if m <= cand + 1e-9:
            return cand * 10 ** e
    return 10.0 ** (e + 1)


_ITER_RE = re.compile(r"iter|passes|n_init|n_neighbors|num_warp|epochs|units")
_SIZE_RE = re.compile(r"size|window|block|radius|patch|tile|grid")
_UNIT_FLOAT_NOT_FRACTION = frozenset({"amount", "scale", "h", "weight", "gain", "mu", "beta",
                                      "upsample", "alpha", "sigma_color"})


def slider_range(s: Any, value: Any = None, *, env: Optional[MetaEnvelope] = None,
                 ph: Optional[Phantom] = None,
                 override: Optional[Tuple[float, float]] = None) -> SliderRange:
    """A deterministic ``(lo, hi, step, decimals)`` for a numeric socket with no declared
    bounds — from its unit, its pick gesture and the magnitude of its starting value, bounded
    by the phantom's own axes and histogram where the quantity is an index or a level."""
    integer = getattr(s, "type", None) is SocketType.INT
    d: Optional[float]
    try:
        d = float(value if value is not None else getattr(s, "default", None))
        if not math.isfinite(d):
            d = None
    except (TypeError, ValueError):
        d = None
    dm = abs(d) if d not in (None, 0.0) else None     # a usable magnitude, or None
    ax = env.axes if env is not None else (ph.axes if ph is not None else AxisSizes())
    md = dict(env.metadata or {}) if env is not None else {}
    px = float(md.get("pixel_size_um") or 0.325)
    zs = float(md.get("z_step_um") or 1.0)
    name = str(getattr(s, "name", "") or "")
    unit = str(getattr(s, "unit", "") or "")
    pk = str(getattr(s, "pick_kind", "") or "")
    lo, hi = 0.0, 1.0
    signed = name.startswith(("shift_", "offset_", "t_shift")) or name == "angle" \
        or (d is not None and d < 0)
    if override is not None:
        lo, hi = float(override[0]), float(override[1])
    elif pk == "level":
        if ph is not None:
            rng = intensity_range(ph)
            lo, hi = rng[0], rng[3]
        else:
            lo, hi = 0.0, max(1.0, (dm or 0.5) * 2)
    elif pk == "percentile" or name.endswith("_pct") or name.startswith("percentile"):
        lo, hi = 0.0, 100.0
    elif pk == "gamma":
        lo, hi = 0.1, 5.0
    elif pk == "channel" or name in ("channel", "ref_c", "secondary_channel", "ref_channel"):
        lo, hi = 0.0, float(max(1, ax.c - 1))
    elif pk in ("frame", "frames") or name in ("reference_frame", "ref_t", "frame"):
        lo, hi = 0.0, float(max(1, ax.t - 1))
    elif pk == "plane" or name == "plane":
        lo, hi = 0.0, float(max(1, ax.z - 1))
    elif pk == "zrange" or name in ("z0", "z1"):
        lo, hi = 0.0, float(max(1, ax.z))
    elif pk == "rect" or name in ("y0", "y1", "x0", "x1"):
        lo, hi = 0.0, float(ax.y if name.startswith("y") else ax.x)
    elif unit == "um":
        hi = _nice_ceil(max(10.0 * (dm or 0.5), 20.0 * px))
    elif unit == "um_axial":
        hi = _nice_ceil(max(10.0 * (dm or 0.5), zs * max(1, ax.z)))
    elif unit in ("um2", "um3"):
        area = ax.y * ax.x * px * px
        hi = _nice_ceil(max(10.0 * (dm or 1.0), area / 20.0))
    elif unit == "px" and integer:
        lo = 1.0 if _SIZE_RE.search(name) else 0.0
        hi = float(min(max(8.0 * (dm or 4.0), 32.0), max(ax.y, ax.x)))
    elif unit == "px":
        hi = _nice_ceil(max(8.0 * (dm or 2.0), 16.0))
    elif unit == "nm":
        lo, hi = 400.0, 750.0
    elif unit == "s":
        hi = _nice_ceil(max(10.0 * (dm or 1.0), 10.0))
    elif unit == "pt":
        lo, hi = 4.0, 24.0
    elif unit == "mm":
        lo, hi = 20.0, 300.0
    elif unit == "dpi":
        lo, hi = 50.0, 600.0
    elif unit == "Pa":
        hi = _nice_ceil(max(10.0 * (dm or 1e4), 1e5))
    elif integer and _ITER_RE.search(name):
        lo, hi = 1.0, float(min(500.0, max(10.0 * (dm or 5.0), 50.0)))
    elif integer:
        lo = 1.0 if (d is not None and d >= 1) else 0.0
        hi = float(max(4.0 * (dm or 2.0), 10.0))
    elif d is not None and 0.0 < d <= 1.0 and name not in _UNIT_FLOAT_NOT_FRACTION:
        lo, hi = 0.0, 1.0
    elif dm is not None and dm > 10.0:
        hi = _nice_ceil(10.0 * dm)
    else:
        hi = _nice_ceil(max(10.0, 10.0 * (dm or 1.0)))
    if signed and override is None:
        if name == "angle":
            lo, hi = -180.0, 180.0
        else:
            a = max(4.0 * (dm or 5.0), 20.0)
            lo, hi = -a, a
    if d is not None:
        if d > hi:
            hi = _nice_ceil(d * 1.5)
        if d < lo:
            lo = d
    if hi <= lo:
        hi = lo + 1.0
    step = float(value_step(s, d, integer=integer))
    if integer:
        step = max(1.0, step)
        lo, hi = math.floor(lo), math.ceil(hi)
        decimals = 0
    else:
        decimals = int(value_decimals(s, d))
        # a span that the unit step would cross in > 1000 ticks gets a coarser step: the
        # spin box keeps the fine one, the slider only needs to be scrubbable
        if (hi - lo) / step > 1000.0:
            step = _nice_ceil((hi - lo) / 1000.0)
    return SliderRange(float(lo), float(hi), float(step), decimals)


def derived_defaults(spec: Any, env: MetaEnvelope, *, channel_index: int = 0) -> Dict[str, Any]:
    """``param → value`` for every input socket with a ``derive``, evaluated against the
    incoming envelope — where a slider starts, never what is sent (the engine derives for
    itself when the param is absent)."""
    out: Dict[str, Any] = {}
    try:
        syms = envelope_symbols(env, channel_index)
    except Exception:                          # noqa: BLE001 — a bad envelope derives nothing
        return out
    for s in spec.inputs:
        expr = getattr(s, "derive", "")
        if not expr:
            continue
        try:
            v = eval_derive(expr, syms)
        except Exception:                      # noqa: BLE001 — the static default stands
            continue
        if v is not None:
            out[s.name] = v
    return out


def derived_modes(spec: Any, env: MetaEnvelope) -> Dict[str, str]:
    """Every mode's resolved default for this envelope, the 2D/3D lever included."""
    from nodegraph.metadata import resolve_dim_default
    state = dict(spec.default_state())
    try:
        syms = envelope_symbols(env)
    except Exception:                          # noqa: BLE001
        syms = {}
    for m in spec.modes:
        if bool(getattr(m, "is_dim_lever", False)):
            v = resolve_dim_default(spec, env)
            if v:
                state[m.name] = v
            continue
        expr = getattr(m, "derive", "")
        if expr:
            try:
                v = eval_derive(expr, syms)
            except Exception:                  # noqa: BLE001
                v = None
            if v in m.choices:
                state[m.name] = str(v)
    return state


# ── the guide text (shared with the palette overview) ─────────────────────────

def _squash(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


def _html_escape(s: Any) -> str:
    import html
    return html.escape(str(s))


def long_description(op_key: str, *, limit: int = 1600) -> str:
    """The compute's docstring, else the owning module's — the "How it works" prose."""
    import sys
    try:
        from nodegraph.nodes import COMPUTES
        long = _squash(getattr(COMPUTES.get(op_key), "__doc__", "") or "")
        if not long:
            owner = NODES.owner(op_key) or ""
            long = _squash(getattr(sys.modules.get(owner), "__doc__", "") or "")
    except Exception:                          # noqa: BLE001 — prose is never fatal
        long = ""
    if limit and len(long) > limit:
        long = long[:limit].rsplit(" ", 1)[0] + " …"
    return long


def guide_html(op_key: str, *, css: str = "", dot: Optional[Callable[[Any], str]] = None,
               socket_color: Optional[Callable[[Any], str]] = None,
               domain_color: Optional[Callable[[Any], str]] = None,
               features: Sequence[str] = (), with_gestures: bool = True,
               head: bool = True) -> str:
    """The node overview as HTML: label, role › stage, description, sockets (with the pick
    gesture each one offers), modes, footprint, "How it works", and the curated features.

    One builder for the palette's Overview card and the demo window's guide, so the two can
    never disagree. Colours are injected (``socket_color(SocketType) -> '#rrggbb'``) so this
    module stays Qt-free."""
    _ensure_ops()
    from nodelab_v2.picker import PICK_ACTION, PICK_HELP
    spec = NODES.get(op_key)
    e = _html_escape
    if spec is None:
        return css + f"<h3>{e(op_key)}</h3><div>Not in the registry.</div>"
    rk, sk = R.role_of(op_key)
    rmeta, smeta = R.role_meta(rk), R.stage_meta(sk)
    state = spec.default_state()
    sc = socket_color or (lambda _t: "#9aa3ad")
    dc = domain_color or (lambda _d: "#9aa3ad")
    dot_of = dot or (lambda col: f'<span style="color:{col}; font-size:13px;">&#9679;</span>')

    def sock_row(s: Any, direction: str) -> str:
        if s.type is SocketType.DATASET:
            doms = (sorted(d.value for d in spec.reads_domains) if direction == "in"
                    else sorted(d.value for d in spec.adds_domains))
            what = "Dataset" + (f" · {'reads' if direction == 'in' else 'adds'} "
                                + ", ".join(doms) if doms else "")
            return (f"<tr><td>{dot_of(sc(SocketType.DATASET))}</td><td><b>{e(s.label or s.name)}"
                    f"</b></td><td class='k'>{e(what)}</td></tr>")
        extra: List[str] = []
        if s.unit:
            extra.append(e(s.unit))
        if s.default is not None and direction == "in":
            extra.append(f"default {e(str(s.default))}")
        if getattr(s, "layer_in", None) is not None:
            extra.append(f"picks a {s.layer_in.value} layer")
        if getattr(s, "layer_out", ()):
            extra.append("names a layer it writes")
        pk = getattr(s, "pick_kind", "")
        if with_gestures and pk:
            extra.append(f"<i>{e(PICK_ACTION.get(pk, 'Pick'))}</i> — {e(PICK_HELP.get(pk, ''))}")
        return (f"<tr><td>{dot_of(sc(s.type))}</td>"
                f"<td><b>{e(s.label or s.name)}</b> <span class='k'>{e(s.type.value)}"
                f"</span></td><td class='k'>{' · '.join(extra)}</td></tr>")

    ins = "".join(sock_row(s, "in") for s in spec.active_inputs(state))
    outs = "".join(sock_row(s, "out") for s in spec.outputs)
    modes = "".join(
        f"<li><b>{e(m.label or m.name)}</b>: {e(', '.join(m.choices))} "
        f"<span class='k'>(default {e(m.resolved_default())})</span>"
        + (f"<div class='k'>{e(_squash(m.description))}</div>" if m.description else "")
        + "</li>"
        for m in spec.modes if m.active_in(state))
    gran = spec.granularity
    if isinstance(gran, dict):
        gran_s = ", ".join(f"{k}: {getattr(v, 'value', v)}" for k, v in gran.items())
    else:
        gran_s = getattr(gran, "value", str(gran)) if gran is not None else "—"
    dims: List[str] = []
    if spec.supports_2d:
        dims.append("2D")
    if spec.supports_true_3d:
        dims.append("true 3D")
    elif spec.three_d_fallback:
        dims.append(f"3D as {spec.three_d_fallback.replace('_', ' ')}")
    long = long_description(op_key)
    feats = "".join(f"<li>{e(f)}</li>" for f in features)
    parts = [css]
    if head:
        parts.append(f"<h3>{e(spec.label)}</h3><div class='op'>{e(op_key)}</div>"
                     f"<div class='k'>{e(str(smeta.get('label', '')))} › "
                     f"<b>{e(str(rmeta.get('label', rk)))}</b></div>"
                     f"<div style='margin-top:4px'>{e(_squash(spec.description))}</div>")
    if feats:
        parts.append(f"<div class='sec'>Key features · how to use</div><ul>{feats}</ul>")
    parts.append(f"<div class='sec'>In</div><table>{ins or '<tr><td class=k>— (source)</td></tr>'}"
                 f"</table><div class='sec'>Out</div><table>{outs}</table>")
    if modes:
        parts.append(f"<div class='sec'>Modes</div><ul>{modes}</ul>")
    parts.append(f"<div class='sec'>Footprint</div><div class='k'>{e(gran_s)}"
                 + (f" · {e(' / '.join(dims))}" if dims else "") + "</div>")
    if long:
        parts.append(f"<div class='sec'>How it works</div><div>{e(long)}</div>")
    return "".join(parts)


# ── running a demo ─────────────────────────────────────────────────────────────

@dataclass
class DemoResult:
    """What one run produced, already reduced to the viewed plane."""
    before: np.ndarray                                     # (Y, X) float
    axes_before: AxisSizes
    axes_after: Optional[AxisSizes] = None
    after_image: Optional[np.ndarray] = None               # (Y', X') float
    after_dataset: Any = None
    label_plane: Optional[np.ndarray] = None               # int (Y', X')
    mask_plane: Optional[np.ndarray] = None                # 0/1 (Y', X')
    scalar_plane: Optional[np.ndarray] = None              # float (Y', X')
    #: ``(y, x, key, layer_index, on_plane, zplane)`` per point on the viewed frame
    points: List[Tuple[float, float, int, int, bool, int]] = field(default_factory=list)
    #: ``(track_id, [(y, x), …], current_index | None, [t, …])`` per track
    tracks: List[Tuple[int, List[Tuple[float, float]], Optional[int], List[int]]] = \
        field(default_factory=list)
    vectors: Optional[np.ndarray] = None                   # (N, 4) y, x, u, v
    #: ``(object_id, [(y, x), …])`` vertices near the viewed plane, per mesh element
    mesh: List[Tuple[int, List[Tuple[float, float]]]] = field(default_factory=list)
    tables: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = field(default_factory=dict)
    #: the layers the demo node itself wrote (by its ``layer_out`` sockets)
    written: Tuple[str, ...] = ()
    value: Any = None                                      # a non-Dataset payload
    view: Tuple[int, int, int] = (0, 0, 0)                 # the (t, z, c) shown
    elapsed_s: float = 0.0
    note: str = ""
    #: Frame-domain layers the node wrote, read at the viewed ``(m, t)`` — a registration's
    #: ``drift_y`` / ``drift_x`` / ``drift_confidence``, so the window can show the transform
    #: it applied to the frame on screen next to the phantom's caption of the true motion
    frame_values: Dict[str, float] = field(default_factory=dict)


_VECTOR_COLUMNS = (("u", "v"), ("u_um", "v_um"), ("dy", "dx"), ("disp_y", "disp_x"))


class DemoSession:
    """One node's demo: owns the phantom, a persistent memo and tile cache, and runs
    ``seed → prelude → node`` with whatever the sliders say. Qt-free; the window drives it
    from a worker thread, the selftest from the gate."""

    def __init__(self, recipe: DemoRecipe, *, memo_bytes: int = 96 << 20,
                 tile_bytes: int = 128 << 20) -> None:
        _ensure_ops()
        self.recipe = recipe
        self.spec = NODES.get(recipe.op)
        if self.spec is None:
            raise KeyError(f"{recipe.op!r} is not registered")
        if recipe.is_guide:
            raise ValueError(f"{recipe.op!r} is a guide-only demo; nothing to run")
        if recipe.scenarios and (not recipe.phantom or recipe.scenario_index < 0):
            recipe = scenario_recipe(recipe, 0)   # the first scenario is the default world
            self.recipe = recipe
        self.phantom: Phantom = phantom(recipe.phantom, **dict(recipe.phantom_kw))
        self.memo = Memo(budget_bytes=memo_bytes)
        self.tiles = TileCache(tile_bytes)
        self._env_in: Optional[MetaEnvelope] = None
        self._state0: Optional[Dict[str, str]] = None

    # ── graph construction ───────────────────────────────────────────────────
    def primary_socket(self, state: Optional[Mapping[str, str]] = None) -> Optional[str]:
        st = dict(state) if state is not None else self.spec.default_state()
        for s in self.spec.active_inputs(st):
            if s.type is SocketType.DATASET:
                return s.name
        return None

    def build_graph(self, params: Mapping[str, Any], modes: Mapping[str, str]
                    ) -> Tuple[Graph, Dict[str, Any], Dict[str, MetaEnvelope]]:
        """The run graph: ``src`` (an ``io.load`` seeded with the phantom, as the GUI seeds
        one) → the prelude → ``demo``, plus any extra inputs the recipe wires."""
        rec = self.recipe
        g = Graph()
        g.add(NodeInstance("src", "io.load", params={"path": ""}))
        seeds: Dict[str, Any] = {"src": self.phantom.dataset}
        metas: Dict[str, MetaEnvelope] = {"src": self.phantom.envelope}
        prev, prev_sock = "src", "image"
        for i, st in enumerate(rec.prelude):
            nid = f"pre{i}"
            g.add(NodeInstance(nid, st.op, params=dict(st.params), modes=dict(st.modes)))
            g.connect(prev, nid, src_socket=prev_sock, dst_socket="data")
            prev, prev_sock = nid, "out"
        state = {**self.spec.default_state(), **rec.fixed_modes, **modes}
        g.add(NodeInstance("demo", rec.op, params={**rec.fixed_params, **params},
                           modes=dict(state)))
        primary = self.primary_socket(state)
        if primary is not None:
            if rec.data_from == "src":
                g.connect("src", "demo", src_socket="image", dst_socket=primary)
            else:
                g.connect(prev, "demo", src_socket=prev_sock, dst_socket=primary)
        for dst, src in rec.extra_inputs.items():
            if "phantom" in src:
                nid = f"src_{dst}"
                ph = phantom(str(src["phantom"]), **dict(src.get("kw") or {}))
                g.add(NodeInstance(nid, "io.load", params={"path": ""}))
                seeds[nid] = ph.dataset
                metas[nid] = ph.envelope
                g.connect(nid, "demo", src_socket="image", dst_socket=dst)
            elif "pre" in src:
                g.connect(f"pre{int(src['pre'])}", "demo", src_socket="out", dst_socket=dst)
        return g, seeds, metas

    def envelope(self) -> MetaEnvelope:
        """The envelope ARRIVING at the demo node with everything at its defaults — what the
        sliders' derived starting values and index-like spans are computed from."""
        if self._env_in is None:
            g, _seeds, metas = self.build_graph({}, {})
            try:
                envs = propagate_meta(g, metas)
            except ValueError:
                envs = {}
            primary = self.primary_socket()
            src = None
            for edge in g.edges:
                if edge.dst == "demo" and (primary is None or edge.dst_socket == primary):
                    src = edge.src
                    break
            self._env_in = envs.get(src) if src else None
            if self._env_in is None:
                self._env_in = self.phantom.envelope
        return self._env_in

    def default_state(self) -> Dict[str, str]:
        if self._state0 is None:
            self._state0 = {**derived_modes(self.spec, self.envelope()),
                            **self.recipe.fixed_modes}
        return dict(self._state0)

    def starting_values(self) -> Dict[str, Any]:
        """``param → value`` every editable socket starts at: the recipe's fixed value, else
        the derived one, else the socket default."""
        out: Dict[str, Any] = {}
        derived = derived_defaults(self.spec, self.envelope())
        for s in self.spec.inputs:
            if s.type is SocketType.DATASET:
                continue
            if s.name in self.recipe.fixed_params:
                out[s.name] = self.recipe.fixed_params[s.name]
            elif s.name in derived:
                out[s.name] = derived[s.name]
            else:
                out[s.name] = s.default
        return out

    def range_for(self, s: Any, value: Any = None) -> SliderRange:
        return slider_range(s, value, env=self.envelope(), ph=self.phantom,
                            override=self.recipe.sliders.get(s.name))

    def missing_requirements(self) -> List[str]:
        """The optional packages this demo needs that are not installed."""
        import importlib.util
        out: List[str] = []
        for mod in self.recipe.requires:
            try:
                found = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                out.append(mod)
        return out

    def is_slow(self, modes: Mapping[str, str]) -> bool:
        """Whether this mode state is one the recipe marks as too slow for live updates."""
        if not self.recipe.live:
            return True
        for mname, choices in self.recipe.slow_modes.items():
            if str(modes.get(mname, "")) in choices:
                return True
        return False

    # ── running ──────────────────────────────────────────────────────────────
    def view_coords(self) -> Tuple[int, int, int]:
        ax = self.phantom.axes
        v = self.recipe.view
        t = min(max(0, int(v.get("t", 0))), ax.t - 1)
        z = min(max(0, int(v.get("z", ax.z // 2))), ax.z - 1)
        c = min(max(0, int(v.get("c", 0))), ax.c - 1)
        return t, z, c

    def run(self, params: Mapping[str, Any], modes: Mapping[str, str]) -> DemoResult:
        """Pull the demo node with these values and reduce the payload to the viewed plane."""
        from nodelab_v2.ops import headless_engine
        t0 = time.perf_counter()
        g, seeds, metas = self.build_graph(params, modes)
        eng = headless_engine(g, seeds=seeds, meta_seeds=metas, memo=self.memo,
                              tiles=self.tiles)
        payload = eng.pull("demo")
        res = self._extract(payload, {**self.recipe.fixed_params, **params})
        res.elapsed_s = time.perf_counter() - t0
        return res

    # ── extraction ───────────────────────────────────────────────────────────
    def _written_layers(self, params: Mapping[str, Any]) -> Tuple[str, ...]:
        names: List[str] = []
        for s in self.spec.inputs:
            if getattr(s, "layer_out", ()):
                v = params.get(s.name, s.default)
                if isinstance(v, str) and v:
                    names.append(v)
        return tuple(dict.fromkeys(names))

    def _extract(self, payload: Any, params: Mapping[str, Any]) -> DemoResult:
        ph = self.phantom
        t, z, c = self.view_coords()
        before = np.asarray(ph.array[0, t, z, c], dtype=float)
        res = DemoResult(before=before, axes_before=ph.axes, view=(t, z, c))
        res.written = self._written_layers(params)
        if not isinstance(payload, Dataset):
            res.value = payload
            res.note = f"output: {type(payload).__name__}"
            return res
        ds = realize(payload)
        res.after_dataset = ds
        ax1 = ds.axes
        res.axes_after = ax1
        tt, zz, cc = min(t, ax1.t - 1), min(z, ax1.z - 1), min(c, ax1.c - 1)
        if ax1.z != ph.axes.z and ax1.z == 1:
            zz = 0
        m = 0
        if ds.image is not None:
            try:
                res.after_image = np.asarray(
                    ds.image.read_region(0, m, tt, zz, cc, 0, ax1.y, 0, ax1.x), dtype=float)
            except Exception as exc:              # noqa: BLE001 — shown, never fatal
                res.note = f"image unreadable: {exc}"
        vox = self._voxel_layers(ds)
        res.label_plane = self._pick_plane(vox, res.written, m, tt, zz, cc, want="labels")
        res.mask_plane = self._pick_plane(vox, res.written, m, tt, zz, cc, want="mask")
        res.scalar_plane = self._pick_plane(vox, res.written, m, tt, zz, cc, want="scalar")
        res.points = self._points(ds, m, tt, zz)
        res.tracks = self._tracks(ds, m, tt)
        res.vectors = self._vectors(ds, m, tt, zz, cc)
        res.mesh = self._mesh(ds, m, tt, zz, cc)
        from nodelab_v2.tables import structure_tables
        res.tables = structure_tables(ds)
        res.frame_values = self._frame_values(ds, m, tt)
        if (ax1.m, ax1.t, ax1.z, ax1.c, ax1.y, ax1.x) != (
                ph.axes.m, ph.axes.t, ph.axes.z, ph.axes.c, ph.axes.y, ph.axes.x):
            res.note = (f"axes (m,t,z,c,y,x): {ph.axes.m},{ph.axes.t},{ph.axes.z},{ph.axes.c},"
                        f"{ph.axes.y},{ph.axes.x} → {ax1.m},{ax1.t},{ax1.z},{ax1.c},{ax1.y},{ax1.x}")
        return res

    @staticmethod
    def _frame_values(ds: Dataset, m: int, t: int) -> Dict[str, float]:
        """Every Frame-domain layer's value at ``(m, t)``, in the order the node wrote them."""
        out: Dict[str, float] = {}
        for attr in ds.layers_on(Domain.FRAME):
            try:
                v = np.asarray(attr.values)
                if v.ndim >= 2 and v.shape[0] > m and v.shape[1] > t:
                    val = v[m, t]
                    if np.ndim(val) == 0 and np.isfinite(float(val)):
                        out[str(attr.name)] = float(val)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _voxel_layers(ds: Dataset) -> List[Tuple[str, np.ndarray]]:
        out: List[Tuple[str, np.ndarray]] = []
        for (dom, _layer, name), attr in ds.attributes.items():
            if dom is Domain.VOXEL:
                vals = np.asarray(attr.values)
                if vals.ndim == 6:
                    out.append((str(name), vals))
        return out

    @staticmethod
    def _pick_plane(vox: Sequence[Tuple[str, np.ndarray]], written: Sequence[str], m: int,
                    t: int, z: int, c: int, *, want: str) -> Optional[np.ndarray]:
        """The viewed plane of the best-matching Voxel layer: integer rasters are label maps
        (``labels``) unless they are 0/1 (``mask``); anything else is a ``scalar``. A layer
        the node itself wrote wins over one it merely carried."""
        cands: List[Tuple[int, str, np.ndarray]] = []
        for name, vals in vox:
            is_int = np.issubdtype(vals.dtype, np.integer) or vals.dtype == np.bool_
            try:
                plane = vals[m, t, z, min(c, vals.shape[3] - 1)]
            except IndexError:
                continue
            if is_int:
                mx = int(plane.max()) if plane.size else 0
                kind = "mask" if mx <= 1 else "labels"
                if want == "mask" and kind == "labels" and name in ("mask",):
                    kind = "mask"
            else:
                kind = "scalar"
            if kind != want:
                continue
            rank = 0 if name in written else (1 if name in ("labels", "mask") else 2)
            cands.append((rank, name, plane))
        if not cands:
            return None
        cands.sort(key=lambda x: (x[0], x[1]))
        plane = cands[0][2]
        if want == "scalar":
            return np.asarray(plane, dtype=float)
        return np.asarray(plane).astype(np.int64)

    @staticmethod
    def _points(ds: Dataset, m: int, t: int, z: int
                ) -> List[Tuple[float, float, int, int, bool, int]]:
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT:
                by_layer.setdefault(layer, {})[name] = attr.values
        out: List[Tuple[float, float, int, int, bool, int]] = []
        for li, layer in enumerate(sorted(by_layer, key=lambda v: str(v))):
            cols = by_layer[layer]
            if "y" not in cols or "x" not in cols:
                continue
            ys, xs = np.asarray(cols["y"], float), np.asarray(cols["x"], float)
            zs = cols.get("z"); ms = cols.get("m"); ts = cols.get("t"); ids = cols.get("id")
            for i in range(len(ys)):
                if ms is not None and int(ms[i]) != m:
                    continue
                if ts is not None and int(ts[i]) != t:
                    continue
                zpl = z if zs is None else int(round(float(zs[i])))
                key = int(ids[i]) if ids is not None else i + 1
                out.append((float(ys[i]), float(xs[i]), key, li, zpl == z, zpl))
        return out

    @staticmethod
    def _tracks(ds: Dataset, m: int, t: int
                ) -> List[Tuple[int, List[Tuple[float, float]], Optional[int], List[int]]]:
        tracks: Dict[Any, Dict[str, np.ndarray]] = {}
        members: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.TRACK:
                tracks.setdefault(layer, {})[name] = attr.values
            elif dom in (Domain.LABEL, Domain.POINT):
                members.setdefault((dom, layer), {})[name] = attr.values
        out: List[Tuple[int, List[Tuple[float, float]], Optional[int], List[int]]] = []
        for _tl, cols in tracks.items():
            if not {"track_id", "t", "member_id"} <= set(cols):
                continue
            tid = np.asarray(cols["track_id"]).astype(np.int64)
            tt = np.asarray(cols["t"]).astype(np.int64)
            mem = np.asarray(cols["member_id"]).astype(np.int64)
            best, best_cov = None, -1
            for key, mc in members.items():
                if not {"id", "y", "x"} <= set(mc):
                    continue
                cov = len(set(np.asarray(mc["id"]).astype(np.int64).tolist()) & set(mem.tolist()))
                if cov > best_cov:
                    best, best_cov = mc, cov
            if best is None:
                continue
            ids = np.asarray(best["id"]).astype(np.int64)
            ys = np.asarray(best["y"], float); xs = np.asarray(best["x"], float)
            ms = np.asarray(best["m"]).astype(np.int64) if "m" in best else None
            ts_m = np.asarray(best["t"]).astype(np.int64) if "t" in best else None
            # a member id is unique per (m, t); index by (t, id) when the table carries t
            index: Dict[Any, int] = {}
            for i, v in enumerate(ids.tolist()):
                k = (int(ts_m[i]), int(v)) if ts_m is not None else int(v)
                index.setdefault(k, i)
            for track in np.unique(tid).tolist():
                sel = tid == track
                order = np.argsort(tt[sel], kind="stable")
                path: List[Tuple[float, float]] = []
                cur: Optional[int] = None
                kept: List[int] = []
                for tv, mv in zip(tt[sel][order].tolist(), mem[sel][order].tolist()):
                    i = index.get((int(tv), int(mv)) if ts_m is not None else int(mv))
                    if i is None or (ms is not None and int(ms[i]) != m):
                        continue
                    path.append((float(ys[i]), float(xs[i])))
                    kept.append(int(tv))
                    if int(tv) == t:
                        cur = len(path) - 1
                if path:
                    out.append((int(track), path, cur, kept))
        return out

    @staticmethod
    def _vectors(ds: Dataset, m: int, t: int, z: int, c: int) -> Optional[np.ndarray]:
        by_layer: Dict[str, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT and layer:
                by_layer.setdefault(str(layer), {})[str(name)] = attr.values
        for layer, cols in by_layer.items():
            uy = ux = None
            for a, b in _VECTOR_COLUMNS:
                if a in cols and b in cols:
                    uy, ux = cols[a], cols[b]
                    break
            if uy is None or "y" not in cols or "x" not in cols:
                continue
            keep = np.ones(len(cols["y"]), dtype=bool)
            for key, want in (("m", m), ("t", t), ("c", c)):
                if key in cols:
                    keep &= np.asarray(cols[key]) == want
            try:
                z_kind = ds.structure_zkind(Domain.POINT, layer)
            except Exception:                  # noqa: BLE001 — provenance is advisory
                z_kind = None
            if z_kind == "plane_index" and "z" in cols:
                keep &= np.asarray(cols["z"]) == z
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            return np.stack([np.asarray(cols["y"], float)[idx], np.asarray(cols["x"], float)[idx],
                             np.asarray(uy, float)[idx], np.asarray(ux, float)[idx]], axis=1)
        return None

    @staticmethod
    def _mesh(ds: Dataset, m: int, t: int, z: int, c: int, *, near: float = 1.0
              ) -> List[Tuple[int, List[Tuple[float, float]]]]:
        from nodegraph.mesh import mesh_part
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.MESH and layer is not None:
                by_layer.setdefault(layer, {})[name] = attr.values
        out: List[Tuple[int, List[Tuple[float, float]]]] = []
        for layer in sorted(by_layer, key=lambda v: str(v)):
            base, part = mesh_part(str(layer))
            if part is not None:
                continue
            el = by_layer[layer]
            vt = by_layer.get(f"{base}/vert")
            if vt is None or not ({"id", "m", "t", "c", "vert_start", "vert_count"} <= set(el)
                                  and {"z", "y", "x"} <= set(vt)):
                continue
            vz, vy, vx = (np.asarray(vt[k], dtype=float) for k in ("z", "y", "x"))
            for row in range(len(np.asarray(el["id"]))):
                if (int(np.asarray(el["m"])[row]) != m or int(np.asarray(el["t"])[row]) != t
                        or int(np.asarray(el["c"])[row]) != c):
                    continue
                vs = int(np.asarray(el["vert_start"])[row])
                vc = int(np.asarray(el["vert_count"])[row])
                keep = np.abs(vz[vs:vs + vc] - float(z)) <= near
                verts = [(float(a), float(b)) for a, b in
                         zip(vy[vs:vs + vc][keep], vx[vs:vs + vc][keep])]
                out.append((int(np.asarray(el["id"])[row]), verts))
        return out


def demo_rows(res: DemoResult, *, prefer: Sequence[str] = ()) -> Tuple[Optional[Tuple[str, Optional[str]]],
                                                                       Dict[str, np.ndarray]]:
    """The one table a ``table`` demo shows: a structure table on a layer the node wrote,
    else the largest. ``(key, columns)`` or ``(None, {})``."""
    if not res.tables:
        return None, {}
    keys = list(res.tables)
    for name in list(prefer) + list(res.written):
        for k in keys:
            if k[1] == name:
                return k, res.tables[k]
    best = max(keys, key=lambda k: (len(next(iter(res.tables[k].values()), ())),
                                    len(res.tables[k])))
    return best, res.tables[best]
