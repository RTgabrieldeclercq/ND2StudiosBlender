"""Figure recipes — prebuilt FIGURE pages a figure can start from (V5.00 step 6).

A **figure recipe** is a figure document (:mod:`nodegraph.figure.schema`) with *slots* in
place of sources: ``{"id": "cells", "needs": "labels"}`` says "an output carrying a label
table", and the document's strings may name what a slot carries — ``"slot:cells"`` for the
Page Output value, ``"slot:cells.labels"`` for its first label layer, ``"slot:cells.points"``,
``"slot:cells.tracks"``, ``"slot:cells.fields"``, ``"slot:cells.fields.column"`` for that
layer's first measured column. :func:`bind_recipe` fills every placeholder from the figure
page's catalog (the rows :func:`nodelab_v2.figure_editor.sources.describe_envelope` builds,
with ``value`` and ``label``) and returns a document the editor commits as the page; a slot
nothing upstream satisfies is reported, not guessed.

Built-in recipes ship in ``nodelab_v2/builtin_figure_recipes/<slug>.json``; the user's own
live in ``~/.nd2studios/figure_recipes/`` and shadow a built-in of the same name. Qt-free:
the start card lists them, the panel applies them, the selftest binds every built-in to a
synthetic catalog.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure import schema as S

RECIPE_FORMAT = "nd2studios.figure-recipe/1"
BUILTIN_DIR = Path(__file__).resolve().parent / "builtin_figure_recipes"
USER_DIR = Path.home() / ".nd2studios" / "figure_recipes"
ENV_DIR = "NODELAB_FIGURE_RECIPES_DIR"
ENV_ENABLED = "NODELAB_FIGURE_RECIPES"

#: what a slot may need: a key of the catalog row's description
NEEDS = ("image", "labels", "points", "tracks", "fields", "mesh", "table")

_SLOT = re.compile(r"^slot:([A-Za-z_][A-Za-z0-9_]*)(?:\.(image|labels|points|tracks|fields|"
                   r"mesh|table)(?:\.(column|column2|name))?)?$")


@dataclass
class FigureRecipe:
    name: str
    description: str
    slots: List[Dict[str, Any]]
    figure: Dict[str, Any]
    path: Optional[str] = None
    builtin: bool = False
    goal: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return slugify(self.name)


def slugify(name: Any) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return s or "recipe"


def user_dir() -> Optional[Path]:
    if str(os.environ.get(ENV_ENABLED, "1")).strip().lower() in ("0", "off", "false", "no"):
        return None
    override = os.environ.get(ENV_DIR)
    return Path(override) if override else USER_DIR


def recipe_from_dict(d: Any, *, path: Optional[str] = None, builtin: bool = False
                     ) -> FigureRecipe:
    """A :class:`FigureRecipe` from a parsed file; ``ValueError`` for anything that is not
    one (the wrong tag, a slot with an unknown need, a figure the schema refuses once its
    placeholders are stubbed)."""
    if not isinstance(d, Mapping) or d.get("format") != RECIPE_FORMAT:
        raise ValueError(f"not a figure recipe (format {RECIPE_FORMAT!r} expected)")
    slots = list(d.get("slots") or [])
    if not slots:
        raise ValueError("a figure recipe needs at least one slot")
    ids = set()
    for k, s in enumerate(slots):
        if not isinstance(s, Mapping) or not isinstance(s.get("id"), str):
            raise ValueError(f"slots[{k}]: needs an id")
        if s.get("needs") not in NEEDS:
            raise ValueError(f"slots[{k}].needs: one of {NEEDS}, not {s.get('needs')!r}")
        if s["id"] in ids:
            raise ValueError(f"slots[{k}].id: duplicate {s['id']!r}")
        ids.add(s["id"])
    fig = d.get("figure")
    if not isinstance(fig, Mapping):
        raise ValueError("a figure recipe needs a `figure` document")
    # the schema sees a stub per slot: a document that fails here would fail on binding
    S.normalize(_stub(dict(fig), [str(s["id"]) for s in slots]))
    name = str(d.get("name") or "").strip() or (Path(path).stem if path else "recipe")
    return FigureRecipe(name, str(d.get("description") or "").strip(),
                        [dict(s) for s in slots], copy.deepcopy(dict(fig)), path, builtin,
                        str(d.get("goal") or fig.get("goal") or ""),
                        tuple(str(t) for t in (d.get("tags") or ())))


def _stub(fig: Dict[str, Any], slot_ids: Sequence[str]) -> Dict[str, Any]:
    """The figure with every placeholder replaced by a syntactically valid stand-in."""
    def sub(v: Any) -> Any:
        if isinstance(v, str):
            m = _SLOT.match(v)
            if m:
                if m.group(2) is None:
                    return f"pg0:{m.group(1)}"
                return "x"
            return v
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v
    out = sub(fig)
    srcs = out.get("sources") or []
    have = {s.get("id") for s in srcs}
    for sid in slot_ids:
        if sid not in have:
            srcs.append({"id": sid, "source": f"pg0:{sid}"})
    out["sources"] = srcs
    return out


def load_recipe(path: Any, *, builtin: bool = False) -> FigureRecipe:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return recipe_from_dict(d, path=str(path), builtin=builtin)


def _scan(folder: Optional[Path], *, builtin: bool) -> List[FigureRecipe]:
    out: List[FigureRecipe] = []
    if folder is None or not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(load_recipe(p, builtin=builtin))
        except Exception:                             # noqa: BLE001 — a damaged file is skipped
            continue
    return out


def list_recipes() -> List[FigureRecipe]:
    """Every figure recipe, built-ins first; a user recipe shadows a built-in of its name."""
    builtins = _scan(BUILTIN_DIR, builtin=True)
    users = _scan(user_dir(), builtin=False)
    names = {r.name.lower() for r in users}
    return [r for r in builtins if r.name.lower() not in names] + users


def find_recipe(name_or_slug: str) -> Optional[FigureRecipe]:
    key = str(name_or_slug).strip().lower()
    for r in list_recipes():
        if r.name.lower() == key or r.slug == key:
            return r
    return None


def recipe_to_dict(recipe: FigureRecipe) -> Dict[str, Any]:
    return {"format": RECIPE_FORMAT, "name": recipe.name, "description": recipe.description,
            "goal": recipe.goal, "tags": list(recipe.tags), "slots": recipe.slots,
            "figure": recipe.figure}


def save_recipe(recipe: FigureRecipe, folder: Optional[Path] = None) -> Path:
    folder = folder or user_dir()
    if folder is None:
        raise ValueError("the user's figure recipes folder is switched off")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{recipe.slug}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(recipe_to_dict(recipe), fh, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def recipe_from_figure(doc: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]], *,
                       name: str, description: str = "") -> FigureRecipe:
    """A recipe from a figure in hand: every source becomes a slot needing what its
    catalog row carries (labels over points over tracks over fields over an image), and
    the source values become placeholders. Layer names stay literal — the figure's own."""
    rows = {str(c.get("value")): c for c in catalog}
    fig = copy.deepcopy(dict(doc))
    slots = []
    for s in fig.get("sources") or []:
        row = rows.get(str(s.get("source")), {})
        needs = next((n for n in ("labels", "points", "tracks", "fields", "image")
                      if row.get(n)), "image")
        slots.append({"id": s["id"], "needs": needs, "label": str(row.get("label") or "")})
        s["source"] = f"slot:{s['id']}"
        s.pop("label", None)
    for key in ("_dismissed",):
        fig.pop(key, None)
    return FigureRecipe(name, description, slots, fig, None, False,
                        str(fig.get("goal") or ""))


# ── binding ───────────────────────────────────────────────────────────────────────
def _satisfies(row: Mapping[str, Any], needs: str) -> bool:
    if needs == "table":
        return bool(row.get("labels") or row.get("points") or row.get("tracks"))
    return bool(row.get(needs))


def applicable(recipe: FigureRecipe, catalog: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the catalog carries what every slot needs."""
    return not bind_recipe(recipe, catalog)[1]


def _first_layer(row: Mapping[str, Any], kind: str) -> Tuple[str, List[str]]:
    if kind == "table":
        for k in ("labels", "points", "tracks"):
            if row.get(k):
                return str(row[k][0][0]), list(row[k][0][1])
        return "", []
    if kind == "image":
        return "", []
    if kind == "mesh":
        return (str(row["mesh"][0]), []) if row.get("mesh") else ("", [])
    got = row.get(kind) or []
    return (str(got[0][0]), list(got[0][1])) if got else ("", [])


def _measured(cols: Sequence[str], skip_first: int = 0) -> str:
    skip = {"id", "m", "t", "c", "z", "y", "x", "b", "label", "parent", "track_id", "file",
            "group", "condition"}
    picks = [c for c in cols if c not in skip and not c.startswith("bbox")]
    for pref in ("mean_intensity", "speed", "area"):     # the usual suspects first
        if pref in picks:
            picks.remove(pref)
            picks.insert(0, pref)
    return picks[skip_first] if len(picks) > skip_first else ""


def bind_recipe(recipe: FigureRecipe, catalog: Sequence[Mapping[str, Any]],
                picks: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """``(figure document, unsatisfied slot ids)``: every slot bound to the first catalog
    row carrying what it needs (``picks`` = ``{slot id: value}`` to choose), every
    placeholder string resolved. A slot with no row keeps its placeholder, so the document
    is still valid once the missing source is bound by hand."""
    rows = list(catalog)
    used: set = set()
    bound: Dict[str, Mapping[str, Any]] = {}
    missing: List[str] = []
    for slot in recipe.slots:
        sid = str(slot["id"])
        want = (picks or {}).get(sid)
        row = next((r for r in rows if str(r.get("value")) == want), None) if want else None
        if row is None:
            row = next((r for r in rows if _satisfies(r, str(slot["needs"]))
                        and str(r.get("value")) not in used), None)
        if row is None:
            row = next((r for r in rows if _satisfies(r, str(slot["needs"]))), None)
        if row is None:
            missing.append(sid)
            continue
        used.add(str(row.get("value")))
        bound[sid] = row

    def sub(v: Any) -> Any:
        if isinstance(v, str):
            m = _SLOT.match(v)
            if not m:
                return v
            sid, kind, what = m.group(1), m.group(2), m.group(3)
            row = bound.get(sid)
            if row is None:
                return v if kind is None else ""
            if kind is None:
                return str(row.get("value"))
            layer, cols = _first_layer(row, kind)
            if what is None or what == "name":
                return layer
            return _measured(cols, 1 if what == "column2" else 0)
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v

    fig = sub(copy.deepcopy(recipe.figure))
    srcs = fig.setdefault("sources", [])
    have = {s.get("id") for s in srcs}
    for slot in recipe.slots:
        sid = str(slot["id"])
        if sid not in have:
            srcs.append({"id": sid, "source": f"slot:{sid}"})
    for s in srcs:
        row = bound.get(str(s.get("id")))
        if row is not None:
            s["source"] = str(row.get("value"))
            if not s.get("label"):
                s["label"] = str(row.get("label") or row.get("value"))
    # channels of an image item left empty: the row's channels in a colour-blind-safe order
    for it in fig.get("items") or []:
        img = it.get("image")
        if not isinstance(img, dict) or img.get("channels"):
            continue
        row = bound.get(str(img.get("source")))
        if row is None:
            continue
        nc = int(row.get("channels") or 1)
        names = list(row.get("channel_names") or [])
        img["channels"] = [{"c": c, "rgb": default_channel_hex(c, nc), "mode": "auto_pct",
                            "label": names[c] if c < len(names) else ""}
                           for c in range(min(nc, 4))]
    if fig.get("goal") is None and recipe.goal:
        fig["goal"] = recipe.goal
    return fig, missing


def default_channel_hex(c: int, nc: int) -> str:
    """White for a lone channel, else green, magenta, cyan, yellow — the merge order a
    colour-blind reader can tell apart."""
    if nc <= 1:
        return "#ffffff"
    return ["#00ff88", "#ff40ff", "#40e0ff", "#ffe040"][c % 4]


__all__ = ["FigureRecipe", "RECIPE_FORMAT", "BUILTIN_DIR", "USER_DIR", "ENV_DIR", "ENV_ENABLED",
           "NEEDS", "list_recipes", "find_recipe", "load_recipe", "recipe_from_dict",
           "recipe_to_dict", "save_recipe", "recipe_from_figure", "bind_recipe",
           "applicable", "default_channel_hex", "slugify"]
