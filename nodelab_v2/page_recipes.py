"""Page recipes — prebuilt PAGE graphs a new page can start from (V4.00 step 11).

A **page recipe** is one page's graph (the ``to_page_dict`` body: nodes, wires, zones, groups,
positions) under a name, a kind and a line of description: "Smooth & threshold" for a
Refinement page, "Label & measure" for a Processing page. Built-in recipes ship with the app
(``nodelab_v2/builtin_page_recipes/<kind>/<slug>.json``); the user's own live in
``~/.nd2studios/page_recipes/`` (*Save as page recipe…* on the page switcher) and shadow a
built-in of the same kind and name. Instantiating one (*New page… ▸ Page recipe*) adds a page,
loads the body into it, binds its Page Inputs to the output the user picked (or the nearest
one) and keeps its Page Output names unique on the page — from there it is an ordinary page,
edited like any other.

This is NOT a LabLink recipe (:mod:`nodelab_v2.lablink.recipe`): that is a whole graph
published for a hub to run; this is a starting point for one page of this editor.

Qt-free: the dialog that lists recipes lives in :mod:`nodelab_v2.new_page_dialog`; the
selftest reaches everything here through the document seam. Tests redirect the user folder
with ``NODELAB_PAGE_RECIPES_DIR`` (or switch it off with ``NODELAB_PAGE_RECIPES=0``) so the
real ``~/.nd2studios`` is never touched.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nodegraph import roles as R
from nodegraph.serialize import page_from_dict
from nodelab_v2.ops import PAGE_INPUT_OP, PAGE_NAME_KEY, PAGE_OUTPUT_OP, PAGE_SOURCE_KEY
from nodelab_v2.version import __version__
from nodelab_v2.workspace import Page, Workspace, kind_label, standard_kinds

#: the file tag every page recipe carries (``layout_store`` names its file the same way)
RECIPE_FORMAT = "nd2studios.page-recipe/1"
#: the recipes shipped with the app, one folder per kind
BUILTIN_DIR = Path(__file__).resolve().parent / "builtin_page_recipes"
#: the user's own recipes (``Save as page recipe…``)
USER_DIR = Path.home() / ".nd2studios" / "page_recipes"
#: redirects the user folder (tests: a tempdir)
ENV_DIR = "NODELAB_PAGE_RECIPES_DIR"
#: ``0`` / ``off`` disables the user folder altogether (built-ins still list)
ENV_ENABLED = "NODELAB_PAGE_RECIPES"

#: how a new page starts (:attr:`NewPageSpec.start`)
START_EMPTY, START_RECIPE, START_LINKED = "empty", "recipe", "linked"

_SLUG_BAD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PageRecipe:
    """One recipe: ``body`` is the page dict (:meth:`GraphDocument.to_page_dict`), ``path``
    the file it came from (``None`` for one made from a page and not saved yet)."""
    name: str
    kind: str
    description: str
    body: Dict[str, Any]
    path: Optional[str] = None
    builtin: bool = False


@dataclass
class NewPageSpec:
    """What the *New page…* dialog returns: the kind and name, how the page starts —
    :data:`START_EMPTY` (one bound Page Input when ``source`` is set), :data:`START_RECIPE`
    (``recipe``), :data:`START_LINKED` (a live linked copy of ``master``, a page id) — and
    the ``source`` (``"<page id>:<name>"``) its Page Input(s) read, or ``None``."""
    kind: str
    name: str = ""
    start: str = START_EMPTY
    recipe: Optional[PageRecipe] = None
    master: Optional[str] = None
    source: Optional[str] = None


# ── folders ──────────────────────────────────────────────────────────────────
def user_dir() -> Optional[Path]:
    """The user's recipe folder, or ``None`` when ``NODELAB_PAGE_RECIPES`` switches it off."""
    if os.environ.get(ENV_ENABLED, "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    override = os.environ.get(ENV_DIR, "").strip()
    return Path(override) if override else USER_DIR


def slugify(name: Any) -> str:
    """``"Smooth & threshold"`` → ``"smooth-threshold"``: the file stem a recipe is saved under."""
    return _SLUG_BAD.sub("-", str(name or "").lower()).strip("-")


# ── reading ──────────────────────────────────────────────────────────────────
def recipe_to_dict(recipe: PageRecipe) -> Dict[str, Any]:
    return {"format": RECIPE_FORMAT, "name": recipe.name, "kind": recipe.kind,
            "description": recipe.description, "app_version": __version__,
            "page": recipe.body}


def recipe_from_dict(d: Any, *, path: Optional[str] = None, builtin: bool = False) -> PageRecipe:
    """A :class:`PageRecipe` from a parsed file; ``ValueError`` for anything that is not one
    (the wrong tag, an unknown kind, a body :func:`page_from_dict` cannot read)."""
    if not isinstance(d, dict) or d.get("format") != RECIPE_FORMAT:
        raise ValueError(f"not a page recipe (format {RECIPE_FORMAT!r} expected)")
    kind = str(d.get("kind") or "").strip()
    if not R.is_page_kind(kind):
        raise ValueError(f"unknown page kind {kind!r}")
    body = d.get("page")
    if not isinstance(body, dict):
        raise ValueError("a page recipe needs a `page` body")
    page_from_dict(body)                              # refuses a body no page could load
    name = str(d.get("name") or "").strip() or (Path(path).stem if path else "recipe")
    return PageRecipe(name, kind, str(d.get("description") or "").strip(), body, path, builtin)


def load_recipe(path: Any, *, builtin: bool = False) -> PageRecipe:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return recipe_from_dict(d, path=str(path), builtin=builtin)


def _scan(folder: Optional[Path], *, builtin: bool) -> List[PageRecipe]:
    out: List[PageRecipe] = []
    if folder is None or not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*.json")):
        try:
            out.append(load_recipe(p, builtin=builtin))
        except Exception:                             # noqa: BLE001 — a damaged file is skipped
            continue
    return out


def list_recipes(kind: Optional[str] = None) -> List[PageRecipe]:
    """Every recipe offered — the built-ins, then the user's, a user recipe shadowing a
    built-in of the same kind and name — for ``kind`` (``None`` = all), by kind order then
    name. A file that is not a recipe is left out, never trusted."""
    by_key: Dict[Tuple[str, str], PageRecipe] = {}
    for r in _scan(BUILTIN_DIR, builtin=True):
        by_key[(r.kind, r.name.lower())] = r
    for r in _scan(user_dir(), builtin=False):
        by_key[(r.kind, r.name.lower())] = r
    out = [r for r in by_key.values() if kind is None or r.kind == kind]

    def order(r: PageRecipe) -> Tuple[int, str]:
        o = R.page_order(r.kind)
        return (o if o is not None else 99, r.name.lower())
    out.sort(key=order)
    return out


# ── writing ──────────────────────────────────────────────────────────────────
def recipe_from_page(page: Page, name: str, description: str = "") -> PageRecipe:
    """The page as a recipe — what the canvas shows (a linked page saves its mirrored graph
    with the overrides applied). Not saved yet: :func:`save_recipe`."""
    body = page.doc.to_page_dict()
    return PageRecipe(str(name or "").strip() or page.name, page.kind,
                      str(description or "").strip(), body, None, False)


def save_recipe(recipe: PageRecipe, folder: Optional[Path] = None) -> Path:
    """Write ``recipe`` under ``<folder or user folder>/<kind>/<slug>.json`` (atomic: a
    ``.part`` renamed into place). ``RuntimeError`` when the user folder is switched off."""
    target_dir = Path(folder) if folder is not None else user_dir()
    if target_dir is None:
        raise RuntimeError(f"page recipes are switched off ({ENV_ENABLED}=0)")
    target = target_dir / recipe.kind / f"{slugify(recipe.name) or 'recipe'}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_text(json.dumps(recipe_to_dict(recipe), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    os.replace(part, target)
    return target


# ── placing a recipe on a page ───────────────────────────────────────────────
def next_kind(kind: Optional[str]) -> Optional[str]:
    """The kind a page reading ``kind`` would have: ``input → refine → process → analyze →
    None``; a Free (or unknown) page → the first kind that reads others."""
    kinds = list(standard_kinds())
    if kind not in kinds:
        return kinds[1] if len(kinds) > 1 else None
    i = kinds.index(kind)
    return kinds[i + 1] if i + 1 < len(kinds) else None


def bind_inputs(ws: Workspace, page_id: str, source: str, *, only_unbound: bool = True) -> int:
    """Point the page's Page Inputs at ``source`` — every one that does not resolve
    (``only_unbound``) or all of them. Goes through ``touch``, so on a linked page the
    source becomes that page's override. Returns how many were set."""
    page = ws.pages[page_id]
    n = 0
    for rec in list(page.doc.nodes.values()):
        if rec.op_key != PAGE_INPUT_OP:
            continue
        if only_unbound and ws.resolve_source(page_id, rec.params.get(PAGE_SOURCE_KEY)):
            continue
        if str(rec.params.get(PAGE_SOURCE_KEY) or "") == source:
            continue
        rec.params[PAGE_SOURCE_KEY] = source
        page.doc.touch(rec.id)
        n += 1
    return n


def unique_output_names(ws: Workspace, page_id: str) -> int:
    """Give the page's Page Outputs distinct names (a guard for a hand-edited recipe: the
    second ``mask`` becomes ``mask2``). Returns how many were renamed."""
    page = ws.pages[page_id]
    seen: set = set()
    n = 0
    for rec in list(page.doc.nodes.values()):
        if rec.op_key != PAGE_OUTPUT_OP:
            continue
        name = str(rec.params.get(PAGE_NAME_KEY) or "").strip()
        if not name:
            continue
        if name.lower() in seen:
            name = ws.unique_output_name(page_id, name)
            rec.params[PAGE_NAME_KEY] = name
            page.doc.touch(rec.id)
            n += 1
        seen.add(name.lower())
    return n


def only_seed(page: Page) -> bool:
    """True when the page holds nothing but Page Inputs — the one the window seeds an empty
    page with — so a recipe or a linked start may take it over."""
    return all(r.op_key == PAGE_INPUT_OP for r in page.doc.nodes.values())


def _take_over(ws: Workspace, page_id: str) -> Page:
    """An existing page about to receive a body: it must be empty but for its seed Input."""
    page = ws.pages[page_id]
    if not only_seed(page):
        raise ValueError(f"page {page.name!r} already holds nodes — start a new page instead")
    if page.master or not getattr(page.doc, "editable_topology", True):
        raise ValueError(f"page {page.name!r} is linked — its graph is its master's")
    for nid in list(page.doc.nodes):
        page.doc.remove_node(nid)
    return page


def instantiate(ws: Workspace, recipe: PageRecipe, *, name: Optional[str] = None,
                kind: Optional[str] = None, source: Optional[str] = None,
                index: Optional[int] = None, into: Optional[str] = None) -> Page:
    """Put ``recipe`` on a page: a new one (``name``, ``kind`` — default the recipe's own —
    at ``index``) or the existing empty page ``into``. Its Page Inputs are bound to
    ``source`` or the page's default source, its Output names made unique."""
    if into is not None:
        page = _take_over(ws, into)
        if name and name != page.name:
            ws.rename_page(page.id, name)
    else:
        page = ws.add_page(name or recipe.name, kind or recipe.kind, index=index)
    ws.load_page_body(page.id, copy.deepcopy(recipe.body))
    if R.op_in_page(PAGE_INPUT_OP, page.kind):
        src = source or ws.default_source(page.id)
        if src:
            bind_inputs(ws, page.id, src, only_unbound=True)
    unique_output_names(ws, page.id)
    return page


def apply_new_page(ws: Workspace, spec: NewPageSpec, *, into: Optional[str] = None) -> Page:
    """The one entry the window calls for *New page…*: an empty page with a bound Page
    Input, a page from a recipe, or a page linked to a master (its Page Inputs re-pointed
    at ``spec.source`` when one was chosen — an override of that page). ``into`` names an
    existing page holding at most its seed Input to fill instead of adding one; a linked
    start removes it (a linked page is always a new page)."""
    from nodelab_v2.ops import ensure_ops
    ensure_ops()
    if spec.start == START_RECIPE:
        if spec.recipe is None:
            raise ValueError("no page recipe chosen")
        return instantiate(ws, spec.recipe, name=spec.name or None, kind=spec.kind,
                           source=spec.source, into=into)
    if spec.start == START_LINKED:
        if not spec.master or spec.master not in ws.pages:
            raise ValueError("no master page chosen")
        page = ws.duplicate_page(spec.master, dependent=True, name=spec.name or None)
        if spec.source:
            bind_inputs(ws, page.id, spec.source, only_unbound=False)
        if (into is not None and into in ws.pages and into != page.id
                and only_seed(ws.pages[into]) and len(ws.pages) > 1):
            ws.remove_page(into)
        return page
    if spec.start != START_EMPTY:
        raise ValueError(f"unknown start {spec.start!r}")
    if into is not None:
        page = _take_over(ws, into)
        if spec.name and spec.name != page.name:
            ws.rename_page(page.id, spec.name)
    else:
        page = ws.add_page(spec.name or kind_label(spec.kind), spec.kind)
    if spec.source and R.op_in_page(PAGE_INPUT_OP, page.kind):
        page.doc.add_node(PAGE_INPUT_OP, x=40.0, y=120.0, params={PAGE_SOURCE_KEY: spec.source})
    elif R.op_in_page(PAGE_INPUT_OP, page.kind):
        ws.seed_input(page.id)
    return page


__all__ = ["RECIPE_FORMAT", "BUILTIN_DIR", "USER_DIR", "ENV_DIR", "ENV_ENABLED",
           "START_EMPTY", "START_RECIPE", "START_LINKED", "PageRecipe", "NewPageSpec",
           "user_dir", "slugify", "recipe_to_dict", "recipe_from_dict", "load_recipe",
           "list_recipes", "recipe_from_page", "save_recipe", "next_kind", "bind_inputs",
           "unique_output_names", "only_seed", "instantiate", "apply_new_page"]
