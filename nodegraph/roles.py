"""The node taxonomy at runtime: the functional ROLES in 5 pipeline STAGES, and — since
ND2Studios V4.00 (2026-10-05) — the typed PAGE KINDS each role's nodes are offered on, all
read from the hand-curated ``codemap/node_roles.json``.

Qt-free. The GUI's palette groups nodes by stage and role through this module and filters
them by the active page's kind; the Workspace orders pages by kind through it; and
``scripts/_node_synopsis.py`` validates the same file against the live registry (every
shipped op in exactly one role, every role on at least one page kind). One file, one
reader, so the palette, the Workspace and the synopsis can never disagree about where a
node belongs.

An op the file does not know (a test fixture, a node added before its role was written)
lands in the ``other`` role of the ``other`` stage and on EVERY page kind rather than
vanishing from the palette: the synopsis check is what fails loudly in that case; the GUI
stays usable.

Page kinds
----------
``pages`` in the file declares the typed kinds with a pipeline ``order`` (``input`` 0 →
``refine`` 1 → ``process`` 2 → ``analyze`` 3). A role's ``pages`` lists the kinds whose
palette offers its ops; ``op_pages`` overrides that per op (``io.dock`` is useful on every
page, ``io.write_tiff`` only on the Analysis page). :data:`FREE_PAGE` is the kind that
applies no filter and is deliberately NOT in the file: it is the absence of an assignment,
the kind a pre-V4 single-graph file opens as.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES_PATH = os.path.join(ROOT, "codemap", "node_roles.json")

#: Where an unclassified op lands. Not in the JSON on purpose — the file must stay a
#: complete assignment, and this is the palette's way of showing that it is not.
OTHER_STAGE = "other"
OTHER_ROLE = "other"
OTHER_STAGE_META = {"label": "Other", "description":
                    "Nodes with no role in codemap/node_roles.json yet — add them there."}
OTHER_ROLE_META = {"label": "Unclassified", "stage": OTHER_STAGE, "description":
                   "Not yet assigned a functional role. The synopsis check "
                   "(scripts/_node_synopsis.py) fails until it is.", "ops": []}

#: The page kind that applies no filter: every node, and a page any other page may feed or
#: read from (while the page graph stays acyclic). Not in the JSON on purpose — it is the
#: absence of a page assignment, and what a graph saved before V4 opens as.
FREE_PAGE = "free"
FREE_PAGE_META = {"label": "Free", "order": None, "description":
                  "Every node, no stage filter — how a graph saved before V4 opens, and a "
                  "scratch page for anything that does not fit one stage."}


@lru_cache(maxsize=1)
def load() -> Dict:
    """The parsed roles file: ``{"stages": {...}, "roles": {...}, "pages": {...},
    "op_pages": {...}}`` (plus its prose keys). Cached; :func:`reload` drops the cache after
    the file is edited."""
    try:
        with open(ROLES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"stages": {}, "roles": {}}
    data.setdefault("stages", {})
    data.setdefault("roles", {})
    data.setdefault("pages", {})
    data.setdefault("op_pages", {})
    return data


def reload() -> Dict:
    load.cache_clear()
    _index.cache_clear()
    _page_index.cache_clear()
    return load()


def stages() -> List[Tuple[str, Dict]]:
    """``[(stage_key, {label, description}), ...]`` in the file's (pipeline) order."""
    return list(load()["stages"].items())


def roles_in(stage: str) -> List[Tuple[str, Dict]]:
    """``[(role_key, {label, stage, description, ops, pages}), ...]`` of one stage, file order."""
    return [(k, r) for k, r in load()["roles"].items() if r.get("stage") == stage]


@lru_cache(maxsize=1)
def _index() -> Dict[str, str]:
    return {op: rk for rk, r in load()["roles"].items() for op in r.get("ops", ())}


def role_of(op_key: str) -> Tuple[str, str]:
    """``(role_key, stage_key)`` for an op; ``("other", "other")`` when unassigned."""
    rk = _index().get(op_key)
    if rk is None:
        return OTHER_ROLE, OTHER_STAGE
    return rk, load()["roles"][rk].get("stage", OTHER_STAGE)


def role_meta(role_key: str) -> Dict:
    return load()["roles"].get(role_key, OTHER_ROLE_META)


def stage_meta(stage_key: str) -> Dict:
    return load()["stages"].get(stage_key, OTHER_STAGE_META)


# ── page kinds (V4.00) ────────────────────────────────────────────────────────

def pages() -> List[Tuple[str, Dict]]:
    """``[(kind, {label, order, description}), ...]`` — the TYPED page kinds in pipeline
    order. :data:`FREE_PAGE` is not among them (it has no order); see :func:`page_meta`."""
    items = list(load()["pages"].items())
    items.sort(key=lambda kv: (kv[1].get("order") is None, kv[1].get("order") or 0))
    return items


def page_kinds() -> List[str]:
    """Every kind a page may have: the typed kinds in order, then ``free``."""
    return [k for k, _ in pages()] + [FREE_PAGE]


def is_page_kind(kind: Optional[str]) -> bool:
    return kind == FREE_PAGE or (kind is not None and kind in load()["pages"])


def page_meta(kind: str) -> Dict:
    if kind == FREE_PAGE:
        return FREE_PAGE_META
    return load()["pages"].get(kind, {"label": kind, "order": None, "description": ""})


def page_order(kind: Optional[str]) -> Optional[int]:
    """Where a kind sits in the pipeline (0 = first), or ``None`` for ``free`` and for an
    unknown kind — an UNORDERED page, which the Workspace lets feed or read any page as long
    as the page graph stays acyclic."""
    if kind is None or kind == FREE_PAGE:
        return None
    o = load()["pages"].get(kind, {}).get("order")
    return int(o) if isinstance(o, int) and not isinstance(o, bool) else None


@lru_cache(maxsize=1)
def _page_index() -> Dict[str, Tuple[str, ...]]:
    data = load()
    declared = tuple(k for k, _ in pages())
    out: Dict[str, Tuple[str, ...]] = {}
    for r in data["roles"].values():
        ps = tuple(p for p in r.get("pages", ()) if p in data["pages"]) or declared
        for op in r.get("ops", ()):
            out[op] = ps
    for op, ps in data["op_pages"].items():
        out[op] = tuple(p for p in (ps or ()) if p in data["pages"]) or declared
    return out


def pages_of(op_key: str) -> Tuple[str, ...]:
    """The typed page kinds whose palette offers ``op_key``: its ``op_pages`` entry, else its
    role's ``pages``, else EVERY kind — an unassigned op stays placeable everywhere, and the
    synopsis check is what fails."""
    ps = _page_index().get(op_key)
    return ps if ps is not None else tuple(k for k, _ in pages())


def op_in_page(op_key: str, kind: Optional[str]) -> bool:
    """Does a page of ``kind`` offer ``op_key``? ``free`` (or no kind) offers everything."""
    if not kind or kind == FREE_PAGE:
        return True
    return kind in pages_of(op_key)


def ops_for_page(kind: Optional[str]) -> List[str]:
    """Every op the roles file assigns to ``kind``, in file order; ``free``/unknown → all."""
    data = load()
    ops = [op for r in data["roles"].values() for op in r.get("ops", ())]
    if not kind or kind == FREE_PAGE or kind not in data["pages"]:
        return ops
    return [op for op in ops if kind in pages_of(op)]


def roles_in_page(kind: Optional[str]) -> List[Tuple[str, Dict]]:
    """The roles with at least one op offered on ``kind`` (file order); all for ``free``."""
    if not kind or kind == FREE_PAGE:
        return list(load()["roles"].items())
    return [(k, r) for k, r in load()["roles"].items()
            if any(kind in pages_of(op) for op in r.get("ops", ()))]


__all__ = ["ROLES_PATH", "OTHER_ROLE", "OTHER_STAGE", "FREE_PAGE", "load", "reload",
           "stages", "roles_in", "role_of", "role_meta", "stage_meta",
           "pages", "page_kinds", "is_page_kind", "page_meta", "page_order", "pages_of",
           "op_in_page", "ops_for_page", "roles_in_page"]
