"""The node taxonomy at runtime: 19 functional ROLES in 5 pipeline STAGES, read from the
hand-curated ``codemap/node_roles.json`` (2026-10-02).

Qt-free. The GUI's palette groups nodes by stage and role through this module, and
``scripts/_node_synopsis.py`` validates the same file against the live registry (every
shipped op in exactly one role). One file, one reader, so the palette and the synopsis can
never disagree about where a node belongs.

An op the file does not know (a test fixture, a node added before its role was written)
lands in the ``other`` role of the ``other`` stage rather than vanishing from the palette:
the synopsis check is what fails loudly in that case; the GUI stays usable.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

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


@lru_cache(maxsize=1)
def load() -> Dict:
    """The parsed roles file: ``{"stages": {...}, "roles": {...}}`` (plus its prose keys).
    Cached; :func:`reload` drops the cache after the file is edited."""
    try:
        with open(ROLES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"stages": {}, "roles": {}}
    data.setdefault("stages", {})
    data.setdefault("roles", {})
    return data


def reload() -> Dict:
    load.cache_clear()
    return load()


def stages() -> List[Tuple[str, Dict]]:
    """``[(stage_key, {label, description}), ...]`` in the file's (pipeline) order."""
    return list(load()["stages"].items())


def roles_in(stage: str) -> List[Tuple[str, Dict]]:
    """``[(role_key, {label, stage, description, ops}), ...]`` of one stage, file order."""
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


__all__ = ["ROLES_PATH", "OTHER_ROLE", "OTHER_STAGE", "load", "reload", "stages",
           "roles_in", "role_of", "role_meta", "stage_meta"]
