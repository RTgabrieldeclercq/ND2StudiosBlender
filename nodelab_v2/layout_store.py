"""Where the window's panel layout lives between sessions (V4.00 step 3) — Qt-free.

One JSON file, ``~/.nd2studios/layout.json`` — the per-user settings folder every other
remembered preference already uses (``overlays.json``, ``lablink-presets.json``). It holds
what Qt's own ``QMainWindow.saveGeometry``/``saveState`` produce (opaque bytes, base64 here)
plus the list of panels that existed when it was written. The list is what makes extra
instances restorable: ``QMainWindow.restoreState`` only re-positions docks that ALREADY
exist under the saved ``objectName``, so a second viewer or a third canvas has to be
re-created from this list first, and only then handed the state that places it.

Two environment switches, both for tests and scripts rather than for users:

* ``NODELAB_LAYOUT_FILE`` — read and write this file instead (the GUI probe uses a temp one);
* ``NODELAB_LAYOUT=0`` — no persistence at all: nothing is read at start-up and nothing is
  written on close. Every probe and script that builds a window sets it, so a test run can
  neither inherit a user's floating panels nor overwrite them.

**Never trusted.** A file that does not parse, carries another ``format``/``version``, or has
the wrong shape is IGNORED — :func:`load_layout` returns ``None`` and the window opens on its
default layout — rather than raising: a layout is a convenience, and a damaged one must never
stop the application from starting. Writes are atomic (``.part`` then ``os.replace``), so a
crash mid-write leaves the previous layout intact instead of half a file.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

#: The layout generation. Bumped when the panel set or the ``objectName`` scheme changes
#: incompatibly, or when the default layout changes in a way an old saved state would hide
#: (V4.00 step 11a → 2: the Viewer is on screen from the start, and a layout saved by a
#: build that launched without one would keep hiding it); it is ALSO the version handed to
#: ``saveState``/``restoreState``, so Qt itself refuses a state blob from another generation.
LAYOUT_VERSION = 2
#: The file's own format tag — a second, independent guard on top of :data:`LAYOUT_VERSION`.
LAYOUT_FORMAT = "nd2studios.layout/2"
#: Where a user's layout lives unless :data:`ENV_FILE` says otherwise.
USER_FILE = Path.home() / ".nd2studios" / "layout.json"
#: Points persistence at another file.
ENV_FILE = "NODELAB_LAYOUT_FILE"
#: ``0``/``false``/``no``/``off`` turns persistence off entirely.
ENV_ENABLED = "NODELAB_LAYOUT"
#: A dock's ``objectName``: its panel KIND and an instance index, ``"viewer:1"``.
DOCK_NAME_RE = re.compile(r"^([a-z][a-z0-9_]*):(\d+)$")


def layout_enabled() -> bool:
    """Whether the window should restore a saved layout and save its own on close."""
    return os.environ.get(ENV_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")


def layout_path() -> Path:
    """The layout file in effect: :data:`ENV_FILE` when set, else :data:`USER_FILE`."""
    p = os.environ.get(ENV_FILE, "").strip()
    return Path(p) if p else USER_FILE


def dock_name(kind: str, index: int) -> str:
    """``"viewer:1"`` — the objectName a panel of ``kind`` gets as instance ``index``."""
    return f"{kind}:{int(index)}"


def parse_dock_name(name: str) -> Optional[tuple]:
    """``("viewer", 1)`` from ``"viewer:1"``; ``None`` for anything else."""
    m = DOCK_NAME_RE.match(str(name or ""))
    return (m.group(1), int(m.group(2))) if m else None


def encode(blob: bytes) -> str:
    return base64.b64encode(bytes(blob)).decode("ascii")


def decode(text: str) -> bytes:
    return base64.b64decode(str(text).encode("ascii"), validate=True)


def make_layout(*, geometry: bytes, state: bytes, docks: List[Mapping[str, Any]],
                app_version: str = "") -> Dict[str, Any]:
    """The JSON-able record of one window's layout. ``docks`` is
    ``[{"name": "viewer:1", "kind": "viewer", "binding": <json or None>}, …]`` — the binding
    is whatever the panel kind needs to show the same thing again (a viewer's page and node,
    a canvas's page), opaque here."""
    out: List[Dict[str, Any]] = []
    for d in docks:
        name = str(d.get("name", ""))
        parsed = parse_dock_name(name)
        if parsed is None:
            raise ValueError(f"dock name {name!r} is not '<kind>:<index>'")
        kind = str(d.get("kind") or parsed[0])
        if kind != parsed[0]:
            raise ValueError(f"dock {name!r} claims kind {kind!r}")
        binding = d.get("binding")
        json.dumps(binding)                       # must be JSON-able — raises if not
        out.append({"name": name, "kind": kind, "binding": binding})
    return {"format": LAYOUT_FORMAT, "version": LAYOUT_VERSION,
            "app_version": str(app_version or ""),
            "geometry": encode(geometry), "state": encode(state), "docks": out}


def validate(d: Any) -> Optional[Dict[str, Any]]:
    """``d`` normalized, or ``None`` when it is not a layout this build can use. Never
    raises. Dock entries that are malformed are DROPPED (the rest still restores); a bad
    geometry/state blob, another format or another version rejects the whole record."""
    try:
        return _validate(d)
    except Exception:                                  # noqa: BLE001 — never raises
        return None


def _validate(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, Mapping):
        return None
    if d.get("format") != LAYOUT_FORMAT or d.get("version") != LAYOUT_VERSION:
        return None
    try:
        geometry = decode(d.get("geometry", ""))
        state = decode(d.get("state", ""))
    except (binascii.Error, ValueError, TypeError, UnicodeEncodeError):
        return None
    if not state:
        return None
    docks: List[Dict[str, Any]] = []
    seen = set()
    listed = d.get("docks")
    for rec in (listed if isinstance(listed, list) else ()):
        if not isinstance(rec, Mapping):
            continue
        name = str(rec.get("name", ""))
        parsed = parse_dock_name(name)
        if parsed is None or name in seen or rec.get("kind", parsed[0]) != parsed[0]:
            continue
        seen.add(name)
        docks.append({"name": name, "kind": parsed[0], "index": parsed[1],
                      "binding": rec.get("binding")})
    return {"format": LAYOUT_FORMAT, "version": LAYOUT_VERSION,
            "app_version": str(d.get("app_version") or ""),
            "geometry": geometry, "state": state, "docks": docks}


def load_layout(path: Optional[Path] = None, *,
                quarantine: bool = False) -> Optional[Dict[str, Any]]:
    """The saved layout, validated (bytes decoded), or ``None`` — missing, unreadable,
    damaged, or from another generation. Never raises.

    ``quarantine`` sets a file that EXISTS but cannot be used aside as
    ``<name>.rejected`` (replacing an older one), so the next save cannot silently destroy
    it — a layout from a newer build, which an older checkout cannot read, survives a
    session of the older one, and a damaged file is kept for whoever wants to look."""
    p = Path(path) if path is not None else layout_path()
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            rec = validate(json.load(f))
    except Exception:                                  # noqa: BLE001 — never raises
        rec = None
    if rec is None and quarantine:
        try:
            os.replace(p, p.with_name(p.name + ".rejected"))
        except OSError:
            pass
    return rec


def save_layout(record: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Write ``record`` (from :func:`make_layout`) atomically. Returns the path written.
    Raises ``OSError`` on a write failure — the caller decides whether that matters (the
    window's close handler swallows it: failing to remember a layout must not block quit)."""
    p = Path(path) if path is not None else layout_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # a name of its own per process, so two windows closing at once cannot interleave
    part = p.with_name(f"{p.name}.{os.getpid()}.part")
    try:
        with open(part, "w", encoding="utf-8") as f:
            json.dump(dict(record), f, indent=2, sort_keys=True)
        os.replace(part, p)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return p


__all__ = ["LAYOUT_VERSION", "LAYOUT_FORMAT", "USER_FILE", "ENV_FILE", "ENV_ENABLED",
           "layout_enabled", "layout_path", "dock_name", "parse_dock_name", "encode",
           "decode", "make_layout", "validate", "load_layout", "save_layout"]
