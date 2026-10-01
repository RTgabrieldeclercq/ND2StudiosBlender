"""Position groups for a source file — detection, and the sidecar that overrides it.

A multipoint acquisition is often several specimens, not one flat list of fields, and
:func:`nodegraph.placement.position_groups` recovers that structure from the stage
coordinates. This module is the layer above: it decides what a *source card* is going to
claim about a given file, and gives the user somewhere to disagree.

**The sidecar.** ``<file>.groups.json``, beside the file the way the ``.b2nd_store`` already
is. When it is present and valid it WINS over detection, and that is its whole reason to
exist: the detector is good (on the lab's CRC file it separates six 3x3 mosaics with a
margin of 7) but it is inferring intent from geometry, and there are real layouts it cannot
get right — two specimens mounted a field apart, one mosaic acquired in two passes, a
control site that belongs with the treatment it pairs with. Hand-editing the JSON is the
answer to all of those, and because the file is read on every open, an edit sticks.

**Nothing writes the sidecar automatically.** Detection is re-run instead, every time. The
alternative — writing the detector's own answer out on first open — would freeze whichever
version of the detector happened to see the file first, and would make a hand edit
indistinguishable from a machine guess. A file with no sidecar is a file nobody has had an
opinion about yet, and that is worth being able to tell.

**Validation is all-or-nothing.** A sidecar whose members are not an exact partition of
``0..M-1`` is REJECTED whole, with the reason, and detection runs instead. A partial one
would assign some positions to the wrong specimen and leave others unassigned, which is
read positionally downstream and so does not look broken — it looks like a result.

Qt-free: this is plain JSON plus the engine's detector, so the headless worker and the
selftest reach it on the same path the GUI does.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.metadata import POSITION_GROUP_KEY, POSITION_NAME_KEY
from nodegraph.placement import GROUP_GAP_FACTOR, GroupPlan, PositionGroup, position_groups

#: Sidecar format version. Bumped only for a change a previous reader would MISREAD; a new
#: optional key does not need one, because an unknown key is ignored.
SIDECAR_VERSION = 1

#: Suffix appended to the source file's name. Beside the file rather than in a central
#: index so that moving or copying the acquisition takes the grouping with it — the same
#: choice the ``.b2nd_store`` makes, and for the same reason.
SIDECAR_SUFFIX = ".groups.json"


def sidecar_path(path: str) -> str:
    """Where ``path``'s group sidecar lives."""
    return f"{path}{SIDECAR_SUFFIX}"


def read_sidecar(path: str, n_positions: int) -> Tuple[Optional[GroupPlan], str]:
    """``(plan, note)`` from ``path``'s sidecar — ``(None, reason)`` when there is none or
    it cannot be trusted.

    ``note`` is always a sentence fit to show the user: either what was loaded or why it was
    not. An unreadable sidecar is never silent — it is a file somebody wrote on purpose, so
    failing to use it is news.
    """
    sc = sidecar_path(path)
    if not os.path.exists(sc):
        return None, ""
    try:
        with open(sc, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, f"{os.path.basename(sc)} could not be read ({exc}); detecting instead."
    groups = raw.get("groups") if isinstance(raw, Mapping) else None
    if not isinstance(groups, (list, tuple)) or not groups:
        return None, f"{os.path.basename(sc)} lists no groups; detecting instead."

    seen: Dict[int, str] = {}
    built: List[PositionGroup] = []
    for i, g in enumerate(groups):
        if not isinstance(g, Mapping):
            return None, f"{os.path.basename(sc)}: entry {i} is not an object."
        key = str(g.get("key") or "").strip() or f"G{i + 1}"
        members = g.get("members")
        if not isinstance(members, (list, tuple)) or not members:
            return None, f"{os.path.basename(sc)}: group {key!r} lists no members."
        picks: List[int] = []
        for v in members:
            try:
                m = int(v)
            except (TypeError, ValueError):
                return None, f"{os.path.basename(sc)}: group {key!r} has a non-integer member."
            if not (0 <= m < n_positions):
                return None, (f"{os.path.basename(sc)}: group {key!r} names position {m}, "
                              f"but this file has {n_positions} (0-{n_positions - 1}).")
            if m in seen:
                return None, (f"{os.path.basename(sc)}: position {m} is in both {seen[m]!r} "
                              f"and {key!r} — a position belongs to one group.")
            seen[m] = key
            picks.append(m)
        built.append(PositionGroup(key=key, members=tuple(sorted(set(picks)))))
    missing = [m for m in range(n_positions) if m not in seen]
    if missing:
        return None, (f"{os.path.basename(sc)}: positions {missing[:8]}"
                      f"{' …' if len(missing) > 8 else ''} are in no group — a sidecar must "
                      f"cover all {n_positions}. Detecting instead.")
    plan = GroupPlan(groups=tuple(built), placed=True)
    return plan, (f"grouping read from {os.path.basename(sc)} "
                  f"({len(built)} groups, {n_positions} positions)")


def write_sidecar(path: str, plan: GroupPlan, *, detected: bool = False) -> str:
    """Write ``plan`` to ``path``'s sidecar and return the file written.

    ``detected`` records whether these groups came from the detector or from a person, so a
    later reader (or a person opening the JSON) can tell an inherited guess from a decision.
    It has no effect on how the file is used — a sidecar wins either way, which is what makes
    "detect once, then correct it" a usable workflow.
    """
    doc = {
        "version": SIDECAR_VERSION,
        "source": os.path.basename(path),
        "n_positions": sum(len(g.members) for g in plan.groups),
        "detected": bool(detected),
        "gap_factor": plan.gap_um or None,
        "margin": (None if plan.margin == float("inf") else round(plan.margin, 3)),
        "groups": [{"key": g.key, "members": list(g.members),
                    "shape": (f"{g.rows}x{g.cols} {g.order}" if g.rows and g.cols else "")}
                   for g in plan.groups],
    }
    sc = sidecar_path(path)
    with open(sc, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return sc


def resolve_plan(path: str, metadata: Mapping[str, Any], axes: Any,
                 gap_factor: float = GROUP_GAP_FACTOR) -> Tuple[GroupPlan, str]:
    """``(plan, note)`` for one source file: the sidecar if it has one, else detection.

    The note is what the source card shows, and it names WHICH of the two answers the user
    is looking at. That matters more than it sounds: "six groups" is the same sentence
    whether a person decided it or a threshold did, and only one of those is worth
    double-checking against the data.
    """
    n = int(getattr(axes, "m", 0) or 0)
    if n <= 0:
        return GroupPlan(), ""
    if path:
        plan, note = read_sidecar(path, n)
        if plan is not None:
            return plan, note
    else:
        note = ""
    detected = position_groups(metadata, axes, gap_factor)
    if not detected.placed:
        return detected, (note or "no stage positions — this file cannot be grouped")
    if len(detected.groups) <= 1:
        msg = f"one group of {n} positions (no gaps wider than one field)"
    else:
        msg = (f"{len(detected.groups)} groups detected "
               f"({', '.join(g.brief() for g in detected.groups[:4])}"
               f"{', …' if len(detected.groups) > 4 else ''})")
        if detected.margin < 2.0:
            msg += (f" — but only just: the gap between groups is {detected.margin:.2f}x the "
                    f"widest gap inside one, so this split depends on the threshold. Check it, "
                    f"and write a {SIDECAR_SUFFIX} sidecar if it is wrong.")
    return detected, ((note + " " if note else "") + msg).strip()


def descriptors(path: str, metadata: Mapping[str, Any], axes: Any) -> List[Dict[str, Any]]:
    """The card's ``[{"key", "size", "shape"}, …]`` for one source — ``[]`` if ungroupable.

    What an ``io.load`` card stores under ``ops.GROUPS_KEY`` and what its ``grpK`` output
    sockets are built from. Deliberately a plain list of plain dicts rather than the
    :class:`~nodegraph.placement.GroupPlan`: it is SERIALIZED into the saved graph, so it has
    to survive a JSON round-trip, and it has to keep meaning something years later when the
    dataclass has gained fields. The three keys it carries are the three a socket label and a
    materialized tap actually need.

    **A single group returns ``[]``**, not one entry. One group is the whole file, so a lone
    ``grp0`` socket would offer the same Dataset the ``image`` output already does, under a
    second name — the ambiguity the channel taps avoid with the same floor.

    Never raises: a file this cannot group is one whose card simply has no group sockets, and
    that must not be the thing that stops it opening.
    """
    try:
        plan, _note = resolve_plan(path, metadata, axes)
    except Exception:                    # noqa: BLE001 — grouping is never load-fatal
        return []
    if not plan.placed or len(plan.groups) < 2:
        return []
    return [{"key": g.key, "size": len(g.members),
             "shape": (f"{g.rows}x{g.cols} {g.order}" if g.rows and g.cols else "")}
            for g in plan.groups]


def group_metadata(plan: GroupPlan, names: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """The per-M metadata a resolved plan contributes: ``{position_group, position_name}``.

    Both are dropped rather than shortened when they cannot cover every position — the
    :data:`~nodegraph.metadata.PER_POSITION_KEYS` rule, which exists because these lists are
    read by INDEX and a short one reports another position's answer rather than admitting it
    does not know.
    """
    out: Dict[str, Any] = {}
    labels = plan.labels()
    if labels:
        out[POSITION_GROUP_KEY] = list(labels)
    if names and labels and len(names) >= len(labels):
        out[POSITION_NAME_KEY] = [str(v) for v in names[:len(labels)]]
    return out


def names_agree(plan: GroupPlan, names: Optional[Sequence[Any]]) -> Optional[bool]:
    """Do the acquisition's own point NAMES agree with ``plan``'s boundaries?

    ``None`` when there is nothing to compare. Otherwise True when every group's names are
    distinct within the group — which is what a per-group counter looks like from outside.
    NIS restarts its numbering at each point group, so the CRC file reads ``#1``…``#9`` six
    times over, and that repetition is an independent witness to exactly the six boundaries
    the geometry finds.

    Advisory only, and deliberately not used to override anything: a user who named every
    point uniquely would fail this test while being perfectly well grouped. It is here so a
    DISAGREEMENT can be reported, not so a tie can be broken.
    """
    if not names or not plan.placed or len(plan.groups) < 2:
        return None
    vals = [str(v) for v in names]
    if len(set(vals)) == len(vals):
        return None                      # all unique: the counter says nothing about groups
    for g in plan.groups:
        got = [vals[m] for m in g.members if m < len(vals)]
        if len(set(got)) != len(got):
            return False
    return True


__all__ = ["SIDECAR_VERSION", "SIDECAR_SUFFIX", "sidecar_path", "read_sidecar",
           "descriptors",
           "write_sidecar", "resolve_plan", "group_metadata", "names_agree"]
