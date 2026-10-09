"""Quick analysis — what to scope an expensive analysis to so a settings change answers in
seconds (2026-10-09). Qt-free, so the selftest checks the picks without a window.

The user's request: "a way to do a quick analysis with the settings chosen … to find the best
set of variables for their data set before the full run". The app already has the machinery:
the **F9 troubleshooting scope** cuts every source seed to picked M / T / Z frames and a
region box (results memo-keyed, so a revisited setting is instant), and the **Iterate** card
sweeps a setting and keeps the best by a metric. What a DIC / PIV / DVC / SerialTrack node
needs on top is the RIGHT picks — and they are not the F9 defaults: a single frame gives a
correlator nothing to compare, and a reference frame left out of the picks makes the scoped
run correlate against the wrong image. :func:`quick_picks` works them out from the node's
own settings (or, for a staged card, from the grid card at the head of its chain):

========  =========================================================================
M         the position the Viewer is on
Z         the plane on screen for a planar method (DIC, PIV, 2-D tracking); the whole
          volume for DVC and 3-D tracking
T         {reference, current} for a fixed reference; {current − 1, current} for
          previous-frame / ensemble; the current frame and the next four for tracking
region    the amber box as the user left it; on first use a centred window about
          16 subsets / windows across (a subset-sized grid of a few hundred points)
========  =========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

#: op → analysis family. The single nodes and every card of the staged chains.
FAMILY: Dict[str, str] = {
    "analysis.dic_correlate": "dic", "analysis.piv": "piv", "analysis.dvc_field": "dvc",
    "track.objects": "track",
    "dic.grid": "dic", "dic.initial_guess": "dic", "dic.refine_mesh": "dic",
    "dic.local_match": "dic", "dic.global_step": "dic", "dic.strain": "dic",
    "piv.first_pass": "piv", "piv.validate": "piv", "piv.deform_pass": "piv",
    "piv.smooth": "piv",
    "dvc.grid": "dvc", "dvc.initial_guess": "dvc", "dvc.local_match": "dvc",
    "dvc.global_step": "dvc", "dvc.strain": "dvc",
}

#: the card of a staged chain that decides channel, reference policy and grid size
_HEAD = {"dic": "dic.grid", "piv": "piv.first_pass", "dvc": "dvc.grid"}

#: the subset / window-size param of each family's settings node
_SUBSET = {"dic": ("winsize", 40), "piv": ("window_size", 32), "dvc": ("subset_size", 32)}

#: frames a quick tracking run links across
TRACK_FRAMES = 5


@dataclass(frozen=True)
class QuickPicks:
    """What the Quick analysis button arms F9 with — or ``error`` saying why it cannot."""
    ms: Tuple[int, ...] = ()
    ts: Tuple[int, ...] = ()
    zs: Tuple[int, ...] = ()
    region: Optional[Tuple[int, int, int, int]] = None
    note: str = ""
    error: str = ""


def is_quick_op(op_key: str) -> bool:
    return op_key in FAMILY


def _settings_node(doc: Any, node_id: str, family: str) -> str:
    """The node whose params decide the reference policy: the node itself for a single
    analysis node, else the chain's head card found by walking upstream."""
    head = _HEAD.get(family)
    rec = doc.nodes.get(node_id)
    if rec is None or head is None or rec.op_key == head or rec.op_key.startswith("analysis."):
        return node_id
    seen, todo = set(), [node_id]
    while todo:
        nid = todo.pop()
        if nid in seen:
            continue
        seen.add(nid)
        r = doc.nodes.get(nid)
        if r is not None and r.op_key == head:
            return nid
        for e in doc.edges:
            if e[2] == nid:                      # (src, src_socket, dst, dst_socket)
                todo.append(e[0])
    return node_id


def _param(rec: Any, name: str, default: Any) -> Any:
    spec = rec.spec()
    sock = spec.input(name) if spec is not None else None
    if name in rec.params:
        return rec.params[name]
    if sock is not None and sock.default is not None:
        return sock.default
    return default


def _mode(rec: Any, name: str, default: str) -> str:
    try:
        return str(rec.state().get(name, default))
    except Exception:  # noqa: BLE001
        return default


def quick_picks(doc: Any, node_id: str, coords: Sequence[int],
                totals: Sequence[int], extent: Optional[Sequence[int]] = None,
                region: Optional[Tuple[int, int, int, int]] = None) -> QuickPicks:
    """The picks for a quick run of ``node_id``. ``coords`` is the Viewer's ``(m, t, z, c)``,
    ``totals`` the source's ``(M, T, Z)``, ``extent`` its ``(Y, X)``, ``region`` the box the
    user already drew (kept when given)."""
    rec = doc.nodes.get(node_id)
    if rec is None or rec.op_key not in FAMILY:
        return QuickPicks(error="Quick analysis applies to DIC, PIV, DVC and SerialTrack nodes")
    family = FAMILY[rec.op_key]
    m, t, z = (int(coords[0]), int(coords[1]), int(coords[2])) if coords else (0, 0, 0)
    M, T, Z = (max(1, int(v)) for v in (tuple(totals) + (1, 1, 1))[:3])
    m, t, z = min(max(0, m), M - 1), min(max(0, t), T - 1), min(max(0, z), Z - 1)
    if T < 2:
        return QuickPicks(error="a quick analysis compares frames, and this source has one "
                                "timepoint — there is nothing to scope down")
    srec = doc.nodes.get(_settings_node(doc, node_id, family)) or rec
    planar = family in ("dic", "piv")
    if family == "track":
        planar = _mode(srec, "st_dim", "2D") != "3D"
        t0 = min(t, max(0, T - TRACK_FRAMES))
        ts = tuple(range(t0, min(T, t0 + TRACK_FRAMES)))
        note = f"frames {ts[0]}–{ts[-1]} of position {m}"
    else:
        default_mode = "previous_frame" if family == "piv" else "fixed_frame"
        ref_mode = _mode(srec, "reference_mode", default_mode)
        if ref_mode == "fixed_frame":
            ref = min(max(0, int(_param(srec, "reference_frame", 0) or 0)), T - 1)
            cur = t if t != ref else (ref + 1 if ref + 1 < T else ref - 1)
            ts = tuple(sorted({ref, cur}))
            note = f"reference t={ref} → t={cur}"
        else:
            cur = max(1, t)
            ts = (cur - 1, cur)
            note = f"t={cur - 1} → t={cur}"
    zs = (z,) if (planar and Z > 1) else ()
    reg = region
    if reg is None and extent is not None and family != "track":
        name, dflt = _SUBSET[family]
        sub = int(_param(srec, name, dflt) or dflt)
        Y, X = int(extent[0]), int(extent[1])
        side = max(128, 16 * max(4, sub))
        h, w = min(Y, side), min(X, side)
        if h < Y or w < X:
            y0, x0 = (Y - h) // 2, (X - w) // 2
            reg = (y0, y0 + h, x0, x0 + w)
    where = f"m={m}" + (f", z={z}" if zs else "")
    box = f", {reg[3] - reg[2]}×{reg[1] - reg[0]} px box" if reg is not None else ""
    return QuickPicks(ms=(m,), ts=ts, zs=zs, region=reg, note=f"{where}, {note}{box}")


__all__ = ["FAMILY", "QuickPicks", "is_quick_op", "quick_picks", "TRACK_FRAMES"]
