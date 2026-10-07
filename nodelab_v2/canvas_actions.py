"""The canvas's action pill (2026-10-07): what it says, and what it offers for what is going on.

The pill sits at the top centre of every canvas. Its TEXT names the context — the page's
graph, the cards or regions selected, a region box being drawn — and clicking it opens a
menu of the actions that make sense there and then: draw a region, group the selected cards
into one, merge selected regions, take cards out of a region, ungroup a region, duplicate it
as a linked tab, reorganize the graph, undo that, fit the view. This module decides all of
it from plain state — the document, the selection, a few flags — so the selftest checks the
offer headless; the window turns each :class:`CanvasAction` into a menu entry and runs it by
its :attr:`~CanvasAction.key` (:meth:`nodelab_v2.window.MainWindow.run_canvas_action`).

An action that cannot run here is still LISTED, greyed out, with the reason as its tooltip —
the pill teaches what it could do. Regions of a linked page are its master's, so a linked
page lists them that way and offers to go to the master instead.

Keys
----
``draw_region`` / ``cancel_region`` — arm or put away the region box. ``group_nodes`` — the
selected cards into a new region. ``add_to_region:<frame id>`` — the selected loose cards into
the one region selected or touched. ``remove_from_region`` — the selected cards out of their
regions. ``merge_regions`` — the selected regions (and loose cards) into the first one (the window
passes the selection in the page's order, so that is the oldest).
``ungroup_regions`` — the selected regions' frames go, their cards stay. ``rename_region`` /
``region_tab`` — the one selected region. ``ungroup_nodegroup`` — a selected node-group card
inlined back. ``goto_region`` — on a region tab, its region on the master. ``reorganize`` /
``undo_layout`` / ``fit``.
"""
from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from nodegraph.groups import group_name_of

#: the menu's sections, in order (a separator between two that both have entries)
SECTIONS = ("selection", "region", "graph")

#: why a region action is refused on a linked page
LINKED_REGION_HINT = ("regions of a linked page are its master's — change them on the "
                      "master page")


class CanvasAction(NamedTuple):
    """One entry of the pill's menu."""
    key: str
    label: str
    tip: str = ""
    enabled: bool = True
    section: str = "graph"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def frames_of(doc, node_id: str) -> List[str]:
    """The frames ``node_id`` is a member of, in frame order."""
    return [fid for fid, fr in doc.frames.items() if node_id in fr.members]


def pill_title(doc, nodes: Sequence[str] = (), frames: Sequence[str] = (), *,
               region_armed: bool = False) -> str:
    """The pill's text: what the menu will act on."""
    if region_armed:
        return "Drag a box around the nodes  ·  Esc"
    nodes = [n for n in nodes if n in doc.nodes]
    frames = [f for f in frames if f in doc.frames]
    if len(frames) == 1 and not nodes:
        return f"Region “{doc.frames[frames[0]].title}”"
    bits = []
    if nodes:
        bits.append(_plural(len(nodes), "node"))
    if frames:
        bits.append(_plural(len(frames), "region"))
    if bits:
        return " · ".join(bits)
    region = getattr(doc, "region", None)
    if region is not None:
        title = doc.region_title() if hasattr(doc, "region_title") else ""
        return f"Region tab · “{title}”" if title else "Region tab"
    return "Graph"


def canvas_actions(doc, nodes: Sequence[str] = (), frames: Sequence[str] = (), *,
                   region_armed: bool = False, can_undo_layout: bool = False,
                   master_name: str = "") -> List[CanvasAction]:
    """The pill's menu for ``doc`` with ``nodes`` and ``frames`` selected. ``region_armed``:
    the region box is waiting for a drag; ``can_undo_layout``: the last reorganize can still
    be taken back; ``master_name``: the master page's name on a linked page."""
    nodes = [n for n in nodes if n in doc.nodes]
    frames = [f for f in frames if f in doc.frames]
    linked = not getattr(doc, "editable_topology", True)
    region = getattr(doc, "region", None)
    shape_ok = not linked
    why = "" if shape_ok else LINKED_REGION_HINT
    out: List[CanvasAction] = []

    def add(key, label, tip="", enabled=True, section="graph"):
        out.append(CanvasAction(key, label, tip if enabled else (why or tip), enabled, section))

    # ── what is selected ──────────────────────────────────────────────────────
    loose = [n for n in nodes if not frames_of(doc, n)]
    framed = [n for n in nodes if frames_of(doc, n)]
    touched = list(dict.fromkeys(list(frames) + [f for n in framed for f in frames_of(doc, n)]))
    if nodes and not frames:
        if loose and not framed:
            add("group_nodes", f"Group {_plural(len(loose), 'node')} into a region",
                "A frame around the selected cards — a REGION, with a port on its border for "
                "every wire crossing it, ready to duplicate as a linked tab.",
                shape_ok, "selection")
        if loose and len(touched) == 1:
            title = doc.frames[touched[0]].title
            add(f"add_to_region:{touched[0]}",
                f"Add {_plural(len(loose), 'node')} to region “{title}”",
                "The region grows round them; its tabs grow too.", shape_ok, "selection")
        if framed:
            add("remove_from_region", f"Remove {_plural(len(framed), 'node')} from "
                + (f"region “{doc.frames[touched[0]].title}”" if len(touched) == 1
                   else "their regions"),
                "The cards leave the frame and stay on the page; a region left empty goes. "
                "Its tabs lose them too.", shape_ok, "selection")
    if len(frames) >= 2:
        extra = f" (and {_plural(len(loose), 'node')})" if loose else ""
        add("merge_regions", f"Group {len(frames)} regions into one{extra}",
            f"One region holding every card of the selected ones, under the first one's "
            f"name “{doc.frames[frames[0]].title}”. Its tabs grow; the other regions' tabs "
            f"keep the cards they showed.", shape_ok, "selection")
    if len(frames) == 1 and loose:
        add(f"add_to_region:{frames[0]}",
            f"Add {_plural(len(loose), 'node')} to region “{doc.frames[frames[0]].title}”",
            "The region grows round them; its tabs grow too.", shape_ok, "selection")
    groups = [n for n in nodes if group_name_of(doc.nodes[n].op_key)]
    if groups:
        add("ungroup_nodegroup", f"Ungroup {_plural(len(groups), 'node group')}",
            "Inline the group's nodes back onto the canvas (Ctrl+Shift+G).",
            shape_ok, "selection")

    # ── the region(s) selected ────────────────────────────────────────────────
    if len(frames) == 1:
        f = frames[0]
        title = doc.frames[f].title
        out.append(CanvasAction(
            "region_tab", "Duplicate region as a linked tab",
            "A new Free page of this region alone: its nodes follow the region here, a Page "
            "Input or Output of its own stands at every port, and its values may differ.",
            True, "region"))
        add("rename_region", "Rename region…", f"Rename “{title}”.", shape_ok, "region")
    if frames:
        add("ungroup_regions", ("Ungroup region" if len(frames) == 1
                                else f"Ungroup {len(frames)} regions"),
            "The frame goes, its cards stay where they are. A tab of the region keeps the "
            "cards it showed.", shape_ok, "region")
    if region is not None:
        out.append(CanvasAction(
            "goto_region", f"Go to the region on “{master_name or 'the master'}”",
            "Show the master page with this region selected — its nodes are edited there, "
            "and every tab of it follows.", True, "region"))

    # ── the graph ─────────────────────────────────────────────────────────────
    if region_armed:
        out.append(CanvasAction("cancel_region", "Put the region box away\tEsc",
                                "Back to selecting and panning.", True, "graph"))
    else:
        add("draw_region", "Draw a region…\tR",
            "Drag a box around a set of nodes: they become a region with a port for every "
            "wire crossing it. Alt+drag draws one straight away.", shape_ok, "graph")
    n = len(doc.nodes)
    if region is not None:
        tip = ("Arrange this tab's own nodes — its Page Inputs and Outputs — round the region, "
               "which stays as its master has it.")
    elif linked:
        tip = (f"Arrange the cards left to right by data flow. The layout is "
               f"“{master_name or 'the master'}”'s, so the master and every page linked to it "
               f"follow.")
    else:
        tip = ("Arrange the cards left to right by data flow, every region kept whole with "
               "room for its ports. Undo reorganize puts them back.")
    out.append(CanvasAction("reorganize", "Reorganize graph",
                            tip if n >= 2 else "Nothing to arrange — fewer than two nodes.",
                            n >= 2, "graph"))
    if can_undo_layout:
        out.append(CanvasAction("undo_layout", "Undo reorganize",
                                "Put every card back where it was before.", True, "graph"))
    out.append(CanvasAction("fit", "Fit graph\tHome", "Fit the view to the nodes.",
                            n >= 1, "graph"))
    return out


def sectioned(actions: Iterable[CanvasAction]) -> List[Tuple[str, List[CanvasAction]]]:
    """The actions grouped by :data:`SECTIONS`, empty sections dropped."""
    acts = list(actions)
    return [(s, [a for a in acts if a.section == s]) for s in SECTIONS
            if any(a.section == s for a in acts)]


def find(actions: Iterable[CanvasAction], key: str) -> Optional[CanvasAction]:
    return next((a for a in actions if a.key == key), None)


__all__ = ["CanvasAction", "canvas_actions", "pill_title", "sectioned", "find", "frames_of",
           "SECTIONS", "LINKED_REGION_HINT"]
