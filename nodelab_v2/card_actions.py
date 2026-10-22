"""Card actions (2026-10-22) — a button on a node's card for something only the GUI can do.

The engine has no notion of a button: a compute takes a Dataset and returns one. Some nodes
still have a natural next step that edits the WORKSPACE rather than the node — Load Table's
*Load file* opens the file its table was exported from on the Image Input page and wires it
in through a Page Input on the node's own page. Those steps are declared here, by op key,
rather than on :class:`~nodegraph.registry.NodeSpec`, because nothing below the GUI can carry
them out; the window does (``MainWindow._on_card_action``).

The card draws one row per node that has actions — a muted note on the left (what the action
would act on) and the buttons on the right — and the inspector shows the same buttons, so
the two surfaces cannot offer different things. Qt-free: the card, the inspector and the
probe all read :func:`actions_for`.
"""
from __future__ import annotations

import os
from typing import Any, List, NamedTuple

from nodegraph import table_file as TF

#: Load Table's op key — named once for the card, the inspector and the window.
LOAD_TABLE_OP = "io.load_table"
#: Load Table's *Load file*: the export's source image onto the Image Input page, wired in.
LOAD_SOURCE_ACTION = "load_source"


class CardAction(NamedTuple):
    """One button: its key (what the window dispatches on), its label, the card row's
    left-hand note, whether it can run now, and its hover text (which says why not)."""

    key: str
    label: str
    note: str
    enabled: bool
    tip: str


def _load_table_action(rec: Any) -> CardAction:
    path = TF.clean_path((getattr(rec, "params", None) or {}).get("path"))
    label = "Load file"
    if not path:
        return CardAction(LOAD_SOURCE_ACTION, label, "choose a table file", False,
                          "Choose a table file first: Load file opens the image it was "
                          "exported from.")
    if not os.path.isfile(path):
        return CardAction(LOAD_SOURCE_ACTION, label, "file not found", False,
                          f"No file at {path}.")
    origin = TF.read_origin(path)
    srcs = TF.source_files(origin)
    summary = TF.origin_summary(origin, path)
    if not srcs:
        return CardAction(LOAD_SOURCE_ACTION, label,
                          "no source recorded" if origin else "no origin record", False,
                          summary + "\n\nThis file names no source image, so there is "
                          "nothing to load — wire one into the node by hand and set Onto to "
                          "image.")
    first = os.path.basename(srcs[0].replace("\\", "/"))
    more = f" +{len(srcs) - 1}" if len(srcs) > 1 else ""
    found = [TF.locate_source(s, path) for s in srcs]
    where = ("" if all(found) else
             "\n\nNot at its recorded path, nor beside the table — Load file will ask "
             "where it is now.")
    return CardAction(
        LOAD_SOURCE_ACTION, label, f"source {first}{more}", True,
        summary + f"\n\nLoad file: open {first}{more} on the Image Input page (or reuse it "
        f"if it is already loaded there), read it here through a Page Input, wire that "
        f"into this node and set Onto to image — so the Viewer draws the table over the "
        f"image it was measured on." + where)


def actions_for(rec: Any) -> List[CardAction]:
    """The card actions of node record ``rec`` (``[]`` for every node without any). Total —
    a probe that fails offers nothing rather than breaking a paint."""
    try:
        if getattr(rec, "op_key", "") == LOAD_TABLE_OP:
            return [_load_table_action(rec)]
    except Exception:  # noqa: BLE001 — a card must always paint
        return []
    return []


__all__ = ["LOAD_TABLE_OP", "LOAD_SOURCE_ACTION", "CardAction", "actions_for"]
