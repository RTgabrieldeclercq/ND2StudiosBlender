"""split grouping — the `groups` socket every split card carries (2026-10-07).

A split card (Split Channels / Split Positions / Split Z / Split T) fans one axis out into
output sockets, one per index while the axis is short enough
(:data:`nodelab_v2.document.FANOUT_CAP`). GROUPING lets the user gather several of those
outputs into one: on the card every data output has a **checkbox**, and **Group selected**
turns the ticked ones into a single group socket carrying that subset; a group socket's
checkbox plus **Ungroup** dissolves it again. The Properties panel shows the same list
with the same buttons, and also *Group every N* for a long axis.

What the gesture writes is this one STRING socket, ``groups`` — ``0-3; 4-7``, a group
optionally named ``top: 0-3`` — the storage, which can also be typed. It is
**presentation-only**: the split's own compute is a pass-through and never reads it. The
GUI resolves the text into synthetic sockets
(:meth:`nodelab_v2.document.GraphDocument.split_groups`) and the run-graph build into taps
(:func:`nodelab_v2.ops.split_group`), both through :func:`nodegraph.metadata.parse_groups`,
so a socket on the card and the tap it becomes cannot disagree about what a group holds.
"""

from __future__ import annotations

from typing import List

from nodegraph.registry import InString, SocketSpec

#: the socket's name, read by the document and the run-graph build
GROUPS_PARAM = "groups"


def grouping_sockets(noun: str, axis: str, tap: str) -> List[SocketSpec]:
    """The ``groups`` socket for a split card whose members are ``noun`` (``"plane"``) on
    ``axis`` (``"Z"``); ``tap`` names what a wired group becomes at run time."""
    return [
        InString(GROUPS_PARAM, "Groups", field=False, default="", presentation=True,
                 description=
                 f"Which {noun}s are gathered into one output. On the card, tick the "
                 f"checkbox beside each {noun} output and press GROUP SELECTED: the ticked "
                 f"{noun}s become one group socket carrying that subset (each materializes "
                 f"into {tap} at run time); tick a group and press Ungroup to dissolve it. "
                 f"This text is where the groups are kept — 0-based indices and inclusive "
                 f"ranges, `;` between groups, each optionally named: `top: 0-3; 4-7` — and "
                 f"may be typed too. A group reaching past the end is labelled so on the "
                 f"card; what lies past the end is dropped at the pull, and a group entirely "
                 f"past it is refused with the real {axis} length. Blank = no groups: one "
                 f"output per {noun}. Never read by the node's own result."),
    ]


__all__ = ["GROUPS_PARAM", "grouping_sockets"]
