"""split grouping — the GROUPING strategy every split card carries (2026-10-07).

A split card (Split Channels / Split Positions / Split Z / Split T) fans one axis out into
output sockets. How many, and what each carries, is a *strategy*:

* ``none``   — one output per index, while the axis is short enough to fan out
  (:data:`nodelab_v2.document.FANOUT_CAP`); the family's original behaviour;
* ``every``  — consecutive groups of ``group_size`` indices from 0, the last one clipped;
* ``ranges`` — the ranges typed into ``groups`` (``0-3; 4-7``, ``top: 0-3``, or ``every 4``).

The Mode and its two gated sockets are declared here once and forwarded by the four nodes,
so the dropdown reads the same on every card. All three are **presentation-only**: the
split's own compute is a pass-through and never reads them — the GUI resolves the strategy
into synthetic sockets (:meth:`nodelab_v2.document.GraphDocument.split_groups`) and the
run-graph build into taps (:func:`nodelab_v2.ops.split_group`), both through
:func:`nodegraph.metadata.split_plan` so the card and the run graph cannot disagree.
"""

from __future__ import annotations

from typing import List

from nodegraph.registry import InInt, InString, Mode, SocketSpec

#: the Mode's name, read by the document and the run-graph build
GROUPING_MODE = "grouping"
GROUPING_CHOICES = ("none", "every", "ranges")


def grouping_mode(noun: str, axis: str) -> Mode:
    """The ``grouping`` Mode for a split card whose members are ``noun`` (``"plane"``) on
    ``axis`` (``"Z"``)."""
    return Mode(
        GROUPING_MODE, list(GROUPING_CHOICES), default="none", label="Grouping",
        description=
        f"How this card fans the {axis} axis out into outputs. `none` is one output per "
        f"{noun}; `every` cuts the axis into consecutive groups of a size you set; `ranges` "
        f"lets you type exactly which {noun}s go together, with names. Each output carries "
        f"its subset of the data; `out` always carries everything. This shapes the card and "
        f"the taps it materializes into, never the node's own result.",
        choice_docs={
            "none":
                f"One output socket per {noun}, labelled with what it carries, while the axis "
                f"has at most 24 members — past that the card offers only `out` and one of the "
                f"other two strategies is the way to split. Each wired socket becomes a "
                f"single-{noun} tap at run time. The family's original behaviour.",
            "every":
                f"Consecutive groups of `Group size` {noun}s, counted from 0: with a size of 10 "
                f"on 95 {noun}s you get ten outputs, the last one holding five. The usual way to "
                f"cut a long axis into windows without typing them; each output is a Crop (or "
                f"a Select Channel) keeping that run at run time.",
            "ranges":
                f"Exactly the groups you type into `Ranges`: `0-3; 4-7; 8-11` (inclusive, "
                f"0-based, `;` between groups), optionally named — `top: 0-3; mid: 4-7` — and "
                f"groups may overlap or skip {noun}s. A group reaching past the end is "
                f"labelled so; what lies past the end is dropped at the pull and a group "
                f"entirely past it is refused with the real {axis} length.",
        })


def grouping_sockets(noun: str, axis: str, tap: str, default_size: int) -> List[SocketSpec]:
    """The two gated sockets: ``group_size`` for ``every``, ``groups`` for ``ranges``."""
    return [
        InInt("group_size", "Group size", unit="", field=False, default=int(default_size),
              presentation=True,
              available_in={GROUPING_MODE: frozenset({"every"})},
              description=
              f"How many consecutive {noun}s each group holds under `every`, counted from "
              f"index 0; the last group is whatever remains. SMALLER means more outputs, each "
              f"carrying less; LARGER means fewer, wider windows. 1 is one output per {noun} "
              f"with no fan-out cap. Each group materializes into {tap} at run time. Only read "
              f"when Grouping is `every`."),
        InString("groups", "Ranges", field=False, default="", presentation=True,
                 available_in={GROUPING_MODE: frozenset({"ranges"})},
                 description=
                 f"Which {noun}s go together under `ranges`, as 0-based indices and inclusive "
                 f"ranges with `;` between groups — `0-3; 4-7; 8-11` — each optionally named "
                 f"(`top: 0-3; mid: 4-7`); `every 4` is accepted here too. The card grows one "
                 f"output PER GROUP, labelled with the name or the members, and each "
                 f"materializes into {tap} at run time. A group reaching past the end is "
                 f"labelled so on the card; what lies past the end is dropped at the pull, and "
                 f"a group entirely past it is refused with the real {axis} length. Blank = no "
                 f"groups yet. Only read when Grouping is `ranges`."),
    ]


__all__ = ["GROUPING_MODE", "GROUPING_CHOICES", "grouping_mode", "grouping_sockets"]
