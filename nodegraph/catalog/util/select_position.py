"""Select Position (``util.select_position``) — keep ONE multipoint position of a Dataset,
by its acquisition name or its 0-based index; the tap Split Positions' per-position outputs
materialize into, and usable on its own."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (POSITION_NAME_KEY, format_indices, position_pick,
                                position_subset, select_position as _meta_select_position)
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import _sampled

# ── Select Position (2026-10-02) ─────────────────────────────────────────────────
#
# The multipoint member of the tap family: ``channel.split`` → ``channel.select``,
# ``io.load``'s group taps → ``util.select_group``, ``util.unbatch`` → ``util.select_batch``,
# and now ``util.split_positions`` → this node. The engine is one-payload-per-node, so K
# distinct per-position payloads come from K real tap nodes; the split card is a
# pass-through.
#
# **The tap carries the position's INDEX** — unlike the group and batch taps, which carry a
# key or a name. A position's index in its file IS its identity: the acquisition's point
# list is fixed when the file is written, every per-M list (`stage_xy_um`, `origin_um`,
# `position_name`) is addressed by it, and a name is an optional label many files do not
# carry at all. Rewiring the split onto a different file legitimately changes which position
# ``pos2`` means — the same way ``ch1`` means "the second channel of whatever is wired".


def _compute_select_position(ctx: EvalContext) -> Dataset:
    """Keep exactly one position of a multipoint Dataset.

    Resolved spec (build-node-v2 §0, 2026-10-02)
    --------------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.select_position"``, category
      ``"utility"``, ``meta_transform=select_position``.
    * **Data contract** ``Dataset → the same Dataset with m = 1``. No calibration changes:
      nothing is cut out of a plane, no axis is re-spaced, and the kept position keeps the
      ``origin_um`` it already had. Everything indexed BY m follows — the per-M metadata
      family (:data:`~nodegraph.metadata.PER_POSITION_KEYS`), lattice layers, structure
      rows (filtered and renumbered) — through the same helpers ``util.select_group`` uses.
    * **2D/3D** no lever: index arithmetic only.
    * **Footprint** ``TILEABLE``, no kernel axes — one output unit reads exactly the
      corresponding input unit of the kept position.
    * **Sockets** ``position`` (name or 0-based index). No modes.
    * **Backend** none — :class:`~nodegraph.provider.FrameSubsetProvider` with one position.

    Resolution is shared with the edit-time transform (:func:`~nodegraph.metadata.position_pick`)
    so the card and the pull agree. EMPTY is a genuine no-op (the input back, untouched); a
    value that names no position is REFUSED with the positions listed — handing back some
    other position's pixels under the name asked for is the failure a selector exists to
    prevent. A single-position input is returned as is, whatever is asked: selecting the
    only position there is, is the identity.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Select Position needs an image on its input Dataset.")
    ax = prov.axes
    raw = ctx.params.get("position")
    want = str(raw if raw is not None else "").strip()
    if not want:
        return ds
    k = position_pick(ds.metadata, int(ax.m), want)
    if k is None:
        raise ValueError(_no_such_position(want, _position_names(ds, int(ax.m))))
    if int(ax.m) <= 1:
        return ds
    new_axes = replace(ax, m=1)
    view = FrameSubsetProvider(prov, [k], tuple(range(ax.t)), None)
    picks = {"m": [k], "t": None, "z": None}
    out = subset_lattice_layers(ds.with_image(view), new_axes, picks)
    out = subset_structure_rows(out, picks)
    out = out.with_metadata(**position_subset(ds.metadata, [k]))
    origin = ctx.calib("origin_um")     # the envelope's, memo-fenced (util.crop does the same)
    if origin is not None:
        out = out.with_metadata(origin_um=origin)
    return _sampled(out, f"m:select_position[{format_indices([k])}]")


def _position_names(ds: Dataset, m: int) -> List[str]:
    got = ds.metadata.get(POSITION_NAME_KEY)
    if isinstance(got, (list, tuple)) and len(got) == m:
        return [str(v) for v in got]
    return [f"m{i}" for i in range(m)]


def _no_such_position(want: str, names: List[str]) -> str:
    listing = "\n".join(f"  {i}  {nm}" for i, nm in enumerate(names))
    return (f"Select Position: this Dataset has no position {want!r}. Its positions are:\n"
            f"{listing}\n"
            f"Name one of those, or give its 0-based index. (If you rewired the Split "
            f"Positions node onto a different file, the slot you picked may be past the end "
            f"of this one.)")


register_node(
    _compute_select_position,
    op_key="util.select_position", label="Select Position", category="utility",
    inputs=[
        InDataset(),
        InString("position", "Position",
                 description=
                 "WHICH POSITION (multipoint / stage location) to keep: its 0-based index "
                 "on the M axis, or the point name the acquisition recorded for it when the "
                 "file carries one (the names the Split Positions card shows). EMPTY keeps "
                 "every position, which makes an unconfigured node a true no-op.\n\n"
                 "A value that names no position is refused, with the real positions listed, "
                 "rather than falling back to one of them. No pixels are copied — this is a "
                 "lazy view — but M really narrows to one, and everything indexed by it "
                 "(stage coordinates, per-position masks, labels and measurement rows) "
                 "follows."),
    ],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_select_position,
    description="Keep one multipoint position, by index or acquisition name. The tap Split "
                "Positions' per-position outputs materialize into, and usable on its own.",
)
