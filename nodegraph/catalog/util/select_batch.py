"""Select Batch Member (``util.select_batch``) — narrow a batch to ONE of its files."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (
    BATCH_FILE_KEY, batch_select as _batch_select, batch_subset as _batch_subset)
from nodegraph.provider import BatchSliceProvider
from nodegraph.registry import Granularity, InDataset, InString, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (
    subset_lattice_layers, subset_structure_rows)

# ── Select Batch Member (V3.01) ──────────────────────────────────────────────────
#
# The tap `util.unbatch`'s synthetic per-member outputs materialize into at graph-build
# time — exactly the arrangement `channel.split` has with `channel.select` and `io.load`'s
# group taps have with `util.select_group`. The engine is one-payload-per-node, so K
# distinct per-member payloads have to come from K real tap nodes; the unbatch card itself
# is a pass-through.
#
# **The tap carries the member's NAME, never its index** — the rule `materialize_group_taps`
# already follows and for the same reason. An index would quietly point at a different file
# if the batch were rewired, with every hash still agreeing; a name either still resolves or
# refuses. It also makes the run graph read the way the user thinks.


def _compute_select_batch(ctx: EvalContext) -> Dataset:
    """Keep exactly one member of a batch, by name (or by 0-based index as a fallback).

    Resolved spec
    -------------
    * **Kind** utility, axis-changing → ``op_key="util.select_batch"``, category
      ``"utility"``. Narrows ``b`` to 1; every other axis is untouched.
    * **2D/3D** no lever — index arithmetic only.
    * **Footprint** ``TILEABLE``, no kernel axes: one output unit reads exactly the
      corresponding member's own unit.
    * **Backend** none — :class:`~nodegraph.provider.BatchSliceProvider` is a lazy pin.

    An empty selection is a genuine no-op (the input is returned untouched), the choice
    ``util.select_group`` and ``util.crop``'s frames mode both make, so an unconfigured
    node costs and stamps nothing. A name that does not resolve is REFUSED with the
    members listed — the alternative is handing back some other file's pixels under the
    name the user asked for.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Select Batch Member needs an image on its input Dataset.")
    ax = prov.axes
    raw = ctx.params.get("member")
    want = str(raw or "").strip()
    if not want:
        return ds

    names = _member_names(ds, ax)
    k = _resolve(want, names)
    if k is None:
        raise ValueError(_no_such_member(want, names))
    if int(ax.b) <= 1:
        # already a single member: selecting the only one it has is an identity, and
        # wrapping it in a view would cost a provider and a memo key for nothing.
        return ds

    new_axes = replace(ax, b=1)
    out = subset_lattice_layers(ds.with_image(BatchSliceProvider(prov, k)),
                                new_axes, {"b": [k]})
    out = subset_structure_rows(out, {"b": [k]})
    return out.with_metadata(**_batch_subset(ds.metadata, [k]))


def _member_names(ds: Dataset, ax: Any) -> List[str]:
    """The batch's member names, from the ``batch_file`` list ``util.batch`` stamped.

    Falls back to slot names for a Dataset whose list is absent or the wrong length —
    the same rule the per-M provenance family uses: a list that does not match the axis
    is not trusted to name anything, because a stale positional list reports the WRONG
    file rather than admitting it does not know.
    """
    got = ds.metadata.get(BATCH_FILE_KEY)
    n = max(1, int(getattr(ax, "b", 1)))
    if isinstance(got, (list, tuple)) and len(got) == n:
        return [str(v) for v in got]
    return [f"file{i}" for i in range(n)]


def _resolve(want: str, names: List[str]) -> Optional[int]:
    """Member index for ``want`` — an exact name, else a bare 0-based index."""
    for i, nm in enumerate(names):
        if nm == want:
            return i
    if want.isdigit():
        i = int(want)
        if 0 <= i < len(names):
            return i
    return None


def _no_such_member(want: str, names: List[str]) -> str:
    listing = "\n".join(f"  {i}  {nm}" for i, nm in enumerate(names))
    return (f"Select Batch Member: this batch has no member {want!r}. Its members are:\n"
            f"{listing}\n"
            f"Name one of those, or give its 0-based index. (If you rewired the Batch "
            f"node, the name you picked may belong to a file that is no longer in it.)")


register_node(
    _compute_select_batch,
    op_key="util.select_batch", label="Select Batch Member", category="utility",
    inputs=[
        InDataset(),
        InString("member", "Member",
                 description=
                 "WHICH FILE of a batch to keep, by the name Batch gave it — normally the "
                 "source file's own name. A 0-based index works too. EMPTY keeps the "
                 "batch whole, which makes an unconfigured node a true no-op.\n\n"
                 "Naming a member that is not there is refused, with the real members "
                 "listed, rather than falling back to one of them: a batch exists to keep "
                 "files apart, so quietly handing back the wrong file would defeat it. No "
                 "pixels are copied — this is a lazy view — but the batch axis really "
                 "narrows to one, and everything indexed by it follows."),
    ],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_batch_select,
    description="Keep one file of a batch, by name. The tap Unbatch's per-file outputs "
                "materialize into, and usable on its own to pull a single member out.",
)
