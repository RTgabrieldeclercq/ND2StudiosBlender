"""Run an eager compute once per batch member, and join the results back (V3.01).

**Why this exists.** Roughly 25 catalog nodes realize a whole ``(m,t,z,c,y,x)`` raster and
loop it by hand. The batch index is keyword-only with a default of 0
(:data:`nodegraph.provider.TileProvider.read_region`), so such a loop reads the FIRST file
and writes it into an array shaped like the whole batch — INV-14's silent-wrong-data bug.
Teaching each of those loops the batch axis is 25 edits, each with its own chance of
getting the indexing subtly wrong, in nodes whose maths nobody wants to touch.

This is the other way round: leave every loop exactly as it is, and give it a dataset it
already understands. :func:`per_member` splits a ``b == K`` Dataset into K one-member
Datasets, runs the unmodified compute on each, and joins the K results on ``b``. The node
never learns what a batch is.

**It is not a shortcut around the semantics — it IS the semantics.** A compute running on
one member sees exactly one file's population, so every data-derived level
(``scope="dataset"`` most of all) is per-file by construction rather than by a scope key
someone remembered to thread. That is the property the batch axis was added for.

**What it costs.** K separate eager computes instead of one, which is what K files honestly
cost; nothing is duplicated except the per-member Python overhead. Members are independent,
so this is also where a batch could fan out across cores — deliberately NOT done here,
because these computes already call :func:`nodegraph.parallel.map_units` internally and
that map is re-entrant-guarded: nesting would serialize the inner one and win nothing.

Qt-free.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from nodegraph.dataset import AttributeLayer, Dataset
from nodegraph.domains import Domain, axes_of, is_lattice
from nodegraph.metadata import BATCH_FILE_KEY
from nodegraph.provider import BatchProvider, BatchSliceProvider
from nodegraph.structure import BATCH_COLUMN

__all__ = ["is_batched", "split_members", "join_members", "per_member", "batch_aware"]


def batch_aware(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a compute so a batched input is run **once per member** and joined.

    The point is that ``fn`` is not modified and does not know. It is handed an ordinary
    one-member Dataset through a rebound :class:`~nodegraph.engine.EvalContext`, so its
    hand-written ``(m,t,z,c)`` loop, its ``dense_output`` shape and its scope keys are all
    correct as written — which is why this is applied at the ``register_node`` call rather
    than inside 25 different loops.

    Unbatched input is passed straight through to ``fn``, unwrapped, so an ordinary graph
    executes exactly the code it always did and memoizes to the same bytes.

    The rebind also swaps the member into ``by_name``, since a compute may reach its
    primary input either way and the two must not disagree about which file it is on.
    Auxiliary Dataset inputs (a ``reference``, a ``raw``) are left alone: they are a
    different wire, and if one is itself batched that is a graph the node has to be told
    about rather than something to silently pair up member-wise.
    """
    def wrapped(ctx):
        ins = tuple(getattr(ctx, "inputs", ()) or ())
        ds = ins[0] if ins else None
        if not is_batched(ds):
            return fn(ctx)

        def _one(member: Dataset):
            by_name = getattr(ctx, "by_name", None)
            swapped = ({k: (member if v is ds else v) for k, v in by_name.items()}
                       if by_name else by_name)
            return fn(replace(ctx, inputs=(member,) + ins[1:], by_name=swapped))

        return join_members([_one(m) for m in split_members(ds)], ds)

    # keep the identity the reloader and the memo's code fingerprint key on
    for _a in ("__name__", "__qualname__", "__doc__", "__module__", "__wrapped__"):
        try:
            setattr(wrapped, _a, getattr(fn, _a, fn))
        except (AttributeError, TypeError):      # pragma: no cover - builtins/partials
            pass
    wrapped.__wrapped__ = fn
    return wrapped


def is_batched(ds: Any) -> bool:
    """True when ``ds`` is a Dataset carrying more than one batch member."""
    return isinstance(ds, Dataset) and int(getattr(ds.axes, "b", 1)) > 1


def split_members(ds: Dataset) -> List[Dataset]:
    """``ds`` as K independent one-member Datasets, in member order.

    The image becomes a :class:`~nodegraph.provider.BatchSliceProvider` — a lazy pin, no
    copy. A lattice layer that carries ``b`` is indexed on its leading axis; one that does
    not (``channel``, ``global``) is shared unchanged, because it is not per-file. Structure
    rows are filtered to the member and their ``b`` column dropped, so the compute sees the
    ordinary un-batched table it was written for.
    """
    nb = int(ds.axes.b)
    one_axes = replace(ds.axes, b=1)
    labels = _batch_member_labels(ds, nb)
    out: List[Dataset] = []
    for b in range(nb):
        member = Dataset(axes=one_axes, metadata=dict(ds.metadata))
        if ds.image is not None:
            member = member.with_image(BatchSliceProvider(ds.image, b))
        # this member's own name, as a one-entry list, so anything downstream that reads
        # the provenance sees a single-file Dataset rather than the whole batch's list
        member = member.with_metadata(**{BATCH_FILE_KEY: [labels[b]]})
        for attr in ds.attributes.values():
            member = _member_attribute(member, ds, attr, b)
        out.append(member)
    return out


def _batch_member_labels(ds: Dataset, nb: int) -> List[str]:
    got = ds.metadata.get(BATCH_FILE_KEY)
    if isinstance(got, (list, tuple)) and len(got) == nb:
        return [str(v) for v in got]
    return [f"file{i}" for i in range(nb)]


def _member_attribute(member: Dataset, ds: Dataset, attr: AttributeLayer,
                      b: int) -> Dataset:
    """Carry one attribute layer onto ``member``, narrowed to batch index ``b``."""
    vals = np.asarray(attr.values)
    if is_lattice(attr.domain):
        if "b" not in (axes_of(attr.domain) or frozenset()):
            return member.with_layer(attr.domain, attr.name, vals, attr.layer)
        if vals.shape == ds.axes.shape_for(attr.domain):
            return member.with_layer(attr.domain, attr.name, vals[b], attr.layer)
        return member                      # already stale against these axes; drop it
    # structure: keep this member's rows, minus the column that identified them
    keep = _member_rows(ds, attr, b)
    if keep is None or attr.name == BATCH_COLUMN:
        return member
    return member.with_layer(attr.domain, attr.name, vals[keep], attr.layer)


def _member_rows(ds: Dataset, attr: AttributeLayer, b: int):
    """Row mask selecting member ``b``'s rows of ``attr``'s table, or ``None`` when the
    table carries no batch column (nothing to split, so nothing is carried)."""
    col = ds.get(attr.domain, BATCH_COLUMN, attr.layer)
    if col is None:
        return None
    return np.asarray(col.values) == b


def join_members(members: Sequence[Dataset], source: Dataset) -> Dataset:
    """K one-member results back into one ``b == K`` Dataset.

    ``source`` is the batched input, read only for its member names. Images re-stack
    through :class:`~nodegraph.provider.BatchProvider`; lattice layers stack on a new
    leading axis; structure rows concatenate with their member stamped into
    :data:`~nodegraph.structure.BATCH_COLUMN` and their ids **offset per member**, because
    every producer numbers its regions from 1 within the data it was given and two members
    would otherwise both hold an id 1 that a downstream join could not tell apart.
    """
    if not members:
        raise ValueError("join_members: nothing to join")
    nb = len(members)
    labels = _batch_member_labels(source, nb)
    first = members[0]
    axes = replace(first.axes, b=nb)
    out = Dataset(axes=axes, metadata=dict(first.metadata))
    out = out.with_metadata(**{BATCH_FILE_KEY: list(labels)})
    if all(m.image is not None for m in members):
        out = out.with_image(BatchProvider([m.image for m in members], labels=labels))

    # lattice layers: stack the ones addressed by b, keep the first member's for the rest
    names = {(a.domain, a.layer, a.name) for m in members for a in m.attributes.values()}
    for (domain, layer, name) in sorted(names, key=lambda k: (k[0].value, k[1] or "", k[2])):
        got = [m.get(domain, name, layer) for m in members]
        if any(g is None for g in got):
            continue                       # not produced for every member — not joinable
        arrs = [np.asarray(g.values) for g in got]
        if is_lattice(domain):
            if "b" in (axes_of(domain) or frozenset()):
                out = out.with_layer(domain, name, np.stack(arrs, axis=0), layer)
            else:
                out = out.with_layer(domain, name, arrs[0], layer)
        else:
            out = out.with_layer(domain, name, np.concatenate(arrs, axis=0), layer)

    out = _stamp_structure_batch(out, members)
    return _carry_struct_zkind(out, members)


def _stamp_structure_batch(out: Dataset, members: Sequence[Dataset]) -> Dataset:
    """Add the batch column to every joined structure table, and make its ids unique.

    Ids are offset by the running row total rather than renumbered, so a member's internal
    references (a Track's ``member_id``, a raster's label values) stay findable — the same
    globally-unique-id discipline every structure producer in the catalog already uses.
    """
    groups: Dict[Any, List[int]] = {}
    for attr in out.attributes.values():
        if not is_lattice(attr.domain):
            groups.setdefault((attr.domain, attr.layer), []).append(0)
    for (domain, layer) in list(groups):
        counts = [_rows_of(m, domain, layer) for m in members]
        if any(c is None for c in counts):
            continue
        bcol = np.concatenate([np.full(int(c), i, dtype=np.int64)
                               for i, c in enumerate(counts)]) if counts else None
        if bcol is None or bcol.size == 0:
            continue
        out = out.with_layer(domain, BATCH_COLUMN, bcol, layer)
        ids = out.get(domain, "id", layer)
        if ids is not None:
            offs, run = [], 0
            for c in counts:
                offs.append(run)
                run += int(c)
            shift = np.concatenate([np.full(int(c), o, dtype=np.int64)
                                    for c, o in zip(counts, offs)])
            out = out.with_layer(domain, "id",
                                 np.asarray(ids.values, dtype=np.int64) + shift, layer)
    return out


def _rows_of(ds: Dataset, domain: Domain, layer: Optional[str]) -> Optional[int]:
    for attr in ds.attributes.values():
        if attr.domain is domain and attr.layer == layer and not is_lattice(domain):
            return int(len(np.asarray(attr.values)))
    return None


def _carry_struct_zkind(out: Dataset, members: Sequence[Dataset]) -> Dataset:
    """Preserve each structure instance's ``z_kind`` provenance across the join — it lives
    in metadata rather than in a column, so the stacking above would drop it and every
    downstream node that INHERITS dimensionality (`wire-node-v2` §7b) would start guessing."""
    zmap: Dict[str, Any] = {}
    for m in members:
        zmap.update(dict(m.metadata.get("__struct_zkind__", {}) or {}))
    return out.with_metadata(__struct_zkind__=zmap) if zmap else out


def per_member(ds: Dataset, fn: Callable[[Dataset], Dataset]) -> Dataset:
    """Run ``fn`` on each batch member of ``ds`` and join the results.

    A pass-through when ``ds`` is not batched, so a node can call this unconditionally and
    a single-file graph pays nothing and keeps byte-identical output.
    """
    if not is_batched(ds):
        return fn(ds)
    return join_members([fn(one) for one in split_members(ds)], ds)
