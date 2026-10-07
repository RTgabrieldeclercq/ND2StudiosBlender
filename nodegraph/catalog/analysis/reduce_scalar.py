"""Reduce → Scalar (``analysis.reduce_scalar``) — Collapse any attribute layer or structure column to ONE number on the Global domain — object count, mean area, total mask volume."""

from __future__ import annotations

import numpy as np

from typing import FrozenSet, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.reducers import reduce as _reduce, reducer_docs
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.units import unit_of, with_unit

from nodegraph.catalog._base import register_node

# ── Reduce → Scalar (any layer/column → one Global number) ─────────────────────
#
# The metric source. Before this node NOTHING in the catalog wrote a `Domain.GLOBAL`
# layer — only selftest fixtures did — so "pick the best iteration" had no score to read
# and `transform.transfer_domain`/`transfer_structure` were the only route to a scalar (a
# three-card chain for "count the objects"). This is that chain as one card.

#: Domains this node can reduce FROM. Global is excluded (reducing a scalar to a scalar is
#: a no-op control) and Mesh is excluded (its CSR strata are not a value column — see
#: `nodegraph.mesh`, which owns the only sanctioned read path).
_RS_DOMAINS: Tuple[str, ...] = (
    "voxel", "plane", "frame", "timepoint", "multipoint", "channel",
    "label", "point", "track")
#: The structure domains among them — the ones whose columns are filed under a table
#: LAYER, so a name can be ambiguous and `table` becomes live.
_RS_STRUCTURE: FrozenSet[str] = frozenset({"label", "point", "track"})
_RS_REDUCERS: Tuple[str, ...] = (
    "count", "mean", "sum", "max", "min", "median")
def _compute_reduce_scalar(ctx: EvalContext) -> Dataset:
    """Reduce one attribute layer / structure column to a single ``Global`` scalar.

    **Resolved spec.** Category ``analysis``; Dataset in, the same Dataset out plus one
    ``Domain.GLOBAL`` layer named by ``name``. No 2D/3D lever — it reads an attribute that
    is already computed, so dimensionality is whatever produced it; ``WHOLE_SERIES`` with
    no kernel axes, because a reduction over *every* axis is by definition not tileable.
    ``reads_domains`` is left EMPTY deliberately: which domain it reads is per-INSTANCE
    (the ``domain`` lever) while the declaration is per-TYPE — the same reason
    ``transform.transfer_domain`` leaves it empty.

    A ``Global`` scalar is what ``flow.iterate``'s ``best`` mode compares, but nothing here
    knows that: this is an ordinary measurement node and the scalar is readable by anything.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    domain = Domain(str(modes.get("domain", "label")))
    reducer = str(modes.get("reducer", "count"))
    src = ctx.layer("source")
    present = ds.layers_on(domain)
    matches = [a for a in present if a.name == src]
    if not matches:
        # **An empty domain is a RESULT, not an error.** A threshold that found nothing
        # leaves no Label table at all, and a sweep must be able to report that iteration
        # as "0 objects" rather than die and take the other iterations with it. The typo
        # case stays a hard error, and the two are distinguishable without guessing: the
        # domain is a closed Mode, so if it carries *other* attributes the name is wrong.
        if present:
            raise ValueError(
                f"reduce_scalar: no {domain.value} attribute {src!r} on the input — it has "
                f"{sorted({a.name for a in present})}")
        empty = 0.0 if reducer in ("count", "sum") else float("nan")
        return with_unit(ds.with_layer(Domain.GLOBAL, ctx.layer("name"), np.asarray(empty)),
                         Domain.GLOBAL, ctx.layer("name"), "" if reducer == "count" else None)
    if len(matches) > 1:
        # One name filed under several structure TABLES (`with_structure` keys each column
        # as (domain, table, name)), so the name alone is ambiguous — say which tables.
        want = str(ctx.params.get("table", "") or "")
        narrowed = [a for a in matches if (a.layer or "") == want] if want else []
        if len(narrowed) != 1:
            raise ValueError(
                f"reduce_scalar: {domain.value} attribute {src!r} exists on "
                f"{sorted(a.layer or '' for a in matches)} — set 'Table' to the one you "
                f"mean")
        matches = narrowed
    arr = np.asarray(matches[0].values, dtype=float)
    value = float(np.asarray(_reduce(arr, tuple(range(arr.ndim)), reducer)).reshape(-1)[0])
    out = ds.with_layer(Domain.GLOBAL, ctx.layer("name"), np.asarray(value))
    # the scalar's UNIT (V4.00 step 12): a count is a plain number; every other reducer is in
    # the attribute's own unit, as its producer recorded it or as its name says
    # (`nodegraph.units.unit_of`) — so a Math card downstream knows what it was handed
    unit = "" if reducer == "count" else unit_of(ds, domain, src, matches[0].layer)
    return with_unit(out, Domain.GLOBAL, ctx.layer("name"), unit)
register_node(
    _compute_reduce_scalar, op_key="analysis.reduce_scalar", label="Reduce → Scalar",
    category="analysis",
    adds_domains=frozenset({Domain.GLOBAL}),
    inputs=[
        InDataset(),
        InString("source", "Attribute", field=False, default="area",
                 layer_in_mode="domain",
                 description=
                 "Which attribute to reduce. It must exist on the domain named by the From "
                 "lever, so the picker follows THAT lever — change From and the offered "
                 "names change with it. Counting objects needs no real column: 'count' on "
                 "any Label column returns how many rows there are."),
        InString("table", "Table", field=False, default="",
                 available_in={"domain": frozenset(_RS_STRUCTURE)},
                 description=
                 "Only needed when the same column name exists on two structure tables "
                 "(two detections both carrying 'area'), which is the only case the node "
                 "cannot resolve on its own. Leave it empty otherwise; if it is needed the "
                 "node refuses and lists the tables to choose from."),
        InString("name", "Output scalar", field=False, default="score",
                 layer_out=(Domain.GLOBAL,),
                 description=
                 "What to call the number on the Global domain. This is the name you type "
                 "into an Iterate node's Metric field, so give it something you will "
                 "recognize there — 'n_cells', 'mean_area' — rather than leaving it "
                 "'score' on all three of them."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("domain", list(_RS_DOMAINS), default="label", label="From",
                description=
                "Which domain the attribute being reduced lives on. It filters the Attribute "
                "picker to the layers actually present on that domain, so set it FIRST — with "
                "the wrong domain selected the name you want is not on the list. Global is "
                "absent (reducing a scalar to a scalar does nothing) and so is Mesh (its "
                "strata are topology, not a value column).",
                choice_docs=domain_docs(_RS_DOMAINS)),
           Mode("reducer", list(_RS_REDUCERS), default="count", label="Reduce",
                description=
                "How the whole layer or column collapses to one number. Every element is "
                "pooled — there is no per-axis grouping here, that is what the Transfer nodes "
                "are for — so this is the statistic the sweep or the report ends up "
                "comparing.",
                choice_docs=reducer_docs(_RS_REDUCERS))],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Collapse any attribute layer or structure column to ONE number on the "
                "Global domain — object count, mean area, total mask volume. The scalar a "
                "sweep compares, and the only node in the catalog that writes Global.")
