"""Transfer Domain (``transform.transfer_domain``) — Move a lattice attribute between domains (reduce coarsens over dropped axes / broadcast refines);."""

from __future__ import annotations


from typing import Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, axes_of, domain_docs, is_lattice
from nodegraph.engine import EvalContext
from nodegraph.reducers import reducer_docs
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _lattice_layers, _resolve_layer

# ── Transfer Domain (move a lattice attribute A→B: reduce / broadcast) ──────────
#
# The from/to dropdowns offer the LATTICE domains ONLY: `execute_transfer` raises
# NotImplementedError on a BridgeStep, so a Label/Point/Track/Mesh entry would be a
# control that always errors. Every lattice pair IS routable (transfer generates a
# reduce/broadcast for any pair), so the cross-product holds no dead choice.
_TRANSFER_DOMAINS: Tuple[str, ...] = tuple(d.value for d in Domain if is_lattice(d))
#: Mirrors nodegraph.reducers.REDUCERS. This node is WHOLE_VOLUME and reduces eagerly,
#: so the non-monoid entries (median / sigma_clip / trimmed_mean) are legal here.
_TRANSFER_REDUCERS: Tuple[str, ...] = (
    "mean", "sum", "max", "min", "median", "count", "first",
    "sigma_clip", "trimmed_mean")
def _compute_transfer_domain(ctx: EvalContext) -> Dataset:
    """Transfer a **lattice** attribute from one domain to another — coarsening reduces
    over the dropped axes, refining broadcasts (a generated :class:`TransferPlan`, wrapped
    as a node). Structure-domain (Label/Point/Track/Mesh) transfers need the geometry
    inputs the bridges carry (``execute_bridge_plan``, C2) and are refused here — which is
    why they are absent from the two dropdowns.

    Resolved spec (2026-07-28 socket fix): ALL FOUR functional params were previously
    unreachable — the node declared only ``InDataset()``, so from the GUI it could only
    ever move ``mask`` from voxel→frame with ``mean``, i.e. it was effectively unusable.
    ``from_domain``/``to_domain``/``reducer`` are **Modes** because each is a closed
    enumeration (the `util.stack`/`analysis.threshold` precedent for a fixed choice list);
    ``attr`` is a layer NAME, so it stays an ``InString`` like every other source-layer
    selector in the catalog."""
    from nodegraph.transfer import lattice_transfer
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})

    # These three were PARAMS before they became Modes. The GUI could never have written
    # them (no sockets — the defect being fixed), but a headless or hand-authored graph
    # could, and nodegraph.selftest itself did. A silent fallback is impossible: the
    # engine passes the RESOLVED mode state (engine.py:404 `dict(state)`, defaults filled
    # in), so a param could only ever be consulted by second-guessing whether a mode value
    # was chosen or defaulted. Refusing is the honest option — a stale caller gets told
    # exactly what to change instead of silently running with the defaults, which is the
    # very failure this whole pass exists to remove.
    _legacy = [k for k in ("from_domain", "to_domain", "reducer") if k in ctx.params]
    if _legacy:
        raise ValueError(
            f"transfer_domain: {_legacy} are Modes now, not params — pass them as "
            f"modes={{{', '.join(f'{k!r}: ...' for k in _legacy)}}} on the NodeInstance. "
            "(They were params with no socket, so the GUI could never set them; only a "
            "headless caller reaches this.)")

    src = Domain(modes.get("from_domain", "voxel"))
    dst = Domain(modes.get("to_domain", "frame"))
    name = ctx.layer("attr")
    reducer = modes.get("reducer", "mean")
    if not (is_lattice(src) and is_lattice(dst)):
        raise ValueError(
            f"transfer_domain moves LATTICE attributes; {src.value}→{dst.value} needs a "
            f"structure bridge — use nodegraph.bridges / execute_bridge_plan (C2)")
    # wire-node-v2 §5b remedy (b): `reducer` only bites when the transfer COARSENS (there
    # are axes to collapse). A pure refinement/identity broadcasts the value untouched, so
    # a non-default reducer there would be a live control the kernel silently ignores —
    # refuse instead of lying. `available_in` can only gate a RECTANGLE of mode values and
    # "src drops an axis vs dst" is not one (it would have to admit frame→voxel to admit
    # voxel→frame), so the declarative remedy (a) is not expressible here.
    # Checked BEFORE the layer lookup: it is a pure configuration error, and the user
    # should see it whether or not the named attribute happens to exist.
    if not (axes_of(src) - axes_of(dst)) and reducer != "mean":
        raise ValueError(
            f"transfer_domain: {src.value}→{dst.value} drops no axis (a pure "
            f"broadcast/identity), so reducer={reducer!r} would be ignored — set "
            f"reducer back to 'mean'")
    layer = ds.get(src, name)
    if layer is None:
        # the one attribute of the source domain on the wire, whatever it is called — the
        # socket ships `mask`, which is right only for `analysis.threshold`'s own default
        name, _note = _resolve_layer(
            _lattice_layers(ds, src), name, node="transfer_domain", socket="attr",
            what=f"{src.value} attribute", where="the `data` input",
            remedy=f"this moves an existing {src.value} attribute to {dst.value}, so run the "
                   f"node that produces one upstream", ctx=ctx)
        layer = ds.get(src, name)
    return ds.with_attribute(lattice_transfer(layer, dst, ds.axes, reducer))
register_node(
    _compute_transfer_domain, op_key="transform.transfer_domain",
    label="Transfer Domain", category="transform",
    # reads/adds stay EMPTY on purpose: both are per-INSTANCE here (whatever from_domain/
    # to_domain say) while NodeSpec's declarations are per-TYPE — the same reason
    # `track.link` leaves reads_domains empty for its Label-OR-Point target mode.
    inputs=[InDataset(),
            # the source DOMAIN is the `from_domain` lever, so the picker resolves
            # it from that mode rather than from a fixed domain
            InString("attr", "Attribute", field=False, default="mask",
                     layer_in_mode="from_domain",
                     description=
                     "Which attribute layer to move between domains. It must exist on the "
                     "domain named by the From lever, so the picker follows THAT lever rather "
                     "than a fixed domain — change From and the offered names change with it. "
                     "The output layer keeps this same name on the To domain. Whether the "
                     "values are averaged, summed or simply copied is the Reduce lever's job: "
                     "going to a coarser domain (voxel → frame) reduces many values into one, "
                     "while going finer broadcasts one value to many and Reduce is then "
                     "ignored.")],
    outputs=[OutDataset()],
    modes=[Mode("from_domain", list(_TRANSFER_DOMAINS), default="voxel", label="From",
                description=
                "The domain the attribute currently lives on. Only LATTICE domains are "
                "offered — Label/Point/Track/Mesh need the geometry a structure bridge "
                "carries, which is Transfer Structure's job — and the Attribute picker "
                "follows this lever, so set it before choosing the layer.",
                choice_docs=domain_docs(_TRANSFER_DOMAINS)),
           Mode("to_domain", list(_TRANSFER_DOMAINS), default="frame", label="To",
                description=
                "The domain to move the attribute ONTO. Compare its axes with From's: fewer "
                "axes means coarsening, so the dropped axes are collapsed with Reduce; more "
                "axes means refining, so each value is broadcast unchanged to every finer "
                "position and Reduce is not used (setting it anyway is refused rather than "
                "silently ignored).",
                choice_docs=domain_docs(_TRANSFER_DOMAINS)),
           Mode("reducer", list(_TRANSFER_REDUCERS), default="mean", label="Reduce",
                description=
                "How many values become one when the transfer COARSENS — applied over exactly "
                "the axes From has and To does not. It is read only in that direction; on a "
                "pure broadcast or an identity transfer anything but the default is an error, "
                "because a control the kernel ignores is worse than one that refuses.",
                choice_docs=reducer_docs(_TRANSFER_REDUCERS))],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Move a lattice attribute between domains (reduce coarsens over dropped "
                "axes / broadcast refines); lattice ONLY — for Label/Point/Track use "
                "Transfer Structure.")
