"""Connected Components (``analysis.label``) — Label a mask into connected regions (2D 4/8-conn vs 3D 6/18/26-conn);."""

from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import Dict

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import fold_units
from nodegraph.registry import DimMode, Granularity, InDataset, InInt, InString, OutDataset
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable, label_components

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _voxel_layers
from nodegraph.catalog._shared.progress import _parallel_progress

def _compute_label(ctx: EvalContext) -> Dataset:
    """Connected-components label a mask into Label regions + a label raster. 2D
    labels each plane (4/8-conn); 3D labels the volume (6/26-conn). Region ids are
    global-unique; per-region attributes are stored as Label-domain layers."""
    ds = ctx.inputs[0]
    ax = ds.axes
    is_3d = ctx.granularity is Granularity.WHOLE_VOLUME
    conn = int(ctx.params.get("connectivity", 26 if is_3d else 8))
    # the one raster on the wire, whatever it is called (`_resolve_layer`)
    mask_layer, _note = _resolve_layer(
        _voxel_layers(ds), ctx.layer("mask"), node="connected components", socket="mask",
        what="Voxel layer", where="the `data` input",
        remedy="this divides a mask's foreground into regions, so run Threshold "
               "(analysis.threshold / analysis.histogram_threshold) upstream", ctx=ctx)
    mask_attr = ds.get(Domain.VOXEL, mask_layer)
    if mask_attr is None:                            # pragma: no cover - _resolve_layer
        raise ValueError(f"no mask attribute {mask_layer!r}")
    mask6 = mask_attr.values
    # Same dense-output ceiling as `analysis.segment`, and reached through the same door:
    # this is Threshold's immediate consumer, so a mask big enough to have been spilled
    # (nodegraph.spill) hands `zeros_like(..., int64)` a request 8× the size of the mask —
    # 315 GiB against the 39.4 GiB uint8 mask on the lab's 42.3-Gvoxel 640 series. Fixing
    # Threshold without this one just moves the _ArrayMemoryError one node to the right.
    # Globally-unique ids are folded in order, so the raster cannot be lazy; above the
    # budget it is a memmapped .npy the layer keeps uncopied (2026-08-04).
    raster_out = dense_output(tuple(mask6.shape), np.int64, tag=f"labels_{ctx.node_id}")
    raster = raster_out.array
    cols: Dict[str, list] = defaultdict(list)
    offset = 0

    def take(lab, tbl):
        nonlocal offset
        for k, v in tbl.columns.items():
            cols[k].extend(((v + offset) if k == "id" else v).tolist())
        return np.where(lab > 0, lab + offset, 0), offset + tbl.n

    # Parallel connected-component pass → SERIAL ORDERED fold (V2.14). `label_components`
    # is pure per unit; `take` is not — it advances the global id `offset` and appends the
    # Label rows, so its order defines the ids. `fold_units` keeps that order exactly, so
    # the raster and the table are byte-identical to the serial loop.
    units = ([(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    tick = _parallel_progress(ctx, len(units), "labelling", frames=ax.t)

    def _label_one(unit):
        m, t, z, c = unit
        if z is None:
            return label_components(mask6[m, t, :, c], conn, m=m, t=t, c=c)
        return label_components(mask6[m, t, z, c], conn, m=m, t=t, c=c, z_index=z)

    def _fold(_i, unit, res):
        nonlocal offset
        m, t, z, c = unit
        lab, tbl = res
        if z is None:
            raster[m, t, :, c], offset = take(lab, tbl)
        else:
            raster[m, t, z, c], offset = take(lab, tbl)
        tick()

    fold_units(_label_one, units, _fold)
    layer = ctx.layer("name")
    out = ds.with_layer(Domain.VOXEL, layer, raster_out.seal())
    if cols.get("id"):
        merged = StructureTable(
            Domain.LABEL, {k: np.array(v) for k, v in cols.items()},
            layer=layer, z_kind=("subpixel" if is_3d else "plane_index"))
        out = out.with_structure(merged)
    return out
register_node(
    _compute_label, op_key="analysis.label", label="Connected Components",
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),   # emits a label RASTER too
    category="analysis",
    inputs=[InDataset(),
            InString("mask", "Mask layer", field=False, default="mask",
                     layer_in=Domain.VOXEL,
                     description=
                     "Which existing Voxel mask layer to label — the output of a Threshold "
                     "or Local Threshold node. Any non-zero voxel counts as foreground, so a "
                     "class-index raster from Multi-Otsu would merge every class into one "
                     "blob; threshold it first. Naming a layer that is not present is an "
                     "error, and the picker offers the layers actually arriving on the wire."),
            InInt("connectivity", "Connectivity", field=False,    # default per dim
                  description=
                  "Which neighbours count as touching, so it decides whether two "
                  "diagonally-adjacent foreground voxels are ONE object or two. In 2D: 4 = "
                  "edge-sharing only (splits diagonal chains), 8 = edges + corners. In 3D: 6 "
                  "= faces only, 18 = faces + edges, 26 = faces + edges + corners. Higher "
                  "merges more aggressively, which reduces object COUNT and increases "
                  "individual areas — the usual cause of two nearly-touching cells being "
                  "reported as one. 0 or unset takes the default: 8 in 2D, 26 in 3D."),
            # one name, TWO domains: a Voxel raster AND the Label table
            InString("name", "Output layer", field=False, default="labels",
                     layer_out=(Domain.VOXEL, Domain.LABEL),
                     description=
                     "Name of the layer this node writes — one name into two domains: a "
                     "Voxel raster where each region carries its own integer id, and a Label "
                     "table with one row per region. Ids are globally unique across every "
                     "(m,t,z,c), so a downstream measurement can join on id without "
                     "collisions. Downstream nodes select the result by this name.")],
    outputs=[OutDataset()], modes=[DimMode()],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    description="Label a mask into connected regions (2D 4/8-conn vs 3D 6/18/26-conn); "
                "connectivity defaults to 8 (2D) / 26 (3D), overridable.",
)
