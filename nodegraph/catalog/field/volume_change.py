"""Volume Change (``field.volume_change``) — the Jacobian ``J = det F`` (area ratio in 2-D,
volume ratio in 3-D), the dilatation ``J - 1`` and the divergence of a field, written back onto
the field's own table from the gradient Field Gradient wrote."""

from __future__ import annotations

from typing import Dict

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.field_nodes import (
    declared_layer, layer_announce, read_gradient, source_socket, write,
)

NODE = "field volume change"
COLUMNS = ("jacobian", "dilatation", "divergence")


def _compute_field_volume_change(ctx: EvalContext) -> Dataset:
    """Jacobian, dilatation and divergence from a field layer's gradient.

    Resolved spec (§0, 2026-10-09):

    * **Kind** analysis on a Point field → ``op_key="field.volume_change"``, category
      ``field``. Reads POINT (``grad_*``), adds POINT columns on the same table;
      WHOLE_SERIES, empty kernel_axes; dimensionality inherited from the gradient.
    * **Columns** ``jacobian = det(I + grad)`` — the local AREA ratio of a 2-D field (the
      out-of-plane stretch is not measured) or the VOLUME ratio of a 3-D one; 1 is no
      change, below 1 compaction. ``dilatation = J - 1``. ``divergence = tr(grad)``, the
      small-strain volumetric strain — equal to the dilatation to first order.
    * **Velocity** only ``divergence`` (1/s), the rate of volume change; a rate has no J.
    * **Backend** :func:`nodegraph.kernels.field_mechanics.volume_change`.
    """
    from nodegraph.kernels.field_mechanics import volume_change

    ds: Dataset = ctx.inputs[0]
    rows, G, q = read_gradient(ctx, ds, NODE)
    vc = volume_change(G)
    if q == "displacement":
        cols: Dict[str, np.ndarray] = dict(vc)
        units = {k: "" for k in cols}
    else:
        cols = {"divergence": vc["divergence"]}
        units = {"divergence": "1/s"}
    j = cols.get("jacobian")
    note = (f"{rows.layer}: median J {float(np.nanmedian(j)):.4g}"
            if j is not None and np.isfinite(j).any()
            else f"{rows.layer}: divergence on {int(np.isfinite(cols['divergence']).sum())} nodes")
    ctx.progress(1, 1, note + (" (2-D: an AREA ratio)" if rows.d == 2 and j is not None else ""))
    return write(ds, rows.layer, cols, units, "volume_change",
                 {"quantity": q, "dim": "3D" if rows.d == 3 else "2D"})


def _columns_field_volume_change(params, modes, incoming):
    return on_layer(Domain.POINT, declared_layer(params, incoming), COLUMNS)


register_node(
    _compute_field_volume_change, op_key="field.volume_change", label="Volume Change",
    category="field",
    adds_columns=_columns_field_volume_change, extra_layers=layer_announce,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(), source_socket()],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="The Jacobian det F (area ratio in 2-D, volume ratio in 3-D), the dilatation "
                "J − 1 and the divergence of a field — from Field Gradient's grad_* columns.")
