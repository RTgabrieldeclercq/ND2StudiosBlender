"""Rotation & Curl (``field.rotation``) — the curl of a field (vorticity, for a velocity) and
the finite rotation of its polar decomposition, written back onto the field's own table from
the gradient Field Gradient wrote."""

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

NODE = "field rotation"
COLUMNS_2D = ("curl", "rotation_deg")
COLUMNS_3D = ("curl_z", "curl_y", "curl_x", "curl_mag", "curl_view", "rotation_deg")


def _compute_field_rotation(ctx: EvalContext) -> Dataset:
    """Curl and rotation from a field layer's gradient.

    Resolved spec (§0, 2026-10-09):

    * **Kind** analysis on a Point field → ``op_key="field.rotation"``, category ``field``.
      Reads POINT (``grad_*``), adds POINT columns on the same table; WHOLE_SERIES, empty
      kernel_axes; dimensionality inherited from the gradient.
    * **2-D** ``curl = du_x/dy - du_y/dx`` in image axes — **positive counter-clockwise as
      seen on screen** (y runs down the rows), twice the small rotation angle in radians;
      ``rotation_deg`` the exact finite rotation of ``F = R U``, same sign.
    * **3-D** ``curl_z, curl_y, curl_x`` in the frame (x right, y down, z = plane index),
      right-handed when z runs away from the viewer; ``curl_mag``; ``curl_view = -curl_z``,
      the on-screen in-plane part with the 2-D sign; ``rotation_deg`` the unsigned angle of
      R.
    * **Velocity** the curl is the vorticity (1/s); a rate has no finite rotation, so
      ``rotation_deg`` is not written.
    * **Backend** :func:`nodegraph.kernels.field_mechanics.curl` / ``rotation_deg``.
    """
    from nodegraph.kernels.field_mechanics import curl, rotation_deg

    ds: Dataset = ctx.inputs[0]
    rows, G, q = read_gradient(ctx, ds, NODE)
    cols: Dict[str, np.ndarray] = dict(curl(G))
    unit = "" if q == "displacement" else "1/s"
    units = {name: unit for name in cols}
    if q == "displacement":
        cols["rotation_deg"] = rotation_deg(G)
        units["rotation_deg"] = "deg"
    ok = int(np.isfinite(next(iter(cols.values()))).sum())
    ctx.progress(1, 1, f"{rows.layer}: {'curl and rotation' if q == 'displacement' else 'vorticity'}"
                       f" on {ok}/{rows.n} nodes")
    return write(ds, rows.layer, cols, units, "rotation", {"quantity": q})


def _columns_field_rotation(params, modes, incoming):
    names = list(dict.fromkeys(COLUMNS_3D + COLUMNS_2D))
    return on_layer(Domain.POINT, declared_layer(params, incoming), names)


register_node(
    _compute_field_rotation, op_key="field.rotation", label="Rotation & Curl",
    category="field",
    adds_columns=_columns_field_rotation, extra_layers=layer_announce,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(), source_socket()],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="The curl of a field (vorticity for a velocity) and the finite rotation of "
                "its polar decomposition, counter-clockwise on screen positive — from Field "
                "Gradient's grad_* columns.")
