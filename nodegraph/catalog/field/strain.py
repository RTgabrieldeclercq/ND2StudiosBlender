"""Strain (``field.strain``) — the strain tensor of a field under a chosen measure, with its
principal values, principal angle, von Mises equivalent, max shear and trace, written back
onto the field's own table from the gradient Field Gradient wrote."""

from __future__ import annotations

from typing import Dict

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.field_nodes import (
    AXES, declared_layer, layer_announce, read_gradient, source_socket, tensor_columns, write,
)

NODE = "field strain"


def strain_columns(d: int):
    axes = AXES[d]
    names = [f"strain_{i}{j}" for i in axes for j in axes]
    names += [f"strain_p{k + 1}" for k in range(d)]
    names += (["strain_angle_deg"] if d == 2 else [])
    return names + ["strain_eq", "strain_max_shear", "strain_vol"]


def _compute_field_strain(ctx: EvalContext) -> Dataset:
    """Strain from a field layer's gradient.

    Resolved spec (§0, 2026-10-09):

    * **Kind** analysis on a Point field → ``op_key="field.strain"``, category ``field``.
      Reads POINT (the ``grad_*`` columns), adds POINT columns on the same table.
    * **Data contract** ``strain_<i><j>`` (symmetric, TENSOR shear — half the engineering
      shear; image axes slowest first), ``strain_p1..pD`` (principal values, largest first),
      ``strain_angle_deg`` (2-D: the largest principal axis, counter-clockwise on screen from
      +x), ``strain_eq`` (von Mises equivalent; 2-D takes e_zz from incompressibility, the
      commercial-DIC convention), ``strain_max_shear`` ((e_max - e_min)/2) and ``strain_vol``
      (the trace). Same names as ``analysis.track_field`` / ``dvc.strain``.
    * **Velocity** a velocity gradient gives the strain RATE ``sym(L)`` under the same names
      in 1/s (the units table says which); a finite measure is refused there.
    * **2D/3D** inherited from the gradient (4 or 9 components); **footprint** WHOLE_SERIES,
      empty kernel_axes; **backend** :func:`nodegraph.kernels.field_mechanics.strain`
      (``field_math.strain_from_gradient`` — the measure code ``dvc.strain`` and Track Field
      use).
    """
    from nodegraph.kernels.field_mechanics import (equivalent_strain, max_shear, principal,
                                                   strain, symmetric_part)

    ds: Dataset = ctx.inputs[0]
    measure = ctx.params.get("__modes__", {}).get("measure", "infinitesimal")
    rows, G, q = read_gradient(ctx, ds, NODE)
    if q == "velocity":
        if measure != "infinitesimal":
            raise ValueError(
                f"{NODE}: {rows.layer!r} holds a VELOCITY gradient (1/s). A rate has no finite "
                f"strain measure — {measure} needs a deformation gradient F = I + grad, which "
                f"only a displacement has. Use `infinitesimal`, which gives the strain rate "
                f"sym(L); or differentiate the displacement instead (Field Gradient → Field: "
                f"displacement).")
        T = symmetric_part(G)
    else:
        T = strain(G, measure)
    vals, ang = principal(T)
    d = rows.d
    cols: Dict[str, np.ndarray] = tensor_columns("strain", T, rows.axes)
    for k in range(d):
        cols[f"strain_p{k + 1}"] = vals[:, k]
    if d == 2 and ang is not None:
        cols["strain_angle_deg"] = ang
    cols["strain_eq"] = equivalent_strain(T)
    cols["strain_max_shear"] = max_shear(T)
    cols["strain_vol"] = np.trace(T, axis1=1, axis2=2)
    unit = "" if q == "displacement" else "1/s"
    units = {name: unit for name in cols}
    if "strain_angle_deg" in units:
        units["strain_angle_deg"] = "deg"
    ok = int(np.isfinite(cols["strain_eq"]).sum())
    ctx.progress(1, 1, f"{rows.layer}: {measure} {'strain' if q == 'displacement' else 'strain rate'} "
                       f"on {ok}/{rows.n} nodes"
                       + (" — a frame-to-frame INCREMENT" if rows.incremental else ""))
    return write(ds, rows.layer, cols, units, "strain",
                 {"measure": measure if q == "displacement" else "rate", "quantity": q})


def _columns_field_strain(params, modes, incoming):
    names = list(dict.fromkeys(strain_columns(3) + strain_columns(2)))
    return on_layer(Domain.POINT, declared_layer(params, incoming), names)


register_node(
    _compute_field_strain, op_key="field.strain", label="Strain", category="field",
    adds_columns=_columns_field_strain, extra_layers=layer_announce,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[InDataset(), source_socket()],
    outputs=[OutDataset()],
    modes=[
        Mode("measure", ["infinitesimal", "green_lagrange", "euler_almansi", "hencky"],
             default="infinitesimal", label="Measure",
             description="Which strain tensor is computed from the gradient. They agree to "
                         "first order; they part ways above a few percent strain or under "
                         "any noticeable rotation.",
             choice_docs={
                 "infinitesimal": "sym(grad) — linear, the classic small-strain tensor. "
                                  "Wrong under rotation: a rigid 10° turn reads as 1.5 % "
                                  "strain. Fine below ~2 % strain with little rotation; the "
                                  "only choice for a velocity (it is then the strain rate).",
                 "green_lagrange": "(FᵀF − I)/2 in the REFERENCE configuration — exact under "
                                   "any rotation, the measure finite-strain solid mechanics "
                                   "and Saint Venant-Kirchhoff use; stretches read larger "
                                   "than compressions of the same size.",
                 "euler_almansi": "(I − F⁻ᵀF⁻¹)/2 in the CURRENT configuration — exact "
                                  "under rotation, referred to the deformed shape; "
                                  "compressions read larger than stretches.",
                 "hencky": "ln(U), the logarithmic (true) strain — rotation-free and "
                           "symmetric between stretch and compression, additive over "
                           "successive stretches; the large-strain measure for soft gels."}),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="The strain tensor of a field under the chosen measure — components, "
                "principal values and angle, von Mises equivalent, max shear, trace — from "
                "Field Gradient's grad_* columns; the strain RATE for a velocity field.")
