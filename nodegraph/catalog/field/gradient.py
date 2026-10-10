"""Field Gradient (``field.gradient``) — the displacement (or velocity) gradient of a measured
field, ``grad_<i><j> = d u_i / d X_j`` on every node, written back onto the field's own table.
The hinge of the Field cards: Strain, Rotation & Curl, Volume Change and Stress all read it."""

from __future__ import annotations

from typing import Dict

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.field_nodes import (
    AXES, POSITIONS_MODE, QUANTITY_MODE, declared_layer, field_layers, grad_columns,
    layer_announce, quantity_of, read_rows, source_socket, tensor_columns, vector_columns,
    write,
)

NODE = "field gradient"


def _compute_field_gradient(ctx: EvalContext) -> Dataset:
    """The gradient of a field layer's displacement (or velocity), per independent field.

    Resolved spec (§0, 2026-10-09):

    * **Kind** analysis on a Point field → ``op_key="field.gradient"``, category ``field``.
      Reads POINT, adds POINT (columns on the SAME table — the field layer). No pixels.
    * **Data contract** adds ``grad_<i><j>`` (``d u_i / d X_j``, image axes slowest first:
      4 columns in 2-D, 9 in 3-D) and ``grad_valid``; dimensionless for a displacement,
      1/s for a velocity (recorded in the units table). Stamps ``__field_mechanics__
      [layer]["gradient"]`` = quantity, method(s), radius, smoothing, positions, dim, whether
      the field is a frame-to-frame increment.
    * **2D/3D** INHERITED from the layer's ``z_kind`` (§7b): ``plane_index`` → every
      (m, t, c, z) plane is its own 2-D field; ``subpixel`` → a (m, t, c) frame is a volume.
      Nothing is ever differentiated across planes, frames, channels or positions.
    * **Footprint** WHOLE_SERIES, empty ``kernel_axes``: it reads every row of the table and
      touches no voxel.
    * **Backend** :func:`nodegraph.kernels.field_mechanics.gradient` — NaN-aware finite
      differences on a regular lattice (correlation grids, holes included), or the
      moving-least-squares plane fit of ``track_field`` for any other layout.
    * **Calibration** ``pixel_size_um`` (+ ``z_step_um`` in 3-D) via ``ctx.calib`` — the
      gradient is per µm; a missing value is refused, never defaulted to 1.
    * **Numba** no: one batched numpy solve or a few array differences per field.
    """
    from nodegraph.kernels.field_mechanics import gradient

    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    rows = read_rows(ctx, ds, NODE, field_layers(ds),
                     remedy="run a DIC, PIV or DVC node (or Track Field, or Load Analysis) "
                            "upstream — they write the field this card differentiates")
    q = quantity_of(rows, modes.get("quantity", "auto"), NODE)
    vec = np.column_stack([np.asarray(rows.cols[c], dtype=float)
                           for c in vector_columns(rows.axes, q)])
    pos = modes.get("positions", "auto")
    current = pos == "current" or (pos == "auto" and "track_id" in rows.cols)
    coords = rows.coords.copy()
    if current and q == "displacement":
        # where each particle WAS: F = I + grad holds only for the material gradient
        coords = coords - np.nan_to_num(vec, nan=0.0) / rows.scale
    method = modes.get("method", "auto")
    radius = float(ctx.params.get("radius", 0.0) or 0.0)
    neighbors = int(ctx.params.get("neighbors", 0) or 0)
    smooth = float(ctx.params.get("smooth", 0.0) or 0.0)
    if radius < 0 or smooth < 0 or neighbors < 0:
        raise ValueError(f"{NODE}: radius, neighbours and smoothing must be ≥ 0 (0 = automatic "
                         f"/ off); got radius={radius}, neighbors={neighbors}, smooth={smooth}")
    d = rows.d
    G = np.full((rows.n, d, d), np.nan)
    used, radii, spacing = set(), [], []
    for k, g in enumerate(rows.groups):
        ok = np.isfinite(coords[g]).all(axis=1)
        gi = g[ok]
        if gi.size >= d + 1:
            res = gradient(coords[gi], rows.scale, vec[gi], method=method, radius_um=radius,
                           neighbors=neighbors, smooth=smooth)
            G[gi] = res.G
            used.add(res.method)
            if np.isfinite(res.radius_um):
                radii.append(res.radius_um)
            if np.isfinite(res.spacing_um):
                spacing.append(res.spacing_um)
        ctx.progress(k + 1, max(1, len(rows.groups)),
                     f"{rows.layer}: field {k + 1}/{len(rows.groups)}")
    valid = np.isfinite(G).all(axis=(1, 2))
    cols: Dict[str, np.ndarray] = tensor_columns("grad", G, rows.axes)
    cols["grad_valid"] = valid.astype(np.int64)
    unit = "" if q == "displacement" else "1/s"
    units = {name: unit for name in cols if name != "grad_valid"}
    units["grad_valid"] = ""
    info = {"quantity": q, "method": "+".join(sorted(used)) or method,
            "radius_um": float(np.median(radii)) if radii else None,
            "spacing_um": float(np.median(spacing)) if spacing else None,
            "smooth": smooth, "positions": "current" if current else "reference",
            "dim": "3D" if d == 3 else "2D", "incremental": bool(rows.incremental)}
    note = (f"{int(valid.sum())}/{rows.n} nodes with a gradient ({info['method']}"
            + (f", radius {info['radius_um']:.3g} µm" if info["radius_um"] else "") + ")"
            + (" — a frame-to-frame INCREMENT field: its gradient is the increment's"
               if rows.incremental else ""))
    ctx.progress(1, 1, note)
    return write(ds, rows.layer, cols, units, "gradient", info)


def _columns_field_gradient(params, modes, incoming):
    names = grad_columns(AXES[3]) + ["grad_valid"]     # ⊇ the 2-D four
    return on_layer(Domain.POINT, declared_layer(params, incoming, ("disp_y", "vy")), names)


register_node(
    _compute_field_gradient, op_key="field.gradient", label="Field Gradient", category="field",
    adds_columns=_columns_field_gradient, extra_layers=layer_announce,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        source_socket(),
        InFloat("radius", "Fit radius", unit="um", field=False, default=0.0,
                pick_kind="distance",
                available_in={"method": frozenset({"auto", "local_fit"})},
                description=
                "Radius of the neighbourhood a plane is fitted over (local fit), in µm — the "
                "gauge length of every quantity derived from the gradient. 0 = automatic: 1.5x "
                "the distance to each node's first full shell of neighbours (about 2 node "
                "spacings; on a stack it reaches the planes above and below). LARGER smooths "
                "and steadies strain but blurs concentrations and loses edge nodes; SMALLER "
                "follows sharp features and is noisier. Inert when the lattice path runs."),
        InInt("neighbors", "Neighbours", unit="", field=False, default=0,
              available_in={"method": frozenset({"auto", "local_fit"})},
              description=
              "How many nearest nodes are considered before the radius cut (local fit). 0 = "
              "automatic from the radius and the field's density. Raise it only if a large "
              "radius on a dense field leaves nodes invalid; it costs memory linearly."),
        InFloat("smooth", "Pre-smoothing", unit="", field=False, default=0.0,
                available_in={"method": frozenset({"auto", "lattice"})},
                description=
                "Gaussian smoothing of the field before differencing, in NODE spacings "
                "(lattice path); 0 = off. Holes neither contribute nor get filled, so a hole "
                "does not drag its neighbours toward zero. 1 suppresses node-to-node noise "
                "that differencing amplifies; it also lowers peak strain values."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("method", ["auto", "lattice", "local_fit"], default="auto", label="Method",
             description="How the field is differentiated.",
             choice_docs={
                 "auto": "Lattice differences when the nodes form a regular grid (DIC, PIV and "
                         "DVC grids, holes allowed), otherwise the local plane fit (a refined "
                         "DIC mesh, tracked particles).",
                 "lattice": "Central differences between neighbouring grid nodes, one-sided "
                            "beside holes and at edges — the sharpest gradient a grid allows. "
                            "Refused when the nodes are not a regular grid.",
                 "local_fit": "A weighted plane fitted through the nodes within the radius — "
                              "any layout, smoother than differences (the radius is the gauge "
                              "length); nodes with too few neighbours come out NaN."}),
        QUANTITY_MODE,
        POSITIONS_MODE,
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="The displacement (or velocity) gradient of a measured field on every node — "
                "grad_* columns written back onto the field's table, by lattice differences or "
                "a local plane fit. Strain, Rotation & Curl, Volume Change and Stress read it.")
