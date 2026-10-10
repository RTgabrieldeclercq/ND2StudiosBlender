"""Stress (``field.stress``) — Cauchy stress from a field's gradient through a stated
constitutive law and its material parameters, with principal stresses, pressure, von Mises,
Tresca max shear and strain-energy density, written back onto the field's own table. Also
registers ``bundle:mechanics`` (Field Gradient → Strain → Stress, wired)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from nodegraph.bundles import Bundle, BundleEdge, BundleNode, register_bundle
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.field_nodes import (
    declared_layer, layer_announce, read_gradient, source_socket, write,
)

NODE = "field stress"
_S3 = ("z", "y", "x")
_ELASTIC = frozenset({"linear_elastic", "saint_venant_kirchhoff", "neo_hookean"})
SUMMARY = ("stress_p1", "stress_p2", "stress_p3", "pressure", "von_mises", "max_shear",
           "strain_energy")


def stress_component_names(is_3d: bool):
    """The full (z, y, x) tensor in 3-D; in 2-D the shear terms onto z are identically zero
    under both plane assumptions and are not written (``analysis.track_field``'s rule)."""
    return [f"stress_{a}{b}" for a in _S3 for b in _S3
            if is_3d or a == b or (a != "z" and b != "z")]


def _compute_field_stress(ctx: EvalContext) -> Dataset:
    """Stress from a field layer's gradient and a material model.

    Resolved spec (§0, 2026-10-09):

    * **Kind** analysis on a Point field → ``op_key="field.stress"``, category ``field``.
      Reads POINT (``grad_*``), adds POINT columns on the same table; WHOLE_SERIES, empty
      kernel_axes; dimensionality inherited from the gradient.
    * **Laws** (:func:`nodegraph.kernels.field_mechanics.stress`): ``linear_elastic`` (Hooke
      on sym(grad), ``track_field``'s), ``saint_venant_kirchhoff`` (Green-Lagrange, pushed
      forward), ``neo_hookean`` (compressible), all from E and ν; ``newtonian_fluid`` — the
      deviatoric viscous stress 2η dev(D) of a VELOCITY field. Elastic laws on a velocity, or
      the fluid on a displacement, are refused.
    * **2-D** ``plane``: ``strain`` (``F_zz = 1``, a thick specimen — a gel slab) or
      ``stress`` (``σ_zz = 0``, a thin free sheet); inert for a 3-D field.
    * **Columns** ``stress_<i><j>`` in (z, y, x) (2-D: zz, yy, yx, xy, xx), ``stress_p1..p3``
      (largest first), ``pressure`` (−tr/3, compression positive; NaN for the fluid, whose
      pressure the velocity does not determine), ``von_mises``, ``max_shear`` (Tresca),
      ``strain_energy`` (elastic laws; J/m³ = Pa). Units: the modulus's (Pa).
    * Names match ``analysis.track_field``'s, so its tables and plots read both alike.
    """
    from nodegraph.kernels.field_mechanics import stress, stress_summary

    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {})
    model = modes.get("model", "linear_elastic")
    plane = modes.get("plane", "strain")
    rows, G, q = read_gradient(ctx, ds, NODE)
    if model in _ELASTIC and q != "displacement":
        raise ValueError(
            f"{NODE}: {model.replace('_', ' ')} is a SOLID's law — stress from how far the "
            f"material has deformed — and {rows.layer!r} carries a velocity gradient. Use "
            f"`newtonian fluid` for a flow, or differentiate the displacement (Field Gradient → "
            f"Field: displacement).")
    if model == "newtonian_fluid" and q != "velocity":
        raise ValueError(
            f"{NODE}: a Newtonian fluid's stress follows from the RATE of deformation, and "
            f"{rows.layer!r} carries a displacement gradient. Differentiate the velocity "
            f"(Field Gradient → Field: velocity; PIV writes vy/vx with its velocity option on), "
            f"or pick an elastic law for a solid.")
    E = float(ctx.params.get("youngs_modulus", 1000.0))
    nu = float(ctx.params.get("poisson_ratio", 0.45))
    eta = float(ctx.params.get("viscosity", 1e-3))
    sig, energy = stress(G, model, youngs_modulus=E, poisson_ratio=nu, plane=plane,
                         viscosity=eta)
    is_3d = rows.d == 3
    cols: Dict[str, np.ndarray] = {}
    for name in stress_component_names(is_3d):
        a, b = _S3.index(name[-2]), _S3.index(name[-1])
        cols[name] = sig[:, a, b]
    summ = stress_summary(sig)
    for k in ("stress_p1", "stress_p2", "stress_p3", "pressure", "von_mises", "max_shear"):
        cols[k] = summ[k]
    if model == "newtonian_fluid":
        cols["pressure"] = np.full(rows.n, np.nan)
    else:
        cols["strain_energy"] = energy
    units = {k: "Pa" for k in cols}
    info = {"model": model, "quantity": q}
    if model in _ELASTIC:
        info.update(youngs_modulus_pa=E, poisson_ratio=nu)
        if not is_3d:
            info["plane"] = plane
    else:
        info["viscosity_pa_s"] = eta
    ok = int(np.isfinite(cols["von_mises"]).sum())
    vm = cols["von_mises"]
    ctx.progress(1, 1, f"{rows.layer}: {model.replace('_', ' ')} on {ok}/{rows.n} nodes"
                       + (f", median von Mises {float(np.nanmedian(vm)):.4g} Pa" if ok else "")
                       + (" — stress INCREMENTS of a frame-to-frame field"
                          if rows.incremental and model in _ELASTIC else ""))
    return write(ds, rows.layer, cols, units, "stress", info)


def _columns_field_stress(params, modes, incoming):
    names = stress_component_names(True) + list(SUMMARY)
    return on_layer(Domain.POINT, declared_layer(params, incoming), names)


register_node(
    _compute_field_stress, op_key="field.stress", label="Stress", category="field",
    adds_columns=_columns_field_stress, extra_layers=layer_announce,
    reads_domains=frozenset({Domain.POINT}), adds_domains=frozenset({Domain.POINT}),
    inputs=[
        InDataset(),
        source_socket(),
        InFloat("youngs_modulus", "Young's modulus", unit="Pa", field=False, default=1000.0,
                available_in={"model": _ELASTIC},
                description=
                "The material's stiffness E, in Pa. It scales every stress column LINEARLY "
                "(and strain energy) and sets their unit — enter it in Pa and the columns are "
                "Pa. The 1000 here is a placeholder, not a recommendation: a soft "
                "polyacrylamide or collagen gel is roughly 0.1-50 kPa; measure yours."),
        InFloat("poisson_ratio", "Poisson ratio", unit="", field=False, default=0.45,
                available_in={"model": _ELASTIC},
                description=
                "ν, from −1 to just under 0.5 (0.5 exactly — incompressible — is refused: the "
                "pressure is then not set by the strain at all; use 0.49 for a "
                "nearly-incompressible gel). Moves `pressure` strongly, `von_mises` far less, "
                "and in 2-D decides how much the out-of-plane direction stiffens the plane."),
        InFloat("viscosity", "Viscosity", unit="Pa*s", field=False, default=1e-3,
                available_in={"model": frozenset({"newtonian_fluid"})},
                description=
                "The fluid's dynamic viscosity η in Pa·s (water ≈ 0.001, glycerol ≈ 1.4). "
                "Every stress column is 2η times the strain-rate deviator, so it scales them "
                "linearly. Only read by the Newtonian-fluid law."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("model", ["linear_elastic", "saint_venant_kirchhoff", "neo_hookean",
                       "newtonian_fluid"],
             default="linear_elastic", label="Material law",
             description="The constitutive law turning the gradient into stress. Every elastic "
                         "law agrees with Hooke below ~1 % strain; they part ways beyond it.",
             choice_docs={
                 "linear_elastic": "Hooke's law on the small strain sym(grad) — stress "
                                   "proportional to strain. Right for small strains with "
                                   "little rotation; a rotation reads as spurious stress.",
                 "saint_venant_kirchhoff": "Hooke's law on the Green-Lagrange strain, pushed "
                                           "to the deformed shape — rotation-proof, but it "
                                           "softens unphysically under large compression.",
                 "neo_hookean": "The compressible neo-Hookean solid — the standard "
                                "large-strain law for rubbery gels and soft tissue; stiffens "
                                "under compression and stays well behaved at large stretch.",
                 "newtonian_fluid": "A viscous fluid: deviatoric stress 2η times the strain "
                                    "rate. Needs a VELOCITY field (PIV with velocity on); the "
                                    "pressure is not determined and is left blank."}),
        Mode("plane", ["strain", "stress"], default="strain", label="2D assumption",
             available_in={"model": _ELASTIC},
             description="What the unmeasured out-of-plane direction does in a 2-D field — it "
                         "changes every stress value. Inert for a 3-D field.",
             choice_docs={
                 "strain": "Plane strain: no out-of-plane stretch (F_zz = 1). A slice through "
                           "a THICK specimen such as a gel slab, where the surrounding "
                           "material holds the plane; σ_zz comes out non-zero.",
                 "stress": "Plane stress: no out-of-plane traction (σ_zz = 0). A THIN free "
                           "sheet or membrane that may thin or thicken freely."}),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Cauchy stress from a field's gradient through a stated material law "
                "(Hooke, Saint Venant-Kirchhoff, neo-Hookean, or a Newtonian fluid) and its "
                "parameters — components, principal stresses, pressure, von Mises, max shear, "
                "strain energy.")


_DX = 270.0

register_bundle(Bundle(
    key="bundle:mechanics", label="Field mechanics · gradient → strain → stress",
    category="field", role="field_mechanics",
    description=
    "The Field cards as a chain, wired in order and framed as one region: Field Gradient → "
    "Strain → Stress. Drop it after a DIC, PIV, DVC, Track Field or Load Analysis card; every "
    "card writes its columns onto the field's own table, so the Viewer's Field overlay can "
    "colour the image by any of them. Rotation & Curl and Volume Change read the same "
    "gradient — add them anywhere after Field Gradient.",
    nodes=(
        BundleNode("field_gradient", "field.gradient", 0.0, 0.0),
        BundleNode("field_strain", "field.strain", _DX, 0.0),
        BundleNode("field_stress", "field.stress", 2 * _DX, 0.0),
    ),
    edges=(
        BundleEdge("field_gradient", "out", "field_strain", "data"),
        BundleEdge("field_strain", "out", "field_stress", "data"),
    ),
    input=("field_gradient", "data"), output=("field_stress", "out"),
    frame_title="Field mechanics",
    extra_ops=("field.rotation", "field.volume_change"),
))
