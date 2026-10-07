"""Select Plane (``util.select_plane``) — keep ONE z plane of a stack, by its 0-based index
(auto = the middle plane; pick it off the viewer); the tap Split Z's per-plane outputs
materialize into, and usable on its own."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import (plane_pick, respaced, select_plane as _meta_select_plane,
                                shift_origin_um, z_home_after)
from nodegraph.provider import FrameSubsetProvider
from nodegraph.registry import Granularity, InDataset, InInt, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.frame_subset import (subset_lattice_layers,
                                                    subset_structure_rows)
from nodegraph.catalog._shared.sampling import Z_STAMP, _sampled

# ── Select Plane (2026-10-07) ───────────────────────────────────────────────────
#
# The z-axis member of the tap family: ``channel.split`` → ``channel.select``,
# ``util.split_positions`` → ``util.select_position``, and now ``util.split_z`` → this node.
# The engine is one-payload-per-node, so K distinct per-plane payloads come from K real tap
# nodes; the split card is a pass-through.
#
# **Why it exists** (the registration workflow, 2026-10-07): the drift of a z-stack is best
# ESTIMATED on one plane the user chooses — the one with the clearest, most stable structure
# — and then APPLIED to every plane. Rather than baking a plane picker into Registration, the
# workflow is three nodes: Select Plane (this) → Registration (its 2D lever estimates on the
# single plane it is handed) → Shift (``align.shift``, which reads Registration's per-frame
# drift layers and moves the WHOLE stack by them). Each node stays small, and each half can
# be swapped: estimate on a z-projection instead, or apply a shift that came from a file.


def _compute_select_plane(ctx: EvalContext) -> Dataset:
    """Keep exactly one z plane of a stack.

    Resolved spec (build-node-v2 §0, 2026-10-07)
    --------------------------------------------
    * **Kind** utility, axis-changing → ``op_key="util.select_plane"``, category
      ``"utility"``, ``meta_transform=select_plane``.
    * **Data contract** ``Dataset → the same Dataset with z = 1``. The z half of
      ``util.crop``'s frames mode exactly: nothing lateral changes, ``z_step_um`` survives
      (one plane keeps the source spacing), ``z_home_index`` follows, and ``origin_um`` moves
      up by the planes cut off the bottom so the plane still sits at its physical z. Lattice
      layers (a mask, a per-plane statistic) are subset with the image and structure rows
      (Point / Label / Track) on other planes are dropped and the survivors re-addressed to
      ``z = 0`` — the same shared helpers ``util.crop`` and ``util.select_position`` use.
    * **2D/3D** no lever: index arithmetic only. The OUTPUT is single-plane, which is what
      makes a dim-lever node downstream derive to 2D on its own.
    * **Footprint** ``TILEABLE``, no kernel axes — one output unit reads exactly the
      corresponding input unit of the kept plane; the rest of the stack is never read.
    * **Sockets** ``plane`` (0-based; AUTO = the middle plane, ``derive="(n_z or 1) // 2"``;
      ``pick_kind="plane"`` adopts the plane the viewer is showing). No modes.
    * **Backend** none — :class:`~nodegraph.provider.FrameSubsetProvider` with one plane.

    Resolution is shared with the edit-time transform (:func:`~nodegraph.metadata.plane_pick`)
    so the card and the pull agree, and the blank/auto case resolves to the SAME middle plane
    the socket's ``derive`` shows. A value past the end is REFUSED with the stack's depth —
    handing back some other plane's pixels under the index asked for is the failure a
    selector exists to prevent. A single-plane input is returned as is, whatever is asked:
    selecting the only plane there is, is the identity.
    """
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Select Plane needs an image on its input Dataset.")
    ax = prov.axes
    raw = ctx.params.get("plane")
    if int(ax.z) <= 1:
        return ds                    # the only plane there is: the identity, whatever is asked
    k = plane_pick(int(ax.z), raw)
    if k is None:
        raise ValueError(_no_such_plane(raw, int(ax.z)))
    new_axes = replace(ax, z=1)
    view = FrameSubsetProvider(prov, tuple(range(ax.m)), tuple(range(ax.t)), (k,))
    picks = {"m": None, "t": None, "z": (k,)}
    out = subset_lattice_layers(ds.with_image(view), new_axes, picks)
    out = subset_structure_rows(out, picks)
    # ── the metadata indexed BY z, computed the way the meta_transform computes it ─────
    md = ds.metadata
    changes: Dict[str, Any] = {"z_step_um": respaced(md.get("z_step_um"), (k,))}
    changes.update(z_home_after(md, (k,)))
    out = out.with_metadata(**changes)
    z_step = md.get("z_step_um")
    if z_step and k:
        # Cutting planes off the BOTTOM moves the field's corner up by the real planes
        # dropped, at the SOURCE step (the same rule as util.crop's frames mode).
        try:
            dz = float(z_step) * int(k)
        except (TypeError, ValueError):
            dz = 0.0
        if dz:
            out = out.with_metadata(**shift_origin_um(out, dz, 0.0, 0.0))
    # ...and the envelope wins for the calibration keys, so it stays the single source of
    # truth and the reads stay memo-fenced; the local computation above is the fallback for
    # an unseeded source and cannot disagree, being the same arithmetic on the same values.
    for key in ("z_step_um", "origin_um"):
        val = ctx.calib(key)
        if val is not None:
            out = out.with_metadata(**{key: val})
    # A z-only stamp: the selection reindexes z and leaves every (y, x) address pointing at
    # the same lateral location, so a consumer comparing this branch with the full stack
    # laterally may drop it (nodegraph.catalog._shared.sampling).
    return _sampled(out, f"{Z_STAMP}select_plane[{k}]")


def _no_such_plane(raw: Any, z: int) -> str:
    valid = "0" if z <= 1 else f"0..{z - 1}"
    return (f"Select Plane: this Dataset has no plane {raw!r} — it has {z} plane(s), so the "
            f"only valid indices are {valid} (0 is the bottom plane; blank or 'auto' is the "
            f"middle one). Negative values are not 'from the top' here, they are simply out "
            f"of range. (If you rewired the Split Z node onto a shallower stack, the slot you "
            f"picked may be past its end.)")


register_node(
    _compute_select_plane,
    op_key="util.select_plane", label="Select Plane", category="utility",
    inputs=[
        InDataset(),
        InInt("plane", "Plane", unit="plane", field=False, default=0,
              derive="(n_z or 1) // 2", pick_kind="plane",
              description=
              "WHICH Z PLANE to keep, 0-based from the bottom of the stack. AUTO is the "
              "middle plane. Scrub Plane z in the Viewer to the plane with the clearest, most "
              "stable structure and press Pick to take the plane on screen. The output has "
              "one plane, so whatever follows works on exactly this plane — Registration "
              "estimates its drift here, a dim-lever node derives to 2D — and the rest of "
              "the stack is never read. A value past the end is refused with the stack's "
              "depth rather than falling back to another plane; on a single-plane input "
              "every value is the identity."),
    ],
    outputs=[OutDataset("out")],
    granularity=Granularity.TILEABLE, kernel_axes=frozenset(),
    meta_transform=_meta_select_plane,
    description="Keep ONE z plane of a stack, chosen on the Viewer (auto = the middle "
                "plane): Z narrows to 1 and everything indexed by z follows — the plane's "
                "physical z, per-plane masks, the detections on it. The tap Split Z's "
                "per-plane outputs materialize into, and the front half of 'estimate the "
                "drift on one plane, apply it to the whole stack' (Select Plane → "
                "Registration → Shift).",
)
