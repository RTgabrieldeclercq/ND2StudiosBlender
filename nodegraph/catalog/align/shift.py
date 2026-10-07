"""Shift (``align.shift``) — move EVERY plane, channel and position of a Dataset by a
per-frame (z, y, x) shift read as Frame layers from a Registration / Drift Correction output
(or from the data's own layers): the APPLY half of registration as a node of its own."""

from __future__ import annotations

import hashlib

import numpy as np

from typing import Any, Dict

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, is_structure
from nodegraph.engine import EvalContext
from nodegraph.provider import ArrayProvider
from nodegraph.registry import DimMode, InBool, InDataset, InString, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.drift_layers import _layers_shift
from nodegraph.catalog._shared.labels import _resolve_layer
from nodegraph.catalog._shared.sampling import _sampled

# ── Shift (2026-10-07) ──────────────────────────────────────────────────────────
#
# Registration (``registration.stabilize``) estimates a per-frame transform AND applies it,
# to the data it estimated on. That couples the two halves: the drift of a z-stack is best
# ESTIMATED on one chosen plane (or a projection, or one channel) and APPLIED to the whole
# stack, and no single node should have to know both which plane to look at and what to
# move. So the apply half is this node. It takes the data to move on one input and the
# shift as a FUNCTION OF TIME on another — the ``drift_y`` / ``drift_x`` (/ ``drift_z``)
# Frame layers Registration and Drift Correction already write — and moves every plane,
# channel and position of the data by its frame's shift. The intended wiring:
#
#     stack ──┬── Select Plane (plane k) ── Registration (2D) ──┐  drift_y / drift_x per frame
#             │                                                  ▼
#             └──────────────────────────────────────────── Shift ──► the stack, stabilized
#
# Sign convention: a layer value is the shift APPLIED to frame t's content, in pixels
# (planes for z), positive toward a higher row / column / plane index. That is exactly what
# Registration stores (its drift_* layers are the correction, the negative of how far the
# content moved), so feeding its layers back through this node reproduces its own alignment
# bit for bit — the selftest asserts it — and ``invert`` puts the measured motion back.

#: Interpolation Mode value → the spline ``order`` scipy's resampler takes (as
#: ``transform.rigid`` spells it; a shared constant would be the only thing shared).
_INTERP_ORDER: Dict[str, int] = {"nearest": 0, "linear": 1, "cubic": 3}


def _compute_shift(ctx: EvalContext) -> Dataset:
    """Apply a per-frame ``(dz, dy, dx)`` shift to every unit of the data.

    Resolved spec (build-node-v2 §0, 2026-10-07)
    --------------------------------------------
    * **Kind** registration (apply) → ``op_key="align.shift"``, category ``"registration"``.
    * **Data contract** axis-preserving. ``data`` → the same Dataset with its image moved;
      Voxel layers (a mask, a label raster) ride along — integer rasters at order 0 so ids
      survive — and a Dataset carrying STRUCTURE TABLES is refused, exactly as
      ``transform.rigid`` refuses it: a table stores ``y``/``x``/``z`` as columns, and moving
      the raster while the rows keep their old addresses reports every object where it used
      to be. Put Shift before the segmentation step, which is where a registration belongs.
      ``origin_um`` is dropped (a per-``(m,t)`` move has no single per-M corner), as
      ``registration.stabilize`` and ``align.drift`` drop it; ``bit_depth`` survives
      (resampling redistributes counts inside their range, §7c).
    * **The shift** comes from the ``shifts`` input's Frame layers when it is wired, else
      from ``data``'s own — ``layer_from="shifts"`` makes the picker offer the right wire's
      layers. Shape ``(M', T)``: ``T`` must equal the data's (the shift is a function of
      time, so the two must be the same series), ``M'`` must equal the data's or be 1
      (applied to every position). A non-finite value (a frame Registration gated) is no
      shift. ``invert`` negates.
    * **2D/3D** ``DimMode``. 2D applies ``(dy, dx)`` per plane; 3D applies ``(dz, dy, dx)``
      per volume, with ``dz`` from the 3D-only ``shift_z`` socket — EMPTY (the default)
      means no axial shift, so a stack whose estimate came from one plane under
      Registration's 2D lever applies cleanly under either lever.
    * **Footprint** ``_DIM_GRAN_GLOBAL`` / ``_DIM_KAX``: a shift fills the vacated edge with
      zeros, so a tile would invent that border mid-image — whole planes, whole volumes.
    * **Lazy apply, as Registration's.** Frame ``t``'s output depends only on frame ``t``'s
      pixels and its shift, so the move is a per-plane / per-volume streaming provider and
      only the units pulled are computed; the shift vector itself is kilobytes, baked
      eagerly. It is folded into the provider fingerprint by content — it comes from a
      SECOND input, which the base provider's fingerprint knows nothing about, so without
      this two different shift sources over the same data would share tile entries.
    * **Backend** :func:`nodegraph.kernels.registration.apply_shift` (``scipy.ndimage.shift``
      on float32, cast back) — the SAME call ``apply_frame`` makes for the translation
      model, which is what makes Shift(data, Registration(data)) == Registration(data).
    * **Stored** the applied shift, per frame, as Frame layers ``shift_y`` / ``shift_x``
      (``shift_z`` when an axial shift was applied), so the window and a table can read
      what moved.
    """
    from scipy.ndimage import shift as nd_shift
    from nodegraph.kernels.registration import apply_shift

    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("Shift needs an image provider on its input Dataset")
    ax = prov.axes
    volumetric = ctx.is_volume
    shifts_ds = ctx.input("shifts")
    src = shifts_ds if shifts_ds is not None else ds
    where = "the Shifts input" if shifts_ds is not None else "the data itself (Shifts is not wired)"
    order = _INTERP_ORDER.get(
        str(ctx.params.get("__modes__", {}).get("interp", "linear")), 1)
    invert = bool(ctx.params.get("invert", False))
    names = {"y": ctx.layer("shift_y"), "x": ctx.layer("shift_x"),
             "z": ctx.layer("shift_z") if volumetric else ""}
    have = sorted(a.name for a in src.layers_on(Domain.FRAME))

    def read(axis: str) -> np.ndarray:
        want = str(names[axis] or "").strip()
        if axis == "z" and not want:
            return np.zeros((ax.m, ax.t), dtype=float)           # no axial shift asked for
        # The shared resolver (the 2026-08-04 sweep's rule for every REQUIRED layer socket):
        # an explicit name that is on the wire wins; the only candidate is taken whatever it
        # is called, with a rail note; nothing is ever chosen BETWEEN candidates.
        remedy = ("wire a Registration or Drift Correction output into `shifts` — its drift_y "
                  "/ drift_x" + (" / drift_z (under its 3D lever)" if axis == "z" else "")
                  + " Frame layers are the per-frame shift"
                  + (", or leave Shift z blank for no axial shift" if axis == "z" else ""))
        name, _note = _resolve_layer(
            have, want, node="Shift", socket=f"shift_{axis}", what="Frame layer",
            where=where, remedy=remedy, ctx=ctx)
        attr = src.get(Domain.FRAME, name)
        v = np.asarray(attr.values, dtype=float)
        if v.ndim != 2:
            raise ValueError(f"Shift: Frame layer {name!r} is not one value per "
                             f"(position, frame) — its shape is {v.shape}.")
        if v.shape[1] != ax.t:
            raise ValueError(
                f"Shift: {name!r} on {where} has {v.shape[1]} timepoint(s) but the data has "
                f"{ax.t}. The shift is a function of time, so the two must come from the "
                f"same series — the same timepoints, in the same order.")
        if v.shape[0] == ax.m:
            return v
        if v.shape[0] == 1:
            return np.repeat(v, ax.m, axis=0)                      # one shift for every position
        raise ValueError(
            f"Shift: {name!r} on {where} covers {v.shape[0]} position(s) but the data has "
            f"{ax.m}. A shift source must match the data's positions, or have exactly one "
            f"(then it is applied to all of them).")

    dy, dx = read("y"), read("x")
    dz = read("z") if volumetric else np.zeros((ax.m, ax.t), dtype=float)
    vec = np.stack([dz, dy, dx], axis=-1)
    vec = np.where(np.isfinite(vec), vec, 0.0) * (-1.0 if invert else 1.0)
    vec = np.ascontiguousarray(vec, dtype=np.float64)              # (M, T, 3): dz, dy, dx
    axial = bool(volumetric and np.any(vec[:, :, 0]))

    structures = sorted({(a.domain.value, a.layer or a.name)
                         for a in ds.attributes.values() if is_structure(a.domain)})
    if structures:
        raise ValueError(
            f"Shift moves pixels, and this Dataset also carries structure tables "
            f"{structures} whose y/x/z COLUMNS record where those objects used to be. "
            f"Moving the raster while leaving the table behind would report every object at "
            f"its old position, so the shift is refused rather than applied to half the "
            f"bundle. Put Shift BEFORE the segmentation / detection step — which is where a "
            f"registration belongs.")

    def move_plane(a: np.ndarray, m: int, t: int) -> np.ndarray:
        sh = vec[m, t, 1:]
        return a if not np.any(sh) else apply_shift(a, sh, order=order)

    def move_volume(v: np.ndarray, m: int, t: int) -> np.ndarray:
        sh = vec[m, t]
        return v if not np.any(sh) else apply_shift(v, sh, order=order)

    cache = ctx.tiles
    if cache is not None:
        # WHOLE_PLANE / WHOLE_VOLUME lazy unit. The shift VALUES are folded in by content:
        # they come from the second input, which the base provider's fingerprint cannot see.
        vec_fp = hashlib.sha1(vec.tobytes()).hexdigest() + f":{vec.shape}:o{order}"
        fp = stream_fp("shift", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (vec_fp,), prov)
        if volumetric:
            out = ds.with_image(VolumeComputeProvider(
                prov, lambda v, m, t, c: move_volume(v, m, t), fp=fp, cache=cache))
        else:
            out = ds.with_image(MapComputeProvider(
                prov, lambda a, m, t, z, c, *_: move_plane(a, m, t),
                unit="plane", fp=fp, cache=cache))
    else:                                          # pre-C1 eager fallback (bare ctx)
        # SAME per-unit call as the lazy path, so the two cannot drift in the last bits.
        buf = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        units = [(m, t, c) for m in range(ax.m) for t in range(ax.t) for c in range(ax.c)]
        for done, (m, t, c) in enumerate(units):
            if volumetric:
                vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                buf[m, t, :, c] = move_volume(np.asarray(vol, dtype=float), m, t)
            else:
                for z in range(ax.z):
                    plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                    buf[m, t, z, c] = move_plane(np.asarray(plane, dtype=float), m, t)
            ctx.progress(done + 1, len(units), "shifting", frames=ax.t)
        out = ds.with_image(ArrayProvider(buf))

    # Voxel layers are lattice arrays over the SAME grid, so they make the same move or a
    # mask would be left describing where its objects used to be. Integer rasters (masks,
    # label maps) resample at order 0 whatever the Mode says — an averaged id is not an id.
    for attr in ds.layers_on(Domain.VOXEL):
        raster = np.asarray(attr.values)
        if tuple(raster.shape) != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
            continue                                   # stale against these axes: untouched
        is_int = not np.issubdtype(raster.dtype, np.floating)
        lorder = 0 if is_int else order
        work = raster.astype(np.uint8) if raster.dtype == bool else raster
        moved = work.copy()
        for m in range(ax.m):
            for t in range(ax.t):
                sh = vec[m, t] if volumetric else vec[m, t, 1:]
                if not np.any(sh):
                    continue
                for c in range(ax.c):
                    if volumetric:
                        moved[m, t, :, c] = nd_shift(work[m, t, :, c], sh, order=lorder,
                                                     mode="constant", cval=0)
                    else:
                        for z in range(ax.z):
                            moved[m, t, z, c] = nd_shift(work[m, t, z, c], sh, order=lorder,
                                                         mode="constant", cval=0)
        out = out.with_layer(Domain.VOXEL, attr.name, moved.astype(raster.dtype), attr.layer)

    # Same reason as Registration: the content has been moved under an unchanged index
    # grid, so the per-M corner no longer describes it. Dropped rather than carried forward.
    out = out.with_metadata(origin_um=None)
    out = _sampled(out, f"align.shift[{'3D' if volumetric else '2D'},o{order}]")
    if axial:
        out = out.with_layer(Domain.FRAME, "shift_z", vec[:, :, 0])
    out = out.with_layer(Domain.FRAME, "shift_y", vec[:, :, 1])
    out = out.with_layer(Domain.FRAME, "shift_x", vec[:, :, 2])
    return out


_DIM_DOCS = {
    "2D": "Apply each frame's (y, x) shift to every plane of that frame, plane by plane. "
          "The cheap path — one output plane costs one source plane — and the right one "
          "when the shift was estimated on a single plane (Select Plane → Registration) and "
          "is meant for the whole stack. An axial shift the Shifts input may carry is "
          "ignored here.",
    "3D": "Apply each frame's (z, y, x) shift to the whole volume of that frame, so a stack "
          "that drifted in focus is moved back along z as well. Needs a Shift z layer "
          "(Registration's drift_z, written under its own 3D lever); leave it blank and only "
          "the lateral shift is applied — the same answer as 2D, computed per volume, so one "
          "viewed plane costs the whole stack.",
}

_LAYER_DOC_TAIL = (
    " — the shift APPLIED to the content, positive toward a higher index, which is the "
    "convention Registration and Drift Correction write (their drift layers are the "
    "correction, the negative of how far the content moved), so their layers reproduce "
    "their own alignment exactly. Offered from the Shifts input when it is wired, else from "
    "the data's own Frame layers. A frame whose value is NaN (a gated estimate) is left "
    "where it is.")

register_node(
    batch_aware(_compute_shift), op_key="align.shift", label="Shift", category="registration",
    extra_layers=_layers_shift,
    reads_domains=frozenset({Domain.FRAME}), adds_domains=frozenset({Domain.FRAME}),
    inputs=[
        InDataset(description=
                  "The data to move: every plane, channel and position of it is shifted by "
                  "its frame's (z, y, x) shift, and its masks and label rasters move with "
                  "it. When Shifts is not wired, the shift is read from this input's own "
                  "Frame layers."),
        InDataset("shifts", label="Shifts", passes_domains=False,
                  description=
                  "Where the per-frame shift comes from: a Registration or Drift Correction "
                  "output, whose drift_y / drift_x (and drift_z) Frame layers are read. "
                  "Nothing else of it is used and it passes nothing downstream. It must have "
                  "the same number of timepoints as the data, and the same number of "
                  "positions or exactly one. Unwired = read the layers off the data itself."),
        InString("shift_y", "Shift y layer", field=False, default="drift_y",
                 layer_in=Domain.FRAME, layer_from="shifts",
                 description=
                 "Which Frame layer holds each frame's shift along y (rows), in pixels"
                 + _LAYER_DOC_TAIL),
        InString("shift_x", "Shift x layer", field=False, default="drift_x",
                 layer_in=Domain.FRAME, layer_from="shifts",
                 description=
                 "Which Frame layer holds each frame's shift along x (columns), in pixels"
                 + _LAYER_DOC_TAIL),
        InString("shift_z", "Shift z layer", field=False, default="",
                 layer_in=Domain.FRAME, layer_from="shifts",
                 available_in={"dim": frozenset({"3D"})},
                 description=
                 "Which Frame layer holds each frame's shift along z, in PLANES. BLANK (the "
                 "default) applies no axial shift, which is right for a stack whose estimate "
                 "came from one plane; Registration under its 3D lever writes drift_z, and "
                 "naming it here moves the focus drift back too. Only read under the 3D "
                 "lever" + _LAYER_DOC_TAIL),
        InBool("invert", "Invert", field=False, default=False,
               description=
               "Apply the NEGATIVE of the layers' shift. The layers Registration writes are "
               "the correction, so applying them as they are reproduces its alignment; "
               "inverted, they put the measured motion back — to undo a correction, or to "
               "impose one series' drift on another. Off by default."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(choice_docs=_DIM_DOCS),
           Mode("interp", ["linear", "nearest", "cubic"], default="linear",
                label="Interpolation",
                description=
                "How pixel values are sampled when a shift lands between pixels. A sub-pixel "
                "shift must interpolate, so this is a genuine choice about what the moved "
                "pixels MEAN; a whole-pixel shift lands exactly and every option gives the "
                "same bytes. Masks and label rasters always move at nearest, whatever is "
                "chosen here, so their ids survive.",
                choice_docs={
                    "linear":
                        "Weighted average of the neighbouring pixels — the usual choice for "
                        "intensity images: smooth, no overshoot, a slight blur that grows "
                        "with each successive resample. The same order Registration uses, "
                        "so its layers reproduce its output exactly.",
                    "nearest":
                        "Copy the closest pixel's value, no averaging. Preserves the exact "
                        "histogram at the cost of jagged edges and up to half a pixel of "
                        "positional error; the right choice when the counts must stay raw.",
                    "cubic":
                        "Third-order spline: the sharpest of the three and the best at "
                        "preserving fine texture, but it can RING — overshoot at a hard "
                        "edge, so a saturated boundary can gain a dark halo — and it is the "
                        "slowest.",
                })],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    description="Move every plane, channel and position of a Dataset by a per-frame "
                "(z, y, x) shift read from a Registration or Drift Correction output's "
                "drift layers (or the data's own) — the apply half of registration as a "
                "node of its own, so the drift can be estimated on one plane (Select Plane "
                "→ Registration) and applied to the whole stack. Masks and label rasters "
                "move with the image; stores the applied shift as Frame layers.")
