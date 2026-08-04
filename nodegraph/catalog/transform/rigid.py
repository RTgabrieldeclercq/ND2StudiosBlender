"""Transform (``transform.rigid``) — Move selected channels' pixels inside a FIXED frame — translate in X/Y (and Z in 3D) and rotate in-plane about the centre — cropping whatever leaves the frame and zero-filling what it vacates."""

from __future__ import annotations

import math
import numpy as np

from typing import Dict, Sequence, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, is_structure
from nodegraph.engine import EvalContext
from nodegraph.metadata import parse_channels
from nodegraph.provider import ArrayProvider
from nodegraph.registry import DimMode, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, VolumeComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.sampling import _sampled
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Transform (rigid — move selected channels' content inside a fixed frame) ───
#
# The one node whose job is to make channels DISAGREE with the file on purpose. A
# chromatic shift, a filter-cube that does not repeat, a dichroic swapped between
# acquisitions: all of them put the same object a few pixels apart in two channels, and
# nothing downstream can tell that apart from real biology — a colocalization coefficient
# just comes back low. Moving one channel back is a statement about the OPTICS, so it
# belongs in the saved graph where it diffs and travels, not in a viewer nudge.
#
# The frame does not grow. Content pushed past an edge is discarded and the vacated strip
# is filled with zeros, which is what makes the output still addressable voxel-for-voxel
# against the channels that did not move — the whole point of the node.

#: Interpolation Mode value → the spline ``order`` scipy's resampler takes.
_INTERP_ORDER: Dict[str, int] = {"nearest": 0, "linear": 1, "cubic": 3}
def _rigid_inverse(angle_deg: float, shift: Sequence[float],
                   shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """The ``(matrix, offset)`` pair :func:`scipy.ndimage.affine_transform` wants in order
    to move the CONTENT by ``shift`` (pixels) and ``angle_deg`` about the array centre.

    ``affine_transform`` samples ``input[matrix @ o + offset]`` for every output index
    ``o``, so what it takes is the **inverse** of the motion the picture should make. In
    index space (y down, x right), with ``R`` the forward rotation and ``c`` the centre::

        p_out = R (p_in − c) + c + s      ⇒      p_in = Rᵀ (p_out − c − s) + c

    hence ``matrix = Rᵀ`` (a rotation is orthogonal, so the transpose IS the inverse) and
    ``offset = c − Rᵀ (c + s)``. Order matters and is fixed here: rotate about the centre
    first, translate second.

    ``R`` is written so a POSITIVE angle carries the +X axis toward +Y — right toward
    down — which is a clockwise turn as the image is displayed, the same sign convention
    ImageJ's Rotate uses. Getting this backwards is invisible on anything symmetric, which
    is why :func:`nodegraph.selftest.test_transform` checks it with a single off-centre
    pixel rather than a blob.

    ``len(shape)`` selects 2-D ``(y, x)`` or 3-D ``(z, y, x)``. The rotation is always in
    the Y–X plane, so the z row of the 3-D matrix stays the identity and its offset reduces
    to ``−s_z`` — a pure axial translation, with no interpolation across the anisotropic
    axis beyond the shift itself.
    """
    n = len(shape)
    th = math.radians(float(angle_deg))
    cos, sin = math.cos(th), math.sin(th)
    r = np.eye(n, dtype=float)
    r[n - 2, n - 2], r[n - 2, n - 1] = cos, sin      # y' =  cos·y + sin·x
    r[n - 1, n - 2], r[n - 1, n - 1] = -sin, cos     # x' = −sin·y + cos·x
    inv = r.T
    centre = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    s = np.asarray(shift, dtype=float)
    return inv, centre - inv @ (centre + s)
def _rigid_warp(a: np.ndarray, matrix: np.ndarray, offset: np.ndarray,
                order: int) -> np.ndarray:
    """Resample ``a`` through the inverse map from :func:`_rigid_inverse`.

    ``mode="constant", cval=0`` is the CROP: anything whose source lies outside the input
    is zero, so content that leaves the frame is gone and the strip it vacated is empty.
    Nothing else in the pipeline needs to know — the array keeps its shape."""
    from scipy.ndimage import affine_transform
    return affine_transform(np.asarray(a), matrix, offset=offset, order=int(order),
                            mode="constant", cval=0.0)
def _compute_transform(ctx: EvalContext) -> Dataset:
    """Translate and rotate SELECTED channels inside a fixed frame; crop what leaves it.

    Resolved spec (§0 grill, 2026-08-03)
    ------------------------------------
    * **Kind** geometric transform of the image content → ``op_key="transform.rigid"``,
      category ``"transform"`` (alongside ``transfer_domain`` / ``rasterize_*``).
    * **Data contract** Image → Image, **axis-preserving**: the frame is fixed, so there is
      no ``meta_transform``. Calibration is untouched as well — ``pixel_size_um`` and
      ``z_step_um`` describe how finely the grid samples, which a move within that grid does
      not change.
    * **Channel scope** the ``channels`` socket names which channels move; the rest pass
      through at their original position **on the same wire**. That is the design: one
      Dataset comes out with a corrected channel sitting beside an untouched one, so
      colocalization / ratio / overlay downstream compare the aligned pixels without any
      merge node. Splitting the bundle and re-merging could not do this — the engine is
      one-payload-per-node and the catalog has no channel merge.
    * **2D/3D** ``DimMode``. 2D transforms each ``(Y,X)`` plane independently (``shift_z``
      is gated away); 3D transforms the ``(Z,Y,X)`` volume, adding the axial translation.
      Rotation is in-plane in both — see the grill note under ``_rigid_inverse``.
    * **Footprint** ``{"2D": WHOLE_PLANE, "3D": WHOLE_VOLUME}`` (``_DIM_GRAN_GLOBAL``). A
      shift or a rotation reads from an arbitrarily distant part of the unit, so a tile
      would have to halo the whole plane; declaring the plane unit is the honest form.
    * **Backend** :func:`scipy.ndimage.affine_transform` (signature re-verified in this env,
      scipy 1.17.1). One compiled call per unit — `wire-node-v2` §12 says leave it alone,
      numba has nothing to add.

    **Why the shifts are in microns.** They describe a physical displacement of specimen
    content, so authoring them in µm makes a correction measured once transfer to another
    objective, another binning, another camera. The `util.crop` exception does not apply:
    a crop's bounds decide the output SIZE and so must be indices, while a shift moves
    content inside a size that is already fixed. An image with no ``pixel_size_um`` resolves
    at 1 µm = 1 px rather than the catalog's usual 0.1 µm guess — inventing a scale here
    would silently move the data ten times too far.

    **``origin_um`` is KEPT**, unlike ``align.drift`` / ``registration.stabilize`` which
    drop it. Their shift is per-``(m,t)``, so no single per-M corner can describe where the
    content ended up. This one is a single deliberate correction applied uniformly, and the
    channels that did NOT move still sit exactly on the declared corner — so the corner
    still describes the field, and an Overlay downstream keeps placing. What IS stamped is
    the sampling provenance (:func:`_sampled`), because the content moved under a fixed
    index grid: that is what makes ``analysis.measure`` refuse a ``raw`` branch which
    bypassed this node, which is correct, since those pixels really are somewhere else.

    **``bit_depth`` is not restamped.** Resampling redistributes intensities inside the
    range they already occupied (`wire-node-v2` §7c) — it neither sums nor rescales. Cubic
    interpolation can overshoot by a fraction of a count at a sharp edge; that is ringing,
    not a wider value scale.

    **Voxel layers ride along**; structure tables are refused. A mask or a distance field
    is a lattice array over the same grid, so it gets the same transform (integer rasters
    at ``order=0``, so ids survive). A Label / Point / Track table stores ``y``/``x``/``z``
    as COLUMNS, and this node cannot honestly rewrite them — a row that rotated out of the
    frame would have to be dropped, breaking the id↔raster correspondence — so a Dataset
    carrying one is refused rather than half-transformed. Transform belongs before the
    segmentation step anyway, which is where a channel registration goes.
    """
    ds = ctx.inputs[0]
    volumetric = ctx.is_volume

    # µm → px. `or 1.0` (not the catalog's 0.1) so an uncalibrated image reads the socket
    # as pixels 1:1 instead of moving ten times too far — see the docstring.
    px = float(ctx.calib("pixel_size_um") or 1.0)
    zs = float(ctx.calib("z_step_um") or 1.0)
    sx = to_pixels_v2(float(ctx.params.get("shift_x", 0.0)), "um", pixel_size_um=px)
    sy = to_pixels_v2(float(ctx.params.get("shift_y", 0.0)), "um", pixel_size_um=px)
    sz = (to_pixels_v2(float(ctx.params.get("shift_z", 0.0)), "um_axial", z_step_um=zs)
          if volumetric else 0.0)
    angle = float(ctx.params.get("angle", 0.0))
    order = _INTERP_ORDER.get(
        str(ctx.params.get("__modes__", {}).get("interp", "linear")), 1)

    # Nothing to do — and nothing to CLAIM. An untouched Dataset must not pick up a
    # sampling stamp, or a `raw` branch would be refused by a node that moved no pixel.
    if angle == 0.0 and sx == 0.0 and sy == 0.0 and sz == 0.0:
        return ds

    prov = ds.image
    if prov is None:
        raise ValueError("transform needs an image provider on its input Dataset")
    ax = prov.axes

    requested = parse_channels(ctx.params.get("channels"))
    if requested:
        channels = frozenset(c for c in requested if 0 <= c < ax.c)
        if not channels:
            raise ValueError(
                f"transform: {sorted(requested)} names no channel of this Dataset, which "
                f"has {ax.c} — the only valid indices are "
                f"{'0' if ax.c == 1 else f'0..{ax.c - 1}'}, and negative values are out of "
                f"range rather than 'from the end'. Clear the socket to move every "
                f"channel.")
    else:
        channels = frozenset(range(ax.c))

    structures = sorted({(a.domain.value, a.layer or a.name)
                         for a in ds.attributes.values() if is_structure(a.domain)})
    if structures:
        raise ValueError(
            f"transform moves pixels, and this Dataset also carries structure tables "
            f"{structures} whose y/x/z COLUMNS record where those objects used to be. "
            f"Moving the raster while leaving the table behind would report every object "
            f"at its old position, so the transform is refused rather than applied to "
            f"half the bundle. Put Transform BEFORE the segmentation / detection step — "
            f"which is where a channel registration belongs — or zero the shifts and the "
            f"angle to pass through.")

    mat3, off3 = _rigid_inverse(angle, (sz, sy, sx), (ax.z, ax.y, ax.x))
    mat2, off2 = _rigid_inverse(angle, (sy, sx), (ax.y, ax.x))

    def move_volume(v: np.ndarray, m: int, t: int, c: int) -> np.ndarray:
        return _rigid_warp(v, mat3, off3, order) if c in channels else v

    def move_plane(a: np.ndarray, c: int) -> np.ndarray:
        return _rigid_warp(a, mat2, off2, order) if c in channels else a

    cache = ctx.tiles
    if cache is not None:
        # WHOLE_PLANE / WHOLE_VOLUME lazy unit: the resampler reads across the whole unit
        # (a shift pulls from anywhere in it), and the geometry is baked from eagerly
        # resolved params, so the closures never touch ctx (V2.04 §6b).
        fp = stream_fp("transform", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (), prov)
        if volumetric:
            out = ds.with_image(VolumeComputeProvider(prov, move_volume, fp=fp,
                                                      cache=cache))
        else:
            out = ds.with_image(MapComputeProvider(
                prov, lambda a, m, t, z, c, *_: move_plane(a, c),
                unit="plane", fp=fp, cache=cache))
    else:                                          # pre-C1 eager fallback (bare ctx)
        buf = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        units = [(m, t, c) for m in range(ax.m) for t in range(ax.t)
                 for c in range(ax.c)]
        for done, (m, t, c) in enumerate(units):
            if volumetric:
                vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
                buf[m, t, :, c] = move_volume(np.asarray(vol, dtype=float), m, t, c)
            else:
                for z in range(ax.z):
                    plane = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
                    buf[m, t, z, c] = move_plane(np.asarray(plane, dtype=float), c)
            ctx.progress(done + 1, len(units), "transforming", frames=ax.t)
        out = ds.with_image(ArrayProvider(buf))

    # Voxel layers are lattice arrays over the SAME grid, so they must make the same move
    # or a mask would be left describing where its objects used to be. Integer rasters
    # (masks, class maps) resample at order 0 whatever the Mode says — averaging ids would
    # invent ones that were never assigned.
    for attr in ds.layers_on(Domain.VOXEL):
        src = np.asarray(attr.values)
        is_int = not np.issubdtype(src.dtype, np.floating)
        lorder = 0 if is_int else order
        work = src.astype(np.int64 if is_int else np.float64)
        moved = work.copy()                        # unselected channels keep their values
        for m in range(ax.m):
            for t in range(ax.t):
                for c in sorted(channels):
                    if volumetric:
                        moved[m, t, :, c] = _rigid_warp(work[m, t, :, c], mat3, off3,
                                                        lorder)
                    else:
                        for z in range(ax.z):
                            moved[m, t, z, c] = _rigid_warp(work[m, t, z, c], mat2, off2,
                                                            lorder)
        out = out.with_layer(Domain.VOXEL, attr.name, moved.astype(src.dtype), attr.layer)

    return _sampled(out, f"transform.rigid[c{sorted(channels)},dz{sz:.4f},dy{sy:.4f},"
                         f"dx{sx:.4f},a{angle:.4f},o{order}]")
#: Shared tail for the three translation sockets. What differs between them is the axis and
#: which way positive points; the rest — that the frame does not grow, that the vacated
#: strip is zero, that the value is physical — is the same paragraph three times over.
_SHIFT_DOC_TAIL = (
    "The frame does NOT grow: content pushed past the edge is discarded and the strip it "
    "vacated is filled with zeros, so a large value permanently loses that band of data. "
    "That is what keeps the output addressable voxel-for-voxel against the channels that "
    "did not move. Pixels are resampled, so anything measured after this node reads "
    "interpolated values rather than the raw counts.")
register_node(
    _compute_transform, op_key="transform.rigid", label="Transform",
    category="transform",
    inputs=[
        InDataset(),
        InString("channels", "Channels", field=False, default="", pick_kind="channels",
                 description=
                 "Which channels MOVE, as 0-based indices — \"1\" moves the second channel "
                 "and leaves every other one exactly where it was. That asymmetry is the "
                 "point: moved and unmoved channels come out on the SAME wire at the same "
                 "axes, so colocalization, a ratio or an overlay downstream sees the "
                 "corrected alignment without any merge step. EMPTY moves every channel, "
                 "which slides the whole picture inside its frame rather than registering "
                 "one channel against another — rarely what you want here. Order and "
                 "duplicates are irrelevant (unlike Select Channel, this never reorders the "
                 "axis); a non-empty list naming no existing channel is refused rather than "
                 "silently doing nothing."),
        InFloat("shift_x", "Shift X", unit="um", field=False, default=0.0,
                description=
                "How far the selected channels move ACROSS the frame, in microns — positive "
                "goes RIGHT, toward higher X. In microns rather than pixels so a chromatic "
                "offset measured once still means the same physical distance at another "
                "objective, binning or camera; an image with no declared pixel size falls "
                "back to 1 µm = 1 px. " + _SHIFT_DOC_TAIL),
        InFloat("shift_y", "Shift Y", unit="um", field=False, default=0.0,
                description=
                "How far the selected channels move DOWN the frame, in microns — positive "
                "goes toward higher Y, which is downward as the image is displayed, since "
                "row 0 is the top. Same physical authoring as Shift X, and the same 1 µm = "
                "1 px fallback on an uncalibrated image. " + _SHIFT_DOC_TAIL),
        InFloat("shift_z", "Shift Z", unit="um_axial", field=False, default=0.0,
                available_in={"dim": frozenset({"3D"})},
                description=
                "How far the selected channels move THROUGH the stack, in microns along Z — "
                "positive goes toward higher plane indices. Measured against the axial step, "
                "which is usually far coarser than the pixel, so check your z spacing before "
                "typing a number: a value below one z step moves the data less than a single "
                "plane and mostly just blurs it between neighbours. 3D ONLY — with the lever "
                "on 2D every plane is transformed independently and nothing crosses z, so "
                "this control is hidden. Falls back to 1 µm = 1 plane if the file declares "
                "no z step. " + _SHIFT_DOC_TAIL),
        InFloat("angle", "Rotation", unit="", field=False, default=0.0,
                description=
                "In-plane rotation of the selected channels, in DEGREES about the centre of "
                "the frame. Positive turns the picture CLOCKWISE as displayed, the same sign "
                "as ImageJ's Rotate. Always in the Y–X plane, in 3D as well — the whole "
                "stack turns about the vertical axis, it never tilts. Applied FIRST, with "
                "the translations added after, so the shifts mean the same thing whatever "
                "the angle. Corners rotate out of the frame and are lost, and the triangles "
                "they leave behind are zero, so even a small angle empties the edges — "
                "crop afterwards if you are going to measure near them."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("interp", ["linear", "nearest", "cubic"], default="linear",
                label="Interpolation",
                description=
                "How pixel values are sampled when the transform lands between voxels. A "
                "sub-pixel shift or any rotation must interpolate, so this is a genuine "
                "choice about what the moved pixels MEAN; a whole-pixel translation lands "
                "exactly and every option gives the same bytes. It also decides whether "
                "label ids survive the move.",
                choice_docs={
                    "linear":
                        "Weighted average of the neighbouring voxels. The usual choice for "
                        "intensity images: smooth, no overshoot, and the error is a slight "
                        "blur that grows with each successive transform. It INVENTS "
                        "intermediate values, so a label raster comes out with meaningless "
                        "ids between the real ones.",
                    "nearest":
                        "Copy the closest voxel's value — no averaging at all. The only safe "
                        "option for LABEL rasters and masks, where an averaged id is not an "
                        "id. On intensity data it preserves the exact histogram at the cost "
                        "of visible jagged edges and up to half a voxel of positional error.",
                    "cubic":
                        "Third-order spline: the sharpest of the three, and the best at "
                        "preserving fine texture through a rotation or a resample. It can "
                        "RING — overshoot above the brightest and below the darkest "
                        "neighbouring value at a hard edge — so a saturated boundary can gain "
                        "a dark halo, and it is the slowest.",
                })],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    description="Move selected channels' pixels inside a FIXED frame — translate in X/Y "
                "(and Z in 3D) and rotate in-plane about the centre — cropping whatever "
                "leaves the frame and zero-filling what it vacates. Channels not named pass "
                "through at their original position, so corrected and raw ones arrive "
                "downstream on one wire and can be compared directly; the usual reason to "
                "reach for it is a chromatic or filter-cube offset before colocalization. "
                "Axes, pixel size and origin are unchanged — only content moves. "
                "Interpolation is linear / nearest / cubic; Voxel layers such as masks ride "
                "along with the same transform (integer rasters always nearest), and a "
                "Label / Point / Track table on the wire is refused because its coordinate "
                "columns cannot be honestly rewritten.")
