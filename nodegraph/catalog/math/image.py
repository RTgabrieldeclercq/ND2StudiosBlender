"""Image Math (``math.image``) — pixel arithmetic between this image and another (or a
constant): add, subtract, multiply, divide, min, max, absolute difference, mean. Lazy per
plane; the value scale (``bit_depth``) follows the operation."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import image_math as _meta_image_math
from nodegraph.provider import ArrayProvider
from nodegraph.registry import Granularity, InDataset, InFloat, Mode, OutDataset
from nodegraph.streaming import MapComputeProvider, stream_fp

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.planes import _each_plane_p
from nodegraph.catalog._shared.sampling import _sampling_of

# ── Image Math (V4.00 step 12) ───────────────────────────────────────────────────
#
# Pixel-wise arithmetic is the one enhancement the catalog did not have: subtract a
# background image computed on another branch, ratio two channels, average two exposures.
# The node is deliberately small — eight operations, one optional second wire, one constant
# — and the thinking is in two places: the second wire's GEOMETRY rule (it may be a
# projection or a single channel of the same field, never a different field), and what the
# operation does to the meaning of the numbers (`metadata.image_math`: a sum can exceed the
# camera's range, a product or a ratio is not counts at all).

_IMAGE_OPS: Tuple[str, ...] = (
    "add", "subtract", "multiply", "divide", "min", "max", "difference", "mean")


def _apply(op: str, a: np.ndarray, b: Any) -> np.ndarray:
    """One operation on float planes. Division by zero is NaN, not 0 and not an error: a
    zero in the divisor is a pixel with no information for a ratio, and NaN is the only
    value a consumer cannot mistake for a measurement."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "add":
            return a + b
        if op == "subtract":
            return a - b
        if op == "multiply":
            return a * b
        if op == "divide":
            bb = np.asarray(b, dtype=float)
            safe = np.where(bb != 0, bb, 1.0)
            return np.where(bb != 0, a / safe, np.nan)
        if op == "min":
            return np.minimum(a, b)
        if op == "max":
            return np.maximum(a, b)
        if op == "difference":
            return np.abs(a - b)
        if op == "mean":
            return 0.5 * (a + b)
    raise ValueError(f"image math: unknown operation {op!r} — one of {list(_IMAGE_OPS)}")


def _check_other(ds: Dataset, other: Dataset) -> None:
    """The second image must address the SAME pixels: y/x equal; m/t/z/c equal or 1 (a
    projection, a single channel, one timepoint broadcast over the series); and the two
    branches must share their sampling provenance on every axis the second one does not
    collapse — a crop, resample or drift correction on one side only is refused."""
    a, b = ds.axes, other.axes
    if (b.y, b.x) != (a.y, a.x):
        raise ValueError(
            f"image math: the `other` image is {b.y}×{b.x} px and the data {a.y}×{a.x} — "
            f"they are combined pixel for pixel, so crop or resample both branches the "
            f"same way (or branch `other` off the point where the geometry already "
            f"matches)")
    bad = [n for n in "mtzc" if getattr(b, n) not in (1, getattr(a, n))]
    if bad:
        raise ValueError(
            f"image math: the `other` image's {bad} axes ({[getattr(b, n) for n in bad]}) "
            f"neither match the data ({[getattr(a, n) for n in bad]}) nor are 1 — a second "
            f"image may be one projection, one timepoint or one channel applied to all, "
            f"not a different count of them")
    ignore = frozenset(n for n in "mtzc" if getattr(b, n) == 1)
    if _sampling_of(ds, ignore_axes=ignore) != _sampling_of(other, ignore_axes=ignore):
        raise ValueError(
            "image math: the `other` branch has had a different crop / resample / "
            "alignment applied than the data, so its pixels sit at different places — "
            "apply the same geometry to both branches, or combine them before it")


def _compute_image_math(ctx: EvalContext) -> Dataset:
    """Combine this image with another image or a constant, pixel by pixel.

    Resolved spec (build-node-v2 §0, V4.00 step 12)
    -----------------------------------------------
    * **Kind** math → ``op_key="math.image"``, category ``"math"``. Image→image; the
      Dataset's layers and tables pass through untouched.
    * **Operands.** ``data``'s image, and either ``other``'s image (when wired —
      :func:`_check_other` says what it may be) or the constant ``value``.
    * **2D/3D** none: pointwise. **Footprint** ``WHOLE_PLANE`` over y,x — the second image
      is read as a whole plane at the same ``(m,t,z,c)`` (broadcast where it is size 1), so
      the unit is a plane. Lazy (:class:`~nodegraph.streaming.MapComputeProvider`), float
      output; eager when there is no tile cache.
    * **Value scale.** ``metadata.image_math`` predicts it (a sum widens ``bit_depth`` by
      one bit; a product or a ratio drops it; the rest keep it) and the payload is SYNCED
      to that envelope through ``ctx.calib("bit_depth")``, never re-derived here.
    """
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("image math needs an image provider on its `data` input")
    modes = ctx.params.get("__modes__", {}) or {}
    op = str(modes.get("op") or "add")
    if op not in _IMAGE_OPS:
        raise ValueError(f"image math: unknown operation {op!r} — one of {list(_IMAGE_OPS)}")
    ax = prov.axes
    other = ctx.input("other")
    raw_value = ctx.params.get("value")
    value = float(raw_value) if raw_value is not None else 1.0
    extra = {}
    if other is not None:
        oprov = getattr(other, "image", None)
        if oprov is None:
            raise ValueError(
                "image math: the `other` input carries no image — wire an image Dataset "
                "into it, or leave it unwired to use the constant `Value`")
        _check_other(ds, other)
        oax = oprov.axes

        def fn(a, m, t, z, c, gy0, gy1, gx0, gx1):
            idx = (m if oax.m > 1 else 0, t if oax.t > 1 else 0,
                   z if oax.z > 1 else 0, c if oax.c > 1 else 0)
            b = np.asarray(oprov.get_region(0, *idx, gy0, gy1, gx0, gx1), dtype=float)
            return _apply(op, a, b)
        # the second image's identity, so two nodes alike in every param but their
        # `other` never share a cached plane
        extra["__other__"] = str(oprov.fingerprint())
    else:
        def fn(a, m, t, z, c, gy0, gy1, gx0, gx1):
            return _apply(op, a, value)
    cache = ctx.tiles
    lazy = cache is not None and ax.y * ax.x * 8 <= cache.budget // 2
    if lazy:
        fp = stream_fp("map", ctx.op_key, {**dict(ctx.params), **extra},
                       ctx.reads.declared_reads(), (), prov)
        out = ds.with_image(MapComputeProvider(prov, fn, halo=0, unit="plane",
                                               fp=fp, cache=cache))
    else:
        arr = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
        for m, t, z, c in _each_plane_p(ctx, ax, "image math"):
            plane = np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), dtype=float)
            arr[m, t, z, c] = fn(plane, m, t, z, c, 0, ax.y, 0, ax.x)
        out = ds.with_image(ArrayProvider(arr))
    # SYNCED from the envelope `image_math` already produced: None removes the key
    return out.with_metadata(bit_depth=ctx.calib("bit_depth"))


register_node(
    _compute_image_math, op_key="math.image", label="Image Math", category="math",
    inputs=[
        InDataset(description=
                  "The image on the left of the operation (A). Its layers, tables and "
                  "calibration pass through; only the pixels change."),
        InDataset("other", label="Other", passes_domains=False,
                  description=
                  "Optional: the image on the right (B) from another branch — a background "
                  "estimate, the other channel, a reference frame. It must cover the same "
                  "pixels: y/x equal, and each of m/t/z/c either equal to the data's or 1 "
                  "(a projection, a single timepoint or a single channel is applied to all). "
                  "Unwired, `Value` is B. Only its pixels are read; its layers do not pass "
                  "downstream."),
        InFloat("value", "Value", field=False, default=1.0, unit="",
                description=
                "The constant B when `Other` is unwired, in the image's own units (camera "
                "counts, or 0–1 after a Normalize): `add` 100 lifts every pixel by 100 "
                "counts, `multiply` 0.5 halves them, `divide` by the exposure time turns "
                "counts into a rate. Ignored while `Other` is wired."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("op", list(_IMAGE_OPS), default="subtract", label="Operation",
             description=
             "What is done to each pixel of A with the matching pixel of B (or the "
             "constant). It also decides what the numbers MEAN afterwards: a sum can exceed "
             "the camera's range and widens the declared depth by a bit; a product or a "
             "ratio is no longer counts and drops it, so a fixed threshold downstream reads "
             "the data's own scale; the rest keep it.",
             choice_docs={
                 "add": "A + B. Sum two images — two exposures, two channels. The result can "
                        "exceed the camera's range, so the declared bit depth widens by one "
                        "bit; a fixed threshold downstream sees that.",
                 "subtract": "A − B. Remove a background image estimated on another branch, "
                             "or difference two timepoints. Negative values are kept (float "
                             "output); clip with `max` against 0 if you need none.",
                 "multiply": "A × B. Weight an image by another (a mask as 0/1, a flat-field "
                             "gain). The product is not camera counts any more, so the bit "
                             "depth is dropped and consumers read the data's own scale.",
                 "divide": "A ÷ B. Ratio two channels, or flat-field by a reference. Where B "
                           "is 0 the result is NaN (no information), never 0 or an error. "
                           "Dimensionless: the bit depth is dropped.",
                 "min": "The smaller of A and B at each pixel. Clamps an image from above by "
                        "another or by a constant; keeps the counts scale.",
                 "max": "The larger of A and B at each pixel. Clamps from below — `max` "
                        "against 0 removes negatives after a subtraction — or merges two "
                        "channels by the brightest; keeps the counts scale.",
                 "difference": "|A − B|, the absolute difference. Change between two frames "
                               "or disagreement between two processings, always ≥ 0; keeps "
                               "the counts scale.",
                 "mean": "(A + B) ÷ 2. Average two images without leaving the camera's "
                         "range — the way to combine two exposures when `add` would clip "
                         "downstream.",
             })],
    granularity=Granularity.WHOLE_PLANE, kernel_axes=frozenset({"y", "x"}),
    meta_transform=_meta_image_math,
    description="Pixel arithmetic between this image and another branch's image, or a "
                "constant: add, subtract, multiply, divide, min, max, absolute difference, "
                "mean. Lazy per plane, float output; the second image may be a projection "
                "or a single channel of the same field, never a different field.")
