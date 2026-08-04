"""raw measure — shared catalog helpers."""

from __future__ import annotations


from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import InDataset, SocketSpec

from nodegraph.catalog._shared.sampling import _sampling_of

#: The optional **raw-intensity** Dataset socket (2026-07-28). Declared AFTER ``data`` so
#: ``data`` stays the primary — ``graph.dataset_preds`` sorts by declared socket position,
#: so calibration/domain propagation keeps flowing from the main chain no matter which edge
#: the user wired first.
def _InRaw() -> SocketSpec:
    return InDataset("raw", label="Raw")
def _intensity_provider(ctx: EvalContext, ds: Dataset):
    """The pixel source for an intensity **measurement**: the optional ``raw`` Dataset
    input when one is wired, else the node's own image. Returns ``(provider, is_raw)``.

    This is the "segment on enhanced, measure on raw" seam. The scope is deliberately
    narrow — ``raw`` overrides **measurement** only, never segmentation: the mask, the
    label raster and every threshold still come from the main input, so the node's
    geometry and its calibration env stay one consistent chain and only the numbers being
    *reported* change. (A node that thresholds raw pixels doesn't need a lever — wire the
    raw Dataset into its main input.)

    Geometry must match **exactly**. A cropped/resampled/projected raw would be read
    voxel-for-voxel against the main raster and report neighbouring objects' intensities,
    which is silent corruption, so a mismatch is refused.

    **Two checks, because size equality is not geometry equality (V2.17).** The shape
    comparison catches a crop or a resample; it does NOT catch anything that keeps the
    axis sizes and moves the content — and ``align.drift`` / ``registration.stabilize``
    are exactly that. So the sampling provenance (:data:`SAMPLING_KEY`, stamped by
    :func:`_sampled`) is compared as well: the two branches must have had the *same
    sequence* of geometry operations applied since the source, which is the property that
    actually makes a voxel-for-voxel read meaningful.

    **A channel tap is exempt when both branches are single-channel** (2026-08-03).
    "Segment channel 0, measure channel 1" is the canonical reason this socket exists, and
    it is two taps off one file: same voxel grid, different channel. Comparing their FULL
    provenance refused it — ``c:channel.select(0,)`` against ``c:channel.select(1,)`` — with
    an error about a shift that does not exist, and advice ("apply the same ones to both")
    that describes the one thing the user must not do. With ``c == 1`` on both sides the
    channel index is 0 in each, so a channel reindex has nothing left to misalign; at
    ``c > 1`` a permutation does, and there those stamps stay compared."""
    prov = ds.image
    raw = ctx.input("raw")
    if raw is None:
        return prov, False
    rprov = getattr(raw, "image", None)
    if rprov is None:
        raise ValueError(
            "the `raw` input carries no image provider — wire an image Dataset (the "
            "unenhanced source) into it, or leave it unwired to measure the main input.")
    if prov is not None and rprov.axes != prov.axes:
        shape = lambda a: (a.m, a.t, a.z, a.c, a.y, a.x)   # AxisSizes is not iterable
        raise ValueError(
            f"the `raw` input's geometry {shape(rprov.axes)} does not match the measured "
            f"input's {shape(prov.axes)} — they are read voxel-for-voxel, so a mismatch "
            "would report the wrong regions' intensities. Apply the same crop / resample "
            "/ z-project to BOTH branches, or branch `raw` off the point in the chain "
            "where the geometry already matches. If only the channel COUNT differs, tap "
            "`raw` down to one channel as well — it may be a DIFFERENT channel from the "
            "measured one, which is the whole point of the socket.")
    # both single-channel ⇒ the c axis is degenerate on each side, so a channel tap cannot
    # misalign the voxel-for-voxel read and its stamp is dropped from the comparison.
    per_channel = ds.axes.c > 1 or raw.axes.c > 1
    mine = _sampling_of(ds, channel_axis=per_channel)
    theirs = _sampling_of(raw, channel_axis=per_channel)
    if mine != theirs:
        only_mine = [s for s in mine if s not in theirs] or ["(none)"]
        only_theirs = [s for s in theirs if s not in mine] or ["(none)"]
        raise ValueError(
            f"the `raw` input went through a DIFFERENT sampling geometry from the measured "
            f"input, so the two do not address the same voxels even though their sizes "
            f"agree. Only on the measured branch: {only_mine}. Only on `raw`: "
            f"{only_theirs}. A shift is the dangerous case — a drift-corrected image and "
            f"its uncorrected source have identical axes, so each object would be measured "
            f"wherever it used to be. Branch `raw` off AFTER the geometry ops the measured "
            f"branch has (enhancement steps in between are fine and are the whole point), "
            f"or apply the same ones to both.")
    return rprov, True
