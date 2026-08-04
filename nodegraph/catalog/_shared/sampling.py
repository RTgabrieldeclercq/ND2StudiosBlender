"""sampling — shared catalog helpers."""

from __future__ import annotations


from typing import Any

from nodegraph.dataset import Dataset

#: Namespaced, non-calibration metadata key holding this Dataset's **sampling grid
#: provenance** — the ordered tuple of geometry operations applied since the source
#: (V2.17). It is what makes "these two branches address the same voxels" checkable.
SAMPLING_KEY = "__sampling__"

#: Prefix marking a stamp that reindexes **only the channel axis** — a channel tap. It is
#: recorded like any other stamp, but it is the one kind a consumer can legitimately ignore:
#: see :func:`_sampling_of`.
CHANNEL_STAMP = "c:"
def _sampled(ds: Dataset, *stamps: str) -> Dataset:
    """Append geometry ``stamps`` to ``ds``'s sampling provenance (§7b stamp-and-inherit).

    A node calls this whenever it changes **which physical location a given (z,y,x) index
    refers to** — a crop (origin moves), a resample (scale changes), a z-project or a
    temporal stack (an axis collapses), a channel tap (c reindexes), a frame slice, or a
    drift/stabilize apply (content shifts under a fixed index grid). Enhancement nodes
    must NOT call it: they rewrite intensities at the same addresses, which is exactly the
    "segment on enhanced, measure on raw" case the `raw` socket exists to support.

    A **channel tap** stamps with the :data:`CHANNEL_STAMP` prefix, because it is the one
    entry in that list whose effect is confined to the ``c`` axis: it leaves every ``(z,y,x)``
    address pointing at the same physical location. Consumers that read a single channel
    drop those stamps (``_sampling_of(ds, channel_axis=False)``) — see there for why.

    Why a stamp rather than a shape comparison: :func:`_intensity_provider` used to accept
    any second Dataset whose ``AxisSizes`` matched, and size equality is not geometry
    equality. ``align.drift`` and ``registration.stabilize`` are shape-PRESERVING and
    origin-MOVING, so the guard waved them through and the node reported each object's
    intensity from wherever that object used to be — measured on the real ND2 with a 40 px
    shift, ``mean_intensity`` came back at median 242.3 against a true 879.8. Worse, the
    old error text told users to wire ``raw`` from *before* any crop/resample, which walks
    straight into it.

    Non-calibration and namespaced, so ``_StrictCalibMetadata`` passes it through and no
    ``meta_transform`` has to predict it — the same contract ``__struct_zkind__`` uses. It
    rides the payload, so a change upstream bumps that node's revision and the consumer's
    ``recipe_hash`` already folds it in; nothing extra is needed for the memo."""
    prior = tuple(ds.metadata.get(SAMPLING_KEY, ()))
    return ds.with_metadata(**{SAMPLING_KEY: prior + tuple(stamps)})
def _sampling_of(ds: Any, *, channel_axis: bool = True) -> tuple:
    """A Dataset's sampling provenance as a tuple (``()`` = untouched since the source).

    ``channel_axis=False`` drops the channel-tap stamps (:data:`CHANNEL_STAMP`), leaving
    only the provenance of the ``(z,y,x)`` grid. A consumer passes it when both Datasets it
    is comparing carry **exactly one** channel, and that is precisely the multi-channel
    workflow this whole guard was blocking (2026-08-03): segment on channel 0, measure the
    intensity of channel 1. Those two branches are two taps off one file — identical voxel
    grid, different channel — yet their full provenance differs (``c:channel.select(0,)``
    against ``c:channel.select(1,)``), so the strict comparison refused the wiring with a
    geometry-mismatch error about a shift that does not exist. Overlaying channel 0 on
    channel 1 was refused the same way.

    Single-channel is the condition, not a courtesy: with ``c == 1`` on both sides the
    channel index is 0 in each and there is nothing left for a reindex to misalign. It
    still holds for ``c > 1``, where a channel PERMUTATION (``"2,0"``) really does change
    which channel a given ``c`` index means — so those stamps stay compared, and a
    reordered branch is still refused."""
    stamps = tuple(getattr(ds, "metadata", {}).get(SAMPLING_KEY, ()))
    if channel_axis:
        return stamps
    return tuple(s for s in stamps if not str(s).startswith(CHANNEL_STAMP))
