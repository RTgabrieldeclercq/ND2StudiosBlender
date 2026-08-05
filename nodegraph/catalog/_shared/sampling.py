"""sampling — shared catalog helpers."""

from __future__ import annotations


from typing import Any

from nodegraph.dataset import Dataset

#: Namespaced, non-calibration metadata key holding this Dataset's **sampling grid
#: provenance** — the ordered tuple of geometry operations applied since the source
#: (V2.17). It is what makes "these two branches address the same voxels" checkable.
SAMPLING_KEY = "__sampling__"

#: The axis letters a stamp may declare itself confined to.
_STAMP_AXES = "mtzcyx"

#: Prefix marking a stamp whose effect is confined to the **channel** axis — a channel tap.
#: It is the ``c`` case of the general form below, kept as a name because three modules spell
#: it.
CHANNEL_STAMP = "c:"

#: Prefix marking a stamp whose effect is confined to the **z** axis: it collapses or
#: selects along z and leaves every ``(y,x)`` address pointing at the same lateral location.
#: A Z-projection always qualifies; a crop qualifies only when its ``y``/``x`` window is the
#: full extent (a pure z-crop), which is why ``util.crop`` decides per call.
Z_STAMP = "z:"
def _stamp_axes(stamp: Any) -> frozenset:
    """The axes a stamp declares its effect confined to (empty = the whole grid).

    The prefix form is ``"<axes>:"`` — ``"c:channel.select(1,)"``, ``"z:zproject[max]"``.
    Empty is the safe default and the parse is deliberately strict: anything that is not
    purely axis letters before the first colon reads as unmarked, so a stamp that happens to
    contain a colon (``"crop[z5:6,y0:64,x0:64]"``) is never mistaken for a declaration."""
    head, sep, _rest = str(stamp).partition(":")
    if not sep or not head or set(head) - set(_STAMP_AXES):
        return frozenset()
    return frozenset(head)
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
def _sampling_of(ds: Any, *, channel_axis: bool = True,
                 ignore_axes: frozenset = frozenset()) -> tuple:
    """A Dataset's sampling provenance as a tuple (``()`` = untouched since the source).

    ``ignore_axes`` drops every stamp whose declared axes (:func:`_stamp_axes`) are a subset
    of it. The rule it encodes is one sentence: **a stamp confined to axes that are singleton
    on both sides being compared cannot misalign anything.** An unmarked stamp is never
    dropped, whatever is passed.

    ``channel_axis=False`` is the original, narrower spelling of ``ignore_axes={"c"}``, kept
    because two callers read naturally that way.

    The channel case is what forced the concept (2026-08-03): segment on channel 0, measure
    the intensity of channel 1. Those two branches are two taps off one file — identical
    voxel grid, different channel — yet their full provenance differs
    (``c:channel.select(0,)`` against ``c:channel.select(1,)``), so a strict comparison
    refused the wiring with a geometry-mismatch error about a shift that does not exist.
    Overlaying channel 0 on channel 1 was refused the same way.

    The **z** case is the same shape and arrived with ``analysis.voronoi``'s second input
    (2026-08-04): dots from a Z-PROJECTION of one channel, areas from a single-plane
    Z-CROP of another. Both branches end at ``z == 1`` and neither moved a ``(y,x)``
    address, so they do address the same voxels — but ``z:zproject[max]`` against
    ``z:crop[z5:6,…]`` is a difference, and the guard refused a graph that was correct.

    Singleton-on-both-sides is the condition, not a courtesy: with ``c == 1`` (or ``z == 1``)
    on each side that index is 0 in both and a reindex has nothing left to misalign. At
    ``c > 1`` a channel PERMUTATION really does change which channel an index means, and at
    ``z > 1`` a z-crop really does change which plane one is — so there the stamps stay
    compared and a mismatched branch is still refused."""
    stamps = tuple(getattr(ds, "metadata", {}).get(SAMPLING_KEY, ()))
    drop = set(ignore_axes) | (set() if channel_axis else {"c"})
    if not drop:
        return stamps
    return tuple(s for s in stamps
                 if not (_stamp_axes(s) and _stamp_axes(s) <= drop))
def _require_same_grid(main: Any, other: Any, *, socket: str, consequence: str) -> None:
    """Refuse unless ``other`` addresses the same voxels as ``main`` (V2.22).

    The check every **second Dataset input** needs, factored out of
    :func:`_intensity_provider` when a second one arrived. Any node that reads two branches
    voxel-for-voxel — ``analysis.measure`` sampling ``raw`` pixels under the main input's
    labels, ``analysis.voronoi`` clipping one branch's seed dots to another branch's areas —
    is silently wrong rather than noisily wrong if the two grids disagree, so the guard has
    to live somewhere both can reach.

    ``consequence`` says what goes wrong for THIS node, because that is the part a shared
    message cannot know and the part that tells the user whether they have a real problem.

    Both halves are load-bearing, and the second is the one that was missing originally:

    * **AxisSizes** catches a crop, a resample or a projection;
    * **sampling provenance** (:func:`_sampling_of`) catches everything that keeps the axis
      sizes and moves the content — ``align.drift`` and ``registration.stabilize`` are
      exactly that, and a shape-only guard waved them straight through.

    **An axis that is singleton on BOTH sides is exempt** — see :func:`_sampling_of`. Two
    channel taps off one file share a voxel grid and differ only in which channel they
    carry; a Z-projection and a single-plane Z-crop both leave every ``(y,x)`` address where
    it was. Those are the workflows these sockets exist for. Where the axis is NOT singleton
    a reindex along it really can misalign, so there the stamps stay compared."""
    a = getattr(main, "axes", None)
    b = getattr(other, "axes", None)
    if a is not None and b is not None and a != b:
        shape = lambda x: (x.m, x.t, x.z, x.c, x.y, x.x)   # AxisSizes is not iterable
        raise ValueError(
            f"the `{socket}` input's geometry {shape(b)} does not match the main input's "
            f"{shape(a)} — they are read voxel-for-voxel, so a mismatch would {consequence}. "
            f"Apply the same crop / resample / z-project to BOTH branches, or branch "
            f"`{socket}` off the point in the chain where the geometry already matches. If "
            f"only the channel COUNT differs, tap `{socket}` down to one channel as well — "
            f"it may be a DIFFERENT channel, which is the whole point of the socket.")
    # `m`/`t` are deliberately NOT in this set. Collapsing them picks a particular position
    # or timepoint, and treating "frame 3" and "frame 7" as one grid is a content decision
    # this guard has no basis to make on the user's behalf; `z`/`c` collapse to a plane or a
    # channel of the SAME field of view, which is exactly what these sockets are for.
    singleton = frozenset(ax for ax in ("z", "c")
                          if (getattr(a, ax, 1) or 1) == 1 and (getattr(b, ax, 1) or 1) == 1)
    mine = _sampling_of(main, ignore_axes=singleton)
    theirs = _sampling_of(other, ignore_axes=singleton)
    if mine != theirs:
        only_mine = [s for s in mine if s not in theirs] or ["(none)"]
        only_theirs = [s for s in theirs if s not in mine] or ["(none)"]
        raise ValueError(
            f"the `{socket}` input went through a DIFFERENT sampling geometry from the main "
            f"input, so the two do not address the same voxels even though their sizes "
            f"agree. Only on the main branch: {only_mine}. Only on `{socket}`: "
            f"{only_theirs}. A shift is the dangerous case — a drift-corrected image and its "
            f"uncorrected source have identical axes, so it would {consequence}. Branch "
            f"`{socket}` off AFTER the geometry ops the main branch has (enhancement steps "
            f"in between are fine and are the whole point), or apply the same ones to both.")
