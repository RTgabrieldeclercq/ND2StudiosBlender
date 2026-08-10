"""Edit-time metadata propagation — the MetaEnvelope pass (nodegraph v2, V2.03 §2 A3).

Directive A makes a node's metadata intelligence come from **the data package on its
own input edge**, as transformed by upstream nodes — not from static file metadata.
But evaluation is lazy demand-driven pull (V2.00 §9): nothing computes until a Viewer
pulls, so at graph-edit time there is no materialized intermediate Dataset to read.

This module supplies the missing piece: a **cheap forward pass that propagates only
metadata** — ``AxisSizes`` + the calibration dict + the attribute-layer catalog —
through each node's declared :attr:`NodeSpec.meta_transform`, **touching no pixels**.
It runs on load and on every graph edit; its output feeds edit-time widget re-seeding
and the 2D/3D lever's default (``z>1 ⇒ 3D``).

It is **advisory** (V2.03 §1): the two-hash memo key stays driven by the *eval-time*
recording ``ReadContext`` (V2.02 §8), resolved upstream-first at pull time. Statically
unknowable sizes (e.g. a stitched Y,X extent) are marked UNKNOWN rather than guessed.

Qt-free; pure standard library + numpy-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence,
                    Tuple)

from nodegraph.dataset import AxisSizes, LayerKey
from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice
from nodegraph.graph import Graph
from nodegraph.registry import layer_value


# ── the envelope ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetaEnvelope:
    """A node's resolved *metadata-only* output: axes, the calibration/metadata dict,
    the attribute-layer catalog, and the set of axes whose size is not statically
    knowable (e.g. a stitched extent — UNKNOWN, never a silent guess)."""

    axes: AxisSizes = field(default_factory=AxisSizes)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    layers: Tuple[LayerKey, ...] = ()
    unknown_axes: FrozenSet[str] = frozenset()
    #: The **accumulated domain-set** present on this node's Dataset output — the
    #: union of every upstream node's ``adds_domains`` (populated by
    #: :func:`propagate_meta`). Drives the GUI's socket domain-rail, the wire tint,
    #: and the domain-mismatch validation. Distinct from ``layers`` (the per-name
    #: attribute catalog, still a stub): this is the coarser domain granularity.
    domains: FrozenSet[Domain] = frozenset()
    #: The **layer catalog** (V2.11): the ``(domain, layer-name)`` pairs a Dataset on this
    #: edge is expected to carry, in first-appearance order. Drives the GUI's layer picker
    #: — a source-layer socket offers the layers actually present upstream instead of
    #: making the user retype a name.
    #:
    #: Distinct from ``layers`` above, which is a per-ATTRIBUTE ``LayerKey`` catalog and
    #: remains a stub. It could NOT be reused: ``LayerKey`` is ``(domain, layer, name)``,
    #: and the user-facing layer name sits in a *different* slot per domain family —
    #: ``with_layer`` leaves ``layer=None`` so a lattice layer is keyed
    #: ``(VOXEL, None, "mask")`` (name in slot 2), while ``with_structure`` stores each
    #: COLUMN separately as ``(POINT, "spots", "y")`` (layer in slot 1). Keying a picker on
    #: ``(domain, LayerKey[1])`` would collapse every Voxel layer in the graph to
    #: ``(VOXEL, None)`` and offer nothing. This field stores the user-facing name per
    #: domain directly, so the projection never arises.
    layer_names: Tuple[Tuple[Domain, str], ...] = ()

    @property
    def is_volumetric(self) -> bool:
        return self.axes.is_volumetric

    def with_domains(self, domains: FrozenSet[Domain]) -> "MetaEnvelope":
        return replace(self, domains=frozenset(domains))

    def with_layer_names(self, names: Sequence[Tuple[Domain, str]]) -> "MetaEnvelope":
        return replace(self, layer_names=tuple(names))

    def layers_in(self, domain: Domain) -> Tuple[str, ...]:
        """The layer names present on this edge for *domain*, in first-appearance
        order (the GUI picker's suggestion list)."""
        return tuple(n for d, n in self.layer_names if d is domain)

    def with_axes(self, axes: AxisSizes, *,
                  unknown: Optional[FrozenSet[str]] = None) -> "MetaEnvelope":
        return replace(self, axes=axes,
                       unknown_axes=self.unknown_axes if unknown is None else unknown)

    def with_metadata(self, **changes: Any) -> "MetaEnvelope":
        """Copy-on-write calibration edit; a ``None`` value removes the key
        (mirrors :meth:`nodegraph.dataset.Dataset.with_metadata`)."""
        new = {**self.metadata}
        for k, v in changes.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        return replace(self, metadata=new)


# ── named meta-transforms (V2.03 §2 A2) ───────────────────────────────────────
#
# Each maps an incoming envelope to the outgoing one given the node's params + its
# resolved mode state (incl. the 2D/3D ``dim`` lever). Axis-changing nodes MUST
# update calibration in lockstep (V2.03 §2 A2). ``params``/``modes`` are plain
# mappings; missing keys degrade to no-op (never a crash).

MetaTransform = Callable[["MetaEnvelope", Mapping[str, Any], Mapping[str, str]],
                         "MetaEnvelope"]


def identity(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    return env


# ── intensity provenance: calibration describes the CURRENT data, not the file ──
#
# The forward walk means every node reads its INPUT envelope, i.e. the most up-to-date
# metadata — so a node that changes what its numbers MEAN must restamp the affected key
# in lockstep, exactly like an axis-changing node restamps pixel size (V2.03 §2 A2).
# ``bit_depth`` is the intensity-domain instance of that rule: summing 16 12-bit frames
# yields 16-bit data, and normalizing to [0,1] yields data that is not integer counts at
# all. A downstream raw-count consumer that read the FILE's 12 bits in either case would
# be wrong. The two helpers below are the whole vocabulary; see `wire-node-v2` §7c.

def bit_depth_after_sum(env: MetaEnvelope, n: int) -> Dict[str, Any]:
    """``{"bit_depth": widened}`` for a reducer that SUMS ``n`` samples — ``b + ceil(log2
    n)`` bits, since n values of at most ``2**b - 1`` can total ``n*(2**b - 1)``. Empty
    when the incoming depth is unknown (nothing to widen) or ``n <= 1``. Mean/median/
    max/min/percentile reducers do NOT widen — they stay inside the input range."""
    b = env.metadata.get("bit_depth")
    if not b or n <= 1:
        return {}
    return {"bit_depth": int(b) + int(math.ceil(math.log2(int(n))))}


def value_rescaled(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """The meta_transform of a node whose output is no longer raw integer counts —
    percentile Normalize and CLAHE both return [0,1] floats. It DROPS ``bit_depth``:
    absent means "no declared integer scale", which is the honest signal every
    raw-count consumer downstream keys on: ``analysis.threshold``'s fixed level falls back to
    its 0.5 normalized-data default, and ``analysis.histogram_threshold`` switches its absolute
    thresholds into the DATA'S OWN units, so a cut of 0.35 means 0.35 (V2.28 — it used to refuse
    such an input outright, which left normalized data with no way to be thresholded at all).
    Axis-preserving — only the intensity meaning changes."""
    return env.with_metadata(bit_depth=None)


def flatten_field(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``enhance.flatten_field``: only the ``ratio`` method leaves the count scale behind.

    Three of the four methods return the image with a background removed or rebalanced —
    still counts, still inside the declared range — so ``bit_depth`` survives. ``ratio``
    divides the image BY its background, which is dimensionless and centred near 1: a raw-
    count consumer downstream (a fixed threshold, a full-range γ) would be badly wrong to
    read the sensor depth after it, so the key is dropped exactly as
    :func:`value_rescaled` does for a percentile Normalize (§7c).

    Axis-preserving either way — only the meaning of the numbers can change."""
    return (env.with_metadata(bit_depth=None)
            if (modes or {}).get("method") == "ratio" else env)


def subtract_background(env: MetaEnvelope, params: Mapping,
                        modes: Mapping) -> MetaEnvelope:
    """``enhance.subtract_background``: only ``combine="divide"`` leaves the count scale.

    The same rule as :func:`flatten_field`, over two Modes instead of one. Subtracting a
    background — clipped or signed — returns counts, so ``bit_depth`` survives; dividing by
    it returns "times the local background", which is dimensionless and centred near 1, so
    the key is dropped exactly as :func:`value_rescaled` does for a percentile Normalize
    (§7c). ``output="background"`` returns the *estimate*, which is in the input's own counts
    whatever the arithmetic would have been — so the ``combine`` state is read only for the
    ``corrected`` output, mirroring the ``available_in`` gate on that socket and the
    compute's own branch. Both defaults are spelled the way the compute spells them, since
    this runs on every keystroke against a state that may predate either Mode.

    Axis-preserving either way — only the meaning of the numbers can change."""
    m = modes or {}
    dimensionless = (str(m.get("output") or "corrected") == "corrected"
                     and str(m.get("combine") or "subtract") == "divide")
    return env.with_metadata(bit_depth=None) if dimensionless else env


# ── spatial provenance: WHERE the data is (origin_um) ──────────────────────────
#
# The companion rule to the intensity one above. `pixel_size_um` says how finely the data
# is sampled; `origin_um` says where that sampling starts on the microscope. Only two
# transforms move it — a crop (the cut corner becomes the new corner) and a stitch (the
# mosaic's corner is the union's) — and the rest MUST leave it alone. In particular a
# resample must not scale it: the field occupies the same patch of stage whether you
# sample it at 0.29 or 1.72 µm/px, and scaling here would double-count the change the
# transform already made to `pixel_size_um` (the V2.03 §2 A2 trap).

def read_origin_um(env: MetaEnvelope) -> Optional[List[List[float]]]:
    """``origin_um`` as a list of ``[z, y, x]`` triples, or ``None`` when unusable.

    Validated rather than trusted: a list that does not cover every multipoint is dropped
    whole, because `origin_um[m]` is addressed by INDEX and a short list silently reports
    some other position's corner — the same rule (and the same reason) as the stage logs in
    :func:`nodelab_v2.nd2_meta.read_nd2_metadata_extended`.
    """
    raw = env.metadata.get("origin_um")
    if not isinstance(raw, (list, tuple)) or len(raw) < max(1, env.axes.m):
        return None
    out: List[List[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        try:
            out.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError):
            return None
    return out


def shift_origin_um(env: MetaEnvelope, dz: float = 0.0, dy: float = 0.0,
                    dx: float = 0.0) -> Dict[str, Any]:
    """``{"origin_um": moved}`` for a transform that cuts into the field, else ``{}``.

    ``{}`` (not ``{"origin_um": None}``) when the key is absent, so a file that never had a
    position log is left alone rather than gaining a key that says "unknown" where nothing
    was ever claimed.
    """
    origins = read_origin_um(env)
    if origins is None:
        return {}
    return {"origin_um": [[o[0] + dz, o[1] + dy, o[2] + dx] for o in origins]}


#: EVERY metadata key that is a **list indexed by multipoint**. The per-M counterpart of
#: :data:`PER_CHANNEL_KEYS`, and it exists for exactly the same reason: these are read
#: POSITIONALLY (``xy[m]``), so a stale full-length list left behind by a node that
#: narrowed M does not look stale — it looks like the wrong POSITION. That is a worse
#: failure than the channel one it mirrors, because a tile placed at another field's stage
#: coordinate reads as a handedness bug, so the user reaches for Stitch's ``flip_x`` /
#: ``flip_y`` and makes it worse.
#:
#: The four of them, and why each is here:
#:
#: * ``origin_um`` — calibration (:data:`~nodegraph.dataset.CALIBRATION_KEYS`), the
#:   transform-maintained corner of voxel ``(m,0,0,0)``. :func:`~nodegraph.placement.field_box`
#:   prefers it over the stage log, so getting it wrong mis-places every placement consumer.
#: * ``stage_xy_um`` / ``stage_z_um`` (:data:`nodelab_v2.ingest.STAGE_KEYS`) — display
#:   provenance: where the camera was, which stops describing the data the moment a node
#:   crops or stitches.
#: * ``__align_um__`` (:data:`~nodegraph.placement.ALIGN_KEY`) and its companion
#:   ``align_to_ncc`` — the per-field correction ``registration.align_to`` measured, one row
#:   per M (``catalog/registration/align_to.py:137-141``), applied inside ``field_box``.
#:
#: ``frame_time_jd`` is deliberately ABSENT: it is indexed by T, not M.
PER_POSITION_KEYS: Tuple[str, ...] = (
    "origin_um", "stage_xy_um", "stage_z_um", "__align_um__", "align_to_ncc",
)


def position_subset(metadata: Mapping[str, Any], keep: Sequence[int]) -> Dict[str, Any]:
    """The ``{key: subset}`` changes that reindex every :data:`PER_POSITION_KEYS` list in
    ``metadata`` onto the multipoints ``keep`` (already validated indices, in output order).

    The per-M twin of :func:`channel_subset`, and shared for the same reason: a node or a
    run-scope that narrows M must subset all of these TOGETHER or the survivors stop
    describing the positions that are left.

    A key that is absent, or not a list, is left alone rather than invented. A list too
    SHORT to cover an index in ``keep`` is dropped whole rather than silently shortened —
    the same rule :func:`read_origin_um` applies, because a partial positional list reports
    some other field's coordinate instead of admitting it does not know.
    """
    changes: Dict[str, Any] = {}
    for key in PER_POSITION_KEYS:
        vals = metadata.get(key)
        if not isinstance(vals, (list, tuple)):
            continue
        changes[key] = ([vals[i] for i in keep] if all(0 <= i < len(vals) for i in keep)
                        else None)
    return changes


#: EVERY metadata key that is a **list indexed by timepoint**. The per-T member of the same
#: family as :data:`PER_CHANNEL_KEYS` and :data:`PER_POSITION_KEYS`, and it exists for the
#: third time for the same reason: ``frame_time_jd[t]`` is read POSITIONALLY, so a
#: full-length list left behind by a node that narrowed T does not look stale — it looks
#: like the wrong TIME. That one is the sharpest of the three, because
#: :func:`~nodegraph.placement.paired_t` uses it as the only clock two files share: a stale
#: list silently pairs frame 0 of one acquisition against frame 0 of the other's ORIGINAL
#: numbering, and channel.merge then reads two different moments as one.
#:
#: One member today. ``frame_timestamps_s`` is deliberately absent: it never reaches a
#: Dataset (:data:`nodelab_v2.ingest.PLACEMENT_KEYS` does not carry it), and ``dt_s`` is a
#: scalar INTERVAL rather than a per-T list, so a subset re-spaces it instead of reindexing
#: it (see :func:`respaced`).
PER_TIME_KEYS: Tuple[str, ...] = ("frame_time_jd",)


def time_subset(metadata: Mapping[str, Any], keep: Sequence[int]) -> Dict[str, Any]:
    """The ``{key: subset}`` changes that reindex every :data:`PER_TIME_KEYS` list in
    ``metadata`` onto the timepoints ``keep`` (already validated indices, in output order).

    The per-T twin of :func:`channel_subset` / :func:`position_subset`, with the same two
    rules: a key that is absent or not a list is left alone rather than invented, and a list
    too SHORT to cover an index in ``keep`` is dropped whole rather than silently shortened,
    because a partial positional list reports some other frame's time instead of admitting
    it does not know.
    """
    changes: Dict[str, Any] = {}
    for key in PER_TIME_KEYS:
        vals = metadata.get(key)
        if not isinstance(vals, (list, tuple)):
            continue
        changes[key] = ([vals[i] for i in keep] if all(0 <= i < len(vals) for i in keep)
                        else None)
    return changes


def respaced(value: Any, keep: Sequence[int]) -> Any:
    """A scalar axis SPACING (``dt_s``, ``z_step_um``) after the axis is subset to ``keep``.

    Three answers, and the middle one is the reason this is not just a pass-through:

    * a **contiguous** run (or a single index) keeps the spacing — nothing was skipped, so
      the interval between surviving neighbours is the source's own;
    * a **uniformly strided** run multiplies it — keeping every 3rd plane of a 0.5 µm stack
      really is a 1.5 µm stack, and a 3D measurement downstream reads this number to turn
      voxels into µm³. Leaving it at 0.5 would under-report every volume by 3×;
    * an **irregular** pick (planes 2, 5, 6) has no single spacing at all, so the key is
      DROPPED (``None`` → removed). Absent means "unknown", which a consumer can refuse or
      degrade on; a fabricated average would be believed.

    ``None`` in, ``None`` out: a file that never carried the spacing does not gain one.
    """
    if value is None or len(keep) < 2:
        return value
    steps = {int(keep[i + 1]) - int(keep[i]) for i in range(len(keep) - 1)}
    if len(steps) != 1:
        return None
    try:
        return float(value) * float(steps.pop())
    except (TypeError, ValueError):
        return None


def z_home_after(metadata: Mapping[str, Any], keep: Sequence[int]) -> Dict[str, Any]:
    """``{"z_home_index": …}`` for a Z axis subset to ``keep``, or ``{}`` if nothing to say.

    ``z_home_index`` names WHICH slice ``stage_z_um`` is the focus of
    (:data:`nodelab_v2.ingest.PLACEMENT_KEYS`), so it is an index into the z axis and a
    subset moves it: the home plane's new address is its position within ``keep``. If the
    home plane was cropped away there is no such position, and the key is dropped rather
    than left pointing at whichever plane inherited its old number — the same
    stale-positional-index failure :func:`position_subset` guards on M.
    """
    home = metadata.get("z_home_index")
    if home is None:
        return {}
    try:
        idx = int(home)
    except (TypeError, ValueError):
        return {}
    picks = [int(z) for z in keep]
    return {"z_home_index": picks.index(idx) if idx in picks else None}


def drop_position_keys(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``{key: None}`` changes that retire every per-M list **except** ``origin_um``.

    For a node that collapses M to a single output whose positions no longer exist
    separately — ``util.stitch``'s mosaic. Subsetting is wrong there: after a stitch there
    is no per-position stage coordinate to keep, only one canvas, and fabricating a
    single-entry ``stage_xy_um`` would state a field CENTRE for something no reader means
    by that.

    ``origin_um`` is excluded because it is the maintained key and the collapsing node
    restamps it itself (``metadata.stitch`` takes the union corner) — which is precisely
    why :func:`~nodegraph.placement.field_box` prefers it and keeps the stage log only as
    a fallback for a Dataset that never crossed the calibration seam.
    """
    return {k: None for k in PER_POSITION_KEYS if k != "origin_um"
            and metadata.get(k) is not None}


def resample(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Rescale: new size = old·scale; pixel size scales inversely (finer when
    upsampling). ``z`` scales only in 3D mode (stack-of-2D leaves z untouched)."""
    sxy = float(params.get("scale_xy", params.get("scale", 1.0)) or 1.0)
    sz = float(params.get("scale_z", 1.0) or 1.0)
    ax = env.axes
    is_3d = modes.get("dim") == "3D"
    new_axes = replace(ax, y=max(1, round(ax.y * sxy)), x=max(1, round(ax.x * sxy)),
                       z=(max(1, round(ax.z * sz)) if is_3d else ax.z))
    changes: Dict[str, Any] = {}
    px = env.metadata.get("pixel_size_um")
    if px is not None and sxy:
        changes["pixel_size_um"] = px / sxy
    zs = env.metadata.get("z_step_um")
    if is_3d and zs is not None and sz:
        changes["z_step_um"] = zs / sz
    return env.with_axes(new_axes).with_metadata(**changes)


def z_project(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Collapse Z→1; drop ``z_step_um`` and mark ``z_collapsed`` provenance so a
    downstream lever defaults to 2D and no metric reads a meaningless z step. A ``sum``
    projection also widens ``bit_depth`` (n_z summed samples), per :func:`bit_depth_after_sum`.

    ``method == "none"`` is the RESET (V2.21): the node hands its input through untouched,
    so the envelope is returned **verbatim** — Z survives at its full extent, ``z_step_um``
    survives, and no ``z_collapsed`` is stamped, which is what lets a downstream lever go
    back to defaulting 3D. It deliberately does not stamp ``z_collapsed=False``: absent and
    False are not the same claim, and rewriting the key would erase a genuine upstream
    collapse (a project → reset chain still happened). The literal is duplicated in
    ``util.zproject``'s compute, whose payload must agree with this prediction for every
    method — ``selftest::test_catalog_ported`` asserts the agreement per method."""
    if modes.get("method") == "none":
        return env
    widen = (bit_depth_after_sum(env, env.axes.z)
             if modes.get("method") == "sum" else {})
    return (env.with_axes(replace(env.axes, z=1))
               .with_metadata(z_step_um=None, z_collapsed=True, **widen))


def stack_time(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Temporal stack T→1; drop ``dt_s``. A ``sum`` combiner also widens ``bit_depth``
    (n_t summed samples) — the "12-bit in, 16-bit out" case."""
    widen = (bit_depth_after_sum(env, env.axes.t)
             if modes.get("method") == "sum" else {})
    return env.with_axes(replace(env.axes, t=1)).with_metadata(dt_s=None, **widen)


def frame_slice(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Per-frame-T slice (``zone.frame``): select one timepoint → T=1. The frame index
    picks *which* frame (a value, not geometry), so axes just collapse to t=1; ``dt_s``
    is kept (it still describes the source series' interval)."""
    return env.with_axes(replace(env.axes, t=1))


def parse_channels(raw) -> Optional[List[int]]:
    """The ``channels`` param → a list of channel indices, or ``None`` for "all".

    Shared by ``channel.select``'s compute and :func:`channel_select` below, which MUST
    agree: the meta_transform predicts the axes at edit time and the compute produces
    them at pull time, and a node whose payload disagrees with its envelope fails the
    build-node-v2 §2 gate. Accepts three forms because the param has three producers:
    a **list of ints** written by the GUI's per-channel tap materializer
    (``nodelab_v2/ops.py``), a **comma-separated string** typed into the socket by a user
    (``SocketType`` has no LIST member), and **empty/None** meaning every channel.
    Non-integer text is ignored rather than raising, so a half-typed "0," keeps the node
    previewing instead of erroring on each keystroke."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, str):
        out: List[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                continue
        return out or None
    items = list(raw)
    return items or None


def parse_indices(raw) -> Optional[List[int]]:
    """An index-list param → sorted unique indices, or ``None`` for "every index".

    The frame-axis counterpart of :func:`parse_channels`, and it accepts one thing that
    parser does not: a **RANGE**, ``"3-8"``, meaning 3 through 8 **inclusive**. Inclusive
    because that is what a human writing a range means, and the alternative — matching the
    exclusive ``y1``/``x1`` slice bounds on the same node — would make ``"0-0"`` select
    nothing. The two spellings compose (``"0-3,7,10-12"``), which is what makes one socket
    serve both "the range I want" and "the sparse set the strips picked".

    Deliberately total, like :func:`parse_channels`: a token that is not an index is
    skipped, and a half-typed ``"3-"`` reads as ``3`` rather than raising, so the node keeps
    previewing while somebody is still typing. Reversed (``"8-3"``) is read as the same span
    — an interval has no direction here, and the axes are always walked in acquisition
    order (:class:`~nodegraph.provider.FrameSubsetProvider` sorts).

    Negative values are simply out of range, not "from the end": ``-`` is the range
    separator, so a leading minus would be ambiguous, and :func:`frame_spec_picks` drops
    out-of-range indices anyway. A list of ints (what a GUI pick commits) passes through.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    out: List[int] = []
    if isinstance(raw, str):
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            lo_txt, sep, hi_txt = tok.partition("-")
            try:
                lo = int(lo_txt)
            except ValueError:
                continue
            if not sep:
                out.append(lo)
                continue
            try:
                hi = int(hi_txt)
            except ValueError:
                out.append(lo)                 # "3-" mid-type: the one index we do have
                continue
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
    else:
        for item in raw:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
    return sorted(set(out)) or None


def format_indices(values: Sequence[int]) -> str:
    """Sorted indices → the shortest index-list string that means them: ``"0-3,7,10-12"``.

    The inverse of :func:`parse_indices`, and it must round-trip through it exactly, which is
    the reason it lives here beside it rather than in the GUI layer that needs it: a picked
    param has to be indistinguishable from a typed one, and a formatter that drifted from the
    parser would produce a value the node then read as something else.

    Canonical: because the input is sorted and de-duplicated (:func:`frame_spec_picks`), one
    selection has exactly one spelling. That is what lets the result be used as sampling
    PROVENANCE, which is compared rather than merely displayed.
    """
    out: List[str] = []
    run: List[int] = []

    def flush() -> None:
        if run:
            out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")

    for v in values:
        if run and int(v) == run[-1] + 1:
            run.append(int(v))
            continue
        flush()
        run = [int(v)]
    flush()
    return ",".join(out)


#: The axes a frame spec may name, in the order :func:`format_frame_spec` writes them.
FRAME_SPEC_AXES: Tuple[str, ...] = ("m", "t", "z")


def parse_frame_spec(raw) -> Dict[str, List[int]]:
    """A frame spec — ``"m0-2,t3,z1-4"`` — → ``{axis: indices}``, absent axis = keep every one.

    ONE param for the whole selection, because the thing the user is selecting is one thing:
    "these frames". Three separate sockets said the same and made the reader assemble it.

    The grammar is whatever a person is likely to type, which means two spellings have to
    work and do: a token that STARTS with an axis letter opens that axis's list, and a token
    with no letter CONTINUES the axis most recently named. Commas and spaces separate tokens
    interchangeably, so ``"m0-2,t3,z1-4"`` and ``"m0,2 t3,7"`` both read correctly — in the
    first the comma divides axes, in the second it divides indices, and neither reading has to
    be guessed at. Within an axis the syntax is :func:`parse_indices`', ranges included.

    A leading token with no axis letter is read as **T**: this project's own vocabulary uses
    "frame" for a timepoint (``pick_kind="frame"`` is "the timepoint the viewer is showing",
    :func:`frame_slice` is the per-T slice), so bare ``"3"`` means timepoint 3 rather than
    quietly doing nothing, which is what a typed number that matched no axis would otherwise
    do. Repeating an axis extends it (``"t1 t5"`` is ``t={1,5}``) rather than replacing it,
    which is the reading that cannot silently discard something the user typed.

    Total, like :func:`parse_indices` and for the same reason: it runs on every keystroke, so
    ``"m"``, ``"m0-"`` and ``"q7"`` all resolve to something rather than raising. An axis
    letter with nothing after it yet contributes no indices, so it reads as "not asked for"
    until a number arrives — the node keeps previewing the whole series instead of flickering
    to an empty selection mid-word."""
    out: Dict[str, List[int]] = {}
    if raw is None:
        return out
    text = raw if isinstance(raw, str) else str(raw)
    axis = "t"
    for tok in text.replace(",", " ").split():
        head = tok[0].lower()
        if head.isalpha():
            if head not in FRAME_SPEC_AXES:
                continue                  # an axis this node cannot subset (c, y, x): ignored
            axis, tok = head, tok[1:]
        idx = parse_indices(tok)
        if idx:
            out.setdefault(axis, []).extend(idx)
    return {a: sorted(set(v)) for a, v in out.items()}


def format_frame_spec(picks: Mapping[str, Sequence[int]]) -> str:
    """``{axis: indices}`` → the canonical spec string ``"m0-2,t3,z1-4"``.

    The inverse of :func:`parse_frame_spec` and the thing a GUI pick commits, so it must
    round-trip exactly: a picked value has to be indistinguishable from a typed one, editable
    in place, and identical for the same selection every time (the sampling stamp compares
    it). Axes are written in :data:`FRAME_SPEC_AXES` order and empty ones omitted."""
    return ",".join(f"{a}{format_indices(sorted(set(picks[a])))}"
                    for a in FRAME_SPEC_AXES if picks.get(a))


def frame_spec_picks(raw, m: int, t: int, z: int
                     ) -> Dict[str, Optional[Tuple[int, ...]]]:
    """A frame spec resolved against real axis lengths: ``{axis: kept indices}``.

    Three distinct answers per axis, and the third is the one that matters:

    * ``None`` — that axis was not named, so every index is kept.
    * a non-empty tuple — the surviving indices, sorted, out-of-range ones dropped.
    * ``()`` — the axis WAS named and nothing survived (``"t9"`` on a 3-frame series).
      Kept distinct from ``None`` so the two halves of the node can differ in the one way
      they must: the compute REFUSES it (an empty axis is a degenerate payload that travels
      until something indexes into it — ``channel.select``'s worked example), while the
      advisory ``meta_transform`` holds the pre-edit envelope, because it re-runs on every
      keystroke and "t9" is seen while somebody types "t90".

    Shared by ``util.crop``'s frames mode and :func:`crop`, so the predicted extent and the
    produced extent cannot drift (build-node-v2 §2)."""
    spec = parse_frame_spec(raw)
    sizes = {"m": int(m), "t": int(t), "z": int(z)}
    return {a: (None if a not in spec
                else tuple(i for i in spec[a] if 0 <= i < sizes[a]))
            for a in FRAME_SPEC_AXES}


#: EVERY metadata key that is a **list indexed by channel**. A node that narrows or
#: reorders the channel axis must subset all of them together or the survivors stop
#: describing the channels that are left — and because they are read POSITIONALLY
#: (``names[c]``), a stale full-length list does not look stale, it looks like the wrong
#: channel. That was the ch1-tap bug (2026-08-03): only ``channel_emission_nm`` was
#: subset, so a tap on channel 1 kept ``channel_names == ["DAPI", "GFP"]`` with ``c == 1``
#: and every positional reader — the Viewer's channel strip, the card's socket labels, the
#: hover readout — reported it as "DAPI", the FIRST channel's name, on both branches.
#:
#: Only ``channel_emission_nm`` is calibration (:data:`~nodegraph.dataset.CALIBRATION_KEYS`);
#: the rest are the Viewer's per-channel display lists, seeded onto the payload by
#: :attr:`nodelab_v2.runner.EngineRunner._channel_display`. They travel together, so they
#: are subset together.
PER_CHANNEL_KEYS: Tuple[str, ...] = (
    "channel_emission_nm", "channel_names",
    "channel_excitation_nm", "channel_colors",
)


def channel_subset(metadata: Mapping[str, Any], keep: Sequence[int]) -> Dict[str, Any]:
    """The ``{key: subset}`` changes that reindex every :data:`PER_CHANNEL_KEYS` list in
    ``metadata`` onto the channels ``keep`` (already validated indices, in output order).

    SHARED by the ``channel_select`` meta_transform and ``channel.select``'s compute so the
    predicted envelope and the produced payload cannot drift (build-node-v2 §2) — the same
    contract :func:`parse_channels` has for the index list itself. A key that is absent, or
    not a list, is left alone rather than invented."""
    changes: Dict[str, Any] = {}
    for key in PER_CHANNEL_KEYS:
        vals = metadata.get(key)
        if isinstance(vals, (list, tuple)):
            changes[key] = [vals[i] for i in keep if i < len(vals)]
    return changes


def channel_select(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Subset/reindex channels; rewrite every per-channel list in lockstep.

    Out-of-range / negative indices are dropped FIRST, so the channel count and the
    per-channel metadata always agree (review #13: len(keep) could otherwise claim more
    channels than the source has)."""
    keep = parse_channels(params.get("channels"))
    if not keep:
        return env
    valid = [i for i in keep if 0 <= i < env.axes.c]     # lockstep count ↔ metadata
    if not valid:
        # A non-empty request that matches nothing is a user error the COMPUTE refuses
        # (``_compute_select_channel``). This transform must stay total — it re-runs on
        # every keystroke, so "0," and "1" are both seen while someone types "10" — hence
        # the pre-edit envelope is held rather than predicting a c=0 Dataset. Same division
        # of labour as ``crop``: the advisory transform degrades, the payload raises.
        return env
    new_axes = replace(env.axes, c=len(valid))
    return env.with_axes(new_axes).with_metadata(**channel_subset(env.metadata, valid))


def crop(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Change spatial extent; pixel size preserved (origin deferred, V2.00 §16).

    ``span`` mirrors the crop node's payload ``bound()`` EXACTLY (fill each missing
    endpoint independently — start→0, end→n — and clamp both into ``[0, n]``), so the
    predicted (header) extent equals the realized payload extent for one-sided and
    out-of-range crops alike (adversarial review 2026-07-21). A ``max(1, …)`` floor
    keeps the advisory transform crash-free where the payload would raise on an empty
    region.

    ``region == "frames"`` is the other half of the node (V2.27) and goes to
    :func:`crop_frames`: the same node narrows M/T/Z by index instead of cutting a window
    out of a plane. The branch is on the MODE rather than on which params are set, so a
    graph carrying stale bounds from the other mode is unaffected by them — exactly what the
    sockets' ``available_in`` gating promises the user."""
    if (modes or {}).get("region") == "frames":
        return crop_frames(env, params, modes)
    ax = env.axes

    def span(a, b, n):
        lo = max(0, min(int(a), n)) if a is not None else 0
        hi = max(0, min(int(b), n)) if b is not None else n
        return max(1, hi - lo)

    new_axes = replace(
        ax, y=span(params.get("y0"), params.get("y1"), ax.y),
        x=span(params.get("x0"), params.get("x1"), ax.x),
        z=(span(params.get("z0"), params.get("z1"), ax.z)
           if modes.get("dim") == "3D" else ax.z))

    # The origin MOVES by the cut — this is the transform the "origin deferred" note in
    # this docstring was waiting for. `lo` mirrors `span`'s clamping exactly (same
    # start-fill, same range clamp) so the predicted corner matches the payload's for
    # one-sided and out-of-range crops alike, and the offset is in µm via the INPUT's
    # sampling (a crop does not change pixel size, so no ordering subtlety arises).
    def lo(a, n):
        return max(0, min(int(a), n)) if a is not None else 0

    px = env.metadata.get("pixel_size_um")
    zs = env.metadata.get("z_step_um")
    dy = lo(params.get("y0"), ax.y) * float(px) if px else 0.0
    dx = lo(params.get("x0"), ax.x) * float(px) if px else 0.0
    dz = (lo(params.get("z0"), ax.z) * float(zs)
          if (zs and modes.get("dim") == "3D") else 0.0)
    return env.with_axes(new_axes).with_metadata(**shift_origin_um(env, dz, dy, dx))


def crop_frames(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.crop``'s frames mode: narrow M / T / Z to the picked indices (V2.27).

    Not a window but a **subset** — the picks may be sparse, because that is what the
    viewer's M/T/Z strips produce and a span would silently re-admit the frames between the
    ones somebody ticked. Every axis is independent and the result is their cross product,
    the same reading :class:`~nodegraph.provider.FrameSubsetProvider` already implements for
    the run scope.

    Four kinds of metadata move with it, and every one of them is a positional-staleness
    trap of the kind that does not LOOK stale:

    * per-M lists (:func:`position_subset`) — a survivor at full length reports another
      field's stage coordinate, which reads as a handedness bug;
    * per-T lists (:func:`time_subset`) — ``frame_time_jd`` is the only clock two files
      share, so a stale one mis-pairs a merge;
    * the axis SPACINGS ``dt_s`` / ``z_step_um`` (:func:`respaced`) — a strided pick really
      does re-space the axis, and this number is what a 3D measurement multiplies by;
    * ``z_home_index`` (:func:`z_home_after`) and ``origin_um`` — both name a position on
      the z axis, so cutting planes off the bottom moves them.

    ``pixel_size_um`` and the lateral extent are untouched: nothing is cut out of a plane
    here. Total, like every transform: an unparseable or fully-out-of-range request holds the
    pre-edit envelope and lets the payload raise the real message
    (:func:`frame_spec_picks`).
    """
    ax = env.axes
    picks = frame_spec_picks(params.get("frames"), ax.m, ax.t, ax.z)
    ms, ts, zs = picks["m"], picks["t"], picks["z"]
    if ms == () or ts == () or zs == ():
        return env
    keep_m = ms if ms is not None else tuple(range(ax.m))
    keep_t = ts if ts is not None else tuple(range(ax.t))
    keep_z = zs if zs is not None else tuple(range(ax.z))
    new_axes = replace(ax, m=len(keep_m), t=len(keep_t), z=len(keep_z))

    changes: Dict[str, Any] = {}
    if ms is not None:
        changes.update(position_subset(env.metadata, keep_m))
    if ts is not None:
        changes.update(time_subset(env.metadata, keep_t))
        changes["dt_s"] = respaced(env.metadata.get("dt_s"), keep_t)
    z_step = env.metadata.get("z_step_um")
    if zs is not None:
        changes["z_step_um"] = respaced(z_step, keep_z)
        changes.update(z_home_after(env.metadata, keep_z))
    # The z shift uses the SOURCE spacing, deliberately: `origin_um` is where plane 0 sits,
    # so the corner moves by however many real planes were dropped below the first kept one.
    # Computing it from the RESPACED step would scale the shift by the stride as well.
    out = env.with_axes(new_axes).with_metadata(**changes)
    if zs is not None and z_step and keep_z and keep_z[0]:
        try:
            dz = float(z_step) * int(keep_z[0])
        except (TypeError, ValueError):
            dz = 0.0
        if dz:
            # Applied to `out`, not `env`: `read_origin_um` validates the list against
            # `axes.m`, and after an M subset only the already-narrowed list on the
            # already-narrowed axes passes that check.
            out = out.with_metadata(**shift_origin_um(out, dz, 0.0, 0.0))
    return out


def stitch(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """Tile stitch: M→1, Y/X grow. The output extent is UNKNOWN unless supplied
    (it depends on estimated registration) — never a silent guess (V2.03 §2 A3)."""
    ax = env.axes
    ny, nx = params.get("out_y"), params.get("out_x")
    unknown = set()
    if not ny:
        unknown.add("y")
    if not nx:
        unknown.add("x")
    new_axes = replace(ax, m=1, y=int(ny) if ny else ax.y, x=int(nx) if nx else ax.x)
    # M→1, so the mosaic's single origin is the UNION corner: the minimum over the tiles
    # that went into it, which is exactly what `_stitch_normalize` places at canvas (0, 0).
    origins = read_origin_um(env)
    changes: Dict[str, Any] = {}
    if origins:
        changes["origin_um"] = [[min(o[0] for o in origins),
                                 min(o[1] for o in origins),
                                 min(o[2] for o in origins)]]
    # The other per-M lists are RETIRED, in lockstep with the compute's own
    # `drop_position_keys` call (build-node-v2 §2 — the predicted envelope and the produced
    # payload may not drift). M→1 means those positions no longer exist separately, and a
    # positional list that outlives its axis reports another field's coordinate.
    changes.update(drop_position_keys(env.metadata))
    return env.with_axes(new_axes, unknown=frozenset(unknown)).with_metadata(**changes)


def overlay(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``view.overlay``: identity in ``display`` mode, **C+1** in ``resample`` mode.

    Exactly ONE channel, and that is a design consequence rather than a simplification. A
    ``meta_transform`` is handed only the PRIMARY edge's envelope (``propagate_meta`` reads
    ``dataset_preds[0]``), so it cannot see how many channels the secondary has — and an
    axis this pass cannot predict would have to be marked UNKNOWN, the way ``stitch`` marks
    its extent. Baking one user-chosen channel instead keeps the prediction exact, which is
    what the whole edit-time widget re-seed depends on, and it is also the useful shape: you
    co-register the channel you are going to measure against, not the whole other file.

    ``bit_depth`` is DROPPED. The output stacks two files' intensity scales, and this pass
    cannot see the second one to check whether they agree, so asserting the primary's depth
    over the pair would be a claim about data it never read. Absent is the honest signal
    (§7c) and every consumer already handles it.

    ``channel_names`` grows in lockstep so the Viewer's channel strip and any name-based
    consumer stay correct; the real name is filled in by the compute, which CAN see the
    secondary, and this pass supplies a placeholder of the right LENGTH — the count is what
    downstream logic indexes by.
    """
    if (modes or {}).get("output") != "resample":
        return env
    ax = env.axes
    names = list(env.metadata.get("channel_names") or [])
    while len(names) < ax.c:
        names.append(f"Ch{len(names) + 1}")
    names.append("overlay")
    return (env.with_axes(replace(ax, c=ax.c + 1))
               .with_metadata(bit_depth=None, channel_names=names))


def merge_channels(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``channel.merge``: C and Z both grow, and both are UNKNOWN.

    This pass is handed only the PRIMARY edge's envelope (``propagate_meta`` reads
    ``dataset_preds[0]``), so it cannot see the second file at all — and both changed axes are
    functions of it:

    * ``c`` grows by however many channels the secondary has;
    * ``z`` becomes the merged grid, whose plane count depends on the secondary's focus range
      and step (:func:`nodegraph.placement.merge_z_grid`).

    ``view.overlay``'s ``resample`` mode solves the same blindness by baking exactly ONE channel,
    which keeps its prediction exact. That is the right trade for an overlay you are going to
    *measure one channel of*, and the wrong one here: this node exists to put both files on one
    axis, so the honest answer is ``stitch``'s — mark the axes unknown rather than guess
    (V2.03 §2 A3). The GUI shows "?" for them until the first pull, which is true.

    ``bit_depth`` is dropped for the same reason it is under ``resample``: the output stacks two
    files' intensity scales and this pass has never seen the second one.

    ``z_step_um`` is dropped too, and that one matters more than it looks. The merged grid's step
    is the finer of the two files', which this pass cannot compute — and leaving the primary's
    step standing would have every downstream µm→plane conversion silently using the wrong
    spacing. Absent is the signal that it must be re-read from the payload.
    """
    ax = env.axes
    names = list(env.metadata.get("channel_names") or [])
    while len(names) < ax.c:
        names.append(f"Ch{len(names) + 1}")
    return (env.with_axes(replace(ax, c=ax.c + 1), unknown=frozenset({"c", "z"}))
               .with_metadata(bit_depth=None, z_step_um=None,
                              channel_names=names + ["merged"]))


def zs_deconvnet(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``enhance.zs_deconvnet``: BOTH kinds of restamp at once, conditionally.

    * **Intensity.** Always drops ``bit_depth``, for the reason :func:`value_rescaled`
      exists: the network is fed percentile-normalized ``[0,1]`` input and its output is
      percentile-normalized again, so the result is not integer counts on any scale — and
      the ``relu`` output layer plus a deconvolution's flux concentration means it has no
      predictable ceiling either, so there is no depth to widen to (the same call
      ``enhance.deconvolve`` makes).

    * **Geometry.** When ``upsample`` is on, the deconvolution head ends in an
      ``UpSampling2D((2,2))`` / ``UpSampling3D((2,2,1))``, so Y and X double and
      ``pixel_size_um`` halves. **Z is never scaled** — the 3D upsampling is deliberately
      lateral-only, because an already-coarse axial axis gains nothing from interpolation —
      so ``z_step_um`` is left exactly alone. That asymmetry is the whole reason this cannot
      reuse :func:`resample`, whose ``scale_z`` would touch it.

    The ``denoised`` output is NOT upsampled even when ``upsample`` is on: stage I is the
    denoiser and runs at the input grid (its head is cropped, never upscaled). So the
    geometry half is conditional on the OUTPUT mode as well as on ``upsample`` — pick
    ``denoised`` and this is a pure intensity transform.

    **``upsample`` may come from the CHECKPOINT** (V2.23). In ``pretrained`` mode an unset
    socket takes the value recorded in the sidecar beside ``weights_path``, because a 2x head
    that disagrees with the trained graph loads silently and predicts nonsense. That makes the
    adopted value part of this prediction, not just of the compute: the axes here must equal
    the axes the payload gets, and they would diverge the moment one of the two consulted the
    sidecar and the other did not. Both call :func:`nodegraph.trained.zs_trained`, so there is
    one answer rather than two that have to be kept in step by hand.
    """
    from nodegraph.trained import zs_trained as _zs_trained
    out = env.with_metadata(bit_depth=None)
    upsample = params.get("upsample")
    if upsample in (None, ""):
        upsample = _zs_trained(params, modes).get("upsample", True)
    if upsample in (False, 0, "0", "false", "False"):
        return out
    if (modes or {}).get("output") == "denoised":
        return out
    ax = out.axes
    changes: Dict[str, Any] = {}
    px = out.metadata.get("pixel_size_um")
    if px is not None:
        changes["pixel_size_um"] = px / 2.0
    return out.with_axes(replace(ax, y=max(1, ax.y * 2), x=max(1, ax.x * 2))) \
              .with_metadata(**changes)


META_TRANSFORMS: Dict[str, MetaTransform] = {
    "overlay": overlay, "merge_channels": merge_channels,
    "identity": identity, "resample": resample, "z_project": z_project,
    "stack_time": stack_time, "frame_slice": frame_slice,
    "channel_select": channel_select, "crop": crop, "stitch": stitch,
    "value_rescaled": value_rescaled, "flatten_field": flatten_field,
    "zs_deconvnet": zs_deconvnet, "subtract_background": subtract_background,
}


def named_meta_transform(name: str) -> Optional[MetaTransform]:
    return META_TRANSFORMS.get(name)


# ── the forward pass ──────────────────────────────────────────────────────────

def propagate_meta(graph: Graph,
                   seeds: Optional[Mapping[str, MetaEnvelope]] = None
                   ) -> Dict[str, MetaEnvelope]:
    """Compute every node's output :class:`MetaEnvelope` by a forward topological
    walk (V2.03 §2 A3). A node's input envelope is its first DATASET-input
    predecessor's output; a root uses ``seeds[node_id]`` (a source declaring its
    file/provider metadata) or an empty envelope. Each node's declared
    ``meta_transform`` (identity by default) produces its output. Pixel-free."""
    seeds = seeds or {}
    out: Dict[str, MetaEnvelope] = {}
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        spec = node.spec()
        dpreds = graph.dataset_preds(nid)
        env_in = out.get(dpreds[0].src, MetaEnvelope()) if dpreds \
            else seeds.get(nid, MetaEnvelope())
        transform = spec.meta_transform if spec is not None else None
        env_out = env_in if transform is None else transform(
            env_in, node.params, node.state(spec))
        # Domain accumulation: union EVERY Dataset predecessor's domain-set (a merge
        # node combines them), then add what this node produces. A root seeds from its
        # own envelope's domain-set. The meta_transform never touches domains, so this
        # is layered on afterward. (V2.06: the socket domain-rail + wire-tint source.)
        if dpreds:
            dom_in: FrozenSet[Domain] = frozenset().union(
                *(out.get(e.src, MetaEnvelope()).domains for e in dpreds))
        else:
            dom_in = env_in.domains
        adds = spec.adds_domains if spec is not None else frozenset()
        env_out = env_out.with_domains(dom_in | adds)
        out[nid] = env_out.with_layer_names(
            _layer_names_out(spec, node, env_in, env_out))
    return out


def _layer_names_out(spec, node, env_in: MetaEnvelope,
                     env_out: MetaEnvelope) -> Tuple[Tuple[Domain, str], ...]:
    """This node's outgoing layer catalog: the input's, minus what its axis change
    invalidates, plus what it writes (V2.11).

    **Total by contract** — this runs inside ``propagate_meta``, which the GUI calls on
    every keystroke, and whose caller catches only ``ValueError``: anything else escapes
    and takes the window down, while even a caught error blanks EVERY node's envelope
    (domain rails and derived spinboxes graph-wide). So every step is defensive, exactly
    like the meta_transforms above ("missing keys degrade to no-op, never a crash")."""
    names: List[Tuple[Domain, str]] = list(env_in.layer_names)

    # ── DROP: the catalog is NOT monotone ────────────────────────────────────────
    # `Dataset.reshaped_axes(drop_stale=True)` silently discards any LATTICE layer whose
    # array no longer matches the new axes, and five nodes rely on it (channel.select,
    # util.zproject, util.crop, util.resample, util.stack). Its rule is exactly "the
    # shape for this domain changed", and a domain's shape is built from `axes_of` — so
    # comparing the envelope's own before/after axes reproduces it for the whole catalog
    # with no per-node declaration. Structure domains are never reshaped, so they survive.
    try:
        changed = {ax for ax in AXIS_ORDER
                   if getattr(env_in.axes, ax, None) != getattr(env_out.axes, ax, None)}
    except Exception:                                    # pragma: no cover - defensive
        changed = set()
    if changed:
        names = [(d, n) for d, n in names
                 if not (is_lattice(d) and (axes_of(d) & changed))]

    # ── ADD: what this node creates ──────────────────────────────────────────────
    if spec is None:
        return tuple(dict.fromkeys(names))
    try:
        state = node.state(spec)
    except Exception:                                    # pragma: no cover - defensive
        state = {}
    params = getattr(node, "params", {}) or {}
    for sock in spec.inputs:
        if not sock.layer_out:
            continue
        try:
            if not sock.active_in(state):
                continue
            # ONE resolver, shared with `EvalContext.layer` — the compute and this
            # prediction must never disagree about which layer a socket denotes.
            value = layer_value(sock, params)
            if not value:
                continue
            for dom in sock.layer_out:
                names.append((dom, value))
        except Exception:                                # pragma: no cover - defensive
            continue
    extra = getattr(spec, "extra_layers", None)
    if extra is not None:
        try:
            for dom, nm in extra(params, state) or ():
                if isinstance(nm, str) and nm:
                    names.append((dom, nm))
        except Exception:                                # pragma: no cover - defensive
            pass
    return tuple(dict.fromkeys(names))                   # de-dup, keep first appearance


# ── derive-symbol source + the metadata-intelligent lever default ─────────────
#
# The SOURCE layer of directive A (V2.03 §2 A5): the symbols a ``derive`` may read,
# sourced from the resolved incoming envelope (per-edge) — NOT from static file
# metadata. Mirrors the V1.91 leaf contract without importing the nd2-coupled
# ``metadata_adapt`` (nodegraph stays nd2-free); the app/ctx layer feeds these
# symbols to ``metadata_adapt.adapt_defaults`` where the pipeline_kit path needs it.

_SAFE_FUNCS: Dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "round": round,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi,
}


def envelope_symbols(env: MetaEnvelope, channel_index: int = 0) -> Dict[str, Any]:
    """Derive symbols from an envelope (axes → counts; calibration → optics). Optics
    are for ``channel_index`` (the node's active channel — V2.03 §2 A6)."""
    md = env.metadata or {}
    emis = md.get("channel_emission_nm")
    emission = (emis[channel_index]
                if isinstance(emis, (list, tuple)) and 0 <= channel_index < len(emis)
                else (emis if not isinstance(emis, (list, tuple)) else None))
    return {
        "pixel_size_um": md.get("pixel_size_um"),
        "z_step_um": md.get("z_step_um"),
        "dt_s": md.get("dt_s"),
        "bit_depth": md.get("bit_depth"),      # significant sensor depth (12 on most ND2s)
        "emission_nm": emission,
        "na": md.get("objective_na"),
        "mag": md.get("objective_magnification"),
        "n_m": env.axes.m, "n_t": env.axes.t, "n_z": env.axes.z, "n_c": env.axes.c,
        "z_collapsed": bool(md.get("z_collapsed", False)),
        "is_3d": env.axes.is_volumetric,
    }


def eval_derive(expr: str, symbols: Mapping[str, Any]) -> Any:
    """Evaluate a trusted, pure-arithmetic ``derive`` expression (empty builtins).
    Mirrors the V1.91 ``metadata_adapt`` leaf contract (V2.03 §2 A5)."""
    ns = dict(_SAFE_FUNCS)
    ns.update(symbols)
    return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 — trusted, no builtins


def resolve_dim_default(spec: Any, env: MetaEnvelope) -> Optional[str]:
    """The metadata-intelligent 2D/3D lever default for ``spec`` given the incoming
    envelope (V2.03 §3 B4): evaluate the lever's ``derive`` (``z>1 ⇒ 3D``) against
    the envelope symbols; fall back to the static default on any failure. ``None``
    if the node bears no lever."""
    lever = spec.dim_lever() if hasattr(spec, "dim_lever") else None
    if lever is None:
        return None
    if not lever.derive:
        return lever.resolved_default()
    try:
        val = eval_derive(lever.derive, envelope_symbols(env))
    except Exception:  # noqa: BLE001 — a bad expression degrades to the static default
        return lever.resolved_default()
    return str(val) if val in ("2D", "3D") else lever.resolved_default()


__all__ = [
    "MetaEnvelope", "MetaTransform", "META_TRANSFORMS", "named_meta_transform",
    "identity", "resample", "z_project", "stack_time", "frame_slice",
    "channel_select", "crop", "stitch", "value_rescaled", "bit_depth_after_sum",
    "propagate_meta", "envelope_symbols", "parse_channels",
    "parse_indices", "format_indices", "crop_frames",
    "FRAME_SPEC_AXES", "parse_frame_spec", "format_frame_spec", "frame_spec_picks",
    "PER_CHANNEL_KEYS", "channel_subset",
    "PER_POSITION_KEYS", "position_subset", "drop_position_keys",
    "PER_TIME_KEYS", "time_subset", "respaced", "z_home_after",
    "eval_derive", "resolve_dim_default",
]
