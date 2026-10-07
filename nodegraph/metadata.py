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
from nodegraph.file_sequence import order as _seq_order
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
    #: The **column catalog** (V2.28): the ``(domain, layer, column)`` triples a Dataset on
    #: this edge is expected to carry on its STRUCTURE tables, in first-appearance order.
    #: Drives the GUI's column picker — a ``column_in`` socket offers the columns actually
    #: measured upstream instead of making the user retype ``mean_intensity`` and discover
    #: at pull time that nothing wrote it.
    #:
    #: Distinct from :attr:`layer_names`, which stops at the layer. A layer name says a
    #: Label table called ``CELLS`` exists; it cannot say whether anyone has measured its
    #: eccentricity yet, and that is exactly the question a per-object condition asks.
    #:
    #: **CLOSED, and therefore complete.** Every node that adds a structure domain
    #: declares ``adds_columns``, enforced by
    #: ``selftest::test_column_catalog_complete`` — so what a table carries is fully
    #: determined by the nodes upstream, and a ``column_in`` socket can render as a
    #: non-editable dropdown rather than a text box with hints. That is the whole point:
    #: the user picks from what the graph measured instead of recalling a name and
    #: learning at pull time that nothing wrote it.
    #:
    #: The obligation runs the other way from the layer catalog's. Because there is no
    #: free-text escape, a producer that declares nothing makes its columns
    #: **unpickable**, not merely unsuggested — which is why the completeness gate
    #: exists and why a new structure producer fails the build until it declares.
    column_names: Tuple[Tuple[Domain, str, str], ...] = ()

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

    def with_column_names(
            self, names: Sequence[Tuple[Domain, str, str]]) -> "MetaEnvelope":
        return replace(self, column_names=tuple(names))

    def columns_in(self, domain: Domain,
                   layer: Optional[str] = None) -> Tuple[str, ...]:
        """The column names known to be on this edge for *domain*, in first-appearance
        order (the GUI column picker's suggestion list).

        ``layer=None`` unions every layer in the domain, which is what a socket whose
        layer is inferred (§4g) has to offer; naming one narrows to it. De-duplicated,
        because two producers writing the same column onto one table (``analysis.measure``
        run twice with different stats) is ordinary, not a conflict."""
        return tuple(dict.fromkeys(
            c for d, lyr, c in self.column_names
            if d is domain and (layer is None or lyr == layer)))

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

    Axis-preserving either way — only the meaning of the numbers can change.

    ``approach="zero_regions"`` (2026-10-02) never changes the scale — a pixel is either
    itself or nothing — so the key survives there whatever ``combine`` still holds behind
    its hidden dropdown. Checked FIRST for that reason: a hidden Mode keeps its value."""
    m = modes or {}
    if str(m.get("approach") or "estimate_surface") == "zero_regions":
        return env
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
#: * ``source_file`` (2026-09-08) — WHICH FILE position ``m`` came from, one name per M.
#:   A single-file load leaves it absent; a **file bundle** stamps it, because a bundle
#:   concatenates K files along M (:class:`~nodegraph.provider.MultiSourceProvider`) and
#:   from that point on ``m`` is the only thing distinguishing one file's data from
#:   another's. It belongs in this family rather than beside the calibration because it is
#:   provenance, not geometry — nothing computes from it — but it is read POSITIONALLY like
#:   every other member, so it fails the same way: a stale list does not report "unknown
#:   file", it reports the WRONG file, onto a spreadsheet row that looks authoritative.
#:   Membership here is what makes it survive a crop correctly and retire on a stitch,
#:   without ``analysis.measure`` or the exporter knowing bundles exist.
#:
#: * ``position_group`` / ``position_name`` (2026-09-15) — WHICH SPECIMEN position ``m``
#:   belongs to, and what the acquisition called it. A multipoint axis is very often not one
#:   flat list of fields but several mosaics with a millimetre between them
#:   (:func:`~nodegraph.placement.position_groups`), and these two keys are where that
#:   recovered structure rides. Both are provenance rather than geometry — nothing computes
#:   a coordinate from them — but they are read POSITIONALLY like every other member here,
#:   and a stale one is the sharpest failure in the family: it does not report "unknown
#:   group", it assigns a field to the WRONG SPECIMEN, which is a conclusion about an
#:   experiment rather than a pixel. Membership here is what makes ``util.select_group``'s
#:   output describe the positions it actually kept.
#:
#: ``frame_time_jd`` is deliberately ABSENT: it is indexed by T, not M.
PER_POSITION_KEYS: Tuple[str, ...] = (
    "origin_um", "stage_xy_um", "stage_z_um", "__align_um__", "align_to_ncc",
    "source_file", "position_group", "position_name",
)

#: The per-M metadata key naming each position's GROUP — the specimen/mosaic it belongs to
#: (see :data:`PER_POSITION_KEYS`). Named once here so the detector, the node, the GUI loader
#: and the sidecar cannot disagree about the spelling.
#:
#: Values are group KEYS (``"G1"``, or whatever the user renamed the group to), not indices.
#: A key survives a crop that drops whole groups; an index would silently renumber, so
#: "group 2" would mean a different specimen before and after — and nothing downstream could
#: tell. The keys are also what a measurement table carries out to a spreadsheet, where
#: ``G3`` is legible and ``2`` is not.
POSITION_GROUP_KEY = "position_group"

#: The per-M metadata key naming each position as the ACQUISITION named it — NIS's point
#: names (``"#1"``…``"#9"``), or whatever the user typed into the point list.
#:
#: Worth carrying because it is a second, independent witness to the grouping: on the CRC
#: file the names restart at ``#1`` six times, which is the microscope's own record of where
#: one specimen ended and the next began. :func:`~nodegraph.placement.position_groups`
#: recovers the same six boundaries from geometry alone, so the two can be checked against
#: each other — and when they disagree, that is a fact about the acquisition worth surfacing
#: rather than a tie to break silently.
POSITION_NAME_KEY = "position_name"

#: The per-M metadata key naming each position's source file — the file bundle's identity
#: (see :data:`PER_POSITION_KEYS`). Named once here so the engine, the exporter and the GUI
#: cannot disagree about the spelling.
SOURCE_FILE_KEY = "source_file"

#: The scalar metadata key naming the experimental CONDITION a Dataset belongs to — stamped by
#: a Page Output (V4.00: blank = its page's name) and read by ``table.concat``, which writes
#: it as a column so rows from several pages say where they came from. Named once here so the
#: GUI-layer op and the catalog cannot disagree about the spelling.
CONDITION_KEY = "condition"

#: Set (``True``) beside :data:`CONDITION_KEY` when the condition was TYPED on a Page Output
#: rather than filled in from the page's name. A later Output left blank keeps a typed
#: condition instead of replacing it with its own page's name — so labelling the dishes once,
#: on the first page, survives every page after it.
CONDITION_SET_KEY = "condition_set"


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


def source_file_runs(metadata: Mapping[str, Any], m: int) -> Optional[List[Tuple[str, int, int]]]:
    """The per-M :data:`SOURCE_FILE_KEY` list collapsed into ``(name, start, count)`` runs
    — one per source FILE — or ``None`` when the list cannot be trusted to say.

    A **file bundle** lays K files end to end on ``m``
    (:class:`~nodegraph.provider.MultiSourceProvider`), so a file contributing ``n``
    positions appears as ``n`` consecutive identical labels. Collapsing them recovers the
    file boundaries that the flat ``m`` axis erased — which is the whole input
    ``util.chain`` re-addresses, and the reason that node needs no "how many files?" param.

    ``None`` (rather than a one-run guess) when the key is absent, is not a list, or is not
    exactly ``m`` long: the same rule :func:`position_subset` and :func:`read_origin_um`
    follow. A positional list of the wrong length does not report "unknown file", it
    reports the WRONG file, and a chain built on it would interleave two acquisitions.

    Runs are **consecutive only** — two non-adjacent runs of one name stay two runs. They
    cannot arise from a bundle (each file is contiguous by construction), and merging them
    would silently reorder positions to make them so.
    """
    vals = metadata.get(SOURCE_FILE_KEY)
    if not isinstance(vals, (list, tuple)) or len(vals) != int(m):
        return None
    out: List[Tuple[str, int, int]] = []
    for i, raw in enumerate(vals):
        name = str(raw)
        if out and out[-1][0] == name and out[-1][1] + out[-1][2] == i:
            name0, start, count = out[-1]
            out[-1] = (name0, start, count + 1)
        else:
            out.append((name, i, 1))
    return out


def stamp_source_file(env: "MetaEnvelope", path: str) -> "MetaEnvelope":
    """``env`` with :data:`SOURCE_FILE_KEY` naming ``path`` for every one of its positions.

    The single-file twin of :func:`nodelab_v2.runner.bundle_envelope`'s own stamp, and the
    reason it exists: until V3.02 only a BUNDLE said which file a position came from, so a
    Dataset loaded from its own card carried its filename nowhere at all — the path lived
    in the node's params, which no compute can see. ``util.chain`` orders files by the
    counting number in their names, and with separately wired cards it had nothing to read.

    Stamped in one place, called from both the pull-time envelope and the GUI's edit-time
    seed, because the two disagreeing about a positional list is the whole failure mode
    :data:`PER_POSITION_KEYS` documents.

    The visible consequence, accepted deliberately: a measurement table from a single-file
    graph now carries a ``file`` column too, where before only a bundle's did. It names the
    file the rows came from, which was never wrong to say — it was only ever absent.
    """
    import os

    name = os.path.basename(str(path)) or str(path)
    if not name:
        return env
    return env.with_metadata(**{SOURCE_FILE_KEY: [name] * max(1, int(env.axes.m))})


#: EVERY metadata key that is a **list indexed by timepoint**. The per-T member of the same
#: family as :data:`PER_CHANNEL_KEYS` and :data:`PER_POSITION_KEYS`, and it exists for the
#: third time for the same reason: ``frame_time_jd[t]`` is read POSITIONALLY, so a
#: full-length list left behind by a node that narrowed T does not look stale — it looks
#: like the wrong TIME. That one is the sharpest of the three, because
#: :func:`~nodegraph.placement.paired_t` uses it as the only clock two files share: a stale
#: list silently pairs frame 0 of one acquisition against frame 0 of the other's ORIGINAL
#: numbering, and channel.merge then reads two different moments as one.
#:
#: Two members: the clock in Julian days and its readable twin ``frame_datetime``
#: (``YYYY-MM-DD HH:MM:SS.mmm``, 2026-10-02), which must reindex together or a frame would
#: show one time and pair by another. ``frame_timestamps_s`` is deliberately absent: it
#: never reaches a Dataset (:data:`nodelab_v2.ingest.PLACEMENT_KEYS` does not carry it),
#: and ``dt_s`` is a scalar INTERVAL rather than a per-T list, so a subset re-spaces it
#: instead of reindexing it (see :func:`respaced`).
PER_TIME_KEYS: Tuple[str, ...] = ("frame_time_jd", "frame_datetime")


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


#: The per-B metadata key naming each batch member's file — the batch's identity, and the
#: exact twin of :data:`SOURCE_FILE_KEY` one axis out (V3.01). ``util.batch`` stamps it and
#: nothing else writes it, which is what lets ``util.unbatch`` know how many members it has
#: and what to CALL them without the member count having to travel as a param.
#:
#: It is deliberately a separate key from ``source_file`` rather than a reuse: a batched
#: bundle has both, and they answer different questions — ``source_file`` says which file
#: position ``m`` came from (within one member), ``batch_file`` says which file member
#: ``b`` IS. Collapsing them would make a batch of bundles unreadable.
BATCH_FILE_KEY = "batch_file"

#: Metadata indexed POSITIONALLY by the batch axis — the per-B family, mirroring
#: :data:`PER_POSITION_KEYS` and :data:`PER_TIME_KEYS`, and failing the same way if left
#: stale: a wrong-length list reports the WRONG file's name rather than admitting it does
#: not know. One member today.
PER_BATCH_KEYS: Tuple[str, ...] = (BATCH_FILE_KEY,)


def batch_subset(metadata: Mapping[str, Any], keep: Sequence[int]) -> Dict[str, Any]:
    """The ``{key: subset}`` changes that reindex every :data:`PER_BATCH_KEYS` list onto
    the batch members ``keep`` — what ``util.select_batch`` applies when it narrows ``b``.

    The per-B twin of :func:`position_subset` / :func:`time_subset`, with the same two
    rules: absent or non-list is left alone rather than invented, and a list too short to
    cover ``keep`` is dropped whole rather than shortened, because a partial positional
    list names some other file instead of admitting it does not know.
    """
    changes: Dict[str, Any] = {}
    for key in PER_BATCH_KEYS:
        vals = metadata.get(key)
        if not isinstance(vals, (list, tuple)):
            continue
        changes[key] = ([vals[i] for i in keep] if all(0 <= i < len(vals) for i in keep)
                        else None)
    return changes


def batch_grow(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.batch``: stacks its inputs on a NEW batch axis (V3.01).

    Handed only INPUT 0's envelope (``propagate_meta`` reads ``dataset_preds[0]``), exactly
    like :func:`merge_grow`, so the member COUNT is not visible here — ``b`` is marked
    UNKNOWN rather than guessed, and the GUI shows ``?`` until a pull resolves it. Every
    other axis is already correct for the result, because the node refuses a member whose
    grid differs (:class:`~nodegraph.provider.BatchProvider`), so nothing else changes.

    The per-member ``batch_file`` list is likewise a pull-time stamp, not a prediction: it
    names files this envelope cannot see.
    """
    return env.with_axes(env.axes, unknown=env.unknown_axes | frozenset({"b"}))


def batch_select(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.select_batch``: narrows ``b`` to one member, or leaves it alone.

    An EMPTY ``member`` is the no-op the compute performs, so the envelope must say so too
    — predicting ``b == 1`` for an unconfigured node would grey out controls and mis-size
    the viewer for a Dataset the pull hands back untouched.

    Which member is selected does not change the geometry, so the name is not read here:
    every member of a batch shares (m,t,z,c,y,x) by the batch node's own refuse-on-mismatch
    contract. Only the extent of ``b`` changes, and it becomes exactly 1.
    """
    want = str((params or {}).get("member") or "").strip()
    if not want:
        return env
    ax = env.axes
    return env.with_axes(replace(ax, b=1),
                         unknown=env.unknown_axes - frozenset({"b"}))


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


def position_group_plan(metadata: Mapping[str, Any], axes: Any,
                        gap_factor: Optional[float] = None):
    """The :class:`~nodegraph.placement.GroupPlan` for ``metadata``/``axes``.

    Two sources, and the order between them is the whole point:

    1. a **stamped** :data:`POSITION_GROUP_KEY` list, if it covers every multipoint. That is
       the answer the GUI resolved at load — from the file's sidecar if the user has edited
       one, else from the detector — and it WINS, because an edit the user made by hand must
       not be re-derived away by a node that looked at the coordinates again.
    2. otherwise the detector, run here on the geometry
       (:func:`~nodegraph.placement.position_groups`).

    So a Dataset that never went through the GUI — a headless fixture, a checkpoint, a raw
    ND2 opened by a script — still groups correctly, and one that did carries the user's
    corrections. Both paths return the same type, so no caller has to know which it got.

    Imported lazily: :mod:`nodegraph.placement` is first-party and dependency-free, but this
    module is imported by the graph layer on every edit and the detector is only needed by
    the one node that groups.
    """
    from nodegraph.placement import (GROUP_GAP_FACTOR, GroupPlan, PositionGroup,
                                     position_groups)

    n = int(getattr(axes, "m", 0) or 0)
    stamped = metadata.get(POSITION_GROUP_KEY)
    if isinstance(stamped, (list, tuple)) and len(stamped) >= n > 0:
        keys = [str(k) for k in stamped[:n]]
        if all(keys):
            order: List[str] = []
            for k in keys:
                if k not in order:
                    order.append(k)
            groups = tuple(
                PositionGroup(key=k,
                              members=tuple(i for i, v in enumerate(keys) if v == k))
                for k in order)
            # `margin`/`gap_um` stay at their "nothing was measured" defaults: these groups
            # were READ, not clustered, so there is no threshold the answer turned on and
            # reporting one would invent a confidence this path never computed.
            return GroupPlan(groups=groups, placed=True)
    return position_groups(metadata, axes,
                           GROUP_GAP_FACTOR if gap_factor is None else gap_factor)


def group_picks(metadata: Mapping[str, Any], axes: Any, raw: Any,
                gap_factor: Optional[float] = None) -> Optional[Tuple[int, ...]]:
    """A group selection resolved against a Dataset: the multipoint indices it keeps.

    The per-M twin of :func:`frame_spec_picks`, with the same three-valued answer and for
    exactly the same reason — ``None`` = nothing selected (keep every position), a non-empty
    tuple = these positions, and ``()`` = something WAS selected and matched nothing, which
    the compute refuses while the advisory ``meta_transform`` holds the pre-edit envelope
    (it re-runs on every keystroke, and ``"G1"`` is seen while somebody types ``"G12"``).

    The spelling is deliberately forgiving, because there is nothing to be gained from
    making somebody remember it: ``"G3"``, ``"g3"`` and bare ``"3"`` all name the third
    group, a comma list (``"G1,G4"``) keeps several, and any name a user gave a group in the
    sidecar works in place of the generated key. Indices in the result are in ascending
    order, never selection order — the output's ``m`` axis is the input's with positions
    removed, and re-ordering it would break every per-M list that rides alongside.

    Shared by ``util.select_group``'s compute and :func:`select_group`, so the predicted
    extent and the produced extent cannot drift (build-node-v2 §2).
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    plan = position_group_plan(metadata, axes, gap_factor)
    if not plan.placed:
        return ()
    by_key = {g.key.casefold(): g for g in plan.groups}
    # The ordinal spelling is resolved against POSITION in the plan, not against the key
    # text, so "3" means the third group even for a sidecar that renamed them all.
    keep: List[int] = []
    for token in text.replace(";", ",").split(","):
        tok = token.strip()
        if not tok:
            continue
        got = by_key.get(tok.casefold())
        if got is None and tok.isdigit():
            i = int(tok) - 1
            got = plan.groups[i] if 0 <= i < len(plan.groups) else None
        if got is not None:
            keep.extend(got.members)
    n = int(getattr(axes, "m", 0) or 0)
    return tuple(sorted({int(m) for m in keep if 0 <= int(m) < n}))


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
#: :attr:`nodelab_v2.runner.EngineRunner._channel_display` — and, since V4.00 step 11g,
#: ``channel_names``/``channel_colors`` onto the edit-time SOURCE envelope as well
#: (:func:`nodelab_v2.ingest.channel_display_seed`), which is what lets a card downstream
#: of a channel tap name the channel it carries. They travel together, so they are subset
#: together.
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


def crop_to_field(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.crop_to``: crop to where ANOTHER file sits in absolute stage µm. M→1, Y/X
    (and, in 3D, Z) shrink by an amount this pass cannot see.

    The one thing to understand about this transform is what it deliberately does **not**
    claim. It is handed only input 0's envelope — ``propagate_meta`` reads
    ``dataset_preds[0]`` — so the reference file's placement, which is the entire input to
    the window calculation, is invisible here. Every published option would be a lie:

    * guessing the extent would put a confident wrong number on the wire and into every
      derived spinbox downstream;
    * holding the input's extent would claim a crop did nothing;
    * so Y/X are marked **UNKNOWN**, which is the same answer :func:`stitch` gives for the
      same reason (V2.03 §2 A3) and the only one a consumer can act on. Z joins them under
      the 3D lever, where the node also cuts planes to the reference's focus span.

    ``m`` is **not** unknown: it is exactly 1: the node always resolves to the single
    position that best covers the region, however many were in the input. So the axis is
    predicted, and only its size — not which position survived — is knowable here.

    ``origin_um`` is DROPPED rather than shifted, and that is the second half of the same
    admission: the cut corner is the intersection of two files' fields, and this pass can
    see one of them. Absent is the honest signal (:func:`overlay` drops ``bit_depth`` on
    the same reasoning). The payload restamps the true corner it measured, so placement
    downstream — which reads the Dataset, not the envelope — stays exact; what degrades is
    only the edit-time prediction, and it degrades to "unknown" rather than to "wrong".

    Everything else is untouched on purpose. Nothing is resampled, so ``pixel_size_um``
    survives; no axis is re-spaced, so ``dt_s`` and ``z_step_um`` survive; no value is
    rewritten, so ``bit_depth`` survives. The remaining per-M lists retire through
    :func:`drop_position_keys` exactly as they do for a stitch, because after this there is
    one field whose stage CENTRE is no longer its field centre — a survivor left at full
    length would report another position's coordinate, and a length-1 one would report the
    uncropped field's.
    """
    ax = env.axes
    unknown = set(env.unknown_axes) | {"y", "x"}
    if (modes or {}).get("dim") == "3D":
        unknown.add("z")
    changes: Dict[str, Any] = dict(drop_position_keys(env.metadata))
    if env.metadata.get("origin_um") is not None:
        changes["origin_um"] = None
    return (env.with_axes(replace(ax, m=1), unknown=frozenset(unknown))
               .with_metadata(**changes))


def select_group(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.select_group``: narrow M to the positions of the chosen specimen group.

    A strict subset of what :func:`crop_frames` does — M only, and the picks come from
    :func:`group_picks` instead of a typed index list. It is a separate transform rather
    than a mode of that one because the two answer different questions and fail differently:
    a frame spec is resolved against an axis LENGTH and cannot be wrong about the data,
    while a group is resolved against the stage GEOMETRY and may legitimately come back
    "these fields cannot be placed".

    That case is the one to get right. When the fields cannot be located, the picks are
    ``()`` and this holds the pre-edit envelope — it does NOT mark ``m`` unknown. The
    difference matters: an unknown axis propagates down the whole graph as "nothing
    downstream can be predicted", which is a heavy thing to do on a node the user is still
    typing into, and the payload will refuse with a message naming the missing stage log
    anyway. Holding is also what ``crop_frames`` does for an unparseable spec.

    Only M moves, so — unlike a frame crop — there is no re-spacing and no origin shift to
    apply: ``dt_s``, ``z_step_um``, ``z_home_index`` and every per-T list describe axes this
    node does not touch, and each surviving position keeps the ``origin_um`` it always had.
    What does move is every per-M list (:func:`position_subset`), and that includes
    :data:`POSITION_GROUP_KEY` itself — after selecting one group the survivors all carry
    that one key, which is correct and is what makes a second Select Group downstream a
    no-op rather than a puzzle.
    """
    ax = env.axes
    keep = group_picks(env.metadata, ax, params.get("group"),
                       params.get("gap_factor"))
    if keep is None or keep == ():
        return env
    return (env.with_axes(replace(ax, m=len(keep)))
               .with_metadata(**position_subset(env.metadata, keep)))


def position_pick(metadata: Mapping[str, Any], m: int, raw: Any) -> Optional[int]:
    """Resolve ``util.select_position``'s ``position`` to one index into ``m`` positions
    (2026-10-02) — shared by its compute and :func:`select_position` so the card and the
    pull cannot disagree. An exact match on :data:`POSITION_NAME_KEY` first (the
    acquisition's own point label, when the file carries one), else a bare 0-based index
    in range. ``None`` for empty (nothing asked for) or for a value that names nothing;
    the caller decides whether that is a no-op or a refusal."""
    want = str(raw if raw is not None else "").strip()
    if not want or m <= 0:
        return None
    names = metadata.get(POSITION_NAME_KEY)
    if isinstance(names, (list, tuple)) and len(names) == m:
        for i, nm in enumerate(names):
            if str(nm) == want:
                return i
    if want.isdigit():
        i = int(want)
        if 0 <= i < m:
            return i
    return None


def select_position(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.select_position``: narrow M to ONE position, by name or 0-based index
    (2026-10-02) — the multipoint twin of :func:`batch_select`, and the tap
    ``util.split_positions``'s per-position outputs materialize into.

    Only M moves, exactly as in :func:`select_group`: no re-spacing, no origin shift, and
    every per-M list follows through :func:`position_subset`. Empty, or a value that
    resolves to nothing, HOLDS the envelope rather than marking ``m`` unknown — the compute
    refuses the second case with the positions listed, and an unknown axis would grey the
    whole downstream graph behind a plausible-looking card."""
    ax = env.axes
    k = position_pick(env.metadata, int(ax.m), params.get("position"))
    if k is None or int(ax.m) <= 1:
        return env
    return (env.with_axes(replace(ax, m=1))
               .with_metadata(**position_subset(env.metadata, [k])))


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


def overlay(env: MetaEnvelope, params: Mapping, modes: Mapping,
            inputs: Optional[Sequence[MetaEnvelope]] = None) -> MetaEnvelope:
    """``view.overlay``: identity in ``display`` mode, **C+1** in ``resample`` mode, and the
    UNION CANVAS under ``canvas=union`` — one blank field over the primary's and the
    secondary's fields, predicted from BOTH envelopes (``wants_inputs``) with the very
    :func:`canvas_union` ``view.canvas`` uses, so the card's size is the pull's.


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
    if (modes or {}).get("canvas") == "union" and (modes or {}).get("output") != "resample":
        envs = [e for e in (list(inputs) if inputs else [env]) if e is not None]
        # The canvas ORIENTATION: this node's Flip X/Y when it starts the canvas; the existing
        # canvas's when it grows one, so a chain keeps the orientation its first Overlay chose.
        from nodegraph.placement import canvas_flip
        if str(env.metadata.get("stage_layout_source", "")) == "canvas":
            fx, fy = canvas_flip(env.metadata)
        else:
            fx = bool((params or {}).get("flip_x", True))
            fy = bool((params or {}).get("flip_y", False))
        return canvas_union(env, {"flip_x": fx, "flip_y": fy}, {}, envs[:2] or [env])
    if (modes or {}).get("output") != "resample":
        return env
    ax = env.axes
    names = list(env.metadata.get("channel_names") or [])
    while len(names) < ax.c:
        names.append(f"Ch{len(names) + 1}")
    names.append("overlay")
    return (env.with_axes(replace(ax, c=ax.c + 1))
               .with_metadata(bit_depth=None, channel_names=names))


def _merge_channels_grow(env: MetaEnvelope) -> MetaEnvelope:
    """The ``merge_axis="C"`` branch of :func:`merge_grow`: C and Z both grow, both UNKNOWN.

    This pass is handed only INPUT 0's envelope (``propagate_meta`` reads
    ``dataset_preds[0]``), so it cannot see any of the other inputs at all — and both
    changed axes are functions of them:

    * ``c`` grows by however many channels the other inputs have, combined;
    * ``z`` becomes the merged grid, whose plane count depends on their focus ranges and
      steps (:func:`nodegraph.placement.merge_z_grid`).

    ``view.overlay``'s ``resample`` mode solves the same blindness by baking exactly ONE
    channel, which keeps its prediction exact. That is the right trade for an overlay you are
    going to *measure one channel of*, and the wrong one here: this node exists to put every
    input on one axis, so the honest answer is ``stitch``'s — mark the axes unknown rather
    than guess (V2.03 §2 A3). The GUI shows "?" for them until the first pull, which is true.

    ``bit_depth`` is dropped for the same reason it is under ``resample``: the output stacks
    several files' intensity scales and this pass has never seen the others.

    ``z_step_um`` is dropped too, and that one matters more than it looks. The merged grid's
    step is the finest of the inputs', which this pass cannot compute — and leaving input 0's
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


def merge_grow(env: MetaEnvelope, params: Mapping, modes: Mapping) -> MetaEnvelope:
    """``util.merge``: grows exactly one axis — C, T, M or Z — picked by the ``merge_axis``
    Mode (the node that absorbed and retired ``channel.merge``, 2026-09-15).

    Handed only INPUT 0's envelope (``propagate_meta`` reads ``dataset_preds[0]``), so every
    other input's contribution to the grown axis is invisible here — the prediction marks
    that ONE axis UNKNOWN rather than guess, the same rule ``z_project``/the old
    ``channel.merge`` already follow (V2.03 §2 A3).

    * ``merge_axis="C"`` — placement-based channel merge: see :func:`_merge_channels_grow`.
      Z becomes unknown too (the merged Z grid depends on the other inputs' focus ranges),
      and ``bit_depth``/``z_step_um`` are dropped.
    * ``merge_axis="T"/"M"/"Z"`` — literal concatenation: the node's refuse-on-mismatch
      contract (:mod:`nodegraph.catalog.util.merge`) means every OTHER axis and every other
      calibration key on input 0 is already correct for the merged result, so only the grown
      axis itself needs marking unknown — nothing else changes.
    """
    axis = str((modes or {}).get("merge_axis", "C"))
    if axis == "C":
        return _merge_channels_grow(env)
    ax_name = axis.lower()
    if ax_name not in ("t", "m", "z"):
        return env
    return env.with_axes(env.axes, unknown=env.unknown_axes | frozenset({ax_name}))
@dataclass(frozen=True)
class ChainMember:
    """One source FILE of a ``util.chain`` — the unit that node lays onto an axis.

    A member is a file wherever it came from, which is what lets one node serve the two
    shapes its input arrives in: a **file bundle** contributes several members that share
    one ``source`` and one ``metadata`` and differ in ``start``, while **separately wired
    files** contribute one member each, from different ``source`` indices, all starting at
    0. Everything downstream — the provider, the metadata lockstep, the resolved-order
    note — reads this and never asks which shape it was.
    """

    #: Index of the wired input this file came from, in the engine's canonical socket
    #: order. Two members sharing it came out of ONE bundle, which is the difference that
    #: decides whether a per-axis list can be concatenated (see :func:`chained_metadata`).
    source: int
    #: That input's whole metadata dict — shared between members of one bundle.
    metadata: Mapping[str, Any]
    #: That input's own axes, so a positional list can be checked before it is sliced and
    #: the member's extent along the CHAINED axis is knowable. Shared within a bundle: the
    #: files of one card necessarily have the same T/Z/C as the card.
    input_axes: AxisSizes
    #: Where this file's positions begin inside its input's ``m``.
    start: int
    #: How many positions this file contributes.
    count: int
    #: The file's name, from :data:`SOURCE_FILE_KEY`; ``""`` when the input carries none.
    name: str = ""

    def extent(self, axis: str) -> int:
        """How much of the CHAINED axis this file contributes."""
        return self.count if axis == "m" else max(1, int(getattr(self.input_axes, axis)))


#: ``util.timeseries``'s ``chain_order`` values. ``time`` (the default since 2026-10-02)
#: sorts the files by their first frame's absolute clock; ``sequence`` by the counting
#: field in their names; ``loaded`` keeps the wiring order.
CHAIN_ORDERS: Tuple[str, ...] = ("time", "sequence", "loaded")


def chain_member_clock(mm: "ChainMember") -> Optional[float]:
    """The Julian day of a member's FIRST frame, or ``None`` when its input carries no
    per-T clock. A member of a bundle shares its input's list, which describes the
    bundle's first file only, so a bundle's members all answer with the same instant —
    they then keep their name order among themselves (the sort is stable) while the
    bundle as a block is placed by that instant."""
    jd = mm.metadata.get("frame_time_jd")
    if not isinstance(jd, (list, tuple)) or not jd:
        return None
    try:
        v = float(jd[0])
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def chain_members(inputs: Sequence[Tuple[Mapping[str, Any], AxisSizes]], axis: str, *,
                  sequence_order: bool = True, order: Optional[str] = None
                  ) -> Tuple[List[ChainMember], int, Optional[str]]:
    """``util.timeseries``'s file members, in OUTPUT order — shared by the compute and
    :func:`chain_grow` so the card and the pull cannot disagree about the result.

    ``order`` is one of :data:`CHAIN_ORDERS`; the older ``sequence_order`` flag maps onto
    ``"sequence"`` / ``"loaded"`` and is kept for callers that predate ``time``.

    ``inputs`` is ``(metadata, axes)`` per wired Dataset, in wiring order. Each input is
    split into its files by :func:`source_file_runs`; an input carrying no usable
    :data:`SOURCE_FILE_KEY` is one unnamed member covering its whole ``m``, which is what
    makes a plain single-file card work without a special case.

    Returns ``(members, positions_per_file, problem)``. ``problem`` is a sentence naming
    what is wrong and how to fix it, and is non-``None`` exactly when the chain cannot be
    built — the compute raises it, and the envelope goes UNKNOWN rather than predicting a
    shape the pull is going to refuse.

    **What has to agree, and what deliberately does not.** Every axis EXCEPT ``m`` and the
    one being chained must already match: the result is rectangular in those. The chained
    axis is exempt because laying files end to end along it is the whole operation — a
    series exported in unequal chunks (5 frames, then 3) is still one series, and refusing
    it would refuse the ordinary case. ``m`` is exempt because each file is read through its
    own window; instead, when chaining onto anything but ``m``, every file must hold the
    same number of POSITIONS, since the result has one position axis and a file with more
    has nowhere to put them.

    **Ordering.** ``time`` sorts the members by their first frame's absolute clock
    (:func:`chain_member_clock`), but only when EVERY member has one — a half-clocked set
    has no coherent order — and falls back to ``sequence`` otherwise; ties (the members of
    one bundle, which share a clock) keep their relative order. ``sequence`` sorts by the
    counting field in the names (:func:`nodegraph.file_sequence.order`), again only when
    EVERY member has a name: a half-named list sorted would put the named files in sequence
    and the rest wherever they fell, which reads as working. Otherwise the order is the
    wiring order — inputs in socket order, each input's own files in bundle order — which
    is the only information that exists in that case. :func:`chain_order_used` names the
    rule that actually applied.
    """
    axis = str(axis).lower()
    if order is None:
        order = "sequence" if sequence_order else "loaded"
    members: List[ChainMember] = []
    for i, (md, ax) in enumerate(inputs):
        runs = source_file_runs(md, int(ax.m)) or [("", 0, int(ax.m))]
        for name, start, count in runs:
            members.append(ChainMember(source=i, metadata=md, input_axes=ax,
                                       start=start, count=count, name=name))
    if not members:
        return [], 0, "Chain has no input wired to it."

    others = [a for a in ("b", "t", "z", "c", "y", "x") if a != axis]
    first = members[0].input_axes
    for mm in members[1:]:
        bad = [a for a in others
               if int(getattr(mm.input_axes, a)) != int(getattr(first, a))]
        if bad:
            want = tuple(int(getattr(first, a)) for a in others)
            got = tuple(int(getattr(mm.input_axes, a)) for a in others)
            return [], 0, (
                f"Chain cannot lay these files onto {axis.upper()}: "
                f"{mm.name or f'input {mm.source}'} does not match "
                f"{members[0].name or f'input {members[0].source}'} on "
                f"{', '.join(bad)} — ({', '.join(others)}) is {got} against {want}. "
                f"Chaining lays files end to end on {axis.upper()} alone, so every OTHER "
                f"axis has to already agree; resample or crop upstream so they match, or "
                f"chain onto a different axis.")

    counts = sorted({mm.count for mm in members})
    if axis != "m" and len(counts) != 1:
        # Name every file and its count, not just the minority: which files are "the odd
        # ones" is the user's judgement, and the fix differs depending which way it runs.
        shown = ", ".join(f"{mm.name or f'input {mm.source}'} holds {mm.count}"
                          for mm in members[:6])
        more = f", and {len(members) - 6} more" if len(members) > 6 else ""
        return [], 0, (
            f"Chain cannot lay these {len(members)} files onto {axis.upper()}: they hold "
            f"different numbers of POSITIONS ({counts[0]} to {counts[-1]}) — {shown}{more}. "
            f"The result has one position axis, so a file with more has nowhere to put "
            f"them. Chain onto M instead to keep them as separate positions, or use "
            f"util.select_group to cut every file down to the same positions first. "
            f"(Files may differ freely on {axis.upper()} itself — that is what is being "
            f"chained.)")
    n = counts[0] if axis != "m" else 0
    used = chain_order_used(members, order)
    if used == "time":
        clocks = [chain_member_clock(mm) for mm in members]
        members = [members[j] for j in sorted(range(len(members)),
                                              key=lambda j: (clocks[j], j))]
    elif used == "sequence":
        members = [members[j] for j in _seq_order([mm.name for mm in members])]
    return members, n, None


def chain_order_used(members: Sequence["ChainMember"], order: str) -> str:
    """Which ordering rule ``order`` resolves to on these members: ``time`` needs a clock
    on every member, ``sequence`` a name on every member; each falls back to the next
    (``time`` → ``sequence`` → ``wired``). ``wired`` is the stamped word for wiring order,
    whether chosen (``loaded``) or fallen back to."""
    if order == "time" and members and all(chain_member_clock(mm) is not None
                                           for mm in members):
        return "time"
    if order in ("time", "sequence") and members and all(mm.name for mm in members):
        return "sequence"
    return "wired"


#: How far one position's recorded LATERAL location may wander across chained files and
#: still be one field, as a fraction of the field's smaller side. A timelapse exported one
#: file per frame revisits each point, and the stage reads back a slightly different
#: coordinate every visit (encoder repeatability, thermal drift of the plate) — so exact
#: equality never holds on real ND2s, and requiring it dropped the log that ``util.stitch``
#: needs. A tenth of a field is the ordinary tile overlap: a disagreement smaller than it
#: still places every tile against its real neighbours, and ``stage+refine`` measures the
#: rest. It is also far below the spacing of two DIFFERENT positions of a mosaic, so it
#: cannot mistake a neighbour for the same field.
CHAIN_SAME_FIELD_FRACTION = 0.1

#: The per-M keys that are coordinates, and how to read ``(x, y)`` / ``z`` out of one
#: entry. Only these are compared with a tolerance; the rest (names, groups, alignment
#: rows) must agree exactly, because "nearly the same name" is not the same name.
_CHAIN_GEOMETRY: Dict[str, Tuple[Optional[Callable[[Any], Tuple[float, float]]],
                                 Optional[Callable[[Any], float]]]] = {
    "stage_xy_um": (lambda v: (float(v[0]), float(v[1])), None),
    "stage_z_um": (None, lambda v: float(v)),
    "origin_um": (lambda v: (float(v[2]), float(v[1])), lambda v: float(v[0])),
}


def _chain_slices(members: Sequence[ChainMember], key: str) -> Optional[List[List[Any]]]:
    """Each member's own slice of a per-M list, or ``None`` if any member cannot say."""
    got: List[List[Any]] = []
    for mm in members:
        vals = mm.metadata.get(key)
        if not isinstance(vals, (list, tuple)) or len(vals) != int(mm.input_axes.m):
            return None
        got.append(list(vals[mm.start:mm.start + mm.count]))
    return got


def chain_position_spread(members: Sequence[ChainMember], key: str
                          ) -> Optional[Tuple[float, float]]:
    """``(lateral_um, axial_um)``: the furthest any file puts a position from where the
    FIRST file puts it, for a coordinate key in :data:`_CHAIN_GEOMETRY` — or ``None`` when
    the key is not one, or is missing, short or non-numeric in any file.

    Measured against the first member rather than a centroid because the first is the
    value that is kept (:func:`chained_metadata`), so this is exactly the error the kept
    value carries for the worst frame. Shared with ``util.chain``'s note so the number the
    card reports is the number the keep/drop decision was made on.
    """
    lat_of, ax_of = _CHAIN_GEOMETRY.get(key, (None, None))
    if lat_of is None and ax_of is None:
        return None
    slices = _chain_slices(members, key)
    if not slices:
        return None
    lat = ax = 0.0
    try:
        for sl in slices[1:]:
            for ref, v in zip(slices[0], sl):
                if lat_of is not None:
                    (x0, y0), (x1, y1) = lat_of(ref), lat_of(v)
                    lat = max(lat, math.hypot(x1 - x0, y1 - y0))
                if ax_of is not None:
                    ax = max(ax, abs(ax_of(v) - ax_of(ref)))
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(lat) and math.isfinite(ax)):
        return None
    return lat, ax


def chain_position_tolerance(members: Sequence[ChainMember]) -> Tuple[float, float]:
    """``(lateral_um, axial_um)`` a position may wander across files and stay one field.

    Lateral is :data:`CHAIN_SAME_FIELD_FRACTION` of the field's smaller side; axial is one
    ``z_step_um`` — a nominal focus that moved by less than one plane still addresses the
    same planes. Either is ``0.0`` (exact agreement only) when the calibration that scales
    it is missing, because a tolerance in pixels means nothing without the pixel size."""
    md = members[0].metadata
    ax = members[0].input_axes
    try:
        px = float(md.get("pixel_size_um") or 0.0)
        dz = float(md.get("z_step_um") or 0.0)
    except (TypeError, ValueError):
        px = dz = 0.0
    lat = CHAIN_SAME_FIELD_FRACTION * px * min(int(ax.y), int(ax.x)) if px > 0 else 0.0
    return lat, (dz if dz > 0 else 0.0)


def chained_metadata(members: Sequence[ChainMember], axis: str, n: int) -> Dict[str, Any]:
    """The metadata changes ``chain_grow`` and ``util.chain``'s compute BOTH apply.

    One function because the two must agree exactly — the standing lockstep rule for an
    axis-changing node (`wire-node-v2` Section 8) — and because each family below fails
    differently if it is left stale.

    **Per-M** (:data:`PER_POSITION_KEYS`). Onto ``m`` the members line up as positions, so
    the lists CONCATENATE and every value keeps the position it described. Onto any other
    axis the result has ``n`` positions, each now assembled from all K files, so a value
    survives only where every file already agreed — the timelapse case this node exists
    for, where K files of one field at different times share one ``stage_xy_um``.
    "Agreed" means exactly for names and alignment rows, but for the COORDINATE keys
    (:data:`_CHAIN_GEOMETRY`) it means "within :func:`chain_position_tolerance`": a stage
    that revisits a point never reads back the same micron twice, so exact equality
    dropped the log on every real ND2 series and left ``util.stitch`` nothing to place by.
    Within tolerance the FIRST file's value is kept. Beyond it the key is DROPPED, not
    taken from the first: a chained position really did come from K different places,
    and naming one puts a coordinate on a voxel that was not acquired there.
    :data:`SOURCE_FILE_KEY` falls out of this by itself — the names never agree — which is
    correct and needs no special case.

    **The chained axis's own family** (:data:`PER_TIME_KEYS` onto T,
    :data:`PER_CHANNEL_KEYS` onto C) is concatenated per member, each contributing its OWN
    length — which is what lets files with different frame counts chain — but only when the
    members come from DISTINCT inputs, so that each carries its own list. Members of one
    bundle share a single list that describes only the FIRST file (``bundle_envelope`` takes
    the non-positional keys from member 0), so:

    * onto **T** it is dropped. Repeating it would say every file was acquired at the same
      instants, which is the one thing a timelapse-per-file series is guaranteed not to be.
    * onto **C** it is tiled anyway. That is not a guess: the loader refuses to bundle
      files whose channel names differ, so block ``i``'s channels genuinely are the same
      stains in the same order.

    ``dt_s``/``z_step_um`` are deliberately left alone. They describe the spacing WITHIN a
    file, which the chain does not change; the gap BETWEEN two files is unknown and this
    schema has nowhere to put an irregular one. See ``util.chain``'s description.
    """
    out: Dict[str, Any] = {}
    k = len(members)
    tol_lat, tol_ax = chain_position_tolerance(members)

    for key in PER_POSITION_KEYS:
        if not any(key in mm.metadata for mm in members):
            continue
        slices = _chain_slices(members, key)
        if slices is None:
            # Present somewhere but not readable everywhere. Dropped rather than carried
            # forward: a positional list that outlives the axis it indexed does not look
            # stale, it looks like the wrong position.
            out[key] = None
        elif axis == "m":
            out[key] = [v for sl in slices for v in sl]
        elif all(sl == slices[0] for sl in slices):
            out[key] = slices[0]
        else:
            # A coordinate that differs by stage jitter is still ONE field -- the first
            # file's reading is kept, the same one-value-per-position a native multi-T
            # ND2 carries (read at T=0). Anything further apart is K different places.
            spread = chain_position_spread(members, key)
            same = spread is not None and spread[0] <= tol_lat + 1e-9 \
                and spread[1] <= tol_ax + 1e-9
            out[key] = slices[0] if same else None

    if axis == "m":
        return out

    distinct = len({mm.source for mm in members}) == k
    family = PER_TIME_KEYS if axis == "t" else PER_CHANNEL_KEYS if axis == "c" else ()
    for key in family:
        if not isinstance(members[0].metadata.get(key), (list, tuple)):
            continue
        if distinct:
            # Each member contributes its OWN length, so unequal chunks concatenate
            # correctly; a list that does not match its own file's extent is not a chunk
            # this can place, so the whole key goes rather than half of it.
            lists = [mm.metadata.get(key) for mm in members]
            if all(isinstance(v, (list, tuple)) and len(v) == mm.extent(axis)
                   for v, mm in zip(lists, members)):
                out[key] = [v for one in lists for v in one]
            else:
                out[key] = None
        elif axis == "c":
            out[key] = list(members[0].metadata[key]) * k
        else:
            out[key] = None
    return out


def canvas_changes(base: Mapping[str, Any], canvas: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``with_metadata`` changes that turn input 0's metadata into an experiment
    CANVAS's — shared by ``view.canvas``'s compute and :func:`canvas_union`, so the payload
    and the edit-time prediction are the same dict (INV-04).

    Every per-position key retires (the canvas is ONE field; its ``origin_um`` and its own
    focus grid come from :func:`nodegraph.placement.union_canvas`), every per-channel key
    retires (the canvas has one blank channel, not input 0's), and the Z anchoring is
    replaced outright — a stale ``z_home_index`` from input 0 would place the canvas's planes
    at another file's focus. The clock (``dt_s``, ``frame_time_jd``) is input 0's, on purpose:
    Overlays onto the canvas pair T as they would against that file."""
    changes: Dict[str, Any] = {k: None for k in PER_POSITION_KEYS if base.get(k) is not None}
    changes.update({k: None for k in PER_CHANNEL_KEYS if base.get(k) is not None})
    for k in ("z_step_um", "z_home_index", "z_bottom_to_top", "z_collapsed",
              "bit_depth", "stage_layout_source"):
        if base.get(k) is not None:
            changes[k] = None
    changes.update(dict(canvas.get("metadata") or {}))
    return changes


def canvas_union(env: MetaEnvelope, params: Mapping, modes: Mapping,
                 inputs: Optional[Sequence[MetaEnvelope]] = None) -> MetaEnvelope:
    """``view.canvas``: one blank field spanning every input's fields at their true stage
    positions (:func:`nodegraph.placement.union_canvas`).

    Opts into every input's envelope (``wants_inputs``), so the card shows the real canvas
    size while you wire — the same function the compute calls, so it cannot differ. Total:
    an input that cannot be placed yet (no envelope, no origin) marks Y/X/Z unknown instead
    of raising, and the compute names the problem when it is pulled."""
    from nodegraph.placement import union_canvas
    envs = [e for e in (list(inputs) if inputs else [env]) if e is not None]
    try:
        uc = union_canvas([(e.metadata, e.axes) for e in envs],
                          pixel_size_um=float(params.get("pixel_size_um", 0.0) or 0.0),
                          margin_um=float(params.get("margin_um", 0.0) or 0.0),
                          flip=(bool(params.get("flip_x", True)),
                                bool(params.get("flip_y", False))))
    except Exception:  # noqa: BLE001 — the edit-time pass must never raise (INV-06)
        uc = {"refusals": ["unplaceable"]}
    if uc.get("refusals"):
        return env.with_axes(env.axes, unknown=env.unknown_axes | frozenset({"y", "x", "z",
                                                                             "m", "c"}))
    ax = env.axes
    new_axes = replace(ax, m=1, t=int(uc["t"]), z=int(uc["z"]), c=1,
                       y=int(uc["y"]), x=int(uc["x"]))
    return env.with_axes(new_axes, unknown=env.unknown_axes - frozenset(
        {"y", "x", "z", "m", "c"})).with_metadata(**canvas_changes(env.metadata, uc))


canvas_union.wants_inputs = True
# `overlay` is defined above `canvas_union` in this module but only CALLS it at edit time;
# it opts into every input's envelope for the canvas=union branch (the secondary's fields
# are half of the canvas), and ignores the extra inputs in every other mode.
overlay.wants_inputs = True


def chain_grow(env: MetaEnvelope, params: Mapping, modes: Mapping,
               inputs: Optional[Sequence[MetaEnvelope]] = None) -> MetaEnvelope:
    """``util.chain``: lay every wired input's FILES onto one axis — T, M, C or Z (V3.02).

    **Predicted exactly, not marked unknown** — the thing that separates this from
    :func:`merge_grow`. That transform sees only input 0 and so cannot know how much the
    other inputs add; this one opts into the whole list (``chain_grow.wants_inputs``, read
    by :func:`propagate_meta`), and every input's per-position :data:`SOURCE_FILE_KEY`
    already names its files. So the node card shows a real ``T=120`` while you are still
    wiring, rather than a ``?`` only a pull could resolve — and it shows the right number
    when the files hold DIFFERENT frame counts, which is a sum rather than a multiple.

    It degrades to the input, unchanged, in the cases where the compute is also a no-op: a
    single file with nothing to chain, or an axis it does not recognise. It marks ``m`` and
    the target axis UNKNOWN only where the compute REFUSES, because predicting the shape of
    a pull that is going to raise would grey the refusal out behind a plausible-looking card.
    """
    axis = str((modes or {}).get("chain_axis", "T")).lower()
    if axis not in ("t", "z", "c", "m"):
        return env
    envs = list(inputs) if inputs else [env]
    # An input whose own m is a guess cannot be split into files -- its `source_file` would
    # be measured against a length that is not real. Stay unknown rather than compound it.
    if any({"m", axis} & set(e.unknown_axes) for e in envs):
        return env.with_axes(env.axes, unknown=env.unknown_axes | frozenset({axis, "m"}))
    members, n, problem = chain_members(
        [(e.metadata, e.axes) for e in envs], axis,
        order=str((modes or {}).get("chain_order", "time")))
    if problem is not None:
        return env.with_axes(env.axes, unknown=env.unknown_axes | frozenset({axis, "m"}))
    if len(members) < 2:
        return env
    ax = env.axes
    total = sum(mm.extent(axis) for mm in members)
    changes = chained_metadata(members, axis, n)
    new_axes = replace(ax, m=total) if axis == "m" else \
        replace(ax, m=n, **{axis: total})
    return env.with_axes(new_axes,
                         unknown=env.unknown_axes - frozenset({axis, "m"})) \
              .with_metadata(**changes)


#: :func:`propagate_meta` hands this transform EVERY dataset input's envelope, not just
#: input 0's. Opt-in per transform (and absent on all the others) so the protocol stays
#: what it was: a node that only ever reads its primary input cannot accidentally start
#: depending on a second one's envelope, and adding this cost no edit to the other
#: fourteen transforms.
chain_grow.wants_inputs = True


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
    "overlay": overlay, "merge_grow": merge_grow,
    "chain_grow": chain_grow,
    "identity": identity, "resample": resample, "z_project": z_project,
    "stack_time": stack_time, "frame_slice": frame_slice,
    "channel_select": channel_select, "crop": crop, "stitch": stitch,
    "select_group": select_group,
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
        if transform is None:
            env_out = env_in
        elif getattr(transform, "wants_inputs", False):
            # A transform that must see EVERY dataset input opts in by name (today only
            # `chain_grow`). Without it the protocol hands over input 0's envelope alone,
            # so an N-input node can do no better than marking the axis it grows unknown
            # -- which is exactly what `merge_grow` settles for.
            env_out = transform(env_in, node.params, node.state(spec),
                                [out.get(e.src, MetaEnvelope()) for e in dpreds])
        else:
            env_out = transform(env_in, node.params, node.state(spec))
        # Domain accumulation: union EVERY Dataset predecessor's domain-set (a merge
        # node combines them), then add what this node produces. A root seeds from its
        # own envelope's domain-set. The meta_transform never touches domains, so this
        # is layered on afterward. (V2.06: the socket domain-rail + wire-tint source.)
        # An input declared `passes_domains=False` is read, not merged: its domains do not
        # reach this node's output (`io.write_movie`'s `source_b`/`source_c`).
        # A node whose output is a NEW Dataset (`NodeSpec.fresh_output` — a plot's Picture)
        # passes nothing on: its domains, layers and columns are only what it adds.
        fresh = bool(getattr(spec, "fresh_output", False)) if spec is not None else False
        if fresh:
            dom_in: FrozenSet[Domain] = frozenset()
        elif dpreds:
            readonly = ({s.name for s in spec.inputs if not getattr(s, "passes_domains", True)}
                        if spec is not None else set())
            dom_in = frozenset().union(
                *(out.get(e.src, MetaEnvelope()).domains for e in dpreds
                  if e.dst_socket not in readonly or e is dpreds[0]))
        else:
            dom_in = env_in.domains
        adds = spec.adds_domains if spec is not None else frozenset()
        env_out = env_out.with_domains(dom_in | adds)
        base = MetaEnvelope() if fresh else env_in
        env_out = env_out.with_layer_names(
            _layer_names_out(spec, node, base, env_out))
        env_out = env_out.with_column_names(
            _column_names_out(spec, node, base, env_out,
                              inputs=tuple((e.dst_socket, out.get(e.src, MetaEnvelope()))
                                           for e in dpreds)))
        out[nid] = _kept_only(spec, node, env_out)
    return out


def _kept_only(spec, node, env: MetaEnvelope) -> MetaEnvelope:
    """Apply ``NodeSpec.keep_layers`` (V4.00 step 11f): only the layers of the kept NAMES
    survive, with their columns; a structure domain survives only if a kept layer lives on
    it, the acquisition lattice always. Total — a declaration that raises keeps everything."""
    keep_fn = getattr(spec, "keep_layers", None) if spec is not None else None
    if keep_fn is None:
        return env
    try:
        keep = keep_fn(getattr(node, "params", {}) or {}, node.state(spec))
    except Exception:                                    # pragma: no cover - defensive
        return env
    if keep is None:
        return env
    keep = frozenset(str(k) for k in keep)
    names = tuple((d, n) for d, n in env.layer_names if n in keep)
    cols = tuple((d, lyr, c) for d, lyr, c in env.column_names if lyr in keep)
    doms = frozenset(d for d in env.domains if is_lattice(d)) | {d for d, _n in names}
    return env.with_domains(doms).with_layer_names(names).with_column_names(cols)


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


def _column_names_out(spec, node, env_in: MetaEnvelope,
                      env_out: MetaEnvelope, *, inputs: Tuple = ()
                      ) -> Tuple[Tuple[Domain, str, str], ...]:
    """This node's outgoing STRUCTURE-column catalog: the input's, minus what its layer
    catalog dropped, plus what its ``adds_columns`` declares (V2.28).

    **Total by contract**, for the same reason as :func:`_layer_names_out` and enforced the
    same way: this runs inside ``propagate_meta`` on every keystroke, whose caller catches
    only ``ValueError`` — anything else takes the window down, and even a caught error
    blanks every node's envelope graph-wide. So a producer whose declaration raises
    contributes nothing and the pass continues; it can never be the reason an edit fails.

    **The drop rule is inherited, not restated.** A column cannot outlive the layer it sits
    on, so anything whose ``(domain, layer)`` pair is no longer in ``env_out.layer_names``
    goes with it. In practice structure domains are never reshaped and this is a no-op —
    but deriving it from the layer catalog rather than asserting that keeps the two from
    drifting if a future node does drop a structure layer.

    **Every input, on request (V4.00).** The catalog flows from input 0 alone, which is right
    for every node that passes one Dataset on — but a node that builds a table out of SEVERAL
    inputs (``table.concat``'s many, ``table.join``'s ``other``) writes columns that only a
    later input carries. A declaration marked ``wants_inputs = True`` is called with a fourth
    argument, ``inputs`` — ``((socket, envelope), ...)`` for each Dataset input in canonical
    order — so it can name them; without it every such column would be unpickable in the
    closed column menus downstream.
    """
    surviving = {(d, n) for d, n in env_out.layer_names}
    cols: List[Tuple[Domain, str, str]] = [
        (d, lyr, c) for d, lyr, c in env_in.column_names if (d, lyr) in surviving]
    if spec is None:
        return tuple(dict.fromkeys(cols))
    adds = getattr(spec, "adds_columns", None)
    if adds is None:
        return tuple(dict.fromkeys(cols))
    try:
        state = node.state(spec)
    except Exception:                                    # pragma: no cover - defensive
        state = {}
    params = getattr(node, "params", {}) or {}
    try:
        got = (adds(params, state, tuple(cols), tuple(inputs))
               if getattr(adds, "wants_inputs", False) else adds(params, state, tuple(cols)))
        for entry in got or ():
            dom, lyr, col = entry
            if isinstance(lyr, str) and isinstance(col, str) and lyr and col:
                cols.append((dom, lyr, col))
    except Exception:                                    # pragma: no cover - defensive
        pass
    return tuple(dict.fromkeys(cols))                    # de-dup, keep first appearance


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
    "select_group", "group_picks", "position_group_plan",
    "POSITION_GROUP_KEY", "POSITION_NAME_KEY",
    "PER_POSITION_KEYS", "position_subset", "drop_position_keys", "SOURCE_FILE_KEY",
    "source_file_runs", "chain_grow", "chained_metadata",
    "ChainMember", "chain_members", "stamp_source_file",
    "PER_TIME_KEYS", "CONDITION_KEY", "CONDITION_SET_KEY", "time_subset", "respaced", "z_home_after",
    "eval_derive", "resolve_dim_default",
]
