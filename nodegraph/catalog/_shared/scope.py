"""scope — the statistics POPULATION vocabulary, shared by every node with a data-derived level.

One question, asked identically by several nodes: *which* values are pooled into the statistic a
parameter is derived from? Until V2.27 each node answered it privately and the answer was
copy-pasted three times — ``analysis.threshold``'s ``_THRESHOLD_SCOPES``/``_threshold_scope_key``,
``analysis.filter_labels``' ``_FILTER_SCOPES``/``_scope_key``, and ``enhance.normalize``'s inline
choice list — with the same four words in each and no shared definition to keep them honest.

**Two things are deliberately kept apart here.**

* the **population** — which values the statistic pools (this module, `role="scope"`);
* the **footprint** — how much of the series one kernel call must read
  (:class:`~nodegraph.registry.Granularity`).

They correlate but they are not the same, and conflating them is what makes a per-label scope
look impossible. ``enhance.normalize`` is the standing proof: its population is a ``scope`` Mode
while its :attr:`~nodegraph.registry.NodeSpec.footprint_mode` is ``bounds``. So the scopes below
*map onto* the existing five Granularity members and **add none** — a new member would read as
"not WHOLE_VOLUME" to the 22 catalog sites that test ``ctx.is_volume`` and as "not lazy" to
``_shared/map_image.py``, which then allocates the whole series as float64.

**The structure scopes need no streaming machinery, and that is a property of the vocabulary
rather than luck.** ``per_label``/``per_roi`` subdivide a population *inside* one unit, so no
population ever spans units and each is complete the moment its unit is read. A per-TRACK scope
would break that (a track pools across t by definition) and is deliberately absent; add it only
with a partial-fold accumulator, never by widening a footprint.

Imports stay narrow on purpose (see ``_shared/__init__``): numpy, ``registry``, ``domains``.
``label_components`` is imported lazily inside :func:`roi_populations` so this module does not
drag the structure spine into the closure of every node that only wants a group key.

Qt-free.
"""

from __future__ import annotations

import numpy as np

from typing import Dict, FrozenSet, Iterator, Mapping, Optional, Sequence, Tuple

from nodegraph.domains import Domain
from nodegraph.registry import Granularity, Mode, ModeSpec

#: Every population, finest → coarsest, lattice before structure. The first four are the
#: historical vocabulary (``analysis.threshold``, ``enhance.normalize``); the last two are the
#: V2.27 structure-relative ones.
SCOPES: Tuple[str, ...] = ("plane", "volume", "series", "dataset", "per_label", "per_roi")
#: Populations addressed by the acquisition lattice — one group per (m,t,z,c) subset. These
#: pool ACROSS units, which is why they are the ones that need a cost strategy.
LATTICE_SCOPES: Tuple[str, ...] = ("plane", "volume", "series", "dataset")
#: Populations addressed by structure — one group per region, subdividing a single unit.
STRUCTURE_SCOPES: Tuple[str, ...] = ("per_label", "per_roi")

#: The honest read footprint each scope implies, from the EXISTING five members.
#:
#: The two structure scopes declare ``WHOLE_VOLUME`` as a conservative UPPER BOUND, and the
#: reason is structural rather than lazy: :meth:`NodeSpec.resolve_granularity` is handed the mode
#: state and nothing else, so it cannot see whether the Label instance on the wire is
#: per-plane (``z_kind="plane_index"``) or volumetric (``"subpixel"``) — and a volumetric region
#: spans z. Over-declaring a footprint is safe (the scheduler reads more than it must);
#: under-declaring is the fatal direction, and is exactly the misdeclaration
#: ``analysis.threshold`` was corrected for. The consequence to remember is that
#: **``ctx.is_volume`` is not the authority under a structure scope** — use
#: :func:`scope_is_3d`, which reads the provenance the segmentation actually recorded.
SCOPE_GRAN: Dict[str, Granularity] = {
    "plane": Granularity.WHOLE_PLANE,        # the plane's own values
    "volume": Granularity.WHOLE_VOLUME,      # pooled over z
    "series": Granularity.WHOLE_SERIES,      # pooled over t (and z)
    "dataset": Granularity.MULTI_VIEW,       # pooled over m as well — the widest read
    "per_label": Granularity.WHOLE_VOLUME,   # a region may span z; see above
    "per_roi": Granularity.WHOLE_VOLUME,     # same
}
#: What each scope REQUIRES on the wire, for ``NodeSpec.reads_domains_by_mode`` — so the card's
#: domain rail turns red on a graph that selected a population its input cannot supply.
#: ``per_label`` needs a whole Label INSTANCE (a raster whose ids divide the foreground into
#: objects, plus the table that proves it); ``per_roi`` needs only a Voxel mask, which is what
#: makes a drawn ROI or a bare threshold usable as an arena.
SCOPE_DOMAINS: Dict[str, FrozenSet[Domain]] = {
    "plane": frozenset(), "volume": frozenset(),
    "series": frozenset(), "dataset": frozenset(),
    "per_label": frozenset({Domain.VOXEL, Domain.LABEL}),
    "per_roi": frozenset({Domain.VOXEL}),
}
#: The card's footprint band renders the selected scope as a pill about 96 px wide. A longer
#: token elides, so the vocabulary is capped rather than the pill being widened into the chip
#: beside it (``nodelab_v2.node_item._foot_pill_rect``).
SCOPE_TOKEN_MAX = 12

#: The canned hover prose, exactly as ``DIM_DESCRIPTION`` is canned for the 2D/3D lever: the
#: control means the same thing on every node that bears it, and N hand-written copies would
#: only differ where one of them was wrong. A node with something extra to say passes its own.
SCOPE_DESCRIPTION = (
    "Which values are pooled into the statistic this node derives its level from — the setting "
    "that decides what a result is reproducible FROM. It does not change which planes are READ "
    "(all of them are); it changes which of them have to agree on one number. It never pools "
    "across channels, because two stains with different dynamic ranges under one level "
    "thresholds the dim one into nothing. The last two choices are not lattice populations at "
    "all: they derive one level per OBJECT, so a bright cell and a dim one are cut at different "
    "absolute levels."
)
#: ``{scope: prose}``. The four lattice entries are carried over VERBATIM from
#: ``analysis.threshold``'s own ``choice_docs`` — they are correct, they are already gated by
#: ``test_option_docs``, and rewording them here would only risk drift with the node that has
#: shipped them since 2026-07-30.
SCOPE_DOCS: Mapping[str, str] = {
    "plane":
        "One level per (Y,X) plane, from that plane's own histogram. The "
        "default, the most adaptive, and what every ImageJ user expects; the "
        "price is that a plane containing no objects gets a level derived "
        "from noise and comes out full of speckle.",
    "volume":
        "One level per (Z,Y,X) volume — per position, timepoint and channel. "
        "A Z stack of one object is cut as one thing, so a dim top slice is "
        "not pushed to its own mid-grey and the mask stays connected through "
        "z.",
    "series":
        "One level per (position, channel), pooled over the whole timelapse. "
        "The mask cannot drift just because the field bleached, which is what "
        "you want before measuring an area time course — a per-plane level "
        "would silently track the bleaching and report constant area.",
    "dataset":
        "One level per channel, pooled over EVERYTHING else including every "
        "multipoint. Right for a tiled acquisition of one continuous "
        "specimen; wrong for a plate, where it makes a well's mask depend on "
        "which other wells share the file. Measured on the lab's 49-position "
        "WellA3 plate: pooling three positions instead of one moved a "
        "position's foreground fraction by −47%. It is also the pre-2026-07-30 "
        "behaviour, kept for graphs tuned against it.",
    "per_label":
        "One level per LABEL REGION, from only the voxels that region owns — so each cell is "
        "cut against its own brightness instead of the frame's. This is the only choice that "
        "can find the bright structure inside a bright cell AND inside a dim one at the same "
        "time; no single global level can, because the dim cell's structure is darker than the "
        "bright cell's background. Needs a label raster and its table on the wire (Segmentation "
        "or Connected Components), and 2D-vs-3D follows that segmentation's own provenance "
        "rather than any lever here. A region too small or too flat to have a histogram is "
        "skipped and reported, never given the frame's level.",
    "per_roi":
        "One level per connected region of a MASK — a drawn ROI, or any binary mask, with no "
        "label table required. Two uses in one choice: several drawn shapes each get their own "
        "level, and a single-blob mask (the common case) yields exactly one population, which "
        "is how you say \"derive the level from the foreground only, ignoring the empty "
        "background that would otherwise drag it down\". Coarser than `per_label` — regions "
        "that touch are one arena here — and correspondingly cheaper, since it needs no "
        "segmentation upstream.",
}


def ScopeMode(choices: Sequence[str] = SCOPES, *, default: str = "plane",
              available_in: Optional[Mapping[str, FrozenSet[str]]] = None,
              description: str = "",
              choice_docs: Optional[Mapping[str, str]] = None) -> ModeSpec:
    """The statistics-population Mode — the ``DimMode()`` of populations (V2.27).

    ``choices`` is this node's **subset** of :data:`SCOPES`, and subsetting is the point: a node
    whose scope groups table ROWS (``analysis.filter_labels``) has no meaningful per-label
    population, because that population would be one row; a node with no label input cannot
    offer ``per_label`` at all. Offering a choice the compute cannot honour is the "live-looking
    control the kernel ignores" the node charter forbids.

    Carries :data:`SCOPE_DESCRIPTION` and the matching slice of :data:`SCOPE_DOCS` unless the
    node overrides them, and stamps ``role="scope"`` — which is what makes the card's footprint
    band an editor for it rather than a readout (:attr:`ModeSpec.is_scope`)."""
    bad = [c for c in choices if c not in SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s) {bad} — the shared vocabulary is {list(SCOPES)}. "
                         f"Add it to nodegraph.catalog._shared.scope (with its footprint and "
                         f"its required domains) rather than inventing one per node.")
    if not choices:
        raise ValueError("ScopeMode needs at least one choice")
    docs = {c: SCOPE_DOCS[c] for c in choices}
    docs.update(choice_docs or {})
    return Mode("scope", list(choices), default=default, label="Scope", role="scope",
                available_in=available_in,
                description=description or SCOPE_DESCRIPTION, choice_docs=docs)


#: The per-plane variant of :data:`SCOPE_GRAN`, for a node that is 2D BY CONSTRUCTION — one
#: whose kernel cannot see a volume at all (``analysis.histogram_threshold`` wraps a segmenter
#: hardcoded to ``is_3d=False``). There the conservative ``WHOLE_VOLUME`` above would be a
#: pessimization *and* a small lie on the card's cost chip: the node genuinely reads one plane
#: at a time, and a population that spanned z is not something it can honour — so it refuses a
#: volumetric structure instance rather than quietly cutting one object at several levels.
SCOPE_GRAN_2D: Dict[str, Granularity] = {
    **SCOPE_GRAN, "volume": Granularity.WHOLE_VOLUME,
    "per_label": Granularity.WHOLE_PLANE, "per_roi": Granularity.WHOLE_PLANE,
}


def scope_declarations(choices: Sequence[str], *, per_plane: bool = False
                       ) -> Tuple[Dict[str, Granularity],
                                  Dict[str, Dict[str, FrozenSet[Domain]]]]:
    """``(granularity, reads_domains_by_mode)`` for a node offering ``choices``.

    Both declarations are derived from one table so they cannot drift, and the granularity map
    is **total over the subset** — which is what ``NodeRegistry._check_footprint`` now requires,
    because a missing key resolves to ``None`` and silently drops the node off the tiled read
    path. Pass the result straight into ``register_node(granularity=…, footprint_mode="scope",
    reads_domains_by_mode=…)``.

    ``per_plane`` selects :data:`SCOPE_GRAN_2D` for the structure scopes — for a node whose
    kernel is 2D by construction. Still one table, so the lockstep gate still holds."""
    table = SCOPE_GRAN_2D if per_plane else SCOPE_GRAN
    gran = {c: table[c] for c in choices}
    per_value = {c: SCOPE_DOMAINS[c] for c in choices if SCOPE_DOMAINS[c]}
    return gran, ({"scope": per_value} if per_value else {})


def scope_key(scope: str, unit: Tuple[int, int, int, int]) -> tuple:
    """The statistics-group key a ``(m,t,z,c)`` unit belongs to, for a LATTICE scope.

    Channel is never pooled over at any scope: two channels are two different stains with
    different dynamic ranges, and one shared cut would threshold the dim one into nothing. That
    matches every other per-channel decision in the catalog (``ctx.channel(c)``) and
    ``enhance.normalize``, whose ``series`` scope is likewise "per (m,c)".

    A **structure** scope raises rather than returning the unit: its populations subdivide the
    unit and are enumerated by :func:`unit_populations`, so a caller reaching here with
    ``per_label`` has a branch it forgot to write — and the failure it would otherwise produce
    (one level for the whole unit) looks exactly like a working per-region threshold."""
    if scope in STRUCTURE_SCOPES:
        raise ValueError(
            f"scope_key: {scope!r} is a per-object population, so it has no unit group key — "
            f"enumerate it with unit_populations() inside each unit instead.")
    m, t, z, c = unit
    if scope == "plane":
        return (m, t, z, c)
    if scope == "volume":
        return (m, t, c)
    if scope == "series":
        return (m, c)
    if scope == "dataset":
        return (c,)
    raise ValueError(f"unknown scope {scope!r} — one of {list(SCOPES)}")


#: Which invariant table columns each lattice scope groups ROWS by (:func:`scope_row_key`).
_ROW_COLS: Dict[str, Tuple[str, ...]] = {
    "plane": ("m", "t", "z", "c"), "volume": ("m", "t", "c"),
    "series": ("m", "c"), "dataset": ("c",),
}


def scope_row_key(scope: str, cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """One integer group code per table ROW, for a LATTICE scope — the vectorized twin of
    :func:`scope_key`, for a node whose members are rows rather than voxels.

    ``cols`` is the table's invariant columns; only the ones :data:`_ROW_COLS` names are read.
    The codes are ``np.unique`` inverse indices rather than a hand-rolled mixed radix: the radix
    form needed ``t.max() + 1`` and so raised on an empty table and could overflow on a wide
    one, and nothing downstream reads the code's VALUE — only which rows share it."""
    if scope in STRUCTURE_SCOPES:
        raise ValueError(
            f"scope_row_key: {scope!r} is a per-object population. On a table whose rows ARE "
            f"the objects it would give every row a population of one, so a level derived from "
            f"it would be that row's own value — offer only the lattice scopes here.")
    names = _ROW_COLS.get(scope)
    if names is None:
        raise ValueError(f"unknown scope {scope!r} — one of {list(SCOPES)}")
    missing = [n for n in names if n not in cols]
    if missing:
        raise ValueError(f"scope_row_key: scope={scope} groups by {list(names)}, but the table "
                         f"is missing {missing}")
    n = int(len(cols[names[0]]))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    stack = np.stack([np.asarray(cols[k], dtype=np.int64) for k in names], axis=1)
    _uniq, inv = np.unique(stack, axis=0, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).reshape(-1)


def unit_populations(pop: np.ndarray, values: np.ndarray
                     ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Enumerate the per-object populations inside ONE unit.

    ``pop`` is an integer array of population ids (a label raster, or the CCL of an ROI mask)
    and ``values`` the co-registered values. Yields ``(pop_id, flat_index, value_vector)`` per
    id, ascending, ids ``<= 0`` treated as background — the ``bridges._group_reduce``
    convention, where Label and Track ids are ``>= 1``.

    ``flat_index`` indexes the FLATTENED unit and is what the caller writes results back
    through. Index it into the 6-D output via ``np.unravel_index`` and a coordinate tuple, never
    by reshaping a slice: ``out[m, t, :, c].reshape(-1)[idx] = 1`` silently discards the write,
    because that slice is non-contiguous and ``reshape`` therefore copies.

    One stable ``argsort`` plus ``np.unique(return_index=True)``, not ``pop == pid`` per id: the
    latter is a full pass over the unit for every region, i.e. O(voxels × regions) on a frame
    that routinely holds hundreds of them."""
    flat_pop = np.asarray(pop).reshape(-1)
    flat_val = np.asarray(values).reshape(-1)
    if flat_pop.shape != flat_val.shape:
        raise ValueError(f"unit_populations: population {flat_pop.shape} and values "
                         f"{flat_val.shape} must cover the same unit")
    nz = np.flatnonzero(flat_pop > 0)
    if nz.size == 0:
        return
    keys = flat_pop[nz]
    order = np.argsort(keys, kind="stable")
    idx = nz[order]
    sorted_key = keys[order]
    ids, starts = np.unique(sorted_key, return_index=True)
    counts = np.diff(np.append(starts, sorted_key.size))
    vals = flat_val[idx]
    for pid, s, cnt in zip(ids.tolist(), starts.tolist(), counts.tolist()):
        yield int(pid), idx[s:s + cnt], vals[s:s + cnt]


def scope_is_3d(z_kind: str, n_z: int) -> bool:
    """Whether a structure scope's populations are VOLUMES rather than planes — inherited from
    the structure's own ``z_kind`` provenance (`wire-node-v2` §7b), never from a lever and never
    from ``ctx.is_volume``.

    The rule ``analysis.measure`` already applies: ``"subpixel"`` means the regions were
    segmented volumetrically, ``"plane_index"`` means each plane's regions are their own
    objects — and a single-plane Dataset is 2D whatever the table claims, because a z==1 volume
    has no third dimension to pool over (the belt-and-braces half, ``measure.py:349``).

    Why not ``ctx.is_volume``: :data:`SCOPE_GRAN` declares the conservative ``WHOLE_VOLUME`` for
    both structure scopes, so ``ctx.is_volume`` reads True even for a per-plane segmentation.
    Trusting it would pool a cell with whatever sits above it in the next plane."""
    return z_kind == "subpixel" and int(n_z or 1) > 1


def roi_populations(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """``per_roi``'s populations: the connected components of ``mask`` as an integer array.

    Ids are per-call (1..K in raster order) — the caller offsets them if it needs global
    uniqueness. ``label_components`` is imported lazily so this module stays out of the import
    closure of nodes that only want a group key."""
    from nodegraph.structure import label_components
    labels, _table = label_components(np.asarray(mask) != 0, connectivity)
    return labels


__all__ = [
    "SCOPES", "LATTICE_SCOPES", "STRUCTURE_SCOPES", "SCOPE_GRAN", "SCOPE_DOMAINS",
    "SCOPE_DESCRIPTION", "SCOPE_DOCS", "SCOPE_TOKEN_MAX",
    "ScopeMode", "scope_declarations", "scope_key", "scope_row_key",
    "unit_populations", "scope_is_3d", "roi_populations",
]
