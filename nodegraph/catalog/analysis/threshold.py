"""Threshold (``analysis.threshold``) — Binarize to a Voxel mask — fixed, or a histogram method (otsu/li/yen/triangle/mean) whose level is derived over the chosen Scope: per plane (default), per volume, per position's whole series, or…"""

from __future__ import annotations

import numpy as np

from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.field import FieldCache, FieldContext
from nodegraph.parallel import map_units
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.map_image import _FIELD_TYPES
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.progress import _parallel_progress

# ── analysis: threshold → label → measure ─────────────────────────────────────

#: histogram-based global threshold methods (skimage.filters.threshold_*).
_THRESHOLD_METHODS = ("fixed", "otsu", "li", "yen", "triangle", "mean")
#: the methods that DERIVE their level from a histogram, i.e. the ones for which the
#: statistics ``scope`` (and the read footprint that follows from it) is live.
_THRESHOLD_HISTOGRAM = frozenset(_THRESHOLD_METHODS) - {"fixed"}
#: statistics populations for a histogram threshold, finest → coarsest. The first three
#: mirror ``enhance.normalize``'s ``scope`` exactly (same words, same meaning, so the two
#: nodes read the same way); ``dataset`` is the extra one that pools across MULTIPOINTS.
_THRESHOLD_SCOPES = ("plane", "volume", "series", "dataset")
#: How much of the series each scope must read — the honest footprint, resolved through
#: ``NodeSpec.footprint_mode="scope"``. ``fixed`` never reads a second plane, but a Mode
#: map is keyed by ONE mode's values, so the ``fixed`` case is covered by ``plane``'s
#: entry being the cheapest that is still correct for it (a fixed cut is pointwise, and
#: ``WHOLE_PLANE`` over-declares harmlessly where ``TILEABLE`` under-declared fatally).
_THRESHOLD_GRAN: Dict[str, Granularity] = {
    "plane": Granularity.WHOLE_PLANE,        # the plane's own histogram
    "volume": Granularity.WHOLE_VOLUME,      # pooled over z
    "series": Granularity.WHOLE_SERIES,      # pooled over t (and z)
    "dataset": Granularity.MULTI_VIEW,       # pooled over m as well — the widest read
}
def _threshold_scope_key(scope: str, unit: Tuple[int, int, int, int]) -> tuple:
    """The statistics-group key a ``(m,t,z,c)`` unit belongs to under ``scope``.

    Channel is never pooled over at any scope: two channels are two different stains with
    different dynamic ranges, and one shared cut would threshold the dim one into nothing.
    That matches every other per-channel decision in the catalog (``ctx.channel(c)``) and
    ``enhance.normalize``, whose ``series`` scope is likewise "per (m,c)"."""
    m, t, z, c = unit
    if scope == "plane":
        return (m, t, z, c)
    if scope == "volume":
        return (m, t, c)
    if scope == "series":
        return (m, c)
    return (c,)                                # dataset: everything but the channel
def _compute_threshold(ctx: EvalContext) -> Dataset:
    """Threshold the image to a binary **mask** (a Voxel-domain integer attribute).
    Pointwise apply is nD-agnostic (no lever). ``method`` = ``fixed`` (the ``threshold``
    param) or a histogram method (otsu/li/yen/triangle/mean). A **Field** wired into the
    ``threshold`` socket thresholds per voxel: it is evaluated per plane through the
    engine's FieldCache with a **windowed** FieldContext (C1 / V2.04 §4 — same-domain Attr
    layers slice to the window; the field memo token is the full unit address).

    **``scope`` (the statistics population) — the 2026-07-30 fix.** A histogram method's
    level used to be derived once from the WHOLE (m,t,z,c) cross product, which made a
    position's mask a function of which *other* positions happened to share the file. On
    the lab's 49-position WellA3 plate, thresholding three positions together instead of
    one at a time moved the third one's foreground fraction from 0.0546 to 0.0288 (−47%)
    under otsu, and by −34% to −47% under li/yen/triangle/mean. That is not a tuning
    difference; it means a well's segmentation is not reproducible from that well's pixels.

    ``scope`` is therefore a Mode over the same population vocabulary
    ``enhance.normalize`` already uses, plus the old behaviour kept as a named
    non-default choice:

    * ``plane``    — one level per (m,t,z,c). The default, and what "Otsu" means to
      everyone who has used it in ImageJ.
    * ``volume``   — one level per (m,t,c) over z: a Z-stack of one object thresholded as
      one thing, so a dim top slice is not pushed to its own mid-grey.
    * ``series``   — one level per (m,c) over t and z: the population is one position's
      whole timelapse, so a mask cannot drift just because the field bleached.
    * ``dataset``  — one level per channel over everything else, INCLUDING every
      multipoint. Legitimate for a tiled acquisition of one continuous specimen, wrong for
      a plate; the description says so.

    The default change is deliberate and follows the project's rule that a node's maths is
    corrected rather than preserved, with the old behaviour retained under a name — a
    saved graph that wants the wide pool sets ``scope="dataset"``.

    One deliberate difference from the pre-fix code: **no scope pools across channels.**
    The old single ``fn(whole.ravel())`` did, and two stains with different dynamic ranges
    sharing one cut thresholds the dim one into nothing — so ``dataset`` reproduces the old
    numbers exactly on a single-channel file (the overwhelming majority, and the file this
    was found on) and deliberately does not on a multi-channel one. Every other
    per-channel decision in the catalog is made the same way (``ctx.channel(c)``,
    ``enhance.normalize``'s per-``(m,c)`` series scope).

    ``fixed`` reads no histogram, so ``scope`` is Mode-gated away under it (§5c) and the
    footprint drops to TILEABLE (``footprint_mode="scope"``)."""
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    method = modes.get("method", "fixed")
    # unset ⇒ the socket's derive: mid-range of the CURRENT declared bit depth, or 0.5
    # when there is none (post-Normalize [0,1] data). channel(0) because the depth is
    # channel-independent; a user-set value overrides, as always.
    thr = float(ctx.channel(0).param("threshold", 0.5))
    thr_field = ctx.input("threshold")
    use_field = isinstance(thr_field, _FIELD_TYPES) and method == "fixed"
    units = list(_each_plane(ax))
    whole = None
    levels: Dict[tuple, float] = {}
    scope = str(modes.get("scope") or "plane")
    if method != "fixed":
        import skimage.filters as skf
        if scope not in _THRESHOLD_SCOPES:
            raise ValueError(f"unknown threshold scope {scope!r} — one of "
                             f"{list(_THRESHOLD_SCOPES)}")
        fn = {"otsu": skf.threshold_otsu, "li": skf.threshold_li,
              "yen": skf.threshold_yen, "triangle": skf.threshold_triangle,
              "mean": skf.threshold_mean}[method]
        # Every plane is read regardless of scope — the population differs, not the reads —
        # so the one preallocated buffer and the parallel fill are unchanged. Filled in
        # parallel into one preallocated buffer rather than np.stack of a list
        # comprehension: same bytes, but the reads overlap and there is no transient list
        # of per-plane arrays alongside the stacked copy.
        whole = np.empty((len(units), ax.y, ax.x), dtype=float)

        def _read(iu):
            i, (m, t, z, c) = iu
            whole[i] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

        map_units(_read, list(enumerate(units)))
        # Group the unit indices by the scope's key, then ONE skimage call per group over
        # that group's pooled intensities. `dataset` collapses to a single group, which is
        # bit-identical to the pre-fix single `fn(whole.ravel())`.
        groups: Dict[tuple, list] = {}
        for i, unit in enumerate(units):
            groups.setdefault(_threshold_scope_key(scope, unit), []).append(i)
        for key, idx in groups.items():
            # 1-D intensities (skimage RGB-shape guard); one contiguous take per group.
            levels[key] = float(fn(whole[idx].ravel()))
        thr = next(iter(levels.values()))       # single-group case, and a sane fallback
    fc = ctx.fields if ctx.fields is not None else FieldCache()
    # uint8, not int64 (V2.14): this layer holds 0/1, so int64 spent 8 bytes on one bit —
    # 2.0 GiB instead of 0.25 GiB on a 268-Mvoxel series, and the `.astype` itself measured
    # 1.58× slower. Every consumer in the catalog reads a mask through `!= 0` / `> 0` and
    # allocates its own output with an explicit dtype (`np.zeros_like(mask6, dtype=...)`),
    # so nothing downstream depends on the width.
    mask = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.uint8)
    tick = _parallel_progress(ctx, len(units), "thresholding", frames=ax.t)

    def _threshold_one(iu):
        i, (m, t, z, c) = iu
        # the stat pass already realized every plane — don't pull the lazy chain twice
        plane = whole[i] if whole is not None \
            else prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        # each unit takes its OWN group's level (one shared entry under `dataset`)
        thr_here = levels.get(_threshold_scope_key(scope, (m, t, z, c)), thr) \
            if levels else thr
        if use_field:
            win = {"m": (m, m + 1), "t": (t, t + 1), "z": (z, z + 1), "c": (c, c + 1)}
            # token = the FULL unit address incl. the input provider's fp (V2.04 §6b):
            # the same field expression consumed by two threshold nodes over different
            # geometry must never share a cached materialization.
            val = fc.evaluate(
                thr_field,
                FieldContext(ds, Domain.VOXEL, ax, window=win),
                token=("thr", prov.fingerprint(), m, t, z, c))
            mask[m, t, z, c] = (plane > np.asarray(val).reshape(ax.y, ax.x)
                                ).astype(np.uint8)
        else:
            mask[m, t, z, c] = (plane > thr_here).astype(np.uint8)

    # No fold: each unit owns a disjoint slice of `mask` and nothing is carried between
    # units, so this one is purely a map — no ordering constraint at all.
    def _one_reporting(iu):
        _threshold_one(iu)
        tick()

    map_units(_one_reporting, list(enumerate(units)))
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), mask)
register_node(
    _compute_threshold, op_key="analysis.threshold", label="Threshold",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(),
            # `fixed`-only: every histogram method DERIVES the cut from the data and
            # ignores this socket (compute: `use_field`/`thr` are overwritten unless
            # method == "fixed"), so showing it under otsu/li/… is a lie.
            # a FIXED level is in the image's own units, so its default must follow the
            # declared intensity scale: mid-range of the current bit depth (2047.5 on
            # 12-bit, 32767.5 on 16-bit), falling back to 0.5 when there is no declared
            # integer scale — i.e. exactly after a Normalize dropped it (wire-node-v2 §7c).
            InFloat("threshold", "Threshold", default=0.5, field=True,
                    pick_kind="level",
                    derive="((2**bit_depth - 1)/2) if bit_depth else 0.5",
                    available_in={"method": frozenset({"fixed"})},
                    description=
                    "The intensity cut: a voxel is foreground when it is STRICTLY GREATER "
                    "than this. In the image's OWN units, so the sensible range follows the "
                    "data — auto starts at the mid-point of the declared bit depth (2047.5 "
                    "on 12-bit) and falls back to 0.5 for [0,1] data, i.e. after a Normalize. "
                    "Raising it shrinks every object and drops the dimmest ones entirely. "
                    "Only shown under the `fixed` method; otsu/li/yen/triangle/mean derive "
                    "their own level from the histogram and ignore this. Can be driven by a "
                    "FIELD for a per-voxel cut, which is how you threshold against a "
                    "background estimate instead of a constant."),
            InString("name", "Output layer", field=False, default="mask",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the Voxel mask layer this node writes (1 = foreground, 0 = "
                     "background). Downstream nodes select it by this name — Connected "
                     "Components and Segmentation's watershed both take a mask layer name. "
                     "Two thresholds in one graph need two names, or the second overwrites "
                     "the first.")],
    outputs=[OutDataset()],
    modes=[Mode("method", list(_THRESHOLD_METHODS),
                description=
                "How the intensity cut is chosen: either you type it (`fixed`) or one of five "
                "histogram methods derives it from the pixels. They differ in what they "
                "ASSUME the histogram looks like, so on data that violates the assumption "
                "they fail in characteristic directions — the usual symptom of the wrong "
                "choice here is a mask that is nearly everything or nearly nothing. All five "
                "derive their level over the population set by Scope.",
                choice_docs={
                    "fixed":
                        "Use the Threshold socket's value verbatim, in the image's own "
                        "intensity units. The only reproducible option across files — nothing "
                        "adapts, so an image that dims is thresholded more harshly — and the "
                        "only one that accepts a FIELD, which is how you cut against a "
                        "per-voxel background estimate instead of a constant.",
                    "otsu":
                        "Maximizes between-class variance: the classic bimodal split, and "
                        "what \"Otsu\" means in ImageJ. Assumes the two classes are both "
                        "well populated, so it is reliable on confluent stains and biased "
                        "LOW (mask too generous, background creeping in) when the foreground "
                        "covers only a few percent of the frame.",
                    "li":
                        "Minimizes cross-entropy between the thresholded and original image, "
                        "iteratively. Handles a SMALL, sparse foreground far better than Otsu "
                        "— the option to try first on sparse puncta or a few cells in an empty "
                        "field — at the cost of being less stable when the histogram is "
                        "genuinely flat.",
                    "yen":
                        "Yen's maximum-correlation criterion. Usually lands HIGHER than Otsu, "
                        "so masks come out tighter and dimmer objects drop out entirely. "
                        "Useful when you want only unambiguously bright structure and can "
                        "afford to lose the faint tail.",
                    "triangle":
                        "Geometric: draws a line from the histogram's peak to its far end and "
                        "cuts where the histogram is furthest from that line. Built for a "
                        "SKEWED, single-peak histogram — one dominant background mode with a "
                        "long bright tail — which is exactly where Otsu's two-class "
                        "assumption breaks down.",
                    "mean":
                        "The mean intensity of the population, used directly as the level. "
                        "Trivially cheap and predictable, but it only lands sensibly when "
                        "foreground and background occupy comparable areas; a mostly-empty "
                        "frame pulls it into the noise and the mask fills with background.",
                }),
           # Gated to the histogram methods (wire-node-v2 §5c): a `fixed` cut reads no
           # histogram, so a population picker under it would be a dead control.
           Mode("scope", list(_THRESHOLD_SCOPES), default="plane", label="Scope",
                available_in={"method": _THRESHOLD_HISTOGRAM},
                description=
                "Which pixels are pooled into the histogram the level is derived from — the "
                "setting that decides what a mask is reproducible FROM. It never changes "
                "which planes are read (all of them are), only which of them have to agree "
                "on one cut, and it never pools across channels: two stains with different "
                "dynamic ranges under one level thresholds the dim one into nothing.",
                choice_docs={
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
                })],
    # Keyed by `scope`, not by a dim lever this node does not have — see
    # NodeSpec.footprint_mode. The shipped TILEABLE was a misdeclaration: every histogram
    # method has always read every plane.
    granularity=_THRESHOLD_GRAN, footprint_mode="scope", kernel_axes=frozenset(),
    description="Binarize to a Voxel mask — fixed, or a histogram method "
                "(otsu/li/yen/triangle/mean) whose level is derived over the chosen "
                "Scope: per plane (default), per volume, per position's whole series, or "
                "one level across the entire dataset. The threshold socket shows only "
                "under `fixed` (the others self-derive); Scope only under the others.",
)
