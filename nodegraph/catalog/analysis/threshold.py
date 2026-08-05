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
from nodegraph.spill import dense_output, spill_budget

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


#: Bins skimage itself uses when a threshold method histograms internally. Matching it is
#: what makes the streaming level EQUAL the staged one rather than merely close.
_SK_NBINS = 256
#: Bins for a method that consumes raw values instead (``li``): finer, because here the
#: histogram is a *stand-in* for the data rather than a reproduction of skimage's own.
_FINE_NBINS = 1 << 16


def _bins_for(method: str, lo: float, hi: float, integral: bool):
    """``(edges, centers)`` for one group's histogram, chosen by what ``method`` does.

    The staged path hands skimage a **float64** buffer (``whole`` is allocated
    ``dtype=float``), so the binning to reproduce is always skimage's *float* branch —
    ``nbins`` even bins over the group's own min..max. Mirroring
    ``skimage.exposure.histogram``'s integer special case instead was wrong for exactly that
    reason and moved otsu/yen by up to 3.8% of voxels (measured 2026-08-04).

    * ``otsu``/``yen``/``triangle`` histogram to :data:`_SK_NBINS` internally, so the same
      256 bins reproduce their answer exactly.
    * ``li`` works on raw values, so its histogram is a surrogate for the data. Integer
      source data gets **one bin per integer value**, which makes
      :func:`_level_from_hist`'s reconstruction the exact multiset of values and therefore
      li exact too; float data gets :data:`_FINE_NBINS`.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        centers = np.array([lo if np.isfinite(lo) else 0.0], dtype=float)
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float), centers
    if method == "li" and integral and (hi - lo) <= _FINE_NBINS:
        centers = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1, dtype=float)
        return np.append(centers - 0.5, centers[-1] + 0.5), centers
    nbins = _FINE_NBINS if method == "li" else _SK_NBINS
    edges = np.linspace(lo, hi, nbins + 1)
    return edges, (edges[:-1] + edges[1:]) / 2.0


#: How many samples the histogram surrogate may reconstruct for a method with no ``hist=``
#: entry point. 32 M float64 = 256 MiB — large enough that the reconstruction is faithful,
#: small enough to be irrelevant beside the read it replaces.
_SURROGATE_MAX = 32 << 20


def _level_from_hist(fn, method: str, counts: np.ndarray, centers: np.ndarray,
                     total: float, ssum: float) -> float:
    """One group's threshold level from its accumulated histogram.

    Every method here is a functional of the histogram — that is what makes streaming them
    possible at all — but skimage only exposes two of them that way:

    * ``otsu``/``yen`` take ``hist=(counts, centers)``, so the level is **exact**: it is
      literally the call the staged path makes, on the same bins.
    * ``mean`` is ``np.mean``, so a running sum and count give it exactly.
    * ``li``/``triangle`` have no ``hist=`` parameter (verified against skimage 0.26 —
      ``threshold_li(image, *, tolerance, initial_guess, iter_callback)`` and
      ``threshold_triangle(image, nbins)``). They are reconstructed from a **surrogate**
      sample: the histogram re-expanded to at most :data:`_SURROGATE_MAX` values with the
      counts scaled down proportionally.

      ``li`` is exact whenever the surrogate is the true multiset of values — integer
      source data under :data:`_SURROGATE_MAX` samples, which is why :func:`_bins_for`
      bins it per integer value — and ``triangle`` is not quite, because skimage
      re-histograms the surrogate over the *surrogate's* min..max (the bin centres), which
      is half a bin narrower than the original range.

    **Measured agreement** with the staged path, over 60 fixtures (integer 12- and 16-bit,
    unimodal and bimodal float), as a fraction of each group's data range::

        otsu      0          (exact)
        yen       0          (exact)
        mean      0          (exact)
        li        4.3e-05
        triangle  1.8e-03

    One 256-bin width is 3.9e-03 of the range, so even the worst case lands inside half a
    bin. Asserted in ``selftest::test_streaming_stats``, exactly and by bound respectively,
    so a future change that degrades either is a failure and not a surprise.
    """
    if method == "mean":
        return float(ssum / total) if total else 0.0
    if method in ("otsu", "yen"):
        # a degenerate (single-valued) group has no two classes to separate; skimage
        # raises on it, and the constant itself is the only defensible cut
        nz = int((counts > 0).sum())
        if nz < 2:
            return float(centers[0]) if len(centers) else 0.0
        return float(fn(hist=(counts, centers)))
    total_i = int(counts.sum())
    if total_i <= 0:
        return float(centers[0]) if len(centers) else 0.0
    if total_i > _SURROGATE_MAX:
        scaled = np.maximum((counts * (_SURROGATE_MAX / total_i)).astype(np.int64),
                            (counts > 0).astype(np.int64))    # keep every occupied bin
    else:
        scaled = counts.astype(np.int64)
    sample = np.repeat(centers.astype(float), scaled)
    if sample.size == 0 or np.ptp(sample) == 0:
        return float(sample[0]) if sample.size else 0.0
    return float(fn(sample))
def _streaming_levels(ctx: EvalContext, prov, ax, units, groups: Dict[tuple, list],
                      fn, method: str) -> Dict[tuple, float]:
    """Per-group threshold levels **without retaining the dataset** (V2.24).

    Two passes over the planes, each holding one plane per worker instead of all of them:

    1. per group, the min/max (and, for ``mean``, a running sum + count — which makes that
       method exact with no histogram at all);
    2. per group, a histogram over that group's own range, binned the way
       :func:`_hist_bins` says skimage would bin it.

    Then :func:`_level_from_hist` per group. A group of ONE unit — which is every group
    under the default ``scope="plane"`` — skips all of it and calls skimage on the plane
    directly, so the common case is not merely close to the staged path, it *is* the staged
    path with one plane in the buffer. That matters: `plane` is the default, so the
    approximation in the surrogate branch only ever applies to a li/triangle level pooled
    over a dataset too large to hold, which previously could not be computed at all.

    Reads are parallel (:func:`~nodegraph.parallel.map_units`) and the folds are guarded —
    a histogram ``+=`` from several threads is a read-modify-write and would lose counts,
    the same hazard :func:`_parallel_progress` exists for."""
    import threading
    levels: Dict[tuple, float] = {}
    single = {k: idx[0] for k, idx in groups.items() if len(idx) == 1}
    multi = {k: idx for k, idx in groups.items() if len(idx) > 1}

    def _plane(i):
        m, t, z, c = units[i]
        return prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

    # ── groups of one: exact, no histogram, no pooling ────────────────────────
    if single:
        keys = list(single)
        tick1 = _parallel_progress(ctx, len(keys), f"{method} level", frames=ax.t)

        def _one(k):
            a = np.asarray(_plane(single[k]), dtype=float).ravel()
            out = (float(a[0]) if a.size and np.ptp(a) == 0
                   else (float(fn(a)) if a.size else 0.0))
            tick1()
            return out

        for k, lv in zip(keys, map_units(_one, keys)):
            levels[k] = lv
    if not multi:
        return levels

    # ── pooled groups: pass 1, range (+ exact sum/count for `mean`) ───────────
    lock = threading.Lock()
    stats = {k: [np.inf, -np.inf, 0.0, 0] for k in multi}     # lo, hi, sum, n
    integral = bool(np.issubdtype(np.asarray(_plane(0)).dtype, np.integer))
    flat = [(k, i) for k, idx in multi.items() for i in idx]
    # `mean` needs no histogram at all — its level IS the running sum/count — so it pays
    # one pass, every other method two.
    passes = 1 if method == "mean" else 2
    tick2 = _parallel_progress(ctx, len(flat) * passes, f"{method} level", frames=ax.t)

    def _range_one(ki):
        k, i = ki
        a = np.asarray(_plane(i), dtype=float)
        finite = a[np.isfinite(a)]
        lo = float(finite.min()) if finite.size else np.inf
        hi = float(finite.max()) if finite.size else -np.inf
        s = float(finite.sum(dtype=np.float64)) if finite.size else 0.0
        with lock:
            st = stats[k]
            st[0] = min(st[0], lo)
            st[1] = max(st[1], hi)
            st[2] += s
            st[3] += int(finite.size)
        tick2()

    map_units(_range_one, flat)
    if method == "mean":
        for k in multi:
            _lo, _hi, ssum, n = stats[k]
            levels[k] = float(ssum / n) if n else 0.0
        return levels

    # ── pass 2: histogram per group over its own range ────────────────────────
    bins = {k: _bins_for(method, stats[k][0], stats[k][1], integral) for k in multi}
    hists = {k: np.zeros(len(bins[k][1]), dtype=np.int64) for k in multi}

    def _hist_one(ki):
        k, i = ki
        a = np.asarray(_plane(i), dtype=float)
        h, _ = np.histogram(a[np.isfinite(a)], bins=bins[k][0])
        with lock:
            hists[k] += h.astype(np.int64)
        tick2()

    map_units(_hist_one, flat)
    for k in multi:
        _lo, _hi, ssum, n = stats[k]
        levels[k] = _level_from_hist(fn, method, hists[k], bins[k][1], float(n), ssum)
    return levels


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
    fn = None
    #: fuse the level pass into the apply pass — see the `scope == "plane"` branch below.
    per_plane_level = False
    scope = str(modes.get("scope") or "plane")
    if method != "fixed":
        import skimage.filters as skf
        if scope not in _THRESHOLD_SCOPES:
            raise ValueError(f"unknown threshold scope {scope!r} — one of "
                             f"{list(_THRESHOLD_SCOPES)}")
        fn = {"otsu": skf.threshold_otsu, "li": skf.threshold_li,
              "yen": skf.threshold_yen, "triangle": skf.threshold_triangle,
              "mean": skf.threshold_mean}[method]
        # Group the unit indices by the scope's key. `dataset` collapses to a single group,
        # which is bit-identical to the pre-fix single `fn(whole.ravel())`.
        groups: Dict[tuple, list] = {}
        for i, unit in enumerate(units):
            groups.setdefault(_threshold_scope_key(scope, unit), []).append(i)
        # Every plane is read regardless of scope — the population differs, not the reads —
        # but whether every plane is RETAINED is exactly the question. Staging the lot as
        # float64 is `n_units × y × x × 8` bytes, which on the lab's 640 series
        # (12×16×210 planes of 1024²) is 315 GiB and an unconditional
        # `_ArrayMemoryError` — the node became unusable the moment an upstream Z-Project
        # was reset to `none`, while working fine at 1.5 GiB with the projection on
        # (2026-08-04). So the staged path is kept for datasets it fits, because it is the
        # exact one, and a streaming path takes over above the budget.
        stage_bytes = len(units) * int(ax.y) * int(ax.x) * 8
        if stage_bytes <= spill_budget():
            # Filled in parallel into one preallocated buffer rather than np.stack of a
            # list comprehension: same bytes, but the reads overlap and there is no
            # transient list of per-plane arrays alongside the stacked copy.
            whole = np.empty((len(units), ax.y, ax.x), dtype=float)

            def _read(iu):
                i, (m, t, z, c) = iu
                whole[i] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

            map_units(_read, list(enumerate(units)))
            for key, idx in groups.items():
                # 1-D intensities (skimage RGB-shape guard); one contiguous take per group.
                levels[key] = float(fn(whole[idx].ravel()))
            thr = next(iter(levels.values()))   # single-group case, and a sane fallback
        elif scope == "plane":
            # FUSED, and this is the case that matters: `plane` is the default, and its
            # groups are single planes, so a plane's level is a function of that plane
            # alone. Deriving levels in their own pass would then read the whole series
            # twice — 84.7 GB decompressed each way on the 640 series, since reading every
            # plane IS reading the file — for a number that the apply pass is holding the
            # pixels for anyway. So it computes the level inline instead. Identical
            # arithmetic (the same `fn` on the same plane), half the I/O.
            per_plane_level = True
        else:
            levels = _streaming_levels(ctx, prov, ax, units, groups, fn, method)
            thr = next(iter(levels.values()))
    fc = ctx.fields if ctx.fields is not None else FieldCache()
    # uint8, not int64 (V2.14): this layer holds 0/1, so int64 spent 8 bytes on one bit —
    # 2.0 GiB instead of 0.25 GiB on a 268-Mvoxel series, and the `.astype` itself measured
    # 1.58× slower. Every consumer in the catalog reads a mask through `!= 0` / `> 0` and
    # allocates its own output with an explicit dtype (`np.zeros_like(mask6, dtype=...)`),
    # so nothing downstream depends on the width.
    #
    # ...and above `spill_budget` even one bit per voxel is too much to hold: 42.3 Gvoxel is
    # 39.4 GiB as uint8. `dense_output` puts it in a memmapped .npy instead, which the layer
    # then keeps WITHOUT copying and the Memo GC counts as zero (nodegraph.spill). Indexed
    # assignment is identical either way, so nothing below this line knows which it got.
    out = dense_output((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), np.uint8,
                       tag=f"mask_{ctx.node_id}")
    mask = out.array
    tick = _parallel_progress(ctx, len(units), "thresholding", frames=ax.t)

    def _threshold_one(iu):
        i, (m, t, z, c) = iu
        # the stat pass already realized every plane — don't pull the lazy chain twice
        plane = whole[i] if whole is not None \
            else prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        # each unit takes its OWN group's level (one shared entry under `dataset`), or
        # derives it right here from the plane already in hand (the fused per-plane path)
        if per_plane_level:
            a = np.asarray(plane, dtype=float).ravel()
            thr_here = (float(a[0]) if a.size and np.ptp(a) == 0
                        else (float(fn(a)) if a.size else 0.0))
        else:
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
    return ds.with_layer(Domain.VOXEL, ctx.layer("name"), out.seal())
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
