"""Multi-Otsu (``analysis.multiotsu``) — Multi-level Otsu → a class-index Voxel raster
(0..K-1), one binary mask per class, or one mask of chosen classes."""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InInt, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.planes import _each_volume_p

#: ``output`` -> what leaves the node, given the per-voxel class index.
_OUTPUTS: Tuple[str, ...] = ("merged", "per_class", "selected")
#: Fewest classes Otsu can split a histogram into.
_MIN_CLASSES = 2


def _class_layer(name: str, i: int) -> str:
    """The per-class mask layer name: ``<name>_<i>``."""
    return f"{name}_{i}"


def _parse_keep(spec: str, k: int) -> Set[int]:
    """``keep`` -> the set of class indices in ``0..k-1`` it names.

    Accepts a comma-separated mix of single indices (``2``), inclusive ranges (``1-2``) and
    open-ended ranges (``1+`` = 1 and every brighter class). Spaces are ignored. Anything
    else, an index outside ``0..k-1``, or a spec that selects nothing is REFUSED with the
    token that failed — a silent empty mask would read as "no objects" one node later."""
    out: Set[int] = set()
    text = str(spec or "").strip()
    if not text:
        raise ValueError("multi-otsu (selected): `keep` is empty — name at least one class, "
                         "e.g. `1+` for every class above the dimmest.")
    for raw in text.split(","):
        tok = raw.strip().replace(" ", "")
        if not tok:
            continue
        try:
            if tok.endswith("+"):
                lo = int(tok[:-1]); hi = k - 1
            elif "-" in tok:
                a, b = tok.split("-", 1); lo, hi = int(a), int(b)
            else:
                lo = hi = int(tok)
        except ValueError:
            raise ValueError(f"multi-otsu (selected): cannot read {raw.strip()!r} in `keep` "
                             f"— use indices, ranges and `N+`, e.g. `0`, `1-2`, `2+`.") from None
        if lo < 0 or lo > k - 1 or hi > k - 1:
            raise ValueError(f"multi-otsu (selected): {tok!r} is outside the classes this "
                             f"node produces (0..{k - 1} for classes={k}).")
        if lo > hi:
            raise ValueError(f"multi-otsu (selected): range {tok!r} runs backwards.")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError("multi-otsu (selected): `keep` selects no class.")
    return out


def _layers_multiotsu(params, modes):
    """Announce the per-class masks the ``per_class`` output writes — their names are
    derived from ``name`` and ``classes``, so no single ``layer_out`` socket can state them.
    Total: never raises, because ``propagate_meta`` runs it on every keystroke."""
    try:
        if str((modes or {}).get("output") or "merged") != "per_class":
            return ()
        name = str((params or {}).get("name") or "classes")
        k = max(_MIN_CLASSES, int((params or {}).get("classes") or 3))
        return tuple((Domain.VOXEL, _class_layer(name, i)) for i in range(k))
    except Exception:                                   # pragma: no cover - defensive
        return ()


def _compute_multiotsu(ctx: EvalContext) -> Dataset:
    """Multi-level Otsu → a **class-index** Voxel raster (0..K-1) for multi-population
    segmentation (K = ``classes``). Thresholds are derived per (m,t,c) over that volume's
    histogram; a volume with too few distinct levels degrades to all-class-0 (no crash).

    Resolved spec for the ``output`` Mode (2026-10-02, build-node-v2 §0):

    * ``merged`` — the class-index raster under ``name``, as before. One layer, K values.
    * ``per_class`` — that same raster PLUS one binary mask per class, ``<name>_0`` …
      ``<name>_K-1``, so each intensity tier can be routed to its own downstream chain
      (Connected Components on the bright tier, Measure on the middle one, …). The masks
      partition the volume: disjoint, and their union is every voxel. Announced to the
      edit-time layer catalog by :func:`_layers_multiotsu`, since their names depend on two
      params.
    * ``selected`` — one binary mask under ``name`` of the classes ``keep`` names (``1+`` by
      default: everything above the dimmest class), for the common case where the tiers
      exist only to pick a foreground. A bad or empty ``keep`` is refused, not emptied.

    Contract otherwise unchanged: Voxel in, Voxel layers out, no axis or scale change, one
    WHOLE_VOLUME histogram per (m,t,c). The three states hash apart because the Mode folds
    into the recipe key; ``keep`` is read only on the ``selected`` path (R1).
    """
    from skimage.filters import threshold_multiotsu
    ds = ctx.inputs[0]
    prov = ds.image
    ax = prov.axes
    modes = ctx.params.get("__modes__", {})
    output = str(modes.get("output") or "merged")
    if output not in _OUTPUTS:
        raise ValueError(f"multi-otsu: unknown output {output!r} — one of {list(_OUTPUTS)}")
    k = max(_MIN_CLASSES, int(ctx.params.get("classes", 3)))
    name = ctx.layer("name")
    keep = _parse_keep(ctx.params.get("keep", "1+"), k) if output == "selected" else None

    out = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=np.int64)
    for m, t, c in _each_volume_p(ctx, ax, "multi-otsu"):
        vol = prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x).astype(float)
        try:
            th = threshold_multiotsu(vol.ravel(), classes=k)
        except ValueError:
            continue                               # too few distinct levels → all class 0
        out[m, t, :, c] = np.digitize(vol, th)

    if output == "selected":
        return ds.with_layer(Domain.VOXEL, name,
                             np.isin(out, sorted(keep)).astype(np.int64))
    result = ds.with_layer(Domain.VOXEL, name, out)
    if output == "per_class":
        for i in range(k):
            result = result.with_layer(Domain.VOXEL, _class_layer(name, i),
                                       (out == i).astype(np.int64))
    return result


register_node(
    batch_aware(_compute_multiotsu), op_key="analysis.multiotsu", label="Multi-Otsu",
    category="analysis",
    reads_domains=frozenset({Domain.VOXEL}), adds_domains=frozenset({Domain.VOXEL}),
    inputs=[InDataset(), InInt("classes", "Classes", default=3, field=False,
                               description=
                               "How many intensity populations to split the histogram into "
                               "— 3 gives background plus two brightness tiers. With the "
                               "`merged` output this is the number of values in the class "
                               "index (0 = dimmest class); with `per_class` it is the number "
                               "of mask layers written; with `selected` it bounds what "
                               "`Keep classes` may name. MORE classes means finer tiers but "
                               "each is estimated from less of the histogram, so they get "
                               "less stable; a volume with fewer distinct levels than classes "
                               "degrades to all-zero rather than failing. Minimum 2. "
                               "Thresholds are derived per (m,t,c) volume, so the same class "
                               "index can mean different intensities in different frames."),
            InString("keep", "Keep classes", field=False, default="1+",
                     available_in={"output": frozenset({"selected"})},
                     description=
                     "Which classes form the single mask the `selected` output writes. A "
                     "comma-separated list of class indices (0 = dimmest), inclusive ranges "
                     "like `1-2`, and open ranges like `1+` (that class and every brighter "
                     "one). The default `1+` keeps everything above the dimmest class, i.e. "
                     "'all foreground tiers' — the usual way Multi-Otsu is used to beat a "
                     "single Otsu cut on a two-population foreground. `2+` keeps only the "
                     "brightest tiers and shrinks every object to its bright core, which "
                     "lowers reported areas. An index outside 0..classes-1, a backwards "
                     "range or an empty selection is REFUSED rather than yielding an empty "
                     "mask that would read as 'no objects' downstream. Only read when the "
                     "output is `selected`."),
            InString("name", "Output layer", field=False, default="classes",
                     layer_out=(Domain.VOXEL,),
                     description=
                     "Name of the Voxel layer this node writes. Under `merged` and "
                     "`per_class` it holds the class index (values 0..K-1, not 0/1) — named "
                     "`classes` rather than `mask` by default as a reminder that it is not "
                     "directly a binary mask; `per_class` additionally writes `<name>_0` … "
                     "`<name>_K-1`, one 0/1 mask per class. Under `selected` this layer IS "
                     "a 0/1 mask of the kept classes, so renaming it to `mask` there keeps "
                     "downstream pickers honest.")],
    outputs=[OutDataset()],
    modes=[
        Mode("output", list(_OUTPUTS), default="merged", label="Output",
             description=
             "What to make of the K intensity classes once Otsu has found them. The "
             "thresholds are identical in all three; the options differ only in how the "
             "class membership is handed downstream — one index layer, one mask per class, "
             "or one mask of the classes you name.",
             choice_docs={
                 "merged":
                     "One class-index layer (`Output layer`), value 0..K-1 per voxel, 0 the "
                     "dimmest. The compact form and the default; a downstream node must then "
                     "decide which indices count as foreground, which is what the other two "
                     "options do for you.",
                 "per_class":
                     "The class-index layer plus one binary mask per class, named "
                     "`<layer>_0` … `<layer>_K-1`. The masks partition the volume (disjoint, "
                     "union = everything), so each tier can feed its own chain — label and "
                     "measure the brightest tier, use the middle tier as a halo, discard the "
                     "dimmest — and the layer picker downstream offers all of them.",
                 "selected":
                     "One binary mask (`Output layer`) of the classes `Keep classes` names, "
                     "`1+` by default. The form to use when the tiers were only ever a means "
                     "to a better foreground than a single Otsu cut gives: everything "
                     "brighter than the dimmest population, as one mask a Connected "
                     "Components or Measure node takes directly.",
             }),
    ],
    extra_layers=_layers_multiotsu,
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset(),
    description="Multi-level Otsu → a class-index Voxel raster (0..K-1), optionally one "
                "binary mask per class so each tier can be processed differently, or one "
                "mask of the classes you keep; needs the whole volume histogram.",
)
