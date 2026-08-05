"""labels — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain

def _structure_layers(ds: Dataset, domain: Domain) -> list:
    """Names of the structure INSTANCES of ``domain`` on ``ds`` — one entry per table.

    A structure attribute stores the instance name in the ``layer`` slot of its key and the
    COLUMN name in the ``name`` slot (``Dataset.with_structure``), which is the opposite of a
    Voxel raster (see :func:`_voxel_layers`) — hence a helper rather than an inline set
    comprehension per node."""
    return sorted({k[1] for k in ds.attributes if k[0] is domain and k[1]})
def _point_layers(ds: Dataset) -> list:
    """Names of the Point structures on ``ds`` (each is one table)."""
    return _structure_layers(ds, Domain.POINT)
def _label_instances(ds: Dataset) -> list:
    """Names that are a whole Label INSTANCE on ``ds`` — a Voxel raster of ids AND the
    Label table those ids index.

    Not simply "the Voxel layers": a binary ``mask``, a ``classes`` raster from multi-Otsu
    and a float ``distance`` field are all Voxel layers and none of them divides the
    foreground into objects. The table's presence is the signal that some node declared this
    raster a label raster, which is the same test :func:`_label_raster` validates with."""
    voxel = {k[2] for k in ds.attributes if k[0] is Domain.VOXEL}
    return sorted(n for n in voxel if n and ds.structure_zkind(Domain.LABEL, n) is not None)
def _label_tables(ds: Dataset) -> list:
    """Names of the Label TABLES on ``ds`` — one entry per Label-domain structure instance.

    Wider than :func:`_label_instances` on purpose: a node that reads the per-region *table*
    (``track.objects``, ``analysis.object_metrics``) does not need the raster, and a graph
    whose raster was dropped by an axis change still has measurable objects."""
    return _structure_layers(ds, Domain.LABEL)
def _lattice_layers(ds: Dataset, domain: Domain) -> list:
    """Names of the LATTICE attribute layers of ``domain`` on ``ds``.

    The mirror of :func:`_structure_layers`: a lattice attribute is one array per name with no
    table around it, so its name lives in the ``name`` slot and its ``layer`` slot is None."""
    return sorted({k[2] for k in ds.attributes if k[0] is domain and k[2]})
def _voxel_layers(ds: Dataset) -> list:
    """Names of the Voxel layers on ``ds`` (any raster — mask, labels, distance field)."""
    return _lattice_layers(ds, Domain.VOXEL)
def _resolve_layer(candidates: list, want: str, *, node: str, socket: str, what: str,
                   where: str, remedy: str, ctx: Optional[object] = None
                   ) -> Tuple[str, str]:
    """Pick the layer a socket means, falling back to *the only candidate* when it can.

    Returns ``(layer, note)`` — ``note`` is non-empty when the answer was inferred rather
    than taken literally, so the caller can say so on the progress rail.

    **Pass ``ctx`` and the rail note is emitted here.** An inference the user cannot see is
    the difference between "the node worked out which layer you meant" and "the node quietly
    ran on something else", and that visibility is what makes overriding a stale name
    defensible at all — so it must not depend on 17 computes each remembering to print it.
    Callers that own a determinate bar (``analysis.voronoi``, ``analysis.measure``,
    ``transform.label_to_points``) still emit it themselves against their real unit count and
    pass no ``ctx``; everything else passes ``ctx`` and gets a bare note.

    **Why a node infers this at all.** A layer-name socket has to carry SOME default, and a
    literal one is right for exactly one upstream producer: ``analysis.voronoi`` shipped
    ``particles`` (what ``detect.particles`` emits) and so refused every graph that seeded it
    from ``detect.spots`` (``spots``) or ``transform.label_to_points`` (``<labels>_points``,
    or whatever the user typed) — with a "no Point layer 'particles'" error that named the
    layer actually present one clause later. There was nothing to decide: one Point table was
    on the wire. Making the user retype its name is the same class of defect as a spatial
    param shipping a pixel constant instead of a ``derive`` — the value is derivable from the
    incoming data, so the node derives it (`wire-node-v2` §7).

    An EXPLICIT name that exists always wins; inference only fills an unset socket or
    replaces a name that is not there. It never chooses BETWEEN candidates — two Point tables
    on one wire is a real question only the user can answer, and it raises listing them.

    **Every REQUIRED consumer of a named input layer goes through here (2026-08-04).**
    Shipping it on ``analysis.voronoi`` alone left the same trap on the other sixteen nodes,
    and the first one a user hit was the very producer named above: ``analysis.segment``
    writing ``CELLS`` → ``transform.label_to_points`` demanding ``labels``, on a two-node wire
    with exactly one Label instance on it. Renaming was not even needed to reach it — the
    shipped defaults of sockets that are routinely wired to each other DISAGREE:
    ``detect.particles`` emits ``particles`` while ``track.link``/``track.objects`` ask for
    ``spots``, ``analysis.tessellate`` emits ``mesh`` while ``analysis.voronoi``'s mesh is
    ``voronoi_mesh``, ``analysis.dvc_field`` emits ``dvc`` and nothing else does.

    **OPTIONAL layer sockets are excluded, deliberately.** ``analysis.dic_correlate``'s
    ``roi`` and ``analysis.segment``'s ``mask`` mean "no ROI / no seed mask" when they do not
    resolve, so inferring the only mask on the wire would silently *restrict* a correlation
    or reseed a segmentation that was asked to run unmasked. Absent must stay absent there;
    inference is only ever right for a socket the compute cannot proceed without.

    Within that scope this can only widen what runs. A socket whose name IS on the wire
    returns unchanged (first branch), so no existing graph changes behaviour; inference only
    replaces an error with either an answer or a better error."""
    want = str(want or "").strip()
    if want and want in candidates:
        return want, ""
    if len(candidates) == 1:
        only = candidates[0]
        note = (f"{what} {only!r} (the only one on {where})" if not want else
                f"{what} {only!r} — the `{socket}` socket said {want!r}, which is not on "
                f"{where}, and this is the only candidate")
        if ctx is not None:
            # a bare 0/1 tick: this runs before the node knows its unit count, and the point
            # is the NOTE, not the fraction. Never allowed to break the compute.
            try:
                ctx.progress(0, 1, "using " + note)
            except Exception:                    # pragma: no cover - defensive
                pass
        return only, note
    if not candidates:
        raise ValueError(
            f"{node}: no {what} on {where}"
            + (f" (the `{socket}` socket asks for {want!r})" if want else "")
            + f" — {remedy}")
    raise ValueError(
        f"{node}: {where} carries {len(candidates)} candidates for {what} ({candidates}), so "
        f"the `{socket}` socket has to say which"
        + (f" — it currently says {want!r}, which is not one of them" if want else "")
        + ". Pick one from its dropdown.")
def _resolve_label_instance(ds: Dataset, want: str, *, node: str, socket: str,
                           remedy: str, ctx: Optional[object] = None) -> Tuple[str, str]:
    """:func:`_resolve_layer` over the Label INSTANCES, for a caller that then goes through
    :func:`_label_raster`.

    Differs from the plain call in one clause: a name that IS a Voxel layer on ``ds`` but not
    a whole Label instance is handed BACK unchanged instead of being replaced. It exists —
    the user pointed at something real — so the specific refusal belongs to
    :func:`_label_raster`, which explains that this raster carries no Label table and would
    therefore collapse its whole foreground into one region. Inferring past it would answer a
    question the user did not ask, and would lose the one message that names the actual
    problem."""
    want = str(want or "").strip()
    if want and want not in _label_instances(ds) and ds.get(Domain.VOXEL, want) is not None:
        return want, ""
    return _resolve_layer(_label_instances(ds), want, node=node, socket=socket,
                          what="Label instance", where="the `data` input", remedy=remedy,
                          ctx=ctx)
def _label_raster(ds: Dataset, layer: str, *, node: str) -> Tuple[np.ndarray, str]:
    """The Voxel raster of the Label instance ``layer``, with its ``z_kind`` — validated.

    Returns ``(raster6, z_kind)``. Every caller that groups voxels by region goes through
    here, because "is this raster actually a label raster?" is a question none of them
    used to ask (2026-07-30).

    A Label INSTANCE is two things stored under one name: a Voxel raster of integer ids,
    and the Label-domain table those ids index. ``analysis.label`` / ``analysis.segment`` /
    ``analysis.histogram_threshold`` all write the pair together, so the table's presence
    is a reliable "this raster is a label raster" signal — and it is the signal that was
    missing. A ``labels`` socket is ``layer_in=Domain.VOXEL``, so the GUI picker offers
    EVERY Voxel raster on the edge: a binary ``mask`` from ``analysis.threshold``, a
    ``classes`` raster from ``analysis.multiotsu``, a float ``distance`` field from
    ``analysis.edt``. Handing a binary mask to ``voxel_to_label`` is not an error to that
    function — it groups by id, and a 0/1 mask has exactly one non-zero id — so a 3-frame
    2-object stack came back as ONE row with ``area=384``, under the ordinary column names,
    and everything downstream consumed it happily.

    Two checks, both cheap:

    * the named Voxel layer exists;
    * it carries Label-table provenance (``structure_zkind``), i.e. some node declared it a
      label raster rather than merely leaving an integer array under that name.

    The dtype is left to :func:`nodegraph.bridges._group_reduce`, which refuses a
    non-integer key array with its own message about id truncation."""
    attr = ds.get(Domain.VOXEL, layer)
    if attr is None:
        have = sorted({k[2] for k in ds.attributes if k[0] is Domain.VOXEL})
        raise ValueError(
            f"{node}: no Voxel layer {layer!r} on the input Dataset"
            + (f" (it carries {have})" if have else " (it carries no Voxel layers)")
            + " — run analysis.segment / analysis.label upstream, or point the layer "
              "socket at the right name.")
    zk = ds.structure_zkind(Domain.LABEL, layer)
    if zk is None:
        labelled = sorted({k[1] for k in ds.attributes
                           if k[0] is Domain.LABEL and k[1]})
        raise ValueError(
            f"{node}: the Voxel layer {layer!r} is not a LABEL raster — it carries no "
            f"Label table, so its non-zero voxels are one undivided region, not objects. "
            f"Measuring it would report a single row for the whole foreground under the "
            f"ordinary per-object column names. "
            + (f"Label instances on this Dataset: {labelled}. " if labelled else "")
            + "Run analysis.label (connected components) or analysis.segment on it first; "
              "those write the raster and its table together under one name.")
    return np.asarray(attr.values), zk
# ── Label regions → Points (one representative dot per region) ─────────────────
#
# The Label→Point CONSTRUCTIVE hop, and the one route the structure spine did not have.
# `transform.transfer_structure` moves an ATTRIBUTE between domains but never creates rows,
# and the geometric Label↔Point bridges (`bridges.points_in_label` / `containing_label`)
# need the points to exist already. So "turn each detected region into a single dot" — the
# seed set every point-cloud consumer wants (`analysis.cluster_points`, `analysis.tessellate`,
# `track.link`'s point target, `analysis.voronoi` below) — had to be done outside the graph.
#
# Dimensionality is INHERITED from the Label instance's own `z_kind` provenance
# (`wire-node-v2` §7b), not chosen with a lever: per-plane 2D labels must yield per-plane
# points, and a lever could disagree with the segmentation that produced them.


def _label_centroids(vol: np.ndarray, ids: np.ndarray,
                     weights: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-label centroid + voxel count of an integer label array, in **voxel index** coords.

    Returns ``(centroids (K, vol.ndim), counts (K,))`` aligned with ``ids``, which must be
    sorted ascending and contain every non-zero value present in ``vol`` (``np.unique``
    gives both).

    ``weights`` (same shape as ``vol``) makes it an intensity-weighted centre of mass,
    falling back to the unweighted centroid for any label whose weights sum to ``<= 0`` — a
    fully-dark region still gets a real position instead of a NaN that would poison every
    downstream distance.

    ``np.bincount`` rather than ``scipy.ndimage.center_of_mass``: this is one vectorized
    pass per axis over the foreground voxels only, with no per-label Python iteration, and
    the per-label loop is exactly what makes ``center_of_mass`` slow on a segmentation with
    thousands of regions — which is the normal case here, once per (m,t,c) unit."""
    k = len(ids)
    flat = np.asarray(vol).reshape(-1)
    # `> 0`, not `!= 0`: a negative value is not a label id, and `searchsorted` would file it
    # under the FIRST id rather than skip it — silently dragging region 1's centroid.
    nz = np.flatnonzero(flat > 0)
    if k == 0 or nz.size == 0:
        return np.zeros((k, np.asarray(vol).ndim), dtype=float), np.zeros(k, dtype=np.int64)
    slot = np.searchsorted(ids, flat[nz])
    counts = np.bincount(slot, minlength=k).astype(np.int64)
    coords = np.unravel_index(nz, np.asarray(vol).shape)
    plain = np.column_stack([
        np.bincount(slot, weights=c.astype(float), minlength=k) / np.maximum(counts, 1)
        for c in coords])
    if weights is None:
        return plain, counts
    w = np.asarray(weights, dtype=float).reshape(-1)[nz]
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    wsum = np.bincount(slot, weights=w, minlength=k)
    weighted = np.column_stack([
        np.bincount(slot, weights=w * c.astype(float), minlength=k)
        / np.maximum(wsum, 1e-12) for c in coords])
    return np.where((wsum > 0.0)[:, None], weighted, plain), counts
