"""labels — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Optional, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain

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
