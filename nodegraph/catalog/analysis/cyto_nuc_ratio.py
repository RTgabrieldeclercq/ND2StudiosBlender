"""Cyto / Nuc Ratio (``analysis.cyto_nuc_ratio``) — a cytoplasm region per nucleus (a ring
grown around it, or another channel's cell segmentation with the nuclei cut out) and, for
every wired signal, the cytoplasm-to-nucleus intensity ratio of each cell (2026-10-22)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.bridges import voxel_to_label
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import map_units
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import on_layer, resolved_layer
from nodegraph.catalog._shared.labels import (
    _label_centroids,
    _label_instances,
    _label_raster,
    _resolve_label_instance,
    _resolve_layer,
    _voxel_layers,
)
from nodegraph.catalog._shared.planes import _each_plane
from nodegraph.catalog._shared.progress import _parallel_progress
from nodegraph.catalog._shared.sampling import _require_same_grid
from nodegraph.catalog._shared.units import to_pixels_v2

#: How many signal sockets the card can grow (``signal``, ``signal_2`` … ``signal_8``).
MAX_SIGNALS = 8
#: The signal sockets, in slot order — slot k's columns are named after its channel, or
#: ``signal<k>``.
SIGNAL_SOCKETS: Tuple[str, ...] = ("signal",) + tuple(
    f"signal_{i}" for i in range(2, MAX_SIGNALS + 1))
_NAME_BAD = re.compile(r"[^0-9A-Za-z]+")


# ── naming: one rule for the compute and the column declaration ─────────────────────────

def _cnr_signal_names(slots: Sequence[int], channel_names: Sequence[Optional[Sequence[Any]]],
                      typed: Any) -> List[str]:
    """The column prefix of each measured signal, in slot order.

    ``slots`` are the 1-based slot numbers that are measured; ``channel_names[i]`` is the
    ``channel_names`` metadata of slot ``slots[i]``'s Dataset (``None`` when unknown).
    ``typed`` is the ``signal_names`` param: comma-separated, entry k naming slot k; a blank
    entry (or none) takes the signal's single channel name, else ``signal<k>``. Every name is
    made column-safe (runs of anything but letters and digits become ``_``) and unique
    (``GFP``, ``GFP_2``), so two taps of the same channel cannot write one column."""
    given = [s.strip() for s in str(typed or "").split(",")]
    out: List[str] = []
    for slot, chans in zip(slots, channel_names):
        name = given[slot - 1] if slot - 1 < len(given) else ""
        if not name:
            names = [str(c) for c in (chans or ()) if str(c).strip()]
            name = names[0] if len(names) == 1 else ""
        name = _NAME_BAD.sub("_", name).strip("_") or f"signal{slot}"
        base, n = name, 2
        while name in out:
            name = f"{base}_{n}"
            n += 1
        out.append(name)
    return out


def _cnr_ratio_column(prefix: str, ratio: str) -> str:
    return f"{prefix}_cyto_nuc_ratio" if ratio == "cyto_over_nuc" else f"{prefix}_nuc_cyto_ratio"


def _cnr_signal_columns(prefix: str, stat: str, background: str, ratio: str) -> List[str]:
    """The nucleus-table columns one signal writes, in order."""
    cols = [f"{prefix}_nuc_{stat}", f"{prefix}_cyto_{stat}"]
    if background == "outside":
        cols.append(f"{prefix}_background")
    cols.append(_cnr_ratio_column(prefix, ratio))
    return cols


# ── the cytoplasm of one unit (a plane, or a volume) ────────────────────────────────────

def _cnr_homes(nuc: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each nucleus's HOME cell — the cell label covering most of its voxels (ties: the
    lower cell id). Returns ``(nucleus ids ascending, home cell ids)``; a nucleus touching no
    cell is absent."""
    both = (nuc > 0) & (cells > 0)
    if not both.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    n = nuc[both].astype(np.int64)
    c = cells[both].astype(np.int64)
    span = int(c.max()) + 1
    keys, cnt = np.unique(n * span + c, return_counts=True)   # one int key per pair
    un, uc = keys // span, keys % span
    order = np.lexsort((uc, -cnt, un))               # by nucleus, most voxels, lower cell
    un, uc = un[order], uc[order]
    first = np.ones(un.size, dtype=bool)
    first[1:] = un[1:] != un[:-1]
    return un[first], uc[first]


def _cnr_unit_cytoplasm(nuc: np.ndarray, cells: Optional[np.ndarray], *, mode: str,
                        cells_are_labels: bool, sampling: Tuple[float, ...], gap_um: float,
                        width_um: float) -> np.ndarray:
    """The cytoplasm label array of one unit: each cytoplasm voxel holds the id of the
    nucleus it belongs to. Nuclear voxels are never cytoplasm — of their own cell or any
    other — and neither is anything within ``gap_um`` of a nucleus.

    ``ring``: every voxel whose distance to the NEAREST nucleus lies in
    ``(gap_um, gap_um + width_um]`` (µm, along the true anisotropic voxel spacing), owned by
    that nearest nucleus — neighbouring rings meet halfway and never overlap.

    ``cells`` with a LABEL raster: each nucleus is homed in the cell covering most of it
    (:func:`_cnr_homes`); a cell's non-nuclear voxels go to its home nucleus, split by
    nearest nucleus when it holds several, and to nobody when it holds none. ``cells`` with
    a plain MASK: the mask's non-nuclear voxels go to their nearest nucleus."""
    from scipy import ndimage as ndi
    out = np.zeros(nuc.shape, dtype=np.int64)
    if not (nuc > 0).any():
        return out
    bg = nuc <= 0
    dist, idx = ndi.distance_transform_edt(bg, sampling=sampling, return_indices=True)
    nearest = nuc[tuple(idx)].astype(np.int64)
    if mode == "ring":
        sel = bg & (dist > gap_um) & (dist <= gap_um + width_um)
        out[sel] = nearest[sel]
        return out
    fg = bg & (cells > 0) & (dist > gap_um)
    if not fg.any():
        return out
    if not cells_are_labels:
        out[fg] = nearest[fg]
        return out
    hn, hc = _cnr_homes(nuc, cells)
    if hn.size == 0:
        return out                                   # no nucleus lies inside any cell
    flat = out.reshape(-1)
    fv = np.flatnonzero(fg.reshape(-1))
    near = nearest.reshape(-1)[fv]
    here = cells.reshape(-1)[fv].astype(np.int64)
    pos = np.minimum(np.searchsorted(hn, near), hn.size - 1)
    home_of_near = np.where(hn[pos] == near, hc[pos], 0)
    ok = home_of_near == here                        # the nearest nucleus lives in this cell
    flat[fv[ok]] = near[ok]
    rest, rest_cell = fv[~ok], here[~ok]
    if rest.size == 0:
        return out
    cell_ids, n_homed = np.unique(hc, return_counts=True)
    single = cell_ids[n_homed == 1]
    if single.size:                                  # one nucleus: the whole cell is its
        only = np.isin(hc, single)
        s_cell, s_nuc = hc[only], hn[only]
        o = np.argsort(s_cell)
        s_cell, s_nuc = s_cell[o], s_nuc[o]
        p = np.minimum(np.searchsorted(s_cell, rest_cell), s_cell.size - 1)
        hit = s_cell[p] == rest_cell
        flat[rest[hit]] = s_nuc[p[hit]]
        rest, rest_cell = rest[~hit], rest_cell[~hit]
    multi = cell_ids[n_homed > 1]
    if rest.size == 0 or multi.size == 0:
        return out                                   # what is left lies in anucleate cells
    coords = np.unravel_index(rest, nuc.shape)
    cell_box = ndi.find_objects(cells.astype(np.int64))
    nuc_box = ndi.find_objects(nuc.astype(np.int64))
    for c in multi.tolist():                         # several nuclei: nearest of THEIRS
        mine = rest_cell == c
        if not mine.any():
            continue
        homed = hn[hc == c]
        # the cell's box grown to hold its nuclei (one may poke out of the cell)
        boxes = [cell_box[c - 1]] + [nuc_box[int(h) - 1] for h in homed]
        lo = [min(b[d].start for b in boxes if b is not None) for d in range(nuc.ndim)]
        hi = [max(b[d].stop for b in boxes if b is not None) for d in range(nuc.ndim)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub = nuc[box]
        local = np.where(np.isin(sub, homed), sub, 0)
        _d, li = ndi.distance_transform_edt(local == 0, sampling=sampling,
                                            return_indices=True)
        got = local[tuple(li)]
        rel = tuple(cc[mine] - l for cc, l in zip(coords, lo))
        flat[rest[mine]] = got[rel]
    return out


# ── the compute ─────────────────────────────────────────────────────────────────────────

def _cnr_table(ds: Dataset, layer: str) -> Dict[str, np.ndarray]:
    return {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.LABEL)
            if a.layer == layer}


def _cnr_gather(ctx: EvalContext, prov: Any, ax: Any, note: str) -> np.ndarray:
    img = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), dtype=float)
    tick = _parallel_progress(ctx, ax.m * ax.t * ax.z * ax.c, note, frames=ax.t)

    def _read(unit):
        m, t, z, c = unit
        img[m, t, z, c] = prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)
        tick()

    map_units(_read, list(_each_plane(ax)))
    return img


def _compute_cyto_nuc_ratio(ctx: EvalContext) -> Dataset:
    """A cytoplasm region per nucleus, and the cytoplasm-to-nucleus ratio of every signal.

    **The cytoplasm** (``cytoplasm`` Mode) is built per unit — a plane when the nuclei were
    segmented per plane (``z_kind = plane_index``), a volume when they were segmented in 3D
    (``subpixel``): the dimensionality is INHERITED from the nuclei's own table (§7b), not
    chosen with a lever that could disagree with it.

    * ``ring`` — grown around each nucleus: every voxel between ``gap`` and
      ``gap + ring_width`` µm of the nearest nucleus edge, true Euclidean distance on the
      anisotropic voxel grid, owned by that nearest nucleus so neighbouring rings meet halfway
      and never overlap.
    * ``cells`` — a cell segmentation from another channel (the ``cells`` input, else a
      second layer on ``data``). Where it overlaps the nucleus segmentation, the nucleus wins:
      no nuclear voxel is ever counted as cytoplasm, so the nuclear signal cannot leak into
      the cytoplasm mean and flatten the ratio. A LABEL raster is matched cell-to-nucleus by
      overlap (a nucleus's home cell covers most of it; a cell holding two nuclei is split
      between them by nearest nucleus; a cell holding none is dropped); a plain MASK is
      split by nearest nucleus.

    **The signals.** ``signal`` … ``signal_8`` each take one channel (a channel tap off the
    same file — the grid must match ``data`` voxel for voxel, refused otherwise); with none
    wired, ``data``'s own image is measured. Per signal, per nucleus: the nucleus and
    cytoplasm ``stat`` (mean or median), optionally the ``outside`` background (median of the
    unit's voxels in neither a nucleus nor a cytoplasm region), and the ratio ``(cyto − bg) /
    (nuc − bg)`` (or its inverse) — NaN where the denominator is not positive or a cell has no
    cytoplasm. Columns are named after the signal's channel (``p65_cyto_mean``), or
    ``signal<k>``, or what ``signal_names`` says.

    **Output.** The nucleus Label table gains the columns (plus ``nuc_voxels`` /
    ``cyto_voxels``); the cytoplasm is a new label instance under ``name`` — a Voxel raster
    whose ids ARE the nucleus ids (the join is the id) and its Label table (id, m, t, c,
    z, y, x, ``area`` in voxels).

    Resolved spec: category analysis; op ``analysis.cyto_nuc_ratio``; reads VOXEL + LABEL
    (the nuclei instance), adds VOXEL + LABEL; footprint WHOLE_VOLUME (a ring needs the
    whole plane or volume around its nucleus); µm params converted along ``pixel_size_um``
    (and ``z_step_um`` for a 3D nucleus on a stack); no numba — the work is two scipy EDTs
    and numpy group reductions."""
    ds = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {}) or {}
    mode = str(modes.get("cytoplasm", "ring"))
    stat = str(modes.get("stat", "mean"))
    background = str(modes.get("background", "none"))
    ratio = str(modes.get("ratio", "cyto_over_nuc"))
    name = ctx.layer("name")

    nuc_layer, nuc_note = _resolve_label_instance(
        ds, ctx.layer("nuclei"), node="cyto/nuc ratio", socket="nuclei",
        remedy="the nuclei must be a label raster AND its table — run analysis.segment "
               "(or analysis.label) on the nuclear channel upstream")
    nuc6, zk = _label_raster(ds, nuc_layer, node="cyto/nuc ratio")
    nuc6 = np.asarray(nuc6)
    if name == nuc_layer:
        raise ValueError(f"cyto/nuc ratio: the cytoplasm layer `name` is {name!r}, the same "
                         f"as the nuclei it is built around — give it another name")
    volumetric = zk == "subpixel"

    # ── where the cytoplasm comes from ──────────────────────────────────────────────
    cds = ctx.input("cells")
    cell6: Optional[np.ndarray] = None
    cells_are_labels = False
    cell_note = ""
    if mode == "ring":
        if cds is not None:
            raise ValueError(
                "cyto/nuc ratio: a cell segmentation is wired into `cells` but Cytoplasm is "
                "`ring`, which grows the cytoplasm from the nuclei and would ignore it — "
                "switch Cytoplasm to `cells`, or unwire it")
    else:
        src = ds if cds is None else cds
        if cds is not None:
            _require_same_grid(ds, cds, socket="cells",
                               consequence="give each nucleus the cytoplasm of a different "
                                           "place")
        want = ctx.layer("cells_layer")
        if want and src.get(Domain.VOXEL, want) is not None:
            cell_layer = want
        else:
            keep = (lambda n: n != nuc_layer) if cds is None else (lambda n: True)
            cands = [n for n in _label_instances(src) if keep(n)] or \
                [n for n in _voxel_layers(src) if keep(n)]
            cell_layer, cell_note = _resolve_layer(
                cands, want, node="cyto/nuc ratio", socket="cells_layer",
                what="cell segmentation",
                where="the `cells` input" if cds is not None else "the `data` input",
                remedy="segment the cell (membrane or cytoplasm) channel and wire it into "
                       "`cells`, or switch Cytoplasm to `ring`")
        cell6 = np.asarray(src.get(Domain.VOXEL, cell_layer).values)
        cells_are_labels = src.structure_zkind(Domain.LABEL, cell_layer) is not None

    # ── µm → the EDT's sampling ─────────────────────────────────────────────────────
    px = float(ctx.calib("pixel_size_um") or 0.1)
    gap_um = max(0.0, float(ctx.params.get("gap", 0.0) or 0.0))
    width_um = (float(ctx.params.get("ring_width", 3.0) or 0.0) if mode == "ring" else 0.0)
    if mode == "ring" and to_pixels_v2(width_um, "um", pixel_size_um=px) < 1.0:
        raise ValueError(
            f"cyto/nuc ratio: `ring_width` is {width_um:g} µm, which is "
            f"{to_pixels_v2(width_um, 'um', pixel_size_um=px):.3g} px at pixel_size_um="
            f"{px:g} — below one voxel, so no ring would be drawn. Raise it above {px:g} µm "
            f"(a 10x plate at ~1.7 µm/px needs a wider ring than this catalog's defaults "
            f"assume).")
    if volumetric:
        zs = float(ctx.calib("z_step_um") or 0.5) if ax.z > 1 else 1.0
        sampling: Tuple[float, ...] = (zs, px, px)
        units = [(m, t, None, c) for m in range(ax.m) for t in range(ax.t)
                 for c in range(ax.c)]
    else:
        sampling = (px, px)
        units = [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
                 for z in range(ax.z) for c in range(ax.c)]

    # ── the cytoplasm raster ────────────────────────────────────────────────────────
    cyto6 = np.zeros(nuc6.shape, dtype=np.int64)
    for note in (nuc_note, cell_note):
        if note:
            ctx.progress(0, len(units), "using " + note, frames=ax.t)
    tick = _parallel_progress(ctx, len(units), f"building cytoplasm ({mode})", frames=ax.t)

    def _sel(m, t, z, c):
        return (m, t, slice(None), c) if z is None else (m, t, z, c)

    def _build(unit):
        key = _sel(*unit)
        cyto6[key] = _cnr_unit_cytoplasm(
            np.asarray(nuc6[key]), None if cell6 is None else np.asarray(cell6[key]),
            mode=mode, cells_are_labels=cells_are_labels, sampling=sampling,
            gap_um=gap_um, width_um=width_um)
        tick()

    map_units(_build, units)

    # ── the signals ─────────────────────────────────────────────────────────────────
    wired = [(i + 1, ctx.input(s)) for i, s in enumerate(SIGNAL_SOCKETS)]
    wired = [(k, sd) for k, sd in wired if sd is not None]
    if not wired:
        wired = [(1, ds)]                    # nothing wired: measure the nuclei's own image
    for k, sd in wired:
        if sd is not ds:
            _require_same_grid(ds, sd, socket=SIGNAL_SOCKETS[k - 1],
                               consequence="report each cell's signal from the wrong place")
        if getattr(sd, "image", None) is None:
            raise ValueError(
                f"cyto/nuc ratio: `{SIGNAL_SOCKETS[k - 1]}` carries no image to measure — "
                f"wire the channel whose signal you want (a channel tap of the same file)")
    prefixes = _cnr_signal_names([k for k, _sd in wired],
                                 [sd.metadata.get("channel_names") for _k, sd in wired],
                                 ctx.params.get("signal_names", ""))

    table = _cnr_table(ds, nuc_layer)
    row_ids = np.asarray(table.get("id", np.zeros(0)), dtype=np.int64)
    n_rows = row_ids.size

    def _by_id(ids: np.ndarray, vals: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(n_rows, fill, dtype=float)
        if ids.size and n_rows:
            p = np.minimum(np.searchsorted(ids, row_ids), ids.size - 1)
            hit = ids[p] == row_ids
            out[hit] = vals[p[hit]]
        return out

    ones = np.ones(nuc6.shape, dtype=float)
    new: Dict[str, np.ndarray] = {
        "nuc_voxels": _by_id(*voxel_to_label(ones, nuc6, "count"), 0.0).astype(np.int64),
        "cyto_voxels": _by_id(*voxel_to_label(ones, cyto6, "count"), 0.0).astype(np.int64),
    }
    del ones
    # each row's unit, for its background
    rm = np.asarray(table.get("m", np.zeros(n_rows)), dtype=np.int64)
    rt = np.asarray(table.get("t", np.zeros(n_rows)), dtype=np.int64)
    rc = np.asarray(table.get("c", np.zeros(n_rows)), dtype=np.int64)
    rz = (np.zeros(n_rows, dtype=np.int64) if volumetric else
          np.rint(np.asarray(table.get("z", np.zeros(n_rows)), dtype=float)).astype(np.int64))
    outside = None
    if background == "outside":
        outside = (nuc6 <= 0) & (cyto6 <= 0)
        if cell6 is not None:
            outside &= cell6 <= 0
    for (k, sd), prefix in zip(wired, prefixes):
        img = _cnr_gather(ctx, sd.image, ax, f"reading {prefix}")
        nuc_v = _by_id(*voxel_to_label(img, nuc6, stat), np.nan)
        cyto_v = _by_id(*voxel_to_label(img, cyto6, stat), np.nan)
        bg = np.zeros(n_rows, dtype=float)
        cols = _cnr_signal_columns(prefix, stat, background, ratio)
        new[cols[0]] = nuc_v
        new[cols[1]] = cyto_v
        if outside is not None:
            level: Dict[tuple, float] = {}
            for (m, t, z, c) in units:
                key = _sel(m, t, z, c)
                vals = img[key][outside[key]]
                level[(m, t, 0 if z is None else z, c)] = (
                    float(np.median(vals)) if vals.size else np.nan)
            bg = np.array([level.get((int(a), int(b), int(z), int(c)), np.nan)
                           for a, b, z, c in zip(rm, rt, rz, rc)], dtype=float)
            new[f"{prefix}_background"] = bg
        num, den = ((cyto_v - bg, nuc_v - bg) if ratio == "cyto_over_nuc"
                    else (nuc_v - bg, cyto_v - bg))
        with np.errstate(divide="ignore", invalid="ignore"):
            new[cols[-1]] = np.where(den > 0, num / den, np.nan)
        del img

    # ── the cytoplasm's own table: ids are the nucleus ids ──────────────────────────
    cids = np.unique(cyto6)
    cids = cids[cids > 0]
    cen, area = _label_centroids(cyto6, cids)
    if cids.size:
        cyto_table = {
            "id": cids.astype(np.int64),
            "m": np.rint(cen[:, 0]).astype(np.int64),
            "t": np.rint(cen[:, 1]).astype(np.int64),
            "c": np.rint(cen[:, 3]).astype(np.int64),
            "z": cen[:, 2] if volumetric else np.rint(cen[:, 2]).astype(np.int64),
            "y": cen[:, 4], "x": cen[:, 5],
            "area": area.astype(np.int64),
        }
    else:
        cyto_table = {k: np.zeros(0, dtype=np.int64) for k in ("id", "m", "t", "c", "z",
                                                               "area")}
        cyto_table.update(y=np.zeros(0), x=np.zeros(0))
    table.update(new)
    return (ds.with_layer(Domain.VOXEL, name, cyto6)
            .with_structure(StructureTable(Domain.LABEL, cyto_table, layer=name, z_kind=zk))
            .with_structure(StructureTable(Domain.LABEL, table, layer=nuc_layer,
                                           z_kind=zk)))


# ── edit-time declarations ──────────────────────────────────────────────────────────────

def _columns_cyto_nuc_ratio(params, modes, incoming, inputs=()):
    """The nucleus table's new columns — per wired signal, named by the SAME rule the
    compute uses (:func:`_cnr_signal_names`, fed the signal envelopes' ``channel_names``) —
    and the cytoplasm table's. Total: anything unexpected declares nothing."""
    try:
        modes = modes or {}
        stat = str(modes.get("stat", "mean"))
        background = str(modes.get("background", "none"))
        ratio = str(modes.get("ratio", "cyto_over_nuc"))
        envs = dict(inputs or ())
        slots = [i + 1 for i, s in enumerate(SIGNAL_SOCKETS) if s in envs]
        chans = [list((envs[SIGNAL_SOCKETS[k - 1]].metadata or {}).get("channel_names") or [])
                 for k in slots]
        if not slots:
            data_env = envs.get("data")
            slots = [1]
            chans = [list(((data_env.metadata if data_env is not None else {}) or {})
                          .get("channel_names") or [])]
        names = ["nuc_voxels", "cyto_voxels"]
        for p in _cnr_signal_names(slots, chans, (params or {}).get("signal_names", "")):
            names += _cnr_signal_columns(p, stat, background, ratio)
        nuc = resolved_layer(params, "nuclei", incoming, Domain.LABEL)
        cyto = str((params or {}).get("name") or "cytoplasm")
        return (on_layer(Domain.LABEL, nuc, names)
                + on_layer(Domain.LABEL, cyto, ("id", "m", "t", "c", "z", "y", "x", "area")))
    except Exception:  # noqa: BLE001 — an edit-time declaration must never raise
        return ()


_columns_cyto_nuc_ratio.wants_inputs = True     # type: ignore[attr-defined]

_SIGNAL_DOC = ("Another signal to measure: one channel, tapped off the same file as `data` "
               "(Channel Select), so its voxels line up with the nuclei — a different grid "
               "is refused. Each signal adds its own nucleus, cytoplasm and ratio columns, "
               "named after its channel. The next empty slot appears once this one is "
               "wired.")

register_node(
    batch_aware(_compute_cyto_nuc_ratio), op_key="analysis.cyto_nuc_ratio",
    label="Cyto / Nuc Ratio", category="analysis",
    adds_columns=_columns_cyto_nuc_ratio,
    reads_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),
    modes=[
        Mode("cytoplasm", ["ring", "cells"], default="ring", label="Cytoplasm",
             description=
             "Where each nucleus's cytoplasm comes from: a ring grown around the nucleus, or "
             "a cell segmentation made on another channel. Either way no nuclear voxel is "
             "ever counted as cytoplasm.",
             choice_docs={
                 "ring":
                     "Grow a ring around every nucleus: the voxels between `gap` and `gap + "
                     "ring width` µm from its edge, owned by the nearest nucleus so "
                     "neighbouring rings meet halfway and never overlap. Needs only the "
                     "nuclei — the usual choice when there is no membrane or cytoplasm "
                     "stain, at the cost of sampling a fixed shell rather than the real cell.",
                 "cells":
                     "Use a cell segmentation from another channel, wired into `cells` (or a "
                     "second layer on `data`). Where it overlaps a nucleus, the nucleus is "
                     "removed from it, so the ratio compares cytoplasm with nucleus and "
                     "never nucleus with itself. A label raster is matched to nuclei by "
                     "overlap (a cell with two nuclei is split between them, one with none "
                     "is dropped); a plain mask is split by nearest nucleus.",
             }),
        Mode("stat", ["mean", "median"], default="mean", label="Statistic",
             description="How each region's intensity is summarised before the ratio is "
                         "taken — it names the columns (`_mean` / `_median`).",
             choice_docs={
                 "mean":
                     "The average voxel value of the nucleus and of the cytoplasm — the "
                     "conventional translocation measure, and the one that matches "
                     "Measure's `mean_intensity`. A few very bright puncta in the cytoplasm "
                     "pull it up.",
                 "median":
                     "The middle voxel value: unmoved by puncta, saturated voxels or a sliver "
                     "of a neighbour's nucleus at the region's edge, so it reports each "
                     "compartment's bulk level. Prefer it on spotty signals.",
             }),
        Mode("background", ["none", "outside"], default="none", label="Background",
             description="What is subtracted from both compartments before the ratio. A "
                         "camera offset left in both pulls every ratio toward 1.",
             choice_docs={
                 "none":
                     "Nothing: the ratio is of the raw values. Right after a Subtract "
                     "Background node, or when the offset is negligible against the signal.",
                 "outside":
                     "Subtract the median of the voxels in neither a nucleus nor a cytoplasm "
                     "region (nor, under `cells`, a cell), per plane — or per volume for 3D "
                     "nuclei — and per signal, written as `<signal>_background`. Under "
                     "`ring` that region includes any cytoplasm beyond the ring, which "
                     "inflates it on confluent cells.",
             }),
        Mode("ratio", ["cyto_over_nuc", "nuc_over_cyto"], default="cyto_over_nuc",
             label="Ratio",
             description="Which way the ratio is taken; it names the column.",
             choice_docs={
                 "cyto_over_nuc":
                     "Cytoplasm ÷ nucleus (`<signal>_cyto_nuc_ratio`): above 1 when the "
                     "protein sits in the cytoplasm, below 1 when it has entered the nucleus.",
                 "nuc_over_cyto":
                     "Nucleus ÷ cytoplasm (`<signal>_nuc_cyto_ratio`), the N/C convention of "
                     "most nuclear-translocation assays (NF-κB, YAP): rises as the protein "
                     "moves in.",
             }),
    ],
    inputs=[
        InDataset(description=
                  "The nuclei: a Dataset carrying the nuclear label raster AND its table "
                  "(Segmentation on the nuclear channel). Their ids are the rows of the "
                  "result, and their dimensionality decides the rings' — per plane for "
                  "2D-segmented nuclei, a 3D shell for 3D ones. With no signal wired, its "
                  "own image is the one measured."),
        InDataset("cells", label="Cells", view_source=True, passes_domains=False,
                  description=
                  "The cell segmentation from another channel (a membrane or cytoplasm "
                  "stain), read under Cytoplasm = `cells` — a label raster (matched to "
                  "nuclei by overlap) or a plain mask (split by nearest nucleus). Must be on "
                  "the same voxel grid as `data`. Unwired, `cells` mode looks for a second "
                  "layer on `data`; wired under `ring`, it is refused rather than ignored."),
        # all eight in ONE grow group: the card shows Signal 1 alone, and each wire reveals
        # the next empty slot (`GraphDocument._grow_filter`)
        *[InDataset(s, label=f"Signal {i + 1}", view_source=True, passes_domains=False,
                    grow_group="signals",
                    description=(
                        "The signal to measure: one channel, tapped off the same file as "
                        "`data` (Channel Select), so its voxels line up with the nuclei — a "
                        "different grid is refused. Its nucleus, cytoplasm and ratio columns "
                        "are named after its channel. Unwired (and no other signal wired), "
                        "the nuclei's own image is measured." if i == 0 else _SIGNAL_DOC))
          for i, s in enumerate(SIGNAL_SOCKETS)],
        InString("nuclei", "Nuclei", field=False, default="", layer_in=Domain.VOXEL,
                 description=
                 "Which label raster on `data` holds the nuclei. Leave it EMPTY and the only "
                 "label instance on the wire is used (the usual case); name one when the wire "
                 "carries two. One result row per nucleus id, so this choice decides what "
                 "gets measured."),
        InString("cells_layer", "Cell layer", field=False, default="",
                 layer_in=Domain.VOXEL, layer_from="cells",
                 available_in={"cytoplasm": frozenset({"cells"})},
                 description=
                 "Which raster on `cells` is the cell segmentation. Leave it EMPTY and the "
                 "only label instance there is used (else the only raster at all). A label "
                 "raster is matched to nuclei by overlap; a mask is split by nearest nucleus "
                 "— naming the mask instead of the labels of the same segmentation changes "
                 "how multi-nucleate cells are divided. Only read under Cytoplasm = `cells`."),
        InFloat("ring_width", "Ring width", unit="um", field=False, default=3.0,
                pick_kind="distance",
                available_in={"cytoplasm": frozenset({"ring"})},
                description=
                "How thick the cytoplasmic ring is, in µm from where the gap ends — a true "
                "Euclidean distance, the same physical length in every direction. WIDER "
                "samples more cytoplasm (steadier means) but reaches past small cells into "
                "background, which drags `cyto` down and the cyto/nuc ratio with it; "
                "narrower stays inside the cell but on fewer voxels. Neighbouring rings stop "
                "where they meet. Refused below one pixel. Only read under Cytoplasm = "
                "`ring`."),
        InFloat("gap", "Gap", unit="um", field=False, default=0.0,
                description=
                "A band left out between each nucleus and its cytoplasm, in µm. The nuclear "
                "edge is blurred by the optics, so the voxels just outside it carry nuclear "
                "signal; a gap of about one PSF width (~0.5 µm at high NA) keeps that out of "
                "the cytoplasm mean and makes the ratio more contrasty. 0 (the default) "
                "starts the cytoplasm at the first voxel past the nucleus. Applies in both "
                "modes."),
        InString("signal_names", "Signal names", field=False, default="",
                 description=
                 "Column names for the signals, comma-separated in slot order (`p65, "
                 "tubulin`). A blank entry takes the signal's channel name, else "
                 "`signal<k>`. Renaming changes only the column names, never a value."),
        InString("name", "Cytoplasm layer", field=False, default="cytoplasm",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "The name of the cytoplasm label raster and its table. Its ids ARE the "
                 "nucleus ids, so the two tables join on `id`; draw it in the Viewer to check "
                 "the rings before trusting a ratio. Must differ from the nuclei layer."),
    ],
    outputs=[OutDataset()],
    granularity=Granularity.WHOLE_VOLUME,
    description="Cytoplasm around each nucleus — a ring grown from the nuclei, or another "
                "channel's cell segmentation with the nuclei cut out — and, for each wired "
                "signal channel, the nucleus and cytoplasm intensity and their ratio per "
                "cell, as columns on the nucleus table; the cytoplasm is kept as a label "
                "layer whose ids are the nucleus ids.")
