"""field_nodes — the shared contract of the ``field.*`` nodes (Field Gradient, Strain,
Rotation & Curl, Volume Change, Stress).

A **field layer** is a Point table whose rows are the nodes of a measured displacement or
velocity field — what ``analysis.dic_correlate`` / ``analysis.piv`` / ``analysis.dvc_field``,
the staged DIC / PIV / DVC cards, ``analysis.track_field`` and ``io.load_analysis`` write:

* a displacement ``disp_y, disp_x[, disp_z]`` in µm, and/or a velocity ``vy, vx[, vz]`` in
  µm/s;
* positions ``y, x`` in pixels and ``z`` either a plane INDEX (``z_kind = plane_index``: every
  (m, t, c, z) plane is its own 2-D field) or a measured depth in plane steps (``subpixel``: a
  (m, t, c) frame is one volume). The dimensionality is INHERITED from that ``z_kind`` (§7b),
  never a lever that could disagree with how the field was produced.

Every Field node ADDS COLUMNS TO THAT SAME LAYER — as ``dic.strain`` does — so one table
carries the whole chain (displacement → gradient → strain → stress) and the Viewer's Field
overlay can colour by any of it. The gradient is the hinge: Field Gradient writes
``grad_<i><j> = d u_i / d X_j`` (image axes, slowest first) and every later card reads it,
which is why a later card on a layer without it refuses rather than silently differentiating
with settings nobody chose. What each card did is recorded per layer under
:data:`FIELD_MARK`; each column's unit under the Dataset's units table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import InString, Mode
from nodegraph.units import with_unit

from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer

#: per-layer record of what the Field cards did: ``{layer: {"gradient": {...}, ...}}``
FIELD_MARK = "__field_mechanics__"
AXES: Dict[int, Tuple[str, ...]] = {2: ("y", "x"), 3: ("z", "y", "x")}
#: the metadata keys whose value ``previous_frame`` marks an INCREMENT field
_REFERENCE_KEYS = ("dic_reference_mode", "piv_reference_mode", "dvc_reference_mode",
                   "of_reference_mode", "track_field_reference")


def vector_columns(axes: Sequence[str], quantity: str) -> List[str]:
    """``disp_<a>`` for a displacement, ``v<a>`` for a velocity."""
    return [f"disp_{a}" for a in axes] if quantity == "displacement" else [f"v{a}" for a in axes]


def grad_columns(axes: Sequence[str]) -> List[str]:
    return [f"grad_{i}{j}" for i in axes for j in axes]


def _has(cols: Iterable[str], names: Iterable[str]) -> bool:
    s = set(cols)
    return all(n in s for n in names)


def table(ds: Dataset, layer: str) -> Dict[str, np.ndarray]:
    return {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
            if a.layer == layer}


def field_layers(ds: Dataset) -> List[str]:
    """The Point layers carrying an in-plane displacement or velocity."""
    out = []
    for name in _point_layers(ds):
        cols = {a.name for a in ds.layers_on(Domain.POINT) if a.layer == name}
        if _has(cols, ("disp_y", "disp_x")) or _has(cols, ("vy", "vx")):
            out.append(name)
    return out


def gradient_layers(ds: Dataset) -> List[str]:
    out = []
    for name in _point_layers(ds):
        cols = {a.name for a in ds.layers_on(Domain.POINT) if a.layer == name}
        if _has(cols, grad_columns(AXES[2])):
            out.append(name)
    return out


# ── sockets and modes every Field card carries ─────────────────────────────────

def source_socket() -> Any:
    return InString(
        "source", "Field layer", field=False, default="", layer_in=Domain.POINT,
        description=
        "Which Point table holds the field — the one a DIC / PIV / DVC node, a staged card, "
        "Track Field or Load Analysis wrote (`dic`, `piv`, `dvc`, …). Blank picks the only "
        "field on the wire; with two, name one. The card writes its columns BACK ONTO THE "
        "SAME ROWS, so every later card and the Viewer's Field overlay see the whole chain "
        "on one table.")


QUANTITY_MODE = Mode(
    "quantity", ["auto", "displacement", "velocity"], default="auto", label="Field",
    description="Which vector the gradient is taken of. The table usually says: a correlation "
                "writes a displacement, PIV can add a velocity beside it.",
    choice_docs={
        "auto": "The displacement (disp_*) when the table has one, else the velocity (v*). "
                "PIV with velocity on carries both, and this takes the displacement.",
        "displacement": "disp_* in µm: the gradient is dimensionless, F = I + grad, and the "
                        "finite strain measures, rotation and volume change all apply.",
        "velocity": "v* in µm/s: the velocity gradient (1/s). Strain becomes the strain RATE, "
                    "curl the vorticity, volume change the divergence; the finite measures "
                    "and a solid's stress do not apply and are refused."})

POSITIONS_MODE = Mode(
    "positions", ["auto", "reference", "current"], default="auto", label="Positions",
    description="Whether a row's y/x are where the material WAS (the reference "
                "configuration) or where it IS now. The gradient must be taken in the "
                "reference configuration for F = I + grad to hold.",
    choice_docs={
        "auto": "Reference for a correlation grid (DIC / PIV / DVC nodes sit on the reference "
                "image), current for a tracked field (a track_id column: Track Field, the "
                "linkers), whose rows move with the particles.",
        "reference": "Fit at y/x as given. Right for every correlation grid; on tracks it "
                     "would fit the SPATIAL gradient, which turns Green-Lagrange into Almansi.",
        "current": "Fit at y/x minus the displacement — where each particle started. Right for "
                   "tracks; on a correlation grid it would move nodes that never moved."})


def layer_announce(params: Mapping, _modes: Mapping) -> Tuple[Tuple[Domain, str], ...]:
    """``extra_layers``: the card writes into the layer ``source`` names (total)."""
    name = str((params or {}).get("source") or "")
    return ((Domain.POINT, name),) if name else ()


def declared_layer(params: Mapping, incoming: Sequence[Tuple[Domain, str, str]],
                   marker: Sequence[str] = ("disp_y", "vy", "grad_yy")) -> str:
    """The layer the card writes at edit time: the explicit ``source``, else the only
    incoming Point layer that carries one of ``marker`` (the compute's own candidate rule —
    mesh tables carry none of them). Total."""
    explicit = str((params or {}).get("source") or "").strip()
    if explicit:
        return explicit
    known = list(dict.fromkeys(lyr for d, lyr, c in incoming
                               if d is Domain.POINT and c in marker))
    return known[0] if len(known) == 1 else ""


# ── reading a field ───────────────────────────────────────────────────────────

@dataclass
class FieldRows:
    layer: str
    cols: Dict[str, np.ndarray]
    z_kind: str
    axes: Tuple[str, ...]
    coords: np.ndarray            # (N, D) index units (z plane steps, y/x px)
    scale: np.ndarray             # (D,) µm per index unit
    groups: List[np.ndarray]      # row indices of each independent field
    incremental: bool
    note: str = ""

    @property
    def d(self) -> int:
        return len(self.axes)

    @property
    def n(self) -> int:
        return self.coords.shape[0]


def _calib(ctx: EvalContext, is_3d: bool, node: str) -> np.ndarray:
    px = ctx.calib("pixel_size_um")
    zs = ctx.calib("z_step_um") if is_3d else 1.0
    if not px or (is_3d and not zs):
        raise ValueError(
            f"{node}: a gradient is µm of displacement per µm of position, and this Dataset "
            f"declares pixel_size_um={px!r}" + (f" / z_step_um={zs!r}" if is_3d else "")
            + ". A placeholder 1.0 would differentiate per PIXEL under a per-µm label (and in "
              "3-D distort every z term by the anisotropy). Type the missing value into the "
              "Load card that opened this file (Parameters, in µm).")
    px = float(px)
    return np.array([float(zs), px, px]) if is_3d else np.array([px, px])


def _groups(cols: Mapping[str, np.ndarray], is_3d: bool, n: int) -> List[np.ndarray]:
    keys = [np.rint(np.asarray(cols[k], dtype=float)).astype(np.int64)
            for k in ("m", "t", "c") if k in cols]
    if not is_3d and "z" in cols:
        keys.append(np.rint(np.asarray(cols["z"], dtype=float)).astype(np.int64))
    if not keys or n == 0:
        return [np.arange(n)] if n else []
    K = np.stack(keys, axis=1)
    order = np.lexsort(tuple(K[:, k] for k in range(K.shape[1] - 1, -1, -1)))
    Ks = K[order]
    bounds = np.flatnonzero(np.any(Ks[1:] != Ks[:-1], axis=1)) + 1
    return [g for g in np.split(order, bounds) if g.size]


def _incremental(md: Mapping[str, Any]) -> bool:
    return any(str(md.get(k) or "") == "previous_frame" for k in _REFERENCE_KEYS)


def read_rows(ctx: EvalContext, ds: Dataset, node: str, candidates: List[str],
              remedy: str) -> FieldRows:
    layer, note = _resolve_layer(
        candidates, ctx.layer("source"), node=node, socket="source", what="field table",
        where="the `data` input", remedy=remedy, ctx=ctx)
    cols = table(ds, layer)
    zk = ds.structure_zkind(Domain.POINT, layer) or (
        "subpixel" if ("disp_z" in cols or "vz" in cols or "grad_zz" in cols) else "plane_index")
    is_3d = zk == "subpixel"
    axes = AXES[3 if is_3d else 2]
    n = len(next(iter(cols.values()))) if cols else 0
    coords = np.column_stack([np.asarray(cols[a], dtype=float) for a in axes]) \
        if n else np.zeros((0, len(axes)))
    return FieldRows(layer, cols, zk, axes, coords, _calib(ctx, is_3d, node),
                     _groups(cols, is_3d, n), _incremental(ds.metadata), note)


def quantity_of(rows: FieldRows, mode: str, node: str) -> str:
    has_d = _has(rows.cols, vector_columns(rows.axes, "displacement"))
    has_v = _has(rows.cols, vector_columns(rows.axes, "velocity"))
    q = mode if mode in ("displacement", "velocity") else ("displacement" if has_d else
                                                           "velocity")
    if not (has_d if q == "displacement" else has_v):
        want = ", ".join(vector_columns(rows.axes, q))
        raise ValueError(
            f"{node}: the field table {rows.layer!r} has no {q} ({want})"
            + (" — this is a 3-D field (z_kind subpixel) and needs the z component too"
               if rows.d == 3 else "")
            + f"; its columns are {', '.join(sorted(rows.cols)[:24])}.")
    return q


def mark(ds: Dataset, layer: str) -> Dict[str, Any]:
    return dict((ds.metadata.get(FIELD_MARK) or {}).get(layer) or {})


def read_gradient(ctx: EvalContext, ds: Dataset, node: str
                  ) -> Tuple[FieldRows, np.ndarray, str]:
    """``(rows, G (N, D, D), quantity)`` from a layer carrying ``grad_*``, or a refusal
    that names the missing card. A gradient no Field Gradient card wrote (the staged DIC
    cards' own ``grad_*``) is a displacement gradient, and the rail says so."""
    cands = gradient_layers(ds)
    if not cands:
        raise ValueError(
            f"{node}: no Point table on the `data` input carries a displacement gradient "
            f"(grad_yy, grad_yx, grad_xy, grad_xx …). This card derives everything it writes "
            f"from that gradient — put a Field Gradient card before this one (its method and "
            f"radius decide how the field is differentiated, which is a choice this card "
            f"should not make silently).")
    rows = read_rows(ctx, ds, node, cands,
                     remedy="put a Field Gradient card before this one")
    names = grad_columns(rows.axes)
    if not _has(rows.cols, names):
        raise ValueError(
            f"{node}: {rows.layer!r} is a 3-D field but carries only the in-plane gradient; "
            f"re-run Field Gradient on it (it writes all nine grad_* columns for a volume).")
    d = rows.d
    G = np.stack([np.asarray(rows.cols[c], dtype=float) for c in names], axis=1)
    G = G.reshape(-1, d, d)
    info = mark(ds, rows.layer).get("gradient") or {}
    q = str(info.get("quantity") or "")
    if not q:
        q = "displacement"
        ctx.progress(0, 1, f"{rows.layer!r}: using the gradient its own solver wrote (no "
                           f"Field Gradient card ran) — read as a displacement gradient")
    return rows, G, q


# ── writing back ──────────────────────────────────────────────────────────────

def write(ds: Dataset, layer: str, cols: Mapping[str, np.ndarray],
          units: Mapping[str, str], stage: str, info: Mapping[str, Any]) -> Dataset:
    """Add ``cols`` to ``layer`` (same rows, same order), record each one's unit, and the
    card's ``info`` under :data:`FIELD_MARK` ``[layer][stage]``."""
    for name, vals in cols.items():
        ds = ds.with_layer(Domain.POINT, name, np.asarray(vals), layer=layer)
    for name, unit in units.items():
        ds = with_unit(ds, Domain.POINT, name, unit, layer=layer)
    marks = dict(ds.metadata.get(FIELD_MARK) or {})
    entry = dict(marks.get(layer) or {})
    entry[stage] = dict(info)
    marks[layer] = entry
    return ds.with_metadata(**{FIELD_MARK: marks})


def tensor_columns(prefix: str, T: np.ndarray, axes: Sequence[str]) -> Dict[str, np.ndarray]:
    """``<prefix>_<i><j>`` for every component of ``T`` ``(N, D, D)``."""
    return {f"{prefix}_{ai}{aj}": T[:, i, j]
            for i, ai in enumerate(axes) for j, aj in enumerate(axes)}


__all__ = [
    "FIELD_MARK", "AXES", "vector_columns", "grad_columns", "table", "field_layers",
    "gradient_layers", "source_socket", "QUANTITY_MODE", "POSITIONS_MODE",
    "layer_announce", "declared_layer", "FieldRows", "read_rows", "quantity_of", "mark",
    "read_gradient", "write", "tensor_columns",
]
