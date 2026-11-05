"""table.register_positions — bring a table's stage positions into another table's stage frame
by the objects the two share: a plate taken off the stage and put back (2026-10-15)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

_RP_DOMAINS = ("label", "point")
_RP_MODELS = ("rigid", "shift")
#: the columns this node adds to every row it passes on
DX_COLUMN, DY_COLUMN, DZ_COLUMN = "reg_dx_um", "reg_dy_um", "reg_dz_um"
RESIDUAL_COLUMN = "reg_residual_um"
ROTATION_COLUMN = "reg_rotation_deg"
_ADDED = (DX_COLUMN, DY_COLUMN, DZ_COLUMN, RESIDUAL_COLUMN, ROTATION_COLUMN)
#: the most row-to-reference vectors the shift vote scores (memory: each is two floats plus
#: a neighbour count)
_RP_MAX_VOTES = 2_000_000


def rp_vote_shift(p: np.ndarray, q: np.ndarray, search: float, tol: float
                  ) -> Optional[np.ndarray]:
    """The XY shift ``t`` (``p + t`` lands on ``q``) that the most row pairs agree on: every
    vector ``q_j - p_i`` no longer than ``search`` votes, a vector scores the number of votes
    within ``tol`` of it, and the best one's neighbourhood is averaged (median). ``None`` when
    no pair is within ``search``. Ties go to the shorter shift; ``ValueError`` when the votes
    would not fit in memory (a dense table and a wide ``search``)."""
    from scipy.spatial import cKDTree
    near_q = cKDTree(q).query_ball_point(p, search)
    count = sum(len(js) for js in near_q)
    if not count:
        return None
    if count > _RP_MAX_VOTES:
        raise ValueError(f"{count} row pairs lie within Search ({search:g}) — too many to "
                         f"vote on; lower Search to just above how far the plate moved")
    v = np.concatenate([q[js] - p[i] for i, js in enumerate(near_q) if js])
    tree = cKDTree(v)
    score = np.asarray(tree.query_ball_point(v, tol, return_length=True))
    best = np.flatnonzero(score == score.max())
    b = best[np.argmin(np.hypot(v[best, 0], v[best, 1]))]
    near = v[tree.query_ball_point(v[b], tol)]
    return np.median(near, axis=0)


def rp_pairs(p: np.ndarray, q: np.ndarray, tol: float) -> List[Tuple[int, int]]:
    """Mutual nearest neighbours of the already-aligned ``p`` and ``q`` closer than ``tol``:
    each row pairs with at most one reference row and the other way round."""
    if not len(p) or not len(q):
        return []
    from scipy.spatial import cKDTree
    dq, jq = cKDTree(q).query(p)
    _dp, ip = cKDTree(p).query(q)
    return [(i, int(j)) for i, (dist, j) in enumerate(zip(dq, jq))
            if dist < tol and int(ip[int(j)]) == i]


def rp_fit(a: np.ndarray, b: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(R, ca, cb)`` with ``a -> R·(a - ca) + cb`` the transform of the paired rows ``a`` onto
    ``b``: least-squares rotation + shift (Kabsch, never a mirror) for ``rigid``, the median
    shift with no rotation for ``shift``."""
    if model == "shift":
        return np.eye(2), np.zeros(2), np.median(b - a, axis=0)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r, ca, cb


def register_xy(p: np.ndarray, q: np.ndarray, *, search: float, tol: float, model: str
                ) -> Dict[str, Any]:
    """Register the XY rows ``p`` onto the reference rows ``q`` (both ``(n, 2)``, finite): the
    voted shift, mutual-nearest pairs within ``tol``, the ``model`` fitted to them, then pairs
    and fit once more under that transform. Returns ``R``, ``ca``, ``cb`` (see :func:`rp_fit`),
    ``pairs`` and each pair's ``residual`` after the fit; raises ``ValueError`` (with the
    numbers) when nothing pairs."""
    t0 = rp_vote_shift(p, q, search, tol)
    if t0 is None:
        raise ValueError(f"no reference row lies within Search ({search:g}) of any row — "
                         f"raise Search to at least how far the plate moved")
    pairs = rp_pairs(p + t0, q, tol)
    need = 2 if model == "rigid" else 1
    if len(pairs) < need:
        raise ValueError(f"only {len(pairs)} row(s) pair within Tolerance ({tol:g}) after the "
                         f"best shift ({t0[0]:.1f}, {t0[1]:.1f}); {model} needs {need}. Widen "
                         f"Tolerance, or check both tables hold the same objects")
    r, ca, cb = rp_fit(p[[i for i, _ in pairs]], q[[j for _, j in pairs]], model)
    for _round in range(2):
        moved = (p - ca) @ r.T + cb
        again = rp_pairs(moved, q, tol)
        if len(again) < need:
            break
        pairs = again
        r, ca, cb = rp_fit(p[[i for i, _ in pairs]], q[[j for _, j in pairs]], model)
    moved = (p - ca) @ r.T + cb
    res = np.array([math.hypot(*(moved[i] - q[j])) for i, j in pairs])
    return {"R": r, "ca": ca, "cb": cb, "pairs": pairs, "residual": res, "vote": t0}


def _rp_column(cols: Mapping[str, np.ndarray], name: str, layer: str, side: str) -> np.ndarray:
    if name not in cols:
        raise ValueError(
            f"Register Positions: the {side} table {layer!r} has no {name!r} column. The stage "
            f"position columns come from Measure (`stage_x_um`, `stage_y_um`, `stage_z_um`, "
            f"written when the image carries a pixel size and a stage log); name other "
            f"columns otherwise. It carries {sorted(cols)}.")
    if TB.is_text(cols[name]):
        raise ValueError(f"Register Positions: {name!r} of the {side} table holds text, not "
                         f"positions.")
    return np.asarray(cols[name], dtype=float)


def _compute_register_positions(ctx: EvalContext) -> Dataset:
    """A table's positions moved into a REFERENCE table's stage frame, fitted on the objects
    both tables contain — the same beads re-imaged after the plate was lifted off the stage
    and put back, a little shifted and turned.

    Resolved spec (2026-10-15, `build-node-v2`):

    * reads ``table`` (``domain`` label / point) on ``data`` and ``ref_table`` (same domain)
      on ``reference``; ``x`` / ``y`` (default ``stage_x_um`` / ``stage_y_um``) place each
      row, in their own units; ``z`` (default ``stage_z_um``, blank = XY only) is shifted;
    * the coarse shift is VOTED, not assumed small: every row-to-reference-row vector no
      longer than ``search`` votes, and the shift most vectors agree on within ``tolerance``
      wins — so a move larger than the objects' spacing still finds its pairs, and objects
      present in only one table cannot outvote the shared ones;
    * pairs are mutual nearest neighbours closer than ``tolerance`` once shifted; ``model``
      ``rigid`` fits a rotation + shift to them (least squares, never a mirror; 2+ pairs),
      ``shift`` the median shift (1+ pair); pairs and fit are refined twice under the fit;
      Z moves by the pairs' median Z difference;
    * EVERY row is moved — paired or not (an object the reference never saw is where the
      transform puts it); a row with no finite x / y keeps its values;
    * the x / y / z columns are REPLACED, so everything downstream that reads `stage_x_um`
      reads the reference frame; ``reg_dx_um`` / ``reg_dy_um`` / ``reg_dz_um`` hold what was
      added to each row (subtract them to get the raw values back), ``reg_residual_um`` how
      far a paired row still is from its twin (NaN unpaired), ``reg_rotation_deg`` the turn;
    * **Landmarks** (2026-10-16, a third session that imaged NEW beads): with the
      ``landmarks`` input wired, the vote, pairs and fit run on ITS table (``landmark_table``,
      same domain and columns) against the reference — the same objects seen in both
      sessions, e.g. the beads of a whole-well scan taken in each — and the transform is then
      applied to every row of ``data``'s table, which need share no object with the
      reference. ``reg_residual_um`` is then NaN on every row (no row was a fitted twin) and
      the landmark pairs' residuals are on the rail; Z moves by the LANDMARK pairs' median
      difference (blank Z when a landmark scan's focus log cannot be trusted);
    * one transform for the whole table (every frame and batch member together); written in
      place (blank ``name``: the same rows of the same image, so a label table keeps its
      raster's name and the Viewer still draws it) or as a new table (a table node's copy,
      refused under a Voxel layer's name); WHOLE_SERIES, no axis or calibration change.
    """
    ds = ctx.inputs[0]
    ref_in = ctx.input("reference")
    if ref_in is None:
        raise ValueError("Register Positions: wire the Reference input — the Dataset whose "
                         "table holds the positions to register onto (the first timepoint's "
                         "beads, say).")
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    model = str(modes.get("model", "rigid"))
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="register positions",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {dom.value} table (Measure writes the stage "
               f"position columns), or switch Domain", ctx=ctx)[0]
    ref_layer = _resolve_layer(
        _structure_layers(ref_in, dom), ctx.layer("ref_table"), node="register positions",
        socket="ref_table", what=f"{dom.value} table", where="the `reference` input",
        remedy=f"wire the Dataset carrying the reference {dom.value} table", ctx=ctx)[0]
    name = ctx.layer("name") or layer
    if name != layer:
        # a COPY under a raster's name would be read as that raster's table; in place, the
        # rows and ids are the table's own, so a label table may keep its raster's name
        TB.refuse_voxel_name(ds, name, "Register Positions")
    cols = TB.read_table(ds, dom, layer)
    rcols = TB.read_table(ref_in, dom, ref_layer)
    lm_in = ctx.input("landmarks")
    lcols, lm_layer = None, ""
    if lm_in is not None:
        lm_layer = _resolve_layer(
            _structure_layers(lm_in, dom), ctx.layer("landmark_table"),
            node="register positions", socket="landmark_table", what=f"{dom.value} table",
            where="the `landmarks` input",
            remedy=f"wire the Dataset whose {dom.value} table holds the landmarks — the same "
                   f"objects the Reference table holds, seen in this run's session", ctx=ctx)[0]
        lcols = TB.read_table(lm_in, dom, lm_layer)
    xname = str(ctx.params.get("x", "stage_x_um") or "stage_x_um")
    yname = str(ctx.params.get("y", "stage_y_um") or "stage_y_um")
    zname = str(ctx.params.get("z", "stage_z_um") or "")
    search = float(ctx.params.get("search", 500.0))
    tol = float(ctx.params.get("tolerance", 25.0))
    for label, v in (("Search", search), ("Tolerance", tol)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"Register Positions: {label} must be a finite distance > 0, "
                             f"got {v}.")
    px, py = _rp_column(cols, xname, layer, "moving"), _rp_column(cols, yname, layer, "moving")
    qx, qy = (_rp_column(rcols, xname, ref_layer, "reference"),
              _rp_column(rcols, yname, ref_layer, "reference"))
    p = np.stack([px, py], axis=1)
    q = np.stack([qx, qy], axis=1)
    # the rows the transform is FITTED on: this table's own, or the landmarks'
    fcols, flayer, fside = (cols, layer, "moving") if lcols is None else \
        (lcols, lm_layer, "landmark")
    fp = p if lcols is None else np.stack(
        [_rp_column(lcols, xname, lm_layer, "landmark"),
         _rp_column(lcols, yname, lm_layer, "landmark")], axis=1)
    pok = np.all(np.isfinite(p), axis=1)
    fok = np.all(np.isfinite(fp), axis=1)
    qok = np.all(np.isfinite(q), axis=1)
    for side, ok in ((fside, fok), ("reference", qok), ("moving", pok)):
        if not ok.any():
            raise ValueError(f"Register Positions: the {side} table has no row with a finite "
                             f"{xname!r} / {yname!r}.")
    pi, fi, qi = np.flatnonzero(pok), np.flatnonzero(fok), np.flatnonzero(qok)
    try:
        fit = register_xy(fp[fi], q[qi], search=search, tol=tol, model=model)
    except ValueError as exc:
        raise ValueError(f"Register Positions: {exc}.") from None
    r, ca, cb = fit["R"], fit["ca"], fit["cb"]
    moved = p.copy()
    moved[pi] = (p[pi] - ca) @ r.T + cb
    n = len(p)
    dz = np.zeros(n)
    dz_fit = 0.0
    if zname:
        pz = _rp_column(cols, zname, layer, "moving")
        fz = _rp_column(fcols, zname, flayer, fside)
        qz = _rp_column(rcols, zname, ref_layer, "reference")
        diffs = np.array([qz[qi[j]] - fz[fi[i]] for i, j in fit["pairs"]])
        diffs = diffs[np.isfinite(diffs)]
        dz_fit = float(np.median(diffs)) if diffs.size else 0.0
        dz = np.where(np.isfinite(pz), dz_fit, 0.0)
    resid = np.full(n, np.nan)
    if lcols is None:
        for (i, _j), d in zip(fit["pairs"], fit["residual"]):
            resid[fi[i]] = d
    rot = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    out: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in cols.items()}
    out[xname], out[yname] = moved[:, 0], moved[:, 1]
    if zname:
        out[zname] = np.asarray(cols[zname], dtype=float) + dz
    out[DX_COLUMN] = moved[:, 0] - px
    out[DY_COLUMN] = moved[:, 1] - py
    out[DZ_COLUMN] = dz
    out[RESIDUAL_COLUMN] = resid
    out[ROTATION_COLUMN] = np.full(n, rot)
    shift = (cb - ca) if model == "rigid" else cb
    res = fit["residual"]
    who = "rows" if lcols is None else f"landmarks ({lm_layer!r}; {int(pok.sum())} rows moved)"
    ctx.progress(1, 1, f"{len(fit['pairs'])} of {int(fok.sum())} {who} paired with the "
                       f"reference; {model}: turned {rot:+.3f}°, shifted ({shift[0]:+.1f}, "
                       f"{shift[1]:+.1f}{f', {dz_fit:+.1f}' if zname else ''}); residual "
                       f"median {float(np.median(res)):.1f}, max {float(res.max()):.1f}")
    zk = ds.structure_zkind(dom, layer) or "plane_index"
    if name != layer:
        return TB.write_table(ds, dom, name, out, zk)
    # in place: the same rows of the same image with new stage columns — replaced without
    # write_table's table-node mark, so the Viewer still draws them over their image
    for a in list(ds.layers_on(dom)):
        if a.layer == layer:
            ds = ds.without(dom, a.name, layer)
    return ds.with_structure(StructureTable(dom, out, layer=layer, z_kind=zk))


def _rp_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    name = TB.param_text(params, "name")
    return ((TB.table_domain(modes), name),) if name else ()


def _rp_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                incoming: Sequence = (), inputs: Sequence = ()):
    """The moving table's columns, then the five ``reg_*`` ones, on the table written (total)."""
    try:
        dom = TB.table_domain(modes)
        envs = {sock: env for sock, env in (inputs or ())}
        env = envs.get("data")
        if env is not None:
            layer = TB.envelope_layer(env, dom, TB.param_text(params, "table"))
            cols = TB.envelope_columns(env, dom, layer)
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            layer = TB.param_text(params, "table") or (known[0] if len(known) == 1 else "")
            cols = [c for d, lyr, c in incoming or () if d is dom and lyr == layer]
        name = TB.param_text(params, "name") or layer
        if not name:
            return ()
        cols = list(cols) + list(_ADDED)
        return tuple((dom, name, c) for c in dict.fromkeys(cols) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_rp_columns.wants_inputs = True


register_node(
    _compute_register_positions, op_key="table.register_positions",
    label="Register Positions", category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table is moved into the reference "
                                       "frame — a later run of the same plate — passed on "
                                       "with the table's positions replaced."),
        InDataset("reference", label="Reference", passes_domains=False,
                  description="The Dataset carrying the table that sets the frame — the run "
                              "your other positions (a spheroid's centroid, a mosaic) were "
                              "measured in. Required."),
        InDataset("landmarks", label="Landmarks", passes_domains=False,
                  description="Optional: the Dataset whose table holds objects seen in BOTH "
                              "sessions — the beads of a whole-well scan taken with this run — "
                              "when this run's own objects are new. The transform is fitted "
                              "on them against the Reference table and applied to this table. "
                              "Unwired: fitted on this table's own rows."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="The table whose positions are moved. Blank uses the only one of "
                             "its kind on the wire, and asks when there are several."),
        InString("landmark_table", "Landmark table", field=False, default="",
                 layer_in_mode="domain", layer_from="landmarks",
                 description="The table on the Landmarks wire the transform is fitted on (same "
                             "X / Y / Z columns). Blank uses the only one of its kind there. "
                             "Only read with Landmarks wired."),
        InString("ref_table", "Reference table", field=False, default="",
                 layer_in_mode="domain", layer_from="reference",
                 description="The table on the Reference wire holding the same objects. Blank "
                             "uses the only one of its kind there."),
        InString("x", "X", field=False, default="stage_x_um", column_in_mode="domain",
                 column_from="table",
                 description="The column placing each row along x, in BOTH tables. The "
                             "default `stage_x_um` (from Measure) is the stage position, the "
                             "one coordinate two runs of one plate share; a per-field column "
                             "like `x_um` would not register anything."),
        InString("y", "Y", field=False, default="stage_y_um", column_in_mode="domain",
                 column_from="table",
                 description="The column placing each row along y in both tables — "
                             "`stage_y_um` by default, for the same reason as X."),
        InString("z", "Z", field=False, default="stage_z_um", column_in_mode="domain",
                 column_from="table",
                 description="The focus column, shifted by the pairs' median difference (a "
                             "plate put back sits a little higher or lower). Blank leaves Z "
                             "alone and registers X / Y only. Changes every height and 3D "
                             "distance computed downstream."),
        InFloat("search", "Search", unit="um", field=False, default=500.0,
                description="The largest move between the two runs to consider, in the X / Y "
                            "columns' units. It must exceed how far the plate actually moved; "
                            "LARGER also lets unrelated objects vote, which only matters when "
                            "the two tables share few objects."),
        InFloat("tolerance", "Tolerance", unit="um", field=False, default=25.0,
                description="How close a row must land to a reference row, once shifted, to "
                            "count as the same object. LARGER pairs objects that re-centred "
                            "less exactly but risks pairing neighbours; SMALLER pairs fewer, "
                            "and below the stage's repeatability (10-20 µm after a re-mount) "
                            "pairs almost none. Keep it under the objects' spacing."),
        InString("name", "Name", field=False, default="",
                 description="Name of the table written. Blank (the default) replaces the "
                             "positions in the table itself, so downstream nodes keep reading "
                             "the same table; a name writes a registered copy beside it."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_RP_DOMAINS), default="label", label="Domain",
             description="Which kind of table both inputs carry.",
             choice_docs=domain_docs(_RP_DOMAINS)),
        Mode("model", list(_RP_MODELS), default="rigid", label="Model",
             description="How the plate may have moved between the two runs.",
             choice_docs={
                 "rigid": "A turn about the paired objects' centre plus a shift, fitted by "
                          "least squares — what lifting a plate off the stage and putting it "
                          "back does (a fraction of a degree moves objects mm away by tens of "
                          "µm). Needs two or more shared objects.",
                 "shift": "A shift only: the median of the pairs' differences, robust to a "
                          "few mismatches. Right for one continuous session where the stage "
                          "was re-zeroed; it leaves a turn's error growing with distance from "
                          "the objects' centre. Works from one shared object.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT})}},
    extra_layers=_rp_layers, adds_columns=_rp_columns,
    description="Move a table's stage positions into a reference table's frame, fitted on the "
                "objects both contain: the shift is voted from every row pair, the objects "
                "are paired within a tolerance, and a rigid turn + shift (or a shift) is "
                "fitted to them — a plate re-imaged after it was taken off the stage.")
