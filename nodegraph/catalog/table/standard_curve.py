"""table.standard_curve — turn a measured signal into a concentration through a standard curve
fitted from standards measured the same way (2026-10-15, IL-8 capture beads read against the
Configuration 1 bead standards)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.figure as FIG
import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

_SC_DOMAINS = ("label", "point", "track")
_STD_DOMAINS = ("label", "point")
_MODELS = ("langmuir", "linear")
_CURVES = ("fit", "typed")
_DEFAULT_NAME = "concentration"
FLAG_SUFFIX = "_flag"
#: provenance: ``{output column: the curve that made it}``
CURVE_KEY = "standard_curve"
#: how many blank standard deviations above the blank mean the detection limit sits
LOD_SDS = 3.0


def parse_where(text: Any) -> List[Tuple[str, str]]:
    """``"config=1, sighting=0"`` → ``[("config", "1"), ("sighting", "0")]`` (total on blank;
    a clause without ``=`` raises with the clause named)."""
    out: List[Tuple[str, str]] = []
    for clause in str(text or "").split(","):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise ValueError(f"Standard Curve: the Standards filter clause {clause!r} is not "
                             f"`column=value` (e.g. `sighting=0`).")
        k, v = clause.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def where_mask(cols: Mapping[str, np.ndarray], clauses: Sequence[Tuple[str, str]],
               layer: str) -> np.ndarray:
    """Rows matching EVERY clause: text compared as text, numbers as numbers."""
    n = len(next(iter(cols.values()))) if cols else 0
    keep = np.ones(n, dtype=bool)
    for k, v in clauses:
        if k not in cols:
            raise ValueError(f"Standard Curve: the Standards filter names {k!r}, which the "
                             f"{layer!r} table does not carry (it has {sorted(cols)}).")
        col = np.asarray(cols[k])
        if TB.is_text(col):
            keep &= np.asarray([str(x).strip() == v for x in col.tolist()])
        else:
            try:
                f = float(v)
            except ValueError:
                raise ValueError(f"Standard Curve: {k!r} is numeric, and {v!r} is not a "
                                 f"number.") from None
            keep &= np.isclose(col.astype(float), f, rtol=0.0, atol=1e-9)
    return keep


def standard_points(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]
                    ) -> Tuple[List[Tuple[float, float, int]], Optional[Dict[str, float]]]:
    """One point per standard concentration ``x > 0`` — the WEIGHTED mean of its rows' signal
    (weight = each row's pixel count reproduces the mean of every pooled pixel, as Plot XY's
    pooled SD draws it) — and the blank (``x <= 0``): its weighted mean, the SD of its rows
    and their count. Rows with a non-finite x, y or weight are left out."""
    ww = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ww) & (ww > 0)
    x, y, ww = x[ok], y[ok], ww[ok]
    pts = []
    for xv in sorted(set(x[x > 0].tolist())):
        m = x == xv
        pts.append((float(xv), float((ww[m] * y[m]).sum() / ww[m].sum()), int(m.sum())))
    blank = None
    b = x <= 0
    if b.any():
        sd = float(np.std(y[b], ddof=1)) if b.sum() >= 2 else float("nan")
        blank = {"mean": float((ww[b] * y[b]).sum() / ww[b].sum()), "sd": sd, "n": int(b.sum())}
    return pts, blank


def invert(y: np.ndarray, curve: Mapping[str, Any]) -> np.ndarray:
    """Concentration from signal: Langmuir ``Kd·y/(Bmax − y)`` (NaN at or above Bmax, where
    the curve never reaches), linear ``(y − a)/b``."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if curve["model"] == "langmuir":
            bmax, kd = float(curve["bmax"]), float(curve["kd"])
            return np.where(y < bmax, kd * y / (bmax - y), np.nan)
        return (y - float(curve["intercept"])) / float(curve["slope"])


def flags(y: np.ndarray, x: np.ndarray, curve: Mapping[str, Any]) -> List[str]:
    """Each row's standing against the curve: ``no signal``, ``above Bmax``, ``below LOD``,
    ``below range`` / ``above range`` (extrapolated past the lowest / highest standard),
    ``in range``."""
    lod = curve.get("lod_signal")
    lo, hi = (list(curve.get("range") or [None, None]) + [None, None])[:2]
    out = []
    for yv, xv in zip(np.asarray(y, dtype=float), np.asarray(x, dtype=float)):
        if not math.isfinite(yv):
            out.append("no signal")
        elif curve["model"] == "langmuir" and yv >= float(curve["bmax"]):
            out.append("above Bmax")
        elif lod is not None and yv < float(lod):
            out.append("below LOD")
        elif lo is not None and xv < float(lo):
            out.append("below range")
        elif hi is not None and xv > float(hi):
            out.append("above range")
        else:
            out.append("in range")
    return out


def _fit_curve(ctx: EvalContext, model: str, sig: str) -> Dict[str, Any]:
    """The curve from the Standards table: its points, blank, fit, range and LOD."""
    std = ctx.input("standards")
    if std is None:
        raise ValueError("Standard Curve: Curve = fit needs the standards — wire the Dataset "
                         "whose table holds them (each bead's signal at a known "
                         "concentration) into Standards, or switch Curve to typed.")
    modes = ctx.params.get("__modes__", {}) or {}
    sdom = TB.table_domain(modes, "std_domain")
    layer = _resolve_layer(
        _structure_layers(std, sdom), ctx.layer("std_table"), node="standard curve",
        socket="std_table", what=f"{sdom.value} table", where="the `standards` input",
        remedy="wire the standards' table (a Table Concat of the wells, labelled by "
               "concentration), or switch Standards domain", ctx=ctx)[0]
    cols = TB.read_table(std, sdom, layer)
    # blank = Table Concat's numeric condition — a fixed default would name a column no
    # producer declares until its inputs are known (an orphan entry in the picker)
    xname = str(ctx.params.get("std_x", "") or "") or TB.CONDITION_VALUE_COLUMN
    yname = str(ctx.params.get("std_y", "") or "") or sig
    wname = str(ctx.params.get("std_weight", "") or "")
    for nm, what in ((xname, "Standards X"), (yname, "Standards Y"), (wname, "Weight")):
        if nm and (nm not in cols or TB.is_text(cols[nm])):
            raise ValueError(f"Standard Curve: {what} {nm!r} is not a numeric column of the "
                             f"standards table {layer!r} (it has {sorted(cols)}).")
    keep = where_mask(cols, parse_where(ctx.params.get("std_where", "")), layer)
    if not keep.any():
        raise ValueError(f"Standard Curve: no row of the standards table {layer!r} matches the "
                         f"filter {ctx.params.get('std_where', '')!r}.")
    x = np.asarray(cols[xname], dtype=float)[keep]
    y = np.asarray(cols[yname], dtype=float)[keep]
    w = np.asarray(cols[wname], dtype=float)[keep] if wname else None
    pts, blank = standard_points(x, y, w)
    if len(pts) < 2:
        raise ValueError(f"Standard Curve: {len(pts)} standard concentration(s) above 0 after "
                         f"the filter — a curve needs at least two.")
    px = np.asarray([p[0] for p in pts])
    py = np.asarray([p[1] for p in pts])
    if model == "langmuir":
        f = FIG.xy_langmuir(px, py)
        if f is None:
            raise ValueError("Standard Curve: the Langmuir fit did not converge on these "
                             "standards — try the linear model.")
        curve: Dict[str, Any] = {"model": "langmuir", "bmax": f["bmax"], "kd": f["kd"],
                                 "r2": f["r2"]}
    else:
        lx = np.append(px, 0.0) if blank is not None else px
        ly = np.append(py, blank["mean"]) if blank is not None else py
        f = FIG.xy_fit(lx, ly)
        if f is None or not f["slope"]:
            raise ValueError("Standard Curve: the linear fit is flat on these standards.")
        curve = {"model": "linear", "slope": f["slope"], "intercept": f["intercept"],
                 "r2": f["r2"]}
    curve["range"] = [float(px.min()), float(px.max())]
    curve["points"] = [list(p) for p in pts]
    curve["source"] = {"table": layer, "x": xname, "y": yname, "weight": wname,
                       "where": str(ctx.params.get("std_where", "") or "")}
    if blank is not None:
        curve["blank"] = blank
        if math.isfinite(blank["sd"]):
            curve["lod_signal"] = blank["mean"] + LOD_SDS * blank["sd"]
    from nodegraph.units import unit_of
    curve["unit"] = unit_of(std, sdom, xname, layer)
    return curve


def _compute_standard_curve(ctx: EvalContext) -> Dataset:
    """A signal column (a bead's background-subtracted MFI) turned into a concentration by a
    standard curve — fitted here from standards measured the same way, or typed.

    Resolved spec (2026-10-15, `build-node-v2`):

    * **Kind** table → ``table.standard_curve``: one table in (``domain`` label / point /
      track, ``table``), the same table out with two columns added IN PLACE (``name`` and
      ``<name>_flag``), so its raster and overlays still match. A Name the table already has
      is refused — a conversion must never overwrite a measurement.
    * **Curve = fit** (default): the ``standards`` input's table (``std_domain``,
      ``std_table``) — one row per standard bead, its known concentration ``std_x`` and its
      signal ``std_y`` (blank = the converted column's own name; ``std_x`` blank = Table
      Concat's ``condition_value``), optionally weighted by
      ``std_weight`` (a pixel count: the weighted mean of bead means IS the mean of every
      pooled pixel, which is what the IL-8 report fitted) — filtered by ``std_where``
      (``column=value`` clauses, all must hold: ``sighting=0`` keeps first imagings). One
      point per concentration > 0: the weighted mean. ``model`` langmuir fits ``Bmax·x /
      (Kd + x)`` through those points in data units (:func:`figure.xy_langmuir`, the fit Plot
      XY draws); linear fits ``a + b·x`` through them AND the blank. The blank rows (x <= 0)
      give the detection limit, mean + 3 SD of the blank rows' signal.
    * **Curve = typed**: ``bmax`` / ``kd`` or ``slope`` / ``intercept``, and an optional
      ``lod_signal``; no range is known, so no range flags.
    * **Inversion** Langmuir ``Kd·y/(Bmax − y)`` (NaN at or above Bmax); linear ``(y − a)/b``.
      **Flags** (text) ``no signal``, ``above Bmax``, ``below LOD``, ``below range`` /
      ``above range`` (extrapolated beyond the lowest / highest standard) or ``in range``.
    * **Provenance** the curve (points, blank, fit, range, LOD) under ``standard_curve`` in
      the metadata, keyed by the output column; the new column carries the standards' x unit.
    * **Footprint** WHOLE_SERIES, empty kernel_axes: tables only, no axis or calibration
      change.
    """
    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    model = str(modes.get("model", "langmuir"))
    how = str(modes.get("curve", "fit"))
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="standard curve",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire the measured objects' {dom.value} table, or switch Domain", ctx=ctx)[0]
    cols = TB.read_table(ds, dom, layer)
    sig = str(ctx.params.get("signal", "") or "")
    if not sig:
        raise ValueError("Standard Curve: name the Signal column to convert — a bead's "
                         "background-subtracted MFI, say.")
    if sig not in cols or TB.is_text(cols[sig]):
        raise ValueError(f"Standard Curve: Signal {sig!r} is not a numeric column of the "
                         f"{layer!r} table (it has {sorted(cols)}).")
    name = str(ctx.params.get("name", _DEFAULT_NAME) or _DEFAULT_NAME)
    for nm in (name, name + FLAG_SUFFIX):
        if nm in cols:
            raise ValueError(f"Standard Curve: the {layer!r} table already has a column {nm!r} "
                             f"— choose another Name; a conversion never overwrites a "
                             f"measurement.")
    if how == "fit":
        curve = _fit_curve(ctx, model, sig)
    else:
        if model == "langmuir":
            bmax = float(ctx.params.get("bmax", 0.0) or 0.0)
            kd = float(ctx.params.get("kd", 0.0) or 0.0)
            if not (bmax > 0 and kd > 0):
                raise ValueError(f"Standard Curve: a typed Langmuir curve needs Bmax > 0 and "
                                 f"Kd > 0 (got {bmax:g}, {kd:g}).")
            curve = {"model": "langmuir", "bmax": bmax, "kd": kd, "r2": None}
        else:
            slope = float(ctx.params.get("slope", 0.0) or 0.0)
            if not (math.isfinite(slope) and slope != 0):
                raise ValueError("Standard Curve: a typed linear curve needs a Slope ≠ 0.")
            curve = {"model": "linear", "slope": slope,
                     "intercept": float(ctx.params.get("intercept", 0.0) or 0.0), "r2": None}
        lod = float(ctx.params.get("lod_signal", 0.0) or 0.0)
        if lod > 0:
            curve["lod_signal"] = lod
        curve["unit"] = None
    # the column converted, so a model fitted to the concentration can be drawn in the
    # signal's units again (Point-Source Fit → Plot XY / Gradient Map, 2026-10-16)
    curve["signal"] = sig
    y = np.asarray(cols[sig], dtype=float)
    conc = invert(y, curve)
    fl = flags(y, conc, curve)
    if curve.get("lod_signal") is not None:
        curve["lod"] = float(invert(np.asarray([curve["lod_signal"]]), curve)[0])
    out = dict(cols)
    out[name] = conc
    out[name + FLAG_SUFFIX] = TB.text_column(fl)
    zk = ds.structure_zkind(dom, layer) or "plane_index"
    res = ds
    for a in list(ds.layers_on(dom)):
        if a.layer == layer:
            res = res.without(dom, a.name, layer)
    res = res.with_structure(StructureTable(dom, out, layer=layer, z_kind=zk))
    prov = dict((ds.metadata or {}).get(CURVE_KEY) or {})
    prov[name] = {k: FIG.json_value(v) if not isinstance(v, (list, dict)) else v
                  for k, v in curve.items()}
    res = res.with_metadata(**{CURVE_KEY: prov})
    if curve.get("unit"):
        from nodegraph.units import with_unit
        res = with_unit(res, dom, name, curve["unit"], layer=layer)
    counts = {k: fl.count(k) for k in dict.fromkeys(fl)}
    if curve["model"] == "langmuir":
        txt = f"Bmax {curve['bmax']:.4g}, Kd {curve['kd']:.4g}"
    else:
        txt = f"y = {curve['slope']:.4g}·x + {curve['intercept']:.4g}"
    if curve.get("r2") is not None:
        txt += f", R² {curve['r2']:.3f}"
    if curve.get("lod") is not None:
        txt += f"; LOD {curve['lod_signal']:.3g} signal = {curve['lod']:.3g}"
    ctx.progress(1, 1, f"{curve['model']} ({how}): {txt}; "
                       + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return res


def _sc_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                incoming: Sequence = (), inputs: Sequence = ()):
    """The converted table's columns plus ``name`` and ``<name>_flag`` (total)."""
    try:
        dom = TB.table_domain(modes)
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        envs = {sock: env for sock, env in (inputs or ())}
        env = envs.get("data")
        if env is not None:
            layer = TB.envelope_layer(env, dom, TB.param_text(params, "table"))
            cols = TB.envelope_columns(env, dom, layer)
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            layer = TB.param_text(params, "table") or (known[0] if len(known) == 1 else "")
            cols = [c for d, lyr, c in incoming or () if d is dom and lyr == layer]
        if not layer:
            return ()
        return tuple((dom, layer, c) for c in dict.fromkeys(list(cols) +
                                                            [name, name + FLAG_SUFFIX]) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_sc_columns.wants_inputs = True

_FIT = {"curve": frozenset({"fit"})}
_TYPED = {"curve": frozenset({"typed"})}

register_node(
    _compute_standard_curve, op_key="table.standard_curve", label="Standard Curve",
    category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table is converted — the measured "
                                       "objects — passed on with two columns added to that "
                                       "table."),
        InDataset("standards", label="Standards", passes_domains=False,
                  description="The Dataset carrying the standards' table: each standard bead's "
                              "signal at a known concentration, measured exactly as the "
                              "objects were. Read only with Curve = fit."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="The table converted. Blank uses the only one of its kind on the "
                             "wire."),
        InString("signal", "Signal", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="The column converted — the same quantity the standards were "
                             "measured in (a bead's background-subtracted R-PE MFI). "
                             "Required."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="The new concentration column (`il8_pg_ml`, say — the unit in "
                             "the name, since pg/mL is not one the units algebra knows); "
                             "`<name>_flag` beside it says whether each row is in the "
                             "standards' range, extrapolated, under the detection limit or "
                             "past the curve. A name the table has is refused."),
        InString("std_table", "Standards table", field=False, default="",
                 layer_in_mode="std_domain", layer_from="standards", available_in=_FIT,
                 description="The standards' table on the Standards wire. Blank uses the only "
                             "one of its kind there."),
        InString("std_x", "Standards X", field=False, default="",
                 column_in_mode="std_domain",
                 column_from="std_table", available_in=_FIT,
                 description="The standards' known concentration. Blank uses Table "
                             "Concat's `condition_value` (its Names as numbers — the wells' "
                             "concentrations). Rows at 0 are the blank: not fitted, they set "
                             "the detection limit."),
        InString("std_y", "Standards Y", field=False, default="", column_in_mode="std_domain",
                 column_from="std_table", available_in=_FIT,
                 description="The standards' signal column. Blank uses the Signal column's "
                             "own name — the same measurement on both tables."),
        InString("std_weight", "Weight", field=False, default="", column_in_mode="std_domain",
                 column_from="std_table", available_in=_FIT,
                 description="Weigh each standard row by this column when averaging a "
                             "concentration — a bead's pixel count (`area`), so the point is "
                             "the mean of every pooled pixel, as Plot XY's pooled SD draws "
                             "it. Blank weighs every bead equally. Moves every fitted value."),
        InString("std_where", "Standards filter", field=False, default="", available_in=_FIT,
                 description="Keep only the standard rows matching every `column=value` "
                             "clause, comma-separated: `sighting=0` fits the first imaging "
                             "only, so photobleaching does not lower the curve. Blank keeps "
                             "every row."),
        InFloat("bmax", "Bmax", unit="", field=False, default=0.0,
                available_in={"curve": frozenset({"typed"}),
                              "model": frozenset({"langmuir"})},
                description="The typed Langmuir plateau, in the signal's units. Signals at "
                            "or above it have no concentration (NaN, flagged)."),
        InFloat("kd", "Kd", unit="", field=False, default=0.0,
                available_in={"curve": frozenset({"typed"}),
                              "model": frozenset({"langmuir"})},
                description="The typed Langmuir half-saturation concentration, in the "
                            "concentration's units; the curve is near-linear well below it."),
        InFloat("slope", "Slope", unit="", field=False, default=0.0,
                available_in={"curve": frozenset({"typed"}), "model": frozenset({"linear"})},
                description="The typed line's signal per unit concentration."),
        InFloat("intercept", "Intercept", unit="", field=False, default=0.0,
                available_in={"curve": frozenset({"typed"}), "model": frozenset({"linear"})},
                description="The typed line's signal at zero concentration (the blank)."),
        InFloat("lod_signal", "LOD signal", unit="", field=False, default=0.0,
                available_in=_TYPED,
                description="A typed detection limit in signal units: rows under it are "
                            "flagged `below LOD`. 0 flags none. A fitted curve takes its own "
                            "from the blank (mean + 3 SD)."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_SC_DOMAINS), default="label", label="Domain",
             description="Which kind of table is converted.",
             choice_docs=domain_docs(_SC_DOMAINS)),
        Mode("model", list(_MODELS), default="langmuir", label="Model",
             description="The curve's shape.",
             choice_docs={
                 "langmuir": "One-site binding, signal = Bmax·c / (Kd + c), through the "
                             "origin — a capture assay that saturates. Fitted in data units "
                             "on the standards above 0 (the blank only sets the detection "
                             "limit); signals at or above Bmax have no concentration.",
                 "linear": "signal = a + b·c, fitted through the standards AND the blank — "
                           "for a curve that has not begun to saturate over the range; it "
                           "extrapolates without bound in both directions.",
             }),
        Mode("curve", list(_CURVES), default="fit", label="Curve",
             description="Where the curve's parameters come from.",
             choice_docs={
                 "fit": "Fitted here from the standards wired into Standards, so the "
                        "parameters follow the standards' data and their filter; the range "
                        "and detection limit come with them.",
                 "typed": "Typed from a report (Bmax and Kd, or a slope and intercept): no "
                          "standards needed, but no range is known — rows are only flagged "
                          "against Bmax and a typed LOD.",
             }),
        Mode("std_domain", list(_STD_DOMAINS), default="label", label="Standards domain",
             available_in=_FIT,
             description="Which kind of table on the Standards wire holds the standards.",
             choice_docs=domain_docs(_STD_DOMAINS)),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_columns=_sc_columns,
    description="Turn a measured signal (a bead's MFI) into a concentration through a "
                "standard curve — Langmuir or linear, fitted from standards measured the same "
                "way and wired in, or typed — with each row flagged in range, extrapolated, "
                "under the detection limit or past the curve.")
