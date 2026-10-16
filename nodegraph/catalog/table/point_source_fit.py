"""table.point_source_fit — Crank's continuous point source fitted to concentrations measured
round a secreting source: the source's production rate (and the analyte's diffusivity) from
capture beads at known distances and times (2026-10-16, IL-8 round a spheroid)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

import nodegraph.catalog._shared.point_source as PS
import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import (Granularity, InDataset, InFloat, InInt, InString, Mode,
                                OutDataset)
from nodegraph.structure import StructureTable

_PSF_DOMAINS = ("label", "point", "track")
_TIME_KINDS = ("clock", "hours", "seconds")
_DIFFUSIVITY = ("fixed", "fit")
_DEFAULT_NAME = "crank"
RATES_SUFFIX = "_rates"
#: provenance: ``{fitted value column: the fit}`` — read by Plot XY's point-source trendline
#: and Gradient Map's point-source trend
FIT_KEY = "point_source_fit"
_BOOT_SEED = 20261016
_RATE_COLUMNS = ("label", "t_h", "t_start_h", "t_end_h", "t_mid_h", "q_fg_s", "q_lo_fg_s",
                 "q_hi_fg_s", "q_ci_fg_s", "q_pg_h", "released_pg", "released_ci_pg",
                 "cumulative_pg", "background_pg_ml",
                 "D_um2_s", "r2", "n")


def _times_s(ctx: EvalContext, cols: Mapping[str, np.ndarray], layer: str) -> np.ndarray:
    """Each row's time since the source started, in seconds."""
    modes = ctx.params.get("__modes__", {}) or {}
    kind = str(modes.get("time_kind", "clock"))
    tname = str(ctx.params.get("time", "") or "") or TB.CLOCK_COLUMN
    if tname not in cols or TB.is_text(cols[tname]):
        raise ValueError(
            f"Point-Source Fit: Time {tname!r} is not a numeric column of the {layer!r} table "
            f"(it has {sorted(cols)}). Table Concat writes `time_jd` (each row's clock) when "
            f"every input has a frame clock; otherwise name a column of elapsed hours.")
    raw = np.asarray(cols[tname], dtype=float)
    start = float(ctx.params.get("start_h", 0.0) or 0.0) * 3600.0
    if kind == "clock":
        fin = raw[np.isfinite(raw)]
        if not fin.size:
            raise ValueError(f"Point-Source Fit: the clock column {tname!r} has no finite "
                             f"value.")
        return (raw - fin.min()) * 86400.0 + start
    if kind == "hours":
        return raw * 3600.0 + start
    return raw + start


def _psf_groups(cols: Mapping[str, np.ndarray], gname: str):
    """``(index per row, labels)`` in first-seen order; blank ``gname`` groups by ``t``."""
    raw = cols[gname] if gname else cols.get("t", np.zeros(len(next(iter(cols.values())))))
    keys = [str(v) for v in np.asarray(raw).tolist()]
    labels = list(dict.fromkeys(keys))
    return np.asarray([labels.index(k) for k in keys], dtype=int), labels


def _bootstrap(r, t, y, gi, labels, rec, kw, n_boot: int) -> List[List[float]]:
    """Each reported rate's 2.5 / 97.5 % bootstrap interval — rows resampled WITHIN each
    group, the fit redone at the fitted D (so the interval is conditional on D)."""
    if n_boot <= 0:
        return []
    rng = np.random.default_rng(_BOOT_SEED)
    by = [np.flatnonzero(gi == k) for k in range(len(labels))]
    draws = []
    for _ in range(int(n_boot)):
        pick = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in by
                               if ix.size])
        try:
            f = PS.fit(r[pick], t[pick], y[pick], gi[pick], labels,
                       **{**kw, "D": rec["D"], "fit_D": False})
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        draws.append([row["q_fg_s"] for row in PS.rate_rows(f)])
    if not draws:
        return []
    arr = np.asarray(draws, dtype=float)
    return [[float(np.nanpercentile(arr[:, j], 2.5)), float(np.nanpercentile(arr[:, j], 97.5))]
            for j in range(arr.shape[1])]


def _compute_point_source_fit(ctx: EvalContext) -> Dataset:
    """Fit Crank's continuous point source to concentrations measured round a source.

    Resolved spec (2026-10-16, `build-node-v2`):

    * **Kind** table → ``table.point_source_fit``: one table in (``domain`` label / point /
      track), passed on with the model's value at every row added IN PLACE (``name``), and a
      new Label table ``<name>_rates`` (one row per rate). Tables only.
    * **Model** :mod:`nodegraph.catalog._shared.point_source`: ``C = q/(4πDr)·erfc(r/2√(Dt))``
      (McGhee et al. 2022 Eq. 2, after Crank), ×2 under ``boundary = half_space`` (a source on
      an impermeable floor). ``rate = constant``: one ``q`` per group (each timepoint's
      average since the source started — the paper's per-timepoint fit); ``piecewise``: one
      per interval between successive group times, all groups fitted together by
      superposition. ``diffusivity = fit`` profiles ``D`` (one, shared) over 0.5–500 µm²/s;
      ``fixed`` uses ``D`` (default 20, the paper's IL-8 value).
    * **Columns** ``distance`` (µm to the source), ``value`` (concentration, pg/mL — the
      model's unit), ``group`` (one time each; blank = ``t``), ``time`` read by
      ``time_kind``: ``clock`` a Julian date (Table Concat's ``time_jd``, each row's own
      acquisition clock), the source started ``start_h`` hours before the EARLIEST row;
      ``hours`` / ``seconds`` elapsed, plus ``start_h``. Rows with a non-finite or non-positive
      distance or time, or a value under ``fit_floor`` (or ≤ 0 under ``relative``), are not
      fitted; every row with a distance and time still gets the model's value.
    * **Weight** ``relative`` fits ``ln C`` (every bead weighs by its relative miss),
      ``sqrt`` fits ``√C`` (between the two), ``absolute`` fits pg/mL. R² is in the space
      fitted.
    * **Background** ``uniform`` (2026-10-16) adds one non-negative level per group to the
      point source — the far field of a FINITE well, whose walls return what an unbounded
      Eq. 2 lets escape, so the profile levels off instead of falling to 0; reported as
      ``background_pg_ml`` and part of the model every reader draws.
    * **Uncertainty** ``bootstrap`` resamples (rows within each group, fixed seed) give each
      rate's 95 % interval, conditional on the fitted D.
    * **Provenance** the fit under ``point_source_fit`` in the metadata, keyed by the value
      column, with the standard curve that made that column (``standard_curve``), so Plot XY
      and Gradient Map can draw the model in the signal's units too.
    * **Footprint** WHOLE_SERIES, empty kernel_axes. Numpy + scipy.
    """
    ds: Dataset = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    rate = str(modes.get("rate", "constant"))
    boundary = str(modes.get("boundary", "infinite"))
    weight = str(modes.get("weight", "relative"))
    fit_D = str(modes.get("diffusivity", "fixed")) == "fit"
    background = str(modes.get("background", "none"))
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="point-source fit",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire the measured objects' {dom.value} table (Standard Curve's output, with "
               f"a concentration column), or switch Domain", ctx=ctx)[0]
    cols = TB.read_table(ds, dom, layer)
    dname = str(ctx.params.get("distance", "distance_um") or "distance_um")
    vname = str(ctx.params.get("value", "") or "")
    gname = str(ctx.params.get("group", "") or "")
    if not vname:
        raise ValueError("Point-Source Fit: name the Value column — a concentration in pg/mL "
                         "(Standard Curve's output, say).")
    for nm, what in ((dname, "Distance"), (vname, "Value")):
        if nm not in cols or TB.is_text(cols[nm]):
            raise ValueError(f"Point-Source Fit: {what} {nm!r} is not a numeric column of the "
                             f"{layer!r} table (it has {sorted(cols)}).")
    if gname and gname not in cols:
        raise ValueError(f"Point-Source Fit: Group {gname!r} is not a column of the {layer!r} "
                         f"table (it has {sorted(cols)}).")
    name = str(ctx.params.get("name", _DEFAULT_NAME) or _DEFAULT_NAME)
    if name in cols:
        raise ValueError(f"Point-Source Fit: the {layer!r} table already has a column {name!r} "
                         f"— choose another Name; a model never overwrites a measurement.")
    TB.refuse_voxel_name(ds, name + RATES_SUFFIX, "Point-Source Fit")
    D = float(ctx.params.get("D", 20.0))
    if not (math.isfinite(D) and D > 0):
        raise ValueError(f"Point-Source Fit: D must be a positive diffusivity (µm²/s), got {D}.")
    floor = float(ctx.params.get("fit_floor", 0.0) or 0.0)
    n_boot = int(ctx.params.get("bootstrap", 200) or 0)

    r = np.asarray(cols[dname], dtype=float)
    y = np.asarray(cols[vname], dtype=float)
    t = _times_s(ctx, cols, layer)
    gi, labels = _psf_groups(cols, gname)
    use = np.isfinite(r) & (r > 0) & np.isfinite(t) & (t > 0) & np.isfinite(y)
    if weight == "relative":
        use &= y > 0
    if floor and math.isfinite(floor):
        use &= y >= floor
    if use.sum() < 2:
        raise ValueError(f"Point-Source Fit: {int(use.sum())} usable row(s) — a fit needs at "
                         f"least two with a distance, a time after the source started and a "
                         f"value{' > 0' if weight == 'relative' else ''}"
                         f"{f' >= {floor:g}' if floor else ''}.")
    kw = {"rate": rate, "D": D, "fit_D": fit_D, "boundary": boundary, "weight": weight,
          "background": background}
    ctx.progress(0, 2, "fitting the point source")
    rec = PS.fit(r[use], t[use], y[use], gi[use], labels, **kw)
    ctx.progress(1, 2, f"bootstrap ({n_boot})" if n_boot > 0 else "rates")
    ci = _bootstrap(r[use], t[use], y[use], gi[use], labels, rec, kw, n_boot)

    # every row with a distance and a time gets the model's value — fitted or not
    model = np.full(len(r), np.nan)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(t) & (t > 0)
    if rate == "constant":
        for k, lab in enumerate(labels):
            m = ok & (gi == k)
            model[m] = PS.predict(r[m], t[m], rec, lab)
    else:
        model[ok] = PS.predict(r[ok], t[ok], rec)
    out_cols = dict(cols)
    out_cols[name] = model
    zk = ds.structure_zkind(dom, layer) or "plane_index"
    res = ds
    for a in list(ds.layers_on(dom)):
        if a.layer == layer:
            res = res.without(dom, a.name, layer)
    res = res.with_structure(StructureTable(dom, out_cols, layer=layer, z_kind=zk))

    rows = PS.rate_rows(rec)
    g_by = {g["label"]: g for g in rec["groups"]}
    n_r = len(rows)
    rt: Dict[str, Any] = {"id": np.arange(1, n_r + 1)}
    for c in ("m", "t", "c", "z"):
        rt[c] = np.zeros(n_r, np.int64)
    rt["y"] = np.zeros(n_r)
    rt["x"] = np.zeros(n_r)
    rt["label"] = TB.text_column(row["label"] for row in rows)
    for c in ("t_start_h", "t_end_h", "t_mid_h", "q_fg_s", "q_pg_h", "released_pg",
              "cumulative_pg"):
        rt[c] = np.asarray([row[c] for row in rows], dtype=float)
    # the time to plot a rate at: a timepoint's own time (an average up to it), an interval's
    # middle (a rate during it)
    rt["t_h"] = rt["t_end_h"] if rate == "constant" else rt["t_mid_h"]
    lo = [c[0] for c in ci] if len(ci) == n_r else [math.nan] * n_r
    hi = [c[1] for c in ci] if len(ci) == n_r else [math.nan] * n_r
    rt["q_lo_fg_s"] = np.asarray(lo, dtype=float)
    rt["q_hi_fg_s"] = np.asarray(hi, dtype=float)
    rt["q_ci_fg_s"] = 0.5 * (rt["q_hi_fg_s"] - rt["q_lo_fg_s"])
    # the same interval on what was released over the row's span (rate x hours, in pg)
    rt["released_ci_pg"] = rt["q_ci_fg_s"] * 3.6 * (rt["t_end_h"] - rt["t_start_h"])
    rt["D_um2_s"] = np.full(n_r, float(rec["D"]))
    rt["background_pg_ml"] = np.asarray([row.get("background", 0.0) for row in rows],
                                        dtype=float)
    rt["r2"] = np.asarray([np.nan if g_by.get(row["label"], {}).get("r2") is None
                           else g_by[row["label"]]["r2"] for row in rows], dtype=float)
    rt["n"] = np.asarray([g_by.get(row["label"], {}).get("n", 0) for row in rows],
                         dtype=np.int64)
    res = TB.write_table(res, Domain.LABEL, name + RATES_SUFFIX, rt, "plane_index")

    curves = (ds.metadata or {}).get("standard_curve") or {}
    curve = curves.get(vname) if isinstance(curves, Mapping) else None
    entry = dict(rec)
    entry.update({"value": vname, "distance": dname, "group": gname, "unit": "pg/mL",
                  "rates_table": name + RATES_SUFFIX,
                  "rate_ci": [[float(a), float(b)] for a, b in zip(lo, hi)]})
    if isinstance(curve, Mapping) and curve.get("signal"):
        entry["signal"] = str(curve["signal"])
        entry["curve"] = {k: curve[k] for k in ("model", "bmax", "kd", "slope", "intercept")
                          if k in curve}
    prov = dict((ds.metadata or {}).get(FIT_KEY) or {})
    prov[vname] = entry
    res = res.with_metadata(**{FIT_KEY: prov})
    qs = ", ".join(f"{row['q_fg_s']:.3g}" for row in rows)
    if background == "uniform":
        qs += "; background [pg/mL] " + ", ".join(f"{row['background']:.3g}" for row in rows)
    ctx.progress(2, 2, f"D {rec['D']:.3g} µm²/s ({'fitted' if fit_D else 'fixed'}), "
                       f"{boundary}, {rate} q [fg/s] {qs}; R² {rec['r2']:.3f} "
                       f"({weight}, n {rec['n']})")
    return res


def _psf_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    return ((Domain.LABEL, TB.param_text(params, "name", _DEFAULT_NAME) + RATES_SUFFIX),)


def _psf_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                 incoming: Sequence = (), inputs: Sequence = ()):
    """The fitted table's columns plus ``name``, and the rates table's (total)."""
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
        out = [] if not layer else [(dom, layer, c) for c in dict.fromkeys(list(cols) + [name])
                                    if c]
        rates = name + RATES_SUFFIX
        out += [(Domain.LABEL, rates, c) for c in ("id", "m", "t", "c", "z", "y", "x")
                + _RATE_COLUMNS]
        return tuple(out)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_psf_columns.wants_inputs = True

_FIXED = {"diffusivity": frozenset({"fixed"})}

register_node(
    _compute_point_source_fit, op_key="table.point_source_fit", label="Point-Source Fit (Crank)",
    category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table holds the concentrations "
                                      "measured round the source (Standard Curve's output); "
                                      "passed on with the model's value per row and a rates "
                                      "table added."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="The table fitted. Blank uses the only one of its kind on the "
                             "wire."),
        InString("distance", "Distance", field=False, default="distance_um",
                 column_in_mode="domain", column_from="table",
                 description="Each row's distance from the source, in µm — the 3-D distance "
                             "to a spheroid's centroid. The model is radial, so a 2-D "
                             "distance on a sample with depth makes far-above beads look "
                             "near and pulls D down."),
        InString("value", "Value", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="The measured concentration, in pg/mL — the unit the model and "
                             "the reported rates (fg/s) assume. Required."),
        InString("group", "Group", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="The column naming each row's timepoint (Table Concat's "
                             "`condition`): one time per group, and under Rate = constant one "
                             "rate per group. Blank groups by frame `t`."),
        InString("time", "Time", field=False, default="", column_in_mode="domain",
                 column_from="table",
                 description="Each row's acquisition time, read as Time kind says. Blank uses "
                             "Table Concat's `time_jd`, each row's own clock — so a bead "
                             "imaged 20 min into a run is fitted at its own time."),
        InFloat("start_h", "Source started", unit="h", field=False, default=0.0,
                description="When the source began releasing. Time kind = clock: hours BEFORE "
                            "the earliest row (a first timepoint 15 h after seeding: 15). "
                            "hours / seconds: added to the column. Moves every rate: the "
                            "same profile read as older needs a smaller rate, and the "
                            "earliest timepoints move most."),
        InFloat("D", "D", unit="um2/s", field=False, default=20.0, available_in=_FIXED,
                description="The analyte's diffusivity in the gel. 20 µm²/s is McGhee et "
                            "al. 2022's measured value for IL-8 with its capture pair "
                            "(± 10). LARGER D spreads the same release farther, so the "
                            "near beads need a larger rate and the far ones a smaller."),
        InFloat("fit_floor", "Fit floor", unit="", field=False, default=0.0,
                description="Values below this (in the value's units) are not fitted — "
                            "readings at the noise floor, which on a log fit would weigh like "
                            "a hundred-fold miss. They still get the model's value. 0 keeps "
                            "every positive value."),
        InInt("bootstrap", "Bootstrap", field=False, default=200,
              description="Resamples (beads within each timepoint) for each rate's 95 % "
                          "interval, at the fitted D. 0 skips it; more steadies the interval "
                          "and costs a refit each."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="The new column holding the model's concentration at each row; "
                             "`<name>_rates` is the Label table of fitted rates (fg/s and pg/h, "
                             "with intervals) a plot can read. A name the table has is "
                             "refused."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_PSF_DOMAINS), default="label", label="Domain",
             description="Which kind of table is fitted.",
             choice_docs=domain_docs(_PSF_DOMAINS)),
        Mode("rate", list(PS.RATE_MODELS), default="constant", label="Rate",
             description="How the production rate may change with time.",
             choice_docs={
                 "constant": "One rate per timepoint, fitted to that timepoint alone: the "
                             "source's AVERAGE rate from its start to that time (McGhee et "
                             "al. 2022's Eq. 2, as published). Stable with few beads; a "
                             "rising average means the rate itself rose.",
                 "piecewise": "One rate per interval between successive timepoints, every "
                              "timepoint fitted together by superposition — the rate's "
                              "history, each profile carrying all that was released before "
                              "it. Needs well-sampled timepoints: with few beads per time, "
                              "or the same beads re-imaged, the intervals trade off and "
                              "swing between 0 and large values.",
             }),
        Mode("diffusivity", list(_DIFFUSIVITY), default="fixed", label="Diffusivity",
             description="Where D comes from.",
             choice_docs={
                 "fixed": "D as typed (default 20 µm²/s, the paper's IL-8 value) — rates "
                          "comparable with the paper's.",
                 "fit": "One D for every timepoint, chosen by profile (0.5–500 µm²/s) with the "
                        "rates refitted at each — what the data say, which also absorbs "
                        "anything else that blurs the profile (depth, bleaching).",
             }),
        Mode("boundary", list(PS.BOUNDARIES), default="infinite", label="Boundary",
             description="What surrounds the source.",
             choice_docs={
                 "infinite": "An unbounded medium on every side of the source — the paper's "
                             "Eq. 2 as published; the rates comparable with the paper's.",
                 "half_space": "The source on an impermeable floor (a spheroid on the well "
                               "bottom): every concentration doubles, so the same data give "
                               "HALF the rate. D is unaffected.",
             }),
        Mode("background", list(PS.BACKGROUNDS), default="none", label="Background",
             description="Whether the profile may level off above zero far from the source.",
             choice_docs={
                 "none": "The point source alone: far from it the model falls to 0 — the "
                         "paper's Eq. 2, an unbounded medium.",
                 "uniform": "Plus one non-negative level per timepoint: what has spread "
                            "through a FINITE well, whose walls return what an unbounded "
                            "medium lets escape — the profile levels off where late "
                            "timepoints read evenly far out. Lowers the near-field rate "
                            "that alone had to explain the far beads.",
             }),
        Mode("weight", list(PS.WEIGHTS), default="relative", label="Weighting",
             description="How each row's miss is counted.",
             choice_docs={
                 "relative": "Fit ln C: every row weighs by its RELATIVE miss, so a profile "
                             "spanning decades is not fitted to its brightest few beads "
                             "(the exponential trendline's default). Rows ≤ 0 are left out.",
                 "sqrt": "Fit √C: each row weighs by its miss over the square root of its "
                         "level (counting noise) — between relative and absolute, so dim beads "
                         "still count but a few near-zero readings cannot pin the tail.",
                 "absolute": "Fit pg/mL: every row weighs by its absolute miss — the brightest "
                             "beads decide, which is how the eye judges a linear plot.",
             }),
        Mode("time_kind", list(_TIME_KINDS), default="clock", label="Time kind",
             description="How the Time column is read.",
             choice_docs={
                 "clock": "A Julian date (Table Concat's `time_jd`): times are measured from "
                          "the earliest row, which is Source started hours after the source "
                          "began.",
                 "hours": "Hours already elapsed since a reference; Source started is added.",
                 "seconds": "Seconds already elapsed since a reference; Source started (h) is "
                            "added.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.LABEL}),
    extra_layers=_psf_layers, adds_columns=_psf_columns,
    description="Fit Crank's continuous point source (McGhee et al. 2022, Eq. 2) to "
                "concentrations measured round a secreting source — capture beads round a "
                "spheroid — for its production rate per timepoint (or per interval) with "
                "bootstrap intervals, and the diffusivity if asked; adds the model's value "
                "per row and a rates table, and records the fit for Plot XY's and Gradient "
                "Map's trend.")
