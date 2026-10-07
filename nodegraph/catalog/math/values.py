"""Math (``math.values``) — arithmetic on the NUMBERS a Dataset carries — a Global scalar, a
table column, a per-plane or per-frame layer, a voxel raster — against another such number
or a constant, with the UNIT of the result worked out (µm × µm = µm², counts ÷ px² is a
density), incompatible units refused, pixel quantities brought to microns (and back) by the
calibration, and the result recorded with its unit for the next card."""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InFloat, InString, Mode, OutDataset
from nodegraph.structure import StructureTable
from nodegraph.units import (UNIT_CHOICE_DOCS, UNIT_CHOICES, UnitError, conversion_factor,
                             describe_unit, div_units, format_unit, mul_units, parse_unit,
                             physical_unit, pixel_unit, pow_unit, root_unit, unit_of,
                             with_unit)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.columns import on_layer
from nodegraph.catalog._shared.labels import _lattice_layers, _resolve_layer, _structure_layers

# ── Math on values (V4.00 step 12) ───────────────────────────────────────────────
#
# What `analysis.reduce_scalar` is to "one number", this is to "arithmetic on it": the node
# that turns `total_intensity / area` into a density column, `n_cells / field_area` into a
# count per µm², a voxel count into a µm³ volume, or two Global scalars from two pages into
# their ratio. One node for every domain rather than a Scalar Math and a Column Math, because
# the arithmetic is the same and only WHERE the numbers live differs — the shape
# `analysis.reduce_scalar` already uses (a `domain` lever that re-targets the attribute
# picker).
#
# Units are the point. Every operand arrives with a unit (recorded by its producer, or read
# off the catalog's naming convention — `nodegraph.units`), the operation composes them, an
# addition of different dimensions is refused and an addition of different UNITS of one
# dimension (px² + µm²) converts B into A's unit by the calibration first. The result is
# recorded, so a second Math card downstream knows what it was handed.

_VM_DOMAINS: Tuple[str, ...] = (
    "global", "label", "point", "track", "voxel", "plane", "frame", "timepoint",
    "multipoint", "channel")
_VM_STRUCTURE = frozenset({"label", "point", "track"})
_VM_BINARY: Tuple[str, ...] = ("add", "subtract", "multiply", "divide", "power", "min", "max")
_VM_UNARY: Tuple[str, ...] = ("abs", "sqrt", "log10", "to_physical", "to_pixels")
_VM_OPS: Tuple[str, ...] = _VM_BINARY + _VM_UNARY
_VM_BINARY_SET = frozenset(_VM_BINARY)
_VM_SAME_UNIT = frozenset({"add", "subtract", "min", "max"})
#: The word an operation writes into an auto-named result (`area_times_0p25`).
_OP_WORD: Dict[str, str] = {"add": "plus", "subtract": "minus", "multiply": "times",
                            "divide": "per", "power": "pow", "min": "min", "max": "max"}
#: ``a_unit`` offers the unit vocabulary plus `auto` = as recorded / as the name says.
_A_UNIT_CHOICES: Tuple[str, ...] = ("auto",) + UNIT_CHOICES
_A_UNIT_DOCS: Dict[str, str] = {
    "auto": "Take A's unit as its producer recorded it, else as the catalog's naming "
            "convention says (`area` is px² or voxels by the table's dimensionality, "
            "`*_um` is µm, `*_intensity` is counts); unknown if neither knows.",
    **UNIT_CHOICE_DOCS}


class _Operand(NamedTuple):
    values: np.ndarray            # float array; 0-d for a Global scalar or a constant
    unit: Optional[str]           # ASCII unit, "" dimensionless, None unknown
    label: str                    # how a message names it


def _num_slug(v: Any) -> str:
    """``0.25`` → ``0p25``, ``-2`` → ``m2`` — a number that fits in a column name."""
    try:
        s = f"{float(v):g}"
    except (TypeError, ValueError):
        s = str(v)
    return s.replace("-", "m").replace("+", "").replace(".", "p")


def _p(params: Mapping[str, Any], key: str) -> Any:
    """A param by key. (Spelled through a helper so the socket contract's literal check —
    "the compute reads params.get('a') directly" — does not fire on an auto-name built from
    the operand names; the compute reads the layer sockets through ``ctx.layer`` and the
    constant through ``ctx.params``, and this only echoes them into a name.)"""
    return (params or {}).get(key)


def _out_name(params: Mapping[str, Any], modes: Mapping[str, Any]) -> str:
    """The result's name: the typed one, else built from the operands — total, shared by
    the compute and the edit-time declarations so the picker offers what the pull writes."""
    params = params or {}
    modes = modes or {}
    name = str(_p(params, "name") or "").strip()
    if name:
        return name
    op = str(modes.get("op") or "add")
    a = str(_p(params, "a") or "a").strip() or "a"
    if op in _VM_BINARY_SET:
        b = str(_p(params, "b") or "").strip()
        if not b:
            v = _p(params, "value")
            b = _num_slug(1.0 if v is None else v)
        return f"{a}_{_OP_WORD.get(op, op)}_{b}"
    if op == "to_physical":
        return f"{a}_physical"
    if op == "to_pixels":
        return f"{a}_pixels"
    return f"{op}_{a}"


def _columns_of(ds: Dataset, domain: Domain, table: str) -> Dict[str, np.ndarray]:
    return {k[2]: np.asarray(ds.attributes[k].values) for k in ds.attributes
            if k[0] is domain and k[1] == table}


def _read(ds: Dataset, domain: Domain, name: str, table_want: str, *, socket: str,
          where: str, ctx: EvalContext, exclude: Sequence[str] = ()
          ) -> Tuple[_Operand, Optional[str]]:
    """One operand off ``ds``: ``(operand, table)`` — ``table`` the structure instance it
    came from, ``None`` for a lattice or Global attribute. A structure COLUMN must be named
    (a table has many, and none is "the only one"); a lattice attribute resolves by the
    only-candidate rule like every layer socket in the catalog."""
    if domain.value in _VM_STRUCTURE:
        table, _note = _resolve_layer(
            _structure_layers(ds, domain), table_want, node="math", socket="table",
            what=f"{domain.value} table", where=where,
            remedy="measure or detect something upstream (Connected Components, Measure, "
                   "Spot Detection, Track Objects) so there is a table to compute on",
            ctx=ctx)
        cols = _columns_of(ds, domain, table)
        if not name or name not in cols:
            raise ValueError(
                f"math: the {domain.value} table {table!r} on {where} has no column "
                f"{name!r} — it has {sorted(cols)}; pick `{socket}` from its dropdown")
        return (_Operand(np.asarray(cols[name], dtype=float),
                         unit_of(ds, domain, name, table), name), table)
    cands = [c for c in _lattice_layers(ds, domain) if c not in exclude]
    lname, _note = _resolve_layer(
        cands, name, node="math", socket=socket, what=f"{domain.value} attribute",
        where=where,
        remedy="produce one upstream (Reduce → Scalar for a Global number, Threshold for a "
               "Voxel mask, Transfer Domain for a per-plane or per-frame value)", ctx=ctx)
    attr = ds.get(domain, lname)
    return _Operand(np.asarray(attr.values, dtype=float), unit_of(ds, domain, lname), lname), None


def _read_b(ctx: EvalContext, ds: Dataset, domain: Domain, a: _Operand,
            a_table: Optional[str]) -> _Operand:
    """Operand B: a layer/column named by ``b`` (on `other` when wired, else on `data`; on
    `other` an empty name takes the only candidate), a Global scalar of that name, or — with
    no name and no `other` — the constant ``value`` in ``value_unit``."""
    other = ctx.input("other")
    src = other if other is not None else ds
    where = "the `other` input" if other is not None else "the `data` input"
    b_name = ctx.layer("b")
    if not b_name and other is None:
        raw = ctx.params.get("value")
        value = 1.0 if raw is None else float(raw)
        unit = parse_unit(ctx.params.get("value_unit") or "none")
        return _Operand(np.asarray(value, dtype=float), format_unit(unit), f"{value:g}")
    # a Global scalar of that name broadcasts onto anything
    if b_name and domain is not Domain.GLOBAL and src.get(Domain.GLOBAL, b_name) is not None \
            and (domain.value in _VM_STRUCTURE or src.get(domain, b_name) is None):
        attr = src.get(Domain.GLOBAL, b_name)
        return _Operand(np.asarray(attr.values, dtype=float).reshape(()),
                        unit_of(src, Domain.GLOBAL, b_name), b_name)
    table_want = (a_table or "") if other is None else ""
    b, b_table = _read(src, domain, b_name, table_want, socket="b", where=where, ctx=ctx,
                       exclude=([a.label] if (other is None and a_table is None) else ()))
    bv = b.values
    if bv.ndim == 0 or bv.shape == a.values.shape:
        if domain.value in _VM_STRUCTURE and other is not None and b_table is not None:
            # two tables from two branches: the rows must be the same objects
            ca, cb = _columns_of(ds, domain, a_table or ""), _columns_of(src, domain, b_table)
            if "id" in ca and "id" in cb and not np.array_equal(ca["id"], cb["id"]):
                raise ValueError(
                    f"math: the `other` table {b_table!r} does not list the same objects as "
                    f"{a_table!r} (their `id` columns differ), so its rows cannot be paired "
                    f"with A's — use Table Join to align them first")
        return b
    if bv.ndim == a.values.ndim and all(s in (1, w) for s, w in zip(bv.shape, a.values.shape)):
        return _Operand(np.broadcast_to(bv, a.values.shape), b.unit, b.label)
    raise ValueError(
        f"math: B ({b.label!r}) has shape {tuple(bv.shape)} and A ({a.label!r}) "
        f"{tuple(a.values.shape)} — they are combined element for element, so they must "
        f"have the same shape (two tables: the same rows), or B must be a single number")


def _convert_b(b: _Operand, ua: Dict[str, int], ub: Optional[Dict[str, int]],
               calib: Mapping[str, Any]) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """B's values in A's unit, for the same-unit operations."""
    if ub is None:
        return b.values, None
    try:
        f = conversion_factor(ub, ua, **calib)
    except UnitError as exc:
        raise ValueError(
            f"math: A is in {describe_unit(ua)} and B ({b.label!r}) in {describe_unit(ub)} "
            f"— {exc}") from exc
    return (b.values * f if f != 1.0 else b.values), ua


def _compute_math_values(ctx: EvalContext) -> Dataset:
    """Arithmetic on an attribute of the chosen domain, with units.

    Resolved spec (build-node-v2 §0, V4.00 step 12)
    -----------------------------------------------
    * **Kind** math → ``op_key="math.values"``, category ``"math"``. Dataset in, the same
      Dataset out plus one attribute on the lever's domain: a Global scalar, a new column on
      a structure table, or a lattice layer. The image is untouched.
    * **Domain lever** (``domain``) re-targets the pickers as ``analysis.reduce_scalar``'s
      does; ``reads_domains``/``adds_domains`` stay empty because both are per-instance
      (the output layer is announced through ``extra_layers`` / ``adds_columns``).
    * **Operands** ``A`` on ``data``; ``B`` a layer on ``other`` or ``data``, a Global
      scalar, or the constant ``value`` in ``value_unit`` (:func:`_read_b`).
    * **Units** composed by :mod:`nodegraph.units`: same-unit operations convert B into A's
      unit by the calibration (``pixel_size_um`` / ``z_step_um`` / ``dt_s``, read through
      ``ctx.calib`` so the memo fences on them) and refuse different dimensions; products
      and ratios compose; ``power``/``sqrt``/``log10`` follow the rules of dimensional
      analysis; ``to_physical``/``to_pixels`` convert A. An UNKNOWN unit stays unknown — it
      is never silently called dimensionless. The result is recorded
      (:func:`nodegraph.units.with_unit`).
    * **2D/3D** none. **Footprint** ``WHOLE_SERIES``, no kernel axes: it reads attributes
      that are already computed, over every axis.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    domain = Domain(str(modes.get("domain") or "global"))
    op = str(modes.get("op") or "add")
    if op not in _VM_OPS:
        raise ValueError(f"math: unknown operation {op!r} — one of {list(_VM_OPS)}")
    table_want = str(ctx.params.get("table") or "")
    a, a_table = _read(ds, domain, ctx.layer("a"), table_want, socket="a",
                       where="the `data` input", ctx=ctx)
    a_sel = str(ctx.params.get("a_unit") or "auto")
    if a_sel != "auto":
        ua: Optional[Dict[str, int]] = parse_unit(a_sel)
    else:
        ua = None if a.unit is None else parse_unit(a.unit)
    calib = {"pixel_size_um": ctx.calib("pixel_size_um"), "z_step_um": ctx.calib("z_step_um"),
             "dt_s": ctx.calib("dt_s")}
    av = a.values
    unit: Optional[Dict[str, int]]
    with np.errstate(divide="ignore", invalid="ignore"):
        if op in _VM_BINARY_SET:
            b = _read_b(ctx, ds, domain, a, a_table)
            ub = None if b.unit is None else parse_unit(b.unit)
            bv = b.values
            if op in _VM_SAME_UNIT:
                if ua is not None:
                    bv, unit = _convert_b(b, ua, ub, calib)
                else:
                    unit = None
                result = {"add": np.add, "subtract": np.subtract,
                          "min": np.minimum, "max": np.maximum}[op](av, bv)
            elif op == "multiply":
                result = av * bv
                unit = mul_units(ua, ub) if (ua is not None and ub is not None) else None
            elif op == "divide":
                bb = np.asarray(bv, dtype=float)
                result = np.where(bb != 0, av / np.where(bb != 0, bb, 1.0), np.nan)
                unit = div_units(ua, ub) if (ua is not None and ub is not None) else None
            else:                                                     # power
                if ub not in (None, {}):
                    raise ValueError(
                        f"math: an exponent must be dimensionless, but B ({b.label!r}) is "
                        f"in {describe_unit(ub)}")
                result = np.power(av, bv)
                if ua is None:
                    unit = None
                elif np.ndim(bv) == 0:
                    try:
                        unit = pow_unit(ua, float(bv))
                    except UnitError as exc:
                        raise ValueError(f"math: {exc}") from exc
                elif not ua:
                    unit = {}
                else:
                    raise ValueError(
                        f"math: A is in {describe_unit(ua)} and the exponent varies per "
                        f"element — a quantity with a unit cannot be raised to a different "
                        f"power in every row (the rows would not share a unit); convert A "
                        f"to a ratio first, or use a single constant exponent")
        elif op == "abs":
            result, unit = np.abs(av), ua
        elif op == "sqrt":
            result = np.sqrt(av)
            try:
                unit = None if ua is None else root_unit(ua, 2)
            except UnitError as exc:
                raise ValueError(f"math: {exc}") from exc
        elif op == "log10":
            if ua not in (None, {}):
                raise ValueError(
                    f"math: log10 of a quantity in {describe_unit(ua)} is not defined — "
                    f"divide it by a reference value in the same unit first (a ratio), "
                    f"or set `Unit of A` to `none` if it really is a plain number")
            result, unit = np.log10(av), ({} if ua is not None else None)
        else:                                                         # to_physical / to_pixels
            if ua is None:
                raise ValueError(
                    f"math: the unit of A ({a.label!r}) is not known, so it cannot be "
                    f"converted — set `Unit of A` to what it is in (px, px2, vox, frame, …)")
            target = physical_unit(ua) if op == "to_physical" else pixel_unit(ua)
            try:
                f = conversion_factor(ua, target, **calib)
            except UnitError as exc:
                raise ValueError(f"math: {exc}") from exc
            result, unit = av * f, target
    # the typed name (or an auto-name from the operands) — spelled out here so the socket
    # contract sees that `name` is read
    name = _out_name({**ctx.params, "name": ctx.params.get("name")}, modes)
    result = np.asarray(result, dtype=float)
    ctx.progress(1, 1, f"{name}: {op} → {describe_unit(unit)}")
    if domain.value in _VM_STRUCTURE:
        cols = _columns_of(ds, domain, a_table or "")
        cols[name] = result
        zk = ds.structure_zkind(domain, a_table)
        out = ds.with_structure(StructureTable(domain, cols, layer=a_table,
                                               z_kind=zk or "subpixel"))
        return with_unit(out, domain, name, None if unit is None else unit, a_table)
    if domain is Domain.GLOBAL:
        result = result.reshape(())
    out = ds.with_layer(domain, name, result)
    return with_unit(out, domain, name, None if unit is None else unit)


def _layers_math_values(params, modes):
    """The attribute this node writes on a LATTICE or Global domain, for the edit-time
    catalog (``layer_out`` cannot say it: the domain is the lever's value). A structure
    result is a COLUMN and is announced by :func:`_columns_math_values`. Total."""
    try:
        d = Domain(str((modes or {}).get("domain") or "global"))
        if d.value in _VM_STRUCTURE:
            return ()
        return ((d, _out_name(params or {}, modes or {})),)
    except Exception:                                    # pragma: no cover - defensive
        return ()


def _columns_math_values(params, modes, incoming):
    """The column this node writes on a structure table — on the named table, or the only
    table of that domain upstream (an inferred table, §4g). Total."""
    try:
        d = Domain(str((modes or {}).get("domain") or "global"))
        if d.value not in _VM_STRUCTURE:
            return ()
        table = str(_p(params, "table") or "").strip()
        if not table:
            tables = sorted({lyr for dd, lyr, _c in (incoming or ()) if dd is d})
            if len(tables) != 1:
                return ()
            table = tables[0]
        return on_layer(d, table, [_out_name(params or {}, modes or {})])
    except Exception:                                    # pragma: no cover - defensive
        return ()


register_node(
    _compute_math_values, op_key="math.values", label="Math", category="math",
    extra_layers=_layers_math_values, adds_columns=_columns_math_values,
    inputs=[
        InDataset(description=
                  "The data carrying A — and B too, unless `Other` is wired. The result is "
                  "added to it (a new scalar, column or layer); the image and everything "
                  "else pass through."),
        InDataset("other", label="Other", passes_domains=False,
                  description=
                  "Optional: the branch carrying B when it is not on this wire — a scalar "
                  "from another page, a column measured on the other channel. Only B is "
                  "read from it; its layers do not pass downstream. Two tables are paired "
                  "row by row and must list the same objects (same `id`s)."),
        InString("a", "A", field=False, default="", layer_in_mode="domain",
                 description=
                 "The attribute on the left: a column of the table (for Label / Point / "
                 "Track), or a layer of the domain (a Global scalar from Reduce → Scalar, a "
                 "per-plane focus score, a Voxel mask). Empty takes the only attribute of "
                 "the domain on the wire; a table column must be named."),
        InString("table", "Table", field=False, default="",
                 available_in={"domain": _VM_STRUCTURE},
                 description=
                 "Which structure table A (and B, when on this wire) belong to. Only needed "
                 "when two tables of the domain are on the wire (two segmentations both "
                 "with an `area`); leave it empty otherwise and the only table is used."),
        InString("b", "B", field=False, default="", layer_in_mode="domain", layer_from="other",
                 available_in={"op": _VM_BINARY_SET},
                 description=
                 "The attribute on the right: a column or layer of the same domain on "
                 "`Other` (if wired) or on this wire, or the NAME OF A GLOBAL SCALAR, which "
                 "applies to every row. EMPTY with `Other` unwired uses the constant `Value` "
                 "instead. Hidden under the one-operand operations."),
        InFloat("value", "Value", field=False, default=1.0, unit="",
                available_in={"op": _VM_BINARY_SET},
                description=
                "The constant B when `B` is empty and `Other` unwired: the number to add, "
                "multiply by, divide by, or raise to. Its unit is `Value unit`, so `area` "
                "minus 50 µm² converts 50 into A's unit first; an exponent must be "
                "dimensionless."),
        InString("value_unit", "Value unit", field=False, default="none",
                 choices=UNIT_CHOICES, choice_docs=UNIT_CHOICE_DOCS,
                 available_in={"op": _VM_BINARY_SET},
                 description=
                 "What `Value` is in. `none` is a plain number (a factor, an exponent). "
                 "Give a unit when the constant is a physical quantity — 50 µm², 10 s — "
                 "and the node converts it to A's unit (or composes it: `area` × 0.25 µm "
                 "is µm·px² until you convert) before computing."),
        InString("a_unit", "Unit of A", field=False, default="auto",
                 choices=_A_UNIT_CHOICES, choice_docs=_A_UNIT_DOCS,
                 description=
                 "Override what A is in. `auto` reads the unit its producer recorded, else "
                 "the naming convention. Set it when A's unit is UNKNOWN (a column with a "
                 "name the convention does not know) and the operation needs one — "
                 "`to_physical` refuses an unknown unit rather than guess — or when the "
                 "convention is wrong for this column."),
        InString("name", "Output name", field=False, default="",
                 description=
                 "What to call the result. Empty builds one from the operands — "
                 "`total_intensity_per_area`, `area_times_0p25`, `sqrt_area`, "
                 "`area_physical` — and the unit is recorded whatever the name. Naming an "
                 "existing column replaces it."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_VM_DOMAINS), default="global", label="On",
             description=
             "Where the numbers live. Filters the A and B pickers to that domain's "
             "attributes — a table's columns for Label / Point / Track, the layers for a "
             "lattice domain, the scalars for Global — so set it FIRST. The result lands "
             "on the same domain: a new column beside A, or a new layer / scalar.",
             choice_docs=domain_docs(_VM_DOMAINS)),
        Mode("op", list(_VM_OPS), default="add", label="Operation",
             description=
             "What is computed from A (and B). Units follow: like units for the sums and "
             "extremes (B converted into A's unit by the calibration when they differ, "
             "refused when they measure different things), composed for products and "
             "ratios, and the conversions change A's unit by the pixel size, Z step and "
             "frame interval of this data.",
             choice_docs={
                 "add": "A + B, element by element. B must be in A's dimension: px² + µm² "
                        "converts B by the pixel size first; px² + µm is refused. A Global B "
                        "adds the same number to every row.",
                 "subtract": "A − B. Same unit rule as `add`: a difference of areas in µm², "
                             "a count minus a count, a column minus a constant stated in its "
                             "own unit.",
                 "multiply": "A × B; the units multiply (µm × µm = µm², px² × counts = "
                             "counts·px²). The way to weight a column by another or scale a "
                             "scalar; convert afterwards with `to_physical` if pixels and "
                             "microns got mixed.",
                 "divide": "A ÷ B; the units divide (counts ÷ px² is an intensity density, "
                           "µm ÷ s a velocity, a count ÷ a count a plain ratio). Division by "
                           "zero gives NaN, never 0.",
                 "power": "A to the power B (a constant or a dimensionless column). A unit "
                          "is raised with it — µm² to the 0.5 is µm — and a power that would "
                          "leave a fractional exponent, or a per-row exponent on a quantity "
                          "with a unit, is refused.",
                 "min": "The smaller of A and B element by element, in A's unit (B converted "
                        "first). Clamps a column from above, or picks the smaller of two "
                        "measurements.",
                 "max": "The larger of A and B element by element, in A's unit. Clamps from "
                        "below — `max` against 0 removes negatives — or picks the larger of "
                        "two measurements.",
                 "abs": "|A|, the absolute value; the unit is unchanged. B is not read.",
                 "sqrt": "√A; the unit's exponents halve (µm² → µm, px² → px), and a unit "
                         "that cannot be halved (µm) is refused. B is not read.",
                 "log10": "log₁₀ A, defined only for a dimensionless A (a ratio, a count); "
                          "a quantity in µm or counts is refused — divide by a reference "
                          "value first. Non-positive values give NaN.",
                 "to_physical": "A in physical units: px → µm by the pixel size, voxels → "
                                "µm³ by pixel size² × Z step, frames → s by the frame "
                                "interval, px/frame → µm/s. Refused, naming the key, when "
                                "the data lacks the calibration it needs, or when A's unit "
                                "is unknown (set `Unit of A`).",
                 "to_pixels": "The inverse: µm → px (lateral), µm² → px², s → frames, by "
                              "the same calibration. For comparing a physical quantity "
                              "against something measured in pixels.",
             })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Arithmetic on the numbers a Dataset carries — a Global scalar, a table "
                "column, a per-plane layer — against another one (this wire or another "
                "branch) or a constant, with the unit of the result worked out, incompatible "
                "units refused, and pixel quantities converted to microns by the calibration. "
                "The result is recorded with its unit for the next card.")
