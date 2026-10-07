"""Units — the small algebra that keeps a number honest about what it measures (V4.00 step 12).

A Dataset carries numbers in three places that arithmetic can reach — a Global scalar
(``analysis.reduce_scalar``'s count or total), a structure-table column (``area``,
``mean_intensity``, ``x_um``), a lattice layer (a mask, a per-plane focus score) — and until
this module nothing said what UNIT any of them was in. The catalog had a convention
(``_um`` suffixes; ``area`` is in voxels; a velocity is µm/s) but no reader of it, so a node
that multiplied two columns could not say what came out, and a node that subtracted a µm
length from a pixel one would have done so silently.

This module is that reader, plus the algebra:

* **A unit is a dict** ``{base symbol: exponent}`` over a handful of bases — ``um``, ``px``
  (a lateral pixel), ``zpx`` (an axial plane step), ``s``, ``frame``, ``counts`` (camera
  intensity), ``rad`` — with a few scaled spellings (``nm``, ``mm``, ``ms``, ``min``, ``h``,
  ``deg``) that reduce to them by a fixed factor. ``vox`` is ``px²·zpx``. Dimensionless is
  the empty dict. :func:`parse_unit` reads every spelling a user or a column name is likely
  to use (``um2``, ``µm²``, ``um^2``, ``um/s``, ``1/s``); :func:`format_unit` writes one
  canonical ASCII spelling (and a pretty one for the screen); :func:`unit_slug` the form that
  fits in a column name (``um2``, ``um_per_s``).
* **Composition** is ordinary: :func:`mul_units`, :func:`div_units`, :func:`pow_unit`,
  :func:`root_unit`. Addition needs the SAME dimension, and :func:`conversion_factor` says
  how to bring one operand into the other's unit — a fixed factor between ``nm`` and ``um``,
  the CALIBRATION between a pixel and a micron (``pixel_size_um`` laterally, ``z_step_um``
  axially, ``dt_s`` between a frame and a second). A conversion that needs a calibration the
  Dataset does not carry is refused by name rather than guessed at 1:1 — the same rule
  :func:`nodegraph.catalog._shared.units.to_pixels_v2`'s callers already live by.
* **Where a number's unit is written down.** Two places, read in this order by
  :func:`unit_of`: an explicit record in the Dataset's metadata under :data:`UNITS_KEY`
  (``{"<domain>:<layer>:<name>": "<unit>"}``, written by :func:`with_unit` — the math nodes
  and ``analysis.reduce_scalar`` record what they produce), and failing that the catalog's
  NAME convention (:func:`unit_of_column`: ``area`` is ``px2`` on a 2D table and ``vox`` on
  a 3D one, ``x``/``y`` are ``px``, ``t`` is ``frame``, ``*_um`` is ``um``, ``*_intensity``
  is ``counts``, …). A name the convention does not know has unit ``None`` — UNKNOWN, which
  the math nodes carry forward as unknown rather than silently calling it dimensionless.

Qt-free; stdlib + the Dataset type only. Nothing here touches a pixel.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain

__all__ = [
    "UNITS_KEY", "UnitError", "Unit", "CANONICAL_BASES", "UNIT_CHOICES", "UNIT_CHOICE_DOCS",
    "parse_unit", "format_unit", "unit_slug", "describe_unit",
    "mul_units", "div_units", "pow_unit", "root_unit", "same_dimension",
    "conversion_factor", "physical_unit", "pixel_unit",
    "unit_of_column", "unit_key", "recorded_unit", "unit_of", "with_unit",
]

#: The metadata key under which a Dataset records the unit of a scalar, column or layer:
#: ``{unit_key(domain, layer, name): "<ascii unit>"}``. Non-calibration provenance — it
#: rides ``ds.metadata`` untouched through every compute that keeps the numbers, like
#: ``source_file`` does, and a stale entry for a layer that was dropped is harmless.
UNITS_KEY = "units"

#: A unit: ``{base symbol: integer exponent}``; ``{}`` is dimensionless.
Unit = Dict[str, int]


class UnitError(ValueError):
    """A unit that cannot be read, composed or converted — always says which and why."""


#: The bases every unit reduces to. ``px`` is a LATERAL pixel and ``zpx`` an axial plane
#: step: they are different lengths on every microscope that is not isotropic, which is why
#: a voxel is ``px²·zpx`` and not ``px³``, and why converting either to microns needs its own
#: calibration key.
CANONICAL_BASES: Tuple[str, ...] = ("um", "px", "zpx", "s", "frame", "counts", "rad")

#: Scaled spellings → ``(canonical base, factor)``: one of these equals ``factor`` of the base.
_SCALED: Dict[str, Tuple[str, float]] = {
    "nm": ("um", 1e-3), "mm": ("um", 1e3),
    "ms": ("s", 1e-3), "min": ("s", 60.0), "h": ("s", 3600.0),
    "deg": ("rad", math.pi / 180.0),
}

#: Every accepted spelling of a base or scaled symbol → the symbol it means. ``vox`` is
#: handled separately (it is a compound).
_ALIASES: Dict[str, str] = {
    "um": "um", "µm": "um", "μm": "um", "micron": "um", "microns": "um", "micrometer": "um",
    "micrometre": "um",
    "nm": "nm", "nanometer": "nm", "nanometre": "nm",
    "mm": "mm", "millimeter": "mm", "millimetre": "mm",
    "px": "px", "pixel": "px", "pixels": "px",
    "zpx": "zpx", "plane": "zpx", "planes": "zpx", "slice": "zpx", "slices": "zpx",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "frame": "frame", "frames": "frame", "fr": "frame",
    "counts": "counts", "count": "counts", "adu": "counts", "intensity": "counts",
    "rad": "rad", "radian": "rad", "radians": "rad",
    "deg": "deg", "degree": "deg", "degrees": "deg",
}
_VOXEL_WORDS = frozenset({"vox", "voxel", "voxels"})
_DIMENSIONLESS_WORDS = frozenset({"", "1", "ratio", "dimensionless", "none", "fraction", "-"})
_SUPERSCRIPTS = {"²": 2, "³": 3, "⁻¹": -1, "⁻²": -2, "⁻³": -3}
#: The order symbols are written in, so one unit has exactly one spelling.
_ORDER = ("um", "nm", "mm", "px", "zpx", "s", "ms", "min", "h", "frame", "counts", "rad", "deg")
_PRETTY = {"um": "µm", "deg": "°"}

_TOKEN_RE = re.compile(r"^([A-Za-zµμ]+?)(?:\^)?(-?\d+)?$")


def _clean(u: Mapping[str, int]) -> Unit:
    return {k: int(v) for k, v in u.items() if int(v) != 0}


def parse_unit(text: Union[str, Mapping[str, int], None]) -> Unit:
    """A unit string → its dict. Accepts ``um2``, ``µm²``, ``um^2``, ``um/s``, ``1/s``,
    ``px2·zpx``, ``vox``, ``counts``, ``um*um/s``; ``""``/``ratio``/``1`` is dimensionless.
    A dict is returned cleaned. Anything else raises :class:`UnitError` naming the token."""
    if text is None:
        return {}
    if isinstance(text, Mapping):
        return _clean(text)
    s = str(text).strip()
    if s.lower() in _DIMENSIONLESS_WORDS:
        return {}
    for sup, val in _SUPERSCRIPTS.items():
        s = s.replace(sup, f"^{val}")
    s = s.replace("·", " ").replace("*", " ").replace("×", " ")
    out: Dict[str, int] = {}
    sign = 1
    for part in re.split(r"(/)", s):
        part = part.strip()
        if part == "/":
            sign = -1
            continue
        if not part:
            continue
        for tok in part.split():
            if tok.lower() in ("1", "one"):
                continue
            m = _TOKEN_RE.match(tok)
            if m is None:
                raise UnitError(f"cannot read the unit {text!r}: {tok!r} is not a unit token")
            word, exp = m.group(1), int(m.group(2) or 1)
            key = word.lower() if word.lower() in _ALIASES or word.lower() in _VOXEL_WORDS \
                else word
            if key in _VOXEL_WORDS or key.lower() in _VOXEL_WORDS:
                out["px"] = out.get("px", 0) + 2 * exp * sign
                out["zpx"] = out.get("zpx", 0) + exp * sign
                continue
            sym = _ALIASES.get(key) or _ALIASES.get(word) or _ALIASES.get(word.lower())
            if sym is None:
                raise UnitError(
                    f"cannot read the unit {text!r}: {word!r} is not a known unit (known: "
                    f"um nm mm px zpx vox s ms min h frame counts rad deg)")
            out[sym] = out.get(sym, 0) + exp * sign
    return _clean(out)


def _exp_text(e: int, *, pretty: bool) -> str:
    if e == 1:
        return ""
    if pretty and e in (2, 3):
        return "²" if e == 2 else "³"
    return str(e) if not pretty else f"^{e}"


def format_unit(unit: Union[str, Mapping[str, int], None], *, pretty: bool = False) -> str:
    """The canonical spelling of a unit — ASCII (``um2``, ``um/s``, ``1/s``, ``vox``, ``""``)
    or pretty for the screen (``µm²``, ``µm/s``). One spelling per unit, so two results
    with the same unit always print the same."""
    u = parse_unit(unit)
    if not u:
        return ""
    if u == {"px": 2, "zpx": 1}:
        return "vox"
    num = [(k, e) for k, e in u.items() if e > 0]
    den = [(k, -e) for k, e in u.items() if e < 0]
    key = lambda kv: _ORDER.index(kv[0]) if kv[0] in _ORDER else len(_ORDER)
    num.sort(key=key)
    den.sort(key=key)
    name = lambda k: (_PRETTY.get(k, k) if pretty else k)
    sep = "·" if pretty else "*"
    top = sep.join(f"{name(k)}{_exp_text(e, pretty=pretty)}" for k, e in num) or "1"
    if not den:
        return top
    bottom = sep.join(f"{name(k)}{_exp_text(e, pretty=pretty)}" for k, e in den)
    return f"{top}/{bottom}"


def unit_slug(unit: Union[str, Mapping[str, int], None]) -> str:
    """The unit as a column-name suffix: ``um2``, ``vox``, ``um_per_s``, ``per_s``,
    ``""`` for dimensionless — the convention :func:`unit_of_column` reads back."""
    s = format_unit(unit)
    if not s:
        return ""
    s = s.replace("*", "_")
    if "/" in s:
        top, bottom = s.split("/", 1)
        return (f"{top}_per_{bottom}" if top != "1" else f"per_{bottom}").replace("-", "m")
    return s


def describe_unit(unit: Union[str, Mapping[str, int], None]) -> str:
    """``µm²`` / ``counts`` / ``dimensionless`` / ``unknown`` — for a message or a hover."""
    if unit is None:
        return "unknown"
    s = format_unit(unit, pretty=True)
    return s or "dimensionless"


# ── composition ───────────────────────────────────────────────────────────────

def mul_units(a: Any, b: Any) -> Unit:
    ua, ub = parse_unit(a), parse_unit(b)
    out = dict(ua)
    for k, e in ub.items():
        out[k] = out.get(k, 0) + e
    return _clean(out)


def div_units(a: Any, b: Any) -> Unit:
    ua, ub = parse_unit(a), parse_unit(b)
    out = dict(ua)
    for k, e in ub.items():
        out[k] = out.get(k, 0) - e
    return _clean(out)


def pow_unit(unit: Any, k: float) -> Unit:
    """``unit**k``. An exponent that would leave a fractional power (``um`` to the 1.5) is
    refused; a dimensionless unit takes any exponent."""
    u = parse_unit(unit)
    if not u:
        return {}
    out: Dict[str, int] = {}
    for sym, e in u.items():
        v = e * float(k)
        if abs(v - round(v)) > 1e-9:
            raise UnitError(
                f"{describe_unit(u)} to the power {k:g} is not a unit (it would need a "
                f"fractional exponent on {sym})")
        out[sym] = int(round(v))
    return _clean(out)


def root_unit(unit: Any, n: int = 2) -> Unit:
    """The ``n``-th root: ``um2`` → ``um``; ``um`` → refused (no µm^½ exists)."""
    return pow_unit(unit, 1.0 / float(n))


def _reduce(u: Mapping[str, int]) -> Tuple[Unit, float]:
    """Scaled symbols → canonical bases, with the factor one of ``u`` equals in them."""
    out: Dict[str, int] = {}
    factor = 1.0
    for sym, e in u.items():
        if sym in _SCALED:
            base, k = _SCALED[sym]
            out[base] = out.get(base, 0) + e
            factor *= k ** e
        else:
            out[sym] = out.get(sym, 0) + e
    return _clean(out), factor


def same_dimension(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` differ at most by a fixed factor (``nm`` vs ``um``)."""
    return _reduce(parse_unit(a))[0] == _reduce(parse_unit(b))[0]


_PHYSICAL = {"px": ("um", "pixel_size_um"), "zpx": ("um", "z_step_um"),
             "frame": ("s", "dt_s")}


def conversion_factor(src: Any, dst: Any, *, pixel_size_um: Optional[float] = None,
                      z_step_um: Optional[float] = None, dt_s: Optional[float] = None
                      ) -> float:
    """The number a value in ``src`` is multiplied by to be in ``dst``.

    ``1.0`` for the same unit; a fixed factor between scaled spellings (``nm``→``um`` is
    ``1e-3``); the CALIBRATION between a pixel and a micron (``px``→``um`` is
    ``pixel_size_um``, ``zpx``→``um`` is ``z_step_um``, ``frame``→``s`` is ``dt_s``, each
    raised to the symbol's exponent). Refused with the missing key named when the
    conversion needs a calibration that was not given, and refused as incompatible when no
    calibration could reconcile the two (``um2`` against ``um``)."""
    su, du = parse_unit(src), parse_unit(dst)
    if su == du:
        return 1.0
    (sc, sf), (dc, df) = _reduce(su), _reduce(du)
    if sc == dc:
        return sf / df
    calib = {"pixel_size_um": pixel_size_um, "z_step_um": z_step_um, "dt_s": dt_s}
    missing = []

    def physical(u: Mapping[str, int], f: float) -> Tuple[Unit, float]:
        out: Dict[str, int] = {}
        for sym, e in u.items():
            if sym in _PHYSICAL:
                base, key = _PHYSICAL[sym]
                val = calib.get(key)
                if val is None or not float(val) > 0:
                    missing.append(key)
                    out[sym] = out.get(sym, 0) + e
                    continue
                out[base] = out.get(base, 0) + e
                f *= float(val) ** e
            else:
                out[sym] = out.get(sym, 0) + e
        return _clean(out), f

    (sp, spf), (dp, dpf) = physical(sc, sf), physical(dc, df)
    if sp == dp:
        return spf / dpf
    if missing:
        need = sorted(set(missing))
        raise UnitError(
            f"converting {describe_unit(su)} to {describe_unit(du)} needs the calibration "
            f"{', '.join(need)}, which this data does not carry — load a file with it, or "
            f"state it on the Load card")
    raise UnitError(
        f"cannot convert {describe_unit(su)} to {describe_unit(du)}: they measure different "
        f"things")


def physical_unit(unit: Any) -> Unit:
    """The unit with every pixel and frame made physical: ``px``/``zpx`` → ``um``,
    ``frame`` → ``s``; scaled spellings → canonical. ``vox`` → ``um3``, ``px/frame`` →
    ``um/s``. The TARGET of a "to microns" conversion."""
    u, _f = _reduce(parse_unit(unit))
    out: Dict[str, int] = {}
    for sym, e in u.items():
        base = _PHYSICAL.get(sym, (sym, ""))[0]
        out[base] = out.get(base, 0) + e
    return _clean(out)


def pixel_unit(unit: Any) -> Unit:
    """The inverse of :func:`physical_unit` for LATERAL lengths and time: ``um`` → ``px``,
    ``s`` → ``frame`` (an axial length cannot be told from a lateral one once it is in
    microns, so ``um`` always goes to ``px``)."""
    u, _f = _reduce(parse_unit(unit))
    out: Dict[str, int] = {}
    back = {"um": "px", "s": "frame"}
    for sym, e in u.items():
        base = back.get(sym, sym)
        out[base] = out.get(base, 0) + e
    return _clean(out)


# ── the catalog's naming convention ───────────────────────────────────────────

#: Column names whose unit the catalog fixes by name. ``area`` is handled by z_kind.
_COLUMN_EXACT: Dict[str, str] = {
    "id": "", "m": "", "c": "", "b": "", "track_id": "", "member_id": "", "parent_id": "",
    "count": "", "n": "", "n_above": "", "n_sub": "", "frac_above": "", "cut": "",
    "eccentricity": "", "solidity": "", "extent": "", "orientation": "rad",
    "t": "frame", "x": "px", "y": "px", "z": "zpx",
    "track_length": "frame",
    "perimeter": "um", "axis_major": "um", "axis_minor": "um",
    "speed": "um/s", "vy": "um/s", "vx": "um/s",
    "neighbor_dist_mean": "um", "neighbor_dist_std": "um",
    "local_divergence": "1/s", "local_curl": "1/s",
    "level": "counts", "intensity": "counts",
}
#: Suffix → unit, tried longest first.
_COLUMN_SUFFIX: Tuple[Tuple[str, str], ...] = (
    ("_um_per_s", "um/s"), ("_um_s", "um/s"), ("_px_per_frame", "px/frame"),
    ("_um2", "um2"), ("_um3", "um3"), ("_px2", "px2"), ("_px3", "px3"),
    ("_per_s", "1/s"), ("_intensity", "counts"), ("_counts", "counts"),
    ("_frames", "frame"), ("_frame", "frame"),
    ("_um", "um"), ("_nm", "nm"), ("_mm", "mm"), ("_px", "px"), ("_vox", "vox"),
    ("_ms", "ms"), ("_s", "s"), ("_deg", "deg"), ("_rad", "rad"), ("_ratio", ""),
)


def unit_of_column(name: str, *, z_kind: Optional[str] = None) -> Optional[str]:
    """The unit the catalog's NAMING convention gives a column, as an ASCII unit string —
    ``""`` for a dimensionless one, ``None`` when the name says nothing.

    ``area``/``area_convex`` are a voxel COUNT: ``px2`` on a 2D table (``z_kind``
    ``plane_index``) and ``vox`` on a 3D one (``subpixel``) — the LABEL_INVARIANT rule.
    Centroids ``x``/``y`` are ``px`` and ``z`` a plane step; ``t`` a frame index; ids and
    fractions dimensionless; Measure's regionprops lengths (``perimeter``, ``axis_*``) are
    µm because it hands skimage the physical spacing; Object Metrics' velocities are µm/s.
    Suffixes carry the rest: ``_um``, ``_um2``, ``_px``, ``_s``, ``_um_per_s``,
    ``_intensity``. Prefix ``n_`` is a count."""
    n = str(name or "")
    if not n:
        return None
    if n in ("area", "area_convex", "volume"):
        return "vox" if z_kind == "subpixel" else "px2"
    if n in _COLUMN_EXACT:
        return _COLUMN_EXACT[n]
    low = n.lower()
    for suffix, unit in _COLUMN_SUFFIX:
        if low.endswith(suffix) and len(low) > len(suffix):
            return unit
    if low.startswith("n_") or low.startswith("frac_") or low.startswith("ratio_"):
        return ""
    return None


def unit_key(domain: Domain, layer: Optional[str], name: str) -> str:
    """The :data:`UNITS_KEY` entry for one scalar, column or layer."""
    return f"{domain.value}:{layer or ''}:{name}"


def recorded_unit(metadata: Optional[Mapping[str, Any]], domain: Domain, name: str,
                  layer: Optional[str] = None) -> Optional[str]:
    """The unit a producer RECORDED for ``(domain, layer, name)``, or ``None``."""
    table = (metadata or {}).get(UNITS_KEY)
    if not isinstance(table, Mapping):
        return None
    val = table.get(unit_key(domain, layer, name))
    return None if val is None else str(val)


def unit_of(ds: Dataset, domain: Domain, name: str, layer: Optional[str] = None
            ) -> Optional[str]:
    """What unit a scalar/column/layer on ``ds`` is in: the recorded one, else the naming
    convention (with the table's own ``z_kind`` deciding ``area``), else ``None`` =
    unknown. A Voxel/lattice layer with no record is unknown: a mask is dimensionless but a
    distance field is µm, and the name alone cannot tell."""
    rec = recorded_unit(ds.metadata, domain, name, layer)
    if rec is not None:
        return rec
    if domain in (Domain.LABEL, Domain.POINT, Domain.TRACK):
        zk = ds.structure_zkind(domain, layer) if layer else None
        return unit_of_column(name, z_kind=zk)
    return unit_of_column(name)


def with_unit(ds: Dataset, domain: Domain, name: str, unit: Any,
              layer: Optional[str] = None) -> Dataset:
    """Record the unit of ``(domain, layer, name)`` on ``ds`` — canonical spelling, ``""``
    for dimensionless; ``None`` removes the record (unknown). Copy-on-write."""
    table = dict(ds.metadata.get(UNITS_KEY) or {})
    key = unit_key(domain, layer, name)
    if unit is None:
        table.pop(key, None)
    else:
        table[key] = format_unit(unit)
    return ds.with_metadata(**{UNITS_KEY: table if table else None})


# ── the closed vocabulary a constant's unit socket offers ─────────────────────

#: ``none`` is the dimensionless entry (a blank dropdown row would read as unset);
#: :func:`parse_unit` reads it as ``{}``.
UNIT_CHOICES: Tuple[str, ...] = (
    "none", "px", "px2", "vox", "um", "um2", "um3", "nm", "s", "ms", "min", "frame",
    "um/s", "px/frame", "1/s", "counts", "rad", "deg")

UNIT_CHOICE_DOCS: Dict[str, str] = {
    "none": "Dimensionless — a plain number: a ratio, a count, an exponent, a scale factor. "
            "Multiplying by it leaves the other operand's unit unchanged.",
    "px": "A lateral pixel of THIS image. Converts to µm by the Dataset's pixel size; a "
          "constant in px scales with the camera, not with the specimen.",
    "px2": "A pixel area (px²) — the unit of `area` on a 2D label table, which counts "
           "pixels. Converts to µm² by the pixel size squared.",
    "vox": "A voxel (px²·zpx) — the unit of `area` on a 3D label table, which counts "
           "voxels. Converts to µm³ by pixel size² × Z step.",
    "um": "A micron — a physical lateral or axial length, independent of the camera. The "
          "unit Measure's `perimeter`, `axis_major` and every `_um` column are in.",
    "um2": "A square micron — a physical area. What `area` becomes after `to physical`, "
           "and the unit to give a constant you subtract from a µm² column.",
    "um3": "A cubic micron — a physical volume, what a 3D `area` (voxels) becomes after "
           "`to physical` using pixel size² × Z step.",
    "nm": "A nanometre (10⁻³ µm) — for a constant stated in nm, an emission wavelength or "
          "a bead size; converts to µm by a fixed factor.",
    "s": "A second — physical time, independent of the frame rate. The unit `dt_s` and "
         "every `_s` column are in.",
    "ms": "A millisecond (10⁻³ s) — for a constant stated in ms, an exposure or a short "
          "interval; converts to seconds by a fixed factor.",
    "min": "A minute (60 s) — for a constant stated in minutes, such as a treatment time; "
           "converts to seconds by a fixed factor.",
    "frame": "A timepoint index of THIS series. Converts to seconds by the Dataset's frame "
             "interval; a constant in frames scales with the acquisition rate.",
    "um/s": "A velocity in microns per second — the unit Object Metrics' `speed`, `vx`, "
            "`vy` and Track Field's velocities are in.",
    "px/frame": "A velocity in pixels per frame — what a displacement between consecutive "
                "frames is in before any calibration; converts to µm/s by pixel size over "
                "frame interval.",
    "1/s": "A rate per second — the unit of a divergence or curl, or of a count divided "
           "by a time in seconds.",
    "counts": "Camera intensity (ADU) — the unit of every `*_intensity` column and of the "
              "image's pixel values; a threshold level is in counts.",
    "rad": "An angle in radians — the unit of `orientation`; converts to degrees by a fixed "
           "factor.",
    "deg": "An angle in degrees — for a constant stated in degrees; converts to radians by "
           "a fixed factor.",
}
