"""Value steps — how far one tick of a numeric editor moves, Qt-free.

Shared by the node card's scrub, the inspector's spin boxes and the demo window's sliders
(:mod:`nodelab_v2.demo_window`), and importable without Qt so the headless recipe module
(:mod:`nodelab_v2.demo_recipes`) and the selftest can size a slider the way the inspector
sizes a spin box. Moved out of :mod:`nodelab_v2.node_item` on 2026-10-07 for that reason;
``node_item`` re-exports every name so nothing that imported from it changed.
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["value_step", "value_decimals", "float_editor_precision", "NUM_MAX_FLOAT",
           "NUM_MAX_INT"]


#: Per-step increment for a scrub, chosen from the socket's type and unit. A physical length
#: in µm and a normalized 0–1 threshold want very different granularity, and the alternative
#: — one step for everything — makes one of them unusable. Shift divides by 10, Ctrl
#: multiplies by 10, so the table only has to be right about the middle case.
_SCRUB_STEP = {
    "px": 1.0, "um": 0.01, "um_axial": 0.01, "um2": 1.0, "um3": 1.0,
    "nm": 1.0, "s": 0.01,
}
#: Step for a unitless FLOAT — thresholds, weights, fractions, almost all 0–1.
_SCRUB_STEP_UNITLESS = 0.005

#: Decimals a FLOAT editor shows unless its magnitude demands more, and the widest step it
#: will ever take. Floors, not fixed values — see :func:`value_step`.
_FLOAT_DECIMALS = 3
_FLOAT_STEP = 0.05

#: Editor bounds. Deliberately far past anything the catalog means, because the WIDGET must
#: never be the thing that limits a value — the compute validates, and a silent clamp is the
#: worst way to be told no. The old caps (1e6 float / 1e5 int) were reachable in practice:
#: an `iterations` of 200000 and a `max_area` on a millimetre-scale colony both hit them and
#: were quietly truncated to the ceiling.
NUM_MAX_FLOAT = 1e12
NUM_MAX_INT = 2_000_000_000             # just under Qt's INT_MAX for QSpinBox


def _magnitude(v) -> Optional[float]:
    """``abs(float(v))`` when that is a usable positive magnitude, else ``None``."""
    try:
        f = abs(float(v))
    except (TypeError, ValueError):
        return None
    return f if f > 0.0 else None


def value_step(spec, current=None, *, integer: bool = False) -> float:
    """One increment for a numeric socket, from its UNIT **and** its MAGNITUDE.

    The unit table stays authoritative — it encodes the granularity the quantity is measured
    in (0.01 µm, 1 px, 0.01 s), and that is a property of the physics, not of the current
    number. Magnitude only **bounds** it, which is what fixes the two symmetric pathologies
    a fixed step has at the extremes:

    * **too coarse for a small value** — a 0.005 step on a 5e-5 learning rate overshoots it
      100× on the first pixel of drag, and the inspector's 0.05 could not express it at all
      (it rounded to ``0.000``, so the widget displayed zero for a non-zero default). Already
      true before this node of ``dic_correlate``'s ``strain_smoothness`` (1e-5) and
      ``disp_smoothness`` (5e-4) and both solvers' ``mu``;
    * **too fine for a large one** — 0.05 on a 2047.5 threshold, or 1 on a 33 000-iteration
      training budget, is tens of thousands of clicks to cross, so the control is decorative.

    The bound is one order of magnitude either side of the value: never coarser than the
    value itself, never finer than a tenth of its leading order. So ``sigma`` at 0.5 µm keeps
    its 0.01 table step exactly, while ``iterations`` at 2000 steps by 100 and a 5e-5 rate
    steps by 1e-5.

    Presentation-only: nothing here reaches ``node_recipe_hash``, so retuning an editor
    cannot invalidate a memo entry or a saved graph.
    """
    base = 1.0 if integer else _SCRUB_STEP.get(getattr(spec, "unit", "") or "",
                                               _SCRUB_STEP_UNITLESS)
    mag = _magnitude(current)
    if mag is None:
        mag = _magnitude(getattr(spec, "default", None))
    if mag is None:
        return base
    import math
    order = 10.0 ** math.floor(math.log10(mag))
    step = min(max(base, order / 10.0), order)
    return max(1.0, float(int(step))) if integer else step


def value_decimals(spec, current=None) -> int:
    """Decimals a FLOAT editor needs: enough for the FINEST value in play, floored at 3.

    Driven by the smallest magnitude of (current, default) rather than by the current value
    alone, so a socket whose default is 5e-5 stays typable after the user has entered 0.5 and
    wants to go back. Floored at the historic 3 rather than reduced for large values: showing
    ``2047.500`` is only cosmetic noise, whereas dropping below 3 decimals would make ``0.001``
    unreachable on a socket that had merely been left at a big number.
    """
    mags = [m for m in (_magnitude(current),
                        _magnitude(getattr(spec, "default", None))) if m is not None]
    if not mags:
        return _FLOAT_DECIMALS
    import math
    step = value_step(spec, min(mags))
    return max(_FLOAT_DECIMALS, min(9, int(math.ceil(-math.log10(step)))))


def float_editor_precision(spec, current=None) -> Tuple[int, float]:
    """``(decimals, step)`` for a FLOAT spin box — :func:`value_decimals` +
    :func:`value_step`. Decimals follow the finest value in play; the step follows where the
    value is NOW, so stepping feels right without anything becoming untypable."""
    return value_decimals(spec, current), value_step(spec, current)
