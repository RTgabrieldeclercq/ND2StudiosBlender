"""full scale — shared catalog helpers."""

from __future__ import annotations


from typing import Optional

from nodegraph.engine import EvalContext

def _declared_full_scale(ctx: EvalContext) -> Optional[float]:
    """The image's **declared full-scale intensity** — ``2**bit_depth - 1`` — or ``None``
    when no integer scale is declared (V2.13).

    The scale-free stand-in for ``np.iinfo(volume.dtype).max``, which Cell-Tracker's γ,
    DoG rescale and morphological-gradient blend all divide by. This engine cannot: every
    compute works in float, and the *significant* sensor depth (12 on most ND2s, §7c) is
    not the container width anyway — a 12-bit frame in a uint16 array would be scaled by
    65535 and come out four bits too dark. ``bit_depth`` is the calibration key that
    carries it, so this read is memo-fenced like any other.

    ``None`` is returned — rather than a guess — when the key is absent, which is the
    honest state after a percentile :func:`enhance.normalize` (whose ``value_rescaled``
    transform drops it) or for a TIFF with no depth tag. Each caller decides how to
    degrade; the catalog's answer is "fall back to the unit's own maximum", which is a
    plane/volume-global statistic and is why those nodes declare WHOLE_PLANE.

    Call it **eagerly** in the compute (never from a lazy closure — the ReadContext
    freezes when the compute returns, V2.04 §6b)."""
    bd = ctx.calib("bit_depth")
    try:
        bits = int(bd) if bd else 0
    except (TypeError, ValueError):          # junk in the envelope must never raise here
        bits = 0
    return float(2 ** bits - 1) if bits > 0 else None
