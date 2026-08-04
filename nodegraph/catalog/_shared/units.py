"""units — shared catalog helpers."""

from __future__ import annotations


from typing import Optional

def to_pixels_v2(value: float, unit: str, *, pixel_size_um: Optional[float] = None,
                 z_step_um: Optional[float] = None, dt_s: Optional[float] = None) -> float:
    """Convert a physical ``value`` to pixels/frames. Extends the V1.91 vocabulary with
    ``um_axial`` (÷ ``z_step_um``) for anisotropic 3D kernels (V2.03 §2 A5). A missing
    calibration degrades to a 1:1 factor (the caller decides whether that is acceptable)."""
    u = (unit or "").lower()
    if u in ("", "px"):
        return value
    if u == "um":
        return value / (pixel_size_um or 1.0)
    if u == "um_axial":
        return value / (z_step_um or 1.0)
    if u == "nm":
        return (value / 1000.0) / (pixel_size_um or 1.0)
    if u == "s":
        return value / (dt_s or 1.0)
    return value
