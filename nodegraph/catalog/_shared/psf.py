"""psf — shared catalog helpers: an optics-metadata-derived point spread function.

**Why this is `_shared/` and not part of ``enhance.deconvolve``.** It used to live in that
node's module, and ``scripts/_split_nodes.py`` recorded the reason: a PSF helper "belongs WITH
the node it documents". That held while exactly one node derived a PSF. ``enhance.zs_deconvnet``
is a second one — its self-supervised deconvolution loss needs the same PSF, from the same
optics, or the two deconvolution nodes in this catalog would model the same microscope
differently — and catalog rule 5 forbids one node module from importing another (importing
executes it, which registers that node early, scrambles catalog order, and welds the two
nodes' live-reload fingerprints together). So the shared half moved here, which is the remedy
that rule prescribes.

Still re-exported by :mod:`nodegraph.nodes` and :mod:`nodegraph` under their original names —
both are public API and neither the names nor the maths changed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def diffraction_sigmas(emission_nm: Optional[float], na: Optional[float],
                       pixel_size_um: Optional[float], z_step_um: Optional[float],
                       is_3d: bool) -> Tuple[float, ...]:
    """Gaussian-approximation PSF sigmas **derived from optics metadata**: lateral
    ``σ_xy ≈ 0.21·λ/NA`` and axial ``σ_z ≈ 0.66·λ·n/NA²`` (n≈1.5 immersion), converted
    to pixels via ``pixel_size_um`` / ``z_step_um``. Returns ``(σ_y,σ_x)`` in 2D or
    ``(σ_z,σ_y,σ_x)`` in 3D. (A Gaussian PSF is the portable default; a Gibson–Lanni /
    measured PSF is a backend swap behind the same derived sampling.)

    Checked against a real measured PSF: ``scripts/_bench_zsdeconvnet.py --zenodo`` fits the
    authors' simulated-optics PSF (525 nm, NA 1.3, 0.0313 µm/px) and compares. Measured
    ``σ_xy`` 2.739 px against 2.710 px derived here — a ratio of 1.01, which is what justifies
    shipping "no PSF file needed" as the default for both deconvolution nodes."""
    lam_um = (emission_nm or 520.0) / 1000.0
    na = na or 1.4
    sxy_um = 0.21 * lam_um / na
    sxy = sxy_um / (pixel_size_um or 0.1)
    if not is_3d:
        return (sxy, sxy)
    sz_um = 0.66 * lam_um * 1.5 / (na * na)
    sz = sz_um / (z_step_um or 0.5)
    return (sz, sxy, sxy)


def gaussian_psf(sigmas: Sequence[float], *, radius_factor: float = 3.0) -> np.ndarray:
    """A normalized n-D Gaussian kernel with the given per-axis ``sigmas`` (pixels)."""
    sig = tuple(max(0.5, float(s)) for s in sigmas)
    radii = [max(1, int(round(radius_factor * s))) for s in sig]
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    g = np.ones_like(grids[0], dtype=float)
    for coord, s in zip(grids, sig):
        g = g * np.exp(-(coord.astype(float) ** 2) / (2.0 * s * s))
    total = g.sum()
    return g / total if total else g


__all__ = ["diffraction_sigmas", "gaussian_psf"]
