"""background_probability — p(background) per voxel, from intensity, as a PROBABILITY.

A threshold answers "is this voxel background?" with a yes or a no. That is the wrong shape
of answer wherever the next step needs to weigh the evidence rather than inherit a decision:
placing an object between two others, deciding whether a divide is real, or asking whether a
detected thing sits in the background or on a body. This kernel returns a number in [0, 1]
instead, and the number is meant to be believed — on the acquisition it was calibrated
against it reaches an expected calibration error of 0.12 with an AUC of 0.97.

## The model, and why it is this small

Intensity is first put on a scale that survives moving between fields, channels and depths::

    u = (I - lo) / (hi - lo)          lo, hi = robust low/high percentiles of the SAME unit

then run through a decreasing logistic::

    p(background | u) = ceiling / (1 + exp((u - midpoint) / width))

Three numbers, and every one of them is interpretable: ``midpoint`` is where a voxel is
equally likely to be background or material, ``width`` is how quickly that flips, and
``ceiling`` is the most this evidence can ever claim — because a deep-background voxel is
not certainly background if the object of interest is merely dim.

The obvious alternative was tried first and rejected on measurement rather than on taste: a
two-component mixture with a Gaussian background peak, whose posterior is read off the
histogram. It discriminates (AUC 0.907) and is badly MIS-CALIBRATED (ECE 0.435, stating 0.29
where the truth was ~1), because a real background is not one Gaussian — it carries haze,
out-of-focus signal and a fluorescence pedestal, so a peak-height estimate of the mixing
weight comes out far too low and the posterior inherits the error. The logistic makes no
claim about the background's shape at all, and matched a non-parametric isotonic fit of the
same data (AUC 0.971/ECE 0.121 against 0.964/0.116) with three parameters instead of 36.

## Fitting it

:func:`fit_logistic` recovers the three parameters from labelled voxels by maximum
likelihood on a coarse grid. It is deliberately a grid rather than a gradient method: the
surface is smooth and three-dimensional, the parameters have hard interpretable bounds, and
a grid cannot land in a local optimum or fail to converge silently.

## Anchors

``lo``/``hi`` are percentiles of the unit being normalised, so attenuation with depth and
brightness differences between fields drop out. They must be computed over the SAME extent
the caller intends the probability to be comparable within — per plane if depth attenuates,
per volume if it does not. This kernel takes the array it is given and normalises it whole;
choosing that extent is the caller's decision, not this function's.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["normalise", "background_probability", "fit_logistic",
           "DEFAULT_MIDPOINT", "DEFAULT_WIDTH", "DEFAULT_CEILING",
           "DEFAULT_LO_PCT", "DEFAULT_HI_PCT"]

#: Defaults MEASURED, not chosen. Fitted by leave-one-field-out on 111k hand-outlined
#: voxels of a packed-particle confocal acquisition (three fields, two stains); each split
#: independently chose midpoint 0.31-0.35 and width 0.07-0.12. They are a starting point for
#: a similar bright-object-on-dark-background acquisition and NOT a universal constant —
#: :func:`fit_logistic` is how a different instrument gets its own.
DEFAULT_MIDPOINT = 0.33
DEFAULT_WIDTH = 0.09
DEFAULT_CEILING = 0.90
DEFAULT_LO_PCT = 5.0
DEFAULT_HI_PCT = 99.0


def normalise(a: np.ndarray, lo_pct: float = DEFAULT_LO_PCT,
              hi_pct: float = DEFAULT_HI_PCT) -> np.ndarray:
    """Intensity onto a robust 0-1 scale: ``u = (a - p_lo) / (p_hi - p_lo)``.

    Percentiles rather than min/max so one hot pixel or one dead one cannot set the scale,
    and ``hi_pct`` deliberately below 100 so that saturation does not compress everything
    else into the bottom of the range. The result is NOT clipped: a voxel brighter than the
    high anchor is legitimately >1 and the logistic maps it to ~0 background, which is
    correct.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a.astype(np.float32)
    lo, hi = np.percentile(a, [float(lo_pct), float(hi_pct)])
    return ((a - lo) / max(float(hi - lo), 1e-9)).astype(np.float32)


def background_probability(a: np.ndarray, *, midpoint: float = DEFAULT_MIDPOINT,
                           width: float = DEFAULT_WIDTH,
                           ceiling: float = DEFAULT_CEILING,
                           lo_pct: float = DEFAULT_LO_PCT,
                           hi_pct: float = DEFAULT_HI_PCT,
                           normalised: bool = False) -> np.ndarray:
    """p(background) per element of ``a``, in [0, ``ceiling``].

    Set ``normalised=True`` when ``a`` has already been through :func:`normalise` — which is
    what the fitter does, so a caller can fit and apply on exactly the same scale.
    """
    u = np.asarray(a, dtype=np.float32) if normalised else normalise(a, lo_pct, hi_pct)
    w = max(float(width), 1e-6)
    # clip the exponent, not the result: exp overflows to inf for a bright voxel and numpy
    # warns on it even though 1/(1+inf) is the right answer.
    z = np.clip((u - float(midpoint)) / w, -60.0, 60.0)
    return (float(ceiling) / (1.0 + np.exp(z))).astype(np.float32)


def fit_logistic(u_background: Sequence[float], u_material: Sequence[float], *,
                 midpoints: Optional[Sequence[float]] = None,
                 widths: Optional[Sequence[float]] = None,
                 ceilings: Optional[Sequence[float]] = None,
                 max_samples: int = 300_000,
                 seed: int = 0) -> Dict[str, float]:
    """Maximum-likelihood (midpoint, width, ceiling) from labelled NORMALISED intensities.

    ``u_background`` and ``u_material`` are :func:`normalise` outputs for voxels known to be
    background and known to be material. Returns the three parameters plus the achieved mean
    negative log-likelihood, so a caller can compare fits.

    Labels near an object's edge should be EXCLUDED by the caller rather than assigned to
    one side: within a voxel or two of a boundary the honest answer is intermediate, and
    scoring it as either class trains the model to be over-confident exactly where it should
    not be.
    """
    b = np.asarray(u_background, dtype=np.float64).ravel()
    m = np.asarray(u_material, dtype=np.float64).ravel()
    if b.size < 2 or m.size < 2:
        raise ValueError("fit_logistic needs at least two labelled samples of each class "
                         f"(got {b.size} background, {m.size} material)")
    u = np.r_[b, m]
    y = np.r_[np.ones(b.size), np.zeros(m.size)]
    if u.size > max_samples:
        idx = np.random.default_rng(seed).choice(u.size, max_samples, replace=False)
        u, y = u[idx], y[idx]
    mids = np.arange(0.05, 0.80, 0.01) if midpoints is None else np.asarray(midpoints, float)
    wids = np.arange(0.01, 0.25, 0.01) if widths is None else np.asarray(widths, float)
    ceils = (np.array([0.80, 0.85, 0.90, 0.95, 1.0]) if ceilings is None
             else np.asarray(ceilings, float))
    best = (np.inf, DEFAULT_MIDPOINT, DEFAULT_WIDTH, DEFAULT_CEILING)
    for mid in mids:
        for wid in wids:
            z = np.clip((u - mid) / max(float(wid), 1e-6), -60.0, 60.0)
            base = 1.0 / (1.0 + np.exp(z))
            for ce in ceils:
                p = np.clip(ce * base, 1e-6, 1.0 - 1e-6)
                nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
                if nll < best[0]:
                    best = (nll, float(mid), float(wid), float(ce))
    return {"midpoint": best[1], "width": best[2], "ceiling": best[3], "nll": best[0]}


def expected_calibration_error(p: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> Tuple[float, list]:
    """(ECE, reliability rows) — how far a stated probability is from the observed rate.

    The number that matters for this kernel. Discrimination (AUC) says whether the ordering
    is right; calibration says whether 0.3 means 0.3, which is what a downstream that
    MULTIPLIES probabilities together depends on.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tot, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if sel.sum() < 2:
            continue
        tot += sel.mean() * abs(p[sel].mean() - y[sel].mean())
        rows.append((float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())))
    return float(tot), rows
