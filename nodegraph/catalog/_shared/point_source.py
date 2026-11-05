"""Crank's continuous point source — the diffusion model of a secreting spheroid (2026-10-16).

The model of McGhee et al., *In vitro models* 1:309-321 (2022), Eq. 2, after Crank, *The
Mathematics of Diffusion*: a source releasing ``q`` (fg/s) into an unbounded medium from
time 0 gives, at distance ``r`` (µm) and time ``t`` (s),

    C(r, t) = q / (4·π·D·r) · erfc( r / (2·√(D·t)) )

with ``D`` the analyte's diffusivity (µm²/s; the paper measured 20 ± 10 for IL-8 with its
capture/detection pair). ``C`` comes out in fg/µm³ = 10⁹ pg/mL. It is the time integral of
Crank's INSTANTANEOUS point source ``M / (8·(π·D·t)^{3/2}) · exp(−r² / (4·D·t))``, which is
what makes a time-varying rate exact by superposition: a source that releases ``q_j`` during
``[b_j, b_{j+1}]`` adds ``q_j·[K(r, t − b_j) − K(r, t − b_{j+1})]``, ``K`` the unit-rate
response above (0 for a source not yet on). On an impermeable floor with the source on it
(``half_space``) every concentration doubles (the method of images), so the same data give
half the rate.

Two rate models, both linear in the rates:

* ``constant``: one rate per group (timepoint), each the source's AVERAGE since seeding — the
  paper's Eq. 2 fitted to each timepoint on its own;
* ``piecewise``: one rate per interval between successive timepoints, all timepoints fitted
  together — the rate's history, so a timepoint's profile carries everything released before
  it.

Fitted in data units (``absolute``, non-negative least squares) or on the logarithm
(``relative``, every point weighed by its relative miss, as the exponential trendline does);
``D`` fixed or chosen by profile over a log grid. :func:`monte_carlo` gives the rates' mean and
SD the paper's way (its Appendix B.3: refits on perturbed concentrations, distances, times and
D). Numpy + scipy (``erfc``, ``nnls``, ``least_squares``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

#: fg/µm³ in pg/mL.
FG_UM3_TO_PG_ML = 1e9
RATE_MODELS = ("piecewise", "constant")
BOUNDARIES = ("infinite", "half_space")
WEIGHTS = ("relative", "sqrt", "absolute")
BACKGROUNDS = ("none", "uniform")
#: The D profile's search range (µm²/s) and grid when D is fitted.
D_RANGE = (0.5, 500.0)
_D_GRID = 49
#: :func:`monte_carlo`'s seed (the paper's publication date), so a run is reproducible.
MC_SEED = 20221107


def unit_response(r: np.ndarray, tau: np.ndarray, D,
                  boundary: str = "infinite") -> np.ndarray:
    """pg/mL at ``r`` µm from a source of 1 fg/s switched on ``tau`` seconds ago: Crank's
    continuous point source ``erfc(r / 2√(Dτ)) / (4πDr)``, doubled on an impermeable floor.
    0 where ``tau <= 0`` (the source not yet on) or ``D <= 0``, NaN where ``r <= 0``. ``D``
    may be an array broadcasting with ``r`` and ``tau`` (one D per Monte Carlo run)."""
    from scipy.special import erfc
    r = np.asarray(r, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dv = np.asarray(D, dtype=float)
    shape = np.broadcast(r, tau, dv).shape
    rb, tb, db = (np.broadcast_to(v, shape) for v in (r, tau, dv))
    out = np.zeros(shape)
    on = (tb > 0) & (rb > 0) & (db > 0)
    rr, tt, dd = rb[on], tb[on], db[on]
    out[on] = erfc(rr / (2.0 * np.sqrt(dd * tt))) / (4.0 * math.pi * dd * rr)
    out[rb <= 0] = np.nan
    return out * FG_UM3_TO_PG_ML * (2.0 if boundary == "half_space" else 1.0)


def design(r: np.ndarray, t: np.ndarray, breaks: Sequence[float], D: float,
           boundary: str = "infinite") -> np.ndarray:
    """``(n, J)``: column ``j`` is each point's pg/mL per fg/s released during
    ``[breaks[j], breaks[j + 1]]`` — the superposition that makes a piecewise rate exact."""
    b = list(breaks)
    cols = [unit_response(r, t - b[j], D, boundary) - unit_response(r, t - b[j + 1], D, boundary)
            for j in range(len(b) - 1)]
    return np.stack(cols, axis=1) if cols else np.zeros((len(r), 0))


def _solve(A: np.ndarray, y: np.ndarray, weight: str) -> np.ndarray:
    """Non-negative coefficients for ``A·q ≈ y``: least squares in data units, on ``√``, or
    on ``ln``."""
    from scipy.optimize import least_squares, nnls
    q0, _res = nnls(A, y, maxiter=50 * max(1, A.shape[1]))
    if weight == "sqrt":
        ry = np.sqrt(y)
        try:
            sol = least_squares(lambda q: np.sqrt(np.maximum(A @ q, 0.0)) - ry,
                                np.maximum(q0, 1e-9), bounds=(0.0, np.inf), x_scale="jac",
                                max_nfev=5000)
            # 'trf' keeps its iterates strictly inside the bounds, so a rate pinned at 0 comes
            # back as ~1e-20 and a legend prints "q = 3.54e-20 fg/s"; a lower bound the
            # solver reports active IS the bound
            q = np.array(sol.x, dtype=float)
            q[np.asarray(sol.active_mask) == -1] = 0.0
            return q
        except (ValueError, RuntimeError):
            return q0
    if weight != "relative":
        return q0
    ly = np.log(y)
    floor = 1e-9
    start = np.log(np.maximum(q0, max(float(q0.max()) * 1e-3, floor) if q0.size else floor))

    def resid(th):
        m = A @ np.exp(th)
        return np.log(np.maximum(m, 1e-300)) - ly

    try:
        sol = least_squares(resid, start, bounds=(math.log(floor), math.log(1e6)),
                            x_scale="jac", max_nfev=2000)
        return np.exp(sol.x)
    except (ValueError, RuntimeError):
        return q0


def _cost(A: np.ndarray, y: np.ndarray, q: np.ndarray, weight: str) -> float:
    m = A @ q
    if weight == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.log(np.maximum(m, 1e-300)) - np.log(y)
    elif weight == "sqrt":
        e = np.sqrt(np.maximum(m, 0.0)) - np.sqrt(np.maximum(y, 0.0))
    else:
        e = m - y
    return float((e * e).sum())


def _r2(y: np.ndarray, m: np.ndarray, weight: str) -> float:
    if weight == "relative":
        y, m = np.log(y), np.log(np.maximum(m, 1e-300))
    elif weight == "sqrt":
        y, m = np.sqrt(np.maximum(y, 0.0)), np.sqrt(np.maximum(m, 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - m) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def _matrix(r, t, gi, times, rate, D, boundary, background=False, n_groups=0):
    if rate == "constant":
        A = np.zeros((len(r), len(times)))
        k = unit_response(r, t, D, boundary)
        A[np.arange(len(r)), gi] = k
    else:
        A = design(r, t, [0.0] + list(times), D, boundary)
    if background:
        # one uniform level per group: what has spread through a FINITE well — its walls send
        # back what an unbounded medium would carry away, so the far field rises everywhere
        B = np.zeros((len(r), n_groups))
        B[np.arange(len(r)), gi] = 1.0
        A = np.concatenate([A, B], axis=1)
    return A


def fit(r: np.ndarray, t: np.ndarray, y: np.ndarray, group: np.ndarray, labels: Sequence[str],
        *, rate: str = "piecewise", D: float = 20.0, fit_D: bool = False,
        boundary: str = "infinite", weight: str = "relative",
        background: str = "none") -> Dict[str, Any]:
    """Fit the point source to points at distance ``r`` (µm), time since the source started
    ``t`` (s) and concentration ``y`` (pg/mL); ``group`` indexes ``labels`` (one timepoint
    each, its time the median of its points'). Points must already be filtered (finite,
    ``r`` and ``t`` > 0, and ``y`` > 0 under ``relative``). Returns the JSON-shaped record
    :func:`predict` reads: ``D``, the groups (label, time, n, R², and under ``constant`` each
    group's rate), the breaks and rates under ``piecewise``, overall R² and n.
    ``background = "uniform"`` adds one non-negative level per group (``background`` in each
    group's record, pg/mL): a finite well's far field, which an unbounded Eq. 2 leaves at 0.
    ``weight = "sqrt"`` fits square roots: between ``relative`` and ``absolute``, each point
    weighing by its miss over the square root of its level (counting noise)."""
    r, t, y = (np.asarray(v, dtype=float) for v in (r, t, y))
    gi = np.asarray(group, dtype=int)
    labels = [str(v) for v in labels]
    times = [float(np.median(t[gi == k])) if (gi == k).any() else float("nan")
             for k in range(len(labels))]
    order = sorted(range(len(labels)), key=lambda k: times[k])
    if rate == "piecewise":
        # the intervals run between successive timepoints, in time order
        times_sorted = [times[k] for k in order]
    else:
        times_sorted = times

    bg = background == "uniform"

    def solve_at(d):
        A = _matrix(r, t, gi, times_sorted, rate, d, boundary, bg, len(labels))
        keep = np.isfinite(A).all(axis=1)
        q = _solve(A[keep], y[keep], weight)
        return q, _cost(A[keep], y[keep], q, weight), A

    if fit_D:
        grid = np.geomspace(D_RANGE[0], D_RANGE[1], _D_GRID)
        costs = [solve_at(float(d))[1] for d in grid]
        i = int(np.nanargmin(costs))
        lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
        from scipy.optimize import minimize_scalar
        sol = minimize_scalar(lambda ld: solve_at(float(math.exp(ld)))[1],
                              bounds=(math.log(lo), math.log(hi)), method="bounded",
                              options={"xatol": 1e-3})
        D = float(math.exp(sol.x)) if sol.success else float(grid[i])
    q, cost, A = solve_at(float(D))
    model = A @ q
    groups: List[Dict[str, Any]] = []
    for k, lab in enumerate(labels):
        m = gi == k
        g: Dict[str, Any] = {"label": lab, "t_s": times[k], "n": int(m.sum()),
                             "r2": _r2(y[m], model[m], weight) if m.sum() > 2 else None}
        if rate == "constant":
            g["q_fg_s"] = float(q[k])
        if bg:
            g["background"] = float(q[len(q) - len(labels) + k])
        groups.append(g)
    rec: Dict[str, Any] = {
        "model": "crank_point_source", "rate": rate, "D": float(D), "fit_D": bool(fit_D),
        "boundary": boundary, "weight": weight, "background": background,
        "groups": groups, "n": int(len(y)),
        "r2": _r2(y, model, weight), "cost": float(cost)}
    if rate == "piecewise":
        rec["breaks_s"] = [0.0] + [float(v) for v in times_sorted]
        rec["q_fg_s"] = [float(v) for v in q[:len(times_sorted)]]
    return rec


def _ls_runs(Y: np.ndarray, F: np.ndarray, level: bool):
    """Per run (row of ``Y``/``F``): the non-negative least-squares ``q`` of ``Y ≈ q·F``, or
    ``(q, b)`` of ``Y ≈ q·F + b`` — the exact two-variable NNLS, picked from the interior
    solution and the three faces of the orthant by their squared miss."""
    syf, sff = (Y * F).sum(axis=1), (F * F).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = np.where(sff > 0, np.maximum(syf / sff, 0.0), 0.0)
    if not level:
        return q1, np.zeros_like(q1)
    n = Y.shape[1]
    sy, sf, syy = Y.sum(axis=1), F.sum(axis=1), (Y * Y).sum(axis=1)

    def sse(q, b):
        return syy - 2 * q * syf - 2 * b * sy + q * q * sff + 2 * q * b * sf + n * b * b

    with np.errstate(divide="ignore", invalid="ignore"):
        det = sff * n - sf * sf
        qu, bu = (syf * n - sf * sy) / det, (sff * sy - sf * syf) / det
    b0 = np.maximum(sy / n, 0.0)
    cands = [(qu, bu, np.isfinite(qu) & np.isfinite(bu) & (qu >= 0) & (bu >= 0)),
             (np.zeros_like(q1), b0, np.ones(q1.shape, bool)),
             (q1, np.zeros_like(q1), np.ones(q1.shape, bool))]
    best_q, best_b = np.zeros_like(q1), np.zeros_like(q1)
    best = sse(best_q, best_b)
    for q, b, ok in cands:
        e = np.where(ok, sse(np.where(ok, q, 0.0), np.where(ok, b, 0.0)), np.inf)
        take = e < best
        best_q, best_b, best = (np.where(take, q, best_q), np.where(take, b, best_b),
                                np.where(take, e, best))
    return best_q, best_b


def monte_carlo(r: np.ndarray, t: np.ndarray, y: np.ndarray, y_sd: np.ndarray,
                group: np.ndarray, n_groups: int, *, D: float, boundary: str = "infinite",
                background: str = "none", n_runs: int = 100_000, D_sd: float = 10.0,
                t_sd: float = 600.0, r_sd: float = 50.0, seed: int = MC_SEED,
                labels: Optional[Sequence[str]] = None,
                batch: int = 4096) -> Dict[str, np.ndarray]:
    """The production rate's uncertainty as McGhee et al. 2022 (Appendix B.3) find it: for
    each group (timepoint) on its own, ``n_runs`` least-squares fits of Eq. 2 (one rate,
    absolute weighting), each on perturbed inputs — every row's concentration ``y + y_sd·ε``,
    distance ``r + r_sd·ε`` (the source margin's uncertainty), time ``t + t_sd·ε``, and the
    run's ``D`` drawn with mean ``D`` and SD ``D_sd``. The paper writes D's draw as normal;
    here it is LOGNORMAL with that mean and SD, because a normal 20 ± 10 µm²/s puts 2 % of
    runs at D ≤ 0 and some just above it, where Eq. 2 says nothing has reached the beads yet
    and a run needs a rate of 10¹⁰⁰ fg/s — one such run sets the mean (Run 4: 10¹²⁰).
    ``background = uniform`` refits the group's level beside its rate. Both are kept
    non-negative, as the nominal fit is.
    Returns per group the mean and SD of ``q`` (fg/s) and of the level (pg/mL), and the run
    count. Each group's random stream is seeded from ``seed`` and its LABEL (``labels``; else
    its index), so a timepoint gets the same result whichever other timepoints are fitted
    beside it, and a rerun repeats exactly."""
    import zlib
    n_runs = int(n_runs)
    level = background == "uniform"
    gi = np.asarray(group, dtype=int)
    out = {k: np.full(n_groups, np.nan) for k in ("q_mean", "q_sd", "b_mean", "b_sd")}
    for g in range(n_groups):
        ix = np.flatnonzero(gi == g)
        if not ix.size:
            continue
        key = str(labels[g]) if labels is not None and g < len(labels) else str(g)
        rng = np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])
        eps = rng.standard_normal(n_runs)
        if D_sd > 0 and D > 0:
            s2 = math.log(1.0 + (float(D_sd) / float(D)) ** 2)
            Ds = np.exp(math.log(float(D)) - 0.5 * s2 + math.sqrt(s2) * eps)
        else:
            Ds = np.full(n_runs, float(D))
        rg, tg, yg, sg = r[ix], t[ix], y[ix], y_sd[ix]
        # sums of each run's offset from the FIRST run's value: the SD of runs that all agree
        # comes out exactly 0, not the rounding residue of Σq² − (Σq)²/n
        acc = np.zeros(4)                      # Σ(q−q₀), Σ(q−q₀)², Σ(b−b₀), Σ(b−b₀)²
        ref = None
        for s in range(0, n_runs, batch):
            d = Ds[s:s + batch]
            nb = d.size
            R = np.maximum(rg + r_sd * rng.standard_normal((nb, ix.size)), 1.0)
            T = np.maximum(tg + t_sd * rng.standard_normal((nb, ix.size)), 1.0)
            Y = yg + sg * rng.standard_normal((nb, ix.size))
            F = unit_response(R, T, d[:, None], boundary)
            q, b = _ls_runs(Y, F, level)
            if ref is None:
                ref = (float(q[0]), float(b[0]))
            dq, db = q - ref[0], b - ref[1]
            acc += (dq.sum(), (dq * dq).sum(), db.sum(), (db * db).sum())
        mq, mb = acc[0] / n_runs, acc[2] / n_runs
        out["q_mean"][g], out["b_mean"][g] = ref[0] + mq, ref[1] + mb
        out["q_sd"][g] = math.sqrt(max(acc[1] / n_runs - mq * mq, 0.0))
        out["b_sd"][g] = math.sqrt(max(acc[3] / n_runs - mb * mb, 0.0))
    out["n_runs"] = np.asarray(n_runs)
    return out


def predict(r: np.ndarray, t: np.ndarray, rec: Mapping[str, Any],
            label: Optional[str] = None) -> np.ndarray:
    """The fitted model's pg/mL at ``r`` µm and ``t`` s. ``constant`` needs the group
    ``label`` (its own rate); ``piecewise`` is one history for every point."""
    D, boundary = float(rec["D"]), str(rec.get("boundary", "infinite"))
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)
    g = next((g for g in rec.get("groups") or () if g["label"] == label), {})
    level = float(g.get("background") or 0.0)
    if rec.get("rate") == "constant":
        q = g.get("q_fg_s")
        if q is None:
            return np.full(np.broadcast(r, t).shape, np.nan)
        return float(q) * unit_response(r, t, D, boundary) + level
    b = list(rec.get("breaks_s") or ())
    out = np.zeros(np.broadcast(r, t).shape)
    for j, qj in enumerate(rec.get("q_fg_s") or ()):
        out = out + float(qj) * (unit_response(r, t - b[j], D, boundary)
                                 - unit_response(r, t - b[j + 1], D, boundary))
    return out + level


def group_time(rec: Mapping[str, Any], label: str) -> Optional[float]:
    """The time (s since the source started) of the group named ``label``."""
    for g in rec.get("groups") or ():
        if g["label"] == str(label):
            return float(g["t_s"])
    return None


def to_signal(c: np.ndarray, curve: Mapping[str, Any]) -> np.ndarray:
    """Concentration → signal through a standard curve (``table.standard_curve``'s record):
    Langmuir ``Bmax·c / (Kd + c)``, linear ``a + b·c`` — the inverse of its conversion."""
    c = np.asarray(c, dtype=float)
    if curve.get("model") == "langmuir":
        bmax, kd = float(curve["bmax"]), float(curve["kd"])
        with np.errstate(divide="ignore", invalid="ignore"):
            return bmax * c / (kd + c)
    return float(curve.get("intercept", 0.0)) + float(curve.get("slope", 1.0)) * c


def rate_rows(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per rate the fit reports: ``piecewise`` one per interval (its start, end and
    midpoint in hours), ``constant`` one per timepoint (from 0 to it: an average)."""
    rows = []
    if rec.get("rate") == "piecewise":
        b = list(rec.get("breaks_s") or ())
        by_t = sorted(rec.get("groups") or (), key=lambda g: g["t_s"])
        for j, qj in enumerate(rec.get("q_fg_s") or ()):
            rows.append({"label": by_t[j]["label"] if j < len(by_t) else "",
                         "t_start_h": b[j] / 3600.0, "t_end_h": b[j + 1] / 3600.0,
                         "q_fg_s": float(qj),
                         "background": float(by_t[j].get("background") or 0.0)
                         if j < len(by_t) else 0.0})
    else:
        for g in sorted(rec.get("groups") or (), key=lambda g: g["t_s"]):
            rows.append({"label": g["label"], "t_start_h": 0.0, "t_end_h": g["t_s"] / 3600.0,
                         "q_fg_s": float(g.get("q_fg_s") or 0.0),
                         "background": float(g.get("background") or 0.0)})
    for row in rows:
        row["t_mid_h"] = 0.5 * (row["t_start_h"] + row["t_end_h"])
        row["q_pg_h"] = row["q_fg_s"] * 3.6
        row["released_pg"] = row["q_pg_h"] * (row["t_end_h"] - row["t_start_h"])
    total = 0.0
    for row in rows:
        total = (total if rec.get("rate") == "piecewise" else 0.0) + row["released_pg"]
        row["cumulative_pg"] = total
    return rows


__all__ = ["unit_response", "design", "fit", "monte_carlo", "predict", "group_time",
           "to_signal", "rate_rows", "RATE_MODELS", "BOUNDARIES", "WEIGHTS", "BACKGROUNDS",
           "FG_UM3_TO_PG_ML", "D_RANGE", "MC_SEED"]
