"""Group comparisons, correlations, stars and bracket layout for charts (V5.00 step 1).

scipy does the tests; this module decides which pairs are compared, spells the result the
way a figure prints it (``***`` / ``p = 0.012``), and lays brackets out so they never cross.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TESTS = ("welch", "student", "mann_whitney", "paired", "wilcoxon", "kruskal", "anova",
         "tukey")

#: p thresholds → stars, highest significance first
STAR_LEVELS: Tuple[Tuple[float, str], ...] = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def stars(p: float, *, levels: Sequence[Tuple[float, str]] = STAR_LEVELS) -> str:
    """``***`` / ``**`` / ``*`` / ``ns`` for ``p``."""
    if p is None or not math.isfinite(p):
        return "n/a"
    for thr, s in levels:
        if p < thr:
            return s
    return "ns"


def p_text(p: float) -> str:
    """``p < 0.001`` / ``p = 0.012`` / ``p = 0.31`` — two significant figures."""
    if p is None or not math.isfinite(p):
        return "p n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.2g}"


def _clean(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    return a[np.isfinite(a)]


def compare(groups: Mapping[str, Sequence[float]], *, test: str = "welch",
            pairs: Optional[Sequence[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
    """Compare the ``groups`` (``{label: values}``) with ``test``.

    Pairwise tests (``welch``, ``student``, ``mann_whitney``, ``paired``, ``wilcoxon``,
    ``tukey``) return one record per pair — ``pairs`` to choose them, else every pair in
    order; the omnibus tests (``kruskal``, ``anova``) return one record with ``a = "all"``.
    Each record: ``{"a", "b", "test", "stat", "p", "n_a", "n_b", "stars", "text"}``. A pair
    with too few values gets ``p = nan`` rather than an exception, so one empty group does
    not blank a figure."""
    from scipy import stats as S
    if test not in TESTS:
        raise ValueError(f"compare: unknown test {test!r}; one of {TESTS}")
    labels = list(groups)
    data = {k: _clean(groups[k]) for k in labels}
    out: List[Dict[str, Any]] = []

    def rec(a: str, b: str, stat: float, p: float) -> Dict[str, Any]:
        return {"a": a, "b": b, "test": test, "stat": float(stat), "p": float(p),
                "n_a": int(data[a].size) if a in data else sum(v.size for v in data.values()),
                "n_b": int(data[b].size) if b in data else 0,
                "stars": stars(p), "text": p_text(p)}

    if test in ("kruskal", "anova"):
        arrays = [data[k] for k in labels if data[k].size >= 2]
        if len(arrays) < 2:
            return [rec("all", "", float("nan"), float("nan"))]
        res = S.kruskal(*arrays) if test == "kruskal" else S.f_oneway(*arrays)
        return [rec("all", "", res.statistic, res.pvalue)]

    if test == "tukey":
        arrays = [data[k] for k in labels if data[k].size >= 2]
        names = [k for k in labels if data[k].size >= 2]
        if len(arrays) < 2:
            return []
        res = S.tukey_hsd(*arrays)
        want = set(tuple(p) for p in pairs) if pairs else None
        for i, j in combinations(range(len(names)), 2):
            if want is not None and (names[i], names[j]) not in want \
                    and (names[j], names[i]) not in want:
                continue
            out.append(rec(names[i], names[j], res.statistic[i, j], res.pvalue[i, j]))
        return out

    chosen = list(pairs) if pairs else list(combinations(labels, 2))
    for a, b in chosen:
        if a not in data or b not in data:
            raise ValueError(f"compare: pair ({a!r}, {b!r}) names a group not given")
        x, y = data[a], data[b]
        try:
            if test == "welch":
                if x.size < 2 or y.size < 2:
                    raise ValueError
                r = S.ttest_ind(x, y, equal_var=False)
            elif test == "student":
                if x.size < 2 or y.size < 2:
                    raise ValueError
                r = S.ttest_ind(x, y, equal_var=True)
            elif test == "mann_whitney":
                if x.size < 1 or y.size < 1:
                    raise ValueError
                r = S.mannwhitneyu(x, y, alternative="two-sided")
            elif test == "paired":
                if x.size != y.size or x.size < 2:
                    raise ValueError
                r = S.ttest_rel(x, y)
            else:                                   # wilcoxon
                if x.size != y.size or x.size < 2:
                    raise ValueError
                r = S.wilcoxon(x, y)
            out.append(rec(a, b, r.statistic, r.pvalue))
        except ValueError:
            out.append(rec(a, b, float("nan"), float("nan")))
    return out


def correlation(x: Sequence[float], y: Sequence[float], *, method: str = "pearson"
                ) -> Dict[str, Any]:
    """``{"r", "p", "n", "method"}`` — Pearson or Spearman over the finite pairs."""
    from scipy import stats as S
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    n = int(xs.size)
    if n < 3 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return {"r": float("nan"), "p": float("nan"), "n": n, "method": method}
    if method == "spearman":
        r = S.spearmanr(xs, ys)
    elif method == "pearson":
        r = S.pearsonr(xs, ys)
    else:
        raise ValueError(f"correlation: method must be pearson or spearman, not {method!r}")
    return {"r": float(r.statistic), "p": float(r.pvalue), "n": n, "method": method}


def bracket_levels(pairs: Sequence[Tuple[int, int]]) -> List[int]:
    """One level (0 = lowest) per bracket so that brackets over overlapping category spans
    stack instead of crossing: shortest spans first, each on the lowest level free over its
    span. Levels index the height steps the chart adds above the tallest mark."""
    spans = [(min(a, b), max(a, b)) for a, b in pairs]
    order = sorted(range(len(spans)), key=lambda k: (spans[k][1] - spans[k][0], spans[k][0]))
    levels = [0] * len(spans)
    taken: List[List[Tuple[int, int]]] = []
    for k in order:
        lo, hi = spans[k]
        lvl = 0
        while True:
            if lvl >= len(taken):
                taken.append([])
            if all(hi < a or lo > b for a, b in taken[lvl]):
                taken[lvl].append((lo, hi))
                levels[k] = lvl
                break
            lvl += 1
    return levels
