"""A figure's summary table (V5.00 step 5): the rows a ``table`` item shows.

Qt-free. :func:`summary_table` reduces the columns of one table — per value of a
``group_by`` column, or over every row — to the statistics asked for, formatted for a
paper (significant digits, not float noise); :func:`table_csv` is the same rows as CSV for
the export bundle.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from nodegraph.figure.charts import fmt_number

#: the statistics a table item may list, in the order they are written
STATS: Tuple[str, ...] = ("n", "mean", "sd", "sem", "median", "min", "max", "sum")

STAT_LABELS: Dict[str, str] = {"n": "n", "mean": "mean", "sd": "SD", "sem": "SEM",
                               "median": "median", "min": "min", "max": "max", "sum": "sum"}


def _reduce(v: np.ndarray, stat: str) -> float:
    v = v[np.isfinite(v)]
    if stat == "n":
        return float(v.size)
    if v.size == 0:
        return math.nan
    if stat == "mean":
        return float(v.mean())
    if stat == "sd":
        return float(v.std(ddof=1)) if v.size > 1 else math.nan
    if stat == "sem":
        return float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else math.nan
    if stat == "median":
        return float(np.median(v))
    if stat == "min":
        return float(v.min())
    if stat == "max":
        return float(v.max())
    if stat == "sum":
        return float(v.sum())
    raise ValueError(f"summary_table: unknown statistic {stat!r}; one of {STATS}")


def _fmt(v: float, stat: str, decimals: int) -> str:
    if stat == "n":
        return str(int(v)) if math.isfinite(v) else ""
    if not math.isfinite(v):
        return "–"
    if decimals >= 0:
        return f"{v:.{decimals}f}"
    return fmt_number(v)


def summary_table(cols: Mapping[str, Any], columns: Sequence[str], *, group_by: str = "",
                  stats: Sequence[str] = ("n", "mean", "sem"), decimals: int = -1
                  ) -> Tuple[List[str], List[List[str]]]:
    """``(header, rows)`` of the summary of ``columns`` of one table (``cols``: ``{name:
    array}``), one row per distinct value of ``group_by`` (sorted; one row in all when
    blank), with each of ``stats`` per column. ``decimals`` < 0 formats to significant
    digits. Columns that are not numeric or not on the table are skipped; a blank
    ``columns`` takes every numeric column except the coordinates."""
    stats = [s for s in stats if s in STATS] or ["n", "mean", "sem"]
    skip = {"id", "m", "t", "c", "z", "y", "x", "b"}
    names = [c for c in columns if c in cols and np.asarray(cols[c]).dtype.kind in "biuf"]
    if not columns:
        names = [c for c in cols if c not in skip and c != group_by
                 and np.asarray(cols[c]).dtype.kind in "biuf"]
    header = ([group_by] if group_by else []) + [
        f"{c} {STAT_LABELS[s]}" if s != "n" else "n" for c in names for s in stats]
    # one 'n' column is enough when every column shares it: keep the first, drop the rest
    seen_n = False
    keep: List[int] = []
    for k, h in enumerate(header):
        if h == "n":
            if seen_n:
                continue
            seen_n = True
        keep.append(k)
    n_rows = len(next(iter(cols.values()))) if cols else 0
    if group_by and group_by in cols:
        g = np.asarray(cols[group_by])
        keys = sorted({_key(v) for v in g.tolist()}, key=lambda s: (_num_key(s), s))
        groups = [(k, np.array([_key(v) == k for v in g.tolist()])) for k in keys]
    else:
        groups = [("", np.ones(n_rows, dtype=bool))]
    rows: List[List[str]] = []
    for label, mask in groups:
        row = [label] if group_by else []
        for c in names:
            v = np.asarray(cols[c], dtype=float)[mask]
            for s in stats:
                row.append(_fmt(_reduce(v, s), s, decimals))
        rows.append([row[k] for k in keep])
    return [header[k] for k in keep], rows


def _key(v: Any) -> str:
    if isinstance(v, float):
        return fmt_number(v) if math.isfinite(v) else "–"
    if isinstance(v, bytes):
        return v.decode("utf8", "replace")
    return str(v)


def _num_key(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.inf


def table_csv(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(header))
    for r in rows:
        w.writerow(list(r))
    return buf.getvalue()


__all__ = ["STATS", "STAT_LABELS", "summary_table", "table_csv"]
