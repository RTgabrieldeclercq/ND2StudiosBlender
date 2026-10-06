"""table_ops — the shared core of the TABLE SYNTHESIS nodes (V4.00 step 9).

``table.concat``, ``table.join`` and ``table.aggregate`` build NEW structure tables out of
existing ones. What they share lives here:

* reading one table of a Dataset as ``{column: array}`` (the plot nodes' idiom);
* TEXT columns — the first producers of them in the engine. A text column is always a
  fixed-width numpy unicode array (``dtype.kind == "U"``), never ``object``: the memo's
  digest refuses object arrays (``nodegraph.memo._canon``) and a dock reload refuses them
  too (``np.load(allow_pickle=False)``), so an object column would break caching and
  baking of every result downstream;
* the per-position provenance columns — ``group`` (``position_group``), ``position_name``
  and ``file`` (``source_file``) — materialised FROM EACH INPUT'S OWN per-M metadata. The
  GUI's tabulator (``nodelab_v2.tables``) fills ``group``/``file`` at display time from the
  OUTPUT Dataset's lists, which after a concatenation are input 0's: a row of input 1 would
  be named after input 0's position with the same index. Writing the columns here, under
  the tabulator's own names, both labels each row from its own source and stops the
  tabulator from re-filling them (it never overwrites an existing column);
* writing the table under a name, dropping any table already of that name first (a table
  re-emitted over an existing one would otherwise keep that one's extra columns at their
  old length — a ragged table, INV-13).

Qt-free; numpy only. Every edit-time helper here is TOTAL (it runs on every keystroke).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nodegraph.catalog._shared.labels import _voxel_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.metadata import (CONDITION_KEY, POSITION_GROUP_KEY, POSITION_NAME_KEY,
                                SOURCE_FILE_KEY)
from nodegraph.structure import StructureTable

#: The provenance columns a concatenation writes from each input's per-M metadata, as
#: ``(metadata key, column name)``. The column names are the GUI tabulator's own
#: (``nodelab_v2.tables.POSITION_GROUP_COLUMN`` / ``SOURCE_FILE_COLUMN``) so it does not add a
#: second, wrongly indexed copy at display time.
PER_M_COLUMNS: Tuple[Tuple[str, str], ...] = (
    (POSITION_GROUP_KEY, "group"),
    (POSITION_NAME_KEY, "position_name"),
    (SOURCE_FILE_KEY, "file"),
)
#: The column naming each row's experimental condition, and the one naming its input.
CONDITION_COLUMN = CONDITION_KEY
INPUT_COLUMN = "input"
#: Each row's time in seconds since its OWN input's first frame (from that input's per-frame
#: clock): after a concatenation the inputs may have been imaged at different intervals,
#: and the Dataset passed on carries input 0's clock only.
TIME_COLUMN = "time_s"
#: Each row's wall-clock time (Julian date) from its own input's per-frame clock — what Plot
#: Time Series' `clock` axis reads after a concatenation.
CLOCK_COLUMN = "time_jd"
#: The metadata key listing the tables a table node WROTE (``[[domain, layer], ...]``): a
#: combined or joined table repeats rows (or holds other inputs' rows) at positions that are
#: not this image's, so the Viewer's point and track overlays leave them out unless picked.
SYNTH_TABLES_KEY = "__table_synth__"


def frame_jd(metadata: Mapping[str, Any], t: np.ndarray) -> Optional[np.ndarray]:
    """Each row's wall-clock time (Julian date) from the Dataset's per-frame clock
    (``frame_time_jd``); ``None`` when it has none. A row whose ``t`` the clock does not
    reach is NaN."""
    jd = (metadata or {}).get("frame_time_jd")
    if not isinstance(jd, (list, tuple)) or not jd:
        return None
    try:
        v = [float(x) for x in jd]
    except (TypeError, ValueError):
        return None
    out = []
    for ti in np.asarray(t, dtype=float).tolist():
        i = int(ti) if math.isfinite(ti) else -1
        out.append(v[i] if 0 <= i < len(v) and math.isfinite(v[i]) else float("nan"))
    return np.asarray(out, dtype=float)


def frame_seconds(metadata: Mapping[str, Any], t: np.ndarray) -> Optional[np.ndarray]:
    """Each row's seconds since its Dataset's FIRST frame (``frame_time_jd``); ``None`` when
    it has no per-frame clock."""
    jd = frame_jd(metadata, t)
    if jd is None:
        return None
    first = float((metadata or {}).get("frame_time_jd")[0])
    return (jd - first) * 86400.0


def table_domain(modes: Mapping[str, Any], key: str = "domain",
                 default: Domain = Domain.LABEL) -> Domain:
    """The Domain a ``domain``-style Mode names; ``default`` for anything unreadable (total)."""
    try:
        return Domain(str((modes or {}).get(key) or default.value))
    except Exception:                                   # noqa: BLE001 — total by contract
        return default


def param_text(params: Mapping[str, Any], name: str, default: str = "") -> str:
    """A string param as typed, stripped; ``default`` when blank or unreadable (total)."""
    try:
        v = (params or {}).get(name)
        s = "" if v is None else str(v).strip()
    except Exception:                                   # noqa: BLE001 — total by contract
        s = ""
    return s or default


def read_table(ds: Dataset, domain: Domain, layer: str) -> Dict[str, np.ndarray]:
    """``{column: array}`` of the ``layer`` table of ``domain`` on ``ds``, in insertion order."""
    return {a.name: np.asarray(a.values) for a in ds.layers_on(domain) if a.layer == layer}


def is_text(arr: np.ndarray) -> bool:
    """A column holding text rather than numbers."""
    return np.asarray(arr).dtype.kind in "USO"


def text_column(values: Iterable[Any]) -> np.ndarray:
    """``values`` as a fixed-width numpy unicode column (``None`` → ``""``) — never object."""
    out = [("" if v is None else str(v)) for v in values]
    return np.asarray(out, dtype=str) if out else np.zeros(0, dtype="<U1")


def per_m_text(metadata: Mapping[str, Any], key: str, m: np.ndarray) -> Optional[np.ndarray]:
    """The per-M metadata list ``key`` read at each row's ``m`` as a text column; ``None`` when
    the Dataset carries no such list. A row whose ``m`` the list does not reach gets ``""``
    (a SHORT list labels what it can rather than mislabelling by a shifted index)."""
    vals = (metadata or {}).get(key)
    if not isinstance(vals, (list, tuple)) or not vals:
        return None
    out = []
    for v in np.asarray(m, dtype=float).tolist():
        i = int(v) if math.isfinite(v) else -1
        out.append(str(vals[i]) if 0 <= i < len(vals) and vals[i] is not None else "")
    return text_column(out)


def missing_like(arr: np.ndarray, n: int) -> np.ndarray:
    """``n`` MISSING values of ``arr``'s kind: ``""`` for text, NaN for numbers."""
    if is_text(arr):
        return np.zeros(n, dtype="<U1")
    return np.full(n, np.nan)


def stack_columns(parts: Sequence[np.ndarray], name: str, node: str) -> np.ndarray:
    """One column from per-input pieces: text stays text, numbers stay numbers (int widened
    to float only when a piece is missing, i.e. NaN-filled). Text beside numbers is refused —
    silently turning a measurement into strings would break every numeric reader."""
    kinds = {("text" if is_text(p) else "num") for p in parts if len(p)}
    if len(kinds) > 1:
        raise ValueError(
            f"{node}: column {name!r} holds text in one input and numbers in another — "
            f"rename one of them upstream, or drop it before combining.")
    if "text" in kinds:
        return text_column(v for p in parts for v in np.asarray(p).tolist())
    return np.concatenate([np.asarray(p) for p in parts]) if parts else np.zeros(0)


def write_table(ds: Dataset, domain: Domain, name: str, columns: Mapping[str, np.ndarray],
                z_kind: str) -> Dataset:
    """``ds`` with ``columns`` as the ``name`` table of ``domain`` — any table already of that
    name is dropped first, so no column of it survives at its old length."""
    out = ds
    for a in list(ds.layers_on(domain)):
        if a.layer == name:
            out = out.without(domain, a.name, name)
    marked = [list(x) for x in (out.metadata or {}).get(SYNTH_TABLES_KEY) or []
              if list(x) != [domain.value, name]]
    out = out.with_metadata(**{SYNTH_TABLES_KEY: marked + [[domain.value, name]]})
    return out.with_structure(StructureTable(domain, dict(columns), layer=name,
                                             z_kind=z_kind or "plane_index"))


def synthesized(metadata: Mapping[str, Any], domain: Domain) -> set:
    """The layer names of ``domain`` a table node wrote (total)."""
    try:
        return {str(lyr) for dom, lyr in ((metadata or {}).get(SYNTH_TABLES_KEY) or [])
                if str(dom) == domain.value}
    except Exception:                                   # noqa: BLE001 — total by contract
        return set()


def refuse_voxel_name(ds: Dataset, name: str, node: str) -> None:
    """Refuse an output table name that is also a Voxel layer on the wire: a table and a raster
    of one name are read as ONE label instance, and the new table's ids are not the raster's."""
    if name in _voxel_layers(ds):
        raise ValueError(
            f"{node}: {name!r} is already a Voxel layer on the wire (a mask or a label "
            f"raster). A table of the same name would be read as that raster's table, and "
            f"its ids are not the raster's — choose another Name.")


def envelope_columns(env: Any, domain: Domain, layer: str) -> List[str]:
    """The columns an edit-time envelope knows on ``(domain, layer)`` (total)."""
    try:
        return [c for d, lyr, c in getattr(env, "column_names", ()) or ()
                if d is domain and lyr == layer]
    except Exception:                                   # noqa: BLE001 — total by contract
        return []


def envelope_layer(env: Any, domain: Domain, want: str) -> str:
    """The table a blank-or-named socket denotes on an edit-time envelope: the name as typed,
    else the only table of ``domain`` the envelope knows (the compute's own rule), else ``""``
    (total)."""
    if want:
        return want
    try:
        known = list(dict.fromkeys(n for d, n in getattr(env, "layer_names", ()) or ()
                                   if d is domain))
    except Exception:                                   # noqa: BLE001 — total by contract
        return ""
    return known[0] if len(known) == 1 else ""


def envelope_has(env: Any, key: str) -> bool:
    """Whether an edit-time envelope carries the per-M list ``key`` (total)."""
    try:
        v = (getattr(env, "metadata", None) or {}).get(key)
        return isinstance(v, (list, tuple)) and bool(v)
    except Exception:                                   # noqa: BLE001 — total by contract
        return False


__all__ = ["PER_M_COLUMNS", "CONDITION_COLUMN", "INPUT_COLUMN", "TIME_COLUMN",
           "CLOCK_COLUMN", "SYNTH_TABLES_KEY", "frame_seconds", "frame_jd", "synthesized", "table_domain",
           "param_text", "read_table", "is_text", "text_column", "per_m_text",
           "missing_like", "stack_columns", "write_table", "refuse_voxel_name",
           "envelope_columns", "envelope_layer", "envelope_has"]
