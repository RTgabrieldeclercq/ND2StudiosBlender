"""Exported table files (2026-10-22) — the one format *File → Export table…* writes and the
Load Table node (``io.load_table``) reads back.

**The rows.** One long-form table: a leading ``domain`` and ``layer`` column naming which
table each row belongs to, then the union of every table's columns, coordinates first
(:func:`nodelab_v2.export._long_rows`). A cell is empty (CSV) or null (Parquet / Arrow) where
that row's table has no such column, so splitting on ``(domain, layer)`` and dropping the
columns that are empty throughout a group gives each table back exactly
(:func:`split_tables`).

**The origin record.** What made the file, as JSON (:func:`origin_record`): when
(``created``) and on which machine, which graph file, page and node, the chain of nodes in
front of that node, the source file(s) the data came from (``source_files`` — what the
node's *Load file* button opens), the Dataset's axes and JSON-safe metadata (calibration,
per-position names, channel display), and one ``tables`` placement per table: its domain,
layer, kind (``structure`` / ``lattice``), ``z_kind``, columns, row count, and the columns the
export DERIVED from metadata (``file``, ``group``), which a reload re-derives rather than
storing twice. Parquet and Arrow carry it in the schema metadata under
:data:`ORIGIN_META_KEY`, so it travels inside the file; a CSV has no such slot, so it sits
beside the file as ``<file>.origin.json`` (:func:`sidecar_path`). A file with no record — a
CSV from elsewhere, a sidecar that did not travel — still loads: placement falls back to its
``domain``/``layer`` columns (or one Point table named after the file), and the axes to the
largest index each table uses (:func:`infer_axes`).

The writer is :mod:`nodelab_v2.export` (it tabulates through the GUI's
:mod:`nodelab_v2.tables`); this module owns the format both sides agree on, so the record's
keys are spelled once. Qt-free; numpy + stdlib, pyarrow lazily imported for Parquet / Arrow.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import re
import threading
from enum import Enum
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain, is_lattice, is_structure

#: The record's ``format`` value and schema version. A reader refuses nothing on a newer
#: version — every key is optional — but the version says which keys to expect.
ORIGIN_FORMAT = "nd2studios.table"
ORIGIN_VERSION = 1
#: The Parquet / Arrow schema-metadata key holding the record (UTF-8 JSON).
ORIGIN_META_KEY = b"nd2studios.origin"
#: A CSV's record sits beside it: ``cells.csv`` → ``cells.csv.origin.json``.
SIDECAR_SUFFIX = ".origin.json"
#: What the Browse… dialog offers.
TABLE_FILE_FILTER = ("Tables (*.parquet *.arrow *.csv);;Parquet (*.parquet);;"
                     "Arrow IPC (*.arrow);;CSV (*.csv);;All files (*)")

#: Metadata Load Table stamps on its output: the file's absolute path, a one-line account of
#: where it came from, and the origin record itself (less its ``metadata`` block, which is
#: already the Dataset's own). A re-export of a loaded table carries the stamp forward, so
#: its new record says it was loaded from this file.
TABLE_FILE_KEY = "table_file"
TABLE_NOTE_KEY = "table_file_note"
TABLE_ORIGIN_KEY = "table_file_origin"

#: The coordinate columns every structure table carries (``nodegraph.structure``) — filled
#: with zeros (``id`` with ``0..n-1``) on a table read from a file that lacks them, and said so.
_COORDS = ("id", "m", "t", "c", "z", "y", "x")
#: One metadata value larger than this (as JSON) is left out of the record and listed under
#: ``metadata_skipped`` — a per-pixel array has no business in a header.
_META_VALUE_LIMIT = 1 << 20

_TRUE = frozenset({"true", "True", "TRUE"})
_FALSE = frozenset({"false", "False", "FALSE"})
_INT_RE = re.compile(r"^[+-]?\d+$")


class TableData(NamedTuple):
    """One table read back from a file: where it goes and its columns."""

    domain: Domain
    layer: Optional[str]
    #: ``"structure"`` (rows are detected elements) or ``"lattice"`` (rows are axis indices)
    kind: str
    columns: Dict[str, np.ndarray]
    #: the structure table's ``z_kind`` when the record names it, else ``None`` (inferred)
    z_kind: Optional[str]
    #: coordinate columns this file did not have, filled with zeros
    filled: Tuple[str, ...] = ()


# ── paths and identity ──────────────────────────────────────────────────────────────────

def clean_path(value: Any) -> str:
    """A path as typed or pasted: whitespace and one layer of quotes stripped."""
    return str(value or "").strip().strip('"').strip("'").strip()


def sidecar_path(path: str) -> str:
    """Where a CSV's origin record lives: ``<path>.origin.json``."""
    return str(path) + SIDECAR_SUFFIX


def file_kind(path: str) -> str:
    """``"parquet"``, ``"arrow"`` or ``"csv"`` — by extension; anything else reads as CSV."""
    low = str(path).lower()
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith((".arrow", ".feather", ".ipc")):
        return "arrow"
    return "csv"


def _stat(path: str) -> Optional[Tuple[int, int]]:
    try:
        st = os.stat(path)
        return int(st.st_size), int(st.st_mtime_ns)
    except OSError:
        return None


def file_identity(path: str) -> Optional[tuple]:
    """``(normalised path, size, mtime_ns[, sidecar size, sidecar mtime_ns])`` — what the
    memo keys a Load Table on (:attr:`nodegraph.registry.NodeSpec.file_identity`), so a file
    re-exported to the same name is a miss. ``None`` when the file does not exist."""
    p = clean_path(path)
    if not p:
        return None
    st = _stat(p)
    if st is None:
        return None
    key: tuple = (os.path.normcase(os.path.abspath(p)),) + st
    side = _stat(sidecar_path(p)) if file_kind(p) == "csv" else None
    return key + (side if side is not None else ())


# ── writing: the record ─────────────────────────────────────────────────────────────────

class _Skip(Exception):
    """A value with no JSON form — left out of the record."""


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON types (numpy scalars and arrays unwrapped, tuples as lists,
    enums by value). Raises :class:`_Skip` for anything with no faithful JSON form — a
    provider, a function, an object — so the caller can leave that key out rather than
    write a ``repr`` that reads back as a string."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUSO":
            raise _Skip(value.dtype)
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if isinstance(k, Enum):
                k = k.value
            if not isinstance(k, (str, int, float, bool)):
                raise _Skip(type(k))
            out[str(k)] = json_safe(v)
        return out
    raise _Skip(type(value))


def _safe_metadata(md: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The JSON-safe part of a Dataset's metadata, and the keys left out."""
    keep: Dict[str, Any] = {}
    skipped: List[str] = []
    for k, v in (md or {}).items():
        try:
            safe = json_safe(v)
            if len(json.dumps(safe)) > _META_VALUE_LIMIT:
                raise _Skip("too large")
        except (_Skip, TypeError, ValueError):
            skipped.append(str(k))
            continue
        keep[str(k)] = safe
    return keep, skipped


def _axes_dict(axes: Any) -> Dict[str, int]:
    return {a: int(getattr(axes, a, 1) or 1) for a in ("b", "m", "t", "z", "c", "y", "x")}


def dataset_origin(dataset: Any, tables: Mapping[Tuple[str, Optional[str]],
                                                  Mapping[str, Any]],
                   derived: Optional[Mapping[Tuple[str, Optional[str]], Sequence[str]]] = None
                   ) -> Dict[str, Any]:
    """The Dataset half of an origin record: its axes, JSON-safe metadata, the source files
    it was read from (``source_path``, first appearance order) and one placement per table
    in ``tables`` (``{(domain, layer): {column: values}}`` — what the export writes).
    ``derived`` names, per table, the columns the tabulation added from metadata."""
    from nodegraph.metadata import SOURCE_PATH_KEY
    md = dict(getattr(dataset, "metadata", None) or {})
    meta, skipped = _safe_metadata(md)
    paths = md.get(SOURCE_PATH_KEY)
    sources = list(dict.fromkeys(str(p) for p in paths if p)) \
        if isinstance(paths, (list, tuple)) else []
    placed: List[Dict[str, Any]] = []
    zfn = getattr(dataset, "structure_zkind", None)
    for (dom, layer), cols in sorted(tables.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        try:
            d = Domain(dom)
        except ValueError:
            continue
        n = max((len(v) for v in cols.values()), default=0)
        entry: Dict[str, Any] = {
            "domain": d.value, "layer": layer or "",
            "kind": "structure" if is_structure(d) else "lattice",
            "columns": list(cols), "rows": int(n),
            "derived": list((derived or {}).get((dom, layer), ()) or ()),
        }
        if is_structure(d) and callable(zfn):
            zk = zfn(d, layer)
            if zk:
                entry["z_kind"] = str(zk)
        placed.append(entry)
    return {
        "source_files": sources,
        "axes": _axes_dict(getattr(dataset, "axes", None)),
        "metadata": meta,
        "metadata_skipped": skipped,
        "tables": placed,
    }


def origin_record(dataset: Any, tables: Mapping[Tuple[str, Optional[str]], Mapping[str, Any]],
                  *, derived: Optional[Mapping[Tuple[str, Optional[str]], Sequence[str]]] = None,
                  context: Optional[Mapping[str, Any]] = None,
                  columns: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The full origin record for one export. ``context`` is what only the caller knows —
    ``graph``, ``page``, ``node``, ``chain``, ``app`` (the GUI fills it from the workspace);
    ``columns`` names the long form's leading keys when they had to be renamed
    (``{"domain": "_domain", "layer": "layer"}``)."""
    import platform
    rec: Dict[str, Any] = {
        "format": ORIGIN_FORMAT,
        "version": ORIGIN_VERSION,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": platform.node(),
    }
    for k, v in (context or {}).items():
        try:
            rec[str(k)] = json_safe(v)
        except _Skip:
            continue
    rec["columns"] = dict(columns or {"domain": "domain", "layer": "layer"})
    rec.update(dataset_origin(dataset, tables, derived))
    return rec


def encode_origin(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=False).encode("utf-8")


# ── reading: the record ─────────────────────────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_ORIGIN_CACHE: Dict[tuple, Dict[str, Any]] = {}
_SCAN_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _decode(raw: Any) -> Dict[str, Any]:
    try:
        got = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    except (ValueError, UnicodeDecodeError, AttributeError):
        return {}
    return got if isinstance(got, dict) else {}


def _read_bytes(path: str) -> bytes:
    # read whole and close: a memory map or a handle pyarrow keeps open would lock the file
    # on Windows, and the next export to the same name would fail
    with open(path, "rb") as fh:
        return fh.read()


def read_origin(path: str) -> Dict[str, Any]:
    """The file's origin record, ``{}`` when it has none (or cannot be read). Cheap: the
    Parquet footer / Arrow schema / CSV sidecar only, cached on :func:`file_identity`."""
    p = clean_path(path)
    ident = file_identity(p)
    if ident is None:
        return {}
    with _CACHE_LOCK:
        hit = _ORIGIN_CACHE.get(ident)
    if hit is not None:
        return hit
    rec: Dict[str, Any] = {}
    kind = file_kind(p)
    try:
        if kind == "csv":
            side = sidecar_path(p)
            if os.path.isfile(side):
                with open(side, "r", encoding="utf-8") as fh:
                    rec = _decode(fh.read())
        else:
            import pyarrow as pa
            if kind == "parquet":
                import pyarrow.parquet as pq
                schema = pq.read_schema(pa.BufferReader(_read_bytes(p)))
            else:
                schema = pa.ipc.open_file(pa.BufferReader(_read_bytes(p))).schema
            raw = (schema.metadata or {}).get(ORIGIN_META_KEY)
            rec = _decode(raw) if raw else {}
    except Exception:  # noqa: BLE001 — a record that will not read is a file without one
        rec = {}
    with _CACHE_LOCK:
        if len(_ORIGIN_CACHE) > 64:
            _ORIGIN_CACHE.clear()
        _ORIGIN_CACHE[ident] = rec
    return rec


# ── reading: the rows ───────────────────────────────────────────────────────────────────

class _Col(NamedTuple):
    values: np.ndarray        # typed (Arrow) or str (CSV)
    null: np.ndarray          # bool mask
    typed: bool               # False: CSV strings, typed per table later


def _object_array(items: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        out[i] = v
    return out


def _arrow_col(col: Any) -> _Col:
    import pyarrow as pa
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    null = np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    t = arr.type
    if pa.types.is_integer(t):
        vals = np.asarray(arr.fill_null(0).to_numpy(zero_copy_only=False))
    elif pa.types.is_floating(t):
        vals = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=float)
    elif pa.types.is_boolean(t):
        vals = np.asarray(arr.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool)
    else:
        vals = _object_array(arr.to_pylist())
    return _Col(vals, null, True)


def read_columns(path: str) -> Tuple[List[str], Dict[str, _Col]]:
    """The long form's header and its columns, in file order."""
    p = clean_path(path)
    kind = file_kind(p)
    if kind == "csv":
        with open(p, "r", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return [], {}
        header = [h.strip() for h in rows[0]]
        body = rows[1:]
        cols: Dict[str, _Col] = {}
        for i, h in enumerate(header):
            if not h or h in cols:
                continue
            raw = [r[i] if i < len(r) else "" for r in body]
            vals = np.asarray(raw, dtype=object)
            cols[h] = _Col(vals, np.asarray([s == "" for s in raw], dtype=bool), False)
        return [h for h in header if h in cols], cols
    import pyarrow as pa
    if kind == "parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(pa.BufferReader(_read_bytes(p)))
    else:
        table = pa.ipc.open_file(pa.BufferReader(_read_bytes(p))).read_all()
    header = list(table.column_names)
    return header, {h: _arrow_col(table.column(h)) for h in header}


def _typed_strings(raw: np.ndarray) -> np.ndarray:
    """A CSV column's cells as the narrowest type every cell parses as: int64, float64,
    bool, else the strings themselves. Empty cells are already excluded by the caller."""
    vals = [str(s) for s in raw]
    if not vals:
        return np.zeros(0, dtype=float)
    if all(_INT_RE.match(s) for s in vals):
        try:
            return np.asarray([int(s) for s in vals], dtype=np.int64)
        except OverflowError:
            pass
    try:
        return np.asarray([float(s) for s in vals], dtype=float)
    except ValueError:
        pass
    if all(s in _TRUE or s in _FALSE for s in vals):
        return np.asarray([s in _TRUE for s in vals], dtype=bool)
    return _object_array(vals)


def _group_values(col: _Col, idx: np.ndarray) -> Optional[np.ndarray]:
    """``col`` restricted to rows ``idx`` — ``None`` when every one of them is empty (the
    column belongs to another table)."""
    null = col.null[idx]
    if null.all():
        return None
    vals = col.values[idx]
    if not col.typed:
        if not null.any():
            return _typed_strings(vals)
        typed = _typed_strings(vals[~null])
        if typed.dtype.kind in "iuf":
            out = np.full(len(vals), np.nan)
            out[~null] = typed
            return out
        out = _object_array([None] * len(vals))
        out[~null] = typed
        return out
    if null.any():
        if vals.dtype.kind in "iu":
            out = vals.astype(float)
            out[null] = np.nan
            return out
        if vals.dtype.kind == "O":
            vals = vals.copy()
            vals[null] = None
    return vals


def _placement(origin: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in origin.get("tables") or ():
        if isinstance(entry, Mapping):
            out[(str(entry.get("domain", "")), str(entry.get("layer", "") or ""))] = dict(entry)
    return out


def _stem(path: str) -> str:
    base = os.path.basename(clean_path(path)) or "table"
    while True:
        root, ext = os.path.splitext(base)
        if not ext or not root:
            return base
        base = root


def split_tables(path: str, header: Sequence[str], cols: Mapping[str, _Col],
                 origin: Optional[Mapping[str, Any]] = None) -> List[TableData]:
    """Cut the long form back into its tables. Grouped on the record's (or the default)
    ``domain``/``layer`` columns; within a group, a column empty on every row belongs to
    another table and is dropped, and so is a column the record says the export DERIVED
    from metadata (a reload re-derives it). A file with no ``domain`` column is one Point
    table named after the file. A structure table the record does not place (a file from
    elsewhere) gets zeros for a missing coordinate column (``id``: ``0..n-1``), named in
    ``filled``; a placed one comes back with exactly its own columns."""
    origin = origin or {}
    keys = origin.get("columns") or {}
    dom_key = str(keys.get("domain") or "domain")
    lay_key = str(keys.get("layer") or "layer")
    place = _placement(origin)
    n = max((len(c.values) for c in cols.values()), default=0)
    if dom_key in cols:
        doms = [str(v) if not nl else "" for v, nl in zip(cols[dom_key].values,
                                                         cols[dom_key].null)]
    else:
        doms = [Domain.POINT.value] * n
    if lay_key in cols:
        lays = [str(v) if not nl else "" for v, nl in zip(cols[lay_key].values,
                                                         cols[lay_key].null)]
    else:
        lays = [_stem(path) if dom_key not in cols else ""] * n
    order: Dict[Tuple[str, str], List[int]] = {}
    for i, k in enumerate(zip(doms, lays)):
        order.setdefault(k, []).append(i)
    data_cols = [h for h in header if h not in (dom_key, lay_key)]
    known = {d.value for d in Domain}
    out: List[TableData] = []
    for (dom, lay), rows in order.items():
        if dom not in known:
            raise ValueError(
                f"{os.path.basename(path)}: rows on domain {dom!r}, which this app does not "
                f"have — the `{dom_key}` column must name one of "
                f"{', '.join(sorted(known))}")
        d = Domain(dom)
        if d is Domain.VOXEL:
            raise ValueError(f"{os.path.basename(path)}: a Voxel table cannot be loaded — "
                             f"per-voxel layers are images, not tables")
        idx = np.asarray(rows, dtype=np.int64)
        where = place.get((dom, lay), {})
        derived = set(str(c) for c in (where.get("derived") or ()))
        table: Dict[str, np.ndarray] = {}
        for h in data_cols:
            if h in derived:
                continue
            got = _group_values(cols[h], idx)
            if got is not None:
                table[h] = got
        filled: List[str] = []
        # only a table the record does not place: an exported table is restored exactly as it
        # was — a mesh's faces or a track summary legitimately has no `y`/`x`
        if is_structure(d) and not where:
            for c in _COORDS:
                if c not in table:
                    table[c] = (np.arange(len(idx), dtype=np.int64) if c == "id"
                                else np.zeros(len(idx), dtype=np.int64))
                    filled.append(c)
        zk = where.get("z_kind")
        out.append(TableData(d, lay or None, "structure" if is_structure(d) else "lattice",
                             table, str(zk) if zk else None, tuple(filled)))
    return out


def read_tables(path: str, origin: Optional[Mapping[str, Any]] = None) -> List[TableData]:
    """Every table in the file at ``path`` (its record read unless ``origin`` is given)."""
    p = clean_path(path)
    header, cols = read_columns(p)
    return split_tables(p, header, cols, read_origin(p) if origin is None else origin)


def infer_z_kind(table: TableData) -> str:
    """A structure table's ``z_kind`` when the file does not say: integer ``z`` throughout
    is a plane index (2D), anything fractional is sub-pixel (3D)."""
    if table.z_kind:
        return table.z_kind
    z = np.asarray(table.columns.get("z", np.zeros(0)))
    if z.dtype.kind in "iub":
        return "plane_index"
    try:
        zf = z.astype(float)
    except (TypeError, ValueError):
        return "plane_index"
    return "plane_index" if np.all(np.isfinite(zf) & (zf == np.round(zf))) else "subpixel"


def _extent(v: np.ndarray) -> int:
    try:
        f = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return 1
    f = f[np.isfinite(f)]
    return int(math.floor(float(f.max()))) + 1 if f.size and f.max() >= 0 else 1


def infer_axes(tables: Sequence[TableData]) -> Dict[str, int]:
    """Axis sizes for a file with no record: the largest index each table uses, plus one —
    ``b``/``m``/``t``/``c``/``z`` from the index and coordinate columns, ``y``/``x`` from a
    structure table's coordinates (so the frame at least contains every point)."""
    ax = {a: 1 for a in ("b", "m", "t", "z", "c", "y", "x")}
    for t in tables:
        for a in ax:
            if a in t.columns and (a not in ("y", "x") or t.kind == "structure"):
                ax[a] = max(ax[a], _extent(t.columns[a]))
    return ax


def scan(path: str) -> Dict[str, Any]:
    """The edit-time view of a table file, cached on :func:`file_identity`: ``origin`` (its
    record), ``axes`` (the record's, else :func:`infer_axes` — which reads the file once),
    and ``tables``: ``[(domain, layer, kind, columns)]``. Total: ``{}`` for a missing file,
    ``{"error": …}`` for one that will not read."""
    p = clean_path(path)
    ident = file_identity(p)
    if ident is None:
        return {}
    with _CACHE_LOCK:
        hit = _SCAN_CACHE.get(ident)
    if hit is not None:
        return hit
    origin = read_origin(p)
    got: Dict[str, Any] = {"origin": origin}
    try:
        placed = [e for e in (origin.get("tables") or ()) if isinstance(e, Mapping)]
        if origin.get("axes") and placed:
            got["axes"] = {a: int(v) for a, v in dict(origin["axes"]).items()}
            got["tables"] = [
                (Domain(str(e["domain"])), str(e.get("layer") or "") or None,
                 str(e.get("kind") or ("structure" if is_structure(Domain(str(e["domain"])))
                                       else "lattice")),
                 tuple(c for c in (e.get("columns") or ())
                       if c not in set(e.get("derived") or ())))
                for e in placed]
        else:
            tables = read_tables(p, origin)
            got["axes"] = infer_axes(tables)
            got["tables"] = [(t.domain, t.layer, t.kind, tuple(t.columns)) for t in tables]
    except Exception as exc:  # noqa: BLE001 — an edit-time read must never raise
        got = {"origin": origin, "error": str(exc)}
    with _CACHE_LOCK:
        if len(_SCAN_CACHE) > 64:
            _SCAN_CACHE.clear()
        _SCAN_CACHE[ident] = got
    return got


# ── describing the record ───────────────────────────────────────────────────────────────

def _when(origin: Mapping[str, Any]) -> str:
    raw = str(origin.get("created") or "")
    try:
        return _dt.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw


def source_files(origin: Mapping[str, Any]) -> List[str]:
    """The source files the record names, in order (``[]`` for none)."""
    got = origin.get("source_files") or []
    return [str(p) for p in got if p] if isinstance(got, list) else []


def locate_source(recorded: str, table_path: str) -> str:
    """Where a recorded source file is NOW: the recorded path when it exists, else a file of
    the same name beside the table file (the pair was moved together, or the table came from
    the other machine), else ``""``."""
    rec = clean_path(recorded)
    if rec and os.path.isfile(rec):
        return rec
    name = os.path.basename(rec.replace("\\", "/")) if rec else ""
    if name:
        here = os.path.join(os.path.dirname(os.path.abspath(clean_path(table_path))), name)
        if os.path.isfile(here):
            return here
    return ""


def _node_name(origin: Mapping[str, Any]) -> str:
    node = origin.get("node") or {}
    if not isinstance(node, Mapping):
        return ""
    label = str(node.get("label") or "")
    op = str(node.get("op") or "")
    return f"“{label}” ({op})" if label and op and label != op else (label or op)


def origin_summary(origin: Mapping[str, Any], path: str = "") -> str:
    """One line on where a table file came from — the node card's note and the payload's
    :data:`TABLE_NOTE_KEY`. Deterministic for one record (the edit-time envelope and the
    pulled payload must carry the same string)."""
    if not origin:
        return (f"{os.path.basename(clean_path(path))}: no origin record — placed by its "
                f"domain/layer columns" if path else "no origin record")
    parts: List[str] = []
    when = _when(origin)
    node = _node_name(origin)
    parts.append(("exported " + when if when else "exported") + (f" from {node}" if node else ""))
    page = origin.get("page") or {}
    if isinstance(page, Mapping) and page.get("name"):
        parts.append(f"page “{page['name']}”")
    graph = str(origin.get("graph") or "")
    if graph:
        parts.append(os.path.basename(graph))
    srcs = source_files(origin)
    if srcs:
        first = os.path.basename(srcs[0].replace("\\", "/"))
        parts.append("source " + first + (f" +{len(srcs) - 1}" if len(srcs) > 1 else ""))
    return " · ".join(parts)


def origin_rows(origin: Mapping[str, Any], path: str = "") -> List[Tuple[str, str]]:
    """``(label, value)`` rows describing the record in full, for the inspector."""
    rows: List[Tuple[str, str]] = []
    if path:
        rows.append(("File", clean_path(path)))
    if not origin:
        rows.append(("Origin", "none recorded — the file was not written by this app's "
                               "export, or its .origin.json sidecar did not travel with it"))
        return rows
    when = _when(origin)
    machine = str(origin.get("machine") or "")
    if when:
        rows.append(("Created", when + (f" on {machine}" if machine else "")))
    app = origin.get("app") or {}
    if isinstance(app, Mapping) and (app.get("name") or app.get("commit")):
        rows.append(("App", " ".join(str(app[k]) for k in ("name", "commit") if app.get(k))))
    graph = str(origin.get("graph") or "")
    rows.append(("Graph", graph or "(unsaved graph)"))
    page = origin.get("page") or {}
    if isinstance(page, Mapping) and (page.get("name") or page.get("id")):
        kind = f" ({page['kind']})" if page.get("kind") else ""
        rows.append(("Page", f"{page.get('name') or page.get('id')}{kind}"))
    node = _node_name(origin)
    if node:
        nid = (origin.get("node") or {}).get("id") if isinstance(origin.get("node"), Mapping) \
            else ""
        rows.append(("Node", node + (f" · {nid}" if nid else "")))
    chain = origin.get("chain") or []
    if isinstance(chain, list) and chain:
        names = [str(c.get("label") or c.get("op") or "?") for c in chain
                 if isinstance(c, Mapping)]
        rows.append(("Chain", " → ".join(names + ([node] if node else []))))
    srcs = source_files(origin)
    for i, s in enumerate(srcs):
        here = locate_source(s, path) if path else ""
        state = "" if not path else (" (found)" if here == clean_path(s) else
                                     f" (missing — found beside the table: {here})" if here
                                     else " (missing)")
        rows.append(("Source" if i == 0 else "", s + state))
    ax = origin.get("axes") or {}
    if isinstance(ax, Mapping) and ax:
        shown = " ".join(f"{a.upper()}{int(ax[a])}" for a in ("b", "m", "t", "z", "c", "y", "x")
                         if a in ax and (a != "b" or int(ax[a]) > 1))
        rows.append(("Axes", shown))
    md = origin.get("metadata") or {}
    if isinstance(md, Mapping):
        cal = [f"{md[k]:g} µm/px" if k == "pixel_size_um" else f"z {md[k]:g} µm"
               for k in ("pixel_size_um", "z_step_um")
               if isinstance(md.get(k), (int, float))]
        if cal:
            rows.append(("Calibration", ", ".join(cal)))
    for e in origin.get("tables") or ():
        if not isinstance(e, Mapping):
            continue
        lay = str(e.get("layer") or "")
        rows.append(("Table", f"{e.get('domain')}{(' · ' + lay) if lay else ''}: "
                              f"{int(e.get('rows') or 0):,} rows × "
                              f"{len(e.get('columns') or ())} columns"))
    return rows


__all__ = [
    "ORIGIN_FORMAT", "ORIGIN_VERSION", "ORIGIN_META_KEY", "SIDECAR_SUFFIX",
    "TABLE_FILE_FILTER", "TABLE_FILE_KEY", "TABLE_NOTE_KEY", "TABLE_ORIGIN_KEY",
    "TableData", "clean_path", "sidecar_path", "file_kind", "file_identity",
    "json_safe", "dataset_origin", "origin_record", "encode_origin", "read_origin",
    "read_columns", "split_tables", "read_tables", "infer_z_kind", "infer_axes", "scan",
    "source_files", "locate_source", "origin_summary", "origin_rows",
]
