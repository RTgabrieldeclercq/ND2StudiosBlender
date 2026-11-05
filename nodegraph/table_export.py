"""Structure-table export (Phase 6 / V2.00 §10 — "the export path reads from here").

Takes the per-``(domain, layer)`` tables that :func:`nodegraph.tables.all_tables` produces
off a pulled Dataset and writes them to disk — the detected structures AND (since V2.17)
the coarse lattice attributes, so anything the spreadsheet can show is also exportable.
Reading through the same one function is what keeps those two in step. Qt-free (a caller
supplies the path): the GUI's Export…, the headless LabLink worker and the
``io.write_table`` node all write through here. numpy + stdlib for CSV, pyarrow **lazily
imported** for Arrow/Parquet so the core has no hard pyarrow dependency.

**It lives in the engine (2026-11-02)**, moved from ``nodelab_v2.export`` (which re-exports
it), because ``io.write_table`` is a catalog node and ``nodegraph`` may not import
``nodelab_v2``.

Two layouts, chosen per table:

* **One long file** — the original layout, unchanged. Every table lands in ``<path>`` with
  leading ``domain``/``layer`` columns and the union of their columns, so a Dataset with a
  Label table and two Point tables is still one sheet.
* **A file of its own** (2026-11-02) — a PIXEL table
  (:data:`nodegraph.tables.PIXEL_TABLES_KEY`, Measure with Members = pixel) or any table
  longer than a spreadsheet can open (:data:`EXCEL_MAX_ROWS`) goes to
  ``<stem>_<layer><ext>``, written COLUMN-wise. Two reasons. The long form unions every
  table's columns, so a million pixel rows would sit under the object table's columns as
  blank cells and the object rows under the pixel columns the same way. And the row-wise
  writer builds one Python value per cell: 12.6 M rows × 20 columns is ~250 M Python objects,
  several GB and minutes before a byte reaches the disk.

Extension picks the format: ``.csv``, ``.parquet`` or ``.arrow`` (Arrow IPC). Every file is
written to ``<stem>.part<ext>`` and moved into place only once complete, so an interrupted
export never leaves a truncated file under the real name.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.tables import Tables, _ordered_columns, all_tables, is_pixel_table

#: Rows a spreadsheet can open, header included (Excel's and LibreOffice's limit). A table
#: with more data rows than ``EXCEL_MAX_ROWS - 1`` is written to a file of its own.
EXCEL_MAX_ROWS = 1_048_576

#: Rows formatted per step of the streaming CSV writers. Bounds their memory to one chunk's
#: Python values whatever the table's length; the value only trades speed against memory.
_CHUNK = 65_536

#: what the extension of an export path writes
_FORMATS = {".csv": "csv", ".parquet": "parquet", ".arrow": "arrow"}


def _n_rows(cols: Dict[str, np.ndarray]) -> int:
    return max((len(v) for v in cols.values()), default=0)


def _key_order(key: Tuple[str, Optional[str]]) -> Tuple[str, str]:
    """Sort key for a ``(domain, layer)`` table key — a ``None`` layer (lattice tables)
    sorts as ``""`` instead of failing to compare with a named one."""
    return (str(key[0]), "" if key[1] is None else str(key[1]))


def _long_header(tables: Tables) -> Tuple[List[str], List[str]]:
    """``(header, attr_names)`` of the long form: leading ``domain``/``layer`` columns, then
    the union of attribute columns with the coordinate columns first."""
    attr_names: List[str] = []
    for cols in tables.values():
        for n in _ordered_columns(cols):
            if n not in attr_names:
                attr_names.append(n)
    # keep coordinate columns first across the union
    attr_names = _ordered_columns({n: None for n in attr_names})
    # the injected leading columns must not collide with an attribute column literally
    # named "domain"/"layer" (real structure tables never do, but stay robust): prefix
    # with "_" until unique.
    dom_key, lay_key = "domain", "layer"
    while dom_key in attr_names:
        dom_key = "_" + dom_key
    while lay_key in attr_names:
        lay_key = "_" + lay_key
    return [dom_key, lay_key, *attr_names], attr_names


def _cells(values: Optional[np.ndarray], start: int, stop: int) -> list:
    """Rows ``start:stop`` of one column as Python values — exactly what a per-cell
    ``.item()`` gave, at ``tolist`` speed. ``None`` (not ``""``) for a missing cell: csv
    renders it empty, and Arrow keeps a numeric column numeric."""
    if values is None:
        return [None] * (stop - start)
    arr = np.asarray(values)
    seg = arr[start:min(stop, len(arr))]
    out = seg.tolist()
    if seg.dtype == object:
        out = [v.item() if isinstance(v, np.generic) else v for v in out]
    if len(out) < stop - start:                 # a ragged column: blank past its end
        out.extend([None] * (stop - start - len(out)))
    return out


def _iter_long_rows(tables: Tables, attr_names: Sequence[str]) -> Iterator[list]:
    for (dom, layer), cols in sorted(tables.items()):
        n = _n_rows(cols)
        lay = layer if layer is not None else ""
        for start in range(0, n, _CHUNK):
            stop = min(n, start + _CHUNK)
            parts = [_cells(cols.get(name), start, stop) for name in attr_names]
            for i in range(stop - start):
                yield [dom, lay, *(p[i] for p in parts)]


def _long_rows(tables: Tables) -> Tuple[List[str], List[list]]:
    """Flatten all tables into (header, rows) long form: leading ``domain``/``layer``
    columns then the union of attribute columns (coordinate columns first)."""
    header, attr_names = _long_header(tables)
    return header, list(_iter_long_rows(tables, attr_names))


def _part_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}.part{ext}"


def _commit(tmp: str, path: str) -> None:
    os.replace(tmp, path)


def export_csv(tables: Tables, path: str) -> int:
    """Write the tables as one long-form CSV; returns the row count. Streamed a chunk at a
    time, so memory does not grow with the table."""
    header, attr_names = _long_header(tables)
    tmp = _part_path(path)
    n = 0
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in _iter_long_rows(tables, attr_names):
            w.writerow(row)
            n += 1
    _commit(tmp, path)
    return n


def export_arrow(tables: Tables, path: str) -> int:
    """Write the tables as a single Parquet/Arrow file (pyarrow lazily imported)."""
    import pyarrow as pa
    header, rows = _long_rows(tables)
    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}
    _write_arrow_table(pa.table(columns), path)
    return len(rows)


def _write_arrow_table(table, path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = _part_path(path)
    if path.lower().endswith(".arrow"):
        with pa.OSFile(tmp, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    else:
        pq.write_table(table, tmp)
    _commit(tmp, path)


def _arrow_column(values: np.ndarray, n: int):
    """One numpy column as an Arrow array of length ``n`` — numeric columns straight from
    their buffer, text as strings, a short column padded with nulls."""
    import pyarrow as pa
    arr = np.asarray(values)
    if arr.dtype == object or arr.dtype.kind in "US":
        vals = [None if v is None else str(v) for v in arr.tolist()]
        vals.extend([None] * (n - len(vals)))
        return pa.array(vals, type=pa.string())
    if len(arr) == n:
        return pa.array(arr)
    mask = np.zeros(n, dtype=bool)
    mask[len(arr):] = True
    full = np.zeros(n, dtype=arr.dtype)
    full[:len(arr)] = arr
    return pa.array(full, mask=mask)


def write_columns(cols: Dict[str, np.ndarray], path: str) -> int:
    """Write ONE table to ``path`` column-wise (format by extension); returns its rows.

    Parquet and Arrow are built from the numpy buffers directly. CSV goes through
    ``pyarrow.csv`` when pyarrow is installed and through a chunked :mod:`csv` writer when
    it is not — slower, but bounded in memory either way."""
    names = _ordered_columns(cols)
    n = _n_rows(cols)
    fmt = _FORMATS.get(os.path.splitext(path)[1].lower(), "csv")
    if fmt != "csv":
        import pyarrow as pa
        _write_arrow_table(pa.table({name: _arrow_column(cols[name], n) for name in names}),
                           path)
        return n
    tmp = _part_path(path)
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names)
            for start in range(0, n, _CHUNK):
                stop = min(n, start + _CHUNK)
                parts = [_cells(cols[name], start, stop) for name in names]
                w.writerows(zip(*parts))
    else:
        table = pa.table({name: _arrow_column(cols[name], n) for name in names})
        pacsv.write_csv(table, tmp)
    _commit(tmp, path)
    return n


def own_file_keys(dataset, tables: Tables) -> List[Tuple[str, Optional[str]]]:
    """The tables written to a file of their own: every pixel table, and any table with more
    rows than a spreadsheet can open. In :func:`all_tables` order (sorted keys)."""
    return [k for k in sorted(tables, key=_key_order)
            if is_pixel_table(dataset, k) or _n_rows(tables[k]) > EXCEL_MAX_ROWS - 1]


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "table"


def table_paths(path: str, keys: Iterable[Tuple[str, Optional[str]]],
                ext: Optional[str] = None) -> Dict[tuple, str]:
    """``{key: file}`` for the tables written to their own files: ``<stem>_<layer><ext>``,
    with the domain added when two keys would share a name. ``ext`` overrides the
    extension of ``path``."""
    stem, path_ext = os.path.splitext(path)
    ext = ext or path_ext
    keys = list(keys)
    names = [_slug(layer or dom) for dom, layer in keys]
    out: Dict[tuple, str] = {}
    for (dom, layer), nm in zip(keys, names):
        if names.count(nm) > 1:
            nm = _slug(f"{dom}_{layer or ''}")
        out[(dom, layer)] = f"{stem}_{nm}{ext or '.csv'}"
    return out


def export_tables(tables: Tables, path: str,
                  own_file: Sequence[Tuple[str, Optional[str]]] = (),
                  own_ext: Optional[str] = None) -> List[Tuple[str, int]]:
    """Write ``tables`` to ``path`` (format by extension) and return ``[(file, rows), …]``.

    The keys in ``own_file`` go to files of their own (:func:`table_paths`), column-wise, in
    ``own_ext``'s format when given (``io.write_table``'s auto format: Parquet for a pixel
    table beside a CSV of the rest); the rest share ``path`` in the long form. When EVERY
    table is an own-file one and there is only one, it is written to ``path`` itself (with
    ``own_ext``) — exporting a lone pixel table should produce the file that was asked for."""
    own = [k for k in sorted(tables, key=_key_order) if k in set(own_file)]
    rest = {k: v for k, v in tables.items() if k not in set(own)}
    written: List[Tuple[str, int]] = []
    if not rest and len(own) == 1:
        if own_ext:
            path = os.path.splitext(path)[0] + own_ext
        return [(path, write_columns(tables[own[0]], path))]
    if rest:
        lower = path.lower()
        n = (export_arrow(rest, path) if lower.endswith((".parquet", ".arrow"))
             else export_csv(rest, path))
        written.append((path, n))
    for key, p in table_paths(path, own, own_ext).items():
        written.append((p, write_columns(tables[key], p)))
    return written


def export_dataset_files(dataset, path: str) -> List[Tuple[str, int]]:
    """Export a pulled Dataset's tabulatable attributes to ``path`` (extension picks the
    format) — structure tables and coarse lattice attributes alike — and return every file
    written with its row count. Pixel tables and over-long tables get files of their own
    (:func:`own_file_keys`). Raises :class:`ValueError` if the Dataset carries nothing."""
    tables = all_tables(dataset)
    if not tables:
        raise ValueError(
            "nothing tabulatable on this output to export — it carries no structure "
            "tables (Label / Point / Track / Mesh) and no coarse lattice attributes "
            "(Frame / Plane / Timepoint / Multipoint / Channel / Global). Per-voxel image "
            "layers are not exported here; view them in the Viewer, gather the voxels of "
            "chosen regions with Measure (Members = pixel), or move a layer to a coarser "
            "domain with transform.transfer_domain first.")
    return export_tables(tables, path, own_file_keys(dataset, tables))


def export_dataset(dataset, path: str) -> int:
    """:func:`export_dataset_files`, returning the total row count written."""
    return sum(n for _p, n in export_dataset_files(dataset, path))


__all__ = ["export_dataset", "export_dataset_files", "export_tables", "export_csv",
           "export_arrow", "write_columns", "own_file_keys", "table_paths",
           "EXCEL_MAX_ROWS", "all_tables"]
