"""io.write_table — write the tables on a wire to CSV / Parquet / Arrow (2026-11-02)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from nodegraph.catalog._base import register_node
from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.table_export import export_tables, own_file_keys, table_paths
from nodegraph.tables import all_tables

#: format → extension
_EXT = {"csv": ".csv", "parquet": ".parquet", "arrow": ".arrow"}
_KNOWN = {".csv": "csv", ".parquet": "parquet", ".arrow": "arrow"}


def _table_target(path: str, fmt: str) -> str:
    """``path`` with its format's extension — added when it has none (or a foreign one),
    refused when it names ANOTHER table format. ``auto`` keeps a known extension and falls
    back to ``.csv``."""
    if os.path.isdir(path) or path.endswith(("/", "\\")):
        raise ValueError(f"Export Table: {path!r} is a folder — add a file name")
    stem, ext = os.path.splitext(path)
    have = _KNOWN.get(ext.lower())
    if fmt == "auto":
        return path if have else path + ".csv"
    if have is None:
        return path + _EXT[fmt]
    if have != fmt:
        raise ValueError(f"Export Table: the file {os.path.basename(path)!r} ends in {ext} "
                         f"but Format is {fmt} — change one so they agree")
    return path


def _table_plan(ds: Dataset, target: str, fmt: str, layout: str, explicit_ext: bool
          ) -> Tuple[Dict, List[tuple], Optional[str], List[str]]:
    """``(tables, own_keys, own_ext, files)`` — what will be written where, decided before
    anything is, so Existing = refuse can check every file first."""
    tables = all_tables(ds)
    if not tables:
        raise ValueError(
            "Export Table: nothing tabulatable on this wire — it carries no structure tables "
            "(Label / Point / Track / Mesh) and no coarse lattice attributes. Wire it after "
            "a Measure, a detector or a table node. To save the image, use Export TIFF.")
    own = (sorted(tables, key=lambda k: (str(k[0]), str(k[1] or "")))
           if layout == "per_table" else own_file_keys(ds, tables))
    # auto with no extension typed: Parquet for a file of its own (a pixel table, an
    # over-long one), CSV for the long file a spreadsheet opens
    own_ext = ".parquet" if (fmt == "auto" and not explicit_ext) else None
    rest = [k for k in tables if k not in set(own)]
    if not rest and len(own) == 1:
        files = [os.path.splitext(target)[0] + own_ext if own_ext else target]
    else:
        files = ([target] if rest else []) + list(table_paths(target, own, own_ext).values())
    return tables, own, own_ext, files


def _compute_write_table(ctx: EvalContext) -> Dataset:
    """Write the tables on this wire, and hand the Dataset on unchanged.

    Resolved spec (2026-11-02, `build-node-v2`):

    * writes exactly what the spreadsheet shows and the GUI's Export… writes —
      :func:`nodegraph.tables.all_tables` through :mod:`nodegraph.table_export` — so a
      graph writes its tables on every run, per file of a batch, without anyone clicking;
    * ``layout`` auto: the ordinary tables share one long file; a PIXEL table (Measure,
      Members = pixel) or one longer than a spreadsheet opens gets a file of its own,
      ``<stem>_<layer><ext>``. per_table: every table its own file;
    * ``format`` auto: the extension typed decides; with none, the shared file is CSV and
      the own files Parquet. csv / parquet / arrow force every file, adding the extension or
      refusing one that names another format;
    * ``path`` required (empty refused, as Export TIFF/Figure do); ``existing`` refuse checks
      EVERY file before writing any; each file is written to ``<stem>.part<ext>`` and moved
      into place when complete; a pass-through tap.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    raw = str(ctx.params.get("path", "") or "").strip().strip('"')
    if not raw:
        raise ValueError("Export Table has no destination: set `path` (the Browse… button "
                         "beside it picks one)")
    fmt = str(modes.get("format", "auto"))
    layout = str(modes.get("layout", "auto"))
    explicit_ext = os.path.splitext(raw)[1].lower() in _KNOWN
    target = _table_target(raw, fmt)
    tables, own, own_ext, files = _table_plan(ds, target, fmt, layout, explicit_ext)
    if str(modes.get("existing", "overwrite")) == "refuse":
        there = [f for f in files if os.path.exists(f)]
        if there:
            raise ValueError(f"Export Table: {', '.join(os.path.basename(f) for f in there)} "
                             f"already exist(s) and Existing is 'refuse' — move them, pick "
                             f"another name, or set Existing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    ctx.progress(0, 1, f"writing {len(files)} file(s) beside {os.path.basename(target)}")
    written = export_tables(tables, target, own, own_ext)
    rows = sum(n for _p, n in written)
    ctx.progress(1, 1, f"wrote {rows:,} rows to "
                       + ", ".join(os.path.basename(p) for p, _n in written))
    return ds


register_node(
    _compute_write_table, op_key="io.write_table", label="Export Table", category="io",
    inputs=[
        InDataset("data", description="The Dataset whose tables are written — every Label, "
                                      "Point and Track table and every coarse per-frame / "
                                      "per-position attribute the spreadsheet shows. It is "
                                      "handed on unchanged."),
        InString("path", "File", field=False, default="", path_kind="save_file",
                 path_filter="CSV (*.csv);;Parquet (*.parquet);;Arrow IPC (*.arrow);;"
                             "All files (*)",
                 path_hint="Browse… to choose where the tables are written",
                 description="Where the tables are written. There is no default and an empty "
                             "value is refused. A pixel table (and with Layout per_table, "
                             "every table) goes to a file of its own beside it, named "
                             "`<this name>_<table>`; nothing else in the folder is touched."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("format", ["auto", "csv", "parquet", "arrow"], default="auto", label="Format",
             description="The file format. CSV opens in any spreadsheet; Parquet and Arrow "
                         "are compact, typed and fast to read back in Python, R or MATLAB.",
             choice_docs={
                 "auto": "Decided by the extension typed in File. With none, the shared "
                         "table file is CSV and a pixel table's own file is Parquet — a "
                         "million-row CSV is about a gigabyte and past what Excel opens.",
                 "csv": "Every file as comma-separated text — readable anywhere, but large "
                        "and slow for a pixel table, and Excel stops at 1,048,576 rows.",
                 "parquet": "Every file as Parquet: columnar and compressed, typically a "
                            "tenth of the CSV, read by pandas, polars, R (arrow) and MATLAB "
                            "(parquetread). Needs pyarrow.",
                 "arrow": "Every file as Arrow IPC (Feather v2): uncompressed and the fastest "
                          "to load back into Python. Needs pyarrow.",
             }),
        Mode("layout", ["auto", "per_table"], default="auto", label="Layout",
             description="How the tables on the wire are spread over files.",
             choice_docs={
                 "auto": "The ordinary tables share one file, each row tagged with its "
                         "`domain` and `layer` — the GUI Export's layout. A pixel table, or "
                         "any table longer than a spreadsheet opens, gets a file of its own.",
                 "per_table": "Every table gets a file of its own, `<name>_<table>` — "
                              "plain columns, no `domain`/`layer` tags, ready for a script "
                              "that reads one kind of table.",
             }),
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description="What to do when a file is already at a path this node writes.",
             choice_docs={
                 "overwrite": "Replace it — the default, since this node only runs again when "
                              "the tables changed. The old file is replaced only once the new "
                              "one is complete.",
                 "refuse": "Raise instead of touching any existing file — checked for every "
                           "file before the first is written, so a refused export writes "
                           "nothing.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Write the tables on this wire — Label, Point and Track tables and the coarse "
                "per-frame attributes — to CSV, Parquet or Arrow, a pixel table to a file of "
                "its own, and hand the Dataset on unchanged.")
