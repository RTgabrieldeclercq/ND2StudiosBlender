"""Structure-table export — moved to :mod:`nodegraph.table_export` (2026-11-02).

Re-exported here unchanged so every ``from nodelab_v2.export import …`` keeps working. The
writer moved into the engine with the tabulation (:mod:`nodegraph.tables`) when the
``io.write_table`` node arrived: a catalog node may not import ``nodelab_v2``, and the GUI's
Export… and the node must write the same files. There is still exactly one implementation.
"""
from __future__ import annotations

from nodegraph.table_export import (  # noqa: F401 — the re-export is the point
    EXCEL_MAX_ROWS, _long_rows, all_tables, export_arrow, export_csv, export_dataset,
    export_dataset_files, export_tables, own_file_keys, table_paths, write_columns,
)
from nodegraph.tables import structure_tables  # noqa: F401 — was re-exported before the move

__all__ = ["export_dataset", "export_dataset_files", "export_tables", "export_csv",
           "export_arrow", "write_columns", "own_file_keys", "table_paths",
           "EXCEL_MAX_ROWS", "structure_tables", "all_tables"]
