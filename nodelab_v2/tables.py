"""Tabulation of a pulled Dataset — moved to :mod:`nodegraph.tables` (2026-11-02).

Re-exported here unchanged so every ``from nodelab_v2.tables import …`` keeps working. The
module moved into the engine when ``io.write_table`` arrived: a catalog node needs the same
tabulation the spreadsheet shows, and ``nodegraph`` may not import ``nodelab_v2``. There is
still exactly one implementation.
"""
from __future__ import annotations

from nodegraph.tables import (  # noqa: F401 — the re-export is the point
    _COORD_ORDER, _PROVENANCE_COLUMNS, _STRUCTURE_NAMES, PIXEL_TABLES_KEY,
    POSITION_GROUP_COLUMN, SOURCE_FILE_COLUMN, Tables, _ordered_columns,
    _with_per_m_column, all_tables, is_pixel_table, lattice_tables, mark_pixel_table,
    pixel_layers, structure_tables, with_batch_file, with_position_group, with_source_file,
)

__all__ = [
    "with_batch_file", "Tables", "structure_tables", "lattice_tables", "all_tables",
    "with_position_group", "POSITION_GROUP_COLUMN", "with_source_file", "SOURCE_FILE_COLUMN",
    "PIXEL_TABLES_KEY", "pixel_layers", "mark_pixel_table", "is_pixel_table",
    "_ordered_columns", "_COORD_ORDER", "_STRUCTURE_NAMES"]
