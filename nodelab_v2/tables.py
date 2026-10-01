"""Tabulation of a pulled Dataset — the Qt-free half of the spreadsheet/export path.

A pulled :class:`~nodegraph.dataset.Dataset` carries its detected structures as
element-indexed :class:`~nodegraph.dataset.AttributeLayer`\\ s on the Label / Point /
Track domains, and its coarse acquisition attributes on the Frame / Plane / Timepoint /
Multipoint / Channel / Global domains. Both are grouped here into ``{(domain, layer):
{column: values}}`` tables — the one shape the spreadsheet panel renders and the export
path writes.

**Why this is its own module.** These four functions read nothing but numpy and
:mod:`nodegraph`, and three consumers need them: the Qt spreadsheet panel, the CSV/Arrow
export, and — since the LabLink worker landed — a headless process with no display at
all. They used to live in :mod:`nodelab_v2.spreadsheet`, which imports ``PySide6`` at
module scope for its widget, so :mod:`nodelab_v2.export` claimed to be "Qt-free" in its
own docstring while transitively dragging in the whole GUI toolkit. That was harmless
while every consumer was the app itself and is not harmless for a worker the hub spawns
per session: Qt is tens of megabytes of import before the handshake, it is one more thing
that can fail on a machine with no display, and a worker that dies before ``hello`` is an
``open_timeout`` whose real cause is invisible.

``Voxel`` is deliberately excluded from :func:`lattice_tables` — a Voxel layer is one row
per voxel (12.6 M for a 3-position 4-frame 1024² series), it is image-shaped by
definition, and the Viewer already renders it.

Qt-free; numpy + stdlib only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice, is_structure
from nodegraph.metadata import BATCH_FILE_KEY, POSITION_GROUP_KEY, SOURCE_FILE_KEY
from nodegraph.structure import BATCH_COLUMN, COORD_COLUMNS

#: The header naming each row's source file, injected by :func:`with_source_file` when the
#: Dataset came from a **file bundle**. Deliberately NOT a member of
#: :data:`~nodegraph.structure.COORD_COLUMNS`: that tuple is the contract that *every*
#: structure table carries those columns, and this one is derived at tabulation time from
#: per-M metadata rather than emitted by the node that built the table.
SOURCE_FILE_COLUMN = "file"

#: Column naming which position GROUP — which specimen — each row came from, filled from the
#: per-M :data:`~nodegraph.metadata.POSITION_GROUP_KEY` list.
#:
#: The counterpart of :data:`SOURCE_FILE_COLUMN` and it exists for the same reason: after the
#: acquisition is loaded there is nothing in a table distinguishing one specimen's rows from
#: another's except ``m``, an integer that means "the 31st position" and names nothing the
#: experiment was about. On the lab's six-mosaic file a 54-position Measure produces rows
#: from six different samples, and without this column, grouping them afterwards means
#: knowing by heart that 27-35 was the fourth one.
POSITION_GROUP_COLUMN = "group"

#: canonical position of each coordinate column, for :func:`_ordered_columns`. ``file`` sits
#: immediately before ``m``, the column it explains: reading left to right gives which file,
#: then which position inside it.
_COORD_ORDER = {
    name: i for i, name in enumerate(
        tuple(c for col in COORD_COLUMNS
              for c in ((SOURCE_FILE_COLUMN, "m") if col == "m" else (col,))))
}

#: domain VALUES whose rows are detected elements rather than axis indices
#: (only affects a caller's status wording).
_STRUCTURE_NAMES = frozenset(d.value for d in Domain if is_structure(d))

#: the table shape every consumer here speaks.
Tables = Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]]


def structure_tables(dataset) -> Tables:
    """Group a Dataset's structure-domain attributes into ``{(domain, layer):
    {col_name: values}}`` — one entry per detected Label/Point/Track instance."""
    tables: Tables = {}
    if dataset is None or not hasattr(dataset, "attributes"):
        return tables
    for (domain, layer, name), attr in dataset.attributes.items():
        if not is_structure(domain):
            continue
        tables.setdefault((domain.value, layer), {})[name] = attr.values
    return tables


def lattice_tables(dataset) -> Tables:
    """Group a Dataset's COARSE lattice attributes into the same ``{(domain, layer):
    {col: values}}`` shape, with the domain's own axes unrolled as leading index columns.

    A lattice attribute's array is shaped by ``axes_of(domain)`` in :data:`AXIS_ORDER`, so
    a Frame layer is ``(M, T)`` and a Global layer is a scalar. Every layer on one domain
    shares that index space, so they merge into one table: ``m, t, drift_y, drift_x``. The
    index columns are named for the axes themselves, which is also what makes the CSV
    self-describing without a separate header.

    ``Voxel`` is excluded — see the module docstring. A layer whose stored shape does not
    match the domain's expected shape is skipped rather than reshaped: that only happens
    mid-edit while an axis change is propagating, and a wrong unrolling would be worse
    than a briefly missing tab."""
    tables: Tables = {}
    if dataset is None or not hasattr(dataset, "attributes"):
        return tables
    axes = getattr(dataset, "axes", None)
    for (domain, layer, name), attr in dataset.attributes.items():
        if not is_lattice(domain) or domain is Domain.VOXEL:
            continue
        dom_axes = tuple(a for a in AXIS_ORDER if a in (axes_of(domain) or frozenset()))
        values = np.asarray(attr.values)
        if axes is not None:
            try:
                if tuple(values.shape) != axes.shape_for(domain):
                    continue                       # stale mid-edit shape — skip, never guess
            except (ValueError, AttributeError):   # pragma: no cover - defensive
                continue
        key = (domain.value, layer)
        cols = tables.setdefault(key, {})
        if dom_axes and "__idx__" not in cols:
            # one row per index tuple, in C order — the same order `ravel()` gives below
            grid = np.indices(values.shape).reshape(len(dom_axes), -1)
            for i, ax in enumerate(dom_axes):
                cols[ax] = grid[i]
            cols["__idx__"] = np.empty(0)          # marker: index columns already built
        cols[name] = values.reshape(-1) if dom_axes else np.asarray([values.reshape(())])
    for cols in tables.values():
        cols.pop("__idx__", None)
    return tables


def _with_per_m_column(dataset, tables: Tables, key: str, column: str) -> Tables:
    """Add ``column`` to every table that has an ``m``, filled from the per-M list ``key``.

    The shared body of :func:`with_source_file` and :func:`with_position_group`. Both answer
    the same question about a row — *which of the things I loaded did this come from?* — off
    two different :data:`~nodegraph.metadata.PER_POSITION_KEYS` lists, and the rule that
    matters is identical for both, so it is written once here.

    **A short or out-of-range list is dropped whole, never partially applied.** The same rule
    :func:`~nodegraph.metadata.read_origin_um` and
    :func:`~nodegraph.metadata.position_subset` follow, and for the sharpest reason of the
    three: a wrong value in this column is not obviously wrong. It is a filename or a
    specimen name sitting next to a measurement, which is exactly what a reader trusts
    without checking. Omitting the column makes them ask; misfilling it does not.

    Mutates ``tables`` in place and returns it (the caller owns freshly-built dicts).
    """
    values = (getattr(dataset, "metadata", None) or {}).get(key)
    if not isinstance(values, (list, tuple)) or not values:
        return tables
    lookup = [str(v) for v in values]
    for cols in tables.values():
        col_m = cols.get("m")
        if col_m is None or column in cols:
            continue
        idx = np.asarray(col_m)
        if idx.size and (idx.min() < 0 or idx.max() >= len(lookup)):
            continue                       # the list does not cover these rows — say nothing
        cols[column] = np.asarray([lookup[int(i)] for i in idx.reshape(-1)], dtype=object)
    return tables


def with_position_group(dataset, tables: Tables) -> Tables:
    """Add a :data:`POSITION_GROUP_COLUMN` naming the specimen each row's position is in.

    A no-op unless the Dataset carries the per-M
    :data:`~nodegraph.metadata.POSITION_GROUP_KEY` list, which the source card stamps when
    grouping resolves (:func:`nodelab_v2.runner._with_position_groups`).

    Survives a ``util.select_group`` upstream, and reads correctly afterwards: the key is in
    ``PER_POSITION_KEYS``, so a selected group's rows all carry that one group's name rather
    than a stale list from before the selection. It is most useful on the branch that did NOT
    select — a Measure over all 54 positions still reports which of the six samples each row
    belongs to, which is the join a spreadsheet needs to group by specimen.
    """
    return _with_per_m_column(dataset, tables, POSITION_GROUP_KEY, POSITION_GROUP_COLUMN)


def with_source_file(dataset, tables: Tables) -> Tables:
    """Add a :data:`SOURCE_FILE_COLUMN` to every table that has an ``m`` column, naming
    the file each row's position came from. A no-op unless the Dataset carries the per-M
    :data:`~nodegraph.metadata.SOURCE_FILE_KEY` list, which only a **file bundle** stamps.

    A bundle lays K files end to end on the multipoint axis
    (:class:`~nodegraph.provider.MultiSourceProvider`), so after it there is nothing in a
    table distinguishing one file's rows from another's except ``m`` — an integer that
    means "the 7th position of the concatenation" and no longer names anything the user
    chose. This turns it back into the filename they picked.

    **A short or out-of-range list is dropped whole, never partially applied** — the same
    rule :func:`~nodegraph.metadata.read_origin_um` and
    :func:`~nodegraph.metadata.position_subset` follow, and for a sharper reason here: a
    wrong name on a spreadsheet row is not obviously wrong. It is a filename next to a
    measurement, which is exactly what a reader trusts without checking. Omitting the
    column makes them ask; misfilling it does not.

    Mutates ``tables`` in place and returns it (the caller owns freshly-built dicts).
    """
    return _with_per_m_column(dataset, tables, SOURCE_FILE_KEY,
                              SOURCE_FILE_COLUMN)


def with_batch_file(dataset, tables: Tables) -> Tables:
    """Fill :data:`SOURCE_FILE_COLUMN` from the per-B batch list, for a batched Dataset.

    The batch twin of :func:`with_source_file`, and it must run INSTEAD of it rather than
    after: on a batch the per-M ``source_file`` list describes one member (a batch member
    is a whole acquisition, so its own list has nothing to do with the batch's ``m``), and
    filling the column from it labels every row of every file with the FIRST file's name.
    That is the precise failure :func:`_with_per_m_column` warns about — a filename beside
    a measurement, trusted without checking — reached by the one route that docstring did
    not anticipate.

    Keyed by the table's own :data:`~nodegraph.structure.BATCH_COLUMN`, so a row is named
    by the member it actually belongs to. A table without that column gets nothing: on a
    batch it cannot say which file it came from, which is exactly what
    ``Dataset.with_structure`` refuses to let happen in the first place.
    """
    md = getattr(dataset, "metadata", None) or {}
    values = md.get(BATCH_FILE_KEY)
    if not isinstance(values, (list, tuple)) or not values:
        return tables
    lookup = [str(v) for v in values]
    for cols in tables.values():
        col_b = cols.get(BATCH_COLUMN)
        if col_b is None or SOURCE_FILE_COLUMN in cols:
            continue
        idx = np.asarray(col_b)
        if idx.size and (idx.min() < 0 or idx.max() >= len(lookup)):
            continue                       # the list does not cover these rows — say nothing
        cols[SOURCE_FILE_COLUMN] = np.asarray(
            [lookup[int(i)] for i in idx.reshape(-1)], dtype=object)
    return tables


def all_tables(dataset) -> Tables:
    """Every tabulatable table on a Dataset — the detected structures AND the coarse
    lattice attributes. The one entry point for the panel, the export and the LabLink
    worker alike, so a layer visible in the spreadsheet is always a layer you can also
    write to CSV or return over a session — and, for a file bundle, one that names the
    file every row came from (:func:`with_source_file`)."""
    tables = structure_tables(dataset)
    tables.update(lattice_tables(dataset))
    # A BATCH names rows by their member; only an unbatched Dataset may name them by the
    # per-M bundle list, because on a batch that list belongs to one member and would put
    # the first file's name on every row (see `with_batch_file`).
    batched = int(getattr(getattr(dataset, "axes", None), "b", 1) or 1) > 1
    tables = (with_batch_file(dataset, tables) if batched
              else with_source_file(dataset, tables))
    return with_position_group(dataset, tables)


def _ordered_columns(cols: Dict[str, np.ndarray]) -> List[str]:
    """Coordinate columns first (canonical order), then the rest alphabetically.

    The two provenance columns sort immediately AFTER the coordinates and before the
    measurements, because they qualify the address rather than measuring anything: ``m`` on
    its own is meaningless in a spreadsheet, and ``file``/``group`` are what turn it back
    into the file and the specimen the user chose. Reading `id, m, t, group, area, …` puts
    the question and its answer next to each other.
    """
    def rank(n: str) -> tuple:
        if n in _COORD_ORDER:
            return (0, _COORD_ORDER[n], n)
        if n in (SOURCE_FILE_COLUMN, POSITION_GROUP_COLUMN):
            return (1, 0, n)
        return (2, 0, n)

    return sorted(cols, key=rank)


__all__ = [
    "with_batch_file","Tables", "structure_tables", "lattice_tables", "all_tables",
           "with_position_group", "POSITION_GROUP_COLUMN",
           "with_source_file", "SOURCE_FILE_COLUMN",
           "_ordered_columns", "_COORD_ORDER", "_STRUCTURE_NAMES"]
