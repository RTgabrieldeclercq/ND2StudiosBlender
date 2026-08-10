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
from nodegraph.structure import COORD_COLUMNS

#: canonical position of each coordinate column, for :func:`_ordered_columns`.
_COORD_ORDER = {name: i for i, name in enumerate(COORD_COLUMNS)}

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


def all_tables(dataset) -> Tables:
    """Every tabulatable table on a Dataset — the detected structures AND the coarse
    lattice attributes. The one entry point for the panel, the export and the LabLink
    worker alike, so a layer visible in the spreadsheet is always a layer you can also
    write to CSV or return over a session."""
    tables = structure_tables(dataset)
    tables.update(lattice_tables(dataset))
    return tables


def _ordered_columns(cols: Dict[str, np.ndarray]) -> List[str]:
    """Coordinate columns first (canonical order), then the rest alphabetically."""
    return sorted(cols, key=lambda n: (_COORD_ORDER.get(n, len(_COORD_ORDER)), n))


__all__ = ["Tables", "structure_tables", "lattice_tables", "all_tables",
           "_ordered_columns", "_COORD_ORDER", "_STRUCTURE_NAMES"]
