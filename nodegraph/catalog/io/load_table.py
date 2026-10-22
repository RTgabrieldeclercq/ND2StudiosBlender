"""Load Table (``io.load_table``) — an exported table file (Parquet / Arrow / CSV) back onto
the graph with the record of what made it (2026-10-22)."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

import nodegraph.table_file as TF
from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain, is_structure
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node

#: ``onto``'s default. Unlike Load Analysis (whose trial list is looked up beside the wired
#: image, so it needs the wire), a table file is named by its own path: dropped and pointed
#: at a file, the node runs with nothing wired. The card's *Load file* button switches it to
#: ``image`` when it wires the original file in.
DEFAULT_ONTO = "tables_only"
_AXES = ("b", "m", "t", "z", "c", "y", "x")


def _path(params: Mapping[str, Any]) -> str:
    return TF.clean_path((params or {}).get("path"))


def _envelope(axes_in: AxisSizes, md_in: Mapping[str, Any], params: Mapping[str, Any],
              modes: Mapping[str, Any]) -> Tuple[AxisSizes, Dict[str, Any]]:
    """The output's axes and metadata — ONE function for the compute and the edit-time
    ``meta_transform``, so the envelope a downstream node derives from and the payload it
    receives cannot disagree.

    ``image``: the wired Dataset's axes and metadata, plus every key of the record's
    metadata the image does not already carry (an analysis' ``dic_*`` / ``track_*`` stamps,
    a calibration a bare TIFF lacks) — never a per-position list, never a private key, and
    never over a value the image has. ``tables_only``: the record's axes and metadata
    exactly as exported (the axes inferred from the rows for a file with no record). Both
    gain the three ``table_file*`` stamps."""
    from nodegraph.metadata import PER_POSITION_KEYS
    path = _path(params)
    onto = str((modes or {}).get("onto", DEFAULT_ONTO) or DEFAULT_ONTO)
    sc = TF.scan(path) if path else {}
    origin = sc.get("origin") or {}
    note = TF.origin_summary(origin, path) if path else ""
    stamps: Dict[str, Any] = {TF.TABLE_FILE_KEY: os.path.abspath(path) if path else ""}
    rec_axes = dict(origin.get("axes") or {})
    if onto == "image":
        md = dict(md_in or {})
        for k, v in (origin.get("metadata") or {}).items():
            if v is None or str(k).startswith("__") or k in PER_POSITION_KEYS:
                continue
            md.setdefault(k, v)
        oy, ox = rec_axes.get("y"), rec_axes.get("x")
        if oy and ox and (int(oy), int(ox)) != (int(axes_in.y), int(axes_in.x)):
            note += (f" · exported on a {int(oy)}×{int(ox)} frame, attached to "
                     f"{int(axes_in.y)}×{int(axes_in.x)} — positions are in the exported "
                     f"frame's pixels")
        axes = axes_in
    else:
        md = dict(origin.get("metadata") or {})
        ax = sc.get("axes") or {}
        axes = AxisSizes(**{a: max(1, int(ax.get(a, 1) or 1)) for a in _AXES})
    stamps[TF.TABLE_NOTE_KEY] = note
    if origin:
        stamps[TF.TABLE_ORIGIN_KEY] = {k: v for k, v in origin.items() if k != "metadata"}
    md.update(stamps)
    return axes, md


def _meta_load_table(env, params, modes):
    """Edit-time twin of :func:`_compute_load_table` — the same :func:`_envelope`. A
    ``tables_only`` output's axes come from the file (its record, cached on the file's
    identity), so the derived defaults and the column pickers downstream are right before
    anything is pulled. Total: a file that will not read leaves the envelope as it was."""
    try:
        axes, md = _envelope(env.axes, env.metadata, params, modes)
    except Exception:  # noqa: BLE001 — an edit-time transform must never raise
        return env
    onto = str((modes or {}).get("onto", DEFAULT_ONTO) or DEFAULT_ONTO)
    if onto == "image":
        return replace(env, metadata=md)
    return replace(env, axes=axes, metadata=md, unknown_axes=frozenset())


def _index(values: np.ndarray, size: int, axis: str, what: str) -> np.ndarray:
    try:
        f = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"Load Table: {what}'s `{axis}` column is not numeric") from None
    if not np.all(np.isfinite(f)) or not np.all(f == np.round(f)):
        raise ValueError(f"Load Table: {what}'s `{axis}` column is not a whole-number index")
    iv = f.astype(np.int64)
    if iv.size and (iv.min() < 0 or iv.max() >= size):
        raise ValueError(
            f"Load Table: {what} reaches {axis}={int(iv.max())} but this Dataset has "
            f"{axis.upper()}={size} — wire the image the table was exported from, or switch "
            f"'onto' to tables_only")
    return iv


def _lattice(ds: Dataset, t: TF.TableData) -> Dataset:
    """A lattice table's attribute columns, each put back into its domain's array shape
    (``axes.shape_for``) at the indices its rows name. Cells no row names are NaN (a whole
    grid keeps its dtype); a Global table is its single row."""
    what = f"the {t.domain.value}" + (f" · {t.layer}" if t.layer else "") + " table"
    dom_axes = tuple(ds.axes.axis_list(t.domain))
    shape = tuple(ds.axes.shape_for(t.domain))
    missing = [a for a in dom_axes if a not in t.columns]
    if missing:
        raise ValueError(f"Load Table: {what} has no {', '.join(missing)} index column(s) — "
                         f"a {t.domain.value} row is addressed by {', '.join(dom_axes)}")
    attrs = [c for c in t.columns if c not in dom_axes]
    if not dom_axes:
        for name in attrs:
            v = np.asarray(t.columns[name])
            ds = ds.with_layer(t.domain, name, v[0] if v.size else np.asarray(np.nan),
                               layer=t.layer)
        return ds
    idx = tuple(_index(t.columns[a], int(ds.axes.size(a)), a, what) for a in dom_axes)
    flat = np.ravel_multi_index(idx, shape) if idx[0].size else np.zeros(0, dtype=np.int64)
    whole = flat.size == int(np.prod(shape)) and np.unique(flat).size == flat.size
    for name in attrs:
        v = np.asarray(t.columns[name])
        if whole:
            out = np.empty(shape, dtype=v.dtype)
        elif v.dtype.kind in "biuf":
            out = np.full(shape, np.nan)
        else:
            out = np.empty(shape, dtype=object)
        out[idx] = v
        ds = ds.with_layer(t.domain, name, out, layer=t.layer)
    return ds


def _attach(ds: Dataset, tables: List[TF.TableData], origin: Mapping[str, Any]) -> Dataset:
    """Every table onto ``ds``: structure tables on their domain and layer (with their
    ``z_kind``, and a correlation grid's mesh mark when the record carries one), lattice
    tables into their domain's arrays."""
    from nodegraph.catalog._shared.analysis_mesh import MESH_MARK
    rec_marks = dict((origin.get("metadata") or {}).get(MESH_MARK) or {})
    marks = dict(ds.metadata.get(MESH_MARK) or {})
    for t in tables:
        if t.kind == "lattice":
            ds = _lattice(ds, t)
            continue
        ds = ds.with_structure(StructureTable(t.domain, dict(t.columns), layer=t.layer,
                                              z_kind=TF.infer_z_kind(t)))
        if t.layer and t.layer in rec_marks and t.layer not in marks:
            marks[t.layer] = rec_marks[t.layer]
    if marks:
        ds = ds.with_metadata(**{MESH_MARK: marks})
    return ds


def _compute_load_table(ctx: EvalContext) -> Dataset:
    """A table file written by *File → Export table…* (or any long-form CSV / Parquet /
    Arrow file) read back onto the graph, with the record of what made it.

    The file is one long-form table — a ``domain`` and ``layer`` column per row, then the
    union of every table's columns (:mod:`nodegraph.table_file`). This node cuts it back into
    its tables: structure tables (Label / Point / Track / Mesh) land on their ORIGINAL
    domain and layer with their original columns and ``z_kind``; lattice tables (Frame /
    Plane / Timepoint / Multipoint / Channel / Global) go back into their domain's arrays at
    the indices their rows name. Columns the export derived from metadata (``file``,
    ``group``) are re-derived, not stored twice.

    **The origin record.** An export writes what made the file — when, on which machine,
    which graph, page and node, the chain in front of it, the source image file(s), the
    axes and the metadata — inside a Parquet / Arrow file, or beside a CSV as
    ``<file>.origin.json``. The output carries it as ``table_file_origin`` (with
    ``table_file`` and a one-line ``table_file_note``), so re-exporting a loaded table
    records that it was loaded from this file; the card and the inspector show it.

    ``onto = tables_only`` (the default) needs no wire: a Dataset with the export's axes
    and metadata and no image — for the Spreadsheet, plots and table nodes. ``onto = image``
    attaches the tables to the wired Dataset — the ORIGINAL image, which the card's *Load
    file* button loads on the Image Input page and wires in through a Page Input — so the
    Viewer draws them over it; one whose positions, frames, planes or batch differ from the
    export's is refused (rows would land on the wrong frame).

    Resolved spec: category io; op ``io.load_table``; reads nothing from the image; adds
    whatever the file holds (structure domains declared, the layers and columns announced
    from the record by ``extra_layers`` / ``adds_columns``); footprint WHOLE_SERIES / {} (it
    reads no pixels); ``file_identity`` = the file's path, size and mtime (and the CSV
    sidecar's), so a table re-exported to the same name is recomputed, not served stale."""
    modes = ctx.params.get("__modes__", {}) or {}
    onto = str(modes.get("onto", DEFAULT_ONTO) or DEFAULT_ONTO)
    path = TF.clean_path(ctx.params.get("path"))
    if not path:
        raise ValueError("Load Table: choose a table file — Browse… for a .parquet, .arrow "
                         "or .csv written by File → Export table…")
    if not os.path.isfile(path):
        raise ValueError(f"Load Table: no file at {path!r}")
    origin = TF.read_origin(path)
    tables = TF.read_tables(path, origin)
    if not tables:
        raise ValueError(f"Load Table: {os.path.basename(path)} holds no rows")
    if onto == "image":
        ds: Dataset = ctx.inputs[0]
        rec_axes = {a: int(v) for a, v in dict(origin.get("axes") or {}).items()}
        bad = [a for a in ("b", "m", "t", "z") if a in rec_axes
               and int(ds.axes.size(a)) != rec_axes[a]]
        if bad:
            raise ValueError(
                f"Load Table: the wired data does not match the table on "
                f"{', '.join(bad)} — data {tuple(int(ds.axes.size(a)) for a in bad)} against "
                f"the export's {tuple(rec_axes[a] for a in bad)}. Its rows would land on the "
                f"wrong positions or frames; wire the image the table was exported from (the "
                f"card's Load file button), or switch 'onto' to tables_only.")
        for t in tables:                       # a file with no record: check its indices
            if t.kind != "structure":
                continue
            for a in ("m", "t", "c"):
                if a in t.columns and a not in t.filled:
                    _index(t.columns[a], int(ds.axes.size(a)), a,
                           f"the {t.domain.value}" + (f" · {t.layer}" if t.layer else "")
                           + " table")
        axes, md = _envelope(ds.axes, ds.metadata, ctx.params, modes)
        out = replace(ds, metadata=md)
    else:
        axes, md = _envelope(AxisSizes(), {}, ctx.params, modes)
        out = Dataset(axes=axes, metadata=md)
    return _attach(out, tables, origin)


def _file_identity_load_table(params, modes):
    """``NodeSpec.file_identity`` — the table file (and a CSV's sidecar) by path, size and
    mtime: a re-export to the same name re-keys the memo."""
    p = _path(params)
    return TF.file_identity(p) if p else None


def _layers_load_table(params, modes):
    """``NodeSpec.extra_layers`` — the layers the chosen file holds, from its record (or a
    one-time read of a file without one): a structure table by its layer name, a lattice
    attribute by its column name. Total — a missing or unreadable file announces nothing."""
    try:
        sc = TF.scan(_path(params)) if _path(params) else {}
        out: List[Tuple[Domain, str]] = []
        for dom, layer, kind, cols in sc.get("tables") or ():
            if kind == "structure":
                if layer:
                    out.append((dom, layer))
            else:
                out.extend((dom, c) for c in cols if c not in _AXES)
        return tuple(dict.fromkeys(out))
    except Exception:  # noqa: BLE001 — an edit-time declaration must never raise
        return ()


def _columns_load_table(params, modes, incoming):
    """``NodeSpec.adds_columns`` — every structure table's columns, from the file's record,
    so the column menus downstream offer them before anything is pulled. Total."""
    from nodegraph.catalog._shared.columns import on_layer
    try:
        sc = TF.scan(_path(params)) if _path(params) else {}
        out: list = []
        for dom, layer, kind, cols in sc.get("tables") or ():
            if kind == "structure" and layer:
                out.extend(on_layer(dom, layer, list(cols)))
        return tuple(out)
    except Exception:  # noqa: BLE001 — an edit-time declaration must never raise
        return ()


register_node(
    _compute_load_table, op_key="io.load_table", label="Load Table", category="io",
    meta_transform=_meta_load_table,
    extra_layers=_layers_load_table,
    adds_columns=_columns_load_table,
    file_identity=_file_identity_load_table,
    reads_domains=frozenset(),
    adds_domains=frozenset({Domain.LABEL, Domain.POINT, Domain.TRACK, Domain.MESH}),
    inputs=[
        InDataset(available_in={"onto": frozenset({"image"})},
                  description=
                  "The image the table was exported from — the same file, through the same "
                  "registration and crop chain — so the Viewer draws the rows where they were "
                  "measured. The card's Load file button wires it for you (a Page Input "
                  "reading the file from the Image Input page). Refused when its positions, "
                  "frames or planes differ from the export's."),
        InString("path", "File", field=False, default="",
                 path_kind="open_file", path_filter=TF.TABLE_FILE_FILTER,
                 path_hint="a .parquet / .arrow / .csv from File → Export table…",
                 description=
                 "The table file to load: a Parquet, Arrow or CSV file written by File → "
                 "Export table… (any long-form file with `domain`/`layer` columns also "
                 "loads; one without them becomes a single Point table named after the "
                 "file). Its origin record — when, which graph and node, which source image "
                 "— is read with it and shown on the card. Re-exporting to the same name is "
                 "picked up on the next pull: the file's size and date are part of what is "
                 "cached."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("onto", ["tables_only", "image"], default=DEFAULT_ONTO, label="Onto",
                description=
                "What the loaded tables are attached to: nothing (a tables-only Dataset, the "
                "default) or the wired image (overlays, and nodes that also need pixels).",
                choice_docs={
                    "tables_only":
                        "No input needed: a Dataset with the export's axes and metadata "
                        "(calibration, position names) but no image, carrying only the "
                        "tables — for the Spreadsheet, plots and table nodes, or a table "
                        "whose image is not at hand.",
                    "image":
                        "Attach the tables to the wired Dataset — the image they were "
                        "exported from — so the Viewer draws points, tracks and labels over "
                        "it. A Dataset whose positions, frames or planes differ is refused "
                        "rather than mislabelled; Load file sets this and wires the image.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Load an exported table file (Parquet / Arrow / CSV) back onto the graph — "
                "its tables on their original layers, its origin (when, which graph and "
                "node, which source image) shown on the card — and, with Load file, onto "
                "the original image.")
