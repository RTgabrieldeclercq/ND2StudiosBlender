"""Load Analysis (``io.load_analysis``) — a saved DIC / DVC / PIV / SerialTrack trial back onto
the graph, without recomputing it (2026-10-09)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import StructureTable

from nodegraph.catalog._base import register_node


def _load_analysis_trial_path(trial: str, ds: Optional[Dataset]) -> str:
    """``trial`` as a folder: an existing path as given, otherwise a trial id looked up in
    the ``<stem>.analysis`` folder beside each of the incoming data's source files, and in
    each file's spare folder (``analysis_store.spare_root``)."""
    from nodegraph.analysis_store import find_trial, trial_roots
    from nodegraph.metadata import SOURCE_PATH_KEY
    t = trial.strip().strip('"').strip("'")
    if os.path.isdir(t) or os.path.isfile(t):
        return t
    sources = [str(p) for p in (ds.metadata.get(SOURCE_PATH_KEY) or [])
               if p] if ds is not None else []
    found = find_trial(sources, t)
    if found:
        return found
    roots: List[str] = []
    for p in sources:
        for r in trial_roots(p):
            if r not in roots:
                roots.append(r)
    where = (" or in " + ", ".join(roots)) if roots else (
        " — and the input carries no source file to look beside (wire the image this trial "
        "was computed on, or give the trial folder's full path)")
    raise ValueError(f"Load Analysis: no saved trial {t!r} as a path{where}")


def _compute_load_analysis(ctx: EvalContext) -> Dataset:
    """A saved analysis trial (:mod:`nodegraph.analysis_store`), read back onto the graph.

    Every analysis node that saves (DIC, DVC, PIV, SerialTrack and their staged cards) writes
    a trial folder beside its source file, one Parquet file per finished unit. This node
    reads one back: its result tables land on their ORIGINAL layers with their original
    columns, and its metadata stamps (``dic_*`` / ``dvc_*`` / ``piv_*`` / ``track_*``) ride
    along, so the Viewer's overlays and every downstream node — strain, Accumulate DVC
    Field, Rasterize Field, Track Field — behave as if the analysis had just run. A trial
    still being written (or one a crash cut short) loads what it holds; the
    ``analysis_trial_units`` stamp says how much.

    ``onto = image`` attaches the tables to the wired Dataset — normally the image the trial
    was computed on, so the overlays sit on it — and refuses one whose positions, frames or
    planes do not match the trial's (rows would land on the wrong frame). ``tables_only``
    needs no wire: a Dataset with the trial's axes, its calibration and no image, for the
    Spreadsheet or table nodes.

    Resolved spec: category io; op ``io.load_analysis``; reads nothing from the image; adds
    POINT / TRACK (whatever the trial holds — not predictable before the file is read, so no
    ``layer_out``); footprint WHOLE_SERIES / {} (it reads no pixels)."""
    from nodegraph.analysis_store import read_trial
    modes = ctx.params.get("__modes__", {}) or {}
    onto = modes.get("onto", "image")
    trial = str(ctx.params.get("trial", "") or "").strip()
    ds: Optional[Dataset] = ctx.inputs[0] if (onto == "image" and ctx.inputs) else None
    if not trial:
        raise ValueError("Load Analysis: choose a saved trial — the list shows the trials "
                         "beside the wired image's source file, or type a trial folder's path")
    path = _load_analysis_trial_path(trial, ds)
    td = read_trial(path)
    hdr = td.header
    hax = {k: int(v) for k, v in (hdr.get("axes") or {}).items()}
    if ds is not None:
        ax = ds.axes
        bad = [a for a in ("m", "t", "z") if a in hax and int(getattr(ax, a)) != hax[a]]
        if bad:
            raise ValueError(
                f"Load Analysis: the wired data does not match the trial on "
                f"{', '.join(bad)} — data {tuple(int(getattr(ax, a)) for a in bad)} against the "
                f"trial's {tuple(hax[a] for a in bad)}. Its rows would land on the wrong "
                f"positions or frames; wire the image the trial was computed on, or switch "
                f"'onto' to tables_only.")
        out = ds
    else:
        out = Dataset(axes=AxisSizes(**{a: max(1, hax.get(a, 1))
                                        for a in ("m", "t", "z", "c", "y", "x")}),
                      metadata=dict(hdr.get("calibration") or {}))
    placement: Dict[str, Dict[str, Any]] = hdr.get("tables") or {}
    for kind, parts in td.tables.items():
        if placement and kind not in placement:
            continue                    # a per-unit working table (SerialTrack's links)
        where = placement.get(kind) or {"domain": "point", "layer": kind,
                                        "z_kind": "plane_index"}
        keys = [k for k in parts[0] if all(k in p for p in parts)]
        cols = {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}
        if not cols:
            continue
        if "id" not in cols:
            cols["id"] = np.arange(len(next(iter(cols.values()))), dtype=np.int64)
        out = out.with_structure(StructureTable(
            Domain(str(where.get("domain", "point"))), cols,
            layer=str(where.get("layer", kind)),
            z_kind=str(where.get("z_kind", "plane_index"))))
        if where.get("mesh_of"):                         # a correlation grid: the Viewer
            from nodegraph.catalog._shared.analysis_mesh import MESH_MARK   # draws it
            marks = dict(out.metadata.get(MESH_MARK) or {})
            marks[str(where.get("layer", kind))] = {"field": str(where["mesh_of"]),
                                                    "kind": str(where.get("mesh_kind", ""))}
            if where.get("mesh_window"):
                marks[str(where.get("layer", kind))]["window"] = list(where["mesh_window"])
            out = out.with_metadata(**{MESH_MARK: marks})
    stamps = dict(hdr.get("stamps") or {})
    stamps.update({
        "analysis_trial": td.path,
        "analysis_trial_note": f"loaded {hdr.get('analysis', hdr.get('op', ''))} trial "
                               f"{os.path.basename(td.path)}",
        "analysis_trial_units": f"{len(td.units)}/{int(hdr.get('units_total') or 0)} saved",
    })
    return out.with_metadata(**stamps)


def _columns_load_analysis(params, modes, incoming):
    """The columns the chosen trial's tables carry, read from its ``trial.json`` when the
    trial is given as a folder path (a bare trial id resolves against the incoming
    source, which this edit-time hook cannot see — then nothing is declared until the
    path form is used). Total: any problem declares nothing."""
    from nodegraph.catalog._shared.columns import on_layer
    try:
        import json
        t = str((params or {}).get("trial", "") or "").strip().strip('"').strip("'")
        hdr_path = t if t.lower().endswith("trial.json") else os.path.join(t, "trial.json")
        if not t or not os.path.isfile(hdr_path):
            return ()
        with open(hdr_path, "r", encoding="utf-8") as fh:
            hdr = json.load(fh)
        out: list = []
        for _kind, where in (hdr.get("tables") or {}).items():
            cols = where.get("columns") or []
            if cols:
                out.extend(on_layer(Domain(str(where.get("domain", "point"))),
                                    str(where.get("layer", _kind)), list(cols)))
        return tuple(out)
    except Exception:  # noqa: BLE001 — an edit-time declaration must never raise
        return ()


register_node(
    _compute_load_analysis, op_key="io.load_analysis", label="Load Analysis", category="io",
    adds_columns=_columns_load_analysis,
    reads_domains=frozenset(), adds_domains=frozenset({Domain.POINT, Domain.TRACK}),
    inputs=[
        InDataset(available_in={"onto": frozenset({"image"})},
                  description=
                  "The image the trial was computed on (the same ND2, the same registration "
                  "chain) — its tables are attached to it, so overlays line up. Its source file "
                  "is also where the trial list is looked up."),
        InString("trial", "Trial", field=False, default="",
                 options_from="analysis_trials",
                 description=
                 "Which saved trial to load: the list shows every trial in the "
                 "`<file>.analysis` folder beside the wired image's source, and in its spare "
                 "folder under `~/.nd2studios/analysis/` (where units go when the source's "
                 "folder refuses them) — analysis, label, how many units are saved, complete "
                 "or partial. Or type a trial folder's full path. Switching entries switches "
                 "results without recomputing anything."),
    ],
    outputs=[OutDataset()],
    modes=[Mode("onto", ["image", "tables_only"], default="image", label="Onto",
                description=
                "What the loaded tables are attached to: the wired image (overlays, and "
                "downstream nodes that also need pixels) or nothing (a tables-only Dataset).",
                choice_docs={
                    "image":
                        "Attach the trial's tables to the wired Dataset — the image it was "
                        "computed on — so the Viewer draws the grid and vectors over it. A "
                        "Dataset whose positions, frames or planes differ is refused rather "
                        "than mislabelled.",
                    "tables_only":
                        "No input needed: a Dataset with the trial's axes and calibration but "
                        "no image, carrying only the tables — for the Spreadsheet, plots and "
                        "table nodes, or a trial whose image is not at hand.",
                })],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    description="Load a saved DIC/DVC/PIV/SerialTrack trial (the `<file>.analysis` folder "
                "beside the image) back onto the graph — its tables on their original layers "
                "and its provenance — without recomputing.")
