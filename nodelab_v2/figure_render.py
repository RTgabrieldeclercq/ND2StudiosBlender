"""Headless figure rendering (V5.00 step 6): a figure page to PNG / SVG / PDF, no window.

::

    python -m nodelab_v2.figure_render workspace.nd2graph.json --out fig.png [--page NAME]
                                       [--dpi 300] [--svg] [--bundle] [--write-job job.json]
    python -m nodelab_v2.figure_render --job job.json --out fig.png [--bundle]

A **figure job** is everything a render needs and nothing that lives only in a window: the
composed run graph (the figure page spliced with every page it reads, as
:func:`nodegraph.serialize.to_dict` writes it), the figure document, the map from its source
and data ids to run ids, and a seed per ``io.load`` node — a file path (ingested the way
the GUI and the LabLink worker ingest it) or the synthetic stack an empty path means in the
editor. :func:`build_job` makes one from a workspace file; the LabLink worker writes one
from the recipe it is running and invokes this module as a subprocess, so Qt never loads
in the worker process. :func:`render_job` runs the job under the ``offscreen`` platform
through the same :class:`~nodelab_v2.figure_editor.panel.FigureEditorPanel` the window
shows, driven by a host that answers from a headless engine — so what this writes is what
the editor exports.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

JOB_FORMAT = "nd2studios.figure-job/1"

#: the stack an ``io.load`` with no path is seeded with (the editor's synthetic source)
SYNTH_AXES = {"m": 1, "t": 3, "z": 4, "c": 2, "y": 256, "x": 256}
SYNTH_META = {"pixel_size_um": 0.325, "z_step_um": 1.0, "dt_s": 60.0,
              "channel_names": ["Ch0", "Ch1"], "channel_emission_nm": [520.0, 640.0]}


# ── jobs ───────────────────────────────────────────────────────────────────────────
def figure_page_id(ws: Any, page: str = "") -> str:
    """The figure page ``page`` names (an id or a name), else the first figure page."""
    pages = list(ws.pages.values())
    if page:
        for p in pages:
            if p.id == page or p.name == page:
                if p.figure is None:
                    raise ValueError(f"page {page!r} is not a figure page")
                return p.id
        raise ValueError(f"no page named {page!r}")
    for p in pages:
        if p.figure is not None:
            return p.id
    raise ValueError("the workspace has no figure page")


def build_job(ws: Any, page: str = "", *, seeds: Optional[Mapping[str, Mapping[str, Any]]]
              = None) -> Dict[str, Any]:
    """A figure job for ``page`` of workspace ``ws``: the composed graph, the figure, the
    run ids, a seed spec per loader (``{"path": …}`` from the node, overridable by
    ``seeds``; a blank path means the synthetic stack)."""
    from nodegraph.serialize import to_dict
    from nodelab_v2.ops import LOAD_OP, ensure_ops
    ensure_ops()
    pid = figure_page_id(ws, page)
    comp = ws.compose(pid)
    fig = ws.pages[pid].figure
    seed_specs: Dict[str, Dict[str, Any]] = {}
    for nid, rec in comp.graph.nodes.items():
        if rec.op_key != LOAD_OP:
            continue
        spec: Dict[str, Any] = {"path": str(rec.params.get("path") or "").strip()}
        env = comp.meta_seeds.get(nid)
        if env is not None:
            spec["metadata"] = _jsonable(dict(getattr(env, "metadata", None) or {}))
            axes = getattr(env, "axes", None)
            if axes is not None:
                spec["axes"] = {k: int(getattr(axes, k, 1) or 1) for k in ("m", "t", "z", "c", "y", "x")}
        if seeds and nid in seeds:
            spec.update(dict(seeds[nid]))
        seed_specs[nid] = spec
    labels = fig.source_labels(ws, pid)
    return {"format": JOB_FORMAT, "page": pid, "page_name": ws.pages[pid].name,
            "graph": to_dict(comp.graph), "figure": fig.to_dict(),
            "run_ids": fig.run_ids(comp, pid), "seeds": seed_specs, "labels": labels}


def _jsonable(d: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[str(k)] = v
        except (TypeError, ValueError):
            out[str(k)] = str(v)
    return out


def build_seeds(graph: Any, specs: Mapping[str, Mapping[str, Any]]
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(seeds, meta_seeds)`` for the job's loaders: a file is ingested
    (:func:`nodelab_v2.ingest.ingest_image`, into ``spec["store"]`` when given), a blank
    path is the synthetic stack; ``spec["metadata"]`` overrides what the file says (the
    sidecar's right), ``spec["axes"]`` sizes a synthetic stack."""
    from nodegraph.dataset import AxisSizes, Dataset
    from nodegraph.metadata import MetaEnvelope
    from nodegraph.provider import SyntheticProvider
    from nodelab_v2.ops import LOAD_OP
    seeds: Dict[str, Any] = {}
    meta_seeds: Dict[str, Any] = {}
    for nid, rec in graph.nodes.items():
        if rec.op_key != LOAD_OP:
            continue
        spec = dict(specs.get(nid) or {})
        path = str(spec.get("path") or rec.params.get("path") or "").strip()
        if path and os.path.exists(path):
            from nodelab_v2.ingest import ingest_image
            provider, envelope = ingest_image(path, spec.get("store"))
            metadata = dict(envelope.metadata)
            metadata.update(dict(spec.get("metadata") or {}))
            envelope = dataclasses.replace(envelope, metadata=dict(metadata))
        else:
            ax = dict(SYNTH_AXES)
            ax.update({k: int(v) for k, v in (spec.get("axes") or {}).items() if k in ax})
            axes = AxisSizes(**ax)
            provider = SyntheticProvider(axes, tile=128)
            metadata = dict(SYNTH_META)
            metadata.update(dict(spec.get("metadata") or {}))
            envelope = MetaEnvelope(axes=axes, metadata=dict(metadata))
        seeds[nid] = Dataset(axes=envelope.axes, metadata=metadata).with_image(provider)
        meta_seeds[nid] = envelope
    return seeds, meta_seeds


# ── the headless host ──────────────────────────────────────────────────────────────
class HeadlessHost:
    """The :class:`FigureEditorPanel` host protocol answered from a job and an engine —
    what ``_FigureHost`` answers from the window. A fetch pulls synchronously and hands
    the panel its payload at once."""

    def __init__(self, job: Mapping[str, Any], engine: Any) -> None:
        from nodelab_v2.figure_document import FigureDocument
        self.job = dict(job)
        self.engine = engine
        self.fig = FigureDocument(job["figure"])
        self.page = str(job.get("page") or "figure")
        self._run_ids: Dict[str, str] = {str(k): str(v)
                                         for k, v in (job.get("run_ids") or {}).items()}
        self.results: Dict[str, Any] = {}
        self.panel: Any = None
        self.messages: List[str] = []

    def figure_state(self, page_id: Optional[str]):
        return self.fig if page_id == self.page else None

    def page_name(self, page_id: str) -> str:
        return str(self.job.get("page_name") or page_id)

    def _env(self, run_id: str):
        try:
            return self.engine.env(run_id) if run_id in self.engine.graph.nodes else None
        except Exception:                  # noqa: BLE001 — an envelope the engine cannot derive
            return None

    def catalog(self, page_id: str) -> List[Dict[str, Any]]:
        labels = dict(self.job.get("labels") or {})
        out = []
        for s in self.fig.sources:
            rid = self._run_ids.get(str(s["id"]), "")
            out.append({"value": s["source"], "label": labels.get(str(s["id"])) or s.get("label")
                        or s["source"], "page": "", "name": str(s["source"]).split(":", 1)[-1],
                        "env": self._env(rid)})
        return out

    def envelopes(self, page_id: str) -> Dict[str, Any]:
        out = {}
        for sid, rid in self._run_ids.items():
            env = self._env(rid)
            if env is not None:
                out[sid] = env
        return out

    def run_ids(self, page_id: str) -> Dict[str, str]:
        return dict(self._run_ids)

    def source_labels(self, page_id: str) -> Dict[str, str]:
        labels = dict(self.job.get("labels") or {})
        return {str(s["id"]): str(labels.get(str(s["id"])) or s.get("label") or s["source"])
                for s in self.fig.sources}

    def finished(self, run_id: str):
        return self.results.get(run_id)

    def fetch(self, run_id: str) -> None:
        if run_id in self.results:
            return
        result = self.engine.pull(run_id)
        self.results[run_id] = result
        if self.panel is not None:
            self.panel.on_fetched(run_id, result)

    def commit(self, page_id: str, doc: Mapping[str, Any]) -> bool:
        return bool(self.fig.replace(doc))

    def viewer_state(self, run_id: str, names) -> Dict[int, Dict[str, Any]]:
        return {}

    def own_layers(self, run_id: str) -> list:
        return []

    def export_path(self, fmt: str) -> str:
        return ""

    def lineage(self, run_id: str) -> str:
        try:
            return str(self.engine.lineage(run_id)) if run_id in self.engine.graph.nodes else ""
        except Exception:                  # noqa: BLE001
            return ""

    def app_version(self) -> str:
        from nodelab_v2.version import PRODUCT, __version__
        return f"{PRODUCT} {__version__}"

    def status(self, text: str) -> None:
        self.messages.append(str(text))


# ── rendering ──────────────────────────────────────────────────────────────────────
def render_job(job: Mapping[str, Any], out: str, *, fmt: str = "", dpi: Optional[int] = None,
               svg: bool = False, bundle: bool = False) -> List[str]:
    """Render ``job`` to ``out`` (its extension picks the format unless ``fmt`` does);
    with ``svg`` an SVG lands beside it, with ``bundle`` the whole export bundle. Returns
    the paths written. Needs a Qt application: one is made on the offscreen platform
    when the process has none."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else "figure"])
    from nodegraph.memo import Memo
    from nodegraph.serialize import from_dict
    from nodelab_v2.figure_editor.fonts import ensure_page_fonts
    from nodelab_v2.figure_editor.panel import FigureEditorPanel
    from nodelab_v2.ops import ensure_ops, headless_engine
    ensure_ops()
    ensure_page_fonts()
    if job.get("format") != JOB_FORMAT:
        raise ValueError(f"not a figure job (format {JOB_FORMAT!r} expected)")
    graph, _zones, _groups = from_dict(job["graph"])
    seeds, meta_seeds = build_seeds(graph, job.get("seeds") or {})
    engine = headless_engine(graph, seeds=seeds, meta_seeds=meta_seeds, memo=Memo())
    host = HeadlessHost(job, engine)
    panel = FigureEditorPanel(host)
    host.panel = panel
    try:
        panel.bind(host.page)
        panel.compute_sources()
        app.processEvents()
        panel.render_all_now()
        fmt = (fmt or os.path.splitext(out)[1].lstrip(".") or "png").lower()
        if bundle:
            written = panel.export_bundle(out, fmt, dpi)
        else:
            written = [panel.export(out, fmt, dpi)]
            if svg and fmt != "svg":
                written.append(panel.export(os.path.splitext(out)[0] + ".svg", "svg", dpi))
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
    return written


def render_workspace(path: str, out: str, *, page: str = "", dpi: Optional[int] = None,
                     svg: bool = False, bundle: bool = False) -> List[str]:
    """Render the figure page of workspace file ``path``."""
    from nodelab_v2.workspace import Workspace
    ws = Workspace()
    ws.load_file(path)
    return render_job(build_job(ws, page), out, dpi=dpi, svg=svg, bundle=bundle)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nodelab_v2.figure_render",
                                 description="Render a figure page without a window.")
    ap.add_argument("workspace", nargs="?", help="a .nd2graph.json workspace file")
    ap.add_argument("--job", help="a figure job (what --write-job writes, or the LabLink worker)")
    ap.add_argument("--page", default="", help="the figure page's id or name (default: the first)")
    ap.add_argument("--out", help="the picture to write (.png / .tiff / .svg / .pdf)")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--svg", action="store_true", help="also write an SVG beside a raster")
    ap.add_argument("--bundle", action="store_true", help="write the export bundle beside it")
    ap.add_argument("--write-job", help="write the job JSON here (and render only if --out)")
    args = ap.parse_args(argv)
    if args.job:
        with open(args.job, encoding="utf8") as fh:
            job = json.load(fh)
    elif args.workspace:
        from nodelab_v2.workspace import Workspace
        ws = Workspace()
        ws.load_file(args.workspace)
        job = build_job(ws, args.page)
    else:
        ap.error("a workspace file or --job is needed")
        return 2
    if args.write_job:
        with open(args.write_job, "w", encoding="utf8") as fh:
            json.dump(job, fh, indent=1, sort_keys=True, ensure_ascii=False)
        print(f"job written: {args.write_job}")
    if not args.out:
        if not args.write_job:
            ap.error("--out is needed")
        return 0
    written = render_job(job, args.out, dpi=args.dpi, svg=args.svg, bundle=args.bundle)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
