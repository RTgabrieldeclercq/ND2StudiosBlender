"""The export bundle (V5.00 step 5): what travels with a figure besides its picture.

Qt-free. :func:`bundle_files` composes, from a normalized figure document and what the
editor knows at export time, the files that let a reader re-draw and cite the figure:

* ``<stem>.figure.json`` — the document, canonical;
* ``<stem>.legend.txt`` — the generated legend paragraph (:mod:`nodegraph.figure.legend_text`);
* ``<stem>.provenance.json`` — each source as ``<page>:<name>`` with its run id, its
  session-stable lineage, the file it was read from (path, size, mtime), the display
  windows the images were drawn with, the app version and the figure digest;
* ``<stem>.data/<name>.csv`` — the values every chart, table and heat layer drew.

:func:`write_bundle` puts them beside the picture. The picture itself is the editor's
(it needs Qt painting); the bundle is the Qt-free rest.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure import legend_text as LT
from nodegraph.figure import schema as S


def _file_record(md: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The files a source was read from, per position, from its metadata (``source_path``
    absolute, ``source_file`` the name) with size and mtime when the path still exists."""
    paths = md.get("source_path") or []
    names = md.get("source_file") or []
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(names, str):
        names = [names]
    out: List[Dict[str, Any]] = []
    for k, p in enumerate(paths):
        rec: Dict[str, Any] = {"path": str(p) if p else "",
                               "name": str(names[k]) if k < len(names) and names[k] else
                               (os.path.basename(str(p)) if p else "")}
        try:
            st = os.stat(str(p))
            rec["size"] = int(st.st_size)
            rec["mtime"] = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        except (OSError, TypeError, ValueError):
            pass
        if rec["path"] or rec["name"]:
            out.append(rec)
    if not out:
        for n in names:
            if n:
                out.append({"path": "", "name": str(n)})
    return out


def _md_of(env: Any) -> Mapping[str, Any]:
    if env is None:
        return {}
    if isinstance(env, tuple) and len(env) == 2:
        return dict(env[1] or {})
    return dict(getattr(env, "metadata", None) or {})


def provenance(doc: Mapping[str, Any], envelopes: Mapping[str, Any], *,
               run_ids: Optional[Mapping[str, str]] = None,
               lineage: Optional[Mapping[str, str]] = None,
               lut_windows: Optional[Mapping[str, Sequence[Tuple[int, float, float]]]] = None,
               app_version: str = "", digest: str = "") -> Dict[str, Any]:
    """The provenance record: sources (id, ``<page>:<name>``, label, run id, lineage,
    files), data entries (what they pool, filter and summarise), the display windows
    (``{item id: [(c, lo, hi), …]}``), the app version, the figure digest, the time."""
    run_ids = run_ids or {}
    lineage = lineage or {}
    srcs = []
    for s in doc.get("sources") or []:
        sid = str(s["id"])
        md = _md_of(envelopes.get(sid))
        srcs.append({"id": sid, "source": s["source"], "label": s.get("label") or "",
                     "item": s.get("item"), "run_id": run_ids.get(sid, ""),
                     "lineage": lineage.get(sid, ""), "files": _file_record(md),
                     "pixel_size_um": md.get("pixel_size_um"), "dt_s": md.get("dt_s"),
                     "z_step_um": md.get("z_step_um"),
                     "channel_names": list(md.get("channel_names") or [])})
    data = []
    for d in doc.get("data") or []:
        data.append({"id": d["id"], "from": list(d.get("from") or []),
                     "domain": d.get("domain"), "table": d.get("table"),
                     "where": d.get("where") or [], "match": d.get("match"),
                     "aggregate": d.get("aggregate"), "run_id": run_ids.get(str(d["id"]), ""),
                     "lineage": lineage.get(str(d["id"]), "")})
    items = [{"id": it["id"], "kind": it["kind"], "letter": it.get("letter") or ""}
             for it in doc.get("items") or []]
    return {"schema": "nd2studios-figure-provenance/1",
            "figure_digest": digest or S.digest(doc), "app_version": app_version,
            "written": _dt.datetime.now().isoformat(timespec="seconds"),
            "page": dict(doc.get("page") or {}), "goal": doc.get("goal"),
            "sources": srcs, "data": data, "items": items,
            "lut_windows": {k: [[int(c), float(lo), float(hi)] for c, lo, hi in v]
                            for k, v in (lut_windows or {}).items()}}


def bundle_files(doc: Mapping[str, Any], envelopes: Mapping[str, Any], *,
                 data_files: Optional[Mapping[str, str]] = None,
                 run_ids: Optional[Mapping[str, str]] = None,
                 lineage: Optional[Mapping[str, str]] = None,
                 lut_windows: Optional[Mapping[str, Sequence[Tuple[int, float, float]]]] = None,
                 app_version: str = "") -> Dict[str, str]:
    """``{relative path: text}`` of the bundle: ``figure.json``, ``legend.txt``,
    ``provenance.json`` and ``data/<name>.csv`` per entry of ``data_files`` (the CSV of
    every chart, table and heat layer, keyed by a file-safe name)."""
    nd = S.normalize(doc)
    out: Dict[str, str] = {"figure.json": S.canonical_json(nd) + "\n"}
    legend = LT.legend_sentence(nd, envelopes, lut_windows=lut_windows)
    out["legend.txt"] = (legend + "\n") if legend else ""
    out["provenance.json"] = json.dumps(
        provenance(nd, envelopes, run_ids=run_ids, lineage=lineage, lut_windows=lut_windows,
                   app_version=app_version, digest=S.digest(nd)),
        indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    for name, text in (data_files or {}).items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name)) or "data"
        if not safe.lower().endswith(".csv"):
            safe += ".csv"
        out[f"data/{safe}"] = text
    return out


def write_bundle(stem: str, files: Mapping[str, str]) -> List[str]:
    """Write ``files`` (from :func:`bundle_files`) beside the picture ``<stem>.<ext>``:
    ``<stem>.figure.json``, ``<stem>.legend.txt``, ``<stem>.provenance.json`` and
    ``<stem>.data/…``. Returns the paths written. An empty text writes nothing."""
    written: List[str] = []
    for rel, text in files.items():
        if not text:
            continue
        if rel.startswith("data/"):
            path = os.path.join(f"{stem}.data", rel[len("data/"):])
        else:
            path = f"{stem}.{rel}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf8", newline="") as f:
            f.write(text)
        written.append(path)
    return written


__all__ = ["provenance", "bundle_files", "write_bundle"]
