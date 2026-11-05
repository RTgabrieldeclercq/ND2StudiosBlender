"""The Report (V5.00 step 7): the figure with its captions and legend as a page to read.

Qt-free. :func:`panel_captions` writes one caption per lettered panel from the document
and the sources' envelopes — what an image shows (source, frame, channels, overlays), what
a chart plots, what a table summarises — in reading order; :func:`report_html` lays the
picture, the captions, the legend paragraph (:mod:`nodegraph.figure.legend_text`) and the
provenance (:mod:`nodegraph.figure.bundle`) out as one self-contained HTML document the
editor writes beside the bundle and prints to PDF.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure import layout as L
from nodegraph.figure import schema as S


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:g}"


def _source_label(doc: Mapping[str, Any], sid: Any, labels: Mapping[str, str]) -> str:
    for s in doc.get("sources") or []:
        if s.get("id") == sid:
            return str(labels.get(str(sid)) or s.get("label") or s.get("source") or sid)
    return str(sid or "")


def _reads(doc: Mapping[str, Any], block: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    if block.get("data"):
        entry = next((d for d in doc.get("data") or [] if d.get("id") == block["data"]), None)
        if entry is not None:
            srcs = ", ".join(_source_label(doc, s, labels) for s in entry.get("from") or [])
            bits = [srcs or str(block["data"])]
            if entry.get("where"):
                bits.append("kept where " + " and ".join(
                    f"{c} {o} {_num(v)}" for c, o, v in entry["where"]))
            agg = entry.get("aggregate")
            if agg:
                gb = agg.get("group_by") or []
                gb = [gb] if isinstance(gb, str) else list(gb)
                bits.append(f"{agg.get('value') or 'value'} per " + (", ".join(gb) or "table")
                            + f" ({', '.join(agg.get('reducers') or [])})")
            return "; ".join(bits)
        return str(block["data"])
    return _source_label(doc, block.get("source"), labels)


def panel_captions(doc: Mapping[str, Any], envelopes: Mapping[str, Any] = (), *,
                   labels: Optional[Mapping[str, str]] = None) -> List[Tuple[str, str]]:
    """``[(letter, caption)]`` for every lettered panel of ``doc`` (a normalized figure
    document) in reading order. Written only from what the document says; a source's
    envelope adds its channel names."""
    labels = dict(labels or {})
    envs = dict(envelopes or {})
    items = {it["id"]: it for it in doc.get("items") or []}
    order = L.reading_order([it for it in items.values() if it.get("letter")])
    out: List[Tuple[str, str]] = []
    for iid in order:
        it = items[iid]
        kind = str(it.get("kind"))
        letter = str(it.get("letter"))
        if kind in ("image", "montage"):
            img = S.effective_image(doc, iid)
            sid = img.get("source")
            env = envs.get(sid)
            md = dict(getattr(env, "metadata", None) or {}) if env is not None else {}
            names = list(md.get("channel_names") or [])
            chans = []
            for ch in img.get("channels") or []:
                c = int(ch.get("c", 0))
                chans.append(str(ch.get("label") or (names[c] if c < len(names) and names[c]
                                                      else f"ch{c + 1}")))
            fr = img.get("frame") or {}
            where = []
            if kind == "montage":
                mt = it.get("montage") or {}
                axis = {"m": "positions", "t": "time points", "z": "planes",
                        "c": "channels"}.get(str(mt.get("tile_by", "m")), "tiles")
                where.append(f"tiles over {axis}")
            else:
                if fr.get("m") not in (None, "cursor", 0):
                    where.append(f"position {fr['m']}")
                if fr.get("t") not in (None, "cursor"):
                    where.append(f"frame {fr['t']}")
                if fr.get("z") == "max":
                    where.append("maximum projection")
                elif fr.get("z") == "mean":
                    where.append("mean projection")
                elif fr.get("z") not in (None, "cursor"):
                    where.append(f"plane {fr['z']}")
            region = (img.get("region") or {}).get("from", "full")
            if region != "full":
                where.append("a region of the field")
            overlays = []
            for Lr in img.get("layers") or []:
                lk = Lr.get("kind")
                if lk == "labels":
                    overlays.append("object outlines" if not (Lr.get("look") or {}).get("fill")
                                    else "object fills")
                elif lk == "field":
                    overlays.append(f"{Lr.get('column') or 'value'} as a colour map")
                elif lk in ("points", "tracks", "vectors", "mesh", "grid", "voxels"):
                    overlays.append({"points": "detections", "tracks": "tracks",
                                     "vectors": "displacement vectors", "mesh": "mesh sections",
                                     "grid": "the correlation grid",
                                     "voxels": "a voxel wash"}[lk])
            text = _source_label(doc, sid, labels)
            if chans:
                text += f" — {', '.join(chans)}"
            if where:
                text += f" ({', '.join(where)})"
            if overlays:
                text += f"; {', '.join(overlays)}"
            sb = (img.get("decor") or {}).get("scale_bar_um")
            if sb:
                text += f". Scale bar {_num(sb)} µm"
            out.append((letter, text + "."))
        elif kind == "chart":
            ch = it.get("chart") or {}
            ck = str(ch.get("kind") or "xy")
            what = {"xy": f"{ch.get('y') or ch.get('value') or 'y'} against {ch.get('x') or 'x'}",
                    "timeseries": f"{ch.get('y') or ch.get('value') or 'value'} over time",
                    "bar": f"{ch.get('y') or ch.get('value') or 'value'} per {ch.get('x') or 'group'}",
                    "distribution": f"the distribution of {ch.get('value') or ch.get('y') or 'value'}"
                                    + (f" ({ch.get('sub_kind')})" if ch.get("sub_kind") else ""),
                    "heatmap": f"{ch.get('value') or 'value'} as a heat map over "
                               f"{ch.get('x') or 'rows'} × {ch.get('group_by') or 'columns'}",
                    "hexbin": f"the density of {ch.get('y') or 'y'} against {ch.get('x') or 'x'}",
                    "msd": "mean squared displacement against lag",
                    "survival": "the fraction of tracks surviving over time",
                    "rose": f"the distribution of {ch.get('value') or 'angle'} as a rose"}.get(
                ck, ck)
            text = what[0].upper() + what[1:]
            if ch.get("group_by") and ck not in ("heatmap",):
                text += f", by {ch['group_by']}"
            fit = (ch.get("fit") or {}).get("model", "none")
            if fit and fit != "none":
                text += f", with a {fit} fit"
            test = (ch.get("stats") or {}).get("test", "none")
            if test and test != "none":
                text += f"; {test} test"
            text += f", from {_reads(doc, ch, labels)}"
            out.append((letter, text + "."))
        elif kind == "table":
            tb = it.get("table") or {}
            cols = list(tb.get("columns") or []) or ["every measured column"]
            stats = ", ".join(tb.get("stats") or ["n", "mean", "sem"])
            text = f"{', '.join(cols)} ({stats})"
            if tb.get("group_by"):
                text += f" per {tb['group_by']}"
            text = text[0].upper() + text[1:] + f", from {_reads(doc, tb, labels)}"
            out.append((letter, text + "."))
        else:
            out.append((letter, f"{kind}."))
    return out


_CSS = """
body { font-family: 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif; color: #111;
       max-width: 190mm; margin: 12mm auto; line-height: 1.45; font-size: 10.5pt; }
h1 { font-size: 15pt; margin: 0 0 4mm 0; }
h2 { font-size: 11.5pt; margin: 7mm 0 2mm 0; border-bottom: 1px solid #ccc; }
figure { margin: 0; }
figure img { width: 100%; height: auto; border: 1px solid #ddd; }
dl.captions dt { font-weight: 700; float: left; clear: left; width: 8mm; }
dl.captions dd { margin: 0 0 1.2mm 9mm; }
p.legend { text-align: justify; }
table.prov { border-collapse: collapse; font-size: 9pt; width: 100%; }
table.prov th, table.prov td { text-align: left; vertical-align: top; padding: 1mm 2mm;
                               border-bottom: 1px solid #e3e3e3; }
table.prov th { background: #f4f4f4; }
.meta { color: #666; font-size: 8.5pt; margin-top: 6mm; }
"""


def report_html(doc: Mapping[str, Any], *, title: str, picture: str, legend: str = "",
                captions: Sequence[Tuple[str, str]] = (), provenance: Optional[Mapping[str, Any]]
                = None, svg: str = "", data_files: Sequence[str] = ()) -> str:
    """One self-contained HTML page: the picture (``picture`` — a relative file name or a
    data URI), the panel captions, the legend paragraph, the provenance table and the
    links to the SVG and the data files."""
    e = html.escape
    parts: List[str] = [
        "<!DOCTYPE html>", "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>{e(title)}</title>", f"<style>{_CSS}</style></head><body>",
        f"<h1>{e(title)}</h1>",
        f"<figure><img src=\"{e(picture)}\" alt=\"{e(title)}\"></figure>"]
    if captions:
        parts.append("<h2>Panels</h2><dl class=\"captions\">")
        for letter, text in captions:
            parts.append(f"<dt>{e(letter)}</dt><dd>{e(text)}</dd>")
        parts.append("</dl>")
    if legend:
        parts.append(f"<h2>Legend</h2><p class=\"legend\">{e(legend)}</p>")
    if provenance:
        parts.append("<h2>Provenance</h2><table class=\"prov\"><tr><th>Source</th>"
                     "<th>Page output</th><th>File</th><th>Run id</th><th>Lineage</th></tr>")
        for s in provenance.get("sources") or []:
            files = "; ".join(f.get("name") or f.get("path") or "" for f in s.get("files") or [])
            parts.append(f"<tr><td>{e(str(s.get('label') or s.get('id')))}</td>"
                         f"<td>{e(str(s.get('source') or ''))}</td><td>{e(files)}</td>"
                         f"<td><code>{e(str(s.get('run_id') or ''))}</code></td>"
                         f"<td><code>{e(str(s.get('lineage') or '')[:16])}</code></td></tr>")
        parts.append("</table>")
        for d in provenance.get("data") or []:
            where = " and ".join(f"{c} {o} {_num(v)}" for c, o, v in d.get("where") or [])
            agg = d.get("aggregate") or {}
            parts.append(f"<p class=\"meta\">Data entry <code>{e(str(d['id']))}</code>: "
                         f"{e(', '.join(d.get('from') or []))}"
                         + (f", kept where {e(where)}" if where else "")
                         + (f", {e(str(agg.get('value')))} per "
                            f"{e(', '.join(agg.get('group_by') or []))} "
                            f"({e(', '.join(agg.get('reducers') or []))})" if agg else "")
                         + ".</p>")
    links = []
    if svg:
        links.append(f"<a href=\"{e(svg)}\">SVG</a>")
    for f in data_files:
        links.append(f"<a href=\"{e(f)}\">{e(f.split('/')[-1])}</a>")
    meta = []
    if provenance:
        if provenance.get("app_version"):
            meta.append(e(str(provenance["app_version"])))
        if provenance.get("written"):
            meta.append(e(str(provenance["written"])))
        if provenance.get("figure_digest"):
            meta.append("figure " + e(str(provenance["figure_digest"])[:12]))
    if links:
        meta.append("files: " + " · ".join(links))
    if meta:
        parts.append("<p class=\"meta\">" + " · ".join(meta) + "</p>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


__all__ = ["panel_captions", "report_html"]
