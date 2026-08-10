"""Render what the bed pipeline LOOKS LIKE in ND2 Studios — offscreen, no display.

Qt lives in this file and nowhere else in the pipeline: `_bed_nodegraph.py` does the engine
work and must never import PySide6, because a heavy numeric run sharing a process with Qt is
a debugging trap. The two meet only through the `*.nd2graph.json` files, which is also what
keeps the pictures honest — the graph screenshotted here is the graph that produced the
numbers, loaded from the same file, not a decorative redraw.

Three shot types, all `QWidget.grab()` or `QGraphicsScene.render()`:
  canvas   the whole wired graph, `GraphView.fit_all()` then grab
  cards    one supersampled PNG per node, `scene.render()` over that NodeItem's rect
  panel    the InspectorPanel bound to one node (the properties form)

Four offscreen hazards, each of which silently ruins the output:
  1. `QT_QPA_PLATFORM=offscreen` must be set BEFORE PySide6 is imported.
  2. The offscreen QPA loads ZERO system fonts — register the Windows TTFs or every glyph
     renders as tofu.
  3. Qt crashes during offscreen teardown AFTER a successful render, so exit via `os._exit`.
  4. Without a metadata seed on the source node, every metadata-derived socket renders as
     the literal string "auto" (`node_item.py:707`) — which is exactly how you can tell a
     decorative screenshot from one of a graph that can actually run.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")     # before PySide6 (hazard 1)
os.environ.setdefault("NODELAB_GL", "0")                  # CPU surface; we shoot cards

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import Qt                                          # noqa: E402
from PySide6.QtGui import QFontDatabase, QImage, QPainter              # noqa: E402
from PySide6.QtWidgets import QApplication                             # noqa: E402

OUT = os.path.join(_ROOT, "CodeLog", "img", "nodes")
GRAPHS = os.path.join(_ROOT, "CodeLog", "graphs")
SUPERSAMPLE = 3

#: the source node is a headless stand-in — the engine takes its payload from `seeds[]`, so
#: the op needs no compute. A reader would wire `io.load`, so that is what gets drawn. Every
#: node downstream is byte-identical to the graph that ran.
SEED_SUBST = ("io.bedseed", "io.load")


def _fonts() -> int:
    n = 0
    for f in ("segoeui.ttf", "consola.ttf", "arial.ttf", "seguisb.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", f)
        if os.path.exists(p) and QFontDatabase.addApplicationFont(p) != -1:
            n += 1
    return n


def _seed_env():
    """A MetaEnvelope carrying THIS acquisition's real axes and calibration.

    Faithful on purpose: 21 positions x 15 timepoints x 7 z x 4 channels at 1.718 µm, so the
    cards show the numbers a reader would see with this file open. `dt_s` is the event-table
    measurement, not the file's own value (which is 4.0009x too large — see
    `_bed_nodegraph.true_dt_s`).
    """
    from nodegraph.dataset import AxisSizes
    from nodegraph.metadata import MetaEnvelope
    cal = {"pixel_size_um": 1.7182777601481225, "z_step_um": 40.0, "dt_s": 2837.2,
           "bit_depth": 12, "objective_na": 0.45, "objective_magnification": 10.0,
           "channel_emission_nm": [499.0, 571.0, 649.0, None]}
    return MetaEnvelope(axes=AxisSizes(m=21, t=15, z=7, c=4, y=1024, x=1024), metadata=cal)


def _topo(nodes, edges):
    """Node ids in DEPENDENCY order.

    `serialize.to_json` sorts nodes by id for clean diffs, so the file order is alphabetical
    and laying out in it puts the loader in the middle of the picture with the edges crossing
    over themselves. A reader follows a pipeline left to right, so the layout has to follow
    the data, not the filename sort.
    """
    ids = [n["id"] for n in nodes]
    preds = {i: set() for i in ids}
    for e in edges:
        if e["dst"] in preds and e["src"] in preds:
            preds[e["dst"]].add(e["src"])
    out, seen = [], set()
    while len(out) < len(ids):
        ready = [i for i in ids if i not in seen and preds[i] <= seen]
        if not ready:                                   # a cycle cannot happen here
            ready = [i for i in ids if i not in seen]
        for i in ready:
            out.append(i)
            seen.add(i)
    return out


def _layout(doc, order):
    """Lay the chain out in dependency order, wrapping so a 7-node graph is legible at page
    width rather than a 1750 px ribbon `fit_all` shrinks past readability."""
    per_row = 4
    for i, nid in enumerate(order):
        row, col = divmod(i, per_row)
        # serpentine: the second row runs right-to-left so the wrap edge is short
        c = col if row % 2 == 0 else (per_row - 1 - col)
        doc.set_pos(nid, 40 + c * 300.0, 40 + row * 330.0)


def _load(path):
    from nodelab_v2.document import GraphDocument
    d = json.load(open(path, encoding="utf-8"))
    swapped = set()
    for n in d.get("graph", {}).get("nodes", []):
        if n.get("op_key") == SEED_SUBST[0]:
            n["op_key"] = SEED_SUBST[1]
            n.setdefault("params", {})["path"] = "ChannelGFP,R-B,Nile Blue,TD_Seq0001.nd2"
            swapped.add(n["id"])
    # `io.load`'s Dataset output is named `image`, not `out` (`nodelab_v2/ops.py:82-97`), so
    # the substitution has to rename the edge's source socket too. Miss this and the very
    # first edge silently fails to draw — the scene has nothing to anchor it to — which is a
    # picture that misrepresents the graph rather than an obvious error.
    for e in d.get("graph", {}).get("edges", []):
        if e.get("src") in swapped and e.get("src_socket", "out") == "out":
            e["src_socket"] = "image"
    doc = GraphDocument()
    doc.load_dict(d)
    order = _topo(d["graph"]["nodes"], d["graph"]["edges"])
    _layout(doc, order)
    for nid in order:
        if doc.nodes[nid].op_key == SEED_SUBST[1]:
            doc.set_meta_seed(nid, _seed_env())
    return doc, order, d


def main() -> int:
    app = QApplication(sys.argv[:1])
    nf = _fonts()
    from nodelab_v2 import theme as T
    from nodelab_v2.ops import ensure_ops
    from nodelab_v2.scene import GraphScene, GraphView
    ensure_ops()                                   # registers io.load / io.dock / view.viewer
    os.makedirs(OUT, exist_ok=True)
    print(f"fonts registered: {nf}/4   theme: {T.MODE}")

    graphs = sorted(f for f in os.listdir(GRAPHS) if f.endswith(".nd2graph.json"))
    if not graphs:
        print(f"SKIP: no graphs in {GRAPHS} — run `_bed_nodegraph.py graphs` first")
        return 0

    seen_cards, manifest = set(), []
    for gfile in graphs:
        slug = gfile.split(".")[0]
        doc, order, _d = _load(os.path.join(GRAPHS, gfile))
        scene = GraphScene(doc)
        view = GraphView(scene)
        view.resize(1500, 720 if len(order) <= 4 else 1000)
        view.show()
        for _ in range(3):
            app.processEvents()
        view.fit_all()
        for _ in range(3):
            app.processEvents()
        try:
            view._max_btn.hide()
        except Exception:
            pass
        for _ in range(2):
            app.processEvents()
        cpath = os.path.join(OUT, f"canvas_{slug}.png")
        pix = view.grab()
        ok = pix.save(cpath)
        print(f"  canvas {slug:4s} {pix.width()}x{pix.height()} ok={ok}  {len(order)} nodes")
        manifest.append(dict(kind="canvas", slug=slug, path=os.path.relpath(cpath, _ROOT),
                             w=pix.width(), h=pix.height(), nodes=len(order)))

        # one card per DISTINCT op across all graphs — the document shows each node once
        for nid in order:
            op = doc.nodes[nid].op_key
            if op in seen_cards:
                continue
            seen_cards.add(op)
            item = scene.node_items.get(nid)
            if item is None:
                continue
            r = item.sceneBoundingRect().adjusted(-14, -14, 14, 14)
            img = QImage(int(r.width() * SUPERSAMPLE), int(r.height() * SUPERSAMPLE),
                         QImage.Format_ARGB32)
            img.fill(T.BG)                        # scene.render draws items only, no grid
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.TextAntialiasing, True)
            scene.render(p, img.rect(), r, Qt.KeepAspectRatio)
            p.end()
            safe = op.replace(".", "_")
            ipath = os.path.join(OUT, f"card_{safe}.png")
            okc = img.save(ipath)
            print(f"    card {op:32s} {img.width()}x{img.height()} ok={okc}")
            manifest.append(dict(kind="card", op=op, slug=slug, node=nid,
                                 path=os.path.relpath(ipath, _ROOT),
                                 w=img.width(), h=img.height()))

    # the inspector form for the node with the most parameters — analysis.segment
    from nodelab_v2.inspector import InspectorPanel
    doc, order, _d = _load(os.path.join(GRAPHS, "r2.nd2graph.json"))
    scene = GraphScene(doc)
    view = GraphView(scene)
    view.resize(900, 900)
    view.show()
    for _ in range(2):
        app.processEvents()
    target = next((n for n in order if doc.nodes[n].op_key == "analysis.segment"), order[-1])
    insp = InspectorPanel()
    insp.resize(376, 1100)
    insp.set_node(scene.node_items[target])
    insp.show()
    for _ in range(4):
        app.processEvents()
    ppath = os.path.join(OUT, "panel_analysis_segment.png")
    pix = insp.grab()
    print(f"  panel {doc.nodes[target].op_key} {pix.width()}x{pix.height()} "
          f"ok={pix.save(ppath)}")
    manifest.append(dict(kind="panel", op=doc.nodes[target].op_key,
                         path=os.path.relpath(ppath, _ROOT),
                         w=pix.width(), h=pix.height()))

    with open(os.path.join(OUT, "shots.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"\n{len(manifest)} images -> {OUT}")
    sys.stdout.flush()
    os._exit(0)                                   # hazard 3: teardown crashes after success


if __name__ == "__main__":
    main()
