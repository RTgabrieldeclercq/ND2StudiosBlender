"""Offscreen end-to-end probe of the Phase-5 GUI slice (G1/G2/G4/G6/G7/G8/G10).

    PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py [out.png]

Boots the real MainWindow offscreen and drives the same code paths the mouse would:
document wiring rules (validity, cycle rejection, non-multi replace), link-search
compatibility, save/load round-trip incl. canvas positions, a REAL EngineRunner pull
on the synthetic source through to viewer pixels, the H11 lever guard, mute
pass-through, and an inspector param edit re-seeding a downstream ƒmd pill (G8).
Asserts + printed checkmarks; exits via os._exit (offscreen teardown crash gotcha).
"""
from __future__ import annotations

import json
import os
import subprocess
import pathlib
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPointF            # noqa: E402
from PySide6.QtGui import QFontDatabase       # noqa: E402
from PySide6.QtWidgets import QApplication    # noqa: E402


def _load_fonts() -> None:
    for name in ("segoeui.ttf", "consola.ttf", "arial.ttf", "seguisb.ttf"):
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


# ── synthetic graphics-scene mouse events (P2) ───────────────────────────────
# `QGraphicsSceneMouseEvent` cannot be constructed from Python, so the card's handlers are
# driven with the small duck-typed surface they actually use. That is the whole contract
# between NodeItem and Qt for these three handlers, so exercising it is a real test of the
# handlers rather than of Qt.

class _FakePress:
    def __init__(self, pos, button=None):
        from PySide6.QtCore import Qt as _Qt
        self._pos, self._btn = pos, button or _Qt.LeftButton

    def pos(self):
        return self._pos

    def button(self):
        return self._btn

    def accept(self):
        pass


class _FakeMove:
    def __init__(self, x, y=0.0, mods=None):
        from PySide6.QtCore import Qt as _Qt
        self._x, self._y = x, y
        self._mods = _Qt.NoModifier if mods is None else mods

    def pos(self):
        from PySide6.QtCore import QPointF as _QPF
        return _QPF(self._x, self._y)

    def modifiers(self):
        return self._mods

    def accept(self):
        pass


class _FakeRelease:
    def button(self):
        from PySide6.QtCore import Qt as _Qt
        return _Qt.LeftButton

    def accept(self):
        pass


def main(argv) -> int:
    out = argv[1] if len(argv) > 1 else "nodelab_v2_phase5.png"
    app = QApplication(sys.argv[:1])
    _load_fonts()

    from nodegraph.dataset import AxisSizes
    from nodegraph.metadata import MetaEnvelope
    from nodelab_v2 import theme as T
    from nodelab_v2.scene import compatible_ops
    from nodelab_v2.window import MainWindow

    win = MainWindow()
    win.resize(1500, 880)
    win.show()
    _seen_fail = []
    win.runner.failed.connect(
        lambda nid, tr: _seen_fail.append((nid, tr.strip().splitlines()[-1])))
    app.processEvents()

    # ── launch state: a blank canvas with the welcome card (no demo graph) ─────
    assert not win.doc.nodes, "the app must open on an empty canvas"
    assert win.welcome.isVisible() and win.welcome.parent() is win.view
    wg, vw, vh = win.welcome.geometry(), win.view.width(), win.view.height()
    assert abs(wg.center().x() - vw / 2) <= 2 and abs(wg.center().y() - vh / 2) <= 2
    assert abs(win.view.transform().m11() - 1.0) < 1e-9   # 1:1, not zoomed into nothing
    win.welcome.op_dropped.emit("enhance.gamma", QPointF(40.0, 40.0))   # drop passthrough
    app.processEvents()
    assert len(win.doc.nodes) == 1 and not win.welcome.isVisible()
    win.file_new()
    app.processEvents()
    assert not win.doc.nodes and win.welcome.isVisible()   # File → New re-welcomes
    _ok("Launch: blank canvas + centred welcome card (drop passthrough places a node)")

    win.build_demo()                       # everything below drives the example graph
    app.processEvents()
    assert len(win.doc.nodes) == 8 and not win.welcome.isVisible()
    doc = win.doc

    # ── G1: wiring rules through the document (what socket drags execute) ──────
    ok, _ = doc.can_connect("n2", "out", "n4", "data")
    assert ok, "dataset→dataset should connect"
    ok, why = doc.can_connect("n4", "out", "n2", "data")
    assert not ok and "cycle" in why, f"cycle must be rejected (got {why!r})"
    ok, _ = doc.can_connect("n2", "out", "n2", "data")
    assert not ok, "self-loop must be rejected"
    # non-multi replace: n4.data currently fed by n3 — reconnect from n2 replaces it
    before = doc.edge_into("n4", "data")
    assert before == ("n3", "out", "n4", "data")
    removed = doc.connect("n2", "out", "n4", "data")
    assert removed == [("n3", "out", "n4", "data")]
    assert doc.edge_into("n4", "data") == ("n2", "out", "n4", "data")
    doc.connect("n3", "out", "n4", "data")          # restore the demo chain
    _ok("G1: validity, cycle/self-loop rejection, non-multi replace")

    # link-drag search offers compatible ops for a Dataset output
    n2_out = win.scene.node_items["n2"].socket("out", "out")
    entries = compatible_ops(n2_out.spec, "out")
    labels = {spec.op_key for spec, _s in entries}
    assert "enhance.gaussian" in labels and "view.viewer" in labels
    assert not any(op.startswith(("zone.", "group.", "test.")) for op in labels)
    _ok(f"G1: link-drag search offers {len(entries)} compatible ops (fixtures hidden)")

    # G1 (crash regression 2026-07-27): DROPPING a wire on a socket — the real mouse
    # path, which no probe covered. `_end_wire` tears the drag down (`_cancel_temp`
    # clears `_drag_fixed`) BEFORE resolving the drop, so the anchor must be passed
    # explicitly; reading it back from state raised AttributeError on every connect.
    from PySide6.QtCore import QPoint as _QPoint
    from nodelab_v2.document import GraphDocument as _GD0
    from nodelab_v2.scene import GraphScene as _GS0
    wdoc = _GD0()
    wsc = _GS0(wdoc)
    wdoc.add_node("io.load", node_id="wa", x=0, y=0)
    wdoc.add_node("enhance.gamma", node_id="wb", x=300, y=0)
    wdoc.add_node("enhance.gamma", node_id="wc", x=600, y=0)
    a_out = wsc.node_items["wa"].socket("out", "image")
    b_in = wsc.node_items["wb"].socket("in", "data")
    wsc.begin_wire(a_out, a_out.anchor())
    assert wsc._drag_fixed is a_out and wsc._temp_wire is not None
    wsc._update_temp(b_in.anchor())                       # hover highlights the target
    assert b_in.highlight is True
    wsc._end_wire(b_in.anchor(), _QPoint(0, 0))           # ← used to raise
    assert ("wa", "image", "wb", "data") in wdoc.edges, wdoc.edges
    assert wsc._drag_fixed is None and wsc._temp_wire is None and b_in.highlight is None
    # dragging FROM a connected non-multi input detaches and re-drags from the source,
    # so dropping it on another node's input moves the wire there
    c_in = wsc.node_items["wc"].socket("in", "data")
    wsc.begin_wire(b_in, b_in.anchor())
    assert wsc._drag_fixed is a_out                       # re-anchored to the source end
    wsc._end_wire(c_in.anchor(), _QPoint(0, 0))
    assert ("wa", "image", "wc", "data") in wdoc.edges
    assert ("wa", "image", "wb", "data") not in wdoc.edges
    # an invalid drop (self-loop) connects nothing and still leaves no dangling drag
    wsc.begin_wire(a_out, a_out.anchor())
    wsc._end_wire(wsc.node_items["wa"].socket("out", "image").anchor(), _QPoint(0, 0))
    assert wsc._drag_fixed is None and len(wdoc.edges) == 1
    _ok("G1: wire DROP on a socket connects (+ detach/re-drag, invalid drop is a no-op)")

    # ── G8: live derive re-seed — the ƒmd pill follows the propagated envelope ──
    g_item = win.scene.node_items["n3"]                    # gaussian (σ derive-less
    d_item = win.scene.node_items["n7"]                    # deconvolve: na derives
    seed_env = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512),
                            metadata={"pixel_size_um": 0.1, "z_step_um": 0.3,
                                      "objective_na": 1.4,
                                      "channel_emission_nm": [520.0, 640.0]})
    doc.set_meta_seed("n1", seed_env)
    na_sock = d_item.spec.input("na")
    assert d_item.resolved(na_sock) == 1.4, d_item.resolved(na_sock)
    doc.set_meta_seed("n1", MetaEnvelope(axes=seed_env.axes,
                                         metadata={**seed_env.metadata,
                                                   "objective_na": 0.45}))
    assert d_item.resolved(na_sock) == 0.45      # pill re-seeded live (G8)
    # the INSPECTOR's auto boxes re-seed too (n7 is the selected node at startup)
    boxes = {s.name: b for _n, s, b in win.inspector._auto_boxes}
    assert "na" in boxes and abs(boxes["na"].value() - 0.45) < 1e-9, \
        {k: b.value() for k, b in boxes.items()}
    _ok("G8: ƒmd pill + inspector auto box re-seed from the envelope (1.4 → 0.45)")

    # ── H11 lever guard: z==1 disables 3D and flags a locked-3D node invalid ────
    assert not g_item.z_is_one() and g_item._switch.allow_3d
    doc.set_meta_seed("n1", MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=2,
                                                        y=512, x=512),
                                         metadata=dict(seed_env.metadata)))
    assert g_item.z_is_one() and not g_item._switch.allow_3d
    assert g_item.dim == "3D" and g_item.dim_invalid()      # demo starts 3D → red badge
    doc.set_meta_seed("n1", seed_env)                        # restore z=5
    assert g_item._switch.allow_3d and not g_item.dim_invalid()
    _ok("H11: z==1 greys the 3D lever + red-badges a locked-3D node; unknown ≠ 1")

    # ── mode-gated params: the CARD and the INSPECTOR both follow the live mode ──
    # n4 is analysis.threshold, whose `threshold` socket is `fixed`-only (a histogram
    # method derives its own cut). Changing the mode through the real inspector combo
    # must drop the socket row from the card AND rebuild the form — the inspector used
    # to keep painting the previous method's params (only `refresh_derived` ran).
    from PySide6.QtWidgets import QComboBox as _QCombo, QLabel as _QLabel
    t_item = win.scene.node_items["n4"]
    win.scene.clearSelection()
    t_item.setSelected(True)
    app.processEvents()
    assert win.inspector._node is t_item

    def _insp_params() -> set:
        return {w.text() for w in win.inspector.findChildren(_QLabel)}

    def _card_ins(item) -> set:
        return {k[1] for k in item._sockets if k[0] == "in"}

    m_combo = next(cb for cb in win.inspector.findChildren(_QCombo)
                   if "otsu" in [cb.itemText(i) for i in range(cb.count())])
    assert t_item.state()["method"] == "fixed"
    assert "threshold" in _card_ins(t_item) and "threshold" in _insp_params()
    m_combo.setCurrentText("otsu")            # the real signal path (currentTextChanged)
    app.processEvents()                       # the rebuild is deferred by one turn
    assert t_item.state()["method"] == "otsu"
    assert "threshold" not in _card_ins(t_item), "the card must drop the dead socket"
    assert "threshold" not in _insp_params(), "the inspector must rebuild off the mode"
    # …and the combo is still live after the rebuild that replaced it
    m_combo = next(cb for cb in win.inspector.findChildren(_QCombo)
                   if "otsu" in [cb.itemText(i) for i in range(cb.count())])
    m_combo.setCurrentText("fixed")
    app.processEvents()
    assert "threshold" in _card_ins(t_item) and "threshold" in _insp_params()
    _ok("mode gate: changing `method` re-resolves the active sockets on the card AND "
        "rebuilds the inspector form (analysis.threshold: fixed-only `threshold`)")

    # ── V2.21: a dropdown explains EVERY OPTION, on the real widgets ─────────────
    # `nodegraph.selftest::test_option_docs` holds the prose and the builders; this holds
    # the wiring, which is where it can silently not arrive. Three separate carriers, each
    # of which has its own way of failing: the ROW tooltip (a Mode row had none at all
    # before this), the per-ITEM Qt.ToolTipRole data (the only surface the user sees while
    # arrowing down the open list), and the node card's popup QMenu — which SWALLOWS action
    # tooltips unless `setToolTipsVisible(True)` is set, so the prose can be written,
    # correct, and invisible.
    from PySide6.QtCore import Qt as _Qt

    m_combo = next(cb for cb in win.inspector.findChildren(_QCombo)
                   if "otsu" in [cb.itemText(i) for i in range(cb.count())])
    _row_tip = m_combo.parent().toolTip()
    assert "method — mode · 6 options" in _row_tip, _row_tip[:200]
    for _opt in ("fixed", "otsu", "li", "yen", "triangle", "mean"):
        assert f"• {_opt} —" in _row_tip, f"{_opt} missing from the Mode row hover"
        _i = m_combo.findText(_opt)
        _item_tip = m_combo.itemData(_i, _Qt.ToolTipRole)
        assert _item_tip and f"<b>{_opt}</b>" in _item_tip, \
            f"combo item {_opt!r} carries no tooltip of its own"
    # the label a user actually aims at carries it too, not just the row
    _lab = next(w for w in m_combo.parent().findChildren(_QLabel) if w.text() == "method")
    assert _lab.toolTip() == _row_tip

    # the card's popup menu, built by the real `_open_menu` on a real mode pill.
    # `_open_menu` resolves `QMenu` from its own module globals, so swapping THAT name for a
    # subclass whose `exec` inspects and returns None runs the real builder without ever
    # entering a modal loop. (Assigning to `QMenu.exec` on the class does not take in
    # PySide6 — the real modal loop opens and the probe hangs with no user to dismiss it.)
    import nodelab_v2.node_item as _NI
    from PySide6.QtWidgets import QMenu as _QMenu
    _mode_ctl = next(c for c in t_item.controls()
                     if c.kind == "mode" and c.obj.name == "method")
    _menus: list = []

    class _PeekMenu(_QMenu):
        def exec(self, *a, **k):               # look, then dismiss (None = no pick)
            _menus.append([(act.text(), act.toolTip()) for act in self.actions()]
                          + [("__visible__", str(self.toolTipsVisible()))])
            return None

    _NI.QMenu = _PeekMenu
    try:
        t_item._open_menu(_mode_ctl, list(_mode_ctl.obj.choices), "fixed")
    finally:
        _NI.QMenu = _QMenu
    assert _menus, "the mode pill did not open a menu"
    assert t_item.state()["method"] == "fixed", "dismissing the menu must change nothing"
    _entries = dict(_menus[-1])
    assert _entries["__visible__"] == "True", \
        "QMenu swallows action tooltips unless setToolTipsVisible(True) — the option " \
        "prose would be written and never shown"
    for _opt in ("otsu", "li"):
        assert f"<b>{_opt}</b>" in _entries[_opt], f"menu action {_opt!r} has no tooltip"

    # the 2D/3D lever hovers like the Mode it is — it used to say only the z==1 refusal
    _lever = win.scene.node_items["n3"]._switch          # enhance.gaussian bears one
    assert _lever is not None, "enhance.gaussian must carry the 2D/3D lever"
    _lever._refresh_tip()
    assert "2D / 3D lever" in _lever.toolTip() and "• 3D —" in _lever.toolTip(), \
        _lever.toolTip()[:200]

    # a vocab TICK LIST has no popup to hang tips off, so each box carries its own
    from PySide6.QtWidgets import QCheckBox as _QCB
    win.scene.clearSelection(); win.scene.node_items["n6"].setSelected(True)
    app.processEvents()                                  # analysis.measure: stats + shape
    _cbs = {cb.text(): cb.toolTip() for cb in win.inspector.findChildren(_QCB)}
    for _tok in ("mean", "count", "solidity"):
        assert f"<b>{_tok}</b>" in _cbs.get(_tok, ""), \
            f"vocab tick box {_tok!r} carries no explanation: {_cbs.get(_tok)!r}"
    win.scene.clearSelection(); t_item.setSelected(True)
    app.processEvents()
    _ok("V2.21 option docs reach the widgets: the Mode row + its label carry the "
        "6-option block, every combo item carries its own Qt.ToolTipRole prose, the "
        "card's popup menu carries it with tooltips VISIBLE, the 2D/3D switch hovers "
        "as a documented lever, and each vocab tick box explains its own token")

    # ── V2.15: a path socket is BROWSABLE — every socket declaring `path_kind` gets a
    # Browse… button that opens the right dialog and commits the pick. The inspector used
    # to match the literal socket name "path", so `analysis.segment`'s two model paths had
    # no browser at all and demanded a hand-typed absolute path.
    from PySide6.QtWidgets import (QFileDialog as _QFD, QLineEdit as _QLE,
                                   QToolButton as _QTB)

    def _browse_btns():
        return [b for b in win.inspector.findChildren(_QTB) if b.text() == "Browse…"]

    picked = []
    _QFD.getOpenFileName = staticmethod(
        lambda parent, cap, start, flt: (picked.append(("file", cap, start, flt))
                                         or (r"C:\data\movie.nd2", "")))
    _QFD.getExistingDirectory = staticmethod(
        lambda parent, cap, start, opts: (picked.append(("dir", cap, start, ""))
                                          or r"C:\models\sd"))

    src_item = win.scene.node_items["n1"]                   # io.load
    win.scene.clearSelection(); src_item.setSelected(True)
    app.processEvents()
    assert len(_browse_btns()) == 1, "io.load's `path` must have one Browse… button"
    row = _browse_btns()[0].parent()
    assert row.findChildren(_QLE)[0].placeholderText().startswith("empty = synthetic"), \
        "the placeholder must say what an EMPTY path means"
    _browse_btns()[0].click()
    app.processEvents()
    assert picked[-1][0] == "file" and "*.nd2" in picked[-1][3], picked[-1]
    assert doc.nodes["n1"].params["path"] == r"C:\data\movie.nd2", doc.nodes["n1"].params
    assert win.inspector._last_dir == r"C:\data"            # seeds the NEXT dialog

    seg = doc.add_node("analysis.segment", node_id="pb", x=1200, y=400)
    app.processEvents()
    win.scene.clearSelection(); win.scene.node_items["pb"].setSelected(True)
    app.processEvents()
    seg.modes["method"] = "stardist"; doc.touch(); win.inspector._rebuild()
    app.processEvents()
    assert len(_browse_btns()) == 1, "stardist exposes exactly one path socket"
    _browse_btns()[0].click()
    app.processEvents()
    assert picked[-1][0] == "dir", "a local StarDist model is a FOLDER, not a file"
    assert seg.params["sd_model_path"] == r"C:\models\sd", seg.params
    seg.modes["method"] = "cellsam"; doc.touch(); win.inspector._rebuild()
    app.processEvents()
    assert len(_browse_btns()) == 1, "cellsam exposes exactly one path socket"
    _browse_btns()[0].click()
    app.processEvents()
    assert picked[-1][0] == "file" and "*.pt" in picked[-1][3], picked[-1]
    assert picked[-1][2] == r"C:\models\sd", "the dialog must start at the last folder"
    assert seg.params["model_path"] == r"C:\data\movie.nd2", seg.params
    # cancelling must not blank a committed path, and a plain string param stays plain
    _QFD.getOpenFileName = staticmethod(lambda *a: ("", ""))
    _browse_btns()[0].click()
    app.processEvents()
    assert seg.params["model_path"] == r"C:\data\movie.nd2", "cancel must be a no-op"
    win.scene.clearSelection(); t_item.setSelected(True)
    app.processEvents()
    assert not _browse_btns(), "a non-path string/layer param must get no Browse… button"
    doc.remove_node("pb")
    doc.nodes["n1"].params["path"] = ""                     # back to the synthetic demo
    doc.touch()
    app.processEvents()
    _ok("V2.15 path browse: every `path_kind` socket gets a Browse… button — io.load "
        "(filtered file), StarDist local model (folder), CellSAM weights (*.pt, seeded "
        "from the last folder); cancel is a no-op; non-path strings unchanged")

    # ── G3: mute pass-through in the run graph ──────────────────────────────────
    doc.set_muted("n4", True)
    g = doc.to_graph(for_run=True)
    assert "n4" not in {e.dst for e in g.edges} and "n4" not in {e.src for e in g.edges}
    assert any(e.src == "n3" and e.dst == "n5" for e in g.edges)   # bypassed around
    doc.set_muted("n4", False)
    _ok("G3: muted node is bypassed (n3 → n5) in the run graph")

    # ── G6: save / load round-trip incl. canvas positions ──────────────────────
    tmp = os.path.join(tempfile.mkdtemp(prefix="nd2graph_"), "t.nd2graph.json")
    doc.set_pos("n3", 123.0, 456.0)
    doc.save_file(tmp)
    with open(tmp, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["format_version"] == "2.0" and "ui" in raw
    n_nodes, n_edges = len(doc.nodes), len(doc.edges)
    doc.load_file(tmp)
    assert len(doc.nodes) == n_nodes and len(doc.edges) == n_edges
    assert (doc.nodes["n3"].x, doc.nodes["n3"].y) == (123.0, 456.0)
    # headless loader reads the same file (GUI extras ignored)
    from nodegraph.serialize import from_dict
    g2, _z, _gr = from_dict(raw)
    assert set(g2.nodes) == set(doc.nodes)
    _ok("G6: save/load round-trip (positions kept; headless loader reads the file)")

    # ── G7 + G4: a real pull on the synthetic source through to viewer pixels ──
    done = {}
    win.runner.finished.connect(lambda nid, *a: done.setdefault("id", nid))
    win.runner.failed.connect(lambda nid, tr: done.setdefault("err", tr))
    win.pull_node("n3")                                     # gaussian (3D on z=5)
    t0 = time.time()
    while not done and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert done.get("err") is None, f"pull failed:\n{done.get('err')}"
    assert done.get("id") == "n3"
    pm = win.viewer._view._item.pixmap()   # viewer is now a QGraphicsView (was a QLabel)
    assert pm is not None and not pm.isNull() and pm.width() > 100
    assert "pulled in" in win.viewer._status.text()
    _ok(f"G7+G4: engine pull off the UI thread → viewer shows {pm.width()}"
        f"×{pm.height()} px ({win.viewer._status.text().split('·')[-1].strip()})")

    # memo persistence: an immediate re-pull is a cache hit (fast)
    done.clear()
    t0 = time.time()
    win.pull_node("n3")
    while not done and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    repull = time.time() - t0
    assert done.get("id") == "n3" and repull < 5.0
    _ok(f"G7: re-pull hits the persistent memo ({repull*1000:.0f} ms)")

    # ── G2: palette content ─────────────────────────────────────────────────────
    win.palette.refill("gauss")
    tree = win.palette._tree
    found = []
    for i in range(tree.topLevelItemCount()):
        head = tree.topLevelItem(i)
        for j in range(head.childCount()):
            found.append(head.child(j).text(0))
    assert any("Gaussian" in t for t in found), found
    win.palette.refill("")
    _ok("G2: palette search filters the registry")

    # ── review regressions (Phase-5 impl review, 2026-07-22) ──────────────────
    from nodegraph.graph import Edge, Graph, NodeInstance
    from nodegraph.zones import Zone
    from nodelab_v2.document import GraphDocument
    from nodelab_v2.ops import ensure_ops, headless_engine

    # R1 (BLOCKER): load_dict reusing ids rebinds NodeItems to the NEW records
    #   (no stale op_key/spec/position). Standalone scene so the demo stays intact.
    from nodelab_v2.scene import GraphScene as _GS
    rdoc = GraphDocument()
    rdoc.add_node("channel.select", node_id="n1", x=10, y=10)
    rscene = _GS(rdoc)
    old_n1 = rscene.node_items["n1"]
    assert old_n1.op_key == "channel.select"
    rdoc.load_dict({
        "format_version": "2.0",
        "graph": {"nodes": [{"id": "n1", "op_key": "enhance.gaussian",
                             "params": {}, "modes": {}}], "edges": []},
        "ui": {"nodes": {"n1": {"x": 999.0, "y": 888.0, "muted": False}}},
    })
    new_n1 = rscene.node_items["n1"]
    assert new_n1 is not old_n1, "stale NodeItem kept after reload"
    assert new_n1.rec is rdoc.nodes["n1"] and new_n1.op_key == "enhance.gaussian"
    assert (new_n1.pos().x(), new_n1.pos().y()) == (999.0, 888.0)
    _ok("R1: reload rebinds cards to fresh records (op/spec/pos correct, no stale)")

    # R5 (BLOCKER): a file with a zone round-trips through the document unharmed
    zdoc = GraphDocument()
    zg = {
        "format_version": "2.0",
        "graph": {"nodes": [
            {"id": "s", "op_key": "enhance.gamma", "params": {}, "modes": {}},
            {"id": "ri", "op_key": "zone.repeat_in", "params": {}, "modes": {}},
            {"id": "ro", "op_key": "zone.repeat_out", "params": {}, "modes": {}}],
            "edges": [{"src": "ri", "dst": "ro", "src_socket": "out",
                       "dst_socket": "data", "kind": "forward"},
                      {"src": "ro", "dst": "ri", "src_socket": "out",
                       "dst_socket": "data", "kind": "back"}]},
        "zones": [{"id": "z1", "kind": "repeat", "in_id": "ri", "out_id": "ro",
                   "body": [], "iterations": 3, "impure": False}],
    }
    zdoc.load_dict(zg)
    assert zdoc.has_unedited_structure
    round_tripped = zdoc.to_dict()
    assert len(round_tripped["zones"]) == 1 and round_tripped["zones"][0]["id"] == "z1"
    assert round_tripped["zones"][0]["iterations"] == 3
    # the back-edge survived too
    assert any(e["kind"] == "back" for e in round_tripped["graph"]["edges"])
    _ok("R5: zones + back-edges preserved verbatim across GUI load→save")

    # R6 (MAJOR): a GUI-authored graph runs HEADLESS (no PySide6) via ops.headless_engine
    ensure_ops()
    from nodegraph.provider import SyntheticProvider
    from nodegraph.dataset import Dataset as _DS
    hg = Graph()
    hg.add(NodeInstance("src", "io.load"))
    hg.add(NodeInstance("v", "view.viewer"))
    hg.connect("src", "v")
    sp = SyntheticProvider(AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32), tile=16)
    seed = _DS(axes=sp.axes, metadata={"pixel_size_um": 0.1}).with_image(sp)
    heng = headless_engine(hg, seeds={"src": seed},
                           meta_seeds={"src": MetaEnvelope(axes=sp.axes)})
    out_ds = heng.pull("v")                        # view.viewer compute is in COMPUTES
    assert out_ds.image is not None
    _ok("R6: view.viewer compute is Qt-free → GUI graph runs headless")

    # R2 (MAJOR): re-seed re-announces when a node's resolved source key CHANGES
    #   (the old code announced once per node-id and then froze forever). Use a
    #   THROWAWAY runner + fake keys so poking its cache can't corrupt the live
    #   window's real ("synthetic",) source.
    from nodelab_v2.runner import EngineRunner as _ER
    r = _ER(GraphDocument())
    r._node_source_key["nX"] = ("fake-a",)
    r._providers[("fake-a",)] = (object(), MetaEnvelope())
    assert dict(r._fresh_envs())  # first resolution announces
    assert not r._fresh_envs()    # unchanged key: no re-announce
    r._node_source_key["nX"] = ("fake-b",)                  # user typed a real path
    r._providers[("fake-b",)] = (object(), MetaEnvelope(axes=AxisSizes(z=9)))
    fresh = dict(r._fresh_envs())
    assert "nX" in fresh and fresh["nX"].axes.z == 9, "re-seed did not re-announce"
    _ok("R2: source re-seed re-announces on a changed source key (not once-only)")

    # ── G5 spreadsheet + Point overlay (Phase-6 inspection) ───────────────────
    from nodegraph.dataset import Dataset as _DS2
    from nodegraph.domains import Domain as _Dom
    from nodegraph.provider import ArrayProvider
    from nodelab_v2.spreadsheet import structure_tables
    ax5 = AxisSizes(m=1, t=1, z=3, c=1, y=64, x=64)
    sds = _DS2(axes=ax5, metadata={"pixel_size_um": 0.1}).with_image(
        ArrayProvider(np.zeros((1, 1, 3, 1, 64, 64))))
    # a Point structure (2 spots on z=1) + a Label table
    for name, vals in {"id": [1, 2], "m": [0, 0], "t": [0, 0], "c": [0, 0],
                       "z": [1.0, 1.0], "y": [10.0, 40.0], "x": [20.0, 50.0],
                       "intensity": [7.5, 9.0]}.items():
        sds = sds.with_layer(_Dom.POINT, name, np.array(vals), layer="spots")
    for name, vals in {"id": [1], "m": [0], "t": [0], "c": [0], "z": [0.0],
                       "y": [5.0], "x": [5.0], "area": [42.0]}.items():
        sds = sds.with_layer(_Dom.LABEL, name, np.array(vals), layer="labels")

    tables = structure_tables(sds)
    assert ("point", "spots") in tables and ("label", "labels") in tables
    assert set(tables[("point", "spots")]) >= {"id", "y", "x", "intensity"}
    win.sheet.show_dataset("spots-test", sds)
    # select the point table (combo is sorted, so label comes first)
    pt_idx = next(i for i in range(win.sheet._pick.count())
                  if win.sheet._pick.itemData(i) == ("point", "spots"))
    win.sheet._pick.setCurrentIndex(pt_idx)
    assert win.sheet._table.rowCount() == 2               # 2 points
    hdrs = [win.sheet._table.horizontalHeaderItem(i).text()
            for i in range(win.sheet._table.columnCount())]
    assert hdrs[0] == "id" and "intensity" in hdrs        # coord columns first
    _ok(f"G5: spreadsheet groups {len(tables)} structure tables (point+label), "
        f"coord columns first")

    # the viewer overlays the 2 z==1 points and NONE at z==0
    win.viewer._dataset = sds
    win.viewer._axes = ax5
    win.viewer._plane = np.zeros((64, 64))
    from PySide6.QtGui import QPixmap as _QPix
    win.viewer._base_pix = _QPix(64, 64)
    win.viewer._sliders["z"].setMaximum(1)   # frame controls are sliders now (were spins)
    win.viewer._sliders["z"].setValue(1)
    assert len(win.viewer._points_here()) == 2
    win.viewer._sliders["z"].setValue(0)
    assert len(win.viewer._points_here()) == 0            # points are on z==1 only
    _ok("G5/G4: Point overlay filters to the viewed (m,t,z,c) plane")

    # ── Track-trajectory overlay (Phase-6 refinement) ─────────────────────────
    from nodegraph.structure import TrackMembership as _TM
    ax6 = AxisSizes(m=1, t=3, z=1, c=1, y=64, x=64)
    tds = _DS2(axes=ax6).with_image(ArrayProvider(np.zeros((1, 3, 1, 1, 64, 64))))
    # two point tracks over 3 timepoints (ids globally unique, as detect.spots emits)
    for name, vals in {"id": [1, 2, 3, 4, 5, 6], "m": [0]*6, "t": [0, 0, 1, 1, 2, 2],
                       "c": [0]*6, "z": [0.0]*6, "y": [10., 40., 15., 40., 20., 40.],
                       "x": [10., 10., 15., 25., 20., 40.]}.items():
        tds = tds.with_layer(_Dom.POINT, name, np.array(vals), layer="spots")
    tds = tds.with_structure(_TM(track_id=[1, 1, 1, 2, 2, 2], t=[0, 1, 2, 0, 1, 2],
                                 member_id=[1, 3, 5, 2, 4, 6],
                                 member_domain=_Dom.POINT).to_table(layer="spots"))
    win.viewer._dataset = tds
    win.viewer._axes = ax6
    win.viewer._ref_plane = np.zeros((64, 64))
    win.viewer._base_pix = _QPix(64, 64)
    win.viewer._sliders["m"].setMaximum(0)
    win.viewer._sliders["z"].setMaximum(0); win.viewer._sliders["z"].setValue(0)
    win.viewer._sliders["t"].setMaximum(2); win.viewer._sliders["t"].setValue(1)
    trajs = win.viewer._tracks_here()
    assert len(trajs) == 2, f"expected 2 trajectories, got {len(trajs)}"
    by_id = {tid: (path, cur, ts) for tid, path, cur, ts in trajs}
    assert by_id[1][0] == [(10., 10.), (15., 15.), (20., 20.)], by_id[1][0]  # ordered by t
    assert by_id[1][1] == 1 and by_id[2][1] == 1, "current-t vertex ≠ viewed T"
    assert by_id[1][2] == [0, 1, 2], by_id[1][2]           # per-vertex t (trail modes)
    win.viewer._sliders["t"].setValue(2)
    assert all(cur == 2 for _tid, _p, cur, _ts in win.viewer._tracks_here())  # follows T
    assert win.viewer._track_color(1).name() != win.viewer._track_color(2).name()
    win.viewer._repaint()                                  # draws without raising
    _ok("Track overlay: membership↔position join, ordered by t, current-T vertex tracks T")

    # ── Overlay system overhaul (2026-07-28) ──────────────────────────────────
    from nodelab_v2 import overlays as _OV
    vw = win.viewer

    # O1: one Overlays button replaces the three checkboxes; the popup has a tab per
    # domain, and any still-reserved one is visibly inert rather than fake-live.
    assert vw._ovl_btn.text() == "◈ Overlays"
    vw.open_overlay_dialog()
    dlg = vw._ovl_dialog
    assert dlg is not None and dlg.isVisible()
    # Driven off TAB_INFO rather than a hardcoded list: the tab set grows when a domain
    # gains a renderer (V2.19 added Vectors and promoted Voxels from reserved to drawn), and
    # a literal here just means editing the probe every time instead of checking anything.
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == \
        [_OV.TAB_BY_KEY[t].title for t in _OV.TABS], "overlay tabs"
    assert all(_OV.TAB_BY_KEY[t].implemented for t in ("points", "labels", "tracks"))
    # every tab's control set is fully wired (a spec ⇒ a widget), and a tab the renderer
    # does NOT draw is inert rather than fake-live. Driven off TAB_INFO.implemented so
    # promoting a reserved domain to a real one needs no probe edit.
    reserved = [t for t in _OV.TABS if not _OV.TAB_BY_KEY[t].implemented]
    for tab in _OV.TABS:
        for spec in _OV.FIELDS[tab]:
            assert (tab, spec.key) in dlg._widgets, (tab, spec.key)
            if tab in reserved:
                assert not dlg._rows[(tab, spec.key)][1].isEnabled(), (tab, spec.key)
    for tab in reserved:
        assert not dlg._enables[tab].isEnabled(), tab
    _ok(f"O1: Overlays popup — {dlg.tabs.count()} domain tabs, "
        f"{sum(len(_OV.FIELDS[t]) for t in _OV.TABS)} spec-driven controls, "
        f"reserved tabs inert ({', '.join(reserved) or 'none'})")

    # O2: dependent rows go dead when the mode ignores them (no live-looking control
    # the renderer never reads — the same charter the node catalog holds itself to)
    dlg._set("points", "color_mode", "single")
    assert dlg._rows[("points", "color")][1].isEnabled()
    dlg._set("points", "color_mode", "per_point")
    assert not dlg._rows[("points", "color")][1].isEnabled()
    assert dlg._rows[("points", "sat")][1].isEnabled()
    dlg._set("tracks", "trail", "all")
    assert not dlg._rows[("tracks", "window")][1].isEnabled()
    dlg._set("tracks", "trail", "window")
    assert dlg._rows[("tracks", "window")][1].isEnabled()
    _ok("O2: enable_if dependencies grey out the rows the chosen mode ignores")

    # O3: a settings edit repaints without re-pulling, and toggling a domain off is the
    # old checkbox behaviour through the settings (one source of truth)
    assert vw._view.overlay_cb is not None, "no overlay callback on the image surface"
    rev0 = vw._ovl_rev
    dlg._set("labels", "width", 4.0)
    assert vw._ovl_rev > rev0 and vw.overlays.labels.width == 4.0
    assert vw.overlay_enabled("tracks")
    vw.set_overlay_enabled("tracks", False)
    assert not vw.overlays.tracks.enabled
    assert len(vw._tracks_here()) == 2       # geometry still joinable while hidden
    vw.set_overlay_enabled("tracks", True)
    _ok("O3: a settings edit bumps the overlay revision and repaints (no re-pull)")

    # O4: zoom-invariance — the SAME settings drawn at two different zooms must put the
    # same number of ink pixels on screen (sizes are screen px, not image px). Render the
    # renderer directly through two mappings differing only in scale.
    from PySide6.QtGui import QImage as _QImg, QPainter as _QPnt, QColor as _QCol
    from PySide6.QtCore import QPointF as _QPt

    def _ink(scale, cx, cy):
        img = _QImg(240, 240, _QImg.Format_ARGB32)
        img.fill(_QCol(0, 0, 0))
        lab = np.zeros((24, 24), np.int32)
        lab[6:14, 6:14] = 1
        fr = _OV.OverlayFrame(
            map_pt=lambda x, y: _QPt(120 + (x - cx) * scale, 120 + (y - cy) * scale),
            plane_wh=(24, 24), label_plane=lab,
            points=[_OV.PointMark(10.0, 10.0, 1, 0)])
        st = _OV.OverlaySettings()
        st.labels.style = "outline"       # fills DO scale (a fill is the region)
        st.tracks.enabled = False
        p = _QPnt(img)
        _OV.OverlayRenderer().paint(p, st, fr)
        p.end()
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(240, 240, 4)
        return int(np.count_nonzero(arr[..., :3].max(axis=2) > 40))

    ink_out, ink_in = _ink(6.0, 10.0, 10.0), _ink(24.0, 10.0, 10.0)
    # a 4× zoom on the same region: the outline gets LONGER (more of it is on screen) but
    # never THICKER, and the point glyph is pixel-identical — so the ink cannot grow 16×
    assert ink_in < ink_out * 6, f"overlay ink grew with zoom: {ink_out} → {ink_in}"
    ren_a, ren_b = _OV.OverlayRenderer(), _OV.OverlayRenderer()
    g_a, size_a = ren_a.glyph(_OV.PointsOverlay(), _OV.qcolor("#ffc83c"), 1.0)
    g_b, size_b = ren_b.glyph(_OV.PointsOverlay(), _OV.qcolor("#ffc83c"), 1.0)
    assert (size_a, g_a.size()) == (size_b, g_b.size())    # zoom-independent glyph size
    _ok(f"O4: overlay sizes are SCREEN px — 4× zoom kept the ink bounded "
        f"({ink_out}→{ink_in}, not 16×) and the glyph identical")

    # O5: the requested point look — a golden star with a bright centre pixel and a
    # brightness gradient down each arm
    ren = _OV.OverlayRenderer()
    ps = _OV.PointsOverlay(spread=3, unit_px=9.0)
    pm, gsize = ren.glyph(ps, _OV.qcolor(ps.color), 1.0)
    gim = pm.toImage()
    ctr = int(gsize / 2)
    arm = [_QCol(gim.pixel(ctr + int(k * 9), ctr)).red() for k in range(4)]
    assert arm[0] > arm[1] > arm[2] > arm[3] > 0, f"arm gradient not monotonic: {arm}"
    assert _QCol(gim.pixel(ctr, ctr)).green() > _QCol(gim.pixel(ctr + 9, ctr)).green()
    assert len(_OV._DIRS["star"]) == 8 and len(_OV._DIRS["cross"]) == 4
    _ok(f"O5: golden star — bright centre + gradient arms {arm} over 8 directions")

    # O6: per-item colouring is deterministic, distinct, and shared by labels/points/
    # tracks (so "each label a different colour" and "each track a different colour" are
    # literally the same rule), and a label's outline and fill agree
    cols = [_OV.distinct_color(i).name() for i in range(1, 25)]
    assert len(set(cols)) == 24, "per-item palette collided"
    assert _OV.distinct_color(7).name() == cols[6]         # stable across calls (over T)
    lab2 = np.zeros((8, 8), np.int32)
    lab2[1:4, 1:4] = 3
    fill = ren._fill_image(_OV.LabelsOverlay(fill_opacity=100), lab2)
    assert _QCol(fill.pixel(2, 2)).name() == _OV.distinct_color(3).name(), "fill≠outline hue"
    _ok("O6: one golden-angle palette for labels/points/tracks; fill hue == outline hue")

    # O6b: the IDENTITY palette. A label id is re-issued from 1 every frame, so colouring
    # by it makes one cell flash a new colour at every T step. Colour follows the TRACK
    # instead — and because the golden angle says nothing about *distant* indices (slots 5
    # and 39 sit 4.7° apart), two objects that are neighbours get pushed apart on the wheel.
    #
    # The scene is built to hit both: three cells over two timepoints, with the first two
    # SIDE BY SIDE and carrying exactly the ids whose hues collide.
    _lp = np.zeros((1, 2, 1, 1, 64, 64), np.int32)
    _lp[0, 0, 0, 0, 8:14, 8:14] = 5          # ┐ neighbours, and 5 vs 39 is a hue collision
    _lp[0, 0, 0, 0, 8:14, 18:24] = 39        # ┘
    _lp[0, 0, 0, 0, 48:54, 48:54] = 7        # far away — free to keep its own colour
    _lp[0, 1, 0, 0, 10:16, 8:14] = 41        # the same three cells, one frame later
    _lp[0, 1, 0, 0, 10:16, 18:24] = 42
    _lp[0, 1, 0, 0, 48:54, 50:56] = 43
    _lax = AxisSizes(m=1, t=2, z=1, c=1, y=64, x=64)
    _lds = _DS2(axes=_lax).with_image(ArrayProvider(np.zeros((1, 2, 1, 1, 64, 64))))
    _lds = _lds.with_layer(_Dom.VOXEL, "labels", _lp)
    for _n, _v in {"id": [5, 39, 7, 41, 42, 43], "m": [0]*6, "t": [0, 0, 0, 1, 1, 1],
                   "c": [0]*6, "z": [0.0]*6,
                   "y": [10.5, 10.5, 50.5, 12.5, 12.5, 50.5],
                   "x": [10.5, 20.5, 50.5, 10.5, 20.5, 52.5]}.items():
        _lds = _lds.with_layer(_Dom.LABEL, _n, np.array(_v), layer="labels")
    _lds = _lds.with_structure(_TM(track_id=[1, 1, 2, 2, 3, 3], t=[0, 1, 0, 1, 0, 1],
                                   member_id=[5, 41, 39, 42, 7, 43],
                                   member_domain=_Dom.LABEL).to_table(layer="labels"))
    vw.show_result("LABELS", {0: np.zeros((64, 64), np.uint16)}, _lax, 0.01, dataset=_lds)
    _keys = vw._label_keys(_lp[0, 0, 0, 0])
    assert _keys is not None, "no palette for a tracked label layer"
    _slot = {i: int(_keys[i]) for i in (5, 39, 7, 41, 42, 43)}
    # one cell, one slot, across T — the whole point
    assert _slot[5] == _slot[41] and _slot[39] == _slot[42] and _slot[7] == _slot[43], _slot
    # three cells, three colours
    assert len({_slot[5], _slot[39], _slot[7]}) == 3, _slot
    # and the two that sit side by side are far apart on the wheel, unlike their raw ids
    _hue = lambda s: (s * _OV.GOLDEN_ANGLE) % 360.0
    _sep = lambda a, b: min(abs(_hue(a) - _hue(b)) % 360.0,
                            360.0 - abs(_hue(a) - _hue(b)) % 360.0)
    assert _sep(5, 39) < 10.0, "fixture no longer exercises a hue collision"
    assert _sep(_slot[5], _slot[39]) >= 25.0, (
        f"neighbouring tracks {_sep(_slot[5], _slot[39]):.1f}° apart", _slot)
    # painted proof: the SAME pixel colour for that cell on both frames, and the untracked
    # rule (colour by raw id) would have given two different ones
    def _cell_rgb(tv, y, x):
        plane = _lp[0, tv, 0, 0]
        im = ren._fill_image(_OV.LabelsOverlay(fill_opacity=100), plane,
                             vw._label_keys(plane))
        return _QCol(im.pixel(x, y)).name()
    assert _cell_rgb(0, 10, 10) == _cell_rgb(1, 12, 10), "a tracked cell changed colour on T"
    assert _OV.distinct_color(5).name() != _OV.distinct_color(41).name()   # the old rule
    # the trajectory is drawn in its cells' colour, not in a palette of its own
    assert vw._track_color(1).name() == _cell_rgb(0, 10, 10), "track ≠ its own regions"
    _ok(f"O6b: identity palette — one colour per tracked cell across T (slots {_slot}), "
        f"neighbours pushed {_sep(_slot[5], _slot[39]):.0f}° apart (raw ids: "
        f"{_sep(5, 39):.0f}°), trajectory matches its regions")

    # O7: "spread to other tabs" copies by ROLE and reports every move
    st = _OV.OverlaySettings()
    st.labels.opacity = 55
    st.labels.width = 3.5
    st.labels.color_mode = "single"
    notes = _OV.spread_settings(st, "labels")
    assert st.tracks.opacity == 55 and st.points.opacity == 55
    assert st.tracks.width == 3.5 and st.points.thickness == 3.5   # role, not key name
    assert st.tracks.color_mode == "single" and st.points.color_mode == "single"
    assert st.labels.style == "both" and st.tracks.trail == "all"   # domain-only untouched
    assert notes and all(":" in n for n in notes)
    assert not _OV.spread_settings(st, "labels")                    # idempotent
    _ok(f"O7: spread copied {len(notes)} role-matched settings, left domain-only alone")

    # O8: persistence — round-trip a file, merge a PARTIAL dict, survive a bad one
    ovl_dir = tempfile.mkdtemp(prefix="nd2ovl_")
    ovl_path = os.path.join(ovl_dir, f"look{_OV.FILE_SUFFIX}")
    st.points.shape = "ring"
    st.points.spread = 5
    _OV.write_json(pathlib.Path(ovl_path), st)
    back = _OV.OverlaySettings()
    back.update_from_dict(_OV.read_json(pathlib.Path(ovl_path)))
    assert back.to_dict() == st.to_dict(), "overlay settings did not round-trip"
    part = _OV.OverlaySettings()
    changed = part.update_from_dict({"overlays": {"labels": {"width": 7.0,
                                                            "bogus_key": 1}}})
    assert changed == ["labels.width"] and part.labels.width == 7.0
    assert part.points.shape == "star"          # a partial file leaves the rest alone
    env_path = os.path.join(ovl_dir, "env.json")
    _OV.write_json(pathlib.Path(env_path), st)
    os.environ[_OV.ENV_VAR] = env_path
    try:
        loaded, sources = _OV.load_defaults()
        assert loaded.points.shape == "ring" and sources and "env.json" in sources[0]
    finally:
        os.environ.pop(_OV.ENV_VAR, None)
    _ok(f"O8: settings round-trip, partial merge ({changed}), and "
        f"{_OV.ENV_VAR} override load")

    # O9: the dialog's Load/Save/default buttons write real files
    dlg._settings.update_from_dict(st.to_dict())
    dlg._settings.labels.width = 6.5            # a marker the points-tab reset must keep
    dlg.reload()
    user_path = os.path.join(ovl_dir, "user.json")
    dlg._write(pathlib.Path(user_path), "test")
    assert os.path.isfile(user_path)
    assert dlg.current_tab() == "points" and dlg._settings.points.shape == "ring"
    dlg._reset_tab()                            # current tab (points) back to built-in
    assert dlg._settings.points.shape == "star", "reset tab did not restore defaults"
    assert dlg._settings.labels.width == 6.5, "reset tab touched another tab"
    dlg._reset_all()
    assert dlg._settings.to_dict() == _OV.OverlaySettings().to_dict()
    dlg.close()
    _ok("O9: dialog save / reset-tab / reset-all act on the live settings")

    # O10: the MESH overlay (V2.08) — a 3-D surface has no single 2-D picture, so it is
    # drawn as its CROSS-SECTION at the viewed Z. Two 8-voxel cubes must each yield one
    # closed loop with exactly the cube's extent on a mid-plane, nothing at all off the
    # mesh, and the vertex style must be empty at a cube's mid-plane (no vertices there)
    # yet populated on a corner plane.
    import nodegraph.mesh as _MSH
    from nodegraph.dataset import Dataset as _MDS
    from nodegraph.provider import ArrayProvider as _MAP
    from PySide6.QtCore import QRectF as _QRect

    def _cube_mesh(z0, y0, x0, s):
        V = np.array([[z0 + dz, y0 + dy, x0 + dx]
                      for dz in (0., s) for dy in (0., s) for dx in (0., s)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        Fs = []
        for a, b, c, d in quads:
            Fs += [[a, b, c], [a, c, d]]
        return V, np.array(Fs, np.int64)

    _mels = []
    for _i, (_yo, _xo) in enumerate([(6., 6.), (6., 26.)]):
        _V, _F = _cube_mesh(4., _yo, _xo, 8.)
        _mels.append(_MSH.MeshElement(m=0, t=0, c=0, src_label=_i, verts_zyx=_V, faces=_F,
                                      centroid_zyx=(8., _yo + 4., _xo + 4.),
                                      volume_um3=1., surface_area_um2=1., density=1.,
                                      n_points=8))
    _mtb = _MSH.build_mesh_tables(_mels, layer="mesh")
    _max = AxisSizes(m=1, t=1, z=16, c=1, y=48, x=48)
    _mds = _MSH.with_mesh(
        _MDS(axes=_max, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
        .with_image(_MAP(np.zeros((1, 1, 16, 1, 48, 48), np.uint16))),
        _mtb, provenance={"boundary": "convex_hull"})
    assert _OV.TAB_BY_KEY["mesh"].implemented, "the mesh tab must be live now"
    vw.show_result("MESH", {0: np.zeros((48, 48), np.uint16)}, _max, 0.01, dataset=_mds)
    assert vw.overlay_enabled("mesh")
    vw._sliders["z"].setValue(8)
    vw._geo_key = None
    vw._ensure_geometry()
    _secs = vw._geo_mesh
    assert len(_secs) == 2, f"expected 2 cross-sections, got {len(_secs)}"
    assert [s.object_id for s in _secs] == [1, 2]
    _ext = []
    for _s in _secs:
        assert all(_s.closed), "a cube's cross-section must close"
        _pts = np.array(_s.loops[0])
        _ext.append((_pts[:, 0].min(), _pts[:, 0].max(),
                     _pts[:, 1].min(), _pts[:, 1].max()))
    assert _ext[0] == (6.0, 14.0, 6.0, 14.0), _ext[0]
    assert _ext[1] == (6.0, 14.0, 26.0, 34.0), _ext[1]
    vw._sliders["z"].setValue(0)                     # off the mesh entirely
    vw._geo_key = None
    vw._ensure_geometry()
    assert not vw._geo_mesh, "the mesh must vanish on a plane it does not cross"
    # every style must put real ink on screen. Painted through an EXPLICIT mapping (as O4
    # does) rather than the live widget transform, so the assertion is about the renderer
    # and not about where the panel happens to be scrolled.
    def _mesh_ink(style, zv):
        vw.overlays.mesh.style = style
        vw._sliders["z"].setValue(zv)
        vw._geo_key = None
        vw._ensure_geometry()
        secs = vw._geo_mesh
        im = _QImg(200, 200, _QImg.Format_ARGB32)
        im.fill(_QCol(0, 0, 0))
        fr = _OV.OverlayFrame(
            map_pt=lambda x, y: _QPt(x * 4.0, y * 4.0), plane_wh=(48, 48), mesh=secs)
        st_m = _OV.OverlaySettings()
        st_m.points.enabled = st_m.labels.enabled = st_m.tracks.enabled = False
        st_m.mesh = vw.overlays.mesh
        p = _QPnt(im)
        _OV.OverlayRenderer().paint(p, st_m, fr)
        p.end()
        arr = np.frombuffer(im.constBits(), np.uint8).reshape(200, 200, 4)
        return len(secs), int(np.count_nonzero(arr[..., :3].max(axis=2) > 40))

    _fill_ink = None
    for _style, _zv, _want in (("wireframe", 8, 2), ("surface", 8, 2),
                               ("points", 8, 0), ("points", 4, 2)):
        _n, _ink = _mesh_ink(_style, _zv)
        assert _n == _want, (_style, _zv, _n)
        if _want:
            assert _ink > 0, f"mesh style {_style} drew nothing"
        if _style == "surface":
            _fill_ink = _ink
    # a filled cross-section must cover strictly more than its outline alone
    _, _outline_ink = _mesh_ink("wireframe", 8)
    assert _fill_ink > _outline_ink, (_fill_ink, _outline_ink)
    vw.overlays.mesh.style = "wireframe"
    _pv = _QImg(240, 160, _QImg.Format_ARGB32)
    _pp = _QPnt(_pv)
    _OV.render_preview(_pp, _QRect(0, 0, 240, 160), vw.overlays, "mesh")
    _pp.end()
    vw.set_overlay_enabled("mesh", False)
    vw._geo_key = None
    vw._ensure_geometry()
    assert not vw._geo_mesh, "a disabled overlay must extract no geometry"
    vw.set_overlay_enabled("mesh", True)
    _ok("O10: MESH overlay draws the Z cross-section (2 closed loops at the exact cube "
        "extents, empty off-plane), all 3 styles paint + preview, on/off honoured")

    # ── Phase-remainder features (2026-07-22) ─────────────────────────────────
    from nodelab_v2.export import export_dataset
    from nodelab_v2.document import GraphDocument as _GD

    # E1: CSV export of the crafted point+label Dataset
    csv_path = os.path.join(tempfile.mkdtemp(prefix="nd2exp_"), "t.csv")
    n_rows = export_dataset(sds, csv_path)
    with open(csv_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert n_rows == 3 and lines[0].startswith("domain,layer,")   # 2 points + 1 label
    assert any("point" in ln for ln in lines) and any("label" in ln for ln in lines)
    _ok(f"Export: CSV wrote {n_rows} rows (long form, domain/layer columns)")

    # E1b: Parquet export (pyarrow present)
    try:
        pq_path = os.path.join(os.path.dirname(csv_path), "t.parquet")
        export_dataset(sds, pq_path)
        import pyarrow.parquet as _pq
        assert _pq.read_table(pq_path).num_rows == 3
        _ok("Export: Parquet round-trips (pyarrow)")
    except ImportError:
        _ok("Export: Parquet SKIPPED (pyarrow absent)")

    # E2: splice-on-wire inserts a node into an existing link
    sdoc = _GD()
    sdoc.add_node("io.load", node_id="a", x=0, y=0)
    sdoc.add_node("enhance.gaussian", node_id="b", x=200, y=0)
    sdoc.connect("a", "image", "b", "data")
    ss = _GS(sdoc)
    ok_sp = ss.splice_onto(sdoc.add_node("enhance.gamma", node_id="g", x=100, y=0).id,
                           ("a", "image", "b", "data"))
    assert ok_sp
    assert sdoc.edge_into("b", "data") == ("g", "out", "b", "data")
    assert ("a", "image", "g", "data") in sdoc.edges         # a → g → b
    assert ("a", "image", "b", "data") not in sdoc.edges     # original edge replaced
    # a Dataset node spliced onto a VALUE-socket wire must be REFUSED without dropping
    # the wire (the second connect, into a value input, is invalid). Construct the
    # value wire directly (the catalog has no value-output node to form one naturally).
    sdoc.edges.append(("a", "image", "b", "sigma"))          # dataset-out → value-in
    spliced = ss.splice_onto(
        sdoc.add_node("enhance.gamma", node_id="g2", x=50, y=200).id,
        ("a", "image", "b", "sigma"))
    assert not spliced and ("a", "image", "b", "sigma") in sdoc.edges   # not dropped
    _ok("Splice: inserts on Dataset wires; refuses (no dropped wire) on a value wire")

    # E3: collapse toggles a compact layout (fewer socket rows, shorter card)
    gi = win.scene.node_items["n3"]
    tall = gi._height
    win.doc.set_collapsed("n3", True)
    assert win.scene.node_items["n3"]._height < tall
    assert win.scene.node_items["n3"].rec.collapsed
    win.doc.set_collapsed("n3", False)
    assert win.scene.node_items["n3"]._height == tall
    _ok("Collapse: compact layout toggles + relayouts (C key path)")

    # E4: light theme rebinds tokens + restyles without error
    dark_bg = T.BG.name()
    win.set_theme("light")
    assert T.MODE == "light" and T.BG.name() != dark_bg
    assert T.INK.lightness() < T.BG.lightness()      # dark ink on light bg
    win.set_theme("dark")
    assert T.MODE == "dark" and T.BG.name() == dark_bg
    _ok("G9: light/dark theme toggle rebinds palette + restyles panels")

    # E5: Repeat-zone creation wraps a linear chain (validated via unroll)
    zdoc2 = _GD()
    zdoc2.add_node("io.load", node_id="src", x=0, y=0)
    zdoc2.add_node("enhance.gamma", node_id="body", x=200, y=0)
    zdoc2.add_node("analysis.threshold", node_id="sink", x=400, y=0)
    zdoc2.connect("src", "image", "body", "data")
    zdoc2.connect("body", "out", "sink", "data")
    zid = zdoc2.wrap_repeat_zone(["body"], iterations=4)
    assert zid and len(zdoc2._zones) == 1 and zdoc2._zones[0].iterations == 4
    assert zdoc2._zones[0].kind == "repeat"
    assert any(op == "zone.repeat_in" for op in
               (r.op_key for r in zdoc2.nodes.values()))
    assert len(zdoc2._back_edges) == 1                # the Out→In feedback edge
    from nodegraph.zones import unroll as _unr
    _unr(zdoc2.to_graph(for_run=False), zdoc2._zones)  # unrolls cleanly
    # a re-save round-trips the zone + back-edge
    assert zdoc2.has_unedited_structure
    rt = zdoc2.to_dict()
    assert len(rt["zones"]) == 1
    # ambiguous selection refuses cleanly (two outputs)
    try:
        zdoc2.wrap_repeat_zone(["src", "body", "sink"])
        raise SystemExit("wrap should have refused a node already in a zone")
    except ValueError:
        pass
    _ok("Zones: Repeat-zone wrap (validated via unroll, round-trips, refuses bad sel)")

    # E6: labelled frames (GUI-only canvas grouping) — create, enclose, follow, persist
    from nodelab_v2.frame_item import FrameItem as _FI
    fdoc = _GD()
    fsc = _GS(fdoc)
    fdoc.add_node("enhance.gamma", node_id="fa", x=100, y=100)
    fdoc.add_node("enhance.gamma", node_id="fb", x=400, y=250)
    fdoc.add_node("enhance.gamma", node_id="fc", x=900, y=100)
    frec = fdoc.add_frame("Preprocess", ["fa", "fb"])
    fit = fsc.frame_items[frec.id]
    assert isinstance(fit, _FI) and fit.zValue() < 0        # behind the nodes
    fr_rect = fit.mapToScene(fit.boundingRect()).boundingRect()
    for nid in ("fa", "fb"):
        ni = fsc.node_items[nid]
        assert fr_rect.contains(ni.mapToScene(ni.boundingRect()).boundingRect())
    nc = fsc.node_items["fc"]
    assert not fr_rect.contains(nc.mapToScene(nc.boundingRect()).boundingRect())
    w0 = fit.mapToScene(fit.boundingRect()).boundingRect().width()
    fsc.node_items["fb"].setPos(700, 520)                   # move a member
    assert fit.mapToScene(fit.boundingRect()).boundingRect().width() > w0  # frame follows
    # save/load round-trip via the ui extras
    fd = fdoc.to_dict()
    assert frec.id in fd["ui"]["frames"] and fd["ui"]["frames"][frec.id]["title"] == "Preprocess"
    fdoc2 = _GD(); fdoc2.load_dict(fd)
    assert fdoc2.frames[frec.id].members == ["fa", "fb"]
    # deleting a member prunes; emptying auto-removes; deleting a frame keeps its nodes
    fdoc.remove_node("fa")
    assert fdoc.frames[frec.id].members == ["fb"]
    fdoc.remove_node("fb")
    assert frec.id not in fdoc.frames and frec.id not in fsc.frame_items
    keep = fdoc.add_frame("Keep", ["fc"])
    fdoc.remove_frame(keep.id)
    assert keep.id not in fdoc.frames and "fc" in fdoc.nodes
    _ok("Frames: create/enclose/follow-move, save-load, prune, delete keeps nodes")

    # E7: reroute — hidden pass-through node, spliced into a wire, renders compact
    rrdoc = _GD()
    rrsc = _GS(rrdoc)
    rrdoc.add_node("enhance.gamma", node_id="ra", x=0, y=0)
    rrdoc.add_node("enhance.gamma", node_id="rb", x=400, y=0)
    rrdoc.connect("ra", "out", "rb", "data")
    from nodelab_v2.scene import visible_specs as _visible_specs
    assert not any(s.op_key == "rr.reroute" for s in _visible_specs())  # hidden from palette
    rr = rrdoc.add_node("rr.reroute", x=200, y=0)
    assert rrsc.splice_onto(rr.id, ("ra", "out", "rb", "data"))         # a → reroute → b
    assert ("ra", "out", rr.id, "data") in rrdoc.edges
    assert rrdoc.edge_into("rb", "data") == (rr.id, "out", "rb", "data")
    rit = rrsc.node_items[rr.id]
    # card_rect is the geometry; boundingRect adds the glow repaint margin on top
    assert rit._is_reroute and rit.card_rect().width() == T.RR_SIZE      # compact dot
    assert rit.boundingRect().width() == T.RR_SIZE + 2 * rit.GLOW_M
    assert rit.socket("in", "data") is not None and rit.socket("out", "out") is not None
    _ok("Reroute: hidden pass-through, splices into a Dataset wire, renders compact")

    # E8: group creation — collapse a linear sub-chain into a reusable group instance
    from nodegraph.groups import group_name_of as _gname
    gdoc = _GD()
    gsc = _GS(gdoc)
    gdoc.add_node("io.load", node_id="gs", x=0, y=0)
    gdoc.add_node("enhance.gamma", node_id="ga", x=200, y=0)
    gdoc.add_node("enhance.gaussian", node_id="gb", x=400, y=0)
    gdoc.add_node("analysis.threshold", node_id="gc", x=600, y=0)
    gdoc.add_node("analysis.label", node_id="gd", x=800, y=0)
    gdoc.connect("gs", "image", "ga", "data")
    gdoc.connect("ga", "out", "gb", "data")
    gdoc.connect("gb", "out", "gc", "data")
    gdoc.connect("gc", "out", "gd", "data")
    ginst = gdoc.make_group(["ga", "gb", "gc"], name="Preprocess")
    assert gdoc.nodes[ginst].op_key == "group:Preprocess"
    assert {"ga", "gb", "gc"}.isdisjoint(gdoc.nodes)         # members left the parent
    assert ("gs", "image", ginst, "data") in gdoc.edges and (ginst, "out", "gd", "data") in gdoc.edges
    gitem = gsc.node_items[ginst]                            # renders as a group card
    assert gitem._is_group and gitem.socket("in", "data") and gitem.socket("out", "out")
    # the run/propagate graph inlines the body (no residual group:* nodes)
    gexp = gdoc.to_graph(for_run=True, materialize=True)
    assert not any(_gname(n.op_key) for n in gexp.nodes.values())
    assert any(n.op_key == "enhance.gaussian" for n in gexp.nodes.values())
    # save/load round-trips the instance + definition (groups are GUI-manageable now)
    gd_dict = gdoc.to_dict()
    assert len(gd_dict["groups"]) == 1 and not gdoc.has_unedited_structure
    _GD().load_dict(gd_dict)                                 # loads without error
    # ungroup restores the interior + reconnects the frontier
    assert gdoc.ungroup(ginst) and ginst not in gdoc.nodes and not gdoc._groups
    assert gdoc.edge_into("gd", "data") is not None          # chain reconnected to sink
    assert sum(1 for n in gdoc.nodes.values() if n.op_key == "enhance.gaussian") == 1
    # guardrail: grouping a source (buries its seed) is refused
    try:
        gdoc.make_group(["gs"], "X"); raise SystemExit("grouping a source should raise")
    except ValueError:
        pass
    _ok("Groups: make_group→instance, run-expand, save-load, ungroup, source guardrail")

    # E8b: deleting a node — all three affordances, plus dissolve (delete + heal) ──
    ddoc = _GD()
    dsc = _GS(ddoc)
    for i, op in enumerate(("io.load", "enhance.gamma", "enhance.gaussian",
                            "analysis.threshold")):
        ddoc.add_node(op, node_id=f"d{i}", x=220 * i, y=0)
    ddoc.connect("d0", "image", "d1", "data")
    ddoc.connect("d1", "out", "d2", "data")
    ddoc.connect("d2", "out", "d3", "data")
    deleted: list = []
    dsc.nodes_deleted.connect(lambda ids: deleted.extend(ids))
    # (1) the hover ✕ badge on the card
    ditem = dsc.node_items["d3"]
    assert not ditem._close.isVisible()                  # hidden until hovered
    from PySide6.QtCore import QEvent as _QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QGraphicsSceneHoverEvent as _Hov
    _h = _Hov(_QEvent.GraphicsSceneHoverEnter); _h.setPos(QPointF(5, 5))
    ditem.hoverEnterEvent(_h)
    assert ditem._close.isVisible()                      # …then it offers itself
    ditem._close.clicked.emit()
    assert "d3" not in ddoc.nodes and deleted == ["d3"]
    # (2) Delete key through the scene (what the canvas keyboard does)
    dsc.clearSelection()
    dsc.node_items["d2"].setSelected(True)
    dsc.keyPressEvent(QKeyEvent(_QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    assert "d2" not in ddoc.nodes
    # (3) the right-click context menu (built, then its Delete action triggered)
    from PySide6.QtWidgets import QMenu as _QMenu
    dsc.clearSelection()
    dmenu = _QMenu()
    dsc._fill_node_menu(dmenu, dsc.node_items["d1"])
    texts = [a.text() for a in dmenu.actions()]
    assert any(t.startswith("Delete node") for t in texts), texts
    assert any("Dissolve" in t for t in texts), texts
    next(a for a in dmenu.actions() if a.text().startswith("Delete node")).trigger()
    assert "d1" not in ddoc.nodes and ddoc.nodes and set(ddoc.nodes) == {"d0"}
    # (4) dissolve heals the chain: src → [mid] → dst becomes src → dst
    hdoc = _GD()
    hsc = _GS(hdoc)
    hdoc.add_node("io.load", node_id="h0", x=0, y=0)
    hdoc.add_node("enhance.gamma", node_id="h1", x=200, y=0)
    hdoc.add_node("analysis.threshold", node_id="h2", x=400, y=0)
    hdoc.add_node("analysis.label", node_id="h3", x=600, y=0)
    hdoc.connect("h0", "image", "h1", "data")
    hdoc.connect("h1", "out", "h2", "data")
    hdoc.connect("h2", "out", "h3", "data")
    assert hsc.dissolve_node("h1")
    assert "h1" not in hdoc.nodes
    assert hdoc.edge_into("h2", "data") == ("h0", "image", "h2", "data")   # healed
    assert hsc.dissolve_node("h0")                        # a SOURCE just leaves a gap
    assert hdoc.edge_into("h2", "data") is None and "h2" in hdoc.nodes
    # the window's Edit action works no matter where the keyboard focus is
    win.viewer.setFocus()
    win.scene.clearSelection()
    win.scene.node_items["n6"].setSelected(True)
    win._sync_edit_actions()
    assert win._del_act.isEnabled() and win._dissolve_act.isEnabled()
    win._del_act.trigger()
    assert "n6" not in win.doc.nodes
    win.scene.clearSelection()
    win._sync_edit_actions()
    assert not win._del_act.isEnabled()                   # nothing selected → greyed out
    win.build_demo()                                      # restore the example graph
    app.processEvents()
    _ok("Delete: hover ✕ badge, Del key, context menu, Edit action (focus-proof); "
        "dissolve deletes a mid-chain node and reconnects the wire through it")

    # E8c: per-node progress — plan → queued → running/cached → done, on the cards ──
    events: list = []
    win.runner.node_progress.connect(
        lambda ev, nid, info: events.append((ev, nid, info.get("fraction"))))
    plans: list = []
    win.runner.plan.connect(lambda t, ids: plans.append((t, sorted(ids))))
    # edit n4's param first, so THIS node is guaranteed to recompute while its upstream
    # is served from the session's persistent memo (earlier probes already pulled it) —
    # which is exactly the mixed recompute/cached picture the cards must show.
    win.doc.nodes["n4"].params["threshold"] = 0.37
    win.doc.touch()
    app.processEvents()
    win.pull_node("n4")                       # Load → Select → Gaussian → Threshold
    t0 = time.time()
    while win.runner._busy and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert plans and plans[-1][0] == "n4"
    assert set(plans[-1][1]) == {"n1", "n2", "n3", "n4"}, plans[-1]
    kinds = [(ev, nid) for ev, nid, _f in events]
    assert ("start", "n4") in kinds and ("done", "n4") in kinds, kinds
    n3_last = max(i for i, k in enumerate(kinds) if k[1] == "n3")
    assert n3_last < kinds.index(("start", "n4")), kinds   # upstream settles first
    assert kinds[n3_last][0] in ("done", "cached"), kinds[n3_last]
    # analysis.threshold is EAGER (per-plane) so it reports real fractions; the last one
    # always lands on 1.0 (the runner never throttles the final update)
    fr = [f for ev, nid, f in events if ev == "progress" and nid == "n4"]
    assert fr and fr[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fr), fr
    items = win.scene.node_items
    assert items["n4"].run_state() == "done" and items["n4"]._run_text().endswith(
        ("ms", "s")), items["n4"]._run_text()
    # the card itself stays graphic (header rail + status dot) — the wall time rides in
    # its tooltip, so the numbers are one hover away instead of printed on the canvas
    assert items["n4"].toolTip().endswith(("ms", "s")), items["n4"].toolTip()
    assert "n4" in items["n4"].toolTip()
    assert items["n3"].run_state() in ("done", "cached")
    assert items["n7"].run_state() == ""          # not in this pull → no stale state
    assert not items["n7"].toolTip()
    assert not any(i.run_state() in ("queued", "running", "decoding")
                   for i in items.values())      # everything settled
    assert not any(e.flow for e in win.scene.edge_items)   # nothing in flight → no flow
    # a re-pull of the same graph is all memo hits → 'cached' cards, no recompute
    events.clear()
    win.pull_node("n4")
    t0 = time.time()
    while win.runner._busy and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert any(ev == "cached" for ev, _n, _f in events), events
    assert items["n3"].run_state() == "cached"
    # ONE shared scene timer animates both the working cards (pulsing dot / sweeping
    # rail) and the flowing wires — and only while something needs animating
    assert not win.scene._anim.isActive()
    win.scene._set_state("n3", "running")         # no fraction → indeterminate sweep
    assert items["n3"].is_running() and win.scene._anim.isActive()
    flowing = [e.model_edge for e in win.scene.edge_items if e.flow]
    assert flowing, "a pull in flight must flow the wires out of produced nodes"
    assert all(e[0] in win.scene._run for e in flowing), flowing
    ph, eph = items["n3"]._phase, win.scene.edge_items[0]._phase
    win.scene._tick_progress()
    assert items["n3"]._phase != ph
    assert any(e._phase != eph for e in win.scene.edge_items if e.flow)
    win.scene.clear_run_states()
    assert not win.scene._anim.isActive() and items["n3"].run_state() == ""
    assert not any(e.flow for e in win.scene.edge_items)
    # the status-bar LED is the footer twin of the card dot: pulses busy, settles idle
    assert win._led_state == "idle" and not win._led_timer.isActive()
    win._set_led("busy")
    assert win._led_timer.isActive() and win._led_on
    win._led_tick()
    assert not win._led_on                        # off-beat of the pulse
    win._set_led("idle")
    assert not win._led_timer.isActive() and win._led_on
    # the status-bar determinate bar: shown ONLY while a real fraction is in hand (a
    # long ingest reports one), hidden for indeterminate work and after the run ends —
    # a bar that invents a position would be worse than no bar.
    assert not win._prog.isVisible()
    win._on_node_progress("progress", "n3",
                          {"fraction": 0.25, "note": "reading ND2", "epoch": 0})
    assert win._prog.isVisible() and win._prog.value() == 250, win._prog.value()
    assert "25%" in win.statusBar().currentMessage()
    assert "reading ND2" in win.statusBar().currentMessage()
    win._on_node_progress("progress", "n3", {"fraction": 1.0, "epoch": 0})
    assert win._prog.value() == win._prog.maximum()      # a bar must land full
    win._on_node_progress("progress", "n3", {"epoch": 0})     # no fraction → indeterminate
    assert not win._prog.isVisible()
    # TWO LEVELS (V2.17): a compute that reports a frame count drives a second bar ABOVE
    # the first — frames finished / total frames — while the lower one becomes the work
    # inside the frame in flight. Both here and on the card.
    assert not win._prog_frame.isVisible()        # nothing reported a frame axis yet
    two = {"epoch": 0, "fraction": 0.31, "note": "t=12 z=2", "done": 62, "total": 200,
           "frames": 40, "frames_done": 12, "frame": 12, "frame_fraction": 0.3,
           "sub_done": 2, "sub_total": 5, "sub_fraction": 0.4}
    win._on_node_progress("progress", "n3", two)
    assert win._prog_frame.isVisible() and win._prog.isVisible()
    assert win._prog_frame.value() == 300, win._prog_frame.value()     # 12/40 frames
    assert win._prog.value() == 400, win._prog.value()                 # 2/5 of the frame
    # the frame bar sits ABOVE the sub bar (the reason the box is a vertical layout)
    assert win._prog_frame.y() < win._prog.y(), (win._prog_frame.y(), win._prog.y())
    msg = win.statusBar().currentMessage()
    assert "frame 13/40" in msg and "t=12 z=2" in msg, msg
    card = win.scene.node_items["n3"]
    assert card._frames == 40 and card._frame == 12
    assert card._frame_frac == 0.3 and card._sub_frac == 0.4
    assert "frame 13/40" in card.toolTip() and "40% of frame" in card.toolTip(), \
        card.toolTip()
    # A frame whose inner work is ONE OPAQUE CALL (CellSAM / StarDist inference): the frame
    # bar holds its position and the sub bar SWEEPS. Freezing it at a percentage for the
    # thirty seconds an inference takes is what reads as a hang.
    opaque = {**two, "sub_done": None, "sub_total": None, "sub_fraction": None}
    win._on_node_progress("progress", "n3", opaque)
    assert win._prog_frame.isVisible() and win._prog_frame.value() == 300
    assert win._prog.isVisible() and win._prog.maximum() == 0, \
        "the sub bar must go INDETERMINATE (range 0,0), not hide and not hold a value"
    assert "(working)" in win.statusBar().currentMessage()
    card = win.scene.node_items["n3"]
    assert card._frame_frac == 0.3 and card._sub_frac is None
    assert "working" in card.toolTip() and "% of frame" not in card.toolTip(), card.toolTip()
    # the card must animate the sweep off the shared timer, with no new events arriving
    assert card.is_running() and win.scene._anim.isActive()
    _ph = card._phase
    win.scene._tick_progress()
    assert card._phase != _ph, "a sweeping sub rail must advance on the shared timer"
    # …and a report WITHOUT a frame count drops back to the single bar rather than
    # leaving a stale frame bar standing at its last value
    win._on_node_progress("progress", "n3", {"fraction": 0.5, "epoch": 0})
    assert not win._prog_frame.isVisible() and win._prog.isVisible()
    assert win._prog.maximum() == 1000, "the sub bar must come back OUT of indeterminate"
    assert win.scene.node_items["n3"]._frame_frac is None
    win._set_progress(0.5)
    win._on_run_failed("n3", "Traceback…\nBoom")              # a failure clears it
    assert not win._prog.isVisible() and not win._prog_frame.isVisible()
    win.scene.clear_run_states()
    _ok("Progress: two-level rails/bars (orange frames above blue within-the-frame; the "
        "blue one SWEEPS through an opaque per-frame call instead of freezing; both drop "
        "to one when no frame axis is reported); plan→queued, per-node "
        "start/done/cached (rail + dot on the card, wall "
        "time in its tooltip), eager fractions, flowing wires, one shared timer, "
        "status-bar determinate bar (ingest fraction; hidden when indeterminate/ended)")

    # E9: maximized canvas + mini-map Viewer + click-to-preview ────────────────
    docked_sizes = win._center.sizes()
    assert win._center.count() == 2 and win._center.widget(0) is win.viewer
    win.set_maximized(True)
    for _ in range(3):
        app.processEvents()                            # splitter re-layout + reposition
    # the SAME viewer widget moved into the overlay (not a copy) and left the splitter
    assert win.viewer.parent() is win.minimap and win.minimap.content is win.viewer
    assert win._center.count() == 1 and win._center.widget(0) is win.view
    assert win.minimap.isVisible() and win.minimap.parent() is win.view
    assert win.view.is_maximized() and win._max_act.isChecked()
    # pinned to the canvas' TOP-LEFT corner, inside it
    mg = win.minimap.geometry()
    assert mg.left() < win.view.width() / 2 and mg.top() < win.view.height() / 2
    assert mg.right() < win.view.width() and mg.bottom() < win.view.height()
    # compact layout: LUT + fps controls give way to the image; the panel can shrink
    assert win.viewer.compact
    assert not any(h.isVisible() for h in win.viewer._hists.values())
    assert not win.viewer._fps_spins["t"].isVisible()
    assert win.viewer._ovl_btn.text() == "◈"           # glyph only while compact
    assert win.viewer.minimumSizeHint().width() <= win.minimap.MIN_W
    _ok(f"Maximize: canvas owns the centre; Viewer re-homed into a "
        f"{mg.width()}×{mg.height()} mini-map at ({mg.left()},{mg.top()})")

    # click-to-preview: selecting a node (what a click does) pulls it into the mini-map
    assert win._follow_act.isChecked()                 # forced on while maximized
    pulled = []
    win.runner.started.connect(lambda nid: pulled.append(nid))
    win.scene.clearSelection()
    win.scene.node_items["n5"].setSelected(True)       # "click" the label node
    t0 = time.time()
    while "n5" not in pulled and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    assert "n5" in pulled, f"click did not preview the node (pulled={pulled})"
    assert win._viewed == "n5" and win.scene.viewed_id == "n5"
    assert win.scene.node_items["n5"]._viewed          # accent spine marks the card
    assert not win.scene.node_items["n3"]._viewed
    assert "n5" in win.minimap._title
    t0 = time.time()
    while win.minimap.state == "busy" and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert win.minimap.state == "live", win.minimap.state
    _ok("Mini-map: a node click pulls it live (debounced), card + header follow")

    # a marquee across several nodes queues ONE pull (the debounce), not one per node
    n_before = len(pulled)
    win.scene.clearSelection()
    for nid in ("n2", "n3", "n4"):
        win.scene.node_items[nid].setSelected(True)
        app.processEvents()
    t0 = time.time()
    while len(pulled) == n_before and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    time.sleep(0.3)
    app.processEvents()
    assert len(pulled) - n_before == 1, f"debounce queued {len(pulled)-n_before} pulls"
    assert pulled[-1] in ("n2", "n3", "n4"), pulled[-1]
    _ok(f"Mini-map: a multi-node selection debounces to a single pull ({pulled[-1]})")

    # the mini-map moves/resizes and re-anchors, then docks back unharmed
    win.minimap.set_frame_size(300, 240)
    win.minimap.move(24, 20)
    win.minimap._reanchor()
    win.view.resize(win.view.width() - 120, win.view.height())
    app.processEvents()
    assert win.minimap.geometry().topLeft().x() == 24  # top-left anchor survives resize
    assert (win.minimap.width(), win.minimap.height()) == (300, 240)
    win.set_maximized(False)
    app.processEvents()
    assert win._center.count() == 2 and win._center.widget(0) is win.viewer
    assert win.viewer.isVisible() and not win.minimap.isVisible()
    assert not win.viewer.compact
    assert all(h.isVisible() for h in win.viewer._hists.values())
    assert win.viewer._ovl_btn.text() == "◈ Overlays"
    assert not win.view.is_maximized() and not win._max_act.isChecked()
    assert win._center.sizes() == docked_sizes, (win._center.sizes(), docked_sizes)
    _ok("Restore: Viewer docks back at its old split size, full controls returned")

    # ── G10 + screenshots (dark + light + maximized) ─────────────────────────
    app.processEvents()
    win.view.fit_all()
    app.processEvents()
    assert win.grab().save(out)
    _ok(f"screenshot (dark) {out}")

    win.set_maximized(True)
    win.pull_node("n3")
    t0 = time.time()
    while win.minimap.state == "busy" and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    win.view.fit_all()
    for _ in range(3):
        app.processEvents()
    max_out = out.replace(".png", "_maximized.png")
    assert win.grab().save(max_out)
    _ok(f"screenshot (maximized + mini-map) {max_out}")
    win.set_maximized(False)
    app.processEvents()
    win.set_theme("light")
    for _ in range(2):
        app.processEvents()
    light_out = out.replace(".png", "_light.png")
    assert win.grab().save(light_out)
    _ok(f"screenshot (light) {light_out}")

    for _ in range(4):
        app.processEvents()
    assert not _seen_fail, f"unexpected runner failures: {_seen_fail}"
    _ok("no spurious runner failures across the session")

    # ── P1: the layer picker (V2.11) ─────────────────────────────────────────
    # A source-layer socket offers the layers actually present on the incoming edge
    # instead of making the user retype a name. Editable, not a closed list: a couple of
    # producers name layers the edit-time pass cannot predict.
    from PySide6.QtCore import QPoint as _QPoint
    from PySide6.QtGui import QWheelEvent
    from nodelab_v2.document import GraphDocument as _PDoc
    from nodelab_v2.inspector import InspectorPanel as _PInsp, _NoWheelCombo
    from nodelab_v2.node_item import NodeItem as _PItem

    pdoc = _PDoc()
    pdoc.add_node("io.load", node_id="PS")
    pdoc.meta_seeds["PS"] = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16))
    pdoc.add_node("analysis.threshold", node_id="PT", params={"name": "m2"})
    pdoc.add_node("analysis.label", node_id="PL", params={"name": "regions"})
    # V2.12: `analysis.watershed` is now the `watershed` METHOD of analysis.segment, and
    # its `mask` socket is gated on that method — so the mode must be set for the picker to
    # exist at all, which is itself the check that `available_in` gating reaches the form.
    pdoc.add_node("analysis.segment", node_id="PW", modes={"method": "watershed"})
    pdoc.connect("PS", "image", "PT", "data")
    pdoc.connect("PT", "out", "PL", "data")
    pdoc.connect("PL", "out", "PW", "data")

    assert pdoc.layer_choices("PW", pdoc.nodes["PW"].spec().input("mask")) \
        == ["m2", "regions"], "picker must offer the upstream Voxel layers"
    assert "labels" not in pdoc.layer_choices(
        "PW", pdoc.nodes["PW"].spec().input("mask")), "never offer a node its own output"
    # the method gates BOTH halves of the form: the `mask` socket and the `level` Mode
    _prec, _pspec = pdoc.nodes["PW"], pdoc.nodes["PW"].spec()
    _sock_names = lambda: {s.name for s in pdoc.input_specs("PW")}
    _mode_names = lambda: {m.name for m in _pspec.active_modes(_prec.state())}
    assert "mask" in _sock_names() and "level" in _mode_names()
    _prec.modes["method"] = "cellsam"
    assert "mask" not in _sock_names(), \
        "a socket the chosen method never reads must vanish from the form"
    assert "level" not in _mode_names(), \
        "and so must a MODE the chosen method never reads (V2.12 ModeSpec.available_in)"
    _prec.modes["method"] = "watershed"
    assert "mask" in _sock_names() and "level" in _mode_names(), "gating is reversible"

    pinsp = _PInsp()
    pitem = _PItem(pdoc.nodes["PW"], pdoc)
    pinsp.set_node(pitem)
    _pick = next((c for c in pinsp.findChildren(_NoWheelCombo) if c.isEditable()
                  and [c.itemText(i) for i in range(c.count())] == ["m2", "regions"]), None)
    assert _pick is not None, "no populated layer picker in the inspector"
    # The Segmentation node's foreground socket defaults to EMPTY — unset means "segment
    # the image", and naming a layer means "split THAT foreground instead".
    assert _pick.currentText() == "", "picker starts at the socket default"
    _pick.setCurrentText("regions")
    _pick.activated.emit(_pick.findText("regions"))
    assert pdoc.nodes["PW"].params.get("mask") == "regions", "picking commits"
    _pick.setEditText("typed_by_hand")
    _pick.lineEdit().editingFinished.emit()
    assert pdoc.nodes["PW"].params.get("mask") == "typed_by_hand", \
        "free text must still commit — an unpredictable layer name is never blocked"

    # Wheel safety. A plain QComboBox CHANGES VALUE and eats the event on an unfocused
    # wheel notch (measured on PySide6 6.10.2), and the inspector is a fixed-height
    # scroll area, so scrolling past a combo used to silently rewrite and pin a param.
    pdoc.nodes["PW"].params["mask"] = "regions"
    _pick.setCurrentText("regions")
    _pick.clearFocus()
    for _ in range(3):
        _wev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), _QPoint(0, -120), _QPoint(0, -120),
                           Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(_pick, _wev)
        assert not _wev.isAccepted(), "an unfocused combo must let the panel scroll"
    assert pdoc.nodes["PW"].params.get("mask") == "regions", \
        "a wheel over an unfocused picker must not change the value"
    _mode = next((c for c in pinsp.findChildren(_NoWheelCombo)
                  if not c.isEditable()), None)
    if _mode is not None:
        _before = dict(pdoc.nodes["PW"].modes)
        _mode.clearFocus()
        _wev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), _QPoint(0, -120), _QPoint(0, -120),
                           Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(_mode, _wev)
        assert pdoc.nodes["PW"].modes == _before, \
            "the same guard must cover the Mode dropdowns (this bug pre-dated the picker)"
    pinsp.set_node(None)
    pinsp.setParent(None)
    _ok("P1 layer picker: offers the upstream layers, commits by pick AND by free text, "
        "never offers a node its own output, and a wheel over an unfocused combo is inert "
        "(fixes a pre-existing Mode-dropdown bug too)")

    # ── T1: the solo-frame troubleshooting scope (F9) ──────────────────────────
    # Needs a source with real frames: the synthetic fallback is t==1, which would make
    # every assertion below vacuously true. Re-point it and drop the cached provider so
    # the next pull re-resolves at the new T (and re-announces its envelope to the
    # document, which is where the frame chooser's extent comes from).
    from nodelab_v2 import runner as _RN
    _RN._SYNTH_AXES = AxisSizes(m=1, t=6, z=3, c=2, y=128, x=128)
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner.invalidate()
    win.build_demo()                                   # known ids again (groups renamed them)
    win.doc.set_meta_seed("n1", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                             metadata=dict(_RN._SYNTH_META)))
    app.processEvents()

    _pulls = []
    win.runner.finished.connect(lambda nid, payload, *a: _pulls.append((nid, payload)))

    def _await_pull(n: int = 1, limit: float = 180.0):
        want = len(_pulls) + n
        t0 = time.time()
        while len(_pulls) < want and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.005)
        assert len(_pulls) >= want, f"pull did not land in {limit}s"
        return _pulls[-1][1]

    win.viewer._sliders["t"].setValue(0)
    app.processEvents()
    _pulls.clear()                                     # a slider move may have queued one
    win.pull_node("n5")                                # label: an EAGER per-frame node
    full = _await_pull()
    assert full.axes.t == 6, f"baseline must see the whole series, got t={full.axes.t}"
    assert win.viewer._sliders["t"].maximum() == 5
    assert not win.runner.solo_frame and win.viewer.solo is None
    assert not win._solo_chip.isVisible()

    win.set_solo_frame(True)                           # ← what F9 / the Run menu does
    solo_t0 = _await_pull()
    assert win.runner.solo_frame and win._solo_act.isChecked()
    assert win.viewer.solo == (1, 6, 3), win.viewer.solo  # chooser spans the SOURCE extent
    assert win.viewer._sliders["t"].maximum() == 5, "the frame chooser must not collapse"
    assert win._solo_chip.isVisible() and win._solo_chip.text() == "SOLO t0"
    assert solo_t0.axes.t == 1 and solo_t0.axes.m == 1, \
        f"a soloed pull must carry ONE frame, got {solo_t0.axes}"
    assert solo_t0.axes.z == 3 and solo_t0.axes.c == 2, "z/c must pass through untouched"
    assert " · solo t0" in win.viewer._status.text(), win.viewer._status.text()

    # moving the frame chooser re-runs the graph on THAT frame (the fast plane path is
    # correctly bypassed: a neighbouring frame is not in this payload at all)
    win.viewer._sliders["t"].setValue(4)
    solo_t4 = _await_pull()
    assert solo_t4.axes.t == 1
    assert win.viewer._sliders["t"].value() == 4 and win._solo_chip.text() == "SOLO t4"
    solo_pixels = {c: np.array(p) for c, p in win.viewer._planes.items()}

    # returning to a frame already run is a MEMO HIT — this is what makes flipping
    # between two frames instant, and it proves the pin re-keys rather than aliases
    win.viewer._sliders["t"].setValue(0)
    _await_pull()
    win.viewer._sliders["t"].setValue(4)
    _await_pull()
    assert win._run_cached > 0 and win._run_computed == 0, \
        f"revisiting a computed frame must be all cache ({win._run_computed} computed)"

    # ...and it is the RIGHT frame: the same node pulled UNSOLOED at t=4 shows the same
    # pixels (the synthetic source varies with t, so a mis-pinned read would differ).
    win.set_solo_frame(False)
    full_again = _await_pull()
    assert full_again.axes.t == 6 and win.viewer.solo is None
    assert not win._solo_chip.isVisible()
    assert win.viewer._sliders["t"].value() == 4, "leaving the scope keeps the frame"
    for c, plane in solo_pixels.items():
        assert np.array_equal(plane, np.asarray(win.viewer._planes[c])), \
            f"soloed frame 4 must be the same pixels as frame 4 of the full series (c={c})"
    _ok("T1 solo frame (F9): 6-frame source → a soloed pull carries t==1 (z/c intact) at "
        "the chooser's frame, the chooser keeps spanning the series, an M/T move re-runs "
        "and a revisit is all memo, and the soloed pixels equal that frame of the full "
        "series")

    # ── T2: PICKED frames — the multi-frame scope a temporal node needs ─────────
    # The T strip's boxes are pickable; the picks (not the cursor) become the run scope,
    # so a tracker downstream sees a real, if short, series instead of one frame.
    _tstrip = win.viewer._sliders["t"]
    assert win.viewer.frame_selection() == ((), (), ()), "nothing picked to start"

    win.set_solo_frame(True)
    _await_pull()
    _tstrip.setSelection((1, 3, 4))                    # ← what ctrl+click on 3 boxes does
    picked = _await_pull()
    assert win.viewer.frame_selection() == ((), (1, 3, 4), ())
    assert picked.axes.t == 3 and picked.axes.m == 1, \
        f"a picked pull must carry the picked frames, got {picked.axes}"
    assert picked.axes.z == 3 and picked.axes.c == 2, \
        "an UNPICKED z is the whole volume (a 3D node needs one) and c is untouched"
    assert win._solo_chip.text() == "SOLO 3T[1, 3–4]", win._solo_chip.text()
    assert win.runner.frame_selection == ((), (1, 3, 4), ())
    # the cursor is at t=4, which IS picked → it addresses the LAST payload frame
    assert win.viewer._sliders["t"].value() == 4
    assert win.viewer._payload_coords()[1] == 2, win.viewer._payload_coords()
    # ...and those pixels are frame 4's, not the payload's frame 0
    for c, plane in solo_pixels.items():
        assert np.array_equal(plane, np.asarray(win.viewer._planes[c])), \
            f"picked frame 4 must still show frame 4's pixels (c={c})"
    # a cursor moved ONTO an unpicked frame does not re-scope (the picks did that) — it
    # shows the nearest picked frame instead of reading off the end of the payload
    win.viewer._sliders["t"].setValue(2)
    app.processEvents()
    assert win.viewer._payload_coords()[1] == 0, "t=2 → the nearest picked frame (t=1)"
    assert win.runner.frame_selection == ((), (1, 3, 4), ()), \
        "scrubbing never edits the picks"

    # clearing the picks falls back to the single-frame scope on the cursor's own frame
    _pulls.clear()
    win.clear_frame_picks()
    cleared = _await_pull()
    assert cleared.axes.t == 1 and win.viewer.frame_selection() == ((), (), ())
    assert win._solo_chip.text() == "SOLO t2", win._solo_chip.text()
    _ok("T2 picked frames: ctrl-picking 3 T boxes scopes the pull to t==3 (the picked "
        "frames, whole volume, c intact) so a temporal node has a series to work on; the "
        "chip and runner follow the picks, the cursor still addresses the right payload "
        "frame (an unpicked one resolves to its nearest picked neighbour), scrubbing "
        "never edits the picks, and clearing them returns to the one-frame scope")

    # ── T3: PICKED planes — z cuts every scoped frame, and the axes compose ─────
    # Z picks are the same mechanism one axis over: they subset z INSIDE each scoped
    # frame, so "3 timepoints × 2 planes" is 3 frames of 2 planes, not 6 of anything.
    _zstrip = win.viewer._sliders["z"]
    _pulls.clear()
    _zstrip.setSelection((0, 2))                       # ← ctrl+click 2 of the 3 z boxes
    zpicked = _await_pull()
    assert win.viewer.frame_selection() == ((), (), (0, 2))
    assert zpicked.axes.z == 2, f"a z pick must shorten the volume, got {zpicked.axes}"
    assert zpicked.axes.t == 1, "…without disturbing the frame scope (cursor's t only)"
    assert win.viewer._sliders["z"].maximum() == 2, \
        "the z strip keeps spanning the SOURCE volume — it is what picks the planes"
    assert win._solo_chip.text() == "SOLO t2·2Z", win._solo_chip.text()

    _tstrip.setSelection((1, 3, 4))                    # both axes at once → cross product
    both = _await_pull()
    assert (both.axes.t, both.axes.z) == (3, 2), f"3 frames × 2 planes, got {both.axes}"
    assert win._solo_chip.text() == "SOLO 3T[1, 3–4]·2Z", win._solo_chip.text()
    # the z cursor addresses the payload the same way t does: picked → its position,
    # unpicked → the nearest picked plane
    win.viewer._sliders["z"].setValue(2)
    app.processEvents()
    assert win.viewer._payload_coords()[2] == 1, "z=2 is the SECOND picked plane"
    win.viewer._sliders["z"].setValue(1)
    app.processEvents()
    assert win.viewer._payload_coords()[2] == 0, "z=1 → the nearest picked plane (z=0)"
    # …and the scoped pixels are the real ones. Checked on the SOURCE (n1, a pure read),
    # NOT on the analysis chain: n3 is a 3D Gaussian, so cutting z legitimately changes
    # what it computes — that is the mode's documented cost, and an equality check there
    # would be asserting a falsehood. The cursor sits at t=2 / z=1, neither picked, so
    # what is on screen is the nearest picked pair: source frame t=1, plane z=0.
    _pulls.clear()
    win.pull_node("n1")
    src_scoped = _await_pull()
    assert (src_scoped.axes.t, src_scoped.axes.z) == (3, 2), src_scoped.axes
    _zpix = {c: np.array(p) for c, p in win.viewer._planes.items()}
    win.set_solo_frame(False)
    full_z = _await_pull()
    assert full_z.axes.t == 6 and full_z.axes.z == 3, "leaving the scope restores both"
    win.viewer._sliders["t"].setValue(1)               # fast plane path (same payload)
    win.viewer._sliders["z"].setValue(0)
    app.processEvents()
    for c, plane in _zpix.items():
        assert np.array_equal(plane, np.asarray(win.viewer._planes[c])), \
            f"the scoped plane must be the source's own (t1, z0) (c={c})"
    win.viewer.clear_frame_selection()
    app.processEvents()
    _ok("T3 picked planes: a z pick cuts every scoped frame to those planes (z==2) "
        "without touching the frame scope, and picking T and Z together gives their "
        "cross product (3 frames × 2 planes); the z strip keeps spanning the source, the "
        "z cursor addresses the payload like t does (nearest picked plane when unpicked), "
        "the chip names the cut, and a scoped SOURCE plane is byte-identical to that "
        "plane of the unscoped series")

    # ── T4: the canvas HUD — you cannot miss that a run is scoped ───────────────
    # A scoped pull leaves the graph, the cards and the progress rails looking IDENTICAL
    # to a full run, so the canvas itself has to say otherwise: an amber frame + a badge.
    _view = win.view
    assert not _view.is_troubleshooting() and not _view._ts_badge.isVisible()
    assert not _view._ts_timer.isActive(), "the pulse must not tick while the mode is off"

    def _left_border_amber() -> bool:
        """Is the canvas wearing its amber frame? Sampled off a real widget grab, so this
        exercises `drawForeground` rather than a flag."""
        img = _view.grab().toImage()
        want = T.DIM2D
        for y in range(int(img.height() * 0.3), int(img.height() * 0.7), 7):
            col = img.pixelColor(_view.TS_INSET, y)
            if (abs(col.red() - want.red()) + abs(col.green() - want.green())
                    + abs(col.blue() - want.blue())) < 60:
                return True
        return False

    assert not _left_border_amber(), "no frame while the mode is off"
    _pulls.clear()
    win.set_solo_frame(True)
    _await_pull()
    app.processEvents()
    assert _view.is_troubleshooting() and _view._ts_badge.isVisible()
    assert _view._ts_timer.isActive(), "the badge dot pulses while the mode is on"
    assert "TROUBLESHOOTING MODE" in _view._ts_badge.text()
    assert win._scope_tag() in _view._ts_badge.text(), \
        "the badge names the same scope as the status chip"
    assert _left_border_amber(), "the canvas must wear the amber frame"
    _bg = _view._ts_badge.geometry()
    assert _bg.left() < 40 and _bg.top() < 40, f"badge belongs top-left, at {_bg}"

    # the pulse only re-renders the badge's TEXT (a 200 px label); a pulsing frame would
    # repaint every node on the canvas twice a second
    _before = _view._ts_badge.text()
    _view._ts_pulse()
    assert _view._ts_badge.text() != _before and \
        "TROUBLESHOOTING MODE" in _view._ts_badge.text()

    # maximized: the mini-map owns that corner, so the badge steps aside — to its right,
    # still on the top strip — instead of hiding under it
    win.set_maximized(True)
    for _ in range(4):
        app.processEvents()
    _bg, _mm = _view._ts_badge.geometry(), win.minimap.geometry()
    assert win.minimap.isVisible() and not _bg.intersects(_mm), \
        f"badge {_bg} must dodge the mini-map {_mm}"
    assert _bg.left() >= _mm.right() and _bg.top() < _mm.top() + 60, \
        "…by moving right of it, not off the top strip"
    assert _left_border_amber(), "the frame survives the maximize"
    win.set_maximized(False)
    for _ in range(3):
        app.processEvents()
    assert _view._ts_badge.geometry().left() < 40, "…and returns to the corner after"

    _pulls.clear()
    win.set_solo_frame(False)
    _await_pull()
    app.processEvents()
    assert not _view.is_troubleshooting() and not _view._ts_badge.isVisible()
    assert not _view._ts_timer.isActive(), "leaving the mode stops the pulse (no idle cost)"
    assert not _left_border_amber(), "and takes the frame with it"
    _ok("T4 canvas HUD: arming the scope paints an amber frame around the graph and a "
        "pulsing top-left TROUBLESHOOTING MODE badge naming the same scope as the chip; "
        "the badge dodges the mini-map (right of it, still top strip) and returns; "
        "leaving the mode removes frame, badge and pulse timer")

    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)   # restore the fallback

    # ── P2: interactive parameters (V2.16) ────────────────────────────────────
    # Both surfaces that can arm a pick (the inspector button, a card's ◎ glyph) must reach
    # the viewer, and the committed value must land on the node PINNED — the arithmetic is
    # covered headlessly by nodegraph.selftest.test_picking; what only a live window can
    # show is that the widgets, signals and document writes are actually connected.
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent, QPainter, QPixmap
    from PySide6.QtWidgets import QCheckBox as _QCheck, QToolButton as _QTool
    from nodelab_v2.picker import Calibration as _Cal, PickRequest as _PickReq

    # (0) BOTH image backends must implement the overlay/pick contract. The panel probes
    # every one of these with getattr, so an omission disables a whole interaction on one
    # backend and nothing says so — which is exactly what happened: `widget_to_plane` went
    # onto the CPU view only, and every on-image gesture was dead on the GPU path (the
    # default in a windowed session) while these offscreen tests, which force NODELAB_GL=0,
    # kept passing. Checked on the CLASSES so no GL context is needed here.
    from nodelab_v2.glview import GLImageView as _GLV
    from nodelab_v2.viewer import _ImageView as _CPUV, check_surface_contract
    check_surface_contract(_CPUV, _GLV)

    # …and the GPU mapping must really be the inverse of its own forward map, under zoom
    # and pan, because that is the arithmetic the dead path was missing.
    _gv = _GLV(); _gv.resize(800, 400); _gv._img_wh = (64, 48)
    _gv._zoom, _gv._pan = 2.5, QPointF(31.0, -17.0)
    for _px, _py in ((0.0, 0.0), (32.0, 24.0), (63.5, 47.5)):
        _rt = _gv.widget_to_plane(_gv.plane_to_widget(_px, _py))
        assert _rt is not None and abs(_rt[0] - _px) < 1e-6 and abs(_rt[1] - _py) < 1e-6, \
            f"GL widget_to_plane is not the inverse of plane_to_widget at {(_px, _py)}"
    # At the default framing a 64×48 image is letterboxed inside an 800×400 pane, so the
    # left margin is genuinely outside the picture — the case that produced ROI shapes with
    # negative vertices. (Under zoom the image can legitimately extend past the widget, so
    # the point has to be chosen from the real geometry, not guessed.)
    _gv._zoom, _gv._pan = 1.0, QPointF(0.0, 0.0)
    _ox, _oy, _dw, _dh = _gv._disp_rect()
    assert _ox > 2.0, f"expected a letterbox margin to test against, got ox={_ox}"
    assert _gv.widget_to_plane(QPointF(_ox - 2.0, _oy + _dh / 2)) is None, \
        "a point in the letterbox margin must be refused, not extrapolated"
    assert _gv.widget_to_plane(QPointF(_ox + _dw / 2, _oy + _dh / 2)) is not None, \
        "…while a point on the image itself must still map"
    _gv.deleteLater()

    # (0b) …and a REAL mouse drag on the live surface must reach the session. Every other
    # check here drives `session.press()` directly, which is precisely how the dead event
    # path went unnoticed — so this one goes through the widget.
    _pv = win.viewer
    _pv.cancel_pick()
    _pv.arm_pick(_PickReq(node_id="QQ", socket="shapes", kind="shapes"), _Cal(um_px=1.0))
    _surf = _pv._pick_targets()[-1]
    _ctr = _surf.rect().center()
    for _t, _dx, _dy, _btn in (
            (QEvent.MouseButtonPress, -20, -15, Qt.LeftButton),
            (QEvent.MouseMove, 25, 18, Qt.LeftButton),
            (QEvent.MouseButtonRelease, 25, 18, Qt.LeftButton)):
        _pt = QPointF(_ctr.x() + _dx, _ctr.y() + _dy)
        app.sendEvent(_surf, QMouseEvent(_t, _pt, _pt, Qt.LeftButton, _btn, Qt.NoModifier))
    assert _pv._pick is not None and len(_pv._pick.shapes) == 1, \
        f"a real drag on the image produced no shape: {_pv._pick and _pv._pick.shapes}"
    _verts = _pv._pick.shapes[0]["vertices"]
    assert all(v >= 0 for pair in _verts for v in pair), \
        f"a drag near the image centre produced out-of-image vertices: {_verts}"
    # and the gesture must actually paint something
    _canvas = QPixmap(_surf.width(), _surf.height())
    _canvas.fill(Qt.black)
    _pp = QPainter(_canvas)
    _pv._paint_overlays(_pp)          # the same callback both surfaces invoke
    _pp.end()
    _img = _canvas.toImage()
    _lit = sum(1 for _yy in range(0, _img.height(), 3) for _xx in range(0, _img.width(), 3)
               if _img.pixelColor(_xx, _yy).value() > 40)
    assert _lit > 20, f"the armed gesture painted nothing ({_lit} lit samples)"
    _pv.cancel_pick()

    pdoc2 = _PDoc()
    pdoc2.add_node("io.load", node_id="QS")
    pdoc2.meta_seeds["QS"] = MetaEnvelope(
        axes=AxisSizes(m=1, t=4, z=3, c=2, y=64, x=64),
        metadata={"pixel_size_um": 0.5, "z_step_um": 2.0,
                  "channel_names": ["DAPI", "GFP"],
                  "channel_emission_nm": [461, 509]})
    pdoc2.add_node("analysis.segment", node_id="QG", modes={"method": "watershed"})
    pdoc2.connect("QS", "image", "QG", "data")
    assert pdoc2.env("QG").metadata.get("pixel_size_um") == 0.5, \
        "the pick's calibration comes from the node's PROPAGATED envelope"

    qinsp = _PInsp()
    qitem = _PItem(pdoc2.nodes["QG"], pdoc2)
    qinsp.set_node(qitem)
    _qreq = []
    qinsp.pick_requested.connect(_qreq.append)
    _pbtns = [b for b in qinsp.findChildren(_QTool) if b.property("role") == "pick"]
    assert _pbtns, "a declared pick_kind must put a Pick button in the inspector"
    _ruler = [b for b in _pbtns if "Measure" in b.text()]
    assert _ruler, f"watershed's min_distance should offer the ruler; got {[b.text() for b in _pbtns]}"
    _ruler[0].click()
    assert _qreq and _qreq[-1].socket == "min_distance" and _qreq[-1].kind == "distance", _qreq

    # the same request from the CARD's ◎ glyph — same builder, same target
    _gl = [c for c in qitem.controls() if c.kind == "pick"]
    assert _gl, "a pickable param must draw a ◎ on the card"
    _qcard = []
    qitem.pick_requested.connect(_qcard.append)
    _minarea = next(c for c in _gl if c.obj.name == "min_area")
    qitem.mousePressEvent(_FakePress(_minarea.rect.center()))
    assert _qcard and _qcard[-1].socket == "min_area" and _qcard[-1].peer == "max_area", _qcard
    assert all(qitem.card_rect().contains(c.rect.center()) for c in qitem.controls()), \
        "every clickable control must lie inside the card it is painted on"

    # arm on the REAL viewer and drive the gesture → the value must land, pinned
    win.viewer.cancel_pick()
    _seg = win.doc.add_node("analysis.segment", node_id="QQ",
                            modes={"method": "watershed"}, x=1200, y=520)
    win.scene.sync()
    win._arm_pick(_qreq[-1].__class__(**{**_qreq[-1].__dict__, "node_id": "QQ"}))
    assert win.viewer.picking(), "the window must arm the viewer"
    _sess = win.viewer._pick
    _sess.calib = _Cal(um_px=0.5, um_z=2.0)
    _sess.press(10, 10); _sess.release(10, 10)
    _sess.press(50, 10); _sess.release(50, 10)
    win.viewer._apply_pick()
    app.processEvents()
    assert _seg.params.get("min_distance") == 20.0, _seg.params
    assert "min_distance" in _seg.locked, "a picked value must be PINNED, like a typed one"
    assert not win.viewer.picking(), "committing disarms"

    # Esc backs out without writing
    _before = dict(_seg.params)
    win._arm_pick(_qreq[-1].__class__(**{**_qreq[-1].__dict__, "node_id": "QQ"}))
    _sess = win.viewer._pick
    _sess.press(0, 0); _sess.release(0, 0)
    win.viewer.cancel_pick()
    assert not win.viewer.picking() and _seg.params == _before, "Esc must not commit"

    # a histogram pick reads the live LUT window
    win.viewer._clim[(win.viewer._node_id or "", 0)] = (200.0, 800.0)
    win.viewer._drange[(win.viewer._node_id or "", 0)] = (0.0, 1000.0)
    win.doc.add_node("enhance.normalize", node_id="QN", x=1200, y=660)
    win.scene.sync()
    from nodelab_v2.picker import request_for as _rfor
    _nspec = win.doc.nodes["QN"].spec()
    win._arm_pick(_rfor("QN", _nspec.input("low_pct"), _nspec.input("high_pct")))
    assert win.viewer.picking() and win.viewer._pick.req.surface == "histogram"
    win.viewer._apply_pick()
    app.processEvents()
    assert win.doc.nodes["QN"].params.get("low_pct") == 20.0, win.doc.nodes["QN"].params
    assert win.doc.nodes["QN"].params.get("high_pct") == 80.0, win.doc.nodes["QN"].params

    # an INSTANT pick never arms — there is nothing to aim
    win.doc.add_node("analysis.dvc_field", node_id="QD", x=1200, y=760)
    win.scene.sync()
    win.viewer._sliders["t"].setValue(2)
    win._arm_pick(_rfor("QD", win.doc.nodes["QD"].spec().input("reference_frame")))
    assert not win.viewer.picking(), "an instant pick commits without a bar"
    assert win.doc.nodes["QD"].params.get("reference_frame") == 2, win.doc.nodes["QD"].params

    # closed `choices` — a dropdown, not free text
    pdoc2.nodes["QG"].modes["method"] = "stardist"
    qinsp.set_node(_PItem(pdoc2.nodes["QG"], pdoc2))
    _sd = next((c for c in qinsp.findChildren(_NoWheelCombo)
                if "2D_versatile_fluo" in [c.itemText(i) for i in range(c.count())]), None)
    assert _sd is not None and not _sd.isEditable(), \
        "a published checkpoint set must be a CLOSED dropdown — a typo is a 404 mid-pull"

    # `vocab` — a tick list that round-trips through the comma string the compute parses
    pdoc2.add_node("analysis.measure", node_id="QM")
    pdoc2.connect("QG", "out", "QM", "data")
    qinsp.set_node(_PItem(pdoc2.nodes["QM"], pdoc2))
    _ticks = {c.text(): c for c in qinsp.findChildren(_QCheck)}
    assert {"mean", "median", "eccentricity"} <= set(_ticks), sorted(_ticks)
    assert _ticks["mean"].isChecked() and not _ticks["median"].isChecked(), \
        "the tick list must start from the socket default"
    _ticks["median"].setChecked(True)
    _stats = pdoc2.nodes["QM"].params.get("stats", "")
    assert "median" in _stats.split(","), _stats
    from nodegraph.nodes import _measure_stats as _ms
    assert "median" in _ms(_stats), "the ticked string must survive the compute's parser"

    # the channel tick list names real channels instead of asking for indices
    pdoc2.add_node("channel.select", node_id="QC")
    pdoc2.connect("QS", "image", "QC", "data")
    qinsp.set_node(_PItem(pdoc2.nodes["QC"], pdoc2))
    _chk = [c.text() for c in qinsp.findChildren(_QCheck)]
    assert any("DAPI" in t for t in _chk) and any("GFP" in t for t in _chk), _chk
    _gfp = next(c for c in qinsp.findChildren(_QCheck) if "GFP" in c.text())
    _dapi = next(c for c in qinsp.findChildren(_QCheck) if "DAPI" in c.text())
    assert _dapi.isChecked() and _gfp.isChecked(), \
        "an empty selector means KEEP ALL, so every box starts ticked"
    _dapi.setChecked(False)
    assert _gfp.isChecked() and pdoc2.nodes["QC"].params.get("channels") == "1", \
        pdoc2.nodes["QC"].params.get("channels")

    # on-canvas editing: scrub a float, toggle a bool, and pin/unpin from the ƒmd badge
    qinsp.set_node(None)
    pdoc2.add_node("enhance.gaussian", node_id="QB")
    pdoc2.connect("QS", "image", "QB", "data")
    gitem = _PItem(pdoc2.nodes["QB"], pdoc2)
    _sig = next(c for c in gitem.controls()
                if c.kind == "value" and c.obj.name == "sigma")
    gitem.mousePressEvent(_FakePress(_sig.rect.center()))
    gitem.mouseMoveEvent(_FakeMove(_sig.rect.center().x() + 30))
    gitem.mouseReleaseEvent(_FakeRelease())
    assert abs(float(pdoc2.nodes["QB"].params["sigma"]) - 0.6) < 1e-9, \
        pdoc2.nodes["QB"].params
    assert "sigma" in pdoc2.nodes["QB"].locked, "a canvas scrub pins, like a typed edit"
    gitem.mousePressEvent(_FakePress(_sig.rect.center()))
    gitem.mouseMoveEvent(_FakeMove(_sig.rect.center().x() - 9999))
    gitem.mouseReleaseEvent(_FakeRelease())
    assert float(pdoc2.nodes["QB"].params["sigma"]) == 0.0, "a scrub clamps at zero"

    sitem = _PItem(pdoc2.nodes["QG"], pdoc2)
    _fh = next(c for c in sitem.controls()
               if c.kind == "value" and c.obj.name == "fill_holes")
    assert not sitem.resolved(_fh.obj)
    sitem.mousePressEvent(_FakePress(_fh.rect.center()))
    assert pdoc2.nodes["QG"].params.get("fill_holes") is True, "a bool pill toggles"

    # a derived param's ƒmd badge pins the derived value and un-pins back to auto
    pdoc2.add_node("detect.spots", node_id="QP")
    pdoc2.connect("QS", "image", "QP", "data")
    ditem = _PItem(pdoc2.nodes["QP"], pdoc2)
    _pin = next((c for c in ditem.controls() if c.kind == "pin"), None)
    assert _pin is not None, "a metadata-derived param must expose its ƒmd badge on the card"
    _auto = ditem.resolved(_pin.obj)
    ditem.mousePressEvent(_FakePress(_pin.rect.center()))
    assert _pin.obj.name in pdoc2.nodes["QP"].locked and \
        pdoc2.nodes["QP"].params.get(_pin.obj.name) == _auto, "the badge pins the live value"
    ditem = _PItem(pdoc2.nodes["QP"], pdoc2)
    _pin = next(c for c in ditem.controls() if c.kind == "pin")
    ditem.mousePressEvent(_FakePress(_pin.rect.center()))
    assert _pin.obj.name not in pdoc2.nodes["QP"].locked, "and un-pins back to auto"

    # (12) the crop rectangle, end to end through a REAL pull: one drag must produce a
    # cropped output whose pixels are byte-identical to the window that was drawn. This is
    # the check that matters most for `rect` — the four bounds are a slice with an exclusive
    # end, so an off-by-one would still look plausible in the pills while quietly shifting
    # the data. Nothing short of comparing pixels catches that.
    # A source big enough to crop a 240×200 window out of (the T-section shrank it to 128²).
    _RN._SYNTH_AXES = AxisSizes(m=1, t=2, z=5, c=1, y=512, x=512)
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner.invalidate()
    win.doc.add_node("util.crop", node_id="QK", x=1200, y=300)
    win.doc.connect("n1", "image", "QK", "data")
    win.doc.set_meta_seed("n1", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                            metadata=dict(_RN._SYNTH_META)))
    win.scene.sync()
    app.processEvents()
    win.pull_node("QK")
    _uncropped = _await_pull()
    _sy, _sx = _uncropped.axes.y, _uncropped.axes.x
    assert _sy >= 260 and _sx >= 340, f"source too small for the crop check ({_sy}×{_sx})"
    _src_plane = np.asarray(next(iter(win.viewer._planes.values())))
    win.scene.clearSelection(); win.scene.node_items["QK"].setSelected(True)
    app.processEvents()
    _cb = [b for b in win.inspector.findChildren(_QTool)
           if b.property("role") == "pick" and "crop rectangle" in b.text()]
    assert _cb, "util.crop must offer the rectangle pick"
    _cb[0].click()
    _cs = win.viewer._pick
    assert _cs is not None and _cs.req.bounds == ("y0", "y1", "x0", "x1"), _cs and _cs.req
    _cs.press(100, 60); _cs.drag(200, 150); _cs.release(340, 260)
    win.viewer._apply_pick()
    app.processEvents()
    assert {k: v for k, v in win.doc.nodes["QK"].params.items() if not k.startswith("__")} \
        == {"y0": 60, "y1": 260, "x0": 100, "x1": 340}, win.doc.nodes["QK"].params
    assert {"y0", "y1", "x0", "x1"} <= win.doc.nodes["QK"].locked, "a picked box must pin"
    win.pull_node("QK")
    _cropped = _await_pull()
    assert (_cropped.axes.y, _cropped.axes.x) == (200, 240), \
        f"the drawn 240×200 box produced {_cropped.axes.x}×{_cropped.axes.y}"
    _crop_plane = np.asarray(next(iter(win.viewer._planes.values())))
    assert np.array_equal(_crop_plane, _src_plane[60:260, 100:340]), \
        "the cropped pixels are not the window that was drawn (off-by-one in the bounds?)"
    # the Z range is a SEPARATE, 3D-only instant pick off the Z strip's multi-select
    win.doc.nodes["QK"].modes["dim"] = "3D"
    win.doc.touch(); win.scene.sync()
    win.scene.clearSelection(); win.scene.node_items["QK"].setSelected(True)
    app.processEvents()
    win.viewer._sliders["z"].setSelection((1, 2, 3))
    _zb = [b for b in win.inspector.findChildren(_QTool)
           if b.property("role") == "pick" and "Z planes" in b.text()]
    assert _zb, "3D crop must offer the Z-range pick"
    _zb[0].click()
    app.processEvents()
    assert (win.doc.nodes["QK"].params.get("z0"),
            win.doc.nodes["QK"].params.get("z1")) == (1, 4), win.doc.nodes["QK"].params
    win.viewer.clear_frame_selection()      # the Z window is a PARAM now, not a run scope
    app.processEvents()
    win.pull_node("QK")
    assert _await_pull().axes.z == 3, "the picked Z planes did not become the Z window"
    win.doc.nodes["QK"].modes["dim"] = "2D"
    win.doc.touch()
    app.processEvents()

    _ok("P2 interactive params (V2.16): BOTH image backends implement the overlay/pick "
        "contract and the GPU map round-trips under zoom+pan (rejecting the letterbox "
        "margin); a REAL mouse drag on the live surface produces a shape with in-image "
        "vertices and paints; Pick buttons + card ring glyphs arm the viewer with "
        "the node's propagated calibration and commit PINNED (ruler 40px@0.5um -> 20um); "
        "Esc backs out clean; the histogram surface commits the live LUT window; an "
        "instant pick needs no bar; closed checkpoint dropdown; vocab ticks round-trip "
        "through the compute's own parser; channels tick by real name; ONE crop drag writes "
        "all four bounds and the PULLED pixels are byte-identical to the drawn window "
        "(240x200), with the 3D-only Z range taken from the Z strip's picks; and on the "
        "canvas a float scrubs (clamped), a bool toggles and the fmd badge pins/unpins")

    # ── P3: the hover readout ───────────────────────────────────────────────────
    # Hovering a pixel must answer four things at once: where it is in the node's grid,
    # where that is in microns, where it is on the STAGE, and what every shown channel
    # reads there — the viewed node's number and the untouched file's number beside it.
    # Driven through a real QMouseEvent on the live surface, because the whole feature is
    # an event path: the panel used to install its filter only while a pick was armed, so
    # a readout wired to that filter would be silently dead in normal use.
    _RN._SYNTH_AXES = AxisSizes(m=3, t=1, z=1, c=2, y=200, x=240)
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner._raw_src.clear()
    win.runner.invalidate()
    # a per-position stage log, the way an ND2's XYPosLoop supplies one (ingest.STAGE_KEYS)
    _STAGE = [(1000.0, -2000.0), (1024.0, -2000.0), (1048.0, -2000.0)]
    win.doc.set_meta_seed("n1", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                             metadata=dict(_RN._SYNTH_META)))
    win.scene.sync()
    app.processEvents()
    win.pull_node("n1")
    _await_pull()
    # The synthetic source carries no position log of its own — supply one on the SAME
    # channel a real file's arrives on (the display dict the runner merges into the seed
    # dataset). Injected AFTER the first pull and without clearing `_providers`: a cleared
    # provider re-resolves, and re-resolving rewrites `_channel_display` from the source.
    win.runner._channel_display[("synthetic",)]["stage_xy_um"] = _STAGE
    # display metadata is not part of any recipe hash (by design — it must never re-key a
    # memo), so the memoized payload has to be dropped for the fresh seed to be seen
    win.runner._memo.clear()
    win.runner.invalidate()
    win.viewer._sliders["m"].setValue(1)
    app.processEvents()
    win.pull_node("n1")
    _await_pull()
    assert win.viewer._dataset.metadata.get("stage_xy_um") == _STAGE, \
        "the stage log must ride on the payload the Viewer reads"

    from nodelab_v2.viewer import _HOVER_HINT
    _hsurf = win.viewer._pick_targets()[-1]

    def _hover_at(px: float, py: float) -> str:
        """Point the REAL mouse at plane pixel (px, py) and return the readout line."""
        wpt = win.viewer._view.plane_to_widget(px, py)
        app.sendEvent(_hsurf, QMouseEvent(QEvent.MouseMove, wpt, wpt,
                                          Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        app.processEvents()
        return win.viewer._hover_lbl.text()

    _cx, _cy = (_RN._SYNTH_AXES.x - 1) / 2.0, (_RN._SYNTH_AXES.y - 1) / 2.0
    # (1a) the EVENT path: a real mouse move must fill the line at all. (The widget
    # round-trip quantizes to whole widget pixels — the CPU view maps through
    # `mapToScene(int(x), int(y))` — so the arithmetic itself is checked at (1b), on the
    # exact plane coordinate, rather than through a coordinate the mouse cannot express.)
    _evt = _hover_at(_cx, _cy)
    assert " px" in _evt and "µm" in _evt and "stage " in _evt, \
        f"a real mouse move over the image produced no readout: {_evt!r}"

    # (1b) the field CENTRE is exactly the logged stage position — the one point where the
    # offset arithmetic cannot hide a sign error or an off-by-half-a-field
    _mid = win.viewer._hover_text((_cx, _cy))
    assert "stage 1024.0, -2000.0" in _mid, f"stage centre is wrong: {_mid!r}"
    assert "x 119, y 99 px" in _mid, _mid
    _plane0 = win.viewer._planes[sorted(win.viewer._planes)[0]]
    _v0 = _plane0[int(round(_cy)), int(round(_cx))]
    assert f"{int(_v0)}" in _mid, f"the pixel value {_v0} is missing from {_mid!r}"
    assert "(raw" not in _mid, \
        f"viewing the LOAD itself, displayed IS raw — a (raw …) here implies a step " \
        f"that never happened: {_mid!r}"
    # every SHOWN channel is read, not just the LUT's one. (A channel toggle takes the
    # decoded-plane fast path — `plane_ready`, not `finished` — so it is awaited on the
    # planes themselves rather than through `_await_pull`.)
    def _await_planes(n: int, limit: float = 60.0) -> None:
        _t0 = time.time()
        while len(win.viewer._planes) != n and time.time() - _t0 < limit:
            app.processEvents()
            time.sleep(0.005)
        assert len(win.viewer._planes) == n, sorted(win.viewer._planes)

    win.viewer._on_channel_toggle(1)
    _await_planes(2)
    _both = win.viewer._hover_text((_cx, _cy))
    assert win.viewer._chan_names[0] in _both and win.viewer._chan_names[1] in _both, \
        f"both active channels must be read: {_both!r}"
    win.viewer._on_channel_toggle(1)
    _await_planes(1)

    # one pixel to the right is exactly one pixel_size_um along the stage
    _right = win.viewer._hover_text((_cx + 1, _cy))
    _sx = float(_right.split("stage ")[1].split(",")[0])
    assert abs(_sx - (1024.0 + _RN._SYNTH_META["pixel_size_um"])) < 1e-6, _right
    # …and in the letterbox margin the line drops back to its hint rather than freezing
    # on a stale coordinate, which would read as a live one
    _tl = win.viewer._view.plane_to_widget(0.0, 0.0)
    _off = QPointF(max(0.0, _tl.x() - 6.0), max(0.0, _tl.y() - 6.0))
    assert win.viewer._view.widget_to_plane(_off) is None, \
        "the probe needs a point OFF the image to test with; the view is not letterboxed"
    app.sendEvent(_hsurf, QMouseEvent(QEvent.MouseMove, _off, _off,
                                      Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    app.processEvents()
    assert win.viewer._hover_lbl.text() == _HOVER_HINT, \
        f"off the image the readout must clear: {win.viewer._hover_lbl.text()!r}"

    # (2) an ENHANCEMENT preserves the geometry, so the raw file value comes along
    win.doc.add_node("enhance.gaussian", node_id="HV", x=1200, y=700)
    win.doc.nodes["HV"].params["sigma"] = 3.0
    win.doc.nodes["HV"].locked.add("sigma")
    win.doc.connect("n1", "image", "HV", "data")
    win.scene.sync()
    app.processEvents()
    win.pull_node("HV")
    _await_pull()
    assert _hover_at(_cx, _cy), "the readout must survive a change of viewed node"
    # Aim at the pixel the blur moved MOST. The synthetic source is a smooth gradient, so
    # over most of the frame a sigma-3 blur changes the value by less than the readout
    # prints — and there the two numbers collapsing to one is correct behaviour, not a
    # missing raw probe. The check needs a pixel where they genuinely differ.
    _rawp = np.asarray(win.runner.raw_plane("HV", *win.viewer.coords()), dtype=float)
    _dispp = np.asarray(win.viewer._planes[sorted(win.viewer._planes)[0]], dtype=float)
    _diff = np.abs(_dispp - _rawp)
    _yy, _xx = np.unravel_index(int(np.argmax(_diff)), _diff.shape)
    assert _diff[_yy, _xx] > 1.0, \
        f"the probe needs a pixel the filter visibly moved; best was {_diff.max()}"
    _enh = win.viewer._hover_text((float(_xx), float(_yy)))
    assert "(raw " in _enh, f"a filtered node must show the file value too: {_enh!r}"
    _shown = float(_enh.split(" (raw ")[0].rsplit(" ", 1)[1])
    _rawv = float(_enh.split(" (raw ")[1].split(")")[0])
    assert abs(_rawv - _rawp[_yy, _xx]) < 0.51, \
        f"the raw number must be the SOURCE pixel {_rawp[_yy, _xx]}, got {_rawv} — {_enh!r}"
    assert abs(_shown - _rawv) > 0.5, \
        f"…and the node's own number must be the one it computed: {_enh!r}"
    # geometry is preserved, so the stage arithmetic still holds — checked at the centre
    assert "stage 1024.0, -2000.0" in win.viewer._hover_text((_cx, _cy)), \
        win.viewer._hover_text((_cx, _cy))

    # (3) a CROP re-addresses (x, y): the raw pixel at the same index is a different
    # pixel and the field centre is no longer the field centre — both must drop out
    # rather than be reported confidently wrong.
    win.doc.add_node("util.crop", node_id="HC", x=1200, y=900)
    win.doc.connect("n1", "image", "HC", "data")
    for _k, _v in (("x0", 40), ("x1", 200), ("y0", 30), ("y1", 170)):
        win.doc.nodes["HC"].params[_k] = _v
        win.doc.nodes["HC"].locked.add(_k)
    win.doc.touch()          # re-run the envelope pass, as every real edit path does
    win.scene.sync()
    app.processEvents()
    win.pull_node("HC")
    _cds = _await_pull()
    assert (_cds.axes.y, _cds.axes.x) == (140, 160), _cds.axes
    _crop = win.viewer._hover_text((60.0, 50.0))
    assert "stage" not in _crop and "(raw" not in _crop, \
        f"a crop re-addresses the grid — stage/raw must be withheld, not guessed: {_crop!r}"
    assert "x 60, y 50 px" in _crop and "µm" in _crop, \
        f"…while position and microns are still true in the node's own grid: {_crop!r}"

    _ok("P3 hover readout: a REAL mouse move over the image reports the pixel's position, "
        "its microns and its ABSOLUTE stage coordinate (exact at the field centre, and one "
        "pixel across moves it by one pixel_size_um) plus every shown channel's value; a "
        "sigma-3 blur puts the untouched file value beside its own; viewing the load itself "
        "claims no processing step; a crop withholds stage AND raw rather than reporting "
        "the wrong pixel; leaving the image clears the line")

    # ── D1 Dock: bake through the real runner, then the whole GUI surface ──────
    # The headless half (the checkpoint format, the graph cut, dormancy, staleness) is
    # `nodegraph.selftest::test_checkpoint_dock`. What only the GUI can prove is the part
    # the user touches: that a bake actually runs off the worker thread and lands, that
    # the document records it and greys the chain, that the memory is released, that the
    # cards and inspector say so, and that it all survives a save/load round-trip.
    from nodegraph.checkpoint import read_manifest as _read_manifest
    from nodelab_v2.ops import DOCK_OP as _DOCK_OP

    win.file_new()
    app.processEvents()
    _dock_dir = tempfile.mkdtemp(prefix="nodelab-dock-")
    win.doc.path = os.path.join(_dock_dir, "docktest.nd2graph.json")
    _src = win.doc.add_node("io.load", x=0, y=0, node_id="DL")
    _blur = win.doc.add_node("enhance.gaussian", x=220, y=0, node_id="DB")
    _dk = win.doc.add_node(_DOCK_OP, x=440, y=0, node_id="DK")
    _thr = win.doc.add_node("analysis.threshold", x=660, y=0, node_id="DT")
    win.doc.connect("DL", "image", "DB", "data")
    win.doc.connect("DB", "out", "DK", "data")
    win.doc.connect("DK", "out", "DT", "data")
    win.scene.sync()
    app.processEvents()

    # the palette offers it (io.load is hidden; the dock must NOT be)
    from nodelab_v2.scene import visible_specs as _vis
    assert any(s.op_key == _DOCK_OP for s in _vis()), \
        "Dock Data must be draggable from the Nodes palette"

    # live: a pass-through, nothing greyed, and the card says nothing special
    assert win.doc.dock_status("DK")[0] == "live"
    assert not win.doc.dormant
    assert not win.scene.node_items["DB"].is_dormant()

    # precision starts UNSET and the bake is refused until it is chosen — asserted on
    # the runner (a modal QMessageBox cannot be clicked offscreen, so drive `runner.bake`
    # the way `_start_bake` would and check the gate separately)
    assert win.doc.nodes["DK"].modes.get("precision", "unset") == "unset"
    _insp_dock = win.scene.node_items["DK"]
    win.inspector.set_node(_insp_dock)
    app.processEvents()
    assert win.doc.default_dock_store("DK").endswith(os.path.join("docktest.docks", "DK"))

    win.doc.nodes["DK"].modes["precision"] = "float32"
    win.doc.touch()
    app.processEvents()

    _pulls.clear()
    _baked = []
    win.runner.baked.connect(lambda nid, spec: _baked.append((nid, spec)))
    _store = win.doc.default_dock_store("DK")
    assert win.runner.bake("DK", store=_store, precision="float32", bake_id="probe-bake",
                           signature=win.doc.dock_signature("DK"))
    _t0 = time.time()
    while not _baked and time.time() - _t0 < 180.0:
        app.processEvents()
        time.sleep(0.005)
    assert _baked, "the bake never landed"
    assert _read_manifest(_store)["bake_id"] == "probe-bake"

    # the window's handler is what a real button press reaches — run it, then check the
    # three things it is responsible for: record, grey, release.
    win._on_baked(*_baked[0])
    app.processEvents()
    assert win.doc.nodes["DK"].modes["state"] == "docked"
    assert win.doc.dock_status("DK") == ("docked", ""), win.doc.dock_status("DK")
    assert win.doc.dormant == frozenset({"DL", "DB"}), sorted(win.doc.dormant)
    win.scene.sync()
    app.processEvents()
    assert win.scene.node_items["DB"].is_dormant()
    assert win.scene.node_items["DK"].dock_status() == "docked"
    assert not win.scene.node_items["DT"].is_dormant(), \
        "the chain BELOW a dock keeps running — only what it replaced goes dark"
    # the run graph really is cut, and the document really is not
    _rg = win.doc.to_graph(for_run=True, materialize=True)
    assert not _rg.preds("DK") and len(win.doc.to_graph().preds("DK")) == 1
    assert win.runner.planned_nodes("DT", _rg) == ["DK", "DT"], \
        win.runner.planned_nodes("DT", _rg)

    # a docked pull returns the checkpoint, and produces the same mask as the live chain
    _pulls.clear()
    win.pull_node("DT")
    _docked_out = _await_pull()
    from nodegraph.domains import Domain as _Dom
    assert _docked_out.get(_Dom.VOXEL, "mask") is not None

    # the inspector shows the dock panel with the real state + a working Re-bake button
    win.inspector.set_node(win.scene.node_items["DK"])
    app.processEvents()
    from PySide6.QtWidgets import QLabel as _QL, QPushButton as _QPB
    _btns = {b.text() for b in win.inspector.findChildren(_QPB)}
    assert "Re-bake" in _btns and "Un-dock" in _btns, _btns
    assert any(_store in lab.text() for lab in win.inspector.findChildren(_QL)), \
        "the inspector must show which folder the bake lives in"

    # staleness: an upstream edit is REPORTED, never acted on
    _orig_params = dict(win.doc.nodes["DB"].params)
    win.doc.nodes["DB"].params["sigma"] = 7.0
    win.doc.touch()
    app.processEvents()
    _st, _why = win.doc.dock_status("DK")
    assert _st == "stale" and "upstream" in _why, (_st, _why)
    assert win.doc.nodes["DK"].modes["state"] == "docked", \
        "a stale dock must keep serving its bake — nothing may recompute behind the user"
    win.scene.sync(); app.processEvents()
    assert win.scene.node_items["DK"].dock_status() == "stale"
    # restoring the params EXACTLY clears it. Note "unset" and "explicitly the default"
    # are deliberately different signatures — an unset param is auto/derived and can move
    # with the metadata, so pinning it to the same number is a real change to the recipe.
    win.doc.nodes["DB"].params.clear()
    win.doc.nodes["DB"].params.update(_orig_params)
    win.doc.touch(); app.processEvents()
    assert win.doc.dock_status("DK")[0] == "docked", "undoing the edit clears staleness"

    # save / load: the dock survives, and its folder is stored RELATIVE so the project
    # folder can be moved
    _graph_path = win.doc.path
    win.doc.save_file(_graph_path)
    _raw = json.loads(pathlib.Path(_graph_path).read_text(encoding="utf-8"))
    _saved_dk = next(n for n in _raw["graph"]["nodes"] if n["id"] == "DK")
    assert not os.path.isabs(_saved_dk["params"]["store"]), \
        f"a dock beside the graph must be saved RELATIVE, got {_saved_dk['params']['store']!r}"
    assert _saved_dk["params"]["__bake__"]["id"] == "probe-bake", \
        "the bake record must round-trip — it is what re-keys the memo and detects staleness"
    assert _saved_dk["modes"]["state"] == "docked"
    win.file_new()
    app.processEvents()
    assert not win.doc.nodes
    win.doc.load_file(_graph_path)
    app.processEvents()
    assert win.doc.nodes["DK"].modes["state"] == "docked"
    assert win.doc.dock_store("DK") == os.path.normpath(_store), \
        f"a relative dock folder must resolve against the graph: {win.doc.dock_store('DK')}"
    assert win.doc.dormant == frozenset({"DL", "DB"}), sorted(win.doc.dormant)
    assert win.doc.dock_status("DK")[0] == "docked", win.doc.dock_status("DK")
    # the loaded dock's envelope comes from the MANIFEST, not from an unknown root —
    # without that every node downstream would describe itself as all-unknown
    _env = win.doc.env("DK")
    assert _env.axes.y > 1 and "c" not in _env.unknown_axes, _env.axes
    assert win.doc.env("DT").axes == _env.axes

    # un-dock puts the chain back exactly as it was
    win.scene.sync(); app.processEvents()
    win._on_dock_action("DK", "undock")
    app.processEvents()
    assert win.doc.dock_status("DK")[0] == "live" and not win.doc.dormant
    assert win.doc.to_graph(for_run=True, materialize=True).preds("DK"), \
        "un-docking restores the in-edge, so the chain runs again"
    win._on_dock_action("DK", "redock")
    app.processEvents()
    assert win.doc.dock_status("DK")[0] == "docked"

    _ok("D1 Dock (V2.18): a REAL bake runs off the worker thread through the runner and "
        "lands a checkpoint; the window records it, flips the node to docked, greys exactly "
        "the two nodes it replaced (never the chain below) and releases their memo entries; "
        "the run graph is CUT (a pull of the downstream node plans 2 nodes, not 4) while the "
        "document keeps the chain; the inspector grows a dock panel with Re-bake/Un-dock and "
        "the folder; an upstream edit is reported as stale on card and panel WITHOUT "
        "recomputing, and undoing it clears; the folder saves RELATIVE and a load resolves "
        "it, restores docked state, dormancy and the manifest-seeded envelope every "
        "downstream node reads; un-dock/re-dock round-trip the in-edge")

    # ── V1 the display fast path: warm inline, cold on the pool, cache across rebuilds ──
    # Three defects that only bite a LAZY provider, all found on one 3D Deconvolve of a
    # 12×16×210×1024² ND2 where one compute unit is a 233 s whole-volume Richardson–Lucy:
    #   (a) the runner rebuilt the Engine — and its TileCache — on every document revision
    #       while keeping the memo. A streaming provider holds its cache by weakref, so a
    #       memo-hit lazy Dataset then read through `_NO_CACHE` and re-ran the WHOLE unit
    #       on every plane, forever. The revision bumps by itself on the first pull (the
    #       source re-seed), so the second pull was already enough to trigger it.
    #   (b) a PlaneCache miss was decoded INLINE on the GUI thread — a frozen application
    #       for the length of one unit, no repaint, no status.
    #   (c) the prefetcher warmed ±8 T-neighbours, i.e. sixteen more whole-volume computes,
    #       behind a cursor that had moved one frame.
    import threading as _threading

    import nodelab_v2.runner as _R
    from nodegraph.streaming import _NO_CACHE as _NOC

    # a source with real z and t to scrub (earlier sections shrank the fallback to z=1)
    _RN._SYNTH_AXES = AxisSizes(m=1, t=3, z=5, c=1, y=128, x=128)
    win.file_new()
    app.processEvents()
    win.build_demo()
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner.invalidate()
    win.doc.set_meta_seed("n1", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                            metadata=dict(_RN._SYNTH_META)))
    win.viewer.clear_frame_selection()               # no scope: the whole series
    win.runner.set_solo_frame(False)
    app.processEvents()
    _gui_tid = _threading.current_thread().ident
    _reads: list = []
    _orig_render = _R.render_plane_native

    def _spy_render(prov, m, t, z, c, **kw):
        _reads.append((_threading.current_thread().ident, m, t, z, c))
        return _orig_render(prov, m, t, z, c, **kw)

    _R.render_plane_native = _spy_render
    _seen: list = []
    win.runner.plane_ready.connect(lambda nid, pl, ax, s: _seen.append(nid))
    try:
        # n3 = enhance.gaussian in 3D → a WHOLE_VOLUME lazy provider, the shape that hurts
        win.pull_node("n3")
        t0 = time.time()
        while win.runner._busy and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        _prov = win.runner._viewer_provider
        assert getattr(_prov, "volume_unit", False), type(_prov).__name__

        # (a) the cache is the RUNNER's and the engine got it
        assert win.runner._engine.tiles is win.runner._tiles, \
            "every Engine must be built on the runner's own TileCache"
        _first_engine = win.runner._engine
        win.doc.touch()                      # an edit: revision bumps, engine rebuilds
        app.processEvents()
        win.pull_node("n3")
        t0 = time.time()
        while win.runner._busy and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        assert win.runner._engine is not _first_engine, "the edit must rebuild the engine"
        assert win.runner._engine.tiles is win.runner._tiles, \
            "the rebuilt engine must keep the SAME tile cache or every memo-hit lazy " \
            "provider re-runs its whole unit per plane"
        assert win.runner._viewer_provider._cache is not _NOC, \
            "the held lazy provider must still read through a LIVE cache after a rebuild"

        # (c) prefetch is gated on what a neighbouring plane costs
        assert win.runner._prefetch_span(_prov) == 0, \
            "a volume-unit compute provider must not warm T-neighbours (one each = a " \
            "whole extra unit)"
        assert win.runner._prefetch_span(win.runner._providers[
            win.runner._node_source_key["n1"]][0]) == 8, \
            "a store-backed provider still prefetches freely — it is a decompress"

        # (b) a COLD frame never decodes on the GUI thread; a warm one never leaves it
        _c = win.viewer.coords()
        _chans = win.viewer.channels()
        _cold = (_c[0], _c[1], (_c[2] + 1) % max(1, win.runner._viewer_axes.z), _c[3])
        _seen.clear()
        _reads.clear()
        assert win.runner._viewer_axes.z > 1, "the fixture needs a z axis to scrub"
        win.runner.request_plane("n3", _cold, _chans)
        assert not _seen, "a cold frame must NOT be painted from the emitting call"
        assert win.runner._decode_busy, "a cold frame must go to the pool"
        t0 = time.time()
        while win.runner._decode_busy and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        assert _seen, "the pool decode must deliver the frame"
        assert _reads and all(tid != _gui_tid for tid, *_ in _reads), \
            f"every provider read must be off the GUI thread: {_reads}"
        _seen.clear()
        _reads.clear()
        win.runner.request_plane("n3", _cold, _chans)       # now warm
        assert _seen and not _reads, "a warm frame is served inline with no provider read"

        # a scrub over cold planes keeps ONE decode in flight + ONE pending (latest-wins):
        # a volume-unit job carries a whole unit's working set, so one per pool thread is
        # not an option
        win.runner._planes.clear()
        _zmax = max(1, win.runner._viewer_axes.z)
        for _dz in range(1, 5):
            win.runner.request_plane(
                "n3", (_c[0], _c[1], (_c[2] + _dz) % _zmax, _c[3]), _chans)
        assert win.runner._decode_busy and win.runner._decode_pending is not None
        assert win.runner._decode_pending[1][2] == (_c[2] + 4) % _zmax, \
            "the pending slot must hold the LAST request, not the first"
        for _ in range(2):                    # the in-flight one, then the pending replay
            t0 = time.time()
            while win.runner._decode_busy and time.time() - t0 < 120:
                app.processEvents()
                time.sleep(0.005)
            app.processEvents()

        # an edit mid-decode retires the frame instead of painting it over the new graph
        win.runner._planes.clear()
        _seen.clear()
        win.runner.request_plane(
            "n3", (_c[0], _c[1], (_c[2] + 7) % _zmax, _c[3]), _chans)
        assert win.runner._decode_busy
        win.runner.invalidate()
        t0 = time.time()
        while win.runner._decode_busy and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        assert not _seen, "a decode superseded by an edit must be dropped, not painted"
    finally:
        _R.render_plane_native = _orig_render

    _ok("V1 display fast path: the runner OWNS the tile cache and hands it to every "
        "rebuilt engine (else a memo-hit lazy provider reads through _NO_CACHE and re-runs "
        "its whole unit per plane, forever, from the second pull onwards); a warm frame is "
        "served inline with zero provider reads while a COLD one decodes on the pool — one "
        "in flight, one latest-wins pending slot, never on the GUI thread — and an edit "
        "mid-decode drops the frame; and prefetch is cost-gated: 8 planes ahead on a "
        "store-backed provider, 0 across frames of a whole-unit compute")

    # ── V2 the store's pyramid: built on ingest, and REPAIRED when a store is short ──
    # An ingest killed between pyramid levels leaves a complete level_0 and no pyramid, and
    # nothing about opening that store says so — which is how the lab's 84.7 GB series ended
    # up serving every zoom level off full-res planes. Level l is a pure function of level
    # l-1, so the runner completes it in place on the next resolve, without reading the
    # source file. Driven here through the REAL `_resolve_source`, on a real file.
    import glob as _glob
    import shutil as _shutil

    import tifffile as _tiff

    from nodelab_v2.ingest import PYRAMID_LEVELS

    _td = tempfile.mkdtemp(prefix="nl2_pyr_")
    try:
        _tpath = os.path.join(_td, "stack.tif")
        _tvol = (np.random.default_rng(5).integers(0, 4000, size=(3, 40, 44))
                 .astype(np.uint16))
        _tiff.imwrite(_tpath, _tvol, metadata={"axes": "ZYX"})
        win.doc.nodes["n1"].params["path"] = _tpath
        win.doc.touch()
        app.processEvents()
        _p1, _e1 = win.runner._resolve_source("n1", {"path": _tpath})
        _store = _glob.glob(os.path.join(_td, "*.b2nd_store"))[0]
        assert _p1.levels == PYRAMID_LEVELS, \
            f"a fresh ingest must build the whole pyramid, got {_p1.levels}"
        _l0 = np.asarray(_p1._arrays[0][...])
        _fp1, _ver1 = _p1.fingerprint(), _p1.version

        # amputate the pyramid, exactly as the interrupted ingest did, and re-resolve
        del _p1
        import gc as _gc

        _gc.collect()
        for _f in _glob.glob(os.path.join(_store, "level_*.b2nd")):
            if not _f.endswith("level_0.b2nd"):
                os.remove(_f)
        from nodelab_v2.ingest import open_store as _open_store
        _short = _open_store(_store)
        assert _short.levels == 1, "the amputated store must open one level short"
        del _short
        win.runner._providers.clear()
        _p2, _e2 = win.runner._resolve_source("n1", {"path": _tpath})
        assert _p2.levels == PYRAMID_LEVELS, \
            f"the runner must complete a short pyramid in place, got {_p2.levels}"
        assert np.array_equal(np.asarray(_p2._arrays[0][...]), _l0), \
            "a repair must not touch level 0"
        assert _p2.fingerprint() == _fp1 and _p2.version == _ver1, \
            "a repair must be MEMO-NEUTRAL — else it discards every result downstream"
        assert _e2.axes == _e1.axes
        del _p2
        _gc.collect()
    finally:
        win.doc.nodes["n1"].params.pop("path", None)
        win.doc.touch()
        app.processEvents()
        _shutil.rmtree(_td, ignore_errors=True)

    _ok("V2 store pyramid: a fresh ingest builds all %d levels; a store amputated to "
        "level_0 (what an ingest killed between levels leaves) is completed IN PLACE on "
        "the next resolve — from level 0 alone, without reading the source file, without "
        "touching level 0, and memo-neutral so nothing downstream is re-run"
        % PYRAMID_LEVELS)

    # ── V3 viewport detail-on-demand ────────────────────────────────────────────
    # A display plane is capped at MAX_DISPLAY_DIM and zoom is a pure view transform over
    # that texture, so without this the cap is the ONLY resolution a big image ever gets —
    # a 13106² stitched mosaic could only be seen at 1/4 scale. The patch re-reads the
    # visible rect at full detail and draws it over the overview.
    #
    # The load-bearing assertion is the LAST one: the patch is a DISPLAY artefact and must
    # never change what a node reads. That is the whole contract — see the pull below.
    import nodelab_v2.runner as _RR
    _detail_seen: list = []
    win.runner.detail_ready.connect(
        lambda nid, planes, rect: _detail_seen.append((nid, planes, rect)))
    _vp = win.viewer
    _surf = _vp._surface()
    _node = "n3"

    def _settle(limit: float = 180.0) -> None:
        t0 = time.time()
        while win.runner._busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()

    win.pull_node(_node)
    _settle()

    # zoomed out: the overview IS the whole image at display resolution, so asking for a
    # patch would re-read the same pixels — the panel must not ask at all
    _surf.fit()
    app.processEvents()
    _vp._request_detail()
    app.processEvents()
    assert not _detail_seen, "a fully zoomed-out view must not request detail"
    assert _vp._detail_rect is None

    # now zoom in and drive the debounce the way a wheel event would
    _prov = win.runner._viewer_provider
    _ax = win.runner._viewer_axes
    assert _prov is not None and _ax is not None
    _rect = (0.30, 0.30, 0.55, 0.55)
    win.runner.request_detail(_node, _vp.coords(), sorted(_vp._planes), _rect,
                              _RR.MAX_DISPLAY_DIM)
    t0 = time.time()
    while not _detail_seen and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.005)
    assert _detail_seen, "no detail patch arrived"
    _nid, _dplanes, _drect = _detail_seen[-1]
    assert _nid == _node and _dplanes, (_nid, list(_dplanes))
    # the rect the pixels really cover is the requested one snapped OUT to whole pixels of
    # the level actually read — never a rect that disagrees with the pixels (that shows up
    # on screen as a seam against the overview underneath)
    assert _drect[0] <= _rect[0] + 1e-6 and _drect[1] <= _rect[1] + 1e-6
    assert _drect[2] >= _rect[2] - 1e-6 and _drect[3] >= _rect[3] - 1e-6

    # THE pixels: the patch must equal that region of the source, at the level it chose
    _ch = sorted(_dplanes)[0]
    _patch = _dplanes[_ch]
    _lv, _lax = 0, _prov.level_axes(0)
    for _l in range(getattr(_prov, "levels", 1)):
        _cand = _prov.level_axes(_l)
        if max((_drect[2] - _drect[0]) * _cand.x,
               (_drect[3] - _drect[1]) * _cand.y) <= _RR.MAX_DISPLAY_DIM:
            _lv, _lax = _l, _cand
            break
    _m, _t, _z, _ = win.runner._clamp_coords(
        win.runner._payload_coords(_vp.coords(), win.runner._viewer_pin), _ax)
    _ref = np.asarray(_prov.get_region(
        _lv, _m, _t, _z, _ch,
        int(round(_drect[1] * _lax.y)), int(round(_drect[3] * _lax.y)),
        int(round(_drect[0] * _lax.x)), int(round(_drect[2] * _lax.x))))
    assert _patch.shape == _RR._fit_plane(_ref, _RR.MAX_DISPLAY_DIM).shape, \
        (_patch.shape, _ref.shape)
    assert np.array_equal(_patch, _RR._fit_plane(_ref, _RR.MAX_DISPLAY_DIM)), \
        "the detail patch is not the pixels of the region it claims to cover"

    # it reaches the surface, and zooming back out drops it
    _vp.on_detail_ready(_nid, _dplanes, _drect)
    app.processEvents()
    assert _vp._detail_rect is not None
    _surf.fit()                               # zoom back out
    app.processEvents()
    _vp._request_detail()
    app.processEvents()
    assert _vp._detail_rect is None, "zooming out must drop the patch"

    # a patch for another node is refused rather than painted over this one's image
    _vp.on_detail_ready("some-other-node", _dplanes, _drect)
    assert _vp._detail_rect is None

    # a graph EDIT retires an in-flight patch: it is being read through the provider the
    # edit is about to drop, so delivering it would paint pre-edit pixels on a post-edit
    # image (the same rule `invalidate` already applies to an in-flight plane decode)
    _detail_seen.clear()
    win.runner.request_detail(_node, _vp.coords(), sorted(_vp._planes), _rect,
                              _RR.MAX_DISPLAY_DIM)
    win.runner.invalidate()
    t0 = time.time()
    while time.time() - t0 < 3.0:
        app.processEvents()
        time.sleep(0.005)
    assert not _detail_seen, "a detail patch superseded by an edit must be dropped"

    # ── THE contract: detail is DISPLAY-ONLY ────────────────────────────────────
    # Everything above happens after the pixels a node reads have been decided. Prove it:
    # pull the node again and check the payload is byte-identical to what it was before any
    # detail patch existed, and that the payload provider still reports its FULL extent.
    _before = win.runner._memo
    _pay = None

    def _grab(nid, payload, plane, axes, secs):
        nonlocal _pay
        if nid == _node:
            _pay = payload
    win.runner.finished.connect(_grab)
    win.runner.pull(_node, _vp.coords(), _vp.channels())
    _settle()
    win.runner.finished.disconnect(_grab)
    assert _pay is not None and _pay.image is not None
    _pax = _pay.image.axes
    assert (_pax.y, _pax.x) == (_ax.y, _ax.x), \
        f"a downstream node must see the FULL extent {(_ax.y, _ax.x)}, got {(_pax.y, _pax.x)}"
    _full = np.asarray(_pay.image.get_region(0, _m, _t, _z, _ch, 0, _pax.y, 0, _pax.x))
    assert _full.shape == (_pax.y, _pax.x), _full.shape
    assert np.array_equal(
        _full, np.asarray(_prov.get_region(0, _m, _t, _z, _ch, 0, _pax.y, 0, _pax.x))), \
        "the node payload changed — detail-on-demand must never touch what a node reads"

    _ok("V3 viewport detail-on-demand: a zoomed-out view asks for nothing (the overview IS "
        "the data at display resolution); zoomed in, the visible rect is re-read OFF the "
        "GUI thread at the finest level that fits the %d px budget and the patch is "
        "byte-identical to that region of the source, over a rect snapped out to whole "
        "pixels of the level it read; it reaches the surface, is dropped on zoom-out and "
        "refused for another node; and — the contract — the node's own payload keeps its "
        "FULL %dx%d extent and byte-identical pixels, because detail is a display artefact "
        "and never reaches what a downstream node reads"
        % (_RR.MAX_DISPLAY_DIM, _ax.y, _ax.x))

    # ── O1 overlay blending: one look, two backends ───────────────────────────────
    #
    # The Viewer draws on the GPU or, if the context is unusable, on the CPU — and the user
    # never chooses. So `blend_one` in the fragment shader and the branch in
    # `composite_with_clim` are two implementations of ONE specification, and the only thing
    # that can keep them in step is a test that runs both. This checks the shader actually
    # builds and links on a real 3.3 core context (a GLSL error would otherwise degrade
    # silently to the CPU path and look like "the blend modes do nothing"), and that the CPU
    # branch reproduces the GLSL arithmetic to within 8-bit quantization.
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
    from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
    import nodelab_v2.glview as _GV
    from nodelab_v2.viewer import composite_with_clim as _cwc

    frag = _GV._build_frag(_GV._MAX_CH)
    assert f"uniform vec3  u_blend[{_GV._MAX_CH}]" in frag, "u_blend not declared"
    # THIS process cannot compile it: the probe runs under QT_QPA_PLATFORM=offscreen with
    # NODELAB_GL=0, so no GL context exists here by construction. A GLSL error would then
    # never be caught — the panel degrades to the CPU path on `gl_failed`, so a broken
    # shader looks exactly like "this machine has no GPU". So the compile runs in a
    # SUBPROCESS on the real platform, which is the only place the driver will answer.
    _compile_src = (
        "import sys;sys.path.insert(0,%r)\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from PySide6.QtGui import QOffscreenSurface,QOpenGLContext,QSurfaceFormat\n"
        "from PySide6.QtOpenGL import QOpenGLShaderProgram,QOpenGLShader\n"
        "from nodelab_v2.glview import _build_frag,_VERT,_MAX_CH\n"
        "a=QApplication([])\n"
        "f=QSurfaceFormat();f.setVersion(3,3);f.setProfile(QSurfaceFormat.CoreProfile)\n"
        "s=QOffscreenSurface();s.setFormat(f);s.create()\n"
        "c=QOpenGLContext();c.setFormat(f)\n"
        "if not (c.create() and c.makeCurrent(s)): print('SKIP');raise SystemExit(0)\n"
        "p=QOpenGLShaderProgram()\n"
        "assert p.addShaderFromSourceCode(QOpenGLShader.Vertex,_VERT),p.log()\n"
        "assert p.addShaderFromSourceCode(QOpenGLShader.Fragment,_build_frag(_MAX_CH)),p.log()\n"
        "assert p.link(),p.log()\n"
        "assert p.uniformLocation('u_blend[0]')>=0,'u_blend optimized away'\n"
        "print('OK')\n" % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    _env["NODELAB_GL"] = "1"
    try:
        _res = subprocess.run([sys.executable, "-c", _compile_src], env=_env,
                              capture_output=True, text=True, timeout=180)
        _out = (_res.stdout or "").strip().splitlines()
    except Exception as _e:                              # noqa: BLE001
        _out, _res = [], None
    if _out and _out[-1] == "OK":
        _compiled = "compiles+links on GL 3.3 core (subprocess, real platform)"
    elif _out and _out[-1] == "SKIP":
        _compiled = "no GL driver on this machine (compile check skipped)"
    else:
        raise AssertionError(
            "the fragment shader failed to build on a real GL context — the Viewer would "
            "silently fall back to the CPU path and every blend mode would look broken:\n"
            + ((_res.stdout + _res.stderr) if _res is not None else "subprocess failed"))

    _base = np.tile(np.linspace(0, 4095, 16, dtype=np.float32), (16, 1))
    _ovl = np.tile(np.linspace(4095, 0, 16, dtype=np.float32), (16, 1)).T
    _n0 = np.clip(_base / 4095.0, 0, 1)
    _n1 = np.clip(_ovl / 4095.0, 0, 1)
    _dst = _n0[..., None] * np.array([0.0, 1.0, 0.0])
    # Tints that OVERLAP in G on purpose. Magenta-on-green is the runtime default
    # and is disjoint in RGB, which makes `add` and `difference` produce byte-
    # identical output — true of the arithmetic, useless for telling the modes
    # apart. A shared component is what makes the comparison mean anything.
    _src = _n1[..., None] * np.array([1.0, 1.0, 0.0])
    _OP = 0.6
    #: the third blend slot means checker density for mode 3 and divider position for 4
    _PARAM = 0.25

    def _glsl_ref(mode):
        """`blend_one` transcribed from the GLSL, independently of the CPU implementation."""
        if mode == 1:
            a = (_n1 * _OP)[..., None]
            return _dst * (1.0 - a) + _src * a
        if mode == 2:
            return np.abs(_dst - _src * _OP)
        if mode == 3:
            ys = np.floor(np.arange(16) / 16 * 8.0)
            k = (np.mod(ys[:, None] + ys[None, :], 2.0) * _OP)[..., None]
            return _dst * (1.0 - k) + _src * k
        if mode == 4:
            k = np.zeros((16, 16)); k[:, :int(round(_PARAM * 16))] = _OP
            return _dst * (1.0 - k[..., None]) + _src * k[..., None]
        return _dst + _src * _OP

    for _name, _mode in (("add", 0), ("over", 1), ("difference", 2), ("checkerboard", 3),
                         ("wipe", 4)):
        _img = _cwc({0: _base, 1: _ovl}, {0: (0, 255, 0), 1: (255, 255, 0)},
                    {0: (0.0, 4095.0), 1: (0.0, 4095.0)}, None,
                    {1: (_mode, _OP, 8.0 if _mode != 4 else _PARAM)})
        _got = (np.frombuffer(_img.constBits(), dtype=np.uint8)
                .reshape(16, -1, 3)[:, :16, :] / 255.0)
        _err = np.abs(_got - np.clip(_glsl_ref(_mode), 0, 1)).max()
        assert _err <= 1.0 / 255 + 1e-6, (_name, _err)
    # every mode really is DIFFERENT — a table that silently fell through to `add` would
    # otherwise pass the agreement check above four times over
    _imgs = []
    for _mode in (0, 1, 2, 3, 4):
        _im = _cwc({0: _base, 1: _ovl}, {0: (0, 255, 0), 1: (255, 255, 0)},
                   {0: (0.0, 4095.0), 1: (0.0, 4095.0)}, None,
                   {1: (_mode, _OP, 8.0 if _mode != 4 else _PARAM)})
        _imgs.append(np.frombuffer(_im.constBits(), dtype=np.uint8).copy())
    for _i in range(5):
        for _j in range(_i + 1, 5):
            assert not np.array_equal(_imgs[_i], _imgs[_j]), (_i, _j, "modes identical")
    # and opacity moves the picture continuously on the default (add) path
    _o = [np.frombuffer(_cwc({0: _base, 1: _ovl}, {0: (0, 255, 0), 1: (255, 255, 0)},
                             {0: (0.0, 4095.0), 1: (0.0, 4095.0)}, None,
                             {1: (0, _v, 8.0)}).constBits(), dtype=np.uint8).astype(float)
          for _v in (0.0, 0.5, 1.0)]
    assert _o[0].sum() < _o[1].sum() < _o[2].sum(), "opacity does not scale the overlay"
    # a channel with NO entry must composite exactly as it did before overlays existed
    _plain = _cwc({0: _base}, {0: (0, 255, 0)}, {0: (0.0, 4095.0)}, None, None)
    _same = _cwc({0: _base}, {0: (0, 255, 0)}, {0: (0.0, 4095.0)}, None, {1: (2, 0.3, 8.0)})
    assert np.array_equal(np.frombuffer(_plain.constBits(), dtype=np.uint8),
                          np.frombuffer(_same.constBits(), dtype=np.uint8))

    _ok("O1 overlay blending (V2.19): the fragment shader carries a per-channel "
        "vec3(mode, opacity, param) and %s; all five modes — add / over / difference / "
        "checkerboard / wipe — agree with the GLSL arithmetic on the CPU mirror to within 8-bit "
        "quantization and are pairwise DISTINCT (so none silently fell through to add); "
        "opacity scales the overlay monotonically; and a channel with no blend entry "
        "composites byte-identically to how it did before overlays existed, which is what "
        "keeps every non-overlay graph unchanged" % _compiled)

    # ── I1 flow.iterate: the GUI half of the parameter sweep (V2.19) ───────────
    #
    # The rewrite itself is covered by nodegraph.selftest::test_iterate. What can only be
    # checked HERE is that the app never *sees* it: the driver wire must be drawable
    # (a loop on the canvas that the cycle guard has to allow), the mode ports must appear
    # and disappear with the Iterate node, the driven editor must be locked, the panel must
    # describe the same sweep the rewrite will run, the iteration strip must select one, and
    # — the one that would silently ruin results — the EDIT-TIME envelope pass must NOT be
    # unrolled, or every node in the cone loses the envelope its widgets and pickers read.
    from PySide6.QtWidgets import (
        QDoubleSpinBox as _QDSB, QLabel as _QLbl, QPushButton as _QBtn)
    from nodelab_v2.document import GraphDocument as _GDoc
    from nodelab_v2.inspector import InspectorPanel as _Insp
    from nodelab_v2.node_item import NodeItem as _NItem
    from nodegraph.iterate import ITERATE_OP as _IT_OP, SWEEP_KEY as _SW_KEY

    idoc = _GDoc()
    idoc.add_node("io.load", node_id="IL")
    idoc.add_node("analysis.threshold", node_id="ITH", modes={"method": "fixed"})
    idoc.add_node("analysis.label", node_id="ILB")
    idoc.add_node("analysis.reduce_scalar", node_id="IRS",
                  params={"source": "area", "name": "n_cells"})
    idoc.add_node(_IT_OP, node_id="ITT",
                  params={"v0_list": "0.2, 0.4, 0.6", "index": 1},
                  modes={"mode": "sweep", "variables": "1", "v0_source": "list",
                         "preserve": "picked"})
    idoc.connect("IL", "image", "ITH", "data")
    idoc.connect("ITH", "out", "ILB", "data")
    idoc.connect("ILB", "out", "IRS", "data")
    idoc.connect("IRS", "out", "ITT", "collect")

    # the driver wire closes a loop on the canvas and MUST be allowed…
    _ok_drv, _why = idoc.can_connect("ITT", "var0", "ITH", "threshold")
    assert _ok_drv, f"a driver wire must be connectable (got {_why!r})"
    idoc.connect("ITT", "var0", "ITH", "threshold")
    # …while a DATA wire closing the same loop is still refused
    assert not idoc.can_connect("IRS", "out", "ITH", "data")[0]

    # mode ports exist only while there is an Iterate node to drive them from
    _mp = [s.name for s in idoc.input_specs("ITH")]
    assert "__mode__:method" in _mp and not any(m.endswith(":dim") for m in _mp), _mp
    assert not idoc.can_connect("ITT", "var0", "ITH", "__mode__:method")[0], \
        "a numeric variable must not reach a dropdown (no float→string conversion)"

    # the EDIT-TIME pass keeps every authored node's envelope (it must not unroll)
    idoc.propagate()
    assert all(n in idoc.envs for n in idoc.nodes), \
        "propagate must not unroll — the cone's node ids are what the GUI looks up"
    # …while the RUN graph does, minting only the kept iteration
    _run = idoc.to_graph(for_run=True, materialize=True, unroll_iterate=True)
    assert sorted(n for n in _run.nodes if "#ITT@" in n) == \
        ["ILB#ITT@1", "IRS#ITT@1", "ITH#ITT@1"]
    assert _run.nodes["ITH#ITT@1"].params["threshold"] == 0.4
    assert not any(e.kind == "driver" for e in _run.edges), \
        "no driver edge may survive into the graph the engine walks"
    assert len([n for n in idoc.to_graph(
        for_run=True, materialize=True, unroll_iterate=True,
        sweep_all=frozenset({"ITT"})).nodes if "#ITT@" in n]) == 9

    # the inspector: a driven editor is locked, and the panel describes the real sweep
    iinsp = _Insp()
    iinsp.set_node(_NItem(idoc.nodes["ITH"], idoc))
    app.processEvents()
    _rows = [w for w in iinsp.findChildren(_QLbl)
             if "driven by ITT" in (w.text() or "")]
    assert _rows, "a driven param must say so in the inspector"
    assert not _rows[0].parent().findChildren(_QDSB)[0].isEnabled(), \
        "a driven param's editor must be dead — the sweep overwrites whatever is typed"
    iinsp.set_node(_NItem(idoc.nodes["ITT"], idoc))
    app.processEvents()
    _txt = " ".join(w.text() or "" for w in iinsp.findChildren(_QLbl))
    assert "3 iterations" in _txt and "1 run" in _txt, _txt[:200]
    assert "ITH.threshold" in _txt, "the panel must name what it drives"
    _btn = next(w for w in iinsp.findChildren(_QBtn) if w.text() == "Run sweep")
    _acts = []
    iinsp.iterate_action.connect(lambda n, a: _acts.append((n, a)))
    _btn.click()
    assert _acts == [("ITT", "sweep")], _acts
    # …and Run sweep must force the cached engine to be rebuilt. `_ensure_engine` keys one
    # Engine per DOCUMENT revision and only refreshes its seeds, so a flag that changes the
    # graph without touching the document would otherwise hand the worker a 3-clone graph
    # while the engine kept walking the 1-clone one it was constructed with — the sweep
    # would report a single row and look like it had simply found nothing to compare.
    win.runner.set_sweep_all({"nope"})
    assert win.runner._engine_rev == -1, \
        "changing the sweep scope must invalidate the cached engine's GRAPH"
    win.runner.set_sweep_all(())
    # the recorded table shows the metric column once a sweep has run
    idoc.nodes["ITT"].params[_SW_KEY] = {"rows": [
        {"iter": 0, "metric": 4.0, "won": False},
        {"iter": 1, "metric": 2.0, "won": True},
        {"iter": 2, "metric": 1.0, "won": False}]}
    iinsp.set_node(_NItem(idoc.nodes["ITT"], idoc))
    app.processEvents()
    _cells = [w.text() for w in iinsp.findChildren(_QLbl)]
    assert "0.4" in _cells and "2" in _cells, _cells[-14:]
    # …and it is a UI annotation: the run graph must not carry it, or every pull would
    # re-key the memo entry that produced it
    assert _SW_KEY not in idoc.to_graph(for_run=True).nodes["ITT"].params
    iinsp.set_node(None)
    iinsp.setParent(None)

    # the viewer's iteration strip selects one, writing index + preserve
    win.viewer.set_iterations(["0.2", "0.4", "0.6"], 1)
    app.processEvents()
    assert win.viewer._iter_row.isVisible() and win.viewer._iter_strip.value() == 1
    _picked = []
    win.viewer.iteration_changed.connect(_picked.append)
    win.viewer._iter_strip.setValue(2)
    app.processEvents()
    assert _picked == [2], _picked
    win.viewer.set_iterations(())
    app.processEvents()
    assert not win.viewer._iter_row.isVisible(), "no Iterate node ⇒ no strip"

    # a save/load round-trip keeps the driver wire as a drawable document edge
    import json as _json
    idoc2 = _GDoc()
    idoc2.load_dict(_json.loads(_json.dumps(idoc.to_dict())))
    assert ("ITT", "var0", "ITH", "threshold") in idoc2.edges
    assert not idoc2.has_unedited_structure, \
        "a driver wire is editable structure, not a preserved-verbatim back-edge"

    _ok("I1 flow.iterate (V2.19): the driver wire closes a loop on the canvas and is "
        "allowed while a DATA wire closing the same loop is still refused; Mode ports "
        "appear only beside an Iterate node, exclude the 2D/3D lever and reject a numeric "
        "variable; the edit-time envelope pass does NOT unroll (every authored node keeps "
        "the envelope its widgets read) while the run graph does, minting ONE clone for "
        "'picked' and N for Run sweep with the value baked into each and no driver edge "
        "surviving; the inspector locks a driven editor, names what the sweep drives, "
        "counts its iterations, emits the sweep action and renders the recorded metric "
        "table — which stays a UI annotation the run graph strips; the Viewer's iteration "
        "strip shows, selects and hides; and the wire round-trips through save/load")

    # ── C1: TWO channel branches, told apart end to end (2026-08-03) ───────────
    #
    # Reported as "running a second channel labels it as the same channel as the first, and
    # they can't both run the same analysis node". Three separate defects, all of which need
    # the real GUI to see: the Split's per-channel sockets fell back to Ch0/Ch1 instead of the
    # file's channel names, the Viewer's channel strip was keyed on the payload's axis SIZES
    # (identical on two branches, so it never rebuilt and the second branch kept the first's
    # name/tint/LUT), and one analysis node fed from two channels was refused outright.
    from nodelab_v2.document import CHANNELS_KEY as _CHK

    win.file_new()
    app.processEvents()
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=1, c=2, y=128, x=128)
    win.runner._providers.clear()          # else the c=1 provider an earlier section cached
    win.runner._announced.clear()          # is reused and the ch1 tap has nothing to select
    win.runner._raw_src.clear()
    win.runner.invalidate()
    cdoc = win.doc
    cdoc.add_node("io.load", node_id="cl", x=0, y=0, params={_CHK: [
        {"name": "DAPI", "emission_nm": 461.0, "color": [0, 0, 255]},
        {"name": "GFP", "emission_nm": 509.0, "color": [0, 255, 0]}]})
    cdoc.set_meta_seed("cl", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                         metadata=dict(_RN._SYNTH_META)))
    cdoc.add_node("channel.split", node_id="csp", x=200, y=0)
    cdoc.add_node("enhance.gaussian", node_id="ce0", x=420, y=-140)
    cdoc.add_node("analysis.segment", node_id="cseg", x=640, y=-140)
    cdoc.add_node("enhance.gaussian", node_id="ce1", x=420, y=140)
    cdoc.add_node("analysis.measure", node_id="cms", x=880, y=0)
    cdoc.connect("cl", "image", "csp", "data")
    cdoc.connect("csp", "ch0", "ce0", "data")
    cdoc.connect("csp", "ch1", "ce1", "data")
    cdoc.connect("ce0", "out", "cseg", "data")
    cdoc.connect("cseg", "out", "cms", "data")
    cdoc.connect("ce1", "out", "cms", "raw")
    app.processEvents()

    # (a) the Split's per-channel sockets carry the FILE's channel names, not Ch0/Ch1 — the
    #     only thing on the canvas that says which branch is which
    _clabels = [s.label for s in cdoc.output_specs("csp") if s.name.startswith("ch")]
    assert _clabels == ["0 · DAPI", "1 · GFP"], _clabels
    assert [c["name"] for c in cdoc.upstream_channel_descriptors("ce0")] == ["DAPI"]
    assert [c["name"] for c in cdoc.upstream_channel_descriptors("ce1")] == ["GFP"]

    # (b) ONE analysis node fed from BOTH channels runs: ch0's objects, ch1's intensity.
    #     This used to raise "a DIFFERENT sampling geometry", naming a shift that a channel
    #     tap cannot cause.
    cdone = {}
    _cfin = win.runner.finished.connect(lambda nid, *a: cdone.setdefault("id", nid))
    _cerr = win.runner.failed.connect(lambda nid, tr: cdone.setdefault("err", tr))
    win.pull_node("cms")
    t0 = time.time()
    while not cdone and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.01)
    assert cdone.get("err") is None, f"two-channel measure failed:\n{cdone.get('err')}"
    assert cdone.get("id") == "cms"

    # (c) the Viewer tells the two branches apart. Same axes on both, so a size-keyed strip
    #     would show the first branch's channel for the second.
    _strips = {}
    for _nid in ("ce0", "ce1"):
        cdone.clear()
        win.pull_node(_nid)
        t0 = time.time()
        while not cdone and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.01)
        assert cdone.get("err") is None, f"{_nid} failed:\n{cdone.get('err')}"
        _md = getattr(win.viewer._dataset, "metadata", {}) or {}
        _strips[_nid] = (list(win.viewer._chan_names),
                         _md.get("channel_names"), _md.get("channel_emission_nm"))
    assert _strips["ce0"][0] != _strips["ce1"][0], (
        f"both channel branches drew the SAME channel strip {_strips['ce0'][0]} — the "
        f"payloads have identical axes, so the strip's rebuild key must include the "
        f"per-channel identity")
    for _nid, _want in (("ce0", 0), ("ce1", 1)):
        _strip, _names, _emis = _strips[_nid]
        # the tap narrowed every per-channel list to the one channel it kept…
        assert _names == [f"Ch{_want}"], (_nid, _names)
        assert _emis == [_RN._SYNTH_META["channel_emission_nm"][_want]], (_nid, _emis)
        assert _strip == _names, (_nid, _strip, _names)   # …and the strip draws it

    win.runner.finished.disconnect(_cfin)
    win.runner.failed.disconnect(_cerr)
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)   # restore the fallback
    _ok("C1 two channel branches (2026-08-03): a Split's per-channel sockets carry the "
        "FILE's channel names (DAPI/GFP, not Ch0/Ch1) and each branch reports its own "
        "upstream channel; ONE analysis node fed ch0's objects + ch1's intensity RUNS "
        "(a channel tap is no longer read as a spatial shift); and the Viewer's channel "
        "strip follows the branch — the two payloads have identical axes, so a size-keyed "
        "strip showed the first branch's name, tint and LUT for the second")

    print("\nALL PHASE-5 GUI PROBES PASSED")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
