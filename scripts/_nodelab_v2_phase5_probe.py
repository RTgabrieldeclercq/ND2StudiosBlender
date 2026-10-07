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
os.environ["NODELAB_LAYOUT"] = "0"               # never read or write the user's panel layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPointF            # noqa: E402
from PySide6.QtGui import QFontDatabase       # noqa: E402
from PySide6.QtWidgets import QApplication    # noqa: E402


def _load_fonts() -> None:
    # seguisym: the panel title bars' glyphs (◉ ◫ ☰ ⇱ ⇲ ✕) name Segoe UI Symbol as fallback
    for name in ("segoeui.ttf", "consola.ttf", "arial.ttf", "seguisb.ttf", "seguisym.ttf"):
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


def _probe_movie_editor(win, app) -> None:
    """The Movie Editor dock (2026-09-30), driven the way the mouse would drive it.

    One graph: a synthetic source ``S``, a threshold ``Th`` whose Voxel mask stands in for a
    label raster, and an Export Movie ``M`` with ``S`` on ``data`` and ``Th`` — through a
    Reroute — on ``source_b``. Checks, in order: selecting ``M`` raises and binds the dock;
    the sources resolve THROUGH the reroute; *Compute sources* fetches without retargeting
    the Viewer or evicting its held views; *Convert to timeline* switches the Mode and starts
    from the flat movie; the loop template is the user's interleave (A's max-Z, then B's
    labelled z sweep, alternating); the monitor's frames equal the EXPORTED frames bit for
    bit; a settled Viewer LUT is stamped into the linked channels; undo/redo round-trip; and
    deleting the only segment is refused."""
    import glob as _glob
    import json as _json
    from PIL import Image as _Image
    from nodegraph.catalog._shared import movie_timeline as MT
    from nodelab_v2.node_item import NodeItem

    def wait(pred, timeout=120.0):
        t0 = time.time()
        while not pred() and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        return pred()

    from nodegraph.dataset import AxisSizes
    from nodegraph.metadata import MetaEnvelope
    from nodelab_v2 import runner as _RN

    win.file_new()
    app.processEvents()
    # A known synthetic stack. The runner caches the synthetic provider by path, so an
    # earlier section's geometry would otherwise leak in (a Z=1 stack makes the z sweep one
    # plane long and every frame count below wrong).
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner._raw_src.clear()
    win.runner.invalidate()
    _RN._SYNTH_AXES = AxisSizes(m=1, t=2, z=4, c=2, y=96, x=128)
    doc = win.doc
    doc.add_node("io.load", node_id="S", x=0, y=0)
    doc.set_meta_seed("S", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                        metadata=dict(_RN._SYNTH_META)))
    doc.add_node("analysis.threshold", node_id="Th", x=260, y=180)
    doc.add_node("rr.reroute", node_id="R", x=420, y=200)
    doc.add_node("io.write_movie", node_id="M", x=560, y=40)
    doc.connect("S", "image", "Th", "data")
    doc.connect("Th", "out", "R", "data")
    doc.connect("S", "image", "M", "data")
    doc.connect("R", "out", "M", "source_b")
    app.processEvents()
    ed = win.movie_editor

    # selecting the node raises and binds the dock — WITHOUT growing the main window. It
    # used to: the dock spanned the full width under the tall side column and raised the
    # window's minimum height past a 1080p screen, so its buttons sat off the bottom edge
    # and "nothing in the movie editor could be clicked" (reported 2026-09-30).
    min_before, size_before = win.minimumSizeHint(), win.size()
    items = {i.node_id: i for i in win.scene.items() if isinstance(i, NodeItem)}
    win.scene.clearSelection()
    items["M"].setSelected(True)
    wait(lambda: False, 0.3)
    assert win._movie_dock.isVisible() and ed.bound() == "M", (ed.bound(),)
    assert win.minimumSizeHint().height() <= max(min_before.height(), size_before.height()) \
        and win.minimumSizeHint().width() <= max(min_before.width(), size_before.width()) \
        and win.size() == size_before, (min_before, win.minimumSizeHint(), win.size())
    assert ed._flat, "a new Export Movie plays its flat settings until converted"
    assert all(b.isEnabled() for b in ed._op_btns), "a flat node locks the editor's buttons"
    srcs = win._movie_sources("M")
    assert srcs["A"]["node"] == "S" and srcs["B"]["node"] == "Th" and srcs["C"]["node"] is None
    assert doc.real_source("M", "source_b") == "Th", "real_source did not walk the reroute"

    # Compute sources: a payload-only fetch — the Viewer is not retargeted
    shown = []
    win.runner.finished.connect(lambda nid, *a: shown.append(nid))
    viewed, held = win._viewed, set(win.runner._views)
    ed.compute_sources()
    assert wait(lambda: ed._payloads.get("A") is not None), "source A never arrived"
    assert win._viewed == viewed and set(win.runner._views) == held \
        and win.runner.run_id("S") not in shown, (
        "a Movie Editor fetch reached the Viewer", win._viewed, shown)
    assert wait(lambda: ed.frame_count() > 0), ed._status.text()

    # Convert to timeline: Sweep flips, the flat movie becomes clip 1, auto channels link
    ed.convert_to_timeline()
    app.processEvents()
    rec = doc.nodes["M"]
    assert rec.modes.get("sweep") == "timeline" and not ed._flat
    spec = MT.normalize_spec(rec.params["timeline"])
    assert len(spec["segments"]) == 1 and spec["segments"][0]["kind"] == "clip"
    assert all(d["link"] == "viewer"
               for d in spec["segments"][0]["panels"][0]["display"].values())
    assert rec.params["timeline"] == MT.canonical_json(spec), "not stored canonically"

    # the loop template: A's max-Z at t, then B's labelled z sweep at t
    ed.select(("seg", 0))
    ed.add("loop")
    app.processEvents()
    spec = MT.normalize_spec(rec.params["timeline"])
    loop = spec["segments"][1]
    assert loop["kind"] == "loop" and loop["source"] == "A", loop
    still, sweep = loop["body"]
    assert still["play"]["axis"] == "none" and still["panels"][0]["z"] == "max"
    assert sweep["play"] == dict(sweep["play"], axis="z", source="B", direction="alternate")
    assert sweep["panels"][0]["render"] == "labels_over_image" and \
        sweep["panels"][0]["layer"] == "mask", sweep["panels"][0]
    ed.compute_sources()
    assert wait(lambda: ed._payloads.get("B") is not None), "source B never arrived"
    a_ax, b_ax = ed._payloads["A"].axes, ed._payloads["B"].axes
    assert (a_ax.t, b_ax.z) == (2, 4), (a_ax, b_ax)
    n_want = a_ax.t + a_ax.t * (1 + b_ax.z)       # clip 1 plays t, then the loop
    assert wait(lambda: ed.frame_count() == n_want), (ed.frame_count(), n_want,
                                                      ed._status.text())

    # a settled Viewer LUT on S is stamped into A's linked channels
    _S = win.runner.run_id("S")                # the viewer keys a node by its run id
    win.viewer._clim[(_S, 0)] = (100.0, 900.0)
    win.viewer._clim_user.add((_S, 0))         # a window the user SET, not an auto one
    win.viewer._gammas[(_S, 0)] = 1.5
    win._on_viewer_display(_S)
    app.processEvents()
    spec = MT.normalize_spec(rec.params["timeline"])
    d0 = spec["segments"][0]["panels"][0]["display"]["0"]
    assert (d0["lo"], d0["hi"], d0["gamma"]) == (100.0, 900.0, 1.5), d0
    assert spec["segments"][1]["body"][0]["panels"][0]["display"]["0"]["lo"] == 100.0

    # the monitor IS the export: write a PNG sequence and compare every frame
    tmp = tempfile.mkdtemp(prefix="nd2sb_movie_editor_")
    rec.params["path"] = os.path.join(tmp, "m.png")
    rec.modes["format"] = "png"
    doc.touch("M")
    app.processEvents()
    done = {}
    win.runner.finished.connect(lambda nid, *a: done.setdefault(nid, True))
    win.runner.failed.connect(lambda nid, tr: done.setdefault("err", tr))
    win.export_movie("M")
    assert wait(lambda: win.runner.run_id("M") in done or "err" in done), \
        "the export never finished"
    assert "err" not in done, done.get("err")
    shots = sorted(_glob.glob(os.path.join(tmp, "m_*.png")))
    assert wait(lambda: ed.frame_count() == n_want)
    assert len(shots) == n_want, (len(shots), n_want)
    for k, f in enumerate(shots):
        assert np.array_equal(np.asarray(_Image.open(f)), ed.render_now(k)), (
            f"monitor frame {k} differs from the exported frame")

    # undo / redo, and the one refusal
    n_before = len(MT.normalize_spec(rec.params["timeline"])["segments"])
    ed.undo()
    assert len(MT.normalize_spec(rec.params["timeline"])["segments"]) == n_before - 1
    ed.redo()
    assert len(MT.normalize_spec(rec.params["timeline"])["segments"]) == n_before
    ed.select(("seg", 1))
    ed.delete()
    ed.select(("seg", 0))
    ed.delete()
    assert len(MT.normalize_spec(rec.params["timeline"])["segments"]) == 1
    assert "cannot delete the only segment" in ed._status.text(), ed._status.text()

    # REAL clicks, not the editor's API: outline rows and property checkboxes rebuild parts
    # of the editor, which must never happen inside the clicked widget's own signal (that
    # deleted the widget under a press Qt was still handling). Afterwards the op buttons
    # must still answer.
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QCheckBox
    tree = ed._tree
    for _round in range(2):
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            QTest.mouseClick(tree.viewport(), _Qt.LeftButton,
                             pos=tree.visualItemRect(it).center())
            wait(lambda: False, 0.05)
    ed.select(("seg", 0))
    wait(lambda: False, 0.1)
    for b in [b for b in ed._props.widget().findChildren(QCheckBox) if b.isVisible()][:3]:
        QTest.mouseClick(b, _Qt.LeftButton)
        wait(lambda: False, 0.05)
    n_seg = len(MT.normalize_spec(rec.params["timeline"])["segments"])
    QTest.mouseClick(ed._op_btns[0], _Qt.LeftButton)          # +Clip, clicked
    wait(lambda: False, 0.1)
    assert len(MT.normalize_spec(rec.params["timeline"])["segments"]) == n_seg + 1
    QTest.mouseClick(ed._undo_btn, _Qt.LeftButton)            # Undo, clicked
    wait(lambda: False, 0.1)
    assert len(MT.normalize_spec(rec.params["timeline"])["segments"]) == n_seg

    # the render thread must be idle before the probe's os._exit: a thread still inside
    # OpenCV when the process tears down crashes it (exit 139, a false failure)
    assert wait(lambda: not ed._renderer.busy, 60), "the render thread never went idle"
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)   # restore the fallback
    _ok("Movie Editor (2026-09-30): selecting an Export Movie raises the dock bound to it "
        "without growing the main window (the dock sits under the canvas and scrolls), and "
        "every control is live on a flat node; its sources resolve through a Reroute; "
        "REAL clicks on outline rows and property checkboxes leave the op buttons "
        "answering; Compute sources fetches payload-only "
        "(Viewer target and held views untouched); Convert to timeline flips Sweep and "
        "starts from the flat movie with auto channels linked to the Viewer; the loop "
        f"template is A's max-Z then B's labelled z sweep, alternating ({n_want} frames on "
        "a T=2, Z=4 stack); a settled Viewer LUT is stamped into the linked channels; every "
        "monitor frame equals the exported PNG bit for bit; undo/redo round-trip; deleting "
        "the only segment is refused")


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
    _rq = win.runner.run_id              # bare node id → page-qualified run id (V4.00 step 2)
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

    # THE CARD'S LAYER PICKER (2026-08-04). The inspector has offered one since V2.11, but
    # clicking a `layer_in` pill on the CANVAS dropped straight into the inline text editor —
    # so on the card the only way to change it was to already know the name. That is exactly
    # the failure the picker exists to prevent: a node kept its factory-default layer name
    # while the branch feeding it produced another, and the mismatch surfaced as a pull error
    # several nodes downstream instead of as a menu with the right answer already in it.
    _m_item = win.scene.node_items["n6"]                  # analysis.measure ← analysis.label
    _lay_ctl = next(c for c in _m_item.controls()
                    if c.kind == "value" and c.obj.name == "labels")
    _offered = win.doc.layer_choices("n6", _lay_ctl.obj)
    assert "labels" in _offered, f"the edit-time pass should predict `labels` here: {_offered}"
    _menus.clear()
    _NI.QMenu = _PeekMenu
    try:
        _m_item._open_layer_menu(_lay_ctl)
    finally:
        _NI.QMenu = _QMenu
    assert _menus, "clicking a layer_in pill on the card opened no menu"
    _lay_entries = dict(_menus[-1])
    assert _lay_entries["__visible__"] == "True", "layer menu tooltips would be swallowed"
    for _nm in _offered:
        assert _nm in _lay_entries, f"{_nm!r} present on the wire but not offered: " \
                                    f"{sorted(_lay_entries)}"
    # free text stays reachable: the prediction is honest but INCOMPLETE (a few producers
    # name layers it cannot foresee), so a closed menu would be a regression, not a fix
    assert "Type a name…" in _lay_entries, sorted(_lay_entries)
    assert "labels" == _m_item.rec.params.get("labels", _lay_ctl.obj.default), \
        "dismissing the layer menu must not write a param"

    # A TWO-INPUT NODE'S SOCKETS HOVER DIFFERENTLY (2026-08-04). The painted domain rail is a
    # NODE-level answer repeated beside every Dataset input, so on `analysis.voronoi` the
    # seeds wire and the areas wire showed identical chips and there was nowhere the card said
    # which was which. Attributing the requirement per wire is not possible from the
    # declarations (a Label instance's LABEL half is wanted by the areas wire while only its
    # VOXEL half is named by a socket), so the per-socket fact that IS available — what that
    # edge actually carries — goes in the hover, beside the node-level requirement rather than
    # replacing it. Dataset sockets also carry `description` now; they could not before.
    import re as _re
    from nodelab_v2.document import GraphDocument as _GDv
    from nodelab_v2.scene import GraphScene as _GSv
    _vdoc = _GDv()
    # the two branches must carry DIFFERENT domains, or "on this wire" cannot be SEEN to
    # differ and the check would be passing on the two descriptions alone
    for _nid, _op in (("vu", "io.load"), ("vd", "detect.spots"),
                      ("vr", "io.load"), ("vt", "analysis.threshold"),
                      ("vl", "analysis.label"), ("vv", "analysis.voronoi")):
        _vdoc.add_node(_op, node_id=_nid)
    _vdoc.connect("vu", "image", "vd", "data")        # seeds branch adds POINT
    _vdoc.connect("vd", "out", "vv", "data")
    _vdoc.connect("vr", "image", "vt", "data")        # areas branch adds VOXEL + LABEL
    _vdoc.connect("vt", "out", "vl", "data")
    _vdoc.connect("vl", "out", "vv", "areas")
    _vsc = _GSv(_vdoc)
    _vsc.sync()
    _vit = _vsc.node_items["vv"]
    _tips = {}
    for _sk in ("data", "areas"):
        _tips[_sk] = _re.sub(r"<[^>]+>", " ", _vit.socket("in", _sk).toolTip())
        assert "on this wire:" in _tips[_sk], _tips[_sk][:200]
        assert "the node requires:" in _tips[_sk], "the requirement must still be stated"
    assert "SEEDS branch" in _tips["data"] and "AREAS branch" in _tips["areas"], \
        "each Dataset socket must carry its OWN prose"
    assert _tips["data"] != _tips["areas"], \
        "two Dataset inputs that hover identically is the defect this closes"
    _wire = lambda s: _tips[s].split("on this wire:")[1][:48]
    assert "point" in _wire("data") and "point" not in _wire("areas"), \
        (_wire("data"), _wire("areas"))
    assert "label" in _wire("areas"), _wire("areas")
    # an UNWIRED aux input says so rather than reporting the node's requirement as content
    _vdoc.disconnect(*_vdoc.edge_into("vv", "areas"))
    assert _vdoc.edge_into("vv", "areas") is None
    _vsc.sync()
    _bare = _re.sub(r"<[^>]+>", " ", _vsc.node_items["vv"].socket("in", "areas").toolTip())
    assert "nothing wired" in _bare, _bare[:200]

    # THE POINTS OVERLAY PICKS ITS LAYER TOO (2026-08-04). It drew EVERY Point table, which is
    # right for one detection and wrong the moment a node publishes a filtered view of
    # another's cloud: `analysis.voronoi` emits `<name>_seeds` holding only the dots that won a
    # territory, so on Auto the display showed those AND the full input — the dropped dots
    # included, which is exactly what was asked to stop showing.
    import numpy as _np
    from nodegraph.dataset import Dataset as _DS
    from nodegraph.domains import Domain as _Dom
    from nodegraph.provider import ArrayProvider as _AP
    from nodegraph.structure import StructureTable as _STp
    _vp = win.viewer
    _pax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    _mk = lambda ids, ys: _STp(_Dom.POINT, {
        "id": _np.asarray(ids, dtype=_np.int64), "m": _np.zeros(len(ids), _np.int64),
        "t": _np.zeros(len(ids), _np.int64), "c": _np.zeros(len(ids), _np.int64),
        "z": _np.zeros(len(ids)), "y": _np.asarray(ys, float),
        "x": _np.asarray(ys, float)}, layer=None, z_kind="plane_index")
    _pds = (_DS(axes=_pax).with_image(_AP(_np.zeros((1, 1, 1, 1, 8, 8))))
            .with_structure(_STp(_Dom.POINT, dict(_mk([1, 2, 3], [1., 3., 5.]).columns),
                                 layer="all_dots", z_kind="plane_index"))
            .with_structure(_STp(_Dom.POINT, dict(_mk([2], [3.]).columns),
                                 layer="kept_dots", z_kind="plane_index")))
    _pkeep = (_vp._dataset, _vp._axes, _vp._ref_plane, _vp.overlays.points.layer,
              _vp.overlays.points.z_project)
    _had_pc2 = "_payload_coords" in _vp.__dict__
    try:
        _vp._dataset, _vp._axes = _pds, _pax
        _vp._ref_plane = _np.zeros((8, 8))
        _vp._payload_coords = lambda: (0, 0, 0, 0)
        _vp.overlays.points.z_project = True
        assert _vp.point_layer_names() == ["all_dots", "kept_dots"], _vp.point_layer_names()
        _vp.overlays.points.layer = ""
        assert len(_vp._point_marks()) == 4, "Auto still draws every table (3 + 1)"
        _vp.overlays.points.layer = "kept_dots"
        assert len(_vp._point_marks()) == 1, \
            "an explicit pick must draw ONLY that table — the filtered view is the point"
        _vp.overlays.points.layer = "renamed_away"
        assert len(_vp._point_marks()) == 4, \
            "a stale pick falls back to all, not to nothing (the Labels picker's rule)"
        # the dialog's provider is PER TAB, or the Points tab would offer label rasters
        assert _vp.overlay_layer_names("points") == ["all_dots", "kept_dots"]
        assert _vp.overlay_layer_names("labels") == []      # this fixture has no raster
        assert _vp.overlay_layer_names("tracks") == []
    finally:
        (_vp._dataset, _vp._axes, _vp._ref_plane, _vp.overlays.points.layer,
         _vp.overlays.points.z_project) = _pkeep
        if not _had_pc2:
            del _vp._payload_coords

    # THE LABELS OVERLAY PICKS ITS LAYER (2026-08-04). It had no selector: `_label_plane`
    # drew whichever integer Voxel raster had the most regions in the viewed plane. Several
    # label rasters on one Dataset is the NORMAL case — `analysis.voronoi` alone emits its
    # territories, inherits the seed branch's labels and copies the areas it clipped to, and
    # two segmentations both default to the name `labels` — so the overlay drew whichever was
    # most fragmented and no click could change it ("I select the segmentation from the Red
    # channel but it still shows the UV labels": there was nothing to select).
    _lax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    # `wanted` holds THREE regions with small ids; `noisy` holds ONE with a huge id. The two
    # rankings disagree on purpose: by region COUNT `wanted` wins (3 > 1), by `plane.max()` —
    # what the fallback used to compute — `noisy` wins on its single id 40. That is the shape
    # of the real failure: a raster carrying another node's numbering outranked the answer.
    _few = _np.zeros((1, 1, 1, 1, 8, 8), dtype=_np.int64)
    _few[0, 0, 0, 0, 1:3, 1:3] = 1
    _few[0, 0, 0, 0, 1:3, 4:6] = 2
    _few[0, 0, 0, 0, 4:6, 1:3] = 3
    _many = _np.zeros((1, 1, 1, 1, 8, 8), dtype=_np.int64)
    _many[0, 0, 0, 0, 5:7, 5:7] = 40                      # one region, far bigger id
    _lds = (_DS(axes=_lax).with_image(_AP(_np.zeros((1, 1, 1, 1, 8, 8))))
            .with_layer(_Dom.VOXEL, "wanted", _few)
            .with_layer(_Dom.VOXEL, "noisy", _many))
    _keep = (_vp._dataset, _vp._axes, _vp._ref_plane, _vp.overlays.labels.layer)
    _had_pc = "_payload_coords" in _vp.__dict__      # restore the BOUND method, not a copy
    try:
        _vp._dataset, _vp._axes = _lds, _lax
        _vp._ref_plane = _np.zeros((8, 8))
        _vp._payload_coords = lambda: (0, 0, 0, 0)
        assert _vp.label_layer_names() == ["noisy", "wanted"], _vp.label_layer_names()
        # AUTO PREFERS THE VIEWED NODE'S OWN OUTPUT (2026-08-04). Ranking the payload's
        # rasters by id drew whichever carried the biggest NUMBERING — on `analysis.voronoi`
        # that was the copied areas raster, whose ids are the source segmentation's (in the
        # hundreds) while only a handful of its regions survive. You view a node to see what
        # it made, so its own declared output wins.
        _vp.own_layers_cb = lambda _nid: ["wanted"]
        _vp._node_id = "probe-node"
        assert _np.array_equal(_vp._label_source(), _few), \
            "Auto must draw the node's OWN output, not the raster with the biggest ids"
        _vp.own_layers_cb = None
        _vp.overlays.labels.layer = ""                     # no preference: count regions
        assert _np.array_equal(_vp._label_source(), _few), \
            "the fallback COUNTS regions (3 vs 1); `plane.max()` ranked the 1-region raster " \
            "first because its single id is 40"
        _vp.overlays.labels.layer = "wanted"               # ...and an explicit pick WINS
        assert _np.array_equal(_vp._label_source(), _few), \
            "an explicit layer pick must override the most-regions guess"
        assert _np.array_equal(_vp._label_plane(), _few[0, 0, 0, 0]), \
            "the PAINTED plane and the picked raster must be the same layer"
        assert _np.array_equal(_vp._label_layer_values(), _few), \
            "the size-probe must count off the layer that is on screen"
        _vp.overlays.labels.layer = "renamed_away"         # a stale pick falls back, not blank
        assert _vp._label_source() is not None, \
            "a pick this payload lacks must fall back to Auto, not draw nothing"
        # the dialog offers exactly those names, Auto first, and keeps a stale pick visible
        from nodelab_v2.overlay_dialog import OverlayDialog as _OD
        # the provider is PER TAB (two tabs now have a `layer` field and they mean different
        # domains), so the dialog is handed the dispatcher, never one tab's list
        _ld = _OD(_vp.overlays, None, None, layer_names=_vp.overlay_layer_names)
        _ld.reload()
        _lw = _ld._widgets[("labels", "layer")]
        _items = [(_lw.itemText(i), _lw.itemData(i)) for i in range(_lw.count())]
        assert _items[0][1] == "" and _items[0][0].startswith("Auto"), _items
        assert [d for _t, d in _items[1:]] == ["noisy", "wanted", "renamed_away"], _items
        assert _lw.currentData() == "renamed_away" and "not on this payload" in _lw.currentText()
        _ld.deleteLater()
    finally:
        (_vp._dataset, _vp._axes, _vp._ref_plane,
         _vp.overlays.labels.layer) = _keep
        if not _had_pc:                  # the stub would otherwise pin every later check to
            del _vp._payload_coords      # (0,0,0,0) — it broke `_points_here` 300 lines on

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
        "as a documented lever, and each vocab tick box explains its own token; and a "
        "`layer_in` pill on the CARD now opens the same picker the inspector has (every "
        "layer the wire actually carries, each explained, plus a `Type a name…` escape "
        "because the prediction is honest but incomplete) instead of dropping into a bare "
        f"text box — it offered {sorted(_offered)} here. And the Labels OVERLAY finally has a "
        "layer picker: it drew whichever integer Voxel raster had the most regions with no way "
        "to override, so on a Dataset carrying several (the normal case) it showed the wrong "
        "segmentation; an explicit pick now wins over the guess, the painted plane and the "
        "size-probe read the SAME layer, a stale pick falls back rather than drawing nothing, "
        "and the combo lists the live payload's rasters with Auto first. A node with TWO "
        "Dataset inputs also hovers them apart at last: the painted rail is one node-level "
        "answer repeated beside each, so each socket's tip now names what ITS OWN edge carries "
        "(POINT on the seeds wire, LABEL on the areas one) beside the node's requirement, "
        "carries its own prose (Dataset sockets could not before), says `nothing wired` when "
        "unplugged — and re-reads on a wiring change, which `refresh` never did, so every tip "
        "used to be frozen at the last relayout")

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
    # V2.23b: picking a model must SAY what it found, because finding nothing is a legitimate
    # outcome (not downloaded, wrong folder) and is otherwise indistinguishable from a broken
    # feature — reported as "loading in the model does not change any of the parameters".
    # `C:\models\sd` does not exist here, which is precisely the silent case: assert the panel
    # names the reason and the fix rather than quietly showing this app's own defaults.
    win.inspector._rebuild()
    app.processEvents()
    _tn = win.inspector._last_trained_note
    assert _tn and "config.json" in _tn, \
        f"a model path with no config.json must explain itself, got {_tn!r}"
    assert not win.scene.node_items["pb"].trained(), "nothing can be adopted from a fake path"
    # ...and the same node with the method that loads no such model says nothing at all,
    # rather than carrying a line about a file it never reads.
    seg.modes["method"] = "threshold"; doc.touch(); win.inspector._rebuild()
    app.processEvents()
    assert not win.inspector._last_trained_note, \
        f"a non-model method must add no model line, got {win.inspector._last_trained_note!r}"
    seg.modes["method"] = "stardist"; doc.touch(); win.inspector._rebuild()
    app.processEvents()
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
    doc.set_muted("n3", True)
    g = doc.to_graph(for_run=True)
    assert "n3" not in {e.dst for e in g.edges} and "n3" not in {e.src for e in g.edges}
    assert any(e.src == "n2" and e.dst == "n4" for e in g.edges)   # bypassed around
    doc.set_muted("n3", False)
    # only a node that keeps the kind of data may be switched off (V4.00 step 11e): a
    # Threshold adds a mask, so muting it would starve everything that reads the mask
    try:
        doc.set_muted("n4", True)
        raise AssertionError("muting a Threshold must be refused")
    except ValueError as _g3:
        assert "cannot be switched off" in str(_g3) and not doc.nodes["n4"].muted
    _ok("G3: a muted node is bypassed (n2 → n4) in the run graph; a node that changes the "
        "kind of data (Threshold adds a mask) cannot be muted")

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

    # ── WS1: the WORKSPACE file (V4.00 step 1) — Save writes format 3.0 with one Free page;
    #    a reload reads it back into the SAME document the canvas is bound to ──────────────
    ws = win.workspace
    assert ws.active and ws.page(ws.active).doc is win.doc and ws.page(ws.active).kind == "input"
    assert win.doc.page_kind == "input" and win.doc.store_tag == ws.active
    tmpw = os.path.join(tempfile.mkdtemp(prefix="nd2ws_"), "w.nd2graph.json")
    n_before = (len(win.doc.nodes), len(win.doc.edges))
    ws.save_file(tmpw)
    with open(tmpw, encoding="utf-8") as f:
        raww = json.load(f)
    assert raww["format_version"] == "3.0" and raww["app_version"], raww.get("format_version")
    assert [p["kind"] for p in raww["workspace"]["pages"]] == \
        ["input", "refine", "process", "analyze"], "File ▸ New is the standard workspace (step 11)"
    assert win.doc.path == tmpw, win.doc.path
    ws.load_file(tmpw)
    assert ws.page(ws.active).doc is win.doc, "a reload must keep the canvas bound to its document"
    assert (len(win.doc.nodes), len(win.doc.edges)) == n_before
    assert (win.doc.nodes["n3"].x, win.doc.nodes["n3"].y) == (123.0, 456.0)
    # the 2.0 file G6 wrote still opens — as one Free page named after the file
    ws.load_file(tmp)
    assert ws.page(ws.active).doc is win.doc and ws.page(ws.active).name == "t"
    assert (len(win.doc.nodes), len(win.doc.edges)) == n_before
    # the palette still offers everything on a Free page (page.* included, never hidden)
    from nodelab_v2.scene import visible_specs as _visible_ws1   # main() rebinds the bare name later
    assert {s.op_key for s in _visible_ws1()} >= {"page.input", "page.output"}
    _ok("WS1 workspace file: Save writes format 3.0 (the four standard pages, app_version "
        "stamped); "
        "reload keeps the canvas bound to the same document; a 2.0 file opens as a Free page")

    # ── WS2: the RUNNER on the workspace (V4.00 step 2) — a pull on a page the canvas is
    #    NOT showing composes its upstream page in and runs under page-qualified ids; the
    #    shared chain is one memo entry across pages; an edit on the shown page cancels
    #    exactly the other page's runs that read it ─────────────────────────────────────
    from nodegraph.metadata import MetaEnvelope as _ME2
    from nodelab_v2 import runner as _RN2
    win.file_new()
    app.processEvents()
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner.invalidate()
    ws = win.workspace
    pA = ws.page(ws.active)                                   # the Free page on the canvas
    assert pA.doc is win.doc
    win.doc.add_node("io.load", node_id="S", x=0, y=0)
    win.doc.set_meta_seed("S", _ME2(axes=_RN2._SYNTH_AXES, metadata=dict(_RN2._SYNTH_META)))
    win.doc.add_node("page.output", node_id="O", x=220, y=0, params={"name": "raw"})
    win.doc.connect("S", "image", "O", "data")
    pB = ws.add_page("Proc", "process")                       # NOT on the canvas
    pB.doc.add_node("page.input", node_id="IN", params={"source": f"{pA.id}:raw"})
    pB.doc.add_node("enhance.gamma", node_id="X", params={"gamma": 1.4})
    pB.doc.connect("IN", "out", "X", "data")
    app.processEvents()
    assert ws.active == pA.id and win.doc is pA.doc
    _rx = f"{pB.id}/X"
    assert win.runner.run_id("S") == f"{pA.id}/S" and win.runner.run_id(_rx) == _rx
    assert win.runner.planned_nodes(_rx) == sorted([f"{pA.id}/S", f"{pA.id}/O", _rx]), \
        win.runner.planned_nodes(_rx)
    assert win.runner.planned_nodes("O") == sorted([f"{pA.id}/S", f"{pA.id}/O"])
    _ws2: dict = {}
    _ev2: list = []
    _c1 = win.runner.finished.connect(lambda nid, *a: _ws2.setdefault("id", nid))
    _c2 = win.runner.failed.connect(lambda nid, tr: _ws2.setdefault("err", tr))
    _c3 = win.runner.node_progress.connect(lambda ev, nid, info: _ev2.append((ev, nid)))
    win.runner.pull(_rx)                                      # the page the canvas is not showing
    t0 = time.time()
    while not _ws2 and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert _ws2.get("err") is None, _ws2.get("err")
    assert _ws2.get("id") == _rx, _ws2
    assert win.runner.finished_result(_rx) is not None, "the other page's result must be remembered"
    assert set(win.runner._engine.graph.nodes) >= {f"{pA.id}/S", f"{pA.id}/O", _rx}, \
        sorted(win.runner._engine.graph.nodes)
    assert ("done", _rx) in _ev2, _ev2
    assert win._viewed is None, "a result for a page not on the canvas must not retarget the Viewer"
    # the shown page's cards DID run (they are the other page's upstream): they say so,
    # nothing is left transient, and the finished run no longer claims them
    _states = {n: it.run_state() for n, it in win.scene.node_items.items()}
    assert _states == {"S": "done", "O": "done"}, _states
    assert win.scene.planned_nodes() == frozenset(), win.scene._plans
    # the shared chain is ONE memo entry: pulling the Output on ITS page is a hit
    _ws2.clear(); _ev2.clear()
    win.pull_node("O")
    t0 = time.time()
    while not _ws2 and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert _ws2.get("err") is None, _ws2.get("err")
    assert _ws2.get("id") == f"{pA.id}/O", _ws2
    assert ("cached", f"{pA.id}/O") in _ev2, _ev2
    # the pulled card wears the run's wall time (`finish_run` stamps it `done`); its memo
    # hit is the `cached` event asserted above, exactly as a same-page re-pull reports
    assert win._viewed == "O" and win.scene.node_items["O"].run_state() in ("done", "cached"), \
        win.scene.node_items["O"].run_state()
    assert win.runner.finished_result("O") is not None and win.runner.finished_result(_rx) is not None
    # a BOUND Page Input card is pullable through the real path: it has no node in the run
    # graph, so `_submit` serves it with the upstream Output and puts the card in the cone
    _rin = f"{pB.id}/IN"
    _ws2.clear()
    win.runner.pull(_rin)
    _live = [r for r, n in win.runner._runs.items() if n == _rin]
    assert _live, ("the Input's pull did not start", win.runner._runs)
    _cone_in = win.runner._run_cones[_live[0]]
    assert _rin in _cone_in and f"{pA.id}/O" in _cone_in, sorted(_cone_in)
    t0 = time.time()
    while not _ws2 and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert _ws2.get("err") is None, _ws2.get("err")
    assert _ws2.get("id") == _rin, _ws2
    assert win.runner.finished_result(_rin).axes == win.runner.finished_result("O").axes, \
        "the Input card shows exactly what the upstream Output passes on"
    # its plan is what serves it, plus the card (the hover readout finds its source there)
    _pin_plan = win.runner.planned_nodes(_rin)
    assert {_rin, f"{pA.id}/S", f"{pA.id}/O"} <= set(_pin_plan), _pin_plan
    # a bake or hold delivered for a page that has gone is reported, not a KeyError
    win._on_baked("pg99/DK", {"store": "x", "manifest": {}})
    win._on_baked("pg99/DK", {"hold": True, "payload": None})
    # …and an ordinary pull reading THROUGH the Input records the Input in its real cone
    _ws2.clear()
    win.runner.pull(_rx)
    _real_cone = next(c for r, c in win.runner._run_cones.items()
                      if win.runner._runs.get(r) == _rx)
    assert _rin in _real_cone and f"{pA.id}/S" in _real_cone, sorted(_real_cone)
    t0 = time.time()
    while not _ws2 and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert _ws2.get("id") == _rx and _ws2.get("err") is None, _ws2
    # an edit on the SHOWN page cancels the other page's run that reads it — and only that
    _canc2: list = []
    _c4 = win.runner.cancelled.connect(_canc2.append)
    win.runner._runs[9400] = _rx
    win.runner._run_cones[9400] = frozenset(win.runner.planned_nodes(_rx))   # reads pA/S
    win.runner._runs[9401] = _rx
    win.runner._run_cones[9401] = frozenset([_rx])                            # does not
    win.doc.nodes["S"].params["path"] = ""
    win.doc.touch("S")
    app.processEvents()
    assert 9400 not in win.runner._runs and 9401 in win.runner._runs, sorted(win.runner._runs)
    assert _canc2 == [_rx], _canc2
    # …and re-pointing the OTHER page's Page Input cancels the run that reads through it,
    # although the Input itself has no node in the run graph (its cone names it explicitly)
    _canc2.clear()
    win.runner._runs.clear(); win.runner._run_cones.clear()
    win.runner._runs[9402] = _rx
    win.runner._run_cones[9402] = _real_cone          # the cone `_submit` really recorded
    pB.doc.nodes["IN"].params["source"] = f"{pA.id}:nope"
    pB.doc.touch("IN")
    app.processEvents()
    assert 9402 not in win.runner._runs and _canc2 == [_rx], (sorted(win.runner._runs), _canc2)
    win.runner._runs.clear(); win.runner._run_cones.clear()
    for sig, c in ((win.runner.finished, _c1), (win.runner.failed, _c2),
                   (win.runner.node_progress, _c3), (win.runner.cancelled, _c4)):
        sig.disconnect(c)
    ws.remove_page(pB.id)
    ws.load_file(tmp)                 # back to the demo graph G7 pulls next
    app.processEvents()
    _ok("WS2 runner on the workspace: a pull on a page the canvas is not showing composes "
        "the upstream page in and finishes under its qualified id without retargeting the "
        "Viewer (the shown page's cards it computed read done, and nothing stays claimed); "
        "the shared Output is a memo hit from its own page; a bound Page Input card is "
        "pullable and shows its upstream Output; the real run cone names the Input it reads "
        "through, so an edit on the shown page or a re-pointed Input cancels exactly the "
        "other page's run that reads it")

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
    assert done.get("id") == _rq("n3"), done
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
    assert done.get("id") == _rq("n3") and repull < 5.0
    _ok(f"G7: re-pull hits the persistent memo ({repull*1000:.0f} ms)")

    # ── G2: palette content ─────────────────────────────────────────────────────
    # Stage → role → node (the taxonomy in codemap/node_roles.json), dots on both sides of
    # every node row, and an overview card for whatever is clicked (2026-10-02).
    from nodegraph import roles as _ROLES
    from nodelab_v2.scene import visible_specs
    from PySide6.QtCore import Qt as _Qt
    tree = win.palette._tree

    def _rows(kind):
        out = []
        for i in range(tree.topLevelItemCount()):
            head = tree.topLevelItem(i)
            if kind == "stage":
                out.append(head)
            for j in range(head.childCount()):
                role = head.child(j)
                if kind == "role":
                    out.append(role)
                for k in range(role.childCount()):
                    if kind == "node":
                        out.append(role.child(k))
        return out

    win.palette.refill("")
    # stage and role rows are spanned bands with their text in column 0 (left-justified,
    # filled background); node rows keep the dots | label | dots columns
    stage_labels = [h.text(0) for h in _rows("stage")]
    assert stage_labels == [m["label"] for _, m in _ROLES.stages()], stage_labels
    for h in _rows("stage"):
        assert h.isFirstColumnSpanned() and h.background(0).color().isValid() \
            and h.background(0).color() != h.background(1).color(), "stage band"
    for r in _rows("role"):
        assert r.isFirstColumnSpanned() and r.text(0) and not r.text(1), "role band"
    visible_ops = {s.op_key for s in visible_specs()}
    roles_with_visible = {rk for rk, r in _ROLES.load()["roles"].items()
                          if any(op in visible_ops for op in r["ops"])}
    assert len(_rows("role")) == len(roles_with_visible), \
        (len(_rows("role")), len(roles_with_visible))   # graph-structure ops are hidden
    nodes = _rows("node")
    assert len(nodes) == len(visible_specs()), (len(nodes), len(visible_specs()))
    gauss = next(r for r in nodes if r.data(0, _Qt.UserRole) == "enhance.gaussian")
    label = next(r for r in nodes if r.data(0, _Qt.UserRole) == "analysis.label")
    for r in (gauss, label):
        assert not r.icon(0).isNull() and not r.icon(2).isNull(), "dots on both sides"
    assert "reads voxel" in label.toolTip(0) and "adds label" in label.toolTip(2), \
        (label.toolTip(0), label.toolTip(2))
    assert "float parameter" in gauss.toolTip(0), gauss.toolTip(0)
    # a single click fills the overview with THAT node, read live from the registry
    tree.setCurrentItem(label)
    app.processEvents()
    assert win.palette.current_op == "analysis.label"
    ov = win.palette.overview_html()
    for needle in ("Connected Components", "analysis.label", "Find structure",
                   "Segmentation", "Footprint", "How it works"):
        assert needle in ov, f"overview lacks {needle!r}"
    tree.setCurrentItem(_rows("role")[0])
    app.processEvents()
    assert win.palette.current_op is None and "Nodes" in win.palette.overview_html()
    # search still filters, and keeps the stage/role scaffolding only where it has a hit
    win.palette.refill("gauss")
    found = [r.text(1) for r in _rows("node")]
    assert any("Gaussian" in t for t in found), found
    assert all(h.childCount() > 0 for h in _rows("stage")), "no empty stage under a filter"
    win.palette.refill("")
    _ok("G2: palette is stage → role → node with every visible op placed, in/out dots "
        "with domain-aware tooltips, an overview that follows the click, and a search "
        "that filters without leaving empty groups")

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

    # ── source calibration override: the typed value must reach BOTH halves ───
    #
    # Regression, found on real data 2026-09-15: the override was applied in
    # `_resolve_source`, but `_all_sources` builds the worker's cfg as an explicit
    # WHITELIST (`path`, access, bundle paths) — so the typed number never left the GUI
    # thread. The edit-time header showed the Z step while the pulled payload still carried
    # None, and `track.objects` refused quoting a value the card claimed to have. Drift
    # between the envelope and the payload is the exact failure an axis-changing node's
    # meta_transform exists to prevent; a SOURCE can drift the same way.
    from nodelab_v2.ops import CALIB_OVERRIDE_KEYS as _CK, LOAD_OP as _LOAD
    _cdoc = GraphDocument()
    _crec = _cdoc.add_node(_LOAD, x=0, y=0, params={"path": "C:/nonexistent/vol.ome.tif"})
    from nodelab_v2.workspace import Workspace as _WSc
    _csrc = _WSc.single(_cdoc)
    class _CR: _source = _csrc                         # the runner reads its GraphSource…
    _ckey = f"{_csrc.active}/{_crec.id}"               # …and keys sources by run id
    assert not any(k in _ER._all_sources(_CR())[_ckey] for k in _CK), \
        "an untouched card must override nothing (0/absent = whatever the file says)"
    _crec.params["z_step_um"] = 1.0                    # the user types it (inspector path)
    _ccfg = _ER._all_sources(_CR())[_ckey]
    assert _ccfg.get("z_step_um") == 1.0, \
        "the typed Z step must reach the worker thread's cfg, or the pull cannot see it"
    from nodelab_v2.runner import _with_card_calib as _wcc
    assert _wcc(MetaEnvelope(metadata={"pixel_size_um": 1.0}), _ccfg).metadata["z_step_um"] \
        == 1.0, "…and be applied to the envelope the payload and meta-seed both come from"
    _cdoc.propagate()
    assert _cdoc.envs[_crec.id].metadata.get("z_step_um") == 1.0, \
        "…while the EDIT-TIME envelope agrees, so derived defaults re-seed as you type"
    _ok("source calibration override (2026-09-15): a Z step typed on the Load card reaches "
        "the worker cfg, the resolved envelope AND the edit-time pass — the three that must "
        "agree, since a plain TIFF records no z_step_um and nothing downstream can infer "
        "one; an untouched card still overrides nothing")

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

    # (the file-bundle export column is covered by nodegraph.selftest's
    # test_bundle_source_file_column — tables/export are Qt-free, so they belong on the
    # Qt-free gate rather than in the middle of a GUI timing sequence.)

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
    assert plans and plans[-1][0] == _rq("n4"), plans[-1]
    assert set(plans[-1][1]) == {_rq(n) for n in ("n1", "n2", "n3", "n4")}, plans[-1]
    kinds = [(ev, win._local(nid)) for ev, nid, _f in events]   # the runner emits run ids
    assert ("start", "n4") in kinds and ("done", "n4") in kinds, kinds
    n3_last = max(i for i, k in enumerate(kinds) if k[1] == "n3")
    assert n3_last < kinds.index(("start", "n4")), kinds   # upstream settles first
    assert kinds[n3_last][0] in ("done", "cached"), kinds[n3_last]
    # analysis.threshold is EAGER (per-plane) so it reports real fractions; the last one
    # always lands on 1.0 (the runner never throttles the final update)
    fr = [f for ev, nid, f in events if ev == "progress" and nid == _rq("n4")]
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
    # …and it carries no RUN note. Since V2.27 every card's tooltip names the node and
    # its footprint whether or not it has ever been pulled (`_apply_card_tip`, called
    # from `_layout`), so "no stale state" is about the run text, not the whole tooltip.
    assert items["n7"]._run_text() == "", items["n7"]._run_text()
    assert "footprint" in items["n7"].toolTip(), items["n7"].toolTip()
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

    # Console: the FULL trace, selectable and copyable (restored 2026-09-15) ───
    #
    # Asked for by name ("errors should be in a console that can be copy/pasted into").
    # The failure above is what opened it: the status bar TRUNCATES to the window width
    # and a tooltip cannot be selected, so this panel is the only place the text a user
    # needs to send someone can actually be got at.
    from PySide6.QtGui import QGuiApplication
    assert win._console_dock.isVisible(), "the first failure of a session must raise it"
    assert win._console_act.isChecked(), "…and the View ▸ Console tick must follow the dock"
    shown = win.console._out.toPlainText()
    assert "n3 FAILED" in shown and "Boom" in shown, shown
    assert win.console._out.isReadOnly(), "a log you can edit is a log you cannot trust"
    assert win.console._out.textInteractionFlags() & Qt.TextSelectableByMouse, \
        "the whole point is that it can be selected"
    win.console.copy_all()
    assert "Boom" in QGuiApplication.clipboard().text(), "Copy all must reach the clipboard"
    # Whitespace is preserved verbatim — the reason the panel formats with
    # QTextCharFormat rather than HTML, since a traceback whose indentation collapsed is
    # not the traceback you were asked to paste.
    win.console.error("outer\n    indented 4")
    assert "\n    indented 4" in win.console._out.toPlainText()
    # A console the USER closed stays closed — someone working through a chain of errors
    # must not have to dismiss it after every one — while still RECORDING every failure.
    win._console_act.setChecked(False)
    assert not win._console_dock.isVisible()
    win._on_run_failed("n3", "Traceback…\nAgain")
    assert not win._console_dock.isVisible(), \
        "a console the user closed must not re-open on each later failure"
    assert "Again" in win.console._out.toPlainText(), "…but it still records them"
    win.scene.clear_run_states()
    win._set_progress(None)
    # Hand the window back exactly as it was found: showing/hiding a bottom dock resizes
    # the CENTRAL splitter, and the maximize/restore check below captures those sizes.
    app.processEvents()
    _ok("Console (asked 2026-09-15): a failed pull's FULL traceback lands in a selectable "
        "monospace log with Copy all — the status bar truncates to the window width and a "
        "tooltip cannot be selected, so this is the only copyable surface; it raises itself "
        "on the first failure of a session, the View ▸ Console tick tracks the dock both "
        "ways, indentation survives to the clipboard verbatim (QTextCharFormat, not HTML), "
        "and a console the user closed keeps recording without re-opening itself")

    # E9: maximized canvas + mini-map Viewer + click-to-preview ────────────────
    # (V4.00 step 4: the Viewer is a dock above the canvas; maximizing hides the docked
    # viewers and re-homes the ACTIVE one into the mini-map)
    from PySide6.QtCore import Qt as _QtE9
    _vd0 = win._viewer_dock(win.viewer)
    assert _vd0 is not None and _vd0.objectName().startswith("viewer:")
    assert not _vd0.isHidden() and win.dockWidgetArea(_vd0) == _QtE9.TopDockWidgetArea
    assert win.centralWidget() is win._main_canvas and win.view is win._main_canvas.view, \
        "the main canvas is the centre; viewers are docks"
    docked_h = _vd0.height()
    win.set_maximized(True)
    for _ in range(3):
        app.processEvents()                            # dock re-layout + reposition
    # the SAME viewer widget moved into the overlay (not a copy) and left its dock
    assert win.viewer.parent() is win.minimap and win.minimap.content is win.viewer
    assert _vd0.isHidden() and _vd0.widget() is None
    assert not _vd0.toggleViewAction().isEnabled(), \
        "View ▸ Panels must not open the mini-map viewer's dock as an empty panel"
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
    while _rq("n5") not in pulled and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    assert _rq("n5") in pulled, f"click did not preview the node (pulled={pulled})"
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
    assert win._local(pulled[-1]) in ("n2", "n3", "n4"), pulled[-1]
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
    for _ in range(3):
        app.processEvents()
    assert _vd0.widget() is win.viewer and not _vd0.isHidden()
    assert _vd0.toggleViewAction().isEnabled()
    assert win.viewer.isVisible() and not win.minimap.isVisible()
    assert not win.viewer.compact
    assert all(h.isVisible() for h in win.viewer._hists.values())
    assert win.viewer._ovl_btn.text() == "◈ Overlays"
    assert not win.view.is_maximized() and not win._max_act.isChecked()
    assert _vd0.height() == docked_h, (_vd0.height(), docked_h)
    _ok("Restore: the Viewer goes back into its dock at its old height, full controls "
        "returned")

    # ── V2.28 → V4.00 step 4: Compare — a second Viewer DOCK beside the active one ──
    #
    # Compare opens (or re-targets) the active viewer's Compare viewer: a dock split beside
    # it. When both results span the same M/T/Z the two share ONE cursor — the leader's
    # strips move both and the Compare viewer drops its own row; different extents give it
    # its own.
    done_cmp: list = []
    win.runner.finished.connect(lambda nid, *a: done_cmp.append(nid))

    def _wait_pull(nid, timeout=120):
        t0 = time.time()
        while _rq(nid) not in done_cmp and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        assert _rq(nid) in done_cmp, f"{nid} never finished (got {done_cmp})"

    win.pull_node("n3")                     # leader: the 3D gaussian (z=5)
    _wait_pull("n3")
    done_cmp.clear()
    _lead = win.viewer
    _ld = win._viewer_dock(_lead)
    win.open_compare("n4")                  # same source chain → same M/T/Z → LINKED
    _wait_pull("n4")
    for _ in range(3):
        app.processEvents()
    assert win._viewed == "n3" and win._viewed2 == "n4"
    assert win.viewer is _lead, "the leader stays the active viewer"
    assert len(win.shell.docks_of("viewer")) == 2
    assert win.viewer2 is not None and win.viewer2 is not _lead
    _cd = win._viewer_dock(win.viewer2)
    assert not _cd.isHidden() and not _cd.isFloating()
    assert win.dockWidgetArea(_cd) == win.dockWidgetArea(_ld)
    assert _cd.geometry().left() > _ld.geometry().left(), "leader left, Compare right"
    assert win._links[win.viewer2] is _lead
    assert win.viewer2.has_image(), "the Compare viewer must show n4's pixels"
    assert "n4" in win.viewer2._status.text()
    assert "pulled in" in win.viewer._status.text()    # the leader KEPT its result
    # metadata match (same M/T/Z) → linked: one cursor, the Compare viewer's row is gone
    assert win.viewer2 in win._linked
    assert not win.viewer2._axes_box.isVisible()
    # (step 11d) the Playback panel shows the active viewer's strips — for a LINKED Compare
    # viewer its leader's, which move both; Channels shows the Compare viewer's own
    _lead_c = win._links[win.viewer2]
    win._activate_viewer(win.viewer2)
    app.processEvents()
    assert win.playback_panel.shown_viewer() is _lead_c
    assert win.channels_panel.shown_viewer() is win.viewer2
    win._activate_viewer(_lead_c)
    app.processEvents()
    assert "linked" in _cd.title_bar.title_text() and "n4" in _cd.title_bar.title_text()
    _ok("Compare (V2.28 → V4.00): a second Viewer DOCK opens beside the active one (leader "
        "left, Compare right), shows its own node's result, and LINKS to one cursor when "
        "the two results' M/T/Z extents match; its title bar names the node and the link")

    # the one cursor: the leader's strip moves the Compare viewer's silently and fetches
    # its plane off the decode lane — no bounce, and no re-pull of either node
    npull = len(pulled)
    win.viewer._sliders["z"].setValue(3)
    for _ in range(5):
        app.processEvents()
    assert win.viewer2._sliders["z"].value() == 3, "linked cursor must mirror"
    t0 = time.time()
    while win.runner.busy and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.005)
    assert len(pulled) == npull, "a linked scrub must not re-pull either node"
    _ok("Compare: the leader's strips move BOTH viewers; scrubbing linked viewers stays "
        "on the coords-only fast path (no pull slot, both nodes' views held at once)")

    # DIFFERENT metadata unlinks: a Z-Project (z 5 → 1) gets its own cursor row back
    zp = doc.add_node("util.zproject", x=1060, y=430)
    doc.connect("n3", "out", zp.id, "data")
    done_cmp.clear()
    win.open_compare(zp.id)
    _wait_pull(zp.id)
    for _ in range(3):
        app.processEvents()
    assert win._viewed2 == zp.id
    assert len(win.shell.docks_of("viewer")) == 2, "re-targeted, not a third viewer"
    assert win.viewer2 not in win._linked
    assert not win.viewer2._axes_box.isHidden(), "different M/T/Z → the viewer's own sliders"
    _lead_c = win._links[win.viewer2]
    win._activate_viewer(win.viewer2)
    app.processEvents()
    assert win.playback_panel.shown_viewer() is win.viewer2, "…in the Playback panel"
    win._activate_viewer(_lead_c)
    app.processEvents()
    assert "own cursor" in win._viewer_dock(win.viewer2).title_bar.title_text()
    _ok("Compare: a result with different M/T/Z (Z-Project, z 5→1) re-targets the same "
        "Compare viewer and gives it its own sliders — the link is re-derived from the "
        "payloads' own axes on every delivery")

    # close: one viewer again
    win.close_compare()
    app.processEvents()
    assert win._viewed2 is None and win.viewer2 is None and not win._links
    assert [d.objectName() for d in win.shell.docks_of("viewer")] == [_ld.objectName()]
    # reopen: beside the leader again
    done_cmp.clear()
    win.open_compare("n4")
    _wait_pull("n4")
    for _ in range(3):
        app.processEvents()
    assert win._links.get(win.viewer2) is _lead and win.viewer2 in win._linked
    # …maximizing hides the Compare viewer with the other docked viewers, and the restore
    # brings both back, still linked
    win.set_maximized(True)
    for _ in range(3):
        app.processEvents()
    assert win._viewer_dock(win.viewer2).isHidden() and win.minimap.content is _lead
    win.set_maximized(False)
    for _ in range(3):
        app.processEvents()
    assert not win._viewer_dock(win.viewer2).isHidden() and win._viewed2 == "n4"
    assert win.viewer2 in win._linked and _ld.widget() is _lead
    # DELETING the compared node closes its viewer: one still showing the result of a node
    # that is no longer on the canvas is a lie, and the one the user cannot detect
    done_cmp.clear()
    win.open_compare(zp.id)
    _wait_pull(zp.id)                       # settle first: deleting mid-pull is a
    assert win._viewed2 == zp.id            # different test, and not this one
    doc.remove_node(zp.id)                  # also leaves the demo graph as the next
    app.processEvents()                     # sections (and the screenshots) expect it
    assert win._viewed2 is None and not win._links, \
        "deleting the compared node must close its viewer"
    assert len(win.shell.docks_of("viewer")) == 1
    _ok("Compare: close leaves one viewer; reopen sits beside the leader again; maximize "
        "hides the Compare viewer with the docks and the restore brings it back linked; "
        "deleting the compared node closes its viewer rather than leaving a result with no "
        "node")

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

    # THE CONDITION COLUMN PICKER (V2.28). The layer picker above is an EDITABLE combo
    # because a couple of producers name layers the edit-time pass cannot predict. The
    # COLUMN picker is the opposite: every structure-producing node declares `adds_columns`
    # (selftest::test_column_catalog_complete), so the columns a table carries really are
    # determined by the nodes upstream and the control is a CLOSED dropdown. That is the
    # whole feature — "which statistics do I have?" answered in the menu instead of by
    # pulling and reading the error — so the closed-ness is asserted here, on the widget.
    pdoc.add_node("analysis.if_else", node_id="PIE",
                  params={"labels": "regions", "column1": "area"},
                  modes={"target": "label"})
    pdoc.connect("PL", "out", "PIE", "data")
    _col_sock = pdoc.nodes["PIE"].spec().input("column1")
    _cols = pdoc.column_choices("PIE", _col_sock)
    assert "area" in _cols and "mean_intensity" not in _cols,         f"a segmentation alone offers its own geometry and nothing measured: {_cols}"
    pdoc.add_node("analysis.measure", node_id="PM2",
                  params={"labels": "regions", "stats": "mean", "shape": "solidity"},
                  modes={"target": "label"})
    pdoc.connect("PL", "out", "PM2", "data")
    pdoc.connect("PM2", "out", "PIE", "data")          # re-wire through Measure
    _cols2 = pdoc.column_choices("PIE", _col_sock)
    assert {"mean_intensity", "solidity"} <= set(_cols2),         f"the palette must GROW when a Measure is inserted upstream: {_cols2}"

    _pie_item = _PItem(pdoc.nodes["PIE"], pdoc)
    _cbox = _PInsp()._column_box(_pie_item, _col_sock)
    assert not _cbox.isEditable(),         "the column control must be a CLOSED dropdown, not a text box with suggestions "         "(nodelab_v2.inspector._names_box editable=False)"
    assert [_cbox.itemText(i) for i in range(_cbox.count())] == list(_cols2),         "the dropdown must list exactly what the edit-time column catalog offers"
    assert _cbox.currentText() == "area", "and start on the value the node actually holds"

    # A value nothing upstream writes must SURVIVE. A closed combo can only emit items in
    # its list, so an orphaned column would otherwise be silently rewritten to index 0 the
    # moment the panel rebuilds — a value the user never chose, on a node that decides
    # which objects survive.
    pdoc.nodes["PIE"].params["column1"] = "ghost_col"
    _cbox2 = _PInsp()._column_box(_PItem(pdoc.nodes["PIE"], pdoc), _col_sock)
    assert _cbox2.currentText() == "ghost_col",         "a column no producer writes must be KEPT, not silently replaced by the first entry"
    assert "NOT on this wire" in _cbox2.toolTip(),         "...and the panel has to say why it is there, or it reads as a working setting"
    pdoc.nodes["PIE"].params["column1"] = "area"

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

    # ── T1b: the troubleshooting REGION box (2026-10-02) ─────────────────────
    # While the scope is on, the Viewer carries an amber box spanning the SOURCE frame;
    # dragging it (here: the programmatic call a completed drag makes) scopes the pull to
    # that window of the source. The payload IS the window, its pixels are the full frame's
    # slice, the chip names it, and clearing it restores the full frame as a memo identity.
    from nodelab_v2 import region_box as _RB
    assert win.viewer.region_extent == (128, 128), win.viewer.region_extent
    assert win.viewer.region is None and win.runner.region is None, "at rest: whole frame"
    assert _RB.clamp((0, 128, 0, 128), (128, 128)) is None, "full frame == no region"
    win.viewer.set_region((32, 96, 16, 80))            # ← what releasing a drag does
    windowed = _await_pull()
    assert win.viewer.region == (32, 96, 16, 80) and win.runner.region == (32, 96, 16, 80)
    assert (windowed.axes.y, windowed.axes.x) == (64, 64), \
        f"a windowed pull must carry the window's extent, got {windowed.axes}"
    assert windowed.axes.t == 1 and windowed.axes.z == 3 and windowed.axes.c == 2
    assert win._solo_chip.text() == "SOLO t4·64×64px", win._solo_chip.text()
    assert "region 64×64@y32,x16" in win.viewer._status.text(), win.viewer._status.text()
    # n5 sits below a 3D Gaussian (n3), whose halo is clipped at the WINDOW's edge exactly
    # as it would be below a Crop node — so the window's outermost halo of pixels differs
    # from the full frame's slice and its interior is bit-identical. Both halves are the
    # claim: the interior proves the window reads the right source pixels, the border
    # proves the caveat the manual states is real rather than theoretical. The halo here is
    # 20 px (sigma 0.5 um at the synthetic 0.1 um/px -> 5 px, int(4*sigma+0.5)).
    HALO = 20
    for c, plane in solo_pixels.items():
        got = np.asarray(win.viewer._planes[c])
        want = plane[32:96, 16:80]
        assert got.shape == (64, 64), f"windowed plane shape {got.shape}, want (64, 64) (c={c})"
        inner = (slice(HALO, -HALO), slice(HALO, -HALO))
        assert np.array_equal(got[inner], want[inner]), \
            (f"the window's interior must be the full frame's slice (c={c}): "
             f"{int((got[inner] != want[inner]).sum())} of {got[inner].size} differ")
        assert not np.array_equal(got, want), \
            f"the window's halo border should differ from the full run (c={c}) — it is clipped"
    # the LOCATOR MAP (2026-10-02): once the window is narrower than the frame, the top-left
    # of the surface carries the whole field with the window drawn on it in amber. Painted
    # through the same overlay callback both surfaces call, onto a black canvas, so the
    # check is on pixels: amber in the map's corner, none there with the window cleared.
    from PySide6.QtGui import QImage as _QI, QPainter as _QP, QPixmap as _QPx
    from PySide6.QtCore import Qt as _Qt2
    _surf = win.viewer._pick_targets()[0]

    def _corner_amber() -> int:
        cv = _QPx(_surf.width(), _surf.height()); cv.fill(_Qt2.black)
        pp = _QP(cv); win.viewer._paint_overlays(pp); pp.end()
        im = cv.toImage().convertToFormat(_QI.Format_RGB32)
        rows = np.frombuffer(bytes(im.constBits()), np.uint8).reshape(
            im.height(), im.bytesPerLine() // 4, 4)[:, :im.width(), :3]
        corner = rows[8:140, 8:180].astype(int)
        # amber: red high, green mid, blue low (T.DIM2D is #e0a13a in the dark theme)
        return int(((corner[..., 2] > 150) & (corner[..., 1] > 110) & (corner[..., 1] < 200)
                    & (corner[..., 0] < 110)).sum())
    assert win.viewer._region is not None
    _amber = _corner_amber()
    assert _amber > 40, f"the locator map must draw the window in amber top-left ({_amber} px)"
    assert win.viewer._region_thumb is None or not win.viewer._region_thumb.isNull()
    # a shapes pick committed under the window is shifted back into SOURCE coordinates and
    # stamped with the viewed frame — what Draw Regions pins a shape to
    _stamped = json.loads(win.viewer._stamp_shapes(json.dumps(
        [{"type": "rect", "op": "add", "vertices": [[0, 0], [10, 10]]},
         {"type": "circle", "op": "add", "center": [5.0, 5.0], "radius": 2.0}])))
    assert _stamped[0]["vertices"] == [[32, 16], [42, 26]], _stamped[0]
    assert _stamped[1]["center"] == [37.0, 21.0], _stamped[1]
    _m, _t, _z, _c = win.viewer.coords()
    assert _stamped[0]["frame"] == [_m, _t, _z] == _stamped[1]["frame"], (_stamped, _m, _t, _z)
    # the box hit-test + drag maths the Viewer's mouse handling is built on
    assert _RB.hit((32, 96, 16, 80), 16, 32, 3) == "nw" and \
        _RB.hit((32, 96, 16, 80), 48, 96, 3) == "s" and \
        _RB.hit((32, 96, 16, 80), 50, 60, 3) == "move" and \
        _RB.hit((32, 96, 16, 80), 5, 5, 3) is None
    assert _RB.drag((32, 96, 16, 80), "move", 100, 100, (128, 128)) == (64, 128, 64, 128), \
        "a move stops at the frame edge"
    assert _RB.drag((32, 96, 16, 80), "e", 500, 0, (128, 128)) == (32, 96, 16, 128)
    assert _RB.drag((32, 96, 16, 80), "w", 500, 0, (128, 128))[2] == 80 - _RB.MIN_SIDE, \
        "a side never crosses its opposite"
    win.clear_region()                                 # Run → Clear troubleshooting region
    unwindowed = _await_pull()
    assert win.viewer.region is None and win.runner.region is None
    assert (unwindowed.axes.y, unwindowed.axes.x) == (128, 128)
    assert win._solo_chip.text() == "SOLO t4"
    for c, plane in solo_pixels.items():
        assert np.array_equal(plane, np.asarray(win.viewer._planes[c])), \
            f"clearing the region must restore the full frame's pixels (c={c})"
    assert win.viewer._region_thumb is None, "clearing the window drops the map's picture"
    assert _corner_amber() == 0, "no window → no locator map"
    _ok("T1b region box: the box spans the 128² source; a 64×64 window pulls a 64×64 "
        "payload whose pixels are the full frame's slice, the chip and status name it, the "
        "hit/drag maths pin handles and edges, and clearing it restores the full frame")

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
    # top-left, on the top strip — beside the page switcher, which owns the very corner
    # (V4.00 step 5)
    assert _bg.top() < 40 and _bg.left() < _view.width() / 2, f"badge belongs top-left, at {_bg}"
    assert not _bg.intersects(_view.page_button.geometry()), "…without covering the switcher"

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
    assert _bg.top() < 40 and not _bg.intersects(_view.page_button.geometry()), \
        "…by moving right of it, not off the top strip"
    assert _left_border_amber(), "the frame survives the maximize"
    win.set_maximized(False)
    for _ in range(3):
        app.processEvents()
    assert _view._ts_badge.geometry().top() < 40 \
        and not _view._ts_badge.geometry().intersects(_view.page_button.geometry()), \
        "…and returns to the corner after"

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
    from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap
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
    # and the gesture must actually paint something.
    #
    # Measured over EVERY pixel, not a 1-in-9 grid thresholded at HSV value 40 (2026-08-05).
    # That earlier form did not measure what it claimed: an ROI rect is a ~1.4 px antialiased
    # stroke plus an alpha-30 fill, so on this 128²-in-246 px pane it put ~22 samples over the
    # threshold against a required 21 — and the fill, at value ~18, counted for nothing. A
    # two-sample margin over a sub-pixel stroke phase is not a test of whether the gesture
    # painted; it is a test of whether the stroke happened to land on the sample lattice. A
    # 51 px change in panel width (780 -> 729, from a legitimate layout change) re-centres the
    # rect, moves the stroke off that lattice and takes the count to ZERO while the renderer
    # is drawing 1645 pixels perfectly well. So: count every painted pixel, and separately
    # require that a full-strength stroke is among them.
    _canvas = QPixmap(_surf.width(), _surf.height())
    _canvas.fill(Qt.black)
    _pp = QPainter(_canvas)
    _pv._paint_overlays(_pp)          # the same callback both surfaces invoke
    _pp.end()
    _img = _canvas.toImage().convertToFormat(QImage.Format_RGB32)
    _rows = np.frombuffer(bytes(_img.constBits()), np.uint8).reshape(
        _img.height(), _img.bytesPerLine() // 4, 4)[:, :_img.width(), :3]
    _painted = int((_rows.max(axis=2) > 0).sum())
    _peak = int(_rows.max())
    # The rect is ~33x46 displayed px, so its fill alone is ~1500 painted pixels; a few
    # hundred is a floor that cannot be met by stray antialiasing yet holds for any framing.
    assert _painted > 300, \
        f"the armed gesture painted nothing ({_painted} painted px, peak value {_peak})"
    # ...and the stroke has to be a visible line, not only the translucent fill (~18).
    assert _peak > 60, f"the gesture's stroke is invisible (peak value {_peak})"
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

    # ── P4: the footprint band is a CONTROL (V2.27) ─────────────────────────────
    #
    # "Clicking the footprint changes whole plane / whole volume to per-label" — the band's
    # granularity chip kept its meaning (a FACT: the read cost, in the cost colour) and gained a
    # PILL beside it that edits the node's statistics population. Everything below is driven
    # through the real handlers, because the whole feature is a hit-test + a menu + a mode write,
    # and each of those has a way to be silently wrong: a rect painted where it cannot be
    # clicked, a menu that writes a PARAM named after a Mode, or a layout filter that disagrees
    # with `refresh` and relayouts the card on every keystroke.
    from nodegraph.registry import NODES as _NODES
    from nodelab_v2.document import GraphDocument as _GDf
    from nodelab_v2.scene import GraphScene as _GSf
    from PySide6.QtWidgets import QGraphicsView as _QGV
    _pop_pair = next(((s, m) for s in _NODES.all() for m in s.modes
                      if getattr(m, "is_scope", False)), None)
    assert _pop_pair is not None, \
        "the engine half must ship at least one role='scope' Mode for the band to edit"
    _pspec, _pmode = _pop_pair
    # analysis.threshold's scope is available_in-gated to the histogram methods, so `fixed`
    # (its default) legitimately has NO population — pick a state where the Mode is live.
    _fdoc = _GDf()
    _frec = _fdoc.add_node("analysis.threshold", node_id="fb", x=0, y=0,
                           modes={"method": "otsu"})
    _gdoc_rec = _fdoc.add_node("enhance.gaussian", node_id="fg", x=200, y=0)
    _zrec = _fdoc.add_node("util.zproject", node_id="fz", x=400, y=0)
    _fitem, _gitem2, _zitem = (_PItem(_frec, _fdoc), _PItem(_gdoc_rec, _fdoc),
                               _PItem(_zrec, _fdoc))
    for _it in (_fitem, _gitem2, _zitem):
        _it._layout()

    # (a) the control exists, and it is inside the band it is painted in
    _fctl = [c for c in _fitem.controls() if c.kind == "scope"]
    assert len(_fctl) == 1, f"expected one scope control on Threshold, got {len(_fctl)}"
    _fctl = _fctl[0]
    assert _fctl.obj is _fitem.spec.scope_mode()
    assert _fitem.card_rect().contains(_fctl.rect.center()), \
        "the population pill must lie inside the card it is painted on"
    assert _fctl.rect.top() >= T.HEADER_H, "the pill must not reach into the header"
    assert _fctl.rect.bottom() <= T.HEADER_H + T.GRAN_H - 4, \
        "the pill must stay ABOVE the band's dashed rule — that is what makes it impossible " \
        "for it to overlap row 0, which starts at HEADER_H + GRAN_H"
    _chip_txt = _fitem._foot_slots()[1]
    assert not _fctl.rect.intersects(_fitem._gran_chip_rect(_chip_txt)), \
        f"the pill overlaps the footprint chip ({_fctl.rect} vs " \
        f"{_fitem._gran_chip_rect(_chip_txt)}) — the cap must be measured from the chip's " \
        f"real right edge, since the chip's width depends on its own text"
    # by VALUE: controls() rebuilds its Ctl tuples per call, so identity never holds — what
    # matters is that hit-testing the painted centre returns the same control
    assert _fitem.control_at(_fctl.rect.center()) == _fctl
    assert all(_fitem.card_rect().contains(c.rect.center()) for c in _fitem.controls())

    # (b) the chip stays a fact, abbreviated so it clears the pill; the pill is the population
    _cap, _chip, _pill = _fitem._foot_slots()
    assert (_cap, _chip, _pill) == ("footprint", T.gran_abbr("whole_plane"), "plane ▾"), \
        f"band slots {(_cap, _chip, _pill)}"
    assert _fitem._footprint_line() == "population: plane → reads whole plane"

    # (c) the population Mode is NOT also a body row, and `refresh` agrees with `_layout`.
    # A disagreement makes `want != have` permanently true, so EVERY doc.touch() destroys and
    # recreates every SocketItem on the card and drops a drag in progress.
    assert _pmode.name not in [r[1].name for r in _fitem._rows if r[0] == "mode"], \
        "the population must not appear as a body pill as well as the band's"
    _sock_id = id(_fitem._sockets[("in", "data")])
    _fitem.refresh(); _fitem.refresh()
    assert id(_fitem._sockets[("in", "data")]) == _sock_id, \
        "refresh() rebuilt the card's sockets — `_layout` and `refresh` are filtering the " \
        "mode rows differently (`_modes_on_rows` must be the single predicate)"

    # (d) the menu: every choice, its prose, AND the footprint each would imply
    _menu_seen = {}

    class _FootMenu(_QMenu):
        def exec(self, *a, **k):                          # noqa: A003 - Qt name
            _menu_seen["items"] = [(x.text(), x.toolTip(), x.isEnabled())
                                   for x in self.actions()]
            _menu_seen["tips"] = self.toolTipsVisible()
            return None

    _scene_for_menu = _GSf(_fdoc)
    _view_for_menu = _QGV(_scene_for_menu)
    _scene_for_menu.sync()
    _mitem = _scene_for_menu.node_items["fb"]
    _mctl = next(c for c in _mitem.controls() if c.kind == "scope")
    _before = dict(_mitem.rec.modes)
    _NI.QMenu = _FootMenu
    try:
        _mitem.mousePressEvent(_FakePress(_mctl.rect.center()))
    finally:
        _NI.QMenu = _QMenu
    assert _menu_seen.get("tips") is True, "a QMenu swallows action tooltips unless told not to"
    _entries = _menu_seen.get("items", [])
    assert _entries and not _entries[0][2], \
        "the first entry must be a DISABLED header naming the footprint being read"
    assert "whole plane" in _entries[0][0], _entries[0][0]
    _texts = [t for t, _tip, en in _entries if en]
    assert _texts == list(_pmode.choices), \
        f"the menu must offer exactly the Mode's choices, got {_texts}"
    for _t, _tip, _en in _entries[1:]:
        assert f"<b>{_t}</b>" in _tip, f"option {_t} lost its prose"
        assert "reads " in _tip, \
            f"option {_t} must be annotated with the footprint it implies — the cost belongs " \
            f"in the menu that changes it"
    assert "whole series" in dict((t, tip) for t, tip, _e in _entries)["series"], \
        "the `series` option must say it reads the whole series"
    assert dict(_mitem.rec.modes) == _before, "dismissing the menu must change nothing"

    # (e) a click commits, and BOTH band strings follow
    class _PickFoot(_QMenu):
        def exec(self, *a, **k):                          # noqa: A003 - Qt name
            return next(x for x in self.actions()
                        if x.isEnabled() and x.text() == "volume")

    _NI.QMenu = _PickFoot
    try:
        _mitem.mousePressEvent(_FakePress(_mctl.rect.center()))
    finally:
        _NI.QMenu = _QMenu
    assert _mitem.rec.modes.get("scope") == "volume", \
        f"the click did not commit ({_mitem.rec.modes})"
    assert _mitem.granularity() == "whole_volume", \
        "the resolved footprint must follow the population — that is the whole point of " \
        "editing it from the footprint band"
    assert _mitem._foot_slots()[1] == T.gran_abbr("whole_volume"), "the chip did not follow"
    assert _mitem._foot_slots()[2] == "volume ▾", "the pill did not follow"
    # and it wrote a MODE, not a param named after one (which would serialize into the graph
    # and fold into the recipe hash while being invisible in the inspector)
    assert "scope" not in _mitem.rec.params, \
        "the band wrote a PARAM called `scope` — _open_scope_menu must own its own write " \
        "rather than falling through _open_menu's _write_param tail"

    # (f) the inert cases stay inert, and each says what decides its footprint
    assert not [c for c in _gitem2.controls() if c.kind == "scope"], \
        "a dim-levered node's band must stay a readout — the lever is 11 px above it, and a " \
        "chip menu would have to re-implement the H11 z==1 refusal to be safe"
    assert "2D / 3D lever" in _gitem2._footprint_line()
    assert not [c for c in _zitem.controls() if c.kind == "scope"], \
        "util.zproject's footprint_mode is `method` — a band bound to footprint_mode would " \
        "offer max/mean/none here, i.e. a footprint control that changes the reducer"
    assert _zitem._foot_slots()[1] == "WHOLE VOLUME" and "`method`" in _zitem._footprint_line(), \
        f"zproject's band must read its footprint, never its reducer ({_zitem._foot_slots()})"
    # a collapsed card and a reroute have no controls at all; the tooltip still carries the
    # footprint, which is the only readout a collapsed card has
    _frec.collapsed = True
    _fitem.refresh()
    assert _fitem.controls() == [] and "population" in _fitem._footprint_line()
    _frec.collapsed = False
    _fitem.refresh()

    # (g) available_in gating: `fixed` has no population, so the band reverts to a readout
    _frec.modes["method"] = "fixed"
    _fitem.refresh()
    assert not [c for c in _fitem.controls() if c.kind == "scope"], \
        "under method=fixed there is no histogram and so no population — the band must be a " \
        "plain readout rather than a control that changes nothing"
    assert _fitem._foot_slots()[2] is None and " " in _fitem._foot_slots()[1], \
        "with no pill the chip goes back to its FULL name (there is nothing to clear)"
    _frec.modes["method"] = "otsu"
    _fitem.refresh()

    # (h) inspector parity: one combo, in Footprint, not two
    _finsp = _PInsp()
    _finsp.set_node(_fitem)
    _scombos = [c for c in _finsp.findChildren(_NoWheelCombo)
                if [c.itemText(i) for i in range(c.count())] == list(_pmode.choices)]
    assert len(_scombos) == 1, \
        f"the population needs exactly ONE combo ({len(_scombos)} found) — it moved into the " \
        f"Footprint section, so the Mode section must exclude it"
    # the row (not the combo) carries the control's own prose; the combo carries per-ITEM tips
    assert "statistics population" in _scombos[0].parentWidget().toolTip(), \
        "the row's hover must name the ROLE — mode_identity's third arm, so the pill, the " \
        "menu and this row all label the control the same way"
    assert _scombos[0].currentText() == _fitem.rec.modes.get("scope", "plane"), \
        "the combo must open on the value the card is showing"
    assert any(_scombos[0].itemData(i, _Qt.ToolTipRole) for i in range(_scombos[0].count())), \
        "each population option needs its own item tip, as every other Mode combo has"
    assert "`scope`" in _finsp._last_footprint_note, \
        f"the note must name the resolver: {_finsp._last_footprint_note!r}"
    _finsp.set_node(_gitem2)
    assert "2D / 3D switch" in _finsp._last_footprint_note
    _finsp.set_node(_zitem)
    assert "`method`" in _finsp._last_footprint_note, \
        "the note used to tell every lever-less node it was 'Dimension-agnostic', which is " \
        "false for the four whose footprint a named mode resolves"

    # (i) a RAGGED structure table must not crash the paint (reported 2026-08-06). A structure
    # instance is one array per column under one layer name, and nothing revalidates that they
    # still agree after a second node writes onto it. When they do not, `_build_palette` sized
    # its row map from `id` and its frame selection from `m`/`t`, and indexed past the end:
    # "IndexError: index 208 is out of bounds for axis 0 with size 208" — a crashed RUN, from a
    # colour. Every consumer here promises the opposite ("a colour is never worth a crash"), so
    # the columns are clipped to the row set they all agree on, and said once on stderr.
    from nodelab_v2.viewer import ViewerPanel as _VP
    from nodegraph.structure import StructureTable as _RST
    from nodegraph.provider import ArrayProvider as _RAP
    from nodegraph.dataset import Dataset as _RDS
    from nodegraph.domains import Domain as _RD
    _rax = AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32)
    _rl6 = np.zeros(_rax.shape_for(_RD.VOXEL), np.int64)
    _rl6[0, 0, 0, 0][2:10, 2:10] = 1
    _rl6[0, 0, 0, 0][14:22, 14:22] = 2
    _rds = (_RDS(axes=_rax, metadata={"pixel_size_um": 0.5})
            .with_image(_RAP(np.zeros(_rax.shape_for(_RD.VOXEL))))
            .with_layer(_RD.VOXEL, "labels", _rl6)
            .with_structure(_RST(_RD.LABEL, {
                "id": np.arange(1, 4, dtype=np.int64),
                "m": np.zeros(3, np.int64), "t": np.zeros(3, np.int64),
                "c": np.zeros(3, np.int64), "z": np.zeros(3, float),
                "y": np.array([5.0, 17.0, 25.0]), "x": np.array([5.0, 17.0, 25.0]),
            }, layer="labels", z_kind="plane_index"))
            # ...then a SHORTER `id`, the way a second node writing against another row set does
            .with_structure(_RST(_RD.LABEL, {"id": np.arange(1, 3, dtype=np.int64),
                                            "level": np.array([400.0, 80.0])},
                                 layer="labels", z_kind="plane_index")))
    _rpane = _VP()
    _rpane._dataset, _rpane._axes = _rds, _rax
    _rmem = _rpane._member_layers()
    assert _rmem, "the ragged layer must still be OFFERED, clipped — not dropped"
    for _k, _c in _rmem.items():
        assert len({len(v) for v in _c.values()}) == 1, \
            f"columns still disagree after clipping: {[(n, len(v)) for n, v in _c.items()]}"
        assert len(_c["id"]) == 2, "clipped to the SHORTEST column, i.e. the rows all agree on"
    _rpane._build_palette()            # the call that crashed; must not raise
    assert _rpane._point_count() >= 0
    assert _rpane._warned_ragged, "a table that disagrees with itself must be reported once"

    _ok("P4 footprint band (V2.27): the card's granularity band is now a CONTROL — the chip "
        "keeps its meaning (the read cost, in the cost colour) and a population pill beside it "
        "edits the node's role='scope' Mode. The pill is inside the card and inside the band "
        "above the dashed rule (so it can never reach row 0), and clears the chip with the cap "
        "measured from the chip's REAL right edge — a constant cap was wrong by 5 px on the "
        "first pair tried. The menu carries a disabled header naming the footprint being read "
        "plus, per option, its prose AND the footprint it would imply, so the one control that "
        "changes the population states the cost at the moment of choosing; dismissing changes "
        "nothing, a click re-resolves the footprint and BOTH band strings follow, and it writes "
        "a MODE (a `scope` param would serialize into the graph and re-key the memo while "
        "staying invisible). The population is absent from the body rows and `refresh` agrees "
        "with `_layout`, so a card is not rebuilt on every touch. Inert where there is no "
        "population: a dim-levered node (the lever owns that state, incl. the H11 refusal), "
        "util.zproject (whose footprint_mode is its REDUCER), a collapsed card and method=fixed "
        "— each naming what decides its footprint instead of going quiet. One inspector combo, "
        "in Footprint, and the note names the real resolver rather than calling every "
        "lever-less node dimension-agnostic")

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

    # every channel is shown by default (2026-09-30), and a rebuild that switches one on
    # asks for its plane — so both arrive without a click. Off and on again through the
    # real toggle, so the fast path is still what is under test.
    _await_planes(2)
    win.viewer._on_channel_toggle(1)
    _await_planes(1)
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
    # V4.00: a page's docks live one folder down, under its page id, so two pages' `n3`
    # never bake into one folder — the single-page window is page `pg1` of its workspace
    assert win.doc.default_dock_store("DK").endswith(
        os.path.join("docktest.docks", win.workspace.active, "DK")), \
        win.doc.default_dock_store("DK")

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
    _rgq = win.workspace.compose(win.workspace.active).graph     # the runner's own (qualified) view
    assert win.runner.planned_nodes("DT", _rgq) == [_rq("DK"), _rq("DT")], \
        win.runner.planned_nodes("DT", _rgq)

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
        # the held display state is per NODE since V2.28 (`_views`, one _HeldView per
        # Viewer pane) rather than a set of `_viewer_*` singletons
        _view = win.runner._view_of(_rq("n3"))
        assert _view is not None, "the pulled node must hold a display view"
        _prov = _view.provider
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
        assert win.runner._view_of(_rq("n3")).provider._cache is not _NOC, \
            "the held lazy provider must still read through a LIVE cache after a rebuild"

        # (c) prefetch is gated on what a neighbouring plane costs
        assert win.runner._prefetch_span(_prov) == 0, \
            "a volume-unit compute provider must not warm T-neighbours (one each = a " \
            "whole extra unit)"
        assert win.runner._prefetch_span(win.runner._providers[
            win.runner._node_source_key[_rq("n1")]][0]) == 8, \
            "a store-backed provider still prefetches freely — it is a decompress"

        # (b) a COLD frame never decodes on the GUI thread; a warm one never leaves it
        _c = win.viewer.coords()
        _chans = win.viewer.channels()
        _vax = win.runner._view_of(_rq("n3")).axes
        _cold = (_c[0], _c[1], (_c[2] + 1) % max(1, _vax.z), _c[3])
        _seen.clear()
        _reads.clear()
        assert _vax.z > 1, "the fixture needs a z axis to scrub"
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
        _zmax = max(1, win.runner._view_of(_rq("n3")).axes.z)
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

    # ── V1b PLAY on a whole-unit chain: instant, and paced by the frames ──────────────
    # Reported 2026-08-06 as "the 3D deconvolved set can't load frames" — pressing play on
    # exactly the provider the section above sets up. V2.23 had made play mean "decode the
    # series, THEN run it", which is right when a frame is a decompress and catastrophic when
    # it is a whole-volume Richardson–Lucy: the preload queued the entire T range at the
    # byte-backed width (four concurrent volumes, ~130 s and ~30 GB of working set each) and
    # HELD playback behind it. Nothing reached the screen, and the four volumes crowded out
    # the frame being displayed. The two halves of the fix are both here.
    win.pull_node("n3")                  # the section above ends on an invalidate
    t0 = time.time()
    while win.runner._busy and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    _view3 = win.runner._view_of(_rq("n3"))
    assert _view3 is not None and getattr(_view3.provider, "volume_unit", False), \
        "this probe needs the whole-unit provider the section above pulled"
    assert win.runner.frames_are_reads("n3") is False, \
        "a frame of a lazy chain is the node RUNNING, not a read — the whole policy hangs " \
        "off this one predicate"
    assert win.runner.preload_series("n3", win.viewer.coords(),
                                     win.viewer.channels()) == 0, \
        "play must not queue a whole-unit series ahead of the cursor — that IS the bug"
    win.viewer._play_btns["t"].setChecked(True)          # → _on_play → playing → _on_playing
    app.processEvents()
    try:
        assert not win.viewer.play_gated(), \
            "playback must start IMMEDIATELY on a computed series — the gate was a wait of " \
            "hours with nothing on screen"
        assert win.viewer._play_timer.isActive(), "…and the timer must actually be running"
        assert win.viewer._play_paced, \
            "a computed series advances on DELIVERY: a wall clock would spin the cursor " \
            "through frames that cannot answer at 8 fps and show one stale volume a couple " \
            "of minutes later"
        # the pacing holds the cursor while a frame is outstanding, and one delivery frees it
        win.viewer._awaiting_frame = True
        win.viewer._awaiting_since = time.perf_counter()
        _t_before = win.viewer._sliders["t"].value()
        win.viewer._tick_play()
        assert win.viewer._sliders["t"].value() == _t_before, \
            "a tick must not advance past a frame that is still computing"
        win.viewer.show_planes("n3", {}, _view3.axes, 0.0)     # the frame lands
        assert not win.viewer._awaiting_frame, "a delivery releases the next advance"
        win.viewer._tick_play()
        assert win.viewer._sliders["t"].value() != _t_before, "…and then it advances"
    finally:
        win.viewer._play_btns["t"].setChecked(False)
        app.processEvents()
    assert not win.viewer._play_timer.isActive() and win.viewer._playing_axis is None

    # …while a store-backed series keeps the V2.23 behaviour it was written for: the frames
    # are bytes, the preload is over in a second or two, and waiting for it is what makes
    # playback smooth instead of "loading every time" (reported 2026-08-05).
    from nodegraph.provider import ArrayProvider as _AP
    from nodelab_v2.runner import EngineRunner as _ERn
    from nodelab_v2.runner import _HeldView as _HV
    _bytes_view = _HV(_AP(np.zeros((1, 4, 1, 1, 32, 32), np.uint16)),
                      AxisSizes(m=1, t=4, z=1, c=1, y=32, x=32),
                      win.runner._rev(_rq("n3")), None, np.uint16)
    win.runner._views[_rq("n3")] = _bytes_view
    assert win.runner.frames_are_reads("n3") is True
    assert _ERn._preload_jobs(_bytes_view.provider) == _ERn.PRELOAD_JOBS

    # …and a PER-PLANE computed series (a stitched mosaic, a Z-projection — the recorded
    # 2026-08-10 stutter) sits between the two: play HOLDS it while its frames prepare at
    # the full preload width, exactly like bytes — but the hold is capped by measured cost
    # (PLAY_PREPARE_MAX_S), so a chain that computes for minutes releases playback rather
    # than holding a blank stare. Cheap identity kernel → the prepare finishes in
    # milliseconds and the deterministic endpoint is "gate down, timer running".
    from nodegraph.streaming import MapComputeProvider as _MCP
    from nodegraph.streaming import TileCache as _TCache
    from nodelab_v2.window import PLAY_PREPARE_MAX_S as _PREP_CAP
    _pbase = _AP(np.zeros((1, 6, 1, 1, 32, 32), np.uint16))
    _mapp = _MCP(_pbase, lambda p, *a: p, fp="probe-perplane", cache=_TCache())
    win.runner._views[_rq("n3")] = _HV(_mapp, _pbase.axes, win.runner._rev(_rq("n3")),
                                        None, np.uint16)
    win.runner._planes.clear()   # planes cached by the sections above share this node id —
    # a warm series would make the preload a no-op and this probe about nothing
    assert _ERn._preload_jobs(_mapp) == _ERn.PRELOAD_JOBS, \
        "a per-plane compute must warm at the full measured width (2026-08-10) — one job " \
        "never outruns playback consuming a frame per frame"
    win.viewer._play_btns["t"].setChecked(True)      # → _on_play → playing → _on_playing
    try:
        # synchronous half: the handler ran inside setChecked, so the gate is up and the
        # timer is parked BEFORE any preload tick can possibly have been delivered
        assert win.viewer.play_gated(), \
            "a per-plane computed series must be HELD while its frames prepare — playing " \
            "it cold is the recorded stutter: every frame at its own decode latency"
        assert not win.viewer._play_timer.isActive(), \
            "…which means the wall-clock timer must not be running yet"
        assert win.viewer._play_paced, \
            "pacing stays on DELIVERY for a computed series — over warm frames it advances " \
            "at the wall clock anyway, and at the cache edge it stays honest"
        _t0 = time.time()
        while win.runner.preloading() and time.time() - _t0 < 60:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()                          # the finished signal drops the gate
        assert not win.viewer.play_gated(), \
            "the preload finished — the gate must drop on its own"
        assert win.viewer._play_timer.isActive(), \
            "…and playback must actually be running off the warm frames"
    finally:
        win.viewer._play_btns["t"].setChecked(False)
        app.processEvents()
    # the ETA half of the cap, pinned as arithmetic — CAPPED holds only, i.e. a series too
    # big to ever be fully resident: two ticks in, a preparation whose measured rate
    # projects past PLAY_PREPARE_MAX_S releases the hold, without cancelling the preload,
    # which keeps warming behind the now-running playback
    win.viewer.set_play_gate(True)
    win._preload_hold_t0 = time.monotonic() - (_PREP_CAP + 1.0)
    win._preload_hold_capped = False                 # fits the budget: waits to completion
    win._on_preload_progress("n3", 2, 1000)
    assert win.viewer.play_gated(), \
        "a series that FITS the budget holds to completion — the cap must not touch it " \
        "(user decision 2026-08-10: pressing play on a series that CAN be made smooth " \
        "means 'make it smooth')"
    win._preload_hold_capped = True                  # larger than the budget: capped
    win._on_preload_progress("n3", 2, 1000)
    assert not win.viewer.play_gated(), \
        "a capped hold whose ETA projects past PLAY_PREPARE_MAX_S must release playback"
    win._drop_play_gate("n3")                        # already down: must be a quiet no-op
    win.runner._views.pop(_rq("n3"), None)

    _ok("V1b play policy by COST (2026-08-06 whole-unit; 2026-08-10 per-plane): a "
        "whole-volume chain queues nothing, raises no gate and starts playing on the spot, "
        "paced by delivery; a per-plane computed series (stitch / Z-projection) is held "
        "like bytes while it warms at the full measured preload width, and the hold is "
        "capped by the preload's own measured rate plus a wall-clock watchdog, so a "
        "long-running chain releases playback instead of holding a blank stare; a "
        "byte-backed series keeps the V2.23 preload that made it smooth")

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
        lambda nid, planes, rect, coords: _detail_seen.append((nid, planes, rect, coords)))
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
    _dview = win.runner._view_of(_rq(_node))
    assert _dview is not None, "the viewed node must hold a display view"
    _prov, _ax = _dview.provider, _dview.axes
    assert _prov is not None and _ax is not None
    _rect = (0.30, 0.30, 0.55, 0.55)
    win.runner.request_detail(_node, _vp.coords(), sorted(_vp._planes), _rect,
                              _RR.MAX_DISPLAY_DIM)
    t0 = time.time()
    while not _detail_seen and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.005)
    assert _detail_seen, "no detail patch arrived"
    _nid, _dplanes, _drect, _dcoords = _detail_seen[-1]
    assert _nid == _rq(_node) and _dplanes, (_nid, list(_dplanes))
    assert tuple(_dcoords)[:3] == tuple(_vp.coords())[:3], \
        "the patch must carry the (m,t,z) it was read at — the panel's staleness check " \
        "hangs off it"
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
        win.runner._payload_coords(_vp.coords(), _dview.pin), _ax)
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

    # a patch that OUTLIVED ITS FRAME is refused too (2026-08-10, "the frames are going
    # back to previously loaded frames"): the generation only advances on a new request,
    # so during playback / a fast scrub a windowed read that took longer than the frame
    # cadence lands generation-current — the (m,t,z) stamp is the guard that catches it
    _m0, _t0c, _z0, _c0 = _vp.coords()
    _vp.on_detail_ready(_nid, _dplanes, _drect, (_m0, _t0c + 1, _z0, _c0))
    assert _vp._detail_rect is None, \
        "a detail patch from a previous frame must not be painted over the current one"
    _vp.on_detail_ready(_nid, _dplanes, _drect, (_m0, _t0c, _z0, _c0))
    assert _vp._detail_rect is not None, "…while the current frame's patch still lands"
    _surf.fit()
    app.processEvents()
    _vp._request_detail()
    app.processEvents()
    assert _vp._detail_rect is None

    # …and PLAYBACK requests none at all: a full-detail windowed read per frame lands
    # after the frame it described and starves the pool the preload and the frame decodes
    # share. The parked frame sharpens on pause instead (ViewerPanel._stop_play).
    _calls: list = []
    _saved_cb = _vp.detail_cb
    _vp.detail_cb = lambda *a: _calls.append(a)
    try:
        _vp._playing_axis = "t"
        _vp._request_detail()
        assert not _calls, "playback must not issue detail reads"
    finally:
        _vp._playing_axis = None
        _vp.detail_cb = _saved_cb

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
        if nid == _rq(_node):
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
    # V2.23: the two SPATIAL comparators are defined on the image, so the quad has to be told
    # where in the image it is. The overview quad is (0,0,1,1) and a viewport detail patch is
    # its own rect — off the quad's own uv a patch restarted the checkerboard and slid the
    # wipe divider to the middle of the zoom.
    assert "uniform vec4  u_rect" in frag and "u_rect.xy + v_uv * u_rect.zw" in frag, \
        "the comparators are computed off the quad's own uv, not the image's"
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
        "assert p.uniformLocation('u_rect')>=0,'u_rect optimized away'\n"
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
    # ...and a PATCH of the image composites to the same pixels that part of the whole image
    # did — which is only true if `region` (the CPU mirror of `u_rect`) reaches both spatial
    # comparators. The detail quad is drawn OVER the overview, so any disagreement here is a
    # visible seam that appears the moment you zoom.
    for _mode, _param in ((3, 8.0), (4, 0.25)):
        _tint = {0: (0, 255, 0), 1: (255, 255, 0)}
        _lim = {0: (0.0, 4095.0), 1: (0.0, 4095.0)}
        _all = np.frombuffer(_cwc({0: _base, 1: _ovl}, _tint, _lim, None,
                                  {1: (_mode, _OP, _param)}).constBits(),
                             dtype=np.uint8).reshape(16, -1, 3)[:, :16, :]
        for _f0, _f1, _sl in ((0.0, 0.5, slice(0, 8)), (0.5, 1.0, slice(8, 16))):
            _half = {0: _base[:, _sl], 1: _ovl[:, _sl]}
            _pimg = _cwc(_half, _tint, _lim, None, {1: (_mode, _OP, _param)},
                         region=(0.0, 1.0, _f0, _f1))
            _pat = np.frombuffer(_pimg.constBits(), dtype=np.uint8).reshape(16, -1, 3)[:, :8, :]
            assert np.array_equal(_pat, _all[:, _sl, :]), (_mode, _f0)
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
        "keeps every non-overlay graph unchanged. V2.23: the two SPATIAL comparators are "
        "computed in IMAGE coordinates (`u_rect`, which the driver confirms survives linking, "
        "mirrored by `region=`), so a zoomed detail patch composites byte-identically to that "
        "part of the whole image instead of restarting the checkerboard and sliding the wipe "
        "divider into the middle of the zoom" % _compiled)

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
    from nodegraph.iterate import (
        ITERATE_OP as _IT_OP, SEG_FROM as _SEG_FROM, SEG_TO as _SEG_TO,
        SWEEP_KEY as _SW_KEY)

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
    idoc.connect("IRS", "out", "ITT", _SEG_TO)      # the SEGMENT's end (V2.22)

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
    # the segment's END keeps its id, wearing the selector: that is what lets an iteration
    # happen automatically when anything below it is pulled, with nothing re-routed
    assert _run.nodes["IRS"].op_key == _IT_OP, "the selector must wear the end node's id"
    assert _run.nodes["ITT"].params.get("__passthrough__") is True, \
        "the card only passes the selected payload on; it must not choose twice"
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
    # the target picker (V2.22): the panel scrapes the chain and WIRES the choice, so the
    # thing to prove is that the combo is not a parallel notion of what is being iterated —
    # it must produce the same driver edge a drag makes, and it must never offer a target
    # the rewrite would then refuse.
    from PySide6.QtWidgets import QComboBox as _QCombo2

    def _target_boxes():
        iinsp.set_node(_NItem(idoc.nodes["ITT"], idoc))
        app.processEvents()
        return [c for c in iinsp.findChildren(_QCombo2)
                if c.count() and c.itemData(0) == ""]

    _boxes = _target_boxes()
    assert len(_boxes) == 2, "V0 plus one spare row"          # variables == 1
    assert "Threshold" in _boxes[0].currentText(), \
        f"the slot's existing driver wire must show as its selection ({_boxes[0].currentText()!r})"
    _spare_items = [_boxes[1].itemText(i) for i in range(_boxes[1].count())]
    assert not any("Threshold · Threshold" in t for t in _spare_items), \
        "a target another slot already drives must leave the other slots' menus"
    assert not any("2D / 3D" in t or "dim" == t for t in _spare_items), _spare_items
    # the spare row adds a SECOND variable: it raises `variables` rather than making the
    # user find that Mode first, and picking a dropdown sets the slot's Type to text
    _k = next(i for i in range(_boxes[1].count())
              if "Connectivity" in _boxes[1].itemText(i))
    _boxes[1].setCurrentIndex(_k)
    app.processEvents()
    assert ("ITT", "var1", "ILB", "connectivity") in idoc.edges, idoc.edges
    assert idoc.nodes["ITT"].modes["variables"] == "2"
    assert idoc.nodes["ITT"].modes["v1_type"] == "number"
    _boxes = _target_boxes()
    _k = next(i for i in range(_boxes[1].count()) if "method" in _boxes[1].itemText(i))
    _boxes[1].setCurrentIndex(_k)
    app.processEvents()
    assert ("ITT", "var1_text", "ITH", "__mode__:method") in idoc.edges, idoc.edges
    assert idoc.nodes["ITT"].modes["v1_type"] == "text", \
        "a Mode is swept by NAME — the picker must move the slot onto its string output"
    assert not any(e[1] == "var1" for e in idoc.edges), \
        "re-pointing a slot must replace its wire, not leave the old one drawn"
    # …and back, so the rest of the section sees the one-variable card it set up
    idoc.set_iterate_target("ITT", 1)
    idoc.nodes["ITT"].modes["variables"] = "1"
    assert [e for e in idoc.edges if e[0] == "ITT"] == \
        [("ITT", "var0", "ITH", "threshold")]

    # the SEGMENT (V2.22): the card is a control, not a stage. Nothing routes through it,
    # the result leaves through the series' own end node, and the two GUI consequences are
    # that the iteration strip belongs to that end node and that viewing a node INSIDE the
    # segment — the most ordinary thing to do while tuning a swept param — resolves to a
    # clone instead of raising on an id the rewrite deleted.
    _starts, _ends = idoc.iterate_segment("ITT")
    assert (_starts, _ends) == ((), ("IRS",)), (_starts, _ends)
    assert idoc.iterate_card_at("IRS") == "ITT", \
        "the strip belongs to the segment's END — that is where the selector sits"
    assert idoc.iterate_card_at("ITT") == "ITT" and idoc.iterate_card_at("ILB") is None, \
        "a node in the MIDDLE is served by one clone; there is nothing to select there"
    _al = idoc.iterate_aliases()
    assert _al == {"ITH": "ITH#ITT@1", "ILB": "ILB#ITT@1"}, _al
    from nodelab_v2.workspace import Workspace as _WS2
    _saved_src = win.runner.source            # the probe's idoc is not the window's
    _iws = _WS2.single(idoc)
    win.runner.source = _iws
    try:
        _runq = _iws.compose(_iws.active).graph          # the composed, page-qualified run graph
        _pq = lambda n: f"{_iws.active}/{n}"             # noqa: E731
        assert win.runner._pull_id("ILB", _runq) == _pq("ILB#ITT@1"), \
            "double-clicking a node inside the segment must pull its clone, not KeyError"
        assert win.runner._pull_id(_pq("ILB"), _runq) == _pq("ILB#ITT@1")
        assert win.runner._pull_id("IRS", _runq) == _pq("IRS") and \
            win.runner._pull_id("IL", _runq) == _pq("IL")
    finally:
        win.runner.source = _saved_src
    # a branch off the END needs no re-routing: it reads the selector
    idoc.add_node("view.viewer", node_id="IVW")
    idoc.connect("IRS", "out", "IVW", "data")
    _run2 = idoc.to_graph(for_run=True, materialize=True, unroll_iterate=True)
    assert [e.src for e in _run2.preds("IVW")] == ["IRS"], \
        "the rewrite must leave a consumer of the end node pointing at the same id"
    idoc.remove_node("IVW")

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
        "table — which stays a UI annotation the run graph strips; the V2.22 target picker "
        "shows the slot's existing wire as its selection, hides what another slot already "
        "took, and BUILDS the same driver edge a drag would — its spare row raising "
        "`variables` and a dropdown target moving the slot onto its string output; the "
        "V2.22 SEGMENT puts the selector on the series' END — the strip belongs to that "
        "node, a branch hanging off it is left pointing at the same id, and a node INSIDE "
        "the segment resolves to the strip's clone instead of raising on an id the rewrite "
        "deleted; the Viewer's iteration strip shows, selects and hides; and the wire "
        "round-trips through save/load")

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
    assert cdone.get("id") == _rq("cms"), cdone

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

    # ── C2: the Points overlay's Z gating + colour modes (2026-08-03 / 08-04) ──
    #
    # Two reports, one root cause. First "particle detection does not show points detected":
    # a 3-D detection's z is a continuous DEPTH, so its particles sit on the planes they were
    # found at (2–34 on the reported stack) and the plane a viewer opens on held none of them
    # — while the status line SUPPRESSED the count when it was zero, so there was no way to
    # tell "detected nothing" from "detected 41 000, none on this plane".
    #
    # The first fix for that was wrong, and this probe pins the corrected one. Making a
    # subpixel layer project REGARDLESS of `z_project` broke two things at once, which became
    # the second report ("Show points from every Z is not selected, it still shows up" and
    # "all beads one colour"): the checkbox went inert, and drawing every plane's markers at
    # once overlaps thousands of glyphs into a flat wash, so a per-point palette stops reading
    # as a palette at all. So `z_project` is authoritative for BOTH z_kinds, and the
    # discoverability problem is carried entirely by the status line — which costs the picture
    # nothing.
    from nodegraph.dataset import Dataset as _DS
    from nodegraph.provider import ArrayProvider as _AP
    from nodegraph.structure import point_table as _pt

    _zax = AxisSizes(m=1, t=1, z=8, c=1, y=32, x=32)
    _zvol = np.zeros((1, 1, 8, 1, 32, 32), dtype=float)
    _zds = _DS(axes=_zax, metadata={"pixel_size_um": 0.2}).with_image(_AP(_zvol))
    # a 3-D cloud whose subpixel z rounds onto planes 3 / 4 / 5 / 5
    _zds = _zds.with_structure(_pt(np.array([[3.2, 8.0, 8.0], [4.1, 12.0, 20.0],
                                             [4.8, 20.0, 10.0], [5.3, 24.0, 24.0]]),
                                   z_kind="subpixel", layer="particles"))
    # …and a per-plane 2-D detection whose two rows belong to planes 1 and 6
    _zds = _zds.with_structure(_pt(np.array([[6.0, 6.0], [26.0, 26.0]]),
                                   z=np.array([1.0, 6.0]), z_kind="plane_index",
                                   layer="perplane"))
    win.viewer.show_result("zpts", {0: _zvol[0, 0, 0, 0]}, _zax, 0.01, dataset=_zds)
    app.processEvents()
    assert win.viewer.overlays.points.enabled
    assert not win.viewer.overlays.points.z_project, "z_project must default OFF"

    _ON = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 2, 6: 1, 7: 0}   # per plane, both layers summed
    for _z, _want in _ON.items():
        win.viewer._sliders["z"].setValue(_z)
        app.processEvents()
        _on, _frame = win.viewer._point_tally()
        # `_point_tally` counts off the DATASET now, so it no longer refreshes the drawn
        # geometry as a side effect — materialise it explicitly before reading _geo_points
        win.viewer._ensure_geometry()
        # (a) OFF draws the viewed plane's own detections and nothing else — the setting gates
        #     a subpixel layer exactly as it gates a plane_index one
        assert len(win.viewer._geo_points) == _want, (
            f"z={_z}: z_project OFF drew {len(win.viewer._geo_points)}, expected {_want} — "
            f"the setting must gate a 3-D layer too")
        assert _on == _want, (_z, _on, _want)
        # (b) the tally still knows the whole frame, which is what keeps an empty plane
        #     distinguishable from an empty result while nothing is projected
        assert _frame == 6, (_z, _frame)
        # (c) ON draws every point, off-plane ones flagged so the renderer dims them
        win.viewer.overlays.points.z_project = True
        win.viewer._geo_key = None
        win.viewer._ensure_geometry()
        _all = win.viewer._geo_points
        assert len(_all) == 6, (_z, len(_all))
        assert sum(1 for mk in _all if mk.on_plane) == _want, (_z, _want)
        win.viewer.overlays.points.z_project = False
        win.viewer._geo_key = None

    # the status line names the off-plane remainder on a plane holding NONE of them
    win.viewer._sliders["z"].setValue(0)
    app.processEvents()
    win.viewer.show_result("zpts", {0: _zvol[0, 0, 0, 0]}, _zax, 0.01, dataset=_zds)
    app.processEvents()
    _txt = win.viewer._status.text()
    assert "0 points" in _txt and "6 on other Z" in _txt, \
        f"an empty plane inside a full volume must still say what was detected: {_txt!r}"

    # each colour mode is what it claims. per_layer being ONE colour per layer is correct
    # rather than broken — with a single Point layer it equals `single`, which is what was
    # reported as "all beads one colour".
    win.viewer._sliders["z"].setValue(4)
    win.viewer.overlays.points.z_project = True          # all 6 marks across both layers
    win.viewer._geo_key = None
    win.viewer._ensure_geometry()
    _mk = win.viewer._geo_points
    _rend = win.viewer._renderer
    # NOT `_seen`: that name is still captured by the `plane_ready` lambda connected far
    # above (the cold/warm decode probe), which appends to it. Rebinding it to a dict here
    # made every later plane_ready raise AttributeError inside the handler — printed as a
    # traceback that reads like a failed probe while every assertion still passed. Whether
    # it fires at all is a timing accident, so it surfaces and vanishes with unrelated edits.
    _colours = {}
    for _mode in ("single", "per_z", "per_point", "per_layer"):
        win.viewer.overlays.points.color_mode = _mode
        _colours[_mode] = len({_rend._point_color(win.viewer.overlays.points, k).name()
                               for k in _mk})
    assert len({mk.layer for mk in _mk}) == 2, "the fixture must carry TWO Point layers"
    assert _colours["single"] == 1, _colours
    assert _colours["per_point"] == 6, (
        f"per_point must give one colour PER POINT, got {_colours} — 4 for 6 marks was the "
        f"cross-layer id collision: every layer numbers ids from 0 and the neighbour graph "
        f"is per layer, so two layers' objects preferred (and kept) the same palette slots")
    assert _colours["per_layer"] == 2, \
        f"per_layer must be one colour per layer, got {_colours}"
    # per_z: one colour per PLANE, over the 5 planes these 6 marks occupy — the cloud rounds
    # onto 3/4/5/5 and the per-plane layer sits on 1 and 6
    assert sorted({mk.zplane for mk in _mk}) == [1, 3, 4, 5, 6],         sorted({m.zplane for m in _mk})
    assert _colours["per_z"] == 5, f"per_z must give one colour per Z plane, got {_colours}"
    # …and with projection OFF every drawn mark is on the viewed plane, so per_z is ONE
    # colour. That is correct, not a repeat of the per_layer confusion — but it is a trap
    # worth pinning, because "colour by Z" showing one colour reads exactly like a bug.
    win.viewer.overlays.points.color_mode = "per_z"
    win.viewer.overlays.points.z_project = False
    win.viewer._geo_key = None
    win.viewer._ensure_geometry()
    _flat = win.viewer._geo_points
    assert _flat and len({mk.zplane for mk in _flat}) == 1
    assert len({_rend._point_color(win.viewer.overlays.points, k).name()
                for k in _flat}) == 1, "per_z with projection off must be one colour"
    win.viewer.overlays.points.color_mode = "single"
    win.viewer.overlays.points.z_project = False
    win.viewer._geo_key = None

    _ok("C2 Points overlay Z gating + colour (2026-08-03/04): reported first as 'particle "
        "detection does not show points detected', then as 'Show points from every Z is not "
        "selected, it still shows up' + 'all beads one colour' — the same root cause twice. A "
        "3-D detection's z is a continuous depth, so its particles sit on the planes they "
        "were found at and the plane a viewer opens on can hold none; the first fix made a "
        "subpixel layer project REGARDLESS of the setting, which left the checkbox inert and "
        "overlapped every plane's glyphs into one flat wash that destroyed the per-point "
        "palette too. Now z_project is authoritative for BOTH z_kinds (proved across 8 "
        "planes, on and off, against a subpixel and a plane_index layer side by side), "
        "discoverability rides on the status line alone — counted off the DATASET rather than "
        "the drawn marks, so the off-plane remainder is still named when nothing is projected "
        "— and all four colour modes are each what they claim: one colour, one per Z PLANE "
        "(V2.23, the mode that makes a projected 3-D cloud read as depth — and legitimately "
        "one colour when nothing is projected, since every drawn mark is then on the viewed "
        "plane), a colour per POINT (6 for 6 marks; it was 4, because every layer numbers ids "
        "from 0 and the neighbour graph is per layer, so two layers preferred and kept the "
        "same palette slots), and one per LAYER (which with a single layer equals `single`)")

    # ── V4 a computed node must LOOK like its raw source ────────────────────────
    # Every streaming provider computes in float64, so Stitch/Gaussian/... reach the Viewer
    # as floats even though their values are still the same integer counts. The LUT slider
    # extent keyed on that dtype instead of on the payload's own `bit_depth` declaration,
    # so a stitched mosaic got a slider spanning its DATA range while the raw source beside
    # it got the SENSOR range — 135-1564 against 0-4095 on real dim 12-bit data.
    from nodelab_v2.glview import pack_u16
    _vp2 = win.viewer
    _bd_save = _vp2._bit_depth

    # (1) the LUT extent follows the DECLARATION, not the dtype
    _u16 = np.full((8, 8), 700, np.uint16)
    _f64 = _u16.astype(np.float64)
    _vp2._bit_depth = 12
    assert _vp2._display_range(_u16) == (0.0, 4095.0)
    assert _vp2._display_range(_f64) == (0.0, 4095.0), \
        "a float plane that still declares bit_depth must get the SENSOR range, like raw"
    # ...and a node that genuinely rescaled (bit_depth dropped) still reads its own range
    _vp2._bit_depth = None
    _norm = np.linspace(0.0, 0.25, 64).reshape(8, 8)
    assert _vp2._display_range(_norm) == (0.0, 0.25), \
        "a rescaled [0,1] image must NOT be given a sensor range it no longer has"
    assert _vp2._display_range(_u16) == (0.0, 65535.0)   # integer, no declaration
    _vp2._bit_depth = _bd_save

    # (2) the texture packing range CANCELS in the shader — pinned here because it looks
    #     like it should be keyed to bit_depth, and pinning it there was tried and reverted.
    #     `_upload` hands its (dmin, dmax) to the shader as u_win, which renormalizes the
    #     LUT window against it, so the rendered value is (v - clim_lo)/(clim_hi - clim_lo)
    #     whatever scale was used. Assert that equivalence directly, on two frames whose
    #     EXTREMES differ — the case where a per-plane scale could have drifted.
    def _rendered(plane, clim):
        u16, dmin, dmax = pack_u16(plane)
        span = max(dmax - dmin, 1e-9)
        vlo, vhi = (clim[0] - dmin) / span, (clim[1] - dmin) / span
        texel = u16.astype(np.float64) / 65535.0
        return np.clip((texel - vlo) / max(vhi - vlo, 1e-6), 0.0, 1.0)

    _clim = (169.0, 1526.0)
    _fA = np.array([[128.0, 700.0, 1572.0]])          # two frames of the same series,
    _fB = np.array([[210.0, 700.0, 1408.0]])          # differing only at the extremes
    _a = _rendered(_fA, _clim)[0][1]                  # the 700-count pixel in each
    _b = _rendered(_fB, _clim)[0][1]
    _raw = _rendered(np.array([[128, 700, 1572]], np.uint16), _clim)[0][1]
    assert abs(_a - _b) < 1e-4, f"one intensity must render alike across frames ({_a},{_b})"
    assert abs(_a - _raw) < 1e-3, \
        f"a computed node must render an intensity like its raw source ({_a} vs {_raw})"
    # integer planes keep their exact, full-width scale
    assert pack_u16(_u16)[1:] == (0.0, 65535.0)
    assert pack_u16(np.full((4, 4), 3, np.uint8))[1:] == (0.0, 255.0)
    # a flat float plane must not divide by zero
    assert pack_u16(np.zeros((4, 4)))[1:] == (0.0, 1.0)

    _ok("V4 computed-node display parity: the LUT slider extent now follows the payload's "
        "`bit_depth` DECLARATION rather than the streaming dtype, so a Stitch/Gaussian gets "
        "the same 0-4095 sensor slider as its raw source instead of a data-derived one "
        "(measured 135-1564 before), while a node that really rescaled has dropped "
        "bit_depth and still reads its own range; and the texture packing range is pinned "
        "as CANCELLING in the shader — one intensity renders identically across frames with "
        "different extremes, and identically to the raw uint16 path, which is why keying it "
        "to bit_depth was tried and reverted as a no-op that only cost clipping headroom")

    # ── C4: two branches run independently and the finished one stays usable ──
    #
    # Asked for as "show that the two are running independently, and if one is done earlier
    # let us view the finished one and manipulate it". Four separate things had to change and
    # each is pinned here, because each failed silently rather than loudly:
    #
    #   1. the second request was DISCARDED, not queued (one `_pending` slot, latest-wins),
    #   2. a result was retired the moment anything else was pending — so the first branch's
    #      payload was thrown away exactly when the user wanted to look at it,
    #   3. one global epoch meant editing the finished branch cancelled the running one,
    #   4. `set_run_plan` cleared every card outside the newest plan, so the branches could
    #      never be shown in different states at the same time.
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=1, c=2, y=64, x=64)
    win.file_new()
    app.processEvents()
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner._raw_src.clear()
    win.runner.invalidate()
    bdoc = win.doc
    bdoc.add_node("io.load", node_id="bl", x=0, y=0)
    bdoc.set_meta_seed("bl", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                         metadata=dict(_RN._SYNTH_META)))
    bdoc.add_node("channel.split", node_id="bs", x=180, y=0)
    bdoc.connect("bl", "image", "bs", "data")
    for _k, _y in ((0, -120), (1, 120)):
        bdoc.add_node("enhance.gamma", node_id=f"bg{_k}", x=380, y=_y,
                      params={"gamma": 0.7 + 0.2 * _k})
        bdoc.connect("bs", f"ch{_k}", f"bg{_k}", "data")
    app.processEvents()

    # (1) a request made while another runs is QUEUED, not dropped
    _q: list = []
    _cancels: list = []
    win.runner.queued.connect(lambda nid, d: _q.append((nid, d)))
    win.runner.cancelled.connect(_cancels.append)
    win.runner._busy = True                       # pretend a long branch is running
    win.runner.pull("bg0")
    win.runner.pull("bg1")
    assert win.runner.queued_nodes() == (_rq("bg0"), _rq("bg1")), win.runner.queued_nodes()
    assert [n for n, _d in _q] == [_rq("bg0"), _rq("bg1")], _q
    win.runner.pull("bg0")                        # a repeat KEEPS its place, adds no second
    assert win.runner.queued_nodes() == (_rq("bg0"), _rq("bg1")), win.runner.queued_nodes()
    assert win.runner.queue_depth() == 2

    # (2) the queued branches' cards say `queued` while the other one holds `running`
    win.scene.set_run_plan("bgX", ["bl", "bs", "bgX"])   # the (fictional) running branch
    win.scene._set_state("bgX", "running")
    win.scene.set_queued("bg0", win._local_ids(win.runner.planned_nodes("bg0")))
    assert win.scene._run.get("bgX", ("",))[0] == "running", win.scene._run.get("bgX")
    assert win.scene._run.get("bg0", ("",))[0] == "queued", win.scene._run.get("bg0")
    win.runner._queue.clear()
    win.runner._busy = False

    # (3) both branches really do produce their OWN result, one after the other
    _pulls.clear()
    win.pull_node("bg0")
    _await_pull()
    win.pull_node("bg1")
    _await_pull()
    _r0 = win.runner._results.get((_rq("bg0"), win.runner._rev(_rq("bg0"))))
    _r1 = win.runner._results.get((_rq("bg1"), win.runner._rev(_rq("bg1"))))
    assert _r0 is not None and _r1 is not None, sorted(win.runner._results)
    _p0 = _r0[0].image.read_region(0, 0, 0, 0, 0, 0, 8, 0, 8)
    _p1 = _r1[0].image.read_region(0, 0, 0, 0, 0, 0, 8, 0, 8)
    assert not np.allclose(_p0, _p1), \
        "both branches produced the SAME pixels — the two channel taps collapsed"

    # (4) the finished branch is served WITHOUT the pull slot, while another run holds it
    win.runner._busy = True
    _served: list = []
    _fin = win.runner.finished.connect(lambda nid, *a: _served.append(nid))
    assert win.runner._serve_finished(_rq("bg0"), None, None), \
        "a finished branch must be viewable while another branch runs"
    app.processEvents()
    assert _served == [_rq("bg0")], _served
    assert win.runner.queue_depth() == 0, "serving from cache must not queue a pull"
    win.runner.finished.disconnect(_fin)
    win.runner._busy = False

    # (5) editing the finished branch does NOT cancel the other one
    win.runner._runs.clear(); win.runner._run_cones.clear()
    win.runner._runs[9001] = _rq("bg1")                  # pretend bg1 is still computing
    win.runner._run_cones[9001] = frozenset(win.runner.planned_nodes("bg1"))
    _cancels.clear()
    bdoc.nodes["bg0"].params["gamma"] = 0.55
    bdoc.touch("bg0")                                    # the inspector's edit path
    app.processEvents()
    assert 9001 in win.runner._runs, \
        "editing the FINISHED branch cancelled the branch still running — one global epoch"
    assert _cancels == [], _cancels
    # ...while an edit inside its OWN cone does cancel it
    bdoc.nodes["bg1"].params["gamma"] = 0.45
    bdoc.touch("bg1")
    app.processEvents()
    assert 9001 not in win.runner._runs, "an edit in a run's own cone must cancel it"
    assert _cancels == [_rq("bg1")], _cancels
    # ...and a structural edit (unknown scope) still cancels everything, as before
    win.runner._runs[9002] = _rq("bg0")
    win.runner._run_cones[9002] = frozenset([_rq("bg0")])
    bdoc.touch()                                          # no node named → assume everything
    app.processEvents()
    assert not win.runner._runs, "an unscoped edit must still cancel every in-flight run"

    # (6) the G8 source re-seed must cancel NOTHING and must not empty the queue.
    #
    # Found on the real 16-position ND2, not here: `set_meta_seed` fires from inside the
    # delivery of a finished pull, loops back through the window's change handler, and used
    # to arrive as "something changed, scope unknown" — which drained the queue, so asking
    # for two branches ran the first, silently dropped the second, and left its card on
    # `queued` for good. It reports an EMPTY touched set, which means "changed nothing a run
    # can see", and that is a different thing from `None`.
    win.runner._runs[9003] = _rq("bg1")
    win.runner._run_cones[9003] = frozenset(win.runner.planned_nodes("bg1"))
    win.runner._queue[_rq("bg0")] = (_rq("bg0"), None, None)
    _cancels.clear()
    bdoc.set_meta_seed("bl", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                          metadata=dict(_RN._SYNTH_META)))
    app.processEvents()
    assert 9003 in win.runner._runs, \
        "the source re-seed cancelled a live run — it resolves metadata that run already had"
    assert win.runner.queued_nodes() == (_rq("bg0"),), \
        f"the source re-seed emptied the queue — {win.runner.queued_nodes()}"
    assert _cancels == [], _cancels
    win.runner._queue.clear(); win.runner._runs.clear(); win.runner._run_cones.clear()

    # (7) a FINISHED branch keeps its badge when the next branch starts.
    #
    # Reported as "queuing is now causing previous nodes to stop displaying their progress".
    # `set_run_plan` cleared every card no live plan claimed — right when the canvas described
    # "the last run" and one pull existed at a time, wrong once each branch has its own
    # answer: finishing branch A and then starting branch B blanked A's `done` card, erasing
    # the result the user had just waited minutes for.
    win.scene.clear_run_states()
    win.scene.set_run_plan("bg0", ["bl", "bs", "bg0"])
    win.scene.finish_run("bg0", seconds=1.0)
    assert win.scene._run.get("bg0", ("",))[0] == "done", win.scene._run.get("bg0")
    win.scene.set_run_plan("bg1", ["bl", "bs", "bg1"])       # the next branch starts
    assert win.scene._run.get("bg0", ("",))[0] == "done", (
        "starting the next branch erased the finished branch's `done` badge")
    assert win.scene._run.get("bg1", ("",))[0] == "queued", win.scene._run.get("bg1")
    # ...but a badge the EDIT invalidated is still retired, on the node and downstream
    assert bdoc.downstream_of(["bs"]) >= {"bs", "bg0", "bg1"}, bdoc.downstream_of(["bs"])
    win.scene.clear_run_states_for(bdoc.downstream_of(["bg0"]))
    assert win.scene._run.get("bg0", ("",))[0] == "", "a stale badge survived its edit"
    win.scene.clear_run_states()

    # (8) COSMETIC edits must not cancel a run, and a preview must not queue.
    #
    # The other half of the same report: with the cooperative cancel in place, an unscoped
    # notify is no longer "drop the result", it ABORTS the pull. Collapsing a card or
    # renaming a canvas frame did that — neither reaches a run graph at all. And
    # click-to-preview (which fires on every settled selection) went from latest-wins to
    # ENQUEUEING, so clicking around during a long run silently committed the machine to
    # every card touched.
    win.runner._runs[9101] = _rq("bg1")
    win.runner._run_cones[9101] = frozenset(_rq(n) for n in ("bl", "bs", "bg1"))
    _cancels.clear()
    _fr = bdoc.add_frame("probe-frame", members=["bg0"])
    bdoc.set_collapsed("bg0", True)
    bdoc.rename_frame(_fr.id, "probe-frame-2")
    app.processEvents()
    assert 9101 in win.runner._runs, \
        "a cosmetic edit (collapse / frame rename) aborted a running pull"
    assert _cancels == [], _cancels
    bdoc.set_collapsed("bg0", False)
    bdoc.remove_frame(_fr.id)
    # a preview is dropped while busy; an explicit pull still queues
    win.runner._busy = True
    win.runner._queue.clear()
    win.runner.pull("bg0", None, None, queue=False)
    assert win.runner.queued_nodes() == (), \
        f"click-to-preview queued a pull: {win.runner.queued_nodes()}"
    win.runner.pull("bg0", None, None)
    assert win.runner.queued_nodes() == (_rq("bg0"),), win.runner.queued_nodes()
    win.runner._busy = False
    win.runner._queue.clear(); win.runner._runs.clear(); win.runner._run_cones.clear()

    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)   # restore the fallback
    _ok("C4 independent branches (2026-08-06): a second branch requested mid-run is QUEUED "
        "(repeat requests keep their place, never duplicate) instead of silently replacing "
        "the pending one; the queued cards read `queued` while the running one reads "
        "`running`; both branches produce their own distinct pixels; a FINISHED branch is "
        "re-served from cache without taking the pull slot, so it can be viewed while the "
        "other still computes; an edit cancels only the runs whose cone contains it — "
        "re-tuning the finished branch leaves the running one alive, where one global epoch "
        "used to kill it with no error and no card state to show for it; and the G8 source "
        "re-seed, which fires from inside a delivery, cancels nothing and leaves the queue "
        "intact (it used to drain it, so the second branch never ran at all); a FINISHED "
        "branch keeps its `done` badge when the next branch starts, while the edit that "
        "actually invalidates it retires it on the node and everything downstream; a "
        "cosmetic edit (collapse, frame rename) no longer ABORTS a running pull now that "
        "cancellation is cooperative; and click-to-preview is dropped rather than queued, so "
        "clicking around during a long run stops committing the machine to every card")

    # ── C5: deleting a node cancels ITS runs — and lets the others run instead ──
    #
    # The 2026-08-06 ask verbatim: "deleting a node should cancel its run state to let
    # other nodes be run instead". Pinned in five parts: (1) run-graph ids map back to the
    # document card that owns them, so a delete matches runs computing its Iterate clones
    # or inlined group bodies; (2) a delete NARROWS — the other branch's run and queued
    # request both survive it; (3) the deleted node's own queued request is dropped on the
    # spot; (4) when the deleted node's run is the one ON the worker, the job's cancel
    # flag latches (a bake is never latched); (5) end to end on the real worker — the
    # running pull of a deleted node aborts mid-compute and the branch queued behind it
    # runs and delivers. The engine half of the abort (should_stop → PullCancelled, no
    # torn memo entry) is gated Qt-free in nodegraph.selftest's test_engine_cancel.

    # (1) run-graph id → document id
    assert _RN._doc_id_of("n7") == "n7"
    assert _RN._doc_id_of("bg0#it2@3") == "bg0"        # an Iterate clone
    assert _RN._doc_id_of("it2#adv@0") == "it2"        # the zone's advance node
    assert _RN._doc_id_of("body%inst") == "inst"       # an inlined group body node
    assert _RN._doc_id_of("b%mid%g7#it1@0") == "g7"    # nested groups inside a zone
    # …and a PAGE-QUALIFIED run id (V4.00 step 2) keeps its page: the cone an edit on a page
    # is matched against is written in these terms
    assert _RN._doc_id_of("pg1/n7") == "pg1/n7"
    assert _RN._doc_id_of("pg1/bg0#it2@3") == "pg1/bg0" and _RN._doc_id_of("pg2/it2#adv@0") == "pg2/it2"
    assert _RN._doc_id_of("pg1/body%inst") == "pg1/inst" and _RN._doc_id_of("pg3/b%mid%g7#it1@0") == "pg3/g7"

    # (2) deleting one branch leaves the other branch's run AND queued request alone
    win.runner._runs[9101] = _rq("bg1")
    win.runner._run_cones[9101] = frozenset(_rq(n) for n in ("bl", "bs", "bg1"))
    win.runner._queue[_rq("bg1")] = (_rq("bg1"), None, None)
    _cancels.clear()
    bdoc.remove_node("bg0")
    app.processEvents()
    assert 9101 in win.runner._runs, \
        "deleting one branch cancelled the OTHER branch's run — a delete must narrow"
    assert win.runner.queued_nodes() == (_rq("bg1"),), win.runner.queued_nodes()
    assert _cancels == [], _cancels

    # (3)+(4) deleting the node whose run is ON the worker: run dead, queued request
    # dropped, and the job's cancel flag latched for the engine to poll
    from types import SimpleNamespace as _SNS
    _act = _SNS(epoch=9101, bake=None, cancelled=False)
    win.runner._active = _act
    bdoc.remove_node("bg1")
    app.processEvents()
    assert 9101 not in win.runner._runs
    assert win.runner.queued_nodes() == (), win.runner.queued_nodes()
    assert _act.cancelled is True, \
        "the deleted node's RUNNING job must latch its cancel flag — otherwise the " \
        "engine grinds the dead pull to completion with the queue waiting behind it"
    assert set(_cancels) == {_rq("bg1")}, _cancels     # its run + its queued request
    # ...while a bake job is never latched: its checkpoint resolves outside staleness
    win.runner._runs[9102] = _rq("bs")
    win.runner._run_cones[9102] = frozenset([_rq("bl"), _rq("bs")])
    _bact = _SNS(epoch=9102, bake={"hold": False}, cancelled=False)
    win.runner._active = _bact
    bdoc.remove_node("bs")
    app.processEvents()
    assert 9102 not in win.runner._runs and _bact.cancelled is False, \
        "a bake must never be cancel-latched — its result is a directory that exists"
    win.runner._active = None

    # (5) end to end: the slow branch's gamma is wrapped to tick progress and sleep per
    # unit — keyed to the one node id, so the fast branch runs the stock compute. Delete
    # the slow node mid-compute; its pull must abort (the wrap records whether the loop
    # ever completed) and the branch queued behind it must run and deliver.
    win.file_new()
    app.processEvents()
    win.runner._providers.clear()
    win.runner._announced.clear()
    win.runner._raw_src.clear()
    win.runner.invalidate()
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=1, c=2, y=64, x=64)
    cdoc = win.doc
    cdoc.add_node("io.load", node_id="cl", x=0, y=0)
    cdoc.set_meta_seed("cl", MetaEnvelope(axes=_RN._SYNTH_AXES,
                                          metadata=dict(_RN._SYNTH_META)))
    cdoc.add_node("channel.split", node_id="cs", x=180, y=0)
    cdoc.connect("cl", "image", "cs", "data")
    for _k in (0, 1):
        cdoc.add_node("enhance.gamma", node_id=f"cg{_k}", x=380, y=_k * 200,
                      params={"gamma": 0.8})
        cdoc.connect("cs", f"ch{_k}", f"cg{_k}", "data")
    app.processEvents()

    from nodegraph.nodes import COMPUTES as _COMP
    _real_gamma = _COMP["enhance.gamma"]
    _slow_done = {"finished": False}

    def _slow_gamma(ctx):
        if ctx.node_id != _rq("cg0"):              # the engine runs page-qualified ids
            return _real_gamma(ctx)
        for _i in range(600):                          # ≥6 s unless the abort fires
            ctx.progress(_i + 1, 600, "slow")
            time.sleep(0.01)
        _slow_done["finished"] = True
        return _real_gamma(ctx)

    _COMP["enhance.gamma"] = _slow_gamma
    _events: list = []
    _prog = win.runner.node_progress.connect(
        lambda ev, nid, info: _events.append((ev, nid)))
    try:
        _pulls.clear()
        win.pull_node("cg0")                           # the slow branch takes the slot
        _t0 = time.time()
        while ("start", _rq("cg0")) not in _events and time.time() - _t0 < 30.0:
            app.processEvents()
            time.sleep(0.005)
        assert ("start", _rq("cg0")) in _events, "the slow pull never started"
        win.runner.pull("cg1")                         # queue the branch we want instead
        assert win.runner.queued_nodes() == (_rq("cg1"),), win.runner.queued_nodes()
        win.scene.delete_nodes(["cg0"])                # the canvas delete path
        _await_pull(limit=60.0)                        # cg1's result, behind the abort
        assert _pulls and _pulls[-1][0] == _rq("cg1"), _pulls
        assert not _slow_done["finished"], \
            "the deleted node's compute ran to completion — the abort never fired"
        assert "cg0" not in cdoc.nodes
    finally:
        _COMP["enhance.gamma"] = _real_gamma
        win.runner.node_progress.disconnect(_prog)
    _RN._SYNTH_AXES = AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512)   # restore the fallback

    _ok("C5 delete cancels the node's own runs (2026-08-06): a delete narrows the "
        "invalidation to the deleted node's cone, so the other branch's run and queued "
        "request survive it; the deleted node's queued request is dropped where it "
        "stands; the running job's cancel flag latches (never for a bake) and the engine "
        "aborts the pull mid-compute — measured end to end: the deleted node's slow pull "
        "unwound without finishing and the branch queued behind it ran and delivered; "
        "run-graph clone ids (`n#it@i`, `body%inst`) map back to their document card so "
        "a delete also cancels runs computing a card's Iterate clones or group body")

    # ── V5: a stale channel request against a per-channel tap (2026-08-15) ────
    # The user's session: view a multi-channel node with only a HIGH channel toggled
    # on, then click a node fed from a per-channel `chK` output (its tap output has
    # c=1). The pull is made with the viewer's PREVIOUS active set (window.pull_node
    # sends viewer.channels()), every index in it is at/above the payload's channel
    # count, and `_plane_addrs` dropped them all — zero planes, "no image on this
    # output", a black viewer — while the very same click a SECOND time displayed
    # fine, because the strip is corrected on delivery, one click too late. The
    # request now degrades to the clamped cursor channel instead of to nothing.
    # (Found live: an ND2's per-channel output into ZS-DeconvNet; node-independent.)
    _pulls.clear()
    win.pull_node("cl")                        # view the 2-channel source
    _await_pull(limit=60.0)
    if 1 not in win.viewer._active_channels:   # leave ONLY channel index 1 active,
        win.viewer._on_channel_toggle(1)       # through the real toggle path
    if 0 in win.viewer._active_channels:
        win.viewer._on_channel_toggle(0)
    _t0 = time.time()
    while time.time() - _t0 < 1.0:             # let the toggles' re-requests land
        app.processEvents()
        time.sleep(0.005)
    assert win.viewer._active_channels == [1], win.viewer._active_channels
    cdoc.add_node("enhance.gamma", node_id="cg2", x=380, y=400, params={"gamma": 0.9})
    cdoc.connect("cs", "ch1", "cg2", "data")
    win.scene.sync()
    app.processEvents()
    _pulls.clear()
    win.pull_node("cg2")                       # FIRST view of the tap-fed node
    _await_pull(limit=60.0)
    _st = win.viewer._status.text()
    assert "no image on this output" not in _st, _st
    assert win.viewer._planes, "first view of a tap-fed node must deliver planes"
    assert "px" in _st, _st
    _ok("V5 stale-channel request vs a per-channel tap (2026-08-15): viewing only a "
        "high channel index and then clicking a node fed from a `chK` output (tap "
        "payload c=1) shows the image on the FIRST click — a request whose every "
        "channel is stale degrades to the cursor channel instead of delivering zero "
        "planes and reading 'no image on this output'")

    # ── V6: channels never switch themselves off (2026-09-30) ──────────────────────
    # Reported: "loading files or going from node to node deactivates one or more
    # channels". The strip used to switch a channel on only the FIRST time its name was
    # seen, so 2-channel -> 1-channel tap -> 2-channel came back with a channel OFF (seen,
    # never re-enabled), and the pull that delivered the rebuild carried only the old
    # strip's channels. Now: every channel is on unless the user switched it off, and the
    # missing planes are fetched. V5 left channel 0 user-off; switch it back on first.
    win.pull_node("cl")
    _await_pull(limit=60.0)
    if 0 not in win.viewer._active_channels:
        win.viewer._on_channel_toggle(0)
    _await_planes(2)
    win.pull_node("cg2")                       # the 1-channel tap
    _await_pull(limit=60.0)
    assert win.viewer._active_channels == [0], win.viewer._active_channels
    win.pull_node("cl")                        # ...and back
    _await_pull(limit=60.0)
    assert win.viewer._active_channels == [0, 1], win.viewer._active_channels
    _await_planes(2)
    assert all(win.viewer._chan_btns[_c].isChecked() for _c in (0, 1))
    # a channel the USER switched off stays off through the same round trip
    win.viewer._on_channel_toggle(1)
    _await_planes(1)
    win.pull_node("cg2")
    _await_pull(limit=60.0)
    win.pull_node("cl")
    _await_pull(limit=60.0)
    assert win.viewer._active_channels == [0], win.viewer._active_channels
    win.viewer._on_channel_toggle(1)           # leave the probe as it found it
    _await_planes(2)
    # a node whose DATA changed under the same id re-auto-contrasts: a stale window
    # from another file is a channel that looks switched off
    _cl = _rq("cl")                            # the viewer keys a node by its run id
    win.viewer._clim[(_cl, 0)] = (1e9, 2e9)
    win.viewer._lut_ident[_cl] = ((2, ("other",), ()), set())
    win.pull_node("cl")
    _await_pull(limit=60.0)
    assert win.viewer._clim.get((_cl, 0), (0.0, 0.0))[0] < 1e9, win.viewer._clim.get((_cl, 0))
    _ok("V6 channels never switch themselves off (2026-09-30): 2-channel -> 1-channel "
        "tap -> 2-channel returns with BOTH channels on and both planes delivered "
        "without a click; a channel the user switched off stays off through the same "
        "round trip; a node whose channel identity changed under the same id drops its "
        "cached LUT and auto-contrasts afresh")

    # ── B1 the golden point (V3.01) ──────────────────────────────────────────────
    # The batch axis' canvas half: Batch and Unbatch render as GOLD CIRCLES rather than
    # cards, the Unbatch grows one output per FILE, and each wired one materializes into a
    # real `util.select_batch` tap. Probed here rather than in the headless selftest
    # because every one of those is a property of the GUI's own layer — the document's
    # socket synthesis, the node item's geometry, and the run-graph rewrite between them.
    from nodelab_v2 import theme as _TB
    from nodelab_v2.ops import materialize_batch_taps as _mbt
    from nodelab_v2.scene import GraphScene as _GSb
    bdoc = GraphDocument()
    bdoc.add_node("util.batch", node_id="bba", x=300, y=0)
    bdoc.add_node("enhance.gaussian", node_id="bg", x=460, y=0)
    bdoc.add_node("util.unbatch", node_id="bun", x=700, y=0)
    for _i in range(3):
        bdoc.add_node("io.load", node_id=f"bf{_i}", x=0, y=_i * 90,
                      params={"path": rf"C:\d\F{_i}.nd2"})
        bdoc.connect(f"bf{_i}", "image", "bba", "data")
    bdoc.connect("bba", "out", "bg", "data")
    bdoc.connect("bg", "out", "bun", "data")
    app.processEvents()

    # (a) the Unbatch grows one output per file, NAMED after it — the only thing on the
    #     canvas that says which wire is which specimen
    _bnames = bdoc.batch_member_names("bun")
    assert _bnames == ["F0.nd2", "F1.nd2", "F2.nd2"], _bnames
    _bsocks = [s.name for s in bdoc.output_specs("bun")]
    assert _bsocks == ["out", "bat0", "bat1", "bat2"], _bsocks
    # ...and the Batch point does NOT: it collects into one multi socket
    assert [s.name for s in bdoc.output_specs("bba")] == ["out"]

    # (b) both render as circles, not cards, and the Unbatch's member sockets are SPREAD
    #     far enough apart to tell one file's wire from another's
    bscene = _GSb(bdoc)
    _bitems = {i.node_id: i for i in bscene.items() if hasattr(i, "node_id")}
    for _nid in ("bba", "bun"):
        _it = _bitems[_nid]
        assert getattr(_it, "_is_dot", False), _nid
        assert _it.card_rect().width() == _it.card_rect().height(), _nid
    assert not getattr(_bitems["bg"], "_is_dot", False), "a filter is still a card"
    _bys = sorted(s.pos().y() for (io, nm), s in _bitems["bun"]._sockets.items()
                  if io == "out" and nm.startswith("bat"))
    _bgaps = [_bys[i + 1] - _bys[i] for i in range(len(_bys) - 1)]
    assert _bgaps and min(_bgaps) >= _TB.SOCKET_PITCH - 0.01, _bgaps

    # (c) a wired member socket becomes a REAL tap carrying the file's NAME, never its
    #     index — the rewire-safety rule the group taps already follow
    bdoc.add_node("view.viewer", node_id="bv", x=900, y=0)
    bdoc.connect("bun", "bat1", "bv", "data")
    _brun = _mbt(bdoc.to_graph())
    _btaps = [n for n in _brun.nodes.values() if n.op_key == "util.select_batch"]
    assert len(_btaps) == 1, _btaps
    assert _btaps[0].params.get("member") == "F1.nd2", _btaps[0].params
    assert not any(e.src_socket.startswith("bat") for e in _brun.edges), "a batK survived"
    _ok("B1 the golden point (V3.01): Batch and Unbatch render as CIRCLES rather than "
        "cards (a batch is not a processing step), the Unbatch grows one output per FILE "
        "named after it while the Batch collects into one multi socket, the member "
        "sockets stay at least SOCKET_PITCH apart so a wire is traceable to its file, and "
        "a wired one materializes into a real util.select_batch tap carrying the file's "
        "NAME — an index would slide onto a different specimen after a rewire with every "
        "hash still agreeing")

    # ── B2 per-file tabs in the measurement table (V3.01) ────────────────────────
    # When several files share one pipeline every row belongs to one of them and the
    # `file` column says which — but reading "the measurements for WellA3" out of a
    # 6000-row table means finding where one name stops. These are that column as tabs.
    from nodegraph.structure import StructureTable as _SB
    from nodegraph.domains import Domain as _DomB
    from nodegraph.dataset import Dataset as _DsB
    from nodegraph.metadata import SOURCE_FILE_KEY as _SFK2
    from nodelab_v2.spreadsheet import SpreadsheetPanel as _SP2
    _tax = AxisSizes(m=3, t=1, z=1, c=1, y=16, x=16)
    _ttbl = _SB(_DomB.LABEL, {
        "id": np.arange(6), "m": np.array([0, 0, 1, 1, 2, 2]),
        "t": np.zeros(6, int), "c": np.zeros(6, int), "z": np.zeros(6, int),
        "y": np.linspace(1, 9, 6), "x": np.linspace(2, 8, 6),
        "area": np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
    }, layer="CELLS", z_kind="plane_index")
    _tds = (_DsB(axes=_tax, metadata={"pixel_size_um": 0.5})
            .with_metadata(**{_SFK2: ["A.nd2", "B.nd2", "C.nd2"]})
            .with_structure(_ttbl))
    _panel = _SP2()
    _panel.show_dataset("measure", _tds)
    _ttabs = [_panel._files.tabText(i) for i in range(_panel._files.count())]
    assert _ttabs == ["All (3 files)", "A.nd2", "B.nd2", "C.nd2"], _ttabs

    def _areas():
        _c = [n for n in range(_panel._table.columnCount())
              if _panel._table.horizontalHeaderItem(n).text() == "area"][0]
        return [_panel._table.item(r, _c).text()
                for r in range(_panel._table.rowCount())]

    assert len(_areas()) == 6, _areas()                       # All
    for _i, _want in ((1, ["10", "11"]), (2, ["20", "21"]), (3, ["30", "31"])):
        _panel._files.setCurrentIndex(_i)
        assert _areas() == _want, (_i, _areas())
        # the row header keeps the ORIGINAL row number, so a filtered row is still
        # findable in the unfiltered CSV the export writes
        assert _panel._table.verticalHeaderItem(0).text() == str((_i - 1) * 2)
    _panel._files.setCurrentIndex(0)
    assert len(_areas()) == 6, _areas()
    # a SINGLE-file table grows no tabs — one tab beside `All` would offer the same rows
    # under two names, the floor the channel and member sockets already use
    _one = (_DsB(axes=AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16),
                    metadata={"pixel_size_um": 0.5})
            .with_metadata(**{_SFK2: ["only.nd2"]})
            .with_structure(_SB(_DomB.LABEL, {
                "id": np.arange(2), "m": np.zeros(2, int), "t": np.zeros(2, int),
                "c": np.zeros(2, int), "z": np.zeros(2, int),
                "y": np.zeros(2), "x": np.zeros(2), "area": np.array([1.0, 2.0]),
            }, layer="CELLS", z_kind="plane_index")))
    _panel.show_dataset("measure1", _one)
    assert _panel._files.count() == 0, [_panel._files.tabText(i)
                                        for i in range(_panel._files.count())]
    _ok("B2 per-file tabs (V3.01): a measurement table whose rows come from several "
        "files grows one tab per file plus `All`, in WIRING order (not sorted — tab 2 "
        "must mean the same file as output 2), each filtering to that file's rows while "
        "the row header keeps the original row number so a row stays findable in the "
        "unfiltered CSV; a single-file table grows none")

    # -- O11 the experiment overlay (2026-09-30): Play all, the source strip, pins, LUTs --
    #
    # The engine and runner halves are proved headlessly (test_overlay_experiment,
    # test_overlay_subtick_cache). What only a real widget can prove: the source strip
    # shows each overlaid file by label with what it is showing; its steppers and Pin
    # buttons emit exactly what the window writes; Play all holds each primary frame for
    # n ticks (the slider moves once per n); the window turns a pin into canonical JSON on
    # the Overlay through the ordinary edit path; a LUT drag keeps an overlay channel's
    # blend; and an overlay channel's LUT is keyed by its SOURCE, not its chain position.
    vw = win.viewer
    if vw._playing_axis is not None:
        vw._stop_play()
    # The widget half runs DETACHED from the window: every emit would otherwise reach the
    # real runner, re-request the viewed node (which overlays nothing) and have the window
    # hand the strip an empty readout mid-assertion. The window half (pin -> params) is
    # driven directly below.
    _o11_sigs = ("request_changed", "overlay_step", "overlay_pin")
    for _sig in _o11_sigs:            # the window's own slots on THIS viewer (V4.00 step 4)
        getattr(vw, _sig).disconnect(win._viewer_slot(vw, _sig))
    vw._sliders["t"].blockSignals(True)
    vw._sliders["t"].setRange(0, 2)
    vw._sliders["t"].setValue(0)
    vw._sliders["t"].blockSignals(False)
    vw._sliders["z"].blockSignals(True)
    vw._sliders["z"].setRange(0, 4)
    vw._sliders["z"].setValue(2)
    vw._sliders["z"].blockSignals(False)
    rows = [{"ovl_id": "OVA", "sec_id": "SRC", "label": "GFP 20x", "t": 4, "n_t": 12,
             "z": 1, "n_z": 3, "t_pinned": False, "z_pinned": False, "offset": (0, 0),
             "sub_ticks": 3}]
    vw.set_overlay_frames(rows, 3)
    app.processEvents()
    assert "OVA" in vw._src_rows and vw._n_sub == 3
    row = vw._src_rows["OVA"]
    assert row["name"].text() == "GFP 20x", row["name"].text()
    assert "t 4/11" in row["info"].text() and "z 1/2" in row["info"].text() \
        and "3x rate" in row["info"].text(), row["info"].text()
    assert "Play all" in vw._play_btns["t"].toolTip()
    got_step, got_pin, reqs = [], [], []
    vw.overlay_step.connect(lambda *a: got_step.append(a))
    vw.overlay_pin.connect(lambda *a: got_pin.append(a))
    vw.request_changed.connect(lambda: reqs.append(vw.sub()))
    vw._step_source("OVA", 0, +1)
    vw._step_source("OVA", 1, -1)
    assert got_step == [("OVA", 1, 0), ("OVA", 0, -1)], got_step
    vw._pin_source("OVA", "t")
    vw._pin_source("OVA", "z")
    assert got_pin == [("OVA", "t", 0, 4), ("OVA", "z", 2, 1)], got_pin

    # Play all: three ticks per primary frame -- the slider moves on the third
    vw._playing_axis = "t"             # what `_on_play` sets, minus the timer and preload
    vw._sync_pin_enabled()
    vw._play_paced = False
    vw._sub_step = 1
    seen = []
    for _ in range(6):
        vw._tick_play()
        seen.append((vw._sliders["t"].value(), vw.sub()))
    assert seen == [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)], seen
    assert all(r["t_pin"].isEnabled() is False for r in vw._src_rows.values()), \
        "no pinning mid-playback: a pin is an edit, and an edit cancels the preload"
    vw._t_sub = 1
    vw._stop_play()
    assert vw.sub() == 0 and vw._src_rows["OVA"]["t_pin"].isEnabled()
    # the tick rate: n_sub x the fps, capped, with several sub-ticks per tick past the cap
    vw._fps_spins["t"].setValue(8.0)
    assert vw._interval_ms("t") == int(1000.0 / 24.0) and vw._sub_step == 1
    vw.set_overlay_frames(rows, 16)
    vw._fps_spins["t"].setValue(30.0)
    vw._interval_ms("t")
    assert vw._sub_step == 8, vw._sub_step            # 480 ticks/s -> 8 per 60 Hz tick
    vw.set_overlay_frames(rows, 3)
    for _sig in _o11_sigs:
        getattr(vw, _sig).connect(win._viewer_slot(vw, _sig))

    # the window writes a pin as canonical JSON, pinned, through the edit path
    ovl = doc.add_node("view.overlay", x=1400, y=900)
    win._viewed = win._viewed or "n3"
    # the overlay strip hands back the RUN id it was given (V4.00 step 2), as here
    _ovl_rid = win.runner.run_id(ovl.id)
    assert _ovl_rid != ovl.id
    win._on_overlay_pin(_ovl_rid, "t", 5, 20)
    win._on_overlay_pin(_ovl_rid, "t", 1, 4)
    win._on_overlay_pin(_ovl_rid, "t", 5, 21)          # re-pinning a frame replaces it
    assert doc.nodes[ovl.id].params["t_pins"] == "[[1,4],[5,21]]", \
        doc.nodes[ovl.id].params["t_pins"]
    assert "t_pins" in doc.nodes[ovl.id].locked
    # ...and the inspector lists them, one remove button each
    win.scene.clearSelection()
    win.scene.node_items[ovl.id].setSelected(True)
    app.processEvents()
    from PySide6.QtWidgets import QToolButton as _QTB
    xs = [b for b in win.inspector.findChildren(_QTB) if b.text() == "✕"]
    assert len(xs) >= 2, "each T pin gets its own remove button"
    xs[0].click()
    app.processEvents()
    app.processEvents()
    assert doc.nodes[ovl.id].params["t_pins"] == "[[5,21]]", doc.nodes[ovl.id].params
    doc.remove_node(ovl.id)

    # an overlay channel's LUT is keyed by its SOURCE, and a LUT drag keeps its blend
    vw._overlay_src = {1: {"node": "SRC", "ch": 0}}
    assert vw._lut_key("n3", 1) == ("ovl", "SRC", 0) and vw._lut_key("n3", 0) == ("n3", 0)
    calls = []

    class _GLStub:
        def set_channel(self, *a, **k):
            calls.append(k)

        def refresh(self):
            pass

    real_gl, real_style, real_planes = vw._gl, vw._overlay_style, vw._planes
    try:
        vw._gl = _GLStub()
        vw._overlay_style = {1: (1, 0.35, 8.0)}             # `over` at 0.35
        vw._planes = {0: np.zeros((4, 4)), 1: np.zeros((4, 4))}
        vw._apply_lut(1)
    finally:
        vw._gl, vw._overlay_style, vw._planes = real_gl, real_style, real_planes
    assert calls and calls[-1].get("blend") == 1 and abs(calls[-1]["opacity"] - 0.35) < 1e-9, \
        calls
    vw.set_overlay_frames([], 1)
    assert not vw._src_box.isVisible() and vw._n_sub == 1
    _ok("O11 experiment overlay (2026-09-30): the source strip shows each overlaid file by "
        "its label with the frame/plane it is showing and its rate; the ◀▶ steppers emit an "
        "absolute display-only offset and Pin T / Pin Z emit (primary, source-on-screen); "
        "Play all holds each primary frame for n ticks (slider 0,0,1,1,1,2 over six ticks) "
        "at n x the fps, several sub-ticks per tick past 60 Hz, pins are disabled while "
        "playing and stopping parks on sub-tick 0; the window writes pins as canonical "
        "sorted later-wins JSON through the pinning edit path and the inspector removes "
        "one with its ✕; an overlay LUT is keyed by source, and a LUT drag no longer resets "
        "an overlay channel to additive at full opacity")

    # ── V1: the Viewer NODE — growable sources, layout modes, scale bar (2026-10-02) ──
    # A display sink: the primary is the payload, every extra wired `source_N` is composited
    # as display channels named after its socket, the card offers exactly one empty slot,
    # `layout` lays the streams out as one picture / per-stream panes / both, and the scale
    # bar is a presentation setting the window pushes to the panel from the document.
    from PySide6.QtCore import QRectF as _QRectF
    win.set_solo_frame(False)
    win.build_demo()
    app.processEvents()
    vdoc = win.doc
    vdoc.add_node("view.viewer", node_id="vv", x=1330, y=430)
    assert [s.name for s in vdoc.input_specs("vv") if s.type.name == "DATASET"] == \
        ["data", "source_2"], "at rest: the primary and ONE empty slot"
    assert not list(vdoc.output_specs("vv")), "a sink has no output socket"
    vdoc.connect("n3", "out", "vv", "data")                 # the blurred image
    vdoc.connect("n2", "out", "vv", "source_2")             # the raw channel beside it
    assert [s.name for s in vdoc.input_specs("vv") if s.type.name == "DATASET"] == \
        ["data", "source_2", "source_3"], "wiring a slot reveals the next"
    _pulls.clear()
    win.pull_node("vv")
    vpay = _await_pull()
    assert vpay.axes.c == 2, f"the payload is the PRIMARY (its own channels), got {vpay.axes}"
    labels = win.runner.overlay_channels("vv")
    assert labels and all(str(l).startswith("source_2:") for l in labels.values()), labels
    assert win.viewer._overlay_chans == labels
    assert win.viewer.source_layout == "merged" and win.viewer._tiles() == [], \
        "merged: one composite, no panes"
    vrec = vdoc.nodes["vv"]
    vrec.modes["layout"] = "tiles"
    win.pull_node("vv")
    _await_pull()
    tiles = win.viewer._tiles()
    assert win.viewer.source_layout == "tiles" and len(tiles) == 2, tiles
    assert tiles[0][0] == "vv" and tiles[1][0] == "source_2", [t[0] for t in tiles]
    assert set(tiles[0][1]) | set(tiles[1][1]) == set(win.viewer._planes), \
        "the panes partition every shown channel"
    vrec.modes["layout"] = "both"
    win.pull_node("vv")
    _await_pull()
    tiles = win.viewer._tiles()
    assert len(tiles) == 3 and tiles[0][0] == "Merged" and \
        set(tiles[0][1]) == set(win.viewer._planes), [t[0] for t in tiles]
    # the scale bar: presentation params, read from the document, drawn inside the image
    assert win.viewer.scalebar is None, "off by default"
    vrec.params["show_scalebar"] = True
    vrec.params["scalebar_um"] = 10.0
    vrec.params["scalebar_corner"] = "top_left"
    win.pull_node("vv")
    _await_pull()
    assert win.viewer.scalebar == {"um": 10.0, "corner": "top_left", "color": "white"}, \
        win.viewer.scalebar
    _surf = win.viewer._pick_targets()[0]
    geo = win.viewer._scalebar_geometry(_QRectF(0, 0, _surf.width(), _surf.height()))
    assert geo is not None, "a calibrated payload must yield a bar"
    bar, label, above = geo
    assert label == "10 µm" and not above, (label, above)
    _mp = win.viewer._view.plane_to_widget
    _H, _W = win.viewer._ref_plane.shape[:2]
    _img = _QRectF(_mp(0.0, 0.0), _mp(float(_W), float(_H))).normalized()
    assert _img.contains(bar), f"the bar must sit inside the image: {bar} vs {_img}"
    # 10 um at 0.1 um/px = 100 source px → the bar spans 100 axes px of the image width,
    # or is capped at 90% of the visible image when the frame is narrower than that (the
    # demo source here is 64 px wide, so the cap is what this exercises)
    _want = min(100.0 / win.viewer._axes.x, 0.9)
    assert abs(bar.width() / _img.width() - _want) < 0.02, \
        (bar.width(), _img.width(), win.viewer._axes.x, _want)
    vrec.params["show_scalebar"] = False
    # the TIMESTAMP (2026-10-02): presentation like the bar; `frame` always reads, `clock`
    # is honest about a payload with no absolute clock, `elapsed` falls back to the frame
    # number on a synthetic source with neither clock nor interval
    vrec.params["show_timestamp"] = True
    vrec.params["timestamp_mode"] = "frame"
    win.pull_node("vv")
    _await_pull()
    assert win.viewer.scalebar is None
    assert win.viewer.timestamp == {"mode": "frame", "corner": "top_left", "color": "white"}
    _tt = win.viewer.timestamp_text()
    assert _tt.startswith("t ") and f"/{win.viewer._axes.t}" in _tt, _tt
    vrec.params["timestamp_mode"] = "clock"
    win.pull_node("vv"); _await_pull()
    _md = getattr(win.viewer._dataset, "metadata", {}) or {}
    if not (_md.get("frame_time_jd") or _md.get("frame_datetime")):
        assert win.viewer.timestamp_text() == "", "no clock on the payload → no fabricated date"
    vrec.params["timestamp_mode"] = "elapsed"
    win.pull_node("vv"); _await_pull()
    assert win.viewer.timestamp_text(), "elapsed always says something"
    from PySide6.QtGui import QFontMetrics as _QFM
    _geo = win.viewer._timestamp_geometry(
        _QRectF(0, 0, win.viewer._view.width(), win.viewer._view.height()), "t 1/4",
        _QFM(win.viewer.font()))
    assert _geo is not None and _geo[0].x() >= 0 and _geo[0].y() > 0, _geo
    vrec.params["show_timestamp"] = False
    win.pull_node("vv"); _await_pull()
    assert win.viewer.timestamp is None
    vdoc.remove_node("vv")
    _ok("VN1 viewer node: primary + one empty source slot that grows as wired, no output; "
        "a second stream composites as `source_2:` display channels on the primary payload; "
        "layout merged/tiles/both gives 0/2/3 panes that partition the channels; the scale "
        "bar is presentation (off by default), 10 um reads '10 µm' top-left inside the "
        "image at the calibrated length, and clears when switched off")

    # ── RD1: the inspector's READY TO RUN block (2026-10-02) ───────────────────
    # A node that cannot run as wired says so under its title, paints the input in question
    # red, and offers the nodes that would fix it; pressing one adds the node AND wires it —
    # on the primary wire for a missing domain, into the side input for a background sample.
    from PySide6.QtWidgets import QLabel as _QL, QToolButton as _QTB
    # on a FREE page: the demo page is an Image Input page since V4.00 step 11, and a page's
    # readiness suggestions offer only the nodes its kind offers (Label is not one of them)
    _rd_home = win.workspace.active
    win.new_page("free")
    app.processEvents()
    _rdoc = win.doc
    _rl = _rdoc.add_node("io.load", node_id="RDL", x=0, y=900)
    _rdoc.meta_seeds["RDL"] = MetaEnvelope(axes=AxisSizes(m=1, t=4, z=1, c=1, y=64, x=64),
                                           metadata={"pixel_size_um": 0.5})
    _rdoc.add_node("enhance.subtract_background", node_id="RDB", x=400, y=900,
                   modes={"approach": "zero_regions"})
    _rdoc.connect("RDL", "image", "RDB", "data")
    win.scene.sync(); app.processEvents()

    def _warns():
        # the problem MESSAGES ("⚠  …", two spaces) — not the red connection line ("⚠ in · data")
        return [l.text() for l in win.inspector.findChildren(_QL) if l.text().startswith("⚠  ")]

    def _red_conns():
        return [l.text() for l in win.inspector.findChildren(_QL) if l.text().startswith("⚠ in")]

    def _adds():
        # the problems' fix buttons — not the "+ Page Output" of the `unpublished` HINT a
        # terminal node on a typed page carries since V4.00 step 11 (RF1 covers that one)
        return [b for b in win.inspector.findChildren(_QTB)
                if b.property("role") == "add" and b.text() != "+ Page Output"]
    win.inspector.set_node(win.scene.node_items["RDB"]); app.processEvents()
    assert any("Background sample" in w for w in _warns()), _warns()
    assert "shapes" in win.inspector._problem_sockets, win.inspector._problem_sockets
    assert [b.text() for b in _adds()] == ["+ Draw Regions"], [b.text() for b in _adds()]
    _adds()[0].click(); app.processEvents()
    _new = [nid for nid, r in _rdoc.nodes.items() if r.op_key == "analysis.draw_regions"]
    assert len(_new) == 1, _new
    assert ("RDL", "image", _new[0], "data") in _rdoc.edges, "fed from the same image"
    assert (_new[0], "out", "RDB", "regions") in _rdoc.edges, "wired into the side input"
    assert ("RDL", "image", "RDB", "data") in _rdoc.edges, "the primary wire is untouched"
    assert _warns() == [] and not _adds(), "the problem is gone once the input is wired"
    assert any(l.text().startswith("✓") for l in win.inspector.findChildren(_QL))
    # a missing domain inserts ON the wire
    _rdoc.add_node("analysis.measure", node_id="RDM", x=800, y=900)
    _rdoc.connect("RDB", "out", "RDM", "data")
    win.scene.sync(); app.processEvents()
    win.inspector.set_node(win.scene.node_items["RDM"]); app.processEvents()
    assert any("label" in w for w in _warns()), _warns()
    assert [b.text() for b in _adds()][0] == "+ Connected Components", [b.text() for b in _adds()]
    _before = set(_rdoc.nodes)
    _adds()[0].click(); app.processEvents()
    _lab = [nid for nid in _rdoc.nodes if nid not in _before]      # the one node it added
    assert len(_lab) == 1 and _rdoc.nodes[_lab[0]].op_key == "analysis.label", _lab
    assert ("RDB", "out", _lab[0], "data") in _rdoc.edges and \
        (_lab[0], "out", "RDM", "data") in _rdoc.edges and \
        ("RDB", "out", "RDM", "data") not in _rdoc.edges, "inserted on the wire"
    assert _warns() == [], _warns()
    # an unwired node: the one problem, and nothing else judged
    _rdoc.add_node("enhance.gaussian", node_id="RDG", x=400, y=1100)
    win.scene.sync(); app.processEvents()
    win.inspector.set_node(win.scene.node_items["RDG"]); app.processEvents()
    assert len(_warns()) == 1 and "nothing is wired" in _warns()[0], _warns()
    assert _red_conns() == ["⚠ in · data"], _red_conns()      # the input itself painted red
    for nid in ("RDG", "RDM", _lab[0], _new[0], "RDB", "RDL"):
        _rdoc.remove_node(nid)
    win.scene.sync(); app.processEvents()
    _ok("RD1 ready-to-run: an empty background sample is reported with its socket painted "
        "red and `+ Draw Regions` adds the node fed from the same image and wired into "
        "`regions`; a missing Label offers Connected Components first and inserts it on the "
        "wire; an unwired node reports that alone; a satisfied node shows the green tick")

    # ── RD2: drawing lives in the NODE'S PANEL, and a node that wants a region goes and
    # draws it on a Draw Regions node in line, then comes back (2026-10-02) ───────────
    from PySide6.QtCore import QEvent as _QEv, QPointF as _QPF
    from PySide6.QtGui import QMouseEvent as _QME
    from nodelab_v2.node_item import NodeItem as _NodeItem
    _rdoc.add_node("io.load", node_id="RD2L", x=0, y=1300)
    _rdoc.add_node("enhance.subtract_background", node_id="RD2B", x=400, y=1300,
                   modes={"approach": "zero_regions"})
    _rdoc.connect("RD2L", "image", "RD2B", "data")
    win.scene.sync(); app.processEvents()
    win.viewer.cancel_pick()
    win._select_only("RD2B"); app.processEvents()
    _pk = [b for b in win.inspector.findChildren(_QTB) if b.property("role") == "pick"]
    assert len(_pk) == 1 and "Draw Regions node" in _pk[0].text(), [b.text() for b in _pk]
    _before = set(_rdoc.nodes)
    _pk[0].click()                               # → add Draw Regions in line, select, pull, arm
    _await_pull(); app.processEvents(); time.sleep(0.05); app.processEvents()
    _dr = [nid for nid in _rdoc.nodes if nid not in _before]
    assert len(_dr) == 1 and _rdoc.nodes[_dr[0]].op_key == "analysis.draw_regions", _dr
    assert ("RD2L", "image", _dr[0], "data") in _rdoc.edges and \
        (_dr[0], "out", "RD2B", "regions") in _rdoc.edges, "in line on the region input"
    _sel = [i.node_id for i in win.scene.selectedItems() if isinstance(i, _NodeItem)]
    assert _sel == [_dr[0]] and win.inspector._node.node_id == _dr[0], (_sel, "switched to it")
    assert win.viewer.picking() and win.viewer.pick_node_id() == _dr[0], "armed on the draw node"
    assert not win.viewer._pick_bar.isVisible(), "NO drawing controls on the image"
    assert win._pick_return == (win.workspace.active, "RD2B")
    _texts = {b.text() for b in win.inspector.findChildren(_QTB)}
    assert {"Undo", "Clear", "Close polygon", "✓  Apply", "Cancel"} <= _texts, _texts
    # the node's own Tool / Operation params drive the gesture
    _drn = win.inspector._node
    win.inspector._set_param(_drn, "tool", "circle"); app.processEvents()
    assert win.viewer._pick.tool == "circle", win.viewer._pick.tool
    win.inspector._set_param(_drn, "op", "cut"); app.processEvents()
    assert win.viewer._pick.op == "cut"
    win.inspector._set_param(_drn, "tool", "rect"); win.inspector._set_param(_drn, "op", "add")
    app.processEvents()
    # a real drag on the image → one shape, and the PANEL's readout says so
    _surf = win.viewer._pick_targets()[-1]
    _ctr = _surf.rect().center()
    for _t, _dx, _dy, _btn in ((_QEv.MouseButtonPress, -15, -10, Qt.LeftButton),
                               (_QEv.MouseMove, 15, 12, Qt.LeftButton),
                               (_QEv.MouseButtonRelease, 15, 12, Qt.LeftButton)):
        _pt = _QPF(_ctr.x() + _dx, _ctr.y() + _dy)
        app.sendEvent(_surf, _QME(_t, _pt, _pt, Qt.LeftButton, _btn, Qt.NoModifier))
    app.processEvents()
    assert win.viewer.pick_shape_count() == 1, win.viewer.pick_shape_count()
    _ro = win.inspector._draw_widgets.get("readout")
    assert _ro is not None and "1 shape" in _ro.text(), (_ro and _ro.text())
    # Apply in the panel: shapes land on the draw node, stamped; we are back on RD2B
    [b for b in win.inspector.findChildren(_QTB) if b.text() == "✓  Apply"][0].click()
    app.processEvents(); _await_pull(); app.processEvents()
    _shp = json.loads(_rdoc.nodes[_dr[0]].params["shapes"])
    assert len(_shp) == 1 and _shp[0]["type"] == "rect" and "frame" in _shp[0], _shp
    _sel = [i.node_id for i in win.scene.selectedItems() if isinstance(i, _NodeItem)]
    assert _sel == ["RD2B"] and win.inspector._node.node_id == "RD2B", _sel
    assert not win.viewer.picking() and win._pick_return is None
    assert not any(l.text().startswith("⚠  ") for l in win.inspector.findChildren(_QL)), \
        "with regions wired and drawn, Subtract Background is ready"
    # on the draw node itself: its panel summarises what it holds, and Draw re-arms there
    win._select_only(_dr[0]); app.processEvents()
    _ro = win.inspector._draw_widgets.get("readout")
    assert _ro is not None and "1 shape" in _ro.text() and "1 frame" in _ro.text(), _ro.text()
    for nid in ("RD2B", _dr[0], "RD2L"):
        _rdoc.remove_node(nid)
    win.scene.sync(); app.processEvents()
    _ok("RD2 draw in the panel: Subtract Background's region button drops a Draw Regions "
        "node in line on `regions`, switches to it and arms its drawing with NO bar on the "
        "image; Tool/Operation params drive the gesture; a real drag makes a shape the panel "
        "readout counts; Apply in the panel writes stamped shapes and returns to Subtract "
        "Background, now ready")
    win._show_page(win._main_canvas, _rd_home)      # back to the demo page
    app.processEvents()

    # ── SP1: Split Positions grows one output per stage position on the LIVE canvas and
    # its wires are drawable (2026-10-02) ───────────────────────────────────────────
    _spdoc = win.doc
    _spdoc.add_node("io.load", node_id="SPL", x=0, y=1500)
    _spdoc.meta_seeds["SPL"] = MetaEnvelope(
        axes=AxisSizes(m=3, t=2, z=1, c=1, y=64, x=64),
        metadata={"pixel_size_um": 0.5, "position_name": ["A1", "B2", "C3"]})
    _spdoc.add_node("util.split_positions", node_id="SPS", x=300, y=1500)
    _spdoc.connect("SPL", "image", "SPS", "data")
    _spdoc.add_node("view.viewer", node_id="SPV", x=600, y=1500)
    _spdoc.connect("SPS", "pos2", "SPV", "data")
    win.scene.sync(); app.processEvents()
    _spitem = win.scene.node_items["SPS"]
    _spouts = [s.name for s in _spdoc.output_specs("SPS")]
    assert _spouts == ["out", "pos0", "pos1", "pos2"], _spouts
    assert [s.label for s in _spdoc.output_specs("SPS")][1:] == ["0 · A1", "1 · B2", "2 · C3"]
    # the card lays the synthetic sockets out (one port item per output, by name)
    _ports = [getattr(p, "name", None) for p in getattr(_spitem, "_outs", {}).values()] \
        if isinstance(getattr(_spitem, "_outs", None), dict) else None
    if _ports is not None:
        assert {"pos0", "pos1", "pos2"} <= set(_ports), _ports
    assert _spdoc.env("SPV").axes.m == 1, "the viewer sees one position through the tap"
    _g = _spdoc.to_graph(for_run=True, materialize=True)
    assert any(n.op_key == "util.select_position" and n.params == {"position": "2"}
               for n in _g.nodes.values()), [(n.id, n.op_key) for n in _g.nodes.values()]
    for nid in ("SPV", "SPS", "SPL"):
        _spdoc.remove_node(nid)
    win.scene.sync(); app.processEvents()
    _ok("SP1 split positions: a 3-position source grows pos0..pos2 on the card with the "
        "file's point names, a wire from pos2 is drawable, the envelope downstream reads "
        "m=1, and the run graph carries a util.select_position tap with position '2'")

    _probe_movie_editor(win, app)

    # ── PG1–PG7: pages in the GUI (V4.00 step 5) ─────────────────────────────────────
    from PySide6.QtCore import QEvent as _PEv, QPointF as _PPF, Qt as _PQt
    from PySide6.QtGui import QAction, QMouseEvent as _PME
    from PySide6.QtWidgets import QMenu as _PMenu
    from nodelab_v2.inspector import _NoWheelCombo as _PCombo
    from nodelab_v2.scene import compatible_ops as _pcompat, visible_specs as _pvis
    from nodelab_v2.workspace import qualify as _pq
    win.set_solo_frame(False)
    win._follow_act.setChecked(False)
    win.file_new()
    win.build_demo()
    app.processEvents()
    _pdone: list = []
    win.runner.finished.connect(lambda nid, *a: _pdone.append(nid))

    def _pwait(rid, timeout=180):
        t0 = time.time()
        while rid not in _pdone and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.005)
        assert rid in _pdone, (rid, _pdone)
        for _ in range(3):
            app.processEvents()

    def _press(view):
        """A real left press (and release) on a canvas's empty corner."""
        vp = view.viewport()
        pt = _PPF(vp.width() - 30.0, vp.height() - 30.0)
        for et, btns in ((_PEv.MouseButtonPress, _PQt.LeftButton),
                         (_PEv.MouseButtonRelease, _PQt.NoButton)):
            QApplication.sendEvent(vp, _PME(et, pt, pt, _PQt.LeftButton, btns,
                                            _PQt.NoModifier))
        app.processEvents()

    def _palette_ops():
        out, stack = set(), [win.palette._tree.topLevelItem(i)
                             for i in range(win.palette._tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            op = it.data(0, _PQt.UserRole)
            if op:
                out.add(op)
            stack.extend(it.child(i) for i in range(it.childCount()))
        return out

    _p1 = win.workspace.active
    _main = win._main_canvas
    # PG1 a Refinement page: the canvas switches to it, empty; the palette, the link search
    # and the window title follow its kind
    _all = {s.op_key for s in _pvis(None)}         # a Free page's set: everything
    _p2 = win.new_page("refine")
    app.processEvents()
    assert win.workspace.active == _p2 and win.canvas is _main and _main.page_id == _p2
    assert not win.doc.nodes and win.welcome.isVisible()
    _ref = _palette_ops()
    assert _ref == {s.op_key for s in _pvis("refine")} and _ref < _all, (len(_ref), len(_all))
    assert "io.load" not in _ref and "enhance.gaussian" in _ref and "page.input" in _ref
    assert "Refinement" in win.palette._kind_chip.text(), win.palette._kind_chip.text()
    assert "Refinement" in win.windowTitle(), win.windowTitle()
    assert _main.view.page_button.isVisible() and \
        _main.view.page_button.text().strip() == win.workspace.page(_p2).name
    _gspec = next(s for s in _pvis(None) if s.op_key == "enhance.gaussian")
    assert all(op in _ref for op in {s.op_key for s, _n in
                                      _pcompat(_gspec.outputs[0], "out", "refine")}), \
        "the link search offers only the page's nodes"
    _ok("PG1 New page ▸ Refinement: the canvas shows the new, empty page (welcome card up); "
        "the palette, the link search and the readiness suggestions offer only the "
        "refinement nodes; the switcher and the window title name the page and its kind")

    # PG7 a named Page Output on page 1, read on the Refinement page through the inspector's
    # Source menu: the card says what it reads, the envelope crosses, a pull runs through it
    win._show_page(_main, _p1)
    app.processEvents()
    _out = win.doc.add_node("page.output", x=560, y=650, params={"name": "raw"})
    win.doc.connect("n2", "out", _out.id, "data")
    win.scene.sync()
    assert win.scene.node_items[_out.id]._page_boundary_label() == "Output · raw"
    win._show_page(_main, _p2)
    app.processEvents()
    _pin = win.doc.add_node("page.input", x=40, y=120)
    assert win.doc.nodes[_pin.id].params.get("source") == f"{_p1}:raw", \
        "a hand-placed Page Input is bound to the default source (step 11)"
    _gam = win.doc.add_node("enhance.gamma", x=320, y=120)
    win.doc.connect(_pin.id, "out", _gam.id, "data")
    win.scene.sync()
    win.scene.clearSelection()
    win.scene.node_items[_pin.id].setSelected(True)
    app.processEvents()
    _src = next((c for c in win.inspector.findChildren(_PCombo)
                 if any(c.itemData(i) == f"{_p1}:raw" for i in range(c.count()))), None)
    assert _src is not None, "the Page Input's Source menu lists the upstream Output"
    _i = next(i for i in range(_src.count()) if _src.itemData(i) == f"{_p1}:raw")
    assert "raw" in _src.itemText(_i) and not _src.isEditable()
    _src.setCurrentIndex(_i)
    _src.activated.emit(_i)
    app.processEvents()
    assert win.doc.nodes[_pin.id].params.get("source") == f"{_p1}:raw"
    assert win.scene.node_items[_pin.id]._page_boundary_label() == \
        f"Input · {win.workspace.page(_p1).name} · raw"
    assert win.doc.env(_pin.id).axes is not None, "the upstream envelope crosses the pages"
    _pdone.clear()
    win.pull_node(_gam.id)
    _pwait(_pq(_p2, _gam.id))
    assert win.viewer.binding == (_p2, _gam.id) and win.viewer.has_image()
    _ok("PG7 a named Page Output on one page is offered by a Page Input's Source menu on a "
        "Refinement page (a closed list, labelled page · name); both cards say what they "
        "carry; the upstream envelope crosses and a pull through the Input runs")

    # PG2 two canvases on two pages: a press on one makes ITS page the active page
    _c2 = win.open_canvas(_p1)
    app.processEvents()
    assert _c2.page_id == _p1 and win.canvas is _c2 and win.workspace.active == _p1
    assert win.view is _c2.view and win.scene is win.scene_for(_p1)
    assert win.shell.dock_of(_c2) is not None, "a second canvas is a dock"
    _press(_main.view)
    assert win.canvas is _main and win.workspace.active == _p2 and win.view is _main.view
    assert "Refinement" in win.palette._kind_chip.text()
    _press(_c2.view)
    assert win.canvas is _c2 and win.workspace.active == _p1
    assert "Image Input" in win.palette._kind_chip.text(), win.palette._kind_chip.text()
    _ok("PG2 a second canvas (a dock) shows another page; a press on a canvas makes its "
        "page the active one — the palette, the inspector and every edit follow")

    # PG6 the same node id on two pages: two viewers, one per page, never share a result
    assert _gam.id in win.workspace.page(_p1).doc.nodes, \
        "the fixture relies on page ids colliding across pages"
    _vb6 = win.viewer
    _vo = win.shell.spawn("viewer", beside=win._viewer_dock(_vb6)).panel
    _pdone.clear()
    win.pull_node(_gam.id, viewer=_vo)                       # the active page: p1
    _pwait(_pq(_p1, _gam.id))
    assert _vo.binding == (_p1, _gam.id) and _vo.showing()[0] == _pq(_p1, _gam.id)
    assert _vb6.binding == (_p2, _gam.id) and _vb6.showing()[0] == _pq(_p2, _gam.id), \
        "the other page's same-id result must not land here"
    win._viewer_dock(_vo).close()
    app.processEvents()
    _ok("PG6 two pages each with node " + _gam.id + ": a viewer bound to one page's node "
        "never shows the other page's (run ids are page-qualified end to end)")

    # PG9 (step 5 review) Shift+F5 pulls again what the active viewer shows — on ITS page,
    # while another page is the active one (`_viewed` names a node of the active page only)
    assert win.workspace.active == _p1 and win._active_viewer() is _vb6
    assert _vb6.binding == (_p2, _gam.id) and win._viewed is None
    _again = next(a for a in win.findChildren(QAction)
                  if a.shortcut().toString() == "Shift+F5")
    _st9: list = []

    def _on_st9(nid):
        _st9.append(nid)

    win.runner.started.connect(_on_st9)
    _pdone.clear()
    _again.trigger()
    _pwait(_pq(_p2, _gam.id))
    win.runner.started.disconnect(_on_st9)
    assert _pq(_p2, _gam.id) in _st9, _st9
    _ok("PG9 Shift+F5 re-pulls the active viewer's node on its own page while another page "
        "is the active one (F9, Hold, Bake and Release re-pull the same way)")

    # PG10 (step 5 review) a pick armed for one page's node is written THERE when another
    # page — a duplicate, holding the same node ids — is active by the time it is applied
    from nodelab_v2.picker import request_for as _pgrq
    _d1 = win.workspace.page(_p1).doc
    _d1.add_node("enhance.gamma", node_id="PKG", x=548, y=650)
    _d1.connect("n2", "out", "PKG", "data")
    _d1.add_node("io.write_movie", node_id="PKM", x=1064, y=430)
    _d1.connect("n3", "out", "PKM", "data")
    _mv_vis = win._movie_dock.isVisible()
    _pd = win.duplicate_page(_p1, canvas=_main)              # the copy, on the main canvas
    app.processEvents()
    _dd = win.workspace.page(_pd).doc
    assert {"PKG", "PKM"} <= set(_dd.nodes)
    _press(_c2.view)                                          # page 1 is the one worked in
    assert win.workspace.active == _p1
    _pdone.clear()
    win.pull_node("PKG")
    _pwait(_pq(_p1, "PKG"))
    _vk = win._active_viewer()
    win._arm_pick(_pgrq("PKG", _d1.nodes["PKG"].spec().input("gamma")))
    assert _vk.pick_node_id() == "PKG" and _vk.pick_page_id() == _p1
    _vk._on_lut_gamma(_vk._lut_channel(), 0.55)               # the user drags the gamma dot
    _press(_main.view)                                        # …glances at the copy
    assert win.workspace.active == _pd
    _vk.apply_pick()                                          # …and applies the pick
    app.processEvents()
    assert _d1.nodes["PKG"].params.get("gamma") == 0.55 and "gamma" in _d1.nodes["PKG"].locked
    assert _dd.nodes["PKG"].params.get("gamma") is None and "gamma" not in _dd.nodes["PKG"].locked
    _ok("PG10 a pick armed for a node of one page is applied to THAT page's node after a "
        "page holding the same node ids became the active one")

    # PG11 (step 5 review) the Movie Editor stays on the page it was opened on: with the
    # copy active, its edits land on its own page's Export Movie and its fetches reach it
    _press(_c2.view)
    win.open_movie_editor("PKM")
    assert win._movie_pid() == _p1 and win.movie_editor.bound() == "PKM"
    _press(_main.view)
    assert win.workspace.active == _pd and win._movie_pid() == _p1
    assert win.workspace.page(_p1).name in win.movie_editor._title.text(), \
        win.movie_editor._title.text()
    win.movie_editor._host.set_sweep("PKM", "timeline")
    app.processEvents()
    assert _d1.nodes["PKM"].modes.get("sweep") == "timeline" and \
        _d1.nodes["PKM"].params.get("timeline"), "converted on the editor's own page"
    assert _dd.nodes["PKM"].modes.get("sweep") != "timeline" and \
        not _dd.nodes["PKM"].params.get("timeline"), "the copy's movie is untouched"
    _fx: list = []
    win.movie_editor.on_fetched = lambda n, p: _fx.append(n)
    try:
        win._on_movie_fetched(_pq(_p1, "n3"), None, 0.0)     # its own page's source
        win._on_movie_fetched(_pq(_pd, "n3"), None, 0.0)     # the copy's: not its business
    finally:
        del win.movie_editor.on_fetched                      # the class method again
    assert _fx == ["n3"], _fx
    _ok("PG11 the Movie Editor keeps editing the Export Movie of the page it was opened on — "
        "its title names that page, its edits land there and its fetched sources reach it — "
        "while a copy with the same node ids is the active page")

    # PG12 (step 5 review) a Hold or a Bake — finished or stopped — retires its run's claims
    # on every page's canvas, as a finished pull does: no card is left 'queued'
    _tg = _pq(_pd, "PKG")
    win._on_run_plan(_tg, [_pq(_p1, "n1"), _pq(_p1, "n2"), _tg])
    assert win.scene_for(_p1)._plans.get(_tg), "the run claims page 1's cards"
    win._on_baked(_tg, {"cancelled": True})                   # a stopped bake
    app.processEvents()
    assert _tg not in win.scene_for(_p1)._plans and "PKG" not in win.scene_for(_pd)._plans
    _tg2 = _pq(_pd, "gone")
    win._on_run_plan(_tg2, [_pq(_p1, "n1"), _tg2])
    assert win.scene_for(_p1)._plans.get(_tg2)
    win._on_held(_tg2, {})                                    # its card went away meanwhile
    app.processEvents()
    assert _tg2 not in win.scene_for(_p1)._plans
    assert not any(v[0] in ("queued", "running") for v in win.scene_for(_p1)._run.values())
    _ok("PG12 a stopped Bake and a Hold retire their run's claims on every page's canvas — no "
        "card is left reading 'queued' on the page the run computed through")
    win._show_page(_main, _p2)
    app.processEvents()
    assert win.delete_page(_pd, confirm=False)
    _d1.remove_node("PKG")
    _d1.remove_node("PKM")
    app.processEvents()
    _press(_c2.view)
    assert win.workspace.active == _p1 and _main.page_id == _p2
    assert win.movie_editor.bound() is None, "its movie node is gone"
    win._movie_dock.setVisible(_mv_vis)
    app.processEvents()

    # PG3 the switcher's menu, and the page operations behind it
    _m = _PMenu()
    win.fill_page_menu(_m, _main)
    _texts = [a.text() for a in _m.actions()]
    _names = [p.name for p in win.workspace.pages.values()]
    assert all(n in _texts for n in _names), (_texts, _names)
    assert {"New page", "Duplicate page", "Rename page…", "Delete page"} <= set(_texts), _texts
    _checked = [a.text() for a in _m.actions() if a.isCheckable() and a.isChecked()]
    assert _checked == [win.workspace.page(_main.page_id).name], _checked
    assert win.rename_page(_p2, "Refine A")
    app.processEvents()
    assert _main.view.page_button.text().strip() == "Refine A"
    _p3 = win.duplicate_page(_p2, canvas=_main)
    app.processEvents()
    assert _main.page_id == _p3 and win.workspace.page(_p3).name == "Refine A copy"
    assert set(win.doc.nodes) == set(win.workspace.page(_p2).doc.nodes)
    assert win.delete_page(_p3, confirm=False)
    app.processEvents()
    assert _p3 not in win.workspace.pages and _main.page_id in win.workspace.pages
    assert _p3 not in win._scenes, "a deleted page's scene goes with it"
    _ok("PG3 the switcher lists every page grouped by kind (the shown one ticked) with New "
        "page, Duplicate, Rename and Delete; a rename re-titles the switcher, a duplicate "
        "copies the graph and is shown, a delete drops the page and its scene")

    # PG13 (step 5 review) a page renamed or deleted from ANOTHER canvas's switcher while the
    # active page stays: the inspector re-reads the page list — a Page Input's Source names
    # the new page name, and once its Output's page is gone it reads unbound, in the menu and
    # in Ready-to-run
    win._show_page(_main, _p2)
    _press(_main.view)
    win.scene.clearSelection()
    win.scene.node_items[_pin.id].setSelected(True)
    app.processEvents()
    assert win.workspace.active == _p2 and win.inspector._node.node_id == _pin.id

    def _srcbox(value):
        return next((c for c in win.inspector.findChildren(_PCombo)
                     if any(c.itemData(i) == value for i in range(c.count()))), None)

    _nm1 = win.workspace.page(_p1).name
    assert win.rename_page(_p1, "Acquired")
    app.processEvents()
    assert win.workspace.active == _p2, "renaming another page leaves the active one"
    _b13 = _srcbox(f"{_p1}:raw")
    assert _b13 is not None and _b13.currentText().startswith("Acquired"), \
        [_b13.itemText(i) for i in range(_b13.count())] if _b13 is not None else None
    assert win.rename_page(_p1, _nm1)
    app.processEvents()
    _px = win.workspace.add_page("Scratch input", "input").id
    win.workspace.page(_px).doc.add_node("page.output", node_id="TO", params={"name": "tmp"})
    win.doc.nodes[_pin.id].params["source"] = f"{_px}:tmp"
    win.doc.touch(_pin.id)
    win.inspector.set_node(win.scene.node_items[_pin.id])
    app.processEvents()
    assert "unbound" not in _srcbox(f"{_px}:tmp").currentText()
    assert not any(p.kind == "unbound" for p in win.inspector._problems)
    assert win.delete_page(_px, confirm=False)
    app.processEvents()
    assert win.workspace.active == _p2
    _b13 = _srcbox(f"{_px}:tmp")
    assert _b13 is not None and "unbound" in _b13.currentText(), \
        [_b13.itemText(i) for i in range(_b13.count())] if _b13 is not None else None
    assert any(p.kind == "unbound" for p in win.inspector._problems), win.inspector._problems
    win.doc.nodes[_pin.id].params["source"] = f"{_p1}:raw"
    win.doc.touch(_pin.id)
    app.processEvents()
    _ok("PG13 a page renamed or deleted from another canvas's switcher: the shown Page "
        "Input's Source follows the new name, and reads unbound — in its menu and in "
        "Ready-to-run — once the page holding its Output is gone")

    # PG4 Ctrl+PgDn / Ctrl+PgUp step the active canvas through the pages
    _order = list(win.workspace.pages)
    _next = next(a for a in win.findChildren(QAction) if a.shortcut().toString() == "Ctrl+PgDown")
    _prev = next(a for a in win.findChildren(QAction) if a.shortcut().toString() == "Ctrl+PgUp")
    _before = win.canvas.page_id
    _next.trigger()
    app.processEvents()
    assert win.canvas.page_id == _order[(_order.index(_before) + 1) % len(_order)]
    _prev.trigger()
    app.processEvents()
    assert win.canvas.page_id == _before
    _ok("PG4 Ctrl+PgDn / Ctrl+PgUp step the active canvas through the workspace's pages")

    # PG5 a page keeps its selection and the canvas its viewpoint across a round trip
    win._show_page(_main, _p1)
    app.processEvents()
    win.scene.clearSelection()
    win.scene.node_items["n3"].setSelected(True)
    _main.view.resetTransform()
    _main.view.scale(1.7, 1.7)
    app.processEvents()
    _z = _main.view.transform().m11()
    win._show_page(_main, _p2)
    app.processEvents()
    win._show_page(_main, _p1)
    app.processEvents()
    assert win.scene.node_items["n3"].isSelected(), "the page kept its selection"
    assert abs(_main.view.transform().m11() - _z) < 1e-9, "the canvas kept its viewpoint"
    _ok("PG5 switching a canvas away from a page and back keeps the page's selection and "
        "the canvas's zoom and position")

    # PG8 a file of several pages, opened over a window whose active page has ANOTHER id:
    # every page comes back on its own scene (a load keeps the active page's document
    # object for the new active page, so a scene cached under the old id is stale)
    _pfile = os.path.join(tempfile.mkdtemp(prefix="nd2pages_"), "pages.nd2graph.json")
    win.workspace.save_file(_pfile)
    _saved_active = win.workspace.active
    _other = next(p for p in win.workspace.pages if p != _saved_active)
    win._show_page(_main, _other)
    app.processEvents()
    win.file_new()
    app.processEvents()
    assert list(win.workspace.pages)[0] == _other and len(win.workspace.pages) == 4
    assert win.workspace.pages[_other].kind == "input"
    win.workspace.load_file(_pfile)
    app.processEvents()
    assert win.workspace.active == _saved_active and win.canvas.page_id == _saved_active
    for _pid, _pg in win.workspace.pages.items():
        _sc = win.scene_for(_pid)
        assert _sc.doc is _pg.doc and set(_sc.node_items) == set(_pg.doc.nodes), \
            (_pid, sorted(_sc.node_items), sorted(_pg.doc.nodes))
    win._show_page(_main, _other)
    app.processEvents()
    assert win.view.scene() is win.scene_for(_other)
    assert set(win.scene.node_items) == set(win.workspace.page(_other).doc.nodes)
    # the dropped scenes listen to nothing: an edit on every page reaches its own scene,
    # and a palette drop draws its card (a dead listener raised before the live ones ran)
    from nodelab_v2.scene import GraphScene as _GSpg
    for _pid, _pg in win.workspace.pages.items():
        assert sum(1 for _fn in _pg.doc._listeners
                   if getattr(_fn, "__func__", None) is _GSpg.sync) == 1, _pid
        _pg.doc.touch()
    _before = set(win.doc.nodes)
    win._on_op_dropped("enhance.gamma", QPointF(300.0, 300.0))
    app.processEvents()
    _new = set(win.doc.nodes) - _before
    assert len(_new) == 1 and _new <= set(win.scene.node_items), (_new, sorted(win.scene.node_items))
    _ok("PG8 a file of several pages opens with every page on its own scene, even over a "
        "window whose active page had another id; the scenes it replaced listen to nothing, "
        "so the next edit and a palette drop reach the live canvas")

    # a docked canvas closes — even while maximized: the viewer in its mini-map goes back
    # to its dock first (the mini-map dies with the canvas); the main canvas cannot close
    if not win.viewer.has_image():
        _pdone.clear()
        win.pull_node("n3")
        _pwait(_pq(win.workspace.active, "n3"))
    win._on_canvas_maximize(_c2, True)
    app.processEvents()
    assert win._maximized and win._max_canvas is _c2 and win.minimap.parentWidget() is _c2.view
    _mv = win._mini_viewer
    _d2 = win.shell.dock_of(_c2)
    _d2.close()
    app.processEvents()
    assert not win._maximized and win._max_canvas is None
    assert win._viewer_dock(_mv) is not None and win._viewer_dock(_mv).widget() is _mv
    assert _mv.has_image(), "the viewer survived its mini-map's canvas"
    assert win.canvases() == [_main] and win.canvas is _main
    assert win.centralWidget() is _main and win.shell.dock_of(_main) is None
    _ok("PG canvases: a docked canvas closes and the main one takes over; the main canvas "
        "is the window's centre, so the last canvas can never be closed")

    # PG14 (step 5 review) an edit on a page that another page does not read leaves that
    # page's viewer whole — its held view, its overlay channels (a Viewer node's second
    # source) and a preload in flight; an edit on its own page still drops them
    win.file_new()
    win.build_demo()
    app.processEvents()
    _q1 = win.workspace.active
    _dq = win.doc
    _dq.add_node("view.viewer", node_id="PGV", x=1330, y=430)
    _dq.connect("n3", "out", "PGV", "data")
    _dq.connect("n2", "out", "PGV", "source_2")
    _q2 = win.new_page("free")
    app.processEvents()
    _dq2 = win.workspace.page(_q2).doc
    _gq = _dq2.add_node("enhance.gamma", x=0, y=0)
    win._show_page(win.canvas, _q1)
    app.processEvents()
    _qr = _pq(_q1, "PGV")
    _pdone.clear()
    win.pull_node("PGV")
    _pwait(_qr)
    _rn = win.runner
    _ql = dict(_rn.overlay_channels(_qr))
    assert _ql, "the second source is composited as overlay channels"
    _qo = sorted(_ql)[0]
    _qp: list = []

    def _on_qp(nid, pl, *_a):
        _qp.append((nid, sorted(pl)))

    _rn.plane_ready.connect(_on_qp)

    def _qserve():
        _qp.clear()
        _rn.request_plane(_qr, (0, 0, 0, 0), (0, _qo))
        t0 = time.time()
        while not any(n == _qr for n, _p in _qp) and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.005)
        return [p for n, p in _qp if n == _qr][-1]

    assert _qo in _qserve()
    _rn._preload_total, _rn._preload_done, _rn._preload_node = 9, 0, _qr   # page 1 plays
    _dq2.nodes[_gq.id].params["gamma"] = 2.0
    _dq2.touch(_gq.id)                                        # page 2 reads nothing of page 1
    app.processEvents()
    assert _rn._view_of(_qr) is not None and dict(_rn.overlay_channels(_qr)) == _ql
    assert _rn.preloading() and _rn._preload_node == _qr, "page 1's preload keeps going"
    assert _qo in _qserve(), "page 1's overlay channel is still served"
    _dq.touch("n3")                                           # an edit on page 1 itself
    app.processEvents()
    assert not _rn.preloading() and _rn._view_of(_qr) is None
    _rn.plane_ready.disconnect(_on_qp)
    _ok("PG14 an edit on a page another page does not read leaves that page's viewer whole "
        "(its held view, overlay channels and a running preload); an edit on its own page "
        "still drops them")

    # ── LK1–LK3: linked pages (V4.00 step 6) ──────────────────────────────────────────
    from PySide6.QtCore import QTimer as _LkTimer
    from nodelab_v2.linked_document import LinkedDocument as _LkDoc, TOPOLOGY_HINT as _LkHint
    win.file_new()
    win.build_demo()
    app.processEvents()
    _lm_id = win.workspace.active
    _lm = win.doc
    _m6 = _PMenu()
    win.fill_page_menu(_m6, win.canvas)
    assert "Duplicate as linked page" in [a.text() for a in _m6.actions()]
    _lp = win.duplicate_page(_lm_id, linked=True)
    app.processEvents()
    _lk = win.doc
    assert win.workspace.active == _lp and isinstance(_lk, _LkDoc) and _lk.master is _lm
    assert win.workspace.page(_lp).master == _lm_id
    assert set(win.scene.node_items) == set(_lm.nodes), "the master's cards"
    _m6 = _PMenu()
    win.fill_page_menu(_m6, win.canvas)
    _t6 = [a.text() for a in _m6.actions()]
    assert {"Go to master page", "Make unique"} <= set(_t6), _t6
    assert any(t.startswith(win.workspace.page(_lp).name) and "(linked · " in t
               for t in _t6), _t6
    # every structural gesture ASKS how to apply it (V4.00 step 11e) — answered Cancel here
    # (the dialog's own section is LE1), so nothing changes and the hint says why
    _asked6: list = []
    win.ask_linked_edit = lambda pid, shape=False, push_note="": (_asked6.append(pid) or "")
    _n6 = set(_lk.nodes)
    win.statusBar().clearMessage()
    win._on_op_dropped("enhance.gamma", QPointF(300.0, 300.0))
    assert win.statusBar().currentMessage() == _LkHint and set(_lk.nodes) == _n6
    assert _asked6 == [_lp], _asked6
    win.statusBar().clearMessage()
    assert win.scene.delete_nodes(["n3"]) is None and "n3" in _lk.nodes
    assert win.statusBar().currentMessage() == _LkHint
    _cm = _PMenu()
    win.scene._fill_node_menu(_cm, win.scene.node_items["n3"])
    win.scene._lock_structural(_cm)
    from nodelab_v2.linked_document import ASK_HINT as _LkAsk
    _del = next(a for a in _cm.actions() if a.text().startswith("Delete"))
    assert _del.isEnabled() and _del.toolTip() == _LkAsk, "it asks — and says it will"
    win.statusBar().clearMessage()
    _LkTimer.singleShot(0, lambda: _lk.add_node("enhance.gamma"))    # no guard: the backstop
    _t0 = time.time()
    while not win.statusBar().currentMessage() and time.time() - _t0 < 5:
        app.processEvents()
    assert win.statusBar().currentMessage() == _LkHint and set(_lk.nodes) == _n6
    # a value edit is this page's override: marked in Properties, reset to the master's
    win.scene.clearSelection()
    win.scene.node_items["n3"].setSelected(True)
    app.processEvents()
    _ins = win.inspector
    assert "Linked to" in _ins._linked_text and "0 overrides" in _ins._linked_text
    _ins._set_param(win.scene.node_items["n3"], "sigma", 3.25)
    app.processEvents()
    assert _lk.is_overridden("n3", "sigma") and _lm.nodes["n3"].params.get("sigma") != 3.25
    assert "sigma" in _lk.nodes["n3"].locked, "an override is a pinned value"
    _ins.set_node(win.scene.node_items["n3"])
    assert ("n3", "sigma") in _ins._override_rows and "1 override" in _ins._linked_text
    _ins.reset_override("n3", "sigma")
    _t0 = time.time()
    while ("n3", "sigma") in _ins._override_rows and time.time() - _t0 < 5:
        app.processEvents()
    assert not _lk.is_overridden("n3", "sigma")
    assert _lk.nodes["n3"].params.get("sigma") == _lm.nodes["n3"].params.get("sigma")
    # a master edit reaches the linked page; a card moved on either page moves on the other
    _lm.nodes["n4"].params["threshold"] = 1234.0
    _lm.touch("n4")
    app.processEvents()
    assert _lk.nodes["n4"].params.get("threshold") == 1234.0
    _it = win.scene.node_items["n4"]
    _it.setPos(_it.pos().x() + 55.0, _it.pos().y() + 21.0)
    app.processEvents()
    _mi = win.scene_for(_lm_id).node_items["n4"]
    assert (_mi.pos().x(), _mi.pos().y()) == (_it.pos().x(), _it.pos().y())
    _pdone.clear()
    win.pull_node("n3")
    _pwait(_pq(_lp, "n3"))
    _ok("LK1 Duplicate as linked page shows the master's cards on a page of its own; adding, "
        "deleting or rewiring nodes asks how to apply it — cancelled, nothing changes and the "
        "hint says why (the context menu's entries say they will ask; a refusal from "
        "anywhere reaches the status bar); a value changed there is an override, marked in "
        "Properties and reset to the master's; master edits and card moves reach it; it pulls")

    # LK2 the file keeps the link; Make unique; deleting a master frees its linked pages
    _lk.nodes["n3"].params["sigma"] = 2.5
    _lk.touch("n3")
    _lfile = os.path.join(tempfile.mkdtemp(prefix="nd2linked_"), "linked.nd2graph.json")
    win.workspace.save_file(_lfile)
    win.file_new()
    app.processEvents()
    win.workspace.load_file(_lfile)
    app.processEvents()
    _lpg = win.workspace.page(_lp)
    assert isinstance(_lpg.doc, _LkDoc) and _lpg.doc.nodes["n3"].params["sigma"] == 2.5
    win._show_page(win.canvas, _lp)
    app.processEvents()
    assert set(win.scene.node_items) == set(_lpg.doc.nodes)
    win.scene.clearSelection()
    win.scene.node_items["n3"].setSelected(True)
    app.processEvents()
    win.inspector.linked_action.emit("unique", "n3")
    app.processEvents()
    assert _lpg.master is None and not isinstance(_lpg.doc, _LkDoc)
    assert win.scene.doc is _lpg.doc and set(win.scene.node_items) == set(_lpg.doc.nodes)
    assert win.inspector._node is None or win.inspector._node.scene() is win.scene
    _n6 = set(_lpg.doc.nodes)
    win._on_op_dropped("enhance.gamma", QPointF(300.0, 300.0))
    assert len(set(_lpg.doc.nodes) - _n6) == 1, "its graph is editable again"
    _l2 = win.duplicate_page(_lm_id, linked=True)
    app.processEvents()
    assert win.workspace.dependents_of(_lm_id) == [_l2]
    assert win.delete_page(_lm_id, confirm=False)
    app.processEvents()
    assert win.workspace.page(_l2).master is None and win.workspace.page(_l2).doc.nodes
    _ok("LK2 a linked page survives save and reopen (master + overrides); Make unique turns "
        "it into a page of its own with an editable graph; deleting a master turns its "
        "linked pages into pages of their own")

    # LK3 (step 6 review) the window's own structural actions act on the ACTIVE page and are
    # refused on a linked one: Edit > Dissolve (bound to the page shown, not the first one)
    # and Edit > Delete keep the hint; Make unique keeps the selection
    win.file_new()
    win.build_demo()
    app.processEvents()
    _lm3 = win.workspace.active
    win.scene.clearSelection()
    win.scene.node_items["n7"].setSelected(True)                 # selected on the master
    _lp3 = win.duplicate_page(_lm3, linked=True)
    app.processEvents()
    win.scene.clearSelection()
    win.scene.node_items["n4"].setSelected(True)
    app.processEvents()
    _n3 = set(win.workspace.page(_lm3).doc.nodes)
    win.statusBar().clearMessage()
    win._sync_edit_actions()                  # what opening the Edit menu does
    assert win._dissolve_act.isEnabled()
    win._dissolve_act.trigger()
    app.processEvents()
    assert set(win.workspace.page(_lm3).doc.nodes) == _n3, "Dissolve never reaches the master"
    assert win.statusBar().currentMessage() == _LkHint
    win.statusBar().clearMessage()
    win.delete_selection()
    assert win.statusBar().currentMessage() == _LkHint and "n4" in win.doc.nodes
    win.scene.clearSelection()
    win.scene.node_items["n5"].setSelected(True)
    app.processEvents()
    assert win.make_unique(_lp3)
    app.processEvents()
    assert [i.node_id for i in win.scene.selectedItems() if isinstance(i, _NodeItem)] == ["n5"]
    assert win.inspector._node is not None and win.inspector._node.node_id == "n5"
    _ok("LK3 Edit > Dissolve and Edit > Delete on a linked page, cancelled, change nothing and say why ("
        "the hint) and never reach the master (Dissolve acts on the page shown); Make unique keeps the "
        "selection and Properties")

    # LE1 (V4.00 step 11e) the three answers to a structural edit on a LINKED page — the
    # question is asked once per page; Keep the change on this page makes a MODIFIED linked
    # page (its own node, the master's other edits still arriving); Add it to the master
    # sends a node there switched off (on here, off on the master and its other pages) and
    # refuses a change the master would compute differently; Make unique swaps the page's
    # document and the gesture runs on its new scene; frames ask with only Make unique
    from nodelab_v2.linked_edit_dialog import LinkedEditDialog as _LED14
    from PySide6.QtWidgets import QMenu as _QM14
    win.file_new()
    win.build_demo()
    app.processEvents()
    _m14 = win.workspace.active
    _md14 = win.doc
    _ans14 = ["modified"]
    _asked14: list = []
    win.ask_linked_edit = lambda pid, shape=False, push_note="": (
        _asked14.append((pid, shape, push_note)) or _ans14[0])
    # (a) keep the change on this page
    _l14 = win.duplicate_page(_m14, linked=True)
    app.processEvents()
    _d14 = win.doc
    _b14 = set(_md14.nodes)
    win._on_op_dropped("enhance.gamma", QPointF(2600.0, 900.0))       # clear of every wire
    app.processEvents()
    _own14 = set(_d14.nodes) - _b14
    assert len(_own14) == 1 and set(_md14.nodes) == _b14 and _d14.edit_mode == "modified"
    assert _asked14 == [(_l14, False, "")], _asked14
    _o14 = next(iter(_own14))
    win.scene.delete_nodes(["n7"])                                    # asks no more
    app.processEvents()
    assert len(_asked14) == 1 and "n7" not in _d14.nodes and "n7" in _md14.nodes
    assert "modified" in win.page_title(_l14)[0], win.page_title(_l14)
    _md14.nodes["n3"].params["sigma"] = 1.7                           # the master still arrives
    _md14.touch("n3")
    app.processEvents()
    assert _d14.nodes["n3"].params["sigma"] == 1.7 and _o14 in _d14.nodes
    _st14 = win.workspace.to_dict()
    _rec14 = next(r for r in _st14["workspace"]["pages"] if r["id"] == _l14)
    assert _rec14["structure"]["removed"] == ["n7"] and _o14 in _rec14["structure"]["nodes"]
    _pr14 = win.pages_panel.node_items(_l14)
    assert _pr14[_o14].text(0).startswith("+ "), _pr14[_o14].text(0)
    # (b) add it to the master, switched off
    _l14b = win.duplicate_page(_m14, linked=True)
    app.processEvents()
    _ans14[0] = "master"
    _b14 = set(_md14.nodes)
    win._on_op_dropped("enhance.unsharp", QPointF(2600.0, 1150.0))
    app.processEvents()
    _push14 = set(_md14.nodes) - _b14
    assert len(_push14) == 1, _push14
    _p14 = next(iter(_push14))
    assert _md14.nodes[_p14].muted and not win.doc.nodes[_p14].muted, "off there, on here"
    assert _d14.nodes[_p14].muted, "…and off on the master's other linked pages"
    assert win.doc.edit_mode == "master" and "edits → master" in win.page_title(_l14b)[0]
    assert win.scene.splice_onto(_p14, ("n3", "out", "n4", "data")), "dropped onto a wire"
    assert ("n3", "out", _p14, "data") in _md14.edges and (_p14, "out", "n4", "data") in         _md14.edges, "the splice reached the master"
    assert any(e.src == "n3" and e.dst == "n4" for e in _md14.to_graph(for_run=True).edges),         "…which still computes n3 → n4"
    win.statusBar().clearMessage()
    win.scene.delete_nodes(["n3"])                                    # on in the master
    app.processEvents()
    assert "n3" in _md14.nodes and "would change what the master computes" in \
        win.statusBar().currentMessage(), win.statusBar().currentMessage()
    win.statusBar().clearMessage()
    _b14 = set(_md14.nodes)
    win._on_op_dropped("analysis.label", QPointF(2600.0, 1400.0))     # cannot be switched off
    app.processEvents()
    assert set(_md14.nodes) == _b14 and "cannot go to the master switched off" in \
        win.statusBar().currentMessage(), win.statusBar().currentMessage()
    # (c) make unique — the gesture runs on the page's new scene
    _l14c = win.duplicate_page(_m14, linked=True)
    app.processEvents()
    _ans14[0] = "unique"
    _sc14 = win.scene
    _sc14.delete_nodes(["n5"])
    app.processEvents()
    _pg14 = win.workspace.page(_l14c)
    assert _pg14.master is None and "n5" not in _pg14.doc.nodes and "n5" in _md14.nodes
    assert win.scene is not _sc14 and win.scene.doc is _pg14.doc
    # (d) a frame: only Make unique is possible, so Keep-on-this-page changes nothing
    _l14d = win.duplicate_page(_m14, linked=True)
    app.processEvents()
    _ans14[0] = "modified"
    win.scene.clearSelection()
    win.scene.node_items["n3"].setSelected(True)
    win.frame_selection()
    app.processEvents()
    assert _asked14[-1] == (_l14d, True, ""), _asked14[-1]
    assert not win.doc.frames and win.doc.edit_mode == ""
    # (e) the dialog: a frame greys both linked answers, a load greys the master one
    _dl14 = _LED14(win, "P (linked)", "P", shape=True)
    assert _dl14.buttons["unique"].isEnabled() and not _dl14.buttons["modified"].isEnabled() \
        and not _dl14.buttons["master"].isEnabled()
    _dl14 = _LED14(win, "P (linked)", "P", push_note="a Load card is where data starts")
    assert _dl14.buttons["modified"].isEnabled() and not _dl14.buttons["master"].isEnabled()
    assert "a Load card is where data starts" in _dl14.buttons["master"].text_label.text()
    _dl14.buttons["modified"].click()
    assert _dl14.choice == "modified" and _dl14.result() == _LED14.Accepted
    # (f) the canvas menu's Muted follows the switch-off rule, and works on a linked page
    _cm14 = _QM14()
    win.scene._fill_node_menu(_cm14, win.scene.node_items["n4"])      # Threshold
    _mu14 = next(a for a in _cm14.actions() if a.text().startswith("Muted"))
    assert not _mu14.isEnabled() and _mu14.toolTip().startswith("Cannot be switched off"), \
        _mu14.toolTip()
    win.scene.clearSelection()
    win.scene.node_items["n3"].setSelected(True)
    win.scene._mute_selection()                                       # M, on a linked page
    app.processEvents()
    assert win.doc.nodes["n3"].muted and not _md14.nodes["n3"].muted
    assert win.doc.is_overridden("n3", "muted") and len(_asked14) == 4, _asked14
    del win.ask_linked_edit
    _ok("LE1 a structural edit on a linked page asks once: Keep the change on this page makes "
        "a modified linked page (its own node and removal saved, the master's edits still "
        "arriving, `+` in the Pages panel); Add it to the master sends a node there switched "
        "off and refuses a delete or a node that would change what the master computes; Make "
        "unique runs the gesture on the page's new scene; a frame offers only Make unique; "
        "the canvas menu's Muted is greyed for a Threshold and M works per page")

    # PL1 (V4.00 step 7) Plot XY on the example graph's measured table: its envelope is ONE
    # RGB picture with no table on it; pulled, the viewer shows it in TRUE colour — R, G and
    # B in their own colours on a fixed 0-255 window — and an Analysis page offers it
    import importlib.util as _ilu
    if _ilu.find_spec("matplotlib") is None:
        _ok("PL1 SKIPPED — matplotlib is not installed")
    else:
        win.file_new()
        win.build_demo()
        app.processEvents()
        _dp = win.doc
        _dp.add_node("plot.xy", node_id="PL", x=1600.0, y=150.0)
        _dp.connect("n6", "out", "PL", "data")
        win.scene.sync()
        app.processEvents()
        _pe = _dp.env("PL")
        assert _pe.axes.c == 3 and _pe.axes.t == 1 and {d.value for d in _pe.domains} == {"voxel"} and \
            not _pe.layer_names, (_pe.axes, _pe.domains, _pe.layer_names)
        _prid = _pq(win.workspace.active, "PL")
        _pdone.clear()
        win.pull_node("PL")
        _pwait(_prid, timeout=300)
        _pv = win.viewer
        assert _pv.binding == (win.workspace.active, "PL") and _pv.has_image()
        assert getattr(_pv, "_picture", False), "the viewer knows it shows a picture"
        assert [_pv._chan_colors[i] for i in range(3)] == [(255, 0, 0), (0, 255, 0),
                                                           (0, 0, 255)], _pv._chan_colors
        assert _pv._planes and all(_pv._clim.get(_pv._lut_key(_prid, c)) == (0.0, 255.0)
                                   for c in _pv._planes), \
            {c: _pv._clim.get(_pv._lut_key(_prid, c)) for c in _pv._planes}
        assert "pixel_size_um" not in (_pv._dataset.metadata or {}), "no scale bar on a chart"
        win.new_page("analyze")
        app.processEvents()
        assert {"plot.xy", "io.write_figure"} <= _palette_ops(), "an Analysis page offers them"
        _ok("PL1 Plot XY of the measured table: its envelope is one RGB picture (its own Voxel "
            "domain) with no table; "
            "the viewer shows it in true colour (R/G/B in their own colours on a fixed 0-255 "
            "window, no calibration); an Analysis page offers Plot XY and Export Figure")

        # PL2 (step 7 review) an optional column (Group by) can go back to blank: Properties
        # and the card both offer "(none)"; the plot's channel descriptors are R/G/B
        win._show_page(win.canvas, [p for p in win.workspace.pages][0])
        app.processEvents()
        win.scene.clearSelection()
        win.scene.node_items["PL"].setSelected(True)
        app.processEvents()
        _dp.nodes["PL"].params["group_by"] = "t"
        _dp.touch("PL")
        win.inspector.set_node(win.scene.node_items["PL"])
        app.processEvents()
        _gb = next(c for c in win.inspector.findChildren(_PCombo)
                   if c.count() and c.itemText(0) == "(none)")
        assert _gb.currentText() == "t"
        _gb.setCurrentIndex(0)
        _gb.activated.emit(0)
        app.processEvents()
        assert _dp.nodes["PL"].params.get("group_by") == "", _dp.nodes["PL"].params
        assert [d["name"] for d in _dp.channel_descriptors("PL")] == ["R", "G", "B"]
        _ok("PL2 an optional column socket offers (none) and goes back to blank; a plot's "
            "channels read R/G/B, not the source file's")

        # PL3 (V4.00 step 8) a PER-FRAME Plot XY in the Viewer: one figure per frame of a
        # three-frame series, the frame chooser spans them, and moving it shows that frame's
        # own figure; Plot Distribution, Heatmap and Time Series are offered on Analysis pages
        from nodelab_v2 import runner as _RN8
        # the synthetic source's caches and size are put back as found: later sections
        # rely on the multi-frame source an earlier one left cached
        _synth_before = _RN8._SYNTH_AXES
        _cache_before = (dict(win.runner._providers), dict(win.runner._announced),
                         dict(win.runner._raw_src))
        win.runner._providers.clear()
        win.runner._announced.clear()
        win.runner._raw_src.clear()
        win.runner.invalidate()
        _RN8._SYNTH_AXES = AxisSizes(m=1, t=3, z=1, c=2, y=96, x=128)
        try:
            win.file_new()
            win.build_demo()
            app.processEvents()
            _d8 = win.doc
            _d8.set_meta_seed("n1", MetaEnvelope(axes=_RN8._SYNTH_AXES,
                                                 metadata=dict(_RN8._SYNTH_META)))
            _d8.nodes["n3"].modes["dim"] = "2D"
            _d8.touch("n3")
            _d8.add_node("plot.xy", node_id="PF", x=1600.0, y=150.0,
                         modes={"per": "frame"})
            _d8.connect("n6", "out", "PF", "data")
            win.scene.sync()
            app.processEvents()
            _fe = _d8.env("PF")
            assert _fe.axes.t == 3 and _fe.axes.c == 3, _fe.axes
            _frid = _pq(win.workspace.active, "PF")
            _pdone.clear()
            win.pull_node("PF")
            _pwait(_frid, timeout=300)
            _fv = win.viewer
            assert _fv.binding == (win.workspace.active, "PF") and _fv.has_image()
            assert _fv._dataset.axes.t == 3 and _fv._sliders["t"].maximum() == 2, \
                (_fv._dataset.axes, _fv._sliders["t"].maximum())
            _f0 = {c: np.array(a, copy=True) for c, a in _fv._planes.items()}
            _fv._sliders["t"].setValue(2)
            _t0 = time.time()
            while time.time() - _t0 < 60:
                app.processEvents()
                time.sleep(0.01)
                if _fv._planes and _fv._payload_coords()[1] == 2 and any(
                        c in _f0 and not np.array_equal(_f0[c], a)
                        for c, a in _fv._planes.items()):
                    break
            assert _fv._payload_coords()[1] == 2 and any(
                c in _f0 and not np.array_equal(_f0[c], a) for c, a in _fv._planes.items()), \
                "frame 2 shows its own figure"
            _fv._sliders["t"].setValue(0)
            app.processEvents()
            win.new_page("analyze")
            app.processEvents()
            assert {"plot.distribution", "plot.heatmap", "plot.timeseries"} <= \
                _palette_ops(), "an Analysis page offers the step 8 plots"
        finally:
            _RN8._SYNTH_AXES = _synth_before
            for _cache, _saved in zip((win.runner._providers, win.runner._announced,
                                       win.runner._raw_src), _cache_before):
                _cache.clear()
                _cache.update(_saved)
            win.runner.invalidate()
        _ok("PL3 a per-frame Plot XY of a three-frame series: its envelope and payload carry "
            "T = 3, the Viewer's frame chooser spans them and frame 2 shows its own figure; "
            "an Analysis page offers Plot Distribution, Heatmap and Time Series")

    # TB1 (V4.00 step 9) Table Aggregate on the example graph's measured table: its summary
    # table is in the envelope (columns a closed menu downstream can offer) and on the pulled
    # result the spreadsheet tabulates; Plot XY draws it; Processing and Analysis pages offer
    # the three table nodes
    if _ilu.find_spec("matplotlib") is None:
        _ok("TB1 SKIPPED — matplotlib is not installed")
    else:
        from nodelab_v2.tables import all_tables as _all_tables
        win.file_new()
        win.build_demo()
        app.processEvents()
        _dt = win.doc
        _dt.add_node("table.aggregate", node_id="TA", x=1600.0, y=320.0)
        _dt.connect("n6", "out", "TA", "data")
        _dt.add_node("plot.xy", node_id="TP", x=1860.0, y=320.0,
                     params={"table": "summary", "x": "t", "y": "area_mean"})
        _dt.connect("TA", "out", "TP", "data")
        win.scene.sync()
        app.processEvents()
        _te = _dt.env("TA")
        assert any(n == "summary" for _d, n in _te.layer_names), _te.layer_names
        assert {"n", "area_mean", "area_sem", "area_count"} <= \
            {c for _d, lyr, c in _te.column_names if lyr == "summary"}
        _trid = _pq(win.workspace.active, "TA")
        _pdone.clear()
        win.pull_node("TA")
        _pwait(_trid, timeout=300)
        _tabs = [t for (_d, lyr), t in _all_tables(win.viewer._dataset).items()
                 if lyr == "summary"]
        assert _tabs and "area_mean" in _tabs[0] and "n" in _tabs[0], list(_tabs[0])
        _tprid = _pq(win.workspace.active, "TP")
        _pdone.clear()
        win.pull_node("TP")
        _pwait(_tprid, timeout=300)
        assert win.viewer.binding == (win.workspace.active, "TP") and win.viewer.has_image()
        for _kind in ("process", "analyze"):
            win.new_page(_kind)
            app.processEvents()
            assert {"table.concat", "table.join", "table.aggregate"} <= _palette_ops(), _kind
        _ok("TB1 Table Aggregate of the measured table: a summary table in the envelope (n, "
            "area_mean, area_sem, area_count) and on the pulled result; Plot XY draws it; "
            "Processing and Analysis pages offer Table Concat, Join and Aggregate")

    # PB1 (V4.00 step 10) the Publish-as-recipe dialog builds — for a page's single graph and
    # for a page that reads other pages (published flattened): its target menu holds the
    # draft's target and Validate's draft keeps it
    from nodegraph.selftest import _ws_fixture as _wsf
    from nodegraph.serialize import from_dict as _fd
    from nodelab_v2.lablink import authoring as _AU
    from nodelab_v2.lablink import recipe as _RCP
    _wsp, _dsp, _envp, _axp = _wsf()
    for _draft in (_RCP.draft_from_document(_wsp.pages["pg1"].doc, name="pb-single"),
                   _RCP.draft_from_workspace(_wsp, "pg3", name="pb-pages")):
        _gp = _fd(_draft.graph_doc)[0]
        _dlg = _AU.RecipeDialog(win, draft=_draft, candidates=_RCP.candidates(_gp))
        app.processEvents()
        assert _dlg._target.count() > 0 and _dlg._target.currentData() == _draft.target, \
            (_draft.name, _dlg._target.currentData(), _draft.target)
        _d2, _pr = _dlg._collect()
        assert _d2 is not None and _d2.target == _draft.target, _pr
        _dlg.deleteLater()
        app.processEvents()
    _ok("PB1 the Publish-as-recipe dialog builds for a page's single graph and for a page "
        "that reads other pages (published flattened); its target menu holds the draft's "
        "target and the collected draft keeps it")

    # ── VW1–VW6: viewers as docks (V4.00 step 4) ─────────────────────────────────────
    from PySide6.QtCore import QEvent as _QEv, QPointF as _QPF, Qt as _QtV
    from PySide6.QtGui import QMouseEvent as _QME
    from nodelab_v2.ops import is_visual_output as _isvis
    from nodelab_v2.picker import request_for as _vrfor
    win.set_solo_frame(False)
    win._follow_act.setChecked(False)
    win.file_new()
    win.build_demo()
    app.processEvents()
    doc = win.doc
    _vv = doc.add_node("view.viewer", x=806, y=250)          # a light visual card, on n3
    doc.connect("n3", "out", _vv.id, "data")
    win.scene.sync()
    _vw_done: list = []
    win.runner.finished.connect(lambda nid, *a: _vw_done.append(nid))

    def _vw_wait(nid, timeout=180):
        rid = _rq(nid)
        t0 = time.time()
        while rid not in _vw_done and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.005)
        assert rid in _vw_done, (nid, _vw_done)
        for _ in range(3):
            app.processEvents()

    # VW1 a second viewer from View ▸ New: its own dock beside the first, empty and ACTIVE;
    # a pull goes to the active viewer and binds it; the first keeps its own result
    _va = win.viewer
    _vw_done.clear()
    win.pull_node("n3")
    _vw_wait("n3")
    _da = win._viewer_dock(_va)
    win._new_menu.aboutToShow.emit()
    _nact = [a for a in win._new_menu.actions() if a.text().endswith("Viewer")]
    assert _nact, [a.text() for a in win._new_menu.actions()]
    _nact[0].trigger()
    app.processEvents()
    _vb = win.viewer
    _db = win._viewer_dock(_vb)
    assert _vb is not _va and _db.objectName() != _da.objectName()
    assert not _db.isHidden() and _db.title_bar.is_active() and not _da.title_bar.is_active()
    assert not _vb.has_image() and _vb.binding is None
    _vw_done.clear()
    win.pull_node("n4")
    _vw_wait("n4")
    assert _vb.binding[1] == "n4" and _vb.has_image() and "n4" in _db.title_bar.title_text()
    assert _va.binding[1] == "n3" and "n3" in _da.title_bar.title_text()
    assert _va.showing()[0] == _rq("n3"), "the other viewer keeps its own result"
    assert win.scene.viewed_id == "n4", "the canvas marks the ACTIVE viewer's card"
    _ok("VW1 View ▸ New ▸ Viewer opens another Viewer dock beside the first, active and "
        "empty; a pull lands in the ACTIVE viewer and binds it (its title names the node); "
        "the first keeps its own result")

    # VW2 selecting a VISUAL card (a Viewer node) shows it in the active viewer even with
    # click-to-preview off; selecting an ordinary card does not
    assert _isvis("view.viewer") and _isvis("plot.xy") and not _isvis("enhance.gaussian")
    win._activate_viewer(_va)
    _vw_done.clear()
    win.scene.clearSelection()
    win.scene.node_items["n5"].setSelected(True)      # an ordinary card: nothing pulls
    for _ in range(5):
        app.processEvents()
    time.sleep(0.25)
    app.processEvents()
    assert _va.binding[1] == "n3" and not _vw_done, (_va.binding, _vw_done)
    win.scene.clearSelection()
    win.scene.node_items[_vv.id].setSelected(True)
    _vw_wait(_vv.id)
    assert _va.binding[1] == _vv.id and _va.showing()[0] == _rq(_vv.id), _va.binding
    assert _vb.binding[1] == "n4", "only the ACTIVE viewer follows the click"
    assert win.scene.viewed_id == _vv.id
    _ok("VW2 selecting a Viewer node shows it in the active viewer at once (click-to-preview "
        "off); selecting an ordinary node does not; the other viewer is left alone")

    # VW2b …so F8 on that card compares it beside what the viewer showed BEFORE the click,
    # not beside itself
    _vw_done.clear()
    win.pull_node("n3", viewer=_va)
    _vw_wait("n3")
    win.scene.clearSelection()
    _vw_done.clear()
    win.scene.node_items[_vv.id].setSelected(True)    # previews it into the active viewer
    _vw_wait(_vv.id)
    assert _va.binding[1] == _vv.id
    _vw_done.clear()
    win.compare_selected()                            # F8
    _vw_wait("n3")
    _vw_wait(_vv.id)
    assert _va.binding[1] == "n3", _va.binding
    assert win.viewer2 is not None and win.viewer2.binding[1] == _vv.id
    win.close_compare()
    app.processEvents()
    assert len(win.shell.docks_of("viewer")) == 2
    _ok("VW2b F8 on a card that selecting it just previewed compares it beside what the "
        "viewer showed before the click")

    # VW3 a press on a viewer's IMAGE makes it the active one: the canvas marks its card,
    # the spreadsheet shows its result, and a pick arms on IT
    _surf = _vb._pick_targets()[-1]
    QApplication.sendEvent(_surf, _QME(_QEv.MouseButtonPress, _QPF(12, 12), _QPF(12, 12),
                                       _QtV.LeftButton, _QtV.LeftButton, _QtV.NoModifier))
    QApplication.sendEvent(_surf, _QME(_QEv.MouseButtonRelease, _QPF(12, 12), _QPF(12, 12),
                                       _QtV.LeftButton, _QtV.NoButton, _QtV.NoModifier))
    app.processEvents()
    assert win.viewer is _vb and _db.title_bar.is_active() and not _da.title_bar.is_active()
    assert win.scene.viewed_id == "n4"
    assert win.sheet.node_id == "n4" and win.sheet.dataset is _vb.showing()[1]
    doc.add_node("enhance.normalize", node_id="VWN", x=1200, y=660)
    win.scene.sync()
    _vn = doc.nodes["VWN"].spec()
    win._arm_pick(_vrfor("VWN", _vn.input("low_pct"), _vn.input("high_pct")))
    assert _vb.picking() and not _va.picking(), "a pick arms on the ACTIVE viewer"
    _vb.cancel_pick()
    app.processEvents()
    # a press on a title bar does the same (it is how a floating viewer is picked up)
    _da.title_bar.mousePressEvent(_QME(_QEv.MouseButtonPress, _QPF(5, 5), _QPF(5, 5),
                                       _QtV.LeftButton, _QtV.LeftButton, _QtV.NoModifier))
    assert win.viewer is _va and win.scene.viewed_id == _va.binding[1]
    _ok("VW3 a press on a viewer's image or title bar makes it the active viewer: the "
        "canvas marks its card, the spreadsheet shows its result, a pick arms on it")

    # VW4 two viewers on ONE node at different cursors: each draws only its own frames
    _vw_done.clear()
    win.pull_node("n3", viewer=_va)
    _vw_wait("n3")
    _vw_done.clear()
    win.pull_node("n3", viewer=_vb)
    _vw_wait("n3")
    assert _va.binding == _vb.binding and _va._axes is not None and _va._axes.z >= 3

    def _vw_plane(v):
        return {c: np.array(a, copy=True) for c, a in (v._planes or {}).items()}

    def _vw_scrub(v, z):
        before = _vw_plane(v)
        v._sliders["z"].setValue(z)
        t0 = time.time()
        while time.time() - t0 < 60:
            app.processEvents()
            now = v._planes or {}
            if now and any(not np.array_equal(now[c], before.get(c)) for c in now):
                break
            time.sleep(0.005)
        for _ in range(3):
            app.processEvents()

    _vw_scrub(_va, 0)
    _vw_scrub(_vb, _vb._axes.z - 1)
    _pa, _pb = _vw_plane(_va), _vw_plane(_vb)
    assert _va._sliders["z"].value() == 0 and _vb._sliders["z"].value() == _vb._axes.z - 1
    assert any(not np.array_equal(_pa[c], _pb[c]) for c in _pa if c in _pb), \
        "the two viewers must show different planes"
    _vw_scrub(_vb, 1)
    assert all(np.array_equal(_va._planes[c], _pa[c]) for c in _pa), \
        "another viewer's frame of the same node must not land here"
    # a result RE-SERVED without planes (the runner busy elsewhere: they follow on the
    # decode lane) leaves the viewers already showing the node with their frames
    _pay = win.runner.finished_result("n3")
    assert _pay is not None
    win._on_run_finished(_rq("n3"), _pay, None, _va._axes, 0.0, None)
    for _ in range(5):
        app.processEvents()
    assert _va.has_image() and _vb.has_image(), "a re-served result must not blank them"
    assert "no image" not in _va._status.text() and "no image" not in _vb._status.text(), \
        (_va._status.text(), _vb._status.text())
    # under the troubleshooting scope (F9) another T is another PULL: two viewers of one
    # node at different T must not pull each other's frame back and forth
    assert _vb._sliders["t"].maximum() > 0, "the demo needs T > 1 for this check"
    _vb.set_cursor(t=1)
    win._on_view_request(viewer=_vb)
    t0 = time.time()
    while time.time() - t0 < 1.0:
        app.processEvents()
        time.sleep(0.01)
    _vw_starts: list = []
    win.runner.started.connect(lambda nid: _vw_starts.append(nid))
    win.set_solo_frame(True)
    t0 = time.time()
    while time.time() - t0 < 4.0:
        app.processEvents()
        time.sleep(0.01)
    assert len(_vw_starts) <= 2, f"a pull loop under F9: {len(_vw_starts)} pulls in 4 s"
    assert _va.has_image() and _vb.has_image()
    win.set_solo_frame(False)
    t0 = time.time()
    while win.runner.busy and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.01)
    _ok("VW4 two viewers bound to the same node at different z: each frame lands only in "
        "the viewer whose cursor asked for it; a result re-served without planes leaves "
        "both frames up; under F9, viewers of one node at different T do not pull each "
        "other's frame back and forth")

    # VW5 every viewer can be closed; the next pull opens a fresh one, on screen
    for _d in list(win.shell.docks_of("viewer")):
        _d.close()
    app.processEvents()
    assert not win.shell.docks_of("viewer") and win._active_viewer() is None
    assert win._viewed is None and not win._asker
    _vw_done.clear()
    win.pull_node("n3")
    _vw_wait("n3")
    _vds = win.shell.docks_of("viewer")
    assert len(_vds) == 1 and _vds[0].objectName() == "viewer:0", \
        [d.objectName() for d in _vds]
    assert not _vds[0].isHidden() and _vds[0].panel.has_image()
    _ok("VW5 the last viewer can be closed; the next pull opens a fresh viewer:0 on screen "
        "and shows the result in it")

    # VW6 a viewer pops out into its own window, keeps receiving, and docks back with its
    # picture (the CPU surface offscreen; the GL context rebuild has its own desktop probe)
    _vd = _vds[0]
    _vd.title_bar.float_btn.click()
    app.processEvents()
    assert _vd.isFloating() and _vd.panel.has_image()
    _vw_done.clear()
    win.pull_node("n4")
    _vw_wait("n4")
    assert _vd.panel.showing()[0] == _rq("n4") and _vd.isFloating()
    _vd.title_bar.float_btn.click()
    app.processEvents()
    assert not _vd.isFloating() and _vd.panel.has_image()
    _ok("VW6 a viewer pops out into its own window, keeps receiving results there, and "
        "docks back with its picture")

    # VW7 quitting while the canvas is maximized saves the DOCKED layout: the viewer comes
    # back on screen at a real height, not as the mini-map left its dock (hidden, empty)
    from nodelab_v2 import layout_store as _VLS
    from nodelab_v2.window import MainWindow as _VMW
    _vlay = os.path.join(tempfile.mkdtemp(prefix="nd2layout_vw_"), "layout.json")
    _venv = os.environ.get(_VLS.ENV_FILE)
    os.environ[_VLS.ENV_FILE] = _vlay
    try:
        _wq = _VMW(persist_layout=True)
        _wq.resize(1200, 760)
        _wq.show()
        _wq.build_demo()
        _qd: list = []
        _wq.runner.finished.connect(lambda nid, *a: _qd.append(nid))
        _wq.pull_node("n3")
        t0 = time.time()
        while _wq.runner.run_id("n3") not in _qd and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(3):
            app.processEvents()
        assert not _wq.shell.docks["viewer:0"].isHidden()
        _wq.set_maximized(True)
        app.processEvents()
        _wq.close()
        app.processEvents()
        _wr = _VMW(persist_layout=True)
        _wr.show()
        app.processEvents()
        _rd = _wr.shell.docks["viewer:0"]
        assert not _rd.isHidden() and _rd.height() > 100, (_rd.isHidden(), _rd.height())
        _wr.close()
        app.processEvents()
    finally:
        if _venv is None:
            os.environ.pop(_VLS.ENV_FILE, None)
        else:
            os.environ[_VLS.ENV_FILE] = _venv
    _ok("VW7 quitting with the canvas maximized saves the docked layout: the viewer comes "
        "back on screen at its height")

    # ── SH1–SH4: the dock shell (V4.00 step 3) ─────────────────────────────────────────
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QDockWidget as _QDW, QLabel as _SHL
    from nodelab_v2 import layout_store as _LS
    from nodelab_v2 import theme as _TH
    from nodelab_v2.shell import DOCK_GLYPH as _DOCKG, PanelDock as _PD, PanelSpec as _PS
    _sh = win.shell
    assert [n for n in _sh.docks if not n.startswith("viewer:")] == [
        "playback:0", "channels:0", "pages:0", "palette:0", "inspector:0", "sheet:0",
        "lablink:0", "console:0", "movie:0"], \
        list(_sh.docks)
    assert _sh.docks_of("viewer"), "the Viewers are shell docks too (V4.00 step 4)"
    _want = (_QDW.DockWidgetMovable | _QDW.DockWidgetFloatable | _QDW.DockWidgetClosable)
    for _d in _sh.docks.values():
        assert isinstance(_d, _PD) and _d.titleBarWidget() is _d.title_bar, _d.objectName()
        assert (_d.features() & _want) == _want, (_d.objectName(), _d.features())
    assert all(isinstance(d, _PD) for d in win.findChildren(_QDW)), \
        "every dock in the window is a shell panel"
    assert _sh.dock_of(win.inspector) is _sh.docks["inspector:0"]
    assert _sh.dock_of(win.movie_editor) is win._movie_dock is _sh.docks["movie:0"], \
        "the Movie Editor is found through its scroll area"
    assert win._console_dock is _sh.docks["console:0"]
    win._panels_menu.aboutToShow.emit()
    _pacts = win._panels_menu.actions()
    assert [a.text().split(" · ")[0] for a in _pacts
            if not a.text().startswith("Viewer")][:7] == [
        "Playback", "Channels", "Pages", "Nodes", "Properties", "Spreadsheet", "LabLink"], \
        [a.text() for a in _pacts]
    assert win._console_act in _pacts, "the Console entry is the Ctrl+` action itself"
    assert win._new_menu.menuAction().isVisible(), "View ▸ New lists the Viewer kind"
    # pop out and dock back from the title bar
    _sd = _sh.docks["sheet:0"]
    _sd.title_bar.float_btn.click()
    app.processEvents()
    assert _sd.isFloating() and _sd.title_bar.float_btn.text() == _DOCKG
    win.set_theme("light")                              # a floating panel restyles too
    app.processEvents()
    assert _TH.BODY.name() in _sd.title_bar.styleSheet()
    win.set_theme("dark")
    _sd.title_bar.float_btn.click()
    app.processEvents()
    assert not _sd.isFloating() and win.dockWidgetArea(_sd) == _Qt.RightDockWidgetArea
    # close from the title bar; View ▸ Panels brings it back
    _ld = _sh.docks["lablink:0"]
    _ld.title_bar.close_btn.click()
    app.processEvents()
    assert _ld.isHidden() and not _ld.toggleViewAction().isChecked()
    _ld.toggleViewAction().trigger()
    app.processEvents()
    assert not _ld.isHidden() and _ld.toggleViewAction().isChecked()
    # the Console's own action and its panel stay one thing
    win._console_act.setChecked(True)
    app.processEvents()
    assert not win._console_dock.isHidden()
    win._console_dock.title_bar.close_btn.click()
    app.processEvents()
    assert win._console_dock.isHidden() and not win._console_act.isChecked()
    # the welcome card's "Browse nodes" finds the palette's panel even when it is closed
    _sh.docks["palette:0"].close()
    win.focus_palette()
    app.processEvents()
    assert not _sh.docks["palette:0"].isHidden()
    _ok("SH1 dock shell: every side panel is a movable/floatable/closable shell dock named "
        "'<kind>:0' with its own title bar; pop-out and dock-back from the title bar (a "
        "floating panel restyles with the theme); ✕ hides and View ▸ Panels brings it back; "
        "the Console's Ctrl+` action is its Panels entry; Browse nodes reopens the palette")

    # the default layout (V4.00 step 11a): the Viewer is ON SCREEN from the start, above the
    # canvas at its share of the canvas column (menu bar to status bar); Nodes left;
    # Properties right with Spreadsheet and LabLink as its tabs; Console and Movie hidden
    from nodelab_v2.window import (INSPECTOR_W as _INSW, MainWindow as _MW1,
                                   PALETTE_W as _PALW, VIEWER_SHARE as _VSHARE)
    _v0d = _sh.docks["viewer:0"]
    assert not _v0d.isHidden() and win.dockWidgetArea(_v0d) == _Qt.TopDockWidgetArea
    _col = win.height() - win.menuBar().height() - win.statusBar().height()
    # the share — or the dock's MINIMUM height when the share is below it: this window is
    # 880 px tall and its viewer shows a picture, whose LUT controls raise the panel's
    # minimum (measured 2026-10-06: 401 px blank, 509 px with an image, at 1500 px wide;
    # 0.45 of this column is 374) — a dock cannot be made smaller than its minimum
    _want, _floor = int(_VSHARE * _col), _v0d.minimumSizeHint().height()
    assert abs(_v0d.height() - max(_want, _floor)) <= 2, (_v0d.height(), _want, _floor, _col)
    assert win.dockWidgetArea(_sh.docks["palette:0"]) == _Qt.LeftDockWidgetArea
    _insp = _sh.docks["inspector:0"]
    assert win.dockWidgetArea(_insp) == _Qt.RightDockWidgetArea
    assert {d.objectName() for d in win.tabifiedDockWidgets(_insp)} >= {"sheet:0", "lablink:0"}, \
        [d.objectName() for d in win.tabifiedDockWidgets(_insp)]
    # …and on a FRESH window large enough for every default size to be reachable (the
    # offscreen screen is 800×800, but a window that restores no layout is not clamped to
    # it): the blank Viewer at 0.40–0.50 of the column, Nodes at its width, Properties at
    # its width or its own minimum — the launch state a first install sees on a big monitor
    _wt = _MW1()
    _extra_t = [_wt]
    _wt.resize(1920, 1400)
    _wt.show()
    for _ in range(3):
        app.processEvents()
    _tv = _wt.shell.docks["viewer:0"]
    _tcol = _wt.height() - _wt.menuBar().height() - _wt.statusBar().height()
    assert not _tv.isHidden() and _wt.dockWidgetArea(_tv) == _Qt.TopDockWidgetArea
    assert not _tv.panel.has_image() and getattr(_tv, "_sized", False)
    # (step 11d) the Viewer dock is the image now — its controls sit under it, in panels
    assert 0.25 * _tcol <= _tv.height() <= 0.35 * _tcol, (_tv.height(), _tcol)
    _tpb = _wt.shell.docks["playback:0"]
    assert not _tpb.isHidden() and _tpb.geometry().top() >= _tv.geometry().bottom() - 2
    assert 0.40 * _tcol <= _tv.height() + _tpb.height() <= 0.55 * _tcol, \
        (_tv.height(), _tpb.height(), _tcol)
    _tp, _ti = _wt.shell.docks["palette:0"], _wt.shell.docks["inspector:0"]
    assert abs(_tp.width() - _PALW) <= 2, (_tp.width(), _PALW)
    assert abs(_ti.width() - max(_INSW, _ti.minimumSizeHint().width())) <= 2, \
        (_ti.width(), _INSW, _ti.minimumSizeHint().width())
    assert _wt.shell.docks["console:0"].isHidden() and _wt.shell.docks["movie:0"].isHidden()
    assert not _wt._layout_restored and _wt._first_show_done
    _wt.close()
    app.processEvents()
    _ok(f"SH1b default layout: the Viewer is on screen from launch, above the canvas, at "
        f"{_VSHARE:.2f} of the canvas column when that is reachable (a fresh 1920×1400 "
        f"window: {_tv.height() / _tcol:.2f}) and at its minimum height otherwise (this "
        f"window: {_v0d.height()} px, minimum {_floor}, column {_col}); Nodes left at "
        f"{_tp.width()} px; Properties right at {_ti.width()} px with Spreadsheet and LabLink "
        f"tabbed behind it; Console and Movie Editor hidden")

    # CH1 dock chrome (V4.00 step 11a): a floating panel is framed and filled from the theme,
    # not from Qt's default light palette (#efefef showed in the frame gutter and through the
    # panels' transparent margins); the panel bodies paint their QSS background; the app
    # palette follows the theme; the title-bar glyphs are in the title bar's font
    from PySide6.QtCore import QPoint as _QPt
    from PySide6.QtGui import QFontMetrics as _QFM, QPalette as _QPal
    from nodelab_v2.shell import FLOAT_GLYPH as _FLOATG
    from nodelab_v2.viewer import ViewerPanel as _VP
    _LIGHT = {"#efefef", "#f0f0f0", "#ffffff"}
    _DARKS = {_TH.BG.name(), _TH.BODY.name(), _TH.PANEL.name()}
    _sd = _sh.docks["sheet:0"]
    _sd.title_bar.float_btn.click()
    for _ in range(3):
        app.processEvents()
    assert _sd.isFloating()
    _img = _sd.grab().toImage()
    _w, _h = _img.width(), _img.height()
    for _x, _y in ((0, _h // 2), (_w - 1, _h // 2), (_w // 2, 0), (_w // 2, _h - 1)):
        assert _img.pixelColor(_x, _y).name() == _TH.BORDER.name(), \
            ("floating frame", (_x, _y), _img.pixelColor(_x, _y).name())
    for _x, _y in ((1, _h // 2), (_w // 2, _h - 2)):
        _c = _img.pixelColor(_x, _y).name()
        assert _c in _DARKS and _c != "#efefef", ("inside the frame", (_x, _y), _c)
    _sd.title_bar.float_btn.click()
    for _ in range(3):
        app.processEvents()
    assert not _sd.isFloating()
    _img = _sd.grab().toImage()
    _w, _h = _img.width(), _img.height()
    for _x, _y in ((0, _h // 2), (_w - 1, _h // 2), (_w // 2, 0), (_w // 2, _h - 1)):
        assert _img.pixelColor(_x, _y).name() not in _LIGHT, \
            ("docked edge", (_x, _y), _img.pixelColor(_x, _y).name())
    # the Viewer's body: its bottom margin and the gap above the controls are PAINTED —
    # both as the panel grabs itself and, the one that matters, as seen THROUGH its dock
    # (a widget's own grab paints its palette background as the root, so only the dock's
    # grab shows what the user saw: the dock behind the panel's transparent margins)
    _vp = _v0d.panel
    assert isinstance(_vp, _VP) and not _v0d.isFloating()
    _pi = _vp.grab().toImage()
    _pw, _ph = _pi.width(), _pi.height()
    _gap = _vp._hover_lbl.geometry().top() - 2      # (11d: the controls are panels now)
    assert _pi.pixelColor(2, _ph - 2).name() == _TH.PANEL.name(), _pi.pixelColor(2, _ph - 2).name()
    assert _pi.pixelColor(_pw // 2, _gap).name() == _TH.PANEL.name(), \
        _pi.pixelColor(_pw // 2, _gap).name()
    _di = _v0d.grab().toImage()
    _o = _vp.mapTo(_v0d, _QPt(0, 0))
    for _x, _y in ((_o.x() + 2, _o.y() + _ph - 2), (_o.x() + _pw // 2, _o.y() + _gap),
                   (_o.x() + 2, _o.y() + _ph // 2)):
        assert _di.pixelColor(_x, _y).name() == _TH.PANEL.name(), \
            ("viewer body through the dock", (_x, _y), _di.pixelColor(_x, _y).name())
    # the Movie Editor's scroll wrapper: its viewport is dark too
    _md = _sh.docks["movie:0"]
    _md.show()
    app.processEvents()
    _md.title_bar.float_btn.click()
    for _ in range(3):
        app.processEvents()
    assert _md.isFloating()
    _mo = _md.widget().viewport().mapTo(_md, _QPt(0, 0))
    _mi = _md.grab().toImage()
    _mc = _mi.pixelColor(_mo.x(), _mo.y()).name()
    assert _mc not in _LIGHT, ("movie viewport corner", (_mo.x(), _mo.y()), _mc)
    _md.title_bar.float_btn.click()
    app.processEvents()
    _md.hide()
    app.processEvents()
    assert _md.isHidden() and not _md.isFloating()
    # the application palette is the theme's, in both modes
    assert app.palette().color(_QPal.Window).name() == _TH.BG.name(), \
        app.palette().color(_QPal.Window).name()
    _dark_bg = _TH.BG.name()
    win.set_theme("light")
    app.processEvents()
    assert app.palette().color(_QPal.Window).name() == _TH.BG.name() != _dark_bg, \
        (app.palette().color(_QPal.Window).name(), _TH.BG.name())
    win.set_theme("dark")
    app.processEvents()
    assert app.palette().color(_QPal.Window).name() == _TH.BG.name() == _dark_bg
    # every glyph a title bar shows is in the title bar's font (Segoe UI Symbol fallback)
    _tb = _sh.docks["palette:0"].title_bar
    _fm = _QFM(_tb._glyph.font())
    for _g in sorted({_FLOATG, _DOCKG, "✕"} | {s.glyph for s in _sh.specs.values() if s.glyph}):
        assert _fm.inFontUcs4(ord(_g)), (_g, hex(ord(_g)), _tb._glyph.font().families())
    assert _tb.close_btn.isEnabled(), "a dock the window does not veto has a live ✕"
    _ok("CH1 dock chrome: a floating panel is framed 1px in the border colour and filled "
        "dark inside it (no #efefef), and shows no light edge once docked back; the Viewer's "
        "margins and the gap above its controls paint the panel colour, seen through the "
        "dock; the Movie Editor's scroll viewport is dark; the application palette follows "
        "the theme in both modes; every title-bar glyph is in the title bar's font; a "
        "non-vetoed dock's ✕ is enabled")

    # SH2 several instances of a kind: '+', the lowest free index, the active one, a veto
    _sh.register(_PS("probe_panel", "Probe", lambda: _SHL("probe"), glyph="◇", multi=True))
    assert win._new_menu.menuAction().isVisible(), "a multi kind appears under View ▸ New"
    _p0 = _sh.spawn("probe_panel")
    _p1 = _sh.spawn("probe_panel", beside=_p0)
    assert (_p0.objectName(), _p1.objectName()) == ("probe_panel:0", "probe_panel:1")
    assert _p1.title_bar.add_btn is not None and _sh.active("probe_panel") is _p1
    _p1.title_bar.add_btn.click()                       # '+' opens another beside it
    app.processEvents()
    assert [d.objectName() for d in _sh.docks_of("probe_panel")] == \
        ["probe_panel:0", "probe_panel:1", "probe_panel:2"]
    _sh.activate(_p0)
    assert _p0.title_bar.is_active() and not _p1.title_bar.is_active()
    app.processEvents()
    # …and the accent is PAINTED, not only flagged: the left edge of the active title bar
    # is the accent colour, the inactive one's is not
    _ia, _ib = _p0.title_bar.grab().toImage(), _p1.title_bar.grab().toImage()
    assert _ia.pixelColor(1, _ia.height() // 2).name() == _TH.ACCENT.name(), \
        _ia.pixelColor(1, _ia.height() // 2).name()
    assert _ib.pixelColor(1, _ib.height() // 2).name() != _TH.ACCENT.name()
    _sh._on_focus(None, _p1.panel)                      # focus moving in activates it
    assert _sh.active("probe_panel") is _p1 and _p1.title_bar.is_active()
    _p1.close()                                         # a multi panel's close destroys it
    app.processEvents()
    assert "probe_panel:1" not in _sh.docks and _sh.active("probe_panel") is _p0
    assert _sh.spawn("probe_panel").objectName() == "probe_panel:1", "lowest free index"
    _sh.allow_close = lambda d: d.kind != "probe_panel"  # e.g. "never the last canvas"
    _sh.sync_close_buttons()
    assert not _p0.title_bar.close_btn.isEnabled(), "a vetoed dock shows its ✕ disabled"
    assert _sh.docks["inspector:0"].title_bar.close_btn.isEnabled(), "…only the vetoed one"
    _p0.close()
    app.processEvents()
    assert "probe_panel:0" in _sh.docks and not _p0.isHidden(), "a vetoed close is refused"
    _p0.toggleViewAction().trigger()                    # the same, from View ▸ Panels
    app.processEvents()
    assert not _p0.isHidden() and _p0.toggleViewAction().isChecked(), \
        "a vetoed close from View ▸ Panels must leave its tick on"
    _sh.allow_close = lambda d: True
    _sh.sync_close_buttons()
    assert _p0.title_bar.close_btn.isEnabled(), "the ✕ is live again once the veto lifts"
    # '+' beside an instance that sits in a TAB GROUP joins the group, in front — Qt's
    # split of a tabbed dock would have taken both out of the group and off screen
    _sh.register(_PS("probe_tab", "ProbeTab", lambda: _SHL("tab"), multi=True,
                     tabify_with="inspector"))
    _t0 = _sh.spawn("probe_tab")
    win.tabifyDockWidget(_sh.docks["inspector:0"], _t0)
    _t0.raise_()
    app.processEvents()
    _t0.title_bar.add_btn.click()
    app.processEvents()
    _t1 = _sh.docks["probe_tab:1"]
    assert {d.objectName() for d in win.tabifiedDockWidgets(_t1)} >= {"inspector:0", "probe_tab:0"}, \
        [d.objectName() for d in win.tabifiedDockWidgets(_t1)]
    assert not _t1.visibleRegion().isEmpty(), "the new tab must be the one in front"
    _t0.raise_()
    app.processEvents()
    assert not _t0.visibleRegion().isEmpty(), "…and the old one is still there"
    for _d in list(_sh.docks_of("probe_tab")):
        _d.close()
    app.processEvents()
    _sh.docks["inspector:0"].raise_()
    for _d in list(_sh.docks_of("probe_panel")):
        _d.close()
    app.processEvents()
    assert not _sh.docks_of("probe_panel")
    _ok("SH2 multi-instance panels: View ▸ New lists a multi kind; '+' on a title bar opens "
        "another beside it — as a tab, in front, when it sits in a tab group; instances are "
        "'<kind>:<n>' with the lowest free n reused; focus makes one the active instance "
        "(its accent painted); closing destroys that instance and the window can veto a "
        "close, from the title bar or from View ▸ Panels (whose tick stays on)")

    # SH3/SH4 the layout survives a restart; Reset layout; a damaged file is harmless.
    # Two more windows on a TEMP layout file — the user's own is never read or written.
    from nodelab_v2.window import MainWindow as _MW
    _lay = os.path.join(tempfile.mkdtemp(prefix="nd2layout_"), "layout.json")
    _envs = {k: os.environ.get(k) for k in (_LS.ENV_FILE, _LS.ENV_ENABLED)}
    os.environ[_LS.ENV_FILE] = _lay
    _extra = []
    try:
        _w2 = _MW(persist_layout=True)
        _extra.append(_w2)
        # the Viewer is on screen from launch (V4.00 step 11a); hidden HERE so the window's
        # minimum height — what the restored-size check below is measured against — is the
        # canvas column's alone, and so a hidden Viewer is one more thing the restore has
        # to bring back as it was
        _w2.shell.docks["viewer:0"].hide()
        # a size that fits the screen: Qt clamps a RESTORED window to its screen (the
        # offscreen one is small), so only a dimension that fits can be compared exactly
        _avail = app.primaryScreen().availableGeometry()
        _minh = _w2.minimumSizeHint()
        _w2.show()
        app.processEvents()
        # a height between the window's minimum and the screen's: a fresh window opens at
        # its minimum, so only a RESTORE can bring this one back. Qt clamps a restored window
        # so that height + title bar fits the screen — the STYLE's title-bar height, which
        # can exceed the frame this platform reports — so that much room is left too.
        from PySide6.QtWidgets import QStyle as _QStyle
        _frame = max(0, _w2.frameGeometry().height() - _w2.height(),
                     _w2.style().pixelMetric(_QStyle.PM_TitleBarHeight))
        _th = min(_avail.height() - _frame - 4, _minh.height() + 30)
        assert _th > _minh.height(), ("no room on this screen to test a restored size",
                                      _minh, _avail, _frame)
        _w2.resize(max(_minh.width(), _avail.width() - 120), _th)
        app.processEvents()
        _s2 = _w2.shell
        assert not _s2.docks["sheet:0"].isFloating(), "no file yet: the default layout"
        _s2.docks["sheet:0"].setFloating(True)
        _s2.docks["console:0"].show()
        _s2.docks["lablink:0"].close()
        # a second Viewer, floating (V4.00 step 4): a multi-instance panel comes back too
        _s2.spawn("viewer").setFloating(True)
        app.processEvents()
        _saved_wh = (_w2.width(), _w2.height())
        _w2.close()
        app.processEvents()
        _rec = _LS.load_layout(_lay)
        assert _rec is not None and [d["name"] for d in _rec["docks"]] == list(_s2.docks), _rec
        _w3 = _MW(persist_layout=True)
        _extra.append(_w3)
        _w3.show()
        app.processEvents()
        _s3 = _w3.shell
        assert _s3.docks["sheet:0"].isFloating(), "a floating panel comes back floating"
        assert not _s3.docks["console:0"].isHidden(), "a shown panel comes back shown"
        assert _s3.docks["lablink:0"].isHidden(), "a closed panel stays closed"
        assert "viewer:1" in _s3.docks and _s3.docks["viewer:1"].isFloating(), \
            "a second, floating Viewer comes back as it was"
        assert _s3.docks["viewer:0"].isHidden(), "a Viewer hidden by hand stays hidden"
        assert _saved_wh[1] != _minh.height(), "the saved height is what a fresh window gets"
        assert _w3.height() == _saved_wh[1], (_w3.height(), _saved_wh, _avail)
        if _saved_wh[0] < _avail.width() - 2:
            assert _w3.width() == _saved_wh[0], (_w3.width(), _saved_wh, _avail)
        assert _w3._layout_restored, "the launcher is told a layout was restored"
        _w3.reset_layout()
        app.processEvents()
        assert not _s3.docks["sheet:0"].isFloating() and _s3.docks["console:0"].isHidden()
        assert not _s3.docks["lablink:0"].isHidden() and _s3.docks["movie:0"].isHidden()
        assert not _s3.docks["viewer:1"].isFloating() and _s3.docks["viewer:1"].isHidden()
        assert _w3.dockWidgetArea(_s3.docks["palette:0"]) == _Qt.LeftDockWidgetArea
        assert _w3.tabifiedDockWidgets(_s3.docks["inspector:0"]), "Properties tabs are back"
        _w3.close()
        app.processEvents()
        with open(_lay, "w", encoding="utf-8") as _f:
            _f.write("{ damaged")
        _w4 = _MW(persist_layout=True)                  # must open, on the default layout
        _extra.append(_w4)
        _w4.show()
        app.processEvents()
        assert not _w4.shell.docks["sheet:0"].isFloating()
        assert _w4.shell.docks["console:0"].isHidden()
        assert not _w4._layout_restored
        with open(_lay + ".rejected", encoding="utf-8") as _f:
            assert _f.read() == "{ damaged", "the unusable file is set aside, intact"
        _w4.close()
        app.processEvents()
        assert _LS.load_layout(_lay) is not None, "closing rewrites a good layout over it"
    finally:
        for _k, _v in _envs.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
    assert not win._persist_layout, "the probe's own window runs with NODELAB_LAYOUT=0"
    _ok("SH3 layout memory: a floating, a shown and a closed panel, a second floating "
        "Viewer, and the window size come back after a restart; View ▸ Reset layout docks "
        "everything back with the "
        "default tabs and hidden panels; SH4 a damaged layout file opens on the default "
        "layout, is kept aside as layout.json.rejected and replaced on close; the probe's own "
        "window never reads or writes one")

    # ── SW1 / LD1 / LD2 / SW3 / RF1 / EX1: the standard workflow (V4.00 step 11) ─────────
    from PySide6.QtCore import QPointF as _QPF11
    from PySide6.QtWidgets import QToolButton as _QTB11
    import tifffile as _tiff11
    from nodelab_v2 import readiness as _RD11
    from nodelab_v2.workspace import qualify as _q11, standard_kinds as _skinds11
    win.set_solo_frame(False)
    win._follow_act.setChecked(False)
    win.file_new()
    app.processEvents()
    _ws11 = win.workspace
    _kinds11 = list(_skinds11())
    assert [p.kind for p in _ws11.pages.values()] == _kinds11 and _kinds11[0] == "input", \
        [(p.id, p.kind) for p in _ws11.pages.values()]
    assert _ws11.page(_ws11.active).kind == "input" and win.canvas.page_id == _ws11.active
    assert "Image Input" in win.windowTitle(), win.windowTitle()
    _tree11 = win.palette._tree
    assert _tree11.topLevelItem(0).text(0) == "Pages", _tree11.topLevelItem(0).text(0)
    _band11 = _tree11.topLevelItem(0).child(0)
    assert [_band11.child(i).text(1) for i in range(_band11.childCount())] == ["Page Output"]
    assert "Image Input" in win.palette._kind_chip.text(), win.palette._kind_chip.text()
    _ok("SW1 File ▸ New: the four standard pages in order, Image Input active and in the "
        "title; the palette leads with the Pages band (Page Output alone on an Input page)")

    # LD1 a load while the Analysis page is active: the card lands on Image Input (the
    # canvas switches there), published as a Page Output named after the file, previewed
    _pg_in11 = _ws11.active
    _pg_an11 = next(p.id for p in _ws11.pages.values() if p.kind == "analyze")
    win._show_page(win._main_canvas, _pg_an11)
    app.processEvents()
    assert _ws11.active == _pg_an11
    _ld11 = tempfile.mkdtemp(prefix="nd2ld_")
    _ldp11 = os.path.join(_ld11, "blobs.tif")
    _tiff11.imwrite(_ldp11, np.random.default_rng(11).integers(0, 4000, size=(3, 40, 44))
                    .astype(np.uint16), metadata={"axes": "ZYX"})
    _done11: list = []
    win.runner.finished.connect(lambda nid, *a: _done11.append(nid))
    win._load_source_paths([_ldp11])
    app.processEvents()
    assert _ws11.active == _pg_in11 and win.canvas.page_id == _pg_in11, "switched to Image Input"
    _loads11 = [r for r in win.doc.nodes.values() if r.op_key == "io.load"]
    _outs11 = [r for r in win.doc.nodes.values() if r.op_key == "page.output"]
    assert len(_loads11) == 1 and len(_outs11) == 1, (len(_loads11), len(_outs11))
    assert _outs11[0].params.get("name") == "blobs", _outs11[0].params
    assert (_loads11[0].id, "image", _outs11[0].id, "data") in set(win.doc.edges), win.doc.edges
    assert _loads11[0].modes.get("access") == "ingest", "a TIFF starts on ingest"
    # (the status line said "published as “blobs”" until the preview pull replaced it)
    # the load pulls the card: a TIFF ingests first, then the run binds the Viewer (File ▸
    # New left the old picture on the surface, so has_image() alone proves nothing)
    _rid11 = _q11(_pg_in11, _loads11[0].id)
    _t011 = time.time()
    while _rid11 not in _done11 and time.time() - _t011 < 240:
        app.processEvents()
        time.sleep(0.005)
    assert _rid11 in _done11, ("the load did not preview its card", _done11[-3:])
    for _ in range(3):
        app.processEvents()
    assert win.viewer.has_image() and win.viewer.binding == (_pg_in11, _loads11[0].id), \
        (win.viewer.has_image(), win.viewer.binding)
    win._load_source_paths([_ldp11])
    app.processEvents()
    assert [n for n, _ in _ws11.outputs_of(_pg_in11)] == ["blobs", "blobs2"]
    assert all(r.id in win.scene.node_items for r in win.doc.nodes.values())
    _ok("LD1 a load while Analysis is active lands on Image Input (the canvas switches), is "
        "published as a Page Output named after the file (blobs, then blobs2), starts a TIFF "
        "on ingest and shows in the Viewer")

    # LD2 a desktop drop: onto the Analysis canvas → Image Input, published; onto a Batch
    # point → stays on that page, nothing published
    win._show_page(win._main_canvas, _pg_an11)
    app.processEvents()
    win._on_files_dropped([_ldp11], _QPF11(100.0, 100.0), "")
    app.processEvents()
    assert _ws11.active == _pg_in11, "a plain drop lands on Image Input"
    assert [n for n, _ in _ws11.outputs_of(_pg_in11)] == ["blobs", "blobs2", "blobs3"]
    _bt11 = win.doc.add_node("util.batch", x=700, y=400)
    _n_out11 = len(_ws11.outputs_of(_pg_in11))
    win._on_files_dropped([_ldp11], _QPF11(700.0, 600.0), _bt11.id)
    app.processEvents()
    assert len(_ws11.outputs_of(_pg_in11)) == _n_out11, "a drop on a Batch point publishes nothing"
    assert any(e[2] == _bt11.id and e[3] == "data" for e in win.doc.edges)
    win.doc.remove_node(_bt11.id)
    app.processEvents()
    _ok("LD2 a desktop drop on another page's canvas lands on Image Input and is published; "
        "a drop on a Batch point stays with the point and publishes nothing")

    # SW3 New page ▸ Refinement arrives with a Page Input already reading the newest Output
    _pg_r11 = win.new_page("refine")
    app.processEvents()
    _ins11 = [r for r in win.doc.nodes.values() if r.op_key == "page.input"]
    assert _ws11.active == _pg_r11 and len(_ins11) == 1, _ins11
    assert _ins11[0].params.get("source") == f"{_pg_in11}:blobs3", _ins11[0].params
    assert win.doc.env(_ins11[0].id).axes is not None, "the envelope crosses"
    _inname11 = _ws11.page(_pg_in11).name
    assert win.scene.node_items[_ins11[0].id]._page_boundary_label() == \
        f"Input · {_inname11} · blobs3"
    assert win.welcome.isVisible(), "the card stays while the page holds only its seeded Input"
    assert f"reads {_inname11} · blobs3" in win.statusBar().currentMessage(), \
        win.statusBar().currentMessage()
    _band11 = win.palette._tree.topLevelItem(0)
    assert _band11.text(0) == "Pages" and \
        [_band11.child(0).child(i).text(1) for i in range(_band11.child(0).childCount())] == \
        ["Page Input", "Page Output"]
    _ok("SW3 New page ▸ Refinement starts with a Page Input bound to the newest upstream "
        "Output (card: Input · page · name, envelope across, status says so); the Pages "
        "band offers both boundary nodes")

    # RF1 readiness one-click fixes: a terminal node's hint appends a wired Page Output; an
    # unbound Input offers Bind buttons (default first) and Go to <page>; a loader nothing
    # publishes gets the same hint
    _pin11 = _ins11[0]
    _g11 = win.doc.add_node("enhance.gaussian", x=320, y=120)
    win.doc.connect(_pin11.id, "out", _g11.id, "data")
    win.scene.sync()
    _pr11 = _RD11.problems(win.doc, _g11.id)
    assert [(p.kind, p.severity) for p in _pr11] == [("unpublished", "hint")], _pr11
    assert _RD11.ready(win.doc, _g11.id), "a hint never blocks"
    win.scene.clearSelection()
    win.scene.node_items[_g11.id].setSelected(True)
    app.processEvents()
    win.inspector.rebuild()
    app.processEvents()
    _btn11 = next((b for b in win.inspector.findChildren(_QTB11)
                   if b.text() == "+ Page Output"), None)
    assert _btn11 is not None, "the hint offers + Page Output"
    _btn11.click()
    app.processEvents()
    _nouts11 = [r for r in win.doc.nodes.values() if r.op_key == "page.output"]
    assert len(_nouts11) == 1 and _nouts11[0].params.get("name") == "out", _nouts11
    assert (_g11.id, "out", _nouts11[0].id, "data") in set(win.doc.edges)
    assert _RD11.problems(win.doc, _g11.id) == []
    win.doc.nodes[_pin11.id].params["source"] = ""
    win.doc.touch(_pin11.id)
    assert win.scene.node_items[_pin11.id]._page_boundary_label() == "Input · (unbound)"
    assert win.scene.node_items[_pin11.id]._boundary_broken()
    win.scene.clearSelection()
    win.scene.node_items[_pin11.id].setSelected(True)
    app.processEvents()
    win.inspector.rebuild()
    app.processEvents()
    _bind11 = [b for b in win.inspector.findChildren(_QTB11) if b.text().startswith("Bind to ")]
    assert _bind11 and _bind11[0].text() == f"Bind to {_inname11} · blobs3", \
        [b.text() for b in _bind11]
    _goto11 = [b for b in win.inspector.findChildren(_QTB11) if b.text().startswith("Go to ")]
    assert _goto11 and _goto11[0].text() == f"Go to {_inname11}", [b.text() for b in _goto11]
    _bind11[0].click()
    app.processEvents()
    assert win.doc.nodes[_pin11.id].params.get("source") == f"{_pg_in11}:blobs3"
    assert not win.scene.node_items[_pin11.id]._boundary_broken()
    win.inspector.rebuild()
    app.processEvents()
    _goto11 = [b for b in win.inspector.findChildren(_QTB11) if b.text().startswith("Go to ")]
    assert _goto11 and _goto11[0].text() == f"Go to {_inname11}", "a bound Input links its page"
    _goto11[0].click()
    app.processEvents()
    assert _ws11.active == _pg_in11 and win.canvas.page_id == _pg_in11, "Go to shows the page"
    _lone11 = [r for r in win.doc.nodes.values() if r.op_key == "io.load"
               and not any(e[0] == r.id for e in win.doc.edges)]
    assert _lone11, "the Batch-drop loader is still unpublished"
    _ph11 = _RD11.problems(win.doc, _lone11[0].id)
    assert [(p.kind, p.severity, p.suggestions[0].wire_to) for p in _ph11] == \
        [("unpublished", "hint", "image")], _ph11
    _nid11 = win._on_append_requested(_lone11[0].id, "page.output", "image")
    app.processEvents()
    assert _nid11 and (_lone11[0].id, "image", _nid11, "data") in set(win.doc.edges)
    assert _RD11.problems(win.doc, _lone11[0].id) == []
    _ok("RF1 one-click fixes: a terminal node's hint appends a wired Page Output; an unbound "
        "Input offers Bind (default first) and Go to <page>, and both act; a loader nothing "
        "publishes gets the same hint and its Output is wired from `image`")

    # EX1 the welcome card's Example graph: the four-page example, the Analysis Viewer pulls
    win.build_example_workspace()
    app.processEvents()
    assert [p.kind for p in _ws11.pages.values()] == _kinds11
    assert _ws11.page(_ws11.active).kind == "input" and win.canvas.page_id == _ws11.active
    _ex_an11 = next(p for p in _ws11.pages.values() if p.kind == "analyze")
    assert {r.op_key for r in _ex_an11.doc.nodes.values()} == \
        {"page.input", "plot.xy", "view.viewer"}, sorted(_ex_an11.doc.nodes)
    for _pg11 in _ws11.pages.values():
        for _r11 in _pg11.doc.nodes.values():
            _errs11 = [p for p in _RD11.problems(_pg11.doc, _r11.id) if p.severity == "error"]
            assert not _errs11, (_pg11.name, _r11.id, [(p.kind, p.message) for p in _errs11])
    _done11.clear()
    win.pull_node("view", page_id=_ex_an11.id)
    _t011 = time.time()
    while _q11(_ex_an11.id, "view") not in _done11 and time.time() - _t011 < 240:
        app.processEvents()
        time.sleep(0.005)
    assert _q11(_ex_an11.id, "view") in _done11, _done11
    assert win.viewer.has_image() and win.viewer.binding == (_ex_an11.id, "view"), \
        win.viewer.binding
    _ok("EX1 the welcome card's Example graph spans the four standard pages with every "
        "boundary named and bound (no error-grade readiness problem); the Analysis page's "
        "Viewer pulls through the whole chain")

    # ── NP1–NP4: New page…, page recipes, masters (V4.00 step 11, part B) ──────────────
    from PySide6.QtCore import QTimer as _QT11
    from PySide6.QtWidgets import QMenu as _QM11
    from nodelab_v2 import page_recipes as _PR11
    from nodelab_v2.new_page_dialog import NewPageDialog as _NPD11
    _rdir11 = tempfile.mkdtemp(prefix="nd2recipes_")
    os.environ[_PR11.ENV_DIR] = _rdir11
    os.environ.pop(_PR11.ENV_ENABLED, None)
    # EX1 left the example workspace: four pages, Image Input active
    _in11 = next(p.id for p in _ws11.pages.values() if p.kind == "input")
    _ref11 = next(p.id for p in _ws11.pages.values() if p.kind == "refine")
    # NP1 the dialog lists recipes, masters and sources; _apply_new_page makes each start
    _dlg = _NPD11(win, _ws11, kind="process")
    assert _dlg._recipe.count() >= 2 and all("(built-in)" in _dlg._recipe.itemText(i)
                                             for i in range(_dlg._recipe.count()))
    assert _dlg._source.count() >= 2 and _dlg._source.currentData() == f"{_ref11}:mask", \
        [_dlg._source.itemData(i) for i in range(_dlg._source.count())]
    assert _dlg._master.count() == 4, "every plain page is offered as a master"
    assert win.set_master_page(_ref11, True)
    app.processEvents()
    _dlg2 = _NPD11(win, _ws11, kind="process")
    assert _dlg2._master.itemText(0).startswith("★ ") and _dlg2._master.itemData(0) == _ref11
    _dlg2._r_recipe.setChecked(True)
    app.processEvents()
    _spec = _dlg2.spec()
    assert _spec.start == "recipe" and _spec.recipe is not None and \
        _spec.source == f"{_ref11}:mask", _spec
    _n0 = len(_ws11.pages)
    _pid_r = win._apply_new_page(_spec, canvas=win._main_canvas)
    assert _pid_r and len(_ws11.pages) == _n0 + 1 and _ws11.active == _pid_r
    assert {r.op_key for r in win.doc.nodes.values()} == \
        {"page.input", "analysis.label", "analysis.measure", "page.output"}, sorted(win.doc.nodes)
    assert win.doc.nodes["in"].params["source"] == f"{_ref11}:mask" and not win.welcome.isVisible()
    assert all(nid in win.scene.node_items for nid in win.doc.nodes), "cards on the canvas"
    _dlg3 = _NPD11(win, _ws11, kind="process", start="linked", master=_ref11)
    assert _dlg3._r_linked.isChecked() and not _dlg3._kind.isEnabled()
    assert _dlg3._kind.currentData() == "refine", "a linked page takes its master's kind"
    _spec3 = _dlg3.spec()
    assert (_spec3.start, _spec3.master, _spec3.kind) == ("linked", _ref11, "refine"), _spec3
    _pid_l = win._apply_new_page(_spec3, canvas=win._main_canvas)
    assert _pid_l and _ws11.pages[_pid_l].master == _ref11
    assert "linked · 0 overrides" in win._main_canvas.view.page_button.text(), \
        win._main_canvas.view.page_button.text()
    _pid_e = win._apply_new_page(_PR11.NewPageSpec(kind="analyze", name="Plots",
                                                   source=f"{_pid_r}:cells"),
                                 canvas=win._main_canvas)
    assert _ws11.pages[_pid_e].name == "Plots"
    assert [r.params.get("source") for r in win.doc.nodes.values()] == [f"{_pid_r}:cells"]
    for _d in (_dlg, _dlg2, _dlg3):
        _d.deleteLater()
    app.processEvents()
    _ok("NP1 New page… lists the kind's page recipes (built-in marked), the masters (★ first) "
        "and the Outputs a new page could read (the nearest preselected); applying the spec "
        "makes a page from a recipe (bound, cards on the canvas), a linked page (its master's "
        "kind, the switcher says linked · 0 overrides) and an empty page with a bound Input")

    # NP2 the switcher menu: New page…, Set as master page (★ on the row and the button),
    # Save as page recipe…; a linked page cannot be set as master
    win._show_page(win._main_canvas, _ref11)
    app.processEvents()
    _m11 = _QM11()
    win.fill_page_menu(_m11, win._main_canvas)
    _texts11 = [a.text() for a in _m11.actions()]
    for _want in ("New page…", "Set as master page", "Save as page recipe…"):
        assert _want in _texts11, (_want, _texts11)
    _mst = next(a for a in _m11.actions() if a.text() == "Set as master page")
    assert _mst.isCheckable() and _mst.isChecked() and _mst.isEnabled()
    assert any(t.startswith("★ ") and _ws11.pages[_ref11].name in t for t in _texts11), _texts11
    assert win._main_canvas.view.page_button.text().strip().startswith("★ ")
    _mst.trigger()
    app.processEvents()
    assert not _ws11.pages[_ref11].is_master
    assert not win._main_canvas.view.page_button.text().strip().startswith("★")
    win._show_page(win._main_canvas, _pid_l)
    app.processEvents()
    _m12 = _QM11()
    win.fill_page_menu(_m12, win._main_canvas)
    assert not next(a for a in _m12.actions() if a.text() == "Set as master page").isEnabled()
    _ok("NP2 the page switcher offers New page…, Set as master page (checkable; ★ on the "
        "row and on the button, both gone when unset; disabled on a linked page) and Save as "
        "page recipe…")

    # NP3 a Page Output's context menu: New page from this output… (disabled while unnamed);
    # accepted, the dialog makes a page of the next kind reading that Output
    win._show_page(win._main_canvas, _in11)
    app.processEvents()
    _out11 = next(r for r in win.doc.nodes.values() if r.op_key == "page.output")
    _m13 = _QM11()
    win.scene._fill_node_menu(_m13, win.scene.node_items[_out11.id])
    _npo = next(a for a in _m13.actions() if a.text() == "New page from this output…")
    assert _npo.isEnabled()
    _un11 = win.doc.add_node("page.output", params={"name": ""})
    win.scene.sync()
    app.processEvents()
    _m14 = _QM11()
    win.scene._fill_node_menu(_m14, win.scene.node_items[_un11.id])
    assert not next(a for a in _m14.actions()
                    if a.text() == "New page from this output…").isEnabled()
    win.doc.remove_node(_un11.id)
    app.processEvents()
    _n1 = len(_ws11.pages)

    _seen11: list = []

    def _accept11(want_recipe=False, into=None, recipe=None):
        for _w in QApplication.topLevelWidgets():
            if isinstance(_w, _NPD11) and _w.isVisible():
                _seen11.append((_w._r_recipe.isChecked(), _w._into))
                if want_recipe:
                    for _i in range(_w._recipe.count()):
                        if _w._recipe.itemData(_i).name == recipe:
                            _w._recipe.setCurrentIndex(_i)
                else:
                    _w._r_empty.setChecked(True)
                _w.accept()
                return
        _QT11.singleShot(50, lambda: _accept11(want_recipe, into, recipe))
    _QT11.singleShot(50, lambda: _accept11(False))
    _npo.trigger()                              # modal: the timer accepts it
    app.processEvents()
    assert len(_ws11.pages) == _n1 + 1, "the dialog added a page"
    _newp = _ws11.pages[_ws11.active]
    assert _newp.kind == "refine", _newp.kind
    assert [r.params.get("source") for r in _newp.doc.nodes.values()] == \
        [f"{_in11}:{_out11.params['name']}"]
    _ok("NP3 a named Page Output's context menu offers New page from this output… (disabled "
        "while unnamed); accepted, a page of the next kind reads that Output")

    # NP4 an empty Refinement page with something upstream: the welcome card offers a page
    # recipe and the recipe takes the seeded page over; Save as page recipe writes under the
    # redirected folder; the Input page's card still says Load image…
    _pid_n = win.new_page("refine")
    app.processEvents()
    assert win.welcome.isVisible() and win.welcome.is_banner, "a seeded page keeps its card"
    assert win.welcome.button("More recipes…") is not None, win.welcome.button_texts()
    _n2 = len(_ws11.pages)
    _seen11.clear()
    _QT11.singleShot(50, lambda: _accept11(True, _pid_n, "Smooth & threshold"))
    win.welcome.button("More recipes…").click()
    app.processEvents()
    assert _seen11 == [(True, _pid_n)], ("the card opens the dialog on Page recipe, into "
                                          "this page", _seen11)
    assert len(_ws11.pages) == _n2 and _ws11.active == _pid_n, "the recipe took the page over"
    assert {r.op_key for r in win.doc.nodes.values()} >= \
        {"page.input", "enhance.gaussian", "analysis.threshold", "page.output"}, sorted(win.doc.nodes)
    assert not win.welcome.isVisible()
    _path11 = win.save_page_as_recipe(_pid_n, name="Probe smooth", description="from the probe")
    assert _path11 and os.path.isfile(_path11) and \
        os.path.abspath(_path11).startswith(os.path.abspath(_rdir11)), _path11
    assert any(r.name == "Probe smooth" and not r.builtin for r in _PR11.list_recipes("refine"))
    win.file_new()
    app.processEvents()
    assert win.welcome.isVisible() and win.welcome._btn_load.text() == "Load image…"
    os.environ.pop(_PR11.ENV_DIR, None)
    _ok("NP4 the start card of a seeded Refinement page offers More recipes… (New page… on "
        "that page), "
        "and the recipe takes the page over; Save as page recipe writes under the recipe "
        "folder and lists as the user's; the Input page's card still loads an image")

    # ── WC1 / PT1 / PP1: the start card per kind, the page tabs, the Pages panel ─────────
    # (V4.00 step 11, after the user's feedback: the card could not be dismissed and asked
    # the same thing on every page; pages were hard to keep track of)
    from PySide6.QtCore import Qt as _Qt12
    _ws12 = win.workspace
    os.environ[_PR11.ENV_DIR] = tempfile.mkdtemp(prefix="nd2recipes_")   # built-ins only
    win.file_new()
    app.processEvents()
    _in12 = _ws12.active
    assert _ws12.page(_in12).kind == "input"
    assert win.welcome.isVisible() and not win.welcome.is_banner
    assert win.welcome.button_texts() == ["Load image…", "Load sequence…", "Example graph"], \
        win.welcome.button_texts()
    win.welcome._close.click()
    app.processEvents()
    assert not win.welcome.isVisible() and _in12 in win._welcome_dismissed
    win._sync_welcome()
    app.processEvents()
    assert not win.welcome.isVisible(), "a dismissed card stays hidden on its page"
    # nothing upstream yet: a later page's banner says so and offers Image Input
    _ref12 = next(p.id for p in _ws12.pages.values() if p.kind == "refine")
    win._show_page(win._main_canvas, _ref12)
    app.processEvents()
    assert not win.doc.nodes, "nothing to read: no Page Input is seeded"
    assert win.welcome.isVisible() and win.welcome.is_banner
    assert win.welcome.button_texts()[0] == "Go to Image Input", win.welcome.button_texts()
    win.welcome.button("Go to Image Input").click()
    app.processEvents()
    assert _ws12.active == _in12 and win.canvas.page_id == _in12
    # something to read: the banner sits at the bottom, below the seeded Page Input, names
    # what it reads and offers the kind's page recipes; a recipe button fills the page
    _L12 = win.doc.add_node("io.load", x=0, y=0)
    _O12 = win.doc.add_node("page.output", x=300, y=0, params={"name": "raw"})
    win.doc.connect(_L12.id, "image", _O12.id, "data")
    app.processEvents()
    win._show_page(win._main_canvas, _ref12)
    app.processEvents()
    assert [r.op_key for r in win.doc.nodes.values()] == ["page.input"], \
        "the page is given its Page Input when it has something to read"
    assert win.welcome.isVisible() and win.welcome.is_banner
    _wg12 = win.welcome.geometry()
    assert _wg12.bottom() >= win.view.height() - 40, (_wg12, win.view.height())
    _seed12 = next(iter(win.doc.nodes.values()))
    _sr12 = win.view.mapFromScene(
        win.scene.node_items[_seed12.id].sceneBoundingRect()).boundingRect()
    assert win.view.viewport().rect().contains(_sr12.center()), "the seeded Input is in view"
    assert not _wg12.intersects(_sr12.translated(win.view.viewport().pos())), \
        ("the banner does not cover the seeded Input", _wg12, _sr12)
    assert f"{_ws12.page(_in12).name} · raw" in win.welcome._sub.text(), win.welcome._sub.text()
    _bt12 = win.welcome.button_texts()
    assert _bt12[:2] == ["Background & deconvolve", "Smooth & threshold"], _bt12
    assert "More recipes…" in _bt12 and _bt12[-1] == "Start empty", _bt12
    assert win.welcome._compact or "Link to a master…" in _bt12, _bt12   # one row when short
    # the seeded Input is given ONCE: deleted, it does not come back on the next click
    win.doc.remove_node(_seed12.id)
    app.processEvents()
    win._activate_canvas(win._main_canvas)
    app.processEvents()
    assert not win.doc.nodes, "a deleted seed does not come back"
    win.welcome.button("Smooth & threshold").click()
    app.processEvents()
    assert {r.op_key for r in win.doc.nodes.values()} == \
        {"page.input", "enhance.gaussian", "analysis.threshold", "page.output"}, \
        sorted(r.op_key for r in win.doc.nodes.values())
    assert win.doc.nodes["in"].params["source"] == f"{_in12}:raw"
    assert not win.welcome.isVisible()
    _vp12 = win.view.viewport().rect()
    assert all(_vp12.intersects(win.view.mapFromScene(it.sceneBoundingRect()).boundingRect())
               for it in win.scene.node_items.values()), "the recipe's cards are in view"
    # Start empty dismisses; File ▸ New brings every card back
    _pro12 = next(p.id for p in _ws12.pages.values() if p.kind == "process")
    win._show_page(win._main_canvas, _pro12)
    app.processEvents()
    assert win.welcome.is_banner and "Image Refinement · mask" in win.welcome._sub.text(), \
        win.welcome._sub.text()
    win.welcome.button("Start empty").click()
    app.processEvents()
    assert not win.welcome.isVisible() and _pro12 in win._welcome_dismissed
    win.file_new()
    app.processEvents()
    assert win.welcome.isVisible() and not win._welcome_dismissed and not win._seeded_pages
    _ok("WC1 the start card asks for what each page needs: a load on Image Input, the kind's "
        "page recipes on a later page (a banner along the bottom, clear of the seeded Page "
        "Input, naming what it reads), Go to Image Input with nothing to read; ✕ and Start "
        "empty dismiss it for the page and File ▸ New brings it back; a recipe button fills "
        "the page in view; a deleted seed does not come back")

    # PT1 the page tabs (two rows since step 11d): a tab per page KIND in pipeline order, and
    # under it the pages of the kind shown; a kind tab shows its page, a sub-tab drag reorders
    # those pages, a rename shows; a new page lands in pipeline order; + presets the kind
    _tabs12 = win._main_canvas.tabs
    _bar12, _kb12 = _tabs12.bar, _tabs12.kind_bar
    _kind12 = lambda p: _ws12.pages[p].kind                                  # noqa: E731
    assert _tabs12.kinds() == ["input", "refine", "process", "analyze"], _tabs12.kinds()
    assert _kb12.tabData(_kb12.currentIndex()) == _kind12(_ws12.active)
    assert _tabs12.page_ids() == [p for p in _ws12.pages if _kind12(p) == _kind12(_ws12.active)]
    assert _bar12.tabData(_bar12.currentIndex()) == _ws12.active
    _ana12 = next(p.id for p in _ws12.pages.values() if p.kind == "analyze")
    _ref12 = next(p.id for p in _ws12.pages.values() if p.kind == "refine")
    _kb12.setCurrentIndex(_tabs12.kinds().index("analyze"))
    for _ in range(3):
        app.processEvents()
    assert _ws12.active == _ana12 and win.canvas.page_id == _ana12
    assert _tabs12.page_ids() == [_ana12] and _tabs12.kind == "analyze"
    _np12 = win.new_page("refine")
    app.processEvents()
    assert list(_ws12.pages).index(_np12) == 2
    assert _tabs12.kind == "refine" and _tabs12.page_ids() == [_ref12, _np12]
    assert _bar12.tabData(_bar12.currentIndex()) == _np12
    assert _kb12.tabText(_tabs12.kinds().index("refine")).endswith("2"), \
        "a kind holding several pages says how many"
    _order12 = list(_ws12.pages)
    _bar12.moveTab(1, 0)
    app.processEvents()
    assert list(_ws12.pages)[1:3] == [_np12, _ref12] and _tabs12.page_ids() == [_np12, _ref12]
    _bar12.moveTab(0, 1)
    app.processEvents()
    assert list(_ws12.pages) == _order12
    win.rename_page(_np12, "Dish B refine")
    app.processEvents()
    assert _bar12.tabText(_tabs12.page_ids().index(_np12)) == "Dish B refine"
    assert win._main_canvas.tabs.add_btn.isVisible()
    assert "Image Refinement page" in _bar12.tabToolTip(0)
    _asked12: list = []
    _npd12 = win.new_page_dialog
    win.new_page_dialog = lambda **k: _asked12.append(k)          # the dialog is modal
    try:
        _tabs12.add_btn.click()
    finally:
        win.new_page_dialog = _npd12
    assert _asked12 and _asked12[0]["kind"] == "refine", _asked12
    _ok("PT1 the page tabs, two rows: one tab per page kind in pipeline order (with a count "
        "when it holds several) and under it the pages of the kind shown; a kind tab shows "
        "its page, dragging a sub-tab reorders those pages, a rename and a new page (placed "
        "in pipeline order) show at once; + opens New page… preset to the kind shown")

    # PP1 the Pages panel: pages by kind in pipeline order, what each reads and publishes; a
    # click shows the page
    _pp12 = win.pages_panel
    _pd12 = win.shell.dock_of(_pp12)
    assert _pd12 is not None and win.dockWidgetArea(_pd12) == _Qt12.LeftDockWidgetArea
    _heads12 = [_pp12.tree.topLevelItem(i).text(0).split("  ·  ")[0]
                for i in range(_pp12.tree.topLevelItemCount())]
    assert _heads12 == ["Image Input", "Image Refinement", "Image Processing", "Analysis"], _heads12
    assert [r.data(0, _Qt12.UserRole) for r in _pp12.page_items()] == list(_ws12.pages)
    win.build_example_workspace()
    app.processEvents()
    _rows12 = {win.workspace.page(r.data(0, _Qt12.UserRole)).kind: r for r in _pp12.page_items()}
    _det12 = [_rows12["refine"].child(j).text(0) for j in range(_rows12["refine"].childCount())]
    assert _det12 == ["⇤ Image Input · raw", "Gaussian Blur", "Threshold",
                      "⇥ mask    read by Image Processing"], _det12
    _ppid12 = _rows12["process"].data(0, _Qt12.UserRole)
    _pp12.tree.itemClicked.emit(_rows12["process"], 0)
    app.processEvents()
    assert win.workspace.active == _ppid12
    assert any(r.font(0).bold() and r.data(0, _Qt12.UserRole) == _ppid12
               for r in _pp12.page_items()), "the active page is marked, not rebuilt"
    assert win._main_canvas.tabs.page_ids() == [
        p for p in win.workspace.pages if win.workspace.pages[p].kind == "process"]
    _ok("PP1 the Pages panel (left, above Nodes) lists the pages by kind in pipeline order "
        "with what each reads and publishes; a click shows the page on the active canvas")

    # SP1 (V4.00 step 11e) a Page Input's Source on its CARD is a menu of every Output it may
    # read — `<page> · <variable>`, a dot in the colour of that page's kind — and the pill
    # names what it reads, edged in that colour; the inspector's Source menu wears the same
    # dots; an Output given a name another Output has is renamed, and the status bar says so
    import nodelab_v2.node_item as _NI13
    from PySide6.QtWidgets import QComboBox as _QCB13, QMenu as _QMenu13
    from nodelab_v2.canvas import PAGE_KIND_COLORS as _PKC13
    _ws13 = win.workspace
    _ref13 = next(p for p in _ws13.pages.values() if p.kind == "refine")
    _inp13 = next(p for p in _ws13.pages.values() if p.kind == "input")
    win._show_page(win._main_canvas, _ref13.id)
    app.processEvents()
    _in13 = next(n for n, r in _ref13.doc.nodes.items() if r.op_key == "page.input")
    _ci13 = win.scene.node_items[_in13]
    _sc13 = next(c for c in _ci13.controls() if c.kind == "value" and c.obj.name == "source")
    assert _ci13._pill_text(_sc13.obj) == "Image Input · raw ▾", _ci13._pill_text(_sc13.obj)
    assert _ci13._source_kind_color().name() == _PKC13["input"]
    _ld13 = next(n for n, r in _inp13.doc.nodes.items() if r.op_key == "io.load")
    _o13 = _inp13.doc.add_node("page.output", x=600.0, y=400.0, params={"name": "raw"})
    app.processEvents()
    assert _o13.params["name"] == "raw2", "a name another Output has is made unique"
    assert "already named “raw”" in win.statusBar().currentMessage(), \
        win.statusBar().currentMessage()
    _inp13.doc.connect(_ld13, "image", _o13.id, "data")
    _seen13: list = []

    class _SrcMenu13(_QMenu13):
        def exec(self, *a, **k):               # look, then pick `raw2`
            _seen13.append([(x.text(), x.data(), not x.icon().isNull())
                            for x in self.actions()])
            return next((x for x in self.actions() if x.data() == f"{_inp13.id}:raw2"), None)

    _NI13.QMenu = _SrcMenu13
    try:
        _ci13._open_source_menu(_sc13)
    finally:
        _NI13.QMenu = _QMenu13
    assert _seen13 and all(icon for _t, _d, icon in _seen13[0]), _seen13
    assert [d for _t, d, _i in _seen13[0]] == [f"{_inp13.id}:raw", f"{_inp13.id}:raw2"], _seen13
    assert _seen13[0][0][0].endswith("✓"), "the one it reads is marked"
    assert _ref13.doc.nodes[_in13].params["source"] == f"{_inp13.id}:raw2"
    win.scene.clearSelection()
    _ci13.setSelected(True)
    app.processEvents()
    _cb13 = next(b for b in win.inspector.findChildren(_QCB13)
                 if any(str(b.itemData(i) or "").startswith(_inp13.id + ":")
                        for i in range(b.count())))
    assert all(not _cb13.itemIcon(i).isNull() for i in range(_cb13.count())
               if str(_cb13.itemData(i) or "")), "every Source entry has its kind's dot"
    _ref13.doc.nodes[_in13].params["source"] = f"{_inp13.id}:raw"
    _ref13.doc.touch(_in13)
    _inp13.doc.remove_node(_o13.id)
    app.processEvents()
    _ok("SP1 a Page Input's Source on its card is a menu of every Output it may read, each "
        "with a dot in its page kind's colour and the one it reads ticked; the pill reads "
        "`Image Input · raw ▾` edged in that colour; the inspector's menu has the dots too; "
        "an Output named like another becomes `raw2`, said on the status bar")

    # PO1 (V4.00 step 11e) the Pages panel lists each page's graph as a hierarchy: a branch
    # nests under the node it leaves, an Output is tinted with its variable in bold; a node
    # that keeps the kind of data has an on/off switch (a Threshold has none, saying why);
    # unticking it switches the node off, and clicking a node shows it on the canvas
    _pp13 = win.pages_panel
    _nr13 = _pp13.node_items(_ref13.id)
    _bl13 = next(n for n, r in _ref13.doc.nodes.items() if r.op_key == "enhance.gaussian")
    _th13 = next(n for n, r in _ref13.doc.nodes.items() if r.op_key == "analysis.threshold")
    _ou13 = next(n for n, r in _ref13.doc.nodes.items() if r.op_key == "page.output")
    assert _nr13[_bl13].checkState(1) == _Qt12.Checked
    assert not (_nr13[_th13].flags() & _Qt12.ItemIsUserCheckable)
    assert _nr13[_th13].toolTip(1).startswith("always on:"), _nr13[_th13].toolTip(1)
    assert _nr13[_ou13].font(0).bold() and _nr13[_ou13].text(0).startswith("⇥ mask")
    assert _nr13[_ou13].background(0).color() != _TH.PANEL
    _an13 = next(p for p in _ws13.pages.values() if p.kind == "analyze")
    _ar13 = _pp13.node_items(_an13.id)
    _ai13 = next(n for n, r in _an13.doc.nodes.items() if r.op_key == "page.input")
    assert {_ar13[_ai13].child(j).data(0, _Qt12.UserRole + 3)
            for j in range(_ar13[_ai13].childCount())} == \
        {n for n, r in _an13.doc.nodes.items() if r.op_key in ("plot.xy", "view.viewer")}, \
        "the Input's two branches nest under it"
    _nr13[_bl13].setCheckState(1, _Qt12.Unchecked)
    for _ in range(3):
        app.processEvents()
    assert _ref13.doc.nodes[_bl13].muted and "switched off" in win.statusBar().currentMessage()
    assert _pp13.node_items(_ref13.id)[_bl13].font(0).strikeOut()
    win._show_page(win._main_canvas, _an13.id)
    app.processEvents()
    _pp13.tree.itemClicked.emit(_pp13.node_items(_ref13.id)[_th13], 0)
    for _ in range(3):
        app.processEvents()
    assert _ws13.active == _ref13.id and win.scene.node_items[_th13].isSelected()
    assert win.set_node_muted(_ref13.id, _bl13, False) and not _ref13.doc.nodes[_bl13].muted
    assert not win.set_node_muted(_ref13.id, _th13, True), "refused: it adds a mask"
    app.processEvents()
    _ok("PO1 the Pages panel shows each page's graph as a hierarchy (an Input's two branches "
        "nest under it; Outputs tinted, the variable name in bold); a filter has an on/off "
        "switch, a Threshold none (its tooltip says why); unticking switches the node off "
        "(struck through), a node row shows that node on the canvas")

    # PT2 closing a page TAB keeps the page (V4.00 step 11d): ✕ takes it off the sub-tab row,
    # the canvas moves to the nearest open page of its kind, the Pages panel lists it "tab
    # closed" and a click there brings it back; with every tab of a kind closed the kind tab
    # stays and reopens the last one shown; the last open tab has no ✕
    _ws15 = win.workspace
    _tabs15 = win._main_canvas.tabs
    _r1 = next(p.id for p in _ws15.pages.values() if p.kind == "refine")
    win._show_page(win._main_canvas, _r1)
    _r2 = win.new_page("refine")
    app.processEvents()
    assert _tabs15.page_ids() == [_r1, _r2] and _ws15.active == _r2
    _tabs15.close_buttons()[_r2].click()
    for _ in range(3):
        app.processEvents()
    assert _r2 in _ws15.pages and _r2 in win._closed_pages
    assert _tabs15.page_ids() == [_r1] and _ws15.active == _r1, (_tabs15.page_ids(), _ws15.active)
    _row15 = next(r for r in _pp12.page_items() if r.data(0, _Qt12.UserRole) == _r2)
    assert "tab closed" in _row15.text(0) and _row15.font(0).italic()
    assert "2 pages" in _tabs15.kind_bar.tabToolTip(_tabs15.kinds().index("refine"))
    _pp12.tree.itemClicked.emit(_row15, 0)
    for _ in range(3):
        app.processEvents()
    assert _ws15.active == _r2 and _r2 not in win._closed_pages
    assert _tabs15.page_ids() == [_r1, _r2]
    assert not any(r.font(0).italic() for r in _pp12.page_items())
    # every Refinement tab closed: the kind tab stays and brings back the one shown last
    win.close_page_tab(_r1)
    win.close_page_tab(_r2)
    app.processEvents()
    assert {_r1, _r2} <= win._closed_pages and _ws15.pages[_ws15.active].kind != "refine"
    assert "refine" in _tabs15.kinds()
    _tabs15.kind_bar.setCurrentIndex(_tabs15.kinds().index("refine"))
    for _ in range(3):
        app.processEvents()
    assert _ws15.active == _r2 and _r2 not in win._closed_pages and _r1 in win._closed_pages
    # the page menu's Close tab; and the last open tab stays
    from PySide6.QtWidgets import QMenu as _QM15
    _m15 = _QM15()
    win.fill_page_menu(_m15, win._main_canvas)
    _ct15 = next(a for a in _m15.actions() if a.text() == "Close tab")
    assert _ct15.isEnabled()
    for _p15 in [p for p in _ws15.pages if p != _ws15.active]:
        win._closed_pages.add(_p15)
    win._refresh_page_views()
    app.processEvents()
    assert not _tabs15.close_buttons(), "the last open tab has no ✕"
    assert win.close_page_tab(_ws15.active) is False
    win.fill_page_menu(_m15, win._main_canvas)
    assert not next(a for a in _m15.actions() if a.text() == "Close tab").isEnabled()
    win.file_new()
    app.processEvents()
    assert not win._closed_pages and len(_tabs15.kinds()) == 4
    _ok("PT2 a closed page tab keeps the page: ✕ moves the canvas to the nearest open page of "
        "its kind, the Pages panel lists the page \"tab closed\" and a click there opens it "
        "again; the kind tab of a kind whose tabs are all closed reopens the last one shown; "
        "Close tab is in the page menu, and the last open tab has no ✕")
    os.environ.pop(_PR11.ENV_DIR, None)

    # ── SP2: a split position read on the next page is what the Viewer shows there ───
    # (user report 2026-10-06: Split Positions → an Output of one position → on Image
    # Refinement the Page Input read that one position, but the Viewer still showed every
    # position — click-to-preview is off, so nothing was pulled on arriving)
    win.file_new()
    app.processEvents()
    _sd13 = tempfile.mkdtemp(prefix="nd2split_")
    _sp13 = []
    for _k13, _v13 in enumerate((1000, 2000, 3000)):
        _p13 = os.path.join(_sd13, f"well{_k13}.tif")
        _tiff11.imwrite(_p13, np.full((2, 24, 28), _v13, np.uint16), metadata={"axes": "ZYX"})
        _sp13.append(_p13)
    _done13: list = []
    win.runner.finished.connect(lambda nid, *a: _done13.append(nid))
    win._load_source_paths(_sp13, group=True)          # one bundle card: three positions
    app.processEvents()
    _pin13 = win.workspace.active
    _bun13 = next(r for r in win.doc.nodes.values() if r.op_key == "io.load")
    _t013 = time.time()
    while _q11(_pin13, _bun13.id) not in _done13 and time.time() - _t013 < 240:
        app.processEvents()
        time.sleep(0.005)
    assert win.doc.env(_bun13.id).axes.m == 3, win.doc.env(_bun13.id).axes
    _ax13 = win.viewer.axes() if callable(win.viewer.axes) else win.viewer.axes
    assert _ax13 is not None and _ax13.m == 3, ("the load previews all three", _ax13)
    _spl13 = win.doc.add_node("util.split_positions", x=300, y=400)
    win.doc.connect(_bun13.id, "image", _spl13.id, "data")
    _out13 = win.doc.add_node("page.output", x=620, y=400, params={"name": "wellB"})
    win.doc.connect(_spl13.id, "pos1", _out13.id, "data")
    app.processEvents()
    _ref13 = next(p.id for p in win.workspace.pages.values() if p.kind == "refine")
    _done13.clear()
    win._show_page(win._main_canvas, _ref13)
    app.processEvents()
    _in13 = next(r for r in win.doc.nodes.values() if r.op_key == "page.input")
    assert _in13.params.get("source") == f"{_pin13}:wellB", _in13.params
    _t013 = time.time()
    while not _done13 and time.time() - _t013 < 240:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(3):
        app.processEvents()
    assert win.viewer.binding == (_ref13, _in13.id), ("arriving shows what the page reads",
                                                      win.viewer.binding)
    _ax13 = win.viewer.axes() if callable(win.viewer.axes) else win.viewer.axes
    assert _ax13 is not None and _ax13.m == 1, ("ONE position on the Refinement page", _ax13)
    # selecting a page boundary card previews it, click-to-preview off
    win._show_page(win._main_canvas, _pin13)
    app.processEvents()
    assert not win._follow_act.isChecked()
    _done13.clear()
    win.scene.clearSelection()
    win.scene.node_items[_out13.id].setSelected(True)
    _t013 = time.time()
    while not _done13 and time.time() - _t013 < 240:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(3):
        app.processEvents()
    assert win.viewer.binding == (_pin13, _out13.id), win.viewer.binding
    _ax13 = win.viewer.axes() if callable(win.viewer.axes) else win.viewer.axes
    assert _ax13.m == 1, _ax13
    _ok("SP2 a split position published on Image Input and read on Image Refinement: "
        "arriving on the page shows what it reads (one position, not the load's three), and "
        "selecting a Page Output or Page Input card previews it with click-to-preview off")

    # ── CT1: a Page Input offers the file's channels, like the Load card (step 11d) ────
    # (user report 2026-10-06: "on the page input, the channels should also be options just
    # as if it was the original IO node for the image")
    win.file_new()
    app.processEvents()
    _cp16 = os.path.join(tempfile.mkdtemp(prefix="nd2chan_"), "two.tif")
    _a16 = np.zeros((2, 2, 24, 28), np.uint16)
    _a16[0, :, 4:12, 4:12] = 1000
    _a16[1, :, 12:20, 14:24] = 3000
    _tiff11.imwrite(_cp16, _a16, metadata={"axes": "CZYX"})
    _done16: list = []
    win.runner.finished.connect(lambda nid, *a: _done16.append(nid))
    win._load_source_paths([_cp16])
    _pin16 = win.workspace.active
    _ld16 = next(r for r in win.doc.nodes.values() if r.op_key == "io.load")
    _t016 = time.time()
    while _q11(_pin16, _ld16.id) not in _done16 and time.time() - _t016 < 240:
        app.processEvents()
        time.sleep(0.005)
    _lab16 = [s.label for s in win.doc.output_specs(_ld16.id) if s.name.startswith("ch")]
    assert len(_lab16) == 2, _lab16
    _rf16 = next(p.id for p in win.workspace.pages.values() if p.kind == "refine")
    win._show_page(win._main_canvas, _rf16)
    app.processEvents()
    _in16 = next(r for r in win.doc.nodes.values() if r.op_key == "page.input")
    _outs16 = win.doc.output_specs(_in16.id)
    assert [s.name for s in _outs16] == ["out", "ch0", "ch1"], [s.name for s in _outs16]
    assert [s.label for s in _outs16[1:]] == _lab16, "named as on the Load card"
    assert [s.name for s in win.scene.node_items[_in16.id]._active_outputs()] == \
        ["out", "ch0", "ch1"], "the card grows them too"
    _g16 = win.doc.add_node("enhance.gaussian", x=360, y=150)
    win.doc.connect(_in16.id, "ch1", _g16.id, "data")
    app.processEvents()
    _done16.clear()
    win.pull_node(_g16.id)
    _t016 = time.time()
    while _q11(_rf16, _g16.id) not in _done16 and time.time() - _t016 < 240:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(3):
        app.processEvents()
    assert not [f for f in _seen_fail if _g16.id in f[0]], _seen_fail[-3:]
    _ax16 = win.viewer.axes() if callable(win.viewer.axes) else win.viewer.axes
    assert _ax16 is not None and _ax16.c == 1, _ax16
    assert float(win.viewer._planes[0].max()) > 1500, "channel 1's 3000 square, blurred"
    _ok("CT1 a Page Input offers one output per channel, named as on the Load card it reads "
        "through; wiring its ch1 into a Gaussian on the Refinement page pulls that one channel")

    # ── DT1: panels grouped as tabs carry their tabs on TOP (step 11d) ─────────────────
    from PySide6.QtWidgets import QTabBar as _QTB16, QTabWidget as _QTW16
    for _ar16 in (_Qt12.LeftDockWidgetArea, _Qt12.RightDockWidgetArea,
                  _Qt12.TopDockWidgetArea, _Qt12.BottomDockWidgetArea):
        assert win.tabPosition(_ar16) == _QTW16.North, _ar16
    _insd16 = win.shell.docks["inspector:0"]
    _insd16.raise_()
    app.processEvents()
    _tbar16 = next(b for b in win.findChildren(_QTB16) if b.parent() is win and b.isVisible()
                   and any(b.tabText(i) == "Properties" for i in range(b.count())))
    assert _tbar16.geometry().bottom() <= _insd16.geometry().top() + 2, \
        (_tbar16.geometry(), _insd16.geometry())
    _ok("DT1 panels grouped as tabs show their tabs on top: Properties, Spreadsheet and "
        "LabLink head their column rather than footing it")

    # ── MS1: the menu bar's Normal | Troubleshooting switch (step 11d) ─────────────────
    from nodelab_v2.mode_switch import NORMAL as _MSN, TROUBLESHOOTING as _MST
    _ms17 = win.mode_switch
    assert win.menuBar().cornerWidget(_Qt12.TopRightCorner) is _ms17 and _ms17.isVisible()
    assert not win.runner.solo_frame and _ms17.buttons[_MSN].isChecked()
    _ms17.buttons[_MST].click()
    app.processEvents()
    assert win.runner.solo_frame and win._solo_act.isChecked() and _ms17.troubleshooting()
    assert _TH.DIM2D.name() in _ms17.styleSheet(), "Troubleshooting lights amber"
    win._solo_act.setChecked(False)                       # F9 / Run ▸ — the switch follows
    app.processEvents()
    assert not win.runner.solo_frame and _ms17.buttons[_MSN].isChecked()
    win.set_solo_frame(True)                              # a programmatic change too
    assert _ms17.troubleshooting() and win._solo_act.isChecked()
    _ms17.buttons[_MSN].click()
    app.processEvents()
    assert not win.runner.solo_frame and not win._solo_act.isChecked()
    _ok("MS1 the menu bar's Normal | Troubleshooting switch: picking a segment turns the "
        "troubleshooting scope on and off (Troubleshooting lit amber), and F9, the Run menu "
        "and a programmatic change all move it")

    # ── FB1: fit-to-nodes beside the canvas's maximize button (step 11d) ──────────────
    _gv17 = win.view
    _fb17 = _gv17._fit_btn
    assert _fb17.isVisible() and _fb17.kind == "fit"
    assert _fb17.geometry().right() < _gv17._max_btn.geometry().left() and \
        abs(_fb17.geometry().top() - _gv17._max_btn.geometry().top()) <= 1, "beside ⛶"
    win.doc.add_node("enhance.gamma", x=3000, y=2400)
    _gv17.resetTransform()
    _gv17.centerOn(-4000, -4000)
    app.processEvents()
    _vis17 = lambda: all(_gv17.viewport().rect().intersects(               # noqa: E731
        _gv17.mapFromScene(it.sceneBoundingRect()).boundingRect())
        for it in win.scene.node_items.values())
    assert win.scene.node_items and not _vis17()
    _fb17.click()
    app.processEvents()
    assert _vis17(), "every card in view after Fit"
    _ok("FB1 a fit-to-nodes button sits beside the canvas's maximize button and brings every "
        "card into view, as Home does")

    # ── VC1: the Viewer's controls as panels of their own (step 11d) ──────────────────
    # (user request 2026-10-06: the M/T/Z and play buttons, and the histograms and colour
    # channel options, each their own window, placed wherever; channels stacked if needed)
    from nodelab_v2.window import CHANNELS_KIND as _CHK16, PLAYBACK_KIND as _PBK16
    _sh16 = win.shell
    for _dx16 in _sh16.docks_of("viewer")[1:]:          # one Viewer, as a fresh session
        _dx16.close()
    app.processEvents()
    _vd16 = _sh16.docks_of("viewer")[0]
    _v16 = _vd16.panel
    win._activate_viewer(_v16)
    win.reset_layout()
    for _ in range(3):
        app.processEvents()
    _pbd16, _chd16 = _sh16.docks[f"{_PBK16}:0"], _sh16.docks[f"{_CHK16}:0"]
    _TOP = _Qt12.TopDockWidgetArea
    assert not _pbd16.isHidden() and not _chd16.isHidden()
    assert win.dockWidgetArea(_pbd16) == win.dockWidgetArea(_chd16) == \
        win.dockWidgetArea(_vd16) == _TOP
    assert _pbd16.geometry().top() >= _vd16.geometry().bottom() - 2, "Playback under the Viewer"
    assert _chd16.geometry().left() >= _pbd16.geometry().right() - 2 and \
        abs(_chd16.geometry().top() - _pbd16.geometry().top()) <= 2, "Channels beside Playback"
    assert _v16.controls_detached and not _v16._controls.isVisible()
    assert win.playback_panel.isAncestorOf(_v16._sliders["z"])
    assert win.playback_panel.isAncestorOf(_v16._play_btns["t"])
    assert win.channels_panel.isAncestorOf(_v16._lut_auto)
    assert win.channels_panel.isAncestorOf(_v16._chan_btns[0])
    # one column per channel, side by side while they fit, stacked when the panel is narrow
    _cols16 = _v16._lut_strip_w
    _show16 = win._show_page
    _show16(win._main_canvas, _pin16)
    app.processEvents()
    _done16.clear()
    win.pull_node(_ld16.id)
    _t016 = time.time()
    while _q11(_pin16, _ld16.id) not in _done16 and time.time() - _t016 < 240:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(3):
        app.processEvents()
    assert len(_cols16.columns()) == 2, len(_cols16.columns())
    _chd16.setFloating(True)
    _chd16.resize(200, 700)
    for _ in range(4):
        app.processEvents()
    _ys16 = [c.geometry().top() for c in _cols16.columns()]
    assert _cols16.per_row() == 1 and _ys16[1] > _ys16[0], ("stacked when narrow", _ys16)
    _chd16.resize(700, 260)
    for _ in range(4):
        app.processEvents()
    _ys16 = [c.geometry().top() for c in _cols16.columns()]
    assert _cols16.per_row() == 2 and _ys16[0] == _ys16[1], ("side by side when wide", _ys16)
    _chd16.setFloating(False)
    app.processEvents()
    # the sections keep the Viewer's look through a theme change
    win.set_theme("light")
    assert _TH.PANEL.name() in _v16.axes_section.styleSheet()
    assert _TH.PANEL.name() in _v16.channel_section.styleSheet()
    win.set_theme("dark")
    assert _TH.PANEL.name() in _v16.channel_section.styleSheet()
    # a second Viewer: both panels follow the ACTIVE one and name it
    _d216 = _sh16.spawn("viewer", beside=_vd16)
    app.processEvents()
    assert win.playback_panel.shown_viewer() is _d216.panel
    assert win.channels_panel.shown_viewer() is _d216.panel
    assert "Viewer 2" in _pbd16.windowTitle() and "Viewer 2" in _chd16.windowTitle()
    win._activate_viewer(_v16)
    app.processEvents()
    assert win.playback_panel.shown_viewer() is _v16 and "Viewer 1" in _pbd16.windowTitle()
    _d216.close()
    app.processEvents()
    assert len(win.playback_panel._sections) == 1 and len(win.channels_panel._sections) == 1
    assert _pbd16.windowTitle() == "Playback", _pbd16.windowTitle()
    # maximized: the docked panels step aside and the mini-map carries the controls, compact
    win.set_maximized(True)
    app.processEvents()
    assert _pbd16.isHidden() and _chd16.isHidden() and not _v16.controls_detached
    assert _v16.isAncestorOf(_v16._sliders["z"]) and not any(
        h.isVisible() for h in _v16._hists.values()), "compact: no histograms in the mini-map"
    win.set_maximized(False)
    app.processEvents()
    assert not _pbd16.isHidden() and not _chd16.isHidden() and _v16.controls_detached
    assert win.playback_panel.shown_viewer() is _v16
    assert win.channels_panel.isAncestorOf(_v16._hists[0]) and all(
        h.isVisible() for h in _v16._hists.values())
    # a panel a saved layout never heard of goes to its default place
    _pbd16.setFloating(True)
    app.processEvents()
    _sh16.place_default(_pbd16)
    app.processEvents()
    assert not _pbd16.isFloating() and win.dockWidgetArea(_pbd16) == _TOP
    _ok("VC1 the Viewer's controls are panels of their own: Playback (M/T/Z, play) under the "
        "Viewer and Channels (tools + a column per channel) beside it; the channel columns "
        "stack when the panel is narrow and sit side by side when it is wide; both follow "
        "the active Viewer and name it once there are several; maximized, the mini-map "
        "carries them compact and the panels come back after; a panel unknown to a saved "
        "layout lands in its default place")

    # VC2 a layout saved BEFORE these panels existed (the step-11c build) restores with them
    # in their default place and everything else as saved. They are taken out of the window
    # before Qt restores the rest and placed after: re-adding a dock that restoreState had
    # laid out without knowing it was an access violation on the native platform (the user's
    # crash, 2026-10-06; offscreen survived it, so this pins the order and the result)
    from nodelab_v2 import layout_store as _LS18
    from nodelab_v2.window import MainWindow as _MW18
    _lay18 = os.path.join(tempfile.mkdtemp(prefix="nd2layout_"), "layout.json")
    _env18 = {k: os.environ.get(k) for k in (_LS18.ENV_FILE, _LS18.ENV_ENABLED)}
    os.environ[_LS18.ENV_FILE] = _lay18
    try:
        _wa18 = _MW18(persist_layout=False)
        _wa18.show()
        app.processEvents()
        for _n18 in ("playback:0", "channels:0"):
            _wa18.removeDockWidget(_wa18.shell.docks[_n18])
        _wa18.shell.docks["sheet:0"].setFloating(True)      # something the user arranged
        app.processEvents()
        _rec18 = _wa18.shell.layout_record()
        _rec18["docks"] = [d for d in _rec18["docks"]
                           if d["name"] not in ("playback:0", "channels:0")]
        _LS18.save_layout(_rec18, _lay18)
        _wa18.close()
        _ev18: list = []
        _rm18, _rs18 = _MW18.removeDockWidget, _MW18.restoreState

        def _spy_rm18(self, d, _e=_ev18):
            _e.append(("remove", d.objectName()))
            return _rm18(self, d)

        def _spy_rs18(self, *a, _e=_ev18):
            _e.append(("restore", ""))
            return _rs18(self, *a)
        _MW18.removeDockWidget, _MW18.restoreState = _spy_rm18, _spy_rs18
        try:
            _wb18 = _MW18(persist_layout=True)
        finally:
            _MW18.removeDockWidget, _MW18.restoreState = _rm18, _rs18
        _wb18.show()
        app.processEvents()
        assert _wb18._layout_restored
        _sb18 = _wb18.shell
        _vb18 = _sb18.docks["viewer:0"]
        _pb18, _cb18 = _sb18.docks["playback:0"], _sb18.docks["channels:0"]
        _at18 = _ev18.index(("restore", ""))
        assert {("remove", "playback:0"), ("remove", "channels:0")} <= set(_ev18[:_at18]) \
            and _ev18[_at18 - 2:_at18] == [("remove", "playback:0"), ("remove", "channels:0")], \
            ("the unknown panels leave the window right before Qt restores", _ev18[-6:])
        assert not _pb18.isHidden() and not _cb18.isHidden()
        assert _wb18.dockWidgetArea(_pb18) == _wb18.dockWidgetArea(_cb18) == \
            _Qt12.TopDockWidgetArea
        assert _pb18.geometry().top() >= _vb18.geometry().bottom() - 2
        assert _sb18.docks["sheet:0"].isFloating(), "the rest of the layout is as saved"
        _wb18._persist_layout = False
        _wb18.close()
        app.processEvents()
    finally:
        for _k18, _v18 in _env18.items():
            if _v18 is None:
                os.environ.pop(_k18, None)
            else:
                os.environ[_k18] = _v18
    _ok("VC2 a layout saved before the Playback and Channels panels existed restores with "
        "them under the Viewer and everything else as saved; they are out of the window while "
        "Qt restores the rest (re-adding such a dock afterwards crashed natively)")

    # a LabLink panel left on screen polls its hub over HTTP on a QThread; a socket connect
    # still in flight when os._exit tears the process down crashes it (exit 139 after every
    # check passed, a false failure — caught with -X faulthandler, 2026-10-06). Stop the
    # polling and let the last request finish, as the app's own close path does.
    from nodelab_v2.lablink.panel import _TaskHost as _LLHost
    for _h in win.findChildren(_LLHost):
        _tm = getattr(_h, "_timer", None)
        if _tm is not None:
            _tm.stop()
        _h.stop_tasks()

    print("\nALL PHASE-5 GUI PROBES PASSED")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
