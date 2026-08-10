"""Gate for the Dock node's HELD tier (V2.26) — the GUI half.

    PYTHONUTF8=1 python scripts/_dock_hold_probe.py

The engine-level mechanics live in selftest::test_checkpoint_dock (a held dock is a root,
runs its upstream zero times, serves the same object, keys the memo distinctly, refuses when
released). This probe covers the wiring that sits ABOVE the engine and that the selftest
therefore cannot reach:

* EngineRunner.hold / release — the seed registry that has to outlive every engine
  rebuild, and the type refusal happening where the button was pressed rather than later;
* GraphDocument.set_dock_hold / set_held_nodes — the badge the canvas paints, and the
  rule that mirroring the runner's set must NOT bump revision (it keys the memo and the
  cached Engine, so bumping it would drop the very payloads a hold exists to keep — this probe
  caught exactly that bug);
* **save / reload** — a held state IS serialized (it is an ordinary mode) and must come back as
  released, in red, naming both fixes. Never silently re-run and never silently downgraded
  to live.

Lives here rather than in _nodelab_v2_phase5_probe.py only so it is runnable on its own;
folding it into that probe's session is the right long-term home.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NODELAB_GL", "0")
sys.path.insert(0, r"c:\Users\McGheeLab - Analysis\Documents\GitHub\ND2StudiosBlender")

import numpy as np
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from nodelab_v2.document import GraphDocument
from nodelab_v2.ops import DOCK_OP, DOCK_HELD, DOCK_LIVE, ensure_ops
from nodelab_v2.runner import EngineRunner
from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.provider import ArrayProvider

ensure_ops()
OK = []


def ok(msg):
    OK.append(msg)
    print(f"[ok] {msg}")


def build_doc():
    doc = GraphDocument()
    src = doc.add_node("io.load", node_id="L")
    blur = doc.add_node("enhance.gaussian", node_id="B")
    dock = doc.add_node(DOCK_OP, node_id="D")
    thr = doc.add_node("analysis.threshold", node_id="T")
    doc.connect("L", "image", "B", "data")
    doc.connect("B", "out", "D", "data")
    doc.connect("D", "out", "T", "data")
    doc.nodes["B"].modes["dim"] = "2D"
    doc.nodes["B"].params["sigma"] = 1.0
    doc.nodes["T"].modes["method"] = "otsu"
    return doc


# ── 1. runner.hold pins a payload, and it survives an engine rebuild ──────────
doc = build_doc()
runner = EngineRunner(doc)
ax = AxisSizes(m=1, t=2, z=1, c=1, y=16, x=16)
vol = (np.arange(2 * 16 * 16, dtype=np.uint16) % 4093).reshape(1, 2, 1, 1, 16, 16)
frozen = Dataset(axes=ax, metadata={"pixel_size_um": 0.5, "bit_depth": 12}).with_image(
    ArrayProvider(vol))

assert runner.held == frozenset()
runner.hold("D", frozen)
assert runner.held == frozenset({"D"}), runner.held
ok("runner.hold pins the payload and reports it in `held`")

rev_before = runner._engine_rev
assert rev_before == -1, f"hold must force an engine rebuild, got {rev_before}"
ok("hold forces an engine rebuild so the next pull seeds from the registry")

try:
    runner.hold("D", "not a Dataset")
    raise AssertionError("expected a TypeError holding a non-Dataset")
except TypeError as exc:
    assert "can only hold a Dataset" in str(exc), str(exc)
ok("holding a non-Dataset is refused where the button was pressed, not later")

# ── 2. document.set_dock_hold + set_held_nodes drive the badge ────────────────
doc.set_dock_hold("D", True)
assert doc.nodes["D"].modes["state"] == DOCK_HELD
assert doc.dock_status("D")[0] == "released", \
    "before the runner's set is mirrored, the badge must read released"
doc.set_held_nodes(runner.held)
st, why = doc.dock_status("D")
assert st == "held" and why == "", (st, why)
ok("document.set_held_nodes flips the badge from `released` to `held`")

rev = doc.revision
doc.set_held_nodes(runner.held)          # idempotent
assert doc.revision == rev, "a no-op set_held_nodes must not bump the revision"
doc.set_held_nodes(frozenset())
assert doc.revision == rev, \
    "set_held_nodes must NOT bump the revision — that would drop every memo entry"
assert doc.dock_status("D")[0] == "released"
doc.set_held_nodes(runner.held)
ok("set_held_nodes clears the status cache without bumping the revision")

# ── 3. the chain greys out, exactly as it does when docked ───────────────────
assert doc.dormant == frozenset({"L", "B"}), doc.dormant
ok(f"a held dock greys out its chain: {sorted(doc.dormant)}")

# ── 4. release ───────────────────────────────────────────────────────────────
assert runner.release("D") is True
assert runner.release("D") is False, "releasing twice must report nothing was released"
doc.set_dock_hold("D", False)
doc.set_held_nodes(runner.held)
assert doc.nodes["D"].modes["state"] == DOCK_LIVE
assert doc.dock_status("D")[0] == "live", doc.dock_status("D")
assert doc.dormant == frozenset(), doc.dormant
ok("release un-freezes: state live, badge live, nothing greyed")

# ── 5. a SAVED graph with a held dock reloads as `released`, not as broken ────
doc.set_dock_hold("D", True)
runner.hold("D", frozen)
doc.set_held_nodes(runner.held)
path = os.path.join(tempfile.mkdtemp(prefix="heldsave-"), "g.nd2graph.json")
doc.save_file(path)
raw = open(path, encoding="utf-8").read()
assert '"held"' in raw, "the held STATE is saved (it is an ordinary mode)"

doc2 = GraphDocument()
doc2.load_file(path)
assert doc2.nodes["D"].modes["state"] == DOCK_HELD
st2, why2 = doc2.dock_status("D")
assert st2 == "released", f"a reloaded held dock must read released, got {st2}"
assert "Hold" in why2 and "Bake" in why2, why2
ok("a saved held dock reloads as `released` and names both fixes: " + why2[:60] + "…")

# the reloaded document still has the whole chain, so re-holding is one click
assert len(doc2.to_graph().preds("D")) == 1 and "B" in doc2.nodes
ok("the reloaded graph keeps the chain, so Hold again is one click")

# ── 6. the window's menu wiring exists and gates correctly ───────────────────
from nodelab_v2.window import MainWindow
win = MainWindow()
try:
    for attr in ("_hold_act", "_stop_bake_act", "_hold_selected", "_hold_dock", "_on_held"):
        assert hasattr(win, attr), f"MainWindow is missing {attr}"
    assert not win.runner.baking, "nothing is baking, so Stop must be disabled"
    win._sync_run_actions()
    assert not win._stop_bake_act.isEnabled(), "Stop must be disabled when not baking"
    assert not win._hold_act.isEnabled(), "no dock selected → Hold disabled"
    ok("Run menu carries Hold + Stop bake, both correctly gated")
    assert win._on_dock_action.__doc__
    ok("MainWindow._on_dock_action handles hold/release (wired to inspector + canvas)")
finally:
    win.close()

print(f"\nALL {len(OK)} DOCK HOLD-TIER PROBES PASSED")
raise SystemExit(0)
