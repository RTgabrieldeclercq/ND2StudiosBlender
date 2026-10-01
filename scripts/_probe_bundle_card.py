"""Render a file-bundle card next to an ordinary source card, to a PNG.

The bundle look is a visual claim; this is how it gets checked rather than asserted.
Run:  .venv\\Scripts\\python.exe -B -u scripts\\_probe_bundle_card.py out.png
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from nodegraph import nodes as _catalog          # noqa: F401 — registers the catalog
from nodelab_v2.document import CHANNELS_KEY, GraphDocument, TITLE_KEY
from nodelab_v2.ops import LOAD_OP, ensure_ops
from nodelab_v2.runner import BUNDLE_PATHS_KEY
from nodelab_v2.scene import GraphScene

ensure_ops()

CHANS = [{"name": "DAPI", "emission_nm": 461, "color": None},
         {"name": "GFP", "emission_nm": 509, "color": None}]


def main(out: str) -> int:
    app = QApplication.instance() or QApplication([])
    doc = GraphDocument()
    doc.add_node(LOAD_OP, node_id="one", x=0, y=0,
                 params={"path": "/data/plateA.nd2", TITLE_KEY: "plateA.nd2",
                         CHANNELS_KEY: CHANS})
    doc.add_node(LOAD_OP, node_id="bundle", x=260, y=0,
                 params={"path": "/data/plateA.nd2",
                         BUNDLE_PATHS_KEY: [f"/data/plate{k}.nd2" for k in "ABCDE"],
                         TITLE_KEY: "5 files", CHANNELS_KEY: CHANS})
    scene = GraphScene(doc)

    one = scene.node_items["one"]
    bun = scene.node_items["bundle"]
    print(f"single card: stack_depth={one.stack_depth()} "
          f"bounding={one.boundingRect().width():.0f}x{one.boundingRect().height():.0f}")
    print(f"bundle card: stack_depth={bun.stack_depth()} "
          f"bounding={bun.boundingRect().width():.0f}x{bun.boundingRect().height():.0f}")
    assert one.stack_depth() == 0, "an ordinary source must not draw a stack"
    assert bun.stack_depth() == 3, bun.stack_depth()      # capped at STACK_MAX
    # the extra room is asymmetric — down/right only, so the glow and the reroute ring
    # keep the reach they had
    b_one, b_bun = one.boundingRect(), bun.boundingRect()
    assert b_bun.left() == b_one.left() and b_bun.top() == b_one.top()
    grew = bun.stack_depth() * bun.STACK_STEP
    assert abs((b_bun.right() - b_one.right()) - grew) < 1e-6, (b_bun, b_one)
    # the card's own geometry is UNCHANGED, so frames and auto-layout do not breathe
    assert one.card_rect() == bun.card_rect(), (one.card_rect(), bun.card_rect())

    rect = scene.itemsBoundingRect().adjusted(-24, -24, 24, 24)
    img = QImage(int(rect.width()) * 2, int(rect.height()) * 2,
                 QImage.Format_ARGB32)
    img.fill(0xFF1B1D21)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    scene.render(p, QRectF(img.rect()), rect)
    p.end()
    img.save(out)
    print(f"wrote {out}")
    print("BUNDLE CARD PROBE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bundle_card.png"))
