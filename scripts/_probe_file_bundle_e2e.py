"""End-to-end probe for the FILE BUNDLE: two real TIFFs on one card, one pipeline,
one spreadsheet that names the file each row came from.

This is the whole feature in one run — the GUI loader's grouping check, the runner's
bundle resolution and ingest, a pull through a real segmentation chain, and the exported
CSV. It uses the runner (so Qt, offscreen) because the bundle plumbing lives there.

Run:  .venv\\Scripts\\python.exe -B -u scripts\\_probe_file_bundle_e2e.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile

from PySide6.QtWidgets import QApplication

from nodegraph.metadata import SOURCE_FILE_KEY
from nodegraph import nodes as _catalog          # noqa: F401 — registers the catalog
from nodelab_v2.document import GraphDocument
from nodelab_v2.export import export_dataset
from nodelab_v2.ops import LOAD_OP, ensure_ops
from nodelab_v2.runner import (BUNDLE_PATHS_KEY, EngineRunner, bundle_key,
                               _clean_source_paths)
from nodelab_v2.tables import SOURCE_FILE_COLUMN, all_tables

ensure_ops()                                     # io.load / view.viewer are GUI-layer ops

H = W = 64


def _blobs(centres, sigma=3.5) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    out = np.zeros((H, W), float)
    for (cy, cx) in centres:
        out += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    return (out * 4000.0).astype(np.uint16)


def _write(path: str, plane) -> str:
    """A single-plane TIFF — one position, which is what a plain 2-D TIFF IS.

    (A stack of planes in one TIFF reads as T, not M: there is nothing in the format
    saying "these are stage positions". Bundling is how several single-position files
    become one multipoint series, which is exactly the case this probe covers.)"""
    tifffile.imwrite(path, plane)
    return path


def main() -> int:
    app = QApplication.instance() or QApplication([])
    tmp = tempfile.mkdtemp(prefix="nd2bundle_e2e_")
    # three single-position files with DIFFERENT object counts, so the per-file rows are
    # distinguishable in the result and a mis-mapped `m` would be visible
    a = _write(os.path.join(tmp, "plateA.tif"), _blobs([(16, 16), (44, 40)]))
    b = _write(os.path.join(tmp, "plateB.tif"),
               _blobs([(12, 40), (40, 14), (48, 48)]))
    c = _write(os.path.join(tmp, "plateC.tif"),
               _blobs([(10, 10), (10, 50), (50, 10), (50, 50)]))
    want = {0: ("plateA.tif", 2), 1: ("plateB.tif", 3), 2: ("plateC.tif", 4)}
    print("wrote plateA/B/C.tif — 2, 3 and 4 objects, one position each")

    doc = GraphDocument()
    rec = doc.add_node(LOAD_OP, node_id="src", x=0, y=0,
                       params={"path": a, BUNDLE_PATHS_KEY: [a, b, c]})
    doc.add_node("analysis.threshold", node_id="thr", x=200, y=0,
                 modes={"method": "otsu", "scope": "plane"})
    doc.add_node("analysis.label", node_id="lab", x=400, y=0)
    doc.add_node("analysis.measure", node_id="mea", x=600, y=0)
    doc.connect("src", "image", "thr", "data")
    doc.connect("thr", "out", "lab", "data")
    doc.connect("lab", "out", "mea", "data")

    assert _clean_source_paths(dict(rec.params)) == [a, b, c]
    runner = EngineRunner(doc)

    # the card's identity is the BUNDLE's, not its first file's
    assert runner.source_state("src") == "cold", runner.source_state("src")
    assert runner._source_key_of([a, b, c]) == bundle_key([a, b, c])

    print("pulling the bundle through threshold -> label -> measure ...")
    import time
    done, err = {}, {}
    runner.finished.connect(
        lambda nid, payload, *rest: done.setdefault(nid, payload))
    runner.failed.connect(lambda nid, tb: err.setdefault(nid, tb))
    runner.pull("mea")
    t0 = time.time()
    while "mea" not in done and not err and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    assert not err, f"the bundle pull FAILED:\n{list(err.values())[0]}"
    ds = done.get("mea")
    assert ds is not None, "the bundle pull produced no payload (timed out)"

    print(f"pulled: axes.m = {ds.axes.m} (1 + 1 + 1)")
    assert ds.axes.m == 3, ds.axes
    names = ds.metadata.get(SOURCE_FILE_KEY)
    assert names == ["plateA.tif", "plateB.tif", "plateC.tif"], names
    print(f"source_file = {names}")

    tabs = all_tables(ds)
    tagged = {k: cols for k, cols in tabs.items() if SOURCE_FILE_COLUMN in cols}
    assert tagged, f"no table carries a {SOURCE_FILE_COLUMN!r} column; {list(tabs)}"
    for key, cols in tagged.items():
        pairs = sorted(set(zip([int(v) for v in cols["m"]],
                               [str(v) for v in cols[SOURCE_FILE_COLUMN]])))
        print(f"  {key}: m->file {pairs}")
        for m, f in pairs:
            assert f == want[m][0], (key, m, f, want[m][0])

    # the rows really are the right file's objects: each plate has its own object count
    lab = tabs.get(("label", "labels")) or tabs.get(("point", "spots"))
    if lab is not None and "m" in lab:
        per_m = {}
        for m in [int(v) for v in lab["m"]]:
            per_m[m] = per_m.get(m, 0) + 1
        print(f"  objects per position = {per_m}")
        for m, (fname, n_obj) in want.items():
            assert per_m.get(m) == n_obj, (
                f"position {m} ({fname}) should hold {n_obj} objects, got "
                f"{per_m.get(m)} — the bundle mapped a position to the wrong file")

    csv_path = os.path.join(tmp, "out.csv")
    n = export_dataset(ds, csv_path)
    with open(csv_path, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
        body = fh.read()
    assert SOURCE_FILE_COLUMN in head, head
    assert "plateA.tif" in body and "plateB.tif" in body
    print(f"exported {n} rows; header = {head}")

    print("\nFILE BUNDLE E2E PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
