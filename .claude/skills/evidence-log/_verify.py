"""End-to-end check of the installed evidence-log skill.

Runs entirely in a temp dir so it never touches the repo's own CodeLog/live.
Verifies: log/ask/confirm/reject/attach, render(), export_markdown(),
export_archive() with real PNGs inlined, the missing-figure warning, and the
labelsheet build + stratified order + read() round-trip.

Two regressions it exists to hold down, both of which silently destroyed evidence:

1. A `table` logged with `cols` and one logged with `headers` must BOTH reach
   index.html and evidence_log.md, and likewise a `kv` logged with `rows` and
   with `items`. Each spelling used to render on exactly one of the two surfaces
   and vanish without a word on the other.
2. A short prefix of the labelling sheet must still straddle the score range.
   Round-robining the strata in band order made the first `n_strata` tiles
   monotonic in the score, so a user who labelled ten and stopped had calibrated
   only the easy end -- the sorted-sheet failure, one level up.

Run it from the repo root:

    .\.venv\Scripts\python.exe -B .claude\skills\evidence-log\_verify.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SKILL = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL)

import devlog as D          # noqa: E402
import labelsheet as LS     # noqa: E402

FAILED: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  OK   " if cond else "  FAIL ") + what)
    if not cond:
        FAILED.append(what)


def fig(path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    f, ax = plt.subplots(figsize=(3.2, 2.2), dpi=110)
    ax.imshow(rng.normal(size=(40, 60)), cmap="magma")
    ax.set_title(f"synthetic field {seed}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    f.tight_layout()
    f.savefig(path)
    plt.close(f)


root = tempfile.mkdtemp(prefix="evlog_verify_")
print(f"root = {root}\n")

# ───────────────────────── devlog ─────────────────────────
print("devlog")
D.configure(os.path.join(root, "live"), title="verify - evidence log")
fig(os.path.join(root, "live", "img", "a.png"), 1)
fig(os.path.join(root, "live", "img", "b.png"), 2)

e1 = D.log(
    kind="found",
    title="Both table spellings and both kv spellings must survive",
    body="Four evidence blocks, one per accepted key spelling.",
    why="Each spelling used to render on one surface only.",
    evidence=[
        {"type": "table", "cols": ["hdr_via_cols", "n"],
         "rows": [["row_cols", "OK 12"], ["row_cols2", "BAD 3"]],
         "caption": "table logged the way SKILL.md documents it"},
        {"type": "table", "headers": ["hdr_via_headers", "n"],
         "rows": [["row_headers", 7]],
         "caption": "table logged the way the HTML renderer used to require"},
        {"type": "table", "rows": [["row_no_header_at_all", 1]],
         "caption": "no header key at all - must still reach the markdown"},
        {"type": "kv", "rows": [["kv_via_rows", "0.798", "note_rows"]],
         "caption": "kv logged as documented"},
        {"type": "kv", "items": [["kv_via_items", "0.93", "note_items"]],
         "caption": "kv logged the way the HTML renderer used to require"},
        {"type": "code", "text": "solidity = area / convex_area  # code_block_marker"},
        {"type": "note", "text": "a **bold** note with `code` - note_block_marker"},
    ],
    images=[{"src": "img/a.png", "caption": "Look at the bright diagonal, not the colours."}],
    status="confirmed",
)
check(e1 == "e001", f"first entry id is e001 (got {e1})")

q = D.ask(title="A pending question", body="b", why="w",
          asks="Confirm or reject this.")
r = D.log(kind="error", title="My own retraction", why="w", status="info")
D.confirm(q, note="user_note_marker: confirmed by hand")
D.attach(e1, [{"src": "img/b.png", "caption": "Second figure, attached after the fact."}])
bad = D.log(kind="did", title="Entry whose figure is missing", status="info",
            images=[{"src": "img/does_not_exist.png", "caption": "c"}])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    D.attach(bad, [{"src": "img/also_missing.png", "caption": "c"}])
check("figure not found" in buf.getvalue(), "missing figure is warned about, not silent")

# log() must APPEND, not rewrite: a concurrent writer must not be able to clobber an
# entry that is already on disk. Simulated by mutating the file underneath devlog and
# checking the pre-existing bytes survive the next log() call.
before = open(D.LOG, encoding="utf-8").read()
with open(D.LOG, "a", encoding="utf-8") as _f:
    _f.write(json.dumps(dict(id="e900", ts="外", kind="did", title="written_by_someone_else",
                             body="", why="", evidence=[], images=[], asks="",
                             status="info", note="")) + "\n")
D.log(kind="did", title="an entry logged after the other writer", status="info")
after = open(D.LOG, encoding="utf-8").read()
check(before in after, "log() appends - it does not rewrite what is already on disk")
check("written_by_someone_else" in after,
      "a concurrent writer's entry survives the next log() call")

page = D.render()
html_txt = open(page, encoding="utf-8").read()
md = open(D.export_markdown(), encoding="utf-8").read()

for tok in ("hdr_via_cols", "hdr_via_headers", "row_no_header_at_all",
            "kv_via_rows", "kv_via_items", "note_rows", "note_items",
            "code_block_marker", "note_block_marker", "user_note_marker"):
    check(tok in html_txt, f"index.html carries {tok}")
    check(tok in md, f"evidence_log.md carries {tok}")

check("img/a.png" in html_txt and "img/b.png" in html_txt,
      "both figures referenced in index.html")
check("AWAITING YOUR CALL" not in html_txt, "resolved question no longer pins as pending")
check(html_txt.index("Entry whose figure is missing") <
      html_txt.index("Both table spellings"), "page is newest-first")
check("http-equiv" not in html_txt.lower() and "meta refresh" not in html_txt.lower(),
      "no meta refresh - the page never reloads itself")

# Header tool links are presence-detected: no dead links before the tools exist.
check("label/index.html" not in html_txt and "gt/index.html" not in html_txt,
      "no tool links in the header before either tool is built")
for rel in ("label/index.html", "gt/index.html"):
    p_ = os.path.join(root, "live", *rel.split("/"))
    os.makedirs(os.path.dirname(p_), exist_ok=True)
    open(p_, "w", encoding="utf-8").write("<html></html>")
html_txt = open(D.render(), encoding="utf-8").read()
check("label/index.html" in html_txt, "labeller link appears once the sheet exists")
check("gt/index.html" in html_txt, "ground-truth link appears once the tool exists")

arch = D.export_archive()
arch_txt = open(arch, encoding="utf-8").read()
check("data:image/jpeg;base64," in arch_txt, "archive inlines the figures")
check("img/a.png" not in arch_txt, "archive has no relative image paths left to break")
check("copy for Claude" not in arch_txt, "archive strips the reply UI")
n_conf = sum(1 for ln in open(os.path.join(root, "live", "log.jsonl"), encoding="utf-8")
             if '"status": "confirmed"' in ln)
check(n_conf == 2, f"two entries ended up confirmed ({n_conf})")
check(f"{n_conf} confirmed" in arch_txt, "archive summary counts the confirmations")
check("0 rejected" in arch_txt, "archive summary counts the retractions")

# ───────────────────────── labelsheet ─────────────────────────
print("\nlabelsheet")
lab = os.path.join(root, "live", "label")
os.makedirs(os.path.join(lab, "img"), exist_ok=True)
manifest = [{"uid": f"M{i // 12:02d}_{i:04d}", "group": f"M{i // 12:02d}",
             "solidity": round(0.60 + 0.39 * i / 47, 4)} for i in range(48)]
for m in manifest:
    fig(os.path.join(lab, "img", f"{m['uid']}.png"), hash(m["uid"]) % 1000)

try:
    LS.build(out_dir=lab, manifest=manifest + [{"uid": "NOPE", "group": "M00"}],
             question="q")
    check(False, "build() refuses a manifest with no tile rendered")
except FileNotFoundError:
    check(True, "build() refuses a manifest with no tile rendered")

lab_page = LS.build(
    out_dir=lab, manifest=manifest,
    question="How many granules are in each object?",
    vocab=[("1", "one", "#34d399"), ("2", "two", "#fbbf24"),
           ("x", "not a granule", "#a78bfa"), ("?", "cannot tell", "#64748b")],
    strata_key="solidity", priority=["M03_0040", "M00_0002"],
    note="Every tile is 400 um across.")
lab_txt = open(lab_page, encoding="utf-8").read()

order = [s.split('"')[0] for s in lab_txt.split('data-uid="')[1:]]
check(order[:2] == ["M03_0040", "M00_0002"], f"priority items come first ({order[:2]})")
sol = {m["uid"]: m["solidity"] for m in manifest}
rest = [sol[u] for u in order[2:]]
check(len(order) == 48, f"every tile is on the page ({len(order)})")
check(rest != sorted(rest, reverse=True) and rest != sorted(rest),
      "the remainder is stratified, not sorted by the score being calibrated")
full = max(rest) - min(rest)
# The promise is that a user who stops early still has a balanced sample. Check it at
# prefixes SHORTER than one full pass over the 24 bands, which is where round-robining the
# bands in order used to degenerate into a plain descending sort.
for k in (6, 10, 15, 24):
    span = max(rest[:k]) - min(rest[:k])
    check(span >= 0.5 * full,
          f"a {k}-tile prefix still straddles the range "
          f"({span:.2f} of {full:.2f} = {span / full:.0%})")
check(rest[:24] != sorted(rest[:24], reverse=True),
      "the first full pass is not monotonic in the score being calibrated")
check(sorted(rest) == sorted(sol[u] for u in order[2:]),
      "reordering neither drops nor duplicates a tile")
for tok in ("0.798", "0.93", "solidity"):
    check(tok not in lab_txt, f"blind: {tok!r} is not on the sheet")

csv = os.path.join(root, "labels.csv")
with open(csv, "w", encoding="utf-8") as f:
    f.write("LABELS  uid,label\nM03_0040,2\nM00_0002,x\nM01_0013,?\n"
            "GHOST_9999,1\n\nNOTES\nthe third one is out of focus\n")
rows, notes = LS.read(csv, manifest)
check(len(rows) == 3, f"read() drops the uid that is not in the manifest ({len(rows)})")
check({r["uid"]: r["label"] for r in rows} ==
      {"M03_0040": "2", "M00_0002": "x", "M01_0013": "?"}, "read() joins labels correctly")
check(rows[0]["solidity"] == sol[rows[0]["uid"]], "read() carries the manifest fields")
check(notes == ["the third one is out of focus"], f"read() picks up the notes ({notes})")

# ───────────────────────── CLI ─────────────────────────
print("\nCLI")
import subprocess
cli = subprocess.run(
    [sys.executable, "-B", os.path.join(SKILL, "devlog.py"),
     "--root", os.path.join(root, "live"), "--archive", "--markdown"],
    capture_output=True, text=True)
check(cli.returncode == 0, f"devlog.py --archive --markdown exits 0\n{cli.stderr}")
check("archived" in cli.stdout and "MISSING" in cli.stdout,
      "CLI reports the inlined count and flags the missing figure")

# ───────────────────────── groundtruth ─────────────────────────
print("\ngroundtruth")
import groundtruth as GT      # noqa: E402

gt_dir = os.path.join(root, "live", "gt")
os.makedirs(os.path.join(gt_dir, "img"), exist_ok=True)
fig(os.path.join(gt_dir, "img", "T1.png"), 7)
gt_page = GT.build(gt_dir, tiles=[{"id": "T1", "file": "img/T1.png", "w": 352, "h": 242,
                                   "um_per_px": 1.7183, "what": "verify fixture"}],
                   question="Outline each object.")
gt_txt = open(gt_page, encoding="utf-8").read()
check(os.path.exists(gt_page), "GT.build writes a page")
check("data:image/png;base64," in gt_txt,
      "the tile is inlined as a data URI - a file:// page cannot getImageData otherwise")
check("img/T1.png" not in gt_txt.replace('"file": "img/T1.png"', ""),
      "no separately-loaded image reference left to taint the canvas")
check("1.7183" in gt_txt, "the physical scale reaches the page, so error can be read in um")

#  A round-trip through the documented JSON shape: tile pixels [x,y] in, (row,col) arrays
#  out. This is the one place the flip happens and it is easy to get backwards.
gt_json = os.path.join(root, "gt_export.json")
with open(gt_json, "w", encoding="utf-8") as f:
    json.dump({"schema": "evidence-log/groundtruth@1", "tiles": {"T1": {
        "id": "T1", "w": 352, "h": 242, "um_per_px": 1.7183, "shapes": [
            #  `p` is [x, y] in TILE PIXELS, per the module's COORDINATES note.
            {"k": "poly", "p": [[10, 20], [40, 20], [40, 60], [10, 60]],
             "c": "F", "a": "F"},
            {"k": "circ", "cx": 200, "cy": 100, "r": 15, "c": "cell"},
            {"k": "rect", "x0": 0, "y0": 0, "x1": 352, "y1": 242}]}}}, f)
try:
    rec = GT.read(gt_json)
    t1 = rec["T1"]
    lab, cls = GT.rasterize(t1)
    check(lab.shape == (242, 352),
          f"rasterize returns (row, col) = (h, w), not (w, h)  [{lab.shape}]")
    check(int(lab.max()) == 1, f"the polygon rasterized ({int(lab.max())} id)")
    #  THE flip: [x,y] in the JSON must come out as (row, col). The polygon spans
    #  x 10..40 and y 20..60, so rows must start near 20 and columns near 10. Getting
    #  this backwards is silent - the array is valid, just transposed.
    ys, xs = np.nonzero(lab == 1)
    check(19 <= ys.min() <= 22 and 9 <= xs.min() <= 12,
          f"[x,y] in the JSON became (row, col) on the way out "
          f"(row {ys.min()}, col {xs.min()}; expected ~20, ~10)")
    check(set(cls.values()) == {"F"}, f"per-id class preserved ({cls})")
    check(len(t1.circles) == 1 and abs(t1.circles[0][0] - 100) < 1e-6,
          f"circle parsed as (cy, cx, r) ({t1.circles})")
    check(len(t1.regions) == 1,
          "the region rectangle survives - without it recall cannot be computed")
except Exception as exc:                              # noqa: BLE001 - this is the report
    check(False, f"GT.read/rasterize round-trip raised {type(exc).__name__}: {exc}")

#  SKILL.md §4 shows `GT.region_mask(...)`. Confirm the documented surface exists rather
#  than discovering mid-analysis that the example does not run.
check(hasattr(GT, "region_mask"),
      "GT.region_mask exists as SKILL.md §4 documents it")

print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED:"))
for f_ in FAILED:
    print("  - " + f_)
sys.exit(1 if FAILED else 0)
