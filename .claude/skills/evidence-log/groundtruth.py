"""Hand-drawn ground truth — the user's eye, as geometry rather than as a category.

WHY THIS EXISTS
    The labelling sheet (`labelsheet.py`) answers *how many* granules are in an object. That
    is a COUNT, and a count is structurally blind to where a boundary lies: a partition can
    score 84% on counts while every divide sits well inside a granule body. That is not
    hypothetical — it happened on this project, and the over-claim ("the partition is
    essentially exact") survived until the user asked a question the metric could not answer.

    A hand-drawn outline is the instrument a count cannot be. It says where the boundary IS,
    so boundary error becomes a distance in micrometres against a known answer rather than a
    comparison against another algorithm. It also gives the split-at-thin-neck move something
    to be right or wrong about: whether two granules meeting at a neck are one object or two
    is a question only the user's eye settles.

THREE PROPERTIES THAT MAKE IT USABLE
    1. THE COLOUR MEANS SOMETHING. The tile is a channel-true composite, so each polygon's
       type defaults to the dominant channel underneath it. The user overrides when the
       colour is wrong — and BOTH the automatic answer and the final one are recorded, so the
       override rate is itself a measurement of how far colour alone gets you.
    2. PARTIAL WORK IS VALID. A "region complete" rectangle marks where every granule was
       drawn. Without it, an un-drawn granule and a missed granule are indistinguishable and
       recall cannot be computed at all. With it, ten minutes of drawing is a usable dataset.
    3. IT IS ONE FILE. Tiles are inlined as data URIs, which is what lets the page read its
       own pixels — a `file://` page cannot call `getImageData` on a separately-loaded image
       in Chrome without tainting the canvas, and the colour default depends on that call.

USAGE
    import sys; sys.path.insert(0, ".claude/skills/evidence-log")
    import groundtruth as GT

    GT.build("CodeLog/live/gt", tiles=[{"id": "M08", "file": "img/M08.png",
                                        "w": 1024, "h": 1024, "um_per_px": 1.7183,
                                        "what": "pure F, MEDIUM"}, ...])

    rec = GT.read("~/Downloads/ground_truth.json")      # -> {tile_id: TileTruth}
    lab, cls = GT.rasterize(rec["M08"])                 # -> label image + per-id class

COORDINATES
    The JSON stores `[x, y]` in TILE PIXELS, which is what a canvas produces and what every
    other annotation format uses. `read()` returns numpy arrays in `(row, col) = (y, x)`
    order, which is what every array in this repo uses. The flip happens in exactly one
    place, here, on purpose.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["build", "read", "rasterize", "TileTruth", "DEFAULT_CLASSES"]

#: The drawing vocabulary. `ch` names the RGB slot a class lives in, which is what makes the
#: type default to the colour; a class with no `ch` can only be chosen by hand. `shape`
#: decides which tool produces it — polygons for bodies, circles for cells.
DEFAULT_CLASSES: List[Dict[str, Any]] = [
    {"key": "F",    "label": "functional granule",  "hint": "red — R-B 571, carries cells",
     "color": "#fb7185", "shape": "polygon", "ch": "r", "hotkey": "1"},
    {"key": "I",    "label": "inert granule",       "hint": "blue — Nile Blue 649",
     "color": "#60a5fa", "shape": "polygon", "ch": "b", "hotkey": "2"},
    {"key": "?",    "label": "cannot tell",         "hint": "ambiguous colour or boundary",
     "color": "#fbbf24", "shape": "polygon", "fallback": True, "hotkey": "3"},
    {"key": "x",    "label": "not a granule",       "hint": "debris, haze, artefact",
     "color": "#a78bfa", "shape": "polygon", "hotkey": "4"},
    {"key": "cell", "label": "cell",                "hint": "green — GFP 499",
     "color": "#34d399", "shape": "circle", "ch": "g", "hotkey": "5"},
]

#: relative margin under which the automatic type is refused and the fallback class used.
#: Deliberately not a hidden constant: it is written into the page and into the export, so a
#: later analysis can re-derive the automatic answer from the recorded channel medians under
#: a different rule without redrawing anything.
AMBIG_MARGIN = 0.25


# ────────────────────────────── build ──────────────────────────────
def _data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext)
    if mime is None:
        raise ValueError(f"unsupported tile format {ext!r} (png or jpeg)")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def build(out_dir: str, tiles: Sequence[Dict[str, Any]], *,
          classes: Sequence[Dict[str, Any]] = tuple(DEFAULT_CLASSES),
          title: str = "Hand-drawn ground truth",
          question: str = "Draw the outline of each granule.",
          note: str = "", json_name: str = "ground_truth.json",
          ambig_margin: float = AMBIG_MARGIN, inline: bool = True,
          prefill: Optional[Dict[str, List[Dict[str, Any]]]] = None,
          review: bool = False, page_name: str = "index.html") -> str:
    """Write ``out_dir/<page_name>`` — a self-contained annotator. Returns the page path.

    Each tile needs ``id``, ``file`` (relative to ``out_dir``), ``w``, ``h`` and
    ``um_per_px``; anything else in the dict is carried through to the export untouched, so
    provenance (which position, which z, how it was scaled) travels with the annotations.

    ``prefill`` seeds each tile with shapes the AGENT produced, and ``review=True`` turns the
    page into a verification pass over them: accept, correct, or reject each one, with the
    verdict recorded. That distinction is the point — a proposal the user accepted and one
    they never looked at are different pieces of evidence, and a tool that cannot tell them
    apart turns unreviewed machine output into "ground truth" by default.

    ``page_name`` exists so a review page can live beside the blank one without colliding:
    browser storage is keyed on the page's path, so two files means two independent sets of
    work and the hand-drawn originals cannot be overwritten by a review session.
    """
    os.makedirs(out_dir, exist_ok=True)
    tl: List[Dict[str, Any]] = []
    for t in tiles:
        rec = dict(t)
        src = os.path.join(out_dir, rec["file"])
        if not os.path.exists(src):
            raise FileNotFoundError(f"tile image missing: {src}")
        rec["src"] = _data_uri(src) if inline else rec["file"]
        tl.append(rec)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"tiles": [{k: v for k, v in t.items() if k != "src"} for t in tl],
                   "classes": list(classes), "ambig_margin": ambig_margin}, f, indent=1)

    cfg = {"tiles": tl, "classes": list(classes), "ambig": ambig_margin,
           "jsonName": json_name, "title": title, "review": bool(review),
           "prefill": {k: list(v) for k, v in (prefill or {}).items()}}
    css = _CSS + "\n".join(
        f'.sw[data-c="{c["key"]}"]{{--k:{c["color"]}}}' for c in classes)
    page = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
            f"<title>{title}</title><style>{css}</style></head><body>\n"
            f"{_BODY}\n<script>const CFG={json.dumps(cfg)};\nconst QUESTION="
            f"{json.dumps(question)};\nconst NOTE={json.dumps(note)};\n{_JS}</script>\n"
            "</body></html>\n")
    out = os.path.join(out_dir, page_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


# ────────────────────────────── read back ──────────────────────────────
class TileTruth:
    """One tile's annotations, in array order.

    ``polys`` are ``(N,2)`` float arrays of ``(row, col)`` vertices — flipped from the
    ``[x, y]`` the page writes, once, here. ``circles`` are ``(row, col, radius)`` in pixels.
    ``regions`` are ``(r0, c0, r1, c1)`` boxes marked exhaustively annotated; anything
    outside every region is UNKNOWN rather than empty, and a recall number that ignores that
    distinction is meaningless.
    """

    def __init__(self, tid: str, meta: Dict[str, Any], shapes: List[Dict[str, Any]]):
        import numpy as np
        self.id = tid
        self.meta = meta
        self.um_per_px = float(meta.get("um_per_px") or 1.0)
        self.shape = (int(meta.get("h") or 0), int(meta.get("w") or 0))
        self.polys: List[Any] = []
        self.poly_cls: List[str] = []
        self.poly_auto: List[Optional[str]] = []
        self.poly_rgb: List[Dict[str, float]] = []
        #: per polygon: None if hand-drawn, else the review verdict of an agent proposal —
        #: 'proposed' (NOT looked at), 'ok', 'fix' or 'no'.
        self.poly_verdict: List[Optional[str]] = []
        self.circles: List[Tuple[float, float, float]] = []
        self.circle_cls: List[str] = []
        self.regions: List[Tuple[float, float, float, float]] = []
        self.notes = str(meta.get("notes") or "")
        for s in shapes:
            k = s.get("k")
            if k == "poly" and len(s.get("p") or ()) >= 3:
                self.polys.append(np.asarray([[p[1], p[0]] for p in s["p"]], dtype=float))
                self.poly_cls.append(str(s.get("c")))
                self.poly_auto.append(s.get("a"))
                self.poly_rgb.append(dict(s.get("m") or {}))
                self.poly_verdict.append(
                    (s.get("v") or "proposed") if s.get("src") == "auto" else None)
            elif k == "circ":
                self.circles.append((float(s["cy"]), float(s["cx"]), float(s["r"])))
                self.circle_cls.append(str(s.get("c")))
            elif k == "rect":
                y0, y1 = sorted((float(s["y0"]), float(s["y1"])))
                x0, x1 = sorted((float(s["x0"]), float(s["x1"])))
                self.regions.append((y0, x0, y1, x1))

    @property
    def n_override(self) -> int:
        """Polygons whose type the user changed away from the colour's answer.

        The interesting number, not a diagnostic: it is the error rate of channel-dominance
        classification measured against the user's eye, for free.
        """
        return sum(1 for c, a in zip(self.poly_cls, self.poly_auto) if a is not None and a != c)

    def confirmed(self, *, include_fixed: bool = True) -> List[int]:
        """Indices of polygons that carry the user's authority — hand-drawn, or a proposal
        they explicitly accepted or corrected.

        A proposal still marked ``proposed`` is the AGENT'S OUTPUT that nobody has checked.
        Scoring against it would be scoring against yourself, and it is the single easiest
        way to manufacture a number that means nothing, so it is excluded here by default
        rather than left to each caller to remember.
        """
        ok = {"ok", "fix"} if include_fixed else {"ok"}
        return [i for i, v in enumerate(self.poly_verdict) if v is None or v in ok]

    def verdict_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.poly_verdict:
            k = "hand" if v is None else v
            out[k] = out.get(k, 0) + 1
        return out

    def __repr__(self) -> str:
        return (f"<TileTruth {self.id}: {len(self.polys)} polygons, "
                f"{len(self.circles)} circles, {len(self.regions)} regions, "
                f"{self.n_override} overrides, {self.verdict_counts()}>")


def read(path: str, *, manifest: Optional[str] = None) -> Dict[str, TileTruth]:
    """Load a saved ground-truth JSON into ``{tile_id: TileTruth}``."""
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        doc = json.load(f)
    if not str(doc.get("schema", "")).startswith("evidence-log/groundtruth"):
        raise ValueError(f"{path}: not a ground-truth export (schema={doc.get('schema')!r})")
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    if manifest:
        with open(os.path.expanduser(manifest), encoding="utf-8") as f:
            for t in json.load(f).get("tiles", []):
                meta_by_id[str(t.get("id"))] = t
    out: Dict[str, TileTruth] = {}
    for tid, rec in (doc.get("tiles") or {}).items():
        meta = dict(meta_by_id.get(tid, {}))
        meta.update({k: v for k, v in rec.items() if k != "shapes"})
        out[tid] = TileTruth(tid, meta, rec.get("shapes") or [])
    return out


def rasterize(truth: TileTruth, *, shape: Optional[Tuple[int, int]] = None,
              classes: Optional[Sequence[str]] = None, fill_holes: bool = False):
    """``(labels, {label_id: class_key})`` — the polygons as an integer label image.

    Later polygons overwrite earlier ones where they overlap, which matches how they were
    drawn: a granule outlined on top of another was seen on top of it.

    ``classes=None`` keeps every polygon. Pass an explicit tuple to filter — the old
    default of ``("F", "I", "?")`` was the granule project's vocabulary and silently
    returned an EMPTY raster for any page built with different class keys.
    """
    import numpy as np
    from skimage.draw import polygon as _poly
    hw = shape or truth.shape
    if not hw or not all(hw):
        raise ValueError(f"{truth.id}: no tile shape; pass shape=(h, w)")
    lab = np.zeros(hw, dtype=np.int32)
    cls: Dict[int, str] = {}
    n = 0
    for pts, ck in zip(truth.polys, truth.poly_cls):
        if classes is not None and ck not in classes:
            continue
        n += 1
        rr, cc = _poly(pts[:, 0], pts[:, 1], shape=hw)
        lab[rr, cc] = n
        cls[n] = ck
    if fill_holes:
        from scipy import ndimage as ndi
        for i in range(1, n + 1):
            m = ndi.binary_fill_holes(lab == i)
            lab[m & (lab == 0)] = i
    return lab, cls


def region_mask(truth: TileTruth, *, shape: Optional[Tuple[int, int]] = None):
    """Boolean mask of where the user drew exhaustively — the only valid recall denominator.

    Returns all-True when no region was marked, and callers should treat that as "coverage
    unknown" rather than "the whole tile is complete".
    """
    import numpy as np
    hw = shape or truth.shape
    m = np.zeros(hw, dtype=bool)
    if not truth.regions:
        return np.ones(hw, dtype=bool)
    for y0, x0, y1, x1 in truth.regions:
        m[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
    return m


_CSS = """
*{box-sizing:border-box}
:root{--bg:#0b0f17;--panel:#111827;--line:#1f2937;--fg:#e5e7eb;--dim:#9ca3af;--ok:#34d399;
 --mono:ui-monospace,Menlo,Consolas,monospace}
:root[data-theme=light]{--bg:#f8fafc;--panel:#fff;--line:#e2e8f0;--fg:#0f172a;--dim:#52627a}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);overflow:hidden;
 font:13px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;display:flex;flex-direction:column}
header{border-bottom:1px solid var(--line);padding:8px 14px;display:flex;gap:12px;
 align-items:center;flex-wrap:wrap;flex:0 0 auto}
h1{font-size:14px;margin:0;white-space:nowrap}
.sub{color:var(--dim);font-size:11.5px}
.grow{flex:1}
button,select{background:transparent;border:1px solid var(--line);color:var(--fg);
 border-radius:7px;padding:5px 9px;font:11.5px var(--mono);cursor:pointer}
button:hover{border-color:var(--fg)}
button.on{border-color:var(--ok);color:var(--ok)}
button.pri{border-color:var(--ok);color:var(--ok)}
main{flex:1;display:flex;min-height:0}
#side{width:252px;flex:0 0 auto;border-right:1px solid var(--line);overflow:auto;
 padding:11px 12px 40px}
#stage{flex:1;position:relative;min-width:0;background:#000}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair}
.sec{font:10px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
 margin:15px 0 6px}
.sec:first-child{margin-top:0}
.tiles{display:flex;flex-direction:column;gap:4px}
.tb{display:flex;justify-content:space-between;align-items:center;gap:7px;text-align:left;
 width:100%;padding:6px 8px;font:11px var(--mono)}
.tb small{color:var(--dim);font-size:9.5px;display:block;font-weight:400}
.tb .n{color:var(--ok);font-size:10.5px}
.sw{display:flex;align-items:center;gap:8px;width:100%;padding:6px 8px;margin-bottom:4px;
 border-left:4px solid var(--k);text-align:left}
.sw.on{border-color:var(--line);border-left-color:var(--k);background:var(--panel)}
.sw.pin{border-color:var(--k);box-shadow:inset 0 0 0 1px var(--k)}
.sw b{color:var(--k);font-size:11.5px;min-width:30px}
.sw span{color:var(--dim);font-size:10px;line-height:1.3;flex:1}
.sw kbd{margin-left:auto;font:9.5px var(--mono);border:1px solid var(--line);
 border-radius:4px;padding:1px 4px;color:var(--dim)}
.tools{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.rv{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:7px}
.rv button:first-child{grid-column:1/-1;border-color:#38bdf8;color:#38bdf8}
#rok{border-color:var(--ok);color:var(--ok)}
#rno{border-color:#fb7185;color:#fb7185}
kbd{font:9.5px var(--mono);border:1px solid var(--line);border-radius:4px;padding:0 3px;
 color:var(--dim);margin-left:4px}
.chip{display:flex;gap:4px}
.chip button{flex:1;padding:4px 0;font-size:10.5px}
#hud{position:absolute;left:10px;top:10px;z-index:5;background:#000a;border-radius:8px;
 padding:7px 10px;font:11px var(--mono);color:var(--fg);pointer-events:none;
 border:1px solid var(--line);max-width:60%}
#hud b{color:var(--ok)}
#hint{position:absolute;left:10px;bottom:10px;z-index:5;background:#000a;border-radius:8px;
 padding:6px 10px;font:11px var(--mono);color:var(--dim);pointer-events:none;
 border:1px solid var(--line)}
#fl{position:fixed;left:50%;transform:translateX(-50%);bottom:26px;z-index:40;
 background:var(--ok);color:#04140d;border-radius:8px;padding:9px 15px;font:12px var(--mono);
 opacity:0;transition:opacity .25s;pointer-events:none;max-width:70vw;text-align:center}
label.rng{display:block;font:10px var(--mono);color:var(--dim);margin:8px 0 2px}
input[type=range]{width:100%}
textarea{width:100%;min-height:54px;background:var(--bg);color:var(--fg);
 border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:12px/1.45 inherit;
 resize:vertical}
.tally{font:10.5px var(--mono);color:var(--dim);line-height:1.7}
.tally b{color:var(--fg)}
.warn{border:1px solid #fbbf24;color:#fbbf24;border-radius:7px;padding:7px 9px;
 font:10.5px var(--mono);line-height:1.5;margin-top:9px}
"""

_BODY = """
<header>
  <h1 id="ttl"></h1><span class="sub" id="sub"></span>
  <span class="grow"></span>
  <button id="undo" title="ctrl+z">undo</button>
  <button id="imp">load file</button>
  <button id="sv" class="pri">save ground truth</button>
  <button id="cp">copy for Claude</button>
  <button id="tt">theme</button>
  <input id="fi" type="file" accept="application/json" style="display:none">
</header>
<main>
  <div id="side">
    <div id="rev" style="display:none">
      <div class="sec">review my proposals</div>
      <div class="rv">
        <button id="rnext">next suspicious <kbd>n</kbd></button>
        <button id="rok">correct <kbd>a</kbd></button>
        <button id="rno">wrong <kbd>d</kbd></button>
      </div>
      <div class="tally" id="rtal"></div>
      <button id="racc" style="width:100%;margin-top:7px">accept the rest of this field</button>
      <label style="display:flex;gap:6px;align-items:center;margin-top:7px;
             font:10.5px var(--mono);color:var(--dim)">
        <input type="checkbox" id="rauto" checked> jump to the next one automatically</label>
    </div>
    <div class="sec">fields</div><div class="tiles" id="tiles"></div>
    <div class="sec">tool</div>
    <div class="tools">
      <button data-t="poly" class="on">polygon <kbd>p</kbd></button>
      <button data-t="circ">circle <kbd>c</kbd></button>
      <button data-t="rect">region <kbd>r</kbd></button>
      <button data-t="sel">select <kbd>v</kbd></button>
    </div>
    <div class="sec" id="csec">type — defaults to the colour</div>
    <div id="cls"></div>
    <div class="sec">display only — never changes what is saved</div>
    <div class="chip" id="chan"></div>
    <label class="rng">brightness <span id="bv">1.0&times;</span></label>
    <input type="range" id="br" min="30" max="400" value="100">
    <div class="sec">this field</div>
    <div class="tally" id="tal"></div>
    <div class="sec">notes for this field</div>
    <textarea id="nt" placeholder="anything I should know — exported with the shapes"></textarea>
    <div class="warn" id="warn"></div>
  </div>
  <div id="stage">
    <canvas id="cv"></canvas>
    <div id="hud"></div><div id="hint"></div>
  </div>
</main>
<div id="fl"></div>
"""

_JS = r"""
// ─────────────── state ───────────────
const CLS = CFG.classes, TILES = CFG.tiles;
const CLSBY = Object.fromEntries(CLS.map(c => [c.key, c]));
const POLYCLS = CLS.filter(c => c.shape === 'polygon');
const CIRCCLS = CLS.filter(c => c.shape === 'circle');
const FALLBACK = (CLS.find(c => c.fallback) || POLYCLS[POLYCLS.length - 1]).key;
const KEY = '_gt3_' + location.pathname + '_';
const SNAP_PX = 5;                 // screen px within which a corner grabs a neighbour's
const S = {
  ti: 0, tool: 'poly', cls: POLYCLS[0].key, pin: null, sel: null, draft: null, drag: null,
  z: 1, ox: 0, oy: 0, pan: null, space: false,
  chan: { r: true, g: true, b: true }, bright: 1, autoNext: true,
  shapes: {}, notes: {}, undo: {}, nextId: {},
};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let srcCan = null, srcData = null, disp = null;   // per-tile pixel caches

const tile = () => TILES[S.ti];
const shapes = () => (S.shapes[tile().id] ||= []);
function loadAll() {
  TILES.forEach(t => {
    let o = null;
    try { const raw = localStorage.getItem(KEY + t.id); o = raw ? JSON.parse(raw) : null; }
    catch (e) { o = null; }
    // Proposals seed a tile only when there is NO saved work for it. Any session that has
    // already touched this tile wins — reseeding would silently throw away review verdicts.
    S.shapes[t.id] = (o && o.shapes) || JSON.parse(JSON.stringify(CFG.prefill[t.id] || []));
    S.notes[t.id] = (o && o.notes) || '';
    S.nextId[t.id] = 1 + Math.max(0, ...S.shapes[t.id].map(s => s.i || 0));
    S.undo[t.id] = [];
  });
}
const isProp = s => s.src === 'auto' && (s.v === 'proposed' || s.v === undefined);
const reviewed = s => s.src === 'auto' && s.v && s.v !== 'proposed';
function verdict(s, v) {
  if (s.src !== 'auto') return;
  s.v = v;
  persist();
}
function persist() {
  const t = tile();
  try {
    localStorage.setItem(KEY + t.id,
      JSON.stringify({ shapes: S.shapes[t.id], notes: S.notes[t.id] || '' }));
  } catch (e) { flash('browser storage is full — use "save ground truth" now'); }
  tally();
}
function snap() {
  const t = tile().id;
  S.undo[t].push(JSON.stringify(S.shapes[t]));
  if (S.undo[t].length > 60) S.undo[t].shift();
}
function undo() {
  const t = tile().id, u = S.undo[t];
  if (!u.length) { flash('nothing to undo on this field'); return; }
  S.shapes[t] = JSON.parse(u.pop()); S.sel = null; S.draft = null; persist(); draw();
}

// ─────────────── image: decode once, re-composite for display ───────────────
function setTile(i) {
  S.ti = i; S.sel = null; S.draft = null;
  const t = tile();
  const im = new Image();
  im.onload = () => {
    srcCan = document.createElement('canvas');
    srcCan.width = im.naturalWidth; srcCan.height = im.naturalHeight;
    const c = srcCan.getContext('2d', { willReadFrequently: true });
    c.drawImage(im, 0, 0);
    try {
      srcData = c.getImageData(0, 0, srcCan.width, srcCan.height);
    } catch (e) {
      srcData = null;                       // tainted: colour default degrades to manual
      flash('cannot read pixels — the type will not default to the colour');
    }
    disp = document.createElement('canvas');
    disp.width = srcCan.width; disp.height = srcCan.height;
    recomposite(); fit(); syncUI();
  };
  im.src = t.src;
}
function recomposite() {
  if (!disp) return;
  const d = disp.getContext('2d');
  if (!srcData) { d.drawImage(srcCan, 0, 0); draw(); return; }
  const n = srcData.data.length, out = d.createImageData(disp.width, disp.height);
  const a = srcData.data, o = out.data, g = S.bright;
  const kr = S.chan.r ? g : 0, kg = S.chan.g ? g : 0, kb = S.chan.b ? g : 0;
  for (let i = 0; i < n; i += 4) {
    o[i] = Math.min(255, a[i] * kr);
    o[i + 1] = Math.min(255, a[i + 1] * kg);
    o[i + 2] = Math.min(255, a[i + 2] * kb);
    o[i + 3] = 255;
  }
  d.putImageData(out, 0, 0);
  draw();
}

// ─────────────── view ───────────────
function resize() {
  const r = cv.parentElement.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  draw();
}
function fit() {
  const r = cv.parentElement.getBoundingClientRect(), t = tile();
  S.z = Math.min(r.width / t.w, r.height / t.h);
  S.ox = (r.width - t.w * S.z) / 2; S.oy = (r.height - t.h * S.z) / 2;
  draw();
}
const toImg = (e) => {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left - S.ox) / S.z, (e.clientY - r.top - S.oy) / S.z];
};

function draw() {
  if (!disp) return;
  const dpr = devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.save();
  ctx.translate(S.ox, S.oy); ctx.scale(S.z, S.z);
  ctx.imageSmoothingEnabled = S.z < 1;
  ctx.drawImage(disp, 0, 0);
  const lw = 1.6 / S.z;
  for (const s of shapes()) drawShape(s, lw, s === S.sel);
  if (S.draft) drawDraft(lw);
  if (S.snapTgt) {   // the ring = "your next click lands HERE, not under the crosshair"
    ctx.lineWidth = 1.4 / S.z; ctx.strokeStyle = '#ffffff'; ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(S.snapTgt[0], S.snapTgt[1], 6 / S.z, 0, 6.2832); ctx.stroke();
  }
  ctx.restore();
  scalebar(); hud();
}
function colorOf(s) { return (CLSBY[s.c] || { color: '#fff' }).color; }
function drawShape(s, lw, sel) {
  ctx.lineWidth = sel ? lw * 2 : lw;
  ctx.strokeStyle = colorOf(s);
  ctx.fillStyle = colorOf(s) + (sel ? '38' : '1e');
  // In a review pass the verdict has to be visible at a glance, or there is no way to see
  // what is left to do: dashed = still mine, solid = you accepted it, thick = you fixed it,
  // ghost = you rejected it (kept, not deleted — a wrong proposal is evidence too).
  if (s.src === 'auto') {
    if (s.v === 'no') { ctx.globalAlpha = 0.22; ctx.setLineDash([2 / S.z, 4 / S.z]); }
    else if (isProp(s)) { ctx.setLineDash([6 / S.z, 4 / S.z]); ctx.fillStyle = 'transparent'; }
    else if (s.v === 'fix') { ctx.lineWidth = (sel ? lw * 2 : lw) * 1.7; }
  }
  ctx.beginPath();
  if (s.k === 'poly') {
    s.p.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
  } else if (s.k === 'circ') {
    ctx.arc(s.cx, s.cy, Math.max(s.r, 0.5), 0, 6.2832);
  } else {
    ctx.setLineDash([7 / S.z, 5 / S.z]);
    ctx.rect(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
    ctx.strokeStyle = '#e5e7eb'; ctx.fillStyle = '#ffffff10';
  }
  ctx.fill(); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
  if (sel && s.k === 'poly') {
    ctx.fillStyle = '#fff';
    for (const p of s.p) { ctx.beginPath(); ctx.arc(p[0], p[1], 3 / S.z, 0, 6.3); ctx.fill(); }
  }
}
function drawDraft(lw) {
  const d = S.draft;
  ctx.lineWidth = lw; ctx.strokeStyle = '#ffffff'; ctx.setLineDash([5 / S.z, 4 / S.z]);
  ctx.beginPath();
  if (d.k === 'poly') {
    d.p.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    if (d.cur) ctx.lineTo(d.cur[0], d.cur[1]);
  } else if (d.k === 'circ') {
    ctx.arc(d.cx, d.cy, Math.max(d.r, 0.5), 0, 6.2832);
  } else {
    ctx.rect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0);
  }
  ctx.stroke(); ctx.setLineDash([]);
  if (d.k === 'poly') {
    ctx.fillStyle = '#fff';
    for (const p of d.p) { ctx.beginPath(); ctx.arc(p[0], p[1], 3 / S.z, 0, 6.3); ctx.fill(); }
  }
}
function scalebar() {
  const t = tile(), dpr = devicePixelRatio || 1;
  let um = 100, px = um / t.um_per_px * S.z;
  while (px > 340) { um /= 2; px /= 2; }
  while (px < 34) { um *= 2; px *= 2; }
  // TOP right: the hint line owns the whole bottom edge and the HUD the top left, so this
  // is the only corner where the bar is not sitting on top of text.
  const y = 34, x = cv.width / dpr - px - 18;
  ctx.fillStyle = '#000a'; ctx.fillRect(x - 8, y - 17, px + 16, 26);
  ctx.fillStyle = '#fff'; ctx.fillRect(x, y, px, 3);
  ctx.font = '11px ui-monospace,monospace';
  ctx.fillText(um >= 1 ? um + ' µm' : um.toFixed(1) + ' µm', x, y - 5);
}
function hud() {
  const t = tile(), n = shapes();
  const np = n.filter(s => s.k === 'poly').length, nc = n.filter(s => s.k === 'circ').length;
  const nr = n.filter(s => s.k === 'rect').length;
  document.getElementById('hud').innerHTML =
    `<b>${t.id}</b> ${t.what || ''} &middot; z=${t.z ?? '?'} t=${t.t ?? '?'} &middot; ` +
    `${(S.z * 100).toFixed(0)}% &middot; <b>${np}</b> outlines &middot; <b>${nc}</b> cells` +
    (nr ? ` &middot; ${nr} region${nr > 1 ? 's' : ''} marked` : '') +
    (t.empty_channels && t.empty_channels.length
      ? `<br><span style="color:#fbbf24">no ${t.empty_channels.map(
        c => ({ r: 'F', g: 'cell', b: 'inert' }[c] || c)).join('/')} signal in this field` +
      ` — by design</span>` : '');
  document.getElementById('hint').textContent = {
    poly: 'click each corner · click the first corner again (or enter / double-click) to close · '
      + 'corners snap to a neighbour\'s (ring shows the grab; hold Alt to place exactly) · '
      + 'backspace undoes a corner · esc cancels',
    circ: 'press at the centre of a cell and drag out · esc cancels',
    rect: 'drag a box around the area where you outlined EVERY granule',
    sel: 'click a shape · drag it or its corners · 1-5 retypes · delete removes',
  }[S.tool] + (CFG.review
    ? '  ·  REVIEW: n = next suspicious · a = correct · d = wrong · drag a corner to fix it'
    : '  ·  space+drag or middle-drag pans · wheel zooms');
}

// ─────────────── the colour default ───────────────
function medianRGB(test, x0, y0, x1, y1) {
  if (!srcData) return null;
  const W = srcData.width, H = srcData.height, d = srcData.data;
  const ax = Math.max(0, Math.floor(x0)), bx = Math.min(W - 1, Math.ceil(x1));
  const ay = Math.max(0, Math.floor(y0)), by = Math.min(H - 1, Math.ceil(y1));
  const R = [], G = [], B = [];
  const step = Math.max(1, Math.floor(Math.sqrt((bx - ax + 1) * (by - ay + 1) / 4000)));
  for (let y = ay; y <= by; y += step) for (let x = ax; x <= bx; x += step) {
    if (!test(x + 0.5, y + 0.5)) continue;
    const i = (y * W + x) * 4;
    R.push(d[i]); G.push(d[i + 1]); B.push(d[i + 2]);
  }
  if (R.length < 4) return null;
  const md = a => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
  return { r: md(R), g: md(G), b: md(B), n: R.length };
}
function inPoly(p) {
  return (x, y) => {
    let hit = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
      const [xi, yi] = p[i], [xj, yj] = p[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
}
/* The type defaults to whichever channel is strongest under the shape, and REFUSES to guess
   when the top two are within CFG.ambig of each other — an ambiguous granule should arrive
   as a question rather than as a confident wrong answer. Both the automatic type and the
   channel medians are recorded, so the rule can be re-derived offline without redrawing. */
function autoClass(m, pool) {
  if (!m) return null;
  const cand = pool.filter(c => c.ch);
  if (!cand.length) return null;
  const sc = cand.map(c => ({ k: c.key, v: m[c.ch] })).sort((a, b) => b.v - a.v);
  if (sc[0].v < 8) return FALLBACK;
  if (sc.length > 1 && (sc[0].v - sc[1].v) / Math.max(sc[0].v, 1) < CFG.ambig) return FALLBACK;
  return sc[0].k;
}

// ─────────────── shape creation ───────────────
function bboxOf(p) {
  const xs = p.map(q => q[0]), ys = p.map(q => q[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
/* Snap a new corner onto a neighbouring outline's existing corner.
   These granules are packed edge to edge, so most boundaries are drawn TWICE — once from
   each side. Without snapping the two versions differ by a few pixels and the gap between
   them is a sliver of no-man's-land that is neither granule, which then shows up as
   boundary error that the annotator invented rather than the algorithm. Snapping makes a
   shared edge exactly shared. */
function snapCandidate(x, y) {
  const e = SNAP_PX / S.z;
  let best = null, bd = e;
  for (const s of shapes()) {
    if (s.k !== 'poly') continue;
    for (const p of s.p) {
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d < bd) { bd = d; best = p; }
    }
  }
  return best;
}
// A snapped point lands AWAY from the crosshair, so the grab must be visible before the
// click (the ring drawn in draw()) and escapable (Alt places exactly at the cursor).
// Invisible snapping read as a broken crosshair to the first real user.
function snapVertex(x, y, alt) {
  if (alt) return [x, y];
  const best = snapCandidate(x, y);
  return best ? [best[0], best[1]] : [x, y];
}
function commitPoly(p) {
  if (p.length < 3) return;
  const bb = bboxOf(p);
  const m = medianRGB(inPoly(p), bb[0], bb[1], bb[2], bb[3]);
  const auto = autoClass(m, POLYCLS);
  snap();
  // Pinning wins over the colour. "Defaults to the colour but I can override" has to work
  // BEFORE the shape exists as well as after it, or every polygon of a type the colour gets
  // wrong has to be drawn and then corrected one at a time.
  const s = {
    i: S.nextId[tile().id]++, k: 'poly', c: S.pin || auto || S.cls, a: auto,
    p: p.map(q => [+q[0].toFixed(2), +q[1].toFixed(2)]), m: m || undefined,
  };
  if (S.pin) s.pin = true;
  shapes().push(s); S.sel = s; persist(); draw();
}
function commitCirc(cx, cy, r) {
  if (r < 0.75) return;
  const m = medianRGB((x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r,
    cx - r, cy - r, cx + r, cy + r);
  snap();
  const s = {
    i: S.nextId[tile().id]++, k: 'circ', c: CIRCCLS.length ? CIRCCLS[0].key : S.cls,
    a: null, cx: +cx.toFixed(2), cy: +cy.toFixed(2), r: +r.toFixed(2), m: m || undefined,
  };
  shapes().push(s); S.sel = s; persist(); draw();
}
function commitRect(x0, y0, x1, y1) {
  if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) return;
  snap();
  const s = {
    i: S.nextId[tile().id]++, k: 'rect', c: '_region', a: null,
    x0: Math.min(x0, x1), y0: Math.min(y0, y1), x1: Math.max(x0, x1), y1: Math.max(y0, y1),
  };
  shapes().push(s); S.sel = s; persist(); draw();
}

// ─────────────── hit testing ───────────────
function hit(x, y) {
  const sh = shapes();
  for (let i = sh.length - 1; i >= 0; i--) {
    const s = sh[i];
    if (s.k === 'poly' && inPoly(s.p)(x, y)) return s;
    if (s.k === 'circ' && (x - s.cx) ** 2 + (y - s.cy) ** 2 <= s.r * s.r) return s;
    if (s.k === 'rect' && x >= s.x0 && x <= s.x1 && y >= s.y0 && y <= s.y1) {
      const e = 6 / S.z;
      if (Math.abs(x - s.x0) < e || Math.abs(x - s.x1) < e ||
        Math.abs(y - s.y0) < e || Math.abs(y - s.y1) < e) return s;
    }
  }
  return null;
}
function vertexAt(s, x, y) {
  if (!s || s.k !== 'poly') return -1;
  const e = 7 / S.z;
  for (let i = 0; i < s.p.length; i++)
    if (Math.abs(s.p[i][0] - x) < e && Math.abs(s.p[i][1] - y) < e) return i;
  return -1;
}

// ─────────────── pointer ───────────────
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('pointerdown', e => {
  cv.setPointerCapture(e.pointerId);
  const [x, y] = toImg(e);
  if (e.button === 1 || S.space) { S.pan = [e.clientX, e.clientY, S.ox, S.oy]; return; }
  if (S.tool === 'poly') {
    if (e.button === 2) { if (S.draft) { commitPoly(S.draft.p); S.draft = null; } return; }
    S.draft = S.draft || { k: 'poly', p: [] };
    const d = S.draft, e2 = SNAP_PX / S.z;
    // clicking the first corner again closes the outline — the gesture everyone expects
    if (d.p.length >= 3 && Math.hypot(d.p[0][0] - x, d.p[0][1] - y) < e2) {
      commitPoly(d.p); S.draft = null; syncUI(); return;
    }
    d.p.push(snapVertex(x, y, e.altKey)); d.cur = [x, y]; draw(); return;
  }
  if (e.button === 2) return;
  if (S.tool === 'circ') { S.draft = { k: 'circ', cx: x, cy: y, r: 0 }; return; }
  if (S.tool === 'rect') { S.draft = { k: 'rect', x0: x, y0: y, x1: x, y1: y }; return; }
  const s = hit(x, y);
  S.sel = s;
  if (s) {
    const vi = vertexAt(s, x, y);
    S.drag = { s, vi, x, y, o: JSON.parse(JSON.stringify(s)) };
    snap();
  }
  syncUI(); draw();
});
cv.addEventListener('pointermove', e => {
  const [x, y] = toImg(e);
  if (S.pan) {
    S.ox = S.pan[2] + (e.clientX - S.pan[0]); S.oy = S.pan[3] + (e.clientY - S.pan[1]);
    draw(); return;
  }
  // show where the NEXT click will land before it happens (Alt suppresses the snap)
  if (S.tool === 'poly' && !S.drag) {
    const tgt = e.altKey ? null : snapCandidate(x, y);
    const changed = JSON.stringify(tgt) !== JSON.stringify(S.snapTgt);
    S.snapTgt = tgt;
    if (changed && !S.draft) draw();
  } else if (S.snapTgt) { S.snapTgt = null; draw(); }
  if (S.draft) {
    const d = S.draft;
    if (d.k === 'poly') d.cur = [x, y];
    else if (d.k === 'circ') d.r = Math.hypot(x - d.cx, y - d.cy);
    else { d.x1 = x; d.y1 = y; }
    draw(); return;
  }
  if (S.drag) {
    const { s, vi, o } = S.drag, dx = x - S.drag.x, dy = y - S.drag.y;
    if (s.k === 'poly') {
      if (vi >= 0) { s.p[vi] = [o.p[vi][0] + dx, o.p[vi][1] + dy]; }
      else s.p = o.p.map(p => [p[0] + dx, p[1] + dy]);
    } else if (s.k === 'circ') { s.cx = o.cx + dx; s.cy = o.cy + dy; }
    else { s.x0 = o.x0 + dx; s.x1 = o.x1 + dx; s.y0 = o.y0 + dy; s.y1 = o.y1 + dy; }
    draw();
  }
});
addEventListener('pointerup', e => {
  if (S.pan) { S.pan = null; return; }
  const d = S.draft;
  if (d && d.k === 'circ') { commitCirc(d.cx, d.cy, d.r); S.draft = null; }
  else if (d && d.k === 'rect') { commitRect(d.x0, d.y0, d.x1, d.y1); S.draft = null; }
  if (S.drag) {
    // moving a proposal's geometry IS the correction — no separate "mark as fixed" step,
    // which would be a step everyone forgets and would leave the record wrong.
    const s = S.drag.s;
    if (s.src === 'auto' && JSON.stringify(s) !== JSON.stringify(S.drag.o)) s.v = 'fix';
    S.drag = null; persist(); syncUI();
  }
});
cv.addEventListener('dblclick', e => {
  if (S.tool === 'poly' && S.draft) {
    S.draft.p.pop();                      // the dblclick's own extra vertex
    commitPoly(S.draft.p); S.draft = null; syncUI();
  }
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = Math.exp(-e.deltaY * 0.0016), nz = Math.max(0.05, Math.min(60, S.z * f));
  S.ox = mx - (mx - S.ox) * (nz / S.z); S.oy = my - (my - S.oy) * (nz / S.z);
  S.z = nz; draw();
}, { passive: false });

// ─────────────── keys ───────────────
addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === ' ') { S.space = true; cv.style.cursor = 'grab'; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { undo(); e.preventDefault(); return; }
  const k = e.key.toLowerCase();
  if (k === 'escape') { S.draft = null; S.sel = null; syncUI(); draw(); return; }
  if (k === 'enter' && S.draft && S.draft.k === 'poly') {
    commitPoly(S.draft.p); S.draft = null; syncUI(); return;
  }
  if (k === 'backspace' || k === 'delete') {
    if (S.draft && S.draft.k === 'poly' && S.draft.p.length) { S.draft.p.pop(); draw(); }
    else if (S.sel) {
      snap();
      // A rejected PROPOSAL is kept as a ghost rather than deleted: "I put an outline here
      // and it should not exist" is a false positive, which is exactly the thing a recall
      // count cannot see. Delete again to remove it for good.
      if (S.sel.src === 'auto' && isProp(S.sel)) { verdict(S.sel, 'no'); rejump(); }
      else { S.shapes[tile().id] = shapes().filter(s => s !== S.sel); S.sel = null; }
      persist(); draw();
    }
    e.preventDefault(); return;
  }
  if (CFG.review) {
    if (k === 'a' && S.sel) { snap(); verdict(S.sel, 'ok'); rejump(); draw(); return; }
    if (k === 'd' && S.sel) { snap(); verdict(S.sel, 'no'); rejump(); draw(); return; }
    if (k === 'n') { jumpNext(); return; }
  }
  if ('pcrv'.includes(k) && !e.ctrlKey) {
    setTool({ p: 'poly', c: 'circ', r: 'rect', v: 'sel' }[k]); return;
  }
  if (k === 'f') { fit(); return; }
  const c = CLS.find(x => x.hotkey === e.key);
  if (c) { pick(c.key); return; }
  if (k === '0') { S.pin = null; syncUI(); flash('type comes from the colour again'); return; }
  if (k === 'arrowright' || k === 'arrowleft') {
    setTile((S.ti + (k === 'arrowright' ? 1 : TILES.length - 1)) % TILES.length);
    e.preventDefault();
  }
});
addEventListener('keyup', e => {
  if (e.key === ' ') { S.space = false; cv.style.cursor = 'crosshair'; }
});

// ─────────────── UI ───────────────
function setTool(t) {
  S.tool = t; S.draft = null;
  document.querySelectorAll('.tools button').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  hud(); draw();
}
/* Review order is by SUSPICION, not by position. `q` is how unlike a single convex body a
   proposal is, so the ones most likely to be wrong come first and a partial pass is still
   worth having — the same reason the labelling sheet stratifies instead of sorting. */
function jumpNext() {
  const todo = shapes().filter(isProp).sort((a, b) => (b.q || 0) - (a.q || 0));
  if (!todo.length) { flash('every proposal on this field has a verdict'); return; }
  const s = todo[0];
  S.sel = s; setTool('sel');
  const bb = bboxOf(s.p || [[s.cx, s.cy]]);
  const r = cv.parentElement.getBoundingClientRect();
  const w = Math.max(bb[2] - bb[0], bb[3] - bb[1], 30);
  S.z = Math.max(0.3, Math.min(9, 0.34 * Math.min(r.width, r.height) / w));
  S.ox = r.width / 2 - (bb[0] + bb[2]) / 2 * S.z;
  S.oy = r.height / 2 - (bb[1] + bb[3]) / 2 * S.z;
  syncUI(); draw();
}
function rejump() { if (S.autoNext) setTimeout(jumpNext, 60); }
function acceptRest() {
  const todo = shapes().filter(isProp);
  if (!todo.length) { flash('nothing left unreviewed on this field'); return; }
  if (!confirm(`Mark the remaining ${todo.length} proposals on this field as CORRECT?\n\n` +
    `Only do this once you have looked over the field — accepting unseen proposals is how ` +
    `machine output turns into "ground truth" without anyone checking it.`)) return;
  snap(); todo.forEach(s => { s.v = 'ok'; }); persist(); S.sel = null; draw(); syncUI();
  flash(`accepted ${todo.length}`);
}
function syncUI() {
  document.querySelectorAll('#tiles .tb').forEach((b, i) => b.classList.toggle('on', i === S.ti));
  const act = S.sel ? S.sel.c : (S.pin || S.cls);
  document.querySelectorAll('#cls .sw').forEach(b => {
    b.classList.toggle('on', b.dataset.c === act);
    b.classList.toggle('pin', b.dataset.c === S.pin);
  });
  document.getElementById('csec').innerHTML = S.pin
    ? `type — <span style="color:${CLSBY[S.pin].color}">pinned to ${S.pin}</span> ` +
    `<span style="text-transform:none">(0 releases)</span>`
    : 'type — defaults to the colour';
  document.getElementById('nt').value = S.notes[tile().id] || '';
  tally(); hud();
}
function tally() {
  const rows = [];
  let gtot = 0;
  for (const t of TILES) {
    const sh = S.shapes[t.id] || [];
    gtot += sh.filter(s => s.k !== 'rect').length;
    const b = document.querySelector(`#tiles .tb[data-i="${TILES.indexOf(t)}"] .n`);
    if (!b) continue;
    if (CFG.review) {
      const a = sh.filter(s => s.src === 'auto'), d = a.filter(reviewed).length;
      b.textContent = a.length ? `${d}/${a.length}` : '';
      b.style.color = a.length && d === a.length ? 'var(--ok)' : 'var(--dim)';
    } else b.textContent = sh.length ? sh.filter(s => s.k !== 'rect').length : '';
  }
  const sh = shapes();
  for (const c of CLS) {
    const n = sh.filter(s => s.c === c.key).length;
    if (n) rows.push(`<b style="color:${c.color}">${n}</b> ${c.label}${n > 1 ? 's' : ''}`);
  }
  const nr = sh.filter(s => s.k === 'rect').length;
  // only shapes whose type the colour was actually ALLOWED to decide count here — a pinned
  // one never had the colour's answer applied, so scoring it as an override would make the
  // colour look worse the more the user pins.
  const judged = sh.filter(s => s.a && !s.pin);
  const ov = judged.filter(s => s.a !== s.c).length;
  rows.push(nr ? `${nr} complete region${nr > 1 ? 's' : ''} marked`
    : '<span style="color:#fbbf24">no complete region marked</span>');
  if (judged.length)
    rows.push(`colour was right ${(100 * (judged.length - ov) / judged.length).toFixed(0)}% ` +
      `of the time (${ov} override${ov === 1 ? '' : 's'} of ${judged.length})`);
  document.getElementById('tal').innerHTML = rows.join('<br>') || 'nothing drawn yet';
  if (CFG.review) {
    const auto = sh.filter(s => s.src === 'auto');
    const c = { proposed: 0, ok: 0, fix: 0, no: 0 };
    auto.forEach(s => c[isProp(s) ? 'proposed' : s.v]++);
    const done = auto.length - c.proposed;
    let gp = 0, gd = 0;
    for (const t of TILES) {
      const a = (S.shapes[t.id] || []).filter(s => s.src === 'auto');
      gp += a.length; gd += a.filter(reviewed).length;
    }
    document.getElementById('rtal').innerHTML =
      `<b>${done}</b> of <b>${auto.length}</b> reviewed on this field` +
      ` &middot; ${gd}/${gp} overall<br>` +
      `<b style="color:var(--ok)">${c.ok}</b> correct &middot; ` +
      `<b style="color:#38bdf8">${c.fix}</b> corrected &middot; ` +
      `<b style="color:#fb7185">${c.no}</b> wrong` +
      (c.proposed ? `<br><span style="color:#fbbf24">${c.proposed} still mine, unchecked` +
        `</span>` : '');
  }
  document.getElementById('sub').textContent =
    `${QUESTION}  ${gtot} shape${gtot === 1 ? '' : 's'} in total`;
  document.getElementById('warn').innerHTML = nr ? '' :
    'Mark at least one <b>region</b> box (key <b>r</b>) around an area where you outlined ' +
    'EVERY granule. Without it there is no way to tell a granule you missed from one you ' +
    'simply had not got to, and recall cannot be computed at all.';
}

document.getElementById('ttl').textContent = CFG.title;
document.getElementById('tiles').innerHTML = TILES.map((t, i) =>
  `<button class="tb" data-i="${i}"><span>${t.id}<small>${t.what || ''}</small></span>` +
  `<span class="n"></span></button>`).join('');
document.querySelectorAll('#tiles .tb').forEach(b =>
  b.onclick = () => setTile(+b.dataset.i));
document.getElementById('cls').innerHTML = CLS.map(c =>
  `<button class="sw" data-c="${c.key}"><b>${c.key}</b>` +
  `<span>${c.label}<br>${c.hint || ''}</span><kbd>${c.hotkey || ''}</kbd></button>`).join('');
/* One click, two meanings, and they do not conflict: with a shape selected it RETYPES that
   shape; with nothing selected it PINS the type for everything drawn next. Clicking the
   pinned type again releases it back to the colour. */
function pick(key) {
  const c = CLSBY[key];
  if (!c) return;
  if (S.sel && S.sel.k !== 'rect') {
    snap();
    if (S.sel.src === 'auto' && S.sel.c !== key) S.sel.v = 'fix';
    S.sel.c = key; persist();
  } else {
    S.pin = (S.pin === key) ? null : key;
  }
  S.cls = key;
  setTool(c.shape === 'circle' ? 'circ' : (S.tool === 'circ' ? 'poly' : S.tool));
  syncUI(); draw();
}
document.querySelectorAll('#cls .sw').forEach(b => b.onclick = () => pick(b.dataset.c));
document.getElementById('chan').innerHTML = [['r', 'R · F'], ['g', 'G · cells'],
['b', 'B · inert']].map(([k, l]) => `<button data-ch="${k}" class="on">${l}</button>`).join('');
document.querySelectorAll('#chan button').forEach(b => b.onclick = () => {
  S.chan[b.dataset.ch] = !S.chan[b.dataset.ch];
  b.classList.toggle('on', S.chan[b.dataset.ch]); recomposite();
});
document.querySelectorAll('.tools button').forEach(b => b.onclick = () => setTool(b.dataset.t));
const br = document.getElementById('br');
br.oninput = () => {
  S.bright = br.value / 100;
  document.getElementById('bv').textContent = S.bright.toFixed(1) + '×';
  recomposite();
};
const nt = document.getElementById('nt');
nt.addEventListener('input', () => { S.notes[tile().id] = nt.value; persist(); });
document.getElementById('undo').onclick = undo;
if (CFG.review) {
  document.getElementById('rev').style.display = '';
  document.getElementById('rnext').onclick = jumpNext;
  document.getElementById('rok').onclick =
    () => { if (S.sel) { snap(); verdict(S.sel, 'ok'); rejump(); draw(); syncUI(); } };
  document.getElementById('rno').onclick =
    () => { if (S.sel) { snap(); verdict(S.sel, 'no'); rejump(); draw(); syncUI(); } };
  document.getElementById('racc').onclick = acceptRest;
  const ra = document.getElementById('rauto');
  ra.onchange = () => { S.autoNext = ra.checked; };
}

// ─────────────── export / import ───────────────
function doc() {
  const tiles = {};
  for (const t of TILES) {
    const sh = S.shapes[t.id] || [];
    if (!sh.length && !(S.notes[t.id] || '').trim()) continue;
    const meta = {};
    for (const k of ['m', 't', 'z', 'w', 'h', 'um_per_px', 'what', 'file', 'empty_channels'])
      if (t[k] !== undefined) meta[k] = t[k];
    tiles[t.id] = Object.assign(meta, { notes: S.notes[t.id] || '', shapes: sh });
  }
  return {
    schema: 'evidence-log/groundtruth@1', title: CFG.title,
    saved: new Date().toISOString(), ambig_margin: CFG.ambig,
    classes: CLS, tiles,
  };
}
function summary() {
  const d = doc(), L = [];
  let np = 0, nc = 0, nr = 0, ov = 0, au = 0;
  const V = { proposed: 0, ok: 0, fix: 0, no: 0 };
  for (const [id, t] of Object.entries(d.tiles)) {
    const p = t.shapes.filter(s => s.k === 'poly'), c = t.shapes.filter(s => s.k === 'circ');
    const r = t.shapes.filter(s => s.k === 'rect');
    np += p.length; nc += c.length; nr += r.length;
    const hand = p.filter(s => s.src !== 'auto');
    au += hand.filter(s => s.a).length;
    ov += hand.filter(s => s.a && s.a !== s.c).length;
    const auto = t.shapes.filter(s => s.src === 'auto');
    auto.forEach(s => V[(s.v && s.v !== 'proposed') ? s.v : 'proposed']++);
    const by = {};
    p.forEach(s => by[s.c] = (by[s.c] || 0) + 1);
    const rv = auto.length
      ? `; of my ${auto.length} proposals ` +
      `${auto.filter(s => s.v === 'ok').length} correct, ` +
      `${auto.filter(s => s.v === 'fix').length} corrected, ` +
      `${auto.filter(s => s.v === 'no').length} wrong, ` +
      `${auto.filter(s => !s.v || s.v === 'proposed').length} unchecked` : '';
    L.push(`${id}: ${p.length} outlines (${Object.entries(by).map(([k, v]) => k + '=' + v).join(' ')})` +
      `, ${c.length} cells, ${r.length} region${r.length === 1 ? '' : 's'}${rv}` +
      (t.notes ? ` — note: ${t.notes.replace(/\s+/g, ' ').trim()}` : ''));
  }
  if (!L.length) return '';
  const head = CFG.review
    ? `PROPOSAL REVIEW  ${V.ok} correct, ${V.fix} corrected, ${V.no} wrong, ` +
    `${V.proposed} UNCHECKED across ${L.length} field(s). Unchecked ones are still my ` +
    `output and must not be scored as ground truth.`
    : `GROUND TRUTH  ${np} outlines, ${nc} cells, ${nr} complete regions across ` +
    `${L.length} field(s). Colour default agreed with me on ` +
    `${au ? (100 * (au - ov) / au).toFixed(0) : '-'}% of outlines (${ov} overrides).`;
  return head + '\n' + L.join('\n') + `\n\nSaved as ${CFG.jsonName} — read it.`;
}
function flash(m) {
  const f = document.getElementById('fl'); f.textContent = m;
  f.style.opacity = 1; setTimeout(() => { f.style.opacity = 0; }, 3200);
}
document.getElementById('sv').onclick = () => {
  const d = doc();
  if (!Object.keys(d.tiles).length) { flash('nothing drawn yet'); return; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(d)], { type: 'application/json' }));
  a.download = CFG.jsonName; a.click();
  flash('saved as ' + CFG.jsonName + ' — tell me to read it');
};
document.getElementById('cp').onclick = async () => {
  const t = summary();
  if (!t) { flash('nothing drawn yet'); return; }
  try { await navigator.clipboard.writeText(t); }
  catch (e) {
    const a = document.createElement('textarea'); a.value = t;
    document.body.appendChild(a); a.select(); document.execCommand('copy'); a.remove();
  }
  flash('copied — but SAVE the file too, the summary has no outlines in it');
};
document.getElementById('imp').onclick = () => document.getElementById('fi').click();
document.getElementById('fi').onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const d = JSON.parse(rd.result);
      let n = 0;
      for (const [id, t] of Object.entries(d.tiles || {})) {
        if (!TILES.some(x => x.id === id)) continue;
        S.shapes[id] = t.shapes || []; S.notes[id] = t.notes || '';
        S.nextId[id] = 1 + Math.max(0, ...S.shapes[id].map(s => s.i || 0));
        n += S.shapes[id].length;
        localStorage.setItem(KEY + id,
          JSON.stringify({ shapes: S.shapes[id], notes: S.notes[id] }));
      }
      flash(`loaded ${n} shapes`); syncUI(); draw();
    } catch (err) { flash('could not read that file: ' + err.message); }
  };
  rd.readAsText(f);
};
const tb = document.getElementById('tt');
const th = localStorage.getItem('_theme');
if (th) document.documentElement.setAttribute('data-theme', th);
tb.onclick = () => {
  const r = document.documentElement;
  const c = r.getAttribute('data-theme') ||
    (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
  const n = c === 'dark' ? 'light' : 'dark';
  r.setAttribute('data-theme', n); localStorage.setItem('_theme', n);
};
addEventListener('resize', resize);
addEventListener('beforeunload', e => { persist(); });
loadAll(); resize(); setTile(0); setTool('poly');
"""
