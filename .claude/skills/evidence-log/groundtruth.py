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

#: What can be WRONG with a proposal. Review works by exception: the user marks only the
#: bad ones and everything left unmarked in a field they have declared finished counts as
#: correct. That inversion is the difference between a review that gets done and one that
#: does not — object-by-object confirmation of thousands of outlines is not a task anyone
#: completes, and an incomplete pass leaves most of the data in an unusable middle state.
#:
#: The categories are failure MODES rather than a verdict, because each one names a
#: different defect and points at a different fix: `merge` is over-segmentation, `split` is
#: under-segmentation, `expand` is a boundary bias, `wrong` is a false positive. "Bad" alone
#: would collapse four different problems into one number.
#:
#: Note what is deliberately NOT here: "you missed one". Every mark below is a property of an
#: outline that exists, so this vocabulary can only ever measure PRECISION — however many
#: fields are signed off, recall stays unquotable. A miss is a claim about empty space with no
#: shape to attach to, so it is a drawing tool (`m`) rather than a fifth mark, and it counts
#: only inside a `region` box. See :meth:`TileTruth.recall`.
DEFAULT_MARKS: List[Dict[str, Any]] = [
    {"key": "merge",  "label": "merge with neighbour",
     "hint": "I split one granule into pieces", "color": "#fbbf24", "hotkey": "1"},
    {"key": "expand", "label": "outline too tight",
     "hint": "the boundary should sit further out", "color": "#38bdf8", "hotkey": "2"},
    {"key": "split",  "label": "this is two granules",
     "hint": "I merged two into one", "color": "#c084fc", "hotkey": "3"},
    {"key": "wrong",  "label": "not a granule",
     "hint": "nothing should be outlined here", "color": "#fb7185", "hotkey": "4"},
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
          review: bool = False, page_name: str = "index.html",
          marks: Sequence[Dict[str, Any]] = tuple(DEFAULT_MARKS)) -> str:
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
           "marks": list(marks),
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
        #: per polygon: None if hand-drawn or if an agent proposal the user left unmarked,
        #: else the failure mode they marked it with ('merge', 'expand', 'split', 'wrong',
        #: 'fixed'). Unmarked only means CORRECT when :attr:`reviewed` is True.
        self.poly_mark: List[Optional[str]] = []
        #: merge group id, shared by the two-or-more outlines that are really one granule
        self.poly_group: List[Optional[int]] = []
        self.poly_auto_src: List[bool] = []
        #: did the user declare this field finished? Without it, "unmarked" means
        #: "not looked at", and counting those as correct is the whole trap.
        self.reviewed: bool = bool(meta.get("reviewed", False))
        self.circles: List[Tuple[float, float, float]] = []
        self.circle_cls: List[str] = []
        self.regions: List[Tuple[float, float, float, float]] = []
        #: ``(row, col)`` points where the user says an object is present and nothing was
        #: proposed. The four marks all describe an outline that EXISTS, so without these the
        #: review can only ever produce a precision; these are the recall numerator's
        #: complement. Only meaningful inside :attr:`regions`.
        self.misses: List[Tuple[float, float]] = []
        self.notes = str(meta.get("notes") or "")
        for s in shapes:
            k = s.get("k")
            if k == "poly" and len(s.get("p") or ()) >= 3:
                self.polys.append(np.asarray([[p[1], p[0]] for p in s["p"]], dtype=float))
                self.poly_cls.append(str(s.get("c")))
                self.poly_auto.append(s.get("a"))
                self.poly_rgb.append(dict(s.get("m") or {}))
                self.poly_mark.append(s.get("mk") or None)
                self.poly_group.append(s.get("g") or None)
                self.poly_auto_src.append(s.get("src") == "auto")
            elif k == "circ":
                self.circles.append((float(s["cy"]), float(s["cx"]), float(s["r"])))
                self.circle_cls.append(str(s.get("c")))
            elif k == "rect":
                y0, y1 = sorted((float(s["y0"]), float(s["y1"])))
                x0, x1 = sorted((float(s["x0"]), float(s["x1"])))
                self.regions.append((y0, x0, y1, x1))
            elif k == "miss":
                self.misses.append((float(s["y"]), float(s["x"])))

    @property
    def n_override(self) -> int:
        """Polygons whose type the user changed away from the colour's answer.

        The interesting number, not a diagnostic: it is the error rate of channel-dominance
        classification measured against the user's eye, for free.
        """
        return sum(1 for c, a in zip(self.poly_cls, self.poly_auto) if a is not None and a != c)

    def confirmed(self, *, for_boundary: bool = True) -> List[int]:
        """Indices of polygons carrying the user's authority — hand-drawn, or an agent
        proposal they left unmarked **in a field they signed off**.

        Two ways this returns fewer than you might expect, both deliberate:

        * A proposal in a field with ``reviewed=False`` is excluded whatever its mark,
          because "unmarked" there means "not looked at". Scoring against your own unchecked
          output is the easiest way there is to manufacture a number that means nothing.
        * ``for_boundary`` (the default) also drops anything marked ``expand``. Those ARE
          real granules — they belong in a detection count — but the user has said the
          boundary is in the wrong place, so including them in a boundary error would be
          averaging in a shape they explicitly rejected. Pass ``for_boundary=False`` when
          counting objects rather than measuring edges.
        """
        drop = {"merge", "split", "wrong"} | ({"expand"} if for_boundary else set())
        return [i for i, (mk, auto) in enumerate(zip(self.poly_mark, self.poly_auto_src))
                if (not auto) or (self.reviewed and mk not in drop)]

    def merge_groups(self) -> Dict[int, List[int]]:
        """``{group id: [polygon indices]}`` — sets the user says are ONE granule.

        This is the direct measurement of over-segmentation: each group of n outlines is one
        granule I broke into n pieces.
        """
        g: Dict[int, List[int]] = {}
        for i, (mk, gid) in enumerate(zip(self.poly_mark, self.poly_group)):
            if mk == "merge" and gid:
                g.setdefault(int(gid), []).append(i)
        return {k: v for k, v in g.items() if len(v) >= 2}

    def recall(self) -> Optional[Tuple[float, int, int]]:
        """``(recall, found, missed)`` inside the marked regions, or ``None``.

        Returns ``None`` — never a number — unless the field is signed off AND at least one
        region box exists. Both conditions are load-bearing and neither is a formality:

        * outside a region, a missing outline and an area nobody examined are the same
          picture, so anything computed there is a guess wearing a percent sign;
        * in a field that is not signed off, an unmarked proposal has not been accepted,
          so the numerator is unknown too.

        The caller must handle ``None`` rather than defaulting to 1.0. A pipeline that
        silently reports perfect recall when nobody looked is the exact failure this method
        exists to make impossible.
        """
        if not self.reviewed or not self.regions:
            return None

        def _in(y: float, x: float) -> bool:
            return any(y0 <= y <= y1 and x0 <= x <= x1 for y0, x0, y1, x1 in self.regions)

        found = 0
        for i in self.confirmed(for_boundary=False):
            p = self.polys[i]
            if _in(float((p[:, 0].min() + p[:, 0].max()) / 2),
                   float((p[:, 1].min() + p[:, 1].max()) / 2)):
                found += 1
        missed = sum(1 for y, x in self.misses if _in(y, x))
        if found + missed == 0:
            return None
        return found / (found + missed), found, missed

    def mark_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for mk, auto in zip(self.poly_mark, self.poly_auto_src):
            k = ("hand" if not auto else (mk or ("ok" if self.reviewed else "unchecked")))
            out[k] = out.get(k, 0) + 1
        return out

    def __repr__(self) -> str:
        r = self.recall()
        return (f"<TileTruth {self.id}: {len(self.polys)} polygons, "
                f"{len(self.circles)} circles, {len(self.regions)} regions, "
                f"{len(self.misses)} missed, reviewed={self.reviewed}, "
                f"{self.mark_counts()}, {len(self.merge_groups())} merge groups, "
                f"recall={'unmeasured' if r is None else f'{r[0]:.3f}'}>")


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
              classes: Sequence[str] = ("F", "I", "?"), fill_holes: bool = False):
    """``(labels, {label_id: class_key})`` — the polygons as an integer label image.

    Later polygons overwrite earlier ones where they overlap, which matches how they were
    drawn: a granule outlined on top of another was seen on top of it.
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
        if ck not in classes:
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
.hint2{font:10.5px/1.45 var(--mono);color:var(--dim);margin-bottom:8px}
.mk{display:flex;align-items:center;gap:8px;width:100%;padding:6px 8px;margin-bottom:4px;
 border-left:4px solid var(--k);text-align:left}
.mk.on{border-color:var(--k);background:var(--panel);box-shadow:inset 0 0 0 1px var(--k)}
.mk b{color:var(--k);font-size:11px;min-width:52px}
.mk span{color:var(--dim);font-size:9.5px;line-height:1.3;flex:1}
.mk i{font-style:normal;color:var(--k);font:10.5px var(--mono);margin-left:auto}
#rectools button{border-color:#fb923c66;color:#fb923c}
#rectools button.on{border-color:#fb923c;background:#fb923c1a}
#rechint{color:#fb923c}
.grp{border:1px solid #fbbf24;color:#fbbf24;border-radius:7px;padding:7px 9px;
 font:10.5px/1.5 var(--mono);margin-bottom:7px}
.grp button{margin-top:5px;width:100%;border-color:#fbbf24;color:#fbbf24}
button.done{border-color:var(--ok);color:var(--ok);background:#34d39914}
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
      <div class="sec">mark only what is WRONG</div>
      <div class="hint2">Everything you leave unmarked counts as correct once you tick the
        field off below. Click a granule to mark it; click it again to unmark.</div>
      <div id="marks"></div>
      <div class="sec" style="color:#fb923c">and what I MISSED</div>
      <div class="hint2">The four marks above all describe an outline that <b>exists</b>, so on
        their own they can only measure how often I am wrong &mdash; never how often I miss
        something. Drag a <b>region</b> box round an area you will check completely, then click
        each thing inside it that I failed to find.</div>
      <div class="tools" id="rectools">
        <button data-t="rect">region <kbd>r</kbd></button>
        <button data-t="miss">missed one <kbd>m</kbd></button>
      </div>
      <div class="hint2" id="rechint" style="margin-top:6px"></div>
      <div id="grpbox" class="grp" style="display:none"></div>
      <div class="sec">this field</div>
      <div class="tally" id="rtal"></div>
      <button id="rdone" style="width:100%;margin-top:8px">
        &#10003; I have been through this field</button>
      <button id="rnext" style="width:100%;margin-top:5px;border-color:#38bdf8;color:#38bdf8">
        show me the ones I am least sure of <kbd>n</kbd></button>
    </div>
    <div class="sec">fields</div><div class="tiles" id="tiles"></div>
    <div class="sec">tool</div>
    <div class="tools">
      <button data-t="poly" class="on">polygon <kbd>p</kbd></button>
      <button data-t="circ">circle <kbd>c</kbd></button>
      <button data-t="rect">region <kbd>r</kbd></button>
      <button data-t="sel">select <kbd>v</kbd></button>
      <button data-t="miss" style="grid-column:1/-1">you missed one <kbd>m</kbd></button>
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
const SNAP_PX = 7;                 // screen px within which a corner grabs a neighbour's
const MISS_COLOR = '#fb923c';      // orange: used by nothing else, so a miss cannot be misread
const S = {
  ti: 0, tool: 'poly', cls: POLYCLS[0].key, pin: null, sel: null, draft: null, drag: null,
  z: 1, ox: 0, oy: 0, pan: null, space: false, hover: null,
  chan: { r: true, g: true, b: true }, bright: 1,
  mark: null, grp: null, done: {},
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
// Review works by EXCEPTION: unmarked means correct, but only once the field is ticked off.
// Until then an unmarked proposal is simply unlooked-at, and the two must never be conflated.
const MARKS = CFG.marks || [];
const MKBY = Object.fromEntries(MARKS.map(m => [m.key, m]));
const isProp = s => s.src === 'auto' && !s.mk;
const marked = s => s.src === 'auto' && !!s.mk;
const fieldDone = (id) => !!S.done[id || tile().id];

function newGroupId() {
  return 1 + Math.max(0, ...shapes().filter(s => s.g).map(s => s.g));
}
/* A merge is a statement about a SET of outlines, so it is collected as a group rather than
   flagged per object: "this one is over-split" is not answerable alone — the question is
   which pieces are one granule. The group stays open until it is closed, and a group that
   ends up with a single member is discarded, because one piece merged with nothing is not
   a claim about anything. */
function closeGroup(quiet) {
  if (!S.grp) return;
  const mem = shapes().filter(s => s.g === S.grp);
  if (mem.length < 2) {
    mem.forEach(s => { delete s.mk; delete s.g; });
    if (!quiet && mem.length) flash('a merge needs two or more — that one was dropped');
  }
  S.grp = null; persist(); syncUI(); draw();
}
function applyMark(s) {
  if (!s || s.src !== 'auto') return;
  snap();
  if (S.mark === 'merge') {
    if (s.mk === 'merge') {                       // clicking a member takes it back out
      const g = s.g;
      delete s.mk; delete s.g;
      const rest = shapes().filter(x => x.g === g);
      if (rest.length === 1 && g !== S.grp) { delete rest[0].mk; delete rest[0].g; }
    } else {
      if (!S.grp) S.grp = newGroupId();
      delete s.mk; delete s.g;
      s.mk = 'merge'; s.g = S.grp;
    }
  } else {
    if (s.mk === S.mark) { delete s.mk; delete s.g; }
    else { delete s.g; s.mk = S.mark; }
  }
  persist(); syncUI(); draw();
}
function setMark(k) {
  if (S.mark !== 'merge') closeGroup(true);
  else if (k !== 'merge') closeGroup();
  S.mark = k; syncUI(); draw();
}
function setDone(v) {
  closeGroup(true);
  S.done[tile().id] = v === undefined ? !fieldDone() : v;
  try { localStorage.setItem(KEY + '__done', JSON.stringify(S.done)); } catch (e) { }
  syncUI(); draw();
  if (fieldDone()) {
    const n = shapes().filter(isProp).length;
    flash(`${tile().id} signed off — ${n} unmarked outlines now count as correct`);
  }
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
    if (s.mk) {
      const mc = (MKBY[s.mk] || {}).color || '#fff';
      ctx.strokeStyle = mc;
      ctx.fillStyle = mc + '3a';
      ctx.lineWidth = (sel ? lw * 2 : lw) * 2.2;
      if (s.g && s.g === S.grp) ctx.setLineDash([7 / S.z, 4 / S.z]);   // group still open
    } else if (!fieldDone(tile().id)) {
      // not yet looked at: hollow and quiet, so a marked one is unmissable
      ctx.setLineDash([6 / S.z, 4 / S.z]); ctx.fillStyle = 'transparent';
      ctx.globalAlpha = 0.75;
    }
    if (s === S.hover) { ctx.lineWidth = Math.max(ctx.lineWidth, lw * 2.4); ctx.globalAlpha = 1; }
  }
  // A miss is drawn as a crosshair rather than an outline BECAUSE it is not an outline: it
  // asserts that something is here, and a ring would read as a claim about its size.
  if (s.k === 'miss') {
    const r = (sel ? 13 : 10) / S.z, w = (sel ? 2.6 : 2.0) / S.z;
    ctx.strokeStyle = MISS_COLOR; ctx.lineWidth = w;
    ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, 6.2832); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(s.x - r * 1.75, s.y); ctx.lineTo(s.x - r * 0.45, s.y);
    ctx.moveTo(s.x + r * 0.45, s.y); ctx.lineTo(s.x + r * 1.75, s.y);
    ctx.moveTo(s.x, s.y - r * 1.75); ctx.lineTo(s.x, s.y - r * 0.45);
    ctx.moveTo(s.x, s.y + r * 0.45); ctx.lineTo(s.x, s.y + r * 1.75);
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
    return;
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
  // the group number, so which pieces belong to which merge is readable without clicking
  if (s.g && s.k === 'poly') {
    const b = bboxOf(s.p), r = 7.5 / S.z;
    ctx.fillStyle = (MKBY[s.mk] || {}).color || '#fff';
    ctx.beginPath(); ctx.arc((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, r, 0, 6.2832); ctx.fill();
    ctx.fillStyle = '#0b0f17';
    ctx.font = `${11 / S.z}px ui-monospace,monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(s.g), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
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
  const nm = n.filter(s => s.k === 'miss').length;
  document.getElementById('hud').innerHTML =
    `<b>${t.id}</b> ${t.what || ''} &middot; z=${t.z ?? '?'} t=${t.t ?? '?'} &middot; ` +
    `${(S.z * 100).toFixed(0)}% &middot; <b>${np}</b> outlines &middot; <b>${nc}</b> cells` +
    (nm ? ` &middot; <b style="color:${MISS_COLOR}">${nm}</b> missed` : '') +
    (nr ? ` &middot; ${nr} region${nr > 1 ? 's' : ''} marked` : '') +
    (t.empty_channels && t.empty_channels.length
      ? `<br><span style="color:#fbbf24">no ${t.empty_channels.map(
        c => ({ r: 'F', g: 'cell', b: 'inert' }[c] || c)).join('/')} signal in this field` +
      ` — by design</span>` : '');
  document.getElementById('hint').textContent = {
    poly: 'click each corner · click the first corner again (or enter / double-click) to close · '
      + 'corners snap to a neighbour\'s · backspace undoes a corner · esc cancels',
    circ: 'press at the centre of a cell and drag out · esc cancels',
    rect: 'drag a box around the area where you outlined EVERY granule',
    sel: 'click a shape · drag it or its corners · 1-5 retypes · delete removes',
    miss: 'click anywhere I should have found something and did not · click a marker again '
      + 'to take it back · only counts inside a region box',
  }[S.tool] + (CFG.review
    ? `  ·  click an outline to mark it "${(MKBY[S.mark] || {}).label || ''}" · 1-4 switch `
    + 'what you are marking · drag a corner to redraw it · space+drag pans'
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
function snapVertex(x, y) {
  const e = SNAP_PX / S.z;
  let best = null, bd = e;
  for (const s of shapes()) {
    if (s.k !== 'poly') continue;
    for (const p of s.p) {
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d < bd) { bd = d; best = p; }
    }
  }
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
/* A MISS is a claim about empty space, not about an outline, which is why it is a tool rather
   than another entry in the mark vocabulary. Every mark answers "what is wrong with THIS
   proposal"; a miss answers "there is a granule here and you produced nothing", and there is
   no shape to attach it to. Without it the review can only ever measure precision — the four
   marks all describe an outline that exists — and recall stays unquotable however many fields
   are signed off. One click, no drawing: the user is asserting presence, not geometry, and
   demanding an outline for it would make the cheap half of the answer cost as much as the
   expensive half. */
function commitMiss(x, y) {
  snap();
  const s = { i: S.nextId[tile().id]++, k: 'miss', c: '_miss', a: null,
              x: +x.toFixed(2), y: +y.toFixed(2) };
  shapes().push(s); S.sel = s; persist(); syncUI(); draw();
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
    // misses are tested FIRST at every index: a marker dropped inside an outline the user is
    // disputing must stay clickable, or it can never be taken back
    if (s.k === 'miss' && Math.hypot(x - s.x, y - s.y) <= 13 / S.z) return s;
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
    d.p.push(snapVertex(x, y)); d.cur = [x, y]; draw(); return;
  }
  if (e.button === 2) return;
  if (S.tool === 'circ') { S.draft = { k: 'circ', cx: x, cy: y, r: 0 }; return; }
  if (S.tool === 'rect') { S.draft = { k: 'rect', x0: x, y0: y, x1: x, y1: y }; return; }
  if (S.tool === 'miss') {
    const h = hit(x, y);
    // clicking an existing marker takes it back, so a misplaced one costs one click, not a
    // tool switch — the same click-again-to-undo the marks already use
    if (h && h.k === 'miss') {
      snap(); S.shapes[tile().id] = shapes().filter(s => s !== h); S.sel = null;
      persist(); syncUI(); draw();
    } else commitMiss(x, y);
    return;
  }
  const s = hit(x, y);
  S.sel = s;
  if (s) {
    const vi = vertexAt(s, x, y);
    // Click vs drag is decided on release, not here: in review a plain click MARKS the
    // outline and a drag EDITS it, and demanding a separate tool for each would put a mode
    // switch between the user and the one action they came to do.
    S.drag = { s, vi, x, y, o: JSON.parse(JSON.stringify(s)), moved: false,
               at: [e.clientX, e.clientY] };
  }
  syncUI(); draw();
});
cv.addEventListener('pointermove', e => {
  const [x, y] = toImg(e);
  if (CFG.review && !S.pan && !S.draft && !S.drag) {
    const h = hit(x, y);
    if (h !== S.hover) { S.hover = h; cv.style.cursor = h ? 'pointer' : 'crosshair'; draw(); }
  }
  if (S.pan) {
    S.ox = S.pan[2] + (e.clientX - S.pan[0]); S.oy = S.pan[3] + (e.clientY - S.pan[1]);
    draw(); return;
  }
  if (S.draft) {
    const d = S.draft;
    if (d.k === 'poly') d.cur = [x, y];
    else if (d.k === 'circ') d.r = Math.hypot(x - d.cx, y - d.cy);
    else { d.x1 = x; d.y1 = y; }
    draw(); return;
  }
  if (S.drag) {
    const { s, vi, o } = S.drag, dx = x - S.drag.x, dy = y - S.drag.y;
    if (!S.drag.moved) {
      if (Math.hypot(e.clientX - S.drag.at[0], e.clientY - S.drag.at[1]) < 4) return;
      S.drag.moved = true; snap();          // only a real drag costs an undo slot
    }
    if (s.k === 'poly') {
      if (vi >= 0) { s.p[vi] = [o.p[vi][0] + dx, o.p[vi][1] + dy]; }
      else s.p = o.p.map(p => [p[0] + dx, p[1] + dy]);
    } else if (s.k === 'circ') { s.cx = o.cx + dx; s.cy = o.cy + dy; }
    else if (s.k === 'miss') { s.x = o.x + dx; s.y = o.y + dy; }
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
    const s = S.drag.s, was = S.drag.moved;
    S.drag = null;
    if (!was && CFG.review && S.mark) applyMark(s);   // a click, not a drag
    else if (was) {
      // moving a proposal's geometry IS the correction — no separate "mark as fixed" step,
      // which would be a step everyone forgets and would leave the record wrong.
      if (s.src === 'auto') { s.mk = 'fixed'; delete s.g; }
      persist(); syncUI();
    }
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
      if (S.sel.src === 'auto' && isProp(S.sel)) { S.sel.mk = 'wrong'; delete S.sel.g; }
      else { S.shapes[tile().id] = shapes().filter(s => s !== S.sel); S.sel = null; }
      persist(); draw();
    }
    e.preventDefault(); return;
  }
  if (CFG.review) {
    const m = MARKS.find(x => x.hotkey === e.key);
    if (m) {
      setMark(m.key);
      const t = S.hover || S.sel;
      if (t) applyMark(t);
      e.preventDefault(); return;
    }
    if (k === 'enter') { closeGroup(); return; }
    if (k === 'n') { jumpNext(); return; }
  }
  if ('pcrvm'.includes(k) && !e.ctrlKey) {
    setTool({ p: 'poly', c: 'circ', r: 'rect', v: 'sel', m: 'miss' }[k]); return;
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
  document.querySelectorAll('#marks .mk').forEach(
    b => b.classList.toggle('on', b.dataset.m === S.mark));
  document.getElementById('nt').value = S.notes[tile().id] || '';
  tally(); hud();
}
function tally() {
  const rows = [];
  let gtot = 0;
  for (const t of TILES) {
    const sh = S.shapes[t.id] || [];
    gtot += sh.filter(s => s.k !== 'rect' && s.k !== 'miss').length;
    const b = document.querySelector(`#tiles .tb[data-i="${TILES.indexOf(t)}"] .n`);
    if (!b) continue;
    if (CFG.review) {
      const a = sh.filter(s => s.src === 'auto'), bad = a.filter(marked).length;
      b.innerHTML = S.done[t.id]
        ? `&#10003; ${bad} wrong` : `${a.length}`;
      b.style.color = S.done[t.id] ? 'var(--ok)' : 'var(--dim)';
    } else b.textContent = sh.length ? sh.filter(s => s.k !== 'rect').length : '';
  }
  const sh = shapes();
  for (const c of CLS) {
    const n = sh.filter(s => s.c === c.key).length;
    if (n) rows.push(`<b style="color:${c.color}">${n}</b> ${c.label}${n > 1 ? 's' : ''}`);
  }
  const nr = sh.filter(s => s.k === 'rect').length;
  const nm = sh.filter(s => s.k === 'miss').length;
  if (nm) rows.push(`<b style="color:${MISS_COLOR}">${nm}</b> I missed`);
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
    const bad = auto.filter(marked);
    const per = {};
    bad.forEach(s => per[s.mk] = (per[s.mk] || 0) + 1);
    const groups = new Set(auto.filter(s => s.g).map(s => s.g)).size;
    const nDone = TILES.filter(t => S.done[t.id]).length;
    const L = [`<b>${auto.length}</b> outlines here, <b style="color:${bad.length ?
      '#fbbf24' : 'var(--ok)'}">${bad.length}</b> marked wrong`];
    for (const m of MARKS) {
      if (per[m.key]) L.push(`<span style="color:${m.color}">${per[m.key]} ${m.label}` +
        (m.key === 'merge' ? ` in ${groups} group${groups === 1 ? '' : 's'}` : '') + `</span>`);
    }
    if (per.fixed) L.push(`<span style="color:#38bdf8">${per.fixed} redrawn by hand</span>`);
    // Recall is the one number the four marks structurally cannot produce, so it is reported
    // here the moment it becomes computable — and reported as NOT computable until then,
    // rather than silently omitted.
    const nmi = sh.filter(s => s.k === 'miss').length;
    let rh;
    if (!nr) {
      rh = nmi
        ? `<b>${nmi} marked, but no region box yet</b> — without one there is no denominator, `
        + `so they cannot become a recall number. Press <b>r</b> and drag a box.`
        : `No region box yet, so recall is <b>not computable</b> for this field. It does not `
        + `have to be large — press <b>r</b> and drag one round any area you will check `
        + `completely.`;
    } else {
      const inR = (px, py) => sh.some(r => r.k === 'rect' && px >= r.x0 && px <= r.x1 &&
        py >= r.y0 && py <= r.y1);
      const miR = sh.filter(s => s.k === 'miss' && inR(s.x, s.y)).length;
      const okR = auto.filter(s => {
        if (s.k !== 'poly' || s.mk) return false;
        const b = bboxOf(s.p);
        return inR((b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
      }).length;
      L.push(`<span style="color:${MISS_COLOR}">${miR} missed inside the region</span>` +
        (fieldDone() && (okR + miR)
          ? ` &rarr; recall <b>${(100 * okR / (okR + miR)).toFixed(0)}%</b>` : ''));
      rh = !fieldDone()
        ? `Region covers <b>${okR}</b> of my outlines and <b>${miR}</b> misses. Sign the field `
        + `off below and that becomes a recall.`
        : (okR + miR
          ? `Recall here is <b>${(100 * okR / (okR + miR)).toFixed(1)}%</b> — ${miR} missed `
          + `against ${okR} found. <b>Zero misses is a real result</b>, not a blank: it says I `
          + `am not dropping objects, only cutting them up.`
          : `The region is empty — move or enlarge it.`);
    }
    document.getElementById('rechint').innerHTML = rh;
    L.push(fieldDone()
      ? `<b style="color:var(--ok)">&#10003; signed off</b> — the other ` +
      `${auto.length - bad.length} count as correct`
      : `<span style="color:#fbbf24">not signed off yet, so nothing here counts as ` +
      `correct</span>`);
    L.push(`<span style="color:var(--dim)">${nDone} of ${TILES.length} fields signed off` +
      `</span>`);
    document.getElementById('rtal').innerHTML = L.join('<br>');
    const db = document.getElementById('rdone');
    db.classList.toggle('done', fieldDone());
    db.innerHTML = fieldDone() ? '&#10003; this field is signed off (click to undo)'
      : '&#10003; I have been through this field';
    const gb = document.getElementById('grpbox');
    if (S.grp) {
      const n = shapes().filter(s => s.g === S.grp).length;
      gb.style.display = '';
      gb.innerHTML = `<b>merge group ${S.grp}</b>: ${n} outline${n === 1 ? '' : 's'} ` +
        `selected${n < 2 ? ' — a merge needs at least two' : ''}` +
        `<button id="gclose">finish this group &nbsp;<kbd>enter</kbd></button>`;
      document.getElementById('gclose').onclick = () => closeGroup();
    } else gb.style.display = 'none';
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
  // a miss has no type — it says "something is here", not what — so retyping must skip it
  if (S.sel && S.sel.k !== 'rect' && S.sel.k !== 'miss') {
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
  document.getElementById('marks').innerHTML = MARKS.map(m =>
    `<button class="mk" data-m="${m.key}" style="--k:${m.color}"><b>${m.label}</b>` +
    `<span>${m.hint || ''}</span><i>${m.hotkey || ''}</i></button>`).join('');
  document.querySelectorAll('#marks .mk').forEach(b =>
    b.onclick = () => setMark(b.dataset.m));
  document.getElementById('rnext').onclick = jumpNext;
  document.getElementById('rdone').onclick = () => setDone();
  try { S.done = JSON.parse(localStorage.getItem(KEY + '__done') || '{}'); }
  catch (e) { S.done = {}; }
  S.mark = MARKS.length ? MARKS[0].key : null;
  setTool('sel');
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
    tiles[t.id] = Object.assign(meta, { notes: S.notes[t.id] || '', shapes: sh,
                                        reviewed: !!S.done[t.id] });
  }
  return {
    schema: 'evidence-log/groundtruth@1', title: CFG.title,
    saved: new Date().toISOString(), ambig_margin: CFG.ambig,
    classes: CLS, tiles,
  };
}
function summary() {
  const d = doc(), L = [];
  let np = 0, nc = 0, nr = 0, ov = 0, au = 0, nmi = 0, recOK = 0, recMI = 0;
  const V = { marked: 0, ok: 0, unchecked: 0 };
  for (const [id, t] of Object.entries(d.tiles)) {
    const p = t.shapes.filter(s => s.k === 'poly'), c = t.shapes.filter(s => s.k === 'circ');
    const r = t.shapes.filter(s => s.k === 'rect');
    const mi = t.shapes.filter(s => s.k === 'miss');
    np += p.length; nc += c.length; nr += r.length; nmi += mi.length;
    // recall counts ONLY inside a region box and ONLY in a signed-off field; outside one, a
    // missing outline is indistinguishable from an area nobody looked at
    if (r.length && t.reviewed) {
      const inR = (px, py) => r.some(b => px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1);
      recMI += mi.filter(s => inR(s.x, s.y)).length;
      recOK += p.filter(s => {
        if (s.mk) return false;
        const xs = s.p.map(q => q[0]), ys = s.p.map(q => q[1]);
        return inR((Math.min(...xs) + Math.max(...xs)) / 2,
                   (Math.min(...ys) + Math.max(...ys)) / 2);
      }).length;
    }
    const hand = p.filter(s => s.src !== 'auto');
    au += hand.filter(s => s.a).length;
    ov += hand.filter(s => s.a && s.a !== s.c).length;
    const auto = t.shapes.filter(s => s.src === 'auto');
    const bad = auto.filter(s => s.mk);
    V.marked += bad.length;
    if (t.reviewed) V.ok += auto.length - bad.length; else V.unchecked += auto.length;
    bad.forEach(s => { V[s.mk] = (V[s.mk] || 0) + 1; });
    const by = {};
    p.forEach(s => by[s.c] = (by[s.c] || 0) + 1);
    const cnt = {};
    bad.forEach(s => cnt[s.mk] = (cnt[s.mk] || 0) + 1);
    const rv = auto.length
      ? `; of my ${auto.length} proposals ${bad.length} marked wrong (` +
      (Object.entries(cnt).map(([k, v]) => `${v} ${k}`).join(', ') || 'none') + `)` +
      (t.reviewed ? `, field SIGNED OFF so the other ${auto.length - bad.length} are correct`
        : `, field NOT signed off — none of it counts yet`) : '';
    L.push(`${id}: ${p.length} outlines (${Object.entries(by).map(([k, v]) => k + '=' + v).join(' ')})` +
      `, ${c.length} cells, ${r.length} region${r.length === 1 ? '' : 's'}` +
      (mi.length ? `, ${mi.length} MISSED` : '') + rv +
      (t.notes ? ` — note: ${t.notes.replace(/\s+/g, ' ').trim()}` : ''));
  }
  if (!L.length) return '';
  const rec = (recOK + recMI)
    ? ` Recall ${(100 * recOK / (recOK + recMI)).toFixed(1)}% (${recMI} missed against ` +
      `${recOK} found, inside marked regions in signed-off fields only).`
    : (nmi ? ` ${nmi} misses marked but no region box in a signed-off field, so recall is ` +
             `still NOT computable.`
           : ` No misses marked, so recall is UNMEASURED — do not read precision as accuracy.`);
  const head = CFG.review
    ? `PROPOSAL REVIEW  ${V.ok} confirmed correct, ${V.marked} marked wrong (` +
    (MARKS.map(m => V[m.key] ? `${V[m.key]} ${m.key}` : '').filter(Boolean).join(', ')
      || 'none') + `), ${V.unchecked} in fields NOT signed off — those are still my output ` +
    `and must not be scored as ground truth.` + rec
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
// Review opens on the SELECT tool, not the polygon tool: the first click in a review pass
// should mark the outline under the cursor, not start drawing a new one.
loadAll(); resize(); setTile(0); setTool(CFG.review ? 'sel' : 'poly');
"""
