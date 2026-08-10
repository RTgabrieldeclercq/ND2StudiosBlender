"""Bake-off harness for the label-reclaim assignment rule (proposed `analysis.reclaim_labels`).

The question this exists to answer: when a segmentation captures the fibroblast body but
misses the thin threads reaching toward each elongated pole, WHICH rule should hand that
leftover signal back to a label?

    brightest  - marker watershed flooding along the BRIGHTEST route
    geodesic   - nearest label by in-mask flood distance, blind to brightness
    fragment   - each connected leftover fragment goes wholly to one label
    euclidean  - straight-line nearest label, then clipped to the mask

`brightest` and `geodesic` are deliberately the same backend and the same connectivity,
differing ONLY in the cost surface, so the comparison isolates the thing being decided.

Stage 1 (`--synthetic`) demonstrates the mechanism on a fixture whose truth is known by
construction, and is what justifies not simply reusing `transform.grow_points`.
Stage 2 (`--real`) runs the same four rules on real frames; that is the one that decides it.

Run from the repo root:

    .\\.venv\\Scripts\\python.exe -B scripts\\_reclaim_bakeoff.py --synthetic

Nothing here writes to the node catalog. It renders figures and appends to the evidence log
at CodeLog/live/segmentation-reclaim so the choice of rule is a decision on the record
rather than one taken quietly inside a compute.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, ".claude", "skills", "evidence-log"))

LOG_ROOT = os.path.join("CodeLog", "live", "segmentation-reclaim")

#: The four candidate rules, in the order they are always presented, so the panels, the
#: tables and the question put to the user all agree.
RULES = ("brightest", "geodesic", "fragment", "euclidean")

#: Shared by the two watershed rules. Pinned here rather than left to each call so that
#: `brightest` vs `geodesic` differs in the COST SURFACE and nothing else -- if they used
#: different neighbourhoods the comparison would be measuring two changes at once.
_CONNECTIVITY = 1


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ the four assignment rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Each takes the same arrays and returns a label raster of the same shape: the input labels
# PLUS whatever leftover that rule claimed. Two invariants are asserted, not trusted: a rule
# may never alter an already-labelled voxel, and never paint outside the leftover mask.

def _frag_label_contacts(frags: np.ndarray, labels: np.ndarray, connectivity: int = 1):
    """Every (fragment, label) adjacency and how many voxels back it, vectorized.

    Replaces a per-fragment Python loop that ran one full-image comparison and one
    `binary_dilation` for EVERY fragment -- O(fragments x image). On a real frame at a low
    leftover cut that is 10,036 fragments over 1024x1024, which did not finish in ten
    minutes. Here each neighbour shift is a single array operation whatever the fragment
    count, so the cost stops depending on how finely the leftover happens to break up.

    Returns ``(frag_id, label_id, n_contact_voxels)``, one entry per distinct touching pair.

    ``connectivity=1`` (face neighbours only) is not a detail: `ndi.label` builds the
    fragments 4-connected by default and `ndi.binary_dilation` defaults to the same cross
    structure, so counting diagonal touches here while the fragments themselves are
    4-connected is internally inconsistent. Using the full 3x3 offset set instead moved 18
    fragments across the bridge/single-label boundary on one real frame and shifted 926 px
    of reported bridge area -- a wrong answer that still looked plausible, which is why the
    equivalence against the original loop is checked rather than assumed.
    """
    K = int(labels.max()) + 1
    N = labels.size
    flat = np.arange(N, dtype=np.int64).reshape(labels.shape)
    seen = []
    for sh in itertools.product((-1, 0, 1), repeat=frags.ndim):
        if sum(abs(v) for v in sh) != connectivity:
            continue
        a_sl, b_sl = [], []
        for d in range(frags.ndim):
            n = frags.shape[d]
            if sh[d] >= 0:
                a_sl.append(slice(sh[d], n))
                b_sl.append(slice(0, n - sh[d]))
            else:
                a_sl.append(slice(0, n + sh[d]))
                b_sl.append(slice(-sh[d], n))
        a, b = frags[tuple(a_sl)], labels[tuple(b_sl)]
        sel = (a > 0) & (b > 0)
        if sel.any():
            #  Key on the LABEL VOXEL's position, not on the adjacency, so that one label
            #  voxel touching a fragment along several neighbours is still counted once.
            #  The loop this replaces did `np.unique(labels[dilated_edge])`, which counts
            #  distinct boundary VOXELS; weighting by adjacency count instead silently moved
            #  the argmax and handed 6-11 fragments per frame to the wrong cell.
            seen.append(a[sel].astype(np.int64) * N + flat[tuple(b_sl)][sel])
    if not seen:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    uv = np.unique(np.concatenate(seen))            # distinct (fragment, label-voxel)
    f_of = uv // N
    l_of = labels.ravel()[uv % N].astype(np.int64)
    uk, counts = np.unique(f_of * K + l_of, return_counts=True)
    return uk // K, uk % K, counts


def _assign(rule: str, img: np.ndarray, labels: np.ndarray, leftover: np.ndarray,
            sampling: tuple) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    # Markers must sit inside the flood region or they cannot seed anything, so the region
    # is the leftover PLUS the existing labels.
    region = leftover | (labels > 0)

    if rule == "brightest":
        # -img so bright = downhill: each label floods along the brightest route, and a
        # thread shared by two cells is cut at its dimmest point.
        got = watershed(-img.astype(np.float64), markers=labels, mask=region,
                        connectivity=_CONNECTIVITY)

    elif rule == "geodesic":
        # A CONSTANT cost surface, so the flood order is in-mask distance and brightness is
        # ignored entirely. Verified against a per-label MCP_Geometric geodesic reference on
        # a two-seed fixture (0 disagreements); MCP's own multi-source `traceback` cannot be
        # used for this -- it returns neighbour-offset indices, not start indices, which is
        # exactly the bug that produced the retracted first geodesic number.
        got = watershed(np.zeros(img.shape, dtype=np.float64), markers=labels, mask=region,
                        connectivity=_CONNECTIVITY)

    elif rule == "fragment":
        # Whole-fragment: every connected leftover piece goes to the label it abuts most.
        # A tie goes to the higher label id (the loop this replaced took the lower); ties
        # are rare and the choice is arbitrary either way, but it is stated rather than
        # left to be discovered.
        got = labels.copy()
        frags, n = ndi.label(leftover)
        fid, lid, cnt = _frag_label_contacts(frags, labels)
        if fid.size:
            order = np.lexsort((cnt, fid))          # ascending count within each fragment
            fs, ls = fid[order], lid[order]
            last = np.ones(fs.size, dtype=bool)
            last[:-1] = fs[1:] != fs[:-1]           # keep the max-count row per fragment
            winner = np.zeros(n + 1, dtype=np.int64)
            winner[fs[last]] = ls[last]
            sel = frags > 0
            got[sel] = winner[frags[sel]]

    elif rule == "euclidean":
        # Straight-line nearest label, blind to connectivity, then clipped to the mask.
        _d, idx = ndi.distance_transform_edt(labels == 0, sampling=sampling,
                                             return_indices=True)
        got = labels[tuple(idx)]

    else:
        raise ValueError(rule)

    keep = labels > 0
    got = np.where(keep, labels, got)
    got[~region] = 0
    assert np.array_equal(got[keep], labels[keep]), f"{rule} altered existing labels"
    assert not got[~region].any(), f"{rule} painted outside the leftover mask"
    return got


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ the synthetic fixture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _thread(img, truth, p0, p1, cid, value, half=1, dip=None, dip_value=None,
            t_out=None):
    """Paint a thin straight thread from ``p0`` to ``p1``, deterministically.

    ``dip`` optionally darkens a stretch centred at that fraction along the thread, which is
    what separates `brightest` from `geodesic`: an OFF-CENTRE intensity minimum gives the two
    rules different places to cut. ``t_out`` records the fraction-along for every voxel
    painted, so the bake-off can report WHERE each rule cut rather than merely whether it
    was right.
    """
    (y0, x0), (y1, x1) = p0, p1
    n = int(2 * max(abs(y1 - y0), abs(x1 - x0))) + 1
    for i in range(n + 1):
        t = i / n
        y, x = int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))
        v = value
        if dip is not None and abs(t - dip) < 0.05:
            v = dip_value
        sy, sx = slice(y - half, y + half + 1), slice(x - half, x + half + 1)
        img[sy, sx] = v
        if cid:
            truth[sy, sx] = cid
        if t_out is not None:
            t_out[sy, sx] = t


def _toy() -> dict:
    """A fibroblast-shaped fixture built so each rule can fail in its OWN way.

    Deterministic (no RNG), per this repo's selftest convention. Five elements, each present
    to discriminate one thing:

    * **cell 1 + cell 2**, elongated ellipse bodies -- the ~80% that IS captured.
    * **an outer thread on each**, unambiguously that cell's -- tests reach.
    * **cell 3**, a small body sitting NEAR cell 1's thread but not connected to it -- tests
      whether a rule steals across a gap. Truth is cell 1, since that is what the signal is
      physically continuous with.
    * **a bridge** joining cell 1 and cell 2, with its intensity minimum at 35% along rather
      than the middle -- the case `fragment` cannot represent at all. Deliberately EXCLUDED
      from the accuracy score and reported separately as *where* each rule cut it, because a
      symmetric claim about a connecting thread is a judgement, not a ground truth.
    * **a debris speck**, real signal connected to nothing -- tests invention.
    """
    H, W = 170, 340
    img = np.zeros((H, W), dtype=np.float64)
    truth = np.zeros((H, W), dtype=np.int64)
    bodies = np.zeros((H, W), dtype=np.int64)
    bridge_t = np.full((H, W), np.nan)
    yy, xx = np.ogrid[:H, :W]

    cells = {1: (60, 85, 38, 14), 2: (110, 250, 38, 14), 3: (30, 24, 9, 9)}
    for cid, (cy, cx, rx, ry) in cells.items():
        ell = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        img[ell] = 210.0
        truth[ell] = cid
        bodies[ell] = cid

    # outer threads -- unambiguous
    _thread(img, truth, (60, 85 - 38), (46, 8), 1, 62.0)          # cell 1, leftward
    _thread(img, truth, (110, 250 + 38), (124, 334), 2, 62.0)     # cell 2, rightward

    # the bridge: cell 1's right pole to cell 2's left pole, dim minimum at 35% along
    bmask_before = ~np.isnan(bridge_t)
    _thread(img, truth, (60, 85 + 36), (110, 250 - 36), 0, 62.0,
            dip=0.35, dip_value=36.0, t_out=bridge_t)
    del bmask_before
    bridge = ~np.isnan(bridge_t) & (bodies == 0)

    # debris -- real signal, connected to nothing, belongs to no cell
    img[22:27, 300:305] = 88.0

    return dict(img=img, bodies=bodies, truth=truth, bridge=bridge, bridge_t=bridge_t)


def _leftover(img: np.ndarray, labels: np.ndarray, level: float) -> np.ndarray:
    """Signal above `level` that the current segmentation did not claim."""
    return (img > level) & (labels == 0)


def _score(got, truth, bodies, bridge) -> dict:
    """Agreement over the voxels actually in play, excluding the bridge.

    Body voxels are excluded because every rule keeps them by construction -- including them
    would push every score toward 1.0 and hide the differences. Bridge voxels are excluded
    because their truth is a convention, not a fact; they get their own metric.
    """
    play = (bodies == 0) & ~bridge
    t, g = truth[play], got[play]
    total = int(play.sum())
    return dict(
        correct=int((g == t).sum()),
        missed=int(((t > 0) & (g == 0)).sum()),                  # real thread left behind
        stolen=int(((t > 0) & (g > 0) & (g != t)).sum()),         # given to the wrong cell
        invented=int(((t == 0) & (g > 0)).sum()),                 # debris/background claimed
        total=total,
        accuracy=int((g == t).sum()) / max(total, 1))


def _bridge_cut(got, bridge, bridge_t) -> dict:
    """Where along the bridge each rule handed over from cell 1 to cell 2."""
    sel = bridge & (got > 0)
    if not sel.any():
        return dict(to_c1=0.0, cut_t=float("nan"), claimed=0.0)
    t = bridge_t[sel]
    g = got[sel]
    to_c1 = float((g == 1).mean())
    cut = t[g == 1]
    return dict(to_c1=to_c1,
                cut_t=float(cut.max()) if cut.size else 0.0,
                claimed=float(sel.sum()) / float(bridge.sum()))


def run_synthetic(level: float = 30.0, retract: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    if retract:
        D.log(kind="error",
              title="RETRACTED: the geodesic row of e001 was my bug, not geodesic's "
                    "behaviour",
              body="e001 reported geodesic at 99.2% with 274 voxels given to the wrong "
                   "cell, and its figure showed BOTH of cell 2's threads coming out in cell "
                   "1's colour. That is not what geodesic assignment does - it is what my "
                   "implementation did.",
              why="I read `MCP_Geometric.find_costs`'s `traceback` as an index into the "
                  "list of start points and used it to look up each seed's label. It is "
                  "not: it holds neighbour-OFFSET indices, -1..7 for an 8-neighbourhood, so "
                  "the lookup indexed an 8-element space into a several-hundred-element seed "
                  "array and returned an unrelated id. Retracted: geodesic's 99.2% / 274 "
                  "figure, and the geodesic panel of e001's figure. The correct multi-source "
                  "form is a watershed on a CONSTANT cost surface, which I checked against "
                  "a per-label MCP_Geometric reference (0 disagreements). Everything else in "
                  "e001 stands, but the fixture it used could not separate the rules anyway "
                  "- see the next entry.",
              evidence=[{"type": "code",
                         "text": "traceback dtype: int16 unique: [-1 0 1 2 3 4 5 6 7]\n"
                                 "n starts: 2          <- only 2 seeds\n"
                                 "offsets len: 8       <- but 8 distinct traceback values\n"
                                 "--> values are neighbour-OFFSET indices, not start "
                                 "indices: True"}],
              status="info")

    t = _toy()
    img, bodies, truth = t["img"], t["bodies"], t["truth"]
    bridge, bridge_t = t["bridge"], t["bridge_t"]
    left = _leftover(img, bodies, level)
    sampling = (1.0, 1.0)

    results = {r: _assign(r, img, bodies, left, sampling) for r in RULES}
    scores = {r: _score(results[r], truth, bodies, bridge) for r in RULES}
    cuts = {r: _bridge_cut(results[r], bridge, bridge_t) for r in RULES}

    lut = ListedColormap([(0.05, 0.06, 0.09), (0.20, 0.83, 0.60),
                          (0.98, 0.45, 0.52), (0.42, 0.65, 0.99)])
    panels = [("signal (what is actually there)", img, "gray"),
              ("current segmentation - bodies only", bodies, lut),
              ("leftover to redistribute", left.astype(int) * 3, lut),
              ("TRUTH by construction (bridge excluded)", truth, lut)]
    panels += [(f"{r}  -  {scores[r]['accuracy']:.1%} right, "
                f"{scores[r]['stolen']} stolen, {scores[r]['invented']} invented",
                results[r], lut) for r in RULES]

    f, axes = plt.subplots(4, 2, figsize=(14, 9.6), dpi=118)
    for ax, (title, arr, cm) in zip(axes.ravel(), panels):
        ax.imshow(arr, cmap=cm, interpolation="nearest",
                  vmin=0, vmax=(3 if cm is lut else None))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    f.suptitle("Four ways to hand leftover signal back to a label - fixture built so each "
               "rule fails in its own way", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = os.path.join(LOG_ROOT, "img", "rules_synthetic_v2.png")
    f.savefig(p1); plt.close(f)

    # â”€â”€ the bridge, zoomed: where did each rule cut it? â”€â”€
    ys, xs = np.where(bridge)
    y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 13, img.shape[0])
    x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 13, img.shape[1])
    f, axes = plt.subplots(5, 1, figsize=(11, 8.4), dpi=118)
    axes[0].imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    axes[0].set_title("the bridge, as signal - the dim notch at 35% along is the "
                      "intensity minimum", fontsize=9)
    for ax, r in zip(axes[1:], RULES):
        ax.imshow(results[r][y0:y1, x0:x1], cmap=lut, vmin=0, vmax=3,
                  interpolation="nearest")
        c = cuts[r]
        ax.set_title(f"{r}  -  {c['to_c1']:.0%} of the bridge to cell 1, "
                     f"handover at {c['cut_t']:.0%} along", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    f.suptitle("Where each rule cuts a thread that JOINS two cells", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = os.path.join(LOG_ROOT, "img", "rules_bridge_zoom.png")
    f.savefig(p2); plt.close(f)

    rows = [[r, f"{scores[r]['accuracy']:.1%}", scores[r]["missed"], scores[r]["stolen"],
             scores[r]["invented"], f"{cuts[r]['to_c1']:.0%}"] for r in RULES]
    clean = [r for r in RULES
             if scores[r]["stolen"] == 0 and scores[r]["invented"] == 0
             and scores[r]["missed"] == 0]

    D.ask(title="Four assignment rules on a discriminating fixture - which one do you want "
                "on your real frames?",
          body="The previous fixture could not tell these apart: its threads were all "
               "unambiguous, so two rules scored 100% for free. This one adds the two cases "
               "that separate them - a small cell sitting near another cell's thread but "
               "not connected to it, and a thread that JOINS two cells with its dim point "
               "off-centre at 35% along.",
          why="This choice fixes the node's sockets and its footprint, so it is worth "
              "getting right before any of it is written. It also settles whether "
              "`transform.grow_points` could have done this job: that node grows by a fixed "
              "physical reach, so reaching a 100 px thread means inflating the cell body by "
              "100 px as well - `missed` and `invented` cannot both go to zero "
              "geometrically. Note the bridge column has no right answer, only "
              "consequences: whatever fraction a rule gives to cell 1 is area that comes "
              "off cell 2 in every downstream measurement.",
          evidence=[{"type": "table",
                     "cols": ["rule", "unambiguous right", "missed", "wrong cell",
                              "invented", "bridge -> cell 1"],
                     "rows": rows,
                     "caption": "First four columns are scored on the unambiguous voxels "
                                "only. `missed` = real thread left as background; `wrong "
                                "cell` = handed to a cell it has no signal path to; "
                                "`invented` = debris claimed. The last column is not right "
                                "or wrong - it is how the bridge got divided."},
                    {"type": "kv",
                     "rows": [["clean on all three error types",
                               ", ".join(clean) or "none",
                               "no missed, no stolen, no invented"],
                              ["bridge dim point", "35% along",
                               "where `brightest` should cut, if brightness is the right "
                               "signal to cut on"],
                              ["bridge midpoint", "50% along",
                               "where `geodesic` should cut - it ignores brightness"]]}],
          images=[{"src": "img/rules_synthetic_v2.png",
                   "caption": "Look at cell 3, the small blue body at top left, and at "
                              "cell 1's thread passing beside it. Under `euclidean` the "
                              "near half of that thread turns blue - stolen across a gap it "
                              "has no signal connection to. Then look at the debris speck "
                              "top right: `euclidean` claims that too."},
                  {"src": "img/rules_bridge_zoom.png",
                   "caption": "One row per rule, same crop. `fragment` gives the ENTIRE "
                              "bridge to one cell - the flat 0% or 100% - which is the "
                              "column that matters if you measure per-cell area. Compare "
                              "where `brightest` hands over (at the dim notch) against "
                              "`geodesic` (near the middle)."}],
          asks="Which rule should I build as the default - and should the others be "
               "selectable modes or left out? If you would rather judge this on your own "
               "frames first, say so and tell me which segmentation method produces your "
               "current ~80% result, and I will run the same four rules on real data before "
               "you decide.")

    print(f"\n{'rule':<11} {'acc':>7} {'missed':>7} {'stolen':>7} {'invented':>9} "
          f"{'bridge->c1':>11}")
    for r in RULES:
        s, c = scores[r], cuts[r]
        print(f"{r:<11} {s['accuracy']:>6.1%} {s['missed']:>7} {s['stolen']:>7} "
              f"{s['invented']:>9} {c['to_c1']:>10.0%}")
    print(f"\nclean on all three error types: {', '.join(clean) or 'none'}")
    print(D.render())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• stage 2: the real frames â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#
# Split in two on purpose. `--segment` runs the operator's own preprocessing chain and one
# CellSAM pull, which is minutes of work, and caches the result; `--real` reads that cache
# and applies the four rules in seconds. So the rules can be re-run, re-scored and
# re-rendered without paying for inference again, and a crash mid-render costs nothing.

#: The operator's chain, read off `Fibroblast_tracking.nd2graph.json` rather than invented,
#: plus the CellSAM segmentation they confirmed (tiled + fast).
ND2 = ("C:/Users/McGheeLab - Analysis/Desktop/TF_ELISA_Final/20260727_193054_245/"
       "WellA3_Time00000_ChannelGFP_Seq0002.nd2")
ZS_WEIGHTS = os.path.join(REPO, "models", "Fibroblast_10x_c0.weights.h5")
CACHE = os.path.join(LOG_ROOT, "cache")


def run_segment(times: int = 3, position: int = 0, out_name: str = "labels") -> None:
    """Pull the real chain once and cache (preprocessed image, labels) to npz.

    The chain is the operator's: ZS-DeconvNet -> temporal gain -> CellSAM (tiled, fast).
    `enhance.flatten_field` is intentionally absent -- they took it out of the bake-off.
    `temporal_gain` fits over T, so more than one timepoint is loaded even when only one is
    wanted for the figures: a frame whose gain was fitted on a single sample would not be the
    image the operator actually looks at.
    """
    import time

    import nd2

    from nodegraph.dataset import AxisSizes, Dataset
    from nodegraph.domains import Domain
    from nodegraph.engine import Engine
    from nodegraph.graph import Graph, NodeInstance
    from nodegraph.metadata import MetaEnvelope
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.registry import OutDataset, define_node

    from nodegraph.kernels import cellsam_segment as CS
    if not CS.cellsam_available():
        raise SystemExit("cellSAM is not importable in this venv - see "
                         "scripts/_cellsam_smoke.py for the install and token steps.")
    print(f"cellsam device -> {CS.resolve_device()}")
    if not os.path.isfile(ZS_WEIGHTS):
        raise SystemExit(f"ZS-DeconvNet weights not found: {ZS_WEIGHTS}")

    # â”€â”€ the pixels, straight from the ND2 (axes T,P,Y,X for this file) â”€â”€
    with nd2.ND2File(ND2) as f:
        d = f.to_dask()
        print(f"nd2 dask shape {d.shape} (T,P,Y,X)")
        sub = np.asarray(d[0:times, position:position + 1])
    arr = np.transpose(sub, (1, 0, 2, 3))[:, :, None, None]      # (M,T,Z,C,Y,X)
    m, t, _z, _c, h, w = arr.shape
    print(f"loaded subset {arr.shape} {arr.dtype}  (position {position}, {t} timepoints)")

    # â”€â”€ calibration from the file itself, never hardcoded â”€â”€
    try:
        from nodelab_v2.ingest import read_calibration
        cal = dict(read_calibration(ND2))
    except Exception as exc:                        # noqa: BLE001 - reported, not hidden
        print(f"  ! read_calibration failed ({type(exc).__name__}: {exc}); "
              f"falling back to the values recorded for this file")
        cal = {"pixel_size_um": 1.7182777601481225, "bit_depth": 12}
    print("  calibration:", {k: cal.get(k) for k in
                             ("pixel_size_um", "bit_depth", "channel_emission_nm")})

    ax = AxisSizes(m=m, t=t, z=1, c=1, y=h, x=w)
    ds = Dataset(axes=ax, metadata=dict(cal)).with_image(ArrayProvider(arr))
    define_node("io.reclaimseed", "Seed", outputs=[OutDataset()])

    g = Graph()
    g.add(NodeInstance("S", "io.reclaimseed"))
    g.add(NodeInstance("n2", "enhance.zs_deconvnet",
                       params={"weights_path": ZS_WEIGHTS, "background": 40.0,
                               "norm_low": 0.0, "tile": 256}))
    g.add(NodeInstance("n3", "enhance.temporal_gain", params={"local_sigma": 10.0}))
    #  `enhance.flatten_field` is deliberately NOT in this chain: the operator dropped it
    #  from the bake-off, so CellSAM sees the temporal-gain output directly. The leftover cut
    #  below therefore sits on un-flattened intensity, which is worth remembering if the
    #  Otsu level looks field-dependent across positions.
    g.add(NodeInstance("n5", "analysis.segment",
                       modes={"dim": "2D", "method": "cellsam"},
                       params={"name": out_name, "tile": True, "fast": True}))
    g.connect("S", "n2"); g.connect("n2", "n3"); g.connect("n3", "n5")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds},
                 meta_seeds={"S": MetaEnvelope(axes=ax, metadata=dict(cal))})

    print("\npulling the preprocessed image (n3, temporal gain)...")
    t0 = time.time()
    pre = eng.pull("n3")
    #  Realize every unit of the preprocessed stack through the provider rather than
    #  reaching for a private array: the node may be backed by a disk store.
    #  Geometry from the PREPROCESSED dataset: ZS-DeconvNet upsamples 2x laterally and
    #  halves pixel_size_um, so the input's h/w are the wrong shape downstream.
    ho, wo = int(pre.axes.y), int(pre.axes.x)
    cal["pixel_size_um"] = float(pre.metadata.get("pixel_size_um")
                                 or cal.get("pixel_size_um") or 0.0)
    if (ho, wo) != (h, w):
        print(f"  preprocessing changed geometry: {h}x{w} -> {ho}x{wo}, now "
              f"{cal['pixel_size_um']:.4f} um/px")
    #  First arg is the pyramid LEVEL; 0 is full resolution.
    prov = pre.image
    img = np.stack([np.stack([np.asarray(prov.get_region(0, mi, ti, 0, 0, 0, ho, 0, wo))
                              for ti in range(t)]) for mi in range(m)])
    print(f"  preprocessed in {time.time() - t0:.1f}s -> {img.shape} {img.dtype}")

    print("pulling CellSAM segmentation (n5) - this is the slow one...")
    t0 = time.time()
    out = eng.pull("n5")
    print(f"  segmented in {time.time() - t0:.1f}s")
    raster = out.get(Domain.VOXEL, out_name)
    if raster is None:
        raise SystemExit(f"no Voxel layer {out_name!r} on the segmentation output")
    lab = np.asarray(raster.values).reshape(m, t, ho, wo)
    print(f"  labels -> {lab.shape}, {int(lab.max())} cells total")
    for ti in range(t):
        print(f"    t={ti}: {len(np.unique(lab[0, ti])) - 1} cells")

    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"real_p{position}_t{t}.npz")
    np.savez_compressed(p, img=img.astype(np.float32), lab=lab.astype(np.int32),
                        pixel_size_um=float(cal.get("pixel_size_um") or 0.0),
                        bit_depth=float(cal.get("bit_depth") or 0.0))
    print(f"\ncached -> {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    print("now run:  --real")


def _fragment_census(labels: np.ndarray, leftover: np.ndarray) -> dict:
    """How many distinct labels each connected leftover fragment abuts.

    THE decisive real-data measurement for `fragment` vs the splitting rules. A fragment
    touching one label is unambiguous and every rule agrees on it. One touching two or more
    is a bridge, and that is precisely where `fragment` must give the whole thing to a single
    cell and move area off its neighbour. One touching none cannot be claimed by anybody
    honestly, so it is a floor on what no rule should recover.
    """
    from scipy import ndimage as ndi
    frags, n = ndi.label(leftover)
    if not n:
        return dict(n=0, touch={0: 0, 1: 0, 2: 0}, area={0: 0, 1: 0, 2: 0})
    fid, _lid, _cnt = _frag_label_contacts(frags, labels)
    #  one row per DISTINCT (fragment,label) pair, so counting rows per fragment counts the
    #  distinct labels it touches
    n_distinct = np.bincount(fid, minlength=n + 1)[1:n + 1]
    sizes = np.bincount(frags.ravel(), minlength=n + 1)[1:n + 1]
    key = np.where(n_distinct == 0, 0, np.where(n_distinct == 1, 1, 2))
    return dict(n=n,
                touch={k: int((key == k).sum()) for k in (0, 1, 2)},
                area={k: int(sizes[key == k].sum()) for k in (0, 1, 2)})


def _segment_plane(plane: np.ndarray, cal: dict, *, deconv: bool,
                   out_name: str = "labels") -> "tuple[np.ndarray, np.ndarray, float]":
    """CellSAM (tiled, fast) on ONE plane, optionally ZS-DeconvNet first.

    The operator's chain is deconv -> temporal gain -> CellSAM. Temporal gain fits over T
    and cannot run on a single frame, so this reproduces as much of it as one plane allows
    and says so rather than pretending the chains match.
    """
    import time

    from nodegraph.dataset import AxisSizes, Dataset
    from nodegraph.domains import Domain
    from nodegraph.engine import Engine
    from nodegraph.graph import Graph, NodeInstance
    from nodegraph.metadata import MetaEnvelope
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.registry import OutDataset, define_node

    h, w = plane.shape
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=h, x=w)
    ds = Dataset(axes=ax, metadata=dict(cal)).with_image(
        ArrayProvider(plane.reshape(1, 1, 1, 1, h, w)))
    define_node("io.reclaimseed", "Seed", outputs=[OutDataset()])

    g = Graph()
    g.add(NodeInstance("S", "io.reclaimseed"))
    last = "S"
    if deconv:
        g.add(NodeInstance("n2", "enhance.zs_deconvnet",
                           params={"weights_path": ZS_WEIGHTS, "background": 40.0,
                                   "norm_low": 0.0, "tile": 256}))
        g.connect("S", "n2")
        last = "n2"
    g.add(NodeInstance("n5", "analysis.segment",
                       modes={"dim": "2D", "method": "cellsam"},
                       params={"name": out_name, "tile": True, "fast": True}))
    g.connect(last, "n5")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds},
                 meta_seeds={"S": MetaEnvelope(axes=ax, metadata=dict(cal))})

    t0 = time.time()
    pre = eng.pull(last)
    #  Read the geometry from the PREPROCESSED dataset, never from the input plane.
    #  ZS-DeconvNet is axis-changing: it returns 2x in each lateral axis (a 1024 frame comes
    #  back 2048) and halves pixel_size_um to match. Reusing the input shape here raised
    #  "cannot reshape array of size 4194304 into shape (1024,1024)" after a 102-minute run.
    ho, wo = int(pre.axes.y), int(pre.axes.x)
    px_out = float(pre.metadata.get("pixel_size_um") or cal.get("pixel_size_um") or 0.0)
    img = np.asarray(pre.image.get_region(0, 0, 0, 0, 0, 0, ho, 0, wo)).astype(np.float64)
    scale = f" [{h}x{w} -> {ho}x{wo}, {cal.get('pixel_size_um'):.4f} -> {px_out:.4f} um/px]" \
        if (ho, wo) != (h, w) else ""
    print(f"  preprocessed ({'deconv' if deconv else 'as-is'}) in "
          f"{time.time() - t0:.1f}s  range {img.min():.4g}-{img.max():.4g}{scale}")

    t0 = time.time()
    out = eng.pull("n5")
    raster = out.get(Domain.VOXEL, out_name)
    if raster is None:
        raise SystemExit(f"no Voxel layer {out_name!r} on the segmentation output")
    lab = np.asarray(raster.values).reshape(ho, wo).astype(np.int64)
    print(f"  CellSAM in {time.time() - t0:.1f}s -> {len(np.unique(lab)) - 1} cells")
    return img, lab, px_out


def run_tif(path: str, deconv: bool = True, level_mult: float = 1.0) -> None:
    """Segment one exported TIF frame and put all four rules on it."""
    import tifffile

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    plane = np.asarray(tifffile.imread(path))
    while plane.ndim > 2:
        plane = plane[0]
    #  Calibration out of the ImageJ header rather than assumed. This file carries
    #  spacing=1.7182777601481225, the WellA3 pixel size, so it is a frame off that
    #  acquisition; the 32..4095 range says raw 12-bit camera data, not a deconvolved float.
    px = 0.0
    with tifffile.TiffFile(path) as tf:
        desc = tf.pages[0].tags.get("ImageDescription")
        for line in (str(desc.value).splitlines() if desc else []):
            if line.startswith("spacing="):
                px = float(line.split("=", 1)[1])
    if not px:
        px = 1.7182777601481225
        print(f"  ! no spacing in the TIF header; assuming {px} um")
    cal = {"pixel_size_um": px, "bit_depth": 12}
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f"{path}\n  {plane.shape} {plane.dtype}, {plane.min()}-{plane.max()}, "
          f"pixel {px:.4f} um")

    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"tif_{stem}_{'dc' if deconv else 'raw'}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        img, labels = z["img"].astype(np.float64), z["lab"].astype(np.int64)
        px = float(z["pixel_size_um"]) or px      # post-preprocessing scale, not the file's
        print(f"  (from cache {cache}; {img.shape} at {px:.4f} um/px)")
    else:
        img, labels, px_out = _segment_plane(plane, cal, deconv=deconv)
        #  The cached pixel size is the POST-preprocessing one -- deconv halves it -- because
        #  every um-based number downstream (tile scale, areas) must use the scale the
        #  labels actually live at.
        px = px_out or px
        np.savez_compressed(cache, img=img.astype(np.float32),
                            lab=labels.astype(np.int32), pixel_size_um=px, bit_depth=12.0)
        print(f"  cached -> {cache}")

    src = (f"`{os.path.basename(path)}`, a single exported frame, segmented at "
           f"{px:.4f} um/px "
           f"({'ZS-DeconvNet (which upsamples 2x, halving the pixel size) then '
              if deconv else 'no deconvolution, '}CellSAM tiled+fast). "
           f"Temporal gain is absent because it fits over T and this is one frame.")
    #  The deconvolved and as-saved runs are two different experiments on the same file, so
    #  they must not overwrite each other's figures.
    _compare_and_log(img, labels, px, tag=f"tif_{stem}_{'dc' if deconv else 'raw'}",
                     source_desc=src, level_mult=level_mult)


def run_real(position: int = 0, times: int = 3, frame: int = 0,
             level_mult: float = 1.0) -> None:
    """Apply the four rules to the cached real ND2 frames, render, and log."""
    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    p = os.path.join(CACHE, f"real_p{position}_t{times}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"no cache at {p} - run --segment first")
    z = np.load(p)
    img_all, lab_all = z["img"], z["lab"]
    px = float(z["pixel_size_um"])
    img, labels = img_all[0, frame].astype(np.float64), lab_all[0, frame].astype(np.int64)
    print(f"frame p{position} t{frame}: {img.shape}, "
          f"{len(np.unique(labels)) - 1} cells, pixel {px:.4f} um")
    src = (f"the ND2 itself, position {position} t={frame}: ZS-DeconvNet -> temporal gain "
           f"-> CellSAM (tiled, fast), the operator's own chain minus `flatten_field`.")
    _compare_and_log(img, labels, px, tag=f"real_p{position}_t{frame}", source_desc=src,
                     level_mult=level_mult)


def run_sweep(tif: str, deconv: bool = False, levels: str = "") -> None:
    """Sweep the leftover cut and watch the rules diverge.

    THE reason this exists: at Otsu the leftover on a real frame was 0.91% of the image and
    all four rules produced visually identical output, because Otsu separates bright cell
    BODIES from everything else and the thin polar processes being recovered are dimmer than
    that. A single-level comparison at the top of the range cannot distinguish the rules, and
    reporting it as if it could is the clipped-sweep failure this project has hit before. So
    the level is swept across the range where the processes actually live, and the sweep is
    reported even where it is unflattering.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.filters import threshold_otsu

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    stem = os.path.splitext(os.path.basename(tif))[0]
    cache = os.path.join(CACHE, f"tif_{stem}_{'dc' if deconv else 'raw'}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"no cache at {cache} - run --tif first")
    z = np.load(cache)
    img, labels = z["img"].astype(np.float64), z["lab"].astype(np.int64)
    px = float(z["pixel_size_um"])
    otsu = float(threshold_otsu(img))
    bg = float(np.percentile(img, 5))
    n_cells = len(np.unique(labels)) - 1
    print(f"{stem}: {n_cells} cells, otsu {otsu:.4g}, 5th pct {bg:.4g}, "
          f"max {img.max():.0f}")

    if levels:
        lv = [float(x) for x in levels.split(",")]
    else:
        #  From just above the background floor up THROUGH Otsu, so the optimum cannot sit
        #  on the edge of the box: if the best level were the lowest or highest tried, the
        #  range itself would be the finding.
        #
        #  NOT rounded to a fixed number of decimals. That assumed an integer count scale:
        #  after ZS-DeconvNet the image is 0..1 floats, `round(x, 1)` collapsed four of the
        #  six levels onto the same value, and the sweep silently reported the same row four
        #  times as though it had covered a range.
        lv = [bg + (otsu - bg) * f for f in (0.10, 0.25, 0.40, 0.60, 0.80, 1.0)]
    print("levels:", lv)

    sampling = (1.0, 1.0)
    base_area = np.bincount(labels.ravel())
    rows, per_level = [], []
    for level in lv:
        left = _leftover(img, labels, level)
        cen = _fragment_census(labels, left)
        tot = max(int(left.sum()), 1)
        bridge_frac = 100.0 * cen["area"][2] / tot
        res = {}
        worst = {}
        for r in RULES:
            got = _assign(r, img, labels, left, sampling)
            res[r] = got
            new_area = np.bincount(got.ravel(), minlength=len(base_area))
            grew = base_area[1:] > 0
            pct = np.zeros(int(grew.size))
            pct[grew] = (100.0 * (new_area[1:][grew] - base_area[1:][grew])
                         / base_area[1:][grew])
            worst[r] = float(pct.max()) if pct.size else 0.0
        #  THE question the whole bake-off is really asking: on THIS data, how often do the
        #  rules actually put a pixel on a different cell? Bridge fraction and worst-cell
        #  growth are proxies for it; this is the thing itself. Measured over pixels at
        #  least one rule claimed, against `brightest` as the reference.
        dis = {}
        for r in RULES:
            if r == "brightest":
                continue
            claimed = (res[r] > 0) | (res["brightest"] > 0)
            claimed &= left
            n_c = max(int(claimed.sum()), 1)
            dis[r] = 100.0 * int((res[r][claimed] != res["brightest"][claimed]).sum()) / n_c

        per_level.append((level, left, res, cen))
        rows.append([f"{level:.4g}", int(left.sum()),
                     f"{100 * left.mean():.1f}%", cen["n"], f"{bridge_frac:.0f}%",
                     f"{dis['geodesic']:.1f}%", f"{dis['fragment']:.1f}%",
                     f"{dis['euclidean']:.1f}%",
                     f"+{worst['brightest']:.0f}%", f"+{worst['fragment']:.0f}%"])
        print(f"  level {level:>9.4g}: leftover {int(left.sum()):>8} px "
              f"({100 * left.mean():>5.1f}%), {cen['n']:>5} frags, "
              f"{bridge_frac:>3.0f}% bridges | vs brightest: geo "
              f"{dis['geodesic']:>4.1f}% frag {dis['fragment']:>4.1f}% eucl "
              f"{dis['euclidean']:>4.1f}% | worst +{worst['brightest']:.0f}%/"
              f"+{worst['fragment']:.0f}%")

    # â”€â”€ figure: one row per level, brightest vs fragment on the same crop â”€â”€
    rng = np.random.default_rng(0)
    lut = ListedColormap(np.vstack([[0.05, 0.06, 0.09],
                                    rng.uniform(0.35, 1.0,
                                                size=(max(int(labels.max()), 1) + 1, 3))]))
    vmax = max(int(labels.max()), 1)
    from scipy import ndimage as ndi
    dens = ndi.uniform_filter(per_level[len(per_level) // 2][1].astype(float), 96)
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    s = 150
    y0, y1 = max(cy - s, 0), min(cy + s, img.shape[0])
    x0, x1 = max(cx - s, 0), min(cx + s, img.shape[1])

    n = len(per_level)
    f, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n), dpi=104)
    for i, (level, left, res, cen) in enumerate(per_level):
        A = axes[i] if n > 1 else axes
        A[0].imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        A[0].set_ylabel(f"level {level:.4g}", fontsize=10)
        A[0].set_title("signal" if i == 0 else "", fontsize=9)
        A[1].imshow(left[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        A[1].set_title(f"leftover  {100 * left.mean():.1f}% of frame"
                       if i == 0 else f"{100 * left.mean():.1f}%", fontsize=9)
        A[2].imshow(res["brightest"][y0:y1, x0:x1], cmap=lut, vmin=0, vmax=vmax,
                    interpolation="nearest")
        A[2].set_title("brightest" if i == 0 else "", fontsize=9)
        A[3].imshow(res["fragment"][y0:y1, x0:x1], cmap=lut, vmin=0, vmax=vmax,
                    interpolation="nearest")
        A[3].set_title("fragment" if i == 0 else "", fontsize=9)
        for ax in A:
            ax.set_xticks([]); ax.set_yticks([])
    f.suptitle(f"Leftover level sweep - {stem} ({'deconv' if deconv else 'as saved'}), "
               f"{2 * s * px:.0f} um crop. Otsu = {otsu:.4g}", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(LOG_ROOT, "img", f"sweep_{stem}_{'dc' if deconv else 'raw'}.png")
    f.savefig(p); plt.close(f)

    D.log(kind="found",
          title="The leftover LEVEL, not the assignment rule, is what decides how much of "
                "a cell you get back",
          body=f"Swept the leftover cut from just above background ({bg:.4g}) up through "
               f"Otsu ({otsu:.4g}) on {stem}, {n_cells} CellSAM cells. At Otsu the leftover "
               f"is under 1% of the frame and every rule produces the same picture; the "
               f"thin polar processes are dimmer than Otsu and are simply not in the "
               f"leftover at that setting.",
          why="I first reported this comparison at Otsu alone, which is the top of the "
              "range, and at that level the four rules are indistinguishable - a "
              "single-point comparison there would have justified picking whichever rule I "
              "preferred. The column that matters is `% of leftover area on bridges`: "
              "leftover pieces touching two or more cells. It is already 57% at Otsu and "
              "rises as the level drops, because dimmer signal joins neighbouring cells into "
              "one connected sheet. Every one of those pixels is area that `fragment` must "
              "give entirely to a single cell.",
          evidence=[{"type": "table",
                     "cols": ["level", "leftover px", "of frame", "fragments",
                              "% area on bridges", "geodesic differs", "fragment differs",
                              "euclidean differs", "worst cell brightest",
                              "worst cell fragment"],
                     "rows": rows,
                     "caption": "The `differs` columns are the fraction of reclaimed pixels "
                                "each rule puts on a DIFFERENT cell than `brightest` does - "
                                "the direct measure of whether this choice matters here, "
                                "rather than a proxy for it. `worst cell` is the largest "
                                "single-cell area increase, and it is a warning: a label "
                                "that grows several-fold has flooded a connected web, which "
                                "is what the reach cap exists to bound."}],
          images=[{"src": f"img/sweep_{stem}_{'dc' if deconv else 'raw'}.png",
                   "caption": "Rows are increasing leftover level, top = dimmest cut. Column "
                              "2 is what is being redistributed: at the bottom row it is "
                              "almost nothing, at the top it is a connected web between "
                              "cells. Compare columns 3 and 4 in the top rows - that is "
                              "where `fragment` starts handing whole shared regions to one "
                              "cell."}],
          status="info")
    print(D.render())


#: Candidate selectors -- how we decide WHICH leftover signal is a plausible extension of a
#: cell, before any rule decides WHOSE it is. The operator's instruction was to improve this
#: first, and the level sweep agrees: the cut dominates the rule.
CANDIDATES = ("global", "attached", "reach", "tubular")


def _candidates(kind: str, img, labels, *, level, px, reach_um=25.0,
                lo_frac=0.45, ridge_keep=0.55):
    """A boolean mask of leftover signal considered a possible cell extension.

    * ``global``   -- everything above `level` that is unlabelled. The baseline, and the one
      that cannot tell a cell process from a speck of debris on the other side of the field.
    * ``attached`` -- hysteresis by ATTACHMENT: take a permissive cut at `lo_frac * level`,
      then keep only the connected pieces that physically touch a cell. This is what
      "extension of a cell" literally means, and it is strictly more informative than a
      brighter global cut: it reaches FURTHER down in intensity along a real process while
      discarding bright debris that touches nothing. On this frame the global cut spent
      ~11-14% of its area on fragments touching no label at all.
    * ``reach``    -- `attached`, then clipped to within `reach_um` of the cell it came from.
      Bounds the runaway growth the sweep exposed (+1067% on one cell) without lowering the
      intensity cut, because a process has a plausible physical length and a connected haze
      does not.
    * ``tubular``  -- `attached`, restricted to a Sato ridge response above `ridge_pct`.
      Thin processes are tubular; cell bodies and diffuse haze are not. Aimed squarely at
      the thin polar threads, at the cost of rejecting broad lamellar extensions.

    Every one of these is a CANDIDATE mask. None of them knows whether the region it selects
    is really part of a cell -- that needs a drawn boundary to score against (SKILL.md §4).
    """
    from scipy import ndimage as ndi

    unl = labels == 0
    if kind == "global":
        return (img > level) & unl

    lo = level * lo_frac
    loose = (img > lo) & unl
    #  keep only pieces touching a cell
    frags, n = ndi.label(loose)
    if not n:
        return np.zeros_like(loose)
    fid, _lid, _cnt = _frag_label_contacts(frags, labels)
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(fid)] = True
    attached = keep[frags]
    if kind == "attached":
        return attached

    if kind == "reach":
        d = ndi.distance_transform_edt(labels == 0, sampling=(px, px))
        return attached & (d <= reach_um)

    if kind == "tubular":
        from skimage.filters import sato
        #  sigmas in PIXELS spanning a plausible process half-width (~1-5 um here).
        sig = [s for s in (1.0, 1.5, 2.0, 3.0) if s * px <= 6.0] or [1.0]
        r = sato(img.astype(float), sigmas=sig, black_ridges=False)
        #  Keep the TOP `ridge_keep` of attached area by ridge response. Written first as a
        #  percentile of `ridge_pct - 90`, i.e. the 7th percentile, which kept 93% of the
        #  mask and made this selector a near-copy of `attached` while claiming to isolate
        #  thin processes.
        thr = (np.percentile(r[attached], 100.0 * (1.0 - ridge_keep))
               if attached.any() else 0.0)
        return attached & (r > thr)

    raise ValueError(kind)


def _tile_font(size: int):
    """A legible font for the tile digits, whatever is installed.

    PIL's built-in bitmap font is ~11 px and cannot be scaled, which made the 1/2 markers
    almost invisible on the first sheet -- the one thing the labeller has to read.
    """
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "segoeuib.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def run_label_sheet(tif: str, deconv: bool = False, level: float = 0.0,
                    max_tiles: int = 260, box_um: float = 165.0, zoom: int = 4) -> None:
    """Build a BLIND labelling sheet over the regions where brightest and geodesic disagree.

    Nothing measured is drawn on a tile. Each one shows the raw signal, the disputed piece,
    and the TWO candidate cells outlined and numbered -- and the two candidates are exactly
    the cells `brightest` and `geodesic` chose, with the 1/2 numbering shuffled per tile by a
    deterministic hash of the uid. So a label of "1" carries no information about which rule
    proposed that cell, and cannot anchor to my prediction; the join back to which rule was
    right happens afterwards, out of the manifest.

    Every tile is rendered at ONE fixed physical size and ONE global intensity mapping. Both
    matter: per-tile autoscaling would turn "this piece is brighter" into an artefact of the
    crop, and per-tile zoom would do the same to "this piece is bigger".
    """
    from PIL import Image, ImageDraw
    from skimage.filters import threshold_otsu

    import labelsheet as LS

    stem = os.path.splitext(os.path.basename(tif))[0]
    cache = os.path.join(CACHE, f"tif_{stem}_{'dc' if deconv else 'raw'}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"no cache at {cache} - run --tif first")
    z = np.load(cache)
    img, labels = z["img"].astype(np.float64), z["lab"].astype(np.int64)
    px = float(z["pixel_size_um"])
    if not level:
        bg, otsu = float(np.percentile(img, 5)), float(threshold_otsu(img))
        level = bg + (otsu - bg) * 0.40     # the sweep's most-contested setting
    print(f"{stem}: level {level:.4g}, pixel {px:.4f} um")

    from scipy import ndimage as ndi
    left = _leftover(img, labels, level)
    sampling = (1.0, 1.0)
    res_b = _assign("brightest", img, labels, left, sampling)
    res_g = _assign("geodesic", img, labels, left, sampling)
    frags, nfrag = ndi.label(left)
    print(f"  {int(left.sum())} leftover px in {nfrag} fragments")

    # â”€â”€ find the contested pieces â”€â”€
    #  A fragment is contested when the two rules give its BULK to different cells. Compared
    #  on the majority label per fragment rather than per pixel, because a one-pixel
    #  disagreement at a seam is not a question worth anyone's time.
    idx = np.argsort(frags.ravel(), kind="stable")
    sorted_f = frags.ravel()[idx]
    starts = np.searchsorted(sorted_f, np.arange(1, nfrag + 1), side="left")
    ends = np.searchsorted(sorted_f, np.arange(1, nfrag + 1), side="right")
    fb, fg = res_b.ravel(), res_g.ravel()

    def majority(vals):
        v = vals[vals > 0]
        if not v.size:
            return 0
        u, c = np.unique(v, return_counts=True)
        return int(u[int(np.argmax(c))])

    cand = []
    for k in range(nfrag):
        sl = idx[starts[k]:ends[k]]
        if sl.size < 6:                                # a handful of pixels decides nothing
            continue
        a, b = majority(fb[sl]), majority(fg[sl])
        if a and b and a != b:
            cand.append((k + 1, a, b, int(sl.size)))
    print(f"  {len(cand)} fragments where brightest and geodesic disagree on the bulk")
    if not cand:
        raise SystemExit("nothing contested at this level - try a lower --label-level")

    cand.sort(key=lambda r: -r[3])
    cand = cand[:max_tiles]

    # â”€â”€ render â”€â”€
    lo, hi = np.percentile(img, (1.0, 99.5))           # ONE mapping for every tile
    box = max(int(round(box_um / px)) | 1, 33)
    half = box // 2
    #  Separate directory per variant. The labeller persists answers in localStorage keyed by
    #  page path + tile uid, and uids are fragment ids -- which are not stable between the
    #  raw and deconvolved runs. Rebuilding in place would silently re-attach an answer given
    #  about one region to a completely different one, which is worse than losing it.
    out_dir = os.path.join(LOG_ROOT, "label_dc" if deconv else "label")
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    cen = ndi.center_of_mass(np.ones_like(frags), frags, [c[0] for c in cand])

    def edge(mask):
        return mask & ~ndi.binary_erosion(mask)

    manifest = []
    for (fid, a, b, area), (cy, cx) in zip(cand, cen):
        uid = f"f{fid:05d}"
        #  Deterministic per-uid shuffle of which candidate is shown as 1 vs 2.
        flip = (hash(uid) & 0xFFFF) % 2 == 1
        one, two = (b, a) if flip else (a, b)
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        y0 = min(max(y0, 0), img.shape[0] - box)
        x0 = min(max(x0, 0), img.shape[1] - box)
        sl = (slice(y0, y0 + box), slice(x0, x0 + box))

        g = np.clip((img[sl] - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        lab_c, frag_c = labels[sl], frags[sl]
        for cid, col in ((one, (255, 176, 32)), (two, (72, 164, 255))):
            rgb[edge(lab_c == cid)] = col
        #  OUTLINE the disputed piece; do not tint its interior. Its brightness is the
        #  evidence `brightest` acts on, so painting over it would hide the very thing the
        #  labeller needs to weigh.
        rgb[edge(frag_c == fid)] = (255, 64, 200)

        im = Image.fromarray(rgb).resize((box * zoom, box * zoom), Image.NEAREST)
        d = ImageDraw.Draw(im)
        font = _tile_font(int(15 * zoom / 3) + 14)
        for cid, col, txt in ((one, (255, 176, 32), "1"), (two, (72, 164, 255), "2")):
            m = lab_c == cid
            if not m.any():
                continue
            yy, xx = np.nonzero(m)
            tx, ty = xx.mean() * zoom, yy.mean() * zoom
            for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):   # dark halo, so the digit
                d.text((tx + ox, ty + oy), txt, fill=(0, 0, 0), font=font,
                       anchor="mm")                              # reads over bright cytoplasm
            d.text((tx, ty), txt, fill=col, font=font, anchor="mm")
        bar = int(round(50.0 / px)) * zoom            # 50 um scale bar, same on every tile
        d.line([(9, box * zoom - 10), (9 + bar, box * zoom - 10)], fill=(255, 255, 255),
               width=3)
        im.save(os.path.join(out_dir, "img", f"{uid}.png"))

        manifest.append(dict(uid=uid, group=("big" if area >= 60 else "small"),
                             frag=int(fid), area_px=int(area),
                             area_um2=round(area * px * px, 1),
                             cell_1=int(one), cell_2=int(two),
                             brightest_is=("1" if one == a else "2"),
                             geodesic_is=("1" if one == b else "2")))

    LS.build(
        out_dir=out_dir, manifest=manifest,
        question="Which cell does the highlighted piece belong to?",
        vocab=[("1", "cell 1 (orange)", "#fbbf24"), ("2", "cell 2 (blue)", "#60b0ff"),
               ("s", "shared - genuinely part of both", "#34d399"),
               ("x", "neither - not cell material", "#a78bfa"),
               ("?", "cannot tell", "#64748b")],
        strata_key="area_px",
        note=(f"Every tile is {box_um:.0f} um across at {px:.2f} um/px, same scale and same "
              f"brightness mapping throughout; the white bar is 50 um. The piece in question "
              f"is outlined in PINK - its interior is left alone so you can judge its "
              f"brightness. The orange 1 and blue 2 are the only two cells it could "
              f"plausibly belong to. Which cell gets which number is shuffled per tile, so "
              f"the numbering tells you nothing about what I predicted."),
        csv_name="reclaim_labels.csv",
        title="Reclaim: which cell owns this piece?")

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")
    D.log(kind="did",
          title=f"Built a blind labelling sheet over {len(manifest)} contested pieces",
          body=f"At level {level:.4g} on {stem}, {len(cand)} leftover fragments have their "
               f"bulk assigned to different cells by `brightest` and `geodesic`. Each tile "
               f"shows the raw signal, the disputed piece tinted white, and the two "
               f"candidate cells outlined in orange and blue.",
          why="Nothing else I can compute ranks these two rules - they disagree on about a "
              "third of reclaimed area and both were clean on the synthetic fixture. Labels "
              "score every rule at once and out-of-sample, and become the target a fixture "
              "has to reproduce. Which candidate is drawn as 1 and which as 2 is shuffled "
              "per tile, so a label cannot anchor to my prediction; the join back to which "
              "rule chose which cell lives in manifest.json and happens after the fact.",
          evidence=[{"type": "kv",
                     "rows": [["contested fragments", len(cand), "bulk assigned differently"],
                              ["tiles built", len(manifest),
                               "largest first, then stratified by area"],
                              ["level used", f"{level:.4g}",
                               "the sweep's most-contested setting (76% of leftover area on "
                               "bridges)"],
                              ["tile size", f"{box_um:.0f} um",
                               f"{box} px at {px:.3f} um/px, fixed for every tile"]]}],
          status="info")
    print(f"\nlabeller -> {os.path.join(out_dir, 'index.html')}")
    print(D.render())


def run_candidates(tif: str, deconv: bool = True, level: float = 0.0,
                   reach_um: float = 25.0) -> None:
    """Compare ways of SELECTING candidate cell extensions, before any rule assigns them.

    Only `brightest` and `fragment` are carried through, per the operator's decision on e004.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    stem = os.path.splitext(os.path.basename(tif))[0]
    cache = os.path.join(CACHE, f"tif_{stem}_{'dc' if deconv else 'raw'}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"no cache at {cache} - run --tif first")
    z = np.load(cache)
    img, labels = z["img"].astype(np.float64), z["lab"].astype(np.int64)
    px = float(z["pixel_size_um"])
    otsu = float(threshold_otsu(img))
    if not level:
        level = otsu
    n_cells = len(np.unique(labels)) - 1
    print(f"{stem}: {n_cells} cells, {px:.4f} um/px, level {level:.4g} (otsu {otsu:.4g})")

    base_area = np.bincount(labels.ravel())
    rows, masks, outs = [], {}, {}
    for kind in CANDIDATES:
        m = _candidates(kind, img, labels, level=level, px=px, reach_um=reach_um)
        masks[kind] = m
        frags, nf = ndi.label(m)
        cen = _fragment_census(labels, m)
        tot = max(int(m.sum()), 1)
        orphan_pct = 100.0 * cen["area"][0] / tot
        #  width: twice the median distance-to-edge inside the candidate, in um. A thin
        #  process reads ~2-4 um; a blob of haze reads much wider.
        wid = ndi.distance_transform_edt(m, sampling=(px, px))
        width_um = float(2 * np.median(wid[m])) if m.any() else 0.0
        dlab = ndi.distance_transform_edt(labels == 0, sampling=(px, px))
        reach95 = float(np.percentile(dlab[m], 95)) if m.any() else 0.0

        got = {r: _assign(r, img, labels, m, (1.0, 1.0)) for r in ("brightest", "fragment")}
        outs[kind] = got
        worst = {}
        for r, g in got.items():
            na = np.bincount(g.ravel(), minlength=len(base_area))
            grew = base_area[1:] > 0
            pct = np.zeros(int(grew.size))
            pct[grew] = 100.0 * (na[1:][grew] - base_area[1:][grew]) / base_area[1:][grew]
            worst[r] = float(pct.max()) if pct.size else 0.0
        both = (got["brightest"] > 0) | (got["fragment"] > 0)
        both &= m
        dis = (100.0 * int((got["brightest"][both] != got["fragment"][both]).sum())
               / max(int(both.sum()), 1))

        rows.append([kind, int(m.sum()), f"{100 * m.mean():.2f}%", nf,
                     f"{orphan_pct:.0f}%", f"{width_um:.1f}", f"{reach95:.0f}",
                     f"+{worst['brightest']:.0f}%", f"+{worst['fragment']:.0f}%",
                     f"{dis:.0f}%"])
        print(f"  {kind:<10} {int(m.sum()):>8} px ({100 * m.mean():>5.2f}%), {nf:>5} frags, "
              f"orphan {orphan_pct:>3.0f}%, width {width_um:>4.1f} um, reach95 "
              f"{reach95:>3.0f} um, worst b+{worst['brightest']:.0f}%/f+{worst['fragment']:.0f}%"
              f", b-vs-f {dis:.0f}%")

    # ── figure ──
    rng = np.random.default_rng(0)
    lut = ListedColormap(np.vstack([[0.05, 0.06, 0.09],
                                    rng.uniform(0.35, 1.0,
                                                size=(max(int(labels.max()), 1) + 1, 3))]))
    vmax = max(int(labels.max()), 1)
    dens = ndi.uniform_filter(masks["attached"].astype(float), 96)
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    s = int(round(180 / px))
    y0, y1 = max(cy - s, 0), min(cy + s, img.shape[0])
    x0, x1 = max(cx - s, 0), min(cx + s, img.shape[1])
    crop = (slice(y0, y1), slice(x0, x1))

    f, axes = plt.subplots(len(CANDIDATES), 3, figsize=(12.5, 4.1 * len(CANDIDATES)),
                           dpi=110)
    for i, kind in enumerate(CANDIDATES):
        A = axes[i]
        base = np.clip((img[crop] - np.percentile(img, 1))
                       / max(np.percentile(img, 99.5) - np.percentile(img, 1), 1e-9), 0, 1)
        rgb = np.repeat(base[:, :, None], 3, axis=2)
        rgb[masks[kind][crop]] = (1.0, 0.25, 0.8)
        A[0].imshow(rgb, interpolation="nearest")
        A[0].set_ylabel(kind, fontsize=11)
        A[0].set_title("candidate (pink) over signal" if i == 0 else "", fontsize=9)
        A[1].imshow(outs[kind]["brightest"][crop], cmap=lut, vmin=0, vmax=vmax,
                    interpolation="nearest")
        A[1].set_title("brightest" if i == 0 else "", fontsize=9)
        A[2].imshow(outs[kind]["fragment"][crop], cmap=lut, vmin=0, vmax=vmax,
                    interpolation="nearest")
        A[2].set_title("fragment" if i == 0 else "", fontsize=9)
        for ax in A:
            ax.set_xticks([]); ax.set_yticks([])
    f.suptitle(f"How to SELECT a candidate extension - {stem} "
               f"({'deconvolved' if deconv else 'as saved'}), "
               f"{2 * s * px:.0f} um crop, level {level:.4g}", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.975))
    p = os.path.join(LOG_ROOT, "img", f"candidates_{stem}.png")
    f.savefig(p); plt.close(f)

    D.log(kind="found",
          title="Requiring a candidate to TOUCH a cell beats raising the intensity cut",
          body=f"Four ways of choosing which leftover signal is a plausible cell extension, "
               f"on the deconvolved frame ({n_cells} cells, {px:.3f} um/px). `attached` "
               f"takes a much more permissive intensity cut ({level * 0.45:.4g} instead of "
               f"{level:.4g}) and then keeps only the pieces physically continuous with a "
               f"cell.",
          why="The operator's instruction was to improve candidate selection before picking "
              "an assignment rule, and the level sweep says the same thing: the cut moves "
              "the result far more than the rule does. Attachment is a better filter than "
              "brightness because it encodes what an extension actually IS - continuous "
              "with the cell - so it can reach further down in intensity along a real "
              "process while discarding bright debris entirely. `orphan` is the share of "
              "selected area sitting on fragments that touch no cell at all: 11-14% under "
              "the global cut, 0% by construction for the rest.",
          evidence=[{"type": "table",
                     "cols": ["selector", "px", "of frame", "fragments", "orphan area",
                              "median width um", "95th pct reach um",
                              "worst cell brightest", "worst cell fragment",
                              "brightest vs fragment"],
                     "rows": rows,
                     "caption": "`median width` separates thin processes from blobs of "
                                "haze; `95th pct reach` is how far from its cell the "
                                "selected signal sits. Both describe WHAT WAS SELECTED - "
                                "neither says the selection is correct."},
                    {"type": "note",
                     "text": "**What these numbers cannot see.** Every column here is a "
                             "property of the candidate mask, not of the truth. None of "
                             "them can tell a real process from a bright artefact that "
                             "happens to touch a cell, and none can say whether a thread "
                             "was recovered along its full length or cut short. That needs "
                             "a drawn boundary to score against - the `x um of boundary "
                             "error` instrument, which does not exist yet on this project."}],
          images=[{"src": f"img/candidates_{stem}.png",
                   "caption": "One row per selector, same crop. Column 1 is what each would "
                              "hand to the assignment rule. Compare `global` against "
                              "`attached`: the pink specks floating in open background under "
                              "`global` are what attachment removes, and the extra pink "
                              "hugging the cell edges is what the lower cut recovers."}],
          status="info")
    print(D.render())


def _compare_and_log(img: np.ndarray, labels: np.ndarray, px: float, *, tag: str,
                     source_desc: str, level_mult: float = 1.0) -> None:
    """Run all four rules on one frame, render two figures, and log the comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.filters import threshold_otsu

    import devlog as D
    D.configure(LOG_ROOT, title="Segmentation reclaim - which assignment rule?")

    # The leftover cut. Otsu on the preprocessed frame is a defensible starting point and,
    # unlike a number picked by hand, it is reproducible -- but it IS the node's future
    # socket, so `level_mult` exists to show the comparison's sensitivity to it.
    otsu = float(threshold_otsu(img))
    level = otsu * level_mult
    left = _leftover(img, labels, level)
    sampling = (1.0, 1.0)
    print(f"otsu {otsu:.1f} x{level_mult} -> level {level:.1f};  leftover "
          f"{int(left.sum())} px ({100 * left.mean():.2f}% of frame)")

    census = _fragment_census(labels, left)
    print(f"leftover fragments: {census['n']}  "
          f"touching 0 labels {census['touch'][0]} ({census['area'][0]} px), "
          f"1 label {census['touch'][1]} ({census['area'][1]} px), "
          f"2+ labels {census['touch'][2]} ({census['area'][2]} px)")

    results, stats = {}, {}
    base_area = np.bincount(labels.ravel())
    for r in RULES:
        got = _assign(r, img, labels, left, sampling)
        results[r] = got
        new_area = np.bincount(got.ravel(), minlength=len(base_area))
        ids = np.arange(1, len(base_area))
        grew = base_area[1:] > 0
        pct = np.zeros(len(ids))
        pct[grew] = 100.0 * (new_area[1:][grew] - base_area[1:][grew]) / base_area[1:][grew]
        stats[r] = dict(
            claimed=int((got > 0).sum() - (labels > 0).sum()),
            unclaimed=int(left.sum() - ((got > 0).sum() - (labels > 0).sum())),
            n_grew=int((pct > 0).sum()),
            median_pct=float(np.median(pct[grew])) if grew.any() else 0.0,
            max_pct=float(pct.max()) if len(pct) else 0.0)
        print(f"  {r:<11} claimed {stats[r]['claimed']:>7} px, "
              f"{stats[r]['n_grew']:>4} cells grew, median +{stats[r]['median_pct']:.1f}%, "
              f"max +{stats[r]['max_pct']:.0f}%")

    # â”€â”€ figures â”€â”€
    rng = np.random.default_rng(0)                  # colour only; no data depends on it
    lut = ListedColormap(np.vstack([[0.05, 0.06, 0.09],
                                    rng.uniform(0.35, 1.0, size=(max(int(labels.max()), 1)
                                                                 + 1, 3))]))
    vmax = max(int(labels.max()), 1)

    def show(ax, arr, title, gray=False):
        if gray:
            ax.imshow(arr, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(arr, cmap=lut, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    f, axes = plt.subplots(2, 4, figsize=(17, 8.2), dpi=112)
    A = axes.ravel()
    show(A[0], img, "the image CellSAM was given", gray=True)
    show(A[1], labels, f"CellSAM labels - {len(np.unique(labels)) - 1} cells")
    show(A[2], left.astype(int), f"leftover above {level:.4g} - {int(left.sum())} px",
         gray=True)
    #  labels-with-leftover-overlaid, so "what was missed" is legible against "what was
    #  found" in a single panel rather than by eye across two.
    show(A[3], np.where(left, vmax, labels),
         "leftover (white) over the labels it must be shared out among")
    for ax, r in zip(A[4:], RULES):
        show(ax, results[r], f"{r} - +{stats[r]['claimed']} px, "
                             f"median cell +{stats[r]['median_pct']:.1f}%")
    f.suptitle(f"{tag} - four reclaim rules on your own CellSAM labels", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.96))
    f.savefig(os.path.join(LOG_ROOT, "img", f"{tag}_full.png")); plt.close(f)

    # zoom on the busiest leftover neighbourhood
    from scipy import ndimage as ndi
    dens = ndi.uniform_filter(left.astype(float), 96)
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    s = 150
    y0, y1 = max(cy - s, 0), min(cy + s, img.shape[0])
    x0, x1 = max(cx - s, 0), min(cx + s, img.shape[1])
    f, axes = plt.subplots(2, 3, figsize=(14.5, 9.4), dpi=118)
    A = axes.ravel()
    show(A[0], img[y0:y1, x0:x1], "signal", gray=True)
    show(A[1], labels[y0:y1, x0:x1], "CellSAM labels")
    for ax, r in zip(A[2:], RULES):
        show(ax, results[r][y0:y1, x0:x1], r)
    f.suptitle(f"Zoom on the densest leftover region - {tag}, "
               f"{2 * s} px = {2 * s * px:.0f} um across", fontsize=11)
    f.tight_layout(rect=(0, 0, 1, 0.96))
    f.savefig(os.path.join(LOG_ROOT, "img", f"{tag}_zoom.png")); plt.close(f)

    rows = [[r, stats[r]["claimed"], stats[r]["unclaimed"], stats[r]["n_grew"],
             f"+{stats[r]['median_pct']:.1f}%", f"+{stats[r]['max_pct']:.0f}%"]
            for r in RULES]

    D.ask(title=f"The four rules on a real frame ({tag}) - which one do you want?",
          body=f"Source: {source_desc} The leftover is signal above Otsu ({otsu:.4g}) that "
               f"CellSAM did not claim. {len(np.unique(labels)) - 1} cells, "
               f"{int(left.sum())} leftover px ({100 * left.mean():.2f}% of the frame).",
          why="On the synthetic fixture three rules were clean and only the bridge column "
              "separated them. What decides it here is the fragment census below: it counts "
              "how often a leftover piece actually touches TWO cells on your data. If that "
              "number is small, `fragment` is the simpler node and loses nothing; if it is "
              "large, only the splitting rules keep per-cell areas honest.",
          evidence=[{"type": "kv",
                     "rows": [["leftover fragments", census["n"], "connected pieces"],
                              ["touching no label", census["touch"][0],
                               f"{census['area'][0]} px - debris no rule should claim"],
                              ["touching one label", census["touch"][1],
                               f"{census['area'][1]} px - unambiguous, all rules agree"],
                              ["touching 2+ labels", census["touch"][2],
                               f"{census['area'][2]} px - BRIDGES; this is the number that "
                               f"decides fragment vs splitting"],
                              ["otsu level", f"{otsu:.1f}",
                               "the leftover cut, reproducible rather than hand-picked - it "
                               "becomes a socket on the node"]]},
                    {"type": "table",
                     "cols": ["rule", "px claimed", "px left behind", "cells that grew",
                              "median area change", "worst cell"],
                     "rows": rows,
                     "caption": "`px left behind` is leftover no rule could reach - for the "
                                "connectivity-aware rules that is mostly the debris. Watch "
                                "the last column: one cell gaining a large fraction usually "
                                "means it swallowed a neighbour's process."}],
          images=[{"src": f"img/{tag}_full.png",
                   "caption": "Whole frame. Compare the four bottom panels against the "
                              "leftover panel: which of them put the thread signal onto the "
                              "cell you would have drawn it onto?"},
                  {"src": f"img/{tag}_zoom.png",
                   "caption": "The densest leftover neighbourhood, where the rules disagree "
                              "most. This is the crop to judge on - look for a thread handed "
                              "to a cell it does not belong to, and for two cells joined "
                              "into one."}],
          asks="Looking at the zoom: which rule assigns the threads the way you would by "
               "hand? If none of them do, say what they get wrong and I will add the case "
               "to the fixture rather than tune to this one frame.")
    print(D.render())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="mechanism demo on a fixture with known truth")
    ap.add_argument("--retract", action="store_true",
                    help="also log the correction retracting e001's geodesic row")
    ap.add_argument("--segment", action="store_true",
                    help="run the real chain + CellSAM once and cache the result")
    ap.add_argument("--real", action="store_true",
                    help="apply the four rules to the cached real ND2 frames")
    ap.add_argument("--tif", default="",
                    help="segment ONE exported TIF frame and run all four rules on it")
    ap.add_argument("--no-deconv", action="store_true",
                    help="skip ZS-DeconvNet on the TIF (segment it exactly as saved)")
    ap.add_argument("--sweep", default="",
                    help="sweep the leftover level on a cached TIF; pass a TIF path")
    ap.add_argument("--sweep-levels", default="",
                    help="explicit comma-separated levels instead of the default span")
    ap.add_argument("--label-sheet", default="",
                    help="build the blind labelling sheet from a cached TIF; pass its path")
    ap.add_argument("--label-level", type=float, default=0.0,
                    help="leftover level for the sheet (default: the sweep's 40% point)")
    ap.add_argument("--max-tiles", type=int, default=260)
    ap.add_argument("--candidates", default="",
                    help="compare candidate-extension selectors on a cached TIF")
    ap.add_argument("--reach-um", type=float, default=25.0)
    ap.add_argument("--level", type=float, default=30.0,
                    help="leftover threshold for the synthetic stage")
    ap.add_argument("--level-mult", type=float, default=1.0,
                    help="scale Otsu for the real stage, to show sensitivity")
    ap.add_argument("--position", type=int, default=0, help="stage position (M) to use")
    ap.add_argument("--times", type=int, default=3,
                    help="timepoints to load (temporal_gain fits over T)")
    ap.add_argument("--frame", type=int, default=0, help="which cached timepoint to render")
    a = ap.parse_args()
    if a.synthetic:
        run_synthetic(a.level, retract=a.retract)
    if a.segment:
        run_segment(times=a.times, position=a.position)
    if a.tif:
        run_tif(a.tif, deconv=not a.no_deconv, level_mult=a.level_mult)
    if a.sweep:
        run_sweep(a.sweep, deconv=not a.no_deconv, levels=a.sweep_levels)
    if a.candidates:
        run_candidates(a.candidates, deconv=not a.no_deconv, level=a.label_level,
                       reach_um=a.reach_um)
    if a.label_sheet:
        run_label_sheet(a.label_sheet, deconv=not a.no_deconv, level=a.label_level,
                        max_tiles=a.max_tiles)
    if a.real:
        run_real(position=a.position, times=a.times, frame=a.frame,
                 level_mult=a.level_mult)
    if not (a.synthetic or a.segment or a.real or a.tif or a.sweep or a.label_sheet
            or a.candidates):
        ap.print_help()
