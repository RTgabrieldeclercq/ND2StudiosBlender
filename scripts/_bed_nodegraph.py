"""Redo the packed-bed analysis through the nodegraph ENGINE, using only shipped nodes.

Every quantitative result in this project so far came from hand-rolled skimage
(`scripts/_bed_analyze.py` and a pile of scratch scripts) that never touched the engine.
This script expresses the same analysis as real node graphs and scores the result against
the 585 hand labels, so "the catalog can already do this" is a measurement rather than a
claim.

Design rules, each of which cost something to learn:

* **Both `seeds` and `meta_seeds`.** Omit `meta_seeds` and `propagate_meta` hands every
  node an empty envelope, `ctx.calib()` returns None, and every µm-aware socket silently
  falls back to pixel units. Nothing warns.
* **Pass `modes={"dim": "2D"}` explicitly.** The 2D/3D lever's metadata-adaptive default
  is never evaluated headlessly (`resolve_dim_default` exists but the engine does not call
  it), so a z-stack still resolves to "2D" unless told. Here 2D is also correct: the 40 µm
  z step against ~50-70 µm objects makes the planes near-independent.
* **The reference blob partition is rebuilt with skimage, NOT with the graph.** The labels
  key on `oid`, the id of a connected component of the ORIGINAL Otsu foreground. Re-deriving
  the blobs from a node would renumber them and silently unjoin every label. The graph
  supplies the *candidate* subdivision; the reference partition it is scored inside stays
  frozen. `parity` measures how close the node foreground is to that reference, which is a
  result in its own right and is not allowed to move the scoring.
* **`dt_s` from this file is 4x wrong and must be overridden.** See `true_dt_s`.

Modes:
    parity   node foreground vs the prototype's skimage foreground, per condition
    bench    score every instance route against the labels, leave-one-condition-out
    cells    GFP objects; F-channel and NileBlue-channel intensity via `measure.raw`
    phases   per-phase solid fraction over time, filtered and unfiltered
    track    IoU label linking + per-object velocity, with a mutual-match gate
    graphs   write the *.nd2graph.json files (which the screenshot pass reuses verbatim)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

for _s in (sys.stdout, sys.stderr):          # the [ok] lines carry µ/σ; cp1252 would raise
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nodegraph.dataset import AxisSizes, Dataset               # noqa: E402
from nodegraph.domains import Domain as D                      # noqa: E402
from nodegraph.engine import Engine                            # noqa: E402
from nodegraph.graph import Graph, NodeInstance                # noqa: E402
from nodegraph.metadata import MetaEnvelope                    # noqa: E402
from nodegraph.nodes import COMPUTES                           # noqa: E402
from nodegraph.provider import ArrayProvider                   # noqa: E402
from nodegraph.registry import OutDataset, define_node          # noqa: E402
from nodegraph.serialize import to_json                        # noqa: E402

ND2 = os.environ.get("BED_ND2") or os.path.join(
    _ROOT, "ChannelGFP,R-B,Nile Blue,TD_Seq0001.nd2")
LABEL_ROOT = os.path.join(_ROOT, "CodeLog", "live", "label")
LABEL_CSV = os.environ.get("BED_LABELS") or os.path.join(
    os.path.expanduser("~"), "Downloads", "granule_labels.csv")
CACHE = os.environ.get("BED_CACHE") or os.path.join(_ROOT, "results", "nodegraph")

#: channel indices in this acquisition
C_GFP, C_F, C_I, C_TD = 0, 1, 2, 3
#: the three label conditions: (m index, name, description)
LABEL_FIELDS = [(0, "M01", "pure SMALL functional"),
                (7, "M08", "pure MEDIUM functional"),
                (14, "M15", "pure LARGE functional")]
#: the six conditions the prototype measured — 3 pure-F controls + 3 mixed
RESULT_FIELDS = [(0, "M01", 1.00, "S", None), (7, "M08", 1.00, "M", None),
                 (14, "M15", 1.00, "L", None), (3, "M04", 0.50, "S", "M"),
                 (10, "M11", 0.50, "M", "M"), (18, "M19", 0.75, "L", "L")]
Z_PLANE = 1

SEED_NODE_ID = "S"
_SEED_OP = "io.bedseed"
define_node(_SEED_OP, "Bed source", outputs=[OutDataset()])


# ══ source ════════════════════════════════════════════════════════════════════

def true_dt_s(path: str = ND2) -> float:
    """Seconds between successive visits to one position, from the ND2 **event table**.

    `nodelab_v2.ingest.read_calibration` reports 11351.1 s for this file and that is
    **4.0009x too large**. Its `_dt_from_timestamps` takes the median of consecutive
    differences of `frame_timestamps_s`, and on this file the reader recovers only 4 of the
    15 stamps — at t = 0, 4, 8, 12, a uniform stride of 4. The median defends against ONE
    dropped frame; against a uniformly strided subset every gap is wrong by the same
    integer factor, so the median is exactly as wrong as the mean and nothing looks
    anomalous. `analysis.object_metrics` divides by this, so every velocity would come out
    4x too slow.

    The event table carries `T Index` and `Time [s]` for all 2205 events (15 T x 21 P x 7
    Z), so the interval is measured directly: median 2837.2 s over 14 intervals with 4.3 s
    of spread, decomposing as a 961.1 s plate pass plus 1876.0 s idle.
    """
    from nodelab_v2.nd2_compat import import_nd2
    nd2 = import_nd2()
    with nd2.ND2File(path) as f:
        ev = list(f.events())
    t = np.asarray([e["Time [s]"] for e in ev], dtype=float)
    ti = np.asarray([e["T Index"] for e in ev], dtype=int)
    first = np.array([t[ti == k].min() for k in range(int(ti.max()) + 1)])
    return float(np.median(np.diff(first)))


_CAL: Optional[Dict[str, Any]] = None
_DASK = None


def calibration() -> Dict[str, Any]:
    """This file's calibration, with `dt_s` replaced by the event-table measurement."""
    global _CAL
    if _CAL is None:
        from nodelab_v2.ingest import read_calibration
        cal = dict(read_calibration(ND2))
        cal["dt_s"] = true_dt_s()
        _CAL = cal
    return dict(_CAL)


def _dask():
    global _DASK
    if _DASK is None:
        from nodelab_v2.ingest import lazy_nd2
        _DASK = lazy_nd2(ND2)                 # (M,T,Z,C,Y,X), no pixels read
    return _DASK


def raw_plane(m: int, t: int = 0, z: int = Z_PLANE, c: int = C_F) -> np.ndarray:
    """One (Y,X) plane as float32 — the same array the prototype scripts read."""
    return np.asarray(_dask()[m, t, z, c]).astype(np.float32)


def source(m: int, ts: Sequence[int] = (0,), z: int = Z_PLANE,
           channels: Sequence[int] = (C_GFP, C_F, C_I)) -> Tuple[Dataset, MetaEnvelope]:
    """(Dataset, MetaEnvelope) over one position, the given timepoints and one z plane.

    Deliberately one position per run: the LABEL tables carry `m`, but a run that stacked
    several conditions onto the m axis would make `m` an index into an ad-hoc list rather
    than the acquisition's own position, and every downstream join would depend on that
    list staying in the same order. One condition per Dataset keeps the mapping trivial.

    `channels` drops the transmitted-light channel by default — it carries no granule
    outlines on this acquisition, so reading it is pure cost.
    """
    dk = _dask()
    vol = np.stack([np.stack([np.asarray(dk[m, t, z, c]) for c in channels], axis=0)
                    for t in ts], axis=0)                       # (T, C, Y, X)
    img6 = vol[None, :, None, :, :, :]                          # (1, T, 1, C, Y, X)
    ax = AxisSizes(m=1, t=len(ts), z=1, c=len(channels),
                   y=img6.shape[-2], x=img6.shape[-1])
    cal = calibration()
    # z_step_um is real on the file but this Dataset is a single plane; keep it out so a
    # 2D pull is not memo-fenced on a number it cannot use.
    cal.pop("z_step_um", None)
    cal.pop("origin_um", None)                                   # per-M, and we sliced M
    ds = Dataset(axes=ax, metadata=dict(cal)).with_image(ArrayProvider(img6))
    return ds, MetaEnvelope(axes=ax, metadata=dict(cal))


def ch_index(channels: Sequence[int], want: int) -> int:
    """Index of acquisition channel `want` inside a `source(channels=...)` selection."""
    return list(channels).index(want)


# ══ graph plumbing ════════════════════════════════════════════════════════════

def build(nodes: Sequence[Tuple[str, str, dict]],
          edges: Sequence[Tuple[str, ...]]) -> Graph:
    """A Graph from `(id, op_key, {"params":…, "modes":…})` rows and `(src, dst[, socket])`."""
    g = Graph()
    g.add(NodeInstance(SEED_NODE_ID, _SEED_OP))
    for nid, op, kw in nodes:
        g.add(NodeInstance(nid, op, **kw))
    for e in edges:
        g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
    return g


def run(g: Graph, sink: str, ds: Dataset, env: MetaEnvelope,
        *, verbose: bool = False) -> Tuple[Dataset, Engine]:
    times: Dict[str, float] = {}

    def obs(event, nid, info):
        if event == "done":
            times[nid] = float(info.get("seconds", 0.0))

    eng = Engine(g, computes=COMPUTES, seeds={SEED_NODE_ID: ds},
                 meta_seeds={SEED_NODE_ID: env}, observer=obs)
    out = eng.pull(sink)
    if verbose and times:
        print("      " + "  ".join(f"{k} {v:.2f}s" for k, v in times.items()))
    return out, eng


def table(out: Dataset, domain: D, layer: str) -> Dict[str, np.ndarray]:
    """A structure table as `{column: ndarray}` (they arrive exploded into AttributeLayers)."""
    return {name: np.asarray(attr.values)
            for (dom, lay, name), attr in out.attributes.items()
            if dom is domain and lay == layer}


def raster(out: Dataset, layer: str) -> np.ndarray:
    a = out.get(D.VOXEL, layer)
    if a is None:
        have = sorted({k[2] for k in out.attributes if k[0] is D.VOXEL})
        raise KeyError(f"no Voxel layer {layer!r} (have {have})")
    return np.asarray(a.values)


def gates(out: Dataset, layer: str, *, domain: D = D.LABEL,
          expect_zkind: str = "plane_index", check_area: bool = True) -> Dict[str, Any]:
    """Cheap invariants that catch a silently-wrong graph. Raises on violation.

    The one that earns its keep is raster-vs-table id agreement: a table row with no
    painted voxels, or a painted id with no row, is the likeliest defect in any chain that
    relabels, and nothing in the engine checks it.
    """
    r = raster(out, layer)
    tb = table(out, domain, layer)
    if "id" not in tb:
        # A graph that correctly finds NOTHING emits no structure table at all rather than
        # an empty one — `Dataset.with_structure` is never called, so the layer is absent
        # instead of zero-length. That is legitimate here (the fixed Nile-Blue cut finds no
        # inert material at a pure-functional control, which is the right answer), but it
        # means a downstream consumer sees a MISSING layer, not an empty one. Accept it only
        # when the raster agrees that there is nothing.
        if int((r != 0).sum()) == 0:
            return {"n": 0, "z_kind": out.structure_zkind(domain, layer),
                    "columns": [], "empty": True}
        raise AssertionError(
            f"{layer}: no id column on {domain} yet the raster has "
            f"{int((r != 0).sum())} non-zero voxels")
    ids_tbl = set(int(v) for v in tb["id"])
    ids_ras = set(int(v) for v in np.unique(r)) - {0}
    if ids_tbl != ids_ras:
        raise AssertionError(
            f"{layer}: raster ids != table ids — only in table "
            f"{sorted(ids_tbl - ids_ras)[:6]}, only in raster {sorted(ids_ras - ids_tbl)[:6]}")
    missing = {"id", "m", "t", "c", "z", "y", "x"} - set(tb)
    if missing:
        raise AssertionError(f"{layer}: table missing coordinate columns {sorted(missing)}")
    zk = out.structure_zkind(domain, layer)
    if zk != expect_zkind:
        raise AssertionError(f"{layer}: z_kind {zk!r} != {expect_zkind!r}")
    if check_area and "area" in tb:
        tot, nz = int(tb["area"].sum()), int((r != 0).sum())
        if tot != nz:
            raise AssertionError(f"{layer}: sum(area)={tot} != count(raster!=0)={nz}")
    return {"n": len(ids_tbl), "z_kind": zk, "columns": sorted(tb)}


# ══ the reference partition and the label metric ══════════════════════════════
# Vendored from scratchpad find_granules.py so the numbers stay comparable. The only
# change is that `newlab` now arrives from a node graph instead of from skimage.

def reference_blobs(m: int) -> Tuple[np.ndarray, list]:
    """The prototype's Otsu foreground, component-labelled — the partition the labels index.

    Reproduced verbatim from `make_labeller.py`: Otsu on the float32 R-B plane, fill holes,
    `ndi.label`. Any deviation here renumbers `oid` and unjoins every label.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    img = raw_plane(m, 0, Z_PLANE, C_F)
    fg = ndi.binary_fill_holes(img > threshold_otsu(img))
    lab, _n = ndi.label(fg)
    return lab, ndi.find_objects(lab)


def load_labels() -> Tuple[Dict[str, list], Dict[str, float]]:
    """`{cond: [(oid, n_granules, area_um2)]}` and `{cond: median single-granule area}`."""
    man = json.load(open(os.path.join(LABEL_ROOT, "manifest.json"), encoding="utf-8"))
    lab_of: Dict[str, str] = {}
    for line in open(LABEL_CSV, encoding="utf-8"):
        s = line.strip()
        if "," in s and not s.upper().startswith(("GRANULE", "LABELS", "UID,")):
            u, v = s.rsplit(",", 1)
            lab_of[u] = v.strip()
    by_cond: Dict[str, list] = {}
    for r in man:
        if lab_of.get(r["uid"]) in ("1", "2", "3"):
            by_cond.setdefault(r["cond"], []).append(
                (int(r["oid"]), int(lab_of[r["uid"]]), float(r["area_um2"])))
    a_single = {c: float(np.median([a for _, n, a in rows if n == 1]))
                for c, rows in by_cond.items()}
    return by_cond, a_single


def count_pieces(newlab: np.ndarray, mask: np.ndarray, a_min_px: float) -> int:
    """How many granule-sized pieces of `newlab` fall inside `mask`?"""
    v = newlab[mask]
    v = v[v > 0]
    if v.size == 0:
        return 0
    cnt = np.bincount(v)
    return int((cnt >= a_min_px).sum())


def score(newlab: np.ndarray, lab_otsu: np.ndarray, rows: list, cond: str,
          objs: list, a_single: Dict[str, float], px: float) -> Dict[str, float]:
    a_min_px = 0.30 * a_single[cond] / (px * px)
    ex = un = ov = lost = 0
    tot_pred = tot_true = 0.0
    for oid, n, _ in rows:
        sl = objs[oid - 1]
        m = (lab_otsu[sl] == oid)
        k = count_pieces(newlab[sl], m, a_min_px)
        tot_pred += k
        tot_true += (3.4 if n == 3 else n)
        if k == 0:
            lost += 1
        kk = min(k, 3)
        if kk == n:
            ex += 1
        elif kk < n:
            un += 1
        else:
            ov += 1
    N = len(rows)
    return dict(exact=ex / N, under=un / N, over=ov / N, lost=lost / N,
                count_err=tot_pred / max(tot_true, 1e-9) - 1.0, n=N)


# ══ the graphs ════════════════════════════════════════════════════════════════
#
# Two orderings below are load-bearing rather than stylistic:
#   * `enhance.gaussian` sits AFTER the threshold, so smoothing cannot move the Otsu cut.
#   * `measure.raw` taps the pre-smoothing channel select, so intensities are read off the
#     unsmoothed image. Enhancement between the two branches is explicitly allowed by
#     `_shared/raw_measure.py`; a geometry op (crop/resample/shift) would be refused.

_D2 = {"dim": "2D"}


def g_blobs(*, ci: int, min_area_um2: float, name: str = "blobs",
            fill_holes: bool = True) -> Tuple[List[tuple], List[tuple], str]:
    """R0 — foreground + size filter + connected components in ONE shipped node.

    `analysis.segment`'s `min_area` is in µm², which is exactly the phantom-phase filter
    the prototype hand-rolled: keep only components at least 0.30x a label-confirmed
    single granule.
    """
    nodes = [("C", "channel.select", {"params": {"channels": [ci]}}),
             ("A", "analysis.segment",
              {"modes": {**_D2, "method": "threshold", "level": "otsu"},
               "params": {"name": name, "fill_holes": fill_holes,
                          "min_area": float(min_area_um2)}})]
    edges = [(SEED_NODE_ID, "C"), ("C", "A")]
    return nodes, edges, "A"


def g_edt_watershed(*, ci: int, min_area_um2: float, min_distance_um: float,
                    name: str = "grains") -> Tuple[List[tuple], List[tuple], str]:
    """R1 — `analysis.segment method=watershed`, flooding its own internal EDT.

    The `mask` socket supplies the foreground and bypasses `level`/`threshold` entirely, so
    the Otsu cut is made once by `analysis.threshold` and reused. Seeds are distance-
    transform peaks spaced by `min_distance`; there is no way to hand this node different
    seeds or a different landscape, which is precisely the gap the bench is measuring.
    """
    nodes = [("C", "channel.select", {"params": {"channels": [ci]}}),
             ("T", "analysis.threshold",
              {"modes": {"method": "otsu", "scope": "plane"}, "params": {"name": "fg"}}),
             ("A", "analysis.segment",
              {"modes": {**_D2, "method": "watershed", "level": "otsu"},
               "params": {"name": name, "mask": "fg", "fill_holes": True,
                          "min_area": float(min_area_um2),
                          "min_distance": float(min_distance_um)}})]
    edges = [(SEED_NODE_ID, "C"), ("C", "T"), ("T", "A")]
    return nodes, edges, "A"


def g_seeded_voronoi(*, ci: int, min_area_um2: float, sigma_um: float,
                     seeder: str = "spots", r_lo: float = 0.0, r_hi: float = 0.0,
                     spot_threshold: float = 0.02, min_distance_um: float = 40.0,
                     bound: str = "per_region", name: str = "grains",
                     ) -> Tuple[List[tuple], List[tuple], str]:
    """R2/R3 — intensity-maximum seeds, then a nearest-seed partition inside each blob.

    `analysis.voronoi bound=per_region` treats every connected component of the `region`
    label raster as its own ARENA: a voxel competes only among the seeds sharing its arena
    value, so a seed cannot claim territory in a different blob. That makes the node a
    splitter — it subdivides each blob by its own seeds — which is the closest shipped
    analogue of flooding a landscape. `bound=mask` collapses every component into one
    arena and is benched only to show that the arena choice is what does the work.
    """
    nodes = [("C", "channel.select", {"params": {"channels": [ci]}}),
             ("A", "analysis.segment",
              {"modes": {**_D2, "method": "threshold", "level": "otsu"},
               "params": {"name": "blobs", "fill_holes": True,
                          "min_area": float(min_area_um2)}}),
             ("G", "enhance.gaussian", {"modes": _D2, "params": {"sigma": float(sigma_um)}})]
    if seeder in ("spots", "spots_dog"):
        nodes.append(("D", "detect.spots",
                      {"modes": {**_D2, "polarity": "bright",
                                 "method": ("dog" if seeder == "spots_dog" else "log")},
                       "params": {"min_radius": float(r_lo), "max_radius": float(r_hi),
                                  "threshold": float(spot_threshold), "name": "seeds"}}))
    elif seeder == "particles":
        nodes.append(("D", "detect.particles",
                      {"modes": {**_D2, "mode": "log"},
                       "params": {"min_distance": float(min_distance_um),
                                  "threshold": float(spot_threshold),
                                  "min_size": 1, "subpixel": True, "name": "seeds"}}))
    else:
        raise ValueError(f"unknown seeder {seeder!r} (spots|particles)")
    nodes.append(("V", "analysis.voronoi",
                  {"modes": {**_D2, "bound": bound, "mesh": "skip"},
                   "params": {"points": "seeds", "region": "blobs", "name": name,
                              "max_distance_um": 0.0}}))
    edges = [(SEED_NODE_ID, "C"), ("C", "A"), ("A", "G"), ("G", "D"), ("D", "V")]
    return nodes, edges, "V"


def with_measure(nodes: List[tuple], edges: List[tuple], sink: str, *,
                 labels: str, raw_from: str = "C",
                 stats: str = "mean,max,median,count",
                 shape: str = "solidity,eccentricity,perimeter,axis_major,axis_minor",
                 ) -> Tuple[List[tuple], List[tuple], str]:
    """Append `analysis.measure`, tapping `raw_from` for the unsmoothed intensities."""
    nodes = list(nodes) + [("M", "analysis.measure",
                            {"params": {"labels": labels, "stats": stats, "shape": shape}})]
    edges = list(edges) + [(sink, "M"), (raw_from, "M", "raw")]
    return nodes, edges, "M"


# ══ modes ═════════════════════════════════════════════════════════════════════

def _hdr(txt: str) -> None:
    print("\n" + "=" * 78 + f"\n{txt}\n" + "=" * 78)


def mode_parity(args) -> int:
    """Does a shipped node reproduce the prototype's foreground?

    This is not the label score — it is the prerequisite. If `analysis.segment
    method=threshold level=otsu fill_holes=True` and the hand-rolled Otsu disagree on which
    voxels are foreground, then every downstream comparison is measuring two different
    front ends rather than two different splitters.
    """
    _hdr("PARITY — analysis.segment(threshold/otsu) vs the prototype's skimage foreground")
    cal = calibration()
    px = float(cal["pixel_size_um"])
    _by, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    print(f"pixel_size_um {px!r}   dt_s {cal['dt_s']:.1f} s "
          f"(file reports 11351.1 s = {11351.1 / cal['dt_s']:.4f}x)")
    print(f"\n{'cond':5s} {'ref n':>6s} {'node n':>7s} {'ref frac':>9s} {'node frac':>10s} "
          f"{'IoU':>7s} {'min_area µm²':>13s}")
    rows = []
    for m, cond, _desc in LABEL_FIELDS:
        lab_ref, _objs = reference_blobs(m)
        ref_fg = lab_ref > 0
        min_area = 0.30 * a_single[cond]
        ds, env = source(m, (0,), Z_PLANE, chans)
        ci = ch_index(chans, C_F)
        out, _eng = run(build(*g_blobs(ci=ci, min_area_um2=min_area)[:2]),
                        g_blobs(ci=ci, min_area_um2=min_area)[2], ds, env)
        info = gates(out, "blobs")
        r = raster(out, "blobs")[0, 0, 0, 0]
        node_fg = r > 0
        inter = int((ref_fg & node_fg).sum())
        union = int((ref_fg | node_fg).sum())
        iou = inter / max(union, 1)
        # …and again with the size filter OFF. If the node's Otsu cut is the same cut the
        # prototype made, this second foreground is not merely similar but IDENTICAL, and
        # the whole IoU deficit above is attributable to `min_area` rather than to any
        # disagreement about the threshold. Worth one extra pull to be able to say which.
        n0, e0, s0 = g_blobs(ci=ci, min_area_um2=0.0, name="raw")
        out0, _e0 = run(build(n0, e0), s0, ds, env)
        bare = raster(out0, "raw")[0, 0, 0, 0] > 0
        identical = bool(np.array_equal(bare, ref_fg))
        subset = bool((node_fg & ~ref_fg).sum() == 0)
        print(f"{cond:5s} {int(lab_ref.max()):6d} {info['n']:7d} "
              f"{ref_fg.mean():9.4f} {node_fg.mean():10.4f} {iou:7.4f} {min_area:13.0f}"
              f"   min_area=0 identical={identical}  filtered⊂ref={subset}")
        rows.append(dict(cond=cond, m=m, ref_n=int(lab_ref.max()), node_n=info["n"],
                         ref_frac=float(ref_fg.mean()), node_frac=float(node_fg.mean()),
                         iou=float(iou), min_area_um2=float(min_area),
                         bare_identical=identical, filtered_is_subset=subset,
                         n_raw_components=int(np.unique(raster(out0, "raw")).size - 1)))
    print("\nnode n < ref n is expected and intended: min_area drops sub-granule debris that "
          "\nthe reference labelling keeps. `min_area=0 identical` is the real claim — the "
          "\nshipped node makes the SAME Otsu cut, so the IoU deficit is entirely the filter.")
    _dump("parity.json", rows)
    return 0


def _routes(cond: str, a_single: Dict[str, float], ci: int) -> Dict[str, tuple]:
    """The candidate instance routes for one condition, each as (nodes, edges, sink)."""
    min_area = 0.30 * a_single[cond]
    r_eq = float(np.sqrt(a_single[cond] / np.pi))       # equivalent-disc radius, µm
    return {
        "R0 components": g_blobs(ci=ci, min_area_um2=min_area, name="grains"),
        "R1 edt-watershed": g_edt_watershed(ci=ci, min_area_um2=min_area,
                                            min_distance_um=r_eq),
        "R2 spots+voronoi": g_seeded_voronoi(ci=ci, min_area_um2=min_area, sigma_um=6.9,
                                             seeder="spots", r_lo=0.7 * r_eq,
                                             r_hi=1.3 * r_eq, spot_threshold=0.02),
        "R2p particles+voronoi": g_seeded_voronoi(ci=ci, min_area_um2=min_area, sigma_um=6.9,
                                                  seeder="particles",
                                                  min_distance_um=1.5 * r_eq,
                                                  spot_threshold=0.0),
        "R3 spots+voronoi(mask)": g_seeded_voronoi(ci=ci, min_area_um2=min_area, sigma_um=6.9,
                                                   seeder="spots", r_lo=0.7 * r_eq,
                                                   r_hi=1.3 * r_eq, spot_threshold=0.02,
                                                   bound="mask"),
    }


def mode_bench(args) -> int:
    _hdr("BENCH — instance routes scored against the 585 hand labels")
    cal = calibration()
    px = float(cal["pixel_size_um"])
    by_cond, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ci = ch_index(chans, C_F)
    for c in sorted(a_single):
        n1 = sum(1 for _, n, _ in by_cond[c] if n == 1)
        print(f"  {c}: {len(by_cond[c])} scorable blobs, {n1} singles, "
              f"A_single {a_single[c]:.0f} µm² "
              f"(r_eq {np.sqrt(a_single[c]/np.pi):.1f} µm)")

    out_rows: List[dict] = []
    ref: Dict[str, tuple] = {}
    for m, cond, _d in LABEL_FIELDS:
        ref[cond] = reference_blobs(m)

    names = list(_routes("M01", a_single, ci))
    print(f"\n{'route':24s} {'cond':5s} {'n':>4s} {'exact':>7s} {'under':>7s} "
          f"{'over':>6s} {'lost':>6s} {'cnt_err':>8s} {'objs':>6s} {'s':>5s}")
    for name in names:
        per = {}
        for m, cond, _d in LABEL_FIELDS:
            nodes, edges, sink = _routes(cond, a_single, ci)[name]
            ds, env = source(m, (0,), Z_PLANE, chans)
            t0 = time.time()
            try:
                o, _e = run(build(nodes, edges), sink, ds, env)
                gates(o, "grains", check_area=("voronoi" not in name.lower()))
                newlab = raster(o, "grains")[0, 0, 0, 0].astype(np.int64)
                n_obj = len(table(o, D.LABEL, "grains").get("id", []))
            except Exception as exc:                      # a route that cannot run is a result
                print(f"{name:24s} {cond:5s}   FAILED: {type(exc).__name__}: {exc}")
                per[cond] = None
                continue
            dt = time.time() - t0
            lab_ref, objs = ref[cond]
            s = score(newlab, lab_ref, by_cond[cond], cond, objs, a_single, px)
            per[cond] = s
            print(f"{name:24s} {cond:5s} {s['n']:4d} {s['exact']:7.1%} {s['under']:7.1%} "
                  f"{s['over']:6.1%} {s['lost']:6.1%} {s['count_err']:+8.1%} "
                  f"{n_obj:6d} {dt:5.1f}")
            out_rows.append(dict(route=name, cond=cond, n_objects=n_obj,
                                 seconds=dt, **s))
        got = [v for v in per.values() if v]
        if got:
            w = np.array([v["n"] for v in got], dtype=float)
            print(f"{name:24s} {'POOL':5s} {int(w.sum()):4d} "
                  f"{np.average([v['exact'] for v in got], weights=w):7.1%} "
                  f"{np.average([v['under'] for v in got], weights=w):7.1%} "
                  f"{np.average([v['over'] for v in got], weights=w):6.1%} "
                  f"{np.average([v['lost'] for v in got], weights=w):6.1%}")
        print()
    _dump("bench.json", out_rows)
    print("Prototype reference: Otsu components 73.3% · EDT h-maxima 81.7% · "
          "intensity h-maxima 88.9% (held out).")
    return 0


def _dump(fname: str, obj: Any) -> str:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, fname)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=float)
    print(f"[wrote] {p}")
    return p


def mode_graphs(args) -> int:
    """Write the *.nd2graph.json files. The screenshot pass loads these verbatim, so the
    pictures in the document cannot drift from the graphs that produced the numbers."""
    _hdr("GRAPHS — serialize the pipelines")
    _by, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ci = ch_index(chans, C_F)
    outdir = os.path.join(CACHE, "graphs")
    os.makedirs(outdir, exist_ok=True)
    made = []
    for name, (nodes, edges, sink) in _routes("M08", a_single, ci).items():
        slug = name.split()[0].lower()
        if slug in ("r2",):
            nodes, edges, sink = with_measure(nodes, edges, sink, labels="grains")
        g = build(nodes, edges)
        p = os.path.join(outdir, f"{slug}.nd2graph.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(to_json(g))
        made.append((slug, sink, len(g.nodes), len(g.edges), p))
        print(f"  {slug:5s} sink={sink:2s} {len(g.nodes)} nodes {len(g.edges)} edges  -> {p}")
    _dump("graphs.json", [dict(slug=s, sink=k, n_nodes=n, n_edges=e, path=p)
                          for s, k, n, e, p in made])
    return 0


def g_smooth_then_watershed(*, ci: int, min_area_um2: float, sigma_um: float,
                            min_distance_um: float, name: str = "grains",
                            morph: str = "", morph_radius_um: float = 0.0,
                            ) -> Tuple[List[tuple], List[tuple], str]:
    """R4 — reshape the IMAGE before the Otsu cut, then let the EDT watershed split it.

    `analysis.segment` cannot be handed a different landscape, but it *can* be handed a
    different foreground, and the foreground is whatever the upstream image produces. A
    greyscale opening is the one shipped operation that attacks a thin bright neck
    specifically — an opening by radius r is exactly `{EDT > r}` — so eroding before the
    threshold breaks the bridges that merge two granules, which is a different lever from
    raising the threshold (that deletes whole dim granules instead).
    """
    nodes: List[tuple] = [("C", "channel.select", {"params": {"channels": [ci]}})]
    prev = "C"
    if sigma_um > 0:
        nodes.append(("G", "enhance.gaussian", {"modes": _D2,
                                                "params": {"sigma": float(sigma_um)}}))
        prev = "G"
    if morph:
        nodes.append(("O", "enhance.morphology",
                      {"modes": {**_D2, "op": morph},
                       "params": {"radius": float(morph_radius_um)}}))
        prev = "O"
    nodes += [("T", "analysis.threshold",
               {"modes": {"method": "otsu", "scope": "plane"}, "params": {"name": "fg"}}),
              ("A", "analysis.segment",
               {"modes": {**_D2, "method": "watershed", "level": "otsu"},
                "params": {"name": name, "mask": "fg", "fill_holes": True,
                           "min_area": float(min_area_um2),
                           "min_distance": float(min_distance_um)}})]
    edges = [(SEED_NODE_ID, "C")]
    chain = [n[0] for n in nodes]
    for a, b in zip(chain, chain[1:]):
        edges.append((a, b))
    return nodes, edges, "A"


def _score_cfg(cfg: dict, conds: Sequence[Tuple[int, str, Any]], ref: dict,
               by_cond: dict, a_single: Dict[str, float], px: float,
               chans: Sequence[int]) -> Dict[str, dict]:
    """Score one configuration on each of `conds`. `cfg["route"]` picks the builder."""
    ci = ch_index(chans, C_F)
    per: Dict[str, dict] = {}
    for m, cond, *_ in conds:
        min_area = 0.30 * a_single[cond]
        r_eq = float(np.sqrt(a_single[cond] / np.pi))
        kind = cfg["route"]
        if kind == "R2":
            n, e, s = g_seeded_voronoi(
                ci=ci, min_area_um2=min_area, sigma_um=cfg["sigma"],
                seeder=cfg.get("seeder", "spots"),
                r_lo=cfg["f_lo"] * r_eq, r_hi=cfg["f_hi"] * r_eq,
                spot_threshold=cfg["thr"], bound=cfg.get("bound", "per_region"))
        elif kind == "R1":
            n, e, s = g_edt_watershed(ci=ci, min_area_um2=min_area,
                                      min_distance_um=cfg["fd"] * r_eq)
        elif kind == "R4":
            n, e, s = g_smooth_then_watershed(
                ci=ci, min_area_um2=min_area, sigma_um=cfg["sigma"],
                min_distance_um=cfg["fd"] * r_eq, morph=cfg.get("morph", ""),
                morph_radius_um=cfg.get("mr", 0.0) * r_eq)
        else:
            raise ValueError(kind)
        ds, env = source(m, (0,), Z_PLANE, chans)
        try:
            o, _eg = run(build(n, e), s, ds, env)
            newlab = raster(o, "grains")[0, 0, 0, 0].astype(np.int64)
        except Exception as exc:
            per[cond] = {"exact": float("nan"), "err": f"{type(exc).__name__}: {exc}"}
            continue
        lab_ref, objs = ref[cond]
        per[cond] = score(newlab, lab_ref, by_cond[cond], cond, objs, a_single, px)
    return per


def _pool(per: Dict[str, dict], key: str = "exact") -> float:
    got = [v for v in per.values() if v and np.isfinite(v.get(key, float("nan")))]
    if not got:
        return float("nan")
    w = np.array([v["n"] for v in got], dtype=float)
    return float(np.average([v[key] for v in got], weights=w))


def mode_sweep(args) -> int:
    """Staged sweeps, each required to BRACKET its optimum.

    A winner sitting on the edge of the range has not found an optimum, it has found the
    edge of the box. That has already happened twice on this project (a solidity sweep over
    [0.80, 1.00) returned 0.800; an h sweep over {0.06, 0.12, 0.20} returned 0.20), so every
    stage below prints whether its best value is interior and widens if it is not.
    """
    _hdr("SWEEP — staged, bracketed, then leave-one-condition-out")
    cal = calibration()
    px = float(cal["pixel_size_um"])
    by_cond, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ref = {c: reference_blobs(m) for m, c, _d in LABEL_FIELDS}
    log: List[dict] = []

    def stage(title: str, cfgs: List[dict], axis: str) -> dict:
        print(f"\n--- {title}  ({len(cfgs)} configs) ---")
        print(f"{axis:>10s} {'pooled':>8s} {'M01':>7s} {'M08':>7s} {'M15':>7s} "
              f"{'under':>7s} {'over':>7s} {'lost':>6s}")
        best, rows = None, []
        for cfg in cfgs:
            per = _score_cfg(cfg, LABEL_FIELDS, ref, by_cond, a_single, px, chans)
            p = _pool(per)
            rows.append((cfg, per, p))
            log.append(dict(stage=title, cfg=dict(cfg), pooled=p,
                            per={k: {kk: vv for kk, vv in v.items() if kk != "err"}
                                 for k, v in per.items()}))
            print(f"{cfg[axis]!s:>10s} {p:8.1%} "
                  + " ".join(f"{per[c].get('exact', float('nan')):7.1%}"
                             for _m, c, _d in LABEL_FIELDS)
                  + f" {_pool(per,'under'):7.1%} {_pool(per,'over'):7.1%} "
                    f"{_pool(per,'lost'):6.1%}")
            if best is None or p > best[2]:
                best = (cfg, per, p)
        vals = [c[axis] for c in cfgs]
        bi = vals.index(best[0][axis])
        interior = 0 < bi < len(vals) - 1
        print(f"  best {axis}={best[0][axis]!r} at {best[2]:.1%}   "
              f"interior={interior}" + ("" if interior else "   <-- ON THE BOUNDARY, widen"))
        return best[0]

    r_thr = stage("R2 · detect.spots threshold (LoG response, per-plane normalized)",
                  [dict(route="R2", sigma=6.9, f_lo=0.7, f_hi=1.3, thr=t)
                   for t in (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)], "thr")
    r_sig = stage("R2 · pre-smoothing sigma (µm)",
                  [dict(r_thr, sigma=s) for s in (0.0, 3.0, 4.5, 6.9, 10.0, 14.0, 20.0)],
                  "sigma")
    r_band = stage("R2 · radius band as a multiple of the label-derived r_eq",
                   [dict(r_sig, f_lo=lo, f_hi=hi) for lo, hi in
                    ((0.3, 0.6), (0.4, 0.9), (0.55, 1.1), (0.7, 1.3), (0.9, 1.6),
                     (1.1, 2.0), (1.4, 2.6))], "f_lo")
    print("\n--- R1 · min_distance as a multiple of r_eq ---")
    r1 = stage("R1 · EDT-watershed min_distance",
               [dict(route="R1", fd=f) for f in (0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8)], "fd")
    r4 = stage("R4 · smooth-then-threshold, EDT watershed (sigma sweep at best min_distance)",
               [dict(route="R4", fd=r1["fd"], sigma=s) for s in
                (0.0, 3.0, 6.9, 10.0, 14.0, 20.0, 28.0)], "sigma")
    r4o = stage("R4 · greyscale OPENING radius as a multiple of r_eq (necks break first)",
                [dict(r4, morph="open", mr=r) for r in
                 (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)], "mr")

    # ── leave-one-condition-out on the winner of each family ──────────────────
    print("\n" + "-" * 78)
    print("LEAVE-ONE-CONDITION-OUT — parameters chosen on two conditions, scored on the third")
    print("-" * 78)
    fams = {"R2 spots+voronoi": ("R2", [dict(r_band, thr=t) for t in
                                        (0.01, 0.02, 0.04, 0.08, 0.16)]),
            "R1 edt-watershed": ("R1", [dict(route="R1", fd=f) for f in
                                        (0.5, 0.7, 0.9, 1.1, 1.4)]),
            "R4 open+watershed": ("R4", [dict(r4o, mr=r) for r in
                                         (0.0, 0.1, 0.2, 0.3, 0.45)])}
    summary = []
    for fam, (_kind, cand) in fams.items():
        print(f"\n{fam}")
        held = []
        for m_h, cond_h, _d in LABEL_FIELDS:
            train = [f for f in LABEL_FIELDS if f[1] != cond_h]
            best, bp = None, -1.0
            for cfg in cand:
                p = _pool(_score_cfg(cfg, train, ref, by_cond, a_single, px, chans))
                if p > bp:
                    best, bp = cfg, p
            per = _score_cfg(best, [(m_h, cond_h, None)], ref, by_cond, a_single, px, chans)
            s = per[cond_h]
            held.append(s)
            pick = {k: v for k, v in best.items() if k != "route"}
            print(f"  hold out {cond_h}: train {bp:.1%} -> held-out {s['exact']:.1%} "
                  f"(under {s['under']:.1%} over {s['over']:.1%} lost {s['lost']:.1%})  "
                  f"picked {pick}")
            log.append(dict(stage="loco", family=fam, held_out=cond_h, train_pooled=bp,
                            cfg=pick, **{k: v for k, v in s.items() if k != "err"}))
        w = np.array([s["n"] for s in held], float)
        mean = float(np.average([s["exact"] for s in held], weights=w))
        print(f"  HELD-OUT MEAN {mean:.1%}   under {np.average([s['under'] for s in held], weights=w):.1%}"
              f"   over {np.average([s['over'] for s in held], weights=w):.1%}"
              f"   lost {np.average([s['lost'] for s in held], weights=w):.1%}")
        summary.append((fam, mean))
    print("\n" + "=" * 78)
    for fam, mean in sorted(summary, key=lambda r: -r[1]):
        print(f"  {fam:22s} held-out {mean:.1%}")
    print(f"  {'prototype (skimage)':22s} held-out 88.9%   [intensity h-maxima, flooded on intensity]")
    print(f"  {'floor: components':22s}          73.4%   [measured above; a component never splits]")
    _dump("sweep.json", log)
    return 0


#: the R2 optimum, picked identically by all three leave-one-condition-out splits
R2_BEST = dict(route="R2", sigma=10.0, f_lo=0.7, f_hi=1.3, thr=0.01)


def mode_diagnose(args) -> int:
    """Whose fault is the over-split — the seeder, or the partition?

    At the R2 optimum the under-split rate matches the prototype (9.1% vs 8.4%) and the
    whole deficit is over-splitting (7.0% vs 2.8%). Those are two very different bills:

    * **too many seeds** → the seeder is mis-tuned, and that is a parameter problem
      solvable inside the shipped catalog;
    * **right number of seeds, too many pieces** → the *partition* is splitting where an
      intensity flood would not, because a nearest-seed diagram always inserts a boundary
      between two seeds whereas a flood only cuts at an intensity saddle. That is the
      missing `height` socket, and no parameter reaches it.

    So count the seeds inside each label-confirmed blob and compare against both the label
    and the number of pieces the partition produced in that same blob.
    """
    _hdr("DIAGNOSE — is the over-split from the seeder or from the partition?")
    cal = calibration()
    px = float(cal["pixel_size_um"])
    by_cond, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ci = ch_index(chans, C_F)
    print(f"config: {R2_BEST}\n")
    print(f"{'cond':5s} {'blobs':>6s} {'seeds=lab':>10s} {'seeds>lab':>10s} {'seeds<lab':>10s}"
          f" {'pieces>lab':>11s} {'over|seeds=lab':>15s} {'over|seeds>lab':>15s}")
    rows = []
    for m, cond, _d in LABEL_FIELDS:
        r_eq = float(np.sqrt(a_single[cond] / np.pi))
        min_area = 0.30 * a_single[cond]
        n, e, s = g_seeded_voronoi(ci=ci, min_area_um2=min_area, sigma_um=R2_BEST["sigma"],
                                  seeder="spots", r_lo=R2_BEST["f_lo"] * r_eq,
                                  r_hi=R2_BEST["f_hi"] * r_eq, spot_threshold=R2_BEST["thr"])
        ds, env = source(m, (0,), Z_PLANE, chans)
        o, _eg = run(build(n, e), s, ds, env)
        newlab = raster(o, "grains")[0, 0, 0, 0].astype(np.int64)
        pts = table(o, D.POINT, "seeds")
        sy = np.rint(pts["y"]).astype(int)
        sx = np.rint(pts["x"]).astype(int)
        lab_ref, objs = reference_blobs(m)
        # seeds per REFERENCE blob (the same blob the label refers to)
        seed_blob = lab_ref[np.clip(sy, 0, lab_ref.shape[0] - 1),
                            np.clip(sx, 0, lab_ref.shape[1] - 1)]
        n_seed = np.bincount(seed_blob, minlength=int(lab_ref.max()) + 1)
        a_min_px = 0.30 * a_single[cond] / (px * px)
        eq = gt = lt = pgt = 0
        over_when_eq = over_when_gt = n_eq = n_gt = 0
        for oid, lab_n, _a in by_cond[cond]:
            sl = objs[oid - 1]
            mask = (lab_ref[sl] == oid)
            k = min(count_pieces(newlab[sl], mask, a_min_px), 3)
            ns = min(int(n_seed[oid]), 3)
            eq += (ns == lab_n); gt += (ns > lab_n); lt += (ns < lab_n)
            pgt += (k > lab_n)
            if ns == lab_n:
                n_eq += 1
                over_when_eq += (k > lab_n)
            elif ns > lab_n:
                n_gt += 1
                over_when_gt += (k > lab_n)
        N = len(by_cond[cond])
        print(f"{cond:5s} {N:6d} {eq/N:10.1%} {gt/N:10.1%} {lt/N:10.1%} {pgt/N:11.1%}"
              f" {over_when_eq/max(n_eq,1):15.1%} {over_when_gt/max(n_gt,1):15.1%}")
        rows.append(dict(cond=cond, n=N, seeds_eq=eq / N, seeds_gt=gt / N, seeds_lt=lt / N,
                         pieces_gt=pgt / N, n_seeds_eq=n_eq, n_seeds_gt=n_gt,
                         over_given_seeds_eq=over_when_eq / max(n_eq, 1),
                         over_given_seeds_gt=over_when_gt / max(n_gt, 1)))
    tot = sum(r["n"] for r in rows)
    w = np.array([r["n"] for r in rows], float)
    print(f"\npooled ({tot} blobs): seeds match the label {np.average([r['seeds_eq'] for r in rows], weights=w):.1%}"
          f" · too many seeds {np.average([r['seeds_gt'] for r in rows], weights=w):.1%}"
          f" · too few {np.average([r['seeds_lt'] for r in rows], weights=w):.1%}")
    n_eq_tot = sum(r["n_seeds_eq"] for r in rows)
    n_gt_tot = sum(r["n_seeds_gt"] for r in rows)
    oe = sum(r["over_given_seeds_eq"] * r["n_seeds_eq"] for r in rows) / max(n_eq_tot, 1)
    og = sum(r["over_given_seeds_gt"] * r["n_seeds_gt"] for r in rows) / max(n_gt_tot, 1)
    print(f"\nP(over-split | seed count was CORRECT) = {oe:.1%}   over {n_eq_tot} blobs")
    print(f"P(over-split | seed count was TOO HIGH) = {og:.1%}   over {n_gt_tot} blobs")
    print("\nRead: the first number is the partition's own error rate — blobs where the "
          "\nseeder got the count right and the nearest-seed diagram still cut them apart. "
          "\nThe second is the seeder's contribution. Whichever dominates is where the "
          "\nremaining 5 points live, and only one of the two is reachable by a parameter.")
    _dump("diagnose.json", rows)
    return 0


def mode_plateau(args) -> int:
    """Resolve the two boundary flags the sweep raised, rather than leaving them ambiguous."""
    _hdr("PLATEAU — are the boundary optima plateaus or unexplored edges?")
    cal = calibration()
    px = float(cal["pixel_size_um"])
    by_cond, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ref = {c: reference_blobs(m) for m, c, _d in LABEL_FIELDS}
    print("detect.spots threshold, extended DOWN by two decades from the sweep's lower bound.")
    print("If the score is flat, the sweep's 'on the boundary' flag marks a saturated")
    print("plateau (every blob already detected) and not a direction left unexplored.\n")
    print(f"{'thr':>10s} {'pooled':>8s}")
    for t in (0.0, 1e-5, 1e-4, 1e-3, 0.005):
        cfg = dict(route="R2", sigma=6.9, f_lo=0.7, f_hi=1.3, thr=t)
        p = _pool(_score_cfg(cfg, LABEL_FIELDS, ref, by_cond, a_single, px, chans))
        print(f"{t:>10g} {p:8.1%}")
    print("\nThe OTHER shipped seeder setting — detect.spots method=dog — swept over the")
    print("same band, since the diagnosis says the score IS the seeder's accuracy:\n")
    print(f"{'f_lo':>10s} {'log':>8s} {'dog':>8s}")
    for lo, hi in ((0.55, 1.1), (0.7, 1.3), (0.9, 1.6)):
        got = []
        for meth in ("spots", "spots_dog"):
            cfg = dict(R2_BEST, f_lo=lo, f_hi=hi, seeder=meth)
            got.append(_pool(_score_cfg(cfg, LABEL_FIELDS, ref, by_cond, a_single,
                                        px, chans)))
        print(f"{lo:>10.2f} {got[0]:8.1%} {got[1]:8.1%}")

    print("\nThe R4 flags (`sigma`=0.0 and opening radius `mr`=0.0) are one-sided by")
    print("construction: 0 means 'do not apply the operation', and there is nothing below")
    print("it. The honest reading is not 'widen the range' but 'both extra operations make")
    print("the result WORSE' — pre-smoothing before the Otsu cut costs 4.1 points by")
    print("sigma=6.9 and a greyscale opening costs 6.0 points by 0.3 r_eq.")
    return 0


CELL_MIN_UM2, CELL_MAX_UM2 = 20.0, 600.0


def g_cells(*, ci_gfp: int, ci_raw: int) -> Tuple[List[tuple], List[tuple], str]:
    """GFP objects, with a SECOND channel's intensity read at those same voxels.

    `analysis.measure`'s `raw` input is the sanctioned cross-channel measurement — its own
    docstring calls "segment channel 0, measure channel 1" the reason the socket exists, and
    the provenance comparison drops the channel stamp exactly when both branches are
    single-channel, which two `channel.select` taps off one source guarantee.

    Note there is no output-name socket on `analysis.measure`: its columns land on the Label
    table named by `labels`. Two measure nodes on one table would collide on
    `mean_intensity`, so a second channel means a second PULL, not a second node.
    """
    nodes = [("C", "channel.select", {"params": {"channels": [ci_gfp]}}),
             ("A", "analysis.segment",
              {"modes": {**_D2, "method": "threshold", "level": "otsu"},
               "params": {"name": "cells", "fill_holes": True,
                          "min_area": CELL_MIN_UM2, "max_area": CELL_MAX_UM2}}),
             ("P", "transform.label_to_points",
              {"modes": {"position": "centroid"},
               "params": {"labels": "cells", "name": "cellpts"}}),
             ("R", "channel.select", {"params": {"channels": [ci_raw]}}),
             ("M", "analysis.measure",
              {"params": {"labels": "cells", "stats": "mean,max,median,count",
                          "shape": "solidity,axis_major,axis_minor"}})]
    edges = [(SEED_NODE_ID, "C"), ("C", "A"), ("A", "P"), ("P", "M"),
             (SEED_NODE_ID, "R"), ("R", "M", "raw")]
    return nodes, edges, "M"


#: Nile-Blue cut in raw DN. Calibrated on the pure-F controls, where the well contains no
#: inert material and therefore every NB count is known to be artefact: 200 DN drives the
#: phantom phase to exactly 0.0000 at all three while retaining 109-137% of what Otsu found
#: at the three mixed conditions. Interior to its range — the mask balloons below it
#: (277% at 100 DN) and real inert dies above it (73% at 300, 0% at 700).
#:
#: Otsu is kept for the FUNCTIONAL channel and replaced only here, and the asymmetry is the
#: point: F carries real signal in every one of the 21 conditions, so a relative rule is
#: safe and was verified bit-identical to the prototype. I is ABSENT BY DESIGN in three of
#: them, and a relative rule has no defence against being handed a channel with no signal —
#: it splits the autofluorescence pedestal from the void and reports the brighter half.
NB_FIXED_DN = 200.0


def g_phase(*, ci: int, min_area_um2: float, level: str = "otsu",
            threshold_dn: float = 0.0) -> Tuple[List[tuple], List[tuple], str]:
    """One body channel: unfiltered 0/1 mask, size-filtered components, and the per-plane
    solid fraction computed by a node rather than by numpy.

    `transform.transfer_domain voxel→plane reducer=mean` on a uint8 0/1 mask IS the area
    fraction. Domain.PLANE carries axes {m,t,z} and no `c`, so it would average the channels
    together on a multi-channel Dataset — the upstream `channel.select` is what makes the
    number per-phase rather than a mixture.
    """
    nodes = [("C", "channel.select", {"params": {"channels": [ci]}}),
             ("T", "analysis.threshold",
              {"modes": {"method": ("fixed" if level == "fixed" else "otsu"),
                         "scope": "plane"},
               "params": {"name": "fg", **({"threshold": float(threshold_dn)}
                                           if level == "fixed" else {})}}),
             ("A", "analysis.segment",
              {"modes": {**_D2, "method": "threshold", "level": level},
               "params": {"name": "blobs", "fill_holes": True,
                          "min_area": float(min_area_um2),
                          **({"threshold": float(threshold_dn)} if level == "fixed" else {})}}),
             ("X", "transform.transfer_domain",
              {"modes": {"from_domain": "voxel", "to_domain": "plane", "reducer": "mean"},
               "params": {"attr": "fg"}})]
    edges = [(SEED_NODE_ID, "C"), ("C", "T"), ("T", "A"), ("A", "X")]
    return nodes, edges, "X"


def mode_phases(args) -> int:
    """Solid fraction per phase per timepoint, filtered and unfiltered.

    The DIFFERENCE between the two is the phantom-phase measurement: at a pure-functional
    control there is no inert material in the well at all, so every inert voxel Otsu finds
    is an artefact of thresholding a channel with no signal in it. `min_area` has to collapse
    that to ~0 AND leave real inert standing where it exists, or it is not a fix.
    """
    _hdr("PHASES — per-phase solid fraction over time (6 conditions x 15 timepoints)")
    _by, a_single = load_labels()
    a_ref = float(np.median(list(a_single.values())))
    ts = tuple(range(15))
    chans = (C_GFP, C_F, C_I)
    rows = []
    print(f"min_area = 0.30 x {a_ref:.0f} µm² = {0.30*a_ref:.0f} µm²  (median of the three "
          f"label-derived single-granule areas; one number for all conditions so the filter "
          f"cannot be tuned per field)\n")
    print(f"{'cond':5s} {'F frac':>17s} {'I frac':>17s} {'void':>7s} {'I kept':>7s}")
    print(f"{'':5s} {'raw    filtered':>17s} {'raw    filtered':>17s}")
    for m, cond, ffrac, fsz, isz in RESULT_FIELDS:
        got = {}
        for tag, cacq in (("F", C_F), ("I", C_I)):
            ds, env = source(m, ts, Z_PLANE, chans)
            n, e, s = g_phase(ci=ch_index(chans, cacq), min_area_um2=0.30 * a_ref,
                              level=("fixed" if tag == "I" else "otsu"),
                              threshold_dn=NB_FIXED_DN)
            o, _eg = run(build(n, e), s, ds, env)
            gates(o, "blobs")
            unfilt = np.asarray(o.get(D.PLANE, "fg").values).reshape(-1)   # (M,T,Z) -> T
            tb = table(o, D.LABEL, "blobs")
            npx = float(np.prod(raster(o, "blobs").shape[-2:]))
            filt = (np.array([tb["area"][tb["t"] == t].sum() / npx for t in ts])
                    if "area" in tb else np.zeros(len(ts)))
            got[tag] = (unfilt, filt)
            rows.append(dict(cond=cond, phase=tag, f_fraction_design=ffrac,
                             f_size=fsz, i_size=isz, m=m,
                             raw=[float(v) for v in unfilt],
                             filtered=[float(v) for v in filt]))
        fu, ff = got["F"]
        iu, ifl = got["I"]
        void = 1.0 - ff.mean() - ifl.mean()
        kept = ifl.mean() / max(iu.mean(), 1e-9)
        print(f"{cond:5s} {fu.mean():8.4f} {ff.mean():8.4f} {iu.mean():8.4f} "
              f"{ifl.mean():8.4f} {void:7.3f} {kept:7.1%}"
              + ("   <- pure F: any inert here is phantom" if isz is None else ""))
    _dump("phases.json", rows)
    print("\nvoid = 1 - F_filtered - I_filtered. The two phases are near-disjoint (measured "
          "\nJaccard 0.004-0.011), which is what licenses the subtraction; there is no shipped "
          "\nboolean op to intersect two masks, so the overlap is the reporting layer's job.")
    return 0


def mode_cells(args) -> int:
    """GFP cells, plus each cell's intensity in a DIFFERENT channel via `measure.raw`."""
    _hdr("CELLS — GFP objects, with F- and NileBlue-channel intensity at the same voxels")
    chans = (C_GFP, C_F, C_I)
    rows = []
    print(f"area window {CELL_MIN_UM2:.0f}-{CELL_MAX_UM2:.0f} µm² on analysis.segment\n")
    print(f"{'cond':5s} {'n cells':>8s} {'d_eq µm':>8s} {'GFP':>7s} {'F ch':>8s} "
          f"{'NB ch':>8s} {'NB/void':>8s}")
    for m, cond, ffrac, fsz, isz in RESULT_FIELDS:
        per = {}
        for tag, craw in (("F", C_F), ("I", C_I)):
            ds, env = source(m, (0,), Z_PLANE, chans)
            n, e, s = g_cells(ci_gfp=ch_index(chans, C_GFP), ci_raw=ch_index(chans, craw))
            o, _eg = run(build(n, e), s, ds, env)
            gates(o, "cells")
            per[tag] = table(o, D.LABEL, "cells")
            per[tag + "_img"] = np.asarray(
                _dask()[m, 0, Z_PLANE, craw]).astype(np.float32)
        tb = per["F"]
        px = float(calibration()["pixel_size_um"])
        area_um2 = tb["area"] * px * px
        d_eq = 2.0 * np.sqrt(area_um2 / np.pi)
        # a cell's own GFP brightness comes from the main branch of the SAME pull, since the
        # measured image there is the GFP tap; `raw` only redirects which pixels are read.
        gfp = tb["max_intensity"]
        f_int = per["F"]["mean_intensity"]
        nb_int = per["I"]["mean_intensity"]
        # local void level for the NileBlue channel: the median of everything BELOW its own
        # Otsu cut is not available as a node output, so use the plane's 25th percentile,
        # which on a bed this sparse is solidly interstitial.
        nb_void = float(np.percentile(per["I_img"], 25))
        rows.append(dict(cond=cond, m=m, n=int(len(tb["id"])), f_fraction_design=ffrac,
                         i_size=isz, d_eq_um=float(np.median(d_eq)),
                         gfp_max=float(np.median(gfp)),
                         f_mean=float(np.median(f_int)), nb_mean=float(np.median(nb_int)),
                         nb_void=nb_void, nb_above_void=float(np.median(nb_int) - nb_void)))
        print(f"{cond:5s} {len(tb['id']):8d} {np.median(d_eq):8.1f} "
              f"{np.median(gfp):7.0f} {np.median(f_int):8.0f} {np.median(nb_int):8.0f} "
              f"{np.median(nb_int)-nb_void:+8.0f}"
              + ("   <- pure F: no dye available to leach" if isz is None else ""))
    _dump("cells.json", rows)
    print("\nNB-above-void is the leaching signal: the cells are GFP-labelled, and where "
          "\nNile-Blue granules exist the dye leaches into them. The three pure-F controls "
          "\nare the experiment's own zero — no inert material, so nothing to take up.")
    return 0


def mode_track(args) -> int:
    """`track.link` on labels (max IoU) and on points (nearest), with a match-rate gate."""
    _hdr("TRACK — IoU label linking and nearest-point cell linking")
    _by, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ts = tuple(range(15))
    dt = float(calibration()["dt_s"])
    print(f"frame_interval = {dt:.1f} s = {dt/60:.2f} min  (event-table measurement; the "
          f"file's own dt_s is 4.0009x larger)\n")
    rows = []
    print(f"{'cond':5s} {'iou':>5s} {'objs':>6s} {'tracks':>7s} {'span>=90%':>10s} "
          f"{'match/frame':>12s} {'cells':>6s} {'ctracks':>8s} {'µm/min':>8s}")
    for m, cond, _ff, _fs, _is in RESULT_FIELDS[:3] + RESULT_FIELDS[3:]:
        r_eq = float(np.sqrt(a_single.get(cond, np.median(list(a_single.values()))) / np.pi))
        min_area = 0.30 * float(np.median(list(a_single.values())))
        for iou in (0.1, 0.3):
            n, e, s = g_seeded_voronoi(
                ci=ch_index(chans, C_F), min_area_um2=min_area, sigma_um=R2_BEST["sigma"],
                seeder="spots", r_lo=R2_BEST["f_lo"] * r_eq, r_hi=R2_BEST["f_hi"] * r_eq,
                spot_threshold=R2_BEST["thr"])
            n = list(n) + [("K", "track.link",
                            {"modes": {"target": "label"},
                             "params": {"labels": "grains", "name": "gtracks",
                                        "iou_threshold": iou}})]
            e = list(e) + [(s, "K")]
            ds, env = source(m, ts, Z_PLANE, chans)
            o, _eg = run(build(n, e), "K", ds, env)
            gt = table(o, D.LABEL, "grains")
            tt = table(o, D.TRACK, "gtracks")
            n_obj = len(gt["id"])
            n_trk = len(np.unique(tt["track_id"])) if "track_id" in tt else 0
            # a track spanning >=90% of the series is an identity that survived; and the
            # per-frame match rate is the number that decides whether motion is measurable
            if "track_id" in tt:
                lens = np.bincount(tt["track_id"].astype(int))
                lens = lens[lens > 0]
                span = float((lens >= 0.9 * len(ts)).mean())
                # NOT a match rate: every object appears in exactly one membership row, so
                # rows/objects is 1.0 by construction whatever the linking did. Kept only to
                # show that it is vacuous — `span` is the number that carries information.
                matched = float(len(tt["track_id"]) / max(n_obj, 1))
            else:
                span = matched = float("nan")
            # cells, point mode
            cn, ce, cs = g_cells(ci_gfp=ch_index(chans, C_GFP), ci_raw=ch_index(chans, C_F))
            # `track.link` attaches a TRACK membership table and NOTHING else — it never
            # writes `track_id` back onto the member layer, in either mode
            # (`track/link.py:86`, a single `ds.with_structure(membership.to_table(...))`).
            # `analysis.object_metrics` requires `track_id` ON the member table, so those two
            # shipped nodes do not compose directly. A third one bridges them:
            # `transfer_structure track->point` is `broadcast_track`, which is exactly the
            # write-back. Three nodes where two looked sufficient.
            cn = list(cn) + [("K2", "track.link",
                              {"modes": {"target": "point"},
                               "params": {"points": "cellpts", "name": "ctracks",
                                          "max_distance": 3.0 * 12.0}}),
                             ("B", "transform.transfer_structure",
                              {"modes": {"from_domain": "track", "to_domain": "point",
                                         "reducer": "mean"},
                               "params": {"attr": "track_id", "source_layer": "ctracks",
                                          "target_layer": "cellpts", "name": "track_id"}}),
                             ("O", "analysis.object_metrics",
                              {"modes": {"target": "point"},
                               "params": {"points": "cellpts", "metrics": "velocity,speed",
                                          "frame_interval": dt}})]
            ce = list(ce) + [(cs, "K2"), ("K2", "B"), ("B", "O")]
            dsc, envc = source(m, ts, Z_PLANE, chans)
            oc, _e2 = run(build(cn, ce), "O", dsc, envc)
            ct = table(oc, D.POINT, "cellpts")
            ctt = table(oc, D.TRACK, "ctracks")
            n_cell = len(ct["id"])
            n_ctrk = len(np.unique(ctt["track_id"])) if "track_id" in ctt else 0
            spd = ct.get("speed")
            um_min = (float(np.nanmedian(spd[np.isfinite(spd) & (spd > 0)])) * 60.0
                      if spd is not None and np.isfinite(spd).any() else float("nan"))
            print(f"{cond:5s} {iou:5.1f} {n_obj:6d} {n_trk:7d} {span:10.1%} "
                  f"{matched:12.1%} {n_cell:6d} {n_ctrk:8d} {um_min:8.3f}")
            rows.append(dict(cond=cond, m=m, iou=iou, n_objects=n_obj, n_tracks=n_trk,
                             span_ge_90pct=span, match_per_frame=matched,
                             n_cells=n_cell, n_cell_tracks=n_ctrk, speed_um_per_min=um_min))
            break                                   # iou=0.3 only if the gate needs it
    _dump("track.json", rows)
    print("\nCell speed 0.14-0.32 µm/min here against the prototype's 0.39 µm/min "
          "(drift-corrected)"
          "\n— same order, and both inside the textbook mammalian crawling range. The node "
          "\nroute reads LOWER, not higher, so 'no drift model inflates it' is not the "
          "\nexplanation; `track.link` point mode is nearest-neighbour within max_distance and "
          "\nthe median is taken over every linked step, which the prototype's drift-corrected "
          "\nper-track step is not. The gap is unattributed and stays that way until measured."
          "\n\nThe load-bearing number above is `span>=90%`: only 2.4-30.7% of granule tracks "
          "\nlast the series, so granule IDENTITY does not survive and per-granule motion is "
          "\nnot measurable from this linking — the same blocker the prototype hit. "
          "`match/frame`"
          "\nis 100.0% at every condition because it is vacuous by construction, not because "
          "\nthe linking is perfect.")
    return 0


def mode_phantom(args) -> int:
    """Why does a µm² size filter only half-remove the phantom inert phase?

    At a pure-functional control the well contains no inert material, so every voxel Otsu
    calls "inert" in the Nile-Blue channel is an artefact. The size filter cut it by half
    and stopped. The candidate explanation is that the residual is not speckle at all: the
    F granules carry a small autofluorescence pedestal into the Nile-Blue channel, so the
    phantom objects are granule-BODY shaped and therefore granule-SIZED, and no area
    threshold can distinguish them from real inert granules.

    That is falsifiable. If it is right, the phantom mask is spatially CONTAINED in the F
    mask; if the phantom were noise it would sit in the void, anti-correlated with F.
    """
    _hdr("PHANTOM — is the residual inert phase the F granules bleeding through?")
    _by, a_single = load_labels()
    a_ref = float(np.median(list(a_single.values())))
    chans = (C_GFP, C_F, C_I)
    print(f"{'cond':5s} {'F frac':>7s} {'I frac':>7s} {'Jaccard':>8s} "
          f"{'I inside F':>11s} {'I in void':>10s} {'expect if random':>17s}")
    rows = []
    for m, cond, _ff, _fs, isz in RESULT_FIELDS:
        masks = {}
        for tag, cacq in (("F", C_F), ("I", C_I)):
            ds, env = source(m, (0,), Z_PLANE, chans)
            n, e, s = g_phase(ci=ch_index(chans, cacq), min_area_um2=0.30 * a_ref)
            o, _eg = run(build(n, e), s, ds, env)
            masks[tag] = raster(o, "blobs")[0, 0, 0, 0] > 0
        F, I = masks["F"], masks["I"]
        inter = float((F & I).sum())
        union = float((F | I).sum())
        jac = inter / max(union, 1.0)
        inside = inter / max(float(I.sum()), 1.0)
        in_void = float((I & ~F).sum()) / max(float(I.sum()), 1.0)
        print(f"{cond:5s} {F.mean():7.4f} {I.mean():7.4f} {jac:8.4f} {inside:11.1%} "
              f"{in_void:10.1%} {F.mean():17.1%}"
              + ("   <- pure F" if isz is None else ""))
        rows.append(dict(cond=cond, m=m, f_frac=float(F.mean()), i_frac=float(I.mean()),
                         jaccard=jac, i_inside_f=inside, i_in_void=in_void,
                         random_expectation=float(F.mean()), pure_f=(isz is None)))
    # ── the fix the diagnosis implies: an ABSOLUTE cut, not a relative one ────────
    # Otsu is a *relative* rule: it splits whatever distribution it is given, so on a
    # channel with no signal it dutifully splits the autofluorescence pedestal from the
    # void and calls the brighter half "inert". `analysis.segment level=fixed` takes a
    # raw-DN threshold instead, and the three pure-F controls are precisely the calibration
    # set for it — every Nile-Blue count they contain is known to be artefact. This is the
    # experiment's own zero control choosing a node parameter.
    print("\n" + "-" * 78)
    print("FIX — analysis.segment level=fixed on the Nile-Blue channel, threshold in raw DN")
    print("calibrated on the pure-F controls, where all NB signal is known to be artefact.")
    print("-" * 78)
    print(f"{'DN':>6s} {'phantom @ pure-F (want ~0)':>34s} {'real inert kept @ mixed':>28s}")
    print(f"{'':6s} {'M01':>11s} {'M08':>11s} {'M15':>11s} {'M04':>9s} {'M11':>9s} {'M19':>9s}")
    base = {}
    for m, cond, _f, _fs, isz in RESULT_FIELDS:
        if isz is not None:
            ds, env = source(m, (0,), Z_PLANE, chans)
            n, e, s = g_phase(ci=ch_index(chans, C_I), min_area_um2=0.30 * a_ref)
            o, _eg = run(build(n, e), s, ds, env)
            base[cond] = float((raster(o, "blobs")[0, 0, 0, 0] > 0).mean())
    fix_rows = []
    for dn in (100, 150, 200, 300, 450, 700):
        cells_out = []
        for m, cond, _f, _fs, isz in RESULT_FIELDS:
            ds, env = source(m, (0,), Z_PLANE, chans)
            nodes = [("C", "channel.select",
                      {"params": {"channels": [ch_index(chans, C_I)]}}),
                     ("A", "analysis.segment",
                      {"modes": {**_D2, "method": "threshold", "level": "fixed"},
                       "params": {"name": "blobs", "fill_holes": True,
                                  "threshold": float(dn),
                                  "min_area": 0.30 * a_ref}})]
            o, _eg = run(build(nodes, [(SEED_NODE_ID, "C"), ("C", "A")]), "A", ds, env)
            frac = float((raster(o, "blobs")[0, 0, 0, 0] > 0).mean())
            cells_out.append((cond, frac, isz))
        pf = [f for _c, f, i in cells_out if i is None]
        mx = [(c, f) for c, f, i in cells_out if i is not None]
        print(f"{dn:6d} " + " ".join(f"{v:11.4f}" for v in pf) + " "
              + " ".join(f"{f/max(base[c],1e-9):9.1%}" for c, f in mx))
        fix_rows.append(dict(threshold_dn=dn,
                             phantom={c: f for c, f, i in cells_out if i is None},
                             kept={c: f / max(base[c], 1e-9) for c, f in mx}))
    _dump("phantom_fix.json", fix_rows)
    _dump("phantom.json", rows)
    print("\n`I inside F` against `expect if random` is the test. Well above the random")
    print("expectation means the inert mask is sitting ON the functional granules, i.e. it")
    print("is those granules' own autofluorescence pedestal and not inert material — which")
    print("also means it is granule-sized, so no area filter can reach it. Well below means")
    print("the two phases genuinely interlock and the Jaccard column licenses void = 1-F-I.")
    return 0


def mode_surface(args) -> int:
    """Is the boundary in the dim SEAM, or just halfway between two seeds?

    The 585 labels say how many granules are in a blob and nothing whatever about where the
    dividing surface lies, so "the partition is 99.6% exact" is a statement about COUNTING
    and cannot be read as a statement about boundary placement. Nothing measured so far
    checks the granules against the intensity at their own surface — the user's "high energy
    areas" tell, which is about the boundary rather than the centre.

    `analysis.voronoi` assigns each voxel to the nearest seed in µm. It never reads the image,
    so its divide sits at the geometric midpoint of two seeds. A flood of the intensity
    landscape puts the divide at the intensity SADDLE instead. With the seeds held identical,
    the difference between the two partitions is exactly the cost of ignoring surface
    intensity — no labels needed, and no other variable moving.

    Three unsupervised checks, none of which need ground truth:
      1. how many foreground voxels the two partitions disagree about, and the mean boundary
         displacement that implies;
      2. how DIM the boundary is in each case, relative to the granule interior — a divide
         sitting in a real seam reads darker than one cutting through a bright body;
      3. the solidity of the resulting objects. These granules are convex polygons, so a
         partition that carves thin slivers is wrong on a shape prior alone.
    """
    _hdr("SURFACE — is the divide in the dim seam, or halfway between the seeds?")
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    cal = calibration()
    px = float(cal["pixel_size_um"])
    _by, a_single = load_labels()
    chans = (C_GFP, C_F, C_I)
    ci = ch_index(chans, C_F)
    print("seeds are IDENTICAL in both columns — only the partition rule differs, so the\n"
          "difference is attributable to surface intensity and to nothing else.\n")
    print(f"{'cond':5s} {'pairs':>6s} {'disagree':>9s} {'mean shift':>11s} "
          f"{'boundary dim: voronoi':>22s} {'flood':>8s} {'solidity: voronoi':>18s} "
          f"{'flood':>8s}")
    rows = []
    for m, cond, _d in LABEL_FIELDS:
        r_eq = float(np.sqrt(a_single[cond] / np.pi))
        n, e, s = g_seeded_voronoi(ci=ci, min_area_um2=0.30 * a_single[cond],
                                  sigma_um=R2_BEST["sigma"], seeder="spots",
                                  r_lo=R2_BEST["f_lo"] * r_eq, r_hi=R2_BEST["f_hi"] * r_eq,
                                  spot_threshold=R2_BEST["thr"])
        ds, env = source(m, (0,), Z_PLANE, chans)
        o, _eg = run(build(n, e), s, ds, env)
        blobs = raster(o, "blobs")[0, 0, 0, 0]
        vor = raster(o, "grains")[0, 0, 0, 0].astype(np.int64)
        tb = table(o, D.LABEL, "grains")
        pts = table(o, D.POINT, "seeds")
        raw = raw_plane(m, 0, Z_PLANE, C_F)
        sm = ndi.gaussian_filter(raw, R2_BEST["sigma"] / px)

        # ── the reference partition: the SAME seeds, flooded on intensity ──────
        # Not a pipeline step — no shipped node can do this — but a legitimate measuring
        # instrument for how far the node route's boundary sits from the intensity-optimal
        # one. Markers are labelled by seed row, so both partitions land in the same id space.
        markers = np.zeros_like(vor)
        sy = np.clip(np.rint(pts["y"]).astype(int), 0, vor.shape[0] - 1)
        sx = np.clip(np.rint(pts["x"]).astype(int), 0, vor.shape[1] - 1)
        keep = blobs[sy, sx] > 0                      # a seed in the void marks nothing
        markers[sy[keep], sx[keep]] = np.arange(1, int(keep.sum()) + 1)
        flood = watershed(-sm, markers=markers, mask=blobs > 0).astype(np.int64)

        # relabel the Voronoi result into that same seed-row space via `point_id`
        seed_row = {int(v): i + 1 for i, v in enumerate(pts["id"][keep])}
        lut = np.zeros(int(vor.max()) + 1, dtype=np.int64)
        for gid, pid in zip(tb["id"].astype(int), tb["point_id"].astype(int)):
            lut[gid] = seed_row.get(pid, 0)
        vor_s = lut[vor]

        fg = (blobs > 0) & (vor_s > 0) & (flood > 0)
        disagree = float(((vor_s != flood) & fg).sum()) / max(float(fg.sum()), 1.0)

        def _bnd(lab):
            """Voxels whose 4-neighbour holds a DIFFERENT non-zero instance."""
            b = np.zeros(lab.shape, bool)
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                r = np.roll(lab, sh, axis=ax)
                b |= (lab > 0) & (r > 0) & (lab != r)
            return b

        bv, bf = _bnd(vor_s), _bnd(flood)
        # mean displacement: the area the two disagree about, spread over the boundary they
        # share — the honest way to turn an area into a distance
        shift_um = disagree * float(fg.sum()) / max(float(bv.sum()), 1.0) * px
        interior = float(np.median(sm[fg]))
        dim_v = float(np.median(sm[bv])) / interior if bv.any() else float("nan")
        dim_f = float(np.median(sm[bf])) / interior if bf.any() else float("nan")

        def _sol(lab):
            from skimage.measure import regionprops
            return float(np.median([p.solidity for p in regionprops(lab.astype(np.int32))
                                    if p.area > 40]))

        sv, sf = _sol(vor_s), _sol(flood)
        n_pairs = int(len({(min(a, b), max(a, b)) for a, b in
                           zip(vor_s[bv].ravel(), np.roll(vor_s, 1, 0)[bv].ravel())
                           if a and b and a != b}))
        print(f"{cond:5s} {n_pairs:6d} {disagree:9.1%} {shift_um:8.1f} µm "
              f"{dim_v:22.3f} {dim_f:8.3f} {sv:18.3f} {sf:8.3f}")
        rows.append(dict(cond=cond, m=m, n_pairs=n_pairs, disagree_frac=disagree,
                         mean_shift_um=shift_um, boundary_dim_voronoi=dim_v,
                         boundary_dim_flood=dim_f, solidity_voronoi=sv,
                         solidity_flood=sf, r_eq_um=r_eq,
                         shift_over_r_eq=shift_um / r_eq))
    _dump("surface.json", rows)
    w = np.array([r["n_pairs"] for r in rows], float)
    print(f"\npooled: {np.average([r['disagree_frac'] for r in rows], weights=w):.1%} of "
          f"foreground assigned differently; mean boundary shift "
          f"{np.average([r['mean_shift_um'] for r in rows], weights=w):.1f} µm "
          f"= {np.average([r['shift_over_r_eq'] for r in rows], weights=w):.1%} of a granule "
          f"radius.")
    print("\n`boundary dim` is the median smoothed intensity ON the divide, over the median "
          "\ninside the objects. Lower is better: it means the divide sits in a real seam. If "
          "\nthe flood column is materially lower than the voronoi column, then the geometric "
          "\npartition is cutting through brighter material than it needs to — i.e. the "
          "\nboundary is NOT being checked against the surface, and that has a size.")
    return 0


_MODES = {"parity": mode_parity, "bench": mode_bench, "graphs": mode_graphs,
          "sweep": mode_sweep, "diagnose": mode_diagnose, "plateau": mode_plateau,
          "phases": mode_phases, "cells": mode_cells, "track": mode_track,
          "phantom": mode_phantom, "surface": mode_surface}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=sorted(_MODES))
    args = ap.parse_args(argv)
    if not os.path.exists(ND2):
        print(f"SKIP: sample not found: {ND2}")
        return 0
    return _MODES[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
