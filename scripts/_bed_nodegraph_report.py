"""Figures + `CodeLog/NODE_PIPELINE.html` from the cached results of `_bed_nodegraph.py`.

No Qt in this file: the GUI screenshots come from `_bed_nodegraph_shots.py` in its own
process, and this one only needs the engine and matplotlib. Run order is
`_bed_nodegraph.py <modes>` → `_bed_nodegraph_shots.py` → this.

The one thing worth stating about the figures: the label rasters are coloured through a
STABLE id→colour hash, so a granule keeps its colour from one stage to the next and a reader
can follow a single object through the chain instead of re-finding it in every panel.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import ListedColormap                           # noqa: E402

import _bed_nodegraph as B                                             # noqa: E402
from nodegraph.domains import Domain as D                              # noqa: E402

IMG = os.path.join(_ROOT, "CodeLog", "img", "nodes")
CACHE = B.CACHE
FIG = dict(dpi=130, facecolor="#14161a")
INK, MUTE = "#e8e8ea", "#9aa0a8"
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE,
                     "ytick.color": MUTE, "axes.edgecolor": "#3a3f47",
                     "axes.facecolor": "#1b1e24", "font.size": 8,
                     "figure.facecolor": "#14161a", "savefig.facecolor": "#14161a"})


def _load(name):
    p = os.path.join(CACHE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def label_cmap(n=4096, seed=7):
    """A stable id→colour map: id k always gets colour k, in every panel and every stage."""
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0.30, 1.0, size=(n, 3))
    cols[0] = (0.06, 0.07, 0.09)                     # background
    return ListedColormap(cols)


CMAP = label_cmap()


def _show(ax, a, *, title, kind="grey", vmax=None, crop=None):
    if crop:
        y0, y1, x0, x1 = crop
        a = a[y0:y1, x0:x1]
    if kind == "labels":
        ax.imshow(a % 4096, cmap=CMAP, interpolation="nearest", vmin=0, vmax=4095)
    elif kind == "mask":
        ax.imshow(a > 0, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
    else:
        hi = vmax if vmax else np.percentile(a, 99.5)
        ax.imshow(a, cmap="gray", interpolation="nearest",
                  vmin=np.percentile(a, 2), vmax=hi)
    ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a3f47")
    return ax


# ══ figure 1: the chain, stage by stage, on real data ═════════════════════════

def fig_stages():
    """One panel per node output, so the reader sees the image being transformed."""
    _by, a_single = B.load_labels()
    cond, m = "M08", 7
    r_eq = float(np.sqrt(a_single[cond] / np.pi))
    chans = (B.C_GFP, B.C_F, B.C_I)
    ci = B.ch_index(chans, B.C_F)
    ds, env = B.source(m, (0,), B.Z_PLANE, chans)
    px = float(B.calibration()["pixel_size_um"])

    n, e, s = B.g_seeded_voronoi(ci=ci, min_area_um2=0.30 * a_single[cond],
                                sigma_um=B.R2_BEST["sigma"], seeder="spots",
                                r_lo=B.R2_BEST["f_lo"] * r_eq, r_hi=B.R2_BEST["f_hi"] * r_eq,
                                spot_threshold=B.R2_BEST["thr"])
    n, e, s = B.with_measure(n, e, s, labels="grains")
    # paint a MEASURED per-object column back onto the raster — a node making its own figure
    n = list(n) + [("W", "transform.transfer_structure",
                    {"modes": {"from_domain": "label", "to_domain": "voxel",
                               "reducer": "mean"},
                     "params": {"attr": "solidity", "source_layer": "grains",
                                "target_layer": "grains", "name": "solidity_paint"}})]
    e = list(e) + [(s, "W")]
    out, _eg = B.run(B.build(n, e), "W", ds, env)

    raw = B.raw_plane(m, 0, B.Z_PLANE, B.C_F)
    blobs = B.raster(out, "blobs")[0, 0, 0, 0]
    grains = B.raster(out, "grains")[0, 0, 0, 0]
    paint = B.raster(out, "solidity_paint")[0, 0, 0, 0].astype(float)
    pts = B.table(out, D.POINT, "seeds")
    tb = B.table(out, D.LABEL, "grains")

    from scipy import ndimage as ndi
    sm = ndi.gaussian_filter(raw, B.R2_BEST["sigma"] / px)

    W = int(round(430.0 / px))
    y0, x0 = 470, 330
    crop = (y0, y0 + W, x0, x0 + W)
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.0))
    A = axes.ravel()
    _show(A[0], raw, title="① io.load → channel.select [1]\nraw R-B, 12-bit", crop=crop)
    _show(A[1], blobs > 0, title="② analysis.segment level=otsu\nforeground (bit-identical "
          "to the prototype)", kind="mask", crop=crop)
    _show(A[2], blobs, title=f"② …same node, components + min_area\n"
          f"{len(np.unique(blobs))-1} blobs kept of 1906 raw", kind="labels", crop=crop)
    _show(A[3], sm, title=f"③ enhance.gaussian σ={B.R2_BEST['sigma']} µm\n"
          "(AFTER the threshold, so the cut cannot move)", crop=crop)
    ax = _show(A[4], sm, title=f"④ detect.spots → {len(pts['id'])} seeds\n"
               f"r ∈ [{B.R2_BEST['f_lo']*r_eq:.0f}, {B.R2_BEST['f_hi']*r_eq:.0f}] µm from "
               f"the LABELS", crop=crop)
    sel = ((pts["y"] >= y0) & (pts["y"] < y0 + W) & (pts["x"] >= x0) & (pts["x"] < x0 + W))
    ax.plot(pts["x"][sel] - x0, pts["y"][sel] - y0, "o", ms=4.5, mfc="none",
            mec="#4dd2ff", mew=1.1)
    _show(A[5], grains, title=f"⑤ analysis.voronoi bound=per_region\n"
          f"{len(tb['id'])} instances — each blob split by ITS OWN seeds",
          kind="labels", crop=crop)
    pm = np.where(grains > 0, paint, np.nan)[crop[0]:crop[1], crop[2]:crop[3]]
    im = A[6].imshow(pm, cmap="viridis", vmin=0.80, vmax=1.0, interpolation="nearest")
    A[6].set_title("⑥ analysis.measure shape=solidity, then\n"
                   "transfer_structure label→voxel to paint it", fontsize=7.5, pad=3)
    A[6].set_xticks([]), A[6].set_yticks([])
    plt.colorbar(im, ax=A[6], fraction=0.046, pad=0.02).set_label("solidity", fontsize=7)
    _show(A[7], grains, title="⑦ the same field, whole plane\n1024² px = 1.76 mm across",
          kind="labels")
    fig.suptitle(f"The chain on real data — {cond} (pure medium functional), "
                 f"z={B.Z_PLANE}, t=0.  Panels ①-⑥ are a {430:.0f} µm crop; "
                 f"colours are a stable id→colour hash, so one granule keeps its colour "
                 f"across ③-⑤.", fontsize=8.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = os.path.join(IMG, "fig_stages.png")
    fig.savefig(p, **FIG)
    plt.close(fig)
    print(f"  {p}")
    return dict(n_blobs=int(len(np.unique(blobs)) - 1), n_seeds=int(len(pts["id"])),
                n_grains=int(len(tb["id"])))


# ══ figure 2: the bench ═══════════════════════════════════════════════════════

def fig_bench():
    sw = _load("sweep.json") or []
    loco = [r for r in sw if r.get("stage") == "loco"]
    fams, order = {}, []
    for r in loco:
        fams.setdefault(r["family"], []).append(r)
        if r["family"] not in order:
            order.append(r["family"])
    names, vals, errs = [], [], []
    for f in order:
        rs = fams[f]
        w = np.array([r["n"] for r in rs], float)
        ex = np.array([r["exact"] for r in rs], float)
        names.append(f.split()[0] + "\n" + " ".join(f.split()[1:]))
        vals.append(float(np.average(ex, weights=w)))
        errs.append(float(ex.max() - ex.min()) / 2)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    xs = np.arange(len(names))
    ax.bar(xs, np.array(vals) * 100, yerr=np.array(errs) * 100, capsize=3,
           color="#4dd2ff", edgecolor="#8fe3ff", width=0.6, alpha=0.9)
    ax.axhline(88.9, color="#ffd166", ls="--", lw=1.4)
    ax.text(len(names) - 0.45, 89.4, "prototype (hand-rolled skimage) 88.9%",
            color="#ffd166", fontsize=7.5, ha="right")
    ax.axhline(73.4, color="#ff6b6b", ls=":", lw=1.4)
    ax.text(len(names) - 0.45, 71.4, "connected components 73.4% — a component never splits",
            color="#ff6b6b", fontsize=7.5, ha="right")
    for x, v in zip(xs, vals):
        ax.text(x, v * 100 + 1.4, f"{v:.1%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs), ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("exact granule count, held out (%)")
    ax.set_ylim(65, 95)
    ax.set_title("Instance segmentation scored against 582 hand-labelled blobs\n"
                 "parameters chosen on two conditions, scored on the third, three ways round",
                 fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(IMG, "fig_bench.png")
    fig.savefig(p, **FIG)
    plt.close(fig)
    print(f"  {p}")
    return dict(zip([n.replace("\n", " ") for n in names], vals))


# ══ figure 3: sweeps + the diagnosis ══════════════════════════════════════════

def fig_sweep_diag():
    sw = _load("sweep.json") or []
    dg = _load("diagnose.json") or []
    stages = {}
    for r in sw:
        if r.get("stage", "").startswith("R2 ·") or r.get("stage", "").startswith("R1 ·"):
            stages.setdefault(r["stage"], []).append(r)
    keys = [("R2 · pre-smoothing sigma (µm)", "sigma", "enhance.gaussian σ  (µm)"),
            ("R2 · radius band as a multiple of the label-derived r_eq", "f_lo",
             "detect.spots  min_radius / r_eq"),
            ("R1 · EDT-watershed min_distance", "fd",
             "analysis.segment  min_distance / r_eq")]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9))
    for ax, (st, axis, xl) in zip(axes, keys):
        rs = sorted(stages.get(st, []), key=lambda r: r["cfg"][axis])
        if not rs:
            continue
        x = [r["cfg"][axis] for r in rs]
        ax.plot(x, [r["pooled"] * 100 for r in rs], "o-", color="#4dd2ff", ms=4, lw=1.5,
                label="exact")
        und = [np.average([v["under"] for v in r["per"].values()],
                          weights=[v["n"] for v in r["per"].values()]) * 100 for r in rs]
        ovr = [np.average([v["over"] for v in r["per"].values()],
                          weights=[v["n"] for v in r["per"].values()]) * 100 for r in rs]
        ax.plot(x, und, "s--", color="#ffd166", ms=3, lw=1, label="under-split")
        ax.plot(x, ovr, "^--", color="#ff6b6b", ms=3, lw=1, label="over-split")
        b = max(rs, key=lambda r: r["pooled"])
        ax.axvline(b["cfg"][axis], color="#8fe3ff", lw=0.9, alpha=0.5)
        ax.set_xlabel(xl, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.12)
        ax.set_title(f"best {b['cfg'][axis]:g} → {b['pooled']:.1%}", fontsize=7.5)
    axes[0].set_ylabel("% of labelled blobs")
    axes[0].legend(fontsize=6.5, framealpha=0.15, loc="center left")
    ax = axes[3]
    if dg:
        w = np.array([r["n"] for r in dg], float)
        eq = np.average([r["seeds_eq"] for r in dg], weights=w)
        gt = np.average([r["seeds_gt"] for r in dg], weights=w)
        lt = np.average([r["seeds_lt"] for r in dg], weights=w)
        n_eq = sum(r["n_seeds_eq"] for r in dg)
        n_gt = sum(r["n_seeds_gt"] for r in dg)
        oe = sum(r["over_given_seeds_eq"] * r["n_seeds_eq"] for r in dg) / max(n_eq, 1)
        og = sum(r["over_given_seeds_gt"] * r["n_seeds_gt"] for r in dg) / max(n_gt, 1)
        ax.bar([0, 1, 2], [eq * 100, gt * 100, lt * 100], width=0.6,
               color=["#5ad19a", "#ff6b6b", "#ffd166"], alpha=0.9)
        for i, v in enumerate([eq, gt, lt]):
            ax.text(i, v * 100 + 1.5, f"{v:.1%}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["seeds = label", "too many", "too few"], fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_title(f"the score IS the seeder's accuracy\n"
                     f"P(over-split | seed count right) = {oe:.1%}   "
                     f"| wrong) = {og:.1%}", fontsize=7.5)
        ax.grid(alpha=0.12, axis="y")
    fig.suptitle("Every sweep bracketed its optimum, and the diagnosis says the partition is "
                 "not the problem: given the right number of seeds the nearest-seed diagram "
                 "returns the right number of pieces 99.6% of the time.", fontsize=8.5,
                 y=1.02)
    fig.tight_layout()
    p = os.path.join(IMG, "fig_sweep.png")
    fig.savefig(p, bbox_inches="tight", **FIG)
    plt.close(fig)
    print(f"  {p}")


# ══ figure 4: phases, phantom, cells ══════════════════════════════════════════

def fig_results():
    ph = _load("phases.json") or []
    pt = _load("phantom.json") or []
    fx = _load("phantom_fix.json") or []
    cl = _load("cells.json") or []
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))

    ax = axes[0]
    conds = []
    for r in ph:
        if r["cond"] not in conds:
            conds.append(r["cond"])
    for c in conds:
        F = next(r for r in ph if r["cond"] == c and r["phase"] == "F")["filtered"]
        I = next(r for r in ph if r["cond"] == c and r["phase"] == "I")["filtered"]
        void = 1.0 - np.array(F) - np.array(I)
        pure = c in ("M01", "M08", "M15")
        ax.plot(np.arange(len(void)) * 47.29 / 60, void, "o-" if pure else "s--", ms=3,
                lw=1.3, label=f"{c}{' (pure F)' if pure else ''}")
    ax.set_xlabel("hours"), ax.set_ylabel("void fraction  1 − F − I")
    ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=6.2, framealpha=0.15, ncol=2)
    ax.grid(alpha=0.12)
    ax.set_title("void per phase over 11.3 h\n(fixed-DN inert cut)", fontsize=8)

    ax = axes[1]
    x = np.arange(len(pt))
    ax.bar(x - 0.19, [r["i_inside_f"] * 100 for r in pt], 0.36, color="#ff6b6b",
           label="inert mask inside F", alpha=0.9)
    ax.bar(x + 0.19, [r["random_expectation"] * 100 for r in pt], 0.36, color="#3a3f47",
           label="expected if unrelated", alpha=0.9)
    ax.set_xticks(x), ax.set_xticklabels([r["cond"] for r in pt], fontsize=7.5)
    ax.set_ylabel("% of inert voxels")
    ax.legend(fontsize=6.5, framealpha=0.15)
    ax.grid(alpha=0.12, axis="y")
    ax.set_title("the phantom phase IS the F granules\n(pure-F: 84-96% vs 24-40% expected)",
                 fontsize=8)

    ax = axes[2]
    if fx:
        dn = [r["threshold_dn"] for r in fx]
        pk = [np.mean(list(r["phantom"].values())) * 100 for r in fx]
        kp = [np.mean(list(r["kept"].values())) * 100 for r in fx]
        ax.plot(dn, pk, "o-", color="#ff6b6b", ms=4, label="phantom @ pure-F (want 0)")
        ax.plot(dn, kp, "s-", color="#5ad19a", ms=4, label="real inert kept @ mixed")
        ax.axvline(B.NB_FIXED_DN, color="#4dd2ff", lw=1.2)
        ax.text(B.NB_FIXED_DN + 12, 200, f"{B.NB_FIXED_DN:.0f} DN", color="#4dd2ff",
                fontsize=7.5)
        ax.set_xlabel("analysis.segment level=fixed  threshold (raw DN)")
        ax.set_ylabel("%")
        ax.legend(fontsize=6.5, framealpha=0.15)
        ax.grid(alpha=0.12)
        ax.set_title("an ABSOLUTE cut fixes it; a size filter cannot\n"
                     "calibrated on the design's own zero controls", fontsize=8)

    ax = axes[3]
    if cl:
        x = np.arange(len(cl))
        col = ["#5ad19a" if r["i_size"] is None else "#ff6b6b" for r in cl]
        ax.bar(x, [r["nb_above_void"] for r in cl], 0.6, color=col, alpha=0.9)
        for i, r in enumerate(cl):
            ax.text(i, r["nb_above_void"] + 25, f"{r['nb_above_void']:.0f}", ha="center",
                    fontsize=7.5, color=INK)
        ax.set_xticks(x), ax.set_xticklabels([r["cond"] for r in cl], fontsize=7.5)
        ax.set_ylabel("cell Nile-Blue above void (DN)")
        ax.grid(alpha=0.12, axis="y")
        ax.set_title("dye leaching, via analysis.measure `raw`\n"
                     "green = pure F, no dye to take up", fontsize=8)
    fig.tight_layout()
    p = os.path.join(IMG, "fig_results.png")
    fig.savefig(p, **FIG)
    plt.close(fig)
    print(f"  {p}")


def fig_surface():
    """The boundary question: geometric midpoint vs the intensity saddle, seeds held fixed."""
    sf = _load("surface.json") or []
    if not sf:
        return
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    _by, a_single = B.load_labels()
    cond, m = "M08", 7
    r_eq = float(np.sqrt(a_single[cond] / np.pi))
    chans = (B.C_GFP, B.C_F, B.C_I)
    px = float(B.calibration()["pixel_size_um"])
    n, e, s = B.g_seeded_voronoi(ci=B.ch_index(chans, B.C_F),
                                min_area_um2=0.30 * a_single[cond],
                                sigma_um=B.R2_BEST["sigma"], seeder="spots",
                                r_lo=B.R2_BEST["f_lo"] * r_eq, r_hi=B.R2_BEST["f_hi"] * r_eq,
                                spot_threshold=B.R2_BEST["thr"])
    ds, env = B.source(m, (0,), B.Z_PLANE, chans)
    o, _eg = B.run(B.build(n, e), s, ds, env)
    blobs = B.raster(o, "blobs")[0, 0, 0, 0]
    vor = B.raster(o, "grains")[0, 0, 0, 0].astype(np.int64)
    tb, pts = B.table(o, D.LABEL, "grains"), B.table(o, D.POINT, "seeds")
    raw = B.raw_plane(m, 0, B.Z_PLANE, B.C_F)
    sm = ndi.gaussian_filter(raw, B.R2_BEST["sigma"] / px)
    markers = np.zeros_like(vor)
    sy = np.clip(np.rint(pts["y"]).astype(int), 0, vor.shape[0] - 1)
    sx = np.clip(np.rint(pts["x"]).astype(int), 0, vor.shape[1] - 1)
    keep = blobs[sy, sx] > 0
    markers[sy[keep], sx[keep]] = np.arange(1, int(keep.sum()) + 1)
    flood = watershed(-sm, markers=markers, mask=blobs > 0).astype(np.int64)
    seed_row = {int(v): i + 1 for i, v in enumerate(pts["id"][keep])}
    lut = np.zeros(int(vor.max()) + 1, dtype=np.int64)
    for gid, pid in zip(tb["id"].astype(int), tb["point_id"].astype(int)):
        lut[gid] = seed_row.get(pid, 0)
    vor_s = lut[vor]

    def bnd(lab):
        b = np.zeros(lab.shape, bool)
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            r = np.roll(lab, sh, axis=ax)
            b |= (lab > 0) & (r > 0) & (lab != r)
        return b

    W = int(round(340.0 / px))
    y0, x0 = 500, 360
    cr = (slice(y0, y0 + W), slice(x0, x0 + W))
    fig, A = plt.subplots(1, 4, figsize=(13.0, 3.35))
    a = A[0]
    a.imshow(raw[cr], cmap="gray", vmin=np.percentile(raw, 2),
             vmax=np.percentile(raw, 99.5), interpolation="nearest")
    yv, xv = np.nonzero(bnd(vor_s)[cr])
    yf, xf = np.nonzero(bnd(flood)[cr])
    a.plot(xv, yv, ".", ms=1.1, color="#ff6b6b", label="voronoi — nearest seed")
    a.plot(xf, yf, ".", ms=1.1, color="#4dd2ff", label="flood — intensity saddle")
    a.legend(fontsize=6.4, framealpha=0.25, markerscale=6, loc="upper right")
    a.set_xticks([]), a.set_yticks([])
    a.set_title("the same seeds, two divides\n340 µm crop, M08", fontsize=8)

    a = A[1]
    x = np.arange(len(sf))
    a.bar(x - 0.19, [r["boundary_dim_voronoi"] for r in sf], 0.36, color="#ff6b6b",
          label="voronoi", alpha=0.9)
    a.bar(x + 0.19, [r["boundary_dim_flood"] for r in sf], 0.36, color="#4dd2ff",
          label="intensity flood", alpha=0.9)
    a.axhline(1.0, color=MUTE, ls=":", lw=1)
    a.text(len(sf) - 0.5, 1.008, "= granule interior brightness", fontsize=6.6,
           color=MUTE, ha="right")
    a.set_xticks(x), a.set_xticklabels([r["cond"] for r in sf], fontsize=7.5)
    a.set_ylim(0.80, 1.06)
    a.set_ylabel("intensity ON the divide ÷ interior")
    a.legend(fontsize=6.6, framealpha=0.15, loc="lower left")
    a.grid(alpha=0.12, axis="y")
    a.set_title("the geometric divide is NOT in a seam\nit sits at body brightness",
                fontsize=8)

    a = A[2]
    a.bar(x - 0.19, [r["solidity_voronoi"] for r in sf], 0.36, color="#ff6b6b",
          label="voronoi", alpha=0.9)
    a.bar(x + 0.19, [r["solidity_flood"] for r in sf], 0.36, color="#4dd2ff",
          label="intensity flood", alpha=0.9)
    a.set_xticks(x), a.set_xticklabels([r["cond"] for r in sf], fontsize=7.5)
    a.set_ylim(0.75, 0.92)
    a.set_ylabel("median solidity of the objects")
    a.legend(fontsize=6.6, framealpha=0.15, loc="lower left")
    a.grid(alpha=0.12, axis="y")
    a.set_title("granules are convex polygons,\nso higher is more correct", fontsize=8)

    a = A[3]
    a.bar(x, [r["shift_over_r_eq"] * 100 for r in sf], 0.5, color="#ffd166", alpha=0.9)
    for i, r in enumerate(sf):
        a.text(i, r["shift_over_r_eq"] * 100 + 1.0,
               f"{r['mean_shift_um']:.1f} µm", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(x), a.set_xticklabels([r["cond"] for r in sf], fontsize=7.5)
    a.set_ylabel("mean boundary shift (% of granule radius)")
    a.set_ylim(0, 55)
    a.grid(alpha=0.12, axis="y")
    a.set_title("how far the divide moves\n6.6% of foreground reassigned", fontsize=8)

    fig.suptitle("Checking the granules against the energy at their own surface — the labels "
                 "cannot do this, because they count granules and never say where the boundary "
                 "is.", fontsize=8.5, y=1.03)
    fig.tight_layout()
    p = os.path.join(IMG, "fig_surface.png")
    fig.savefig(p, bbox_inches="tight", **FIG)
    plt.close(fig)
    print(f"  {p}")


def main() -> int:
    os.makedirs(IMG, exist_ok=True)
    if not os.path.exists(B.ND2):
        print(f"SKIP: sample not found: {B.ND2}")
        return 0
    print("figures:")
    st = fig_stages()
    bn = fig_bench()
    fig_sweep_diag()
    fig_results()
    fig_surface()
    json.dump(dict(stages=st, bench=bn),
              open(os.path.join(CACHE, "report.json"), "w", encoding="utf-8"), indent=1)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
