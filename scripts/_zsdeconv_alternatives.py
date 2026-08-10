"""Is ZS-DeconvNet worth 4.2 h? Score it against cheap classical denoisers on the same metrics."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import sato, gaussian, apply_hysteresis_threshold
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma, richardson_lucy
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label
from sklearn.metrics import roc_auc_score

IMG = "CodeLog/live/zsdeconv-fiber/img"
RAW = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif"
DEC = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_Deconv.ome.tif"
PX = 0.107136166827053


def norm01(a, lo=0.1, hi=99.9):
    p0, p1 = np.percentile(a, [lo, hi]); return np.clip((a - p0) / max(1e-12, p1 - p0), 0, 1)


def block2(a):
    h, w = a.shape
    return a[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean((1, 3))


def ridge(img, px):
    sig_um = np.array([0.08, 0.12, 0.18, 0.26])
    r = sato(img, sigmas=list(np.maximum(1.0, sig_um / px)), black_ridges=False)
    return r / (r.max() + 1e-12)


def gauss_psf(px):
    s = 0.21 * 0.525 / 1.4 / px
    n = max(3, int(2 * round(3 * s) + 1))
    g = np.exp(-((np.arange(n) - n // 2) ** 2) / (2 * s * s))
    k = np.outer(g, g); return (k / k.sum()).astype(np.float32)


def metrics(r, px, fiber, bg, sel, y):
    auc = roc_auc_score(y, r[sel].ravel())
    fr = float(r[bg].mean() / (r[fiber].mean() + 1e-12))
    hi, lo = np.percentile(r, 97), np.percentile(r, 88)
    m = remove_small_objects(apply_hysteresis_threshold(r, lo, hi), int(0.3 / (px * px)))
    sk = skeletonize(m)
    nb = np.zeros_like(sk, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb += np.roll(np.roll(sk, dy, 0), dx, 1).astype(np.uint8)
    ends = int(((nb == 1) & sk).sum()); L = sk.sum() * px
    return dict(auc=auc, fr=fr, ends=100 * ends / max(L, 1e-9),
                width=m.sum() * px * px / max(L, 1e-9), objs=int(label(m).max()))


FRAMES = (10, 30, 54, 80, 100)
acc, times = {}, {}
for T in FRAMES:
    raw = tifffile.imread(RAW, key=T).astype(np.float32)
    dec = tifffile.imread(DEC, key=T).astype(np.float32)
    rn = norm01(raw, 1, 99.8)
    sm = gaussian(rn, sigma=0.5 / PX)
    hi_t, lo_t = np.percentile(sm, [88, 45])
    fiber, bg = sm >= hi_t, sm <= lo_t
    sel = fiber | bg; y = fiber[sel].ravel()

    sig = float(estimate_sigma(rn))
    cands = {}
    t0 = time.time(); cands["raw"] = rn; times.setdefault("raw", []).append(time.time()-t0)
    t0 = time.time(); cands["gaussian 0.5px"] = gaussian(rn, sigma=0.5); times.setdefault("gaussian 0.5px", []).append(time.time()-t0)
    t0 = time.time(); cands["TV chambolle"] = denoise_tv_chambolle(rn, weight=0.02); times.setdefault("TV chambolle", []).append(time.time()-t0)
    t0 = time.time(); cands["NL-means"] = denoise_nl_means(rn, h=1.15*sig, sigma=sig, fast_mode=True, patch_size=5, patch_distance=6); times.setdefault("NL-means", []).append(time.time()-t0)
    t0 = time.time()
    nlm = cands["NL-means"]
    cands["NL-means + RL(10)"] = richardson_lucy(np.clip(nlm, 1e-6, 1), gauss_psf(PX), num_iter=10, clip=False)
    times.setdefault("NL-means + RL(10)", []).append(time.time()-t0)
    cands["ZS-DeconvNet"] = block2(norm01(dec, 1, 99.8))
    times.setdefault("ZS-DeconvNet", []).append(15000.0/114)

    for k, v in cands.items():
        acc.setdefault(k, []).append(metrics(ridge(np.clip(v, 0, 1), PX), PX, fiber, bg, sel, y))
    print("frame", T, "done")

print()
order = ["raw", "gaussian 0.5px", "TV chambolle", "NL-means", "NL-means + RL(10)", "ZS-DeconvNet"]
hdr = f"{'method':<20}{'AUC':>8}{'falseridge':>12}{'ends/100um':>12}{'width um':>10}{'objs':>7}{'s/frame':>10}"
print(hdr); print("-" * len(hdr))
rows = []
for k in order:
    a = {m: float(np.mean([d[m] for d in acc[k]])) for m in ("auc", "fr", "ends", "width", "objs")}
    t = float(np.mean(times[k]))
    print(f"{k:<20}{a['auc']:>8.4f}{a['fr']:>12.4f}{a['ends']:>12.1f}{a['width']:>10.3f}"
          f"{a['objs']:>7.0f}{t:>10.2f}")
    rows.append([k, f"{a['auc']:.4f}", f"{a['fr']:.4f}", f"{a['ends']:.1f}",
                 f"{a['width']:.3f}", f"{a['objs']:.0f}", f"{t:.2f}"])
np.save("CodeLog/live/zsdeconv-fiber/alt_rows.npy", np.array(rows, dtype=object),
        allow_pickle=True)

# figure
fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
xs = np.arange(len(order))
for a, (key, ttl, better) in zip(ax, [
        ("auc", "ridge AUC (fiber vs bg)\nhigher better", 1),
        ("fr", "false-ridge ratio\nlower better", -1),
        ("ends", "free ends / 100 um\nlower better", -1),
        ("width", "mean fiber width (um)\nlower = sharper", -1)]):
    vals = [float(np.mean([d[key] for d in acc[k]])) for k in order]
    cols = ["#64748b"] * len(order); cols[order.index("ZS-DeconvNet")] = "#fb7185"
    cols[order.index("raw")] = "#0ea5e9"
    a.bar(xs, vals, color=cols)
    a.set_xticks(xs); a.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    a.set_title(ttl, fontsize=10); a.grid(alpha=.3, axis="y")
    if key == "auc":
        a.set_ylim(0.80, 0.88)
fig.suptitle("ZS-DeconvNet (red, 132 s/frame) vs classical denoisers (grey, <2 s/frame), "
             "same ridge pipeline, 5 frames", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/08_alternatives.png", dpi=115, bbox_inches="tight")
plt.close(fig)

# visual panel
T = 54
raw = tifffile.imread(RAW, key=T).astype(np.float32); rn = norm01(raw, 1, 99.8)
dec = tifffile.imread(DEC, key=T).astype(np.float32)
sig = float(estimate_sigma(rn))
nlm = denoise_nl_means(rn, h=1.15*sig, sigma=sig, fast_mode=True, patch_size=5, patch_distance=6)
show = {"RAW": rn, "NL-means (1.5 s)": nlm,
        "NL-means + RL(10) (2 s)": richardson_lucy(np.clip(nlm,1e-6,1), gauss_psf(PX), num_iter=10, clip=False),
        "ZS-DeconvNet (132 s)": block2(norm01(dec, 1, 99.8))}
y0, x0, S = 150, 150, 128
fig, ax = plt.subplots(2, 4, figsize=(17, 8.8))
for j, (k, v) in enumerate(show.items()):
    ax[0, j].imshow(np.clip(v, 0, 1)[y0:y0+S, x0:x0+S], cmap="gray"); ax[0, j].set_title(k)
    ax[1, j].imshow(ridge(np.clip(v, 0, 1), PX)[y0:y0+S, x0:x0+S], cmap="magma", vmin=0, vmax=.35)
    ax[1, j].set_title("ridge response")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Same 13.7 um field. Bottom row: how clean is the ridge response the "
             "segmentation actually consumes?", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/09_alt_visual.png", dpi=112, bbox_inches="tight")
plt.close(fig)
print("figures written")
