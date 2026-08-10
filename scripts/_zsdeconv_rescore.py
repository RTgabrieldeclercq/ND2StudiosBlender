"""Re-score the 20000-iteration retrain against v1 and the classical methods.

Identical frames, label source, threshold rule and metrics as the earlier run, so the
numbers are directly comparable to log entries e006 and e008.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import sato, gaussian, apply_hysteresis_threshold, sobel
from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label
from sklearn.metrics import roc_auc_score
from scipy.ndimage import map_coordinates

IMG = "CodeLog/live/zsdeconv-fiber/img"
RAW = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif"
V1 = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_Deconv.ome.tif"
V2 = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_Deconv_final.ome.ome.tif"
PX = 0.107136166827053


def n01(x, lo=0.1, hi=99.9):
    p0, p1 = np.percentile(x, [lo, hi]); return np.clip((x - p0) / max(1e-12, p1 - p0), 0, 1)


def block2(a):
    h, w = a.shape
    return a[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean((1, 3))


def ridge(img):
    sig_um = np.array([0.08, 0.12, 0.18, 0.26])
    r = sato(img, sigmas=list(np.maximum(1.0, sig_um / PX)), black_ridges=False)
    return r / (r.max() + 1e-12)


def gauss_psf():
    s = max(0.5, 0.21 * 0.525 / 1.4 / PX)
    n = max(3, int(2 * round(3 * s) + 1))
    g = np.exp(-((np.arange(n) - n // 2) ** 2) / (2 * s * s))
    k = np.outer(g, g); return (k / k.sum()).astype(np.float32)


def seg_metrics(r, fiber, bg, sel, y):
    auc = roc_auc_score(y, r[sel].ravel())
    fr = float(r[bg].mean() / (r[fiber].mean() + 1e-12))
    hi, lo = np.percentile(r, 97), np.percentile(r, 88)
    m = remove_small_objects(apply_hysteresis_threshold(r, lo, hi), int(0.3 / (PX * PX)))
    sk = skeletonize(m)
    nb = np.zeros_like(sk, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb += np.roll(np.roll(sk, dy, 0), dx, 1).astype(np.uint8)
    ends = int(((nb == 1) & sk).sum()); L = sk.sum() * PX
    return dict(auc=auc, fr=fr, ends=100 * ends / max(L, 1e-9),
                width=m.sum() * PX * PX / max(L, 1e-9), objs=int(label(m).max())), m


FRAMES = (10, 30, 54, 80, 100)
ORDER = ["raw", "NL-means", "NL-means + RL(10)",
         "ZS v1 (2k it, upsample)", "ZS v2 (20k it, no upsample)"]
acc, times = {k: [] for k in ORDER}, {k: [] for k in ORDER}
keep = None

for T in FRAMES:
    raw = tifffile.imread(RAW, key=T).astype(np.float32)
    v1 = tifffile.imread(V1, key=T).astype(np.float32)
    v2 = tifffile.imread(V2, key=T).astype(np.float32)
    rn = n01(raw)
    sig = float(estimate_sigma(rn))

    cands, tt = {}, {}
    cands["raw"] = rn; tt["raw"] = 0.0
    t0 = time.time()
    nlm = denoise_nl_means(rn, h=1.15*sig, sigma=sig, fast_mode=True, patch_size=5,
                           patch_distance=6)
    tt["NL-means"] = time.time() - t0; cands["NL-means"] = nlm
    t0 = time.time()
    cands["NL-means + RL(10)"] = richardson_lucy(np.clip(nlm, 1e-6, 1), gauss_psf(),
                                                 num_iter=10, clip=False)
    tt["NL-means + RL(10)"] = time.time() - t0
    cands["ZS v1 (2k it, upsample)"] = block2(v1); tt["ZS v1 (2k it, upsample)"] = 131.6
    cands["ZS v2 (20k it, no upsample)"] = v2
    tt["ZS v2 (20k it, no upsample)"] = 4.3          # measured: tile 256, no upsample

    # ONE common intensity scale for every method before any metric (the e011 lesson)
    cands = {k: n01(np.clip(v, 0, None)) for k, v in cands.items()}

    sm = gaussian(rn, sigma=0.5 / PX)
    hi_t, lo_t = np.percentile(sm, [88, 45])
    fiber, bg = sm >= hi_t, sm <= lo_t
    sel = fiber | bg; y = fiber[sel].ravel()

    masks = {}
    for k in ORDER:
        m, mask = seg_metrics(ridge(cands[k]), fiber, bg, sel, y)
        # scale-invariant sharpness: 10-90% rise across strong edges
        acc[k].append(m); times[k].append(tt[k]); masks[k] = mask
    if T == 54:
        keep = (cands, masks)
    print("frame", T, "done")

print()
hdr = (f"{'method':<30}{'AUC':>8}{'falseridge':>12}{'ends/100um':>12}"
       f"{'width um':>10}{'objs':>7}{'s/frame':>10}")
print(hdr); print("-" * len(hdr))
rows = []
for k in ORDER:
    a = {m: float(np.mean([d[m] for d in acc[k]])) for m in ("auc","fr","ends","width","objs")}
    t = float(np.mean(times[k]))
    print(f"{k:<30}{a['auc']:>8.4f}{a['fr']:>12.4f}{a['ends']:>12.1f}"
          f"{a['width']:>10.3f}{a['objs']:>7.0f}{t:>10.2f}")
    rows.append([k, f"{a['auc']:.4f}", f"{a['fr']:.4f}", f"{a['ends']:.1f}",
                 f"{a['width']:.3f}", f"{a['objs']:.0f}", f"{t:.1f}"])
np.save("CodeLog/live/zsdeconv-fiber/rescore_rows.npy", np.array(rows, dtype=object),
        allow_pickle=True)

# ---------------------------------------------------------------- figures
cands, masks = keep
y0, x0, S = 150, 150, 128
fig, ax = plt.subplots(3, 5, figsize=(20, 12.4))
for j, k in enumerate(ORDER):
    ax[0, j].imshow(cands[k][y0:y0+S, x0:x0+S], cmap="gray")
    ax[0, j].set_title(k, fontsize=10)
    ax[1, j].imshow(ridge(cands[k])[y0:y0+S, x0:x0+S], cmap="magma", vmin=0, vmax=.35)
    ax[1, j].set_title("ridge response", fontsize=9)
    ax[2, j].imshow(masks[k][y0:y0+S, x0:x0+S], cmap="gray")
    ax[2, j].set_title("hysteresis mask", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Frame 54, 13.7 um. Did 20000 iterations without upsampling fix the blur?",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{IMG}/13_rescore.png", dpi=108, bbox_inches="tight")
plt.close(fig)


def rpsd(a):
    a = (a - a.mean()) / (a.std() + 1e-12)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(yy - h/2, xx - w/2); nb = min(h, w)//2
    pr = np.bincount(rr.astype(int).ravel(), F.ravel(), minlength=nb+1)[:nb]
    ct = np.bincount(rr.astype(int).ravel(), minlength=nb+1)[:nb]
    return np.arange(nb)/nb*(1/(2*PX)), pr/np.maximum(ct, 1)


fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for k in ORDER:
    f, p = rpsd(cands[k])
    for a in ax:
        a.semilogy(f, p/p[1], lw=1.6, label=k)
for a in ax:
    a.axvline(2.68, color="g", ls=":", lw=1.2, label="raw signal=noise 2.68/um")
    a.set_xlabel("spatial frequency (1/um)"); a.set_ylabel("radial power (norm at low f)")
    a.grid(alpha=.3)
ax[0].set_xlim(0, 4.67); ax[0].legend(fontsize=7.5); ax[0].set_title("to raw Nyquist")
ax[1].set_xlim(0, 2.0); ax[1].set_title("the band that carries signal")
fig.suptitle("Frame 54: did v2 put power back into the mid frequencies?", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/14_rescore_psd.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("figures written")
