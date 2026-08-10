"""Compute-only: threshold-free separability + hysteresis segmentation. Prints, no logging."""
from __future__ import annotations
import sys, os
sys.path.insert(0, ".claude/skills/evidence-log")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import sato, gaussian, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label
from sklearn.metrics import roc_auc_score

IMG = "CodeLog/live/zsdeconv-fiber/img"
RAW = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif"
DEC = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_Deconv.ome.tif"
PX_RAW = 0.107136166827053; PX_DEC = PX_RAW / 2.0


def norm01(a, lo=0.1, hi=99.9):
    p0, p1 = np.percentile(a, [lo, hi]); return np.clip((a - p0) / max(1e-12, p1 - p0), 0, 1)


def block2(a):
    h, w = a.shape
    return a[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean((1, 3))


def ridge(img, px):
    sig_um = np.array([0.08, 0.12, 0.18, 0.26])
    r = sato(norm01(img, 1, 99.8), sigmas=list(np.maximum(1.0, sig_um / px)),
             black_ridges=False)
    return r / (r.max() + 1e-12)


ALL = []
for T in (10, 30, 54, 80, 100):
    raw = tifffile.imread(RAW, key=T).astype(np.float32)
    dec = tifffile.imread(DEC, key=T).astype(np.float32)
    r_raw, r_dec = ridge(raw, PX_RAW), ridge(dec, PX_DEC)
    r_dec_ds = block2(r_dec)

    # common, unbiased label source: heavily smoothed RAW (sigma 0.5 um)
    sm = gaussian(norm01(raw, 1, 99.8), sigma=0.5 / PX_RAW)
    hi_t, lo_t = np.percentile(sm, [88, 45])
    fiber = sm >= hi_t
    bg = sm <= lo_t
    sel = fiber | bg
    y = fiber[sel].ravel()
    auc_raw = roc_auc_score(y, r_raw[sel].ravel())
    auc_dec = roc_auc_score(y, r_dec_ds[sel].ravel())

    # false-ridge density: ridge response inside genuinely dark background
    fr_raw = float(r_raw[bg].mean() / (r_raw[fiber].mean() + 1e-12))
    fr_dec = float(r_dec_ds[bg].mean() / (r_dec_ds[fiber].mean() + 1e-12))

    # hysteresis segmentation at matched physical scale
    def seg(r, px):
        hi = np.percentile(r, 97); lo = np.percentile(r, 88)
        m = apply_hysteresis_threshold(r, lo, hi)
        m = remove_small_objects(m, int(0.3 / (px * px)))
        sk = skeletonize(m)
        nb = np.zeros_like(sk, np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    nb += np.roll(np.roll(sk, dy, 0), dx, 1).astype(np.uint8)
        ends = int(((nb == 1) & sk).sum()); L = sk.sum() * px
        return (label(m).max(), L, 100 * ends / max(L, 1e-9),
                m.sum() * px * px / max(L, 1e-9), m)

    o_r, L_r, e_r, w_r, m_r = seg(r_raw, PX_RAW)
    o_d, L_d, e_d, w_d, m_d = seg(r_dec, PX_DEC)
    ALL.append(dict(T=T, auc_raw=auc_raw, auc_dec=auc_dec, fr_raw=fr_raw, fr_dec=fr_dec,
                    o_r=o_r, o_d=o_d, e_r=e_r, e_d=e_d, w_r=w_r, w_d=w_d,
                    L_r=L_r, L_d=L_d))
    print(f"T={T:3d} AUC raw {auc_raw:.4f} dec {auc_dec:.4f} | bg/fiber ridge raw {fr_raw:.4f} "
          f"dec {fr_dec:.4f} | objs {o_r}/{o_d} ends/100um {e_r:.1f}/{e_d:.1f} "
          f"width {w_r:.3f}/{w_d:.3f}")
    if T == 54:
        keep = (raw, dec, r_raw, r_dec, m_r, m_d, fiber, bg)

print()
for k in ("auc", "fr", "e", "w", "o"):
    a = np.mean([x[f"{k}_raw" if k in ("auc", "fr") else f"{k}_r"] for x in ALL])
    b = np.mean([x[f"{k}_dec" if k in ("auc", "fr") else f"{k}_d"] for x in ALL])
    print(f"MEAN {k}: raw {a:.4f}  deconv {b:.4f}  delta {b-a:+.4f}")

raw, dec, r_raw, r_dec, m_r, m_d, fiber, bg = keep
y0, x0, S = 150, 150, 128
fig, ax = plt.subplots(2, 3, figsize=(15, 10.2))
ax[0,0].imshow(norm01(raw[y0:y0+S,x0:x0+S],1,99.8), cmap="gray"); ax[0,0].set_title("RAW")
ax[0,1].imshow(r_raw[y0:y0+S,x0:x0+S], cmap="magma", vmin=0, vmax=.35)
ax[0,1].set_title("RAW ridge response\n(note the mesh of noise ridges in dark areas)")
ax[0,2].imshow(m_r[y0:y0+S,x0:x0+S], cmap="gray"); ax[0,2].set_title("RAW hysteresis mask")
ax[1,0].imshow(norm01(dec[2*y0:2*y0+2*S,2*x0:2*x0+2*S],1,99.8), cmap="gray"); ax[1,0].set_title("DECONV")
ax[1,1].imshow(r_dec[2*y0:2*y0+2*S,2*x0:2*x0+2*S], cmap="magma", vmin=0, vmax=.35)
ax[1,1].set_title("DECONV ridge response\n(dark areas are clean)")
ax[1,2].imshow(m_d[2*y0:2*y0+2*S,2*x0:2*x0+2*S], cmap="gray"); ax[1,2].set_title("DECONV hysteresis mask")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Corrected: hysteresis segmentation, matched physical scale, 13.7 um field", fontsize=13)
fig.tight_layout(); fig.savefig(f"{IMG}/06_seg_corrected.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# summary bar figure
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
Ts = [x["T"] for x in ALL]; xx = np.arange(len(Ts)); w = .38
ax[0].bar(xx-w/2, [x["auc_raw"] for x in ALL], w, label="RAW")
ax[0].bar(xx+w/2, [x["auc_dec"] for x in ALL], w, label="DECONV")
ax[0].set_ylim(.8, 1.0); ax[0].set_title("ridge-response AUC\n(fiber vs background, threshold-free)")
ax[1].bar(xx-w/2, [x["fr_raw"] for x in ALL], w, label="RAW")
ax[1].bar(xx+w/2, [x["fr_dec"] for x in ALL], w, label="DECONV")
ax[1].set_title("false-ridge ratio\nmean ridge in background / on fiber (lower=better)")
ax[2].bar(xx-w/2, [x["e_r"] for x in ALL], w, label="RAW")
ax[2].bar(xx+w/2, [x["e_d"] for x in ALL], w, label="DECONV")
ax[2].set_title("free ends per 100 um\n(lower = less fragmented)")
for a in ax:
    a.set_xticks(xx); a.set_xticklabels([f"t={t}" for t in Ts]); a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{IMG}/07_metrics.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("figures written")
