"""All four methods on Fibro_test.tif, including a PROPERLY TRAINED ZS-DeconvNet
(models/Fibroblast_10x_c0: 33000 iterations, upsample off, 98 train units)."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import gaussian, sobel
from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy
from scipy.ndimage import map_coordinates

from nodegraph.kernels import zsdeconvnet as zsk
from nodegraph.catalog._shared.psf import diffraction_sigmas

IMG = "CodeLog/live/zsdeconv-fiber/img"
SRC = r"C:\Users\McGheeLab - Analysis\Downloads\Fibro_test.tif"
W = "models/Fibroblast_10x_c0.weights.h5"
PX = 1.7182777601481225

raw = tifffile.imread(SRC).astype(np.float32)


def n01(x, lo=0.5, hi=99.7):
    p0, p1 = np.percentile(x, [lo, hi]); return np.clip((x - p0) / max(1e-12, p1 - p0), 0, 1)


rn = n01(raw)
sig = float(estimate_sigma(rn))
print("estimated sigma", sig)

psf = zsk.gaussian_psf_from_sigmas(diffraction_sigmas(525.0, 0.45, PX, 0.0, False))
print("psf", psf.shape, "sum2", float((psf.astype(np.float64) ** 2).sum()))

out, times = {}, {}
out["raw"] = rn; times["raw"] = 0.0

t0 = time.time()
nlm = denoise_nl_means(rn, h=1.15 * sig, sigma=sig, fast_mode=True, patch_size=5,
                       patch_distance=6)
times["NL-means"] = time.time() - t0; out["NL-means"] = nlm

t0 = time.time()
out["NL-means + RL(10)"] = richardson_lucy(np.clip(nlm, 1e-6, 1), psf, num_iter=10, clip=False)
times["NL-means + RL(10)"] = time.time() - t0

t0 = time.time()
den, dec = zsk.infer_2d(raw, arch="unet2d", weights_path=W, tile=256, overlap=20,
                        upsample=False, insert_xy=16, norm_low=0.0)
times["ZS-DeconvNet 33k it"] = time.time() - t0
out["ZS-DeconvNet 33k it"] = np.clip(dec, 0, 1)
out["ZS-DeconvNet denoised head"] = np.clip(den, 0, 1)
times["ZS-DeconvNet denoised head"] = times["ZS-DeconvNet 33k it"]
print("zs out", dec.shape, dec.min(), dec.max())

# EVERY method is put on the SAME display/measurement scale before anything is compared.
# Without this the ZS output (already min-max normalized by infer_2d, so its 4095-count
# dust specks crush everything else) is measured on a different scale from the raw, and
# `bgnoise` becomes a comparison of normalizations rather than of noise.
out = {k: n01(v) for k, v in out.items()}

for k, v in times.items():
    print(f"  {k:<28} {v:8.2f} s")

# ---------------------------------------------------------------- instruments
# 1. background noise sigma, in genuinely dark regions (from the raw)
sm = gaussian(rn, sigma=3)
bg = sm <= np.percentile(sm, 30)
cell = sm >= np.percentile(sm, 80)

# 2. edge sharpness: 10-90% rise distance across strong cell boundaries, in um
edge = sobel(gaussian(rn, 2))
ys, xs = np.where(edge > np.percentile(edge, 99.7))
rs = np.random.default_rng(0).choice(len(ys), size=min(400, len(ys)), replace=False)
gy, gx = np.gradient(gaussian(rn, 2))


def rise_um(img, n=41, half=10.0):
    """Mean 10-90% transition distance (um) along the local gradient at strong edges."""
    ws = []
    for i in rs:
        y, x = ys[i], xs[i]
        d = np.hypot(gy[y, x], gx[y, x])
        if d < 1e-6:
            continue
        uy, ux = gy[y, x] / d, gx[y, x] / d
        tt = np.linspace(-half, half, n)
        prof = map_coordinates(img, [y + tt * uy, x + tt * ux], order=1, mode="nearest")
        a, b = prof[0], prof[-1]
        if abs(b - a) < 0.05:
            continue
        lo, hi = a + 0.1 * (b - a), a + 0.9 * (b - a)
        p = prof if b > a else prof[::-1]
        lo, hi = (lo, hi) if b > a else (hi, lo)
        i1 = np.argmax(p >= lo); i2 = np.argmax(p >= hi)
        if i2 > i1:
            ws.append((tt[i2] - tt[i1]) * PX)
    return float(np.mean(ws)), len(ws)


rows = []
for k in ["raw", "NL-means", "NL-means + RL(10)", "ZS-DeconvNet denoised head",
          "ZS-DeconvNet 33k it"]:
    v = out[k]
    nsd = float(v[bg].std())
    cnr = float((v[cell].mean() - v[bg].mean()) / (nsd + 1e-12))
    w, nedge = rise_um(v)
    rows.append([k, f"{nsd:.5f}", f"{cnr:.1f}", f"{w:.2f}", f"{times[k]:.2f}"])
    print(f"{k:<28} bgnoise {nsd:.5f}  CNR {cnr:7.1f}  10-90% rise {w:5.2f} um  "
          f"({nedge} edges)  {times[k]:.2f}s")

np.save("CodeLog/live/zsdeconv-fiber/fibro_rows.npy",
        np.array(rows, dtype=object), allow_pickle=True)

# ---------------------------------------------------------------- figures
keys = ["raw", "NL-means", "NL-means + RL(10)", "ZS-DeconvNet denoised head",
        "ZS-DeconvNet 33k it"]
y0, x0, S = 440, 440, 128
fig, ax = plt.subplots(3, 5, figsize=(21, 12.6))
for j, k in enumerate(keys):
    v = out[k]
    ax[0, j].imshow(v, cmap="gray"); ax[0, j].set_title(f"{k}\n{times[k]:.1f} s", fontsize=10)
    ax[0, j].add_patch(plt.Rectangle((380, 380), 256, 256, ec="yellow", fc="none", lw=1.1))
    ax[1, j].imshow(v[380:636, 380:636], cmap="gray", interpolation="nearest")
    ax[1, j].set_title("440 um", fontsize=9)
    ax[2, j].imshow(v[y0:y0+S, x0:x0+S], cmap="gray", interpolation="nearest")
    ax[2, j].set_title("220 um", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Fibro_test.tif (1.7183 um/px, 1760 um field) - four methods, same display scale",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{IMG}/11_fibro_four.png", dpi=108, bbox_inches="tight")
plt.close(fig)

# radial PSD
def rpsd(a, px):
    a = (a - a.mean()) / (a.std() + 1e-12)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(yy - h / 2, xx - w / 2); nb = min(h, w) // 2
    pr = np.bincount(rr.astype(int).ravel(), F.ravel(), minlength=nb + 1)[:nb]
    ct = np.bincount(rr.astype(int).ravel(), minlength=nb + 1)[:nb]
    return np.arange(nb) / nb * (1 / (2 * px)), pr / np.maximum(ct, 1)


fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for k in keys:
    f, p = rpsd(out[k], PX)
    for a in ax:
        a.semilogy(f, p / p[1], lw=1.6, label=k)
ax[0].set_xlim(0, 0.29); ax[1].set_xlim(0, 0.10)
for a in ax:
    a.set_xlabel("spatial frequency (1/um)"); a.set_ylabel("radial power (norm at low f)")
    a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_title("full range to Nyquist (0.291/um)"); ax[1].set_title("structure band")
fig.suptitle("Fibro_test: where each method puts its energy", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/12_fibro_psd.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("figures written")
