"""The applicability formula: measure, on three real datasets, the quantities that
predicted whether ZS-DeconvNet helped. Then apply it to Monolayer_stain Red."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy
from skimage.filters import gaussian

from nodelab_v2.nd2_compat import import_nd2

IMG = "CodeLog/live/zsdeconv-fiber/img"


def rpsd(a, px):
    a = (a - a.mean()) / (a.std() + 1e-12)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(yy - h / 2, xx - w / 2); nb = min(h, w) // 2
    pr = np.bincount(rr.astype(int).ravel(), F.ravel(), minlength=nb + 1)[:nb]
    ct = np.bincount(rr.astype(int).ravel(), minlength=nb + 1)[:nb]
    return np.arange(nb) / nb * (1 / (2 * px)), pr / np.maximum(ct, 1)


def diagnose(img, px, lam_nm, na, name):
    """The four numbers that decide whether ZS-DeconvNet has anything to do."""
    f, p = rpsd(img.astype(np.float32), px)
    nyq = 1.0 / (2 * px)
    cutoff = 2.0 * na / (lam_nm / 1000.0)          # coherent OTF cutoff, 1/um
    sigma_px = 0.21 * lam_nm / na / (px * 1000.0)  # diffraction sigma in PIXELS
    floor = float(np.median(p[f > 0.82 * nyq]))    # white-noise plateau
    cross = float(f[np.argmax(((p - floor) <= 0) & (f > 0.15 * nyq))])
    # how much of the SAMPLED band is noise-dominated -> the denoising prize
    noise_frac = float((f > cross).mean())
    # is there a band the optics support that the data has not reached?
    recoverable = max(0.0, min(cutoff, nyq) - cross)
    return dict(name=name, px=px, lam=lam_nm, na=na, nyq=nyq, cutoff=cutoff,
                sigma_px=sigma_px, cross=cross, noise_frac=noise_frac,
                recoverable=recoverable, f=f, p=p, floor=floor)


D = []

# 1. LAGFP actin, 60x NA 1.4, 0.1071 um/px, em 525
lag = tifffile.imread(
    r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif",
    key=54).astype(np.float32)
D.append(diagnose(lag, 0.107136166827053, 525.0, 1.4, "LAGFP actin 60x/1.4"))

# 2. Fibroblast 10x, 1.7183 um/px, em 525, NA 0.45
fib = tifffile.imread(r"C:\Users\McGheeLab - Analysis\Downloads\Fibro_test.tif"
                      ).astype(np.float32)
D.append(diagnose(fib, 1.7182777601481225, 525.0, 0.45, "Fibroblast 10x/0.45"))

# 3. Monolayer_stain RED (c1), 20x NA 0.75, 0.32463 um/px, em 600
nd2 = import_nd2()
PXM, LAMM, NAM = 0.324628161961367, 600.0, 0.75
with nd2.ND2File(r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\Monolayer_stain.nd2") as f:
    d = f.to_dask()
    print("monolayer dask", d.shape)
    mono = np.asarray(d[0, 5, 1], dtype=np.float32)      # P0, z5, c1 = Red
print("mono frame", mono.shape, mono.min(), mono.max(), np.median(mono))
D.append(diagnose(mono, PXM, LAMM, NAM, "Monolayer Red 20x/0.75"))

hdr = (f"{'dataset':<24}{'um/px':>8}{'sigma_px':>10}{'Nyq':>8}{'OTFcut':>8}"
       f"{'sig=noise':>11}{'noise band':>12}{'recoverable':>12}")
print(); print(hdr); print("-" * len(hdr))
for d in D:
    print(f"{d['name']:<24}{d['px']:>8.4f}{d['sigma_px']:>10.2f}{d['nyq']:>8.2f}"
          f"{d['cutoff']:>8.2f}{d['cross']:>11.2f}{d['noise_frac']*100:>11.0f}%"
          f"{d['recoverable']:>12.2f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
for a, d in zip(ax, D):
    a.semilogy(d["f"], d["p"] / d["p"][1], lw=1.6, color="#0ea5e9")
    a.axhline(d["floor"] / d["p"][1], color="r", ls="--", lw=1.2, label="noise floor")
    a.axvline(d["cross"], color="g", lw=1.6, label=f"signal=noise {d['cross']:.2f}/um")
    a.axvline(d["nyq"], color="grey", ls=":", lw=1.2, label=f"Nyquist {d['nyq']:.2f}")
    if d["cutoff"] <= d["f"][-1] * 1.6:
        a.axvline(d["cutoff"], color="k", ls=":", lw=1.3, label=f"OTF cutoff {d['cutoff']:.2f}")
    a.axvspan(d["cross"], min(d["cutoff"], d["nyq"]), color="#fbbf24", alpha=.25)
    a.set_title(f"{d['name']}\nPSF sigma = {d['sigma_px']:.2f} px", fontsize=10)
    a.set_xlabel("spatial frequency (1/um)"); a.grid(alpha=.3); a.legend(fontsize=7.5)
    a.set_xlim(0, min(d["cutoff"] * 1.15, d["nyq"] * 1.05))
ax[0].set_ylabel("radial power (norm at low f)")
fig.suptitle("The yellow band is what a deconvolution could recover: optics support it, "
             "the data has not reached it", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/15_applicability.png", dpi=115, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- classical on Monolayer Red
def n01(x, lo=0.5, hi=99.7):
    p0, p1 = np.percentile(x, [lo, hi]); return np.clip((x - p0) / max(1e-12, p1 - p0), 0, 1)


crop = mono[700:1212, 700:1212]
rn = n01(crop)
sg = float(estimate_sigma(rn))
s = max(0.5, 0.21 * LAMM / NAM / (PXM * 1000))
n = max(3, int(2 * round(3 * s) + 1))
g = np.exp(-((np.arange(n) - n // 2) ** 2) / (2 * s * s))
psf = np.outer(g, g); psf = (psf / psf.sum()).astype(np.float32)
print("mono psf", psf.shape, "sigma", s)

t0 = time.time(); nlm = denoise_nl_means(rn, h=1.15*sg, sigma=sg, fast_mode=True,
                                         patch_size=5, patch_distance=6)
t_nlm = time.time() - t0
t0 = time.time(); rl = richardson_lucy(np.clip(nlm, 1e-6, 1), psf, num_iter=10, clip=False)
t_rl = time.time() - t0
outs = {"raw": rn, f"NL-means ({t_nlm:.1f}s)": nlm, f"NL-means + RL(10) ({t_rl:.1f}s)": rl}
outs = {k: n01(v) for k, v in outs.items()}

sm = gaussian(rn, sigma=2)
bg = sm <= np.percentile(sm, 30); cellm = sm >= np.percentile(sm, 85)
for k, v in outs.items():
    nsd = float(v[bg].std())
    print(f"  {k:<28} bgnoise {nsd:.5f}  CNR {(v[cellm].mean()-v[bg].mean())/(nsd+1e-12):7.1f}")

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for j, (k, v) in enumerate(outs.items()):
    ax[0, j].imshow(v, cmap="gray"); ax[0, j].set_title(k, fontsize=10)
    ax[0, j].add_patch(plt.Rectangle((160, 160), 160, 160, ec="yellow", fc="none", lw=1.2))
    ax[1, j].imshow(v[160:320, 160:320], cmap="gray", interpolation="nearest")
    ax[1, j].set_title("52 um zoom", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Monolayer_stain Red (c1), 20x/0.75, 0.3246 um/px - 166 um field", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/16_monolayer.png", dpi=112, bbox_inches="tight")
plt.close(fig)
print("figures written")
