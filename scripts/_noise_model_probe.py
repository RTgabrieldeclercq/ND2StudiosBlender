"""Two photon-transfer estimators, cross-checked on LAGFP (known-good: gain 4.7,
background 120, beta2 25) then applied to Monolayer Red and the fibroblasts."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nodelab_v2.nd2_compat import import_nd2

IMG = "CodeLog/live/zsdeconv-fiber/img"


def pedestal(img):
    """Camera offset = histogram mode of the darkest fifth of the frame.

    Robust and automatable: in a fluorescence frame the dark background is dominated by
    pedestal + read noise, so its mode is the offset. A low percentile would sit on the
    noise tail and read a few counts low.
    """
    a = np.asarray(img, np.float64).ravel()
    dark = a[a <= np.percentile(a, 20)]
    lo, hi = np.percentile(dark, [1, 99])
    if hi <= lo:
        return float(np.median(dark))
    h, e = np.histogram(dark, bins=max(16, int(hi - lo)), range=(lo, hi))
    return float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))


def pt_adjacent(img, nbins=28):
    """Method A: half the variance of horizontally adjacent differences, binned by mean.
    Valid only where the PSF is sampled at >= ~0.7 px sigma, else real structure leaks in."""
    a = np.asarray(img, np.float64)
    d = a[:, 1:] - a[:, :-1]
    mu = 0.5 * (a[:, 1:] + a[:, :-1])
    lo, hi = np.percentile(a, [2, 96])
    edges = np.linspace(lo, hi, nbins)
    xs, ys = [], []
    for l, h in zip(edges[:-1], edges[1:]):
        m = (mu >= l) & (mu < h)
        if m.sum() < 400:
            continue
        dd = d[m]
        s = 1.4826 * np.median(np.abs(dd - np.median(dd)))
        xs.append(mu[m].mean()); ys.append(0.5 * s * s)
    return np.array(xs), np.array(ys)


def pt_blocks(img, blk=4, pct=10, nbins=28):
    """Method B: LOW percentile of local variance in blk x blk blocks, binned by mean.
    Flat patches dominate a low percentile, so structure contaminates far less. Works at
    any sampling, which is what makes it the safe default."""
    a = np.asarray(img, np.float64)
    h, w = a.shape
    a = a[:h // blk * blk, :w // blk * blk]
    b = a.reshape(h // blk, blk, w // blk, blk).transpose(0, 2, 1, 3).reshape(-1, blk * blk)
    mu, va = b.mean(1), b.var(1, ddof=1)
    lo, hi = np.percentile(a, [2, 96])
    edges = np.linspace(lo, hi, nbins)
    xs, ys = [], []
    for l, h2 in zip(edges[:-1], edges[1:]):
        m = (mu >= l) & (mu < h2)
        if m.sum() < 60:
            continue
        xs.append(mu[m].mean()); ys.append(np.percentile(va[m], pct))
    return np.array(xs), np.array(ys)


def fit(xs, ys):
    A = np.vstack([xs, np.ones_like(xs)]).T
    g, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(g), float(c)


def model(img, name, sigma_px):
    bg = pedestal(img)
    xa, ya = pt_adjacent(img); ga, ca = fit(xa, ya)
    xb, yb = pt_blocks(img);   gb, cb = fit(xb, yb)
    # beta2 = variance left at the pedestal = read noise, floored at 0
    b2a = max(0.0, ca + ga * bg)
    b2b = max(0.0, cb + gb * bg)
    print(f"\n=== {name}   (sigma_px {sigma_px:.2f})")
    print(f"  pedestal (hist mode of darkest 20%) = {bg:.1f} ADU")
    print(f"  A adjacent-diff : gain {ga:6.2f} ADU/e-   beta2 {b2a:7.1f}")
    print(f"  B block-lowpct  : gain {gb:6.2f} ADU/e-   beta2 {b2b:7.1f}")
    print(f"  ratio A/B = {ga/max(gb,1e-9):.2f}   (agreement means structure is not leaking in)")
    return dict(name=name, bg=bg, ga=ga, gb=gb, b2a=b2a, b2b=b2b,
                xa=xa, ya=ya, xb=xb, yb=yb, sigma_px=sigma_px)


R = []
lag = tifffile.imread(
    r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif",
    key=54).astype(np.float32)
R.append(model(lag, "LAGFP actin 60x/1.4 (known: 4.7 / 120 / 25)", 0.74))

nd2 = import_nd2()
with nd2.ND2File(r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\Monolayer_stain.nd2") as f:
    dk = f.to_dask()
    mono = np.asarray(dk[0, 5, 1], dtype=np.float32)
    mono2 = np.asarray(dk[7, 5, 1], dtype=np.float32)
R.append(model(mono, "Monolayer Red 20x/0.75  P0", 0.52))
R.append(model(mono2, "Monolayer Red 20x/0.75  P7", 0.52))

fib = tifffile.imread(r"C:\Users\McGheeLab - Analysis\Downloads\Fibro_test.tif").astype(np.float32)
R.append(model(fib, "Fibroblast 10x/0.45", 0.14))

fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
for a, r in zip(ax, R):
    a.plot(r["xa"], r["ya"], "o-", ms=3, lw=1.2, label=f"A adjacent  g={r['ga']:.2f}")
    a.plot(r["xb"], r["yb"], "s-", ms=3, lw=1.2, label=f"B blocks    g={r['gb']:.2f}")
    a.axvline(r["bg"], color="k", ls=":", lw=1.2, label=f"pedestal {r['bg']:.0f}")
    a.set_title(f"{r['name']}\nsigma_px {r['sigma_px']:.2f}", fontsize=9)
    a.set_xlabel("mean intensity (ADU)"); a.grid(alpha=.3); a.legend(fontsize=7.5)
ax[0].set_ylabel("noise variance (ADU^2)")
fig.suptitle("Photon-transfer: slope = gain (beta1), intercept at the pedestal = beta2. "
             "The two estimators agree only where the PSF is sampled.", fontsize=12)
fig.tight_layout(); fig.savefig(f"{IMG}/17_noise_model.png", dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nfigure written")
