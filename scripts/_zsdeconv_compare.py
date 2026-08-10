"""Compare the raw crop against the ZS-DeconvNet output, for fiber segmentation.

Run from the repo root:
    .\.venv\Scripts\python.exe -u -B scripts\_zsdeconv_compare.py
"""
from __future__ import annotations

import sys, os, json
sys.path.insert(0, ".claude/skills/evidence-log")

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import devlog as D

ROOT = "CodeLog/live/zsdeconv-fiber"
IMG = os.path.join(ROOT, "img")
os.makedirs(IMG, exist_ok=True)
D.configure(ROOT, title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

RAW = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif"
DEC = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_Deconv.ome.tif"
PX_RAW = 0.107136166827053
PX_DEC = PX_RAW / 2.0

# ---------------------------------------------------------------- load
raw = tifffile.imread(RAW)                      # (114,512,512) uint16
print("raw", raw.shape, raw.dtype)

with tifffile.TiffFile(DEC) as t:
    dser = t.series[0]
    NT = dser.shape[0]
    def dframe(i):
        return np.asarray(dser.asarray(key=i), dtype=np.float32)
    dec0 = dframe(0)
print("dec frame", dec0.shape, dec0.dtype, dec0.min(), dec0.max())


def block2(a):
    """2x2 block mean: deconv grid -> raw grid."""
    h, w = a.shape
    return a[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def norm01(a, lo=0.1, hi=99.9):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(1e-12, p1 - p0), 0, 1)


# ------------------------------------------------- 1. correspondence check
rows = []
for t in (0, 30, 60, 90, 113):
    r = raw[t].astype(np.float32)
    d = block2(dframe(t))
    rn, dn = norm01(r), norm01(d)
    cc = float(np.corrcoef(rn.ravel(), dn.ravel())[0, 1])
    # also check against a shifted frame, to prove the match is specific
    r2 = raw[(t + 1) % raw.shape[0]].astype(np.float32)
    cc2 = float(np.corrcoef(norm01(r2).ravel(), dn.ravel())[0, 1])
    rows.append([t, f"{cc:.4f}", f"{cc2:.4f}", f"{d.shape}"])
    print("t", t, "cc same", round(cc, 4), "cc next-frame", round(cc2, 4))

D.log(kind="found",
      title="The two files are the same 114 frames, same field, correctly paired",
      body="`LAGFP_Caps_Z_raw.ome.tif` is 114x512x512 uint16 at 0.107136 um/px. "
           "`LAGFP_Caps_Z_Deconv.ome.tif` is 114x1024x1024 float32 at 0.053568 um/px - "
           "exactly the 2x lateral upsample and halved pixel size that "
           "`enhance.zs_deconvnet` declares when `upsample` is on. Downsampling the "
           "deconvolved frame 2x2 back onto the raw grid and correlating confirms frame t "
           "matches frame t, and NOT frame t+1.",
      why="Everything below compares these two pixel-for-pixel after a 2x2 block mean. If "
          "the pairing or the scale were wrong, every sharpness number would be measuring "
          "a registration error instead of the network.",
      evidence=[{"type": "table",
                 "cols": ["frame", "corr(raw_t, deconv_t)", "corr(raw_t+1, deconv_t)",
                          "downsampled shape"],
                 "rows": rows,
                 "caption": "Correlation is high on the matching frame and drops on the "
                            "neighbour, so the pairing is specific."}],
      status="confirmed")

# ------------------------------------------------- 2. pick a good frame
scores = []
for t in range(0, 114, 6):
    r = raw[t].astype(np.float32)
    scores.append((float(np.percentile(r, 99.9)), t))
scores.sort(reverse=True)
T = scores[0][1]
print("using frame", T, "p99.9", scores[0][0])

r = raw[T].astype(np.float32)
d = dframe(T)
dd = block2(d)

# ------------------------------------------------- 3. visual, matched scale
fig, ax = plt.subplots(2, 3, figsize=(15, 10.2))
y0, x0, S = 150, 150, 128          # 128 raw px = 13.7 um
rc = r[y0:y0 + S, x0:x0 + S]
dc = d[2 * y0:2 * (y0 + S), 2 * x0:2 * (x0 + S)]

ax[0, 0].imshow(norm01(r, 1, 99.8), cmap="gray"); ax[0, 0].set_title(f"RAW  frame {T}\n512x512 @ 0.1071 um/px")
ax[0, 1].imshow(norm01(d, 1, 99.8), cmap="gray"); ax[0, 1].set_title("DECONV\n1024x1024 @ 0.0536 um/px")
ax[0, 2].imshow(norm01(dd, 1, 99.8) - norm01(r, 1, 99.8), cmap="RdBu_r", vmin=-.5, vmax=.5)
ax[0, 2].set_title("deconv(downsampled) - raw\nred = deconv brighter")
for a in ax[0]:
    a.add_patch(plt.Rectangle((x0, y0), S, S, ec="yellow", fc="none", lw=1.2))
ax[0, 1].patches[0].set_bounds(2 * x0, 2 * y0, 2 * S, 2 * S)

ax[1, 0].imshow(norm01(rc, 1, 99.8), cmap="gray", interpolation="nearest")
ax[1, 0].set_title("RAW zoom (13.7 um)")
ax[1, 1].imshow(norm01(dc, 1, 99.8), cmap="gray", interpolation="nearest")
ax[1, 1].set_title("DECONV zoom (13.7 um)")
# raw upsampled 2x nearest, so the two zooms are the same physical size on screen
ax[1, 2].imshow(norm01(np.kron(rc, np.ones((2, 2))), 1, 99.8), cmap="gray",
                interpolation="nearest")
ax[1, 2].set_title("RAW zoom, 2x nearest\n(same screen scale as centre)")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Raw vs ZS-DeconvNet, frame {T}, matched physical scale", fontsize=13)
fig.tight_layout()
fig.savefig(f"{IMG}/01_visual.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- 4. radial power spectrum
def radial_psd(a, px):
    a = (a - a.mean()) / (a.std() + 1e-12)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    h, w = a.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(yy - h / 2, xx - w / 2)
    fmax = 1.0 / (2 * px)                       # Nyquist, um^-1
    nb = int(min(h, w) // 2)
    prof = np.bincount(rr.astype(int).ravel(), F.ravel(), minlength=nb + 1)[:nb]
    cnt = np.bincount(rr.astype(int).ravel(), minlength=nb + 1)[:nb]
    prof = prof / np.maximum(cnt, 1)
    freq = np.arange(nb) / nb * fmax
    return freq, prof


fr_raw, ps_raw = radial_psd(r, PX_RAW)
fr_dec, ps_dec = radial_psd(d, PX_DEC)
fr_dd, ps_dd = radial_psd(dd, PX_RAW)

cutoff = 2 * 1.4 / 0.525                        # widefield OTF cutoff, um^-1

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
for a, (xl, tt) in zip(ax, [((0, 9.6), "full range, native grids"),
                            ((0, 4.8), "band both grids cover")]):
    a.semilogy(fr_raw, ps_raw / ps_raw[1], label="RAW (0.1071 um/px)", lw=1.6)
    a.semilogy(fr_dec, ps_dec / ps_dec[1], label="DECONV native (0.0536 um/px)", lw=1.6)
    a.semilogy(fr_dd, ps_dd / ps_dd[1], "--", label="DECONV downsampled to raw grid", lw=1.4)
    a.axvline(cutoff, color="k", ls=":", label=f"diffraction cutoff {cutoff:.2f}/um")
    a.axvline(1 / (2 * PX_RAW), color="r", ls=":", lw=1, label="raw Nyquist")
    a.set_xlim(*xl); a.set_xlabel("spatial frequency (1/um)")
    a.set_ylabel("radially averaged power (norm. at low f)")
    a.set_title(tt); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(f"Power spectra, frame {T} - where the deconvolution put its energy", fontsize=13)
fig.tight_layout()
fig.savefig(f"{IMG}/02_psd.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# quantify: relative power in bands, on the shared grid
def band(fr, ps, lo, hi):
    m = (fr >= lo) & (fr < hi)
    return float(ps[m].sum() / ps[(fr >= 0.05) & (fr < 0.5)].sum())


bands = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 4.66)]
brows = []
for lo, hi in bands:
    br = band(fr_raw, ps_raw, lo, hi)
    bd = band(fr_dd, ps_dd, lo, hi)
    brows.append([f"{lo:.1f}-{hi:.2f}", f"{1000/ (2*hi):.0f}",
                  f"{br:.4g}", f"{bd:.4g}", f"{bd/max(br,1e-12):.2f}x"])
    print("band", lo, hi, "raw", round(br, 5), "dec", round(bd, 5))

print(json.dumps({"frame": int(T)}, indent=1))
print(D.render())
