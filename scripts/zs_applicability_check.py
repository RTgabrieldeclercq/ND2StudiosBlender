"""Should ZS-DeconvNet be used on THIS dataset, and with what settings?

Computes the three quantities that predicted the outcome on every dataset tested so far
(see CodeLog/live/zsdeconv-fiber, entry e016) and prints a verdict.

    .\.venv\Scripts\python.exe -u -B scripts\zs_applicability_check.py <image> <um_per_px> <em_nm> <NA>

Accepts any 2D image tifffile can read (or pass an .nd2 with --nd2 P Z C).
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import numpy as np


def radial_psd(a, px):
    a = np.asarray(a, np.float32)
    a = (a - a.mean()) / (a.std() + 1e-12)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    h, w = a.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(yy - h / 2, xx - w / 2)
    nb = min(h, w) // 2
    pr = np.bincount(rr.astype(int).ravel(), F.ravel(), minlength=nb + 1)[:nb]
    ct = np.bincount(rr.astype(int).ravel(), minlength=nb + 1)[:nb]
    return np.arange(nb) / nb * (1 / (2 * px)), pr / np.maximum(ct, 1)


def applicability(img, px_um, em_nm, na):
    """The three gates. Returns a dict of numbers plus a verdict string."""
    f, p = radial_psd(img, px_um)
    nyq = 1.0 / (2 * px_um)
    cutoff = 2.0 * na / (em_nm / 1000.0)
    sigma_px = 0.21 * em_nm / na / (px_um * 1000.0)
    floor = float(np.median(p[f > 0.82 * nyq]))
    cross = float(f[np.argmax(((p - floor) <= 0) & (f > 0.15 * nyq))])
    noise_frac = float((f > cross).mean())

    # gate 1 - is there optical blur for stage II to remove, at the OUTPUT grid?
    if sigma_px < 0.7:
        g1 = ("NO deconvolution possible: sigma is below the node's max(0.5, sigma) clamp, "
              "so the PSF it trains against is the clamp, not your optics. Stage II can "
              "only smooth. Use output=denoised, or use a plain denoiser.")
    elif sigma_px < 1.5:
        g1 = ("WEAK deconvolution: expect denoising benefit, no resolution gain, and "
              "roughly 10% object-width inflation.")
    else:
        g1 = "REAL deconvolution possible - this is the regime the paper operates in."

    # gate 2 - how big is the denoising prize?
    if noise_frac < 0.15:
        g2 = "SMALL denoising prize: the data is already clean. The node adds blur for little."
    elif noise_frac < 0.35:
        g2 = "MODERATE denoising prize: expect real continuity/false-ridge gains."
    else:
        g2 = "LARGE denoising prize: this is where the node earns its compute."

    # gate 3 - is upsampling worth it?
    ratio = cross / nyq
    up = ratio >= 0.9
    g3 = (f"upsample=True  (signal reaches {ratio:.0%} of Nyquist - sampling-limited)"
          if up else
          f"upsample=False (signal dies at {ratio:.0%} of Nyquist - NOISE-limited, so a "
          f"finer grid would carry nothing)")

    return dict(sigma_px=sigma_px, nyq=nyq, cutoff=cutoff, cross=cross,
                noise_frac=noise_frac, cross_over_nyq=ratio, upsample=up,
                gate1=g1, gate2=g2, gate3=g3)


def report(name, img, px_um, em_nm, na):
    r = applicability(img, px_um, em_nm, na)
    print(f"\n=== {name}")
    print(f"  pixel {px_um:.4f} um | em {em_nm:.0f} nm | NA {na}")
    print(f"  PSF sigma          {r['sigma_px']:.2f} px")
    print(f"  Nyquist            {r['nyq']:.2f} /um")
    print(f"  OTF cutoff         {r['cutoff']:.2f} /um")
    print(f"  signal = noise at  {r['cross']:.2f} /um  ({r['cross_over_nyq']:.0%} of Nyquist)")
    print(f"  noise-dominated    {r['noise_frac']:.0%} of the sampled band")
    print(f"  [1] {r['gate1']}")
    print(f"  [2] {r['gate2']}")
    print(f"  [3] {r['gate3']}")
    print(f"  -> settings: upsample={r['upsample']}, iterations=20000, hess_weight=0.005, "
          f"tile={'128' if r['upsample'] else '256'}, "
          f"output={'denoised' if r['sigma_px'] < 0.7 else 'deconvolved'}")
    return r


if __name__ == "__main__":
    import tifffile
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    path, px, em, na = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    img = tifffile.imread(path)
    while img.ndim > 2:
        img = img[img.shape[0] // 2]
    report(path, img, px, em, na)
