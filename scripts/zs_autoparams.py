"""Auto-fill every ZS-DeconvNet socket from a file's own metadata + one measured frame.

    .\.venv\Scripts\python.exe -u -B scripts\zs_autoparams.py <file.nd2> [channel]

Derived in CodeLog/live/zsdeconv-fiber (entries e016, e018). Validated against the two
runs whose settings are known to work:
  * LAGFP_Caps_Z (60x/1.4, 0.1071 um/px): auto -> beta1 4.3, background 115, beta2 18
    against the hand-set 4.7 / 120 / 25 that produced the accepted v2 result.
  * Fibroblast_10x: auto -> output=denoised, upsample=False, matching the configuration
    the shipped 33000-iteration model was trained with.

The three GATES decide the shape of the run; the NOISE MODEL fills the physics.
"""
from __future__ import annotations
import sys, re
sys.path.insert(0, ".")
import numpy as np

# Camera conversion gain in ADU per electron, keyed by the ND2's own camera-mode strings.
# These are camera constants, not image properties - reading them beats fitting them,
# because a photon-transfer fit on an undersampled PSF is contaminated by real structure
# (measured: the two estimators disagree by 1.5x once sigma_px < 0.7).
_CAMERA_GAIN = {
    ("12-bit (CMS)", "Sensitivity"): (4.5, 1.0),      # Kinetix 12-bit CMS: ~0.22 e-/ADU, 1.0 e- rms
    ("16-bit", "HDR"):               (0.25, 1.6),
    ("12-bit (Speed)", "Speed"):     (1.0, 2.3),
}
_DEFAULT_GAIN = (1.0, 1.5)          # photon-counting assumption, the node's own default


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


def photon_transfer(img, nbins=28):
    """Fit noise variance = g*I + c by robust adjacent-pixel differences.

    Returns (g, c). The signal must be smooth at 1 px for this to be pure noise, which
    holds once the PSF sigma is >= ~0.7 px; below that it over-reads and the caller should
    prefer the camera-metadata gain.
    """
    a = np.asarray(img, np.float64)
    d = a[:, 1:] - a[:, :-1]
    mu = 0.5 * (a[:, 1:] + a[:, :-1])
    lo, hi = np.percentile(a, [2, 96])
    xs, ys = [], []
    for l, h in zip(np.linspace(lo, hi, nbins)[:-1], np.linspace(lo, hi, nbins)[1:]):
        m = (mu >= l) & (mu < h)
        if m.sum() < 400:
            continue
        dd = d[m]
        s = 1.4826 * np.median(np.abs(dd - np.median(dd)))
        xs.append(mu[m].mean()); ys.append(0.5 * s * s)
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    g, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(g), float(c)


def noise_model(img, gain_adu_per_e, read_noise_e):
    """(background, beta1, beta2) for the node's `var = beta1*(I-bg) + beta2` model.

    Three unknowns, one fitted line - so the read noise (a camera constant) supplies the
    third constraint and the pedestal falls out of the intercept. Fitting the pedestal from
    a dark-region histogram instead reads ~13 counts high, because even the darkest fifth
    of a confluent frame still carries signal.
    """
    g, c = photon_transfer(img)
    beta1 = float(gain_adu_per_e)
    beta2 = float((read_noise_e * beta1) ** 2)
    background = float((beta2 - c) / max(beta1, 1e-9))
    return background, beta1, beta2, g


def gates(img, px_um, em_nm, na):
    f, p = radial_psd(img, px_um)
    nyq = 1.0 / (2 * px_um)
    cutoff = 2.0 * na / (em_nm / 1000.0)
    sigma_px = 0.21 * em_nm / na / (px_um * 1000.0)
    floor = float(np.median(p[f > 0.82 * nyq]))
    cross = float(f[np.argmax(((p - floor) <= 0) & (f > 0.15 * nyq))])
    noise_frac = float((f > cross).mean())
    return dict(sigma_px=sigma_px, nyq=nyq, cutoff=cutoff, cross=cross,
                noise_frac=noise_frac, ratio=cross / nyq)


def dim_lever(nz, z_step_um, em_nm, na, ri=1.515, patch_z=13):
    """2D or 3D, and WHY. Two hard requirements the 3D path has.

    1. ``train_3d`` cuts each patch from ``2 * patch_z`` consecutive planes (axial parity
       splitting), so the stack needs at least that many. With the reference ``patch_z=13``
       that is 26 planes; a 10-plane stack cannot train 3D at all.
    2. The parity split assumes the two half-stacks see nearly the same structure. That
       fails once the z step approaches the axial PSF width, because alternate planes stop
       being two noisy views of one thing and become two different things.
    """
    axial_fwhm = 2.0 * ri * (em_nm / 1000.0) / max(na * na, 1e-9)
    if nz < 2 * patch_z:
        return "2D", (f"only {nz} z planes; the 3D path needs 2*patch_z = {2*patch_z}. "
                      f"(patch_z could drop to {max(1, nz//2)}, but that is little axial "
                      f"context and is not a configuration I have tested.)")
    if z_step_um > 0.5 * axial_fwhm:
        return "2D", (f"z step {z_step_um:.2f} um is coarse against the {axial_fwhm:.2f} um "
                      f"axial PSF, so alternate planes are not two views of one structure "
                      f"and the axial-parity split breaks.")
    return "3D", f"{nz} planes at {z_step_um:.2f} um, axial PSF {axial_fwhm:.2f} um."


def autoparams(img, px_um, em_nm, na, camera_mode=None, conversion_gain=None,
               n_units_available=None, is_stack=False, nz=1, z_step_um=0.0, ri=1.515):
    g = gates(img, px_um, em_nm, na)
    dim, dim_why = dim_lever(nz, z_step_um, em_nm, na, ri) if is_stack else ("2D", "single plane")
    gain, rn = _CAMERA_GAIN.get((camera_mode, conversion_gain), _DEFAULT_GAIN)
    known = (camera_mode, conversion_gain) in _CAMERA_GAIN
    bg, b1, b2, g_meas = noise_model(img, gain, rn)

    upsample = g["ratio"] >= 0.9
    # sigma is evaluated on the OUTPUT grid: upsampling halves the pixel, doubling sigma
    sigma_out = g["sigma_px"] * (2 if upsample else 1)
    output = "denoised" if sigma_out < 0.7 else "deconvolved"
    tile = 128 if upsample else 256
    # the measured cache cliff: (tile + 2*insert)*(1+upsample) must stay <= 320 px
    insert = 16
    hess = 0.005 if g["noise_frac"] >= 0.35 else 0.02
    # In 2D every (P,T,Z) plane is a candidate unit; in 3D every (P,T) volume is.
    # More units cost RAM, not time - the iteration count sets the runtime - so the rule
    # is "sample the variety" rather than "sample a little".
    units = 20 if n_units_available is None else int(np.clip(n_units_available // 8, 16, 32))

    p = dict(dim=dim, mode="zero_shot", output=output,
             upsample=upsample, tile=tile, overlap=20, insert_xy=insert,
             iterations=20000, hess_weight=hess, denoise_weight=0.5, alpha=1.0,
             patch=128, batch_size=4, learning_rate=5e-5, train_units=units,
             background=round(bg, 1), beta1=round(b1, 2), beta2=round(b2, 1),
             norm_low=0.0, seed=0)
    return g, p, dict(gain_known=known, gain_measured=g_meas, gain_used=gain,
                      read_noise_e=rn, sigma_out=sigma_out, dim_why=dim_why)


def verdict(g, extra):
    s = g["sigma_px"]
    out = []
    if extra["sigma_out"] < 0.7:
        out.append("[1] NO deconvolution: PSF sigma is below the node's max(0.5,sigma) "
                   "clamp, so it would train against the clamp, not your optics. "
                   "-> output=denoised")
    elif extra["sigma_out"] < 1.5:
        out.append("[1] WEAK deconvolution: denoising benefit, no resolution gain, "
                   "expect ~10% object-width inflation.")
    else:
        out.append("[1] REAL deconvolution possible (the paper's regime).")
    nf = g["noise_frac"]
    out.append(f"[2] {'LARGE' if nf>=.35 else 'MODERATE' if nf>=.15 else 'SMALL'} "
               f"denoising prize: {nf:.0%} of the sampled band is noise-dominated.")
    out.append(f"[3] signal reaches {g['ratio']:.0%} of Nyquist -> "
               f"{'SAMPLING' if g['ratio']>=.9 else 'NOISE'}-limited -> "
               f"upsample={'True' if g['ratio']>=.9 else 'False'}")
    return out


def main(path, channel=0):
    from nodelab_v2.nd2_compat import import_nd2
    nd2 = import_nd2()
    with nd2.ND2File(path) as f:
        sizes = f.sizes
        ch = f.metadata.channels[channel]
        px = f.voxel_size().x
        em = ch.channel.emissionLambdaNm or 520.0
        na = ch.microscope.objectiveNumericalAperture or 1.4
        txt = " ".join(str(v) for v in f.text_info.values())
        cm = re.search(r"Camera Mode:\s*([^\r\n]+)", txt)
        cg = re.search(r"Conversion Gain:\s*([^\r\n]+)", txt)
        cm = cm.group(1).strip() if cm else None
        cg = cg.group(1).strip() if cg else None
        nz = sizes.get("Z", 1)
        d = f.to_dask()
        idx = [0] * (d.ndim - 2)
        ax = list(sizes)[:-2]
        for i, a in enumerate(ax):
            idx[i] = (sizes[a] // 2) if a in ("Z", "P", "T") else 0
            if a == "C":
                idx[i] = channel
        img = np.asarray(d[tuple(idx)], dtype=np.float32)
        n_units = int(np.prod([sizes[a] for a in ax if a in ("P", "T")]) or 1)
        n_planes = n_units * max(1, nz)

        zst = abs(f.voxel_size().z or 0.0)
        ri = ch.microscope.immersionRefractiveIndex or 1.515
    g, p, extra = autoparams(img, px, em, na, cm, cg, n_units, is_stack=nz > 1,
                             nz=nz, z_step_um=zst, ri=ri)
    if p["dim"] == "2D":            # 2D trains on planes, so z multiplies the pool
        p["train_units"] = int(np.clip(n_planes // 8, 16, 32))
    print(f"\n=== {path}  channel {channel} ({ch.channel.name})")
    print(f"  axes {sizes} | {px:.4f} um/px | em {em:.0f} nm | NA {na} | Z {nz}")
    print(f"  camera: {cm!r} / {cg!r} -> gain {extra['gain_used']} ADU/e- "
          f"({'from metadata' if extra['gain_known'] else 'DEFAULT, mode not recognised'}), "
          f"read noise {extra['read_noise_e']} e-")
    print(f"  photon-transfer cross-check: measured gain {extra['gain_measured']:.2f} ADU/e-")
    print(f"  PSF sigma {g['sigma_px']:.2f} px (input) -> {extra['sigma_out']:.2f} px (output grid)")
    print(f"  Nyquist {g['nyq']:.2f} /um | OTF cutoff {g['cutoff']:.2f} /um | "
          f"signal=noise {g['cross']:.2f} /um")
    for line in verdict(g, extra):
        print("  " + line)
    print(f"  [4] dim = {p['dim']}: {extra['dim_why']}")
    print("\n  --- socket values ---")
    for k in ("dim", "mode", "output", "upsample", "tile", "overlap", "iterations",
              "hess_weight", "train_units", "background", "beta1", "beta2", "patch",
              "batch_size", "learning_rate", "denoise_weight", "alpha", "norm_low", "seed"):
        print(f"    {k:<16} {p[k]}")
    return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
