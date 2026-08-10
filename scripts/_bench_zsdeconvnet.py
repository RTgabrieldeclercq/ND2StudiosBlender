"""ZS-DeconvNet validation bench — GOLDEN-OUTPUT PARITY against the authors' own results.

WHY THIS EXISTS
    `nodegraph/kernels/zsdeconvnet.py` is a port of a Keras-2 (TF 2.5) reference
    implementation onto Keras 3 (TF 2.21). The dangerous failure mode is not a crash: a
    legacy `.h5` checkpoint is matched to a graph by the TOPOLOGICAL ORDER of its
    weight-bearing layers, with no name check, so a mis-ordered rebuild of the network
    loads "successfully" and then predicts plausible-looking nonsense. RCAN3D in
    particular has 38 Conv3D layers, most of them the same shape, so a permutation among
    them is invisible to every shape assertion.

    Nothing in `nodegraph.selftest` can see that. The only check that can is running the
    authors' PUBLISHED weights on the authors' PUBLISHED input and comparing against the
    authors' PUBLISHED output — which is what this script does. The selftest covers the
    maths that does NOT need a 52 MB checkpoint (re-corruption, tiling, PSF, metadata),
    and stays fast; this covers the rest, and is manual because it needs a ~2 GB download.

DATA (both free, no account needed)
    * Pre-trained models + their reference inference results — the `saved_models` folder
      linked from the repo's ReadMe:
        https://drive.google.com/drive/folders/1XAOuLYXYFCxlElRwvik_fs7TqZlRixGv
      Point --models at the extracted `saved_models` directory.
    * Raw training data + PSFs (Zenodo record 7261163, `2D data.zip` is only 12 MB):
        https://zenodo.org/records/7261163
      Point --zenodo at the extracted `2D data` directory. Used by the PSF check, which
      compares this repo's metadata-DERIVED Gaussian PSF against a real measured PSF whose
      optics are recorded in its filename (`psf_emLambda525_dxy0.0313_NA1.3.tif`).

WHAT "PARITY" MEANS HERE
    The reference saves `uint16(1e4 * prctile_norm(x, 3, 100))`, so its published TIFFs are
    quantized to 1 part in 10 000. Agreement is therefore reported in those units: a max
    absolute deviation of 1 count is float-rounding at the truncation boundary and is the
    best achievable, not an approximation.

    The 2D case reproduces `infer_demo_2D.sh` exactly and lands there. The 3D case cannot
    be run at the authors' own tile geometry on a CPU build — one activation is 4.29 GiB and
    TensorFlow's `mklcpu` allocator caps its arena at 64 GiB regardless of installed RAM —
    and the shipped 3D demo output is additionally stored MIRRORED along axis 1 relative to
    its input (`Infer_3D.py` flips on load and again on save; the published file kept only
    one of those), so both orientations are scored and the better reported.

Manual (NOT in nodegraph.selftest — it needs a multi-GB download):
    PYTHONUTF8=1 python scripts/_bench_zsdeconvnet.py --models <dir>/saved_models
    PYTHONUTF8=1 python scripts/_bench_zsdeconvnet.py --models <dir> --case 2d
    PYTHONUTF8=1 python scripts/_bench_zsdeconvnet.py --zenodo "<dir>/2D data"   # PSF only
    PYTHONUTF8=1 python scripts/_bench_zsdeconvnet.py --zenodo "<dir>/2D data" --train 200
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodegraph.kernels import zsdeconvnet as zsk           # noqa: E402


def _seg_window(extent: int, n_win: int, overlap: int) -> int:
    """The reference's tile SIZE from a tile COUNT — `Infer_2D.py:78` / `Infer_3D.py:107`.

    Its knob is `--num_seg_window_*` (how many tiles); ours is a tile size. Converting here
    is what lets the bench reproduce a published invocation verbatim.
    """
    return math.ceil((extent + (n_win - 1) * overlap) / n_win)


def _score(name: str, mine: np.ndarray, golden: np.ndarray) -> float:
    """Compare in the units the reference SAVES: `uint16(1e4 * x)`. Returns Pearson r."""
    a = (1e4 * np.asarray(mine, dtype=np.float64)).astype(np.uint16).astype(np.float64)
    b = np.asarray(golden).astype(np.float64)
    if a.shape != b.shape:
        print(f"   {name:9s} SHAPE MISMATCH mine={a.shape} golden={b.shape}")
        return -1.0
    d = np.abs(a - b)
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print(f"   {name:9s} maxabs={d.max():7.1f}  rmse={np.sqrt((d ** 2).mean()):7.2f}  "
          f"r={r:.6f}  bit-identical={100.0 * float((d == 0).mean()):5.2f}%  "
          f"(golden range {b.min():.0f}-{b.max():.0f} of 9999)")
    return r


def case_2d(models: str) -> bool:
    """`infer_demo_2D.sh`, WF lysosome: 1 tile, insert_xy=16, upsample_flag=1."""
    import tifffile
    d = os.path.join(models, "WF2D_Lysosome")
    inp = tifffile.imread(os.path.join(d, "test_data", "NoisyInput.tif"))
    gd = tifffile.imread(os.path.join(d, "saved_model", "Inference_demo",
                                      "img0_denoised.tif"))
    gc = tifffile.imread(os.path.join(d, "saved_model", "Inference_demo",
                                      "img0_deconved.tif"))
    w = os.path.join(d, "saved_model", "weights_20000.h5")
    print(f"2D  WF2D_Lysosome  input {inp.shape} {inp.dtype}  weights_20000.h5")
    h, wd = inp.shape
    t = time.time()
    den, dec = zsk.infer_2d(inp, arch="unet2d", weights_path=w,
                            tile_y=_seg_window(h, 1, 20), tile_x=_seg_window(wd, 1, 20),
                            overlap=20, upsample=True, insert_xy=16, norm_low=3.0)
    print(f"   inference {time.time() - t:.1f} s")
    ok = _score("denoised", den, gd) > 0.999 and _score("deconved", dec, gc) > 0.999
    print(f"   2D: {'PARITY' if ok else 'MISMATCH'}")
    return ok


def case_3d(models: str, z_window: int = 0) -> bool:
    """`infer_demo_3D.sh`, LLS mitochondria: RCAN3D, bg=100, Fourier damping 450/1."""
    import tifffile
    d = os.path.join(models, "LLS3D_Mitochondria")
    inp = tifffile.imread(os.path.join(d, "test_data", "NoisyWF.tif"))
    gd = tifffile.imread(os.path.join(d, "saved_model", "Inference_demo", "00_den.tif"))
    gc = tifffile.imread(os.path.join(d, "saved_model", "Inference_demo", "00_dec.tif"))
    w = os.path.join(d, "saved_model", "weights_10000.h5")
    nz, hh, ww = inp.shape
    wz = int(z_window) or _seg_window(nz, 2, 4)
    print(f"3D  LLS3D_Mitochondria  input {inp.shape}  weights_10000.h5  "
          f"z-window {wz} (reference {_seg_window(nz, 2, 4)})")
    t = time.time()
    den, dec = zsk.infer_3d(np.flip(inp, axis=1), arch="rcan3d", weights_path=w,
                            tile_z=wz, tile_y=_seg_window(hh, 2, 20),
                            tile_x=_seg_window(ww, 1, 20), overlap=20, overlap_z=4,
                            upsample=False, insert_xy=8, insert_z=2, background=100.0,
                            norm_low=3.0, damping_length=450, damping_width=1)
    print(f"   inference {time.time() - t:.1f} s")
    best = -1.0
    for label, dn, dc in (("as computed (mirrored)", den, dec),
                          ("un-flipped", np.flip(den, axis=1), np.flip(dec, axis=1))):
        print(f"   -- {label} --")
        r = min(_score("denoised", dn, gd), _score("deconved", dc, gc))
        best = max(best, r)
    ok = best > 0.999
    print(f"   3D: {'PARITY' if ok else 'MISMATCH'}  (best r={best:.6f})")
    return ok


def case_metrics(models: str, zenodo: str = "") -> bool:
    """TIER 0 — validate the METRIC before trusting it on data with no ground truth.

    `WF2D_Lysosome` is the one published case that ships a high-SNR `ClearGT.tif` beside its
    `NoisyInput.tif`, so the paper's Eq. 13-14 protocol can be run exactly as described and
    the answer is known in advance: the authors' own ZS-DeconvNet output must score BETTER
    against the clear reference than the raw noisy input does. That is their Fig. 1d claim,
    and reproducing its ORDERING is what says this implementation of PSNR / RSP / RSE is
    right. Without this step, a validation report on WellA3 would be numbers with no
    demonstrated meaning.

    The lysosome PSF ships only as an `.mrc` OTF, which this port deliberately does not read,
    so the PSF is derived from the acquisition's optics (TIRF, 1.49 NA objective per the
    paper's Methods, 560 nm emission per the OTF filename, 0.0313 um/px on the 2x SR grid).
    When `--zenodo` is given the check is REPEATED with the real measured microtubule PSF: the
    ordering must not depend on which PSF is used, and if it did, the metric would be
    measuring the PSF rather than the image.
    """
    import tifffile
    from nodegraph.catalog._shared.psf import diffraction_sigmas
    d = os.path.join(models, "WF2D_Lysosome")
    gt = np.asarray(tifffile.imread(os.path.join(d, "test_data", "ClearGT.tif")),
                    dtype=np.float64)
    noisy = np.asarray(tifffile.imread(os.path.join(d, "test_data", "NoisyInput.tif")),
                       dtype=np.float64)
    dec = np.asarray(tifffile.imread(os.path.join(d, "saved_model", "Inference_demo",
                                                  "img0_deconved.tif")), dtype=np.float64)
    den = np.asarray(tifffile.imread(os.path.join(d, "saved_model", "Inference_demo",
                                                  "img0_denoised.tif")), dtype=np.float64)
    print("TIER 0  metric validation on WF2D_Lysosome (ships a high-SNR ClearGT)")
    print(f"   ClearGT {gt.shape}  NoisyInput {noisy.shape}  "
          f"published deconved {dec.shape}  denoised {den.shape}")

    psfs = [("derived Gaussian (1.49 NA, 560 nm, 0.0313 um/px)",
             zsk.gaussian_psf_from_sigmas(
                 diffraction_sigmas(560.0, 1.49, 0.0313, None, False)))]
    if zenodo:
        for root, _dirs, files in os.walk(zenodo):
            for fn in files:
                if fn.lower().startswith("psf") and fn.lower().endswith((".tif", ".tiff")):
                    psfs.append((f"measured {fn}",
                                 zsk.crop_psf(zsk.load_psf_tif(os.path.join(root, fn)))))
                    break
    ok = True
    for label, psf in psfs:
        print(f"   -- PSF: {label}  {psf.shape} --")
        rows = []
        # `blur` per CANDIDATE, not per run: only the 2x deconvolved head is super-resolved
        # and so needs pushing back through the optics. Blurring the other two would add a
        # second PSF the ClearGT never had and quietly suppress their noise (see
        # `degrade_to_reference`) — which is exactly how a first cut made the raw noisy frame
        # score level with ZS-DeconvNet's own output.
        for name, img, blur in (("raw noisy input", noisy, False),
                                ("published denoised", den, False),
                                ("published deconved", dec, True)):
            m = zsk.resolution_scaled_metrics(img, gt, psf, blur=blur)
            rows.append((name, m))
            print(f"      {name:20s} PSNR {m['psnr']:6.2f} dB   RSP {m['rsp']:.4f}   "
                  f"RSE {m['rse']:.4f}")
        base = rows[0][1]["psnr"]
        better = [n for n, m in rows[1:] if m["psnr"] > base]
        good = len(better) == 2
        ok = ok and good
        print(f"      -> both ZS-DeconvNet heads beat the raw input: "
              f"{'YES' if good else 'NO (' + ', '.join(better) + ' only)'}")

    # and the SQUIRREL reading of the same machinery: reference = the NOISY input, i.e. no
    # ground truth at all. This is the mode WellA3 has to use, so check it agrees in ORDER.
    print("   -- SQUIRREL mode (reference = the noisy input, NO ground truth) --")
    psf = psfs[0][1]
    for name, img, blur in (("published denoised", den, False),
                            ("published deconved", dec, True)):
        m = zsk.resolution_scaled_metrics(img, noisy, psf, blur=blur)
        print(f"      {name:20s} RSP {m['rsp']:.4f}   RSE {m['rse']:.4f}   "
              f"(self-consistency, no GT)")
    fr = zsk.faint_signal_retention(noisy, dec, psf=psf)
    print(f"      faint-signal retention on the published output: n={fr['n']} points, "
          f"median contrast retained {fr['median_retained']:.2f}, "
          f"lost >50%: {100 * fr['lost_fraction']:.1f}%")
    print(f"   TIER 0: {'PASS' if ok else 'FAIL'}")
    return ok


def case_validate(models: str, inp: str, weights: str, *, tile: int, upsample: bool,
                  background: float, na: float, emission: float, pixel_um: float,
                  limit: int, out_dir: str) -> bool:
    """TIER 1/2 — validate YOUR OWN trained model's output, with no ground truth.

    Runs inference on `--input` with `--weights`, then for each plane reports:
      * **self-consistency** — re-blur the output with the PSF and compare to the measured
        input (the reference's `out_mul_otf`, and the paper's Eq. 5 degradation term read as
        a score). Catches structure the network INVENTED.
      * **SQUIRREL** RSP / RSE and an error MAP, per the Discussion's recommendation.
      * **faint-signal retention** — the paper's Supplementary Fig. 28a failure, which
        re-blurring is blind to.

    PNGs go to `--out-dir` so the error map can be looked at rather than summarized: a single
    RSP cannot say "the disagreement is all on the three brightest cells".
    """
    import tifffile
    from nodegraph.catalog._shared.psf import diffraction_sigmas
    planes = []
    if inp.lower().endswith(".nd2"):
        import nd2
        with nd2.ND2File(inp) as f:
            arr = f.asarray()
        flat = arr.reshape((-1,) + arr.shape[-2:])
        step = max(1, len(flat) // max(1, limit))
        planes = [np.asarray(flat[i], dtype=np.float32)
                  for i in range(0, len(flat), step)][:limit]
    else:
        a = np.asarray(tifffile.imread(inp), dtype=np.float32)
        planes = [a] if a.ndim == 2 else [np.asarray(p) for p in a[:limit]]
    if not planes:
        print("validate: no planes read from --input")
        return False
    psf = zsk.gaussian_psf_from_sigmas(
        diffraction_sigmas(emission, na, pixel_um, None, False))
    print(f"TIER 1/2  validating {os.path.basename(weights)} on {len(planes)} plane(s) of "
          f"{os.path.basename(inp)}")
    print(f"   PSF sigma {tuple(round(s, 3) for s in zsk.psf_sigma(psf))} px, "
          f"kernel {psf.shape}  (derived: {emission} nm, NA {na}, {pixel_um} um/px)")
    os.makedirs(out_dir, exist_ok=True)
    rsps, rses, rets = [], [], []
    for i, pl in enumerate(planes):
        t = time.time()
        den, dec = zsk.infer_2d(pl, arch="unet2d", weights_path=weights, tile=tile,
                                overlap=20, upsample=upsample, insert_xy=16, norm_low=0.0)
        out = dec if upsample else den
        m = zsk.resolution_scaled_metrics(out, np.maximum(pl - background, 0.0), psf)
        fr = zsk.faint_signal_retention(np.maximum(pl - background, 0.0), out,
                                        psf=psf)
        rsps.append(m["rsp"]); rses.append(m["rse"]); rets.append(fr["median_retained"])
        print(f"   plane {i + 1}/{len(planes)} ({time.time() - t:5.1f}s)  "
              f"RSP {m['rsp']:.4f}  RSE {m['rse']:.4f}  "
              f"faint retained {fr['median_retained']:.2f} "
              f"(lost>50%: {100 * fr['lost_fraction']:4.1f}% of {fr['n']})")
        if i == 0:
            try:
                from PIL import Image

                def png(a, name, lo=1, hi=99.8):
                    v = np.asarray(a, dtype=np.float64)
                    p0, p1 = np.percentile(v, lo), np.percentile(v, hi)
                    u = np.clip((v - p0) / max(p1 - p0, 1e-9), 0, 1)
                    Image.fromarray((u * 255).astype(np.uint8)).save(
                        os.path.join(out_dir, name))
                png(pl, "01_input.png")
                png(out, "02_output.png")
                png(m["degraded"], "03_output_reblurred.png")
                png(m["error_map"], "04_error_map.png", 0, 99.5)
                print(f"      wrote input / output / re-blurred / error-map PNGs to "
                      f"{out_dir}")
            except Exception as exc:                      # noqa: BLE001
                print("      (PNG report skipped:", exc, ")")
    print(f"   MEAN over {len(planes)} plane(s): RSP {np.nanmean(rsps):.4f}  "
          f"RSE {np.nanmean(rses):.4f}  faint retained {np.nanmean(rets):.2f}")
    # Thresholds are advisory, and deliberately loose: RSP is a self-consistency score, not
    # an accuracy score, so a high value only says "nothing was invented" — it says nothing
    # about whether the sharpening is real. Named so the report cannot be over-read.
    ok = bool(np.nanmean(rsps) > 0.9 and np.nanmean(rets) > 0.5)
    print(f"   TIER 1/2: {'PLAUSIBLE (self-consistent, faint signal retained)' if ok else 'SUSPECT — inspect 04_error_map.png'}")
    print("   NOTE: RSP near 1 means the output is consistent with the measurement. It is "
          "NOT evidence of a resolution gain — for that you need a high-SNR reference.")
    return ok


def case_psf(zenodo: str) -> bool:
    """Is the metadata-DERIVED Gaussian PSF a fair stand-in for a MEASURED one?

    The node's default PSF comes from `_shared/psf.py::diffraction_sigmas` — the
    Gaussian approximation sigma_xy ~ 0.21*lambda/NA. The Zenodo microtubule set ships a
    real simulated-optics PSF whose parameters are in its filename, so the two can be
    compared at the SAME optics. Passing means the derived width is within a factor of ~1.5
    of the measured one, which is the honest bar for a Gaussian approximation to a
    diffraction PSF (it has no Airy rings, so it can never match exactly) — and it is what
    justifies shipping "no PSF file needed" as the default.
    """
    import tifffile
    from nodegraph.catalog._shared.psf import diffraction_sigmas
    hits = []
    for root, _dirs, files in os.walk(zenodo):
        for fn in files:
            if fn.lower().endswith((".tif", ".tiff")) and "psf" in fn.lower():
                hits.append(os.path.join(root, fn))
    if not hits:
        print("PSF: no PSF TIFF found under --zenodo (expected e.g. "
              "Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif)")
        return False
    ok = True
    for p in sorted(hits):
        base = os.path.basename(p)
        m_lam = re.search(r"emLambda(\d+(?:\.\d+)?)", base)
        m_dxy = re.search(r"dxy(\d*\.?\d+)", base)
        m_na = re.search(r"NA(\d*\.?\d+)", base)
        psf = zsk.load_psf_tif(p)
        sig = zsk.psf_sigma(psf)
        print(f"PSF {base}\n     shape {psf.shape}  measured sigma (px) "
              f"{tuple(round(s, 3) for s in sig)}")
        if not (m_lam and m_dxy and m_na):
            print("     filename does not record lambda/dxy/NA — no derived comparison")
            continue
        lam, dxy, na = float(m_lam.group(1)), float(m_dxy.group(1)), float(m_na.group(1))
        want = diffraction_sigmas(lam, na, dxy, None, False)
        lateral = sig[-1]
        ratio = lateral / want[-1]
        print(f"     optics from filename: {lam} nm, NA {na}, {dxy} um/px")
        print(f"     derived sigma_xy = {want[-1]:.3f} px   measured = {lateral:.3f} px   "
              f"ratio = {ratio:.2f}")
        # resampling onto a coarser grid must scale the width proportionally
        rs = zsk.resample_psf(psf, dxy, dxy * 2.0)
        rsig = zsk.psf_sigma(rs)[-1]
        print(f"     resampled to {dxy * 2:.4f} um/px: sigma {rsig:.3f} px "
              f"(expect ~{lateral / 2:.3f})")
        good = (0.66 < ratio < 1.5) and abs(rsig - lateral / 2) < max(0.5, lateral * 0.25)
        ok = ok and good
        print(f"     -> {'OK' if good else 'OUT OF TOLERANCE'}")
    print(f"   PSF: {'OK' if ok else 'MISMATCH'}")
    return ok


def case_train(zenodo: str, iterations: int) -> bool:
    """Does zero-shot training actually LEARN on real data (loss decreases)?

    Trains a 2D model on the Zenodo microtubule/lysosome frames with a metadata-derived PSF
    and reports the loss trend. Not a quality claim — a few hundred CPU iterations is far
    short of the paper's 50 000 — but it does prove the re-corruption -> patch -> dual-loss
    -> optimizer chain is wired up and converging rather than merely running.
    """
    import tifffile
    from nodegraph.catalog._shared.psf import diffraction_sigmas
    frames = []
    for root, _d, files in os.walk(zenodo):
        if os.path.basename(root).lower() in ("train_data", "traindata"):
            for fn in sorted(files)[:4]:
                if fn.lower().endswith((".tif", ".tiff")):
                    a = np.asarray(tifffile.imread(os.path.join(root, fn)),
                                   dtype=np.float32)
                    if a.ndim == 3:
                        a = a.sum(axis=0)
                    frames.append(a)
        if len(frames) >= 4:
            break
    if not frames:
        print("train: no train_data/*.tif found under --zenodo")
        return False
    print(f"train: {len(frames)} frame(s) of {frames[0].shape}, "
          f"{iterations} iterations on CPU")
    psf = zsk.gaussian_psf_from_sigmas(diffraction_sigmas(525.0, 1.3, 0.0313, None, False))
    b2 = zsk.estimate_beta2(frames, bg=100.0)
    print(f"   estimated beta2 (read-noise variance) = {b2:.2f} counts^2")
    hist = []
    t = time.time()
    zsk.train_2d(frames, psf, iterations=int(iterations), batch_size=4, patch=128,
                 insert_xy=16, upsample=True, beta2=b2, background=100.0, seed=0,
                 progress=lambda s, n, loss: hist.append(loss))
    dt = time.time() - t
    k = max(1, len(hist) // 5)
    first, last = float(np.mean(hist[:k])), float(np.mean(hist[-k:]))
    print(f"   {dt:.1f} s ({dt / max(1, len(hist)):.2f} s/iter); loss "
          f"{first:.4g} -> {last:.4g} ({100.0 * (1 - last / max(first, 1e-12)):+.1f}%)")
    ok = last < first
    print(f"   train: {'LEARNING' if ok else 'NOT CONVERGING'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="extracted `saved_models` directory")
    ap.add_argument("--zenodo", default="", help="extracted Zenodo `2D data` directory")
    ap.add_argument("--case", default="all", choices=("all", "2d", "3d", "metrics"))
    ap.add_argument("--validate", default="",
                    help="validate YOUR trained model: path to an .nd2/.tif to run on "
                         "(requires --weights)")
    ap.add_argument("--weights", default="", help="the checkpoint --validate should use")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=0)
    ap.add_argument("--background", type=float, default=40.0)
    ap.add_argument("--na", type=float, default=0.45)
    ap.add_argument("--emission", type=float, default=499.0)
    ap.add_argument("--pixel-um", type=float, default=1.7182777601481225)
    ap.add_argument("--limit", type=int, default=4,
                    help="how many planes --validate samples (evenly strided)")
    ap.add_argument("--out-dir", default="zsdeconv_validation",
                    help="where --validate writes its PNG report")
    ap.add_argument("--z-window", type=int, default=0,
                    help="override the 3D z tile (the reference's 78 needs >64 GiB)")
    ap.add_argument("--train", type=int, default=0,
                    help="also run N zero-shot training iterations on --zenodo data")
    args = ap.parse_args()
    if not args.models and not args.zenodo and not args.validate:
        ap.error("give --models (golden parity / metrics), --zenodo (PSF / training) "
                 "and/or --validate <image> --weights <ckpt>")
    if args.validate and not args.weights:
        ap.error("--validate needs --weights (the checkpoint to validate)")
    results = {}
    root = ""
    if args.models:
        root = args.models
        if not os.path.isdir(os.path.join(root, "WF2D_Lysosome")) and \
                os.path.isdir(os.path.join(root, "saved_models")):
            root = os.path.join(root, "saved_models")     # tolerate the wrapper folder
        if args.case in ("all", "2d"):
            results["2D golden"] = case_2d(root)
        if args.case in ("all", "3d"):
            results["3D golden"] = case_3d(root, args.z_window)
        if args.case in ("all", "metrics"):
            results["metrics (tier 0)"] = case_metrics(root, args.zenodo)
    if args.validate:
        # Tier 0 first when it is available, so the report below is numbers whose meaning has
        # just been demonstrated rather than asserted.
        if root and "metrics (tier 0)" not in results:
            results["metrics (tier 0)"] = case_metrics(root, args.zenodo)
        results["validate (tier 1/2)"] = case_validate(
            root, args.validate, args.weights, tile=args.tile,
            upsample=bool(args.upsample), background=args.background, na=args.na,
            emission=args.emission, pixel_um=args.pixel_um, limit=args.limit,
            out_dir=args.out_dir)
    if args.zenodo:
        results["PSF"] = case_psf(args.zenodo)
        if args.train:
            results["train"] = case_train(args.zenodo, args.train)
    print("\n" + "=" * 70)
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")
    bad = [k for k, v in results.items() if not v]
    print("=" * 70)
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    print("ALL ZS-DECONVNET BENCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
