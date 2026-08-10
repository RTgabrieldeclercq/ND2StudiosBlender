"""Time one training iteration and one inference tile, with and without upsample."""
from __future__ import annotations
import sys; sys.path.insert(0, ".")     # script dir is sys.path[0], not the repo root
import time, numpy as np, tifffile
from nodegraph.kernels import zsdeconvnet as zsk
from nodegraph.catalog._shared.psf import diffraction_sigmas

RAW = r"C:\Users\McGheeLab - Analysis\Desktop\Alex data\LAGFP_Caps_Z_raw.ome.tif"
PX = 0.107136166827053
planes = [tifffile.imread(RAW, key=t).astype(np.float32) for t in (10, 30, 54, 80)]

for ups in (True, False):
    px_eff = PX / 2 if ups else PX
    psf = zsk.gaussian_psf_from_sigmas(diffraction_sigmas(525.0, 1.4, px_eff, 0.0, False))
    t0 = time.time()
    w = zsk.train_2d(planes, psf, iterations=12, batch_size=4, patch=128, insert_xy=16,
                     upsample=ups, background=120.0, beta1=4.7, beta2=25.0, seed=0)
    t1 = time.time()
    # warm: 12 more, so the first-call graph build is excluded
    t2 = time.time()
    zsk.train_2d(planes, psf, iterations=12, batch_size=4, patch=128, insert_xy=16,
                 upsample=ups, background=120.0, beta1=4.7, beta2=25.0, seed=1)
    t3 = time.time()
    per = (t3 - t2) / 12
    print(f"upsample={ups}: psf {psf.shape}  cold12 {t1-t0:.1f}s  warm12 {t3-t2:.1f}s  "
          f"-> {per:.2f} s/iter | 2000 it = {per*2000/60:.0f} min | "
          f"10000 it = {per*10000/3600:.1f} h | 20000 it = {per*20000/3600:.1f} h")

    tile = 128 if ups else 256
    img = planes[2]
    t4 = time.time()
    out = zsk.infer_2d(img, arch="unet2d", weights=w, tile=tile, overlap=20,
                       upsample=ups, insert_xy=16, norm_low=0.0)
    t5 = time.time()
    print(f"   inference tile={tile}: {t5-t4:.1f} s for one 512^2 plane "
          f"-> 114 frames = {(t5-t4)*114/60:.0f} min ; out {out[1].shape}")
