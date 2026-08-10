"""Prototype: fill enhance.zs_deconvnet sockets from ND2 metadata + pixel statistics.

The formula (see evidence-log entry for derivations and sources):
  emission_nm  <- channel metadata (emissionLambdaNm)
  na           <- ObjectiveNA (text_info / unstructured metadata)
  psf_pixel_um <- 0 (derive the PSF from NA + emission; only nonzero for PSF FILES)
  background   <- ~1st percentile of the channel's dimmest frame (camera pedestal)
  beta1        <- camera gain in ADU per electron (photon-transfer or spec sheet);
                  1.0 when unknown (right order for modern sCMOS)
  beta2        <- read-noise variance in counts^2 = (read_noise_e- * beta1)^2;
                  0.0 when unknown, ~1-2 for Kinetix/AX-class sCMOS
  train_units  <- min(8, number of available planes)
  iterations   <- time_budget_s / t_step_s (calibrate t_step with a ~50-step run;
                  TensorFlow is CPU-only on native Windows - do NOT assume the
                  paper's 50k-on-GPU is reachable)
  everything else: published 2D reference values already in the node defaults
  (learning_rate 5e-5, batch 4, patch 128, denoise_weight 0.5, hess_weight 0.02,
  upsample True, damping 0/1, alpha 1.0).

NOT wired into the node yet - that lands via build-node-v2 as metadata-aware
defaults (same mechanism emission_nm already documents).
"""
import re

import numpy as np


def autofill(nd2_path: str, channel_name: str, camera_hints: dict | None = None) -> dict:
    import nd2

    with nd2.ND2File(nd2_path) as h:
        chans = [c.channel.name for c in h.metadata.channels]
        ci = chans.index(channel_name)
        em = getattr(h.metadata.channels[ci].channel, "emissionLambdaNm", None)
        ti = h.text_info.get("description", "")
        m = re.search(r"Numerical Aperture:\s*([0-9.]+)", ti)
        na = float(m.group(1)) if m else None
        cam = re.search(r"Camera Name:\s*(.+)", ti)
        sizes = h.sizes
        # dimmest available frame of this channel for the pedestal estimate
        arr = np.asarray(h.asarray())
        # collapse leading axes to (N, C, Y, X) regardless of P/T/Z layout
        arr = arr.reshape(-1, sizes["C"], sizes["Y"], sizes["X"])
        ch = arr[:, ci].astype(np.float32)
        background = float(np.percentile(ch, 1.0))
        n_planes = ch.shape[0]

    hints = camera_hints or {}
    beta1 = hints.get("gain_adu_per_e", 1.0)
    read_e = hints.get("read_noise_e", None)
    beta2 = round((read_e * beta1) ** 2, 2) if read_e else 0.0

    return {
        "file": nd2_path, "channel": channel_name,
        "camera": cam.group(1).strip() if cam else None,
        "params": {
            "emission_nm": em, "na": na, "psf_pixel_um": 0.0,
            "background": round(background, 1),
            "beta1": beta1, "beta2": beta2, "alpha": 1.0,
            "learning_rate": 5e-5, "batch_size": 4, "patch": 128,
            "train_units": min(8, n_planes),
            "upsample": True, "denoise_weight": 0.5, "hess_weight": 0.02,
            "damping_length": 0, "damping_width": 1, "seed": 0,
        },
        "notes": {
            "n_planes_available": n_planes,
            "beta_source": "camera_hints" if hints else
                           "DEFAULTS - unverified; measure via photon transfer or dark frames",
            "iterations": "budget / t_step; calibrate with a 50-step run first",
        },
    }


if __name__ == "__main__":
    import json

    out = {}
    out["hk_2025_af647"] = autofill(
        "H:/HK_segmentation/2025.09.07_EDF_Count_Intensity_ColocalizationDrawROI.nd2",
        "AF647")
    out["monolayer_red"] = autofill(
        "C:/Users/McGheeLab - Analysis/Desktop/Alex data/Monolayer_stain.nd2",
        "W1 - Red",
        camera_hints={"gain_adu_per_e": 1.18, "read_noise_e": 1.2})  # Kinetix spec ballpark
    print(json.dumps(out, indent=2))
    with open("CodeLog/live/cellsam-custom-model/zs_params_filled.json", "w") as f:
        json.dump(out, f, indent=2)
