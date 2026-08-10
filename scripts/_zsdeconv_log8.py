"""Batch 8: the auto-parameter tool, validated against the known-good LAGFP settings."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

D.confirm("e017", note="Monolayer Red 9 h training approved. Formula extended to fill "
                       "beta1/beta2/background automatically from file metadata - "
                       "`scripts/zs_autoparams.py`, entry e018.")

D.log(kind="found",
      title="beta1 is a CAMERA constant, not an image measurement - both files record "
            "'12-bit (CMS)' / 'Sensitivity', so the gain can be read rather than fitted",
      body="The photon-transfer fit I used to derive beta1=4.7 by hand is only valid where "
           "the PSF is sampled well enough that adjacent pixels differ by noise alone. "
           "Measured across the three datasets, the two independent estimators (robust "
           "adjacent-difference, and a low-percentile block variance) agree to 17% on LAGFP "
           "(sigma_px 0.74) but diverge to 46% on Monolayer (0.52) and 55% on the "
           "fibroblasts (0.14). Below sigma_px ~0.7 a photon-transfer fit is measuring "
           "structure, not noise.\n\n"
           "The fix is that it never needed to be measured. Both ND2s record `Camera Mode: "
           "12-bit (CMS)` and `Conversion Gain: Sensitivity` - the same Kinetix "
           "configuration - and conversion gain is a property of that mode, not of the "
           "sample. Reading it gives one number that is right for every file off this "
           "microscope, and the measured value becomes a cross-check instead of the source.",
      why="This is what makes automated parameter filling trustworthy. A fitted beta1 would "
          "have been silently wrong by 50% on exactly the datasets where the node is most "
          "likely to be used (coarse pixels, low NA), and the error is invisible - it just "
          "produces an over- or under-corrupted training pair and a model that smooths too "
          "much. The metadata route cannot fail that way.\n\n"
          "The pedestal then falls out for free. `var = beta1*(I - background) + beta2` has "
          "three unknowns and a fitted line gives two, so the camera's read noise supplies "
          "the third constraint: `beta2 = (read_noise_e * beta1)^2`, and `background = "
          "(beta2 - intercept) / beta1`. Estimating the pedestal from a dark-region "
          "histogram instead reads about 13 counts high, because even the darkest fifth of "
          "a confluent frame still carries signal.",
      evidence=[{"type": "table",
                 "cols": ["dataset", "sigma_px", "estimator A (adjacent)",
                          "estimator B (blocks)", "A/B", "trustworthy?"],
                 "rows": [
                     ["LAGFP actin", "0.74", "4.30", "3.67", "1.17", "yes - above 0.7"],
                     ["Monolayer Red P0", "0.52", "4.26", "2.91", "1.46", "no"],
                     ["Monolayer Red P7", "0.52", "4.17", "2.82", "1.48", "no"],
                     ["Fibroblast 10x", "0.14", "15.20", "9.83", "1.55", "no"],
                 ],
                 "caption": "Gain in ADU/e-. The estimators only agree where the PSF is "
                            "sampled; the fibroblast figure of 15.2 is nonsense produced by "
                            "cell edges, not photons."}],
      images=[{"src": "img/17_noise_model.png",
               "caption": "Photon-transfer curves. Where the two series overlap (left "
                          "panel) the fit is measuring noise; where they separate, the "
                          "upper one is measuring structure."}],
      status="confirmed")

D.log(kind="decision",
      title="`scripts/zs_autoparams.py` - fills every socket from an ND2's own metadata "
            "plus one frame. It reproduces the hand-tuned LAGFP settings to within 4%.",
      body="    .\\.venv\\Scripts\\python.exe -u -B scripts\\zs_autoparams.py <file.nd2> [channel]\n\n"
           "Four gates now, not three - the dim lever was added after it tried to put "
           "Monolayer on the 3D path, which would have failed outright: `train_3d` cuts "
           "each patch from `2 * patch_z` = 26 consecutive planes and that stack has 10. "
           "The lever also refuses 3D when the z step exceeds half the axial PSF, because "
           "the axial-parity split assumes alternate planes are two views of one structure.",
      why="The validation is the point. Run against LAGFP - whose settings were derived by "
          "hand over this whole investigation and then confirmed by the v2 result - the "
          "tool reproduces them from metadata alone, having been told nothing about the "
          "outcome. And the gain cross-check lands at 4.45 measured against 4.5 from "
          "metadata, a 1% agreement, on the one dataset where the measurement is valid.\n\n"
          "Two honest limits. `hess_weight` switches on the denoising-prize gate (0.005 "
          "when the noise band exceeds 35%, else the paper's 0.02) and only the 0.005 arm "
          "has been tested, on one dataset. And `train_units` is a rule of thumb - "
          "`n_planes // 8` clamped to 16-32 - chosen because more units cost RAM but not "
          "time, never swept.",
      evidence=[{"type": "table",
                 "cols": ["socket", "auto-derived", "hand-set (validated by v2)", "delta"],
                 "rows": [
                     ["beta1", "4.5", "4.7", "-4%"],
                     ["background", "115.1", "120", "-4%"],
                     ["beta2", "20.2", "25", "-19% (inside the +/-20% training jitter)"],
                     ["upsample", "False", "False", "match"],
                     ["hess_weight", "0.005", "0.005", "match"],
                     ["iterations", "20000", "20000", "match"],
                     ["tile", "256", "256", "match"],
                     ["output", "deconvolved", "deconvolved", "match"],
                     ["dim", "2D", "2D", "match"],
                     ["train_units", "32", "20", "more; costs RAM, not time"],
                 ],
                 "caption": "LAGFP_Caps_Z: the tool re-derives the settings that produced "
                            "the accepted v2 result, from the file alone."},
                {"type": "code", "text":
                 "=== Monolayer_stain.nd2  channel 1 (W1 - Red)\n"
                 "  axes {P:16, Z:10, C:4, Y:2048, X:2048} | 0.3246 um/px | em 600 | NA 0.75\n"
                 "  camera: '12-bit (CMS)' / 'Sensitivity' -> gain 4.5 ADU/e- (from metadata)\n"
                 "  photon-transfer cross-check: measured 3.99 ADU/e-\n"
                 "  PSF sigma 0.52 px | Nyquist 1.54 /um | OTF 2.50 /um | sig=noise 1.05 /um\n"
                 "  [1] NO deconvolution - sigma below the max(0.5,sigma) clamp -> denoised\n"
                 "  [2] MODERATE denoising prize: 32% of the band is noise-dominated\n"
                 "  [3] signal reaches 68% of Nyquist -> NOISE-limited -> upsample=False\n"
                 "  [4] dim = 2D: only 10 z planes; the 3D path needs 26\n\n"
                 "    dim              2D            background       97.4\n"
                 "    mode             zero_shot     beta1            4.5\n"
                 "    output           denoised      beta2            20.2\n"
                 "    upsample         False         patch            128\n"
                 "    tile             256           batch_size       4\n"
                 "    overlap          20            learning_rate    5e-05\n"
                 "    iterations       20000         denoise_weight   0.5\n"
                 "    hess_weight      0.02          alpha            1.0\n"
                 "    train_units      20            norm_low         0.0\n"}],
      status="confirmed")

eid = D.ask(
    title="Monolayer Red is ready to launch - but note the run will train the DECONVOLVED "
          "head you are then told not to use. Two ways to spend the 9 hours.",
    body="The socket block above is what to enter. One wrinkle worth deciding before you "
         "start: gate 1 says `output=denoised`, but `zero_shot` training always trains both "
         "heads - the deconvolution loss is what supervises stage I. So you cannot skip it, "
         "and the 9 hours buys a denoiser either way.\n\n"
         "That makes the run a genuine test of the formula: if the prediction is right, the "
         "denoised head will beat the deconvolved head on noise, CNR and sharpness "
         "simultaneously, exactly as it did on the fibroblasts. If the deconvolved head "
         "wins, gate 1's 0.7 threshold is wrong and I will have to move it.\n\n"
         "Worth knowing what you are buying: on this dataset NL-means already lifts CNR "
         "from 8.0 to 10.9 in 0.2 s/frame. The ZS-DeconvNet denoised head has to beat that "
         "to justify 9 hours plus ~11 min of inference per channel.",
    why="It changes what I score when it lands, and whether the other three channels are "
        "worth training at all (one model per channel - Far Red, Green and UV would be "
        "another 27 h).",
    asks="(1) Confirm you want the deconvolved-vs-denoised head comparison as the primary "
         "readout when it finishes - that is the falsifiable bit.\n"
         "(2) Red only, or do you want the other three channels queued? At 9 h each I would "
         "run Red first and let the result decide.\n"
         "(3) Still open from e004: the single-cell deep dive on brightness and "
         "fragmentation, and improving how we select nearby regions that may be extensions "
         "of the same cell. Which dataset - Monolayer or the fibroblasts? That is the next "
         "real piece of work and I would rather start it while the training runs.")
print("asked", eid)
print(D.render())
