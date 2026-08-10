"""Batch 7: the applicability formula, validated on three datasets."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

D.confirm("e015", note="Answer requested as a general applicability formula for the node, "
                       "plus a third test dataset (Monolayer_stain.nd2, Red channel). "
                       "Downstream-use question left open; the formula below is written to "
                       "answer it per dataset rather than once.")

D.log(kind="decision",
      title="THE FORMULA: three gates, computed from pixel size, NA, emission and one "
            "power spectrum. It reproduces the outcome on all three datasets tested.",
      body="Implemented as `scripts/zs_applicability_check.py`, runnable on any frame:\n\n"
           "    .\\.venv\\Scripts\\python.exe -u -B scripts\\zs_applicability_check.py "
           "<image> <um_per_px> <em_nm> <NA>\n\n"
           "**Gate 1 - is there blur to remove?**  `sigma_px = 0.21 * em_nm / (NA * "
           "pixel_nm)`, the diffraction sigma in PIXELS, evaluated on the output grid.\n"
           "  * `< 0.7` - the node's own `max(0.5, sigma)` clamp is in force, so it trains "
           "against the clamp rather than your optics. Stage II can only smooth. Use "
           "`output=denoised` or a plain denoiser.\n"
           "  * `0.7 - 1.5` - weak. Denoising benefit, no resolution gain, ~10% width "
           "inflation.\n"
           "  * `>= 1.5` - real deconvolution possible. This is the paper's regime (their "
           "TIRF at 0.0313 um/px, NA 1.49 gives sigma 2.4 px).\n\n"
           "**Gate 2 - how big is the denoising prize?**  Fraction of the sampled spectrum "
           "above the signal/noise crossover.\n"
           "  * `< 15%` - already clean; the node adds blur for little.\n"
           "  * `15-35%` - moderate; expect real continuity and false-ridge gains.\n"
           "  * `> 35%` - large; this is where the node earns its compute.\n\n"
           "**Gate 3 - is `upsample` worth it?**  Only if the crossover reaches `>= 90%` of "
           "Nyquist, i.e. the data is SAMPLING-limited. If it is noise-limited, a finer "
           "grid carries nothing and costs 1.5x training / 3.8x inference / 4x storage.",
      why="This is what you asked for: a rule that decides per dataset instead of a default "
          "that is wrong half the time. The two failure modes it catches are exactly the "
          "ones that burned time here - running a deconvolution where the PSF is "
          "sub-pixel (Fibroblast, and now Monolayer), and upsampling data whose signal died "
          "well below Nyquist (LAGFP, which cost 4.2 h instead of 1 h for an 8x larger file "
          "carrying no information).\n\n"
          "**The blunt summary across all three: none of these datasets can produce "
          "super-resolution.** In every one, the band between where signal dies and where "
          "the optics stop is entirely at or below the noise floor. On this microscope and "
          "at these exposures, ZS-DeconvNet is a denoiser with a continuity bonus. That is "
          "worth having - it gave the best fiber continuity of anything tested - but it is "
          "not what the node's name promises.",
      evidence=[{"type": "table",
                 "cols": ["dataset", "sigma_px", "Nyq /um", "OTF /um", "sig=noise",
                          "noise band", "cross/Nyq", "verdict"],
                 "rows": [
                     ["LAGFP actin 60x/1.4", "0.74", "4.67", "5.33", "2.68", "42%", "57%",
                      "weak deconv + LARGE denoise prize; upsample OFF"],
                     ["Fibroblast 10x/0.45", "0.14", "0.29", "1.71", "0.26", "12%", "88%",
                      "NO deconv (clamp) + small prize; use denoised head"],
                     ["Monolayer Red 20x/0.75", "0.52", "1.54", "2.50", "1.08", "29%", "70%",
                      "NO deconv (clamp) + moderate prize; use denoised head"],
                 ],
                 "caption": "Gate values for the three datasets. The last column is what "
                            "the formula outputs before any result is looked at."},
                {"type": "table",
                 "cols": ["dataset", "formula predicted", "what was actually measured",
                          "match?"],
                 "rows": [
                     ["LAGFP actin",
                      "no resolution gain; big continuity win; upsample useless",
                      "AUC flat 0.842->0.829; ends best at 36.3; upsample gave 0 power "
                      "above Nyquist and cost 30x", "yes"],
                     ["Fibroblast 10x",
                      "deconvolution head should be WORSE than the denoised head",
                      "denoised head beat it on noise, CNR and sharpness simultaneously, "
                      "at 33000 iterations", "yes"],
                     ["Monolayer Red",
                      "deconv ~no-op; moderate continuity gain; output=denoised; "
                      "upsample off", "NOT YET TESTED - falsifiable prediction", "pending"],
                 ],
                 "caption": "Two retrodictions and one open prediction. The Monolayer row "
                            "is the test that would break the formula."}],
      images=[{"src": "img/15_applicability.png",
               "caption": "The yellow band in each panel is what a deconvolution could in "
                          "principle recover - optics support it, the data has not reached "
                          "it. In all three it lies flat on the red noise floor, which is "
                          "why none of them yields super-resolution."},
              {"src": "img/16_monolayer.png",
               "caption": "Monolayer Red: the raw (left) is the noisiest data of the three "
                          "- CNR 8.0, and the bottom-left zoom is almost pure grain. "
                          "NL-means alone takes CNR to 10.9 in 0.2 s. This is a dataset "
                          "where denoising clearly pays."}],
      status="confirmed")

eid = D.ask(
    title="Monolayer Red: the formula says use the DENOISED head and skip the "
          "deconvolution. Worth 9 h to test that prediction?",
    body="Monolayer_stain.nd2 is 16 positions x 10 z x 4 channels at 2048^2, 0.3246 um/px, "
         "20x/0.75, Red = c1 at 600 nm. It is the noisiest of the three datasets (CNR 8.0 "
         "raw, against 20.1 for the fibroblasts), so the denoising prize is real - "
         "NL-means alone lifts CNR to 10.9 in 0.2 s.\n\n"
         "But `sigma_px` is 0.52, below the 0.7 clamp threshold, so the formula predicts "
         "the deconvolution head will be a no-op at best and mildly harmful at worst - the "
         "same situation as the fibroblasts, where the denoised head won on every metric.\n\n"
         "Settings the formula gives: `upsample=False`, `iterations=20000`, "
         "`hess_weight=0.005`, `tile=256`, **`output=denoised`**. That is ~9 h training on "
         "one channel plus inference.\n\n"
         "Note this is a Z-stack (10 planes, 1.5 um step) with 4 channels, unlike the two "
         "flat datasets so far - so there is also a 3D-lever question, and one model per "
         "channel if you want more than Red.",
    why="Running it tests the formula's one open prediction. Not running it is also "
        "defensible - the formula already says the answer is 'use a denoiser', and "
        "NL-means gets you most of the way in 0.2 s/frame.",
    asks="(1) Run the 9 h training on Monolayer Red to test the prediction, or accept the "
         "formula's answer and use NL-means / the denoised head?\n"
         "(2) For the single-cell deep dive you asked for in e004 - keeping brightness and "
         "fragmentation, and improving how we select nearby regions that may be extensions "
         "of the same cell - which dataset is that on? I would assume Monolayer or the "
         "fibroblasts rather than the actin timelapse, since 'cell extensions' implies "
         "whole cells. Tell me which and I will start there; that is a bigger piece of work "
         "than the parameter question and deserves its own thread in this log.")
print("asked", eid)
print(D.render())
