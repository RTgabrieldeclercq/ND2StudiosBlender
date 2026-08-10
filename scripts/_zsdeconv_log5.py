"""Batch 5: resolve e010/e001, add the Fibro_test four-method comparison."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

D.confirm("e010", note="Picked A - the 20000-iteration retrain is already running. Also "
                       "asked to bring Fibro_test.tif in as a second, real example across "
                       "all four methods before any default is set.")
D.confirm("e001", note="Superseded as the working example by Fibro_test.tif at the user's "
                       "request. The pairing check itself stands and still underwrites the "
                       "LAGFP measurements in e006-e009.")

D.log(kind="error",
      title="MY ERROR - my first Fibro_test noise numbers compared normalizations, not "
            "noise. ZS-DeconvNet is not the quietest; it is the noisiest.",
      body="`infer_2d` already percentile-normalizes its output, so the ZS result arrived "
           "on a min-max scale set by a handful of 4095-count dust specks, while the raw "
           "and the classical methods were on a 0.5-99.7 percentile scale. I measured "
           "background sigma across those different scales and got 0.01020 for "
           "ZS-DeconvNet against 0.01652 for raw - which I was about to report as 'best "
           "noise suppression, -38%'. Putting every method on one common scale first "
           "reverses it: 0.01917 for ZS-DeconvNet, the WORST of the five, noisier than the "
           "untouched raw.",
      why="RETRACTED before it reached you, but it is logged because the trap is generic "
          "and will recur: any node that normalizes its own output cannot be compared to "
          "an un-normalized one on any absolute-intensity metric. CNR and the 10-90% rise "
          "distance are ratios and were unaffected - only the noise figure moved.",
      evidence=[{"type": "kv", "rows": [
          ["wrong (mixed scales)", "ZS bgnoise 0.01020 vs raw 0.01652", "would read as -38%"],
          ["correct (common scale)", "ZS bgnoise 0.01917 vs raw 0.01652", "+16%, worse"],
          ["unaffected", "CNR, 10-90% rise", "both are ratios"],
          ["fix", "n01() applied to every method before any metric", "in the script"]]}],
      status="confirmed")

D.log(kind="found",
      title="Fibro_test.tif, all methods: a PROPERLY trained ZS-DeconvNet (33000 iterations) "
            "still blurs by 2%, and its DENOISED head beats its own deconvolved head on "
            "every metric",
      body="`Fibro_test.tif` is a single 1024x1024 frame at **1.7183 um/px** - a 1760 um "
           "field of roughly a thousand fibroblast cell bodies. The repo already had "
           "`models/Fibroblast_10x_c0`, trained at **33000 iterations, upsample off, 98 "
           "train units**, and its recorded `psf_sum2` of 0.411339 matches this pixel size "
           "exactly, so it is the right model for this image. That is the properly-trained "
           "data point the LAGFP comparison was missing.\n\n"
           "Sharpness is the 10-90% intensity rise distance across ~340 strong cell "
           "boundaries, measured along the local gradient - scale-invariant, so the "
           "normalization trap above does not touch it.",
      why="Three things follow, and the first is the one that matters for your retrain.\n\n"
          "**More iterations did not remove the blur.** At 33000 iterations the deconvolved "
          "head is still the blurriest of the five (4.37 um rise vs 4.28 raw). On the LAGFP "
          "data at 2000 iterations I attributed the +12% fiber width mainly to "
          "undertraining. This is evidence against that being the whole story - though see "
          "the caveat, because the two datasets differ in a way that matters.\n\n"
          "**Stage II actively degrades this image.** The denoised head is better than the "
          "deconvolved head on noise, CNR and sharpness simultaneously. If you use "
          "ZS-DeconvNet at this scale, `output = denoised` is strictly the better socket.\n\n"
          "**And there is a mechanical reason.** At 1.7183 um/px the true diffraction sigma "
          "is 0.04-0.21 px, far below one pixel, so `gaussian_psf_from_sigmas` returns its "
          "`max(0.5, sigma)` CLAMP rather than the optics. The degradation term in the loss "
          "is then almost the identity, leaving stage II trained against little more than "
          "the Hessian smoothness penalty. There is no optical blur to remove at this "
          "sampling, so a deconvolution can only cost you.",
      evidence=[{"type": "table",
                 "cols": ["method", "bg noise (lower)", "CNR (higher)",
                          "10-90% rise um (lower=sharper)", "seconds"],
                 "rows": [
                     ["raw", "0.01652", "20.1", "4.28", "0"],
                     ["NL-means", "0.01499", "22.3", "4.08  (best)", "1.6"],
                     ["NL-means + RL(10)", "0.01730", "19.3", "4.12", "1.4"],
                     ["ZS-DeconvNet denoised head", "0.01561", "21.4", "4.19", "19.7"],
                     ["ZS-DeconvNet 33k, deconvolved", "0.01917", "16.7", "4.37 (worst)", "19.7"],
                 ],
                 "caption": "All five on one common intensity scale. NL-means is sharpest, "
                            "quietest-but-one, highest CNR, and 12x faster."},
                {"type": "note", "text":
                 "**The caveat that stops this generalising, and it is a big one.** This "
                 "image is 16x coarser than the LAGFP actin data (1.7183 vs 0.107 um/px) "
                 "and the objects are whole cell bodies, not actin filaments. Crucially the "
                 "PSF here is the 0.5 px CLAMP, whereas on the LAGFP data it is a genuine "
                 "1.465 px sigma on the upsampled grid. So this result shows what "
                 "ZS-DeconvNet does when there is no blur to remove; it does NOT predict "
                 "what your 20000-iteration LAGFP run will do, where there is real optical "
                 "blur for stage II to work against. **A default tuned on this frame must "
                 "not be carried over to the 60x data, or the reverse.**"}],
      images=[{"src": "img/11_fibro_four.png",
               "caption": "Bottom row, 220 um. Columns 2-3 (NL-means, +RL) keep the crisp "
                          "cell edges; column 5 (deconvolved head) has visibly softer "
                          "boundaries and flatter cell interiors than column 4 (its own "
                          "denoised head). All five are on one display scale."},
              {"src": "img/10_fibro_raw.png",
               "caption": "What the frame is: ~1000 spindle-shaped fibroblasts over a "
                          "1760 um field. The elongated objects are CELL BODIES, tens of "
                          "microns across - not the sub-micron actin filaments of the "
                          "LAGFP dataset."},
              {"src": "img/12_fibro_psd.png",
               "caption": "The deconvolved head (red) sits below the others across the "
                          "whole structure band - it removed power everywhere rather than "
                          "boosting any of it."}],
      status="confirmed")

eid = D.ask(
    title="Your reason for picking A was that ZS-DeconvNet's resolution beats conventional "
          "deconvolution - but every sharpness measurement I have says the opposite. Worth "
          "reconciling before the retrain finishes.",
    body="You wrote that you can see it has potential 'because the resolution is better "
         "than the conventional deconvolve method'. Both instruments I have disagree:\n\n"
         "- **LAGFP actin**: mean fiber width 0.553 um for ZS-DeconvNet vs **0.479 um** for "
         "NL-means+RL - the classical route was 13% thinner, and identical to the raw.\n"
         "- **Fibro_test**: 10-90% edge rise 4.37 um for ZS-DeconvNet (33k iterations!) vs "
         "**4.12 um** for NL-means+RL and 4.28 um for the raw.\n\n"
         "Both say ZS-DeconvNet is the softer of the two. I would rather find out I am "
         "measuring the wrong thing than have you choose on a number I got wrong.\n\n"
         "The most likely honest explanation is that we mean different things by "
         "'resolution'. ZS-DeconvNet produces a far cleaner, more continuous image, and a "
         "clean image genuinely reads as better resolved - it is also the one metric where "
         "it won outright on the actin data (-40% false ridges, -23% fragmentation). "
         "'Conventional deconvolve' may also mean the `enhance.deconvolve` node at its own "
         "defaults rather than the NL-means+RL chain I built, and those are not the same "
         "thing - plain Richardson-Lucy on raw noisy data amplifies noise badly.",
    why="If you are seeing sharper structure somewhere I am not measuring, my instrument is "
        "wrong and so is the recommendation that follows from it. If we just mean different "
        "things by resolution, then the retrain should be judged on continuity, not "
        "sharpness - and I should score it that way when it lands.",
    asks="Two things. (1) Which comparison were you looking at - the `enhance.deconvolve` "
         "node, or something else? Point me at a region and I will measure that exact spot "
         "both ways. (2) Confirm the retrain's settings, since it is already running: did "
         "you set `upsample=False`, `iterations=20000`, `hess_weight=0.005`, tile 256? If "
         "`upsample` is still on, it is costing 1.5x training time and 3.8x inference for a "
         "grid we showed carries no information.",
    images=[{"src": "img/11_fibro_four.png",
             "caption": "Column 3 vs column 5 is the disagreement in one picture. Tell me "
                        "which one you read as better resolved, and why - that is the "
                        "fastest way to find out whose instrument is wrong."}])
print("asked", eid)
print(D.render())
