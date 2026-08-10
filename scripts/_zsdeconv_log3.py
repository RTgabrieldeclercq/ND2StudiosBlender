"""Batch 3: corrections + the real result + the parameter recommendation."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

# ------------------------------------------------------------------ error 1
D.log(kind="error",
      title="MY ERROR - I overstated the segmentation result. I wrote 'far less "
            "fragmentation and much higher ridge contrast' from ONE frame through a "
            "threshold that had segmented almost nothing.",
      body="The earlier segmentation entry used Otsu on the Sato ridge response of frame 54 "
           "only. Looking at the figure, both masks were nearly empty - Otsu sat so high "
           "that it kept only the few brightest bundles. The numbers I quoted (93 vs 87 "
           "objects, 59.4 vs 58.2 free ends per 100 um) were describing that broken "
           "threshold, not fiber segmentation quality, and on their face they showed almost "
           "NO difference while my sentence claimed a large one.",
      why="RETRACTED: 'far less fragmentation and much higher ridge contrast'. Replaced by "
          "the 5-frame hysteresis measurement in the next entry, which does show a real but "
          "MODEST continuity gain (-23% free ends) and, importantly, NO gain in "
          "separability. I wrote the interpretation into the script before I had run it - "
          "exactly the failure this log exists to catch.",
      evidence=[{"type": "kv", "rows": [
          ["retracted claim", "'far less fragmentation, much higher ridge contrast'",
           "from 1 frame, broken Otsu"],
          ["what the numbers actually said", "59.4 vs 58.2 ends/100um",
           "a 2% difference, not 'far less'"],
          ["root cause", "Otsu on ridge response",
           "kept ~2% of pixels; the metric could not see fibers at all"],
          ["replaced by", "hysteresis (88/97 pct), 5 frames", "entry below"]]}],
      status="confirmed")

# ------------------------------------------------------------------ error 2
D.log(kind="error",
      title="MY ERROR - '3.5x short of the diffraction limit' is wrong. It is 2.0x short of "
            "the OTF cutoff, and the data is essentially AT the classical resolution limit.",
      body="The measured signal/noise crossover is 2.68 um-1. I wrote '3.5x short' as "
           "prose before the number existed. The correct comparisons: the coherent OTF "
           "cutoff is 2*NA/lambda = 5.33 um-1, so the crossover is 2.0x short of it, not "
           "3.5x. And the crossover corresponds to a 187 nm half-period, which is "
           "essentially the classical FWHM resolution of these optics (0.51*lambda/NA = "
           "191 nm).",
      why="This changes the framing in the user's favour, so it matters. The dataset is not "
          "hopelessly starved - it already resolves to about the conventional diffraction "
          "limit. What it cannot do is go BEYOND it, because the band between 2.68 and "
          "5.33 um-1 (187 nm down to 94 nm) is where super-resolution would have to come "
          "from and that band is at or below the noise floor.",
      evidence=[{"type": "kv", "rows": [
          ["measured signal = noise", "2.68 um-1", "187 nm half-period"],
          ["classical FWHM resolution", "0.51*525/1.4 = 191 nm", "the data is at this"],
          ["OTF cutoff", "2*1.4/0.525 = 5.33 um-1", "94 nm half-period"],
          ["retracted", "'3.5x short of the diffraction limit'", "correct value: 2.0x"]]}],
      status="confirmed")

# ------------------------------------------------------------------ the real result
D.log(kind="found",
      title="The real result over 5 frames: the deconvolution suppresses false ridges (-40%) "
            "and fragmentation (-23%), adds NO separability (AUC unchanged), and makes "
            "fibers 12% wider",
      body="Same Sato ridge pipeline on both, sigmas set in microns so the two grids see the "
           "same physical filter, hysteresis threshold at the 88th/97th percentile, small "
           "objects removed below 0.3 um^2. Frames 10, 30, 54, 80, 100. The AUC uses a "
           "label source common to both images (raw smoothed at sigma = 0.5 um; top 12% = "
           "fiber, bottom 45% = background) so neither input is favoured.",
      why="This is the honest verdict, and it splits three ways.\n\n"
          "**It genuinely helps.** Cutting the false-ridge ratio by 40% and free ends by 23% "
          "is exactly what fiber segmentation needs - the raw ridge response is filled with "
          "a mesh of noise-induced ridges in the dark regions, and those become spurious "
          "fragments in any mask.\n\n"
          "**It adds no information.** AUC is 0.8433 raw vs 0.8394 deconvolved - a hair "
          "WORSE. The deconvolution does not make it easier to tell fiber from background; "
          "it only makes the answer cleaner-looking. Everything it gained, a good denoiser "
          "would also gain.\n\n"
          "**It costs width.** Fibers come out 12% thicker. That is the blur you saw, and "
          "it is real, not a display artefact.",
      evidence=[{"type": "table",
                 "cols": ["metric", "RAW", "DECONV", "change", "reading"],
                 "rows": [
                     ["ridge AUC (fiber vs bg)", "0.8433", "0.8394", "-0.004",
                      "no gain - the key negative result"],
                     ["false-ridge ratio (bg/fiber)", "0.1837", "0.1095", "-40.4%",
                      "real gain, the main benefit"],
                     ["free ends / 100 um", "54.6", "42.3", "-22.6%", "real gain"],
                     ["objects (fragments)", "62.6", "49.4", "-21.1%", "real gain"],
                     ["mean fiber width (um)", "0.479", "0.537", "+12.2%",
                      "the blur, quantified"],
                 ],
                 "caption": "Means over 5 frames. Every metric moved the same direction on "
                            "every individual frame except AUC, which is flat."},
                {"type": "note", "text":
                 "**What these metrics are blind to.** None of them can see a HALLUCINATED "
                 "fiber. A network that invented a plausible filament in a dark region "
                 "would score *better* on false-ridge ratio and fragmentation, not worse. "
                 "Nothing measured here can distinguish 'cleaned up' from 'made up'. That "
                 "needs hand-drawn ground truth, which is the pending question below."}],
      images=[{"src": "img/06_seg_corrected.png",
               "caption": "The middle column is the point. Top: the raw ridge response is "
                          "filled with a mesh of noise ridges in the dark areas. Bottom: "
                          "those are gone. Then compare the two masks on the right - the "
                          "deconvolved one is continuous but visibly fatter."},
              {"src": "img/07_metrics.png",
               "caption": "Per-frame. AUC (left) is flat - that is the negative result. "
                          "False-ridge ratio (middle) and fragmentation (right) improve on "
                          "every frame."}],
      status="confirmed")

# ------------------------------------------------------------------ diagnosis
D.log(kind="found",
      title="Diagnosis: the network denoised but never deconvolved. It ran at the DEFAULT "
            "2000 iterations, and the 2x upsample carries no information at all.",
      body="`models/LAGFP_Caps_Z_c0.json` records the training signature, and that signature "
           "only stores params the user EXPLICITLY set. It contains background=120, "
           "beta1=4.7, beta2=25, train_units=20, upsample=true - and no `iterations` key, "
           "which means iterations was left at its default of 2000. The node's own socket "
           "documentation says 2000 gives 'a usable, visibly denoised and sharpened "
           "preview, NOT paper-quality super-resolution'; the paper uses 50000 for 2D.\n\n"
           "The power spectra show what that produced. Below 1.2 um-1 the raw and "
           "deconvolved curves lie exactly on top of each other - the network left the "
           "signal band untouched. Above it, the deconvolved curve simply falls away where "
           "the raw flattens onto its noise floor. There is no frequency boost anywhere, "
           "which is what a deconvolution IS. Above the raw Nyquist (4.67 um-1) the "
           "deconvolved spectrum drops to ~5e-11 and then RISES again toward 8-9 um-1: that "
           "rise is an upsampling artefact, not recovered detail.",
      why="Two consequences for the parameters.\n\n"
          "First, `upsample=True` bought nothing here. It cost 4x the pixels, 4x the "
          "inference time and an 8x larger file (1.43 GB vs 179 MB), and put zero "
          "information above the raw Nyquist. Turning it off is close to free quality-wise "
          "and is the single biggest speed lever.\n\n"
          "Second, 2000 iterations on a photon-starved dataset converges to the safest "
          "thing the loss allows, which is a smoothed estimate - that is precisely the +12% "
          "fiber width. The Hessian regulariser (0.02) is also still fully active with a "
          "network that has not yet learned to sharpen.",
      evidence=[{"type": "code", "text":
                 '{\n "__inference__": {"arch": "unet2d", "dim": "2D",\n'
                 '                    "insert_xy": 16, "upsample": true},\n'
                 ' "arch": "unet2d",\n "background": 120.0,\n "beta1": 4.7,\n'
                 ' "beta2": 25.0,\n "dim": "2D",\n "psf_shape": [9, 9],\n'
                 ' "psf_sum2": 0.037088376,\n "train_units": 20.0,\n "upsample": true\n}\n'
                 '   <- no "iterations" key => it ran at the default 2000'},
                {"type": "kv", "rows": [
                    ["psf_sum2 0.037088", "=> sigma 1.465 px", "exactly 2x the 0.735 px "
                     "diffraction sigma"],
                    ["why 2x", "PSF sampled on the UPSAMPLED grid (0.0536 um/px)",
                     "correct - the loss convolves at 2x then downsamples"],
                    ["power ratio 0.5-1.0 um-1", "1.10x", "negligible sharpening"],
                    ["power ratio 2.0-3.0 um-1", "0.010x", "100x suppression"],
                    ["above raw Nyquist", "artefact rise at 8-9 um-1", "not information"]]}],
      images=[{"src": "img/02_psd.png",
               "caption": "Left panel: the orange (deconvolved) curve tracks the blue (raw) "
                          "exactly below 1.2/um, then falls away. It never rises above the "
                          "raw anywhere - no deconvolution happened. Note the artefact bump "
                          "at 8-9/um, past the raw Nyquist."},
              {"src": "img/01_visual.png",
               "caption": "Bottom row, same physical scale: the deconvolved zoom (centre) is "
                          "smoother than the raw (left) but the fibers are not thinner or "
                          "better separated - they are softer."}],
      status="confirmed")

print(D.render())
