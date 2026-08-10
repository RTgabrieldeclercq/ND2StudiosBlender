"""Batch 6: the v2 re-score - the answer to option A."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

D.log(kind="found",
      title="Option A answered. The retrain is a 30x SPEEDUP and the best fiber continuity "
            "of any method tested - but 10x the iterations did not close the separability "
            "gap, and it is still 11% blurrier than the raw.",
      body="`models/LAGFP_Caps_Z_c0.json` confirms the run used exactly the recommended "
           "settings: `upsample=false`, `iterations=20000`, `hess_weight=0.005`, with "
           "background 120 / beta1 4.7 / beta2 25 / train_units 20 carried over. "
           "`psf_sum2` is 0.150276, i.e. sigma 0.728 px - the correct NATIVE-grid "
           "diffraction sigma now that upsampling is off. Output is 114x512x512 float32 at "
           "0.107136 um/px, on the same grid as the raw, and 358 MB instead of 1.43 GB.\n\n"
           "Re-scored on the identical five frames, label source, threshold rule and "
           "metrics as e006/e008, with every method put on one common intensity scale "
           "first.",
      why="**What the parameter changes bought.** The speed result is unambiguous and large: "
          "131.6 to 4.3 seconds per frame, a 30x speedup, while every quality metric moved "
          "in the right direction. Turning off `upsample` was free quality and is now "
          "settled. At 4.3 s/frame the full 725-frame ND2 becomes about 52 minutes of "
          "inference instead of 26 hours - this is the difference between a usable node and "
          "an unusable one.\n\n"
          "**What it did not buy.** AUC went 0.8276 to 0.8287, still flat and still BELOW "
          "the untouched raw (0.8423). Ten times the iterations and a four-times-lower "
          "Hessian weight did not make the network any better at telling fiber from "
          "background. The blur narrowed but did not close: fiber width 0.553 to 0.531 um, "
          "against 0.480 um for both the raw and NL-means+RL. So your original observation "
          "stands after the retrain - it is still the softer image.\n\n"
          "**Where it now wins outright.** 36.3 free ends per 100 um is the best "
          "fragmentation figure of anything tested, better than NL-means+RL's 42.6. If what "
          "you need is fibers that survive as single connected objects for tracing, v2 is "
          "the best input available.",
      evidence=[{"type": "table",
                 "cols": ["method", "AUC (higher)", "false-ridge (lower)",
                          "ends/100um (lower)", "width um (lower)", "s/frame"],
                 "rows": [
                     ["raw", "0.8423", "0.1827", "54.1", "0.480", "0"],
                     ["NL-means", "0.8875", "0.0542", "40.9", "0.566", "0.23"],
                     ["NL-means + RL(10)", "0.8988 (best)", "0.0578", "42.6", "0.480 (best)", "0.18"],
                     ["ZS v1 (2k it, upsample)", "0.8276", "0.1075", "39.4", "0.553", "131.6"],
                     ["ZS v2 (20k it, no upsample)", "0.8287", "0.1009", "36.3 (best)", "0.531", "4.3"],
                 ],
                 "caption": "v2 beats v1 on every column and is 30x faster. It still trails "
                            "NL-means+RL on separability and sharpness, and leads it on "
                            "continuity."},
                {"type": "kv", "rows": [
                    ["v1 -> v2 AUC", "0.8276 -> 0.8287", "+0.001, no change"],
                    ["v1 -> v2 width", "0.553 -> 0.531 um", "-4%, still +11% vs raw"],
                    ["v1 -> v2 ends/100um", "39.4 -> 36.3", "-8%, now best overall"],
                    ["v1 -> v2 speed", "131.6 -> 4.3 s/frame", "30x, the headline result"],
                    ["full 725-frame ND2", "26 h -> 52 min", "what makes it usable"]]}],
      images=[{"src": "img/13_rescore.png",
               "caption": "Column 5 (v2) against column 4 (v1): the fibers are visibly "
                          "thinner and better separated, and the mask at the bottom is both "
                          "more continuous and less bloated. The retrain is a real "
                          "improvement over v1 - it just does not overtake column 3."},
              {"src": "img/14_rescore_psd.png",
               "caption": "Left: v2 (purple) sits well above v1 (red) through the mid "
                          "frequencies - it holds signal to ~1.7/um where v1 gave up at "
                          "~1.2/um. That is the width improvement, in the frequency domain. "
                          "Both still fall below the raw and NL-means+RL."}],
      status="confirmed")

eid = D.ask(
    title="v2 is the right ZS-DeconvNet configuration - now which do you want as the "
          "default, and for which job?",
    body="The retrain settled the parameter question. `upsample=False`, "
         "`iterations=20000`, `hess_weight=0.005`, tile 256, background 120, beta1 4.7, "
         "beta2 25, train_units 20 is the configuration to keep, and I would write those "
         "into the node's docstring citing this entry.\n\n"
         "What it did not settle is which tool wins, because the two candidates are now "
         "genuinely good at different things:\n\n"
         "- **ZS-DeconvNet v2** - best fiber CONTINUITY (36.3 ends/100 um). Choose it if "
         "the downstream step traces or tracks fibers and broken filaments are the failure "
         "mode. 4.3 s/frame, ~52 min for the full ND2.\n"
         "- **NL-means + RL(10)** - best SEPARABILITY (AUC 0.8988) and sharpest (0.480 um, "
         "equal to raw). Choose it if the downstream step measures fiber width, orientation "
         "or spacing, where an 11% width inflation is a systematic error. 0.18 s/frame, "
         "~2 min for the full ND2.\n\n"
         "There is also a real hybrid: run the segmentation on v2 for connectivity, then "
         "measure the resulting fibers on the RAW pixels, so the mask comes from the clean "
         "image and the numbers come from the unmodified data.",
    why="This decides what I wire as the default enhancement chain and what goes in the "
        "node docs. It depends entirely on what the fiber segmentation feeds into, which "
        "you know and I do not.",
    asks="What happens downstream of the fiber mask - tracing/tracking, or measuring "
         "width/orientation/density? If it is measurement, the 11% width inflation matters "
         "and NL-means+RL wins. If it is tracing, v2 wins.\n\n"
         "And the offer still stands: ~15 minutes drawing fiber outlines on a handful of "
         "tiles would let me score all five against real boundaries instead of against each "
         "other. Every number in this log is still blind to a hallucinated fiber, and v2 - "
         "the smoothest, most continuous output - is exactly the one where that risk is "
         "highest.")
print("asked", eid)
print(D.render())
