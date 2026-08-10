"""Batch 4: the alternatives result, the cost model, and the decision to put to the user."""
from __future__ import annotations
import sys
sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D

D.configure("CodeLog/live/zsdeconv-fiber",
            title="ZS-DeconvNet for fiber segmentation - raw vs deconvolved")

D.log(kind="found",
      title="A 0.3-second classical pipeline beats this 132-second-per-frame ZS-DeconvNet "
            "run on almost every segmentation metric, including sharpness",
      body="Same ridge pipeline, same five frames, same threshold rule, same common label "
           "source. NL-means denoising followed by 10 Richardson-Lucy iterations against "
           "the same diffraction Gaussian PSF gives the best separability of anything "
           "tested (AUC 0.9002 vs 0.8283), a lower false-ridge ratio (0.058 vs 0.109), and "
           "fibers that are exactly as thin as the raw (0.479 um, versus 0.553 um for "
           "ZS-DeconvNet). ZS-DeconvNet wins on one metric only: fragmentation, 39.4 free "
           "ends per 100 um against 44.1.",
      why="This reframes the question you asked. Tuning ZS-DeconvNet's parameters is worth "
          "doing only if it can clear a bar that a two-second pipeline already clears. As "
          "run, it does not - its AUC is below even the untouched raw image. For the goal "
          "of segmenting fibers, the cheap route currently wins.",
      evidence=[{"type": "table",
                 "cols": ["method", "AUC (higher)", "false-ridge (lower)",
                          "ends/100um (lower)", "width um (lower)", "s/frame"],
                 "rows": [
                     ["raw", "0.8433", "0.1837", "54.6", "0.479", "0"],
                     ["gaussian 0.5 px", "0.8430", "0.1652", "48.1", "0.506", "0.01"],
                     ["TV chambolle", "0.8741", "0.0995", "45.6", "0.529", "0.24"],
                     ["NL-means", "0.8881", "0.0549", "40.8", "0.565", "0.44"],
                     ["NL-means + RL(10)", "0.9002", "0.0576", "44.1", "0.479", "0.29"],
                     ["ZS-DeconvNet (this run)", "0.8283", "0.1088", "39.4", "0.553", "131.6"],
                 ],
                 "caption": "Means over frames 10/30/54/80/100. Best AUC and best width "
                            "both belong to NL-means + Richardson-Lucy."},
                {"type": "note", "text":
                 "**Three things this comparison is blind to, stated plainly.**\n\n"
                 "1. **It is not a fair test of the METHOD, only of THIS RUN.** ZS-DeconvNet "
                 "ran at its default 2000 iterations, which its own documentation calls a "
                 "preview. The classical methods are at sensible settings. A properly "
                 "trained network may well overtake them - that is the pending question.\n\n"
                 "2. **The AUC label source is derived from the raw image** (raw smoothed at "
                 "sigma 0.5 um). Methods that stay close to the raw may be flattered by "
                 "this. The false-ridge, fragmentation and width metrics do not depend on "
                 "it, and they agree, but the AUC column deserves that asterisk.\n\n"
                 "3. **Nothing here can see a hallucinated fiber.** A network that invents "
                 "a plausible filament scores BETTER on every metric in this table. Only "
                 "hand-drawn ground truth can catch that."}],
      images=[{"src": "img/09_alt_visual.png",
               "caption": "Bottom row is what the segmentation consumes. Columns 2 and 3 "
                          "(NL-means, NL-means+RL) have the darkest backgrounds and the "
                          "crispest fibers; column 4 (ZS-DeconvNet) still shows a faint "
                          "mesh in the dark areas and visibly fatter ridges."},
              {"src": "img/08_alternatives.png",
               "caption": "Red is ZS-DeconvNet. It is last on AUC and second-worst on width, "
                          "and only wins the fragmentation panel."}],
      status="confirmed")

D.log(kind="found",
      title="Measured cost model: turning OFF upsample makes training 1.5x and inference "
            "3.8x faster, for output we showed carries no information anyway",
      body="Timed on this machine with the real planes, patch 128, insert_xy 16, batch 4.",
      why="This is the cheapest quality-neutral change available. The 2x upsampled grid was "
          "shown to hold no signal above the raw Nyquist - only an artefact bump at 8-9 "
          "um-1 - so the 4x pixels, 3.8x inference time and 8x file size (1.43 GB vs 179 MB) "
          "buy nothing. It also makes a long training run affordable: 20000 iterations "
          "without upsample costs about what 9 hours overnight can absorb.",
      evidence=[{"type": "table",
                 "cols": ["setting", "s / training iter", "2000 it", "10000 it", "20000 it",
                          "inference, 114 frames"],
                 "rows": [
                     ["upsample=True, tile 128", "2.44", "81 min", "6.8 h", "13.6 h", "31 min"],
                     ["upsample=False, tile 256", "1.63", "54 min", "4.5 h", "9.1 h", "8 min"],
                 ],
                 "caption": "The bottom row is the recommended configuration."},
                {"type": "note", "text":
                 "My benchmark predicts about 1.9 h for the run you did (81 min training + "
                 "31 min inference), against the ~15000 s (4.2 h) you observed. I have not "
                 "explained that gap - it could be machine load, a different `tile`, or the "
                 "training-unit reads. I am flagging it rather than pretending the model is "
                 "complete."}],
      status="confirmed")

eid = D.ask(
    title="Which way do you want to go? The evidence points away from ZS-DeconvNet for this "
          "dataset, but there is one test that would settle it.",
    body="Everything the sidecar did not record was left at default, and that includes the "
         "big one: `iterations` sat at 2000. The parameters you and I set (background 120, "
         "beta1 4.7, beta2 25, train_units 20) all came through correctly and are, as far "
         "as I can measure, right. What went wrong is not a bad parameter - it is that a "
         "2000-iteration network on a photon-limited dataset converges to a smoothed "
         "estimate, which is exactly the +12% fiber width you saw as blur.\n\n"
         "**Option A - settle it properly.** Retrain with `upsample=False`, "
         "`iterations=20000`, `hess_weight=0.005`, tile 256. That is ~9 h of training plus "
         "8 min of inference, so one overnight run, and it is the only way to know whether "
         "ZS-DeconvNet can beat NL-means+RL here. I would re-score it against the exact "
         "same five frames and metrics.\n\n"
         "**Option B - take the cheap win now.** Use NL-means + Richardson-Lucy as the "
         "segmentation front end. It already scores better on separability and equal on "
         "sharpness, runs in 0.3 s/frame, and would process all 725 frames of the full ND2 "
         "in about 4 minutes instead of 26 hours.\n\n"
         "**Option C - both.** Ship B now so the fiber work is unblocked, run A overnight "
         "as the check.",
    why="Whichever you pick changes what I build next: A means a training run and a "
        "re-score, B means wiring a classical enhancement chain into the graph.",
    asks="Pick A, B or C. Also: do you want to draw ground-truth fiber outlines on a few "
         "tiles? It is the ONLY way to catch hallucinated fibers - every metric in this log "
         "rewards a network that invents a plausible filament. About 15 minutes of drawing "
         "would give all of these numbers a denominator they currently lack.",
    images=[{"src": "img/06_seg_corrected.png",
             "caption": "What ZS-DeconvNet did buy: the noise mesh in the raw ridge "
                        "response (top middle) is gone (bottom middle). That gain is real - "
                        "it is just also available from a 0.4 s denoiser."}])

print("asked", eid)
print(D.render())
