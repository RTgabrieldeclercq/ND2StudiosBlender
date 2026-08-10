# §7c. Calibration describes the CURRENT data, not the file (2026-07-28)

*A leaf of the **wire-node-v2** skill. Cited as `§7c`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


**Every node reads its INPUT envelope, so calibration is always the most up-to-date value
— which means a node that changes what its numbers MEAN must restamp the affected key in
lockstep.** This is the same law as §8 (an axis-changing node restamps pixel size); it
just also applies to keys that are not geometry. Read it as: *the envelope is a running
description of the data on this wire, never a record of the acquisition.*

**`bit_depth` is the intensity-domain instance.** It is a calibration key (ND2
`bitsPerComponentSignificant` — most files are **12**-bit, not 16), so:

- A node that **widens the value scale** restamps it. Summing `n` samples of a `b`-bit
  signal needs `b + ceil(log2 n)` bits → `metadata.bit_depth_after_sum(env, n)`, used by
  `z_project`/`stack_time` for their `sum` combiners only (mean/median/max/min stay inside
  the input range). A 12-bit series summed over T=8 becomes 15-bit *for every node after
  it*, and chains (`sum` → `sum` → 17-bit).
- A node whose output **is no longer integer counts** DROPS it: `metadata.value_rescaled`
  is the ready-made meta_transform (percentile `enhance.normalize` → `[0,1]` floats).
  Absent `bit_depth` is the honest signal "no declared integer scale", and consumers must
  handle it: `analysis.threshold`'s fixed level derives mid-range from the depth and falls
  back to `0.5` without one; `analysis.histogram_threshold` refuses `[0,1]` input outright.
- A node that only **redistributes** intensities inside the same range does NOT restamp —
  `enhance.clahe` rescales its equalized result back to the input's `[min,max]`, and γ
  (`(a/mx)**g * mx`) preserves the plane max. Check the backend's actual output range
  before deciding (skimage's `equalize_adapthist` returns `[0,1]`; the node's own
  post-scaling is what saves it).

**Both halves, as always.** Declare the `meta_transform` so *edit-time* propagation
predicts the new value (the GUI's widget re-seed and every `derive` read it before any
pull), and stamp the payload in the compute so the pulled Dataset agrees. And per §8, do
NOT re-derive a relative change in the compute — `ctx.calib` already returns the
**post-transform** value, so sync the payload to it (`bd_out = ctx.calib("bit_depth")`)
or a re-pull would widen twice.

**Consumers read it through `ctx.calib("bit_depth")`**, which memo-fences the read — so
re-ingesting the same file at another depth, or flipping an upstream combiner to `sum`,
invalidates exactly the nodes that cared.
