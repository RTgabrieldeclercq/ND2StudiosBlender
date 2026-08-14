# Candidate recipe — deconv + fusion cell segmentation (HK-class IF monolayers)

**Status: CANDIDATE** (adopted by user decision on e027, 2026-08-09). Confirmed in-sample
on one field; awaiting (a) out-of-sample scoring on the incoming annotated files and
(b) the boundary-anchor verdict from the hand-drawn GT page (`gt/index.html`).

Applies to: fixed IF monolayers, whole-cell stain + DAPI, ~0.1–0.35 µm/px, NA 0.75–0.8
widefield/EDF. Everything below cites the evidence-log entry that established it
(`CodeLog/live/cellsam-custom-model/`).

## Pipeline

1. **Enhance — `enhance.zs_deconvnet`** (mode `zero_shot`, 2D, output `deconvolved`)
   on the whole-cell channel. Parameters by the auto-fill formula (e020,
   `zs_autofill_prototype.py`):
   - `emission_nm`, `na` ← ND2 metadata; `psf_pixel_um` 0 (derive PSF)
   - `background` ← ~p1 of the channel (pedestal)
   - `beta1` ← camera gain ADU/e⁻, `beta2` ← read-noise variance counts²
     (photon-transfer / spec; defaults 1.0/0.0 acceptable but see the stipple caveat)
   - `iterations` ← time budget ÷ ~1.37 s/step on this CPU (e022); 23 000 ≈ 8.4 h.
     **Set `cache_path`** — the trained model is reusable, retraining is not needed
     per run (e025).
   - Output: 2× lateral grid, `pixel_size_um` halved. Keep it — the 2× grid is part
     of the detection gain (e026: 1×-grid control scored 0.969 vs 0.990).
2. **Fusion input** — DAPI (bicubic 2× to match) in CellSAM's *nuclear* slot +
   deconvolved whole-cell channel in the *whole-cell* slot.
   **Not yet expressible in the node graph** — `analysis.segment` deliberately does not
   expose multi-channel fusion (Label-domain channel-key decision pending; see
   `cellsam_segment.md` §6). Until the node work lands, the kernel-level path in
   `scratchpad/compare_deconv.py::run()` is the reference implementation.
3. **Segment — CellSAM `cellsam_extra`**, thresholds `bbox 0.4 / mask_threshold 0.5 /
   mask_quality 0.8` for deconvolved input (e026; raw-input optimum is `bbox 0.3`,
   e023). `remove_boundaries=True` (GT gap convention). One model pull per process.
4. **Post-rules** (from the blind labels, e019): exclude border-touching unmatched
   objects and objects < 1 µm²; both are user-verified artifact classes.

## Measured performance (2025 field, 42-cell GT v2, in-sample)

| input | corrected F1 | plain F1@0.5 | boundary med | miss/merge/split |
|---|---|---|---|---|
| raw fusion @ best | 0.990 | 0.854 | 0.767 µm | 1 / 1 / 2 |
| **deconv fusion @ best** | **0.990** | **0.884** | 0.842 µm | **0 / 0 / 0** |

Boundary-vs-GT favors raw — but GT was drawn on the raw (PSF-dilated) image; the
hand-drawn-on-sharpened page resolves which anchor is right (e026/e027).

## EDF / processed-input branch (added after e037–e039)

The HK "raw" files are **EDF composites**, not camera frames — the Poisson–Gaussian
noise model is mis-specified for them (photon-transfer on the image returns unphysical
betas). For EDF/processed inputs the formula switches to: `beta1 0.65, beta2 0,
hess_weight 0.08, iterations ≤ 6000` (detect via `'EDF'` in the filename or a
noise-slope estimate ≪ 1; see `zs_autofill_prototype.py`).

Validated on the 2025 field (e039): the v2 model cuts stipple ~40% (band-pass energy
16.9 → 10.9×10⁻³; raw = 5.7) and gives the best boundary agreement of any variant
(0.718 µm, F1@0.75 0.688), at a ~1-cell detection cost (F1c 0.970 vs 0.990, one miss,
one split). Both differences are single-cell / sub-human-floor scale on this field.

**Roles until out-of-sample data lands:** v1 (23k, `zs_hk2025_af647_c0`) remains the
segmentation input — its 0-miss/0-merge/0-split profile is the recipe's core win.
v2 (`zs_hk2025_af647_v2_c0`) is the human-facing rendering for drawing/review surfaces.
Full stipple elimination needs more training material (train one model across the
incoming annotated fields), not more parameter search on one plane.

## Known caveats

- Single-field, in-sample numbers throughout (e023/e026 blindness notes).
- Residual stipple in both HK models (v1 strong, v2 reduced) — single-plane
  self-supervision is the root cause; revisit when more fields arrive (e025/e039).
- Boundary differences between raw/v1/v2 (0.72–0.84 µm) are all below the measured
  ~1.1 µm annotator repeatability floor (e036) — do not rank variants on them.
- CellSAM weights: non-commercial academic license (e003).

## Engineering follow-ups (queued)

1. Expose two-channel fusion on `analysis.segment` (build-node-v2; Label-domain
   channel-key decision).
2. Wire the e020 auto-fill formula into `enhance.zs_deconvnet` as metadata-aware
   defaults (build-node-v2).
