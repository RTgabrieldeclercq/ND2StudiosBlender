# Registration — integration contract

Kernel module: `nodegraph/kernels/registration.py`
Node: **Registration** (`registration.stabilize`); `registration.align_to` borrows its
filter primitives.

> **2026-10-07.** The kernel is no longer a verbatim vendoring. A synthetic known-answer
> bench (`scripts/registration_synthetic_bench.py` — star field, star volume, deforming
> mass, affine mass, simulated landmark clicks) found a sign bug in the ECC seed and showed
> that phase-whitened correlation was the wrong default for microscopy data. §12 lists what
> changed and the numbers; the rest of this document describes the kernel as it is now.

---

## 1. Purpose & where the real math lives

Estimate per-frame spatial transforms that align every frame of a `(T, H, W)` — or
`(T, Z, H, W)` — single-channel series onto a common anchor (drift correction /
stabilization), then resample **any** channel through those transforms. The design is
*register-once, apply-to-all*: estimate on one reference channel, then apply the identical
transform bundle to every other channel so colocalization is preserved.

**The real math is in-repo** — plain Python over numpy plus standard library primitives.
It delegates to:

- `scipy.ndimage` — sub-pixel resampling (`shift`, n-D) and the band-pass (`gaussian_filter`)
- `skimage.registration.phase_cross_correlation` — sub-pixel translation, 2-D or 3-D
- `skimage.feature.ORB`, `skimage.measure.ransac`, `skimage.transform.*` — the feature model
  and the landmark fits (`estimate_transform`)
- `scipy.stats` — the F-test / χ² behind the landmark model choice
- `cv2` (OpenCV) — `findTransformECC`, `warpAffine` / `warpPerspective`

Three transform families: **translation** (band-passed cross-correlation, the workhorse;
the only one that is also 3-D), **euclidean/affine** (ECC, seeded from the feature match
when that is trusted, else from the correlation shift), and **feature** (ORB + RANSAC, for
large motion / rotation / scale — ~0.3–0.5 px on its own, which is why it is now mostly a
seed).

---

## 2. Entry points

Primary (series-level):

```python
estimate_series(
    series: np.ndarray,               # (T, H, W) or (T, Z, H, W)
    model: str = "translation",       # "translation" | "euclidean" | "affine" | "feature" | "auto"
    reference: str = "previous",      # "first" | "previous" | "mean" | "template"
    upsample: int = 20,
    highpass_sigma: float = 2.0,
    min_confidence: float = 0.0,
    normalize: str = "none",          # "none" | "zscore"
    roi: Optional[Dict[str, Any]] = None,
    feature_transform: str = "affine",  # "euclidean" | "similarity" | "affine"
    min_inliers: int = 8,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
    lowpass_sigma: float = 0.0,       # matched-filter smoothing before the high-pass (the node passes 1.0)
    correlation: str = "cross",       # "cross" | "phase"
    landmarks: Optional[Dict[str, Any]] = None,   # {"t": k, "src": [[y,x],…], "dst": [[y,x],…]}
    seed_from_features: bool = True,
    click_sigma: float = 1.0,         # per-click precision the landmark NON-RIGID verdict is judged against
) -> Dict[str, Any]                   # the transform bundle (see §5)

apply_series(series (T,H,W), transforms, interp_order=1, progress_cb=None) -> (T,H,W)
apply_frame(frame (H,W), transforms, t, interp_order=1) -> (H,W)      # the planar unit
apply_volume(volume (Z,H,W), transforms, t, interp_order=1) -> (Z,H,W) # the 3-D unit
```

Leaf functions (usable standalone, pairwise):

```python
estimate_translation(reference, moving, upsample=20, highpass_sigma=2.0, window=True,
                     mask=None, bbox=None, lowpass_sigma=0.0, correlation="cross")
    -> (shift (2,) | (3,) float64, ncc float)                 # 2-D planes or 3-D volumes
ecc_align(reference, moving, model="euclidean", init_shift=None, iters=200, eps=1e-6,
          gauss=5, interp_order=1, mask=None, init_warp=None)
    -> (warp (2,3) float32, cc float, aligned)
estimate_features(reference, moving, transform="affine", mask=None, bbox=None,
                  n_keypoints=800, min_inliers=8, residual_threshold=2.0) -> (warp (2,3), conf)
estimate_from_landmarks(src_yx, dst_yx, model="auto", click_sigma=1.0, alpha=0.05) -> dict  # §7
parse_landmarks(text) -> {"t", "src", "dst"} | None                                         # §7
apply_shift(image, shift, order=1) -> np.ndarray           # n-D, dtype preserved
apply_warp(image, warp_matrix, motion=None, output_shape=None, interp_order=1) -> np.ndarray
common_translation_crop(shifts, shape, inset_edges=True) -> (y0,y1,x0,x1) | None
```

The node wiring is: `estimate_series` on the reference channel → `apply_frame` (2D) or
`apply_volume` (3D) per streamed unit on every channel, with the returned bundle.

---

## 3. Inputs

### `estimate_series`

| name | Python type | array shape | dtype | axis order | units | required? / default | meaning & constraints |
|---|---|---|---|---|---|---|---|
| `series` | `np.ndarray` | `(T, H, W)` or `(T, Z, H, W)` | any (computed in float) | T[, Z], row(y), col(x) | pixels | **required** | Single-channel series. 3-D or 4-D, else `ValueError`. Caller has done all prep (I/O, per-M/per-C extraction, crop, exclusion). |
| `roi` | `dict` \| `None` | — | — | — | pixels | default `None` (whole frame) | See §4. Restricts *estimation* to a region; the transform is still applied full-frame. |
| `landmarks` | `dict` \| `None` | — | — | `(row, col)` | pixels | default `None` | Clicked correspondences between frame 0 and frame `t` (§7). |
| `progress_cb` / `cancelled_cb` | callable | — | — | — | — | `None` | Progress 0–100; cancel → remaining frames keep identity. |

### `apply_series` / `apply_frame` / `apply_volume`

| name | type | shape | meaning |
|---|---|---|---|
| `series` / `frame` / `volume` | `np.ndarray` | `(T,H,W)` / `(H,W)` / `(Z,H,W)` | Any channel; output dtype = input dtype. |
| `transforms` | `dict` | — | The exact bundle `estimate_series` returned. |
| `t` | `int` | — | Frame index; out of range → unchanged. |

### leaf `reference` / `moving`

`(H, W)` planes, or `(Z, H, W)` volumes for `estimate_translation`. `mask` is a bool
`(H, W)` (broadcast over z for a volume) in full-frame coordinates; `bbox=(y0,y1,x0,x1)`
crops the last two axes.

---

## 4. Parameters

| name | type | default | valid range / choices | semantics |
|---|---|---|---|---|
| `model` | str | `"translation"` | `translation`, `euclidean`, `affine`, `feature`, `auto` | DOF of the transform. `translation` → band-passed cross-correlation (2-D or 3-D; returns `shifts`, `warps=None`). `euclidean`/`affine` → ECC seeded by the feature match or the correlation shift (2×3 `warps`). `feature` → ORB+RANSAC. `auto` → the family the landmarks imply (§7); without landmarks, `translation`. `similarity` from a landmark fit runs as `affine` (ECC has no similarity motion). |
| `reference` | str | `"previous"` | `first`, `previous`, `mean`, `template` | Anchor. `first` = every frame → frame 0. `previous` = cumulative, composed to absolute. `mean` = every frame → series mean. `template` = two-pass (rough-stabilize → average) anchor. |
| `upsample` | int | `20` | ≥ 1 | Correlation sub-pixel factor; accuracy ≈ 1/`upsample` px. |
| `lowpass_sigma` | float | `0.0` (node: `1.0`) | ≥ 0 | Gaussian **matched-filter** smoothing before the high-pass — the band-pass's lower edge. ~the spot radius for point-like data. Must stay below `highpass_sigma`. |
| `highpass_sigma` | float | `2.0` | ≥ 0 (0 = off) | Background removal before correlating. **Live again**: under `correlation="phase"` any filter applied to both images cancels, so this did nothing until the default changed. |
| `correlation` | str | `"cross"` | `cross`, `phase` | Plain cross-correlation of the band-passed images vs skimage's phase whitening. Cross is 3–15× more precise on band-limited noisy images (§12). |
| `min_confidence` | float | `0.0` | typically 0–1 | Frames whose confidence (NCC / ECC cc / RANSAC inlier fraction) is below this are gated (§6.4). A warp estimate that **failed** (cc 0) is gated regardless. |
| `normalize` | str | `"none"` | `none`, `zscore` | Per-frame normalization for *estimation only*. |
| `roi` | dict \| None | `None` | see below | Estimation region. |
| `feature_transform` / `min_inliers` | str / int | `"affine"` / `8` | — | RANSAC family and consensus floor for `model="feature"`. |
| `landmarks` | dict \| None | `None` | `{"t", "src", "dst"}` | §7. |
| `seed_from_features` | bool | `True` | — | Seed ECC from the ORB+RANSAC warp when ≥ 50 % of matches agree on it (§6.5). |
| `click_sigma` | float | `1.0` | > 0 | Per-click precision (px) for the landmark non-rigid verdict (§7); the family choice does not depend on it. 2 px clicks judged at 1 px flag a pure drift as non-rigid. |
| `interp_order` | int | `1` | 0 / 1 / 3 | Resampling order in the apply functions. |

**ROI spec** (`roi` → `roi_to_mask`):

- `{"kind":"rect", "x","y","w","h"}` → box mask **and** bbox. Translation uses the bbox
  (sub-pixel crop); ECC/feature use the mask.
- `{"kind":"shapes", "shapes":[{"type":"rect"|"ellipse"|"polygon", "vertices":[[y,x],…]}]}`
  → rasterized freeform mask, no bbox; masked correlation is **integer-pixel**.
- `{"kind":"mask", "mask": bool (H,W)}` → a mask the caller rasterized itself (the node
  does this from the Draw tool's richer shape vocabulary).
- falsy / unknown → whole frame.

---

## 5. Output

`estimate_series` returns a **dict bundle**:

| field | type | shape | meaning |
|---|---|---|---|
| `model` | str | — | The model that RAN (resolved from `auto`). |
| `reference` | str | — | Echo. |
| `shifts` | `np.ndarray` | `(T, 2)` or `(T, 3)` | **Absolute** per-frame translation, `(row, col)` or `(z, row, col)`, in the `apply_shift` convention (`reference ≈ apply_shift(moving, shift)`). For warp models the planar part is the warp's translation `(warp[1,2], warp[0,2])` and, for a volume, the leading entry is the axial shift. |
| `warps` | `np.ndarray` \| `None` | `(T, 2, 3)` | `None` for translation. Otherwise the absolute planar 2×3 affine mapping **reference → moving** in `(x, y)`. |
| `confidence` | `np.ndarray` | `(T,)` | NCC (translation), ECC cc, or RANSAC inlier fraction. Anchor frame = 1.0. |
| `gated` | `np.ndarray` bool | `(T,)` | Frames whose estimate was held / skipped. |
| `landmarks` | dict \| None | — | The `estimate_from_landmarks` result when landmarks were given (§7). |

---

## 6. Conventions & GOTCHAS

1. **Shifts are `(row, col)` = `(y, x)`** (numpy order), and `(z, row, col)` for a volume.
   `shifts[t]` is the vector applied to the moving frame to align it onto the reference.

2. **ECC / feature warps map reference → moving** and are applied with `WARP_INVERSE_MAP`
   (`apply_warp` does this). Do not re-invert.

3. **The ECC seed is `−shift`** (fixed 2026-10-07). The aligned image samples `moving` at
   `p + d`, `d` being how far the content moved; `init_shift` moves content *back*, so the
   seed translation is its negative. The vendored `+shift` put the seed at the mirror image
   of the answer, `2·|shift|` away — harmless on a big smooth texture (ECC still pulled in),
   catastrophic on a sparse field, where non-convergence returns the seed and the result was
   wrong by exactly twice the drift. `init_warp` (a full 2×3) overrides `init_shift`.

4. **Confidence gating differs by mode.** With `min_confidence > 0`: absolute modes
   (`first`/`mean`/`template`) **hold the last-good transform**; `previous` **skips the
   increment**. A warp estimate that failed outright (ECC did not converge, converged to an
   implausible warp — `_warp_is_sane`: a singular value outside [0.5, 2] or a translation past
   the frame — or the feature fit found no consensus) is gated **whatever** `min_confidence`
   is: there is no estimate to apply.

5. **ECC seeding order of trust** (`_ecc_seeded`): a caller-supplied landmark warp, else the
   ORB+RANSAC warp when ≥ 50 % of the cross-checked matches agree with it (a geometrically
   verified vote), else the correlation translation; whichever seed fails, the other is
   tried. ECC from a translation seed converges to a WRONG optimum with a HIGH cc once the
   true rotation passes ~10°, so cc alone cannot pick the seed — the feature vote can.

6. **`estimate_series` returns ABSOLUTE per-frame transforms.** In `previous` mode the
   increments are composed (`cum_shift`; `_homog(warp) @ h_abs`) — do not re-accumulate.

7. **Anchor / start frame.** `first`/`previous` leave frame 0 fixed (`start=1`);
   `mean`/`template` register every frame (`start=0`).

8. **ROI: rect vs shapes vs mask.** A rect drives translation via a sub-pixel bbox crop but
   ECC/feature via a mask (keeps the rotation centre right). Freeform shapes / a caller mask
   give masked correlation, which is integer-pixel for translation.

9. **A volume under a warp model** estimates its planar warp on the mid-z plane and takes
   only `dz` from the 3-D correlation; `apply_volume` warps every plane then shifts along z.
   The two commute (the warp is in-plane). Only the translation model is natively 3-D.

10. **Axial precision is set by the data, not the algorithm.** On a 16-plane stack with a
    2-plane axial PSF the bench recovers `dz` to ~0.45 plane (0.24 with a 1.2-plane PSF); the
    3-D band-pass + 3-D Hann the kernel uses was the best of the filtering variants tried.

11. **`normalize="zscore"` affects estimation only.** Geometry is identical.

12. **`common_translation_crop` is translation-only and optional.**

13. **Dtype is preserved end-to-end** (`_cast_like`). **Border handling** zero-pads vacated
    borders. **Identity short-circuits**: a zero shift / identity warp returns the input
    untouched.

14. **`apply_frame` is defined once** (the module used to define it twice; merged). A
    3-component shift applied to a plane uses its planar part.

---

## 7. Landmarks — fitting the transform from clicked correspondences

```python
lm = estimate_from_landmarks(src_yx, dst_yx, model="auto", click_sigma=1.0, alpha=0.05)
# lm["model"]      the family chosen: translation | euclidean | similarity | affine
# lm["warp"]       (2,3) float32 reference→moving in (x, y) — the ECC seed / direct transform
# lm["residuals"]  {family: dof-corrected RMS px}   lm["pvalues"] {family: p vs the richest}
# lm["nonrigid"]   True when even the richest informative family cannot explain the clicks
# lm["summary"]    one line for the user
```

- `src_yx[i]` on the REFERENCE frame (frame 0), `dst_yx[i]` the same physical place on the
  moving frame, `(row, col)`.
- Each family with `2n > p` (translation 2, euclidean 3, similarity 4, affine 6) is fitted
  by least squares and scored by `sqrt(RSS / (2n − p))`. **Model choice is a nested F-test**:
  walking from the simplest up, the first family the richest does not improve on
  significantly (`p > alpha`) wins — the clicks' own scatter is the yardstick, so no guess
  about click precision is needed to tell a drift from a rotation from a shear.
- **Non-rigid verdict**: a χ² test of the richest family's RSS against `2·click_sigma²`
  (two clicks per pair). The detectable deformation is therefore about `2.2 × click_sigma`
  at five pairs — the user must click ON the part that deforms, and 1 px clicks cannot
  reveal a 1 px bulge.
- Minimum pairs: translation 1, euclidean/similarity 2, affine 3 — but with the minimum the
  richest family fits exactly and nothing can be judged; **five well-spread pairs** is the
  practical floor (3 cannot separate the families — affine fits any three exactly).
- `parse_landmarks` reads `t=5; row,col -> row,col; …` or JSON
  `{"t": 5, "src": [[y,x],…], "dst": [[y,x],…]}`. `t` is required (≥ 1).
- In `estimate_series` the fitted warp seeds every frame's ECC: absolute modes get
  `_interp_warp(warp, t / k)` (angle and drift scale linearly, the symmetric part by a
  matrix power — a 40° turn at half the motion is a 20° turn), `previous` gets one increment
  `1 / k`. A seed is what matters: a 42° rotation ECC cannot find from a translation seed
  registers to 0.02 px from a five-click seed with 1 px click jitter.

---

## 8. Dependencies

| pip package | import name | why | when needed |
|---|---|---|---|
| numpy | `numpy` | arrays, linear algebra | import-time |
| scikit-image | `skimage` | correlation, ORB/RANSAC, `estimate_transform`, `draw` | import-time (`draw`), lazy elsewhere |
| scipy | `scipy` | `ndimage.shift`/`gaussian_filter`, `stats.f`/`chi2` | lazy |
| opencv-python | `cv2` | ECC + warping | lazy (euclidean/affine, `apply_warp`) |

---

## 9. Failure modes / edge cases

- **Wrong-rank `series`** → `ValueError("estimate_series expects (T,H,W) or (T,Z,H,W) …")`.
- **Unknown `model` / `reference`** → `ValueError` listing the choices.
- **Blank frame** (`std < 1e-6` after the band-pass) → `estimate_translation` returns
  `(zeros, 0.0)`; ECC returns the seed with cc 0 → gated.
- **ECC non-convergence or implausible warp** → seed, `cc=0.0`, unaligned moving → gated.
- **Feature model, too few keypoints/matches/inliers** → `(eye(2,3), 0.0)` → gated.
- **Landmarks**: too few pairs for an explicit family → `ValueError` naming the minimum;
  mismatched / non-finite points → `ValueError`; missing `t` → `ValueError`.
- **ROI degenerate** → whole-frame estimation.
- **`common_translation_crop` with drift exceeding the frame** → `None`.

---

## 10. Minimal runnable example

```python
import numpy as np
from scipy.ndimage import shift as nd_shift
from nodegraph.kernels import registration as reg

H = W = 64; T = 4
base = np.random.default_rng(0).random((H, W)).astype(np.float32)
series = np.stack([nd_shift(base, (2.0 * t, -1.5 * t), order=1) for t in range(T)])

tf = reg.estimate_series(series, model="translation", reference="first", lowpass_sigma=1.0)
print(np.round(tf["shifts"][1], 2))   # -> [-2.  1.5]   (row, col) that undoes frame-1 drift
aligned = reg.apply_series(series, tf)

# a volume: (T, Z, H, W) → 3-component shifts
vol = np.stack([nd_shift(np.random.default_rng(1).random((12, H, W)), (0.5 * t, 2.0 * t, -1.5 * t), order=1)
                for t in range(T)])
tf3 = reg.estimate_series(vol, "translation", "first")      # tf3["shifts"].shape == (4, 3)
one = reg.apply_volume(vol[2], tf3, 2)

# landmarks: five clicked pairs, frame 0 ↔ frame 3
lm = reg.estimate_from_landmarks([[10, 10], [10, 50], [50, 10], [50, 50], [30, 30]],
                                 [[16, 5.5], [16, 45.5], [56, 5.5], [56, 45.5], [36, 25.5]])
print(lm["model"], lm["summary"])     # translation …
```

---

## 11. Pipeline wiring

- **Upstream:** the node supplies the reference channel's `(T, H, W)` mid-plane series (2D)
  or `(T, Z, H, W)` volume series (3D), per multipoint, plus the kernel `roi` built from the
  drawn `region` socket and the `landmarks` parsed from its socket.
- **This kernel:** `estimate_series` → bundle → `apply_frame` / `apply_volume` per streamed
  unit on **every** channel.
- **Downstream:** the aligned Dataset carries `drift_y`/`drift_x` (+ `drift_z`) and
  `drift_confidence` as Frame layers and `registration_model` / `registration_landmarks` as
  metadata; `common_translation_crop` can trim the shared border for export.

---

## 12. What the 2026-10-07 bench changed, and the evidence

`scripts/registration_synthetic_bench.py` (`--legacy` re-creates the old behaviour). EPE =
RMS endpoint error of the recovered transform against the exact true motion, in px; the
*family floor* is the best any member of that model family could do.

| finding | before | after |
|---|---|---|
| ECC seed sign (`euclidean`/`affine` on a 12–60-star field, drift 1.5 px/frame) | 2–14 px (cc 0.3, returning the mirrored seed) | 0.02–0.08 px, cc 0.99 |
| phase-whitened vs cross-correlation, sparse stars (n=12) | 0.18 px | 0.007 px |
| …dense stars (n=60) | 0.06 px | 0.000 px |
| …textured mass, pure translation | 0.33 px | 0.000 px |
| …`highpass_sigma` | no effect at all under phase whitening | live (0 → 1.1 px under a drifting illumination gradient; 2 → 0.04 px) |
| matched-filter `lowpass_sigma` on 40 faint stars, peak SNR ≈ 8 | 156 px (`first`) / 2.2 px (`previous`) | 0.09 px / 0.22 px |
| rotation in `first` mode (2°/frame, 14° by t=7) | 3 px, 3 of 7 frames gated | 0.03 px |
| rotation 6°/frame (42° by t=7), `first` | 14 px | 0.02 px |
| axial drift on a 16-plane volume (0.5 plane/frame) | 100 % uncorrected (2D path), lateral 0.21 px | `dz` to 0.45 plane, lateral 0.01 px |
| deforming mass (0.4 px/frame bulge + drift), drift estimate bias | 0.61 px (`first`) / 1.14 px (`previous`) | 0.45 px / 0.44 px; **0.04 px** with the estimate restricted to the rigid half |

**Non-registrable conditions** the bench establishes (unchanged by any fix, and now flagged
by `confidence` / `gated` / the landmark verdict rather than silently wrong):

- fewer than ~12 point-like features per frame with nothing else in the field;
- peak SNR below ~3 on a sparse field (0.55 px at SNR 3.5, >5 px at 1.3);
- cumulative shift beyond ~50 % of the frame with `reference="first"` (no overlap; use
  `previous` or `template`);
- rotation beyond ~75° cumulative in `first` mode (the ORB seed loses consensus; use
  `previous`);
- any specimen that deforms: a global family's floor is the limit (0.45 px for a 0.4 px/frame
  bulge under translation, 1.9 px for a 1.2 px/frame bulge) and the deformation biases the
  drift estimate by about its mean pull — restrict the estimate to a rigid region;
- a very thin stack: axial precision ≈ 0.4–0.5 plane at 16 planes with a 2-plane PSF.

---

## 13. Provenance

Originally vendored verbatim (branch **Version-1.45**) from
`nd2studios/backend/registration/estimate.py` (the kernel body),
`nd2studios/backend/stitch/register.py` (`_highpass`, `_hann2d`, `_ncc`) and
`nd2studios/backend/analysis/manual_mask.py` (`rasterize_shapes`, `_rasterize`); the two
`nd2studios` imports were replaced by in-file definitions. Refined 2026-10-07 as listed in
§12 and in the module docstring.
