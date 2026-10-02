# Subtract Background: zero-regions approach; Multi-Otsu: per-class and selected outputs

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the team-sync entry)
- **Base:** 6fbd719

## What changed

**`enhance.subtract_background`** gains an `Approach` mode. `estimate_surface` is the node as
it was: fit a smooth background and remove it arithmetically. `zero_regions` is new: decide
which pixels are background and set exactly those to nothing (0 on a dark background, the
white level on a light one), leaving every object pixel bit-for-bit untouched. A `Detector`
mode picks the rule:

- `sampled_region`: the user draws one or more patches of pure background with the same Pick
  tool the ROI Mask node uses (`Background sample`, a `shapes` JSON socket). Every pixel no
  brighter than the sample's mean plus `Tolerance` standard deviations is background. The
  region is drawn in (Y, X) and applied to every Z plane; statistics pool over the unit.
  Nothing drawn, malformed JSON, or a region under two pixels is refused.
- `adaptive`: each pixel's local mean under a Gaussian of `Block size` (and `Block size Z` in
  3D), plus `Tolerance` robust noise widths, clamped into `[Lower limit, Upper limit]` typed in
  the image's own intensity units. The clamp is the "within a user-defined range" half of the
  request and is what protects the interior of an object wider than the block.

`Output` keeps its meaning: `corrected` zeroes the background, `background` zeroes everything
else, so the diagnostic shows exactly the pixels about to be removed. `presmooth` governs the
decision only. The estimator sockets, `bg_floor`, the `Estimator` and `Arithmetic` dropdowns
are hidden under `zero_regions`; the sample socket shows only with `sampled_region`, the block
and range sockets only with `adaptive`. The meta-transform in `nodegraph/metadata.py` now
checks the approach first so `bit_depth` survives this approach even when `divide` is left
behind its hidden dropdown. Footprint unchanged (whole plane / whole volume).

**`analysis.multiotsu`** gains an `Output` mode. `merged` is the class-index raster as before.
`per_class` adds one 0/1 mask per class beside it, named `<name>_0` … `<name>_K-1`, a disjoint
and complete partition, announced to the edit-time layer catalog through a new `extra_layers`
hook so downstream pickers offer them. `selected` writes one 0/1 mask of the classes a new
`Keep classes` socket names, with a small grammar (`1+` default, `0,2`, `1-2`); an empty,
backwards or out-of-range selection is refused rather than yielding an empty mask.

Tests: `test_subtract_background_zero_regions` and `test_multiotsu_outputs` added and
registered in `main()`. Manual rows for both nodes updated. Catalog baseline re-blessed,
codemap and node synopsis regenerated.

## Why

Both were direct requests. Subtract Background's nine estimators all answer "what does this
image look like with the objects taken out" by moving every pixel; the request was the other
operation, "which pixels are background, remove them entirely", which no node offered: ROI
Mask needs the user to outline the objects, and Threshold produces a mask rather than a
cleaned image. It belongs on this node rather than a new one because it is the same user
intent under the same name, and the polarity, pre-smoothing and diagnostic-output machinery
carry over unchanged. The estimators and the detectors are kept apart as two Approaches
rather than nine-plus-two Estimators because the arithmetic dropdown is meaningless for a
decision, and a control the selected path ignores is what the node charter forbids.

The adaptive detector first used Niblack's local standard deviation as its margin. Measured
on the test fixture, that zeroed a compact object whose core sat three times above its local
mean at the default tolerance, because a local spread balloons next to any bright object. The
margin is now one robust noise width per unit (1.4826 times the median absolute high-pass
residual), which follows shading through the local mean without inflating at objects. The
test pins both facts: a compact object survives unclamped, and the upper limit rescues an
object wider than the block.

Multi-Otsu produced a class index that no downstream node could consume without a manual
"which indices are foreground" step. Per-class masks let each intensity tier feed its own
chain, which was the request; the `selected` form covers the far more common case where the
tiers only ever existed to get a better foreground than a single Otsu cut.

## Files

- `nodegraph/catalog/enhance/subtract_background.py` — approach/detector modes, two detectors, six sockets, gating (hotspot: catalog)
- `nodegraph/metadata.py` — `subtract_background` meta-transform checks the approach first
- `nodegraph/catalog/analysis/multiotsu.py` — output mode, `keep` grammar, `extra_layers`
- `nodegraph/selftest.py` — two new tests, registered in `main()` (hotspot)
- `MANUAL.md` — §15 rows for both nodes
- `scripts/catalog_baseline.json` — re-blessed (hotspot, generated)
- `codemap/gen/*`, `codemap/STATE.md`, `codemap/node_synopsis.json` — regenerated

## How to verify

```
python -c "import nodegraph.selftest as s; s.test_subtract_background_zero_regions(); s.test_multiotsu_outputs()"
python scripts/_catalog_snapshot.py        # CATALOG IDENTICAL
python scripts/_node_synopsis.py show enhance.subtract_background
```

In the GUI: Subtract Background, Approach = zero_regions, Detector = sampled_region, press
Pick on `Background sample`, draw a patch of empty field, pull. Switch Output to `background`
to see exactly what was removed.

## Gates

- [x] New tests pass; `test_subtract_background`, `test_catalog`, `test_metadata_pass`,
  `test_param_socket_contract`, `test_socket_docs`, `test_option_docs`,
  `test_column_catalog_complete`, `test_live_reload_contract` pass when run directly.
- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` as a whole is still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry (9 stale curated codemap entries abort
  the run at `test_codemap`; `test_write_movie` fails on a codec tolerance). Neither is
  touched by this change.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (after `save`)
- [x] `python scripts/_codemap.py` -> gen/ current
- [x] `python scripts/_node_synopsis.py` -> SYNOPSIS CURRENT
