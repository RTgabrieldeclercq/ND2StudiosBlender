# V4.00 step 11i — a position remembers which one of the file it was (`position_index`)

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11h

## What changed

- **`position_index`, a new member of `PER_POSITION_KEYS`** (`nodegraph/metadata.py`): which
  position OF THE FILE each `m` was, one integer per M. Absent on a source — until something
  narrows M a stream's index is the file's — and written by `position_subset` on the FIRST
  narrowing (the `keep` indices themselves), then reindexed on every later one like any other
  member of the family. Every position tap, `util.select_position`, `util.select_group`,
  `util.crop` on M, `util.batch` and the Viewer's frame pin already go through
  `position_subset`, so the envelope and the pulled payload carry the same list with no
  further change; a stitch retires it with the rest of the family; a bundle's members have
  none, so a bundle starts clean. Provenance, not geometry: nothing computes from it, and it
  is not a table column (only `source_file` and `position_group` are).
- **A position with no point name is named by its index in the file.** The fallback for a
  file that carries no `position_name` used to be `m{i}` over the stream's OWN length — `m0`
  for every single position, three pages down (`GraphDocument._env_position_names`, the
  Split Positions card's socket labels). It now reads `position_index`
  (`_position_fallback_names`), so one position of three reads `m1` on every later page and
  the Output it feeds is auto-named `m1`; and since the name already says it, the card
  prints `m1`, not `m1 · m0`.
- **Tests**: a block in `test_stream_identity` (the engine's `position_subset` invents and
  reindexes the list; a no-names file's `pos1` is `m1` at edit time on the next page and
  `position_index == [1]` on the pulled payload). **Docs**: MANUAL §4 (the position bullet),
  a §18 row; codemap CON-23 (a sentence); the `PER_POSITION_KEYS` and `position_subset`
  docstrings.

## Why

The user, right after 11h: "it seems its getting the location wrong downstream, I parsed the
location data from the input node into multiple m locations. I passed the m1 location to the
output. but the downstream node graphs are showing m0 as the location." Their file has no
point names, so the sockets read `0 · m0`, `1 · m1`, … — and after the `pos1` tap the stream
is one position long, so the fallback name, built from the stream's own index, was `m0` for
whichever position had been taken. The pixels were right (verified headless: the plane
downstream of `pos1` is position 1's); only the name lied, which is the one failure the
per-M family exists to prevent.

Why in the engine rather than the GUI: the GUI cannot recover the file index after the fact
— nothing in a one-position envelope says which one it was — so the fact has to be written
at the moment it is still known, by the function that narrows M, and travel as a member of
the family that already knows how to be subset, concatenated and retired. Putting it in
`position_subset` rather than stamping `range(m)` on every source envelope keeps the seeds
unchanged (a stamped seed would differ from the probes' `_SYNTH_META` seeds and re-pull after
every first pull) and makes a bundle correct for free (bundle-wide indices at its first
narrowing, instead of concatenated per-member ones).

## Files

- `nodegraph/metadata.py` — `PER_POSITION_KEYS` gains `position_index`; `position_subset`
  writes it when absent
- `nodelab_v2/document.py` — `_position_fallback_names`; `_env_position_names` and
  `position_descriptors` use it
- `nodegraph/selftest.py` (hotspot: tail of `main()` — a block in `test_stream_identity`)
- `MANUAL.md`, `codemap/concepts.md`, `codemap/curated.lock.json`, `codemap/gen/*`,
  `codemap/STATE.md`

## How to verify

`python run.py`, load a multipoint ND2 whose points are unnamed (sockets read `0 · m0`,
`1 · m1`, …). Split Positions → wire `1 · m1` into a Page Output: it reads `Output · m1`. On
the next page its Page Input's `out`, and every card after it, read `m1`; hover: `position:
m1 — 1 of N positions`. Pull a card there: the Viewer shows position 1's field.

## Gates

- [x] selftest — 189 tests run; only `test_write_movie` fails (pre-existing H.264 rounding).
  No test asserted an exact per-M metadata dict, so the new key broke nothing
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 153 checks, run alone, exit 0
- [x] catalog snapshot — CATALOG IDENTICAL, 104 ops (no op declaration changed); synopsis
  CURRENT
- [x] codemap — CODEMAP CURRENT (CON-23 blessed; every node's `fp` moved because
  `nodegraph/metadata.py` is in every node's fingerprint — expected, not a finding)
- [ ] sync check — run before the push
