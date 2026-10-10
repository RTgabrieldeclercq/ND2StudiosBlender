# Field mechanics nodes

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

A new `field` category of five cards, plus `bundle:mechanics`, a new kernel, `Pa` units and one
fix to DIC's strain columns.

**Where they work.** The cards run on any measured displacement or velocity field: the Point table
written by DIC, PIV, DVC, the staged cards, Track Field or Load Analysis. Each card **adds columns to
that same table**, so one table carries displacement → gradient → strain → stress.

**The cards.**
- **`field.gradient` (Field Gradient)** writes `grad_<i><j> = ∂u_i/∂X_j` and `grad_valid`. It uses
  lattice differences on a regular grid (one-sided beside holes) or a local moving-least-squares
  (MLS) plane fit for any other layout. On tracked particles it fits where each particle was.
- **`field.strain` (Strain)** writes the symmetric tensor under infinitesimal, Green-Lagrange,
  Almansi or Hencky strain, plus principal values and angle, von Mises equivalent, max shear and
  trace. On a velocity field these are strain rates.
- **`field.rotation` (Rotation & Curl)** writes curl or vorticity and the finite rotation, positive
  counter-clockwise on screen.
- **`field.volume_change` (Volume Change)** writes `J = det F`, `J − 1` and the divergence.
- **`field.stress` (Stress)** writes Cauchy stress, principal stresses, pressure, von Mises, Tresca
  and strain energy, through one of four laws: Hooke, Saint Venant-Kirchhoff, compressible
  neo-Hookean, or a Newtonian fluid (velocity only). In 2-D it takes plane strain or plane stress.

**How they behave.**
- 2-D vs 3-D is inherited from the table's `z_kind`.
- Columns record their units (dimensionless, 1/s, deg, Pa).
- Each card's settings go under `__field_mechanics__[layer]`.
- Strain, Rotation, Volume Change and Stress refuse a table without `grad_*`.

**The kernel.** `nodegraph/kernels/field_mechanics.py` (with its `.md` contract) reuses
`field_math.strain_from_gradient` and `track_field`'s MLS gauge, Hooke's law and invariants.

**Units.** `nodegraph/units.py` gains `Pa` / `kPa` / `MPa` / `mPa`, and Math's unit choices gain
`Pa` and `kPa`.

**DIC fix.** DIC's `strain_xy` / `strain_yx` (`analysis.dic_correlate` with `compute_strain`, and
`dic.strain`) are now the symmetric tensor shear `(dv/dx + du/dy)/2`. Before, they were the raw
gradient terms while stamped "infinitesimal strain", unlike DVC and Track Field.

## Why

The request was for nodes that take a correlation's displacement or velocity field further, into
strain, curl, dilatation and a Jacobian for volume change, and then into stress through material
parameters, all with a visual output. This commit is the computing half of the approved plan; the
Viewer overlay, the vector fixes and `plot.field` follow in later commits.

**Why one gradient card instead of a gradient inside each node.** Every quantity depends on how the
field was differentiated: the method and the radius set the gauge length. Making that choice once,
visibly, on one card means Strain, Rotation and Stress cannot silently differentiate the same field
three different ways. The cards then re-key independently: changing the strain measure re-runs only
Strain.

**Why columns go on the same table.** It matches `dic.strain`, and the Viewer can colour by any
column of one layer.

**Why the DIC fix.** The cross terms were mislabelled, which would have made DIC's `strain_xy`
disagree with the new Strain card's on the same field.

**Snapshot gates for `3af1653` (the previous commit),** run in a worktree:
- the full selftest passed up to the known pre-existing `test_write_movie` failure, and every test
  after it passed;
- the GUI probe printed ALL PHASE-5 GUI PROBES PASSED, then exited 139 at teardown — the intermittent
  post-PASS crash seen before.

## Files

- `MANUAL.md`
- `codemap/STATE.md`
- `codemap/concepts.md`
- `codemap/curated.lock.json`
- `codemap/gen/MANIFEST.json`
- `codemap/gen/imports.jsonl`
- `codemap/gen/modules.jsonl`
- `codemap/gen/nodes.jsonl`
- `codemap/gen/sockets.jsonl`
- `codemap/gen/symbols.jsonl`
- `codemap/invariants.md`
- `codemap/node_demos.json`
- `codemap/node_roles.json`
- `codemap/node_synopsis.json`
- `nodegraph/catalog/__init__.py`
- `nodegraph/catalog/analysis/dic_correlate.py`
- `nodegraph/catalog/analysis/dic_stages.py`
- `nodegraph/kernels/README.md`
- `nodegraph/kernels/dic_correlate.md`
- `nodegraph/kernels/dic_correlate.py`
- `nodegraph/selftest.py`
- `nodegraph/units.py`
- `nodelab_v2/demo_recipes.py`
- `nodelab_v2/theme.py`
- `scripts/catalog_baseline.json`
- `nodegraph/catalog/_shared/field_nodes.py`
- `nodegraph/catalog/field/__init__.py`
- `nodegraph/catalog/field/gradient.py`
- `nodegraph/catalog/field/rotation.py`
- `nodegraph/catalog/field/strain.py`
- `nodegraph/catalog/field/stress.py`
- `nodegraph/catalog/field/volume_change.py`
- `nodegraph/kernels/field_mechanics.md`
- `nodegraph/kernels/field_mechanics.py`

## How to verify

`PYTHONUTF8=1 .venv/Scripts/python.exe -B -c "import nodegraph.selftest as s; s.test_field_mechanics()"`
checks the following, through the Engine and on analytic fields:
- an affine gradient is exact (lattice and local fit; 2-D and an anisotropic 3-D stack);
- a 10° on-screen rotation reads +10° with zero finite strain;
- `J` is 1.21 / 1.331; simple-shear curl is +γ;
- plane-stress `σ_zz` = 0 for Saint Venant-Kirchhoff and neo-Hookean;
- the Newtonian shear stress is ηγ̇;
- the velocity branch, the tracked-field branch and every refusal behave.

In the app, drop `bundle:mechanics` after a PIV card on the speckle demo, or open any Field card's
demo window.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` -> ALL NODEGRAPH SELF-TESTS PASSED
- [ ] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (baseline saved: 5 new ops, the DIC descriptions, Math's Pa / kPa units)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT (CON-28 new; INV-10 extended; INV-11 re-read and blessed)
- [ ] `python scripts/_sync_check.py` -> IN SYNC (rebased on origin/Blender before push)

<!-- Only the gates that apply need ticking: a docs-only change does not run the GUI probe.
     Say which you skipped and why. -->

Focused selftests run on this tree, all passing:
- `test_field_mechanics`, `test_registry`, `test_track_field`, `test_catalog_dic`, `test_dic_stages`;
- `test_column_catalog_complete`, `test_catalog_import_hygiene`, `test_units`, `test_math_nodes`;
- `test_socket_docs`, `test_option_docs`, `test_param_socket_contract`, `test_codemap`;
- `test_node_demos` (the five new demos run live).

The full selftest and the GUI probe run on a snapshot of this commit; their results go in the next
entry.

Hotspots touched: `nodegraph/selftest.py`, `nodegraph/catalog/__init__.py`, `codemap/gen/*`,
`scripts/catalog_baseline.json`.
