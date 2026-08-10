# §5b. `available_in` gates ANY mode, not just `dim` (2026-07-28)

*A leaf of the **wire-node-v2** skill. Cited as `§5b`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


**A socket the selected mode never reads must be HIDDEN, not shown and discarded.** The
key of `available_in` is any mode name — `{"method": frozenset({"fixed"})}`,
`{"boundary": frozenset({"alpha_shape"})}` — and a socket may gate on several modes at
once (`analysis.histogram_threshold` gates each threshold on **method × direction**, so
1–2 live fields show instead of all 8). When adding a node with a non-dim `Mode`, read
the kernel and gate every param that only one branch consumes; a param that is live in
*every* branch stays ungated (`detect.particles`: `min_size` filters blobs in both LoG
and components — leave it alone; `analysis.boundary_band`: `band_voxels` is dilation
iterations *and* the `edt` fallback threshold, so only `band_um` is gated).

Two things this does **not** change: the engine still passes every declared param to the
compute (gating is edit-time only, so a compute's `ctx.params.get(name, default)`
fallback must stay correct), and the mode already folds into the recipe hash, so hiding
a socket never alters a memo key.

`available_in` can only see **mode state**, not whether an optional socket is wired — so
a param whose liveness depends on a wire (DVC/DIC `reference_frame`: dead under
`previous_frame`, but live again when an external `reference` Dataset is connected)
stays ungated on purpose. Don't gate those; hiding a control that still has an effect is
the same bug in reverse.

Both halves must be verified: `NodeSpec.active_inputs(state)` in `nodegraph.selftest`,
**and** the GUI — the node card re-lays-out via `scene.sync → item.refresh`, and the
inspector rebuilds via `_set_mode`'s deferred `_rebuild` (covered by the "mode gate"
check in `scripts/_nodelab_v2_phase5_probe.py`).
