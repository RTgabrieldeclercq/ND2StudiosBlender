# §7b. Provenance inheritance — a node stamps context a downstream node reads

*A leaf of the **wire-node-v2** skill. Cited as `§7b`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


When a downstream node's behaviour is **determined by how an upstream node ran** — not
by a value the user should re-type — the upstream node should **stamp that config** and
the downstream node should **inherit it**, instead of exposing a redundant/conflictable
control. This is metadata intelligence at the *structural* (non-calibration) level.

**The built-in generic case — structure dimensionality (`z_kind`).** A structure table's
`z_kind` (`"plane_index"` = 2D per-plane, `"subpixel"` = 3D) is lost when
`with_structure` explodes the table into per-column `AttributeLayer`s. So `with_structure`
**automatically preserves it** into a namespaced `__struct_zkind__` metadata map (keyed by
domain+layer) — **every structure-producing node self-describes its dimensionality for
free, no per-node code**. A consumer inherits it with **`ds.structure_zkind(domain, layer)`**
(→ `"plane_index"`/`"subpixel"`/`None`). This is why `transform.rasterize_field` has **no
`DimMode` lever**: it routes 2D-per-plane vs 3D-volumetric off the *field's own* `z_kind`,
so a 2D per-plane field on a `z>1` image is never misread as a 3D grid (its plane-index
`z` treated as a coordinate). `analysis.accumulate_field` inherits its dim the same way.

**The general pattern (for config beyond dim):**
- **Stamp on the output** with `ds.with_metadata(key=value, ...)` — the sanctioned write
  path. Use a **namespaced** non-calibration key (e.g. `dvc_reference_mode`,
  `dvc_strain_type`) so it never collides with `CALIBRATION_KEYS`. Example: `analysis.dvc_field`
  stamps `dvc_reference_mode` / `dvc_strain_*` alongside its Point output (dim is NOT
  stamped here — it rides the generic `z_kind`).
- **Inherit in the consumer** by reading `ctx.inputs[0].metadata.get("key")` **directly**
  (or `ds.structure_zkind(...)` for dim). Allowed *because the key is not a calibration
  key* — `_StrictCalibMetadata` passes non-calibration reads through (only `CALIBRATION_KEYS`
  trip C7). Memo-safe without recording: the marker rides the upstream payload, so a change
  upstream bumps the upstream **revision**, which already folds into the consumer's
  `recipe_hash` (and `output_fingerprint` folds `__struct_zkind__` via `metadata`).
- **Derive, don't ask.** A consumer that inherits its dim/reference/measure from the
  provenance needs **no `DimMode` lever and no reference param** — a lever could disagree
  with the data and silently corrupt it. Fix a plain `granularity`/`kernel_axes` (a superset
  like `frozenset({"z","y","x"})`) instead; the compute loops the units internally.
- **Refuse the wrong input.** Inherited provenance also lets the consumer **validate**:
  `analysis.accumulate_field` raises if the marker is absent (not a DVC field) or says
  `fixed_frame` (already cumulative — nothing to accumulate), and re-stamps its own output
  `dvc_reference_mode="cumulative"` so a second accumulate is refused too.
- Contrast with **calibration** provenance: pixel/z/dt still flow through the envelope and
  are read via `ctx.calib` (§7) — only *structural config that isn't a physical unit* uses
  this stamp-and-inherit pattern.
