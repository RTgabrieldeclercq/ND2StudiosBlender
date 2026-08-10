# §4d. Filesystem-path sockets — `path_kind` (V2.15)

*A leaf of the **wire-node-v2** skill. Cited as `§4d`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


A STRING socket whose value is a **path on the machine that runs the graph** is not free
text either. Declare it and the inspector puts a **Browse…** button beside the field:

```python
InString("model_path", "Local weights", field=False, default="",
         path_kind="open_file",                       # open_file | save_file | directory
         path_filter="PyTorch checkpoint (*.pt *.pth);;All files (*)",
         path_hint="empty = published · or Browse…")  # placeholder while empty
```

* `path_kind` is the **only** thing the GUI keys on. Before V2.15 the inspector matched the
  literal socket *name* `"path"`, so `sd_model_path` / `model_path` were typing-only —
  which is exactly the failure this declaration exists to prevent. A typo or a non-STRING
  socket is **refused at registration** (`NodeRegistry.register`, `registry.PATH_KINDS`),
  because a silently-ignored value would just look like a plain text field.
* `directory` gets `getExistingDirectory`, not a file dialog — a StarDist local model is a
  *folder* (`config.json` + `weights_best.h5`) and no file dialog can return one.
* `path_hint` is the place to say what an **empty** value means; it differs per socket
  (`io.load` → the synthetic demo; a model path → fall back to the pretrained name).
* All three are **presentation-only, never hashed** (like `description`), so annotating an
  existing socket cannot invalidate a memo entry or a saved graph.

**Two traps if you touch the catalog itself:**

* **Do not key anything on `LayerKey`.** It is `(domain, layer, name)`, but the user-facing
  name is in a *different slot per family*: `with_layer` leaves `layer=None`, so a lattice
  layer is `(VOXEL, None, "mask")`; `with_structure` files each COLUMN as
  `(POINT, "spots", "y")`. Keying on slot 1 collapses every Voxel layer to `(VOXEL, None)`.
  `layer_names` stores the user-facing name per domain to sidestep this.
* **The catalog is NOT monotone.** `reshaped_axes(drop_stale=True)` discards lattice layers
  whose shape no longer matches, so an axis-changing node (crop/resample/zproject/stack/
  channel.select) *removes* layers. The drop is derived centrally from the axis delta — you
  do not declare it — but do not assume "layers only accumulate".
