# `scene_viewer` — layer geometry + self-contained WebGL scene page

Kernel module: `nodegraph/kernels/scene_viewer.py` (+ `scene_viewer_template.html`).
Nodes: `io.write_scene_viewer` (the only caller) fed by the `scene.*` taps
(`nodegraph/catalog/scene/`). Design record: `CodeLog/ClaudesPlan/V4.10_scene_viewer.md`.

## 1. Entry points

```python
L = volume_layer(frames, spacing_um, origin_um, name=..., render=..., ...)      # [(t, vol), …]
L = vectors_layer(frames, name=..., units=..., ...)                               # [(t, pos, vec), …]
L = objects_layer(frames, name=..., ...)                                          # [(t, pos, rad, val), …]
L = tracks_layer(tracks, name=..., ...)                                           # [(id, times, pos), …]
L = series_layer(curves, name=..., ...)                                           # [(label, x, y), …]
scene = pack([L, …], title=..., subtitle=..., time_unit="s" | "frame")
html = render_html(scene, title=..., subtitle=..., snapshot_name=...)
report = scene_report(scene)
```

Stage functions are public: `downsample_factors`, `block_mean`, `to_uint8`, `isosurface`,
`grid_pitch`, `streamlines`, `parse_color`, `colormap_index`, `b64`.

## 2. Inputs

Every position is **µm in one world frame**, image convention: `(x, y, z)` with +y DOWN the
image and z up the stack. Time is a float per frame in the scene's unit (seconds or frame
index); the kernel only sorts and unions it.

| builder | input | notes |
|---|---|---|
| `volume_layer` | `frames = [(t, vol (nz, ny, nx) float), …]`, `spacing_um (sz, sy, sx)`, `origin_um (oz, oy, ox)` | pooled by integer blocks to `voxel_budget` (near-isotropic in µm, `downsample_factors`), windowed by two percentiles over EVERY frame → uint8 bricks; `render="iso"` → marching cubes at `iso_level_pct` instead |
| `vectors_layer` | `[(t, pos (N,3), vec (N,3)), …]` | any units (labelled); `scale` = drawn µm per unit, 0 → auto so the median glyph spans one grid pitch; `with_streamlines` integrates the FIRST frame |
| `objects_layer` | `[(t, pos (N,3), radius (N) µm, value (N) or None), …]` | non-finite / non-positive radii dropped |
| `tracks_layer` | `[(track_id, times (K), pos (K,3)), …]` | sorted by time; speed = forward difference per time unit |
| `series_layer` | `[(label, x (K), y (K)), …]` | `x_is_time` ties the chart cursor to the timeline |

## 3. Outputs

A layer is a dict: `kind`, `name`, style keys, `bounds [[x0,y0,z0],[x1,y1,z1]]` and base64
buffers (`Float32` positions ×3, `Uint32` ranges/indices, `Uint8` bricks). `pack` returns
`{meta, layers}` — `meta.bounds` (padded union), `meta.times` (sorted union of frame
times), `meta.dynamic`, `meta.time_unit` — with **y flipped** on every buffer
(`y' = y0 + y1 − y`, bricks reversed along y, normals negated) so the page is right-handed
with z up. `render_html` fills `__SCENE__`, `__TITLE__`, `__SUBTITLE__`, `__SNAPNAME__`.

## 4. The page

`scene_viewer_template.html`: WebGL2, no dependencies. A layer panel (toggle, opacity,
render mode / colourmap / window / slice positions for volumes, length / width / colour
mode for vectors, size / colour for objects, tail for tracks), a timeline (play, speed; each
layer shows its last frame at or before the cursor; static layers always), a 2-D chart
panel for series layers with a time cursor, orbit / pan / zoom camera with view buttons, µm
axes + grid with tick labels, z-stretch, fog, white/dark, PNG snapshot (labels baked in),
`?embed=1`, and the saved-view API of the flow viewer (`window.__getConfig`,
`__applyConfig`, `__setRunning`, `__setTime`, `#cfg=<base64 JSON>`). Volumes are 3-D
textures ray-marched as MIP (additive) or emission–absorption (`cloud`), or sampled on three
movable slice quads; a dynamic volume uploads the current frame's brick on demand.

## 5. Failure modes

`volume_layer` / `vectors_layer` / `objects_layer` raise on no frames or an unknown
mode; `isosurface` returns empty arrays when the level is outside the data; `streamlines`
returns empty arrays for a degenerate field; `pack` raises on an empty scene or when no
layer has geometry; `render_html` raises `FileNotFoundError` without the template.

## 6. Performance / size

A brick costs `4/3` bytes per voxel of page (base64); the default 2.5 M-voxel budget is
~3.3 MB per frame, so a 24-frame dynamic volume is ~80 MB — budgets and `max_frames` are
sockets. Marching cubes on a 2.5 M brick takes ~1 s per frame. Glyph and object layers
cost 28 bytes per row per frame before base64. The page uploads one 3-D texture per
volume layer and re-uploads on frame change only.

## 7. Determinism

Fixed RNG seeds (vector / object subsampling, streamline seeds); the same inputs give the
same page bytes.

## 8. Dependencies

numpy; scipy (`ndimage.map_coordinates`, streamlines only); scikit-image
(`measure.marching_cubes`, iso only) — both lazily imported.

## 9. Pipeline wiring

`scene.place` (optional) → `scene.volume` / `scene.vectors` / `scene.objects` /
`scene.tracks` / `scene.series` (any number, on any number of streams) →
`io.write_scene_viewer` (grow-group inputs). The node converts pixels and plane indices to µm
with each input's calibration and stage log (`catalog/_shared/scene.py`); the kernel never
sees calibration.
