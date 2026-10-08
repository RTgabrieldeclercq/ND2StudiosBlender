"""Shared helpers for the 3-D HTML viewers (``io.write_flow_viewer``, the ``scene.*`` layer
nodes and ``io.write_scene_viewer``, V4.10).

Three things live here because two or more nodes need the same answer:

* **Placement** — where each multipoint position sits on one canvas, from the payload's
  ``stage_xy_um`` with Stitch's mapping and handedness flags (:func:`tile_offsets`), and the
  pitch of a gridded point field (:func:`grid_pitch`).
* **Layer specs on the wire** — a ``scene.*`` node is a tap that appends one small dict to
  the Dataset's ``scene_layers`` metadata and hands the payload through; the sink reads the
  list. The same function writes the spec at edit time (``meta_transform``) and at pull time
  (the compute), which is what keeps the envelope and the payload in lockstep
  (:func:`appended_layers`, :func:`spec_meta_transform`).
* **The stream's frame** — ``scene.place`` writes ``scene_frame`` (offsets, flips, layout);
  :func:`frame_spec` reads it with the defaults every other stream gets.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

#: Dataset / envelope metadata key holding the tuple of layer specs
SCENE_LAYERS_KEY = "scene_layers"
#: Dataset / envelope metadata key holding the stream's placement
SCENE_FRAME_KEY = "scene_frame"

DEFAULT_FRAME: Dict[str, Any] = {
    "layout": "stage", "flip_x": True, "flip_y": False, "offset_um": [0.0, 0.0, 0.0],
}


def tile_offsets(metadata: Optional[Mapping[str, Any]], n_m: int, px_um: float, *,
                 flip_x: bool = True, flip_y: bool = False, layout: str = "stage",
                 node: str = "Export Scene Viewer") -> List[Tuple[int, int]]:
    """Per-position ``(oy, ox)`` pixel offsets of each tile on one canvas.

    ``layout == "stage"``: from ``stage_xy_um`` exactly as Stitch places tiles — image +x/+y
    run along stage +x/+y, mirrored per flag — with one position at ``(0, 0)``; refuses by
    name when the log is missing for any position. ``layout == "origin"``: every position at
    ``(0, 0)`` (one-position data, or positions that really are the same place)."""
    n_m = int(max(1, n_m))
    if n_m == 1 or layout == "origin":
        return [(0, 0)] * n_m
    xy = (metadata or {}).get("stage_xy_um") or []
    if len(xy) < n_m:
        raise ValueError(
            f"{node}: {n_m} positions but no stage log to place them with — the Dataset "
            f"carries no `stage_xy_um` for every position. Load the original file (ND2 "
            f"stage coordinates ride the payload), export one position at a time (Select "
            f"Position), or set Scene: Place to layout 'origin' to stack them.")
    xs = np.array([float(p[0]) for p in xy[:n_m]])
    ys = np.array([float(p[1]) for p in xy[:n_m]])
    u = (xs.max() - xs) if flip_x else (xs - xs.min())
    v = (ys.max() - ys) if flip_y else (ys - ys.min())
    oy = np.rint(v / float(px_um)).astype(int)
    ox = np.rint(u / float(px_um)).astype(int)
    oy -= oy.min()
    ox -= ox.min()
    return list(zip(oy.tolist(), ox.tolist()))


def grid_pitch(x: np.ndarray, y: np.ndarray) -> float:
    """The window pitch of a gridded point field in pixels: the median spacing of the
    distinct x (then y) coordinates. One distinct value → 1 px, so a degenerate table still
    exports."""
    for arr in (x, y):
        u = np.unique(np.round(np.asarray(arr, dtype=float), 3))
        if u.size >= 2:
            d = np.diff(u)
            d = d[d > 1e-6]
            if d.size:
                return float(np.median(d))
    return 1.0


def layer_specs(metadata: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The layer specs a stream carries, oldest first (a list copy; never the stored tuple)."""
    raw = (metadata or {}).get(SCENE_LAYERS_KEY) or ()
    return [dict(s) for s in raw if isinstance(s, Mapping)]


def appended_layers(metadata: Optional[Mapping[str, Any]], spec: Mapping[str, Any]
                    ) -> Tuple[Dict[str, Any], ...]:
    """The ``scene_layers`` value after appending ``spec`` — a tuple, so the metadata stays
    hashable and copy-on-write like every other key."""
    return tuple(layer_specs(metadata)) + (dict(spec),)


def frame_spec(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The stream's placement: ``scene.place``'s values, else :data:`DEFAULT_FRAME`."""
    out = dict(DEFAULT_FRAME)
    raw = (metadata or {}).get(SCENE_FRAME_KEY)
    if isinstance(raw, Mapping):
        out.update({k: raw[k] for k in DEFAULT_FRAME if k in raw})
    out["offset_um"] = [float(v) for v in list(out.get("offset_um") or [0, 0, 0])[:3]]
    while len(out["offset_um"]) < 3:
        out["offset_um"].append(0.0)
    return out


def _layer_names(op_key: str, keys: Tuple[str, ...], params: Mapping[str, Any]) -> Dict[str, str]:
    """Resolve layer-name sockets the way the compute does (:func:`nodegraph.registry
    .layer_value`: the user's override, else the socket's declared default), so the
    edit-time spec and the payload spec name the same layer and the default lives once."""
    from nodegraph.registry import NODES, layer_value
    spec = NODES.get(op_key) if op_key else None
    socks = {s.name: s for s in (spec.inputs if spec is not None else ())}
    return {k: layer_value(socks.get(k), params or {}) for k in keys}


def spec_meta_transform(build: Callable[..., Dict[str, Any]], name: str, *, op_key: str = "",
                        layer_keys: Tuple[str, ...] = ()):
    """A ``meta_transform`` that appends ``build(params, modes, **layer_names)`` to the envelope's
    ``scene_layers`` — the edit-time half of a layer node. Total: a builder that raises on
    half-typed params leaves the envelope unchanged rather than breaking the inspector."""
    def transform(env, params, modes):
        try:
            spec = build(params or {}, modes or {}, **_layer_names(op_key, layer_keys, params or {}))
        except Exception:                        # pragma: no cover - defensive, edit time
            return env
        return env.with_metadata(**{SCENE_LAYERS_KEY: appended_layers(env.metadata, spec)})
    transform.__name__ = name
    transform.__doc__ = f"Append the {name} layer spec to `scene_layers` (edit time)."
    return transform


def frame_meta_transform(build: Callable[[Mapping[str, Any], Mapping[str, str]], Dict[str, Any]]):
    """The edit-time half of ``scene.place``: set the envelope's ``scene_frame``."""
    def scene_place(env, params, modes):
        try:
            frame = build(params or {}, modes or {})
        except Exception:                        # pragma: no cover - defensive, edit time
            return env
        return env.with_metadata(**{SCENE_FRAME_KEY: frame})
    scene_place.__doc__ = "Set the stream's `scene_frame` placement (edit time)."
    return scene_place


__all__ = [
    "SCENE_LAYERS_KEY", "SCENE_FRAME_KEY", "DEFAULT_FRAME", "tile_offsets", "grid_pitch",
    "layer_specs", "appended_layers", "frame_spec", "spec_meta_transform",
    "frame_meta_transform",
]
