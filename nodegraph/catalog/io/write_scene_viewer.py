"""Export Scene Viewer (``io.write_scene_viewer``) — write the layers that up to eight streams
carry (``scene.*`` nodes) as one self-contained 3-D WebGL scene page, and hand the first
Dataset through."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.scene import frame_spec, layer_specs, tile_offsets

MAX_SCENES = 8
_NODE = "Export Scene Viewer"

_SLOT_DOC = (
    "Another stream whose `scene.*` layers join the same page, in the same µm frame (placed "
    "by its own calibration, stage log and Scene: Place). Slots appear one at a time as you "
    "fill them. A stream with no layer node but an image gets a default volume layer.")


def _pick_frames(n_t: int, max_frames: int) -> List[int]:
    n_t = max(1, int(n_t)); k = max(1, int(max_frames))
    if n_t <= k:
        return list(range(n_t))
    return sorted({int(round(i * (n_t - 1) / (k - 1))) for i in range(k)}) if k > 1 else [0]


def _reduce(values: np.ndarray, how: str) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if how == "count":
        return float(v.size)
    if v.size == 0:
        return float("nan")
    return float({"mean": np.mean, "median": np.median, "sum": np.sum, "max": np.max,
                  "min": np.min}.get(how, np.mean)(v))


class _Frame:
    """One input's world mapping: pixels + plane index → µm in the scene frame."""

    def __init__(self, ds: Dataset, index: int, px: float, dz: float, dt: Optional[float],
                 time_unit: str):
        self.ds, self.index, self.px, self.dz, self.dt = ds, index, float(px), float(dz), dt
        self.time_unit = time_unit
        fr = frame_spec(ds.metadata)
        self.offset = fr["offset_um"]
        offs = tile_offsets(ds.metadata, ds.axes.m, px, flip_x=bool(fr["flip_x"]),
                            flip_y=bool(fr["flip_y"]), layout=str(fr["layout"]), node=_NODE)
        self.oy = np.array([o[0] for o in offs], dtype=float)
        self.ox = np.array([o[1] for o in offs], dtype=float)

    def world(self, m: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        m = np.clip(np.asarray(m, dtype=int), 0, len(self.ox) - 1)
        X = self.offset[0] + (self.ox[m] + np.asarray(x, float)) * self.px
        Y = self.offset[1] + (self.oy[m] + np.asarray(y, float)) * self.px
        Z = self.offset[2] + np.asarray(z, float) * self.dz
        return np.stack([X, Y, Z], axis=1)

    def time(self, t: Any) -> float:
        t = float(t)
        return t * float(self.dt) if (self.time_unit == "s" and self.dt) else t

    def origin_canvas(self) -> Tuple[float, float, float]:
        """``(oz, oy, ox)`` µm of the canvas's voxel (0, 0, 0) — the tile offsets are
        already zero-based, so this is the stream's offset alone."""
        return (self.offset[2], self.offset[1], self.offset[0])


def _table(ds: Dataset, dom: Domain, layer: str, node: str) -> Dict[str, np.ndarray]:
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(dom) if a.layer == layer}
    if not cols:
        raise ValueError(f"{node}: no {dom.value} table {layer!r} on this input — its layer "
                         f"node ran on a wire that no longer carries it. Re-pull the layer node.")
    for req in ("m", "t", "z", "y", "x"):
        if req not in cols:
            raise ValueError(f"{node}: {dom.value} table {layer!r} lacks the coordinate column {req!r}")
    return cols


def _read_pooled(prov, m: int, t: int, c: int, ax, factors: Tuple[int, int, int], block_mean):
    fz, fy, fx = factors
    out = []
    for z0 in range(0, ax.z, fz):
        planes = [np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x), dtype=np.float32)
                  for z in range(z0, min(z0 + fz, ax.z))]
        stack = np.stack(planes, 0)
        out.append(block_mean(stack, (stack.shape[0], fy, fx)))
    return np.concatenate(out, 0)


def _build_volume(SV, F: _Frame, spec: Dict[str, Any], progress: Callable) -> List[Dict[str, Any]]:
    ds = F.ds; prov = ds.image
    if prov is None:
        raise ValueError(f"{_NODE}: input {F.index + 1} asks for a volume layer but carries no image")
    ax = prov.axes
    c = int(spec.get("channel", 0))
    if not (0 <= c < max(1, ax.c)):
        raise ValueError(f"{_NODE}: volume layer {spec.get('name')!r}: channel {c} is out of range "
                         f"for a {ax.c}-channel image")
    n_m = max(1, ax.m)
    budget = max(1000, int(spec.get("voxel_budget", 2_500_000)))
    ts = _pick_frames(ax.t, int(spec.get("max_frames", 24)))
    # ONE brick per channel: every position is pasted onto the canvas its stage offsets
    # span (the Stitch layout), so a 36-position chip is one volume layer, not 36 panels.
    # The budget is the canvas's, and the pooling block is chosen on the canvas shape.
    oy_max = int(F.oy.max()) if n_m > 1 else 0
    ox_max = int(F.ox.max()) if n_m > 1 else 0
    canvas_shape = (ax.z, oy_max + ax.y, ox_max + ax.x)
    factors = SV.downsample_factors(canvas_shape, (F.dz, F.px, F.px), budget)
    fz, fy, fx = factors
    spacing = (F.dz * fz, F.px * fy, F.px * fx)
    pooled_shape = tuple(int(math.ceil(n / f)) for n, f in zip(canvas_shape, factors))
    frames = []
    for t in ts:
        canvas = np.zeros(pooled_shape, np.float32)
        for m in range(n_m):
            progress(f"volume {spec.get('name')!r} position {m} t={t}")
            brick = _read_pooled(prov, m, t, c, ax, factors, SV.block_mean)
            y0 = int(F.oy[m]) // fy if n_m > 1 else 0
            x0 = int(F.ox[m]) // fx if n_m > 1 else 0
            z1 = min(brick.shape[0], pooled_shape[0])
            y1 = min(y0 + brick.shape[1], pooled_shape[1])
            x1 = min(x0 + brick.shape[2], pooled_shape[2])
            region = canvas[:z1, y0:y1, x0:x1]
            np.maximum(region, brick[:z1, :y1 - y0, :x1 - x0], out=region)   # overlaps: brightest wins
        frames.append((F.time(t), canvas))
    name = str(spec.get("name") or f"channel {c}")
    L = SV.volume_layer(
        frames, spacing, F.origin_canvas(), name=name, render=spec.get("render", "mip"),
        colormap=spec.get("colormap", "gray"), color=spec.get("color", "#ffffff"),
        opacity=float(spec.get("opacity", 0.8)),
        window_pct=(float(spec.get("window_low", 0.5)), float(spec.get("window_high", 99.8))),
        iso_level_pct=float(spec.get("iso_level", 90.0)), voxel_budget=10 ** 12)
    L["downsample"] = [int(f) for f in factors]
    L["n_positions"] = int(n_m)
    return [L]


def _build_vectors(SV, F: _Frame, spec: Dict[str, Any], progress: Callable) -> List[Dict[str, Any]]:
    cols = _table(F.ds, Domain.POINT, str(spec.get("source", "")), _NODE)
    if "vx" in cols and "vy" in cols:
        u, v, units = cols["vx"], cols["vy"], "µm/s"
        w = cols.get("vz")
    elif "disp_x" in cols and "disp_y" in cols:
        u, v, units = cols["disp_x"], cols["disp_y"], "µm"
        w = cols.get("disp_z")
    else:
        raise ValueError(f"{_NODE}: Point table {spec.get('source')!r} carries no `vx`/`vy` or "
                         f"`disp_x`/`disp_y` columns")
    u = np.asarray(u, float); v = np.asarray(v, float)
    w = np.asarray(w, float) if w is not None else np.zeros_like(u)
    min_sn = float(spec.get("min_sn", 0.0) or 0.0)
    if min_sn > 0 and "qfactor" in cols:
        bad = ~(np.asarray(cols["qfactor"], float) >= min_sn)
        u = np.where(bad, np.nan, u); v = np.where(bad, np.nan, v); w = np.where(bad, np.nan, w)
    pos = F.world(cols["m"], cols["x"], cols["y"], cols["z"])
    vec = np.stack([u, v, w], 1)
    frames = []
    for t in np.unique(np.asarray(cols["t"], int)):
        sel = np.asarray(cols["t"], int) == t
        frames.append((F.time(t), pos[sel], vec[sel]))
    progress(f"vectors {spec.get('name')!r}")
    return [SV.vectors_layer(
        frames, name=str(spec.get("name") or spec.get("source")), units=units,
        color_by=spec.get("color_by", "magnitude"), colormap=spec.get("colormap", "turbo"),
        color=spec.get("color", "#31d0c6"), scale=float(spec.get("scale", 0.0) or 0.0),
        max_vectors=int(spec.get("max_vectors", 200_000)), with_streamlines=bool(spec.get("lines")),
        max_lines=int(spec.get("max_lines", 800)))]


def _build_objects(SV, F: _Frame, spec: Dict[str, Any], progress: Callable) -> List[Dict[str, Any]]:
    dom = Domain.LABEL if str(spec.get("table", "point")) == "label" else Domain.POINT
    layer = str(spec.get("layer", ""))
    cols = _table(F.ds, dom, layer, _NODE)
    n = len(cols["x"])
    r_um = float(spec.get("radius_um", 0.0) or 0.0)
    if r_um > 0:
        rad = np.full(n, r_um)
    elif "area" in cols:
        area = np.asarray(cols["area"], float)
        if F.ds.structure_zkind(dom, layer) == "subpixel" and F.ds.axes.z > 1:
            rad = np.cbrt(3.0 * area * F.px * F.px * F.dz / (4.0 * np.pi))
        else:
            rad = np.sqrt(area * F.px * F.px / np.pi)
    else:
        rad = np.ones(n)
    val = None
    if str(spec.get("color_by", "uniform")) == "value":
        col = str(spec.get("value_column", ""))
        if col not in cols:
            raise ValueError(f"{_NODE}: {dom.value} table {layer!r} has no column {col!r}")
        val = np.asarray(cols[col], float)
    pos = F.world(cols["m"], cols["x"], cols["y"], cols["z"])
    frames = []
    for t in np.unique(np.asarray(cols["t"], int)):
        sel = np.asarray(cols["t"], int) == t
        frames.append((F.time(t), pos[sel], rad[sel], val[sel] if val is not None else None))
    progress(f"objects {spec.get('name')!r}")
    return [SV.objects_layer(
        frames, name=str(spec.get("name") or layer), color_by=spec.get("color_by", "uniform"),
        colormap=spec.get("colormap", "viridis"), color=spec.get("color", "#ffd43b"),
        value_label=str(spec.get("value_column", "")), size_scale=float(spec.get("size_scale", 1.0) or 1.0),
        max_objects=int(spec.get("max_objects", 200_000)))]


def _build_tracks(SV, F: _Frame, spec: Dict[str, Any], progress: Callable) -> List[Dict[str, Any]]:
    ds = F.ds
    src = str(spec.get("source", ""))
    tcols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.TRACK) if a.layer == src}
    for req in ("track_id", "t", "member_id"):
        if req not in tcols:
            raise ValueError(f"{_NODE}: Track table {src!r} lacks the column {req!r}")
    dom = Domain.LABEL if str(spec.get("members", "point")) == "label" else Domain.POINT
    mcols = _table(ds, dom, str(spec.get("member_layer", "")), _NODE)
    if "id" not in mcols:
        raise ValueError(f"{_NODE}: member table {spec.get('member_layer')!r} has no `id` column")
    mid = np.asarray(mcols["id"]); mt = np.asarray(mcols["t"], int)
    by_id_t: Dict[Tuple[int, int], int] = {}
    by_id: Dict[int, int] = {}
    for i in range(len(mid)):
        by_id_t.setdefault((int(mid[i]), int(mt[i])), i)
        by_id.setdefault(int(mid[i]), i)
    pos = F.world(mcols["m"], mcols["x"], mcols["y"], mcols["z"])
    tid = np.asarray(tcols["track_id"]); tt = np.asarray(tcols["t"], int); mem = np.asarray(tcols["member_id"])
    order = np.lexsort((tt, tid))
    tracks: List[Tuple[int, np.ndarray, np.ndarray]] = []
    cur, times, pts = None, [], []
    for k in order:
        row = by_id_t.get((int(mem[k]), int(tt[k])), by_id.get(int(mem[k])))
        if row is None:
            continue
        if cur is not None and tid[k] != cur:
            tracks.append((int(cur), np.asarray(times), np.asarray(pts)))
            times, pts = [], []
        cur = tid[k]
        times.append(F.time(tt[k])); pts.append(pos[row])
    if cur is not None and pts:
        tracks.append((int(cur), np.asarray(times), np.asarray(pts)))
    progress(f"tracks {spec.get('name')!r}")
    return [SV.tracks_layer(
        tracks, name=str(spec.get("name") or src), color_by=spec.get("color_by", "track"),
        colormap=spec.get("colormap", "turbo"), color=spec.get("color", "#e599f7"),
        tail=float(spec.get("tail", 0.0) or 0.0), max_tracks=int(spec.get("max_tracks", 5000)))]


def _build_series(SV, F: _Frame, spec: Dict[str, Any], progress: Callable) -> List[Dict[str, Any]]:
    dom = Domain.LABEL if str(spec.get("table", "point")) == "label" else Domain.POINT
    layer = str(spec.get("layer", ""))
    cols = _table(F.ds, dom, layer, _NODE)
    col = str(spec.get("column", ""))
    if col not in cols:
        raise ValueError(f"{_NODE}: {dom.value} table {layer!r} has no column {col!r}")
    vals = np.asarray(cols[col], float); t = np.asarray(cols["t"], int); m = np.asarray(cols["m"], int)
    how = str(spec.get("reducer", "mean"))
    groups = sorted(set(m.tolist())) if bool(spec.get("per_position")) else [None]
    curves = []
    for g in groups:
        sel = np.ones(len(t), bool) if g is None else (m == g)
        ts = np.unique(t[sel])
        y = np.array([_reduce(vals[sel & (t == tv)], how) for tv in ts])
        x = np.array([F.time(tv) for tv in ts])
        curves.append((f"P{g}" if g is not None else f"{how}({col})", x, y))
    progress(f"series {spec.get('name')!r}")
    return [SV.series_layer(
        curves, name=str(spec.get("name") or f"{col} ({layer})"),
        x_label="time (s)" if F.time_unit == "s" else "frame", y_label=f"{how} {col}",
        x_is_time=True, color=spec.get("color", "#31d0c6"))]


_BUILDERS = {"volume": _build_volume, "vectors": _build_vectors, "objects": _build_objects,
             "tracks": _build_tracks, "series": _build_series}


def _compute_write_scene_viewer(ctx: EvalContext) -> Dataset:
    """Write every wired stream's scene layers as one 3-D WebGL page and hand the first
    Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** io / side effect → ``op_key="io.write_scene_viewer"``, category ``io`` — the
      composable sibling of ``io.write_flow_viewer``, and like it a TAP: input 1 passes
      through byte-identical plus a ``scene_viewer_report``.
    * **Data contract** up to eight Datasets through a grow group. Each carries the specs
      its ``scene.*`` nodes appended to ``scene_layers`` (a stream with no spec but an image
      gets a default MIP volume; one with neither is refused by slot number) and its
      ``scene_frame`` placement. Every layer is converted to world geometry in µm with that
      input's own calibration: ``pixel_size_um`` (required), ``z_step_um`` (required for
      more than one plane — never a guessed z-stretch), positions by ``stage_xy_um``
      (Stitch's mapping) unless Scene: Place says otherwise. Time is seconds when every
      dynamic input carries ``dt_s``, else the frame index for all. Input 1's calibration is
      read through ``ctx.calib`` so the memo fences on it; the others ride their payloads,
      whose revisions already fold into the recipe hash.
    * **What is read** volumes one ``(m, t, c)`` plane at a time through ``get_region``,
      mean-pooled to the layer's voxel budget as they stream in; structure tables whole.
    * **Footprint** ``MULTI_VIEW`` / every axis.
    * **Backend** :mod:`nodegraph.kernels.scene_viewer` (numpy, scipy map_coordinates for
      streamlines, scikit-image marching cubes); the page is ``scene_viewer_template.html``
      (WebGL2, no dependencies).
    * **Params read** ``path``, ``title``, ``subtitle``; mode ``existing``.
    """
    from nodegraph.kernels import scene_viewer as SV

    inputs = [d for d in ctx.inputs if isinstance(d, Dataset)]
    if not inputs:
        raise ValueError(f"{_NODE}: nothing wired — connect at least one stream carrying scene layers")
    modes = ctx.params.get("__modes__", {}) or {}
    existing = str(modes.get("existing", "overwrite"))
    raw = str(ctx.params.get("path", "") or "").strip().strip('"')
    if not raw:
        raise ValueError(f"{_NODE} has no destination: set `path` (the Browse… button beside it "
                         f"picks one). There is deliberately no default.")
    if os.path.isdir(raw) or raw.endswith(("/", "\\")):
        raise ValueError(f"{_NODE}: {raw!r} is a folder — add a file name")
    target = raw if raw.lower().endswith((".html", ".htm")) else raw + ".html"
    if existing == "refuse" and os.path.exists(target):
        raise ValueError(f"{_NODE}: {target} already exists and Existing is 'refuse' — move it, "
                         f"pick another name, or set Existing to overwrite")
    title = str(ctx.params.get("title", "") or "").strip() or os.path.splitext(os.path.basename(target))[0]
    subtitle = str(ctx.params.get("subtitle", "") or "").strip()

    def calib(i: int, key: str):
        return ctx.calib(key) if i == 0 else (inputs[i].metadata or {}).get(key)

    # the time unit is one decision for the whole page
    dynamic = [i for i, d in enumerate(inputs)
               if d.axes.t > 1 or any(a.domain is Domain.TRACK for a in d.attributes.values())]
    time_unit = "s" if dynamic and all(calib(i, "dt_s") for i in dynamic) else "frame"

    plans: List[Tuple[_Frame, Dict[str, Any]]] = []
    for i, d in enumerate(inputs):
        specs = layer_specs(d.metadata)
        if not specs:
            if d.image is None:
                raise ValueError(
                    f"{_NODE}: scene input {i + 1} carries no scene layer (no `scene.*` node "
                    f"upstream) and no image to show by default — add Scene: Volume / Vectors / "
                    f"Objects / Tracks / Series on that stream, or unwire it")
            specs = [{"kind": "volume", "name": "", "channel": 0, "render": "mip", "colormap": "gray",
                      "color": "#ffffff", "opacity": 0.8, "window_low": 0.5, "window_high": 99.8,
                      "voxel_budget": 2_500_000, "max_frames": 24}]
        px = calib(i, "pixel_size_um")
        if not px:
            raise ValueError(f"{_NODE}: scene input {i + 1} has no `pixel_size_um` — the scene is "
                             f"in µm, so a pixel size is required (set it with Set Calibration)")
        dz = calib(i, "z_step_um")
        if d.axes.z > 1 and not dz:
            raise ValueError(f"{_NODE}: scene input {i + 1} has {d.axes.z} planes but no "
                             f"`z_step_um` — a z-stack cannot be placed without its plane spacing "
                             f"(set it with Set Calibration; a guessed z-stretch is never used)")
        F = _Frame(d, i, float(px), float(dz or px), calib(i, "dt_s"), time_unit)
        for spec in specs:
            kind = str(spec.get("kind", ""))
            if kind not in _BUILDERS:
                raise ValueError(f"{_NODE}: unknown scene layer kind {kind!r} on input {i + 1}")
            plans.append((F, spec))

    total = max(1, len(plans))
    done = [0]
    layers: List[Dict[str, Any]] = []

    def progress(note: str):
        ctx.progress(done[0], total, note)

    for F, spec in plans:
        layers.extend(_BUILDERS[str(spec["kind"])](SV, F, spec, progress))
        done[0] += 1
        ctx.progress(done[0], total, f"{spec['kind']} {spec.get('name')!r} ready")
    scene = SV.pack(layers, title=title, subtitle=subtitle, time_unit=time_unit)
    html = SV.render_html(scene, title=title, subtitle=subtitle,
                          snapshot_name=os.path.splitext(os.path.basename(target))[0] + ".png")
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    part = target + ".part"
    with open(part, "w", encoding="utf-8") as fh:
        fh.write(html)
    os.replace(part, target)
    rep = SV.scene_report(scene)
    rep.update(path=target, bytes=os.path.getsize(target), time_unit=time_unit, n_inputs=len(inputs))
    ctx.progress(total, total, f"wrote {os.path.basename(target)} ({rep['bytes'] / 1e6:.1f} MB)")
    return inputs[0].with_metadata(scene_viewer_report=rep)


register_node(
    _compute_write_scene_viewer, op_key="io.write_scene_viewer",
    label="Export Scene Viewer (3D HTML)", category="io",
    inputs=[
        InDataset("scene", label="Scene 1",
                  description="The first stream whose `scene.*` layers go on the page; it is "
                              "the Dataset handed through (plus the `scene_viewer_report`). "
                              "Its calibration is read through the memo fence."),
        *[InDataset(f"scene_{i}", label=f"Scene {i}", grow_group="scenes", passes_domains=False,
                    description=_SLOT_DOC) for i in range(2, MAX_SCENES + 1)],
        InString("path", "File", field=False, default="", path_kind="save_file",
                 path_filter="HTML (*.html);;All files (*)",
                 path_hint="Browse… to choose where the scene page is written",
                 description="Where the self-contained page is written (one `.html`, opens in "
                             "any browser with no server). No default; empty is refused; "
                             "`.html` is added when missing."),
        InString("title", "Title", field=False, default="",
                 description="The heading in the page's panel and browser tab. Empty uses "
                             "the file name. Cosmetic."),
        InString("subtitle", "Subtitle", field=False, default="",
                 description="One line under the title — the experiment, the date, what the "
                             "layers are. Cosmetic."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description="What to do when a file is already at the path.",
             choice_docs={
                 "overwrite": "Replace it (written to `.part` first, so an interrupted export "
                              "never leaves a truncated page under the real name).",
                 "refuse": "Raise instead of touching an existing file, so a page already "
                           "published is never silently replaced.",
             }),
    ],
    granularity=Granularity.MULTI_VIEW, kernel_axes=frozenset({"m", "t", "z", "c", "y", "x"}),
    description="Write the scene layers of up to eight streams (volumes, vector fields, "
                "objects, tracks, charts — each placed in one µm frame by its own calibration "
                "and stage log) as one self-contained 3-D WebGL page with a timeline, and "
                "hand the first Dataset through unchanged.")
