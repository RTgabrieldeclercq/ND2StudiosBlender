"""Export Flow Viewer (``io.write_flow_viewer``) — write a PIV / DIC point field as the lab's
self-contained 3-D WebGL flow viewer (one HTML file), and hand the Dataset through."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (Granularity, InBool, InDataset, InFloat, InInt, InString,
                               Mode, OutDataset)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _point_layers, _resolve_layer
from nodegraph.catalog._shared.scene import grid_pitch as _grid_pitch
from nodegraph.catalog._shared.scene import tile_offsets

#: Downsampling of the grain planes handed to the detector (the deployed pipeline's value).
_GRAIN_DS = 4


def _tile_offsets(ds: Dataset, n_m: int, px_um: float, tile_h: int, tile_w: int,
                  flip_x: bool, flip_y: bool) -> List[Tuple[int, int]]:
    """Per-multipoint ``(oy, ox)`` pixel offsets of each tile on one canvas — the shared
    Stitch mapping (:func:`nodegraph.catalog._shared.scene.tile_offsets`), refusing by
    this node's name when the stage log is missing."""
    return tile_offsets(getattr(ds, "metadata", {}) or {}, n_m, px_um, flip_x=flip_x,
                        flip_y=flip_y, node="Export Flow Viewer")


def _compute_write_flow_viewer(ctx: EvalContext) -> Dataset:
    """Write the 3-D flow viewer of a point velocity field and hand the Dataset through.

    Resolved spec (`build-node-v2` §0, 2026-10-08)
    ----------------------------------------------
    * **Kind** io / side effect → ``op_key="io.write_flow_viewer"``, category ``io`` — the
      presentation sibling of Export TIFF / Export Movie, and like them a TAP: the Dataset
      passes through byte-identical so a Viewer after it still shows the field.
    * **Data contract** reads ONE Point field (``source``, a PIV / DIC output: rows on the
      interrogation-window grid with ``vx``/``vy`` in µm/s, or ``disp_x``/``disp_y`` in µm
      when no velocity was computed) across every position, timepoint and plane, plus —
      for the grain pack — a Voxel mask (nonzero = grain) or the image's own channel.
      Writes one HTML file; adds nothing to the wire, so no ``meta_transform``.
    * **How it is built** (the lab's deployed granular-flow pipeline, ported to
      :mod:`nodegraph.kernels.flow_viewer`): the per-plane vectors are placed on ONE regular
      grid whose cell is the window pitch — positions laid out by their stage coordinates
      like Stitch (``flip_x``/``flip_y`` are its handedness flags) — timepoints averaged per
      cell; the out-of-plane velocity ``w`` is estimated from continuity
      (``dw/dz = -(du/dx + dv/dy)``, regularised least squares or plain integration) at the
      plane spacing the calibration gives (``s = z_step_um / cell_um``, so no guessed
      z-stretch); the volume is PCHIP-interpolated in z; streamlines, the high-speed
      iso-surface, 3-D grains (per-plane watershed, linked across planes), DTI-style bundles
      and the vessel skeleton are precomputed and embedded as base64 buffers in the page.
      Velocities keep their units (the page labels them); coordinates are grid cells.
    * **2D/3D** no lever: a single plane exports a flat field (``w = 0``, ``s = 1``); a
      z-stack exports a volume. Derived from the data, never asked.
    * **Footprint** ``MULTI_VIEW`` / ``{m, t, z, y, x}``: the point rows of every position,
      timepoint and plane are read at once (they are one table), and the grain planes are
      read one ``(m, 0, z, c)`` plane at a time through ``get_region``, so the image is
      never materialised.
    * **Backend** numpy + scipy (sparse CG for ``w``, PCHIP, RK4 streamlines, EDT),
      scikit-image (marching cubes, watershed, 3-D skeletonize); no numba — the hot paths
      are vectorised or compiled library calls.
    * **Params read** ``path``, ``title``, ``source``, ``min_sn``, ``flip_x``, ``flip_y``,
      ``grain_mask`` (mask branch), ``grain_channel`` (image branch), ``grain_min_diameter``
      (both grain branches), ``smooth``,
      ``lambda_z``/``lambda_xy`` (regularised branch), ``z_upsample``, ``iso_percentile``,
      ``vessel_percentile``, ``seed_stride``, ``max_lines``, ``seed_min_fraction``; modes
      ``w_mode``, ``grains``, ``existing``.
    * **Caveat carried into the page** ``w`` is a structural estimate: all measured 2-D
      divergence is attributed to out-of-plane flow, and a granular pack also compacts; the
      page's stats panel shows ``|w|/|u|`` and the high-speed volume fraction so the
      reader can judge it. Written atomically through ``<path>.part``.
    """
    from nodegraph.kernels import flow_viewer as FV

    ds = ctx.inputs[0]
    ax = ds.axes
    modes = ctx.params.get("__modes__", {}) or {}
    w_mode = str(modes.get("w_mode", "regularized"))
    grains_mode = str(modes.get("grains", "image"))
    existing = str(modes.get("existing", "overwrite"))

    raw = str(ctx.params.get("path", "") or "").strip().strip('"')
    if not raw:
        raise ValueError("Export Flow Viewer has no destination: set `path` (the Browse… "
                         "button beside it picks one). There is deliberately no default.")
    if os.path.isdir(raw) or raw.endswith(("/", "\\")):
        raise ValueError(f"Export Flow Viewer: {raw!r} is a folder — add a file name")
    target = raw if raw.lower().endswith((".html", ".htm")) else raw + ".html"
    if existing == "refuse" and os.path.exists(target):
        raise ValueError(f"Export Flow Viewer: {target} already exists and Existing is "
                         f"'refuse' — move it, pick another name, or set Existing to overwrite")

    # ── the point field ───────────────────────────────────────────────────────
    source, _note = _resolve_layer(
        _point_layers(ds), ctx.layer("source"), node="Export Flow Viewer", socket="source",
        what="Point table", where="the `data` input",
        remedy="this draws a PIV / DIC vector field, so run analysis.piv (or DIC) upstream",
        ctx=ctx)
    col = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
           if a.layer == source}
    for req in ("m", "t", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} is missing the coordinate column {req!r}")
    if "vx" in col and "vy" in col:
        u_col, v_col, units = col["vx"], col["vy"], "µm/s"
    elif "disp_x" in col and "disp_y" in col:
        u_col, v_col, units = col["disp_x"], col["disp_y"], "µm per frame"
    else:
        raise ValueError(
            f"Point layer {source!r} carries no velocity: it needs `vx`/`vy` (µm/s — turn "
            f"on Velocity in PIV) or `disp_x`/`disp_y` (µm). Columns here: "
            f"{sorted(col)}")
    u_col = u_col.astype(float); v_col = v_col.astype(float)
    min_sn = float(ctx.params.get("min_sn", 0.0) or 0.0)
    if min_sn > 0 and "qfactor" in col:
        bad = ~(np.asarray(col["qfactor"], float) >= min_sn)
        u_col = np.where(bad, np.nan, u_col); v_col = np.where(bad, np.nan, v_col)

    # ── calibration → cell size, plane spacing, z stretch ─────────────────────
    px = ctx.calib("pixel_size_um")
    px_um = float(px) if px else 1.0
    pitch_px = _grid_pitch(col["x"], col["y"])
    cell_um = pitch_px * px_um
    nz = int(ax.z)
    zs = ctx.calib("z_step_um")
    if nz > 1 and not zs:
        raise ValueError(
            f"Export Flow Viewer: the field spans {nz} planes but the Dataset has no "
            f"`z_step_um`, so the plane spacing (and the out-of-plane velocity) cannot be "
            f"placed. Load a file that records its z step, or Z-project first.")
    s = float(zs) / cell_um if nz > 1 else 1.0

    # ── one canvas: positions placed by their stage coordinates ───────────────
    flip_x = bool(ctx.params.get("flip_x", True))
    flip_y = bool(ctx.params.get("flip_y", False))
    offs = _tile_offsets(ds, int(ax.m), px_um, int(ax.y), int(ax.x), flip_x, flip_y)
    m_all = np.asarray(col["m"]).astype(int)
    oy = np.array([o[0] for o in offs], float); ox = np.array([o[1] for o in offs], float)
    m_ok = (m_all >= 0) & (m_all < len(offs))
    gx = (ox[np.clip(m_all, 0, len(offs) - 1)] + np.asarray(col["x"], float)) / pitch_px
    gy = (oy[np.clip(m_all, 0, len(offs) - 1)] + np.asarray(col["y"], float)) / pitch_px
    ix = np.rint(gx - gx[m_ok].min()).astype(int) if m_ok.any() else np.zeros(0, int)
    iy = np.rint(gy - gy[m_ok].min()).astype(int) if m_ok.any() else np.zeros(0, int)
    iz = np.rint(np.asarray(col["z"], float)).astype(int)
    canvas_h = max(o[0] for o in offs) + int(ax.y)
    canvas_w = max(o[1] for o in offs) + int(ax.x)
    nx_cells = int(np.ceil(canvas_w / pitch_px)); ny_cells = int(np.ceil(canvas_h / pitch_px))
    if m_ok.any():
        nx_cells = max(nx_cells, int(ix[m_ok].max()) + 1)
        ny_cells = max(ny_cells, int(iy[m_ok].max()) + 1)
    ctx.progress(0, 8, f"placing {int(m_ok.sum())} vectors on a {nz}×{ny_cells}×{nx_cells} grid")
    U, V, _n = FV.assemble_grid(iz[m_ok], iy[m_ok], ix[m_ok], u_col[m_ok], v_col[m_ok],
                                (nz, ny_cells, nx_cells))
    if not np.isfinite(U).any():
        raise ValueError("Export Flow Viewer: the point field has no finite velocity to draw "
                         "(every vector was dropped — lower `min_sn`, or check the field)")
    # cell offset of the canvas origin: vectors sit at window CENTRES, so the first row/col
    # of the cell grid starts at the minimum vector coordinate
    gx0 = float(gx[m_ok].min()) * pitch_px; gy0 = float(gy[m_ok].min()) * pitch_px

    # ── grain planes (downsampled canvas per z) ───────────────────────────────
    grain_planes: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None
    if grains_mode != "off":
        gh = int(np.ceil((ny_cells * pitch_px) / _GRAIN_DS)) + 1
        gw = int(np.ceil((nx_cells * pitch_px) / _GRAIN_DS)) + 1
        mask_arr = None; prov = None; ch = 0
        if grains_mode == "mask":
            layer = ctx.layer("grain_mask")
            if not layer:
                raise ValueError("Export Flow Viewer: Grains is 'mask' but `grain_mask` names "
                                 "no Voxel layer — name the grain mask, or switch Grains to "
                                 "'image' / 'off'")
            got = ds.get(Domain.VOXEL, layer)
            if got is None:
                have = sorted({a.name for a in ds.layers_on(Domain.VOXEL)})
                raise ValueError(f"no Voxel layer {layer!r} for the grains — this Dataset "
                                 f"carries {have or 'none'}")
            mask_arr = np.asarray(got.values)
            if mask_arr.shape != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
                raise ValueError(f"Voxel layer {layer!r} has shape {mask_arr.shape}, not the "
                                 f"Dataset's {(ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)}")
        else:
            prov = ds.image
            if prov is None:
                raise ValueError("Export Flow Viewer: Grains is 'image' but the Dataset has "
                                 "no image to detect them on — switch Grains to 'off', or "
                                 "name a mask")
            ch = int(ctx.params.get("grain_channel", 0) or 0)
            if not (0 <= ch < int(ax.c)):
                raise ValueError(f"grain_channel {ch} is out of range for {int(ax.c)} channel(s)")
        grain_planes = []
        for z in range(nz):
            ctx.progress(1, 8, f"grain plane {z + 1}/{nz}")
            canvas = np.zeros((gh, gw), np.float32)
            for m in range(int(ax.m)):
                if mask_arr is not None:
                    plane = np.asarray(mask_arr[m, 0, z, 0] != 0, np.float32)
                else:
                    plane = np.asarray(prov.get_region(0, m, 0, z, ch, 0, int(ax.y), 0,
                                                       int(ax.x)), np.float32)
                py0 = int(round((offs[m][0] - gy0) / _GRAIN_DS))
                px0 = int(round((offs[m][1] - gx0) / _GRAIN_DS))
                sub = plane[::_GRAIN_DS, ::_GRAIN_DS]
                y0, x0 = max(py0, 0), max(px0, 0)
                sy, sx = y0 - py0, x0 - px0
                y1 = min(gh, py0 + sub.shape[0]); x1 = min(gw, px0 + sub.shape[1])
                if y1 > y0 and x1 > x0:
                    canvas[y0:y1, x0:x1] = np.maximum(
                        canvas[y0:y1, x0:x1], sub[sy:sy + (y1 - y0), sx:sx + (x1 - x0)])
            if mask_arr is not None:
                grain_planes.append((None, canvas > 0.5))
            else:
                grain_planes.append((canvas, None))

    # ── the pipeline ──────────────────────────────────────────────────────────
    smooth_um = float(ctx.params.get("smooth", 0.0) or 0.0)
    smooth_sigma = (smooth_um / cell_um) if smooth_um > 0 else 1.5
    # grain size floor: µm → the detector's area / peak-separation in downsampled px
    d_um = float(ctx.params.get("grain_min_diameter", 0.0) or 0.0)
    if d_um <= 0:
        d_um = 3.0 * cell_um
    ds_px_um = _GRAIN_DS * px_um
    grain_min_area = max(1, int(round(np.pi * (0.5 * d_um / ds_px_um) ** 2)))
    grain_min_dist = max(1, int(round(0.5 * d_um / ds_px_um)))
    plane_names = ([f"z {k * float(zs):g} µm" for k in range(nz)] if nz > 1 and zs
                   else ["plane"])
    geometry, report = FV.build_geometry(
        U, V, s=s, px_per_cell=pitch_px, plane_names=plane_names, w_mode=w_mode,
        smooth_sigma=smooth_sigma,
        lam_z=float(ctx.params.get("lambda_z", 0.2)),
        lam_xy=float(ctx.params.get("lambda_xy", 0.1)),
        z_upsample=int(ctx.params.get("z_upsample", 8)),
        iso_pctl=float(ctx.params.get("iso_percentile", 90.0)),
        vessel_pctl=float(ctx.params.get("vessel_percentile", 70.0)),
        seed_stride=max(1, int(ctx.params.get("seed_stride", 11))),
        max_lines=max(1, int(ctx.params.get("max_lines", 1400))),
        seed_min_fraction=float(ctx.params.get("seed_min_fraction", 0.1)),
        grain_planes=grain_planes, grain_ds=_GRAIN_DS, grain_min_area=grain_min_area,
        grain_min_dist=grain_min_dist,
        progress=lambda k, n, text: ctx.progress(2 + k, 9, text))

    title = str(ctx.params.get("title", "") or "").strip()
    if not title:
        title = os.path.splitext(os.path.basename(target))[0].replace("_", " ") + " — 3D flow"
    sub = (f"{int(ax.m)} position(s) · {nz} plane(s) · {int(report['n_lines'])} streamlines · "
           f"cell {cell_um:.3g} µm · velocities in {units}.")
    html = FV.render_html(geometry, report, title=title, subtitle=sub,
                          units=f"flow speed ({units})", cell_um=cell_um,
                          snapshot_name=os.path.splitext(os.path.basename(target))[0] + ".png")
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    part = target + ".part"
    ctx.progress(8, 9, f"writing {os.path.basename(target)}")
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    os.replace(part, target)
    ctx.progress(9, 9, f"wrote {os.path.basename(target)} ({len(html) / 1e6:.1f} MB)")
    return ds.with_metadata(flow_viewer_report={**report, "path": target, "units": units,
                                               "cell_um": round(cell_um, 6)})


register_node(
    _compute_write_flow_viewer, op_key="io.write_flow_viewer",
    label="Export Flow Viewer (3D HTML)", category="io",
    reads_domains=frozenset({Domain.POINT}),
    reads_domains_by_mode={"grains": {"mask": frozenset({Domain.VOXEL})}},
    inputs=[
        InDataset("data", description="The Dataset carrying the point velocity field (a PIV or "
                                      "DIC output) — and, for the grain pack, a Voxel mask or "
                                      "the image. It is handed on unchanged."),
        InString("path", "File", field=False, default="", path_kind="save_file",
                 path_filter="HTML (*.html);;All files (*)",
                 path_hint="Browse… to choose where the viewer page is written",
                 description="Where the self-contained viewer is written (one `.html` file, a "
                             "few MB, opens in any browser with no server). There is no "
                             "default and an empty value is refused; `.html` is added when "
                             "missing."),
        InString("title", "Title", field=False, default="",
                 description="The heading shown in the page's control panel and browser tab. "
                             "Empty uses the file name. Cosmetic: nothing in the data changes."),
        InString("source", "Point layer", field=False, default="piv", layer_in=Domain.POINT,
                 description="Which Point field to draw — a PIV output (one row per "
                             "interrogation window with `vx`/`vy` in µm/s, or `disp_x`/`disp_y` "
                             "in µm when Velocity was off) or a DIC field. With one point table "
                             "on the wire the name is inferred; two need the name."),
        InFloat("min_sn", "Min S/N", unit="", field=False, default=0.0,
                description="Drop vectors whose correlation signal-to-noise (`qfactor`) is "
                            "below this before the field is gridded; the hole is inpainted "
                            "from its neighbours or, with nothing near, read as solid. 0 "
                            "keeps every vector. Inert on a field without a `qfactor` column "
                            "(DIC). RAISING it removes junk from featureless windows but "
                            "thins sparse regions."),
        InBool("flip_x", "Flip X", field=False, default=True,
               description="Whether a position's image +x runs along stage −x when several "
                           "positions are laid out on one canvas by their stage "
                           "coordinates (the same flag as Stitch, same default — this "
                           "lab's scopes). Wrong = tiles mirrored and the field torn at "
                           "every seam. Inert for a single position."),
        InBool("flip_y", "Flip Y", field=False, default=False,
               description="As Flip X, for the Y axis: image +y along stage −y when ON. "
                           "Default OFF matches Stitch and this lab's scopes. Inert for a "
                           "single position."),
        InString("grain_mask", "Grain mask", field=False, default="", layer_in=Domain.VOXEL,
                 available_in={"grains": frozenset({"mask"})},
                 description="The Voxel mask whose nonzero voxels are GRAINS (a Threshold of the "
                             "granule channel, or Mask Math's inversion of the PIV pore ROI). "
                             "Its t = 0 plane per position is used. Only read when Grains is "
                             "'mask'; empty is then refused."),
        InInt("grain_channel", "Grain channel", unit="", field=False, default=0,
              pick_kind="channel", available_in={"grains": frozenset({"image"})},
              description="Which image channel (0-based) shows the grains as BRIGHT objects "
                          "— they are found by Otsu threshold + watershed on a 4× "
                          "downsampled plane, as the deployed pipeline did on brightfield. "
                          "Only read when Grains is 'image'."),
        InFloat("grain_min_diameter", "Min grain size", unit="um", field=False, default=0.0,
                available_in={"grains": frozenset({"image", "mask"})},
                description="The smallest object drawn as a grain, as a diameter: it sets the "
                            "watershed's minimum area and how close two grain centres may "
                            "be before they count as one. 0 = auto (three grid cells). "
                            "LOWER keeps small fragments and splits big grains into pieces; "
                            "HIGHER merges touching grains and drops fragments. Only read "
                            "when Grains is 'image' or 'mask'. Display only."),
        InFloat("smooth", "Smoothing", unit="um", field=False, default=0.0,
                description="Gaussian smoothing of the velocity planes before the divergence "
                            "is taken and holes are inpainted. 0 = auto (1.5 grid cells, the "
                            "deployed pipeline's value). LARGER suppresses vector noise in "
                            "`w` and straightens streamlines but erases real gradients "
                            "narrower than it; it never changes the exported velocities' "
                            "units, only their smoothness."),
        InFloat("lambda_z", "λ vertical", unit="", field=False, default=0.2,
                available_in={"w_mode": frozenset({"regularized"})},
                description="Weight of the vertical-smoothness penalty in the regularised "
                            "`w` solve. LARGER makes `w` vary more gently between planes "
                            "(and under-fit a real jump); 0 trusts continuity alone. Only "
                            "read when W mode is 'regularized'; 0.2 is the deployed value."),
        InFloat("lambda_xy", "λ in-plane", unit="", field=False, default=0.1,
                available_in={"w_mode": frozenset({"regularized"})},
                description="Weight of the in-plane-smoothness penalty on `w`. LARGER "
                            "spreads `w` sideways into broader patches; 0 lets it follow the "
                            "divergence cell by cell. Only read when W mode is "
                            "'regularized'; 0.1 is the deployed value."),
        InInt("z_upsample", "Z upsample", unit="", field=False, default=8,
              description="Interpolated planes inserted between each pair of measured planes "
                          "(PCHIP) before streamlines, iso-surface and vessels are computed. "
                          "MORE gives smoother geometry at roughly linear cost in memory and "
                          "time; 0 keeps the measured planes only. Inert on a single plane."),
        InFloat("iso_percentile", "Channel percentile", unit="", field=False, default=90.0,
                description="The speed percentile of the whole volume that the translucent "
                            "'channel' iso-surface encloses. LOWER wraps more of the pore "
                            "space, HIGHER keeps only the fastest cores. Display only."),
        InFloat("vessel_percentile", "Vessel percentile", unit="", field=False, default=70.0,
                description="The speed percentile above which the volume counts as vessel "
                            "lumen for the vasculature skeleton. LOWER grows a denser, "
                            "finer tree; HIGHER keeps the main trunks. A visual analogy, "
                            "not a model."),
        InInt("seed_stride", "Seed spacing", unit="", field=False, default=11,
              description="Streamline seed spacing in grid cells (one cell = one PIV window "
                          "pitch), at three depths. SMALLER seeds more lines, denser and "
                          "slower; the total is capped by Max lines."),
        InInt("max_lines", "Max lines", unit="", field=False, default=1400,
              description="Upper bound on streamlines kept, after slow seeds are skipped. "
                          "The page's size and frame rate scale with it; 1400 is the "
                          "deployed value (~4 MB)."),
        InFloat("seed_min_fraction", "Min seed speed", unit="", field=False, default=0.1,
                description="Seeds whose in-plane speed is below this FRACTION of the field's "
                            "95th-percentile speed are skipped, so lines start in moving "
                            "fluid rather than inside grains or dead zones. 0 seeds "
                            "everywhere; 0.1 is a quiet default. Display only."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("w_mode", ["regularized", "trapezoid", "off"], default="regularized",
             label="W mode",
             description="How the out-of-plane velocity `w` is estimated from the in-plane "
                         "field via mass continuity. It is a structural estimate either way: "
                         "all measured 2-D divergence is attributed to flow through the "
                         "planes, and a granular pack also compacts and dilates.",
             choice_docs={
                 "regularized": "Regularised 3-D least squares (vertical + in-plane "
                                "smoothness, confidence-weighted by speed) solved by "
                                "conjugate gradient — spatially coherent and noise-tolerant; "
                                "the deployed pipeline's choice. Costs seconds to a minute "
                                "on a large grid.",
                 "trapezoid": "Plain per-column trapezoidal integration of −div from the "
                              "first plane upward: transparent and instant, but every noisy "
                              "vector's divergence accumulates upward, so use it to "
                              "cross-check the regularised answer rather than to publish.",
                 "off": "No out-of-plane velocity: streamlines stay in their planes and "
                        "the page's |w|/|u| reads 0. For a single plane, or when you want "
                        "to show exactly what was measured."}),
        Mode("grains", ["image", "mask", "off"], default="image", label="Grains",
             description="Where the grain pack drawn around the flow comes from.",
             choice_docs={
                 "image": "Detect grains on the image channel `Grain channel` (bright "
                          "objects, Otsu + watershed on a 4× downsampled plane) — what the "
                          "deployed pipeline did on a brightfield mosaic; needs the image.",
                 "mask": "Take grains from the Voxel mask `Grain mask` (nonzero = grain) "
                         "— the honest choice when a Threshold / Mask Math already defined "
                         "them for the PIV ROI; needs that layer on this wire.",
                 "off": "Draw no grains: flow geometry only. The fastest export, and the "
                        "only option on a Dataset with neither an image nor a mask."}),
        Mode("existing", ["overwrite", "refuse"], default="overwrite", label="Existing",
             description="What to do when a file is already at the path.",
             choice_docs={
                 "overwrite": "Replace it (written to `.part` first, so an interrupted "
                              "export never leaves a truncated page under the real name).",
                 "refuse": "Raise instead of touching an existing file, so a page already "
                           "published is never silently replaced."}),
    ],
    granularity=Granularity.MULTI_VIEW, kernel_axes=frozenset({"m", "t", "z", "y", "x"}),
    description="Write a PIV / DIC point velocity field as the lab's self-contained 3-D WebGL "
                "flow viewer (one HTML file: streamlines, flow particles, channel "
                "iso-surface, grains, vessels, DTI-style bundles; positions laid out by stage, "
                "w from continuity at the calibrated plane spacing) and hand the Dataset "
                "through unchanged.")
