"""Write the field-mechanics playground: bead gels with a KNOWN deformation, plus a three-page
graph that runs them, and the chip-flow PIV, through the Field cards.

    .venv\\Scripts\\python.exe scripts/make_field_playground.py               # data + graph
    .venv\\Scripts\\python.exe scripts/make_field_playground.py --graph-only  # graph only

Outputs:
- **Data** go to ``sample_data/field_mechanics/``. That folder is git-ignored, so regenerate
  the data on each machine.
- **The graph** goes to ``graphs/field_mechanics_playground.nd2graph.json``, with absolute paths
  to that data.

The pages:

1. **2-D gel (synthetic).** A bead-gel movie, 512² px at 0.5 µm/px, 6 frames, deformed by
   three superposed features that grow linearly from frame 0 to frame 5:
   - a cell contracting by 12 % at its centre;
   - a vortex turning 10° counter-clockwise on screen at its centre;
   - a 1 % background stretch along x.

   It runs DIC → Field Gradient → Strain (Green–Lagrange) → Rotation & Curl → Volume Change →
   Stress (neo-Hookean, plane stress, E = 1 kPa) → four Plot Field figures.
2. **Chip flow PIV (real data).** The bead flow through the granular pack: ensemble PIV with
   the pore mask, as in ``piv_alia_chip_flow``. The velocity field gives the strain RATE,
   vorticity, divergence and the viscous stress of water.
3. **3-D gel (synthetic).** A bead-gel volume, 40 × 160² at 0.5 µm/px with a 1 µm z step,
   3 frames. A spherical cell contracts 10 % and twists 6° about the optical axis, over a 1.5 %
   x stretch. It runs DVC → the same chain → figures.

On every page the Viewer's playback strip picks the position, frame and plane each figure
shows, as it does for the image.

The beads are rendered at their exactly displaced positions; the image is not warped. So the
only error in what the correlations recover is their own. ``truth.json`` beside the data gives
the analytic values at the feature centres and in the far field, for the last frame.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "sample_data", "field_mechanics")
GRAPH = os.path.join(ROOT, "graphs", "field_mechanics_playground.nd2graph.json")
CHIP = "S:/alia chip flow data"
#: a Load card reads an ND2 in place ("direct", the default) but must ingest a TIFF
TIFF_ACCESS = {"access": "ingest"}

# ── 2-D gel ──────────────────────────────────────────────────────────────────────────────
Y2 = X2 = 512
PX2 = 0.5                       # µm per pixel
T2 = 6
CELL = (180.0, 170.0)           # (y, x) px
R_CELL, A_CELL = 48.0, 0.12     # Gaussian radius (px); strain at the centre (contraction)
VORTEX = (340.0, 350.0)
R_VORTEX, THETA_DEG = 60.0, 10.0
STRETCH_X = 0.01
FAR_2D = (440.0, 80.0)          # a point neither feature reaches (both weights < e^-11)

# ── 3-D gel ──────────────────────────────────────────────────────────────────────────────
Z3, Y3, X3 = 40, 160, 160
PX3, DZ3 = 0.5, 1.0             # µm per pixel, µm per plane
T3 = 3
C3 = (20.0, 40.0, 40.0)         # (z, y, x) µm: the volume's centre
R3, A3, TWIST_DEG = 12.0, 0.10, 6.0
STRETCH_X3 = 0.015
FAR_3D = (20.0, 72.0, 8.0)      # µm


def disp_2d(y, x, s):
    """Displacement (dy, dx) in px at reference positions (y, x) px, at ramp ``s`` ∈ [0, 1]."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ry, rx = y - CELL[0], x - CELL[1]
    g = np.exp(-(ry ** 2 + rx ** 2) / (2 * R_CELL ** 2))
    dy = -A_CELL * s * ry * g
    dx = -A_CELL * s * rx * g
    ry, rx = y - VORTEX[0], x - VORTEX[1]
    th = np.deg2rad(THETA_DEG) * s * np.exp(-(ry ** 2 + rx ** 2) / (2 * R_VORTEX ** 2))
    c, sn = np.cos(th), np.sin(th)
    # counter-clockwise ON SCREEN (y runs down): a point right of the centre moves up
    dy = dy + ry * (c - 1) - rx * sn
    dx = dx + rx * (c - 1) + ry * sn
    dx = dx + STRETCH_X * s * (x - X2 / 2)
    return dy, dx


def disp_3d(z, y, x, s):
    """Displacement (dz, dy, dx) in µm at reference positions in µm, at ramp ``s``."""
    rz, ry, rx = z - C3[0], y - C3[1], x - C3[2]
    g = np.exp(-(rz ** 2 + ry ** 2 + rx ** 2) / (2 * R3 ** 2))
    k = 1.0 - A3 * s * g                        # contract first...
    th = np.deg2rad(TWIST_DEG) * s * g          # ...then twist about the optical (z) axis,
    c, sn = np.cos(th), np.sin(th)              # so the centre's rotation is exactly TWIST
    dz = (k - 1.0) * rz
    dy = k * (ry * c - rx * sn) - ry
    dx = k * (rx * c + ry * sn) - rx
    dx = dx + STRETCH_X3 * s * (x - X3 * PX3 / 2)
    return dz, dy, dx


def _render(shape, centres, amp, sigma, rng):
    """Gaussian beads at sub-pixel ``centres`` (N, D), summed onto ``shape``; background
    100, read noise σ 12, clipped to uint16."""
    img = np.zeros(shape, np.float64)
    d = len(shape)
    r = [int(np.ceil(3 * s)) for s in sigma]
    base = np.floor(centres).astype(int)
    grids = np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing="ij")
    for off in zip(*[g.ravel() for g in grids]):
        idx = base + np.asarray(off)
        w = amp.copy()
        for a in range(d):
            w *= np.exp(-((idx[:, a] - centres[:, a]) ** 2) / (2 * sigma[a] ** 2))
        ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
        np.add.at(img, tuple(idx[ok].T), w[ok])
    img += 100.0 + rng.normal(0.0, 12.0, shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def make_2d(path, rng):
    m = 24
    n = int((Y2 + 2 * m) * (X2 + 2 * m) / 40)                   # one bead per ~40 px²
    ref = np.column_stack([rng.uniform(-m, Y2 + m, n), rng.uniform(-m, X2 + m, n)])
    amp = rng.uniform(500, 1500, n)
    frames = []
    for t in range(T2):
        dy, dx = disp_2d(ref[:, 0], ref[:, 1], t / (T2 - 1))
        frames.append(_render((Y2, X2), ref + np.column_stack([dy, dx]), amp, (1.3, 1.3),
                              rng))
    _write_tiff(path, np.stack(frames), "TYX", PX2)


def make_3d(path, rng):
    m = 6.0
    ext = np.array([Z3 * DZ3, Y3 * PX3, X3 * PX3])               # µm
    n = int(np.prod(ext + 2 * m) / (DZ3 * PX3 * PX3) / 120)      # one bead per ~120 voxels
    ref = np.column_stack([rng.uniform(-m, e + m, n) for e in ext])
    amp = rng.uniform(500, 1500, n)
    step = np.array([DZ3, PX3, PX3])
    frames = []
    for t in range(T3):
        u = np.column_stack(disp_3d(ref[:, 0], ref[:, 1], ref[:, 2], t / (T3 - 1)))
        frames.append(_render((Z3, Y3, X3), (ref + u) / step, amp, (1.0, 1.2, 1.2), rng))
    _write_tiff(path, np.stack(frames), "TZYX", PX3, DZ3)


def _write_tiff(path, arr, axes, px, dz=None):
    import tifffile
    md = {"axes": axes, "unit": "um"}
    if dz is not None:
        md["spacing"] = dz
    tifffile.imwrite(path, arr, imagej=True, resolution=(1.0 / px, 1.0 / px),
                     resolutionunit=1, metadata=md)


def _grad(fn, p, h=1e-3):
    """∂u_i/∂X_j at ``p`` by central differences, in ``fn``'s own units."""
    p = np.asarray(p, float)
    d = len(p)
    G = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        G[:, j] = (np.array(fn(*(p + e))) - np.array(fn(*(p - e)))) / (2 * h)
    return G


def _summary(G):
    d = G.shape[0]
    F = np.eye(d) + G
    E = 0.5 * (F.T @ F - np.eye(d))
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    names = "zyx"[-d:]
    out = {"jacobian": round(float(np.linalg.det(F)), 4),
           "dilatation": round(float(np.linalg.det(F) - 1), 4)}
    for i, a in enumerate(names):
        out[f"strain_{a}{a} (green_lagrange)"] = round(float(E[i, i]), 4)
        out[f"strain_{a}{a} (infinitesimal)"] = round(float(G[i, i]), 4)
    iy, ix = d - 2, d - 1
    out["curl_view" if d == 3 else "curl"] = round(float(G[ix, iy] - G[iy, ix]), 4)
    out["rotation_deg (in plane, + = counter-clockwise on screen)"] = round(
        float(np.degrees(np.arctan2(-R[iy, ix], R[iy, iy]))), 3)
    return out


def truth():
    f2 = lambda y, x: disp_2d(y, x, 1.0)
    f3 = lambda z, y, x: disp_3d(z, y, x, 1.0)
    return {
        "2-D gel, last frame (5), reference = frame 0": {
            "cell centre (y 180, x 170 px)": _summary(_grad(f2, CELL)),
            "vortex centre (y 340, x 350 px)": _summary(_grad(f2, VORTEX)),
            "far field (y 440, x 80 px)": _summary(_grad(f2, FAR_2D)),
        },
        "3-D gel, last frame (2), reference = frame 0": {
            "cell centre (z 20, y 80, x 80 px)": _summary(_grad(f3, C3)),
            "far field (z 20, y 144, x 16 px)": _summary(_grad(f3, FAR_3D)),
        },
        "note": ("Gradients come from the analytic field at a point. A correlation averages "
                 "over its subset (and Field Gradient over its stencil), so recovered peaks "
                 "are somewhat lower at the feature centres; the far field should match "
                 "closely."),
    }


# ── the graph ────────────────────────────────────────────────────────────────────────────
def build_graph(out_path, path_2d, path_3d):
    sys.path.insert(0, ROOT)
    from nodegraph import nodes as _catalog          # noqa: F401 — registers the catalog
    from nodelab_v2.document import GraphDocument
    from nodelab_v2.ops import ensure_ops
    from nodelab_v2.workspace import FREE, Workspace
    ensure_ops()
    ws = Workspace()

    def wire(doc, pairs):
        for s, d in pairs:
            ss, ds_ = "out", "data"
            if isinstance(s, tuple):
                s, ss = s
            if isinstance(d, tuple):
                d, ds_ = d
            doc.connect(s, ss, d, ds_)

    def mechanics(doc, upstream, x0, quantity, measure, stress_params, stress_modes):
        """Field Gradient → Strain → Rotation → Volume Change → Stress, from ``upstream``."""
        ids = ["grad", "strain", "rot", "vol", "stress"]
        doc.add_node("field.gradient", node_id="grad", x=x0, y=0,
                     modes={"quantity": quantity})
        doc.add_node("field.strain", node_id="strain", x=x0 + 280, y=0,
                     modes={"measure": measure})
        doc.add_node("field.rotation", node_id="rot", x=x0 + 560, y=0)
        doc.add_node("field.volume_change", node_id="vol", x=x0 + 840, y=0)
        doc.add_node("field.stress", node_id="stress", x=x0 + 1120, y=0,
                     params=stress_params, modes=stress_modes)
        wire(doc, [(upstream, "grad"), ("grad", "strain"), ("strain", "rot"), ("rot", "vol"),
                   ("vol", "stress")])
        return ids

    def plots(doc, x0, specs, **common):
        ids = []
        for i, (nid, value, title, extra) in enumerate(specs):
            params = dict(common, value=value, title=title, **extra)
            doc.add_node("plot.field", node_id=nid, x=x0, y=-330 + 220 * i, params=params)
            wire(doc, [("stress", nid)])
            ids.append(nid)
        return ids

    # 1 ── 2-D gel (synthetic) ───────────────────────────────────────────────────────────
    d = GraphDocument()
    ws.add_page("2-D gel (synthetic)", FREE, doc=d, page_id="pg1")
    d.add_node("io.load", node_id="gel2d", x=-620, y=0, params={"path": path_2d},
               modes=TIFF_ACCESS)
    d.add_node("analysis.dic_correlate", node_id="dic", x=-300, y=0,
               params={"name": "dic"}, modes={"reference_mode": "fixed_frame"})
    wire(d, [(("gel2d", "image"), "dic")])
    chain = mechanics(d, "dic", 20, "displacement", "green_lagrange",
                      {"youngs_modulus": 1000.0, "poisson_ratio": 0.45},
                      {"model": "neo_hookean", "plane": "stress"})
    figs = plots(d, 1520, [
        ("fig_strain", "strain_eq", "Equivalent strain", {}),
        ("fig_rot", "rotation_deg", "Rotation (deg, + = CCW)", {}),
        ("fig_J", "jacobian", "Area change J", {"contours": 6}),
        ("fig_vm", "von_mises", "von Mises stress (Pa)", {}),
    ], arrow_every=2, scale_bar_um=50.0)
    d.add_frame("Synthetic bead gel: a cell contracts 12 % (upper left), a vortex turns "
                "10° CCW (lower right), 1 % stretch in x; frames 0→5 ramp it in",
                ["gel2d", "dic"])
    d.add_frame("Mechanics chain: each card adds columns to the DIC table", chain)
    d.add_frame("Figures: the playback strip picks the frame (5 = full deformation)", figs)

    # 2 ── chip flow PIV (real data) ─────────────────────────────────────────────────────
    d = GraphDocument()
    ws.add_page("Chip flow PIV (real data)", FREE, doc=d, page_id="pg2")
    srcs = []
    for k in range(10):
        nid = f"src{k + 1}"
        d.add_node("io.load", node_id=nid, x=-1500, y=-560 + 110 * k, params={
            "path": f"{CHIP}/Count{k:05d}_ChannelFITC_Seq{k + 1:04d}.nd2"})
        srcs.append(nid)
    d.add_node("util.timeseries", node_id="series", x=-1180, y=-60,
               modes={"chain_axis": "T", "chain_order": "time"})
    d.add_node("io.load", node_id="rb", x=-1500, y=620,
               params={"path": f"{CHIP}/ChannelR-B_Seq0000.nd2"})
    d.add_node("analysis.threshold", node_id="granules", x=-1180, y=620,
               params={"name": "granules"}, modes={"method": "otsu", "scope": "plane"})
    d.add_node("math.mask", node_id="pore", x=-880, y=0,
               params={"a": "", "b": "granules", "name": "pore"}, modes={"op": "subtract"})
    d.add_node("analysis.piv", node_id="piv", x=-580, y=0, params={
        "channel": 0, "frame_interval": 0.2014, "name": "piv", "overlap": 0.5, "passes": 2,
        "replace": True, "roi": "pore", "s2n_threshold": 1.3, "velocity": True,
        "window_size": 32}, modes={"correlation": "circular", "reference_mode": "ensemble"})
    wire(d, [(("src1", "image"), "series")]
         + [((s, "image"), ("series", f"file_{i + 2}")) for i, s in enumerate(srcs[1:])]
         + [(("rb", "image"), "granules"), ("series", "pore"), ("granules", ("pore", "other")),
            ("pore", "piv")])
    chain = mechanics(d, "piv", -260, "velocity", "infinitesimal", {"viscosity": 1.0e-3},
                      {"model": "newtonian_fluid"})
    figs = plots(d, 1240, [
        ("fig_speed", "speed", "Bead speed (um/s)", {}),
        ("fig_curl", "curl", "Vorticity (1/s, + = CCW)", {}),
        ("fig_rate", "strain_eq", "Equivalent strain rate (1/s)", {}),
        ("fig_div", "divergence", "In-plane divergence (1/s)", {"arrows": False}),
    ], arrow_every=2, scale_bar_um=100.0)
    d.add_frame("Bead flow through the granular pack: 10 burst files chained on T, PIV "
                "only inside the pores (ensemble, velocity)",
                srcs + ["series", "rb", "granules", "pore", "piv"])
    d.add_frame("Velocity → strain RATE, vorticity, divergence, viscous stress (water)", chain)
    d.add_frame("Figures: the playback strip picks position and plane (position 24, "
                "plane 3 has the fastest flow)", figs)

    # 3 ── 3-D gel (synthetic) ───────────────────────────────────────────────────────────
    d = GraphDocument()
    ws.add_page("3-D gel (synthetic)", FREE, doc=d, page_id="pg3")
    d.add_node("io.load", node_id="gel3d", x=-620, y=0, params={"path": path_3d},
               modes=TIFF_ACCESS)
    d.add_node("analysis.dvc_field", node_id="dvc", x=-300, y=0, params={
        "name": "dvc", "subset_size": 24, "subset_size_z": 12, "subset_spacing": 10,
        "subset_spacing_z": 5}, modes={"reference_mode": "fixed_frame"})
    wire(d, [(("gel3d", "image"), "dvc")])
    chain = mechanics(d, "dvc", 20, "displacement", "green_lagrange",
                      {"youngs_modulus": 1000.0, "poisson_ratio": 0.45},
                      {"model": "neo_hookean"})
    figs = plots(d, 1520, [
        ("fig_J", "jacobian", "Volume change J", {"contours": 6}),
        ("fig_rot", "rotation_deg", "Rotation (deg)", {}),
        ("fig_p", "pressure", "Pressure (Pa, + = compression)", {}),
        ("fig_vm", "von_mises", "von Mises stress (Pa)", {}),
    ], scale_bar_um=20.0)
    d.add_frame("Synthetic bead volume: a cell at the centre contracts 10 % and twists 6° "
                "about z, over a 1.5 % stretch in x; frames 0→2",
                ["gel3d", "dvc"])
    d.add_frame("Mechanics chain on the 3-D DVC field", chain)
    d.add_frame("Figures: the playback strip picks frame and plane (frame 2, plane 20 = "
                "the cell's centre)", figs)

    ws.active = "pg1"
    ws.save_file(out_path)
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph-only", action="store_true",
                    help="rewrite the graph, keep the data already written")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    p2 = os.path.join(DATA, "gel_2d_cell_vortex.tif").replace("\\", "/")
    p3 = os.path.join(DATA, "gel_3d_cell_twist.tif").replace("\\", "/")
    if not a.graph_only:
        rng = np.random.default_rng(20261009)
        make_2d(p2, rng)
        make_3d(p3, rng)
        with open(os.path.join(DATA, "truth.json"), "w", encoding="utf-8") as f:
            json.dump(truth(), f, indent=2, ensure_ascii=False)
        print("data:", p2, p3, sep="\n  ")
    build_graph(GRAPH, p2, p3)
    print("graph:", GRAPH)


if __name__ == "__main__":
    main()
