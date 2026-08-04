"""Tessellate (``analysis.tessellate``) — Tessellate a labeled Point cloud (convex/concave/Voronoi boundary) or a Voxel label raster (marching cubes) → a MESH: one closed surface per label, with analytic volume/area/density per element;."""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, Tuple

from nodegraph.dataset import AxisSizes, Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import (
    Granularity,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)

from nodegraph.catalog._base import register_node

# ── Tessellate points / labels → a MESH (3D) ───────────────────────────────────
#
# The tessellation stage on its own (V2.08; ports granule_tessellate). v1 always had this
# as its own node — `special:granule_tessellate` → `special:granule_mask` — and v2 fused
# the two because there was no domain able to carry a mesh across a wire. With
# ``Domain.MESH`` (nodegraph/mesh.py) the v1 split is restored: this node produces the
# surface, ``transform.rasterize_mesh`` consumes it.
#
# GENERAL node: it tessellates ANY labeled 3-D point cloud, or the surface of ANY Voxel
# label raster — the granule density-MERGE stays off (``merge_tol=0``) so input labels map
# one-to-one. Downstream: rasterize_mesh → boundary_band / measure.


def _tess_points(ctx: EvalContext, ds: Dataset, ax: AxisSizes, mode: str,
                 vox: Tuple[float, float, float]) -> list:
    """POINTS path — one mesh element per label of a Point layer's cluster-id column.

    The vendored kernel takes VOXEL ``(z,y,x)`` points and returns world ``(x,y,z)`` µm
    vertices, so the only conversion here is on the way out."""
    from nodegraph.kernels.granule_tessellate import tessellate_granules
    from nodegraph.kernels.mesh_raster import verts_um_to_zyx
    from nodegraph.mesh import MeshElement, surface_area_um2

    source = ctx.layer("points")
    labels_col = ctx.params.get("cluster", "cluster")
    if mode == "voronoi":
        tp: Dict[str, Any] = {"tess_mode": "voronoi"}
    elif mode == "alpha_shape":
        a_um = float(ctx.params.get("alpha_um", 0.0))
        tp = {"tess_mode": "alpha_shape", "alpha": (a_um if a_um > 0 else float("inf"))}
    else:                                                      # convex_hull (default)
        tp = {"tess_mode": "alpha_shape", "alpha": float("inf")}
    # MERGE OFF — general node: one element per input label, no density folding.
    tp["merge_tol"] = 0.0
    tp["min_granule_points"] = max(1, int(ctx.params.get("min_points", 4)))

    pts = [a for a in ds.layers_on(Domain.POINT) if a.layer == source]
    if not pts:
        raise ValueError(f"tessellate needs a Point layer {source!r} "
                         "(run detect.particles → cluster_points first)")
    col = {a.name: np.asarray(a.values) for a in pts}
    for req in ("m", "t", "c", "z", "y", "x"):
        if req not in col:
            raise ValueError(f"Point layer {source!r} missing coordinate column {req!r}")
    if labels_col not in col:
        raise ValueError(f"Point layer {source!r} missing label column {labels_col!r} "
                         "(run cluster_points, or provide a per-point label column)")
    m_all = col["m"].astype(int); t_all = col["t"].astype(int); c_all = col["c"].astype(int)
    z_all = col["z"].astype(float); y_all = col["y"].astype(float); x_all = col["x"].astype(float)
    lab_all = col[labels_col].astype(int)

    elements: list = []
    for m in sorted(set(m_all.tolist())):
        for t in sorted(set(t_all[m_all == m].tolist())):
            for c in sorted(set(c_all[(m_all == m) & (t_all == t)].tolist())):
                sel = (m_all == m) & (t_all == t) & (c_all == c)
                if not sel.any():
                    continue
                pzyx = np.column_stack([z_all[sel], y_all[sel], x_all[sel]])
                plab = lab_all[sel]
                tess = tessellate_granules(pzyx, plab, vox, tp)
                for oid, bnd in tess.boundaries.items():
                    verts = verts_um_to_zyx(bnd.vertices_um, vox)
                    mem = plab == oid
                    elements.append(MeshElement(
                        m=m, t=t, c=c, src_label=int(oid),
                        verts_zyx=verts, faces=bnd.faces,
                        centroid_zyx=(float(np.mean(z_all[sel][mem])),
                                      float(np.mean(y_all[sel][mem])),
                                      float(np.mean(x_all[sel][mem]))),
                        # the kernel's ANALYTIC volume (alpha-filtered where relevant) —
                        # no voxel-quantization loss, unlike a rasterized count
                        volume_um3=float(bnd.enclosed_volume_um3),
                        surface_area_um2=surface_area_um2(verts, bnd.faces, vox),
                        density=float(bnd.density), n_points=int(bnd.n_points)))
    return elements
def _tess_label_surface(ctx: EvalContext, ds: Dataset, ax: AxisSizes,
                        vox: Tuple[float, float, float]) -> list:
    """LABELS path — one mesh element per region of a Voxel label raster, via marching
    cubes. ``marching_cubes`` already returns voxel ``(z,y,x)`` vertices, so this path
    needs no coordinate conversion at all."""
    from skimage.measure import marching_cubes
    from nodegraph.mesh import MeshElement, enclosed_volume_um3, surface_area_um2

    src = ctx.layer("labels")
    level = float(ctx.params.get("iso_level", 0.5))
    step = max(1, int(ctx.params.get("decimate", 1)))
    lay = ds.get(Domain.VOXEL, src)
    if lay is None:
        raise ValueError(f"tessellate (label_surface) needs a Voxel layer {src!r} "
                         "(run analysis.label / analysis.threshold first)")
    raster = np.asarray(lay.values)
    elements: list = []
    for m in range(ax.m):
        for t in range(ax.t):
            for c in range(ax.c):
                vol = raster[m, t, :, c]
                ids = np.unique(vol)
                for g in ids[ids != 0].tolist():
                    binary = (vol == g)
                    if binary.sum() < 8:              # too thin for a closed surface
                        continue
                    try:
                        v, f, _n, _v = marching_cubes(
                            binary.astype(float), level=level, step_size=step)
                    except (RuntimeError, ValueError):
                        continue                      # degenerate region — skip, not fail
                    zc, yc, xc = (float(a.mean()) for a in np.nonzero(binary))
                    elements.append(MeshElement(
                        m=m, t=t, c=c, src_label=int(g), verts_zyx=v, faces=f,
                        centroid_zyx=(zc, yc, xc),
                        volume_um3=enclosed_volume_um3(v, f, vox),
                        surface_area_um2=surface_area_um2(v, f, vox),
                        density=0.0, n_points=int(binary.sum())))
    return elements
def _compute_tessellate(ctx: EvalContext) -> Dataset:
    """Tessellate a labeled Point cloud **or** a Voxel label raster into a **MESH** —
    one closed boundary surface per input label (ports v1 ``granule_tessellate``).

    Resolved spec (V2.08): category analysis; op ``analysis.tessellate``; adds MESH;
    **3D-only** (a surface is volumetric — ``WHOLE_VOLUME``, ``kernel_axes={z,y,x}``,
    ``z<2`` is a hard error). One four-choice ``boundary`` lever selects both the source
    kind and the algorithm: ``convex_hull`` (alpha=∞) / ``alpha_shape`` (finite
    ``alpha_um``, concave) / ``voronoi`` read the ``points`` layer's ``cluster`` column;
    ``label_surface`` marching-cubes the ``labels`` Voxel raster. It is ONE mode rather
    than a ``source`` × ``boundary`` pair because ``available_in`` gates sockets on modes
    but nothing gates a *mode* on another mode — a separate source lever would leave three
    selectable-but-meaningless combinations.

    ``reads_domains`` is empty on purpose (the ``track.objects`` / ``boundary_band``
    precedent): the required domain is POINT in three modes and VOXEL in the fourth, and
    ``reads_domains`` has no per-mode form, so declaring either would light a false red
    chip in the other. The compute raises with the exact layer to wire instead.

    Vertices are stored in **voxel ``(z,y,x)``** (the Mesh domain convention — see
    :mod:`nodegraph.mesh`); ``voxel_size_um=(z_step_um, pixel_size_um, pixel_size_um)`` is
    read via ``ctx.calib`` (memo-fenced) for the µm geometry columns. The boundary mode +
    source are **stamped as mesh provenance** so ``transform.rasterize_mesh`` derives its
    interior test instead of exposing a lever that could disagree with the data
    (`wire-node-v2` §7b)."""
    from nodegraph.mesh import build_mesh_tables, with_mesh

    ds = ctx.inputs[0]
    ax = ds.axes
    if ax.z < 2:
        raise ValueError("tessellate needs a 3D volume (z>1); a boundary surface is "
                         "volumetric (use a Z-stack)")
    mode = ctx.params.get("__modes__", {}).get("boundary", "convex_hull")
    name = ctx.layer("name")
    px = ctx.calib("pixel_size_um") or 0.1
    zs = ctx.calib("z_step_um") or 0.5
    vox = (zs, px, px)

    if mode == "label_surface":
        elements = _tess_label_surface(ctx, ds, ax, vox)
        src_kind, src_layer = "labels", ctx.layer("labels")
    else:
        elements = _tess_points(ctx, ds, ax, mode, vox)
        src_kind, src_layer = "points", ctx.layer("points")

    tables = build_mesh_tables(elements, layer=name, z_kind="subpixel")
    return with_mesh(ds, tables, provenance={
        "boundary": mode, "source": src_kind, "src_layer": str(src_layer),
        "voxel_size_um": [float(v) for v in vox]})
register_node(
    _compute_tessellate, op_key="analysis.tessellate", label="Tessellate",
    category="analysis",
    # deliberately empty — see the compute docstring (POINT in three modes, VOXEL in the
    # fourth, and reads_domains has no per-mode form)
    reads_domains=frozenset(),
    adds_domains=frozenset({Domain.MESH}),
    inputs=[InDataset(),
            InString("points", "Point layer", field=False, default="particles",
                     layer_in=Domain.POINT,
                     available_in={"boundary": frozenset(
                         {"convex_hull", "alpha_shape", "voronoi"})},
                     description=
                     "Which Point cloud to build surfaces from — a Spot or Particle Detection "
                     "output, usually after Cluster Points. Hidden under `label_surface`, which "
                     "meshes a voxel raster instead of points."),
            InString("cluster", "Label column", field=False, default="cluster",
                     available_in={"boundary": frozenset(
                         {"convex_hull", "alpha_shape", "voronoi"})},
                     description=
                     "Which per-point column groups the cloud into separate elements — one mesh "
                     "element per distinct value, normally the column written by Cluster Points. "
                     "Without meaningful groups the whole cloud becomes ONE surface, which is "
                     "rarely what you want. Hidden under `label_surface`."),
            # alpha is the ALPHA-SHAPE radius only: convex_hull pins alpha=inf, voronoi
            # takes no radius, label_surface does not use the kernel at all.
            InFloat("alpha_um", "Alpha (concave)", unit="um", field=True, default=0.0,
                    pick_kind="radius",
                    available_in={"boundary": frozenset({"alpha_shape"})},
                    description=
                    "Probe radius in microns — how deeply the surface may cave inward. This is "
                    "the knob that makes alpha-shape different from a convex hull: SMALL follows "
                    "concavities tightly and, taken too far, fragments one cluster into pieces "
                    "or punches holes through it; LARGE smooths them out until the result IS the "
                    "convex hull. It therefore changes enclosed VOLUME directly. 0 means auto. "
                    "Alpha-shape only — `convex_hull` fixes the radius at infinity."),
            InInt("min_points", "Min points/element", unit="", field=False, default=4,
                  available_in={"boundary": frozenset(
                      {"convex_hull", "alpha_shape", "voronoi"})},
                  description=
                  "Smallest group that still gets a mesh; smaller groups are skipped. 4 is the "
                  "floor for a closed 3D surface — three points define only a triangle, with no "
                  "volume — so lowering it cannot help and raising it discards sparse clusters "
                  "that would otherwise produce slivers with near-zero volume. Hidden under "
                  "`label_surface`."),
            InString("labels", "Voxel label layer", field=False, default="labels",
                     layer_in=Domain.VOXEL,
                     available_in={"boundary": frozenset({"label_surface"})},
                     description=
                     "Which label raster to mesh — one closed surface per region, so it converts "
                     "a voxel segmentation into geometry. This is the socket the "
                     "`label_surface` boundary reads; the point-cloud boundaries use the Point "
                     "layer instead."),
            InFloat("iso_level", "Iso level", unit="", field=True, default=0.5,
                    available_in={"boundary": frozenset({"label_surface"})},
                    description=
                    "Where the surface is placed across the label boundary, on a 0-to-1 scale "
                    "between background and region. 0.5 puts it midway — the standard choice, "
                    "and the one that reproduces the voxel volume most faithfully. LOWER pushes "
                    "the surface outward and inflates the enclosed volume; HIGHER pulls it in "
                    "and shrinks it, so this directly biases every volume you measure from the "
                    "mesh. `label_surface` only."),
            InInt("decimate", "Vertex step", unit="", field=False, default=1,
                  available_in={"boundary": frozenset({"label_surface"})},
                  description=
                  "Keep only every Nth vertex — the mesh-size/fidelity trade-off. 1 keeps the "
                  "full surface. HIGHER shrinks the mesh sharply (memory and downstream cost "
                  "fall roughly with the square) at the price of a coarser, faceted surface, "
                  "and because decimation cuts corners it systematically REDUCES enclosed "
                  "volume — so raise it for display, keep it at 1 when the mesh is being "
                  "measured. `label_surface` only."),
            InString("name", "Output mesh", field=False, default="mesh",
                     layer_out=(Domain.MESH,),
                     description=
                     "Name of the Mesh this node writes (vertices, faces and per-element rows). "
                     "Rasterize Mesh and any mesh consumer select it by this name.")],
    outputs=[OutDataset()],
    modes=[Mode("boundary", ["convex_hull", "alpha_shape", "voronoi", "label_surface"],
                default="convex_hull", label="Boundary",
                description=
                "What kind of surface is wrapped around each label — and, because the four "
                "algorithms do not take the same input, which SOURCE is read: the first three "
                "wrap a clustered Point cloud, the last one traces an existing Voxel label "
                "raster. One dropdown rather than a source × algorithm pair, since three of "
                "those combinations would be meaningless.",
                choice_docs={
                    "convex_hull":
                        "The tightest CONVEX wrapper around each cluster's points — an alpha "
                        "shape with infinite alpha. Always closed, always watertight, and "
                        "impossible to get wrong, which is why it is the default; it also "
                        "bridges every concavity, so a cupped or branched cluster is reported "
                        "larger than it is.",
                    "alpha_shape":
                        "A CONCAVE wrapper: the hull is carved back wherever a sphere of radius "
                        "Alpha fits between points, so dents and channels survive. The accurate "
                        "choice for a non-convex cluster, and the fragile one — too small an "
                        "alpha fragments the surface or opens holes, too large returns the "
                        "convex hull.",
                    "voronoi":
                        "Bound each cluster by the Voronoi partition of the point set: "
                        "neighbouring clusters share flat interfaces and their volumes TILE the "
                        "space with no gap between them. Use it when the clusters are meant to "
                        "fill a region (packed granules, a tissue) rather than sit in it.",
                    "label_surface":
                        "Marching cubes over an existing Voxel label raster — no point cloud at "
                        "all. The most faithful option, because the surface follows the "
                        "segmentation you already checked instead of being inferred from dots, "
                        "and the one to use whenever a 3-D segmentation exists. Its vertex "
                        "count scales with surface area, so meshes get large.",
                })],
    granularity=Granularity.WHOLE_VOLUME, kernel_axes=frozenset({"z", "y", "x"}),
    description="Tessellate a labeled Point cloud (convex/concave/Voronoi boundary) or a "
                "Voxel label raster (marching cubes) → a MESH: one closed surface per "
                "label, with analytic volume/area/density per element; 3D (ports v1 "
                "granule_tessellate). Feed transform.rasterize_mesh to get a Label volume.")
