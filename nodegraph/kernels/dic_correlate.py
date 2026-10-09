"""dic_correlate — vendored 2D AL-DIC (pyALDIC) kernel + result adapter.

PURPOSE
    Self-contained, portable extraction of the ND2Studios "DIC (pyALDIC)" node
    math kernel, for wiring into a different software's node system. Given an
    already-prepared reference/deformed image pair (or an ordered series), it
    runs the external ``al-dic`` (pyALDIC) 2D Augmented-Lagrangian DIC solver and
    resamples the result onto a regular grid packaged as a ``DVCResult``.

WHERE THE REAL MATH LIVES
    The actual 2D correlation math — local IC-GN subset matching + a global ADMM
    solve over an adaptive quadtree finite-element mesh — is 100% inside the
    EXTERNAL ``al-dic`` (pyALDIC) package (``al_dic.core.pipeline.run_aldic``),
    which is imported LAZILY. The IN-REPO math vendored here is ONLY the thin
    adapters around it:
      * input:  normalize an image to float64 [0,1] (``_to_float01``); map node
                params -> an al_dic ``DICPara`` with pow2/even snapping
                (``_build_dicpara`` / ``_snap_pow2``); build a boolean ROI mask
                from a serializable shape list (``build_roi_mask``).
      * output: de-interleave pyALDIC's ``U=[u,v,...]`` and scipy.griddata-
                resample the scattered adaptive-FE-mesh result onto a regular
                grid, with the LOAD-BEARING [x,y]->[y,x] / [u,v]->[dy,dx] axis
                swap (``_frame_to_dvcresult`` / ``_interp_to_grid`` /
                ``_regular_grid`` / ``_mesh_coords``).

PROVENANCE (branch: Version-1.45)
    * nd2studios/backend/dic/engine.py   — run_pyaldic_pair, run_pyaldic_series,
      _frame_to_dvcresult, _interp_to_grid, _regular_grid, _mesh_coords,
      _build_dicpara, _to_float01, _snap_pow2, _refinement_policy,
      _require_al_dic, al_dic_available, _INSTALL_HINT (verbatim).
    * nd2studios/backend/dic/roi.py      — build_roi_mask, _rasterize_region,
      has_region, OP_ADD, OP_CUT (verbatim).
    * nd2studios/core/dvc_registry.py    — DVCResult dataclass ONLY (verbatim).

VENDORED VERBATIM; IMPORTS NOTHING FROM nd2studios; CALLER OWNS ALL PREP
    (no file I/O, no per-multipoint/per-timepoint looping, no crop / downsample /
    registration / exclusion — the caller prepares the (H,W) arrays and masks).

DROPPED UI/REGISTRY-ONLY MEMBERS (not on the compute path)
    * DVCMethod (ABC) + DVCParams + the ``@DVCMethod.register`` registry and the
      ``ParamSpec`` import from nd2studios.core.dvc_registry — discovery/UI only.
    * No private-helper renames were needed (no name collisions between the three
      concatenated source modules).

VERBATIM IMPORT STRATEGY
    numpy is top-level. ``al_dic`` stays LAZY (imported inside _require_al_dic /
    _refinement_policy). scipy.interpolate.griddata stays imported inside
    _interp_to_grid, and skimage.draw stays imported inside _rasterize_region —
    exactly as the sources had them.

DEVIATIONS FROM VERBATIM (2026-07-31 — validated against pyALDIC's own synthetic suite)
    Measured with ``scripts/dic_synthetic_bench.py``, which replicates
    ``tests/test_integration/test_synthetic.py`` from github.com/zachtong/pyALDIC.
    The v1 adapter was ACCURATE (11/12 upstream cases inside upstream's own tolerance,
    matching or beating a direct ``run_aldic`` call) but wasteful. Four changes, none of
    which alter the solved field beyond float noise:

    1. ``_LazyFrameProvider`` — a series is now ONE ``run_aldic`` call over the whole
       ordered stack instead of N independent pair calls, so al_dic's reference bundle,
       subpb1 precompute and 6-DOF IC-GN context are computed once and reused, and the
       FFT search radius it learns carries across frames. **Measured 1.8x at both 256²/T=9
       and 1024²/T=5, for BIT-IDENTICAL fields** (max|diff| exactly 0.0 px). Frames are
       pulled and normalized on demand, so peak memory stays at ~two planes rather than
       the whole T×H×W stack.
    2. **Exact lattice resample.** al_dic's default uniform mesh is exactly the regular
       lattice ``_regular_grid`` reconstructs, so the old ``griddata`` linear+nearest
       pass was a provable no-op costing up to **~3.3 s/frame at 2048²**. It is now an
       index scatter, with griddata kept as the fallback for a genuinely irregular
       (quadtree-refined) mesh. ``resample="nodes"`` skips the step entirely and emits
       the FE nodes themselves — exact, and it PRESERVES adaptive-refinement nodes that
       resampling to the coarse lattice used to discard.
    3. **ROI-bbox grid tightening.** With a mask, the correlation grid is limited to the
       mask's bounding box instead of the full frame — which is what upstream's own
       scripted example does (``examples/batch_process.py:221``). Untightened, al_dic lays
       a grid over the whole frame and returns NaN for every node the mask excludes: on a
       disc ROI covering ~28% of a 512² frame that was **68% of the emitted points**.
    4. **Strain / convergence are no longer dropped.** ``strain_field`` is filled from
       al_dic's FEM ``StrainResult`` when ``compute_strain`` is on; ``converged`` and
       ``iterations`` are read from the result instead of echoed back from the params.

    ``_to_float01`` is retained for the documented input contract but is NO LONGER on
    the al_dic path: al_dic z-scores every frame itself (``normalize_one``:
    ``(img-mean_roi)/std_roi``), and a z-score is invariant under the affine min-max
    rescale, so the extra pass was provably redundant (measured identical to 2.8e-14 px).
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# DVCResult — vendored verbatim from nd2studios/core/dvc_registry.py
# (the DVCMethod ABC / DVCParams / ParamSpec import were dropped — UI/registry only)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DVCResult:
    """Structured output from a :meth:`DVCMethod.run` call.

    Shapes (``d`` = 2 for 2D DIC, 3 for 3D DVC):

    - ``grid_coords``: ``(*grid, d)`` subset-center coordinates **in voxels**,
      where ``grid`` is ``(Gy, Gx)`` (2D) or ``(Gz, Gy, Gx)`` (3D).
    - ``displacement_field``: ``(*grid, d)`` displacement **in voxels**, axis
      order matching ``grid_coords`` (i.e. ``[..., 0]`` is the slowest spatial
      axis: y in 2D, z in 3D).
    - ``strain_field``: ``(*grid, n_components)`` or ``None`` until computed.

    ``voxel_size_um`` is ``(y, x)`` (2D) or ``(z, y, x)`` (3D); use
    :meth:`displacement_um` to convert the displacement field to micrometers.
    """
    dim: int
    grid_coords: np.ndarray
    displacement_field: np.ndarray
    voxel_size_um: Tuple[float, ...] = ()

    strain_field: Optional[np.ndarray] = None
    strain_type: str = ""

    qfactor: Optional[np.ndarray] = None        # (*grid,) correlation confidence
    converged: bool = False
    iterations: int = 0
    mu: float = 0.0
    beta: float = 0.0

    method: str = ""
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── convenience accessors ──
    @property
    def magnitude(self) -> np.ndarray:
        """``(*grid,)`` displacement magnitude in voxels."""
        return np.sqrt(np.sum(np.square(self.displacement_field), axis=-1))

    def displacement_um(self) -> np.ndarray:
        """Displacement field converted to micrometers (per-axis scaling)."""
        if not self.voxel_size_um or len(self.voxel_size_um) != self.dim:
            return self.displacement_field
        scale = np.asarray(self.voxel_size_um, dtype=np.float64)
        return self.displacement_field * scale

    def magnitude_um(self) -> np.ndarray:
        return np.sqrt(np.sum(np.square(self.displacement_um()), axis=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Engine + adapters — vendored verbatim from nd2studios/backend/dic/engine.py
# ═══════════════════════════════════════════════════════════════════════════════
ProgressCb = Optional[Callable[[int], None]]
CancelledCb = Optional[Callable[[], bool]]

_INSTALL_HINT = (
    "The 2D DIC node needs the optional 'al-dic' package (pyALDIC).\n"
    "Install it with:  pip install al-dic\n"
    "(it pulls numba + PySide6>=6.6; it is intentionally not a core dependency)."
)


def al_dic_available() -> bool:
    """True if the optional ``al_dic`` package is importable."""
    return importlib.util.find_spec("al_dic") is not None


def _require_al_dic():
    """Import and return the ``al_dic`` entry points, or raise a friendly error."""
    if not al_dic_available():
        raise ImportError(_INSTALL_HINT)
    config = importlib.import_module("al_dic.core.config")
    pipeline = importlib.import_module("al_dic.core.pipeline")
    return config, pipeline


def _refinement_policy(refinement: Optional[Dict[str, Any]], half_win: int):
    """Build an ``al_dic`` ``RefinementPolicy`` from the refine node's spec, or None."""
    if not refinement:
        return None
    try:
        refine = importlib.import_module("al_dic.mesh.refinement")
    except Exception:  # noqa: BLE001 — refinement is best-effort
        return None
    crit = refinement.get("criteria", {}) if isinstance(refinement, dict) else {}
    brush = refinement.get("brush")
    mask = None
    if brush is not None:
        mask = np.asarray(brush).astype(np.float64)
    try:
        return refine.build_refinement_policy(
            refine_inner_boundary=bool(crit.get("mask_boundary", False)),
            refine_outer_boundary=bool(crit.get("roi_edge", False)),
            refinement_mask=(mask if bool(crit.get("brush", mask is not None)) else None),
            min_element_size=int(refinement.get("min_element_size", 8) or 8),
            half_win=int(half_win),
        )
    except Exception:  # noqa: BLE001 — never fail the solve on a bad policy
        return None


# ── input normalization ──────────────────────────────────────────────────────
def _to_float01(img: np.ndarray) -> np.ndarray:
    """Grayscale ``(H, W)`` float64 in [0, 1] (dtype-aware scaling)."""
    a = np.asarray(img)
    if a.ndim == 3:                       # collapse an accidental channel axis
        a = a.max(axis=0) if a.shape[0] <= 4 else a[a.shape[0] // 2]
    a = a.astype(np.float64)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _as_plane(img: np.ndarray) -> np.ndarray:
    """``(H, W)`` float64 view of one frame — the dtype/shape half of ``_to_float01``.

    Does the accidental-channel-axis collapse and the float64 cast but NOT the min-max
    rescale, because al_dic's own ``normalize_one`` z-scores the frame immediately
    afterwards and a z-score is invariant under any affine rescale of its input."""
    a = np.asarray(img)
    if a.ndim == 3:                       # collapse an accidental channel axis
        a = a.max(axis=0) if a.shape[0] <= 4 else a[a.shape[0] // 2]
    return a.astype(np.float64, copy=False)


class _LazyFrameProvider:
    """al_dic ``FrameProvider`` over a lazily-indexed frame sequence.

    ``run_aldic`` accepts any object exposing ``__len__`` / ``shape`` / ``clamped_roi``
    / ``get_normalized`` in place of a frame list, and calls ``get_normalized`` exactly
    ONCE per frame (once to build the reference bundle, once for that frame's FFT
    image — ``core/pipeline.py`` lines 987 and 1002). So a sequence whose
    ``__getitem__`` pulls a plane from a reader streams an arbitrarily long series at
    roughly TWO planes of memory, instead of the ``T x H x W`` float64 stack that
    ``ListFrameProvider`` materializes up front.

    Normalization is al_dic's own ``normalize_one`` — ``(img - mean_roi) / std_roi`` —
    applied to raw input values. Verified identical to the old
    ``_to_float01``-then-normalize path to 2.8e-14 px."""

    def __init__(self, images: Sequence[np.ndarray], roi_range) -> None:
        from al_dic.io.image_ops import compute_clamped_roi
        self._images = images
        self._first = _as_plane(images[0])          # frame 0 is always read at least once
        self._shape = (int(self._first.shape[0]), int(self._first.shape[1]))
        self._roi = compute_clamped_roi(self._shape, roi_range)

    def __len__(self) -> int:
        return len(self._images)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def clamped_roi(self):
        return self._roi

    def get_normalized(self, idx: int) -> np.ndarray:
        from al_dic.io.image_ops import normalize_one
        plane = self._first if idx == 0 else _as_plane(self._images[idx])
        if plane.shape != self._shape:
            raise ValueError(
                f"AL-DIC frame {idx} is {plane.shape}, but frame 0 is {self._shape}; "
                "every frame in a series must have the same Y/X")
        return normalize_one(plane, self._roi)


def _snap_pow2(v: int) -> int:
    """Nearest power of two >= 2 (``winstepsize`` / ``winsize_min`` must be pow2)."""
    v = max(2, int(v))
    return int(2 ** round(np.log2(v)))


def _mask_bbox(masks: Sequence[np.ndarray], grow: int,
               img_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """``(y0, y1, x0, x1)`` inclusive bbox of the union of every mask's ON pixels.

    ``None`` when the masks are empty or already cover the frame (nothing to tighten).

    ``grow`` is 0 in practice — matching upstream's own ``roi_range_from`` (a bare
    ``xs.min()/xs.max()`` bbox, ``examples/batch_process.py:131``) — and exists only to
    make the trade-off explicit: the solver insets whatever range it is given by
    ``winsize//2`` (``integer_search`` clamps to ``[half_w, dim-1-half_w]``), so padding
    the bbox by a half-window reproduces the untightened grid EXACTLY and buys nothing.
    Dropping the pad costs at most the ring of nodes within a half-window of the mask
    edge — nodes whose subset is mostly outside the ROI anyway. Measured: on an annular
    ROI the tight bbox solves the SAME 72 nodes as the full frame from 144 instead of 196
    candidates; on an off-centre rectangle it solves 63 where the full frame solved 48,
    because the grid now lands inside the ROI instead of straddling it. Note the grid
    origin therefore MOVES when a mask is present — the node positions are subset centres,
    not physical landmarks, so this changes sampling, not measurements."""
    H, W = int(img_size[0]), int(img_size[1])
    y0 = x0 = np.inf
    y1 = x1 = -np.inf
    for m in masks:
        a = np.asarray(m)
        rows = np.flatnonzero(a.any(axis=1))
        cols = np.flatnonzero(a.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        y0, y1 = min(y0, rows[0]), max(y1, rows[-1])
        x0, x1 = min(x0, cols[0]), max(x1, cols[-1])
    if not np.isfinite(y0) or not np.isfinite(x0):
        return None                                   # every mask is empty
    y0 = int(max(0, y0 - grow)); y1 = int(min(H - 1, y1 + grow))
    x0 = int(max(0, x0 - grow)); x1 = int(min(W - 1, x1 + grow))
    if y0 <= 0 and x0 <= 0 and y1 >= H - 1 and x1 >= W - 1:
        return None                                   # already the whole frame
    return (y0, y1, x0, x1)


def _build_dicpara(config, params: Dict[str, Any], img_size: Tuple[int, int],
                   reference_mode: str, use_masks: bool,
                   roi_bbox: Optional[Tuple[int, int, int, int]] = None):
    """Map node params → an ``al_dic`` ``DICPara`` (validated by ``dicpara_default``)."""
    winsize = max(2, int(params.get("winsize", 40) or 40))
    if winsize % 2:                       # winsize must be even
        winsize += 1
    winstep = _snap_pow2(int(params.get("winstepsize", 16) or 16))
    winmin = min(_snap_pow2(int(params.get("winsize_min", 8) or 8)), winstep)
    overrides: Dict[str, Any] = {
        "winsize": winsize,
        "winstepsize": winstep,
        "winsize_min": winmin,
        "init_guess_mode": str(params.get("init_guess_mode", "auto") or "auto"),
        "mu": float(params.get("mu", 1e-3) or 1e-3),
        "tol": float(params.get("tol", 1e-2) or 1e-2),
        "admm_max_iter": max(1, int(params.get("admm_max_iter", 3) or 3)),
        "icgn_max_iter": max(1, int(params.get("icgn_max_iter", 100) or 100)),
        "disp_smoothness": max(0.0, float(params.get("disp_smoothness", 5e-4) or 0.0)),
        "strain_smoothness": max(0.0, float(params.get("strain_smoothness", 1e-5) or 0.0)),
        "reference_mode": reference_mode,
        "img_size": (int(img_size[0]), int(img_size[1])),
        # al-dic >=0.7 requires the correlation ROI range EXPLICITLY when run_aldic() is
        # called directly (it no longer auto-derives it — it defaults to a zero-size box,
        # which yields "No grid points generated"). With no mask, the FULL image extent;
        # the solver insets by winsize//2 itself (integer_search min/max_x = clamp to
        # [half_w, dim-1-half_w]). Passed as (x, y) so gridx spans width = img_size[1],
        # gridy spans height = img_size[0].
        # With a mask, `roi_bbox` narrows the range to the mask's own bounding box — which
        # is exactly what upstream's own scripted example does
        # (examples/batch_process.py:221 `gridxy_roi_range=roi_range_from(masks[0])`,
        # whose docstring notes the field "has no usable default ... a script must set it
        # explicitly"). Untightened, the solver lays a grid over the whole frame and then
        # returns NaN for every node the mask excludes, paying full IC-GN cost for answers
        # it discards: on a disc ROI covering ~28% of a 512² frame that was 68% of the
        # emitted points.
        "gridxy_roi_range": (
            config.GridxyROIRange(gridx=(0, int(img_size[1])),
                                  gridy=(0, int(img_size[0])))
            if roi_bbox is None else
            config.GridxyROIRange(gridx=(int(roi_bbox[2]), int(roi_bbox[3])),
                                  gridy=(int(roi_bbox[0]), int(roi_bbox[1])))),
        # ± half-range in px of the FFT integer search that seeds IC-GN. al_dic's own
        # default is 20 and its example configs raise it to 30 for a rotation dataset;
        # it must cover the largest per-pair displacement or the local solve starts from
        # the wrong speckle. run_aldic SILENTLY SHRINKS it when
        # 2*search + winsize > min(H, W)/4 (warning "Auto-scaled FFT search region").
        "size_of_fft_search_region": max(1, int(
            params.get("size_of_fft_search_region", 20) or 20)),
        # INERT in al-dic 0.7.2 — `use_masks` is declared at data_structures.py:288 and
        # referenced nowhere else in the package; real masking flows through the `masks`
        # list, which run_aldic installs as para.img_ref_mask per frame (pipeline.py:1004).
        # Kept because it is the honest value, and because the caller's own "is there a
        # real mask?" test (which drives roi_bbox above) computes it anyway.
        "use_masks": bool(use_masks),
        # AL-DIC (default) alternates the local IC-GN matches with a global FEM step;
        # False is plain Local DIC — roughly 2x faster and, on smooth fields, no less
        # accurate (upstream's own case9). Everything the global step alone reads (mu,
        # admm_max_iter, disp_smoothness) is dead when this is False.
        "use_global_step": bool(params.get("use_global_step", True)),
        # Keep displacement in *pixels* — DVCResult stores voxels and converts to
        # µm itself via ``voxel_size_um`` (matching the DVC engine).
        "um2px": 1.0,
        "show_plots": False,
    }
    return config.dicpara_default(**overrides)


# ── output adapter (adaptive FE mesh → regular grid) ─────────────────────────
def _mesh_coords(mesh) -> np.ndarray:
    """``(N, 2)`` node coordinates ``[x, y]`` from an ``al_dic`` ``DICMesh``."""
    return np.asarray(getattr(mesh, "coordinates_fem"), dtype=float).reshape(-1, 2)


def _regular_grid(coords_xy: np.ndarray, step: int
                  ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Regular grid (pitch = ``step``) spanning the node bbox.

    Returns ``(Yq, Xq, (Gy, Gx))`` where ``Yq``/``Xq`` are ``(Gy, Gx)`` meshes in
    image coordinates (row = y, col = x)."""
    x_min, y_min = coords_xy[:, 0].min(), coords_xy[:, 1].min()
    x_max, y_max = coords_xy[:, 0].max(), coords_xy[:, 1].max()
    step = max(1, int(step))
    xs = np.arange(x_min, x_max + 1e-6, step)
    ys = np.arange(y_min, y_max + 1e-6, step)
    if xs.size < 1:
        xs = np.asarray([x_min])
    if ys.size < 1:
        ys = np.asarray([y_min])
    Xq, Yq = np.meshgrid(xs, ys)          # (Gy, Gx)
    return Yq, Xq, (int(ys.size), int(xs.size))


def _lattice_index(coords_xy: np.ndarray, Yq: np.ndarray,
                   Xq: np.ndarray) -> Optional[np.ndarray]:
    """``(Gy*Gx,)`` node index per query cell when the mesh IS exactly the query lattice.

    al_dic's default ``mesh_type="uniform"`` mesh is a perfect regular lattice of pitch
    ``winstepsize``, which is precisely what :func:`_regular_grid` reconstructs — so the
    scattered-data interpolation the adapter used to run was an identity map paid for
    with a Delaunay triangulation (measured ~0.8 s per component at 2048²/step 16).
    When the two coincide, resampling is an index permutation.

    Returns ``None`` — meaning "fall back to real interpolation" — for any mesh that is
    not that lattice, which is the adaptive quadtree-refined case."""
    xs = np.asarray(Xq)[0, :]
    ys = np.asarray(Yq)[:, 0]
    n = coords_xy.shape[0]
    if n != xs.size * ys.size or n == 0:
        return None
    ix = np.searchsorted(xs, coords_xy[:, 0])
    iy = np.searchsorted(ys, coords_xy[:, 1])
    if ix.max(initial=0) >= xs.size or iy.max(initial=0) >= ys.size:
        return None
    if not (np.allclose(xs[ix], coords_xy[:, 0], atol=1e-6)
            and np.allclose(ys[iy], coords_xy[:, 1], atol=1e-6)):
        return None
    flat = iy * xs.size + ix
    inv = np.full(flat.size, -1, dtype=np.int64)
    inv[flat] = np.arange(flat.size, dtype=np.int64)
    if (inv < 0).any():                   # duplicate/missing node ⇒ not a clean lattice
        return None
    return inv


def _interp_to_grid(coords_xy: np.ndarray, values: np.ndarray,
                    Yq: np.ndarray, Xq: np.ndarray,
                    lattice: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample node ``values`` onto the query grid.

    ``lattice`` is the :func:`_lattice_index` permutation when the mesh already IS the
    query grid — the common uniform-mesh case, where this is exact and O(N). Otherwise
    falls back to the original scattered interpolation (linear + nearest hole fill)."""
    if lattice is not None:
        return np.asarray(values, dtype=np.float64)[lattice].reshape(Yq.shape)
    from scipy.interpolate import griddata
    pts = coords_xy                       # (N, 2) as (x, y)
    query = np.column_stack([Xq.ravel(), Yq.ravel()])
    out = np.full(query.shape[0], np.nan, dtype=np.float64)
    if pts.shape[0] >= 4:
        out = griddata(pts, values, query, method="linear")
    holes = ~np.isfinite(out)
    if holes.any() and pts.shape[0] >= 1:
        fill = griddata(pts, values, query, method="nearest")
        out[holes] = fill[holes]
    return out.reshape(Yq.shape)


def _strain_components(strain) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]]:
    """``(dudx, dudy, dvdx, dvdy)`` from an al_dic ``StrainResult``, in IMAGE axes.

    These are al_dic's FEM/plate-fit nodal displacement gradients — notably better than
    the raw per-subset ``F`` the IC-GN step reports (on the 2% biaxial case ``dudx``
    lands at 0.01984 against truth 0.02, where raw ``F11`` gives 0.01928).

    THE TWO CROSS TERMS ARE NEGATED. al_dic returns the y-derivative and the
    v-component under a y-UP convention while its *displacements* are plain image-row
    (y-down), so ``dudy`` and ``dvdx`` — the terms with an odd number of y/v factors —
    come back with the opposite sign to ``dudx``/``dvdy``. Measured on three analytic
    cases (u = 0.015·(y−cy) shear: al_dic ``dudy`` = −0.01494 for truth +0.015; 2°
    rotation: ``dudy`` = +0.03486 / ``dvdx`` = −0.03486 for truth −0.03490 / +0.03490;
    2% biaxial: both diagonals correct at +0.0198).

    Upstream's own strain tests cannot catch this — ``test_synthetic.py`` asserts
    case5_shear ``F12`` = 0.015 with ``strain_tol`` = 0.04 and case10_rotation with
    ``strain_tol`` = 0.08, and in both the tolerance exceeds *twice* the signal, so a
    fully sign-flipped cross term still passes."""
    if strain is None:
        return None
    got = tuple(getattr(strain, k, None) for k in ("dudx", "dudy", "dvdx", "dvdy"))
    if any(g is None for g in got):
        return None
    dudx, dudy, dvdx, dvdy = (np.asarray(g, dtype=float).reshape(-1) for g in got)
    return dudx, -dudy, -dvdx, dvdy


def _frame_to_dvcresult(mesh, U: np.ndarray, step: int,
                        voxel_size_um: Tuple[float, ...], method: str,
                        params: Dict[str, Any], reference_mode: str,
                        notes: str = "", *, resample: str = "grid",
                        strain=None, converged: bool = True,
                        iterations: Optional[int] = None) -> DVCResult:
    """Convert one pyALDIC frame result (mesh + interleaved U) to a 2D DVCResult.

    pyALDIC: coords ``[x, y]``, ``U = [u0, v0, u1, v1, ...]`` (u = x-disp, v = y-disp).
    DVCResult: ``grid_coords[..., 0] = y, [..., 1] = x``; displacement
    ``[..., 0] = dy, [..., 1] = dx`` — hence the swap below.

    ``resample="grid"`` (default, the historical contract) returns a regular
    ``(Gy, Gx, …)`` field of pitch ``step``. ``resample="nodes"`` returns the FE mesh
    nodes themselves as an ``(N, …)`` point set: exact, no interpolation, and it keeps
    the extra nodes an adaptive quadtree refinement produced — resampling those onto the
    coarse lattice silently discarded exactly the detail the refinement was for."""
    coords = _mesh_coords(mesh)           # (N, 2) [x, y]
    u = np.asarray(U, dtype=float).reshape(-1)
    n = coords.shape[0]
    if u.size >= 2 * n and n > 0:
        u_x = u[0:2 * n:2]                # x-displacement per node
        v_y = u[1:2 * n:2]               # y-displacement per node
    else:                                 # defensive: degenerate result
        u_x = np.zeros(n)
        v_y = np.zeros(n)
    grads = _strain_components(strain)
    if grads is not None and grads[0].size != n:
        grads = None                      # defensive: strain not on this mesh

    if resample == "nodes":
        grid_coords = np.stack([coords[:, 1], coords[:, 0]], axis=-1)   # (N, 2) [y, x]
        disp = np.stack([v_y, u_x], axis=-1)                            # (N, 2) [dy, dx]
        grid_shape: Tuple[int, ...] = (int(n),)
        strain_field = None
        if grads is not None:
            dudx, dudy, dvdx, dvdy = grads
            # strain[i, j] = d(disp_i)/d(axis_j) in DVCResult's own [y, x] axis order.
            strain_field = np.stack([np.stack([dvdy, dvdx], axis=-1),
                                     np.stack([dudy, dudx], axis=-1)], axis=-2)
    else:
        Yq, Xq, grid_shape = _regular_grid(coords, step)
        lat = _lattice_index(coords, Yq, Xq)
        dy = _interp_to_grid(coords, v_y, Yq, Xq, lat)     # (Gy, Gx) y-disp
        dx = _interp_to_grid(coords, u_x, Yq, Xq, lat)     # (Gy, Gx) x-disp
        grid_coords = np.stack([Yq, Xq], axis=-1)          # (Gy, Gx, 2) [y, x]
        disp = np.stack([dy, dx], axis=-1)                 # (Gy, Gx, 2) [dy, dx]
        strain_field = None
        if grads is not None:
            g = [_interp_to_grid(coords, c, Yq, Xq, lat) for c in grads]
            dudx, dudy, dvdx, dvdy = g
            strain_field = np.stack([np.stack([dvdy, dvdx], axis=-1),
                                     np.stack([dudy, dudx], axis=-1)], axis=-2)

    return DVCResult(
        dim=2,
        grid_coords=np.asarray(grid_coords, dtype=np.float64),
        displacement_field=np.asarray(disp, dtype=np.float64),
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        # (Gy, Gx, 2, 2) / (N, 2, 2) nodal displacement gradient, or None when the
        # caller left compute_strain off — al_dic's own FEM strain, not a finite
        # difference taken on the coarse grid downstream.
        strain_field=strain_field,
        strain_type="infinitesimal",
        # qfactor stays None: al-dic 0.7.2 keeps its per-point NCC quality factors
        # inside the FFT integer search (solver/integer_search.py) and does not surface
        # them on PipelineResult, so there is no honest per-node confidence to report.
        converged=bool(converged),
        iterations=int(iterations if iterations is not None
                       else (params.get("admm_max_iter", 3) or 3)),
        mu=float(params.get("mu", 1e-3) or 1e-3),
        method=method,
        notes=notes,
        diagnostics={
            "engine": "pyALDIC (al-dic)",
            "n_nodes": int(n),
            "grid_shape": tuple(int(v) for v in grid_shape),
            "winsize": int(params.get("winsize", 40) or 40),
            "winstepsize": int(step),
            "reference_mode": reference_mode,
            "resample": str(resample),
            "solver": ("aldic" if params.get("use_global_step", True) else "local"),
            "has_strain": bool(strain_field is not None),
        },
    )


# ── public entry points ──────────────────────────────────────────────────────
def run_pyaldic_series(
    images: Sequence[np.ndarray],
    masks: Optional[Sequence[np.ndarray]],
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, ...],
    *,
    reference_mode: str = "accumulative",
    refinement: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCb = None,
    cancelled_cb: CancelledCb = None,
    resample: str = "grid",
) -> List[Dict[str, DVCResult]]:
    """Run AL-DIC over an ordered image series and adapt the results.

    ``images[0]`` is the reference. Returns a list of length ``len(images) - 1``
    (one entry per deformed frame, in order) — each a dict
    ``{"primary": DVCResult, "increment": DVCResult}`` where *primary* is the
    cumulative field (from pyALDIC's ``U_accum`` when present) and *increment* is
    the raw per-step field (``U``). Cumulative vs incremental behaviour is chosen
    natively by ``reference_mode`` ("accumulative" | "incremental").

    **Prefer this over N calls to** :func:`run_pyaldic_pair`. One call over the whole
    ordered stack lets al_dic keep its reference bundle, subpb1 precompute and 6-DOF
    IC-GN context (all keyed on the reference frame index, all expensive) across every
    frame, and remembers the FFT search radius that worked. Measured 1.8x at 256²/T=9 and
    1024²/T=5 with ``init_guess_mode="fft"``, for BIT-IDENTICAL per-frame fields.

    Note ``init_guess_mode``: ``"fft"`` gives every frame its own coarse integer search,
    which is what upstream recommends and what a pair loop did implicitly. ``"auto"``
    (al_dic maps it to ``"previous"``) warm-starts each frame from the last solution — a
    little faster on smoothly-growing motion, but it cannot find a large FIRST step, and
    in accumulative mode a leading self-pair will hand ~0 forward as that guess.

    ``images`` is indexed lazily — any object with ``__len__``/``__getitem__`` works, so
    a caller with a plane reader can hand in a view that pulls frames on demand and keep
    peak memory at ~two planes instead of the whole ``T x H x W`` float64 stack.

    ``resample`` — ``"grid"`` for the regular-lattice field (the historical contract),
    ``"nodes"`` for the FE mesh nodes themselves (exact, and refinement-preserving).

    Raises ``ImportError`` (friendly) when ``al-dic`` is not installed and
    ``RuntimeError`` when the user cancels.
    """
    config, pipeline = _require_al_dic()
    n_img = len(images)
    if n_img < 2:
        raise ValueError("AL-DIC needs at least 2 images (reference + deformed).")
    H, W = _as_plane(images[0]).shape[:2]
    roi_bbox = None
    if masks is None:
        # ONE shared all-ones array referenced n times, not n distinct allocations:
        # al_dic only reads masks, and a per-frame 2048² float64 is 33 MB apiece.
        ones = np.ones((H, W), dtype=np.float64)
        mask_list = [ones] * n_img
        use_masks = False
    else:
        mask_list = [np.asarray(m).astype(np.float64) for m in masks]
        use_masks = any(float(m.min()) < 1.0 for m in mask_list)
        if use_masks:
            roi_bbox = _mask_bbox(mask_list, 0, (H, W))
    para = _build_dicpara(config, params, (H, W), reference_mode, use_masks, roi_bbox)
    winsize = int(getattr(para, "winsize", params.get("winsize", 40)))
    policy = _refinement_policy(refinement, half_win=max(1, winsize // 2))

    def _progress(frac: float, _msg: str = "") -> None:
        if progress_cb is not None:
            progress_cb(int(max(0.0, min(1.0, float(frac))) * 100))

    def _stop() -> bool:
        return bool(cancelled_cb()) if cancelled_cb is not None else False

    want_strain = bool(params.get("compute_strain", True))
    result = pipeline.run_aldic(
        para, _LazyFrameProvider(images, para.gridxy_roi_range), mask_list,
        progress_fn=_progress, stop_fn=_stop,
        compute_strain=want_strain,
        refinement_policy=policy,
    )

    disp = list(getattr(result, "result_disp", []) or [])
    meshes = list(getattr(result, "result_fe_mesh_each_frame", []) or [])
    strains = list(getattr(result, "result_strain", []) or [])
    canonical = getattr(result, "dic_mesh", None)
    step = int(getattr(para, "winstepsize", params.get("winstepsize", 16)))
    # A frame after an early stop was never solved; only frames before it converged.
    stopped_at = getattr(result, "stopped_at_frame", None)
    method = "pyALDIC"
    out: List[Dict[str, DVCResult]] = []
    for i, fr in enumerate(disp):
        mesh = meshes[i] if i < len(meshes) and meshes[i] is not None else canonical
        if mesh is None:
            continue
        u_incr = getattr(fr, "U", None)
        u_accum = getattr(fr, "U_accum", None)
        if u_accum is None:
            u_accum = u_incr
        sr = strains[i] if want_strain and i < len(strains) else None
        ok = not (getattr(result, "stopped_early", False)
                  and stopped_at is not None and i >= int(stopped_at))
        iters = int(getattr(para, "admm_max_iter", 3)) if para.use_global_step else 1
        common = dict(resample=resample, converged=ok, iterations=iters)
        primary = _frame_to_dvcresult(
            mesh, u_accum, step, voxel_size_um, method, params, reference_mode,
            notes="cumulative", strain=sr, **common)
        increment = _frame_to_dvcresult(
            mesh, u_incr, step, voxel_size_um, method, params, reference_mode,
            notes="increment", **common)
        # the FE mesh this frame was solved on (2026-10-09) — `[x, y]` node coordinates
        # and the Q8 elements, so the node can draw the grid (incl. adaptive refinement)
        out.append({"primary": primary, "increment": increment,
                    "mesh": {"coords": _mesh_coords(mesh),
                             "elements": np.asarray(getattr(mesh, "elements_fem",
                                                            np.zeros((0, 8))),
                                                    dtype=np.int64)}})
    return out


def run_pyaldic_pair(
    ref_img: np.ndarray,
    def_img: np.ndarray,
    voxel_size_um: Tuple[float, ...],
    params: Dict[str, Any],
    progress_cb: ProgressCb = None,
    cancelled_cb: CancelledCb = None,
    *,
    roi_mask: Optional[np.ndarray] = None,
    refinement: Optional[Dict[str, Any]] = None,
    resample: str = "grid",
) -> DVCResult:
    """Single reference/deformed pair convenience (headless + the DVCMethod ABC).

    Returns the cumulative :class:`DVCResult` for ``def_img`` vs ``ref_img``.

    For an ordered SERIES call :func:`run_pyaldic_series` once instead of looping this —
    the per-pair loop re-pays al_dic's reference precompute on every frame (2.4x
    measured at 256²)."""
    masks = None
    if roi_mask is not None:
        m = np.asarray(roi_mask).astype(np.float64)
        masks = [m, m]
    series = run_pyaldic_series(
        [ref_img, def_img], masks, params, voxel_size_um,
        reference_mode="accumulative", refinement=refinement,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb, resample=resample)
    if not series:
        raise RuntimeError("AL-DIC produced no result for the image pair.")
    return series[0]["primary"]


# ═══════════════════════════════════════════════════════════════════════════════
# ROI rasterization — vendored verbatim from nd2studios/backend/dic/roi.py
# ═══════════════════════════════════════════════════════════════════════════════
# Marks a shape whose ``op`` is Cut (subtract from the mask) rather than Add.
OP_ADD = "add"
OP_CUT = "cut"


def _rasterize_region(shape: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """Rasterize a single shape to a boolean ``(H, W)`` region (True = painted)."""
    from skimage.draw import disk as sk_disk
    from skimage.draw import ellipse as sk_ellipse
    from skimage.draw import polygon as sk_polygon

    region = np.zeros((H, W), dtype=bool)
    s_type = str(shape.get("type", ""))
    verts = shape.get("vertices") or []
    v = np.asarray(verts, dtype=float) if verts else np.empty((0, 2))

    if s_type == "rect" and v.shape[0] == 2:
        y0, y1 = sorted([v[0, 0], v[1, 0]])
        x0, x1 = sorted([v[0, 1], v[1, 1]])
        iy0, iy1 = max(0, int(round(y0))), min(H, int(round(y1)) + 1)
        ix0, ix1 = max(0, int(round(x0))), min(W, int(round(x1)) + 1)
        if iy1 > iy0 and ix1 > ix0:
            region[iy0:iy1, ix0:ix1] = True
        return region

    if s_type == "ellipse" and v.shape[0] == 2:
        cy = (v[0, 0] + v[1, 0]) / 2.0
        cx = (v[0, 1] + v[1, 1]) / 2.0
        ry = abs(v[1, 0] - v[0, 0]) / 2.0
        rx = abs(v[1, 1] - v[0, 1]) / 2.0
        if ry > 0 and rx > 0:
            rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W))
            region[rr, cc] = True
        return region

    if s_type == "circle":
        c = shape.get("center")
        r = float(shape.get("radius", 0) or 0)
        if c is not None and r > 0:
            rr, cc = sk_disk((float(c[0]), float(c[1])), r, shape=(H, W))
            region[rr, cc] = True
        return region

    if s_type == "polygon" and v.shape[0] >= 3:
        rr, cc = sk_polygon(v[:, 0], v[:, 1], shape=(H, W))
        region[rr, cc] = True
        return region

    if s_type == "brush" and v.shape[0] >= 1:
        r = max(1.0, float(shape.get("radius", 8) or 8))
        # Stamp a disk at each vertex and along each segment (sampled at ~r/2)
        # so a freehand stroke paints a continuous thick band.
        pts: List[np.ndarray] = []
        for i in range(v.shape[0]):
            pts.append(v[i])
            if i + 1 < v.shape[0]:
                seg = v[i + 1] - v[i]
                dist = float(np.hypot(seg[0], seg[1]))
                n = int(dist / max(1.0, r / 2.0))
                for k in range(1, n):
                    pts.append(v[i] + seg * (k / float(n)))
        for p in pts:
            rr, cc = sk_disk((float(p[0]), float(p[1])), r, shape=(H, W))
            region[rr, cc] = True
        return region

    return region


def build_roi_mask(shapes: Optional[Sequence[Dict[str, Any]]],
                   H: int, W: int) -> np.ndarray:
    """Replay a shape list into a boolean ``(H, W)`` mask (True = inside).

    Actions apply in order: Add ``|=`` region, Cut ``&= ~`` region, Invert flips
    the whole mask, Clear zeros it. An empty / missing list yields an all-False
    mask (the caller decides whether "no ROI" means the full frame)."""
    mask = np.zeros((int(H), int(W)), dtype=bool)
    for shape in (shapes or []):
        if not isinstance(shape, dict):
            continue
        s_type = str(shape.get("type", ""))
        if s_type == "invert":
            mask = ~mask
            continue
        if s_type == "clear":
            mask[:] = False
            continue
        region = _rasterize_region(shape, int(H), int(W))
        if str(shape.get("op", OP_ADD)) == OP_CUT:
            mask &= ~region
        else:
            mask |= region
    return mask


def has_region(shapes: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """True if the shape list contains at least one drawable (non-clear) action."""
    for shape in (shapes or []):
        if isinstance(shape, dict) and str(shape.get("type", "")) not in ("", "clear"):
            return True
    return False


# ── staged entry points (2026-10-08): pyALDIC's sections as separate calls ─────
# `run_aldic` (al-dic 0.9.0 `core/pipeline.py:700-1871`) is Sections 2b-8 of the ALDIC
# paper in one ~900-line function. The staged DIC nodes — `dic.grid` → `dic.initial_guess`
# → `dic.local_match` → `dic.global_step` → `dic.strain` — call the functions below one at
# a time; each reproduces its slice of that driver (the same calls in the same order, the
# same `para` rewrites, the mesh trimmed by the mask exactly as the driver trims it). The
# staged chain therefore agrees with `run_pyaldic_series` on every node for the paths it
# covers: the FFT seed (`init_guess_mode="fft"`, what this repo's single node uses), no
# pre-solve mesh refinement, no seed propagation. Everything crosses this boundary in
# pyALDIC's own layout — ``U`` interleaved ``[u0, v0, u1, v1, …]`` with ``u`` along x
# (columns) and ``v`` along y (rows), ``F`` interleaved ``[du/dx, dv/dx, du/dy, dv/dy]``,
# node coordinates ``(N, 2)`` as ``[x, y]`` — and the nodes convert to this repo's
# ``(y, x)`` columns at their edge, as `_frame_to_dvcresult` does.


def _al_dic_modules() -> Dict[str, Any]:
    """Every pyALDIC module a staged call touches, imported lazily and by path."""
    config, pipeline = _require_al_dic()
    m = importlib.import_module
    return {
        "config": config, "pipeline": pipeline,
        "ds": m("al_dic.core.data_structures"),
        "image_ops": m("al_dic.io.image_ops"),
        "mesh_setup": m("al_dic.mesh.mesh_setup"),
        "bridging": m("al_dic.mesh.mark_bridging"),
        "rasterize": m("al_dic.mesh.rasterize"),
        "init_disp": m("al_dic.solver.init_disp"),
        "isearch": m("al_dic.solver.integer_search"),
        "local": m("al_dic.solver.local_icgn"),
        "sub1": m("al_dic.solver.subpb1_solver"),
        "sub2": m("al_dic.solver.subpb2_solver"),
        "strain": m("al_dic.strain.compute_strain"),
        "nsfem": m("al_dic.strain.nodal_strain_fem"),
        "platefit": m("al_dic.strain.platefit_kernel"),
        "regions": m("al_dic.utils.region_analysis"),
        "seedprop": m("al_dic.solver.seed_prop_pipeline"),
    }


@dataclass
class DICReference:
    """One reference frame as every later section sees it (`pipeline.py:981-1004`):
    the normalised frame (``f_img_raw``), its masked copy (``f_img``), the mask (all ones
    without an ROI), the 7-point image gradients and the ``para`` with ``img_ref_mask``
    installed. ``shape`` is ``(H, W)``."""
    para: Any
    shape: Tuple[int, int]
    f_img_raw: np.ndarray
    f_img: np.ndarray
    f_mask: np.ndarray
    Df: Any


def dic_stage_para(params: Dict[str, Any], shape: Tuple[int, int], *,
                   reference_mode: str = "accumulative",
                   roi_mask: Optional[np.ndarray] = None):
    """Section 2b's parameter rewrite: the node's params → a validated ``DICPara`` with
    the correlation ROI clamped to the image (and to the mask's bounding box when there
    is one), ``img_size`` set and the FFT search region auto-scaled to what the image can
    hold — what `run_aldic` does at `pipeline.py:773-799` before any frame is read.
    Masks are per frame in pyALDIC; the staged nodes use one static ROI per
    ``(m, z)`` line, so one ``roi_mask`` (truthy = inside) stands for every frame."""
    M = _al_dic_modules()
    from dataclasses import replace
    H, W = int(shape[0]), int(shape[1])
    use_masks = roi_mask is not None and bool(np.any(~np.asarray(roi_mask, dtype=bool)))
    roi_bbox = None
    if use_masks:
        mk = np.asarray(roi_mask, dtype=np.float64)
        roi_bbox = _mask_bbox([mk], 0, (H, W))
    para = _build_dicpara(M["config"], dict(params or {}), (H, W), reference_mode, use_masks,
                          roi_bbox)
    clamped = M["image_ops"].compute_clamped_roi((H, W), para.gridxy_roi_range)
    para = replace(para, gridxy_roi_range=clamped, img_size=(H, W))
    max_safe = max(10, min(H, W) // 4 - int(para.winsize))
    if int(para.size_of_fft_search_region) > max_safe:
        para = replace(para, size_of_fft_search_region=int(max_safe))
    return para


def dic_stage_normalize(img: np.ndarray, para) -> np.ndarray:
    """``(img − mean_roi) / std_roi`` over the clamped correlation ROI — pyALDIC's own
    ``normalize_one``, the only thing its frame providers do to a frame."""
    M = _al_dic_modules()
    return np.asarray(M["image_ops"].normalize_one(_as_plane(img), para.gridxy_roi_range),
                      dtype=np.float64)


def dic_stage_grid(para, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The correlation lattice ``(x0, y0)`` for ``para`` on an image of ``shape`` —
    ``build_grid_for_roi``, the same lattice ``integer_search`` lays its windows on, so a
    grid card can show the measurement density before any correlation is paid for."""
    M = _al_dic_modules()
    x0, y0 = M["seedprop"].build_grid_for_roi(para, int(shape[0]), int(shape[1]))
    return np.asarray(x0, dtype=np.float64), np.asarray(y0, dtype=np.float64)


def dic_stage_reference(ref_norm: np.ndarray, para, shape: Tuple[int, int],
                        roi_mask: Optional[np.ndarray] = None) -> DICReference:
    """Section 2b-3 for ONE reference: the masked frame, its gradients and the ``para``
    that carries the mask (`pipeline.py:985-1004`). ``ref_norm`` is the frame AFTER
    :func:`dic_stage_normalize`."""
    M = _al_dic_modules()
    from dataclasses import replace
    H, W = int(shape[0]), int(shape[1])
    f_raw = np.ascontiguousarray(np.asarray(ref_norm, dtype=np.float64))
    if f_raw.shape != (H, W):
        raise ValueError(f"reference frame has shape {f_raw.shape}, expected {(H, W)}")
    f_mask = (np.asarray(roi_mask, dtype=np.float64) if roi_mask is not None
              else np.ones((H, W), dtype=np.float64))
    f_img = f_raw * f_mask
    Df = M["image_ops"].compute_image_gradient(f_img, f_mask, img_raw=f_raw)
    para = replace(para, img_ref_mask=f_mask)
    return DICReference(para=para, shape=(H, W), f_img_raw=f_raw, f_img=f_img, f_mask=f_mask,
                        Df=Df)


def dic_stage_seed(ref: DICReference, g_norm: np.ndarray, *, search: Optional[int] = None
                   ) -> Dict[str, Any]:
    """Section 3's integer seed: the FFT (or pyramid NCC) search of every lattice window
    with pyALDIC's auto-expanding retry (`pipeline.py:1088-1151`), then ``init_disp`` —
    inpainting, three outlier passes, the lattice re-ordered into mesh node order →
    ``{"x0", "y0", "U0", "cc", "search", "info"}``: the lattice, the interleaved seed
    ``(2N,)`` in node order, the correlation peak per node (node order), the search
    radius that finally covered the motion (pass it back in for the next frame against
    the same reference — the driver remembers it per reference) and the search's info."""
    M = _al_dic_modules()
    from dataclasses import replace
    para = ref.para
    if search is not None and int(search) != int(para.size_of_fft_search_region):
        para = replace(para, size_of_fft_search_region=int(search))
    g = np.ascontiguousarray(np.asarray(g_norm, dtype=np.float64))
    use_pyramid = int(para.init_fft_search_method) >= 2
    if use_pyramid:
        x0, y0, u_grid, v_grid, info = M["isearch"].integer_search_pyramid(ref.f_img_raw, g, para)
        current = int(para.size_of_fft_search_region)
    else:
        x0, y0, u_grid, v_grid, info = M["isearch"].integer_search(ref.f_img_raw, g, para)
        current = int(para.size_of_fft_search_region)
        if para.fft_auto_expand_search:
            H, W = ref.shape
            cap = max(32, min(H, W) // 2)
            for _retry in range(6):
                if not info.get("peak_clipped", False):
                    break
                needed = int(np.ceil(float(info["max_abs_disp"]) * 2.0)) + 2
                new_search = max(needed, current * 2)
                if new_search >= cap:
                    new_search = cap
                    if new_search == current:
                        break
                x0, y0, u_grid, v_grid, info = M["isearch"].integer_search(
                    ref.f_img_raw, g, replace(para, size_of_fft_search_region=new_search))
                current = new_search
                if new_search >= cap:
                    break
    U0 = np.asarray(M["init_disp"].init_disp(u_grid, v_grid, info["cc_max"], x0, y0),
                    dtype=np.float64)
    cc = np.asarray(info["cc_max"], dtype=np.float64)
    # init_disp lays U0 out as u_clean.T.ravel(): node = ix*ny + iy — the same transpose
    # takes the (ny, nx) peak grid into node order
    cc_nodes = cc.T.reshape(-1) if cc.ndim == 2 else cc.reshape(-1)
    return {"x0": np.asarray(x0, float), "y0": np.asarray(y0, float), "U0": U0, "cc": cc_nodes,
            "search": int(current), "info": info}


def dic_stage_mesh(ref: DICReference, x0: np.ndarray, y0: np.ndarray):
    """Section 3's mesh: ``mesh_setup`` on the lattice, then the elements more than half
    masked or bridging a crack dropped (`trimmed_keep_indices`, `pipeline.py:1160-1207`).
    Nodes are never removed — only elements — so the node count is the lattice's."""
    M = _al_dic_modules()
    DICMesh = M["ds"].DICMesh
    base = M["mesh_setup"].mesh_setup(np.asarray(x0, float), np.asarray(y0, float), ref.para)
    keep = M["bridging"].trimmed_keep_indices(base.coordinates_fem, base.elements_fem, ref.f_mask)
    elems = (base.elements_fem[keep] if len(keep) < base.elements_fem.shape[0]
             else base.elements_fem.copy())
    return DICMesh(coordinates_fem=base.coordinates_fem.copy(), elements_fem=elems,
                   irregular=base.irregular, mark_coord_hole_edge=base.mark_coord_hole_edge.copy(),
                   x0=base.x0, y0=base.y0, element_min_size=base.element_min_size)


def dic_stage_refine(ref: DICReference, mesh, U0: Optional[np.ndarray],
                     refinement: Optional[Dict[str, Any]]):
    """Pre-solve adaptive refinement (``pipeline.py:1374-1404``) as its own step
    (2026-10-09): the criteria of ``refinement`` (the same spec ``run_pyaldic_series``
    takes — ``criteria.mask_boundary`` / ``roi_edge`` / ``brush``, ``brush`` (H, W),
    ``min_element_size``) applied to ``mesh`` with ``al_dic.mesh.refinement.refine_mesh``,
    which quadtree-splits the marked elements, rebuilds the Q8 connectivity with hanging
    nodes, removes in-hole elements and interpolates ``U0`` onto the new nodes.

    → ``(mesh, U0)``: the input mesh and ``U0`` (or zeros) unchanged when no criterion is
    on. Refinement depends on the mesh, the mask and the criteria — never on ``U0`` — so a
    stage that needs the refined mesh again (Local Match, Global Step, Strain) rebuilds it
    exactly by calling this with ``U0=None``; that is how the staged chain carries a refined
    mesh without serialising pyALDIC's object."""
    n = int(mesh.coordinates_fem.shape[0])
    u0 = np.zeros(2 * n) if U0 is None else np.asarray(U0, dtype=np.float64).reshape(-1)
    policy = _refinement_policy(refinement, half_win=max(1, int(ref.para.winsize) // 2))
    if policy is None or not policy.has_pre_solve:
        return mesh, u0.copy()
    refine = importlib.import_module("al_dic.mesh.refinement")
    ctx = refine.RefinementContext(mesh=mesh, mask=ref.f_mask)
    h, w = (int(v) for v in np.asarray(ref.f_mask).shape[:2])
    return refine.refine_mesh(mesh, list(policy.pre_solve), ctx, u0, mask=ref.f_mask,
                              img_size=(h, w))


def dic_stage_seed_on_mesh(ref: DICReference, mesh, U0: np.ndarray, x0: np.ndarray,
                           y0: np.ndarray) -> np.ndarray:
    """The seed as Section 4 receives it: interpolated onto the mesh when the FFT lattice
    and the mesh disagree in size (`pipeline.py:1248-1269`), and NaN at every node whose
    pixel the mask excludes (`:1285-1296`)."""
    n = int(mesh.coordinates_fem.shape[0])
    U = np.asarray(U0, dtype=np.float64).copy()
    if U.size != 2 * n:
        from scipy.interpolate import RegularGridInterpolator
        ny, nx = len(y0), len(x0)
        u2 = U[0::2].reshape(nx, ny).T
        v2 = U[1::2].reshape(nx, ny).T
        iu = RegularGridInterpolator((y0, x0), u2, method="linear", bounds_error=False,
                                     fill_value=np.nan)
        iv = RegularGridInterpolator((y0, x0), v2, method="linear", bounds_error=False,
                                     fill_value=np.nan)
        xy = mesh.coordinates_fem
        q = np.column_stack([xy[:, 1], xy[:, 0]])
        U = np.zeros(2 * n, dtype=np.float64)
        U[0::2] = iu(q)
        U[1::2] = iv(q)
    H, W = ref.shape
    node_x = np.clip(np.round(mesh.coordinates_fem[:, 0]).astype(int), 0, W - 1)
    node_y = np.clip(np.round(mesh.coordinates_fem[:, 1]).astype(int), 0, H - 1)
    nan_nodes = np.where(ref.f_mask[node_y, node_x] < 1.0)[0]
    if len(nan_nodes):
        U[2 * nan_nodes] = np.nan
        U[2 * nan_nodes + 1] = np.nan
    return U


def dic_stage_local(ref: DICReference, mesh, g_norm: np.ndarray, U0: np.ndarray
                    ) -> Dict[str, Any]:
    """Section 4: the 6-DOF IC-GN of every subset from the seed (`pipeline.py:1426-1452`)
    → ``{"U", "F", "conv_iter", "bad_pt_num", "mark_hole_strain", "node_region_map",
    "solve_time"}``; ``U`` ``(2N,)`` and ``F`` ``(4N,)`` interleaved, bad points already
    IDW-filled inside ``local_icgn``, ``mark_hole_strain`` the nodes whose subset is
    mostly masked. The node-region map (connected regions of the mask) is computed here
    because every later section reads it."""
    M = _al_dic_modules()
    para = ref.para
    g = np.ascontiguousarray(np.asarray(g_norm, dtype=np.float64))
    ctx = M["local"].local_icgn_precompute(mesh.coordinates_fem, ref.Df, ref.f_img_raw, para)
    U1, F1, t_local, conv_iter, bad_pt_num, mark_hole = M["local"].local_icgn(
        np.asarray(U0, dtype=np.float64), mesh.coordinates_fem, ref.Df, ref.f_img_raw, g,
        para, para.tol, ctx=ctx)
    regions = M["regions"].precompute_node_regions(mesh.coordinates_fem, ref.f_mask, ref.shape)
    return {"U": np.asarray(U1, dtype=np.float64), "F": np.asarray(F1, dtype=np.float64),
            "conv_iter": np.asarray(conv_iter), "bad_pt_num": bad_pt_num,
            "mark_hole_strain": np.asarray(mark_hole, dtype=np.int64),
            "node_region_map": regions, "solve_time": float(t_local)}


def dic_stage_global(ref: DICReference, mesh, g_norm: np.ndarray, U1: np.ndarray,
                     F1: np.ndarray, mark_hole_strain: np.ndarray, node_region_map, *,
                     beta: Optional[float] = None) -> Dict[str, Any]:
    """Sections 5-6 (`pipeline.py:1465-1673`, verbatim): optional smoothing of the local
    result, β by the L-curve unless given, the global FEM solve on the mesh with the
    hole-strain elements trimmed, the post-solve corrections, then the ADMM alternation
    (3-DOF local re-solve ↔ global solve, duals reassigned each step, stop on
    ``admm_tol``) → ``{"U", "F", "beta", "n_steps", "update_global", "update_local"}``.
    ``beta`` is tuned once per reference by the driver and reused — pass the first
    frame's back in for the next frame against the same reference."""
    M = _al_dic_modules()
    from dataclasses import replace
    P = M["pipeline"]
    para = ref.para
    g = np.ascontiguousarray(np.asarray(g_norm, dtype=np.float64))
    n = int(mesh.coordinates_fem.shape[0])
    U1 = np.asarray(U1, dtype=np.float64).copy()
    F1 = np.asarray(F1, dtype=np.float64).copy()
    mhs = np.asarray(mark_hole_strain, dtype=np.int64)
    if para.disp_smoothness > 1e-6:
        U1 = P._smooth_disp(U1, mesh.coordinates_fem, para, node_region_map, mesh=mesh)
    if para.strain_smoothness > 1e-6:
        F1 = P._smooth_strain_field(F1, mesh.coordinates_fem, para, node_region_map, mesh=mesh)
    mu = float(para.mu)
    grad_dual = np.zeros(4 * n, dtype=np.float64)
    disp_dual = np.zeros(2 * n, dtype=np.float64)
    alpha = para.alpha
    if para.beta is not None:
        beta_v = float(para.beta)
    elif beta is not None:
        beta_v = float(beta)
    else:
        beta_v = float(P._auto_tune_beta(mesh, para, mu, U1, F1, mark_hole_strain=mhs))
    mesh2 = mesh
    if len(mhs) > 0:
        mhs_set = set(mhs.tolist())
        trimmed = mesh.elements_fem.copy()
        for e in range(trimmed.shape[0]):
            for j in range(trimmed.shape[1]):
                if trimmed[e, j] >= 0 and trimmed[e, j] in mhs_set:
                    trimmed[e, :] = -1
                    break
        mesh2 = M["ds"].DICMesh(coordinates_fem=mesh.coordinates_fem, elements_fem=trimmed,
                                mark_coord_hole_edge=mesh.mark_coord_hole_edge)
    cache = M["sub2"].precompute_subpb2(mesh2, para.gauss_pt_order, beta_v, mu, alpha)
    U2 = M["sub2"].subpb2_solver(mesh, para.gauss_pt_order, beta_v, mu, U1, F1, grad_dual,
                                 disp_dual, alpha, para.winstepsize, precomputed=cache)
    F2 = M["nsfem"].global_nodal_strain_fem(mesh2, para, U2)
    U2, F2 = P._apply_post_solve_corrections(U2, F2, U1, F1, mesh, para, node_region_map, mhs)
    grad_dual = F2 - F1
    disp_dual = U2 - U1
    tol2 = para.admm_tol
    U2_prev, U1_prev = U2.copy(), U1.copy()
    n_steps = 1
    upd_g: List[float] = []
    upd_l: List[float] = []
    para_w = replace(para, winsize_list=np.full((n, 2), para.winsize, dtype=np.float64))
    pre1 = M["sub1"].precompute_subpb1(mesh.coordinates_fem, ref.Df, ref.f_img_raw, para_w)
    for step in range(2, int(para.admm_max_iter) + 1):
        n_steps = step
        U1, _t1, _conv, _bad = M["sub1"].subpb1_solver(
            U2, F2, disp_dual, grad_dual, mesh.coordinates_fem, ref.Df, ref.f_img_raw, g,
            mu, beta_v, para_w, para.tol, precomputed=pre1)
        F1 = F2.copy()
        U2 = M["sub2"].subpb2_solver(mesh, para.gauss_pt_order, beta_v, mu, U1, F1, grad_dual,
                                     disp_dual, alpha, para.winstepsize, precomputed=cache)
        F2 = M["nsfem"].global_nodal_strain_fem(mesh2, para_w, U2)
        U2, F2 = P._apply_post_solve_corrections(U2, F2, U1, F1, mesh, para_w, node_region_map,
                                                 mhs)
        du_g = float(np.linalg.norm(U2_prev - U2) / np.sqrt(len(U2)))
        du_l = float(np.linalg.norm(U1_prev - U1) / np.sqrt(len(U1)))
        upd_g.append(du_g)
        upd_l.append(du_l)
        grad_dual = F2 - F1
        disp_dual = U2 - U1
        U2_prev, U1_prev = U2.copy(), U1.copy()
        if du_g < tol2 or du_l < tol2:
            break
    return {"U": np.asarray(U2, dtype=np.float64), "F": np.asarray(F2, dtype=np.float64),
            "beta": beta_v, "n_steps": int(n_steps), "update_global": tuple(upd_g),
            "update_local": tuple(upd_l)}


def dic_stage_strain(ref: DICReference, mesh, U: np.ndarray, *,
                     deformed_mask: Optional[np.ndarray] = None):
    """Section 8 for one frame (`pipeline.py:1746-1827`): the mesh's elements under the
    deformed mask (``crack_mask_from_deformed``), the region map, three smoothing passes
    when ``para.smoothness`` asks, then pyALDIC's ``compute_strain`` (plane fit or FEM,
    the strain measure from ``para``) → its ``StrainResult``."""
    M = _al_dic_modules()
    from dataclasses import replace
    P = M["pipeline"]
    para = ref.para
    coords = mesh.coordinates_fem
    elems = mesh.elements_fem
    U = np.asarray(U, dtype=np.float64)
    fallback = ref.f_mask
    mask_i, elems_i = fallback, elems
    if deformed_mask is not None and U.size == 2 * coords.shape[0]:
        mk, el = M["rasterize"].crack_mask_from_deformed(
            coords, elems, U, np.asarray(deformed_mask, dtype=np.float64), ref.shape)
        if np.any(mk):
            mask_i, elems_i = mk, el
    regions = M["regions"].precompute_node_regions(coords, mask_i, ref.shape)
    smesh = M["ds"].DICMesh(coordinates_fem=coords, elements_fem=elems_i,
                            mark_coord_hole_edge=mesh.mark_coord_hole_edge)
    para_s8 = replace(para, img_ref_mask=mask_i)
    neighbors = None
    if int(para.method_to_compute_strain) == 2 and coords.shape[0] > 0:
        neighbors = M["platefit"].build_neighbor_cache(coords, para.strain_plane_fit_rad)
    U_local = U.copy()
    if para.smoothness > 0:
        for _ in range(3):
            U_local = P._smooth_disp(U_local, coords, para, regions, mesh=smesh)
    return M["strain"].compute_strain(smesh, para_s8, U_local, regions, neighbors=neighbors)


def dic_split_u(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interleaved ``(2N,)`` → ``(u_x, v_y)`` per node (x along columns, y along rows)."""
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    return U[0::2].copy(), U[1::2].copy()


def dic_merge_u(u_x: np.ndarray, v_y: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(u_x), dtype=np.float64)
    out[0::2] = u_x
    out[1::2] = v_y
    return out


def dic_split_f(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interleaved ``(4N,)`` → ``(dudx, dvdx, dudy, dvdy)`` per node — pyALDIC's order."""
    F = np.asarray(F, dtype=np.float64).reshape(-1)
    return F[0::4].copy(), F[1::4].copy(), F[2::4].copy(), F[3::4].copy()


def dic_merge_f(dudx, dvdx, dudy, dvdy) -> np.ndarray:
    out = np.empty(4 * len(dudx), dtype=np.float64)
    out[0::4], out[1::4], out[2::4], out[3::4] = dudx, dvdx, dudy, dvdy
    return out


def dic_stage_lattice_nodes(para, shape: Tuple[int, int]) -> np.ndarray:
    """The mesh NODES ``(N, 2)`` as ``[x, y]`` in pyALDIC's node order (``node = ix·ny + iy``)
    for ``para`` on an image of ``shape`` — ``mesh_setup`` on the lattice, no mask, no
    frame read. The row order every staged DIC table keeps; trimming by a mask drops
    elements only, never nodes, so this order holds through the whole chain."""
    M = _al_dic_modules()
    x0, y0 = dic_stage_grid(para, shape)
    mesh = M["mesh_setup"].mesh_setup(x0, y0, para)
    return np.asarray(mesh.coordinates_fem, dtype=np.float64).copy()
