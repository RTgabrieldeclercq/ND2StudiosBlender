"""aldvc_field — official pyALDVC (``al-dvc``) 3D DVC kernel + result adapter.

PURPOSE
    Self-contained, portable adapter that runs the EXTERNAL ``al-dvc`` (pyALDVC)
    Augmented-Lagrangian Digital Volume Correlation solver over an ordered stack
    of volumes and packages each frame pair as a :class:`DVCResult` on a regular
    subset grid. Given already-prepared ``(Z, Y, X)`` volumes it returns dense
    displacement (voxels) + strain on the solver's own node grid.

WHERE THE REAL MATH LIVES
    100% inside the EXTERNAL ``al-dvc`` package (``al_dvc.core.pipeline.run_aldvc``),
    imported LAZILY. pyALDVC is the official Python AL-DVC from the author of
    pyALDIC — https://github.com/zachtong/pyALDVC, docs at
    https://zachtong.github.io/pyALDVC/, PyPI ``al-dvc``, DOI 10.5281/zenodo.22883767.
    It implements MATLAB ``main_ALDVC.m`` Sections 2–8: node grid, pyramid-NCC
    initial guess, 12-DOF local IC-GN (subproblem 1), the global compatibility
    solve (subproblem 2) with L-curve β auto-tuning, the ADMM outer loop, and
    strain. The IN-REPO math here is ONLY the adapters around it:

      * input:  node params → an ``al_dvc`` ``DVCPara`` with the ``(z,y,x)``→``(x,y,z)``
                reversal and even/≥4 subset snapping (:func:`build_dvcpara`);
                a lazy ``VolumeProvider`` over a caller-supplied frame getter
                (:class:`LazyVolumeProvider`).
      * output: the LOAD-BEARING ``[x,y,z]``→``[z,y,x]`` axis reversal of node
                coordinates, displacement and the strain tensor, reshaped from
                pyALDVC's flat ``(N, …)`` node arrays onto the ``(Gz,Gy,Gx, …)``
                grid this repo's Point flattener expects
                (:func:`frame_to_dvcresult`).

HISTORY — WHAT THIS REPLACED (2026-09-25)
    Until now this module was a 2998-line clean-room numpy/scipy port of
    FranckLab's MATLAB ALDVC, written inside ND2Studios and vendored here. It was
    replaced WHOLESALE by the upstream package: pyALDVC is the maintained
    implementation by the pyALDIC author, it is numba/CUDA accelerated, and it
    carries its own validation against the MATLAB reference — none of which the
    in-repo port could keep up with. The port's dimension-agnostic leftovers that
    other nodes depend on (the ``DVCResult`` contract, the subset ``Grid``, the
    strain-measure stack and Lagrangian accumulation) moved to
    :mod:`nodegraph.kernels.field_math`; nothing else survived.

    **This kernel is 3D-only**, because pyALDVC is: ``winsize`` must be an even
    triple ≥ 4 per axis and masks must be 3-D, so a single-plane volume cannot be
    correlated at all. 2D correlation is :mod:`nodegraph.kernels.dic_correlate`
    (the ``al-dic`` / pyALDIC sibling package), exposed as ``analysis.dic_correlate``.

CALLER OWNS ALL PREP
    No file I/O, no per-multipoint/per-timepoint looping beyond the ordered stack
    you hand it, no crop/downsample/registration/exclusion. You prepare the
    ``(Z, Y, X)`` arrays; the kernel normalizes them the way pyALDVC does
    (``normalize_volume`` — z-score over the VOI) and nothing more.

LOAD-BEARING CONVENTIONS — READ BEFORE USE
    pyALDVC and this repo order their axes OPPOSITELY, and every adapter below
    exists to bridge that:

    =======================  ==========================  ==========================
    quantity                 pyALDVC native              what this kernel returns
    =======================  ==========================  ==========================
    volume array             ``(nz, ny, nx)``            same — no change
    ``DVCPara`` triples      ``(x, y, z)``               caller passes ``(z, y, x)``
    node coordinates         ``(N, 3)`` ``[x, y, z]``    ``(Gz,Gy,Gx,3)`` ``[z,y,x]``
    displacement ``U``       ``(N, 3)`` ``[u, v, w]``    ``(Gz,Gy,Gx,3)`` ``[dz,dy,dx]``
    gradient ``F[i,j]``      ``du_i/dx_j``, ``i,j∈xyz``  strain ``[i,j]``, ``i,j∈zyx``
    =======================  ==========================  ==========================

    The reversal is a full ``[::-1]`` on the component axis — and on BOTH tensor
    axes for strain. Reversing one and not the other transposes the strain tensor,
    which is invisible on any symmetric fixture.

VERIFIED SIGN CONVENTION (2026-09-25) — pyALDVC is NOT pyALDIC here
    ``dic_correlate`` has to NEGATE pyALDIC's two strain cross-terms (that kernel's
    docstring records the measurement). **pyALDVC does not need this.** Probed
    against analytic truth on a 72³ bead volume, three fixtures:

      * pure translation ``t=(+2,+1,-1.5)`` in ``(x,y,z)`` → ``U`` median
        ``[+1.9999, +1.0000, -1.5001]``. Sign and component order match directly.
      * simple shear ``du/dy=+0.02``, all else 0 → ``F[0,1]=+0.0202``,
        ``F[1,0]=+1e-5``. Correct term, correct sign, no transpose.
      * ANTISYMMETRIC ``du/dy=+0.02, dv/dx=-0.02`` → ``F[0,1]=+0.0202``,
        ``F[1,0]=-0.0202``. This is the fixture that would expose a negation or a
        transpose, and it exposes neither.

    ``StrainResult``'s ``exy/exz/eyz`` are TENSOR shear (½ the engineering shear):
    the ``du/dy=+0.02`` fixture reports ``exy=+0.0101``. That matches
    ``field_math.strain_from_gradient``'s ``½(G+Gᵀ)``, so the two are interchangeable
    and the Point columns mean the same thing they always did.

UNITS
    ``displacement_field`` is in VOXELS (pyALDVC's ``FrameResult.U``, which stays in
    voxels regardless of ``voxel_size``). ``voxel_size_um`` is recorded on the result
    but NOT applied — the catalog's ``_shared.dvc._dvc_rows`` does that multiply.
    STRAIN, however, is taken from ``StrainResult``, which pyALDVC computes in
    PHYSICAL units (``scale_to_physical(U, F, para.voxel_size)``). That is deliberate
    and load-bearing for anisotropic voxels: a confocal z-step of 0.5 µm against a
    0.1 µm xy pixel makes every off-diagonal strain term wrong by the 5× anisotropy
    ratio unless the solver is told the voxel size. So ``voxel_size_um`` IS passed
    through to ``DVCPara.voxel_size``, and strain comes back dimensionless-correct.

DEPENDENCY
    ``al-dvc`` is OPTIONAL and lazily imported, exactly like ``al-dic`` in
    ``dic_correlate``: importing this module never imports it, so the catalog loads
    on a machine without it and only a pull fails, with :data:`INSTALL_HINT`.
    Install: ``pip install al-dvc`` (or ``al-dvc[gpu]`` for the CUDA local solver).
    Verified against al-dvc 1.2.0 / numpy 2.4.6 / scipy 1.18.0 / numba 0.66.0.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── the optional dependency ───────────────────────────────────────────────────

INSTALL_HINT = (
    "The 3D DVC node needs the optional 'al-dvc' package (pyALDVC).\n"
    "Install it with:  pip install al-dvc\n"
    "  (or 'pip install al-dvc[gpu]' for the CUDA-accelerated local solver)\n"
    "Project: https://github.com/zachtong/pyALDVC"
)


def al_dvc_available() -> bool:
    """True if the optional ``al_dvc`` package is importable."""
    return importlib.util.find_spec("al_dvc") is not None


def _require_al_dvc():
    """Import and return ``(config, pipeline, volume_ops)``, or raise a friendly error."""
    if not al_dvc_available():
        raise RuntimeError(INSTALL_HINT)
    return (
        importlib.import_module("al_dvc.core.config"),
        importlib.import_module("al_dvc.core.pipeline"),
        importlib.import_module("al_dvc.io.volume_ops"),
    )


# ── DVCResult — the repo-wide correlation output contract ─────────────────────
# Kept in-file (not imported from field_math) so this kernel stays byte-portable,
# matching `dic_correlate` and `piv_field`, which each carry their own copy. The
# four copies are deliberate: a kernel that imports another kernel cannot be lifted
# out on its own, which is the whole point of `kernels/README.md`'s doctrine.

@dataclass
class DVCResult:
    """Structured output of one correlated frame pair.

    Shapes (``d`` = 3 — this kernel is 3D-only):

    - ``grid_coords``: ``(Gz, Gy, Gx, 3)`` subset-center coordinates **in voxels**,
      ``[..., 0]`` the SLOWEST spatial axis (z).
    - ``displacement_field``: ``(Gz, Gy, Gx, 3)`` displacement **in voxels**, same
      axis order as ``grid_coords`` (``[..., 0]`` is dz).
    - ``strain_field``: ``(Gz, Gy, Gx, 3, 3)`` symmetric strain tensor
      ``[i, j]`` over axes ``(z, y, x)``, or ``None``.
    - ``qfactor``: ``(Gz, Gy, Gx)`` per-subset ZNCC.

    ``voxel_size_um`` is ``(z, y, x)``; :meth:`displacement_um` applies it.
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


# ── param mapping: node dict → al_dvc DVCPara ─────────────────────────────────

def _snap_even(v: int, lo: int = 4) -> int:
    """Nearest even integer ``>= lo``. ``DVCPara`` validation REQUIRES this of every
    ``winsize`` component (``config.validate_dvcpara``: "must be an even integer >= 4"),
    and raises rather than rounding, so the snap has to happen on this side."""
    n = int(round(float(v)))
    if n % 2:
        n += 1
    return max(int(lo), n)


def _triple_zyx_to_xyz(lateral: int, axial: Optional[int], *, even: bool,
                       lo: int = 1) -> Tuple[int, int, int]:
    """``(x, y, z)`` triple for ``DVCPara`` from a lateral value + optional axial one.

    The node speaks ``(z, y, x)`` slowest-first and offers one lateral socket plus a
    ``_z`` companion; ``DVCPara`` wants ``(x, y, z)``. ``axial`` of ``None``/``0``
    means "same as lateral" (the isotropic case).
    """
    lat = _snap_even(lateral, lo) if even else max(int(lo), int(round(float(lateral))))
    if axial is None or int(axial) <= 0:
        ax = lat
    else:
        ax = _snap_even(axial, lo) if even else max(int(lo), int(round(float(axial))))
    return (lat, lat, ax)


# Node mode value → the string `DVCPara` actually validates. The node keeps the
# hyphenated spellings the old kernel used for strain so saved graphs still load;
# pyALDVC validates underscored ones.
_STRAIN_TYPE_ALIASES = {
    "infinitesimal": "infinitesimal", "small": "infinitesimal",
    "engineering": "infinitesimal",
    "green-lagrange": "green_lagrange", "green_lagrange": "green_lagrange",
    "green": "green_lagrange", "lagrange": "green_lagrange",
    "almansi": "euler_almansi", "euler-almansi": "euler_almansi",
    "euler_almansi": "euler_almansi", "eulerian-almansi": "euler_almansi",
    "hencky": "hencky", "log": "hencky", "logarithmic": "hencky",
}


def normalize_strain_type(name: str) -> str:
    """Map this repo's strain-measure spelling onto ``DVCPara.strain_type``."""
    key = str(name or "infinitesimal").strip().lower().replace(" ", "_")
    try:
        return _STRAIN_TYPE_ALIASES[key]
    except KeyError:
        try:
            return _STRAIN_TYPE_ALIASES[key.replace("_", "-")]
        except KeyError:
            raise ValueError(
                f"unknown strain_type {name!r}; expected one of "
                "infinitesimal / green_lagrange / euler_almansi / hencky") from None


def build_dvcpara(params: Dict[str, Any], voxel_size_um: Sequence[float],
                  *, n_frames: int = 2, ref_indices: Optional[Sequence[int]] = None):
    """Map node params → a validated ``al_dvc`` ``DVCPara``.

    ``voxel_size_um`` is ``(z, y, x)`` slowest-first (this repo's order); it is
    REVERSED to ``DVCPara``'s ``(x, y, z)``. Every key is optional; the fallback is
    pyALDVC's own default, so an empty dict reproduces ``dvcpara_default()``.

    ``ref_indices`` (per deformed frame, ``0 <= ref[i] <= i``) becomes a
    ``FrameSchedule``; ``None`` leaves ``reference_mode`` to decide.
    """
    config, _pipeline, _vops = _require_al_dvc()
    g = params.get

    vox = np.asarray(voxel_size_um, dtype=float).reshape(-1)
    if vox.size != 3:
        raise ValueError(
            f"voxel_size_um must have 3 components (z, y, x) for the 3D DVC kernel, "
            f"got {vox.size}")
    if not np.all(np.isfinite(vox)) or np.any(vox <= 0):
        raise ValueError(f"voxel_size_um must be positive and finite, got {tuple(vox)}")

    kw: Dict[str, Any] = {
        # (x, y, z) triples ------------------------------------------------------
        "voxel_size": tuple(float(v) for v in vox[::-1]),
        "units": "um",
        "winsize": _triple_zyx_to_xyz(int(g("subset_size", 32)),
                                      g("subset_size_z"), even=True, lo=4),
        "winstepsize": _triple_zyx_to_xyz(int(g("subset_spacing", 16)),
                                          g("subset_spacing_z"), even=False, lo=1),
        # initial guess ----------------------------------------------------------
        "init_guess_method": str(g("init_guess", "pyramid")),
        "global_shift": bool(g("global_shift", True)),
        "init_coarse_factor": max(1, min(8, int(g("init_coarse_factor", 1)))),
        "prefilter_sigma": max(0.0, float(g("prefilter_sigma", 0.0))),
        # local IC-GN ------------------------------------------------------------
        "interp_method": str(g("interp_method", "cubic")),
        "icgn_max_iter": max(1, int(g("icgn_max_iter", 100))),
        # ADMM -------------------------------------------------------------------
        "use_global_step": bool(g("use_global_step", True)),
        "mu": float(g("mu", 1e-3)),
        "admm_max_iter": max(1, int(g("admm_iterations", 4))),
        # smoothing --------------------------------------------------------------
        "disp_smoothing": max(0.0, float(g("disp_smoothing", 0.0))),
        "strain_smoothing": max(0.0, float(g("strain_smooth", 0.0))),
        # strain -----------------------------------------------------------------
        "strain_method": str(g("strain_method", "plane_fit")),
        "strain_type": normalize_strain_type(g("strain_type", "infinitesimal")),
        "strain_plane_fit_halfwidth": max(1, int(g("strain_halfwidth", 1))),
        # compute ----------------------------------------------------------------
        "backend": str(g("backend", "auto")),
        "n_threads": max(0, int(g("n_threads", 0))),
        "tile_local": max(0, int(g("tile_local", 0))),
        "verbose": False,
    }

    # `search_radius` 0 means "pyALDVC's own default" rather than "search nothing":
    # a 0 radius would make the NCC seed a single-point lookup. The node's socket
    # documents 0 as "leave it to the solver"; upstream's default is 8.
    sr = int(g("search_radius", 0))
    if sr > 0:
        kw["search_radius"] = (sr, sr, sr)

    # `subset_stride` must leave >= 5 samples per axis or DVCPara refuses; clamp
    # rather than raise, so a large stride with a small subset degrades gracefully.
    stride = max(1, int(g("subset_stride", 1)))
    kw["subset_stride"] = min(stride, max(1, min(kw["winsize"]) // 4))

    # `beta = 0` is this node's spelling of "auto-tune by L-curve", which upstream
    # spells `None`. A negative or zero beta is refused by DVCPara, so it cannot be
    # passed through literally.
    beta = float(g("beta", 0.0))
    kw["beta"] = beta if beta > 0 else None

    if ref_indices is not None:
        kw["frame_schedule"] = config.FrameSchedule(
            ref_indices=tuple(int(i) for i in ref_indices))
    else:
        kw["reference_mode"] = str(g("reference_mode", "accumulative"))

    return config.dvcpara_default(**kw)


# ── a lazy VolumeProvider over a caller-supplied frame getter ─────────────────

class LazyVolumeProvider:
    """``al_dvc`` ``VolumeProvider`` over a lazily-indexed volume sequence.

    pyALDVC's own ``ListVolumeProvider`` wants every volume materialized up front.
    This repo's frames come from a lazy image provider, and a 49-position confocal
    series would hold the whole stack in RAM for no reason, so this hands pyALDVC
    one frame at a time and caches the normalized float32 copies in a small LRU —
    the same shape as ``ListVolumeProvider`` (default 3: the reference, the current
    deformed frame and the previous one).

    ``get_volume(i)`` must return a ``(nz, ny, nx)`` array of the SAME shape for
    every ``i``. Normalization is pyALDVC's own ``normalize_volume`` —
    ``(v - mean_voi) / std_voi`` — so the numbers the solver sees are identical to
    what it would compute from a list.

    Masks are not supported here (``get_mask`` returns ``None`` and ``has_masks``
    is False): this repo's DVC node correlates whole volumes, and pyALDVC's mask
    path additionally drives ``subset_split`` and the VOI, which the node does not
    expose. Wire a mask by cropping before you call.
    """

    def __init__(self, get_volume: Callable[[int], np.ndarray], n_frames: int,
                 shape: Tuple[int, int, int], *, voi=None, cache_size: int = 3):
        if int(n_frames) < 2:
            raise ValueError(f"at least 2 volumes are required, got {n_frames}")
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError(f"volumes must be 3-D, got shape {shape}")
        _config, _pipeline, self._vops = _require_al_dvc()
        from al_dvc.core.data_structures import VOIRange
        self._get = get_volume
        self._n = int(n_frames)
        self._shape: Tuple[int, int, int] = shape           # type: ignore[assignment]
        self._voi = (voi or VOIRange()).clamp(self._shape)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    def __len__(self) -> int:
        return self._n

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def clamped_voi(self):
        return self._voi

    def get_normalized(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        vol = np.asarray(self._get(int(idx)))
        if tuple(vol.shape) != self._shape:
            raise ValueError(
                f"volume {idx} has shape {tuple(vol.shape)}, expected {self._shape}; "
                "the reference and every deformed volume must match in Z/Y/X")
        out = self._vops.normalize_volume(vol, self._voi)
        self._cache[idx] = out
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def get_mask(self, idx: int):
        return None

    @property
    def has_masks(self) -> bool:
        return False


# ── output adapter: pyALDVC node arrays → DVCResult ───────────────────────────

def _strain_tensor_zyx(strain) -> np.ndarray:
    """``(N, 3, 3)`` symmetric strain in ``(z, y, x)`` index order from a ``StrainResult``.

    pyALDVC names its components in ``(x, y, z)``; this repo indexes ``(z, y, x)``.
    Reversing BOTH tensor axes maps one onto the other, which for a symmetric tensor
    is exactly the relabelling below — written out rather than done with ``[::-1,::-1]``
    so the mapping is readable and testable:

        S_zyx = [[ezz, eyz, exz],
                 [eyz, eyy, exy],
                 [exz, exy, exx]]
    """
    exx = np.asarray(strain.exx, dtype=np.float64)
    eyy = np.asarray(strain.eyy, dtype=np.float64)
    ezz = np.asarray(strain.ezz, dtype=np.float64)
    exy = np.asarray(strain.exy, dtype=np.float64)
    exz = np.asarray(strain.exz, dtype=np.float64)
    eyz = np.asarray(strain.eyz, dtype=np.float64)
    n = exx.shape[0]
    out = np.empty((n, 3, 3), dtype=np.float64)
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = ezz, eyz, exz
    out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = eyz, eyy, exy
    out[:, 2, 0], out[:, 2, 1], out[:, 2, 2] = exz, exy, exx
    return out


def frame_to_dvcresult(mesh, frame, strain, voxel_size_um: Sequence[float], *,
                       strain_type: str = "", cumulative: bool = False,
                       notes: str = "") -> DVCResult:
    """Convert one pyALDVC frame result (mesh + ``FrameResult`` + ``StrainResult``)
    into a :class:`DVCResult` on the ``(Gz, Gy, Gx)`` grid.

    THE AXIS REVERSAL LIVES HERE. ``mesh.coordinates`` is ``(N, 3)`` ``[x, y, z]``
    and ``frame.U`` is ``(N, 3)`` ``[u, v, w]``; both get their component axis
    reversed to ``[z, y, x]`` / ``[dz, dy, dx]``. ``mesh.grid_shape`` is already
    ``(nz, ny, nx)`` node counts and node ``n = iz*ny*nx + iy*nx + ix``, so a plain
    C-order reshape lands on the right grid — no transpose needed, only the
    component reversal.

    ``cumulative`` selects ``frame.U_accum`` (displacement from frame 0, which the
    pipeline fills for an incremental schedule) over ``frame.U`` (this pair only).
    """
    gshape = tuple(int(s) for s in mesh.grid_shape)          # (nz, ny, nx) nodes

    coords = np.asarray(mesh.coordinates, dtype=np.float64)[:, ::-1]   # → [z,y,x]
    grid_coords = coords.reshape(*gshape, 3)

    U = frame.U_accum if (cumulative and frame.U_accum is not None) else frame.U
    disp = np.asarray(U, dtype=np.float64)[:, ::-1]                    # → [dz,dy,dx]
    displacement = disp.reshape(*gshape, 3)

    strain_field = None
    if strain is not None:
        strain_field = _strain_tensor_zyx(strain).reshape(*gshape, 3, 3)

    qfactor = None
    if frame.zncc is not None:
        qfactor = np.asarray(frame.zncc, dtype=np.float64).reshape(*gshape)

    # `converged` is a whole-frame verdict, so it is the per-node status collapsed:
    # STATUS_CONVERGED == 0 for every node of the final local pass. A frame with any
    # non-converged node is reported unconverged rather than silently averaged.
    converged = False
    if frame.status is not None:
        converged = bool(np.all(np.asarray(frame.status) == 0))

    admm = frame.admm
    mu = float(admm.mu) if admm is not None else 0.0
    beta = float(admm.beta) if admm is not None else 0.0
    iterations = int(admm.n_steps) if admm is not None else 0

    diagnostics: Dict[str, Any] = {
        "engine": "pyALDVC (al-dvc)",
        "ref_frame": int(frame.ref_frame),
        "n_nodes": int(mesh.n_nodes),
        "grid_shape": gshape,
    }
    if frame.status is not None:
        st = np.asarray(frame.status)
        diagnostics["n_converged"] = int(np.sum(st == 0))
        diagnostics["n_nodes_bad"] = int(np.sum(st != 0))
    if frame.outlier is not None:
        diagnostics["n_outlier"] = int(np.sum(np.asarray(frame.outlier)))
    if qfactor is not None:
        diagnostics["median_zncc"] = float(np.nanmedian(qfactor))

    return DVCResult(
        dim=3,
        grid_coords=grid_coords,
        displacement_field=displacement,
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        strain_field=strain_field,
        strain_type=str(strain_type or (strain.strain_type if strain is not None else "")),
        qfactor=qfactor,
        converged=converged,
        iterations=iterations,
        mu=mu,
        beta=beta,
        method="pyALDVC (al-dvc)" + (" cumulative" if cumulative else ""),
        notes=notes,
        diagnostics=diagnostics,
    )


# ── entry points ──────────────────────────────────────────────────────────────

def run_aldvc_series(
    get_volume: Callable[[int], np.ndarray],
    n_frames: int,
    shape: Tuple[int, int, int],
    *,
    voxel_size_um: Sequence[float],
    params: Optional[Dict[str, Any]] = None,
    ref_indices: Optional[Sequence[int]] = None,
    compute_strain: bool = True,
    cumulative: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    stop_cb: Optional[Callable[[], bool]] = None,
) -> List[DVCResult]:
    """Correlate an ordered stack in ONE ``al_dvc.run_aldvc`` call → one
    :class:`DVCResult` per deformed frame (``n_frames - 1`` of them, for frames
    ``1 … n_frames-1``).

    One call rather than N pair calls is the point: pyALDVC builds the node grid
    once, caches each reference frame's bundle (normalized volume + its three
    gradient volumes) across every frame that references it, and auto-tunes β once
    per reference rather than once per pair. On an accumulative schedule — every
    frame against frame 0 — that is a single reference bundle for the whole series.

    ``get_volume(i)`` returns the ``i``-th ``(nz, ny, nx)`` volume; frame 0 is the
    first reference. ``ref_indices[i]`` is the reference for deformed frame ``i+1``
    and must satisfy ``0 <= ref_indices[i] <= i``; ``None`` falls back to
    ``params["reference_mode"]`` (``accumulative`` / ``incremental``).

    ``progress_cb(fraction, message)`` is pyALDVC's own callback, forwarded as-is.
    """
    _config, pipeline, _vops = _require_al_dvc()
    params = dict(params or {})

    para = build_dvcpara(params, voxel_size_um, n_frames=n_frames,
                         ref_indices=ref_indices)
    provider = LazyVolumeProvider(get_volume, n_frames, shape)

    res = pipeline.run_aldvc(
        para, provider,
        progress_fn=progress_cb,
        stop_fn=stop_cb,
        compute_strain=bool(compute_strain),
        checkpoint_dir=None,
        resume=False,
    )

    strain_type = str(para.strain_type)
    out: List[DVCResult] = []
    for k, frame in enumerate(res.result_disp):
        strain = res.result_strain[k] if (compute_strain and k < len(res.result_strain)) else None
        out.append(frame_to_dvcresult(
            res.dvc_mesh, frame, strain, voxel_size_um,
            strain_type=strain_type, cumulative=cumulative,
            notes=(res.stop_reason or "")))
    return out


def run_aldvc(
    ref_vol: np.ndarray,
    def_vol: np.ndarray,
    *,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    params: Optional[Dict[str, Any]] = None,
    compute_strain: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> DVCResult:
    """Correlate ONE reference/deformed ``(Z, Y, X)`` volume pair → a :class:`DVCResult`.

    The convenience entry point; :func:`run_aldvc_series` is what the node uses and
    what you want for a series (it reuses the reference bundle across frames).

    ``voxel_size_um`` is ``(z, y, x)`` slowest-first. ``params`` keys are the node's
    own spelling — see :func:`build_dvcpara`; an empty dict gives pyALDVC's defaults
    (subset 32, spacing 16, pyramid seed, 4 ADMM iterations, auto β).
    """
    ref = np.asarray(ref_vol)
    dfm = np.asarray(def_vol)
    if ref.ndim != 3 or dfm.ndim != 3:
        raise ValueError(
            f"pyALDVC is 3D-only: got reference {ref.ndim}D and deformed {dfm.ndim}D. "
            "For 2D correlation use nodegraph.kernels.dic_correlate (pyALDIC).")
    if ref.shape != dfm.shape:
        raise ValueError(
            f"reference shape {ref.shape} != deformed {dfm.shape}; both volumes must "
            "match in Z/Y/X")

    vols = (ref, dfm)
    out = run_aldvc_series(
        lambda i: vols[i], 2, tuple(int(s) for s in ref.shape),
        voxel_size_um=voxel_size_um, params=params,
        compute_strain=compute_strain, progress_cb=progress_cb)
    return out[0]


# ── staged entry points (2026-10-08): pyALDVC's sections as separate calls ─────
# `run_aldvc` is Sections 2-8 of the ALDVC paper in ONE function (al-dvc 1.2.0
# `core/pipeline.py:124-390`). The staged DVC nodes — `dvc.grid` → `dvc.initial_guess` →
# `dvc.local_match` → `dvc.global_step` → `dvc.strain` — call the functions below one at
# a time so a user can pull, inspect and re-tune each section on its own. Each function
# reproduces its slice of that driver verbatim (the same calls in the same order, the
# same `para` rewrites, the same `mesh.node_valid` side effect), so the staged chain and
# `run_aldvc_series` agree on every node. Everything crosses this boundary in THIS
# repo's (z, y, x) order; the reversal to pyALDVC's (x, y, z) happens here and only here
# (see §6a of the contract — one and only one adapter).
#
# What a stage needs from the volumes is rebuilt from the volumes: a reference "context"
# (normalised volume, its gradients, the per-node Hessians, the global operators) is NOT
# handed from node to node — it is a few seconds to rebuild and would not survive the
# memo, the spill or a saved file. The displacement field between stages is the Point
# table the nodes carry, in voxels here (the nodes keep it in µm on the wire).


def _al_dvc_modules() -> Dict[str, Any]:
    """Every pyALDVC module a staged call touches, imported lazily and by path — the
    `al_dvc.solver` package re-exports the *functions* `local_icgn` / `subpb1_solver`
    under the names of their modules, so attribute access on the package is ambiguous."""
    if not al_dvc_available():
        raise RuntimeError(INSTALL_HINT)
    m = importlib.import_module
    return {
        "config": m("al_dvc.core.config"),
        "ds": m("al_dvc.core.data_structures"),
        "vops": m("al_dvc.io.volume_ops"),
        "grid": m("al_dvc.mesh.grid_mesh"),
        "tiling": m("al_dvc.solver.tiling"),
        "local": m("al_dvc.solver.local_icgn"),
        "gops": m("al_dvc.solver.global_operators"),
        "init": m("al_dvc.solver.init_disp"),
        "coarse": m("al_dvc.solver.coarse_init"),
        "beta": m("al_dvc.solver.beta_tuning"),
        "sub1": m("al_dvc.solver.subpb1_solver"),
        "sub2": m("al_dvc.solver.subpb2_solver"),
        "unc": m("al_dvc.solver.uncertainty"),
        "strain": m("al_dvc.strain.compute_strain"),
        "valid": m("al_dvc.utils.validation"),
        "numba": m("al_dvc._numba_compat"),
    }


@dataclass
class DVCReference:
    """Everything the local and global stages need about ONE reference volume on ONE
    grid — pyALDVC's `get_reference` cache entry (`pipeline.py:190-222`) as a value.

    ``mesh`` is the grid AFTER `precompute_local_context` wrote its validity into it
    (`mesh.node_valid = ctx.valid`), which is the mesh every later section reads.
    ``ops`` is ``None`` when the global step is off or the operators could not be
    built (pyALDVC logs and disables the global step in that case; so do we)."""
    para: Any
    shape: Tuple[int, int, int]
    mesh: Any
    source: Any
    ctx: Any
    ops: Any

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.n_nodes)

    @property
    def valid(self) -> np.ndarray:
        return np.asarray(self.mesh.node_valid, dtype=bool)


def stage_para(params: Dict[str, Any], voxel_size_um: Sequence[float],
               shape: Tuple[int, int, int]):
    """Section 2's parameter rewrite: the node's params → a validated ``DVCPara`` with
    the VOI clamped to ``shape``, ``volume_shape`` set and ``gradient_mode="auto"``
    resolved — exactly what `run_aldvc` does at `pipeline.py:124-151` before anything
    else runs. Raises pyALDVC's own ``ValueError`` for a volume too small for the
    subset (every axis ≥ 16, ``winsize + 11 ≤ extent``)."""
    M = _al_dvc_modules()
    from dataclasses import replace
    shape = tuple(int(s) for s in shape)
    para = build_dvcpara(dict(params or {}), voxel_size_um)
    M["valid"].validate_para_against_volume(para, shape)
    para = replace(para, voi=M["ds"].VOIRange().clamp(shape), volume_shape=shape)
    resolved = M["vops"].resolve_gradient_mode(para.gradient_mode, shape)
    if resolved != para.gradient_mode:
        para = replace(para, gradient_mode=resolved)
    return para


def stage_grid(para, shape: Tuple[int, int, int]) -> Tuple[Any, np.ndarray]:
    """Section 2: the node grid for ``shape`` under ``para`` →
    ``(mesh, coords_zyx)`` with ``coords_zyx`` the ``(N, 3)`` node centres in voxels,
    ``[z, y, x]``, in pyALDVC's node order ``n = iz·ny·nx + iy·nx + ix`` — the row
    order every stage table keeps. Pure geometry: no volume is read, so a grid node
    can show the measurement density before any correlation is paid for."""
    M = _al_dvc_modules()
    shape = tuple(int(s) for s in shape)
    x0, y0, z0 = M["grid"].build_grid_axes(para.voi, shape, para.winsize, para.winstepsize)
    mesh = M["grid"].mesh_setup(x0, y0, z0)
    coords = np.ascontiguousarray(np.asarray(mesh.coordinates, dtype=np.float64)[:, ::-1])
    return mesh, coords


def stage_reference(ref_vol: np.ndarray, para, shape: Tuple[int, int, int]) -> DVCReference:
    """Sections 2-3 for ONE reference: normalise + pre-smooth the volume, build its
    gradient bundle, the grid, the per-node local context (Hessians, validity) and —
    when the global step is on — the FEM/FD operators (`pipeline.py:190-222`)."""
    M = _al_dvc_modules()
    shape = tuple(int(s) for s in shape)
    vol = np.asarray(ref_vol)
    if tuple(vol.shape) != shape:
        raise ValueError(f"reference volume has shape {tuple(vol.shape)}, expected {shape}")
    f = M["vops"].presmooth_volume(M["vops"].normalize_volume(vol, para.voi),
                                   para.prefilter_sigma)
    source = M["tiling"].ReferenceSource(f=f, mask=None, gradient_mode=para.gradient_mode)
    if not int(getattr(para, "tile_local", 0) or 0):
        source.bundle_for(M["tiling"].whole_box_tile(source.shape))
    base_mesh, _coords = stage_grid(para, shape)
    mesh = M["grid"].apply_mask_to_mesh(base_mesh, None, para.winsize, para.min_valid_ratio,
                                        cut_bridging=bool(getattr(para, "subset_split", False)))
    if para.n_threads > 0:
        M["numba"].set_num_threads(para.n_threads)
    ctx = M["local"].precompute_local_context(mesh, source, para)
    mesh.node_valid = ctx.valid.copy()
    ops = None
    if para.use_global_step:
        try:
            ops = M["gops"].build_global_operators(mesh, para.subpb2_method, para.gauss_pt_order)
        except ValueError:
            ops = None
    return DVCReference(para=para, shape=shape, mesh=mesh, source=source, ctx=ctx, ops=ops)


def stage_deformed(def_vol: np.ndarray, para, shape: Tuple[int, int, int]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """One deformed volume → ``(g_norm, g_prep)``: the normalised, pre-smoothed volume
    the NCC seed searches, and the interpolation-ready copy (B-spline coefficients for
    ``bspline``) the IC-GN passes sample (`pipeline.py:255-262`, no mask)."""
    M = _al_dvc_modules()
    vol = np.asarray(def_vol)
    if tuple(vol.shape) != tuple(shape):
        raise ValueError(f"deformed volume has shape {tuple(vol.shape)}, expected {tuple(shape)}")
    g_norm = M["vops"].presmooth_volume(M["vops"].normalize_volume(vol, para.voi),
                                        para.prefilter_sigma)
    g_prep = M["vops"].prepare_deformed(g_norm, para.interp_method, mask=None)
    return g_norm, g_prep


def _u_to_xyz(u_zyx: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if u_zyx is None:
        return None
    return np.ascontiguousarray(np.asarray(u_zyx, dtype=np.float64)[:, ::-1])


def _u_to_zyx(u_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(u_xyz, dtype=np.float64)[:, ::-1])


def _f_to_xyz(f_zyx: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if f_zyx is None:
        return None
    return np.ascontiguousarray(np.asarray(f_zyx, dtype=np.float64)[:, ::-1, ::-1])


def _f_to_zyx(f_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(f_xyz, dtype=np.float64)[:, ::-1, ::-1])


def stage_initial_guess(ref: DVCReference, g_norm: np.ndarray, g_prep: np.ndarray, *,
                        previous_zyx: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Section 3 → ``(U0_zyx, F0_zyx | None, info)``: the integer-plus seed for every
    node (voxels, ``[dz, dy, dx]``), a coarse-lattice gradient seed when
    ``init_coarse_factor > 1`` (and no previous field), and pyALDVC's info dict
    (``method``, ``global_shift``, per-node ``cc`` / ``pce`` / ``ok`` where the method
    searched, ``n_bad``). ``previous_zyx`` is the last solved field against the SAME
    reference, which `run_aldvc` hands to the seed only in that case
    (`pipeline.py:266-272`)."""
    M = _al_dvc_modules()
    para = ref.para
    previous = _u_to_xyz(previous_zyx)
    F0 = None
    if int(para.init_coarse_factor) > 1 and previous is None:
        U0, F0, info = M["coarse"].coarse_initial_guess(ref.source, g_norm, g_prep, ref.mesh, para)
    else:
        U0, info = M["init"].compute_initial_guess(
            ref.source.f, g_norm, ref.mesh, para, previous=previous,
            split_fraction=ref.ctx.split_fraction)
    return _u_to_zyx(U0), (None if F0 is None else _f_to_zyx(F0)), dict(info)


def stage_local(ref: DVCReference, g_prep: np.ndarray, U0_zyx: np.ndarray, *,
                F0_zyx: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Section 4: the 12-DOF IC-GN on every subset, seeded by ``U0`` (and ``F0``) →
    ``{"U", "F", "zncc", "status", "n_iter", "bad", "U_std"}`` in ``(z, y, x)`` voxel
    order: ``U`` ``(N,3)``, ``F`` ``(N,3,3)`` with ``F[n,i,j] = ∂u_i/∂x_j``, the per-node
    ZNCC, pyALDVC's status code (0 converged … 7 stalled), the iteration count, the
    median-test rejections that were inpainted (`fill_bad_nodes` runs inside
    `local_icgn`), and the displacement uncertainty ``(N,3)`` (`pipeline.py:277-279`,
    `:377`)."""
    M = _al_dvc_modules()
    U1, F1, info, bad = M["local"].local_icgn(ref.ctx, ref.source, g_prep, _u_to_xyz(U0_zyx),
                                             ref.para, ref.mesh, F0=_f_to_xyz(F0_zyx))
    U_std = M["unc"].displacement_uncertainty(ref.ctx, info.zncc, info.status)
    return {
        "U": _u_to_zyx(U1), "F": _f_to_zyx(F1),
        "zncc": np.asarray(info.zncc, dtype=np.float64),
        "status": np.asarray(info.status, dtype=np.int64),
        "n_iter": np.asarray(info.n_iter, dtype=np.int64),
        "bad": np.asarray(bad, dtype=bool),
        "U_std": _u_to_zyx(U_std),
        "solve_time": float(info.solve_time),
    }


def stage_global(ref: DVCReference, g_prep: np.ndarray, U1_zyx: np.ndarray, F1_zyx: np.ndarray,
                 bad_local: np.ndarray, *, beta: Optional[float] = None) -> Dict[str, Any]:
    """Sections 5-6: the global compatibility step and the ADMM iterations
    (`pipeline.py:282-368`, verbatim) → ``{"U", "F", "zncc", "status", "outlier",
    "U_std", "beta", "mu", "n_steps", "update_global", "update_local"}`` in ``(z, y, x)``.

    ``beta`` ``None`` auto-tunes by the L-curve on THIS pair (``para.beta`` wins when it
    is set); the driver tunes once per reference and reuses it, so a caller correlating
    several frames against one reference should pass the first pair's ``beta`` back in.
    Refuses (``ValueError``) when the reference holds no global operators — the global
    step is off in ``para`` or the mesh could not build them."""
    M = _al_dvc_modules()
    para = ref.para
    if ref.ops is None:
        raise ValueError("the global step needs the FEM/FD operators: use_global_step is off "
                         "for this reference, or the grid could not build them")
    mesh, ctx, ops = ref.mesh, ref.ctx, ref.ops
    U1, F1 = _u_to_xyz(U1_zyx), _f_to_xyz(F1_zyx)
    bad_local = np.asarray(bad_local, dtype=bool)
    exclude = np.zeros(mesh.n_nodes, dtype=bool)
    exclude[mesh.boundary_nodes] = True
    exclude |= bad_local
    beta_sweep = None
    if para.beta is not None:
        beta_v = float(para.beta)
    elif beta is not None:
        beta_v = float(beta)
    else:
        beta_v, beta_sweep = M["beta"].auto_tune_beta(ops, para, para.mu, U1, F1, exclude)
    mu = float(para.mu)
    system = M["sub2"].build_global_system(ops, beta_v, mu, para.alpha, para)
    W = np.zeros((mesh.n_nodes, 3, 3))
    v = np.zeros((mesh.n_nodes, 3))
    U2, _s2 = M["sub2"].solve_subpb2(system, U1, F1, W, v)
    F2 = M["gops"].nodal_gradient(ops, U2)
    F2 = np.where(np.isfinite(F2), F2, F1)
    W = F2 - F1
    v = U2 - U1

    def _rms(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

    upd_g: List[float] = []
    upd_l: List[float] = []
    zncc = status = None
    n_steps = 1
    bad_last = bad_local
    for step in range(2, int(para.admm_max_iter) + 1):
        n_steps = step
        U1_prev = U1
        U1, info_s1, bad_last = M["sub1"].subpb1_solver(ctx, ref.source, g_prep, U2, F2, v, mu,
                                                        para, mesh)
        F1 = F2
        zncc, status = info_s1.zncc, info_s1.status
        U2_new, _ = M["sub2"].solve_subpb2(system, U1, F1, W, v)
        F2_new = M["gops"].nodal_gradient(ops, U2_new)
        F2_new = np.where(np.isfinite(F2_new), F2_new, F1)
        du_g = _rms(U2_new - U2)
        du_l = _rms(U1 - U1_prev)
        upd_g.append(du_g)
        upd_l.append(du_l)
        if para.dual_update == "accumulate":
            W = W + (F2_new - F1)
            v = v + (U2_new - U1)
        else:
            W = F2_new - F1
            v = U2_new - U1
        U2, F2 = U2_new, F2_new
        if du_g < para.admm_tol or du_l < para.admm_tol:
            break
    if zncc is None:
        # a single ADMM iteration: the status and quality are the local pass's, which the
        # caller holds — report them as unknown here rather than inventing a pass
        zncc = np.full(mesh.n_nodes, np.nan)
        status = np.full(mesh.n_nodes, -1, dtype=np.int64)
    status_arr = np.asarray(status, dtype=np.int64)
    outlier = np.asarray(bad_last, dtype=bool) & (status_arr == M["ds"].STATUS_CONVERGED)
    U_std = M["unc"].displacement_uncertainty(
        ctx, np.where(np.isfinite(zncc), zncc, 0.0), np.where(status_arr < 0, 0, status_arr))
    return {
        "U": _u_to_zyx(U2), "F": _f_to_zyx(F2),
        "zncc": np.asarray(zncc, dtype=np.float64), "status": status_arr,
        "outlier": outlier, "U_std": _u_to_zyx(U_std),
        "beta": beta_v, "mu": mu, "n_steps": int(n_steps),
        "update_global": tuple(upd_g), "update_local": tuple(upd_l),
        "beta_sweep": beta_sweep,
    }


def stage_strain(para, shape: Tuple[int, int, int], U_zyx: np.ndarray, *,
                 F_direct_zyx: Optional[np.ndarray] = None,
                 valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Section 8 on a grid rebuilt from ``para`` + ``shape`` (pure geometry — no volume
    is read) → ``(strain_zyx (N,3,3), strain_valid (N,), strain_type)``. ``U_zyx`` is the
    field to differentiate (voxels); ``F_direct_zyx`` the solver's own gradient, used by
    ``strain_method="direct"`` and only meaningful against frame 0 (`pipeline.py:410`);
    ``valid`` the node validity the local stage wrote. The tensor comes back in this
    repo's index order via the same relabelling `frame_to_dvcresult` uses."""
    M = _al_dvc_modules()
    mesh, _coords = stage_grid(para, shape)
    if valid is not None:
        mesh.node_valid = np.asarray(valid, dtype=bool).copy()
    sr = M["strain"].compute_strain(mesh, para, _u_to_xyz(U_zyx), F_direct=_f_to_xyz(F_direct_zyx),
                                    valid=np.asarray(mesh.node_valid, dtype=bool), ops=None)
    return _strain_tensor_zyx(sr), np.asarray(sr.strain_valid, dtype=bool), str(sr.strain_type)
