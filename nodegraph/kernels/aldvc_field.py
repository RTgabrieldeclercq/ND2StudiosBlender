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
