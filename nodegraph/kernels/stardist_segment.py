"""stardist_segment — StarDist 2-D-per-frame nuclear segmentation (vendored kernel).

PURPOSE
    Segment nuclei in a SINGLE already-prepared 2-D frame with StarDist, then
    (optionally) area-filter + relabel the resulting label image. This is the
    2-D-per-frame math kernel only; the caller owns all prep (file I/O, Z/
    channel selection, per-timepoint / per-multipoint looping, crop, downsample,
    registration, exclusion) and all orchestration.

WHERE THE REAL MATH LIVES
    The actual star-convex-polygon detection + non-maximum suppression is NOT in
    this file and NOT in the ND2Studios repo. It lives entirely inside the
    external `stardist` + `tensorflow` + `csbdeep` packages (StarDist2D neural
    net + trained weights). The in-repo math vendored here is thin glue:
      * csbdeep percentile normalization to (1, 99.8),
      * an auto-tiling heuristic (tile if max dimension > 1024),
      * a single call to StarDist2D.predict_instances,
      * a np.bincount + lookup-table area filter / contiguous relabel.

PROVENANCE (branch: Version-1.45)
    * segment_frame, get_stardist_model, and their private helpers
      (_ensure_tf_threading, _patch_windows_symlink) + the module-global model
      singleton — from nd2studios/backend/celltracker/segmentation.py
    * filter_and_relabel — from nd2studios/backend/analysis/source_utils.py

    Vendored verbatim; imports nothing from nd2studios; caller owns all prep.

DROPPED MEMBERS (not on the compute path)
    * segment_timeseries (the T-loop convenience wrapper) — orchestration, not
      kernel; the app iterates frames itself via plane_runner.
    * source_shape / read_plane (from source_utils.py) — frame-source shape
      helpers, caller-side prep, not part of this kernel.
    No renames were needed (no name collision between the two source modules).

NOTE ON THIRD-PARTY IMPORTS
    tensorflow / stardist / csbdeep are imported LAZILY (inside
    get_stardist_model and segment_frame), exactly as in the source. This is
    load-bearing: `import stardist_segment` therefore succeeds WITHOUT
    tensorflow installed — the heavy deps are only required when you actually
    segment. numpy is the only top-level dependency. (The source modules import
    only numpy at top level; scikit-image is NOT used by either vendored
    function despite being listed as a nominal dep — see the .md.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# StarDist model singleton — loaded once on first use.
_stardist_model = None
_stardist_model_name = None
_tf_configured = False

def _ensure_tf_threading(disable_gpu: bool = False) -> None:
    """Configure TF threading on first use. Must happen before any TF ops.

    Op-parallelism is pinned to a **single** inter/intra-op thread — this is exactly
    how CellTracker's original repository runs StarDist, and it is dramatically
    faster here than TF's multi-threaded default: StarDist tiles a large frame into
    many small ``predict_instances`` passes, and multi-threaded intra-op on those
    small ops oversubscribes the cores and thrashes (removing the cap measured ~10×
    slower). The setting only takes effect before TF is initialised, so it is applied
    on the first model load, before any inference.

    When *disable_gpu* is True, ``CUDA_VISIBLE_DEVICES`` is cleared **before** the
    first TensorFlow import so inference stays on CPU (toggling it after import has
    no effect).
    """
    global _tf_configured
    if _tf_configured:
        return
    import os
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf_configured = True


def _patch_windows_symlink() -> None:
    """On Windows, os.symlink requires elevated privileges or Developer Mode.
    Fall back to copying the directory tree so csbdeep model loading still works."""
    import sys
    if sys.platform != "win32":
        return
    import os
    _orig = os.symlink

    def _symlink_or_copy(src, dst, target_is_directory=False, *, dir_fd=None):
        try:
            _orig(src, dst, target_is_directory, dir_fd=dir_fd)
        except OSError:
            import shutil
            from pathlib import Path
            dst_path = Path(str(dst))
            src_path = (dst_path.parent / src) if not Path(str(src)).is_absolute() else Path(str(src))
            if src_path.is_dir():
                shutil.copytree(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

    os.symlink = _symlink_or_copy


def get_stardist_model(model_name: str = "2D_versatile_fluo", disable_gpu: bool = False,
                       *, dim: int = 2, model_path: str = ""):
    """Get or load the StarDist model (singleton).

    ``dim`` picks ``StarDist2D`` vs ``StarDist3D`` (V2.13 — the 3D lever). ``model_path``
    loads a LOCALLY TRAINED model directory instead of a pretrained name, which is not a
    nicety in 3D: ``3D_demo`` is the only 3D model StarDist registers and it is a demo, so
    real volumetric work means training your own (Weigert et al., WACV 2020) and pointing
    here at it. csbdeep's convention for that is ``StarDist3D(None, name=<dir>,
    basedir=<parent>)``.

    The singleton key includes ``dim`` and the path, so a 2D and a 3D StarDist node in one
    graph do not silently hand each other the wrong network. They still THRASH — one global
    slot, so alternating pulls reload — the same caveat this kernel and
    ``cellsam_segment`` have always carried. Pass an explicit ``model`` to avoid it.
    """
    global _stardist_model, _stardist_model_name

    key = (int(dim), str(model_path) or str(model_name))
    if _stardist_model is not None and _stardist_model_name == key:
        return _stardist_model

    _patch_windows_symlink()
    _ensure_tf_threading(disable_gpu=disable_gpu)
    from stardist.models import StarDist2D, StarDist3D
    cls = StarDist3D if int(dim) == 3 else StarDist2D
    if model_path:
        import os
        path = os.path.abspath(os.path.expanduser(str(model_path).strip().strip('"')))
        if not os.path.isdir(path):
            raise ValueError(
                f"StarDist model_path must be a trained-model DIRECTORY (the folder holding "
                f"config.json + weights_best.h5), got {path!r}")
        # csbdeep loads by (basedir, name); config=None reads the directory's own config.json,
        # so a model trained at any anisotropy/ray count loads without restating it here.
        _stardist_model = cls(None, name=os.path.basename(path.rstrip("/\\")),
                              basedir=os.path.dirname(path.rstrip("/\\")))
    else:
        _stardist_model = cls.from_pretrained(model_name)
    _stardist_model_name = key
    return _stardist_model


def segment_frame(
    image: np.ndarray,
    model=None,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    scale: Optional[float] = None,
    model_name: str = "2D_versatile_fluo",
    disable_gpu: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Segment nuclei in a single 2D frame using StarDist.

    Parameters
    ----------
    image : 2D array, any dtype
    model : StarDist2D instance, or None to use singleton
    prob_thresh : float, object probability threshold
    nms_thresh : float, non-maximum suppression threshold
    scale : float or None, rescale factor
    model_name : str, pretrained model name
    disable_gpu : bool, clear CUDA_VISIBLE_DEVICES before the first TF import

    Returns
    -------
    labels : 2D int array, each nucleus has a unique ID
    details : dict with prob, coord, etc.
    """
    from csbdeep.utils import normalize

    if model is None:
        model = get_stardist_model(model_name, disable_gpu=disable_gpu)

    img_norm = normalize(image, 1, 99.8)

    # Tiling for large images
    h, w = img_norm.shape
    n_tiles = None
    if max(h, w) > 1024:
        n_tiles = (max(1, h // 512), max(1, w // 512))

    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        scale=scale,
        n_tiles=n_tiles,
    )

    return labels, details


def segment_volume(
    volume: np.ndarray,
    model=None,
    *,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    scale: Optional[float] = None,
    scale_z: Optional[float] = None,
    model_name: str = "3D_demo",
    model_path: str = "",
    disable_gpu: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Segment nuclei in a single prepared 3-D volume with StarDist3D (V2.13).

    The 3-D counterpart of :func:`segment_frame`. StarDist3D predicts star-convex
    POLYHEDRA (rays on a golden-spiral sphere) rather than 2-D polygons, so the objects it
    returns are genuinely z-connected — this is not a stack of per-plane results, which is
    exactly why the node's 3D lever can now route here instead of refusing.

    Parameters
    ----------
    volume : 3-D array ``(Z, Y, X)``, any dtype
        One prepared volume — one channel, one timepoint. ``axes="ZYX"`` is passed
        explicitly rather than left to csbdeep's guesser, which would mis-read a volume
        whose Z happens to be small as a channel axis.
    scale, scale_z : float or None
        Internal resize factor. ``scale`` applies laterally; ``scale_z`` applies along Z and
        defaults to ``scale`` when omitted. Splitting them is what makes ANISOTROPIC data
        usable: a pretrained network learned nuclei at one apparent aspect ratio, and
        confocal voxels are typically several times taller than they are wide, so the same
        nucleus is squashed along z relative to what the model expects. Pass
        ``scale_z = z_step_um / pixel_size_um`` (times any lateral scale) to present the
        model with near-isotropic objects. ``None`` for both = no resizing.
    model_path : str
        A locally trained model directory — see :func:`get_stardist_model`. Strongly
        recommended in 3D: the only registered 3D model is a demo.

    Returns
    -------
    labels : 3-D int array ``(Z, Y, X)``, one id per nucleus, z-connected.
    details : dict from StarDist (probabilities, ray coordinates, …).
    """
    from csbdeep.utils import normalize

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"stardist segment_volume expects a 3-D (Z,Y,X) volume, got shape "
                         f"{vol.shape} — the caller owns the (m,t,c) loop")
    if model is None:
        model = get_stardist_model(model_name, disable_gpu=disable_gpu, dim=3,
                                   model_path=model_path)

    img_norm = normalize(vol, 1, 99.8)

    # Tiling heuristic, the 3-D form of segment_frame's. A volume is the product of three
    # axes, so it hits memory limits far sooner than a frame does; tile whenever an axis
    # exceeds 256, and never tile Z below 1 (a 12-plane stack must stay one tile).
    nz, ny, nx = img_norm.shape
    n_tiles = None
    if max(nz, ny, nx) > 256:
        n_tiles = (max(1, nz // 128), max(1, ny // 256), max(1, nx // 256))

    # scale must be a per-AXIS triple when the two differ; a scalar would scale z with the
    # lateral factor and defeat the anisotropy correction above.
    sc = None
    if scale is not None or scale_z is not None:
        lat = float(scale) if scale is not None else 1.0
        ax = float(scale_z) if scale_z is not None else lat
        sc = (ax, lat, lat)

    labels, details = model.predict_instances(
        img_norm,
        axes="ZYX",
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        scale=sc,
        n_tiles=n_tiles,
    )
    return labels, details


def filter_and_relabel(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Drop objects outside ``[min_area, max_area]`` (px) and relabel ``1..K``.

    Single ``O(pixels)`` pass: object areas come from ``np.bincount`` over the
    label image, a lookup table maps each surviving label to a fresh contiguous
    id, and ``lut[mask]`` applies the filter + relabel at once. This replaces the
    old per-object ``mask[mask == label] = 0`` / ``out[mask == id] = new`` loops,
    which scanned the whole frame **once per object** — ``O(n_objects × pixels)``,
    tens of seconds on a 4096² frame with thousands of nuclei. Returns an
    ``int32`` array of the same shape.
    """
    mask = np.asarray(mask)
    if mask.size == 0 or int(mask.max()) == 0:
        return mask.astype(np.int32, copy=False)
    counts = np.bincount(mask.ravel())
    lut = np.zeros(counts.shape[0], dtype=np.int32)
    new_id = 1
    for lbl in range(1, counts.shape[0]):
        c = int(counts[lbl])
        if c and min_area <= c <= max_area:
            lut[lbl] = new_id
            new_id += 1
    return lut[mask]
