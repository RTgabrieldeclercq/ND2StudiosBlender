"""Segmentation (``analysis.segment``) — THE segmentation node: image → a Voxel label raster + a Label table, with the algorithm as a `method` Mode — threshold+CCL, distance-transform watershed, StarDist (CNN), or CellSAM…"""

from __future__ import annotations

import numpy as np
import os

from collections import defaultdict
from typing import Dict, Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.parallel import cpu_budget, fold_units, tile_cache_bytes
from nodegraph.registry import (
    DimMode,
    InBool,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.spill import dense_output
from nodegraph.structure import StructureTable, label_components, seeded_watershed
from nodegraph.trained import (stardist_config, stardist_model_dir,
                              stardist_note, stardist_trained)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.batch import batch_aware
from nodegraph.catalog._shared.columns import LABEL_INVARIANT, on_layer
from nodegraph.catalog._shared.dim_footprint import _DIM_GRAN_GLOBAL, _DIM_KAX
from nodegraph.catalog._shared.progress import _UnitBar
from nodegraph.catalog._shared.units import to_pixels_v2

# ── Watershed (split a mask by EDT-peak markers → Labels) ───────────────────────

def _labeled_table(raster: np.ndarray, *, m: int, t: int, c: int, layer: str,
                   is_3d: bool, z_index: int = 0) -> StructureTable:
    """A Label StructureTable (invariant ``id,m,t,c,area,z,y,x`` schema) from an
    ALREADY-labelled raster (ids are the label values). Region-props for nodes that
    produce labels by a means other than fresh CCL (e.g. watershed)."""
    from scipy import ndimage as ndi
    labs = np.unique(raster)
    labs = labs[labs != 0]
    k = len(labs)
    zk = "subpixel" if is_3d else "plane_index"
    if k == 0:
        cols = {n: np.zeros(0, dtype=(np.int64 if n in ("id", "m", "t", "c", "area")
                                      else float))
                for n in ("id", "m", "t", "c", "area", "z", "y", "x")}
        return StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk)
    ones = np.ones_like(raster, dtype=float)
    areas = np.asarray(ndi.sum_labels(ones, raster, labs))
    coms = np.asarray(ndi.center_of_mass(ones, raster, labs), dtype=float).reshape(k, -1)
    if is_3d:
        cz, cy, cx = coms[:, 0], coms[:, 1], coms[:, 2]
    else:
        cy, cx = coms[:, 0], coms[:, 1]
        cz = np.full(k, float(z_index))
    cols = {
        "id": labs.astype(np.int64), "m": np.full(k, m, np.int64),
        "t": np.full(k, t, np.int64), "c": np.full(k, c, np.int64),
        "area": areas.astype(np.int64), "z": cz, "y": cy, "x": cx,
    }
    return StructureTable(Domain.LABEL, cols, layer=layer, z_kind=zk)
# ── Segmentation — the ONE image→instance-labels node (`method` = the algorithm) ─
#
# V2.12. Every segmentation shares one data contract: an image in, a Voxel **label
# raster** + a per-object **Label table** out, with globally-unique ids. Only the
# algorithm that finds the objects differs. So the catalog carries ONE Segmentation node
# with a ``method`` Mode instead of one node per algorithm: ``analysis.watershed`` and
# ``detect.stardist_nuclei`` were folded in here and deleted, **CellSAM** is new, and
# every shared stage — the µm²/µm³ size filter, hole filling, the contiguous relabel, the
# global id offset, the Label table, the 2D/3D lever — is written exactly ONCE instead of
# once per method (which is how the old pair drifted: stardist filtered by area and
# watershed did not, stardist had no lever and watershed did).
#
# The methods are deliberately heterogeneous in what they need, and ``available_in`` keeps
# each method's controls to itself (wire-node-v2 §5b): two classical methods that cut a
# foreground and split it (``threshold``, ``watershed``) and two learned detectors that
# take an image and return instances directly (``stardist``, ``cellsam``).

_SEGMENT_METHODS = ("threshold", "watershed", "stardist", "cellsam")
#: Foreground level for the classical methods — the `analysis.threshold` menu, defaulting
#: to `otsu` because a segmentation node should work on an unseen image with no numbers.
_SEGMENT_LEVELS = ("otsu", "li", "yen", "triangle", "mean", "fixed")
#: Methods that CUT a foreground and therefore read the level controls.
_SEGMENT_CLASSICAL = frozenset({"threshold", "watershed"})
#: Methods whose backend is a 2-D-per-plane detector (StarDist2D; CellSAM's ViT + SAM
#: decoder). They cannot produce z-connected 3D instances, so 3D mode is REFUSED rather
#: than quietly returning a stack of per-plane labels while the lever claims 3D
#: (wire-node-v2 §5 — never silently loop 2D over Z while claiming 3D). 3D consensus
#: fusion of 2D slices is a real algorithm (u-Segment3D, CellSAM paper Fig. 3d), not
#: something to fake here.
#: **StarDist left this set on 2026-07-30.** It is genuinely volumetric: ``StarDist3D``
#: predicts star-convex POLYHEDRA (rays on a golden-spiral sphere, ``Config3D.rays``) and
#: returns z-connected objects, so routing the 3D lever to
#: ``stardist_segment.segment_volume`` is real 3D — not the stack-of-2D fake this set exists
#: to forbid (Weigert et al., "Star-convex Polyhedra for 3D Object Detection and
#: Segmentation in Microscopy", WACV 2020; github.com/stardist/stardist).
#: CellSAM stays: its decoder is 2-D by construction, and fusing its slices is u-Segment3D's
#: job (CellSAM paper Fig. 3d), not something to fake here.
_SEGMENT_2D_ONLY = frozenset({"cellsam"})
def _segment_level(arr: np.ndarray, level: str, fixed: float) -> float:
    """The foreground cut for ONE segmentation unit (a plane in 2D, a volume in 3D).

    ``fixed`` is the user's level in the image's own intensity units; every other choice
    derives it from that unit's own histogram — **per unit**, which is what the node's
    declared footprint says it reads (WHOLE_PLANE in 2D / WHOLE_VOLUME in 3D). One level
    for the whole dataset is a different (also valid) recipe, and it is what
    ``analysis.threshold`` does: compose it with this node's ``watershed`` ``mask`` socket
    when you want a single global cut.

    A flat unit gets no histogram level: skimage's methods return the constant itself (or
    warn and divide by zero), and a cut AT the constant paints the WHOLE plane as one
    giant object. Returning just above the maximum yields an empty foreground, which is
    the honest answer for a blank plane.

    **Non-finite pixels are excluded from the histogram** (V2.12): a SINGLE NaN or inf —
    which a deconvolution, a normalize of a flat region or a resampled edge can produce —
    otherwise makes every skimage method raise ``autodetected range … is not finite``, and
    NaN also defeats the flat-unit guard above (``nan == nan`` is False). Cutting on the
    finite population is the useful answer; NaN voxels then fall out of the foreground on
    their own, since ``nan > level`` is False."""
    if level == "fixed":
        return float(fixed)
    import skimage.filters as skf
    fn = {"otsu": skf.threshold_otsu, "li": skf.threshold_li, "yen": skf.threshold_yen,
          "triangle": skf.threshold_triangle, "mean": skf.threshold_mean}[level]
    flat = np.asarray(arr, dtype=float).ravel()      # 1-D (skimage RGB-shape guard)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("inf")                          # nothing to cut ⇒ empty foreground
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        return hi + 1.0                              # blank unit ⇒ empty foreground
    return float(fn(finite))
def _segment_fill_holes(raster: np.ndarray) -> np.ndarray:
    """Fill each label region's interior holes — the "hole filling" half of the
    postprocess Cellpose and CellSAM both apply (CellSAM, *Nat. Methods* 22:2585, Methods
    → "CellSAM postprocessing"), here shared by every method.

    **Non-destructive**, unlike the upstream idiom. Cellpose's
    ``fill_holes_and_remove_small_masks`` writes ``masks[slc][filled] = id`` across the
    object's whole bounding box, which steals voxels that already belong to a NEIGHBOURING
    label whenever two objects share a box. Only voxels that are currently background are
    painted here, so filling can never move a boundary between two objects."""
    from scipy import ndimage as ndi
    out = np.asarray(raster)
    if out.size == 0 or int(out.max()) == 0:
        return out
    out = out.copy()
    for i, slc in enumerate(ndi.find_objects(out), start=1):
        if slc is None:                              # id absent (non-contiguous labels)
            continue
        sub = out[slc]                               # a VIEW — assignment writes through
        holes = ndi.binary_fill_holes(sub == i) & (sub == 0)
        if holes.any():
            sub[holes] = i
    return out
def _segment_size_filter(raster: np.ndarray, min_px: int, max_px: int) -> np.ndarray:
    """Drop objects outside ``[min_px, max_px]`` **voxels** (inclusive; ``0`` disables
    either bound) and relabel the survivors ``1..K`` — one ``O(voxels)`` bincount + LUT
    pass, the shape of ``stardist_segment.filter_and_relabel``.

    The relabel runs even with both bounds off, and that is load-bearing: the caller makes
    ids globally unique by adding a running offset per unit, which is only correct if each
    unit's own ids are contiguous from 1."""
    lab = np.asarray(raster).astype(np.int64, copy=False)
    top = int(lab.max()) if lab.size else 0
    if top == 0:
        return np.zeros(lab.shape, dtype=np.int64)
    counts = np.bincount(lab.ravel(), minlength=top + 1)
    keep = counts > 0
    keep[0] = False                                  # background is never an object
    if min_px > 0:
        keep &= counts >= min_px
    if max_px > 0:
        keep &= counts <= max_px
    lut = np.zeros(top + 1, dtype=np.int64)
    lut[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.int64)
    return lut[lab]
def _segment_watershed_split(fg: np.ndarray, sampling, footprint: np.ndarray) -> np.ndarray:
    """Split a foreground mask into touching objects by seeding a watershed at the peaks
    of the (anisotropic) distance transform — the classic "separate touching objects" step,
    carried over verbatim from the folded-in ``analysis.watershed``.

    Peak suppression is a **physical** radius expressed as a per-axis ``footprint``, not
    the isotropic index-unit ``min_distance``: on an anisotropic volume (z_step > pixel)
    an index-unit radius suppresses far too aggressively along Z and under-segments.

    **Adjacent peaks are ONE marker** (V2.12 fix). A distance transform is full of
    plateaus — a disk on an integer grid has several pixels at the same maximum, and an
    elongated object has a whole ridge of them — and ``peak_local_max`` returns *every*
    pixel of a plateau. The folded-in ``analysis.watershed`` numbered each returned pixel
    as its own marker, so one object was shattered into as many basins as its plateau had
    pixels (measured: four synthetic disks became 19 fragments per plane, the smallest of
    them 1 voxel). Connected-component labelling the peak mask first — the recipe from
    scikit-image's own watershed example — collapses each plateau to a single seed while
    leaving genuinely distinct maxima separate, which is the whole point of the node.

    That merge uses **full** connectivity (8 in 2D / 26 in 3D), not scipy's cross-shaped
    default: a plateau is frequently a diagonal ring — the equidistant crest inside any
    object with a hole in it — and a cross-connected label breaks such a ring into one
    marker per diagonal step (the same four disks then yield 11 basins instead of 4).

    **`exclude_border=False` is load-bearing** (V2.12 fix). ``peak_local_max`` defaults to
    excluding a border shell ``min_distance`` wide on EVERY axis — and ``min_distance`` is a
    parameter this call does not even use, since the physical suppression is the per-axis
    ``footprint``. Leaving the default on discards peaks near the frame edge, and in 3D on a
    volume with ``z <= 2`` there is **no interior z plane at all**, so it returns ZERO peaks:
    the fallback below then plants a single marker and the watershed collapses every object
    in the volume into one (measured on two disjoint disks over z=1 and z=2 — 1 object
    instead of 2, silently). Objects at the image border are real objects."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int64)
    edt = ndi.distance_transform_edt(fg, sampling=sampling)
    peaks = peak_local_max(edt, footprint=footprint, labels=fg.astype(np.int64),
                           exclude_border=False)
    seeds = np.zeros(fg.shape, dtype=bool)
    if len(peaks):
        seeds[tuple(np.asarray(peaks).T)] = True
    full = ndi.generate_binary_structure(seeds.ndim, seeds.ndim)
    markers = np.asarray(ndi.label(seeds, structure=full)[0], dtype=np.int64)
    if markers.max() == 0:                           # no separable peak → one basin
        markers[np.unravel_index(int(np.argmax(edt)), edt.shape)] = 1
    return np.asarray(seeded_watershed(fg, markers, sampling=sampling), dtype=np.int64)
# ── what the loaded StarDist checkpoint says about itself (V2.23) ──────────────

def _as_num(value) -> "float | None":
    """``value`` as a finite float, or ``None`` for anything that is not one.

    Used on ``model.thresholds`` fields, which are namedtuple entries built from a JSON
    file — so a field can legitimately be absent on an unexpected StarDist build, and the
    reporting path must degrade to "say nothing" rather than raise while formatting a
    progress note."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _stardist_note(params, modes) -> str:
    """``NodeSpec.trained_note`` — see :func:`nodegraph.trained.stardist_note`.

    Silent for the three methods that load no such model, so the panel gains a line
    only where there is a model record to talk about."""
    if str((modes or {}).get("method") or "threshold") != "stardist":
        return ""
    return stardist_note(params, modes)


def _stardist_trained(params, modes) -> dict:
    """``NodeSpec.trained_params`` for this node: the values the LOADED StarDist checkpoint
    was trained with, for the sockets that have one.

    A thin delegate to :func:`nodegraph.trained.stardist_trained` — the resolver lives in
    ``nodegraph/`` rather than here because ``metadata.py`` and the GUI both reach it, and
    because keeping it out of the catalog module means reading a model's JSON never pulls
    scipy/skimage in behind it.

    **Only for the ``stardist`` method.** The other three speak for nothing here: the two
    classical methods have no model at all, and CellSAM's checkpoint is a bare torch state
    dict whose configuration lives in the ``cellSAM`` package rather than beside the
    weights. Returning values for a method that is not selected would put a
    checkpoint-derived number under a socket the run will never consult.
    """
    if str((modes or {}).get("method") or "threshold") != "stardist":
        return {}
    return stardist_trained(params, modes)


def _sd_check(ctx: EvalContext, *, is_3d: bool, px: float, zs: float) -> None:
    """Refuse — or warn about — a StarDist checkpoint whose own ``config.json`` disagrees
    with how this node is set up.

    Called BEFORE the model loads, because every failure it catches is otherwise expensive
    or silent:

    * **``n_dim``.** A 2D checkpoint handed to ``StarDist3D`` (or the reverse) fails deep
      inside csbdeep with a shape or axes error that names neither the model nor the lever.
      The mismatch is stated in one line of the model's own record, so read it.
    * **``n_channel_in``.** This node feeds single-channel planes (each channel is segmented
      independently, by design). A checkpoint trained on RGB — ``2D_versatile_he`` is one —
      cannot consume that, and the failure again arrives as a shape error.
    * **``anisotropy``.** NOT an error and NOT auto-applied. A StarDist3D model's rays are
      built on a sphere scaled by the training anisotropy (``Rays_GoldenSpiral(n,
      anisotropy=…)``, and ``edt_prob`` uses it too), so the network learned distances on a
      grid of that aspect — but rescaling the volume at inference to match is a correction
      StarDist's own documentation does not prescribe (upstream's advice is to TRAIN at your
      data's anisotropy). So this reports the disagreement and names the ``scale_z`` that
      would close it, and leaves the decision with the user rather than silently changing
      what a 3D segmentation returns.

    Reads nothing if there is no model directory on disk yet — a pretrained checkpoint that
    has not been downloaded states nothing until the loader below fetches it.
    """
    cfg = stardist_config(ctx.params, {"dim": "3D" if is_3d else "2D"})
    if not cfg:
        return
    where = stardist_model_dir(ctx.params, {"dim": "3D" if is_3d else "2D"})
    name = os.path.basename(where.rstrip("/\\")) or where
    want_dim = 3 if is_3d else 2
    got_dim = _as_num(cfg.get("n_dim"))
    if got_dim is not None and int(got_dim) != want_dim:
        raise ValueError(
            f"segmentation (stardist): {name!r} is a {int(got_dim)}-D checkpoint but the "
            f"2D/3D lever is set to {'3D' if is_3d else '2D'}, which loads "
            f"StarDist{want_dim}D. Its own config.json says n_dim={int(got_dim)}. A "
            f"{int(got_dim)}-D checkpoint cannot load into StarDist{want_dim}D — set the "
            f"lever to {int(got_dim)}D, or point at a {want_dim}-D model.")
    n_ch = _as_num(cfg.get("n_channel_in"))
    if n_ch is not None and int(n_ch) != 1:
        raise ValueError(
            f"segmentation (stardist): {name!r} was trained on {int(n_ch)}-channel input "
            f"(config.json n_channel_in={int(n_ch)}) but this node segments ONE channel at a "
            f"time — each channel independently, so a fused multi-channel segmentation has "
            f"no single `c` to file its Label rows under. Use a single-channel checkpoint "
            f"(`2D_versatile_fluo` for fluorescence); `2D_versatile_he` is the RGB one.")
    # anisotropy: report, never apply.
    aniso = cfg.get("anisotropy")
    if not (is_3d and isinstance(aniso, (list, tuple)) and len(aniso) == 3 and px > 0):
        return
    az, ay = _as_num(aniso[0]), _as_num(aniso[1])
    if not az or not ay or az <= 0 or ay <= 0:
        return
    trained_aspect = az / ay
    data_aspect = (zs / px) if px > 0 else 0.0
    if data_aspect <= 0:
        return
    # 15%: below that the ray geometry difference is far smaller than the segmentation's own
    # run-to-run spread, and a warning that fires on every ordinary volume trains the user to
    # ignore the rail.
    if abs(data_aspect - trained_aspect) <= 0.15 * trained_aspect:
        return
    ctx.progress(0, 1,
                 f"segmentation (stardist): {name!r} was trained at z:xy anisotropy "
                 f"{trained_aspect:.3g} but this volume's is {data_aspect:.3g} "
                 f"(z_step {zs:.4g} / pixel {px:.4g} um). Its star-convex rays were built "
                 f"for the trained aspect, so objects here are a different shape than it "
                 f"learned. Set `scale_z` to {data_aspect / trained_aspect:.3g} to present "
                 f"it the trained aspect, or retrain at this anisotropy (which is what "
                 f"StarDist recommends).")


# ── process-pool workers for the learned segmenters (V2.14) ────────────────────
#
# StarDist / CellSAM are the textbook case for PROCESS parallelism, and the reason is in
# their own kernels: `stardist_segment._ensure_tf_threading` pins TensorFlow to a SINGLE
# inter/intra-op thread (its docstring records that removing the cap measured ~10× SLOWER,
# because StarDist tiles a frame into many small predict passes and multi-threaded intra-op
# oversubscribes). One-thread inference is therefore the fast configuration — and the only
# way to then use twelve cores is twelve of them, one per frame. Frames are independent:
# both are 2-D-per-plane detectors that refuse 3D outright.
#
# These must be MODULE-LEVEL (pickled by qualified name) and take only picklable
# arguments, which is why the model arrives as an *identity* rather than an object: each
# worker's own module singleton loads it once per process and reuses it for every frame it
# is handed. Measured on the analogous heavy per-plane workload: 5.23× on processes vs
# 1.71× on threads (skimage/framework glue holds the GIL).

def _seg_worker_stardist(payload: tuple) -> np.ndarray:
    """One frame → StarDist labels, in a pool process. ``payload`` carries the model
    NAME, not the model: a Keras/TF graph is not picklable, and shipping one per frame
    would dwarf the inference anyway."""
    from nodegraph.kernels.stardist_segment import get_stardist_model, segment_frame
    arr, model_id, prob, nms, scale, cpu = payload
    model = get_stardist_model(model_id, disable_gpu=cpu)      # per-process singleton
    return np.asarray(segment_frame(arr, model=model, prob_thresh=prob,
                                    nms_thresh=nms, scale=scale)[0], dtype=np.int64)
# There is deliberately NO `_seg_worker_cellsam`. CellSAM runs on THREADS (see
# `_segment_backend`), so it needs no picklable module-level worker — and a pickle worker for
# a backend that never selects "process" would be dead code that looks live, the exact thing
# this file's socket contract forbids for controls. If CellSAM is ever moved back to
# processes, it needs one shaped like `_seg_worker_stardist`: the model identity
# `(model, path)` in the payload, never the loaded `nn.Module`.


#: Resident bytes to budget per PROCESS worker for a model-bearing segmenter — the model
#: plus its framework, not the pixels. StarDist is the only entry because it is the only
#: method that selects the process backend (CellSAM runs on threads, sharing one model).
#:
#: 2 GiB is a deliberately CONSERVATIVE estimate rather than a measurement: a TensorFlow
#: process is its weights (~tens of MB for a StarDist U-Net) plus the CUDA/oneDNN runtime,
#: the graph, and an allocator arena that TF grows and does not return. Under-estimating
#: here costs a swapping machine or an OOM mid-pull — a failure the user reads as "the app
#: is broken" — while over-estimating costs some parallelism on a small box. Those are not
#: symmetric, so this errs high. Override per machine with NODEGRAPH_SEG_MODEL_BYTES.
_SEG_MODEL_WORKER_BYTES = {"stardist": 2 * 1024**3}
def _seg_proc_lane_cap(method: str) -> int:
    """How many PROCESS workers this method's model footprint allows.

    Half of installed RAM, divided by the per-worker estimate above, floored at 1 (never
    zero — a zero-lane map would do nothing) and only applied to methods that actually
    carry a model. Half rather than all: the parent process is simultaneously holding the
    output raster, the tile cache and the source, and a budget that assumes the workers own
    the machine is the one that OOMs."""
    per = _SEG_MODEL_WORKER_BYTES.get(method)
    if not per:
        return 1 << 30                      # not model-bearing: no cap of this kind
    env = os.environ.get("NODEGRAPH_SEG_MODEL_BYTES", "").strip()
    if env:
        try:
            per = max(1, int(float(env)))
        except ValueError:
            pass                            # unparsable override: keep the estimate
    from nodegraph.parallel import total_ram_bytes
    return max(1, int((total_ram_bytes() // 2) // per))
def _segment_backend(method: str) -> str:
    """Which pool ``analysis.segment`` should run its per-unit work on.

    ``"process"`` for a **CPU** learned segmenter (see above). ``"serial"`` for a
    **GPU** one — deliberately: there is one device, the framework already saturates it,
    and eight processes each building a CUDA context would exhaust card memory (a ViT-B
    plus a TF arena, eight times over) to run work that was never CPU-bound. ``"thread"``
    for the classical methods, whose distance transform / watershed / connected-component
    passes are scipy calls that release the GIL."""
    if method == "stardist":
        cpu = os.environ.get("NODELAB_STARDIST_CPU", "") in ("1", "true", "yes")
        return "process" if cpu else "serial"
    if method == "cellsam":
        try:
            from nodegraph.kernels.cellsam_segment import resolve_device
            # THREADS, not processes, on CPU (2026-07-30). The process argument that holds
            # for StarDist does NOT transfer, because the two backends have opposite
            # threading models:
            #   * StarDist's kernel PINS TensorFlow to one inter/intra-op thread, so a single
            #     inference uses one core and the only way to use twelve is twelve processes.
            #   * CellSAM is torch, which is internally multi-threaded on CPU and releases
            #     the GIL inside the ViT's matmuls — ONE process already saturates the
            #     machine. N processes would therefore buy no extra FLOPs while multiplying a
            #     ~375 MB ViT-B + SAM decoder by N resident copies (`_worker_init` divides
            #     OMP threads among workers, so the arithmetic is the same work, N× the RAM).
            # A thread pool gets the same parallelism against ONE model, which is what the
            # kernel's process-singleton was designed to provide.
            #
            # Thread-safety of the shared model: every lane in a pull writes the SAME
            # threshold values onto it (`segment_plane` assigns mask_threshold/iou_threshold,
            # upstream assigns bbox_threshold), because all lanes read one params dict. That
            # makes the write race benign — same value, any order. It stops being benign the
            # moment any of those becomes per-unit, so if that ever changes, hoist the
            # assignment out of the per-plane call before re-reading this line.
            return "thread" if resolve_device() == "cpu" else "serial"
        except Exception:      # noqa: BLE001 — torch absent/odd: stay on the safe path
            return "serial"
    return "thread"
def _compute_segment(ctx: EvalContext) -> Dataset:
    """**Segmentation** — image → a Voxel **label raster** + a per-object **Label table**
    (ids globally unique across every unit); ``method`` picks the algorithm.

    THE segmentation node. Input and output domains are identical for every method, so the
    algorithm is a Mode rather than a separate node type, and the surrounding stages are
    shared:

    ``method``
        * **threshold** — cut a foreground at ``level`` (otsu/li/yen/triangle/mean, or a
          ``fixed`` level in the image's own units) and connected-component label it
          (``connectivity`` 4/8 in 2D, 6/18/26 in 3D; ``0`` = per-dim default).
        * **watershed** — the same foreground, split at the peaks of the anisotropic
          distance transform (``min_distance``, µm). An existing binary/label layer can be
          used as the foreground instead by naming it in ``mask`` — that is how the
          removed ``analysis.watershed`` behaved, and it is what to use when the mask comes
          from ``analysis.threshold`` / ``analysis.roi_mask`` /
          ``analysis.histogram_threshold``.
        * **stardist** — StarDist star-convex CNN (``prob_thresh``/``nms_thresh``/
          ``scale``/``model_name``), kernel :mod:`nodegraph.kernels.stardist_segment`.
        * **cellsam** — CellSAM: a SAM ViT-B whose mask decoder is prompted by CellFinder
          (Anchor-DETR) box detections, i.e. a *generalist* segmenter that needs no
          per-dataset tuning and no seeds (``bbox_threshold`` is the precision/recall
          knob; ``normalize``/``postprocess``/``remove_boundaries``/``tile``…), kernel
          :mod:`nodegraph.kernels.cellsam_segment`.

    Shared by all four: the output layer ``name`` (one name, two domains — the Voxel
    raster and the Label table), hole filling, the size filter in **µm² (2D) / µm³ (3D)**,
    the contiguous relabel, the global id offset, and the Label table's invariant
    ``id,m,t,c,area,z,y,x`` schema with ``area`` in voxels.

    **2D vs 3D is what the lever means here:** 2D segments each ``(m,t,z,c)`` plane
    independently and emits per-plane instances (``z_kind="plane_index"``); 3D segments
    each ``(m,t,c)`` volume and emits z-connected instances (``z_kind="subpixel"``). The
    two learned methods are 2-D-per-plane detectors and **refuse** 3D rather than
    pretending a stack of per-plane labels is a 3D segmentation.

    Reads ``pixel_size_um`` (and ``z_step_um`` in 3D) for the size filter and the seed
    radius; both are recorded, so the memo re-checks them.

    Per-channel: each channel is segmented independently, as every other structure
    producer in the catalog does. CellSAM's ``(blank, nuclear, whole-cell)`` multi-channel
    fusion is deliberately NOT exposed — a fused segmentation has no single ``c`` to file
    its Label rows under, and inventing one silently would corrupt every downstream
    per-channel join. Select the marker channel upstream (``channel.select``)."""
    ds = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("segmentation needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.is_volume
    modes = ctx.params.get("__modes__", {})
    method = str(modes.get("method") or "threshold")
    if method not in _SEGMENT_METHODS:
        raise ValueError(f"unknown segmentation method {method!r} — one of "
                         f"{list(_SEGMENT_METHODS)}")
    if is_3d and method in _SEGMENT_2D_ONLY:
        raise ValueError(
            f"segmentation: {method!r} is a 2-D-per-plane detector and cannot produce "
            "z-connected 3D instances. Set the 2D/3D lever to 2D to segment every plane "
            "independently (per-plane instances), or z-project first. (Fusing 2D slices "
            "into true 3D objects is its own algorithm — u-Segment3D in the CellSAM "
            "paper — not something this node fakes.) The `threshold` and `watershed` "
            "methods do run volumetrically in 3D.")
    layer = ctx.layer("name")

    # ── the shared size filter: µm² (2D) / µm³ (3D) → voxels ──────────────────
    px = ctx.calib("pixel_size_um") or 0.1
    # The four size params are read with LITERAL keys, per dim, deliberately: the socket
    # contract's index is an AST pass that only resolves string constants (and helper args),
    # so a `ctx.params.get(key_variable)` read would make all four sockets look DEAD and
    # fail the guard — the one place where saying it twice is required, not sloppy.
    if is_3d:
        zs = ctx.calib("z_step_um") or 0.5
        vox, unit, lo_key, hi_key = px * px * zs, "µm³", "min_volume", "max_volume"
        lo_um = float(ctx.params.get("min_volume", 0.0))
        hi_um = float(ctx.params.get("max_volume", 0.0))
    else:
        vox, unit, lo_key, hi_key = px * px, "µm²", "min_area", "max_area"
        lo_um = float(ctx.params.get("min_area", 0.0))
        hi_um = float(ctx.params.get("max_area", 0.0))
    min_px, max_px = int(round(lo_um / vox)), int(round(hi_um / vox))
    # 0 is the OFF sentinel for both bounds, which is safe for the lower one (every object
    # has at least one voxel, so a sub-voxel minimum filters nothing either way) but
    # INVERTS the upper one: a sub-voxel maximum would quantize to "no upper limit" and
    # keep everything instead of dropping all but sub-voxel specks. Refuse, exactly as
    # analysis.histogram_threshold does with the same arithmetic.
    if hi_um > 0.0 and max_px == 0:
        raise ValueError(
            f"segmentation: {hi_key}={hi_um:g} {unit} is under one voxel ({vox:g} {unit}) "
            f"— it quantizes to 0, which is this filter's OFF value, so it would keep "
            f"every object instead of dropping the large ones. Raise it, or set exactly 0 "
            f"to disable the upper bound deliberately.")
    if max_px > 0 and min_px > max_px:
        raise ValueError(
            f"segmentation: {lo_key} ({min_px} voxels) exceeds {hi_key} ({max_px} voxels) "
            f"— the size window is empty, so every object would be discarded. Widen it "
            f"({hi_key} 0 = no upper limit).")
    fill = bool(ctx.params.get("fill_holes", False))

    # ── per-method setup (everything read here, once, not per unit) ────────────
    level = str(modes.get("level") or "otsu")
    fixed = 0.5
    conn = 26 if is_3d else 8
    mask6 = None
    footprint = None
    sampling = (zs, px, px) if is_3d else (px, px)
    if method in _SEGMENT_CLASSICAL:
        if level not in _SEGMENT_LEVELS:
            raise ValueError(f"unknown threshold level {level!r} — one of "
                             f"{list(_SEGMENT_LEVELS)}")
        # unset ⇒ the socket's derive: mid-range of the CURRENT declared bit depth, or 0.5
        # when there is none (post-Normalize [0,1] data). channel(0) because the depth is
        # channel-independent (§7c).
        fixed = float(ctx.channel(0).param("threshold", 0.5))
    if method == "threshold":
        # A QSpinBox cannot express "unset", so it commits 0 the moment it is touched —
        # and 0 is not a legal connectivity. Treat it as "per-dim default" instead of
        # letting `_connectivity_rank` raise on a value the user never chose.
        conn = int(ctx.params.get("connectivity", 0) or 0) or (26 if is_3d else 8)
    if method == "watershed":
        src = ctx.layer("mask")
        if src:
            attr = ds.get(Domain.VOXEL, src)
            if attr is None:
                raise ValueError(
                    f"segmentation (watershed): no Voxel layer {src!r} to use as the "
                    "foreground. Pick an existing mask/label layer, or clear the `mask` "
                    "socket to cut the foreground from the image with the level controls.")
            mask6 = np.asarray(attr.values)
        min_um = float(ctx.params.get("min_distance", 0.3))
        rxy = max(1, int(round(to_pixels_v2(min_um, "um", pixel_size_um=px))))
        if is_3d:
            rz = max(1, int(round(to_pixels_v2(min_um, "um_axial", z_step_um=zs))))
            footprint = np.ones((2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), dtype=bool)
        else:
            footprint = np.ones((2 * rxy + 1, 2 * rxy + 1), dtype=bool)

    model = None
    model_id = ""
    if method == "stardist":
        from nodegraph.kernels.stardist_segment import (get_stardist_model, segment_frame,
                                                        segment_volume)
        # Per-dim model socket, the same shape as min_area/min_volume: a 2D checkpoint cannot
        # load into StarDist3D, so one socket with one default could only ever be right in
        # one dim. `sd_path` (a locally trained model directory) is live in BOTH dims and
        # WINS over the name — in 3D it is effectively required, since `3D_demo` is the only
        # registered 3D checkpoint and it is a demo.
        sd_path = str(ctx.params.get("sd_model_path") or "").strip()
        model_id = str((ctx.params.get("model_name_3d") if is_3d
                        else ctx.params.get("model_name"))
                       or ("3D_demo" if is_3d else "2D_versatile_fluo"))
        # UNSET means "use what this checkpoint was trained with" (V2.23), which is why the
        # sentinel is `None` and not a number. Every StarDist model directory ships a
        # `thresholds.json` that `optimize_thresholds` wrote after measuring THAT network
        # against its own validation set, and `predict_instances(prob_thresh=None)` reads it
        # (`StarDistBase._predict_sparse_generator`: `if prob_thresh is None: prob_thresh =
        # self.thresholds.prob`). Passing a number unconditionally — which this node did
        # until V2.23 — overrode that tuning with a value nobody measured: on the registered
        # `3D_demo` the trained prob is 0.708 against the 0.5 that was being forced, and
        # StarDist's own nms fallback is 0.4 against the forced 0.3. The socket's `description`
        # had said so in as many words while the code did the opposite.
        #
        # `ctx.params.get` with NO fallback is the load-bearing part: params are raw
        # overrides (never default-filled), so absence is exactly "the user did not pin
        # this" — the same signal the inspector's auto/pin box draws from, resolved through
        # the shared `trained.stardist_trained` so the number shown and the number used
        # cannot drift.
        prob = ctx.params.get("prob_thresh")
        prob = float(prob) if prob not in (None, "") else None
        nms = ctx.params.get("nms_thresh")
        nms = float(nms) if nms not in (None, "") else None
        scale = ctx.params.get("scale")
        scale = float(scale) if scale not in (None, "") else 0.0
        scale = scale if scale > 0 else None          # a 0 spin-box means "no rescale"
        scale_z = ctx.params.get("scale_z")
        scale_z = float(scale_z) if scale_z not in (None, "") else 0.0
        scale_z = scale_z if scale_z > 0 else None
        # `disable_gpu` is an ENVIRONMENT knob, never a socket: the loader is a process
        # singleton that ignores the flag after the first call, and TF must see
        # CUDA_VISIBLE_DEVICES before its first import. As a param it would be a control
        # that silently stops working, and it would make the memo non-deterministic
        # (identical recipe hash, different device). Same argument as NODELAB_CELLSAM_DEVICE.
        import os as _os
        _cpu = _os.environ.get("NODELAB_STARDIST_CPU", "") in ("1", "true", "yes")
        # Read the checkpoint's OWN record before touching TensorFlow, so a dim or channel
        # mismatch is one sentence naming the fix instead of a shape error thrown from
        # somewhere inside csbdeep's resizer several seconds into a load (V2.23).
        _sd_check(ctx, is_3d=is_3d, px=px, zs=(zs if is_3d else 0.0))
        try:
            model = get_stardist_model(model_id, disable_gpu=_cpu,
                                       dim=(3 if is_3d else 2), model_path=sd_path)
        except Exception as exc:                      # noqa: BLE001 — one clear message
            raise ImportError(f"StarDist model "
                              f"{(sd_path or model_id)!r} unavailable "
                              f"(tensorflow/stardist/csbdeep + weights): {exc}") from exc
        if sd_path:
            model_id = sd_path        # provenance must name the checkpoint that RAN
        # The EFFECTIVE thresholds, read off the loaded model rather than re-derived from the
        # JSON. `model.thresholds` is the authority — csbdeep parsed the file, applied its own
        # `0 < x < 1` validity test and substituted its built-in 0.5/0.4 for anything that
        # failed — so this is the only value that is guaranteed to be what the network uses.
        # Reported on the rail and stamped into provenance below, because "which threshold
        # actually ran" is otherwise unanswerable from the graph once the socket is on auto.
        _th = getattr(model, "thresholds", None)
        eff_prob = float(prob) if prob is not None else _as_num(getattr(_th, "prob", None))
        eff_nms = float(nms) if nms is not None else _as_num(getattr(_th, "nms", None))
        if prob is None or nms is None:
            ctx.progress(0, 1, "segmentation (stardist): "
                               f"{'prob %.4g' % eff_prob if eff_prob is not None else ''}"
                               f"{' / ' if eff_prob is not None and eff_nms is not None else ''}"
                               f"{'nms %.4g' % eff_nms if eff_nms is not None else ''}"
                               " from the checkpoint's own thresholds.json")
    if method == "cellsam":
        from nodegraph.kernels.cellsam_segment import get_cellsam_model, segment_plane
        model_id = str(ctx.params.get("cellsam_model") or "cellsam_general")
        weights = str(ctx.params.get("model_path") or "")
        cs = dict(bbox_threshold=float(ctx.params.get("bbox_threshold", 0.4)),
                  # The other two of the paper's three inference thresholds. Upstream gives
                  # them no keyword — the kernel assigns them onto the model per call.
                  mask_threshold=float(ctx.params.get("mask_threshold", 0.4)),
                  mask_quality=float(ctx.params.get("mask_quality", 0.5)),
                  normalize=bool(ctx.params.get("normalize", True)),
                  postprocess=bool(ctx.params.get("postprocess", False)),
                  remove_boundaries=bool(ctx.params.get("remove_boundaries", False)),
                  tile=bool(ctx.params.get("tile", False)),
                  tile_size=int(ctx.params.get("tile_size", 512)),
                  overlap=int(ctx.params.get("tile_overlap", 56)),
                  tile_iou=float(ctx.params.get("tile_iou", 0.5)),
                  fast=bool(ctx.params.get("fast", False)))
        # Loaded ONCE for the whole pull (a checkpoint read + ViT build per plane would
        # dominate a time series); the kernel keys its singleton on (model, path, device).
        # This ONE object is then shared by every lane: CellSAM runs on threads, not
        # processes (`_segment_backend`), so "once per pull" stays literally true instead of
        # becoming once per worker.
        model = get_cellsam_model(model_id, model_path=weights)
        if weights:
            # `model_path` WINS inside the kernel (get_local_model), so the provenance must
            # name the checkpoint that actually ran, not the published-model socket the
            # loader ignored.
            model_id = weights

    # ── the unit loop: one segmentation per plane (2D) / per volume (3D) ───────
    def segment_unit(arr: np.ndarray, unit: tuple) -> np.ndarray:
        """One prepared unit → a label array of the same shape, ids contiguous from 1."""
        m, t, c = unit[0], unit[1], unit[-1]
        z = unit[2] if len(unit) == 4 else None
        if method == "stardist":
            if is_3d:
                # Genuinely volumetric: StarDist3D's polyhedra are z-connected, so `arr` is
                # the whole (Z,Y,X) volume and the ids that come back already span z.
                return np.asarray(
                    segment_volume(arr.astype(np.float32), model=model, prob_thresh=prob,
                                   nms_thresh=nms, scale=scale, scale_z=scale_z)[0],
                    dtype=np.int64)
            return np.asarray(segment_frame(arr.astype(np.float32), model=model,
                                            prob_thresh=prob, nms_thresh=nms,
                                            scale=scale)[0], dtype=np.int64)
        if method == "cellsam":
            # `progress_cb` ONLY when tiling: that is the one branch with a real inner loop
            # (a dask graph over blocks) and so the one branch whose percentage means
            # anything. Un-tiled, the kernel could only report "started"/"finished", and
            # handing that to a determinate bar would freeze it at 8% for the whole
            # inference — `unit_started`'s sweep is the honest rendering of that case.
            return np.asarray(
                segment_plane(arr, model=model, **cs,
                              progress_cb=(unit_sub if (solo and cs.get("tile")) else None)),
                dtype=np.int64)
        if mask6 is not None:
            fg = (mask6[m, t, :, c] if z is None else mask6[m, t, z, c]) != 0
        else:
            fg = arr > _segment_level(arr, level, fixed)
            if arr.dtype.kind == "f":
                # A non-finite voxel is "no valid measurement here", so it is background —
                # never an object. NaN falls out on its own (`nan > level` is False) but
                # `inf > level` is True, and an inf from a deconvolution or a 0/0 would
                # otherwise appear as a phantom one-voxel cell in the table. Treating the
                # two the same is the only defensible reading; the guard is skipped on an
                # integer image, which cannot carry either.
                fg &= np.isfinite(arr)
        if method == "watershed":
            return _segment_watershed_split(fg, sampling, footprint)
        return np.asarray(label_components(fg, conn)[0], dtype=np.int64)

    # The label raster is the one output that cannot be made lazy even in principle: ids are
    # GLOBALLY unique, assigned by folding the units in order, so plane n's labels depend on
    # how many objects the previous n-1 units found. It is therefore always a dense array
    # over the whole grid — 42.3 Gvoxel on the lab's 640 series, which is 315 GiB at int64
    # and an unconditional `_ArrayMemoryError` (2026-08-04, the failure that read as an
    # upstream Z-Project's `none` being broken). Above `spill_budget` it is written through a
    # memmapped .npy instead, which the layer keeps without copying and the Memo GC counts
    # as zero (nodegraph.spill). `fold` assigns into it exactly as before either way, and the
    # fold stays serial and ordered, so ids and rows are byte-identical.
    raster_out = dense_output((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x), np.int64,
                              tag=f"labels_{ctx.node_id}")
    raster = raster_out.array
    cols: Dict[str, list] = defaultdict(list)
    offset = 0

    def take(lab: np.ndarray, m: int, t: int, c: int, z_index: int) -> np.ndarray:
        """Clean up one unit's labels, append its Label rows, and shift its ids into the
        globally-unique range."""
        nonlocal offset
        lab = _segment_size_filter(_segment_fill_holes(lab) if fill else lab,
                                   min_px, max_px)
        tbl = _labeled_table(lab, m=m, t=t, c=c, layer=layer, is_3d=is_3d,
                             z_index=z_index)
        for k, v in tbl.columns.items():
            cols[k].extend(((v + offset) if k == "id" else v).tolist())
        shifted = np.where(lab > 0, lab + offset, 0)
        offset += tbl.n
        return shifted

    # ── run it: parallel per-unit segmentation → SERIAL ORDERED fold (V2.14) ───
    #
    # `segment_unit` is pure — one unit in, one label array out — but `take` is NOT: it
    # advances a global id `offset` and appends Label rows. So the ids a unit receives
    # depend on how many objects every EARLIER unit found, which makes fold order part of
    # the result. Splitting the loop in two keeps the expensive half parallel and the
    # order-bearing half exactly as it was: `fold_units` preserves submission order, so
    # the ids, the row order and the raster are byte-identical to the serial loop.
    note = "segmenting (%s)" % method
    units = ([(m, t, c) for m in range(ax.m) for t in range(ax.t)
              for c in range(ax.c)] if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    n_units = len(units)

    def read_unit(unit: tuple) -> np.ndarray:
        if is_3d:
            m, t, c = unit
            return prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x)
        m, t, z, c = unit
        return prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

    backend = _segment_backend(method)
    prepare = None
    if backend == "process" and method == "stardist":
        worker = _seg_worker_stardist
        prepare = lambda u: (read_unit(u).astype(np.float32), model_id,   # noqa: E731
                             prob, nms, scale, _cpu)
    else:
        def worker(unit: tuple) -> np.ndarray:      # threads / serial: no pickling at all
            # sweep the sub bar while THIS unit's read + inference runs; without it the bar
            # would not move again until the unit lands, which on CellSAM is tens of seconds
            unit_started()
            return segment_unit(read_unit(unit), unit)

    # In-flight results are bounded by the batch, so peak memory stays at the output
    # raster plus a few units instead of a second copy of the whole series (the eager
    # full-raster hazard). Sized off the machine-derived cache budget.
    unit_bytes = max(1, (ax.z if is_3d else 1) * ax.y * ax.x * 8)
    lanes = 1 if backend == "serial" else max(
        1, min(cpu_budget(), int(tile_cache_bytes() // unit_bytes)))
    if backend == "process":
        # A SECOND cap, on a resource the one above cannot see. `tile_cache_bytes // unit`
        # bounds the in-flight PIXEL data; a process worker running a learned segmenter also
        # holds a whole model — its own copy, because `_seg_worker_stardist` loads through
        # the worker's module singleton (that is the design: an nn.Module/Keras graph cannot
        # be pickled per task). So peak RSS is lanes × model, a term absent from the pixel
        # arithmetic, and on a 12-core box the pixel cap alone happily asks for 10 workers ×
        # a TensorFlow process each.
        lanes = max(1, min(lanes, _seg_proc_lane_cap(method)))

    # ── two-level progress, including INSIDE one unit (V2.17) ─────────────────
    # The fold ticks once per FINISHED unit, and on a learned segmenter a unit is tens of
    # seconds — long enough that a bar which only moves on completion reads as a hang. So a
    # unit also reports when it STARTS (sweeping the sub bar: "working, position unknown"),
    # and CellSAM's *tiled* path reports its dask blocks as they land, which is the one place
    # in either segmenter where genuine per-iteration progress exists. Un-tiled CellSAM and
    # StarDist are a single opaque inference call with no upstream hook, so they sweep rather
    # than pretend to a percentage.
    #
    # Gated on ONE IN-PROCESS lane, and both halves matter: with several planes in flight a
    # sub-fraction cannot be attributed to any single unit (the fold is waiting on one
    # specific unit, not on whichever worker last reported), and a process worker cannot
    # reach `ctx` at all. With the gate off, the streamed fold ticks ARE the signal —
    # `fold_units` folds each result as it lands precisely so they arrive one per unit
    # instead of a batch at a time.
    bar = _UnitBar(ctx, frames=ax.t, units_per_frame=n_units // max(1, ax.t), note=note)
    solo = lanes == 1 and backend != "process"

    def unit_started() -> None:
        if solo:
            bar.emit_unknown()

    def unit_sub(pct: int) -> None:
        if solo:
            bar.emit(pct)

    def fold(i: int, unit: tuple, lab: np.ndarray) -> None:
        m, t, c = unit[0], unit[1], unit[-1]
        z = unit[2] if len(unit) == 4 else None
        if z is None:
            raster[m, t, :, c] = take(lab, m, t, c, 0)
        else:
            raster[m, t, z, c] = take(lab, m, t, c, z)
        bar.finish_unit()

    fold_units(worker, units, fold, prepare=prepare,
               proc=(backend == "process"), workers=lanes, batch=lanes)

    out = ds.with_layer(Domain.VOXEL, layer, raster_out.seal())
    if cols.get("id"):
        out = out.with_structure(StructureTable(
            Domain.LABEL, {k: np.array(v) for k, v in cols.items()},
            layer=layer, z_kind=("subpixel" if is_3d else "plane_index")))
    # Provenance (§7b), the `track.objects` shape: namespaced non-calibration keys naming
    # HOW these labels were made, so a downstream node or a reader can tell a CNN
    # segmentation from a threshold without re-deriving it.
    prov_md = {"segment_method": method}
    if model_id:
        prov_md["segment_model"] = model_id
    if method == "stardist":
        # The thresholds that ACTUALLY ran, whether they came from a pinned socket or the
        # checkpoint's own thresholds.json (V2.23). Stamped because the socket alone no
        # longer answers the question: on auto it holds nothing, and the value in force is a
        # property of the model file. A reader comparing two runs — or a future self asking
        # why one graph found 30% more nuclei — needs the number, not the provenance of the
        # number. Namespaced non-calibration keys, the `track.objects` shape (wire-node-v2 §7b).
        if eff_prob is not None:
            prov_md["segment_prob_thresh"] = eff_prob
        if eff_nms is not None:
            prov_md["segment_nms_thresh"] = eff_nms
    return out.with_metadata(**prov_md)
#: The pretrained checkpoint names each learned backend REGISTERS — closed, published sets,
#: which is why the three sockets below declare them as ``choices`` (a dropdown) rather than
#: taking free text. These names are not ours to invent: each one is a key into the
#: backend's own model zoo, so a typo is not a validation error the graph can report but a
#: download that 404s at pull time, several seconds into a run. The sets are small and
#: stable enough to name here; a model of your OWN goes through the local-path socket
#: beside each one, which overrides the choice entirely.
_STARDIST_2D_MODELS: Tuple[str, ...] = (
    "2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018", "2D_demo")
_STARDIST_3D_MODELS: Tuple[str, ...] = ("3D_demo",)
_CELLSAM_MODELS: Tuple[str, ...] = ("cellsam_general", "cellsam_extra")

def _columns_segment(params, modes, incoming):
    """The invariant Label schema every segmentation backend emits (V2.28). Independent of
    the `method` lever: the table is assembled by `structure._label_table` whichever
    detector found the regions. Total by contract (runs on every keystroke)."""
    return on_layer(Domain.LABEL, str((params or {}).get("name") or "labels"),
                    LABEL_INVARIANT)

register_node(
    batch_aware(_compute_segment), op_key="analysis.segment", label="Segmentation",
    category="analysis",
    adds_columns=_columns_segment,
    # STATIC on purpose, and left alone by the V2.22 per-mode sweep: this is the image
    # domain every source already supplies (`io.load` adds VOXEL), not a layer requirement
    # that varies by method. The only method-specific Voxel layer is watershed's `mask`,
    # and its default is `""` — "cut the foreground out of the image with the level
    # controls" — so no branch demands anything the others do not.
    reads_domains=frozenset({Domain.VOXEL}),
    adds_domains=frozenset({Domain.VOXEL, Domain.LABEL}),   # a label RASTER and a table
    inputs=[
        InDataset(),
        # ── shared by every method ────────────────────────────────────────────
        InString("name", "Output layer", field=False, default="labels",
                 layer_out=(Domain.VOXEL, Domain.LABEL),
                 description=
                 "Name of the layer this node writes — one name into two domains: a Voxel "
                 "label raster and a Label table (id, m, t, c, area, z, y, x, with area in "
                 "voxels). Downstream nodes select this result by name, so two segmentation "
                 "nodes in one graph need two different names or the later one shadows the "
                 "earlier."),
        InBool("fill_holes", "Fill holes", field=False, default=False,
               description=
               "Fill enclosed background holes inside each object, before the size filter. "
               "Only voxels that are currently background get painted, so filling can never "
               "steal a voxel from a neighbouring cell or move a shared boundary. For "
               "CellSAM this sits on top of upstream's own always-on hole filling and "
               "usually changes little; it earns its keep on a whole-cell mask with a "
               "genuine interior gap (a vacuole, an unstained nucleus). It can only "
               "INCREASE a reported area."),
        # A 2D object has an area and a 3D object has a volume — different units, so
        # different sockets rather than one whose meaning silently changes with the lever.
        InFloat("min_area", "Min area", unit="um2", field=False, default=0.0,
                pick_kind="area", pick_peer="max_area",
                available_in={"dim": frozenset({"2D"})},
                description=
                "Discard objects smaller than this, in µm² — converted to pixels with "
                "pixel_size_um from the file's metadata, so one value means the same "
                "physical cell at any magnification. 0 = no lower bound. Applied AFTER the "
                "method runs. For CellSAM it stacks on an unconditional 25-PIXEL floor "
                "inside upstream, so the smallest surviving object is whichever of the two "
                "is stricter. Survivors are renumbered contiguously, so this changes ids as "
                "well as membership."),
        InFloat("max_area", "Max area", unit="um2", field=False, default=0.0,
                pick_kind="area", pick_peer="min_area",
                available_in={"dim": frozenset({"2D"})},
                description=
                "Discard objects larger than this, in µm² (converted with pixel_size_um; "
                "0 = no upper bound). In practice this throws away merged clumps and debris "
                "rather than cells — with CellSAM it is the usual answer to two touching "
                "cells that came back as one box. Applied after the method runs; survivors "
                "are renumbered contiguously."),
        InFloat("min_volume", "Min volume", unit="um3", field=False, default=0.0,
                pick_kind="area", pick_peer="max_volume",
                available_in={"dim": frozenset({"3D"})},
                description=
                "Discard objects smaller than this, in µm³ — converted with pixel_size_um "
                "and z_step_um, so it stays physical under anisotropic voxels. 0 = no lower "
                "bound. 3D only, so it never applies to StarDist or CellSAM: both are "
                "2-D-per-plane detectors and refuse the 3D lever."),
        InFloat("max_volume", "Max volume", unit="um3", field=False, default=0.0,
                pick_kind="area", pick_peer="min_volume",
                available_in={"dim": frozenset({"3D"})},
                description=
                "Discard objects larger than this, in µm³ (converted with pixel_size_um and "
                "z_step_um; 0 = no upper bound). 3D only, so it never applies to StarDist or "
                "CellSAM, which refuse the 3D lever."),
        # ── threshold + watershed: the foreground cut ─────────────────────────
        # a FIXED level is in the image's own units, so its default follows the declared
        # intensity scale: mid-range of the current bit depth (2047.5 on 12-bit), falling
        # back to 0.5 when there is no declared integer scale — i.e. exactly after a
        # Normalize dropped it (wire-node-v2 §7c).
        InFloat("threshold", "Level", unit="", field=False, default=0.5,
                pick_kind="level",
                derive="((2**bit_depth - 1)/2) if bit_depth else 0.5",
                available_in={"method": frozenset({"threshold", "watershed"}),
                              "level": frozenset({"fixed"})},
                description=
                "The foreground cut for the classical methods: a voxel belongs to an object "
                "when it is STRICTLY GREATER than this. In the image's OWN intensity units, "
                "so auto starts at the mid-point of the declared bit depth (2047.5 on 12-bit) "
                "and falls back to 0.5 for [0,1] data — i.e. after a Normalize. Raising it "
                "shrinks every object, splits ones joined by a dim bridge, and drops the "
                "faintest entirely; lowering it merges neighbours. Only shown when the Level "
                "mode is `fixed`: the automatic levels (otsu and friends) derive the cut from "
                "the histogram and ignore this. Not used by the learned methods at all."),
        # ── threshold only ───────────────────────────────────────────────────
        InInt("connectivity", "Connectivity", unit="", field=False, default=0,
              available_in={"method": frozenset({"threshold"})},
              description=
              "Which neighbours count as touching when grouping foreground voxels into "
              "objects, so it decides whether a diagonal contact makes one object or two. In "
              "2D: 4 = edge-sharing only, 8 = edges + corners. In 3D: 6 = faces, 18 = faces + "
              "edges, 26 = everything. HIGHER merges more, which lowers the object count and "
              "raises individual areas — the usual reason two nearly-touching cells come back "
              "as one. 0 takes the default (8 in 2D, 26 in 3D). The `watershed` method exists "
              "precisely to split objects that connectivity cannot separate."),
        # ── watershed only ───────────────────────────────────────────────────
        # EMPTY by default: a Segmentation node segments the image. Naming a layer here
        # splits THAT foreground instead (the folded-in `analysis.watershed` behaviour).
        InString("mask", "Foreground layer", field=False, default="",
                 layer_in=Domain.VOXEL,
                 available_in={"method": frozenset({"watershed"})},
                 description=
                 "Optional: split an EXISTING mask layer instead of thresholding this node's "
                 "own image. Leave EMPTY — the default — and the node thresholds the incoming "
                 "image itself, which is what you want when Segmentation is doing the whole "
                 "job. Name a layer when the foreground was already produced upstream (by "
                 "Threshold, ROI Mask, or Histogram Threshold) and you only want watershed to "
                 "cut it apart; the Level control then plays no part."),
        InFloat("min_distance", "Min seed distance", unit="um", field=False, default=0.3,
                pick_kind="distance",
                available_in={"method": frozenset({"watershed"})},
                description=
                "Minimum spacing between watershed seeds, in microns — in practice, THE "
                "over/under-splitting knob. Seeds are the peaks of the distance transform, so "
                "this says how close two object centres may be before they are treated as one "
                "object. TOO SMALL and every bump in a cell's outline seeds its own fragment "
                "(over-segmentation); TOO LARGE and genuinely touching cells merge. Start near "
                "the radius of your smallest object and adjust from the result. Physical, so "
                "it survives a change of magnification."),
        # ── stardist only ────────────────────────────────────────────────────
        # StarDist's own parameter semantics, from `StarDist2D.predict_instances`'s docstring
        # and the model registry in `stardist/models/__init__.py` (github.com/stardist/stardist).
        InFloat("prob_thresh", "Prob threshold", unit="", field=False, default=0.5,
                available_in={"method": frozenset({"stardist"})},
                description=
                "Only pixels whose predicted object probability exceeds this become candidate "
                "nuclei — the detection sensitivity knob. LOWER finds more objects, including "
                "faint and spurious ones; HIGHER keeps only confident detections and misses dim "
                "nuclei. Lower it first if a well-focused image returns too few nuclei. "
                "ON AUTO it takes the value the loaded checkpoint was TRAINED with — every "
                "StarDist model directory ships a thresholds.json that its author tuned against "
                "their own validation set, and that is the number the network is meant to run "
                "at (the registered `3D_demo` says 0.708, not 0.5). Pin it only to deviate from "
                "that deliberately; the value in force is recorded on the output as "
                "segment_prob_thresh either way."),
        InFloat("nms_thresh", "NMS threshold", unit="", field=False, default=0.3,
                available_in={"method": frozenset({"stardist"})},
                description=
                "Non-maximum suppression treats two candidate shapes as the SAME object when "
                "their area overlap exceeds this. So it decides how much overlap neighbouring "
                "nuclei may have before one is discarded: LOWER suppresses aggressively and can "
                "delete a genuinely touching or partly-overlapping nucleus, HIGHER keeps more "
                "and risks reporting one nucleus twice. Raise it for densely packed samples — "
                "separating crowded nuclei is what StarDist's star-convex shapes are for. Like "
                "Prob threshold it comes from the checkpoint's own thresholds.json ON AUTO, and "
                "StarDist's fallback when a model states none is 0.4 rather than the 0.3 shown "
                "here as the pinned starting point."),
        InFloat("scale", "Scale", unit="", field=False, default=0.0,
                available_in={"method": frozenset({"stardist"})},
                description=
                "Resize the image internally by this factor before prediction, then map the "
                "result back to the original resolution. 0 disables it. It exists because a "
                "pretrained model learned nuclei at ONE apparent size: if yours are much larger "
                "or smaller in pixels, scaling them toward the training scale recovers accuracy "
                "far more cheaply than retraining. Set it to (training nucleus diameter) / "
                "(your nucleus diameter) — roughly 0.5 when your nuclei look twice as big as the "
                "model expects. Output geometry is unaffected, so measurements stay in your "
                "image's own coordinates."),
        InString("model_name", "StarDist model", field=False,
                 choices=_STARDIST_2D_MODELS,
                 default="2D_versatile_fluo",
                 available_in={"method": frozenset({"stardist"}),
                               "dim": frozenset({"2D"})},
                 description=
                 "Which pretrained 2-D checkpoint to load. The registered ones are "
                 "`2D_versatile_fluo` (fluorescent nuclear markers — the default and the right "
                 "choice for DAPI/Hoechst-style data), `2D_versatile_he` (brightfield H&E "
                 "histology), `2D_paper_dsb2018` (the original publication's DSB-2018 model) and "
                 "`2D_demo` (a small toy model, not for real data). Weights download on first "
                 "use. Ignored when a local model directory is set. 2D only — a 2-D checkpoint "
                 "cannot load into StarDist3D, which is why 3D has its own socket.",
                 choice_docs={
                     "2D_versatile_fluo":
                         "Trained on a broad mix of FLUORESCENT nuclear images — DAPI, Hoechst, "
                         "H2B and friends. The default and the right first choice for almost "
                         "all fluorescence work; it expects bright nuclei on a dark background, "
                         "so inverted (brightfield) contrast confuses it.",
                     "2D_versatile_he":
                         "Trained on H&E-stained BRIGHTFIELD histology, where nuclei are dark "
                         "purple on pale tissue. Use it for stained sections; on fluorescence "
                         "it finds little, because the contrast polarity it learned is the "
                         "opposite one.",
                     "2D_paper_dsb2018":
                         "The original publication's model, trained on the Data Science Bowl "
                         "2018 nuclei set. Keep it for reproducing paper numbers or comparing "
                         "against published results; `2D_versatile_fluo` generalizes better on "
                         "new data.",
                     "2D_demo":
                         "A tiny toy model shipped so the code path can be exercised without "
                         "a download. Treat any output as a smoke test — it is not trained "
                         "for real data and its objects should not be measured.",
                 }),
        InString("model_name_3d", "StarDist 3D model", field=False, default="3D_demo",
                 choices=_STARDIST_3D_MODELS,
                 available_in={"method": frozenset({"stardist"}),
                               "dim": frozenset({"3D"})},
                 description=
                 "Which pretrained 3-D checkpoint to load. **`3D_demo` is the only one "
                 "StarDist registers, and it is a DEMO** — trained on a small toy dataset, so "
                 "treat any result from it as a smoke test, not a measurement. For real "
                 "volumetric work train a model on your own data (the WACV 2020 paper's "
                 "procedure) and point the local model directory at it instead; that socket "
                 "overrides this one. 3D only.",
                 choice_docs={
                     "3D_demo":
                         "The ONLY 3-D checkpoint StarDist registers, and it is a demo — "
                         "trained on a small toy volume. It proves the 3-D path runs; it does "
                         "not segment real specimens. For volumetric work train your own model "
                         "and set the local model directory, which overrides this.",
                 }),
        InString("sd_model_path", "Local model dir", field=False, default="",
                 available_in={"method": frozenset({"stardist"})},
                 path_kind="directory",
                 path_hint="empty = pretrained · or Browse…",
                 description=
                 "Path to a locally TRAINED StarDist model directory — the folder holding "
                 "`config.json` and `weights_best.h5`. Overrides the pretrained name above and "
                 "skips the download. Its own `config.json` supplies the ray count and training "
                 "anisotropy, so a model trained at any geometry loads without restating "
                 "anything here. Empty = use the pretrained name. This is the ONLY route to "
                 "usable 3-D segmentation, since the sole registered 3-D checkpoint is a demo. "
                 "When set, this path (not the model name) is what the segment_model provenance "
                 "records."),
        InFloat("scale_z", "Scale Z", unit="", field=False, default=0.0,
                available_in={"method": frozenset({"stardist"}),
                              "dim": frozenset({"3D"})},
                description=
                "Axial resize factor, separate from the lateral Scale so ANISOTROPIC volumes "
                "can be corrected — the single most important 3-D control after the model "
                "itself. A network learned nuclei at one apparent aspect ratio, and confocal "
                "voxels are typically several times taller than they are wide, so the same "
                "nucleus arrives squashed along z and the polyhedra fit badly. Set it to "
                "z_step_um / pixel_size_um (times any lateral scale) to present near-isotropic "
                "objects. 0 follows the lateral Scale, which is right only for already-isotropic "
                "data. Output geometry is unaffected — measurements stay in your own voxel "
                "coordinates. 3D only."),
        # ── cellsam only (socket names stay disjoint from stardist's: the node card
        #    relayouts on the active socket NAME list, so a same-named socket with a
        #    different default would not redraw when the method changes) ─────────
        InFloat("bbox_threshold", "Box threshold", unit="", field=False, default=0.4,
                available_in={"method": frozenset({"cellsam"})},
                description=
                "CellFinder's box-confidence cut, and the main precision/recall knob for "
                "this method. CellFinder proposes up to 3,500 boxes per pass; every box "
                "that clears this confidence is prompted through SAM's mask decoder and "
                "becomes at most one cell — so LOWER finds more cells and more false "
                "positives, HIGHER finds fewer and cleaner ones. It is not applied raw: "
                "CellSAM k-means-clusters (k=2) the box confidences of each image and "
                "blends them with this value, T_box = 0.66·T + 0.33·T_cluster, so the "
                "effective cut adapts per image and this socket shifts it rather than "
                "setting it. The paper uses 0.4 for every dataset it reports; lower it "
                "first when an out-of-distribution image comes back empty."),
        # The paper names THREE inference thresholds (Nat. Methods 22:2585, Methods →
        # Thresholding); `bbox_threshold` above is the first. These are the other two.
        # Upstream's `segment_cellular_image` takes no keyword for either — they are read
        # off the model inside `CellSAM.predict` — so the kernel assigns them onto the model
        # on every call. Defaults are the SHIPPED values, not the paper's, so exposing them
        # reproduces existing results bit-for-bit; where the two disagree the socket
        # description says so.
        InFloat("mask_threshold", "Mask cut", unit="", field=False, default=0.4,
                available_in={"method": frozenset({"cellsam"})},
                description=
                "The per-pixel sigmoid cut on the mask decoder's output — in effect, how "
                "far each cell's mask extends from its centre. LOWER grows every mask, "
                "HIGHER shrinks it, so this is the knob that moves reported µm² areas "
                "without changing WHICH cells are found (that is Box threshold). Range "
                "(0,1); 0 would keep every pixel and 1 nothing, and both ends are refused. "
                "The default 0.4 is the value CellSAM actually ships, which is why it is "
                "the default here — but the paper's Methods state 0.5 for the published "
                "results, so set 0.5 if you are reproducing the publication or comparing "
                "against its numbers."),
        InFloat("mask_quality", "Min mask quality", unit="", field=False, default=0.5,
                available_in={"method": frozenset({"cellsam"})},
                description=
                "The smallest predicted mask quality a detection may have and still be "
                "kept — SAM's mask decoder scores its own output with an IoU-prediction "
                "head, and anything below this is dropped outright. A SECOND recall knob, "
                "independent of Box threshold: a box can clear the confidence cut and still "
                "be discarded here, which is the usual explanation for cells that are "
                "visibly detected but missing from the output. Raise it to shed ragged, "
                "low-confidence masks; lower it to keep them. 0.5 in both the shipped code "
                "and the paper."),
        InString("cellsam_model", "CellSAM model", field=False,
                 choices=_CELLSAM_MODELS,
                 default="cellsam_general",
                 available_in={"method": frozenset({"cellsam"})},
                 description=
                 "Which published checkpoint to load. 'cellsam_general' was trained only "
                 "on the datasets in the paper — use it to reproduce published numbers. "
                 "'cellsam_extra' adds further training data and upstream recommends it "
                 "for domains beyond the paper, at the cost of no longer matching the "
                 "published benchmarks. Ignored when Local weights is set. Weights land "
                 "in ~/.deepcell/models on first use (a DeepCell API token is needed once; "
                 "loading is offline afterwards) and are licensed for non-commercial "
                 "academic use. Whichever model runs is recorded in the output's "
                 "segment_model provenance.",
                 choice_docs={
                     "cellsam_general":
                         "The paper's checkpoint, trained only on the datasets reported there. "
                         "The one to use when the result has to match published CellSAM "
                         "benchmarks, or when you want the behaviour the paper's figures "
                         "describe.",
                     "cellsam_extra":
                         "The same architecture with additional training data. Upstream "
                         "recommends it for domains beyond the paper — the option to try when "
                         "`cellsam_general` underperforms on your modality — with the explicit "
                         "cost that its numbers no longer correspond to the published "
                         "benchmarks.",
                 }),
        InString("model_path", "Local weights", field=False, default="",
                 available_in={"method": frozenset({"cellsam"})},
                 path_kind="open_file",
                 path_filter="PyTorch checkpoint (*.pt *.pth);;All files (*)",
                 path_hint="empty = published · or Browse…",
                 description=
                 "Path to a local CellSAM .pt checkpoint. Takes priority over CellSAM "
                 "model and skips the DeepCell download and API token entirely — the "
                 "offline route, and the way past a corporate proxy or antivirus TLS "
                 "interception that blocks the fetch. Empty = use the published checkpoint "
                 "named above. When set, this path rather than the model name is what the "
                 "segment_model provenance records."),
        InBool("normalize", "Normalize", field=False, default=True,
               available_in={"method": frozenset({"cellsam"})},
               description=
               "Run CellSAM's own preprocessing: clip at the 99.9th percentile, rescale "
               "each channel to [0,1], then CLAHE with a 128-px kernel. LEAVE THIS ON for "
               "raw microscopy — it is not an optional refinement. The detector branch "
               "converts the image with ToPILImage, which multiplies by 255 and casts to "
               "uint8 with NO clipping, so a plane whose values exceed 1.0 (i.e. every raw "
               "16-bit ND2 plane) wraps into noise, CellFinder proposes nothing, and the "
               "result is a silently EMPTY segmentation rather than an error. Turn it off "
               "only when the incoming plane is already scaled to [0,1] — for example "
               "after a Normalize node."),
        InBool("postprocess", "Postprocess", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})},
               description=
               "Extra per-cell morphological cleanup inside CellSAM: drop small holes and "
               "islands, open then close with a 2-px disk, dilate then erode with a 10-px "
               "disk, blur at sigma=3 and re-threshold. Upstream recommends it for noisy "
               "images. It SMOOTHS and slightly INFLATES every mask, so it moves the areas "
               "this node reports and therefore the µm² filter above. This is NOT the "
               "paper's 'CellSAM postprocessing' (hole filling + island removal), which "
               "always runs whatever this is set to. Leave it off unless the images are "
               "noisy: on a prediction containing no non-zero label, upstream raises a "
               "zero-size-array error."),
        InBool("remove_boundaries", "Separate touching", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})},
               description=
               "Erode a one-pixel gap between touching cells so no two labels share a "
               "border — use it when downstream work needs provably separated objects. It "
               "shrinks every cell by roughly one pixel of perimeter, which lowers every "
               "reported area slightly and hits small cells hardest."),
        # Tiling is a memory/compute knob in the MODEL's own pixel space (CellSAM resizes
        # every tile to 1024²), not a physical extent — hence `px`, declared rather than
        # hidden as a bare constant. `tile_size`/`tile_overlap` stay visible while `tile`
        # is off because liveness that depends on another SOCKET's value cannot be
        # expressed in `available_in`, which sees mode state only (wire-node-v2 §5b).
        InBool("tile", "Tiled inference", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})},
               description=
               "Segment each plane as overlapping blocks stitched back together by label "
               "IoU, instead of in one pass. Two reasons to switch it on: peak memory on a "
               "large FOV, and CellFinder's hard ceiling of 3,500 detection queries per "
               "pass (the paper sized it at ~3.5x the 1,000 cells it expects per image) — "
               "past roughly 3,000 cells in one field real cells simply go undetected, and "
               "tiling is the only fix. Two costs: the blocks run SEQUENTIALLY, so this "
               "bounds memory without shortening the run, and a block that fails is "
               "replaced with empty background and logged rather than raised — so a broken "
               "configuration shows up as missing cells, not as an error."),
        InInt("tile_size", "Tile size", unit="px", field=False, default=512,
              pick_kind="grid", pick_peer="tile_overlap",
              available_in={"method": frozenset({"cellsam"})},
              description=
              "Block edge, in PIXELS rather than µm on purpose: CellSAM resizes every block "
              "to 1024² internally, so tile geometry belongs to the model's input space and "
              "to memory, not to physical extent. The paper trained on 512-px tiles "
              "upsampled to 1024², so the 512 default keeps inference at the training "
              "scale; smaller blocks mean more of them (slower, more seams to stitch) but "
              "fewer cells per pass. Only read when Tiled inference is on. Values below 64 "
              "are clamped up."),
        InInt("tile_overlap", "Tile overlap", unit="px", field=False, default=56,
              pick_kind="grid", pick_peer="tile_size",
              available_in={"method": frozenset({"cellsam"})},
              description=
              "How far neighbouring blocks overlap, in PIXELS. It must be wider than a "
              "typical cell: a cell crossing a seam is only rejoined if it appears whole "
              "inside the overlap, and this same value is reused as the IoU-matching depth "
              "that decides whether the two halves are one cell (they merge above IoU 0.5). "
              "Larger is safer but costs redundant compute — each block is really "
              "tile_size + 2×overlap px. Only read when Tiled inference is on. Clamped to "
              "[1, tile_size − 1]."),
        # NOT named `iou_threshold`: three distinct IoU quantities are now reachable from
        # this catalog (this tile-merge cut, `mask_quality`'s predicted-mask IoU, and
        # `track.objects`' own `iou_threshold` socket), and the bare name distinguished none
        # of them.
        InFloat("tile_iou", "Tile merge IoU", unit="", field=False, default=0.5,
                available_in={"method": frozenset({"cellsam"})},
                description=
                "When stitching tiles, how much two labels on either side of a seam must "
                "overlap to be declared the same cell. LOWER merges more eagerly — too low "
                "and two neighbouring cells that straddle a seam fuse into one; too high "
                "and a single cell split by a seam stays two, each a partial mask with a "
                "wrong area. Only read when Tiled inference is on, and only matters where "
                "blocks meet, so it cannot affect a run that fits in one pass. 0.5 is "
                "upstream's default."),
        InBool("fast", "Fast inference", field=False, default=False,
               available_in={"method": frozenset({"cellsam"})},
               description=
               "Run CellSAM's mask decoder on batches of 32 cell proposals instead of one "
               "at a time, and upsample the masks on the GPU instead of the CPU. Measured "
               "5.7x faster on a 1024² block of 441 cells (10.1 s → 1.8 s, RTX 3090): "
               "upstream calls the decoder once per detected cell, which leaves the GPU "
               "idle waiting on launch overhead. A 26-minute stitched-mosaic run becomes "
               "about 5 minutes. NOT bit-identical, which is why it is off by default — "
               "batched matmuls reduce in a different order, so a logit can land the other "
               "side of the mask cut. On that block 11 of 443 cells changed area, each by "
               "EXACTLY 1 pixel against a median cell of 556 px (0.18% worst case), with no "
               "cell gained, lost or renumbered; that is ~100x smaller than moving "
               "mask_threshold one step, but it is not zero, so leave it OFF to reproduce a "
               "published number and turn it ON while exploring. Off, this node calls "
               "upstream's own function untouched. Needs a CUDA GPU to be worth anything."),
    ],
    outputs=[OutDataset()],
    modes=[DimMode(),
           Mode("method", list(_SEGMENT_METHODS), default="threshold", label="Method",
                description=
                "Which algorithm finds the objects. Everything around it is shared — hole "
                "filling, the µm²/µm³ size filter, globally-unique ids, the Label table — so "
                "switching method changes only how instances are found, never the shape of "
                "the result, and each method's own controls appear and disappear with it. Two "
                "are classical (they cut a foreground and split it, work anywhere, need no "
                "download) and two are learned detectors (they recognise cells, need weights, "
                "and are far slower per plane).",
                choice_docs={
                    "threshold":
                        "Cut a foreground at the Level, then label each connected blob. The "
                        "simplest and by far the fastest: no model, no training, fully "
                        "predictable. Its limitation is absolute — two cells that TOUCH become "
                        "one object, because nothing here separates them. Use it for "
                        "well-separated objects, beads, or as the sanity check before "
                        "reaching for anything else.",
                    "watershed":
                        "The same foreground, then split at the peaks of the distance "
                        "transform — so touching, convex objects ARE separated, with Min "
                        "distance setting how close two centres may be. The standard answer "
                        "for confluent nuclei. It over-splits elongated or bent objects "
                        "(each lobe gets a peak), and it can take an existing mask layer as "
                        "its foreground instead of thresholding again.",
                    "stardist":
                        "A star-convex CNN: it predicts each object's outline directly, so "
                        "touching cells come apart without any distance heuristic and dim "
                        "objects the histogram would miss are still found. Needs the "
                        "`stardist` package and weights (downloaded on first use). "
                        "Genuinely volumetric in 3D (star-convex polyhedra), but objects it "
                        "cannot represent as star-convex — branched neurons, rings — come out "
                        "wrong by construction.",
                    "cellsam":
                        "A foundation model: SAM's ViT prompted by CellFinder box detections. "
                        "The generalist — it segments cells across modalities with no "
                        "per-dataset tuning and no seeds, which makes it the option to try "
                        "on data the other three keep failing on. In exchange it is the "
                        "slowest by an order of magnitude, needs a checkpoint (and a one-time "
                        "DeepCell token, or local weights), and is 2-D only: 3D is refused "
                        "rather than faked from per-plane labels.",
                }),
           # the foreground cut belongs to the classical methods only — a learned detector
           # never thresholds, so the dropdown is GATED AWAY rather than shown and ignored
           # (V2.12 `ModeSpec.available_in`, the Mode-level half of wire-node-v2 §5b).
           Mode("level", list(_SEGMENT_LEVELS), default="otsu", label="Level",
                available_in={"method": frozenset(_SEGMENT_CLASSICAL)},
                description=
                "How the foreground cut is chosen for the two classical methods — the same "
                "menu as the Threshold node, but derived PER UNIT here (per plane in 2D, per "
                "volume in 3D) rather than over a scope you choose. For one global cut across "
                "the whole series, threshold upstream with Threshold and name that layer in "
                "`mask` instead. Shown only under `threshold` and `watershed`; the learned "
                "detectors never cut a histogram.",
                choice_docs={
                    "otsu":
                        "Maximizes between-class variance — the classic bimodal split and the "
                        "default, because it needs no numbers and works on an unseen image. "
                        "Biased LOW when the objects cover only a few percent of the frame, "
                        "which shows up as background blobs joining the label table.",
                    "li":
                        "Minimum cross-entropy, iteratively. Handles a SPARSE foreground much "
                        "better than Otsu — a few cells in an empty field — so it is the first "
                        "alternative to try when Otsu's masks come out too generous.",
                    "yen":
                        "Yen's maximum-correlation criterion, which usually lands HIGHER than "
                        "Otsu. Masks come out tighter, dim cells drop out of the table "
                        "entirely, and objects shrink — so areas measured downstream shrink "
                        "with them.",
                    "triangle":
                        "Geometric: cuts where the histogram is furthest from the line joining "
                        "its peak to its far end. For a SKEWED single-peak histogram — one "
                        "dominant background mode with a long bright tail — where Otsu's "
                        "two-class assumption does not hold.",
                    "mean":
                        "The unit's mean intensity, used directly. Cheap and predictable, but "
                        "only sensible when objects and background cover comparable area; on a "
                        "mostly-empty plane it lands in the noise and the mask fills with "
                        "background.",
                    "fixed":
                        "Use the Threshold socket's number, in the image's own intensity "
                        "units. The only reproducible choice across planes and files — nothing "
                        "adapts, so a plane that dims yields smaller objects instead of the "
                        "same ones — and the one to use when the cut has to be identical "
                        "everywhere.",
                })],
    granularity=_DIM_GRAN_GLOBAL, kernel_axes=_DIM_KAX,
    # The StarDist thresholds default to what the LOADED CHECKPOINT was trained with rather
    # than to a number this catalog invented (V2.23). See `_stardist_trained`.
    trained_params=_stardist_trained,
    # ...and one line saying WHICH file they came from, or why there are none:
    # an empty answer is legitimate (not downloaded yet, wrong folder picked) and
    # indistinguishable from a broken feature without it (V2.23b).
    trained_note=_stardist_note,
    description="THE segmentation node: image → a Voxel label raster + a Label table, "
                "with the algorithm as a `method` Mode — threshold+CCL, "
                "distance-transform watershed, StarDist (CNN), or CellSAM (SAM + "
                "CellFinder foundation model). Shared across every method: hole filling, "
                "the µm²/µm³ size filter, globally-unique ids and the region table. 2D "
                "segments each plane independently, 3D each volume (the two learned "
                "methods are 2D-per-plane and refuse the 3D lever).")
