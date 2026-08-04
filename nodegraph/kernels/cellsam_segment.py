"""cellsam_segment — CellSAM foundation-model 2-D instance segmentation (thin glue).

PURPOSE
    Segment cells in a SINGLE already-prepared 2-D plane with CellSAM and return a
    contiguous ``int32`` label image of the same ``(H, W)`` shape. This is the
    2-D-per-plane math kernel ONLY: the caller owns all prep (file I/O, Z/channel
    selection, the per-``(m,t,z,c)`` loop, crop, enhancement) and every physical-unit
    decision (the µm² / µm³ size filters) — exactly the division of labour used by
    :mod:`nodegraph.kernels.stardist_segment`.

WHERE THE REAL MATH LIVES
    NOT here, and not in this repo. CellSAM is a SAM ViT-B backbone whose features feed
    (a) **CellFinder**, an Anchor-DETR set-prediction detector that emits one bounding
    box per cell, and (b) a fine-tuned SAM mask decoder that turns those boxes into
    instance masks — i.e. automatic prompt engineering for SAM (Marks, Israel et al.,
    "CellSAM: a foundation model for cell segmentation", *Nature Methods* 22:2585–2593,
    2025; https://github.com/vanvalenlab/cellSAM). All of it lives in the external
    ``cellSAM`` + ``segment_anything`` + ``torch`` packages plus the downloaded
    ``cellsam_general`` / ``cellsam_extra`` weights.

    The glue vendored here is thin and deliberately so:
      * a **process-singleton** model loader, so the weights are read once per process
        instead of once per plane,
      * the ``(H, W)`` → CellSAM channel-slot convention,
      * one ``segment_cellular_image`` call per plane, or ``segment_wsi`` when tiling a
        large FOV,
      * normalization of the upstream quirks below (7 of them, of which 2 and 6 are the
        ones that would otherwise cost a day),
      * a contiguous ``1..K`` relabel so the caller can offset ids globally across planes.

WHY NOT ``cellsam_pipeline``
    ``cellSAM.cellsam_pipeline`` is the documented one-shot entry point, but it calls
    ``get_model()`` **on every invocation**. Per-plane that re-reads the checkpoint for
    every frame of a time series, which dominates the run. It also min-max normalizes
    into ``img`` in place, and its ``low_contrast_enhancement=True`` branch is broken
    upstream (``cellSAM.utils.enhance_low_contrast`` assigns ``model.bbox_threshold``
    with no ``model`` in scope → ``NameError``). So this kernel drives the two lower-level
    entry points directly and keeps the model itself cached.

UPSTREAM QUIRKS THIS KERNEL NORMALIZES (verified against master, cellSAM 0.0.dev1)
    1. ``segment_cellular_image(img, model, ...)`` takes ``model`` as a REQUIRED
       positional argument, even though the project README shows
       ``segment_cellular_image(img, device='cuda')``. The README call raises
       ``TypeError``; we always pass a loaded model.
    2. **The no-cells path is broken upstream, in two layers.**
       (a) ``segment_cellular_image``'s guard is ``if preds is None``, but
       ``CellSAM.predict`` returns the 4-tuple ``(None, None, None, None)`` when no box
       survives — so the guard NEVER fires (upstream issue #98). Execution unpacks the
       tuple and calls ``fill_holes_and_remove_small_masks(None)``, raising
       ``AttributeError: 'NoneType' object has no attribute 'ndim'``. A blank plane, an
       empty FOV or the dark end slices of a z-stack would crash the whole pull, so
       :func:`segment_plane` absorbs exactly that AttributeError into an empty plane —
       repairing the guard, not inventing behaviour ("no cells" IS an empty segmentation,
       which is what upstream's own dead branch returns).
       (b) That dead branch is itself mis-shaped: it returns ``np.zeros(img.shape[1:])``
       where ``img`` has by then been replaced by the ``(1, 3, H, W)`` torch tensor, i.e.
       ``(3, H, W)`` rather than ``(H, W)``. :func:`segment_plane` also collapses any 3-D
       return to one plane, so an upstream fix for (a) lands safely.
       (c) The tiled path never CRASHES from either — ``cellSAM.wsi.segment_chunk`` wraps
       each block in ``try/except Exception`` and substitutes zeros — and that is the
       hazard, not the cure: a block that failed for an unrelated reason (CUDA OOM, a torn
       read) is zeroed just the same, so a populated region is reported as containing no
       cells and the pull SUCCEEDS. Observed in the wild on a stitched 13106² mosaic: 3 of
       196 blocks logged this and there was no way to tell which kind they were.
       :func:`_chunk_error_watch` reads the log record upstream leaves behind — the only
       evidence that survives its ``except`` — and classifies it, so the tiled path now
       holds exactly the line the untiled one does: absorb the no-cells bug, RAISE anything
       else. It also suppresses both messages from the console, because "no cells in this
       block" was never an ``ERROR``.
    3. ``fill_holes_and_remove_small_masks`` (always applied inside
       ``segment_cellular_image``, ``min_size=25`` px) mutates its argument in place.
       Harmless here — the array is upstream-local — but it is why the returned ids are
       already re-numbered and why we never rely on the pre-filter numbering.
    4. ``postprocess=True`` runs ``cellSAM.model.postprocess_predictions``, which ends in
       ``np.max(new_masks, axis=0)`` over a list built from ``np.unique(...)[1:]``. If the
       decoder returned a non-empty prediction that nonetheless contains no non-zero
       label, that list is empty and numpy raises ``ValueError: zero-size array``. We do
       NOT swallow it: a crash from a documented upstream edge is more honest than a
       silently blank plane. Leave ``postprocess`` off unless the images are noisy.
    5. ``postprocess=True`` also FLOODS the log. ``postprocess_predictions`` calls four
       ``skimage.morphology`` functions that scikit-image deprecated in 0.26
       (``binary_opening``/``binary_closing``/``binary_dilation``/``binary_erosion``) and
       calls them once **per cell per plane**, so a 200-cell time series emits thousands of
       identical ``FutureWarning``s and buries every real message.
       :func:`segment_plane` filters exactly those four (see
       :data:`_MORPHOLOGY_DEPRECATION_RE`) around the upstream call and nothing else.
       The 0.28 REMOVAL is the sharper problem and cannot be filtered: cellSAM imports
       those names at MODULE level (``cellSAM/model.py``), so scikit-image >= 0.28 breaks
       ``import cellSAM`` outright — not just this option. :func:`_require_cellsam`
       translates that into a message naming the pin instead of a bare "cannot import
       name" from three frames down.

    6. ``normalize=False`` SILENTLY CORRUPTS THE DETECTOR, and the two failure modes
       compose into "no cells, no error". ``CellSAM.predict`` runs two preprocessing paths
       and only one of them is range-safe:
         * the EMBEDDING path (``prep_2(percentile=True)``) applies
           ``AnchorDETR.transforms.PercentileThreshold``, which ``rescale_intensity``s to
           ``[0,1]`` — range-agnostic, always fine;
         * the BOX path (``sam_bbox_preprocessing(..., percentile=False)``) skips that and
           calls ``torchvision.transforms.ToPILImage()``, whose ``to_pil_image`` does
           ``(npimg * 255).astype(np.uint8)`` with **no clipping**.
       So the ONLY thing that puts the image in the range ``ToPILImage`` assumes is
       ``normalize_image``. With ``normalize=False`` a uint16 plane (max ≈ 4000) becomes
       ``4000·255`` cast to ``uint8`` → wraparound noise → CellFinder proposes no box →
       quirk 2's no-cells path → :func:`segment_plane` returns an empty plane, exactly as
       designed. Nothing raises. The requirement is therefore **not** "match the paper's
       CLAHE" but "already be in ``[0,1]``". Not repaired here: rescaling behind the user's
       back would be inventing preprocessing, and the honest fix is the socket
       documentation on the node (``analysis.segment``'s ``normalize`` description).

    7. ``segment_cellular_image`` declares ``fast: bool = False`` ("batched inference…
       alpha feature") and **never uses it** — it is not forwarded to ``predict``. Not
       exposed here because it does nothing; do not advertise a batched path.

    PAPER vs SHIPPED CODE (checked against Nat. Methods 22:2585–2593, Methods →
    "Thresholding", 2026-07-29). The paper names **three** inference thresholds:
      * CellFinder box confidence **0.4**, dynamically adjusted by k-means (k=2) on each
        image's box confidences: ``T_box = (2/3)·T + (1/3)·T_μ``. Shipped code matches, up
        to writing the weights as ``0.66``/``0.33`` (``sam_inference.py``) — they sum to
        0.99, not 1, which is upstream's rounding, not ours. This is the socket
        ``bbox_threshold``.
      * the mask decoder's IoU-prediction-head score **0.5** — shipped
        ``CellSAM.iou_threshold = 0.5``. **Matches.** Drops a box entirely
        (``predict``'s ``if iou_predictions[0][0] < self.iou_threshold: continue``), so it
        is a second recall knob independent of ``bbox_threshold``.
      * the per-pixel sigmoid cut **0.5** — shipped ``CellSAM.mask_threshold = 0.4``.
        **DOES NOT MATCH THE PAPER.** This is the cut that decides each mask's extent, so
        it moves every reported ``area``; the shipped default is more permissive (larger
        masks) than the published configuration.
    Neither of the last two is settable through ``segment_cellular_image`` — it assigns only
    ``model.bbox_threshold``. **Both are now exposed** as the ``mask_threshold`` and
    ``mask_quality`` parameters, which :func:`segment_plane` assigns onto the model object
    on every call (see the body comment for why "every call" is load-bearing). Their
    defaults are the SHIPPED values (0.4 / 0.5), not the paper's, so exposing them changes
    no existing result; the node's socket description points at 0.5 for anyone reproducing
    the publication. The tile-stitch IoU is ``tile_iou``, renamed from ``iou_threshold``
    precisely so the three IoU-ish quantities in this catalog stay distinguishable.

    Also from the paper, and the reason ``tile`` exists: CellFinder is configured with
    ``num_query_position = 3500`` queries, sized at "3.5 times the maximum number of cells"
    for images "generally no more than 1,000" cells. That is a HARD per-pass ceiling on
    detections — beyond roughly 3,000 cells in one field, cells go undetected and tiling is
    the only fix.

    And one naming trap: the paper's "CellSAM postprocessing" is *hole filling + island
    removal* (Cellpose's), which is ``fill_holes_and_remove_small_masks`` and runs
    **unconditionally**. The ``postprocess`` flag is a DIFFERENT, extra morphological
    cleanup that the paper does not describe.

    Not a quirk but worth recording: ``cellSAM/modelconfig.yaml`` declares
    ``device: cuda``, yet ``AnchorDETR.build_inference`` never reads ``args.device`` and
    ``CellSAM.predict`` takes its device from ``next(self.parameters()).device``. A
    CPU-only torch build therefore works — the model simply stays where it was loaded.

CHANNEL CONVENTION (why a single plane lands in the LAST slot)
    CellSAM was trained on 3-channel input ordered ``(blank, nuclear, whole-cell)``, and
    ``cellSAM.utils.format_image_shape`` right-aligns whatever it is given
    (``out[:, :, -C:] = img``). A 1-channel plane therefore occupies the whole-cell slot,
    which is exactly how the paper handles nuclear-only datasets ("We moved the green
    channel to blue for nuclear-only datasets … to keep the blue channel always
    occupied", Methods → Dataset construction). We pass each plane through as-is and let
    upstream place it, rather than re-inventing the padding.

MODEL WEIGHTS / LICENCE
    ``get_model()`` downloads to ``$HOME/.deepcell/models`` on first use and needs a
    DeepCell API token in ``DEEPCELL_ACCESS_TOKEN`` (https://users.deepcell.org). The
    weights are licensed for **non-commercial academic use**. ``model_path`` bypasses the
    download entirely and loads a local ``.pt`` via ``cellSAM.get_local_model``.

THIRD-PARTY IMPORTS
    ``cellSAM`` / ``torch`` are imported LAZILY inside the loader, so
    ``import nodegraph.kernels.cellsam_segment`` succeeds with neither installed — the
    whole node palette still registers, and only pulling the node raises the install
    hint. numpy is the sole top-level dependency, and availability is probed with
    ``importlib.util.find_spec`` (no import side effects) exactly as
    :mod:`nodegraph.kernels.dic_correlate` probes ``al_dic``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import threading
import warnings
from typing import Callable, Optional, Tuple

import numpy as np

#: CellSAM model singleton — (model, model_path, device) → loaded ``nn.Module``.
#: Loading reads a multi-hundred-MB checkpoint and builds a ViT-B, so it must happen
#: once per process, not once per plane (the same reason ``stardist_segment`` caches).
_cellsam_model = None
_cellsam_key: Optional[Tuple[str, str, str]] = None

#: Environment override for the inference device: ``auto`` (default) | ``cpu`` | ``cuda``.
#: This is an environment knob rather than a node socket for the same reason
#: ``NODELAB_STARDIST_CPU`` is: the model is a process singleton, so a per-graph device
#: control would silently stop taking effect after the first pull, and it would make the
#: memo non-deterministic (identical recipe hash, different device).
DEVICE_ENV = "NODELAB_CELLSAM_DEVICE"

_INSTALL_HINT = (
    "CellSAM is not installed. Install the package and its weights:\n"
    "    pip install git+https://github.com/vanvalenlab/cellSAM.git\n"
    "Model weights download to ~/.deepcell/models on first use and require a DeepCell "
    "API token (non-commercial academic licence):\n"
    "    set DEEPCELL_ACCESS_TOKEN to the token from https://users.deepcell.org\n"
    "Alternatively set the node's `model_path` to a local CellSAM .pt checkpoint, which "
    "skips the download (but still needs the cellSAM package)."
)

#: Raised when the package IS importable but its WEIGHTS could not be obtained — a
#: different problem with a different fix, so it must not print the install hint (which
#: sent one debugging session chasing a phantom missing package while the real cause was
#: TLS interception).
_WEIGHTS_HINT = (
    "The cellSAM package is installed, so this is a WEIGHTS problem, not an install one.\n"
    "FIX IT IN ONE COMMAND:  python scripts/setup_cellsam.py\n"
    "(it fetches + verifies the weights, walks you through the token, and handles the\n"
    "TLS-interception case). What can be wrong:\n"
    "  * no DEEPCELL_ACCESS_TOKEN, or an expired/mistyped one (paste it bare — no <>, no "
    "quotes) — create one at https://users.deepcell.org\n"
    "  * no network, or TLS interception (corporate proxy / antivirus HTTPS scanning). "
    "Python 3.13 verifies strictly, so an AV-generated CA can be rejected even when the "
    "browser and PowerShell accept it — download the archive with a client that uses the "
    "OS trust store and unpack it into ~/.deepcell/models, or `pip install truststore` "
    "and inject it.\n"
    "  * once ~/.deepcell/models/cellsam_v<ver>/<model>.pt exists, loading is OFFLINE and "
    "no token is consulted at all.\n"
    "  * or point the node's `model_path` at a local .pt to bypass all of the above."
)


#: Quirk 5. ``warnings.filterwarnings`` matches this against the START of the message, so
#: it pins the four deprecated ``skimage.morphology`` calls made by upstream's
#: ``postprocess_predictions`` (``cellSAM/model.py`` ~190-196) and nothing else — a cellSAM
#: or skimage warning about anything at all still reaches the user. Suppression is the only
#: lever available: the calls are inside upstream, there is no flag for them, and they fire
#: once per cell per plane. It is NOT a fix — see the 0.28 removal note in quirk 5.
_MORPHOLOGY_DEPRECATION_RE = r"`binary_(opening|closing|dilation|erosion)` is deprecated"

#: ``sklearn.metrics.confusion_matrix`` warns whenever a seam face carries a single label.
#: ``cellSAM.wsi._across_block_label_iou`` calls it on EVERY pair of touching block faces to
#: merge labels across seams, so an all-background face — which is most of them on a mosaic
#: with physical gaps — emits one. Guaranteed noise from a step that then does the right
#: thing (no labels to merge), and it buried the real content of a run 21 lines deep.
_SKLEARN_SINGLE_LABEL_RE = r"A single label was found in 'y_true' and 'y_pred'"

#: The message ``cellSAM.wsi.segment_chunk`` logs for the no-cells bug (quirk 2a). It
#: catches EVERY exception per block and substitutes zeros, so this string is the only
#: thing distinguishing "this block had no cells" from "this block failed and its pixels
#: were dropped" — see :func:`_chunk_error_watch`.
_NO_CELLS_CHUNK_RE = r"'NoneType' object has no attribute 'ndim'"

#: Prefix of that log record, from ``segment_chunk``'s ``logging.error`` call.
_CHUNK_ERROR_PREFIX = "Error segmenting chunk:"


#: How many box prompts the accelerated path pushes through the mask decoder at once.
#: 32 measured fastest on an RTX 3090 (batch 16/32/64/128/256 → 2.81/2.22/2.45/2.37/2.37 s
#: for the same 529-box block); the curve is flat past 32 because the decoder is a small
#: two-layer transformer that saturates on launch overhead, not on arithmetic. Not a socket:
#: it changes only how the identical work is scheduled, so there is nothing for a user to
#: tune and one more control would be one more thing to get wrong.
_FAST_BATCH = 32

#: Serializes the accelerated tiled path, which must monkey-patch a module attribute inside
#: ``cellSAM.wsi`` (see :func:`_accelerated`). Two segmentation nodes evaluating at once
#: would otherwise install each other's patch; they would also be fighting over one GPU, so
#: serializing them costs nothing real.
_ACCEL_LOCK = threading.Lock()


def _assemble_labels(masks: list, shape: tuple) -> np.ndarray:
    """Per-cell binary masks → one label image, ids ``1..N``, later masks winning overlaps.

    Exactly upstream's ``np.max(stack * arange(1, N+1)[:, None, None], axis=0)``, and
    verified bit-identical against it — but upstream materializes an ``(N, H, W)`` **int64**
    array to get there. At 443 cells on a 1024² block that is a **3.89 GB** transient (plus
    a 464 MB uint8 stack) to produce a 4 MB result, and dask runs several blocks at once.
    Writing the ids in ascending order into one canvas is the same answer — a pixel ends up
    holding the largest id that covers it either way — for 4 MB and half the time
    (1132 ms → 543 ms measured on 443 masks)."""
    lab = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        np.putmask(lab, m.astype(bool, copy=False), i)
    return lab


def _segment_image_fast(img: np.ndarray, model, *, normalize: bool = True,
                        postprocess: bool = False, remove_boundaries: bool = False,
                        bbox_threshold: float = 0.4, device: str = "cpu",
                        batch: int = _FAST_BATCH):
    """``cellSAM.model.segment_cellular_image``, with the per-box loop batched and the
    mask upsample kept on the GPU. Same 3-tuple return, same call signature.

    **Measured on a 1024² block with 441 cells (RTX 3090), against upstream as reference:**

    ==========================================  ======  =======  ==================
    variant                                     time    speedup  differing pixels
    ==========================================  ======  =======  ==================
    upstream (batch 1, CPU upsample)            10.14s   1.00x   reference
    + in-place label assembly                    7.20s   1.41x   0  (bit-identical)
    + batch 32                                   3.17s   3.20x   10  (0.0010%)
    + batch 32 and GPU upsample  (**this**)      1.78s   5.70x   10  (0.0010%)
    + bf16 autocast                              1.78s   5.71x   242857  (23.2%)  ✗
    ==========================================  ======  =======  ==================

    **Why it is not bit-identical, and what that costs.** Batched matmuls reduce in a
    different order than batch-1 ones, so a logit can land either side of the sigmoid cut.
    On that block: 11 of 443 cells changed area, each by **exactly 1 pixel** against a
    median cell of 556 px (0.18% worst case), no cell was gained or lost, and the id
    numbering was unchanged. That is roughly two orders of magnitude below what moving
    ``mask_threshold`` one step does — but it is not zero, which is why this path is
    **opt-in** and the default stays upstream's own function, untouched.

    **bf16 autocast was tried and REJECTED**, on measurement rather than principle: it is
    no faster than this (the decoder is launch-bound, not arithmetic-bound) and it corrupts
    23% of the pixels. Do not re-add it.

    Everything outside the loop — ``format_image_shape``, ``normalize_image``,
    ``fill_holes_and_remove_small_masks(min_size=25)``, ``subtract_boundaries``,
    ``postprocess_predictions`` — is upstream's, called here, so only the SCHEDULING of the
    decoder changes. The no-cells case returns zeros of the right ``(H, W)`` shape rather
    than raising, which is what upstream's own dead branch was written to do (quirk 2)."""
    import torch
    from torch import nn
    from cellSAM.model import (
        fill_holes_and_remove_small_masks, format_image_shape, normalize_image,
        postprocess_predictions, subtract_boundaries,
    )

    model = model.eval()
    model.bbox_threshold = bbox_threshold
    arr = format_image_shape(img)
    if normalize:
        arr = normalize_image(arr)
    arr = arr.transpose((2, 0, 1))
    images = torch.from_numpy(arr).float().unsqueeze(0)
    if "cuda" in device:
        model, images = model.to(device), images.to(device)
    dev = next(model.parameters()).device
    mdl = model.model_cp if getattr(model, "adv_mode", False) else model.model
    h, w = int(images.shape[-2]), int(images.shape[-1])

    with torch.no_grad():
        emb, paddings = model.generate_embeddings(images, device=dev)
        boxes = model.generate_bounding_boxes(images, device=dev)
        boxes = boxes[0] if len(boxes) else []
        kept: list = []
        if len(boxes):
            pe_dense = mdl.prompt_encoder.get_dense_pe()
            in_size = torch.tensor([1024 - paddings[0][0], 1024 - paddings[0][1]]).to(dev)
            for start in range(0, len(boxes), max(1, int(batch))):
                chunk = boxes[start:start + max(1, int(batch))]
                bx = torch.stack([torch.as_tensor(b, device=dev).float()
                                  for b in chunk]).unsqueeze(1)          # (B, 1, 4)
                sparse, dense = mdl.prompt_encoder(points=None, boxes=bx, masks=None)
                low, iou = mdl.mask_decoder(
                    image_embeddings=emb[0].unsqueeze(0), image_pe=pe_dense,
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                    multimask_output=False)
                sel = (iou[:, 0] >= model.iou_threshold).nonzero().flatten().tolist()
                if not sel:
                    continue
                # ONE batched upsample, on the device. Upstream calls ``.cpu()`` first and
                # then interpolates 256²→1024² per mask on the CPU, which is 60% of what
                # this function saves.
                up = mdl.postprocess_masks(low[sel].float(), input_size=in_size,
                                           original_size=(h, w))
                th = (torch.sigmoid(up[:, 0]) > model.mask_threshold).cpu().numpy()
                kept.extend(th[i][:h, :w].astype(np.uint8) for i in range(th.shape[0]))

    if not kept:
        return np.zeros((h, w), dtype=np.int32), None, None
    seg = _assemble_labels(kept, (h, w))
    if postprocess:
        seg = postprocess_predictions(seg)
    mask = fill_holes_and_remove_small_masks(seg, min_size=25)
    if remove_boundaries:
        mask = subtract_boundaries(mask)
    return mask, None, None


@contextlib.contextmanager
def _accelerated(enabled: bool):
    """Route ``cellSAM.wsi``'s per-block segmentation through :func:`_segment_image_fast`.

    The tiled path is dask inside ``segment_wsi``, which calls the ``segment_cellular_image``
    it imported at module scope — there is no hook, no argument, and reimplementing the
    tiling to get one would mean owning the block/overlap/IoU-merge logic too. Rebinding
    that one module attribute for the duration of the compute is the smallest change that
    reaches every block, and it is restored in a ``finally``.

    A no-op when ``enabled`` is false, so the default path never has a patch installed at
    all — the reason "exact" here means *upstream's own function ran*, not "we reproduced
    it". Held under :data:`_ACCEL_LOCK` so two concurrent segmentations cannot install each
    other's patch."""
    if not enabled:
        yield
        return
    import cellSAM.wsi as wsi
    with _ACCEL_LOCK:
        original = wsi.segment_cellular_image

        def patched(chunk, model=None, **kw):
            kw.pop("fast", None)                    # upstream's dead kwarg (quirk 7)
            return _segment_image_fast(chunk, model, batch=_FAST_BATCH, **kw)

        wsi.segment_cellular_image = patched
        try:
            yield
        finally:
            wsi.segment_cellular_image = original


@contextlib.contextmanager
def _chunk_error_watch():
    """Intercept ``cellSAM.wsi.segment_chunk``'s per-block error log; yield ``(benign,
    real)`` message lists, filled by the time the block exits.

    **Why this exists.** ``segment_chunk`` wraps each block in a bare
    ``except Exception``, logs one line, and substitutes ``np.zeros`` — so a block that
    genuinely failed (CUDA OOM, a torn read, a shape upstream never expected) reports *no
    cells* and the pull SUCCEEDS. On a stitched mosaic that is a hole in the results with
    nothing but a log line to mark it, and the user cannot tell it from the ordinary
    no-cells case because both print the same ``ERROR:root:Error segmenting chunk``.

    The untiled path already draws this line: it absorbs the documented no-cells
    ``AttributeError`` (quirk 2a) and re-raises everything else. The tiled path could not,
    because upstream swallows the exception before this kernel ever sees it — the log
    record is the only remaining evidence, so that is what we read.

    So the two are separated here and the caller applies the SAME rule: benign ones are
    counted and their (non-)error lines suppressed, real ones are raised. Attaching a
    ``Filter`` to the root logger rather than a handler is deliberate — a filter that
    returns ``False`` stops the record reaching any handler, which is what keeps a
    not-an-error off the console. Filters live on the logger the record was made through,
    and ``segment_chunk`` uses the module-level ``logging.error``, i.e. root.

    Attached in BOTH places a record can pass through, because the two routings are
    filtered at different points and upstream picks one by writing ``logging.error(...)``
    versus ``logging.getLogger(__name__).error(...)``:

    * on the **root logger** — catches today's module-level ``logging.error`` (a logger's
      filters run only for records made through that logger, never for propagated ones);
    * on root's **handlers** and ``logging.lastResort`` — catches a record propagating up
      from a named child logger, which a logger-level filter would never see.

    Covering only the first would mean an upstream refactor to a named logger silently
    turns this guard off — and a guard that can stop working without saying so is the very
    failure it exists to prevent. Records are marked on first sight, so a record that
    reaches several handlers is still counted once.

    Thread-safe by lock: dask runs blocks on a worker pool, so the filter is called
    concurrently. Detached in a ``finally`` — a leaked filter would silently eat every
    future chunk error in the process."""
    import logging
    import re
    import threading

    benign: list = []
    real: list = []
    lock = threading.Lock()
    pattern = re.compile(_NO_CELLS_CHUNK_RE)
    _MARK = "_nodegraph_chunk_seen"

    class _Watch(logging.Filter):
        def filter(self, record: "logging.LogRecord") -> bool:
            try:
                msg = record.getMessage()
            except Exception:      # noqa: BLE001 — a broken record must not break a run
                return True
            if not msg.startswith(_CHUNK_ERROR_PREFIX):
                return True
            if not getattr(record, _MARK, False):
                setattr(record, _MARK, True)
                with lock:
                    (benign if pattern.search(msg) else real).append(msg)
            return False           # neither case is an ERROR the user should read raw

    watcher = _Watch()
    root = logging.getLogger()
    targets = [root, *root.handlers]
    last = getattr(logging, "lastResort", None)
    if last is not None:
        targets.append(last)
    for tgt in targets:
        tgt.addFilter(watcher)
    try:
        yield benign, real
    finally:
        for tgt in targets:
            try:
                tgt.removeFilter(watcher)
            except Exception:      # noqa: BLE001 — never mask the body's own exception
                pass


def cellsam_available() -> bool:
    """True if the optional ``cellSAM`` package is importable — a side-effect-free probe
    (``find_spec``), so a GUI/selftest can ask without paying a multi-second torch import.
    Note this says nothing about the model WEIGHTS, which are a separate failure mode
    (:func:`get_cellsam_model` raises for those)."""
    return importlib.util.find_spec("cellSAM") is not None


def _require_cellsam():
    """Import and return ``(get_model, get_local_model, segment_cellular_image)``, or
    raise the friendly install hint."""
    if not cellsam_available():
        raise ImportError(_INSTALL_HINT)
    try:
        mod = importlib.import_module("cellSAM")
    except ImportError as exc:
        # find_spec found the package, so this is not an install problem: one of cellSAM's
        # OWN imports moved under it. The known, dated candidate is quirk 5 —
        # ``from skimage.morphology import binary_opening, ...`` at cellSAM/model.py:13,
        # deprecated in scikit-image 0.26 and REMOVED in 0.28. Because it is a module-level
        # import, that upgrade takes out the whole method (every call, not just
        # ``postprocess=True``), and the raw message says nothing about which pin to move.
        raise ImportError(
            f"cellSAM is installed but failed to import: {type(exc).__name__}: {exc}\n"
            "If that names skimage.morphology.binary_*, the cause is scikit-image >= 0.28, "
            "which REMOVED the deprecated `binary_opening`/`binary_closing`/"
            "`binary_dilation`/`binary_erosion` that cellSAM/model.py imports at module "
            "level. Pin `scikit-image<0.28` until cellSAM moves to the replacements "
            "(`opening`/`closing`/`dilation`/`erosion`); nothing in this repo calls the "
            "removed names.") from exc
    return mod.get_model, mod.get_local_model, mod.segment_cellular_image


def resolve_device(requested: str = "") -> str:
    """The torch device string to run on: ``requested`` (or :data:`DEVICE_ENV`, or
    ``auto``) resolved against what torch can actually do.

    ``auto`` picks CUDA when it is available and CPU otherwise. An explicit ``cuda``
    that is not available is a hard error rather than a silent CPU fallback — a user who
    asked for the GPU wants to know the 10× slowdown happened. Raises ``ImportError``
    when torch is missing, which is the same failure the caller already handles.
    """
    want = (requested or os.environ.get(DEVICE_ENV, "") or "auto").strip().lower()
    if want not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {want!r} "
                         f"(set {DEVICE_ENV} to override)")
    if want == "cpu":
        return "cpu"                # nothing to probe — do not pay the torch import
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - env dependent
        raise ImportError(_INSTALL_HINT) from exc
    available = bool(torch.cuda.is_available())
    if want == "auto":
        return "cuda" if available else "cpu"
    if want == "cuda" and not available:
        raise RuntimeError(
            "CellSAM was asked for device 'cuda' but torch reports no CUDA device "
            f"(this build is {torch.__version__}). Unset {DEVICE_ENV} to fall back to "
            "CPU automatically.")
    return want


def get_cellsam_model(model: str = "cellsam_general", *, model_path: str = "",
                      device: str = ""):
    """Get or load the CellSAM model (process singleton keyed by the load parameters).

    Parameters
    ----------
    model : str
        ``"cellsam_general"`` (the published generalist, for reproducing the paper) or
        ``"cellsam_extra"`` (extra training data, recommended for domains outside the
        paper). Ignored when *model_path* is given.
    model_path : str
        Path to a local ``.pt`` checkpoint. Skips the DeepCell download + API token.
    device : str
        ``auto`` | ``cpu`` | ``cuda`` — see :func:`resolve_device`.

    Raises
    ------
    ImportError
        With an actionable install hint when ``cellSAM`` (or torch) is unavailable, or
        when the weights cannot be fetched.
    """
    global _cellsam_model, _cellsam_key
    # probe the package BEFORE torch, so an absent cellSAM reports the install hint
    # rather than whatever torch happens to say.
    get_model, get_local_model, _ = _require_cellsam()
    dev = resolve_device(device)
    key = (str(model), str(model_path), dev)
    if _cellsam_model is not None and _cellsam_key == key:
        return _cellsam_model
    try:
        net = get_local_model(model_path) if model_path else get_model(model)
    except Exception as exc:                        # noqa: BLE001 - surface as one hint
        # The package imported (``_require_cellsam`` succeeded above), so this is about the
        # WEIGHTS: token, network, TLS, or a bad path. Do NOT print the install hint here.
        raise ImportError(f"CellSAM weights unavailable "
                          f"({'local ' + model_path if model_path else model}): "
                          f"{type(exc).__name__}: {exc}\n{_WEIGHTS_HINT}") from exc
    net = net.eval()
    if dev != "cpu":
        # Move once at load. ``segment_cellular_image`` also does ``model.to(device)``
        # per call, which is then a no-op instead of a per-plane host→device copy.
        net = net.to(dev)
    _cellsam_model, _cellsam_key = net, key
    return net


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Renumber an integer label image to a contiguous ``1..K`` (background stays 0).

    One ``O(pixels)`` bincount + LUT pass, the same shape as
    ``stardist_segment.filter_and_relabel`` minus the filtering (the caller applies the
    physical-unit size filter, which needs calibration this kernel does not have).
    Contiguity is what lets the caller offset ids per plane into globally unique ones.
    """
    lab = np.asarray(labels)
    if lab.size == 0 or int(lab.max()) == 0:
        return lab.astype(np.int32, copy=False)
    present = np.flatnonzero(np.bincount(lab.ravel()))
    present = present[present > 0]
    lut = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lut[lab]


@contextlib.contextmanager
def _dask_ticker(tick: Callable[[int], None], *, lo: int, hi: int):
    """Report ``lo``→``hi`` across a dask ``compute()``, one step per finished task.

    ``cellSAM.wsi.segment_wsi`` returns a lazy dask graph rather than an array, so the
    per-block loop that a tiled plane really does run belongs to dask's scheduler — and
    dask's ``Callback`` hooks are the only way to observe it. ``_start`` gives the task
    count up front (so the fraction has a real denominator, not a guess) and ``_posttask``
    fires once per completed task.

    Degrades to a no-op context if dask's callback API is missing or has moved: tiled
    inference must still run, just without the finer bar. The scheduler counts *graph
    tasks*, not blocks — a block is several tasks (read, predict, stitch) — which makes the
    bar finer than per-block, never coarser, and still monotone."""
    try:
        from dask.callbacks import Callback
    except Exception:  # noqa: BLE001 — no dask callbacks → no sub-bar, not an error
        yield
        return

    span = max(1, int(hi) - int(lo))

    class _Ticker(Callback):
        def _start(self, dsk):
            self._n = max(1, len(dsk))
            self._k = 0

        def _posttask(self, key, result, dsk, state, worker_id):
            self._k += 1
            tick(int(lo) + min(span, span * self._k // self._n))

    try:
        ticker = _Ticker()
    except Exception:  # noqa: BLE001 — a hook that won't even build → no sub-bar
        yield
        return
    # NOT wrapped in try/except: the body's own exceptions must propagate untouched (a
    # failed segmentation is a real failure), and catching around a `yield` would try to
    # resume an already-finished generator.
    with ticker:
        yield


def segment_plane(
    image: np.ndarray,
    model=None,
    *,
    bbox_threshold: float = 0.4,
    mask_threshold: float = 0.4,
    mask_quality: float = 0.5,
    normalize: bool = True,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    model_name: str = "cellsam_general",
    model_path: str = "",
    device: str = "",
    tile: bool = False,
    tile_size: int = 512,
    overlap: int = 56,
    tile_iou: float = 0.5,
    fast: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Segment cells in a single 2-D plane with CellSAM.

    Parameters
    ----------
    image : 2-D array (H, W), any dtype
        One prepared plane. Placed in CellSAM's whole-cell channel slot upstream (see
        the module docstring).
    model : nn.Module, optional
        A loaded CellSAM model; ``None`` uses/loads the singleton.
    bbox_threshold : float
        CellFinder box confidence cut — *the* precision/recall knob. Upstream default
        0.4; lower it for out-of-distribution images. (CellSAM then blends it with a
        per-image k-means split of the box confidences, ``0.66·T + 0.33·T_cluster``,
        which is the paper's dynamic ``T_box``.) Forwarded as a keyword; upstream assigns
        it onto the model itself.
    mask_threshold : float
        The per-pixel **sigmoid cut** applied to the mask decoder's logits, i.e. how far
        each mask extends. In ``(0, 1)``. Lower ⇒ larger masks, so this moves every AREA
        the caller measures. Shipped upstream default 0.4; **the paper says 0.5** (Methods
        → Thresholding) — 0.4 is kept as the default here so exposing the knob changes no
        existing result, but 0.5 is what reproduces the publication.
    mask_quality : float
        Minimum **predicted mask quality** (the mask decoder's IoU-prediction-head score)
        for a detection to be kept at all — upstream's ``CellSAM.iou_threshold``, 0.5 in
        both the code and the paper. In ``[0, 1]``. A second recall knob, INDEPENDENT of
        ``bbox_threshold``: a box can clear the confidence cut and still be discarded here.
        Raise it to drop ragged masks, lower it when cells are being detected but not
        returned.

        Both of the above are set **on the model object** rather than passed as keywords —
        ``segment_cellular_image`` accepts neither. See the note in :func:`segment_plane`'s
        body for why they must be re-assigned on every call.
    normalize : bool
        Apply CellSAM's own preprocessing (99.9-percentile clip + per-channel rescale +
        CLAHE, kernel 128) — the paper's Methods pipeline (Nat. Methods 22:2585, Methods
        → dataset preparation).
        **Effectively mandatory for data outside [0,1]** — see quirk 6. It is NOT merely
        "skip it if you already matched the paper": with it off, a raw 16-bit plane is
        destroyed on the way into CellFinder and the plane comes back EMPTY, with no error.
    postprocess : bool
        Upstream morphological cleanup, "recommended for noisy images". See quirks 4
        (a zero-size crash on an all-background prediction) and 5 (its deprecated
        ``skimage`` calls, whose per-cell warning spam this function filters).
    remove_boundaries : bool
        Erode a one-pixel gap between touching cells.
    tile : bool
        Segment the plane in overlapping blocks and stitch by IoU
        (``cellSAM.wsi.segment_wsi``) instead of in one pass. Needed for large FOVs /
        very many cells (upstream suggests tiling above roughly 3000 cells per image).
    tile_size, overlap : int
        Block edge and overlap **in pixels** (the caller converts from µm). ``overlap``
        must be wide enough to contain a typical cell, and it doubles as upstream's
        ``iou_depth`` — which upstream requires to be ``<= overlap``, so passing the
        same value is both legal and maximal.
    tile_iou : float
        IoU above which two blocks' labels are merged into one cell when stitching tiles.
        Named ``tile_iou`` rather than ``iou_threshold`` deliberately: three different IoU
        quantities are now in play (this one, ``mask_quality``'s predicted-mask IoU, and
        ``track.objects``' own ``iou_threshold`` socket elsewhere in the catalog), and the
        bare name gave no way to tell them apart. Only read when ``tile=True``.
    progress_cb : callable(int) or None
        Called with ``0..100`` as this **one plane** progresses, for a caller driving a
        sub-bar inside its per-plane loop.

        How much resolution you get depends entirely on ``tile``, and the difference is not
        cosmetic. With ``tile=True`` the plane is a dask graph over blocks and this reports
        **one tick per finished block** — genuine per-iteration progress. With
        ``tile=False`` a plane is a *single opaque* ``segment_cellular_image`` call with no
        upstream hook of any kind, so all this can honestly say is "started" / "finished":
        it emits 8 on entry and 92 when inference returns, and NOTHING in between. A caller
        must not render that gap as a stalled determinate bar — it has no idea how far
        along it is, and :meth:`nodegraph.engine.EvalContext.progress`'s
        ``sub_unknown=True`` exists for exactly this case.

    Returns
    -------
    labels : 2-D ``int32`` array, contiguous ids ``1..K``, 0 = background.
    """
    plane = np.asarray(image)
    if plane.ndim != 2:
        raise ValueError(f"cellsam segment_plane expects a 2-D plane, got shape "
                         f"{plane.shape} — the caller owns the (m,t,z,c) loop")

    def _tick(pct: int) -> None:
        """Report into the caller's sub-bar, never raising — a progress sink must not be
        able to fail a segmentation."""
        if progress_cb is None:
            return
        try:
            progress_cb(max(0, min(100, int(pct))))
        except Exception:  # noqa: BLE001
            pass

    _tick(0)
    if model is None:
        model = get_cellsam_model(model_name, model_path=model_path, device=device)
    dev = resolve_device(device)
    kwargs = dict(normalize=bool(normalize), postprocess=bool(postprocess),
                  remove_boundaries=bool(remove_boundaries),
                  bbox_threshold=float(bbox_threshold), device=dev)

    # ── the two thresholds upstream gives no keyword for ──────────────────────────
    # ``segment_cellular_image`` accepts only ``bbox_threshold`` and assigns it onto the
    # model (``model.bbox_threshold = bbox_threshold``). The other two documented inference
    # thresholds — the per-pixel sigmoid cut and the mask-decoder IoU-head gate — are read
    # straight off ``self`` inside ``CellSAM.predict`` and have no argument at all, so the
    # only way to drive them is to assign them here.
    #
    # RE-ASSIGNED ON EVERY CALL, deliberately. The model is a process SINGLETON: set once
    # and plane 2 of a pull would silently inherit plane 1's thresholds, and two graphs in
    # one session would contaminate each other — a memo hit would then depend on execution
    # ORDER, which is exactly the non-determinism the recipe hash exists to prevent. Paying
    # two attribute writes per plane buys back that determinism. This mirrors what upstream
    # itself does with ``bbox_threshold``.
    mt, mq = float(mask_threshold), float(mask_quality)
    if not 0.0 < mt < 1.0:
        # Upstream asserts ``self.mask_threshold > 0`` inside ``predict``; >= 1 is a cut no
        # sigmoid output can clear, so it would return an empty plane rather than an error.
        raise ValueError(f"mask_threshold must be in (0, 1), got {mt} — it is a cut on a "
                         f"sigmoid, so 0 keeps everything and 1 keeps nothing")
    if not 0.0 <= mq <= 1.0:
        raise ValueError(f"mask_quality must be in [0, 1], got {mq} — it is compared "
                         f"against the decoder's predicted-IoU score")
    for attr in ("mask_threshold", "iou_threshold"):
        # Fail LOUDLY if upstream ever renames these. Assigning a name the model does not
        # have would silently create a dead attribute, leaving two live-looking node sockets
        # that change nothing — the precise failure this repo's socket contract forbids.
        if not hasattr(model, attr):
            raise AttributeError(
                f"this CellSAM model has no {attr!r}, so the node's mask_threshold / "
                f"mask_quality sockets would silently do nothing. Upstream defines both in "
                f"``CellSAM.__init__`` (cellSAM/sam_inference.py); if a new version renamed "
                f"them, re-map them here rather than leaving the sockets dead.")
    model.mask_threshold = mt
    model.iou_threshold = mq

    # QUIRK 5 — silence upstream's per-cell skimage deprecation spam, and ONLY that, for
    # the duration of the call. ``postprocess=True`` emits four FutureWarnings per cell per
    # plane from inside ``cellSAM.model.postprocess_predictions``; on a real time series
    # that is thousands of identical lines. The filter is scoped to the call and pinned to
    # those four messages, so anything else upstream says still reaches the user.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_MORPHOLOGY_DEPRECATION_RE,
                                category=FutureWarning)
        if tile:
            try:
                from cellSAM.wsi import segment_wsi
            except ImportError as exc:              # dask-image / sklearn ride along
                raise ImportError(f"CellSAM tiled inference needs cellSAM.wsi and its "
                                  f"dask-image / scikit-learn dependencies: {exc}\n"
                                  f"{_INSTALL_HINT}") from exc
            block = max(64, int(tile_size))
            lap = max(1, min(int(overlap), block - 1))
            # The seam merge warns once per single-label block face; on a mosaic with
            # physical gaps that is most of them (see _SKLEARN_SINGLE_LABEL_RE).
            warnings.filterwarnings("ignore", message=_SKLEARN_SINGLE_LABEL_RE,
                                    category=UserWarning)
            # QUIRK 2a, TILED — hold the SAME line the untiled branch holds below: absorb
            # the documented no-cells bug, refuse anything else. Upstream zeroes a failed
            # block and carries on, so without this a genuine failure is reported as "this
            # region contained no cells" and the pull succeeds.
            with _chunk_error_watch() as (benign_chunks, failed_chunks), \
                    _accelerated(bool(fast)):
                out = segment_wsi(plane, block, lap, lap, float(tile_iou),
                                  model=model, **kwargs)
                _tick(8)
                # REAL per-iteration progress, and the only place in this kernel it exists:
                # `segment_wsi` returns a lazy dask graph, so the blocks have not run yet
                # and dask will tell us as each one lands. Scoped to this compute by the
                # context manager, so it cannot leak onto another thread's dask work.
                with _dask_ticker(_tick, lo=8, hi=92):
                    lab = np.asarray(getattr(out, "compute", lambda: out)())
            if failed_chunks:
                shown = sorted({m.split(_CHUNK_ERROR_PREFIX, 1)[-1].strip()
                                for m in failed_chunks})[:5]
                raise RuntimeError(
                    f"CellSAM tiled inference: {len(failed_chunks)} block(s) FAILED and "
                    f"upstream replaced each with an empty mask, so those regions would "
                    f"have been reported as containing no cells. Refusing that silently. "
                    f"Distinct errors: {'; '.join(shown)}. "
                    f"(A block that merely contains no cells is normal and is not counted "
                    f"here — {len(benign_chunks)} of those were seen and ignored.) If this "
                    f"is memory, lower tile_size; the whole plane is {plane.shape}.")
        else:
            segment_cellular_image = _require_cellsam()[2]
            # ONE opaque call — upstream exposes no callback, no tile count, nothing. 8 in,
            # 92 out, and a caller that draws the gap as a determinate bar is lying; see the
            # `progress_cb` docs.
            _tick(8)
            if fast:
                # No patching needed off the tiled path — call ours directly, then fall
                # through to the SAME shape/relabel tail every branch uses. It already
                # returns zeros for the no-cells case, so the quirk-2 repair below is
                # unreachable from here; that `except` still guards the upstream branch.
                lab = np.asarray(_segment_image_fast(plane, model, batch=_FAST_BATCH,
                                                     **kwargs)[0])
            else:
                try:
                    lab = np.asarray(segment_cellular_image(plane, model, **kwargs)[0])
                except AttributeError as exc:
                    # QUIRK 2b — the REAL no-cells path (upstream issue #98).
                    # ``CellSAM.predict`` returns the 4-tuple ``(None, None, None, None)``
                    # when no box survives the confidence/IoU filter, so
                    # ``segment_cellular_image``'s ``if preds is None`` guard never fires;
                    # it unpacks the tuple and calls
                    # ``fill_holes_and_remove_small_masks(None)`` → ``AttributeError`` on
                    # ``masks.ndim`` (or later on ``x.cpu()``). A blank plane, an empty FOV
                    # or the dark end slices of a stack would otherwise CRASH the whole
                    # pull. "No cells" is an empty segmentation, which is what upstream's
                    # own unreachable branch returns, so this repairs a broken guard rather
                    # than inventing a behaviour — and it stays narrow: only an
                    # AttributeError ON NoneType is absorbed.
                    if "NoneType" not in str(exc):
                        raise
                    lab = np.zeros(plane.shape, dtype=np.int64)
    _tick(92)

    if lab.ndim == 3:
        # Quirk 2: the no-cells path returns the (C, H, W) tensor shape. Any genuine
        # per-channel stack would be identical across slots, so folding by max is safe
        # and keeps the all-zero case zero.
        lab = lab.max(axis=0) if lab.shape[0] <= 3 else lab.max(axis=-1)
    if lab.shape != plane.shape:
        raise ValueError(f"CellSAM returned a {lab.shape} mask for a {plane.shape} "
                         f"plane — refusing to guess the alignment")
    out_lab = relabel_contiguous(lab.astype(np.int64, copy=False))
    _tick(100)
    return out_lab


def _reset_model_cache() -> None:
    """Drop the singleton — for tests that swap the backing module."""
    global _cellsam_model, _cellsam_key
    _cellsam_model, _cellsam_key = None, None
