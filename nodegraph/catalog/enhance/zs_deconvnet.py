"""ZS-DeconvNet (``enhance.zs_deconvnet``) — self-supervised denoising + deconvolution: a two-stage CNN trained on the incoming data ITSELF (no ground truth), or a published checkpoint run in inference only; 2D U-Net per plane vs volumetric 3D RCAN/U-Net."""

from __future__ import annotations

import json
import os

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.engine import EvalContext
from nodegraph.metadata import zs_deconvnet as _meta_zs_deconvnet
from nodegraph.provider import ArrayProvider
from nodegraph.registry import (
    DimMode,
    Granularity,
    InBool,
    InDataset,
    InFloat,
    InInt,
    InString,
    Mode,
    OutDataset,
)
from nodegraph.spill import dense_output
from nodegraph.trained import (
    INFERENCE_KEY as _INFERENCE_KEY,
    read_json as _read_model_json,
    zs_signature_of as _zs_signature_of,
    zs_trained as _zs_trained_of,
)

from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.progress import _UnitBar
from nodegraph.catalog._shared.psf import diffraction_sigmas

#: The two ways to get a network. ``zero_shot`` is the paper's method — train on the data in
#: front of you; ``pretrained`` is the Fiji plugin's "predict" button.
_MODES = ("zero_shot", "pretrained")
#: Which of the dual-stage network's two heads becomes the output image.
_OUTPUTS = ("deconvolved", "denoised")
#: 3D backbones. The paper's Fig. 3a backbone (and the reference's default) is the RCAN.
_ARCH_3D = ("rcan3d", "unet3d")


def _psf_for_channel(ctx: EvalContext, c: int, *, is_3d: bool, px: float,
                     zs: float) -> np.ndarray:
    """The PSF the DECONVOLUTION LOSS is trained against, for channel ``c``.

    Two sources, in priority order:

    * **A measured / simulated PSF file** (``psf_path``), resampled from its own sampling
      (``psf_pixel_um``, ``psf_z_um``) onto this data's grid. One file serves every channel,
      because a PSF file is a measurement of one configuration and the node has no way to
      know which channel it belongs to.
    * **Derived from the optics metadata** — the default, and the reason this node needs no
      files at all. ``diffraction_sigmas`` comes from ``_shared/psf.py`` rather than being
      re-derived so that the two deconvolution nodes in this catalog model the same
      microscope the same way; emission λ and NA resolve PER CHANNEL through
      ``ctx.channel(c).param`` (C8/H12), so a two-colour stack gets two different PSF widths
      instead of channel 0's applied to both.

    The PSF matters more here than the iteration count: the paper's Supplementary Fig. 28c
    shows a mismatched PSF produces either no resolution gain or ringing artifacts, and
    unlike Richardson–Lucy the mistake is baked into trained weights rather than re-derived
    per pull.
    """
    from nodegraph.kernels import zsdeconvnet as zsk
    path = str(ctx.params.get("psf_path") or "").strip()
    if path:
        psf = zsk.load_psf_tif(path)
        want_3d = bool(is_3d)
        if psf.ndim == 3 and not want_3d:
            psf = psf[psf.shape[0] // 2]              # centre plane for a 2D model
        if psf.ndim == 2 and want_3d:
            raise ValueError(
                f"zs-deconvnet: {os.path.basename(path)} is a 2D PSF but the lever is set to "
                "3D, whose deconvolution loss convolves in z as well. Supply a 3D (Z,Y,X) "
                "PSF stack, or clear `psf_path` to derive an anisotropic Gaussian PSF from "
                "the file's own NA / emission / z step.")
        src = float(ctx.params.get("psf_pixel_um", 0.0) or 0.0)
        src_z = float(ctx.params.get("psf_z_um", 0.0) or 0.0)
        if src:
            psf = zsk.resample_psf(psf, src, px,
                                   src_z_um=(src_z or None), dst_z_um=(zs or None))
        return zsk.crop_psf(psf)
    sig = diffraction_sigmas(ctx.channel(c).param("emission_nm"),
                             ctx.channel(c).param("na"), px, zs, bool(is_3d))
    return zsk.gaussian_psf_from_sigmas(sig)


def _cache_paths(base: str, c: int) -> Tuple[str, str]:
    """``(weights, sidecar)`` for channel ``c`` under the user's cache path.

    Per channel because the paper trains one model per emission wavelength ("Independent
    ZS-DeconvNet models were trained for each biological structure and emission wavelength"),
    so one path has to fan out into one file per channel or channel 1 would silently reuse
    channel 0's optics.
    """
    root = base[:-len(".weights.h5")] if base.endswith(".weights.h5") else \
        (base[:-3] if base.endswith(".h5") else base)
    return f"{root}_c{int(c)}.weights.h5", f"{root}_c{int(c)}.json"


#: The largest lateral activation extent the CPU keeps in cache. Past it the second stage's
#: 128-channel convolutions fall off a cliff that is nothing like linear — MEASURED on a
#: 1024² plane: a 288 px padded tile is 0.39 s, while 544 and 576 px are 140 s and 133 s,
#: i.e. **340× per pixel**. That is what turns a 2.1-hour series into a 724-hour one, and it
#: is invisible in the parameters (`tile` 256 is fine until `upsample` doubles it to 576).
_FAST_EXTENT_PX = 320


def _cost_warning(*, tile: int, insert_xy: int, upsample: bool, is_3d: bool,
                  units: int) -> str:
    """A warning naming the cheaper setting when this configuration is in the slow regime.

    Not a refusal: a big tile has fewer seams and someone may legitimately want it on a
    single frame. But at 784 units the difference is 2 hours against a month, and the symptom
    is a bar that appears frozen, so the node says so up front rather than letting the user
    discover it by waiting.
    """
    extent = (int(tile) + 2 * int(insert_xy)) * (2 if upsample else 1)
    if extent <= _FAST_EXTENT_PX:
        return ""
    cheaper = ("turn `upsample` OFF" if upsample else
               f"lower `tile` to {max(64, _FAST_EXTENT_PX - 2 * int(insert_xy))}")
    return (f"SLOW CONFIGURATION: tile {tile} + {2 * insert_xy} px padding"
            f"{' x2 upsampling' if upsample else ''} = {extent} px of activation, past the "
            f"~{_FAST_EXTENT_PX} px cache limit — measured ~340x slower PER PIXEL there "
            f"(0.39 s vs 133 s for one tile). {units} unit(s) at this setting is a very long "
            f"run; {cheaper} to stay in the fast regime.")


def _checkpoint_sidecar(weights_path: str) -> str:
    """The ``.json`` this node writes beside a checkpoint it trained, if any.

    ``zero_shot`` + ``cache_path`` writes ``<root>_c0.weights.h5`` and ``<root>_c0.json``, so
    the sidecar is derivable from the weights path alone. Its whole purpose in ``pretrained``
    mode is :func:`_check_checkpoint`.
    """
    for suffix in (".weights.h5", ".hdf5", ".h5"):
        if weights_path.endswith(suffix):
            return weights_path[:-len(suffix)] + ".json"
    return weights_path + ".json"


def _zs_trained(params, modes) -> Dict[str, Any]:
    """``NodeSpec.trained_params`` for this node: the graph-shaping settings the checkpoint
    on ``weights_path`` was trained with, so its sockets sit on the right values by default
    instead of on numbers the user had to transcribe (V2.23).

    A thin delegate to :func:`nodegraph.trained.zs_trained`; the resolver lives in
    ``nodegraph/`` because :func:`nodegraph.metadata.zs_deconvnet` must consult the SAME
    answer — an adopted ``upsample`` changes the output axes, so the edit-time prediction and
    the compute have to agree about it (wire-node-v2 §8).
    """
    return _zs_trained_of(params, modes)


def _zs_inference_record(*, arch: str, is_3d: bool, upsample: bool, insert_xy: int,
                         insert_z: int) -> Dict[str, Any]:
    """The EFFECTIVE graph-shaping settings of a training run, for the sidecar's
    :data:`nodegraph.trained.INFERENCE_KEY` block.

    Effective, not overridden — and that distinction is the point. :func:`_train_signature`
    records only params the user explicitly set (it iterates ``ctx.params``, which holds
    overrides), because that is all an equality check needs to prove a cached model matches.
    But a checkpoint trained entirely at defaults then records almost nothing, so a later
    ``pretrained`` run had nothing to adopt. This block states the values that were actually
    used, whether they came from a socket or a default.

    It is a SEPARATE top-level key rather than more signature entries so that adding it
    cannot invalidate a cached model: the reuse check compares
    :func:`nodegraph.trained.zs_signature_of`, which strips exactly this key, so a sidecar
    written before this existed still compares byte-for-byte as it always did. Retraining is
    tens of minutes to hours per channel on this build — not a cost to impose for a
    bookkeeping change.
    """
    lateral = "insert_xy_3d" if is_3d else "insert_xy"
    rec: Dict[str, Any] = {"arch": arch, "dim": "3D" if is_3d else "2D",
                           "upsample": bool(upsample), lateral: int(insert_xy)}
    if is_3d:
        rec["insert_z"] = int(insert_z)
    return rec


def _check_checkpoint(weights_path: str, *, arch: str, is_3d: bool,
                      params: Mapping[str, Any]) -> str:
    """Refuse a checkpoint whose own sidecar says it was trained a different way — for the
    settings the user must resolve, having ADOPTED the ones the node can resolve itself.

    This exists because the failure it catches is SILENT. A legacy/Keras ``.h5`` is matched to
    the graph by the topological order of weight-BEARING layers, and ``UpSampling2D`` /
    ``UpSampling3D`` carry none — so a model trained with ``upsample=False`` loads into an
    ``upsample=True`` graph without a murmur and then produces a 2x image its second stage was
    never trained to produce. Nothing downstream can tell; it just looks like a bad result
    (and, per :func:`_cost_warning`, takes 340x longer to get).

    **What is refused vs adopted (V2.23).** The two are split by whether the node can tell
    "unset" from "explicitly the default":

    * ``arch`` and ``dim`` are **Modes**. The engine hands a compute the *resolved* mode
      state, so a deliberate ``rcan3d`` is indistinguishable from an untouched one — adopting
      would silently overwrite a choice the user may have made on purpose. These still raise.
    * ``upsample`` / the padding margins are **sockets**, and params are raw overrides that
      are never default-filled, so absence really does mean "not pinned". Those are adopted
      by :func:`_zs_trained`, and only a **pinned** value that contradicts the sidecar raises
      here — the socket keeps its ability to disagree, it just has to say so deliberately.

    Only checkpoints THIS node trained carry a sidecar. The authors' published models have
    none, so their absence is not an error — it just means the user is responsible for
    matching ``arch``/``dim``/``upsample`` themselves, which the socket docs say.

    Returns a note for the progress rail when the sidecar confirms a match.
    """
    side = _checkpoint_sidecar(weights_path)
    sig = _read_model_json(side)
    if not sig:
        return ""                    # absent or unreadable: fall back to trusting it
    bad: Dict[str, Any] = {}
    # (a) the Modes — refused whether pinned or not, because "pinned" is not observable.
    for k, v in (("arch", arch), ("dim", "3D" if is_3d else "2D")):
        if k in sig and sig[k] != v:
            bad[k] = (sig[k], v)
    # (b) the sockets — refused only when the user PINNED a contradicting value. An unset
    # socket is not a disagreement; `_zs_trained` has already made it agree.
    adopted = _zs_trained_of(params, {"mode": "pretrained"})
    for k, want in adopted.items():
        got = params.get(k)
        if got in (None, ""):
            continue                                    # on auto: adopted, not a conflict
        same = (bool(got) == bool(want) if isinstance(want, bool)
                else _num_eq(got, want))
        if not same:
            bad[k] = (want, got)
    if bad:
        parts = ", ".join(f"{k}: trained with {t!r} but this node is set to {n!r}"
                          for k, (t, n) in sorted(bad.items()))
        raise ValueError(
            f"zs-deconvnet: {os.path.basename(weights_path)} was trained with different "
            f"settings — {parts}. Its own record is {os.path.basename(side)}. This would "
            f"NOT fail on its own: up-sampling layers carry no weights, so the checkpoint "
            f"loads into the wrong graph and quietly returns an image its second stage was "
            f"never trained to produce. Match the setting(s) above (unpin a socket to take "
            f"the checkpoint's own value), or point `weights_path` at a checkpoint trained "
            f"the way this node is configured.")
    return f"checkpoint verified against {os.path.basename(side)}"


def _num_eq(a: Any, b: Any) -> bool:
    """Numeric equality that tolerates the int/float/str round trip a param takes through
    JSON and a Qt spin box — ``8`` from a QSpinBox, ``8.0`` out of a sidecar, ``"8"`` from a
    hand-edited saved graph all mean the same padding margin. Anything non-numeric compares
    by string, so a genuinely different value still registers as a conflict."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _train_signature(ctx: EvalContext, *, is_3d: bool, arch: str, psf: np.ndarray,
                     upsample: bool) -> Dict[str, Any]:
    """Everything that changes what training produces, as a JSON-able dict.

    Written beside a cached checkpoint so a reused one can be PROVEN to belong to the
    current settings. Without it, editing `iterations` (or the PSF, or the patch size) would
    silently return the previously trained model: the engine's memo would correctly decide to
    re-run the compute, and the compute would then load a stale file and report success. The
    PSF enters as a rounded checksum rather than an array so a re-derived-but-identical PSF
    still matches.
    """
    keys = ("iterations", "batch_size", "patch", "patch_3d", "patch_z",
            "learning_rate", "learning_rate_3d", "hess_weight", "hess_weight_3d",
            "denoise_weight", "background", "alpha", "beta1", "beta2", "insert_xy",
            "insert_xy_3d", "insert_z", "train_units", "seed")
    sig: Dict[str, Any] = {"arch": arch, "dim": "3D" if is_3d else "2D",
                           "upsample": bool(upsample),
                           "psf_shape": list(psf.shape),
                           "psf_sum2": round(float((psf.astype(np.float64) ** 2).sum()), 9)}
    for k in keys:
        v = ctx.params.get(k)
        if v is not None and v != "":
            sig[k] = round(float(v), 9) if isinstance(v, (int, float)) else str(v)
    return sig


def _read_units(prov, ax, c: int, *, is_3d: bool, limit: int) -> List[np.ndarray]:
    """Up to ``limit`` training units for channel ``c``, evenly strided over the dataset.

    Strided rather than "the first N": a time series' first N frames are the most correlated
    slice of it, and a 49-position plate's first N are one corner of the well. An even stride
    over ``(m, t, z)`` (2D) or ``(m, t)`` (3D) samples the variety the model has to generalize
    over, which is the whole reason the paper trains on all frames and then applies the result
    to all of them.
    """
    if is_3d:
        addr = [(m, t) for m in range(ax.m) for t in range(ax.t)]
    else:
        addr = [(m, t, z) for m in range(ax.m) for t in range(ax.t) for z in range(ax.z)]
    n = max(1, int(limit))
    if len(addr) > n:
        step = len(addr) / float(n)
        addr = [addr[min(len(addr) - 1, int(i * step))] for i in range(n)]
    out: List[np.ndarray] = []
    for a in addr:
        if is_3d:
            m, t = a
            out.append(np.asarray(
                prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x),
                dtype=np.float32))
        else:
            m, t, z = a
            out.append(np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x),
                                  dtype=np.float32))
    return out


def _compute_zs_deconvnet(ctx: EvalContext) -> Dataset:
    """**ZS-DeconvNet** — the paper's zero-shot deconvolution network, as one node.

    Qiao et al., *Nat. Commun.* 15:4180 (2024). A dual-stage CNN: stage I denoises, stage II
    deconvolves, and the pair is trained **without any ground truth** by a physics-informed
    self-supervised loss — the sharp prediction is re-blurred with the PSF and compared to a
    second, noise-independent copy of the same measurement. Kernel:
    :mod:`nodegraph.kernels.zsdeconvnet`.

    ``mode``
        * **zero_shot** — the actual method. Trains a model on THIS dataset (2D:
          re-corruption, paper Eq. 9-12; 3D: axial parity splitting, parameter-free) and then
          runs it. One model per channel, because the PSF differs per emission wavelength.
        * **pretrained** — load a published ``.h5`` checkpoint and infer only. Fast, needs no
          PSF at all (the PSF is a *training-time* term and never enters inference), but the
          checkpoint must have been trained for optics like yours.

    ``output`` picks which head becomes the image: **deconvolved** (the point of the node,
    optionally 2× upsampled) or **denoised** (stage I, always at the input grid).

    **2D vs 3D.** 2D runs the two-stage U-Net on each ``(m,t,z,c)`` plane. 3D runs a
    genuinely volumetric backbone — the RCAN of the paper's Fig. 3a, all ``Conv3D``, no
    stack-of-2D fakery — on each ``(m,t,c)`` volume, and its up-sampling is deliberately
    lateral-only (``(2,2,1)``), so ``z_step_um`` never changes.

    **It drops ``bit_depth`` (§7c)** and, when ``upsample`` is on, doubles Y/X and halves
    ``pixel_size_um``: input is percentile-normalized to ``[0,1]`` before the network sees it
    and the output is percentile-normalized again, so the result is not integer counts on any
    scale. Both halves are predicted by :func:`nodegraph.metadata.zs_deconvnet`.

    **What this costs.** Measured on this build (TensorFlow 2.21, CPU only — TF ≥ 2.11 has no
    native-Windows CUDA): training is ~1.4 s/iteration for the 2D default patch, so the
    paper's 50 000 iterations is ~19 h and the shipped 2 000 is ~45 min *per channel*. Set
    ``cache_path`` and that is paid once, not once per pull. 2D inference is memory-bound and
    scales WORSE than linearly in tile area above ~128 px (0.8 s for a 128 px tile with
    upsampling, 109 s at 256, 470 s at 512), which is why the default tile is small.
    """
    from nodegraph.kernels import zsdeconvnet as zsk
    ds: Dataset = ctx.inputs[0]
    prov = ds.image
    if prov is None:
        raise ValueError("zs-deconvnet needs an image provider on its input Dataset")
    ax = prov.axes
    is_3d = ctx.is_volume
    modes = ctx.params.get("__modes__", {})
    mode = str(modes.get("mode") or "zero_shot")
    if mode not in _MODES:
        raise ValueError(f"zs-deconvnet: unknown mode {mode!r} — one of {list(_MODES)}")
    which = str(modes.get("output") or "deconvolved")
    if which not in _OUTPUTS:
        raise ValueError(f"zs-deconvnet: unknown output {which!r} — one of {list(_OUTPUTS)}")
    arch = (str(modes.get("arch_3d") or "rcan3d") if is_3d else "unet2d")

    # What the checkpoint on `weights_path` was TRAINED with — `{}` in `zero_shot` mode and
    # for any checkpoint with no sidecar (V2.23). Resolved here, before the sockets that
    # consult it, and through the same `nodegraph.trained` resolver the inspector's auto box
    # and `metadata.zs_deconvnet` call, so the number displayed, the number predicted and the
    # number used are one value by construction rather than by three copies agreeing.
    adopted = _zs_trained(ctx.params, modes)

    def _zs_sock(name: str, default: Any) -> Any:
        """One graph-shaping socket: the user's PINNED value, else what the model was
        trained with, else the socket default.

        `ctx.params.get(name)` with no fallback is the load-bearing part — params are raw
        overrides that the engine never default-fills, so absence is exactly "the user did
        not pin this", which is the same signal the inspector's auto/pin box draws from.
        `False` and `0` are legitimate pinned values and must not read as absent, which is
        why the test is against `(None, "")` and not truthiness.
        """
        v = ctx.params.get(name)
        return v if v not in (None, "") else adopted.get(name, default)

    upsample = bool(_zs_sock("upsample", True))
    tile = int(ctx.params.get("tile", 128) or 128)
    overlap = int(ctx.params.get("overlap", 20) or 0)
    norm_low = float(ctx.params.get("norm_low", 3.0))
    background = float(ctx.params.get("background", 100.0))
    # The four per-dim params are read with LITERAL keys inside an `is_3d` branch,
    # deliberately: the socket contract's index is an AST pass that only resolves string
    # constants, so a `ctx.params.get(key_variable)` read would make all eight sockets look
    # DEAD and fail the guard. Same pattern (and same reason) as `analysis.segment`'s
    # min_area/min_volume pair. `_zs_sock` keeps the literal for the same reason.
    if is_3d:
        insert_xy = int(_zs_sock("insert_xy_3d", 8))
        t_patch = int(ctx.params.get("patch_3d", 64))
        t_lr = float(ctx.params.get("learning_rate_3d", 1e-4))
        t_hess = float(ctx.params.get("hess_weight_3d", 0.1))
    else:
        insert_xy = int(_zs_sock("insert_xy", 16))
        t_patch = int(ctx.params.get("patch", 128))
        t_lr = float(ctx.params.get("learning_rate", 5e-5))
        t_hess = float(ctx.params.get("hess_weight", 0.02))
    insert_z = int(_zs_sock("insert_z", 2))
    px = ctx.calib("pixel_size_um") or 0.1
    zs_um = (ctx.calib("z_step_um") or 0.5) if is_3d else 0.0

    # `upsample` scales the LATERAL axes only, and only for the deconvolution head — stage I
    # is the denoiser and never leaves the input grid. Kept in lockstep with
    # `metadata.zs_deconvnet`, whose prediction the payload below must equal (§8).
    up = 2 if (upsample and which == "deconvolved") else 1
    out_y, out_x = ax.y * up, ax.x * up

    weights_path = ""
    trained: Dict[int, List[np.ndarray]] = {}
    if mode == "pretrained":
        weights_path = str(ctx.params.get("weights_path") or "").strip()
        if not weights_path:
            raise ValueError(
                "zs-deconvnet: `pretrained` mode needs a checkpoint — set `weights_path` to "
                "a ZS-DeconvNet .h5 file (the authors publish one per modality), or switch "
                "`mode` to `zero_shot` to train a model on this data instead.")
        if not os.path.isfile(weights_path):
            raise ValueError(f"zs-deconvnet: no such checkpoint {weights_path!r}")
        # Refuse a checkpoint whose sidecar says it was trained another way — the one failure
        # in this node that is otherwise completely silent (see `_check_checkpoint`). It is
        # handed the raw params rather than the resolved values, because what it has to decide
        # is precisely whether the user PINNED a contradicting one: an unset socket has
        # already been made to agree by `adopted` above and is not a conflict.
        verified = _check_checkpoint(weights_path, arch=arch, is_3d=is_3d,
                                     params=ctx.params)
        if verified:
            ctx.progress(0, 1, f"zs-deconvnet: {verified}")
        if adopted:
            ctx.progress(0, 1, "zs-deconvnet: took "
                               + ", ".join(f"{k}={adopted[k]}" for k in sorted(adopted))
                               + " from the checkpoint's own training record")
    else:
        cache = str(ctx.params.get("cache_path") or "").strip()
        iters = int(ctx.params.get("iterations", 2000))
        if iters < 1:
            raise ValueError(
                f"zs-deconvnet: `iterations`={iters} would train nothing, leaving a randomly "
                "initialized network that outputs noise. Set at least 1, or switch `mode` to "
                "`pretrained`.")
        seed = int(ctx.params.get("seed", 0))
        n_units = int(ctx.params.get("train_units", 8))
        for c in range(ax.c):
            psf = _psf_for_channel(ctx, c, is_3d=is_3d, px=px, zs=zs_um)
            sig = _train_signature(ctx, is_3d=is_3d, arch=arch, psf=psf,
                                   upsample=upsample)
            wpath, spath = _cache_paths(cache, c) if cache else ("", "")
            if wpath and os.path.isfile(wpath) and os.path.isfile(spath):
                try:
                    with open(spath, "r", encoding="utf-8") as fh:
                        # `zs_signature_of` strips the `__inference__` block before comparing,
                        # which is what makes that block ADDITIVE: a sidecar written before it
                        # existed reduces to itself and compares exactly as it always did, so
                        # no already-trained model is invalidated by a bookkeeping change
                        # (V2.23). Everything the signature has always covered still compares
                        # by equality, so changing `iterations` or the PSF still retrains.
                        if _zs_signature_of(json.load(fh)) == sig:
                            ctx.progress(0, 1, f"c{c}: reusing cached model "
                                               f"{os.path.basename(wpath)}")
                            trained[c] = None          # sentinel: load from wpath below
                            continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass                               # unreadable sidecar: retrain
            units = _read_units(prov, ax, c, is_3d=is_3d, limit=n_units)
            note = f"training c{c} ({arch}, {iters} it)"

            def _tick(step: int, total: int, loss: float, _n=note) -> None:
                ctx.progress(step, total, f"{_n} loss={loss:.4g}")

            if is_3d:
                w = zsk.train_3d(
                    units, psf, iterations=iters, arch=arch,
                    batch_size=int(ctx.params.get("batch_size", 3)),
                    patch=t_patch, patch_z=int(ctx.params.get("patch_z", 13)),
                    insert_xy=insert_xy, insert_z=insert_z,
                    upsample=upsample, start_lr=t_lr, hess_weight=t_hess,
                    seed=seed + c, progress=_tick)
            else:
                w = zsk.train_2d(
                    units, psf, iterations=iters,
                    batch_size=int(ctx.params.get("batch_size", 4)),
                    patch=t_patch, insert_xy=insert_xy, upsample=upsample,
                    start_lr=t_lr, hess_weight=t_hess,
                    denoise_weight=float(ctx.params.get("denoise_weight", 0.5)),
                    background=background,
                    beta1=float(ctx.params.get("beta1", 1.0)),
                    beta2=float(ctx.params.get("beta2", 0.0)),
                    alpha=float(ctx.params.get("alpha", 1.0)),
                    seed=seed + c, progress=_tick)
            trained[c] = w
            if wpath:
                shape = ((t_patch + 2 * insert_xy, t_patch + 2 * insert_xy,
                          int(ctx.params.get("patch_z", 13)) + 2 * insert_z, 1) if is_3d
                         else (t_patch + 2 * insert_xy, t_patch + 2 * insert_xy, 1))
                model = zsk.get_model(arch, shape, upsample=upsample,
                                      insert_xy=insert_xy, insert_z=insert_z, weights=w)
                zsk.save_weights(model, wpath)
                # The signature (what proves a cached model may be REUSED) plus the
                # inference record (what a later `pretrained` run ADOPTS). Two blocks
                # because they answer different questions and have different back-compat
                # obligations — see `_zs_inference_record`.
                with open(spath, "w", encoding="utf-8") as fh:
                    json.dump({**sig,
                               _INFERENCE_KEY: _zs_inference_record(
                                   arch=arch, is_3d=is_3d, upsample=upsample,
                                   insert_xy=insert_xy, insert_z=insert_z)},
                              fh, indent=1, sort_keys=True)

    # ── inference over every unit ────────────────────────────────────────────
    # Dense rather than a lazy provider: `upsample` CHANGES the lateral shape, which the
    # shape-preserving streaming providers cannot express, and a trained-per-pull model is
    # not something to re-run per displayed tile. `dense_output` spills to a memmap above
    # the spill budget, so a big series degrades to disk instead of MemoryError.
    out = dense_output((ax.m, ax.t, ax.z, ax.c, out_y, out_x), np.float32,
                       tag=f"zsdeconv_{ctx.node_id}")
    buf = out.array
    units = ([(m, t, c) for m in range(ax.m) for t in range(ax.t) for c in range(ax.c)]
             if is_3d else
             [(m, t, z, c) for m in range(ax.m) for t in range(ax.t)
              for z in range(ax.z) for c in range(ax.c)])
    total = len(units)
    take = 1 if which == "deconvolved" else 0
    cache_root = str(ctx.params.get("cache_path") or "").strip()
    # Two-level progress, because ONE unit is minutes of work here and a bar that only moves
    # when a unit lands reads as a hang. On the WellA3 series that is 784 units, so the first
    # per-unit tick is 1/784 = 0.13 % — it rounds to 0 %, and the node looked frozen for the
    # whole of the first plane. `_UnitBar` reports the tile fraction INSIDE the unit in
    # flight, which is real per-iteration progress (a plane is 25 tiles at tile=256), and the
    # model build + weight load gets its own sweep because it precedes the first tile and on
    # the 13 M-parameter 2D U-Net is not instant.
    warn = _cost_warning(tile=tile, insert_xy=insert_xy, upsample=upsample,
                         is_3d=is_3d, units=total)
    if warn:
        ctx.progress(0, 1, f"zs-deconvnet: {warn}")
    bar = _UnitBar(ctx, frames=max(1, ax.t),
                   units_per_frame=max(1, total // max(1, ax.t)),
                   note=f"{'deconvolving' if take else 'denoising'} ({arch})")
    for i, unit in enumerate(units):
        c = unit[-1]
        # Three ways a channel's weights arrive, and exactly one applies:
        #   pretrained            -> the user's checkpoint path
        #   zero_shot, trained    -> the in-memory arrays this pull produced
        #   zero_shot, cache hit  -> the cached file (`trained[c]` is the None sentinel)
        cw = trained.get(c) if mode == "zero_shot" else None
        wp = weights_path
        if mode == "zero_shot" and cw is None:
            if not cache_root:                 # unreachable: a hit requires a cache path
                raise ValueError(
                    f"zs-deconvnet: no weights for channel {c} — trained nothing and no "
                    "`cache_path` to load from. This is a bug; report the graph.")
            wp = _cache_paths(cache_root, c)[0]
        where = f"unit {i + 1}/{total}"
        # Building the graph and loading 52 MB of weights happens on the FIRST unit, before
        # any tile reports — say so, and sweep rather than sit at 0 %.
        if i == 0:
            bar.emit_unknown(f"{bar.note}: loading the model")

        def tick(done: int, n_tiles: int, _w=where) -> None:
            bar.emit(int(100 * done / max(1, n_tiles)),
                     f"{bar.note} {_w} — tile {done}/{n_tiles}")

        if is_3d:
            m, t, c = unit
            vol = np.asarray(prov.get_region_volume(0, m, t, c, 0, ax.z, 0, ax.y, 0, ax.x),
                             dtype=np.float32)
            res = zsk.infer_3d(
                vol, arch=arch, weights_path=wp, weights=cw, tile=tile,
                tile_z=int(ctx.params.get("tile_z", 32) or 0), overlap=overlap,
                overlap_z=int(ctx.params.get("overlap_z", 4) or 0), upsample=upsample,
                insert_xy=insert_xy, insert_z=insert_z,
                background=background, norm_low=norm_low,
                damping_length=int(ctx.params.get("damping_length", 0) or 0),
                damping_width=int(ctx.params.get("damping_width", 1) or 1),
                progress=tick)[take]
            buf[m, t, :, c] = res
        else:
            m, t, z, c = unit
            plane = np.asarray(prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x),
                               dtype=np.float32)
            res = zsk.infer_2d(plane, arch=arch, weights_path=wp, weights=cw, tile=tile,
                               overlap=overlap, upsample=upsample, insert_xy=insert_xy,
                               norm_low=norm_low, progress=tick)[take]
            buf[m, t, z, c] = res
        bar.finish_unit()
    # Through the BAR, not a bare ctx.progress: a plain call omits `frames`, so the two-level
    # report vanishes on the very last tick and the UI's frame bar would blank just as it
    # should read full. `_UnitBar.emit` already pins the sub bar full once done == total.
    bar.emit(100, f"{arch}: {total} unit(s) done")

    from dataclasses import replace as _replace
    result = ds.with_image(ArrayProvider(out.seal()))
    if up != 1:
        # Axes and calibration move in LOCKSTEP, and the pixel size is SYNCED from the
        # post-transform envelope rather than re-derived: `ctx.calib` already reflects
        # `metadata.zs_deconvnet`, so dividing by 2 again here would halve it twice (§8).
        result = result.reshaped_axes(_replace(ax, y=out_y, x=out_x))
        result = result.with_metadata(pixel_size_um=ctx.calib("pixel_size_um"))
    prov_md: Dict[str, Any] = {"bit_depth": None, "zsdeconv_mode": mode,
                               "zsdeconv_arch": arch, "zsdeconv_output": which}
    if mode == "pretrained" and weights_path:
        prov_md["zsdeconv_model"] = weights_path
    return result.with_metadata(**prov_md)


register_node(
    _compute_zs_deconvnet,
    op_key="enhance.zs_deconvnet", label="ZS-DeconvNet", category="enhancement",
    inputs=[
        InDataset(),
        # ── inference: read in BOTH modes ─────────────────────────────────────
        InBool("upsample", "2x upsample", field=False, default=True,
               description=
               "Whether the deconvolution head ends in a 2x lateral upsampling, doubling Y "
               "and X and halving pixel_size_um. This is how the paper reaches ~1.5x past "
               "the diffraction limit: the sharpened detail needs a finer grid to sit on. It "
               "COSTS the most of any control here — the final two 128-channel convolutions "
               "run at 4x the pixel count — and on this CPU-only build that is the difference "
               "between 0.3 s and 0.8 s for a 128 px tile. In `pretrained` mode it is NOT "
               "free to choose: it must match how the checkpoint was trained, or the weights "
               "will not load. Never applies to the `denoised` output, which always stays on "
               "the input grid."),
        InInt("tile", "Tile size", unit="px", field=False, default=128,
              description=
              "Lateral tile the network runs on, in pixels; tiles are fused on half-overlap "
              "seams. Bigger tiles mean fewer seams and slightly better continuity, but the "
              "cost grows WORSE than linearly with tile area because the wide activation "
              "maps stop fitting in cache: measured here with upsampling on, 0.8 s at 128 px, "
              "109 s at 256, 470 s at 512. So 128 is not a quality compromise so much as the "
              "only tractable choice on a CPU — raise it only if you can see seams. Clamped "
              "to the image size, and rounded down to an even number for the pooling "
              "architectures."),
        InInt("overlap", "Tile overlap", unit="px", field=False, default=20,
              description=
              "How much neighbouring tiles overlap laterally. Each output pixel is taken from "
              "whichever tile saw the most context around it (the seam is cut at the middle "
              "of the overlap), so this is the knob for seam artifacts: too small and a faint "
              "grid appears at tile boundaries, larger costs proportionally more tiles. 20 is "
              "the reference default. Ignored when the image fits in one tile."),
        InFloat("norm_low", "Output low percentile", unit="", field=False, default=3.0,
                description=
                "The low percentile the OUTPUT is normalized against, so it directly sets how "
                "much of the dim end is clipped to zero: at the default 3, the faintest 3 % of "
                "pixels become exactly 0. This is what the reference does before saving "
                "(prctile_norm(x,3,100)), and reproducing it is what makes results comparable "
                "with the published figures — but it DOES destroy dim signal, so set 0 for a "
                "plain min-max normalization when the result feeds intensity measurements. "
                "Some normalization is unavoidable: the network was trained on [0,1] input and "
                "its output scale is arbitrary."),
        # NO `derive` (fixed 2026-08-05). It carried `derive="0.0"`, a bare constant — the only
        # such socket in the catalog — while the compute's fallback is 100.0. A `derive` is what
        # the inspector keys the greyed "ƒ auto" box on, so the panel displayed 0 for a run that
        # used 100: the one number a user checks before trusting a pedestal subtraction was the
        # one number that was wrong. Removing it (rather than moving the compute to 0) keeps
        # every existing graph computing exactly what it computed before and makes the panel
        # agree with it.
        #
        # And nothing better is possible here: `envelope_symbols` carries no camera-pedestal
        # key — it cannot, since a pedestal is a property of the detector, not of the geometry
        # or optics an ND2 records — so any expression would be a constant wearing the badge of
        # a metadata-derived value. This param is measured from the data (a dark corner) or
        # taken from a camera calibration; that is a user decision, and a plain field is the
        # honest control for it.
        InFloat("background", "Background offset", unit="", field=False, default=100.0,
                description=
                "Camera pedestal subtracted before the data is normalized, in the image's own "
                "intensity units. Real sCMOS frames sit on an offset the network was never "
                "trained to see, and leaving it in compresses the useful range. INERT for 2D "
                "inference, which follows the reference in not subtracting anything; it is "
                "read for 3D inference and for 2D zero-shot training, where it also sets the "
                "point below which the Poisson noise term stops growing. The reference uses "
                "100 for its LLSM data and 0 for confocal — check a dark corner of your own "
                "frame."),
        # Per-dim sockets, the `min_area`/`min_volume` pattern: the reference uses a DIFFERENT
        # value in 2D and 3D for each of these four, so one socket with one default could only
        # ever be right in one dim — and the default has to live in the SocketSpec exactly once
        # (socket contract clause 5), which rules out a per-dim fallback in the compute.
        InInt("insert_xy", "Padding margin", unit="px", field=False, default=16,
              available_in={"dim": frozenset({"2D"})},
              description=
              "Zero-padding margin added around every tile before it enters the network, and "
              "cropped off after. It exists because a deconvolution has no valid answer at an "
              "edge it cannot see past, so without it every tile boundary rings. The ACTUAL "
              "margin used may be larger: the pooling architectures need the padded tile to be "
              "a multiple of 16, so it is grown until it is. 16 is the reference's 2D value. In "
              "`pretrained` mode it should match the checkpoint's training value or the edges "
              "of each tile will be interpreted differently than in training. 2D only — 3D has "
              "its own socket, because the reference pads 3D volumes less."),
        InInt("insert_xy_3d", "Padding margin", unit="px", field=False, default=8,
              available_in={"dim": frozenset({"3D"})},
              description=
              "The same lateral padding margin, for 3D. Half the 2D value (the reference uses "
              "8) because a padded voxel in a volume costs a whole extra plane of computation "
              "per pixel of margin, and the 3D backbones have a shorter receptive field than "
              "the 2D U-Net so they need less context. Must match the checkpoint in "
              "`pretrained` mode — the authors' 3D models were all trained at 8. 3D only."),
        # ── mode = pretrained ────────────────────────────────────────────────
        InString("weights_path", "Checkpoint", field=False, default="",
                 path_kind="open_file",
                 path_filter="Keras weights (*.h5 *.hdf5 *.weights.h5);;All files (*)",
                 path_hint="required in `pretrained` mode — a ZS-DeconvNet .h5",
                 available_in={"mode": frozenset({"pretrained"})},
                 description=
                 "The trained network to run. Required in `pretrained` mode. The authors "
                 "publish one checkpoint per modality and structure (WF lysosome, LLSM "
                 "mitochondria, confocal microtubule, ...); a model trained on your own data "
                 "by `zero_shot` + `cache_path` also goes here. Legacy Keras-2 .h5 files are "
                 "matched to the graph by layer ORDER with no name check, so a checkpoint for "
                 "the wrong architecture can appear to load and then predict plausible "
                 "nonsense — set `arch_3d`, `upsample` and the dim lever to whatever the "
                 "checkpoint was trained with. The paper's own warning applies: a model "
                 "applied to optics unlike its training data has a real risk of "
                 "hallucination (Supplementary Fig. 28b)."),
        # ── mode = zero_shot ─────────────────────────────────────────────────
        InString("cache_path", "Save/reuse model", field=False, default="",
                 path_kind="save_file",
                 path_filter="Keras weights (*.weights.h5);;All files (*)",
                 path_hint="empty = retrain on every recompute (slow)",
                 available_in={"mode": frozenset({"zero_shot"})},
                 description=
                 "Where to persist the model this node trains — and, on a later run, where to "
                 "reuse it from instead of training again. LEAVE THIS SET: training is the "
                 "expensive part (tens of minutes to hours per channel), and without a cache "
                 "any recompute pays it again. One file per channel is written (`_c0`, `_c1`, "
                 "...) because the PSF differs per emission wavelength, alongside a small "
                 ".json recording the training settings — a cached model is reused ONLY when "
                 "those still match, so changing `iterations` or the PSF retrains rather than "
                 "silently returning the old network."),
        InInt("iterations", "Iterations", field=False, default=2000,
              available_in={"mode": frozenset({"zero_shot"})},
              description=
              "Training steps, per channel. The single biggest quality/time lever, and "
              "runtime is LINEAR in it. The paper uses 50000 for 2D and 10000 for 3D on an "
              "RTX 3090 (~1 h and ~2 h); this build has NO GPU — TensorFlow >= 2.11 dropped "
              "native-Windows CUDA — and measures ~1.4 s per 2D iteration, so 50000 is about "
              "19 hours and the default 2000 about 45 minutes. 2000 gives a usable, visibly "
              "denoised and sharpened preview, NOT paper-quality super-resolution; raise it "
              "(with `cache_path` set) when the preview looks worth the wait. The paper's own "
              "test-time-adaptation trick is only 50 extra steps on top of a trained model, "
              "which is why fine-tuning a cached checkpoint is cheap."),
        InInt("train_units", "Training units", field=False, default=8,
              available_in={"mode": frozenset({"zero_shot"})},
              description=
              "How many planes (2D) or volumes (3D) are sampled from this dataset to train "
              "on, spread by an even stride over positions, timepoints and z rather than "
              "taken from the start — the first frames of a series are its most redundant "
              "part. More units means a model that generalizes across the whole dataset "
              "instead of over-fitting one field; it does NOT cost more time, since the "
              "iteration count is what sets that. Each unit is held in RAM for the duration "
              "of training, so a large value on big volumes is a memory cost."),
        InInt("seed", "Random seed", field=False, default=0,
              available_in={"mode": frozenset({"zero_shot"})},
              description=
              "Seeds weight initialization, patch selection and the re-corruption noise, so "
              "the same graph retrains to the same model. It is a real parameter, not a "
              "convenience: without a fixed seed this node would not be a pure function of "
              "its inputs and the memo could hand back a payload no re-run reproduces. Change "
              "it to see how much of the result is training variance — on a short run, more "
              "than you might expect."),
        InInt("batch_size", "Batch size", field=False, default=4,
              available_in={"mode": frozenset({"zero_shot"})},
              description=
              "Patches per training step (reference: 4 for 2D, 3 for 3D). Larger batches make "
              "each step's gradient less noisy but proportionally slower, so at a FIXED "
              "iteration count raising this costs time without covering more ground — on a "
              "short CPU budget prefer more iterations over bigger batches. Memory scales "
              "with it too, which is the binding constraint in 3D."),
        InInt("patch", "Patch size", unit="px", field=False, default=128,
              available_in={"mode": frozenset({"zero_shot"}),
                            "dim": frozenset({"2D"})},
              description=
              "Lateral size of the training patches, cut around detected foreground so they "
              "land on structure rather than empty coverslip. It sets how much spatial context "
              "the network learns from — too small and it cannot see a whole object — and cost "
              "grows with its AREA. 128 is the reference's 2D value. 2D only; 3D trains on "
              "smaller patches because a volume of the same width costs depth times as much."),
        InInt("patch_3d", "Patch size", unit="px", field=False, default=64,
              available_in={"mode": frozenset({"zero_shot"}),
                            "dim": frozenset({"3D"})},
              description=
              "Lateral size of the 3D training patches (reference: 64). Half the 2D width "
              "because the patch is a VOLUME — at 64x64x13 it already holds more voxels than a "
              "128x128 plane — so this is the main cost and memory control during 3D training. "
              "With the `unet3d` architecture this plus twice the padding margin must be a "
              "multiple of 16, so the default 64 + 2x8 = 80 is REFUSED there (use 48 or 112); "
              "the default `rcan3d` never pools and accepts any size. 3D only."),
        InFloat("learning_rate", "Learning rate", unit="", field=False, default=5e-5,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "Initial Adam step size; halved partway through training on the reference's "
                "schedule. 5e-5 is the published 2D value. Raising it makes a short run "
                "converge further before the budget runs out, at the risk of a diverging or "
                "artifact-prone model — the safer way to buy quality is `iterations`. Leave it "
                "alone unless the training loss is visibly flat. 2D only."),
        InFloat("learning_rate_3d", "Learning rate", unit="", field=False, default=1e-4,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"3D"})},
                description=
                "Initial Adam step size for 3D — twice the 2D value (the reference uses 1e-4), "
                "which goes with 3D's much smaller iteration budget: 10000 steps against "
                "50000, so each one has to move further. Halved at the halfway and "
                "three-quarter marks. 3D only."),
        InFloat("hess_weight", "Hessian weight", unit="", field=False, default=0.02,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "Weight of the Hessian regularizer on the deconvolved output — the term that "
                "keeps a deconvolution from turning noise into speckle and bright edges into "
                "rings. HIGHER gives a smoother, safer result and eventually erases genuine "
                "fine detail; LOWER sharpens and eventually produces the artifacts the "
                "regularizer exists to prevent. 0.02 is the paper's 2D value, and it reports "
                "the result is stable over a wide range of it (Supplementary Fig. 1b-e). 0 "
                "disables it. 2D only."),
        InFloat("hess_weight_3d", "Hessian weight", unit="", field=False, default=0.1,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"3D"})},
                description=
                "The same Hessian regularizer, for 3D, where the paper sets it FIVE TIMES "
                "higher (0.1) — a volumetric deconvolution has an extra axis to ring along and "
                "the axial direction is the worst-conditioned one, so it needs more smoothing "
                "to stay artifact-free. Lower it if genuine axial detail looks washed out. "
                "3D only."),
        InFloat("denoise_weight", "Denoise loss weight", unit="", field=False, default=0.5,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "How the total loss is split between the two stages: this much on stage I's "
                "denoising term, the rest on stage II's deconvolution term (the paper's mu, "
                "Eq. 3). At 0 the denoiser is untrained and stage II is fed noise; at 1 "
                "nothing trains the deconvolution at all. The paper sets 0.5 and shows "
                "performance is stable across a wide range of it (Supplementary Fig. 29). 2D "
                "only — the 3D losses weight their two terms equally by construction."),
        # ── zero_shot + 2D: the re-corruption noise model (paper Eq. 9-12) ────
        InFloat("alpha", "Recorrupt alpha", unit="", field=False, default=1.0,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "Splits one noisy image into the two noise-independent copies training needs: "
                "the input gets alpha x sigma of extra noise and the target gets sigma/alpha "
                "of the opposite. It therefore trades how hard the input is to denoise against "
                "how noisy the target is. The paper proves 1 is optimal (Supplementary Note 1, "
                "Figs. 3-4) and this node keeps it, jittering +/-50 % per step as augmentation "
                "exactly as the authors' MATLAB does. 2D only: the 3D path splits axial "
                "parities instead and needs no noise model at all."),
        InFloat("beta1", "Poisson factor", unit="", field=False, default=1.0,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "Scales the SIGNAL-dependent (shot-noise) half of the noise model: the "
                "estimated variance is beta1 x (smoothed intensity - background) + beta2. For "
                "a photon-counting detector the true value is 1, which the paper also derives "
                "as optimal, so change it only if your camera's gain means one count is not "
                "one photon. Too high over-corrupts bright structure and the model learns to "
                "smooth it away. 2D only."),
        InFloat("beta2", "Read-noise variance", unit="", field=False, default=0.0,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"2D"})},
                description=
                "The signal-INDEPENDENT half of the noise model — your camera's read-noise "
                "variance in counts squared, the one re-corruption parameter that is not "
                "theoretically 1 and genuinely depends on hardware. 0 means pure shot noise, "
                "which under-corrupts a real sCMOS frame and leaves fine noise the model "
                "treats as signal. The authors estimate it from background pixels of the data "
                "itself (their demo lands near 10-15 counts squared for 12-bit sCMOS); a "
                "camera calibration is better. 2D only."),
        # ── zero_shot + 3D ───────────────────────────────────────────────────
        InInt("patch_z", "Patch depth", field=False, default=13,
              available_in={"mode": frozenset({"zero_shot"}),
                            "dim": frozenset({"3D"})},
              description=
              "Planes per training patch, per axial parity — so each patch is cut from 2x "
              "this many consecutive planes and split into an input half and a target half. "
              "The volume must therefore have at least twice this many planes. It sets how "
              "much axial context the network learns, and memory scales with it. Reference: "
              "13 (i.e. 26 source planes). 3D only."),
        InInt("insert_z", "Axial margin", field=False, default=2,
              available_in={"dim": frozenset({"3D"})},
              description=
              "Zero-padded margin in PLANES at the top and bottom of every volume, cropped "
              "off afterwards — the axial counterpart of the lateral padding margin, and it "
              "is what stops the first and last planes from ringing. Small because z is "
              "coarsely sampled and each padded plane is a whole plane of wasted computation. "
              "Reference: 2. 3D only."),
        InInt("tile_z", "Tile depth", field=False, default=32,
              available_in={"dim": frozenset({"3D"})},
              description=
              "Planes per inference tile; 0 processes the whole stack at once. This is the "
              "main MEMORY control in 3D — the RCAN never pools, so it holds 64-channel "
              "activations at FULL tile resolution and depth multiplies that directly. The "
              "ceiling is lower than the machine's: TensorFlow's CPU allocator caps its arena "
              "at 64 GiB no matter how much RAM is installed, and the authors' own 151-plane "
              "demo tiling exceeds it (one activation there is 4.3 GiB). Axial seams are fused "
              "like lateral ones, so tiling costs continuity only at the `overlap_z` "
              "boundaries. 3D only."),
        InInt("overlap_z", "Tile depth overlap", field=False, default=4,
              available_in={"dim": frozenset({"3D"})},
              description=
              "Planes of overlap between axial tiles, cut at the middle like the lateral "
              "seams. Smaller than the lateral overlap because planes are expensive and the "
              "network's axial receptive field is short. Reference: 4. Ignored when `tile_z` "
              "is 0 or covers the stack. 3D only."),
        InInt("damping_length", "Fourier damping length", field=False, default=0,
              available_in={"dim": frozenset({"3D"})},
              description=
              "Removes sCMOS fixed-pattern stripes by zeroing a vertical band through each "
              "plane's Fourier transform, this many frequency rows tall from each end; 0 "
              "disables it. Column-to-column gain differences are IDENTICAL in both training "
              "copies, so the self-supervised scheme cannot learn them away and the "
              "deconvolution stage amplifies them into visible stripes. The reference enables "
              "this for lattice light-sheet data (length 450) and not for confocal. It DELETES "
              "real spatial frequencies along one axis, so it is off here by default and "
              "calibrating the camera is the proper fix. 3D only."),
        InInt("damping_width", "Fourier damping width", field=False, default=1,
              available_in={"dim": frozenset({"3D"})},
              description=
              "Half-width of that damping band: 2 x this + 1 frequency columns are zeroed. "
              "Widen it only if stripes survive at width 1 — every extra column removes more "
              "genuine horizontal structure along with the artifact. Inert while "
              "`damping_length` is 0. 3D only."),
        # ── the PSF (training-time only — inference never uses it) ───────────
        InString("psf_path", "PSF file", field=False, default="",
                 path_kind="open_file",
                 path_filter="PSF image (*.tif *.tiff);;All files (*)",
                 path_hint="empty = derive a Gaussian PSF from NA / emission / pixel size",
                 available_in={"mode": frozenset({"zero_shot"})},
                 description=
                 "A measured or simulated PSF as a TIFF, used ONLY while training (the "
                 "deconvolution loss re-blurs the network's output with it; inference never "
                 "touches a PSF). Leave EMPTY and an anisotropic Gaussian PSF is derived from "
                 "the file's own NA, emission wavelength and pixel/z size — the same "
                 "derivation `Deconvolve` uses, resolved per channel, and enough for most "
                 "work. Supply a file when you have measured your system's PSF with beads: "
                 "the paper's Supplementary Fig. 28c shows a WRONG PSF gives either no "
                 "resolution gain or ringing, and here that mistake is baked into the trained "
                 "weights. .mrc OTFs are refused — they need microscope-specific radial "
                 "sampling constants this port does not carry."),
        InFloat("psf_pixel_um", "PSF pixel size", unit="um", field=False, default=0.0,
                available_in={"mode": frozenset({"zero_shot"})},
                description=
                "The lateral pixel size the PSF FILE was sampled at, which is usually finer "
                "than the data's — the authors' simulated PSFs are 0.0313 um. The PSF is "
                "rescaled by this / your pixel size before use, so getting it wrong stretches "
                "or shrinks the assumed blur and mis-sharpens by the same factor. 0 means "
                "\"already on my grid\" and skips rescaling. Inert unless `psf_path` is set."),
        InFloat("psf_z_um", "PSF z step", unit="um_axial", field=False, default=0.0,
                available_in={"mode": frozenset({"zero_shot"}),
                              "dim": frozenset({"3D"})},
                description=
                "The axial step the 3D PSF file was sampled at (the authors' simulated stacks "
                "use 0.05 um), rescaled to your z step the same way the lateral size is. "
                "Axial and lateral sampling are independent on a real microscope, so one "
                "factor cannot serve both. 0 skips axial rescaling. Inert unless `psf_path` is "
                "set. 3D only."),
        InFloat("na", "NA", unit="", field=True, derive="na or 1.4",
                available_in={"mode": frozenset({"zero_shot"})},
                description=
                "Numerical aperture, one half of the derived PSF's width. A HIGHER NA means a "
                "tighter PSF, so the loss assumes less blur and asks for a gentler "
                "deconvolution; too low and it assumes more blur than the optics produced and "
                "trains an over-sharpening, ring-prone model. Reads the file's own NA on auto "
                "(falling back to 1.4); one value applies to every channel. Inert when "
                "`psf_path` is set, and unused entirely in `pretrained` mode."),
        InFloat("emission_nm", "Emission wavelength", unit="nm", field=True,
                derive="emission_nm or 520",
                available_in={"mode": frozenset({"zero_shot"})},
                description=
                "Emission wavelength, the other half of the derived PSF width — longer "
                "wavelengths diffract more, giving a wider PSF and a stronger correction. "
                "Resolved PER CHANNEL from the file's own emission list on auto, which is why "
                "a two-colour stack trains two properly different models; setting it here "
                "overrides every channel with one value, so prefer auto. Falls back to 520 nm. "
                "Inert when `psf_path` is set, and unused in `pretrained` mode."),
    ],
    outputs=[OutDataset()],
    modes=[
        DimMode(),
        Mode("mode", list(_MODES), default="zero_shot", label="Model",
             description=
             "Where the network comes from — the difference between the paper's zero-shot "
             "method and simply running someone else's trained model.",
             choice_docs={
                 "zero_shot":
                     "TRAIN on the data in front of you, then run it. This is the actual "
                     "method: no ground truth and no reference dataset, because the training "
                     "pairs are manufactured from the measurement itself (2D by re-corrupting "
                     "it into two noise-independent copies, 3D by splitting alternate z "
                     "planes). Needs a PSF, which it derives from your metadata by default. "
                     "Costs tens of minutes to hours per channel on this CPU-only build, so "
                     "set `cache_path` and pay it once.",
                 "pretrained":
                     "Load a .h5 checkpoint and infer only — seconds to minutes, no PSF, no "
                     "training. Use it for the authors' published models or to re-apply a "
                     "model this node trained earlier. The catch is generalization: the paper "
                     "reports noticeable degradation and a real hallucination risk when a "
                     "model meets optics unlike its training data (Supplementary Fig. 28b), "
                     "and the authors' checkpoints are TIRF/LLSM/confocal at 0.03-0.09 um/px.",
             }),
        Mode("output", list(_OUTPUTS), default="deconvolved", label="Output",
             description=
             "Which of the dual-stage network's two heads becomes this node's image. Both are "
             "computed either way — the choice costs nothing, it only selects.",
             choice_docs={
                 "deconvolved":
                     "Stage II: denoised AND sharpened, the result the node exists for, and "
                     "the one the paper's resolution claims refer to. Optionally 2x upsampled, "
                     "which changes the lateral axes and pixel size. Being a deconvolution it "
                     "is the more interpretive of the two — every artifact risk in the paper's "
                     "limitations section applies to this head.",
                 "denoised":
                     "Stage I only: noise removed, resolution untouched, always on the input "
                     "grid so axes and pixel size never change. The conservative choice, and "
                     "the right one when the next node measures intensities or you want the "
                     "denoising benefit without trusting a deconvolution. Comparable to a "
                     "dedicated self-supervised denoiser (paper, Supplementary Fig. 15).",
             }),
        Mode("arch_3d", list(_ARCH_3D), default="rcan3d", label="3D backbone",
             available_in={"dim": frozenset({"3D"})},
             description=
             "Which volumetric network the 3D lever uses. Both are genuinely 3D (Conv3D "
             "throughout, z-connected), and both upsample laterally only.",
             choice_docs={
                 "rcan3d":
                     "Residual channel-attention network — the paper's 3D backbone (Fig. 3a) "
                     "and the reference's default. NO pooling anywhere, so any tile size works "
                     "and there is no divisibility rule to satisfy; ~2 M parameters, far "
                     "smaller and cheaper per voxel than the 2D U-Net. The right choice unless "
                     "you have a checkpoint that says otherwise.",
                 "unet3d":
                     "A 3D U-Net with lateral-only pooling, offered because the reference does "
                     "and because some published checkpoints use it. Pooling means the padded "
                     "tile AND the training patch must be multiples of 16 laterally, which is "
                     "refused rather than silently adjusted, and its wider layers cost more "
                     "memory than the RCAN.",
             }),
    ],
    granularity={"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME},
    kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    supports_2d=True, supports_true_3d=True,
    meta_transform=_meta_zs_deconvnet,
    # In `pretrained` mode the graph-shaping sockets default to what the CHECKPOINT was
    # trained with, from the sidecar beside it (V2.23). `metadata.zs_deconvnet` consults the
    # same resolver, so an adopted `upsample` moves the predicted axes and the produced axes
    # together (§8). See `_zs_trained`.
    trained_params=_zs_trained,
    description="Zero-shot deconvolution network (Qiao et al. 2024): a dual-stage CNN "
                "trained self-supervised on the incoming data itself — no ground truth — or "
                "a published checkpoint run inference-only. Denoises and sharpens past the "
                "diffraction limit, optionally on a 2x finer grid. Output is percentile-"
                "normalized, so the declared bit depth is dropped.",
)
