"""trained — read the parameters a MODEL was trained with, from the JSON beside its weights.

WHY THIS EXISTS
    Three of the catalog's nodes run someone else's trained network, and every one of
    them ships a JSON file recording how it was trained:

    * a StarDist model directory carries ``thresholds.json`` (the ``prob``/``nms``
      thresholds ``optimize_thresholds`` tuned for THAT checkpoint) and ``config.json``
      (``n_dim``, ``n_channel_in``, the training ``anisotropy``, the ray count, …);
    * a ZS-DeconvNet checkpoint this app trained carries the sidecar
      ``<root>_c<N>.json`` that ``enhance.zs_deconvnet`` writes.

    Until V2.23 the nodes ignored those files and shipped static socket defaults, so a
    control that LOOKED like a neutral starting point silently overrode the value the
    checkpoint's own author measured — ``analysis.segment`` passed ``prob_thresh=0.5``
    over the ``3D_demo`` model's tuned **0.708**, and its own socket documentation
    admitted it. This module is the one place that reads those files, so the GUI and the
    compute can never disagree about what a loaded model was trained with.

THE CONTRACT — every function here is TOTAL
    These run inside ``propagate_meta`` and the inspector rebuild, i.e. on **every
    keystroke**, exactly like ``NodeSpec.extra_layers``. So nothing here may raise, and
    nothing here may be slow: every read goes through :func:`read_json` , which caches on
    ``(abspath, st_mtime_ns, st_size)`` and returns ``None`` for anything it cannot parse.
    A missing, unreadable or nonsense file is not an error — it means "this model states
    nothing", and the caller falls back to its socket default.

    Keying on mtime+size rather than path alone is load-bearing: **retraining to the same
    filename must re-key**, which is the same argument
    ``kernels.zsdeconvnet.get_model`` makes for its own model cache.

WHAT IT DELIBERATELY DOES NOT DO
    It never imports tensorflow, torch, stardist or csbdeep — reading a small JSON must
    not cost a framework import in the GUI process, and this module is reached from
    ``metadata.py`` (Qt-free, dependency-light) during edit-time propagation. It also
    never *decides* anything: it reports what the file says, and the node decides whether
    to adopt it, warn about it, or refuse.

    CellSAM has no per-model JSON (its checkpoint is a bare torch state dict and its
    configuration lives in the ``cellSAM`` package's own ``modelconfig.yaml``, not beside
    the weights), so there is no CellSAM resolver here and ``analysis.segment`` adopts
    nothing for that method.
"""

from __future__ import annotations

import json
import os
import threading

from typing import Any, Dict, Mapping, Optional, Tuple

# ── the total, mtime-keyed JSON reader ────────────────────────────────────────

#: ``(abspath, mtime_ns, size) -> parsed dict``. Unbounded, and that is fine: one entry is
#: a handful of keys and the keys are model files a session touches a few of. Guarded by a
#: lock because the inspector rebuild (GUI thread) and a pull (worker thread) can both
#: resolve the same node's params at once.
_CACHE: Dict[Tuple[str, int, int], Optional[Dict[str, Any]]] = {}
_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop the parsed-JSON cache. For the selftest, which writes a model JSON, reads it,
    rewrites it and reads it again — inside one ``st_mtime_ns`` tick on a filesystem whose
    timestamp granularity is coarser than the test is fast, the (path, mtime, size) key
    would otherwise be identical for two different contents."""
    with _LOCK:
        _CACHE.clear()


def read_json(path: str) -> Optional[Dict[str, Any]]:
    """The parsed JSON object at ``path``, or ``None`` — never raises.

    ``None`` covers every way this can fail to produce a mapping: the file is absent, it is
    a directory, it is unreadable, it is not valid JSON, or it parses to something that is
    not an object (a bare list or number). All of those mean the same thing to every caller
    — "this model states nothing" — so they are deliberately not distinguished.
    """
    if not path:
        return None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    parsed: Optional[Dict[str, Any]] = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            parsed = raw
    except (OSError, ValueError):
        parsed = None
    with _LOCK:
        _CACHE[key] = parsed
    return parsed


def _clean_path(value: Any) -> str:
    """A user-typed path, normalized the way the inspector's own committer does — a
    Windows "Copy as path" paste arrives wrapped in double quotes, and a path socket's
    value reaches here straight from the params dict."""
    return str(value or "").strip().strip('"').strip()


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None      # reject NaN / inf


# ── StarDist ──────────────────────────────────────────────────────────────────

#: The keras/csbdeep model-zoo class directory for each dim. ``csbdeep.models.pretrained.
#: get_model_folder`` builds ``<keras cache>/models/<cls.__name__>/<key>`` and extracts the
#: downloaded zip into it, so the class name is part of the path and 2D/3D never collide.
_SD_CLASS = {2: "StarDist2D", 3: "StarDist3D"}


def _keras_cache_root() -> str:
    """Where ``keras.utils.get_file`` puts downloads — ``$KERAS_HOME``, else ``~/.keras``.

    Replicated rather than imported: reaching it through keras would mean importing
    TensorFlow to learn a directory name, and this module is called from the GUI's
    edit-time propagation where that is unacceptable. The consequence of the replica
    drifting is benign in exactly one direction — the directory is not found, so a
    pretrained model reports no trained values and its sockets show their static defaults,
    which is where they were before this module existed.
    """
    home = os.environ.get("KERAS_HOME", "").strip()
    if home:
        return home
    return os.path.join(os.path.expanduser("~"), ".keras")


def stardist_model_dir(params: Mapping[str, Any], modes: Mapping[str, Any]) -> str:
    """The directory holding ``config.json``/``thresholds.json`` for the model
    ``analysis.segment`` would load right now, or ``""`` if there is not one on disk.

    Mirrors ``kernels.stardist_segment.get_stardist_model``'s own resolution order, and it
    has to: a value shown beside a socket that came from a DIFFERENT checkpoint than the
    pull will load is worse than showing nothing.

    * ``sd_model_path`` wins in both dims (a locally trained model directory);
    * otherwise the pretrained name for the active dim, looked up in the keras cache.

    A pretrained model that has never been downloaded has no directory yet, so this
    returns ``""`` and the sockets keep their static defaults until the first pull fetches
    it. That is honest — the file genuinely does not exist — and it costs the RUN nothing,
    because the compute leaves an unset threshold as ``None`` and StarDist then reads the
    freshly-downloaded ``thresholds.json`` itself.
    """
    local = _clean_path(params.get("sd_model_path"))
    if local:
        # `config.json` is required here, not merely a directory, so that this function means
        # ONE thing on both branches: "a directory that is actually a StarDist model". csbdeep
        # writes config.json for every model it trains, so a real local model always passes —
        # and a path that does not is better answered with "states nothing" (sockets keep their
        # defaults) than with a directory whose thresholds.json is then looked for in vain.
        # The node still refuses such a path at pull time, with a message naming what is
        # missing; that refusal is `get_stardist_model`'s job, not this reporter's.
        path = os.path.abspath(os.path.expanduser(local))
        return path if os.path.isfile(os.path.join(path, "config.json")) else ""
    is_3d = str((modes or {}).get("dim") or "2D").upper() == "3D"
    name = str(params.get("model_name_3d") if is_3d else params.get("model_name")
               or "").strip()
    if not name:
        name = "3D_demo" if is_3d else "2D_versatile_fluo"
    base = os.path.join(_keras_cache_root(), "models", _SD_CLASS[3 if is_3d else 2], name)
    # Two candidate leaf names, both real: Keras < 3.6 extracted to ``<key>/`` while
    # >= 3.6 extracts to ``<key>_extracted/`` and symlinks ``<key>`` at it. The symlink
    # needs a privilege Windows does not grant by default, so on this platform the
    # ``_extracted`` directory is frequently the only one that exists.
    for leaf in (name, f"{name}_extracted"):
        cand = os.path.join(base, leaf)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return ""


def stardist_config(params: Mapping[str, Any],
                    modes: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The model's own ``config.json`` — the architecture record csbdeep wrote at training
    time (``n_dim``, ``axes``, ``n_channel_in``, ``n_rays``, ``grid``, ``anisotropy``, and
    the whole ``train_*`` block). ``None`` when there is no model directory.

    Read for **validation and reporting**, never to set a socket: everything in here that
    describes the network is already applied by csbdeep when it loads the directory with
    ``config=None``. What the node uses it for is telling the user when their data does not
    match what the checkpoint expects — a mismatch that otherwise surfaces as a shape error
    from inside TensorFlow, or as a quietly worse segmentation.
    """
    d = stardist_model_dir(params, modes)
    return read_json(os.path.join(d, "config.json")) if d else None


def stardist_note(params: Mapping[str, Any], modes: Mapping[str, Any]) -> str:
    """One line for the GUI saying WHICH file the StarDist auto values came from — or why
    there are none (V2.23b).

    This exists because the silent case is indistinguishable from a broken feature. Reported
    as "loading in the model does not change any of the parameters": a path that is not a
    model directory, and a pretrained checkpoint that has not been downloaded yet, both
    resolve to "this model states nothing" — which is the correct answer for
    :func:`stardist_trained` and a useless one for the person looking at the panel. The
    values silently stayed at their socket defaults with no badge and no reason.
    """
    is_3d = str((modes or {}).get("dim") or "2D").upper() == "3D"
    local = _clean_path(params.get("sd_model_path"))
    d = stardist_model_dir(params, modes)
    if not d:
        if local:
            return (f"{os.path.basename(local.rstrip('/\\')) or local} is not a StarDist "
                    f"model folder (no config.json in it) — pick the folder that HOLDS "
                    f"config.json and weights_best.h5")
        name = str((params.get("model_name_3d") if is_3d else params.get("model_name"))
                   or ("3D_demo" if is_3d else "2D_versatile_fluo"))
        return (f"{name} is not downloaded yet — its tuned thresholds appear here after the "
                f"first run fetches it")
    got = stardist_trained(params, modes)
    where = os.path.basename(d.rstrip("/\\")) or d
    if not got:
        return (f"{where} records no usable thresholds.json — the values below are this "
                f"app's defaults, not the checkpoint's")
    return (f"prob/nms below come from {where}'s own thresholds.json "
            f"({', '.join(f'{k.split(chr(95))[0]} {v:g}' for k, v in sorted(got.items()))})")


def stardist_trained(params: Mapping[str, Any],
                     modes: Mapping[str, Any]) -> Dict[str, Any]:
    """``{socket_name: trained_value}`` for ``analysis.segment``'s StarDist sockets.

    Only the two detection thresholds, because they are the only entries in a StarDist
    model directory that correspond to a socket on this node: ``thresholds.json`` is what
    ``StarDist*.optimize_thresholds`` writes after measuring the checkpoint against its own
    validation set, and it is the value StarDist itself uses when you pass ``None``.

    The ``0 < x < 1`` test is **StarDist's own** (``StarDistBase.__init__`` rejects an
    out-of-range entry and falls back to its built-in 0.5/0.4). Replicating it here is what
    keeps the number the GUI displays equal to the number the network will actually use —
    if the file says ``prob: 3``, StarDist ignores it, so this must ignore it too rather
    than showing the user a 3 that never applies.
    """
    d = stardist_model_dir(params, modes)
    if not d:
        return {}
    th = read_json(os.path.join(d, "thresholds.json"))
    if not th:
        return {}
    out: Dict[str, Any] = {}
    for socket, key in (("prob_thresh", "prob"), ("nms_thresh", "nms")):
        v = _as_float(th.get(key))
        if v is not None and 0.0 < v < 1.0:
            out[socket] = v
    return out


# ── ZS-DeconvNet ──────────────────────────────────────────────────────────────

#: The sidecar block holding the EFFECTIVE inference-relevant parameters of a training
#: run, written by ``enhance.zs_deconvnet`` beside a checkpoint it trained.
#:
#: It is a separate top-level key rather than more entries in the training signature for one
#: reason: the signature is compared for **equality** to decide whether a cached model may
#: be reused, and a sidecar written by an older build would then stop matching and silently
#: trigger a retrain — which on this CPU-only build is tens of minutes to hours per channel.
#: Excluding this key from that comparison (:func:`zs_signature_of`) keeps every existing
#: cache entry valid while new sidecars gain the full record.
INFERENCE_KEY = "__inference__"

#: Sockets a ZS-DeconvNet sidecar can speak for. The test each one passes is narrow and
#: deliberate: **it describes the trained GRAPH, so a value that disagrees with the
#: checkpoint is simply wrong, not merely different.**
#:
#: * ``upsample`` — the 2x head. Its up-sampling layers carry no weights, so a mismatch
#:   loads CLEANLY and then predicts an image the second stage was never trained to
#:   produce. That silent failure is ``_check_checkpoint``'s whole reason for existing.
#: * ``insert_xy`` / ``insert_xy_3d`` / ``insert_z`` — the padding margins, which are baked
#:   into the graph as per-axis crops. Their socket docs already told the user to match the
#:   checkpoint by hand, and nothing checked that they had.
#:
#: NOT here, and the omissions matter:
#:
#: * ``tile``/``overlap``/``norm_low`` are free inference choices — the training run says
#:   nothing about them and a bigger tile is not "wrong".
#: * ``iterations``/``patch``/``learning_rate``/the re-corruption noise model are
#:   training-only and inert once weights exist.
#: * ``background`` is recorded in the signature (it entered the training loss) but is
#:   deliberately NOT adopted: it is a property of the CAMERA that produced the pixels being
#:   processed now, not of the network. Running a published checkpoint on your own data is
#:   precisely the case where the training set's pedestal is the wrong number, so adopting it
#:   would import a stranger's camera offset into your intensities.
ZS_INFERENCE_SOCKETS: Tuple[str, ...] = (
    "upsample", "insert_xy", "insert_xy_3d", "insert_z")


def zs_sidecar_path(weights_path: str) -> str:
    """The ``.json`` written beside a ZS-DeconvNet checkpoint, derived from the weights
    path alone (``<root>.weights.h5`` -> ``<root>.json``).

    Mirrors ``_cache_paths``: ``zero_shot`` + ``cache_path`` writes ``<root>_c<N>.weights.h5``
    plus ``<root>_c<N>.json``, so the channel suffix is already part of the weights name the
    user later points ``weights_path`` at.
    """
    p = _clean_path(weights_path)
    if not p:
        return ""
    for suffix in (".weights.h5", ".hdf5", ".h5"):
        if p.endswith(suffix):
            return p[:-len(suffix)] + ".json"
    return p + ".json"


def zs_signature_of(sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    """A loaded sidecar reduced to its TRAINING SIGNATURE — everything except
    :data:`INFERENCE_KEY`.

    This is the half ``_compute_zs_deconvnet`` compares for equality when deciding whether
    a cached model may be reused. Routing that comparison through here (rather than
    comparing the whole loaded dict) is what makes the inference block additive: a sidecar
    from before it existed reduces to itself, so it compares exactly as it always did.
    """
    return {k: v for k, v in dict(sidecar).items() if k != INFERENCE_KEY}


def zs_note(params: Mapping[str, Any], modes: Mapping[str, Any]) -> str:
    """One line for the GUI saying which sidecar the ZS-DeconvNet auto values came from — or
    why there are none (V2.23b). Same reason as :func:`stardist_note`.

    The most important case is the one that is NOT a mistake: the authors' published
    checkpoints carry no sidecar, so nothing can be adopted and every setting is the user's to
    match. Saying so is the difference between a feature that looks broken and one that has
    told you what it knows.
    """
    if str((modes or {}).get("mode") or "zero_shot") != "pretrained":
        return ""                      # zero_shot is training a model; nothing to adopt
    wp = _clean_path(params.get("weights_path"))
    if not wp:
        return ""                      # no checkpoint chosen yet — the socket says that
    side = zs_sidecar_path(wp)
    doc = read_json(side)
    if not doc:
        return (f"no training record beside {os.path.basename(wp)} — the authors' published "
                f"checkpoints carry none, so `arch`, the 2D/3D lever, `upsample` and the "
                f"padding margins are yours to match to how it was trained")
    got = zs_trained(params, modes)
    if not got:
        return (f"{os.path.basename(side)} records nothing this node can adopt — check "
                f"`arch`, the lever and `upsample` against how it was trained")
    if not isinstance(doc.get(INFERENCE_KEY), dict):
        return (f"{os.path.basename(side)} is an OLDER record — only `upsample` is stated "
                f"({got.get('upsample')}); the padding margins are yours to match")
    return (f"auto values below come from {os.path.basename(side)}, written when this "
            f"checkpoint was trained ("
            f"{', '.join(f'{k} {v}' for k, v in sorted(got.items()))})")


def zs_trained(params: Mapping[str, Any], modes: Mapping[str, Any]) -> Dict[str, Any]:
    """``{socket_name: trained_value}`` for ``enhance.zs_deconvnet``, from the sidecar
    beside ``weights_path``.

    **Only in ``pretrained`` mode.** ``zero_shot`` is *training a model now*, so the
    parameters on its sockets are inputs to that training, not facts about an existing
    checkpoint — adopting a stale sidecar's values there would silently change what gets
    trained. (The cached-model reuse path has its own, stricter check: it compares the whole
    training signature and retrains on any difference.)

    Falls back to the sidecar's **top-level** ``upsample`` when the inference block is
    absent, which is what every sidecar written before V2.23 looks like. The authors'
    published checkpoints have no sidecar at all and so report nothing — correctly: this
    node knows how a model IT trained was configured, and refuses to guess about one it
    did not (the same reason ``arch_3d`` stays a socket the user sets).
    """
    if str((modes or {}).get("mode") or "zero_shot") != "pretrained":
        return {}
    side = read_json(zs_sidecar_path(params.get("weights_path", "")))
    if not side:
        return {}
    block = side.get(INFERENCE_KEY)
    src: Mapping[str, Any] = block if isinstance(block, dict) else side
    out: Dict[str, Any] = {}
    for name in ZS_INFERENCE_SOCKETS:
        if name not in src:
            continue
        v = src[name]
        if name == "upsample":
            # The same truthiness table `metadata.zs_deconvnet` uses, because the two must
            # agree about what a sidecar's `false` means or the predicted axes and the
            # produced axes disagree (wire-node-v2 §8).
            out[name] = v not in (False, 0, "0", "false", "False")
        else:
            f = _as_float(v)
            if f is not None and f >= 0:
                out[name] = int(round(f))
    return out


__all__ = [
    "read_json", "clear_cache",
    "stardist_model_dir", "stardist_config", "stardist_trained", "stardist_note",
    "INFERENCE_KEY", "ZS_INFERENCE_SOCKETS",
    "zs_sidecar_path", "zs_signature_of", "zs_trained", "zs_note",
]
