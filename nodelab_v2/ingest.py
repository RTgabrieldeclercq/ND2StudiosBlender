"""ND2 → nodegraph v2 ingest (C4) — the app-layer, nd2-coupled reader.

``nodegraph`` stays nd2-free; this module is the seam that turns a real microscope
``.nd2`` into the two things the v2 engine consumes:

* a lazy voxel source — a :class:`~nodegraph.provider.B2ndProvider` (in-memory, or an
  on-disk planar-block ``.b2nd`` store written once and re-opened lazily), and
* a source :class:`~nodegraph.metadata.MetaEnvelope` (canonical axes + the calibration
  dict that drives every metadata-intelligent param).

The ND2 pixels are read via ``nd2.ND2File.to_dask()`` (the frame-wise ``read_frame``
path **segfaults** on the sample, per V2.01) and transposed into the canonical
``(M,T,Z,C,Y,X)`` order, inserting size-1 axes for absent dimensions. Calibration reuses
the proven ``read_nd2_metadata_extended`` (optics parsing is fiddly — don't re-derive it;
vendored from the retired v1 backend into :mod:`nodelab_v2.nd2_meta`), filtered to the v2
``CALIBRATION_KEYS`` vocabulary. Qt-free.

Every ``nd2`` import here goes through :func:`nodelab_v2.nd2_compat.import_nd2`, which
applies the SDK bug shims (a zero-range ZStackLoop divides by zero in ``nd2 <= 0.11.3`` and
takes the whole file with it) before any ``ND2File`` is opened. It stays a *lazy*, in-function
import: this module must remain importable — and its TIFF half usable — with no SDK present.

``.nd3`` (the MEBP HDF5 container) dispatches from the same four seams to
:mod:`nodelab_v2.nd3_ingest`, which is lazy about h5py for the same reason.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes, CALIBRATION_KEYS
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import B2ndProvider

#: A progress sink: ``(fraction 0→1, phase note)``. Ingest reports through this so the
#: GUI can show a determinate bar; ``None`` everywhere keeps the readers silent.
ProgressFn = Callable[[float, str], None]

#: Share of an ingest's wall time the READ phase gets on the combined bar. A rough constant
#: is fine — the phase NOTE tells the user where they are; the split only stops the bar from
#: stalling or jumping.
#:
#: Applies to the paths that still have two phases: a TIFF (``tifffile`` has no lazy read)
#: and an in-memory ingest. An ND2 written to a store is now ONE streamed phase (V2.20 — see
#: :func:`ingest_image`) and spends the whole bar on the write.
#:
#: Re-measured 2026-07-31 on a 1.64 GiB slice of the 640 series (24 cores) after the V2.20
#: parallel pyramid reduce: decode 2.0 s (821 MB/s), whole 3-level write 6.3 s (266 MB/s), so
#: the read is ~32% of a two-phase ingest rather than the ~14% this constant was set from.
#: Left at 0.15 deliberately — it only shapes a progress bar, and both remaining users of it
#: (TIFF, in-memory) have a *different* read profile from the ND2 the old figure came from.
_READ_SHARE = 0.15

#: How many slabs to aim for when realizing a lazy volume, so the bar moves smoothly
#: without paying dask-graph overhead per plane.
_SLAB_TARGET = 64

#: Pyramid depth every image store is ingested to, and the depth
#: :func:`ensure_store_levels` repairs an existing store up to. Levels above 0 are a
#: DISPLAY convenience — every compute reads level 0 — so this is chosen for the Viewer:
#: three levels take a 2048² plane down to 512², below the Viewer's ``max_dim``. A series
#: whose planes cannot be halved that far simply gets a shorter pyramid.
PYRAMID_LEVELS = 3

#: ND2 axis letters → the canonical nodegraph axes (``P`` = position/multipoint → M).
#: ``S`` is the SDK's RGB/sample-plane axis (``nd2.AXIS.RGB``); it is read as channels,
#: exactly as :data:`_TIFF_TO_ND` already does for a TIFF's sample planes — so a colour-
#: camera ND2 loads as a 3-channel image instead of failing. ``nd2`` also emits ``U`` for
#: an axis it could not classify (a CustomLoop); that one is deliberately absent, because
#: guessing which canonical axis an *unknown* loop is would silently mis-shape the volume.
_ND_TO_CANON = {"P": "m", "T": "t", "Z": "z", "C": "c", "S": "c", "Y": "y", "X": "x"}
_CANON: Tuple[str, ...] = ("m", "t", "z", "c", "y", "x")


def _nd2_dims(dims: Sequence[str]) -> list:
    """ND2 axis letters → canonical axes, refusing an axis we cannot place.

    The mirror of :func:`_tiff_dims`, and it exists for the same reason: ``_to_6d`` used to
    index ``_ND_TO_CANON`` directly, so any unmapped letter escaped as a bare
    ``KeyError: 'S'`` from inside a list comprehension — naming neither the file nor the
    layout, and thrown too late to be caught by the File-menu's friendly error path. Both
    reachable cases are real (``nd2/_util.py`` defines ``RGB`` and ``UNKNOWN``): a colour
    camera and a CustomLoop acquisition.

    A duplicate mapping is refused too — with ``S`` now folding onto ``c``, a file carrying
    both ``C`` and ``S`` would otherwise transpose two source axes onto one canonical slot
    and lose data."""
    out: list = []
    for d in dims:
        canon = _ND_TO_CANON.get(str(d).upper())
        if canon is None:
            raise ValueError(
                f"unsupported ND2 axis {d!r} in layout {list(dims)!r} — this reader maps "
                f"P/T/Z/C/S/Y/X onto (M,T,Z,C,Y,X) and cannot place {d!r}. An 'U' axis is "
                f"an unclassified custom acquisition loop; there is no safe guess for which "
                f"canonical axis it is, so it is refused rather than mis-shaped.")
        if canon in out:
            raise ValueError(
                f"ND2 layout {list(dims)!r} maps two axes onto {canon!r} (S is read as a "
                f"channel axis, so a file with both C and S cannot be flattened without "
                f"losing one of them).")
        out.append(canon)
    if "y" not in out or "x" not in out:
        raise ValueError(f"ND2 layout {list(dims)!r} has no Y/X image plane")
    return out


def _to_6d(arr: Any, dims: list) -> Any:
    """Reshape an array whose axes are the ND2 ``dims`` (a subset of P,T,Z,C,S,Y,X, in
    file order) into canonical ``(M,T,Z,C,Y,X)`` — size-1 axes for absent dimensions.
    Works for both numpy and dask (``np.transpose`` dispatches)."""
    present = _nd2_dims(dims)
    missing = [ax for ax in _CANON if ax not in present]
    for _ in missing:
        arr = arr[..., None]                       # append the absent axes as size-1
    order = present + missing
    return np.transpose(arr, [order.index(ax) for ax in _CANON])


def _dt_from_timestamps(md: Dict[str, Any]) -> Any:
    """The frame interval in **seconds** from an ND2's per-frame timestamps, or ``None``.

    ``read_nd2_metadata_extended`` reads ``relativeTimeMs`` per timepoint into
    ``frame_timestamps_s``; ``dt_s`` is simply the **median** of its consecutive
    differences. Median, not mean: a timelapse's first interval often absorbs the
    acquisition start-up and a dropped frame shows up as one double-length gap, and
    neither should move the number every velocity in the graph is divided by. On the lab's
    WellA3 file the 16 stamps give 1417.7–1432.4 s → 1424.6 s.

    Why this has to happen HERE rather than in ``nd2_meta``: that module is vendored
    verbatim from the retired v1 backend and its docstring says not to re-derive it, so the
    ND2 read stays untouched and the *interpretation* — timestamps → the one calibration
    scalar the engine's ``unit="s"`` vocabulary needs — lives in the v2 seam that owns
    :data:`CALIBRATION_KEYS`.

    ``None`` (key omitted) whenever the answer would be a guess: fewer than two stamps,
    or a non-positive median (a single-timepoint file, or stamps the SDK did not fill in).
    Absent is the honest signal — see :func:`nodegraph.nodes._compute_object_metrics`,
    which refuses a velocity rather than dividing by a fabricated 1.0.
    """
    ts = md.get("frame_timestamps_s") or []
    if len(ts) < 2:
        return None
    diffs = sorted(float(b) - float(a) for a, b in zip(ts, ts[1:]))
    mid = len(diffs) // 2
    dt = diffs[mid] if len(diffs) % 2 else 0.5 * (diffs[mid - 1] + diffs[mid])
    return float(dt) if dt > 0 else None


def read_calibration(path: str) -> Dict[str, Any]:
    """The v2 calibration dict for ``path`` (ND2 **or** TIFF) — filtered to
    :data:`~nodegraph.dataset.CALIBRATION_KEYS` (dropping absent / ``None`` scalars;
    ``channel_emission_nm`` stays a per-channel list even when some channels have no
    emission — e.g. a transmitted-light channel is ``None``). ND2 reuses the proven v1
    ``read_nd2_metadata_extended``; TIFF is best-effort (:func:`_tiff_calibration`);
    ND3 maps the container's own ``meta_json`` (:func:`nodelab_v2.nd3_ingest.nd3_calibration`)."""
    if _is_nd3(path):
        from nodelab_v2.nd3_ingest import nd3_calibration
        return nd3_calibration(path)
    if _is_tiff(path):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            return _tiff_calibration(tf)
    from nodelab_v2.nd2_meta import read_nd2_metadata_extended
    md = read_nd2_metadata_extended(path)
    calib = {k: md[k] for k in CALIBRATION_KEYS
             if md.get(k) is not None}
    # the significant sensor depth is not part of `read_nd2_metadata_extended`'s dict, but
    # it IS calibration now (nodes read it — a 12-bit ND2 must not be treated as 16-bit)
    bits = _nd2_bit_depth(path)
    if bits:
        calib["bit_depth"] = int(bits)
    # Nor is `dt_s`: the reader produces per-frame TIMESTAMPS, and every consumer wants the
    # interval. Without this the key was absent on EVERY ND2, so `analysis.object_metrics`
    # divided by its 1.0 fallback and reported µm/frame under a µm/s label — a factor of
    # 1424.6 on the lab's 6-hour WellA3 timelapse.
    dt = _dt_from_timestamps(md)
    if dt is not None:
        calib["dt_s"] = dt
    # `z_step_um` is fabricated by the SDK for a file with NO Z axis (`voxel_size().z`
    # defaults to 1.0), and a real-looking 1.0 satisfies every `or <fallback>` guard in a
    # `derive` and feeds `spacing` in analysis.measure's 3D walk. A single plane has no
    # spacing, so drop it and let the absence say so.
    if int(md.get("n_zslices", 1) or 1) <= 1:
        calib.pop("z_step_um", None)
    origin = _origin_um_from_stage(md)
    if origin is not None:
        calib["origin_um"] = origin
    return calib


def _origin_um_from_stage(md: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Seed ``origin_um`` from the file's stage logs — the minimum corner of each
    multipoint's field, in µm of the microscope frame, as ``[[z, y, x], …]``.

    This is the one place the stage log is turned into something transform-safe. Nikon
    records the field **CENTRE**, so the corner is the centre less half the extent; the Z
    component needs the ZStackLoop anchoring, because ``stagePositionUm.z`` is the
    position's nominal focus and is constant down a stack.

    ``None`` unless the logs cover **every** multipoint. A partial origin is worse than
    none: it is addressed by index, so a short list hands field *m* some other position's
    corner, and the result looks exactly like a complete one — the failure
    :func:`nodegraph.nodes._stitch_stage_xy` already refuses for the same reason.

    Z is optional: a file with XY but no focus log gets ``z = 0.0`` and is still placeable
    laterally, which is the common widefield-montage case. Absent Z would otherwise cost
    the whole origin, and lateral placement is what tile selection actually needs.
    """
    xy = md.get("stage_xy_um") or []
    n_m = int(md.get("n_multipoints", 1) or 1)
    ps = md.get("pixel_size_um")
    try:
        ps = float(ps)
    except (TypeError, ValueError):
        return None
    if len(xy) < n_m or not (ps > 0):
        return None
    half_y = 0.5 * float(md.get("height", 0) or 0) * ps
    half_x = 0.5 * float(md.get("width", 0) or 0) * ps
    zs = md.get("stage_z_um") or []
    home = md.get("z_home_index")
    step = md.get("z_step_um")
    n_z = int(md.get("n_zslices", 1) or 1)
    out: List[List[float]] = []
    for m in range(n_m):
        try:
            cx, cy = float(xy[m][0]), float(xy[m][1])
        except (TypeError, ValueError, IndexError):
            return None
        z0 = 0.0
        if m < len(zs):
            try:
                z_nom = float(zs[m])
            except (TypeError, ValueError):
                z_nom = None
            if z_nom is not None:
                if n_z <= 1:
                    z0 = z_nom              # one plane IS the nominal focus
                elif home is not None and step:
                    span = (n_z - 1) * float(step)
                    top = z_nom - int(home) * float(step)
                    # bottom_to_top False means index runs DOWN in µm, so the minimum
                    # corner is at the far end of the stack rather than at slice `home`.
                    z0 = top if md.get("z_bottom_to_top", True) else top - span
        out.append([z0, cy - half_y, cx - half_x])
    return out


#: extra per-channel display keys (names/optics/native color) the Viewer wants but that
#: are NOT part of the engine calibration schema (:data:`CALIBRATION_KEYS`).
CHANNEL_DISPLAY_KEYS = ("channel_names", "channel_emission_nm",
                        "channel_excitation_nm", "channel_colors")

#: per-POSITION stage geometry: ``stage_xy_um[m]`` is the (x, y) stage coordinate of
#: multipoint ``m``'s field CENTRE, in microns of the microscope's own frame, and
#: ``stage_z_um[m]`` that position's nominal focus. Carried alongside the channel display
#: keys for the same reason — the Viewer needs it (the hover readout turns a pixel into an
#: absolute stage coordinate) but it is per-M geometry, not one of the locked
#: :data:`CALIBRATION_KEYS` scalars, and no ``meta_transform`` knows how to keep it true
#: across a crop or a resample. Treat it as *display* provenance: the readout labels it as
#: read from the file.
STAGE_KEYS = ("stage_xy_um", "stage_z_um")

#: the rest of the **placement vocabulary** (2026-07-31) — what it takes to say where a
#: voxel of THIS file is on the microscope, and when it was acquired, in terms another
#: file can be compared against:
#:
#: * ``z_home_index`` / ``z_bottom_to_top`` — which slice ``stage_z_um`` names, and the
#:   stack's direction. Without them a stack has a focus but no way to place slice *k*.
#: * ``frame_time_jd`` — per-T absolute Julian day, the only cross-file clock
#:   (``frame_timestamps_s`` is relative to each file's own origin; on the WellA3 pair
#:   those origins differ by 1207.3 s).
#: * ``stage_layout_source`` — how the position log was obtained, or ``"missing"``.
#: * ``acquisition_start`` — the human date string, for readouts only.
#:
#: Same status as :data:`STAGE_KEYS`: non-calibration provenance riding the payload
#: (`wire-node-v2` §7b), read by consumers straight off ``ds.metadata``.
PLACEMENT_KEYS = ("z_home_index", "z_bottom_to_top", "frame_time_jd",
                  "stage_layout_source", "acquisition_start")

#: ND3 plate/dataset provenance (2026-08-08) — where an ``.nd3`` image sits on
#: the *plate* and which acquisition profile wrote it: ``plate_frame`` (the
#: A1-relative frame dict, verbatim from ``meta_json``), ``wells_um`` (mapped
#: well centres in absolute stage µm), ``plate_id`` / ``well`` (identity), and
#: ``nd3_profile`` (e.g. ``"mebp.fluor_well/1"``). Same status as
#: :data:`STAGE_KEYS` / :data:`PLACEMENT_KEYS`: non-calibration provenance
#: riding the payload for readouts and future tools — not part of the locked
#: :data:`~nodegraph.dataset.CALIBRATION_KEYS`, and no ``meta_transform``
#: keeps it true across a crop or resample. ``channel_display_levels`` (the
#: file's frozen per-channel ``[lo, hi]`` display window, or ``None`` per
#: channel) rides alongside for a future LUT-seeding feature.
ND3_PLATE_KEYS = ("plate_frame", "wells_um", "plate_id", "well",
                  "nd3_profile")


def _nd2_bit_depth(path: str) -> Any:
    """The ND2's *significant* bit depth (e.g. 12) — the real sensor range, which the
    pixel values alone can't reveal (a dim 12-bit frame may max out below 1024). Read
    straight from ``nd2``'s attributes; ``None`` if unavailable."""
    try:
        from nodelab_v2.nd2_compat import import_nd2
        with import_nd2().ND2File(path) as f:
            a = f.attributes
            for name in ("bitsPerComponentSignificant", "bitsPerComponentInMemory"):
                v = getattr(a, name, None)
                if v:
                    return int(v)
    except Exception:                        # noqa: BLE001 — bit depth is best-effort
        return None
    return None


def _tiff_bit_depth(path: str) -> Any:
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            bps = tf.pages[0].bitspersample
            return int(bps[0] if isinstance(bps, (tuple, list)) else bps)
    except Exception:                        # noqa: BLE001
        return None


def read_channel_display(path: str) -> Dict[str, Any]:
    """The *display* metadata for ``path`` — per-channel names, emission/excitation
    wavelengths, the native color, the significant **bit depth**, the per-position
    stage geometry (:data:`STAGE_KEYS`) and the rest of the placement vocabulary
    (:data:`PLACEMENT_KEYS`) — used to label and tint the channel toggles, to
    size the LUT range (so 12-bit data windows to 4095, not just to its brightest pixel),
    to answer "where on the stage is this pixel?" in the Viewer's hover readout, and to let
    ``view.overlay`` place one file's voxels inside another's field. A
    superset of the engine calibration; kept separate so the engine envelope stays the
    locked calibration schema. TIFF carries no optics or stage log, so it degrades to
    ``Ch0…`` names (emission absent → a neutral grey tint downstream) and cannot be
    placed at all — which the overlay node reports rather than guesses."""
    if _is_nd3(path):
        from nodelab_v2.nd3_ingest import nd3_channel_display
        return nd3_channel_display(path)
    if _is_tiff(path):
        ax = _tiff_axes(path)
        out: Dict[str, Any] = {"channel_names": [f"Ch{i}" for i in range(ax.c)]}
        bits = _tiff_bit_depth(path)
        if bits:
            out["bit_depth"] = bits
        return out
    from nodelab_v2.nd2_meta import read_nd2_metadata_extended
    md = read_nd2_metadata_extended(path)
    out = {k: md[k] for k in CHANNEL_DISPLAY_KEYS if md.get(k) is not None}
    # An EMPTY stage list is dropped, not carried: `read_nd2_metadata_extended` returns
    # `[]` when the SDK had no position log (this file's `stage_z_um`), and an absent key
    # is what the readout tests for before it offers an absolute coordinate.
    for k in STAGE_KEYS:
        if md.get(k):
            out[k] = md[k]
    # The placement scalars take `is not None`, NOT truthiness: `z_home_index` is legitimately
    # **0** on the WellA3 640 stack (nominal focus names slice 0) and `z_bottom_to_top` is a
    # bool — a truthiness filter would drop exactly the two values that anchor a stack, and
    # the loss would look identical to a file that never carried them.
    for k in PLACEMENT_KEYS:
        v = md.get(k)
        if v is None or v == [] or v == "":
            continue
        out[k] = v
    bits = _nd2_bit_depth(path)
    if bits:
        out["bit_depth"] = bits
    return out


def _materialize_6d(arr: Any, progress: Optional[ProgressFn] = None,
                    note: str = "reading") -> np.ndarray:
    """Realize a 6-D ``(M,T,Z,C,Y,X)`` array (dask or numpy) into ONE contiguous numpy
    buffer, filled slab-at-a-time over the leading axes.

    Two reasons not to write ``np.ascontiguousarray(np.asarray(arr))``: it is a single
    opaque call, so there is nothing to report a bar from; and because ``_to_6d``
    transposes, the ``ascontiguousarray`` copies — holding the volume TWICE. On a >50 GB
    series that second copy is what drives the machine into its pagefile. Filling slabs
    into a pre-allocated buffer keeps peak memory at the result plus one slab."""
    shape = tuple(int(n) for n in arr.shape)
    out = np.empty(shape, dtype=arr.dtype)
    # Group over as few leading axes as still gives a smooth bar: per-plane would mean
    # thousands of separate dask computes, one slab means no progress at all.
    k = 1
    while k < 4 and int(np.prod(shape[:k])) < _SLAB_TARGET:
        k += 1
    lead = shape[:k]
    n = max(1, int(np.prod(lead)))
    for i, idx in enumerate(np.ndindex(*lead), start=1):
        out[idx] = np.asarray(arr[idx])
        if progress is not None:
            progress(i / n, note)
    return out


def lazy_nd2(path: str) -> Any:
    """A **lazy** canonical ``(M,T,Z,C,Y,X)`` view of ``path`` — ``nd2``'s ``to_dask()``
    transposed by :func:`_to_6d`, with no pixels read.

    ``nd2`` hands back a ``ResourceBackedDaskArray``, which re-opens the file for each
    block it is asked for, so the view stays valid after the ``ND2File`` context closes and
    the caller need not hold the handle. Slicing it and calling ``np.asarray`` on the slice
    reads exactly that slab — which is what makes :func:`ingest_image`'s write streaming."""
    from nodelab_v2.nd2_compat import import_nd2
    with import_nd2().ND2File(path) as f:
        return _to_6d(f.to_dask(), list(f.sizes.keys()))


def read_nd2(path: str, progress: Optional[ProgressFn] = None
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an ``.nd2`` into a canonical ``(M,T,Z,C,Y,X)`` numpy array + its calibration
    dict. Realizes the whole volume (ingest is a one-time cost); slice ``to_dask`` before
    calling if only a crop is needed. ``progress`` reports ``0→1`` over the read.

    The disk-store path in :func:`ingest_image` no longer goes through this — it streams
    :func:`lazy_nd2` straight into the store — but an in-memory ingest and any caller that
    genuinely wants the array still do."""
    calib = read_calibration(path)
    vol = _materialize_6d(lazy_nd2(path), progress, "reading ND2")
    return vol, calib


#: file extensions this ingest layer reads (case-insensitive).
_ND2_EXT = (".nd2",)
_TIFF_EXT = (".tif", ".tiff")
_ND3_EXT = (".nd3",)

#: tifffile axis letters → the ND2 letter space understood by :func:`_to_6d`
#: (``S`` sample-planes read as channels; ``I``/``Q`` sequence axes read as time).
_TIFF_TO_ND = {"T": "T", "Z": "Z", "C": "C", "Y": "Y", "X": "X",
               "S": "C", "I": "T", "Q": "T"}


def _tiff_dims(axes: str) -> list:
    """Translate a tifffile ``series.axes`` string to the ``_to_6d`` letter list,
    rejecting an axis we can't place (so a surprise layout fails loudly, not silently
    mis-shaped)."""
    dims = []
    for a in axes:
        nd = _TIFF_TO_ND.get(a.upper())
        if nd is None:
            raise ValueError(f"unsupported TIFF axis {a!r} in layout {axes!r}")
        if nd in dims:
            raise ValueError(f"TIFF layout {axes!r} maps two axes onto {nd!r}")
        dims.append(nd)
    if "Y" not in dims or "X" not in dims:
        raise ValueError(f"TIFF layout {axes!r} has no Y/X image plane")
    return dims


def _tiff_calibration(tf: Any) -> Dict[str, Any]:
    """Best-effort calibration for a TIFF: ImageJ ``spacing`` → z step, the XY
    resolution tag → pixel size (µm). Absent tags are simply omitted — a plain TIFF
    carries no optics, and the metadata-intelligent params fall back to their guards."""
    calib: Dict[str, Any] = {}
    ij = getattr(tf, "imagej_metadata", None) or {}
    unit = str(ij.get("unit", "")).lower()
    if ij.get("spacing") and unit in ("um", "micron", "microns", "µm", ""):
        try:
            calib["z_step_um"] = float(ij["spacing"])
        except (TypeError, ValueError):
            pass
    try:
        page = tf.pages[0]
        xres = page.tags.get("XResolution")
        if xres is not None and xres.value and xres.value[0]:
            num, den = xres.value
            per_unit = (num / den) if den else 0.0
            if per_unit:
                px = 1.0 / per_unit                 # distance per pixel, in the tag unit
                ru = page.tags.get("ResolutionUnit")
                if ru is not None and int(ru.value) == 3:   # 3 = centimeter
                    px *= 1.0e4                              # cm → µm
                elif ru is not None and int(ru.value) == 2:  # 2 = inch
                    px *= 25400.0                            # in → µm
                calib["pixel_size_um"] = float(px)
    except Exception:  # noqa: BLE001 — a missing/odd tag must never break ingest
        pass
    try:                                        # bits-per-sample = the CONTAINER depth
        bps = tf.pages[0].bitspersample
        calib["bit_depth"] = int(bps[0] if isinstance(bps, (tuple, list)) else bps)
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in calib.items() if k in CALIBRATION_KEYS}


def read_tiff(path: str, progress: Optional[ProgressFn] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a ``.tif``/``.tiff`` into a canonical ``(M,T,Z,C,Y,X)`` numpy array + a
    best-effort calibration dict (see :func:`_tiff_calibration`). Uses the first series.

    ``progress`` covers only the transpose-into-canonical pass: ``tifffile``'s own
    ``series.asarray()`` is one opaque call with no hook, so a big TIFF sits at 0% for
    that part. ND2 (the lazy ``to_dask`` path) reports throughout."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        vol = _materialize_6d(_to_6d(arr, _tiff_dims(series.axes)), progress,
                              "reading TIFF")
        calib = _tiff_calibration(tf)
    return vol, calib


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TIFF_EXT


def _is_nd3(path: str) -> bool:
    """True for an ``.nd3`` path, **fragment-aware**: ``well.nd3#DAPI`` (the
    one-image escape hatch, see :func:`nodelab_v2.nd3_ingest.split_fragment`)
    is an nd3 path even though ``splitext`` sees the fragment as part of the
    extension."""
    from nodelab_v2.nd3_ingest import split_fragment
    return os.path.splitext(split_fragment(path)[0])[1].lower() in _ND3_EXT


def _tiff_axes(path: str) -> AxisSizes:
    """Canonical :class:`AxisSizes` for a TIFF from its series shape/axes — **no pixel
    read** (uses ``series.shape``, not ``asarray``)."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        dims = _tiff_dims(series.axes)
        size = {nd: n for nd, n in zip(dims, series.shape)}
    canon = {"m": size.get("P", 1), "t": size.get("T", 1), "z": size.get("Z", 1),
             "c": size.get("C", 1), "y": size.get("Y", 1), "x": size.get("X", 1)}
    return AxisSizes(**{k: int(v) for k, v in canon.items()})


def read_image(path: str, progress: Optional[ProgressFn] = None
               ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an ``.nd2``, ``.nd3`` **or** ``.tif``/``.tiff`` into a canonical
    ``(M,T,Z,C,Y,X)`` numpy array + its calibration dict — the
    format-dispatching reader."""
    if _is_nd3(path):
        from nodelab_v2.nd3_ingest import read_nd3
        return read_nd3(path, progress)
    return (read_tiff(path, progress) if _is_tiff(path)
            else read_nd2(path, progress))


def read_meta_only(path: str) -> Tuple[AxisSizes, Dict[str, Any], Dict[str, Any]]:
    """``(axes, calibration, channel_display)`` for ``path`` **without realizing pixels**
    — the cheap read the File-menu loader uses to seed a node's envelope + per-channel
    output sockets the instant a file is picked (the heavy ingest happens lazily on the
    first pull). Works for ND2, ND3 and TIFF."""
    if _is_nd3(path):
        from nodelab_v2.nd3_ingest import nd3_meta_only
        return nd3_meta_only(path)
    calib = read_calibration(path)
    disp = read_channel_display(path)
    if _is_tiff(path):
        axes = _tiff_axes(path)
    else:
        from nodelab_v2.nd2_compat import import_nd2
        with import_nd2().ND2File(path) as f:
            s = f.sizes
        # Validate through the SAME table the pixel read uses, so a layout this reader
        # cannot handle fails HERE — at file-pick time, inside the File menu's friendly
        # error path — instead of handing the GUI a confidently wrong AxisSizes that only
        # blows up later, from inside the pull, as a bare KeyError. (Before this,
        # `s.get("C", 1)` reported 1 channel for a 3-component RGB ND2 and the volume was
        # sized at a third of its real voxel count.)
        _nd2_dims(list(s.keys()))
        axes = AxisSizes(m=int(s.get("P", 1)), t=int(s.get("T", 1)),
                         z=int(s.get("Z", 1)),
                         c=int(s.get("C", s.get("S", 1))),
                         y=int(s.get("Y", 1)), x=int(s.get("X", 1)))
    return axes, calib, disp


def ingest_image(path: str, store_path: Optional[str] = None,
                 *, levels: int = PYRAMID_LEVELS,
                 progress: Optional[ProgressFn] = None
                 ) -> Tuple[B2ndProvider, MetaEnvelope]:
    """Ingest an ``.nd2`` **or** ``.tif``/``.tiff`` → ``(provider, source_envelope)``
    ready to seed the engine.

    With ``store_path`` the volume is persisted to an on-disk planar-block ``.b2nd``
    store (a directory) and a lazy disk-backed provider is returned (re-open later with
    :meth:`B2ndProvider.open` — no re-ingest); without it, an in-memory provider. The
    returned :class:`MetaEnvelope` is the source seed for ``Engine(meta_seeds=...)`` and
    the edit-time ``propagate_meta`` pass.

    ``progress`` receives ONE monotonic ``0→1`` fraction plus a phase note, so a caller
    drives a single determinate bar over the whole one-time ingest.

    **An ND2 written to a store never materializes** (V2.20): the lazy
    :func:`lazy_nd2` view goes straight to :meth:`B2ndProvider.write`, whose level-0 loop
    already reads one z-slab per write, so peak memory is a slab instead of the series. The
    old path allocated the whole 6-D volume first — 84.7 GB of ``np.empty`` for the lab's
    640 series, held live while blosc2 compressed out of it — and a run that dies anywhere
    in those tens of minutes leaves a torn store behind. That is the failure this repairs at
    the source; :func:`verify_store` is the half that *detects* one already on disk.

    The other three combinations keep the read-then-write split (:data:`_READ_SHARE`):
    ``tifffile``'s ``series.asarray()`` has no lazy form, and an in-memory ingest realizes
    the array by definition."""
    # nd3 materializes (v1): payloads are modest (mosaic canvases are
    # downscaled), and a raw h5py dataset dies with its File — the lazy path
    # would need a reopen-per-slab wrapper. See read_nd3's docstring.
    stream = bool(store_path) and not (_is_tiff(path) or _is_nd3(path))
    read_p = write_p = None
    if progress is not None:
        share = 0.0 if stream else _READ_SHARE
        # The streamed path reads and compresses in one pass, so calling that phase
        # "writing" would read as if the (dominant) decode were free. Name the source.
        note_w = (f"ingesting {os.path.basename(path)}" if stream else
                  f"writing {os.path.basename(store_path)}" if store_path
                  else "building pyramid")

        def read_p(f: float, note: str) -> None:          # noqa: F811 — 0 → _READ_SHARE
            progress(f * _READ_SHARE, note)

        def write_p(f: float) -> None:                    # noqa: F811 — share → 1
            progress(share + f * (1.0 - share), note_w)

    if stream:
        calib = read_calibration(path)
        vol6d: Any = lazy_nd2(path)
    else:
        vol6d, calib = read_image(path, read_p)
    m, t, z, c, y, x = vol6d.shape
    provider = (B2ndProvider.write(vol6d, store_path, levels=levels, progress=write_p)
                if store_path
                else B2ndProvider.from_array(vol6d, levels=levels, progress=write_p))
    envelope = MetaEnvelope(axes=AxisSizes(m=m, t=t, z=z, c=c, y=y, x=x), metadata=calib)
    if progress is not None:
        progress(1.0, "done")
    return provider, envelope


#: backwards-compatible alias (ND2-only callers) — now format-dispatching.
ingest_nd2 = ingest_image


def verify_store(prov: B2ndProvider, store_path: str) -> None:
    """Refuse an ingest store whose ``level_0`` stops partway through (V2.20).

    :meth:`B2ndProvider.open` already refuses a level 0 *marked* torn, but a store written
    before that marker existed is trusted as ``"legacy"`` — and the lab's 84.7 GB
    ``WellA3…Channel640`` store was exactly that: a legacy ``level_0`` holding 7 899 of its
    40 320 chunks, so positions m≥2 loaded as solid black while the file itself was fine.
    Nothing said so. The pyramid repair then read those zeros and dutifully wrote two more
    levels of them, marked complete.

    So for a legacy level 0 the marker is not evidence and the chunk census is
    (:meth:`B2ndProvider.blank_tail`): it asks blosc2 which chunks were ever written, which
    costs no decompression and self-sunsets — every store written from V2.19 on is marked
    and skips it.

    This lives at the **ingest** seam, not in :meth:`B2ndProvider.open`, because it is a
    statement about the *data*: an ND2/TIFF ingest is dense (raw camera frames have a noise
    floor; an all-zero plane does not occur), whereas a Dock checkpoint's raster may be a
    mask that is legitimately blank almost everywhere. :mod:`nodegraph.checkpoint` opens the
    same class and must keep its own manifest-based completeness rule.

    Raises :class:`ValueError`, which is what every caller wants: the runner's resolve
    catches it and falls back to a real re-ingest from the source file (:meth:`
    B2ndProvider.write` rebuilds with ``mode="w"``)."""
    if prov.level_state(0) != "legacy":
        return                    # V2.19-marked: `complete` lands only after the last chunk
    tail = prov.blank_tail(0)
    if tail is None:
        return
    written, total = tail
    raise ValueError(
        f"torn b2nd store: {store_path!r} was only written as far as chunk {written} of "
        f"{total} ({100.0 * written / max(1, total):.1f}%), so the rest of the series "
        f"reads back as solid zeros — an ingest that was killed or cancelled partway. It "
        f"predates the completeness marker, so nothing on disk said so. Re-ingest from the "
        f"source file; deleting {store_path!r} forces that.")


def open_store(store_path: str, *, verify: bool = True) -> B2ndProvider:
    """Re-open a previously-written on-disk ``.b2nd`` store (no re-ingest). Raises when the
    store is unusable — no ``level_0``, a ``level_0`` marked torn, or (unless ``verify`` is
    off) a legacy ``level_0`` the chunk census shows was never finished
    (:func:`verify_store`) — which is the caller's signal to re-ingest from the source
    file."""
    prov = B2ndProvider.open(store_path)
    if verify:
        verify_store(prov, store_path)
    return prov


def ensure_store_levels(store_path: str, levels: int = PYRAMID_LEVELS,
                        progress: Optional[Callable[[float], None]] = None
                        ) -> B2ndProvider:
    """Re-open a store and **complete its pyramid in place** if it is short (V2.19).

    Level *l* is a pure function of level *l-1*, so a store with an intact ``level_0`` can
    regain its display pyramid without the source file being read — or existing. That is
    the repair for a store whose ingest died between levels: the lab's 84.7 GB ND2 left
    exactly that behind (a complete ``level_0``, no pyramid), because the old writer
    downsampled each whole level in RAM. Idempotent and memo-neutral — the source's
    identity comes from level 0, which this never touches.

    ``progress`` receives a monotonic ``0→1`` over the levels being added, and is not
    called at all when there is nothing to do.

    Verifies level 0 first (:func:`verify_store`) — a repair reads level 0 and writes what
    it finds, so running it over a torn one manufactures a pyramid of zeros and stamps it
    complete. That is not hypothetical: it is what happened to the lab's 640 series."""
    verify_store(B2ndProvider.open(store_path), store_path)
    return B2ndProvider.ensure_levels(store_path, levels, progress=progress)


def split_fragment(path: str):
    """Re-export of :func:`nodelab_v2.nd3_ingest.split_fragment` so callers
    (the runner's source resolution) keep importing one ingest module."""
    from nodelab_v2.nd3_ingest import split_fragment as _sf
    return _sf(path)


__all__ = ["read_calibration", "read_channel_display", "lazy_nd2", "read_nd2",
           "read_tiff", "read_image", "read_meta_only", "ingest_image", "ingest_nd2",
           "open_store", "verify_store", "ensure_store_levels", "PYRAMID_LEVELS",
           "CHANNEL_DISPLAY_KEYS", "STAGE_KEYS", "PLACEMENT_KEYS",
           "ND3_PLATE_KEYS", "split_fragment"]
