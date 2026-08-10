"""ND3 → nodegraph v2 ingest — the app-layer, nd3-coupled reader.

The `.nd3` half of the format seam (:mod:`nodelab_v2.ingest` owns the dispatch;
``nodegraph`` stays format-free). It turns an MEBP ``.nd3`` container — HDF5,
spec ``docs/ND3_SPEC.md`` in the MEBP repository, schema 1.0 — into the same
two things the ND2/TIFF readers produce: a canonical ``(M,T,Z,C,Y,X)`` volume
and a :data:`~nodegraph.dataset.CALIBRATION_KEYS` dict, plus the display /
provenance extras (:data:`nodelab_v2.ingest.ND3_PLATE_KEYS`).

The low-level reader is :mod:`nodelab_v2.nd3` — **vendored byte-identical**
from MEBP ``SupportClasses/ND3.py`` (v7.16, schema 1.0, copied 2026-08-08).
Vendoring verbatim is sanctioned and test-enforced on the MEBP side (the module
must stay stdlib+numpy+h5py only); to upgrade, re-copy the whole file — never
patch it locally. A selftest pins ``nd3.SCHEMA_VERSION == "1.0"`` so a silent
MAJOR bump on re-vendor trips the gate.

Every ``nodelab_v2.nd3`` import here is lazy (inside functions), for the same
reason :mod:`nodelab_v2.ingest` imports ``nd2``/``tifffile`` lazily: this
module must stay importable — and the rest of the ingest seam usable — with no
h5py installed.

Two honesty rules dominate everything below (both are the difference between a
correct measurement and a plausible-looking wrong one):

* **Scale** comes from the stored ``pixel_to_stage_um`` matrix diagonal when
  present (authoritative, spec §8.4), else ``scale.um_per_px`` — and NEVER
  from ``captured_um_per_px``: a plate-mosaic canvas is downscaled from camera
  resolution, and reading the camera pitch mis-scales every µm figure ~75×
  (spec §8.5, the named mosaic pitfall).
* **Placement** (``origin_um``) comes from the matrix translation ONLY — the
  matrix is the one field guaranteed registration-shift-subtracted (§8.2) —
  and is *omitted* whenever the file says it cannot be trusted
  (``shift_known: false``, or no transforms at all, which covers
  ``pixels_stage_aligned: false``). Absent means "cannot be placed", which is
  exactly what :data:`~nodegraph.dataset.CALIBRATION_KEYS` documents for it.

Multi-image containers (``mebp.fluor_well/1`` writes ONE IMAGE PER CHANNEL,
with **alphabetical** ids) stack along C only when the stack is honest — same
shape/dtype/pixel-size, origins agreeing to half a pixel — and are otherwise
refused with the ``path.nd3#<image_id>`` remediation spelled out in the
message. A stack along C asserts pixel (y, x) is the same physical point in
every channel; when the per-channel geometry disagrees, the stack itself is a
lie, so this reader refuses rather than resamples.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AxisSizes

ProgressFn = Callable[[float, str], None]

#: pixel formats whose S-axis sample order this reader knows (spec §5.3
#: registry). Anything else carrying an S axis is refused — "readers MUST NOT
#: guess an unknown format's sample order".
_S_FORMATS = ("RGB", "BGR", "RGBA", "BGRA")

#: sample-index reorder applied so every S-folded volume is R,G,B(,A) in
#: channel order regardless of how the file interleaved it.
_S_REORDER = {"RGB": None, "RGBA": None,
              "BGR": [2, 1, 0], "BGRA": [2, 1, 0, 3]}

#: fallback per-sample display colors (R, G, B, A) for S-folded images —
#: 3-lists, the shape the Viewer's native-color check expects.
_S_COLORS = ([255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255])

#: half a pixel — the lateral tolerance for per-channel origins of one stacked
#: well. The stage does not move between channels of a well and the matrix
#: origin is raw (shift-subtracted) stage position, so honest origins agree to
#: rounding; more than half a pixel means different fields, and stacking them
#: would misregister the C axis visibly.
_ORIGIN_TOL_PX = 0.5

_MAG_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[xX×]")
_INT_RE = re.compile(r"\s*(\d+)")


# --------------------------------------------------------------------------
# path fragments — the one-image escape hatch
# --------------------------------------------------------------------------

def split_fragment(path: str) -> Tuple[str, Optional[str]]:
    """Split a ``file.nd3#image_id`` path into ``(file_part, image_id|None)``.

    ``#`` is a legal filename character, so the whole path is probed first: an
    existing file named ``a#b.nd3`` is never mis-split. A fragment is honored
    only when the part before the last ``#`` exists on disk — or, when neither
    exists, ends in ``.nd3`` (so a file-not-found error downstream names the
    file the user actually meant, not the ``#``-suffixed string)."""
    if "#" not in path or os.path.isfile(path):
        return path, None
    file_part, frag = path.rsplit("#", 1)
    if not frag:
        return path, None
    if os.path.isfile(file_part):
        return file_part, frag
    if os.path.splitext(file_part)[1].lower() == ".nd3":
        return file_part, frag
    return path, None


# --------------------------------------------------------------------------
# pure calibration helpers — plain dicts in, values out; unit-tested without
# h5py or a file (nodegraph.selftest.test_nd3_calibration_mapping)
# --------------------------------------------------------------------------

def _pixel_size_um(meta: Dict[str, Any]) -> Optional[float]:
    """This image's pixel pitch in µm/px — matrix diagonal when stored
    (authoritative), else ``scale.um_per_px``. ``captured_um_per_px`` is never
    read (the §8.5 mosaic pitfall). ``None`` when the file does not say."""
    mat = (meta.get("transforms") or {}).get("pixel_to_stage_um")
    if mat is not None:
        try:
            p = float(mat[0][0])
            if p > 0:
                return p
        except (TypeError, ValueError, IndexError):
            pass
    v = (meta.get("scale") or {}).get("um_per_px")
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def _origin_um(meta: Dict[str, Any]) -> Optional[List[float]]:
    """The trusted ``[z, y, x]`` µm origin of pixel (0, 0)'s outer corner, or
    ``None`` whenever emitting one would be a guess.

    Gates, in order (each → ``None``, which downstream reads as "cannot be
    placed" — the load-bearing absence of ``origin_um``):

    * no stored ``pixel_to_stage_um`` — covers ``pixels_stage_aligned: false``
      (transforms are only written when honest) and legacy files. The scalar
      ``stage_frame.extent_um`` is deliberately NOT a fallback: only the
      matrix is guaranteed to have the registration shift subtracted (§8.2).
    * ``stage_frame.shift_known == false`` — the shift is *unknown*, not zero;
      such an image is fine for translation-invariant measurement but its
      absolute position may be off by up to half a field of view.

    Z is ``planes[0].focus_um`` when recorded, else 0.0 — the same precedent
    as the ND2 reader: a missing focus log must not cost lateral placement,
    which is what tile selection and overlay actually need."""
    mat = (meta.get("transforms") or {}).get("pixel_to_stage_um")
    if mat is None:
        return None
    if (meta.get("stage_frame") or {}).get("shift_known") is False:
        return None
    try:
        ox, oy = float(mat[0][2]), float(mat[1][2])
    except (TypeError, ValueError, IndexError):
        return None
    planes = meta.get("planes") or []
    z = 0.0
    if planes:
        try:
            z = float(planes[0].get("focus_um", 0.0))
        except (TypeError, ValueError):
            z = 0.0
    return [z, oy, ox]


def _stacked_origin_um(origins: Sequence[Optional[List[float]]],
                       pixel_size_um: Optional[float]
                       ) -> Optional[List[float]]:
    """One origin for a C-stacked well from the per-channel origins, in
    stacked order.

    * any channel's origin unknown → ``None`` (the stack still loads, for
      translation-invariant work, but is never placed by "the channels that
      know" — a partially-trusted origin looks identical to a full one);
    * all known but laterally more than :data:`_ORIGIN_TOL_PX` pixels apart →
      ``ValueError`` (the caller refuses the whole stack: those are different
      fields, and stacking them would silently misregister channels);
    * otherwise the **first** stacked channel's origin (deterministic, and any
      pick is honest to under half a pixel). Per-channel *focus* differences
      are routine (channels are refocused), so the tolerance is lateral only.
    """
    if any(o is None for o in origins):
        return None
    if not origins:
        return None
    if pixel_size_um is None or not (pixel_size_um > 0):
        return None
    tol = _ORIGIN_TOL_PX * float(pixel_size_um)
    ys = [o[1] for o in origins]
    xs = [o[2] for o in origins]
    dy = max(ys) - min(ys)
    dx = max(xs) - min(xs)
    if dy > tol or dx > tol:
        raise ValueError(
            f"per-channel stage origins differ by ({dx:.3f}, {dy:.3f}) µm — "
            f"more than half a pixel ({tol:.3f} µm) — so these images are not "
            "the same field and stacking them along C would misregister the "
            "channels")
    return list(origins[0])


def _bit_depth(meta: Dict[str, Any]) -> Optional[int]:
    """The SENSOR's significant bit depth from ``acquisition.bit_depth``
    (leading integer of e.g. ``12`` or ``"12-bit"``). NEVER derived from the
    array dtype — the dtype is the *container* depth, and a uint16 container
    of a 12-bit sensor mis-windows every LUT and threshold downstream."""
    v = (meta.get("acquisition") or {}).get("bit_depth")
    if v is None:
        return None
    m = _INT_RE.match(str(v))
    return int(m.group(1)) if m else None


def _objective_na(meta: Dict[str, Any]) -> Optional[float]:
    v = (meta.get("acquisition") or {}).get("numerical_aperture")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _objective_magnification(meta: Dict[str, Any]) -> Optional[float]:
    """Leading ``<number>x`` from ``acquisition.magnification`` /
    ``objective`` / ``objective_label`` (e.g. ``"10x Plan Fluor"`` → 10.0);
    ``None`` when unparseable — recorded, never guessed."""
    acq = meta.get("acquisition") or {}
    for key in ("magnification", "objective", "objective_label"):
        m = _MAG_RE.match(str(acq.get(key, "")))
        if m:
            return float(m.group(1))
    return None


def _per_channel(channels: Sequence[Dict[str, Any]], key: str,
                 n: int) -> Optional[List[Any]]:
    """A per-channel list of ``key`` values (``None`` holes kept — a
    transmitted-light channel legitimately has no emission), or ``None`` when
    no channel carries the key at all, or the entry count does not match the
    channel axis (a mismatched list would address the wrong channel, which is
    worse than absence)."""
    if len(channels) != n:
        return None
    vals = [ch.get(key) for ch in channels]
    return vals if any(v is not None for v in vals) else None


def _dt_s(meta: Dict[str, Any], t_count: int) -> Optional[float]:
    """Median frame interval in seconds from ``planes[].t_iso`` (first stamp
    per distinct T, in T order), or ``None``. Median for the same reason the
    ND2 reader uses it (:func:`nodelab_v2.ingest._dt_from_timestamps`): the
    first interval absorbs start-up and a dropped frame is one double gap.
    Mixed naive/aware or unparseable stamps → ``None`` — never a guess."""
    if t_count < 2:
        return None
    seen: Dict[int, str] = {}
    for p in meta.get("planes") or []:
        t, iso = p.get("t"), p.get("t_iso")
        if t is None or not iso or int(t) in seen:
            continue
        seen[int(t)] = str(iso)
    if len(seen) < 2:
        return None
    try:
        times = [datetime.fromisoformat(seen[k]) for k in sorted(seen)]
        secs = [(x - times[0]).total_seconds() for x in times]
    except (ValueError, TypeError):
        return None
    from nodelab_v2.ingest import _dt_from_timestamps
    return _dt_from_timestamps({"frame_timestamps_s": secs})


def stack_order(id_meta_pairs: Sequence[Tuple[str, Dict[str, Any]]]
                ) -> List[str]:
    """The C-stacking order for a multi-image container: by the microscope's
    ``channels[0].channel_number`` (filter position) iff every image carries
    one and they are all distinct, else by image id — never a mixed sort
    (half-known ordering is a guess).

    This exists because ``ND3Reader.image_ids()`` is **alphabetical**:
    ``Bright_Field`` sorts before ``DAPI`` but is filter channel 4, and
    ingesting in id order would put DAPI pixels under a Bright_Field label
    while looking entirely normal. MEBP's own LabLink sidecar builder re-sorts
    the same way, so the sidecar's channel list and this stack agree by
    construction."""
    nums = []
    for _id, meta in id_meta_pairs:
        chans = meta.get("channels") or []
        n = chans[0].get("channel_number") if chans else None
        if not isinstance(n, int):
            nums = None
            break
        nums.append(n)
    ids = [i for i, _ in id_meta_pairs]
    if nums is not None and len(set(nums)) == len(nums):
        return [i for _, i in sorted(zip(nums, ids))]
    return sorted(ids)


def calibration_from_metas(metas: Sequence[Dict[str, Any]], *,
                           stacked: bool, c_count: int,
                           t_count: int = 1) -> Dict[str, Any]:
    """The :data:`~nodegraph.dataset.CALIBRATION_KEYS` dict from parsed
    ``meta_json`` dicts (in stacked order when ``stacked``). Absent values
    stay absent — every key here has an honest "the file does not say" state
    that downstream guards test for.

    Raises :class:`ValueError` when stacked origins disagree beyond half a
    pixel (see :func:`_stacked_origin_um`) — the caller refuses the stack."""
    meta0 = metas[0]
    calib: Dict[str, Any] = {}

    ps = _pixel_size_um(meta0)
    if ps is not None:
        calib["pixel_size_um"] = ps

    if stacked:
        origin = _stacked_origin_um([_origin_um(m) for m in metas], ps)
        channels = [(m.get("channels") or [{}])[0] for m in metas]
    else:
        origin = _origin_um(meta0)
        channels = meta0.get("channels") or []
    if origin is not None:
        calib["origin_um"] = [origin]

    emission = _per_channel(channels, "emission_nm", c_count)
    if emission is not None:
        calib["channel_emission_nm"] = emission

    for key, fn in (("bit_depth", _bit_depth), ("objective_na", _objective_na),
                    ("objective_magnification", _objective_magnification)):
        for m in metas:
            v = fn(m)
            if v is not None:
                calib[key] = v
                break

    if not stacked:
        dt = _dt_s(meta0, t_count)
        if dt is not None:
            calib["dt_s"] = dt
    # z_step_um deliberately absent (v1): no MEBP profile writes Z stacks, and
    # a fabricated spacing feeds every µm³ figure downstream. Deriving it from
    # per-z planes[].focus_um diffs can come when a producer exists.
    return calib


# --------------------------------------------------------------------------
# the plan — the ONE place stacking / refusal / ordering is decided, shared
# by metadata reads and the pixel read so they can never disagree
# --------------------------------------------------------------------------

class _Nd3Plan:
    """How one ``.nd3`` container (or one ``#image_id`` of it) becomes one
    canonical volume: which images, in what order, stacked or not."""

    def __init__(self, path: str, images: List[Any], stacked: bool,
                 dataset_meta: Dict[str, Any]):
        self.path = path
        self.images = images                  # ND3Image handles, stacked order
        self.stacked = stacked
        self.dataset_meta = dataset_meta
        self.metas = [img.meta for img in images]

    # -- canonical geometry ---------------------------------------------------

    def axis_sizes(self) -> AxisSizes:
        if self.stacked:
            y, x = self.images[0].shape
            return AxisSizes(m=1, t=1, z=1, c=len(self.images), y=y, x=x)
        img = self.images[0]
        size = dict(zip(img.axes, img.shape))
        c = size.get("C", size.get("S", 1))
        return AxisSizes(m=1, t=int(size.get("T", 1)), z=int(size.get("Z", 1)),
                         c=int(c), y=int(size["Y"]), x=int(size["X"]))

    @property
    def s_folded(self) -> bool:
        return (not self.stacked) and "S" in self.images[0].axes

    def calibration(self) -> Dict[str, Any]:
        ax = self.axis_sizes()
        try:
            return calibration_from_metas(
                self.metas, stacked=self.stacked,
                c_count=1 if self.s_folded else ax.c, t_count=ax.t)
        except ValueError as exc:
            raise ValueError(self._refusal(str(exc))) from None

    def _refusal(self, why: str) -> str:
        ids = [img.id for img in self.images]
        return (f"cannot load {self.path!r} as one volume: {why}. Load one "
                f"image instead with '<path>#<image_id>' — ids in this file: "
                f"{ids}")


def _open(path: str):
    from nodelab_v2 import nd3 as _nd3
    return _nd3.open_nd3(path)


def _check_s_format(img: Any) -> None:
    if "S" in img.axes and img.pixel_format not in _S_FORMATS:
        raise ValueError(
            f"images/{img.id} has interleaved samples (axes {img.axes!r}) "
            f"with pixel_format {img.pixel_format!r}, which is not in the "
            f"known registry {_S_FORMATS} — the sample order of an unknown "
            "format must not be guessed (ND3 spec §5.3)")


def _plan(reader: Any, path: str, image_id: Optional[str]) -> _Nd3Plan:
    """Decide how this container loads. All Q1 rules live here."""
    ids = reader.image_ids()
    if not ids:
        raise ValueError(f"{path!r} is a valid .nd3 container but holds no "
                         "images")
    dataset_meta = reader.dataset_meta

    if image_id is not None:
        if image_id not in ids:
            raise ValueError(
                f"no image {image_id!r} in {path!r} — ids in this file: {ids}")
        img = reader.image(image_id)
        _check_s_format(img)
        return _Nd3Plan(path, [img], False, dataset_meta)

    if len(ids) == 1:
        img = reader.image(ids[0])
        _check_s_format(img)
        return _Nd3Plan(path, [img], False, dataset_meta)

    # N images → a C stack, allowed only when honest.
    images = {i: reader.image(i) for i in ids}
    plan = _Nd3Plan(path, list(images.values()), True, dataset_meta)

    def refuse(why: str) -> ValueError:
        return ValueError(plan._refusal(why))

    for img in images.values():
        if img.axes != "YX":
            raise refuse(
                f"images/{img.id} has axes {img.axes!r} — a multi-image "
                "container stacks along C only when every image is a plain "
                "YX plane")
    shapes = {img.shape for img in images.values()}
    if len(shapes) > 1:
        detail = ", ".join(f"{i}: {images[i].shape}" for i in ids)
        raise refuse(f"images differ in shape ({detail})")
    dtypes = {img.dtype for img in images.values()}
    if len(dtypes) > 1:
        detail = ", ".join(f"{i}: {images[i].dtype}" for i in ids)
        raise refuse(f"images differ in dtype ({detail})")
    sizes = {i: _pixel_size_um(images[i].meta) for i in ids}
    if any(v is None for v in sizes.values()):
        missing = [i for i, v in sizes.items() if v is None]
        raise refuse(f"images {missing} carry no pixel size")
    ref = next(iter(sizes.values()))
    if not all(np.isclose(v, ref, rtol=1e-6) for v in sizes.values()):
        detail = ", ".join(f"{i}: {sizes[i]}" for i in ids)
        raise refuse(
            f"images differ in µm/px ({detail}) — stacking them would put "
            "two scales under one calibration and every µm² figure would be "
            "wrong for some channel")

    order = stack_order([(i, images[i].meta) for i in ids])
    plan.images = [images[i] for i in order]
    plan.metas = [img.meta for img in plan.images]
    plan.calibration()          # runs the stacked-origin tolerance gate now
    return plan


# --------------------------------------------------------------------------
# display / provenance
# --------------------------------------------------------------------------

def _display_from_plan(plan: _Nd3Plan) -> Dict[str, Any]:
    """The Viewer-facing dict: channel names/colors/optics, sensor bit depth,
    frozen display levels, plate provenance, and the field-centre
    ``stage_xy_um`` the hover readout uses — all non-calibration keys riding
    the payload the way :data:`nodelab_v2.ingest.PLACEMENT_KEYS` do."""
    ax = plan.axis_sizes()
    out: Dict[str, Any] = {}

    if plan.s_folded:
        img = plan.images[0]
        chans = img.channels or []
        base = str(chans[0].get("name") or "") if len(chans) == 1 else ""
        letters = "RGBA"[:ax.c]
        out["channel_names"] = [f"{base} {L}".strip() if base else L
                                for L in letters]
        out["channel_colors"] = [list(_S_COLORS[i]) for i in range(ax.c)]
    else:
        if plan.stacked:
            entries = [(m.get("channels") or [{}])[0] for m in plan.metas]
            fallback = [img.id for img in plan.images]
        else:
            entries = list(plan.metas[0].get("channels") or [])
            fallback = [f"Ch{i}" for i in range(ax.c)]
            if len(entries) != ax.c:
                entries = [{} for _ in range(ax.c)]
        out["channel_names"] = [
            str(e.get("name") or fallback[i]) for i, e in enumerate(entries)]
        colors = [e.get("color_rgb") for e in entries]
        if any(c is not None for c in colors):
            out["channel_colors"] = [
                list(c) if isinstance(c, (list, tuple)) and len(c) == 3
                else None for c in colors]
        for src, dst in (("emission_nm", "channel_emission_nm"),
                         ("excitation_nm", "channel_excitation_nm")):
            vals = _per_channel(entries, src, ax.c)
            if vals is not None:
                out[dst] = vals
        levels = [[e.get("display_lo"), e.get("display_hi")]
                  if e.get("display_lo") is not None
                  and e.get("display_hi") is not None else None
                  for e in entries]
        if any(v is not None for v in levels):
            out["channel_display_levels"] = levels

    for m in plan.metas:
        bits = _bit_depth(m)
        if bits is not None:
            out["bit_depth"] = bits
            break

    # plate / dataset provenance (ND3_PLATE_KEYS) — lifted verbatim
    dm = plan.dataset_meta or {}
    frame = next((m.get("plate_frame") for m in plan.metas
                  if m.get("plate_frame")), None)
    if frame:
        out["plate_frame"] = frame
    for key in ("wells_um", "plate_id", "well"):
        v = dm.get(key) or (frame or {}).get(key)
        if v:
            out[key] = v
    if dm.get("profile"):
        out["nd3_profile"] = dm["profile"]

    # field centre for the hover readout — only when honestly placeable
    calib = plan.calibration()
    origin = calib.get("origin_um")
    ps = calib.get("pixel_size_um")
    if origin and ps:
        z0, oy, ox = origin[0]
        out["stage_xy_um"] = [[ox + 0.5 * ax.x * ps, oy + 0.5 * ax.y * ps]]
        out["stage_z_um"] = [z0]
    return out


# --------------------------------------------------------------------------
# public surface — mirrors the ND2/TIFF reader shapes in nodelab_v2.ingest
# --------------------------------------------------------------------------

def nd3_calibration(path: str) -> Dict[str, Any]:
    """The v2 calibration dict for an ``.nd3`` path (``#image_id`` honored)."""
    file_part, frag = split_fragment(path)
    with _open(file_part) as reader:
        return _plan(reader, file_part, frag).calibration()


def nd3_channel_display(path: str) -> Dict[str, Any]:
    """Display/provenance metadata for an ``.nd3`` path — the nd3 counterpart
    of :func:`nodelab_v2.ingest.read_channel_display`."""
    file_part, frag = split_fragment(path)
    with _open(file_part) as reader:
        return _display_from_plan(_plan(reader, file_part, frag))


def nd3_meta_only(path: str) -> Tuple[AxisSizes, Dict[str, Any],
                                      Dict[str, Any]]:
    """``(axes, calibration, channel_display)`` with **no pixel reads** — all
    three off one :class:`_Nd3Plan`, so they can never disagree about channel
    order or count."""
    file_part, frag = split_fragment(path)
    with _open(file_part) as reader:
        plan = _plan(reader, file_part, frag)
        return plan.axis_sizes(), plan.calibration(), _display_from_plan(plan)


def _to_canonical(arr: np.ndarray, axes: str, pixel_format: str,
                  image_id: str) -> np.ndarray:
    """One image's raw array → canonical ``(M,T,Z,C,Y,X)``: reorder unknown-
    free S samples to R,G,B(,A), then reuse the shared axis transposer (ND3's
    T/C/Z/Y/X/S letters are a subset of the letter space it already maps)."""
    reorder = _S_REORDER.get(pixel_format)
    if "S" in axes and reorder is not None:
        arr = arr[..., reorder]
    from nodelab_v2.ingest import _to_6d
    try:
        return _to_6d(arr, list(axes))
    except ValueError as exc:
        raise ValueError(f"images/{image_id}: {exc}") from None


def read_nd3(path: str, progress: Optional[ProgressFn] = None
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an ``.nd3`` into a canonical ``(1,T,Z,C,Y,X)`` numpy array + its
    calibration dict. Materializes (v1): real MEBP payloads are modest — plate
    mosaics are *downscaled* canvases, wells are a handful of planes. If huge
    containers appear, the upgrade is a small lazy wrapper whose
    ``__getitem__`` opens ``h5py.File`` per slab (a raw h5py dataset dies with
    its ``File``, unlike ND2's resource-backed dask view), slotted into
    :func:`nodelab_v2.ingest.ingest_image`'s streaming branch."""
    file_part, frag = split_fragment(path)
    with _open(file_part) as reader:
        plan = _plan(reader, file_part, frag)
        calib = plan.calibration()
        n = len(plan.images)
        if plan.stacked:
            planes = []
            for i, img in enumerate(plan.images, start=1):
                planes.append(img.array())
                if progress is not None:
                    progress(i / n, "reading ND3")
            vol = _to_canonical(np.stack(planes, axis=0), "CYX", "", "stack")
        else:
            img = plan.images[0]
            vol = _to_canonical(img.array(), img.axes, img.pixel_format,
                                img.id)
            if progress is not None:
                progress(1.0, "reading ND3")
    return vol, calib


__all__ = ["split_fragment", "nd3_calibration", "nd3_channel_display",
           "nd3_meta_only", "read_nd3", "stack_order",
           "calibration_from_metas"]
