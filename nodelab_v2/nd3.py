"""ND3.py — the .nd3 multidimensional image container (HDF5-based).

v7.16, from the operator's ask: "lets make the best file system that stores all
that we need to reconstruct images elsewhere. we can call the file system .nd3
we will use this in other software like nd2 studios blender, and lab link."

An .nd3 file is a valid HDF5 file with a defined tree (normative spec:
docs/ND3_SPEC.md — third parties implement readers from THAT document)::

    /                        attrs: format="nd3", schema_version="1.0",
                                    created_iso, generator_json
    /dataset_json            bytes dataset — free-form dataset metadata
    /images/<id>/data        pixel dataset (chunked, gzip, Fletcher32)
                             attrs: axes ("YX"/"CYX"/"TYXS"/...), pixel_format
    /images/<id>/meta_json   bytes dataset — per-image metadata document
    /images/<id>/preview_png optional pre-encoded PNG bytes
    /attachments/<name>      bytes datasets, attr media_type

⚠ This module MUST stay importable with ONLY the standard library + numpy +
h5py — no other SupportClasses, no cv2, no Qt. It is vendored verbatim into
Blender / LabLink readers, and a test walks its AST to keep it that way.
Anything that needs the MEBP stores or OpenCV belongs in ND3Export.py.

Two conventions carried from the rest of this codebase:

* **Refuse, never sanitize.** Image ids / attachment names outside
  ``[A-Za-z0-9_.-]+`` raise ``ND3ValidationError`` — a many-to-one sanitizer
  lets two names silently collide (the v7.12 plate-store lesson).
* **Absent means unknown.** Nothing here fabricates a frame, a shift, or a
  default; an unknown value is an absent key, and honesty flags such as
  ``shift_known`` / ``needs_plate_frame`` are written by the EXPORTERS, not
  invented here.

Bytes payloads (JSON documents, PNG previews, attachments) are stored as 1-D
uint8 datasets — fixed-length HDF5 strings are unsafe for binary because a
payload whose last byte is 0x00 (possible in a PNG CRC) is indistinguishable
from NUL padding; readers accept scalar-bytes datasets too (spec §3).
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - import guard exercised only where h5py is absent
    import h5py
except Exception as _exc:  # noqa: N816
    h5py = None
    _H5PY_IMPORT_ERROR = _exc
else:
    _H5PY_IMPORT_ERROR = None

FORMAT_NAME = "nd3"
SCHEMA_VERSION = "1.0"
WRITER_ID = "ND3.py/1.0"

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

AXES_ORDER = "TCZYXS"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Structural keys the writer owns inside meta_json; user meta may not shadow
# them (two homes for one fact is how metadata desyncs from pixels).
_STRUCTURAL_META_KEYS = frozenset(
    {"id", "axes", "shape", "dtype", "pixel_format", "channels", "planes"})

# dtype whitelist: kind -> allowed itemsizes.  Object / structured / complex /
# datetime dtypes are refused outright (spec §5.1).
_ALLOWED_DTYPES = {
    "b": {1},                    # bool
    "u": {1, 2, 4, 8},           # uint8..uint64
    "i": {1, 2, 4, 8},           # int8..int64
    "f": {2, 4, 8},              # float16/32/64
}

_YX_TILE = 1024                  # max Y/X chunk edge (partial reads of mosaics)
_GZIP_LEVEL = 4


class ND3Error(Exception):
    """Base class for every error this module raises deliberately."""


class ND3FormatError(ND3Error):
    """Not an .nd3 file (not HDF5 / missing marker attrs / malformed tree)."""


class ND3VersionError(ND3FormatError):
    """File's schema MAJOR is newer than this reader supports."""


class ND3ValidationError(ND3Error):
    """Bad writer input.  Always a refusal — never sanitized."""


class ND3IntegrityError(ND3Error):
    """Stored data failed a checksum / could not be read back."""


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _require_h5py() -> None:
    if h5py is None:
        raise ND3Error(
            "h5py is required for .nd3 files (pip install h5py); "
            f"import failed with: {_H5PY_IMPORT_ERROR!r}")


def _validate_id(name: str, *, what: str = "image id") -> str:
    if not isinstance(name, str) or not _ID_RE.match(name) or name in (".", ".."):
        raise ND3ValidationError(
            f"invalid {what} {name!r}: must match [A-Za-z0-9_.-]+ and not be "
            "'.'/'..' (refused, never sanitized)")
    return name


def _validate_axes(axes: str, shape: tuple) -> None:
    if not isinstance(axes, str) or not axes:
        raise ND3ValidationError("axes must be a non-empty string")
    if len(set(axes)) != len(axes):
        raise ND3ValidationError(f"axes {axes!r} has repeated characters")
    try:
        positions = [AXES_ORDER.index(c) for c in axes]
    except ValueError:
        bad = [c for c in axes if c not in AXES_ORDER]
        raise ND3ValidationError(
            f"axes {axes!r} contains unknown character(s) {bad}; allowed: "
            f"subsequence of {AXES_ORDER!r}")
    if positions != sorted(positions):
        raise ND3ValidationError(
            f"axes {axes!r} out of canonical order (must be a subsequence of "
            f"{AXES_ORDER!r})")
    if "Y" not in axes or "X" not in axes:
        raise ND3ValidationError(f"axes {axes!r} must contain both Y and X")
    if "S" in axes:
        if axes[-1] != "S":
            raise ND3ValidationError(
                f"axes {axes!r}: S (interleaved samples) may only be the "
                "final axis")
        s_size = shape[axes.index("S")]
        if not (2 <= s_size <= 4):
            raise ND3ValidationError(
                f"S axis size must be 2..4, got {s_size}")
    if len(axes) != len(shape):
        raise ND3ValidationError(
            f"axes {axes!r} has {len(axes)} characters but the array has "
            f"{len(shape)} dimensions")
    for dim, size in zip(axes, shape):
        if size < 1:
            raise ND3ValidationError(f"axis {dim} has size {size} (< 1)")


def _validate_dtype(dtype: "np.dtype") -> None:
    dt = np.dtype(dtype)
    sizes = _ALLOWED_DTYPES.get(dt.kind)
    if sizes is None or dt.itemsize not in sizes:
        raise ND3ValidationError(
            f"dtype {dt!r} is not allowed (spec §5.1: bool, u/int 8-64, "
            "float16/32/64)")


def _validate_pixel_format(pixel_format: str, axes: str, shape: tuple) -> None:
    if not isinstance(pixel_format, str) or not re.match(
            r"^[A-Za-z0-9_]+$", pixel_format or ""):
        raise ND3ValidationError(
            f"pixel_format {pixel_format!r} must match [A-Za-z0-9_]+ and is "
            "required (the field is the contract — readers must not guess)")
    s_size = shape[axes.index("S")] if "S" in axes else None
    three = {"RGB", "BGR"}
    four = {"RGBA", "BGRA"}
    if pixel_format in three and s_size != 3:
        raise ND3ValidationError(
            f"pixel_format {pixel_format!r} requires an S axis of size 3")
    if pixel_format in four and s_size != 4:
        raise ND3ValidationError(
            f"pixel_format {pixel_format!r} requires an S axis of size 4")
    if s_size is not None and pixel_format not in (three | four):
        raise ND3ValidationError(
            f"interleaved image (S axis) needs an interleaved pixel_format, "
            f"got {pixel_format!r}")


def _non_spatial_count(axes: str, shape: tuple) -> int:
    return math.prod(
        (size for dim, size in zip(axes, shape) if dim not in "YXS"),
        start=1)


def _validate_planes(planes, axes: str, shape: tuple) -> None:
    expected = _non_spatial_count(axes, shape)
    if len(planes) != expected:
        raise ND3ValidationError(
            f"planes has {len(planes)} records but the non-spatial axes "
            f"require exactly {expected}")
    axis_sizes = {dim: size for dim, size in zip(axes, shape)}
    key_for_axis = {"t": "T", "c": "C", "z": "Z"}
    for i, rec in enumerate(planes):
        if not isinstance(rec, dict):
            raise ND3ValidationError(f"planes[{i}] is not a dict")
        for key, axis in key_for_axis.items():
            if key in rec:
                if axis not in axis_sizes:
                    raise ND3ValidationError(
                        f"planes[{i}][{key!r}] given but the image has no "
                        f"{axis} axis")
                v = rec[key]
                if not isinstance(v, int) or isinstance(v, bool) or not (
                        0 <= v < axis_sizes[axis]):
                    raise ND3ValidationError(
                        f"planes[{i}][{key!r}]={v!r} out of range for axis "
                        f"{axis} (size {axis_sizes[axis]})")


def _validate_channels(channels, axes: str, shape: tuple) -> None:
    c_size = shape[axes.index("C")] if "C" in axes else None
    if c_size is not None and len(channels) != c_size:
        raise ND3ValidationError(
            f"channels has {len(channels)} entries but the C axis has size "
            f"{c_size}")
    if c_size is None and len(channels) > 1:
        raise ND3ValidationError(
            f"image without a C axis may carry at most one channel entry, "
            f"got {len(channels)}")
    for i, ch in enumerate(channels):
        if not isinstance(ch, dict) or not isinstance(ch.get("name"), str) \
                or not ch["name"]:
            raise ND3ValidationError(
                f"channels[{i}] must be a dict with a non-empty 'name'")


def _validate_user_meta(meta: dict) -> None:
    if not isinstance(meta, dict):
        raise ND3ValidationError("meta must be a dict")
    clash = _STRUCTURAL_META_KEYS & set(meta)
    if clash:
        raise ND3ValidationError(
            f"meta may not shadow writer-owned structural keys: {sorted(clash)}")
    try:
        json.dumps(meta)
    except (TypeError, ValueError) as exc:
        raise ND3ValidationError(f"meta is not JSON-serializable: {exc}")


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _read_bytes_dataset(ds) -> bytes:
    """Normalise the three shapes a bytes dataset can come back as (spec §3)."""
    val = ds[()]
    if isinstance(val, bytes):
        return val
    if isinstance(val, np.void):
        return val.tobytes()
    if isinstance(val, np.ndarray):
        return val.tobytes()
    if isinstance(val, str):
        return val.encode("utf-8")
    raise ND3FormatError(
        f"unsupported bytes-dataset payload type {type(val)!r} at {ds.name}")


def _chunks_for(axes: str, shape: tuple) -> tuple:
    """One plane per leading-dim chunk; Y/X tiled for partial reads."""
    out = []
    for dim, size in zip(axes, shape):
        if dim in "YX":
            out.append(min(size, _YX_TILE))
        elif dim == "S":
            out.append(size)
        else:
            out.append(1)
    return tuple(out)


def _parse_schema_version(text) -> tuple:
    try:
        major_s, minor_s = str(text).split(".")
        return int(major_s), int(minor_s)
    except Exception:
        raise ND3FormatError(f"malformed schema_version {text!r}")


def _attr_str(attrs, key) -> Optional[str]:
    if key not in attrs:
        return None
    val = attrs[key]
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    return str(val)


def _prepare_array(array) -> "np.ndarray":
    arr = np.asarray(array)
    _validate_dtype(arr.dtype)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return arr


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

class ND3Writer:
    """Write an .nd3 file atomically.

    Context-manager semantics: a clean exit commits (``close()`` — tmp file is
    os.replace'd onto the final path); an exception aborts (``abort()`` — the
    final path is never touched).
    """

    def __init__(self, path, *, dataset_meta: Optional[dict] = None,
                 compress: bool = True, overwrite: bool = False,
                 generator: Optional[dict] = None):
        _require_h5py()
        self._path = Path(path)
        if self._path.exists() and not overwrite:
            raise ND3ValidationError(
                f"{self._path} already exists (pass overwrite=True to replace)")
        if dataset_meta is not None:
            try:
                json.dumps(dataset_meta)
            except (TypeError, ValueError) as exc:
                raise ND3ValidationError(
                    f"dataset_meta is not JSON-serializable: {exc}")
        self._dataset_meta = dataset_meta
        self._compress = bool(compress)
        self._tmp = self._path.with_name(self._path.name + ".tmp")
        self._ids_lower: set = set()
        self._attachment_names_lower: set = set()
        self._open_appenders: list = []
        self._closed = False

        self._f = h5py.File(self._tmp, "w")
        self._f.attrs["format"] = FORMAT_NAME
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["created_iso"] = datetime.now().astimezone().isoformat()
        gen = {"software": "ND3", "writer": WRITER_ID}
        if generator:
            gen.update(generator)
        self._f.attrs["generator_json"] = json.dumps(gen, sort_keys=True)
        self._f.create_group("images")

    # -- internal ----------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ND3Error("writer is closed")

    def _claim_id(self, image_id: str) -> None:
        _validate_id(image_id)
        low = image_id.lower()
        if low in self._ids_lower:
            raise ND3ValidationError(
                f"image id {image_id!r} collides (case-insensitively) with an "
                "existing image — extraction on Windows/macOS would clobber")
        self._ids_lower.add(low)

    def _write_bytes(self, group, name: str, data: bytes):
        payload = np.frombuffer(bytes(data), dtype=np.uint8)
        return group.create_dataset(name, data=payload)

    def _dataset_kwargs(self, axes: str, shape: tuple, compress) -> dict:
        use_gzip = self._compress if compress is None else bool(compress)
        kwargs = {"chunks": _chunks_for(axes, shape), "fletcher32": True}
        if use_gzip:
            kwargs["compression"] = "gzip"
            kwargs["compression_opts"] = _GZIP_LEVEL
        return kwargs

    def _compose_meta(self, image_id, axes, shape, dtype, pixel_format,
                      meta, channels, planes) -> dict:
        doc = {
            "id": image_id,
            "axes": axes,
            "shape": [int(s) for s in shape],
            "dtype": str(np.dtype(dtype)),
            "pixel_format": pixel_format,
        }
        if meta:
            doc.update(meta)
        if channels is not None:
            doc["channels"] = channels
        if planes is not None:
            doc["planes"] = planes
        return doc

    def _validate_preview(self, preview_png) -> None:
        if not isinstance(preview_png, (bytes, bytearray)) or \
                not bytes(preview_png).startswith(PNG_MAGIC):
            raise ND3ValidationError(
                "preview_png must be pre-encoded PNG bytes (missing PNG magic "
                "— did you pass a raw array?)")

    # -- public ------------------------------------------------------------

    def add_image(self, image_id: str, array, *, axes: str, pixel_format: str,
                  meta: Optional[dict] = None,
                  channels: Optional[list] = None,
                  planes: Optional[list] = None,
                  preview_png: Optional[bytes] = None,
                  compress: Optional[bool] = None) -> None:
        self._check_open()
        arr = _prepare_array(array)
        _validate_axes(axes, arr.shape)
        _validate_pixel_format(pixel_format, axes, arr.shape)
        if meta is not None:
            _validate_user_meta(meta)
        if channels is not None:
            _validate_channels(channels, axes, arr.shape)
        if planes is not None:
            _validate_planes(planes, axes, arr.shape)
        if preview_png is not None:
            self._validate_preview(preview_png)
        self._claim_id(image_id)

        grp = self._f["images"].create_group(image_id)
        ds = grp.create_dataset(
            "data", data=arr, **self._dataset_kwargs(axes, arr.shape, compress))
        ds.attrs["axes"] = axes
        ds.attrs["pixel_format"] = pixel_format
        doc = self._compose_meta(image_id, axes, arr.shape, arr.dtype,
                                 pixel_format, meta, channels, planes)
        self._write_bytes(grp, "meta_json", _json_bytes(doc))
        if preview_png is not None:
            self._write_bytes(grp, "preview_png", bytes(preview_png))

    def begin_stack(self, image_id: str, *, axes: str, dtype,
                    frame_shape: tuple, pixel_format: str,
                    meta: Optional[dict] = None,
                    channels: Optional[list] = None,
                    preview_png: Optional[bytes] = None,
                    compress: Optional[bool] = None) -> "ND3StackAppender":
        """Create an extendable image and return an appender (time-lapse).

        ``axes`` must start with ``T``; frames appended via
        ``ND3StackAppender.append(frame, plane=None)`` grow the T axis by one.
        Metadata is finalized by ``finish()`` (called automatically by
        ``ND3Writer.close()`` if still open).
        """
        self._check_open()
        if not axes or axes[0] != "T":
            raise ND3ValidationError(
                f"begin_stack requires axes starting with T, got {axes!r}")
        frame_shape = tuple(int(s) for s in frame_shape)
        probe_shape = (1,) + frame_shape
        _validate_axes(axes, probe_shape)
        _validate_pixel_format(pixel_format, axes, probe_shape)
        _validate_dtype(dtype)
        if meta is not None:
            _validate_user_meta(meta)
        if channels is not None:
            _validate_channels(channels, axes, probe_shape)
        if preview_png is not None:
            self._validate_preview(preview_png)
        self._claim_id(image_id)

        grp = self._f["images"].create_group(image_id)
        ds = grp.create_dataset(
            "data", shape=(0,) + frame_shape,
            maxshape=(None,) + frame_shape,
            dtype=np.dtype(dtype).newbyteorder("<"),
            **self._dataset_kwargs(axes, (1,) + frame_shape, compress))
        ds.attrs["axes"] = axes
        ds.attrs["pixel_format"] = pixel_format
        appender = ND3StackAppender(self, grp, ds, image_id, axes, frame_shape,
                                    pixel_format, meta, channels, preview_png)
        self._open_appenders.append(appender)
        return appender

    def add_attachment(self, name: str, data, *,
                       media_type: str = "application/octet-stream") -> None:
        self._check_open()
        _validate_id(name, what="attachment name")
        low = name.lower()
        if low in self._attachment_names_lower:
            raise ND3ValidationError(
                f"attachment name {name!r} collides (case-insensitively) with "
                "an existing attachment")
        if not isinstance(data, (bytes, bytearray)):
            raise ND3ValidationError("attachment data must be bytes")
        self._attachment_names_lower.add(low)
        grp = self._f.require_group("attachments")
        ds = self._write_bytes(grp, name, bytes(data))
        ds.attrs["media_type"] = str(media_type)

    def close(self) -> None:
        """Finalize and atomically commit the file."""
        if self._closed:
            return
        for appender in list(self._open_appenders):
            appender.finish()
        if self._dataset_meta is not None:
            self._write_bytes(self._f, "dataset_json",
                              _json_bytes(self._dataset_meta))
        self._f.flush()
        self._f.close()
        self._closed = True
        os.replace(self._tmp, self._path)

    def abort(self) -> None:
        """Discard everything; the final path is never touched.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._f.close()
        except Exception:
            pass
        try:
            self._tmp.unlink()
        except OSError:
            pass

    def __enter__(self) -> "ND3Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class ND3StackAppender:
    """Frame-by-frame appender for a ``begin_stack`` image (T axis grows)."""

    def __init__(self, writer, group, dataset, image_id, axes, frame_shape,
                 pixel_format, meta, channels, preview_png):
        self._writer = writer
        self._group = group
        self._ds = dataset
        self._image_id = image_id
        self._axes = axes
        self._frame_shape = frame_shape
        self._pixel_format = pixel_format
        self._meta = meta
        self._channels = channels
        self._preview_png = preview_png
        self._planes: list = []
        self._any_plane = False
        self._finished = False

    @property
    def frames_written(self) -> int:
        return int(self._ds.shape[0])

    def append(self, frame, plane: Optional[dict] = None) -> None:
        if self._finished:
            raise ND3Error("stack appender already finished")
        arr = _prepare_array(frame)
        if arr.shape != self._frame_shape:
            raise ND3ValidationError(
                f"frame shape {arr.shape} != declared {self._frame_shape}")
        if np.dtype(arr.dtype).newbyteorder("<") != self._ds.dtype:
            raise ND3ValidationError(
                f"frame dtype {arr.dtype} != declared {self._ds.dtype}")
        if plane is not None and not isinstance(plane, dict):
            raise ND3ValidationError("plane must be a dict")
        n = self._ds.shape[0]
        self._ds.resize(n + 1, axis=0)
        self._ds[n] = arr
        if plane is not None:
            self._any_plane = True
        self._planes.append(dict(plane) if plane else {})

    def finish(self) -> None:
        """Write meta_json (with collected planes) and seal the stack."""
        if self._finished:
            return
        self._finished = True
        if self in self._writer._open_appenders:
            self._writer._open_appenders.remove(self)
        shape = tuple(int(s) for s in self._ds.shape)
        planes = None
        if self._any_plane:
            for i, rec in enumerate(self._planes):
                rec.setdefault("t", i)
            _validate_planes(self._planes, self._axes, shape)
            planes = self._planes
        if self._channels is not None and shape[0] >= 1:
            _validate_channels(self._channels, self._axes, shape)
        doc = self._writer._compose_meta(
            self._image_id, self._axes, shape, self._ds.dtype,
            self._pixel_format, self._meta, self._channels, planes)
        self._writer._write_bytes(self._group, "meta_json", _json_bytes(doc))
        if self._preview_png is not None:
            self._writer._write_bytes(self._group, "preview_png",
                                      bytes(self._preview_png))


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def sniff(path) -> bool:
    """True iff ``path`` looks like an .nd3 file.  Never raises."""
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != HDF5_MAGIC:
                return False
        if h5py is None:
            return False
        with h5py.File(path, "r") as f:
            return _attr_str(f.attrs, "format") == FORMAT_NAME
    except Exception:
        return False


def open_nd3(path) -> "ND3Reader":
    return ND3Reader(path)


class ND3Image:
    """Lazy handle: constructing it reads metadata only, never pixel data."""

    def __init__(self, reader, image_id, group):
        self._reader = reader
        self._group = group
        self.id = image_id
        if "data" not in group:
            raise ND3FormatError(f"images/{image_id} has no 'data' dataset")
        ds = group["data"]
        self.axes = _attr_str(ds.attrs, "axes") or ""
        self.pixel_format = _attr_str(ds.attrs, "pixel_format") or ""
        self.shape = tuple(int(s) for s in ds.shape)
        self.dtype = np.dtype(ds.dtype)
        if "meta_json" not in group:
            raise ND3FormatError(f"images/{image_id} has no meta_json")
        try:
            self.meta = json.loads(
                _read_bytes_dataset(group["meta_json"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ND3FormatError(
                f"images/{image_id}/meta_json is not valid JSON: {exc}")
        self.channels = self.meta.get("channels", [])
        self.planes = self.meta.get("planes", [])

    def _live_group(self):
        # An h5py object tied to a closed file is falsy; h5py's own error for
        # it ("invalid identifier type to function") is too cryptic to ship.
        if not self._group:
            raise ND3Error(
                f"images/{self.id}: the ND3Reader that produced this handle "
                "is closed — read pixel data while the reader is open")
        return self._group

    def array(self) -> "np.ndarray":
        try:
            return self._live_group()["data"][()]
        except OSError as exc:
            raise ND3IntegrityError(
                f"images/{self.id}/data failed to read (checksum?): {exc}")

    def section(self, slices) -> "np.ndarray":
        """Partial read, e.g. ``img.section(np.s_[0, 100:200, 100:200])``."""
        try:
            return self._live_group()["data"][slices]
        except OSError as exc:
            raise ND3IntegrityError(
                f"images/{self.id}/data failed to read (checksum?): {exc}")

    def preview_png(self) -> Optional[bytes]:
        grp = self._live_group()
        if "preview_png" not in grp:
            return None
        return _read_bytes_dataset(grp["preview_png"])

    def _transform(self, key) -> Optional["np.ndarray"]:
        mat = (self.meta.get("transforms") or {}).get(key)
        if mat is None:
            return None
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ND3FormatError(
                f"images/{self.id} transforms[{key!r}] is not 3x3")
        return arr

    def pixel_to_stage_um(self) -> Optional["np.ndarray"]:
        return self._transform("pixel_to_stage_um")

    def pixel_to_plate_mm(self) -> Optional["np.ndarray"]:
        return self._transform("pixel_to_plate_mm")


class ND3Reader:
    def __init__(self, path):
        _require_h5py()
        self._path = Path(path)
        try:
            self._f = h5py.File(self._path, "r")
        except Exception as exc:
            raise ND3FormatError(f"{self._path} is not readable HDF5: {exc}")
        fmt = _attr_str(self._f.attrs, "format")
        if fmt != FORMAT_NAME:
            self._f.close()
            raise ND3FormatError(
                f"{self._path} is HDF5 but not nd3 (format attr = {fmt!r})")
        version = _attr_str(self._f.attrs, "schema_version")
        try:
            major, _minor = _parse_schema_version(version)
        except ND3FormatError:
            self._f.close()
            raise
        supported_major = _parse_schema_version(SCHEMA_VERSION)[0]
        if major > supported_major:
            self._f.close()
            raise ND3VersionError(
                f"{self._path} has schema_version {version}; this reader "
                f"supports major {supported_major}")
        self.schema = version

    # -- metadata ------------------------------------------------------------

    @property
    def dataset_meta(self) -> dict:
        if "dataset_json" not in self._f:
            return {}
        try:
            return json.loads(
                _read_bytes_dataset(self._f["dataset_json"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ND3FormatError(f"dataset_json is not valid JSON: {exc}")

    def image_ids(self) -> list:
        if "images" not in self._f:
            return []
        return sorted(self._f["images"].keys())

    def image(self, image_id: str) -> ND3Image:
        if "images" not in self._f or image_id not in self._f["images"]:
            raise ND3Error(f"no image {image_id!r} in {self._path}")
        return ND3Image(self, image_id, self._f["images"][image_id])

    def attachment_names(self) -> list:
        if "attachments" not in self._f:
            return []
        return sorted(self._f["attachments"].keys())

    def read_attachment(self, name: str) -> bytes:
        if "attachments" not in self._f or name not in self._f["attachments"]:
            raise ND3Error(f"no attachment {name!r} in {self._path}")
        return _read_bytes_dataset(self._f["attachments"][name])

    # -- verification ---------------------------------------------------------

    def verify(self, *, deep: bool = True) -> list:
        """Return problem strings ([] = OK).  ``deep`` reads every dataset,
        which drives HDF5's Fletcher32 checksum validation."""
        problems: list = []
        try:
            self.dataset_meta
        except ND3Error as exc:
            problems.append(f"dataset_json: {exc}")
        for image_id in self.image_ids():
            grp = self._f["images"][image_id]
            try:
                img = ND3Image(self, image_id, grp)
            except ND3Error as exc:
                problems.append(f"images/{image_id}: {exc}")
                continue
            try:
                _validate_axes(img.axes, img.shape)
                _validate_pixel_format(img.pixel_format, img.axes, img.shape)
            except ND3ValidationError as exc:
                problems.append(f"images/{image_id}: {exc}")
            if deep:
                try:
                    ds = grp["data"]
                    if ds.chunks:
                        for chunk_slice in ds.iter_chunks():
                            ds[chunk_slice]
                    else:
                        ds[()]
                except OSError as exc:
                    problems.append(
                        f"images/{image_id}/data: read failed "
                        f"(checksum?): {exc}")
                if "preview_png" in grp:
                    try:
                        _read_bytes_dataset(grp["preview_png"])
                    except (ND3Error, OSError) as exc:
                        problems.append(
                            f"images/{image_id}/preview_png: {exc}")
        if deep:
            for name in self.attachment_names():
                try:
                    self.read_attachment(name)
                except (ND3Error, OSError) as exc:
                    problems.append(f"attachments/{name}: {exc}")
        return problems

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "ND3Reader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def verify(path, *, deep: bool = True) -> list:
    """Open + verify in one call; format/version problems come back as
    strings too (so a caller can point at ANY bad file without try/except)."""
    try:
        with open_nd3(path) as reader:
            return reader.verify(deep=deep)
    except ND3Error as exc:
        return [str(exc)]
