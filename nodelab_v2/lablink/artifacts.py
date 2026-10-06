"""What a LabLink session sends back: tables, quicklooks, thumbnails and metrics.

The hub is standard-library-only and **must never decode or generate an image** — that is
the property that lets it sit on an instrument PC — so producing every returnable byte is
the worker's job, and this module is that job. One function per artifact kind declared in
:data:`~nodelab_v2.lablink.protocol.ARTIFACT_KINDS`, each returning the ``artifact`` event
dict the hub stores and publishes.

Three rules the shapes here exist to satisfy:

* **Bulk data never rides the pipe.** Every artifact is a file under the session
  directory and the event carries its path, its size and its sha256. The hub's line limit
  is 1 MiB and a line over it kills the worker, so a "just inline the CSV" shortcut is not
  a shortcut.
* **A preview is not the table.** A measurement CSV can be a million rows; the hub shows
  the node a few. So a table artifact writes a *separate* small preview file and reports
  ``total_rows`` alongside the truncated count, which is what lets a UI say "3 of 118 407"
  instead of implying the table has three rows.
* **PNG from zlib and struct, not from an imaging library.** ~15 lines of standard
  library, and it removes Pillow/OpenCV from the path that decides whether a session can
  return anything at all. The reference worker in the LabLink repo does the same, for the
  same reason: a thumbnail contract that has never been exercised is a contract that does
  not work.

**Downsampling a label raster strides, it never interpolates.** A quicklook of a
segmentation is reduced by taking every Nth pixel, because averaging label ids invents
regions that were never segmented — id 4 next to id 6 averages to 5, a real object
somewhere else in the image. Grayscale intensity is strided too, for one honest rule
rather than two.

Qt-free; numpy + standard library (``nodelab_v2.export`` for the CSV, itself Qt-free).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nodegraph.domains import Domain

#: Longest edge of a quicklook, in pixels. Big enough to see whether a segmentation is
#: sane, small enough that returning one is never the reason a session felt slow.
QUICKLOOK_MAX_PX = 1024

#: Longest edge of a thumbnail — the inline preview a console or panel renders in a list.
THUMB_MAX_PX = 128

#: Rows a table artifact's preview carries.
PREVIEW_ROWS = 20

#: Distinct hues for label colouring, as uint8 RGB. A qualitative ramp: adjacent ids get
#: unrelated colours, so two touching regions are visibly two regions. Deliberately short
#: and cycled — a quicklook answers "did this segment sensibly", not "which id is which".
_LABEL_COLORS = np.array([
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (26, 188, 156), (241, 196, 15), (52, 73, 94), (231, 76, 60),
], dtype=np.uint8)


@dataclass
class Artifact:
    """One returnable file, and the ``artifact`` event that announces it."""

    name: str
    kind: str
    path: str
    nbytes: int
    sha256: str
    policy: str = "auto"
    preview: Optional[Dict[str, Any]] = None
    thumb: Optional[Dict[str, Any]] = None
    dims: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def event(self) -> Dict[str, Any]:
        """The ``ev="artifact"`` payload. ``held`` mirrors the policy because the hub
        keys its publish/withhold decision on it and a node reads it to know whether a
        file is already on the out channel or waiting for a ``pull``."""
        ev: Dict[str, Any] = {
            "name": self.name, "kind": self.kind, "path": self.path,
            "bytes": self.nbytes, "sha256": self.sha256,
            "policy": self.policy, "held": self.policy == "pull",
        }
        if self.preview is not None:
            ev["preview"] = self.preview
        if self.thumb is not None:
            ev["thumb"] = self.thumb
        if self.dims is not None:
            ev["dims"] = self.dims
        ev.update(self.extra)
        return ev


# ── primitives ──────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, *, chunk: int = 1 << 20) -> Tuple[str, int]:
    """``(hex digest, size)`` read in chunks, so hashing a 3 GB volume costs a buffer.

    The size comes from what was actually read rather than from ``stat``: the two can
    disagree if anything is still writing, and the digest must describe the bytes it saw.
    """
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            total += len(block)
    return h.hexdigest(), total


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as a PNG, using only zlib and struct.

    Truecolour, 8 bits, filter type 0 per row — the simplest conforming encoding, which is
    what keeps this short enough to be obviously correct. Level 6 compression: a quicklook
    is written once and read once, so squeezing the last few percent is not worth the CPU
    inside a session's critical path.
    """
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"png_bytes expects (H, W, 3) uint8; got {arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    # one leading filter byte per row, then the row's RGB triplets
    raw = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = arr.reshape(height, width * 3)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + chunk(b"IEND", b""))


def _stride_to(plane: np.ndarray, max_px: int) -> np.ndarray:
    """Reduce a 2-D plane so its longest edge is ``<= max_px``, by STRIDING.

    Never averages — see the module docstring. Integer label ids and float intensities go
    through the same path so there is one rule to reason about.
    """
    h, w = int(plane.shape[0]), int(plane.shape[1])
    longest = max(h, w)
    if longest <= max_px or max_px <= 0:
        return plane
    step = int(np.ceil(longest / float(max_px)))
    return plane[::step, ::step]


def _to_gray_u8(plane: np.ndarray) -> np.ndarray:
    """A 2-D plane of any numeric dtype → uint8, on a robust percentile window.

    2nd–98th percentile rather than min–max: one hot pixel (a cosmic ray, a saturated
    speck of debris) compresses a min–max stretch into the bottom few levels and the
    quicklook comes back black, which reads as "the pipeline produced nothing".
    A constant plane maps to mid-grey, because a 0-width window would divide by zero.
    """
    finite = np.asarray(plane, dtype=np.float64)
    good = finite[np.isfinite(finite)]
    if good.size == 0:
        return np.zeros(finite.shape, dtype=np.uint8)
    lo, hi = (float(np.percentile(good, 2.0)), float(np.percentile(good, 98.0)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(finite.shape, 128, dtype=np.uint8)
    scaled = (np.nan_to_num(finite, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _colorize_labels(labels: np.ndarray, under: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """An integer label plane → ``(H, W, 3)`` uint8, optionally over a grayscale image.

    Background (id 0) shows the underlying image where there is one, so the quicklook
    answers the question actually being asked: did the regions land on the objects.
    Regions are blended rather than painted flat for the same reason — a solid fill hides
    whether a region covers a cell or the space beside it.
    """
    ids = np.asarray(labels)
    base = (np.repeat(_to_gray_u8(under)[:, :, None], 3, axis=2)
            if under is not None and under.shape == ids.shape
            else np.zeros(ids.shape + (3,), dtype=np.uint8))
    fg = ids > 0
    if not np.any(fg):
        return base
    # id -> colour by modular index, so ids need not be dense or start at 1
    idx = (ids[fg].astype(np.int64) - 1) % len(_LABEL_COLORS)
    tint = _LABEL_COLORS[idx]
    out = base.copy()
    out[fg] = (0.45 * base[fg].astype(np.float32)
               + 0.55 * tint.astype(np.float32)).astype(np.uint8)
    return out


# ── choosing what to draw ───────────────────────────────────────────────────────

def _plane_coords(axes: Any) -> Tuple[int, int, int, int]:
    """``(m, t, z, c)`` for the plane a quicklook shows: first position, first timepoint,
    MIDDLE z, first channel.

    Mid-z rather than z=0 because the top plane of a z-stack is routinely empty space
    above the sample, and a quicklook of empty space is indistinguishable from a failed
    run."""
    z = max(0, int(getattr(axes, "z", 1)) // 2)
    return 0, 0, z, 0


def _label_rasters(dataset: Any) -> Dict[str, np.ndarray]:
    """Every integer 6-D Voxel layer on a Dataset, by name — the label rasters.

    Same test the Viewer's label picker uses (integer dtype, 6-D, named), so a raster the
    app would draw is a raster a quicklook draws.
    """
    out: Dict[str, np.ndarray] = {}
    if dataset is None or not hasattr(dataset, "attributes"):
        return out
    for (dom, _layer, name), attr in dataset.attributes.items():
        if dom is not Domain.VOXEL or not name:
            continue
        vals = attr.values
        if np.issubdtype(np.asarray(vals).dtype, np.integer) and np.asarray(vals).ndim == 6:
            out[str(name)] = vals
    return out


def _image_plane(dataset: Any, m: int, t: int, z: int, c: int,
                 max_px: int) -> Optional[np.ndarray]:
    """A 2-D plane off the Dataset's lazy image provider, read at a COARSE pyramid level.

    Reading level 0 and then throwing 99% of it away is the obvious implementation and the
    wrong one: on the lab's 12 000 px mosaics that is hundreds of MB decoded to produce a
    1024 px picture. The provider already holds a pyramid, so pick the coarsest level whose
    longest edge still covers ``max_px`` and read that.
    """
    prov = getattr(dataset, "image", None)
    if prov is None:
        return None
    level = 0
    try:
        for cand in range(0, 16):
            ax = prov.level_axes(cand)
            if max(int(ax.y), int(ax.x)) < max_px:
                break
            level = cand
        ax = prov.level_axes(level)
        # the coordinates are level-invariant for m/t/c; z is not resampled by the pyramid
        zz = min(int(z), max(0, int(ax.z) - 1))
        return np.asarray(prov.read_region(level, int(m), int(t), zz, int(c),
                                          0, int(ax.y), 0, int(ax.x)))
    except Exception:       # noqa: BLE001 — a quicklook must never fail a run
        return None


# ── the artifact writers ────────────────────────────────────────────────────────

def write_table(dataset: Any, out_dir: str, name: str, *, policy: str = "auto",
                fmt: str = "csv") -> Optional[Artifact]:
    """Write a pulled Dataset's tables as one long-form CSV, plus a small preview.

    Returns ``None`` when the Dataset carries nothing tabulatable — a legitimate outcome
    (a threshold that found no objects is a scientific result, not a fault), so the caller
    reports an empty table rather than failing the command.
    """
    from nodelab_v2.export import export_csv
    from nodelab_v2.tables import all_tables

    tables = all_tables(dataset)
    if not tables:
        return None
    os.makedirs(out_dir, exist_ok=True)
    stem = name[:-4] if name.lower().endswith(".csv") else name
    filename = stem + (".csv" if fmt in ("", "csv") else "." + fmt)
    path = os.path.join(out_dir, filename)
    rows = export_csv(tables, path)
    digest, nbytes = sha256_file(path)

    # the preview is a separate FILE, so the hub can hand a node a few rows without
    # publishing (or reading) a table that may be hundreds of MB
    preview_path = os.path.join(out_dir, stem + ".preview.csv")
    kept = 0
    with open(path, "r", encoding="utf-8", newline="") as src, \
            open(preview_path, "w", encoding="utf-8", newline="") as dst:
        for i, line in enumerate(src):
            if i > PREVIEW_ROWS:
                break
            dst.write(line)
            kept = i                      # i=0 is the header; kept ends at the last row
    return Artifact(
        name=filename, kind="table", path=path, nbytes=nbytes, sha256=digest,
        policy=policy,
        preview={"kind": "csv", "path": preview_path, "rows": kept,
                 "total_rows": rows, "truncated": rows > kept},
        extra={"rows": rows, "tables": len(tables)})


def _picture_rgb(dataset: Any, m: int, t: int, max_px: int):
    """A PICTURE's (a plot's figure) R, G and B planes as one uint8 image — or ``None`` for
    anything else. A chart is not a microscope image: it has no channel to grey-stretch."""
    md = getattr(dataset, "metadata", None) or {}
    axes = getattr(dataset, "axes", None)
    prov = getattr(dataset, "image", None)
    if not md.get("picture") or axes is None or prov is None or getattr(axes, "c", 0) < 3:
        return None
    planes = [_stride_to(np.asarray(prov.get_region(0, m, t, 0, ch, 0, axes.y, 0, axes.x)),
                         max_px) for ch in range(3)]
    return np.clip(np.stack(planes, axis=-1), 0, 255).astype(np.uint8)


def write_quicklook(dataset: Any, out_dir: str, name: str, *, policy: str = "auto",
                    max_px: int = QUICKLOOK_MAX_PX) -> Optional[Artifact]:
    """Render one plane of a pulled Dataset as a PNG, with a thumbnail beside it.

    Prefers a label raster over the raw image, because the node asking for a quicklook of
    a segmentation wants to see the segmentation. Returns ``None`` if there is neither —
    a Dataset carrying only a table has no picture to draw and that is not an error.
    """
    axes = getattr(dataset, "axes", None)
    if axes is None:
        return None
    m, t, z, c = _plane_coords(axes)
    pic = _picture_rgb(dataset, m, t, max_px)       # a plot's figure: its own RGB as drawn
    rasters = {} if pic is not None else _label_rasters(dataset)
    under = None if pic is not None else _image_plane(dataset, m, t, z, c, max_px)

    if pic is not None:
        rgb, drew = pic, "picture"
    elif rasters:
        # a stable pick: the first by name, so two runs of one recipe draw the same layer
        chosen = sorted(rasters)[0]
        vol = np.asarray(rasters[chosen])
        zz = min(int(z), max(0, int(vol.shape[2]) - 1))
        cc = min(int(c), max(0, int(vol.shape[3]) - 1))
        plane = _stride_to(vol[min(m, vol.shape[0] - 1), min(t, vol.shape[1] - 1),
                               zz, cc], max_px)
        # the image under the labels must be reduced to the SAME grid or the blend is
        # meaningless — a shape mismatch just drops it (see _colorize_labels)
        rgb = _colorize_labels(plane, _stride_to(under, max_px)
                               if under is not None else None)
        drew = f"labels:{chosen}"
    elif under is not None:
        rgb = np.repeat(_to_gray_u8(_stride_to(under, max_px))[:, :, None], 3, axis=2)
        drew = "image"
    else:
        return None

    os.makedirs(out_dir, exist_ok=True)
    filename = name if name.lower().endswith(".png") else name + ".png"
    stem = filename[:-4]
    path = os.path.join(out_dir, filename)
    blob = png_bytes(rgb)
    with open(path, "wb") as fh:
        fh.write(blob)

    thumb_rgb = _stride_to(rgb, THUMB_MAX_PX)
    thumb_path = os.path.join(out_dir, stem + ".thumb.png")
    thumb_blob = png_bytes(thumb_rgb)
    with open(thumb_path, "wb") as fh:
        fh.write(thumb_blob)

    return Artifact(
        name=filename, kind="quicklook", path=path, nbytes=len(blob),
        sha256=sha256_bytes(blob), policy=policy,
        thumb={"name": os.path.basename(thumb_path), "path": thumb_path,
               "bytes": len(thumb_blob), "w": int(thumb_rgb.shape[1]),
               "h": int(thumb_rgb.shape[0])},
        dims={"shape": [int(getattr(axes, a, 1)) for a in ("m", "t", "z", "c", "y", "x")],
              "plane": {"m": m, "t": t, "z": z, "c": c},
              "drew": drew,
              "voxel_um": [getattr(dataset, "metadata", {}).get("z_step_um"),
                           getattr(dataset, "metadata", {}).get("pixel_size_um"),
                           getattr(dataset, "metadata", {}).get("pixel_size_um")]},
        extra={"w": int(rgb.shape[1]), "h": int(rgb.shape[0])})


def write_metrics(payload: Dict[str, Any], out_dir: str, name: str, *,
                  policy: str = "auto") -> Artifact:
    """Write a small JSON metrics blob.

    ``allow_nan=False``: NaN is not JSON, and one of them reaching a stored record makes
    that document unparseable for every client that is not Python. The caller has already
    neutralised non-finite values via :func:`~nodelab_v2.lablink.worker.sanitize`, so this
    raising would be a bug rather than a data condition — which is exactly why it is left
    able to raise.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = name if name.lower().endswith(".json") else name + ".json"
    path = os.path.join(out_dir, filename)
    blob = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(blob)
    return Artifact(name=filename, kind="metrics", path=path, nbytes=len(blob),
                    sha256=sha256_bytes(blob), policy=policy)


def write_image_stack(dataset: Any, out_dir: str, name: str, *,
                      policy: str = "pull") -> Optional[Artifact]:
    """Write a label raster as a real multi-page TIFF, for a recipe that asks for one.

    Defaults to ``policy="pull"`` and the default is the point: a full-resolution label
    volume is the artifact most likely to be gigabytes and least likely to be opened, so
    the hub holds it until a node asks by name. ``tifffile`` is a hard dependency of the
    app's ingest path, so this needs no extra install; it is imported here rather than at
    module scope so a session that returns no stack never pays for it.
    """
    rasters = _label_rasters(dataset)
    if not rasters:
        return None
    import tifffile

    chosen = sorted(rasters)[0]
    vol = np.asarray(rasters[chosen])
    os.makedirs(out_dir, exist_ok=True)
    filename = name if name.lower().endswith((".tif", ".tiff")) else name + ".tif"
    path = os.path.join(out_dir, filename)
    # squeeze the singleton acquisition axes so the file opens as a plain stack in
    # ImageJ/Fiji rather than as a 6-D hyperstack nothing renders by default
    tifffile.imwrite(path, np.squeeze(vol))
    digest, nbytes = sha256_file(path)
    return Artifact(name=filename, kind="image_stack", path=path, nbytes=nbytes,
                    sha256=digest, policy=policy,
                    extra={"layer": chosen, "dtype": str(vol.dtype),
                           "shape": [int(s) for s in vol.shape]})


#: Dispatch by the ``kind`` a recipe's output declares. A kind absent here is refused at
#: ``open`` as a tier-2 ``bad_recipe``, which is the whole reason ``hello`` reports
#: ``artifact_kinds``: the alternative is a missing file after the run.
WRITERS = {
    "table": write_table,
    "quicklook": write_quicklook,
    "image_stack": write_image_stack,
    # "metrics" is not here on purpose: it takes a payload the run assembles, not a
    # Dataset, so the run loop calls write_metrics directly.
}


__all__ = [
    "Artifact", "QUICKLOOK_MAX_PX", "THUMB_MAX_PX", "PREVIEW_ROWS", "WRITERS",
    "sha256_bytes", "sha256_file", "png_bytes",
    "write_table", "write_quicklook", "write_metrics", "write_image_stack",
]
