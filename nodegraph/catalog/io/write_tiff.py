"""Export TIFF (``io.write_tiff``) — stream the Dataset on the wire to a TIFF on disk, one
plane at a time, carrying the calibration envelope out verbatim as OME-XML."""

from __future__ import annotations


import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nodegraph.dataset import Dataset
from nodegraph.metadata import position_group_plan
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.memo import value_digest
from nodegraph.registry import (Granularity, InBool, InDataset, InInt, InString, Mode,
                               OutDataset)

from nodegraph.catalog._base import register_node

# ── Export TIFF (io — the pipeline's write end) ────────────────────────────────
#
# The counterpart to `io.load`, and the first node in the catalog whose product is a FILE
# rather than a Dataset. Two properties drive every decision in here:
#
#   1. It is a SIDE EFFECT inside a pure-function engine. Everything else in the catalog can
#      be re-run for free; this cannot. §"Re-running" below is the whole story.
#   2. It is the boundary where the metadata envelope stops being an internal model and
#      becomes something Fiji, QuPath and napari have to agree with. So the envelope is
#      transcribed, never re-derived — see `_ome_metadata`.
#
# It passes the Dataset through unchanged, so it is a TAP rather than a terminal: drop it
# mid-chain, keep wiring downstream, and a Viewer after it still shows what was written.

#: Private TIFF tag holding this node's export STAMP — a digest of everything that
#: determines the file's bytes (see :func:`_stamp`). Written into the first IFD of every
#: file this node produces, in all three metadata models, and read back to decide whether an
#: existing file is already this exact export (`existing="skip"`).
#:
#: 65000 is in the private/reusable range (65000-65535) that the TIFF spec leaves to
#: applications, so no reader is confused by it and no standard tag is displaced. Chosen over
#: a sidecar file because an export is something the user hands to someone else, and a second
#: file beside it is clutter that goes missing exactly when it is needed. Verified to survive
#: `ome=True`, `imagej=True` and a bare write.
STAMP_TAG = 65000

#: ``model`` mode → the writer keyword that selects the metadata container.
_MODEL_KW = {"ome": "ome", "imagej": "imagej", "plain": ""}

#: ``compression`` mode → (tifffile codec name, accepts ``level``, ``COMPRESSION`` member).
#:
#: ``lzma`` takes no level: its encoder signature has no ``preset`` (verified in-env against
#: tifffile 2026.5.15, which raises ``TypeError: lzma_encode() got an unexpected keyword
#: argument 'preset'``), which is why `level` is `available_in`-gated away from it rather
#: than accepted and dropped.
#:
#: The third field exists because tifffile's own name→codec coercion is **not** reusable
#: here: ``enumarg(COMPRESSION, "zlib")`` and ``COMPRESSION["ZLIB"]`` both raise, and only
#: ``TiffWriter.write``'s private branch knows that ``"zlib"`` means ``ADOBE_DEFLATE`` (tag
#: 8). So the enum member is spelled out once, beside the string it corresponds to, rather
#: than guessed by upper-casing — which is what made the availability probe below report
#: "zlib unavailable" on an installation that encodes zlib perfectly well.
_CODECS = {"none": (None, False, "NONE"), "zlib": ("zlib", True, "ADOBE_DEFLATE"),
           "lzma": ("lzma", False, "LZMA"), "zstd": ("zstd", True, "ZSTD")}

#: Bytes above which the file is written as BigTIFF. Plain TIFF's offsets are 32-bit, so the
#: hard ceiling is 4 GiB; the margin covers the IFDs and the OME-XML block, which are written
#: last and whose size is not known when the container is chosen. Derived rather than
#: socketed: the answer follows from the axes and the dtype, and a user who sets it wrong
#: gets either a refused write or an unnecessarily exotic file. ImageJ hyperstacks cannot be
#: BigTIFF at all, which is a refusal, not a choice.
_BIGTIFF_BYTES = 3_900_000_000


def _codec_usable(mode_name: str) -> bool:
    """Whether tifffile can actually ENCODE with the ``compression`` mode ``mode_name`` here.

    Not the same question as whether tifffile *knows* the codec: ``COMPRESSION.ZSTD in
    TIFF.COMPRESSORS`` answers ``True`` on this machine and then fails at write time, because
    the fallback shim imports ``compression.zstd`` (Python 3.14+) and ``imagecodecs`` is not
    installed. So the only honest test is to run the encoder on a few bytes — which is cheap,
    and is what turns a ``ModuleNotFoundError`` thrown partway through a multi-gigabyte export
    into a refusal, up front, that names the package to install.
    """
    entry = _CODECS.get(mode_name)
    if entry is None or entry[0] is None:
        return True
    try:
        from tifffile import COMPRESSION
        from tifffile.tifffile import TIFF
        TIFF.COMPRESSORS[COMPRESSION[entry[2]]](b"\0" * 64)
        return True
    except Exception:      # noqa: BLE001 — any failure means "cannot encode with this"
        return False


def _compute_write_tiff(ctx: EvalContext) -> Dataset:
    """Stream the Dataset (or one of its Voxel layers) to a TIFF, and hand it through.

    Resolved spec (§0 grill, 2026-08-05)
    ------------------------------------
    * **Kind** io / side effect → ``op_key="io.write_tiff"``, category ``"io"``. The op_key
      is the one the V3.00 roadmap froze for this slot (W6-P1).
    * **Data contract** ``Dataset -> the SAME Dataset``, byte-identical: same provider, same
      axes, same calibration, same layers. Not axis-changing, so **no** ``meta_transform``,
      and nothing downstream can tell an export happened. That makes it a tap rather than a
      terminal — a Viewer after it still draws what was written, which is the only way to
      check an export without opening the file.
    * **2D/3D** no lever. The node neither reads a neighbourhood nor changes dimensionality;
      it writes whatever axes are on the wire. A lever here could only disagree with the
      data (`wire-node-v2` §5: derive, don't ask).
    * **Footprint** ``WHOLE_PLANE`` / ``kernel_axes={"y","x"}``. This is the streaming
      decision made honest: the compute reads exactly one full ``(Y,X)`` plane at a time via
      ``provider.get_region`` and hands it straight to an open ``TiffWriter``, so peak
      memory is one plane no matter how large the series. The roadmap had pencilled in
      ``WHOLE_SERIES``, which is what materializing the whole 6-D array first would have
      required — and what makes the lab's 49-position 6554² files unexportable.
    * **Sockets** ``path`` (``save_file``), ``layer`` (Voxel, empty = the image), ``level``
      (gated to the codecs that accept one). Modes: ``split``, ``model``
      (ome/imagej/plain), ``compression``, ``existing``.
    * **Backend** tifffile 2026.5.15, already a dependency. Re-verified in-env: an iterator
      of planes with ``shape``/``dtype`` streams; one ``write`` per multipoint gives one OME
      ``Image`` per position; ``maxworkers`` parallelizes the encode behind the iterator;
      every OME field below round-trips. **LZW, packbits and zstd are NOT usable here** —
      they need ``imagecodecs``, which is not installed (see :func:`_codec_usable`).
    * **Numba** no. The hot path is tifffile's C encoder plus provider reads; there is no
      Python loop over small arrays to fuse.

    Calibration is resolved **eagerly**, before the writer opens. Not stylistic: the export
    is a side effect performed inside the compute, and a ``ctx.calib`` call from inside the
    plane iterator would run after ``ReadContext.freeze`` and be a hard error (C1 / V2.04
    §6b) — the hazard the roadmap flagged against this node by name.

    Re-running
    ----------
    A memo HIT returns the cached payload without entering this function at all
    (``engine.py``), so an unchanged graph re-pulled in the same session writes nothing.
    That is the fast behaviour and it is correct: nothing changed. The case that needs care
    is the opposite one — the compute IS entered, because something upstream changed, and a
    file with that name already exists. Skipping it there would leave a stale export sitting
    under the name of a fresh one, which is why ``existing="skip"`` does not mean "a file is
    there, leave it": it means *"a file is there and its stamp proves it is already this
    exact export"*. Anything else is overwritten. See :func:`_stamp`.

    The write goes to ``<path>.part`` and is renamed on success, so an interrupted export
    cannot leave a truncated file wearing the real name — the same reasoning as
    ``nodegraph.checkpoint``'s completeness marker.
    """
    import tifffile

    ds = ctx.inputs[0]
    path = str(ctx.params.get("path", "") or "").strip()
    if not path:
        raise ValueError(
            "Export TIFF has no destination: set `path` (the Browse… button beside it opens "
            "a save dialog). There is deliberately no default — a node that invented a "
            "filename would write megabytes somewhere the user never chose.")

    modes = ctx.params.get("__modes__", {})
    model = str(modes.get("model", "ome"))
    comp_name = str(modes.get("compression", "zlib"))
    existing = str(modes.get("existing", "skip"))
    split = str(modes.get("split", "none"))
    if ctx.params.get("split_positions"):
        raise ValueError(
            "Export TIFF: `split_positions` has been replaced by the Split mode, which can "
            "also write one file per GROUP. Set Split to 'position' for what the tick box "
            "did, then clear the old value. Refused rather than translated because the two "
            "are not the same control: a graph saved with the box ticked would otherwise "
            "keep exporting per position while the header said 'none'.")
    level = int(ctx.params.get("level", 1))

    codec, takes_level, _enum = _CODECS[comp_name]
    if not _codec_usable(comp_name):
        alt = [n for n in _CODECS if n != comp_name and _codec_usable(n)]
        raise ValueError(
            f"compression={comp_name!r} is not available in this Python: tifffile has no "
            f"working {comp_name} encoder here. It needs the `imagecodecs` package (not "
            f"installed — `pip install imagecodecs`), or Python 3.14's built-in zstd. "
            f"Usable here: {', '.join(alt) or 'none'}.")

    # ── what gets written: the image, or one Voxel raster ──────────────────────
    ax = ds.axes
    layer = ctx.layer("layer")
    arr: Optional[np.ndarray] = None
    if layer:
        got = ds.get(Domain.VOXEL, layer)
        if got is None:
            have = sorted({a.name for a in ds.layers_on(Domain.VOXEL)})
            raise ValueError(
                f"no Voxel layer {layer!r} to export — this Dataset carries {have or 'none'}. "
                f"Leave `layer` empty to export the IMAGE instead.")
        arr = np.asarray(got.values)
        if arr.shape != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
            raise ValueError(
                f"Voxel layer {layer!r} has shape {arr.shape}, which does not match the "
                f"Dataset's axes {(ax.m, ax.t, ax.z, ax.c, ax.y, ax.x)}.")
    prov = ds.image
    if arr is None and prov is None:
        raise ValueError("Export TIFF has nothing to write: no image on the input Dataset "
                         "and no `layer` naming a Voxel raster.")

    def plane(m: int, t: int, z: int, c: int) -> np.ndarray:
        if arr is not None:
            return arr[m, t, z, c]
        return prov.get_region(0, m, t, z, c, 0, ax.y, 0, ax.x)

    # The dtype and the on-disk element size come from a real plane rather than from the
    # provider's declared dtype: a provider is free to return float32 for an integer store
    # (a normalized chain does exactly that), and declaring uint16 there would silently
    # truncate every value on the way out.
    probe = np.asarray(plane(0, 0, 0, 0))
    dt = probe.dtype
    if dt == np.float64:
        # No TIFF reader in this lab's toolchain handles float64 usefully, and it doubles the
        # file for precision the camera never had. float32 is lossless for anything that came
        # from an integer sensor through float maths.
        dt = np.dtype(np.float32)

    # ── calibration, resolved EAGERLY (before any writer/iterator exists) ──────
    px = ctx.calib("pixel_size_um")
    zs = ctx.calib("z_step_um")
    dt_s = ctx.calib("dt_s")
    emis = ctx.calib("channel_emission_nm")
    bits = ctx.calib("bit_depth")
    origin = ctx.calib("origin_um")
    na = ctx.calib("objective_na")
    mag = ctx.calib("objective_magnification")
    ch_names = ctx.meta("channel_names")

    meta_per_m = [_ome_metadata(model, ax, m, px=px, zs=zs, dt_s=dt_s, emis=emis,
                                bits=bits, dtype=dt, origin=origin, na=na, mag=mag,
                                ch_names=ch_names, layer=layer, split=split)
                  for m in range(ax.m)]

    # ── target files ──────────────────────────────────────────────────────────
    if split == "group":
        plan = position_group_plan(ds.metadata, ax)
        if not plan.placed:
            raise ValueError(
                f"Export TIFF: Split = 'group' needs to know which positions belong "
                f"together, and this Dataset carries no usable field geometry for its "
                f"{ax.m} positions (it needs `pixel_size_um` plus a per-position "
                f"`origin_um` or `stage_xy_um`). Use Split = 'position' for one file each, "
                f"or 'none' for one file holding all of them.")
        targets = [(_group_path(path, g.key), list(g.members)) for g in plan.groups]
    elif split == "position" or (model != "ome" and ax.m > 1):
        if split == "none":
            raise ValueError(
                f"the {model!r} metadata model cannot hold {ax.m} positions in one file — "
                f"only OME-TIFF has multiple images per file. Set Split to 'position' (or "
                f"'group') to write more than one file, or switch Model to 'ome'.")
        targets = [(_split_path(path, m, ax.m), [m]) for m in range(ax.m)]
    else:
        targets = [(path, list(range(ax.m)))]

    plane_bytes = int(ax.y) * int(ax.x) * int(dt.itemsize)
    per_file = {p: len(ms) * ax.t * ax.z * ax.c for p, ms in targets}

    stamps = {p: _stamp(ds, arr, layer, ms, path=p, model=model, comp=comp_name,
                        level=level if takes_level else None,
                        meta=[meta_per_m[m] for m in ms], dtype=dt)
              for p, ms in targets}

    if existing == "refuse":
        clash = [p for p, _ in targets if os.path.exists(p)]
        if clash:
            raise ValueError(
                "Export TIFF refuses to overwrite an existing file (Existing = 'refuse'): "
                + ", ".join(clash[:3]) + (" …" if len(clash) > 3 else "")
                + ". Change the path, or set Existing to 'overwrite'/'skip'.")
    if existing == "skip" and all(_read_stamp(p) == stamps[p] for p, _ in targets):
        ctx.progress(1, 1, f"already exported: {os.path.basename(path)} is byte-for-byte "
                           f"this export (stamp matches) — nothing rewritten")
        return ds

    # ── write ─────────────────────────────────────────────────────────────────
    total = sum(per_file.values())
    done = 0
    kw = _MODEL_KW[model]
    written: List[str] = []
    for target, ms in targets:
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        big = (per_file[target] * plane_bytes) > _BIGTIFF_BYTES
        if big and model == "imagej":
            raise ValueError(
                f"an ImageJ hyperstack cannot exceed 4 GB and this would be "
                f"~{per_file[target] * plane_bytes / 1e9:.1f} GB. Switch Model to 'ome' "
                f"(BigTIFF, and Fiji reads it through Bio-Formats), or set Split to "
                f"'position'/'group', or crop first.")
        opts: Dict[str, Any] = {"bigtiff": big} if kw != "imagej" else {}
        if kw:
            opts[kw] = True
        part = target + ".part"
        try:
            with tifffile.TiffWriter(part, **opts) as tw:
                for m in ms:
                    n = ax.t * ax.z * ax.c

                    def planes(_m: int = m) -> Any:
                        # C-fastest order, which is what axes="TZCYX" declares.
                        for t, z, c in itertools.product(range(ax.t), range(ax.z),
                                                         range(ax.c)):
                            p = np.asarray(plane(_m, t, z, c))
                            yield p if p.dtype == dt else p.astype(dt, copy=False)

                    # Progress is ticked from OUTSIDE the generator: tifffile pulls it
                    # through a thread pool when maxworkers > 1, so a report from inside
                    # would arrive out of order and from the wrong thread.
                    tw.write(
                        _counting(planes(), ctx, done, total, ax.t,
                                  f"writing {os.path.basename(target)}"),
                        shape=(ax.t, ax.z, ax.c, ax.y, ax.x), dtype=dt,
                        photometric="minisblack",
                        compression=codec,
                        compressionargs=({"level": level} if (codec and takes_level)
                                         else None),
                        maxworkers=(os.cpu_count() or 1) if codec else 1,
                        resolution=((1.0 / px, 1.0 / px) if px else None),
                        resolutionunit=("MICROMETER" if px else None),
                        extratags=[(STAMP_TAG, "s", 0, stamps[target], True)],
                        metadata=(meta_per_m[m] if kw else None))
                    done += n
            os.replace(part, target)
            written.append(target)
        finally:
            if os.path.exists(part):
                os.remove(part)          # an interrupted write never keeps the real name

    ctx.progress(total, total,
                 f"exported {len(written)} file(s), {sum(per_file.values())} planes: "
                 f"{os.path.basename(written[0]) if written else path}")
    return ds


def _counting(it: Any, ctx: EvalContext, base: int, total: int, frames: int, note: str):
    """Wrap a plane iterator so each pull ticks the progress rail.

    Yields exactly what it is given. The count is the honest one — a plane reported here has
    been READ and handed to the encoder, which is the work this node owns; the encoder's own
    latency lands a moment later and is bounded by tifffile's buffer.
    """
    n = base
    for item in it:
        n += 1
        ctx.progress(n, total, note, frames=max(1, int(frames)))
        yield item


def _split_path(path: str, m: int, n: int) -> str:
    """``out.ome.tif`` → ``out_m003.ome.tif`` — the per-position filename.

    The index is zero-padded to the width of the largest one so the files sort in position
    order in every file browser and every glob, which is what makes a 49-position export
    usable. Compound extensions (``.ome.tif``, ``.ome.tiff``) are kept whole; splitting on
    the last dot alone would produce ``out.ome_m003.tif`` and cost the OME suffix its
    meaning.
    """
    lower = path.lower()
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if lower.endswith(suffix):
            stem, ext = path[: -len(suffix)], path[-len(suffix):]
            break
    else:
        stem, ext = path, ""
    return f"{stem}_m{m:0{len(str(max(0, n - 1)))}d}{ext}"


def _group_path(path: str, key: str) -> str:
    """``out.ome.tif`` → ``out_G3.ome.tif`` — the per-GROUP filename.

    The group's own key rather than an index, because the key is what the user typed into
    ``util.select_group`` and what a renamed group in a ``.groups.json`` sidecar is called:
    an export named ``_g02`` while everything else calls it ``treated`` would be one more
    thing to map by hand. Sanitised to what every filesystem accepts, since a sidecar key is
    free text.
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(key)).strip("_")
    lower = path.lower()
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if lower.endswith(suffix):
            stem, ext = path[: -len(suffix)], path[-len(suffix):]
            break
    else:
        stem, ext = path, ""
    return f"{stem}_{safe or 'group'}{ext}"


def _stamp(ds: Dataset, arr: Optional[np.ndarray], layer: str, ms: List[int], *,
           path: str, model: str, comp: str, level: Optional[int],
           meta: List[dict], dtype: Any) -> str:
    """A digest of everything that determines this file's bytes.

    This is what makes ``existing="skip"`` safe rather than a way to ship a stale export.
    The engine gives a node no access to its own recipe hash, and the filesystem cannot be
    folded into the memo's read set (``_reads_valid`` re-validates ENVELOPE keys only), so
    "is the file on disk already the file I am about to write?" has to be answered by the
    node, from the same inputs the write consumes:

    * the pixel source's identity — the provider's ``fingerprint()``, which already folds
      content and version (C5), or the layer array's bytes when a raster is being exported;
    * which positions land in this file, and the full metadata block for each;
    * the container and codec choices, which change the bytes without changing the pixels.

    A mismatch on any of those rewrites. So the sequence the user asked for holds — a second
    pull of an unchanged graph does no I/O, deleting the file brings it back, and changing
    anything upstream produces a new file rather than leaving the old one in place.
    """
    if arr is not None:
        src: Any = ("layer", layer, value_digest(arr.tobytes()), arr.shape, str(arr.dtype))
    else:
        src = ("image", ds.image.fingerprint())
    return value_digest((src, tuple(ms), os.path.basename(path), model, comp, level,
                         str(np.dtype(dtype)), meta))


def _read_stamp(path: str) -> str:
    """The export stamp in ``path``'s first IFD, or ``""`` (absent, unreadable, not a TIFF).

    Header-only: opening a TIFF reads its IFDs, never its pixel data, so this costs a couple
    of seeks even for a 100 GB file. Every failure answers ``""`` — the safe direction, since
    an unreadable stamp means "cannot prove this is already the right file" and the export
    proceeds.
    """
    if not os.path.exists(path):
        return ""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            tag = tf.pages[0].tags.get(STAMP_TAG)
            return str(tag.value) if tag is not None else ""
    except Exception:      # noqa: BLE001 — any failure means "unproven", i.e. rewrite
        return ""


def _ome_metadata(model: str, ax: Any, m: int, *, px, zs, dt_s, emis, bits, dtype,
                  origin, na, mag, ch_names, layer: str, split: bool) -> dict:
    """The metadata block for one position, transcribed from the envelope.

    **Every value here is the envelope's own, unrounded and un-re-derived.** That is the
    point of the node: `wire-node-v2` §7c says calibration describes the data on THIS wire
    rather than the acquisition, so what gets written is what a crop, a resample or a
    z-project left behind — which is what makes the exported file correctly calibrated
    instead of merely stamped with the source file's numbers. A key whose envelope value is
    absent is **omitted**, never defaulted: no ``PhysicalSizeZ`` on a single plane, no
    ``TimeIncrement`` on a file whose timestamps the SDK never filled in. An absent field
    reads as "unknown" to every OME consumer; a fabricated 1.0 reads as a measurement.

    ``EmissionWavelength`` is written only when EVERY channel has one, because OME's
    ``Channel`` fields are positional lists and a transmitted-light channel's ``None`` would
    either shift every subsequent wavelength onto the wrong channel or have to be invented.

    ``objective_na`` / ``objective_magnification`` have no home in the ``Image`` block
    tifffile builds (they belong to OME's ``Instrument``, which its writer does not emit), so
    rather than drop them they go into the ``Description`` free-text field alongside the
    provenance line. Present and readable beats structured and absent.
    """
    if model == "imagej":
        # ImageJ's own vocabulary, which is what makes Fiji open the stack calibrated
        # without going through Bio-Formats. `spacing`/`unit` are the Z calibration,
        # `finterval` the frame interval; XY comes from the TIFF resolution tags.
        md: Dict[str, Any] = {"axes": "TZCYX"}
        if zs:
            md["spacing"] = float(zs)
        if px or zs:
            md["unit"] = "um"
        if dt_s:
            md["finterval"] = float(dt_s)
        if ax.c > 1:
            md["mode"] = "composite"
        return md

    md = {"axes": "TZCYX"}
    if model != "ome":
        return md          # `plain`: axes only; calibration rides the resolution tags

    if px:
        md.update(PhysicalSizeX=float(px), PhysicalSizeXUnit="µm",
                  PhysicalSizeY=float(px), PhysicalSizeYUnit="µm")
    if zs:
        md.update(PhysicalSizeZ=float(zs), PhysicalSizeZUnit="µm")
    if dt_s:
        md.update(TimeIncrement=float(dt_s), TimeIncrementUnit="s")
    # SignificantBits is the SENSOR's depth, not the container's — a 12-bit ND2 stored in
    # uint16 must say 12, or every reader's auto-contrast stretches against 65535 and the
    # image looks black. Integer data only: it means nothing for a float payload, which by
    # then has no declared integer scale anyway (§7c).
    if bits and np.issubdtype(np.dtype(dtype), np.integer):
        md["SignificantBits"] = int(bits)

    names = list(ch_names or [])
    chan: Dict[str, Any] = {}
    if any(names[:ax.c]):
        chan["Name"] = [str(names[c]) if c < len(names) and names[c] else f"Ch{c + 1}"
                        for c in range(ax.c)]
    ems = list(emis or [])
    if len(ems) >= ax.c and all(e is not None for e in ems[:ax.c]):
        chan["EmissionWavelength"] = [float(ems[c]) for c in range(ax.c)]
        chan["EmissionWavelengthUnit"] = ["nm"] * ax.c
    if chan:
        md["Channel"] = chan

    n = ax.t * ax.z * ax.c
    plane_md: Dict[str, Any] = {}
    org = _origin_of(origin, m)
    if org is not None:
        oz, oy, ox = org
        # Per-plane stage position in the microscope's own frame. X/Y are the field corner
        # (constant down a stack); Z advances with the slice, which is the one component that
        # genuinely differs plane to plane and the reason this is a Plane field rather than
        # an Image one.
        plane_md["PositionX"] = [float(ox)] * n
        plane_md["PositionY"] = [float(oy)] * n
        plane_md["PositionZ"] = [float(oz) + (float(zs) * z if zs else 0.0)
                                 for _t, z, _c in itertools.product(
                                     range(ax.t), range(ax.z), range(ax.c))]
        for k in ("PositionXUnit", "PositionYUnit", "PositionZUnit"):
            plane_md[k] = ["µm"] * n
    if dt_s:
        plane_md["DeltaT"] = [float(dt_s) * t for t, _z, _c in itertools.product(
            range(ax.t), range(ax.z), range(ax.c))]
        plane_md["DeltaTUnit"] = ["s"] * n
    if plane_md:
        md["Plane"] = plane_md

    notes = ["written by ND2StudiosBlender io.write_tiff"]
    if layer:
        notes.append(f"content: Voxel layer {layer!r} (not the image)")
    if na:
        notes.append(f"objective NA {float(na):g}")
    if mag:
        notes.append(f"objective magnification {float(mag):g}x")
    md["Description"] = " · ".join(notes)
    if ax.m > 1 and not split:
        md["Name"] = f"position {m}"
    return md


def _origin_of(origin: Any, m: int) -> Optional[Tuple[float, float, float]]:
    """``origin_um[m]`` as ``(z, y, x)``, or ``None``.

    ``origin_um`` is per-multipoint and only ever seeded when the stage logs cover EVERY
    position (`dataset.CALIBRATION_KEYS`), so a short list is not something to index into —
    it would hand position *m* some other field's corner and look exactly like a complete
    answer. Refuse to guess; the Plane block is simply omitted.
    """
    try:
        row = list(origin)[m]
        z, y, x = (float(v) for v in row)
        return (z, y, x)
    except Exception:      # noqa: BLE001 — absent, short, or not a triple list
        return None


register_node(
    _compute_write_tiff, op_key="io.write_tiff", label="Export TIFF", category="io",
    inputs=[
        InDataset("data"),
        InString("path", "File", field=False, default="",
                 path_kind="save_file",
                 path_filter="OME-TIFF (*.ome.tif);;TIFF (*.tif *.tiff);;All files (*)",
                 path_hint="Browse… to choose where this writes",
                 description=
                 "Where the file is written, on the machine that runs the graph. There is no "
                 "default and an empty value is refused rather than guessed — this node is "
                 "the one thing in the graph that puts bytes somewhere permanent, and it "
                 "must be somewhere you picked. With Split on, this is the TEMPLATE: "
                 "position 3 of 49 becomes `<name>_m03.<ext>` (the index padded so the "
                 "files sort in position order), and group G3 becomes `<name>_G3.<ext>`. "
                 "A `.ome.tif` suffix is kept whole when the suffix is inserted."),
        InString("layer", "Layer", field=False, default="", layer_in=Domain.VOXEL,
                 description=
                 "Export a Voxel raster — a mask, a label image, a distance field — instead "
                 "of the image. Leave it EMPTY (the default) to write the image, which is "
                 "what you want most of the time. Set it to a layer name and the file "
                 "contains that raster's own values at the same geometry and calibration, "
                 "which is how a segmentation gets into Fiji or QuPath as a label image. "
                 "The layer's own dtype is written, so integer label ids survive as ids "
                 "rather than being rescaled."),
        InInt("level", "Level", unit="", field=False, default=1,
              available_in={"compression": frozenset({"zlib", "zstd"})},
              description=
              "How hard the codec works, 1 (fastest) to 9. The default 1 is deliberate: on "
              "microscope data almost all of the size win is in the first level, and 9 can "
              "cost several times the write time for a few percent. Raise it only when the "
              "file is going somewhere slow — an archive, a share — and the write is not the "
              "thing you are waiting on. Only read for the codecs that accept a level; lzma "
              "has no level in tifffile's encoder, so this field is hidden there."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("split", ["none", "position", "group"], default="none", label="Split",
             description=
             "Whether this writes ONE file or several. A multipoint export is the only "
             "place the answer is not obvious: the positions can be one experiment or "
             "several specimens, and only you know which.",
             choice_docs={
                 "none":
                     "One file holding every position — a single OME-TIFF with one OME "
                     "Image per multipoint. Tidy, and the form a viewer reopens as one "
                     "experiment. Refused by the 'imagej' and 'plain' models when there is "
                     "more than one position, because neither can hold them.",
                 "position":
                     "One file per multipoint, named `<name>_m03.<ext>` with the index "
                     "zero-padded so the files sort in position order. What you want when "
                     "the positions are handled separately downstream, or when a single "
                     "file would be uncomfortably large. On a 54-position series this "
                     "writes 54 files.",
                 "group":
                     "One file per position GROUP — the 3x3 mosaic, the plate well — named "
                     "`<name>_G3.<ext>` after the group's own key, so a group renamed in a "
                     "`.groups.json` sidecar exports under that name. This is the one that "
                     "matches how a multi-specimen acquisition is actually analysed: a "
                     "54-position file of six mosaics writes six files of nine positions, "
                     "each reopening as one specimen. Needs the stage geometry, and refuses "
                     "rather than guessing when it is missing.",
             }),
        Mode("model", ["ome", "imagej", "plain"], default="ome", label="Model",
             description=
             "Which metadata container the calibration is written into — i.e. which reader "
             "is guaranteed to open the file correctly calibrated. The PIXELS are identical "
             "in all three; what differs is how much of the envelope survives the trip.",
             choice_docs={
                 "ome":
                     "OME-XML in the first IFD: physical XY and Z sizes, frame interval, "
                     "significant bits, per-channel names and emission wavelengths, and a "
                     "per-plane stage position and time offset. The only option that can "
                     "hold several positions in one file, and the only one that carries "
                     "where the data was on the stage. Read by Fiji (Bio-Formats), QuPath "
                     "and napari; BigTIFF when needed, so there is no size ceiling. The "
                     "default, and the one to pick if you are unsure.",
                 "imagej":
                     "An ImageJ hyperstack header — `spacing`, `unit`, `finterval` plus the "
                     "resolution tags. Fiji opens it natively, instantly, with the Z spacing "
                     "and frame interval already right and no Bio-Formats import dialog. In "
                     "exchange it holds ONE position, drops stage positions and channel "
                     "wavelengths, and cannot exceed 4 GB (a larger export is refused, not "
                     "silently truncated). Pick it when the destination is Fiji and the "
                     "stack is modest.",
                 "plain":
                     "A bare TIFF stack: pixels, the XY resolution tags, and nothing else. "
                     "Nothing to misinterpret, so it is the safe answer for a reader that "
                     "chokes on OME-XML or an ImageJ header, and for handing an array to "
                     "code that just wants planes. Z spacing, frame interval, channel "
                     "identity and stage position are all LOST — do not pick it for data "
                     "anyone will need to measure in physical units.",
             }),
        Mode("compression", ["none", "zlib", "lzma", "zstd"], default="zlib",
             label="Compression",
             description=
             "How the planes are packed. All four are LOSSLESS — the pixels read back "
             "bit-for-bit — so this trades write time against file size and nothing else. "
             "Encoding runs across all cores behind the streaming write, so the cost is "
             "well under what the ratios suggest.",
             choice_docs={
                 "none":
                     "Store the planes raw. The fastest possible write — measured on this "
                     "machine at ~1.4 GB/s, i.e. disk bandwidth — and the largest file, and "
                     "the one every reader on earth opens. Pick it when the export is a "
                     "scratch hand-off you will delete, or when the destination is a fast "
                     "local disk with room to spare.",
                 "zlib":
                     "Deflate, the most widely supported compressed TIFF there is — always "
                     "available here, no extra package. Measured at ~230 MB/s on this "
                     "machine at Level 1, about a sixth of the raw write speed; how much "
                     "SMALLER the file gets depends entirely on the data, so watch the first "
                     "export rather than trusting a number. The default: it is the best "
                     "size-per-risk on offer, since a file no reader can open is worth "
                     "nothing.",
                 "lzma":
                     "Usually the smallest of the three that work here, and by far the "
                     "slowest — measured at ~33 MB/s, roughly 40x slower than an "
                     "uncompressed write, and slow to READ again every time. For an archive "
                     "you write once and rarely open. Has no level setting in tifffile's "
                     "encoder, so the Level field is hidden.",
                 "zstd":
                     "The codec that would be the right default — deflate's ratio at close "
                     "to raw write speed. **Not usable in this installation**: tifffile "
                     "needs either `imagecodecs` (not installed) or Python 3.14's built-in "
                     "zstd, and this is Python 3.12. Selecting it is refused up front with "
                     "that explanation rather than failing partway through a large export; "
                     "`pip install imagecodecs` makes it work.",
             }),
        Mode("existing", ["skip", "overwrite", "refuse"], default="skip", label="Existing",
             description=
             "What to do when a file is already at the target path. The node writes a "
             "private stamp into every file it produces — a digest of the pixel source, the "
             "positions and the full metadata block — so this can distinguish 'the same "
             "export already ran' from 'a different file happens to have this name', which "
             "is the difference between skipping safely and shipping a stale result.",
             choice_docs={
                 "skip":
                     "Rewrite unless the stamp proves the existing file is ALREADY this "
                     "exact export. So a re-run costs no I/O, deleting the file brings it "
                     "back, and changing anything upstream still produces a fresh file — a "
                     "stale export is not reachable. The default. Note that an unchanged "
                     "graph re-pulled in one session never reaches this node at all: the "
                     "memo answers first, which is cheaper still.",
                 "overwrite":
                     "Always rewrite, stamp or no stamp. Costs a full re-encode on every "
                     "run that reaches the node, and is the option to pick when something "
                     "OUTSIDE the graph may have touched the file — an editor, a sync "
                     "client, another tool — because the stamp only proves what this node "
                     "last wrote, not what the file contains now.",
                 "refuse":
                     "Raise rather than touch an existing file, even one this node wrote. "
                     "For a destination that must be written exactly once — a published "
                     "result, a shared drive — where an accidental overwrite is worse than "
                     "a failed run. You then move or rename the old file by hand.",
             }),
    ],
    # WHOLE_PLANE, not WHOLE_SERIES: the compute reads exactly one (Y,X) plane at a time and
    # streams it into an open writer, so the scheduler should route it down the per-plane
    # provider path. Declaring WHOLE_SERIES would ask for the whole 6-D array it deliberately
    # never builds.
    granularity=Granularity.WHOLE_PLANE,
    kernel_axes=frozenset({"y", "x"}),
    reads_domains=frozenset({Domain.VOXEL}),
    description="Write the Dataset on this wire to a TIFF, streaming one plane at a time so "
                "peak memory is a single plane no matter how large the series. The "
                "calibration envelope is transcribed verbatim into OME-XML — physical XY/Z "
                "sizes, frame interval, significant bits, channel names and emission "
                "wavelengths, per-plane stage position and time offset — so the exported "
                "file describes the pixels it contains rather than the file they came from. "
                "Hands the Dataset through unchanged, so it can sit mid-chain.")
