"""On-disk **Dataset checkpoint** — the persistence layer behind the Dock node (V2.18).

A checkpoint is a whole :class:`~nodegraph.dataset.Dataset` written to a directory and
re-openable as a *lazy* one: the image raster becomes a planar-block ``.b2nd`` store (the
same format ND2 ingest writes, so it re-opens with per-tile decompression), full-size
Voxel attribute layers become **memory-mapped** ``.npy`` files, and everything small — the
coarse lattice attributes and the Label/Point/Track/Mesh structure columns — lands in one
``.npz``. Axes, calibration and the layer catalog ride in ``manifest.json``.

Its reason to exist: on a large series a long node chain is expensive twice over — every
realized intermediate is retained (a Voxel mask is a full ``(M,T,Z,C,Y,X)`` array in RAM,
not a lazy provider), and every memo eviction re-runs the whole upstream chain to get a
plane back. Checkpointing cuts the chain: the downstream half reads *decompressed pixels
off disk* instead of recomputing twenty nodes, and the upstream half can be dropped from
memory entirely. See ``io.dock`` in :mod:`nodelab_v2.ops` for the node that drives this.

**Layout** (the manifest is written LAST — its presence is the completeness marker, so an
interrupted bake leaves a directory :func:`read_manifest` correctly reports as unusable,
rather than a torn store that dead-ends every later open)::

    <dir>/manifest.json          version · axes · metadata · layer index · bake id
    <dir>/image/level_*.b2nd     the raster, planar-block, one file per pyramid level
    <dir>/voxel/NNN_*.npy        one file per Voxel attribute layer (memmapped on open)
    <dir>/tables.npz             every non-Voxel attribute layer, keyed by manifest index

**Writing is streamed plane-by-plane**, never through a realized 6-D array: peak memory is
one z-slab per pyramid level (bounded by :data:`~nodegraph.provider._CHUNK_TARGET_BYTES`),
so a series far larger than RAM checkpoints fine. That is the whole point — a bake that
had to materialize the series first would OOM on exactly the files this feature exists for.

Qt-free; numpy + stdlib, with ``blosc2`` lazily imported (same as the provider).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.dataset import AttributeLayer, AxisSizes, Dataset
from nodegraph.domains import AXIS_ORDER, Domain
from nodegraph.metadata import MetaEnvelope
from nodegraph.provider import B2ndProvider

#: bumped when the on-disk layout changes incompatibly; :func:`read_manifest` refuses a
#: checkpoint from a newer writer rather than mis-reading it.
CHECKPOINT_VERSION = 1

MANIFEST_NAME = "manifest.json"
IMAGE_DIR = "image"
VOXEL_DIR = "voxel"
TABLES_NAME = "tables.npz"

#: The precisions a checkpoint may be written at. There is deliberately **no default** —
#: the Dock node makes the user pick, because the honest answer depends on what the chain
#: upstream produced (see :func:`target_dtype`).
PRECISIONS: Tuple[str, ...] = ("float32", "float64", "uint16")

#: ``progress(fraction_0_to_1, note)`` — the same shape ND2 ingest reports, so a caller
#: drives one determinate bar across the whole bake.
ProgressFn = Callable[[float, str], None]

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ── small helpers ─────────────────────────────────────────────────────────────

def _slug(s: Optional[str]) -> str:
    return _SLUG_RE.sub("_", str(s or ""))[:48]


def _json_safe(o: Any) -> Any:
    """``o`` reduced to JSON-encodable types. numpy scalars/arrays become Python
    values; anything else that will not encode is dropped rather than failing the bake —
    a stray unserializable metadata value must not cost the user a multi-minute write.
    Keys are coerced to ``str`` (a JSON object has no other kind)."""
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            try:
                out[str(k)] = _json_safe(v)
            except Exception:  # noqa: BLE001 — a bad value must not fail the bake
                continue
        return out
    try:
        json.dumps(o)
        return o
    except Exception:  # noqa: BLE001
        return None


def target_dtype(src: np.dtype, precision: str) -> np.dtype:
    """The stored dtype for source dtype ``src`` at ``precision``.

    **Integer and boolean data always pass through untouched** — a label raster, a mask or
    a raw uint16 camera frame already *is* its own exact representation, and re-typing one
    would either waste space (a bool mask as float64) or corrupt ids. The precision choice
    only ever applies to floating-point data, which is what a filter chain produces.
    """
    src = np.dtype(src)
    if src.kind not in "fc":                     # integer / bool / anything non-float
        return src
    if precision == "float32":
        return np.dtype(np.float32)
    if precision == "uint16":
        return np.dtype(np.uint16)
    return np.dtype(np.float64)


def _cast(a: np.ndarray, dt: np.dtype) -> np.ndarray:
    """``a`` as ``dt``. A float→uint16 conversion **rounds and clips** into ``[0, 65535]``
    rather than rescaling: rescaling would need the series' global range (a whole extra
    pass over a lazy chain) and would silently re-mean every number a downstream table
    reports. :func:`_guard_uint16_range` refuses the case where clipping would destroy
    the data, so this stays predictable rather than merely convenient."""
    a = np.asarray(a)
    if a.dtype == dt:
        return a
    if dt.kind in "ui" and a.dtype.kind == "f":
        info = np.iinfo(dt)
        return np.clip(np.rint(a), info.min, info.max).astype(dt)
    return a.astype(dt)


def _guard_uint16_range(probe: np.ndarray, precision: str) -> None:
    """Refuse a ``uint16`` bake of data that clipping would destroy, checked on the FIRST
    plane read so the user finds out in a second rather than after a long write.

    The case that matters is normalized ``[0, 1]`` float data (anything after
    ``enhance.normalize``, and most ``[0,1]`` skimage outputs): rounding it to integers
    collapses every pixel to 0 or 1. Negative data (a difference-of-gaussians, a
    background-subtracted frame) loses its entire negative half the same way."""
    if precision != "uint16" or probe.dtype.kind != "f":
        return
    finite = probe[np.isfinite(probe)]
    if finite.size == 0:
        return
    hi, lo = float(finite.max()), float(finite.min())
    if hi <= 1.5:
        raise ValueError(
            f"uint16 precision would destroy this data: the first plane's values top out "
            f"at {hi:.4g}, so rounding to integers collapses the image to 0/1. This is "
            f"normalized or [0,1] float data — bake it at float32 instead, or scale it "
            f"back into camera counts upstream.")
    if lo < -0.5:
        raise ValueError(
            f"uint16 precision would clip away this data's negative half (the first "
            f"plane reaches {lo:.4g}). Bake at float32 instead, or add an offset "
            f"upstream so the values are non-negative.")


def _downsample_2x(a: np.ndarray) -> np.ndarray:
    """Mean-pool the trailing ``(Y, X)`` of a ``(…, Y, X)`` slab by 2×, via
    :func:`nodegraph.provider._plane_mean_2x` — **the** pyramid arithmetic, in one place.

    It used to re-spell that expression here, which is precisely the drift
    ``provider._plane_mean_2x``'s docstring claims cannot happen ("Every pyramid path routes
    its per-plane reduction through this function so … :mod:`nodegraph.checkpoint` cannot
    drift apart"). It could, because nothing enforced it and this module never called it.

    Kept as a thin wrapper rather than deleted: it is the reference the selftest compares
    the streamed pyramid against, and the checkpoint writer no longer calls it at all — the
    coarse levels are streamed out of the finished ``level_0`` by
    :meth:`nodegraph.provider.B2ndProvider._append_level`, which is pooled and bounded.
    """
    from nodegraph.provider import _plane_mean_2x
    y, x = a.shape[-2], a.shape[-1]
    if y < 2 or x < 2:
        return a
    return _plane_mean_2x(a)


# ── the layer index (what goes in the manifest) ───────────────────────────────

def _user_layer_name(attr: AttributeLayer) -> str:
    """The user-facing layer name for the picker catalog. The name lives in a **different
    slot per family** (the ``LayerKey`` trap, `wire-node-v2` §4c): ``with_layer`` files a
    lattice layer as ``(VOXEL, None, "mask")`` — name in slot 2 — while ``with_structure``
    files each COLUMN as ``(POINT, "spots", "y")``, where the layer name is slot 1."""
    return attr.name if attr.layer is None else attr.layer


def derived_layer_names(ds: Dataset) -> Tuple[Tuple[Domain, str], ...]:
    """The ``(domain, user-facing name)`` catalog of ``ds``, first-appearance order —
    the fallback when a caller does not hand over the source envelope's own catalog."""
    out: List[Tuple[Domain, str]] = []
    for attr in ds.attributes.values():
        pair = (attr.domain, _user_layer_name(attr))
        if pair not in out:
            out.append(pair)
    return tuple(out)


def derived_domains(ds: Dataset) -> frozenset:
    """The domain set present on ``ds`` — every domain carrying an attribute, plus
    ``VOXEL`` when it has an image (the image *is* the Voxel domain)."""
    doms = {a.domain for a in ds.attributes.values()}
    if ds.image is not None:
        doms.add(Domain.VOXEL)
    return frozenset(doms)


# ── write ─────────────────────────────────────────────────────────────────────

def write_checkpoint(ds: Dataset, dirpath: str, *, precision: str,
                     bake_id: str = "", levels: int = 3,
                     domains: Optional[frozenset] = None,
                     layer_names: Optional[Sequence[Tuple[Domain, str]]] = None,
                     progress: Optional[ProgressFn] = None,
                     should_cancel: Optional[Callable[[], bool]] = None
                     ) -> Optional[Dict[str, Any]]:
    """Write ``ds`` to the checkpoint directory ``dirpath`` and return its manifest.

    ``precision`` (one of :data:`PRECISIONS`) applies to **floating-point** data only —
    integer rasters, label ids and boolean masks always store as themselves
    (:func:`target_dtype`). ``bake_id`` is an opaque stamp the caller folds into the
    node's params so the memo can tell two bakes of the same path apart.

    ``domains``/``layer_names`` let the caller hand over the *source envelope's* own
    catalog (what ``propagate_meta`` computed for the edge being checkpointed), which is
    strictly better than re-deriving it from the payload; both fall back to a derivation
    when omitted.

    ``should_cancel()`` is polled at every block/layer boundary; when it returns true this
    returns **None** having written no manifest — which is the same state an interrupted bake
    already leaves, i.e. one :func:`read_manifest` correctly reports as unusable. A caller
    that gets ``None`` must not record a bake.

    Any pre-existing content at ``dirpath`` is overwritten. The manifest is written last,
    so an interrupted bake leaves a directory that reads as *absent*, never as valid."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    ax = ds.axes
    os.makedirs(dirpath, exist_ok=True)
    # A stale manifest must not survive a failed rewrite and describe the new store.
    try:
        os.remove(os.path.join(dirpath, MANIFEST_NAME))
    except OSError:
        pass

    voxel = [a for a in ds.attributes.values() if a.domain is Domain.VOXEL]
    small = [a for a in ds.attributes.values() if a.domain is not Domain.VOXEL]

    # Weight the bar by bytes so the fraction tracks real work rather than step count.
    img_bytes = int(ax.m) * ax.t * ax.z * ax.c * ax.y * ax.x if ds.image is not None else 0
    vox_bytes = sum(int(a.values.size) for a in voxel)
    total = max(1, img_bytes + vox_bytes)

    def report(done: float, note: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, done / total)), note)

    image_meta: Optional[Dict[str, Any]] = None
    if ds.image is not None:
        image_meta = _write_image(ds.image, ax, os.path.join(dirpath, IMAGE_DIR),
                                  precision=precision, levels=levels,
                                  report=report, total_units=img_bytes,
                                  should_cancel=should_cancel)
        if image_meta is None:
            return None                  # cancelled: no manifest, so this reads as absent

    voxel_index = _write_voxel_layers(voxel, os.path.join(dirpath, VOXEL_DIR),
                                      precision=precision, report=report,
                                      base=img_bytes, should_cancel=should_cancel)
    if voxel_index is None:
        return None
    table_index = _write_tables(small, os.path.join(dirpath, TABLES_NAME))
    if should_cancel is not None and should_cancel():
        return None

    md = dict(ds.metadata)
    if image_meta is not None and image_meta["restamped_bit_depth"]:
        # §7c: float data rounded into 16-bit counts IS a value-scale change, so the
        # depth downstream nodes read must describe the checkpoint, not the source.
        md["bit_depth"] = 16
    doms = derived_domains(ds) if domains is None else frozenset(domains)
    names = derived_layer_names(ds) if layer_names is None else tuple(layer_names)

    manifest = {
        "version": CHECKPOINT_VERSION,
        "bake_id": str(bake_id),
        "precision": precision,
        "axes": {a: int(getattr(ax, a)) for a in ("m", "t", "z", "c", "y", "x")},
        "metadata": _json_safe(md),
        "image": image_meta,
        "voxel_layers": voxel_index,
        "tables": table_index,
        "domains": sorted(d.value for d in doms),
        "layer_names": [[d.value, n] for d, n in names],
    }
    with open(os.path.join(dirpath, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if progress is not None:
        progress(1.0, "done")
    return manifest


#: Set ``NODEGRAPH_DOCK_COPY=0`` to force every bake through the full re-encode, bypassing
#: :func:`_copy_levels`. A kill-switch rather than an opt-in: the copy is guarded on exact
#: equality of everything that could differ, so the honest default is to take it — but a
#: comparison ("is the copy really producing the same store?") needs a way to turn it off, and
#: so does a bug report.
def _copy_enabled() -> bool:
    return (os.environ.get("NODEGRAPH_DOCK_COPY", "1").strip().lower()
            not in ("0", "false", "no", "off"))


def _copy_levels(prov: Any, ax: AxisSizes, imgdir: str, *, dt: np.dtype, tile: int,
                 report: Callable[[float, str], None], total_units: int,
                 want_levels: int = 1,
                 should_cancel: Optional[Callable[[], bool]] = None
                 ) -> Optional[Tuple[Any, int]]:
    """Copy the source store's pyramid files verbatim instead of decoding and re-encoding
    them — returning ``(deepest copied array, how many levels were copied)``, or ``None`` when
    any guard fails and the caller must do the real write.

    **Why this exists.** The cheapest bake a user can ask for is "freeze the load so I stop
    re-ingesting", and it was the one that paid full price: nothing tested what the input
    provider *was*, so a dock straight after ``io.load`` decompressed a store off disk and
    compressed it straight back, to produce bytes that were already there. Copying moves the
    COMPRESSED bytes and skips both codec passes, so it is bounded by the disk rather than by
    a core: on the 1.25 GiB fixture, 866 MiB of level 0 copies in well under a second against
    ~3.3 s to re-encode it, and the saving grows with the file.

    **Every guard is an exact-equality test, because "close enough" here means silently wrong
    pixels.** The path is taken only when the destination bytes would be *identical*:

    * the input is a **disk-backed** :class:`~nodegraph.provider.B2ndProvider` (an in-memory
      one has no file to copy, and every other provider computes its pixels);
    * its axes equal the payload's, so no node between the store and the dock reshaped,
      subset or re-addressed anything (those all wrap the provider in another class, but a
      metadata-only reshape would not, and the manifest's axes must describe the bytes);
    * ``target_dtype`` is a **no-op** at this precision — i.e. the data is integer/bool, which
      passes through untouched. A float source at ``float32``/``uint16`` is a real conversion
      and must go the long way;
    * ``tile`` matches, since it is the block geometry a reader decompresses by, and it is
      part of the provider's fingerprint;
    * the source's level 0 is **sound** — neither ``torn`` by its marker nor showing the
      blank-chunk tail of a pre-marker interrupted write. This is the guard that matters most:
      copying is the one path that would propagate a half-written source verbatim, and a tail
      of zero planes is exactly what a killed ingest leaves.

    **Every sound level is copied, not just level 0**, up to the depth this bake wants. The
    first cut copied level 0 only and rebuilt the pyramid, which left the rebuild dominating
    the whole operation — it decompresses all of level 0 again to produce coarse levels that
    were already sitting on disk beside it, and measured only 1.3× faster than a full
    re-encode. Copying them is sound for the same reason ``open`` can be trusted about them:
    it takes only the leading run that is complete-or-legacy and DROPS a torn level, so
    ``prov.levels`` already excludes anything it could not vouch for — and each one is put
    through the blank-chunk census here as well. Levels the source does not have are still
    built by :meth:`~nodegraph.provider.B2ndProvider._append_level` from the deepest one it
    did have.

    The copied coarse levels keep the SOURCE's chunk framing, which differs from what this
    writer would choose (an ingest store does not batch thin Z). That is a framing difference,
    not a pixel one — both sides derive from :func:`~nodegraph.provider._plane_mean_2x` — and
    the manifest reports the framing that is actually on disk rather than the one that was
    requested.
    """
    from nodegraph.provider import B2ndProvider
    if not _copy_enabled() or not isinstance(prov, B2ndProvider):
        return None
    src_dir = getattr(prov, "_urlpath", None)
    if not src_dir:
        return None                      # an in-memory store: nothing on disk to copy
    arrs = getattr(prov, "_arrays", None)
    if not arrs:
        return None
    src0 = arrs[0]
    if tuple(src0.shape) != (ax.m, ax.t, ax.z, ax.c, ax.y, ax.x):
        return None                      # the payload is not this store's own geometry
    if np.dtype(src0.dtype) != np.dtype(dt):
        return None                      # a real precision conversion
    if int(getattr(prov, "tile", -1)) != int(tile):
        return None                      # different read granularity
    # How many levels are BOTH sound at the source and wanted by this bake. `prov.levels`
    # already excludes anything `open` would not vouch for; the census catches the pre-marker
    # interrupted write that `open` trusts by age.
    n = min(int(getattr(prov, "levels", 1)), max(1, int(want_levels)))
    for lv in range(n):
        a = arrs[lv]
        if B2ndProvider._level_state(a) == "torn" or B2ndProvider._blank_tail(a) is not None:
            n = lv                       # this level and everything under it get rebuilt
            break
    if n < 1:
        return None                      # not even level 0 is trustworthy

    srcs = [os.path.join(src_dir, f"level_{lv}.b2nd") for lv in range(n)]
    dsts = [os.path.join(imgdir, f"level_{lv}.b2nd") for lv in range(n)]
    if any(not os.path.isfile(s) for s in srcs):
        return None
    if any(os.path.abspath(s) == os.path.abspath(d) for s, d in zip(srcs, dsts)):
        return None                      # baking a store onto itself
    total_bytes = max(1, sum(os.path.getsize(s) for s in srcs))

    # Copied in chunks rather than through `shutil.copyfile` so the bar moves and a stop is
    # honoured: this is one operation on files that can total hundreds of GB, and a
    # determinate bar that sits still for minutes reads as a hang.
    step = 32 << 20
    moved = 0
    try:
        for s, d in zip(srcs, dsts):
            with open(s, "rb") as fin, open(d, "wb") as fout:
                while True:
                    if should_cancel is not None and should_cancel():
                        raise InterruptedError
                    buf = fin.read(step)
                    if not buf:
                        break
                    fout.write(buf)
                    moved += len(buf)
                    report(int(total_units * (moved / total_bytes)),
                           f"copying the source store · "
                           f"{moved * 100 // total_bytes}%")
    except (InterruptedError, OSError):
        # A copy that cannot complete (cancelled, no space, a locked file) must not leave a
        # truncated level behind for `open` to adopt. Remove every file this attempt made and
        # fall through to the real write.
        for d in dsts:
            _unlink(d)
        return None
    import blosc2
    out = None
    for lv, d in enumerate(dsts):
        try:
            arr = blosc2.open(d, mode="a")
        except Exception:  # noqa: BLE001 — an unopenable copy is not a checkpoint
            for dd in dsts:
                _unlink(dd)
            return None
        # The source may pre-date the marker (`legacy`), so stamp every copy: from here on
        # these are levels THIS writer produced, and they must read as complete rather than
        # as trusted-by-age.
        B2ndProvider._mark(arr, lv, True)
        out = arr
    return (out, n)


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _write_image(prov: Any, ax: AxisSizes, imgdir: str, *, precision: str, levels: int,
                 report: Callable[[float, str], None], total_units: int,
                 should_cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Stream the raster into a planar-block ``.b2nd`` pyramid.

    Never realizes the 6-D volume: one ``(t-batch, z-slab)`` block is read from the
    (possibly lazy, possibly twenty-nodes-deep) provider and written to level 0. Peak
    memory is that block — bounded by :data:`~nodegraph.provider._CHUNK_TARGET_BYTES` —
    which is what lets a series larger than RAM be checkpointed at all.

    Four things changed here, and the first two are where the time was going on exactly the
    data a Dock gets pointed at (a Z-projection, a stitch, a merge — all ``z == 1``):

    1. **Geometry comes from** :meth:`~nodegraph.provider.B2ndProvider._store_kwargs`,
       per level, with ``batch_thin_z=True``. This module used to compute ``cz`` **once**
       from level-0 plane bytes and reuse it for every level, which both mis-framed the
       coarse levels (4× and 16× smaller chunks than the store the ingest writes) and, on a
       thin-Z series, pinned one 8 MiB plane per chunk. Measured: 143 → 552 MiB/s.
    2. **The pyramid is streamed out of the finished level 0** by
       :meth:`~nodegraph.provider.B2ndProvider._append_level` instead of being mean-pooled
       here per slab. That call is already pooled across planes, already chunk-aligned,
       already ``_mark``ed, and it routes through :func:`~nodegraph.provider._plane_mean_2x`
       — so the pyramid is bit-identical to an ingest store's *by construction* rather than
       by two copies of one expression agreeing. The old serial float64 mean-pool was ~22%
       of a bake on its own.
    3. **Every level is stamped** (:data:`~nodegraph.provider._LEVEL_META`) before its first
       byte and again when complete. Manifest-last already made a torn *checkpoint* read as
       absent, but it could not make a torn *level* read as torn — and
       :meth:`~nodegraph.provider.B2ndProvider.open` globs the leading run of level files
       rather than trusting the manifest's count, so a re-bake that produced fewer levels
       left a stale higher level to be adopted as trusted-``legacy``.
    4. **It is cancellable**, checked at the block boundary and always *before* the
       manifest — so a cancelled bake lands in the state this module already documents as
       correct for an interrupted one: readable as absent.

    Returns the image manifest entry; ``None`` when the write was cancelled."""
    import blosc2
    os.makedirs(imgdir, exist_ok=True)
    from nodegraph.parallel import fold_units
    from nodegraph.provider import B2ndProvider, pyramid_cparams

    def cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    tile = int(getattr(prov, "tile", 512) or 512)
    probe = np.asarray(prov.get_region(0, 0, 0, 0, 0, 0, min(ax.y, 64), 0, min(ax.x, 64)))
    _guard_uint16_range(probe, precision)
    dt = target_dtype(probe.dtype, precision)
    restamped = bool(probe.dtype.kind == "f" and dt.kind in "ui")
    cparams = pyramid_cparams()

    shape0 = (ax.m, ax.t, ax.z, ax.c, int(ax.y), int(ax.x))

    want = len(B2ndProvider._pyramid_shapes(shape0, max(1, int(levels))))

    # ── the fast path: the input IS a store already, so copy it ────────────────
    copied = _copy_levels(prov, ax, imgdir, dt=dt, tile=tile, report=report,
                          total_units=total_units, want_levels=want,
                          should_cancel=should_cancel)
    if copied is not None:
        last, n_levels = copied
        # The framing actually ON DISK, which for a copy is the source's, not the one this
        # writer would have chosen. Reporting the request instead would make the manifest
        # describe bytes that are not there.
        chunks = [int(v) for v in last.chunks] if n_levels == 1 else \
            [int(v) for v in blosc2.open(os.path.join(imgdir, "level_0.b2nd")).chunks]
        done = total_units
    else:
        if cancelled():
            return None
        kw0 = B2ndProvider._store_kwargs(shape0, dt.itemsize, tile=tile, cparams=cparams,
                                         urlpath=imgdir, level=0, batch_thin_z=True)
        chunks = [int(v) for v in kw0["chunks"]]
        last = _write_level0(prov, ax, imgdir, dt=dt, kw0=kw0, report=report,
                             total_units=total_units, cancelled=cancelled,
                             fold_units=fold_units, B2ndProvider=B2ndProvider,
                             blosc2=blosc2)
        if last is None:
            return None
        n_levels, done = 1, total_units

    # ── whatever the copy did not supply, built from the deepest level we have ──
    for lv in range(n_levels, want):
        if cancelled():
            return None
        report(done, f"baking pyramid · level {lv}")
        nxt = B2ndProvider._append_level(last, lv, tile=tile, cparams=cparams,
                                         urlpath=imgdir, batch_thin_z=True)
        if nxt is None:
            break                       # cannot halve further — stop adding levels
        last = nxt
        n_levels += 1
    return {"levels": n_levels, "dtype": str(dt), "tile": tile,
            "restamped_bit_depth": restamped, "chunks": chunks,
            # How many levels came from the source verbatim: 0 for a real write, and the
            # difference between this and `levels` is what was rebuilt. Recorded because it is
            # the one thing about a bake that a reader cannot infer from the bytes, and it is
            # what a "why is this store framed oddly?" question resolves to.
            "copied_levels": (copied[1] if copied is not None else 0),
            "copied_level0": copied is not None}


def _write_level0(prov: Any, ax: AxisSizes, imgdir: str, *, dt: np.dtype, kw0: dict,
                  report: Callable[[float, str], None], total_units: int,
                  cancelled: Callable[[], bool], fold_units: Any, B2ndProvider: Any,
                  blosc2: Any) -> Optional[Any]:
    """Stream level 0 out of ``prov``, block by block. ``None`` when cancelled.

    Split out of :func:`_write_image` when the copy fast path arrived, so the two ways to
    produce level 0 read as the alternatives they are instead of one of them being buried in
    an ``else``."""
    ct, cz = int(kw0["chunks"][1]), int(kw0["chunks"][2])
    shape0 = (ax.m, ax.t, ax.z, ax.c, int(ax.y), int(ax.x))
    arr0 = blosc2.empty(shape0, dtype=dt, **kw0)
    B2ndProvider._mark(arr0, 0, False)

    # One reusable destination buffer per block, allocated once rather than per iteration.
    # `fold_units` reads the planes on the pool and this thread writes them into the block
    # in submission order, so the buffer is only ever touched by one thread at a time — the
    # canonical eager shape (`parallel.fold_units`), with a small `batch` because a unit
    # here is a whole plane and in-flight results are what bound peak memory.
    done = 0
    for im in range(ax.m):
        for ic in range(ax.c):
            for t0 in range(0, ax.t, ct):
                t1 = min(t0 + ct, ax.t)
                for z0 in range(0, ax.z, cz):
                    if cancelled():
                        return None
                    z1 = min(z0 + cz, ax.z)
                    block = np.empty((t1 - t0, z1 - z0, int(ax.y), int(ax.x)), dtype=dt)
                    units = [(j, k) for j in range(t1 - t0) for k in range(z1 - z0)]

                    def read(unit, _im=im, _ic=ic, _t0=t0, _z0=z0):
                        j, k = unit
                        return np.asarray(prov.get_region(
                            0, _im, _t0 + j, _z0 + k, _ic,
                            0, int(ax.y), 0, int(ax.x)))

                    def place(_i, unit, plane, _b=block):
                        j, k = unit
                        # `casting="unsafe"` into a typed destination replaces `_cast`'s
                        # allocate-and-return for the pass-through case, but a float→integer
                        # narrowing still has to round and clip rather than truncate — so
                        # that one keeps going through `_cast`.
                        if plane.dtype == dt:
                            np.copyto(_b[j, k], plane)
                        else:
                            np.copyto(_b[j, k], _cast(plane, dt), casting="unsafe")

                    fold_units(read, units, place, batch=max(1, min(len(units), 8)))
                    arr0[im:im + 1, t0:t1, z0:z1, ic:ic + 1, :, :] = \
                        block.reshape(1, t1 - t0, z1 - z0, 1, int(ax.y), int(ax.x))
                    done += (t1 - t0) * (z1 - z0) * ax.y * ax.x
                    report(done, f"baking image · {done * 100 // max(1, total_units)}%")
    B2ndProvider._mark(arr0, 0, True)
    return arr0


def _write_voxel_layers(layers: Sequence[AttributeLayer], voxdir: str, *,
                        precision: str, report: Callable[[float, str], None],
                        base: int,
                        should_cancel: Optional[Callable[[], bool]] = None
                        ) -> Optional[List[Dict[str, Any]]]:
    """Write each Voxel attribute layer to its own ``.npy``, copied **per timepoint** so
    a dtype conversion never allocates a second copy of the whole layer.

    Plain ``.npy`` rather than a compressed store on purpose: the file is re-opened with
    ``mmap_mode="r"``, which needs a flat on-disk buffer, and that memory map is the
    entire reason a docked mask costs no RAM. A compressed layer would have to be
    inflated whole on open — exactly the retained full raster this feature removes."""
    if not layers:
        return []
    os.makedirs(voxdir, exist_ok=True)
    index, done = [], base
    for i, attr in enumerate(layers):
        dt = target_dtype(attr.values.dtype, precision)
        fname = f"{i:03d}_{_slug(attr.layer)}_{_slug(attr.name)}.npy"
        path = os.path.join(voxdir, fname)
        src = attr.values
        out = np.lib.format.open_memmap(path, mode="w+", dtype=dt, shape=src.shape)
        try:
            # (M,T,…) — copy one t at a time; the cast's temporary is then one frame,
            # not one series. A 0-length axis just skips the loop, which is correct.
            if src.ndim >= 2 and src.shape[0] and src.shape[1]:
                for im in range(src.shape[0]):
                    for it in range(src.shape[1]):
                        if should_cancel is not None and should_cancel():
                            return None
                        out[im, it] = _cast(src[im, it], dt)
                        done += int(src[im, it].size)
                        report(done, f"baking layer {attr.name!r}")
            else:
                out[...] = _cast(src, dt)
                done += int(src.size)
                report(done, f"baking layer {attr.name!r}")
            out.flush()
        finally:
            del out                      # close the map before anyone re-opens the file
        index.append({"domain": attr.domain.value, "layer": attr.layer,
                      "name": attr.name, "file": f"{VOXEL_DIR}/{fname}",
                      "dtype": str(dt), "shape": [int(s) for s in src.shape]})
    return index


def _write_tables(layers: Sequence[AttributeLayer], path: str) -> List[Dict[str, Any]]:
    """Write every non-Voxel attribute layer into one ``.npz``. These are the coarse
    lattice values (Frame/Plane/Channel/Global) and the exploded structure columns
    (Label/Point/Track/Mesh) — KB to MB, so one archive is the right granularity."""
    if not layers:
        try:
            os.remove(path)              # a re-bake must not leave the old tables behind
        except OSError:
            pass
        return []
    index, arrays = [], {}
    for i, attr in enumerate(layers):
        key = f"a{i}"
        arrays[key] = np.asarray(attr.values)
        index.append({"domain": attr.domain.value, "layer": attr.layer,
                      "name": attr.name, "key": key})
    with open(path, "wb") as f:
        np.savez(f, **arrays)
    return index


# ── read ──────────────────────────────────────────────────────────────────────

def read_manifest(dirpath: str) -> Optional[Dict[str, Any]]:
    """The checkpoint's manifest, or ``None`` when ``dirpath`` holds no *complete* one.

    Returns ``None`` rather than raising for every "not a usable checkpoint" case —
    absent directory, interrupted bake (no manifest), unreadable/corrupt JSON — because
    every caller's response is the same: treat it as un-baked and offer to bake. A
    checkpoint from a **newer** writer is the one hard error, since silently mis-reading
    a future layout would produce wrong pixels instead of a clear message."""
    p = os.path.join(dirpath or "", MANIFEST_NAME)
    if not dirpath or not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(man, dict):
        return None
    v = int(man.get("version", 0) or 0)
    if v > CHECKPOINT_VERSION:
        raise ValueError(
            f"the dock at {dirpath!r} was written by a newer version of this program "
            f"(checkpoint format {v}, this build reads {CHECKPOINT_VERSION}) — update, "
            f"or re-bake it here.")
    return man


def is_checkpoint(dirpath: str) -> bool:
    """True when ``dirpath`` holds a complete, readable checkpoint."""
    try:
        return read_manifest(dirpath) is not None
    except ValueError:                   # a future-format store IS one, just unreadable
        return True


def checkpoint_envelope(dirpath: str) -> Optional[MetaEnvelope]:
    """The :class:`~nodegraph.metadata.MetaEnvelope` a docked node should seed the
    edit-time propagation with — axes, calibration, domain set and layer catalog, read
    from the manifest alone (no pixels, no ``.npy`` map). Cheap enough for the GUI to
    call on every re-propagation; ``None`` when there is no complete checkpoint."""
    man = read_manifest(dirpath)
    if man is None:
        return None
    return MetaEnvelope(
        axes=_axes_of(man),
        metadata=dict(man.get("metadata", {}) or {}),
        domains=frozenset(_domain(d) for d in man.get("domains", []) if _domain(d)),
        layer_names=tuple((_domain(d), str(n))
                          for d, n in man.get("layer_names", []) if _domain(d)),
    )


def _domain(value: Any) -> Optional[Domain]:
    try:
        return Domain(value)
    except ValueError:                   # a domain this build does not know — skip it
        return None


def _axes_of(man: Dict[str, Any]) -> AxisSizes:
    """Axis sizes from a checkpoint manifest, driven off :data:`AXIS_ORDER` rather than a
    literal key tuple so a new axis cannot be silently dropped here (V3.01 added ``b``).

    A manifest written before an axis existed simply does not carry it, and the ``1``
    default is then exactly right: a pre-batch checkpoint IS a one-member batch. That is
    what makes this forward-compatible without a schema version bump."""
    a = man.get("axes", {}) or {}
    return AxisSizes(**{k: int(a.get(k, 1) or 1) for k in AXIS_ORDER})


def open_checkpoint(dirpath: str) -> Dataset:
    """Re-open a checkpoint as a **lazy** :class:`~nodegraph.dataset.Dataset`.

    The image is a disk-backed :class:`~nodegraph.provider.B2ndProvider` (tiles
    decompress on read) and every Voxel layer is a read-only ``np.memmap``, so opening a
    100 GB checkpoint costs kilobytes and the pages a consumer actually touches. Small
    tables load eagerly — they are small by construction.

    Raises :class:`FileNotFoundError` when there is no complete checkpoint, and
    :class:`ValueError` (from ``with_attribute``'s shape check) if a stored lattice layer
    disagrees with the manifest's axes — an integrity check worth keeping loud."""
    man = read_manifest(dirpath)
    if man is None:
        raise FileNotFoundError(
            f"no complete checkpoint in {dirpath!r} — it was never baked, or a bake was "
            f"interrupted before it finished. Re-bake it.")
    ds = Dataset(axes=_axes_of(man), metadata=dict(man.get("metadata", {}) or {}))
    if man.get("image"):
        ds = ds.with_image(B2ndProvider.open(os.path.join(dirpath, IMAGE_DIR)))
    for ent in man.get("voxel_layers", []) or []:
        dom = _domain(ent.get("domain"))
        if dom is None:
            continue
        arr = np.load(os.path.join(dirpath, str(ent["file"]).replace("/", os.sep)),
                      mmap_mode="r")
        ds = ds.with_attribute(AttributeLayer(dom, str(ent["name"]), arr, ent.get("layer")))
    tables = man.get("tables", []) or []
    if tables:
        with np.load(os.path.join(dirpath, TABLES_NAME), allow_pickle=False) as z:
            for ent in tables:
                dom = _domain(ent.get("domain"))
                if dom is None:
                    continue
                ds = ds.with_attribute(
                    AttributeLayer(dom, str(ent["name"]), z[str(ent["key"])],
                                   ent.get("layer")))
    return ds


def checkpoint_bytes(dirpath: str) -> int:
    """Total bytes the checkpoint occupies on disk (0 when absent) — what the GUI shows
    beside the dock so the cost of keeping it is never invisible."""
    total = 0
    for root, _dirs, files in os.walk(dirpath or ""):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def remove_checkpoint(dirpath: str) -> None:
    """Delete a checkpoint directory (un-dock discards). Refuses anything that is not
    recognisably a checkpoint, so a mis-set path can never take a user's data with it —
    the directory must hold a manifest, or the ``image``/``voxel`` layout of a bake that
    was interrupted before its manifest landed."""
    import shutil
    if not dirpath or not os.path.isdir(dirpath):
        return
    looks_baked = (os.path.isfile(os.path.join(dirpath, MANIFEST_NAME))
                   or os.path.isdir(os.path.join(dirpath, IMAGE_DIR))
                   or os.path.isdir(os.path.join(dirpath, VOXEL_DIR)))
    if not looks_baked:
        raise ValueError(
            f"refusing to delete {dirpath!r} — it holds no checkpoint (no manifest, no "
            f"image/ or voxel/ directory), so it is not something this bake wrote.")
    shutil.rmtree(dirpath, ignore_errors=True)
    if os.path.isdir(dirpath):
        # Windows will not unlink a file that is still MAPPED, and this checkpoint's
        # Voxel layers are memory-mapped by every Dataset that opened it. `rmtree` with
        # ignore_errors reports success either way, so without this check a "discard"
        # would silently leave the whole thing on disk. Say so instead — the fix is to
        # let go of the open Dataset (un-dock, or re-pull) before deleting.
        raise OSError(
            f"could not delete {dirpath!r} — some of it is still open. A docked "
            f"checkpoint's layers stay memory-mapped while anything is using them; "
            f"un-dock the node (or restart) and try again.")


__all__ = [
    "CHECKPOINT_VERSION", "PRECISIONS", "ProgressFn", "target_dtype",
    "write_checkpoint", "open_checkpoint", "read_manifest", "is_checkpoint",
    "checkpoint_envelope", "checkpoint_bytes", "remove_checkpoint",
    "derived_domains", "derived_layer_names",
]
