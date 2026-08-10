"""The image-job sidecar — the metadata that travels WITH an image to a hub.

A TIFF does not merely omit the optical metadata an analysis derives from. It **invents**
it: the container's 16-bit depth stands in for a 12-bit sensor, and a channel called
``GFP`` arrives as ``Ch0``. A missing value can be detected and refused; an invented one
is indistinguishable from a measured one, and produces a plausible wrong answer with no
warning from any layer. So every image sent to a hub travels with a ``*.job.json``
sidecar whose values **override** whatever the file claims, and a recipe declares in its
``requires.metadata`` which fields its graph cannot run correctly without.

The upload call is content-agnostic and stays that way, so **a successful upload never
means the sidecar was valid**. Writing one is this module's job; refusing a run that lacks
a required field is :mod:`nodelab_v2.lablink.worker`'s.

Three rules this module exists to hold:

* **It never invents.** :func:`draft` reports what it genuinely read, what the reader
  supplied as a placeholder, and what is simply absent — as three separate tuples — so the
  GUI can ask a person for the rest instead of writing a number nobody measured. A TIFF's
  ``Ch0``/``Ch1`` names are a placeholder, not a name, and are classified as such.
* **The unit is in the field name.** There is no ``pixel_size`` with a companion unit
  field, on either side of the wire. A mismatched unit pair is the most expensive silent
  error available here.
* **The ``job`` block is provenance, never input.** LabLink specifies that a sidecar whose
  ``job`` disagrees with the API's knobs should be refused, and records that the check is
  not implemented. Until it is, knobs live in exactly one place — the command — and this
  module writes ``job`` for a human reader and never reads it back as a value.

Qt-free. The writer needs :mod:`nodelab_v2.ingest` (metadata only, no pixels); the reader
and :func:`missing_metadata` are pure standard library, so the worker can refuse a job
without importing a reader.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from nodelab_v2.lablink import protocol as P


class SidecarError(Exception):
    """A sidecar that cannot be trusted — absent, unparseable, or a format we do not know."""


# ── reading ─────────────────────────────────────────────────────────────────────

def sidecar_path_for(image_path: str) -> str:
    """``x.tif`` -> ``x.tif.job.json``.

    The suffix appends rather than replaces, and pairing is on the full name rather than the
    stem: an image beside an unrelated ``x.json`` is an ordinary thing to find in a folder,
    and pairing on a bare ``.json`` would silently adopt it as this file's calibration.
    """
    return image_path + P.SIDECAR_SUFFIX


def is_sidecar(name: str) -> bool:
    """True for a name the hub handed us that is a sidecar rather than an image.

    The hub labels every input with the recipe's declared ``role`` regardless of filename,
    so the suffix is the only thing that distinguishes the two — and the worker must know
    before it tries to open one as an image.
    """
    return str(name).lower().endswith(P.SIDECAR_SUFFIX)


def read_sidecar(path: str) -> Dict[str, Any]:
    """Parse a sidecar and return its document.

    Refuses rather than guesses: an absent or unrecognised ``format`` raises. A sidecar is
    a claim that overrides the file's own metadata, so accepting one we cannot interpret
    would be the one failure mode this whole mechanism exists to remove.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError as exc:
        raise SidecarError(f"no sidecar at {path}") from exc
    except (OSError, ValueError) as exc:
        raise SidecarError(f"{os.path.basename(path)} is not readable JSON: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise SidecarError(f"{os.path.basename(path)}: top level must be an object")
    fmt = doc.get("format")
    if not fmt:
        raise SidecarError(
            f"{os.path.basename(path)} has no 'format' — refusing to guess that it is "
            f"{P.SIDECAR_FORMAT}, because a wrong guess silently overrides the image's "
            f"own calibration")
    if fmt != P.SIDECAR_FORMAT:
        raise SidecarError(
            f"{os.path.basename(path)} declares format {fmt!r}; this build reads "
            f"{P.SIDECAR_FORMAT!r}")
    return dict(doc)


def sidecar_metadata(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a sidecar document into the engine's calibration vocabulary.

    The sidecar nests per-channel facts (the shape a person writing one gets right); the
    engine wants parallel per-channel lists (the shape a node reading one gets right).
    :data:`~nodelab_v2.lablink.protocol.SIDECAR_CHANNEL_FIELDS` is the mapping between them.

    A channel list is emitted **whole or not at all**, and keeps ``None`` for channels that
    genuinely have no value — a transmitted-light channel has no emission wavelength, and
    dropping it would shift every later channel's value onto the wrong channel.
    """
    image = doc.get("image")
    if not isinstance(image, Mapping):
        raise SidecarError("sidecar has no 'image' block")

    out: Dict[str, Any] = {}
    for key in P.SIDECAR_IMAGE_FIELDS:
        value = image.get(key)
        if value is not None:
            out[key] = value

    channels = image.get("channels")
    if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
        rows = [c if isinstance(c, Mapping) else {} for c in channels]
        for src, dst in P.SIDECAR_CHANNEL_FIELDS.items():
            column = [row.get(src) for row in rows]
            # All-absent is "not stated"; partly-absent is a real per-channel list with
            # holes, and the holes are meaningful.
            if any(v is not None for v in column):
                out[dst] = column
    return out


def missing_metadata(required: Iterable[str],
                     metadata: Mapping[str, Any]) -> List[str]:
    """Which of ``required`` the resolved metadata does not supply, in declared order.

    Pure standard library, so the worker can refuse before importing a reader. A field is
    supplied only if it is present AND not ``None``; for the per-channel lists, only if at
    least one channel carries a value — an all-``None`` list is a list-shaped absence, and
    treating it as present is exactly the fail-open this function exists to prevent.
    """
    out: List[str] = []
    for name in required:
        value = metadata.get(name)
        if value is None:
            out.append(name)
            continue
        if isinstance(value, (list, tuple)) and not any(v is not None for v in value):
            out.append(name)
    return out


def unknown_requirements(required: Iterable[str]) -> List[str]:
    """Requirements naming a field no sidecar can ever supply.

    A recipe asking for a field outside
    :data:`~nodelab_v2.lablink.protocol.REQUIRABLE_METADATA` is not a job that needs a
    better file — it is a recipe that can never be satisfied, and saying so at validation
    time beats refusing every job forever with a message about the data.
    """
    known = set(P.REQUIRABLE_METADATA)
    return [str(n) for n in required if str(n) not in known]


# ── writing ─────────────────────────────────────────────────────────────────────

#: Channel names a reader manufactured rather than read. :func:`ingest.read_channel_display`
#: degrades a TIFF to ``Ch0…`` so the GUI has something to label a toggle with — useful
#: there, and a fabrication here.
def _is_placeholder_names(names: Sequence[Any]) -> bool:
    return bool(names) and all(
        isinstance(n, str) and n == f"Ch{i}" for i, n in enumerate(names))


@dataclass
class SidecarDraft:
    """A sidecar built from an image, plus an honest account of where each field came from.

    The three tuples are the point. ``absent`` and ``invented`` are what a person still has
    to answer for, and keeping them apart from ``from_file`` is what stops this module from
    quietly writing a number nobody measured.
    """

    image_name: str
    doc: Dict[str, Any]
    from_file: Tuple[str, ...] = ()
    invented: Tuple[str, ...] = ()
    absent: Tuple[str, ...] = ()
    channel_count: int = 0
    notes: Tuple[str, ...] = ()

    @property
    def metadata(self) -> Dict[str, Any]:
        """The flat engine view of this draft — what the worker would resolve."""
        return sidecar_metadata(self.doc)

    def unmet(self, required: Iterable[str]) -> List[str]:
        """Which of ``required`` this draft does not yet answer."""
        return missing_metadata(required, self.metadata)

    def with_values(self, values: Mapping[str, Any]) -> "SidecarDraft":
        """A copy with ``values`` (flat engine keys) written in — what a prompt returns.

        Anything set this way moves out of ``invented``/``absent``, because a person
        supplying a value is the authority the file was not.
        """
        doc = json.loads(json.dumps(self.doc))       # deep copy of plain JSON
        image = doc.setdefault("image", {})
        channels = image.setdefault("channels", [])
        filled: List[str] = []

        per_channel = {v: k for k, v in P.SIDECAR_CHANNEL_FIELDS.items()}
        for key, value in values.items():
            if value is None:
                continue
            if key in P.SIDECAR_IMAGE_FIELDS:
                image[key] = value
                filled.append(key)
            elif key in per_channel:
                field_name = per_channel[key]
                column = list(value) if isinstance(value, (list, tuple)) else [value]
                while len(channels) < len(column):
                    channels.append({})
                for row, item in zip(channels, column):
                    if item is not None:
                        row[field_name] = item
                filled.append(key)

        return SidecarDraft(
            image_name=self.image_name,
            doc=doc,
            from_file=self.from_file,
            invented=tuple(k for k in self.invented if k not in filled),
            absent=tuple(k for k in self.absent if k not in filled),
            channel_count=max(self.channel_count, len(channels)),
            notes=self.notes,
        )


def draft(image_path: str, *, recipe: str = "",
          original_name: str = "") -> SidecarDraft:
    """Build a sidecar for ``image_path`` from the file's own metadata.

    Reads metadata only — :func:`nodelab_v2.ingest.read_calibration` and
    :func:`~nodelab_v2.ingest.read_channel_display` both open the file without decoding
    pixels, so this is cheap enough to run on a multi-gigabyte stack before uploading it.

    Every requirable field the file does not answer lands in ``absent``, and every one the
    reader manufactured lands in ``invented``. Neither is written as though measured.
    """
    from nodelab_v2 import ingest

    name = os.path.basename(image_path)
    notes: List[str] = []
    calib: Dict[str, Any] = {}
    display: Dict[str, Any] = {}
    axes = None
    try:
        # One call for the axes AND both metadata dicts, and it decodes no pixels.
        axes, calib, display = ingest.read_meta_only(image_path)
        calib, display = dict(calib), dict(display)
    except Exception as exc:                        # noqa: BLE001 — an unreadable file must
        # still yield a draft, with everything in `absent`, so the GUI can say what is
        # missing rather than failing before it can ask.
        notes.append(f"metadata unreadable ({type(exc).__name__}: {exc})")

    merged: Dict[str, Any] = {}
    merged.update({k: v for k, v in calib.items() if v is not None})
    merged.update({k: v for k, v in display.items() if v is not None})

    # A single plane has no spacing. `read_calibration` already drops a fabricated
    # `z_step_um` for an ND2 with no Z axis, but the TIFF path has no such guard and hands
    # back the SDK's default -- on a one-plane TIFF that arrives equal to the pixel size,
    # which is a real-looking number that would satisfy a `requires.metadata` check for
    # `z_step_um` and then feed the 3D spacing of every measurement.
    if axes is not None and int(getattr(axes, "z", 1) or 1) <= 1:
        if merged.pop("z_step_um", None) is not None:
            notes.append("the image has one z plane, so z_step_um is not written")

    from_file: List[str] = []
    invented: List[str] = []
    absent: List[str] = []

    image: Dict[str, Any] = {}
    for key in P.SIDECAR_IMAGE_FIELDS:
        value = merged.get(key)
        if value is None:
            absent.append(key)
            continue
        image[key] = value
        from_file.append(key)

    names = merged.get("channel_names") or []
    count = len(names) if isinstance(names, (list, tuple)) else 0
    for key in P.SIDECAR_CHANNEL_FIELDS.values():
        column = merged.get(key)
        if isinstance(column, (list, tuple)):
            count = max(count, len(column))

    channels: List[Dict[str, Any]] = [{} for _ in range(count)]
    for src, engine_key in P.SIDECAR_CHANNEL_FIELDS.items():
        column = merged.get(engine_key)
        if not isinstance(column, (list, tuple)) or not any(v is not None for v in column):
            absent.append(engine_key)
            continue
        if engine_key == "channel_names" and _is_placeholder_names(column):
            # `Ch0`, `Ch1`, ... is what the reader shows on a toggle for a TIFF that names
            # nothing. Writing it here would assert it IS the channel's name, and a recipe
            # selecting a channel by name would then match a label nobody chose.
            invented.append(engine_key)
            notes.append(
                "the file names no channels; 'Ch0', 'Ch1', ... is this reader's "
                "placeholder and is not written as a name")
            continue
        for row, value in zip(channels, column):
            if value is not None:
                row[src] = value
        from_file.append(engine_key)

    if channels:
        image["channels"] = channels

    doc: Dict[str, Any] = {"format": P.SIDECAR_FORMAT, "image": image}
    source: Dict[str, Any] = {}
    if original_name and original_name != name:
        source["original_name"] = original_name
    if notes:
        source["note"] = "; ".join(notes)
    if source:
        doc["source"] = source
    if recipe:
        # Provenance for a human reading the folder later. Deliberately no `knobs` here:
        # the command carries those, and two copies of a knob set is two things to
        # disagree.
        doc["job"] = {"recipe": recipe}

    return SidecarDraft(
        image_name=name,
        doc=doc,
        from_file=tuple(from_file),
        invented=tuple(invented),
        absent=tuple(absent),
        channel_count=count,
        notes=tuple(notes),
    )


def write_sidecar(draft_or_doc: Any, dest_dir: str, *,
                  image_name: str = "") -> str:
    """Write a sidecar beside where its image will be, and return the path.

    ``dest_dir`` is a staging directory, not the image's own folder: the sidecar is uploaded
    as a second file on the same channel, so it needs a name derived from the *repaired*
    image name the hub will see rather than from the original on disk.
    """
    if isinstance(draft_or_doc, SidecarDraft):
        doc = draft_or_doc.doc
        name = image_name or draft_or_doc.image_name
    else:
        doc = dict(draft_or_doc)
        name = image_name
    if not name:
        raise ValueError("write_sidecar needs the image name the sidecar pairs with")

    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(name) + P.SIDECAR_SUFFIX)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return path


__all__ = [
    "SidecarError", "SidecarDraft",
    "sidecar_path_for", "is_sidecar", "read_sidecar", "sidecar_metadata",
    "missing_metadata", "unknown_requirements", "draft", "write_sidecar",
]
