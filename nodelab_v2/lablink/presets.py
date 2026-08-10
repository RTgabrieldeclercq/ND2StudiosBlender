"""Remembering what worked — knob presets, and the record of what actually ran.

Two halves of one problem. Tuning a recipe against a warm session is cheap, so a person
will try twenty knob settings in a few minutes; without somewhere to put the one that
worked, all twenty are lost. And a folder of results that does not say what produced it
cannot be picked up again a week later, which is the difference between a processed image
you can build on and a processed image you can only look at.

:class:`Preset`
    A named knob set for one recipe, stored on this machine. Instant to save, needs no
    operator, and never leaves here. Promoting one into a real derived recipe is
    :mod:`nodelab_v2.lablink.recipe`'s job.

:class:`RunRecord`
    What one command actually ran with, written **into the results folder** so the folder
    describes itself, and appended to a local history.

**A preset stores the full effective set, including explicit nulls.** Storing only the
values that differ from the recipe's defaults would make a preset mean different things
before and after an operator edited the recipe, and re-applying a partial set onto a warm
session would leave whatever an earlier attempt set still in force for the knobs it omits.

**What cannot be recorded, and is therefore said out loud.** A recipe has no version field
and no content hash on the wire, so a run record names its recipe and can never prove that
recipe is byte-for-byte the one that runs today. :attr:`RunRecord.fidelity` says which tier
of evidence a record was built from rather than letting all three look alike.

Qt-free; standard library only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: This machine's presets. Beside ``overlays.json``, which set the convention.
PRESETS_FILE = Path.home() / ".nd2studios" / "lablink-presets.json"

#: Local history of every run this machine drove, newest last.
RUNS_DIR = Path.home() / ".nd2studios" / "lablink-runs"

#: What a self-describing results folder is called.
RUN_RECORD_NAME = "lablink-run.json"

#: Where a recipe this app generated is staged before and after submission, so there is
#: always a local copy of exactly what was sent.
OUTBOX_DIR = Path.home() / ".nd2studios" / "lablink-outbox"

_PRESETS_FORMAT = "nd2studios.lablink-presets/1"
_RUN_FORMAT = "nd2studios.lablink-run/1"


def _slug(text: str, *, fallback: str = "preset") -> str:
    """A filename- and recipe-name-safe slug. Recipe directory names must survive this."""
    out = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return out[:48] or fallback


# ── presets ─────────────────────────────────────────────────────────────────────

@dataclass
class Preset:
    """A named knob set for one recipe on one hub's workflow."""

    name: str
    workflow: str
    recipe: str
    #: knob name -> value, the FULL effective set. ``None`` means "derive from the file",
    #: and is stored rather than omitted — see the module docstring.
    knobs: Dict[str, Any] = field(default_factory=dict)
    #: Free text from whoever saved it: "HeLa, 20x, works on the dim wells".
    note: str = ""
    #: The hub it was tuned against, for provenance only. A preset is not scoped to a hub —
    #: the same recipe on another hub takes the same knobs.
    hub: str = ""
    #: ISO-8601, supplied by the caller (this module never reads the clock, so a record is
    #: reproducible and a test does not depend on the time of day).
    saved: str = ""
    #: How the source data was calibrated when this was tuned, so a preset applied to a
    #: differently-calibrated file can warn rather than silently mean something else.
    pixel_size_um: Optional[float] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.workflow, self.recipe)

    def to_dict(self) -> Dict[str, Any]:
        out = {"name": self.name, "workflow": self.workflow, "recipe": self.recipe,
               "knobs": dict(self.knobs)}
        for attr in ("note", "hub", "saved"):
            if getattr(self, attr):
                out[attr] = getattr(self, attr)
        if self.pixel_size_um is not None:
            out["pixel_size_um"] = self.pixel_size_um
        return out

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Preset":
        return cls(
            name=str(doc.get("name") or ""),
            workflow=str(doc.get("workflow") or ""),
            recipe=str(doc.get("recipe") or ""),
            knobs=dict(doc.get("knobs") or {}),
            note=str(doc.get("note") or ""),
            hub=str(doc.get("hub") or ""),
            saved=str(doc.get("saved") or ""),
            pixel_size_um=doc.get("pixel_size_um"),
        )

    def slug(self) -> str:
        return _slug(self.name)


@dataclass
class AppliedPreset:
    """A preset checked against a recipe as it is *now*, with what no longer fits.

    Applying is validated rather than trusted because a recipe can change under a preset:
    an operator may rename a knob, narrow a bound, or install a different recipe under the
    same name — and a recipe has no version to compare against. Silently dropping the knobs
    that no longer fit would hand back a preset that quietly does something else.
    """

    knobs: Dict[str, Any] = field(default_factory=dict)
    #: Knobs the recipe no longer declares.
    unknown: Tuple[str, ...] = ()
    #: ``(knob, reason)`` for values the recipe would now refuse.
    out_of_range: Tuple[Tuple[str, str], ...] = ()
    #: Knobs cleared to ``None`` because their ``applies_when`` is no longer met.
    inapplicable: Tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.unknown or self.out_of_range)

    def warnings(self) -> List[str]:
        """One sentence per problem, for a status line."""
        out: List[str] = []
        if self.unknown:
            out.append(f"this recipe no longer has {', '.join(self.unknown)}")
        for name, why in self.out_of_range:
            out.append(f"{name}: {why}")
        if self.inapplicable:
            out.append(f"not applicable now, so left derived: "
                       f"{', '.join(self.inapplicable)}")
        return out


def apply_preset(preset: Preset, recipe_meta: Mapping[str, Any]) -> AppliedPreset:
    """Fit ``preset`` to ``recipe_meta`` as the hub publishes it today.

    Everything that still fits comes back in :attr:`AppliedPreset.knobs`; everything that
    does not is reported rather than dropped.
    """
    from nodelab_v2.lablink.client import HubClient, LabLinkError, knob_applies

    declared = HubClient.declared_knobs(dict(recipe_meta))
    if not declared:
        return AppliedPreset(knobs=dict(preset.knobs))

    unknown = tuple(sorted(set(preset.knobs) - set(declared)))
    wanted = {k: v for k, v in preset.knobs.items() if k in declared}

    inapplicable: List[str] = []
    for name, spec in declared.items():
        if name in wanted and wanted[name] is not None \
                and not knob_applies(spec, wanted, declared):
            wanted[name] = None
            inapplicable.append(name)

    bad: List[Tuple[str, str]] = []
    good: Dict[str, Any] = {}
    for name, value in wanted.items():
        try:
            HubClient.check_knobs(dict(recipe_meta), {name: value})
        except LabLinkError as exc:
            bad.append((name, str(exc)))
            continue
        good[name] = value
    return AppliedPreset(knobs=good, unknown=unknown,
                         out_of_range=tuple(bad),
                         inapplicable=tuple(sorted(inapplicable)))


class PresetStore:
    """The presets on this machine, as one JSON file.

    Read on demand rather than cached, because the editor is not the only thing that may
    write this file (a second window, a script) and a stale in-memory copy would silently
    drop somebody else's save on the next write.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else PRESETS_FILE

    # ── whole file ──────────────────────────────────────────────────────────────
    def load(self) -> List[Preset]:
        """Every preset, or an empty list. Never raises: a corrupt file must not stop the
        editor from opening, and :meth:`problem` is how the GUI says so."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return []
        if not isinstance(doc, Mapping):
            return []
        out = []
        for row in (doc.get("presets") or []):
            if isinstance(row, Mapping) and row.get("name"):
                out.append(Preset.from_dict(row))
        return out

    def problem(self) -> str:
        """Why the file could not be read, or ``""``. Distinguishes "no presets yet" from
        "your presets are there but unreadable", which look identical to :meth:`load`."""
        if not self.path.exists():
            return ""
        try:
            with open(self.path, encoding="utf-8") as fh:
                json.load(fh)
        except (OSError, ValueError) as exc:
            return f"{self.path} could not be read: {exc}"
        return ""

    def _write(self, presets: Sequence[Preset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"format": _PRESETS_FORMAT,
               "presets": [p.to_dict() for p in presets]}
        tmp = self.path.with_suffix(self.path.suffix + ".partial")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.path)

    # ── one preset ──────────────────────────────────────────────────────────────
    def for_recipe(self, workflow: str, recipe: str) -> List[Preset]:
        """This recipe's presets, by name."""
        return sorted((p for p in self.load() if p.key == (workflow, recipe)),
                      key=lambda p: p.name.lower())

    def get(self, workflow: str, recipe: str, name: str) -> Optional[Preset]:
        for preset in self.for_recipe(workflow, recipe):
            if preset.name == name:
                return preset
        return None

    def save(self, preset: Preset) -> Preset:
        """Add or replace a preset, matched on ``(workflow, recipe, name)``."""
        if not preset.name.strip():
            raise ValueError("a preset needs a name")
        keep = [p for p in self.load()
                if not (p.key == preset.key and p.name == preset.name)]
        keep.append(preset)
        self._write(keep)
        return preset

    def delete(self, workflow: str, recipe: str, name: str) -> bool:
        before = self.load()
        keep = [p for p in before
                if not (p.key == (workflow, recipe) and p.name == name)]
        if len(keep) == len(before):
            return False
        self._write(keep)
        return True

    def rename(self, workflow: str, recipe: str, old: str, new: str) -> Optional[Preset]:
        preset = self.get(workflow, recipe, old)
        if preset is None:
            return None
        self.delete(workflow, recipe, old)
        return self.save(replace(preset, name=new))


# ── run records ─────────────────────────────────────────────────────────────────

#: How much of a run record is actually known, worst to best. Kept explicit because the
#: three are not interchangeable and a UI that shows them identically is lying.
FIDELITY = ("recipe-name-only", "metrics-artifact", "full")


@dataclass
class RunRecord:
    """What one command ran with, and how well that is known."""

    recipe: str = ""
    workflow: str = ""
    hub: str = ""
    #: knob name -> value, the hub's own echo where we have it: it includes the recipe's
    #: defaults and the ones left derived, which is what makes it a reproducible record.
    knobs: Dict[str, Any] = field(default_factory=dict)
    #: knob name -> ``node`` / ``graph`` / ``derive``, when the worker reported it.
    knob_sources: Dict[str, str] = field(default_factory=dict)
    cmd_id: str = ""
    cmd_seq: int = 0
    session: str = ""
    duration_s: Optional[float] = None
    cached_steps: int = 0
    computed_steps: int = 0
    #: ``[{"name", "sha256", "role", "sidecar": {...}}]``
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    #: ``[{"name", "sha256", "kind", "bytes"}]``
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    software_version: str = ""
    worker_version: str = ""
    requires_metadata: List[str] = field(default_factory=list)
    ran: str = ""
    fidelity: str = "full"
    #: Where the evidence came from, for a UI that has to explain itself.
    source: str = ""

    @property
    def reproducible(self) -> bool:
        """Whether these knobs can be replayed as a complete set.

        False for a record recovered from a ``metrics`` artifact written by an older worker,
        which recorded only the knobs a caller pinned — so replaying it would silently take
        the recipe's *current* defaults for everything else.
        """
        return self.fidelity == "full" and bool(self.knobs)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"format": _RUN_FORMAT}
        for attr in ("recipe", "workflow", "hub", "cmd_id", "session", "ran",
                     "software_version", "worker_version", "fidelity", "source"):
            if getattr(self, attr):
                out[attr] = getattr(self, attr)
        out["knobs"] = dict(self.knobs)
        if self.knob_sources:
            out["knob_sources"] = dict(self.knob_sources)
        if self.cmd_seq:
            out["cmd_seq"] = self.cmd_seq
        if self.duration_s is not None:
            out["duration_s"] = self.duration_s
        if self.cached_steps or self.computed_steps:
            out["steps"] = {"cached": self.cached_steps, "computed": self.computed_steps}
        if self.inputs:
            out["inputs"] = list(self.inputs)
        if self.artifacts:
            out["artifacts"] = list(self.artifacts)
        if self.requires_metadata:
            out["requires_metadata"] = list(self.requires_metadata)
        return out

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any], *, source: str = "") -> "RunRecord":
        steps = doc.get("steps") or {}
        return cls(
            recipe=str(doc.get("recipe") or ""),
            workflow=str(doc.get("workflow") or ""),
            hub=str(doc.get("hub") or ""),
            knobs=dict(doc.get("knobs") or {}),
            knob_sources={str(k): str(v) for k, v in
                          (doc.get("knob_sources") or {}).items()},
            cmd_id=str(doc.get("cmd_id") or ""),
            cmd_seq=int(doc.get("cmd_seq") or 0),
            session=str(doc.get("session") or ""),
            duration_s=doc.get("duration_s"),
            cached_steps=int(steps.get("cached") or 0),
            computed_steps=int(steps.get("computed") or 0),
            inputs=list(doc.get("inputs") or []),
            artifacts=list(doc.get("artifacts") or []),
            software_version=str(doc.get("software_version") or ""),
            worker_version=str(doc.get("worker_version") or ""),
            requires_metadata=[str(f) for f in (doc.get("requires_metadata") or [])],
            ran=str(doc.get("ran") or ""),
            fidelity=str(doc.get("fidelity") or "full"),
            source=source or str(doc.get("source") or ""),
        )

    def as_preset(self, name: str, *, note: str = "") -> Preset:
        """This run's knobs as a saveable preset — "the settings that made this"."""
        return Preset(name=name, workflow=self.workflow, recipe=self.recipe,
                      knobs=dict(self.knobs), note=note, hub=self.hub, saved=self.ran)


def write_run_record(record: RunRecord, results_dir: str, *,
                     history: bool = True) -> str:
    """Write ``record`` into ``results_dir`` and (by default) append it to the history.

    Into the results folder deliberately: the hub's own per-file metadata carries the recipe
    name in the *channel listing* only, so it is gone the moment a file is downloaded. A
    folder that does not describe itself is a folder nobody can pick up later.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, RUN_RECORD_NAME)
    doc = record.to_dict()
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)

    if history:
        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = _slug(record.ran or "run", fallback="run")
            name = f"{stamp}-{_slug(record.recipe, fallback='recipe')}"
            if record.cmd_id:
                name += f"-{_slug(record.cmd_id, fallback='cmd')}"
            with open(RUNS_DIR / f"{name}.json", "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2, sort_keys=True)
                fh.write("\n")
        except OSError:
            pass          # history is a convenience; the folder's own copy is the record
    return path


def read_run_record(path: str) -> Optional[RunRecord]:
    """Recover what produced a results folder, by the best evidence available.

    Three tiers, tried in order, and the record says which one answered:

    1. ``lablink-run.json`` — written by this app, the hub's full knob echo. Reproducible.
    2. a ``*_metrics.json`` artifact — the worker's own blob. Complete from a worker that
       records every knob's source; from an older one, only the knobs a caller pinned, so
       it is marked ``metrics-artifact`` and **not** reproducible.
    3. nothing but a recipe name.

    ``path`` may be the folder or any file in it.
    """
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        return None

    own = os.path.join(folder, RUN_RECORD_NAME)
    if os.path.isfile(own):
        try:
            with open(own, encoding="utf-8") as fh:
                doc = json.load(fh)
            if isinstance(doc, Mapping):
                record = RunRecord.from_dict(doc, source=RUN_RECORD_NAME)
                record.fidelity = str(doc.get("fidelity") or "full")
                return record
        except (OSError, ValueError):
            pass

    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or "metrics" not in name:
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(doc, Mapping) or not doc.get("recipe"):
            continue
        sources = {str(k): str(v) for k, v in (doc.get("knob_sources") or {}).items()}
        record = RunRecord(
            recipe=str(doc.get("recipe") or ""),
            knobs=dict(doc.get("knobs") or {}),
            knob_sources=sources,
            duration_s=doc.get("seconds"),
            cached_steps=int(doc.get("nodes_cached") or 0),
            computed_steps=int(doc.get("nodes_computed") or 0),
            software_version=str(doc.get("software_version") or ""),
            worker_version=str(doc.get("worker_version") or ""),
            requires_metadata=[str(f) for f in (doc.get("requires_metadata") or [])],
            # `knob_sources` is what tells us the worker recorded EVERY knob rather than
            # only the pinned ones. Without it the knob map is a subset of what ran, and
            # replaying it would silently pick up today's defaults for the rest.
            fidelity="full" if sources else "metrics-artifact",
            source=name)
        for role, sent in (doc.get("inputs") or {}).items():
            if isinstance(sent, Mapping):
                record.inputs.append({"role": role, **dict(sent)})
        return record

    return None


def results_look_processed(folder: str) -> bool:
    """Whether ``folder`` looks like a LabLink results folder at all."""
    if not os.path.isdir(folder):
        return False
    if os.path.isfile(os.path.join(folder, RUN_RECORD_NAME)):
        return True
    return any("metrics" in n and n.endswith(".json") for n in os.listdir(folder))


__all__ = [
    "PRESETS_FILE", "RUNS_DIR", "RUN_RECORD_NAME", "OUTBOX_DIR", "FIDELITY",
    "Preset", "AppliedPreset", "apply_preset", "PresetStore",
    "RunRecord", "write_run_record", "read_run_record", "results_look_processed",
]
