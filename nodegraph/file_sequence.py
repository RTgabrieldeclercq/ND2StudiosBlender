"""Filename-sequence detection — the shared half of ``util.chain`` and the GUI's
**File -> Load file sequence...** action.

A microscope that exports one file per frame writes a *recurring* name: the stem is
constant, one numeric field counts up, and every other numeric field stays put::

    WellA3_t001_z002.nd2
    WellA3_t002_z002.nd2
    WellA3_t003_z002.nd2
           ^^^^        varies  -> the sequence index
                ^^^^   constant -> part of the stem

**Which number is the sequence is derived, never assumed** (`wire-node-v2` Section 7b,
"derive, don't ask"). Taking the last numeric run — the obvious shortcut — picks ``z002``
above and orders three timepoints by a field that never moves, which is a silent
mis-ordering: the pull succeeds, the movie plays, and frame 2 is somewhere it was not
acquired. So the field is chosen by *comparing the names to each other*, and when two
fields both vary this module REFUSES and names them both rather than picking one. The
GUI's scan dialog then shows the detected pattern and the resulting order and lets the
user override it, which is the only place the ambiguity can actually be resolved.

Qt-free, standard library only: the node re-derives the same order from the per-position
``source_file`` metadata that the loader stamped, so the canvas and the pull cannot
disagree about which file is frame 0.

**Why this sits at the top of ``nodegraph/`` and not in ``catalog/_shared/``.** Three
layers need it — :func:`nodegraph.metadata.chain_grow` (edit-time axis prediction), the
``util.chain`` compute (pull time) and the GUI's scan dialog — and the first of those is
core. A ``catalog`` import from :mod:`nodegraph.metadata` would be a cycle: importing any
catalog module runs ``nodegraph.catalog.__init__``, which imports every node module, each
of which imports ``metadata`` back while it is still half-initialised. Importing nothing
from this package is what keeps that impossible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: One numeric run or one non-numeric run. Splitting a name on this alternation is what
#: makes ``t9`` and ``t10`` the SAME field (both are "the digits after ``_t``") instead of
#: two names with different shapes — a fixed-width assumption would break the moment an
#: acquisition rolled over from 9 to 10, which is the first thing every real series does.
_RUN = re.compile(r"\d+|\D+")

#: The token a user-supplied pattern uses for the counting field, e.g.
#: ``WellA3_t{}_z002.nd2``. Chosen over ``*`` because a pattern has exactly ONE counting
#: field by construction and ``*`` invites a second one that this module cannot honour.
SEQ_TOKEN = "{}"


@dataclass(frozen=True)
class SequenceSpec:
    """A detected sequence: which numeric field counts, and what the rest of the name is.

    ``field`` indexes into :func:`segments`' output, not into the name's characters, so it
    survives the width change from ``t009`` to ``t010``.
    """

    #: Index into :func:`segments` of the numeric run that varies.
    field: int
    #: The first name's segments — the stem, with ``segments[field]`` the counting one.
    segments: Tuple[str, ...]
    #: Every numeric field that varied, when more than one did (ambiguous detection).
    #: Empty on a clean detection. Non-empty means ``field`` is a GUESS the caller must
    #: show the user before acting on it.
    ambiguous: Tuple[int, ...] = ()

    @property
    def width(self) -> int:
        """The counting field's zero-padded width in the first name (``3`` for ``t001``).

        Used only to rebuild a pattern for display; matching never relies on it.
        """
        return len(self.segments[self.field])

    def pattern(self) -> str:
        """The name with the counting field replaced by :data:`SEQ_TOKEN` — what the scan
        dialog shows and what the user edits."""
        return "".join(SEQ_TOKEN if i == self.field else s
                       for i, s in enumerate(self.segments))

    def index_of(self, name: str) -> Optional[int]:
        """``name``'s sequence number, or ``None`` if it does not fit this pattern.

        Matched structurally (same literal segments, same field count), not by regex on
        the rebuilt pattern, so a stem containing a regex metacharacter -- ``well[3].nd2``,
        ``A3 (2).nd2`` -- is compared as the literal text it is.
        """
        segs = segments(name)
        if len(segs) != len(self.segments):
            return None
        for i, (a, b) in enumerate(zip(segs, self.segments)):
            if i == self.field:
                if not a.isdigit():
                    return None
                continue
            if a.isdigit() != b.isdigit():
                return None
            # a non-counting numeric field must match by VALUE, not by text, so that a
            # series whose constant field is written `z2` in one file and `z02` in the
            # next is still one series rather than two that silently never meet.
            if a != b and not (a.isdigit() and int(a) == int(b)):
                return None
        return int(segs[self.field])


def segments(name: str) -> Tuple[str, ...]:
    """``name`` split into alternating numeric / non-numeric runs."""
    return tuple(_RUN.findall(str(name)))


def _shape(segs: Sequence[str]) -> Tuple[object, ...]:
    """A name's comparable skeleton: each numeric run collapses to ``True``, each literal
    run stays itself. Two names share a shape iff they differ only in their numbers."""
    return tuple(True if s.isdigit() else s for s in segs)


def detect(names: Sequence[str], *, whole: bool = False) -> Optional[SequenceSpec]:
    """The sequence ``names`` form, or ``None`` if they do not form one.

    Returns ``None`` — rather than a guess — when the names have different shapes (a
    folder holding two unrelated acquisitions) or when no numeric field varies at all
    (the same file listed twice). When SEVERAL fields vary, the left-most is returned
    with every varying field recorded in :attr:`SequenceSpec.ambiguous`, because the
    caller can only resolve that by showing the user; acting on it silently is the
    mis-ordering this module exists to prevent.

    ``names`` may be full paths; only the basename is examined, so two directories whose
    files share a naming scheme do not read as one sequence through their parent names.
    Pass ``whole=True`` to compare the entire string instead — what :func:`order` retries
    with when the basenames are identical and the counter must therefore be in the path.
    """
    bases = [str(n) if whole else os.path.basename(str(n)) for n in names if str(n)]
    if len(bases) < 2:
        return None
    segs = [segments(b) for b in bases]
    shape = _shape(segs[0])
    if any(_shape(s) != shape for s in segs[1:]):
        return None
    fields = [i for i, s in enumerate(segs[0]) if s.isdigit()]
    varying = tuple(i for i in fields
                    if len({int(s[i]) for s in segs}) > 1)
    if not varying:
        return None
    return SequenceSpec(field=varying[0], segments=segs[0],
                        ambiguous=varying if len(varying) > 1 else ())


def order(names: Sequence[str]) -> List[int]:
    """The permutation that puts ``names`` into sequence order — ``order(names)[k]`` is
    the index in ``names`` of the k-th file of the series.

    Falls back to a **natural sort** (digit runs compared as numbers, so ``t9`` precedes
    ``t10``) when :func:`detect` finds no single counting field. That fallback is the
    honest answer for a set this module cannot read as a sequence: it is still stable and
    still beats ASCII order, and the caller that cares -- the GUI dialog -- shows the
    resulting order rather than trusting it.

    Ties keep their input order (``sorted`` is stable), so two files that genuinely carry
    the same index stay in the order the caller supplied.

    **Where the counter is in the DIRECTORY** — a series stored one file per numbered
    folder, ``t001/img.nd2``, ``t002/img.nd2`` — the basenames are all identical and carry
    nothing to sort by. That is exactly when a bundle's labels arrive already carrying
    their parent segments (:func:`nodelab_v2.runner._unique_labels` lengthens a label only
    to break a basename collision), so the retry below reads the counter out of the path
    instead. It is deliberately a retry rather than the first attempt: comparing full paths
    up front would let a constant parent directory's own digits look like a counting field.
    """
    strs = [str(n) for n in names]
    bases = [os.path.basename(s) for s in strs]
    collide = len(set(bases)) < len(bases)
    for cand, whole in ([(bases, False), (strs, True)] if collide else [(bases, False)]):
        spec = detect(cand, whole=whole)
        if spec is None:
            continue
        keys: List[Tuple[int, ...]] = []
        for name in cand:
            idx = spec.index_of(name)
            keys.append((0, idx) if idx is not None else (1, 0))
        return sorted(range(len(cand)), key=lambda i: keys[i])
    src = strs if collide else bases
    return sorted(range(len(src)), key=lambda i: natural_key(src[i], whole=collide))


def natural_key(name: str, *, whole: bool = False) -> Tuple:
    """A sort key that compares digit runs numerically — ``img9`` before ``img10``.

    Each segment becomes a ``(kind, value)`` pair so a number and a literal never compare
    against each other (which would raise on Python 3). ``whole=True`` keeps the directory
    part, for the case :func:`order` documents.
    """
    text = str(name) if whole else os.path.basename(str(name))
    return tuple((0, int(s), "") if s.isdigit() else (1, 0, s)
                 for s in segments(text))


def compile_pattern(pattern: str) -> Optional[SequenceSpec]:
    """A user-typed ``pattern`` containing exactly one :data:`SEQ_TOKEN` -> a
    :class:`SequenceSpec`, or ``None`` if it does not contain exactly one.

    This is the override half of the GUI dialog: whatever :func:`detect` decided, the user
    can retype the stem and move the ``{}``, and the result is the same kind of object the
    detector returns — so exactly one code path expands it.
    """
    text = os.path.basename(str(pattern))
    if text.count(SEQ_TOKEN) != 1:
        return None
    head, tail = text.split(SEQ_TOKEN)
    # The token must stand for a WHOLE numeric field. `t00{}` -- a token glued to literal
    # digits -- is refused rather than honoured, because the two would segment into one
    # run and the pattern would then match `t99` as readily as `t00099`: a stem the user
    # typed precisely to exclude things. Refusing says so; matching it does not.
    if head[-1:].isdigit() or tail[:1].isdigit():
        return None
    # A placeholder digit stands in for the counting field so the stem segments exactly as
    # a real member's does; `index_of` overwrites the value and never reads this one.
    segs = segments(head + "0" + tail)
    field = len(segments(head))
    if field >= len(segs) or not segs[field].isdigit():
        return None
    return SequenceSpec(field=field, segments=segs)


def scan(path: str, spec: Optional[SequenceSpec] = None) -> Tuple[List[str], Optional[SequenceSpec]]:
    """Every sibling of ``path`` belonging to its sequence, in sequence order.

    ``path`` is one member the user picked; its directory is listed and each entry is
    tested against ``spec`` (or against a spec detected from the directory's own
    like-named files when ``spec`` is ``None``). Returns ``(paths, spec)``; ``paths`` is
    ``[path]`` alone when nothing matches, which is a correct answer — a single file is a
    sequence of one — and lets the caller report "1 file" rather than raise.

    Only files whose extension matches ``path``'s are considered, so a folder holding
    ``run_t001.nd2`` beside ``run_t001.xml`` sidecars yields three ND2s, not six entries.
    """
    path = str(path)
    folder = os.path.dirname(path) or "."
    ext = os.path.splitext(path)[1].lower()
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return [path], spec
    same_ext = [e for e in entries
                if os.path.splitext(e)[1].lower() == ext
                and os.path.isfile(os.path.join(folder, e))]
    if spec is None:
        base = os.path.basename(path)
        # Detect against only the files that SHARE this one's shape -- an unrelated
        # acquisition in the same folder would otherwise collapse the shape test and
        # return None for a sequence that is plainly there.
        shape = _shape(segments(base))
        spec = detect([e for e in same_ext if _shape(segments(e)) == shape])
    if spec is None:
        return [path], None
    hits: List[Tuple[int, str]] = []
    for e in same_ext:
        idx = spec.index_of(e)
        if idx is not None:
            hits.append((idx, os.path.join(folder, e)))
    if not hits:
        return [path], spec
    hits.sort(key=lambda kv: kv[0])
    return [p for _, p in hits], spec


def describe(names: Sequence[str]) -> str:
    """One line naming the sequence — what the node card and the scan dialog both show.

    Deliberately reports the COUNT and the endpoints rather than the whole list: a
    120-file series is the case this feature exists for, and a card that printed 120
    names would be unreadable exactly when it matters most.
    """
    bases = [os.path.basename(str(n)) for n in names]
    if not bases:
        return "no files"
    if len(bases) == 1:
        return bases[0]
    spec = detect(bases)
    ordered = [bases[i] for i in order(bases)]
    span = f"{ordered[0]} .. {ordered[-1]}"
    if spec is None:
        return f"{len(bases)} files, {span} (natural order - no counting field found)"
    return f"{len(bases)} files, {spec.pattern()}, {span}"


__all__ = ["SEQ_TOKEN", "SequenceSpec", "segments", "detect", "order", "natural_key",
           "compile_pattern", "scan", "describe"]
