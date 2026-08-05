"""The placement RECIPE ENTRY — one builder, shared by every node that co-registers two files.

Two nodes place a second Dataset inside a first one's field by absolute stage position, and a
third builder lives in the Viewer:

* ``view.overlay`` stamps an entry so the Viewer can composite the secondary at draw time;
* ``channel.merge`` samples through an entry to serve the secondary's pixels as real channels;
* :meth:`nodelab_v2.runner.EngineRunner._replan` rebuilds one at DISPLAY time, because a
  stamped plan is a cache keyed on the primary's geometry AT the node and a Stitch or a Crop
  afterwards changes exactly that.

They must not drift, so the entry has exactly one constructor and it lives here rather than in
any one node's module — the catalog forbids a node module importing another (importing executes
it, which registers that node early and welds the two fingerprints together).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def already_stage_placed(ds: Any) -> bool:
    """Whether this Dataset's pixels have ALREADY been laid out in stage coordinates.

    ``flip_x``/``flip_y`` describe how the camera is mounted — whether image ``+x`` runs along
    stage ``+x`` — and they exist because no file records it. But ``util.stitch`` has to answer
    that same question to place its tiles, and once it has, its canvas IS in stage space:
    applying the flip again mirrors the whole mosaic inside its own footprint. Measured on the
    WellA3 pair, a stitched 640 secondary lands **784 µm** (456 px at the primary's 1.718 µm/px)
    out in x with the flip on, against 17 µm with it off.

    So the flip is not a preference for such an input, it is inapplicable — which is why this is
    derived from the data rather than left to a default the user has to discover by eye
    (`wire-node-v2` §7b). Read off the sampling provenance, which is where ``util.stitch``
    records itself (``stitch[stage[49 tiles]]``); a ``grid`` layout is not stage-placed and is
    deliberately not matched.
    """
    try:
        from nodegraph.catalog._shared.sampling import SAMPLING_KEY
        stamps: Tuple[str, ...] = tuple(ds.metadata.get(SAMPLING_KEY, ()))
    except Exception:  # noqa: BLE001 — a placement hint, never a failure
        return False
    return any(str(s).startswith("stitch[stage") for s in stamps)


def overlay_entry(node_id: str, plan: Any, settings: Dict[str, Any],
                  context: Any = ()) -> Dict[str, Any]:
    """One recipe entry from a resolved plan — the SINGLE builder.

    Three callers, and they must not drift. A compute stamps this onto the payload (the record:
    it serializes, it survives a checkpoint, and ``resample``/``channel.merge`` sample from it);
    the Viewer's runner builds it again at DISPLAY time against whatever geometry is actually
    being viewed.

    Plain JSON-shaped values only: the entry rides in ``metadata``, which folds into
    ``output_fingerprint``, so anything exotic here would make the fingerprint depend on object
    identity.
    """
    return {
        "node": node_id,
        "blend": str(settings.get("blend", "add")),
        # `opacity` / `wipe_pos` / `flicker_hz` are PRESENTATION sockets and deliberately do NOT
        # appear here. They are excluded from the recipe hash so dragging one is a repaint rather
        # than a re-run, and a value that is not in the key must not be in the payload either — a
        # memo hit would otherwise hand back a recipe stamped with the value the slider used to
        # have. The GUI reads them live from the document.
        "flip_x": bool(settings.get("flip_x", True)),
        "flip_y": bool(settings.get("flip_y", False)),
        "scale": plan.scale,
        "t_shift": int(settings.get("t_shift", 0)),
        "offset_um": [float(v) for v in settings.get("offset_um", (0.0, 0.0, 0.0))],
        # int keys would not survive a JSON round-trip through a saved checkpoint, so the
        # per-field maps are emitted as sorted lists of pairs.
        "tiles": [[m, [[j, round(f, 6)] for j, f in plan.tiles[m]]]
                  for m in sorted(plan.tiles)],
        "coverage": [[m, round(plan.coverage[m], 6)] for m in sorted(plan.coverage)],
        "z_offset_um": [[m, plan.z_offset_um[m]] for m in sorted(plan.z_offset_um)],
        "t_pairs": [[t, j, e] for t, j, e in plan.t_pairs],
        "placed_by": plan.placed_by,
        # The CONTEXT extent, recorded per field so the Viewer can widen to it without
        # recomputing the placement. Display-only by construction.
        "context": context,
        "warnings": list(plan.warnings),
        # An override's warnings ride IN the note, not merely beside it: the note is what reaches
        # the status line under the picture, and "placed by index" is the one thing a person
        # looking at that picture must not have to go and look up.
        "note": plan.describe(0) + (
            "  ·  " + plan.warnings[0] if (plan.placed_by != "stage" and plan.warnings)
            else ""),
    }


def handedness_for(sec: Any, flip_x: bool, flip_y: bool) -> Tuple[bool, bool, Tuple[str, ...]]:
    """``(flip_x, flip_y, warnings)`` for this secondary — both forced off, with a reason, when
    it is already stage-placed (:func:`already_stage_placed`)."""
    if not already_stage_placed(sec):
        return bool(flip_x), bool(flip_y), ()
    if not (flip_x or flip_y):
        return False, False, ()
    return False, False, (
        "the secondary is an already-stitched canvas, which is laid out in stage coordinates "
        "— so Flip X/Y are inapplicable here and were NOT applied. Applying them again would "
        "mirror the mosaic inside its own footprint (measured 784 µm out on the WellA3 pair). "
        "The stitch itself is where that handedness belongs.",)


__all__ = ["overlay_entry", "already_stage_placed", "handedness_for"]
