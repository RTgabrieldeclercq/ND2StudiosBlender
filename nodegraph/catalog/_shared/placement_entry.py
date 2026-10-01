"""The placement RECIPE ENTRY — one builder, shared by every node that co-registers two files.

Two nodes place a second Dataset inside a first one's field by absolute stage position, and a
third builder lives in the Viewer:

* ``view.overlay`` stamps an entry so the Viewer can composite the secondary at draw time;
* ``util.merge`` (``merge_axis="C"``, retired ``channel.merge``'s successor) samples through
  an entry to serve a later input's pixels as real channels;
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


def overlay_settings(src: Any, modes: Dict[str, Any]) -> Dict[str, Any]:
    """An Overlay node's params + modes → everything placement needs — the SINGLE reader.

    Two callers read an Overlay's settings: its compute (``src`` = the ``EvalContext``), and
    the Viewer's runner re-planning at display time (``src`` = the document's node record,
    :meth:`nodelab_v2.runner.EngineRunner._replan`). Anything with a ``.params`` mapping. The
    runner used to carry its own hard-coded list, which meant every new setting was honoured
    by the stamped record and silently ignored by the picture. One reader, so that cannot
    happen again. (Read as ``src.params.get("key")`` literally, which is also the form the
    socket-contract gate follows through a helper.)

    Raises ``ValueError`` on malformed pins (:func:`nodegraph.placement.parse_pins`); the
    compute lets that reach the node card, the runner falls back to the stamped entry.
    """
    from nodegraph.placement import parse_pins
    modes = dict(modes or {})
    return {
        "blend": str(modes.get("blend", "add")),
        "t_shift": int(src.params.get("t_shift", 0) or 0),
        "offset_um": (float(src.params.get("offset_z", 0.0) or 0.0),
                      float(src.params.get("offset_y", 0.0) or 0.0),
                      float(src.params.get("offset_x", 0.0) or 0.0)),
        "min_coverage": float(src.params.get("min_coverage", 0.0) or 0.0),
        "on_unplaceable": str(modes.get("unplaceable", "refuse")),
        "t_pairing": str(modes.get("t_pairing", "index")),
        "rate": float(src.params.get("rate", 0.0) or 0.0),
        "t_pins": parse_pins(src.params.get("t_pins", ""), axis="t"),
        "z_pins": parse_pins(src.params.get("z_pins", ""), axis="z"),
        "z_sampling": str(modes.get("z_sampling", "nearest")),
        "flip_x": bool(src.params.get("flip_x", True)),
        "flip_y": bool(src.params.get("flip_y", False)),
    }


def plan_kwargs(s: Dict[str, Any]) -> Dict[str, Any]:
    """The :func:`nodegraph.placement.plan_placement` keywords from :func:`overlay_settings`."""
    return {"t_shift": s["t_shift"], "offset_um": s["offset_um"],
            "min_coverage": s["min_coverage"], "on_unplaceable": s["on_unplaceable"],
            "t_pairing": s["t_pairing"], "rate": s["rate"], "t_pins": s["t_pins"],
            "z_pins": s["z_pins"]}


def overlay_entry(node_id: str, plan: Any, settings: Dict[str, Any],
                  context: Any = ()) -> Dict[str, Any]:
    """One recipe entry from a resolved plan — the SINGLE builder.

    Three callers, and they must not drift. A compute stamps this onto the payload (the record:
    it serializes, it survives a checkpoint, and ``resample``/``util.merge`` sample from it);
    the Viewer's runner builds it again at DISPLAY time against whatever geometry is actually
    being viewed.

    Plain JSON-shaped values only: the entry rides in ``metadata``, which folds into
    ``output_fingerprint``, so anything exotic here would make the fingerprint depend on object
    identity.

    The T map, the sub-tick count, the Z sampling and the Z pins are written ONLY when they
    differ from the historical behaviour, so an overlay that uses none of them stamps exactly
    the entry it always did.
    """
    entry = _base_entry(node_id, plan, settings, context)
    t_map = getattr(plan, "t_map", None)
    if t_map is not None:
        entry["t_map"] = {"knots": [[float(u), float(s)] for u, s in t_map["knots"]],
                          "rate": float(t_map["rate"]), "n_src": int(t_map["n_src"])}
        entry["sub_ticks"] = int(getattr(plan, "sub_ticks", 1) or 1)
    if str(settings.get("z_sampling", "nearest")) == "linear":
        entry["z_sampling"] = "linear"
    z_pins = settings.get("z_pins") or ()
    if z_pins:
        entry["z_pins"] = [[int(r[0]), int(r[1]),
                            None if r[2] is None else float(r[2]),
                            None if r[3] is None else float(r[3])] for r in z_pins]
    return entry


def _base_entry(node_id: str, plan: Any, settings: Dict[str, Any],
                context: Any) -> Dict[str, Any]:
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


def z_pick(entry: Dict[str, Any], sec_md: Any, sec_axes: Any, sec_m: int,
           pri_md: Any, pri_axes: Any, pri_k: int, z_um: Any) -> Any:
    """``[(secondary slice, weight)]`` for primary plane ``pri_k`` under this entry's Z
    settings — the ONE call the Viewer's compositor and the resample bake both make, so the
    picture and the baked channel sample the same planes with the same weights."""
    from nodegraph.placement import _z_step, secondary_z_weights
    dz = float((entry.get("offset_um") or (0.0, 0.0, 0.0))[0])
    return secondary_z_weights(
        sec_md, sec_axes, int(sec_m), z_um, dz=dz,
        linear=str(entry.get("z_sampling", "nearest")) == "linear",
        z_pins=tuple(tuple(r) for r in (entry.get("z_pins") or ())),
        pri_k=int(pri_k), pri_step_um=_z_step(pri_md, pri_axes))


__all__ = ["overlay_entry", "already_stage_placed", "handedness_for", "overlay_settings",
           "plan_kwargs", "z_pick"]
