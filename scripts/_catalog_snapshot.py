"""Canonical snapshot of the whole node catalog — the behaviour-preservation gate for a
refactor that MOVES node definitions between modules (the per-node split, V2.20).

    python scripts/_catalog_snapshot.py            # check (the default)
    python scripts/_catalog_snapshot.py check      # against the committed baseline
    python scripts/_catalog_snapshot.py save       # re-bless it (only when a
                                                  # catalog change is intended)

The selftest proves the nodes still *work*. This proves the catalog is still the *same
catalog*: every op_key, in the same registration order, with byte-identical socket
declarations (name, type, direction, unit, derive, default, domain, dims, availability,
layer/path/pick annotations, hover prose), the same modes, the same footprint and domain
declarations, and the same compute wired to each key. A mechanical split that drops a
socket's ``unit``, reorders two sockets, silently re-registers a node twice, or attaches the
wrong compute is invisible to a test that only runs a few nodes end-to-end — and every one
of those is a plausible slip when 12,000 lines move.

Deliberately records what must NOT change and omits what must: a callable's
``__module__``/``__qualname__`` is excluded (moving a compute to its own module is the whole
point), while its ``__name__`` is kept, because a *renamed* or *swapped* callable is a real
defect. Registration ORDER is recorded because ``NodeRegistry`` is insertion-ordered and the
link-search menu enumerates it.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fn(v):
    """A callable's identity for snapshot purposes: its name only.

    NOT its module or qualname — a per-node split moves every compute to a new module by
    design, so including those would make the gate fail on the very change it is meant to
    validate. The name still catches the slips that matter: a node wired to the wrong
    compute, or a helper renamed mid-move."""
    if v is None:
        return None
    return getattr(v, "__name__", type(v).__name__)


def _socket(s) -> dict:
    """Every declared field of a SocketSpec, so a dropped annotation cannot slip through.

    Built by reading the dataclass's own field list rather than a hand-written key list:
    a field added to SocketSpec later is then covered automatically instead of silently
    escaping the gate."""
    import dataclasses
    out = {}
    for f in dataclasses.fields(s):
        v = getattr(s, f.name)
        if hasattr(v, "value") and hasattr(v, "name") and not isinstance(v, (str, int)):
            v = f"{type(v).__name__}.{v.name}"            # Enum -> stable string
        elif isinstance(v, frozenset):
            v = sorted(str(x) for x in v)
        elif isinstance(v, dict):
            v = {k: sorted(str(x) for x in vv) if isinstance(vv, (set, frozenset)) else vv
                 for k, vv in sorted(v.items())}
        elif isinstance(v, tuple):
            v = [f"{type(x).__name__}.{x.name}" if hasattr(x, "name") and not isinstance(x, str)
                 else x for x in v]
        elif callable(v):
            v = _fn(v)
        out[f.name] = v
    return out


def _mode(m) -> dict:
    """Every declared field of a ModeSpec.

    Hand-written (unlike :func:`_socket`, which reads the dataclass) so it must be kept in
    step by hand — ``description``/``choice_docs`` were added in V2.21 and are covered here
    for the same reason the socket sweep is exhaustive: a dropped annotation must not slip
    through a refactor. Docs are presentation-only, so a diff in them is a documentation
    change, never a behaviour change."""
    return {
        "name": m.name, "choices": list(m.choices), "default": m.default,
        "label": m.label, "presentation": m.presentation, "role": m.role,
        "derive": m.derive,
        "available_in": (None if not m.available_in else
                         {k: sorted(v) for k, v in sorted(m.available_in.items())}),
        "description": m.description,
        "choice_docs": dict(sorted((m.choice_docs or {}).items())),
    }


def _gran(g):
    if g is None:
        return None
    if isinstance(g, dict):
        return {k: (v.value if hasattr(v, "value") else v) for k, v in sorted(g.items())}
    return g.value if hasattr(g, "value") else g


def _kax(k):
    if k is None:
        return None
    if isinstance(k, dict):
        return {kk: sorted(vv) for kk, vv in sorted(k.items())}
    return sorted(k)


def snapshot() -> dict:
    import nodegraph.nodes as NN
    import nodelab_v2.ops as OPS
    OPS.ensure_ops()                    # the GUI-layer ops belong in the catalog too
    from nodegraph.registry import NODES

    specs = []
    for spec in NODES.all():            # insertion order — deliberately not sorted
        specs.append({
            "op_key": spec.op_key,
            "label": spec.label,
            "category": spec.category,
            "description": spec.description,
            "inputs": [_socket(s) for s in spec.inputs],
            "outputs": [_socket(s) for s in spec.outputs],
            "modes": [_mode(m) for m in spec.modes],
            "granularity": _gran(spec.granularity),
            "kernel_axes": _kax(spec.kernel_axes),
            "footprint_mode": spec.footprint_mode,
            "meta_transform": _fn(spec.meta_transform),
            "extra_layers": _fn(spec.extra_layers),
            "supports_2d": spec.supports_2d,
            "supports_true_3d": spec.supports_true_3d,
            "three_d_fallback": spec.three_d_fallback,
            "reads_domains": sorted(d.value for d in spec.reads_domains),
            # the CONDITIONAL half (V2.22) — recorded separately rather than flattened
            # into the line above, because a flattened union is exactly the over-claiming
            # declaration the field replaced and the snapshot would then be unable to tell
            # "requires both" from "requires one or the other".
            "reads_domains_by_mode": {
                mode: {value: sorted(d.value for d in domains)
                       for value, domains in sorted(per_value.items())}
                for mode, per_value in sorted(spec.reads_domains_by_mode.items())},
            "adds_domains": sorted(d.value for d in spec.adds_domains),
            "has_compute": spec.op_key in NN.COMPUTES,
            "compute": _fn(NN.COMPUTES.get(spec.op_key)),
            # WHICH module registered it. Recorded even though the point of the split is that
            # definitions move, because the alternative leaves a real blind spot: everything
            # else here is deliberately module-agnostic, so a facade that kept registering the
            # whole catalog *alongside* the split modules would produce a byte-identical
            # snapshot while every fingerprint collapsed back to one. Owner is compared
            # loosely (see `_diff`) — it must be a catalog module, not a particular one.
            "owner": NODES.owner(spec.op_key),
        })
    return {
        "n_ops": len(specs),
        "order": [s["op_key"] for s in specs],
        "computes_only": sorted(set(NN.COMPUTES) - {s["op_key"] for s in specs}),
        "specs": {s["op_key"]: s for s in specs},
    }


def _catalog_owner(owner: str) -> bool:
    """Whether ``owner`` is a module the reloader can fingerprint and reload."""
    return bool(owner) and (owner.startswith("nodegraph.catalog.")
                            or owner == "nodegraph.nodes")


def _diff(a: dict, b: dict) -> list:
    """Every difference between two snapshots, as readable paths."""
    out = []
    if a["order"] != b["order"]:
        sa, sb = set(a["order"]), set(b["order"])
        if sa - sb:
            out.append(f"OPS LOST: {sorted(sa - sb)}")
        if sb - sa:
            out.append(f"OPS ADDED: {sorted(sb - sa)}")
        if sa == sb:
            moved = [(i, x, y) for i, (x, y) in enumerate(zip(a["order"], b["order"]))
                     if x != y]
            out.append(f"REGISTRATION ORDER changed at {len(moved)} position(s); "
                       f"first: index {moved[0][0]} {moved[0][1]!r} -> {moved[0][2]!r}")
    if a["computes_only"] != b["computes_only"]:
        out.append(f"computes with no spec: {a['computes_only']} -> {b['computes_only']}")
    for op in sorted(set(a["specs"]) & set(b["specs"])):
        sa, sb = a["specs"][op], b["specs"][op]
        for key in sorted(set(sa) | set(sb)):
            va, vb = sa.get(key, "<missing>"), sb.get(key, "<missing>")
            if va == vb:
                continue
            if key == "owner":
                # A move between modules is the refactor, not a regression. What must NOT
                # happen is an op drifting to an owner that is not a node module at all —
                # ``nodelab_v2.ops`` (unreloadable) or the facade (which would mean the old
                # monolith is still registering) — because such an op is silently
                # un-fingerprinted and a live edit to it serves stale memoized results.
                ok_a, ok_b = _catalog_owner(va), _catalog_owner(vb)
                if ok_a == ok_b:
                    continue
                out.append(f"{op}.owner: {va!r} -> {vb!r} — no longer owned by a node module, "
                           f"so it can no longer be fingerprinted or reloaded")
                continue
            if key in ("inputs", "outputs") and isinstance(va, list) and isinstance(vb, list):
                na = [s.get("name") for s in va]
                nb = [s.get("name") for s in vb]
                if na != nb:
                    out.append(f"{op}.{key}: socket list {na} -> {nb}")
                    continue
                for s_a, s_b in zip(va, vb):
                    for f in sorted(set(s_a) | set(s_b)):
                        if s_a.get(f) != s_b.get(f):
                            out.append(f"{op}.{key}[{s_a.get('name')}].{f}: "
                                       f"{s_a.get(f)!r} -> {s_b.get(f)!r}")
                continue
            out.append(f"{op}.{key}: {va!r} -> {vb!r}")
    return out


def main() -> int:
    # `check`, not `save`. The old default meant that running this bare — the obvious thing to
    # do when you want to know whether the catalog moved — silently overwrote the baseline it
    # exists to defend and then printed a success line. There is no reading of "I ran the gate
    # with no arguments" that means "destroy the reference copy".
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "catalog_baseline.json")
    snap = snapshot()
    if mode == "save":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=1, sort_keys=True)
        print(f"saved {snap['n_ops']} ops to {path}")
        return 0
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)
    diffs = _diff(base, snap)
    if not diffs:
        print(f"CATALOG IDENTICAL — {snap['n_ops']} ops, same order, same declarations")
        return 0
    print(f"CATALOG DIFFERS from {path} — {len(diffs)} difference(s):")
    for d in diffs[:200]:
        print(f"  {d}")
    if len(diffs) > 200:
        print(f"  … and {len(diffs) - 200} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
