"""Node bundles — a palette entry that places SEVERAL pre-wired nodes (2026-10-08).

A complex analysis — DVC, DIC, PIV — is one algorithm but several *sections*, and a
user who wants to see the seed field before paying for the refinement, or re-tune the
strain without re-correlating, needs each section as its own card. A **bundle** is the
catalog's way to offer that as ONE palette row: a named, documented recipe of nodes
with relative positions and the wires between them. Placing it (double-click, or a
drag onto the canvas) drops the member nodes on the page, wires them, and wraps them in
a *frame* (a canvas region, which gets border ports and can become a linked tab), so
what the user holds afterwards is ordinary nodes — each pullable, each editable, saved
as nodes. The engine never sees a bundle; nothing of it survives placement except the
nodes, which is the point: a bundle is a *gesture*, not a node type.

This is deliberately NOT a :mod:`nodegraph.groups` group: a group instance is one opaque
card with one Dataset in and one out, run as a unit, and that hides exactly the stages a
bundle exists to expose.

Qt-free. The GUI (`nodelab_v2.document.GraphDocument.place_bundle`, the palette) reads
:data:`BUNDLES`; the catalog registers into it at import, beside the nodes it names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

__all__ = ["Bundle", "BundleNode", "BundleEdge", "BUNDLES", "BUNDLE_PREFIX",
           "register_bundle", "is_bundle_key", "bundle_of", "validate_bundle"]

#: Every bundle key starts with this, so a palette row / a drag payload that is a bundle
#: is told apart from an op key by prefix alone (op keys are ``<category>.<name>``).
BUNDLE_PREFIX = "bundle:"


@dataclass(frozen=True)
class BundleNode:
    """One member: ``id`` is the bundle-local name (made unique on placement), ``x``/``y``
    the position RELATIVE to the drop point, ``params``/``modes`` the values to place it
    with (anything unset takes the node's own defaults)."""
    id: str
    op_key: str
    x: float = 0.0
    y: float = 0.0
    params: Mapping[str, object] = field(default_factory=dict)
    modes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BundleEdge:
    """A wire between two members, by bundle-local ids and socket names."""
    src: str
    src_socket: str
    dst: str
    dst_socket: str


@dataclass(frozen=True)
class Bundle:
    """A placeable recipe. ``input`` / ``output`` name the member socket a wire the
    bundle is dropped ONTO is spliced into (its source feeds ``input``, ``output`` feeds
    its destination); ``role`` is the ``codemap/node_roles.json`` role whose palette
    section lists the bundle (bundles are not ops, so they do not appear in that file
    themselves); ``frame_title`` names the region the members are placed in."""
    key: str
    label: str
    category: str
    role: str
    description: str
    nodes: Tuple[BundleNode, ...]
    edges: Tuple[BundleEdge, ...]
    input: Tuple[str, str]
    output: Tuple[str, str]
    frame_title: str = ""
    #: Ops that belong to this family but are not placed by default — an optional stage
    #: (PIV Smooth) the user adds where wanted. The palette lists them under the bundle
    #: with the placed members; placement ignores them.
    extra_ops: Tuple[str, ...] = ()

    @property
    def op_keys(self) -> Tuple[str, ...]:
        """The placed members' ops, in placement order, repeats included (a PIV ladder
        places Validate after every pass)."""
        return tuple(n.op_key for n in self.nodes)

    @property
    def family_ops(self) -> Tuple[str, ...]:
        """Every op of the family ONCE, in first-placement order, then the extras — what
        the palette lists under the bundle row."""
        seen: Dict[str, None] = {}
        for op in self.op_keys + tuple(self.extra_ops):
            seen.setdefault(op, None)
        return tuple(seen)


#: The registry, keyed by bundle key, in registration order.
BUNDLES: Dict[str, Bundle] = {}


def is_bundle_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(BUNDLE_PREFIX)


def bundle_of(key: str) -> Bundle:
    try:
        return BUNDLES[key]
    except KeyError:
        raise KeyError(f"no node bundle {key!r}; known: {sorted(BUNDLES)}") from None


def validate_bundle(bundle: Bundle, known_ops=None) -> List[str]:
    """Every structural problem with ``bundle`` as a list of messages (empty = sound):
    the key prefix, unique member ids, edges and the input/output that name members,
    and — when ``known_ops`` (an op-key container, e.g. ``NODES``) is given — members
    whose op is not registered."""
    out: List[str] = []
    if not is_bundle_key(bundle.key):
        out.append(f"bundle key {bundle.key!r} must start with {BUNDLE_PREFIX!r}")
    ids = [n.id for n in bundle.nodes]
    if not ids:
        out.append("a bundle needs at least one node")
    if len(set(ids)) != len(ids):
        out.append(f"duplicate member ids in {bundle.key}: {ids}")
    for e in bundle.edges:
        for end in (e.src, e.dst):
            if end not in ids:
                out.append(f"{bundle.key}: edge {e} names unknown member {end!r}")
    for what, (nid, _sock) in (("input", bundle.input), ("output", bundle.output)):
        if nid not in ids:
            out.append(f"{bundle.key}: {what} names unknown member {nid!r}")
    if known_ops is not None:
        for n in bundle.nodes:
            if n.op_key not in known_ops:
                out.append(f"{bundle.key}: member {n.id!r} uses unregistered op {n.op_key!r}")
        for op in bundle.extra_ops:
            if op not in known_ops:
                out.append(f"{bundle.key}: extra op {op!r} is not registered")
    return out


def register_bundle(bundle: Bundle) -> Bundle:
    """Add ``bundle`` to :data:`BUNDLES`; a structural problem or a re-used key raises
    at import, like a bad ``register_node`` does. Member ops are NOT checked here —
    the catalog registers a bundle in the same module as its nodes, after them, but the
    selftest is where every bundle is proven against the live registry."""
    problems = validate_bundle(bundle)
    if problems:
        raise ValueError("; ".join(problems))
    if bundle.key in BUNDLES:
        raise ValueError(f"node bundle {bundle.key!r} is already registered")
    BUNDLES[bundle.key] = bundle
    return bundle
