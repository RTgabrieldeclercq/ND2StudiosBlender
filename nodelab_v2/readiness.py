"""Readiness — can this node run as wired, and if not, what is missing and what fixes it.

Qt-free (the selftest reaches it through the document seam). The inspector paints the
result at the top of the node's properties: each :class:`Problem` names the input it is
about, which the panel highlights in red, and carries :class:`Suggestion` rows — nodes
whose output would satisfy the gap — which the panel turns into *Add* buttons and the
window turns into a node inserted upstream and wired in.

Seven kinds of problem, in the order they are reported:

* ``unbound`` — a ``page.input`` whose Source names no output an earlier page offers
  (V4.00): nothing else about it can be judged, so it is the only problem reported.
* ``duplicate_output`` — a ``page.output`` whose name another Output on the same page
  already uses, so a later page could not tell them apart (an unnamed Output is a
  ``validation`` problem: nothing can pick it).
* ``unwired`` — the node's primary Dataset input has no wire. Nothing else can be judged
  until it does, so this is the only problem reported when it holds.
* ``domain`` — a domain the node reads (``reads_domains``, resolved against its mode
  state) that nothing upstream produces: the red domain chip on the card, explained, with
  the nodes that produce it ranked by :data:`PREFERRED` (the usual producer first, then
  the rest of the registry that adds it, by label).
* ``empty`` — a drawn-shape socket the node REFUSES empty, with the Dataset input that
  is the richer alternative and the node to feed it (:data:`EMPTY_PICKS`). Declared per
  (op, socket) rather than inferred: whether an empty drawing means "whole frame" (ROI
  Mask) or "nothing to compare against" (Subtract Background) is the node's own semantics.
* ``validation`` — the 3D lever on data known to have one plane (the card's red badge).
* ``unpublished`` — a HINT (``severity="hint"``, never blocks :func:`ready`; V4.00 step 11):
  on a page whose kind feeds later pages, a loader nothing publishes yet, or the terminal
  node of a page that has no Page Output at all — later pages read this page by its
  Outputs, so the suggestion appends one.

A :class:`Suggestion` is an ``action``: ``add`` inserts the node upstream and wires it in
(the original), ``append`` adds it downstream (a Page Output after a node), ``set_param``
writes ``value`` into ``param`` (bind an unbound Page Input, name an unnamed Output) —
one click each, in the inspector.

What this does NOT do: run anything, read pixels, or guess at values. A node with no
problems may still fail at pull time for a reason only its compute can see; this is the
edit-time answer, the same information the card's chips and badges carry, in words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nodegraph import roles as R
from nodegraph.domains import Domain
from nodegraph.registry import NODES
from nodegraph.sockets import SocketType
from nodelab_v2.ops import (
    HIDDEN_OP_PREFIXES, LOAD_OP, PAGE_INPUT_OP, PAGE_NAME_KEY, PAGE_OUTPUT_OP, PAGE_SOURCE_KEY)


@dataclass(frozen=True)
class Suggestion:
    """One click that fixes (part of) a problem. ``action="add"`` (the original): a node of
    ``op_key`` that would supply what is missing, inserted upstream and wired into
    ``wire_to``; ``"append"``: a node of ``op_key`` added DOWNSTREAM, fed from this node's
    ``wire_to`` output; ``"set_param"``: write ``value`` into this node's ``param``
    (``op_key`` is then ``""``). ``reason`` is the one-line why, the button's tooltip."""
    op_key: str
    label: str
    wire_to: str
    reason: str = ""
    action: str = "add"          # "add" | "append" | "set_param"
    param: str = ""
    value: Any = None


@dataclass(frozen=True)
class Problem:
    kind: str                    # "unbound" | "duplicate_output" | "unwired" | "domain" |
                                 # "empty" | "validation" | "unpublished"
    socket: Optional[str]        # the input this is about (highlighted); None = node-level
    message: str
    suggestions: Tuple[Suggestion, ...] = ()
    severity: str = "error"      # "error" blocks ready(); "hint" is advice


#: the usual producer of each structure domain, first. Anything else in the registry that
#: adds the domain follows, by label. A domain not listed here is ranked by label alone.
PREFERRED: Dict[Domain, Tuple[str, ...]] = {
    Domain.LABEL: ("analysis.label", "analysis.segment", "analysis.draw_regions"),
    Domain.VOXEL: ("analysis.threshold", "analysis.segment", "analysis.histogram_threshold",
                   "analysis.roi_mask"),
    Domain.POINT: ("detect.spots", "detect.particles", "transform.label_to_points"),
    Domain.TRACK: ("track.link", "track.objects"),
}

#: ``(op_key, shapes socket) -> (the Dataset input that is the richer alternative, the node
#: to feed it, what the node needs it for)``. Only sockets the node REFUSES empty belong
#: here — an empty ROI Mask means "whole frame" and is not a problem.
EMPTY_PICKS: Dict[Tuple[str, str], Tuple[str, str, str]] = {
    ("enhance.subtract_background", "shapes"):
        ("regions", "analysis.draw_regions",
         "a sample of pure background — draw it here, or on a Draw Regions node wired into "
         "`Background regions` (per frame, and visible on its own node)"),
    ("analysis.draw_regions", "shapes"):
        ("", "", "the regions themselves — nothing is drawn yet, so the output layer is empty"),
}

#: how many producers a domain problem offers
MAX_SUGGESTIONS = 3

#: ops that END a chain on purpose — a Viewer, a plot, a file writer, a page boundary — so
#: a page whose last node is one of these is not "unpublished"
_SINK_PREFIXES = ("view.", "plot.", "io.write", "page.")


def _defaults(doc, op_key: str) -> Dict[str, Any]:
    """What the page would give a new ``op_key`` node (:attr:`GraphDocument.node_defaults`);
    ``{}`` outside a workspace."""
    try:
        return dict(doc.node_defaults(op_key) or {})
    except Exception:                       # noqa: BLE001 — a bare document
        return {}


def _visible(op_key: str) -> bool:
    return not op_key.startswith(HIDDEN_OP_PREFIXES)


def producers_of(domain: Domain, kind: Optional[str] = None) -> List[Any]:
    """Visible node specs whose output adds ``domain``, preferred ones first — on a page of
    ``kind``, only those that page offers (V4.00 step 5)."""
    pref = PREFERRED.get(domain, ())
    specs = [s for s in NODES.all()
             if domain in getattr(s, "adds_domains", frozenset()) and _visible(s.op_key)
             and R.op_in_page(s.op_key, kind)]

    def rank(s):
        try:
            return (0, pref.index(s.op_key), s.label)
        except ValueError:
            return (1, 0, s.label)
    return sorted(specs, key=rank)


def _is_source(spec) -> bool:
    return not any(s.type is SocketType.DATASET for s in getattr(spec, "inputs", ()))


def _wired_inputs(doc, node_id: str) -> set:
    return {e[3] for e in doc.edges if e[2] == node_id}


def _empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        return s in ("", "[]", "{}", "null")
    try:
        return len(v) == 0
    except TypeError:
        return False


def problems(doc, node_id: str) -> List[Problem]:
    """Everything that stops ``node_id`` from running as it is wired, most blocking first.
    Empty when the node is ready (as far as the edit-time view can tell)."""
    rec = doc.nodes.get(node_id)
    spec = rec.spec() if rec is not None else None
    if rec is None or spec is None:
        return []
    out: List[Problem] = []
    active = list(doc.input_specs(node_id))
    ds_in = [s for s in active if s.type is SocketType.DATASET]
    wired = _wired_inputs(doc, node_id)

    # 0. page boundaries (V4.00): an Input must name an output an earlier page offers; an
    #    Output must carry a name no sibling Output on this page already uses
    if spec.op_key == PAGE_INPUT_OP:
        value = str(rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
        choices = list(doc.source_choices(node_id))
        offered = {v for v, _label in choices}
        if value in offered:
            return []
        if not value:
            msg = "no source chosen — pick a named output of an earlier page in `Source`"
        elif not offered:
            msg = (f"`{value}` cannot be read here — no earlier page has a named Page "
                   f"Output yet")
        else:
            msg = (f"`{value}` is not an output any earlier page offers — renamed, deleted, "
                   f"or on a page that cannot feed this one")
        # one click per offered Output, the page's own default first (V4.00 step 11)
        default = str(_defaults(doc, PAGE_INPUT_OP).get(PAGE_SOURCE_KEY, "") or "")
        ordered, seen = [], set()
        for c in sorted(choices, key=lambda c: c[0] != default):
            if c[0] not in seen:            # two Outputs of one name: one button
                seen.add(c[0])
                ordered.append(c)
        sugg = tuple(Suggestion(
            "", f"Bind to {label}", PAGE_SOURCE_KEY,
            f"read `{label}`" + (" — the nearest earlier page's newest Output"
                                 if v == default else ""),
            action="set_param", param=PAGE_SOURCE_KEY, value=v)
            for v, label in ordered[:MAX_SUGGESTIONS])
        return [Problem("unbound", PAGE_SOURCE_KEY, msg, sugg)]
    if spec.op_key == PAGE_OUTPUT_OP:
        name = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
        fresh = str(_defaults(doc, PAGE_OUTPUT_OP).get(PAGE_NAME_KEY, "") or "") or "out"
        rename = (Suggestion("", f"Name it “{fresh}”", PAGE_NAME_KEY,
                             "a later page's Page Input picks this Output by its name",
                             action="set_param", param=PAGE_NAME_KEY, value=fresh),)
        if not name:
            out.append(Problem("validation", PAGE_NAME_KEY,
                               "this output has no name — later pages pick outputs by name",
                               rename))
        elif any(r.id != node_id and r.op_key == PAGE_OUTPUT_OP
                 and str(r.params.get(PAGE_NAME_KEY, "") or "").strip() == name
                 for r in doc.nodes.values()):
            out.append(Problem("duplicate_output", PAGE_NAME_KEY,
                               f"another Page Output on this page is also named `{name}` — "
                               f"a later page could not tell them apart", rename))

    # 1. the primary Dataset input — the first declared, never a display-only tap
    primary = next((s for s in ds_in if not getattr(s, "view_source", False)), None)
    if primary is not None and primary.name not in wired and not _is_source(spec):
        has_source = any(_is_source(r.spec()) for r in doc.nodes.values()
                         if r.spec() is not None and r.id != node_id)
        sugg: Tuple[Suggestion, ...] = ()
        if not has_source and R.op_in_page("io.load", getattr(doc, "page_kind", None)):
            sugg = (Suggestion("io.load", "Load", primary.name,
                               "the graph has no source yet — a Load node reads the file "
                               "this node will work on"),)
        elif not has_source:
            # a later page reads its data from an earlier one, not from a file (V4.00)
            sugg = (Suggestion(PAGE_INPUT_OP, "Page Input", primary.name,
                               "this page has no source yet — a Page Input reads a named "
                               "Output of an earlier page"),)
        out.append(Problem(
            "unwired", primary.name,
            f"nothing is wired into `{primary.label or primary.name}` — this node has no "
            f"image to work on", sugg))
        return out                      # nothing downstream of this can be judged yet

    # 2. domains the node reads that nothing upstream produces
    for d in sorted(doc.missing_domains(node_id), key=lambda d: d.value):
        prods = producers_of(d, getattr(doc, "page_kind", None))[:MAX_SUGGESTIONS]
        where = primary.name if primary is not None else None
        out.append(Problem(
            "domain", where,
            f"needs {d.value} structure upstream and nothing wired in produces it",
            tuple(Suggestion(
                s.op_key, s.label, where or "data",
                f"{s.label} adds {d.value} to the wire") for s in prods)))

    # 3. drawn-shape sockets the node refuses empty, with their richer alternative
    active_names = {s.name for s in active}
    for (op, sock), (alt, alt_op, need) in EMPTY_PICKS.items():
        if op != spec.op_key or sock not in active_names:
            continue
        if alt and alt in wired:
            continue
        if not _empty_value(rec.params.get(sock)):
            continue
        s = spec.input(sock)
        label = (s.label or s.name) if s is not None else sock
        sugg = ()
        alt_spec = NODES.get(alt_op) if alt and alt_op else None
        if alt_spec is not None:
            a = spec.input(alt)
            alt_label = (a.label or a.name) if a is not None else alt
            sugg = (Suggestion(alt_op, alt_spec.label, alt,
                               f"wired into `{alt_label}`, fed from this node's own image"),)
        out.append(Problem("empty", sock, f"`{label}` is empty — the node needs {need}", sugg))

    # 4. the 3D lever on known single-plane data
    try:
        z = int(doc.env(node_id).axes.z)
    except Exception:                       # noqa: BLE001 — an un-propagated node
        z = 0
    if spec.has_dim_lever() and rec.state().get("dim") == "3D" and z == 1:
        out.append(Problem(
            "validation", None,
            "set to 3D but the incoming data has a single Z plane — switch the lever to 2D "
            "or feed it a stack"))

    # 5. (a hint) nothing on this page is named for later pages yet (V4.00 step 11)
    hint = _unpublished(doc, node_id, spec)
    if hint is not None:
        out.append(hint)
    return out


def _unpublished(doc, node_id: str, spec) -> Optional[Problem]:
    """The ``unpublished`` hint, or ``None``: only on a typed page whose kind may hold a Page
    Output; for a loader with no Page Output on its wire, or for the terminal node (a
    Dataset output, no outgoing wire, not a sink) of a page that has no Page Output at all."""
    kind = getattr(doc, "page_kind", None)
    if not kind or kind == R.FREE_PAGE or not R.op_in_page(PAGE_OUTPUT_OP, kind):
        return None
    if spec.op_key.startswith(_SINK_PREFIXES):
        return None
    out_sock = next((s.name for s in getattr(spec, "outputs", ())
                     if s.type is SocketType.DATASET), None)
    if out_sock is None:
        return None
    outgoing = [e for e in doc.edges if e[0] == node_id]
    if spec.op_key == LOAD_OP:
        if any(getattr(doc.nodes.get(e[2]), "op_key", "") == PAGE_OUTPUT_OP for e in outgoing):
            return None
        msg = "this image is not named for later pages yet — a Page Output publishes it"
    else:
        if outgoing:
            return None
        if any(r.op_key == PAGE_OUTPUT_OP for r in doc.nodes.values()):
            return None
        if not any(not r.op_key.startswith("page.") for r in doc.nodes.values()
                   if r.id != node_id):
            pass                            # a one-node page: still worth saying
        msg = "later pages read this page by its Outputs — nothing here is named yet"
    return Problem("unpublished", None, msg, (Suggestion(
        PAGE_OUTPUT_OP, "Page Output", out_sock,
        "names what this node produces so a later page's Page Input can read it",
        action="append"),), severity="hint")


def ready(doc, node_id: str) -> bool:
    """Nothing blocking: hints (``severity="hint"``) do not count."""
    return not any(p.severity == "error" for p in problems(doc, node_id))


def socket_problems(probs: Sequence[Problem]) -> Dict[str, Problem]:
    """``{socket name: its first problem}`` — what the panel highlights."""
    out: Dict[str, Problem] = {}
    for p in probs:
        if p.socket and p.socket not in out:
            out[p.socket] = p
    return out
