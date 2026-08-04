"""Node definition API + the unified registry (nodegraph v2).

A node **type** is a :class:`NodeSpec`: an ``op_key``, a label/category, its input
and output **socket specs**, its in-body **modes** (dropdowns that may reconfigure
sockets), and its data-access declarations. Value inputs carry the metadata-
intelligence fields (``unit``/``derive``) reused from V1.91.

**V2.03 directive additions** (per the 2026-07-21 metadata/2D-3D-toggle directive):

* **Variant sockets** — ``SocketSpec.available_in`` tags a socket as present only in
  certain mode states; ``NodeSpec.active_sockets(state)`` resolves the live socket
  set. This is the declarative backing that lets a mode/toggle reconfigure a node's
  sockets (V2.03 §3 B1), replacing the docstring-only promise.
* **The 2D/3D lever** — a distinguished in-body :class:`ModeSpec` with
  ``role="dim_lever"`` + ``presentation="header"`` and an optional metadata ``derive``
  default; it is a Mode in substance (not a socket, hashed as an in-body option), so
  memo/serialization are unchanged (V2.03 §3 B2).
* **Data-access footprint** — ``granularity`` and ``kernel_axes`` may be static or a
  ``{dim_value: value}`` map resolved per mode state (V2.03 §3 B3), plus the
  true-3D-vs-stack capability flags (V2.03 §3 B3 / H15).
* **Metadata propagation** — ``meta_transform`` declares a node's calibration/axes
  effect for the edit-time MetaEnvelope pass (V2.03 §2 A2; see :mod:`nodegraph.metadata`).

Qt-free; pure standard library.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union,
)

from nodegraph.domains import Domain
from nodegraph.sockets import Direction, Socket, SocketType


# ── data-access footprint (V2.02 §7b / V2.03 §3 B3) ──────────────────────────

class Granularity(Enum):
    """Which axes a node must consume *whole* vs per-element — gates tiling, halo,
    and memo granularity in the scheduler (V2.02 §7b). For a 2D/3D-toggle node this
    is resolved per mode state (2D → TILEABLE/WHOLE_PLANE, 3D → WHOLE_VOLUME)."""

    TILEABLE = "tileable"          # pointwise / small-stencil 2D — honors the tile provider
    WHOLE_PLANE = "whole_plane"    # a full (Y,X) plane per (m,t,z,c)
    WHOLE_VOLUME = "whole_volume"  # a full (Z,Y,X) volume per (m,t,c)
    WHOLE_SERIES = "whole_series"  # the full T series per (m,c)
    MULTI_VIEW = "multi_view"      # several M (tile stitching / multi-view fusion)


#: The reserved mode name for the 2D/3D header lever (V2.03 §3 B2).
DIM_MODE = "dim"

#: Legal :attr:`SocketSpec.path_kind` values (V2.15). Validated at registration because a
#: typo would otherwise fail SILENTLY — the GUI simply wouldn't draw the Browse… button,
#: and the socket would look like an ordinary text field.
PATH_KINDS: FrozenSet[str] = frozenset({"open_file", "save_file", "directory"})

#: Legal :attr:`SocketSpec.pick_kind` values (V2.16) — the INTERACTIVE gesture a param
#: offers instead of (never *only* instead of — the number stays editable) hand-typing:
#:
#: * ``shapes``     — draw rect/ellipse/polygon/brush regions; commits the ROI shape list
#: * ``area``       — click a segmented object, or draw a blob; commits its µm²/µm³
#: * ``level``      — eyedropper: click a pixel; commits its intensity
#: * ``radius``     — drag a circle over a feature; commits its µm radius
#: * ``distance``   — two-click ruler; commits the µm separation
#: * ``grid``       — drag a subset box / its spacing against the real pixel grid
#: * ``rect``       — drag ONE rectangle; commits a whole pixel bounding box (see
#:                    ``pick_bounds``) — the crop window
#: * ``channel``    — adopt the channel the viewer is showing
#: * ``channels``   — tick the channels to keep, by their real names
#: * ``frame``      — adopt the timepoint the viewer is showing
#: * ``zrange``     — adopt the Z planes picked on the viewer's Z strip
#: * ``percentile`` — adopt the contrast window the histogram handles are sitting on
#: * ``gamma``      — adopt the histogram's gamma dot
#:
#: Validated at registration for the same reason as ``path_kind``: an unrecognized value
#: fails SILENTLY (the GUI just wouldn't draw the Pick button), which is indistinguishable
#: from "this param was never annotated".
PICK_KINDS: FrozenSet[str] = frozenset({
    "shapes", "area", "level", "radius", "distance", "grid", "rect",
    "channel", "channels", "frame", "zrange", "percentile", "gamma",
})

#: Pick kinds that write a whole GROUP of sockets from one gesture and therefore require
#: :attr:`SocketSpec.pick_bounds`. ``pick_peer`` covers the two-socket case where the pair is
#: an interval aimed in two phases; these are different — one gesture yields every member at
#: once (a rectangle *is* four numbers), so there is nothing to order and no second phase.
BOUND_PICK_KINDS: FrozenSet[str] = frozenset({"rect", "zrange"})

#: Which SocketTypes each pick kind may annotate. A gesture produces a particular KIND of
#: number — a ruler produces a physical length, an eyedropper an intensity — so putting
#: ``distance`` on a STRING or ``shapes`` on a FLOAT is a declaration bug, not a style
#: choice. ``level`` and ``grid`` legitimately span INT and FLOAT: a threshold is a float
#: on normalized data but an integer count on the raw-histogram node, and a subset grid is
#: px (INT) for DVC/DIC but µm (FLOAT) for the object-field grid.
_PICK_SOCKET_TYPES: Dict[str, Tuple[SocketType, ...]] = {
    "shapes": (SocketType.STRING,),
    "channels": (SocketType.STRING,),
    "channel": (SocketType.INT,),
    "frame": (SocketType.INT,),
    "rect": (SocketType.INT,),
    "zrange": (SocketType.INT,),
    "grid": (SocketType.INT, SocketType.FLOAT),
    "level": (SocketType.INT, SocketType.FLOAT),
    "area": (SocketType.FLOAT,),
    "radius": (SocketType.FLOAT,),
    "distance": (SocketType.FLOAT,),
    "percentile": (SocketType.FLOAT,),
    "gamma": (SocketType.FLOAT,),
}


# ── socket / mode specs ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SocketSpec:
    """Blueprint for one socket on a node type.

    ``available_in`` (V2.03 §3 B1): ``{mode_name: {allowed values}}``. The socket is
    active only when the node's current state matches every entry; ``None`` = always
    active. e.g. ``{"dim": frozenset({"3D"})}`` marks a ``sigma_z`` socket 3D-only.
    """

    name: str
    type: SocketType
    direction: Direction
    label: str = ""
    #: Hover documentation for this one socket: what the param does and **how it moves the
    #: result**. Presentation-only — the GUI renders it as the tooltip on the inspector row
    #: and on the node-card port (:func:`nodelab_v2.node_item.socket_hover_text`). It is
    #: deliberately NOT part of any hash: ``node_recipe_hash`` keys on the *params*, never on
    #: the spec, so writing/editing a description can never invalidate a memo entry or a
    #: saved graph. Empty = no tooltip beyond the name/type/unit line the GUI always shows.
    #: Prose, not a type restatement — "0.4; lower finds more cells" earns its space,
    #: "the bbox threshold" does not.
    description: str = ""
    is_field: bool = False
    multi: bool = False
    unit: str = ""
    derive: str = ""
    default: Any = None
    domain: Optional[Domain] = None   # for a field input: the domain it evaluates on
    dims: int = 3                     # VECTOR arity (2 or 3)
    available_in: Optional[Mapping[str, FrozenSet[str]]] = None
    #: ── layer-name sockets (V2.11) ────────────────────────────────────────────
    #: A STRING socket carrying an attribute-LAYER NAME rather than free text.
    #: ``layer_in`` = it SELECTS a layer that must already exist on the incoming
    #: Dataset, in this domain → the GUI offers the layers actually present.
    #: ``layer_in_mode`` names a Mode whose *value* is the domain instead (only
    #: ``transform.transfer_domain``, whose source domain is its ``from_domain`` lever).
    #: ``layer_out`` = it NAMES a layer this node CREATES, in each listed domain —
    #: several nodes write one name into two domains (``analysis.label`` emits both a
    #: Voxel raster and a Label table called ``labels``), which is why it is a tuple.
    #: These drive :func:`nodegraph.metadata.propagate_meta`'s edit-time layer catalog.
    layer_in: Optional[Domain] = None
    layer_in_mode: str = ""
    layer_out: Tuple[Domain, ...] = ()
    #: ── filesystem-path sockets (V2.15) ──────────────────────────────────────
    #: A STRING socket whose value is a PATH on the machine that runs the graph, not free
    #: text. ``path_kind`` says what the user is choosing — ``"open_file"`` (must exist),
    #: ``"save_file"`` (may not exist yet), ``"directory"`` — and is the ONLY thing the GUI
    #: keys on to put a **Browse…** button beside the field, so a path param that omits it
    #: silently demands hand-typing (the bug this replaced: the inspector matched the
    #: literal socket name ``"path"``, so ``sd_model_path``/``model_path`` had no browser).
    #: ``path_filter`` is the Qt name filter for the file kinds (ignored for a directory);
    #: ``path_hint`` is the one-line placeholder shown while the field is empty — the place
    #: to say what an empty value MEANS, since that differs per socket (``io.load``: the
    #: synthetic demo; a model path: fall back to the pretrained name).
    #: Presentation-only, exactly like :attr:`description`: never hashed, so annotating an
    #: existing socket cannot invalidate a memo entry or a saved graph.
    path_kind: str = ""
    path_filter: str = ""
    path_hint: str = ""
    #: ── interactive picks + closed vocabularies (V2.16) ──────────────────────
    #: ``pick_kind`` names the GESTURE this param offers on the image / histogram / frame
    #: strip instead of a typed number — one of :data:`PICK_KINDS`. It is the ONLY thing
    #: the GUI keys on to draw the **Pick** button under an inspector row and the ◎ glyph
    #: on the node card, so an un-annotated param is typing-only (the crop bounds were the
    #: motivating case: six px spin boxes for a rectangle the user can see).
    #:
    #: ``pick_peer`` names ANOTHER input socket on the same node that the same gesture sets
    #: at the same time. It exists because several of these params are only meaningful as a
    #: PAIR — a spot detector's min/max radius are two rings around one spot, a hysteresis
    #: threshold's low/high are two handles on one histogram, a correlation grid's box and
    #: stride are one lattice. Picking either end arms both, so the user compares them
    #: against each other on the real data instead of typing two numbers that have to agree.
    #:
    #: ``choices`` turns a STRING param into a CLOSED dropdown — for a socket whose legal
    #: values are a fixed published set (the StarDist / CellSAM checkpoint names). Free text
    #: there bought nothing and cost a silent runtime download failure on a typo. Use it
    #: only when the set really is closed; a layer name is open (see :attr:`layer_in`).
    #:
    #: ``vocab`` turns a STRING param into a MULTI-SELECT over a fixed set, serialized as
    #: the comma-joined string the compute already parses. Same reasoning: ``stats`` /
    #: ``metrics`` / ``fields`` are menus of eight-ish tokens, and a typo in a comma list
    #: silently produces no column rather than an error.
    #:
    #: All four are presentation-only, exactly like :attr:`description` and
    #: :attr:`path_kind`: ``node_recipe_hash`` keys on the PARAMS, never on the spec, so
    #: annotating an existing socket cannot invalidate a memo entry or a saved graph. A pick
    #: writes an ordinary param value through the ordinary edit path — there is no second
    #: kind of value, and every picked param stays editable by hand.
    #: ``pick_bounds`` is the ordered, COMPLETE set of sockets a
    #: :data:`BOUND_PICK_KINDS` gesture writes at once — ``("y0","y1","x0","x1")`` for a
    #: crop rectangle, ``("z0","z1")`` for a Z range. Declared identically on every member,
    #: so arming from any row produces the same pick and there is no privileged socket.
    #: Registration additionally requires every member to share one SocketType and one unit,
    #: which is what lets the session convert the whole group with the armed socket's
    #: unit/type instead of carrying a per-name table.
    pick_kind: str = ""
    pick_peer: str = ""
    pick_bounds: Tuple[str, ...] = ()
    choices: Tuple[str, ...] = ()
    vocab: Tuple[str, ...] = ()
    #: ── per-OPTION hover documentation (V2.21) ────────────────────────────────
    #: ``{option: prose}`` for the entries of :attr:`choices` (pick one) or
    #: :attr:`vocab` (pick many) — the dropdown/tick-list counterpart of
    #: :attr:`description`, which can only ever describe the control as a whole.
    #:
    #: A dropdown is where the socket contract's "a live control the user cannot read"
    #: defect hides most easily: ``description`` says what the param selects, and then the
    #: menu offers six tokens (``otsu``, ``li``, ``yen``, …) whose differences are exactly
    #: what the user is trying to decide between. Naming a method is not explaining it, so
    #: every option carries its own line: what it assumes about the data, and which way the
    #: result moves if you pick it. Rendered as the per-item tooltip in the inspector combo
    #: / tick list and in the node card's popup menu, and appended to the param's own hover.
    #:
    #: Keys are validated at registration against the live option list, because a key that
    #: matches nothing fails SILENTLY — the tooltip simply never appears, which is
    #: indistinguishable from never having been written. Presentation-only and memo-neutral
    #: like :attr:`description`.
    choice_docs: Mapping[str, str] = field(default_factory=dict)
    #: PRESENTATION-only (V2.19): the value changes how a result is DRAWN and never what it
    #: contains, so it is excluded from ``node_recipe_hash`` and moving it cannot invalidate
    #: a memo entry. That is the difference between dragging an overlay's opacity being a
    #: repaint and being a re-run of whatever the node does — which for ``view.overlay``'s
    #: ``resample`` mode is re-reading four tiles and resampling them.
    #:
    #: The bar for setting it is absolute: the compute must not let the value reach the
    #: PAYLOAD either, or a memo hit would serve pixels (or a stamped recipe) made with the
    #: old value while the socket shows the new one. A presentation socket's only legitimate
    #: consumer is the GUI, reading it live from the document.
    presentation: bool = False
    kernel_param: bool = False        # influences the spatial kernel (radius/σ): a
    #: NON-Const Field on this socket is spatially varying and breaks tile translation-
    #: invariance + halo sizing, so a consumer must stream at the plane unit, not tiled
    #: (V2.04 §6b Fork B — the kernel-param field gate).

    def instantiate(self) -> Socket:
        return Socket(
            name=self.name, type=self.type, direction=self.direction,
            is_field=self.is_field, multi=self.multi, unit=self.unit,
            derive=self.derive, default=self.default, dims=self.dims,
            domain=self.domain,
        )

    def active_in(self, state: Mapping[str, str]) -> bool:
        """True if this socket is present in mode ``state`` (V2.03 §3 B1)."""
        if not self.available_in:
            return True
        return all(state.get(m) in allowed for m, allowed in self.available_in.items())


@dataclass(frozen=True)
class ModeSpec:
    """An in-body enum dropdown (not a socket; may reconfigure sockets).

    ``presentation`` ("body" default | "header") controls rendering — the 2D/3D lever
    uses "header" (top-right). ``role`` ("dim_lever" for the toggle) lets the engine/
    GUI find it. ``derive`` is a metadata-intelligent default expression (e.g. the
    lever: ``"'3D' if n_z>1 else '2D'"``) — the toggle default is metadata-adaptive
    exactly like a value-socket default (V2.03 §1 / H9).

    ``description`` and ``choice_docs`` (V2.21) are the hover documentation, and a Mode
    needs BOTH: the description says what the dropdown selects, and ``{choice: prose}``
    says what each option does — the difference between ``otsu`` and ``li`` is the whole
    reason the user opened the menu, and a bare token list does not tell them. Same
    contract as :attr:`SocketSpec.description`/:attr:`SocketSpec.choice_docs`:
    presentation-only, memo-neutral, keys checked at registration.

    ``available_in`` (V2.12) gates one Mode on ANOTHER Mode's value, exactly as
    :attr:`SocketSpec.available_in` gates a socket: ``{mode_name: {allowed values}}``,
    ``None`` = always shown. It exists because a node that unifies several algorithms
    behind one ``method`` Mode can have a *second* Mode only some methods read —
    ``analysis.segment``'s ``level`` (the foreground cut) means nothing to its learned
    detectors. Without gating that dropdown would sit there doing nothing, which is the
    same "live-looking control the selected kernel ignores" the node charter forbids for
    sockets. Gating is **edit-time only**: the resolved mode state still carries every
    mode (a hidden one keeps its value) and still folds into the recipe hash, so hiding a
    Mode never changes a memo key — the same rule as a hidden socket.
    """

    name: str
    choices: Sequence[str]
    default: str = ""
    label: str = ""
    presentation: str = "body"
    role: str = ""
    derive: str = ""
    available_in: Optional[Mapping[str, FrozenSet[str]]] = None
    #: What this dropdown selects, and how the choice moves the result (V2.21).
    description: str = ""
    #: ``{choice: prose}`` — one line per option (V2.21). See the class docstring.
    choice_docs: Mapping[str, str] = field(default_factory=dict)

    def resolved_default(self) -> str:
        return self.default or (self.choices[0] if self.choices else "")

    @property
    def is_dim_lever(self) -> bool:
        return self.role == "dim_lever"

    def active_in(self, state: Mapping[str, str]) -> bool:
        """True if this Mode is shown in mode ``state`` (V2.12) — mirrors
        :meth:`SocketSpec.active_in`."""
        if not self.available_in:
            return True
        return all(state.get(m) in allowed for m, allowed in self.available_in.items())


@dataclass(frozen=True)
class NodeSpec:
    op_key: str
    label: str
    category: str = "general"
    inputs: Sequence[SocketSpec] = field(default_factory=tuple)
    outputs: Sequence[SocketSpec] = field(default_factory=tuple)
    modes: Sequence[ModeSpec] = field(default_factory=tuple)
    description: str = ""
    # V2.03 data-access + propagation declarations
    granularity: Union[Granularity, Mapping[str, Granularity], None] = None
    kernel_axes: Union[FrozenSet[str], Mapping[str, FrozenSet[str]], None] = None
    #: Which Mode a **Mapping** ``granularity``/``kernel_axes`` is keyed by. Defaults to
    #: the 2D/3D lever, which is the overwhelmingly common case (``{"2D": …, "3D": …}``)
    #: and what every levered node in the catalog relies on. A node whose footprint is
    #: chosen by a DIFFERENT closed Mode names it here — ``analysis.threshold`` sets
    #: ``"scope"``, because how much of the series its histogram pass must read is decided
    #: by the statistics scope, not by a dimensionality it does not have. Without this the
    #: only expressible declaration would be the single worst case, which would either
    #: over-claim (a per-plane threshold declared MULTI_VIEW) or lie (the shipped
    #: ``TILEABLE`` on a node that read every plane).
    footprint_mode: str = DIM_MODE
    meta_transform: Optional[Callable[..., Any]] = None
    supports_2d: bool = True
    supports_true_3d: bool = True
    three_d_fallback: str = ""        # e.g. "stack_of_2d" when supports_true_3d is False
    # Domain interface (the GUI's socket domain-rail + wire tint; edit-time domain
    # propagation in metadata.propagate_meta). ``reads_domains`` = the domains this
    # node's compute REQUIRES present in the incoming Dataset (a missing one is a
    # validation error); ``adds_domains`` = the domains it PRODUCES/adds to the bundle
    # (unioned into the accumulated set that flows downstream). Both default empty —
    # an un-annotated node is domain-transparent (passes the upstream set through).
    reads_domains: FrozenSet[Domain] = frozenset()
    adds_domains: FrozenSet[Domain] = frozenset()
    #: Layers this node creates that no ``layer_out`` socket can describe (V2.11):
    #: ``(params, modes) -> ((Domain, name), ...)``. Needed by the handful of producers
    #: that name a layer with NO socket at all (``align.drift``/``registration.stabilize``
    #: write the literals ``drift_y``/``drift_x``), derive the name from ANOTHER param
    #: (``analysis.extract_boundary`` -> ``f"{labels}_boundary"``), or write into the layer
    #: their READ socket names (``analysis.measure`` adds Label columns to the raster it
    #: measures). MUST be total — see ``propagate_meta``, which runs on every keystroke.
    extra_layers: Optional[Callable[..., Any]] = None

    def input(self, name: str) -> Optional[SocketSpec]:
        return next((s for s in self.inputs if s.name == name), None)

    def output(self, name: str) -> Optional[SocketSpec]:
        return next((s for s in self.outputs if s.name == name), None)

    # ── mode state ────────────────────────────────────────────────────────────
    def default_state(self) -> Dict[str, str]:
        """The mode state with every mode at its resolved default."""
        return {m.name: m.resolved_default() for m in self.modes}

    def dim_lever(self) -> Optional[ModeSpec]:
        """The 2D/3D lever mode, if this node bears one (V2.03 §3 B2)."""
        return next((m for m in self.modes if m.is_dim_lever), None)

    def has_dim_lever(self) -> bool:
        return self.dim_lever() is not None

    # ── variant resolution (V2.03 §3 B1) ──────────────────────────────────────
    def active_inputs(self, state: Mapping[str, str]) -> tuple:
        return tuple(s for s in self.inputs if s.active_in(state))

    def active_outputs(self, state: Mapping[str, str]) -> tuple:
        return tuple(s for s in self.outputs if s.active_in(state))

    def active_sockets(self, state: Mapping[str, str]) -> tuple:
        return self.active_inputs(state) + self.active_outputs(state)

    def active_modes(self, state: Mapping[str, str]) -> tuple:
        """The Modes shown in ``state`` (V2.12). ``default_state`` deliberately still
        includes the hidden ones: a gated-away Mode keeps its value, so the compute's
        ``__modes__`` lookup and the memo key are unaffected by what the GUI draws."""
        return tuple(m for m in self.modes if m.active_in(state))

    # ── footprint resolution (V2.03 §3 B3) ────────────────────────────────────
    def resolve_granularity(self, state: Mapping[str, str]) -> Optional[Granularity]:
        g = self.granularity
        if isinstance(g, Mapping):
            return g.get(state.get(self.footprint_mode, ""))
        return g

    def resolve_kernel_axes(self, state: Mapping[str, str]) -> Optional[FrozenSet[str]]:
        k = self.kernel_axes
        if isinstance(k, Mapping):
            return k.get(state.get(self.footprint_mode, ""))
        return k

    # ── domain interface ──────────────────────────────────────────────────────
    def out_domains(self, incoming: FrozenSet[Domain]) -> FrozenSet[Domain]:
        """The accumulated domain-set on this node's Dataset output given the set
        ``incoming`` on its Dataset input(s): the upstream set unioned with what this
        node adds (domain-transparent by default)."""
        return incoming | self.adds_domains

    def missing_domains(self, incoming: FrozenSet[Domain]) -> FrozenSet[Domain]:
        """Required domains not present upstream — the GUI's red validation chips."""
        return self.reads_domains - incoming


# ── socket / mode factories (the §11 sketch) ─────────────────────────────────

def InDataset(name: str = "data", *, multi: bool = False, label: str = "",
              available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> SocketSpec:
    return SocketSpec(name, SocketType.DATASET, Direction.IN, label=label,
                      multi=multi, available_in=available_in)


def OutDataset(name: str = "out", *, label: str = "",
               available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> SocketSpec:
    return SocketSpec(name, SocketType.DATASET, Direction.OUT, label=label,
                      available_in=available_in)


def layer_value(sock: Optional[SocketSpec], params: Mapping[str, Any]) -> str:
    """Resolve a layer-name socket to the name it denotes: the user's override, else the
    socket's declared **default**. The empty string counts as unset.

    THE single resolution path, and the reason it lives here rather than in either caller:
    a layer name is resolved in two places that must never disagree — the compute at pull
    time (via ``EvalContext.layer``) and :func:`nodegraph.metadata.propagate_meta` at edit
    time, which predicts the layer catalog the GUI picker offers. They used to be
    independent, with each compute repeating its socket's default inline
    (``ctx.params.get("mask", "mask")``) because the engine passes params as raw
    OVERRIDES and never default-fills them. That made every default a *third* copy —
    socket, compute, envelope rule — with nothing keeping the three in step. Routing both
    through this function leaves exactly one copy: the ``SocketSpec`` default."""
    if sock is None:
        return ""
    value = params.get(sock.name)
    if value is None or value == "":
        value = sock.default
    return value if isinstance(value, str) else ""


def _in_value(t: SocketType):
    def make(name: str, label: str = "", *, default: Any = None, field: bool = True,
             unit: str = "", derive: str = "", domain: Optional[Domain] = None,
             multi: bool = False, dims: int = 3,
             available_in: Optional[Mapping[str, FrozenSet[str]]] = None,
             layer_in: Optional[Domain] = None, layer_in_mode: str = "",
             layer_out: Tuple[Domain, ...] = (),
             kernel_param: bool = False, description: str = "",
             path_kind: str = "", path_filter: str = "",
             path_hint: str = "", pick_kind: str = "", pick_peer: str = "",
             pick_bounds: Sequence[str] = (), choices: Sequence[str] = (),
             vocab: Sequence[str] = (),
             choice_docs: Optional[Mapping[str, str]] = None,
             presentation: bool = False) -> SocketSpec:
        return SocketSpec(name, t, Direction.IN, label=label, is_field=field,
                          unit=unit, derive=derive, default=default, domain=domain,
                          multi=multi, dims=dims, available_in=available_in,
                          layer_in=layer_in, layer_in_mode=layer_in_mode,
                          layer_out=tuple(layer_out), kernel_param=kernel_param,
                          description=description, path_kind=path_kind,
                          path_filter=path_filter, path_hint=path_hint,
                          pick_kind=pick_kind, pick_peer=pick_peer,
                          pick_bounds=tuple(pick_bounds),
                          choices=tuple(choices), vocab=tuple(vocab),
                          choice_docs=dict(choice_docs or {}),
                          presentation=bool(presentation))
    return make


InFloat = _in_value(SocketType.FLOAT)
InInt = _in_value(SocketType.INT)
InBool = _in_value(SocketType.BOOL)
InVector = _in_value(SocketType.VECTOR)
InColor = _in_value(SocketType.COLOR)
InString = _in_value(SocketType.STRING)


def OutValue(name: str, t: SocketType, label: str = "", *, field: bool = True,
             dims: int = 3, description: str = "",
             available_in: Optional[Mapping[str, FrozenSet[str]]] = None) -> SocketSpec:
    """A value OUTPUT socket. ``available_in`` gates it on mode state exactly as it gates
    an input (``NodeSpec.active_outputs``) — ``flow.iterate`` needs it because a variable's
    driver port must be FLOAT or STRING depending on that slot's type, and ``STRING`` has no
    implicit conversion to anything, so one port cannot serve both."""
    return SocketSpec(name, t, Direction.OUT, label=label, is_field=field, dims=dims,
                      description=description, available_in=available_in)


def Mode(name: str, choices: Sequence[str], default: str = "", label: str = "",
         *, presentation: str = "body", role: str = "", derive: str = "",
         available_in: Optional[Mapping[str, FrozenSet[str]]] = None,
         description: str = "",
         choice_docs: Optional[Mapping[str, str]] = None) -> ModeSpec:
    return ModeSpec(name, tuple(choices), default, label,
                    presentation=presentation, role=role, derive=derive,
                    available_in=available_in, description=description,
                    choice_docs=dict(choice_docs or {}))


#: The 2D/3D lever's hover documentation. Canned here rather than written per node because
#: the lever means the SAME thing on all 24 nodes that bear one — it is one control with one
#: contract, and 24 hand-written copies would only differ where one of them was wrong.
#: :func:`DimMode` accepts overrides for the rare node with something extra to say.
DIM_DESCRIPTION = (
    "Whether this node treats each Z plane as its own 2D image or the Z stack as one 3D "
    "volume. It also decides which parameters are live (a Z-only radius or σ appears only "
    "on the 3D side) and how much data the scheduler reads per step. It folds into the memo "
    "key, so the 2D and the 3D result are cached separately instead of overwriting each "
    "other — flipping back is free."
)
DIM_CHOICE_DOCS: Mapping[str, str] = {
    "2D": "Plane by plane. The operation runs independently on every (Y,X) plane of every "
          "z, timepoint, channel and position, so nothing crosses a plane boundary: a "
          "feature spanning three planes is seen as three unrelated 2D features. Cheapest, "
          "tiles, ignores the Z spacing entirely, and the only sane choice for data that is "
          "a single plane or has very coarse Z steps.",
    "3D": "The whole (Z,Y,X) volume at once, using the file's real Z spacing, so a feature "
          "spanning planes stays ONE feature and measurements come out as volumes rather "
          "than per-slice areas. Needs the volume resident and is the slower, more "
          "memory-hungry side; it greys out when the incoming data has z == 1.",
}


def DimMode(*, default: str = "2D",
            derive: str = "'3D' if (n_z or 1) > 1 else '2D'",
            description: str = "",
            choice_docs: Optional[Mapping[str, str]] = None) -> ModeSpec:
    """The 2D/3D header lever (V2.03 §3 B2): an in-body Mode with header rendering,
    ``role="dim_lever"``, and a metadata-adaptive default (z>1 ⇒ 3D). Carries
    :data:`DIM_DESCRIPTION` / :data:`DIM_CHOICE_DOCS` unless a node overrides them."""
    return Mode(DIM_MODE, ["2D", "3D"], default=default, label="2D / 3D",
                presentation="header", role="dim_lever", derive=derive,
                description=description or DIM_DESCRIPTION,
                choice_docs=dict(choice_docs or DIM_CHOICE_DOCS))


# ── the registry ─────────────────────────────────────────────────────────────

#: Frames belonging to the registration plumbing itself, skipped when
#: :func:`_defining_module` walks the stack looking for the module that *authored* a node.
#: ``nodegraph.registry`` is skipped wholesale (``None`` = any function in it); a shared
#: spec+compute wrapper is skipped by ``(module, function)`` so that the module which called
#: it — the node's real author — is the one recorded. Without those entries every node in the
#: catalog would be attributed to whichever module happens to host the wrapper, including the
#: GUI-layer ops that ``nodelab_v2.ops`` defines.
#:
#: Wrappers **declare themselves** via :func:`registration_helper` rather than being listed
#: here by hand. That is not ceremony: this list was previously hard-coded with
#: ``("nodegraph.nodes", "register_node")``, and moving that function into
#: ``nodegraph.catalog._base`` (V2.20) silently re-attributed all 63 catalog nodes to the new
#: host module — a failure whose only symptom is a live reload deleting the wrong nodes.
_PLUMBING: List[Tuple[str, Optional[str]]] = [("nodegraph.registry", None)]


def registration_helper(module: str, func: str) -> None:
    """Declare ``module.func`` to be a registration WRAPPER, not a node's author.

    Call at import time from any module that defines a helper which calls
    :func:`define_node` on someone else's behalf (see ``nodegraph.catalog._base``). Idempotent."""
    entry = (module, func)
    if entry not in _PLUMBING:
        _PLUMBING.append(entry)


def _defining_module() -> str:
    """The dotted name of the module that is registering a node — its **owner**.

    Provenance exists for exactly one consumer: :mod:`nodegraph.hotreload`, which reloads
    a node-defining module and must then drop the specs that module used to define and no
    longer does. Without an owner it could only diff the whole registry, which would delete
    every op registered by some *other* module (``io.load``, ``view.viewer``, the selftest
    fixtures) on the first reload."""
    f = sys._getframe(1)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if not any(mod == m and (fn is None or f.f_code.co_name == fn) for m, fn in _PLUMBING):
            return mod
        f = f.f_back
    return ""


def _check_choice_docs(where: str, docs: Mapping[str, str],
                       options: Sequence[str], what: str) -> None:
    """Validate a ``choice_docs`` mapping against the options it documents (V2.21).

    Shared by the socket and the Mode check because the failure is the same on both and is
    completely silent: a key that matches no option (a renamed method, a typo, a stale entry
    left behind when an option was dropped) draws no tooltip, and no tooltip is exactly what
    an undocumented option looks like. An empty value is caught for the same reason — it
    reads as "documented" to any coverage sweep and shows the user nothing."""
    if not docs:
        return
    if not options:
        raise ValueError(f"{where}: choice_docs without {what} — there are no options for "
                         f"it to document")
    unknown = sorted(set(docs) - set(options))
    if unknown:
        raise ValueError(f"{where}: choice_docs documents {unknown}, which is not in "
                         f"{list(options)} (a key that matches nothing renders no tooltip)")
    blank = sorted(k for k, v in docs.items() if not str(v).strip())
    if blank:
        raise ValueError(f"{where}: choice_docs entries {blank} are empty — delete them or "
                         f"write the prose")


class NodeRegistry:
    """Insertion-ordered ``{op_key: NodeSpec}`` (palette order for un-sorted kinds).

    Each entry additionally records its **owner** (the module that registered it) and a
    monotonic registration **sequence number**. Both serve live reload
    (:mod:`nodegraph.hotreload`): after re-executing a module, the ops it owns whose
    sequence predates the reload are the ones its author deleted, so they are the ones to
    remove. Nothing else reads them, and neither takes part in any hash — a spec's identity
    for memo purposes is its ``op_key`` plus the code fingerprint in
    :mod:`nodegraph.revision`, never its registration order."""

    def __init__(self) -> None:
        self._by_key: Dict[str, NodeSpec] = {}
        self._owner: Dict[str, str] = {}
        self._seq: Dict[str, int] = {}
        self._counter = 0

    def register(self, spec: NodeSpec) -> NodeSpec:
        in_names = frozenset(s.name for s in spec.inputs)
        for s in tuple(spec.inputs) + tuple(spec.outputs):
            if s.path_kind:
                if s.path_kind not in PATH_KINDS:
                    raise ValueError(
                        f"{spec.op_key}.{s.name}: path_kind={s.path_kind!r} is not one of "
                        f"{sorted(PATH_KINDS)}")
                if s.type is not SocketType.STRING:
                    raise ValueError(
                        f"{spec.op_key}.{s.name}: path_kind is only meaningful on a STRING "
                        f"socket, got {s.type.name}")
            self._check_interaction(spec, s, in_names)
        for m in spec.modes:
            self._check_mode(spec, m)
        self._by_key[spec.op_key] = spec
        self._owner[spec.op_key] = _defining_module()
        self._counter += 1
        self._seq[spec.op_key] = self._counter
        return spec

    # ── provenance + atomic replacement (nodegraph.hotreload) ─────────────────
    def mark(self) -> int:
        """The current registration sequence — pass to :meth:`stale_owned` afterwards."""
        return self._counter

    def owner(self, op_key: str) -> str:
        """The module that registered ``op_key`` (``""`` if unknown)."""
        return self._owner.get(op_key, "")

    def keys_owned_by(self, module: str) -> List[str]:
        return [k for k, m in self._owner.items() if m == module]

    def stale_owned(self, modules: Sequence[str], since: int) -> List[str]:
        """Ops owned by ``modules`` that have **not** been re-registered since ``since``.

        After a reload these are precisely the node types the author deleted or renamed:
        re-executing the module re-registers everything it still defines with a fresh
        sequence number, so anything left behind is gone from the source."""
        mods = set(modules)
        return [k for k, m in self._owner.items()
                if m in mods and self._seq.get(k, 0) <= since]

    def remove(self, op_key: str) -> Optional[NodeSpec]:
        """Unregister ``op_key`` (no-op if absent); returns the spec that was dropped."""
        self._owner.pop(op_key, None)
        self._seq.pop(op_key, None)
        return self._by_key.pop(op_key, None)

    def snapshot(self) -> Tuple[dict, dict, dict, int]:
        """An opaque copy of the whole catalog, for :meth:`restore`."""
        return (dict(self._by_key), dict(self._owner), dict(self._seq), self._counter)

    def restore(self, snap: Tuple[dict, dict, dict, int]) -> None:
        """Put the catalog back exactly as :meth:`snapshot` found it — the rollback a
        failed reload needs, so a node file that raises halfway through leaves the running
        session on the catalog it already had rather than on a half-registered one. The
        container identity is preserved (cleared and refilled, never rebound), because the
        GUI and the engine both hold long-lived references to this registry."""
        by_key, owner, seq, counter = snap
        self._by_key.clear()
        self._by_key.update(by_key)
        self._owner.clear()
        self._owner.update(owner)
        self._seq.clear()
        self._seq.update(seq)
        self._counter = counter

    @staticmethod
    def _check_bounds(spec: NodeSpec, s: SocketSpec) -> None:
        """Validate a :data:`BOUND_PICK_KINDS` group (``pick_bounds``).

        Four rules, each protecting one silent failure: the group must be declared (a
        boundless ``rect`` would arm a gesture that writes nothing), every member must exist
        and agree on the group and the kind (a renamed socket would leave one row writing a
        param nothing reads), and every member must share a SocketType and unit — which is
        the invariant the session leans on when it converts the whole group using the armed
        socket's own type, rather than threading a per-name table through every call."""
        where = f"{spec.op_key}.{s.name}"
        if s.pick_kind in BOUND_PICK_KINDS and not s.pick_bounds:
            raise ValueError(f"{where}: pick_kind={s.pick_kind!r} writes a whole group and "
                             f"needs pick_bounds")
        if s.pick_bounds and s.pick_kind not in BOUND_PICK_KINDS:
            raise ValueError(f"{where}: pick_bounds is only meaningful for "
                             f"{sorted(BOUND_PICK_KINDS)}, not {s.pick_kind!r}")
        if not s.pick_bounds:
            return
        if s.name not in s.pick_bounds:
            raise ValueError(f"{where}: pick_bounds {list(s.pick_bounds)} must include the "
                             f"socket itself — it is the COMPLETE group, not the others")
        if s.pick_peer:
            raise ValueError(f"{where}: pick_bounds (one gesture, whole group) and pick_peer "
                             f"(an interval aimed in two phases) are mutually exclusive")
        for name in s.pick_bounds:
            other = spec.input(name)
            if other is None:
                raise ValueError(f"{where}: pick_bounds names {name!r}, which is not an "
                                 f"input socket on this node")
            if other.pick_kind != s.pick_kind or other.pick_bounds != s.pick_bounds:
                raise ValueError(
                    f"{where}: bound member {name!r} declares "
                    f"({other.pick_kind!r}, {list(other.pick_bounds)}) — every member must "
                    f"declare the identical kind and group")
            if other.type is not s.type or other.unit != s.unit:
                raise ValueError(
                    f"{where}: bound member {name!r} is {other.type.name}/{other.unit!r} "
                    f"but {s.name} is {s.type.name}/{s.unit!r} — a bound group must be "
                    f"homogeneous (the session converts it with one type and one unit)")

    @staticmethod
    def _check_interaction(spec: NodeSpec, s: SocketSpec, in_names: FrozenSet[str]) -> None:
        """Validate the V2.16 interaction declarations (``pick_kind`` / ``pick_peer`` /
        ``choices`` / ``vocab``). Every one of these fails SILENTLY if it is wrong — a bad
        ``pick_kind`` just doesn't draw a button, a ``pick_peer`` naming a socket that was
        renamed writes a param nothing reads, and a ``default`` outside ``choices`` gives a
        dropdown that opens on a blank row. Registration is the only place that sees all of
        it at once, so it is the only place that can catch it."""
        where = f"{spec.op_key}.{s.name}"
        if s.pick_kind:
            if s.pick_kind not in PICK_KINDS:
                raise ValueError(f"{where}: pick_kind={s.pick_kind!r} is not one of "
                                 f"{sorted(PICK_KINDS)}")
            if s.direction is not Direction.IN:
                raise ValueError(f"{where}: pick_kind is only meaningful on an INPUT "
                                 f"socket (a pick WRITES a param)")
            allowed = _PICK_SOCKET_TYPES[s.pick_kind]
            if s.type not in allowed:
                raise ValueError(
                    f"{where}: pick_kind={s.pick_kind!r} needs a "
                    f"{'/'.join(t.name for t in allowed)} socket, got {s.type.name}")
        NodeRegistry._check_bounds(spec, s)
        if s.pick_peer:
            if not s.pick_kind:
                raise ValueError(f"{where}: pick_peer={s.pick_peer!r} without a pick_kind "
                                 f"— nothing would ever co-pick it")
            if s.pick_peer == s.name:
                raise ValueError(f"{where}: pick_peer names the socket itself")
            if s.pick_peer not in in_names:
                raise ValueError(f"{where}: pick_peer={s.pick_peer!r} is not an input "
                                 f"socket on this node")
        if s.choices:
            if s.type is not SocketType.STRING:
                raise ValueError(f"{where}: choices is only meaningful on a STRING socket, "
                                 f"got {s.type.name}")
            if s.default not in (None, "") and s.default not in s.choices:
                raise ValueError(f"{where}: default={s.default!r} is not in choices "
                                 f"{list(s.choices)}")
        if s.vocab:
            if s.type is not SocketType.STRING:
                raise ValueError(f"{where}: vocab is only meaningful on a STRING socket, "
                                 f"got {s.type.name}")
            if s.choices:
                raise ValueError(f"{where}: choices (pick ONE) and vocab (pick MANY) are "
                                 f"mutually exclusive")
            unknown = [t for t in str(s.default or "").split(",")
                       if t.strip() and t.strip() not in s.vocab]
            if unknown:
                raise ValueError(f"{where}: default token(s) {unknown} are not in vocab "
                                 f"{list(s.vocab)}")
        _check_choice_docs(where, s.choice_docs, tuple(s.choices) + tuple(s.vocab),
                           "choices/vocab")

    @staticmethod
    def _check_mode(spec: NodeSpec, m: ModeSpec) -> None:
        """Validate one :class:`ModeSpec` (V2.21). Modes went unvalidated until per-option
        docs arrived, and both checks here are for failures with NO symptom: a
        ``choice_docs`` key that matches no option renders no tooltip (identical to never
        writing one), and a ``default`` outside ``choices`` opens the dropdown on a blank
        row — the same defect ``_check_interaction`` already catches for a ``choices``
        socket, which a Mode had no reason to be exempt from."""
        where = f"{spec.op_key}[{m.name}]"
        if not m.choices:
            raise ValueError(f"{where}: a Mode with no choices is a dropdown that cannot "
                             f"be moved")
        if m.default and m.default not in m.choices:
            raise ValueError(f"{where}: default={m.default!r} is not in choices "
                             f"{list(m.choices)}")
        _check_choice_docs(where, m.choice_docs, tuple(m.choices), "choices")

    def get(self, op_key: str) -> Optional[NodeSpec]:
        return self._by_key.get(op_key)

    def all(self) -> List[NodeSpec]:
        return list(self._by_key.values())

    def keys(self) -> List[str]:
        return list(self._by_key)

    def __contains__(self, op_key: str) -> bool:
        return op_key in self._by_key


NODES = NodeRegistry()


def define_node(op_key: str, label: str, *, category: str = "general",
                inputs: Sequence[SocketSpec] = (), outputs: Sequence[SocketSpec] = (),
                modes: Sequence[ModeSpec] = (), description: str = "",
                granularity: Union[Granularity, Mapping[str, Granularity], None] = None,
                kernel_axes: Union[FrozenSet[str], Mapping[str, FrozenSet[str]], None] = None,
                footprint_mode: str = DIM_MODE,
                meta_transform: Optional[Callable[..., Any]] = None,
                supports_2d: bool = True, supports_true_3d: bool = True,
                three_d_fallback: str = "",
                reads_domains: FrozenSet[Domain] = frozenset(),
                adds_domains: FrozenSet[Domain] = frozenset(),
                extra_layers: Optional[Callable[..., Any]] = None) -> NodeSpec:
    """Build and register a :class:`NodeSpec`."""
    return NODES.register(NodeSpec(
        op_key=op_key, label=label, category=category,
        inputs=tuple(inputs), outputs=tuple(outputs), modes=tuple(modes),
        description=description, granularity=granularity, kernel_axes=kernel_axes,
        footprint_mode=footprint_mode,
        meta_transform=meta_transform, supports_2d=supports_2d,
        supports_true_3d=supports_true_3d, three_d_fallback=three_d_fallback,
        reads_domains=frozenset(reads_domains), adds_domains=frozenset(adds_domains),
        extra_layers=extra_layers,
    ))


__all__ = [
    "SocketSpec", "ModeSpec", "NodeSpec", "NodeRegistry", "NODES",
    "Granularity", "DIM_MODE", "DIM_DESCRIPTION", "DIM_CHOICE_DOCS",
    "PATH_KINDS", "PICK_KINDS", "BOUND_PICK_KINDS",
    "InDataset", "OutDataset", "OutValue", "Mode", "DimMode",
    "InFloat", "InInt", "InBool", "InVector", "InColor", "InString",
    "define_node", "layer_value",
]
