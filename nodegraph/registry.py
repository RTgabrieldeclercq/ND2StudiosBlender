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
#: * ``frames``     — adopt the M/T/Z selection from the strips (the frame the cursor is on,
#:                    for an axis with nothing ticked) as one ``"m0-2,t3"`` spec
#: * ``percentile`` — adopt the contrast window the histogram handles are sitting on
#: * ``gamma``      — adopt the histogram's gamma dot
#: * ``nudge_xy``   — two clicks, a feature in the primary then the same feature in an
#:                    overlaid secondary; commits the µm nudge that lines them up (see
#:                    ``pick_bounds``) — ``view.overlay``'s ``offset_y``/``offset_x``
#:
#: Validated at registration for the same reason as ``path_kind``: an unrecognized value
#: fails SILENTLY (the GUI just wouldn't draw the Pick button), which is indistinguishable
#: from "this param was never annotated".
PICK_KINDS: FrozenSet[str] = frozenset({
    "shapes", "area", "level", "radius", "distance", "grid", "rect",
    "channel", "channels", "frame", "zrange", "frames", "percentile", "gamma",
    "nudge_xy",
})

#: Pick kinds that write a whole GROUP of sockets from one gesture and therefore require
#: :attr:`SocketSpec.pick_bounds`. ``pick_peer`` covers the two-socket case where the pair is
#: an interval aimed in two phases; these are different — one gesture yields every member at
#: once (a rectangle *is* four numbers), so there is nothing to order and no second phase.
#:
#: ``frames`` is deliberately NOT here: a selection across three axes is one VALUE (``util.crop``'s
#: ``"m0-2,t3"`` spec) rather than three sockets written together, so there is no group to
#: declare — which is the simpler shape wherever the picked thing is one thing.
BOUND_PICK_KINDS: FrozenSet[str] = frozenset({"rect", "zrange", "nudge_xy"})

#: Which SocketTypes each pick kind may annotate. A gesture produces a particular KIND of
#: number — a ruler produces a physical length, an eyedropper an intensity — so putting
#: ``distance`` on a STRING or ``shapes`` on a FLOAT is a declaration bug, not a style
#: choice. ``level`` and ``grid`` legitimately span INT and FLOAT: a threshold is a float
#: on normalized data but an integer count on the raw-histogram node, and a subset grid is
#: px (INT) for DVC/DIC but µm (FLOAT) for the object-field grid.
_PICK_SOCKET_TYPES: Dict[str, Tuple[SocketType, ...]] = {
    "shapes": (SocketType.STRING,),
    "channels": (SocketType.STRING,),
    # A whole SELECTION ("m0-2,t3"), so STRING like ``channels`` and unlike ``frame``/
    # ``zrange``: the picks span three axes and can be sparse on each, which no number holds.
    "frames": (SocketType.STRING,),
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
    # a µm offset pair, written together from one two-click gesture
    "nudge_xy": (SocketType.FLOAT,),
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
    #: WHICH Dataset input a ``layer_in`` socket picks from (V2.22). Empty = the primary,
    #: which is what every layer socket meant until a node needed to combine structure
    #: produced by two DIFFERENT branches — Voronoi Cells takes its seed dots from one
    #: chain and the areas it clips them to from another, and a graph cannot put a Point
    #: table and someone else's Label raster on one wire.
    #:
    #: The GUI picker follows the primary edge on purpose (``document.layer_choices``):
    #: the payload flows from ``dataset_preds[0]`` and a ``raw``/``reference`` input's
    #: layers must never be offered as if they were on it. So a socket that genuinely
    #: reads the SECOND input has to say so, or the picker would offer names from a wire
    #: its compute never looks at — which reads as "the layer picker is broken" and is
    #: indistinguishable from a typo'd layer name.
    #:
    #: Falls back to the primary when the named input is UNWIRED, which is what keeps the
    #: single-wire graphs that predate the second input working unchanged.
    layer_from: str = ""
    #: ── column-name sockets (V2.28) ───────────────────────────────────────────
    #: A STRING socket carrying the name of a COLUMN on a structure table, rather than a
    #: layer name or free text. ``column_in`` = the domain whose table the column must be
    #: on — the GUI then offers the columns ``propagate_meta`` knows were measured
    #: upstream (``MetaEnvelope.columns_in``), instead of a bare text box in which
    #: ``mean_intensity`` is a guess the user only learns is wrong at pull time.
    #:
    #: ``column_in_mode`` names a Mode whose VALUE is the domain instead, for a node whose
    #: member domain is a lever (a per-object condition tests Label rows or Point rows
    #: depending on its ``target``) — the ``layer_in_mode`` arrangement, one level down.
    #:
    #: ``column_from`` names the LAYER-name socket on this same node that says which
    #: layer's columns to offer; empty unions every layer in the domain, which is what a
    #: socket whose layer is inferred (§4g) must do.
    #:
    #: The offered list is CLOSED and the combo is not editable, which is the whole
    #: point: every structure-producing node declares ``adds_columns``
    #: (``selftest::test_column_catalog_complete``), so a name absent from the list is a
    #: column no node on this wire writes, and typing it would only defer the refusal to
    #: the pull. Contrast ``layer_in``, whose combo MUST stay editable because a couple
    #: of producers name layers the edit-time pass cannot predict.
    #:
    #: Presentation-only and memo-neutral all the same — the value is an ordinary
    #: string param, and the compute still validates it against the real table.
    column_in: Optional[Domain] = None
    column_in_mode: str = ""
    column_from: str = ""
    #: EXTRA domains whose columns this socket also offers, beyond ``column_in``. A
    #: per-object condition on Label rows can legitimately test ``track_length``, which
    #: lives on the TRACK table and reaches the label rows by the ``member_id`` join the
    #: compute performs — so the picker has to offer it even though it is not a Label
    #: column. Without this the tracking half of the condition palette would be invisible
    #: to the exact user who asked for it, and reachable only by typing.
    column_join: Tuple[Domain, ...] = ()
    #: On an AUXILIARY Dataset input: its image is a viewer SOURCE, composited under the
    #: primary's when this node is viewed (V2.22, reported 2026-08-04).
    #:
    #: The engine is one-payload-per-node and a payload has one image, so a node that reads
    #: structure off a second branch shows only the primary's channel — "I can only see the
    #: UV channel" on a graph whose areas came from the Red one. The display machinery to
    #: composite two sources already exists (it is what ``view.overlay`` drives); what was
    #: missing is any way for a socket to say it *is* one.
    #:
    #: Declared per socket rather than inferred for every second input, because most of them
    #: must NOT composite: ``analysis.measure``'s ``raw`` is the unenhanced version of the
    #: same pixels and would draw the field twice, and a ``reference`` is another timepoint
    #: of the same channel. The ones that qualify carry genuinely different content — a
    #: different channel, segmented independently — which is exactly the case where the node
    #: reads a DOMAIN off the wire rather than intensities.
    view_source: bool = False
    #: On an AUXILIARY Dataset input: whether the domains arriving on this wire are part of
    #: what the node's OUTPUT carries (2026-09-30). ``propagate_meta`` unions the domain
    #: sets of every Dataset input, which is right for a node that merges its inputs and
    #: wrong for one that only READS a second input and passes its primary through.
    #: ``io.write_movie`` is that case: it draws a segmentation wired into ``source_b``, but
    #: what leaves the node is ``data`` byte-for-byte, so without this flag the edit-time
    #: envelope would promise B's Label and Voxel layers to everything downstream, and the
    #: pull would then fail to deliver them.
    passes_domains: bool = True
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
    #: ── growable input groups (2026-10-02) ───────────────────────────────────
    #: Dataset inputs that share a ``grow_group`` name reveal themselves ONE AT A TIME on
    #: the card: the first is always shown, and each later one appears only once the one
    #: before it is wired (or it is wired itself) — Blender's "virtual socket", so a node
    #: that takes any number of streams (``view.viewer``) always offers exactly one empty
    #: slot instead of a column of six. Engine-neutral: the specs all exist and an unwired
    #: one is simply absent from ``ctx.inputs``; only :meth:`nodelab_v2.document
    #: .GraphDocument.input_specs` reads it, and it is the one place that knows the wires.
    grow_group: str = ""
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

    @property
    def is_scope(self) -> bool:
        """True for the STATISTICS-POPULATION mode (V2.27) — ``role="scope"``.

        The second role, and the mirror of :attr:`is_dim_lever`: the lever says how much of
        the data one kernel call sees, this says which voxels are pooled into the *statistic*
        a data-derived parameter is computed from — the plane's histogram, the volume's, or
        one population per label region.

        It exists as a role rather than a naming convention because the GUI edits it from the
        card's footprint band (``nodelab_v2.node_item``), and the band cannot be keyed on
        :attr:`NodeSpec.footprint_mode` instead: on three of the four nodes that declare a
        non-default one, that Mode is not a population at all. ``util.zproject``'s is
        ``method``, so a band bound to it would offer max/mean/**none** — i.e. a footprint
        control that changes the reducer, or switches the node off. ``enhance.normalize`` is
        the clean proof of the split: its population IS a ``scope`` Mode while its
        ``footprint_mode`` is ``bounds``, and the two must stay independent.

        The vocabulary itself lives in :mod:`nodegraph.catalog._shared.scope` — not here,
        because this module must not import the catalog. ``selftest::test_scope_facility``
        closes that loop by checking every declared scope Mode against it."""
        return self.role == "scope"

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
    #: CONDITIONAL reads, unioned onto ``reads_domains`` per mode state (V2.22):
    #: ``{mode_name: {mode_value: frozenset(Domain)}}``. A node whose branches read
    #: DIFFERENT domains — Voronoi Cells needs a Label instance under ``per_region`` and
    #: nothing at all under ``frame``; Track Objects needs Label under ``target=label``
    #: and Point under ``target=point`` — could previously express only the single static
    #: worst case, so the catalog split into nodes that over-claimed (a false red chip on
    #: a graph that is fine) and nodes that declared ``frozenset()`` and warned about
    #: nothing. Both halves were silent: the rail is advisory, so nobody noticed that the
    #: node telling you it reads Points is the one that also needs your labels.
    #:
    #: It is a UNION over every listed mode rather than the single-mode ``Mapping`` form
    #: ``granularity`` uses, because the real requirements are not keyed by one dropdown:
    #: ``transform.transfer_structure`` reads the domain named by ``from_domain`` AND the
    #: one named by ``to_domain``, and ``flow.iterate`` needs a Global under
    #: ``preserve=best`` OR under ``mode=feedback`` — a disjunction a single key cannot
    #: state (the same limit that makes the socket's own ``available_in`` approximate
    #: there). An unlisted value contributes nothing, which is how "this branch requires
    #: no structure at all" is said.
    reads_domains_by_mode: Mapping[str, Mapping[str, FrozenSet[Domain]]] = field(
        default_factory=dict)
    adds_domains: FrozenSet[Domain] = frozenset()
    #: The output is a NEW Dataset this node makes (V4.00 step 7 — a plot's Picture), not its
    #: input carried on: at edit time no input domain, structure layer or column passes
    #: through to it, only what it adds itself. Without it the envelope would keep
    #: promising the input's tables downstream (layer and column pickers offering what the
    #: payload no longer has) — ``passes_domains=False`` does this for an AUXILIARY input
    #: only, since the primary input's domains always pass.
    fresh_output: bool = False
    #: Layers this node creates that no ``layer_out`` socket can describe (V2.11):
    #: ``(params, modes) -> ((Domain, name), ...)``. Needed by the handful of producers
    #: that name a layer with NO socket at all (``align.drift``/``registration.stabilize``
    #: write the literals ``drift_y``/``drift_x``), derive the name from ANOTHER param
    #: (``analysis.extract_boundary`` -> ``f"{labels}_boundary"``), or write into the layer
    #: their READ socket names (``analysis.measure`` adds Label columns to the raster it
    #: measures). MUST be total — see ``propagate_meta``, which runs on every keystroke.
    extra_layers: Optional[Callable[..., Any]] = None
    #: STRUCTURE COLUMNS this node writes (V2.28): ``(params, modes, incoming) ->
    #: [(Domain, layer_name, column_name), ...]``, feeding ``propagate_meta``'s edit-time
    #: column catalog so a downstream ``column_in`` socket can OFFER them.
    #:
    #: ``incoming`` is the caller's accumulated catalog — the same triples, as they stand
    #: on this node's input edge. It takes a third argument where its sibling
    #: ``extra_layers`` takes two because columns FLOW in a way layer names do not: a node
    #: that re-emits a table under a new name (``analysis.filter_labels`` → ``labels_kept``)
    #: carries every column with it, and could not name one of them without being told what
    #: is already there. Declaring only the columns it invents would silently amputate the
    #: catalog at exactly the node a user filters with.
    #:
    #: The layer half of this question is already answered by ``layer_out``/``extra_layers``;
    #: this is the half below it. "A Label table called CELLS exists" is what the layer
    #: catalog knows, and it is not enough to populate a menu of conditions — whether
    #: anyone has measured its eccentricity yet is a different fact, and the one the user
    #: is actually choosing between.
    #:
    #: Same total-function obligation as ``extra_layers``, for the same reason: it runs
    #: inside ``propagate_meta`` on every keystroke. It is **NOT optional**: the column
    #: picker is a closed dropdown with no free-text escape, so a producer that declares
    #: nothing makes every column it writes **unpickable** — not merely unsuggested.
    #: ``selftest::test_column_catalog_complete`` fails the build for any node that adds
    #: a structure domain without one, which is the only thing keeping the dropdown
    #: honest as the catalog grows.
    adds_columns: Optional[Callable[..., Any]] = None
    #: Socket values this node's LOADED MODEL was trained with (V2.23):
    #: ``(params, modes) -> {socket_name: value}``, read from the JSON beside the weights
    #: (``nodegraph.trained``). Same shape and the same total-function obligation as
    #: ``extra_layers`` — it runs inside ``propagate_meta`` and the inspector rebuild, i.e.
    #: on every keystroke, so it must never raise and never be slow.
    #:
    #: It declares a THIRD source for a param's value, between the two that already
    #: existed: an explicit param still wins, and the ``SocketSpec`` default still applies
    #: when nothing else speaks, but in between, a socket named here shows (and resolves to)
    #: what the checkpoint on the wire was trained with. That is why it lives on the spec
    #: rather than purely inside the compute — the inspector needs it to draw the auto/pin
    #: box, and a value the GUI derived independently could disagree with the pull.
    #:
    #: It is NOT a substitute for ``SocketSpec.derive``, and the two are deliberately
    #: separate: ``derive`` is a pure-arithmetic expression over the metadata envelope
    #: (``eval_derive``, empty builtins, no I/O), while this reads a file whose path comes
    #: from another param. Keeping them apart is what stops every ``derive`` read in the
    #: app from becoming a possible disk touch.
    #:
    #: A **Mode** can never be sourced this way. The engine hands a compute the *resolved*
    #: mode state, so "unset" and "explicitly the default" are indistinguishable there and
    #: adopting would silently overwrite a deliberate choice — a checkpoint/mode
    #: disagreement is refused instead (``enhance.zs_deconvnet``'s ``arch_3d`` and the dim
    #: lever).
    trained_params: Optional[Callable[..., Any]] = None
    #: One line for the GUI naming WHICH file :attr:`trained_params` read, or why it found
    #: nothing (V2.23b): ``(params, modes) -> str``, ``""`` for nothing to say.
    #:
    #: It exists because the empty answer is the ambiguous one. "This model has no training
    #: record" and "this feature is not working" look identical in a panel — reported exactly
    #: that way ("loading in the model does not change any of the parameters") for the two
    #: legitimate empty cases: a published checkpoint that carries no sidecar, and a model path
    #: that is not a model directory. A resolver returning ``{}`` is correct in both; saying so
    #: is what makes it usable.
    #:
    #: Presentation-only and memo-neutral, exactly like ``description``: nothing hashes it and
    #: no compute may read it, so its wording is a zero-risk edit.
    trained_note: Optional[Callable[..., Any]] = None

    def note_for_trained(self, params: Mapping[str, Any],
                         modes: Mapping[str, Any]) -> str:
        """Resolve :attr:`trained_note`, swallowing everything — ``""`` when there is no
        hook, it raises, or it returns a non-string. Same total-function seam as
        :meth:`trained`, and for the same reason: this runs on every inspector rebuild, and a
        broken explanation must never take the panel down with it."""
        fn = self.trained_note
        if fn is None:
            return ""
        try:
            got = fn(dict(params or {}), dict(modes or {}))
        except Exception:  # noqa: BLE001 — a note must never break the panel
            return ""
        return got if isinstance(got, str) else ""

    def trained(self, params: Mapping[str, Any],
                modes: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve :attr:`trained_params`, swallowing everything. ``{}`` for a node that
        declares none.

        The blanket ``except`` is the same call the GUI already makes around
        ``extra_layers``, for the same reason: this runs on every keystroke on the edit-time
        path, and a model file that turns out to be a directory (or a resolver bug) must
        degrade to "the model states nothing" — i.e. static socket defaults — rather than
        take down the inspector or the propagation pass. Every function in
        ``nodegraph.trained`` is already total; this is the backstop that makes that a
        property of the SEAM instead of a promise each resolver has to keep.
        """
        fn = self.trained_params
        if fn is None:
            return {}
        try:
            got = fn(dict(params or {}), dict(modes or {}))
        except Exception:  # noqa: BLE001 — a resolver must never break edit-time propagation
            return {}
        return dict(got) if isinstance(got, Mapping) else {}

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

    def scope_mode(self) -> Optional[ModeSpec]:
        """The statistics-population mode, if this node bears one (V2.27) — see
        :attr:`ModeSpec.is_scope`. At most one per spec (``_check_footprint``)."""
        return next((m for m in self.modes if m.is_scope), None)

    def has_scope_mode(self) -> bool:
        return self.scope_mode() is not None

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
        if self.fresh_output:
            return frozenset(self.adds_domains)    # a NEW Dataset: only what it adds
        return incoming | self.adds_domains

    def resolve_reads_domains(self,
                              state: Optional[Mapping[str, str]] = None
                              ) -> FrozenSet[Domain]:
        """The domains this node requires in mode ``state`` — the unconditional
        :attr:`reads_domains` unioned with every :attr:`reads_domains_by_mode` entry the
        state selects (V2.22). ``state=None`` resolves against :meth:`default_state`.

        Mirrors :meth:`resolve_granularity`: the declaration is the honest per-branch one
        and every consumer resolves it, so no caller has to know whether a given node
        happens to be conditional."""
        if not self.reads_domains_by_mode:
            return self.reads_domains
        st = self.default_state() if state is None else state
        out = set(self.reads_domains)
        for mode, table in self.reads_domains_by_mode.items():
            out |= set(table.get(st.get(mode, ""), ()))
        return frozenset(out)

    def missing_domains(self, incoming: FrozenSet[Domain],
                        state: Optional[Mapping[str, str]] = None) -> FrozenSet[Domain]:
        """Required domains not present upstream — the GUI's red validation chips."""
        return self.resolve_reads_domains(state) - incoming


# ── socket / mode factories (the §11 sketch) ─────────────────────────────────

def InDataset(name: str = "data", *, multi: bool = False, label: str = "",
              view_source: bool = False, description: str = "",
              available_in: Optional[Mapping[str, FrozenSet[str]]] = None,
              passes_domains: bool = True, grow_group: str = "") -> SocketSpec:
    """A Dataset input. ``description`` is the hover text, and it earns its place on a node
    with SEVERAL Dataset inputs: the domain rail is a node-level answer painted identically
    beside each one, so the card cannot say which wire wants what. A value socket has carried
    prose since V2.13; a Dataset socket could not, which is why `areas`, `raw`, `secondary`
    and `reference` all hovered as bare names. ``passes_domains=False`` marks an input the
    node only reads (see :attr:`SocketSpec.passes_domains`)."""
    return SocketSpec(name, SocketType.DATASET, Direction.IN, label=label,
                      multi=multi, view_source=view_source, description=description,
                      available_in=available_in, passes_domains=passes_domains,
                      grow_group=grow_group)


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
             layer_out: Tuple[Domain, ...] = (), layer_from: str = "",
             column_in: Optional[Domain] = None, column_in_mode: str = "",
             column_from: str = "", column_join: Tuple[Domain, ...] = (),
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
                          layer_out=tuple(layer_out), layer_from=layer_from,
                          column_in=column_in, column_in_mode=column_in_mode,
                          column_from=column_from, column_join=tuple(column_join),
                          kernel_param=kernel_param,
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
            self._check_layer_from(spec, s)
            self._check_interaction(spec, s, in_names)
        for m in spec.modes:
            self._check_mode(spec, m)
        self._check_reads_domains(spec)
        self._check_footprint(spec)
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
    def _check_layer_from(spec: NodeSpec, s: SocketSpec) -> None:
        """Validate :attr:`SocketSpec.layer_from` (V2.22) — which Dataset input a layer
        socket picks from.

        Silent both ways if wrong, and in the most confusing possible manner: a name that
        matches no socket falls straight back to the primary edge, so the picker offers a
        plausible list of names from the wrong wire while the compute reads the other one.
        The user sees a valid-looking layer name that "doesn't exist"."""
        if s.view_source:
            where = f"{spec.op_key}.{s.name}"
            ds_ins = [i.name for i in spec.inputs if i.type is SocketType.DATASET]
            if s.type is not SocketType.DATASET:
                raise ValueError(f"{where}: view_source is only meaningful on a Dataset "
                                 f"input, got {s.type.name}")
            if ds_ins and s.name == ds_ins[0]:
                raise ValueError(f"{where}: view_source on the PRIMARY input — the primary's "
                                 f"image is what the viewer already shows; this marks an "
                                 f"AUXILIARY one as a second source")
        if not s.layer_from:
            return
        where = f"{spec.op_key}.{s.name}"
        if not (s.layer_in or s.layer_in_mode):
            raise ValueError(f"{where}: layer_from={s.layer_from!r} without layer_in — "
                             f"there is no picker for it to redirect")
        ds_ins = [i.name for i in spec.inputs if i.type is SocketType.DATASET]
        if s.layer_from not in ds_ins:
            raise ValueError(f"{where}: layer_from={s.layer_from!r} is not a Dataset input "
                             f"on this node (it has {ds_ins}) — it would silently fall back "
                             f"to the primary edge")
        if ds_ins and s.layer_from == ds_ins[0]:
            raise ValueError(f"{where}: layer_from={s.layer_from!r} IS the primary input — "
                             f"that is already the default; spelling it out suggests a "
                             f"different input was meant")

    @staticmethod
    def _check_reads_domains(spec: NodeSpec) -> None:
        """Validate :attr:`NodeSpec.reads_domains_by_mode` (V2.22).

        Every failure here is silent at runtime — the domain rail is advisory, so a mode
        name or value that matches nothing simply contributes no domains and the node goes
        on under-claiming exactly as it did before the declaration was written. That is
        indistinguishable from not having written it, which is the whole class of defect
        this field exists to end."""
        table = spec.reads_domains_by_mode
        if not table:
            return
        by_name = {m.name: m for m in spec.modes}
        for mode_name, per_value in table.items():
            where = f"{spec.op_key}[{mode_name}]"
            mode = by_name.get(mode_name)
            if mode is None:
                raise ValueError(
                    f"{where}: reads_domains_by_mode names no such mode (this node has "
                    f"{sorted(by_name)}) — it would contribute no domains in every state")
            unknown = sorted(set(per_value) - set(mode.choices))
            if unknown:
                raise ValueError(
                    f"{where}: reads_domains_by_mode keys {unknown} are not choices of "
                    f"this mode {list(mode.choices)} — they can never be selected")
            for value, domains in per_value.items():
                bad = [d for d in domains if not isinstance(d, Domain)]
                if bad:
                    raise ValueError(
                        f"{where}={value!r}: reads_domains_by_mode holds {bad!r}, which "
                        f"is not a Domain")

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

    @staticmethod
    def _check_footprint(spec: NodeSpec) -> None:
        """Validate the footprint declaration (V2.27) — ``granularity`` / ``kernel_axes`` /
        ``footprint_mode``, which went **entirely unvalidated** until now.

        Every failure below is silent AND expensive, which is the combination that earns a
        registration check. :meth:`NodeSpec.resolve_granularity` is
        ``g.get(state.get(self.footprint_mode, ""))`` — total, and ``None`` on any miss. So a
        Mapping that forgets one of its Mode's choices, or a ``footprint_mode`` naming a Mode
        that does not exist, resolves to ``None`` in that state and nothing raises. Downstream,
        ``None`` is not in ``_shared/map_image.py``'s lazy whitelist, so the node abandons the
        tiled path and allocates the WHOLE SERIES as float64 — 5 GiB per raster on a 16-position
        2048²×10 stack, reached without a single error message. The GUI is worse than silent:
        the card paints the green ``TILEABLE`` chip, i.e. "cheapest possible read", for a
        footprint that failed to resolve.

        Four checks, each closing one of those:

        * a Mapping ``granularity``/``kernel_axes`` requires ``footprint_mode`` to name a real
          Mode on this spec;
        * the Mapping must carry a key for **every** choice of that Mode — coverage, not merely
          overlap, because the uncovered value is exactly the one that resolves to ``None``;
        * every ``granularity`` value is a real :class:`Granularity` member;
        * at most one ``role="scope"`` Mode per spec, since
          :meth:`NodeSpec.scope_mode` returns the first and a second one would be a control
          that silently does nothing.

        Audited against the shipped catalog when this landed: all 79 registered ops pass with
        no fixes, so this is a ratchet on new work rather than a migration."""
        mode_names = {m.name for m in spec.modes}
        for field_name, value in (("granularity", spec.granularity),
                                  ("kernel_axes", spec.kernel_axes)):
            if not isinstance(value, Mapping):
                continue
            if spec.footprint_mode not in mode_names:
                raise ValueError(
                    f"{spec.op_key}: {field_name} is keyed per mode value, but "
                    f"footprint_mode={spec.footprint_mode!r} is not a Mode on this node "
                    f"({sorted(mode_names) or 'it has none'}) — every state would resolve to "
                    f"None, which drops the node off the tiled read path and allocates the "
                    f"whole series eagerly, with no error. Name the Mode that decides the "
                    f"footprint, or declare a single {field_name} value.")
            choices = tuple(next(m.choices for m in spec.modes
                                 if m.name == spec.footprint_mode))
            missing = [c for c in choices if c not in value]
            if missing:
                raise ValueError(
                    f"{spec.op_key}: {field_name} is keyed by {spec.footprint_mode!r} but has "
                    f"no entry for {missing} — those values resolve to None, which reads as "
                    f"'undeclared' (whole-series eager realize, and a green TILEABLE chip on "
                    f"the card). Give every choice a value; over-declaring is safe, absent is "
                    f"not.")
        gran = spec.granularity
        vals = list(gran.values()) if isinstance(gran, Mapping) else \
            ([gran] if gran is not None else [])
        bad = [v for v in vals if not isinstance(v, Granularity)]
        if bad:
            raise ValueError(
                f"{spec.op_key}: granularity holds {bad!r}, which is not a Granularity member")
        scopes = [m.name for m in spec.modes if m.is_scope]
        if len(scopes) > 1:
            raise ValueError(
                f"{spec.op_key}: two modes claim role='scope' ({scopes}) — NodeSpec.scope_mode "
                f"returns the first, so the second would be a live-looking population control "
                f"that nothing reads, and the card's footprint band could only edit one.")

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
                reads_domains_by_mode: Optional[
                    Mapping[str, Mapping[str, FrozenSet[Domain]]]] = None,
                adds_domains: FrozenSet[Domain] = frozenset(),
                fresh_output: bool = False,
                extra_layers: Optional[Callable[..., Any]] = None,
                adds_columns: Optional[Callable[..., Any]] = None,
                trained_params: Optional[Callable[..., Any]] = None,
                trained_note: Optional[Callable[..., Any]] = None) -> NodeSpec:
    """Build and register a :class:`NodeSpec`."""
    return NODES.register(NodeSpec(
        op_key=op_key, label=label, category=category,
        inputs=tuple(inputs), outputs=tuple(outputs), modes=tuple(modes),
        description=description, granularity=granularity, kernel_axes=kernel_axes,
        footprint_mode=footprint_mode,
        meta_transform=meta_transform, supports_2d=supports_2d,
        supports_true_3d=supports_true_3d, three_d_fallback=three_d_fallback,
        reads_domains=frozenset(reads_domains),
        reads_domains_by_mode={m: {v: frozenset(d) for v, d in per.items()}
                               for m, per in (reads_domains_by_mode or {}).items()},
        adds_domains=frozenset(adds_domains),
        fresh_output=bool(fresh_output),
        extra_layers=extra_layers,
        adds_columns=adds_columns,
        trained_params=trained_params,
        trained_note=trained_note,
    ))


__all__ = [
    "SocketSpec", "ModeSpec", "NodeSpec", "NodeRegistry", "NODES",
    "Granularity", "DIM_MODE", "DIM_DESCRIPTION", "DIM_CHOICE_DOCS",
    "PATH_KINDS", "PICK_KINDS", "BOUND_PICK_KINDS",
    "InDataset", "OutDataset", "OutValue", "Mode", "DimMode",
    "InFloat", "InInt", "InBool", "InVector", "InColor", "InString",
    "define_node", "layer_value",
]
