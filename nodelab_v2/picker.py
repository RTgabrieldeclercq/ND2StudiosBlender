"""Interactive parameter picking (NodeLab v2, V2.16) — the headless half.

A **pick** is a gesture on the viewed data that produces a parameter value: drag a circle
over a nucleus to set a blur σ, click the smallest cell you want to keep to set a minimum
area, draw the region to correlate. The alternative — typing a number into a spin box — asks
the user to answer in the machine's terms ("is that spot 0.2 or 0.6 µm across?") a question
they can see the answer to. Which params offer which gesture is DECLARED on the socket
(:attr:`nodegraph.registry.SocketSpec.pick_kind`), so this module never names a node.

**What lives here vs. in the viewer.** This module owns the *state machine* and the *maths*:
what a gesture means, how plane pixels become microns, what value gets committed. It imports
no Qt and touches no pixels, so every conversion in it is checkable headlessly (see
``nodegraph.selftest.test_picker``). :mod:`nodelab_v2.viewer` owns only the parts that
genuinely need a widget — mouse events, painting the rubber band, sampling the plane under
the cursor — and feeds sampled numbers in as the ``probe`` argument.

**A pick is an ordinary edit.** :meth:`PickSession.values` returns a plain
``{socket_name: value}`` dict that the window writes through the same path as a typed edit,
including the sticky pin. There is no second kind of parameter value, nothing new in the
recipe hash, and every picked param stays editable by hand afterwards.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nodegraph.registry import PICK_KINDS

#: Kinds resolved from the viewer's CURRENT state with no gesture at all — "use the channel
#: I am looking at", "use this timepoint". They commit the moment they are armed, because
#: there is nothing to aim: the answer is already on screen.
INSTANT_KINDS = frozenset({"channel", "frame", "channels", "zrange", "frames"})

#: Kinds committed off the intensity histogram rather than the image. The user is already
#: dragging these handles to make the image readable; the pick is the missing path from
#: "that looked right" to the parameter.
HISTOGRAM_KINDS = frozenset({"percentile", "gamma"})

#: Kinds driven by a mouse gesture on the image surface.
CANVAS_KINDS = frozenset({"shapes", "area", "level", "radius", "distance", "grid", "rect",
                          "nudge_xy"})

if INSTANT_KINDS | HISTOGRAM_KINDS | CANVAS_KINDS != PICK_KINDS:
    # Import-time, and a raise rather than an assert so ``-O`` cannot silence it. A kind
    # added to the registry but to none of the three sets above would otherwise reach the
    # GUI as a Pick button that arms a session no surface knows how to drive.
    raise RuntimeError(
        "pick kinds not routed to a surface: "
        f"{sorted(PICK_KINDS ^ (INSTANT_KINDS | HISTOGRAM_KINDS | CANVAS_KINDS))}")

#: The instruction line the viewer shows while a pick is armed. Written as an imperative
#: sentence naming the GESTURE, because the banner is the only place the interaction is
#: explained — a user who has just pressed "Pick" needs to know whether to click or drag.
PICK_HELP: Dict[str, str] = {
    "shapes": "Drag to draw the region. Pick a tool below; Cut removes from the "
              "selection. Apply when the shape list looks right.",
    "area": "Click an object to take its area, or drag to draw a blob and use that. "
            "Shift-drag adds the second bound.",
    "level": "Click a pixel to take its intensity as the level.",
    "radius": "Press at the centre and drag out to the edge of the feature.",
    "distance": "Click two points — the value is the distance between them.",
    "grid": "Drag a box the size of one correlation subset.",
    "rect": "Drag the rectangle to keep. Everything outside it is cropped away.",
    "channel": "Taking the channel the viewer is showing.",
    "channels": "Taking the channels the viewer has switched on.",
    "frame": "Taking the timepoint the viewer is showing.",
    "zrange": "Taking the Z planes picked on the Z strip (all of them if none are picked).",
    "frames": "Taking the M / T / Z boxes ticked on the strips — or, for an axis with "
              "nothing ticked, the frame you are looking at.",
    "percentile": "Set the histogram handles where you want them, then Apply.",
    "gamma": "Drag the histogram's gamma dot, then Apply.",
    "nudge_xy": "Click a feature in the PRIMARY image, then click the same feature where the "
                "overlaid source shows it — the nudge moves the source onto the primary.",
}

#: The marker that means "this parameter can be picked", used in the inspector's button
#: text and in the hover prose. A plain WHITE CIRCLE because the shipped Windows UI fonts
#: (Segoe UI / Consolas) have no BULLSEYE — ``◎`` rendered as a tofu box. The node card
#: paints its own ring with QPainter, so it is not exposed to font coverage at all.
PICK_GLYPH = "○"

#: The Pick button's own text, per kind — a short imperative naming the gesture. Shared by
#: the inspector row and the node card's glyph tooltip so the two never describe the same
#: gesture with different words.
PICK_ACTION: Dict[str, str] = {
    "shapes": "Draw the region",
    "area": "Click or draw the size",
    "level": "Eyedropper",
    "radius": "Drag the radius",
    "distance": "Measure on the image",
    "grid": "Drag the grid",
    "rect": "Drag the crop rectangle",
    "channel": "Use the viewed channel",
    "channels": "Use the viewer's channels",
    "frame": "Use the current frame",
    "zrange": "Use the picked Z planes",
    "frames": "Use the selected frames",
    "percentile": "Take the histogram window",
    "gamma": "Take the histogram gamma",
    "nudge_xy": "Align by two clicks",
}

#: What a BOUND group is, as a noun phrase, for the pick bar's title. A group gesture is not
#: "picking y0" — it writes four numbers at once — so naming one member there would misreport
#: what Apply is about to change.
PICK_GROUP_LABEL: Dict[str, str] = {
    "rect": "the crop window",
    "zrange": "the Z range",
    "nudge_xy": "the XY nudge",
}

#: Name fragments that mark a socket as the LOW or HIGH end of a co-picked pair. Checked as
#: a prefix and as a suffix so both catalog spellings are covered (``min_area`` /
#: ``percentile_low``). Used only to ORDER a pair — see :func:`ordered_pair`.
_LOW_WORDS = ("min", "low")
_HIGH_WORDS = ("max", "high")


def _side(name: str) -> str:
    """``"lo"`` / ``"hi"`` / ``""`` — which end of a min/max pair this socket name reads as."""
    parts = name.lower().split("_")
    if parts[0] in _LOW_WORDS or parts[-1] in _LOW_WORDS:
        return "lo"
    if parts[0] in _HIGH_WORDS or parts[-1] in _HIGH_WORDS:
        return "hi"
    return ""


def ordered_pair(socket: str, peer: str, first: float, second: float
                 ) -> Dict[str, float]:
    """Assign two picked magnitudes to a socket and its peer, smallest to whichever name
    reads as the LOW end.

    Every co-picked pair in the catalog is an interval — a spot's min/max radius, a
    hysteresis threshold's low/high, an area window. The gesture order is the user's, not
    the interval's: someone who arms ``max_radius`` and drags the small ring first means the
    small one to be the minimum. Sorting here means a pick can never leave the pair
    inverted, which for most of these consumers is a silent empty result rather than an
    error. Pairs whose names carry no order (a grid's box and stride) are assigned in
    gesture order, which is what the banner promised."""
    lo_first = _side(socket), _side(peer)
    if lo_first == ("lo", "hi"):
        return {socket: min(first, second), peer: max(first, second)}
    if lo_first == ("hi", "lo"):
        return {socket: max(first, second), peer: min(first, second)}
    return {socket: first, peer: second}


# ── calibration ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Calibration:
    """Plane pixels → physical units, read from the node's propagated envelope.

    ``um_px`` of 0 means UNCALIBRATED — the file carried no ``pixel_size_um`` — and every
    conversion below then falls back to 1.0 µm/px, i.e. it reports pixels while the socket's
    unit label still says µm. That is a real limitation of the data, not something a picker
    can fix, so :meth:`PickSession.readout` says so out loud rather than quietly handing
    back a number that means something else than it claims."""

    um_px: float = 0.0
    um_z: float = 0.0

    @classmethod
    def from_metadata(cls, md: Optional[Dict[str, Any]]) -> "Calibration":
        def num(key: str) -> float:
            try:
                v = float((md or {}).get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
            return v if math.isfinite(v) and v > 0 else 0.0
        return cls(um_px=num("pixel_size_um"), um_z=num("z_step_um"))

    @property
    def calibrated(self) -> bool:
        return self.um_px > 0.0

    @property
    def lateral(self) -> float:
        """µm per lateral pixel, with the uncalibrated fallback applied."""
        return self.um_px or 1.0

    @property
    def axial(self) -> float:
        """µm per z step, falling back to the lateral size (an isotropic guess) and then
        to 1.0 — a volume pick on a file with no z spacing is still better than refusing."""
        return self.um_z or self.lateral


# ── the request ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PickRequest:
    """One armed pick: which node's which param, by which gesture.

    Carries ``node_id`` explicitly rather than letting the viewer assume the selected node.
    The inspector shows the SELECTED node while the viewer shows the PULLED one and those
    routinely differ — a pick that wrote to "whatever is selected" would silently retarget
    when the user clicked elsewhere mid-gesture."""

    node_id: str
    socket: str
    kind: str
    unit: str = ""
    peer: str = ""
    #: For a :data:`~nodegraph.registry.BOUND_PICK_KINDS` gesture, the complete ordered group
    #: this pick writes — ``("y0","y1","x0","x1")`` for a crop rectangle. Registration
    #: guarantees the members share one SocketType and one unit, so :attr:`unit` and
    #: :attr:`integral` describe every one of them and no per-name table is needed.
    bounds: Tuple[str, ...] = ()
    label: str = ""
    peer_label: str = ""
    peer_unit: str = ""
    #: True when the socket is INT-typed, so the committed value is rounded and stored as
    #: an int (a px subset size, a channel index) instead of a float.
    integral: bool = False
    peer_integral: bool = False
    #: The bound members' CURRENT values, ``((name, value), ...)`` — for a gesture that commits
    #: a CHANGE rather than a fresh value (``nudge_xy`` adds its delta to the nudge already
    #: set). Filled by the window, which owns the document; ``()`` means "all zero".
    base: Tuple[Tuple[str, float], ...] = ()
    #: The viewed canvas's orientation (``canvas_flip``) for ``nudge_xy``, or ``None`` when
    #: what is viewed is not a canvas: on a canvas that runs toward stage −x / −y a screen
    #: delta runs against stage µm, so the nudge must reverse it. Filled by the window.
    mirror: Optional[Tuple[bool, bool]] = None

    @property
    def surface(self) -> str:
        """``"instant"`` | ``"histogram"`` | ``"canvas"`` — which surface hosts this pick."""
        if self.kind in INSTANT_KINDS:
            return "instant"
        if self.kind in HISTOGRAM_KINDS:
            return "histogram"
        return "canvas"

    @property
    def help_text(self) -> str:
        return PICK_HELP.get(self.kind, "")

    @property
    def action_text(self) -> str:
        return PICK_ACTION.get(self.kind, "Pick")

    @property
    def title(self) -> str:
        """What the pick bar says it is aiming. A bound group names the GROUP, because a
        rectangle is not "y0" — every member changes on Apply."""
        if self.bounds:
            return PICK_GROUP_LABEL.get(self.kind, ", ".join(self.bounds))
        what = self.label or self.socket
        if self.peer:
            what += f" + {self.peer_label or self.peer}"
        return what

    @property
    def leads_group(self) -> bool:
        """True unless this socket is a NON-first member of a bound group.

        The whole group arms the same gesture, so the affordance is drawn once — on the
        group's first member — instead of stacking four identical buttons down the panel.
        Every member still advertises it in hover text, which is where someone looking at
        ``x1`` and wondering whether they can draw it will look."""
        return not self.bounds or self.socket == self.bounds[0]

    def unit_for(self, name: str) -> str:
        """The unit to interpret a magnitude in, for whichever half of the pick is being
        aimed. An unset ``peer_unit`` INHERITS the primary's rather than defaulting to
        pixels: two sockets that co-pick are two ends of one quantity, and silently
        measuring the second in px while the first was in µm produces a number that is
        wrong by the pixel size with nothing on screen to show it."""
        if name and name == self.peer:
            return self.peer_unit or self.unit
        return self.unit

    def integral_for(self, name: str) -> bool:
        if name and name == self.peer:
            return self.peer_integral
        return self.integral


def request_for(node_id: str, spec: Any, peer_spec: Any = None) -> PickRequest:
    """Build the request for one annotated socket. THE single place a ``SocketSpec``'s
    declarations become a pick, so the inspector's button and the node card's ◎ glyph arm
    identical sessions — two builders would eventually disagree about, say, whether a socket
    is integral, and the divergence would only show up as a rounded value in one of them."""
    from nodegraph.sockets import SocketType
    peer = peer_spec if (peer_spec is not None and spec.pick_peer) else None
    return PickRequest(
        node_id=node_id,
        socket=spec.name,
        kind=spec.pick_kind,
        unit=spec.unit,
        bounds=tuple(getattr(spec, "pick_bounds", ()) or ()),
        peer=peer.name if peer is not None else "",
        label=spec.label or spec.name,
        peer_label=(peer.label or peer.name) if peer is not None else "",
        peer_unit=peer.unit if peer is not None else "",
        integral=spec.type is SocketType.INT,
        peer_integral=(peer.type is SocketType.INT) if peer is not None else False,
    )


# ── the session ───────────────────────────────────────────────────────────────

@dataclass
class PickSession:
    """The live state of one armed canvas pick.

    Points are **(x, y) plane pixels** throughout — the coordinate the image surface hands
    back. The ROI shape schema the kernel replays is ``[y, x]``, so the flip happens in
    exactly one place (:meth:`_shape_verts`) instead of at every call site, which is how the
    axis order stops being a coin flip.
    """

    req: PickRequest
    calib: Calibration = field(default_factory=Calibration)
    #: the in-progress gesture's points, (x, y) plane px
    pts: List[Tuple[float, float]] = field(default_factory=list)
    #: which half of a paired pick is being aimed (0 = the armed socket, 1 = its peer)
    phase: int = 0
    #: completed magnitudes, in gesture order, one per phase
    picked: List[float] = field(default_factory=list)
    #: accumulated ROI shapes (the ``shapes`` kind only)
    shapes: List[dict] = field(default_factory=list)
    #: the value the caller sampled under the cursor (intensity for ``level``, an object's
    #: voxel count for ``area``); ``None`` when the caller had nothing to sample
    probe: Optional[float] = None
    #: ``shapes`` tool + operation, driven by the viewer's little toolbar
    tool: str = "rect"
    op: str = "add"
    brush_px: float = 8.0
    #: True once every phase this pick needs has been aimed
    done: bool = False
    #: set when the gesture was refused, for the banner to explain (e.g. a click that
    #: landed on background while the node has no labels to sample)
    note: str = ""
    #: where the pointer is now, for the live rubber band on the CLICK-driven kinds
    #: (``distance``, a polygon in progress) whose geometry is not in :attr:`pts` yet
    hover_pt: Optional[Tuple[float, float]] = None
    _dragging: bool = False
    _moved: bool = False

    # ── the gesture ───────────────────────────────────────────────────────────
    def press(self, x: float, y: float, *, probe: Optional[float] = None) -> None:
        self.note = ""
        self._dragging = True
        self._moved = False
        if probe is not None:
            self.probe = probe
        if self.req.kind in ("distance", "nudge_xy"):
            if len(self.pts) >= 2:
                self.pts = []                   # a third click restarts the measurement
            self.pts.append((x, y))
            return
        if self.req.kind == "shapes" and self.tool == "polygon":
            self.pts.append((x, y))
            return
        self.pts = [(x, y)]

    def hover(self, x: float, y: float) -> None:
        """Track the pointer for the click-driven kinds' preview (no state change)."""
        self.hover_pt = (x, y)

    def drag(self, x: float, y: float) -> None:
        if not self._dragging:
            return
        self._moved = True
        self.hover_pt = (x, y)
        kind = self.req.kind
        if kind in ("distance", "nudge_xy"):
            return                              # two clicks, not a drag — see press()
        if kind == "shapes" and self.tool == "polygon":
            return                              # a polygon grows on clicks, not on drags
        if kind == "area" or (kind == "shapes" and self.tool == "brush"):
            # Freehand: the whole trace IS the geometry. `area`'s drag path is always
            # freehand — drawing the blob you mean is the gesture that works before
            # anything has been segmented, which is the point of offering it.
            self.pts.append((x, y))
            return
        if len(self.pts) < 2:
            self.pts.append((x, y))
        else:
            self.pts[-1] = (x, y)               # rubber-band the moving corner

    def release(self, x: float, y: float, *, probe: Optional[float] = None) -> None:
        """End the gesture. Returns nothing; read :attr:`done` and :meth:`values`."""
        if not self._dragging:
            return
        self._dragging = False
        if probe is not None:
            self.probe = probe
        kind = self.req.kind
        if kind == "shapes":
            self._end_shape(x, y)
            return
        if kind in ("distance", "nudge_xy"):
            if len(self.pts) >= 2:
                self.done = True
            return
        if kind == "level":
            if self.probe is None:
                self.note = "nothing to sample there"
                return
            self.picked = [float(self.probe)]
            self.done = True
            return
        if kind == "area":
            self._end_area(x, y)
            return
        if kind == "rect":
            if not self._moved or len(self.pts) < 2:
                self.note = "drag a rectangle — a click has no extent"
                return
            # Anchor the far corner on the RELEASE position, not on wherever the last move
            # happened to land. With a real mouse the two coincide, but relying on that makes
            # the committed window depend on event coalescing — and any caller that releases
            # without a final move would silently crop to the second-to-last position.
            self.pts[-1] = (x, y)
            self.done = True
            return
        # radius / grid: a magnitude per phase
        mag = self._magnitude(x, y)
        if mag <= 0:
            self.note = "that gesture had no extent — press and drag"
            return
        self.picked.append(mag)
        if self.req.peer and self.phase == 0:
            self.phase = 1
            self.pts = self.pts[:1]             # keep the centre / origin for phase 2
        else:
            self.done = True

    def cancel_gesture(self) -> None:
        """Abandon the in-progress stroke without ending the session (Esc once)."""
        self._dragging = False
        self.pts = self.pts[:1] if self.phase else []
        self.note = ""

    # ── shapes ────────────────────────────────────────────────────────────────
    def _end_shape(self, x: float, y: float) -> None:
        """Commit one drawn shape into the accumulating list. ``shapes`` never sets
        :attr:`done` — the list is the value, so the user decides when it is finished."""
        if self.tool == "polygon":
            return                              # closed explicitly (close_polygon)
        if self.tool == "brush":
            if len(self.pts) >= 1:
                self.shapes.append({"type": "brush", "op": self.op,
                                    "radius": float(self.brush_px),
                                    "vertices": self._shape_verts(self.pts)})
            self.pts = []
            return
        if len(self.pts) < 2 or not self._moved:
            self.pts = []
            self.note = "drag to give the shape an extent"
            return
        (x0, y0), (x1, y1) = self.pts[0], self.pts[-1]
        if self.tool == "circle":
            r = math.hypot(x1 - x0, y1 - y0)
            if r > 0:
                self.shapes.append({"type": "circle", "op": self.op,
                                    "center": [round(y0, 2), round(x0, 2)],
                                    "radius": round(r, 2)})
        else:                                   # rect | ellipse — a two-corner box
            self.shapes.append({"type": self.tool, "op": self.op,
                                "vertices": self._shape_verts([(x0, y0), (x1, y1)])})
        self.pts = []

    def close_polygon(self) -> None:
        """Finish a polygon in progress (double-click / Enter)."""
        if self.req.kind != "shapes" or self.tool != "polygon":
            return
        if len(self.pts) >= 3:
            self.shapes.append({"type": "polygon", "op": self.op,
                                "vertices": self._shape_verts(self.pts)})
        else:
            self.note = "a polygon needs at least three points"
        self.pts = []

    def add_action(self, kind: str) -> None:
        """Append a stateful whole-mask action (``invert`` / ``clear``)."""
        if self.req.kind == "shapes" and kind in ("invert", "clear"):
            self.shapes.append({"type": kind})

    def undo_shape(self) -> None:
        if self.pts:
            self.pts = []
        elif self.shapes:
            self.shapes.pop()

    @staticmethod
    def _shape_verts(pts: Sequence[Tuple[float, float]]) -> List[List[float]]:
        """(x, y) plane points → the kernel's ``[[y, x], …]`` vertex list."""
        return [[round(y, 2), round(x, 2)] for x, y in pts]

    # ── area ──────────────────────────────────────────────────────────────────
    def _end_area(self, x: float, y: float) -> None:
        """A click takes the clicked object's measured size; a drag takes the drawn blob's.

        The click path is the better one when it is available (the probe is the object's own
        voxel count from the Label table, so it is the same number the size filter will
        compare against) but it needs something already segmented. The draw path always
        works, which is what makes this usable while building a graph from scratch — before
        any labels exist there is nothing to click."""
        drew = self._moved and len(self.pts) >= 3
        if drew:
            px_area = polygon_area(self.pts)
            if px_area <= 0:
                self.note = "that outline enclosed no pixels"
                return
            self.picked.append(self._area_to_unit(px_area, extruded=True))
        elif self.probe is not None and self.probe > 0:
            self.picked.append(self._area_to_unit(float(self.probe), extruded=False))
        else:
            self.note = ("no object there to measure — drag to draw the size instead"
                         if self.probe is None else "that object measured zero")
            return
        if self.req.peer and self.phase == 0:
            self.phase = 1
            self.pts = []
        else:
            self.done = True

    def _area_to_unit(self, count: float, *, extruded: bool) -> float:
        """A pixel/voxel COUNT → the socket's unit.

        ``um2``: count × (µm/px)². ``um3``: the same, times the z step — correct for a
        clicked object (whose probe is a true voxel count) and, for a DRAWN outline, an
        explicit one-plane extrusion, which :meth:`readout` labels as such rather than
        letting it pass for a measured volume."""
        lat = self.calib.lateral
        area = count * lat * lat
        if self.req.unit == "um3":
            return area * self.calib.axial
        return area

    # ── rect (a bounding box, all four bounds from one drag) ──────────────────
    def box(self) -> Optional[Tuple[float, float, float, float]]:
        """The dragged rectangle as ``(x0, y0, x1, y1)`` in plane pixels, normalized so
        ``x0 <= x1``. ``None`` until two corners exist. Shared by the value maths and the
        viewer's rubber band, so the numbers committed are the ones drawn."""
        if len(self.pts) < 2:
            return None
        (ax, ay), (bx, by) = self.pts[0], self.pts[-1]
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    def _rect_values(self) -> Dict[str, Any]:
        """The box → the node's four bound params, as an integer PIXEL window.

        ``floor`` the starts and ``ceil`` the ends: the crop node's end bounds are EXCLUSIVE
        (a Python slice), so this makes ``[start:end]`` cover exactly the pixels the
        rectangle was drawn over — no off-by-one, and no need for the user to reason about
        which edge is inclusive. Negative values cannot arise: the surfaces refuse clicks
        outside the image, and a degenerate axis is widened to one pixel rather than
        committing an empty window the compute would refuse."""
        b = self.box()
        if b is None or not self.req.bounds:
            return {}
        x0, y0, x1, y1 = b
        iy0, iy1 = int(math.floor(y0)), int(math.ceil(y1))
        ix0, ix1 = int(math.floor(x0)), int(math.ceil(x1))
        iy1 = max(iy1, iy0 + 1)
        ix1 = max(ix1, ix0 + 1)
        by_name = {"y0": iy0, "y1": iy1, "x0": ix0, "x1": ix1}
        # Keyed by NAME, and only the names this node declared — a node whose bound group is
        # spelt differently gets nothing rather than a wrong guess.
        return {n: by_name[n] for n in self.req.bounds if n in by_name}

    # ── nudge_xy (two clicks → the µm offset pair that lines them up) ────────────
    def _nudge_values(self) -> Dict[str, Any]:
        """The two clicks → the new ``(offset_y, offset_x)``: the current nudge PLUS the move
        that carries the second click (the feature as the overlay draws it) onto the first
        (the same feature in the primary). Through the engine's own
        :func:`nodegraph.placement.nudge_delta_um`, which is flip-independent — a flip
        mirrors which source pixel the secondary's box samples, never where the box sits.
        The bound group is ``(y name, x name)`` in that order."""
        from nodegraph.placement import nudge_delta_um
        if len(self.pts) < 2 or len(self.req.bounds) != 2:
            return {}
        (xa, ya), (xb, yb) = self.pts[0], self.pts[1]
        md = {"pixel_size_um": self.calib.lateral}
        if self.req.mirror is not None:
            md["canvas_flip"] = list(self.req.mirror)
        got = nudge_delta_um(md, None, (ya, xa), (yb, xb))
        if got is None:
            return {}
        base = dict(self.req.base or ())
        ny, nx = self.req.bounds
        return {ny: round(float(base.get(ny, 0.0)) + got[0], 4),
                nx: round(float(base.get(nx, 0.0)) + got[1], 4)}

    # ── radius / grid magnitudes ──────────────────────────────────────────────
    def _magnitude(self, x: float, y: float) -> float:
        """The gesture's extent in the socket's own unit."""
        if not self.pts:
            return 0.0
        x0, y0 = self.pts[0]
        if self.req.kind == "radius":
            px = math.hypot(x - x0, y - y0)
        else:                                   # grid: a square box, so the longer edge
            px = max(abs(x - x0), abs(y - y0))
        if self._unit_is_px():
            return px
        return px * self.calib.lateral

    def _unit_is_px(self) -> bool:
        """A px-unit socket takes the raw pixel count; anything else is physical."""
        return self.req.unit_for(self._target()) in ("px", "")

    # ── the result ────────────────────────────────────────────────────────────
    def values(self) -> Dict[str, Any]:
        """The param edits this pick commits — ``{socket_name: value}``, empty until
        :attr:`done` (except ``shapes``, whose accumulated list is always committable)."""
        req = self.req
        if req.kind == "shapes":
            return {req.socket: json.dumps(self.shapes)}
        if not self.done:
            return {}
        if req.kind == "rect":
            return self._rect_values()
        if req.kind == "nudge_xy":
            return self._nudge_values()
        if req.kind == "distance":
            (x0, y0), (x1, y1) = self.pts[0], self.pts[1]
            d = math.hypot(x1 - x0, y1 - y0)
            return self._typed({req.socket: d if self._unit_is_px()
                                else d * self.calib.lateral})
        if len(self.picked) >= 2 and req.peer:
            return self._typed(ordered_pair(req.socket, req.peer,
                                            self.picked[0], self.picked[1]))
        if self.picked:
            return self._typed({req.socket: self.picked[0]})
        return {}

    def _typed(self, vals: Dict[str, float]) -> Dict[str, Any]:
        """Round each value to its socket's type — an INT socket must not receive 16.4."""
        return {name: (int(round(v)) if self.req.integral_for(name)
                       else round(float(v), 4))
                for name, v in vals.items()}

    # ── the banner ────────────────────────────────────────────────────────────
    def readout(self) -> str:
        """The live one-line value preview under the banner."""
        if self.note:
            return self.note
        req = self.req
        if req.kind == "shapes":
            n = len(self.shapes)
            live = f", drawing {self.tool}" if self.pts else ""
            return f"{n} shape{'' if n == 1 else 's'}{live}"
        if req.kind == "nudge_xy":
            if len(self.pts) < 2:
                return ("click the feature in the PRIMARY" if not self.pts
                        else "now click the same feature in the overlaid source")
            v = self._nudge_values()
            base = dict(req.base or ())
            parts = [f"{n} {float(base.get(n, 0.0)):+.2f} → {v[n]:+.2f} µm" for n in v]
            tail = ("" if self.calib.calibrated
                    else "  ⚠ uncalibrated file — values are in pixels")
            return " · ".join(parts) + tail
        if req.kind == "rect":
            # The box reads as a window plus its size, which is what a crop is about: the
            # four raw bounds are in the pills already, and "512 × 384 px" is the number the
            # user is actually judging while dragging.
            b = self.box()
            if b is None:
                return self._prompt()
            x0, y0, x1, y1 = b
            v = self._rect_values() if self.done else None
            iy0, iy1 = int(math.floor(y0)), max(int(math.ceil(y1)), int(math.floor(y0)) + 1)
            ix0, ix1 = int(math.floor(x0)), max(int(math.ceil(x1)), int(math.floor(x0)) + 1)
            if v:
                iy0, iy1 = v.get("y0", iy0), v.get("y1", iy1)
                ix0, ix1 = v.get("x0", ix0), v.get("x1", ix1)
            return (f"y {iy0}–{iy1} · x {ix0}–{ix1}   "
                    f"({ix1 - ix0} × {iy1 - iy0} px kept)")
        vals = self.values()
        if not vals:
            live = self._live_magnitude()
            if live is None:
                return self._prompt()
            return f"{self._label_for(self._target())} ≈ {self._fmt(live)}"
        parts = [f"{self._label_for(k)} = {self._fmt(v, name=k)}" for k, v in vals.items()]
        tail = ""
        if req.unit == "um3" and self._moved:
            tail = "  (drawn outline, one plane thick)"
        if not self.calib.calibrated and req.unit in ("um", "um2", "um3"):
            tail += "  ⚠ uncalibrated file — values are in pixels"
        return " · ".join(parts) + tail

    def _prompt(self) -> str:
        if self.req.peer and self.phase == 1:
            return f"now aim {self.req.peer_label or self.req.peer}"
        return ""

    def _target(self) -> str:
        return (self.req.peer if self.phase == 1 and self.req.peer else self.req.socket)

    def _label_for(self, name: str) -> str:
        if name == self.req.peer:
            return self.req.peer_label or name
        return self.req.label or name

    def _live_magnitude(self) -> Optional[float]:
        """The magnitude of the gesture as it currently stands (nothing released yet)."""
        if len(self.pts) < 2:
            return None
        if self.req.kind in ("radius", "grid"):
            return self._magnitude(*self.pts[-1])
        if self.req.kind == "distance":
            (x0, y0), (x1, y1) = self.pts[0], self.pts[-1]
            d = math.hypot(x1 - x0, y1 - y0)
            return d if self._unit_is_px() else d * self.calib.lateral
        if self.req.kind == "area" and len(self.pts) >= 3:
            return self._area_to_unit(polygon_area(self.pts), extruded=True)
        return None

    def _fmt(self, v: float, *, name: str = "") -> str:
        target = name or self._target()
        unit = self.req.unit_for(target)
        txt = (f"{int(round(v))}" if self.req.integral_for(target)
               else f"{v:.3f}".rstrip("0").rstrip("."))
        return f"{txt} {UNIT_TEXT.get(unit, unit)}".strip()


#: Unit labels for the banner (the inspector's own table is Qt-side; this one is plain text
#: so the readout is checkable headlessly).
UNIT_TEXT = {"um": "µm", "um_axial": "µm(z)", "um2": "µm²", "um3": "µm³",
             "nm": "nm", "s": "s", "px": "px", "": ""}


def polygon_area(pts: Sequence[Tuple[float, float]]) -> float:
    """The absolute area of the closed polygon through ``pts``, in square pixels
    (the shoelace formula). Fewer than three points enclose nothing."""
    n = len(pts)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


# ── the instant + histogram surfaces ─────────────────────────────────────────

def instant_values(req: PickRequest, *, channel: int, frame: int,
                   channels: Sequence[int], z_picks: Sequence[int] = (),
                   z_total: int = 0, m_picks: Sequence[int] = (),
                   t_picks: Sequence[int] = (), position: int = 0,
                   plane: int = 0) -> Dict[str, Any]:
    """Resolve an :data:`INSTANT_KINDS` pick from the viewer's current cursor state."""
    if req.kind == "channel":
        return {req.socket: int(channel)}
    if req.kind == "frame":
        return {req.socket: int(frame)}
    if req.kind == "channels":
        return {req.socket: ",".join(str(int(c)) for c in channels)}
    if req.kind == "zrange":
        return _zrange_values(req, z_picks, z_total)
    if req.kind == "frames":
        return _frames_values(req, m_picks, t_picks, z_picks,
                             cursor=(int(position), int(frame), int(plane)))
    return {}


def _frames_values(req: PickRequest, m_picks: Sequence[int], t_picks: Sequence[int],
                   z_picks: Sequence[int], *,
                   cursor: Tuple[int, int, int]) -> Dict[str, Any]:
    """The M/T/Z strip selections → ONE frame-spec string (``"m0-2,t3,z1-4"``).

    One value, not three, because the thing being selected is one thing: "these frames". It
    is also what makes the gesture work for a SINGLE frame — an axis with nothing ticked
    takes the index **the cursor is on**, so looking at position 2, timepoint 7 and pressing
    Pick crops to exactly the frame on screen without ticking anything first. That includes
    ``z``: what is displayed is one plane, so "the frame I am looking at" is that plane, and
    since the spec lands in a visible, editable socket, deleting the ``z`` section is how you
    say "the whole volume of that frame".

    The picks stay SPARSE, which is the whole reason the socket takes text: the strips are
    multi-selects, so ticking timepoints 2, 5 and 9 is a statement about three frames, and
    taking their span (what :func:`_zrange_values` must do for a pair of INT bounds) would
    silently re-admit the six in between. Contiguous runs collapse to ``0-3`` through the
    canonical formatter, so one selection has exactly one spelling and the value stays
    hand-editable, indistinguishable from a typed one.
    """
    # The formatter lives next to the PARSER (`nodegraph.metadata`) rather than here, and is
    # deliberately not `framestrip.compact_list`, which looks like the same function and is
    # not: that one is a tooltip formatter (en-dashes, ", " separators, "—" for empty, and
    # elision past six runs), so its output is both unparseable and, on a long selection,
    # incomplete. This is a VALUE, and it must round-trip through the socket's own parser.
    from nodegraph.metadata import format_frame_spec
    picked = {"m": m_picks, "t": t_picks, "z": z_picks}
    return {req.socket: format_frame_spec(
        {a: (sorted({int(i) for i in picked[a]}) or [cur])
         for a, cur in zip(("m", "t", "z"), cursor)})}


def _zrange_values(req: PickRequest, z_picks: Sequence[int],
                   z_total: int) -> Dict[str, Any]:
    """The Z strip's picked planes → a ``(z0, z1)`` slice window.

    The strip is a multi-select, so the picks can be sparse (planes 2, 5, 9); a crop window
    is necessarily contiguous, so this takes their SPAN — the smallest window containing
    every picked plane. That is the only reading that cannot silently drop a plane the user
    ticked. Nothing picked means the whole stack, matching the socket's own unset default
    rather than committing an empty range."""
    zs = sorted({int(z) for z in z_picks})
    if zs:
        z0, z1 = zs[0], zs[-1] + 1        # end is EXCLUSIVE, like the crop bounds
    else:
        z0, z1 = 0, max(1, int(z_total))
    by_name = {"z0": z0, "z1": z1}
    return {n: by_name[n] for n in req.bounds if n in by_name}


def histogram_values(req: PickRequest, *, lo: float, hi: float, vmin: float, vmax: float,
                     gamma: float) -> Dict[str, Any]:
    """Resolve a :data:`HISTOGRAM_KINDS` pick from the live LUT state.

    ``percentile`` converts the contrast WINDOW (``lo``/``hi``, in native intensity units)
    into the 0–100 positions it occupies within the data range ``vmin``…``vmax``. That is a
    linear position, not a true data percentile — the honest name for what the handles
    encode — and it is what the consumers of these sockets want: ``enhance.normalize`` maps
    its low/high percentages onto the same span. A degenerate range commits the full 0–100
    rather than dividing by zero."""
    if req.kind == "gamma":
        return {req.socket: round(float(gamma), 4)}
    span = float(vmax) - float(vmin)
    if not (span > 0 and math.isfinite(span)):
        p_lo, p_hi = 0.0, 100.0
    else:
        p_lo = max(0.0, min(100.0, (float(lo) - float(vmin)) / span * 100.0))
        p_hi = max(0.0, min(100.0, (float(hi) - float(vmin)) / span * 100.0))
    if req.peer:
        return {k: round(v, 4)
                for k, v in ordered_pair(req.socket, req.peer, p_lo, p_hi).items()}
    return {req.socket: round(p_lo, 4)}


__all__ = [
    "PickRequest", "PickSession", "Calibration", "PICK_HELP", "PICK_ACTION", "PICK_GLYPH",
    "UNIT_TEXT",
    "INSTANT_KINDS", "HISTOGRAM_KINDS", "CANVAS_KINDS",
    "instant_values", "histogram_values", "ordered_pair", "polygon_area", "request_for",
]
