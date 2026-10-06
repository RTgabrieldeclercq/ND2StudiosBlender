"""The start card on an empty page (2026-07-27; per page kind since V4.00 step 11).

A page that holds no node yet — or nothing but the Page Input it was started with — shows
this card. What it offers depends on the page's KIND, because the first move is different
on each:

* an **Image Input** page invites a load (*Load image…*, *Load sequence…*) — what is loaded
  there is published for the later pages — or the four-page *Example graph*;
* a **Refinement**, **Processing** or **Analysis** page says what its Page Input reads and
  offers that kind's **page recipes** as one-click buttons, *More recipes…* (the New page
  dialog), *Link to a master…* and *Start empty*; with nothing upstream yet it says so and
  offers *Go to Image Input*. On these pages the card is a BANNER along the canvas's bottom
  edge, so the Page Input it was seeded with stays in view above it;
* a **Free** page (and a document outside a workspace) keeps the original card: *Load
  image… / Browse nodes / Example graph*.

Every card can be dismissed (✕, or *Start empty*); the window remembers that per page for
the session. It hides itself once the page holds a node of its own.

It is chrome only: it owns no document state, and it forwards a palette **drag-and-drop
that lands on it** to the canvas underneath (:data:`op_dropped`) — otherwise the card would
swallow the very drop it is asking for, sitting where you would aim it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout,
                               QWidget)

from nodelab_v2 import theme as T

#: the palette's drag payload (mirrors :class:`~nodelab_v2.scene.GraphView`)
OP_MIME = "application/x-nd2studios-op"
#: how many of a kind's page recipes get their own button (the rest: *More recipes…*)
MAX_RECIPE_BUTTONS = 3
#: the banner's widest extent, and its gap from the canvas's bottom edge
BANNER_MAX_W, BANNER_MARGIN = 780, 16
#: a canvas shorter than this gets the one-row banner, so the Page Input framed at its
#: top left stays clear of it
COMPACT_BELOW = 380

_KIND_LABELS = {"input": "Image Input", "refine": "Image Refinement",
                "process": "Image Processing", "analyze": "Analysis", "free": "Free"}


class WelcomeCard(QWidget):
    """The start card of an empty page — a centred card, or a bottom banner on a page that
    reads earlier pages (:meth:`configure`)."""

    SIZE = (452, 248)
    GLYPH_H = 66                  # painted "drop a node here" square above the text

    load_image_requested = Signal()
    load_sequence_requested = Signal()
    browse_nodes_requested = Signal()
    example_requested = Signal()
    #: *More recipes…*: the New page dialog, filling this page
    recipe_requested = Signal()
    #: a recipe button: fill this page from the page recipe of that name
    recipe_chosen = Signal(str)
    #: *Link to a master…*: the New page dialog on its linked start
    link_requested = Signal()
    #: *Go to Image Input* (nothing upstream to read yet)
    goto_input_requested = Signal()
    #: ✕ / *Start empty*
    dismissed = Signal()
    op_dropped = Signal(str, QPointF)     # (op_key, scene position) — drop passthrough

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("welcome")
        self.setAcceptDrops(True)
        self.setAutoFillBackground(False)
        self._banner = False
        self._compact = False
        self._sig: Optional[tuple] = None
        self._body: Optional[QWidget] = None
        self._title = QLabel()
        self._sub = QLabel()
        self._hints: List[QLabel] = []
        self._buttons: List[QPushButton] = []
        self._btn_load: Optional[QPushButton] = None     # the card's primary button
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._close = QToolButton(self)
        self._close.setText("✕")
        self._close.setToolTip("Hide this card for this page — the page switcher's New page… "
                               "offers the same starts")
        self._close.setCursor(Qt.PointingHandCursor)
        self._close.setAutoRaise(True)
        self._close.clicked.connect(self.dismissed.emit)
        self.resize(*self.SIZE)
        self.configure("")
        parent.installEventFilter(self)
        self.hide()

    # ── what the card offers ─────────────────────────────────────────────────
    def configure(self, kind: str, *, reads: str = "", has_upstream: bool = False,
                  recipes: Sequence[Tuple[str, str]] = (), description: str = "") -> None:
        """Word the card for a page of ``kind``: ``reads`` — what its Page Input reads (a
        label, ``""`` when unbound); ``has_upstream`` — an earlier page offers a named Output;
        ``recipes`` — ``(name, description)`` of the kind's page recipes; ``description`` —
        the kind's one-line purpose. Rebuilt only when something changed."""
        banner = kind in ("refine", "process", "analyze")
        sig = (kind, reads, bool(has_upstream), tuple(recipes), description, banner)
        if sig == self._sig:
            return
        self._sig, self._banner = sig, banner
        self._compact = banner and self._want_compact()
        self._rebuild_body()
        self.restyle()
        self.recenter()

    def _want_compact(self) -> bool:
        """A banner on a SHORT canvas is one row (:data:`COMPACT_BELOW`), so it stays clear of
        the Page Input framed at the canvas's top left."""
        par = self.parentWidget()
        return par is not None and 0 < par.height() < COMPACT_BELOW

    def _rebuild_body(self) -> None:
        kind, reads, has_upstream, recipes, description, banner = self._sig
        if self._body is not None:
            # hidden and detached NOW: deleteLater alone leaves it painting under the new
            # body until the event loop next deletes deferred objects
            self._outer.removeWidget(self._body)
            self._body.hide()
            self._body.setParent(None)
            self._body.deleteLater()
        self._hints, self._buttons = [], []
        if not banner:
            self._body = self._build_card(kind)
        elif self._compact:
            self._body = self._build_compact(kind, reads, has_upstream, recipes)
        else:
            self._body = self._build_banner(kind, reads, has_upstream, recipes, description)
        self._outer.addWidget(self._body)

    def _build_compact(self, kind: str, reads: str, has_upstream: bool,
                       recipes: Sequence[Tuple[str, str]]) -> QWidget:
        """The banner in one line of text and one row of buttons, for a short canvas: the
        kind's first two recipes, *More recipes…* (which also offers *Linked to a master*),
        *Start empty*."""
        body = QWidget(self)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 8, 34, 8)
        lay.setSpacing(6)
        label = _KIND_LABELS.get(kind, kind)
        if reads:
            tail = f" — reads <b>{reads}</b>"
        elif has_upstream:
            tail = " — its Page Input reads nothing yet"
        else:
            tail = " — nothing to read yet: load an image on Image Input"
        self._title = QLabel(f"<b>Start the {label} page</b>{tail}", body)
        self._title.setTextFormat(Qt.RichText)
        self._sub = self._title
        lay.addWidget(self._title)
        row = QHBoxLayout()
        row.setSpacing(8)
        if not has_upstream:
            row.addWidget(self._button("Go to Image Input", "Show the Image Input page",
                                       self.goto_input_requested.emit, primary=True))
        for i, (name, desc) in enumerate(list(recipes)[:2]):
            row.addWidget(self._button(
                name, f"Page recipe: {desc}" if desc else "Page recipe",
                lambda n=name: self.recipe_chosen.emit(n),
                primary=(i == 0 and has_upstream)))
        row.addWidget(self._button("More recipes…", "New page… on this page: every page recipe "
                                   "for this kind, linking to a master page, the Output to "
                                   "read, a name", self.recipe_requested.emit))
        row.addStretch(1)
        row.addWidget(self._button("Start empty", "Hide this card and build the page by hand "
                                   "(the Nodes palette's Pages band holds Page Input / Output)",
                                   self.dismissed.emit))
        lay.addLayout(row)
        self._btn_load = self._buttons[0]
        return body

    def _button(self, text: str, tip: str, signal, *, primary: bool = False) -> QPushButton:
        # "&" is literal ("Smooth & threshold"), not a keyboard mnemonic
        b = QPushButton(text.replace("&", "&&"))
        b.setProperty("label", text)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        if primary:
            b.setProperty("role", "primary")
        b.clicked.connect(lambda _=False: signal())
        self._buttons.append(b)
        return b

    def _build_card(self, kind: str) -> QWidget:
        body = QWidget(self)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(26, self.GLYPH_H, 26, 20)
        lay.setSpacing(3)
        self._title = QLabel(body)
        self._sub = QLabel(body)
        for lab in (self._title, self._sub):
            lab.setAlignment(Qt.AlignCenter)
            lab.setWordWrap(True)
            lay.addWidget(lab)
        lay.addSpacing(9)
        if kind == "input":
            self._title.setText("Load your images")
            self._sub.setText("This is the Image Input page: what you load here is published "
                              "for the later pages.")
            hints = ("<b>Ctrl+L</b> loads ND2/TIFF files — each becomes a Page Output named "
                     "after its file",
                     "<b>Ctrl+Shift+L</b> loads a numbered file sequence as one series",
                     "Files dropped on any canvas land here too")
            buttons = [("Load image…", "File → Load ND2/TIFF file… (Ctrl+L)",
                        self.load_image_requested.emit, True),
                       ("Load sequence…", "File → Load file sequence… (Ctrl+Shift+L)",
                        self.load_sequence_requested.emit, False),
                       ("Example graph", "A small analysis spread over the four standard pages",
                        self.example_requested.emit, False)]
        else:
            self._title.setText("Start your graph")
            self._sub.setText("The canvas is empty — place a node to begin.")
            hints = ("Double-click a node in the <b>Nodes</b> palette — or drag it onto the "
                     "canvas",
                     "<b>Ctrl+L</b> loads an ND2/TIFF and drops a source node with its channels",
                     "Double-click any node to preview it · <b>Ctrl+Space</b> maximizes the "
                     "canvas")
            buttons = [("Load image…", "File → Load ND2/TIFF file… (Ctrl+L)",
                        self.load_image_requested.emit, True),
                       ("Browse nodes", "Jump to the Nodes palette search",
                        self.browse_nodes_requested.emit, False),
                       ("Example graph", "A small analysis spread over the four standard pages",
                        self.example_requested.emit, False)]
        for text in hints:
            lab = QLabel(text, body)
            lab.setAlignment(Qt.AlignCenter)
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lay.addWidget(lab)
            self._hints.append(lab)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.setSpacing(8)
        for text, tip, sig, primary in buttons:
            row.addWidget(self._button(text, tip, sig, primary=primary))
        lay.addLayout(row)
        self._btn_load = self._buttons[0]
        return body

    def _build_banner(self, kind: str, reads: str, has_upstream: bool,
                      recipes: Sequence[Tuple[str, str]], description: str) -> QWidget:
        body = QWidget(self)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 12, 34, 12)
        lay.setSpacing(4)
        label = _KIND_LABELS.get(kind, kind)
        self._title = QLabel(f"Start the {label} page", body)
        lay.addWidget(self._title)
        if reads:
            sub = f"Its Page Input reads <b>{reads}</b>."
        elif has_upstream:
            sub = "Its Page Input reads nothing yet — pick a Source on it, or start from a recipe."
        else:
            sub = ("Nothing to read yet — load an image on the Image Input page first; its "
                   "Output is what this page reads.")
        self._sub = QLabel(sub, body)
        self._sub.setTextFormat(Qt.RichText)
        self._sub.setWordWrap(True)
        lay.addWidget(self._sub)
        if description:
            hint = QLabel(description, body)
            hint.setWordWrap(True)
            lay.addWidget(hint)
            self._hints.append(hint)
        lay.addSpacing(4)
        # two rows, so the buttons never run past the banner's edge: the starts first (Go to
        # Image Input, the kind's page recipes), then the other ways to begin
        first = QHBoxLayout()
        first.setSpacing(8)
        if not has_upstream:
            first.addWidget(self._button("Go to Image Input", "Show the Image Input page",
                                         self.goto_input_requested.emit, primary=True))
        for i, (name, desc) in enumerate(list(recipes)[:MAX_RECIPE_BUTTONS]):
            first.addWidget(self._button(
                name, f"Page recipe: {desc}" if desc else "Page recipe",
                lambda n=name: self.recipe_chosen.emit(n),
                primary=(i == 0 and has_upstream)))
        first.addStretch(1)
        lay.addLayout(first)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._button("More recipes…", "New page… on this page: every page recipe "
                                   "for this kind, the Output to read, a name",
                                   self.recipe_requested.emit))
        row.addWidget(self._button("Link to a master…", "Make this page a linked copy of a "
                                   "master page — its graph, values of its own",
                                   self.link_requested.emit))
        row.addStretch(1)
        row.addWidget(self._button("Start empty", "Hide this card and build the page by hand "
                                   "(the Nodes palette's Pages band holds Page Input / Output)",
                                   self.dismissed.emit))
        lay.addLayout(row)
        self._btn_load = self._buttons[0]
        return body

    def button(self, text: str) -> Optional[QPushButton]:
        """The card's button labelled ``text`` (``None`` when it offers none)."""
        return next((b for b in self._buttons if b.property("label") == text), None)

    def button_texts(self) -> List[str]:
        return [str(b.property("label")) for b in self._buttons]

    @property
    def is_banner(self) -> bool:
        return self._banner

    # ── placement ──────────────────────────────────────────────────────────────
    def eventFilter(self, obj, ev) -> bool:        # noqa: N802 — Qt override
        if obj is self.parentWidget() and ev.type() == QEvent.Resize:
            self.recenter()
        return False

    def recenter(self) -> None:
        """A card sits in the middle of the canvas (shrinking if the canvas is smaller); a
        banner spans the bottom edge, so whatever the page already holds stays visible."""
        par = self.parentWidget()
        if par is None:
            return
        if self._banner:
            if self._want_compact() != self._compact and self._sig is not None:
                self._compact = not self._compact      # the canvas grew or shrank past it
                self._rebuild_body()
                self.restyle()
            w = max(260, min(BANNER_MAX_W, par.width() - 2 * BANNER_MARGIN))
            body = self._body
            h = 0
            if body is not None:
                body.ensurePolished()
                h = max(body.minimumSizeHint().height(),
                        body.heightForWidth(w) if body.hasHeightForWidth()
                        else body.sizeHint().height())
            floor = 60 if self._compact else 110
            h = max(floor, min(h, max(floor, par.height() - 2 * BANNER_MARGIN)))
            if (w, h) != (self.width(), self.height()):
                self.resize(w, h)
            self.move(max(0, (par.width() - w) // 2), max(0, par.height() - h - BANNER_MARGIN))
        else:
            w = min(self.SIZE[0], max(240, par.width() - 24))
            h = min(self.SIZE[1], max(180, par.height() - 24))
            if (w, h) != (self.width(), self.height()):
                self.resize(w, h)
            self.move(max(0, (par.width() - w) // 2), max(0, (par.height() - h) // 2))
        self._close.move(self.width() - self._close.width() - 6, 6)
        self._close.raise_()

    def resizeEvent(self, e) -> None:              # noqa: N802 — Qt override
        super().resizeEvent(e)
        self._close.move(self.width() - self._close.width() - 6, 6)

    def setVisible(self, on: bool) -> None:        # noqa: N802 — Qt override
        if on:
            self.recenter()
        super().setVisible(on)

    # ── palette drop passthrough ───────────────────────────────────────────────
    def dragEnterEvent(self, e) -> None:           # noqa: N802 — Qt override
        e.acceptProposedAction() if e.mimeData().hasFormat(OP_MIME) else e.ignore()

    def dragMoveEvent(self, e) -> None:            # noqa: N802 — Qt override
        e.acceptProposedAction() if e.mimeData().hasFormat(OP_MIME) else e.ignore()

    def dropEvent(self, e) -> None:                # noqa: N802 — Qt override
        """A node dropped ON the card lands on the canvas beneath it, at that point."""
        view = self.parentWidget()
        if not e.mimeData().hasFormat(OP_MIME) or view is None:
            e.ignore()
            return
        op = bytes(e.mimeData().data(OP_MIME)).decode("utf-8")
        pt = self.mapTo(view, e.position().toPoint())
        self.op_dropped.emit(op, view.mapToScene(pt))
        e.acceptProposedAction()

    # ── paint ──────────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:              # noqa: N802 — Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        p.setPen(Qt.NoPen)
        p.setBrush(T.PANEL)
        p.drawRoundedRect(r, 13, 13)
        p.setPen(QPen(T.BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, 13, 13)
        if not self._banner:
            # the "place a node here" affordance: a dashed accent square with a plus
            side = 40.0
            box = QRectF(r.center().x() - side / 2, r.top() + 16, side, side)
            pen = QPen(T.alpha(T.ACCENT, 170), 1.6, Qt.DashLine)
            pen.setDashPattern([3.0, 2.6])
            p.setPen(pen)
            p.drawRoundedRect(box, 7, 7)
            p.setPen(QPen(T.ACCENT, 2.0, Qt.SolidLine, Qt.RoundCap))
            c = box.center()
            p.drawLine(QPointF(c.x() - 8, c.y()), QPointF(c.x() + 8, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() + 8))
        else:
            # a kind-coloured accent bar down the banner's left edge
            p.setPen(Qt.NoPen)
            p.setBrush(T.ACCENT)
            p.drawRoundedRect(QRectF(1, 10, 3, self.height() - 20), 1.5, 1.5)
        p.end()

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        if self._compact:
            self._title.setFont(QFont(T.SANS, 10))     # rich text carries its own bold
        else:
            tf = QFont(T.SANS, 12 if self._banner else 13)
            tf.setBold(True)
            self._title.setFont(tf)
            self._sub.setFont(QFont(T.SANS, 9))
        for lab in self._hints:
            lab.setFont(QFont(T.SANS, 8))
        self.setStyleSheet(f"""
            QWidget#welcome {{ background:transparent; }}
            QWidget {{ background:transparent; }}
            QLabel {{ background:transparent; color:{T.INK_2.name()}; }}
        """ + T.controls_qss() + f"""
            QPushButton[role="primary"] {{ background:{T.ACCENT.name()};
                color:{T.ACCENT_INK.name()}; border:1px solid {T.ACCENT.name()}; }}
            QPushButton[role="primary"]:hover {{
                background:{T.mix(T.ACCENT, T.INK, 0.12).name()}; }}
            QToolButton {{ background:transparent; border:0; color:{T.MUTED.name()};
                padding:2px 5px; }}
            QToolButton:hover {{ color:{T.INK.name()}; }}
        """)
        self._title.setStyleSheet(f"color:{T.INK.name()}; background:transparent;")
        for lab in self._hints:
            lab.setStyleSheet(f"color:{T.MUTED.name()}; background:transparent;")
        self.update()


__all__ = ["WelcomeCard", "OP_MIME", "MAX_RECIPE_BUTTONS", "COMPACT_BELOW"]
