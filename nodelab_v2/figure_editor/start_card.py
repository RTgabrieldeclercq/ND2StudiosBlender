"""The start card of an empty figure page (V5.00 step 3): *What do you want to make?*

Six goal cards, each saying what it needs and what the upstream pages provide right now
(from the envelopes of their Page Outputs — no pull), and one button per image source to
start with. Hidden once the figure holds an item.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from nodelab_v2 import theme as T

GOALS: Tuple[Tuple[str, str, str, str], ...] = (
    ("overlay", "Overlay", "An image with results drawn on it — labels, tracks, a heat "
                           "layer, vectors, a scale bar.", "an image stream"),
    ("montage", "Montage", "Tiles of one image along positions, time, planes or channels, "
                           "one shared display range.", "an image stream"),
    ("composite", "Plot + Overlay", "Panels of images and charts on one journal page, "
                                    "lettered.", "an image stream and a table"),
    ("plot", "Plot", "A chart of measured columns — scatter, bars, distributions, time "
                     "series, MSD.", "a table"),
    ("table", "Table", "A summary table per group, formatted for a paper.", "a table"),
    ("movie", "Movie", "A movie of the result — opens the Movie Editor on a Free page.",
     "an image stream"),
)

#: goals the editor can start in this build (the others name the step that adds them)
AVAILABLE = {"overlay": "", "montage": "", "composite": "",
             "plot": "charts arrive in step 4", "table": "tables arrive in step 5",
             "movie": "the movie hand-off arrives in step 7"}


class _GoalCard(QFrame):
    chosen = Signal(str)

    def __init__(self, key: str, title: str, blurb: str) -> None:
        super().__init__()
        self.key = key
        self.setObjectName("goalCard")
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        self.title = QLabel(title)
        self.title.setProperty("role", "title")
        self.blurb = QLabel(blurb)
        self.blurb.setWordWrap(True)
        self.blurb.setProperty("role", "muted")
        self.have = QLabel("")
        self.have.setWordWrap(True)
        self.have.setProperty("role", "have")
        for w in (self.title, self.blurb, self.have):
            lay.addWidget(w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def mousePressEvent(self, e) -> None:                                 # noqa: N802
        if self.isEnabled():
            self.chosen.emit(self.key)
        super().mousePressEvent(e)


class StartCard(QWidget):
    """The card. ``goal_chosen(key)``, ``source_chosen(value)`` (an image source to start
    with) and ``dismissed`` are what the panel answers."""

    goal_chosen = Signal(str)
    source_chosen = Signal(str)
    dismissed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("startCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(10)
        self.heading = QLabel("What do you want to make?")
        self.heading.setProperty("role", "heading")
        outer.addWidget(self.heading)
        self.sub = QLabel("A figure page reads what the pages before it publish. Pick a goal; "
                          "the sources it needs are listed on the left, and dragging one "
                          "onto the page places it.")
        self.sub.setWordWrap(True)
        self.sub.setProperty("role", "muted")
        outer.addWidget(self.sub)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.cards: Dict[str, _GoalCard] = {}
        for k, (key, title, blurb, _needs) in enumerate(GOALS):
            card = _GoalCard(key, title, blurb)
            card.chosen.connect(self.goal_chosen)
            self.cards[key] = card
            grid.addWidget(card, k // 3, k % 3)
        outer.addLayout(grid)
        self.sources_label = QLabel("Start with an image:")
        self.sources_label.setProperty("role", "muted")
        outer.addWidget(self.sources_label)
        self.sources_row = QHBoxLayout()
        self.sources_row.setSpacing(6)
        outer.addLayout(self.sources_row)
        row = QHBoxLayout()
        row.addStretch(1)
        self.skip = QPushButton("Start empty")
        self.skip.setCursor(Qt.PointingHandCursor)
        self.skip.clicked.connect(self.dismissed)
        row.addWidget(self.skip)
        outer.addLayout(row)
        outer.addStretch(1)
        self._source_buttons: List[QPushButton] = []
        self.restyle()

    def configure(self, catalog: Sequence[Mapping[str, Any]]) -> None:
        """``catalog`` rows: ``{"value", "label", "image": bool, "channels": int, "labels":
        [...], "points": [...], "tracks": [...], "fields": [...], "page": name}``."""
        images = [c for c in catalog if c.get("image")]
        tables = [c for c in catalog if c.get("labels") or c.get("points") or c.get("tracks")]
        fields = [c for c in catalog if c.get("fields")]
        n_img, n_tab = len(images), len(tables)

        def have(text: str) -> str:
            return f"You have: {text}" if text else "You have: nothing upstream yet — load " \
                                                   "an image on the Image Input page"

        img_txt = (f"{n_img} image source{'s' if n_img != 1 else ''}"
                   + (f" ({sum(int(c.get('channels') or 1) for c in images)} channels)"
                      if n_img else ""))
        tab_txt = (f"{n_tab} source{'s' if n_tab != 1 else ''} with tables"
                   + (f" — {', '.join(sorted({str(l) for c in tables for l, _c in c.get('labels', [])}))}"
                      if tables and any(c.get("labels") for c in tables) else ""))
        fld_txt = f"{len(fields)} measured field{'s' if len(fields) != 1 else ''}"
        for key, card in self.cards.items():
            if key in ("overlay", "montage", "movie"):
                card.have.setText(have(img_txt if n_img else ""))
                ok = n_img > 0
            elif key in ("plot", "table"):
                card.have.setText(have(tab_txt if n_tab else ""))
                ok = n_tab > 0
            else:
                parts = [p for p, n in ((img_txt, n_img), (tab_txt, n_tab)) if n]
                if fields:
                    parts.append(fld_txt)
                card.have.setText(have("; ".join(parts)))
                ok = n_img > 0
            note = AVAILABLE.get(key, "")
            card.setEnabled(ok and not note)
            card.setToolTip(note or ("" if ok else "nothing upstream carries what this needs"))
        for b in self._source_buttons:
            b.setParent(None)
            b.deleteLater()
        self._source_buttons = []
        for c in images[:6]:
            b = QPushButton(str(c.get("label") or c.get("value")))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, v=str(c["value"]): self.source_chosen.emit(v))
            self.sources_row.addWidget(b)
            self._source_buttons.append(b)
        self.sources_label.setVisible(bool(images))

    def restyle(self) -> None:
        def h(c) -> str:
            return c.name()
        self.setStyleSheet(f"""
            QWidget#startCard {{ background:{h(T.PANEL)}; border:1px solid {h(T.BORDER)};
                border-radius:10px; }}
            QLabel {{ color:{h(T.INK)}; }}
            QLabel[role="heading"] {{ font-size:16px; font-weight:700; }}
            QLabel[role="muted"] {{ color:{h(T.MUTED)}; }}
            QLabel[role="title"] {{ font-weight:700; font-size:12px; }}
            QLabel[role="have"] {{ color:{h(T.INK_2)}; font-size:10px; }}
            QFrame#goalCard {{ background:{h(T.BODY)}; border:1px solid {h(T.BORDER)};
                border-radius:8px; }}
            QFrame#goalCard:hover {{ border-color:{h(T.ACCENT)}; }}
            QFrame#goalCard:disabled {{ background:{h(T.BG)}; }}
            QPushButton {{ background:{h(T.BODY)}; color:{h(T.INK)}; border:1px solid
                {h(T.BORDER)}; border-radius:6px; padding:4px 10px; }}
            QPushButton:hover {{ border-color:{h(T.ACCENT)}; }}
        """)


__all__ = ["StartCard", "GOALS", "AVAILABLE"]
