"""The **New page…** and **Save as page recipe…** dialogs (V4.00 step 11).

*New page…* asks four things a new page needs settled before it exists: its **kind** and
**name**, how it **starts** — empty (one Page Input, bound), from a **page recipe** (a
prebuilt page graph, built-in or saved by the user), or **linked to a master page** (a live
copy whose values are its own: see MANUAL §2 *Linked pages*) — and which upstream Output its
Page Input **reads**. It returns a :class:`~nodelab_v2.page_recipes.NewPageSpec`; the window
applies it through :func:`~nodelab_v2.page_recipes.apply_new_page`, so the probe can drive
the same code without a modal dialog.

*Save as page recipe…* names and describes the page being saved and says where the file goes.

Qt lives here only; everything the dialogs list comes from the Qt-free
:mod:`nodelab_v2.page_recipes` and :class:`~nodelab_v2.workspace.Workspace`.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QRadioButton, QVBoxLayout, QWidget,
)

from nodegraph import roles as R
from nodelab_v2 import page_recipes as PR
from nodelab_v2 import theme as T
from nodelab_v2.canvas import kind_icon
from nodelab_v2.ops import PAGE_INPUT_OP
from nodelab_v2.workspace import Page, Workspace, kind_label

#: the *Read output* menu's first entry
NO_SOURCE_TEXT = "(none — add a Page Input later)"


def _muted(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "muted")
    lab.setStyleSheet(f"color:{T.MUTED.name()}; background:transparent;")
    f = lab.font()
    f.setPointSize(8)
    lab.setFont(f)
    return lab


class NewPageDialog(QDialog):
    """Settle a new page: kind, name, start (empty / page recipe / linked to a master) and
    the Output its Page Input reads. :meth:`spec` is meaningful after ``exec()`` returned
    :attr:`QDialog.Accepted`."""

    def __init__(self, parent: Optional[QWidget], ws: Workspace, *, kind: str,
                 source: Optional[str] = None, start: str = PR.START_EMPTY,
                 into: Optional[str] = None, master: Optional[str] = None,
                 recipe: Optional[str] = None) -> None:
        super().__init__(parent)
        self._ws = ws
        self._into = into
        self._preset_source = source
        self.setWindowTitle("New page" if into is None else "Start this page")
        self.setMinimumWidth(520)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._kind = QComboBox()
        for k in R.page_kinds():
            self._kind.addItem(kind_icon(k), kind_label(k), k)
            self._kind.setItemData(self._kind.count() - 1,
                                   str(R.page_meta(k).get("description") or ""), Qt.ToolTipRole)
        if kind in R.page_kinds():
            self._kind.setCurrentIndex(R.page_kinds().index(kind))
        self._kind.setToolTip("What the page is for — which nodes its palette offers and "
                              "which earlier pages it may read.")
        form.addRow("Kind", self._kind)

        self._name = QLineEdit()
        self._name.setPlaceholderText(kind_label(kind))
        self._name.setToolTip("Blank: the kind's name, numbered when taken.")
        if into is not None and into in ws.pages:
            self._name.setText(ws.pages[into].name)
        form.addRow("Name", self._name)

        start_box = QWidget()
        sl = QVBoxLayout(start_box)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        self._r_empty = QRadioButton("Empty page")
        self._r_empty.setToolTip("A page with one Page Input reading the Output chosen below.")
        self._r_recipe = QRadioButton("Page recipe")
        self._r_recipe.setToolTip("A prebuilt page graph — built-in, or one you saved with "
                                  "*Save as page recipe…* — loaded onto the page and bound to "
                                  "the Output chosen below. NOT a LabLink recipe.")
        self._r_linked = QRadioButton("Linked to a master page")
        self._r_linked.setToolTip("A live copy of a master page: its graph follows every edit "
                                  "of the master, its parameter values are its own.")
        for r in (self._r_empty, self._r_recipe, self._r_linked):
            sl.addWidget(r)
        form.addRow("Start from", start_box)

        self._recipe = QComboBox()
        self._recipe.setToolTip("Recipes for this kind of page; a saved one shadows a built-in "
                                "of the same name.")
        self._recipe_desc = _muted()
        rbox = QWidget()
        rl = QVBoxLayout(rbox)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(3)
        rl.addWidget(self._recipe)
        rl.addWidget(self._recipe_desc)
        self._recipe_row = form.rowCount()
        form.addRow("Recipe", rbox)

        self._master = QComboBox()
        self._master.setToolTip("★ = set as a master page (the page switcher's *Set as master "
                                "page*); any plain page may be chosen.")
        self._master_row = form.rowCount()
        form.addRow("Master", self._master)

        self._source = QComboBox()
        self._source.setToolTip("The named Output of an earlier page the new page's Page Input "
                                "reads; more Inputs can be added on the page at any time.")
        self._source_row = form.rowCount()
        form.addRow("Read output", self._source)
        self._form = form
        lay.addLayout(form)

        self._note = _muted()
        lay.addWidget(self._note)

        row = QHBoxLayout()
        row.addStretch(1)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText(
            "Add page" if into is None else "Start page")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        row.addWidget(self._buttons)
        lay.addLayout(row)

        self._kind.currentIndexChanged.connect(self._on_kind)
        for r in (self._r_empty, self._r_recipe, self._r_linked):
            r.toggled.connect(self._sync)
        self._recipe.currentIndexChanged.connect(self._sync)
        self._master.currentIndexChanged.connect(self._on_master)

        self._fill_masters(kind, master)
        self._fill_recipes(kind, recipe)
        self._fill_sources(kind, source)
        chosen = {PR.START_RECIPE: self._r_recipe, PR.START_LINKED: self._r_linked}.get(
            start, self._r_empty)
        if chosen is self._r_recipe and self._recipe.count() == 0:
            chosen = self._r_empty
        if chosen is self._r_linked and self._master.count() == 0:
            chosen = self._r_empty
        chosen.setChecked(True)
        self.restyle()
        self._sync()

    # ── lists ────────────────────────────────────────────────────────────────
    def _fill_recipes(self, kind: str, prefer: Optional[str] = None) -> None:
        self._recipe.blockSignals(True)
        self._recipe.clear()
        for r in PR.list_recipes(kind):
            self._recipe.addItem(r.name + ("  (built-in)" if r.builtin else ""), r)
        self._recipe.blockSignals(False)
        if prefer:
            for i in range(self._recipe.count()):
                if self._recipe.itemData(i).name == prefer:
                    self._recipe.setCurrentIndex(i)
                    break

    def _fill_masters(self, kind: str, prefer: Optional[str] = None) -> None:
        self._master.blockSignals(True)
        self._master.clear()
        for p in self._ws.masters(kind):
            if self._into is not None and p.id == self._into:
                continue                      # the page being filled is no master for itself
            text = ("★ " if p.is_master else "") + p.name
            if p.kind != kind:
                text += f"  · {kind_label(p.kind)}"
            self._master.addItem(kind_icon(p.kind), text, p.id)
        self._master.blockSignals(False)
        if prefer:
            i = self._master.findData(prefer)
            if i >= 0:
                self._master.setCurrentIndex(i)

    def _fill_sources(self, kind: str, prefer: Optional[str] = None) -> None:
        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItem(NO_SOURCE_TEXT, "")
        for value, label in self._ws.sources_for_kind(kind):
            self._source.addItem(label, value)
        want = prefer if prefer is not None else self._ws.default_source_for_kind(kind)
        i = self._source.findData(want) if want else -1
        self._source.setCurrentIndex(i if i >= 0 else 0)
        self._source.blockSignals(False)

    # ── behaviour ────────────────────────────────────────────────────────────
    def _current_kind(self) -> str:
        return str(self._kind.currentData() or R.FREE_PAGE)

    def _on_kind(self, *_a) -> None:
        kind = self._current_kind()
        self._name.setPlaceholderText(kind_label(kind))
        self._fill_recipes(kind)
        self._fill_sources(kind, self._preset_source)
        if not self._r_linked.isChecked():
            self._fill_masters(kind)
        self._sync()

    def _on_master(self, *_a) -> None:
        if self._r_linked.isChecked():
            pid = self._master.currentData()
            page = self._ws.pages.get(pid) if pid else None
            if page is not None and page.kind != self._current_kind():
                i = self._kind.findData(page.kind)
                if i >= 0:
                    self._kind.blockSignals(True)
                    self._kind.setCurrentIndex(i)
                    self._kind.blockSignals(False)
                    self._on_kind()
                    return
        self._sync()

    def _set_row_visible(self, row: int, on: bool) -> None:
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
            item = self._form.itemAt(row, role)
            if item is not None and item.widget() is not None:
                item.widget().setVisible(on)

    def _sync(self, *_a) -> None:
        kind = self._current_kind()
        recipe, linked = self._r_recipe.isChecked(), self._r_linked.isChecked()
        self._set_row_visible(self._recipe_row, recipe)
        self._set_row_visible(self._master_row, linked)
        reads = R.op_in_page(PAGE_INPUT_OP, kind)
        self._set_row_visible(self._source_row, reads)
        self._kind.setEnabled(not linked)      # a linked page has its master's kind
        if linked:
            self._on_master_kind()
        r = self._recipe.currentData()
        self._recipe_desc.setText(r.description if (recipe and r is not None) else "")
        self._recipe_desc.setVisible(recipe and r is not None and bool(r.description))
        ok = True
        note = ""
        if recipe and self._recipe.count() == 0:
            ok, note = False, f"No page recipe for a {kind_label(kind)} page yet — save one with the switcher's *Save as page recipe…*."
        elif linked and self._master.count() == 0:
            ok, note = False, "No page to link to yet."
        elif reads and self._source.count() <= 1:
            note = ("No earlier page has a named Output yet — the page starts without a bound "
                    "Page Input; load an image on Image Input first, or add a Page Output there.")
        self._note.setText(note)
        self._note.setVisible(bool(note))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def _on_master_kind(self) -> None:
        pid = self._master.currentData()
        page = self._ws.pages.get(pid) if pid else None
        if page is not None and page.kind != self._current_kind():
            i = self._kind.findData(page.kind)
            if i >= 0:
                self._kind.blockSignals(True)
                self._kind.setCurrentIndex(i)
                self._kind.blockSignals(False)
                self._name.setPlaceholderText(kind_label(page.kind))
                self._fill_sources(page.kind, self._preset_source)
                self._set_row_visible(self._source_row, R.op_in_page(PAGE_INPUT_OP, page.kind))

    # ── result ───────────────────────────────────────────────────────────────
    def spec(self) -> PR.NewPageSpec:
        kind = self._current_kind()
        start = (PR.START_RECIPE if self._r_recipe.isChecked() else
                 PR.START_LINKED if self._r_linked.isChecked() else PR.START_EMPTY)
        source = str(self._source.currentData() or "") if self._source.isVisible() or \
            R.op_in_page(PAGE_INPUT_OP, kind) else ""
        return PR.NewPageSpec(
            kind=kind, name=self._name.text().strip(), start=start,
            recipe=self._recipe.currentData() if start == PR.START_RECIPE else None,
            master=str(self._master.currentData() or "") or None if start == PR.START_LINKED
            else None,
            source=source or None)

    def restyle(self) -> None:
        self.setStyleSheet(T.controls_qss())


class SavePageRecipeDialog(QDialog):
    """Name and describe the page being saved as a page recipe; says where the file goes
    and refuses when the user folder is switched off."""

    def __init__(self, parent: Optional[QWidget], page: Page) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save as page recipe")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self._name = QLineEdit(page.name)
        self._name.setToolTip("The recipe's name in the New page… dialog; a saved recipe with "
                              "the name of a built-in replaces it in the list.")
        form.addRow("Name", self._name)
        self._desc = QLineEdit()
        self._desc.setPlaceholderText("What the page does to the data — one line")
        form.addRow("Description", self._desc)
        form.addRow("Kind", QLabel(kind_label(page.kind)))
        folder = PR.user_dir()
        self._folder = _muted(f"Saved under {folder}" if folder is not None else
                              f"Page recipes are switched off ({PR.ENV_ENABLED}=0).")
        form.addRow("", self._folder)
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addStretch(1)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Save recipe")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(folder is not None)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        row.addWidget(self._buttons)
        lay.addLayout(row)
        self._name.textChanged.connect(
            lambda s: self._buttons.button(QDialogButtonBox.Ok).setEnabled(
                folder is not None and bool(s.strip())))
        self.setStyleSheet(T.controls_qss())

    def name(self) -> str:
        return self._name.text().strip()

    def description(self) -> str:
        return self._desc.text().strip()


__all__ = ["NewPageDialog", "SavePageRecipeDialog", "NO_SOURCE_TEXT"]
