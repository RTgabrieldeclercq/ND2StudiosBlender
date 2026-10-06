"""Publishing a graph as a LabLink recipe — the authoring dialog.

A recipe decides which code a hub executes, so getting one installed is deliberately an act
with a person in the loop: this dialog gets a candidate to the point where it is *verifiably*
correct, writes it where it can be inspected, and hands it to the operator's channel. It
never writes into a hub's ``recipes_dir`` and there is no endpoint that would let it —
installation is a command an operator runs, and that boundary is what makes them willing to
point a hub at their microscope data at all.

What the dialog is actually for is the part a person cannot do reliably by hand: choosing
which of a graph's params a remote caller may turn, and getting each knob's ``kind``,
``unit`` and condition right. Those three are all silent failures —
:mod:`nodelab_v2.lablink.recipe` derives them from the node catalogue so they cannot be
wrong, and this dialog asks only for what genuinely needs a decision: which params to offer,
and what bounds to allow.

**Validate before Submit, and both tiers.** Tier 1 is manifest-versus-graph and tier 2 needs
the node catalogue, which lives here and not on the hub. Running them now is the difference
between finding a problem in this window and finding it at a stranger's session start, after
they have transferred a file.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from nodelab_v2.lablink import presets as PR
from nodelab_v2.lablink import protocol as P
from nodelab_v2.lablink import recipe as RC
from nodelab_v2.lablink.client import HubClient, LabLinkError

#: The channel a candidate recipe is submitted to. An agreed name rather than a protocol
#: one — the exchange has no notion of a recipe, which is exactly why submitting needs no
#: new endpoint.
DEFAULT_INBOX = "recipe-inbox"

_COLUMNS = ("expose", "knob name", "node", "param", "kind", "type", "unit", "min", "max")


class RecipeDialog(QDialog):
    """Choose which of a graph's params become knobs, then validate and submit."""

    def __init__(self, parent: Optional[QWidget], *, draft: RC.RecipeDraft,
                 candidates: Sequence[RC.KnobCandidate],
                 hub: Optional[HubClient] = None,
                 fixed_graph: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Publish as a LabLink recipe")
        self.resize(980, 720)
        self._draft = draft
        self._candidates = list(candidates)
        self._hub = hub
        self._fixed_graph = fixed_graph
        self._written = ""
        self._clean = False

        root = QVBoxLayout(self)
        root.addWidget(self._build_header())

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._build_table())
        split.addWidget(self._build_report())
        split.setSizes([420, 200])
        root.addWidget(split, 1)

        root.addWidget(self._build_buttons())
        self._fill_table()
        self._describe()

    # ── header ──────────────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        box = QGroupBox("Recipe")
        form = QFormLayout(box)
        self._name = QLineEdit(self._draft.name)
        self._name.setToolTip(
            "Also the directory name — tier 1 requires the two to agree, because a caller "
            "asks by name and the hub finds it by directory.")
        self._name.textChanged.connect(self._invalidate)
        self._title = QLineEdit(self._draft.title)
        self._desc = QLineEdit(self._draft.description)
        self._desc.setPlaceholderText("what this pipeline is for, in one sentence")
        self._target = QComboBox()
        nodes = sorted((self._draft.graph_doc.get("graph") or {}).get("nodes") or [],
                       key=lambda n: str(n.get("id")))
        for node in nodes:
            self._target.addItem(f"{node.get('id')}  ({node.get('op_key')})",
                                 str(node.get("id")))
        hit = self._target.findData(self._draft.target)
        if hit >= 0:
            self._target.setCurrentIndex(hit)
        self._target.setToolTip("The node a run pulls — what this recipe is for.")
        self._target.currentIndexChanged.connect(self._invalidate)

        self._workflow = QLineEdit(self._draft.workflow)
        self._workflow.setToolTip(
            "Must be a workflow the hub has configured, or it is refused at install.")
        self._typical = QSpinBox()
        self._typical.setRange(0, 100000)
        self._typical.setValue(self._draft.typical_s)
        self._typical.setSuffix(" s")
        self._worst = QSpinBox()
        self._worst.setRange(0, 100000)
        self._worst.setValue(self._draft.worst_s)
        self._worst.setSuffix(" s")
        runtime = QHBoxLayout()
        runtime.addWidget(QLabel("typical"))
        runtime.addWidget(self._typical)
        runtime.addWidget(QLabel("worst"))
        runtime.addWidget(self._worst)
        runtime.addStretch(1)

        form.addRow("Name", self._name)
        form.addRow("Title", self._title)
        form.addRow("Description", self._desc)
        form.addRow("Workflow", self._workflow)
        form.addRow("Target node", self._target)
        form.addRow("Expected runtime", _wrap(runtime))
        self._derived = QLabel("")
        self._derived.setWordWrap(True)
        self._derived.setStyleSheet("color: palette(mid);")
        form.addRow(self._derived)
        if self._fixed_graph:
            fixed = QLabel(
                "This is a variant: the parent's pipeline is used unchanged, and only the "
                "published defaults differ. Knobs bind to graph node ids, so a variant that "
                "altered the graph would not be a variant of anything.")
            fixed.setWordWrap(True)
            form.addRow(fixed)
        return box

    def _build_table(self) -> QWidget:
        box = QGroupBox("Knobs a caller may turn")
        layout = QVBoxLayout(box)
        note = QLabel(
            "Everything but the bounds is read from the node catalogue, so a knob's unit and "
            "kind cannot disagree with its socket. Bounds are yours: they say what a remote "
            "caller may ask for, which is not the same question as what the maths accepts.")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c.title() for c in _COLUMNS])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        head = self._table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.itemChanged.connect(lambda *_: self._invalidate())
        layout.addWidget(self._table, 1)

        row = QHBoxLayout()
        for text, fn in (("Expose all", lambda: self._set_all(True)),
                         ("Expose none", lambda: self._set_all(False)),
                         ("Expose suggested", self._set_suggested)):
            button = QPushButton(text)
            button.clicked.connect(fn)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_report(self) -> QWidget:
        box = QGroupBox("Verdict")
        layout = QVBoxLayout(box)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText(
            "Validate runs tier 1 (manifest against graph) and tier 2 (against this build's "
            "node catalogue: does the socket exist, does its unit agree, is that mode value "
            "real, can we produce that output kind). The hub cannot run tier 2 — it has no "
            "catalogue — so this is the only place it can happen before submission.")
        layout.addWidget(self._report)
        return box

    def _build_buttons(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        self._validate = QPushButton("Validate")
        self._validate.clicked.connect(self._do_validate)
        self._write = QPushButton("Save to folder…")
        self._write.setToolTip("Write the validated pair somewhere you choose.")
        self._write.clicked.connect(self._do_write)
        self._write.setEnabled(False)
        self._submit = QPushButton("Submit to hub…")
        self._submit.setToolTip(
            "Upload the pair to the operator's review channel. Installing it is a command "
            "they run — this never writes into a hub's recipes_dir.")
        self._submit.clicked.connect(self._do_submit)
        self._submit.setEnabled(False)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        row.addWidget(self._validate)
        row.addWidget(self._write)
        row.addWidget(self._submit)
        row.addStretch(1)
        row.addWidget(close)
        return bar

    # ── the table ───────────────────────────────────────────────────────────────
    def _fill_table(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._candidates))
        chosen = {k.candidate.key: k for k in self._draft.knobs}
        # Named as a SET: two nodes with a 2D/3D lever both want to be `dim`, and tier 1
        # refuses two knobs sharing a name.
        suggested = RC.assign_names(self._candidates)
        for row, cand in enumerate(self._candidates):
            picked = chosen.get(cand.key)
            name = picked.name if picked else suggested[cand.key]
            lo, hi = ((picked.minimum, picked.maximum) if picked
                      else RC.suggest_bounds(cand))

            expose = QTableWidgetItem()
            expose.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            expose.setCheckState(Qt.CheckState.Checked if picked
                                 else Qt.CheckState.Unchecked)
            tip = "\n".join(cand.notes) if cand.notes else (cand.help or "")
            expose.setToolTip(tip)
            self._table.setItem(row, 0, expose)
            self._table.setItem(row, 1, _item(name, editable=True, tip=cand.help))
            self._table.setItem(row, 2, _item(f"{cand.node_id}", tip=cand.op_key))
            self._table.setItem(row, 3, _item(cand.param, tip=cand.help))
            self._table.setItem(row, 4, _item(cand.kind, tip=(
                "'param' writes the node's params and 'mode' writes its modes. They are "
                "read from different places, so the wrong one is a silent no-op — which is "
                "why this is derived from the catalogue and not typed.")))
            self._table.setItem(row, 5, _item(cand.type))
            self._table.setItem(row, 6, _item(cand.unit or "—", tip=(
                "Copied from the socket's own declared unit. A knob claiming a different "
                "one is refused by tier 2, because the number would be accepted and would "
                "mean something else entirely.")))
            numeric = cand.type in ("float", "int")
            self._table.setItem(row, 7, _item("" if lo is None else f"{lo:g}",
                                              editable=numeric))
            self._table.setItem(row, 8, _item("" if hi is None else f"{hi:g}",
                                              editable=numeric))
            if cand.notes and any("NEVER READ" in n for n in cand.notes):
                for column in range(len(_COLUMNS)):
                    item = self._table.item(row, column)
                    if item is not None:
                        item.setForeground(Qt.GlobalColor.red)
        self._table.blockSignals(False)

    def _set_all(self, on: bool) -> None:
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            self._table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._table.blockSignals(False)
        self._invalidate()

    def _set_suggested(self) -> None:
        self._table.blockSignals(True)
        for row, cand in enumerate(self._candidates):
            want = cand.suggested or cand.kind == "mode"
            self._table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if want else Qt.CheckState.Unchecked)
        self._table.blockSignals(False)
        self._invalidate()

    # ── assembling ──────────────────────────────────────────────────────────────
    def _collect(self) -> Tuple[Optional[RC.RecipeDraft], List[str]]:
        """``(draft, problems)`` from the form as it stands."""
        problems: List[str] = []
        name = RC.slugify(self._name.text().strip())
        if not name:
            problems.append("the recipe needs a name")
        knobs: List[RC.KnobDraft] = []
        seen: Dict[str, int] = {}
        for row, cand in enumerate(self._candidates):
            if self._table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue
            knob_name = str(self._table.item(row, 1).text()).strip()
            if not knob_name:
                problems.append(f"{cand.node_id}.{cand.param}: the knob needs a name")
                continue
            seen[knob_name] = seen.get(knob_name, 0) + 1
            lo = _number(self._table.item(row, 7).text())
            hi = _number(self._table.item(row, 8).text())
            if cand.type in ("float", "int") and (lo is None or hi is None):
                # Refused rather than defaulted: an unbounded numeric knob published to a
                # remote caller is a decision, and it is not this dialog's to make quietly.
                problems.append(
                    f"{knob_name}: a numeric knob needs both a min and a max. They are what "
                    f"turn a bad value into an instant refusal instead of a wasted run.")
            if lo is not None and hi is not None and lo > hi:
                problems.append(f"{knob_name}: min {lo:g} is above max {hi:g}")
            knobs.append(RC.KnobDraft(candidate=cand, name=knob_name,
                                      minimum=lo, maximum=hi,
                                      derive=cand.can_derive))
        for knob_name, count in seen.items():
            if count > 1:
                problems.append(f"knob name {knob_name!r} is used {count} times")

        draft = RC.RecipeDraft(
            name=name, workflow=self._workflow.text().strip() or P.WORKER_NAME,
            title=self._title.text().strip(), description=self._desc.text().strip(),
            target=str(self._target.currentData() or ""), knobs=knobs,
            allow_zones=self._draft.allow_zones,
            requires_metadata=self._draft.requires_metadata,
            capabilities=self._draft.capabilities,
            typical_s=self._typical.value(), worst_s=self._worst.value(),
            graph_doc=self._draft.graph_doc, loaders=self._draft.loaders)
        return draft, problems

    def _describe(self) -> None:
        needs = ", ".join(self._draft.requires_metadata) or "nothing"
        text = (f"Derived from the graph: {len(self._draft.loaders)} loader(s), "
                f"requires metadata: {needs}. A recipe that derives from a field the file "
                f"does not supply is refused rather than run with defaults, so this list is "
                f"measured from the catalogue rather than typed.")
        if self._draft.conditional_metadata:
            # Said rather than declared: a static requirement here would refuse every honest
            # 2D job on single-plane data, which legitimately has no z spacing at all.
            text += (f"\n\nA caller who turns this recipe to 3D would additionally need "
                     f"{', '.join(self._draft.conditional_metadata)}. The manifest format "
                     f"has no conditional requirement, so that cannot be declared — if 3D "
                     f"runs matter here, say so in the description.")
        self._derived.setText(text)

    def _invalidate(self) -> None:
        self._clean = False
        self._submit.setEnabled(False)
        self._write.setEnabled(False)

    # ── validate ────────────────────────────────────────────────────────────────
    def _do_validate(self) -> None:
        draft, problems = self._collect()
        lines: List[str] = []
        if problems:
            lines.append(f"FAIL  {len(problems)} problem(s) in this form:")
            lines += [f"        - {p}" for p in problems]
            self._report.setPlainText("\n".join(lines))
            self._invalidate()
            return

        staging = tempfile.mkdtemp(prefix="nd2s-recipe-")
        try:
            directory = RC.write_recipe(draft, staging)
            ok, found = RC.check_recipe(directory)
        except Exception as exc:                         # noqa: BLE001
            self._report.setPlainText(f"the check itself failed: {type(exc).__name__}: {exc}")
            self._invalidate()
            return
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        if ok:
            lines.append(f"ok    tier 1 + tier 2: recipe {draft.name!r}, "
                         f"{len(draft.knobs)} knob(s), "
                         f"{len(draft.outputs())} output(s)")
            lines.append("      every knob resolves against this build's node catalogue.")
            lines.append("This recipe is ready to submit.")
            for cand in self._candidates:
                for note in cand.notes:
                    if "NEVER READ" in note or "cannot grey" in note:
                        lines.append(f"note  {cand.node_id}.{cand.param}: {note}")
            self._clean = True
            self._submit.setEnabled(self._hub is not None)
            self._write.setEnabled(True)
        else:
            lines.append(f"FAIL  {len(found)} problem(s) against this node catalogue:")
            lines += [f"        - {p}" for p in found]
            self._invalidate()
        self._report.setPlainText("\n".join(lines))

    # ── write / submit ──────────────────────────────────────────────────────────
    def _do_write(self) -> None:
        draft, problems = self._collect()
        if problems or not self._clean:
            QMessageBox.warning(self, "Not validated",
                                "Validate the recipe first — the button enables when it is "
                                "clean.")
            return
        target = QFileDialog.getExistingDirectory(
            self, "Write the recipe directory into…", str(PR.OUTBOX_DIR))
        if not target:
            return
        try:
            directory = RC.write_recipe(draft, target)
        except OSError as exc:
            QMessageBox.warning(self, "Could not write", str(exc))
            return
        self._written = directory
        self._report.appendPlainText(f"\nwrote {directory}")

    def _do_submit(self) -> None:
        draft, problems = self._collect()
        if problems or not self._clean or self._hub is None:
            return
        channel, ok = QInputDialog.getText(
            self, "Submit to the hub",
            "The channel the operator collects candidate recipes from:",
            text=DEFAULT_INBOX)
        if not ok or not channel.strip():
            return
        channel = channel.strip()

        # Into our own outbox FIRST, so there is always a local copy of exactly what was
        # sent — the hashes below are meaningless without something to compare them to.
        try:
            PR.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
            directory = RC.write_recipe(draft, str(PR.OUTBOX_DIR))
        except OSError as exc:
            QMessageBox.warning(self, "Could not stage the recipe", str(exc))
            return

        sent = []
        try:
            for filename in (RC.MANIFEST_FILENAME, RC.GRAPH_FILENAME):
                ref = self._hub.put_file(channel, os.path.join(directory, filename))
                sent.append(ref)
        except LabLinkError as exc:
            QMessageBox.warning(
                self, "Submit failed",
                f"{exc}\n\nThe recipe is still staged at {directory}.")
            return

        self._written = directory
        lines = [f"\nsubmitted to channel {channel!r}:"]
        for ref in sent:
            lines.append(f"        {ref.name}  sha256 {ref.sha256[:16]}…  {ref.size} bytes")
        lines.append(f"      staged locally at {directory}")
        lines.append("")
        lines.append("Installing it is an operator's command, on the hub:")
        lines.append(f"  python -m lablink hub recipe install --config hub.json "
                     f"--root <store> --from-channel {channel}")
        lines.append("It stages the candidate, runs tier 1, prints the knob table and who "
                     "submitted it, and asks for confirmation. It does NOT run tier 2 — "
                     "that happened here.")
        self._report.appendPlainText("\n".join(lines))
        self._submit.setEnabled(False)

    @property
    def written(self) -> str:
        return self._written


# ── entry points ────────────────────────────────────────────────────────────────

def publish_document(parent: Optional[QWidget], doc: Any, *,
                     hub: Optional[HubClient] = None,
                     suggested_name: str = "", workspace: Any = None,
                     page_id: str = "") -> Optional[str]:
    """Publish the editor's live graph as a recipe. Returns the directory written, if any.

    The graph half is serialized through the same ``nodegraph.serialize`` the editor writes,
    so there is no second format and nothing to hand-author. A page that reads other pages
    (it holds a Page Input) is published WITH them, flattened into one graph (V4.00), because
    the page alone would arrive with its inputs unbound.
    """
    if not getattr(doc, "nodes", None):
        QMessageBox.information(parent, "Publish as a recipe",
                                "There is nothing on the canvas to publish yet.")
        return None
    name = RC.slugify(suggested_name or _name_from(doc) or "my-recipe")
    try:
        if workspace is not None and page_id and RC.page_reads_other_pages(doc):
            draft = RC.draft_from_workspace(workspace, page_id, name=name)
        else:
            draft = RC.draft_from_document(doc, name=name)
    except Exception as exc:                             # noqa: BLE001
        QMessageBox.warning(parent, "Cannot publish this graph",
                            f"{type(exc).__name__}: {exc}")
        return None
    if not draft.target:
        QMessageBox.warning(parent, "Cannot publish this graph",
                            "This graph has no node a run could pull.")
        return None

    from nodegraph.serialize import from_dict
    graph, _z, _g = from_dict(draft.graph_doc)
    dialog = RecipeDialog(parent, draft=draft, candidates=RC.candidates(graph), hub=hub)
    dialog.exec()
    return dialog.written or None


def promote_preset(parent: Optional[QWidget], *, hub: Optional[HubClient],
                   recipe_meta: Dict[str, Any], knobs: Dict[str, Any],
                   say: Callable[[str], None] = print) -> Optional[str]:
    """Turn tuned knob values into a **derived recipe** — the parent's pipeline, new defaults.

    This needs the parent recipe's graph on this machine, and there is no way to fetch one:
    ``GET /workflows`` publishes a recipe's knobs but never its graph, and the manifest format
    has no inheritance. So the parent is resolved from what this machine has — our own outbox
    first, then the repo's starter kit — and if neither holds it, that is said plainly rather
    than half-done.
    """
    name = str(recipe_meta.get("name") or "")
    found = _find_parent(name)
    if found is None:
        QMessageBox.information(
            parent, "Cannot promote this one",
            f"Promoting needs {name!r}'s own graph on this machine, and a hub publishes a "
            f"recipe's knobs but never its graph — so there is nothing to derive from "
            f"here.\n\nAsk the operator for the recipe directory, put it in "
            f"{PR.OUTBOX_DIR}, and this will work.\n\nYour preset still works as it is; it "
            f"just cannot be shared as a recipe.")
        return None

    parent_dir, manifest, graph_doc = found
    variant, ok = QInputDialog.getText(
        parent, "Promote to a recipe",
        "A name for the variant (it becomes the recipe's directory name):",
        text=f"{name}-variant")
    if not ok or not variant.strip():
        return None
    variant = RC.slugify(variant.strip())

    derived = RC.derived_from(manifest, graph_doc, name=variant, knobs=knobs,
                             description=(
                                 f"Derived from {name} on "
                                 f"{time.strftime('%Y-%m-%d')} with these knob values as "
                                 f"its defaults."))
    staging = tempfile.mkdtemp(prefix="nd2s-variant-")
    try:
        directory = os.path.join(staging, variant)
        os.makedirs(directory, exist_ok=True)
        RC._write_json(os.path.join(directory, RC.MANIFEST_FILENAME), derived)
        RC._write_json(os.path.join(directory, RC.GRAPH_FILENAME), graph_doc)
        good, problems = RC.check_recipe(directory)
        if not good:
            say(f"the derived recipe does not validate ({len(problems)} problem(s)):")
            for problem in problems:
                say(f"    - {problem}")
            QMessageBox.warning(
                parent, "The variant does not validate",
                "\n".join(problems[:6])
                + ("\n…" if len(problems) > 6 else ""))
            return None
        PR.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        final = os.path.join(str(PR.OUTBOX_DIR), variant)
        if os.path.isdir(final):
            shutil.rmtree(final)
        shutil.copytree(directory, final)
    except OSError as exc:
        QMessageBox.warning(parent, "Could not write the variant", str(exc))
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    say(f"wrote the derived recipe {variant!r} to {final}")
    say(f"  its graph is {name}'s, unchanged; {len(knobs)} knob default(s) differ.")
    if hub is None:
        say("  connect to a hub to submit it for installation.")
        return final

    channel, ok = QInputDialog.getText(
        parent, "Submit the variant",
        "The channel the operator collects candidate recipes from:", text=DEFAULT_INBOX)
    if not ok or not channel.strip():
        return final
    try:
        for filename in (RC.MANIFEST_FILENAME, RC.GRAPH_FILENAME):
            ref = hub.put_file(channel.strip(), os.path.join(final, filename))
            say(f"  sent {ref.name} (sha256 {ref.sha256[:16]}…)")
    except LabLinkError as exc:
        say(f"  submit failed: {exc}")
        say(f"  the variant is still staged at {final}")
        return final
    say("On the hub, the operator runs:")
    say(f"  python -m lablink hub recipe install --config hub.json --root <store> "
        f"--from-channel {channel.strip()}")
    return final


def _find_parent(name: str) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """``(dir, manifest, graph)`` for a recipe this machine holds, or ``None``."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in (str(PR.OUTBOX_DIR),
                 os.path.join(repo, "lablink_recipes", P.WORKER_NAME)):
        directory = os.path.join(base, name)
        manifest_path = os.path.join(directory, RC.MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            graph_rel = str(manifest.get("graph") or RC.GRAPH_FILENAME)
            with open(os.path.join(directory, graph_rel), encoding="utf-8") as fh:
                graph_doc = json.load(fh)
        except (OSError, ValueError):
            continue
        return directory, manifest, graph_doc
    return None


def _name_from(doc: Any) -> str:
    path = str(getattr(doc, "path", "") or "")
    if not path:
        return ""
    stem = os.path.basename(path)
    for suffix in (".nd2graph.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _item(text: str, *, editable: bool = False, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if editable:
        flags |= Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    if tip:
        item.setToolTip(tip)
    return item


def _number(text: str) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _wrap(layout: QHBoxLayout) -> QWidget:
    holder = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    holder.setLayout(layout)
    return holder


__all__ = ["RecipeDialog", "publish_document", "promote_preset", "DEFAULT_INBOX"]
