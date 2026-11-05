"""Export what a Viewer shows as a standalone interactive HTML view (2026-11-05).

The Viewer's **→ HTML** button lands here: the display state the Viewer holds — which
channels are on, the windows and gammas the user set, the colours picked, the overlay
settings, the scale bar and timestamp, the frame — is read off the panel and handed to
:func:`nodegraph.htmlview.build_view`, so the page opens looking like the Viewer did and
then lets the reader scrub, re-window and toggle for themselves. The dialog asks the four
things that decide the file's size: the pixel budget, the T frames, the Z planes and
whether the overlays travel.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                               QWidget)

from nodelab_v2 import overlays as OV


def viewer_state(v: Any, node_id: Optional[str] = None) -> Dict[str, Any]:
    """The Viewer's display state in :func:`build_view`'s ``state`` form: per channel
    whether it is on, the window where the USER set it (an auto window is left to the
    page's own Auto), the gamma and the picked colour; the overlay settings per tab; the
    coords."""
    node = node_id or getattr(v, "_node_id", None)
    chans: Dict[int, Dict[str, Any]] = {}
    nc = len(getattr(v, "_chan_names", []) or [])
    active = set(getattr(v, "_active_channels", []) or [])
    clim = getattr(v, "_clim", {}) or {}
    clim_user = getattr(v, "_clim_user", set()) or set()
    gammas = getattr(v, "_gammas", {}) or {}
    colors = getattr(v, "_chan_colors", {}) or {}
    for c in range(nc):
        key = v._lut_key(node, c) if hasattr(v, "_lut_key") else (node, c)
        d: Dict[str, Any] = {"on": c in active}
        if key in clim and key in clim_user:
            d["lo"], d["hi"] = float(clim[key][0]), float(clim[key][1])
        if key in gammas:
            d["gamma"] = float(gammas[key])
        if c in colors:
            d["rgb"] = [int(x) for x in colors[c]]
        chans[c] = d
    settings = getattr(v, "overlays", None)
    looks: Dict[str, Dict[str, Any]] = {}
    if isinstance(settings, OV.OverlaySettings):
        for tab in OV.TABS:
            try:
                looks[tab] = dataclasses.asdict(settings.group(tab))
            except Exception:                  # noqa: BLE001
                continue
    try:
        coords = tuple(int(x) for x in v.coords())
    except Exception:                          # noqa: BLE001
        coords = None
    return {"channels": chans, "overlays": looks, "coords": coords,
            "active": sorted(active) if active else None}


class HtmlExportDialog(QDialog):
    """What to embed: the file, the pixel budget, the T frames, the Z planes, the
    overlays and the size cap."""

    def __init__(self, parent: Optional[QWidget], *, n_t: int, n_z: int, cur_t: int,
                 cur_z: int, suggested: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Export an interactive HTML view")
        form = QFormLayout(self)
        self.path = QLineEdit(suggested)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        form.addRow("File", row)
        self.max_px = QComboBox()
        for v in (512, 1024, 2048, 4096):
            self.max_px.addItem(f"{v} px", v)
        self.max_px.setCurrentIndex(1)
        form.addRow("Longest side", self.max_px)
        self.frames = QComboBox()
        self.frames.addItem(f"every frame ({n_t})", "")
        self.frames.addItem(f"this frame only (t {cur_t + 1})", str(cur_t))
        self.frames.addItem("a range…", "range")
        self.frames_text = QLineEdit(f"0-{max(0, n_t - 1)}")
        self.frames_text.setPlaceholderText("0-99 · 0-99:3 · 5,9,12")
        self.frames_text.setEnabled(False)
        self.frames.currentIndexChanged.connect(
            lambda _i: self.frames_text.setEnabled(self.frames.currentData() == "range"))
        if n_t <= 1:
            self.frames.setEnabled(False)
        form.addRow("T frames", self.frames)
        form.addRow("", self.frames_text)
        self.planes = QComboBox()
        self.planes.addItem(f"every plane ({n_z})", "all")
        self.planes.addItem(f"this plane only (z {cur_z + 1})", str(cur_z))
        self.planes.addItem("maximum projection", "max")
        self.planes.addItem("mean projection", "mean")
        if n_z <= 1:
            self.planes.setEnabled(False)
        form.addRow("Z", self.planes)
        self.overlays = QCheckBox("embed the overlays (points, labels, tracks, fields)")
        self.overlays.setChecked(True)
        form.addRow("", self.overlays)
        self.budget = QSpinBox()
        self.budget.setRange(1, 4000)
        self.budget.setValue(250)
        self.budget.setSuffix(" MB")
        form.addRow("Refuse above", self.budget)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def _browse(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Export an interactive HTML view",
                                               self.path.text(), "HTML page (*.html)")
        if path:
            self.path.setText(path)

    def options(self) -> Dict[str, Any]:
        frames = self.frames.currentData()
        if frames == "range":
            frames = self.frames_text.text().strip()
        z = self.planes.currentData()
        if z not in ("all", "max", "mean"):
            z = int(z)
        return {"path": self.path.text().strip(), "max_px": int(self.max_px.currentData()),
                "t_frames": frames or None, "z": z, "overlays": self.overlays.isChecked(),
                "budget_mb": float(self.budget.value())}


def export_viewer_html(v: Any, path: str, *, title: str = "", node_id: Optional[str] = None,
                       max_px: int = 1024, t_frames: Any = None, z: Any = "all",
                       overlays: bool = True, budget_mb: float = 250.0,
                       provenance: Optional[Dict[str, Any]] = None) -> str:
    """Write the page of what Viewer ``v`` shows; returns the path written."""
    from nodegraph.htmlview import build_view, view_html, write_html
    ds = getattr(v, "_dataset", None)
    if ds is None:
        raise ValueError("the Viewer shows nothing yet")
    state = viewer_state(v, node_id)
    m = state["coords"][0] if state.get("coords") else 0
    doc = build_view(ds, title=title or str(node_id or getattr(v, "_node_id", "") or "view"),
                     max_px=max_px, t_frames=t_frames, m_frames=[m] if int(ds.axes.m) > 1 else None,
                     z=z, overlays=overlays, state=state, budget_mb=budget_mb,
                     scalebar=getattr(v, "_scalebar", None) or True,
                     timestamp=getattr(v, "_timestamp", None) or "elapsed",
                     stage_ok=getattr(v, "raw_plane_cb", None) is not None,
                     own_layers=None, provenance=provenance)
    return write_html(path, view_html(doc))


def default_name(node_id: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(node_id or "view"))
    return os.path.join(os.path.expanduser("~"), f"{stem}.html")


__all__ = ["viewer_state", "export_viewer_html", "HtmlExportDialog", "default_name"]
