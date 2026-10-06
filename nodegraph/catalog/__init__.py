"""The node catalog: one module per node type (nodegraph v2, V2.20).

Importing this package registers the whole catalog — every module in :data:`MODULES` is
imported in order, and each registers its own node type(s) into
:data:`~nodegraph.registry.NODES` and :data:`nodegraph.catalog._base.COMPUTES`.

**Why one module per node.** So a node can be edited and reloaded on its own while the GUI
is running (:mod:`nodegraph.hotreload`). The catalog used to be one 12,000-line module, which
meant a live reload could only ever re-key the memo for *every* node at once — retuning one
filter threw away a segmentation you had already paid for. A module per node makes the
dependency set of each node explicit in its own imports, which is what lets the reloader
fingerprint them independently.

**:data:`MODULES` is ORDERED, and the order is load-bearing.**
:class:`~nodegraph.registry.NodeRegistry` is insertion-ordered, and that order is what the
canvas' link-search menu enumerates. The list therefore preserves the registration order the
single-module catalog had, and is not sorted — see ``scripts/_catalog_snapshot.py``, which
gates exactly this.

Qt-free.
"""
from __future__ import annotations

import importlib
import os
from typing import List, Tuple

#: Node modules, in registration order (see the note above — do not sort).
MODULES: Tuple[str, ...] = (
    "channel.select",
    "channel.split",
    "rr.reroute",
    "enhance.deconvolve",
    "enhance.gamma",
    "enhance.gaussian",
    "analysis.threshold",
    "analysis.multiotsu",
    "analysis.threshold_local",
    "analysis.label",
    "analysis.measure",
    "enhance.median",
    "enhance.morphology",
    "enhance.tophat",
    "enhance.dog",
    "enhance.unsharp",
    "enhance.tv_denoise",
    "enhance.wavelet_denoise",
    "enhance.clahe",
    "enhance.normalize",
    "detect.spots",
    "util.zproject",
    "util.crop",
    "util.select_group",
    "util.split_positions",
    "util.select_position",
    "enhance.morphological_gradient",
    "enhance.bilateral",
    "enhance.nlm",
    "analysis.edt",
    "analysis.segment",
    "util.resample",
    "util.stack",
    "util.stitch",
    "align.drift",
    "analysis.extract_boundary",
    "track.link",
    "transform.transfer_domain",
    "transform.transfer_structure",
    "zone.frame",
    "detect.particles",
    "analysis.histogram_threshold",
    "registration.stabilize",
    "analysis.boundary_band",
    "analysis.cluster_points",
    "analysis.tessellate",
    "transform.rasterize_mesh",
    "transform.label_to_points",
    "analysis.voronoi",
    "analysis.roi_mask",
    "analysis.draw_regions",
    "analysis.dvc_field",
    "transform.rasterize_field",
    "analysis.accumulate_field",
    "analysis.dic_correlate",
    "track.objects",
    "enhance.flatten_field",
    "enhance.temporal_gain",
    "enhance.remove_blobs",
    "analysis.object_metrics",
    "analysis.object_field",
    "analysis.reduce_scalar",
    "flow.iterate",
    "flow.advance",
    "zone.boundary",
    "group.interface",
    "registration.align_to",
    "view.overlay",
    "transform.rigid",
    # Appended, not slotted in beside `transform.label_to_points` whose inverse it is: this
    # list's order is what the link-drag search menu enumerates, so inserting mid-list would
    # shift every node after it and re-key nothing usefully. Last is also where
    # `module_order()` would put it on its own.
    "transform.grow_points",
    # `channel.merge` sat here until 2026-09-15 and was RETIRED, not renamed: its op_key is
    # deregistered outright (a saved graph using it silently stops merging on its next pull —
    # see `nodegraph/catalog/util/merge.py`'s module docstring). Its full behaviour, plus
    # T/M/Z concatenation it never had, lives on as `util.merge`'s `merge_axis="C"` branch.
    # Appended for the same reason, not slotted beside `enhance.deconvolve` whose learned
    # counterpart it is: the two share the PSF derivation but nothing else, and moving
    # `enhance.deconvolve` off position 4 would shift every node after it in the link-drag
    # search menu for no gain.
    "enhance.zs_deconvnet",
    # Appended for the same reason, not slotted beside the other `analysis.threshold*` nodes:
    # this list's order is the link-drag search menu's order, so inserting mid-list shifts
    # every node after it and re-keys nothing useful.
    #
    # `analysis.threshold_per_label` sat here until V2.27 and was REMOVED, not renamed: its one
    # capability — derive the level inside each label — is now the `scope` Mode that
    # `analysis.threshold` and `analysis.histogram_threshold` both carry, edited from the card's
    # footprint band. Histogram Threshold absorbed its full product (sub-objects with a
    # `parent_id` join, the per-parent level/n_above/frac_above/n_sub columns, `min_pixels`), so
    # keeping the node as well would have been a third way to do one thing.
    "analysis.filter_labels",
    "analysis.if_else",
    # Appended last, like everything since V2.20 — the list's order is the link-drag search
    # menu's order, so a mid-list insert shifts every node after it for no gain. Last is also
    # where this one BELONGS: it is the write end of the pipeline, and the first catalog node
    # whose product is a file rather than a Dataset.
    "io.write_tiff",
    # Appended last for the same reason, not slotted beside `enhance.tophat` /
    # `enhance.flatten_field` whose background-removal neighbour it is: this list's order is
    # the link-drag search menu's order, so a mid-list insert shifts every node after it and
    # re-keys nothing useful.
    "enhance.subtract_background",
    # Appended last for the same reason, not slotted beside `analysis.dic_correlate` /
    # `analysis.dvc_field` whose correlation sibling it is (V3.00 roadmap W5-P2).
    "analysis.piv",
    # Appended last for the same reason — analysis.piv's dense per-pixel sibling.
    "analysis.optical_flow",
    # Appended last for the same reason: this list's order is the link-drag search menu's
    # order, so a mid-list insert (beside `channel.merge`'s old slot, or beside
    # `util.stack`/`util.stitch`) would shift every node after it for no gain.
    "util.merge",
    # Appended last for the same reason, not slotted beside `track.objects` whose
    # post-processing half it is nor beside `analysis.dvc_field` whose output schema it
    # shares: this list's order is the link-drag search menu's order.
    "analysis.track_field",
    # Appended last for the same reason, not slotted beside `util.crop` whose sibling it
    # plainly is: this list's order is the link-drag search menu's order, and moving every
    # node after `util.crop` one place down to put the two together would cost more than
    # the adjacency is worth.
    "util.crop_to",
    # Appended last for the same reason, not slotted beside `io.write_tiff` whose sibling it
    # is: this list's order is the link-drag search menu's order. The two are the pipeline's
    # write end read two ways — `write_tiff` exports the pixels for measuring, this one
    # exports a picture of them for showing.
    "io.write_movie",
    # The batch axis's three nodes (V3.01), appended in the order you meet them: stack the
    # files, split them again, and the tap the split materializes into. Appended rather than
    # slotted next to `util.merge` — whose Multipoint branch they deliberately are NOT, see
    # `util/batch.py` — because this list's order is the link-drag search menu's order.
    "util.batch",
    "util.unbatch",
    "util.select_batch",
    # Appended last for the same reason, not slotted beside `util.merge` whose one-input
    # counterpart it is: this list's order is the link-drag search menu's order. Merge lays
    # N WIRED Datasets onto an axis; this splits ONE multi-file source card's M axis onto
    # one, which is the same arithmetic reached from the other end.
    "util.timeseries",
    # Appended last, not slotted beside `view.overlay` whose primary it exists to be: this
    # list's order is the link-drag search menu's order. Overlay places a file inside the
    # primary's field; this makes a primary whose field is EVERY file's, so files that share
    # no field can still be overlaid into one view.
    "view.canvas",
    # V4.00 step 7: the plot nodes and the figure writer, appended (this list's order is the
    # link-drag search menu's order, and appending keeps every earlier position).
    "plot.xy",
    "io.write_figure",
    # V4.00 step 8: the other plots, appended for the same reason
    "plot.distribution",
    "plot.heatmap",
    "plot.timeseries",
    # V4.00 step 9: table synthesis, appended for the same reason
    "table.concat",
    "table.join",
    "table.aggregate",
)


def discover() -> Tuple[str, ...]:
    """Every node module present ON DISK, as dotted paths relative to this package.

    :data:`MODULES` fixes the *order* of the nodes that ship; the filesystem decides what
    *exists*. Dropping a new ``.py`` into ``nodegraph/catalog/<category>/`` is therefore
    enough to add a node — no second edit, and nothing to forget. The alternative (a
    hand-maintained list as the only truth) fails in the least helpful way possible: the file
    is right there, the node simply never appears, and nothing says why.

    Skips ``__init__.py``, the ``_shared`` prelude and any ``_``-prefixed module — those are
    support code, not node types, and importing ``_base`` here would be circular."""
    root = os.path.dirname(os.path.abspath(__file__))
    found: List[str] = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d != "__pycache__" and not d.startswith("_")]
        rel = os.path.relpath(dirpath, root)
        prefix = "" if rel == "." else rel.replace(os.sep, ".") + "."
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py" and not f.startswith("_"):
                found.append(prefix + f[:-3])
    return tuple(found)


def module_order() -> Tuple[str, ...]:
    """What :func:`load` imports, in the order it imports it: :data:`MODULES` first (the
    curated registration order), then anything else found on disk, sorted.

    New modules go LAST rather than being slotted in, because their registration position is
    genuinely unknown — order is observable in the link-drag search menu, and inventing a
    position would be a guess presented as a fact. Add the module to :data:`MODULES` to place
    it deliberately."""
    known = [m for m in MODULES if m in set(discover())]
    extra = [m for m in discover() if m not in set(MODULES)]
    return tuple(known + extra)


def load() -> None:
    """Import every node module, in order. Idempotent: a module already in
    ``sys.modules`` is not re-executed, and registration overwrites by ``op_key`` anyway.

    Called explicitly by :mod:`nodegraph.nodes` rather than run at package-import time. That
    is deliberate and load-bearing: importing *any* submodule (say
    ``nodegraph.catalog._base``, which every node module needs) executes this ``__init__``
    first, so registering here would register the whole catalog at whatever moment something
    first reached for a base name — putting the catalog's registration order at the mercy of
    an import statement's position. An explicit call keeps the order in one place."""
    for name in module_order():
        importlib.import_module(f"{__name__}.{name}")


__all__ = ["MODULES", "discover", "module_order", "load"]
