"""The node catalog's public surface — a facade over :mod:`nodegraph.catalog`.

Importing this module **registers the whole catalog** (that is what
:func:`nodegraph.catalog.load` does, and several callers import this module purely for that
side effect — see ``nodelab_v2/window.py``). The node definitions themselves live one per
module under ``nodegraph/catalog/``, so that a single node can be edited and reloaded into a
running session without re-keying every other node's memoized results
(:mod:`nodegraph.hotreload`).

Everything below is a **re-export**. The names are the compatibility surface this module has
always had: ``COMPUTES``, ``register_node``, ``to_pixels_v2``, ``diffraction_sigmas``,
``gaussian_psf`` and ``OVERLAY_KEY`` are public API (the first five are also re-exported by
``nodegraph/__init__``); the rest are internals that ``nodegraph/selftest.py`` and the
GUI probes pin directly, kept reachable so a verification gate reports on the split instead
of failing to import.

**A caveat for anything that reasons about a compute's origin.** A re-export makes a name
reachable; it does not change where the object was defined. A compute's ``__module__`` is now
its own node module, and this module's AST contains no ``FunctionDef`` at all — so a check
written as ``fn.__module__ == "nodegraph.nodes"``, or one that parses
``inspect.getsource(nodegraph.nodes)``, does not merely break: it silently matches nothing and
reports success over an empty set. Ask :func:`nodegraph.hotreload.is_catalog_op` (registration
provenance) instead, and see ``nodegraph.selftest._catalog_modules`` for the source-level case.

Qt-free.
"""
from __future__ import annotations

import nodegraph.catalog as _catalog

# Registers every node module, in the order `nodegraph.catalog.MODULES` fixes.
#
# Called HERE rather than in the catalog package's own __init__ so that importing
# any one node module (or `_base`, which they all need) cannot trigger the whole
# catalog at an arbitrary moment.
#
# And called BEFORE the re-exports below, which is load-bearing rather than
# stylistic: `from nodegraph.catalog.analysis.histogram_threshold import _snap_bit_depth`
# EXECUTES that module, so with the re-exports first, the handful of nodes this
# facade happens to borrow a helper from would register ahead of everything else.
# Registration order is observable (the link-drag search menu enumerates the
# registry) and `scripts/_catalog_snapshot.py` gates it — it caught exactly this.
# With load() first, every module is already in sys.modules and the re-exports
# are pure attribute lookups.
_catalog.load()

from nodegraph.catalog._base import COMPUTES, register_node  # noqa: E402
from nodegraph.catalog._shared.dim_footprint import _DIM_KAX  # noqa: E402
from nodegraph.catalog._shared.labels import _label_centroids  # noqa: E402
from nodegraph.catalog._shared.map_image import _map_image  # noqa: E402
from nodegraph.catalog._shared.sampling import CHANNEL_STAMP, SAMPLING_KEY  # noqa: E402
from nodegraph.catalog._shared.units import to_pixels_v2  # noqa: E402
from nodegraph.catalog.analysis.histogram_threshold import _snap_bit_depth  # noqa: E402
from nodegraph.catalog.analysis.measure import (  # noqa: E402,F401
    _MEASURE_COLUMNS,
    _MEASURE_SHAPE,
    _measure_stats,
)
from nodegraph.catalog.analysis.object_field import _OBJECT_FIELDS  # noqa: E402
from nodegraph.catalog.analysis.object_metrics import _OBJECT_METRIC_COLUMNS  # noqa: E402
from nodegraph.catalog.analysis.segment import _SEGMENT_2D_ONLY  # noqa: E402
from nodegraph.catalog.enhance.deconvolve import (  # noqa: E402,F401
    _gauss_blur_zero,
    _rl,
    _rl_gaussian,
    diffraction_sigmas,
    gaussian_psf,
)
from nodegraph.catalog.enhance.temporal_gain import (  # noqa: E402,F401
    _exp_fit_t,
    _rolling_mean_t,
    _temporal_gain_field,
    _tile_means,
)
from nodegraph.catalog.view.overlay import OVERLAY_KEY  # noqa: E402

__all__ = [
    "CHANNEL_STAMP", "COMPUTES", "OVERLAY_KEY", "SAMPLING_KEY", "_DIM_KAX", "_MEASURE_COLUMNS",
    "_MEASURE_SHAPE", "_OBJECT_FIELDS", "_OBJECT_METRIC_COLUMNS", "_SEGMENT_2D_ONLY",
    "_exp_fit_t", "_gauss_blur_zero", "_label_centroids", "_map_image", "_measure_stats",
    "_rl", "_rl_gaussian", "_rolling_mean_t", "_snap_bit_depth", "_temporal_gain_field",
    "_tile_means", "diffraction_sigmas", "gaussian_psf", "register_node", "to_pixels_v2"
]
