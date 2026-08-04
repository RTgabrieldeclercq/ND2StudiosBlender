"""Overlay compositing for the Viewer — a secondary Dataset's pixels, in the primary's grid.

``view.overlay`` records WHERE the secondary goes; this module is what actually fetches it.
Given the recipe entry the node stamped, the primary's display plane, and the secondary's
provider, it returns one array shaped like the primary's plane, carrying the secondary's
pixels resampled onto the primary's field.

**This is display work, not data work.** It runs in the runner's plane path, beside the
existing decimation to ``MAX_DISPLAY_DIM`` — the same category of transformation, for the
same reason, with the same guarantee: nothing it produces reaches a downstream node. The
overlay node's own output Dataset is untouched, which is what keeps "look at it" and
"measure it" the same pipeline.

Three properties worth stating, because each was a choice:

* **Nearest-neighbour, and separable.** The row and column maps are independent, so the
  whole resample is two 1-D index arrays and one fancy-index — no interpolation pass over a
  2-D grid. At the WellA3 pair's 5.985× magnification a smooth interpolation would invent
  detail the GFP file does not have; blocky is the honest rendering of 171 real pixels
  stretched over 1024, and it also makes the sampling grid visible, which is a feature when
  you are checking an alignment.
* **Several tiles, painted in coverage order.** A primary field can straddle up to four
  secondary tiles (WellA3 ``m10`` does). Each is mapped independently and written where it
  lands; uncovered pixels keep the fill value, so a genuine gap reads as a gap.
* **Handedness applied HERE.** ``flip_x``/``flip_y`` describe how the camera is mounted and
  no file records them, so they never entered the envelope (see
  :data:`nodegraph.dataset.CALIBRATION_KEYS`). This is the sampling step, so this is where
  they belong — matching :func:`nodegraph.nodes._stitch_offsets_stage`'s convention exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.placement import (  # noqa: F401 — re-exported
    axis_map, compose_secondary_plane, field_box, paired_t,
    secondary_z_index)

#: Every name here is re-exported from `nodegraph.placement`. The compositing moved down to
#: the engine layer when `view.overlay` gained its `resample` output mode: that mode bakes
#: the SAME resample into a real Dataset channel, and a node cannot import the GUI package.
#: This module is now the display path's alias for it, so the picture you look at and the
#: pixels you measure are produced by one function rather than two that agree today.
__all__ = ["axis_map", "compose_secondary_plane", "secondary_z_index", "paired_t"]


