"""dim footprint — shared catalog helpers."""

from __future__ import annotations


from nodegraph.registry import Granularity

#: the per-dim footprint every dim-lever spatial filter shares (2D per-plane, 3D volume).
_DIM_GRAN = {"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME}
_DIM_KAX = {"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})}
#: per-dim footprint for ops with a plane/volume-GLOBAL statistic or solver (min/max
#: normalization, global σ estimate, iterative solver): they stream at the plane/
#: volume unit, never per tile — a tile would see the wrong statistical population
#: (the misdeclared-TILEABLE traps, C1 audit / V2.04 §7).
_DIM_GRAN_GLOBAL = {"2D": Granularity.WHOLE_PLANE, "3D": Granularity.WHOLE_VOLUME}
