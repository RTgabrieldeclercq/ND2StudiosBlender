"""objects — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Tuple

from nodegraph.dataset import Dataset
from nodegraph.domains import Domain
from nodegraph.engine import EvalContext
from nodegraph.registry import InFloat, SocketSpec

def _object_velocity(track: np.ndarray, tt: np.ndarray, y_um: np.ndarray,
                     x_um: np.ndarray, dt_s: float, *,
                     z_um: np.ndarray | None = None):
    """Per-object velocity ``(vz, vy, vx)`` in µm/s from consecutive detections of the
    same track. Returns three ``(N,)`` arrays aligned with the input rows.

    ``z_um`` is optional: pass it for a volumetric member table to get a real axial
    component, omit it and ``vz`` comes back all-NaN. NaN rather than zero for the same
    reason a track's first row is NaN — a fabricated 0 reads as "measured, not moving"
    and drags every mean toward zero, which is exactly the failure a 2D-only caller
    would never notice. Axial velocity is only as good as the z sampling: on a stack
    whose z step exceeds the object size, consecutive planes do not see the same object
    and ``vz`` is noise, not motion.

    Each object is differenced against its own track's PREVIOUS detection and divided by
    the real frame gap, so a gap-filled track — the ``fingerprint`` and ``overlap`` linkers
    both bridge missed frames — is not reported as a sudden jump. That is
    ``scripts/mean_velocity.py``'s definition; ``backend/fields.py`` instead differences
    only strictly adjacent frames and silently drops every gapped step, so it disagrees
    with the app's own velocity script. The gap-aware form is used for both nodes here so
    ``analysis.object_metrics`` and ``analysis.object_field`` cannot report different
    speeds for the same data.

    A track's FIRST detection has no predecessor, so its velocity is **NaN** — not 0,
    which would read as a stationary object and drag every mean toward zero. Untracked
    rows (``track_id <= 0``, the write-back's background value) are NaN throughout."""
    n = len(track)
    vz = np.full(n, np.nan, dtype=float)
    vy = np.full(n, np.nan, dtype=float)
    vx = np.full(n, np.nan, dtype=float)
    live = np.flatnonzero(np.asarray(track) > 0)
    if live.size < 2:
        return vz, vy, vx
    order = live[np.lexsort((tt[live], track[live]))]       # by track, then time
    tid = np.asarray(track)[order]
    edge = np.concatenate(([True], tid[1:] != tid[:-1]))    # first row of each track
    dt_frames = np.diff(tt[order]).astype(float)
    step = np.concatenate(([np.nan], dt_frames))            # gap to the previous row
    step = np.where(edge, np.nan, step)                     # a track's first row has none
    span = step * float(dt_s)
    dy = np.concatenate(([np.nan], np.diff(y_um[order])))
    dx = np.concatenate(([np.nan], np.diff(x_um[order])))
    good = np.isfinite(span) & (span > 0)
    vy[order[good]] = dy[good] / span[good]
    vx[order[good]] = dx[good] / span[good]
    if z_um is not None:
        dz = np.concatenate(([np.nan], np.diff(np.asarray(z_um, dtype=float)[order])))
        vz[order[good]] = dz[good] / span[good]
    return vz, vy, vx
def _InFrameInterval() -> Tuple[SocketSpec, ...]:
    """The shared ``frame_interval`` escape hatch for the two per-object motion nodes.

    ``0`` means "use the file's own ``dt_s``", which is what a real acquisition supplies;
    a positive value overrides it for data whose timing the reader could not recover (a
    plain TIFF stack carries no timestamps at all). It is NOT ``unit="s"``: that unit means
    *convert seconds to frames by dividing by* ``dt_s``, and this param **is** ``dt_s`` —
    tagging it would divide the interval by itself."""
    return (InFloat("frame_interval", "Frame interval (s)", unit="", field=False,
                    default=0.0, derive="dt_s or 0.0",
                    description=
                    "Seconds between consecutive timepoints, used to turn a displacement "
                    "into a RATE. 0 means take it from the file's own metadata, which is "
                    "what you want for an ND2 — auto shows the value that will be used. Set "
                    "it only when the file carries no timing (a plain TIFF stack) or you "
                    "know the recorded interval is wrong. It scales every velocity, speed, "
                    "divergence and curl LINEARLY, so getting it wrong scales your reported "
                    "µm/s by exactly the same factor; it is inert if you asked for none of "
                    "those metrics."),)
def _frame_interval_s(ctx: EvalContext, *, needed: bool, wanted, node: str) -> float:
    """The seconds-per-frame every rate in an object-analysis node divides by.

    Resolution order — the ``frame_interval`` socket if the user set one, else the file's
    ``dt_s`` (memo-fenced through :meth:`EvalContext.calib`), else **refuse**.

    Refusing is the point. This used to be ``(ctx.calib("dt_s") or 1.0)``, and because
    ``dt_s`` was absent from every ND2 (:func:`nodelab_v2.ingest._dt_from_timestamps` now
    supplies it) that fallback silently turned µm/frame into a column labelled µm/s — a
    factor of 1424.6 on the lab's 6-hour WellA3 timelapse, i.e. cells reported at 9.9 mm/h.
    A wrong number wearing a unit is worse than no number, and this node already refuses
    its other missing prerequisites (a ``track_id`` column, an intensity column) rather
    than degrading, so this is the same rule applied to the one input that was exempt.

    ``needed`` is False when no requested metric is a rate; the return value is then unused
    and ``dt_s`` is deliberately NOT read, so the memo does not fence the pull on a key the
    node could not have consumed."""
    if not needed:
        return 1.0
    override = float(ctx.params.get("frame_interval", 0.0) or 0.0)
    if override > 0:
        return override
    dt = ctx.calib("dt_s")
    if dt and float(dt) > 0:
        return float(dt)
    raise ValueError(
        f"{node}: {sorted(wanted)} are RATES (per second), and this Dataset declares no "
        f"frame interval — dividing by a placeholder 1.0 would report µm/FRAME under a "
        f"µm/s label. Set the `frame_interval` socket to the seconds between timepoints, "
        f"or re-open the source if it should carry timing (an ND2's per-frame timestamps "
        f"become `dt_s` at ingest; a plain TIFF has none). Metrics that are not rates "
        f"('neighbors', 'density', 'mean_area', 'intensity', 'frame_fold') need no "
        f"interval and work as they are.")
def _object_table(ctx: EvalContext, ds: Dataset, *, node: str,
                  allow_3d: bool = False):
    """Pull and validate the Label/Point member table an object-analysis node reads.

    Returns ``(domain, layer, columns, z_kind)`` with ``columns`` holding at least the
    invariant ``id,m,t,c,z,y,x``. Shared by ``analysis.object_metrics`` and
    ``analysis.object_field`` so their refusals are identical, and shaped after
    ``track.objects``' own validation.

    ``allow_3d`` selects which dimensionality rule applies, and the default is the
    restrictive one so a new caller cannot silently opt itself in:

    * ``False`` (``analysis.object_field``) — a ``subpixel`` table is refused outright.
      That node's whole output is a 2-D grid of in-plane estimators: a 2-D curl is a
      scalar where a 3-D one is a vector, and the velocity gradient is fitted in-plane.
      There is no correct 3-D answer to give, so it refuses rather than measuring in a
      geometry the maths does not describe.
    * ``True`` (``analysis.object_metrics``) — a volumetric table is accepted here and
      the **caller** refuses only the individual metrics that are genuinely planar,
      which is the ``analysis.measure`` precedent (it rejects just its 2D-only shape
      entries rather than the whole volume). Blanket-refusing this node meant a 3-D
      dataset had *no* per-object motion column at all, not even a centroid speed that
      is perfectly well defined in a volume.
    """
    target = ctx.params.get("__modes__", {}).get("target", "label")
    domain = Domain.LABEL if target == "label" else Domain.POINT
    layer = ctx.layer("labels") if target == "label" else ctx.layer("points")
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(domain)
            if a.layer == layer}
    if "id" not in cols:
        raise ValueError(
            f"{node}: no {domain.value} structure {layer!r} on the input Dataset. Run "
            f"analysis.segment / analysis.label (Label members) or detect.spots "
            f"(Point members) upstream, or point the layer socket at the right layer.")
    missing = sorted(k for k in ("m", "t", "c", "z", "y", "x") if k not in cols)
    if missing:
        raise ValueError(
            f"{node}: the {domain.value} layer {layer!r} is missing the invariant "
            f"column(s) {missing}, so its objects have no position to measure.")
    n = len(cols["id"])
    ragged = sorted(k for k, v in cols.items() if len(v) != n)
    if ragged:
        raise ValueError(
            f"{node}: column(s) {ragged} on layer {layer!r} disagree in length with 'id' "
            f"({n}) — every metric would be built from misaligned rows.")
    zk = ds.structure_zkind(domain, layer) or "plane_index"
    if zk == "subpixel" and not allow_3d:
        raise ValueError(
            f"{node} is 2D-only: it ports Cell-Tracker's in-plane neighbourhood "
            f"estimators, and neither a 2-D curl nor a 2-D velocity gradient is defined on "
            f"a volume. The {domain.value} layer {layer!r} is 3D (z_kind='subpixel'). "
            f"Segment or detect with the 2D/3D lever on 2D so each plane's objects are "
            f"their own, or z-project first. (track.objects refuses a 3D table for the "
            f"same reason.)")
    return domain, layer, cols, zk
