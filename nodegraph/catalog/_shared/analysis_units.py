"""analysis_units — run an expensive analysis unit by unit: across cores, saved as it goes,
resumed from disk (2026-10-09).

The DIC, DVC, PIV and SerialTrack nodes share one shape: the work splits into independent
**units** — a stage position ``m`` for volumetric methods, a position × plane ``(m, z)`` for
planar ones — and every unit runs the WHOLE time series, because time is the one axis these
methods compare across and so can never be split. This module is that shape, once:

* :func:`unit_list` — the units, ``(m, z)`` or ``(m,)``.
* :func:`open_node_trial` — the node's saved trial beside its source file
  (:mod:`nodegraph.analysis_store`), honouring the node's ``results`` socket
  (``reuse`` / ``recompute`` / ``off``) and refusing to save a scoped (quick-analysis) pull.
* :func:`run_units` — loads the units the trial already holds, then computes the rest: in
  this thread when one lane is all the machine affords (keeping the kernel's own in-unit
  progress), otherwise across worker processes through
  :func:`nodegraph.parallel.adaptive_fold`, whose width the resource governor decides unit by
  unit. Each worker saves its own unit the moment it finishes. Results come back **in unit
  order**, so ids assigned afterwards are identical to a serial run.
* :class:`ShippedArray` — a big unit input (a DVC volume series) handed to a worker as a
  memory-mapped temp file instead of a multi-GB pickle.

A unit's result is a dict ``{"tables": {kind: {column: array}}, "stats": {...}}`` — the
tables are what the store writes (one Parquet file per kind) and what the node folds into
its output; ``stats`` carries per-unit counts the node aggregates into its stamps.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.engine import EvalContext
from nodegraph.registry import InString

#: Modules a pyALDVC worker imports one at a time before its first unit: ``al_dvc``'s
#: import-time cache probe writes and deletes a file in its own ``__pycache__``, and workers
#: starting together collide on it — the losers turn numba's disk cache off and recompile
#: every kernel (:func:`nodegraph.governor.serialized_import`).
ALDVC_PREIMPORT: Tuple[str, ...] = ("al_dvc._numba_compat",)

#: The two store-control sockets every saving node carries. Excluded from the trial key
#: (``ctx.lineage(exclude=STORE_PARAMS)``) so changing them never moves a trial.
STORE_PARAMS: Tuple[str, ...] = ("results", "trial_label")

_RESULTS_DOCS = {
    "reuse": ("Load every unit (position, or position × plane) already saved for exactly "
              "these settings and this input, compute only the missing ones, and save each "
              "new unit the moment it finishes. A cancelled or crashed run resumes where it "
              "stopped; changing a setting starts (or finds) a different trial."),
    "recompute": ("Ignore what is saved for these settings and compute every unit again, "
                  "overwriting the trial's files as each unit finishes — use it after fixing "
                  "something upstream whose code (not settings) changed, which the trial key "
                  "cannot see."),
    "off": ("Neither read nor write saved results: compute in memory only, as before the "
            "store existed. Use it for throwaway runs you do not want written beside the "
            "data."),
}


def store_sockets(available_in: Optional[Dict[str, Any]] = None) -> list:
    """The ``results`` and ``trial_label`` sockets, declared once for every saving node.
    ``available_in`` gates them to the mode values that save (SerialTrack, on a tracker
    whose other methods are fast enough not to)."""
    return [
        InString("results", "Saved results", field=False, default="reuse",
                 available_in=available_in,
                 choices=("reuse", "recompute", "off"), choice_docs=_RESULTS_DOCS,
                 description=(
                     "Whether this analysis saves its result beside the source file "
                     "(`<file>.analysis/`, one Parquet file per finished position) and reuses "
                     "it. A unit that folder refuses (a read-only share, a full or failing "
                     "disk) is saved in a spare folder under `~/.nd2studios/analysis/` "
                     "instead, and found there next time. "
                     "`reuse` loads what is already saved for these exact settings and "
                     "computes only the rest; `recompute` redoes everything; `off` keeps it "
                     "in memory only. Does not change the numbers — only whether they are "
                     "read from and written to disk. Quick-analysis (F9-scoped) pulls never "
                     "save.")),
        InString("trial_label", "Trial label", field=False, default="",
                 available_in=available_in,
                 description=(
                     "A name for this combination of settings, shown in the saved trial's "
                     "`trial.json`, the folder's `index.json` and Load Analysis's list — so "
                     "\"subset 32, tol 1e-3\" can be told from \"subset 48\" when you switch "
                     "between trials. Purely a label: it changes neither the result nor which "
                     "saved trial these settings match.")),
    ]


def unit_list(ax: Any, *, per_plane: bool) -> List[Tuple[int, ...]]:
    """The independent units of an analysis over ``ax``: ``(m, z)`` for a planar method
    (DIC, PIV, 2-D tracking — every plane is its own problem), ``(m,)`` for a volumetric one
    (DVC, 3-D tracking). Never split in T."""
    if per_plane:
        return [(m, z) for m in range(int(ax.m)) for z in range(int(ax.z))]
    return [(m,) for m in range(int(ax.m))]


# ── the quick-analysis scope ─────────────────────────────────────────────────────

def scoped_frame(md: Any, ref_frame: int, n_t: int, *, mode: str = "fixed_frame",
                 node: str = "") -> int:
    """``ref_frame`` — a SOURCE timepoint, as the user set it — as its index in the payload
    actually wired in (2026-10-09).

    Under the F9 / Quick-analysis scope the source seed holds only the picked frames,
    renumbered from 0, and the runner stamps the source index of each as ``scope_t``. A
    fixed reference frame of 5 with frames {2, 5} picked is payload frame 1 — read as
    frame 5 it would be clamped onto the LAST picked frame and correlate against the wrong
    image without a word. Unscoped, a ``previous_frame`` policy, or a ``scope_t`` that no
    longer matches the axis (a node upstream changed T) → ``ref_frame`` unchanged; a
    reference the picks leave out is refused, naming the fix."""
    if mode != "fixed_frame":
        return ref_frame
    st = md.get("scope_t") if hasattr(md, "get") else None
    if not st:
        return ref_frame
    picks = [int(x) for x in st]
    if len(picks) != int(n_t):
        return ref_frame
    if int(ref_frame) in picks:
        return picks.index(int(ref_frame))
    raise ValueError(
        f"{node or 'this analysis'}: the quick-analysis scope holds frames {picks}, which do "
        f"not include the reference frame {ref_frame} — every frame is correlated against "
        f"it. Pick it on the Viewer's T strip too (the Quick analysis button does), or set "
        f"Reference frame to one of the picked frames.")


# ── the trial ────────────────────────────────────────────────────────────────────

def _is_scoped(md: Any) -> bool:
    return any(k in md for k in ("scope_m", "scope_t", "scope_z", "scope_region"))


def _source_paths(md: Any) -> List[str]:
    from nodegraph.metadata import SOURCE_PATH_KEY
    got = md.get(SOURCE_PATH_KEY)
    if isinstance(got, (list, tuple)):
        return [str(p) for p in got if p]
    return []


def _store_root(md: Any) -> Optional[str]:
    """Beside the first position's source file; with no file but ``NODEGRAPH_STORE_DIR``
    set (headless runs, the selftest), an ``unsourced.analysis`` folder there."""
    from nodegraph.analysis_store import STORE_SUFFIX, trial_root
    paths = _source_paths(md)
    if paths:
        return trial_root(paths[0])
    override = os.environ.get("NODEGRAPH_STORE_DIR", "").strip().strip('"').strip("'")
    if override:
        return os.path.join(override, "unsourced" + STORE_SUFFIX)
    return None


def _spare_root(md: Any, root: str) -> str:
    """The spare folder for the first position's source file (``""`` with no file, or when
    it is ``root`` itself): where a unit goes when ``root`` refuses it."""
    from nodegraph.analysis_store import spare_root
    paths = _source_paths(md)
    if not paths:
        return ""
    sp = spare_root(paths[0])
    same = os.path.normcase(os.path.abspath(sp)) == os.path.normcase(os.path.abspath(root))
    return "" if same else sp


@dataclass
class TrialOpen:
    """What :func:`open_node_trial` decided: the trial (or ``None``) and why, in words."""
    trial: Any
    note: str


def open_node_trial(ctx: EvalContext, ds: Any, units: Sequence[Tuple[int, ...]], *,
                    axes_names: Sequence[str], analysis: str) -> TrialOpen:
    """The saved trial this pull reads from and writes to, or ``None`` with the reason:
    ``results = off``, a scoped quick-analysis pull, or no source file to sit beside.

    The trial lives beside the source, with the spare folder (``analysis_store.spare_root``)
    as its second home: a unit the source's folder refuses is saved there, and both are
    searched on every run. When the source's folder cannot hold the trial at all, the spare
    becomes the trial's home and the source's folder is still read."""
    import nodegraph.analysis_store as S
    mode = str(ctx.params.get("results", "reuse") or "reuse")
    if mode == "off":
        return TrialOpen(None, "results not saved (off)")
    md = ds.metadata
    if _is_scoped(md):
        return TrialOpen(None, "quick analysis — results not saved")
    root = _store_root(md)
    if root is None:
        return TrialOpen(None, "no source file to save beside")
    lineage = ctx.lineage(exclude=STORE_PARAMS)
    key = S.trial_key(ctx.op_key, lineage)
    names = [S.unit_name(u, axes_names) for u in units]
    files = []
    for p in dict.fromkeys(_source_paths(md)):
        try:
            st = os.stat(p)
            files.append({"path": p, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        except OSError:
            files.append({"path": p})
    settings = {k: v for k, v in ctx.params.items()
                if not str(k).startswith("__") and k not in STORE_PARAMS}
    ax = ds.image.axes if getattr(ds, "image", None) is not None else ds.axes
    calib = {}
    for k in ("pixel_size_um", "z_step_um", "dt_s"):
        try:
            v = ctx.calib(k)
        except Exception:  # noqa: BLE001 — provenance only
            v = None
        if v is not None:
            calib[k] = v
    header = {
        "analysis": analysis,
        "label": str(ctx.params.get("trial_label", "") or ""),
        "settings": settings,
        "modes": dict(ctx.params.get("__modes__", {}) or {}),
        "lineage": lineage,
        "upstream": ctx.upstream_chain(),
        "source_files": files,
        "axes": {a: int(getattr(ax, a)) for a in ("m", "t", "z", "c", "y", "x")},
        "unit_axes": list(axes_names),
        "calibration": calib,
        "about": (f"{analysis} results, saved one unit ({' × '.join(axes_names)}) per file "
                  "under data/ as each finished. 'settings' and 'modes' are this node's; "
                  "'upstream' is the chain that fed it; 'source_files' the image(s) analysed."),
    }
    spare = _spare_root(md, root)
    note = f"trial {S.trial_folder_name(ctx.op_key, key)}"
    try:
        tr = S.open_trial(root, op_key=ctx.op_key, key=key, units=names, header=header,
                          alt_root=spare)
    except OSError as exc:
        if not spare:
            return TrialOpen(None, f"cannot write beside the source ({exc}) — not saved")
        try:
            tr = S.open_trial(spare, op_key=ctx.op_key, key=key, units=names, header=header,
                              alt_root=root)
        except OSError as exc2:
            return TrialOpen(None, f"cannot write beside the source ({exc}) nor in the "
                                   f"spare folder {spare} ({exc2}) — not saved")
        note = (f"{note}, saved in the spare folder {spare}: the source's folder refused "
                f"it ({exc})")
    if mode == "recompute":
        tr.clear()
    return TrialOpen(tr, note)


# ── handing a big array to a worker ─────────────────────────────────────────────

#: Unit inputs above this go to a worker as a memory-mapped temp file, not a pickle: a
#: pickled 2 GB volume series costs a second copy in the parent, another in the pipe and a
#: third in the worker, and seconds of serialization on each side.
SHIP_BYTES = 256 << 20


@dataclass
class ShippedArray:
    """A ``.npy`` in local temp space standing in for an array; :meth:`get` memory-maps it
    (read-only) in whichever process holds it. :func:`ship` makes one; the parent deletes
    the file once the unit is folded (:meth:`discard`)."""
    path: str
    shape: Tuple[int, ...]
    dtype: str

    def get(self) -> np.ndarray:
        return np.load(self.path, mmap_mode="r")

    def discard(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


def ship(frames: Sequence[np.ndarray] | np.ndarray, *, force: bool = False):
    """``frames`` (an array, or a sequence of equal-shape arrays read one at a time) as
    something a worker can receive: the stacked array itself when small, a
    :class:`ShippedArray` written frame by frame (never holding the whole stack) when it
    is larger than :data:`SHIP_BYTES` or ``force``."""
    if isinstance(frames, np.ndarray):
        if frames.nbytes <= SHIP_BYTES and not force:
            return frames
        first, n, rest = frames[0], frames.shape[0], None
    else:
        frames = list(frames) if not hasattr(frames, "__len__") else frames
        n = len(frames)
        first = np.asarray(frames[0])
        if first.nbytes * n <= SHIP_BYTES and not force:
            return np.stack([first] + [np.asarray(frames[i]) for i in range(1, n)])
    fd, path = tempfile.mkstemp(prefix="ng-unit-", suffix=".npy")
    os.close(fd)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=first.dtype,
                                   shape=(n,) + tuple(first.shape))
    for i in range(n):
        mm[i] = np.asarray(frames[i])
    mm.flush()
    del mm
    return ShippedArray(path=path, shape=(n,) + tuple(first.shape), dtype=str(first.dtype))


def unship(x: Any) -> np.ndarray:
    """The array behind :func:`ship`'s result, in whichever process holds it."""
    return x.get() if isinstance(x, ShippedArray) else x


def _discard_shipped(payload: Any) -> None:
    if isinstance(payload, ShippedArray):
        payload.discard()
    elif isinstance(payload, dict):
        for v in payload.values():
            _discard_shipped(v)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            _discard_shipped(v)


# ── the run ──────────────────────────────────────────────────────────────────────

def _unit_task(payload: Tuple[Callable[[Any], dict], Any, str, str, str]) -> dict:
    """The task a worker runs: the node's own unit worker, then — in that same process —
    the unit's save, so a cancel or crash in the parent loses nothing that finished. The
    save goes to the trial's folder, or to its spare when that folder refuses it."""
    work, inner, data_dir, alt_dir, name = payload
    res = work(inner)
    if data_dir:
        from nodegraph.analysis_store import save_unit_to
        try:
            refused = save_unit_to(data_dir, alt_dir, name, res.get("tables") or {},
                                   res.get("stats") or {})
            if refused:
                res = dict(res, saved_spare=refused)
        except OSError as exc:
            # the result is finished. A store that cannot be written is a lost
            # convenience, not a lost analysis: say so on the output, never fail the run
            res = dict(res, save_error=f"{type(exc).__name__}: {exc}")
    return res


def _save_note(res: dict) -> str:
    """What the progress line says about a unit's save when it did not simply land."""
    if res.get("save_error"):
        return " · NOT saved"
    if res.get("saved_spare"):
        return " · saved in the spare folder"
    return ""


@dataclass
class UnitRun:
    """What :func:`run_units` returns: every unit's result in unit order, and how it went."""
    results: List[dict]
    loaded: int
    computed: int
    governor: Any
    lanes_note: str
    unit_seconds: Optional[float]

    @property
    def save_errors(self) -> List[str]:
        """Why each unit that was computed but could not be written to the trial failed."""
        return [str(r["save_error"]) for r in self.results if r.get("save_error")]

    @property
    def spare_saves(self) -> List[str]:
        """Why the trial's folder refused each unit that was saved in the spare instead."""
        return [str(r["saved_spare"]) for r in self.results if r.get("saved_spare")]


def run_units(ctx: EvalContext, units: Sequence[Tuple[int, ...]], *,
              read: Callable[[Tuple[int, ...]], Any],
              work: Callable[[Any], dict],
              read_local: Optional[Callable[[Tuple[int, ...]], Any]] = None,
              names: Sequence[str],
              trial: Any = None,
              worker_bytes: int,
              min_inner: int = 1,
              numba: bool = True,
              note: str = "analysing",
              in_thread: Optional[Callable[[Any, Callable[[int], None]], dict]] = None,
              preimport: Sequence[str] = (),
              ) -> UnitRun:
    """Every unit's result, in unit order — loaded from ``trial`` where it is saved,
    computed otherwise.

    ``read(unit)`` builds the unit's payload on THIS thread (it is the one that may touch
    the engine's providers); ``work(payload)`` is the node's module-level unit worker, run
    in a worker process when the governor affords more than one lane. ``read_local(unit)``,
    when given, replaces ``read`` on the in-process path — a payload that may hold lazy,
    unpicklable readers (provider closures), so one lane reads frame by frame instead of
    shipping a stacked copy it does not need. ``in_thread(payload,
    progress_cb)`` is an optional in-process variant that reports the kernel's own 0-100
    progress — used on the one-lane path, where a long unit would otherwise show a frozen
    bar. ``worker_bytes`` is the first guess at one worker's memory; the governor replaces
    it with the measured peak after the first unit."""
    import nodegraph.parallel as P
    from nodegraph.catalog._shared.progress import _UnitBar
    from nodegraph.governor import Governor, UnitStats, peak_rss_bytes

    n = len(units)
    results: List[Optional[dict]] = [None] * n
    bar = _UnitBar(ctx, frames=max(1, n), units_per_frame=1, note=note)
    todo: List[int] = []
    for i, u in enumerate(units):
        if trial is not None and trial.has(names[i]):
            try:
                tables, stats = trial.load(names[i])
                results[i] = {"tables": tables, "stats": stats, "loaded": True}
                bar.finish_unit(note=f"{names[i]}: loaded from the saved trial")
                continue
            except Exception:  # noqa: BLE001 — an unreadable unit is recomputed, not fatal
                pass
        todo.append(i)
    loaded = n - len(todo)
    if not todo:
        return UnitRun([r for r in results if r is not None], loaded, 0, None,
                       "all units loaded", None)

    gov = Governor(len(todo), worker_bytes, min_inner=min_inner)
    serial = (gov.plan.lanes <= 1 or len(todo) < 2 or not P.enabled()
              or P.in_parallel_region() or P.mode() == "thread")
    if serial:
        for i in todo:
            payload = (read_local or read)(units[i])
            t0 = time.perf_counter()
            try:
                res = in_thread(payload, bar.emit) if in_thread is not None else work(payload)
            finally:
                _discard_shipped(payload)
            if trial is not None:
                try:
                    refused = trial.save(names[i], res.get("tables") or {},
                                         res.get("stats") or {})
                    if refused:
                        res = dict(res, saved_spare=refused)
                except OSError as exc:
                    res = dict(res, save_error=f"{type(exc).__name__}: {exc}")
            gov.record(UnitStats(seconds=time.perf_counter() - t0,
                                 peak_bytes=peak_rss_bytes(), pid=os.getpid()))
            results[i] = res
            bar.finish_unit(note=f"{names[i]} ({time.perf_counter() - t0:.1f} s)"
                                 f"{_save_note(res)}")
        return UnitRun([r for r in results if r is not None], loaded, len(todo), gov,
                       "1 lane (in this process)", gov.mean_seconds())

    data_dir = trial.data_dir if trial is not None else ""
    alt_dir = trial.alt_data_dir if trial is not None else ""
    shipped: Dict[int, Any] = {}

    def prepare(i: int):
        payload = read(units[i])
        shipped[i] = payload
        return (work, payload, data_dir, alt_dir, names[i])

    def fold(_k: int, i: int, res: dict) -> None:
        results[i] = res
        _discard_shipped(shipped.pop(i, None))
        bar.finish_unit(note=f"{names[i]} · {gov.note()}{_save_note(res)}")

    try:
        P.adaptive_fold(_unit_task, todo, fold, governor=gov, prepare=prepare,
                        numba=numba, on_wait=lambda: bar.emit_unknown(note=gov.note()),
                        preimport=preimport)
    finally:
        for p in shipped.values():
            _discard_shipped(p)
    return UnitRun([r for r in results if r is not None], loaded, len(todo), gov,
                   gov.fallback or gov.plan.note(), gov.mean_seconds())


# ── quality metrics and the full-run estimate (the quick analysis) ─────────────────

#: The Global scalars each node writes — what an Iterate card's Metric field names to keep
#: the best setting of a sweep. Every unit-running node also writes ``analysis_est_full_s``.
QUALITY: Dict[str, Tuple[str, ...]] = {
    "analysis.dic_correlate": ("dic_solved_frac",),
    "analysis.piv": ("piv_valid_frac", "piv_median_s2n"),
    "analysis.dvc_field": ("dvc_valid_frac", "dvc_median_q"),
    # the staged cards that read frames — each scores its own section
    "dic.initial_guess": ("dic_median_cc",),
    "dic.local_match": ("dic_solved_frac", "dic_hole_frac", "dic_median_iter"),
    "dic.global_step": ("dic_solved_frac",),
    "piv.first_pass": ("piv_valid_frac", "piv_median_s2n"),
    "piv.deform_pass": ("piv_valid_frac", "piv_median_s2n"),
    "dvc.initial_guess": ("dvc_seed_ok_frac",),
    "dvc.local_match": ("dvc_valid_frac", "dvc_median_q"),
    "dvc.global_step": ("dvc_valid_frac", "dvc_median_q"),
}


def _nanmedian(v) -> Optional[float]:
    a = np.asarray(v, dtype=float)
    return float(np.nanmedian(a)) if a.size and np.isfinite(a).any() else None


def _frac(mask) -> Optional[float]:
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if m.size else None
EST_KEY = "analysis_est_full_s"


def quality_layers(op_key: str) -> Tuple[Tuple[Any, str], ...]:
    """``((Domain.GLOBAL, name), …)`` for ``extra_layers`` — the scalars ``op_key`` writes."""
    from nodegraph.domains import Domain
    return tuple((Domain.GLOBAL, n) for n in QUALITY.get(op_key, ()) + (EST_KEY,))


def with_quality(ds: Any, metrics: Dict[str, Optional[float]]) -> Any:
    """``ds`` with each metric as a dimensionless Global scalar (the estimate in seconds) —
    the shape ``analysis.reduce_scalar`` writes, so ``flow.iterate`` reads them as metrics.
    A metric that could not be measured is written as NaN rather than left out, so a sweep
    sees every iteration."""
    from nodegraph.domains import Domain
    from nodegraph.units import with_unit
    for name, v in metrics.items():
        val = float("nan") if v is None else float(v)
        ds = with_unit(ds.with_layer(Domain.GLOBAL, name, np.asarray(val)),
                       Domain.GLOBAL, name, "s" if name == EST_KEY else "")
    return ds


def full_run_estimate(md: Any, ax: Any, run: "UnitRun", *, per_plane: bool,
                      min_inner: int = 1) -> Optional[float]:
    """Seconds the FULL run would take on this machine, from a quick (scoped) run — or
    ``None`` when the pull was not scoped or nothing was computed.

    The scoped unit's measured time is scaled to a full unit by the frame count and the
    analysed area (volume, for DVC) the scope cut away — correlation cost grows with the
    number of grid points and of frame pairs — then spread over the full unit count on
    the lane plan the governor would choose (:func:`nodegraph.governor.estimate_seconds`).
    An estimate, not a promise: shown so the user knows whether to start the full run now
    or overnight."""
    from nodegraph.governor import estimate_seconds, machine_profile, plan_lanes
    of = md.get("scope_of") if hasattr(md, "get") else None
    if not of or run.unit_seconds is None or run.computed == 0:
        return None
    M, T, Z, Y, X = (max(1, int(v)) for v in of)
    n_full = M * Z if per_plane else M
    t_ratio = T / max(1, int(ax.t))
    area = (Y * X) / max(1, int(ax.y) * int(ax.x))
    if not per_plane:
        area *= Z / max(1, int(ax.z))
    unit_full = float(run.unit_seconds) * t_ratio * area
    gov = run.governor
    peak = int(getattr(gov, "peak_bytes", 0) or 0)
    wb = int(max(peak, 1) * max(1.0, area) * 1.25) if peak else (1 << 30)
    plan = plan_lanes(n_full, wb, machine_profile(), min_inner=min_inner)
    measured = None if (gov is not None and gov.plan.lanes > 1) else plan.usable_cores
    return estimate_seconds(unit_full, n_full, plan, measured_threads=measured)


def describe_seconds(s: Optional[float]) -> str:
    if s is None:
        return ""
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ── staged cards: a unit's frames, its rows, and the shared run ─────────────────────

class _LazyFrames:
    """``reader(t)`` for each ``t`` of ``ts`` as an indexable sequence — what :func:`ship`
    streams to disk one frame at a time instead of stacking the whole series in RAM."""

    def __init__(self, reader: Callable[[int], np.ndarray], ts: Sequence[int]) -> None:
        self._reader, self._ts = reader, list(ts)

    def __len__(self) -> int:
        return len(self._ts)

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self._reader(self._ts[i]))


class UnitFrames:
    """The frames one unit reads, addressed by SOURCE timepoint. Built on the pull thread:
    in-process it wraps the provider reader (lazy, nothing copied); for a worker it holds
    the frames as an array or a :class:`ShippedArray` (picklable, read lazily there).
    ``get(t)`` returns the frame in its native dtype — the stage converts it exactly as its
    serial loop did, so results stay bit-identical."""

    def __init__(self, ts: Sequence[int], *, frames: Any = None,
                 reader: Optional[Callable[[int], np.ndarray]] = None) -> None:
        self.ts = [int(t) for t in ts]
        self._index = {t: i for i, t in enumerate(self.ts)}
        self._frames = frames
        self._reader = reader
        self._opened: Any = None

    def __getstate__(self):
        return {"ts": self.ts, "_index": self._index, "_frames": self._frames,
                "_reader": None, "_opened": None}

    def get(self, t: int) -> np.ndarray:
        if self._reader is not None:
            return np.asarray(self._reader(int(t)))
        if self._opened is None:
            self._opened = unship(self._frames)
        return np.asarray(self._opened[self._index[int(t)]])


def unit_frames(ts: Sequence[int], reader: Callable[[int], np.ndarray], *,
                local: bool) -> UnitFrames:
    """The :class:`UnitFrames` for ``ts`` — lazy in-process, shipped for a worker."""
    ts = sorted({int(t) for t in ts})
    if local:
        return UnitFrames(ts, reader=reader)
    return UnitFrames(ts, frames=ship(_LazyFrames(reader, ts)) if ts else None)


@dataclass
class StagePart:
    """One unit of a staged card, folded: the table rows it covers (in the order the worker
    produced them) and the columns it wrote there."""
    unit: Tuple[int, ...]
    rows: np.ndarray
    cols: Dict[str, np.ndarray]
    stats: dict


def stage_result(payload: dict, new_cols: Dict[str, np.ndarray],
                 stats: Optional[dict] = None) -> dict:
    """What a staged card's unit worker returns: the unit's FULL rows with its new columns
    overwritten (so a saved stage loads back as a complete table), its grid rows passed
    through, and per-unit stats."""
    field = dict(payload.get("_rows") or {})
    field.update({k: np.asarray(v) for k, v in new_cols.items()})
    tables = {"field": field} if field else {}
    mesh = payload.get("_mesh")
    if mesh:
        tables["mesh"] = mesh
    return {"tables": tables, "stats": dict(stats or {})}


def run_stage_units(ctx: EvalContext, ds: Any, layer: str, units: Sequence[Tuple[int, ...]],
                    *, axes_names: Sequence[str],
                    rows_of: Callable[[Tuple[int, ...]], np.ndarray],
                    payload_of: Callable[[Tuple[int, ...], bool], dict],
                    work: Callable[[Any], dict], analysis: str, worker_bytes: int,
                    min_inner: int = 1, numba: bool = True, note: str = "analysing",
                    in_thread: Optional[Callable[[Any, Callable[[int], None]], dict]] = None,
                    z_kind: str = "plane_index", preimport: Sequence[str] = (),
                    ) -> Tuple[List[StagePart], "TrialOpen", "UnitRun"]:
    """A staged card's units through :func:`run_units` and the store: each unit's payload
    (``payload_of(unit, local)``) gets the unit's rows of the input table (``rows_of`` — the
    row order the worker writes back in) and its grid rows, the units run across lanes /
    load from the card's trial, and the parts come back in unit order for the card to write.
    The trial records the field and grid placement so Load Analysis reads a stage's trial as
    the complete table at that stage."""
    from nodegraph.analysis_store import unit_name as _uname
    from nodegraph.domains import Domain
    cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT) if a.layer == layer}
    mesh_name = f"{layer}_mesh"
    mesh_cols = {a.name: np.asarray(a.values) for a in ds.layers_on(Domain.POINT)
                 if a.layer == mesh_name}
    names = [_uname(u, axes_names) for u in units]
    opened = open_node_trial(ctx, ds, units, axes_names=axes_names, analysis=analysis)
    rows = {tuple(u): np.asarray(rows_of(u), dtype=np.int64) for u in units}

    def build(u, local: bool) -> dict:
        p = payload_of(u, local)
        idx = rows[tuple(u)]
        p["_rows"] = {k: v[idx] for k, v in cols.items()}
        if mesh_cols:
            keep = np.ones(len(next(iter(mesh_cols.values()))), dtype=bool)
            for a, i in zip(axes_names, u):
                if a in mesh_cols:
                    keep &= np.rint(np.asarray(mesh_cols[a], float)) == int(i)
            p["_mesh"] = {k: v[keep] for k, v in mesh_cols.items()}
        return p

    run = run_units(ctx, units, read=lambda u: build(u, False),
                    read_local=lambda u: build(u, True), work=work, names=names,
                    trial=opened.trial, worker_bytes=worker_bytes, min_inner=min_inner,
                    numba=numba, note=note, in_thread=in_thread, preimport=preimport)
    parts = [StagePart(tuple(u), rows[tuple(u)], (r.get("tables") or {}).get("field", {}),
                       r.get("stats") or {}) for u, r in zip(units, run.results)]
    return parts, opened, run


def finish_stage(opened: "TrialOpen", run: "UnitRun", layer: str, *, stamps: Dict[str, Any],
                 z_kind: str = "plane_index", mesh_kind: str = "",
                 mesh_window: Optional[Sequence[float]] = None) -> None:
    """:func:`finish_trial` for a staged card — its field, and its grid when it has one."""
    tables: Dict[str, Dict[str, Any]] = {
        "field": {"domain": "point", "layer": layer, "z_kind": z_kind}}
    if mesh_kind:
        tables["mesh"] = {"domain": "point", "layer": f"{layer}_mesh", "z_kind": z_kind,
                          "mesh_of": layer, "mesh_kind": mesh_kind}
        if mesh_window is not None:
            tables["mesh"]["mesh_window"] = [float(w) for w in mesh_window]
    finish_trial(opened, run, stamps=stamps, tables=tables)


def concat_tables(results: Sequence[dict], kind: str) -> Dict[str, np.ndarray]:
    """One column dict from every unit's ``kind`` table, in unit order (empty units skipped;
    a column missing from some units is dropped — a partial column would misalign rows)."""
    parts = [r["tables"][kind] for r in results
             if kind in (r.get("tables") or {}) and r["tables"][kind]
             and len(next(iter(r["tables"][kind].values()))) > 0]
    if not parts:
        return {}
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}


def finish_trial(opened: TrialOpen, run: UnitRun, *, stamps: Dict[str, Any],
                 tables: Dict[str, Dict[str, Any]],
                 quality: Optional[Dict[str, Any]] = None) -> None:
    """Record, in the trial's header, what a full read-back needs: the output's metadata
    ``stamps``, how each table kind lands (``{kind: {"domain", "layer", "z_kind"}}``),
    the quality metrics, and ``complete`` / ``partial``."""
    tr = opened.trial
    if tr is None:
        return
    tables = {k: dict(v) for k, v in tables.items()}
    for kind, where in tables.items():                 # the columns, for Load Analysis
        cols = concat_tables(run.results, kind)
        if cols:
            where["columns"] = sorted(set(cols) | {"id"})
    done = len(tr.done_units())
    tr.update(status="complete" if done >= len(tr.units) else "partial",
              stamps=stamps, tables=tables, quality=dict(quality or {}),
              unit_seconds=run.unit_seconds, lanes=run.lanes_note)


def stamp_trial(md_out: Dict[str, Any], opened: TrialOpen, run: UnitRun) -> Dict[str, Any]:
    """The provenance a saving node adds to its output: where the trial lives (or why it is
    not saved) and how many units came from disk."""
    md_out = dict(md_out)
    tr = opened.trial
    md_out["analysis_trial"] = tr.path if tr is not None else ""
    md_out["analysis_trial_note"] = opened.note
    spare = run.spare_saves
    if spare and tr is not None:
        md_out["analysis_trial_note"] += (f"; {len(spare)} unit(s) saved in the spare folder "
                                          f"{tr.alt_root} because the trial's folder refused "
                                          f"them: {spare[0]}")
    errs = run.save_errors
    if errs:
        md_out["analysis_trial_note"] += (f"; {len(errs)} unit(s) NOT saved (results are "
                                          f"complete, the trial is partial): {errs[0]}")
    total = run.loaded + run.computed
    md_out["analysis_trial_units"] = (f"{run.loaded}/{total} loaded, {run.computed} computed"
                                      f" ({run.lanes_note})")
    return md_out


__all__ = [
    "STORE_PARAMS", "ALDVC_PREIMPORT", "store_sockets", "unit_list", "scoped_frame",
    "TrialOpen",
    "open_node_trial",
    "SHIP_BYTES", "ShippedArray", "ship", "unship", "UnitRun", "run_units",
    "concat_tables", "finish_trial", "stamp_trial", "QUALITY", "EST_KEY",
    "UnitFrames", "unit_frames", "StagePart", "stage_result", "run_stage_units",
    "finish_stage",
    "quality_layers", "with_quality", "full_run_estimate", "describe_seconds",
    "_nanmedian", "_frac",
]
