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


def store_sockets() -> list:
    """The ``results`` and ``trial_label`` sockets, declared once for every saving node."""
    return [
        InString("results", "Saved results", field=False, default="reuse",
                 choices=("reuse", "recompute", "off"), choice_docs=_RESULTS_DOCS,
                 description=(
                     "Whether this analysis saves its result beside the source file "
                     "(`<file>.analysis/`, one Parquet file per finished position) and reuses "
                     "it. `reuse` loads what is already saved for these exact settings and "
                     "computes only the rest; `recompute` redoes everything; `off` keeps it "
                     "in memory only. Does not change the numbers — only whether they are "
                     "read from and written to disk. Quick-analysis (F9-scoped) pulls never "
                     "save.")),
        InString("trial_label", "Trial label", field=False, default="",
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


@dataclass
class TrialOpen:
    """What :func:`open_node_trial` decided: the trial (or ``None``) and why, in words."""
    trial: Any
    note: str


def open_node_trial(ctx: EvalContext, ds: Any, units: Sequence[Tuple[int, ...]], *,
                    axes_names: Sequence[str], analysis: str) -> TrialOpen:
    """The saved trial this pull reads from and writes to, or ``None`` with the reason:
    ``results = off``, a scoped quick-analysis pull, or no source file to sit beside."""
    from nodegraph import analysis_store as S
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
    try:
        tr = S.open_trial(root, op_key=ctx.op_key, key=key, units=names, header=header)
    except OSError as exc:
        return TrialOpen(None, f"cannot write beside the source ({exc}) — not saved")
    if mode == "recompute":
        tr.clear()
    return TrialOpen(tr, f"trial {tr.folder}")


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

def _unit_task(payload: Tuple[Callable[[Any], dict], Any, str, str]) -> dict:
    """The task a worker runs: the node's own unit worker, then — in that same process —
    the unit's save, so a cancel or crash in the parent loses nothing that finished."""
    work, inner, data_dir, name = payload
    res = work(inner)
    if data_dir:
        from nodegraph.analysis_store import save_unit
        save_unit(data_dir, name, res.get("tables") or {}, res.get("stats") or {})
    return res


@dataclass
class UnitRun:
    """What :func:`run_units` returns: every unit's result in unit order, and how it went."""
    results: List[dict]
    loaded: int
    computed: int
    governor: Any
    lanes_note: str
    unit_seconds: Optional[float]


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
    from nodegraph import parallel as P
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
                    trial.save(names[i], res.get("tables") or {}, res.get("stats") or {})
                except OSError:
                    pass
            gov.record(UnitStats(seconds=time.perf_counter() - t0,
                                 peak_bytes=peak_rss_bytes(), pid=os.getpid()))
            results[i] = res
            bar.finish_unit(note=f"{names[i]} ({time.perf_counter() - t0:.1f} s)")
        return UnitRun([r for r in results if r is not None], loaded, len(todo), gov,
                       "1 lane (in this process)", gov.mean_seconds())

    data_dir = trial.data_dir if trial is not None else ""
    shipped: Dict[int, Any] = {}

    def prepare(i: int):
        payload = read(units[i])
        shipped[i] = payload
        return (work, payload, data_dir, names[i])

    def fold(_k: int, i: int, res: dict) -> None:
        results[i] = res
        _discard_shipped(shipped.pop(i, None))
        bar.finish_unit(note=f"{names[i]} · {gov.note()}")

    try:
        P.adaptive_fold(_unit_task, todo, fold, governor=gov, prepare=prepare,
                        numba=numba, on_wait=lambda: bar.emit_unknown(note=gov.note()))
    finally:
        for p in shipped.values():
            _discard_shipped(p)
    return UnitRun([r for r in results if r is not None], loaded, len(todo), gov,
                   gov.plan.note(), gov.mean_seconds())


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
    total = run.loaded + run.computed
    md_out["analysis_trial_units"] = (f"{run.loaded}/{total} loaded, {run.computed} computed"
                                      f" ({run.lanes_note})")
    return md_out


__all__ = [
    "STORE_PARAMS", "store_sockets", "unit_list", "TrialOpen", "open_node_trial",
    "SHIP_BYTES", "ShippedArray", "ship", "unship", "UnitRun", "run_units",
    "concat_tables", "finish_trial", "stamp_trial",
]
