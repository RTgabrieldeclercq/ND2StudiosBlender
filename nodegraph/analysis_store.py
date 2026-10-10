"""The analysis store — expensive results saved beside their .nd2 as they are produced, and
loaded back instead of recomputed (2026-10-09).

A DIC, DVC, PIV or SerialTrack run can take hours, and until now its only home was the
in-memory memo: a crash, a cancel or a restart repeated all of it. This module is where
those results live on disk.

Layout — one folder beside the source file, one sub-folder per **trial** (a settings
combination), one file per finished **unit** (a position, or a position × plane)::

    Channel640_Seq0001.nd2
    Channel640_Seq0001.analysis/
      index.json                     every trial: id, analysis, label, settings, status, k/n
      dic_correlate-3f9a2c1b/
        trial.json                   what made it: op, full settings, the upstream chain,
                                     the source file(s) with size + mtime, axes, units,
                                     the result's metadata stamps, quality metrics
        data/
          m0003_z0000.field.parquet  the unit's rows of the result table(s)
          m0003_z0000.mesh.parquet   …one file per table kind…
          m0003_z0000.done.json      written LAST: present ⇔ the unit is complete

Why this shape:

* **A unit is the save point.** The analysis nodes run unit by unit (position, or
  position × plane — never split in time), in parallel worker processes. Each worker writes
  its own unit's files the moment it finishes, so a cancelled or crashed run keeps every
  finished unit and the next pull resumes at the first missing one. Files are written to a
  temporary name and renamed into place (atomic on every platform), and the ``.done.json``
  marker goes last — a unit with no marker is simply not there.
* **Parquet** for the tables: typed, columnar, compressed, and readable directly by pandas
  (``pd.read_parquet(folder)``), R and MATLAB — a trial is useful outside this app.
* **One folder per trial**, keyed by a hash of the node's settings and its input's
  :meth:`~nodegraph.engine.Engine.lineage` (the upstream chain and the source file's
  identity, stable across sessions). Different settings → a different folder; settings
  changed back → the earlier folder is found and loaded, which is how a user flips between
  trials. ``trial.json`` explains each folder in words a person can read; ``index.json``
  lists them all and is only a convenience — it is rebuilt from the ``trial.json`` files
  whenever it is missing or stale, so two writers can never corrupt anything that matters.
* Honours ``NODEGRAPH_STORE_DIR`` exactly as ingest stores and docks do
  (:func:`nodegraph.parallel.store_dir`): beside the file by default, one fast directory when
  the source sits on slow media.
* **A spare folder** (:data:`SPARE_DIR`, per user, local). When the folder beside the source
  refuses a unit — a read-only share, a full disk, a failing drive — the unit is saved there
  instead, and every run looks in both. Until 2026-10-09 such a unit was only reported as
  "not saved", so a restart repeated hours of work on exactly the disks that can least
  afford it. The two halves of a split trial name each other in ``trial.json``
  (``also_saved_in``), so Load Analysis reads the whole trial from either.

Qt-free. pyarrow is imported lazily, on first read or write.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Bumped when the on-disk layout or a table's meaning changes; part of every trial key,
#: so an old trial is never read as a new one.
SCHEMA = 1

STORE_SUFFIX = ".analysis"
INDEX_NAME = "index.json"
TRIAL_NAME = "trial.json"
DATA_DIR = "data"
DONE_SUFFIX = ".done.json"


# ── atomic file writes ──────────────────────────────────────────────────────────

def _discard_tmp(tmp: str) -> None:
    """Remove a temp file whose write failed. A failed save no longer fails the run
    (``analysis_units._unit_task``), so without this a full or failing store would
    collect half-written ``*.tmp`` files."""
    try:
        os.remove(tmp)
    except OSError:
        pass


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
    except BaseException:
        _discard_tmp(tmp)
        raise
    os.replace(tmp, path)


def _json_safe(v: Any) -> Any:
    """``v`` as plain JSON: numpy scalars/arrays to Python, tuples to lists, anything else
    unknown to its ``repr`` (a trial header must always be writable)."""
    if isinstance(v, (str, bool)) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v if not (isinstance(v, float) and not np.isfinite(v)) else None
    if isinstance(v, np.generic):
        return _json_safe(v.item())
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()] if v.size <= 4096 else repr(v)
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_json_safe(x) for x in v]
    return repr(v)


def _write_json(path: str, obj: Any) -> None:
    _atomic_write_bytes(path, json.dumps(_json_safe(obj), indent=2, sort_keys=False)
                        .encode("utf-8"))


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


# ── where and what ──────────────────────────────────────────────────────────────

def trial_root(source_path: str) -> str:
    """The store folder for ``source_path``: ``<stem>.analysis`` beside it, or — when
    ``NODEGRAPH_STORE_DIR`` redirects stores — a folder of that name, made unique by a digest
    of the full path, inside the redirect."""
    from nodegraph.parallel import store_dir
    stem = os.path.splitext(os.path.abspath(str(source_path)))[0]
    base = stem + STORE_SUFFIX
    target = store_dir(os.path.dirname(base))
    if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.dirname(base)):
        return base
    tag = hashlib.blake2b(base.lower().encode("utf-8"), digest_size=6).hexdigest()
    return os.path.join(target, f"{os.path.basename(base)}.{tag}")


#: Where a unit goes when the folder beside its source refuses the write. Per user and on
#: the local disk, so a long analysis survives a restart whatever the data's disk is doing.
#: A module attribute so a test can move it; worker processes are handed the path itself.
SPARE_DIR = os.path.join(os.path.expanduser("~"), ".nd2studios", "analysis")


def spare_root(source_path: str) -> str:
    """The spare store folder for ``source_path``: ``<stem>.analysis.<digest>`` inside
    :data:`SPARE_DIR`. The digest of the full path keeps two same-named files from different
    folders apart, as :func:`trial_root`'s redirect does."""
    stem = os.path.splitext(os.path.abspath(str(source_path)))[0]
    base = stem + STORE_SUFFIX
    tag = hashlib.blake2b(base.lower().encode("utf-8"), digest_size=6).hexdigest()
    return os.path.join(SPARE_DIR, f"{os.path.basename(base)}.{tag}")


def trial_roots(source_path: str) -> List[str]:
    """Every folder that may hold ``source_path``'s trials: :func:`trial_root`, then
    :func:`spare_root` (listed once when the two are the same folder)."""
    out = [trial_root(source_path)]
    sp = spare_root(source_path)
    if os.path.normcase(os.path.abspath(sp)) != os.path.normcase(os.path.abspath(out[0])):
        out.append(sp)
    return out


def trial_folder_name(op_key: str, key: str) -> str:
    """``analysis.dic_correlate`` + key → ``dic_correlate-3f9a2c1b``; ``dic.local_match`` →
    ``dic_local_match-…`` — the analysis readable first, the settings hash after."""
    short = op_key[len("analysis."):] if op_key.startswith("analysis.") else op_key
    return f"{short.replace('.', '_')}-{key[:8]}"


def trial_key(op_key: str, lineage: str) -> str:
    """The trial identity: the schema, the op and the node's lineage taken WITHOUT its
    store-control sockets (so changing ``results``/``trial_label`` never moves a trial)."""
    from nodegraph.memo import digest
    return digest("trial", SCHEMA, op_key, lineage)


def unit_name(unit: Sequence[int], axes: Sequence[str] = ("m", "z")) -> str:
    """``(3, 0)`` → ``m0003_z0000`` (or ``m0003`` for a one-axis unit)."""
    return "_".join(f"{a}{int(i):04d}" for a, i in zip(axes, unit))


# ── tables ⇄ parquet ────────────────────────────────────────────────────────────

def _to_arrow(cols: Mapping[str, np.ndarray]):
    import pyarrow as pa
    arrays, names = [], []
    for k, v in cols.items():
        a = np.asarray(v)
        if a.dtype == object:
            a = a.astype(str)
        arrays.append(pa.array(a))
        names.append(str(k))
    return pa.Table.from_arrays(arrays, names=names)


def write_table(path: str, cols: Mapping[str, np.ndarray]) -> None:
    import pyarrow.parquet as pq
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        pq.write_table(_to_arrow(cols), tmp, compression="zstd")
    except BaseException:
        _discard_tmp(tmp)
        raise
    os.replace(tmp, path)


def read_table(path: str) -> Dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return {name: np.asarray(t.column(name).to_numpy(zero_copy_only=False))
            for name in t.column_names}


def save_unit(data_dir: str, name: str, tables: Mapping[str, Mapping[str, np.ndarray]],
              stats: Optional[Mapping[str, Any]] = None) -> None:
    """Write one finished unit: every table as ``<name>.<kind>.parquet``, then the
    ``<name>.done.json`` marker that makes it exist. Pure file I/O, so a worker process
    calls it directly with nothing but the folder path."""
    os.makedirs(data_dir, exist_ok=True)
    kinds = []
    for kind, cols in tables.items():
        write_table(os.path.join(data_dir, f"{name}.{kind}.parquet"), cols)
        kinds.append(kind)
    _write_json(os.path.join(data_dir, name + DONE_SUFFIX),
                {"unit": name, "tables": kinds, "stats": dict(stats or {}),
                 "saved": time.strftime("%Y-%m-%dT%H:%M:%S")})


def save_unit_to(data_dir: str, alt_dir: str, name: str,
                 tables: Mapping[str, Mapping[str, np.ndarray]],
                 stats: Optional[Mapping[str, Any]] = None) -> str:
    """:func:`save_unit` into ``data_dir`` or, when that folder refuses the write and
    ``alt_dir`` is given, into ``alt_dir``.

    Returns ``""`` when the unit landed in ``data_dir``, otherwise why ``data_dir`` refused
    it. Raises :class:`OSError` only when both refuse. A unit half-written into
    ``data_dir`` has no ``.done.json`` there, so it does not exist in that folder."""
    try:
        save_unit(data_dir, name, tables, stats)
        return ""
    except OSError as exc:
        if not alt_dir:
            raise
        refused = f"{type(exc).__name__}: {exc}"
    try:
        save_unit(alt_dir, name, tables, stats)
    except OSError as exc:
        raise OSError(f"{refused}; the spare folder refused it too: "
                      f"{type(exc).__name__}: {exc}") from exc
    return refused


def unit_done(data_dir: str, name: str) -> bool:
    return os.path.isfile(os.path.join(data_dir, name + DONE_SUFFIX))


def load_unit(data_dir: str, name: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], dict]:
    """``(tables, stats)`` of a finished unit."""
    done = _read_json(os.path.join(data_dir, name + DONE_SUFFIX)) or {}
    tables = {kind: read_table(os.path.join(data_dir, f"{name}.{kind}.parquet"))
              for kind in done.get("tables", [])}
    return tables, dict(done.get("stats") or {})


# ── a trial ─────────────────────────────────────────────────────────────────────

@dataclass
class Trial:
    """One trial, held in one folder or split across two. Plain data — picklable — so a
    worker can be handed one.

    ``alt_root``, when set, is a second root holding the same trial folder: units are looked
    up in both and saved there when ``root`` refuses the write. Normally ``root`` is beside
    the source and ``alt_root`` the spare (:func:`spare_root`); when the source's folder
    cannot even be created, the two swap, so what is already saved beside the source is
    still read."""
    root: str
    folder: str
    key: str
    op_key: str
    units: List[str] = field(default_factory=list)
    alt_root: str = ""

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.folder)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.path, DATA_DIR)

    @property
    def alt_path(self) -> str:
        return os.path.join(self.alt_root, self.folder) if self.alt_root else ""

    @property
    def alt_data_dir(self) -> str:
        return os.path.join(self.alt_path, DATA_DIR) if self.alt_root else ""

    def _dir_of(self, name: str) -> str:
        if unit_done(self.data_dir, name):
            return self.data_dir
        if self.alt_root and unit_done(self.alt_data_dir, name):
            return self.alt_data_dir
        return ""

    def has(self, name: str) -> bool:
        return bool(self._dir_of(name))

    def done_units(self) -> List[str]:
        return [u for u in self.units if self.has(u)]

    def alt_units(self) -> List[str]:
        """The finished units only the alternate folder holds."""
        if not self.alt_root:
            return []
        return [u for u in self.units
                if unit_done(self.alt_data_dir, u) and not unit_done(self.data_dir, u)]

    def load(self, name: str):
        return load_unit(self._dir_of(name) or self.data_dir, name)

    def save(self, name: str, tables, stats=None) -> str:
        """Save one unit (:func:`save_unit_to`): ``""``, or why ``root`` refused it and the
        unit went to ``alt_root``."""
        return save_unit_to(self.data_dir, self.alt_data_dir, name, tables, stats)

    def header(self) -> dict:
        got = _read_json(os.path.join(self.path, TRIAL_NAME))
        if got is None and self.alt_root:
            got = _read_json(os.path.join(self.alt_path, TRIAL_NAME))
        return got or {}

    def update(self, **fields: Any) -> None:
        """Merge ``fields`` into ``trial.json`` and refresh the index — in both folders once
        the alternate one holds a unit, each header naming the other (``also_saved_in``).
        Best-effort: a header that cannot be written (a read-only share, a file open in an
        editor) never fails the analysis — the unit files are the record, the header only
        explains them."""
        hdr = self.header()
        hdr.update(fields)
        hdr["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        hdr["units_done"] = len(self.done_units())
        hdr.pop("also_saved_in", None)
        places = [(self.root, self.path, "")]
        if self.alt_units() or (self.alt_root and
                                os.path.isfile(os.path.join(self.alt_path, TRIAL_NAME))):
            places = [(self.root, self.path, self.alt_path),
                      (self.alt_root, self.alt_path, self.path)]
        for root, path, other in places:
            try:
                _write_json(os.path.join(path, TRIAL_NAME),
                            dict(hdr, also_saved_in=other) if other else hdr)
                refresh_index(root)
            except OSError:
                pass

    def clear(self) -> None:
        """Forget every saved unit, in both folders (``results = recompute``)."""
        for ddir in (self.data_dir, self.alt_data_dir):
            if not ddir:
                continue
            try:
                for fn in os.listdir(ddir):
                    if fn.endswith(DONE_SUFFIX):
                        os.remove(os.path.join(ddir, fn))
            except OSError:
                pass


def open_trial(root: str, *, op_key: str, key: str, units: Sequence[str],
               header: Mapping[str, Any], alt_root: str = "") -> Trial:
    """Open (creating if needed) the trial for ``key`` under ``root``, with ``alt_root`` as
    its second folder (:class:`Trial`). A new trial writes its ``trial.json`` from
    ``header``; an existing one keeps its header and refreshes only the fields a new run may
    legitimately change (the label, the unit list). Raises :class:`OSError` when ``root``
    cannot hold the trial folder at all."""
    folder = trial_folder_name(op_key, key)
    tr = Trial(root=root, folder=folder, key=key, op_key=op_key, units=list(units),
               alt_root=alt_root)
    os.makedirs(tr.data_dir, exist_ok=True)
    hdr_path = os.path.join(tr.path, TRIAL_NAME)
    existing = _read_json(hdr_path)
    if existing is None and alt_root:
        existing = _read_json(os.path.join(tr.alt_path, TRIAL_NAME))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if existing is None or existing.get("key") != key:
        hdr = {"schema": SCHEMA, "key": key, "id": folder, "op": op_key,
               "created": now, "status": "running"}
        hdr.update(header)
        hdr["units"] = list(units)
        hdr["units_total"] = len(units)
        hdr["units_done"] = len(tr.done_units())
        hdr["updated"] = now
        try:
            _write_json(hdr_path, hdr)
        except OSError:
            pass
        refresh_index(root)
    else:
        tr.update(label=header.get("label", existing.get("label", "")),
                  units=list(units), units_total=len(units))
    return tr


# ── the index ───────────────────────────────────────────────────────────────────

def _summary(hdr: Mapping[str, Any]) -> str:
    """A one-line settings summary for a list row: the node's own non-default-looking
    params, short."""
    s = hdr.get("settings") or {}
    parts = []
    for k, v in s.items():
        if k.startswith("__") or k in ("results", "trial_label") or isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={v}")
    out = ", ".join(parts)
    return out if len(out) <= 160 else out[:157] + "…"


def _unit_dirs(path: str, hdr: Mapping[str, Any]) -> List[str]:
    """The data folders holding the trial at ``path``: its own, and the other half's when
    the header names one (``also_saved_in``)."""
    dirs = [os.path.join(path, DATA_DIR)]
    other = str(hdr.get("also_saved_in") or "")
    if other and os.path.normcase(os.path.abspath(other)) != os.path.normcase(
            os.path.abspath(path)):
        dirs.append(os.path.join(other, DATA_DIR))
    return dirs


def list_trials(root: str) -> List[dict]:
    """Every trial under ``root``, newest first, read from the ``trial.json`` files (the
    source of truth) with the finished-unit count re-counted from disk."""
    out: List[dict] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        hdr = _read_json(os.path.join(root, name, TRIAL_NAME))
        if not hdr:
            continue
        units = list(hdr.get("units") or [])
        dirs = _unit_dirs(os.path.join(root, name), hdr)
        done = sum(1 for u in units if any(unit_done(d, u) for d in dirs))
        total = int(hdr.get("units_total") or len(units))
        status = hdr.get("status", "")
        if status == "complete" and done < total:
            status = "partial"
        out.append({"id": name, "path": os.path.join(root, name), "op": hdr.get("op", ""),
                    "analysis": hdr.get("analysis", hdr.get("op", "")),
                    "label": hdr.get("label", ""), "status": status,
                    "units_done": done, "units_total": total,
                    "created": hdr.get("created", ""), "updated": hdr.get("updated", ""),
                    "settings": _summary(hdr), "quality": hdr.get("quality") or {},
                    "source_files": hdr.get("source_files") or []})
    out.sort(key=lambda r: r.get("updated", ""), reverse=True)
    return out


def refresh_index(root: str) -> None:
    """Rewrite ``index.json`` from the trial folders (best-effort, see :meth:`Trial.update`)."""
    try:
        _write_json(os.path.join(root, INDEX_NAME),
                    {"schema": SCHEMA,
                     "about": ("Saved analysis trials for the image beside this folder. Each "
                               "sub-folder is one trial (one combination of settings); its "
                               "trial.json says what made it, and data/ holds one Parquet "
                               "file per finished position (and plane) per result table."),
                     "trials": list_trials(root)})
    except OSError:
        pass


def list_source_trials(source_paths: Iterable[str]) -> List[dict]:
    """:func:`list_trials` over every folder that may hold the trials of ``source_paths``
    (:func:`trial_roots`), one row per trial: of a trial found in two folders, the row
    counting more finished units. Newest first."""
    best: Dict[str, dict] = {}
    seen = set()
    for sp in source_paths:
        if not sp:
            continue
        for root in trial_roots(str(sp)):
            if root in seen:
                continue
            seen.add(root)
            for r in list_trials(root):
                cur = best.get(r["id"])
                if cur is None or r["units_done"] > cur["units_done"]:
                    best[r["id"]] = r
    out = list(best.values())
    out.sort(key=lambda r: r.get("updated", ""), reverse=True)
    return out


def find_trial(source_paths: Iterable[str], trial_id: str) -> Optional[str]:
    """The folder of the trial ``trial_id`` among ``source_paths``' trials
    (:func:`list_source_trials`), or ``None``."""
    return next((r["path"] for r in list_source_trials(source_paths) if r["id"] == trial_id),
                None)


def trial_choices(source_paths: Iterable[str]) -> List[Tuple[str, str]]:
    """``[(label, trial_path)]`` for the trials of ``source_paths``, beside each file or in
    the spare folder — what the Load Analysis node's dropdown lists."""
    out: List[Tuple[str, str]] = []
    for r in list_source_trials(source_paths):
        name = r["label"] or r["id"]
        text = (f"{name} · {r['analysis']} · {r['units_done']}/{r['units_total']} "
                f"{r['status']}")
        out.append((text, r["path"]))
    return out


@dataclass
class TrialData:
    """A trial read back whole: its header and every finished unit's tables."""
    path: str
    header: dict
    units: List[str]
    tables: Dict[str, List[Dict[str, np.ndarray]]]     # kind → per-unit column dicts
    stats: Dict[str, dict]


def read_trial(path: str) -> TrialData:
    """Every finished unit of the trial at ``path`` (a trial folder, or its ``trial.json``),
    including those its other half holds (``also_saved_in``)."""
    if os.path.basename(path).lower() == TRIAL_NAME:
        path = os.path.dirname(path)
    hdr = _read_json(os.path.join(path, TRIAL_NAME))
    if hdr is None:
        raise FileNotFoundError(f"{path!r} is not an analysis trial folder (no {TRIAL_NAME})")
    dirs = _unit_dirs(path, hdr)
    where = {}
    for u in hdr.get("units") or []:
        d = next((d for d in dirs if unit_done(d, u)), "")
        if d:
            where[u] = d
    units = list(where)
    tables: Dict[str, List[Dict[str, np.ndarray]]] = {}
    stats: Dict[str, dict] = {}
    for u in units:
        t, st = load_unit(where[u], u)
        stats[u] = st
        for kind, cols in t.items():
            tables.setdefault(kind, []).append(cols)
    return TrialData(path=path, header=hdr, units=units, tables=tables, stats=stats)


__all__ = [
    "SCHEMA", "STORE_SUFFIX", "SPARE_DIR", "trial_root", "spare_root", "trial_roots",
    "trial_folder_name", "trial_key", "unit_name",
    "write_table", "read_table", "save_unit", "save_unit_to", "unit_done", "load_unit",
    "Trial", "open_trial", "list_trials", "list_source_trials", "find_trial", "refresh_index",
    "trial_choices",
    "TrialData", "read_trial",
]
