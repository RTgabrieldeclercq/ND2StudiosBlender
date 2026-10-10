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

#: The largest single ``write`` the store issues. Some drives refuse one big write call
#: with ``EINVAL``. The lab's 13 TB S: drive takes 16 MiB but refuses 24 MiB (measured
#: 2026-10-09). pyarrow writes a whole Parquet file in a call that size, so every unit of a
#: real PIV run failed to save, and that failure failed the run. Writing in chunks costs
#: nothing measurable.
WRITE_CHUNK = 4 << 20


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    view = memoryview(data)
    try:
        with open(tmp, "wb") as fh:
            for i in range(0, len(view), WRITE_CHUNK):
                fh.write(view[i:i + WRITE_CHUNK])
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
    except BaseException:
        try:                                # a refused write leaves no half file behind
            os.remove(tmp)
        except OSError:
            pass
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
    """One table as zstd Parquet, built in memory, then written in :data:`WRITE_CHUNK`
    pieces and renamed into place, never handed to pyarrow's own one-call file write."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    buf = pa.BufferOutputStream()
    pq.write_table(_to_arrow(cols), buf, compression="zstd")
    _atomic_write_bytes(path, buf.getvalue().to_pybytes())


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
    """One trial folder. Plain data — picklable — so a worker can be handed one."""
    root: str
    folder: str
    key: str
    op_key: str
    units: List[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.folder)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.path, DATA_DIR)

    def has(self, name: str) -> bool:
        return unit_done(self.data_dir, name)

    def done_units(self) -> List[str]:
        return [u for u in self.units if self.has(u)]

    def load(self, name: str):
        return load_unit(self.data_dir, name)

    def save(self, name: str, tables, stats=None) -> None:
        save_unit(self.data_dir, name, tables, stats)

    def header(self) -> dict:
        return _read_json(os.path.join(self.path, TRIAL_NAME)) or {}

    def update(self, **fields: Any) -> None:
        """Merge ``fields`` into ``trial.json`` and refresh the index. Best-effort: a header
        that cannot be written (a read-only share, a file open in an editor) never fails
        the analysis — the unit files are the record, the header only explains them."""
        try:
            hdr = self.header()
            hdr.update(fields)
            hdr["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            hdr["units_done"] = len(self.done_units())
            _write_json(os.path.join(self.path, TRIAL_NAME), hdr)
            refresh_index(self.root)
        except OSError:
            pass

    def clear(self) -> None:
        """Forget every saved unit (``results = recompute``)."""
        try:
            for fn in os.listdir(self.data_dir):
                if fn.endswith(DONE_SUFFIX):
                    os.remove(os.path.join(self.data_dir, fn))
        except OSError:
            pass


def open_trial(root: str, *, op_key: str, key: str, units: Sequence[str],
               header: Mapping[str, Any]) -> Trial:
    """Open (creating if needed) the trial for ``key`` under ``root``. A new trial writes its
    ``trial.json`` from ``header``; an existing one keeps its header and refreshes only the
    fields a new run may legitimately change (the label, the unit list)."""
    folder = trial_folder_name(op_key, key)
    tr = Trial(root=root, folder=folder, key=key, op_key=op_key, units=list(units))
    os.makedirs(tr.data_dir, exist_ok=True)
    hdr_path = os.path.join(tr.path, TRIAL_NAME)
    existing = _read_json(hdr_path)
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
        ddir = os.path.join(root, name, DATA_DIR)
        done = sum(1 for u in units if unit_done(ddir, u))
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


def trial_choices(source_paths: Iterable[str]) -> List[Tuple[str, str]]:
    """``[(label, trial_path)]`` for the trials beside any of ``source_paths`` — what the
    Load Analysis node's dropdown lists."""
    seen = set()
    out: List[Tuple[str, str]] = []
    for sp in source_paths:
        if not sp:
            continue
        root = trial_root(sp)
        if root in seen:
            continue
        seen.add(root)
        for r in list_trials(root):
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
    """Every finished unit of the trial at ``path`` (a trial folder, or its ``trial.json``)."""
    if os.path.basename(path).lower() == TRIAL_NAME:
        path = os.path.dirname(path)
    hdr = _read_json(os.path.join(path, TRIAL_NAME))
    if hdr is None:
        raise FileNotFoundError(f"{path!r} is not an analysis trial folder (no {TRIAL_NAME})")
    ddir = os.path.join(path, DATA_DIR)
    units = [u for u in (hdr.get("units") or []) if unit_done(ddir, u)]
    tables: Dict[str, List[Dict[str, np.ndarray]]] = {}
    stats: Dict[str, dict] = {}
    for u in units:
        t, st = load_unit(ddir, u)
        stats[u] = st
        for kind, cols in t.items():
            tables.setdefault(kind, []).append(cols)
    return TrialData(path=path, header=hdr, units=units, tables=tables, stats=stats)


__all__ = [
    "SCHEMA", "STORE_SUFFIX", "trial_root", "trial_folder_name", "trial_key", "unit_name",
    "write_table", "read_table", "save_unit", "unit_done", "load_unit",
    "Trial", "open_trial", "list_trials", "refresh_index", "trial_choices",
    "TrialData", "read_trial",
]
