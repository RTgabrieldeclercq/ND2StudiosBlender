"""Shim. The engine now lives in the `evidence-log` skill; this directory is the DATA.

Kept so `sys.path.insert(0, "CodeLog/live"); import devlog` keeps working. New code should
use the skill directly:

    import sys; sys.path.insert(0, ".claude/skills/evidence-log")
    import devlog as D
    D.configure("CodeLog/live", title="V2.22 - granule workflow")

Loaded via importlib under a distinct module name on purpose: a plain
`from devlog import *` here would re-enter THIS module, since it is already registered in
sys.modules as "devlog" while its own body is still running.
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".claude", "skills",
                                      "evidence-log", "devlog.py"))
if not _os.path.exists(_SRC):
    raise ImportError(f"evidence-log skill engine not found at {_SRC}")

_spec = _ilu.spec_from_file_location("_evidence_log_engine", _SRC)
_eng = _ilu.module_from_spec(_spec)
_sys.modules["_evidence_log_engine"] = _eng
_spec.loader.exec_module(_eng)

_eng.configure(_HERE, title="V2.22 · granule workflow")

# re-export the public surface
log = _eng.log
ask = _eng.ask
attach = _eng.attach
confirm = _eng.confirm
reject = _eng.reject
render = _eng.render
configure = _eng.configure
export_archive = _eng.export_archive
export_markdown = _eng.export_markdown
_read = _eng._read
_resolve = _eng._resolve
KINDS = _eng.KINDS
STATUS = _eng.STATUS


def __getattr__(name):                    # ROOT / IMG / LOG etc. stay live after reconfig
    return getattr(_eng, name)
