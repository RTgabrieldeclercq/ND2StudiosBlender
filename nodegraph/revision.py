"""Process-monotonic revision counter — the memo *identity* for nodegraph v2.

Every derived value that participates in memoization — an :class:`~nodegraph.
dataset.AttributeLayer`, a memo entry, a computed tile — is stamped with a
``revision``: a strictly increasing integer minted by :func:`next_revision`.

Identity is this cheap monotonic counter, deliberately **not**:

* Python ``id()`` — recycled by the allocator, so a freed-then-reused object
  would collide with a stale memo key (V2.02 §6 cross-cutting invariant).
* a content hash of the payload — an 85 MB plane must never be read just to be
  *identified*; content hashing is reserved for the separate output-fingerprint
  (cutoff/dedup), not for identity (V2.02 §I.a / §6).

Because a recomputed value gets a *fresh* revision, any downstream memo key that
embeds an input's revision auto-invalidates the moment that input is replaced
(V2.02 §4). ``Dataset.with_attribute`` structural-shares via ``replace``, so a
new layer carries a new revision and stale keys fall out naturally.

Thread-safe: the Phase-2 pull scheduler may mint revisions from worker threads.
Qt-free; pure standard library.

**Code fingerprints (live reload).** A revision tracks when a *value* changed; it says
nothing about the *code* that produced it. Live node reload (:mod:`nodegraph.hotreload`)
needs the second axis: once a node's compute has been edited under a running session, every
memo entry computed by the previous version is stale even though nothing upstream moved.
This module therefore also holds the ``op_key -> code fingerprint`` table that
:func:`~nodegraph.memo.node_recipe_hash` folds into its lookup key. It lives here rather
than in ``hotreload`` so ``memo``/``engine`` can read it without importing the reloader.

The fingerprint is a **content digest of the source**, not a counter, so reverting a node
file to a version you already ran restores the memo entries from that run instead of
recomputing them. The empty string means "never stamped" and is omitted from the hash
entirely, which is what keeps a plain headless session's keys identical to what they were
before this table existed.
"""
from __future__ import annotations

import threading
from typing import Dict, Mapping

_lock = threading.Lock()
_current = 0

#: op_key -> source-content fingerprint of the code behind it. Absent/``""`` = unstamped.
_code_fp: Dict[str, str] = {}


def next_revision() -> int:
    """Return the next strictly-increasing revision (thread-safe, never 0)."""
    global _current
    with _lock:
        _current += 1
        return _current


def peek_revision() -> int:
    """The most-recently-issued revision without consuming one (0 before any)."""
    with _lock:
        return _current


# ── code fingerprints (live reload) ──────────────────────────────────────────

def set_code_fingerprints(stamps: Mapping[str, str]) -> None:
    """Record ``{op_key: fingerprint}`` for the ops whose code was just (re)loaded."""
    with _lock:
        for op_key, fp in stamps.items():
            if fp:
                _code_fp[op_key] = fp
            else:
                _code_fp.pop(op_key, None)


def code_fingerprint(op_key: str) -> str:
    """The fingerprint of the code behind ``op_key`` — ``""`` when unstamped, in which
    case the memo key omits it (see :func:`~nodegraph.memo.node_recipe_hash`)."""
    return _code_fp.get(op_key, "")


def clear_code_fingerprints() -> None:
    """Forget every stamp — back to unstamped keys (used by tests)."""
    with _lock:
        _code_fp.clear()


__all__ = ["next_revision", "peek_revision",
           "set_code_fingerprints", "code_fingerprint", "clear_code_fingerprints"]
