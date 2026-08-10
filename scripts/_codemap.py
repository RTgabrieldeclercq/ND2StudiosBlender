"""Generate and gate the agent codemap — the grep-first index under ``codemap/``.

    python scripts/_codemap.py                 # check (THE DEFAULT)
    python scripts/_codemap.py check
    python scripts/_codemap.py write           # regenerate codemap/gen/ + STATE.md
    python scripts/_codemap.py write --verified 2026-08-05   # ALSO record a green gate run
    python scripts/_codemap.py bless WF-02 INV-07            # re-pin named curated entries
    python scripts/_codemap.py bless --all                   # only for the first cut

**The default is ``check``, and that is deliberate.** Its sibling
``scripts/_catalog_snapshot.py`` defaulted to ``save``, so running it bare silently overwrote
the very baseline it existed to defend and then reported success. That footgun cost a real
regression window and is fixed at source now; this tool is built so it never existed: the
read path is the default, and the write path has to be asked for by name.

The gate that actually runs is ``nodegraph.selftest.test_codemap`` — the same check, wired
into the suite everyone already runs, and structurally unable to re-bless because it imports
no write path at all. This script is how you FIX a red gate, not how you notice one.

``write`` never touches ``codemap/curated.lock.json``. Refreshing a pin means a human or an
agent re-read that entry and decided the prose is still true; if a bulk regeneration did it
silently, the whole pinning mechanism would become a rubber stamp.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodegraph import codemap as CM       # noqa: E402


def _with_gui() -> bool:
    """Register the three GUI-layer ops (``io.load``, ``view.viewer``, ``io.dock``).

    They are real, wireable nodes that a graph cannot start without, so a map that omitted
    them would send an agent hunting for ``io.load`` in a catalog that has never contained it.
    ``nodegraph`` may not import Qt, so the GUI package is pulled in HERE, by the caller, and
    ``codemap.build()`` simply sees whatever is registered."""
    try:
        import nodelab_v2.ops as OPS
        OPS.ensure_ops()
        return True
    except Exception as exc:                                    # pragma: no cover
        print(f"! PySide6/nodelab_v2 unavailable ({type(exc).__name__}) — the 3 GUI-layer ops "
              f"are omitted from this run; their records are skipped, not treated as removed.")
        return False


def _strip_gui(recs):
    """Drop GUI-op records so a Qt-less run compares like with like instead of reporting
    three phantom deletions."""
    gui = {r["op"] for r in recs.get("nodes", []) if r.get("gui_only")}
    return {
        kind: [r for r in rows
               if not (kind == "nodes" and r.get("op") in gui)
               and not (kind == "sockets" and r.get("op") in gui)]
        for kind, rows in recs.items()
    }


def cmd_check(gui: bool) -> int:
    built = CM.build()
    live = built["records"]
    committed = CM.read_committed()
    if not gui:
        committed = _strip_gui(committed)
    diffs = CM.compare(committed, live, strict=False)
    cosmetic = len(CM.compare(committed, live)) - len(diffs)
    stale = CM.check_curated(built)

    if not any(committed.values()):
        print("CODEMAP MISSING — codemap/gen/ has no records. "
              "Run: python scripts/_codemap.py write")
        return 1
    if not diffs and not stale:
        m = built["manifest"]
        print(f"CODEMAP CURRENT — {m['n_catalog_ops']} catalog ops, "
              f"{m['counts']['sockets']} sockets, {m['counts']['modules']} modules, "
              f"{m['counts']['symbols']} symbols; fp {m['fp_all']}")
        if cosmetic:
            # Reported, never silent: the map is right about every contract but its line
            # numbers have moved, and an agent that Reads at a stale offset lands in the
            # wrong function. Not a failure, because blocking on it would train bypass.
            print(f"  ({cosmetic} cosmetic drift(s) — line numbers / LOC / closure "
                  f"fingerprints. Not a gate failure; `write` refreshes them.)")
        return 0
    if diffs:
        print(f"CODEMAP STALE — {len(diffs)} difference(s) between the code and codemap/gen/:")
        for d in diffs[:200]:
            print(f"  {d}")
        if len(diffs) > 200:
            print(f"  … and {len(diffs) - 200} more")
        print("\n  Fix: python scripts/_codemap.py write     "
              "(then READ `git diff codemap/` — an unexpected line there is a finding)")
    if stale:
        print(f"\nCURATED PROSE UNVERIFIED — {len(stale)} entr(y/ies) need re-reading:")
        for s in stale:
            print(f"  {s}")
    return 1


def cmd_write(gui: bool, verified: str) -> int:
    built = CM.build()
    files = CM.render(built)
    files["codemap/STATE.md"] = CM.state_markdown(built, verified or CM.current_verified())
    os.makedirs(CM.GEN_DIR, exist_ok=True)
    changed = []
    for rel, text in sorted(files.items()):
        path = os.path.join(CM.ROOT, rel.replace("/", os.sep))
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as fh:
                old = fh.read()
        if old != text:
            changed.append(rel)
        # newline="" so Python does not translate \n to \r\n on Windows: the committed bytes
        # must be identical on every platform or the gate fails for the wrong reason.
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    m = built["manifest"]
    print(f"wrote {len(files)} files — {m['n_catalog_ops']} catalog ops, "
          f"{m['counts']['sockets']} sockets, {m['counts']['modules']} modules, "
          f"{m['counts']['symbols']} symbols, {m['counts']['imports']} import edges")
    print(f"changed: {', '.join(changed) if changed else '(nothing — map was already current)'}")
    if not gui:
        print("! the 3 GUI-layer ops were NOT registered in this run — the map you just wrote "
              "is incomplete. Re-run where PySide6 imports.")
        return 1
    return 0


def cmd_bless(ids, all_: bool, today: str) -> int:
    built = CM.build()
    entries = CM.parse_curated()
    lock = CM.read_lock()
    if all_:
        ids = sorted(entries)
    if not ids:
        print("bless needs entry ids (e.g. WF-02 INV-07) or --all")
        return 2
    for eid in ids:
        if eid not in entries:
            print(f"! no curated entry called {eid}")
            return 1
        anchors = {}
        for a in entries[eid]["anchors"]:
            h = CM.anchor_hash(a, built)
            if h is None:
                print(f"! {eid}: anchor {a} does not resolve — fix the anchor before blessing")
                return 1
            anchors[a] = h
        lock[eid] = {"anchors": anchors, "file": entries[eid]["file"], "blessed": today}
        print(f"blessed {eid} ({len(anchors)} anchor(s))")
    for eid in sorted(set(lock) - set(entries)):
        del lock[eid]
        print(f"dropped {eid} (entry no longer exists)")
    with open(CM.LOCK_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(CM.render_lock(lock))
    return 0


def main(argv) -> int:
    args = list(argv)
    mode = "check"
    if args and not args[0].startswith("-"):
        mode = args.pop(0)
    verified = ""
    if "--verified" in args:
        i = args.index("--verified")
        verified = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    all_ = "--all" in args
    if all_:
        args.remove("--all")

    if mode in ("check", "write"):
        gui = _with_gui()
        return cmd_check(gui) if mode == "check" else cmd_write(gui, verified)
    if mode == "bless":
        _with_gui()
        if not verified:
            print("bless needs --verified YYYY-MM-DD (the date you re-read the entry); "
                  "it is recorded in the lockfile so a future reader knows how old the "
                  "judgement is")
            return 2
        return cmd_bless(args, all_, verified)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
