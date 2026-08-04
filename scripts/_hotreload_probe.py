"""Offscreen end-to-end probe of live PER-NODE reload (``nodegraph.hotreload`` + the GUI).

    PYTHONUTF8=1 python scripts/_hotreload_probe.py

Boots the real :class:`~nodelab_v2.window.MainWindow` offscreen, then edits real node source
on disk the way a developer would and drives Run → *Reload node code*. Asserts what the
feature promises, and in particular the claim the per-node split exists to make good:

1. a changed compute runs on the next pull, in the already-open window;
2. **only the edited node re-keys** — every other node keeps its cached results, which is the
   whole point of one module per node (before the split, one edit re-keyed all 63);
3. a changed **socket set** appears on the placed card and in the palette, without the card
   being recreated (its wires and position survive);
4. editing a **shared helper** re-keys exactly its users, and re-EXECUTES them — a spec built
   from ``_InRadius(...)`` is a product of that helper, so an alias rebind cannot repair it;
5. editing a **kernel** re-keys only the nodes that import it;
6. a **syntax error** is reported and ignored, with nothing re-keyed and the retry still armed;
7. a node type deleted from the source leaves the palette but not the canvas;
8. the watcher path and the refuse-while-a-pull-is-running path both behave;
9. every file touched is restored byte-for-byte, from a ``finally``.

Asserts + printed checkmarks; exits via ``os._exit`` (offscreen teardown crash gotcha, as
in ``_nodelab_v2_phase5_probe.py``).
"""
from __future__ import annotations

import os
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication      # noqa: E402

import nodegraph.nodes as NN                    # noqa: E402  (the facade — registers all)
from nodegraph import hotreload as H            # noqa: E402
from nodegraph.registry import NODES            # noqa: E402
from nodegraph.revision import code_fingerprint  # noqa: E402

#: the node the probe edits — a one-param pointwise enhancement, cheap and with a socket list
#: short enough that adding one is unambiguous on the card.
OP = "enhance.gamma"
NODE_MOD = "nodegraph.catalog.enhance.gamma"
#: a shared helper module whose users build their SPECS from it (``_InRadius``)
SHARED_MOD = "nodegraph.catalog._shared.kernel_radius"


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _path(mod_name: str) -> str:
    import importlib
    return os.path.abspath(importlib.import_module(mod_name).__file__)


def _read(path: str) -> str:
    # newline="" on BOTH read and write keeps the round-trip byte-exact; the default would
    # collapse this tree's CRLF to LF on read and re-emit LF, rewriting every line ending.
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    from nodelab_v2.window import MainWindow

    targets = [_path(NODE_MOD), _path(SHARED_MOD)]
    backups = {p: p + ".probebak" for p in targets}
    for p, b in backups.items():
        shutil.copy2(p, b)
    before = {p: _read(p) for p in targets}
    try:
        return _run(app, MainWindow)
    except BaseException:                      # noqa: BLE001 — print, then still restore
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # The probe edits real source, so the restore is unconditional and byte-level: a
        # failed assertion must not leave the working tree holding a probe's edit.
        for p, b in backups.items():
            shutil.copy2(b, p)
            os.remove(b)
        H.reload_nodes()                       # leave the process on the restored code
        bad = [p for p in targets if _read(p) != before[p]]
        assert not bad, f"RESTORE FAILED — not as the probe found them: {bad}"
        print("[ok] every edited file restored byte-for-byte")


def _run(app, MainWindow) -> int:
    node_py, shared_py = _path(NODE_MOD), _path(SHARED_MOD)
    pristine_node, pristine_shared = _read(node_py), _read(shared_py)
    nl = "\r\n" if "\r\n" in pristine_node else "\n"

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()

    spec0 = NODES.get(OP)
    assert spec0 is not None, f"{OP} missing from the catalog"
    n_in0, n_ops0 = len(spec0.inputs), len(NODES.keys())
    assert win._watcher is not None, "auto-reload is ticked by default but no watcher started"
    assert NODES.owner(OP) == NODE_MOD, f"{OP} is owned by {NODES.owner(OP)!r}"
    watched = H.watch_paths()
    assert any(p == node_py for p in watched), "the node's own module is not watched"
    _ok(f"window up: {n_ops0} ops; {OP} owned by its own module; {len(watched)} paths watched")

    # ── the headline claim: distinct fingerprints per node ────────────────────
    fp_all0 = {op: code_fingerprint(op) for op in NODES.keys() if code_fingerprint(op)}
    distinct = len(set(fp_all0.values()))
    assert distinct > 50, (f"only {distinct} distinct code fingerprints across "
                           f"{len(fp_all0)} ops — the split did not buy granularity")
    _ok(f"{distinct} distinct code fingerprints across {len(fp_all0)} stamped ops "
        f"(one monolithic module gave exactly 1)")

    # ── place a source + the node under test, and wire them ───────────────────
    src_id = win.doc.add_node("io.load", x=0, y=0).id
    node_id = win.doc.add_node(OP, x=260, y=0).id
    win.doc.connect(src_id, "image", node_id, "data")
    app.processEvents()
    card = win.scene.node_items[node_id]
    pos0 = (card.pos().x(), card.pos().y())
    ports0 = sorted(k[1] for k in card._sockets if k[0] == "in")
    compute0 = NN.COMPUTES[OP]
    _ok(f"placed {OP} as {node_id} with in-ports {ports0}")

    # ── EDIT 1: a node module — new socket + new module body ──────────────────
    edited = pristine_node.replace(
        'inputs=[InDataset(), InFloat("gamma"',
        'inputs=[InDataset(), InFloat("probe_added", "Probe", default=1.0),'
        f'{nl}            InFloat("gamma"', 1)
    assert edited != pristine_node, "socket injection failed"
    edited = edited.replace(
        "def _compute_gamma(", f"def _probe_marker():{nl}"
        f'    """Probe marker — proves this module body re-executed."""{nl}'
        f'    return "reloaded"{nl}{nl}{nl}def _compute_gamma(', 1)
    _write(node_py, edited)

    assert H.pending() == (NODE_MOD,), f"only the edited module should be pending: {H.pending()}"
    _ok("pending() named the ONE edited module")

    assert win.reload_node_code(quiet=True), "reload reported nothing changed"
    app.processEvents()
    spec1 = NODES.get(OP)
    assert len(spec1.inputs) == n_in0 + 1, f"new socket not registered: {spec1.inputs}"
    assert NN.COMPUTES[OP] is not compute0, "COMPUTES still holds the pre-reload compute"
    import importlib
    assert getattr(importlib.import_module(NODE_MOD), "_probe_marker", None) is not None
    _ok(f"reload: {OP} has {len(spec1.inputs)} inputs; compute + module body swapped")

    # THE claim: only this node re-keyed
    fp_all1 = {op: code_fingerprint(op) for op in NODES.keys() if code_fingerprint(op)}
    moved = sorted(op for op in fp_all0 if fp_all1.get(op) != fp_all0[op])
    assert moved == [OP], f"a one-node edit re-keyed {len(moved)} nodes, not 1: {moved[:8]}"
    _ok(f"ONE node edited -> exactly ONE node re-keyed ({OP}); the other "
        f"{len(fp_all0) - 1} keep their cached results")

    # the card must show it, on the same item, same place, wires intact
    assert win.scene.node_items[node_id] is card, "the card was recreated"
    ports1 = sorted(k[1] for k in card._sockets if k[0] == "in")
    assert "probe_added" in ports1, f"new port absent from the card: {ports1}"
    assert (card.pos().x(), card.pos().y()) == pos0, "the card moved"
    assert len(win.doc.edges) == 1, "the wire was dropped"
    assert card.spec is spec1, "the card still points at the stale spec"
    labels = _palette_labels(win)
    assert spec1.label in labels, "palette lost the node"
    _ok("card relayouted in place (new port, same position, wire intact); palette rebuilt")

    _write(node_py, pristine_node)
    assert win.reload_node_code(quiet=True), "revert failed"
    assert code_fingerprint(OP) == fp_all0[OP], "revert did not restore the original key"
    _ok("revert restored the node and its original memo key")

    # Re-baseline HERE, after the revert. Comparing the next edit against `fp_all1` (taken
    # while gamma still carried the probe edit) would count gamma's revert as a change caused
    # by the shared-helper edit — a test artefact that looks exactly like the product bug this
    # step is meant to catch.
    fp_base = {op: code_fingerprint(op) for op in NODES.keys() if code_fingerprint(op)}

    # ── EDIT 2: a SHARED helper — its users must re-key AND re-execute ────────
    users = [op for op in NODES.keys()
             if SHARED_MOD in H.dependency_closure(NODES.owner(op) or "nodegraph.nodes")]
    assert 3 <= len(users) <= 20, f"implausible user set for {SHARED_MOD}: {users}"
    non_users = [op for op in fp_base if op not in users]
    radius_specs = {op: NODES.get(op) for op in users}
    shared_edited = pristine_shared.replace(
        "def _win(", f"def _probe_shared_marker():{nl}"
        f'    """Probe marker."""{nl}    return 1{nl}{nl}{nl}def _win(', 1)
    assert shared_edited != pristine_shared, "shared marker injection failed"
    _write(shared_py, shared_edited)
    rep = H.reload_nodes()
    assert rep.ok, rep.summary()
    # dependencies-first, and the users re-executed rather than merely being re-aliased
    assert SHARED_MOD in rep.reloaded, rep.reloaded
    reloaded_users = [m for m in rep.reloaded if m != SHARED_MOD]
    assert reloaded_users, ("a shared-helper edit reloaded nothing else — the nodes whose "
                            "SPECS were built from it would keep specs made by the old code")
    fp_all2 = {op: code_fingerprint(op) for op in NODES.keys() if code_fingerprint(op)}
    moved2 = {op for op in fp_base if fp_all2.get(op) != fp_base.get(op)}
    assert set(users) <= moved2, f"a helper's users did not all re-key: {set(users) - moved2}"
    untouched = [op for op in non_users if fp_all2.get(op) != fp_base.get(op)]
    assert not untouched, f"a helper edit re-keyed nodes that do not use it: {untouched[:8]}"
    for op in users:
        assert NODES.get(op) is not radius_specs[op], \
            f"{op}'s spec object was not rebuilt, so it is still a product of the old helper"
    _ok(f"shared-helper edit: {len(users)} users re-keyed AND re-executed (specs rebuilt); "
        f"the other {len(non_users)} untouched")

    _write(shared_py, pristine_shared)
    assert H.reload_nodes().ok
    assert code_fingerprint(OP) == fp_all0[OP], "revert of the helper did not restore keys"
    _ok("revert of the shared helper restored every key")

    # ── the inspector's per-node ⟳ button ────────────────────────────────────
    # The reported failure this exists for: "I changed a node's parameters and backend, and
    # reopening the node in the graph did not change it." Re-selecting a node never re-read
    # anything, so there was no gesture that fixed it. This drives the real button, with the
    # watcher deliberately left out of it.
    from PySide6.QtWidgets import QToolButton
    w_node = win.scene.node_items[node_id]
    win.inspector.set_node(w_node)
    app.processEvents()
    btn = [b for b in win.inspector.findChildren(QToolButton) if b.text() == "⟳"]
    assert len(btn) == 1, f"expected exactly one reload button, found {len(btn)}"
    assert btn[0].isEnabled(), "the reload button is disabled for a real catalog node"

    def _rows() -> list:
        from nodelab_v2.inspector import QLabel as _QL
        return [x.text() for x in win.inspector.findChildren(_QL)]

    # the inspector labels a row with the socket NAME, not its display label
    assert "probe_added" not in _rows(), "the probe param is already shown"
    _write(node_py, edited)                        # the same socket-adding edit as EDIT 1
    win._reload_timer.stop()                       # no watcher involvement — the BUTTON only
    btn[0].click()
    app.processEvents()
    assert "probe_added" in _rows(), \
        "the inspector still shows the old parameters after pressing Reload"
    assert "probe_added" in [k[1] for k in w_node._sockets if k[0] == "in"], \
        "the card did not grow the new port"
    assert len(NODES.get(OP).inputs) == n_in0 + 1
    _ok("inspector ⟳ button: edit the .py, press it, the panel and card show the new param")

    # and it is unconditional — pressing it with NOTHING changed still re-registers, which is
    # what makes it a repair rather than another thing that can answer "nothing changed".
    _write(node_py, pristine_node)
    btn[0].click()
    app.processEvents()
    assert len(NODES.get(OP).inputs) == n_in0, "the button did not pick the revert back up"
    spec_a = NODES.get(OP)
    btn[0].click()
    app.processEvents()
    assert NODES.get(OP) is not spec_a, "a second press was a no-op — it must be forced"
    assert code_fingerprint(OP) == fp_all0[OP], "a forced no-op reload moved the memo key"
    _ok("⟳ is unconditional (re-registers even with no file change) yet memo-neutral")

    # ── the palette's ⟳ : the node LIST re-read from disk ────────────────────
    # Distinct from every reload above, which can only refresh modules already imported. A
    # node file created while the window is open has never been imported, so it is in no list
    # and in no palette — and one whose file is deleted keeps offering itself from the palette
    # long after its source is gone. Both are resolved against the filesystem.
    new_py = os.path.join(os.path.dirname(node_py), "brand_new_probe.py")
    NEW_OP = "enhance.brand_new_probe"
    try:
        n_pal0 = len(_palette_labels(win))
        _write(new_py, nl.join([
            '"""Brand New (probe)."""',
            "from __future__ import annotations",
            "from nodegraph.catalog._base import register_node",
            "from nodegraph.registry import Granularity, InDataset, InFloat, OutDataset",
            "",
            "",
            "def _compute_brand_new_probe(ctx):",
            "    return ctx.inputs[0]",
            "",
            "",
            "register_node(",
            f'    _compute_brand_new_probe, op_key="{NEW_OP}", label="Brand New",',
            '    category="enhancement",',
            '    inputs=[InDataset(), InFloat("amount", "Amount", default=1.0,',
            '                                description="How much. A probe node.")],',
            "    outputs=[OutDataset()], granularity=Granularity.TILEABLE,",
            '    description="A probe node created while the window was open.",',
            ")", ""]))
        assert win.refresh_node_list(), "refresh reported nothing"
        app.processEvents()
        assert NODES.get(NEW_OP) is not None, "a new node file was not picked up"
        assert "Brand New" in _palette_labels(win), "the new node is missing from the palette"
        assert len(_palette_labels(win)) == n_pal0 + 1
        # It must be a FULL citizen, not just a palette entry: fingerprinted (or editing it
        # live would serve stale memo results) and placeable with its declared sockets.
        assert code_fingerprint(NEW_OP), "the new node is un-fingerprinted"
        new_id = win.doc.add_node(NEW_OP, x=520, y=0).id
        app.processEvents()
        assert sorted(k[1] for k in win.scene.node_items[new_id]._sockets
                      if k[0] == "in") == ["amount", "data"]
        _ok("palette ⟳: a node file ADDED at runtime is registered, fingerprinted, listed "
            "and placeable — no restart")

        # a broken new file is reported, not half-registered
        _write(new_py, "def oops(:\n")
        rep = H.refresh_catalog()
        assert not rep.ok and "brand_new_probe" in rep.error, rep
        assert NODES.get(NEW_OP) is not None, "a broken EDIT dropped the working registration"
        _ok(f"a new file that does not compile is reported, not half-registered")

        os.remove(new_py)
        win.doc.remove_node(new_id)
        assert win.refresh_node_list()
        app.processEvents()
        assert NODES.get(NEW_OP) is None, "a deleted node file stayed registered"
        assert NEW_OP not in NN.COMPUTES, "a deleted node file kept its compute"
        assert "Brand New" not in _palette_labels(win), "the palette kept the deleted node"
        assert len(_palette_labels(win)) == n_pal0
        _ok("palette ⟳: a node file DELETED at runtime leaves the registry and the palette")
    finally:
        if os.path.exists(new_py):
            os.remove(new_py)
        H.refresh_catalog()

    # ── kernel granularity, statically ───────────────────────────────────────
    # A kernel is imported INSIDE its compute's body, so an edit-and-reload test would first
    # have to make the node run (numba/tensorflow/torch, model weights). The property that
    # matters is visible without running anything: the AST closure that feeds the fingerprint
    # must contain a kernel for the nodes that call it and only those. This is the check that
    # would have failed had the closure been built from RUNTIME imports, which cannot see a
    # function-body import until after the node has been executed once.
    kern = "nodegraph.kernels.dic_correlate"
    users = [op for op in NODES.keys()
             if kern in H.dependency_closure(NODES.owner(op) or "")]
    assert users, f"no node's closure contains {kern} — kernel deps are invisible"
    assert OP not in users, f"{OP} must not depend on {kern}"
    assert len(users) <= 4, f"{kern} reaches too many nodes to be per-node: {users}"
    _ok(f"kernel granularity: {kern.split('.')[-1]} is in the closure of {users} "
        f"and no other node — so editing it re-keys only those")

    # ── EDIT 3: a syntax error is reported and ignored ────────────────────────
    live_spec, live_compute = NODES.get(OP), NN.COMPUTES[OP]
    live_fp = dict((op, code_fingerprint(op)) for op in NODES.keys())
    _write(node_py, pristine_node + nl + "def nope(:" + nl)
    rep = H.reload_nodes()
    assert not rep.ok and not rep.broken, f"syntax error mishandled: {rep}"
    assert "gamma.py" in rep.error, rep.error
    assert NODES.get(OP) is live_spec, "a rejected reload changed the catalog"
    assert NN.COMPUTES[OP] is live_compute, "a rejected reload swapped a compute"
    assert len(NODES.keys()) == n_ops0, "a rejected reload changed the op count"
    assert all(code_fingerprint(op) == live_fp[op] for op in live_fp), \
        "a rejected reload re-keyed the memo"
    assert H.pending() == (NODE_MOD,), "a rejected reload consumed the baseline"
    _ok(f"syntax error rejected safely, nothing re-keyed, retry still armed "
        f"({rep.error.split(os.sep)[-1][:44]}…)")

    # ── EDIT 4: a deleted node type leaves the palette, not the canvas ───────
    gone = pristine_node.replace(f'op_key="{OP}"', 'op_key="enhance.gammaRenamed"', 1)
    assert gone != pristine_node, "rename injection failed"
    _write(node_py, gone)
    rep = H.reload_nodes()
    assert rep.ok, rep.summary()
    assert OP in rep.removed and "enhance.gammaRenamed" in rep.added, (rep.removed, rep.added)
    assert NODES.get(OP) is None and OP not in NN.COMPUTES, "deleted op survived"
    orphaned = win.scene.resync_specs()
    assert node_id in orphaned, "the orphaned card was not reported"
    assert node_id in win.scene.node_items, "the orphaned card was deleted with its wires"
    assert len(win.doc.edges) == 1, "the orphan's wire was dropped"
    _ok("deleted op left the registry/palette; its placed card kept its wires")

    # ── the watcher + the refuse-while-running paths ─────────────────────────
    _write(node_py, pristine_node)
    win._autoreload_tick()
    app.processEvents()
    assert not H.pending(), "the watcher tick left the edit unloaded"
    assert NODES.get(OP) is not None, "the watcher tick did not restore the node"
    fp_quiet = code_fingerprint(OP)
    win._autoreload_tick()
    assert code_fingerprint(OP) == fp_quiet, "an idle watcher tick re-keyed the memo"
    _ok("watcher tick reloaded the save on its own; an idle tick changes nothing")

    _write(node_py, pristine_node.replace('label="Gamma"', 'label="Gamma "', 1))
    win.runner._busy = True
    try:
        assert win.reload_node_code(quiet=True) is False, "reloaded during a pull"
        assert NODES.get(OP).label == "Gamma", "the refusal did not hold the reload back"
        assert win._reload_timer.isActive(), "the deferred reload was dropped"
    finally:
        win.runner._busy = False
    assert win.reload_node_code(quiet=True), "the deferred reload did not apply once idle"
    assert NODES.get(OP).label == "Gamma ", "the deferred reload did not take effect"
    _ok("reload refused mid-pull, rescheduled, applied once the pull finished")

    _write(node_py, pristine_node)
    assert H.reload_nodes().ok
    assert not win.scene.resync_specs(), "the card did not recover"
    assert len(NODES.keys()) == n_ops0, f"op count drifted: {len(NODES.keys())} vs {n_ops0}"
    assert code_fingerprint(OP) == fp_all0[OP]
    _ok("final revert: catalog, card and memo keys all back to the starting state")

    print("\nAll live per-node reload probes passed.")
    return 0


def _palette_labels(win) -> list:
    tree = win.palette._tree
    out = []
    for i in range(tree.topLevelItemCount()):
        head = tree.topLevelItem(i)
        out += [head.child(j).text(0) for j in range(head.childCount())]
    return out


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    finally:
        sys.stdout.flush()
        os._exit(code)
