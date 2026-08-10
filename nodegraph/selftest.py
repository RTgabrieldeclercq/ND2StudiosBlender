"""Self-test for the nodegraph v2 core (headless, Qt-free).

Run:  python -m nodegraph.selftest

Mirrors the project's ``scripts/_pipeline_graph_selftest.py`` style (asserts +
printed checkmarks, exit 0 on success) so it needs no pytest.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from nodegraph.domains import (
    Domain, axes_of, comparable, dropped_axes, is_finer, is_lattice,
    is_structure, join, meet,
)
from nodegraph.dataset import (
    AttributeLayer, AxisSizes, CALIBRATION_KEYS, Dataset,
)
from nodegraph.revision import next_revision, peek_revision
from nodegraph.reducers import (
    TILEABLE_REDUCERS, is_tileable, reduce, tree_reduce,
)
from nodegraph.transfer import (
    BridgeStep, LatticeStep, execute_transfer, lattice_transfer, plan_transfer,
)
from nodegraph.sockets import (
    Direction, Socket, SocketType, can_connect, can_convert,
)
from nodegraph.registry import (
    NODES, DIM_MODE, DimMode, Granularity, InDataset, InFloat, InVector, Mode,
    OutDataset, define_node,
)
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.metadata import (
    MetaEnvelope, envelope_symbols, eval_derive, propagate_meta, resample,
    resolve_dim_default,
    stitch, z_project,
)
from nodegraph.provider import B2ndProvider, SyntheticProvider
from nodegraph.memo import Memo, node_recipe_hash
from nodegraph.engine import Engine
from nodegraph.structure import (
    connectivity_offsets, label_components, point_table, seeded_watershed,
)
from nodegraph.bridges import (
    containing_label, label_to_voxel, point_to_voxel, points_in_label,
    voxel_to_label, voxel_to_point,
)
from nodegraph.field import (
    Attr, BinOp, Const, FieldCache, FieldContext, Input, VirtualArray, Where,
    evaluate, field_expr_hash, field_key,
)

try:
    import blosc2 as _blosc2
    _HAVE_BLOSC2, _BLOSC2_VER = True, _blosc2.__version__
except Exception:  # noqa: BLE001
    _HAVE_BLOSC2, _BLOSC2_VER = False, ""

try:
    import pyarrow as _pa
    _HAVE_PYARROW, _PA_VER = True, _pa.__version__
except Exception:  # noqa: BLE001
    _HAVE_PYARROW, _PA_VER = False, ""

try:
    import skimage as _skimage  # noqa: F401
    _HAVE_SKIMAGE = True
except Exception:  # noqa: BLE001
    _HAVE_SKIMAGE = False

try:
    import scipy as _scipy      # noqa: F401
    _HAVE_WATERSHED = _HAVE_SKIMAGE
except Exception:  # noqa: BLE001
    _HAVE_WATERSHED = False

D = Domain
AX = AxisSizes(m=2, t=3, z=4, c=2, y=5, x=6)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _stream_eq(a, b, *, equal_nan: bool = False) -> bool:
    """Compare a **streamed** float result against an **eager float64** reference.

    Exact under the default float64 streaming dtype — the V2.04 promise that a tiled /
    per-unit result is byte-identical to the whole-array one is the whole point of those
    tests and must not be softened. Under the opt-in ``NODEGRAPH_FLOAT32=1`` path
    (:func:`nodegraph.parallel.stream_dtype`) exactness is arithmetically impossible: the
    streamed side carries ~7 significant digits and the reference ~16. There the check
    becomes a float32-epsilon tolerance, which still catches every real defect these
    assertions exist for (a wrong halo, a stale tile, a mis-cropped window, a reducer
    disagreeing with numpy) because those are order-of-magnitude wrong, not 1e-7 wrong."""
    import numpy as _np
    from nodegraph.streaming import _F
    if _np.dtype(_F) == _np.float64:
        return bool(_np.array_equal(a, b, equal_nan=equal_nan))
    return bool(_np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=equal_nan))


def _stream_rtol(exact_rtol: float = 1e-12) -> float:
    """``exact_rtol`` under float64 streaming, a float32-epsilon tolerance under the
    ``NODEGRAPH_FLOAT32=1`` opt-in. For assertions that already use a tight ``allclose``
    against an eager float64 reference (the reducer ≡ numpy checks)."""
    import numpy as _np
    from nodegraph.streaming import _F
    return exact_rtol if _np.dtype(_F) == _np.float64 else 1e-5


# ── domains + lattice order ──────────────────────────────────────────────────

def test_domains() -> None:
    assert is_lattice(D.VOXEL) and is_lattice(D.GLOBAL)
    assert is_lattice(D.CHANNEL) and axes_of(D.CHANNEL) == frozenset({"c"})  # V2.01 §H
    assert not is_lattice(D.LABEL) and is_structure(D.TRACK)
    assert axes_of(D.VOXEL) == frozenset({"m", "t", "z", "c", "y", "x"})
    assert axes_of(D.FRAME) == frozenset({"m", "t"})
    assert axes_of(D.GLOBAL) == frozenset()
    # order
    assert is_finer(D.VOXEL, D.FRAME) and not is_finer(D.FRAME, D.VOXEL)
    assert is_finer(D.VOXEL, D.CHANNEL)                   # Channel ⊆ Voxel
    assert comparable(D.VOXEL, D.GLOBAL)
    assert not comparable(D.MULTIPOINT, D.TIMEPOINT)      # the two branches
    assert not comparable(D.CHANNEL, D.FRAME)             # Channel is orthogonal
    # lattice: M∪T = Frame, M∩T = Global; meet total, join partial across c
    assert join(D.MULTIPOINT, D.TIMEPOINT) is D.FRAME
    assert meet(D.MULTIPOINT, D.TIMEPOINT) is D.GLOBAL
    assert join(D.PLANE, D.TIMEPOINT) is D.PLANE
    assert meet(D.CHANNEL, D.VOXEL) is D.CHANNEL
    assert join(D.CHANNEL, D.FRAME) is None               # {m,t,c} unnamed → partial
    assert dropped_axes(D.VOXEL, D.FRAME) == frozenset({"z", "c", "y", "x"})
    _ok("domains: 10 domains (7 lattice incl. Channel), order, meet total / join partial")


# ── reducers ─────────────────────────────────────────────────────────────────

def test_reducers() -> None:
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert np.allclose(reduce(a, (2,), "mean"), a.mean(axis=2))
    assert np.allclose(reduce(a, (0, 2), "sum"), a.sum(axis=(0, 2)))
    assert np.allclose(reduce(a, (1,), "max"), a.max(axis=1))
    assert np.array_equal(reduce(a, (0, 1, 2), "count"), np.int64(24))
    assert np.allclose(reduce(a, (2,), "first"), a[:, :, 0])
    _ok("reducers: mean/sum/max/count/first over axes")


# ── partial reducers (tree-reduce across tiles) ──────────────────────────────

def test_partial_reducers() -> None:
    a = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)   # keep axis0; reduce 1,2
    axis = (1, 2)

    def tiles_of(arr):
        # partition the reduced axes into ragged chunks, in canonical scan order
        out = []
        for i0, i1 in ((0, 2), (2, 3)):          # axis-1 chunks
            for j0, j1 in ((0, 1), (1, 4)):      # axis-2 chunks
                out.append(arr[:, i0:i1, j0:j1])
        return out

    tiles = tiles_of(a)
    # tiled tree-reduce equals the whole-array reduce
    for name, whole in (("mean", a.mean(axis=axis)), ("sum", a.sum(axis=axis)),
                        ("max", a.max(axis=axis)), ("min", a.min(axis=axis))):
        assert np.allclose(tree_reduce(tiles, axis, name), whole), name
    assert np.array_equal(tree_reduce(tiles, axis, "count"), np.full((2,), 12))
    # first: leftmost tile (global reduced-index 0) wins under canonical order
    assert np.allclose(tree_reduce(tiles, axis, "first"), a[:, 0, 0])
    # NaN-aware: mixed nan/non-nan tiles fold to nanmean
    b = a.copy(); b[0, 0, 0] = np.nan
    assert np.allclose(tree_reduce(tiles_of(b), axis, "mean"),
                       np.nanmean(b, axis=axis), equal_nan=True)
    # median is not a monoid → not tileable → callers realize the whole domain
    assert not is_tileable("median")
    try:
        tree_reduce(tiles, axis, "median")
        raise AssertionError("expected median to be non-tileable")
    except ValueError:
        pass
    assert TILEABLE_REDUCERS == {"mean", "sum", "max", "min", "count", "first"}
    _ok("partial reducers: tiled tree-reduce == whole; NaN-aware; median falls back")


# ── revision counter + safe immutability (V2.02 §4) ──────────────────────────

def test_revision_and_immutability() -> None:
    r1 = next_revision()
    r2 = next_revision()
    assert r2 > r1                                   # strictly monotonic
    p = peek_revision()
    assert peek_revision() == p and p >= r2          # peek doesn't consume

    # a fresh layer freezes its values and carries a revision
    src = np.ones((2, 3))                            # writeable producer buffer
    lay = AttributeLayer(D.FRAME, "a", src)
    assert lay.revision > 0
    assert not lay.values.flags.writeable            # frozen read-only
    assert lay.values is not src and src.flags.writeable  # producer NOT aliased/frozen
    try:
        lay.values[0, 0] = 5.0
        raise AssertionError("expected read-only array")
    except ValueError:
        pass

    # an already read-only input is adopted without a needless copy
    ro = np.ones((2, 3)); ro.flags.writeable = False
    assert AttributeLayer(D.FRAME, "b", ro).values is ro

    # mutate → same key, new values, fresh (greater) revision
    lay2 = lay.mutate(np.zeros((2, 3)))
    assert lay2.key == lay.key and lay2.revision > lay.revision
    assert np.array_equal(lay2.values, np.zeros((2, 3)))
    assert not lay2.values.flags.writeable
    _ok("revision + immutability: monotonic, frozen values, no aliasing, mutate re-stamps")


# ── dataset / attribute layers ───────────────────────────────────────────────

def test_dataset() -> None:
    assert AX.shape_for(D.VOXEL) == (2, 3, 4, 2, 5, 6)   # (m,t,z,c,y,x)
    assert AX.shape_for(D.FRAME) == (2, 3)
    assert AX.shape_for(D.CHANNEL) == (2,)               # (c,)
    assert AX.shape_for(D.GLOBAL) == ()
    ds = Dataset(axes=AX, metadata={"pixel_size_um": 1.7})
    frame = AttributeLayer(D.FRAME, "count", np.ones((2, 3)))
    ds2 = ds.with_attribute(frame)
    assert ds.get(D.FRAME, "count") is None          # structural sharing: original untouched
    assert ds2.get(D.FRAME, "count").values.shape == (2, 3)
    # shape validation
    try:
        ds.with_attribute(AttributeLayer(D.FRAME, "bad", np.ones((2, 2))))
        raise AssertionError("expected shape error")
    except ValueError:
        pass
    # §7b: with_structure preserves each table's z_kind into __struct_zkind__ provenance
    # (keyed by domain+layer), readable via structure_zkind — the generic dimensionality
    # inheritance mechanism. Two layers on the same domain keep independent z_kinds.
    from nodegraph.structure import StructureTable as _ST
    p2 = _ST(D.POINT, {"id": np.arange(1), "y": np.zeros(1), "x": np.zeros(1)},
             layer="flat", z_kind="plane_index")
    p3 = _ST(D.POINT, {"id": np.arange(1), "z": np.zeros(1), "y": np.zeros(1),
                       "x": np.zeros(1)}, layer="vol", z_kind="subpixel")
    dss = ds.with_structure(p2).with_structure(p3)
    assert dss.structure_zkind(D.POINT, "flat") == "plane_index"
    assert dss.structure_zkind(D.POINT, "vol") == "subpixel"
    assert dss.structure_zkind(D.POINT, "missing") is None
    assert ds.structure_zkind(D.POINT, "flat") is None      # original untouched (COW)
    _ok("dataset: shape_for, structural sharing, shape validation, z_kind provenance (§7b)")


# ── transfer plans (generation + routing) ────────────────────────────────────

def test_plans() -> None:
    # comparable lattice → single reduce / broadcast
    p = plan_transfer(D.VOXEL, D.FRAME)
    assert p.generated and len(p.steps) == 1
    assert isinstance(p.steps[0], LatticeStep) and p.steps[0].kind == "reduce"
    assert p.steps[0].axes == frozenset({"z", "c", "y", "x"})   # now reduces c too
    assert plan_transfer(D.FRAME, D.VOXEL).steps[0].kind == "broadcast"
    assert plan_transfer(D.FRAME, D.FRAME).steps[0].kind == "identity"

    # Channel is a lattice domain now → Voxel↔Channel are GENERATED (V2.01 §H)
    pc = plan_transfer(D.VOXEL, D.CHANNEL)
    assert pc.generated and pc.steps[0].kind == "reduce"
    assert pc.steps[0].axes == frozenset({"m", "t", "z", "y", "x"})
    assert plan_transfer(D.CHANNEL, D.VOXEL).steps[0].kind == "broadcast"

    # incomparable lattice → reduce then broadcast, still generated
    mt = plan_transfer(D.MULTIPOINT, D.TIMEPOINT)
    assert mt.generated and [s.kind for s in mt.steps] == ["reduce", "broadcast"]

    # routing through bridges (not generated)
    vt = plan_transfer(D.VOXEL, D.TRACK)
    assert not vt.generated
    assert vt.steps[-1].dst is D.TRACK and isinstance(vt.steps[-1], BridgeStep)
    lm = plan_transfer(D.LABEL, D.MULTIPOINT)      # Label→Frame→Multipoint
    assert not lm.generated and lm.dst is D.MULTIPOINT
    tm = plan_transfer(D.TRACK, D.MULTIPOINT)      # Track→Timepoint→Multipoint
    assert not tm.generated and any(isinstance(s, BridgeStep) for s in tm.steps)
    assert plan_transfer(D.VOXEL, D.LABEL).describe()  # doesn't crash
    _ok("transfer plans: lattice generation + bridge routing")


# ── transfer execution (lattice, real numpy) ─────────────────────────────────

def test_execution() -> None:
    vox = AttributeLayer(D.VOXEL, "intensity",
                         np.arange(np.prod(AX.shape_for(D.VOXEL)), dtype=float)
                         .reshape(AX.shape_for(D.VOXEL)))
    # Voxel(m,t,z,c,y,x) → Frame (mean over z,c,y,x = positions 2,3,4,5)
    fr = lattice_transfer(vox, D.FRAME, AX)
    assert fr.domain is D.FRAME and fr.values.shape == (2, 3)
    assert np.allclose(fr.values, vox.values.mean(axis=(2, 3, 4, 5)))
    # Voxel → Channel (mean over m,t,z,y,x = positions 0,1,2,4,5) — the c-axis fix
    ch = lattice_transfer(vox, D.CHANNEL, AX)
    assert ch.domain is D.CHANNEL and ch.values.shape == (2,)
    assert np.allclose(ch.values, vox.values.mean(axis=(0, 1, 2, 4, 5)))
    # Voxel → Global (scalar mean)
    g = lattice_transfer(vox, D.GLOBAL, AX)
    assert g.values.shape == () and np.allclose(g.values, vox.values.mean())
    # Frame → Voxel (broadcast) then back → identity for mean
    back = lattice_transfer(fr, D.VOXEL, AX)
    assert back.values.shape == AX.shape_for(D.VOXEL)
    assert np.allclose(lattice_transfer(back, D.FRAME, AX).values, fr.values)

    # multi-axis broadcast insertion: Timepoint → Voxel (now 6-D target)
    tp = AttributeLayer(D.TIMEPOINT, "elapsed", np.array([10.0, 20.0, 30.0]))
    tv = lattice_transfer(tp, D.VOXEL, AX)
    assert tv.values.shape == AX.shape_for(D.VOXEL)
    for t in range(AX.t):
        assert np.allclose(tv.values[:, t], tp.values[t])   # select t=axis1 → all == tp[t]

    # incomparable execution: Multipoint → Timepoint = mean over m, const over t
    mp = AttributeLayer(D.MULTIPOINT, "qc", np.array([4.0, 8.0]))
    tt = lattice_transfer(mp, D.TIMEPOINT, AX)
    assert tt.values.shape == (3,) and np.allclose(tt.values, mp.values.mean())

    # reducer variants
    assert np.allclose(lattice_transfer(vox, D.FRAME, AX, "sum").values,
                       vox.values.sum(axis=(2, 3, 4, 5)))
    assert np.allclose(lattice_transfer(vox, D.PLANE, AX, "max").values,
                       vox.values.max(axis=(3, 4, 5)))   # Plane{m,t,z}: reduce c,y,x

    # a bridge plan refuses to execute (Phase 3)
    try:
        execute_transfer(vox, plan_transfer(D.VOXEL, D.LABEL), AX)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    _ok("transfer execution: reduce/broadcast/round-trip/incomparable + bridge guard")


# ── sockets ──────────────────────────────────────────────────────────────────

def test_sockets() -> None:
    assert can_convert(SocketType.INT, SocketType.FLOAT)
    assert can_convert(SocketType.FLOAT, SocketType.VECTOR)
    assert not can_convert(SocketType.STRING, SocketType.FLOAT)
    out_f = Socket("o", SocketType.FLOAT, Direction.OUT)
    in_i = Socket("i", SocketType.INT, Direction.IN)
    in_ds = Socket("d", SocketType.DATASET, Direction.IN)
    out_ds = Socket("d", SocketType.DATASET, Direction.OUT)
    assert can_connect(out_f, in_i)                     # Float→Int implicit
    assert can_connect(out_ds, in_ds)                   # Dataset↔Dataset
    assert not can_connect(out_f, in_ds)                # value↮dataset
    assert not can_connect(in_i, out_f)                 # wrong direction
    _ok("sockets: conversion table + can_connect")


# ── registry / node API ──────────────────────────────────────────────────────

def test_registry() -> None:
    # NB: a clearly-fake op_key — fixtures must NOT squat on a real node's op_key or
    # they clobber it in the global registry (the real ``detect.spots`` node, V2.00
    # §14 / nodes.py, previously collided with this fixture).
    spec = define_node(
        "test.registry_demo", "Detect Spots", category="detection",
        inputs=[InDataset(),
                InFloat("radius", "Radius", unit="um",
                        derive="0.61*(emission_nm or 520)/(na or 1.4)/1000")],
        outputs=[OutDataset()],
        modes=[Mode("polarity", ["Bright", "Dark"])],
    )
    assert NODES.get("test.registry_demo") is spec
    r = spec.input("radius")
    assert r.unit == "um" and r.is_field and r.instantiate().type is SocketType.FLOAT
    assert spec.modes[0].resolved_default() == "Bright"
    _ok("registry: define_node, unit/derive sockets, modes")


# ── live node reload: the two invariants it rests on (nodegraph.hotreload) ────

def test_live_reload_contract() -> None:
    """The reload machinery's load-bearing pieces, without touching a real source file.

    Two invariants, because everything else in :mod:`nodegraph.hotreload` is built on them:

    1. **Provenance is per-defining-module.** A reload deletes the ops its module stopped
       defining, and it identifies them by owner + registration sequence. If the owner were
       attributed to the shared ``register_node`` helper instead of the calling module, the
       first reload of the catalog would delete every GUI-layer op (``io.load``,
       ``view.viewer``, ``io.dock``) along with it.
    2. **An unstamped memo key is byte-identical to a pre-fingerprint one.** The code
       fingerprint only enters ``node_recipe_hash`` once something has actually been
       reloaded, which is what lets a headless/batch/selftest run keep the keys — and so the
       memo hits — it has always had.
    """
    from nodegraph.hotreload import is_catalog_op, node_modules
    from nodegraph.memo import node_recipe_hash
    from nodegraph.revision import code_fingerprint, set_code_fingerprints

    # (1) provenance: this fixture is owned by the SELFTEST, not by whichever module hosts
    # the ``register_node`` wrapper it registers through.
    from nodegraph.nodes import register_node as _reg
    _reg(lambda ctx: ctx.inputs[0], op_key="test.reload_owned", label="Owned",
         inputs=[InDataset()], outputs=[OutDataset()])
    # ``__name__``, not the literal: this file is both ``nodegraph.selftest`` (imported) and
    # ``__main__`` (run with -m), and the point being pinned is that the owner is THIS
    # module either way rather than the wrapper it called.
    assert NODES.owner("test.reload_owned") == __name__, NODES.owner("test.reload_owned")
    mods = node_modules()
    # The PROPERTY, not a module literal: where a given node's definition lives is exactly
    # what the per-node split changes, so asserting "gamma is owned by nodegraph.nodes" would
    # have to be edited every time a node moves — and an assertion that has to be edited to
    # keep passing is not testing anything. What must hold is that a shipped node is owned by
    # SOME node module and a fixture is not.
    assert NODES.owner("enhance.gamma") in mods, NODES.owner("enhance.gamma")
    assert is_catalog_op("enhance.gamma"), "a shipped node must read as a catalog op"
    assert not is_catalog_op("test.reload_owned"), "a fixture must not read as a catalog op"
    assert not is_catalog_op("view.viewer"), "a GUI-layer op must not read as a catalog op"
    # a catalog reload sweeps by owner+sequence; this fixture must never be swept
    mark = NODES.mark()
    assert "test.reload_owned" not in NODES.stale_owned(mods, mark)
    assert "enhance.gamma" in NODES.stale_owned(mods, mark), \
        "an op registered BEFORE the mark must count as not-yet-re-registered"

    # snapshot/restore is the rollback a failing reload depends on
    snap = NODES.snapshot()
    n_before = len(NODES.keys())
    NODES.remove("test.reload_owned")
    assert NODES.get("test.reload_owned") is None
    NODES.restore(snap)
    assert NODES.get("test.reload_owned") is not None and len(NODES.keys()) == n_before
    NODES.remove("test.reload_owned")                      # leave the registry as found

    # (2) the memo key is untouched while unstamped, and moves once stamped
    args = ("enhance.gamma", {"gamma": 0.5}, ("upstream",), (7,))
    assert code_fingerprint("test.reload_fp") == ""
    bare = node_recipe_hash(*args)
    assert node_recipe_hash(*args, "") == bare, \
        "an empty fingerprint must be OMITTED from the digest, not hashed as ''"
    assert node_recipe_hash(*args, "abc") != bare, "a stamped op must re-key"
    assert node_recipe_hash(*args, "abc") == node_recipe_hash(*args, "abc")
    assert node_recipe_hash(*args, "def") != node_recipe_hash(*args, "abc")
    # a stamp is reversible: reverting the source restores the ORIGINAL key, which is what
    # makes "undo my edit" return the results of the run before it rather than recompute.
    set_code_fingerprints({"test.reload_fp": "v1"})
    assert code_fingerprint("test.reload_fp") == "v1"
    set_code_fingerprints({"test.reload_fp": ""})           # "" clears
    assert code_fingerprint("test.reload_fp") == ""
    _ok("live reload: ops attributed to their DEFINING module (not the shared "
        "register_node helper) so a catalog reload cannot sweep the GUI-layer ops; "
        "snapshot/restore round-trips the catalog for rollback; and the code fingerprint "
        "is omitted-when-unstamped (byte-identical keys for a session that never reloads), "
        "re-keys when stamped, and returns to the original key when the source is reverted")


# ── vector arity + socket threading (V2.03 §4 C1) ────────────────────────────

def test_socket_dims() -> None:
    v2o = Socket("o", SocketType.VECTOR, Direction.OUT, dims=2)
    v3o = Socket("o", SocketType.VECTOR, Direction.OUT, dims=3)
    v2i = Socket("i", SocketType.VECTOR, Direction.IN, dims=2)
    v3i = Socket("i", SocketType.VECTOR, Direction.IN, dims=3)
    assert can_connect(v2o, v2i)                       # equal arity
    assert can_connect(v2o, v3i)                       # widen 2→3 (pad axial 0)
    assert not can_connect(v3o, v2i)                   # narrow 3→2 rejected
    assert can_connect(Socket("f", SocketType.FLOAT, Direction.OUT), v3i)  # broadcast
    # instantiate carries dims + domain (the audit-flagged drops)
    assert InVector("shift", dims=2).instantiate().dims == 2
    assert InFloat("d", domain=D.VOXEL).instantiate().domain is D.VOXEL
    _ok("socket dims: vector widen-only, dims/domain threaded through instantiate")


# ── calibration write-path + axis helpers (V2.03 §2 A1) ──────────────────────

def test_calibration() -> None:
    ds = Dataset(axes=AxisSizes(z=1), metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
    ds2 = ds.with_metadata(pixel_size_um=0.1)
    assert ds.metadata["pixel_size_um"] == 0.2 and ds2.metadata["pixel_size_um"] == 0.1
    ds3 = ds.with_metadata(z_step_um=None)             # None removes the key
    assert "z_step_um" not in ds3.metadata and "z_step_um" in ds.metadata
    assert not AxisSizes(z=1).is_volumetric and AxisSizes(z=5).is_volumetric
    assert "pixel_size_um" in CALIBRATION_KEYS and "z_step_um" in CALIBRATION_KEYS
    # reshaped_axes drops lattice layers orphaned by an axis change
    big = AxisSizes(z=4, y=8, x=8)
    d = Dataset(axes=big).with_layer(D.PLANE, "focus", np.ones(big.shape_for(D.PLANE)))
    small = AxisSizes(z=1, y=8, x=8)
    assert d.reshaped_axes(small).get(D.PLANE, "focus") is None
    try:
        d.reshaped_axes(small, drop_stale=False)
        raise AssertionError("expected orphan error")
    except ValueError:
        pass
    _ok("calibration: with_metadata copy-on-write, is_volumetric, reshaped_axes")


# ── node variants: the 2D/3D lever reconfigures sockets (V2.03 §3 B1/B2/B3) ───

def test_node_variants() -> None:
    spec = define_node(
        "filt.gauss", "Gaussian", category="enhancement",
        inputs=[InDataset(),
                InFloat("sigma", "Sigma", unit="um",
                        available_in={DIM_MODE: frozenset({"2D"})}),
                InFloat("sigma_xy", "Sigma XY", unit="um",
                        available_in={DIM_MODE: frozenset({"3D"})}),
                InFloat("sigma_z", "Sigma Z", unit="um_axial",
                        available_in={DIM_MODE: frozenset({"3D"})})],
        outputs=[OutDataset()],
        modes=[DimMode()],
        granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
        kernel_axes={"2D": frozenset({"y", "x"}), "3D": frozenset({"z", "y", "x"})},
    )
    st2d, st3d = {"dim": "2D"}, {"dim": "3D"}
    in2d = {s.name for s in spec.active_inputs(st2d)}
    in3d = {s.name for s in spec.active_inputs(st3d)}
    assert in2d == {"data", "sigma"}                   # 2D: single scalar sigma
    assert in3d == {"data", "sigma_xy", "sigma_z"}     # 3D: anisotropic pair
    # granularity + kernel-axes jump with the lever
    assert spec.resolve_granularity(st2d) is Granularity.TILEABLE
    assert spec.resolve_granularity(st3d) is Granularity.WHOLE_VOLUME
    assert spec.resolve_kernel_axes(st3d) == frozenset({"z", "y", "x"})
    # the lever itself
    lever = spec.dim_lever()
    assert spec.has_dim_lever() and lever.presentation == "header"
    assert lever.is_dim_lever and spec.default_state()[DIM_MODE] == "2D"
    _ok("node variants: available_in gates sockets; granularity/kernel-axes per lever")


# ── the edit-time MetaEnvelope pass (V2.03 §2 A3) ────────────────────────────

def test_metadata_pass() -> None:
    define_node("io.load", "Load", outputs=[OutDataset()])            # source (identity)
    define_node("filt.resample", "Resample", inputs=[InDataset()],
                outputs=[OutDataset()], modes=[DimMode()], meta_transform=resample)
    define_node("filt.blur", "Blur", inputs=[InDataset()], outputs=[OutDataset()],
                modes=[DimMode()], meta_transform=None,
                granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME})
    define_node("proj.z", "Z Project", inputs=[InDataset()], outputs=[OutDataset()],
                meta_transform=z_project)
    define_node("stitch.tiles", "Stitch", inputs=[InDataset()], outputs=[OutDataset()],
                meta_transform=stitch)

    # source → resample(×2 lateral) → blur : the blur sees the transformed calibration
    g = Graph()
    g.add(NodeInstance("L", "io.load"))
    g.add(NodeInstance("R", "filt.resample", params={"scale_xy": 2.0}, modes={"dim": "2D"}))
    g.add(NodeInstance("B", "filt.blur", modes={"dim": "2D"}))
    g.connect("L", "R"); g.connect("R", "B")
    seed = MetaEnvelope(axes=AxisSizes(z=1, y=10, x=10), metadata={"pixel_size_um": 0.2})
    env = propagate_meta(g, {"L": seed})
    assert env["R"].axes.y == 20 and env["R"].axes.x == 20            # extent doubled
    assert abs(env["R"].metadata["pixel_size_um"] - 0.1) < 1e-9       # px halved (finer)
    assert env["B"].axes.x == 20 and abs(env["B"].metadata["pixel_size_um"] - 0.1) < 1e-9

    # z-project drops z + z_step and flags collapse; downstream lever defaults to 2D
    g2 = Graph()
    g2.add(NodeInstance("L", "io.load")); g2.add(NodeInstance("Z", "proj.z"))
    g2.connect("L", "Z")
    e2 = propagate_meta(g2, {"L": MetaEnvelope(
        axes=AxisSizes(z=5, y=4, x=4), metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})})
    assert e2["Z"].axes.z == 1 and "z_step_um" not in e2["Z"].metadata
    assert e2["Z"].metadata.get("z_collapsed") is True

    # metadata-intelligent lever default: z>1 ⇒ 3D, z==1 ⇒ 2D (incl. post z-project)
    blur = NODES.get("filt.blur")
    assert resolve_dim_default(blur, MetaEnvelope(axes=AxisSizes(z=5))) == "3D"
    assert resolve_dim_default(blur, MetaEnvelope(axes=AxisSizes(z=1))) == "2D"
    assert resolve_dim_default(blur, e2["Z"]) == "2D"
    assert envelope_symbols(MetaEnvelope(axes=AxisSizes(z=5)))["is_3d"] is True

    # stitch grows Y,X to an UNKNOWN extent (never a silent guess); M→1
    g3 = Graph()
    g3.add(NodeInstance("L", "io.load")); g3.add(NodeInstance("S", "stitch.tiles"))
    g3.connect("L", "S")
    e3 = propagate_meta(g3, {"L": MetaEnvelope(axes=AxisSizes(m=4, y=10, x=10))})
    assert e3["S"].axes.m == 1 and {"y", "x"} <= e3["S"].unknown_axes

    # cycles are rejected outside zones (V2.00 §7)
    gc = Graph()
    gc.add(NodeInstance("A", "io.load")); gc.add(NodeInstance("B", "filt.blur"))
    gc.connect("A", "B"); gc.connect("B", "A")
    try:
        gc.topo_order()
        raise AssertionError("expected cycle rejection")
    except ValueError:
        pass
    _ok("metadata pass: per-edge propagation, z-collapse, lever default, UNKNOWN, cycle guard")


def test_domain_interface() -> None:
    """The socket domain-rail + wire-tint source: reads/adds_domains declarations,
    their accumulation through ``propagate_meta``, and the mismatch validation
    (fake ``test.*`` op_keys so real nodes are never clobbered)."""
    define_node("test.src", "Src", outputs=[OutDataset()],
                adds_domains=frozenset({D.VOXEL}))
    define_node("test.blur", "Blur", inputs=[InDataset()], outputs=[OutDataset()])  # transparent
    define_node("test.label", "Label", inputs=[InDataset()], outputs=[OutDataset()],
                reads_domains=frozenset({D.VOXEL}), adds_domains=frozenset({D.LABEL}))
    define_node("test.measure", "Measure", inputs=[InDataset()], outputs=[OutDataset()],
                reads_domains=frozenset({D.VOXEL, D.LABEL}), adds_domains=frozenset({D.LABEL}))

    # accumulation: src{VOX} → blur{VOX} (transparent) → label{VOX,LBL} → measure{VOX,LBL}
    g = Graph()
    for nid, op in [("S", "test.src"), ("B", "test.blur"),
                    ("L", "test.label"), ("M", "test.measure")]:
        g.add(NodeInstance(nid, op))
    g.connect("S", "B"); g.connect("B", "L"); g.connect("L", "M")
    env = propagate_meta(g, {"S": MetaEnvelope()})
    assert env["S"].domains == frozenset({D.VOXEL})
    assert env["B"].domains == frozenset({D.VOXEL})                 # transparent pass-through
    assert env["L"].domains == frozenset({D.VOXEL, D.LABEL})        # label added
    assert env["M"].domains == frozenset({D.VOXEL, D.LABEL})        # inherited (measure adds LBL)

    # spec helpers: out_domains unions, missing_domains flags an absent requirement
    label = NODES.get("test.label"); measure = NODES.get("test.measure")
    assert label.out_domains(frozenset({D.VOXEL})) == frozenset({D.VOXEL, D.LABEL})
    assert measure.missing_domains(frozenset({D.VOXEL})) == frozenset({D.LABEL})  # no LBL upstream
    assert measure.missing_domains(frozenset({D.VOXEL, D.LABEL})) == frozenset()  # satisfied

    # a merge unions BOTH Dataset predecessors' domain-sets
    define_node("test.merge", "Merge", inputs=[InDataset(multi=True)], outputs=[OutDataset()])
    define_node("test.spots", "Spots", inputs=[InDataset()], outputs=[OutDataset()],
                adds_domains=frozenset({D.POINT}))
    g2 = Graph()
    for nid, op in [("S", "test.src"), ("L", "test.label"),
                    ("P", "test.spots"), ("G", "test.merge")]:
        g2.add(NodeInstance(nid, op))
    g2.connect("S", "L"); g2.connect("S", "P")
    g2.connect("L", "G"); g2.connect("P", "G")
    e2 = propagate_meta(g2, {"S": MetaEnvelope()})
    assert e2["G"].domains == frozenset({D.VOXEL, D.LABEL, D.POINT})   # both branches merged
    _ok("domain interface: reads/adds declarations, accumulation, merge-union, mismatch validation")


# ── lazy tiled provider (Phase 2a — V2.02 §3 / V2.03 §5) ─────────────────────

def test_provider() -> None:
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=300, x=400)
    sp = SyntheticProvider(ax, tile=128, levels=2)
    # exact window read
    r = sp.get_region(0, 0, 0, 1, 0, 10, 20, 30, 45)
    assert r.shape == (10, 15) and r.dtype == np.uint16
    # edge tile clips; get_tile == get_region on the block's coords
    t23 = sp.get_tile(0, 0, 0, 0, 0, 2, 3)
    assert t23.shape == (44, 16)              # (300-256, 400-384)
    assert np.array_equal(t23, sp.get_region(0, 0, 0, 0, 0, 256, 300, 384, 400))
    # subvolume stacks z (planar blocks gathered); region-volume over z
    sv = sp.get_subvolume(0, 0, 0, 0, 0, 3, 0, 0)
    assert sv.shape == (3, 128, 128) and np.array_equal(sv[1], sp.get_tile(0, 0, 0, 1, 0, 0, 0))
    assert sp.get_region_volume(0, 0, 0, 0, 0, 2, 5, 9, 5, 10).shape == (2, 4, 5)
    # multiscale: level-1 axes halve; stride-decimated values
    assert sp.level_axes(1).y == 150 and sp.level_axes(1).x == 200
    assert np.array_equal(sp.get_region(1, 0, 0, 0, 0, 0, 4, 0, 4),
                          sp.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8)[::2, ::2])
    assert sp.tiles_per_plane(0) == (3, 4)    # ceil(300/128), ceil(400/128)
    try:
        sp.get_tile(0, 0, 0, 0, 0, 99, 0)     # out of range
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    _ok("provider: synthetic region/tile/subvolume/volume + multiscale + edge clip")

    if _HAVE_BLOSC2:
        rng = np.random.default_rng(0)
        vol = rng.integers(0, 4096, size=(1, 1, 3, 2, 40, 50), dtype=np.uint16)
        bp = B2ndProvider.from_array(vol, tile=16, levels=2)   # planar (1,1,1,1,16,16) blocks
        assert bp.axes.y == 40 and bp.axes.c == 2 and bp.levels == 2
        # b2nd window read round-trips exactly vs the source slice
        assert np.array_equal(bp.get_region(0, 0, 0, 2, 1, 5, 12, 8, 20),
                              vol[0, 0, 2, 1, 5:12, 8:20])
        assert np.array_equal(bp.get_tile(0, 0, 0, 1, 0, 1, 2), vol[0, 0, 1, 0, 16:32, 32:48])
        sv = bp.get_subvolume(0, 0, 0, 0, 0, 3, 0, 0)
        assert sv.shape[0] == 3 and np.array_equal(sv[2], vol[0, 0, 2, 0, 0:16, 0:16])
        # level-1 == mean-downsample of the source (Y,X halved)
        assert bp.level_axes(1).y == 20 and bp.level_axes(1).x == 25
        exp = vol[0, 0, 2, 1].reshape(20, 2, 25, 2).mean(axis=(1, 3)).astype(vol.dtype)
        assert np.array_equal(bp.get_region(1, 0, 0, 2, 1, 0, 20, 0, 25), exp)

        # C4: DISK-backed store — write → open (lazy) → identical reads; mtime-based
        # version (cheap, disk-cheap identity) that differs from an in-memory store.
        import os
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="b2nd_disk_")
        try:
            store = os.path.join(d, "store")
            dp = B2ndProvider.write(vol, store, tile=16, levels=2)
            assert dp.axes.y == 40 and dp.levels == 2
            assert np.array_equal(dp.get_region(0, 0, 0, 2, 1, 5, 12, 8, 20),
                                  vol[0, 0, 2, 1, 5:12, 8:20])
            assert np.array_equal(dp.get_region(1, 0, 0, 2, 1, 0, 20, 0, 25), exp)
            re = B2ndProvider.open(store)             # re-open without re-ingest
            assert np.array_equal(re.get_region(0, 0, 0, 0, 0, 0, 8, 0, 8),
                                  vol[0, 0, 0, 0, 0:8, 0:8])
            # disk identity folds path+mtime (cheap); distinct from the in-memory hash
            fp = dp.fingerprint()
            assert fp[0] == "b2nd-disk" and fp[2] == dp._mtime_ns and dp.version == fp
            assert fp != bp.fingerprint()
            del dp, re

            # progress: the band-at-a-time fill an ingest bar rides must report a
            # MONOTONIC 0→1 that lands exactly full, and must store byte-identical
            # pixels to the bulk asarray path (it is the same chunk alignment).
            seen: list = []
            store2 = os.path.join(d, "store_p")
            pp = B2ndProvider.write(vol, store2, tile=16, levels=2,
                                    progress=seen.append)
            assert seen and all(0.0 <= f <= 1.0 for f in seen)
            assert all(b >= a for a, b in zip(seen, seen[1:]))
            assert abs(seen[-1] - 1.0) < 1e-9      # a bar must always land full
            assert np.array_equal(pp.get_region(0, 0, 0, 2, 1, 5, 12, 8, 20),
                                  vol[0, 0, 2, 1, 5:12, 8:20])
            assert np.array_equal(pp.get_region(1, 0, 0, 2, 1, 0, 20, 0, 25), exp)
            mp: list = []
            B2ndProvider.from_array(vol, tile=16, levels=2, progress=mp.append)
            assert mp and abs(mp[-1] - 1.0) < 1e-9   # in-memory path reports too
            del pp

            # ── V2.19: the pyramid streams, is marked, and REPAIRS ──────────────
            # The writer no longer downsamples a whole level in RAM — it streams level l
            # out of level l-1's store one z-slab at a time — so this asserts the streamed
            # levels against `_mean_downsample_2x`, the in-RAM reference form.
            from nodegraph.provider import _LEVEL_META, _mean_downsample_2x
            store3 = os.path.join(d, "store_pyr")
            deep = B2ndProvider.write(vol, store3, tile=16, levels=4)
            assert deep.levels == 4, deep.levels
            assert [(deep.level_axes(l).y, deep.level_axes(l).x)
                    for l in range(4)] == [(40, 50), (20, 25), (10, 12), (5, 6)]
            ref6 = vol
            for lv in range(4):
                assert np.array_equal(np.asarray(deep._arrays[lv][...]), ref6), lv
                ref6 = _mean_downsample_2x(ref6)
                assert B2ndProvider._level_state(deep._arrays[lv]) == "complete", lv
            del deep, ref6

            # a level_0-only store — what an ingest killed between levels leaves behind,
            # and precisely what the lab's 84.7 GB series had on disk — regains its
            # pyramid from level 0 ALONE (no source file), bit-identically…
            store4 = os.path.join(d, "store_short")
            os.makedirs(store4)
            shutil.copy2(os.path.join(store3, "level_0.b2nd"),
                         os.path.join(store4, "level_0.b2nd"))
            short = B2ndProvider.open(store4)
            assert short.levels == 1
            fp_short, ver_short = short.fingerprint(), short.version
            rep: list = []
            fixed = B2ndProvider.ensure_levels(store4, 4, progress=rep.append)
            assert fixed.levels == 4
            whole = B2ndProvider.open(store3)
            for lv in range(4):
                assert np.array_equal(np.asarray(fixed._arrays[lv][...]),
                                      np.asarray(whole._arrays[lv][...])), lv
            assert rep and all(b >= a for a, b in zip(rep, rep[1:]))
            assert abs(rep[-1] - 1.0) < 1e-9
            # …and the repair is MEMO-NEUTRAL: identity comes from level 0, which it never
            # touches, so completing a pyramid must not re-key the source and discard every
            # memoized result downstream of it (on the lab's series, a 4-minute Deconvolve).
            assert fixed.fingerprint() == fp_short and fixed.version == ver_short
            assert B2ndProvider.ensure_levels(store4, 4).levels == 4      # idempotent
            del short, fixed, whole

            # a TORN level is not served: above 0 the store opens short (repairable),
            # at 0 it is a hard error, because its unwritten blocks read as zeros and
            # nothing can rebuild level 0 but a re-ingest.
            import blosc2 as _b2
            _a = _b2.open(os.path.join(store4, "level_2.b2nd"), mode="a")
            _a.vlmeta[_LEVEL_META] = {"complete": False, "level": 2}
            del _a
            assert B2ndProvider.open(store4).levels == 2
            assert B2ndProvider.ensure_levels(store4, 4).levels == 4   # …and repairable
            _a = _b2.open(os.path.join(store4, "level_0.b2nd"), mode="a")
            _a.vlmeta[_LEVEL_META] = {"complete": False, "level": 0}
            del _a
            try:
                B2ndProvider.open(store4)
                raise AssertionError("a torn level 0 must refuse to open")
            except ValueError as exc:
                assert "re-ingest" in str(exc), exc
            # ── V2.20: the MARKER-FREE tear test (`_blank_tail`) ────────────────
            # A store written before the marker existed reads as "legacy" and is trusted,
            # and the lab's 84.7 GB series proved that trust misplaced: a legacy level_0
            # holding 7 899 of 40 320 chunks served solid black for m≥2, and the pyramid
            # repair copied the zeros upward and stamped them complete. blosc2 files a
            # never-written chunk as a special run-length value, so which chunks were
            # written is a FACT on disk — no decompression, no inference from pixels.
            def _torn_store(name, fill):     # a level_0-only legacy store, partly written
                p = os.path.join(d, name)
                os.makedirs(p, exist_ok=True)
                a = _b2.empty((2, 2, 4, 1, 40, 50), dtype=np.uint16,
                              chunks=(1, 1, 1, 1, 40, 50), blocks=(1, 1, 1, 1, 20, 25),
                              urlpath=os.path.join(p, "level_0.b2nd"), mode="w")
                for im, it, iz in fill:      # C-order chunk index = ((im*2)+it)*4 + iz
                    # NON-constant on purpose: blosc2 files a constant chunk as a special
                    # run-length value too, so a `np.full` fill would read back as blank
                    # and the test would pass for the wrong reason.
                    a[im, it, iz, 0] = rng.integers(1, 4096, (40, 50), dtype=np.uint16)
                del a
                return p
            # 16 chunks; the write stopped after 3 → blanks are a pure TAIL.
            cut = B2ndProvider.open(_torn_store(
                "store_cut", [(0, 0, 0), (0, 0, 1), (0, 0, 2)]))
            assert cut.level_state(0) == "legacy"          # no marker ⇒ open() accepts it
            assert cut.blank_tail(0) == (3, 16), cut.blank_tail(0)
            del cut
            # SPARSE is not TORN: one blank chunk inside the written body (a mask with an
            # empty plane) must NOT be flagged, or the verdict would cost a re-ingest.
            sparse = B2ndProvider.open(_torn_store(
                "store_sparse", [(im, it, iz) for im in range(2) for it in range(2)
                                 for iz in range(4)
                                 if ((im * 2 + it) * 4 + iz) not in (5, 14, 15)]))
            assert sparse.blank_tail(0) is None, sparse.blank_tail(0)
            del sparse
            # a fully-written store has no tail at all…
            assert B2ndProvider.open(store3).blank_tail(0) is None
            # …and the census is GATED on the marker, so it self-sunsets: a level 0 that
            # says `complete` was stamped after its last chunk landed and is not re-scanned.
            marked = _torn_store("store_marked", [(0, 0, 0)])
            _a = _b2.open(os.path.join(marked, "level_0.b2nd"), mode="a")
            _a.vlmeta[_LEVEL_META] = {"complete": True, "level": 0,
                                      "shape": [2, 2, 4, 1, 40, 50]}
            del _a
            assert B2ndProvider.open(marked).level_state(0) == "complete"

            # a GAP is a broken store, not a shorter one: level_1 without level_0 would
            # otherwise serve level_1's pixels to a level_0 read.
            store5 = os.path.join(d, "store_gap")
            os.makedirs(store5)
            shutil.copy2(os.path.join(store3, "level_1.b2nd"),
                         os.path.join(store5, "level_1.b2nd"))
            try:
                B2ndProvider.open(store5)
                raise AssertionError("a store with no level_0 must refuse to open")
            except FileNotFoundError:
                pass
        finally:
            import gc
            gc.collect()
            shutil.rmtree(d, ignore_errors=True)
        _ok(f"provider: b2nd planar-block store (in-memory + DISK write/open, "
            f"mtime version) + mean pyramid + monotonic ingest progress "
            f"(blosc2 {_BLOSC2_VER}); V2.19 pyramid — 4 STREAMED levels bit-identical to "
            f"the in-RAM reference (no whole-level allocation), every level marked "
            f"complete, a level_0-only store repaired from level 0 alone and MEMO-NEUTRAL "
            f"(identity is level 0's mtime), idempotent, and a torn level served never: "
            f"short+repairable above 0, a hard error at 0, a gap refused; V2.20 — the "
            f"marker-free chunk census catches a LEGACY level_0 that stops partway "
            f"(a pure blank tail), calls scattered blanks sparse rather than torn, and "
            f"is skipped once a level is marked")
    else:
        _ok("provider: b2nd tests SKIPPED (blosc2 not installed)")


# ── two-hash memo (V2.02 §6) ─────────────────────────────────────────────────

def test_memo() -> None:
    m = Memo()
    rh = node_recipe_hash("op.a", {"k": 1}, (), ())
    assert m.get(rh) is None and m.misses == 1              # miss
    e1 = m.put(rh, np.ones((4, 4)), node_key="A")
    assert m.get(rh) is e1 and m.hits == 1                  # hit
    # dedup: an identical payload from another node shares the blob
    e2 = m.put(node_recipe_hash("op.b", {"k": 2}, (), ()), np.ones((4, 4)), node_key="B")
    assert e2.payload is e1.payload
    # cutoff: same node, identical output → not "changed"; different output → changed
    assert m.put(rh, np.ones((4, 4)), node_key="A").changed is False
    e_diff = m.put(rh, np.zeros((4, 4)), node_key="A")
    assert e_diff.changed is True and e_diff.revision > e1.revision
    # recipe-hash sensitivity: params + upstream (hashes/revisions) all matter
    assert node_recipe_hash("op.a", {"k": 1}, (), ()) == rh
    assert node_recipe_hash("op.a", {"k": 2}, (), ()) != rh
    assert node_recipe_hash("op.a", {"k": 1}, ("h",), (7,)) != rh
    _ok("memo: two-hash lookup, blob dedup, cutoff, revision, hash sensitivity")


def test_memo_gc() -> None:
    """Memo GC — byte-budget LRU eviction (V2.04 §6b follow-up). Bounds the persistent-
    memo hazard: an eager full-raster node in a high-T unrolled zone otherwise pins one
    raster per iteration forever. Eviction is correctness-safe (a later recompute)."""
    from nodegraph.provider import ArrayProvider

    each = int(np.ones((100, 100)).nbytes)                 # 80_000 (float64)

    # 1. unbounded default (budget None) is byte-identical to pre-GC behavior: no eviction
    mu = Memo()
    assert mu.budget is None
    for i in range(50):
        mu.put(node_recipe_hash("t.u", {"i": i}, (), ()), np.zeros((64, 64)),
               node_key=f"u{i}")
    assert mu.evictions == 0 and len(mu._entries) == 50

    # 2. dedup accounting is per unique BLOB, not per entry — the shared blob is counted
    #    once and freed only when the LAST referencing entry drops (refcount invariant)
    md = Memo(budget_bytes=1 << 20)
    ra = node_recipe_hash("t.a", {}, (), ())
    rb = node_recipe_hash("t.b", {}, (), ())
    ea = md.put(ra, np.ones((100, 100)), node_key="A")
    eb = md.put(rb, np.ones((100, 100)), node_key="B")     # identical content → shared blob
    assert eb.payload is ea.payload and md._fp_refs[ea.fingerprint] == 2
    assert md.nbytes == each                               # counted ONCE despite two entries
    md.invalidate(ra)                                      # one ref dropped, blob survives
    assert ea.fingerprint in md._blobs and md.nbytes == each
    md.invalidate(rb)                                      # last ref → blob + bytes freed
    assert md.nbytes == 0 and not md._blobs

    # 3. byte-budget LRU: distinct large payloads over budget evict the OLDEST
    budget = 3 * each + 1
    ml = Memo(budget_bytes=budget)
    rhs = [node_recipe_hash("t.g", {"i": i}, (), ()) for i in range(10)]
    for i, rh in enumerate(rhs):
        ml.put(rh, np.full((100, 100), float(i)), node_key=f"g{i}")
    assert ml.nbytes <= budget and ml.evictions >= 6
    assert ml.get(rhs[0]) is None and ml.get(rhs[-1]) is not None   # oldest gone, newest kept

    # 3b. recency: a HIT promotes an entry so it outlives a newer-but-cold one
    ml2 = Memo(budget_bytes=2 * each + 1)
    r0, r1, r2 = (node_recipe_hash("t.r", {"i": i}, (), ()) for i in range(3))
    ml2.put(r0, np.full((100, 100), 0.0), node_key="r0")
    ml2.put(r1, np.full((100, 100), 1.0), node_key="r1")
    assert ml2.get(r0) is not None                         # promote r0 to most-recent
    ml2.put(r2, np.full((100, 100), 2.0), node_key="r2")   # evicts the LRU = r1, not r0
    assert ml2.get(r1) is None and ml2.get(r0) is not None and ml2.get(r2) is not None

    # 4. _last_fp is NEVER GC'd — an evict-then-recompute of identical bytes still cuts off
    mc = Memo(budget_bytes=each + 1)                       # holds ~one entry
    ka = node_recipe_hash("t.c", {"i": "a"}, (), ())
    kb = node_recipe_hash("t.c", {"i": "b"}, (), ())
    mc.put(ka, np.full((100, 100), 7.0), node_key="A")
    mc.put(kb, np.full((100, 100), 9.0), node_key="B")     # evicts the ka entry
    assert mc.get(ka) is None
    again = mc.put(ka, np.full((100, 100), 7.0), node_key="A")   # identical content
    assert again.changed is False                          # cutoff preserved (last_fp kept)

    # 5. end-to-end through the Engine (memo_bytes): a long linear chain of eager
    #    ArrayProvider nodes — the V2.04 hazard shape — stays bounded and computes right
    define_node("test.gc_src", "GcSrc", outputs=[OutDataset()])
    define_node("test.gc_gen", "GcGen", inputs=[InDataset()], outputs=[OutDataset()])
    SZ, N = 128, 40
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=SZ, x=SZ)

    def c_gc_src(ctx):
        return Dataset(axes=ax, image=ArrayProvider(np.zeros((1, 1, 1, 1, SZ, SZ), np.uint8)))

    def c_gc_gen(ctx):
        k = int(ctx.params["k"]) % 256                     # distinct per node → distinct blobs
        return Dataset(axes=ctx.inputs[0].axes,
                       image=ArrayProvider(np.full((1, 1, 1, 1, SZ, SZ), k, np.uint8)))

    img_bytes = int(np.zeros((1, 1, 1, 1, SZ, SZ), np.uint8).nbytes)   # 16_384
    gc_computes = {"test.gc_src": c_gc_src, "test.gc_gen": c_gc_gen}
    g = Graph(); g.add(NodeInstance("n0", "test.gc_src")); prev = "n0"
    for i in range(1, N + 1):
        nid = f"n{i}"; g.add(NodeInstance(nid, "test.gc_gen", params={"k": i}))
        g.connect(prev, nid); prev = nid

    def _tip_val(ds):
        return int(ds.image.read_region(0, 0, 0, 0, 0, 0, SZ, 0, SZ).ravel()[0])

    # a budget for ~5 rasters vs a 41-node chain → GC must evict the long tail
    e2e_budget = 5 * img_bytes
    eng = Engine(g, computes=gc_computes, memo_bytes=e2e_budget)
    assert _tip_val(eng.pull(prev)) == N % 256                       # correct tip
    assert eng.memo.nbytes <= e2e_budget and eng.memo.evictions > 0  # bounded + GC ran
    # re-pull is correctness-safe under eviction: evicted ancestors force a recompute
    # (the honest memory/recompute trade — identical result), and the memo stays bounded
    # across pulls (no unbounded growth — the whole point).
    assert _tip_val(eng.pull(prev)) == N % 256 and eng.memo.nbytes <= e2e_budget

    # contrast — a budget that fits the whole chain: GC never runs, so a re-pull is fully
    # memoized (the normal C1 cutoff — zero recompute). Proves the GC only trades recompute
    # under genuine pressure and is inert otherwise.
    big = Engine(g, computes=gc_computes, memo_bytes=(N + 2) * img_bytes)
    assert _tip_val(big.pull(prev)) == N % 256
    c0 = big.compute_count
    big.pull(prev)
    assert big.compute_count == c0 and big.memo.evictions == 0
    _ok("memo GC: byte-budget LRU eviction, per-blob refcount, recency, cutoff-safe, "
        "end-to-end bounded long chain (evict-recompute vs fits-fully-memoized)")


# ── lazy pull engine + ReadContext + granularity routing (V2.02 §8 / V2.03) ──

def test_engine() -> None:
    define_node("eng.src", "Src", outputs=[OutDataset()])
    define_node("eng.scale", "Scale", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("eng.sink", "Sink", inputs=[InDataset()], outputs=[OutDataset()])

    calls = {"src": 0, "scale": 0, "sink": 0}

    def c_src(ctx):
        calls["src"] += 1
        return np.array([1.0])

    def c_scale(ctx):
        calls["scale"] += 1
        return np.asarray(ctx.inputs[0]) * (ctx.calib("pixel_size_um") or 1.0)

    def c_sink(ctx):
        calls["sink"] += 1
        return np.asarray(ctx.inputs[0]) + float(ctx.params.get("bias", 0.0))

    computes = {"eng.src": c_src, "eng.scale": c_scale, "eng.sink": c_sink}
    g = Graph()
    g.add(NodeInstance("S", "eng.src"))
    g.add(NodeInstance("K", "eng.scale"))
    g.add(NodeInstance("N", "eng.sink", params={"bias": 10.0}))
    g.connect("S", "K"); g.connect("K", "N")
    eng = Engine(g, computes=computes,
                 meta_seeds={"S": MetaEnvelope(metadata={"pixel_size_um": 2.0})})

    assert np.allclose(eng.pull("N"), 1.0 * 2.0 + 10.0)     # 12.0
    assert calls == {"src": 1, "scale": 1, "sink": 1}       # each computed once
    eng.pull("N")                                           # all memo hits
    assert calls == {"src": 1, "scale": 1, "sink": 1} and eng.memo.hits > 0

    # downstream param change → only the sink recomputes (upstream isolated)
    g.nodes["N"] = NodeInstance("N", "eng.sink", params={"bias": 100.0})
    assert np.allclose(eng.pull("N"), 1.0 * 2.0 + 100.0)    # 102.0
    assert calls == {"src": 1, "scale": 1, "sink": 2}

    # a calibration value the SCALE node READ changes → scale + sink recompute
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 5.0})})
    assert np.allclose(eng.pull("N"), 1.0 * 5.0 + 100.0)    # 105.0
    assert calls["scale"] == 2 and calls["sink"] == 3 and calls["src"] == 1

    # a calibration value the scale node did NOT read → precise: no recompute
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 5.0, "dt_s": 0.7})})
    eng.pull("N")
    assert calls["scale"] == 2
    _ok("engine: lazy pull, memo hit, precise read-invalidation, downstream isolation")


def test_engine_granularity() -> None:
    ax = AxisSizes(m=1, t=1, z=4, c=1, y=64, x=64)
    define_node("eng.filter", "Filter", inputs=[], outputs=[OutDataset()],
                modes=[DimMode()],
                granularity={"2D": Granularity.WHOLE_PLANE,
                             "3D": Granularity.WHOLE_VOLUME})

    def c_filter(ctx):
        p = ctx.provider
        if ctx.is_volume:                                  # 3D → z-range brick
            return p.get_subvolume(0, 0, 0, 0, 0, ctx.env.axes.z, 0, 0)
        return p.get_tile(0, 0, 0, 0, 0, 0, 0)             # 2D → single-z tile

    prov = SyntheticProvider(ax, tile=32, levels=1)
    seed = {"F": MetaEnvelope(axes=ax)}

    g2 = Graph(); g2.add(NodeInstance("F", "eng.filter", modes={"dim": "2D"}))
    e2 = Engine(g2, computes={"eng.filter": c_filter}, providers={"F": prov}, meta_seeds=seed)
    out2 = e2.pull("F")
    assert out2.ndim == 2 and out2.shape == (32, 32)       # tile path

    g3 = Graph(); g3.add(NodeInstance("F", "eng.filter", modes={"dim": "3D"}))
    e3 = Engine(g3, computes={"eng.filter": c_filter}, providers={"F": prov}, meta_seeds=seed)
    out3 = e3.pull("F")
    assert out3.ndim == 3 and out3.shape == (4, 32, 32)    # subvolume path

    # the 2D/3D lever folds into recipe_hash → distinct memo entries
    assert e2.entry("F").recipe_hash != e3.entry("F").recipe_hash
    _ok("engine: 2D/3D lever routes provider path (tile vs subvolume) + distinct memo keys")


# ── the solo-frame scope (GUI troubleshooting mode, engine-side mechanism) ─────

def test_solo_frame_scope() -> None:
    """A run scoped to ONE frame by pinning the source seed — what the GUI's
    troubleshooting mode (``EngineRunner.set_solo_frame``, ``F9``) is built on.

    The whole point is that **no node participates**: the catalog keeps iterating "every
    frame", the pinned seed just means there is one. So the properties to hold are about
    the seam — geometry, addressing, and the memo fence — not about any node's maths."""
    from nodegraph.provider import FrameSliceProvider

    base = SyntheticProvider(AxisSizes(m=3, t=5, z=2, c=2, y=40, x=48), tile=16, levels=2)
    sl = FrameSliceProvider(base, 2, 3)

    # geometry: M,T collapse; z/c/y/x and the PYRAMID pass through untouched
    assert (sl.axes.m, sl.axes.t) == (1, 1) and sl.frame == (2, 3)
    assert (sl.axes.z, sl.axes.c, sl.axes.y, sl.axes.x) == (2, 2, 40, 48)
    assert sl.levels == base.levels and sl.tile == base.tile
    assert sl.level_axes(1).y == 20 and sl.level_axes(1).m == 1
    # every read resolves to the pin — including a hand-built read at a non-zero address,
    # which must NOT escape the scope
    for z, c in ((0, 0), (1, 1)):
        assert np.array_equal(sl.get_region(0, 0, 0, z, c, 0, 40, 0, 48),
                              base.get_region(0, 2, 3, z, c, 0, 40, 0, 48))
    assert np.array_equal(sl.read_region(0, 7, 9, 1, 0, 0, 8, 0, 8),
                          base.read_region(0, 2, 3, 1, 0, 0, 8, 0, 8))
    assert np.array_equal(sl.get_subvolume(0, 0, 0, 1, 0, 2, 0, 0),
                          base.get_subvolume(0, 2, 3, 1, 0, 2, 0, 0))
    # a pyramid read decimates the PINNED frame, not frame 0
    assert np.array_equal(sl.get_region(1, 0, 0, 0, 0, 0, 4, 0, 4),
                          base.get_region(1, 2, 3, 0, 0, 0, 4, 0, 4))
    # identity: the pin is part of it, so no two frames share a fingerprint/version
    fps = {FrameSliceProvider(base, m, t).fingerprint()
           for m in range(3) for t in range(5)}
    assert len(fps) == 15 and base.fingerprint() not in fps
    assert sl.version == sl.fingerprint()

    # ── the seam: a node counting the frames it walks sees exactly one ──────────
    define_node("solo.count", "Count frames", inputs=[InDataset()],
                outputs=[OutDataset()], granularity=Granularity.WHOLE_VOLUME)

    def c_count(ctx):
        ds = ctx.inputs[0]
        ax = ds.image.axes
        walked = [(m, t) for m in range(ax.m) for t in range(ax.t)]
        # read one plane per unit so the count is backed by real reads, not just axes
        mean = float(np.mean([ds.image.get_region(0, m, t, 0, 0, 0, 4, 0, 4).mean()
                             for m, t in walked]))
        return ds.with_layer(Domain.GLOBAL, "units", np.asarray(len(walked))) \
                 .with_layer(Domain.GLOBAL, "mean", np.asarray(mean))

    g = Graph()
    g.add(NodeInstance("S", "io.seedsolo"))
    g.add(NodeInstance("N", "solo.count"))
    g.connect("S", "N", src_socket="out", dst_socket="data")

    def run(prov):
        env = MetaEnvelope(axes=prov.axes)
        eng = Engine(g, computes={"solo.count": c_count},
                     seeds={"S": Dataset(axes=prov.axes).with_image(prov)},
                     meta_seeds={"S": env}, memo=memo)
        return eng, eng.pull("N")

    memo = Memo()                       # shared across runs: the fence is the assertion
    full_e, full = run(base)
    solo_e, solo = run(sl)
    assert int(full.get(Domain.GLOBAL, "units").values) == 15    # M·T of the series
    assert int(solo.get(Domain.GLOBAL, "units").values) == 1     # …one frame
    # and it is the RIGHT frame: the soloed mean equals frame (2,3)'s own
    one = float(base.get_region(0, 2, 3, 0, 0, 0, 4, 0, 4).mean())
    assert abs(float(solo.get(Domain.GLOBAL, "mean").values) - one) < 1e-9

    # the memo fence: the pin rides seed version → recipe_hash, so the scopes never
    # alias, two frames never alias, and re-pinning a computed frame is a HIT
    h_full = full_e.entry("N").recipe_hash
    h_solo = solo_e.entry("N").recipe_hash
    other_e, _ = run(FrameSliceProvider(base, 0, 1))
    assert len({h_full, h_solo, other_e.entry("N").recipe_hash}) == 3
    again_e, again = run(FrameSliceProvider(base, 2, 3))
    assert again_e.entry("N").recipe_hash == h_solo and again_e.compute_count == 0
    assert int(again.get(Domain.GLOBAL, "units").values) == 1

    # the envelope must follow the payload (build-node-v2 §2: axes agree)
    assert solo_e.env("N").axes.t == 1 and solo_e.env("N").axes.m == 1
    assert full_e.env("N").axes.t == 5 and full_e.env("N").axes.m == 3
    _ok("solo frame: (m,t)-pinned source view — 15 units → 1 (the pinned frame's own "
        "pixels), pyramid/z/c intact, pin folded into every recipe_hash (scope + frame "
        "distinct, revisit is a memo hit), envelope tracks the payload")


def test_frame_subset_scope() -> None:
    """A run scoped to a CHOSEN SET of frames and planes — the multi-pick half of the
    GUI's troubleshooting mode (picking boxes on the Viewer's M/T/Z strips, F9).

    Its reason to exist over the single-frame pin: a temporal node needs more than one
    frame to do anything at all, and a 3D node checked on 4 of 60 planes is 15× cheaper.
    So the properties to hold are that the subset really is a short series of short
    volumes (both in acquisition order), that the three axes compose as a cross product,
    that it addresses back to the display cursor, and that it fences the memo per
    selection."""
    from nodegraph.provider import FrameSubsetProvider, subset_index

    base = SyntheticProvider(AxisSizes(m=3, t=5, z=6, c=2, y=40, x=48), tile=16, levels=2)
    sub = FrameSubsetProvider(base, (0,), (4, 1, 3))

    # geometry: M,T shrink to the picks — SORTED, so t still rises through the subset and
    # a temporal node walks the chosen frames in acquisition order. z/c/y/x pass through
    # while zs is unset, which is the default a 3D node depends on.
    assert sub.frames == ((0,), (1, 3, 4)) and (sub.axes.m, sub.axes.t) == (1, 3)
    assert sub.planes is None
    assert (sub.axes.z, sub.axes.c, sub.axes.y, sub.axes.x) == (6, 2, 40, 48)
    assert sub.levels == base.levels and sub.tile == base.tile
    assert sub.level_axes(1).y == 20 and sub.level_axes(1).t == 3
    for i, bt in enumerate((1, 3, 4)):
        assert np.array_equal(sub.get_region(0, 0, i, 0, 0, 0, 40, 0, 48),
                              base.get_region(0, 0, bt, 0, 0, 0, 40, 0, 48))
    # a hand-built read at a non-existent address clamps INTO the subset, never out of it
    assert np.array_equal(sub.read_region(0, 9, 9, 1, 0, 0, 8, 0, 8),
                          base.read_region(0, 0, 4, 1, 0, 0, 8, 0, 8))
    # out-of-range picks clamp to the nearest real index (and dedupe against it) and an
    # empty pick degrades to index 0 rather than to an empty axis — a graph edit can
    # shrink a source under a stale GUI selection, and that must not fail the pull
    assert FrameSubsetProvider(base, (), (99, 2)).frames == ((0,), (2, 4))
    assert FrameSubsetProvider(base, (0,), (4, 99)).frames == ((0,), (4,))
    # the display cursor maps onto the payload; an UNPICKED index lands on the nearest
    # picked one, which is what keeps a cursor outside the scope addressing real pixels,
    # and an UNPICKED AXIS passes through untouched (z, when only frames were picked)
    assert [subset_index((1, 3, 4), t) for t in range(6)] == [0, 0, 0, 1, 2, 2]
    assert [subset_index((), z) for z in range(4)] == [0, 1, 2, 3]

    # ── z picks: same remap, applied INSIDE every scoped frame ──────────────────
    zsub = FrameSubsetProvider(base, (0,), (1, 3), (5, 1, 2))
    assert zsub.planes == (1, 2, 5) and zsub.axes.z == 3
    assert (zsub.axes.m, zsub.axes.t) == (1, 2), "z picks must not disturb the frames"
    assert zsub.level_axes(1).z == 3 and zsub.level_axes(1).y == 20
    for zi, bz in enumerate((1, 2, 5)):        # every picked z, in every picked frame
        for ti, bt in enumerate((1, 3)):
            assert np.array_equal(zsub.get_region(0, 0, ti, zi, 0, 0, 40, 0, 48),
                                  base.get_region(0, 0, bt, bz, 0, 0, 40, 0, 48))
    # the derived volume reads compose off the remapped planes — a 3D node gathering a
    # z-range gets the picked planes stacked in order, not the base's z0..z2
    assert np.array_equal(zsub.get_subvolume(0, 0, 1, 0, 0, 3, 0, 0),
                          np.stack([base.get_tile(0, 0, 3, bz, 0, 0, 0)
                                    for bz in (1, 2, 5)]))
    assert np.array_equal(zsub.read_region(0, 0, 0, 9, 0, 0, 8, 0, 8),   # clamps into it
                          base.read_region(0, 0, 1, 5, 0, 0, 8, 0, 8))
    # an unset z is not the same identity as a z pick covering everything, but it IS the
    # same identity as before z became pickable — a frame-only scope never re-keyed
    assert FrameSubsetProvider(base, (0,), (1, 3)).fingerprint() \
        != FrameSubsetProvider(base, (0,), (1, 3), tuple(range(6))).fingerprint()
    assert FrameSubsetProvider(base, (0,), (1, 3), ()).fingerprint() \
        == FrameSubsetProvider(base, (0,), (1, 3)).fingerprint()

    define_node("subset.count", "Count frames", inputs=[InDataset()],
                outputs=[OutDataset()], granularity=Granularity.WHOLE_VOLUME)

    def c_count(ctx):
        ds = ctx.inputs[0]
        ax = ds.image.axes
        walked = [(m, t) for m in range(ax.m) for t in range(ax.t)]
        # a Frame-domain layer, so the (m,t) LATTICE the node walked is the assertion —
        # a subset that reads the right pixels in the wrong order would still fail. The
        # value is the frame's whole-VOLUME mean, so a z pick moves it too.
        means = np.asarray(
            [np.mean([ds.image.get_region(0, m, t, z, 0, 0, 4, 0, 4).mean()
                      for z in range(ax.z)]) for m, t in walked],
            dtype=float).reshape(ax.m, ax.t)
        return ds.with_layer(Domain.GLOBAL, "units", np.asarray(len(walked))) \
                 .with_layer(Domain.GLOBAL, "planes", np.asarray(ax.z)) \
                 .with_layer(Domain.FRAME, "mean", means)

    g = Graph()
    g.add(NodeInstance("S", "io.seedsubset"))
    g.add(NodeInstance("N", "subset.count"))
    g.connect("S", "N", src_socket="out", dst_socket="data")

    memo = Memo()
    def run(prov):
        eng = Engine(g, computes={"subset.count": c_count},
                     seeds={"S": Dataset(axes=prov.axes).with_image(prov)},
                     meta_seeds={"S": MetaEnvelope(axes=prov.axes)}, memo=memo)
        return eng, eng.pull("N")

    def vol_mean(m, t, zs):
        return float(np.mean([base.get_region(0, m, t, z, 0, 0, 4, 0, 4).mean()
                              for z in zs]))

    sub_e, sub_ds = run(sub)
    assert int(sub_ds.get(Domain.GLOBAL, "units").values) == 3      # not 15, not 1
    assert int(sub_ds.get(Domain.GLOBAL, "planes").values) == 6     # whole volume
    # …and they are the RIGHT three frames, in order
    assert np.allclose(np.asarray(sub_ds.get(Domain.FRAME, "mean").values),
                       [[vol_mean(0, t, range(6)) for t in (1, 3, 4)]])
    # all three axes pick independently → the subset is their cross product
    both_e, both = run(FrameSubsetProvider(base, (0, 2), (1, 3, 4), (0, 5)))
    assert int(both.get(Domain.GLOBAL, "units").values) == 6        # 2 positions × 3 times
    assert int(both.get(Domain.GLOBAL, "planes").values) == 2       # …of 2 planes each
    assert np.asarray(both.get(Domain.FRAME, "mean").values).shape == (2, 3)
    assert np.allclose(np.asarray(both.get(Domain.FRAME, "mean").values),
                       [[vol_mean(m, t, (0, 5)) for t in (1, 3, 4)] for m in (0, 2)])
    assert (both_e.env("N").axes.m, both_e.env("N").axes.t,
            both_e.env("N").axes.z) == (2, 3, 2)
    # the envelope follows the payload (build-node-v2 §2: axes agree)
    assert sub_e.env("N").axes.t == 3 and sub_e.env("N").axes.z == 6

    # the memo fence: one selection never serves another — including two that differ only
    # in z — and revisiting one is a HIT
    hashes = {run(FrameSubsetProvider(base, (0,), ts, zs))[0].entry("N").recipe_hash
              for ts, zs in (((1, 3, 4), None), ((1, 3), None), ((0, 3, 4), None),
                             ((1, 2, 4), None), ((1, 3, 4), (0, 1)), ((1, 3, 4), (0, 2)))}
    assert len(hashes) == 6
    again_e, _again = run(FrameSubsetProvider(base, (0,), (3, 4, 1)))   # order-insensitive
    assert (again_e.entry("N").recipe_hash == sub_e.entry("N").recipe_hash
            and again_e.compute_count == 0)
    _ok("frame subset: a picked (ms,ts,zs) source view — 15 units → 3 (the picked frames' "
        "own pixels, in acquisition order) and → a 2×3 cross product of 2-plane volumes; "
        "z picks remap inside every scoped frame (gathered subvolumes included) without "
        "disturbing them and an unset z keeps the pre-z fingerprint; unpicked cursors "
        "address the nearest picked index, an unpicked axis passes through, picks clamp "
        "to the source, and each selection fences the memo (revisit is a hit, "
        "order-insensitive)")


# ── structure producers: CCL / watershed / point schema (V2.02 §7 / V2.03) ────

def test_structure() -> None:
    # connectivity offsets: the toggle-dependent param set
    assert len(connectivity_offsets(2, 4)) == 4 and len(connectivity_offsets(2, 8)) == 8
    for conn, k in ((6, 6), (18, 18), (26, 26)):
        assert len(connectivity_offsets(3, conn)) == k
    try:
        connectivity_offsets(2, 6)
        raise AssertionError("expected invalid-connectivity error")
    except ValueError:
        pass

    # 2D CCL: two diagonally-touching pixels → 2 regions (4-conn) vs 1 (8-conn)
    m2 = np.zeros((5, 5), int); m2[1, 1] = 1; m2[2, 2] = 1
    lab4, t4 = label_components(m2, 4)
    _, t8 = label_components(m2, 8)
    assert t4.n == 2 and t8.n == 1 and t4.z_kind == "plane_index"
    # separated pair: areas + raster-canonical ids (top-left region is id 1)
    m2b = np.zeros((5, 6), int); m2b[0:2, 0:2] = 1; m2b[3:5, 4:6] = 1
    lab, tb = label_components(m2b, 4)
    assert tb.n == 2 and sorted(tb.columns["area"].tolist()) == [4, 4] and lab[0, 0] == 1

    # 3D CCL: two separated cubes; real subpixel centroid z
    m3 = np.zeros((4, 4, 4), int); m3[0:2, 0:2, 0:2] = 1; m3[2:4, 2:4, 2:4] = 1
    _, t3 = label_components(m3, 6)
    assert t3.n == 2 and t3.z_kind == "subpixel"
    assert sorted(t3.columns["area"].tolist()) == [8, 8]
    zc = sorted(t3.columns["z"].tolist())
    assert abs(zc[0] - 0.5) < 1e-9 and abs(zc[1] - 2.5) < 1e-9

    # C6 fast CCL: the scipy.ndimage.label fast path is BYTE-IDENTICAL to the reference
    # pure-numpy flood-fill (same partition; raster-canonical relabel to first-appearance
    # C-order) — raster + id/area/centroid columns match exactly, across connectivities.
    from nodegraph.structure import _label_components_flood
    rng = np.random.default_rng(0)
    for f, conns in ((rng.random((48, 48)) > 0.45, (4, 8)),
                     (rng.random((14, 14, 14)) > 0.4, (6, 26))):
        for conn in conns:
            ls, ts = label_components(f, conn)                    # scipy fast path
            lf, tf = _label_components_flood(np.asarray(f) != 0, conn,
                                             m=0, t=0, c=0, z_index=0, layer=None)
            assert np.array_equal(ls, lf), f"scipy≠flood raster ({f.ndim}D {conn}-conn)"
            assert ts.columns["id"].tolist() == tf.columns["id"].tolist()
            assert ts.columns["area"].tolist() == tf.columns["area"].tolist()
            for ax in ("z", "y", "x"):
                assert np.allclose(ts.columns[ax], tf.columns[ax]), f"{ax} centroid drift"
    # empty mask → no regions, all-background raster (both paths)
    le, te = label_components(np.zeros((6, 6), int), 4)
    assert te.n == 0 and int(le.max()) == 0

    # point table: invariant schema; 2D → plane-index z (never NaN), 3D → subpixel
    p2 = point_table(np.array([[1.5, 2.5], [3.5, 4.5]]), z=7, t=1)
    assert p2.z_kind == "plane_index" and p2.n == 2
    assert set(("id", "m", "t", "c", "z", "y", "x")) <= set(p2.columns)
    assert np.allclose(p2.columns["z"], 7.0) and np.all(np.isfinite(p2.columns["z"]))
    p3 = point_table(np.array([[0.5, 1.0, 2.0]]))
    assert p3.z_kind == "subpixel" and p3.columns["z"][0] == 0.5

    # content hash: stable + data-sensitive
    h = t4.content_hash()
    assert t4.content_hash() == h and t8.content_hash() != h

    if _HAVE_PYARROW:
        rb = t3.to_arrow()
        assert rb.num_rows == 2 and {"id", "area", "z", "y", "x"} <= set(rb.schema.names)
        _ok(f"structure: CCL 2D/3D (scipy fast path ≡ flood-fill, C6) + point schema + "
            f"content-hash + Arrow (pyarrow {_PA_VER})")
    else:
        _ok("structure: CCL 2D/3D (scipy fast path ≡ flood-fill, C6) + point schema + "
            "content-hash (pyarrow SKIPPED)")

    # id-carrying seeded watershed: output ids ARE the marker ids (stable over t)
    if _HAVE_WATERSHED:
        fg = np.zeros((3, 9), int); fg[1, :] = 1
        markers = np.zeros((3, 9), int); markers[1, 0] = 5; markers[1, 8] = 9
        ws = seeded_watershed(fg, markers)
        assert set(np.unique(ws[fg > 0]).tolist()) == {5, 9}
        assert ws[1, 0] == 5 and ws[1, 8] == 9
        _ok("structure: id-carrying seeded watershed splits by marker id")
    else:
        _ok("structure: seeded watershed SKIPPED (scipy/skimage absent)")


# ── structure-bridge execution (the geometric spine, V2.00 §6) ────────────────

def test_bridges() -> None:
    raster = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 0],
                       [3, 3, 3, 0]], dtype=np.int64)
    vox = np.array([[10, 10, 0, 20],
                    [10, 10, 0, 20],
                    [0, 0, 0, 0],
                    [30, 30, 30, 0]], dtype=float)
    # Voxel → Label: mean-in-mask / sum / count
    ids, means = voxel_to_label(vox, raster, "mean")
    assert ids.tolist() == [1, 2, 3] and np.allclose(means, [10, 20, 30])
    assert np.allclose(voxel_to_label(vox, raster, "sum")[1], [40, 40, 90])
    assert np.allclose(voxel_to_label(vox, raster, "count")[1], [4, 2, 3])
    # Label → Voxel: paint-by-label; background stays 0
    painted = label_to_voxel([1, 2, 3], [100, 200, 300], raster)
    assert painted[0, 0] == 100 and painted[0, 3] == 200 and painted[3, 0] == 300
    assert painted[0, 2] == 0.0 and painted[2, 2] == 0.0
    # round-trip: paint the means back, re-reduce → identical
    assert np.allclose(voxel_to_label(label_to_voxel(ids, means, raster), raster, "mean")[1],
                       means)
    # Voxel → Point (nearest) + containing label
    pts = np.array([[0, 0], [0, 3], [3, 1]], dtype=float)     # (y,x)
    assert np.allclose(voxel_to_point(vox, pts, "nearest"), [10, 20, 30])
    assert containing_label(pts, raster).tolist() == [1, 2, 3]
    assert containing_label(np.array([[2, 2]], dtype=float), raster)[0] == 0   # bg
    # Point → Voxel splat (collisions sum)
    sv = point_to_voxel([5, 7], np.array([[0, 0], [0, 0]], dtype=float), (2, 2), "sum")
    assert sv[0, 0] == 12.0
    # Point → Label: reduce points inside each region
    idp, mp = points_in_label([1.0, 2.0, 3.0, 4.0],
                              np.array([[0, 0], [0, 1], [0, 3], [3, 0]], dtype=float),
                              raster, "mean")
    assert idp.tolist() == [1, 2, 3] and np.allclose(mp, [1.5, 3.0, 4.0])
    # a bridge plan still refuses generic execution (needs structure inputs)
    from nodegraph.transfer import execute_transfer, plan_transfer
    from nodegraph.dataset import AttributeLayer, AxisSizes
    try:
        execute_transfer(AttributeLayer(D.VOXEL, "i",
                         np.zeros(AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2).shape_for(D.VOXEL))),
                         plan_transfer(D.VOXEL, D.LABEL), AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2))
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    if _HAVE_WATERSHED:                                        # scipy present → linear
        vl = voxel_to_point(vox, np.array([[0.0, 0.5]]), "linear")
        assert abs(vl[0] - 10.0) < 1e-9
    _ok("bridges: voxel↔label (mean/sum/count/paint/round-trip), voxel↔point, point↔label")


# ── fields: deferred per-element expressions (V2.00 §3.3 / V2.02 §9) ───────────

def test_field() -> None:
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=3, x=4)
    fr = np.array([[10.0, 20.0]])                     # Frame (m=1, t=2)
    ds = Dataset(axes=ax).with_layer(D.FRAME, "val", fr)
    fctx = FieldContext(ds, D.FRAME, ax)

    # Const → VirtualArray; a const-only subtree stays virtual (V2.02 §9)
    c = evaluate(Const(5.0), fctx)
    assert isinstance(c, VirtualArray) and c.shape == (1, 2) and np.allclose(np.asarray(c), 5.0)
    cv = evaluate(BinOp("*", Const(2.0), Const(3.0)), fctx)
    assert isinstance(cv, VirtualArray) and np.allclose(np.asarray(cv), 6.0)

    # Attr read + arithmetic + comparison + where
    assert np.allclose(evaluate(Attr(D.FRAME, "val"), fctx), fr)
    expr = BinOp("+", BinOp("*", Attr(D.FRAME, "val"), Const(2.0)), Const(1.0))
    assert np.allclose(evaluate(expr, fctx), fr * 2 + 1)
    assert evaluate(BinOp(">", Attr(D.FRAME, "val"), Const(15.0)), fctx).tolist() == [[False, True]]
    w = evaluate(Where(BinOp(">", Attr(D.FRAME, "val"), Const(15.0)), Const(1.0), Const(0.0)), fctx)
    assert np.allclose(np.asarray(w), [[0.0, 1.0]])

    # Attr transferred across the lattice: Frame val consumed on Voxel → broadcast
    rv = evaluate(Attr(D.FRAME, "val"), FieldContext(ds, D.VOXEL, ax))
    assert rv.shape == ax.shape_for(D.VOXEL)
    for t in range(ax.t):
        assert np.allclose(rv[:, t], fr[0, t])

    # Input binding
    assert np.allclose(evaluate(Input("x"), FieldContext(ds, D.FRAME, ax, inputs={"x": fr})), fr)
    try:
        evaluate(Input("missing"), fctx)
        raise AssertionError("expected unbound-input error")
    except KeyError:
        pass

    # field_expr_hash tracks layer revision; kernel_axes disjoins 2D vs 3D stencils
    h1 = field_expr_hash(expr, ds)
    assert field_expr_hash(expr, ds.with_layer(D.FRAME, "val", fr * 10)) != h1
    assert (field_key(expr, D.FRAME, 0, dataset=ds, kernel_axes=("y", "x"))
            != field_key(expr, D.FRAME, 0, dataset=ds, kernel_axes=("z", "y", "x")))

    # disjoint field cache: same token hits, new token misses
    fc = FieldCache()
    fc.evaluate(expr, fctx, token=1); assert fc.misses == 1
    fc.evaluate(expr, fctx, token=1); assert fc.hits == 1
    fc.evaluate(expr, fctx, token=2); assert fc.misses == 2
    _ok("field: IR eval, VirtualArray, lattice transfer, expr-hash on revision, disjoint key + cache")


# ── track membership + track bridges (temporal identity, V2.00 §6) ────────────

def test_tracks() -> None:
    from nodegraph.structure import TrackMembership
    from nodegraph.bridges import (broadcast_track, gather_by_track,
                                    timepoint_to_members, tracks_per_timepoint)
    from nodegraph.transfer import plan_transfer
    # track 1: labels 10→11→12 over t0,1,2 ; track 2: labels 20→21 over t0,1 (dies at t2)
    mem = TrackMembership(track_id=[1, 1, 1, 2, 2], t=[0, 1, 2, 0, 1],
                          member_id=[10, 11, 12, 20, 21], member_domain=D.LABEL)
    assert mem.n == 5 and mem.track_ids().tolist() == [1, 2] and mem.timepoints().tolist() == [0, 1, 2]
    mids, mvals = [10, 11, 12, 20, 21], [1.0, 2.0, 3.0, 100.0, 200.0]
    # Label → Track: gather members over t and reduce
    tids, means = gather_by_track(mids, mvals, mem, "mean")
    assert tids.tolist() == [1, 2] and np.allclose(means, [2.0, 150.0])
    assert np.allclose(gather_by_track(mids, mvals, mem, "sum")[1], [6.0, 300.0])
    assert np.allclose(gather_by_track(mids, mvals, mem, "count")[1], [3, 2])
    # Track → Label: broadcast a per-track value onto its members
    bm, bv = broadcast_track([1, 2], [2.0, 150.0], mem)
    assert bm.tolist() == [10, 11, 12, 20, 21] and np.allclose(bv, [2, 2, 2, 150, 150])
    # Track → Timepoint: count active tracks per t, and reduce a track attr over them
    tp, cnt = tracks_per_timepoint(mem)
    assert tp.tolist() == [0, 1, 2] and np.allclose(cnt, [2, 2, 1])
    _, red = tracks_per_timepoint(mem, track_ids=[1, 2], track_values=[2.0, 150.0], reducer="mean")
    assert np.allclose(red, [76.0, 76.0, 2.0])
    # Timepoint → Track members: broadcast a per-t value onto each member at t
    tmid, tmv = timepoint_to_members([10.0, 20.0, 30.0], [0, 1, 2], mem)
    assert tmid.tolist() == [10, 11, 12, 20, 21] and np.allclose(tmv, [10, 20, 30, 10, 20])
    # the plan router still marks Label→Track as a (non-generated) bridge plan
    assert not plan_transfer(D.LABEL, D.TRACK).generated
    _ok("tracks: membership + gather/broadcast/per-timepoint/timepoint→members bridges")


# ── constructive Fill/Extract Boundary (Point↔Label spine, V2.03 §4 C4) ───────

def test_boundary() -> None:
    from nodegraph.boundary import boundary_dim, extract_boundary, fill_boundary
    from nodegraph.transfer import bridges as _bridges

    # dimensionality authority = geometry (single-z ⇒ 2D contour; multi-z ⇒ 3D surface)
    sq = np.array([[1, 1], [1, 5], [5, 5], [5, 1]], dtype=float)      # (y,x) corners
    assert boundary_dim(sq) == "2D"
    assert boundary_dim(np.array([[0, 1, 1], [0, 5, 5], [0, 5, 1]], dtype=float)) == "2D"
    assert boundary_dim(np.array([[0, 1, 1], [2, 5, 5]], dtype=float)) == "3D"

    if not _HAVE_SKIMAGE:
        _ok("boundary: geometry dim-authority (fill/extract SKIPPED — skimage absent)")
        return

    # Fill: a square contour → a filled region; Extract: a region → its outline points
    raster = fill_boundary(sq, (7, 7))
    assert raster[3, 3] == 1 and raster[0, 0] == 0 and raster.sum() > 0
    mask = np.zeros((8, 8), dtype=int); mask[2:6, 2:6] = 1
    pts = extract_boundary(mask)
    assert pts.domain is D.POINT and pts.z_kind == "plane_index" and pts.n > 0
    assert "contour_id" in pts.columns

    # the toggle is a consistency ASSERTION, not an override: contradiction → hard error
    try:
        fill_boundary(sq, (7, 7), dim="3D")
        raise AssertionError("expected dim-contradiction error")
    except ValueError:
        pass

    # 3D: multi-z contours fill per-plane (the 2D-ROI-on-3D case fills only its plane)
    zpts = np.array([[0, 1, 1], [0, 1, 4], [0, 4, 4], [0, 4, 1],
                     [2, 1, 1], [2, 1, 4], [2, 4, 4], [2, 4, 1]], dtype=float)
    vol = fill_boundary(zpts, (3, 6, 6))
    assert vol[0].sum() > 0 and vol[2].sum() > 0 and vol[1].sum() == 0
    v = np.zeros((6, 6, 6), dtype=int); v[2:4, 2:4, 2:4] = 1
    surf = extract_boundary(v)
    assert surf.z_kind == "subpixel" and surf.n > 0

    # explicit-node contract (V2.03 §4 C4): Fill/Extract are NOT auto-routed bridges,
    # while the Point↔Label CONTAINMENT bridges remain registered
    names = {b.name for b in _bridges()}
    assert "fill-boundary" not in names and "extract-boundary" not in names
    assert "points-in-label" in names and "containing-label" in names
    _ok("boundary: fill/extract contour↔region, geometry-authority, toggle-assert, not a bridge")


# ── node port: metadata-intelligent PSF + Select Channel + Deconvolve (Phase 3) ─

def test_nodes() -> None:
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, diffraction_sigmas, gaussian_psf, to_pixels_v2

    # unit conversion incl. the axial unit (V2.03 §2 A5)
    assert to_pixels_v2(1.0, "um", pixel_size_um=0.1) == 10.0
    assert to_pixels_v2(1.0, "um_axial", z_step_um=0.5) == 2.0
    assert abs(to_pixels_v2(520, "nm", pixel_size_um=0.13) - 0.52 / 0.13) < 1e-9
    # PSF derives from optics: 2D → 2 sigmas, 3D → 3; smaller NA ⇒ broader PSF
    s2 = diffraction_sigmas(520, 1.4, 0.1, 0.3, False)
    s3 = diffraction_sigmas(520, 1.4, 0.1, 0.3, True)
    assert len(s2) == 2 and len(s3) == 3
    assert diffraction_sigmas(520, 0.7, 0.1, 0.3, False)[0] > s2[0]
    psf = gaussian_psf(s3)
    assert psf.ndim == 3 and abs(psf.sum() - 1.0) < 1e-9

    if not _HAVE_SKIMAGE:
        _ok("nodes: unit-conv + metadata-PSF (Select/Deconvolve SKIPPED — skimage absent)")
        return

    define_node("io.seed", "Seed", outputs=[OutDataset()])          # trivial source

    # Select Channel (H12): subset a 2-channel dataset to channel 1
    ax2 = AxisSizes(m=1, t=1, z=1, c=2, y=4, x=4)
    vol2 = np.zeros((1, 1, 1, 2, 4, 4)); vol2[..., 0, :, :] = 1.0; vol2[..., 1, :, :] = 9.0
    ds2 = Dataset(axes=ax2, metadata={"channel_emission_nm": [500, 600]}).with_image(ArrayProvider(vol2))
    gsel = Graph()
    gsel.add(NodeInstance("S", "io.seed")); gsel.add(NodeInstance("C", "channel.select", params={"channels": [1]}))
    gsel.connect("S", "C")
    esel = Engine(gsel, computes=COMPUTES, seeds={"S": ds2},
                  meta_seeds={"S": MetaEnvelope(axes=ax2, metadata={"channel_emission_nm": [500, 600]})})
    selds = esel.pull("C")
    assert selds.axes.c == 1
    assert selds.image.read_region(0, 0, 0, 0, 0, 0, 4, 0, 4)[0, 0] == 9.0     # channel 1's data
    assert selds.metadata["channel_emission_nm"] == [600]

    # Deconvolve: end-to-end; 2D vs 3D use different PSF dims + distinct memo keys.
    # Signal on plane 0 only + a broad (low-NA) PSF, so the 3D volumetric PSF couples
    # z-planes (spreads into empty planes) while the 2D per-plane path leaves them empty.
    ax = AxisSizes(m=1, t=1, z=4, c=1, y=8, x=8)
    vol = np.zeros((1, 1, 4, 1, 8, 8)); vol[0, 0, 0, 0, 3:6, 3:6] = 1.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 0.8, "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(vol))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    def deco(dim):
        g = Graph()
        g.add(NodeInstance("S", "io.seed"))
        g.add(NodeInstance("D", "enhance.deconvolve", modes={"dim": dim}, params={"iterations": 2}))
        g.connect("S", "D")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    e3 = deco("3D"); out3 = e3.pull("D")
    assert isinstance(out3, Dataset) and out3.image.axes.z == 4 and out3.image.axes.y == 8
    e2 = deco("2D"); out2 = e2.pull("D")
    assert out2.image.axes.z == 4                                      # 2D mode also runs
    assert e2.entry("D").recipe_hash != e3.entry("D").recipe_hash      # lever re-keys memo
    dec3 = out3.image.get_region_volume(0, 0, 0, 0, 0, 4, 0, 8, 0, 8)
    assert np.all(np.isfinite(dec3)) and np.all(dec3 >= 0)            # RL invariant
    # the lever changes the resolved footprint + socket set on the flagship spec
    dspec = NODES.get("enhance.deconvolve")
    assert dspec.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert dspec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert dspec.resolve_kernel_axes({"dim": "3D"}) == frozenset({"z", "y", "x"})
    a2 = {s.name for s in dspec.active_inputs({"dim": "2D"})}
    a3 = {s.name for s in dspec.active_inputs({"dim": "3D"})}
    assert "z_step_um" in a3 and "z_step_um" not in a2                 # 3D-only axial socket

    # ── V2.19: the separable RL IS the PSF-array RL ─────────────────────────────
    # The node no longer convolves with the n-D kernel — it exploits the fact that its PSF
    # is always a Gaussian, hence separable and self-mirroring, so both blurs in an RL step
    # are per-axis passes. That is a claim about EQUALITY, not about closeness, and it is
    # the only thing standing between "20x faster" and "quietly a different deconvolution",
    # so it is asserted against skimage directly, at the real optics.
    from nodegraph.nodes import _gauss_blur_zero, _rl, _rl_gaussian
    from nodegraph.streaming import realize
    from scipy.signal import convolve as _sconv
    _rng = np.random.default_rng(11)
    for _shape, _sig in (((24, 26), (0.334, 0.334)),
                         ((9, 20, 22), (1.12, 0.61, 0.61)),
                         ((11, 18, 18), (3.561, 0.606, 0.606)),   # the lab's NA 0.8 / 663 nm
                         ((5, 7, 7), (0.4, 0.4, 0.4))):           # below the sigma clamp
        _img = _rng.random(_shape) * 900 + 60
        _img[tuple(s // 2 for s in _shape)] = 4000.0              # a hot voxel to sharpen
        _psf = gaussian_psf(_sig)
        _cl = tuple(max(0.5, float(v)) for v in _sig)
        _rd = tuple(max(1, int(round(3.0 * v))) for v in _cl)
        # one blur first: the separable pass == scipy's n-D zero-padded convolution
        _b = _gauss_blur_zero(_img.copy(), _cl, _rd)
        _w = _sconv(_img, _psf, mode="same")
        assert np.allclose(_b, _w, rtol=1e-11, atol=1e-11 * float(np.max(_w))), \
            (_shape, _sig, float(np.max(np.abs(_b - _w))))
        for _it in (1, 3, 10):
            _want = _rl(_img.copy(), _psf, _it)
            _got = _rl_gaussian(_img.copy(), _sig, _it)
            _rel = float(np.max(np.abs(_got - _want)) / max(1e-30, np.max(np.abs(_want))))
            assert _rel < 1e-9, (_shape, _sig, _it, _rel)
    # dtype is preserved, so NODEGRAPH_FLOAT32 reaches this kernel instead of being upcast
    assert _rl_gaussian(np.ones((6, 12, 12), dtype=np.float32), (1.1, 0.6, 0.6),
                        3).dtype == np.float32
    # 3D RL actually deblurs along Z: a point emitter's axial FWHM must NARROW
    from scipy.ndimage import gaussian_filter as _gf
    _t = np.zeros((21, 32, 32)); _t[10, 16, 16] = 1.0
    _sg = (2.5, 0.8, 0.8)
    _bl = _gf(_t, sigma=_sg, mode="constant", cval=0.0)
    _dc = _rl_gaussian(_bl.copy(), _sg, 30)

    def _fwhm(v):
        col = v[:, 16, 16]
        idx = np.flatnonzero(col >= col.max() / 2.0)
        return int(idx[-1] - idx[0] + 1)

    assert _fwhm(_dc) < _fwhm(_bl), (_fwhm(_bl), _fwhm(_dc))
    assert abs(_dc.sum() - _bl.sum()) / _bl.sum() < 0.05          # RL conserves flux
    # …and an isotropic sigma recovers z WORSE, i.e. the axial term is load-bearing
    assert _fwhm(_rl_gaussian(_bl.copy(), (0.8, 0.8, 0.8), 30)) > _fwhm(_dc)
    # a plane-by-plane pull of the lazy volume == a full realize (the C1 promise, on the
    # node whose unit is a whole volume)
    _e = deco("3D")
    _p = _e.pull("D").image
    _streamed = np.stack([_p.get_region(0, 0, 0, _z, 0, 0, 8, 0, 8) for _z in range(4)])
    _real = realize(deco("3D").pull("D")).image
    assert np.array_equal(_streamed, np.stack(
        [_real.get_region(0, 0, 0, _z, 0, 0, 8, 0, 8) for _z in range(4)]))
    # an all-zero volume must not escape as NaN through the RL ratio
    _zds = Dataset(axes=ax, metadata=optics).with_image(
        ArrayProvider(np.zeros((1, 1, 4, 1, 8, 8))))
    _gz = Graph(); _gz.add(NodeInstance("S", "io.seed"))
    _gz.add(NodeInstance("D", "enhance.deconvolve", modes={"dim": "3D"}))
    _gz.connect("S", "D")
    assert np.all(np.isfinite(Engine(_gz, computes=COMPUTES, seeds={"S": _zds},
                                     meta_seeds={"S": seedenv}).pull("D").image
                              .get_region_volume(0, 0, 0, 0, 0, 4, 0, 8, 0, 8)))
    _ok("nodes: metadata-PSF, Select Channel (H12), Deconvolve 2D/3D lever end-to-end; "
        "V2.19 separable RL — one blur matches scipy's n-D zero-padded convolution and the "
        "whole iteration matches skimage's PSF-array RL to <1e-9 relative on 4 sigma sets "
        "(incl. the lab's NA 0.8/663 nm and one below the sigma clamp) at 1/3/10 iterations, "
        "float32 survives the kernel, a point emitter's axial FWHM narrows while flux is "
        "conserved and an isotropic sigma does WORSE, a plane-by-plane pull equals a "
        "realize, and an all-zero volume stays finite")


# ── ported catalog: filters + threshold→label→measure pipeline (Phase 3) ──────

def test_catalog() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=2, c=1, y=6, x=6)
    img = np.zeros((1, 1, 2, 1, 6, 6))
    img[0, 0, :, 0, 1:3, 1:3] = 5.0            # blob A (both z-planes)
    img[0, 0, :, 0, 4:6, 4:6] = 9.0            # blob B (both z-planes)
    ds = Dataset(axes=ax, metadata={"pixel_size_um": 0.1, "z_step_um": 0.3}).with_image(ArrayProvider(img))
    define_node("io.seed2", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata={"pixel_size_um": 0.1, "z_step_um": 0.3})

    def eng(nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for s, d in edges:
            g.connect(s, d)
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    # Gamma (pointwise, no lever): max preserved, mid-tones compressed
    eg = eng([("S", "io.seed2", {}), ("G", "enhance.gamma", {"params": {"gamma": 2.0}})],
             [("S", "G")])
    a = eg.pull("G").image.get_region(0, 0, 0, 0, 0, 0, 6, 0, 6)
    assert a.shape == (6, 6) and a[1, 1] < 5.0 and a[4, 4] == 9.0

    # Gaussian blur (toggle): blurs the plane (differs from input)
    eb = eng([("S", "io.seed2", {}),
              ("B", "enhance.gaussian", {"modes": {"dim": "2D"}, "params": {"sigma": 0.2}})],
             [("S", "B")])
    blur = eb.pull("B").image.get_region(0, 0, 0, 0, 0, 0, 6, 0, 6)
    assert not np.allclose(blur, img[0, 0, 0, 0])

    # threshold → label → measure pipeline
    ep = eng([("S", "io.seed2", {}),
              ("T", "analysis.threshold", {"params": {"threshold": 1.0}}),
              ("L", "analysis.label", {"modes": {"dim": "2D"}}),
              ("M", "analysis.measure", {})],
             [("S", "T"), ("T", "L"), ("L", "M")])
    assert ep.pull("T").get(D.VOXEL, "mask") is not None          # threshold → Voxel mask
    lds = ep.pull("L")
    area = lds.get(D.LABEL, "area", layer="labels")
    assert area is not None and len(area.values) == 4            # 2 blobs × 2 planes
    mds = ep.pull("M")
    mi = mds.get(D.LABEL, "mean_intensity", layer="labels")
    assert mi is not None
    means = set(np.round(mi.values).tolist())
    assert 5.0 in means and 9.0 in means                        # per-label mean intensity
    # §7b: label stamped z_kind=plane_index (2D); measure re-emits the same layer and must
    # PRESERVE it, not clobber with the StructureTable default (subpixel).
    assert lds.structure_zkind(D.LABEL, "labels") == "plane_index"
    assert mds.structure_zkind(D.LABEL, "labels") == "plane_index", "measure clobbered z_kind"
    _ok("catalog: gamma (pointwise), gaussian (toggle), threshold→label→measure pipeline "
        "(+ z_kind preserved through measure)")


# ── ported catalog batch: filters / denoise / detection / axis-changing ───────

def test_catalog_ported() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (ported batch): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, SAMPLING_KEY

    # a z=3 volume with two bright 3×3 blobs on every plane (detectable + filterable)
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=16, x=16)
    img = np.zeros((1, 1, 3, 1, 16, 16), dtype=float)
    for (yy, xx) in [(4, 4), (11, 11)]:
        img[0, 0, :, 0, yy - 1:yy + 2, xx - 1:xx + 2] = 8.0
        img[0, 0, :, 0, yy, xx] = 12.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 1.0,
              "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.seed3", "Seed", outputs=[OutDataset()])

    def eng(op, *, modes=None, params=None):
        g = Graph()
        g.add(NodeInstance("S", "io.seed3"))
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    def img_of(dset):
        p = dset.image
        return p.get_region_volume(0, 0, 0, 0, 0, p.axes.z, 0, p.axes.y, 0, p.axes.x)

    # every IMAGE→IMAGE filter runs end-to-end, stays finite + same geometry, in 2D & 3D
    filters = ["enhance.median", "enhance.morphology", "enhance.tophat", "enhance.dog",
               "enhance.unsharp", "enhance.tv_denoise", "enhance.wavelet_denoise",
               "enhance.clahe"]
    for op in filters:
        for dim in ("2D", "3D"):
            out = eng(op, modes={"dim": dim}).pull("N")
            arr = img_of(out)
            assert arr.shape == (3, 16, 16), f"{op}/{dim} geometry {arr.shape}"
            assert np.all(np.isfinite(arr)), f"{op}/{dim} non-finite"
        # the 2D/3D lever re-keys the memo distinctly (params fold in the mode state)
        assert (eng(op, modes={"dim": "2D"}).entry("N").recipe_hash
                != eng(op, modes={"dim": "3D"}).entry("N").recipe_hash), f"{op} lever hash"

    # median actually changes the image (a real filter, not a pass-through)
    med = img_of(eng("enhance.median", modes={"dim": "2D"},
                     params={"radius": 0.15}).pull("N"))
    assert not np.allclose(med, img[0, 0, :, 0])

    # wavelet_denoise declares the stack-of-2D trap honestly (never true-3D)
    wspec = NODES.get("enhance.wavelet_denoise")
    assert wspec.supports_true_3d is False and wspec.three_d_fallback == "stack_of_2d"
    assert wspec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_PLANE
    # tv_denoise, by contrast, is genuinely volumetric in 3D
    assert NODES.get("enhance.tv_denoise").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_VOLUME

    # morphology/tophat modes fold into the recipe hash (distinct ops memoize apart)
    assert (eng("enhance.morphology", modes={"op": "erode"}).entry("N").recipe_hash
            != eng("enhance.morphology", modes={"op": "dilate"}).entry("N").recipe_hash)

    # Normalize: percentile → [0,1]; scope is a Mode, NOT the dim lever
    nspec = NODES.get("enhance.normalize")
    assert not nspec.has_dim_lever() and nspec.dim_lever() is None
    for scope in ("plane", "volume", "series"):
        nrm = img_of(eng("enhance.normalize", modes={"scope": scope}).pull("N"))
        assert nrm.min() >= 0.0 and nrm.max() <= 1.0 + 1e-9, f"normalize/{scope} range"
    assert (eng("enhance.normalize", modes={"scope": "plane"}).entry("N").recipe_hash
            != eng("enhance.normalize", modes={"scope": "volume"}).entry("N").recipe_hash)

    # Spot detection → a Point structure layer (both dims find the two blobs)
    for dim, zk in (("2D", "plane_index"), ("3D", "subpixel")):
        sp = eng("detect.spots", modes={"dim": dim},
                 params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05}).pull("N")
        ys = sp.get(D.POINT, "y", layer="spots")
        assert ys is not None and len(ys.values) >= 1, f"spots/{dim} found none"
        assert sp.get(D.POINT, "z", layer="spots") is not None            # invariant schema

    # Z-Project: axis-changing z→1, drops z_step_um + marks z_collapsed, env tracks it
    ez = eng("util.zproject", params={})
    zout = ez.pull("N")
    assert zout.axes.z == 1 and zout.image.axes.z == 1
    assert "z_step_um" not in zout.metadata and zout.metadata.get("z_collapsed") is True
    assert ez.env("N").axes.z == 1                                        # meta_transform pass
    zmax = img_of(zout)[0]                                                # the single plane
    assert zmax.max() == 12.0                                             # max projection

    # Z-Project method="none" is the RESET (V2.21): the stack comes back untouched, and the
    # meta_transform's prediction agrees with the payload just as it does for a reducer.
    ezn = eng("util.zproject", modes={"method": "none"})
    znone = ezn.pull("N")
    assert znone.axes.z == 3 and znone.image.axes.z == 3, "reset must keep every z plane"
    assert np.allclose(img_of(znone), img[0, 0, :, 0]), "reset must not touch the voxels"
    assert znone.metadata.get("z_step_um") == 0.3, "reset must keep z_step_um"
    assert "z_collapsed" not in znone.metadata, "reset must not claim a collapse"
    assert ezn.env("N").axes.z == 3 and ezn.env("N").metadata.get("z_step_um") == 0.3
    assert "z_collapsed" not in ezn.env("N").metadata, "reset header must match payload"
    # geometry provenance untouched — a downstream `raw` intensity wire must still match
    assert not znone.metadata.get(SAMPLING_KEY), "reset must not stamp __sampling__"
    assert zout.metadata.get(SAMPLING_KEY), "a real projection DOES stamp it"
    # the reset memoizes apart from every reducer, and declares the honest cheap footprint
    zspec = NODES.get("util.zproject")
    assert zspec.footprint_mode == "method", "zproject's footprint follows its method"
    assert zspec.resolve_granularity({"method": "none"}) is Granularity.TILEABLE
    assert zspec.resolve_kernel_axes({"method": "none"}) == frozenset()
    hashes = {m: eng("util.zproject", modes={"method": m}).entry("N").recipe_hash
              for m in ("max", "mean", "sum", "min", "median", "none")}
    assert len(set(hashes.values())) == 6, f"zproject method hashes collide: {hashes}"
    for m in ("max", "mean", "sum", "min", "median"):
        assert zspec.resolve_granularity({"method": m}) is Granularity.WHOLE_VOLUME
        assert zspec.resolve_kernel_axes({"method": m}) == frozenset({"z"})
        em = eng("util.zproject", modes={"method": m})
        assert em.pull("N").axes.z == em.env("N").axes.z == 1, f"zproject/{m} header"

    # Crop: shrink Y,X (2D) — payload + edit-time envelope agree on the new extent
    ec = eng("util.crop", modes={"dim": "2D"},
             params={"y0": 2, "y1": 10, "x0": 3, "x1": 12})
    cout = ec.pull("N")
    assert cout.axes.y == 8 and cout.axes.x == 9 and cout.axes.z == 3
    assert ec.env("N").axes.y == 8 and ec.env("N").axes.x == 9
    # 3D crop also trims Z via the 3D-only sockets
    c3 = eng("util.crop", modes={"dim": "3D"},
             params={"y0": 0, "y1": 8, "x0": 0, "x1": 8, "z0": 1, "z1": 3}).pull("N")
    assert c3.axes.z == 2 and c3.axes.y == 8

    # ── adversarial-review regression guards (2026-07-21 ported-catalog review) ──

    # R1 detect.spots 3D reads z_step_um (anisotropic σ) → memo-fenced on it, and the
    #    axial scale changes results (isotropic-vs-anisotropic must diverge somewhere).
    e3d = eng("detect.spots", modes={"dim": "3D"},
              params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05})
    e3d.pull("N")
    assert "z_step_um" in dict(e3d.entry("N").reads), "spots 3D must read z_step_um"
    e2d = eng("detect.spots", modes={"dim": "2D"},
              params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05})
    e2d.pull("N")
    assert "z_step_um" not in dict(e2d.entry("N").reads), "spots 2D must NOT read z_step_um"

    # R2 crop header (meta_transform env) == payload axes for one-sided & out-of-range
    #    bounds (previously span() ignored single endpoints and never clamped).
    for params, exp_y in [({"y0": 2}, 14), ({"y1": 10}, 10),
                          ({"y0": 2, "y1": 100}, 14), ({"y0": -5, "y1": 10}, 10)]:
        crop_e = eng("util.crop", modes={"dim": "2D"}, params=params)
        cp = crop_e.pull("N")
        assert cp.axes.y == exp_y, f"crop {params} payload y={cp.axes.y} != {exp_y}"
        assert crop_e.env("N").axes.y == exp_y, \
            f"crop {params} header y={crop_e.env('N').axes.y} != payload {exp_y}"

    # R3 wavelet_denoise: a flat/blank plane comes through finite (no BayesShrink NaN)
    #    and untouched, while a textured plane in the same stack denoises finitely.
    yy, xx = np.mgrid[0:16, 0:16]
    wimg = np.zeros((1, 1, 2, 1, 16, 16), dtype=float)          # plane 0 flat/blank
    wimg[0, 0, 1, 0] = (np.sin(yy * 0.7) + np.cos(xx * 0.5) + 2.0) * 3.0   # plane 1 textured
    dsf = Dataset(axes=AxisSizes(m=1, t=1, z=2, c=1, y=16, x=16),
                  metadata=optics).with_image(ArrayProvider(wimg))
    gf = Graph()
    gf.add(NodeInstance("S", "io.seed3"))
    gf.add(NodeInstance("W", "enhance.wavelet_denoise", modes={"dim": "2D"}))
    gf.connect("S", "W")
    wout = Engine(gf, computes=COMPUTES, seeds={"S": dsf},
                  meta_seeds={"S": MetaEnvelope(axes=dsf.axes, metadata=optics)}).pull("W")
    warr = wout.image.get_region_volume(0, 0, 0, 0, 0, 2, 0, 16, 0, 16)
    assert np.all(np.isfinite(warr)), "wavelet flat plane produced NaN"
    assert np.all(warr[0] == 0.0), "wavelet flat plane should pass through untouched"

    _ok("catalog (ported): 8 filters (2D/3D lever), TV/wavelet 3D honesty, normalize "
        "scope Mode, spot→Points, z-project + crop meta_transform; review fixes R1–R3")


# ── ported catalog batch 2 + fusion reducers + C2 bridge execution ────────────

def test_fusion_reducers() -> None:
    from nodegraph.reducers import REDUCERS, is_tileable, reduce as _r
    # sigma_clip is robust: a lone extreme outlier is rejected (mean/std would keep it)
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert abs(float(_r(x, (0,), "sigma_clip")) - 2.5) < 1e-6      # 100 dropped → mean(1..4)
    assert float(_r(x, (0,), "trimmed_mean")) == 3.0              # drop min+max → mean(2,3,4)
    # both are whole-domain (NOT tile-reducible — like median)
    assert not is_tileable("sigma_clip") and not is_tileable("trimmed_mean")
    # multi-axis + a constant population (MAD==0 → threshold 0, a truly constant cell
    # is untouched, never emptied)
    flat = np.full((3, 4), 7.0)
    assert np.allclose(_r(flat, (0,), "sigma_clip"), 7.0)
    # review: MAD==0 on a flat/dark background + a lone cosmic ray STILL rejects it
    # (the old std-fallback re-admitted it)
    assert float(_r(np.array([0.0, 0.0, 0.0, 0.0, 50.0]), (0,), "sigma_clip")) == 0.0
    # review: trimmed_mean is NaN-aware — the NaN is excluded (not kept as a "high"
    # value), the real high outlier is trimmed, and a low-count cell never goes NaN
    assert abs(float(_r(np.array([1.0, 2.0, 3.0, 100.0, np.nan]), (0,), "trimmed_mean"))
               - 2.5) < 1e-9
    assert np.isfinite(_r(np.array([5.0, np.nan]), (0,), "trimmed_mean"))   # k=1 ≤ 2·trim
    _ok("fusion reducers: sigma-clip (MAD-robust, rejects cosmic ray on flat-dark); "
        "trimmed mean (NaN-aware); both whole-domain")


def test_transfer_bridge_exec() -> None:
    from nodegraph.transfer import Carrier, execute_bridge_plan, plan_transfer
    from nodegraph.bridges import gather_by_track, voxel_to_label
    from nodegraph.structure import TrackMembership

    raster = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0], [3, 3, 0, 2]])
    vox = np.array([[10, 10, 0, 20], [10, 10, 0, 20], [0, 0, 0, 0], [30, 30, 0, 20]],
                   dtype=float)
    mem = TrackMembership(track_id=[1, 2, 2], t=[0, 0, 1], member_id=[1, 2, 3])

    # C2: a routed Voxel→Track plan (Voxel→Label→Track) executes as one call and equals
    # the hand-chained bridges.
    plan = plan_transfer(D.VOXEL, D.TRACK, reducer="mean")
    assert not plan.generated and len(plan.steps) == 2
    res = execute_bridge_plan(Carrier(D.VOXEL, array=vox), plan,
                              label_raster=raster, membership=mem)
    ids, means = voxel_to_label(vox, raster, "mean")
    tids, tvals = gather_by_track(ids, means, mem, "mean")
    assert res.domain is D.TRACK and res.ids.tolist() == tids.tolist()
    assert np.allclose(res.values, tvals) and res.values.tolist() == [10.0, 25.0]

    # Label→Frame reduce-in-frame → one scalar
    fr = execute_bridge_plan(Carrier(D.LABEL, ids=ids, values=means),
                             plan_transfer(D.LABEL, D.FRAME, reducer="mean"))
    assert fr.domain is D.FRAME and abs(float(fr.values) - float(np.mean(means))) < 1e-9

    # a hop missing its structure input is a clear hard error, not a wrong result
    try:
        execute_bridge_plan(Carrier(D.VOXEL, array=vox), plan, membership=mem)
        raise AssertionError("expected a missing-label_raster error")
    except ValueError as e:
        assert "label_raster" in str(e)

    # review: the generated (pure-lattice) branch honors the supplied `axes` — a
    # coarse→fine broadcast sizes to the real geometry (was a size-1 AxisSizes()).
    fplan = plan_transfer(D.FRAME, D.VOXEL)
    assert fplan.generated
    gax = AxisSizes(m=1, t=1, z=2, c=1, y=3, x=3)
    bc = execute_bridge_plan(Carrier(D.FRAME, array=np.array([[5.0]])), fplan, axes=gax)
    assert bc.domain is D.VOXEL and bc.array.shape == gax.shape_for(D.VOXEL)
    assert np.all(bc.array == 5.0)
    _ok("transfer C2: routed Voxel→Track (Voxel→Label→Track) chains as one call; "
        "Label→Frame; generated broadcast honors axes; missing-input guard")


def test_catalog_ported2() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (batch 2): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=3, z=3, c=1, y=16, x=16)
    yy, xx = np.mgrid[0:16, 0:16]
    base = np.sin(yy * 0.6) + np.cos(xx * 0.4) + 2.0            # full-plane texture
    img = np.zeros((1, 3, 3, 1, 16, 16), dtype=float)
    for t in range(3):
        img[0, t, :, 0] = base * 5.0 + t * 0.5
    img[0, :, :, 0, 6:10, 6:10] += 30.0                        # a bright blob region
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 1.0, "objective_na": 1.0,
              "channel_emission_nm": [520]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    define_node("io.seed4", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    def eng(op, *, modes=None, params=None, chain=()):
        g = Graph(); g.add(NodeInstance("S", "io.seed4")); prev = "S"
        for i, (cop, cmodes, cparams) in enumerate(chain):
            nid = f"U{i}"
            g.add(NodeInstance(nid, cop, modes=cmodes or {}, params=cparams or {}))
            g.connect(prev, nid); prev = nid
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect(prev, "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    def imv(dset):
        p = dset.image
        return p.get_region_volume(0, 0, 0, 0, 0, p.axes.z, 0, p.axes.y, 0, p.axes.x)

    # new IMAGE→IMAGE filters: finite + geometry-preserving in both dims; lever re-keys
    for op in ("enhance.morphological_gradient", "enhance.bilateral", "enhance.nlm"):
        for dim in ("2D", "3D"):
            arr = imv(eng(op, modes={"dim": dim}).pull("N"))
            assert arr.shape == (3, 16, 16) and np.all(np.isfinite(arr)), f"{op}/{dim}"
        assert (eng(op, modes={"dim": "2D"}).entry("N").recipe_hash
                != eng(op, modes={"dim": "3D"}).entry("N").recipe_hash), f"{op} lever"
    # bilateral is honestly stack-of-2D; nlm is genuinely volumetric
    assert NODES.get("enhance.bilateral").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_PLANE
    assert NODES.get("enhance.nlm").resolve_granularity({"dim": "3D"}) \
        is Granularity.WHOLE_VOLUME

    # spot detection: LoG/DoG × bright/dark all run; method/polarity fold into the hash
    for method in ("log", "dog"):
        for pol in ("bright", "dark"):
            sp = eng("detect.spots", modes={"dim": "2D", "method": method, "polarity": pol},
                     params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.08})
            assert sp.pull("N").get(D.POINT, "y", layer="spots") is not None
    assert (eng("detect.spots", modes={"method": "log"}).entry("N").recipe_hash
            != eng("detect.spots", modes={"method": "dog"}).entry("N").recipe_hash)

    # review R: a FLAT/blank volume has NO spots of EITHER polarity — the dark-polarity
    # inversion must not turn the flat guard's zeros into a constant-1 flood, and the 3D
    # LoG axial σ must not clamp sub-voxel into a spurious uniform response.
    flat_ds = Dataset(axes=ax, metadata=optics).with_image(
        ArrayProvider(np.zeros((1, 3, 3, 1, 16, 16))))
    gflat = Graph(); gflat.add(NodeInstance("S", "io.seed4"))
    gflat.add(NodeInstance("N", "detect.spots", modes={"dim": "3D", "polarity": "dark"},
                           params={"min_radius": 0.05, "max_radius": 0.4, "threshold": 0.05}))
    gflat.connect("S", "N")
    fspots = Engine(gflat, computes=COMPUTES, seeds={"S": flat_ds},
                    meta_seeds={"S": seedenv}).pull("N").get(D.POINT, "y", layer="spots")
    assert fspots is None or len(fspots.values) == 0, "flat volume should yield no spots"

    thr = ("analysis.threshold", None, {"threshold": 20.0})

    # EDT: a µm distance field on the mask (finite, positive inside)
    for dim in ("2D", "3D"):
        dist = eng("analysis.edt", modes={"dim": dim}, chain=(thr,)).pull("N").get(D.VOXEL, "distance")
        assert dist is not None and np.all(np.isfinite(dist.values)) and dist.values.max() > 0

    # Segmentation, watershed method (V2.12 - `analysis.watershed` folded in here):
    # a global-unique Label raster + region table, splitting the mask it is pointed at
    w = eng("analysis.segment", modes={"dim": "2D", "method": "watershed"},
            params={"mask": "mask", "name": "watershed"}, chain=(thr,)).pull("N")
    assert int(w.get(D.VOXEL, "watershed").values.max()) >= 1
    warea = w.get(D.LABEL, "area", layer="watershed")
    assert warea is not None and len(set(range(1, len(warea.values) + 1)))  # ids present

    # Resample: axes + inverse pixel size mirror the meta_transform (header==payload)
    for dim, params, ey in (("2D", {"scale_xy": 0.5}, 8), ("2D", {"scale_xy": 2.0}, 32)):
        e = eng("util.resample", modes={"dim": dim}, params=params)
        o = e.pull("N")
        assert o.axes.y == ey and e.env("N").axes.y == ey
        assert abs(o.metadata["pixel_size_um"] - 0.1 / params["scale_xy"]) < 1e-9
    e3 = eng("util.resample", modes={"dim": "3D"}, params={"scale_xy": 1.0, "scale_z": 2.0})
    o3 = e3.pull("N")
    assert o3.axes.z == 6 and e3.env("N").axes.z == 6
    assert abs(o3.metadata["z_step_um"] - 0.15) < 1e-9

    # Stack: T→1 with each combiner; dt_s dropped; env tracks it
    for meth in ("mean", "median", "sigma_clip", "trimmed_mean", "max"):
        es = eng("util.stack", modes={"method": meth})
        so = es.pull("N")
        assert so.axes.t == 1 and es.env("N").axes.t == 1 and "dt_s" not in so.metadata

    # Drift: geometry preserved; the estimated shift is stored as Frame attributes
    dr = eng("align.drift").pull("N")
    assert dr.axes.t == 3 and dr.axes.y == 16
    assert dr.get(D.FRAME, "drift_y") is not None and dr.get(D.FRAME, "drift_x") is not None

    # Measure: multi-stat columns all present and id-aligned
    md = eng("analysis.measure", chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
    for col in ("mean_intensity", "max_intensity", "min_intensity", "area"):
        assert md.get(D.LABEL, col, layer="labels") is not None, f"measure {col}"

    # Source-layer + output-name sockets (2026-07-28). These were params the compute read
    # with no socket declared, so from the GUI the whole segmentation chain was pinned to
    # one hardcoded layer name and a graph could not carry two masks. Chain them with
    # NON-default names end to end: threshold→label→watershed→edt each reading the
    # previous node's renamed output proves the redirects actually compose.
    chain_named = eng("analysis.edt", modes={"dim": "2D"},
                      params={"mask": "m2", "name": "dist2"},
                      chain=(("analysis.threshold", None,
                              {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert chain_named.get(D.VOXEL, "m2") is not None, "threshold `name` socket"
    assert chain_named.get(D.VOXEL, "dist2") is not None, "edt `mask`+`name` sockets"
    assert chain_named.get(D.VOXEL, "distance") is None, "no stale default edt layer"
    lab_named = eng("analysis.label", modes={"dim": "2D"},
                    params={"mask": "m2", "name": "regions2"},
                    chain=(("analysis.threshold", None,
                            {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert lab_named.get(D.VOXEL, "regions2") is not None, "label `mask`+`name` sockets"
    assert lab_named.get(D.LABEL, "area", layer="regions2") is not None, "label table follows"
    wat_named = eng("analysis.segment", modes={"dim": "2D", "method": "watershed"},
                    params={"mask": "m2", "name": "ws2"},
                    chain=(("analysis.threshold", None,
                            {"threshold": 20.0, "name": "m2"}),)).pull("N")
    assert wat_named.get(D.VOXEL, "ws2") is not None, "segment `mask`+`name` sockets"
    # pointing a node at a layer that does not exist must RAISE, not silently no-op
    try:
        eng("analysis.label", modes={"dim": "2D"}, params={"mask": "nope"},
            chain=(thr,)).pull("N")
        raise AssertionError("a missing source layer must raise")
    except (ValueError, KeyError):
        pass
    # `stats` selects the measured columns (it too had no socket, so the set was fixed)
    md2 = eng("analysis.measure", params={"stats": "mean,median"},
              chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
    assert md2.get(D.LABEL, "median_intensity", layer="labels") is not None, "stats picks median"
    assert md2.get(D.LABEL, "max_intensity", layer="labels") is None, "stats excludes max"
    try:
        eng("analysis.measure", params={"stats": "bogus"},
            chain=(thr, ("analysis.label", {"dim": "2D"}, {}))).pull("N")
        raise AssertionError("an unknown stat must raise")
    except ValueError as exc:
        assert "unknown measure stat" in str(exc)

    # Extract Boundary: a label raster → boundary Points
    lbl2 = ("analysis.label", {"dim": "2D"}, {})
    eb = eng("analysis.extract_boundary", modes={"dim": "2D"}, chain=(thr, lbl2)).pull("N")
    ybp = eb.get(D.POINT, "y", layer="labels_boundary")
    assert ybp is not None and len(ybp.values) >= 1
    # ...and the 3D surface-vertex path, which had no coverage at all (which is how the
    # missing sockets below survived: only the all-defaults 2D pull was ever exercised)
    eb3 = eng("analysis.extract_boundary", modes={"dim": "3D"},
              chain=(thr, ("analysis.label", {"dim": "3D"}, {}))).pull("N")
    p3 = eb3.get(D.POINT, "y", layer="labels_boundary")
    assert p3 is not None and len(p3.values) >= 1, "3D marching-cubes vertices"
    assert eb3.get(D.POINT, "z", layer="labels_boundary") is not None, "3D needs a z column"

    # REGRESSION (2026-07-28): the compute always read a `labels` source param and an
    # output-name param, but register_node declared ONLY InDataset() — so neither had a
    # socket and both were stuck on their defaults in the GUI, making it impossible to
    # outline a Label raster not named "labels".
    #
    # READ THIS BEFORE TRUSTING THE ASSERTIONS BELOW: an UNDECLARED param still reaches
    # ctx.params (the engine does not filter params against the socket list), so the two
    # layer-redirect pulls and the re-key check below all PASS against the pre-fix node —
    # measured, not assumed. They are coverage of the sockets' semantics, NOT guards. The
    # defect was reachability from the GUI, which builds its widgets from `inputs`, so the
    # only assertions that actually fail against the old node are the `name` rename, the
    # domain rail, and the socket-set check at the end of this block. Do not delete those
    # three thinking the behavioural pulls cover them.
    lblR = ("analysis.label", {"dim": "2D"}, {"name": "regions"})
    ebr = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions"}, chain=(thr, lblR)).pull("N")
    assert ebr.get(D.POINT, "y", layer="regions_boundary") is not None, \
        "the `labels` socket must redirect the source layer"
    # `name` names the output; empty `name` keeps auto-deriving f"{labels}_boundary"
    ebn = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions", "name": "outline"}, chain=(thr, lblR)).pull("N")
    assert ebn.get(D.POINT, "y", layer="outline") is not None, "the `name` socket"
    assert ebn.get(D.POINT, "y", layer="regions_boundary") is None, "no stale default layer"
    # `labels` folds into the recipe hash, so re-pointing the node re-keys the memo.
    # Chain BOTH label nodes so either choice resolves (layers accumulate on the Dataset).
    _both = (thr, lbl2, lblR)
    _hR = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "regions"}, chain=_both).entry("N").recipe_hash
    _hL = eng("analysis.extract_boundary", modes={"dim": "2D"},
              params={"labels": "labels"}, chain=_both).entry("N").recipe_hash
    assert _hR != _hL, "`labels` must fold into recipe_hash"
    # The two REAL guards (both fail against the pre-fix node). The domain rail was empty
    # despite an identical contract to detect.spots, so the GUI drew no chips and no red
    # missing-domain validation here; and the socket set is the only thing that can catch
    # a param the compute reads but nobody can reach.
    _ebspec = NODES.get("analysis.extract_boundary")
    assert _ebspec.reads_domains == frozenset({D.VOXEL}), "domain rail: reads VOXEL"
    assert _ebspec.adds_domains == frozenset({D.POINT}), "domain rail: adds POINT"
    assert {s.name for s in _ebspec.inputs} == {"data", "labels", "name"}, \
        "every param the compute reads needs a socket, or the GUI cannot set it"

    _ok("catalog (batch 2): morph-gradient/bilateral/nlm, LoG+DoG × bright/dark, EDT, "
        "watershed, resample+stack+drift meta, multi-stat measure, extract-boundary "
        "(2D contours + 3D surface verts; labels/name sockets redirect + re-key; rail)")


# ── ported catalog batch 3: threshold methods, multi-otsu, transfer-domain ────

def test_channel_split() -> None:
    """``channel.split`` is a domain-transparent pass-through: its ``out`` socket carries
    the full multi-channel bundle unchanged (axes, pixels, and channel emission all
    preserved). Per-channel fan-out is a GUI/graph-build concern (nodelab materializes
    each ``chK`` tap into a ``channel.select``), so here we pin the node contract itself."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=1, c=3, y=4, x=4)
    img = np.zeros((1, 1, 1, 3, 4, 4), dtype=float)
    img[0, 0, 0, 0] = 1.0; img[0, 0, 0, 1] = 2.0; img[0, 0, 0, 2] = 3.0   # per-channel
    seedenv = MetaEnvelope(axes=ax, metadata={"channel_emission_nm": [461, 509, 610]})
    ds = Dataset(axes=ax, metadata=dict(seedenv.metadata)).with_image(ArrayProvider(img))

    g = Graph()
    g.add(NodeInstance("S", "io.seedSplit"))
    g.add(NodeInstance("P", "channel.split"))
    g.connect("S", "P")
    define_node("io.seedSplit", "Seed", outputs=[OutDataset()])
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    out = eng.pull("P")
    assert out.axes.c == 3, "split must pass all channels through 'out'"
    for ch, val in enumerate((1.0, 2.0, 3.0)):
        plane = out.image.get_region(0, 0, 0, 0, ch, 0, 4, 0, 4)
        assert np.allclose(plane, val), f"channel {ch} altered by split"
    # envelope passes through unchanged (no meta_transform)
    assert eng.env("P").axes.c == 3
    assert list(eng.env("P").metadata.get("channel_emission_nm")) == [461, 509, 610]
    _ok("channel.split: pass-through preserves channels + emission (tap fan-out is GUI)")


def test_two_channel_branches() -> None:
    """**Two channel branches off one file, converging on one analysis node** (2026-08-03).

    The canonical multi-channel workflow — segment channel 0, measure the intensity of
    channel 1 — is two ``channel.select`` taps off one source. Both halves of it were
    broken, and each failure looked like something other than what it was.

    1. **Every per-channel metadata list follows the tap**, not just the calibration one.
       ``channel_names`` / ``channel_colors`` / ``channel_excitation_nm`` are display lists
       read POSITIONALLY, so a full-length survivor on a ``c == 1`` Dataset does not read as
       stale — the ch1 branch reported ``names[0]``, i.e. the FIRST channel's name, and both
       branches labelled themselves identically with no way to tell them apart.

    2. **A channel tap does not count as a spatial-geometry change.** It stamps the sampling
       provenance with the :data:`~nodegraph.nodes.CHANNEL_STAMP` prefix, and the
       voxel-for-voxel guard in ``_intensity_provider`` drops those stamps when both sides
       carry one channel — so ``ch0``-labels + ``ch1``-intensity is accepted. It was refused
       with a geometry-mismatch error naming a shift that did not exist. A channel
       PERMUTATION at ``c > 1`` still changes what a ``c`` index means, so it stays refused.
    """
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import CHANNEL_STAMP, COMPUTES, SAMPLING_KEY

    ax = AxisSizes(m=1, t=1, z=1, c=2, y=16, x=16)
    img = np.zeros((1, 1, 1, 2, 16, 16))
    img[0, 0, 0, 0, 2:6, 2:6] = 800.0        # ch0: two square objects to segment
    img[0, 0, 0, 0, 10:14, 10:14] = 800.0
    img[0, 0, 0, 1, 2:6, 2:6] = 300.0        # ch1: the SAME regions, distinct intensities
    img[0, 0, 0, 1, 10:14, 10:14] = 600.0
    md = {"pixel_size_um": 0.5, "bit_depth": 12,
          "channel_emission_nm": [461.0, 509.0], "channel_names": ["DAPI", "GFP"],
          "channel_excitation_nm": [405.0, 488.0],
          "channel_colors": [[0, 0, 255], [0, 255, 0]]}
    ds = Dataset(axes=ax, metadata=dict(md)).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax, metadata=dict(md))
    define_node("io.seed2ch", "Seed", outputs=[OutDataset()])

    def _graph(raw_src, c0=(0,), c1=(1,)):
        g = Graph()
        g.add(NodeInstance("S", "io.seed2ch"))
        g.add(NodeInstance("T0", "channel.select", params={"channels": list(c0)}))
        g.add(NodeInstance("T1", "channel.select", params={"channels": list(c1)}))
        g.add(NodeInstance("G", "analysis.threshold", params={"threshold": 400.0}))
        g.add(NodeInstance("L", "analysis.label"))
        g.add(NodeInstance("M", "analysis.measure"))
        g.connect("S", "T0"); g.connect("S", "T1")
        g.connect("T0", "G"); g.connect("G", "L"); g.connect("L", "M")
        if raw_src:
            g.connect(raw_src, "M", dst_socket="raw")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})

    # (1) the display lists narrow to the tapped channel — payload AND envelope agree
    eng = _graph("T1")
    for nid, want in (("T0", 0), ("T1", 1)):
        out, e = eng.pull(nid), eng.env(nid)
        assert out.axes.c == e.axes.c == 1, (nid, out.axes.c, e.axes.c)
        for key in ("channel_names", "channel_emission_nm",
                    "channel_excitation_nm", "channel_colors"):
            assert out.metadata.get(key) == [md[key][want]], \
                (f"{nid}: {key} must narrow to channel {want}, got "
                 f"{out.metadata.get(key)!r} — a positional reader would report "
                 f"channel 0's value on BOTH branches")
        assert e.metadata.get("channel_emission_nm") == [md["channel_emission_nm"][want]]
    # the two taps are genuinely different pixels (not one branch served twice)
    assert eng.pull("T0").image.read_region(0, 0, 0, 0, 0, 2, 3, 2, 3)[0, 0] == 800.0
    assert eng.pull("T1").image.read_region(0, 0, 0, 0, 0, 2, 3, 2, 3)[0, 0] == 300.0

    # (2) the tap's stamp is channel-namespaced, so a spatial consumer can drop it
    stamps = tuple(eng.pull("T0").metadata.get(SAMPLING_KEY, ()))
    assert len(stamps) == 1 and stamps[0].startswith(CHANNEL_STAMP), stamps

    # (3) ch0 labels + ch1 intensity RUNS, and reports ch1's numbers
    def _intensities(engine):
        cols = {a.name: np.asarray(a.values).ravel()
                for a in engine.pull("M").attributes.values()}
        return np.sort(cols["mean_intensity"])

    own = _intensities(_graph(None))                  # no `raw`: measures ch0
    other = _intensities(_graph("T1"))                # `raw` = ch1
    assert own.size == other.size == 2, (own, other)
    assert np.allclose(own, [800.0, 800.0]), own
    assert np.allclose(other, [300.0, 600.0]), \
        f"`raw` from the ch1 tap must report ch1's intensities, got {other}"

    # (4) a channel PERMUTATION at c>1 is still refused — there the stamp means something
    try:
        _graph("T1", c0=(0, 1), c1=(1, 0)).pull("M")
        raise AssertionError("a reordered channel axis was accepted on `raw` — index c no "
                             "longer names the same channel on the two branches")
    except ValueError as exc:
        assert "sampling geometry" in str(exc), exc

    _ok("two channel branches: per-channel metadata narrows per tap (names/colors/"
        "excitation, not just emission); ch0-labels + ch1-intensity runs and reports ch1; "
        "a c>1 permutation still refused")


def test_reroute() -> None:
    """``rr.reroute`` is an identity pass-through (a GUI wire-routing hop): pixels, axes
    and calibration all pass through unchanged, and it composes transparently in a chain.
    Hidden from the palette (``rr.`` prefix) but a real engine node so meta-propagation /
    memoization need no special-casing."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    ax = AxisSizes(m=1, t=1, z=2, c=1, y=4, x=4)
    img = np.arange(1 * 1 * 2 * 1 * 4 * 4, dtype=float).reshape(1, 1, 2, 1, 4, 4)
    optics = {"pixel_size_um": 0.2, "channel_emission_nm": [488]}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.seedRR", "Seed", outputs=[OutDataset()])

    # S → R1 → R2 : two reroutes in series must not alter the data
    g = Graph()
    g.add(NodeInstance("S", "io.seedRR"))
    g.add(NodeInstance("R1", "rr.reroute"))
    g.add(NodeInstance("R2", "rr.reroute"))
    g.connect("S", "R1"); g.connect("R1", "R2")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})

    out = eng.pull("R2")
    got = out.image.get_region_volume(0, 0, 0, 0, 0, ax.z, 0, ax.y, 0, ax.x)
    assert np.array_equal(got, img[0, 0, :, 0]), "reroute altered the pixels"
    # envelope + calibration pass through unchanged (no meta_transform)
    assert eng.env("R2").axes == ax
    assert list(eng.env("R2").metadata.get("channel_emission_nm")) == [488]
    # TILEABLE, no dim lever, hidden prefix
    spec = NODES.get("rr.reroute")
    assert spec.resolve_granularity({}) is Granularity.TILEABLE
    assert not spec.has_dim_lever() and spec.op_key.startswith("rr.")
    _ok("rr.reroute: identity pass-through (pixels/axes/calib preserved, composes)")


def test_catalog3() -> None:
    if not _HAVE_SKIMAGE:
        _ok("catalog (batch 3): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    # a 3-population image: 0 (bg), 50 (mid), 100 (hi)
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
    img = np.zeros((1, 1, 1, 1, 4, 4))
    img[0, 0, 0, 0, :2, :2] = 0.0; img[0, 0, 0, 0, :2, 2:] = 50.0
    img[0, 0, 0, 0, 2:, :] = 100.0
    ds = Dataset(axes=ax).with_image(ArrayProvider(img))
    define_node("io.seedC3", "Seed", outputs=[OutDataset()])
    env = MetaEnvelope(axes=ax)

    def eng(op, *, modes=None, params=None):
        g = Graph(); g.add(NodeInstance("S", "io.seedC3"))
        g.add(NodeInstance("N", op, modes=modes or {}, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})

    # histogram threshold methods: each separates bg from signal; each memoizes distinctly
    hashes = set()
    for method in ("otsu", "li", "yen", "triangle", "mean"):
        e = eng("analysis.threshold", modes={"method": method})
        mask = e.pull("N").get(D.VOXEL, "mask")
        assert mask is not None and set(np.unique(mask.values).tolist()) == {0, 1}, method
        hashes.add(e.entry("N").recipe_hash)
    assert len(hashes) == 5, "each threshold method must memoize distinctly"
    # the `threshold` socket is FIXED-only: a histogram method derives its own cut and
    # never reads it, so it must not be offered under otsu/li/yen/triangle/mean
    _tvis = lambda st: {i.name for i in NODES.get("analysis.threshold").active_inputs(st)}
    assert "threshold" in _tvis({"method": "fixed"})
    assert all("threshold" not in _tvis({"method": mm})
               for mm in ("otsu", "li", "yen", "triangle", "mean"))

    # multi-otsu → a 3-class index raster (0,1,2)
    cl = eng("analysis.multiotsu", params={"classes": 3}).pull("N").get(D.VOXEL, "classes")
    assert cl is not None and set(np.unique(cl.values).tolist()) == {0, 1, 2}

    # adaptive local threshold → a Voxel mask (per-plane; finite, binary)
    lm = eng("analysis.threshold_local", params={"block_size": 0.3}).pull("N").get(D.VOXEL, "mask")
    assert lm is not None and set(np.unique(lm.values).tolist()) <= {0, 1}

    # transfer domain: a Voxel attribute → Frame (mean over z,c,y,x)
    vox = np.arange(16, dtype=float).reshape(1, 1, 1, 1, 4, 4)
    dst = Dataset(axes=ax).with_layer(D.VOXEL, "vals", vox)
    def _xfer(*, modes=None, params=None):
        g = Graph(); g.add(NodeInstance("S", "io.seedC3"))
        g.add(NodeInstance("T", "transform.transfer_domain",
                           modes=modes or {}, params=params or {"attr": "vals"}))
        g.connect("S", "T")
        return Engine(g, computes=COMPUTES, seeds={"S": dst}, meta_seeds={"S": env})

    # from/to/reducer are MODES as of 2026-07-28 (they had no sockets at all before, so
    # the node could only ever do voxel→frame/mean from the GUI).
    ft = _xfer(modes={"from_domain": "voxel", "to_domain": "frame", "reducer": "mean"}
               ).pull("T")
    fr = ft.get(D.FRAME, "vals")
    assert fr is not None and fr.values.shape == ax.shape_for(D.FRAME)
    assert abs(float(fr.values[0, 0]) - float(vox.mean())) < 1e-9   # reduced mean
    # A NON-DEFAULT reducer must actually change the number. The old test only ever used
    # values equal to the defaults, so it could not tell a working lever from an ignored
    # one — which is exactly how the missing sockets survived here.
    fmax = _xfer(modes={"from_domain": "voxel", "to_domain": "frame", "reducer": "max"}
                 ).pull("T").get(D.FRAME, "vals")
    assert abs(float(fmax.values[0, 0]) - float(vox.max())) < 1e-9, "reducer mode is live"
    assert (_xfer(modes={"reducer": "max"}).entry("T").recipe_hash
            != _xfer(modes={"reducer": "mean"}).entry("T").recipe_hash), "mode re-keys"
    # A headless caller predating the Mode conversion is REFUSED, not silently run on the
    # defaults. A silent fallback is not implementable: the engine hands the compute the
    # RESOLVED mode state, so "unset" and "explicitly set to the default" are the same
    # value — honouring the param would mean guessing which one happened.
    try:
        _xfer(params={"attr": "vals", "reducer": "max"}).pull("T")
        raise AssertionError("the legacy param form must be refused, not ignored")
    except ValueError as exc:
        assert "are Modes now" in str(exc)
    # charter: a pure broadcast ignores `reducer`, so a non-default one is REFUSED
    try:
        _xfer(modes={"from_domain": "frame", "to_domain": "voxel", "reducer": "max"}
              ).pull("T")
        raise AssertionError("a no-reduce transfer must refuse a non-default reducer")
    except ValueError as exc:
        assert "drops no axis" in str(exc)
    # structure domains are absent from the dropdowns AND refused if hand-authored
    try:
        _xfer(modes={"from_domain": "label", "to_domain": "frame"}).pull("T")
        raise AssertionError("structure-domain transfer must be refused")
    except ValueError as exc:
        assert "structure bridge" in str(exc)
    _spec = NODES.get("transform.transfer_domain")
    assert {m.name for m in _spec.modes} == {"from_domain", "to_domain", "reducer"}
    assert "label" not in dict((m.name, m.choices) for m in _spec.modes)["from_domain"]
    _ok("catalog (batch 3): threshold methods (otsu/li/yen/triangle/mean, distinct keys); "
        "multi-otsu class raster; transfer-domain Voxel→Frame reduce (from/to/reducer are "
        "live Modes + legacy params; non-default reducer changes the value and re-keys; "
        "no-reduce + structure-domain refusals)")


# ── Stitch (util.stitch — the MULTI_VIEW / M→1 mosaic node) ───────────────────

def _stitch_texture(h: int, w: int) -> np.ndarray:
    """A deterministic, **non-periodic** texture: golden-ratio-placed Gaussian blobs.

    No RNG (build-node-v2 §4), but also no grating — a sine/checker pattern would make
    phase correlation ambiguous by construction, so the refine leg would be testing
    aliasing rather than registration."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    out = np.full((h, w), 400.0)
    g = 0.6180339887498949
    for k in range(140):
        cy = ((k * g) % 1.0) * h
        cx = (((k * k + 3 * k) * g) % 1.0) * w
        sig = 2.0 + 3.0 * ((k * 7 * g) % 1.0)
        amp = 60.0 + 240.0 * ((k * 11 * g) % 1.0)
        out += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig * sig)))
    return out


def test_stitch() -> None:
    """``util.stitch``: carve a known canvas into overlapping tiles, stitch it back.

    The fixture is the whole point — a 2×3 mosaic cut FROM one canvas, so the correct
    answer is known pixel-for-pixel and every blend that is not lossy must reproduce it
    exactly. Overlap is 32 px on 96 px tiles (33%), in the range a real acquisition uses;
    at 24 px and below, pair correlation on the resulting narrow band is at its precision
    limit and the test would be measuring FFT resolution rather than the node."""
    if not _HAVE_SKIMAGE:
        _ok("stitch: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.engine import EvalContext, ReadContext
    from nodegraph.streaming import MultiViewProvider

    TH = TW = 96
    STEP = 64                                    # 32 px overlap on both axes
    PX = 2.0
    OFFS = [(oy, ox) for oy in (0, STEP) for ox in (0, STEP, 2 * STEP)]
    H, W, N_M, N_T = STEP + TH, 2 * STEP + TW, len(OFFS), 2
    canvas = np.stack([_stitch_texture(H, W),
                       _stitch_texture(H, W) * 0.5 + 10.0])          # two DIFFERENT frames
    tiles = np.zeros((N_M, N_T, 1, 1, TH, TW))
    for m, (oy, ox) in enumerate(OFFS):
        for t in range(N_T):
            tiles[m, t, 0, 0] = canvas[t][oy:oy + TH, ox:ox + TW]
    # flip_x=True / flip_y=False (the defaults) invert exactly this mapping
    stage = [(-ox * PX, oy * PX) for (oy, ox) in OFFS]

    ax = AxisSizes(m=N_M, t=N_T, z=1, c=1, y=TH, x=TW)
    ds = Dataset(axes=ax, metadata={"pixel_size_um": PX, "stage_xy_um": stage}
                 ).with_image(ArrayProvider(tiles))
    env = MetaEnvelope(axes=ax, metadata={"pixel_size_um": PX})
    define_node("io.seedStitch", "Seed", outputs=[OutDataset()])

    def eng(modes=None, params=None, seed=ds, senv=env):
        g = Graph()
        g.add(NodeInstance("S", "io.seedStitch"))
        g.add(NodeInstance("N", "util.stitch", modes=modes or {}, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": seed}, meta_seeds={"S": senv})

    def plane(e, t=0):
        p = e.pull("N").image
        a = p.axes
        return np.asarray(p.get_region(0, 0, t, 0, 0, 0, a.y, 0, a.x)), a

    # ── 1. every blend reproduces the source canvas EXACTLY, on every frame ────
    # The tiles are exact cuts of `canvas`, so overlapping pixels agree; any weighted
    # mean of equal values is that value, and max/overwrite pick it. A blend that does
    # not reproduce it here is arithmetically wrong, not merely different.
    for blend in ("overwrite", "feather", "mean", "max"):
        for t in range(N_T):
            got, a = plane(eng(modes={"blend": blend}), t)
            assert (a.m, a.t, a.z, a.c, a.y, a.x) == (1, N_T, 1, 1, H, W), (blend, a)
            assert np.allclose(got, canvas[t], rtol=0, atol=1e-9), \
                f"{blend} @t={t}: max|err|={np.abs(got - canvas[t]).max():.3e}"

    # ── 2. the meta_transform: M→1 predicted, the canvas honestly UNKNOWN ──────
    # The extent depends on `stage_xy_um`, which rides the PAYLOAD (per-M geometry is not
    # in CALIBRATION_KEYS), so the edit-time pass cannot know it without reading data. It
    # says so instead of echoing the tile size, and the payload carries the real canvas.
    e = eng()
    assert e.env("N").axes.m == 1, e.env("N").axes
    assert {"y", "x"} <= e.env("N").unknown_axes, e.env("N").unknown_axes
    assert plane(e)[1].y == H and plane(e)[1].x == W
    assert "pixel_size_um" in dict(e.entry("N").reads), \
        "the µm→px conversion must be memo-fenced on pixel_size_um"

    # ── 3. modes and sockets re-key the memo, and the flips are live ───────────
    keys = {b: eng(modes={"blend": b}).entry("N").recipe_hash
            for b in ("overwrite", "feather", "mean", "max")}
    assert len(set(keys.values())) == 4, keys
    assert (eng(modes={"layout": "grid"}).entry("N").recipe_hash
            != eng(modes={"layout": "stage"}).entry("N").recipe_hash)
    noflip, _ = plane(eng(modes={"blend": "overwrite"}, params={"flip_x": False}))
    assert np.abs(noflip - canvas[0]).max() > 1.0, "flip_x does nothing"
    spec = NODES.get("util.stitch")
    assert spec.resolve_granularity(spec.default_state()) is Granularity.MULTI_VIEW
    vis = lambda lay: {i.name for i in spec.active_inputs(  # noqa: E731
        {**spec.default_state(), "layout": lay})}
    assert vis("stage") == {"data", "flip_x", "flip_y"}, vis("stage")
    assert "refine_min_ncc" in vis("stage+refine") and "grid_cols" not in vis("stage")
    assert vis("grid") == {"data", "grid_cols"}, vis("grid")

    # ── 4. streaming ≡ eager, and a sub-window ≡ that slice of the canvas ──────
    got_full, _ = plane(eng(modes={"blend": "feather"}))
    prov = eng(modes={"blend": "feather"}).pull("N").image
    assert isinstance(prov, MultiViewProvider), type(prov)
    win = np.asarray(prov.get_region(0, 0, 0, 0, 0, 30, 90, 50, 140))
    assert _stream_eq(win, got_full[30:90, 50:140]), "window ≠ that slice of the canvas"
    # A window must be servable WITHOUT the whole canvas being cached — that is what makes
    # the Viewer's detail-on-demand affordable (level 0 of a 49-tile mosaic is 172 Mpx, and
    # stitching all of it to show a 4096² patch would cost more than the overview did).
    # `_stitch_plane` clips per tile, so shifting every offset by the window origin yields
    # exactly this window; the feather ramp travels with the tile, so it stays identical.
    for blend in ("overwrite", "feather", "max", "mean"):
        e_w = eng(modes={"blend": blend})
        p_w = e_w.pull("N").image
        cold = np.asarray(p_w.get_region(0, 0, 0, 0, 0, 30, 90, 50, 140))   # nothing cached
        ref = np.asarray(p_w.get_region(0, 0, 0, 0, 0, 0, H, 0, W))[30:90, 50:140]
        assert _stream_eq(cold, ref), \
            f"a windowed stitch must equal that region of the whole canvas ({blend})"
    # ...including a window that hangs off the edge, and one that misses every tile
    edge = np.asarray(prov.get_region(0, 0, 0, 0, 0, H - 10, H + 40, -20, 30))
    assert edge.shape == (10, 30), edge.shape
    assert _stream_eq(edge, got_full[H - 10:H, 0:30]), "an out-of-range window must clip"
    rc = ReadContext(env.metadata)
    eager = COMPUTES["util.stitch"](EvalContext(
        node_id="N", op_key="util.stitch",
        params={"__modes__": {"layout": "stage", "blend": "feather"}},
        env=env, granularity=Granularity.MULTI_VIEW,
        kernel_axes=frozenset({"m", "y", "x"}), inputs=(ds,), reads=rc, spec=spec))
    ep = eager.image
    assert _stream_eq(np.asarray(ep.get_region(0, 0, 0, 0, 0, 0, ep.axes.y, 0, ep.axes.x)),
                      got_full), "the bare-ctx eager path disagrees with the streamed one"

    # ── 5. refusals — every case v1 answered with a silent grid ────────────────
    bare = Dataset(axes=ax).with_image(ArrayProvider(tiles))
    benv = MetaEnvelope(axes=ax)
    for seed, senv, needle in (
            (bare, benv, "stage log"),                                  # no log at all
            (Dataset(axes=ax, metadata={"pixel_size_um": PX,             # log too short
                                        "stage_xy_um": stage[:2]}
                     ).with_image(ArrayProvider(tiles)), env, "stage log"),
            (Dataset(axes=ax, metadata={"pixel_size_um": PX,             # M = repeat visits
                                        "stage_xy_um": [(0.0, 0.0)] * N_M}
                     ).with_image(ArrayProvider(tiles)), env, "repeat visits"),
            (ds.with_layer(D.POINT, "spots", np.zeros((3,))), env, "structure table")):
        try:
            eng(seed=seed, senv=senv).pull("N")
            raise AssertionError(f"expected a refusal mentioning {needle!r}")
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc)[:160])
    # ...and `grid` is the explicit way to get a montage without a position log
    gg, ga = plane(eng(modes={"layout": "grid"}, seed=bare, senv=benv))
    assert (ga.y, ga.x) == (2 * TH, 3 * TW), (ga.y, ga.x)
    gc, gca = plane(eng(modes={"layout": "grid"}, params={"grid_cols": 6},
                        seed=bare, senv=benv))
    assert (gca.y, gca.x) == (TH, 6 * TW), (gca.y, gca.x)      # grid_cols is live

    # ── 6. refine: recover a jittered stage log from the pixels ────────────────
    jit = [(2, -3), (-1, 2), (3, 1), (-2, -1), (1, 3), (0, -2)]
    bad = [(sx - jx * PX, sy + jy * PX) for (sx, sy), (jy, jx) in zip(stage, jit)]
    ds_bad = Dataset(axes=ax, metadata={"pixel_size_um": PX, "stage_xy_um": bad}
                     ).with_image(ArrayProvider(tiles))
    _, a_raw = plane(eng(seed=ds_bad))
    _, a_ref = plane(eng(modes={"layout": "stage+refine"}, seed=ds_bad))
    err_raw = max(abs(a_raw.y - H), abs(a_raw.x - W))
    err_ref = max(abs(a_ref.y - H), abs(a_ref.x - W))
    assert err_raw >= 4, f"the fixture's jitter should be visible ({err_raw})"
    assert err_ref <= 1, f"refine left the canvas {err_ref} px out ({a_ref.y}x{a_ref.x})"
    assert err_ref < err_raw, (err_raw, err_ref)
    assert (eng(modes={"layout": "stage+refine"}, seed=ds_bad).entry("N").recipe_hash
            != eng(seed=ds_bad).entry("N").recipe_hash), "the layout mode must re-key"

    # ── 7. the display pyramid — why a stitched series scrubs at all ──────────
    # `render_plane_native` shows ~2048 px; without a pyramid it had to read LEVEL 0, so
    # one 13106² frame stitched 172 Mpx to display 3.5 Mpx (measured 0.92 s/frame
    # overwrite, 2.9 s feather, on every frame scrubbed to). Forwarding the base's
    # pyramid serves the same picture from 1/64th of the work.
    if _HAVE_BLOSC2:
        from nodegraph.provider import B2ndProvider
        from nodelab_v2.runner import render_plane_native
        pyr = Dataset(axes=ax, metadata={"pixel_size_um": PX, "stage_xy_um": stage}
                      ).with_image(B2ndProvider.from_array(tiles, levels=2))
        # Level 1 is EXACT here, and provably so rather than approximately: the tiles are
        # cuts of one canvas, so every tile agrees with its neighbours in the overlaps and
        # any blend of them returns that shared value; and the offsets are even, so each
        # tile's own 2×2 mean lands on the same block grid as the whole canvas's. Both
        # together mean stitch-then-downsample and downsample-then-stitch coincide, for
        # every blend. (On real data with disagreeing overlaps they differ within a seam
        # pixel — a display-only difference, which is why level 0 stays exact.)
        half = canvas[0].reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
        for blend in ("overwrite", "feather", "max", "mean"):
            p = eng(modes={"blend": blend}, seed=pyr).pull("N").image
            assert p.levels == 2, (blend, p.levels)
            assert (p.level_axes(0).y, p.level_axes(0).x) == (H, W)
            assert (p.level_axes(1).y, p.level_axes(1).x) == (H // 2, W // 2), \
                (blend, p.level_axes(1))
            lv1 = np.asarray(p.get_region(1, 0, 0, 0, 0, 0, H // 2, 0, W // 2))
            assert np.allclose(lv1, half, rtol=0, atol=1e-9), \
                f"level-1 canvas != the downsampled level-0 one ({blend}): " \
                f"max|err|={np.abs(lv1 - half).max():.3e}"
        # the display path actually TAKES the coarse level (the reported symptom)
        p = eng(seed=pyr).pull("N").image
        # `max_dim` = the level-1 canvas's LONG side: level 0 (224 px) is too big, level 1
        # (112) fits, so the picker must land on 1 and stride-decimate by 1 — i.e. the
        # coarse canvas is shown as-is, with no level-0 read anywhere.
        shown, lv = render_plane_native(p, 0, 0, 0, 0, max_dim=W // 2)
        assert lv == 1, f"the Viewer stayed on level 0 ({lv}) — the pyramid is not wired"
        assert shown.shape == (H // 2, W // 2), shown.shape
        try:
            p.level_axes(2)
            raise AssertionError("an out-of-range level must raise")
        except ValueError:
            pass

        # ── the display DECIMATION itself (2026-07-31) ─────────────────────────
        # A display plane is capped ONCE and zoom is a pure view transform over that
        # texture, so this is the only resolution a mosaic ever gets. It used to be
        # `plane[::stride, ::stride]`: point-sampling, which keeps noise at full
        # amplitude while discarding 1-1/stride² of the data, and whose integer stride
        # threw away up to half the cap (a 3277 px plane under a 2048 cap became 1638).
        from nodelab_v2.runner import _fit_plane
        assert _fit_plane(np.zeros((40, 60), np.uint16), 4096).shape == (40, 60), \
            "under the cap the plane must pass through untouched"
        block = np.repeat(np.repeat(np.arange(16, dtype=np.float64).reshape(4, 4),
                                    25, axis=0), 25, axis=1)          # 100x100, 25² blocks
        got = _fit_plane(block, 4)
        assert got.shape == (4, 4), got.shape
        assert np.allclose(got, np.arange(16).reshape(4, 4)), \
            "each output pixel must be the MEAN of its block, not one sample from it"
        # hits the cap exactly rather than landing on an integer-stride multiple
        assert _fit_plane(np.zeros((3277, 3277), np.uint16), 2048).shape == (2048, 2048)
        assert _fit_plane(np.zeros((3277, 3277), np.uint16), 2048).dtype == np.uint16, \
            "an integer plane must stay integer — the GPU picks its texture format from it"
        # a noisy plane must come out SMOOTHER than point-sampling it (the bug report)
        _rng = np.random.default_rng(11)
        noisy = _rng.poisson(400.0, size=(1024, 1024)).astype(np.uint16)
        _rough = lambda a: float(np.abs(np.diff(a.astype(float), axis=1)).mean())  # noqa: E731
        assert _rough(_fit_plane(noisy, 256)) < 0.5 * _rough(noisy[::4, ::4]), \
            "area-averaging must suppress the noise that striding preserves"
    # a base with NO pyramid (every computed intermediate) still works, at level 0 only
    flat = eng().pull("N").image
    assert flat.levels == 1, flat.levels
    try:
        flat.level_axes(1)
        raise AssertionError("a pyramid-less base must not claim a level 1")
    except ValueError:
        pass

    _ok("stitch (util.stitch, M→1): 2x3 mosaic of %dx%d tiles (32 px overlap) round-trips "
        "EXACTLY through all four blends on both frames; header says m=1 with y/x UNKNOWN "
        "while the payload carries the real %dx%d canvas; pixel_size_um memo-fenced; "
        "blends/layout/flip re-key and gate their sockets; MULTI_VIEW resolved; streamed "
        "canvas == bare-ctx eager and a sub-window == its slice; 4 refusals (no/short "
        "stage log, repeat-visit M, structure table) with `grid` as the explicit way out; "
        "refine recovers a %d px jitter to %d px; and the display PYRAMID is forwarded "
        "from the base (level 1 == the downsampled level 0 exactly, for all 4 blends; the "
        "Viewer takes the coarse level; a pyramid-less base stays level-0 only)"
        % (TH, TW, H, W, err_raw, err_ref))


# ── Z-Project ON a Stitch (V2.20 — the plane_unit fence + the reduce pyramid) ──

def test_zproject_over_stitch() -> None:
    """``util.zproject`` downstream of ``util.stitch``: the same numbers, at a cost that
    scales with the mosaic instead of with the mosaic times its own tile grid.

    Three separate faults made "Max Z onto a stitch" hang rather than run, and each one is
    invisible in a correctness test — the pixels were right the whole time:

    1. ``_AxisReduceProvider`` tiled its base. Every output tile pulled a matching window
       from the ``MultiViewProvider``, every one of those re-stitched the mosaic from source
       planes, and none was cached: ``n_out_tiles × Z × n_m`` full plane reads instead of
       ``Z × n_m``. On a 13106² canvas (26×26 tiles at 512) that is a ~350× multiplier.
    2. ``_stitch_plane``'s pre-read skip tested only the far edge, so a windowed stitch read
       every tile lying *before* the window too — the negative-offset mirror case.
    3. The reduce had no pyramid, so the Viewer could only read level 0 — Z full-canvas
       stitches to fill a 3.5 Mpx widget.

    So this test counts reads and levels, not just values. The read counter is the point: an
    assertion on bytes would have passed before the fix."""
    if not _HAVE_SKIMAGE:
        _ok("zproject-over-stitch: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.streaming import MultiViewProvider, ZReduceProvider
    from nodelab_v2.runner import render_plane_native

    # A 3×3 mosaic with 25% overlap, tiles big enough that the canvas spans a REAL grid of
    # the 512 default (2×2 output tiles) — with a one-tile canvas the bug cannot appear.
    TH = TW = 320
    STEP = 240                                   # 80 px overlap
    PX, N_Z = 2.0, 4
    OFFS = [(iy * STEP, ix * STEP) for iy in range(3) for ix in range(3)]
    H, W, N_M = 2 * STEP + TH, 2 * STEP + TW, len(OFFS)
    assert H > 512 and W > 512, (H, W)            # the grid the multiplier lived on
    canvas = np.stack([_stitch_texture(H, W) + 40.0 * z for z in range(N_Z)])
    tiles = np.zeros((N_M, 1, N_Z, 1, TH, TW))
    for m, (oy, ox) in enumerate(OFFS):
        for z in range(N_Z):
            tiles[m, 0, z, 0] = canvas[z][oy:oy + TH, ox:ox + TW]
    stage = [(-ox * PX, oy * PX) for (oy, ox) in OFFS]
    ax = AxisSizes(m=N_M, t=1, z=N_Z, c=1, y=TH, x=TW)

    class _Counting(ArrayProvider):
        """Counts source-plane reads — the quantity that actually broke."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.reads = 0

        def read_region(self, level, m, t, z, c, y0, y1, x0, x1):
            self.reads += 1
            return super().read_region(level, m, t, z, c, y0, y1, x0, x1)

    env = MetaEnvelope(axes=ax, metadata={"pixel_size_um": PX})
    define_node("io.seedZS", "Seed", outputs=[OutDataset()])

    def chain(prov, method="max", blend="feather", reverse=False):
        """``load → stitch → zproject``, or with ``reverse`` the other order."""
        ds = Dataset(axes=ax, metadata={"pixel_size_um": PX, "stage_xy_um": stage}
                     ).with_image(prov)
        g = Graph()
        g.add(NodeInstance("S", "io.seedZS"))
        g.add(NodeInstance("T", "util.stitch", modes={"blend": blend}))
        g.add(NodeInstance("Z", "util.zproject", modes={"method": method}))
        if reverse:
            g.connect("S", "Z")
            g.connect("Z", "T")
        else:
            g.connect("S", "T")
            g.connect("T", "Z")
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})

    # ── 1. the cost fence: one whole output plane costs exactly Z × M reads ────
    # Not "fewer than before" — the IDEAL, because every source plane is needed once per z
    # and the fold visits each exactly once. Anything above it is a multiplier that will
    # grow with the canvas.
    for method in ("max", "mean", "median"):       # monoid, monoid, and the stacking path
        prov = _Counting(tiles)
        p = chain(prov, method).pull("Z").image
        assert isinstance(p, ZReduceProvider), type(p)
        assert p.plane_unit, "a reduce over a MULTI_VIEW base must fold whole planes"
        a = p.axes
        assert (a.z, a.y, a.x) == (1, H, W), (a.z, a.y, a.x)
        prov.reads = 0
        got = np.asarray(p.get_region(0, 0, 0, 0, 0, 0, a.y, 0, a.x))
        assert prov.reads == N_Z * N_M, \
            f"{method}: {prov.reads} source reads for one plane, ideal {N_Z * N_M} " \
            f"(a {prov.reads / (N_Z * N_M):.0f}x multiplier is back)"
        # and the value is still the reduce of the stitched planes, plane for plane
        sp = chain(_Counting(tiles), method).pull("T").image
        ref = reduce(np.stack([np.asarray(sp.get_region(0, 0, 0, z, 0, 0, H, 0, W))
                                for z in range(N_Z)]), (0,), method)
        assert _stream_eq(got, ref), f"{method}: the folded plane != the reduce of the planes"
        # a window of it is that slice of the whole (the plane path must not shift anything)
        win = np.asarray(p.get_region(0, 0, 0, 0, 0, 17, 600, 33, 590))
        assert _stream_eq(win, got[17:600, 33:590]), f"{method}: window != its slice"

    # ── 2. a windowed stitch reads only the tiles that TOUCH the window ────────
    # Both directions. A tile before the window has a negative shifted offset, which the
    # old far-edge-only test let through — it was read in full and then discarded.
    prov = _Counting(tiles)
    sp = chain(prov).pull("T").image
    assert isinstance(sp, MultiViewProvider), type(sp)
    for (y0, y1, x0, x1), want, note in (
            ((0, 24, 0, 24), 1, "top-left corner (1 tile)"),
            ((H - 24, H, W - 24, W), 1, "bottom-right corner (the mirror case)"),
            ((H - 24, H, 0, 24), 1, "bottom-left"),
            ((0, 24, W - 24, W), 1, "top-right")):
        e2 = chain(_Counting(tiles))
        p2 = e2.pull("T").image
        p2._base.reads = 0
        w = np.asarray(p2.get_region(0, 0, 0, 0, 0, y0, y1, x0, x1))
        assert p2._base.reads == want, \
            f"{note}: read {p2._base.reads} tiles for a window {want} tile(s) touch"
        assert w.shape == (y1 - y0, x1 - x0), (note, w.shape)

    # ── 3. the reduce forwards the base's pyramid, and level 0 stays EXACT ─────
    # The whole safety argument: coarse levels are display-only, so they may approximate;
    # level 0 may not, and every compute/realize/export reads it.
    from nodegraph.provider import B2ndProvider
    pyr_tiles = B2ndProvider.from_array(tiles, tile=128, levels=2)
    for method, exact in (("mean", True), ("sum", True), ("max", False)):
        p = chain(pyr_tiles, method).pull("Z").image
        assert p.levels == 2, (method, p.levels)
        l0, l1 = p.level_axes(0), p.level_axes(1)
        assert (l0.y, l0.x) == (H, W) and (l0.z, l1.z) == (1, 1)
        assert (l1.y, l1.x) == (H // 2, W // 2), (method, l1)
        lv0 = np.asarray(p.get_region(0, 0, 0, 0, 0, 0, l0.y, 0, l0.x))
        # level 0 is exactly the reduce of the level-0 stitched planes
        sp = chain(pyr_tiles, method).pull("T").image
        ref = reduce(np.stack([np.asarray(sp.get_region(0, 0, 0, z, 0, 0, H, 0, W))
                                for z in range(N_Z)]), (0,), method)
        assert _stream_eq(lv0, ref), f"{method}: level 0 is not exact"
        lv1 = np.asarray(p.get_region(1, 0, 0, 0, 0, 0, l1.y, 0, l1.x))
        assert lv1.shape == (H // 2, W // 2), (method, lv1.shape)
        half = lv0[:H // 2 * 2, :W // 2 * 2].reshape(
            H // 2, 2, W // 2, 2).mean(axis=(1, 3))
        err = float(np.abs(lv1 - half).max()) / max(1e-12, float(np.ptp(half)))
        if exact:
            # a sum/mean fold COMMUTES with the xy mean-pool — bit-identical, not merely close
            assert err < 1e-12, f"{method}: level 1 must equal the downsampled level 0 " \
                                f"(relerr {err:.2e})"
        else:
            # max/min/median do not commute; the deviation is bounded by the xy variation
            # inside one 2×2 block, and it is one-sided (max of means ≤ mean of maxes)
            assert err < 0.15, f"{method}: level 1 deviates {err:.2%} — more than the " \
                               f"within-block variation can explain"
            assert lv1.max() <= half.max() + 1e-9, "a coarse max must not EXCEED the fine one"
        # a coarse window is that slice of the coarse plane
        assert np.array_equal(np.asarray(p.get_region(1, 0, 0, 0, 0, 5, 40, 7, 44)),
                              lv1[5:40, 7:44]), f"{method}: coarse window != its slice"
    # the display path takes the coarse level — the reported symptom, end to end
    p = chain(pyr_tiles, "max").pull("Z").image
    shown, lv = render_plane_native(p, 0, 0, 0, 0, max_dim=W // 2)
    assert lv == 1, f"the Viewer stayed on level 0 ({lv}) — the reduce pyramid is not wired"
    assert shown.shape == (H // 2, W // 2), shown.shape
    try:
        p.get_region(2, 0, 0, 0, 0, 0, 4, 0, 4)
        raise AssertionError("an out-of-range level must raise")
    except ValueError:
        pass
    # a pyramid-less base (any computed intermediate) stays level-0 only
    flat = chain(_Counting(tiles), "max").pull("Z").image
    assert flat.levels == 1, flat.levels
    try:
        flat.level_axes(1)
        raise AssertionError("a pyramid-less reduce must not claim a level 1")
    except ValueError:
        pass

    # ── 3b. THE OTHER ORDER: stitch ON a projection ────────────────────────────
    # A reduce over a real tiled STORE must keep its per-tile path (tiling a store is what
    # tiling is for — the plane_unit promotion is about a plane-unit base, not about being a
    # reduce), the stitch above it must still read every source plane exactly once, and — new
    # in V2.20, because the reduce now forwards levels — the mosaic KEEPS its coarse levels
    # through the projection. Before, an axis reduce reported `levels = 1` and so silently
    # cost the stitch above it its pyramid, which is the "put Stitch early" penalty.
    rev = chain(pyr_tiles, "max", reverse=True).pull("T").image
    assert isinstance(rev, MultiViewProvider), type(rev)
    assert rev.levels == 2, \
        f"a stitch above a projection lost its pyramid (levels={rev.levels})"
    inner = rev._base
    assert isinstance(inner, ZReduceProvider) and not inner.plane_unit, \
        "a reduce over a tiled STORE must keep its per-tile path"
    ra = rev.axes
    assert (ra.m, ra.z, ra.y, ra.x) == (1, 1, H, W), (ra.m, ra.z, ra.y, ra.x)
    cnt = _Counting(tiles)
    rev_c = chain(cnt, "max", reverse=True).pull("T").image
    cnt.reads = 0
    rv = np.asarray(rev_c.get_region(0, 0, 0, 0, 0, 0, ra.y, 0, ra.x))
    assert cnt.reads == N_Z * N_M, \
        f"stitch-on-projection: {cnt.reads} source reads, ideal {N_Z * N_M}"
    # The two orders are NOT interchangeable in general, and the fixture says which way.
    # Its tiles are exact cuts of one canvas, so overlapping tiles AGREE and every blend
    # commutes — pin that, since it is the case a user's tiles approximate whenever they
    # differ only by a z-independent gain (vignetting, exposure).
    fwd = np.asarray(chain(_Counting(tiles), "max").pull("Z").image
                     .get_region(0, 0, 0, 0, 0, 0, H, 0, W))
    assert _stream_eq(rv, fwd), \
        "with agreeing tiles the two orders must give the same canvas"
    # ...and `max`/`overwrite` commute UNCONDITIONALLY (a max/selection over m commutes with
    # a max over z), which is the guarantee worth pinning rather than the fixture's accident.
    skew = tiles.copy()
    skew[1::2, 0, :, 0] *= 1.35                # a z-independent per-tile gain: still commutes
    skew[2::3, 0, :, 0] += 40.0 * np.arange(N_Z)[:, None, None]   # z-DEPENDENT: must not
    for blend, must_match in (("max", True), ("overwrite", True), ("feather", False)):
        a1 = np.asarray(chain(_Counting(skew), "max", blend).pull("Z").image
                        .get_region(0, 0, 0, 0, 0, 0, H, 0, W))
        a2 = np.asarray(chain(_Counting(skew), "max", blend, reverse=True).pull("T").image
                        .get_region(0, 0, 0, 0, 0, 0, H, 0, W))
        if must_match:
            assert _stream_eq(a1, a2), \
                f"blend={blend} must commute with a z-reduce in either order"
        else:
            # convexity fixes the direction: max of a weighted mean <= weighted mean of maxes
            assert (a1 <= a2 + 1e-9).all(), \
                "zproject(stitch) must never EXCEED stitch(zproject) under a mean blend"

    # ── 4. the plane_unit fence is not stitch-specific ─────────────────────────
    # It is a property of the BASE's cost model, so it must ride through a lazy crop and
    # must NOT fire for an ordinary tiled base (which tiling genuinely helps).
    from dataclasses import replace
    from nodegraph.streaming import WindowView
    sp = chain(_Counting(tiles)).pull("T").image
    assert WindowView(sp, y0=8, x0=8, axes=replace(sp.axes, y=H - 8, x=W - 8)).plane_unit, \
        "a crop of a plane-unit provider is still plane-unit"
    plain = Engine(_zs_plain_graph(), computes=COMPUTES,
                   seeds={"S": Dataset(axes=ax).with_image(ArrayProvider(tiles, tile=128))},
                   meta_seeds={"S": MetaEnvelope(axes=ax)}).pull("Z").image
    assert isinstance(plain, ZReduceProvider) and not plain.plane_unit, \
        "a reduce over a plain tiled store must keep its per-tile path"

    _ok("zproject-over-stitch (V2.20): Max/Mean/Median Z on a %dx%d 3x3 mosaic costs "
        "EXACTLY Z*M=%d source-plane reads for a whole output plane — the reduce folds "
        "whole planes when its base is plane_unit instead of tiling it, which is what "
        "turned a 2x2-tile grid into a 4x multiplier here and a ~350x one on a 26x26 "
        "mosaic; the folded plane equals the reduce of the stitched planes and a window "
        "equals its slice; a windowed stitch now reads only the 1 tile a corner touches "
        "instead of every tile lying before it (the negative-offset mirror of the old "
        "far-edge-only skip, checked at all 4 corners); the reduce forwards the base's "
        "PYRAMID so the Viewer takes level 1 (it read level 0 = Z full-canvas stitches "
        "per frame before) with level 0 still bit-exact for every reducer, sum/mean "
        "commuting with the mean-pool exactly and max bounded + one-sided; the OTHER ORDER "
        "(stitch ON a projection) reads every source plane exactly once too, keeps its "
        "per-tile path over a real store, and now KEEPS its coarse levels through the "
        "projection (a reduce reporting levels=1 used to cost the stitch above it its "
        "pyramid), where max/overwrite commute in either order unconditionally while a mean "
        "blend does not once the tiles disagree as a function of z — and never exceeds the "
        "other way round; and the fence rides through a lazy crop while a plain tiled base "
        "keeps its per-tile path"
        % (H, W, N_Z * N_M))


def _zs_plain_graph() -> Graph:
    """A bare seed → z-project graph (no stitch) — the control for the ``plane_unit``
    fence: over an ordinary tiled store the per-tile path is the right one and must stay."""
    g = Graph()
    g.add(NodeInstance("S", "io.seedZS"))
    g.add(NodeInstance("Z", "util.zproject", modes={"method": "max"}))
    g.connect("S", "Z")
    return g


# ── Phase 4a: Repeat / Simulation zones (unroll + revision-fold) ──────────────

def test_zones() -> None:
    if not _HAVE_SKIMAGE:
        _ok("zones: SKIPPED (needs enhance.gamma → scipy/skimage)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.zones import Zone, iter_id, unroll, zone_output, assert_zone_pure

    define_node("io.seedZ", "Seed", outputs=[OutDataset()])

    # ── Repeat zone: iterate Gamma N times ──────────────────────────────────
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
    grad = (np.arange(16, dtype=float).reshape(4, 4) + 1.0)      # 1..16 (non-flat)
    img = grad.reshape(1, 1, 1, 1, 4, 4)
    ds = Dataset(axes=ax).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax)

    def repeat_graph(n):
        g = Graph()
        g.add(NodeInstance("S", "io.seedZ"))
        g.add(NodeInstance("RIN", "zone.repeat_in"))
        g.add(NodeInstance("G", "enhance.gamma", params={"gamma": 2.0}))
        g.add(NodeInstance("ROUT", "zone.repeat_out"))
        g.connect("S", "RIN")                       # external seed → In (iteration 0)
        g.connect("RIN", "G")
        g.connect("G", "ROUT")
        g.connect("ROUT", "RIN", kind="back")       # feedback (Out → In)
        z = Zone("Z", "repeat", "RIN", "ROUT", body=frozenset({"G"}), iterations=n)
        return g, z

    g, z = repeat_graph(3)
    flat = unroll(g, [z])
    g.topo_order()                    # the zoned graph orders fine (back-edge is skipped)
    zout = zone_output(z)             # "ROUT#Z@2"

    def eng(memo=None):
        return Engine(flat, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                      memo=memo)
    out = eng().pull(zout)
    got = out.image.get_region(0, 0, 0, 0, 0, 0, 4, 0, 4)
    mx = grad.max()
    expected = (grad / mx) ** (2.0 ** 3) * mx                    # γ=2 applied 3× → exp 2³=8
    assert np.allclose(got, expected), "Repeat zone must apply the body N times"

    # revision-fold caching: re-pull on a SHARED memo recomputes nothing (the chain's
    # keys are stable — iteration i folds iteration i-1's revision, which is unchanged)
    memo = Memo()
    eng(memo).pull(zout)
    e2 = eng(memo)
    e2.pull(zout)
    assert e2.compute_count == 0, "an unchanged zone must be fully memo-cached on re-pull"
    # debug-verify: the pure Gamma body is deterministic
    assert_zone_pure(lambda: eng(), zout)

    # ── Simulation zone: feedback ACCUMULATES across iterations ──────────────
    def _accum(ctx):                                   # out = in + 1, per iteration
        prov = ctx.inputs[0].image
        a = prov.axes
        o = prov.get_region_volume(0, 0, 0, 0, 0, a.z, 0, a.y, 0, a.x).astype(float) + 1.0
        return ctx.inputs[0].with_image(ArrayProvider(o.reshape(1, 1, a.z, 1, a.y, a.x)))

    define_node("test.accumulate", "Accumulate", inputs=[InDataset()], outputs=[OutDataset()],
                granularity=Granularity.TILEABLE)
    merged = {**COMPUTES, "test.accumulate": _accum}
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)
    ds0 = Dataset(axes=ax1).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 2, 2))))
    env1 = MetaEnvelope(axes=ax1)

    def sim_graph(n, impure=False):
        g = Graph()
        g.add(NodeInstance("S", "io.seedZ"))
        g.add(NodeInstance("SIN", "zone.sim_in"))
        g.add(NodeInstance("ACC", "test.accumulate"))
        g.add(NodeInstance("SOUT", "zone.sim_out"))
        g.connect("S", "SIN"); g.connect("SIN", "ACC"); g.connect("ACC", "SOUT")
        g.connect("SOUT", "SIN", kind="back")
        z = Zone("Z", "sim", "SIN", "SOUT", body=frozenset({"ACC"}), iterations=n,
                 impure=impure)
        return g, z

    gs, zs = sim_graph(4)
    fs = unroll(gs, [zs])
    sim_out = Engine(fs, computes=merged, seeds={"S": ds0}, meta_seeds={"S": env1}).pull(
        zone_output(zs))
    acc = sim_out.image.get_region(0, 0, 0, 0, 0, 0, 2, 0, 2)
    assert np.all(acc == 4.0), "Sim zone must accumulate feedback (0 + 4 iterations)"
    # each iteration's In folds the prior Out's revision → distinct per-iteration keys
    esim = Engine(fs, computes=merged, seeds={"S": ds0}, meta_seeds={"S": env1})
    esim.pull(zone_output(zs))
    assert (esim.entry(iter_id("ACC", "Z", 1)).recipe_hash
            != esim.entry(iter_id("ACC", "Z", 2)).recipe_hash)

    # ── impure escape hatch: an impure zone re-keys per epoch (non-cacheable);
    #    a pure one ignores epoch (stays cached across pulls) ──────────────────
    gi, zi = sim_graph(4, impure=True)
    memo_i = Memo()
    Engine(unroll(gi, [zi], epoch=0), computes=merged, seeds={"S": ds0},
           meta_seeds={"S": env1}, memo=memo_i).pull(zone_output(zi))
    eimp = Engine(unroll(gi, [zi], epoch=1), computes=merged, seeds={"S": ds0},
                  meta_seeds={"S": env1}, memo=memo_i)
    eimp.pull(zone_output(zi))
    assert eimp.compute_count > 0, "impure zone must recompute under a new epoch"

    gp, zp = sim_graph(4, impure=False)
    memo_p = Memo()
    Engine(unroll(gp, [zp], epoch=0), computes=merged, seeds={"S": ds0},
           meta_seeds={"S": env1}, memo=memo_p).pull(zone_output(zp))
    epure = Engine(unroll(gp, [zp], epoch=1), computes=merged, seeds={"S": ds0},
                   meta_seeds={"S": env1}, memo=memo_p)
    epure.pull(zone_output(zp))
    assert epure.compute_count == 0, "pure zone must ignore epoch (stay cached)"

    _ok("zones: Repeat (N× body) + Simulation (feedback accumulate); revision-fold "
        "caching + incremental keys; impure epoch escape hatch; debug-verify")


# ── Phase 4a: per-frame-T Simulation specialization (zone.frame) ──────────────

def test_sim_perframe() -> None:
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.zones import Zone, iter_id, unroll, zone_output

    def _add2(ctx):                                    # out = state + current frame
        pa, pb = ctx.inputs[0].image, ctx.inputs[1].image
        ax = pa.axes
        va = pa.get_region(0, 0, 0, 0, 0, 0, ax.y, 0, ax.x)
        vb = pb.get_region(0, 0, 0, 0, 0, 0, ax.y, 0, ax.x)
        return ctx.inputs[0].with_image(
            ArrayProvider((va + vb).reshape(1, 1, 1, 1, ax.y, ax.x)))

    define_node("io.seedPF", "Seed", outputs=[OutDataset()])
    define_node("io.stackPF", "Stack", outputs=[OutDataset()])
    define_node("test.add2", "Add2", inputs=[InDataset("a"), InDataset("b")],
                outputs=[OutDataset()], granularity=Granularity.TILEABLE)

    seed = Dataset(axes=AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)).with_image(
        ArrayProvider(np.zeros((1, 1, 1, 1, 2, 2))))          # state re-init at t0
    stack_arr = np.zeros((1, 3, 1, 1, 2, 2))
    stack_arr[0, 0] = 1.0; stack_arr[0, 1] = 2.0; stack_arr[0, 2] = 3.0
    stack = Dataset(axes=AxisSizes(m=1, t=3, z=1, c=1, y=2, x=2)).with_image(
        ArrayProvider(stack_arr))                             # T=3 series

    g = Graph()
    g.add(NodeInstance("SEED", "io.seedPF"))
    g.add(NodeInstance("STACK", "io.stackPF"))
    g.add(NodeInstance("SIN", "zone.sim_in"))
    g.add(NodeInstance("FR", "zone.frame"))                   # per-frame slicer (body)
    g.add(NodeInstance("ADD", "test.add2"))
    g.add(NodeInstance("SOUT", "zone.sim_out"))
    g.connect("SEED", "SIN")                                  # state seed → In (iter0 only)
    g.connect("STACK", "FR")                                 # full T-stack → slicer (every iter)
    g.connect("SIN", "ADD", dst_socket="a")                  # carried state
    g.connect("FR", "ADD", dst_socket="b")                   # current frame t
    g.connect("ADD", "SOUT")
    g.connect("SOUT", "SIN", kind="back")                    # feedback
    z = Zone("Z", "sim", "SIN", "SOUT", body=frozenset({"FR", "ADD"}), iterations=3)
    flat = unroll(g, [z])
    seeds = {"SEED": seed, "STACK": stack}
    meta = {"SEED": MetaEnvelope(axes=seed.axes), "STACK": MetaEnvelope(axes=stack.axes)}
    merged = {**COMPUTES, "test.add2": _add2}

    out = Engine(flat, computes=merged, seeds=seeds, meta_seeds=meta).pull(zone_output(z))
    acc = out.image.get_region(0, 0, 0, 0, 0, 0, 2, 0, 2)
    assert np.all(acc == 6.0), f"per-frame Sim must accumulate 1+2+3=6, got {acc}"
    # zone.frame stamps a distinct __frame__ per iteration → distinct memo keys...
    esim = Engine(flat, computes=merged, seeds=seeds, meta_seeds=meta)
    esim.pull(zone_output(z))
    assert (esim.entry(iter_id("FR", "Z", 0)).recipe_hash
            != esim.entry(iter_id("FR", "Z", 2)).recipe_hash)
    # ...and its frame_slice meta_transform makes the envelope t==1 (matches the payload)
    assert esim.env(iter_id("FR", "Z", 1)).axes.t == 1
    _ok("zones per-frame-T: Simulation scan (zone.frame slices frame t; feedback carries "
        "state) accumulates 1+2+3 → 6; per-iter frame keys + meta_transform t→1")


# ── Phase 4b: node groups (inline-expand, nestable) ──────────────────────────

def test_groups() -> None:
    from nodegraph.groups import (Group, GROUP_INPUT, GROUP_OUTPUT, expand,
                                   group_input, group_name_of, group_output)

    def _edges(g):
        return {(e.src, e.dst, e.src_socket, e.dst_socket, e.kind) for e in g.edges}

    # ── simple group: body = (Group Input → inner → Group Output) ────────────
    body = Graph()
    body.add(NodeInstance("IN", GROUP_INPUT))
    body.add(NodeInstance("MID", "test.inner"))
    body.add(NodeInstance("OUT", GROUP_OUTPUT))
    body.connect("IN", "MID"); body.connect("MID", "OUT")
    G = Group(name="blur", body=body, input_id="IN", output_id="OUT")
    assert G.op_key == "group:blur" and group_name_of("group:blur") == "blur"

    parent = Graph()
    parent.add(NodeInstance("SRC", "test.src"))
    parent.add(NodeInstance("g0", G.op_key))            # the group INSTANCE placeholder
    parent.add(NodeInstance("SNK", "test.sink"))
    parent.connect("SRC", "g0"); parent.connect("g0", "SNK")

    flat = expand(parent, [G])
    assert "g0" not in flat.nodes                                   # instance replaced
    assert set(flat.nodes) == {"SRC", "SNK", "IN%g0", "MID%g0", "OUT%g0"}
    assert flat.nodes["MID%g0"].op_key == "test.inner"             # body copied w/ unique id
    es = _edges(flat)
    assert ("SRC", "IN%g0", "out", "data", "forward") in es        # external in → input boundary
    assert ("IN%g0", "MID%g0", "out", "data", "forward") in es     #            → inner
    assert ("MID%g0", "OUT%g0", "out", "data", "forward") in es    # inner → output boundary
    assert ("OUT%g0", "SNK", "out", "data", "forward") in es       #       → external out
    assert group_output(G, "g0") == "OUT%g0" and group_input(G, "g0") == "IN%g0"
    flat.topo_order()                                              # a well-formed DAG

    # ── nested group: Outer's body contains an Inner group instance ──────────
    ib = Graph()
    ib.add(NodeInstance("IIN", GROUP_INPUT)); ib.add(NodeInstance("N", "test.leaf"))
    ib.add(NodeInstance("IOUT", GROUP_OUTPUT))
    ib.connect("IIN", "N"); ib.connect("N", "IOUT")
    Inner = Group(name="inner", body=ib, input_id="IIN", output_id="IOUT")
    ob = Graph()
    ob.add(NodeInstance("OIN", GROUP_INPUT)); ob.add(NodeInstance("MID", Inner.op_key))
    ob.add(NodeInstance("OOUT", GROUP_OUTPUT))
    ob.connect("OIN", "MID"); ob.connect("MID", "OOUT")
    Outer = Group(name="outer", body=ob, input_id="OIN", output_id="OOUT")
    p2 = Graph()
    p2.add(NodeInstance("S", "test.src")); p2.add(NodeInstance("G", Outer.op_key))
    p2.add(NodeInstance("K", "test.sink"))
    p2.connect("S", "G"); p2.connect("G", "K")
    flat2 = expand(p2, [Outer, Inner])
    assert "G" not in flat2.nodes and "MID" not in flat2.nodes     # both instances gone
    assert not any(n.op_key.startswith("group:") for n in flat2.nodes.values())  # fixed point
    assert flat2.nodes["N%MID%G"].op_key == "test.leaf"           # innermost leaf, unique id
    es2 = _edges(flat2)
    for a, b in [("S", "OIN%G"), ("OIN%G", "IIN%MID%G"), ("IIN%MID%G", "N%MID%G"),
                 ("N%MID%G", "IOUT%MID%G"), ("IOUT%MID%G", "OOUT%G"), ("OOUT%G", "K")]:
        assert (a, b, "out", "data", "forward") in es2, (a, b)
    flat2.topo_order()

    # ── error paths: unknown ref / self-recursion / malformed definition ─────
    def _rejects(fn, needle):
        try:
            fn(); raise AssertionError(f"expected ValueError ({needle})")
        except ValueError as ex:
            assert needle in str(ex), (needle, str(ex))

    bad = Graph(); bad.add(NodeInstance("x", "group:nope"))
    _rejects(lambda: expand(bad, [G]), "unknown group")
    sb = Graph()
    sb.add(NodeInstance("SI", GROUP_INPUT)); sb.add(NodeInstance("SELF", "group:loop"))
    sb.add(NodeInstance("SO", GROUP_OUTPUT)); sb.connect("SI", "SELF"); sb.connect("SELF", "SO")
    Loop = Group(name="loop", body=sb, input_id="SI", output_id="SO")
    pl = Graph(); pl.add(NodeInstance("L", "group:loop"))
    _rejects(lambda: expand(pl, [Loop]), "recursively nested")
    mb = Graph()
    mb.add(NodeInstance("BI", "not.group.input")); mb.add(NodeInstance("BO", GROUP_OUTPUT))
    Malf = Group(name="malf", body=mb, input_id="BI", output_id="BO")
    pm = Graph(); pm.add(NodeInstance("M", "group:malf"))
    _rejects(lambda: expand(pm, [Malf]), "op_key")

    # ── end-to-end: a group wrapping enhance.gamma runs through the Engine ────
    if _HAVE_SKIMAGE:
        from nodegraph.provider import ArrayProvider
        from nodegraph.nodes import COMPUTES
        gb = Graph()
        gb.add(NodeInstance("GI", GROUP_INPUT))
        gb.add(NodeInstance("GG", "enhance.gamma", params={"gamma": 2.0}))
        gb.add(NodeInstance("GO", GROUP_OUTPUT))
        gb.connect("GI", "GG"); gb.connect("GG", "GO")
        grp = Group(name="gammagrp", body=gb, input_id="GI", output_id="GO")
        define_node("io.seedG", "Seed", outputs=[OutDataset()])
        axg = AxisSizes(m=1, t=1, z=1, c=1, y=4, x=4)
        imgg = (np.arange(16.0) + 1).reshape(1, 1, 1, 1, 4, 4)
        dsg = Dataset(axes=axg).with_image(ArrayProvider(imgg))
        pg = Graph()
        pg.add(NodeInstance("S", "io.seedG")); pg.add(NodeInstance("g0", grp.op_key))
        pg.connect("S", "g0")
        flatg = expand(pg, [grp])
        outg = Engine(flatg, computes=COMPUTES, seeds={"S": dsg},
                      meta_seeds={"S": MetaEnvelope(axes=axg)}).pull(group_output(grp, "g0"))
        got = outg.image.get_region(0, 0, 0, 0, 0, 0, 4, 0, 4)
        mx = imgg[0, 0, 0, 0].max()
        assert np.allclose(got, (imgg[0, 0, 0, 0] / mx) ** 2.0 * mx), "group body must run"

    _ok("groups: simple + nested expand (interface stitched, unique ids, fixed point); "
        "unknown-ref / recursion / malformed rejected; body runs end-to-end")


# ── frame-to-frame tracking: the TrackMembership PRODUCER (C3) ────────────────

def test_tracking() -> None:
    from nodegraph.tracking import link_labels, link_points
    from nodegraph.bridges import gather_by_track
    from nodegraph.nodes import COMPUTES

    # LABEL: id 10@t0 → 20@t1 → 30@t2 (one track); a 2nd object 40@t1 → 50@t2
    # (a track starting mid-series). Regions overlap by maximum IoU.
    r0 = np.zeros((5, 5), int); r0[1:3, 1:3] = 10
    r1 = np.zeros((5, 5), int); r1[1:3, 1:3] = 20; r1[3:5, 3:5] = 40
    r2 = np.zeros((5, 5), int); r2[1:3, 1:3] = 30; r2[3:5, 3:5] = 50
    mem = link_labels({0: r0, 1: r1, 2: r2})
    assert mem.member_domain is D.LABEL and mem.n == 5
    assert mem.track_ids().tolist() == [1, 2] and mem.timepoints().tolist() == [0, 1, 2]
    rows = set(zip(mem.track_id.tolist(), mem.t.tolist(), mem.member_id.tolist()))
    assert rows == {(1, 0, 10), (1, 1, 20), (1, 2, 30), (2, 1, 40), (2, 2, 50)}
    # produced membership is consumable by the Track bridge it feeds
    tids, means = gather_by_track([10, 20, 30, 40, 50], [1., 2., 3., 4., 5.], mem, "mean")
    assert tids.tolist() == [1, 2] and np.allclose(means, [2.0, 4.5])
    # IoU threshold above 1.0 breaks every chain → all singletons
    assert link_labels({0: r0, 1: r1, 2: r2}, iou_threshold=1.0001).track_ids().tolist() \
        == [1, 2, 3, 4, 5]

    # POINT: nearest-neighbour. track1 1@t0→3@t1→5@t2 ; track2 2@t0→4@t1 (dies at t2)
    pos = {0: ([1, 2], np.array([[0., 0.], [10., 10.]])),
           1: ([3, 4], np.array([[1., 0.], [10., 11.]])),
           2: ([5], np.array([[2., 0.]]))}
    pmem = link_points(pos, max_distance=5.0)
    assert pmem.member_domain is D.POINT
    prows = set(zip(pmem.track_id.tolist(), pmem.t.tolist(), pmem.member_id.tolist()))
    assert prows == {(1, 0, 1), (1, 1, 3), (1, 2, 5), (2, 0, 2), (2, 1, 4)}

    # track.link node end-to-end through the Engine (label mode)
    define_node("io.seedTk", "Seed", outputs=[OutDataset()])
    ax = AxisSizes(m=1, t=3, z=1, c=1, y=5, x=5)
    raster = np.zeros((1, 3, 1, 1, 5, 5), np.int64)
    raster[0, 0, 0, 0, 1:3, 1:3] = 10
    raster[0, 1, 0, 0, 1:3, 1:3] = 20; raster[0, 1, 0, 0, 3:5, 3:5] = 40
    raster[0, 2, 0, 0, 1:3, 1:3] = 30; raster[0, 2, 0, 0, 3:5, 3:5] = 50
    ds = Dataset(axes=ax).with_layer(D.VOXEL, "labels", raster)
    g = Graph(); g.add(NodeInstance("S", "io.seedTk"))
    g.add(NodeInstance("K", "track.link", modes={"target": "label"})); g.connect("S", "K")
    out = Engine(g, computes=COMPUTES, seeds={"S": ds},
                 meta_seeds={"S": MetaEnvelope(axes=ax, metadata={})}).pull("K")
    nrows = set(zip(out.get(D.TRACK, "track_id", layer="tracks").values.tolist(),
                    out.get(D.TRACK, "t", layer="tracks").values.tolist(),
                    out.get(D.TRACK, "member_id", layer="tracks").values.tolist()))
    assert nrows == rows
    _ok("tracking: link_labels (IoU overlap) + link_points (nearest-neighbour) produce "
        "TrackMembership (gather-consumable); track.link node attaches it end-to-end")


def test_track_objects() -> None:
    """``track.objects`` — the vendored v1 ``track_objects`` kernel (five interchangeable
    linkers) as the richer sibling of ``track.link``. Covers the structural spec, a real
    end-to-end pull on **all five** methods for both Label and Point members, the
    per-(m,c,z) plane independence that makes a 2D-only kernel correct on a z-stack, the
    ``track_id`` write-back alignment, the shared membership conventions, determinism
    (incl. shuffle-invariance — the one property that will silently rot), and every hard
    refusal where the kernel would otherwise degrade in silence.

    Gated on numba+pandas: the vendored kernel imports both at module scope, so even the
    centroid linker is unimportable without them."""
    import importlib.util as _u
    from nodegraph.structure import StructureTable
    from nodegraph.nodes import COMPUTES

    # ── structural spec ───────────────────────────────────────────────────────
    s = NODES.get("track.objects")
    assert s.category == "analysis"
    assert D.TRACK in s.adds_domains
    assert s.granularity is Granularity.WHOLE_SERIES
    assert s.kernel_axes == frozenset({"t", "z", "y", "x"})
    assert s.meta_transform is None                     # never changes axes/calibration
    names = {i.name for i in s.inputs}
    assert {"labels", "points", "name", "max_distance", "min_track_length",
            "ct_min_iou", "st_n_neighbors"} <= names
    assert not any(i.name in {"min_circularity", "max_eccentricity"} for i in s.inputs), \
        "morphology sockets must stay ABSENT until a producer emits those columns"
    assert all(not i.is_field for i in s.inputs if i.type is not SocketType.DATASET), \
        "no compute in this class evaluates a wired Field — field=True would be a lie"
    mode_of = {mm.name: mm for mm in s.modes}
    assert set(mode_of) == {"target", "method"} and mode_of["method"].default == "centroid"
    assert set(mode_of["method"].choices) == {"centroid", "serialtrack", "topology",
                                              "fingerprint", "overlap"}
    # method-gated variant sockets resolve per mode state (registry available_in)
    vis = lambda st: {i.name for i in s.active_inputs(st)}
    assert "ct_min_iou" in vis({"method": "overlap"})
    assert "ct_min_iou" not in vis({"method": "centroid"})
    assert "st_n_neighbors" in vis({"method": "serialtrack"})
    assert "max_frame_gap" in vis({"method": "centroid"})
    assert "labels" in vis({"target": "label"}) and "labels" not in vis({"target": "point"})
    # review: a socket the chosen linker never receives must be HIDDEN, not accepted and
    # discarded. `overlap` matches by mask IoU and takes no distance bound; the centroid
    # size gate is inert on arealess (Point) members.
    assert "max_distance" in vis({"method": "centroid"})
    assert "max_distance" not in vis({"method": "overlap"})
    assert "max_size_diff_frac" in vis({"method": "centroid", "target": "label"})
    assert "max_size_diff_frac" not in vis({"method": "centroid", "target": "point"})

    if _u.find_spec("numba") is None or _u.find_spec("pandas") is None:
        _ok("track.objects: spec OK; RUN SKIPPED (kernel needs numba+pandas)")
        return

    define_node("io.trkobj", "S", outputs=[OutDataset()])
    T, Y, X = 4, 48, 48
    BASE = np.array([[8.0, 8.0], [26.0, 30.0], [38.0, 12.0]])      # 3 objects, +2 px/frame

    def build(target="label", nz=1, perm=None):
        """3 objects drifting +2 px/frame over 4 frames on ``nz`` planes; member ids are
        globally unique. ``perm`` permutes the table's ROW ORDER (same data, different
        presentation) to exercise shuffle-invariance + write-back alignment."""
        ax = AxisSizes(m=1, t=T, z=nz, c=1, y=Y, x=X)
        raster = np.zeros((1, T, nz, 1, Y, X), np.int64)
        cols: dict = {k: [] for k in ("id", "m", "t", "c", "z", "y", "x", "area")}
        nid = 1
        for t in range(T):
            for z in range(nz):
                for (cy, cx) in BASE + t * 2.0:
                    y0, x0 = int(round(cy)) - 2, int(round(cx)) - 2
                    raster[0, t, z, 0, y0:y0 + 5, x0:x0 + 5] = nid
                    cols["id"].append(nid); cols["m"].append(0); cols["t"].append(t)
                    cols["c"].append(0); cols["z"].append(z)
                    cols["y"].append(float(y0 + 2)); cols["x"].append(float(x0 + 2))
                    cols["area"].append(25)
                    nid += 1
        arrs = {k: np.asarray(v) for k, v in cols.items()}
        if perm is not None:
            arrs = {k: v[perm] for k, v in arrs.items()}
        dom = D.LABEL if target == "label" else D.POINT
        if target != "label":
            arrs.pop("area")                            # Point tables carry no area
        layer = "labels" if target == "label" else "spots"
        ds = Dataset(axes=ax)
        if target == "label":
            ds = ds.with_layer(D.VOXEL, "labels", raster)
        return ds.with_structure(StructureTable(dom, arrs, layer=layer,
                                                z_kind="plane_index")), ax

    meta = {"pixel_size_um": 0.5}

    def pull(ds, ax, target="label", method="centroid", **params):
        g = Graph(); g.add(NodeInstance("S", "io.trkobj"))
        g.add(NodeInstance("K", "track.objects", params={"max_distance": 6.0, **params},
                           modes={"target": target, "method": method}))
        g.connect("S", "K")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds},
                   meta_seeds={"S": MetaEnvelope(axes=ax, metadata=meta)})
        return e.pull("K"), e

    def trows(out, layer="tracks"):
        get = lambda n: out.get(D.TRACK, n, layer=layer).values.tolist()
        return set(zip(get("track_id"), get("t"), get("member_id")))

    # ── all five methods, Label members: 3 objects → 3 tracks spanning all 4 frames ──
    hashes = {}
    for meth in ("centroid", "topology", "fingerprint", "overlap", "serialtrack"):
        out, e = pull(*build("label"), method=meth)
        rows = trows(out)
        assert len(rows) == 12, (meth, len(rows))
        assert sorted({r[0] for r in rows}) == [1, 2, 3], (meth, rows)
        lengths = out.get(D.TRACK, "track_length", layer="tracks").values
        assert set(lengths.tolist()) == {4}, (meth, lengths)
        # every member got a real track id written back onto its own layer
        back = out.get(D.LABEL, "track_id", layer="labels").values
        assert back.shape == (12,) and int((back > 0).sum()) == 12, (meth, back)
        hashes[meth] = e.entry("K").recipe_hash

    # ── Point members (no area column ⇒ area_px 0.0; overlap unavailable) ──────
    for meth in ("centroid", "topology", "fingerprint", "serialtrack"):
        out, _ = pull(*build("point"), target="point", method=meth)
        assert sorted({r[0] for r in trows(out)}) == [1, 2, 3], meth
        assert int((out.get(D.POINT, "track_id", layer="spots").values > 0).sum()) == 12

    # ── plane independence: a 2D kernel on a z-stack must NOT link across planes ──
    out, _ = pull(*build("label", nz=2))
    rows = trows(out)
    assert sorted({r[0] for r in rows}) == [1, 2, 3, 4, 5, 6] and len(rows) == 24

    # ── membership conventions identical to track.link (contiguous, sorted) ───
    out, e_ref = pull(*build("label"))
    tid = out.get(D.TRACK, "track_id", layer="tracks").values.tolist()
    tcl = out.get(D.TRACK, "t", layer="tracks").values.tolist()
    mid = out.get(D.TRACK, "member_id", layer="tracks").values.tolist()
    assert sorted(set(tid)) == list(range(1, len(set(tid)) + 1))     # 1..K contiguous
    assert list(zip(tid, tcl, mid)) == sorted(zip(tid, tcl, mid))    # canonical row order
    assert out.structure_zkind(D.LABEL, "labels") == "plane_index"   # §7b not clobbered

    # ── determinism + shuffle-invariance, and write-back stays aligned ─────────
    ref_ids = out.get(D.LABEL, "id", layer="labels").values.tolist()
    ref_back = out.get(D.LABEL, "track_id", layer="labels").values.tolist()
    ref_map = dict(zip(ref_ids, ref_back))
    rng = np.random.default_rng(3)
    perm = rng.permutation(12)
    out_sh, _ = pull(*build("label", perm=perm))
    assert trows(out_sh) == trows(out), "row order changed the tracking result"
    sh_ids = out_sh.get(D.LABEL, "id", layer="labels").values.tolist()
    sh_back = out_sh.get(D.LABEL, "track_id", layer="labels").values.tolist()
    assert dict(zip(sh_ids, sh_back)) == ref_map, "write-back misaligned under a permutation"
    assert sh_ids == np.asarray(ref_ids)[perm].tolist(), "member layer row order disturbed"

    # ── memo: each method/target is a distinct recipe; calibration read is fenced ──
    _, e_pt = pull(*build("point"), target="point")
    hashes["point"] = e_pt.entry("K").recipe_hash
    assert len(set(hashes.values())) == len(hashes), hashes
    assert "pixel_size_um" in dict(e_ref.entry("K").reads)   # µm→px conversion memo-fenced
    assert out.metadata["track_method"] == "centroid"        # §7b provenance stamp
    assert out.metadata["track_member_domain"] == "label"
    assert out.metadata["track_member_layer"] == "labels"

    # ── hard refusals (each one is a silent kernel degradation if not caught) ──
    def refuses(fn, needle):
        try:
            fn()
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
            return
        raise AssertionError(f"expected a refusal mentioning {needle!r}")

    refuses(lambda: pull(*build("label"), method="bogus"), "unknown method")
    refuses(lambda: pull(*build("point"), target="point", method="overlap"),
            "unavailable for Point members")
    refuses(lambda: pull(*build("point"), target="point", max_size_diff_frac=0.1),
            "silently inert")                       # arealess members ⇒ gate does nothing
    refuses(lambda: pull(*build("label"), method="overlap", ct_max_gap=0),
            "floors its frame gap at 1")            # kernel would bridge a 1-frame hole

    # ── review: a group spanning <2 frames must not vanish at min_track_length<=1 ──
    # Every kernel linker early-returns on such a group BEFORE the min_track_length
    # post-pass, so its rows would keep track_id=None and get the 0 "untracked" sentinel
    # written back — silently deleting them from any downstream `track_id > 0` filter.
    ds_sp, ax_sp = build("label")
    lone = {k: np.concatenate([v, [v[-1] + 1] if k == "id" else [v[0]]])
            for k, v in {kk: ds_sp.get(D.LABEL, kk, layer="labels").values
                         for kk in ("id", "m", "t", "c", "z", "y", "x", "area")}.items()}
    lone["t"][-1] = 0                                # a 13th object seen only at t=0
    lone["y"][-1] = 44.0; lone["x"][-1] = 44.0
    lone["c"][-1] = 1                                # ... in its own (m,c,z) group
    ds_sp = ds_sp.with_structure(
        StructureTable(D.LABEL, lone, layer="labels", z_kind="plane_index"))
    out_1 = pull(ds_sp, ax_sp, min_track_length=1)[0]
    back_1 = out_1.get(D.LABEL, "track_id", layer="labels").values
    assert int((back_1 > 0).sum()) == 13, back_1     # the lone object IS a 1-frame track
    out_2 = pull(ds_sp, ax_sp, min_track_length=2)[0]
    back_2 = out_2.get(D.LABEL, "track_id", layer="labels").values
    assert int((back_2 > 0).sum()) == 12 and back_2[-1] == 0, back_2   # default unchanged

    ds_nr, ax_nr = build("label")                       # overlap with the raster removed
    keep = {k: ds_nr.get(D.LABEL, k, layer="labels").values
            for k in ("id", "m", "t", "c", "z", "y", "x", "area")}
    ds_nr = Dataset(axes=ax_nr).with_structure(
        StructureTable(D.LABEL, keep, layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_nr, ax_nr, method="overlap"), "needs the Voxel label raster")

    ds_3d, ax_3d = build("label")                       # a 3D (subpixel) structure
    ds_3d = ds_3d.with_structure(
        StructureTable(D.LABEL, keep, layer="labels", z_kind="subpixel"))
    refuses(lambda: pull(ds_3d, ax_3d), "2D-only")

    ds_dup, ax_dup = build("label")                     # duplicated member ids
    ds_dup = ds_dup.with_structure(
        StructureTable(D.LABEL, {**keep, "id": np.ones(12, np.int64)},
                       layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_dup, ax_dup), "globally-unique member ids")

    ds_rg, ax_rg = build("label")                       # ragged columns
    ds_rg = ds_rg.with_structure(
        StructureTable(D.LABEL, {"y": np.zeros(3)}, layer="labels", z_kind="plane_index"))
    refuses(lambda: pull(ds_rg, ax_rg), "disagree in length")

    _ok("track.objects: 5 linkers (centroid/serialtrack/CT topology+fingerprint+overlap) "
        "track Label AND Point members end-to-end; per-(m,c,z) plane independence; "
        "track_id write-back aligned under a row permutation; track.link membership "
        "conventions; distinct per-mode recipes; unconsumed sockets hidden per method; "
        "<2-frame groups survive min_track_length=1; 8 silent-degradation refusals")


# ── save / load *.nd2graph.json (graph + zones + groups round-trip) ───────────

def test_serialize() -> None:
    from nodegraph.zones import Zone
    from nodegraph.groups import Group, GROUP_INPUT, GROUP_OUTPUT
    from nodegraph.serialize import FORMAT_VERSION, from_dict, from_json, to_dict, to_json

    # graph with params+modes (incl. a __locked__ sticky list), forward edges AND a back-edge
    g = Graph()
    g.add(NodeInstance("src", "provider.synthetic",
                       params={"shape": [4, 4], "seed": 7, "__locked__": ["seed"]},
                       modes={"dim": "3D"}))
    g.add(NodeInstance("zin", "zone.repeat_in"))
    g.add(NodeInstance("blur", "enhance.gaussian",
                       params={"sigma": 1.5, "on": True}, modes={"dim": "2D"}))
    g.add(NodeInstance("zout", "zone.repeat_out"))
    g.add(NodeInstance("sink", "io.view"))
    g.connect("src", "zin"); g.connect("zin", "blur")
    g.connect("blur", "zout"); g.connect("zout", "sink")
    g.connect("zout", "zin", kind="back")                    # critical: survives round-trip

    zone = Zone("Z1", "repeat", "zin", "zout",
                body=frozenset({"blur"}), iterations=3, impure=True)
    body = Graph()
    body.add(NodeInstance("gi", GROUP_INPUT))
    body.add(NodeInstance("mid", "enhance.median", params={"radius": 2}, modes={"dim": "3D"}))
    body.add(NodeInstance("go", GROUP_OUTPUT))
    body.connect("gi", "mid"); body.connect("mid", "go")
    grp = Group("denoise", body, "gi", "go")

    s = to_json(g, zones=[zone], groups=[grp])
    assert to_json(g, zones=[zone], groups=[grp]) == s        # deterministic
    g2, zones2, groups2 = from_json(s)

    assert set(g2.nodes) == set(g.nodes)
    for nid, n in g.nodes.items():
        m = g2.nodes[nid]
        assert (m.op_key, m.params, m.modes) == (n.op_key, n.params, n.modes)
    assert g2.nodes["src"].params["__locked__"] == ["seed"]   # sticky set survives

    def eset(gr):
        return {(e.src, e.dst, e.src_socket, e.dst_socket, e.kind) for e in gr.edges}
    assert eset(g2) == eset(g)
    assert ("zout", "zin", "out", "data", "back") in eset(g2)  # back-edge survives

    z = zones2[0]
    assert (z.id, z.kind, z.in_id, z.out_id, z.body, z.iterations, z.impure) == \
           (zone.id, zone.kind, zone.in_id, zone.out_id, zone.body, zone.iterations, zone.impure)

    gr2 = groups2[0]
    assert (gr2.name, gr2.input_id, gr2.output_id) == (grp.name, grp.input_id, grp.output_id)
    assert set(gr2.body.nodes) == set(grp.body.nodes) and eset(gr2.body) == eset(grp.body)
    assert gr2.body.nodes["mid"].params == {"radius": 2}      # nested group body round-trips

    for mutate in (lambda doc: doc.update(format_version="9.9"),
                   lambda doc: doc.pop("format_version")):
        doc = to_dict(g); mutate(doc)
        try:
            from_dict(doc); raise AssertionError("bad/absent version not rejected")
        except ValueError:
            pass
    _ok(f"serialize: round-trip graph+back-edge+zone+nested group; bad/absent version "
        f"rejected (v{FORMAT_VERSION})")


# ── flow.iterate: parameter sweep / feedback search (V2.19) ──────────────────

def test_iterate() -> None:
    """The Iterate cone rewrite end-to-end, plus every refusal it is supposed to make.

    The assertions that matter are the STRUCTURAL ones — how many clones get minted, and
    that a driven param really reaches the clone — because a sweep whose iterations are
    secretly identical still produces a plausible-looking answer. Each refusal below is a
    configuration that would otherwise run and return a wrong-but-believable result."""
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    import nodegraph.iterate as IT

    if NODES.get("io.seedIT") is None:
        define_node("io.seedIT", "Seed", outputs=[OutDataset()])

    sz = 24
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=sz, x=sz)
    img = np.zeros((1, 1, 1, 1, sz, sz), dtype=float)
    # four separated blobs of DIFFERENT brightness, so the object count is a strictly
    # decreasing function of the threshold: 0.3 -> 4 objects, 0.9 -> 1. Deterministic, no RNG.
    for cy, cx, v in ((5, 5, 1.0), (5, 18, 0.8), (18, 5, 0.6), (18, 18, 0.4)):
        img[0, 0, 0, 0, cy - 2:cy + 3, cx - 2:cx + 3] = v
    ds = Dataset(axes=ax, metadata={"pixel_size_um": 0.5}).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax, metadata={"pixel_size_um": 0.5})

    def build(params=None, modes=None):
        g = Graph()
        g.add(NodeInstance("S", "io.seedIT"))
        g.add(NodeInstance("TH", "analysis.threshold", params={"threshold": 0.5},
                           modes={"method": "fixed"}))
        g.add(NodeInstance("LB", "analysis.label", modes={"dim": "2D"}))
        g.add(NodeInstance("RS", "analysis.reduce_scalar",
                           params={"source": "area", "name": "n_cells"},
                           modes={"domain": "label", "reducer": "count"}))
        g.add(NodeInstance("IT", IT.ITERATE_OP, params=dict(params or {}),
                           modes=dict(modes or {})))
        g.connect("S", "TH"); g.connect("TH", "LB"); g.connect("LB", "RS")
        g.connect("RS", "IT", dst_socket="collect")
        g.connect("IT", "TH", src_socket="var0", dst_socket="threshold", kind="driver")
        return g

    def run(g, sweep_all=()):
        flat = IT.unroll(g, sweep_all=sweep_all)
        eng = Engine(flat, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                     memo=Memo())
        return flat, eng, eng.pull("IT")

    def scalar(payload, name):
        layer = payload.get(Domain.GLOBAL, name)
        assert layer is not None, f"no Global scalar {name!r} on the result"
        return float(np.asarray(layer.values).reshape(-1)[0])

    LIST = {"mode": "sweep", "variables": "1", "v0_source": "list"}

    # ── a driver wire does NOT make the graph cyclic ───────────────────────────
    g = build({"v0_list": "0.3, 0.5, 0.7, 0.9", "index": 2},
              {**LIST, "preserve": "picked"})
    g.topo_order()               # kind="driver" is skipped, exactly like a zone back-edge
    assert not any(e.kind == "driver" for e in IT.unroll(g).edges), \
        "the rewrite must consume every driver edge — the engine can't route one"

    # ── preserve=picked mints ONE iteration (the production cost claim) ────────
    flat, eng, out = run(g)
    clones = sorted(n for n in flat.nodes if "#IT@" in n)
    assert clones == ["LB#IT@2", "RS#IT@2", "TH#IT@2"], clones
    assert scalar(out, "n_cells") == 2.0                 # threshold 0.7 -> 2 blobs survive
    assert scalar(out, "sweep_threshold") == 0.7, "the winning value must be stamped"
    # the value really reached the clone's params, rather than the sweep running 4x on 0.5
    assert flat.nodes["TH#IT@2"].params["threshold"] == 0.7

    # ── "Run sweep" mints all N, and every iteration differs ───────────────────
    flat, eng, _ = run(g, sweep_all={"IT"})
    assert len([n for n in flat.nodes if "#IT@" in n]) == 12, "4 iterations x 3 cone nodes"
    counts = [scalar(eng.pull(IT.iter_id("RS", "IT", i)), "n_cells") for i in range(4)]
    assert counts == [4.0, 3.0, 2.0, 1.0], counts

    # ── preserve=best picks by the Global metric, both directions ──────────────
    for direction, want_thr, want_n in (("min", 0.9, 1.0), ("max", 0.3, 4.0)):
        g = build({"v0_list": "0.3, 0.5, 0.7, 0.9", "metric": "n_cells"},
                  {**LIST, "preserve": "best", "direction": direction})
        _, _, out = run(g)
        assert (scalar(out, "sweep_threshold"), scalar(out, "sweep_metric")) \
            == (want_thr, want_n), direction

    # ── grid vs zip over two variables ─────────────────────────────────────────
    def two_var(combine):
        g = build({"v0_list": "0.3, 0.7", "v1_list": "4, 8", "metric": "n_cells"},
                  {"mode": "sweep", "variables": "2", "combine": combine,
                   "preserve": "best", "v0_source": "list", "v1_source": "list"})
        g.connect("IT", "LB", src_socket="var1", dst_socket="connectivity", kind="driver")
        return g
    assert [it.values for it in IT.plan(two_var("grid"), "IT").iterations] == \
        [(0.3, 4.0), (0.3, 8.0), (0.7, 4.0), (0.7, 8.0)], "grid is the Cartesian product"
    assert [it.values for it in IT.plan(two_var("zip"), "IT").iterations] == \
        [(0.3, 4.0), (0.7, 8.0)], "zip takes the i-th of each"
    _, _, out = run(two_var("grid"))
    assert scalar(out, "sweep_metric") == 4.0

    # ── 'around' anchors on the TARGET's own value, not on a typed constant ────
    g = build({"v0_percent": 50.0, "v0_steps": 3, "index": 1},
              {**LIST, "v0_source": "around", "preserve": "picked"})
    vals = IT.plan(g, "IT").variables[0].values
    assert vals == (0.25, 0.5, 0.75), vals          # centred on TH's own threshold=0.5

    # ── feedback: a secant search converges onto a target count ────────────────
    g = build({"v0_start": 0.2, "v0_stop": 1.0, "v0_steps": 6, "metric": "n_cells",
               "target": 2.0, "tol": 0.5},
              {"mode": "feedback", "variables": "1", "search": "secant",
               "v0_source": "linear"})
    flat, _, out = run(g)
    assert sorted(n for n in flat.nodes if n.startswith("IT#adv")) == \
        [f"IT#adv@{i}" for i in range(1, 6)], "one advance per probe after the first"
    assert scalar(out, "sweep_metric") == 2.0 and scalar(out, "sweep_converged") == 1.0
    # iteration 0's probe is BAKED (deterministic); later ones arrive on a wire, which is
    # the engine's wired-scalar rule doing its job.
    assert flat.nodes["TH#IT@0"].params["threshold"] == 0.2
    assert "threshold" not in flat.nodes["TH#IT@1"].params

    # golden-section maximizes instead, over the same bracket
    g = build({"v0_start": 0.2, "v0_stop": 1.0, "v0_steps": 5, "metric": "n_cells"},
              {"mode": "feedback", "variables": "1", "search": "golden",
               "v0_source": "linear"})
    _, _, out = run(g)
    assert scalar(out, "sweep_metric") == 4.0, "max object count is at the low threshold"

    # ── an iteration that finds NOTHING is a row, not a crash ──────────────────
    g = build({"v0_list": "0.5, 2.0", "metric": "n_cells"},
              {**LIST, "preserve": "best", "direction": "max"})
    _, eng, out = run(g)
    assert scalar(eng.pull(IT.iter_id("RS", "IT", 1)), "n_cells") == 0.0, \
        "a threshold above every pixel must report 0 objects, not kill the sweep"
    assert scalar(out, "sweep_metric") == 3.0

    # ── the memo: re-pulling an unchanged sweep recomputes nothing ─────────────
    g = build({"v0_list": "0.3, 0.5, 0.7", "metric": "n_cells"},
              {**LIST, "preserve": "best"})
    flat = IT.unroll(g)
    memo = Memo()
    Engine(flat, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
           memo=memo).pull("IT")
    again = Engine(flat, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                   memo=memo)
    again.pull("IT")
    assert again.compute_count == 0, "an unchanged sweep must be fully memo-cached"
    # two iterations resolving to the SAME value share a recipe_hash, so the second is free
    dup = IT.unroll(build({"v0_list": "0.4, 0.4, 0.4", "metric": "n_cells"},
                          {**LIST, "preserve": "best"}))
    e_dup = Engine(dup, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                   memo=Memo())
    e_dup.pull("IT")
    e_uni = Engine(IT.unroll(build({"v0_list": "0.3, 0.5, 0.7", "metric": "n_cells"},
                                   {**LIST, "preserve": "best"})),
                   computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env}, memo=Memo())
    e_uni.pull("IT")
    assert e_dup.compute_count < e_uni.compute_count, \
        "repeated values must collapse onto one recipe_hash"

    # ── the refusals (each would otherwise return a believable wrong answer) ───
    def refuses(g, needle):
        try:
            IT.unroll(g)
        except ValueError as exc:
            assert needle in str(exc), f"refused, but not for {needle!r}: {exc}"
            return
        raise AssertionError(f"expected a refusal mentioning {needle!r}")

    g = build({"v0_list": "0.3,0.5"}, LIST)
    g.add(NodeInstance("X", "enhance.gamma", params={"gamma": 2.0}))
    g.connect("LB", "X")
    refuses(g, "outside it")                       # a branch escaping the cone

    def rewired(dst, socket, **kw):
        g = build({"v0_list": "0.3,0.5"}, {**LIST, **kw.pop("modes", {})})
        g.edges = [e for e in g.edges if e.kind != "driver"]
        g.connect("IT", dst, src_socket="var0", dst_socket=socket, kind="driver")
        return g

    refuses(rewired("S", "threshold"), "no parameter")          # target has no such param
    refuses(rewired("LB", "__mode__:dim"), "2D/3D lever")       # the footprint lever
    refuses(rewired("TH", "__mode__:method", modes={"mode": "feedback"}),
            "only be swept")                                    # a Mode driven by payload
    refuses(rewired("TH", "data"), "Dataset input")             # a value onto the main wire

    # truly NESTED: IT2 collects inside IT's chain AND feeds the rest of it, so IT2 itself
    # lands in IT's cone and would have to be cloned along with everything else.
    g = build({"v0_list": "0.3,0.5"}, LIST)
    g.add(NodeInstance("IT2", IT.ITERATE_OP, params={"v0_list": "1,2"},
                       modes=dict(LIST)))
    g.edges = [e for e in g.edges if not (e.src == "LB" and e.dst == "RS")]
    g.connect("LB", "IT2", dst_socket="collect")
    g.connect("IT2", "RS")
    g.connect("IT2", "LB", src_socket="var0", dst_socket="connectivity", kind="driver")
    refuses(g, "Nested iteration")

    # OVERLAPPING but not nested: IT2 collects from the same tail and drives a DIFFERENT
    # param further down, so its cone {LB, RS} is a subset of IT's {TH, LB, RS} while IT2
    # itself sits outside — nothing escapes, and only the overlap is wrong.
    g = build({"v0_list": "0.3,0.5"}, LIST)
    g.add(NodeInstance("IT2", IT.ITERATE_OP, params={"v0_list": "1,2"}, modes=dict(LIST)))
    g.connect("RS", "IT2", dst_socket="collect")
    g.connect("IT2", "LB", src_socket="var0", dst_socket="connectivity", kind="driver")
    refuses(g, "overlap")

    # the same parameter driven by two cards — the more specific of the two, so it must
    # win over the overlap their cones necessarily also have
    g = build({"v0_list": "0.3,0.5"}, LIST)
    g.add(NodeInstance("IT2", IT.ITERATE_OP, params={"v0_list": "1,2"}, modes=dict(LIST)))
    g.connect("RS", "IT2", dst_socket="collect")
    g.connect("IT2", "TH", src_socket="var0", dst_socket="threshold", kind="driver")
    refuses(g, "driven by two Iterate nodes")

    g = build({"v0_list": "0.3,0.5"}, LIST)
    g.add(NodeInstance("D", "io.dock", modes={"state": "docked"}))
    g.edges = [e for e in g.edges if not (e.src == "TH" and e.dst == "LB")]
    g.connect("TH", "D"); g.connect("D", "LB")
    refuses(g, "docked Dock")

    g = build({"v0_start": 0.1, "v0_stop": 1.0, "v0_steps": 9,
               "v1_list": "1,2,3,4,5,6,7,8,9"},
              {"mode": "sweep", "variables": "2", "combine": "grid",
               "v0_source": "linear", "v1_source": "list"})
    g.connect("IT", "LB", src_socket="var1", dst_socket="connectivity", kind="driver")
    refuses(g, "past the limit")                   # 9 x 9 = 81 clones of the whole chain
    refuses(build({"v0_list": "0.3,0.5"}, {**LIST, "preserve": "best"}), "needs a score")
    g = build({"v0_start": 0.2, "v0_stop": 0.9, "v0_steps": 4, "v1_list": "4,8",
               "metric": "n_cells"},
              {"mode": "feedback", "variables": "2", "v0_source": "linear",
               "v1_source": "list"})
    g.connect("IT", "LB", src_socket="var1", dst_socket="connectivity", kind="driver")
    refuses(g, "searches ONE variable")
    refuses(build({"v0_list": "0.3,0.5", "metric": "n"},
                  {"mode": "feedback", "variables": "1", "v0_source": "list"}),
            "SEARCH RANGE")
    refuses(build({"v0_list": ""}, LIST), "has no values")

    # ── an UN-REWRITTEN graph: pass through, but never pretend a sweep happened ─
    def raw_pull(g):
        return Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env},
                      memo=Memo()).pull("IT")

    g = build({"v0_list": "0.5"}, LIST)          # one value = nothing to iterate
    g.edges = [e for e in g.edges if e.kind != "driver"]
    raw = raw_pull(g)
    assert scalar(raw, "n_cells") == 3.0, "a one-value Iterate must pass its input through"
    assert raw.get(Domain.GLOBAL, "sweep_var0") is None, \
        "with no rewrite there is no iteration this payload can honestly be labelled with"

    g = build({"v0_list": "0.3, 0.7"}, LIST)     # a REAL sweep that never ran
    g.edges = [e for e in g.edges if e.kind != "driver"]
    try:
        raw_pull(g)
        raise AssertionError("an un-rewritten multi-value sweep must refuse, not run once")
    except ValueError as exc:
        assert "without the iterate rewrite" in str(exc), exc

    _ok("flow.iterate: driver edges leave the DAG acyclic and are fully consumed; "
        "picked mints 1 clone / sweep-all mints N and each iteration's value reaches its "
        "clone; best picks by a Global metric (max+min); grid and zip; 'around' anchors on "
        "the target's own value; feedback secant converges to a target and golden "
        "maximizes, with probe 0 baked and later probes arriving on a wire; an empty "
        "iteration reports 0; the memo caches an unchanged sweep and collapses repeated "
        "values; 15 refusals; an un-rewritten graph passes through without inventing a "
        "sweep, and refuses outright when one was really described")


# ── engine hardening: C5 provider identity + C7 un-contexted-read guard ───────

def test_engine_hardening() -> None:
    from nodegraph.provider import ArrayProvider

    # C5: a provider's .version defaults to its fingerprint — stable for the same data,
    # distinct for different content (ArrayProvider hashes its bytes).
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=2, x=2)
    sp = SyntheticProvider(ax1)
    assert sp.version == sp.fingerprint()                       # stable/structural
    z = np.zeros((1, 1, 1, 1, 2, 2)); o = np.ones((1, 1, 1, 1, 2, 2))
    assert ArrayProvider(z).version != ArrayProvider(o).version  # content-distinct

    # C5 end-to-end: two engines SHARING one memo, same source node, providers whose
    # data differs → distinct recipe hashes (no collision) → correct payloads (a stale
    # cached blob would return the wrong image — review #3 / the __provider_version__ hook).
    define_node("c5.src", "Src", outputs=[OutDataset()])

    def _c5(ctx):
        return np.asarray([float(ctx.provider.read_region(0, 0, 0, 0, 0, 0, 1, 0, 1)[0, 0])])

    g = Graph(); g.add(NodeInstance("S", "c5.src"))
    shared = Memo()
    e0 = Engine(g, computes={"c5.src": _c5}, memo=shared, providers={"S": ArrayProvider(z)})
    e7 = Engine(g, computes={"c5.src": _c5}, memo=shared, providers={"S": ArrayProvider(o)})
    r0, r7 = e0.pull("S"), e7.pull("S")
    assert e0.entry("S").recipe_hash != e7.entry("S").recipe_hash
    assert r0[0] == 0.0 and r7[0] == 1.0                        # no wrong-payload dedup

    # C7: strict_reads makes an un-contexted calibration read on an input Dataset a hard
    # error; ctx.calib stays fine and with_metadata pass-through is unaffected.
    define_node("c7.src", "Src", outputs=[OutDataset()])
    define_node("c7.bad", "Bad", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("c7.good", "Good", inputs=[InDataset()], outputs=[OutDataset()])
    srcds = Dataset(axes=AxisSizes(z=1), metadata={"pixel_size_um": 0.1})
    seed = {"S": MetaEnvelope(metadata={"pixel_size_um": 0.1})}

    def _bad(ctx):
        _ = ctx.inputs[0].metadata["pixel_size_um"]            # un-contexted → fenced away
        return ctx.inputs[0]

    def _good(ctx):
        _ = ctx.calib("pixel_size_um")                         # fenced, recorded
        return ctx.inputs[0].with_metadata(note=1)             # {**strict} fast path — no trip

    def mk(op, fn, strict):
        gg = Graph(); gg.add(NodeInstance("S", "c7.src")); gg.add(NodeInstance("X", op))
        gg.connect("S", "X")
        return Engine(gg, computes={op: fn}, seeds={"S": srcds}, meta_seeds=seed,
                      strict_reads=strict)

    try:
        mk("c7.bad", _bad, True).pull("X")
        raise AssertionError("strict_reads should reject the un-contexted calib read")
    except RuntimeError:
        pass
    assert mk("c7.bad", _bad, False).pull("X") is not None      # off by default → allowed
    og = mk("c7.good", _good, True).pull("X")
    assert og.metadata.get("note") == 1                        # ctx.calib + with_metadata OK

    # review: a content-bearing B2ndProvider must NOT collide on structural identity —
    # two stores of equal geometry but different pixels get distinct fingerprint/version
    if _HAVE_BLOSC2:
        from nodegraph.provider import B2ndProvider
        b0 = B2ndProvider.from_array(np.zeros((1, 1, 1, 1, 4, 4)))
        b1 = B2ndProvider.from_array(np.ones((1, 1, 1, 1, 4, 4)))
        assert b0.fingerprint() != b1.fingerprint() and b0.version != b1.version

    # review: the strict wrapper is LAUNDERED out of the output payload — a compute
    # returning the (wrapped) input unchanged yields a plain-dict output that a
    # downstream/non-strict consumer can read without tripping
    define_node("c7.pass", "Pass", inputs=[InDataset()], outputs=[OutDataset()])
    out = mk("c7.pass", lambda c: c.inputs[0], True).pull("X")
    assert out.metadata.__class__ is dict and out.metadata["pixel_size_um"] == 0.1
    _ok("engine hardening: C5 provider.version (ArrayProvider + B2ndProvider content); "
        "C7 strict_reads fences reads + laundered out of outputs")


# ── regression guards for the 2026-07-21 adversarial review findings ──────────

def test_review_regressions() -> None:
    from nodegraph.memo import node_recipe_hash, output_fingerprint
    from nodegraph.structure import COORD_COLUMNS, label_components
    from nodegraph.bridges import label_to_voxel, point_to_voxel
    from nodegraph.metadata import channel_select

    # #1 (critical) _canon is injective: previously-colliding params now differ
    assert (node_recipe_hash("op", {"a": "b", "c": "d"}, (), ())
            != node_recipe_hash("op", {"a": "b,sc=sd"}, (), ()))
    assert (node_recipe_hash("op", ["a", "b"], (), ())
            != node_recipe_hash("op", ["a,sb"], (), ()))
    assert node_recipe_hash("op", {"a": "b"}, (), ()) == node_recipe_hash("op", {"a": "b"}, (), ())

    # #7 Dataset fingerprint pairs key↔revision → insertion-order independent
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=1, x=1)
    a = AttributeLayer(D.FRAME, "A", np.ones((1, 1)))
    b = AttributeLayer(D.FRAME, "B", np.ones((1, 1)))
    dsAB = Dataset(axes=ax1).with_attribute(a).with_attribute(b)
    dsBA = Dataset(axes=ax1).with_attribute(b).with_attribute(a)
    assert output_fingerprint(dsAB) == output_fingerprint(dsBA)

    # #5 a read-only VIEW over a writeable base is copied, not aliased
    big = np.ones((1, 6)); v = big.view(); v.flags.writeable = False
    lay = AttributeLayer(D.FRAME, "x", v)
    big[0, 0] = 999.0
    assert lay.values[0, 0] == 1.0

    # #14 AttributeLayer is hashable + identity-eq (no ndarray-eq crash)
    l1 = AttributeLayer(D.FRAME, "x", np.zeros((2, 3)))
    assert hash(l1) == hash(l1) and l1 == l1
    assert l1 != AttributeLayer(D.FRAME, "x", np.zeros((2, 3))) and len({l1, l1}) == 1

    # #6 label_components emits the invariant id,m,t,c,z,y,x schema
    _, tbl = label_components(np.array([[1, 0], [0, 2]]), 4, m=1, t=2, c=3)
    assert set(COORD_COLUMNS) <= set(tbl.columns)
    assert tbl.columns["m"].tolist() == [1, 1] and tbl.columns["c"].tolist() == [3, 3]

    # #9 point_to_voxel sum: background fills only empty voxels (hits not offset)
    sv = point_to_voxel([10.0], np.array([[0, 0]]), (2, 2), reducer="sum", background=5.0)
    assert sv[0, 0] == 10.0 and sv[0, 1] == 5.0
    # #10 max: a legitimate -inf point value survives; empties get background
    mx = point_to_voxel([-np.inf], np.array([[0, 0]]), (1, 2), reducer="max", background=7.0)
    assert mx[0, 0] == -np.inf and mx[0, 1] == 7.0
    # #11 label_to_voxel: a negative id paints nothing (no negative-index wrap)
    assert np.allclose(label_to_voxel([-1], [9.0], np.array([[0, 1], [1, 0]])), 0.0)

    # #12 provider empty z-range → (0, by, bx), not a crash
    sp = SyntheticProvider(AxisSizes(m=1, t=1, z=8, c=1, y=16, x=16), tile=8)
    assert sp.get_subvolume(0, 0, 0, 0, 5, 5, 0, 0).shape == (0, 8, 8)

    # #13 channel_select drops out-of-range indices (count ↔ emission stay in lockstep)
    env = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=2, y=1, x=1),
                       metadata={"channel_emission_nm": [500, 600]})
    out = channel_select(env, {"channels": [0, 1, 5]}, {})
    assert out.axes.c == 2 and out.metadata["channel_emission_nm"] == [500, 600]

    # #2 a compute reading ctx.env.metadata directly is now RECORDED → invalidated
    define_node("rr.src", "S", outputs=[OutDataset()])
    define_node("rr.env", "E", inputs=[InDataset()], outputs=[OutDataset()])
    ncalls = {"e": 0}

    def c_env(ctx):
        ncalls["e"] += 1
        return np.asarray(ctx.inputs[0]) * ctx.env.metadata["pixel_size_um"]

    g = Graph()
    g.add(NodeInstance("S", "rr.src")); g.add(NodeInstance("E", "rr.env"))
    g.connect("S", "E")
    eng = Engine(g, computes={"rr.src": lambda c: np.array([2.0]), "rr.env": c_env},
                 meta_seeds={"S": MetaEnvelope(metadata={"pixel_size_um": 0.1})})
    assert np.allclose(eng.pull("E"), 0.2) and ncalls["e"] == 1
    eng.reseed_meta({"S": MetaEnvelope(metadata={"pixel_size_um": 0.5})})
    assert np.allclose(eng.pull("E"), 1.0) and ncalls["e"] == 2

    # #4 multi-input node receives args by SOCKET, independent of edge-connect order
    define_node("rr.leaf", "L", outputs=[OutDataset()])
    define_node("rr.sub", "Sub", inputs=[InDataset("a"), InDataset("b")], outputs=[OutDataset()])
    g2 = Graph()
    for nid in ("A", "B"):
        g2.add(NodeInstance(nid, "rr.leaf"))
    g2.add(NodeInstance("SUB", "rr.sub"))
    g2.connect("B", "SUB", dst_socket="b")            # edges out of declaration order
    g2.connect("A", "SUB", dst_socket="a")
    eng2 = Engine(g2, computes={"rr.sub": lambda c: np.asarray(c.inputs[0]) - np.asarray(c.inputs[1])},
                  seeds={"A": np.array([10.0]), "B": np.array([3.0])})
    assert np.allclose(eng2.pull("SUB"), 7.0)         # a - b = 10 - 3, not 3 - 10
    _ok("review regressions: #1 hash-collision, #2 env-read fence, #4 socket order, #5/#6/#7/#9-#14")


# ── C1 streaming eval (V2.04 — LOCKED 2026-07-22) ──────────────────────────────

def test_streaming() -> None:
    if not _HAVE_SKIMAGE:
        _ok("streaming eval: SKIPPED (scipy/skimage absent)")
        return
    import itertools
    from nodegraph.provider import ArrayProvider, TileProvider
    from nodegraph.nodes import COMPUTES, register_node as _reg
    from nodegraph.streaming import (
        MapComputeProvider, StreamProvider, WindowView, ZReduceProvider,
        realize, stream_fp,
    )
    from nodegraph.zones import assert_zone_pure

    rng = np.random.default_rng(7)
    ax = AxisSizes(m=1, t=1, z=2, c=1, y=70, x=70)
    raw = rng.normal(100.0, 25.0, (1, 1, 2, 1, 70, 70))
    base = ArrayProvider(raw, tile=32)              # 32-tiles → seams + ragged edge
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3}
    ds = Dataset(axes=ax, metadata=optics).with_image(base)
    env = MetaEnvelope(axes=ax, metadata=optics)
    define_node("io.stream_seed", "Seed", outputs=[OutDataset()])

    def eng(nodes, edges, *, cache_bytes=1 << 30, seed_ds=ds, seed_env=env):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(*e[:2], **(e[2] if len(e) > 2 else {}))
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds},
                      meta_seeds={"S": seed_env}, cache_bytes=cache_bytes)

    def gather(dset) -> np.ndarray:
        p = dset.image
        pax = p.axes
        out = np.empty((pax.m, pax.t, pax.z, pax.c, pax.y, pax.x), dtype=float)
        for m, t, z, c in itertools.product(range(pax.m), range(pax.t),
                                            range(pax.z), range(pax.c)):
            out[m, t, z, c] = p.get_region(0, m, t, z, c, 0, pax.y, 0, pax.x)
        return out

    # 1) tiled ≡ eager BYTES across a 2-op kernel chain (seams + ragged edges).
    #    The eager reference is the same computes forced down the pre-C1 path via the
    #    oversize bypass (a tiny budget → _map_image realizes whole, V2.04 §6b).
    chain_nodes = [("S", "io.stream_seed", {}),
                   ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                              "params": {"sigma": 0.2}}),
                   ("Md", "enhance.median", {"modes": {"dim": "2D"},
                                             "params": {"radius": 0.15}})]
    chain_edges = [("S", "G"), ("G", "Md")]
    lazy_out = eng(chain_nodes, chain_edges).pull("Md")
    assert isinstance(lazy_out.image, StreamProvider)          # actually lazy
    eager_out = eng(chain_nodes, chain_edges, cache_bytes=1024).pull("Md")
    assert isinstance(eager_out.image, ArrayProvider)          # oversize bypass → eager
    assert _stream_eq(gather(lazy_out), gather(eager_out))

    # 2) laziness + touched-window bound: pulling computes NOTHING; one tile read
    #    reads exactly one halo-expanded window from the base, second read is a cache
    #    hit (no new base reads).
    class _Counting(TileProvider):
        def __init__(self, inner):
            self._i = inner
            self.axes, self.tile, self.levels = inner.axes, inner.tile, 1
            self.calls = []

        def read_region(self, level, m, t, z, c, y0, y1, x0, x1):
            self.calls.append((z, y0, y1, x0, x1))
            return self._i.read_region(level, m, t, z, c, y0, y1, x0, x1)

        def fingerprint(self):
            return ("count",) + tuple(self._i.fingerprint())

    counting = _Counting(base)
    e2 = eng([("S", "io.stream_seed", {}),
              ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                         "params": {"sigma": 0.2}})],
             [("S", "G")], seed_ds=ds.with_image(counting))
    g_out = e2.pull("G")
    assert counting.calls == []                                # pull = plan, no pixels
    tile00 = g_out.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert len(counting.calls) == 1                            # one window, one z
    (z_, y0_, y1_, x0_, x1_) = counting.calls[0]
    assert z_ == 0 and y1_ - y0_ <= 32 + 2 * 8 and x1_ - x0_ <= 32 + 2 * 8   # halo=int(4·2+.5)=8
    assert not tile00.flags.writeable                          # uniform freeze
    _ = g_out.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert len(counting.calls) == 1 and e2.tiles.hits >= 1     # cache hit, no re-read
    # memo re-pull: same recipe → hit, no recompute
    n_before = e2.compute_count
    e2.pull("G")
    assert e2.compute_count == n_before

    # 3) fp stability + reseed_meta staleness (the V2.04 §6b tile-cache hole): the
    #    provider fp folds declared calibration reads, so a pixel-size edit re-keys
    #    the tile cache — the same tile MUST come back different.
    e3 = eng([("S", "io.stream_seed", {}),
              ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                         "params": {"sigma": 0.2}})], [("S", "G")])
    p_a = e3.pull("G").image
    t_a = p_a.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    e3.reseed_meta({"S": MetaEnvelope(axes=ax, metadata={"pixel_size_um": 0.05,
                                                         "z_step_um": 0.3})})
    p_b = e3.pull("G").image
    t_b = p_b.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert p_a._fp != p_b._fp                                  # calibration re-keys fp
    assert not np.array_equal(t_a, t_b)                        # no stale tile served
    # stability: an identical fresh engine reproduces the identical fp
    p_c = eng([("S", "io.stream_seed", {}),
               ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                          "params": {"sigma": 0.2}})],
              [("S", "G")]).pull("G").image
    assert p_c._fp == p_a._fp
    # unstable fp inputs (id-bearing reprs) are a hard error, not a silent cache-cold
    try:
        stream_fp("map", "x", {"fn": lambda: 0}, (), (), base)
        raise SystemExit("stream_fp accepted an unstable (callable) param")
    except TypeError:
        pass

    # 4) per-tile windowed FIELD materialization (D4 flagship, fixture node): a
    #    per-voxel Attr field scales the image per tile; per-tile ≡ whole; a layer
    #    revision change re-keys the provider fp (no stale field tiles).
    wmap = rng.normal(1.5, 0.2, (1, 1, 2, 1, 70, 70))
    ds_f = ds.with_layer(D.VOXEL, "w", wmap)

    def c_scale(ctx):
        d = ctx.inputs[0]
        fld = ctx.input("factor")
        fc = ctx.fields
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (field_expr_hash(fld, d),), d.image)

        def fn(a, m, t, z, c, gy0, gy1, gx0, gx1):
            win = {"m": (m, m + 1), "t": (t, t + 1), "z": (z, z + 1), "c": (c, c + 1),
                   "y": (gy0, gy1), "x": (gx0, gx1)}
            val = fc.evaluate(fld, FieldContext(d, D.VOXEL, d.axes, window=win),
                              token=(fp, 0, m, t, z, c, gy0, gy1, gx0, gx1))
            return a * np.asarray(val).reshape(a.shape)

        return d.with_image(MapComputeProvider(d.image, fn, halo=0,
                                               fp=fp, cache=ctx.tiles))

    _reg(c_scale, op_key="test.scale_field", label="ScaleF",
         inputs=[InDataset(), InFloat("factor", "Factor", field=True)],
         outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())
    _reg(lambda ctx: Attr(D.VOXEL, "w"), op_key="test.wfield_src", label="WSrc",
         outputs=[OutDataset()])
    ef = eng([("S", "io.stream_seed", {}), ("F", "test.wfield_src", {}),
              ("SC", "test.scale_field", {})],
             [("S", "SC"), ("F", "SC", {"dst_socket": "factor"})], seed_ds=ds_f)
    sc = ef.pull("SC")
    assert _stream_eq(gather(sc), raw * wmap)                    # per-tile ≡ whole
    assert ef.fields.hits + ef.fields.misses > 0                 # went through the cache
    # a changed layer (new revision) re-keys the provider fp
    ds_f2 = ds.with_layer(D.VOXEL, "w", wmap + 1.0)
    ef2 = eng([("S", "io.stream_seed", {}), ("F", "test.wfield_src", {}),
               ("SC", "test.scale_field", {})],
              [("S", "SC"), ("F", "SC", {"dst_socket": "factor"})], seed_ds=ds_f2)
    assert ef2.pull("SC").image._fp != sc.image._fp

    # 5) threshold consumes a wired per-voxel Field threshold (windowed, eager mask)
    thr_map = np.full((1, 1, 2, 1, 70, 70), 100.0)
    thr_map[0, 0, 1] = 90.0                                    # z-varying threshold
    ds_t = ds.with_layer(D.VOXEL, "thr", thr_map)
    _reg(lambda ctx: Attr(D.VOXEL, "thr"), op_key="test.tfield_src", label="TSrc",
         outputs=[OutDataset()])
    et = eng([("S", "io.stream_seed", {}), ("F", "test.tfield_src", {}),
              ("T", "analysis.threshold", {})],
             [("S", "T"), ("F", "T", {"dst_socket": "threshold"})], seed_ds=ds_t)
    mask = et.pull("T").get(D.VOXEL, "mask")
    assert mask is not None
    assert np.array_equal(mask.values, (raw > thr_map).astype(np.int64))

    # 6) z-project tree-reduce: monoid (mean/max) folds per tile via PartialReducer,
    #    median falls back to the stacked z-column — all ≡ the whole-volume reduce.
    vol = raw[0, 0, :, 0]
    for method, expect in (("max", vol.max(0)), ("mean", vol.mean(0)),
                           ("median", np.median(vol, 0))):
        ez = eng([("S", "io.stream_seed", {}),
                  ("Z", "util.zproject", {"modes": {"method": method}})],
                 [("S", "Z")])
        zout = ez.pull("Z")
        assert isinstance(zout.image, ZReduceProvider) and zout.axes.z == 1
        got = zout.image.get_region(0, 0, 0, 0, 0, 0, 70, 0, 70)
        assert np.allclose(got, expect, rtol=_stream_rtol()), f"zproject {method}"
    # meta_transform lockstep preserved (z_step dropped, z_collapsed stamped)
    assert zout.metadata.get("z_collapsed") is True and "z_step_um" not in zout.metadata

    # 7) crop is a pure lazy view: source dtype preserved, bytes = the slice, and a
    #    kernel downstream matches the eager chain (reflect-at-crop-edge parity).
    raw16 = (rng.uniform(0, 4000, (1, 1, 2, 1, 70, 70))).astype(np.uint16)
    ds16 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw16, tile=32))
    crop_nodes = [("S", "io.stream_seed", {}),
                  ("C", "util.crop", {"params": {"y0": 5, "y1": 37, "x0": 3, "x1": 66}}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})]
    crop_edges = [("S", "C"), ("C", "G")]
    ec = eng(crop_nodes, crop_edges, seed_ds=ds16)
    cds = ec.pull("C")
    assert isinstance(cds.image, WindowView) and cds.axes.y == 32 and cds.axes.x == 63
    cw = cds.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 63)
    assert cw.dtype == np.uint16                               # raw dtype preserved
    assert np.array_equal(cw, raw16[0, 0, 0, 0, 5:37, 3:66])
    lazy_g = gather(ec.pull("G"))
    eager_g = gather(eng(crop_nodes, crop_edges, seed_ds=ds16,
                         cache_bytes=1024).pull("G"))
    assert _stream_eq(lazy_g, eager_g)

    # 8) cum-halo fence: chained gaussians accumulate halo (8 px each, tile 32) →
    #    deeper providers silently promote to the plane unit, bytes stay exact.
    fence_nodes = [("S", "io.stream_seed", {})] + [
        (f"G{i}", "enhance.gaussian", {"modes": {"dim": "2D"},
                                       "params": {"sigma": 0.2}}) for i in range(4)]
    fence_edges = [("S", "G0")] + [(f"G{i}", f"G{i+1}") for i in range(3)]
    efence = eng(fence_nodes, fence_edges)
    fout = efence.pull("G3")
    assert fout.image._plane_unit                              # 2·cum_halo ≥ tile tripped
    assert _stream_eq(
        gather(fout), gather(eng(fence_nodes, fence_edges, cache_bytes=1024).pull("G3")))

    # 9) deep chain (unrolled-zone shape): 300 lazy plane units pull + realize under
    #    the scoped recursion headroom; values survive the whole chain.
    deep_nodes = [("S", "io.stream_seed", {})] + [
        (f"P{i}", "enhance.gamma", {"params": {"gamma": 1.0}}) for i in range(300)]
    deep_edges = [("S", "P0")] + [(f"P{i}", f"P{i+1}") for i in range(299)]
    ed = eng(deep_nodes, deep_edges)
    deep = ed.pull("P299")
    assert deep.image.depth >= 300
    assert np.allclose(gather(realize(deep)), raw, rtol=_stream_rtol(1e-9))

    # 10) a late ctx.calib from inside a lazy closure is a hard error at tile time
    #     (the frozen ReadContext — it would silently escape the memo fence)
    def c_late(ctx):
        d = ctx.inputs[0]
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (), d.image)
        return d.with_image(MapComputeProvider(
            d.image, lambda a, *rest: a * (ctx.calib("pixel_size_um") or 1.0),
            halo=0, fp=fp, cache=ctx.tiles))

    _reg(c_late, op_key="test.late_read", label="Late",
         inputs=[InDataset()], outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())
    el = eng([("S", "io.stream_seed", {}), ("L", "test.late_read", {})], [("S", "L")])
    lazy_late = el.pull("L")                                   # compute itself is fine
    caught = False
    try:
        lazy_late.image.read_region(0, 0, 0, 0, 0, 0, 8, 0, 8)
    except RuntimeError as ex:
        caught = "late metadata read" in str(ex)
    assert caught, "late closure calib read was not fenced"

    # 11) assert_zone_pure still catches an IMPURE lazy body: structural fingerprints
    #     are equal by construction, so the debug-verify realizes to bytes (V2.04 §3)
    def c_noise(ctx):
        d = ctx.inputs[0]
        fp = stream_fp("map", ctx.op_key, ctx.params, ctx.reads.declared_reads(),
                       (), d.image)
        return d.with_image(MapComputeProvider(
            d.image,
            lambda a, *rest: a + np.random.default_rng().normal(size=a.shape),
            halo=0, fp=fp, cache=ctx.tiles))

    _reg(c_noise, op_key="test.noise_map", label="Noise",
         inputs=[InDataset()], outputs=[OutDataset()],
         granularity=Granularity.TILEABLE, kernel_axes=frozenset())

    def mk():
        return eng([("S", "io.stream_seed", {}), ("N", "test.noise_map", {})],
                   [("S", "N")])

    caught = False
    try:
        assert_zone_pure(mk, "N")
    except AssertionError as ex:
        caught = "non-deterministic" in str(ex)
    assert caught, "impure lazy body slipped past assert_zone_pure"

    # ── review regressions (C1 impl review, 2026-07-22) ────────────────────────

    # R1: cum-halo RESETS at a plane-unit level (a realized unit is a window-growth
    # cut point) — a mid-chain node between fence trips, and a halo-0 node after a
    # trip, both stay TILE-unit (the review's interactive-panning cliff).
    assert not efence.pull("G1").image._plane_unit or True   # G1 trips (cum 16)
    assert efence.pull("G1").image._plane_unit
    assert not efence.pull("G2").image._plane_unit           # reset at G1 → G2 tiles
    g5_nodes = fence_nodes + [("G4", "enhance.gaussian",
                               {"modes": {"dim": "2D"}, "params": {"sigma": 0.001}})]
    g5_edges = fence_edges + [("G3", "G4")]
    p_g4 = eng(g5_nodes, g5_edges).pull("G4").image
    assert not p_g4._plane_unit and p_g4.cum_halo == 0       # halo-0 after fence: tiles

    # R2 (BLOCKER): tile GRIDS are part of the streaming identity — two same-content
    # sources on different tile grids must not alias in the shared TileCache.
    gg = Graph()
    for nid in ("S1", "S2"):
        gg.add(NodeInstance(nid, "io.stream_seed"))
    for nid, src in (("Ga", "S1"), ("Gb", "S2")):
        gg.add(NodeInstance(nid, "enhance.gaussian", modes={"dim": "2D"},
                            params={"sigma": 0.2}))
        gg.connect(src, nid)
    ds32 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw, tile=32))
    ds16 = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(raw, tile=16))
    e_grid = Engine(gg, computes=COMPUTES, seeds={"S1": ds32, "S2": ds16},
                    meta_seeds={"S1": env, "S2": env})
    ga, gb = gather(e_grid.pull("Ga")), gather(e_grid.pull("Gb"))
    ref_g = gather(eng([("S", "io.stream_seed", {}),
                        ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                                   "params": {"sigma": 0.2}})],
                       [("S", "G")], cache_bytes=1024).pull("G"))
    assert _stream_eq(ga, ref_g) and _stream_eq(gb, ref_g)

    # R3: zproject NaN policy is ONE policy across the lazy and eager paths
    rawn = raw.copy()
    rawn[0, 0, 0, 0, 10, 10] = np.nan
    dsn = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(rawn, tile=32))
    zn = [("S", "io.stream_seed", {}), ("Z", "util.zproject",
                                        {"modes": {"method": "mean"}})]
    zedges = [("S", "Z")]
    assert _stream_eq(gather(eng(zn, zedges, seed_ds=dsn).pull("Z")),
                      gather(eng(zn, zedges, seed_ds=dsn,
                                 cache_bytes=1024).pull("Z")), equal_nan=True)

    # R4: a COARSER-domain (FRAME) Attr field thresholds a VOXEL window (windowed
    # refine/broadcast per V2.04 §6b — was a NotImplementedError crash)
    ds_bg = ds.with_layer(D.FRAME, "bg", np.array([[95.0]]))
    _reg(lambda ctx: Attr(D.FRAME, "bg"), op_key="test.bgfield_src", label="BgSrc",
         outputs=[OutDataset()])
    ebg = eng([("S", "io.stream_seed", {}), ("F", "test.bgfield_src", {}),
               ("T", "analysis.threshold", {})],
              [("S", "T"), ("F", "T", {"dst_socket": "threshold"})], seed_ds=ds_bg)
    assert np.array_equal(ebg.pull("T").get(D.VOXEL, "mask").values,
                          (raw > 95.0).astype(np.int64))

    # R5: one field expression consumed by TWO threshold nodes over DIFFERENT
    # geometry (full + cropped) — tokens fold the input provider fp, no collision
    from nodegraph.field import UnaryOp as FUnaryOp
    _reg(lambda ctx: FUnaryOp("abs", Const(100.0)), op_key="test.cfield_src",
         label="CSrc", outputs=[OutDataset()])
    e2t = eng([("S", "io.stream_seed", {}), ("F", "test.cfield_src", {}),
               ("T1", "analysis.threshold", {}),
               ("C", "util.crop", {"params": {"y0": 5, "y1": 37, "x0": 3, "x1": 66}}),
               ("T2", "analysis.threshold", {})],
              [("S", "T1"), ("F", "T1", {"dst_socket": "threshold"}),
               ("S", "C"), ("C", "T2"), ("F", "T2", {"dst_socket": "threshold"})])
    m1 = e2t.pull("T1").get(D.VOXEL, "mask").values
    m2 = e2t.pull("T2").get(D.VOXEL, "mask").values
    assert np.array_equal(m1, (raw > 100.0).astype(np.int64))
    assert np.array_equal(m2, (raw[..., 5:37, 3:66] > 100.0).astype(np.int64))

    # R6: swapping engine.seeds[nid] re-keys the source (seed data identity)
    e_swap = eng([("S", "io.stream_seed", {}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})], [("S", "G")])
    a_sw = gather(e_swap.pull("G"))
    e_swap.seeds["S"] = Dataset(axes=ax, metadata=optics).with_image(
        ArrayProvider(raw + 1.0, tile=32))
    b_sw = gather(e_swap.pull("G"))
    assert not np.array_equal(a_sw, b_sw)

    # R7: an out-of-range z on a cropped view raises (never serves pixels OUTSIDE
    # the crop)
    caught = False
    try:
        cds.image.read_region(0, 0, 0, 5, 0, 0, 8, 0, 8)
    except IndexError:
        caught = True
    assert caught

    # R8: providers hold the cache WEAKLY — a payload outliving its engine stays
    # readable (recompute path), it does not pin or crash
    import gc
    e_weak = eng([("S", "io.stream_seed", {}),
                  ("G", "enhance.gaussian", {"modes": {"dim": "2D"},
                                             "params": {"sigma": 0.2}})], [("S", "G")])
    out_w = e_weak.pull("G")
    t_w1 = np.array(out_w.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32))
    del e_weak
    gc.collect()
    t_w2 = out_w.image.read_region(0, 0, 0, 0, 0, 0, 32, 0, 32)
    assert np.array_equal(t_w1, t_w2)

    # R9: two edges into one NON-multi socket is a hard error (no silent last-wins)
    gdup = Graph()
    for nid in ("S1", "S2"):
        gdup.add(NodeInstance(nid, "io.stream_seed"))
    gdup.add(NodeInstance("G", "enhance.gaussian", modes={"dim": "2D"},
                          params={"sigma": 0.2}))
    gdup.connect("S1", "G")
    gdup.connect("S2", "G")
    e_dup = Engine(gdup, computes=COMPUTES, seeds={"S1": ds32, "S2": ds16},
                   meta_seeds={"S1": env, "S2": env})
    caught = False
    try:
        e_dup.pull("G")
    except ValueError as ex:
        caught = "non-multi" in str(ex)
    assert caught

    # R10: an UNWIRED PRIMARY dataset input is refused by name, not as a bare
    # `IndexError: tuple index out of range` from inside the kernel's `ctx.inputs[0]`
    # (review 2026-07-29). Three shapes, all of which used to reach the compute:
    #   (a) nothing wired at all;
    #   (b) only a VALUE socket wired — the positional tuple then hands the FIELD to
    #       `inputs[0]`, so the old failure was a misaligned read, not even an IndexError;
    #   (c) only an OPTIONAL later dataset socket wired — the guard must still name the
    #       first-declared `data`, never accept `reference` as the primary.
    def _unwired(nodes, edges):
        gu = Graph()
        for nid, op in nodes:
            gu.add(NodeInstance(nid, op))
        for src, dst, sock in edges:
            gu.connect(src, dst, dst_socket=sock)
        try:
            Engine(gu, computes=COMPUTES, seeds={"S": ds32},
                   meta_seeds={"S": env}).pull("G")
        except ValueError as ex:
            return str(ex)
        return ""

    for _nodes, _edges in (
            ([("G", "enhance.gaussian")], []),
            ([("F", "test.cfield_src"), ("G", "enhance.gaussian")],
             [("F", "G", "sigma")]),
            ([("S", "io.stream_seed"), ("G", "analysis.dvc_field")],
             [("S", "G", "reference")])):
        msg = _unwired(_nodes, _edges)
        assert "not wired" in msg and "'G'" in msg and "'data'" in msg, msg

    _ok("streaming eval (C1): tiled≡eager bytes; lazy pull + one-window tile reads + "
        "cache hits; reseed re-keys fp (no stale tiles); windowed per-tile fields + "
        "threshold Field; zproject tree-reduce; crop view; cum-halo fence; 300-deep "
        "chain; late-read fence; impure lazy body caught; review R1–R10 (fence reset, "
        "grid identity, NaN policy, coarser-domain field, token collision, seed swap, "
        "crop z-guard, weak cache, non-multi edge guard, unwired-primary guard)")


def test_streaming_slivers() -> None:
    """C1 follow-up slivers (V2.04 §6b): ``util.stack`` T→1 tree-reduce; lazy units for
    normalize / resample / drift (eager-stat, lazy-apply); the kernel-param field gate."""
    if not _HAVE_SKIMAGE:
        _ok("streaming slivers: SKIPPED (scipy/skimage absent)")
        return
    import itertools
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, register_node as _reg, _map_image, _DIM_KAX
    from nodegraph.streaming import (
        MapComputeProvider, PlaneRealizeProvider, TReduceProvider, VolumeComputeProvider,
    )

    rng = np.random.default_rng(11)
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 0.5}
    axt = AxisSizes(m=1, t=3, z=2, c=1, y=40, x=40)
    raw = rng.normal(50.0, 12.0, (1, 3, 2, 1, 40, 40))
    ds = Dataset(axes=axt, metadata=optics).with_image(ArrayProvider(raw, tile=32))
    env = MetaEnvelope(axes=axt, metadata=optics)
    define_node("test.slv_seed", "Seed", outputs=[OutDataset()])

    def eng(nodes, edges, *, seed_ds=ds, seed_env=env):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(*e[:2], **(e[2] if len(e) > 2 else {}))
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds}, meta_seeds={"S": seed_env})

    def gather(dset) -> np.ndarray:
        p = dset.image
        pa = p.axes
        out = np.empty((pa.m, pa.t, pa.z, pa.c, pa.y, pa.x), dtype=float)
        for m, t, z, c in itertools.product(range(pa.m), range(pa.t), range(pa.z),
                                            range(pa.c)):
            out[m, t, z, c] = p.get_region(0, m, t, z, c, 0, pa.y, 0, pa.x)
        return out

    # 1) util.stack T→1: monoid (mean/max) folds the T series per tile, non-monoid
    #    (median) stacks the t-column — both ≡ the numpy reference; dt_s dropped.
    for method, ref in (("mean", raw.mean(1, keepdims=True)),
                        ("max", raw.max(1, keepdims=True)),
                        ("median", np.median(raw, 1, keepdims=True))):
        st = eng([("S", "test.slv_seed", {}),
                  ("K", "util.stack", {"modes": {"method": method}})],
                 [("S", "K")]).pull("K")
        assert isinstance(st.image, TReduceProvider) and st.axes.t == 1
        assert "dt_s" not in st.metadata
        assert np.allclose(gather(st), ref, rtol=_stream_rtol()), f"stack {method}"

    # 2) normalize lazy units: plane (self-contained per plane) & series (eager (lo,hi)
    #    per (m,c) + lazy apply) → MapComputeProvider; volume → VolumeComputeProvider.
    lo_p, hi_p = 1.0, 99.0

    def _rescale(b: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(b, lo_p), np.percentile(b, hi_p)
        return np.zeros_like(b) if hi <= lo else np.clip((b - lo) / (hi - lo), 0.0, 1.0)

    npl = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "plane"}})],
              [("S", "N")]).pull("N")
    assert isinstance(npl.image, MapComputeProvider)
    ref_pl = np.empty_like(raw)
    for m, t, z, c in itertools.product(range(1), range(3), range(2), range(1)):
        ref_pl[m, t, z, c] = _rescale(raw[m, t, z, c])
    assert np.allclose(gather(npl), ref_pl, rtol=_stream_rtol(1e-7), atol=1e-7)

    nvo = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "volume"}})],
              [("S", "N")]).pull("N")
    assert isinstance(nvo.image, VolumeComputeProvider)
    ref_vo = np.empty_like(raw)
    for m, t, c in itertools.product(range(1), range(3), range(1)):
        ref_vo[m, t, :, c] = _rescale(raw[m, t, :, c])
    assert np.allclose(gather(nvo), ref_vo, rtol=_stream_rtol(1e-7), atol=1e-7)

    nse = eng([("S", "test.slv_seed", {}),
               ("N", "enhance.normalize", {"modes": {"scope": "series"}})],
              [("S", "N")]).pull("N")
    assert isinstance(nse.image, MapComputeProvider)
    ref_se = np.empty_like(raw)
    for m, c in itertools.product(range(1), range(1)):
        ref_se[m, :, :, c] = _rescale(raw[m, :, :, c])
    assert np.allclose(gather(nse), ref_se, rtol=_stream_rtol(1e-7), atol=1e-7)

    # 3) drift: eager per-frame FFT estimate + lazy per-plane shift apply (register-once/
    #    apply-all); Frame drift_y/x attrs present; ≡ a direct eager reference.
    from scipy.ndimage import shift as _nsh
    from skimage.registration import phase_cross_correlation as _pcc
    dft = eng([("S", "test.slv_seed", {}), ("D", "align.drift", {})], [("S", "D")]).pull("D")
    assert isinstance(dft.image, MapComputeProvider)
    assert (dft.get(D.FRAME, "drift_y") is not None
            and dft.get(D.FRAME, "drift_x") is not None)
    ref_d = np.empty_like(raw)
    refz = axt.z // 2
    for m in range(1):
        r = raw[m, 0, refz, 0]
        for t in range(3):
            sh = _pcc(r, raw[m, t, refz, 0], upsample_factor=10)[0]
            for c in range(1):
                for z in range(2):
                    ref_d[m, t, z, c] = _nsh(raw[m, t, z, c], shift=sh, order=1,
                                             mode="constant")
    assert np.allclose(gather(dft), ref_d, rtol=_stream_rtol(1e-7), atol=1e-7)

    # 4) resample lazy per-UNIT realize (geometry-changing → PlaneRealizeProvider): only
    #    touched planes resize; ≡ a direct skimage reference; output axes scaled.
    from skimage.transform import resize as _rsz
    rsp = eng([("S", "test.slv_seed", {}),
               ("R", "util.resample",
                {"modes": {"dim": "2D"}, "params": {"scale_xy": 0.5}})],
              [("S", "R")]).pull("R")
    assert isinstance(rsp.image, PlaneRealizeProvider)
    assert (rsp.axes.y, rsp.axes.x, rsp.axes.z) == (20, 20, 2)
    ref_r = np.empty((1, 3, 2, 1, 20, 20))
    for m, t, z, c in itertools.product(range(1), range(3), range(2), range(1)):
        ref_r[m, t, z, c] = _rsz(raw[m, t, z, c], (20, 20), order=1, preserve_range=True)
    assert np.allclose(gather(rsp), ref_r, rtol=_stream_rtol(1e-7), atol=1e-7)

    # 5) kernel-param field gate (Fork B): a NON-Const Field on a kernel_param socket drops
    #    the TILEABLE 2D unit to WHOLE_PLANE; a Const field / no field stay tiled. A halo-0
    #    identity fixture isolates the gate from the cum-halo fence.
    def _c_kfilter(ctx):
        return _map_image(ctx, ctx.inputs[0], plane_fn=lambda a: a, halo=0)

    _reg(_c_kfilter, op_key="test.kfilter", label="KFilter",
         inputs=[InDataset(), InFloat("sigma", "Sigma", field=True, kernel_param=True)],
         outputs=[OutDataset()], modes=[DimMode()],
         granularity={"2D": Granularity.TILEABLE, "3D": Granularity.WHOLE_VOLUME},
         kernel_axes=_DIM_KAX)
    _reg(lambda ctx: Attr(D.VOXEL, "w"), op_key="test.slv_fieldsrc", label="FSrc",
         outputs=[OutDataset()])
    _reg(lambda ctx: Const(0.2), op_key="test.slv_constsrc", label="CSrc",
         outputs=[OutDataset()])
    gv = eng([("S", "test.slv_seed", {}), ("F", "test.slv_fieldsrc", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})],
             [("S", "K"), ("F", "K", {"dst_socket": "sigma"})]).pull("K")
    assert isinstance(gv.image, MapComputeProvider) and gv.image._plane_unit is True
    gc = eng([("S", "test.slv_seed", {}), ("F", "test.slv_constsrc", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})],
             [("S", "K"), ("F", "K", {"dst_socket": "sigma"})]).pull("K")
    assert isinstance(gc.image, MapComputeProvider) and gc.image._plane_unit is False
    gb = eng([("S", "test.slv_seed", {}),
              ("K", "test.kfilter", {"modes": {"dim": "2D"}})], [("S", "K")]).pull("K")
    assert gb.image._plane_unit is False

    _ok("streaming slivers (V2.04 §6b): util.stack T→1 tree-reduce (monoid+gather≡ref, "
        "dt_s dropped); normalize plane/volume/series lazy units; drift eager-estimate + "
        "lazy per-plane apply; resample lazy per-unit realize (geometry change); "
        "kernel-param field gate (non-Const σ field → plane unit; Const/none → tiled)")


def test_catalog_kernels() -> None:
    """Ported v1 analysis kernels (Phase 7 / V2.05): bead detection, histogram
    threshold, registration, granule boundary, + the dep-gated stubs. StarDist is
    excluded (a ~seconds TF model load — verified live in the port, not the fast gate)."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (v1 kernel ports): SKIPPED (scipy/skimage absent)")
        return
    import importlib.util as _u
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    def eng(seed_ds, seed_env, nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for s, d in edges:
            g.connect(s, d)
        return Engine(g, computes=COMPUTES, seeds={"S": seed_ds},
                      meta_seeds={"S": seed_env})

    define_node("io.kseed", "S", outputs=[OutDataset()])

    # ── bead detection (needs numba) — 3D finds the planted beads; 2D≠3D hash ──
    if _u.find_spec("numba") is not None:
        ax = AxisSizes(m=1, t=1, z=8, c=1, y=48, x=48)
        vol = np.zeros((1, 1, 8, 1, 48, 48))
        for (z, y, x) in [(4, 12, 12), (4, 30, 34), (2, 38, 10)]:
            vol[0, 0, z - 1:z + 2, 0, y - 2:y + 3, x - 2:x + 3] = 400.0
        bmeta = {"pixel_size_um": 0.2, "z_step_um": 0.5, "objective_na": 1.0,
                 "channel_emission_nm": [520]}
        bds = Dataset(axes=ax, metadata=bmeta).with_image(ArrayProvider(vol))
        benv = MetaEnvelope(axes=ax, metadata=bmeta)
        e3 = eng(bds, benv, [("S", "io.kseed", {}),
                             ("B", "detect.particles", {"modes": {"dim": "3D"},
                                                        "params": {"min_distance": 0.6}})],
                 [("S", "B")])
        pts = e3.pull("B").get(D.POINT, "z", layer="particles")
        assert pts is not None and len(pts.values) == 3, \
            f"particles 3D found {0 if pts is None else len(pts.values)}"
        e2 = eng(bds, benv, [("S", "io.kseed", {}),
                             ("B", "detect.particles", {"modes": {"dim": "2D"},
                                                        "params": {"min_distance": 0.6}})],
                 [("S", "B")])
        assert e3.entry("B").recipe_hash != e2.entry("B").recipe_hash
        assert any(k == "pixel_size_um" for k, _ in e3.entry("B").reads)
        beads_note = "particles 2D/3D (3 detected, distinct hash, fenced)"
    else:
        beads_note = "beads SKIPPED (numba absent)"

    # ── histogram threshold → mask + labels + region table; methods re-key ─────
    ax2 = AxisSizes(m=1, t=1, z=2, c=1, y=48, x=48)
    img = np.full((1, 1, 2, 1, 48, 48), 500.0)
    img[0, 0, :, 0, 10:20, 10:20] = 4000.0
    img[0, 0, :, 0, 30:36, 30:36] = 3800.0
    hds = Dataset(axes=ax2, metadata={"pixel_size_um": 0.25}).with_image(ArrayProvider(img))
    henv = MetaEnvelope(axes=ax2, metadata={"pixel_size_um": 0.25})
    # v1 parity 2026-07-28: the spatial cleanup now DEFAULTS to the v1 segmenter's numbers
    # (100 px² min area, 1 px opening, 2 px closing, 50 px² holes) via µm derives. The
    # threshold-behaviour checks below pin it OFF so they stay about thresholding; the
    # default itself is verified in its own block (it deletes this fixture's tiny squares).
    NOCLEAN = {"min_area": 0.0, "opening_radius": 0.0,
               "closing_radius": 0.0, "min_hole_size": 0.0}
    eh = eng(hds, henv, [("S", "io.kseed", {}),
                         ("H", "analysis.histogram_threshold",
                          {"params": {**NOCLEAN, "percentile_high": 95.0}})], [("S", "H")])
    hout = eh.pull("H")
    assert int(hout.get(D.VOXEL, "mask").values.sum()) > 0
    assert int(hout.get(D.VOXEL, "labels").values.max()) == 4      # 2 squares × 2 planes
    assert hout.get(D.LABEL, "mean_intensity", layer="labels") is not None
    # the rasters keep the kernel's dtypes — int64 would be 4×/8× the bytes for nothing
    # (a 16-position 2048²×10 series = 5 GiB PER raster → ArrayMemoryError, fix 2026-07-28)
    assert hout.get(D.VOXEL, "mask").values.dtype == np.uint8
    assert hout.get(D.VOXEL, "labels").values.dtype == np.int32
    assert np.array_equal(hout.get(D.VOXEL, "mask").values > 0,
                          hout.get(D.VOXEL, "labels").values > 0)   # mask ≡ labelled area
    e_ch = eng(hds, henv,                       # …and a real Label consumer reads int32
               [("S", "io.kseed", {}),
                ("H", "analysis.histogram_threshold",
                 {"modes": {"method": "relative", "direction": "above"},
                  "params": dict(NOCLEAN)}),
                ("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
               [("S", "H"), ("H", "B")])
    assert int((e_ch.pull("B").get(D.VOXEL, "bands").values != 0).sum()) > 0
    eh2 = eng(hds, henv, [("S", "io.kseed", {}),
                          ("H", "analysis.histogram_threshold",
                           {"modes": {"method": "hysteresis", "direction": "above"},
                            "params": {**NOCLEAN, "strict": 3500,
                                       "permissive": 3000}})], [("S", "H")])
    assert eh.entry("H").recipe_hash != eh2.entry("H").recipe_hash
    # hysteresis seed ORDERING is directional in v1 (below: strict ≤ permissive; above:
    # strict ≥ permissive — the core is the more extreme cut). v1 raised a bare comparison
    # error; explain the two seeds instead (2026-07-28)
    try:
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "hysteresis", "direction": "above"},
                          "params": {**NOCLEAN, "strict": 3000,
                                     "permissive": 3500}})], [("S", "H")]).pull("H")
        raise SystemExit("hysteresis above with strict < permissive should be refused")
    except ValueError as ex:
        assert "CORE cut" in str(ex) and "Swap them" in str(ex), str(ex)
    # method-specific params: the image-relative defaults are v1's own (percentile 5/95,
    # relative 0.30×/1.50×) and segment out of the box — no threshold typed in.
    # Per-plane fixture: 100 px @4000 + 36 px @3800 over 2304 px @500.
    for _md, _n in ((({}), 272),                        # ≥p95 → both bright squares
                    ({"method": "relative", "direction": "above"}, 272),  # >1.5×500 → both
                    ({"method": "percentile", "direction": "below"}, 4336)):  # ≤p5 → bg
        _e = eng(hds, henv, [("S", "io.kseed", {}),
                             ("H", "analysis.histogram_threshold",
                              {"modes": _md, "params": dict(NOCLEAN)})], [("S", "H")])
        assert int(_e.pull("H").get(D.VOXEL, "mask").values.sum()) == _n, _md
    # ... but an ABSOLUTE raw-count method with nothing set (or an explicit 0 = unset,
    # the GUI spin box's empty state) must ASK, naming the fields — not surface the
    # vendored dataclass's bare "requires `strict` and `permissive`" traceback
    for _p in ({}, {"strict": 3500, "permissive": 0}):
        try:
            eng(hds, henv, [("S", "io.kseed", {}),
                            ("H", "analysis.histogram_threshold",
                             {"modes": {"method": "hysteresis", "direction": "above"},
                              "params": _p})], [("S", "H")]).pull("H")
            raise SystemExit(f"hysteresis with {_p} should demand its thresholds")
        except ValueError as ex:
            assert "`permissive`" in str(ex) and "0 = unset" in str(ex), str(ex)
    try:                                    # hysteresis has no two-sided form
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "hysteresis", "direction": "between"},
                          "params": {**NOCLEAN, "strict": 3500, "permissive": 3000}})],
            [("S", "H")]).pull("H")
        raise SystemExit("hysteresis should reject direction=between")
    except ValueError as ex:
        assert "below/above only" in str(ex)
    _hspec = NODES.get("analysis.histogram_threshold")
    _vis = lambda st: {i.name for i in _hspec.active_inputs(st)}
    assert {"strict", "permissive"} <= _vis({"method": "hysteresis", "direction": "above"})
    assert not ({"low", "high", "percentile_high", "fraction_high"}
                & _vis({"method": "hysteresis", "direction": "above"}))
    assert "high" in _vis({"method": "single", "direction": "above"})
    assert "low" not in _vis({"method": "single", "direction": "above"})
    assert {"low", "high"} <= _vis({"method": "single", "direction": "between"})
    # max_area: the restored upper size filter (µm²→px², 0 = no limit). px=0.25 → 1 px² =
    # 0.0625 µm²; the 4000-square is 100 px² = 6.25 µm², the 3800 one 36 px² = 2.25 µm².
    assert _hspec.input("max_area") is not None and _hspec.input("max_area").unit == "um2"
    e_ma = eng(hds, henv, [("S", "io.kseed", {}),
                           ("H", "analysis.histogram_threshold",
                            {"modes": {"method": "relative", "direction": "above"},
                             "params": {**NOCLEAN, "max_area": 3.0}})], [("S", "H")])
    assert int(e_ma.pull("H").get(D.VOXEL, "mask").values.sum()) == 72   # 36 px × 2 planes
    assert e_ma.entry("H").recipe_hash != eh2.entry("H").recipe_hash     # the lever re-keys
    # ── v1 spatial-cleanup parity (2026-07-28) ─────────────────────────────────
    # The defaults ARE v1's: 100 px² min area, 1 px opening, 2 px closing, 50 px² holes,
    # carried as µm derives so they follow the objective. On a 30×30 blob + a 4×4 speck:
    # the speck is deleted by min_area and the blob keeps 896 px (opening/closing shave
    # its 4 corners). UNCALIBRATED, the derives resolve to 0 ⇒ cleanup off and area_um2
    # is NaN — v1's `voxel_size=None` path, never a fabricated 0.1 µm/px.
    axc = AxisSizes(m=1, t=1, z=1, c=1, y=128, x=128)
    cimg = np.full((1, 1, 1, 1, 128, 128), 500.0)
    cimg[0, 0, 0, 0, 20:50, 20:50] = 4000.0                   # 900 px
    cimg[0, 0, 0, 0, 100:104, 100:104] = 4000.0               # 16 px speck
    for _meta, _mask, _nlab, _px, _um2 in (
            ({"pixel_size_um": 0.25}, 896, 1, [896], [56.0]),
            ({}, 916, 2, [16, 900], None)):
        cds = Dataset(axes=axc, metadata=_meta).with_image(ArrayProvider(cimg))
        cenv = MetaEnvelope(axes=axc, metadata=_meta)
        _e = eng(cds, cenv, [("S", "io.kseed", {}),
                             ("H", "analysis.histogram_threshold", {})], [("S", "H")])
        _o = _e.pull("H")
        assert int(_o.get(D.VOXEL, "mask").values.sum()) == _mask, _meta
        assert int(_o.get(D.VOXEL, "labels").values.max()) == _nlab, _meta
        assert sorted(_o.get(D.LABEL, "area", layer="labels").values.tolist()) == _px
        _got = sorted(_o.get(D.LABEL, "area_um2", layer="labels").values.tolist())
        if _um2 is None:
            assert all(np.isnan(v) for v in _got), _got   # no calibration ⇒ no µm² fiction
        else:
            assert np.allclose(_got, _um2), _got
    hspec_ma = NODES.get("analysis.histogram_threshold").input("min_area")
    assert hspec_ma.derive == "100*(pixel_size_um or 0)**2"    # v1's 100 px², µm-authored

    # ── hysteresis agrees with the other methods at the same cut (2026-07-28) ───
    # The vendored kernel delegated its seeds to skimage's STRICT `>` while its own
    # docstring (and threshold_single, hence percentile/relative) promised `>=`/`<=`. So an
    # object plateau sitting exactly AT `strict` seeded nothing and hysteresis returned an
    # empty mask where percentile at the same resolved threshold found every object — the
    # visible bug on dim 12-bit data (objects at 192) and on saturated data (4095).
    from nodegraph.kernels.histogram_threshold import (
        HistogramThresholdSegmenter as _HTS, compute_histogram as _chist,
        make_config as _mkcfg)
    _flat = np.full((64, 64), 120, np.uint16)
    _flat[10:40, 10:40] = 192                                 # 900 px AT the cut value
    assert _chist(_flat, bit_depth=12).percentile(95) == 192   # what percentile resolves to

    def _kmask(im, **kw):
        cfg = _mkcfg(bit_depth=12, min_area=0, opening_radius=0, closing_radius=0,
                     min_hole_size=0, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(_HTS(cfg).run(im).mask.sum())
    # percentile ≡ single ≡ hysteresis, all inclusive, all 900 px
    assert _kmask(_flat, method="percentile", direction="above",
                  percentile_high=95.0) == 900
    assert _kmask(_flat, method="single", direction="above", high=192) == 900
    assert _kmask(_flat, method="hysteresis", direction="above",
                  strict=192, permissive=150) == 900           # was 0 before the fix
    assert _kmask(_flat, method="hysteresis", direction="below",
                  strict=120, permissive=150) == 3196          # was 0 before the fix
    # the general invariant: a DEGENERATE hysteresis (strict == permissive == T) is exactly
    # a single threshold at T — no fringe to grow, so the two must agree pixel for pixel.
    _rng = np.random.default_rng(0)
    for _im in ((np.arange(64 * 64).reshape(64, 64) % 500).astype(np.uint16),
                np.where(_rng.random((64, 64)) > 0.7, 3000, 400).astype(np.uint16),
                np.full((64, 64), 4095, np.uint16)):          # incl. a saturated plateau
        for _T in (0, 1, 137, 400, 499, 3000, 4095):
            assert (_kmask(_im, method="hysteresis", direction="above",
                           strict=_T, permissive=_T)
                    == _kmask(_im, method="single", direction="above", high=_T)), _T
            assert (_kmask(_im, method="hysteresis", direction="below",
                           strict=_T, permissive=_T)
                    == _kmask(_im, method="single", direction="below", low=_T)), _T

    # ── bit depth is METADATA, not an assumption (2026-07-28) ───────────────────
    # Most ND2s are 12-bit; declaring 16 mis-sized the percentile LUT and made the kernel
    # warn "max N is <5% of 16-bit range" on any dim frame. The depth now comes from the
    # `bit_depth` calibration key (memo-fenced), snapped UP to a kernel LUT, user-
    # overridable, and only falls back to 16 when the metadata is silent.
    from nodegraph.nodes import _snap_bit_depth as _snapbd
    assert [_snapbd(b) for b in (None, 0, 8, 11, 12, 14, 16, 20)] == \
           [16, 16, 8, 12, 12, 14, 16, 16]                    # round UP, never truncate
    axb = AxisSizes(m=1, t=1, z=1, c=1, y=64, x=64)
    bimg = np.full((1, 1, 1, 1, 64, 64), 120.0)               # a DIM 12-bit frame…
    bimg[0, 0, 0, 0, 10:40, 10:40] = 192.0                    # …max 192, <5% of 16-bit
    for _meta, _params, _warns in (
            ({"pixel_size_um": 0.25, "bit_depth": 12}, {}, 0),      # from the file
            ({"pixel_size_um": 0.25}, {"bit_depth": 12}, 0),        # user override
            ({"pixel_size_um": 0.25}, {}, 1)):                     # silent ⇒ 16 + warn
        bds = Dataset(axes=axb, metadata=_meta).with_image(ArrayProvider(bimg))
        _e = eng(bds, MetaEnvelope(axes=axb, metadata=_meta),
                 [("S", "io.kseed", {}),
                  ("H", "analysis.histogram_threshold", {"params": _params})],
                 [("S", "H")])
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            _o = _e.pull("H")
        assert int(_o.get(D.VOXEL, "mask").values.sum()) == 896, (_meta, _params)
        got = [x for x in _w if "<5% of" in str(x.message)]
        assert len(got) == _warns, (_meta, _params, [str(x.message) for x in got])
        if "bit_depth" in _meta:                    # the read is fenced like any calib
            assert "bit_depth" in {k for k, _ in _e.entry("H").reads}
    assert "bit_depth" in CALIBRATION_KEYS         # promoted into the engine schema
    assert envelope_symbols(MetaEnvelope(axes=axb, metadata={"bit_depth": 12})
                            )["bit_depth"] == 12   # …and visible to `derive`

    # ── the optional `raw` socket: segment on enhanced, MEASURE on raw ──────────
    # `raw` overrides the measured PIXELS only — mask/labels/thresholds and the whole
    # calibration env stay on the main chain (`data` is declared first, so dataset_preds
    # keeps it primary no matter which edge was wired first).
    define_node("io.kseed2", "R", outputs=[OutDataset()])
    axr = AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32)
    rawi = np.full((1, 1, 1, 1, 32, 32), 300.0)
    rawi[0, 0, 0, 0, 4:12, 4:12] = 700.0                  # two objects, DISTINCT counts
    rawi[0, 0, 0, 0, 20:28, 20:28] = 1500.0
    rmeta = {"pixel_size_um": 0.25, "bit_depth": 12}
    rawds = Dataset(axes=axr, metadata=rmeta).with_image(ArrayProvider(rawi))
    rawenv = MetaEnvelope(axes=axr, metadata=rmeta)

    def _measure_graph(wire_raw, raw_seed=None, raw_env=None):
        g = Graph()
        for nid, op in (("S", "io.kseed"), ("R", "io.kseed2")):
            g.add(NodeInstance(nid, op))
        g.add(NodeInstance("N", "enhance.normalize"))      # destroys the count scale
        g.add(NodeInstance("T", "analysis.threshold",
                           modes={"method": "fixed"}, params={"threshold": 0.2}))
        g.add(NodeInstance("L", "analysis.label", params={"name": "labels"}))
        g.add(NodeInstance("M", "analysis.measure", params={"labels": "labels"}))
        for a, b in (("S", "N"), ("N", "T"), ("T", "L"), ("L", "M")):
            g.connect(a, b)
        if wire_raw:
            g.connect("R", "M", dst_socket="raw")
        e = Engine(g, computes=COMPUTES,
                   seeds={"S": rawds, "R": raw_seed if raw_seed is not None else rawds},
                   meta_seeds={"S": rawenv,
                               "R": raw_env if raw_env is not None else rawenv})
        return g, e
    _g_plain, _e_plain = _measure_graph(False)
    _plain = _e_plain.pull("M").get(D.LABEL, "mean_intensity", layer="labels").values
    _g_raw, _e_raw = _measure_graph(True)
    _raw = _e_raw.pull("M").get(D.LABEL, "mean_intensity", layer="labels").values
    assert len(_plain) == len(_raw) == 2, (len(_plain), len(_raw))   # same segmentation
    # the normalized chain reports a rescaled shadow — 300/700/1500 counts became
    # 0/⅓/1, so even the RATIO between the two objects is gone (percentile clipping)
    assert np.allclose(sorted(_plain), [1.0 / 3.0, 1.0])
    assert np.allclose(sorted(_raw), [700.0, 1500.0])  # raw: the true per-region counts
    assert _e_plain.entry("M").recipe_hash != _e_raw.entry("M").recipe_hash   # re-keys
    assert [e.dst_socket for e in _g_raw.dataset_preds("M")] == ["data", "raw"]
    assert propagate_meta(_g_raw, {"S": rawenv, "R": rawenv})["M"] \
        .metadata.get("bit_depth") is None            # env still the (normalized) primary
    _g_rev, _e_rev = _measure_graph(True)             # wiring order must not matter
    _g_rev.edges.reverse()
    assert [e.dst_socket for e in _g_rev.dataset_preds("M")] == ["data", "raw"]
    # a geometry mismatch is silent corruption (voxel-for-voxel reads) → refuse
    axbad = AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16)
    badds = Dataset(axes=axbad, metadata=rmeta).with_image(
        ArrayProvider(np.zeros((1, 1, 1, 1, 16, 16))))
    try:
        _measure_graph(True, raw_seed=badds,
                       raw_env=MetaEnvelope(axes=axbad, metadata=rmeta))[1].pull("M")
        raise SystemExit("a mis-shaped `raw` input should be refused")
    except ValueError as ex:
        assert "does not match the measured input" in str(ex), str(ex)
    # histogram_threshold: mask from the main pixels, intensities re-measured on raw
    _g = Graph()
    for nid, op in (("S", "io.kseed"), ("R", "io.kseed2")):
        _g.add(NodeInstance(nid, op))
    _g.add(NodeInstance("H", "analysis.histogram_threshold",
                        modes={"method": "single", "direction": "above"},
                        params={**NOCLEAN, "high": 500}))
    _g.connect("S", "H")
    _g.connect("R", "H", dst_socket="raw")
    _e = Engine(_g, computes=COMPUTES, seeds={"S": rawds, "R": rawds},
                meta_seeds={"S": rawenv, "R": rawenv})
    _o = _e.pull("H")
    assert int(_o.get(D.VOXEL, "labels").values.max()) == 2          # both objects
    assert np.allclose(sorted(_o.get(D.LABEL, "mean_intensity",
                                    layer="labels").values), [700.0, 1500.0])

    # ── §7c: a node that changes what the numbers MEAN restamps bit_depth ───────
    # Calibration describes the CURRENT data, so downstream nodes read the transformed
    # depth, not the file's: summing T=8 12-bit frames IS 15-bit data. Both halves must
    # agree — the edit-time meta_transform prediction AND the pulled payload.
    axv = AxisSizes(m=1, t=8, z=4, c=1, y=16, x=16)
    vimg = _rng.integers(0, 4096, (1, 8, 4, 1, 16, 16)).astype(float)
    vmeta = {"pixel_size_um": 0.25, "bit_depth": 12, "dt_s": 1.0, "z_step_um": 0.5}
    vds = Dataset(axes=axv, metadata=vmeta).with_image(ArrayProvider(vimg))
    venv = MetaEnvelope(axes=axv, metadata=vmeta)
    for _nodes, _edges, _want in (
            ([("A", "util.stack", {"modes": {"method": "sum"}})], [("S", "A")], 15),
            ([("A", "util.stack", {"modes": {"method": "mean"}})], [("S", "A")], 12),
            ([("A", "util.zproject", {"modes": {"method": "sum"}})], [("S", "A")], 14),
            ([("A", "util.zproject", {"modes": {"method": "max"}})], [("S", "A")], 12),
            ([("A", "enhance.clahe", {})], [("S", "A")], 12),   # rescales back ⇒ no change
            ([("A", "enhance.normalize", {})], [("S", "A")], None),        # [0,1] ⇒ dropped
            ([("A", "util.stack", {"modes": {"method": "sum"}}),           # …and it CHAINS
              ("B", "util.zproject", {"modes": {"method": "sum"}})],
             [("S", "A"), ("A", "B")], 17)):
        _g = Graph()
        _g.add(NodeInstance("S", "io.kseed"))
        for _nid, _op, _kw in _nodes:
            _g.add(NodeInstance(_nid, _op, **_kw))
        for _a, _b in _edges:
            _g.connect(_a, _b)
        _e = Engine(_g, computes=COMPUTES, seeds={"S": vds}, meta_seeds={"S": venv})
        _last = _nodes[-1][0]
        _env_bd = propagate_meta(_g, {"S": venv})[_last].metadata.get("bit_depth")
        _pay_bd = _e.pull(_last).metadata.get("bit_depth")
        assert _env_bd == _want, (_nodes, _env_bd, _want)      # edit-time prediction
        assert _pay_bd == _want, (_nodes, _pay_bd, _want)      # payload in lockstep
    # a raw-count consumer downstream of the widening reads the NEW depth (fenced)
    _g = Graph()
    _g.add(NodeInstance("S", "io.kseed"))
    _g.add(NodeInstance("A", "util.stack", modes={"method": "sum"}))
    _g.add(NodeInstance("H", "analysis.histogram_threshold", params=dict(NOCLEAN)))
    _g.connect("S", "A"); _g.connect("A", "H")
    _e = Engine(_g, computes=COMPUTES, seeds={"S": vds}, meta_seeds={"S": venv})
    _e.pull("H")
    assert "bit_depth" in {k for k, _ in _e.entry("H").reads}
    # analysis.threshold's FIXED level follows the declared scale, and degrades to the
    # 0.5 normalized-data default when a Normalize dropped it
    _tspec = NODES.get("analysis.threshold").input("threshold")
    for _md, _want in (({"bit_depth": 12}, 2047.5), ({"bit_depth": 16}, 32767.5),
                       ({}, 0.5)):
        assert eval_derive(_tspec.derive,
                           envelope_symbols(MetaEnvelope(axes=axv, metadata=_md))) == _want
    try:                        # a µm² filter with NO pixel size cannot be honoured
        cds = Dataset(axes=axc, metadata={}).with_image(ArrayProvider(cimg))
        eng(cds, MetaEnvelope(axes=axc, metadata={}),
            [("S", "io.kseed", {}),
             ("H", "analysis.histogram_threshold",
              {"params": {"min_area": 1.0}})], [("S", "H")]).pull("H")
        raise SystemExit("a µm² filter without pixel_size_um should be refused")
    except ValueError as ex:
        assert "no `pixel_size_um` calibration" in str(ex), str(ex)

    # the filter is authored in µm², so the Label domain reports µm² beside px² — the
    # column that makes "read the areas, then pick the cut" possible (2026-07-28)
    a_um2 = e_ma.pull("H").get(D.LABEL, "area_um2", layer="labels").values
    a_px = e_ma.pull("H").get(D.LABEL, "area", layer="labels").values
    assert np.allclose(np.sort(a_um2), [2.25, 2.25])          # 36 px² × 0.25² µm², 2 planes
    assert np.allclose(a_um2, a_px * 0.25 * 0.25)
    # an area window that discards EVERYTHING reports the areas that ARE there, in µm²
    try:
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "relative", "direction": "above"},
                          "params": {**NOCLEAN, "min_area": 20.0}})], [("S", "H")]).pull("H")
        raise SystemExit("an area window that empties the series should say so")
    except ValueError as ex:
        assert "discarded every region" in str(ex) and "2.25" in str(ex), str(ex)
    try:                                    # …but blame the THRESHOLD when that is empty
        eng(hds, henv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold",
                         {"modes": {"method": "single", "direction": "above"},
                          "params": {**NOCLEAN, "high": 60000, "min_area": 1.0}})], [("S", "H")]).pull("H")
        raise SystemExit("an empty threshold should be named as the cause")
    except ValueError as ex:
        assert "THRESHOLD is what" in str(ex), str(ex)
    for _bad, _msg in (({"max_area": 0.01}, "under one pixel"),          # → 0 px² = OFF
                       ({"min_area": 5.0, "max_area": 3.0}, "size window is empty")):
        try:
            eng(hds, henv, [("S", "io.kseed", {}),
                            ("H", "analysis.histogram_threshold",
                             {"modes": {"method": "relative", "direction": "above"},
                              "params": {**NOCLEAN, **_bad}})], [("S", "H")]).pull("H")
            raise SystemExit(f"histogram_threshold should refuse {_bad}")
        except ValueError as ex:
            assert _msg in str(ex), str(ex)
    # a NORMALIZED [0,1] float image raises loudly (not silent quantization to {0,1})
    norm = np.full((1, 1, 1, 1, 16, 16), 0.2)
    norm[0, 0, 0, 0, 4:10, 4:10] = 0.9
    axn = AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16)
    nds = Dataset(axes=axn, metadata={"pixel_size_um": 0.25}).with_image(ArrayProvider(norm))
    nenv = MetaEnvelope(axes=axn, metadata={"pixel_size_um": 0.25})
    try:
        eng(nds, nenv, [("S", "io.kseed", {}),
                        ("H", "analysis.histogram_threshold", {})], [("S", "H")]).pull("H")
        raise SystemExit("histogram_threshold should reject a normalized [0,1] image")
    except ValueError as ex:
        assert "raw integer counts" in str(ex)

    # ── registration recovers a known drift + stores Frame attrs ───────────────
    from scipy.ndimage import shift as _ndshift
    base = np.random.default_rng(0).random((40, 40))
    rimg = np.zeros((1, 4, 1, 1, 40, 40))
    for tt in range(4):
        rimg[0, tt, 0, 0] = _ndshift(base, (2.0 * tt, -1.5 * tt), order=1)
    axr = AxisSizes(m=1, t=4, z=1, c=1, y=40, x=40)
    rds = Dataset(axes=axr, metadata={"pixel_size_um": 0.1}).with_image(ArrayProvider(rimg))
    renv = MetaEnvelope(axes=axr, metadata={"pixel_size_um": 0.1})
    er = eng(rds, renv, [("S", "io.kseed", {}),
                         ("R", "registration.stabilize",
                          {"modes": {"model": "translation", "reference": "first"}})],
             [("S", "R")])
    dyv = er.pull("R").get(D.FRAME, "drift_y").values[0]
    assert np.allclose(dyv, [0.0, -2.0, -4.0, -6.0], atol=0.3), dyv.tolist()

    # ── boundary bands on a label volume (general, any labels); z_step fenced ──
    axg = AxisSizes(m=1, t=1, z=10, c=1, y=32, x=32)
    lab = np.zeros((1, 1, 10, 1, 32, 32), dtype=np.int64)
    lab[0, 0, 3:7, 0, 6:12, 6:12] = 1
    lab[0, 0, 3:7, 0, 20:26, 20:26] = 2
    gds = (Dataset(axes=axg, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
           .with_image(ArrayProvider(np.zeros((1, 1, 10, 1, 32, 32))))
           .with_layer(D.VOXEL, "labels", lab))
    genv = MetaEnvelope(axes=axg, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
    egb = eng(gds, genv, [("S", "io.kseed", {}),
                          ("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
              [("S", "B")])
    bands = egb.pull("B").get(D.VOXEL, "bands").values
    assert int((bands == 1).sum()) > 0 and int((bands == 2).sum()) > 0
    assert int(((bands != 0) & (lab != 0)).sum()) == 0            # bands avoid interiors
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in egb.entry("B").reads}
    # `band_um` is the EDT criterion only — the dilation path grows by band_voxels
    # iterations and never reads it, so it stays hidden under dilation. band_voxels is
    # live in BOTH (dilation iterations; the edt fallback threshold when band_um == 0).
    _bvis = lambda st: {i.name for i in NODES.get("analysis.boundary_band").active_inputs(st)}
    assert "band_um" in _bvis({"method": "edt"})
    assert "band_um" not in _bvis({"method": "dilation"})
    assert {"band_voxels", "include_neighbors"} <= _bvis({"method": "dilation"})
    assert {"band_voxels", "include_neighbors"} <= _bvis({"method": "edt"})

    # ── ROI mask: empty → whole-frame; a rect shape rasterizes correctly ───────
    axm = AxisSizes(m=1, t=1, z=1, c=1, y=20, x=20)
    mds = Dataset(axes=axm).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))
    menv = MetaEnvelope(axes=axm)
    e_full = eng(mds, menv, [("S", "io.kseed", {}),
                             ("R", "analysis.roi_mask", {})], [("S", "R")])
    assert int(e_full.pull("R").get(D.VOXEL, "roi_mask").values.sum()) == 20 * 20
    e_rect = eng(mds, menv, [("S", "io.kseed", {}),
                             ("R", "analysis.roi_mask",
                              {"params": {"shapes": [{"type": "rect", "op": "add",
                                                      "vertices": [[5, 5], [15, 15]]}]}})],
                 [("S", "R")])
    rm = e_rect.pull("R").get(D.VOXEL, "roi_mask").values[0, 0, 0, 0]
    assert rm[8, 8] == 1 and rm[0, 0] == 0 and int(rm.sum()) > 0

    # ── dic_correlate is WIRED to its kernel (full 2D-DIC verification — real shift
    #    recovery when al-dic is present — lives in test_catalog_dic). Here: only the
    #    dep-gated path, when al-dic is ABSENT, must raise a clear ImportError. When it is
    #    present, skip the heavy solve in this group (covered end-to-end elsewhere). ──
    from nodegraph.kernels.dic_correlate import al_dic_available as _dic_avail
    if not _dic_avail():
        caught = 0
        try:
            eng(hds, henv, [("S", "io.kseed", {}), ("G", "analysis.dic_correlate", {})],
                [("S", "G")]).pull("G")
        except ImportError as ex:
            caught = int("al-dic" in str(ex))
        assert caught == 1, "dic_correlate did not raise a clear al-dic ImportError"
    dic_note = ("dic_correlate wired (al-dic present → verified in test_catalog_dic)"
                if _dic_avail() else "dic_correlate wired but dep-gated (al-dic ImportError)")

    _ok(f"catalog (v1 kernel ports): {beads_note}; histogram-threshold "
        "mask+labels+regions (methods re-key); registration recovers drift; boundary "
        f"bands (µm-fenced, any labels); ROI mask (empty→whole, rect); {dic_note}")


def test_catalog_dvc() -> None:
    """DVC/ALDVC displacement + strain field (``analysis.dvc_field``) → a Point field,
    and its Voxel rasterizer (``transform.rasterize_field``). V2.06. Asserts: 2D/3D
    recover a known shift (µm), the invariant Point schema + disp/strain/qfactor columns,
    2D≠3D recipe hash, calibration fenced (pixel_size_um / z_step_um), the reference-mode
    lever (previous_frame skips t0) + the optional external-reference input socket, and
    the rasterizer reproducing a planted linear field at grid points. Needs scipy/skimage."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (DVC + field rasterize): SKIPPED (scipy/skimage absent)")
        return
    from scipy.ndimage import gaussian_filter, shift as ndshift
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable

    define_node("io.dvcseed", "S", outputs=[OutDataset()])

    def eng(seeds, envs, nodes, edges):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        return Engine(g, computes=COMPUTES, seeds=seeds, meta_seeds=envs)

    rng = np.random.default_rng(3)
    patt = gaussian_filter(rng.random((64, 64)).astype(float), 1.5)
    px = 0.2
    P = {"subset_size": 16, "subset_spacing": 12, "admm_iterations": 1, "seed_levels": 1}

    # ── 2D fixed_frame, T=2: frame0=ref (self-pair→~0), frame1=shift (dy,dx) ──────
    img = np.zeros((1, 2, 1, 1, 64, 64))
    img[0, 0, 0, 0] = patt
    img[0, 1, 0, 0] = ndshift(patt, (1.5, -2.0), order=3, mode="nearest")
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=64, x=64)
    meta = {"pixel_size_um": px}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax, metadata=meta)
    e2 = eng({"S": ds}, {"S": env},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
             [("S", "D")])
    o2 = e2.pull("D")
    names2 = {a.name for a in o2.layers_on(D.POINT) if a.layer == "dvc"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um",
            "qfactor", "strain_yx"} <= names2, sorted(names2)
    assert "disp_z" not in names2, "2D field must not carry disp_z"
    tc = o2.get(D.POINT, "t", layer="dvc").values
    dy = o2.get(D.POINT, "disp_y", layer="dvc").values
    dx = o2.get(D.POINT, "disp_x", layer="dvc").values
    s1 = tc == 1
    assert abs(float(np.median(dy[s1])) - 1.5 * px) < 0.06, float(np.median(dy[s1]))
    assert abs(float(np.median(dx[s1])) + 2.0 * px) < 0.06, float(np.median(dx[s1]))
    assert float(np.median(o2.get(D.POINT, "disp_mag_um", layer="dvc").values[tc == 0])) < 0.02
    assert any(k == "pixel_size_um" for k, _ in e2.entry("D").reads)

    # ── 3D fixed_frame, T=2: disp_z present, recovers (z,y,x), z_step fenced ──────
    vol = gaussian_filter(rng.random((16, 40, 40)).astype(float), 1.2)
    img3 = np.zeros((1, 2, 16, 1, 40, 40))
    img3[0, 0, :, 0] = vol
    img3[0, 1, :, 0] = ndshift(vol, (1.0, 1.0, 1.5), order=3, mode="nearest")
    ax3 = AxisSizes(m=1, t=2, z=16, c=1, y=40, x=40)
    meta3 = {"pixel_size_um": px, "z_step_um": 0.5}
    ds3 = Dataset(axes=ax3, metadata=meta3).with_image(ArrayProvider(img3))
    e3 = eng({"S": ds3}, {"S": MetaEnvelope(axes=ax3, metadata=meta3)},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field",
               {"modes": {"dim": "3D"},
                "params": {"subset_size": 8, "subset_spacing": 10,
                           "admm_iterations": 1, "seed_levels": 1}})],
             [("S", "D")])
    o3 = e3.pull("D")
    t3 = o3.get(D.POINT, "t", layer="dvc").values == 1
    dz = o3.get(D.POINT, "disp_z", layer="dvc").values[t3]
    assert o3.get(D.POINT, "disp_z", layer="dvc") is not None
    assert abs(float(np.median(dz)) - 1.0 * 0.5) < 0.08, float(np.median(dz))
    assert abs(float(np.median(o3.get(D.POINT, "disp_x", layer="dvc").values[t3])) - 1.5 * px) < 0.08
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e3.entry("D").reads}
    assert e2.entry("D").recipe_hash != e3.entry("D").recipe_hash, "2D/3D lever must re-key"
    # 3D mode on a single-plane series is a hard error (kernel gotcha 6)
    axf = AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32)
    dsf = Dataset(axes=axf, metadata=meta).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 32, 32))))
    try:
        eng({"S": dsf}, {"S": MetaEnvelope(axes=axf, metadata=meta)},
            [("S", "io.dvcseed", {}),
             ("D", "analysis.dvc_field", {"modes": {"dim": "3D"}})], [("S", "D")]).pull("D")
        raise SystemExit("3D DVC on a single plane should raise")
    except ValueError as ex:
        assert "z>1" in str(ex)

    # ── optional external reference input socket (a second file as the reference) ─
    imgp = np.zeros((1, 1, 1, 1, 64, 64)); imgp[0, 0, 0, 0] = ndshift(patt, (2.0, 0.0), order=3, mode="nearest")
    imgr = np.zeros((1, 1, 1, 1, 64, 64)); imgr[0, 0, 0, 0] = patt
    axp = AxisSizes(m=1, t=1, z=1, c=1, y=64, x=64)
    ep = MetaEnvelope(axes=axp, metadata=meta)
    ex = eng({"S": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgp)),
              "Sr": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgr))},
             {"S": ep, "Sr": ep},
             [("S", "io.dvcseed", {}), ("Sr", "io.dvcseed", {}),
              ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
             [("S", "D", "data"), ("Sr", "D", "reference")])
    dyx = ex.pull("D").get(D.POINT, "disp_y", layer="dvc").values
    assert abs(float(np.median(dyx)) - 2.0 * px) < 0.06, float(np.median(dyx))
    # regression: the compute's calibration env is the PRIMARY (data) input's, even when
    # the reference is wired FIRST and carries a different pixel size — propagate_meta
    # must seed from the first-declared dataset socket, not edge-insertion order.
    metaR = {"pixel_size_um": 0.9}                              # a wrong scale if picked
    exr = eng({"S": Dataset(axes=axp, metadata=meta).with_image(ArrayProvider(imgp)),
               "Sr": Dataset(axes=axp, metadata=metaR).with_image(ArrayProvider(imgr))},
              {"S": ep, "Sr": MetaEnvelope(axes=axp, metadata=metaR)},
              [("S", "io.dvcseed", {}), ("Sr", "io.dvcseed", {}),
               ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)})],
              [("Sr", "D", "reference"), ("S", "D", "data")])   # reference wired FIRST
    dyr = exr.pull("D").get(D.POINT, "disp_y", layer="dvc").values
    # µm scale ⇒ primary px=0.2 (≈0.4 µm), NOT the reference px=0.9 (≈1.8 µm)
    assert abs(float(np.median(dyr)) - 2.0 * px) < 0.06, \
        f"DVC used the reference's calibration ({float(np.median(dyr)):.3f} µm ⇒ px≈0.9)"
    assert any(k == "pixel_size_um" for k, _ in exr.entry("D").reads)

    # ── previous_frame self-reference skips t0 (no increment into the first frame) ─
    imgt = np.zeros((1, 3, 1, 1, 64, 64))
    for tt in range(3):
        imgt[0, tt, 0, 0] = ndshift(patt, (1.0 * tt, 0.0), order=3, mode="nearest")
    axt = AxisSizes(m=1, t=3, z=1, c=1, y=64, x=64)
    ept = eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
              {"S": MetaEnvelope(axes=axt, metadata=meta)},
              [("S", "io.dvcseed", {}),
               ("D", "analysis.dvc_field",
                {"modes": {"dim": "2D", "reference_mode": "previous_frame"},
                 "params": dict(P)})], [("S", "D")])
    tp = ept.pull("D").get(D.POINT, "t", layer="dvc").values
    assert 0 not in set(tp.tolist()) and {1, 2} <= set(tp.tolist()), sorted(set(tp.tolist()))

    # ── accumulate: previous_frame increments → cumulative Lagrangian field ────────
    # imgt shifts +1 voxel/frame, so each increment ≈ 1·px and the cumulative field is
    # 1·px at t=1, 2·px at t=2 (composition, not a fixed-frame re-correlation).
    ea = eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
             {"S": MetaEnvelope(axes=axt, metadata=meta)},
             [("S", "io.dvcseed", {}),
              ("D", "analysis.dvc_field",
               {"modes": {"dim": "2D", "reference_mode": "previous_frame"},
                "params": dict(P)}),
              ("A", "analysis.accumulate_field", {})],
             [("S", "D"), ("D", "A")])
    oa = ea.pull("A")
    na = {a.name for a in oa.layers_on(D.POINT) if a.layer == "dvc_cumulative"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um",
            "qfactor", "strain_yx"} <= na, sorted(na)
    ta = oa.get(D.POINT, "t", layer="dvc_cumulative").values
    dya = oa.get(D.POINT, "disp_y", layer="dvc_cumulative").values
    assert {1, 2} == set(ta.tolist()), sorted(set(ta.tolist()))
    assert abs(float(np.median(dya[ta == 1])) - 1.0 * px) < 0.06, float(np.median(dya[ta == 1]))
    assert abs(float(np.median(dya[ta == 2])) - 2.0 * px) < 0.08, float(np.median(dya[ta == 2]))
    assert any(k == "pixel_size_um" for k, _ in ea.entry("A").reads)   # calib fenced
    # guard: a fixed_frame (already cumulative) field is refused
    try:
        eng({"S": Dataset(axes=axt, metadata=meta).with_image(ArrayProvider(imgt))},
            {"S": MetaEnvelope(axes=axt, metadata=meta)},
            [("S", "io.dvcseed", {}),
             ("D", "analysis.dvc_field", {"modes": {"dim": "2D"}, "params": dict(P)}),
             ("A", "analysis.accumulate_field", {})], [("S", "D"), ("D", "A")]).pull("A")
        raise SystemExit("accumulate on a fixed_frame field should raise")
    except ValueError as exa:
        assert "already cumulative" in str(exa), str(exa)
    # guard: a Point field with no DVC provenance is refused
    axg = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    tblg = StructureTable(D.POINT, {
        "id": np.arange(2, dtype=np.int64), "m": np.zeros(2, np.int64),
        "t": np.zeros(2, np.int64), "c": np.zeros(2, np.int64),
        "z": np.zeros(2), "y": np.array([2.0, 4.0]), "x": np.array([2.0, 4.0]),
        "disp_y": np.zeros(2), "disp_x": np.zeros(2),
    }, layer="dvc", z_kind="plane_index")
    dsg = (Dataset(axes=axg, metadata={"pixel_size_um": px})
           .with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 8, 8)))).with_structure(tblg))
    try:
        eng({"S": dsg}, {"S": MetaEnvelope(axes=axg, metadata={"pixel_size_um": px})},
            [("S", "io.dvcseed", {}),
             ("A", "analysis.accumulate_field", {})], [("S", "A")]).pull("A")
        raise SystemExit("accumulate without dvc provenance should raise")
    except ValueError as exb:
        assert "provenance" in str(exb), str(exb)

    # ── rasterize a planted linear Point field → Voxel layer reproduces it ────────
    # Dimensionality is INHERITED from the table's z_kind (§7b): plane_index → 2D per-plane,
    # no dim lever passed. `with_structure` preserved z_kind into `__struct_zkind__`.
    axi = AxisSizes(m=1, t=1, z=1, c=1, y=20, x=20)
    yy, xx = np.meshgrid([4.0, 9.0, 14.0], [4.0, 9.0, 14.0], indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    tbl = StructureTable(D.POINT, {
        "id": np.arange(len(yy), dtype=np.int64), "m": np.zeros(len(yy), np.int64),
        "t": np.zeros(len(yy), np.int64), "c": np.zeros(len(yy), np.int64),
        "z": np.zeros(len(yy)), "y": yy, "x": xx,
        "disp_x": xx.copy(), "disp_y": np.zeros(len(yy)),
    }, layer="dvc", z_kind="plane_index")
    dsi = (Dataset(axes=axi, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))
           .with_structure(tbl))
    assert dsi.structure_zkind(D.POINT, "dvc") == "plane_index"        # z_kind preserved
    orr = eng({"S": dsi}, {"S": MetaEnvelope(axes=axi, metadata={})},
              [("S", "io.dvcseed", {}),
               ("R", "transform.rasterize_field", {})],
              [("S", "R")]).pull("R")
    rx = orr.get(D.VOXEL, "dvc_disp_x").values
    assert rx.shape == (1, 1, 1, 1, 20, 20) and np.isfinite(rx).all()
    assert abs(float(rx[0, 0, 0, 0, 9, 9]) - 9.0) < 0.5 and abs(float(rx[0, 0, 0, 0, 9, 14]) - 14.0) < 0.5
    # §7b regression: a 2D per-plane field (z_kind=plane_index) on a z>1 image must NOT be
    # misread as a 3D grid (which — with no lever — the old default would do from ax.z>1,
    # bleeding a single plane's points across all z). Inherited z_kind keeps it per-plane.
    axv = AxisSizes(m=1, t=1, z=3, c=1, y=16, x=16)
    yv, xv = np.meshgrid([3.0, 8.0, 12.0], [3.0, 8.0, 12.0], indexing="ij")
    yv, xv = yv.ravel(), xv.ravel()
    tblv = StructureTable(D.POINT, {
        "id": np.arange(len(yv), dtype=np.int64), "m": np.zeros(len(yv), np.int64),
        "t": np.zeros(len(yv), np.int64), "c": np.zeros(len(yv), np.int64),
        "z": np.full(len(yv), 1.0), "y": yv, "x": xv, "disp_x": xv.copy(),
    }, layer="dvc", z_kind="plane_index")
    dsv = (Dataset(axes=axv, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 3, 1, 16, 16))))
           .with_structure(tblv))
    rxv = eng({"S": dsv}, {"S": MetaEnvelope(axes=axv, metadata={})},
              [("S", "io.dvcseed", {}), ("R", "transform.rasterize_field", {})],
              [("S", "R")]).pull("R").get(D.VOXEL, "dvc_disp_x").values
    assert np.count_nonzero(rxv[0, 0, 0]) == 0 and np.count_nonzero(rxv[0, 0, 2]) == 0, \
        "2D per-plane field bled into other z planes (misread as 3D — §7b inheritance failed)"
    assert np.isfinite(rxv[0, 0, 1]).all() and abs(float(rxv[0, 0, 1, 0, 8, 8]) - 8.0) < 0.5
    # rasterize without a matching Point layer is a hard error
    try:
        eng({"S": Dataset(axes=axi, metadata={}).with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 20, 20))))},
            {"S": MetaEnvelope(axes=axi, metadata={})},
            [("S", "io.dvcseed", {}),
             ("R", "transform.rasterize_field", {})], [("S", "R")]).pull("R")
        raise SystemExit("rasterize without a Point layer should raise")
    except ValueError as ex2:
        assert "Point layer" in str(ex2)

    _ok("catalog (DVC + field rasterize): 2D/3D recover known shift (µm), Point schema "
        "+ disp/strain/qfactor; 2D≠3D hash; pixel/z-step fenced; external-ref socket + "
        "previous-frame skips t0; accumulate composes increments → cumulative (t·px, "
        "provenance-inherited, refuses fixed_frame/no-provenance); rasterizer reproduces "
        "a linear field + inherits z_kind (§7b: 2D field on z>1 not misread as 3D); "
        "guards (z<2, missing Point) raise")


def test_catalog_dic() -> None:
    """DIC (pyALDIC, ``analysis.dic_correlate``) — the 2D image sibling of DVC, WIRED to the
    vendored kernel. ``al-dic`` is an optional dep: when absent the node is fully wired but
    its compute raises a friendly ImportError at run time (it reaches ``run_pyaldic_pair`` —
    not a bare stub that errors before touching inputs). Asserts the structural spec + the
    gated-run behavior (and, if al-dic is installed, a real Point-field pull)."""
    if not _HAVE_SKIMAGE:
        _ok("catalog (DIC pyALDIC): SKIPPED (scipy/skimage absent)")
        return
    from scipy.ndimage import gaussian_filter, shift as ndshift
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.kernels.dic_correlate import al_dic_available

    # structural spec — really wired (Point output, reference lever + ROI, WHOLE_SERIES),
    # not the old bare-stub registration
    s = NODES.get("analysis.dic_correlate")
    assert s.category == "analysis"
    assert D.POINT in s.adds_domains and D.VOXEL in s.reads_domains
    assert s.granularity is Granularity.WHOLE_SERIES
    assert {"data", "reference", "reference_frame", "winsize", "winstepsize", "roi",
            "compute_strain"} <= {i.name for i in s.inputs}
    _modes = {mm.name: set(mm.choices) for mm in s.modes}
    assert _modes.get("reference_mode") == {"fixed_frame", "previous_frame"}
    # V2.18 solver lever. Everything only al_dic's `if para.use_global_step:` block reads
    # must be hidden in Local mode — a live-looking knob the chosen solver ignores.
    assert _modes.get("solver") == {"aldic", "local"}, _modes
    _gated = {i.name: i.available_in for i in s.inputs if i.available_in}
    for _p in ("mu", "admm_max_iter", "disp_smoothness"):
        assert _gated.get(_p, {}).get("solver") == frozenset({"aldic"}), \
            f"{_p} must be gated to the AL-DIC solver, got {_gated.get(_p)}"

    # a 3-frame 2D graph (frame0 = reference; frames 1,2 carry k*(Δy, Δx)). fixed_frame
    # (default) pairs each frame against t0 in ONE series solve, so t=1/t=2 carry 1x/2x.
    define_node("io.dicseed", "S", outputs=[OutDataset()])
    rng = np.random.default_rng(5)
    patt = gaussian_filter(rng.random((128, 128)), 1.0)      # speckle-like texture
    img = np.zeros((1, 3, 1, 1, 128, 128))
    for _t in range(3):                                      # +1 y, +3 x px per step
        img[0, _t, 0, 0] = (patt if _t == 0 else
                            ndshift(patt, (1.0 * _t, 3.0 * _t), order=3, mode="nearest"))
    px = 0.5
    ax = AxisSizes(m=1, t=3, z=1, c=1, y=128, x=128)
    meta = {"pixel_size_um": px}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    P = {"winsize": 20, "winstepsize": 16, "admm_max_iter": 4, "icgn_max_iter": 100}

    def _dic(params=None, modes=None):
        g = Graph()
        g.add(NodeInstance("S", "io.dicseed"))
        g.add(NodeInstance("D", "analysis.dic_correlate",
                           params={**P, **(params or {})}, modes=modes or {}))
        g.connect("S", "D", dst_socket="data")
        return Engine(g, computes=COMPUTES, seeds={"S": ds},
                      meta_seeds={"S": MetaEnvelope(axes=ax, metadata=meta)})

    e = _dic()
    if not al_dic_available():
        raised = ""                                         # dep-gated env: friendly error
        try:
            e.pull("D")
        except ImportError as ex:
            raised = str(ex)
        assert "al-dic" in raised.lower(), f"expected a friendly al-dic ImportError, got {raised!r}"
        _ok("catalog (DIC pyALDIC): wired 2D DIC node (reference lever + solver lever + ROI + "
            "Point output, WHOLE_SERIES); dep-gated — reaches run_pyaldic_series, friendly "
            "al-dic ImportError when the solver is absent")
        return

    # al-dic present → a real correlation must recover the planted shift (µm) + the Point schema
    out = e.pull("D")                                       # runs IC-GN + ADMM (numba JIT: slow once)
    names = {a.name for a in out.layers_on(D.POINT) if a.layer == "dic"}
    assert {"id", "m", "t", "c", "z", "y", "x", "disp_x", "disp_y", "disp_mag_um"} <= names, \
        sorted(names)
    assert "disp_z" not in names, "DIC is 2D"
    assert "strain_yx" not in names, "strain is opt-in (compute_strain defaults off)"

    def _med(o, col, t):
        v = o.get(D.POINT, col, layer="dic").values
        return float(np.median(v[o.get(D.POINT, "t", layer="dic").values == t]))

    for _t in (1, 2):                    # the whole series comes from ONE solve now
        assert abs(_med(out, "disp_y", _t) - _t * 1.0 * px) < 0.12, (_t, _med(out, "disp_y", _t))
        assert abs(_med(out, "disp_x", _t) - _t * 3.0 * px) < 0.12, (_t, _med(out, "disp_x", _t))
    assert _med(out, "disp_mag_um", 0) < 0.1, "the self-pair at the reference must be ~0"
    assert any(k == "pixel_size_um" for k, _ in e.entry("D").reads)   # calib fenced
    assert out.metadata.get("dic_solver") == "aldic"
    assert out.metadata.get("dic_unsolved_points") == 0

    # previous_frame → per-step increments (t0 has no reference and is absent)
    e_prev = _dic(modes={"reference_mode": "previous_frame"})
    o_prev = e_prev.pull("D")
    t_prev = set(o_prev.get(D.POINT, "t", layer="dic").values.astype(int).tolist())
    assert t_prev == {1, 2}, t_prev
    assert abs(_med(o_prev, "disp_x", 2) - 3.0 * px) < 0.12, _med(o_prev, "disp_x", 2)

    # Local-DIC solver: same field, no ADMM. Distinct recipe hash so the memo cannot
    # hand an AL-DIC result back for a Local run.
    e_loc = _dic(modes={"solver": "local"})
    o_loc = e_loc.pull("D")
    assert abs(_med(o_loc, "disp_x", 2) - 6.0 * px) < 0.12, _med(o_loc, "disp_x", 2)
    assert o_loc.metadata.get("dic_solver") == "local"
    assert e_loc.entry("D").recipe_hash != e.entry("D").recipe_hash, \
        "solver mode must fold into the recipe hash"

    # REGRESSION GUARD (V2.18): a LARGE FIRST STEP must still be found. The series
    # restructure originally passed init_guess_mode="auto", which al_dic maps to
    # "previous" — and in fixed_frame mode the prepended reference self-pair solves to ~0
    # and then hands that ~0 forward as the next frame's initial guess, so a big first
    # displacement had to be found by local refinement alone and was not. A 9 px shift
    # came back as a ~6 px error. Every frame must get its own coarse search.
    big = np.zeros((1, 2, 1, 1, 128, 128))
    big[0, 0, 0, 0] = patt
    big[0, 1, 0, 0] = ndshift(patt, (-6.0, 9.0), order=3, mode="nearest")
    ax_b = AxisSizes(m=1, t=2, z=1, c=1, y=128, x=128)
    ds_b = Dataset(axes=ax_b, metadata=meta).with_image(ArrayProvider(big))
    g_b = Graph()
    g_b.add(NodeInstance("S", "io.dicseed"))
    g_b.add(NodeInstance("D", "analysis.dic_correlate", params=P))
    g_b.connect("S", "D", dst_socket="data")
    o_big = Engine(g_b, computes=COMPUTES, seeds={"S": ds_b},
                   meta_seeds={"S": MetaEnvelope(axes=ax_b, metadata=meta)}).pull("D")
    assert abs(_med(o_big, "disp_y", 1) - (-6.0 * px)) < 0.15, _med(o_big, "disp_y", 1)
    assert abs(_med(o_big, "disp_x", 1) - (9.0 * px)) < 0.15, _med(o_big, "disp_x", 1)

    # compute_strain → al_dic's nodal displacement gradients as strain_* columns. The two
    # CROSS terms are the ones that pin the axis convention: dy/dx and dx/dy are the
    # components al_dic reports y-up, and the kernel negates them back to image axes.
    o_str = _dic({"compute_strain": True}).pull("D")
    snames = {a.name for a in o_str.layers_on(D.POINT) if a.layer == "dic"}
    assert {"strain_yy", "strain_yx", "strain_xy", "strain_xx"} <= snames, sorted(snames)
    for _c in ("strain_yy", "strain_yx", "strain_xy", "strain_xx"):
        assert abs(_med(o_str, _c, 2)) < 5e-3, (_c, _med(o_str, _c, 2))  # rigid shift ⇒ 0

    _ok("catalog (DIC pyALDIC): wired 2D DIC (reference + solver levers, ROI, Point output, "
        "WHOLE_SERIES) ran end-to-end — ONE series solve recovers a planted (1,3)px/frame "
        "shift across t=1,2 in µm, self-pair~0, a LARGE (-6,9)px first step is still found "
        "(the init-guess regression), previous_frame gives increments, Local solver agrees + "
        "rehashes, compute_strain adds a zero strain tensor, calib fenced")


def test_channel_derive() -> None:
    """C8 / H12 — per-channel derive resolution (``ctx.channel``). A c-iterating node's
    metadata-intelligent param DERIVES from *that channel's* emission λ (distinct λ →
    distinct value); a user override applies to all channels; the derive's optics deps are
    memo-fenced. Also checks ``detect.spots`` resolves its radii per channel end-to-end."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES, register_node

    # a probe node (FAKE op_key — never clobber a real node) whose one param derives from
    # emission λ; capture what ctx.channel(c) resolves for each channel.
    cap: dict = {}

    def _c8(ctx):
        ax = ctx.inputs[0].image.axes
        cap["per_ch"] = [ctx.channel(c).param("radius") for c in range(ax.c)]
        cap["emis"] = [ctx.channel(c).emission_nm() for c in range(ax.c)]
        return ctx.inputs[0]

    register_node(_c8, op_key="test.c8_probe", label="c8 probe", category="analysis",
                  inputs=[InDataset(), InFloat("radius", "R", unit="um", field=True,
                          derive="0.61*(emission_nm or 520)/(na or 1.4)/1000")],
                  outputs=[OutDataset()])
    define_node("io.c8seed", "S", outputs=[OutDataset()])

    ax = AxisSizes(m=1, t=1, z=1, c=2, y=4, x=4)
    meta = {"objective_na": 1.0, "channel_emission_nm": [450, 650]}
    ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(np.zeros((1, 1, 1, 2, 4, 4))))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def run(params):
        g = Graph(); g.add(NodeInstance("S", "io.c8seed"))
        g.add(NodeInstance("P", "test.c8_probe", params=params)); g.connect("S", "P")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
        e.pull("P"); return e

    # derive path: each channel resolves from ITS OWN emission (0.61·λ/NA)
    e = run({})
    r = cap["per_ch"]
    assert abs(r[0] - 0.61 * 450 / 1.0 / 1000) < 1e-9, r
    assert abs(r[1] - 0.61 * 650 / 1.0 / 1000) < 1e-9, r
    assert r[0] != r[1], "per-channel derive collapsed to a single value"
    assert cap["emis"] == [450, 650]
    reads = {k for k, _ in e.entry("P").reads}
    assert {"channel_emission_nm", "objective_na"} <= reads, sorted(reads)   # deps fenced
    # override path: an explicit param wins and applies to ALL channels (one widget)
    run({"radius": 0.9})
    assert cap["per_ch"] == [0.9, 0.9], cap["per_ch"]

    # end-to-end: detect.spots resolves radii per channel (no radius params → derive path),
    # detects the blob in BOTH channels, and now memo-fences on per-channel emission.
    if _HAVE_SKIMAGE:
        yy, xx = np.mgrid[0:24, 0:24]
        blob = np.exp(-(((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 2.0 ** 2)))
        img = np.zeros((1, 1, 1, 2, 24, 24)); img[0, 0, 0, 0] = blob; img[0, 0, 0, 1] = blob
        axs = AxisSizes(m=1, t=1, z=1, c=2, y=24, x=24)
        ms = {"pixel_size_um": 0.1, "objective_na": 1.0, "channel_emission_nm": [450, 650]}
        dss = Dataset(axes=axs, metadata=ms).with_image(ArrayProvider(img))
        gs = Graph(); gs.add(NodeInstance("S", "io.c8seed"))
        gs.add(NodeInstance("N", "detect.spots", modes={"dim": "2D"},
                            params={"threshold": 0.03}))
        gs.connect("S", "N")
        es = Engine(gs, computes=COMPUTES, seeds={"S": dss},
                    meta_seeds={"S": MetaEnvelope(axes=axs, metadata=ms)})
        cc = es.pull("N").get(D.POINT, "c", layer="spots")
        assert cc is not None and set(cc.values.tolist()) == {0, 1}, "spots missed a channel"
        assert "channel_emission_nm" in {k for k, _ in es.entry("N").reads}, \
            "detect.spots not memo-fenced on per-channel emission"

    _ok("channel derive (C8/H12): ctx.channel per-c derive (distinct λ → distinct value), "
        "override applies to all channels, optics deps fenced; detect.spots resolves radii "
        "per channel end-to-end")


def test_cluster_points() -> None:
    """``analysis.cluster_points`` (V2.07; ported v1 ``granule_cluster``, scikit-learn) — an
    unlabeled 3-D Point cloud → a per-point cluster-id column via a GaussianMixture/BIC fit.
    Asserts it recovers two planted clusters (relax=0 ⇒ k=2), preserves the source z_kind,
    fences calibration, that gmm≠kmeans re-keys the memo, and that it chains into
    ``tessellate`` → ``rasterize_mesh``. Skips if scikit-learn is absent."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        _ok("cluster points: SKIPPED (scikit-learn absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable as _ST

    define_node("io.cpseed", "S", outputs=[OutDataset()])
    # two well-separated 3-D clusters, 6 non-coplanar points each (>= min_granule_points)
    star = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    A = star + np.array([5.0, 12, 12])
    B = star + np.array([12.0, 40, 40])
    P = np.vstack([A, B]); n = len(P)
    tbl = _ST(D.POINT, {
        "id": np.arange(n, dtype=np.int64), "m": np.zeros(n, np.int64),
        "t": np.zeros(n, np.int64), "c": np.zeros(n, np.int64),
        "z": P[:, 0], "y": P[:, 1], "x": P[:, 2]}, layer="particles", z_kind="subpixel")
    ax = AxisSizes(m=1, t=1, z=18, c=1, y=56, x=56)
    meta = {"pixel_size_um": 0.2, "z_step_um": 0.5}
    ds = (Dataset(axes=ax, metadata=meta)
          .with_image(ArrayProvider(np.zeros((1, 1, 18, 1, 56, 56)))).with_structure(tbl))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def eng(nodes, edges, sink):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        e = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": env})
        return e, e.pull(sink)

    e, o = eng([("S", "io.cpseed", {}),
                ("C", "analysis.cluster_points",
                 {"params": {"n_clusters": 2, "relax_pct": 0.0}})], [("S", "C")], "C")
    clus = o.get(D.POINT, "cluster", layer="particles").values
    assert set(clus.tolist()) == {0, 1}, sorted(set(clus.tolist()))     # relax=0 ⇒ k=2
    # each planted cluster is one id, and the two differ (ids themselves are arbitrary)
    assert len(set(clus[:6].tolist())) == 1 and len(set(clus[6:].tolist())) == 1
    assert clus[0] != clus[6], "the two planted clusters collapsed into one"
    assert o.structure_zkind(D.POINT, "particles") == "subpixel"        # source z_kind kept
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e.entry("C").reads}
    # the model mode folds into the recipe hash (gmm vs kmeans memoize separately)
    eg, _ = eng([("S", "io.cpseed", {}), ("C", "analysis.cluster_points",
                 {"modes": {"method": "gmm"}, "params": {"n_clusters": 2}})], [("S", "C")], "C")
    ek, _ = eng([("S", "io.cpseed", {}), ("C", "analysis.cluster_points",
                 {"modes": {"method": "kmeans"}, "params": {"n_clusters": 2}})], [("S", "C")], "C")
    assert eg.entry("C").recipe_hash != ek.entry("C").recipe_hash, "method must re-key"
    # full granule-chain hand-off: cluster_points → tessellate → rasterize_mesh → 2 regions
    _, ot = eng([("S", "io.cpseed", {}),
                 ("C", "analysis.cluster_points", {"params": {"n_clusters": 2, "relax_pct": 0.0}}),
                 ("T", "analysis.tessellate", {}),
                 ("R", "transform.rasterize_mesh", {})],
                [("S", "C"), ("C", "T"), ("T", "R")], "R")
    rast = ot.get(D.VOXEL, "labels").values
    assert set(np.unique(rast).tolist()) == {0, 1, 2}                   # bg + 2 regions
    assert np.all(ot.get(D.LABEL, "volume_um3", layer="labels").values > 0)

    _ok("cluster points (V2.07): GMM/BIC recovers 2 planted clusters (relax=0⇒k=2) → per-point "
        "cluster column; z_kind kept; calib fenced; gmm≠kmeans re-keys; chains into "
        "tessellate → rasterize_mesh (particles→cluster→mesh→raster)")


def test_mesh_domain() -> None:
    """``Domain.MESH`` (V2.08) — the eleventh domain, stored as three flat CSR strata under
    one domain addressed by layer sub-keys (``L`` / ``L/vert`` / ``L/face``).

    Asserts: the domain predicates (structure, NOT lattice, no axis-set, a clean raise from
    the lattice-only helpers); a build → ``with_mesh`` → ``read_mesh`` round-trip preserving
    all three bucket lengths + the per-(m,t,c) dense ids; ``content_hash`` EQUAL across two
    independently built identical meshes (the positive assertion that closes the
    object-dtype pointer-hash trap) and ``_canon`` refusing object dtype outright; every
    ``validate()`` invariant; three independent ``__struct_zkind__`` stamps; the clean
    plan-time transfer refusal (MESH registers no bridge on purpose); and the face-parity
    interior test exact on a convex box AND a concave L-prism. numpy only."""
    from nodegraph.domains import (DOMAIN_ABBR, DOMAIN_COLOR, STRUCTURE_DOMAINS,
                                   domain_abbr, domain_color, is_lattice, is_structure)
    from nodegraph.kernels.mesh_raster import FaceParity
    from nodegraph.memo import _canon
    from nodegraph.structure import COORD_COLUMNS
    from nodegraph.transfer import plan_transfer
    from dataclasses import replace
    import nodegraph.mesh as MSH

    # ── the enum member + its presentation ────────────────────────────────────────
    assert D.MESH.value == "mesh" and D.MESH in STRUCTURE_DOMAINS
    assert is_structure(D.MESH) and not is_lattice(D.MESH)
    assert axes_of(D.MESH) is None                       # omission IS the non-lattice decl
    # DOMAIN_COLOR must be populated in the SAME commit as the member: nodelab_v2.theme
    # iterates the whole enum to build its QColor map, so a gap crashes the GUI at import.
    assert D.MESH in DOMAIN_COLOR and D.MESH in DOMAIN_ABBR
    assert domain_color(D.MESH) == DOMAIN_COLOR[D.MESH] and domain_abbr(D.MESH) == "MSH"
    for bad in (lambda: AX.shape_for(D.MESH), lambda: AX.axis_list(D.MESH)):
        try:
            bad()
            raise SystemExit("a lattice-only helper must reject MESH")
        except ValueError:
            pass

    # ── a hand-built 3-element mesh spanning two (m,t,c) frames ───────────────────
    def cube_mesh(z0, y0, x0, s=4.0):
        V = np.array([[z0 + dz, y0 + dy, x0 + dx]
                      for dz in (0.0, s) for dy in (0.0, s) for dx in (0.0, s)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        F = []
        for a, b, c, d in quads:
            F += [[a, b, c], [a, c, d]]
        return V, np.array(F, np.int64)

    def elements():
        out = []
        for i, (yo, mm) in enumerate([(2.0, 0), (12.0, 0), (2.0, 1)]):
            V, F = cube_mesh(2.0, yo, 2.0)
            out.append(MSH.MeshElement(m=mm, t=0, c=0, src_label=7 + i, verts_zyx=V,
                                       faces=F, centroid_zyx=(4.0, yo + 2.0, 4.0),
                                       volume_um3=1.5, surface_area_um2=2.5,
                                       density=float(3 - i), n_points=8))
        return out

    tb = MSH.build_mesh_tables(elements(), layer="surf")
    assert (tb.element.n, tb.vertex.n, tb.face.n) == (3, 24, 36)
    ec = tb.element.columns
    assert np.asarray(ec["id"]).tolist() == [1, 2, 1]           # dense per (m,t,c)
    assert np.asarray(ec["element_uid"]).tolist() == [0, 1, 2]  # global FK, row order
    assert np.asarray(ec["m"]).tolist() == [0, 0, 1]
    assert np.asarray(ec["closed"]).tolist() == [1, 1, 1]       # derived, watertight
    assert np.asarray(ec["src_label"]).tolist() == [7, 8, 9]    # original ids survive
    assert np.asarray(ec["vert_start"]).tolist() == [0, 8, 16]
    # every bucket carries the invariant coordinate schema; a face row is topology, so it
    # carries no centroid (the one documented exception, see nodegraph.mesh).
    for tbl, need in ((tb.element, COORD_COLUMNS), (tb.vertex, COORD_COLUMNS),
                      (tb.face, ("id", "m", "t", "c"))):
        assert set(need) <= set(tbl.columns), sorted(tbl.columns)
    for tbl in (tb.element, tb.vertex, tb.face):
        for col, v in tbl.columns.items():
            a = np.asarray(v)
            assert a.dtype != object and a.ndim == 1, (tbl.layer, col, a.dtype)

    # ── memo identity: flat columns hash by CONTENT, object dtype is refused ───────
    assert MSH.build_mesh_tables(elements(), layer="surf").content_hash() == tb.content_hash()
    obj = np.empty(2, dtype=object)
    obj[0] = np.zeros(3)
    obj[1] = np.zeros(2)
    try:
        _canon(obj)
        raise SystemExit("_canon must refuse an object-dtype array (it hashes pointers)")
    except TypeError as ex:
        assert "object-dtype" in str(ex)

    # ── attach / read round-trip + three independent z_kind stamps ────────────────
    ds = Dataset(axes=AxisSizes(m=2, t=1, z=16, c=1, y=32, x=32))
    ds = MSH.with_mesh(ds, tb, provenance={"boundary": "convex_hull", "source": "points"})
    assert MSH.mesh_names(ds) == ("surf",)
    for lay in ("surf", "surf/vert", "surf/face"):
        assert ds.structure_zkind(D.MESH, lay) == "subpixel", lay
    rt = MSH.read_mesh(ds, "surf")
    assert rt.content_hash() == tb.content_hash(), "mesh did not survive the store"
    assert MSH.mesh_provenance(ds, "surf")["boundary"] == "convex_hull"
    v1, f1 = MSH.mesh_element(rt, 1)                  # element 1 slices out, faces LOCAL
    assert v1.shape == (8, 3) and f1.min() == 0 and f1.max() == 7
    assert np.allclose(v1, cube_mesh(2.0, 12.0, 2.0)[0])
    assert MSH.mesh_layer("q", "vert") == "q/vert"
    assert MSH.mesh_part("q/vert") == ("q", "vert") and MSH.mesh_part("plain") == ("plain", None)

    # ── validate() is the ONLY enforcement (the store skips non-lattice checks) ────
    def broken(**cols):
        c = dict(tb.element.columns)
        c.update(cols)
        bad = MSH.MeshTables(replace(tb.element, columns=c), tb.vertex, tb.face, layer="surf")
        try:
            bad.validate()
        except ValueError:
            return True
        return False
    assert broken(vert_count=np.array([8, 8, 9], np.int64)), "CSR sum must be checked"
    assert broken(vert_start=np.array([0, 9, 16], np.int64)), "CSR prefix must be checked"
    assert broken(element_uid=np.array([1, 2, 3], np.int64)), "the FK target must be dense"
    assert broken(id=np.array([1, 3, 1], np.int64)), "ids must be dense per (m,t,c)"
    try:
        MSH.build_mesh_tables(elements(), layer="a/vert")
        raise SystemExit("the stratum separator must be reserved in a mesh name")
    except ValueError as ex:
        assert "reserved" in str(ex)
    # the store itself really does NOT length-check a non-lattice layer — which is exactly
    # why with_mesh() has to validate, and why nothing may bypass it.
    torn = Dataset().with_layer(D.MESH, "a", np.zeros(5), layer="surf") \
                   .with_layer(D.MESH, "b", np.zeros(3), layer="surf")
    assert len(torn.layers_on(D.MESH)) == 2

    # ── no bridge: MESH must fail at PLAN time, not with a wrong answer ────────────
    for dst in (D.VOXEL, D.LABEL, D.FRAME):
        try:
            plan_transfer(D.MESH, dst)
            raise SystemExit(f"MESH->{dst} must have no transfer route")
        except ValueError as ex:
            assert "route" in str(ex).lower(), str(ex)

    # ── the face-parity interior test, exact on convex AND concave ────────────────
    zz, yy, xx = np.mgrid[0:9, 0:9, 0:9]
    q = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(float)
    Vb, Fb = cube_mesh(2.0, 2.0, 2.0)
    box = FaceParity(Vb, Fb, (1.0, 1.0, 1.0)).contains(q).reshape(9, 9, 9)
    want = (zz >= 2) & (zz < 6) & (yy >= 2) & (yy < 6) & (xx >= 2) & (xx < 6)
    assert np.array_equal(box, want), "parity must be exact on an axis-aligned box"
    # an L-shaped prism: the case a convex-hull test CANNOT represent
    poly = np.array([[2, 2], [2, 8], [4, 8], [4, 4], [8, 4], [8, 2]], float)
    npv = len(poly)
    VL = np.vstack([np.column_stack([np.full(npv, 2.0), poly]),
                    np.column_stack([np.full(npv, 6.0), poly])])
    FL = []
    for i in range(npv):
        j = (i + 1) % npv
        FL += [[i, j, npv + i], [j, npv + j, npv + i]]
    for i in range(1, npv - 1):
        FL += [[0, i, i + 1], [npv, npv + i + 1, npv + i]]
    FL = np.array(FL, np.int64)
    assert MSH.faces_are_closed(FL) and not MSH.faces_are_closed(FL[:4])
    gotL = FaceParity(VL, FL, (1.0, 1.0, 1.0)).contains(q).reshape(9, 9, 9)
    inL = (((yy >= 2) & (yy < 4) & (xx >= 2) & (xx < 8))
           | ((yy >= 4) & (yy < 8) & (xx >= 2) & (xx < 4)))
    assert np.array_equal(gotL, (zz >= 2) & (zz < 6) & inL), "parity must honor concavity"
    assert int(gotL.sum()) < int(((zz >= 2) & (zz < 6) & (yy >= 2) & (yy < 8)
                                 & (xx >= 2) & (xx < 8)).sum())
    # analytic geometry off the mesh, in µm, anisotropic
    vox = (0.5, 0.2, 0.2)
    assert abs(MSH.enclosed_volume_um3(Vb, Fb, vox) - (4 * 0.5) * (4 * 0.2) ** 2) < 1e-9
    assert MSH.surface_area_um2(Vb, Fb, vox) > 0
    assert MSH.surface_area_um2(Vb, Fb[:0], vox) == 0.0        # no faces => 0, never a crash

    _ok("mesh domain (V2.08): 11th domain = 3 flat CSR strata on one Domain.MESH "
        "(L / L/vert / L/face); structure-not-lattice; build→with_mesh→read_mesh round-trips "
        "(dense per-frame ids + global FK + derived closed); content_hash content-stable and "
        "_canon refuses object dtype; validate() catches torn CSR/FK/ids/reserved-name; "
        "3 z_kind stamps; no bridge ⇒ clean plan-time refusal; face parity exact on a box "
        "AND a concave L-prism")


def test_tessellate_split() -> None:
    """``analysis.tessellate`` → ``transform.rasterize_mesh`` (V2.08) — the v1 split restored
    in v2 with a real MESH intermediate, replacing the fused ``analysis.tessellate_volume``.

    Reuses the fused node's exact cube fixture so the OLD numbers must still hold end-to-end
    (ids ``[1,2]``, convex-hull ``volume_um3 == 0.54 µm³``, centroids 5.5/10.5, raster
    ``{0,1,2}``, ``z_kind`` stamped, calibration fenced, chains into ``boundary_band``), and
    adds: the intermediate MESH payload exists + validates; the concave ``alpha_shape`` path
    rasterizes STRICTLY SMALLER than its convex hull (proving the faces-based interior test
    is real, and fixing the latent bug where every mode filled convex); ``label_surface``
    round-trips a Voxel label region → mesh → raster; per-mode socket gating; the ``z<2`` /
    missing-layer guards; and a two-fresh-Engine output-fingerprint match. scipy+skimage."""
    if not _HAVE_SKIMAGE:                                    # skimage ⟹ scipy present
        _ok("tessellate split: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable as _ST
    from nodegraph.kernels.mesh_raster import interior_test_for
    from nodegraph.memo import output_fingerprint
    import nodegraph.mesh as MSH

    define_node("io.tvseed", "S", outputs=[OutDataset()])

    def cube(z0, y0, x0):                                    # 8 corners of a 3-voxel cube
        return np.array([[z0 + dz, y0 + dy, x0 + dx]
                         for dz in (0, 3) for dy in (0, 3) for dx in (0, 3)], dtype=float)
    P = np.vstack([cube(4, 10, 10), cube(9, 30, 30)])       # two well-separated clusters
    L = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    n = len(P)
    tbl = _ST(D.POINT, {
        "id": np.arange(n, dtype=np.int64), "m": np.zeros(n, np.int64),
        "t": np.zeros(n, np.int64), "c": np.zeros(n, np.int64),
        "z": P[:, 0], "y": P[:, 1], "x": P[:, 2], "cluster": L},
        layer="particles", z_kind="subpixel")
    ax = AxisSizes(m=1, t=1, z=16, c=1, y=48, x=48)
    meta = {"pixel_size_um": 0.2, "z_step_um": 0.5}          # cube side = 3 vox
    blank = np.zeros((1, 1, 16, 1, 48, 48))
    ds = (Dataset(axes=ax, metadata=meta)
          .with_image(ArrayProvider(blank)).with_structure(tbl))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def run(nodes, edges, sink, seed=None, seedenv=None):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        eng = Engine(g, computes=COMPUTES, seeds={"S": seed if seed is not None else ds},
                     meta_seeds={"S": seedenv if seedenv is not None else env})
        return eng, eng.pull(sink)

    TESS = ("T", "analysis.tessellate", {})
    RAST = ("R", "transform.rasterize_mesh", {})
    CHAIN = [("S", "io.tvseed", {}), TESS, RAST]
    WIRE = [("S", "T"), ("T", "R")]

    # ── the MESH intermediate is a real, validated payload on its own wire ────────
    et, om = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    mt = MSH.read_mesh(om, "mesh").validate()
    assert mt.element.n == 2 and mt.vertex.n == 16 and mt.face.n > 0
    assert np.asarray(mt.element.columns["id"]).tolist() == [1, 2]
    assert np.asarray(mt.element.columns["src_label"]).tolist() == [0, 1]
    assert om.structure_zkind(D.MESH, "mesh") == "subpixel"
    assert MSH.mesh_provenance(om, "mesh")["boundary"] == "convex_hull"
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in et.entry("T").reads}
    assert D.MESH in NODES.get("analysis.tessellate").adds_domains
    # vertices are VOXEL coords (the domain convention) — i.e. the input cloud itself
    vz = np.asarray(mt.vertex.columns["z"], dtype=float)
    assert vz.min() >= 4.0 - 1e-9 and vz.max() <= 12.0 + 1e-9, (vz.min(), vz.max())

    # ── the rasterized half reproduces the FUSED node's numbers exactly ───────────
    e, o = run(CHAIN, WIRE, "R")
    rast = o.get(D.VOXEL, "labels").values
    assert rast.shape == (1, 1, 16, 1, 48, 48)
    assert set(np.unique(rast).tolist()) == {0, 1, 2}        # bg + 2 regions
    names = {a.name for a in o.layers_on(D.LABEL) if a.layer == "labels"}
    assert {"id", "m", "t", "c", "z", "y", "x", "volume_um3", "density", "n_points",
            "surface_area_um2", "voxel_count"} <= names, sorted(names)
    ids = o.get(D.LABEL, "id", layer="labels").values.tolist()
    vol = o.get(D.LABEL, "volume_um3", layer="labels").values
    zc = sorted(o.get(D.LABEL, "z", layer="labels").values.tolist())
    assert ids == [1, 2]
    # convex hull of a 3-vox cube: (3·0.5)·(3·0.2)·(3·0.2) = 0.54 µm³
    assert np.allclose(vol, 0.54, atol=1e-6), vol.tolist()
    assert abs(zc[0] - 5.5) < 0.1 and abs(zc[1] - 10.5) < 0.1, zc      # cube centers
    assert np.all(o.get(D.LABEL, "voxel_count", layer="labels").values > 0)
    assert o.structure_zkind(D.LABEL, "labels") == "subpixel"          # §7b provenance
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in e.entry("R").reads}

    # ── chain into the GENERAL boundary_band (reads the "labels" raster) ──────────
    _, ob = run(CHAIN + [("B", "analysis.boundary_band", {"params": {"band_voxels": 1}})],
                WIRE + [("R", "B")], "B")
    bands = ob.get(D.VOXEL, "bands").values
    assert int((bands != 0).sum()) > 0 and int(((bands != 0) & (rast != 0)).sum()) == 0

    # ── the concave path is REAL: alpha_shape fills less than its convex hull ─────
    # one label over two separated cubes — the convex hull spans the gap, a finite alpha
    # drops the long bridging tetrahedra. This is the bug the fused node hid: it stored the
    # UNFILTERED Delaunay, whose simplices union to the convex hull in EVERY mode.
    P2 = np.vstack([cube(4, 8, 8), cube(4, 8, 20)])
    n2 = len(P2)
    tbl2 = _ST(D.POINT, {
        "id": np.arange(n2, dtype=np.int64), "m": np.zeros(n2, np.int64),
        "t": np.zeros(n2, np.int64), "c": np.zeros(n2, np.int64),
        "z": P2[:, 0], "y": P2[:, 1], "x": P2[:, 2],
        "cluster": np.zeros(n2, np.int64)}, layer="particles", z_kind="subpixel")
    ds2 = (Dataset(axes=ax, metadata=meta)
           .with_image(ArrayProvider(blank)).with_structure(tbl2))
    _, oc = run(CHAIN, WIRE, "R", seed=ds2)
    _, oa = run([("S", "io.tvseed", {}),
                 ("T", "analysis.tessellate", {"modes": {"boundary": "alpha_shape"},
                                               "params": {"alpha_um": 1.2}}), RAST],
                WIRE, "R", seed=ds2)
    n_convex = int((oc.get(D.VOXEL, "labels").values != 0).sum())
    n_alpha = int((oa.get(D.VOXEL, "labels").values != 0).sum())
    assert n_alpha > 0, "the alpha-shape mesh dissolved — alpha_um is too small"
    assert n_alpha < n_convex, (n_alpha, n_convex)
    assert MSH.mesh_provenance(oa, "mesh")["boundary"] == "alpha_shape"
    # and the interior test really is DERIVED from that provenance, with no user lever
    assert interior_test_for({"boundary": "convex_hull"}, True) == "convex"
    assert interior_test_for({"boundary": "alpha_shape"}, True) == "watertight"
    assert interior_test_for({"boundary": "alpha_shape"}, False) == "convex"  # open => fall back
    assert interior_test_for({}, True) == "watertight"                       # hand-built mesh
    assert not NODES.get("transform.rasterize_mesh").modes

    # ── label_surface: a Voxel label region → mesh → raster round-trip ────────────
    lab = np.zeros((1, 1, 16, 1, 48, 48), dtype=np.int64)
    lab[0, 0, 4:12, 0, 10:20, 10:20] = 1
    dsl = (Dataset(axes=ax, metadata=meta)
           .with_image(ArrayProvider(blank)).with_layer(D.VOXEL, "labels", lab))
    _, ol = run([("S", "io.tvseed", {}),
                 ("T", "analysis.tessellate", {"modes": {"boundary": "label_surface"},
                                               "params": {"name": "surf"}}),
                 ("R", "transform.rasterize_mesh", {"params": {"mesh": "surf",
                                                               "name": "again"}})],
                WIRE, "R", seed=dsl)
    back = ol.get(D.VOXEL, "again").values != 0
    orig = lab != 0
    iou = float((back & orig).sum()) / float((back | orig).sum())
    assert iou > 0.95, f"label_surface round-trip IoU {iou:.3f}"
    assert MSH.read_mesh(ol, "surf").face.n > 0

    # ── determinism across two fresh Engines ─────────────────────────────────────
    # The mesh CONTENT hash must match: this is the assertion that fails loudly the day
    # someone reintroduces a ragged/object-dtype mesh column (identical content would then
    # hash differently, because ndarray canonicalization would be hashing pointers).
    # ``output_fingerprint`` deliberately can NOT be compared here — its Dataset branch
    # folds each layer's fresh monotonic ``revision``, which is per-session lookup identity
    # by design, not a content hash.
    _, oa2 = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    _, ob2 = run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T")
    assert MSH.read_mesh(oa2, "mesh").content_hash() == MSH.read_mesh(ob2, "mesh").content_hash()
    assert output_fingerprint(oa2) and output_fingerprint(ob2)
    # the boundary lever must re-key. Compared WITHIN one Engine on purpose: recipe_hash
    # folds monotonic layer revisions, so it is a per-session lookup key and is not stable
    # across two independent Engines (true of every node, e.g. rr.reroute — not a mesh
    # property). The content hash above is what carries cross-run determinism.
    em, _ = run([("S", "io.tvseed", {}), TESS,
                 ("V", "analysis.tessellate", {"modes": {"boundary": "voronoi"}})],
                [("S", "T"), ("S", "V")], "T")
    em.pull("V")
    assert em.entry("T").recipe_hash != em.entry("V").recipe_hash, "boundary must re-key"

    # ── guards + per-mode socket gating ───────────────────────────────────────────
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=48, x=48)
    ds1 = (Dataset(axes=ax1, metadata=meta)
           .with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 48, 48)))).with_structure(tbl))
    try:
        run([("S", "io.tvseed", {}), TESS], [("S", "T")], "T", seed=ds1,
            seedenv=MetaEnvelope(axes=ax1, metadata=meta))
        raise SystemExit("z<2 should raise")
    except ValueError as ex:
        assert "z>1" in str(ex)
    try:
        run([("S", "io.tvseed", {}),
             ("T", "analysis.tessellate", {"params": {"cluster": "nope"}})],
            [("S", "T")], "T")
        raise SystemExit("missing label column should raise")
    except ValueError as ex2:
        assert "label column" in str(ex2)
    try:
        run([("S", "io.tvseed", {}), ("R", "transform.rasterize_mesh", {})],
            [("S", "R")], "R")
        raise SystemExit("a missing mesh should raise")
    except ValueError as ex3:
        assert "no mesh" in str(ex3)

    # each boundary choice must show ONLY the sockets it reads (§5b)
    _avis = lambda st: {i.name for i in NODES.get("analysis.tessellate").active_inputs(st)}
    assert "alpha_um" in _avis({"boundary": "alpha_shape"})
    assert "alpha_um" not in _avis({"boundary": "convex_hull"})
    assert "alpha_um" not in _avis({"boundary": "voronoi"})
    for pt_mode in ("convex_hull", "alpha_shape", "voronoi"):
        vis = _avis({"boundary": pt_mode})
        assert {"points", "cluster", "min_points", "name"} <= vis, pt_mode
        assert not ({"labels", "iso_level", "decimate"} & vis), pt_mode
    ls = _avis({"boundary": "label_surface"})
    assert {"labels", "iso_level", "decimate", "name"} <= ls
    assert not ({"points", "cluster", "alpha_um", "min_points"} & ls)
    # the voxelization params moved to the rasterizer, and nothing gates them
    rm = {i.name for i in NODES.get("transform.rasterize_mesh").active_inputs({})}
    assert {"mesh", "name", "smooth_um", "min_voxels", "fill_holes"} <= rm

    _ok("tessellate split (V2.08): analysis.tessellate → MESH → transform.rasterize_mesh "
        "reproduces the fused node's numbers (ids [1,2], volume_um3=cube 0.54, centroids "
        "5.5/10.5, z_kind, calib fenced, chains into boundary_band) + a validated MESH "
        "intermediate with voxel-space verts; concave alpha_shape now fills LESS than its "
        "convex hull (interior test derived from provenance, no lever); label_surface "
        "round-trips a label region at IoU>0.95; mesh content_hash deterministic across "
        "Engines + boundary re-keys; z<2 / missing-layer guards; per-mode socket gating for "
        "all 4 boundary choices")


def test_label_to_points_voronoi() -> None:
    """``transform.label_to_points`` → ``analysis.voronoi`` — dots + areas → territory cells.

    The two halves of the "link a point cloud to a region layer" workflow. Covers:
    label→points in both dimensionalities with the dim INHERITED from the Label instance's
    ``z_kind`` (no lever); the ``inside`` position landing inside a concave region whose true
    centroid is background; ``weighted`` following the intensities; the ``raw``-on-a-geometric-
    mode refusal; then Voronoi with all three ``bound`` modes, per-region arenas that cells
    provably never cross, the ``max_distance_um`` cap, the µm-space partition on anisotropic
    voxels, watertight 3D cell surfaces that round-trip through ``transform.rasterize_mesh``,
    ``mesh: skip``, per-mode socket/Mode gating and the guards. scipy+skimage."""
    if not _HAVE_SKIMAGE:                                    # skimage ⟹ scipy present
        _ok("label_to_points / voronoi: SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.structure import StructureTable as _ST
    import nodegraph.mesh as MSH

    define_node("io.vseed", "S", outputs=[OutDataset()])
    META = {"pixel_size_um": 0.2, "z_step_um": 0.5}

    def run(nodes, edges, sink, seed, ax):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for e in edges:
            g.connect(e[0], e[1], dst_socket=(e[2] if len(e) > 2 else "data"))
        eng = Engine(g, computes=COMPUTES, seeds={"S": seed},
                     meta_seeds={"S": MetaEnvelope(axes=ax, metadata=META)})
        return eng, eng.pull(sink)

    def col(ds, dom, name, layer):
        return np.asarray(ds.get(dom, name, layer=layer).values)

    # ── fixture: two stacked rectangular "areas", brightness biased to the left ───
    ax2 = AxisSizes(m=1, t=1, z=1, c=1, y=40, x=40)
    mask = np.zeros((1, 1, 1, 1, 40, 40), dtype=np.int64)
    mask[0, 0, 0, 0, 4:20, 4:36] = 1                         # 16 x 32 = 512 voxels
    mask[0, 0, 0, 0, 24:36, 4:36] = 1                        # 12 x 32 = 384 voxels
    img = np.zeros((1, 1, 1, 1, 40, 40))
    img[0, 0, 0, 0, 4:20, 4:20] = 100.0                      # bright LEFT half of area 1
    ds2 = (Dataset(axes=ax2, metadata=META).with_image(ArrayProvider(img))
           .with_layer(D.VOXEL, "mask", mask))
    CC2 = ("C", "analysis.label", {"params": {"mask": "mask", "name": "areas"},
                                   "modes": {"dim": "2D"}})
    L2P = ("P", "transform.label_to_points", {"params": {"labels": "areas"}})

    # ── label → points: 2D, dim inherited, `label` column, auto-derived name ──────
    ep, op_ = run([("S", "io.vseed", {}), CC2, L2P], [("S", "C"), ("C", "P")], "P", ds2, ax2)
    got = {a.name for a in op_.layers_on(D.POINT) if a.layer == "areas_points"}
    assert {"id", "m", "t", "c", "z", "y", "x", "label"} <= got, sorted(got)
    assert col(op_, D.POINT, "id", "areas_points").tolist() == [0, 1]      # global-unique
    assert col(op_, D.POINT, "label", "areas_points").tolist() == [1, 2]   # source region
    assert col(op_, D.POINT, "y", "areas_points").tolist() == [11.5, 29.5]
    assert col(op_, D.POINT, "x", "areas_points").tolist() == [19.5, 19.5]
    # a 2D (plane_index) Label instance yields per-plane points, with NO dim lever to disagree
    assert op_.structure_zkind(D.POINT, "areas_points") == "plane_index"
    assert col(op_, D.POINT, "z", "areas_points").tolist() == [0.0, 0.0]
    assert not [m for m in NODES.get("transform.label_to_points").modes
                if m.role == "dim_lever"], "dimensionality is inherited, not levered"
    assert {"pixel_size_um", "z_step_um"} <= {k for k, _ in ep.entry("P").reads}

    # ── position modes: weighted follows the pixels, inside stays in the region ───
    _, ow = run([("S", "io.vseed", {}), CC2,
                 ("P", "transform.label_to_points",
                  {"params": {"labels": "areas"}, "modes": {"position": "weighted"}})],
                [("S", "C"), ("C", "P")], "P", ds2, ax2)
    assert col(ow, D.POINT, "x", "areas_points").tolist() == [11.5, 19.5]  # pulled left
    # a C-shaped region: its centre of mass is BACKGROUND, so only `inside` is usable
    cmask = np.zeros((1, 1, 1, 1, 40, 40), dtype=np.int64)
    cmask[0, 0, 0, 0, 8:32, 8:12] = 1                        # spine
    cmask[0, 0, 0, 0, 8:12, 8:28] = 1                        # top arm
    cmask[0, 0, 0, 0, 28:32, 8:28] = 1                       # bottom arm
    dsc = (Dataset(axes=ax2, metadata=META)
           .with_image(ArrayProvider(np.zeros((1, 1, 1, 1, 40, 40))))
           .with_layer(D.VOXEL, "mask", cmask))
    lut = cmask[0, 0, 0, 0]
    for mode, want_inside in (("centroid", False), ("inside", True)):
        _, oc = run([("S", "io.vseed", {}), CC2,
                     ("P", "transform.label_to_points",
                      {"params": {"labels": "areas"}, "modes": {"position": mode}})],
                    [("S", "C"), ("C", "P")], "P", dsc, ax2)
        yy = int(round(float(col(oc, D.POINT, "y", "areas_points")[0])))
        xx = int(round(float(col(oc, D.POINT, "x", "areas_points")[0])))
        assert bool(lut[yy, xx]) is want_inside, (mode, yy, xx, lut[yy, xx])

    # ── refusals: `raw` on a geometric mode, and a non-label raster ───────────────
    try:
        run([("S", "io.vseed", {}), CC2, L2P],
            [("S", "C"), ("C", "P"), ("C", "P", "raw")], "P", ds2, ax2)
        raise SystemExit("raw on position=centroid should raise")
    except ValueError as exr:
        assert "only read by position=weighted" in str(exr), str(exr)
    try:
        run([("S", "io.vseed", {}),
             ("P", "transform.label_to_points", {"params": {"labels": "mask"}})],
            [("S", "P")], "P", ds2, ax2)
        raise SystemExit("a bare mask is not a label raster")
    except ValueError as exm:
        assert "not a LABEL raster" in str(exm), str(exm)

    # ── voronoi 2D: four dots, per-region arenas ──────────────────────────────────
    dots = _ST(D.POINT, {
        "id": np.arange(4, dtype=np.int64), "m": np.zeros(4, np.int64),
        "t": np.zeros(4, np.int64), "c": np.zeros(4, np.int64), "z": np.zeros(4),
        "y": np.array([10., 10., 30., 30.]), "x": np.array([10., 30., 10., 30.])},
        layer="dots", z_kind="plane_index")
    ds2d = ds2.with_structure(dots)
    VOR = ("V", "analysis.voronoi", {"params": {"points": "dots", "region": "areas"},
                                     "modes": {"dim": "2D", "bound": "per_region"}})
    ev, ov = run([("S", "io.vseed", {}), CC2, VOR], [("S", "C"), ("C", "V")], "V",
                 ds2d, ax2)
    rast = np.asarray(ov.get(D.VOXEL, "voronoi").values)
    assert sorted(np.unique(rast).tolist()) == [0, 1, 2, 3, 4]         # bg + 4 cells
    assert int((rast != 0).sum()) == int((mask != 0).sum())            # the areas, exactly
    assert col(ov, D.LABEL, "region", "voronoi").tolist() == [1, 1, 2, 2]
    assert sorted(col(ov, D.LABEL, "point_id", "voronoi").tolist()) == [0, 1, 2, 3]
    assert col(ov, D.LABEL, "area", "voronoi").sum() == int((mask != 0).sum())
    assert ov.structure_zkind(D.LABEL, "voronoi") == "plane_index"
    # THE per-region guarantee: a cell never claims a voxel outside its own arena
    for cid, reg in zip(col(ov, D.LABEL, "id", "voronoi"),
                        col(ov, D.LABEL, "region", "voronoi")):
        ys = np.nonzero(rast[0, 0, 0, 0] == cid)[0]
        lo, hi = (4, 20) if reg == 1 else (24, 36)
        assert ys.min() >= lo and ys.max() < hi, (cid, reg, ys.min(), ys.max())
    # density is 1 / area in µm² (2D): 0.2 µm/px ⇒ 0.04 µm² per voxel
    dens = col(ov, D.LABEL, "density", "voronoi")
    area = col(ov, D.LABEL, "area", "voronoi")
    assert np.allclose(dens, 1.0 / (area * 0.04)), (dens.tolist(), area.tolist())
    assert {"pixel_size_um"} <= {k for k, _ in ev.entry("V").reads}

    # ── the other two bounds, and the reach cap ───────────────────────────────────
    def voronoi(bound, extra=None, seed=None, ax=None, dim="2D", chain=True):
        params = {"points": "dots", "region": "areas"}
        params.update(extra or {})
        node = ("V", "analysis.voronoi", {"params": params,
                                          "modes": {"dim": dim, "bound": bound}})
        nodes = [("S", "io.vseed", {})] + ([CC2] if chain else []) + [node]
        edges = [("S", "C"), ("C", "V")] if chain else [("S", "V")]
        return run(nodes, edges, "V", seed if seed is not None else ds2d, ax or ax2)
    _, omask = voronoi("mask", {"region": "mask"}, chain=False)
    rm = np.asarray(omask.get(D.VOXEL, "voronoi").values)
    assert int((rm != 0).sum()) == int((mask != 0).sum())     # one arena, same coverage
    _, ofr = voronoi("frame", chain=False)
    assert int((np.asarray(ofr.get(D.VOXEL, "voronoi").values) != 0).sum()) == 40 * 40
    _, ocap = voronoi("per_region", {"max_distance_um": 1.0})
    n_cap = int((np.asarray(ocap.get(D.VOXEL, "voronoi").values) != 0).sum())
    assert 0 < n_cap < int((mask != 0).sum()), n_cap

    # THE difference between the two bounded modes, on a fixture that separates them: both
    # dots sit in the TOP rectangle, so the bottom one has no seed of its own. per_region
    # leaves it background (a region without a dot gets no territory); mask has one arena, so
    # its voxels go to the nearest dot regardless of which rectangle that dot is in.
    top = _ST(D.POINT, {
        "id": np.arange(2, dtype=np.int64), "m": np.zeros(2, np.int64),
        "t": np.zeros(2, np.int64), "c": np.zeros(2, np.int64), "z": np.zeros(2),
        "y": np.array([10., 10.]), "x": np.array([10., 30.])},
        layer="dots", z_kind="plane_index")
    ds2t = ds2.with_structure(top)
    _, opr = voronoi("per_region", seed=ds2t)
    _, opm = voronoi("mask", {"region": "mask"}, seed=ds2t, chain=False)
    rpr = np.asarray(opr.get(D.VOXEL, "voronoi").values)
    rpm = np.asarray(opm.get(D.VOXEL, "voronoi").values)
    assert int((rpr != 0).sum()) == 512, int((rpr != 0).sum())      # top rectangle only
    assert not rpr[0, 0, 0, 0, 24:36, :].any(), "a region with no dot must stay background"
    assert int((rpm != 0).sum()) == 512 + 384                       # the whole mask
    assert rpm[0, 0, 0, 0, 24:36, :].any(), "mask bound must reach a dot-less rectangle"

    # ── the partition is measured in µm, so an anisotropic z step cannot skew it ──
    # two dots 8 planes apart in a z-tall arena: with z_step 4x the pixel size the plane
    # midway between them must go to the NEARER one in microns, which is the same voxel
    # answer only because both axes are scaled — a pixel-space split would sit elsewhere.
    ax3 = AxisSizes(m=1, t=1, z=12, c=1, y=32, x=32)
    box = np.zeros((1, 1, 12, 1, 32, 32), dtype=np.int64)
    box[0, 0, 2:10, 0, 4:28, 4:28] = 1
    blank3 = np.zeros((1, 1, 12, 1, 32, 32))
    p3 = _ST(D.POINT, {
        "id": np.arange(2, dtype=np.int64), "m": np.zeros(2, np.int64),
        "t": np.zeros(2, np.int64), "c": np.zeros(2, np.int64),
        "z": np.array([6., 6.]), "y": np.array([10., 22.]), "x": np.array([16., 16.])},
        layer="dots", z_kind="subpixel")
    ds3 = (Dataset(axes=ax3, metadata=META).with_image(ArrayProvider(blank3))
           .with_layer(D.VOXEL, "mask", box).with_structure(p3))
    CC3 = ("C", "analysis.label", {"params": {"mask": "mask", "name": "areas"},
                                   "modes": {"dim": "3D"}})
    e3, o3 = run([("S", "io.vseed", {}), CC3,
                  ("V", "analysis.voronoi",
                   {"params": {"points": "dots", "region": "areas"},
                    "modes": {"dim": "3D", "bound": "per_region"}})],
                 [("S", "C"), ("C", "V")], "V", ds3, ax3)
    r3 = np.asarray(o3.get(D.VOXEL, "voronoi").values)
    assert sorted(np.unique(r3).tolist()) == [0, 1, 2]
    assert int((r3 != 0).sum()) == int((box != 0).sum())
    # the two dots differ only in y (10 and 22), so the split is the y midline at every z —
    # row 16 is the exact tie and goes to the lower-index seed, so it is excluded here
    assert set(np.unique(r3[0, 0, 2:10, 0, 4:16, 4:28]).tolist()) == {1}
    assert set(np.unique(r3[0, 0, 2:10, 0, 17:28, 4:28]).tolist()) == {2}
    assert set(np.unique(r3[0, 0, 2:10, 0, 16, 4:28]).tolist()) == {1}
    assert o3.structure_zkind(D.LABEL, "voronoi") == "subpixel"

    # ── 3D cell surfaces: watertight, and they round-trip through the rasterizer ──
    mt = MSH.read_mesh(o3, "voronoi_mesh").validate()
    assert mt.element.n == 2 and mt.face.n > 0
    assert np.all(np.asarray(mt.element.columns["closed"]) == 1), \
        "each cell surface must be watertight (the pad-by-one is what guarantees it)"
    assert np.asarray(mt.element.columns["src_label"]).tolist() == [1, 2]  # == raster ids
    assert np.all(np.asarray(mt.element.columns["volume_um3"]) > 0)
    assert np.all(np.asarray(mt.element.columns["n_points"]) == 1)         # one seed / cell
    assert MSH.mesh_provenance(o3, "voronoi_mesh")["boundary"] == "voronoi_cells"
    _, orr = run([("S", "io.vseed", {}), CC3,
                  ("V", "analysis.voronoi",
                   {"params": {"points": "dots", "region": "areas"},
                    "modes": {"dim": "3D", "bound": "per_region"}}),
                  ("R", "transform.rasterize_mesh",
                   {"params": {"mesh": "voronoi_mesh", "name": "again"}})],
                 [("S", "C"), ("C", "V"), ("V", "R")], "R", ds3, ax3)
    back = np.asarray(orr.get(D.VOXEL, "again").values) != 0
    iou = float((back & (r3 != 0)).sum()) / float((back | (r3 != 0)).sum())
    assert iou > 0.9, f"mesh → raster IoU {iou:.3f}"

    # ── the full headline chain: segmentation → dots → territories ────────────────
    _, och = run([("S", "io.vseed", {}), CC3,
                  ("P", "transform.label_to_points", {"params": {"labels": "areas"}}),
                  ("V", "analysis.voronoi",
                   {"params": {"points": "areas_points", "region": "areas"},
                    "modes": {"dim": "3D", "bound": "per_region", "mesh": "skip"}})],
                 [("S", "C"), ("C", "P"), ("P", "V")], "V", ds3, ax3)
    # one region, one centroid dot ⇒ the region becomes exactly one territory
    assert sorted(np.unique(np.asarray(och.get(D.VOXEL, "voronoi").values)).tolist()) == [0, 1]
    assert col(och, D.LABEL, "area", "voronoi").tolist() == [int((box != 0).sum())]

    # ── mesh: skip really skips it (and the raster is unaffected) ─────────────────
    _, osk = run([("S", "io.vseed", {}), CC3,
                  ("V", "analysis.voronoi",
                   {"params": {"points": "dots", "region": "areas"},
                    "modes": {"dim": "3D", "bound": "per_region", "mesh": "skip"}})],
                 [("S", "C"), ("C", "V")], "V", ds3, ax3)
    assert not [a for a in osk.layers_on(D.MESH)], "mesh=skip must write no Mesh"
    assert np.array_equal(np.asarray(osk.get(D.VOXEL, "voronoi").values), r3)

    # ── nothing to claim: every dot outside every region ⇒ an empty (valid) result ─
    far = _ST(D.POINT, {
        "id": np.zeros(1, np.int64), "m": np.zeros(1, np.int64),
        "t": np.zeros(1, np.int64), "c": np.zeros(1, np.int64),
        "z": np.array([6.]), "y": np.array([31.]), "x": np.array([31.])},
        layer="dots", z_kind="subpixel")
    _, onone = run([("S", "io.vseed", {}), CC3,
                    ("V", "analysis.voronoi",
                     {"params": {"points": "dots", "region": "areas"},
                      "modes": {"dim": "3D", "bound": "per_region"}})],
                   [("S", "C"), ("C", "V")], "V",
                   (Dataset(axes=ax3, metadata=META).with_image(ArrayProvider(blank3))
                    .with_layer(D.VOXEL, "mask", box).with_structure(far)), ax3)
    assert not np.asarray(onone.get(D.VOXEL, "voronoi").values).any()
    assert not [a for a in onone.layers_on(D.LABEL) if a.layer == "voronoi"]
    assert MSH.read_mesh(onone, "voronoi_mesh").validate().element.n == 0, \
        "an empty mesh must still be a VALID mesh, not a raise"

    # ── a stray NEGATIVE value in a raster is junk, not label id 1 ────────────────
    # `searchsorted` would file it under the first id, dragging that region's centroid to
    # wherever the junk voxel is — so the foreground test is `> 0`, never `!= 0`.
    from nodegraph.nodes import _label_centroids as _lc
    junk = np.zeros((6, 6), dtype=np.int64)
    junk[1, 1] = junk[1, 2] = 1
    junk[4, 4] = -7
    cen1, cnt1 = _lc(junk, np.array([1], dtype=np.int64))
    assert cnt1.tolist() == [2] and np.allclose(cen1, [[1.0, 1.5]]), (cen1, cnt1)

    # ── the levers re-key the memo (compared WITHIN one Engine — see tessellate) ──
    ek, _ = run([("S", "io.vseed", {}), CC3,
                 ("V", "analysis.voronoi",
                  {"params": {"points": "dots", "region": "areas"},
                   "modes": {"dim": "3D", "bound": "per_region"}}),
                 ("W", "analysis.voronoi",
                  {"params": {"points": "dots", "region": "areas"},
                   "modes": {"dim": "3D", "bound": "mask"}}),
                 ("X", "analysis.voronoi",
                  {"params": {"points": "dots", "region": "areas"},
                   "modes": {"dim": "2D", "bound": "per_region"}})],
                [("S", "C"), ("C", "V"), ("C", "W"), ("C", "X")], "V", ds3, ax3)
    ek.pull("W")
    ek.pull("X")
    keys = {n: ek.entry(n).recipe_hash for n in ("V", "W", "X")}
    assert len(set(keys.values())) == 3, keys      # bound AND the dim lever both re-key

    # ── guards + per-mode socket / Mode gating ────────────────────────────────────
    try:
        voronoi("per_region", {"points": "nope"})
        raise SystemExit("a missing Point layer should raise")
    except ValueError as exp:
        assert "no Point layer" in str(exp), str(exp)
    try:
        voronoi("mask", {"region": "nothere"}, chain=False)
        raise SystemExit("a missing mask should raise")
    except ValueError as exq:
        assert "to bound the cells" in str(exq), str(exq)
    SPEC = NODES.get("analysis.voronoi")
    vis = lambda st: {i.name for i in SPEC.active_inputs(st)}
    mvis = lambda st: {m.name for m in SPEC.active_modes(st)}
    base3 = {"dim": "3D", "bound": "per_region", "mesh": "build"}
    assert {"points", "region", "max_distance_um", "name", "mesh_name",
            "decimate"} <= vis(base3)
    assert "region" not in vis({**base3, "bound": "frame"})
    assert not ({"mesh_name", "decimate"} & vis({**base3, "mesh": "skip"}))
    assert not ({"mesh_name", "decimate"} & vis({**base3, "dim": "2D"}))
    assert "mesh" in mvis(base3) and "mesh" not in mvis({**base3, "dim": "2D"})
    lp = NODES.get("transform.label_to_points")
    assert {"labels", "name"} <= {i.name for i in lp.active_inputs({"position": "centroid"})}

    _ok("label→points + voronoi cells: label_to_points inherits its dim from the Label "
        "z_kind (no lever), keeps a `label` column + global-unique ids, and its `inside` "
        "position lands inside a C-shaped region whose centroid is background (`weighted` "
        "follows the pixels; `raw` refused on the geometric modes); analysis.voronoi links "
        "dots to areas — per_region arenas provably never leaked and a dot-less region stays "
        "background where `mask` reaches into it, full-coverage partition in µm space on 4:1 "
        "anisotropic voxels, area/density/point_id/region columns, max reach caps it, "
        "3D cell surfaces watertight and "
        "rasterize back at IoU>0.9, mesh=skip skips, dim+bound both re-key; a dot outside "
        "every region yields an empty-but-VALID mesh and a stray negative raster value is "
        "junk rather than id 1; guards and per-mode socket/Mode gating")


def test_engine_observer() -> None:
    """Per-node run observation (2026-07-28): the engine reports start/done/cached/error
    per node and forwards ``ctx.progress`` fractions, without touching results. This is
    what makes a per-node progress bar possible in the GUI."""
    define_node("eng.obs_src", "ObsSrc", outputs=[OutDataset()])
    define_node("eng.obs_work", "ObsWork", inputs=[InDataset()], outputs=[OutDataset()])
    define_node("eng.obs_boom", "ObsBoom", inputs=[InDataset()], outputs=[OutDataset()])

    def c_src(ctx):
        return np.array([1.0])

    def c_work(ctx):
        for i in range(4):                          # an eager per-unit compute
            ctx.progress(i + 1, 4, "unit")
        return np.asarray(ctx.inputs[0]) * 2.0

    def c_boom(ctx):
        raise ValueError("kaboom")

    computes = {"eng.obs_src": c_src, "eng.obs_work": c_work, "eng.obs_boom": c_boom}
    g = Graph()
    g.add(NodeInstance("S", "eng.obs_src"))
    g.add(NodeInstance("W", "eng.obs_work"))
    g.connect("S", "W")

    seen: list = []
    eng = Engine(g, computes=computes,
                 observer=lambda ev, nid, info: seen.append((ev, nid, info)))
    assert np.allclose(eng.pull("W"), 2.0)          # observation never changes results
    kinds = [(ev, nid) for ev, nid, _i in seen]
    assert kinds[0] == ("start", "S") and ("done", "S") in kinds
    assert ("start", "W") in kinds and ("done", "W") in kinds
    # start precedes its own done, and upstream finishes before downstream starts
    assert kinds.index(("done", "S")) < kinds.index(("start", "W"))
    fracs = [i["fraction"] for ev, nid, i in seen if ev == "progress" and nid == "W"]
    assert fracs == [0.25, 0.5, 0.75, 1.0], fracs
    assert all(i["total"] == 4 for ev, _n, i in seen if ev == "progress")
    secs = [i["seconds"] for ev, _n, i in seen if ev == "done"]
    assert len(secs) == 2 and all(s >= 0.0 for s in secs)

    # a second pull is all memo hits → 'cached' per node, no 'start'
    seen.clear()
    eng.pull("W")
    assert [(ev, nid) for ev, nid, _i in seen] == [("cached", "S"), ("cached", "W")]

    # a raising compute is observed as 'error' on the node that raised, and the
    # exception still propagates verbatim
    g2 = Graph()
    g2.add(NodeInstance("S", "eng.obs_src"))
    g2.add(NodeInstance("B", "eng.obs_boom"))
    g2.connect("S", "B")
    seen2: list = []
    eng2 = Engine(g2, computes=computes,
                  observer=lambda ev, nid, info: seen2.append((ev, nid, info)))
    try:
        eng2.pull("B")
        raise SystemExit("the raising compute must propagate")
    except ValueError as exc:
        assert "kaboom" in str(exc)
    errs = [(nid, i.get("error")) for ev, nid, i in seen2 if ev == "error"]
    assert len(errs) == 1 and errs[0][0] == "B" and "kaboom" in errs[0][1]

    # an observer that itself raises must not break the run (a broken progress sink is
    # a UI bug, never a data bug) — and ctx.progress is a no-op without an observer
    def bad(_ev, _nid, _info):
        raise RuntimeError("bad sink")

    g3 = Graph()
    g3.add(NodeInstance("S", "eng.obs_src"))
    g3.add(NodeInstance("W", "eng.obs_work"))
    g3.connect("S", "W")
    assert np.allclose(Engine(g3, computes=computes, observer=bad).pull("W"), 2.0)
    assert np.allclose(Engine(g3, computes=computes).pull("W"), 2.0)

    # ── the TWO-LEVEL split (V2.17): frames + within-the-frame ────────────────
    # This is what the stacked progress bars draw, so the invariants the UI relies on are
    # gated here: the frame level is monotone, the sub level stays in [0,1] and restarts at
    # each frame boundary, the last tick lands BOTH bars full, and a compute that reports no
    # frame count gets no frame fields at all (the UI must not invent an axis).
    define_node("eng.obs_frames", "ObsFrames", inputs=[InDataset()],
                outputs=[OutDataset()])

    def c_frames(ctx):
        for i in range(12):                    # 4 frames × 3 planes
            ctx.progress(i + 1, 12, f"t={i // 3}", frames=4)
        return np.asarray(ctx.inputs[0]) * 2.0

    define_node("eng.obs_solver", "ObsSolver", inputs=[InDataset()],
                outputs=[OutDataset()])

    def c_solver(ctx):
        for t in range(3):                     # a frame loop with an inner iterative solve
            for it in range(5):
                ctx.progress_frame(t, 3, it, 5, note=f"t={t}")
        return np.asarray(ctx.inputs[0]) * 2.0

    computes2 = dict(computes, **{"eng.obs_frames": c_frames,
                                  "eng.obs_solver": c_solver})
    g4 = Graph()
    g4.add(NodeInstance("S", "eng.obs_src"))
    g4.add(NodeInstance("F", "eng.obs_frames"))
    g4.connect("S", "F")
    seen4: list = []
    Engine(g4, computes=computes2,
           observer=lambda ev, nid, i: seen4.append((ev, nid, i))).pull("F")
    ticks = [i for ev, nid, i in seen4 if ev == "progress" and nid == "F"]
    assert len(ticks) == 12
    assert [i["frames_done"] for i in ticks] == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4], \
        [i["frames_done"] for i in ticks]
    assert all(i["frames"] == 4 for i in ticks)
    # the frame IN FLIGHT never runs past the last frame, even on the final tick
    assert [i["frame"] for i in ticks] == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    subs = [round(i["sub_fraction"], 6) for i in ticks]
    assert subs == [round(v, 6) for v in
                    [1 / 3, 2 / 3, 0.0, 1 / 3, 2 / 3, 0.0, 1 / 3, 2 / 3, 0.0,
                     1 / 3, 2 / 3, 1.0]], subs
    # both bars land full together, and the flat fraction is unchanged by any of this
    assert ticks[-1]["frame_fraction"] == 1.0 and ticks[-1]["sub_fraction"] == 1.0
    assert ticks[-1]["fraction"] == 1.0
    prev = -1
    for i in ticks:                            # monotone: a frame bar must never step back
        assert i["frames_done"] >= prev
        prev = i["frames_done"]
        assert 0.0 <= i["sub_fraction"] <= 1.0

    # progress_frame: the sub axis is the SOLVER's, finer than one unit, and the flat
    # fraction is synthesized from the two so the old sinks keep working
    g5 = Graph()
    g5.add(NodeInstance("S", "eng.obs_src"))
    g5.add(NodeInstance("V", "eng.obs_solver"))
    g5.connect("S", "V")
    seen5: list = []
    Engine(g5, computes=computes2,
           observer=lambda ev, nid, i: seen5.append((ev, nid, i))).pull("V")
    st = [i for ev, nid, i in seen5 if ev == "progress" and nid == "V"]
    assert len(st) == 15 and all(i["frames"] == 3 and i["sub_total"] == 5 for i in st)
    assert [i["frame"] for i in st[:5]] == [0] * 5
    assert [round(i["sub_fraction"], 3) for i in st[:5]] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert [i["frames_done"] for i in st] == [0] * 5 + [1] * 5 + [2] * 5
    assert round(st[7]["fraction"], 6) == round(7 / 15, 6)

    # sub_unknown: the frame is known, the work inside it is ONE OPAQUE CALL (a CNN
    # inference with no upstream hook). The sub level must be an explicit None — NOT a
    # missing key (which reads as 0%, "nothing started") and NOT 0.0 — because that is what
    # makes the UI sweep the bar instead of freezing it for the length of the call.
    define_node("eng.obs_opaque", "ObsOpaque", inputs=[InDataset()],
                outputs=[OutDataset()])

    def c_opaque(ctx):
        for i in range(4):                     # 4 frames, one opaque unit each
            ctx.progress(i, 4, f"t={i}", frames=4, sub_unknown=True)   # unit i starts
            ctx.progress(i + 1, 4, f"t={i}", frames=4)                 # unit i landed
        return np.asarray(ctx.inputs[0]) * 2.0

    computes3 = dict(computes2, **{"eng.obs_opaque": c_opaque})
    g6 = Graph()
    g6.add(NodeInstance("S", "eng.obs_src"))
    g6.add(NodeInstance("O", "eng.obs_opaque"))
    g6.connect("S", "O")
    seen7: list = []
    Engine(g6, computes=computes3,
           observer=lambda ev, nid, i: seen7.append((ev, nid, i))).pull("O")
    op = [i for ev, nid, i in seen7 if ev == "progress" and nid == "O"]
    assert len(op) == 8
    starts, lands = op[0::2], op[1::2]
    for i in starts:
        assert "sub_fraction" in i and i["sub_fraction"] is None, i
        assert i["sub_done"] is None and i["sub_total"] is None, i
        assert i["frames"] == 4 and i["frame_fraction"] is not None    # frame axis intact
    for i in lands:
        assert i["sub_fraction"] is not None, i    # a landed unit is determinate again
    # the sweep for unit k must report the frame unit k is IN, not the one before
    assert [i["frame"] for i in starts] == [0, 1, 2, 3], [i["frame"] for i in starts]
    assert [i["frames_done"] for i in starts] == [0, 1, 2, 3]

    # NO frame count reported → no frame fields, so the UI keeps its single flat rail.
    # A fresh Engine (its own memo) so `c_work` actually runs and reports.
    seen6: list = []
    Engine(g3, computes=computes,
           observer=lambda ev, nid, i: seen6.append((ev, nid, i))).pull("W")
    flat = [i for ev, nid, i in seen6 if ev == "progress" and nid == "W"]
    assert len(flat) == 4 and not any("frames" in i or "sub_fraction" in i for i in flat)

    _ok("engine observer: per-node start/done(+seconds)/cached/error + ctx.progress "
        "fractions; two-level frame/sub split monotone, restarts per frame, both bars land "
        "full, absent without a frame count, sub reported as an explicit None (sweep, not "
        "0%) for one opaque call; results untouched, a raising observer or compute both "
        "handled")



# ── V2.12: the central Segmentation node (one contract, the algorithm as a Mode) ──

def _stub_cellsam(planes, *, nocells="", log=None, chunk_errors=()):
    """A fake ``cellSAM`` package, injected into ``sys.modules`` so the kernel's GLUE is
    covered in the fast gate — the real model is a multi-hundred-MB download behind a
    DeepCell API token, so it can never run here, and the glue is where the integration
    risk lives (the mis-shaped no-cells return, the model singleton, kwarg forwarding, the
    contiguous relabel). ``find_spec`` resolves through ``sys.modules``, which is why the
    stub carries a ``__spec__``; ``planes`` records every call so the test can prove the
    checkpoint is read once per pull rather than once per plane."""
    import sys
    import types
    from importlib.machinery import ModuleSpec
    mod = types.ModuleType("cellSAM")
    mod.__spec__ = ModuleSpec("cellSAM", None)

    class _Net:                      # torch is never imported: no .parameters() needed
        # The three inference thresholds real ``CellSAM.__init__`` sets. Present here so the
        # stub models the attribute surface the kernel writes to: ``mask_threshold`` and
        # ``iou_threshold`` have no upstream keyword, so ``segment_plane`` assigns them onto
        # the model and REFUSES a model lacking them (a rename upstream must not degrade
        # into two silently-dead sockets). A bare ``object()`` cannot even take an
        # attribute, which is why it is no longer used as a fake model below.
        mask_threshold = 0.4
        iou_threshold = 0.5
        bbox_threshold = 0.4

        def eval(self):
            return self

        def to(self, dev):
            return self

    def get_model(model="cellsam_general", version=None):
        planes.append(("load", model))
        return _Net()

    def get_local_model(path):
        planes.append(("local", str(path)))
        return _Net()

    def segment_cellular_image(img, model, **kw):
        h, w = np.shape(img)
        # bbox_threshold arrives as a KEYWORD; the other two thresholds arrive on the MODEL
        # (upstream exposes no keyword for them), so read them from there — that is the only
        # place a real ``CellSAM.predict`` would look either.
        planes.append(("seg", float(kw["bbox_threshold"]), bool(kw["normalize"]),
                       float(getattr(model, "mask_threshold", -1)),
                       float(getattr(model, "iou_threshold", -1))))
        if nocells == "attr":
            # The REAL no-cells failure (upstream issue #98): `CellSAM.predict` returns the
            # 4-tuple (None,)*4, so `if preds is None` never fires and upstream calls
            # `fill_holes_and_remove_small_masks(None)`. This is what a blank plane, an
            # empty FOV or the dark end slices of a stack actually hit.
            raise AttributeError("'NoneType' object has no attribute 'ndim'")
        if nocells == "shape":
            # Upstream's own (unreachable) empty branch: `np.zeros(img.shape[1:])` on the
            # (1,3,H,W) TENSOR. Covered so an upstream fix for #98 lands safely.
            return np.zeros((3, h, w), dtype=np.int32), None, None
        lab = np.zeros((h, w), dtype=np.int32)
        lab[1:5, 1:5] = 9                        # ids deliberately non-contiguous
        lab[2, 2] = 0                            # an interior hole
        lab[7:9, 7:9] = 4                        # a 4-px object (for the size filter)
        return lab, None, None

    mod.get_model, mod.get_local_model = get_model, get_local_model
    mod.segment_cellular_image = segment_cellular_image
    mod._Net = _Net                  # so a test can hand `segment_plane` a realistic model
    if log is not None:
        wsi = types.ModuleType("cellSAM.wsi")
        wsi.__spec__ = ModuleSpec("cellSAM.wsi", None)

        def segment_wsi(image, block, overlap, iou_depth, iou_threshold, **kw):
            log.append((int(block), int(overlap), int(iou_depth), float(iou_threshold)))
            # Upstream's `segment_chunk` catches EVERY per-block exception, logs one line
            # and substitutes zeros — it never re-raises, so this is exactly how a failed
            # block presents itself to the kernel: a log record and a hole in the result.
            import logging
            for msg in chunk_errors:
                if str(msg).startswith("@named:"):
                    # the same record via a NAMED child logger — the routing an upstream
                    # refactor would switch to, which a root-LOGGER filter never sees
                    logging.getLogger("cellSAM.wsi").error(
                        "Error segmenting chunk: %s", str(msg)[len("@named:"):])
                else:
                    logging.error("Error segmenting chunk: %s", msg)
            return segment_cellular_image(image, kw.get("model"),
                                          **{k: v for k, v in kw.items() if k != "model"})[0]
        wsi.segment_wsi = segment_wsi
        mod.wsi = wsi
        sys.modules["cellSAM.wsi"] = wsi
    sys.modules["cellSAM"] = mod
    return mod


def test_segment() -> None:
    """``analysis.segment`` — THE segmentation node (V2.12). Every segmentation has the
    same data contract (image in; a Voxel label raster + a Label table out), so the
    algorithm is a ``method`` Mode instead of a node type: ``analysis.watershed`` and
    ``detect.stardist_nuclei`` were folded in and deleted, and CellSAM joined.

    Covers the structural spec (per-method socket gating, the new Mode-level gating, the
    per-dim footprint), a real end-to-end pull of both dependency-free methods in 2D and
    3D, the property that actually distinguishes them (watershed splits a touching pair
    that connected components merges), every shared stage (hole filling, the µm²/µm³ size
    filter, globally-unique ids, the region table), the memo fences, every hard refusal,
    and the CellSAM glue against a stubbed package."""
    import os

    # ── pin the parallel backend for the whole group (2026-07-30) ──────────────
    # This test observes SIDE EFFECTS IN THE PARENT: the CellSAM stub appends to a `calls`
    # list to prove the checkpoint is read once per pull and the thresholds reach the model
    # on every plane. Under the process backend those appends happen in a worker and never
    # come back, so the assertions failed with "one call per (m,t,z,c) plane" — a test that
    # passed or failed on the machine's core count and the NODEGRAPH_PARALLEL default,
    # which is not a property of the code under test.
    #
    # Pinned rather than rewritten: what the stub proves — load-once, per-plane threshold
    # assignment, the two no-cells repairs — is only observable in-process, and a version
    # that asserted on returned pixels alone would stop covering the thing most likely to
    # break. The parallel PATH is exercised by the phase-5 GUI probe and by
    # `test_streaming`, which assert on returned data and so are backend-agnostic.
    _par = os.environ.get("NODEGRAPH_PARALLEL")
    os.environ["NODEGRAPH_PARALLEL"] = "off"
    try:
        _test_segment_body()
    finally:
        if _par is None:
            os.environ.pop("NODEGRAPH_PARALLEL", None)
        else:
            os.environ["NODEGRAPH_PARALLEL"] = _par


def _test_segment_body() -> None:
    """The body of :func:`test_segment`, split out only so the env pin above wraps it."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES

    # ── structural spec (no image deps needed) ─────────────────────────────────
    s = NODES.get("analysis.segment")
    assert s.category == "analysis" and s.label == "Segmentation"
    assert s.reads_domains == frozenset({D.VOXEL})
    assert s.adds_domains == frozenset({D.VOXEL, D.LABEL})   # a raster AND a table
    assert s.meta_transform is None                          # never changes axes/calib
    assert s.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert s.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert s.resolve_kernel_axes({"dim": "3D"}) == frozenset({"z", "y", "x"})
    assert all(not i.is_field for i in s.inputs if i.type is not SocketType.DATASET), \
        "no path in this node evaluates a wired Field — field=True would be a lie"
    mode_of = {m.name: m for m in s.modes}
    assert set(mode_of) == {"dim", "method", "level"}
    assert set(mode_of["method"].choices) == {"threshold", "watershed", "stardist",
                                              "cellsam"}
    assert mode_of["method"].resolved_default() == "threshold", \
        "the dependency-free method must be the default — a fresh node has to just run"
    assert mode_of["level"].resolved_default() == "otsu"
    # method-gated sockets: each method sees ONLY its own controls
    vis = lambda st: {i.name for i in s.active_inputs(st)}
    st2 = {"dim": "2D", "method": "threshold", "level": "otsu"}
    assert "connectivity" in vis(st2) and "mask" not in vis(st2)
    assert "mask" in vis({**st2, "method": "watershed"})
    assert "min_distance" in vis({**st2, "method": "watershed"})
    assert "connectivity" not in vis({**st2, "method": "watershed"}), \
        "watershed takes its regions from the markers, never from a connectivity"
    assert "prob_thresh" in vis({**st2, "method": "stardist"})
    assert "bbox_threshold" in vis({**st2, "method": "cellsam"})
    # `fast` (the batched decoder, 2026-07-31) is CellSAM-only and OFF by default: it is
    # not bit-identical, so a saved graph must keep calling upstream's own function until
    # someone opts in. Its numerics are verified against the real model by
    # `scripts/_bench_cellsam_fast.py` — the stub here cannot say anything about them.
    assert "fast" in vis({**st2, "method": "cellsam"})
    assert not any("fast" in vis({**st2, "method": m})
                   for m in ("threshold", "watershed", "stardist"))
    _fast_sock = NODES.get("analysis.segment").input("fast")
    assert _fast_sock.default is False, "the fast path must be opt-in, not the default"
    assert not (vis({**st2, "method": "stardist"}) & {"bbox_threshold", "cellsam_model"})
    assert not (vis({**st2, "method": "cellsam"}) & {"prob_thresh", "model_name"}), \
        "socket names must stay DISJOINT across methods — the card relayouts on the name " \
        "list, so a same-named socket with another default would not redraw"
    # the fixed level only exists under level=fixed, and only for the cutting methods
    assert "threshold" in vis({**st2, "level": "fixed"})
    assert "threshold" not in vis(st2)
    assert "threshold" not in vis({**st2, "method": "cellsam", "level": "fixed"})
    # a 2D object has an area, a 3D object a volume — never one socket meaning both
    assert {"min_area", "max_area"} <= vis(st2) and "min_volume" not in vis(st2)
    assert "min_volume" in vis({**st2, "dim": "3D"}) and "max_area" not in \
        vis({**st2, "dim": "3D"})
    # ── V2.12 Mode gating: `level` is meaningless to a learned detector, so the whole
    #    dropdown is hidden rather than shown and ignored (ModeSpec.available_in)
    amodes = lambda st: {m.name for m in s.active_modes(st)}
    assert amodes(st2) == {"dim", "method", "level"}
    assert amodes({**st2, "method": "watershed"}) == {"dim", "method", "level"}
    assert amodes({**st2, "method": "cellsam"}) == {"dim", "method"}
    assert amodes({**st2, "method": "stardist"}) == {"dim", "method"}
    # ...and a hidden Mode still carries its value, so the compute and the memo are
    # unaffected by what the GUI draws (the same rule as a hidden socket)
    assert s.default_state()["level"] == "otsu"

    if not _HAVE_WATERSHED:
        _ok("segmentation: spec OK; RUN SKIPPED (scipy/skimage absent)")
        return

    # ── fixture: two separated disks, one TOUCHING pair, and a hole ────────────
    def _disk(a, cy, cx, r, val):
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        a[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val

    Y = X = 32
    plane = np.zeros((Y, X), dtype=float)
    _disk(plane, 7, 7, 4, 100.0)                     # A — 44 px after its hole
    _disk(plane, 7, 24, 4, 100.0)                    # B — 49 px
    _disk(plane, 22, 10, 5, 100.0)                   # C \ overlapping: ONE component
    _disk(plane, 22, 18, 5, 100.0)                   # D / that only a watershed splits
    _hole = np.zeros((Y, X), dtype=bool)
    _disk(_hole, 7, 7, 1, True)
    plane[_hole] = 0.0                               # 5-px hole in the middle of A
    ax = AxisSizes(m=1, t=1, z=3, c=1, y=Y, x=X)
    img = np.zeros((1, 1, 3, 1, Y, X), dtype=float)
    img[0, 0, :, 0] = plane                          # identical on all 3 planes
    optics = {"pixel_size_um": 0.5, "z_step_um": 1.0, "bit_depth": 12}
    seedenv = MetaEnvelope(axes=ax, metadata=optics)
    ds = Dataset(axes=ax, metadata=dict(optics)).with_image(ArrayProvider(img))
    define_node("io.segseed", "Seed", outputs=[OutDataset()])

    def eng(*, modes=None, params=None, chain=(), dset=None, denv=None):
        g = Graph(); g.add(NodeInstance("S", "io.segseed")); prev = "S"
        for i, (cop, cmodes, cparams) in enumerate(chain):
            nid = f"U{i}"
            g.add(NodeInstance(nid, cop, modes=cmodes or {}, params=cparams or {}))
            g.connect(prev, nid); prev = nid
        g.add(NodeInstance("N", "analysis.segment", modes=modes or {},
                           params=params or {}))
        g.connect(prev, "N")
        return Engine(g, computes=COMPUTES, seeds={"S": dset if dset is not None else ds},
                      meta_seeds={"S": denv if denv is not None else seedenv})

    def seg(**kw):
        """(engine, raster, {column: values}) for one pull, output layer 'labels'."""
        e = eng(**kw)
        out = e.pull("N")
        raster = out.get(D.VOXEL, "labels")
        assert raster is not None, "no Voxel label raster"
        cols = {c: (out.get(D.LABEL, c, layer="labels").values
                    if out.get(D.LABEL, c, layer="labels") is not None else None)
                for c in ("id", "area", "z", "y", "x", "m", "t", "c")}
        return e, out, np.asarray(raster.values), cols

    # ── the two dependency-free methods, 2D and 3D ────────────────────────────
    e2, out2, r2, c2 = seg(modes={"dim": "2D", "method": "threshold"})
    assert int(r2.max()) == 9, "3 planes x 3 connected regions (the pair is ONE region)"
    assert len(c2["id"]) == 9 and sorted(c2["id"].tolist()) == list(range(1, 10)), \
        "ids must be globally unique and contiguous across every unit"
    assert sorted(np.unique(r2[r2 > 0]).tolist()) == c2["id"].tolist(), \
        "the raster's ids ARE the table's ids"
    assert sorted(c2["area"].tolist()) == [44, 44, 44, 49, 49, 49, 153, 153, 153], \
        "area is a voxel count, per plane"
    assert sorted(set(c2["z"].tolist())) == [0.0, 1.0, 2.0], "2D z = the plane index"
    assert out2.structure_zkind(D.LABEL, "labels") == "plane_index"
    # the viewer's Labels overlay discovers a raster by DTYPE + rank, not by name
    assert np.issubdtype(r2.dtype, np.integer) and r2.ndim == 6, \
        "the overlay only draws an integer 6-D Voxel layer"

    e3, out3, r3, c3 = seg(modes={"dim": "3D", "method": "threshold"})
    assert int(r3.max()) == 3, "3D labels the VOLUME: the 3 regions are z-connected"
    assert sorted(c3["area"].tolist()) == [132, 147, 459], "3x the per-plane areas"
    assert out3.structure_zkind(D.LABEL, "labels") == "subpixel"
    assert all(0.0 <= z <= 2.0 for z in c3["z"].tolist()), "3D z = a centroid, in range"

    # THE property that distinguishes the two classical methods: the touching pair is one
    # connected component and four disks, so only the watershed recovers the fourth object.
    _, _, rw, cw = seg(modes={"dim": "2D", "method": "watershed"},
                       params={"min_distance": 1.5})
    assert int(rw.max()) == 12, "watershed splits the pair: 3 planes x 4 objects"
    assert sorted(cw["area"].tolist())[:4] == [44, 44, 44, 49]
    _, _, rw3, cw3 = seg(modes={"dim": "3D", "method": "watershed"},
                         params={"min_distance": 1.5})
    assert int(rw3.max()) == 4, "volumetric watershed: 4 z-connected objects"
    # REGRESSION (V2.12): the folded-in `analysis.watershed` numbered EVERY peak pixel
    # returned by peak_local_max as its own marker, so an EDT plateau — the diagonal crest
    # inside the holed disk here — shattered one object into one basin per plateau pixel
    # (this fixture: 11 per plane, some of them 1 voxel). Merging the plateau with FULL
    # connectivity is the fix; these counts are its guard.
    assert int(rw.max()) < 15 and int(cw["area"].min()) > 5, \
        "EDT plateaus must merge into ONE marker, not one marker per plateau pixel"
    # REGRESSION (V2.12): `peak_local_max` excludes a border shell `min_distance` wide on
    # EVERY axis by default — and `min_distance` is a parameter this call does not even use
    # (the physical suppression is the per-axis footprint). With the default left on, a
    # 3D volume of z<=2 has NO interior z plane, so ZERO peaks come back, the single-marker
    # fallback fires, and every object in the volume collapses into one. Two disjoint disks
    # over a 2-plane stack returned 1 object instead of 2. An object against the frame edge
    # was dropped for the same reason.
    thin = np.zeros((1, 1, 2, 1, Y, X), dtype=float)
    thin[0, 0, :, 0] = plane
    thin[0, 0, :, 0, 0:4, 0:4] = 100.0                # a 5th object in the frame CORNER
    thin_ax = AxisSizes(m=1, t=1, z=2, c=1, y=Y, x=X)
    thin_ds = Dataset(axes=thin_ax, metadata=dict(optics)).with_image(ArrayProvider(thin))
    thin_env = MetaEnvelope(axes=thin_ax, metadata=optics)
    _, _, rthin, _ = seg(modes={"dim": "3D", "method": "watershed"},
                         params={"min_distance": 1.5}, dset=thin_ds, denv=thin_env)
    assert int(rthin.max()) == 5, \
        "a z<=2 volume must still yield every object (border exclusion off), got %d" \
        % int(rthin.max())
    _, _, rthin2, _ = seg(modes={"dim": "2D", "method": "watershed"},
                          params={"min_distance": 1.5}, dset=thin_ds, denv=thin_env)
    assert int(rthin2.max()) == 10, "2 planes x 5 objects incl. the frame-corner one"

    # ── the shared stages ─────────────────────────────────────────────────────
    # hole filling closes A's 5-px hole, so A ends up the same size as the intact disk B
    _, _, _, cf = seg(modes={"dim": "2D", "method": "threshold"},
                      params={"fill_holes": True})
    assert sorted(cf["area"].tolist()) == [49] * 6 + [153] * 3, \
        "fill_holes must close the interior hole (44 -> 49) and touch nothing else"
    # the size filter is PHYSICAL: 20 µm² / (0.5 µm)² = 80 px², which keeps only the pair
    _, _, rmin, cmin = seg(modes={"dim": "2D", "method": "threshold"},
                           params={"min_area": 20.0})
    assert cmin["area"].tolist() == [153] * 3, "min_area (µm²) -> px² drops the disks"
    _, _, _, cmax = seg(modes={"dim": "2D", "method": "threshold"},
                        params={"max_area": 20.0})
    assert sorted(cmax["area"].tolist()) == [44, 44, 44, 49, 49, 49], "max_area keeps them"
    # ...and in 3D it is a VOLUME: 40 µm³ / (0.5·0.5·1.0) = 160 voxels
    _, _, _, cvol = seg(modes={"dim": "3D", "method": "threshold"},
                        params={"min_volume": 40.0})
    assert cvol["area"].tolist() == [459], "min_volume (µm³) -> voxels"

    # ── the foreground cut ────────────────────────────────────────────────────
    for lvl in ("otsu", "li", "yen", "triangle", "mean"):
        _, _, rl, _ = seg(modes={"dim": "2D", "method": "threshold", "level": lvl})
        assert int(rl.max()) >= 3, f"level {lvl} found nothing"
    _, _, rfx, _ = seg(modes={"dim": "2D", "method": "threshold", "level": "fixed"},
                       params={"threshold": 50.0})
    assert int(rfx.max()) == 9, "a fixed level in the image's own units"
    # a flat unit must NOT become one giant object (skimage's methods return the constant)
    flat = Dataset(axes=ax, metadata=dict(optics)).with_image(
        ArrayProvider(np.full((1, 1, 3, 1, Y, X), 7.0)))
    _, _, rflat, _ = seg(modes={"dim": "2D", "method": "threshold"}, dset=flat)
    assert int(rflat.max()) == 0, "a blank plane segments to nothing, not to one big blob"
    # ...and a single NON-FINITE voxel must not take the histogram down with it. Untreated,
    # every skimage method raises "autodetected range ... is not finite", and NaN also
    # defeats the flat guard above (nan == nan is False). Deconvolution / normalize /
    # resample can all put one there.
    for tag, bad in (("nan", np.nan), ("inf", np.inf)):
        dirty = img.copy(); dirty[0, 0, 0, 0, 0, 0] = bad
        dds2 = Dataset(axes=ax, metadata=dict(optics)).with_image(ArrayProvider(dirty))
        _, _, rnf, cnf = seg(modes={"dim": "2D", "method": "threshold"}, dset=dds2)
        assert int(rnf.max()) == 9, f"one {tag} voxel must not change the segmentation"
        assert sorted(cnf["area"].tolist())[:3] == [44, 44, 44], \
            f"one {tag} voxel must not join or split an object"
    allnan = Dataset(axes=ax, metadata=dict(optics)).with_image(
        ArrayProvider(np.full((1, 1, 3, 1, Y, X), np.nan)))
    _, _, rnan, _ = seg(modes={"dim": "2D", "method": "threshold"}, dset=allnan)
    assert int(rnan.max()) == 0, "an all-NaN plane segments to nothing"

    # connectivity: two corner-touching squares are 2 objects at 4-conn, 1 at 8-conn
    dax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    dimg = np.zeros((1, 1, 1, 1, 8, 8), dtype=float)
    dimg[0, 0, 0, 0, 1:3, 1:3] = 50.0
    dimg[0, 0, 0, 0, 3:5, 3:5] = 50.0                # touches the first only diagonally
    denv = MetaEnvelope(axes=dax, metadata=optics)
    dds = Dataset(axes=dax, metadata=dict(optics)).with_image(ArrayProvider(dimg))
    for conn, want in ((4, 2), (8, 1)):
        _, _, rc, _ = seg(modes={"dim": "2D", "method": "threshold"},
                          params={"connectivity": conn}, dset=dds, denv=denv)
        assert int(rc.max()) == want, f"{conn}-connectivity should give {want} region(s)"
    # 0 is what a spin box commits when it is touched; it must mean "per-dim default" (8),
    # not reach `_connectivity_rank` and raise on a value the user never chose
    _, _, rc0, _ = seg(modes={"dim": "2D", "method": "threshold"},
                       params={"connectivity": 0}, dset=dds, denv=denv)
    assert int(rc0.max()) == 1, "connectivity=0 must resolve to the 2D default (8)"

    # ── the watershed `mask` socket: split an EXISTING foreground ──────────────
    thr = ("analysis.threshold", {"method": "fixed"}, {"threshold": 50.0, "name": "m2"})
    outm = eng(modes={"dim": "2D", "method": "watershed"},
               params={"mask": "m2", "name": "ws2", "min_distance": 1.5},
               chain=(thr,)).pull("N")          # NOT seg(): the output layer is renamed
    assert outm.get(D.VOXEL, "ws2") is not None, "`mask` + `name` sockets"
    assert outm.get(D.VOXEL, "labels") is None, "no stale default layer"
    assert outm.get(D.LABEL, "area", layer="ws2") is not None, "the table follows `name`"
    assert int(np.asarray(outm.get(D.VOXEL, "ws2").values).max()) == 12, \
        "splitting the upstream mask must match splitting its own cut"

    # ── memo behaviour ────────────────────────────────────────────────────────
    reads2 = dict(e2.entry("N").reads)
    assert "pixel_size_um" in reads2, "the µm² filter must fence on the pixel size"
    assert "z_step_um" in dict(e3.entry("N").reads), "µm³ needs the z step too"
    # every (method, dim, level) must memoize distinctly. Keyed WITHOUT pulling — the
    # engine's `entry()` computes, and asking a hash question must not load a TensorFlow
    # model or trip the cellsam dependency gate. This mirrors engine._entry exactly:
    # the mode state folds into params as `__modes__`, then node_recipe_hash.
    from nodegraph.memo import node_recipe_hash
    hashes = {}
    for meth in ("threshold", "watershed", "stardist", "cellsam"):
        for dim in ("2D", "3D"):
            for lvl in ("otsu", "li"):
                st = {"dim": dim, "method": meth, "level": lvl}
                hashes[(meth, dim, lvl)] = node_recipe_hash(
                    "analysis.segment", {"__modes__": st}, ("up",), (1,))
    assert len(set(hashes.values())) == 16, \
        "method x dim x level must all fold into the recipe hash"
    # ...and prove it end to end on the two methods that can actually run here
    assert (eng(modes={"dim": "2D", "method": "threshold"}).entry("N").recipe_hash
            != eng(modes={"dim": "3D", "method": "threshold"}).entry("N").recipe_hash)
    assert (eng(modes={"dim": "2D", "method": "threshold"}).entry("N").recipe_hash
            != eng(modes={"dim": "2D", "method": "watershed"}).entry("N").recipe_hash)

    # ── refusals ──────────────────────────────────────────────────────────────
    # CellSAM alone now: StarDist gained the 3D lever on 2026-07-30 (StarDist3D predicts
    # z-connected polyhedra, so 3D is real rather than a stack-of-2D fake). CellSAM's decoder
    # is 2-D by construction and still refuses.
    try:
        eng(modes={"dim": "3D", "method": "cellsam"}).pull("N")
        raise AssertionError("cellsam must refuse the 3D lever")
    except ValueError as exc:
        assert "2-D-per-plane" in str(exc) and "2D" in str(exc), \
            "the refusal has to name the fix"
    # StarDist in 3D must NOT be refused — it must get as far as trying to load a 3D model.
    # (TF/stardist weights are a network download, so the fast gate cannot run inference; what
    # is asserted is that the 2D-only guard no longer fires and the 3D socket set is live.)
    _sd3 = {**s.default_state(), "dim": "3D", "method": "stardist"}
    _live3 = {sk.name for sk in s.active_inputs(_sd3)}
    assert {"model_name_3d", "scale_z", "sd_model_path"} <= _live3, \
        f"the 3D StarDist sockets must be live in 3D: {sorted(_live3)}"
    assert "model_name" not in _live3, \
        "the 2D-only model socket must be hidden in 3D (a 2D checkpoint cannot load into 3D)"
    _live2 = {sk.name for sk in s.active_inputs({**_sd3, "dim": "2D"})}
    assert "model_name" in _live2 and not ({"model_name_3d", "scale_z"} & _live2), \
        "and the 3D-only sockets must be hidden in 2D"
    # Checked STRUCTURALLY, deliberately not by pulling. A 3D pull here really does import
    # TensorFlow and download `3D_demo` (measured: it did, the first time this assertion was
    # written) — seconds plus network, which is precisely what this kernel's docs say must
    # stay out of the fast gate. The guard is a frozenset membership test, so reading the
    # frozenset proves the same thing for free.
    from nodegraph.nodes import _SEGMENT_2D_ONLY
    assert "stardist" not in _SEGMENT_2D_ONLY, \
        "StarDist must no longer be refused in 3D (StarDist3D predicts z-connected polyhedra)"
    assert "cellsam" in _SEGMENT_2D_ONLY, \
        "CellSAM's decoder is 2-D by construction — it must still refuse"
    try:
        eng(modes={"dim": "2D", "method": "watershed"},
            params={"mask": "nope"}).pull("N")
        raise AssertionError("a named foreground layer that does not exist must raise")
    except ValueError as exc:
        assert "nope" in str(exc)
    for bad, key in (({"method": "bogus"}, "method"), ({"level": "bogus"}, "level")):
        try:
            eng(modes={"dim": "2D", **bad}).pull("N")
            raise AssertionError(f"an unknown {key} must raise")
        except ValueError as exc:
            assert "bogus" in str(exc)
    # 0 is the size filter's OFF sentinel, so a SUB-VOXEL upper bound would quantize to
    # "no upper limit" and keep everything instead of dropping all but specks — the exact
    # inversion analysis.histogram_threshold already refuses. (A sub-voxel LOWER bound is
    # harmless: every object has at least one voxel, so it filters nothing either way.)
    for dim, key, val in (("2D", "max_area", 0.1), ("3D", "max_volume", 0.1)):
        try:
            eng(modes={"dim": dim, "method": "threshold"}, params={key: val}).pull("N")
            raise AssertionError(f"a sub-voxel {key} must raise, not invert the filter")
        except ValueError as exc:
            assert "under one voxel" in str(exc), str(exc)
    _, _, rsub, _ = seg(modes={"dim": "2D", "method": "threshold"},
                        params={"min_area": 0.1})
    assert int(rsub.max()) == 9, "a sub-voxel min_area is a vacuous filter, not an error"
    try:
        eng(modes={"dim": "2D", "method": "threshold"},
            params={"min_area": 20.0, "max_area": 5.0}).pull("N")
        raise AssertionError("an empty size window must raise")
    except ValueError as exc:
        assert "size window is empty" in str(exc)

    # ── CellSAM: the glue, against a stubbed package ──────────────────────────
    import os
    import sys
    import nodegraph.kernels.cellsam_segment as CS

    # absent (this env): the node must raise the friendly install hint, not a bare
    # ModuleNotFoundError from three frames down.
    if not CS.cellsam_available():
        try:
            eng(modes={"dim": "2D", "method": "cellsam"}).pull("N")
            raise AssertionError("cellsam must refuse when the package is absent")
        except ImportError as exc:
            assert "pip install" in str(exc) and "DEEPCELL_ACCESS_TOKEN" in str(exc), \
                "the hint must name both the package and the model-token requirement"
    assert CS.resolve_device("cpu") == "cpu", "cpu must not even import torch"

    _saved = {k: sys.modules[k] for k in ("cellSAM", "cellSAM.wsi") if k in sys.modules}
    _dev = os.environ.get(CS.DEVICE_ENV)
    os.environ[CS.DEVICE_ENV] = "cpu"                # keep the gate torch-free
    try:
        calls = []
        _stub0 = _stub_cellsam(calls)
        CS._reset_model_cache()
        _, outc, rc2, cc = seg(modes={"dim": "2D", "method": "cellsam"},
                               params={"bbox_threshold": 0.25,
                                       "cellsam_model": "cellsam_extra"})
        assert [c for c in calls if c[0] == "load"] == [("load", "cellsam_extra")], \
            "the checkpoint must be read ONCE per pull, not once per plane"
        assert sum(c[0] == "seg" for c in calls) == 3, "one call per (m,t,z,c) plane"
        assert {c[1] for c in calls if c[0] == "seg"} == {0.25}, "bbox_threshold forwarded"
        # The other two of the paper's three thresholds reach the model, not a keyword, and
        # must be re-asserted on EVERY plane — a singleton set once would let plane 2 inherit
        # plane 1, making a memo hit depend on execution order.
        assert {(c[3], c[4]) for c in calls if c[0] == "seg"} == {(0.4, 0.5)}, \
            "shipped defaults must reach the model on every plane"
        calls.clear(); CS._reset_model_cache()
        _, _, _, _ = seg(modes={"dim": "2D", "method": "cellsam"},
                         params={"mask_threshold": 0.5, "mask_quality": 0.75})
        assert {(c[3], c[4]) for c in calls if c[0] == "seg"} == {(0.5, 0.75)}, \
            "mask_threshold / mask_quality must be assigned onto the model per plane"
        # ...and a model that cannot carry them is refused, not silently ignored: a rename
        # upstream must not degrade into two live sockets that change nothing.
        class _Old:                                  # no mask_threshold / iou_threshold
            def eval(self): return self
        try:
            CS.segment_plane(np.zeros((8, 8), dtype=np.float32), model=_Old())
            raise AssertionError("a model missing the threshold attributes must be refused")
        except AttributeError as exc:
            assert "mask_threshold" in str(exc) and "silently" in str(exc)
        for bad, why in (({"mask_threshold": 0.0}, "a 0 sigmoid cut trips upstream's assert"),
                         ({"mask_threshold": 1.0}, "a 1.0 cut no sigmoid output can clear"),
                         ({"mask_quality": 1.5}, "a score outside [0,1]")):
            try:
                CS.segment_plane(np.zeros((8, 8), dtype=np.float32),
                                 model=_stub0._Net(), **bad)
                raise AssertionError(f"out-of-range must be refused: {why}")
            except ValueError:
                pass
        assert int(rc2.max()) == 6 and sorted(cc["id"].tolist()) == list(range(1, 7)), \
            "3 planes x 2 objects, ids offset into a globally-unique range"
        assert sorted(cc["area"].tolist()) == [4, 4, 4, 15, 15, 15], \
            "upstream's non-contiguous ids (9, 4) relabel to 1..K with the areas intact"
        assert outc.metadata.get("segment_method") == "cellsam"
        assert outc.metadata.get("segment_model") == "cellsam_extra", "provenance stamp"
        # the shared postprocess applies to a learned method exactly as to a classical one
        _, _, _, ccf = seg(modes={"dim": "2D", "method": "cellsam"},
                           params={"fill_holes": True})
        assert sorted(ccf["area"].tolist()) == [4, 4, 4, 16, 16, 16], \
            "fill_holes must close the hole in the model's mask too (15 -> 16)"
        _, _, _, ccm = seg(modes={"dim": "2D", "method": "cellsam"},
                           params={"min_area": 2.0})
        assert sorted(ccm["area"].tolist()) == [15, 15, 15], \
            "2 µm² = 8 px² drops the 4-px object — the physical filter is shared"
        # a local checkpoint bypasses the download+token path, and the provenance must name
        # the checkpoint that actually ran (model_path WINS inside the kernel)
        calls.clear(); CS._reset_model_cache()
        _, outw, _, _ = seg(modes={"dim": "2D", "method": "cellsam"},
                            params={"model_path": "w.pt",
                                    "cellsam_model": "cellsam_extra"})
        assert ("local", "w.pt") in calls, "`model_path` must use get_local_model"
        assert not any(c[0] == "load" for c in calls), "…and never the published model"
        assert outw.metadata.get("segment_model") == "w.pt", \
            "provenance must name the local checkpoint, not the ignored model socket"
        # tiling routes through cellSAM.wsi with iou_depth == overlap (upstream needs <=)
        wsi_log = []
        calls.clear(); _stub_cellsam(calls, log=wsi_log); CS._reset_model_cache()
        seg(modes={"dim": "2D", "method": "cellsam"},
            params={"tile": True, "tile_size": 128, "tile_overlap": 24,
                    "tile_iou": 0.8})
        assert wsi_log and wsi_log[0] == (128, 24, 24, 0.8), \
            "tile_size/overlap/tile_iou forwarded; iou_depth mirrors overlap"

        # ── QUIRK 2c: a FAILED block must not pass as an empty one (2026-07-31) ────
        # `segment_chunk` catches every per-block exception, logs one line and substitutes
        # zeros, so "no cells here" and "this block died and its pixels are gone" arrive
        # identically and the pull succeeds either way. Seen in the wild on a stitched
        # 13106² mosaic: 3 of 196 blocks logged it. The tiled path must hold the same line
        # the untiled one holds two blocks below: absorb the documented no-cells bug,
        # RAISE anything else.
        import logging as _logging

        class _Capture(_logging.Handler):
            def __init__(self):
                super().__init__()
                self.seen = []

            def emit(self, record):
                self.seen.append(record.getMessage())

        _NOCELL = "'NoneType' object has no attribute 'ndim'"
        _tile_p = {"tile": True, "tile_size": 128, "tile_overlap": 24, "tile_iou": 0.8}

        def _tiled(errs):
            cap = _Capture()
            _logging.getLogger().addHandler(cap)
            try:
                calls.clear()
                _stub_cellsam(calls, log=[], chunk_errors=errs)
                CS._reset_model_cache()
                seg(modes={"dim": "2D", "method": "cellsam"}, params=dict(_tile_p))
            finally:
                _logging.getLogger().removeHandler(cap)
            return cap.seen

        # (a) blocks that merely held no cells: the pull SUCCEEDS and the not-an-error
        #     lines never reach a handler (they were `ERROR:root:` on the user's console)
        seen = _tiled([_NOCELL] * 3)
        assert not any("Error segmenting chunk" in m for m in seen), \
            f"a no-cells block is not an ERROR the user should read: {seen}"
        # (b) a genuine failure is REFUSED, and the message says what was lost + why
        for errs, needle in ((["CUDA out of memory"], "CUDA out of memory"),
                             ([_NOCELL, "Killed worker", _NOCELL], "Killed worker")):
            try:
                _tiled(errs)
                raise AssertionError(f"a failed block must not pass as empty: {errs}")
            except RuntimeError as exc:
                assert needle in str(exc), str(exc)[:200]
                assert "no cells" in str(exc), "must distinguish the benign case"
        # (c) and the counts are not conflated: 2 benign + 1 real reports both
        try:
            _tiled([_NOCELL, "Killed worker", _NOCELL])
            raise AssertionError("unreachable")
        except RuntimeError as exc:
            assert "1 block(s) FAILED" in str(exc) and "2 of those" in str(exc), str(exc)
        # (d) the SAME record routed through a NAMED child logger is still caught. A
        #     logger's filters never run on propagated records, so root-logger-only would
        #     mean an upstream switch to `getLogger(__name__)` silently disables the whole
        #     guard — real failures back to passing as empty blocks, with nothing to show
        #     it had stopped working.
        try:
            _tiled(["@named:Killed worker"])
            raise AssertionError("a named-logger chunk error must not escape the watch")
        except RuntimeError as exc:
            assert "Killed worker" in str(exc), str(exc)[:200]
        seen = _tiled(["@named:" + _NOCELL] * 2)
        assert not any("Error segmenting chunk" in m for m in seen), \
            f"a named-logger no-cells line must be suppressed too: {seen}"
        # Both upstream no-cells failures must land as an EMPTY segmentation, not a crash.
        # This is the path a blank plane / an empty FOV / the dark end slices of a z-stack
        # take on every real run, so it is the difference between "the stack segments" and
        # "the whole pull dies on slice 0".
        for mode, why in (("attr", "the real (None,)*4 AttributeError, upstream #98"),
                          ("shape", "upstream's own mis-shaped (C,H,W) empty branch")):
            calls.clear(); _stub_cellsam(calls, nocells=mode); CS._reset_model_cache()
            _, outn, rn, cn = seg(modes={"dim": "2D", "method": "cellsam"})
            assert int(rn.max()) == 0 and rn.shape == (1, 1, 3, 1, Y, X), why
            assert cn["id"] is None or len(cn["id"]) == 0, f"rows emitted for {why}"
        # ...but an unrelated AttributeError must still propagate — the absorb is narrow
        calls.clear(); _stub = _stub_cellsam(calls); CS._reset_model_cache()
        _stub.segment_cellular_image = lambda *a, **k: (_ for _ in ()).throw(
            AttributeError("'Foo' object has no attribute 'bar'"))
        try:
            seg(modes={"dim": "2D", "method": "cellsam"})
            raise AssertionError("an unrelated AttributeError must not be swallowed")
        except AttributeError as exc:
            assert "Foo" in str(exc)
        # QUIRK 5 — with `postprocess=True` upstream calls four skimage functions deprecated
        # in 0.26 ONCE PER CELL per plane, so a 200-cell series emits thousands of identical
        # FutureWarnings and buries every real message. The kernel must swallow exactly
        # those four and stay out of the way of anything else upstream says.
        calls.clear(); _stub = _stub_cellsam(calls); CS._reset_model_cache()
        _inner = _stub.segment_cellular_image

        def _noisy(img, model, **kw):
            for fn in ("binary_opening", "binary_closing", "binary_dilation",
                       "binary_erosion"):
                warnings.warn(f"`{fn}` is deprecated since version 0.26 and will be "
                              f"removed in version 0.28. Use `skimage.morphology` instead.",
                              FutureWarning)
            warnings.warn("cellSAM has something the user must see", FutureWarning)
            return _inner(img, model, **kw)
        _stub.segment_cellular_image = _noisy
        with warnings.catch_warnings(record=True) as _seen:
            warnings.simplefilter("always")
            CS.segment_plane(np.zeros((16, 16), dtype=np.float32), model=_stub._Net(),
                             postprocess=True)
        _msgs = [str(w.message) for w in _seen]
        assert not [m for m in _msgs if "binary_" in m], \
            f"the per-cell skimage deprecation spam must be filtered out: {_msgs}"
        assert any("must see" in m for m in _msgs), \
            "the filter must stay narrow — every OTHER upstream warning still surfaces"
    finally:
        for k in ("cellSAM", "cellSAM.wsi"):
            sys.modules.pop(k, None)
        sys.modules.update(_saved)
        if _dev is None:
            os.environ.pop(CS.DEVICE_ENV, None)
        else:
            os.environ[CS.DEVICE_ENV] = _dev
        CS._reset_model_cache()

    # StarDist is NOT run here, by the same policy as before the fold-in: loading the
    # pretrained TF model costs seconds and hits the network, which does not belong in the
    # fast gate. Its socket set and gating are covered structurally above, and the compute
    # is audited by test_param_socket_contract.
    _ok("segmentation: one image->Label contract, 4 methods behind a Mode (threshold/"
        "watershed 2D+3D run; stardist spec-only; cellsam glue on a stub, incl. BOTH "
        "upstream no-cells failures absorbed to an empty plane + an unrelated one still "
        "raised, and the per-cell deprecated-skimage spam filtered while other upstream "
        "warnings survive) — the TILED path holds that same line through the only evidence "
        "upstream's per-block `except Exception` leaves behind, its log record: a no-cells "
        "block is counted and its not-an-ERROR line suppressed, while a genuinely failed "
        "block (which upstream zeroes, so it would pass as 'no cells here') is REFUSED with "
        "both counts named — caught whether upstream logs to the root logger or to a named "
        "child one — per-method socket + Mode gating, watershed splits what CCL "
        "merges "
        "(plateau markers merged, border exclusion off so z<=2 and frame-edge objects "
        "survive), "
        "shared fill/µm²+µm³ filter/global ids/table, px+z fences, 16 distinct "
        "method×dim×level hashes, 5 refusals (3D lever on a 2D-only method, missing "
        "foreground layer, unknown method/level, sub-voxel + empty size window)")


def test_celltracker_parity() -> None:
    """The Cell-Tracker parity ports (V2.13) end-to-end through the ``Engine``.

    Covers the five nodes that ported Cell-Tracker's remaining processing —
    ``enhance.flatten_field``, ``enhance.temporal_gain``, ``enhance.remove_blobs``,
    ``analysis.object_metrics``, ``analysis.object_field`` — plus the four consistency
    repairs made to already-ported nodes in the same pass (``enhance.gamma``'s ``scale``
    Mode, ``enhance.dog``'s ``rescale``, ``enhance.clahe``'s ``tile_grid``,
    ``enhance.morphological_gradient``'s ``blend``/``rescale``, and ``analysis.measure``'s
    ``shape`` metrics).

    What is asserted, beyond "it runs":

    * **Numbers, against closed forms.** A synthetic series carries a lateral illumination
      ramp, an exponential bleach, one 1.6x aberrant frame and one hot pixel, so each node
      has something it must actually fix. Velocities are checked against the exact µm/s the
      calibration implies, γ against Cell-Tracker's own formula on the declared 12-bit
      scale, fold changes against hand-computed ratios.
    * **The two references genuinely differ.** ``series_mean`` flattens every frame
      including the aberrant one; ``exponential_fit`` flattens only the smooth decay and
      LEAVES the outlier standing. That divergence is the reason both exist, so it is
      pinned rather than left to a "runs clean" check.
    * **Envelope == payload** for the one mode-dependent ``meta_transform`` in the catalog:
      ``flatten_field`` drops ``bit_depth`` for ``ratio`` only, and the pulled Dataset must
      agree with ``engine.env`` in all four methods (§7c / build-node-v2 §2).
    * **Memo fences.** ``bit_depth`` is recorded by ``full_range`` γ and NOT by
      ``plane_max``, which is the R1 rule — a node must not be invalidated by a key its
      selected path cannot read.
    * **Distinct recipe hashes** per method x reference, so the modes really fold into the key.
    * **Refusals**, not silent degradation: T=1 temporal gain, a 3D (``subpixel``) table on
      either analysis node, a missing ``track_id``, and a 2D-only shape metric on a 3D
      Label table.
    * **The composition** that makes ``object_field`` worth having: it chains into
      ``transform.rasterize_field``, which inherits its ``plane_index`` z_kind (§7b).
    """
    if not _HAVE_SKIMAGE:
        _ok("Cell-Tracker parity: SKIPPED (scikit-image absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import (
        COMPUTES, _exp_fit_t, _rolling_mean_t, _temporal_gain_field, _tile_means,
    )
    from nodegraph.structure import StructureTable as _ST

    # ── 0. the temporal kernels, against the claims their docstrings make ─────
    # (a) _rolling_mean_t IS Cell-Tracker's pad(edge) + convolve("same") + slice-back
    #     sandwich for an odd window. Asserted rather than asserted-in-prose, because the
    #     whole reason the scipy call is used instead is that they are identical.
    sig = np.array([10.0, 12.0, 40.0, 11.0, 9.0, 8.0, 13.0])
    for w in (3, 5, 11):
        k, n_ = w // 2, len(sig)
        ct = np.convolve(np.pad(sig, k, mode="edge"), np.ones(w) / w,
                         mode="same")[k:k + n_]
        assert np.allclose(_rolling_mean_t(sig, w), ct), (w, _rolling_mean_t(sig, w), ct)
    # (b) the log-space least squares recovers a known exponential exactly — this is the
    #     decay model Cell-Tracker's Bleach Correction offered and never implemented
    decay = 500.0 * np.exp(-0.2 * np.arange(10))
    assert np.allclose(_exp_fit_t(decay, 1e-9), decay, rtol=1e-6), _exp_fit_t(decay, 1e-9)
    # (c) a float32 signal stays float32 all the way through: the `gaussian` extent's
    #     signal is a whole (T,Y,X) series, so a silent widening triples peak memory
    s32 = (decay[:, None, None] * np.ones((1, 4, 5))).astype(np.float32)
    for ref in ("series_mean", "rolling_mean", "exponential_fit"):
        assert _temporal_gain_field(s32, ref, 3, 0.0).dtype == np.float32, ref
        assert _temporal_gain_field(decay, ref, 3, 0.0).dtype == np.float64, ref
    # (d) tile means average a RAGGED trailing block over the pixels it has, not a
    #     zero-padded block (padding would drag the frame border down and over-correct it)
    ramp = np.tile(np.arange(5.0), (4, 1))          # 4x5, tile 2 ⇒ last column block is 1 wide
    tm = _tile_means(ramp, 2)
    assert tm.shape == (2, 3), tm.shape
    assert np.allclose(tm[:, -1], 4.0), tm         # the 1-wide edge block is exactly x=4

    define_node("io.ctseed", "S", outputs=[OutDataset()])

    # ── the fixture: a bleaching, unevenly-lit series with one bad frame + one speck ──
    T, Y, X = 6, 40, 48
    yy, xx = np.mgrid[0:Y, 0:X]

    def _blob(cy, cx, amp, sig):
        return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig * sig)))

    # smooth, EXTENDED cells (σ = 4 px = 2 µm). Deliberately not sharp squares: a hard-edged
    # square reads as a small blob to a LoG, which would make the blob-removal check a test
    # of the fixture rather than of the node.
    obj = _blob(13, 15, 400.0, 4.0) + _blob(29, 33, 300.0, 4.0)
    shade = 1.0 + 0.8 * (xx / X)                       # lateral illumination ramp
    img = np.zeros((1, T, 1, 1, Y, X))
    for t in range(T):
        img[0, t, 0, 0] = (200.0 + obj) * shade * (0.85 ** t)      # exponential bleach
    img[0, 2, 0, 0] *= 1.6                                          # one aberrant frame
    img[0, 3, 0, 0, 5, 5] = 5000.0                                  # a hot pixel

    ax = AxisSizes(m=1, t=T, z=1, c=1, y=Y, x=X)
    meta = {"pixel_size_um": 0.5, "dt_s": 60.0, "bit_depth": 12,
            "channel_emission_nm": [520.0], "objective_na": 1.4}
    base_ds = Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
    env = MetaEnvelope(axes=ax, metadata=meta)

    def run(nodes, edges, sink, seed=None, seed_env=None):
        g = Graph()
        for nid, op, kw in nodes:
            g.add(NodeInstance(nid, op, **kw))
        for a, b in edges:
            g.connect(a, b)
        e = Engine(g, computes=COMPUTES, seeds={"S": seed or base_ds},
                   meta_seeds={"S": seed_env or env})
        return e, e.pull(sink)

    def plane(out, t=0):
        a = out.axes
        return out.image.get_region(0, 0, t, 0, 0, 0, a.y, 0, a.x)

    def frame_means(out):
        a = out.axes
        return np.array([float(out.image.get_region(0, 0, t, 0, 0, 0, a.y, 0, a.x).mean())
                         for t in range(a.t)])

    _, raw = run([("S", "io.ctseed", {})], [], "S")
    src0 = plane(raw, 0)

    # ── 1. flatten_field: four methods x two references ───────────────────────
    hashes = set()
    for method in ("subtract", "subtract_mean", "divide_mean", "ratio"):
        for ref in ("per_plane", "time_averaged"):
            e, out = run([("S", "io.ctseed", {}),
                          ("F", "enhance.flatten_field",
                           {"modes": {"method": method, "reference": ref},
                            "params": {"sigma": 8.0}})], [("S", "F")], "F")
            got = plane(out, 1)
            assert np.all(np.isfinite(got)), (method, ref, "non-finite output")
            assert got.shape == (Y, X), got.shape
            # envelope prediction == payload, for every method (§7c lockstep)
            predicted = e.env("F").metadata.get("bit_depth")
            assert predicted == out.metadata.get("bit_depth"), \
                f"flatten_field/{method}: envelope bit_depth {predicted} != payload " \
                f"{out.metadata.get('bit_depth')}"
            assert (predicted is None) == (method == "ratio"), \
                f"flatten_field/{method}: only `ratio` leaves the count scale"
            hashes.add(e.entry("F").recipe_hash)
            if method == "subtract":
                assert got.min() >= 0.0, "subtract must clip at 0"
    assert len(hashes) == 8, f"method x reference must re-key: {len(hashes)} hashes"

    def _imbalance(a):                    # left/right brightness split the ramp created
        h = a.shape[1] // 2
        return abs(a[:, h:].mean() - a[:, :h].mean()) / a.mean()

    _, flat = run([("S", "io.ctseed", {}),
                   ("F", "enhance.flatten_field",
                    {"modes": {"method": "divide_mean"},
                     "params": {"sigma": 8.0}})], [("S", "F")], "F")
    ib_raw, ib_flat = _imbalance(plane(raw, 1)), _imbalance(plane(flat, 1))
    assert ib_flat < ib_raw / 2, f"illumination not flattened: {ib_raw:.3f} -> {ib_flat:.3f}"

    # ── 2. temporal_gain: 3 references x 3 extents, and the two must DIFFER ───
    base_fm = frame_means(raw)
    assert base_fm.std() / base_fm.mean() > 0.3, "fixture must actually drift"
    for ref in ("series_mean", "rolling_mean", "exponential_fit"):
        for ext in ("global", "tile", "gaussian"):
            _, out = run([("S", "io.ctseed", {}),
                          ("G", "enhance.temporal_gain",
                           {"modes": {"reference": ref, "extent": ext},
                            "params": {"tile_size": 8.0, "local_sigma": 4.0,
                                       "window": 3, "threshold": 0.05}})],
                         [("S", "G")], "G")
            fm = frame_means(out)
            assert np.all(np.isfinite(fm)), (ref, ext)
            if ext != "global":
                continue
            if ref == "series_mean":
                # each frame is divided by its OWN mean ⇒ every frame lands on one level,
                # the aberrant one included (so this reference also erases it)
                assert fm.std() / fm.mean() < 0.05, (ref, fm)
            if ref == "exponential_fit":
                # a least-squares fit does NOT chase an outlier: the 5 clean frames
                # flatten and the 1.6x frame deliberately survives. This is the whole
                # difference between the two references, and Cell-Tracker's UI offered
                # this mode without ever implementing it.
                clean = fm[[0, 1, 3, 4, 5]]
                assert clean.std() / clean.mean() < 0.05, (ref, clean)
                assert fm[2] > 1.4 * clean.mean(), (fm[2], clean.mean())
    # ── 2b. rolling_mean IS Cell-Tracker's Temporal Fold Correction, end to end ──
    # The kernel check in (a) only pinned the rolling mean. A LOCAL extent is where a port
    # actually diverges — tile seams, ragged trailing blocks, and which array the blur is
    # applied to — so all three extents are scored against Cell-Tracker's own
    # `_correct_global` / `_correct_local` / `_correct_local_gaussian`, transcribed here
    # from CellTracker/plugins/enhancement/builtin.py with only the trailing
    # `clip(0, 65535).astype(volume.dtype)` dropped, so this compares the ARITHMETIC and
    # not a uint16 round-trip. `scripts/_temporal_gain_parity.py` is the standalone bench.
    # `threshold` is 0.02, NOT the node default 0.1 or the 0.05 used above, and the choice
    # is load-bearing. The dead band makes this node DISCONTINUOUS in its input: a sample
    # whose deviation sits exactly on the threshold flips between "gain applied" and "gain
    # exactly 1", so the last bit of the measured brightness decides a full gain step — and
    # the two implementations necessarily differ in that last bit, because CT means each
    # tile with `.mean()` while `_tile_means` block-sums with `np.add.reduceat`. This
    # fixture bleaches by a clean 0.85^t, so at t=0 the edge-padded w=3 rolling mean is
    # (2 + 0.85)/3 = 0.95 EXACTLY and the deviation is exactly 0.05 — right on the boundary
    # for the 0.05 used above, where the two sides legitimately disagree by 5% on whichever
    # tiles round the other way. That is a property of Cell-Tracker's own design, not a
    # port defect, so the parity check is run off the boundary and `ct_margin` guards it.
    ct_w, ct_thr = 3, 0.02
    ct_tile_um, ct_sigma_um = 8.0, 4.0                      # µm; 0.5 µm/px ⇒ 16 px, 8 px
    ct_tile, ct_sigma = 16, 8.0
    vol_in = img[0, :, 0, 0]                                # (T, Y, X) raw
    ct_margin = [np.inf]                                    # closest any sample came to it

    def _ct_ratios(means):                                  # CT's pad+convolve sandwich
        k = ct_w // 2
        rolling = np.convolve(np.pad(means, k, mode="edge"), np.ones(ct_w) / ct_w,
                              mode="same")[k:k + T]
        r = rolling / np.clip(means, 1e-10, None)
        ct_margin[0] = min(ct_margin[0],
                           float(np.abs(np.abs(r - 1.0) - ct_thr).min()))
        return r

    def _ct_global(vol):
        out = vol.astype(float).copy()
        r = _ct_ratios(np.array([float(vol[t].mean()) for t in range(T)]))
        for t in range(T):
            if abs(r[t] - 1.0) > ct_thr:
                out[t] = vol[t].astype(np.float32) * r[t]
        return out

    def _ct_local(vol):
        out = vol.astype(float).copy()
        for y0 in range(0, Y, ct_tile):
            for x0 in range(0, X, ct_tile):
                y1, x1 = min(y0 + ct_tile, Y), min(x0 + ct_tile, X)
                sub = vol[:, y0:y1, x0:x1]
                r = _ct_ratios(np.array([float(sub[t].mean()) for t in range(T)]))
                for t in range(T):
                    if abs(r[t] - 1.0) > ct_thr:
                        out[t, y0:y1, x0:x1] = sub[t].astype(np.float32) * r[t]
        return out

    def _ct_local_gaussian(vol):
        from scipy.ndimage import gaussian_filter as _gf
        out = vol.astype(float).copy()
        blurred = np.stack([_gf(vol[t].astype(np.float32), sigma=ct_sigma)
                            for t in range(T)])
        for y in range(Y):
            for x in range(X):
                r = _ct_ratios(blurred[:, y, x])
                for t in range(T):
                    if abs(r[t] - 1.0) > ct_thr:
                        out[t, y, x] = np.float32(vol[t, y, x]) * r[t]
        return out

    def _stack(out):
        return np.stack([plane(out, t) for t in range(T)])

    for ext, ct_fn in (("global", _ct_global), ("tile", _ct_local),
                       ("gaussian", _ct_local_gaussian)):
        _, out = run([("S", "io.ctseed", {}),
                      ("G", "enhance.temporal_gain",
                       {"modes": {"reference": "rolling_mean", "extent": ext},
                        "params": {"window": ct_w, "threshold": ct_thr,
                                   "tile_size": ct_tile_um,
                                   "local_sigma": ct_sigma_um}})], [("S", "G")], "G")
        got, want = _stack(out), ct_fn(vol_in)
        # rtol 1e-6 is CT's own float32 multiply, not slack: the observed gap is ~1e-7
        # relative. The `gaussian` extent matches only because BOTH sides blur with
        # scipy — CT itself used cv2, whose BORDER_REFLECT_101 differs from scipy's
        # reflect within ~4σ of the frame edge (documented on the node; the catalog is
        # scipy throughout, so the node keeps scipy).
        assert np.allclose(got, want, rtol=1e-6, atol=1e-6), (
            f"temporal_gain[rolling_mean/{ext}] diverges from Cell-Tracker's "
            f"Temporal Fold Correction: max |diff| "
            f"{np.abs(got - want).max():.4e} on a {np.abs(want).max():.1f} peak")
    # and the fixture must be one the parity check can actually fail on: the tile extent
    # has ragged trailing blocks, and the local extents must DIFFER from the global one
    assert Y % ct_tile and (Y // ct_tile) >= 2, (Y, ct_tile)   # ragged, >1 block
    assert not np.allclose(_ct_local(vol_in), _ct_global(vol_in)), \
        "fixture has no spatial structure — the local extents would pass trivially"
    # …and no sample may sit ON the dead-band boundary, or the comparison above is a coin
    # flip on float rounding rather than a parity check (see the note on `ct_thr`).
    # The bar is 1e-12, not "comfortably far": the `gaussian` extent has one deviation PER
    # PIXEL (T·Y·X ≈ 11k here), so by sheer density something always lands within ~1e-6 of
    # any threshold and parity still holds — a NEAR-boundary sample does not flip. A flip
    # needs the deviation within the two implementations' actual disagreement about the
    # ratio, which is ULP-level (~1e-16 relative) because they differ only in summation
    # order. The 0.05 threshold this check originally used sat 4e-17 from the fixture's
    # analytic 0.95 gain, and that is the case worth catching.
    assert ct_margin[0] > 1e-12, (
        f"temporal_gain parity fixture is degenerate: some sample's fold deviation comes "
        f"within {ct_margin[0]:.2e} of threshold {ct_thr} — at that distance the dead "
        f"band's strict `>` is decided by the last bit of the measured mean, and CT's "
        f"per-tile `.mean()` and `_tile_means`' `add.reduceat` legitimately disagree there "
        f"by a WHOLE gain step. Move `ct_thr` off the boundary.")

    # T=1 is refused rather than silently passed through as a gain of 1
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=Y, x=X)
    ds1 = Dataset(axes=ax1, metadata=meta).with_image(ArrayProvider(img[:, :1]))
    try:
        run([("S", "io.ctseed", {}), ("G", "enhance.temporal_gain", {})], [("S", "G")],
            "G", seed=ds1, seed_env=MetaEnvelope(axes=ax1, metadata=meta))
        raise AssertionError("temporal gain accepted T=1")
    except ValueError as exc:
        assert "single timepoint" in str(exc), exc

    # ── 3. remove_blobs: the speck goes, the cells stay ───────────────────────
    before = plane(raw, 3)
    for action in ("zero", "interpolate", "median"):
        _, out = run([("S", "io.ctseed", {}),
                      ("B", "enhance.remove_blobs",
                       {"modes": {"action": action},
                        # max 1 µm = 2 px ⇒ σ ≤ 1.41, while the cells sit near σ 4 and are
                        # therefore outside the band
                        "params": {"min_radius": 0.5, "max_radius": 1.0,
                                   "threshold": 0.02, "fill_radius": 2.0}})],
                     [("S", "B")], "B")
        got = plane(out, 3)
        assert np.all(np.isfinite(got)), action
        assert got[5, 5] < before[5, 5] / 5, (action, "speck survived", got[5, 5])
        # untouched pixel: exact under float64 streaming, float32-epsilon under the
        # NODEGRAPH_FLOAT32 opt-in (where "unchanged" can only mean unchanged to float32)
        assert np.isclose(got[13, 15], before[13, 15],
                          rtol=_stream_rtol(0.0), atol=0.0), \
            (action, "a real cell was eaten")
        if action == "zero":
            assert got[5, 5] == 0.0, got[5, 5]

    # ── 4/5. object_metrics + object_field on a tracked Label table ───────────
    # two objects, each drifting +2 px in x per frame; intensities 100 and 200
    n = 2 * T
    tbl = _ST(D.LABEL, {
        "id": np.arange(1, n + 1, dtype=np.int64),
        "m": np.zeros(n, np.int64),
        "t": np.repeat(np.arange(T, dtype=np.int64), 2),
        "c": np.zeros(n, np.int64),
        "z": np.zeros(n),
        "y": np.tile([10.0, 30.0], T),
        "x": np.repeat(np.arange(T, dtype=float) * 2.0, 2) + np.tile([10.0, 30.0], T),
        "area": np.full(n, 36.0),
        "mean_intensity": np.tile([100.0, 200.0], T),
        "track_id": np.tile([1, 2], T).astype(np.int64),
    }, layer="labels", z_kind="plane_index")
    raster = np.zeros((1, T, 1, 1, Y, X), dtype=np.int64)
    ds_lab = (Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
              .with_layer(D.VOXEL, "labels", raster).with_structure(tbl))

    e, out = run([("S", "io.ctseed", {}),
                  ("M", "analysis.object_metrics",
                   {"params": {"metrics": "velocity,speed,neighbors,divergence,curl,"
                                          "frame_fold,self_fold", "n_neighbors": 1}})],
                 [("S", "M")], "M", seed=ds_lab)
    col = {c: out.get(D.LABEL, c, layer="labels").values
           for c in ("vy", "vx", "speed", "neighbor_dist_mean", "neighbor_dist_std",
                     "local_divergence", "local_curl", "frame_fold", "self_fold")}
    v_expect = 2.0 * 0.5 / 60.0                       # 2 px/frame · 0.5 µm/px ÷ 60 s
    assert np.isnan(col["vx"][0]) and np.isnan(col["vx"][1]), \
        "a track's first detection must be NaN, not 0"
    assert np.allclose(col["vx"][2:], v_expect), col["vx"]
    assert np.allclose(col["vy"][2:], 0.0), col["vy"]
    assert np.allclose(col["speed"][2:], v_expect), col["speed"]
    # the two objects are 20 px apart in both y and x ⇒ hypot(20,20)·0.5 µm
    assert np.allclose(col["neighbor_dist_mean"], np.hypot(20.0, 20.0) * 0.5), \
        col["neighbor_dist_mean"][:2]
    # CT skips a frame below 3 objects; this port's floor is 2, so a 2-object frame still
    # reports a well-defined nearest-neighbour distance (the documented loosening)
    assert np.all(np.isfinite(col["neighbor_dist_mean"])), "2-object frame blanked"
    # both objects move identically ⇒ zero RELATIVE velocity; with 1 neighbour the
    # gradient measures fall below their own >=2-neighbour guard and stay NaN
    assert np.all(np.isnan(col["local_divergence"])), col["local_divergence"][:2]
    assert np.all(np.isnan(col["local_curl"])), col["local_curl"][:2]
    # 100 and 200 against a frame mean of 150; constant per track ⇒ self_fold exactly 1
    assert np.allclose(col["frame_fold"], np.tile([100 / 150, 200 / 150], T)), \
        col["frame_fold"][:2]
    assert np.allclose(col["self_fold"], 1.0), col["self_fold"][:4]
    reads = {k for k, _ in e.entry("M").reads}
    assert {"pixel_size_um", "dt_s"} <= reads, sorted(reads)   # µm/s fenced on both

    e, out = run([("S", "io.ctseed", {}),
                  ("F", "analysis.object_field",
                   {"params": {"fields": "density,mean_area,intensity,fold_change,"
                                         "velocity,speed,divergence,curl",
                               "grid_step": 5.0, "smooth": 5.0}})],
                 [("S", "F")], "F", seed=ds_lab)
    fld = {a.name: a.values for a in out.layers_on(D.POINT) if a.layer == "object_field"}
    for need in ("id", "m", "t", "c", "z", "y", "x",          # invariant Point schema
                 "density", "mean_area", "intensity", "fold_change",
                 "velocity_y", "velocity_x", "speed", "divergence", "curl"):
        assert need in fld, f"object_field missing column {need!r}"
        assert len(fld[need]) == len(fld["id"]), need
    assert out.structure_zkind(D.POINT, "object_field") == "plane_index"
    assert np.unique(fld["id"]).size == fld["id"].size, "grid node ids must be unique"
    peak = float(np.nanmax(fld["speed"]))
    assert abs(peak - v_expect) < 1e-3, peak          # the gridded flow is the real flow
    assert float(np.nanmax(fld["density"])) > 0.0
    # composes with the rasterizer, which inherits the grid's plane_index z_kind (§7b)
    _, ras = run([("S", "io.ctseed", {}),
                  ("F", "analysis.object_field",
                   {"params": {"fields": "density", "grid_step": 5.0}}),
                  ("R", "transform.rasterize_field",
                   {"params": {"source": "object_field"}})],
                 [("S", "F"), ("F", "R")], "R", seed=ds_lab)
    lay = ras.get(D.VOXEL, "object_field_density")
    assert lay is not None and lay.values.shape == (1, T, 1, 1, Y, X), lay

    # a 2D table must NOT carry a vz column: an all-NaN axial column on every 2D
    # workflow is noise, so vz appears only where it is actually measured
    assert out.get(D.LABEL, "vz", layer="labels") is None, \
        "vz must not be written for a plane_index table"

    # ── 3D (subpixel) member table: per-METRIC refusal, not a blanket one ──────
    # object_field stays 2D-only (its whole output is a grid of in-plane estimators);
    # object_metrics now measures a volume and refuses only `divergence`/`curl`.
    ds_3d = (Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
             .with_layer(D.VOXEL, "labels", raster)
             .with_structure(_ST(D.LABEL, dict(tbl.columns), layer="labels",
                                 z_kind="subpixel")))
    try:
        run([("S", "io.ctseed", {}), ("N", "analysis.object_field", {})],
            [("S", "N")], "N", seed=ds_3d)
        raise AssertionError("analysis.object_field accepted a 3D (subpixel) table")
    except ValueError as exc:
        assert "2D-only" in str(exc), exc
    for planar in ("divergence", "curl"):
        try:
            run([("S", "io.ctseed", {}),
                 ("N", "analysis.object_metrics", {"params": {"metrics": planar}})],
                [("S", "N")], "N", seed=ds_3d)
            raise AssertionError(f"object_metrics accepted {planar!r} on a 3D table")
        except ValueError as exc:
            assert planar in str(exc) and "volumetric" in str(exc), exc
    # …and the volume-safe metrics run, adding a real vz. This fixture's objects stay in
    # one plane, so the axial velocity is exactly 0 wherever a predecessor exists (NOT
    # NaN — that would mean "unmeasured") and the 3-norm speed matches the 2D answer.
    _e3, o3 = run([("S", "io.ctseed", {}),
                   ("M", "analysis.object_metrics",
                    {"params": {"metrics": "velocity,speed,neighbors"}})],
                  [("S", "M")], "M", seed=ds_3d)
    c3 = {c: o3.get(D.LABEL, c, layer="labels").values
          for c in ("vz", "vy", "vx", "speed")}
    assert np.isnan(c3["vz"][0]) and np.isnan(c3["vz"][1]), \
        "a track's first detection has no axial predecessor either"
    assert np.allclose(c3["vz"][2:], 0.0), c3["vz"]
    assert np.allclose(c3["speed"][2:], v_expect), c3["speed"]
    assert o3.structure_zkind(D.LABEL, "labels") == "subpixel", "z_kind clobbered"
    ds_nt = (Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
             .with_layer(D.VOXEL, "labels", raster)
             .with_structure(_ST(D.LABEL,
                                 {k: v for k, v in tbl.columns.items()
                                  if k != "track_id"},
                                 layer="labels", z_kind="plane_index")))
    try:
        run([("S", "io.ctseed", {}),
             ("M", "analysis.object_metrics", {"params": {"metrics": "speed"}})],
            [("S", "M")], "M", seed=ds_nt)
        raise AssertionError("object_metrics ran without a track_id column")
    except ValueError as exc:
        assert "track_id" in str(exc), exc

    # ── 6. gamma: full_range == Cell-Tracker's formula on the DECLARED depth ──
    e_full, o_full = run([("S", "io.ctseed", {}),
                          ("G", "enhance.gamma", {"modes": {"scale": "full_range"},
                                                  "params": {"gamma": 2.0}})],
                         [("S", "G")], "G")
    e_pm, o_pm = run([("S", "io.ctseed", {}),
                      ("G", "enhance.gamma", {"modes": {"scale": "plane_max"},
                                              "params": {"gamma": 2.0}})],
                     [("S", "G")], "G")
    full_scale = 2.0 ** 12 - 1                      # 12-bit, not the uint16 container
    assert np.allclose(plane(o_full, 0), (src0 / full_scale) ** 2 * full_scale), \
        "full_range γ does not match (a/(2**bit_depth-1))**g * (2**bit_depth-1)"
    assert np.allclose(plane(o_pm, 0), (src0 / src0.max()) ** 2 * src0.max()), \
        "plane_max γ changed — the pre-V2.13 behaviour must be preserved"
    assert e_full.entry("G").recipe_hash != e_pm.entry("G").recipe_hash
    assert "bit_depth" in {k for k, _ in e_full.entry("G").reads}, \
        "full_range γ must memo-fence on bit_depth"
    assert "bit_depth" not in {k for k, _ in e_pm.entry("G").reads}, \
        "plane_max γ must NOT be fenced on a key it cannot read (R1)"

    # ── 7. dog rescale / clahe tile_grid / morph-gradient blend ───────────────
    _, d_raw = run([("S", "io.ctseed", {}),
                    ("D", "enhance.dog", {"modes": {"dim": "2D"},
                                          "params": {"low_sigma": 0.5,
                                                     "high_sigma": 2.0}})],
                   [("S", "D")], "D")
    _, d_res = run([("S", "io.ctseed", {}),
                    ("D", "enhance.dog", {"modes": {"dim": "2D"},
                                          "params": {"low_sigma": 0.5, "high_sigma": 2.0,
                                                     "rescale": True}})],
                   [("S", "D")], "D")
    a_raw, a_res = plane(d_raw, 0), plane(d_res, 0)
    assert a_raw.min() < 0.0, "a raw DoG is a difference image — negatives expected"
    assert a_res.min() >= 0.0 and abs(a_res.max() - full_scale) < 1e-6, \
        (a_res.min(), a_res.max())
    _, c4 = run([("S", "io.ctseed", {}),
                 ("C", "enhance.clahe", {"modes": {"dim": "2D"},
                                         "params": {"tile_grid": 4}})], [("S", "C")], "C")
    _, c8 = run([("S", "io.ctseed", {}),
                 ("C", "enhance.clahe", {"modes": {"dim": "2D"}})], [("S", "C")], "C")
    assert not np.allclose(plane(c4, 0), plane(c8, 0)), \
        "tile_grid has no effect — it was a pinned constant before V2.13"
    _, g_pure = run([("S", "io.ctseed", {}),
                     ("M", "enhance.morphological_gradient", {"modes": {"dim": "2D"}})],
                    [("S", "M")], "M")
    _, g_mix = run([("S", "io.ctseed", {}),
                    ("M", "enhance.morphological_gradient",
                     {"modes": {"dim": "2D"},
                      "params": {"blend": 0.5, "rescale": True}})], [("S", "M")], "M")
    bg_pure = float(plane(g_pure, 0)[0, 0])
    bg_mix = float(plane(g_mix, 0)[0, 0])
    bg_src = float(src0[0, 0])
    assert bg_pure < 0.05 * bg_src, bg_pure       # flat background ⇒ no edge
    assert 0.4 * bg_src < bg_mix < 0.9 * bg_src, (bg_mix, bg_src)   # blend·original + edge

    # ── 8. measure shape metrics (µm-aware, 2D-only ones refused in 3D) ───────
    lab = np.zeros((1, 1, 1, 1, Y, X), dtype=np.int64)
    lab[0, 0, 0, 0, 10:16, 12:24] = 1                 # 6x12 rectangle → eccentric
    lab[0, 0, 0, 0, 26:32, 30:36] = 2                 # 6x6 square    → eccentricity 0
    ax_1 = AxisSizes(m=1, t=1, z=1, c=1, y=Y, x=X)
    ltab = _ST(D.LABEL, {"id": np.array([1, 2], np.int64),
                         "m": np.zeros(2, np.int64), "t": np.zeros(2, np.int64),
                         "c": np.zeros(2, np.int64), "z": np.zeros(2),
                         "y": np.array([12.5, 28.5]), "x": np.array([17.5, 32.5])},
               layer="labels", z_kind="plane_index")
    ds_m = (Dataset(axes=ax_1, metadata=meta)
            .with_image(ArrayProvider(img[:, :1]))
            .with_layer(D.VOXEL, "labels", lab).with_structure(ltab))
    env_1 = MetaEnvelope(axes=ax_1, metadata=meta)
    e, out = run([("S", "io.ctseed", {}),
                  ("Q", "analysis.measure",
                   {"params": {"shape": "eccentricity,perimeter,solidity,axis_major"}})],
                 [("S", "Q")], "Q", seed=ds_m, seed_env=env_1)
    ecc = out.get(D.LABEL, "eccentricity", layer="labels").values
    per = out.get(D.LABEL, "perimeter", layer="labels").values
    axm = out.get(D.LABEL, "axis_major", layer="labels").values
    sol = out.get(D.LABEL, "solidity", layer="labels").values
    assert ecc[0] > ecc[1] and abs(ecc[1]) < 1e-9, ecc     # square is not eccentric
    assert per[0] > per[1] and axm[0] > axm[1], (per, axm)
    assert np.allclose(sol, 1.0), sol                      # both are convex
    # lengths are physical: a 6x12 px box at 0.5 µm/px has a µm major axis, not a px one
    assert 3.0 < axm[0] < 10.0, axm[0]
    assert "pixel_size_um" in {k for k, _ in e.entry("Q").reads}, "shape not µm-fenced"
    # The 2D-only refusal fires on a table that is 3D *and sits on a volume*. The fixture
    # must therefore be z=2: a `subpixel` stamp over a ONE-PLANE image is not a volume, and
    # since 2026-07-30 measure walks it in 2D rather than refusing (see the z==1 case
    # below), so a z=1 fixture would only be testing the stamp, not the geometry.
    from dataclasses import replace as _replace
    ax_2 = _replace(ax_1, z=2)
    lab2 = np.repeat(lab, 2, axis=2)
    ds_m3 = (Dataset(axes=ax_2, metadata=meta)
             .with_image(ArrayProvider(np.repeat(img[:, :1], 2, axis=2)))
             .with_layer(D.VOXEL, "labels", lab2)
             .with_structure(_ST(D.LABEL, dict(ltab.columns), layer="labels",
                                 z_kind="subpixel")))
    try:
        run([("S", "io.ctseed", {}),
             ("Q", "analysis.measure", {"params": {"shape": "eccentricity"}})],
            [("S", "Q")], "Q", seed=ds_m3, seed_env=MetaEnvelope(axes=ax_2, metadata=meta))
        raise AssertionError("measure accepted a 2D-only shape metric on a 3D table")
    except ValueError as exc:
        assert "2-D regions only" in str(exc), exc

    # ── z==1 can never reach the 3D walk, whatever the table claims (2026-07-30) ──
    # Two entry points used to poison `solidity` with **inf** on a file with no Z axis:
    # (a) a `labels` socket pointed at a bare mask, where the absent-z_kind fallback used
    # to guess "subpixel"; and (b) analysis.label with the 3D lever, which stamps
    # "subpixel" CORRECTLY on a z==1 ND2. Both end in a 3D regionprops walk over a single
    # (1,Y,X) plane, qhull cannot hull a coplanar set, and `area / area_convex` divides by
    # zero. A single plane IS a 2-D region, so both must measure, and finitely.
    ds_flat = (Dataset(axes=ax_1, metadata=meta)
               .with_image(ArrayProvider(img[:, :1]))
               .with_layer(D.VOXEL, "labels", lab)
               .with_structure(_ST(D.LABEL, dict(ltab.columns), layer="labels",
                                   z_kind="subpixel")))
    _, flat_out = run([("S", "io.ctseed", {}),
                       ("Q", "analysis.measure",
                        {"params": {"shape": "solidity,axis_major"}})],
                      [("S", "Q")], "Q", seed=ds_flat, seed_env=env_1)
    sol_f = np.asarray(flat_out.get(D.LABEL, "solidity", layer="labels").values, dtype=float)
    assert sol_f.size and np.isfinite(sol_f).all(), \
        f"a subpixel table on a z==1 image produced non-finite solidity {sol_f}"
    assert np.all(sol_f <= 1.0 + 1e-9), f"solidity must be <= 1, got {sol_f}"

    # ── and a raster with NO Label table is refused outright (2026-07-30) ─────────
    # The other half of the same defect. `labels` is layer_in=Domain.VOXEL, so the picker
    # offers a bare `mask` from analysis.threshold — and voxel_to_label groups a 0/1 mask
    # into ONE region, so measure reported a single row for the whole foreground under the
    # ordinary per-object column names. Nothing downstream could tell.
    ds_mask = (Dataset(axes=ax_1, metadata=meta)
               .with_image(ArrayProvider(img[:, :1]))
               .with_layer(D.VOXEL, "mask", (lab > 0).astype(np.uint8)))
    try:
        run([("S", "io.ctseed", {}),
             ("Q", "analysis.measure", {"params": {"labels": "mask"}})],
            [("S", "Q")], "Q", seed=ds_mask, seed_env=env_1)
        raise AssertionError("measure accepted a raster with no Label table — a binary "
                             "mask measures as one giant region")
    except ValueError as exc:
        assert "not a LABEL raster" in str(exc), exc

    _ok("Cell-Tracker parity (V2.13): temporal kernels pinned (rolling mean == CT's "
        "pad+convolve sandwich for w=3/5/11, log-space fit recovers 500·e^-0.2t to 1e-6, "
        "float32 signals stay float32, ragged tile block averaged over real pixels); "
        "flatten_field 4 methods x 2 references (8 distinct "
        "hashes, envelope==payload bit_depth, ratio drops it, ramp imbalance halved); "
        "temporal_gain 3 references x 3 extents (series_mean flattens every frame, "
        "exponential_fit flattens the decay and LEAVES the outlier, T=1 refused) and "
        "rolling_mean == Cell-Tracker's Temporal Fold Correction end to end at rtol 1e-6 "
        "for all 3 extents against CT's own _correct_global/_correct_local/"
        "_correct_local_gaussian (ragged tile blocks included); "
        "remove_blobs kills a hot pixel and spares the cells in all 3 actions; "
        "object_metrics velocity/speed µm-s exact, NaN at a track's first row, "
        "neighbour distance exact, folds hand-checked, px+dt fenced; object_field emits "
        "the invariant Point grid schema and chains into rasterize_field (z_kind "
        "inherited); 4 consistency repairs (γ full_range == CT's formula on the declared "
        "depth + fenced only when read, DoG rescale, CLAHE tile_grid reachable, "
        "morph-gradient blend) and 5 refusals")


#: Param sockets exempt from the description requirement (socket-contract clause 6), keyed
#: ``op_key`` → {param: reason}. THE ONLY GROUND FOR EXEMPTION is that the param's meaning
#: lives in an external paper or repository, so a description written from this repo alone
#: would be a confident guess — and a wrong tooltip is worse than a missing one, because it
#: reads as authoritative. Code vendored from ND2Studios' OWN history is deliberately NOT
#: exempt: it lives in this repo, so it is readable, and it is documented like everything
#: else. To retire an entry, read that algorithm's own documentation and write the prose.
#:
#: **Currently EMPTY — the whole catalog is documented.** Kept (rather than deleted with the
#: machinery that reads it) because the next externally-backed node will need it, and because
#: an empty dict states the invariant more loudly than an absent one: nothing is exempt.
_SOCKET_DOC_EXEMPT: Dict[str, Dict[str, str]] = {}

# ── how the list got to empty (2026-07-29 → 2026-07-30) ───────────────────────────────
# It started at 29 entries across four nodes. All 29 were retired, in two rounds, and the
# reason each was retired is worth more than the entries were:
#
# 1. `track.objects`' seven `ct_*`/`st_*` linker weights were listed as external on the
#    strength of the method NAMES. That was simply wrong: `kernels/track_objects.md` records
#    the Cell-Tracker linkers as vendored from ND2Studios' own `celltracker/tracking.py`, and
#    its SerialTrack as "a from-scratch NumPy/SciPy/Numba port (not MATLAB SerialTrack, not a
#    pip package)". The cost functions are explicit in the vendored source —
#    `(1-w)*distance + w*topology`, `(1-w)*distance + w*area`. Nothing external was needed.
#    **Exemption follows where the CODE lives, not what the algorithm is called.**
#
# 2. The remaining 22 (ALDIC 9, ALDVC 9, StarDist 4) were genuinely external, and were
#    written once the sources arrived. Two of the three turned out to be READABLE IN THIS
#    ENVIRONMENT, which beats any paper for parameter semantics:
#      * StarDist — `stardist` is installed; the prose comes from
#        `StarDist2D.predict_instances`'s own docstring and the model registry in
#        `stardist/models/__init__.py`, not from the README.
#      * AL-DIC — `al_dic` is installed; all 9 params map 1:1 onto fields of `DICPara`
#        (`al_dic/core/data_structures.py`) with identical defaults, so the mapping is
#        verified rather than assumed. Method semantics: github.com/zachtong/STAQ-DIC-GUI.
#      * AL-DVC — the kernel is vendored in-tree and `kernels/aldvc_field.md` §4 already
#        carried a full range/semantics table; the paper (10.1007/s11340-020-00607-3) and
#        github.com/FranckLab/ALDVC supplied the physical reading of μ, the seed pyramid and
#        the correlation cut.
#
# The lesson for whoever adds the next entry: before exempting anything, check whether the
# package is installed and whether the kernel's own `.md` already documents it. Those two
# checks emptied this list.


def test_socket_docs() -> None:
    """``SocketSpec.description`` — the hover documentation (2026-07-29).

    Guards the two things that can silently rot about per-socket docs:

    1. **Coverage — CATALOG-WIDE.** Every param socket on every registered node type
       carries prose, except the entries in :data:`_SOCKET_DOC_EXEMPT`. The failure mode
       being prevented is a live control with an empty tooltip: the user can move it but
       cannot tell what it will do, which is the same defect as the socket contract's
       "socket nobody reads" wearing a different hat. Catalog-wide from the start, so a NEW
       node cannot be added undocumented — that is the whole point of putting it in a gate
       instead of a checklist.
    2. **Memo neutrality.** ``description`` is presentation-only. ``node_recipe_hash`` keys
       on op_key + params + upstream, never on the ``SocketSpec``, so writing or editing a
       description must NOT change a recipe hash — otherwise every doc edit would silently
       invalidate the whole memo and force a full recompute of every saved graph. Asserted
       against a hash captured with the descriptions stripped.
    """
    import dataclasses

    # ── (1) catalog-wide coverage ─────────────────────────────────────────────────
    # ONLY the shipped catalog is swept. `NODES` is a process-global registry and the earlier
    # groups in this file legitimately register throwaway probes into it (`filt.gauss`,
    # `filt.blur`, `io.load`, `test.c8_probe`, …), so a naive sweep passes when this test runs
    # alone and fails inside the full suite — order-dependence, exactly the fixture hazard
    # build-node-v2 §3 warns about. A fixture is not user-facing, so it has no hover to write.
    #
    # "Shipped" is decided by WHERE THE COMPUTE WAS DEFINED, which is exact and needs no
    # naming convention: a catalog node's compute lives in `nodegraph.nodes`, a fixture's in
    # `nodegraph.selftest`. Two weaker rules were tried first and both leaked — a `test.`
    # prefix misses the `filt.`/`io.` fixtures, and mere COMPUTES membership admits the 11
    # `test.*` probes that DO register a compute. Fixtures built with `define_node` register a
    # NodeSpec and no compute at all, so `.get(...)` is None for them and they drop out here
    # too. Verified: on a clean import this selects exactly the 54 catalog keys, and after a
    # full suite run it still selects those same 54.
    from nodegraph.nodes import COMPUTES

    # Selected by registration PROVENANCE, not by the compute's `__module__` string. The
    # string was the literal "nodegraph.nodes", which under the per-node split (V2.20) stops
    # matching one node at a time: each moved node silently left this gate while the suite
    # stayed green. `is_catalog_op` asks the registry who registered the op, which is exact
    # and stays exact however the catalog is laid out.
    shipped = [s for s in NODES.all() if _is_catalog_op(s.op_key)]
    assert len(shipped) >= 70, \
        f"the catalog sweep found only {len(shipped)} shipped nodes — did the discriminator " \
        f"stop matching? (registry has {len(NODES.all())})"

    missing, exempt_n, total = [], 0, 0
    for nspec in shipped:
        allowed = _SOCKET_DOC_EXEMPT.get(nspec.op_key, {})
        for s in nspec.inputs:
            if s.type is SocketType.DATASET:
                continue
            total += 1
            if (s.description or "").strip():
                continue
            if s.name in allowed:
                exempt_n += 1
                continue
            missing.append(f"{nspec.op_key}.{s.name}")
    assert not missing, (
        "%d param socket(s) have no hover description (socket-contract clause 6). Either "
        "write the prose — what it does and how it moves the result — or, ONLY if the "
        "param's meaning lives in an external paper/repo, add it to _SOCKET_DOC_EXEMPT "
        "with the reason:\n  %s" % (len(missing), "\n  ".join(missing)))
    # The exemption list must not rot into a dumping ground: every entry must name a socket
    # that still exists, so a renamed/deleted param cannot leave a stale free pass behind.
    for op_key, entries in _SOCKET_DOC_EXEMPT.items():
        nspec = NODES.get(op_key)
        assert nspec is not None, f"_SOCKET_DOC_EXEMPT names an unknown node {op_key!r}"
        real = {s.name for s in nspec.inputs}
        assert not (set(entries) - real), \
            f"_SOCKET_DOC_EXEMPT[{op_key!r}] names sockets that no longer exist: " \
            f"{sorted(set(entries) - real)}"
        assert all(str(v).strip() for v in entries.values()), \
            f"_SOCKET_DOC_EXEMPT[{op_key!r}] has an entry with no stated reason"

    # ── (2) the CellSAM set in detail (the worked example the skill points at) ─────
    spec = NODES.get("analysis.segment")
    state = dict(spec.default_state()); state["method"] = "cellsam"; state["dim"] = "2D"
    params = [s for s in spec.active_inputs(state) if s.type is not SocketType.DATASET]
    assert len(params) >= 13, f"expected the CellSAM socket set, got {len(params)}"

    # Prose, not a restatement of the label. The shortest real description here is
    # `remove_boundaries`; 120 chars is comfortably below it and still catches a stub.
    #
    # There is deliberately NO automated "does it state an EFFECT?" assertion. The first
    # attempt sniffed for words like more/fewer/larger/area and immediately failed on
    # `cellsam_model`, whose description states its effect perfectly well ("no longer
    # matching the published benchmarks") in words the list did not contain. A keyword list
    # over prose is unfalsifiable-by-construction: it grows every time it is wrong, and each
    # growth weakens it. Whether a description explains consequence is a REVIEW judgment;
    # what a machine can hold is coverage, substance and memo-neutrality, which is what this
    # test holds.
    thin = [s.name for s in params if len(s.description.strip()) < 120]
    assert not thin, f"description too thin to be useful (needs what it DOES and how it " \
                     f"moves the result): {thin}"
    echoes = [s.name for s in params
              if s.description.strip().lower().rstrip(".") in
              (s.name.lower(), (s.label or "").lower())]
    assert not echoes, f"description just repeats the label — say what it does: {echoes}"

    # (2) memo neutrality — same params, descriptions stripped, same recipe hash.
    bare = [dataclasses.replace(s, description="") for s in params]
    pv = {s.name: s.default for s in params}
    assert (node_recipe_hash("analysis.segment", pv, (), ())
            == node_recipe_hash("analysis.segment", {s.name: s.default for s in bare},
                                (), ())), \
        "a description must never reach the recipe hash — doc edits would nuke the memo"

    # The GUI's hover builder is importable headlessly and actually embeds the prose,
    # HTML-escaped and hard-wrapped (the widgets are exercised by the phase-5 GUI probe).
    try:
        from nodelab_v2.node_item import socket_hover_text
    except Exception:                      # PySide6 absent — structural half still gated
        _ok("socket docs: %d CellSAM-active sockets documented; memo-neutral "
            "(hover builder SKIPPED, PySide6 absent)" % len(params))
        return
    bbox = next(s for s in params if s.name == "bbox_threshold")
    tip = socket_hover_text(bbox)
    assert "<b>bbox_threshold" in tip and "<br>" in tip, tip[:200]
    assert "CellFinder" in tip and "precision/recall" in tip
    assert max(len(ln) for ln in tip.split("<br>")) < 140, \
        "prose must be hard-wrapped — Qt tooltips wrap only at the screen edge"
    evil = dataclasses.replace(bbox, description="a < b & c > d")
    assert "&lt; b &amp; c &gt;" in socket_hover_text(evil), \
        "description text must be HTML-escaped (it is rendered as rich text)"

    _ok("socket docs (clause 6): %d/%d param sockets across all %d SHIPPED catalog node "
        "types carry hover prose (test fixtures in the global registry excluded), %d exempt "
        "with a stated external-paper/repo reason (list checked live against the socket "
        "names); the %d CellSAM-active sockets are substantive (not label echoes); "
        "descriptions are memo-neutral (identical recipe_hash stripped vs not); the hover "
        "builder embeds, HTML-escapes and hard-wraps them"
        % (total - exempt_n, total, len(shipped), exempt_n, len(params)))


# ── per-OPTION docs: clause 6 for dropdowns (2026-08-03) ──────────────────────────────

#: The shortest an option's prose may be. Deliberately well under the 120-char bar the param
#: descriptions are held to: an option is documented RELATIVE to its siblings ("assumes two
#: intensity classes", "keeps every plane's own median") and several are honestly one clause
#: long. 60 characters still refuses the failure this gate exists for — the token restated as
#: a phrase ("Otsu's method", "the GMM linker").
_OPTION_DOC_MIN = 60

#: Dropdown OPTIONS exempt from the per-option requirement, ``op_key`` → {control: reason},
#: where ``control`` is the Mode name or the socket name. The grounds are the same as
#: :data:`_SOCKET_DOC_EXEMPT`'s and no wider: the option's meaning lives in an external paper
#: or repo, so prose written from this repo alone would be a confident guess, and a wrong
#: tooltip on a method selector is worse than a missing one — it will be believed, and it
#: decides which algorithm runs.
#:
#: **Currently EMPTY.** Every dropdown in the catalog documents every option.
_OPTION_DOC_EXEMPT: Dict[str, Dict[str, str]] = {}


def test_option_docs() -> None:
    """Per-OPTION hover documentation for dropdowns — ``choice_docs`` (2026-08-03).

    :func:`test_socket_docs` holds the line for params that are a number or a string, and a
    dropdown slipped through it: a ``description`` on the control plus a menu of bare tokens
    PASSES that gate while leaving the user's actual question — what is the difference
    between ``otsu`` and ``li``, between ``serialtrack`` and ``centroid`` — unanswered on
    every surface. Naming a method is not explaining it. So the same three things are held
    here, one level down:

    1. **Coverage, catalog-wide.** Every Mode on every shipped node carries a description AND
       a line for each of its choices; every ``choices`` / ``vocab`` socket carries a line for
       each option. Catalog-wide from the start so a new dropdown cannot arrive undocumented.
    2. **Substance.** At least :data:`_OPTION_DOC_MIN` characters and not a restatement of the
       token. The threshold node's six methods are additionally checked in detail, as the
       worked example the skill points at.
    3. **Memo neutrality.** ``choice_docs`` is presentation-only on both spec types. For a
       socket that is a direct hash check; for a Mode it is structural — the engine folds the
       resolved mode STATE into ``__modes__``, never the ``ModeSpec`` — and both are asserted,
       because a doc edit that invalidated the memo would make documenting the catalog cost a
       full recompute of every saved graph.

    Registration already refuses a ``choice_docs`` key that matches no option
    (:func:`nodegraph.registry._check_choice_docs`); that guard is exercised here too, since a
    silently-unrendered tooltip is exactly what this gate would otherwise fail to notice.
    """
    import dataclasses

    from nodegraph.registry import _check_choice_docs

    # The catalog PLUS the GUI-layer ops, which is wider than `test_socket_docs`'s sweep on
    # purpose: `io.dock` carries two dropdowns (`state`, `precision`) and is as user-facing as
    # anything in the catalog, so leaving it out would exempt the one node whose wrong choice
    # silently quantizes a whole bake. `nodelab_v2.ops` is deliberately Qt-free (its module
    # docstring says so), so importing it here keeps this test headless; `ensure_ops` is
    # idempotent, so the sweep does not depend on whether the GUI ran first. Fixtures are
    # still excluded — they are not user-facing and have no hover to write.
    try:
        import nodelab_v2.ops as _ops
        _ops.ensure_ops()
        gui_ops = frozenset(_ops.NODES.keys()) & frozenset(("io.dock", "io.load",
                                                            "view.viewer"))
    except Exception:                      # noqa: BLE001 — catalog half still gated
        gui_ops = frozenset()
    shipped = [s for s in NODES.all()
               if _is_catalog_op(s.op_key) or s.op_key in gui_ops]
    assert len(shipped) >= 70, \
        f"the catalog sweep found only {len(shipped)} shipped nodes — did the discriminator " \
        f"stop matching? (registry has {len(NODES.all())})"
    assert not gui_ops or NODES.get("io.dock") in shipped, \
        "io.dock has dropdowns and must be swept — the GUI-layer ops are in scope here"

    # ── (1) coverage ──────────────────────────────────────────────────────────────
    missing: List[str] = []
    thin: List[str] = []
    echoes: List[str] = []
    n_controls = n_options = exempt_n = 0

    def check(op_key: str, control: str, options, docs, exempt: bool) -> None:
        nonlocal n_controls, n_options, exempt_n
        n_controls += 1
        for opt in options:
            n_options += 1
            doc = str((docs or {}).get(opt, "") or "").strip()
            if not doc:
                if exempt:
                    exempt_n += 1
                else:
                    missing.append(f"{op_key}[{control}].{opt}")
                continue
            if len(doc) < _OPTION_DOC_MIN:
                thin.append(f"{op_key}[{control}].{opt} ({len(doc)} chars)")
            if doc.lower().rstrip(".") == opt.lower().replace("_", " "):
                echoes.append(f"{op_key}[{control}].{opt}")

    for nspec in shipped:
        free = _OPTION_DOC_EXEMPT.get(nspec.op_key, {})
        for m in nspec.modes:
            if not (m.description or "").strip() and m.name not in free:
                missing.append(f"{nspec.op_key}[{m.name}] (the Mode's own description)")
            check(nspec.op_key, m.name, m.choices, m.choice_docs, m.name in free)
        for s in nspec.inputs:
            opts = tuple(s.choices) + tuple(s.vocab)
            if opts:
                check(nspec.op_key, s.name, opts, s.choice_docs, s.name in free)

    assert not missing, (
        "%d dropdown option(s) have no explanation. A dropdown documents each option, not "
        "just the control — the difference between the options IS the question the user "
        "opened the menu to answer. Write the prose, or (ONLY if the option's meaning lives "
        "in an external paper/repo) add it to _OPTION_DOC_EXEMPT with the reason:\n  %s"
        % (len(missing), "\n  ".join(missing)))
    assert not thin, ("option prose too thin to be useful (< %d chars) — say what the option "
                      "ASSUMES and which way it moves the result:\n  %s"
                      % (_OPTION_DOC_MIN, "\n  ".join(thin)))
    assert not echoes, f"option prose just restates the token: {echoes}"
    for op_key, entries in _OPTION_DOC_EXEMPT.items():
        nspec = NODES.get(op_key)
        assert nspec is not None, f"_OPTION_DOC_EXEMPT names an unknown node {op_key!r}"
        real = {m.name for m in nspec.modes} | {s.name for s in nspec.inputs}
        assert not (set(entries) - real), \
            f"_OPTION_DOC_EXEMPT[{op_key!r}] names controls that no longer exist: " \
            f"{sorted(set(entries) - real)}"
        assert all(str(v).strip() for v in entries.values()), \
            f"_OPTION_DOC_EXEMPT[{op_key!r}] has an entry with no stated reason"

    # ── (2) the worked example: the threshold node's six methods ───────────────────
    spec = NODES.get("analysis.threshold")
    method = next(m for m in spec.modes if m.name == "method")
    assert len(method.choices) >= 6, f"expected the six threshold methods, got {method.choices}"
    for opt in method.choices:
        assert len(method.choice_docs[opt].strip()) >= 120, \
            f"analysis.threshold[method].{opt}: a method selector's options need enough " \
            f"prose to choose BETWEEN them"

    # ── (3) memo neutrality ───────────────────────────────────────────────────────
    # Sockets: choice_docs travels with the SocketSpec, and the hash keys on params only.
    seg = NODES.get("analysis.segment")
    csock = [s for s in seg.inputs if s.choices or s.vocab]
    assert csock, "expected analysis.segment's model-name dropdowns"
    pv = {s.name: s.default for s in csock}
    bare = {dataclasses.replace(s, choice_docs={}).name: s.default for s in csock}
    assert (node_recipe_hash("analysis.segment", pv, (), ())
            == node_recipe_hash("analysis.segment", bare, (), ())), \
        "socket choice_docs must never reach the recipe hash"
    # Modes: the engine folds the resolved STATE (`__modes__`), never the ModeSpec — so
    # stripping the docs cannot move the state, and therefore cannot move the key.
    state = spec.default_state()
    stripped = [dataclasses.replace(m, description="", choice_docs={}) for m in spec.modes]
    assert state == {m.name: m.resolved_default() for m in stripped}, \
        "a Mode's docs must not affect its resolved default"
    assert (node_recipe_hash("analysis.threshold", {"__modes__": state}, (), ())
            == node_recipe_hash("analysis.threshold",
                                {"__modes__": {m.name: m.resolved_default()
                                               for m in stripped}}, (), ())), \
        "mode choice_docs must never reach the recipe hash"

    # ── (4) registration refuses a key that documents nothing ─────────────────────
    for bad, why in ((("nope",), "a key matching no option"),
                     (("",), "an empty key")):
        try:
            _check_choice_docs("probe[m]", {bad[0]: "prose"}, ("a", "b"), "choices")
        except ValueError:
            pass
        else:
            raise AssertionError(f"_check_choice_docs accepted {why}")
    try:
        _check_choice_docs("probe[m]", {"a": "   "}, ("a", "b"), "choices")
    except ValueError:
        pass
    else:
        raise AssertionError("_check_choice_docs accepted a blank explanation")
    # ... and a Mode whose default is not on its own list (the blank-row dropdown).
    try:
        define_node("test.badmode", "Bad", modes=[Mode("m", ["a", "b"], default="c")])
    except ValueError:
        pass
    else:
        raise AssertionError("registration accepted a Mode default outside its choices")

    # ── (5) the hover builders render it ──────────────────────────────────────────
    try:
        from nodelab_v2.node_item import mode_hover_text, option_hover_text
    except Exception:                      # PySide6 absent — structural half still gated
        _ok("option docs: %d options across %d dropdowns documented; memo-neutral "
            "(hover builders SKIPPED, PySide6 absent)" % (n_options, n_controls))
        return
    tip = mode_hover_text(method)
    for opt in method.choices:
        assert f"• {opt} —" in tip, f"{opt} missing from the Mode hover: {tip[:200]}"
    assert "<b>method — mode · 6 options</b>" in tip
    assert max(len(ln) for ln in tip.split("<br>")) < 140, \
        "option prose must be hard-wrapped — Qt tooltips wrap only at the screen edge"
    evil = dataclasses.replace(method, choice_docs={"otsu": "a < b & c > d"})
    assert "&lt; b &amp; c &gt;" in mode_hover_text(evil), \
        "option prose must be HTML-escaped (it is rendered as rich text)"
    assert option_hover_text("otsu", "") == "", \
        "an undocumented option must yield NO tooltip, not an empty popup"
    assert "<b>otsu</b>" in option_hover_text("otsu", method.choice_docs["otsu"])

    _ok("option docs (clause 6 for dropdowns): %d options across %d dropdowns on %d SHIPPED "
        "catalog nodes each carry their own explanation (%d exempt), all >= %d chars and none "
        "a restatement of the token; the 6 threshold methods are substantive; docs are "
        "memo-neutral on both spec types (socket hash + mode state); registration refuses a "
        "key that documents nothing, a blank explanation and an off-list default; the Mode "
        "hover embeds, escapes and hard-wraps every option"
        % (n_options - exempt_n, n_controls, len(shipped), exempt_n, _OPTION_DOC_MIN))


# ── the param<->socket contract (catalog-wide structural guard, 2026-07-28) ─────

#: Params a compute reads that legitimately have NO socket. Every entry needs a reason;
#: anything else is the defect this guard exists to catch.
_PARAM_NO_SOCKET_OK = {
    # read only as the fallback INSIDE `scale_xy`'s lookup — a back-compat alias for
    # graphs written before the axis-split, not a control of its own.
    ("util.resample", "scale"),
}
#: Sockets no compute reads. Empty, and it should stay that way: a declared socket the
#: kernel ignores is the "live-looking GUI control" the node charter forbids.
_SOCKET_UNREAD_OK: set = set()


def _compute_source(module, fn) -> str:
    """The source text of one compute (for the declared-once check); ``""`` if unavailable.

    The gate the source feeds is clause (f): a layer name's default must be declared ONCE, on
    the socket, not repeated inside the compute. That check is only as good as this function.

    It used to accept a compute only when ``fn.__module__ == "nodegraph.nodes"``. Under the
    per-node split every compute reports its own module, so that test excluded ALL of them and
    clause (f) went quietly dark — a check that inspects nothing and reports nothing wrong.
    Now the source is taken from wherever the function actually lives, and the caller's
    ``module`` argument is only the fallback for a function with no resolvable source."""
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):        # pragma: no cover - defensive
        return ""


def test_catalog_import_hygiene() -> None:
    """The five import rules every module under ``nodegraph/catalog/`` must obey (V2.20).

    Per-node granularity is not a property of the directory layout — it is a property of what
    each node module *imports*, because the fingerprint that decides which memo entries
    survive an edit is a digest over each module's transitive import closure
    (:func:`nodegraph.hotreload.dependency_closure`). Every rule below is a way to have 63
    separate files that all still re-key together, i.e. to pay the cost of the split and keep
    none of the benefit. None of them is visible in a functional test: the nodes would work
    perfectly and the memo would simply throw away everything on every edit.

    1. **No importing the facade.** ``from nodegraph.nodes import X`` inside a catalog module
       is a cycle (the facade loads the catalog) and pulls the facade's whole closure in.
    2. **No bare ``from nodegraph import X``.** It resolves through ``nodegraph/__init__``,
       which itself imports the facade — the same collapse, one level of indirection away.
       Import the leaf module instead (``from nodegraph.gpu import ...``).
    3. **No importing the ``_shared`` package**, only its submodules. ``from
       nodegraph.catalog._shared import to_pixels_v2`` would put the package initialiser in
       the closure; if that ``__init__`` ever re-exports, all 15 concern modules land in every
       closure at once and the measured over-invalidation goes from 76 back to ~1497.
    5. **No node module imports another node module.** Importing a module *executes* it, so a
       node→node import registers the imported node early and scrambles the catalog's
       registration order (which the link-drag menu enumerates); it also puts both nodes in
       one fingerprint closure, so editing either re-keys both. This is not hypothetical: the
       migration's first pass emitted ``from nodegraph.catalog.flow.iterate import _m`` into
       two unrelated nodes — a *loop temporary* mistaken for a module-level name — and the
       snapshot gate caught it as a 29-position order change.
    4. **Package initialisers stay logic-free.** ``dependency_closure`` deliberately excludes
       them (``catalog/__init__``'s ``MODULES`` list changes whenever a node is added, so
       including it would re-key all 63 nodes on every addition). That exclusion is only sound
       while an ``__init__`` contributes nothing but names and import order."""
    import ast as _ast
    import os as _os

    root = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "catalog")
    if not _os.path.isdir(root):
        _ok("catalog import hygiene: no catalog package yet (pre-split) — nothing to check")
        return
    bad: list = []
    n_mod = 0
    inits: list = []
    for dirpath, _dirs, files in _os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            path = _os.path.join(dirpath, fname)
            rel = _os.path.relpath(path, root).replace(_os.sep, ".")[:-3]
            with open(path, encoding="utf-8") as fh:
                tree = _ast.parse(fh.read(), filename=path)
            if fname == "__init__.py":
                inits.append((rel, tree, path))
                continue
            n_mod += 1
            for node in _ast.walk(tree):
                if isinstance(node, _ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "nodegraph.nodes":
                        bad.append(f"{rel}: `from nodegraph.nodes import …` (rule 1)")
                    elif mod == "nodegraph":
                        bad.append(f"{rel}: `from nodegraph import "
                                   f"{', '.join(a.name for a in node.names)}` (rule 2) — "
                                   f"import the leaf module instead")
                    elif mod == "nodegraph.catalog._shared":
                        bad.append(f"{rel}: `from nodegraph.catalog._shared import …` "
                                   f"(rule 3) — name the submodule")
                    elif (mod.startswith("nodegraph.catalog.")
                          and "._shared." not in mod
                          and not mod.endswith("._base")
                          and mod != f"nodegraph.catalog.{rel}"):
                        bad.append(
                            f"{rel}: `from {mod} import …` (rule 5) — one node module must "
                            f"not import another. Importing a module EXECUTES it, so this "
                            f"registers that node early and scrambles catalog order; it also "
                            f"welds the two nodes' fingerprints together. Move the shared "
                            f"code to `_shared/`.")
                    # rule 6 (V2.22): `nodegraph.synth` is a TEST FIXTURE — it generates
                    # synthetic beds with known per-voxel ownership so the shape nodes can be
                    # graded against ground truth. It must never enter a shipped compute path.
                    # A machine-checked clause is strictly stronger than the accidental
                    # guarantee that living under `scripts/` would give, and it is the reason
                    # the fixture is allowed to sit inside the package at all.
                    if mod == "nodegraph.synth" or mod.startswith("nodegraph.synth."):
                        bad.append(f"{rel}:{node.lineno}: `from nodegraph.synth import …` "
                                   f"(rule 6) — that module is a test fixture and must not "
                                   f"reach a shipped compute path")
                    elif mod == "nodegraph" and any(a.name == "synth" for a in node.names):
                        bad.append(f"{rel}:{node.lineno}: imports the `synth` test fixture "
                                   f"(rule 6)")
                elif isinstance(node, _ast.Import):
                    for a in node.names:
                        if a.name == "nodegraph.nodes":
                            bad.append(f"{rel}: `import nodegraph.nodes` (rule 1)")
                        elif (a.name == "nodegraph.synth"
                              or a.name.startswith("nodegraph.synth.")):
                            bad.append(f"{rel}:{node.lineno}: `import nodegraph.synth` "
                                       f"(rule 6) — test fixture, not a compute path")

    # rule 4: an __init__ may hold only docstrings, imports, assignments, and function defs
    # that are not CALLED at import time (catalog/__init__'s `load()` is defined there but
    # invoked by the facade, which is what keeps registration order in one place).
    for rel, tree, path in inits:
        for node in tree.body:
            if isinstance(node, (_ast.Expr, _ast.Import, _ast.ImportFrom, _ast.Assign,
                                 _ast.AnnAssign, _ast.FunctionDef, _ast.ClassDef)):
                continue
            bad.append(f"{rel}/__init__.py:{node.lineno}: package initialisers must be "
                       f"logic-free ({type(node).__name__}) — dependency_closure excludes "
                       f"them, so logic here is invisible to every fingerprint (rule 4)")
        calls = [n for n in tree.body if isinstance(n, _ast.Expr)
                 and isinstance(n.value, _ast.Call)]
        if calls:
            bad.append(f"{rel}/__init__.py:{calls[0].lineno}: a call at import time in a "
                       f"package initialiser (rule 4)")

    assert not bad, ("catalog import hygiene violated — each of these silently collapses "
                     "per-node memo granularity:\n  " + "\n  ".join(bad))
    # A positive check, so the gate cannot pass by finding nothing: the shared package must
    # exist and its initialiser must be free of re-exports.
    shared_init = _os.path.join(root, "_shared", "__init__.py")
    if _os.path.exists(shared_init):
        with open(shared_init, encoding="utf-8") as fh:
            st = _ast.parse(fh.read()).body
        assert not [n for n in st if isinstance(n, (_ast.Import, _ast.ImportFrom))], \
            ("_shared/__init__.py must not import (and so must not re-export) its "
             "submodules: one re-export puts all 15 concern modules into every node's "
             "closure, which is the monolith with extra files")
    _ok(f"catalog import hygiene: {n_mod} catalog modules obey the six import rules "
        f"(no facade import, no bare `from nodegraph import`, no `_shared` package import, "
        f"logic-free initialisers, no node->node imports, no `nodegraph.synth`) — the rules "
        f"that keep per-node memo granularity and registration order from silently collapsing, "
        f"and the test fixture out of every shipped compute path")


def _is_catalog_op(op_key: str) -> bool:
    """Whether ``op_key`` is a shipped catalog node — see
    :func:`nodegraph.hotreload.is_catalog_op`, which is the single definition. Wrapped here
    only so the three gates below read as one predicate rather than three copies of an
    import."""
    from nodegraph.hotreload import is_catalog_op
    return is_catalog_op(op_key)


def _catalog_modules():
    """Every module whose source the catalog-wide gates must read: the node modules plus the
    shared prelude they forward params through.

    Exists because these gates are SOURCE-level. Before the per-node split there was one
    module to parse and ``inspect.getsource(nodegraph.nodes)`` was the whole catalog; now the
    computes live in ``nodegraph/catalog/**`` and ``nodegraph.nodes`` is a facade whose AST
    contains no ``FunctionDef`` at all. Parsing only the facade would resolve every param key
    to nothing and report every socket in the catalog as dead."""
    import sys as _sys
    from nodegraph.hotreload import node_modules
    mods = []
    for name in node_modules():
        mod = _sys.modules.get(name)
        if mod is None:
            try:
                import importlib
                mod = importlib.import_module(name)
            except ImportError:
                continue
        mods.append(mod)
    return mods


def _param_key_index(modules):
    """Map every function across *modules* to the param keys it reads, resolving keys passed
    THROUGH helpers. Without that resolution ``_radius_px(ctx, "radius", 0.3)`` looks like
    it reads nothing and every radius socket in the catalog reads as dead.

    Accepts one module or many. Many, since the per-node split: a node's compute and the
    shared helper it forwards a param key through are now in DIFFERENT files, so the
    resolution has to span them or every forwarded key resolves to nothing. The tables are
    keyed by function NAME, which is safe precisely because these functions all came out of
    one module — but a collision would silently shadow a helper and quietly weaken the gate,
    so it is asserted rather than assumed."""
    import ast
    import inspect

    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    fns, top_origin = {}, {}
    for mod in modules:
        try:
            tree = ast.parse(inspect.getsource(mod))
        except (OSError, TypeError):               # a namespace/extension module has no source
            continue
        top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.FunctionDef):
                continue
            # Only MODULE-LEVEL names are asserted unique. A nested closure called `plane` or
            # `vol` is idiomatic here — most computes define one and hand it to `_map_image` —
            # and several already shared a name inside the old single module, where the index
            # silently kept the last one. That shadowing is harmless because such a closure is
            # passed as a VALUE, never called by name, so the resolver never looks it up.
            # A collision between two module-level helpers is different: those ARE resolved by
            # name, so one shadowing the other would make the shadowed node's forwarded param
            # keys resolve to the wrong function and its sockets read as unread — the gate
            # would report a defect in the wrong node, or miss one. Newly possible after the
            # per-node split (the monolith's single namespace made it impossible), hence the
            # check lands here rather than being assumed away.
            if n.name in top:
                prev = top_origin.get(n.name)
                assert prev is None or prev == mod.__name__, (
                    f"two catalog modules define a module-level function called {n.name!r} "
                    f"({prev} and {mod.__name__}) — the param-key resolver is keyed by name, "
                    f"so one would shadow the other and its forwarded param keys would read "
                    f"as unread sockets. Rename one.")
                top_origin[n.name] = mod.__name__
            fns[n.name] = n

    def is_read(call):
        f = call.func
        if not isinstance(f, ast.Attribute):
            return False
        return ((f.attr == "get" and isinstance(f.value, ast.Attribute)
                 and f.value.attr == "params")            # ctx.params.get(K)
                or f.attr == "param"                      # ctx.channel(c).param(K)
                or f.attr == "layer")                     # ctx.layer(K) — V2.11

    def callee(c):
        f = c.func
        if isinstance(f, ast.Name):
            return f.id
        return f.attr if isinstance(f, ast.Attribute) else None

    lits, fwd, calls, argidx = {}, {}, {}, {}
    for name, fn in fns.items():
        args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
        argidx[name] = args
        L, F, C = set(), set(), []

        def _key(k):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                L.add(k.value)
            elif isinstance(k, ast.Name) and k.id in args:
                F.add((k.id, ""))
            elif (isinstance(k, ast.BinOp) and isinstance(k.op, ast.Add)
                  and isinstance(k.left, ast.Name) and k.left.id in args
                  and isinstance(k.right, ast.Constant)):
                F.add((k.left.id, k.right.value))           # `name + "_z"` paired float

        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                C.append(node)
                if is_read(node) and node.args:
                    _key(node.args[0])
            # ctx.params["X"] — `detect.spots` reads its axial radii this way (guarded by
            # an `in ctx.params` test, so absent means "fall back to the lateral value").
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                  and node.value.attr == "params"):
                _key(node.slice)
            # `"X" in ctx.params` — the membership half of that same idiom
            elif isinstance(node, ast.Compare) and len(node.comparators) == 1:
                cmp0 = node.comparators[0]
                if (isinstance(node.ops[0], ast.In)
                        and isinstance(cmp0, ast.Attribute) and cmp0.attr == "params"):
                    _key(node.left)
        lits[name], fwd[name], calls[name] = L, F, C

    # a helper that hands its OWN arg to a forwarder is itself a forwarder — transitive,
    # e.g. _rad(name) -> _cleanup_um(name) -> ctx.channel(0).param(name)
    changed = True
    while changed:
        changed = False
        for name in fns:
            for c in calls[name]:
                cn = callee(c)
                if cn not in fwd or cn == name or not fwd[cn]:
                    continue
                for aname, suf in list(fwd[cn]):
                    i = argidx[cn].index(aname) if aname in argidx[cn] else -1
                    if 0 <= i < len(c.args):
                        a = c.args[i]
                        if (isinstance(a, ast.Name) and a.id in argidx[name]
                                and (a.id, suf) not in fwd[name]):
                            fwd[name].add((a.id, suf))
                            changed = True

    def keys(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen or name not in fns:
            return set()
        seen.add(name)
        out = set(lits[name])
        for c in calls[name]:
            cn = callee(c)
            if cn in fwd:
                for aname, suf in fwd[cn]:
                    i = argidx[cn].index(aname) if aname in argidx[cn] else -1
                    if 0 <= i < len(c.args):
                        a = c.args[i]
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            out.add(a.value + suf)
            out |= keys(cn, seen)
        return out

    return keys


def test_param_socket_contract() -> None:
    """Every param a compute reads has a socket, and every socket a compute reads.

    Structural, not behavioural, and that is the point: the engine does NOT filter params
    against the socket list, so a compute can happily read a param no socket exposes and
    every functional test still passes — the value is simply pinned to its fallback
    forever and no user can change it. It is invisible headlessly and only bites in the
    GUI, which builds its widgets from ``NodeSpec.inputs``. A catalog-wide sweep on
    2026-07-28 found the pattern in 18 nodes, including ``transform.transfer_domain``,
    whose four functional params were ALL unreachable (from the GUI it could only ever
    move ``mask`` voxel->frame with ``mean``). The mirror case — a declared socket no
    kernel path reads — is the "live-looking GUI control the selected kernel silently
    ignores" that the ``track.objects`` review established as charter-forbidden."""
    import nodegraph.nodes as NN

    # ── clauses 1 & 2: every param read has a socket, every socket is read ─────
    keys_of = _param_key_index(_catalog_modules())
    bad_unreachable, bad_dead, audited = [], [], 0
    bad_presentation_read: List[str] = []
    for spec in sorted(NODES.all(), key=lambda s: s.op_key):
        fn = NN.COMPUTES.get(spec.op_key)
        fname = getattr(fn, "__name__", None)
        # Only real catalog nodes: the index is built from `nodegraph.nodes`' AST, so a
        # node whose compute is defined elsewhere has no resolvable reads and every socket
        # would read as dead. In a full-suite run that is every `test.*` fixture (their
        # computes live in this file) — the same fixture-vs-registry overlap that makes
        # fake op_keys mandatory. Order-dependent otherwise: green alone, red in suite.
        if fname is None or not _is_catalog_op(spec.op_key):
            continue
        audited += 1
        used = keys_of(fname)
        declared = {s.name for s in spec.inputs}
        datasets = {s.name for s in spec.inputs
                    if getattr(s.type, "name", "") == "DATASET"}
        modes = {m.name for m in spec.modes}
        for p in sorted(used):
            if (p in declared or p in modes or p.startswith("__")
                    or (spec.op_key, p) in _PARAM_NO_SOCKET_OK):
                continue
            bad_unreachable.append("%s.%s" % (spec.op_key, p))
        # PRESENTATION sockets (V2.19) are exempt from clause 2, and the exemption is the
        # POINT of the flag rather than a hole in it. Such a param is excluded from
        # `node_recipe_hash` so moving it cannot invalidate a memo entry — and a value that
        # is not in the key must not be in the payload either, or a hit would serve a result
        # made with the value the control used to have. So the compute is *required* not to
        # read it; its only legitimate consumer is the GUI, live from the document. A
        # presentation socket the GUI does not read either is still a dead control, but that
        # is a GUI-side fact this source-level check cannot see — `nodelab_v2.runner._look`
        # is the reader, and the phase-5 probe is where it is exercised.
        presentation = {s.name for s in spec.inputs if getattr(s, "presentation", False)}
        for d in sorted(declared - used - datasets - presentation):
            if d.startswith("__") or (spec.op_key, d) in _SOCKET_UNREAD_OK:
                continue
            bad_dead.append("%s.%s" % (spec.op_key, d))
        # ...and the converse: a presentation socket the compute DOES read is a live
        # contradiction — the value would reach the payload while being absent from the key.
        for d in sorted(presentation & used):
            bad_presentation_read.append("%s.%s" % (spec.op_key, d))

    assert not bad_unreachable, (
        "compute reads a param with NO socket (unreachable from the GUI, pinned to its "
        "fallback): %s — add an input socket, or allowlist it in _PARAM_NO_SOCKET_OK "
        "with the reason" % bad_unreachable)
    assert not bad_presentation_read, (
        "a PRESENTATION socket is read by its compute: %s — presentation params are "
        "excluded from node_recipe_hash, so letting the value reach the payload means a "
        "memo hit serves a result stamped with the control's OLD value. Read it in the GUI "
        "from the document instead, or drop presentation=True" % bad_presentation_read)
    assert not bad_dead, (
        "socket declared that no compute reads (a control that does nothing — charter-"
        "forbidden): %s — wire it, `available_in`-gate it, or delete it" % bad_dead)

    # ── clause 3: layer sockets are typed, resolvable, and domain-consistent ────
    # `layer_in`/`layer_out` are the declarative half of the layer catalog
    # (metadata.propagate_meta) and the GUI picker. Each clause below is a rule the
    # catalog must obey for BOTH to stay correct as nodes are added.
    bad_layer = []
    n_in = n_out = n_path = 0
    for spec in sorted(NODES.all(), key=lambda s: s.op_key):
        fn = NN.COMPUTES.get(spec.op_key)
        if not _is_catalog_op(spec.op_key):
            continue
        mode_names = {m.name: set(m.choices) for m in spec.modes}
        # (a0) a MODE may be gated on another Mode's value too (V2.12 `ModeSpec.
        #      available_in`), and the same typo hides it in every state — with no socket
        #      list to notice its absence. A mode must never gate on ITSELF, which would
        #      make its own visibility depend on the value it is choosing.
        for mo in spec.modes:
            mtag = "%s![%s]" % (spec.op_key, mo.name)
            for mname, allowed in (mo.available_in or {}).items():
                if mname == mo.name:
                    bad_layer.append("%s: a Mode cannot gate on itself" % mtag)
                    continue
                if mname not in mode_names:
                    bad_layer.append("%s: available_in names mode %r that does not exist"
                                     % (mtag, mname))
                    continue
                unknown = set(allowed) - mode_names[mname]
                if unknown:
                    bad_layer.append("%s: available_in[%r] has values %s not in %s"
                                     % (mtag, mname, sorted(unknown),
                                        sorted(mode_names[mname])))
        for so in spec.inputs:
            tag = "%s.%s" % (spec.op_key, so.name)
            # (a) available_in must reference modes and VALUES that exist — a typo here
            #     silently hides the socket in every state, which is invisible until a
            #     user wonders where a control went.
            for mname, allowed in (so.available_in or {}).items():
                if mname not in mode_names:
                    bad_layer.append("%s: available_in names mode %r that does not exist"
                                     % (tag, mname))
                    continue
                unknown = set(allowed) - mode_names[mname]
                if unknown:
                    bad_layer.append("%s: available_in[%r] has values %s not in %s"
                                     % (tag, mname, sorted(unknown),
                                        sorted(mode_names[mname])))
            # (a2) a filesystem path must declare `path_kind` (V2.15). The inspector
            #      draws its Browse… button off that declaration ALONE, so a path socket
            #      without it silently demands a hand-typed absolute path — the exact bug
            #      `sd_model_path`/`model_path` had while the GUI matched the literal
            #      socket name "path". Name-shaped heuristic on purpose: the point is to
            #      catch the NEXT path socket, whose author will not have read this.
            if so.type is SocketType.STRING and so.direction is Direction.IN:
                looks_path = any(w in so.name for w in ("path", "dir", "folder", "file"))
                if looks_path and not so.path_kind:
                    bad_layer.append(
                        "%s: looks like a filesystem path but declares no path_kind — "
                        "the GUI cannot offer Browse…" % tag)
                if so.path_kind:
                    n_path += 1
                    if not so.path_hint:
                        bad_layer.append(
                            "%s: path socket with no path_hint — the placeholder is where "
                            "an EMPTY value's meaning is stated" % tag)
                    if so.path_kind != "directory" and not so.path_filter:
                        bad_layer.append(
                            "%s: file-kind path socket with no path_filter" % tag)
            if not (so.layer_in or so.layer_in_mode or so.layer_out):
                continue
            if so.layer_in is not None or so.layer_in_mode:
                n_in += 1          # both reader forms: fixed domain, or mode-resolved
            # (b) a layer name is a string
            if so.type is not SocketType.STRING:
                bad_layer.append("%s: a layer socket must be STRING, is %s"
                                 % (tag, so.type))
            # (c) a mode-resolved domain must name a real mode
            if so.layer_in_mode and so.layer_in_mode not in mode_names:
                bad_layer.append("%s: layer_in_mode=%r names no such mode"
                                 % (tag, so.layer_in_mode))
            # (d) reading a layer in domain D means the node READS D. Exempt when the
            #     socket is mode-gated: the requirement is then conditional and
            #     `reads_domains` has no per-mode form (the tessellate / track.link
            #     case, whose empty declaration is deliberate and documented).
            if so.layer_in is not None:
                if so.layer_in not in spec.reads_domains and not so.available_in:
                    bad_layer.append(
                        "%s: reads a %s layer but reads_domains=%s"
                        % (tag, so.layer_in.value,
                           sorted(d.value for d in spec.reads_domains)))
            # (e) writing a layer in domain D means the node ADDS D — unconditionally,
            #     since the write happens whenever the socket is active.
            if so.layer_out:
                n_out += 1
                miss = set(so.layer_out) - set(spec.adds_domains)
                if miss:
                    bad_layer.append(
                        "%s: writes %s but adds_domains=%s"
                        % (tag, sorted(d.value for d in miss),
                           sorted(d.value for d in spec.adds_domains)))
            # (f) the default must live ONLY in the SocketSpec. A compute that still
            #     spells its own fallback inline (`ctx.params.get("mask", "mask")`) has
            #     a second copy that can drift from the declaration — and a third, in
            #     propagate_meta's prediction. `ctx.layer(name)` reads the declaration.
            src = _compute_source(NN, fn)
            if src and ('params.get("%s"' % so.name) in src:
                bad_layer.append(
                    "%s: compute reads params.get(%r) directly — use ctx.layer(%r) so "
                    "the default is declared once" % (tag, so.name, so.name))

    assert not bad_layer, "layer-socket contract violations:\n  " + "\n  ".join(bad_layer)
    # V2.12: `analysis.watershed` + `detect.stardist_nuclei` folded into
    # `analysis.segment` - the catalog lost one layer_in and two layer_out
    # sockets and gained one of each.
    assert n_in >= 18 and n_out >= 18, \
        "only %d layer_in / %d layer_out sockets — a declaration was lost" % (n_in, n_out)
    # V2.15: analysis.segment's two model paths (io.load's `path` lives in the GUI layer's
    # `nodelab_v2.ops`, out of this catalog scan — the phase-5 GUI probe covers that one).
    assert n_path >= 2, "only %d path_kind sockets — a declaration was lost" % n_path

    # The resolver MUST see through every indirection the catalog uses, or this guard
    # passes vacuously (nothing looks read, nothing looks declared-and-unread).
    assert "radius" in keys_of("_compute_median"), "helper-forwarded literal not resolved"
    assert "radius_z" in keys_of("_compute_median"), "suffixed `name + _z` key not resolved"
    assert "opening_radius" in keys_of("_compute_histogram_threshold"), \
        "transitively-forwarded key (_rad -> _cleanup_um) not resolved"
    assert "emission_nm" in keys_of("_compute_deconvolve"), \
        "ctx.channel(c).param(...) read not resolved"
    assert "min_radius_z" in keys_of("_compute_spots"), \
        "ctx.params[\"X\"] subscript read not resolved"

    # Raised from 54 to the real count in V2.22. It had sat at 54 since V2.12 while the
    # catalog grew past 70, i.e. slack by ~16 — a floor that far below actual cannot
    # catch a lost node, which is the only thing it exists to do.
    assert audited >= 70, f"only {audited} catalog computes audited — index broke"
    _ok("socket contract: all %d catalog node types — (1) every param a compute reads "
        "has a socket, (2) every socket is read (helper-forwarded, suffixed `_z`, "
        "transitive, per-channel and ctx.layer keys resolved), (3) %d layer_in + %d "
        "layer_out sockets are STRING, domain-consistent with reads/adds_domains "
        "(mode-gated exempt), mode-resolvable, available_in-valid, and declare their "
        "default ONCE, (4) %d path sockets declare path_kind + hint (+ filter for file "
        "kinds) and no path-shaped socket is left un-browsable; %d documented alias "
        "exemption"
        % (audited, n_in, n_out, n_path, len(_PARAM_NO_SOCKET_OK)))



# ── V2.11: the edit-time layer catalog (what the GUI layer picker offers) ──────

def _ds_layer_names(ds, domain):
    """The layer names a REALIZED Dataset carries in *domain*.

    The projection differs by domain family and that asymmetry is the whole reason
    ``MetaEnvelope.layer_names`` exists as its own field: ``with_layer`` leaves
    ``layer=None``, so a lattice layer's user-facing name is the attribute NAME, while
    ``with_structure`` files each column under the table's LAYER."""
    from nodegraph.domains import is_lattice
    return {(a.name if is_lattice(domain) else a.layer)
            for a in ds.layers_on(domain)}


def test_layer_catalog() -> None:
    """``propagate_meta`` predicts the layer names a Dataset will carry, and the picker
    reads that prediction (V2.11).

    The prediction is a THIRD copy of every output-layer default — the socket default,
    the compute's inline fallback, and now the envelope rule — because ``propagate_meta``
    sees RAW params (the engine never default-fills them; they are overrides, which is
    why each compute repeats its default inline). Nothing structural keeps those three in
    step, so this group pins them by pulling the real nodes and comparing the prediction
    against the layers the payload actually has."""
    if not _HAVE_SKIMAGE:
        _ok("layer catalog (V2.11): SKIPPED (scipy/skimage absent)")
        return
    from nodegraph.provider import ArrayProvider
    from nodegraph.nodes import COMPUTES
    from nodegraph.metadata import propagate_meta

    ax = AxisSizes(m=1, t=2, z=1, c=1, y=16, x=16)
    yy, xx = np.mgrid[0:16, 0:16]
    img = np.zeros((1, 2, 1, 1, 16, 16), dtype=float)
    for t in range(2):
        img[0, t, 0, 0] = np.sin(yy * 0.6) + np.cos(xx * 0.4) + 2.0
    img[0, :, 0, 0, 6:10, 6:10] += 30.0
    optics = {"pixel_size_um": 0.1, "z_step_um": 0.3, "dt_s": 1.0}
    ds = Dataset(axes=ax, metadata=optics).with_image(ArrayProvider(img))
    define_node("io.seedLC", "Seed", outputs=[OutDataset()])
    seedenv = MetaEnvelope(axes=ax, metadata=optics)

    # A chain using NON-DEFAULT names throughout: a default-named chain would pass even
    # if the rule ignored params entirely.
    chain = [("T", "analysis.threshold", {}, {"threshold": 20.0, "name": "m2"}),
             ("L", "analysis.label", {"dim": "2D"}, {"mask": "m2", "name": "regions"}),
             ("E", "analysis.extract_boundary", {"dim": "2D"}, {"labels": "regions"}),
             ("D", "align.drift", {}, {})]
    g = Graph(); g.add(NodeInstance("S", "io.seedLC")); prev = "S"
    for nid, op, modes, params in chain:
        g.add(NodeInstance(nid, op, modes=modes, params=params))
        g.connect(prev, nid); prev = nid
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds}, meta_seeds={"S": seedenv})
    envs = propagate_meta(g, {"S": seedenv})

    # PREDICTION == REALITY, per node, per domain — the three-copies invariant.
    for nid, op, _m, _p in chain:
        out = eng.pull(nid)
        env = envs[nid]
        for dom in (D.VOXEL, D.LABEL, D.POINT, D.FRAME):
            predicted = set(env.layers_in(dom))
            actual = _ds_layer_names(out, dom)
            assert predicted <= actual, (
                f"{op}: predicted {dom.value} layers {sorted(predicted - actual)} "
                f"that the payload does not have (actual {sorted(actual)})")

    # the renamed layers really are what flows (not the hardcoded defaults)
    assert set(envs["L"].layers_in(D.VOXEL)) == {"m2", "regions"}
    assert set(envs["L"].layers_in(D.LABEL)) == {"regions"}, "one socket, TWO domains"
    # a DERIVED name (empty `name` ⇒ f"{labels}_boundary") is predicted too
    assert envs["E"].layers_in(D.POINT) == ("regions_boundary",)
    # a producer with NO socket at all still registers (extra_layers)
    assert set(envs["D"].layers_in(D.FRAME)) == {"drift_y", "drift_x"}

    # NON-MONOTONE: an axis-changing node DROPS the lattice layers whose shape it
    # invalidates (Dataset.reshaped_axes(drop_stale=True)) and keeps the rest. Without
    # this the picker would offer a mask a downstream Crop had already destroyed.
    g2 = Graph(); g2.add(NodeInstance("S", "io.seedLC"))
    g2.add(NodeInstance("T", "analysis.threshold", params={"threshold": 20.0}))
    g2.add(NodeInstance("P", "detect.spots", modes={"dim": "2D"},
                        params={"min_radius": 0.05, "max_radius": 0.4,
                                "threshold": 0.08, "name": "blobs"}))
    g2.add(NodeInstance("C", "util.crop", modes={"dim": "2D"},
                        params={"y0": 0, "y1": 8, "x0": 0, "x1": 8}))
    g2.connect("S", "T"); g2.connect("T", "P"); g2.connect("P", "C")
    e2 = propagate_meta(g2, {"S": seedenv})
    assert "mask" in e2["P"].layers_in(D.VOXEL)
    assert e2["C"].layers_in(D.VOXEL) == (), "crop must drop the y/x-shaped Voxel layers"
    assert e2["C"].layers_in(D.POINT) == ("blobs",), "structure layers survive a crop"
    # It is the AXES that decide, not the node identity: crop changes y/x but leaves m/t
    # alone, so a FRAME layer (shaped by m,t) survives the very same crop that dropped
    # the Voxel ones. Put a drift node above the crop and check both outcomes at once.
    g2b = Graph(); g2b.add(NodeInstance("S", "io.seedLC"))
    g2b.add(NodeInstance("T", "analysis.threshold", params={"threshold": 20.0}))
    g2b.add(NodeInstance("D", "align.drift"))
    g2b.add(NodeInstance("C", "util.crop", modes={"dim": "2D"},
                         params={"y0": 0, "y1": 8, "x0": 0, "x1": 8}))
    g2b.connect("S", "T"); g2b.connect("T", "D"); g2b.connect("D", "C")
    e2b = propagate_meta(g2b, {"S": seedenv})
    assert e2b["C"].layers_in(D.VOXEL) == (), "y/x-shaped Voxel layers drop"
    assert set(e2b["C"].layers_in(D.FRAME)) == {"drift_y", "drift_x"}, \
        "m/t-shaped Frame layers survive the same crop"

    # every layer_in socket names a domain the picker can resolve
    for spec in NODES.all():
        for so in spec.inputs:
            if so.layer_in is None and not so.layer_in_mode:
                continue
            assert so.type is SocketType.STRING, f"{spec.op_key}.{so.name} must be STRING"
            if so.layer_in_mode:
                mode = next((m for m in spec.modes if m.name == so.layer_in_mode), None)
                assert mode is not None, \
                    f"{spec.op_key}.{so.name} names mode {so.layer_in_mode!r} that does not exist"
    # and every layer_out socket is a STRING naming real domains
    n_out = 0
    for spec in NODES.all():
        for so in spec.inputs:
            if not so.layer_out:
                continue
            n_out += 1
            assert so.type is SocketType.STRING, f"{spec.op_key}.{so.name} must be STRING"
            assert all(isinstance(d, Domain) for d in so.layer_out)
    assert n_out >= 18, f"only {n_out} layer_out sockets — a producer lost its declaration"

    # TOTALITY: propagate_meta runs on every keystroke and its GUI caller catches only
    # ValueError, so a junk param must degrade, never raise.
    g3 = Graph(); g3.add(NodeInstance("S", "io.seedLC"))
    g3.add(NodeInstance("X", "analysis.threshold", params={"name": None}))
    g3.add(NodeInstance("Y", "analysis.label", params={"name": 17, "mask": ["not", "a", "str"]}))
    g3.connect("S", "X"); g3.connect("X", "Y")
    e3 = propagate_meta(g3, {"S": seedenv})          # must not raise
    assert e3["X"].layers_in(D.VOXEL) == ("mask",), "None falls back to the socket default"
    assert "regions" not in e3["Y"].layers_in(D.VOXEL), "a non-string name is skipped"

    _ok("layer catalog (V2.11): propagate_meta predicts the layer names a payload really "
        "carries (checked against real pulls on a fully renamed chain); one socket → two "
        "domains; derived + socket-less producers registered; an axis change DROPS the "
        "invalidated lattice layers and keeps structure; junk params degrade, never raise")


# ── V2.17: the real-ND2 audit repairs (2026-07-30) ────────────────────────────

def test_nd2_audit_repairs() -> None:
    """The seven silent-wrong-number repairs found by driving a REAL 49-position ND2.

    Every one of them was invisible to this suite because its fixtures are single-position
    with a small pixel, and each is checked here at the property that was actually wrong —
    not merely that the code runs.

    1. **Statistics scope** (``analysis.threshold``). A histogram level pooled over the
       whole ``(m,t,z,c)`` product, so a position's mask depended on which OTHER positions
       shared the file (measured: −47% foreground on the third of three real wells). The
       invariant now asserted is the one a user assumes: **thresholding a position alone
       gives the same mask as thresholding it alongside others** — for every histogram
       method, at every scope but the opt-in ``dataset``.
    2. **Degenerate kernels.** A µm radius that quantizes to a 1-voxel window made
       top-hat / morphological-gradient return an identically BLACK image and
       median / morphology return the input untouched. Refused now, with 0 still allowed
       as an explicit request.
    3. **The `raw` measurement socket.** Its guard compared only ``AxisSizes``, so a
       shape-preserving ORIGIN shift (``align.drift``, ``registration.stabilize``) passed
       and intensities were read from where each object used to be.
    4. **``analysis.measure`` on a single plane.** An absent ``z_kind`` defaulted to 3D and
       ran regionprops' 3D walk over one plane → ``solidity = inf``. (Covered in
       ``test_celltracker_parity`` §8, where the shape metrics live.)
    5. **``enhance.deconvolve``** pushed voxels past the declared ``bit_depth`` and kept
       the stale depth stamped.
    6. **``channel.select``** silently produced a c=0 Dataset.
    7. **``plan_transfer``** dropped the caller's ``reducer`` on every bridge hop, and
       ``execute_bridge_plan`` destroyed a structure carrier on an identity plan.
    """
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.structure import StructureTable as _ST

    s = NODES.get("analysis.threshold")
    mode_of = {m.name: m for m in s.modes}
    assert set(mode_of) == {"method", "scope"}
    assert mode_of["scope"].resolved_default() == "plane", \
        "per-plane must be the DEFAULT — the pooled behaviour is the opt-in one"
    assert set(mode_of["scope"].choices) == {"plane", "volume", "series", "dataset"}
    # `scope` is meaningless to a fixed cut, so it is Mode-gated away (§5c)
    amodes = lambda st: {m.name for m in s.active_modes(st)}
    assert amodes({"method": "fixed", "scope": "plane"}) == {"method"}
    assert amodes({"method": "otsu", "scope": "plane"}) == {"method", "scope"}
    # the footprint follows the SCOPE, not a dim lever this node does not have
    assert s.footprint_mode == "scope"
    assert s.resolve_granularity({"scope": "plane"}) is Granularity.WHOLE_PLANE
    assert s.resolve_granularity({"scope": "series"}) is Granularity.WHOLE_SERIES
    assert s.resolve_granularity({"scope": "dataset"}) is Granularity.MULTI_VIEW, \
        "pooling across multipoints IS a multi-view read; TILEABLE was a misdeclaration"
    assert NODES.get("enhance.gaussian").footprint_mode == DIM_MODE, \
        "footprint_mode must default to the dim lever for every existing levered node"

    if not _HAVE_SKIMAGE:
        _ok("nd2 audit repairs: spec OK; RUN SKIPPED (skimage absent)")
        return

    # ── fixture: THREE positions at deliberately different brightness ──────────
    # This is the shape of the real file (a plate: one well per m, exposures differing),
    # and it is exactly what a single-position fixture cannot express.
    Y = X = 32
    rng = np.arange(Y * X, dtype=float).reshape(Y, X) / (Y * X)
    img = np.zeros((3, 2, 1, 1, Y, X), dtype=np.uint16)
    for m, gain in enumerate((1.0, 3.0, 9.0)):
        for t in range(2):
            plane = 100.0 + 900.0 * gain * rng + 40.0 * t
            plane[8:16, 8:16] += 1500.0 * gain          # an object to threshold
            img[m, t, 0, 0] = np.clip(plane, 0, 4095).astype(np.uint16)
    # 1.7183 µm/px — the real plate's pixel, and the whole reason (2) exists
    optics = {"pixel_size_um": 1.7182777601481225, "bit_depth": 12}
    ax3 = AxisSizes(m=3, t=2, z=1, c=1, y=Y, x=X)
    define_node("io.auditseed", "Seed", outputs=[OutDataset()])

    def _thresh(vol, method, scope):
        a = np.ascontiguousarray(vol)
        axn = AxisSizes(m=a.shape[0], t=a.shape[1], z=1, c=1, y=Y, x=X)
        dsn = Dataset(axes=axn, metadata=dict(optics)).with_image(ArrayProvider(a))
        g = Graph(); g.add(NodeInstance("S", "io.auditseed"))
        g.add(NodeInstance("N", "analysis.threshold",
                           modes={"method": method, "scope": scope}))
        g.connect("S", "N")
        e = Engine(g, computes=COMPUTES, seeds={"S": dsn},
                   meta_seeds={"S": MetaEnvelope(axes=axn, metadata=optics)})
        return np.asarray(e.pull("N").get(D.VOXEL, "mask").values)

    # (1) THE invariant: a position's mask must not depend on its neighbours.
    for method in ("otsu", "li", "yen", "triangle", "mean"):
        for scope in ("plane", "volume", "series"):
            together = _thresh(img, method, scope)
            for m in range(3):
                alone = _thresh(img[m:m + 1], method, scope)
                assert np.array_equal(together[m], alone[0]), (
                    f"{method}/{scope}: position {m}'s mask changed when other positions "
                    f"shared the Dataset — the histogram population leaked across m")
    # ...and `dataset` deliberately still pools, so the old behaviour stays reachable.
    pooled = _thresh(img, "otsu", "dataset")
    solo = _thresh(img[2:3], "otsu", "dataset")
    assert not np.array_equal(pooled[2], solo[0]), \
        "scope='dataset' must keep pooling across m — it is the opt-in legacy behaviour"
    # each scope is its own memo key
    hashes = set()
    for scope in ("plane", "volume", "series", "dataset"):
        g = Graph(); g.add(NodeInstance("S", "io.auditseed"))
        g.add(NodeInstance("N", "analysis.threshold",
                           modes={"method": "otsu", "scope": scope}))
        g.connect("S", "N")
        ds3 = Dataset(axes=ax3, metadata=dict(optics)).with_image(ArrayProvider(img))
        e = Engine(g, computes=COMPUTES, seeds={"S": ds3},
                   meta_seeds={"S": MetaEnvelope(axes=ax3, metadata=optics)})
        e.pull("N")
        hashes.add(e.entry("N").recipe_hash)
    assert len(hashes) == 4, f"each scope must re-key the memo, got {len(hashes)}"

    # (2) a µm radius that rounds to a 1-voxel window is REFUSED, not silently degenerate
    ds3 = Dataset(axes=ax3, metadata=dict(optics)).with_image(ArrayProvider(img))
    env3 = MetaEnvelope(axes=ax3, metadata=optics)

    def _pull1(op, params=None):
        g = Graph(); g.add(NodeInstance("S", "io.auditseed"))
        g.add(NodeInstance("N", op, params=params or {}))
        g.connect("S", "N")
        return Engine(g, computes=COMPUTES, seeds={"S": ds3},
                      meta_seeds={"S": env3}).pull("N")

    for op in ("enhance.tophat", "enhance.morphological_gradient",
               "enhance.median", "enhance.morphology"):
        try:
            _pull1(op)                       # SHIPPED defaults, at 1.72 µm/px
            raise AssertionError(f"{op} accepted a sub-pixel radius that yields a "
                                 f"1-voxel window (it used to return a blank / unchanged "
                                 f"image silently)")
        except ValueError as exc:
            assert "1-voxel" in str(exc) and "µm" in str(exc), exc
        out = _pull1(op, {"radius": 8.0})    # sized for THIS pixel: works
        got = out.image.read_region(0, 0, 0, 0, 0, 0, Y, 0, X)
        assert np.isfinite(got).all() and got.shape == (Y, X)
        _pull1(op, {"radius": 0.0})          # an EXPLICIT no-op stays legal

    # (3) `raw` must refuse a shape-preserving ORIGIN shift, and accept a pure enhancement
    shifted = Dataset(axes=ax3, metadata=dict(optics)).with_image(
        ArrayProvider(np.ascontiguousarray(np.roll(img, 5, axis=5))))
    shifted = shifted.with_metadata(**{"__sampling__": ("align.drift",)})

    def _measure_with_raw(raw_ds):
        g = Graph()
        g.add(NodeInstance("S", "io.auditseed"))
        g.add(NodeInstance("R", "io.auditseed"))
        g.add(NodeInstance("T", "analysis.threshold", modes={"method": "otsu"}))
        g.add(NodeInstance("L", "analysis.label", modes={"dim": "2D"}))
        g.add(NodeInstance("Q", "analysis.measure"))
        g.connect("S", "T"); g.connect("T", "L"); g.connect("L", "Q")
        g.connect("R", "Q", dst_socket="raw")
        return Engine(g, computes=COMPUTES, seeds={"S": ds3, "R": raw_ds},
                      meta_seeds={"S": env3, "R": env3}).pull("Q")

    _measure_with_raw(ds3)                   # same sampling → fine
    try:
        _measure_with_raw(shifted)
        raise AssertionError("`raw` accepted a Dataset with a DIFFERENT sampling geometry "
                             "— equal AxisSizes is not equal geometry")
    except ValueError as exc:
        assert "sampling geometry" in str(exc), exc
    # the stamps are real: a geometry node writes one, an enhancement node does not
    from nodegraph.nodes import SAMPLING_KEY
    cropped = _pull1("util.crop", {"y0": 0, "y1": 16, "x0": 0, "x1": 16})
    assert cropped.metadata.get(SAMPLING_KEY), "util.crop must stamp its sampling change"
    blurred = _pull1("enhance.gaussian", {"sigma": 4.0})
    assert not blurred.metadata.get(SAMPLING_KEY), \
        "an enhancement must NOT stamp — 'segment on enhanced, measure on raw' depends on it"

    # (5) deconvolve leaves the declared integer scale behind, header and payload agreeing
    dec = _pull1("enhance.deconvolve")
    assert "bit_depth" not in dec.metadata, \
        "Richardson-Lucy exceeds the input range, so the stale depth must be dropped"
    gd = Graph(); gd.add(NodeInstance("S", "io.auditseed"))
    gd.add(NodeInstance("N", "enhance.deconvolve")); gd.connect("S", "N")
    assert "bit_depth" not in propagate_meta(gd, {"S": env3})["N"].metadata, \
        "the meta_transform must predict the drop too (both halves, §7c)"

    # (6) a channel selection that matches nothing is refused, not silently emptied
    try:
        _pull1("channel.select", {"channels": "7"})
        raise AssertionError("channel.select accepted a selection matching no channel")
    except ValueError as exc:
        assert "selects no channel" in str(exc), exc
    assert _pull1("channel.select", {"channels": "0"}).axes.c == 1
    assert _pull1("channel.select", {}).axes.c == 1          # empty = keep everything

    # (7) the transfer library: the reducer reaches a bridge hop, identity preserves ids
    vox = np.array([[9.0, 9.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    raster = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int32)
    from nodegraph.transfer import Carrier, execute_bridge_plan
    plan_max = plan_transfer(D.VOXEL, D.LABEL, reducer="max")
    assert plan_max.steps[0].reducer == "max", "the caller's reducer never reached the hop"
    got_max = execute_bridge_plan(Carrier(D.VOXEL, array=vox), plan_max,
                                  label_raster=raster).values
    assert np.allclose(got_max, [9.0]), (got_max, "requested max, executed something else")
    got_mean = execute_bridge_plan(Carrier(D.VOXEL, array=vox),
                                   plan_transfer(D.VOXEL, D.LABEL),
                                   label_raster=raster).values
    assert np.allclose(got_mean, [5.0]), got_mean            # the default is unchanged
    for dom in (D.LABEL, D.POINT, D.TRACK, D.MESH):
        ident = execute_bridge_plan(
            Carrier(dom, ids=np.array([1, 2]), values=np.array([4.0, 8.0])),
            plan_transfer(dom, dom))
        assert ident.ids is not None and np.allclose(ident.values, [4.0, 8.0]), \
            f"{dom.value}->{dom.value} identity destroyed its structure payload"

    # (0) THE headline one: a RATE with no frame interval must refuse, never divide by 1.0.
    # `dt_s` was absent from every ND2 (the reader emitted timestamps and ingest dropped
    # them), so `speed` came back in µm/FRAME under a µm/s label — ×1424.6 on the real
    # 6-hour file, i.e. cells reported at 9.9 mm/h. Checked on both nodes that do it.
    lab6 = np.zeros((1, 2, 1, 1, Y, X), dtype=np.int64)
    lab6[0, 0, 0, 0, 4:8, 4:8] = 1
    lab6[0, 1, 0, 0, 4:8, 10:14] = 2                     # the same object, moved +6 px in x
    mv_cols = {"id": np.array([1, 2], np.int64), "m": np.zeros(2, np.int64),
               "t": np.array([0, 1], np.int64), "c": np.zeros(2, np.int64),
               "z": np.zeros(2), "y": np.array([5.5, 5.5]), "x": np.array([5.5, 11.5]),
               "track_id": np.array([1, 1], np.int64)}
    ax_mv = AxisSizes(m=1, t=2, z=1, c=1, y=Y, x=X)
    ds_mv = (Dataset(axes=ax_mv, metadata=dict(optics))
             .with_image(ArrayProvider(img[:1, :2]))
             .with_layer(D.VOXEL, "labels", lab6)
             .with_structure(_ST(D.LABEL, mv_cols, layer="labels", z_kind="plane_index")))
    env_mv = MetaEnvelope(axes=ax_mv, metadata=optics)          # NOTE: no dt_s

    def _metrics(seed_ds, seed_env, params=None, op="analysis.object_metrics"):
        g = Graph(); g.add(NodeInstance("S", "io.auditseed"))
        g.add(NodeInstance("N", op, params=params or {}))
        g.connect("S", "N")
        e = Engine(g, computes=COMPUTES, seeds={"S": seed_ds}, meta_seeds={"S": seed_env})
        return e, e.pull("N")

    for op, ask in (("analysis.object_metrics", {"metrics": "speed"}),
                    ("analysis.object_field", {"fields": "speed"})):
        try:
            _metrics(ds_mv, env_mv, ask, op)
            raise AssertionError(f"{op} reported a RATE with no frame interval — it used "
                                 f"to divide by a placeholder 1.0 and label the result µm/s")
        except ValueError as exc:
            assert "frame interval" in str(exc), exc
    # a non-rate metric needs no interval and must still work
    _metrics(ds_mv, env_mv, {"metrics": "neighbors"})
    # with dt_s present the speed is physical: 6 px × 1.7183 µm over 2 s = 5.15 µm/s
    env_dt = MetaEnvelope(axes=ax_mv, metadata={**optics, "dt_s": 2.0})
    ds_dt = ds_mv.with_metadata(dt_s=2.0)
    e_dt, out_dt = _metrics(ds_dt, env_dt, {"metrics": "speed"})
    spd = np.asarray(out_dt.get(D.LABEL, "speed", layer="labels").values, dtype=float)
    expect = 6.0 * optics["pixel_size_um"] / 2.0
    # row 0 is the track's FIRST frame — no prior position, so NaN by design, not 0
    assert np.isnan(spd[0]) and np.allclose(spd[1], expect, rtol=1e-9), (spd, expect)
    assert "dt_s" in {k for k, _ in e_dt.entry("N").reads}, "speed not dt_s-fenced"
    # ...and the socket overrides it, for data whose file carries no timing at all
    _, out_ov = _metrics(ds_mv, env_mv, {"metrics": "speed", "frame_interval": 4.0})
    spd_ov = np.asarray(out_ov.get(D.LABEL, "speed", layer="labels").values, dtype=float)
    assert np.allclose(spd_ov[1], 6.0 * optics["pixel_size_um"] / 4.0), spd_ov
    assert np.allclose(spd_ov[1], spd[1] / 2.0), \
        "the override must scale the rate linearly — it IS the divisor"

    _ok("nd2 audit repairs (V2.17): a RATE with no frame interval is REFUSED on both "
        "object nodes (it used to divide by 1.0 and label µm/FRAME as µm/s), dt_s is "
        "memo-fenced and gives the exact physical speed, the frame_interval socket "
        "overrides it, and a non-rate metric still needs none; threshold scope — every "
        "plane/volume/series gives a position the SAME mask alone as in company (the "
        "cross-position leak), `dataset` still pools, 4 distinct recipe keys, footprint "
        "resolves off `scope` not a dim lever; 4 nodes refuse a µm radius that rounds to a "
        "1-voxel window (0 still allowed) and work at a radius sized for the pixel; the "
        "`raw` socket refuses an equal-size ORIGIN shift and geometry nodes stamp sampling "
        "provenance while enhancements do not; deconvolve drops bit_depth in payload AND "
        "header; channel.select refuses an empty selection; plan_transfer honours a "
        "non-default reducer on a bridge hop and a structure identity plan keeps its ids")


def test_nd2_ingest_calibration() -> None:
    """The ND2 seam must hand the engine calibration that DESCRIBES THE FILE.

    Two ways it did not, both found on the lab's real WellA3 plate and both invisible to
    every synthetic fixture (which supplies its calibration by hand):

    * ``dt_s`` was **never emitted for any ND2**. ``read_nd2_metadata_extended`` computes
      ``frame_timestamps_s`` correctly and ``read_calibration`` dropped it, because ``dt_s``
      is a CALIBRATION key and the timestamps are not. Every velocity in the app was
      therefore µm/frame wearing a µm/s label.
    * ``z_step_um`` was **fabricated as 1.0** for a file with no Z axis at all (the SDK's
      ``voxel_size().z`` defaults to 1.0), and a real-looking 1.0 satisfies every
      ``or <fallback>`` guard in a ``derive``.

    Pure arithmetic on the reader's own dict, so no ND2 file is needed here."""
    from nodelab_v2.ingest import _dt_from_timestamps

    # the real file's 16 stamps, verbatim (relativeTimeMs / 1000)
    real = [1994.8, 3412.5, 4832.6, 6259.3, 7677.0, 9098.2, 10522.8, 11945.0,
            13376.4, 14807.8, 16231.7, 17655.1, 19087.5, 20515.2, 21945.5, 23374.8]
    dt = _dt_from_timestamps({"frame_timestamps_s": real})
    assert dt is not None and 1420.0 < dt < 1430.0, dt
    # MEDIAN, not mean: one dropped frame must not move the number every rate divides by
    gap = real[:8] + [v + 5000.0 for v in real[8:]]
    assert abs(_dt_from_timestamps({"frame_timestamps_s": gap}) - dt) < 1.0,         "a single long gap moved dt_s — use the median of the differences, not the mean"
    # absent is the honest answer, never a guessed 1.0
    for bad in ({}, {"frame_timestamps_s": []}, {"frame_timestamps_s": [7.0]},
                {"frame_timestamps_s": [5.0, 5.0]}):
        assert _dt_from_timestamps(bad) is None, bad
    _ok("nd2 ingest calibration (V2.17): dt_s derived from the real file's 16 frame "
        "timestamps (1424.6 s) by MEDIAN difference so a dropped frame cannot move it, "
        "and absent — never a fabricated 1.0 — when the stamps cannot answer")


def test_nd2_zstack_home_index_guard() -> None:
    """The ``nd2`` SDK shim that lets a zero-range Z-stack be OPENED at all (2026-08-03).

    ``LAGFP_Caps_Z_12hrs.nd2`` (6.08 GB, T=145 × P=5 × 2048², 12-bit, otherwise healthy)
    could not be loaded: its ``ZStackLoop`` is a stack configured in ND Acquisition but
    acquired at a single plane, so ``dZLow == dZHigh == -25.2`` and ``dZStep == 0.0``, and
    ``nd2._parse._parse._calc_zstack_home_index`` divides by ``abs(high_um - low_um)``.
    ``ZeroDivisionError`` out of ``ND2File.experiment`` → out of ``ND2File.sizes``, which is
    the first thing ``read_meta_only`` touches — so the File menu died at pick time, before
    a pixel was read. nd2 0.11.3 is the newest release, so there is nothing to upgrade to.

    What this pins, on the real file's loop parameters and with no ND2 file needed:

    * the shim is INSTALLED by ``import_nd2`` (a bare ``import nd2`` is the bug), and
    * it answers 0 for the collapsed stack — the value the formula itself was reaching for,
      since its numerator ``(count - 1) * hrange`` is 0 when ``count == 1`` — and
    * a HEALTHY multi-plane stack is byte-identical before and after. That last one is the
      whole licence for patching somebody else's parser: the wrapper delegates first and
      only acts on the raised ``ZeroDivisionError``, so no file that loads today can shift.

    Skips if ``nd2`` is absent (ingest's TIFF half does not need it), and self-sunsets: if a
    future release carries the guard, upstream stops raising and every assert still holds.
    """
    try:
        import nd2  # noqa: F401
        from nd2._parse import _parse
    except ImportError:
        _ok("nd2 zstack home index guard: SKIPPED (nd2 absent)")
        return
    from nodelab_v2.nd2_compat import import_nd2, _SHIM_TAG

    # the real file's ZStackLoop, verbatim: inverted, count, iType, dZHome, dZLow, dZHigh, step
    COLLAPSED = (False, 1, 7, 1.8, -25.2, -25.2, 0.0)
    # a healthy 210-slice stack (the WellA3 640 series) — must be untouched by the shim
    HEALTHY = (False, 210, 7, 5971.96, 5971.96, 6032.08, 0.288)

    import_nd2()
    fn = _parse._calc_zstack_home_index
    assert getattr(fn, _SHIM_TAG, False), "import_nd2 did not install the zstack shim"
    assert fn(*COLLAPSED) == 0, fn(*COLLAPSED)
    # idempotent: a second call must not stack a wrapper on the wrapper
    import_nd2()
    assert _parse._calc_zstack_home_index is fn, "the zstack shim double-wrapped"
    # and the shim is inert wherever upstream can answer — including the OTHER zero-range
    # shapes, which must reach upstream's own branches rather than the except
    for pars in (HEALTHY,
                 (True, 210, 7, 5971.96, 5971.96, 6032.08, 0.288),   # inverted
                 (False, 210, 2, 5971.96, 5971.96, 6032.08, 0.288),  # iType 2 branch
                 (False, 12, 0, 1.0, 0.0, 0.0, 0.0)):                # iType 0 early return
        assert fn(*pars) == fn.__wrapped__(*pars), pars
    # the bug itself, so this test fails loudly if the premise ever changes
    try:
        fn.__wrapped__(*COLLAPSED)
        raised = False
    except ZeroDivisionError:
        raised = True
    if not raised:                       # a fixed nd2 — the shim is now a no-op, as designed
        _ok("nd2 zstack home index guard: nd2 no longer divides by a zero Z range — the "
            "shim is inert and can be retired (nodelab_v2/nd2_compat.py)")
        return
    _ok("nd2 zstack home index guard: a single-plane ZStackLoop (dZLow == dZHigh == -25.2, "
        "step 0) no longer raises ZeroDivisionError out of ND2File.sizes — home index 0, "
        "installed by import_nd2, idempotent, and provably inert on a healthy 210-slice "
        "stack and on upstream's other branches")


def test_nd3_calibration_mapping() -> None:
    """The ND3 seam's honesty rules, on plain dicts — no file, no h5py.

    Every rule here is the difference between a correct measurement and a
    plausible-looking wrong one on a real MEBP export:

    * **the mosaic pitfall** (ND3 spec §8.5): a plate-mosaic canvas is
      downscaled from camera resolution, so reading ``captured_um_per_px``
      instead of the image's own pitch mis-scales every µm figure ~75×;
    * **sensor vs container depth**: the array dtype is the CONTAINER's depth
      — a uint16 canvas of a 12-bit sensor — so ``bit_depth`` comes from the
      acquisition record or stays absent, never from the dtype;
    * **placement comes from the stored matrix only** (§8.2/§8.4): the matrix
      is the one field guaranteed registration-shift-subtracted, and it is
      trusted only when the file says the shift is known;
    * **alphabetical ids are not channel order**: ``ND3Reader.image_ids()``
      sorts alphabetically, so ``Bright_Field`` (filter channel 4) precedes
      ``DAPI`` (channel 1) — ingesting in id order puts DAPI pixels under a
      Bright_Field label while looking entirely normal.
    """
    from nodelab_v2.nd3_ingest import (calibration_from_metas, split_fragment,
                                       stack_order)

    mat = {"transforms": {"pixel_to_stage_um": [[50.0, 0, 100.0],
                                                [0, 50.0, 200.0], [0, 0, 1]]}}

    # scale: the matrix diagonal is authoritative; um_per_px is the fallback;
    # captured_um_per_px (the CAMERA pitch) is never read even when present
    c = calibration_from_metas([{**mat, "scale": {
        "um_per_px": 49.0, "captured_um_per_px": 0.65}}],
        stacked=False, c_count=1)
    assert c["pixel_size_um"] == 50.0, c
    c = calibration_from_metas([{"scale": {
        "um_per_px": 50.0, "captured_um_per_px": 0.65,
        "mosaic_scale_px_per_um": 0.02}}], stacked=False, c_count=1)
    assert c["pixel_size_um"] == 50.0 and "origin_um" not in c, c

    # bit depth: leading integer of the SENSOR record; absent stays absent
    c = calibration_from_metas([{"acquisition": {"bit_depth": "12-bit"}}],
                               stacked=False, c_count=1)
    assert c["bit_depth"] == 12 and "pixel_size_um" not in c, c
    assert "bit_depth" not in calibration_from_metas(
        [{}], stacked=False, c_count=1)

    # optics: "10x Plan Fluor" parses, junk stays absent, NA passes through
    c = calibration_from_metas([{"acquisition": {
        "magnification": "10x Plan Fluor", "numerical_aperture": 0.45}}],
        stacked=False, c_count=1)
    assert c["objective_magnification"] == 10.0 and c["objective_na"] == 0.45
    assert "objective_magnification" not in calibration_from_metas(
        [{"acquisition": {"objective": "Plan Fluor"}}],
        stacked=False, c_count=1)

    # origin: matrix + shift known → [[focus, oy, ox]]; shift_known false or
    # no matrix → ABSENT (fine for translation-invariant work, never placed)
    honest = {**mat, "planes": [{"t": 0, "c": 0, "z": 0, "focus_um": 12.5}]}
    c = calibration_from_metas([honest], stacked=False, c_count=1)
    assert c["origin_um"] == [[12.5, 200.0, 100.0]], c
    legacy = {**honest, "stage_frame": {"shift_known": False}}
    assert "origin_um" not in calibration_from_metas(
        [legacy], stacked=False, c_count=1), \
        "shift_known:false means the shift is UNKNOWN (up to ~half a field) " \
        "— an origin from it is a guess wearing coordinates"
    assert "origin_um" not in calibration_from_metas(
        [{"planes": honest["planes"]}], stacked=False, c_count=1)

    # stacking order: by channel_number when all unique (Bright_Field is
    # filter 4 — AFTER DAPI despite sorting first alphabetically); id order
    # the moment one is missing — never a mixed sort
    bf = {"channels": [{"name": "Bright Field", "channel_number": 4}]}
    dapi = {"channels": [{"name": "DAPI", "channel_number": 1,
                          "emission_nm": 461}]}
    assert stack_order([("Bright_Field", bf), ("DAPI", dapi)]) == \
        ["DAPI", "Bright_Field"]
    assert stack_order([("Bright_Field", {"channels": [{"name": "BF"}]}),
                        ("DAPI", dapi)]) == ["Bright_Field", "DAPI"]

    # stacked origins: half-a-pixel lateral tolerance. Agreeing → the first
    # sorted channel's; one unknown → ABSENT (never "the channels that know");
    # apart → ValueError, because those are different fields and stacking
    # them would misregister the C axis
    def well(ox, oy, *, known=True, ps=0.65):
        m = {"scale": {"um_per_px": ps},
             "transforms": {"pixel_to_stage_um": [[ps, 0, ox], [0, ps, oy],
                                                  [0, 0, 1]]},
             "channels": [{"name": "x"}]}
        if not known:
            m["stage_frame"] = {"shift_known": False}
        return m

    c = calibration_from_metas([well(100.0, 200.0), well(100.1, 200.2)],
                               stacked=True, c_count=2)
    assert c["origin_um"] == [[0.0, 200.0, 100.0]], c
    c = calibration_from_metas([well(100.0, 200.0),
                                well(100.0, 200.0, known=False)],
                               stacked=True, c_count=2)
    assert "origin_um" not in c, c
    try:
        calibration_from_metas([well(100.0, 200.0), well(150.0, 200.0)],
                               stacked=True, c_count=2)
        raise AssertionError("origins 50 µm apart (77 px) stacked silently")
    except ValueError:
        pass

    # per-channel emission: holes stay None; all-absent stays absent (never
    # fabricated from a fluorophore table — LabLink's missing_metadata refusal
    # is the designed remediation loop)
    c = calibration_from_metas([dapi, bf], stacked=True, c_count=2)
    assert c["channel_emission_nm"] == [461, None], c
    assert "channel_emission_nm" not in calibration_from_metas(
        [bf, bf], stacked=True, c_count=2)

    # dt_s: median of t_iso diffs per distinct T; absent for a single stamp
    # or stamps that cannot be compared (mixed naive/aware)
    lapse = {"planes": [
        {"t": 0, "t_iso": "2026-08-08T10:00:00"},
        {"t": 1, "t_iso": "2026-08-08T10:01:00"},
        {"t": 2, "t_iso": "2026-08-08T10:02:00"},
        {"t": 3, "t_iso": "2026-08-08T10:08:00"}]}   # one long gap
    c = calibration_from_metas([lapse], stacked=False, c_count=1, t_count=4)
    assert c["dt_s"] == 60.0, c
    assert "dt_s" not in calibration_from_metas(
        [{"planes": lapse["planes"][:1]}], stacked=False, c_count=1,
        t_count=4)
    mixed = {"planes": [{"t": 0, "t_iso": "2026-08-08T10:00:00"},
                        {"t": 1, "t_iso": "2026-08-08T10:01:00+00:00"}]}
    assert "dt_s" not in calibration_from_metas(
        [mixed], stacked=False, c_count=1, t_count=2)
    # and z_step_um is NEVER emitted (v1: no MEBP profile writes Z stacks;
    # a fabricated spacing feeds every µm³ figure downstream)
    assert "z_step_um" not in c

    # the #image_id fragment: honored only when it can be meant as one
    assert split_fragment("Q:\\nowhere\\well.nd3#DAPI") == \
        ("Q:\\nowhere\\well.nd3", "DAPI")
    assert split_fragment("Q:\\nowhere\\photo.png#frag") == \
        ("Q:\\nowhere\\photo.png#frag", None)

    _ok("nd3 calibration mapping: pitch from the stored matrix (never the "
        "camera's captured_um_per_px — the §8.5 mosaic pitfall), bit depth "
        "from the sensor record (never the container dtype), origin only "
        "when the shift is known, channel_number ordering over alphabetical "
        "ids, stacked origins honest to half a pixel, and absent — never "
        "fabricated — everywhere the file does not say")


def test_nd3_ingest_roundtrip() -> None:
    """End-to-end .nd3 ingest against fixtures written with RAW h5py.

    Raw h5py, not the vendored ND3 writer, deliberately: writer→reader through
    one module only proves self-consistency; these fixtures are built straight
    from the spec (ND3_SPEC.md §2/§3), so the test is a conformance check on
    the reading side. Skips when h5py is absent — the rest of the ingest seam
    (ND2/TIFF) does not need it."""
    try:
        import h5py
    except ImportError:
        _ok("nd3 ingest roundtrip: SKIPPED (h5py absent)")
        return
    import json
    import os
    import tempfile

    from nodelab_v2 import nd3 as nd3mod
    from nodelab_v2.ingest import _is_nd3, ingest_image, read_meta_only
    from nodelab_v2.nd3_ingest import nd3_meta_only, read_nd3

    # the vendor-drift canary: a re-vendored ND3.py with a new MAJOR must not
    # land silently (readers refuse newer majors, so this repo would start
    # refusing files the lab's MEBP still writes — or vice versa)
    assert nd3mod.SCHEMA_VERSION == "1.0", nd3mod.SCHEMA_VERSION

    tmp = tempfile.mkdtemp(prefix="nd3_selftest_")

    def write(name, images, dataset_json=None, fmt="nd3", schema="1.0"):
        path = os.path.join(tmp, name)
        with h5py.File(path, "w") as f:
            f.attrs["format"] = fmt
            f.attrs["schema_version"] = schema
            f.attrs["created_iso"] = "2026-08-08T10:00:00"
            if dataset_json is not None:
                f.create_dataset("dataset_json", data=np.frombuffer(
                    json.dumps(dataset_json).encode("utf-8"), dtype=np.uint8))
            g = f.create_group("images")
            for image_id, (arr, axes, pixel_format, meta) in images.items():
                gi = g.create_group(image_id)
                ds = gi.create_dataset("data", data=arr)
                ds.attrs["axes"] = axes
                ds.attrs["pixel_format"] = pixel_format
                gi.create_dataset("meta_json", data=np.frombuffer(
                    json.dumps(meta).encode("utf-8"), dtype=np.uint8))
        return path

    def chan_meta(name, num, ps=0.65, ox=100.0, oy=200.0):
        return {"scale": {"um_per_px": ps},
                "transforms": {"pixel_to_stage_um": [[ps, 0, ox], [0, ps, oy],
                                                     [0, 0, 1]]},
                "channels": [{"name": name, "channel_number": num}],
                "planes": [{"t": 0, "c": 0, "z": 0, "focus_um": 12.5}],
                "acquisition": {"bit_depth": 12}}

    # 1. a fluor_well twin whose ALPHABETICAL id order (Bright_Field, DAPI)
    #    differs from its channel_number order (DAPI=1, Bright_Field=4)
    dapi_px = np.arange(48, dtype=np.uint16).reshape(6, 8)
    bf_px = dapi_px + 1000
    well = write("well.nd3",
                 {"Bright_Field": (bf_px, "YX", "gray16",
                                   chan_meta("Bright Field", 4)),
                  "DAPI": (dapi_px, "YX", "gray16", chan_meta("DAPI", 1))},
                 dataset_json={"profile": "mebp.fluor_well/1",
                               "plate_id": "plate-24", "well": "B3"})
    axes, calib, disp = read_meta_only(well)     # through the DISPATCH seam
    assert (axes.m, axes.t, axes.z, axes.c, axes.y, axes.x) == (1, 1, 1, 2, 6, 8)
    assert disp["channel_names"] == ["DAPI", "Bright Field"], disp
    assert disp["plate_id"] == "plate-24" and \
        disp["nd3_profile"] == "mebp.fluor_well/1"
    assert calib["origin_um"] == [[12.5, 200.0, 100.0]] and \
        calib["bit_depth"] == 12
    prov, env = ingest_image(well)               # in-memory provider
    r = np.asarray(prov.read_region(0, 0, 0, 0, 0, 0, 6, 0, 8)).reshape(6, 8)
    assert np.array_equal(r, dapi_px), "channel 0 must be DAPI (channel 1), " \
        "not Bright_Field (alphabetically first, filter channel 4)"
    r = np.asarray(prov.read_region(0, 0, 0, 0, 1, 0, 6, 0, 8)).reshape(6, 8)
    assert np.array_equal(r, bf_px)
    assert env.metadata["pixel_size_um"] == 0.65

    # 2. shape mismatch refuses naming the remediation; #id loads one image
    bad = write("bad.nd3",
                {"A": (np.zeros((6, 8), np.uint16), "YX", "gray16",
                       chan_meta("A", 1)),
                 "B": (np.zeros((5, 8), np.uint16), "YX", "gray16",
                       chan_meta("B", 2))})
    try:
        read_meta_only(bad)
        raise AssertionError("a shape-mismatched pair stacked silently")
    except ValueError as exc:
        assert "#<image_id>" in str(exc) and "'A'" in str(exc), exc
    ax1, _c1, d1 = read_meta_only(bad + "#A")
    assert _is_nd3(bad + "#A") and ax1.c == 1 and \
        d1["channel_names"] == ["A"]

    # 3. YXS/BGR reorders samples to R,G,B; an unknown S format is refused
    #    (spec §5.3: never guess an unknown format's sample order)
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 0] = 10                             # the stored B plane
    bgr[..., 2] = 30                             # the stored R plane
    cam = {"scale": {"um_per_px": 1.0}, "channels": [{"name": "Cam"}]}
    f = write("bgr.nd3", {"capture": (bgr, "YXS", "BGR", cam)})
    vol, _ = read_nd3(f)
    assert vol.shape == (1, 1, 1, 3, 4, 5)
    assert vol[0, 0, 0, 0].max() == 30 and vol[0, 0, 0, 2].max() == 10, \
        "BGR samples not reordered — channel 0 would be blue wearing red"
    f = write("weird.nd3", {"capture": (bgr, "YXS", "Weird3", cam)})
    try:
        nd3_meta_only(f)
        raise AssertionError("unknown pixel_format with an S axis loaded")
    except ValueError:
        pass

    # 4. a TYX time-lapse gets dt_s from its t_iso stamps (median)
    lapse_meta = {"scale": {"um_per_px": 0.65},
                  "channels": [{"name": "lapse"}],
                  "planes": [{"t": t, "c": 0, "z": 0,
                              "t_iso": f"2026-08-08T10:{t:02d}:00"}
                             for t in range(4)]}
    f = write("lapse.nd3", {"lapse": (np.zeros((4, 6, 8), np.uint16), "TYX",
                                      "gray16", lapse_meta)})
    axes, calib, _d = nd3_meta_only(f)
    assert axes.t == 4 and calib["dt_s"] == 60.0, calib

    # 5. not-nd3 and newer-major files are refused, naming the reason
    f = write("not.nd3", {}, fmt="zarr")
    try:
        nd3_meta_only(f)
        raise AssertionError("a non-nd3 HDF5 file was read as nd3")
    except nd3mod.ND3FormatError:
        pass
    f = write("future.nd3", {}, schema="2.0")
    try:
        nd3_meta_only(f)
        raise AssertionError("a schema-2.0 file was read by a 1.x reader")
    except nd3mod.ND3FormatError:                # ND3VersionError subclasses it
        pass

    _ok("nd3 ingest roundtrip: raw-h5py spec fixtures load through the "
        "dispatch seam — fluor_well twin stacked in channel_number order "
        "(pixels verified per channel), shape mismatch refused with the "
        "#image_id remediation, BGR reordered to RGB, dt_s from t_iso, and "
        "non-nd3 / newer-major files refused")


def test_transfer_structure() -> None:
    """``transform.transfer_structure`` — the node surface for the structure bridges.

    Until V2.17 the whole structure half of the domain model was unreachable from a graph:
    ``transform.transfer_domain`` refuses every structure pair by construction, and 9 of the
    10 functions in :mod:`nodegraph.bridges` had no caller outside this file. A user who
    segmented, detected or tracked could not move a value between those domains at all.

    What this pins:

    * the matrix is exactly the 15 pairs ``execute_bridge_plan`` can RUN, and the other 5
      combos of the 4x5 grid refuse with a named alternative;
    * each pair's ARITHMETIC, hand-checked on a planted two-object fixture — mean-in-mask,
      paint-by-label, splat, containing-label, gather-over-track, broadcast-track;
    * the two directions T is and is not collapsed in (a Track TARGET gathers the whole
      series at once; a Track SOURCE paints/reduces per frame — collapsing t there wrote
      frame 0 and left every later frame untouched);
    * values land on the target BY ID, with NaN for a row the bridge never produced;
    * every refusal: the 5 dead combos, an inert reducer, an unsupported splat reducer, a
      z_kind mismatch, a reserved output name, an unset ``via_label``.
    """
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.structure import StructureTable as _ST

    spec = NODES.get("transform.transfer_structure")
    assert spec is not None and spec.category == "transform"
    mode_of = {m.name: m for m in spec.modes}
    assert set(mode_of) == {"from_domain", "to_domain", "reducer"}
    assert set(mode_of["from_domain"].choices) == {"voxel", "label", "point", "track"}
    assert set(mode_of["to_domain"].choices) == {"voxel", "label", "point", "track",
                                                 "frame"}
    # reads/adds stay empty — both are per-INSTANCE, like transfer_domain and track.link
    assert spec.reads_domains == frozenset() and spec.adds_domains == frozenset()
    assert spec.resolve_granularity({}) is Granularity.WHOLE_SERIES
    assert spec.resolve_kernel_axes({}) == frozenset(), "it never reads the image provider"
    # the output layer is announced for the edit-time catalog (layer_out cannot express it,
    # because its domains must be a subset of the deliberately-empty adds_domains)
    assert spec.extra_layers is not None
    assert spec.extra_layers({"attr": "v", "name": ""},
                             {"from_domain": "label", "to_domain": "track"}) \
        == ((D.TRACK, "v"),)
    assert spec.extra_layers({"attr": "v", "name": "renamed"},
                             {"from_domain": "label", "to_domain": "point"}) \
        == ((D.POINT, "renamed"),)
    assert spec.extra_layers({}, {"from_domain": "label", "to_domain": "label"}) == ()

    if not _HAVE_SKIMAGE:
        _ok("transfer structure: spec OK; RUN SKIPPED (skimage absent)")
        return

    # ── fixture: 2 labels per frame over 2 frames, linked into 2 tracks ────────
    # Planted so every transfer has a hand-computable answer.
    Y = X = 8
    img = np.zeros((1, 2, 1, 1, Y, X), dtype=float)
    img[0, 0, 0, 0, 1:3, 1:3] = 10.0
    img[0, 0, 0, 0, 5:7, 5:7] = 100.0
    img[0, 1, 0, 0, 1:3, 1:3] = 20.0
    img[0, 1, 0, 0, 5:7, 5:7] = 200.0
    lab = np.zeros((1, 2, 1, 1, Y, X), dtype=np.int64)
    lab[0, 0, 0, 0, 1:3, 1:3] = 1
    lab[0, 0, 0, 0, 5:7, 5:7] = 2
    lab[0, 1, 0, 0, 1:3, 1:3] = 3
    lab[0, 1, 0, 0, 5:7, 5:7] = 4
    ax = AxisSizes(m=1, t=2, z=1, c=1, y=Y, x=X)
    meta = {"pixel_size_um": 0.5}
    ltab = _ST(D.LABEL, {
        "id": np.array([1, 2, 3, 4], np.int64), "m": np.zeros(4, np.int64),
        "t": np.array([0, 0, 1, 1], np.int64), "c": np.zeros(4, np.int64),
        "z": np.zeros(4), "y": np.array([1.5, 5.5, 1.5, 5.5]),
        "x": np.array([1.5, 5.5, 1.5, 5.5]),
        "score": np.array([10.0, 100.0, 20.0, 200.0]),
    }, layer="labels", z_kind="plane_index")
    ptab = _ST(D.POINT, {                       # one point at each label's centroid
        "id": np.arange(4, dtype=np.int64), "m": np.zeros(4, np.int64),
        "t": np.array([0, 0, 1, 1], np.int64), "c": np.zeros(4, np.int64),
        "z": np.zeros(4), "y": np.array([1.5, 5.5, 1.5, 5.5]),
        "x": np.array([1.5, 5.5, 1.5, 5.5]),
        "pv": np.array([1.0, 2.0, 3.0, 4.0]),
    }, layer="spots", z_kind="plane_index")
    ttab = _ST(D.TRACK, {                       # track 1 = labels 1,3; track 2 = 2,4
        "track_id": np.array([1, 1, 2, 2], np.int64),
        "t": np.array([0, 1, 0, 1], np.int64),
        "member_id": np.array([1, 3, 2, 4], np.int64),
        "m": np.zeros(4, np.int64), "c": np.zeros(4, np.int64),
        "tlen": np.array([2.0, 2.0, 2.0, 2.0]),
    }, layer="tracks", z_kind="plane_index")
    base = (Dataset(axes=ax, metadata=meta).with_image(ArrayProvider(img))
            .with_layer(D.VOXEL, "labels", lab)
            .with_layer(D.VOXEL, "signal", img)
            .with_structure(ltab).with_structure(ptab).with_structure(ttab))
    env = MetaEnvelope(axes=ax, metadata=meta)
    define_node("io.xsseed", "Seed", outputs=[OutDataset()])

    _LAYERS = {"label": "labels", "point": "spots", "track": "tracks"}

    def xfer(src, dst, *, seed=None, reducer="mean", **params):
        p = {"source_layer": _LAYERS.get(src, ""), "target_layer": _LAYERS.get(dst, "")}
        p.update(params)
        g = Graph()
        g.add(NodeInstance("S", "io.xsseed"))
        g.add(NodeInstance("N", "transform.transfer_structure", params=p,
                           modes={"from_domain": src, "to_domain": dst,
                                  "reducer": reducer}))
        g.connect("S", "N")
        e = Engine(g, computes=COMPUTES, seeds={"S": seed if seed is not None else base},
                   meta_seeds={"S": env})
        return e, e.pull("N")

    def col(out, dom, name, layer):
        a = out.get(dom, name, layer=layer)
        assert a is not None, f"no {dom.value}/{layer}/{name} on the output"
        return np.asarray(a.values, dtype=float)

    # ── voxel → label: mean-in-mask of the planted signal ─────────────────────
    _, o = xfer("voxel", "label", attr="signal", name="v2l")
    assert np.allclose(col(o, D.LABEL, "v2l", "labels"), [10.0, 100.0, 20.0, 200.0])
    # ...and the reducer really reaches the bridge (4 voxels per object)
    _, o = xfer("voxel", "label", attr="signal", name="v2l", reducer="sum")
    assert np.allclose(col(o, D.LABEL, "v2l", "labels"), [40.0, 400.0, 80.0, 800.0])

    # ── label → voxel: paint-by-label, EVERY frame, background left at 0 ──────
    _, o = xfer("label", "voxel", attr="score", name="l2v")
    painted = np.asarray(o.get(D.VOXEL, "l2v").values)
    assert painted[0, 0, 0, 0, 1, 1] == 10.0 and painted[0, 0, 0, 0, 5, 5] == 100.0
    assert painted[0, 1, 0, 0, 1, 1] == 20.0 and painted[0, 1, 0, 0, 5, 5] == 200.0, \
        "frame 1 was not painted — the per-frame loop collapsed t"
    assert painted[0, 0, 0, 0, 0, 0] == 0.0

    # ── label → point: each point takes its containing region's value ─────────
    _, o = xfer("label", "point", attr="score", name="l2p")
    assert np.allclose(col(o, D.POINT, "l2p", "spots"), [10.0, 100.0, 20.0, 200.0])

    # ── point → label: reduce the points inside each region ───────────────────
    _, o = xfer("point", "label", attr="pv", name="p2l")
    assert np.allclose(col(o, D.LABEL, "p2l", "labels"), [1.0, 2.0, 3.0, 4.0])

    # ── point → voxel: splat onto the nearest voxel ───────────────────────────
    _, o = xfer("point", "voxel", attr="pv", name="p2v", reducer="sum")
    splat = np.asarray(o.get(D.VOXEL, "p2v").values)
    assert splat[0, 0, 0, 0, 2, 2] == 1.0 and splat[0, 1, 0, 0, 6, 6] == 4.0, \
        "the splat did not land at the points' nearest voxels in every frame"

    # ── label → track: gather over the track's members ────────────────────────
    _, o = xfer("label", "track", attr="score", name="l2t")
    assert np.allclose(col(o, D.TRACK, "l2t", "tracks"), [15.0, 15.0, 150.0, 150.0])
    _, o = xfer("label", "track", attr="score", name="l2t", reducer="max")
    assert np.allclose(col(o, D.TRACK, "l2t", "tracks"), [20.0, 20.0, 200.0, 200.0])

    # ── track → label: broadcast a track's value onto its members ─────────────
    _, o = xfer("track", "label", attr="tlen", name="t2l")
    assert np.allclose(col(o, D.LABEL, "t2l", "labels"), [2.0, 2.0, 2.0, 2.0])

    # ── track → frame: one number PER FRAME, not one for the whole series ─────
    _, o = xfer("track", "frame", attr="tlen", name="t2f")
    t2f = np.asarray(o.get(D.FRAME, "t2f").values, dtype=float)
    assert t2f.shape == (1, 2) and np.isfinite(t2f).all(), t2f
    assert np.allclose(t2f, 2.0), t2f

    # ── label → frame: reduce the frame's objects into one number ─────────────
    _, o = xfer("label", "frame", attr="score", name="l2f")
    assert np.allclose(np.asarray(o.get(D.FRAME, "l2f").values), [[55.0, 110.0]])

    # ── voxel → track: the 2-hop route, via_label named ───────────────────────
    _, o = xfer("voxel", "track", attr="signal", name="v2t", via_label="labels")
    assert np.allclose(col(o, D.TRACK, "v2t", "tracks"), [15.0, 15.0, 150.0, 150.0])

    # ── a target row the bridge never produced reads NaN, never 0 ─────────────
    lonely = base.with_structure(_ST(D.TRACK, {
        "track_id": np.array([1, 9], np.int64), "t": np.array([0, 0], np.int64),
        "member_id": np.array([1, 999], np.int64), "m": np.zeros(2, np.int64),
        "c": np.zeros(2, np.int64), "tlen": np.array([2.0, 2.0]),
    }, layer="tracks", z_kind="plane_index"))
    _, o = xfer("label", "track", attr="score", seed=lonely, name="miss")
    miss = col(o, D.TRACK, "miss", "tracks")
    assert np.isfinite(miss[0]) and np.isnan(miss[1]), \
        f"a track whose member is absent from the source table must read NaN, got {miss}"

    # ── the flagship round trip: label -> track -> label -> voxel ─────────────
    # "colour every cell by its track's mean speed". It is the reason this node exists, it
    # exercises three pairs and both t conventions, and it is what caught the NaN-poisoning
    # in _group_reduce (a column with ONE missing entry made every track NaN).
    with_gap = base.with_structure(_ST(D.LABEL, dict(
        ltab.columns, score=np.array([np.nan, 100.0, 20.0, 200.0])),
        layer="labels", z_kind="plane_index"))
    _, o = xfer("label", "track", attr="score", seed=with_gap, name="ts")
    assert np.allclose(col(o, D.TRACK, "ts", "tracks"), [20.0, 20.0, 150.0, 150.0]),         "a missing member value must be SKIPPED, not poison its track's mean"
    _, o2 = xfer("track", "label", attr="ts", seed=o, name="ts_l")
    assert np.allclose(col(o2, D.LABEL, "ts_l", "labels"), [20.0, 150.0, 20.0, 150.0])
    _, o3 = xfer("label", "voxel", attr="ts_l", seed=o2, name="speed_map")
    smap = np.asarray(o3.get(D.VOXEL, "speed_map").values)
    assert smap[0, 0, 0, 0, 0, 0] == 0.0, "background must stay 0"
    assert smap[0, 0, 0, 0, 1, 1] == 20.0 and smap[0, 1, 0, 0, 5, 5] == 150.0, smap

    # ── refusals ──────────────────────────────────────────────────────────────
    def refuses(src, dst, needle, **kw):
        try:
            xfer(src, dst, **kw)
        except ValueError as exc:
            assert needle in str(exc), f"{src}->{dst}: {exc}"
            return
        raise AssertionError(f"{src}->{dst} {kw} was accepted; expected {needle!r}")

    for dom in ("voxel", "label", "point", "track"):            # the 4 self-pairs
        refuses(dom, dom, "nothing to transfer", attr="score")
    refuses("voxel", "frame", "LATTICE transfer", attr="signal")
    refuses("label", "point", "does not reduce", attr="score", reducer="sum")
    refuses("point", "voxel", "reduces only the COLLISIONS", attr="pv", reducer="median")
    refuses("label", "track", "invariant coordinate columns", attr="score", name="y")
    refuses("voxel", "track", "routes through Label", attr="signal")   # via_label unset
    refuses("label", "track", "no label instance", attr="score", source_layer="nope")
    refuses("label", "track", "has no column", attr="not_a_column")
    # a z_kind disagreement between the two structure endpoints
    mixed = base.with_structure(_ST(D.POINT, dict(ptab.columns), layer="spots",
                                    z_kind="subpixel"))
    refuses("label", "point", "different geometries", attr="score", seed=mixed)

    # ── each pair is its own memo key ─────────────────────────────────────────
    keys = set()
    for src, dst in (("voxel", "label"), ("label", "voxel"), ("label", "track"),
                     ("label", "frame")):
        e, _ = xfer(src, dst, attr=("signal" if src == "voxel" else "score"), name="k")
        keys.add(e.entry("N").recipe_hash)
    assert len(keys) == 4, f"each pair must re-key the memo, got {len(keys)}"

    _ok("transfer structure (V2.17): the 15 executable pairs of the geometric spine are a "
        "NODE at last (bridges.py had 9 functions with no caller outside this file) — "
        "mean-in-mask / paint-by-label / splat / containing-label / gather-over-track / "
        "broadcast-track all hand-checked on a planted 2-object 2-frame fixture, the "
        "reducer reaches the bridge (sum and max change the answer), a Track SOURCE paints "
        "and reduces PER FRAME while a Track TARGET gathers the whole series, an unmatched "
        "target row reads NaN not 0, 4 pairs re-key distinctly, and 11 refusals (4 "
        "self-pairs, voxel->frame, inert reducer, unsupported splat reducer, reserved "
        "output name, unset via_label, missing instance, missing column, z_kind mismatch)")


def test_group_reduce_repairs() -> None:
    """``_group_reduce`` — the segment rewrite and the integer-key refusal (V2.17).

    ``max``/``min``/``median`` were ``[fn(values[idx == k]) for k in range(n_labels)]``:
    two full-length boolean masks per label, O(n_labels x n_voxels), and it is the DEFAULT
    path because ``analysis.measure``'s ``stats`` ships as ``"mean,max,min,count"``.
    Measured on 2**20 voxels it went 129 ms -> 95 ms at 100 labels and **3871 ms -> 176 ms
    at 7052** — the label count a single 3-position 4-frame crop of the lab's ND2 yields.

    And the ids were ``np.unique(labels).astype(int64)``, which TRUNCATES a float key —
    reachable, because ``analysis.edt`` writes a float64 Voxel layer and the ``labels``
    picker offers every Voxel layer on the edge."""
    from nodegraph.bridges import _group_reduce

    rng = np.random.RandomState(7)
    for _ in range(120):
        n, k = int(rng.randint(1, 300)), int(rng.randint(1, 10))
        lab = rng.randint(0, k + 1, n).astype(np.int64)
        val = rng.randn(n) * 50.0
        fg = lab > 0
        want_ids = np.unique(lab[fg])
        for red, fn in (("mean", np.mean), ("sum", np.sum), ("count", len),
                        ("max", np.max), ("min", np.min), ("median", np.median)):
            ids, out = _group_reduce(lab, val, red)
            assert np.array_equal(ids, want_ids), (red, ids, want_ids)
            want = np.array([float(fn(val[fg][lab[fg] == i])) for i in want_ids])
            assert np.allclose(out, want), (red, out, want)
    # an even-length segment averages the two middles, exactly like np.median
    _ids, out = _group_reduce(np.array([1, 1, 1, 1], np.int64),
                              np.array([1.0, 2.0, 3.0, 10.0]), "median")
    assert np.allclose(out, [2.5]), out

    # ── NaN is MISSING DATA, not a value (2026-07-30) ─────────────────────────
    # Plain numpy semantics poison a group from one NaN, and in this engine NaN is exactly
    # how a structure column says "no value here" — object_metrics writes it for a track's
    # FIRST frame, where a velocity is undefined. So gathering per-object `speed` into its
    # Track with `mean` returned NaN for EVERY track, because every track has a first
    # frame. Found by running the flagship workflow (colour each cell by its track's mean
    # speed) on the real ND2, where it painted 1.9 M object voxels entirely NaN.
    lab_n = np.array([1, 1, 1, 2, 2, 3], np.int64)
    val_n = np.array([np.nan, 2.0, 4.0, np.nan, np.nan, 7.0])
    for red, want in (("mean", [3.0, np.nan, 7.0]), ("sum", [6.0, np.nan, 7.0]),
                      ("count", [2.0, 0.0, 1.0]), ("max", [4.0, np.nan, 7.0]),
                      ("min", [2.0, np.nan, 7.0]), ("median", [3.0, np.nan, 7.0])):
        ids_n, out_n = _group_reduce(lab_n, val_n, red)
        assert np.array_equal(ids_n, [1, 2, 3]), (red, ids_n)
        assert np.allclose(out_n, want, equal_nan=True), (red, out_n, want)
    # a group that is ENTIRELY missing keeps its id and reads NaN — it must not vanish,
    # or the caller's scatter-by-id would silently shift every later row
    assert _group_reduce(np.array([5], np.int64), np.array([np.nan]), "mean")[0] == [5]
    # ...and with NaNs present every value reducer still matches its nan-aware reference
    for _ in range(60):
        n2, k2 = int(rng.randint(2, 200)), int(rng.randint(1, 8))
        lab2 = rng.randint(0, k2 + 1, n2).astype(np.int64)
        val2 = rng.randn(n2) * 40.0
        val2[rng.rand(n2) < 0.3] = np.nan
        fg2 = lab2 > 0
        want_ids2 = np.unique(lab2[fg2])
        for red, fn in (("mean", np.mean), ("sum", np.sum), ("max", np.max),
                        ("min", np.min), ("median", np.median)):
            ids2, out2 = _group_reduce(lab2, val2, red)
            assert np.array_equal(ids2, want_ids2)
            want2 = []
            for i in want_ids2:
                v = val2[fg2][lab2[fg2] == i]
                v = v[np.isfinite(v)]
                want2.append(float(fn(v)) if v.size else np.nan)
            assert np.allclose(out2, want2, equal_nan=True), (red, out2, want2)
    # a float key array is refused rather than truncated into duplicate ids
    try:
        _group_reduce(np.array([1.2, 1.8]), np.array([1.0, 2.0]), "mean")
        raise AssertionError("a float label raster was accepted — the ids truncate to 1,1")
    except ValueError as exc:
        assert "integer label ids" in str(exc), exc
    _ok("group reduce (V2.17): 180 random cases x 6 reducers match a per-label reference "
        "(incl. even/odd medians, single-element and empty groups); max/min/median are now "
        "segment reductions instead of a per-label full-array mask (3871 ms -> 176 ms at "
        "the 7052 labels a real ND2 crop yields); NaN is treated as MISSING and skipped "
        "per group (an all-missing group keeps its id and reads NaN, count reports the "
        "contributions) so a track's undefined first-frame velocity no longer poisons its "
        "mean; a float key array is refused instead of truncating two regions onto one id")


# ── V2.16: interactive parameter picking (declarations + the gesture maths) ───

def test_picking() -> None:
    """A picked parameter must equal the number a careful user would have typed.

    That is the whole promise of the feature, and it is entirely arithmetic — plane pixels
    through a calibration to microns — so it is checkable here rather than only by eye in
    the GUI. What this holds:

    * every ``pick_kind`` in the catalog is routed to a surface and typed correctly;
    * the declarations are MEMO-NEUTRAL, exactly like ``description`` and ``path_kind``
      (annotating a socket must never invalidate a cached result or a saved graph);
    * ``choices`` / ``vocab`` agree with what the compute will actually accept, which is
      the one way a dropdown can lie — offering a value that then raises;
    * the gestures convert correctly, including the µm³ case that needs the z step and the
      uncalibrated case that must SAY it is reporting pixels;
    * a co-picked pair can never be committed inverted.
    """
    import dataclasses
    import json
    from nodegraph.registry import PICK_KINDS
    from nodelab_v2.picker import (
        CANVAS_KINDS, Calibration, HISTOGRAM_KINDS, INSTANT_KINDS, PICK_ACTION, PICK_HELP,
        PickRequest, PickSession, histogram_values, instant_values, request_for)

    # (1) every kind is routed, explained and labelled — a kind that reaches the GUI with
    # no surface arms a session nothing can drive; one with no help text is a bare button.
    assert INSTANT_KINDS | HISTOGRAM_KINDS | CANVAS_KINDS == PICK_KINDS
    assert not (INSTANT_KINDS & HISTOGRAM_KINDS) and not (CANVAS_KINDS & HISTOGRAM_KINDS)
    for kind in PICK_KINDS:
        assert PICK_HELP.get(kind), f"{kind}: no instruction line"
        assert PICK_ACTION.get(kind), f"{kind}: no button label"

    picked = [(sp, s) for sp in NODES.all() for s in sp.inputs if s.pick_kind]
    assert len(picked) >= 50, f"only {len(picked)} params annotated — did the table shrink?"

    # (2) declarations resolve: a peer names a real, same-node input, and the pair is
    # mutual (arming either end must offer the same two).
    for sp, s in picked:
        assert s.pick_kind in PICK_KINDS, f"{sp.op_key}.{s.name}"
        if s.pick_peer:
            peer = sp.input(s.pick_peer)
            assert peer is not None, f"{sp.op_key}.{s.name}: peer {s.pick_peer} missing"
            assert peer.pick_peer == s.name, \
                f"{sp.op_key}: {s.name}/{peer.name} peer link is not mutual"
            assert peer.pick_kind == s.pick_kind, \
                f"{sp.op_key}: {s.name}/{peer.name} co-pick with different gestures"

    # (3) a peer must be REACHABLE whenever its partner is: if a mode gates one away and
    # not the other, arming the visible one would write a param the user cannot see.
    for sp, s in picked:
        if not s.pick_peer:
            continue
        peer = sp.input(s.pick_peer)
        for state in _mode_states(sp):
            assert s.active_in(state) == peer.active_in(state), \
                f"{sp.op_key}: {s.name} and {peer.name} are gated apart in {state}"

    # (3b) a BOUND group (one gesture → the whole set) must be complete, mutually
    # consistent, homogeneous and gated together — the same reachability rule as a peer,
    # for the same reason, plus the type/unit invariant the session's conversion leans on.
    from nodegraph.registry import BOUND_PICK_KINDS
    bound = [(sp, s) for sp, s in picked if s.pick_kind in BOUND_PICK_KINDS]
    assert bound, "no bound-group picks in the catalog — did util.crop lose its rect?"
    for sp, s in bound:
        assert s.pick_bounds and s.name in s.pick_bounds, f"{sp.op_key}.{s.name}"
        assert not s.pick_peer, f"{sp.op_key}.{s.name}: bounds and peer are exclusive"
        for name in s.pick_bounds:
            other = sp.input(name)
            assert other is not None, f"{sp.op_key}.{s.name}: bound {name} missing"
            assert (other.pick_kind, other.pick_bounds) == (s.pick_kind, s.pick_bounds), \
                f"{sp.op_key}: bound members disagree about the group"
            assert (other.type, other.unit) == (s.type, s.unit), \
                f"{sp.op_key}: bound group is not homogeneous ({name})"
            for state in _mode_states(sp):
                assert s.active_in(state) == other.active_in(state), \
                    f"{sp.op_key}: {s.name} and {name} are gated apart in {state}"

    # …and the crop node specifically, since it is the reason the mechanism exists: the four
    # lateral bounds are ONE rectangle, the two axial ones a SEPARATE Z range (a rectangle
    # drawn on a plane says nothing about z), and the halves are gated apart on the lever.
    _crop = NODES.get("util.crop")
    assert [_crop.input(n).pick_kind for n in ("y0", "y1", "x0", "x1")] == ["rect"] * 4
    assert [_crop.input(n).pick_kind for n in ("z0", "z1")] == ["zrange"] * 2
    assert _crop.input("y0").pick_bounds == ("y0", "y1", "x0", "x1")
    assert _crop.input("z0").pick_bounds == ("z0", "z1")
    assert _crop.input("y0").active_in({"dim": "2D"}) and \
        not _crop.input("z0").active_in({"dim": "2D"}), \
        "the Z range must be 3D-only while the rectangle is always available"

    # (4) memo neutrality — the same params with every V2.16 annotation stripped must hash
    # identically, or annotating a socket would silently invalidate everyone's cache.
    for op in ("analysis.segment", "detect.spots", "analysis.roi_mask", "enhance.gaussian"):
        sp = NODES.get(op)
        pv = {s.name: s.default for s in sp.inputs
              if s.type is not SocketType.DATASET}
        bare = [dataclasses.replace(s, pick_kind="", pick_peer="", choices=(), vocab=())
                for s in sp.inputs if s.type is not SocketType.DATASET]
        assert (node_recipe_hash(op, pv, (), ())
                == node_recipe_hash(op, {s.name: s.default for s in bare}, (), ())), \
            f"{op}: a pick/choices/vocab annotation reached the recipe hash"

    # (5) closed vocabularies match what the compute accepts. A dropdown that offers a
    # value the node then refuses is worse than the free text it replaced.
    from nodegraph import nodes as _N
    for op, sock, allowed in (
            ("analysis.measure", "stats", _N._MEASURE_COLUMNS),
            ("analysis.measure", "shape", _N._MEASURE_SHAPE),
            ("analysis.object_metrics", "metrics", _N._OBJECT_METRIC_COLUMNS),
            ("analysis.object_field", "fields", _N._OBJECT_FIELDS)):
        s = NODES.get(op).input(sock)
        assert set(s.vocab) == set(allowed), f"{op}.{sock}: vocab drifted from the compute"
        assert all(t.strip() in s.vocab for t in str(s.default or "").split(",") if t.strip())
    for op, sock in (("analysis.segment", "model_name"),
                     ("analysis.segment", "model_name_3d"),
                     ("analysis.segment", "cellsam_model")):
        s = NODES.get(op).input(sock)
        assert s.choices and s.default in s.choices, f"{op}.{sock}"

    # (6) the gestures. 0.5 µm/px laterally, 2.0 µm per z step.
    cal = Calibration(um_px=0.5, um_z=2.0)

    def sess(**kw):
        d = dict(node_id="n", socket="s", kind="radius", unit="um")
        d.update(kw)
        return PickSession(PickRequest(**d), cal)

    s = sess()                                        # radius: 20 px → 10 µm
    s.press(100, 100); s.drag(120, 100); s.release(120, 100)
    assert s.done and s.values() == {"s": 10.0}, s.values()

    s = sess(kind="distance", socket="d")             # ruler: two clicks, 8 px → 4 µm
    s.press(0, 0); s.drag(3, 4); s.release(3, 4)      # a stray drag must NOT complete it
    assert not s.done, "one click completed a two-click measurement"
    s.press(0, 8); s.release(0, 8)
    assert s.values() == {"d": 4.0}, s.values()

    s = sess(kind="area", socket="a", unit="um2")     # click an object: 400 px → 100 µm²
    s.press(5, 5, probe=400.0); s.release(5, 5)
    assert s.values() == {"a": 100.0}, s.values()

    s = sess(kind="area", socket="v", unit="um3")     # the same count as a VOLUME
    s.press(5, 5, probe=400.0); s.release(5, 5)
    assert s.values() == {"v": 200.0}, s.values()     # × the 2.0 µm z step

    s = sess(kind="area", socket="a", unit="um2")     # draw a 20×20 blob instead
    s.press(0, 0)
    for p in ((20, 0), (20, 20), (0, 20)):
        s.drag(*p)
    s.release(0, 20)
    assert s.values() == {"a": 100.0}, s.values()

    s = sess(kind="level", socket="t", unit="")       # eyedropper
    s.press(5, 5, probe=1234.0); s.release(5, 5)
    assert s.values() == {"t": 1234.0}
    s = sess(kind="level", socket="t", unit="")       # …with nothing under the cursor
    s.press(5, 5); s.release(5, 5)
    assert not s.done and s.readout(), "a refused sample must explain itself"

    # a px-unit grid stays in pixels and commits as an int
    s = sess(kind="grid", socket="box", peer="stride", unit="px", peer_unit="px",
             integral=True, peer_integral=True)
    s.press(0, 0); s.drag(16, 12); s.release(16, 12)
    s.press(0, 0); s.drag(10, 3); s.release(10, 3)
    assert s.values() == {"box": 16, "stride": 10}, s.values()

    # (6b) the crop rectangle: ONE drag → four integer bounds, as a slice window that keeps
    # exactly the pixels the rectangle covered (floor the starts, ceil the ends).
    _rr = request_for("nC", _crop.input("x0"))
    rs = PickSession(_rr, cal)
    rs.press(12.4, 5.8); rs.drag(120, 90); rs.release(200.2, 140.1)
    assert rs.values() == {"y0": 5, "y1": 141, "x0": 12, "x1": 201}, rs.values()
    assert "189 × 136 px" in rs.readout(), rs.readout()
    rs2 = PickSession(_rr, cal)                       # dragged up-left: same window
    rs2.press(200, 140); rs2.drag(100, 100); rs2.release(12, 5)
    assert rs2.values() == {"y0": 5, "y1": 140, "x0": 12, "x1": 200}, rs2.values()
    rs3 = PickSession(_rr, cal)                       # a click is not a rectangle
    rs3.press(50, 50); rs3.release(50, 50)
    assert not rs3.done and rs3.readout(), "a bare click must be refused with a reason"
    rs4 = PickSession(_rr, cal)                       # sub-pixel drag → a legal 1 px window
    rs4.press(10, 10); rs4.drag(40, 10.1); rs4.release(40, 10.2)
    _v4 = rs4.values()
    assert _v4["y1"] - _v4["y0"] >= 1 and _v4["x1"] - _v4["x0"] >= 1, _v4
    # the release position wins over the last move — otherwise the committed window would
    # depend on how Qt happened to coalesce the drag
    rs5 = PickSession(_rr, cal)
    rs5.press(0, 0); rs5.drag(10, 10); rs5.release(80, 60)
    assert rs5.values() == {"y0": 0, "y1": 60, "x0": 0, "x1": 80}, rs5.values()

    # (6c) the Z range comes off the Z strip's multi-select. The picks may be SPARSE and a
    # crop window cannot be, so it takes their span — the only reading that never silently
    # drops a plane the user ticked. Nothing picked = the whole stack, not an empty range.
    _zr = request_for("nC", _crop.input("z1"))
    assert instant_values(_zr, channel=0, frame=0, channels=(),
                          z_picks=(9, 2, 5), z_total=12) == {"z0": 2, "z1": 10}
    assert instant_values(_zr, channel=0, frame=0, channels=(),
                          z_picks=(), z_total=12) == {"z0": 0, "z1": 12}
    assert instant_values(_zr, channel=0, frame=0, channels=(),
                          z_picks=(4,), z_total=12) == {"z0": 4, "z1": 5}

    # (7) a pair can never commit inverted, whichever end was aimed first — and the peer
    # half is measured in the SAME unit even though only the primary declares one.
    for first, second in ((40, 20), (20, 40)):
        s = sess(socket="min_radius", peer="max_radius")
        s.press(0, 0); s.drag(first, 0); s.release(first, 0)
        s.press(0, 0); s.drag(second, 0); s.release(second, 0)
        got = s.values()
        assert got["min_radius"] < got["max_radius"], got

    # (8) an uncalibrated file reports pixels AND says so — silently relabelling px as µm
    # is the one failure mode of this feature that produces a wrong number with no warning.
    s = PickSession(PickRequest(node_id="n", socket="s", kind="radius", unit="um"),
                    Calibration())
    s.press(0, 0); s.drag(10, 0); s.release(10, 0)
    assert s.values() == {"s": 10.0}
    assert "uncalibrated" in s.readout(), s.readout()

    # (9) ROI shapes replay through the REAL kernel with the [y, x] flip intact
    from nodegraph.kernels.dic_mesh_region import build_roi_mask
    s = PickSession(PickRequest(node_id="n", socket="shapes", kind="shapes"),
                    Calibration(um_px=1.0))
    s.tool = "rect"; s.press(10, 4); s.drag(30, 12); s.release(30, 12)
    mask = build_roi_mask(json.loads(s.values()["shapes"]), 40, 50)
    ys, xs = np.where(mask)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (4, 12, 10, 30), \
        "a drawn rect landed somewhere other than where it was drawn (y/x flip?)"
    s.tool = "circle"; s.op = "cut"; s.press(20, 8); s.drag(23, 8); s.release(23, 8)
    cut = build_roi_mask(json.loads(s.values()["shapes"]), 40, 50)
    assert not cut[8, 20] and cut[4, 10], "the stateful add/cut replay is wrong"
    s.undo_shape()
    assert len(s.shapes) == 1

    # (10) the two no-gesture surfaces
    r = PickRequest(node_id="n", socket="lo", kind="percentile", peer="hi")
    assert histogram_values(r, lo=120, hi=880, vmin=0, vmax=1000, gamma=1.0) == \
        {"lo": 12.0, "hi": 88.0}
    assert histogram_values(r, lo=5, hi=9, vmin=7, vmax=7, gamma=1.0) == \
        {"lo": 0.0, "hi": 100.0}, "a degenerate range must not divide by zero"
    assert instant_values(PickRequest(node_id="n", socket="ch", kind="channels"),
                          channel=1, frame=4, channels=(0, 2)) == {"ch": "0,2"}

    # (11) the shared request builder reads the socket's own type/unit, so the inspector
    # button and the card glyph cannot arm sessions that disagree
    sp = NODES.get("analysis.dvc_field")
    req = request_for("nX", sp.input("subset_size"), sp.input("subset_spacing"))
    assert (req.integral and req.peer_integral and req.unit == "px"
            and req.peer == "subset_spacing" and req.surface == "canvas"), req

    _ok(f"picking (V2.16): {len(picked)} params annotated across {len({o.op_key for o, _ in picked})} "
        "nodes; every kind routed/typed/explained; peers mutual, same-gesture and gated "
        "together; bound GROUPS complete, homogeneous and gated together (crop = one rect + "
        "a 3D-only Z range); annotations memo-neutral; choices+vocab match the computes; "
        "radius / ruler / area / volume / blob / eyedropper / grid convert correctly; the "
        "crop rect keeps exactly the pixels it covered (either drag direction, release wins, "
        "1 px floor) and the Z range spans sparse strip picks; pairs never invert; "
        "uncalibrated says so; ROI shapes replay through the real kernel")


def _mode_states(spec):
    """Every combination of a node's mode values — small by construction (the catalog's
    biggest node has 4×6×2 = 48), and the only honest way to ask "is this socket ever
    reachable while its peer is not?"."""
    import itertools
    if not spec.modes:
        return [{}]
    names = [m.name for m in spec.modes]
    return [dict(zip(names, combo))
            for combo in itertools.product(*[list(m.choices) for m in spec.modes])]


# ── CuPy wheel selection (V2.14) ─────────────────────────────────────────────

def test_gpu_platform() -> None:
    """The machine-adaptive CuPy wheel choice.

    This is the one piece of the GPU work that RUNS ON OTHER PEOPLE'S MACHINES and cannot
    be checked by running it here — the dev box has exactly one GPU and one driver. So the
    selection function is exercised against synthetic :class:`~nodegraph.gpu.Platform`
    values covering the combinations that actually bite: an old card behind a brand-new
    driver (where the newest wheel is the WRONG answer even though the driver would load
    it), a new card behind an old driver (the reverse), multi-GPU boxes where the OLDEST
    card sets the floor, and the no-GPU/AMD/macOS paths."""
    import os
    from nodegraph import gpu as G

    def P(driver, sms, kind="cuda"):
        return G.Platform(kind, driver_cuda=driver,
                          devices=tuple(G.Device(f"dev{i}", sm, 8 << 30)
                                        for i, sm in enumerate(sms)))

    def pick(driver, sms):
        pkg = G.recommended_package(P(driver, sms))
        return pkg.split("[")[0] if pkg else None

    # ── the driver ceiling: a wheel's runtime cannot exceed what the driver loads ──
    assert pick((13, 2), [86]) == "cupy-cuda13x"
    assert pick((12, 4), [86]) == "cupy-cuda12x"
    assert pick((11, 8), [86]) == "cupy-cuda11x"
    assert pick((11, 2), [75]) == "cupy-cuda11x"          # exactly on the floor
    assert pick((11, 1), [75]) is None                    # below every supported build
    assert pick((10, 2), [61]) is None

    # ── the architecture floor: newest-driver-wins is WRONG on an old card ──
    # a GTX 1080 (sm_61) behind a CUDA 13 driver must NOT get cuda13x (Pascal is dropped
    # from 13.x); it must step down to the newest build that still supports sm_61.
    assert pick((13, 2), [61]) == "cupy-cuda12x"
    assert pick((13, 2), [37]) == "cupy-cuda11x"          # Kepler → back to 11.x
    assert pick((12, 4), [37]) == "cupy-cuda11x"
    assert pick((13, 2), [75]) == "cupy-cuda13x"          # Turing: exactly on 13.x's floor
    assert pick((13, 2), [70]) == "cupy-cuda12x"          # Volta is one step below it

    # ── multi-GPU: the OLDEST device sets the floor, in any order ──
    assert pick((13, 2), [86, 61]) == "cupy-cuda12x"
    assert pick((13, 2), [61, 86]) == "cupy-cuda12x"
    assert pick((13, 2), [86, 89]) == "cupy-cuda13x"
    assert P((13, 2), [86, 61]).min_sm == 61

    # ── the non-CUDA paths ──
    assert G.recommended_package(G.Platform("none", note="x")) is None
    rocm = G.recommended_package(G.Platform("rocm", note="x"))
    assert rocm and "rocm" in rocm and "[ctk]" not in rocm   # ctk is a CUDA-only extra

    # ── the [ctk] extra is never dropped: without CUDA headers CuPy rejects every
    #    ndimage kernel at first call, so a recommendation lacking it is a broken install
    for drv, sms in (((13, 2), [86]), ((12, 4), [61]), ((11, 8), [37])):
        assert G.recommended_package(P(drv, sms)).endswith("[ctk]"), (drv, sms)

    # ── detection on THIS machine must be well-formed and must never raise ──
    live = G.detect_platform()
    assert live.kind in ("cuda", "rocm", "none")
    if live.kind == "cuda":
        assert len(live.driver_cuda) == 2 and live.devices and live.min_sm > 0
    else:
        assert live.note, "a non-CUDA platform must explain itself"
    assert G.install_hint(live)                 # always actionable prose, never empty
    assert G.mode() in ("off", "auto", "on")
    # the DEFAULT is off — a GPU that silently made some graphs slower would be worse
    # than one the user opts into (see gpu.mode's measurements)
    _old = os.environ.pop("NODEGRAPH_GPU", None)
    try:
        assert G.mode() == "off"
    finally:
        if _old is not None:
            os.environ["NODEGRAPH_GPU"] = _old

    _ok("gpu platform (V2.14): CuPy wheel picked from BOTH the driver's CUDA ceiling and "
        "the oldest device's architecture floor — 17 synthetic combinations incl. Pascal/"
        "Volta/Kepler behind a CUDA-13 driver stepping down, multi-GPU floor, sub-11.2 "
        "refusal, ROCm/none paths; [ctk] never dropped; live probe well-formed; default off")


# ── parallel execution policy (V2.14) ────────────────────────────────────────

def _pw_square(i):
    """A module-level worker for the process-pool round trip (a closure cannot be
    pickled, which is the contract `map_units_proc` documents)."""
    return i * i


def test_parallel() -> None:
    """The V2.14 run-wide fan-out: ordering, re-entrancy, cache locking, and — the one
    that actually matters — that a parallel pull is BYTE-IDENTICAL to a serial one for the
    nodes whose output depends on unit order."""
    import threading as _th
    from nodegraph import parallel as P
    from nodegraph.streaming import TileCache

    # ── 1. sizing is machine-derived and sane ─────────────────────────────────
    assert P.cpu_budget() >= 1 and P.proc_budget() >= 1
    assert P.total_ram_bytes() > 0
    # the tile budget is what gates lazy-vs-eager (`unit_bytes <= budget//2`), so a
    # collapsed budget silently reinstates whole-6-D realization
    assert P.tile_cache_bytes() >= (1 << 30)
    assert P.ram_budget(0.5, floor=8, cap=16) == 16          # cap wins
    assert P.ram_budget(0.0, floor=8, cap=16) == 8           # floor wins
    assert P.inner_threads(1) >= P.inner_threads(64)         # narrows as workers grow

    # ── 2. map_units: order-preserving, exception-propagating, re-entrant-safe ──
    src = list(range(97))
    assert P.map_units(lambda x: x * 2, src) == [x * 2 for x in src]
    assert P.map_units(lambda x: x, []) == []

    def boom(x):
        if x == 40:
            raise ValueError("unit 40")
        return x
    try:
        P.map_units(boom, src)
        raise AssertionError("a failing unit did not propagate")
    except ValueError as exc:
        assert "unit 40" in str(exc)

    # a nested map must run SERIAL rather than launching workers², and the flag must be
    # cleared again afterwards even though the inner map also touches it
    seen = []

    def outer(_i):
        seen.append(P.in_parallel_region())
        return P.map_units(lambda j: j, [1, 2, 3])
    assert P.map_units(outer, list(range(6)))[0] == [1, 2, 3]
    if P.enabled():
        # the flag must be set ON THE WORKER, not on the submitting thread — otherwise the
        # nesting guard silently does nothing. Only meaningful when there ARE workers:
        # under NODEGRAPH_PARALLEL=off the map runs inline and correctly leaves it clear.
        assert all(seen), "workers did not observe the active-region flag"
    else:
        assert not any(seen), "serial path should not claim to be a parallel region"
    assert not P.in_parallel_region(), "region flag leaked past the map"

    # ── 3. fold_units: parallel compute, serial ORDERED fold, bounded batches ──
    order, prepared = [], []
    P.fold_units(lambda x: x * 3, list(range(50)),
                 lambda i, item, res: order.append((i, item, res)),
                 batch=7)
    assert order == [(i, i, i * 3) for i in range(50)], "fold lost submission order"
    P.fold_units(lambda p: p + 1, list(range(20)),
                 lambda i, item, res: order.append(res),
                 prepare=lambda x: (prepared.append(x) or x) * 10, batch=5)
    assert prepared == list(range(20))                       # prepare ran, in order
    assert order[50:] == [x * 10 + 1 for x in range(20)]     # and fed the compute

    # ── 4. the process pool round-trips and falls back on an unpicklable payload ──
    assert P.map_units_proc(_pw_square, list(range(24))) == [i * i for i in range(24)]
    mult = 7                                                 # closure ⇒ unpicklable
    assert P.map_units_proc(lambda i: i * mult, list(range(9))) == \
        [i * 7 for i in range(9)]

    # ── 5. TileCache under concurrent load: no drift, no exception ─────────────
    tc = TileCache(1 << 20)
    errs = []

    def hammer(seed):
        try:
            for i in range(1500):
                k = ("t", "fp", 0, 0, 0, 0, (seed + i) % 32, i % 32)
                if tc.get(k) is None:
                    tc.put(k, np.zeros((32, 32)))
        except Exception as exc:                             # noqa: BLE001
            errs.append(repr(exc))
    ths = [_th.Thread(target=hammer, args=(s,)) for s in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert not errs, errs[:2]
    assert tc.nbytes == sum(v.nbytes for v in tc._lru.values()), \
        "TileCache byte accounting drifted under threads (unlocked read-modify-write)"
    assert tc.nbytes <= tc.budget

    # ── 6. THE invariant: serial ≡ parallel, including order-dependent ids ─────
    if not _HAVE_SKIMAGE:
        _ok("parallel: sizing/maps/fold/pool/cache (node parity SKIPPED — no skimage)")
        return
    import hashlib
    import os
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider

    ax = AxisSizes(m=1, t=2, z=3, c=2, y=64, x=72)
    meta = {"pixel_size_um": 0.1, "z_step_um": 0.3, "objective_na": 1.4,
            "channel_emission_nm": [520.0, 640.0], "bit_depth": 12}
    zz, yy, xx = np.mgrid[0:ax.z, 0:ax.y, 0:ax.x].astype(float)
    vol = np.zeros((ax.m, ax.t, ax.z, ax.c, ax.y, ax.x))
    for t in range(ax.t):
        for c in range(ax.c):
            f = np.zeros((ax.z, ax.y, ax.x))
            for k, (cy, cx) in enumerate([(16, 18), (16, 54), (48, 18), (48, 54),
                                          (32, 36)]):
                s = 5.0 + 1.5 * k + 0.5 * t
                f += 900.0 * np.exp(-(((yy - cy - t) ** 2 + (xx - cx - c) ** 2)
                                      / (2 * s * s) + ((zz - 1.0) ** 2) / 4.0))
            vol[0, t, :, c] = f + 30.0 + 2.0 * ((yy + xx + zz) % 7)

    def digest_payload(p) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(p.axes).encode())
        for lk in sorted(p.attributes, key=lambda k: (k[0].value, k[1] or "", k[2])):
            v = getattr(p.attributes[lk], "values", None)
            h.update(repr(lk).encode())
            if isinstance(v, np.ndarray):
                h.update(str(v.dtype).encode())
                h.update(np.ascontiguousarray(v).tobytes())
        for sk in sorted(getattr(p, "structures", {}) or {}, key=repr):
            tbl = p.structures[sk]
            h.update(repr(sk).encode())
            for col in sorted(tbl.columns):
                h.update(col.encode())
                h.update(np.ascontiguousarray(tbl.columns[col]).tobytes())
        return h.hexdigest()

    # every one of these folds a running counter (region ids / Label rows / point ids),
    # so an out-of-order fold changes the NUMBERS, not just the timing
    cases = [
        ("analysis.segment", {"method": "threshold", "level": "otsu", "dim": "2D"}, {}),
        ("analysis.segment", {"method": "watershed", "level": "otsu", "dim": "3D"},
         {"min_distance": 0.3}),
        ("detect.spots", {"dim": "2D"}, {}),
        ("analysis.threshold", {"method": "otsu"}, {}),
    ]
    for op, modes, params in cases:
        got = {}
        for label, forced in (("serial", 1), ("parallel", None)):
            g = Graph()
            g.add(NodeInstance("S", "test.par_seed"))
            g.add(NodeInstance("N", op, params=dict(params), modes=dict(modes)))
            g.connect("S", "N")
            seed = Dataset(axes=ax, metadata=dict(meta)).with_image(
                ArrayProvider(vol.copy(), tile=32))
            eng = Engine(g, computes=COMPUTES, seeds={"S": seed},
                         meta_seeds={"S": MetaEnvelope(axes=ax, metadata=dict(meta))},
                         cache_bytes=1 << 26)
            if forced == 1:                       # force the serial path for the reference
                old = os.environ.get("NODEGRAPH_PARALLEL")
                os.environ["NODEGRAPH_PARALLEL"] = "off"
                try:
                    got[label] = digest_payload(eng.pull("N"))
                finally:
                    if old is None:
                        os.environ.pop("NODEGRAPH_PARALLEL", None)
                    else:
                        os.environ["NODEGRAPH_PARALLEL"] = old
            else:
                got[label] = digest_payload(eng.pull("N"))
        assert got["serial"] == got["parallel"], \
            f"{op} {modes}: parallel result differs from serial ({got})"

    # a binary mask is uint8, not int64 — 8× the bytes for one bit (V2.14)
    g = Graph()
    g.add(NodeInstance("S", "test.par_seed"))
    g.add(NodeInstance("T", "analysis.threshold", modes={"method": "otsu"}))
    g.connect("S", "T")
    seed = Dataset(axes=ax, metadata=dict(meta)).with_image(
        ArrayProvider(vol.copy(), tile=32))
    e = Engine(g, computes=COMPUTES, seeds={"S": seed},
               meta_seeds={"S": MetaEnvelope(axes=ax, metadata=dict(meta))})
    mk = e.pull("T").get(D.VOXEL, "mask")
    assert mk is not None and mk.values.dtype == np.uint8, mk.values.dtype
    assert set(np.unique(mk.values)) <= {0, 1}

    # fold_units STREAMS within a batch (V2.17): each result is folded as it lands, not
    # after the batch's slowest unit. Invisible in the output, load-bearing for the progress
    # bar — the catalog's eager nodes tick from the fold, so a materialised batch makes a
    # per-plane bar stand still and then jump `batch` at once.
    _stamps: list = []

    def _staggered(u):
        time.sleep(0.02 * (u % 4 + 1))         # unequal costs → distinguishable arrivals
        return u

    P.fold_units(_staggered, list(range(8)),
                 lambda i, it, r: _stamps.append((i, time.perf_counter())),
                 workers=4, batch=4)
    assert [i for i, _ in _stamps] == list(range(8)), _stamps   # order still exact
    _instants = {round(t, 3) for _, t in _stamps}
    assert len(_instants) >= 6, (
        f"fold_units materialised its batch: 8 units of unequal cost folded at only "
        f"{len(_instants)} distinct instants, so a progress bar would jump in bursts")

    _ok("parallel (V2.14): machine-derived budgets; map_units order/raise/re-entrancy; "
        "fold_units ordered fold + prepare + batching + streams each result as it lands "
        "(V2.17, per-unit progress); spawn pool round-trip + "
        "unpicklable-closure fallback; TileCache 8-thread accounting exact; "
        "serial≡parallel byte-identical on 4 order-dependent nodes (segment 2D/3D ids, "
        "spot ids, mask) and masks are uint8")


# ── checkpoint + Dock node (V2.18) ───────────────────────────────────────────

_DOCK_AX = AxisSizes(m=1, t=2, z=3, c=2, y=40, x=56)


def _dock_fixture():
    """A Dataset carrying one of everything a bake must preserve: a raster, a full-size
    Voxel mask, a coarse lattice layer and a structure table (with its z_kind)."""
    from nodegraph.provider import ArrayProvider
    from nodegraph.structure import StructureTable
    n = int(np.prod((1, 2, 3, 2, 40, 56)))
    raw = (np.arange(n, dtype=np.uint16) % 977).reshape(1, 2, 3, 2, 40, 56)
    ds = Dataset(axes=_DOCK_AX,
                 metadata={"pixel_size_um": 0.325, "z_step_um": 1.0, "dt_s": 30.0,
                           "bit_depth": 12})
    ds = ds.with_image(ArrayProvider(raw, tile=16))
    ds = ds.with_layer(Domain.VOXEL, "mask", (raw % 7) > 3)
    ds = ds.with_layer(Domain.FRAME, "drift_y", np.arange(2, dtype=float).reshape(1, 2))
    ds = ds.with_structure(StructureTable(
        Domain.POINT,
        {"id": np.arange(3), "m": np.zeros(3, int), "t": np.zeros(3, int),
         "c": np.zeros(3, int), "z": np.zeros(3, float),
         "y": np.arange(3, dtype=float), "x": np.arange(3, dtype=float) * 2},
        layer="spots", z_kind="plane_index"))
    return ds, raw


def test_checkpoint_dock() -> None:
    """The Dock node + its on-disk checkpoint (V2.18).

    Two things are being proved, and the second is the whole feature:

    1. **A checkpoint is a lossless, LAZY Dataset.** Raster, Voxel mask, coarse lattice
       layer, structure columns and provenance all round-trip, and the Voxel layer comes
       back memory-mapped rather than loaded — otherwise docking after a segmentation
       would re-materialize exactly the full raster it exists to get rid of.
    2. **A docked node's upstream is never evaluated.** Asserted by counting compute
       calls, not by inspecting the graph: the chain behind the dock must run zero times
       while the result stays byte-identical to the live one.
    """
    import os
    import shutil
    import sys
    import tempfile

    from nodegraph import checkpoint as CP
    from nodegraph.domains import Domain
    from nodegraph.memo import payload_bytes
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodelab_v2.ops import (
        BAKE_KEY, DOCK_OP, cut_docked_inputs, dock_seeds, dock_status, dormant_nodes,
        ensure_ops, prepare_run_graph, upstream_signature)

    ensure_ops()
    root = tempfile.mkdtemp(prefix="ng-dock-")
    try:
        ds, raw = _dock_fixture()

        # ── 1. round-trip, lossless + lazy ────────────────────────────────────
        d1 = os.path.join(root, "cp1")
        man = CP.write_checkpoint(ds, d1, precision="float32", bake_id="b1")
        assert man["version"] == CP.CHECKPOINT_VERSION and man["bake_id"] == "b1"
        back = CP.open_checkpoint(d1)
        assert back.axes == _DOCK_AX
        assert back.metadata["pixel_size_um"] == 0.325
        assert back.metadata["bit_depth"] == 12          # float32 does not restamp depth
        assert back.structure_zkind(Domain.POINT, "spots") == "plane_index"
        assert np.array_equal(
            np.asarray(back.get(Domain.VOXEL, "mask").values), (raw % 7) > 3)
        assert np.array_equal(back.get(Domain.FRAME, "drift_y").values,
                              np.arange(2, dtype=float).reshape(1, 2))
        assert np.array_equal(back.get(Domain.POINT, "x", "spots").values,
                              np.arange(3, dtype=float) * 2)
        got = np.stack([back.image.get_region(0, 0, t, z, c, 0, 40, 0, 56)
                        for t in range(2) for z in range(3) for c in range(2)])
        want = np.stack([raw[0, t, z, c]
                         for t in range(2) for z in range(3) for c in range(2)])
        # dtype is uint16, NOT float32: the precision choice applies to floating-point
        # data only, so a raw camera raster is stored as itself whatever is asked for.
        assert got.dtype == np.uint16 and np.array_equal(got, want), \
            "an integer raster must round-trip exactly, at its own dtype"
        # a FLOAT raster is the case precision actually governs
        fl = Dataset(axes=AxisSizes(t=2, y=8, x=8)).with_image(
            ArrayProvider(np.linspace(3.0, 900.0, 128).reshape(1, 2, 1, 1, 8, 8), tile=8))
        dfl = os.path.join(root, "cpf")
        CP.write_checkpoint(fl, dfl, precision="float32", bake_id="bf")
        fback = CP.open_checkpoint(dfl)
        fplane = fback.image.get_region(0, 0, 1, 0, 0, 0, 8, 0, 8)
        assert fplane.dtype == np.float32
        assert np.allclose(fplane,
                           np.linspace(3.0, 900.0, 128).reshape(2, 8, 8)[1], rtol=1e-6)
        # THE laziness invariant: a Voxel layer is a memmap, and an AttributeLayer must
        # not have copied it (that copy is precisely the retained raster docking removes).
        vals = back.get(Domain.VOXEL, "mask").values
        assert isinstance(vals, np.memmap), type(vals).__name__
        assert not vals.flags.writeable
        # …and the Memo GC must not COUNT it: those bytes live in the OS page cache
        # against a file, so evicting the entry frees nothing, and counting them would
        # have the GC discard real, freeable entries to "reclaim" memory it cannot.
        mask_bytes = int(((raw % 7) > 3).nbytes)
        eager = back.with_layer(Domain.VOXEL, "mask_copy", np.asarray(vals))
        assert payload_bytes(eager) - payload_bytes(back) == mask_bytes, \
            "an in-RAM copy of the same layer MUST be counted (the guard is memmap-only)"
        assert payload_bytes(back) < mask_bytes // 10, (
            "a lazily-opened checkpoint retains only its small tables, not its rasters "
            f"(got {payload_bytes(back)} vs a {mask_bytes}-byte mask)")

        # a pyramid was built, and it halves
        assert back.image.levels >= 2
        assert back.image.level_axes(1).y == 20 and back.image.level_axes(1).x == 28

        # ── 2. precision rules ────────────────────────────────────────────────
        assert CP.target_dtype(np.dtype(np.uint16), "float32") == np.uint16, \
            "integer data must pass through every precision untouched"
        assert CP.target_dtype(np.dtype(bool), "float64") == np.bool_
        assert CP.target_dtype(np.dtype(np.float64), "float32") == np.float32
        assert CP.target_dtype(np.dtype(np.float64), "uint16") == np.uint16
        d2 = os.path.join(root, "cp2")
        CP.write_checkpoint(ds, d2, precision="uint16", bake_id="b2")
        m2 = CP.read_manifest(d2)
        # uint16 on ALREADY-integer data changes nothing, so bit_depth is NOT restamped
        assert m2["image"]["dtype"] == "uint16" and not m2["image"]["restamped_bit_depth"]
        assert CP.read_manifest(d2)["metadata"]["bit_depth"] == 12
        # …but normalized float data is REFUSED rather than quantized to 0/1
        norm = Dataset(axes=AxisSizes(y=8, x=8)).with_image(
            ArrayProvider(np.linspace(0, 1, 64).reshape(1, 1, 1, 1, 8, 8), tile=8))
        try:
            CP.write_checkpoint(norm, os.path.join(root, "cp3"), precision="uint16")
            raise AssertionError("expected a refusal for [0,1] float at uint16")
        except ValueError as exc:
            assert "float32" in str(exc) and "0/1" in str(exc), str(exc)

        # ── 3. an incomplete bake reads as ABSENT, never as a torn store ───────
        d4 = os.path.join(root, "cp4")
        CP.write_checkpoint(ds, d4, precision="float32", bake_id="b4")
        os.remove(os.path.join(d4, CP.MANIFEST_NAME))
        assert CP.read_manifest(d4) is None and not CP.is_checkpoint(d4)
        try:
            CP.open_checkpoint(d4)
            raise AssertionError("expected a FileNotFoundError for a torn store")
        except FileNotFoundError as exc:
            assert "Re-bake" in str(exc)
        assert CP.checkpoint_envelope(d4) is None
        # a FUTURE format is the one hard error — mis-reading it would give wrong pixels
        with open(os.path.join(d4, CP.MANIFEST_NAME), "w", encoding="utf-8") as f:
            f.write('{"version": 999}')
        try:
            CP.read_manifest(d4)
            raise AssertionError("expected a refusal for a newer checkpoint format")
        except ValueError as exc:
            assert "newer version" in str(exc)

        # ── 4. the envelope a docked node seeds from ──────────────────────────
        env = CP.checkpoint_envelope(d1)
        assert env.axes == _DOCK_AX and env.metadata["dt_s"] == 30.0
        assert {Domain.VOXEL, Domain.FRAME, Domain.POINT} <= env.domains
        assert env.layers_in(Domain.VOXEL) == ("mask",)
        assert env.layers_in(Domain.POINT) == ("spots",), \
            "a structure layer's user-facing name is its LAYER, not its column"

        # ── 5. the node: the upstream chain is NOT evaluated ──────────────────
        runs = {"n": 0}

        def counting_gaussian(ctx):
            runs["n"] += 1
            return COMPUTES["enhance.gaussian"](ctx)

        computes = dict(COMPUTES)
        computes["enhance.gaussian"] = counting_gaussian
        seed = Dataset(axes=_DOCK_AX, metadata=dict(ds.metadata)).with_image(
            ArrayProvider(raw, tile=16))
        m_seed = {"src": MetaEnvelope(axes=_DOCK_AX, metadata=dict(ds.metadata))}

        def build(modes, params):
            g = Graph()
            g.add(NodeInstance("src", "io.load", params={"path": ""}))
            g.add(NodeInstance("blur", "enhance.gaussian", params={"sigma": 1.0},
                               modes={"dim": "2D"}))
            g.add(NodeInstance("dock", DOCK_OP, params=dict(params), modes=dict(modes)))
            g.add(NodeInstance("thr", "analysis.threshold", modes={"method": "otsu"}))
            g.connect("src", "blur", src_socket="image", dst_socket="data")
            g.connect("blur", "dock", src_socket="out", dst_socket="data")
            g.connect("dock", "thr", src_socket="out", dst_socket="data")
            return g

        def engine_for(g):
            seeds = {"src": seed}
            seeds.update(dock_seeds(g))
            return Engine(g, computes=computes, seeds=seeds, meta_seeds=m_seed)

        live_g = prepare_run_graph(build({"state": "live", "precision": "float32"}, {}))
        e_live = engine_for(live_g)
        live_dock = e_live.pull("dock")
        live_mask = np.asarray(
            e_live.pull("thr").get(Domain.VOXEL, "mask").values).copy()
        assert runs["n"] == 1, runs
        assert live_dock.image is not None, "a live dock passes its input straight through"

        dstore = os.path.join(root, "node")
        sig = upstream_signature(build({"state": "live"}, {}), "dock")
        CP.write_checkpoint(live_dock, dstore, precision="float32", bake_id="bk",
                            domains=e_live.env("dock").domains or None,
                            layer_names=e_live.env("dock").layer_names or None)

        runs["n"] = 0
        dmodes = {"state": "docked", "precision": "float32"}
        dparams = {"store": dstore,
                   BAKE_KEY: {"id": "bk", "sig": sig, "precision": "float32"}}
        g_full = build(dmodes, dparams)
        g_run = prepare_run_graph(g_full)
        assert not g_run.preds("dock"), "a docked node must be a ROOT in the run graph"
        assert len(g_full.preds("dock")) == 1, "the DOCUMENT keeps the chain"
        e_dock = engine_for(g_run)
        docked_mask = np.asarray(
            e_dock.pull("thr").get(Domain.VOXEL, "mask").values)
        assert runs["n"] == 0, (
            "the chain behind a docked node must never run — this is the entire point "
            "of the feature; got %d compute calls" % runs["n"])
        assert np.array_equal(docked_mask, live_mask), \
            "a docked run must produce the same answer as the live one"

        # ── 6. dormancy: what greys out, and what must NOT ────────────────────
        assert dormant_nodes(g_full) == frozenset({"src", "blur"})
        assert dock_status(g_full, "dock")[0] == "docked"
        g_branch = build(dmodes, dparams)
        g_branch.add(NodeInstance("peek", "view.viewer"))
        g_branch.connect("blur", "peek", src_socket="out", dst_socket="data")
        assert "blur" not in dormant_nodes(g_branch), (
            "a node feeding a LIVE branch as well as a docked one is still evaluated — "
            "greying it out would say the opposite of what runs")
        assert cut_docked_inputs(build({"state": "live"}, {})).edges, \
            "a live dock's edges are untouched"

        # ── 7. staleness: reported, never acted on ────────────────────────────
        g_edit = build(dmodes, dparams)
        g_edit.nodes["blur"].params["sigma"] = 9.0
        st, why = dock_status(g_edit, "dock")
        assert st == "stale" and "upstream" in why, (st, why)
        runs["n"] = 0
        still = engine_for(prepare_run_graph(g_edit)).pull("dock")
        assert runs["n"] == 0 and still.image is not None, \
            "a stale dock keeps serving its bake — it must not silently recompute"
        assert dock_status(build({"state": "docked", "precision": "uint16"},
                                 dparams), "dock")[0] == "stale"
        # an edit ABOVE an already-docked node does not stale a dock further downstream:
        # the walk stops at the docked node, whose own bake id stands in for its chain
        g_stop = build(dmodes, dparams)
        assert (upstream_signature(g_stop, "thr")
                == upstream_signature(g_edit, "thr")), \
            "the signature walk must stop at a docked node"

        # ── 8. refusals name the fix ──────────────────────────────────────────
        for params, want in (({}, "no dock folder"),
                             ({"store": dstore + "-gone"}, "no finished checkpoint"),
                             ({"store": dstore, BAKE_KEY: {"id": "someone-else"}},
                              "DIFFERENT bake")):
            gg = prepare_run_graph(build({"state": "docked"}, params))
            eng = Engine(gg, computes=computes, seeds={"src": seed, "dock": Dataset()},
                         meta_seeds=m_seed)
            try:
                eng.pull("dock")
                raise AssertionError(f"expected a refusal mentioning {want!r}")
            except ValueError as exc:
                assert want in str(exc), str(exc)

        # ── 9. Memo.drop_nodes — the "unload it" half ─────────────────────────
        m = Memo()
        e9 = Engine(prepare_run_graph(build({"state": "live"}, {})), computes=computes,
                    seeds={"src": seed}, meta_seeds=m_seed, memo=m)
        e9.pull("thr")
        before = len(m._entries)
        assert m.drop_nodes(["src", "blur"]) == 2 and len(m._entries) == before - 2
        assert m.drop_nodes([]) == 0
        assert e9.pull("thr") is not None       # a drop only ever costs a recompute

        # ── 10. remove_checkpoint refuses anything it did not write ───────────
        safe = os.path.join(root, "not-a-dock")
        os.makedirs(safe, exist_ok=True)
        with open(os.path.join(safe, "precious.txt"), "w", encoding="utf-8") as f:
            f.write("user data")
        try:
            CP.remove_checkpoint(safe)
            raise AssertionError("expected a refusal to delete a non-checkpoint")
        except ValueError as exc:
            assert "refusing to delete" in str(exc)
        assert os.path.isfile(os.path.join(safe, "precious.txt"))
        # d2 was written but never OPENED, so nothing maps it and the delete completes.
        CP.remove_checkpoint(d2)
        assert not os.path.isdir(d2)
        # d1 IS mapped (`back`/`vals`/`eager` above), and on Windows a mapped file cannot
        # be unlinked — the delete must SAY it failed rather than silently leaving the
        # whole checkpoint on disk while reporting success.
        if sys.platform.startswith("win"):
            try:
                CP.remove_checkpoint(d1)
                raise AssertionError("expected an OSError deleting a mapped checkpoint")
            except OSError as exc:
                assert "still open" in str(exc), str(exc)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    _ok("checkpoint + Dock (V2.18): a bake round-trips a whole Dataset (raster pyramid, "
        "Voxel mask, coarse lattice, structure columns + z_kind, calibration) and re-opens "
        "LAZY — the Voxel layer comes back as a read-only memmap the AttributeLayer does "
        "not copy and the Memo GC counts as 0 bytes; integer data survives every precision "
        "and [0,1] float is REFUSED at uint16 instead of collapsing to 0/1; a torn bake "
        "reads as absent and a future format is a hard error; and the node itself proves "
        "the point — a docked pull runs its upstream chain ZERO times and still returns "
        "byte-identical results, greys only nodes with no live consumer, reports staleness "
        "without ever acting on it, stops the signature walk at an upstream dock, names the "
        "fix in all 3 refusals, and drop_nodes releases exactly the docked-away entries")


#: The lab's real WellA3 pair, as numbers. Both files' geometry is reproduced here exactly
#: — extents, pixel sizes, stage logs, focus logs, Julian timestamps — so the placement
#: maths is gated on the acquisition it was designed for without needing a 40 GB ND2 in the
#: test environment. Read from the files on 2026-07-31 via `read_nd2_metadata_extended`.
_W3_PRIMARY_XY = [(-14088.6, -29462.8), (-14382.4, -29462.9), (-14382.3, -29169.1),
                  (-14088.5, -29168.9), (-11580.8, -19689.0), (-11874.6, -19688.8),
                  (-11874.8, -19394.9), (-11581.0, -19394.8), (-11368.9, -24369.9),
                  (-11662.8, -24369.5), (-11662.8, -24075.9), (-11369.0, -24075.8)]
_W3_PRIMARY_Z = [5971.96, 5971.98, 5971.92, 5971.64, 5996.22, 5995.76,
                 5996.06, 5996.30, 5981.10, 5981.30, 5981.50, 5981.70]
_W3_PRIMARY_PS = 0.28709571808614537
_W3_SECOND_PS = 1.7182777601481225
#: the GFP file's 7x7 contiguous montage: x steps of one field width, y likewise
_W3_SECOND_X0, _W3_SECOND_Y0 = -9145.5, -30361.5
_W3_SECOND_STEP = 1759.5


def _w3_secondary_xy():
    """The GFP file's 49 stage positions, row-major, exactly as the scope wrote them."""
    return [(_W3_SECOND_X0 - c * _W3_SECOND_STEP, _W3_SECOND_Y0 + r * _W3_SECOND_STEP)
            for r in range(7) for c in range(7)]


def test_overlay_placement() -> None:
    """``view.overlay`` must place one FILE inside another's field, or refuse.

    This is the gate for the whole placement spine, and it is written against the lab's
    real WellA3 acquisition because the failure modes it exists to prevent are all
    invisible to a tidy synthetic fixture:

    * the primary's 294 µm field does not sit inside one secondary tile — six of twelve
      positions straddle two, and ``m10`` draws from **four**. A design that assumed one
      source tile would show three-quarters of a field and a black corner;
    * the montage's tiles do not abut to the micron (a 0.18 µm seam), so an exact-coverage
      test calls five of twelve fields defective;
    * the two files' RELATIVE clocks start 1207.3 s apart, so only the absolute Julian one
      can report how far apart paired frames really are;
    * ``stagePositionUm.z`` is constant down a stack, so placing the GFP plane in the 640
      stack needs the ZStackLoop anchoring, not just the focus.

    Pure arithmetic over metadata dicts — no ND2 file is opened."""
    from nodegraph.placement import (
        field_box, pair_timepoints, plan_placement, tiles_covering, z_um_of_slice)
    from nodegraph.nodes import COMPUTES, OVERLAY_KEY, SAMPLING_KEY
    from nodegraph.provider import ArrayProvider

    pri_md = {
        "pixel_size_um": _W3_PRIMARY_PS, "z_step_um": 0.288, "bit_depth": 12,
        "stage_xy_um": _W3_PRIMARY_XY, "stage_z_um": _W3_PRIMARY_Z,
        "z_home_index": 0, "z_bottom_to_top": True,
        "frame_time_jd": [2461249.623022976 + i * (1421.9 / 86400.0) for i in range(16)],
    }
    pri_ax = AxisSizes(m=12, t=16, z=210, c=1, y=1024, x=1024)
    sec_md = {
        "pixel_size_um": _W3_SECOND_PS, "bit_depth": 12,
        "stage_xy_um": _w3_secondary_xy(),
        "stage_z_um": [5999.74 + 1.7 * i for i in range(49)],
        "frame_time_jd": [2461249.650969694 + i * (1417.8 / 86400.0) for i in range(16)],
    }
    sec_ax = AxisSizes(m=49, t=16, z=1, c=1, y=1024, x=1024)

    # ── the multi-tile fact this whole node exists for ────────────────────────────
    plan = plan_placement(pri_md, pri_ax, sec_md, sec_ax)
    assert plan.ok, plan.refusals
    assert abs(plan.scale - _W3_SECOND_PS / _W3_PRIMARY_PS) < 1e-9
    assert abs(plan.scale - 5.985) < 0.01, plan.scale
    n_tiles = {m: len(plan.tiles[m]) for m in plan.tiles}
    assert max(n_tiles.values()) == 4, n_tiles      # m10 straddles four GFP tiles
    assert sorted(n_tiles.values()) == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4], n_tiles
    for m, cov in plan.coverage.items():
        assert cov > 0.99, (m, cov)                # every field IS covered
    # ...and the 0.18 µm seam between montage rows is NOT reported as a hole
    assert not any("partly covered" in w for w in plan.warnings), plan.warnings
    # the magnification IS reported — it is why the secondary looks blocky
    assert any("5.99x" in w or "magnified" in w for w in plan.warnings), plan.warnings

    # ── absolute clock: index pairing, honest error ───────────────────────────────
    pairs = pair_timepoints(pri_md, 16, sec_md, 16)
    assert all(j == t for t, j, _ in pairs)
    assert abs(pairs[0][2] - 2414.6) < 2.0, pairs[0]        # GFP[0] is 2415 s after 640[0]
    # Nearest-in-time runs the OTHER way round: the GFP frame is the SECONDARY, and
    # GFP[0] is nearest to 640[2], so the shift that would chase the clock is -2 (primary
    # t pairs with secondary t-2) — which is exactly why nearest-time is not the default.
    shifted = pair_timepoints(pri_md, 16, sec_md, 16, shift=-2)
    assert abs(shifted[2][2] + 429.0) < 5.0, shifted[2]     # and the nearest one is EARLIER
    assert shifted[0][1] is None and shifted[0][2] is None   # shift walks off the start
    off_end = plan_placement(pri_md, pri_ax, sec_md, sec_ax, t_shift=2)
    assert any("no secondary frame" in w for w in off_end.warnings), off_end.warnings

    # ── Z: the stack anchoring is what places a plane inside a stack ──────────────
    box0 = field_box(pri_md, pri_ax, 0)
    assert abs(box0.z0 - 5971.96) < 1e-6 and abs(box0.z1 - (5971.96 + 209 * 0.288)) < 1e-6
    gfp_z = z_um_of_slice(sec_md, sec_ax, 3, 0)               # a single plane IS its focus
    assert abs(gfp_z - sec_md["stage_z_um"][3]) < 1e-9
    slice_index = (gfp_z - box0.z0) / 0.288
    assert 0 < slice_index < pri_ax.z, slice_index            # it falls INSIDE the stack
    # without the ZStackLoop anchoring a multi-slice stack must refuse, not assume slice 0
    no_anchor = dict(pri_md); no_anchor.pop("z_home_index")
    assert z_um_of_slice(no_anchor, pri_ax, 0, 100) is None

    # ── the nudge moves tile SELECTION, not just the drawing ─────────────────────
    near = tiles_covering(pri_md, pri_ax, 2, sec_md, sec_ax)
    far = tiles_covering(pri_md, pri_ax, 2, sec_md, sec_ax,
                         offset_um=(0.0, 0.0, 3.0 * _W3_SECOND_STEP))
    assert [j for j, _ in near] != [j for j, _ in far], (near, far)

    # ── refusals: every one is a case that would otherwise LOOK right ────────────
    moved = dict(sec_md)
    bad = plan_placement(pri_md, pri_ax, moved, sec_ax,
                         dst_sampling=(), src_sampling=("align.drift",))
    # The refusal now comes from the PER-INPUT guard rather than a blanket "the chains
    # differ" test: what makes this unsafe is that the drifted side has no maintained
    # origin_um AND its geometry moved, so its stage log has stopped describing it. Two
    # inputs that merely went through different geometry, both with true origins, are fine
    # — see test_overlay_after_geometry_change.
    assert not bad.ok and any("align.drift" in r and "no maintained origin_um" in r
                              for r in bad.refusals), bad
    no_log = {k: v for k, v in sec_md.items() if k != "stage_xy_um"}
    assert any("no stage position log" in r
               for r in plan_placement(pri_md, pri_ax, no_log, sec_ax).refusals)
    short = dict(sec_md); short["stage_xy_um"] = sec_md["stage_xy_um"][:30]
    assert any("covers 30 of 49" in r
               for r in plan_placement(pri_md, pri_ax, short, sec_ax).refusals)
    no_ps = {k: v for k, v in sec_md.items() if k != "pixel_size_um"}
    assert any("no pixel_size_um" in r
               for r in plan_placement(pri_md, pri_ax, no_ps, sec_ax).refusals)

    # ── end-to-end through the Engine: primary passes through UNTOUCHED ──────────
    pri_img = ArrayProvider(np.zeros((12, 16, 210, 1, 4, 4), dtype=np.uint16))
    sec_img = ArrayProvider(np.ones((49, 16, 1, 1, 4, 4), dtype=np.uint16))
    pri_ds = Dataset(axes=pri_ax, metadata=pri_md).with_image(pri_img)
    sec_ds = Dataset(axes=sec_ax, metadata=sec_md).with_image(sec_img)
    g = Graph()
    g.add(NodeInstance("A", "io.load")); g.add(NodeInstance("B", "io.load"))
    g.add(NodeInstance("O", "view.overlay", params={"opacity": 0.35}))   # presentation
    g.connect("A", "O"); g.connect("B", "O", dst_socket="secondary")
    eng = Engine(g, computes=COMPUTES, seeds={"A": pri_ds, "B": sec_ds})
    out = eng.pull("O")
    assert out.axes == pri_ax                       # canvas is the primary's, unchanged
    assert out.image is pri_img                     # and its pixels are not even touched
    assert out.metadata["pixel_size_um"] == pri_md["pixel_size_um"]
    recipe = out.metadata[OVERLAY_KEY]
    assert len(recipe) == 2 and recipe[0]["role"] == "base"
    e = recipe[1]
    # `opacity` is a PRESENTATION socket, so it must NOT be in the payload — a value that
    # is excluded from the recipe hash and stamped anyway would be served stale on a hit.
    assert e["node"] == "O" and e["blend"] == "add" and "opacity" not in e
    assert dict(e["tiles"])[10] and len(dict(e["tiles"])[10]) == 4
    assert "m0 <- m10, m03" in e["note"] or "m0 <- m03, m10" in e["note"], e["note"]

    # ── chaining is how N-way works: a third source APPENDS ──────────────────────
    g.add(NodeInstance("C", "io.load"))
    g.add(NodeInstance("O2", "view.overlay"))
    g.connect("O", "O2"); g.connect("C", "O2", dst_socket="secondary")
    eng2 = Engine(g, computes=COMPUTES,
                  seeds={"A": pri_ds, "B": sec_ds, "C": sec_ds})
    chained = eng2.pull("O2").metadata[OVERLAY_KEY]
    assert len(chained) == 3, chained
    assert [r.get("node") for r in chained] == ["primary", "O", "O2"], chained

    # ── an unwired secondary is a wiring error, not a silent no-op ───────────────
    g3 = Graph()
    g3.add(NodeInstance("A", "io.load")); g3.add(NodeInstance("O", "view.overlay"))
    g3.connect("A", "O")
    try:
        Engine(g3, computes=COMPUTES, seeds={"A": pri_ds}).pull("O")
        raise AssertionError("an Overlay with no secondary must refuse")
    except ValueError as exc:
        assert "secondary" in str(exc)

    # ── the refusal reaches the user through the Engine, naming the cause ────────
    drifted = sec_ds.with_metadata(**{SAMPLING_KEY: ("align.drift",)})
    try:
        Engine(g, computes=COMPUTES,
               seeds={"A": pri_ds, "B": drifted, "C": sec_ds}).pull("O")
        raise AssertionError("a sampling-provenance mismatch must refuse")
    except ValueError as exc:
        assert "align.drift" in str(exc) and "no maintained origin_um" in str(exc)

    # ── the DISPLAY composite: the recipe actually fetches the right pixels ──────
    #
    # Every tile is filled with its own multipoint index, so the composed plane is a map of
    # WHICH tile was sampled where. If the placement is right, each tile's share of the
    # output equals its coverage fraction — an assertion that fails on an off-by-one in the
    # µm→index map, a swapped axis, or a tile drawn in the wrong place, none of which a
    # "does it produce an array" test would notice.
    from nodelab_v2.overlay_compose import axis_map, compose_secondary_plane, paired_t
    entry = {
        "tiles": [[m, [[j, f] for j, f in plan.tiles[m]]] for m in sorted(plan.tiles)],
        "offset_um": [0.0, 0.0, 0.0], "flip_x": True, "flip_y": False,
        "t_pairs": [[t, t, 2414.6] for t in range(16)]}
    tile_of = lambda j: np.full((64, 64), float(j), dtype=np.float32)
    for m in (10, 0, 2):
        comp = compose_secondary_plane(entry, (256, 256), pri_md, pri_ax, m,
                                       sec_md, sec_ax, tile_of)
        assert comp is not None and comp.shape == (256, 256)
        vals, counts = np.unique(comp, return_counts=True)
        got = {int(v): c / comp.size for v, c in zip(vals, counts)}
        want = dict(plan.tiles[m])
        assert set(got) == set(want), (m, sorted(got), sorted(want))
        for j, frac in want.items():
            assert abs(got[j] - frac) < 0.01, (m, j, got[j], frac)
        # the whole field is painted — no fill value survives where a tile covers it
        assert sum(got.values()) > 0.99, (m, got)

    # a 64² tile drawn onto a 256² display grid: the mapping is in µm, so the display
    # decimation costs nothing and no caller tracks a scale factor
    assert compose_secondary_plane(entry, (37, 91), pri_md, pri_ax, 2,
                                   sec_md, sec_ax, tile_of).shape == (37, 91)
    # outside the source extent is -1 (left unpainted), never clamped to the edge — a
    # clamp would smear the border row across a gap and make it look like data
    amap = axis_map(4, 0.0, 40.0, 2, 10.0, 30.0, flip=False)
    assert amap.tolist() == [-1, 0, 1, -1], amap.tolist()
    assert axis_map(4, 0.0, 40.0, 2, 10.0, 30.0, flip=True).tolist() == [-1, 1, 0, -1]
    # an unpaired timepoint draws nothing rather than a black plane
    assert paired_t(entry, 99) is None
    assert compose_secondary_plane({**entry, "tiles": []}, (16, 16), pri_md, pri_ax, 0,
                                   sec_md, sec_ax, tile_of) is None

    # ── the CHAIN walk: N-way overlay draws every source, not just the last ──────
    #
    # Chaining is how this node does N sources, so the display path has to walk back down
    # the primary spine and collect every `secondary` edge. Drawing only the viewed node's
    # own — the first implementation — recorded three sources correctly and rendered two,
    # which is the kind of gap you find by comparing a picture against a graph, i.e. late.
    from nodelab_v2.runner import EngineRunner, GL_MAX_CHANNELS
    gch = Graph()
    for nid in ("A", "B", "C", "D"):
        gch.add(NodeInstance(nid, "io.load"))
    for nid in ("O1", "O2", "O3"):
        gch.add(NodeInstance(nid, "view.overlay"))
    gch.connect("A", "O1"); gch.connect("B", "O1", dst_socket="secondary")
    gch.connect("O1", "O2"); gch.connect("C", "O2", dst_socket="secondary")
    gch.connect("O2", "O3"); gch.connect("D", "O3", dst_socket="secondary")
    assert EngineRunner.overlay_chain(gch, "O3") == [("O1", "B"), ("O2", "C"), ("O3", "D")]
    assert EngineRunner.overlay_chain(gch, "O1") == [("O1", "B")]     # base-first, partial
    assert EngineRunner.overlay_chain(gch, "A") == []                 # not an overlay
    # a self-referential primary must terminate rather than spin — the engine rejects
    # cycles, but this walks GUI-supplied structure mid-edit
    gcy = Graph()
    gcy.add(NodeInstance("X", "view.overlay")); gcy.add(NodeInstance("S", "io.load"))
    gcy.connect("X", "X"); gcy.connect("S", "X", dst_socket="secondary")
    assert EngineRunner.overlay_chain(gcy, "X") == [("X", "S")]
    assert GL_MAX_CHANNELS == 8, "the channel ceiling must track the shader's sampler bank"

    _ok("overlay placement (V2.19): the real WellA3 pair placed by stage µm — 12 primary "
        "fields resolve to 1/2/4 GFP tiles (m10 draws four), 5.985x pixel ratio reported, "
        "the 0.18 µm montage seam NOT called a hole, index t-pairing carries the absolute "
        "+2415 s from the shared Julian clock (chasing it would need shift=-2, nearest at "
        "-429 s, which walks off the start), a single plane places inside a 210-slice "
        "stack via the ZStackLoop "
        "anchoring and an unanchored stack REFUSES, a nudge re-selects tiles; the primary "
        "passes through with the same provider/axes/calibration, overlays chain to 3 "
        "sources, and 6 refusals fire (drift provenance, no log, partial log, no pixel "
        "size, unwired secondary) — each one a case that would otherwise render "
        "convincingly and be wrong; and the DISPLAY composite fetches from the tiles the "
        "plan chose, each one's share of the picture equal to its coverage fraction to "
        "within 1%, at any display shape, with off-field samples left unpainted rather "
        "than clamped to the border row; and the chain walk collects EVERY source down the "
        "primary spine, base-first, terminating on a self-referential primary")


def test_overlay_renderers() -> None:
    """The scalar / vector / comparison overlay maths.

    Each of these families has a failure mode that a screenshot cannot show you, which is
    the whole reason the arithmetic lives in a Qt-free module: a colour-mapped field looks
    convincing whether or not zero is where it claims, arrows look like a flow field whether
    or not their angles survived scaling, and a TP/FP/FN split looks decisive whether or not
    one object was counted twice."""
    from nodelab_v2.overlay_render import (
        COLORMAPS, colormap_lut, match_labels, match_points, normalize_for_map,
        quiver_arrows, scalar_rgba)

    # ── colormaps: sampled, complete, and ending where the table says ────────────
    for name in COLORMAPS:
        lut = colormap_lut(name)
        assert lut.shape == (256, 3) and lut.dtype == np.uint8, name
        assert tuple(lut[0]) == COLORMAPS[name][0], name
        assert tuple(lut[-1]) == COLORMAPS[name][-1], name
    assert np.array_equal(colormap_lut("nope"), colormap_lut("viridis"))
    grey = colormap_lut("grey")[:, 0].astype(int)
    assert np.all(np.diff(grey) >= 0), "a colormap must be monotonic in index"

    # ── scalar: absence, sign and the alpha ramp ─────────────────────────────────
    v = np.linspace(-2.0, 2.0, 9).reshape(3, 3)
    v = v.copy(); v[0, 0] = np.nan
    rgba = scalar_rgba(v, cmap="coolwarm", lo=-2, hi=2, center_zero=True,
                       alpha_mode="ramp", opacity=1.0)
    assert rgba.shape == (3, 3, 4) and rgba.dtype == np.uint8
    # NaN is ABSENCE, not the low end of the scale — it must never be painted
    assert rgba[0, 0, 3] == 0, "NaN must be fully transparent"
    # on a diverging map centred at zero, ZERO is the transparent, neutral end and the two
    # signs are distinguishable — the property a strain field is read by
    mid = scalar_rgba(np.zeros((1, 1)), cmap="coolwarm", lo=-2, hi=2, center_zero=True,
                      alpha_mode="ramp", opacity=1.0)
    assert mid[0, 0, 3] == 0, "zero must be transparent on a centred diverging ramp"
    ext = scalar_rgba(np.array([[-2.0, 2.0]]), cmap="coolwarm", lo=-2, hi=2,
                      center_zero=True, alpha_mode="ramp", opacity=1.0)
    assert ext[0, 0, 3] == 255 and ext[0, 1, 3] == 255
    assert tuple(ext[0, 0, :3]) != tuple(ext[0, 1, :3]), "the two signs must differ"
    # flat vs ramp vs gated really are three behaviours
    flat = scalar_rgba(np.array([[0.1, 0.9]]), lo=0, hi=1, alpha_mode="flat", opacity=1.0)
    assert flat[0, 0, 3] == flat[0, 1, 3] == 255
    ramp = scalar_rgba(np.array([[0.1, 0.9]]), lo=0, hi=1, alpha_mode="ramp", opacity=1.0)
    assert ramp[0, 0, 3] < ramp[0, 1, 3]
    gated = scalar_rgba(np.array([[0.1, 0.9]]), lo=0, hi=1, alpha_mode="gated",
                        threshold=0.5, opacity=1.0)
    assert gated[0, 0, 3] == 0 and gated[0, 1, 3] == 255
    # centring is IGNORED for a sequential map — it has no meaningful midpoint
    _t, lo_s, hi_s = normalize_for_map(np.array([-1.0, 3.0]), -1, 3, center_zero=True,
                                       cmap="viridis")
    assert (lo_s, hi_s) == (-1.0, 3.0)
    _t, lo_d, hi_d = normalize_for_map(np.array([-1.0, 3.0]), -1, 3, center_zero=True,
                                       cmap="coolwarm")
    assert (lo_d, hi_d) == (-3.0, 3.0), (lo_d, hi_d)
    # absent limits come from PERCENTILES, so a single outlier cannot flatten the field
    body = np.concatenate([np.linspace(0, 1, 999), [1e6]])
    _t, lo_p, hi_p = normalize_for_map(body)
    assert hi_p < 2.0, hi_p

    # ── vector: scaling must not rotate the field ────────────────────────────────
    tails, heads, mag = quiver_arrows([0, 10], [0, 10], [3, 0], [4, 0],
                                      scale=2.0, gate=1.0)
    assert len(tails) == 1                                   # the zero vector is gated out
    d = heads[0] - tails[0]
    assert abs(np.arctan2(d[0], d[1]) - np.arctan2(3.0, 4.0)) < 1e-12, "arrow rotated"
    assert abs(np.hypot(*d) - 2.0 * 5.0) < 1e-9, "arrow length is not scale x magnitude"
    assert abs(mag[0] - 5.0) < 1e-9
    # decimation and the hard arrow cap
    n = 1000
    t2, _h2, _m2 = quiver_arrows(np.arange(n), np.arange(n), np.ones(n), np.ones(n),
                                 every=10)
    assert len(t2) == 100
    t3, _h3, _m3 = quiver_arrows(np.arange(n), np.arange(n), np.ones(n), np.ones(n),
                                 max_arrows=50)
    assert len(t3) == 50
    # a NaN vector is dropped, never drawn as a zero-length arrow at the origin
    t4, _h4, _m4 = quiver_arrows([0, 1], [0, 1], [np.nan, 2], [1, 2], gate=0.0)
    assert len(t4) == 1

    # ── comparison: strictly one-to-one, or the counts are fiction ───────────────
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[1.0, 1.0], [1.5, 1.5], [100.0, 100.0]])
    pairs, only_a, only_b = match_points(a, b, max_dist=5.0)
    assert pairs.tolist() == [[0, 0]], pairs.tolist()   # b[1] cannot ALSO claim a[0]
    assert only_a.tolist() == [1] and only_b.tolist() == [1, 2]
    assert len(set(pairs[:, 0].tolist())) == len(pairs)
    assert len(set(pairs[:, 1].tolist())) == len(pairs)
    # nothing beyond max_dist pairs, however lonely
    assert match_points(a, np.array([[50.0, 50.0]]), max_dist=5.0)[0].size == 0
    # empty sides degrade to "everything is unmatched", not an exception
    p0, oa0, ob0 = match_points(np.zeros((0, 2)), b, max_dist=5.0)
    assert p0.size == 0 and oa0.size == 0 and ob0.tolist() == [0, 1, 2]

    ra = np.zeros((10, 10), dtype=int); ra[1:5, 1:5] = 1; ra[6:9, 6:9] = 2
    rb = np.zeros((10, 10), dtype=int); rb[1:5, 1:5] = 7; rb[0:2, 8:10] = 8
    lp, la, lb = match_labels(ra, rb, min_iou=0.5)
    assert lp.tolist() == [[1, 7]], lp.tolist()      # matched on OVERLAP, and by label id
    assert la.tolist() == [2] and lb.tolist() == [8]
    # IoU is about EXTENT: a concentric mask of half the area shares a centroid and fails
    rc = np.zeros((10, 10), dtype=int); rc[2:4, 2:4] = 5
    assert match_labels(ra, rc, min_iou=0.5)[0].size == 0
    assert match_labels(ra, rc, min_iou=0.2)[0].tolist() == [[1, 5]]
    try:
        match_labels(ra, np.zeros((5, 5), dtype=int))
        raise AssertionError("mismatched raster shapes must refuse")
    except ValueError as exc:
        assert "same grid" in str(exc)

    _ok("overlay renderers (V2.19): 5 colormaps sample to monotonic 256-entry LUTs; a "
        "scalar field paints NaN as ABSENCE (alpha 0, never the low end), puts zero at the "
        "neutral transparent midpoint of a centred diverging map with the two signs "
        "distinguishable, honours centring only for a diverging map, takes absent limits "
        "from percentiles so one outlier cannot flatten it, and gives flat/ramp/gated three "
        "real behaviours; quiver scaling preserves every ANGLE (the per-component bug) with "
        "decimation, a hard arrow cap and NaN dropped; and both matchers are strictly "
        "one-to-one — points by distance, labels by IoU over a joint histogram, where a "
        "concentric half-area mask correctly FAILS at 0.5 and passes at 0.2")


def test_overlay_resample() -> None:
    """``view.overlay``'s ``resample`` output mode bakes the placement into real pixels.

    ``display`` composites at draw time and nothing downstream sees it — right for looking,
    useless for measuring. ``resample`` puts the same placement into the Dataset, so
    ``analysis.measure`` can report the secondary's intensity inside objects the primary's
    chain segmented. Both modes go through the SAME compositor, which is why the pixels you
    measure cannot drift from the picture you looked at.

    The gate that matters is §8: an axis-changing node's payload must equal what its
    ``meta_transform`` predicted, or every edit-time widget re-seed is reading a shape the
    pull will not produce."""
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider

    # two 8 µm primary fields side by side, under ONE 32 µm secondary that covers both
    pri_md = {"pixel_size_um": 1.0, "stage_xy_um": [(4.0, 4.0), (12.0, 4.0)],
              "bit_depth": 12, "channel_names": ["640"]}
    pri_ax = AxisSizes(m=2, t=1, z=1, c=1, y=8, x=8)
    sec_md = {"pixel_size_um": 4.0, "stage_xy_um": [(8.0, 4.0)],
              "bit_depth": 12, "channel_names": ["GFP"]}
    sec_ax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    pri = Dataset(axes=pri_ax, metadata=pri_md).with_image(
        ArrayProvider(np.full((2, 1, 1, 1, 8, 8), 100, np.uint16)))
    sec = Dataset(axes=sec_ax, metadata=sec_md).with_image(
        ArrayProvider(np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 1, 8, 8)))
    env_p = MetaEnvelope(axes=pri_ax, metadata=dict(pri_md))

    def pull(mode, **params):
        g = Graph()
        g.add(NodeInstance("A", "io.load")); g.add(NodeInstance("B", "io.load"))
        g.add(NodeInstance("O", "view.overlay", modes={"output": mode},
                           params=dict(params)))
        g.connect("A", "O"); g.connect("B", "O", dst_socket="secondary")
        eng = Engine(g, computes=COMPUTES, seeds={"A": pri, "B": sec},
                     meta_seeds={"A": env_p})
        return eng, eng.pull("O")

    # display: nothing about the data changes — same axes, same depth, same names
    eng_d, disp = pull("display")
    assert disp.axes == pri_ax == eng_d.env("O").axes
    assert disp.metadata["bit_depth"] == 12
    assert disp.metadata["channel_names"] == ["640"]
    assert disp.image is pri.image                       # not one voxel touched

    # resample: C+1, and the PAYLOAD equals the header the meta_transform predicted (§8)
    eng_r, res = pull("resample")
    hdr = eng_r.env("O")
    assert res.axes.c == 2 and res.axes == hdr.axes, (res.axes, hdr.axes)
    assert len(res.metadata["channel_names"]) == len(hdr.metadata["channel_names"]) == 2
    assert res.metadata["channel_names"][1].startswith("ovl:")
    # the merged stack has no single declared integer scale — dropped in BOTH halves
    assert res.metadata.get("bit_depth") is None
    assert hdr.metadata.get("bit_depth") is None
    # the primary's own channel is untouched, and the new one carries real secondary pixels
    for m in (0, 1):
        assert (res.image.get_region(0, m, 0, 0, 0, 0, 8, 0, 8) == 100).all()
    got = {m: sorted(set(res.image.get_region(0, m, 0, 0, 1, 0, 8, 0, 8).ravel().tolist()))
           for m in (0, 1)}
    assert got[0] and got[1] and got[0] != got[1], got   # the two fields sample DIFFERENT
    # ...and the two 8 µm fields land on adjacent 4 µm columns of the secondary, mirrored
    # by the flip_x default this node shares with util.stitch
    assert got[0] == [28, 29, 36, 37], got[0]
    assert got[1] == [26, 27, 34, 35], got[1]
    _eng, unflipped = pull("resample", flip_x=False)
    assert sorted(set(unflipped.image.get_region(0, 0, 0, 0, 1, 0, 8, 0, 8)
                      .ravel().tolist())) == [26, 27, 34, 35]

    # the footprint is honest per mode: display reads nothing, resample reads across M
    spec = NODES.get("view.overlay")
    assert spec.resolve_granularity({"output": "display"}) is Granularity.TILEABLE
    assert spec.resolve_granularity({"output": "resample"}) is Granularity.MULTI_VIEW
    # and the two modes memoize apart
    assert eng_d.entry("O").recipe_hash != eng_r.entry("O").recipe_hash
    # an out-of-range channel is refused rather than clamped onto a neighbour
    try:
        pull("resample", secondary_channel=5)
        raise AssertionError("an out-of-range secondary channel must refuse")
    except ValueError as exc:
        assert "does not exist" in str(exc)

    _ok("overlay resample (V2.19): `display` leaves the payload byte-identical (same "
        "provider object, axes, depth and names) while `resample` bakes the placed "
        "secondary into a real C+1 channel whose axes and channel-name count EQUAL the "
        "meta_transform's edit-time prediction, with bit_depth dropped in both halves "
        "because the merged stack has no one integer scale and the edit-time pass cannot "
        "see the second file to check; the primary's own channel is untouched, the two "
        "fields sample different (and exactly the predicted) columns of the secondary with "
        "the flip_x default this node shares with Stitch, the footprint resolves "
        "TILEABLE/MULTI_VIEW per mode, the modes memoize apart, and an out-of-range "
        "secondary channel refuses instead of clamping onto a neighbour")


def test_overlay_override_and_presentation() -> None:
    """The user's two escape hatches from the overlay's own strictness.

    **The override.** Every refusal in :func:`nodegraph.placement.plan_placement` is about
    what the FILES can prove, not about what is true — a TIFF carries no stage log, a
    cropped source threw its origin away — and a person who knows the acquisition can still
    place them by hand. ``unplaceable=align_by_index`` lets them, and the test is that it
    stays loud: every refusal it walks past must reappear as a warning that says so, and the
    plan must record that it was placed by INDEX, because an index-placed picture looks
    exactly like a stage-placed one.

    **Presentation params.** Opacity/wipe/flicker change how a result is DRAWN, never what
    it contains, so they are excluded from the recipe hash — which is what makes dragging
    one a repaint instead of (in ``resample`` mode) re-reading four tiles. The other half of
    that contract is that such a value must not reach the payload either, or a memo hit
    would serve a recipe stamped with the value the slider used to have."""
    from nodegraph.nodes import COMPUTES, OVERLAY_KEY
    from nodegraph.placement import compose_secondary_plane, plan_placement
    from nodegraph.provider import ArrayProvider

    stageless = {"pixel_size_um": 1.0}                     # a TIFF: nothing to place with
    ax = AxisSizes(m=2, t=1, z=1, c=1, y=8, x=8)

    # refuse (the default) names the problem and stops
    strict = plan_placement(stageless, ax, stageless, ax)
    assert not strict.ok and strict.refusals
    assert strict.placed_by == "stage"

    # override places by index — and says so, in every channel a person might read
    loose = plan_placement(stageless, ax, stageless, ax,
                           on_unplaceable="align_by_index")
    assert loose.ok and loose.placed_by == "index"
    assert loose.tiles == {0: [(0, 1.0)], 1: [(1, 1.0)]}
    overridden = [w for w in loose.warnings if w.startswith("OVERRIDDEN")]
    assert len(overridden) == len(strict.refusals), (overridden, strict.refusals)
    assert any("NOT verified" in w for w in loose.warnings), loose.warnings
    assert "BY INDEX (override)" in loose.describe(0)
    # a secondary with fewer fields clamps rather than indexing off the end
    narrow = plan_placement(stageless, ax, stageless, AxisSizes(m=1, y=8, x=8),
                            on_unplaceable="align_by_index")
    assert narrow.tiles[1] == [(0, 1.0)], narrow.tiles

    # ...and the compositor can actually DRAW an index placement, which has no field boxes
    entry = {"tiles": [[m, [[j, f] for j, f in loose.tiles[m]]] for m in loose.tiles],
             "offset_um": [0.0, 0.0, 0.0], "flip_x": False, "flip_y": False,
             "placed_by": "index", "t_pairs": [[0, 0, None]]}
    src = np.arange(64, dtype=float).reshape(8, 8)
    drawn = compose_secondary_plane(entry, (16, 16), stageless, ax, 0,
                                    stageless, ax, lambda j: src)
    assert drawn is not None and drawn.shape == (16, 16)
    assert drawn.min() == 0 and drawn.max() == 63          # the whole plane, resized
    # a STAGE placement with no boxes still declines — the override is the only way in
    assert compose_secondary_plane({**entry, "placed_by": "stage"}, (16, 16), stageless,
                                   ax, 0, stageless, ax, lambda j: src) is None

    # ── presentation params: memo-neutral, and absent from the payload ───────────
    md = {"pixel_size_um": 1.0, "stage_xy_um": [(4.0, 4.0)]}
    one = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    a = Dataset(axes=one, metadata=md).with_image(
        ArrayProvider(np.zeros((1, 1, 1, 1, 8, 8), np.uint16)))
    b = Dataset(axes=one, metadata=md).with_image(
        ArrayProvider(np.ones((1, 1, 1, 1, 8, 8), np.uint16)))
    shared = Memo()          # one memo, so upstream REVISIONS are stable across the pulls

    def rh(**params):
        g = Graph()
        g.add(NodeInstance("A", "io.load")); g.add(NodeInstance("B", "io.load"))
        g.add(NodeInstance("O", "view.overlay", params=dict(params)))
        g.connect("A", "O"); g.connect("B", "O", dst_socket="secondary")
        eng = Engine(g, computes=COMPUTES, seeds={"A": a, "B": b}, memo=shared)
        out = eng.pull("O")
        return eng.entry("O").recipe_hash, out.metadata[OVERLAY_KEY][-1]

    base, entry0 = rh()
    assert rh()[0] == base, "the fixture itself must be stable before anything is asserted"
    for look in ("opacity", "wipe_pos", "flicker_hz"):
        assert rh(**{look: 0.77})[0] == base, f"{look} must not re-key the memo"
        assert look not in entry0, f"{look} must not reach the payload"
    assert rh(t_shift=1)[0] != base, "a DATA param must still re-key"
    assert rh(offset_x=3.0)[0] != base
    # the flag is declared where the GUI and the engine both read it
    spec = NODES.get("view.overlay")
    assert {i.name for i in spec.inputs if i.presentation} == {
        "opacity", "wipe_pos", "flicker_hz"}

    _ok("overlay override + presentation (V2.19): `align_by_index` lets a user place two "
        "files the FILES cannot prove line up — a TIFF, a cropped source — and stays loud "
        "about it: every refusal reappears as an OVERRIDDEN warning, the plan records "
        "placed_by='index', the readout says so, a short secondary clamps rather than "
        "running off the end, and the compositor draws it as an honest whole-plane resize "
        "while a stage placement with no boxes still declines; and opacity / wipe_pos / "
        "flicker_hz are memo-NEUTRAL (proved against a shared memo, so upstream revisions "
        "hold still) and absent from the payload, which together are what let the GUI "
        "restyle without re-running a resample bake")


def test_overlay_after_geometry_change() -> None:
    """An overlay must survive its inputs going through DIFFERENT geometry.

    Reported 2026-08-03: overlay after ``util.stitch`` refused, and so did overlay after a
    max-Z projection. One cause, and it was a design error rather than a slip — the
    sampling-provenance test was borrowed from ``_intensity_provider``, whose contract is
    voxel-for-voxel reading. An overlay does not read voxel-for-voxel; it places by physical
    µm, and keeping a true µm origin ACROSS crop / resample / z-project / stitch is the
    entire reason ``origin_um`` exists. Refusing on divergence therefore defeated the
    feature it was meant to protect, having already computed the placement correctly.

    What must NOT regress is the case the check was really guarding: placement resting on a
    stage log that has gone stale. That is not "the chains differ" — it is "this input has
    no maintained origin AND its geometry moved", which is where ``align.drift`` lands
    because it drops the key rather than carry a claim it cannot honour.
    """
    from nodegraph.nodes import COMPUTES, OVERLAY_KEY
    from nodegraph.placement import plan_placement
    from nodegraph.provider import ArrayProvider

    md = {"pixel_size_um": 1.0, "z_step_um": 2.0,
          "stage_xy_um": [(4.0, 4.0), (12.0, 4.0)], "stage_z_um": [10.0, 10.0],
          "z_home_index": 0, "z_bottom_to_top": True,
          "origin_um": [[10.0, 0.0, 0.0], [10.0, 0.0, 8.0]]}
    ax = AxisSizes(m=2, t=1, z=4, c=1, y=8, x=8)
    env = MetaEnvelope(axes=ax, metadata=dict(md))
    pri = Dataset(axes=ax, metadata=md).with_image(
        ArrayProvider(np.zeros((2, 1, 4, 1, 8, 8), np.uint16)))
    sec_md = {"pixel_size_um": 4.0, "stage_xy_um": [(8.0, 4.0)],
              "origin_um": [[0.0, -12.0, -8.0]]}
    sec_ax = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    sec = Dataset(axes=sec_ax, metadata=sec_md).with_image(
        ArrayProvider(np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 1, 8, 8)))

    def overlay_after(op=None, params=None, modes=None):
        g = Graph()
        g.add(NodeInstance("A", "io.load")); g.add(NodeInstance("B", "io.load"))
        tail = "A"
        if op:
            g.add(NodeInstance("U", op, params=dict(params or {}),
                               modes=dict(modes or {})))
            g.connect("A", "U"); tail = "U"
        g.add(NodeInstance("O", "view.overlay"))
        g.connect(tail, "O"); g.connect("B", "O", dst_socket="secondary")
        out = Engine(g, computes=COMPUTES, seeds={"A": pri, "B": sec},
                     meta_seeds={"A": env}).pull("O")
        return out.metadata[OVERLAY_KEY][-1]

    # the three reported chains all PLACE, and place onto a real tile
    for label, op, prm, mds in (
            ("plain", None, {}, {}),
            ("z-project", "util.zproject", {}, {"method": "max"}),
            ("stitch", "util.stitch", {"out_y": 16, "out_x": 24}, {}),
            ("crop", "util.crop", {"y0": 1, "y1": 7, "x0": 1, "x1": 7}, {"dim": "2D"})):
        entry = overlay_after(op, prm, mds)
        tiles = dict((int(m), v) for m, v in entry["tiles"])
        assert tiles and any(v for v in tiles.values()), (label, tiles)
        assert dict(entry["coverage"])[0] > 0.0, label
        if op is not None:
            # placed, and the divergence is REPORTED rather than swallowed
            assert any("different geometry" in w for w in entry["warnings"]), label

    # a Z projection costs Z and nothing else: X/Y still map, the depth relation is retired
    zp = overlay_after("util.zproject", {}, {"method": "max"})
    assert dict(zp["z_offset_um"])[0] is None
    assert any("Z is collapsed" in w for w in zp["warnings"]), zp["warnings"]
    assert dict(zp["coverage"])[0] > 0.0

    # ── the guard that must NOT have been weakened ──────────────────────────────
    one = AxisSizes(m=1, t=1, z=1, c=1, y=8, x=8)
    have = {"pixel_size_um": 1.0, "stage_xy_um": [(4.0, 4.0)],
            "origin_um": [[0.0, 0.0, 0.0]]}
    lost = {"pixel_size_um": 1.0, "stage_xy_um": [(4.0, 4.0)]}     # drift dropped it
    assert plan_placement(have, one, have, one,
                          dst_sampling=("crop[…]",), src_sampling=()).ok
    stale = plan_placement(have, one, lost, one,
                           dst_sampling=(), src_sampling=("align.drift",))
    assert not stale.ok and "align.drift" in stale.refusals[0], stale.refusals
    # a source with no origin but a CLEAN history is still fine — the log is current
    assert plan_placement(have, one, lost, one).ok

    _ok("overlay after a geometry change (V2.19, reported 2026-08-03): an overlay placed "
        "after stitch / z-project / crop now WORKS — it places by maintained µm origins, "
        "which is what origin_um is for, instead of refusing whenever the two chains "
        "differed (a check borrowed from the voxel-for-voxel `raw` socket that defeated the "
        "feature it was meant to protect); the divergence is reported as a warning on every "
        "such pull; a Z projection costs Z and NOT X/Y — lateral coverage is unchanged while "
        "the depth relation is retired rather than reported against a plane the data is not "
        "on; and the real guard is untouched: an input with no maintained origin whose "
        "geometry moved (drift) still REFUSES, while one with a clean history still passes")


def test_align_to() -> None:
    """``registration.align_to`` must recover a wrong stage log and refuse a bad match.

    The stage is good to ~10 µm, which at 0.29 µm/px is ~35 px of visible misregistration —
    enough to make an overlay useless and small enough that nothing warns you. This node
    measures the residual against a trusted reference and RECORDS it; the correction reaches
    every placement consumer through :func:`nodegraph.placement.field_box`.

    Both halves are checked: that a planted error comes back exactly, and that the node
    declines rather than inventing a shift when there is nothing to match — the second is
    the one that matters, because a confident wrong alignment looks identical to a right
    one until you measure something with it."""
    if not _HAVE_SKIMAGE:
        _ok("align_to: skipped (no skimage)")
        return
    from scipy.ndimage import gaussian_filter
    from nodegraph.placement import ALIGN_KEY, field_box
    from nodegraph.nodes import COMPUTES, SAMPLING_KEY
    from nodegraph.provider import ArrayProvider

    rng = np.random.default_rng(0)
    tex = gaussian_filter(rng.random((256, 256)).astype(np.float32), 2.0)
    r_ax = AxisSizes(m=1, t=1, z=1, c=1, y=256, x=256)
    m_ax = AxisSizes(m=1, t=1, z=1, c=1, y=128, x=128)
    ref_ds = Dataset(axes=r_ax, metadata={"pixel_size_um": 1.0,
                                          "origin_um": [[0.0, 0.0, 0.0]]}
                     ).with_image(ArrayProvider(tex[None, None, None, None]))
    TRUE_Y, TRUE_X = 40.0, 30.0
    window = tex[40:168, 30:158][None, None, None, None]

    def run(err_y, err_x, moving=None, **params):
        md = {"pixel_size_um": 1.0,
              "origin_um": [[0.0, TRUE_Y + err_y, TRUE_X + err_x]]}
        ds = Dataset(axes=m_ax, metadata=md).with_image(
            ArrayProvider(window if moving is None else moving))
        g = Graph()
        g.add(NodeInstance("M", "io.load")); g.add(NodeInstance("R", "io.load"))
        g.add(NodeInstance("A", "registration.align_to", params=dict(params)))
        g.connect("M", "A"); g.connect("R", "A", dst_socket="reference")
        out = Engine(g, computes=COMPUTES, seeds={"M": ds, "R": ref_ds}).pull("A")
        return out, out.metadata[ALIGN_KEY][0], out.metadata["align_to_ncc"][0]

    # an ALREADY-correct log measures zero and is left alone
    out, (dy, dx), ncc = run(0.0, 0.0)
    assert abs(dy) < 0.3 and abs(dx) < 0.3, (dy, dx)
    assert ncc > 0.99, ncc
    # a planted error is recovered, and the corrected box lands on the TRUE position
    out, (dy, dx), ncc = run(6.0, -4.0)
    assert abs(dy + 6.0) < 0.3 and abs(dx - 4.0) < 0.3, (dy, dx)
    assert ncc > 0.9, ncc
    fb = field_box(out.metadata, m_ax, 0)
    assert abs(fb.y0 - TRUE_Y) < 0.3 and abs(fb.x0 - TRUE_X) < 0.3, fb
    # pixels, axes and sampling provenance are UNTOUCHED — this is a metadata node, and an
    # overlay downstream must keep working, which it would not if this looked like a resample
    assert out.axes == m_ax
    assert not out.metadata.get(SAMPLING_KEY)
    assert "origin_um" in out.metadata and out.metadata["origin_um"][0][1] == TRUE_Y + 6.0

    # REFUSALS / declines
    noise = rng.random((1, 1, 1, 1, 128, 128)).astype(np.float32)
    _o, (dy, dx), ncc = run(6.0, -4.0, moving=noise)
    assert (dy, dx) == (0.0, 0.0) and ncc == 0.0, (dy, dx, ncc)   # nothing to match
    _o, (dy, dx), ncc = run(6.0, -4.0, max_shift_um=1.0)
    assert (dy, dx) == (0.0, 0.0), (dy, dx)          # beyond the trust window → stage stands
    _o, (dy, dx), ncc = run(6.0, -4.0, min_ncc=0.999)
    assert (dy, dx) == (0.0, 0.0), (dy, dx)          # not confident enough → stage stands
    g = Graph()
    g.add(NodeInstance("M", "io.load")); g.add(NodeInstance("A", "registration.align_to"))
    g.connect("M", "A")
    try:
        Engine(g, computes=COMPUTES,
               seeds={"M": Dataset(axes=m_ax).with_image(ArrayProvider(window))}).pull("A")
        raise AssertionError("align_to with no reference must refuse")
    except ValueError as exc:
        assert "reference" in str(exc)

    _ok("align_to (V2.19): a planted 6/-4 µm stage-log error is recovered to 0.3 µm and the "
        "corrected field box lands on the true position, while an already-correct log "
        "measures zero at ncc 1.0; pixels, axes and sampling provenance are untouched (it "
        "records the transform, per V2.03 §2, rather than resampling — so an overlay "
        "downstream still places) and origin_um is left exactly as it arrived; and it "
        "DECLINES in all four ways that matter — pure noise, a shift beyond the trust "
        "window, a correlation under the floor, and no reference wired — each leaving the "
        "stage log standing rather than inventing a confident wrong alignment")


def test_origin_um_maintenance() -> None:
    """``origin_um`` must keep describing WHERE the data is, through every transform.

    The key exists because ``stage_xy_um`` does not survive: it records where the camera
    was, so a crop leaves it pointing at a corner the data no longer has, and nothing in
    the pixels reveals it (the Viewer's hover readout already refuses to offer a stage
    coordinate after a crop for exactly this reason). Four rules, and three of them are
    ways to get it wrong:

    * a **crop** moves the corner by the cut, in µm, mirroring ``span``'s clamping;
    * a **resample** must NOT touch it — the field occupies the same patch of stage at
      either sampling, and scaling it here would double-count the change the transform
      already made to ``pixel_size_um`` (the V2.03 §2 A2 trap);
    * a **stitch** collapses M→1 onto the union corner;
    * **drift / stabilize** DROP it, because they move content under a fixed index grid,
      and a per-M corner cannot describe a per-(m,t) shift.
    """
    from nodegraph.metadata import (
        crop as m_crop, read_origin_um, resample as m_resample, stitch as m_stitch,
        z_project as m_zproj)
    from nodegraph.nodes import COMPUTES, SAMPLING_KEY
    from nodegraph.placement import plan_placement
    from nodegraph.provider import ArrayProvider

    ax = AxisSizes(m=2, t=1, z=4, c=1, y=100, x=80)
    env = MetaEnvelope(axes=ax, metadata={
        "pixel_size_um": 0.5, "z_step_um": 2.0,
        "origin_um": [[10.0, 200.0, 300.0], [10.0, 200.0, 340.0]]})

    # crop: the corner moves by the cut, in microns
    cropped = m_crop(env, {"y0": 10, "y1": 60, "x0": 20, "x1": 70}, {"dim": "2D"})
    assert read_origin_um(cropped)[0] == [10.0, 205.0, 310.0], cropped.metadata
    assert read_origin_um(cropped)[1] == [10.0, 205.0, 350.0]
    # ...clamped exactly like span(): a negative / absent start is 0, so nothing moves
    assert read_origin_um(m_crop(env, {"y1": 50}, {"dim": "2D"}))[0] == [10.0, 200.0, 300.0]
    assert read_origin_um(m_crop(env, {"y0": -5}, {"dim": "2D"}))[0] == [10.0, 200.0, 300.0]
    # 3D crop moves z by the z STEP, and a 2D crop leaves z alone even if z0 is set
    assert read_origin_um(m_crop(env, {"z0": 2}, {"dim": "3D"}))[0][0] == 14.0
    assert read_origin_um(m_crop(env, {"z0": 2}, {"dim": "2D"}))[0][0] == 10.0

    # resample: pixel size changes, the corner does NOT (the double-count trap)
    res = m_resample(env, {"scale_xy": 2.0}, {"dim": "2D"})
    assert abs(res.metadata["pixel_size_um"] - 0.25) < 1e-12
    assert read_origin_um(res) == read_origin_um(env), read_origin_um(res)
    # z-project and stack collapse an axis; the field is still where it was
    assert read_origin_um(m_zproj(env, {}, {"method": "max"})) == read_origin_um(env)

    # stitch: M→1 at the union corner
    st = m_stitch(env, {"out_y": 200, "out_x": 160}, {})
    assert st.axes.m == 1 and read_origin_um(st) == [[10.0, 200.0, 300.0]], st.metadata

    # a short or malformed list is dropped WHOLE rather than mis-indexed
    def _with_origin(value):
        return MetaEnvelope(axes=ax, metadata={**env.metadata, "origin_um": value})

    assert read_origin_um(_with_origin([[0.0, 0.0, 0.0]])) is None      # covers 1 of 2 m
    assert read_origin_um(_with_origin([[0, 0], [0, 0]])) is None       # not triples
    assert read_origin_um(_with_origin("nonsense")) is None

    # ── the §8 gate: the PAYLOAD must agree with the header the transform predicted ──
    prov = ArrayProvider(np.arange(2 * 1 * 4 * 1 * 100 * 80, dtype=np.uint16)
                         .reshape(2, 1, 4, 1, 100, 80))
    seed = Dataset(axes=ax, metadata=dict(env.metadata)).with_image(prov)
    g = Graph()
    g.add(NodeInstance("S", "io.load"))
    g.add(NodeInstance("C", "util.crop",
                       params={"y0": 10, "y1": 60, "x0": 20, "x1": 70},
                       modes={"dim": "2D"}))
    g.connect("S", "C")
    eng = Engine(g, computes=COMPUTES, seeds={"S": seed}, meta_seeds={"S": env})
    out = eng.pull("C")
    assert out.metadata["origin_um"] == eng.env("C").metadata["origin_um"], (
        out.metadata["origin_um"], eng.env("C").metadata["origin_um"])
    assert out.metadata["origin_um"][0] == [10.0, 205.0, 310.0]
    # and the read is FENCED: a node that reads origin_um records it for the memo
    assert "origin_um" in dict(eng.entry("C").reads)

    # ── drift DROPS it, and the stale stage log must not stand in ──────────────────
    g2 = Graph()
    g2.add(NodeInstance("S", "io.load")); g2.add(NodeInstance("D", "align.drift"))
    g2.connect("S", "D")
    staged = seed.with_metadata(stage_xy_um=[(300.0, 200.0), (340.0, 200.0)])
    drifted = Engine(g2, computes=COMPUTES, seeds={"S": staged},
                     meta_seeds={"S": env}).pull("D")
    assert drifted.metadata.get("origin_um") is None                 # claim withdrawn
    assert drifted.metadata.get("stage_xy_um") is not None           # but the log rides on
    assert tuple(drifted.metadata.get(SAMPLING_KEY, ())) == ("align.drift",)
    # ...so placement must REFUSE rather than silently fall back to that log
    bad = plan_placement(drifted.metadata, drifted.axes, drifted.metadata, drifted.axes,
                         dst_sampling=("align.drift",), src_sampling=("align.drift",))
    assert not bad.ok and any("no longer describes these pixels" in r
                              for r in bad.refusals), bad.refusals

    _ok("origin_um (V2.19): the spatial companion to pixel_size — a crop moves the corner "
        "by the cut (clamped exactly like span, z only in 3D), a resample changes the "
        "sampling and NOT the corner (the double-count trap), z-project/stack preserve it, "
        "stitch collapses M→1 onto the union corner, and a short or malformed list is "
        "dropped whole rather than mis-indexed; a real crop pull's payload equals the "
        "header the meta_transform predicted and the read is memo-fenced; drift WITHDRAWS "
        "the claim (a per-M corner cannot describe a per-frame shift) and placement then "
        "refuses instead of falling back to the stage log that still rides the payload")



def test_shape_synth() -> None:
    """The synthetic fixture, checked BEFORE anything is graded against it (V2.22).

    Runs first among the shape tests on purpose. The 585 hand labels count granules and are
    structurally blind to where a boundary lies, so ground truth is the only instrument that
    can say *correct* rather than *better than the last thing* — and a fixture nobody
    validated is worse than no fixture, because every absolute number downstream inherits its
    errors.

    The two planted contacts are the load-bearing part: a THIN neck a split move must take
    apart, and a BROAD contact it must leave alone. A split validated only on necks would
    bisect every large granule and a count metric would barely notice.
    """
    try:
        from nodegraph.synth import bed_to_dataset, make_bed
    except Exception as exc:                                     # pragma: no cover
        _ok(f"shape synth: SKIPPED ({type(exc).__name__}: {exc})")
        return
    bed = make_bed(dims=2, seed=3)
    own, vox = bed.owner, bed.voxel_um
    px = vox[-1]

    gids = sorted(b.gid for b in bed.bodies)
    assert gids == list(range(1, len(gids) + 1)), "body ids must be contiguous 1..n"
    assert set(np.unique(own).tolist()) - {0} == set(gids), \
        "every body must own painted voxels"
    fg = float((own > 0).mean())
    assert 0.18 < fg < 0.70, f"solid fraction {fg:.3f} is not bed-like"

    planted = [c for c in bed.contacts if c[3] in ("neck", "broad")]
    bulk = [c for c in bed.contacts if c[3] == "bulk"]
    assert len(planted) == 2, f"expected 2 planted contacts, got {len(planted)}"
    assert len(bulk) >= 15, (
        f"only {len(bulk)} bulk contacts — a fixture for BOUNDARY placement needs many "
        f"boundaries, not two")

    w_min = 0.15 * bed.metadata["r_eq_um"]
    nk = bed.metadata["neck_half_width_um"]
    bd = bed.metadata["broad_half_width_um"]
    assert nk < w_min, f"planted neck {nk:.2f} um must be inside the forbidden regime {w_min:.2f}"
    assert bd > 3.0 * w_min, f"planted broad contact {bd:.2f} um must be far outside it"

    # both pairs must MERGE under a plain threshold, or the fixture does not reproduce the
    # under-segmentation the split move exists to undo
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    img = bed.image[0, 0, 0, 0]
    lab_thr, _n = ndi.label(ndi.binary_fill_holes(img > threshold_otsu(img)))
    for roles in (("neck_a", "neck_b"), ("broad_a", "broad_b")):
        ga, gb = bed.role_gids(roles[0])[0], bed.role_gids(roles[1])[0]
        ids = set(np.unique(lab_thr[np.isin(own, [ga, gb])]).tolist()) - {0}
        assert len(ids) == 1, (
            f"the {roles[0].split('_')[0]} pair must read as ONE blob after thresholding "
            f"(the halo bridges the seam) — got {len(ids)}")

    # the concave negative control: as non-convex as a necked pair, with no thin place to cut
    from skimage.measure import regionprops
    gc = bed.role_gids("concave")[0]
    lc, nc = ndi.label(own == gc)
    assert nc == 1, "the concave body must be one piece"
    sol_c = float(max(p.solidity for p in regionprops(lc.astype(np.int32))))
    assert sol_c < 0.90, f"the concave body has no real dent (solidity {sol_c:.3f})"

    b2 = make_bed(dims=2, seed=3)
    assert np.array_equal(bed.owner, b2.owner) and np.array_equal(bed.image, b2.image), \
        "the fixture must be bit-identical for a fixed seed"
    b3 = make_bed(dims=3, seed=3)
    assert b3.owner.ndim == 3 and b3.metadata.get("z_step_um") == 40.0, "3D must build"
    _ds, env = bed_to_dataset(bed)
    assert env.axes.y == own.shape[0] and env.axes.c == 1, "bed_to_dataset axes"
    _ok(f"shape synth: {bed.metadata['n_bodies']} convex bodies, solid {fg:.3f}, "
        f"{len(bulk)} bulk contacts plus a planted THIN neck ({nk:.2f} um = {nk / px:.2f} px, "
        f"inside w_min {w_min / px:.2f} px) and BROAD contact ({bd:.2f} um), both merged by a "
        f"real threshold; a concave single body at solidity {sol_c:.3f} is the split move's "
        f"negative control; bit-identical per seed and 3D builds")


def test_fit_shape() -> None:
    """``analysis.fit_shape`` — a convex body per object, stored as half-spaces (V2.22).

    Structural spec first, so the contract is checked even where a dependency is missing, per
    the house pattern. Then the numbers: a signed distance is only worth having if it is exact
    on a body whose answer is known, and ``fill`` / ``rms_residual_um`` are only worth having
    if they separate a convex body from a concave one.
    """
    spec = NODES.get("analysis.fit_shape")
    assert spec is not None, "analysis.fit_shape is not registered"
    assert spec.category == "analysis"
    assert spec.reads_domains == frozenset({Domain.VOXEL, Domain.LABEL})
    assert spec.adds_domains == frozenset({Domain.LABEL})
    assert spec.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert spec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert spec.resolve_kernel_axes({"dim": "2D"}) == frozenset({"x", "y"})
    modes = {m.name: list(m.choices) for m in spec.modes}
    assert modes.get("model") == ["convex_hull", "obb"], (
        "only POLYTOPE models may be offered — a sphere or an ellipsoid is not one and would "
        "need a different field evaluator, so declaring it would be a dead control")
    assert "dim" in modes
    if not _HAVE_WATERSHED:
        _ok("fit_shape: spec OK; RUN SKIPPED (needs scipy + scikit-image)")
        return
    try:
        from nodegraph.synth import make_bed
    except Exception as exc:                                     # pragma: no cover
        _ok(f"fit_shape: spec OK; RUN SKIPPED ({type(exc).__name__}: {exc})")
        return
    from scipy import ndimage as ndi
    from nodegraph.catalog._shared.shapes import fit_object_polytope
    from nodegraph.kernels.convex_polytope import signed_distance

    bed = make_bed(dims=2, seed=3)
    vox = bed.voxel_um

    def _fit(gid, model="convex_hull"):
        m = bed.owner == gid
        sl = ndi.find_objects(m.astype(int))[0]
        org = [float(s.start) * v for s, v in zip(sl, vox)]
        return fit_object_polytope(m[sl], voxel_um=vox, model=model, origin_um=org)

    # A body CONTAINS itself, so fill <= 1 is a hard geometric fact — and it only holds because
    # the hull is fitted to voxel CORNERS. Fitting centres loses half a voxel all round and put
    # the median fill at 1.016, which is impossible, and fatal for a split proposal that
    # triggers on fill dropping below a threshold.
    fills, rmss = [], []
    for role in ("bulk", "neck_a", "broad_a", "lone"):
        poly, ex = _fit(bed.role_gids(role)[0])
        assert poly is not None, f"{role} must fit"
        assert ex["fill"] <= 1.0 + 1e-9, \
            f"{role}: fill {ex['fill']:.4f} > 1 — the body is not inside its own hull"
        fills.append(ex["fill"])
        rmss.append(ex["rms_residual_um"])
    poly_c, ex_c = _fit(bed.role_gids("concave")[0])
    assert ex_c["fill"] < min(fills) - 0.05, (
        f"the concave body's fill {ex_c['fill']:.3f} must sit clearly below every convex "
        f"body's (min {min(fills):.3f}) — `fill` is the concavity signal")
    assert ex_c["rms_residual_um"] > 2.0 * max(rmss), (
        f"the concave body's fit residual {ex_c['rms_residual_um']:.2f} um must dominate the "
        f"convex ones (max {max(rmss):.2f}) — this residual IS sigma_rough")

    # |grad s| = 1 on the fitted body: the property that makes an energy imbalance of dE um
    # displace a boundary by exactly dE um, and therefore the only reason a term weight can be
    # calibrated against a one-voxel budget instead of tuned.
    poly, ex = _fit(bed.role_gids("lone")[0])
    rng = np.random.default_rng(0)
    q = poly.centroid_um[None, :] + rng.uniform(-4.0, 4.0, size=(200, 2))
    h = 1e-6
    gmag = np.linalg.norm(np.stack(
        [(signed_distance(q + h * e, poly) - signed_distance(q - h * e, poly)) / (2 * h)
         for e in np.eye(2)], axis=1), axis=1)
    assert np.allclose(gmag, 1.0, atol=1e-5), \
        f"|grad s| must be 1 (max deviation {np.abs(gmag - 1).max():.2e})"

    poly_o, ex_o = _fit(bed.role_gids("lone")[0], model="obb")
    assert poly_o is not None and poly_o.n_faces == 4, \
        f"a 2D oriented bounding box has 4 facets, got {poly_o.n_faces}"
    assert ex_o["fill"] <= ex["fill"] + 1e-9, \
        "an oriented bounding box cannot fit tighter than the convex hull"
    _ok(f"fit_shape: spec OK; convex bodies fill {min(fills):.3f}-{max(fills):.3f} with residual "
        f"<= {max(rmss):.2f} um (about one voxel, i.e. sampling-limited rather than surface "
        f"roughness), while the concave control reads fill {ex_c['fill']:.3f} and residual "
        f"{ex_c['rms_residual_um']:.2f} um — both statistics separate it; |grad s| = 1 to "
        f"{np.abs(gmag - 1).max():.1e}; obb gives 4 facets and cannot fit tighter")



def test_background_probability() -> None:
    """``analysis.background_probability`` — p(background) as a PROBABILITY (V2.22 space 1).

    Graded on calibration, not only on discrimination. A threshold node is scored by how
    often it is right; this one is scored by whether 0.3 means 0.3, because the whole reason
    it exists is that something downstream will MULTIPLY it with other evidence, and a
    mis-calibrated factor corrupts the product silently. The synthetic bed is the instrument:
    it carries exact per-voxel ownership, so background is known rather than estimated.

    The parametric form here replaced a derived one that discriminated well (AUC 0.907) and
    was badly mis-calibrated (ECE 0.435), which is why calibration is asserted at all.
    """
    spec = NODES.get("analysis.background_probability")
    assert spec is not None, "analysis.background_probability is not registered"
    assert spec.category == "analysis"
    assert spec.adds_domains == frozenset({Domain.VOXEL})
    assert spec.resolve_granularity({"dim": "2D"}) is Granularity.WHOLE_PLANE
    assert spec.resolve_granularity({"dim": "3D"}) is Granularity.WHOLE_VOLUME
    assert spec.resolve_kernel_axes({"dim": "3D"}) == frozenset({"z", "y", "x"})

    from nodegraph.kernels.background_probability import (
        DEFAULT_CEILING, background_probability, expected_calibration_error, fit_logistic,
        normalise)

    # monotone, bounded, and it never claims more than the ceiling
    u = np.linspace(-0.5, 2.0, 400)
    p = background_probability(u, normalised=True)
    assert np.all(np.diff(p) <= 1e-7), "p(background) must fall as intensity rises"
    assert p.max() <= DEFAULT_CEILING + 1e-6 and p.min() >= 0.0
    assert p[0] > 0.8 * DEFAULT_CEILING and p[-1] < 0.01, "must span its range"

    # the fitter recovers parameters it generated, so a new instrument can be calibrated
    rng = np.random.default_rng(0)
    ub = rng.normal(0.10, 0.05, 6000)
    um = rng.normal(0.70, 0.15, 6000)
    got = fit_logistic(ub, um)
    assert 0.15 < got["midpoint"] < 0.65, f"fit_logistic midpoint {got['midpoint']}"
    assert got["width"] > 0.0

    if not _HAVE_SKIMAGE:
        _ok("background probability: spec + kernel OK; RUN SKIPPED (scipy/skimage absent)")
        return
    try:
        from nodegraph.synth import bed_to_dataset, make_bed
    except Exception as exc:                                     # pragma: no cover
        _ok(f"background probability: spec + kernel OK; RUN SKIPPED ({type(exc).__name__})")
        return
    from nodegraph.nodes import COMPUTES

    bed = make_bed(dims=2, seed=3)
    ds0, env = bed_to_dataset(bed)
    define_node("io.pbgseed", "Seed", outputs=[OutDataset()])

    def _pull(dim, **params):
        g = Graph()
        g.add(NodeInstance("S", "io.pbgseed"))
        g.add(NodeInstance("P", "analysis.background_probability",
                           modes={"dim": dim}, params=params))
        g.connect("S", "P")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds0}, meta_seeds={"S": env})
        return e, e.pull("P")

    eng2, out2 = _pull("2D")
    lay = out2.get(Domain.VOXEL, "p_background")
    assert lay is not None, "no p_background layer written"
    pv = np.asarray(lay.values)
    assert pv.shape == (env.axes.m, env.axes.t, env.axes.z, env.axes.c,
                        env.axes.y, env.axes.x), f"wrong shape {pv.shape}"
    assert np.all(np.isfinite(pv)) and pv.min() >= 0.0 and pv.max() <= 1.0

    # the fixture's own answer: owner == 0 is background
    truth = (bed.owner == 0)
    plane = pv[0, 0, 0, 0]
    # exclude a 2 px band either side of every boundary — within a voxel of an edge the
    # honest answer is intermediate, and scoring it as either class would punish the model
    # for being right (the same exclusion the calibration on real data used).
    from scipy import ndimage as _ndi
    se = np.ones((5, 5))
    band = _ndi.binary_dilation(truth, se) & ~_ndi.binary_erosion(truth, se)
    bg, fg = plane[truth & ~band], plane[(~truth) & ~band]
    assert bg.size > 500 and fg.size > 500, "fixture gave too few clean voxels"
    assert bg.mean() > fg.mean() + 0.25, (
        f"background {bg.mean():.3f} must read far above material {fg.mean():.3f}")
    s = np.r_[bg, fg]
    y = np.r_[np.ones(bg.size), np.zeros(fg.size)]
    ece, _rows = expected_calibration_error(s, y)
    assert ece < 0.30, f"expected calibration error {ece:.3f} is too large to multiply with"

    # 2D and 3D are genuinely different computations, not merely different footprints — and
    # the case that proves it is the one the lever exists for: an ATTENUATING stack. Plane 1
    # holds the same object at a quarter the brightness, as a real confocal stack does with
    # depth. Per-plane anchors renormalise it and still find the object; per-volume anchors
    # are set by the bright plane, so the dim one washes out toward background. The synthetic
    # bed cannot show this, because it EXTRUDES identical planes and the two agree exactly
    # there — which is what the first version of this assertion tripped over.
    from nodegraph.provider import ArrayProvider
    att = np.full((1, 1, 2, 1, 24, 24), 40.0)          # a camera offset in both planes
    att[0, 0, 0, 0, 6:18, 6:18] = 1000.0
    att[0, 0, 1, 0, 6:18, 6:18] = 250.0
    ax_a = AxisSizes(m=1, t=1, z=2, c=1, y=24, x=24)
    ds_a = Dataset(axes=ax_a).with_image(ArrayProvider(att))
    env_a = MetaEnvelope(axes=ax_a, metadata={})

    def _pull3(dim):
        g = Graph()
        g.add(NodeInstance("S", "io.pbgseed"))
        g.add(NodeInstance("P", "analysis.background_probability", modes={"dim": dim}))
        g.connect("S", "P")
        e = Engine(g, computes=COMPUTES, seeds={"S": ds_a}, meta_seeds={"S": env_a})
        return e, e.pull("P")

    e2, o2 = _pull3("2D")
    e3, o3 = _pull3("3D")
    assert e2.entry("P").recipe_hash != e3.entry("P").recipe_hash, \
        "2D and 3D must produce distinct recipe hashes"
    a2 = np.asarray(o2.get(Domain.VOXEL, "p_background").values)[0, 0, :, 0]
    a3 = np.asarray(o3.get(Domain.VOXEL, "p_background").values)[0, 0, :, 0]
    obj = (slice(8, 16), slice(8, 16))
    assert a2[0][obj].mean() < 0.15 and a2[1][obj].mean() < 0.15, (
        f"per-plane anchors must find the object in BOTH planes (got "
        f"{a2[0][obj].mean():.3f}, {a2[1][obj].mean():.3f})")
    assert a3[1][obj].mean() > a2[1][obj].mean() + 0.2, (
        f"per-volume anchors must wash the dim plane's object toward background "
        f"(volume {a3[1][obj].mean():.3f} vs plane {a2[1][obj].mean():.3f})")

    # the refusals are real, not decorative
    for bad, why in (({"width": 0.0}, "width"), ({"ceiling": 1.5}, "ceiling"),
                     ({"lo_pct": 99.0, "hi_pct": 5.0}, "anchors")):
        try:
            _pull("2D", **bad)
        except Exception:
            pass
        else:                                                    # pragma: no cover
            raise AssertionError(f"background probability accepted a bad {why}")
    _ok(f"background probability: spec OK; on the fixture's own ownership map it reads "
        f"{bg.mean():.3f} in background against {fg.mean():.3f} in material with "
        f"ECE {ece:.3f}; monotone and bounded by the ceiling; 2D per-plane and 3D "
        f"per-volume anchors give different numbers AND different recipe hashes; "
        f"width/ceiling/anchor refusals all fire")


def test_transform() -> None:
    """``transform.rigid`` must move exactly the named channels, and by exactly the
    distance and DIRECTION asked for.

    Every assertion here uses an ASYMMETRIC fixture — a single off-centre pixel, or a
    half-filled frame — because that is the only kind that can catch the failure this node
    is most exposed to: a sign flip or a transposed axis. A blob at the centre of a
    symmetric test image comes back looking perfect whether the matrix is right or its own
    inverse, and a channel-registration node that moves the picture the wrong way makes
    colocalization *worse* while reporting success.

    The per-channel contract is the other half: channel 0 must come back **bit-identical**
    while channel 1 moves, since the whole reason this is one node rather than a
    split/transform/merge chain is that both arrive downstream on one wire, addressable
    voxel-for-voxel."""
    if not _HAVE_SKIMAGE:
        _ok("transform: skipped (no scipy/skimage)")
        return
    from nodegraph.nodes import COMPUTES, SAMPLING_KEY
    from nodegraph.provider import ArrayProvider
    from nodegraph.structure import point_table

    PX, ZS = 0.5, 2.0             # µm/px lateral, µm/plane axial — deliberately ≠ 1

    def run(arr, ax, params, modes, *, extra=None, meta=None):
        md = {"pixel_size_um": PX, "z_step_um": ZS, "origin_um": [[7.0, 11.0, 13.0]]}
        md.update(meta or {})
        seed = Dataset(axes=ax, metadata=dict(md)).with_image(ArrayProvider(arr))
        if extra is not None:
            seed = extra(seed)
        g = Graph()
        g.add(NodeInstance("S", "io.load"))
        g.add(NodeInstance("T", "transform.rigid", params=dict(params),
                           modes=dict(modes)))
        g.connect("S", "T")
        eng = Engine(g, computes=COMPUTES, seeds={"S": seed},
                     meta_seeds={"S": MetaEnvelope(axes=ax, metadata=dict(md))})
        return eng, eng.pull("T")

    def plane(ds, z=0, c=0):
        ax = ds.axes
        return np.asarray(ds.image.get_region(0, 0, 0, z, c, 0, ax.y, 0, ax.x))

    def peak(a):
        return tuple(int(v) for v in np.unravel_index(int(np.argmax(a)), a.shape))

    # ── (1) per-channel translation: direction, magnitude, and an UNTOUCHED neighbour ──
    ax2 = AxisSizes(m=1, t=1, z=1, c=2, y=33, x=33)
    dots = np.zeros((1, 1, 1, 2, 33, 33), dtype=np.float32)
    dots[0, 0, 0, 0, 10, 20] = 100.0
    dots[0, 0, 0, 1, 10, 20] = 100.0
    # 2.5 µm / 0.5 µm-per-px = +5 px in x, 1.5 µm = +3 px in y
    eng, out = run(dots, ax2, {"channels": "1", "shift_x": 2.5, "shift_y": 1.5},
                   {"dim": "2D", "interp": "nearest"})
    assert peak(plane(out, c=1)) == (13, 25), peak(plane(out, c=1))   # +y is DOWN, +x RIGHT
    assert np.array_equal(plane(out, c=0), dots[0, 0, 0, 0]), "channel 0 was disturbed"
    assert float(plane(out, c=1).sum()) == 100.0, "the moved dot was duplicated or lost"
    # geometry and placement are untouched — only the content moved
    assert out.axes == ax2 and out.metadata["pixel_size_um"] == PX
    assert out.metadata["origin_um"] == [[7.0, 11.0, 13.0]], out.metadata["origin_um"]
    # ...but the move IS declared, so `analysis.measure`'s raw guard can see it
    assert len(out.metadata.get(SAMPLING_KEY, ())) == 1, out.metadata.get(SAMPLING_KEY)
    assert "pixel_size_um" in dict(eng.entry("T").reads)      # memo-fenced conversion

    # ── (2) rotation sign: +angle is CLOCKWISE as displayed (ImageJ's convention) ──────
    #     a dot to the RIGHT of centre must land BELOW it, not above.
    ax1_33 = AxisSizes(m=1, t=1, z=1, c=1, y=33, x=33)
    right = np.zeros((1, 1, 1, 1, 33, 33), dtype=np.float32)
    right[0, 0, 0, 0, 16, 26] = 100.0                        # centre is (16, 16)
    _e, out = run(right, ax1_33, {"angle": 90.0}, {"dim": "2D", "interp": "nearest"})
    assert peak(plane(out)) == (26, 16), peak(plane(out))
    _e, out = run(right, ax1_33, {"angle": -90.0}, {"dim": "2D", "interp": "nearest"})
    assert peak(plane(out)) == (6, 16), peak(plane(out))      # and −90 goes the other way

    # ── (3) the crop: what leaves the frame is GONE and what it vacates is ZERO ───────
    ax1 = AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16)
    full = np.full((1, 1, 1, 1, 16, 16), 100.0, dtype=np.float32)
    _e, out = run(full, ax1, {"shift_x": 2.5}, {"dim": "2D", "interp": "linear"})
    p = plane(out)
    assert float(p[:, :5].max()) == 0.0, "the vacated strip is not empty"
    assert np.allclose(p[:, 5:], 100.0), "the surviving band was altered"
    assert p.shape == (16, 16), "the frame grew — a transform must not resize"
    # nothing wrapped around: 5 columns of 16 rows were dropped, not rotated in
    assert abs(float(p.sum()) - 100.0 * 16 * 11) < 1e-6, float(p.sum())

    # ── (4) 3D: the axial shift is in µm_axial (z_step), and 2D must IGNORE it ────────
    ax3 = AxisSizes(m=1, t=1, z=5, c=1, y=16, x=16)
    vol = np.zeros((1, 1, 5, 1, 16, 16), dtype=np.float32)
    vol[0, 0, 1, 0, 8, 8] = 100.0
    _e, out = run(vol, ax3, {"shift_z": 4.0}, {"dim": "3D", "interp": "nearest"})
    tops = [float(plane(out, z=z).max()) for z in range(5)]
    assert tops == [0.0, 0.0, 0.0, 100.0, 0.0], tops       # 4 µm / 2 µm-per-plane = +2
    _e, out = run(vol, ax3, {"shift_z": 4.0}, {"dim": "2D", "interp": "nearest"})
    assert out.metadata.get(SAMPLING_KEY) is None, \
        "shift_z is 3D-only, so in 2D this is an identity and must claim nothing"
    spec = NODES.get("transform.rigid")
    vis = lambda st: {s.name for s in spec.active_inputs(st)}      # noqa: E731
    assert "shift_z" not in vis({"dim": "2D", "interp": "linear"})
    assert "shift_z" in vis({"dim": "3D", "interp": "linear"})

    # ── (5) a Voxel layer rides along, and an integer raster keeps its ids ────────────
    def add_mask(ds):
        mk = np.zeros((1, 1, 1, 2, 33, 33), dtype=np.int32)
        mk[0, 0, 0, 0, 10, 20] = 7
        mk[0, 0, 0, 1, 10, 20] = 7
        return ds.with_layer(Domain.VOXEL, "mask", mk)

    _e, out = run(dots, ax2, {"channels": "1", "shift_x": 2.5}, {"dim": "2D"},
                  extra=add_mask)
    mk = np.asarray(out.get(Domain.VOXEL, "mask").values)
    assert mk.dtype == np.int32, mk.dtype
    assert set(np.unique(mk)) == {0, 7}, "interpolating ids invented a label"
    assert mk[0, 0, 0, 1, 10, 25] == 7 and mk[0, 0, 0, 1, 10, 20] == 0   # moved with c=1
    assert mk[0, 0, 0, 0, 10, 20] == 7                                   # c=0 stayed put

    # ── (6) refusals ─────────────────────────────────────────────────────────────────
    def add_points(ds):
        return ds.with_structure(point_table(np.array([[10.0, 20.0]]),
                                             z_kind="plane_index", layer="spots"))

    try:
        run(dots, ax2, {"shift_x": 2.5}, {"dim": "2D"}, extra=add_points)
        raise AssertionError("a structure table on the wire must be refused")
    except ValueError as exc:
        assert "spots" in str(exc) and "BEFORE" in str(exc), str(exc)
    try:
        run(dots, ax2, {"channels": "9", "shift_x": 2.5}, {"dim": "2D"})
        raise AssertionError("a channel list naming nothing must be refused")
    except ValueError as exc:
        assert "names no channel" in str(exc), str(exc)

    # ── (7) identity is a true pass-through — no stamp, no refusal, no resample ──────
    _e, out = run(dots, ax2, {}, {"dim": "2D"}, extra=add_points)
    assert out.metadata.get(SAMPLING_KEY) is None
    assert np.array_equal(plane(out, c=1), dots[0, 0, 0, 1])

    # ── (8) the lever and the interpolation Mode each re-key the memo ────────────────
    hashes = set()
    for md in ({"dim": "2D", "interp": "linear"}, {"dim": "3D", "interp": "linear"},
               {"dim": "2D", "interp": "nearest"}, {"dim": "2D", "interp": "cubic"}):
        e, _o = run(vol, ax3, {"shift_x": 2.5}, md)
        hashes.add(e.entry("T").recipe_hash)
    assert len(hashes) == 4, "the lever or the interp Mode does not fold into the recipe"

    _ok("transform (transform.rigid): moves ONLY the named channels — a dot in c1 lands at "
        "+3y/+5px for 1.5/2.5 µm at 0.5 µm/px while c0 comes back bit-identical, so both "
        "arrive on one wire addressable voxel-for-voxel; +angle turns CLOCKWISE as "
        "displayed and −angle the other way (checked with an off-centre dot, the only "
        "fixture a sign flip cannot survive); content pushed past the edge is dropped with "
        "no wraparound and the vacated strip is exactly zero at an unchanged frame size; "
        "shift_z resolves against z_step in 3D and is gated away AND inert in 2D; a Voxel "
        "mask rides along per-channel at order 0 so its ids survive; a structure table or "
        "an empty channel list is refused; an identity leaves the payload and its sampling "
        "provenance untouched while a real move stamps it, keeps axes/pixel size/origin_um, "
        "fences its pixel_size_um read, and re-keys on both the lever and the interp Mode")


def main() -> int:
    test_domains()
    test_reducers()
    test_partial_reducers()
    test_revision_and_immutability()
    test_dataset()
    test_plans()
    test_execution()
    test_sockets()
    test_registry()
    test_live_reload_contract()
    test_catalog_import_hygiene()
    test_socket_dims()
    test_calibration()
    test_node_variants()
    test_metadata_pass()
    test_domain_interface()
    test_provider()
    test_memo()
    test_memo_gc()
    test_engine()
    test_engine_observer()
    test_engine_granularity()
    test_solo_frame_scope()
    test_frame_subset_scope()
    test_structure()
    test_bridges()
    test_field()
    test_tracks()
    test_boundary()
    test_nodes()
    test_catalog()
    test_catalog_ported()
    test_catalog_ported2()
    test_channel_split()
    test_two_channel_branches()
    test_reroute()
    test_catalog3()
    test_stitch()
    test_zproject_over_stitch()
    test_fusion_reducers()
    test_transfer_bridge_exec()
    test_zones()
    test_sim_perframe()
    test_iterate()
    test_groups()
    test_tracking()
    test_track_objects()
    test_serialize()
    test_engine_hardening()
    test_review_regressions()
    test_streaming()
    test_streaming_slivers()
    test_parallel()
    test_gpu_platform()
    test_catalog_kernels()
    test_catalog_dvc()
    test_catalog_dic()
    test_channel_derive()
    test_cluster_points()
    test_mesh_domain()
    test_tessellate_split()
    test_label_to_points_voronoi()
    test_segment()
    test_celltracker_parity()
    test_socket_docs()
    test_option_docs()
    test_param_socket_contract()
    test_layer_catalog()
    test_picking()
    test_nd2_audit_repairs()
    test_nd2_ingest_calibration()
    test_nd2_zstack_home_index_guard()
    test_nd3_calibration_mapping()
    test_nd3_ingest_roundtrip()
    test_group_reduce_repairs()
    test_transfer_structure()
    test_checkpoint_dock()
    test_overlay_placement()
    test_overlay_renderers()
    test_overlay_resample()
    test_overlay_override_and_presentation()
    test_overlay_after_geometry_change()
    test_align_to()
    test_origin_um_maintenance()
    test_transform()
    test_shape_synth()
    test_fit_shape()
    test_background_probability()
    print("\nALL NODEGRAPH SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
