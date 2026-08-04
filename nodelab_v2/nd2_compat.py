"""``nd2`` SDK compatibility shims — reader bugs that block a real lab file outright.

Every ``import nd2`` in this package goes through :func:`import_nd2` rather than importing
the module directly. It applies the shims below **once** per process and hands the module
back. It is a function, not module-level patching, because ``nd2`` is imported *lazily*
everywhere on purpose — :mod:`nodelab_v2.ingest` must stay importable in an environment
without the SDK (the TIFF half of it works fine there), and a top-level patch would break
that contract.

Shims are written as **delegating wrappers that only intervene on the exact failure**: the
upstream function runs unchanged first, and the shim takes over only once it has actually
thrown. A file that loads today therefore loads bit-for-bit identically — the shim cannot
regress it, which is the property that makes patching somebody else's parser acceptable at
all. Each shim is idempotent (tagged with :data:`_SHIM_TAG`) so a re-import cannot stack
wrappers.

Shim 1 — **a zero-range Z-stack loop divides by zero**
------------------------------------------------------
Found 2026-08-03 on ``LAGFP_Caps_Z_12hrs.nd2`` (6.08 GB, T=145 × P=5 × 2048², 12-bit),
which could not be opened at all. ``nd2._parse._parse._calc_zstack_home_index`` ends in::

    if step_um <= 0:
        return min(int((count - 1) * hrange / abs(high_um - low_um)), count - 1)

and that file's ``ZStackLoop`` is one a user really can produce — a Z-stack configured in ND
Acquisition but acquired at a **single plane**::

    uiCount = 1   dZLow = -25.2   dZHigh = -25.2   dZStep = 0.0
    dZHome = 1.8  iType = 7       bZInverted = False

``dZLow == dZHigh``, so the Z range is exactly zero. ``_parse_z_stack_loop`` only synthesizes
a step when ``count > 1``, so ``step_um`` stays ``0.0`` and takes that branch; ``iType = 7``
selects ``hrange = abs(dZHigh - dZHome) = 27.0``; the denominator is ``0.0`` →
``ZeroDivisionError``. It fires from ``ND2File.experiment``, hence from ``ND2File.sizes`` —
the first thing :func:`nodelab_v2.ingest.read_meta_only` touches — so the File menu failed at
pick time, before a pixel was read, on a file that is otherwise perfectly healthy.

The correct answer is **0**, and not as a fallback: the numerator ``(count - 1) * hrange`` is
itself ``0`` when ``count == 1``, so 0 is the value the formula was reaching for. For the
degenerate ``count > 1`` case (also zero-range — ``_parse_z_stack_loop`` would have set
``step`` to ``abs(hi - low) / (count - 1) == 0``) every slice sits at the same Z, so slice 0
is the only defensible home. ``homeIndex`` is inert for this particular file anyway:
:func:`nodelab_v2.ingest._origin_um_from_stage` takes its ``n_z <= 1`` branch and uses the
nominal focus directly.

Affects ``nd2 <= 0.11.3`` — the latest release on PyPI as of 2026-08-03, so there is no
fixed version to pin to instead. Reported upstream; drop this shim once a release carries
the guard (:func:`import_nd2` will then simply never see the exception).

A note on what is NOT shimmed: that file also reports ``acquisition_start`` of
``7/31/2609`` with a matching ``absoluteJulianDayNumber`` of 2887693.53. That is what the
microscope PC wrote into the file, not a parse error, so it is left alone — ``frame_time_jd``
is honest about what it read. The consequence is real but belongs to the data: this file
cannot be placed on the cross-file clock ``view.overlay`` uses. Relative ``dt_s`` (299.99 s)
is unaffected, so rates and tracking are fine.
"""
from __future__ import annotations

import functools
from typing import Any

#: Attribute stamped on a wrapper so :func:`import_nd2` can tell an already-shimmed SDK from
#: a fresh one. Guards against double-wrapping if the module is reloaded.
_SHIM_TAG = "_nd2studios_shim"

#: Set once the shims have been applied in this process, so the repeated lazy
#: ``import_nd2()`` calls on the ingest hot path cost one bool test.
_applied = False


def _patch_zstack_home_index() -> bool:
    """Wrap ``nd2._parse._parse._calc_zstack_home_index`` so a zero Z range yields home
    index 0 instead of ``ZeroDivisionError`` (shim 1 in the module docstring).

    Returns whether a wrapper was installed. ``False`` covers both "already shimmed" and
    "this ``nd2`` has no such private function" — a future release that renames or fixes it.
    Neither is an error: the shim is a repair for a specific version range, and a version
    outside it needs nothing.

    ``_parse_z_stack_loop`` calls the function through a module-global lookup, so replacing
    the module attribute is enough to reach it.
    """
    try:
        from nd2._parse import _parse
    except Exception:            # noqa: BLE001 — private path; absence is not an error
        return False
    orig = getattr(_parse, "_calc_zstack_home_index", None)
    if orig is None or getattr(orig, _SHIM_TAG, False):
        return False

    @functools.wraps(orig)
    def guarded(*args: Any, **kwargs: Any) -> int:
        try:
            return orig(*args, **kwargs)
        except ZeroDivisionError:
            # abs(high_um - low_um) == 0: a Z-stack whose range collapsed to a single
            # plane. The numerator is 0 too, so 0 is the formula's own answer.
            return 0

    setattr(guarded, _SHIM_TAG, True)
    _parse._calc_zstack_home_index = guarded
    return True


def apply_shims() -> None:
    """Apply every shim in this module, once per process. Never raises — a shim that cannot
    find its target simply does not install (see :func:`_patch_zstack_home_index`), because
    failing to patch must not be worse than not having tried."""
    global _applied
    if _applied:
        return
    _applied = True
    _patch_zstack_home_index()


def import_nd2() -> Any:
    """Import the ``nd2`` SDK with this module's shims applied, and return it.

    Use this instead of a bare ``import nd2`` anywhere in the package, at the same lazy
    point the bare import sat — it must run before any ``ND2File`` is constructed, since the
    shimmed parse happens on first metadata access. Propagates ``ImportError`` unchanged when
    the SDK is genuinely absent, which is what the ND2 paths already expect."""
    import nd2

    apply_shims()
    return nd2


__all__ = ["import_nd2", "apply_shims"]
