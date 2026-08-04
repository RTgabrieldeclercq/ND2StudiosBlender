"""Helpers shared by more than one catalog node, grouped by CONCERN — one module per concern.

**This file must stay empty of imports, and that is a correctness constraint rather than a
style preference.** Each node's memo key carries a fingerprint of its module's transitive
import closure (:func:`nodegraph.hotreload.dependency_closure`), so a node must reach a helper
through its *concern module* and nothing wider:

    from nodegraph.catalog._shared.units import to_pixels_v2      # correct
    from nodegraph.catalog._shared import to_pixels_v2            # collapses the design

If this initialiser re-exported its submodules, the second form would work and would put all
sixteen concern modules into the closure of every node that used any one helper. Measured on
this catalog, that takes the over-invalidation from 76 node-keys per helper edit back to ~1497
— i.e. every helper edit re-keys almost the whole catalog again, which is exactly the
monolithic behaviour the split was done to remove. The nodes would still work; the memo would
just quietly throw everything away on every edit. ``test_catalog_import_hygiene`` enforces both
halves (rule 3 rejects the package-level import; a positive check rejects imports here).

The grouping itself follows one rule: **a helper's home is the smallest module whose entire
user set wants to be re-keyed together.** ``units`` (18 users) is a leaf with no first-party
imports at all, so its reach is harmless; ``map_image`` (14 users, the tiling/halo/GPU policy)
is the one genuinely wide module, and it is kept out of ``nodegraph/streaming.py`` because
living there would widen it to every node instead.

Qt-free.
"""
