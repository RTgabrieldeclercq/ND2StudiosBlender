"""The catalog's registration surface: ``COMPUTES`` + :func:`register_node`.

This module exists to break a cycle. Every per-node module in :mod:`nodegraph.catalog`
needs ``register_node``, and ``nodegraph.nodes`` — where it used to live — must import the
catalog in order to populate it. Putting the two names in a leaf module that imports nothing
from either side leaves the graph acyclic:

    nodegraph.catalog.<node>  ->  nodegraph.catalog._base
    nodegraph.nodes (facade)  ->  nodegraph.catalog  ->  nodegraph.catalog.<node>

``COMPUTES`` keeps its identity for the lifetime of the process even across a live reload:
:mod:`nodegraph.hotreload` reconciles into this dict object rather than rebinding it, because
``nodelab_v2.ops``, ``nodegraph/__init__`` and every constructed
:class:`~nodegraph.engine.Engine` hold a reference to it. That is also why this module is
**not** itself reloadable — re-executing it would mint a fresh dict and orphan every one of
those references. It holds no node logic, so there is nothing in it worth reloading.

Qt-free.
"""
from __future__ import annotations

from typing import Any, Dict

from nodegraph.engine import Compute
from nodegraph.registry import define_node, registration_helper

#: op_key → compute(ctx) — the engine's compute lookup for the whole catalog.
COMPUTES: Dict[str, Compute] = {}


def register_node(compute: Compute, **spec_kwargs: Any):
    """Register a node: build+register its :class:`NodeSpec` and record its compute."""
    spec = define_node(**spec_kwargs)
    COMPUTES[spec.op_key] = compute
    return spec


# This function registers nodes on BEHALF of the module that calls it, so registration
# provenance (which the live reloader uses to decide whose nodes to drop) must look through
# this frame to the caller. Declared here rather than listed inside the registry so that
# moving this helper again cannot silently re-attribute the entire catalog.
registration_helper(__name__, register_node.__name__)


__all__ = ["COMPUTES", "register_node"]
