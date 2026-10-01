"""columns — the edit-time STRUCTURE-COLUMN catalog helpers (V2.28).

``NodeSpec.adds_columns`` lets a producer say which columns it writes, so a downstream
``column_in`` socket can OFFER them instead of making the user type ``mean_intensity``
and find out at pull time that nothing measured it.

The picker built on it is a CLOSED dropdown, so declaring is **mandatory** for any node
that adds a structure domain: an undeclared producer makes its columns unpickable rather
than merely unsuggested. ``selftest::test_column_catalog_complete`` fails the build
otherwise. This module holds the invariant
schemas every structure producer emits and the two combinators the declarations use.

**Every function here must be total.** They run inside ``propagate_meta``, on every
keystroke, whose caller catches only ``ValueError`` — see ``_column_names_out``.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from nodegraph.domains import Domain

#: The invariant Label schema every region producer emits (``structure._label_table``).
#: ``area`` is in VOXELS (px^2 in 2D, px^3 in 3D) — the micron areas are Object Metrics'.
LABEL_INVARIANT: Tuple[str, ...] = ("id", "m", "t", "c", "area", "z", "y", "x")

#: The invariant Point schema every detector emits (``structure.point_table``).
POINT_INVARIANT: Tuple[str, ...] = ("id", "m", "t", "c", "z", "y", "x")

#: The three membership arrays every Track table carries (``TrackMembership.to_table``).
#: ``track.objects`` adds more; ``track.link`` emits exactly these.
TRACK_MEMBERSHIP: Tuple[str, ...] = ("track_id", "t", "member_id")


def on_layer(domain: Domain, layer: str,
             names: Iterable[str]) -> Tuple[Tuple[Domain, str, str], ...]:
    """``(domain, layer, col)`` for each name — the declaration building block.

    An empty or non-string ``layer`` yields NOTHING rather than a triple keyed on ``""``:
    a socket whose layer is inferred (\u00a74g) legitimately has no name to predict at edit
    time, and a blank key would offer every one of its columns under a layer that does not
    exist, which reads as a broken picker."""
    if not isinstance(layer, str) or not layer:
        return ()
    return tuple((domain, layer, n) for n in names if isinstance(n, str) and n)


def carried_over(incoming: Sequence[Tuple[Domain, str, str]], domain: Domain,
                 src_layer: str, dst_layer: str,
                 *, extra: Iterable[str] = ()) -> Tuple[Tuple[Domain, str, str], ...]:
    """The columns of ``src_layer`` re-offered under ``dst_layer``, plus ``extra``.

    For a node that re-emits a table under a NEW name — ``analysis.filter_labels`` keeps
    every column of the labels it cut and adds ``cut``. Declaring only the invented column
    would amputate the catalog at exactly the node a user filters with, so the carried ones
    have to be restated under the new key; the catalog is keyed by ``(domain, layer)`` and
    nothing else walks the provenance back to the source layer."""
    if not isinstance(dst_layer, str) or not dst_layer:
        return ()
    carried = [c for d, lyr, c in incoming if d is domain and lyr == src_layer]
    return on_layer(domain, dst_layer, list(carried) + [e for e in extra])


def member_layer(params: Mapping, modes: Mapping, *,
                 label_socket: str = "labels", point_socket: str = "points",
                 mode_name: str = "target") -> Tuple[Domain, str]:
    """``(domain, layer)`` of the MEMBER table a label-or-point node is working on.

    The ``target`` Mode picks the domain and the matching layer socket names the instance —
    the shape ``analysis.measure`` / ``analysis.object_metrics`` / ``track.*`` all share."""
    if (modes or {}).get(mode_name) == "point":
        return Domain.POINT, str((params or {}).get(point_socket) or "")
    return Domain.LABEL, str((params or {}).get(label_socket) or "")


def resolved_layer(params: Mapping, socket: str,
                   incoming: Sequence[Tuple[Domain, str, str]], domain: Domain,
                   fallback: str = "") -> str:
    """The layer a possibly-EMPTY layer socket denotes, by the catalog's own only-candidate
    rule (§4g).

    Several nodes ship their structure socket blank and resolve "the ONE instance of this
    domain on the wire" at pull time (``analysis.measure``'s ``points``,
    ``transform.label_to_points``'s ``labels``). At edit time there is no Dataset to look
    at, but the accumulated catalog names the same instances — so when the socket is blank
    and exactly ONE layer of the domain is known, that is the one the compute will pick.

    Deliberately silent when there are two or more: the compute REFUSES an ambiguous wire,
    so offering the union would suggest a graph that does not run."""
    explicit = str((params or {}).get(socket) or "").strip()
    if explicit:
        return explicit
    known = list(dict.fromkeys(lyr for d, lyr, _c in incoming if d is domain))
    if len(known) == 1:
        return known[0]
    return fallback


#: The columns of a MESH element table (``nodegraph.mesh.build_mesh_tables``).
MESH_ELEMENT: Tuple[str, ...] = ("id", "m", "t", "c", "z", "y", "x", "element_uid",
                                 "src_label", "vert_start", "vert_count", "face_start",
                                 "face_count")


def dvc_field_columns(*, is_3d: bool) -> Tuple[str, ...]:
    """The Point columns ``_shared.dvc._dvc_rows`` emits, which every correlation node
    (``analysis.piv`` / ``dvc_field`` / ``dic_correlate`` / ``accumulate_field``) builds its
    table from.

    Derived from the same rule the flattener uses rather than transcribed, so the two cannot
    drift: the axis names are ``(z,y,x)`` in 3D and ``(y,x)`` in 2D, one ``disp_<axis>`` per
    component, and one ``strain_<i><j>`` per ordered axis PAIR — 4 of them in 2D, 9 in 3D.

    ``strain`` and ``qfactor`` are only present when the solver produced them, and the
    per-node declarations that call this say so; over-declaring here would offer a column a
    strain-free run does not carry. That is the one direction this catalog must not err in,
    so each caller gates them on its own params."""
    axes = ("z", "y", "x") if is_3d else ("y", "x")
    cols = ["id", "m", "t", "c", "z", "y", "x"]
    cols += [f"disp_{a}" for a in axes]
    cols += ["disp_mag_um"]
    cols += [f"strain_{i}{j}" for i in axes for j in axes]
    return tuple(dict.fromkeys(cols))
