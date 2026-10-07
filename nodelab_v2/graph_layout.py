"""Reorganize a page's node graph (2026-10-07): a layered left-to-right layout that keeps every REGION in one block.

What the canvas's action pill runs as *Reorganize graph*. Qt-free and pure: it reads plain
data — node ids, wires, card sizes, frames, current positions — and returns new card
positions, so the selftest drives it headless and the window applies it with
:meth:`~nodelab_v2.document.GraphDocument.set_pos`.

The layout
----------
Data flows left to right. A card's COLUMN is the length of the longest chain of wires in
front of it; a source with no input sits one column before its first consumer (a value card
or an Iterate beside the node it drives, not at the far left). Within a column the cards
keep the order they had top to bottom, then a few barycentre sweeps pull each card toward
the cards it is wired to, and a chain of single wires comes out straight. Cards never
overlap: every card keeps :data:`ROW_GAP` from the one under it and :data:`COL_GAP` from the
next column.

Regions stay whole
------------------
Every frame — a region — is laid out as ONE BLOCK: its members are arranged inside it with
the same rules, and the block is placed in the page layout like a single card, with room
for the frame's title bar, its padding and the labels of its ports
(:data:`FRAME_PAD`, :data:`FRAME_TITLE`, :data:`FRAME_SIDE`). So after a reorganize no card
lies inside a region it does not belong to. Two frames that share a card are one block. A
region whose wires leave it and come back into it through outside cards cannot be one
column span; it is laid out card by card instead and named in :attr:`Layout.loose`.

A FROZEN region keeps the arrangement inside it and only moves as a whole; an ANCHORED
region does not move at all, so everything else is placed around it. A region tab uses both
(:func:`layout_document`): the region's cards are its master's, so only the tab's own Page
Inputs and Outputs move.

The page stays where it was: the result's top-left corner is the old graph's top-left.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

#: a card's size when the caller does not know it (the canvas card width, a typical height)
DEFAULT_SIZE: Tuple[float, float] = (214.0, 140.0)
#: the horizontal gap between two columns
COL_GAP = 110.0
#: the vertical gap between two cards (or blocks) stacked in one column
ROW_GAP = 36.0
#: a frame's padding round its members (:attr:`nodelab_v2.frame_item.FrameItem.PAD`)
FRAME_PAD = 22.0
#: a frame's title bar, above the padding (:attr:`nodelab_v2.frame_item.FrameItem.TITLE_H`)
FRAME_TITLE = 24.0
#: room beside a frame for its port labels, which are drawn outside it
FRAME_SIDE = 64.0
#: barycentre sweeps (each one down the columns and back up)
SWEEPS = 4

Pos = Tuple[float, float]
Size = Tuple[float, float]


class Layout(NamedTuple):
    """What :func:`layout` computed: ``positions`` — every card's new top-left corner — and
    ``loose`` — the regions that could not be kept as one block (their wires leave and
    re-enter them through outside cards), laid out card by card."""
    positions: Dict[str, Pos]
    loose: Tuple[str, ...]


# ── one level: a layered placement of boxes ──────────────────────────────────
def _layers(ids: Sequence[str], succ: Mapping[str, Set[str]],
            pred: Mapping[str, Set[str]], rank: Mapping[str, float]) -> Dict[str, int]:
    """Each id's column: the longest path to it from a source (a cycle — which the document
    never builds — is broken at the lowest-ranked id), then every SOURCE moved right to sit
    one column before its nearest consumer."""
    indeg = {i: len(pred[i]) for i in ids}
    ready = sorted((i for i in ids if indeg[i] == 0), key=lambda i: rank[i])
    order: List[str] = []
    left = set(ids)
    while left:
        if not ready:                                   # a cycle: break it
            ready = [min(left, key=lambda i: rank[i])]
        i = ready.pop(0)
        if i not in left:
            continue
        left.discard(i)
        order.append(i)
        for j in sorted(succ[i], key=lambda j: rank[j]):
            if j in left:
                indeg[j] -= 1
                if indeg[j] <= 0:
                    ready.append(j)
        ready.sort(key=lambda j: rank[j])
    layer: Dict[str, int] = {}
    pos = {i: k for k, i in enumerate(order)}
    for i in order:
        before = [layer[p] for p in pred[i] if p in layer and pos[p] < pos[i]]
        layer[i] = 1 + max(before) if before else 0
    for i in reversed(order):
        if not pred[i] and succ[i]:
            layer[i] = max(0, min(layer[j] for j in succ[i]) - 1)
    return layer


def _order(cols: Dict[int, List[str]], succ, pred, rank) -> None:
    """Order each column in place: by the current top-to-bottom order, then barycentre
    sweeps toward the neighbouring columns (a card with no neighbour there keeps its
    place)."""
    for k in cols:
        cols[k].sort(key=lambda i: rank[i])
    keys = sorted(cols)

    def sweep(seq, nbrs):
        for k in seq:
            prev = {i: n for c in cols.values() for n, i in enumerate(c)}
            col = cols[k]

            def key(item):
                n, i = item
                ps = [prev[j] for j in nbrs[i] if j in prev and j not in col]
                return (sum(ps) / len(ps)) if ps else float(n)
            cols[k] = [i for _n, i in sorted(enumerate(col), key=lambda t: (key(t), t[0]))]

    for _ in range(SWEEPS):
        sweep(keys[1:], pred)
        sweep(list(reversed(keys[:-1])), succ)


def _resolve(order: List[str], want: Dict[str, float], h: Mapping[str, float]) -> Dict[str, float]:
    """Centre heights for ``order`` (top to bottom) as close to ``want`` as they can be with
    :data:`ROW_GAP` between neighbours: pushed down where they would overlap, then the whole
    column shifted back by the mean displacement (which keeps every gap)."""
    out: Dict[str, float] = {}
    last: Optional[str] = None
    for i in order:
        y = want[i]
        if last is not None:
            y = max(y, out[last] + h[last] / 2 + ROW_GAP + h[i] / 2)
        out[i] = y
        last = i
    if order:
        shift = sum(want[i] - out[i] for i in order) / len(order)
        for i in order:
            out[i] += shift
    return out


def _place(ids: Sequence[str], sizes: Mapping[str, Size], pairs: Iterable[Tuple[str, str]],
           rank: Mapping[str, float]) -> Dict[str, Pos]:
    """A layered left-to-right placement of the boxes ``ids`` (top-left corners, the
    smallest at 0, 0). ``pairs`` are ``(src, dst)`` wires between them; ``rank`` their
    current top-to-bottom order."""
    if not ids:
        return {}
    succ: Dict[str, Set[str]] = {i: set() for i in ids}
    pred: Dict[str, Set[str]] = {i: set() for i in ids}
    for s, d in pairs:
        if s in succ and d in succ and s != d:
            succ[s].add(d)
            pred[d].add(s)
    layer = _layers(ids, succ, pred, rank)
    cols: Dict[int, List[str]] = {}
    for i in ids:
        cols.setdefault(layer[i], []).append(i)
    _order(cols, succ, pred, rank)
    keys = sorted(cols)
    w = {i: float(sizes[i][0]) for i in ids}
    h = {i: float(sizes[i][1]) for i in ids}
    xs: Dict[int, float] = {}
    x = 0.0
    for k in keys:
        xs[k] = x
        x += max(w[i] for i in cols[k]) + COL_GAP
    cy: Dict[str, float] = {}
    # a first stack of every column, centred on 0
    for k in keys:
        tot = sum(h[i] for i in cols[k]) + ROW_GAP * (len(cols[k]) - 1)
        y = -tot / 2
        for i in cols[k]:
            cy[i] = y + h[i] / 2
            y += h[i] + ROW_GAP
    # left to right: each card toward the cards feeding it
    for k in keys[1:]:
        want = {}
        for i in cols[k]:
            ps = [cy[p] for p in pred[i] if layer[p] < k]
            want[i] = sum(ps) / len(ps) if ps else cy[i]
        cy.update(_resolve(cols[k], want, h))
    # right to left: a SOURCE toward the cards it feeds (a Load level with its chain)
    for k in reversed(keys[:-1]):
        want = {}
        for i in cols[k]:
            ss = [cy[s] for s in succ[i] if layer[s] > k]
            want[i] = (sum(ss) / len(ss)) if (ss and not pred[i]) else cy[i]
        cy.update(_resolve(cols[k], want, h))
    top = min(cy[i] - h[i] / 2 for i in ids)
    return {i: (xs[layer[i]], cy[i] - h[i] / 2 - top) for i in ids}


# ── the page: regions as blocks ──────────────────────────────────────────────
def _block_groups(frames: Mapping[str, Sequence[str]], present: Set[str]) -> List[List[str]]:
    """Frames that share a card merged (union-find), each group's frame ids in frame
    order. A frame with no present member is left out."""
    fids = [f for f, mem in frames.items() if any(m in present for m in mem)]
    parent = {f: f for f in fids}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f
    owner: Dict[str, str] = {}
    for f in fids:
        for m in frames[f]:
            if m not in present:
                continue
            if m in owner:
                a, b = find(owner[m]), find(f)
                if a != b:
                    parent[b] = a
            else:
                owner[m] = f
    groups: Dict[str, List[str]] = {}
    for f in fids:
        groups.setdefault(find(f), []).append(f)
    return list(groups.values())


def _acyclic(blocks: Mapping[str, Sequence[str]], block_of: Mapping[str, str],
             pairs: Sequence[Tuple[str, str]]) -> Optional[str]:
    """``None`` when the block graph has no cycle, else one REGION block on a cycle."""
    succ: Dict[str, Set[str]] = {b: set() for b in blocks}
    for s, d in pairs:
        bs, bd = block_of[s], block_of[d]
        if bs != bd:
            succ[bs].add(bd)
    state: Dict[str, int] = {}
    stack_path: List[str] = []
    found: List[str] = []

    def visit(b: str) -> bool:
        state[b] = 1
        stack_path.append(b)
        for c in sorted(succ[b]):
            st = state.get(c, 0)
            if st == 1:
                cyc = stack_path[stack_path.index(c):]
                found.extend(x for x in cyc if x.startswith("F:"))
                return True
            if st == 0 and visit(c):
                return True
        stack_path.pop()
        state[b] = 2
        return False

    for b in sorted(blocks):
        if state.get(b, 0) == 0 and visit(b):
            return found[0] if found else None
    return None


def layout(nodes: Sequence[str], edges: Iterable[Tuple[str, str, str, str]], *,
           sizes: Optional[Mapping[str, Size]] = None,
           frames: Optional[Mapping[str, Sequence[str]]] = None,
           current: Optional[Mapping[str, Pos]] = None,
           frozen: Iterable[str] = (), anchor: Optional[str] = None) -> Layout:
    """New top-left positions for ``nodes`` (document order), wired by ``edges``
    ``(src, src_socket, dst, dst_socket)``, with every frame in ``frames`` (frame id →
    member ids) kept as one block. ``sizes`` are card sizes (default :data:`DEFAULT_SIZE`);
    ``current`` the positions now, which seed the top-to-bottom order and where the result
    lands. A region in ``frozen`` keeps its inside arrangement; the ``anchor`` region (one
    of ``frozen``) does not move."""
    nodes = list(dict.fromkeys(nodes))
    present = set(nodes)
    sizes = dict(sizes or {})
    cur = {n: (float(p[0]), float(p[1])) for n, p in (current or {}).items() if n in present}
    for k, n in enumerate(nodes):
        cur.setdefault(n, (0.0, float(k) * (DEFAULT_SIZE[1] + ROW_GAP)))
    size = {n: tuple(float(v) for v in sizes.get(n, DEFAULT_SIZE)) for n in nodes}
    pairs = list(dict.fromkeys((e[0], e[2]) for e in edges
                               if e[0] in present and e[2] in present and e[0] != e[2]))
    frames = {f: [m for m in mem if m in present] for f, mem in (frames or {}).items()}
    frozen = set(frozen)
    order_ix = {n: k for k, n in enumerate(nodes)}

    def node_rank(n: str) -> float:          # top to bottom, then left to right, then doc
        return cur[n][1] + cur[n][0] * 1e-4 + order_ix[n] * 1e-9

    groups = _block_groups(frames, present)
    loose: List[str] = []
    while True:
        blocks: Dict[str, List[str]] = {}
        block_of: Dict[str, str] = {}
        gfrozen: Dict[str, bool] = {}
        gfids: Dict[str, List[str]] = {}
        for g in groups:
            bid = "F:" + g[0]
            mem = list(dict.fromkeys(m for f in g for m in frames[f]))
            blocks[bid] = mem
            gfids[bid] = g
            gfrozen[bid] = any(f in frozen for f in g)
            for m in mem:
                block_of[m] = bid
        for n in nodes:
            if n not in block_of:
                bid = "N:" + n
                blocks[bid] = [n]
                block_of[n] = bid
        bad = _acyclic(blocks, block_of, pairs)
        if bad is None:
            break
        groups = [g for g in groups if "F:" + g[0] != bad]   # that region goes card by card
        loose.extend(gfids[bad])
    # inside each block
    inner: Dict[str, Dict[str, Pos]] = {}
    bsize: Dict[str, Size] = {}
    for bid, mem in blocks.items():
        if bid.startswith("N:"):
            inner[bid] = {mem[0]: (0.0, 0.0)}
            bsize[bid] = size[mem[0]]
            continue
        if gfrozen[bid]:
            x0 = min(cur[m][0] for m in mem)
            y0 = min(cur[m][1] for m in mem)
            rel = {m: (cur[m][0] - x0, cur[m][1] - y0) for m in mem}
        else:
            ms = set(mem)
            rel = _place(mem, size, [p for p in pairs if p[0] in ms and p[1] in ms],
                         {m: node_rank(m) for m in mem})
        iw = max(rel[m][0] + size[m][0] for m in mem)
        ih = max(rel[m][1] + size[m][1] for m in mem)
        ox, oy = FRAME_PAD + FRAME_SIDE, FRAME_PAD + FRAME_TITLE
        inner[bid] = {m: (rel[m][0] + ox, rel[m][1] + oy) for m in mem}
        bsize[bid] = (iw + 2 * (FRAME_PAD + FRAME_SIDE), ih + FRAME_PAD * 2 + FRAME_TITLE)

    def block_rank(bid: str) -> float:
        return min(node_rank(m) for m in blocks[bid])

    bpairs = list(dict.fromkeys((block_of[s], block_of[d]) for s, d in pairs
                                if block_of[s] != block_of[d]))
    outer = _place(list(blocks), bsize, bpairs, {b: block_rank(b) for b in blocks})
    pos: Dict[str, Pos] = {}
    for bid, (bx, by) in outer.items():
        for m, (ix, iy) in inner[bid].items():
            pos[m] = (bx + ix, by + iy)
    # where the result lands: the anchored region exactly where it is, else the old top-left
    abid = None
    if anchor is not None:
        abid = next((b for b, fs in gfids.items() if anchor in fs), None)
    if abid is not None:
        m0 = blocks[abid][0]
        dx, dy = cur[m0][0] - pos[m0][0], cur[m0][1] - pos[m0][1]
    else:
        dx = min(p[0] for p in cur.values()) - min(p[0] for p in pos.values())
        dy = min(p[1] for p in cur.values()) - min(p[1] for p in pos.values())
    return Layout({n: (round(pos[n][0] + dx, 1), round(pos[n][1] + dy, 1)) for n in nodes},
                  tuple(loose))


def layout_document(doc, sizes: Optional[Mapping[str, Size]] = None) -> Layout:
    """:func:`layout` for a :class:`~nodelab_v2.document.GraphDocument`: its nodes, its
    wires, its frames and its current positions. On a REGION TAB the region is frozen and
    anchored — its cards are the master's, so only the tab's own nodes move."""
    frames = {fid: list(fr.members) for fid, fr in doc.frames.items()}
    frozen: List[str] = []
    anchor = None
    region = getattr(doc, "region", None)
    if region is not None:
        members = [m for m in doc.region_members() if m in doc.nodes]
        if members:
            frames.setdefault(region, members)
            frames[region] = list(dict.fromkeys(list(frames[region]) + members))
            frozen, anchor = [region], region
    return layout(list(doc.nodes), doc.edges, sizes=sizes, frames=frames,
                  current={n: (r.x, r.y) for n, r in doc.nodes.items()},
                  frozen=frozen, anchor=anchor)


__all__ = ["Layout", "layout", "layout_document", "DEFAULT_SIZE", "COL_GAP", "ROW_GAP",
           "FRAME_PAD", "FRAME_TITLE", "FRAME_SIDE"]
