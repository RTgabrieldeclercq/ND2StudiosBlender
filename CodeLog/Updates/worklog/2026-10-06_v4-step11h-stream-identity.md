# V4.00 step 11h — a stream carries its whole identity: position, channel, and the name it was given; a Page Output names itself from its first wire

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11g

## What changed

- **Fix: each item of a several-item Output is its own channel on the next page.** Masks
  made per channel on Refinement (a Split, a Threshold per channel, their `mask only` parts
  into ONE Page Output) came through to Processing all calling themselves by the FIRST
  item's channel — on the cards' `out` (their input rows were right). The inherited-name walk
  (`_inherited_channel_descriptors`) crossed the page boundary by asking the Output for its
  channels, which are its first item's; it now crosses by the ITEM's wire
  (`page_channel_scope` with the item). `upstream_channel_descriptors` (the channel picker)
  reads the socket the wire leaves (`socket_channels`) for the same reason. `socket_channels`
  now answers for any socket — a plot's picture R/G/B included, the picker needs them — and
  the "not a channel of the file's" exclusion for a `fresh_output` node moved to
  `channel_subset`, the tag.
- **A position labels a stream as a channel does.** Past `1 · B03` on a Split Positions card
  the cards read `B03`, and `B03 · Cy5` with a channel tap too, before and after a page
  boundary: `GraphDocument.socket_positions` / `position_subset` / `position_tag` /
  `source_position_total` mirror the channel trio, reading the stream's position names off
  its envelope (`_env_position_names`: the file's `position_name`, narrowed with M by every
  position tap, else `m{i}`); the file's position total crosses a page through
  `Workspace.input_position_scope` (hook `doc.page_position_scope`), the twin of the channel
  scope. No dot tint: positions have no colour.
- **The name a stream was given is its name on the next page.** `GraphDocument.stream_name`
  is the variable (`control`) — or the item (`mask_cy5`) — of the Page Output a stream last
  came through, read off the Page Input it enters by and followed back along the PRIMARY
  Dataset wire through any node after it. `stream_identity` composes what a socket prints:
  the name, then the position(s) and channel(s) the name does not already say —
  `control · B03 · Cy5`, `mask_cy5 · B03`, just `DAPI` for an Output named `DAPI` on the DAPI
  channel. `socket_text` prints it (a generic `data`/`out` is replaced, a role socket adds it,
  a synthetic socket keeps its label), and the hover gained `position:` and `named:` lines
  beside `channel:` (`NodeItem._stream_facts`, `SocketItem.set_domain_tip(facts=)`).
- **A Page Output names itself from its first wire** while its name is still the
  placeholder (`out`, `out2`) or blank (`GraphDocument._name_output_from_wire`, called by
  `connect` for the first slot): the position (`B03`), the channel (`Cy5`), a part (`mask`),
  a node's title (`gaussian_blur`; `gaussian_blur_b03_cy5` on a one-position, one-channel
  stream) — the same answer an item gets (`_item_default`), settled unique like any name
  (`B032`). A typed name is never replaced. A Page Input already bound to the placeholder
  (a new page auto-binds to the newest Output, wired or not) follows the rename through the
  new `doc.output_renamed` hook → `Workspace._follow_renames`. A `LinkedDocument` sets
  `AUTO_NAMES_OUTPUTS = False` (its names are its master's; in `EDIT_MASTER` the wire lands
  on the master, which names as usual). The readiness hint's *+ Page Output* therefore lands
  named after the node it was appended to.
- **Items from every tap are named by what they carry** (`_tap_name`): a position's name
  (`B03`, was `split_positions_pos1`), a group's key, a batch member's name, as a channel's
  already was; a node's item appends the one position and one channel its title does not say.
- **Tests**: `test_stream_identity` (new), the several-item case added to
  `test_channel_provenance`; probe CP2 (the next page's Input and the card wired to it read
  `mask`, the hover names the Output, a dropped Output takes its first wire's name
  `median`), and the readiness section now expects its appended Output to be named
  `gaussian_blur`. **Docs**: MANUAL §4 rewritten as *A stream knows where it came from*
  (positions, the name, auto-name), §2 Pages (auto-name), two §18 rows; codemap CON-23
  rewritten (stream identity) with new anchors.

## Why

The user, testing 11g: "on the image input node graph, i want the output node to be able to
name the data. for example i split the nd2 file into location data m split, then i make an
output node per m file. i want to name each m file a different name" — and: "when i get to
the image processing node, the data doesnt match the stream. I made masks for each channel,
then passed the masks to the image processing node, and all of the data streams called the
data GFP".

The second was a bug in 11g's inheritance across a page boundary (above). The first is what
11g left out: identity stopped at the channel. A position of a multipoint file is exactly as
much "where the data came from" as a channel is, and the name the user gives an Output is
the identity THEY chose — so after the boundary the stream should carry it, and an Output
fed a position should not need a name typed to be told apart. Asked how, the user chose: the
given name plus the facts it does not already say (over name-only, or channel-only with the
name on the Input's title); auto-naming from the first wire while the name is a placeholder
(over typing every name); and positions labelling streams like channels (over reaching later
pages only through the Output's name).

Why this shape: one resolution (`stream_identity`) feeds the row text, the hover, the item
default and the Output's default name, so they cannot disagree; the position trio copies the
channel trio rather than generalising both into one abstraction, because the two differ in
what a socket reads (a descriptor list with colours vs. a name list) and a shared abstraction
would have been the harder read for the next person; the auto-name lives in `connect` — the
one place a first wire arrives from every editor — and only ever replaces a placeholder, so
no name the user typed can be lost; and the rename follows readers because the standard
workflow auto-binds a new page to the newest Output before it is wired.

## Files

- `nodelab_v2/document.py` — `_join_names`, `_PLACEHOLDER_NAME_RE`; hooks
  `page_position_scope`, `output_renamed`; `connect` → `_name_output_from_wire`,
  `AUTO_NAMES_OUTPUTS`; `_inherited_channel_descriptors` crosses by the item;
  `upstream_channel_descriptors` / `socket_channels` / `channel_subset` (the fresh-output
  rule moved); `_env_position_names`, `_env_m`, `source_position_total`, `socket_positions`,
  `position_subset`, `position_tag`, `stream_name`, `stream_identity`, `_page_position_scope`;
  `socket_text` prints the identity; `_item_default` + `_tap_name`
- `nodelab_v2/workspace.py` — `input_position_scope`; the two hooks in `_attach` / `_detach`
- `nodelab_v2/linked_document.py` — `AUTO_NAMES_OUTPUTS = False`
- `nodelab_v2/node_item.py` — `_stream_facts` (was `_channel_phrase`),
  `set_domain_tip(facts=)`
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/curated.lock.json`, `codemap/gen/*`,
  `codemap/STATE.md`

## How to verify

`python run.py`, load a multipoint, multi-channel ND2. Drop a Split Positions after the Load
card and a Page Output after each of its `K · name` sockets: the Outputs read `Output · A01`,
`Output · B03`, … as the wires land; type `control` into one — it stays. On Image Refinement
each Page Input bound to one of them has `out` reading `B03` (or `control · B03`), and a
Gaussian after it the same; add a `2 · Cy5` channel tap and the cards read `control · B03 ·
Cy5`. Split the channels, threshold each, wire the three `mask only` parts into one Page
Output: on Image Processing the Input's item sockets each carry their own channel and a node
wired to each reads it on both its sockets. Hover any socket for `position:` / `channel:` /
`named:`.

## Gates

- [x] selftest — 189 tests run (new: `test_stream_identity`); only `test_write_movie` fails
  (pre-existing H.264 rounding). `test_channel_provenance`'s cross-page expectations moved
  with the behaviour (`raw · Cy5`, the item's name on an item)
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 153 checks (new: CP2; RF1's appended
  Output now `gaussian_blur`), run alone, exit 0
- [x] catalog snapshot — CATALOG IDENTICAL, 104 ops (no op changed); synopsis CURRENT
- [x] codemap — CODEMAP CURRENT (CON-23 rewritten and blessed with its new anchors)
- [ ] sync check — run before the push
