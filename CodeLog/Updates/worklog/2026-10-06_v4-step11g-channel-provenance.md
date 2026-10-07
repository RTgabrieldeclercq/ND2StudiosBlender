# V4.00 step 11g — a data stream names the channel it came from

- **Date:** 2026-10-06
- **Author:** hyper
- **Branch:** v4-step11-standard-workflow
- **Base:** b28bc05 (v4-step10-docs), on top of steps 11a–11f

## What changed

- **A stream names its channel on every socket.** Wire the Load card's `2 · Cy5` socket into
  a Gaussian and the Gaussian's sockets read **`Cy5`**, not `data` / `out` — and so does every
  card after it, through Threshold, Label, Measure, the Page Output and the next page's Page
  Input. On a generic socket (`data`, `out`, a source's `image`) the channel REPLACES the name;
  on a socket with a role of its own (`raw`, `reference`, `areas`) it is added — `raw · DAPI`
  (`GraphDocument.socket_text`, `GENERIC_DATASET_SOCKETS`). Two channels read `DAPI · GFP`,
  three or more `DAPI +2`. The full bundle is not a channel: a card fed everything keeps
  `data` / `out`, exactly as its wire is not tinted. A synthetic socket (`0 · DAPI`,
  `mask only`, `item · cells`) keeps its label — the channel is on the `out` above it. The
  socket DOT is tinted by the channel(s) too, inputs included (`NodeItem._tint_channel_socket`,
  `desc_qcolor`: native colour, else emission) — re-applied on every `refresh`, since a card
  is laid out at the drop and its wire arrives after (the GUI probe caught a plain dot beside
  a row that already read `DAPI`) — and the hover says it in words
  (`channel: Cy5 — 1 of 3 channels`, `SocketItem.set_domain_tip(channel=)`). The wire reads
  the same resolution (`EdgeItem._channel_colors` → `GraphDocument.channel_subset`), so the
  wire, its two ends and their names cannot disagree; a wire whose file carries a native
  channel colour now takes it, as the `chK` dot always did.
- **The names ride the data.** The root cause: the edit-time source envelope carried only
  `channel_emission_nm`, so past a `chK` wire a card could tint by emission but could only call
  its stream `Ch0` (the names lived on the Load card's captured `__channels__` and on the
  pulled payload; the inherited-name walk stopped at a `chK` edge). Now `channel_names` and
  `channel_colors` ride the source envelope — the Load card's seed at file-pick
  (`_add_source_node`, `_add_bundle_node`) and the runner's resolved envelope (`_ingest_locked`
  both branches, `_open_direct`, `_resolve_bundle`, the synthetic fallback; stamped on the
  cached `_providers` entry so the `_fresh_envs` re-seed agrees) — through ONE function,
  `ingest.channel_display_seed` / `with_channel_display`. Both keys were already in
  `metadata.PER_CHANNEL_KEYS`, so every `channel.select` narrows them with the axis.
  `_SYNTH_META` lists the synthetic names so a probe seeding from it matches the resolved
  envelope.
- **The channel total crosses the page boundary.** `source_channel_total` at a root
  `page.input` asks the Workspace for the FILE's total on the page it reads from
  (`Workspace.input_channel_scope`, hook `doc.page_channel_scope`, recursive through pages),
  so a one-channel stream on a later page knows it is one of three and names itself; before,
  the Input's own envelope was the root and "one of one" meant nothing to tint or name. The
  same hook answers for one ITEM of a several-item Output (the wire into that slot).
- **Older graphs get the names past a tap too.** `_inherited_channel_descriptors` narrows
  through a `chK` wire to channel K of the upstream list instead of stopping there, so a seed
  from before the names rode it (a graph opened from an older session, until its first pull)
  still yields the file's name and native colour downstream of a tap. A `channel.select` by
  param on such a seed falls back to `Ch0 · Ch1` until the first pull re-seeds.
- **A Page Output item wired from a channel is named by it**: `Cy5` from a `chK` socket;
  `gaussian_blur_cy5` for a node on that single-channel stream (`_item_default`).
- **Tests**: `test_channel_provenance` (Qt-free: the seed function, every socket's text, the
  subset rule, the synthetic sockets, the cross-page total, an item, the fresh-output
  exclusion, the inherited narrowing without a seed, the item default); probe CP1 (the C1
  two-branch graph: the cards' row texts, the tinted input dots, the wire's colours, the
  hover line; it edits nothing, so C2 after it is undisturbed). **Docs**: MANUAL §4 *A
  stream names its channel*, a §18 row; codemap CON-23 (new), a CON-17 sentence; the stale
  prose in `channel_descriptors`, `read_channel_display`, `PER_CHANNEL_KEYS` and the runner's
  seed comment, which all said the names were kept off the seed.

## Why

The user: "ensure all data streams know where it came from. for example after we use a cy5
channel, into a node, it comes out as data. the wires are colored per channel, but why do we
just call it data, it should still be associated with the channel it came from."

The tint was right and the name was not because they came from different places: the wire
read the envelope's emission list, which the `channel_select` meta_transform narrows per tap,
while the names were a GUI-side capture on the Load card that the inherited walk refused to
carry through a `chK` edge, and the envelope — "kept to the calibration schema" — had no
names to fall back on. Fixing the label alone (walk through the tap) would have left a
`channel.select` by param, a bundle, a Page Input on a later page and the hover all wrong in
their own ways. Putting the names on the source envelope is the general fix: the engine
already treats them as per-channel lists, every narrowing is already in lockstep, and the GUI
then has one place to ask. The seed's schema rule was kept where it matters — `ctx.calib`
still refuses a non-calibration key — and relaxed only for two display lists the engine never
reads, with a single shared function so the file-pick seed and the pull's envelope cannot
drift (a drift would re-pull everything after the first pull of every file).

The display rule follows the tint's: a strict subset of the file's channels is "a channel",
the full bundle is not. Replacing a generic name and appending to a role name keeps the one
piece of information each socket has that the other does not.

## Files

- `nodelab_v2/ingest.py` — `channel_display_seed`, `with_channel_display`; docstrings
- `nodelab_v2/runner.py` — `_SYNTH_META` names; the display seed on every resolved envelope
  (`_ingest_locked`, `_open_direct`, `_resolve_bundle`); the seed comment
- `nodelab_v2/window.py` — the Load and bundle cards' seeds carry the display seed
- `nodegraph/metadata.py` — `PER_CHANNEL_KEYS` prose
- `nodelab_v2/document.py` — `GENERIC_DATASET_SOCKETS`, `_SYNTHETIC_SOCKET_RE`, the
  `page_channel_scope` hook, `socket_channels` / `channel_subset` / `channel_tag` /
  `socket_text` / `_page_scope`, `source_channel_total` across pages,
  `_inherited_channel_descriptors` narrowing, `_item_default`, `channel_descriptors` prose
- `nodelab_v2/workspace.py` — `input_channel_scope`; the hook in `_attach` / `_detach`
- `nodelab_v2/node_item.py` — `desc_qcolor`, `avg_qcolor`, `_tint_channel_socket(io)`,
  `_channel_phrase`, `_input_row_text` / `_output_row_text`, `set_domain_tip(channel=)`
- `nodelab_v2/edge_item.py` — `_channel_colors` through the document's resolution
- `nodegraph/selftest.py` (hotspot: tail of `main()`), `scripts/_nodelab_v2_phase5_probe.py`
- `MANUAL.md`, `codemap/concepts.md`, `codemap/curated.lock.json`, `codemap/gen/*`,
  `codemap/STATE.md`

## How to verify

`python run.py`, load a multi-channel ND2. On the Load card drag `2 · Cy5` into a Gaussian:
the Gaussian's left socket reads `Cy5` with a Cy5-tinted dot, its right socket `Cy5`; wire a
Threshold after it — `Cy5` again, with `image only` / `mask only` under it. Wire `0 · DAPI`
into a Measure's `raw` beside the Threshold on `data`: the rows read `Cy5` and `raw · DAPI`.
Hover a socket: `channel: Cy5 — 1 of 3 channels`. Wire the Load card's `image` (everything)
into a Median: `data` / `out`, untinted. Put a Page Output on the Cy5 branch and open the next
page: its Page Input's `out` reads `Cy5`. Pull anything: nothing re-labels.

## Gates

- [x] selftest — 188 tests run; only `test_write_movie` fails (pre-existing H.264 rounding).
  `test_codemap` and `test_channel_provenance` re-run on the final code and pass
- [x] GUI probe — ALL PHASE-5 GUI PROBES PASSED, 152 checks (new: CP1), run alone, exit 0.
  Its first run caught the input dot untinted (the tint was set only in `_layout`; fixed in
  `refresh`), its second a C2 failure caused by CP1 editing the live document (CP1 now edits
  nothing)
- [x] catalog snapshot — CATALOG IDENTICAL, 104 ops (no op changed); synopsis CURRENT
- [x] codemap — CODEMAP CURRENT (CON-23 new; WF-07 re-read, one sentence added, blessed)
- [ ] sync check — run before the push
