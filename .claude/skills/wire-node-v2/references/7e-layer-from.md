# §7e. A second input for a second DOMAIN — `layer_from` (V2.22)

*A leaf of the **wire-node-v2** skill. Cited as `§7e`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


`raw`/`reference` bring in *pixels*. The other reason to take a second Dataset is that the
node **combines two domains that different branches produce**: `analysis.voronoi` needs a
Point table (the seed dots) *and* a Label raster (the areas to clip them to), and no single
wire can carry a Point table plus someone else's labels. Reported as "I can only connect
one data at a time" (2026-08-03).

Same four rules as above, plus one:

- **The layer socket must name the input it reads** — `layer_from="areas"` on the
  `layer_in` socket. `document.layer_choices` follows the **primary** edge by design (a
  `raw`/`reference` input's layers must never be offered as if they were on the payload
  wire), so without this the picker lists names off a wire the compute never reads and the
  user sees a valid-looking layer that "does not exist". Registration refuses a
  `layer_from` naming a non-existent input, or the primary (already the default).
- **The picker exists on BOTH surfaces.** `inspector._layer_box` (an editable combo) and
  `node_item._open_layer_menu` (the card's pill popup). Until 2026-08-04 only the inspector
  had one, so on the canvas a layer socket could only be typed — and a node left holding its
  factory-default layer name while the wire carried another failed at pull time, several
  nodes downstream, with a menu one click away that had the right answer in it. Both keep
  free text reachable: the edit-time prediction is honest but incomplete.
- **Falls back to the primary when unwired**, in the compute *and* the picker — that is
  what keeps every single-wire graph that predates the socket working unchanged.
- **Refuse the input in a mode that ignores it.** `analysis.voronoi` raises if `areas` is
  wired under `bound=frame`, which reads no area layer at all — clause 2 of the socket
  contract applied to a Dataset input.
- **Do NOT `available_in`-gate a Dataset input.** No node in the catalog does: hiding a
  socket that already has a wire leaves the edge dangling. Refuse instead.

- **Put what it used on the OUTPUT.** The payload is built on `dataset_preds[0]`, so it inherits
  the primary wire's layers and *nothing* from the auxiliary one. Viewing `analysis.voronoi`
  therefore showed the seeds' branch labels and points with no trace of the areas the cells were
  clipped to — and the inherited raster is usually *also* called `labels`, so the overlay looked
  like the area layer while showing a different branch's. Copy it under a name of the node's own
  choosing (`f"{name}_areas"`), declared via `extra_layers`; never under the source name, which
  collides with what the primary branch already means by it.
- **Declare it `view_source=True` if its IMAGE should be visible.** One payload has one image, so
  the Viewer can only draw the primary's channel — "I can only see the UV channel" on a graph
  whose areas came from the Red one. `InDataset("areas", view_source=True)` makes the runner
  composite it (`overlay_chain` yields it; `_view_source_entry` synthesizes the placement, since
  these nodes stamp no recipe — display config in a payload would ride the memo key).
  **Opt-in per socket, and most sockets must NOT have it:** `raw` is the unenhanced version of
  the *same* pixels and would draw the field twice, and a `reference` is another timepoint of
  the same channel. The test is whether the wire carries genuinely different content, which is
  the same condition as reading a *domain* off it rather than intensities. Placement is
  field-for-field at scale 1 (`on_unplaceable="index"`), not by stage position: stage placement
  refuses without a stage log, which a sibling branch does not need and a TIFF never has.

The domain rail composes for free: `document.input_domains` unions **every** Dataset
predecessor, so the LABEL arriving on the second wire satisfies the `reads_domains_by_mode`
requirement declared for `bound=per_region` (§4f).
