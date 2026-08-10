# §4f. `reads_domains_by_mode` — the domain rail per BRANCH (V2.22)

*A leaf of the **wire-node-v2** skill. Cited as `§4f`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


`NodeSpec.reads_domains` says which domains must be present on the incoming Dataset; it drives
the card's input rail, the wire tint, and the red missing-domain chips. A node whose **branches
read different structure** cannot state that once, and the catalog had drifted both ways:

* **over-claiming** — `analysis.object_field` declared the static union `{LABEL, POINT}`, so a
  pure-Label graph that was fine got a red PT chip;
* **under-claiming** — `analysis.voronoi` declared `{POINT}` while `bound=per_region` also goes
  through `_label_raster` and needs a whole Label instance. The node that combines dots with
  labels advertised no labels requirement at all (reported 2026-08-03).

Both are silent — nothing at pull time reads the rail, so it just tells the user the wrong thing.
Declare the conditional half instead:

```python
reads_domains=frozenset({Domain.POINT}),          # unconditional (the seeds)
reads_domains_by_mode={"bound": {
    "per_region": frozenset({Domain.VOXEL, Domain.LABEL}),   # _label_raster wants the table
    "mask":       frozenset({Domain.VOXEL}),                 # any non-zero raster
    "frame":      frozenset(),                               # reads no layer at all
}},
```

* It is a **UNION over every listed mode**, not the single-key `Mapping` `granularity` uses —
  because the requirements are not keyed by one dropdown. `transform.transfer_structure` reads
  what `from_domain` names AND what `to_domain` names; `flow.iterate` needs a Global under
  `preserve=best` **OR** `mode=feedback`. An unlisted value contributes nothing.
* Resolve it with **`spec.resolve_reads_domains(state)`** / **`missing_domains(incoming, state)`**
  — never read the raw field. The GUI already does (`node_item.reads_domains`,
  `document.missing_domains`).
* Registration refuses a mode name that does not exist, a value that mode cannot take, and a
  non-`Domain` entry — each would contribute nothing in every state, i.e. look exactly like
  never having written it.
* **Don't reach for it when the domain is the image.** `analysis.segment` keeps a static
  `{VOXEL}`: that is the image domain every source supplies, not a per-method layer.
