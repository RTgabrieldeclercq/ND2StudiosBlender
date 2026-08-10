# §4g. A layer name is DERIVABLE — infer it (`_resolve_layer`, 2026-08-04)

*A leaf of the **wire-node-v2** skill. Cited as `§4g`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


A layer-name socket has to carry some default, and a **literal** default is right for exactly
one upstream producer. `analysis.voronoi` shipped `points="particles"` — what `detect.particles`
emits — so a graph seeded from `detect.spots` (`spots`) or `transform.label_to_points`
(`<labels>_points`) was refused with *"no Point layer 'particles' on the input Dataset (it
carries ['points'])"*: an error that prints the right answer one clause after declining to use
it. There was nothing to decide; a single Point table was on the wire.

Treat that the way you treat a spatial param: **derivable from the incoming data ⇒ the node
derives it** (§7). Use `_resolve_layer(candidates, want, …)` (`_shared/labels.py`):

| state | behaviour |
|---|---|
| socket set, name **present** | use it — an explicit name always wins |
| socket **empty**, one candidate | use it, and report on the progress rail |
| socket set but name **absent**, one candidate | use the candidate, saying the socket is stale |
| **zero** candidates | raise, naming what to wire upstream |
| **two or more** candidates | raise, listing them — never guess |

* **Default `""`, not a literal.** Say in the `description` what empty means ("leave it empty and
  the only Point table on `data` is used"), the same job `path_hint` does for a path socket.
* **Candidates are per-purpose, not per-domain.** `per_region` needs a whole Label **instance**
  (`_label_instances`: a raster *plus* the table that divides it into objects), so a plain `mask`
  or a `distance` field beside a label raster is still unambiguous. `mask` takes any raster
  (`_voxel_layers`) and so is ambiguous more often — correctly.
* **Report the inference.** `ctx.progress(…, "using seeds: Point table 'points' (the only one on
  the `data` input)")`. Inferring silently is how you get a node that quietly analyses the wrong
  layer.
* Never infer across a **domain** boundary or between candidates: two Point tables on one wire is
  a real question, and picking one would tessellate the wrong cloud with no symptom.
