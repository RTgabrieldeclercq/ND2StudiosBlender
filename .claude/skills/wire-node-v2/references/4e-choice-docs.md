# §4e. `choice_docs` — one explanation per DROPDOWN OPTION (V2.21)

*A leaf of the **wire-node-v2** skill. Cited as `§4e`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


A `description` can only describe the control. For a dropdown that is not enough: the menu
then offers six bare tokens (`otsu`, `li`, `yen`, …) whose *differences* are the entire
reason the user opened it, and naming a method is not explaining it. So **every dropdown
documents every option**, on both spec types, and `Mode` additionally carries its own
`description` (it had none before V2.21):

* `ModeSpec.description` + `ModeSpec.choice_docs` — `{choice: prose}`.
* `SocketSpec.choice_docs` — the same, for `choices` (pick one) and `vocab` (pick many).

```python
Mode("method", ["otsu", "li"], label="Level",
     description="How the intensity cut is chosen — the methods differ in what they "
                 "ASSUME the histogram looks like.",
     choice_docs={
         "otsu": "Maximizes between-class variance: the classic bimodal split. Biased LOW "
                 "(mask too generous) when the foreground covers only a few percent.",
         "li":   "Minimum cross-entropy. Handles a SMALL, sparse foreground far better than "
                 "Otsu — the first thing to try when Otsu's masks come out too generous.",
     })
```

**Write it RELATIVE to the siblings.** What this option assumes about the data, and which
way the result moves if you pick it — that is what a menu is for. Say when an option is
cheap/expensive, when it needs something the others do not (a track column, a raster, a
download), and when it is refused in some state. Option prose is held to a lower length bar
than a param description (`_OPTION_DOC_MIN`, 60 chars) because "assumes two intensity
classes" is a complete answer.

Three vocabularies are **canned centrally** — do not re-write them per node:
`registry.DIM_CHOICE_DOCS` (the 2D/3D lever, carried by `DimMode()` for free),
`domains.domain_docs(names)`, and `reducers.reducer_docs(names)`. Each lives beside the
thing it describes, for the reason `DOMAIN_COLOR` does: it is a property of the data model,
not of a node, and five hand-written copies would only differ where one had rotted.

Same contract as `description`: **presentation-only and memo-neutral** — a Mode's docs
cannot reach the recipe hash, because the engine folds the resolved mode STATE (`__modes__`)
and never the `ModeSpec`. Registration refuses a `choice_docs` key that matches no option
(it would render no tooltip, indistinguishable from never writing one), a blank explanation,
and — new in V2.21, since Modes were previously unvalidated — a Mode `default` outside its
own `choices`. Rendered by `mode_hover_text` / `option_hover_text` on four surfaces: the
inspector row, each combo item, each vocab tick box, and the node card's popup menu (which
needs `setToolTipsVisible(True)`, or QMenu swallows the prose). Enforced by
`selftest::test_option_docs`, exemptions in `_OPTION_DOC_EXEMPT` on the same
external-paper-only grounds.
