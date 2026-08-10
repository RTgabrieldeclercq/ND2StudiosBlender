# §5c. A MODE can be gated on another Mode too — `ModeSpec.available_in` (V2.12)

*A leaf of the **wire-node-v2** skill. Cited as `§5c`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


`available_in` is **not socket-only**. A `ModeSpec` takes the same mapping, resolved by
`ModeSpec.active_in(state)` / **`NodeSpec.active_modes(state)`**, and it exists because a
node that unifies several algorithms behind one `method` Mode usually has a *second* Mode
only some methods read. `analysis.segment`'s `level` (which foreground cut: otsu/li/…/fixed)
is gated to `{"method": {"threshold", "watershed"}}` — its two learned detectors never
threshold, so the dropdown is **hidden**, not shown and ignored. Ungated it would be the
same dead control §4b clause 2 forbids for sockets, one level up.

```python
modes=[DimMode(),
       Mode("method", ["threshold", "watershed", "stardist", "cellsam"]),
       Mode("level", [...], available_in={"method": frozenset({"threshold", "watershed"})})]
```

Rules, all enforced by `selftest::test_param_socket_contract` clause (a0):

* the referenced mode and values must exist (a typo hides the Mode in *every* state — and
  unlike a socket there is no `inputs` list where its absence is obvious);
* **a Mode must never gate on itself** — its visibility would depend on the value it is
  choosing.

**Gating is edit-time only, exactly like a socket's.** `default_state()` still includes
hidden Modes, so a gated-away Mode keeps its value: the compute's
`ctx.params["__modes__"].get(name)` still resolves, and the mode still folds into the
recipe hash, so hiding one never changes a memo key. Both GUI halves read
`active_modes`: the inspector's Mode section (`inspector._rebuild`) and the card's mode
rows (`node_item._layout`) — and `NodeItem.refresh()` compares the active **mode** name
list alongside the socket lists, or a method switch that gates only a Mode would not
relayout.

**What gating cannot fix:** a lever whose *value* the compute still resolves. Do not gate
the `dim` lever away for a 2D-only method — the state would still resolve `dim` from its
`derive` and hand the compute `WHOLE_VOLUME`. Refuse instead, naming the fix
(`analysis.segment` refuses 3D for stardist/cellsam and says "set the lever to 2D").
