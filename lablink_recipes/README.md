# LabLink recipes — a starter kit, not a shipped configuration

A **recipe** is what a remote machine is allowed to ask a LabLink hub to run: a curated
pipeline plus the handful of knobs it may turn. Two files per recipe, in one directory:

```
nd2studios/<name>/recipe.json          the manifest — knobs, inputs, outputs, limits
nd2studios/<name>/graph.nd2graph.json  the pipeline — exactly what NodeLab saves
```

**Recipes are hub-owned, and that is the security boundary.** They live on the machine that
runs the hub, curated by the operator who owns it; a node names one and can never supply a
command, a path, an argument or a graph of its own. So the directory here is a *starter kit*
to copy or point a hub at, not something ND2Studios activates by itself. Nothing in this
folder does anything until a `hub.json` names it:

```json
"recipes_dir": "C:/path/to/ND2StudiosBlender/lablink_recipes/nd2studios"
```

Full setup, the two-tier validation model and the knob rules are in
[`MANUAL.md` §17b](../MANUAL.md#17b-lablink-mode--serve-the-lab-or-send-work-out).

## What is here

| Recipe | What it is for |
|---|---|
| `nd2studios/selftest-synthetic` | Proves the whole loop with **no data files at all** — an empty `io.load` path yields the deterministic synthetic source, so the result is reproducible and can be pinned by hash. It is what `lablink hub doctor` runs, what `nodegraph.selftest`'s `test_lablink` re-validates against the live node catalog, and the worked example to copy. Send it a real ND2 or TIFF and the same five nodes run on that instead. |

## Writing your own

1. **Build the graph in NodeLab and save it.** The recipe's graph is a plain
   `*.nd2graph.json` read by the same `nodegraph.serialize` the editor writes, so there is no
   second format to learn and no hand-authoring to get wrong.
2. **Copy `selftest-synthetic/recipe.json` and edit it.** Set `workflow` to the workflow name
   in your `hub.json`, `targets.primary` to the node a run should pull, and one `inputs`
   entry per `io.load` node in the graph.
3. **Declare each knob against the real socket.** `kind` must say whether it writes the
   node's `params` or its `modes` — they are read from different places, so the wrong one is a
   silent no-op — and `unit` must match the socket's own or the number quietly means
   something else. The node reference in `MANUAL.md` §15 lists both.
4. **Check it, then prove it.**

   ```powershell
   python -m lablink hub --check-config hub.json    # tier 1: manifest vs graph
   python -m lablink hub doctor --config hub.json --root lablink_data
   ```

   `--check-config` deliberately reports what it has *not* verified: the four checks that
   need the node catalogue (socket exists, unit agrees, mode value is real, output kind is
   producible, plus every 2D/3D lever set explicitly) happen in the worker at session start.
   `doctor` is the one that exercises them, because it runs a real session.

## Two mistakes worth naming

* **`unset_means: "derive"` on a param the graph already pins.** The hub refuses this
  outright. As written the recipe asks for the value to come from the file's own
  calibration while the graph fixes it — so the microscope is ignored and nothing reports
  it.
* **A node with a 2D/3D lever whose `dim` mode is unset.** Refused by the worker at `open`.
  Left unset it runs the default on whatever arrives, so a z-stack is processed plane by
  plane and the result looks plausible.

## Methods that need weights

`analysis.segment` really does offer `stardist` and `cellsam`, and `selftest-synthetic`
deliberately whitelists only `threshold` and `watershed`. A recipe that wants a model-based
method should name it in `requires.capabilities` (`"stardist"`, `"cellsam"`) so the hub
refuses the session up front, with the worker's own reason, on a machine where the import or
the weights are missing — rather than triggering a multi-gigabyte download because a remote
node turned a dropdown.
