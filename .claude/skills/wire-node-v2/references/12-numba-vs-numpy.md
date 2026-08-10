# §12. Compute performance — numba vs numpy/scipy (when to reach for it)

*A leaf of the **wire-node-v2** skill. Cited as `§12`; the summary and the pointer here live in [SKILL.md](../SKILL.md).*

---


A compute body is fast **by default** when it bottoms out in one vectorized numpy op
or a scipy/skimage/cv2 call — those are already compiled C/Fortran/MKL and numba
**cannot beat them**. Reach for numba **only** when the hot cost is a *Python loop of
many small array ops* or a *sequential, data-dependent algorithm that won't vectorize*.

**The decision rule** — ask of the hot path: *is wall-time dominated by one big compiled
call, or by a Python loop doing thousands of tiny ops?*

| The compute's hot path… | Use | Why |
|---|---|---|
| one big `ndimage.*` / `skimage.*` / `cv2.*` / `np.fft` / vectorized numpy | **numpy/scipy** | already C/MKL; numba ties or loses **and** adds compile latency |
| element-wise math numpy expands into N temporaries (`a*b + c*d - e`) | **numba** `njit` | fuses one pass, no temp arrays (memory-bandwidth bound) |
| a Python `for` over 10³–10⁶ items, each a few small numpy calls | **numba** `njit` | erases per-call dispatch + alloc overhead (the real win: 10–100×) |
| sequential/data-dependent (greedy NMS, region-grow, per-subset solve, ODE step) | **numba** `njit` | can't vectorize; numpy forces slow-Python or wasteful over-compute |
| the above **and** embarrassingly parallel per element | **numba** `njit(parallel=True)` + `prange` | frees the GIL, uses all cores in-process |

**Codebase idiom** (match it — see `kernels/bead_detect.py`, `kernels/track_objects.py`):
`import numba as nb`; module-level `@nb.njit(cache=True)` kernels with explicit dtypes;
`@nb.njit(parallel=True, cache=True)` + `nb.prange` for the parallel ones. Keep the
kernel a **pure numeric helper** (arrays in, arrays out) — no `ctx`, no `Dataset`, no
scipy calls inside `njit` (numba can't call them; keep `map_coordinates`/`label`/FFT
outside the kernel and pass the arrays in).

**Non-negotiable gotchas for THIS engine** (streaming + spawn):
- **`cache=True` is mandatory.** v2 does per-tile streaming eval and the DVC path fans
  across processes with `ProcessPoolExecutor` — Windows uses **spawn**, so every worker
  is a fresh interpreter that re-JITs on first call (~0.3–2 s each). `cache=True` writes
  the compiled artifact to disk so workers/tiles/re-runs load instead of recompiling.
  Without it numba can go **net-negative** on short interactive pulls.
- **First-ever call still pays one cold compile.** Fine for a long analysis solve; weigh
  it for a tiny per-tile pointwise op (there, a numba port may not be worth it at all —
  vectorized numpy already wins those).
- **Confirm the numba cache dir is writable** in the packaged/frozen app, or every run
  recompiles cold.
- **Don't stack redundant parallelism.** If the node already fans across processes
  (DVC IC-GN), `parallel=True` inside the kernel competes with the pool — prefer a
  serial `njit` kernel per worker, or `prange` only on the non-fanned path.

Backends stay **lazily imported inside the compute** (§2); a numba kernel is a
module-level helper, so importing it at module top is fine (numba is a declared dep) —
but never let a kernel's presence pull scipy/skimage to module top.
