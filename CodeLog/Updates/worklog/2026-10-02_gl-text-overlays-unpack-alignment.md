# GL image surface: text over the image rendered as white blocks with black stripes

- **Date:** 2026-10-02
- **Author:** McGheeLab (Claude Fable 5.1 session)
- **Branch:** Blender (built before the team-sync integration step; see the team-sync entry)
- **Base:** 6fbd719

## What changed

`GLImageView._upload` now restores `GL_UNPACK_ALIGNMENT` to the GL default of 4 straight
after the `glTexImage2D` that needed it at 1. One line plus the comment that explains it.
Nothing else about the upload, the shader or the overlays changed. The manual's
troubleshooting table gained a row for the symptom.

## Why

Reported symptom: text drawn over the image in the Viewer (region ids, track ids, split-pane
names, the scale-bar label, the SOLO chip) showed as white blocks with black stripes, while
some labels still read fine.

Cause: the R16 plane upload (2026-08-10) set the unpack alignment to 1 so odd-width 16-bit
rows are not sheared, and never put it back. Pixel-store state belongs to the context, and
QPainter's text engine shares that context: it uploads every newly used glyph into its
glyph-cache texture with 4-byte-aligned scanlines, assuming the default alignment, and never
sets the alignment itself. So every glyph first rasterised *after* the first frame was read
with the wrong stride and landed in the cache as a striped block. The cache is never
refilled, so the damage lasted for the session, and glyphs cached before the first upload
(the chrome drawn before any image arrived) stayed intact, which is why only *some* text
broke. Reproduced off-screen on this machine with a `QOffscreenSurface` + FBO on a 3.3 core
context: alignment left at 1, then text made of glyphs the process had never drawn, renders
every glyph as a striped block; alignment restored to 4 first renders them cleanly. A
third render showed the persistence: glyphs cached while the alignment was 1 stay broken
even after it is restored, and only glyphs new to the cache come out right.

Restoring the alignment at the upload site, rather than wrapping the native draw in
`beginNativePainting`/`endNativePainting`, because Qt's native-painting bracket documents the
state it restores and the pixel store is not in that list; the fix has to be explicit either
way, and the place that changes a piece of context state is the place that should undo it.

## Files

- `nodelab_v2/glview.py` — `_upload` restores `GL_UNPACK_ALIGNMENT` after the texture upload
- `MANUAL.md` — §18 troubleshooting row for the symptom
- `codemap/gen/*`, `codemap/STATE.md` — regenerated (line numbers in `glview.py`)

## How to verify

In the GUI with the GL surface active (the default): load any file so a frame is uploaded,
then turn on label ids, a split view or the Viewer node's scale bar. Every label is legible.
Before the fix, any glyph not already drawn before the first frame rendered as a striped
block for the rest of the session.

Off-screen: the scratch script `glyph_align_check.py` from this session (not committed; a
`QOffscreenSurface` + `QOpenGLFramebufferObject`, set alignment 1, optionally restore 4, draw
unseen glyphs via QPainter, save the FBO as PNG) shows striped blocks with `RESTORE=0` and
readable text with `RESTORE=1`. The pixel correlation against a CPU `QImage` render that the
script also prints is NOT a usable pass/fail: GL and raster text differ in sub-pixel layout,
so it reads 0.33 broken versus 0.49 fixed. Judge the PNGs.

## Gates

- [ ] `PYTHONUTF8=1 python -B -m nodegraph.selftest` — not re-run for this one-line GL change;
  the selftest is Qt-free and never touches `glview.py`. Still red at HEAD for the
  pre-existing reasons recorded in the team-sync entry.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png` -> ALL PHASE-5 GUI PROBES PASSED
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no node changed)
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> pending the team-sync integration step
