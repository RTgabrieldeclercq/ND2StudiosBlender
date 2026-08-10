---
name: evidence-log
description: >-
  The collaborative analysis loop: stream what you are doing and thinking onto a local HTML
  page WITH THE IMAGE EVIDENCE, let the user confirm/reject/correct each step in a reply box,
  and keep an append-only record of every decision point so it can be shown to someone else
  later. Includes a blind per-object labelling sheet that turns the user's eye into ground
  truth, a hand-drawing tool for when the answer is a BOUNDARY rather than a category
  (polygons, circles, per-shape type defaulting to the colour), and exports the whole record
  as one self-contained HTML or as markdown. Invoke whenever an analysis involves judgment
  calls on real data — measurements that will set a default, a threshold, a size prior, a
  segmentation rule — or when the user wants to see your reasoning rather than your
  conclusion. Triggers:
  "show me what you are doing", "show me the evidence", "stream your reasoning", "log this",
  "add it to the dashboard", "am I on track", "let me confirm", "make this collaborative",
  "I need to tell you what is right or wrong with each image", "label these", "let me mark
  them", "let me draw the ground truth", "I'll show you what the shape should be", "outline
  these", "trace them", "hand-annotate", "keep a log for posterity", "why did we decide
  that", "export the log".
---

# The evidence log — collaborative analysis with a durable record

Two jobs, one mechanism.

**Now:** the user sees what you measured, *as a picture*, and can stop you before a wrong
number becomes a structural decision.
**Later:** the record explains why every threshold, prior and rule is what it is — including
the ones that were wrong — so it can be replayed, quoted, or shown to a third party.

This skill exists because of a specific failure. A size prior was measured by an automated
pipeline, never looked at, written into a plan as fact, and was wrong by several fold. The
user's instruction afterwards was: *"I want you to show me what you are doing and thinking.
I want to see the evidence for your reasoning."* Then, when the first version was text-only:
*"I don't just want text, the entire point is that you show me the image evidence."*

---

## §0 — The rules. These are not style preferences; each one comes from a specific failure.

1. **Every claim that will change what you build carries a figure.** A number in a table is
   not evidence, it is an assertion. Render it, *look at it yourself*, and attach it. If you
   cannot produce a figure for a claim, that is a signal the claim is not ready.

2. **Look at the output before it drives a structural decision.** Not after. The origin
   failure was not bad arithmetic — it was trusting an unviewed pipeline whose front end was
   already known to be broken.

3. **Never auto-refresh a page someone is reading.** A page that reloads itself moves the
   scroll position out from under the reader mid-sentence. Auto-reload is opt-in, off by
   default, and suppressed while they are typing. `render()` already does this; do not
   re-add a meta refresh.

4. **The log is append-only.** Never rewrite or delete an entry. A claim you withdraw stays,
   marked `rejected`, with the reason in its note. The record of *why a decision changed* is
   worth as much as the decision.

5. **Announce your own errors with `kind="error"`, prominently.** The `MY ERROR` badge exists
   so a correction is not buried. Correct yourself the moment you notice, and state which of
   your own numbers is retracted and what replaces it.

6. **When you need a judgment call, ask for a decision — not for a threshold.** Do not ask
   "where should the cut be?" Build a labelling sheet (§3) and ask them to label. Labels
   calibrate every candidate rule at once, out-of-sample, and become the target a synthetic
   fixture must reproduce. A threshold they name is still a guess.

7. **Never tune a rule to reproduce one example.** If the user corrects a single object,
   that is one data point. Fitting the threshold to it teaches you nothing about the others.

8. **A sweep whose winner sits on a boundary has not found an optimum, it has found the
   edge of the box.** This happened twice in one session: a solidity sweep over
   `arange(0.80, 1.00)` reported `0.800` as best, and a parameter sweep over `h ∈ {0.06,
   0.12, 0.20}` reported `0.20` as best. Both numbers were artefacts of the range. Widen
   until the optimum is *interior*, and say in the figure that it is.

9. **Cross-validate anything fitted to labels.** Fit on a subset, score on the held-out
   part, report out-of-sample numbers only. If the data has natural groups (conditions,
   subjects, fields), hold out a whole group — and if every split independently picks the
   same parameter, say so, because that is much stronger evidence than the score itself.

10. **State what your instrument is BLIND to, every time you quote it.** A count metric
    cannot see a boundary. On this project a partition scored 83.8% on counts and was
    reported as "essentially exact" while every divide sat inside a granule body; the
    over-claim survived until the user asked a question the metric could not answer. If the
    thing you are building is not the thing the metric measures, say so in the same
    sentence as the number — and if no instrument you have can see it, build one (§4)
    rather than quoting the one you have.

---

## §1 — Set it up

The engine is `devlog.py` in this skill directory. The **data** lives in the project
(`CodeLog/live/` by convention), so the code is versioned once and each investigation keeps
its own record.

```python
import sys; sys.path.insert(0, ".claude/skills/evidence-log")
import devlog as D
D.configure("CodeLog/live", title="V2.22 · granule workflow")   # once per script
```

Layout it creates and expects:

```
CodeLog/live/
  log.jsonl                append-only, one JSON object per entry — THE record
  index.html               regenerated by render(); needs img/ beside it
  img/*.png                the figures, referenced relatively
  label/                   a labelling sheet, if one exists (auto-linked in the header)
  gt/                      the hand-drawing tool, if one exists (auto-linked in the header)
  evidence_archive.html    export_archive() — one shareable file, figures inlined
  evidence_log.md          export_markdown() — for the repo or a paper
```

Write each batch of entries as a small script rather than inline calls, so the log is
reproducible and a mistake in the text is fixable before it is appended.

---

## §2 — Log as you work

```python
D.log(kind="found",
      title="One sentence that states the finding, not the activity",
      body="What it is.",
      why="Why it matters and what it changes. This is the part the user reads.",
      evidence=[{"type": "table", "cols": [...], "rows": [...], "caption": "..."},
                {"type": "kv", "rows": [["what", "value", "note"], ...]},
                {"type": "code", "text": "..."},
                {"type": "note", "text": "markdown"}],
      images=[{"src": "img/thing.png", "caption": "What to look at in this figure."}],
      status="confirmed")
```

`kind`: `did` · `found` · `decision` · `question` · `plan` · `error`
`status`: `info` · `confirmed` · `rejected` · `pending`

```python
eid = D.ask(title="...", body="...", why="...", asks="What I need from you.",
            images=[...])          # status=pending; pins to the top of the page
D.confirm(eid, note="...")         # or D.reject(eid, note="why it was withdrawn")
D.attach(eid, [{"src": ..., "caption": ...}])   # add figures to an earlier entry
print(D.render())                  # rebuild index.html — call at the end of every script
```

**Figure captions must say what to look at**, not what the figure is. "Top row: the six
largest objects; note the thin bright bridges joining them" beats "segmentation results".

**How the user replies.** Every card has a *your reply* box that persists in the browser.
They write in it and hit **copy for Claude**, which yields:

```
### e024  <the entry title>
their reply
```

That arrives as a normal chat message. Resolve the entry it names, and if they corrected
you, log the correction as its own `kind="error"` entry — do not quietly fix it.

---

## §3 — When a judgment call needs their eye: the labelling sheet

Use this the moment you find yourself about to invent a threshold, or when the user says
some version of *"I need a way to tell you what is right or wrong with each one."*

```python
import labelsheet as LS
# 1. YOU render one tile per item to <out>/img/<uid>.png — only you know what the
#    evidence should look like. Render every tile at ONE fixed physical scale so
#    relative size is a real cue and not an artefact of per-item zoom.
# 2. then:
LS.build(out_dir="CodeLog/live/label",
         manifest=[{"uid": "M15_0001", "group": "M15", "solidity": 0.93, ...}, ...],
         question="How many granules are in each object?",
         vocab=[("1", "one granule", "#34d399"), ("2", "two", "#fbbf24"),
                ("3", "three or more", "#fb7185"),
                ("x", "not a granule", "#a78bfa"), ("?", "cannot tell", "#64748b")],
         strata_key="solidity",            # the score you are calibrating
         priority=["M15_0658", ...],       # items already discussed — first, flagged
         note="Every tile is 400 um across; white bar = 100 um.")

rows, notes = LS.read("~/Downloads/labels.csv", manifest)   # join back
```

Two properties do the work, and both are easy to get wrong:

- **Blind.** The tile shows the raw evidence and nothing else — no measurement, no score, no
  verdict of yours. Put everything you measured in the manifest, join it afterwards. If your
  prediction is visible on the tile, the labels anchor to it and calibrate nothing.
- **Stratified, not sorted.** Ordering by the score being calibrated puts the whole
  interesting transition hundreds of tiles away, so partial work is useless.
  `strata_key` round-robins across contiguous bands, so **any** prefix is a balanced sample.
  (The first version of this sorted, and it had to be rebuilt.)

Keyboard labelling auto-advances to the next unlabelled tile; arrows navigate; there is a
notes box; labels persist so they can stop and resume. Ask for a `?` option and a
"not valid" option — *what the user cannot decide* and *what is not the thing at all* are
both findings.

**Then score honestly.** Report per-feature precision/recall/F1 cross-validated, say which
of your rules the labels destroyed, and state plainly when nothing reaches the gate. On this
project the labels retracted a "89% of area is multi-granule" claim down to 26.6%, showed
the best hand-picked cut was 0.93 when the optimum was 0.798, and showed the merge rate ran
opposite to the direction that had been argued.

---

## §4 — When the judgment call is WHERE, not WHICH: hand-drawn ground truth

The labelling sheet answers *how many* / *which category*. It cannot answer **where the
boundary is**, and no amount of it ever will — that is a property of the instrument, not of
the sample size. When what you are building is a boundary, a shape, a split or a merge, ask
the user to **draw the answer**.

```python
import groundtruth as GT
# 1. YOU render one image per field into <out>/img/<id>.png. Make the COLOUR MEAN SOMETHING:
#    map each channel to an RGB slot directly, because each shape's type defaults to the
#    colour underneath it. Use a FIXED intensity scale across fields — an adaptive one makes
#    the same physical brightness look different from field to field and biases where the
#    boundary gets drawn.
# 2. then:
GT.build("CodeLog/live/gt",
         tiles=[{"id": "M08", "file": "img/M08.png", "w": 1024, "h": 1024,
                 "um_per_px": 1.7183, "what": "pure F, MEDIUM"}, ...],
         question="Outline each granule; circle each cell.")

truth = GT.read("~/Downloads/ground_truth.json")     # {tile_id: TileTruth}
lab, cls = GT.rasterize(truth["M08"])                # label image + per-id class
inside  = GT.region_mask(truth["M08"])               # where they drew EXHAUSTIVELY
```

Polygons for bodies, circles for point-like things, and a **region rectangle** marking where
they outlined everything. The page is one self-contained file (tiles inlined as data URIs,
which is also what lets it read its own pixels — a `file://` page cannot `getImageData` a
separately-loaded image without tainting the canvas, and the colour default needs that call).

Four properties earn their complexity, and three of them exist because the naive version
produces a dataset that quietly cannot be scored:

- **The type defaults to the colour, and the user overrides it.** Typing hundreds of shapes
  by hand is what stops someone finishing. `a` (what the colour said) and `c` (what they
  settled on) are BOTH exported, so the override rate is a free measurement of how far
  channel dominance alone gets you — report it, it is a real result. An explicitly *pinned*
  type wins over the colour and is flagged, so it is not scored as a disagreement.
- **Ambiguity comes back as a question.** When the top two channels are within a margin the
  automatic type refuses to guess and falls back to *cannot tell*. A confident wrong default
  is worse than no default, because it is the one nobody re-checks.
- **The region rectangle is not optional bookkeeping.** Without it, a granule they missed and
  a granule they simply had not reached yet are the same thing in the file, and **recall
  cannot be computed at all**. With it, ten minutes of drawing is a valid dataset. The page
  nags until at least one is drawn.
- **Corners snap to a neighbour's corner.** Packed objects share edges, so most boundaries
  get drawn twice, once from each side. Unsnapped, the two versions differ by a few pixels
  and leave a sliver belonging to neither — boundary error the annotator invented, which then
  shows up in your score as if the algorithm had made it.

**Then hold yourself to it.** Ground truth turns "better than the last version" into an
absolute error in real units. Report boundary error in µm, not IoU alone — IoU on a large
object hides a boundary that is a full radius out of place. And say how much was drawn: a
number from six outlines is an anecdote.

### The other half: you label, they mark what is wrong

Drawing from scratch does not scale past a few hundred objects, and it is not what the user's
time is best spent on. Once a seed set exists, calibrate on it, label **everything**, and hand
it back for marking:

```python
GT.build("CodeLog/live/gt", tiles, prefill={"M08": proposals}, review=True,
         page_name="review.html")           # a separate page: their originals stay untouched
truth = GT.read("~/Downloads/granule_reviewed.json")
idx   = truth["M08"].confirmed()            # what carries their authority — see below
bad   = truth["M08"].merge_groups()         # the sets they say are ONE granule
```

**Review works by exception: they mark only the mistakes.** Asking for a verdict on every
object is a task nobody finishes, and a half-finished pass leaves most of the data in an
unusable middle state. So unmarked means correct — but *only* after they tick the field off.
Until then unmarked means **not looked at**, and those two must never be conflated:
`confirmed()` returns nothing from a field that was not signed off, because scoring your own
unchecked output against itself is the easiest way there is to manufacture a meaningless
number.

**Mark failure MODES, not a verdict.** `merge` (you over-split), `split` (you under-split),
`expand` (your boundary is biased), `wrong` (false positive). "Bad" would collapse four
different defects, with four different fixes, into one number. `expand` in particular is a
real granule with a wrong edge, so it counts for detection and not for boundary error —
`confirmed(for_boundary=False)` is the difference.

**A merge is a claim about a SET**, so it is collected as a numbered group of two or more, not
flagged per object: "this one is over-split" is not answerable alone, since the question is
*which pieces* are one granule. A group left with a single member is discarded.

**Order what you show them by your own uncertainty**, and calibrate that against their labels
rather than inventing it. A first attempt here used made-up thresholds and flagged 63% of
everything, which is not a priority order at all; scoring each proposal by *the fraction of
real granules more convex than it* made it one.

**Every mark above describes an outline that EXISTS, so the four together measure precision and
nothing else** — however many fields get signed off, *recall stays unquotable*. This is a
structural blind spot of review-by-exception, not an oversight to be fixed by marking harder,
and it went unnoticed here through 2725 reviewed outlines because "100% precision" reads like
an accuracy. So the tool carries a fifth thing, and it is a **drawing tool (`m`), not a fifth
mark**, because a miss is a claim about empty space with no shape to attach it to:

```python
truth["M08"].misses          # (row, col) points: "there is one here and you found nothing"
truth["M08"].recall()        # (recall, found, missed) — or None, and None is the point
```

`recall()` returns **`None` rather than a number** unless the field is signed off *and* carries
a region box. Both refusals are load-bearing: outside a region a missing outline and an
unexamined area are the same picture, and in an unsigned field the numerator is unknown too.
Handle the `None` — a pipeline that silently reports perfect recall when nobody looked is the
failure this exists to prevent.

---

## §5 — Posterity

```bash
# one self-contained HTML: figures inlined, reply UI stripped, safe to move or send
python .claude/skills/evidence-log/devlog.py --root CodeLog/live --archive

# the same record as markdown, for the repo or a methods section
python .claude/skills/evidence-log/devlog.py --root CodeLog/live --markdown
```

`--archive` re-encodes figures to fit (`--max-px`, default 1500) so the file stays sendable.
The archive is **frozen**: no reply boxes, no auto-reload, no relative paths to break. It
opens with a summary — entry counts by kind, how many confirmed, how many rejected — so a
reader sees immediately that the record includes the wrong turns.

What to commit: `log.jsonl`, `img/`, `evidence_log.md`, and any `label/manifest.json` plus
the label CSV. Those are the record. The archive HTML and the label tiles are regenerable
and belong in `.gitignore`.

**Reference the log from durable documents.** When a plan or a docstring states a threshold,
cite the entry id that established it (`see e031`). That is what makes the record useful
six months later instead of merely large.

---

## §6 — Checklist before you hand the page back

- [ ] Every entry that makes a claim has a figure, and you looked at each figure yourself.
- [ ] Captions say what to look at.
- [ ] Anything you got wrong is logged as `kind="error"`, not silently edited away.
- [ ] Pending questions state what you need and what changes depending on the answer.
- [ ] Any number fitted to labels is cross-validated, and its sweep range is not clipped.
- [ ] Every metric quoted says what it is blind to, and nothing is scored against a metric
      that cannot see the thing being built.
- [ ] `render()` ran; you told the user to reload (the page will not do it by itself).
- [ ] Claims that reached a plan or docstring cite the entry id behind them.
