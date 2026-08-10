"""Evidence log — the surface an agent streams its reasoning and image evidence onto,
and a permanent record of why each decision was taken.

WHY THIS EXISTS
    A structural decision was once made from an automated measurement nobody looked at.
    It was wrong by several fold and it reached the plan as fact. The rule that came out
    of it: show the work, show the *picture*, and wait for a confirm or reject before
    building anything on top of it.

    The second reason is posterity. Six months later nobody remembers why a threshold is
    0.80 rather than 0.93, which claims were retracted, or which numbers the user's own
    eye overruled. `log.jsonl` is append-only and keeps all of it, including the wrong
    turns, so the record can be replayed, exported, and shown to someone else.

HOW IT WORKS
    Append-only `log.jsonl` + a regenerated `index.html` beside it. No server and no
    fetch(), so it works straight off `file://`. Images are referenced relatively out of
    `img/` because the evidence is image-heavy and inlining it would bloat every rewrite;
    `export_archive()` inlines them once, on demand, when you want one shareable file.

    The page does NOT reload itself. Auto-reload is an opt-in button, off by default, and
    it refuses to fire while the reader is typing — an auto-refreshing page moves the
    scroll position out from under someone mid-sentence, which the user reported as the
    single most annoying thing about the first version.

USAGE
    import sys; sys.path.insert(0, ".claude/skills/evidence-log")
    import devlog as D
    D.configure("CodeLog/live")            # where log.jsonl and img/ live; call once
    D.log(kind="found", title="...", why="...", evidence=[...], images=[...])
    print(D.render())

    Entry kinds:  did | found | decision | question | plan | error
    Status:       info | confirmed | rejected | pending
    `ask(...)`      a pending question; pins to the top of the page.
    `confirm(id)` / `reject(id, note)` resolve a pending entry, keeping its original text.
    `attach(id, images)` adds figures to an entry written earlier.
    `render()`      rebuild index.html.
    `export_archive()` one self-contained HTML with the images inlined.
    `export_markdown()` the same record as markdown, for the repo or a paper.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("EVIDENCE_LOG_ROOT") or os.path.join(os.getcwd(), "CodeLog", "live")
LOG = os.path.join(ROOT, "log.jsonl")
PAGE = os.path.join(ROOT, "index.html")
IMG = os.path.join(ROOT, "img")
REFRESH_S = 30          # only used when you switch auto-reload ON; off by default
TITLE = "evidence log"


def configure(root: str, *, title: Optional[str] = None) -> str:
    """Point the log at `root`.  Creates `root` and `root/img` if missing."""
    global ROOT, LOG, PAGE, IMG, TITLE
    ROOT = os.path.abspath(root)
    LOG = os.path.join(ROOT, "log.jsonl")
    PAGE = os.path.join(ROOT, "index.html")
    IMG = os.path.join(ROOT, "img")
    if title:
        TITLE = title
    os.makedirs(IMG, exist_ok=True)
    return ROOT

KINDS = {
    "did":      ("DID",       "#38bdf8"),
    "found":    ("FOUND",     "#a78bfa"),
    "decision": ("DECISION",  "#22d3ee"),
    "question": ("NEEDS YOU", "#fbbf24"),
    "plan":     ("PLAN",      "#94a3b8"),
    "error":    ("MY ERROR",  "#fb7185"),
}
STATUS = {
    "info":      ("", ""),
    "confirmed": ("CONFIRMED", "ok"),
    "rejected":  ("REJECTED",  "bad"),
    "pending":   ("AWAITING YOUR CALL", "wait"),
}


# ────────────────────────────── writing ──────────────────────────────
def _read() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG):
        return []
    out = []
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _write_all(entries: List[Dict[str, Any]]) -> None:
    with open(LOG, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _append(entry: Dict[str, Any]) -> None:
    """Append ONE line, without rewriting what is already there.

    `log()` used to read the whole file and write it back for every entry. That is O(n^2),
    but the reason it had to change is a data-loss window: two processes logging at once --
    a long pull writing its result in one terminal while a quick correction is written in
    another -- each read the same list, and whichever finished second wrote its own copy over
    the other's entry. For an append-only record whose whole promise is that nothing is ever
    silently lost, that is the wrong failure to leave in.

    A residual race remains and is deliberately left: the entry ID is derived from the
    current line count, so two simultaneous writers can pick the same `eNNN`. That duplicates
    a label rather than destroying a record, and both entries survive.
    """
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _warn_missing(images: Optional[List[Dict[str, str]]]) -> None:
    """Print a warning for any figure path that is not on disk.

    A broken `src` renders as an empty box, which the reader interprets as "there is no
    evidence" rather than as "the file moved". That is the one failure mode this module
    cannot afford to be quiet about, so it is reported at the moment it is logged, while
    you can still fix the path.
    """
    for i in images or []:
        src = i.get("src", "")
        if src and not src.startswith("data:") \
                and not os.path.exists(os.path.join(ROOT, src)):
            print(f"  ! figure not found: {os.path.join(ROOT, src)}")


def log(*, kind: str = "did", title: str, body: str = "", why: str = "",
        evidence: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, str]]] = None,
        asks: str = "", status: str = "info", eid: str = "",
        render_now: bool = True) -> str:
    """Append one entry and (by default) regenerate the page. Returns the entry id."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    if status not in STATUS:
        raise ValueError(f"status must be one of {sorted(STATUS)}")
    _warn_missing(images)
    eid = eid or f"e{len(_read()) + 1:03d}"
    _append(dict(id=eid, ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind,
                 title=title, body=body, why=why, evidence=evidence or [],
                 images=images or [], asks=asks, status=status, note=""))
    if render_now:
        render()
    return eid


def ask(*, title: str, body: str = "", why: str = "", asks: str,
        evidence=None, images=None) -> str:
    """A question that pins to the top of the page until resolved."""
    return log(kind="question", title=title, body=body, why=why, asks=asks,
               evidence=evidence, images=images, status="pending")


def _resolve(eid: str, status: str, note: str = "") -> None:
    entries = _read()
    for e in entries:
        if e["id"] == eid:
            e["status"], e["note"] = status, note
            break
    else:
        raise KeyError(f"no entry {eid!r}")
    _write_all(entries)
    render()


def attach(eid: str, images: List[Dict[str, str]], *, prepend: bool = False) -> None:
    """Add figures to an entry that already exists.

    Every entry that reports a measurement should carry the picture the measurement was
    read off. This exists so a text-only entry can be retrofitted rather than rewritten.
    """
    _warn_missing(images)
    entries = _read()
    for e in entries:
        if e["id"] == eid:
            cur = e.get("images") or []
            e["images"] = (list(images) + cur) if prepend else (cur + list(images))
            break
    else:
        raise KeyError(f"no entry {eid!r}")
    _write_all(entries)
    render()


def confirm(eid: str, note: str = "") -> None:
    _resolve(eid, "confirmed", note)


def reject(eid: str, note: str = "") -> None:
    _resolve(eid, "rejected", note)


# ────────────────────────────── rendering ──────────────────────────────
def _md(t: str) -> str:
    """Deliberately tiny markdown: **bold**, `code`, blank-line paragraphs."""
    t = html.escape(t or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.S)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t, flags=re.S)
    paras = [p.strip().replace("\n", " ") for p in t.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paras)


def _cells(ev: Dict[str, Any], *names: str) -> List[Any]:
    """The first non-empty of several accepted key spellings.

    Table headers have been written both as `cols` (what SKILL.md documents, and what
    export_markdown read) and as `headers` (what the HTML renderer read); `kv` payloads
    likewise as `rows` and as `items`. Each spelling rendered on one surface and vanished
    without a word on the other — a table logged per the docs came out of render() with an
    empty header row, and a `kv` block came out completely blank. Silently dropping
    evidence is the precise failure this module exists to prevent, so both spellings are
    now accepted on both surfaces.
    """
    for n in names:
        v = ev.get(n)
        if v:
            return list(v)
    return []


def _ev(ev: Dict[str, Any]) -> str:
    kind = ev.get("type")
    cap = f'<div class="cap">{_md(ev["caption"])}</div>' if ev.get("caption") else ""
    if kind == "table":
        head = "".join(f"<th>{html.escape(str(h))}</th>"
                       for h in _cells(ev, "cols", "headers"))
        rows = []
        for r in ev.get("rows", []):
            tds = []
            for c in r:
                s = str(c)
                cls = ""
                if s.startswith("OK "):
                    cls, s = ' class="ok"', s[3:]
                elif s.startswith("BAD "):
                    cls, s = ' class="bad"', s[4:]
                tds.append(f"<td{cls}>{html.escape(s)}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
        return (f'{cap}<div class="scroll"><table><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')
    if kind == "kv":
        items = "".join(
            f'<div class="kv"><span class="k">{html.escape(str(k))}</span>'
            f'<span class="v">{html.escape(str(v))}</span>'
            f'<span class="n">{_md(str(n)) if n else ""}</span></div>'
            for k, v, *rest in _cells(ev, "rows", "items")
            for n in [rest[0] if rest else ""])
        return f'{cap}<div class="kvs">{items}</div>'
    if kind == "code":
        return f'{cap}<pre class="code">{html.escape(ev.get("text", ""))}</pre>'
    return f'{cap}<div class="note">{_md(ev.get("text", ""))}</div>'


def _entry(e: Dict[str, Any], pinned: bool = False) -> str:
    label, colour = KINDS.get(e["kind"], ("?", "#888"))
    slabel, scls = STATUS.get(e.get("status", "info"), ("", ""))
    badge = (f'<span class="status {scls}">{slabel}</span>' if slabel else "")
    ev = "".join(_ev(x) for x in e.get("evidence", []))
    imgs = "".join(
        f'<figure><a href="{html.escape(i["src"])}" target="_blank">'
        f'<img src="{html.escape(i["src"])}" loading="lazy" alt=""></a>'
        f'<figcaption>{_md(i.get("caption", ""))}</figcaption></figure>'
        for i in e.get("images", []))
    why = (f'<div class="why"><span class="lbl">why</span>{_md(e["why"])}</div>'
           if e.get("why") else "")
    asks = (f'<div class="asks"><span class="lbl">what I need from you</span>'
            f'{_md(e["asks"])}</div>' if e.get("asks") else "")
    note = (f'<div class="note yours"><span class="lbl">your note</span>'
            f'{_md(e["note"])}</div>' if e.get("note") else "")
    reply = (f'<details class="reply"{" open" if pinned else ""}>'
             f'<summary>your reply</summary>'
             f'<textarea data-eid="{e["id"]}" '
             f'data-title="{html.escape(e["title"], quote=True)}" '
             f'placeholder="Write your response here — agree, disagree, correct me, or ask '
             f'for something. It saves as you type; use Copy for Claude at the bottom right '
             f'when you are done."></textarea><div class="st"></div></details>')
    return f"""<article class="card {'pin' if pinned else ''}" id="{e['id']}">
  <header>
    <span class="kind" style="--c:{colour}">{label}</span>
    <h2>{html.escape(e['title'])}</h2>
    {badge}
    <span class="ts">{e['ts']} · {e['id']}</span>
  </header>
  {_md(e.get('body', ''))}
  {why}{ev}{imgs}{asks}{note}{reply}
</article>"""


CSS = """
*{box-sizing:border-box}
:root{
  --bg:#0b0f17; --panel:#111827; --panel2:#0f1520; --line:#1f2937;
  --fg:#e5e7eb; --dim:#9ca3af; --ok:#34d399; --bad:#fb7185; --wait:#fbbf24;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:light){
  :root{--bg:#f8fafc;--panel:#fff;--panel2:#f1f5f9;--line:#e2e8f0;
        --fg:#0f172a;--dim:#52627a;--ok:#059669;--bad:#dc2626;--wait:#b45309}
}
:root[data-theme=light]{--bg:#f8fafc;--panel:#fff;--panel2:#f1f5f9;--line:#e2e8f0;
  --fg:#0f172a;--dim:#52627a;--ok:#059669;--bad:#dc2626;--wait:#b45309}
:root[data-theme=dark]{--bg:#0b0f17;--panel:#111827;--panel2:#0f1520;--line:#1f2937;
  --fg:#e5e7eb;--dim:#9ca3af;--ok:#34d399;--bad:#fb7185;--wait:#fbbf24}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 80px}
header.top{position:sticky;top:0;z-index:10;background:var(--bg);
  border-bottom:1px solid var(--line);padding:16px 0 12px;margin-bottom:22px}
header.top .row{max-width:1080px;margin:0 auto;padding:0 20px;
  display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:17px;margin:0;letter-spacing:.2px}
.sub{color:var(--dim);font-size:13px}
.live{margin-left:auto;display:flex;align-items:center;gap:7px;
  color:var(--dim);font-size:12px;font-family:var(--mono)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);
  animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.pending-bar{background:color-mix(in srgb,var(--wait) 14%,transparent);
  border:1px solid var(--wait);border-radius:10px;padding:11px 15px;margin:0 0 22px;
  font-size:14px}
.pending-bar b{color:var(--wait)}
.pending-bar a{color:var(--fg);text-decoration:underline;text-underline-offset:2px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:18px 20px;margin:0 0 18px}
.card.pin{border-color:var(--wait);box-shadow:0 0 0 1px var(--wait) inset}
.card header{display:flex;align-items:center;gap:11px;flex-wrap:wrap;margin-bottom:10px}
.kind{font:600 10.5px/1 var(--mono);letter-spacing:.09em;color:var(--c);
  border:1px solid var(--c);border-radius:5px;padding:4px 7px;white-space:nowrap}
.card h2{font-size:15.5px;margin:0;font-weight:620;flex:1 1 260px}
.status{font:600 10.5px/1 var(--mono);letter-spacing:.07em;padding:4px 7px;
  border-radius:5px;white-space:nowrap}
.status.ok{color:var(--ok);border:1px solid var(--ok)}
.status.bad{color:var(--bad);border:1px solid var(--bad)}
.status.wait{color:var(--wait);border:1px solid var(--wait)}
.ts{color:var(--dim);font:11.5px/1 var(--mono);white-space:nowrap}
.card p{margin:.5em 0}
.why,.asks,.note{background:var(--panel2);border-left:3px solid var(--line);
  border-radius:0 8px 8px 0;padding:10px 14px;margin:12px 0}
.asks{border-left-color:var(--wait)}
.note.yours{border-left-color:var(--ok)}
.lbl{display:block;font:600 10px/1 var(--mono);letter-spacing:.1em;
  text-transform:uppercase;color:var(--dim);margin-bottom:5px}
.why p,.asks p,.note p{margin:.3em 0}
.scroll{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font:12.5px/1.5 var(--mono);width:100%}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}
th{background:var(--panel2);text-align:right;font-weight:600;color:var(--dim)}
td:first-child,th:first-child{text-align:left}
td.ok{color:var(--ok)}td.bad{color:var(--bad)}
.kvs{display:grid;gap:1px;margin:12px 0;background:var(--line);
  border:1px solid var(--line);border-radius:8px;overflow:hidden}
.kv{display:grid;grid-template-columns:minmax(140px,1fr) minmax(90px,auto) 2fr;
  gap:12px;background:var(--panel);padding:7px 12px;align-items:baseline}
@media(max-width:620px){.kv{grid-template-columns:1fr}}
.kv .k{color:var(--dim);font-size:13px}
.kv .v{font:600 13px/1.4 var(--mono)}
.kv .n{color:var(--dim);font-size:12.5px}
.kv .n p{margin:0}
pre.code{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  padding:12px 14px;overflow-x:auto;font:12.5px/1.55 var(--mono);margin:12px 0}
code{background:var(--panel2);border-radius:4px;padding:1px 5px;
  font:.9em var(--mono)}
.cap{color:var(--dim);font-size:12.5px;margin:12px 0 5px}
.cap p{margin:0}
figure{margin:14px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:8px;background:#000}
figcaption{color:var(--dim);font-size:12.5px;margin-top:6px}
figcaption p{margin:0}
.toggle{background:none;border:1px solid var(--line);color:var(--dim);
  border-radius:6px;padding:4px 9px;font:11px var(--mono);cursor:pointer}
.toggle:hover{border-color:var(--ok);color:var(--fg)}
.toggle.pri{border-color:var(--ok);color:var(--ok)}
.reply{margin-top:15px;border-top:1px dashed var(--line);padding-top:12px}
.reply summary{cursor:pointer;font:600 10px/1 var(--mono);letter-spacing:.1em;
  text-transform:uppercase;color:var(--dim);list-style:none;user-select:none}
.reply summary::-webkit-details-marker{display:none}
.reply summary:before{content:"\\25B8  "}
.reply[open] summary:before{content:"\\25BE  "}
.reply summary:hover{color:var(--ok)}
.reply textarea{width:100%;min-height:74px;margin-top:10px;resize:vertical;
  background:var(--panel2);color:var(--fg);border:1px solid var(--line);
  border-radius:8px;padding:10px 13px;font:14.5px/1.6 inherit}
.reply textarea:focus{outline:none;border-color:var(--ok)}
.reply.has summary{color:var(--ok)}
.reply.has textarea{border-color:var(--ok)}
.reply .st{color:var(--dim);font:10.5px var(--mono);margin-top:5px;height:13px}
.dock{position:fixed;right:18px;bottom:18px;z-index:40;display:flex;gap:8px;
  align-items:center;background:var(--panel);border:1px solid var(--line);
  border-radius:11px;padding:9px 12px;box-shadow:0 10px 30px #0007;font:12px var(--mono)}
.dock .n{color:var(--dim)}
.dock .n b{color:var(--ok);font-size:13px}
.dock button{background:var(--panel2);border:1px solid var(--line);color:var(--fg);
  border-radius:7px;padding:6px 10px;font:11.5px var(--mono);cursor:pointer}
.dock button:hover{border-color:var(--ok);color:var(--ok)}
.dock button.pri{border-color:var(--ok);color:var(--ok)}
#fl{position:fixed;right:18px;bottom:66px;z-index:41;background:var(--ok);color:#04140d;
  border-radius:8px;padding:8px 13px;font:12px var(--mono);opacity:0;
  transition:opacity .25s;pointer-events:none;max-width:340px}
@media(max-width:620px){.dock{left:12px;right:12px;flex-wrap:wrap;justify-content:center}}
"""

JS = r"""
// ── the page no longer reloads itself.  Reload is a button; auto is opt-in. ──
history.scrollRestoration = 'manual';
addEventListener('beforeunload', () => sessionStorage.setItem('_y', scrollY));
addEventListener('load', () => {
  const y = sessionStorage.getItem('_y');
  if (y) scrollTo(0, +y);
});

// theme, remembered
const th = localStorage.getItem('_theme');
if (th) document.documentElement.setAttribute('data-theme', th);
const tb = document.getElementById('tt');
if (tb) tb.onclick = () => {
  const r = document.documentElement;
  const cur = r.getAttribute('data-theme') ||
    (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
  const nxt = cur === 'dark' ? 'light' : 'dark';
  r.setAttribute('data-theme', nxt);
  localStorage.setItem('_theme', nxt);
};

const rb = document.getElementById('rl');
if (rb) rb.onclick = () => location.reload();

// opt-in auto-reload, OFF unless you turn it on, and never while you are typing
const ab = document.getElementById('ar');
let timer = null;
function setAuto(on) {
  localStorage.setItem('_auto', on ? '1' : '0');
  if (ab) {
    ab.textContent = on ? ('auto ' + AUTO_S + 's') : 'auto off';
    ab.classList.toggle('pri', on);
  }
  if (timer) { clearInterval(timer); timer = null; }
  if (on) timer = setInterval(() => {
    if (document.querySelector('.reply textarea:focus')) return;   // never mid-sentence
    if (collect().length) return;                                  // never with unsent replies
    location.reload();
  }, AUTO_S * 1000);
}
if (ab) ab.onclick = () => setAuto(localStorage.getItem('_auto') !== '1');

// ── your replies ──
const K = '_reply_';
function boxes() { return [...document.querySelectorAll('textarea[data-eid]')]; }
function collect() {
  return boxes().map(t => ({ id: t.dataset.eid, title: t.dataset.title, v: t.value.trim() }))
                .filter(r => r.v);
}
function refresh() {
  const n = collect().length;
  const c = document.getElementById('cnt');
  if (c) c.textContent = n;
  boxes().forEach(t => t.closest('.reply').classList.toggle('has', !!t.value.trim()));
}
boxes().forEach(t => {
  const k = K + t.dataset.eid;
  const s = localStorage.getItem(k);
  if (s) { t.value = s; t.closest('details').open = true; }
  let tid = null;
  t.addEventListener('input', () => {
    localStorage.setItem(k, t.value);
    refresh();
    const st = t.parentElement.querySelector('.st');
    if (st) {
      st.textContent = 'saved in this browser';
      clearTimeout(tid);
      tid = setTimeout(() => { st.textContent = ''; }, 1800);
    }
  });
});
function asText() {
  const r = collect();
  if (!r.length) return '';
  return r.map(x => '### ' + x.id + '  ' + x.title + '\n' + x.v).join('\n\n');
}
function flash(m) {
  const f = document.getElementById('fl');
  if (!f) return;
  f.textContent = m;
  f.style.opacity = 1;
  setTimeout(() => { f.style.opacity = 0; }, 3000);
}
const cp = document.getElementById('cp');
if (cp) cp.onclick = async () => {
  const t = asText();
  if (!t) { flash('nothing written yet — type in a "your reply" box first'); return; }
  try { await navigator.clipboard.writeText(t); }
  catch (e) {
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove();
  }
  flash('copied — paste it straight into the chat');
};
const sv = document.getElementById('sv');
if (sv) sv.onclick = () => {
  const t = asText();
  if (!t) { flash('nothing written yet'); return; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([t], { type: 'text/markdown' }));
  a.download = 'claude_replies.md';
  a.click();
  flash('saved as claude_replies.md — say "read my replies" and I will pick it up');
};
const cl = document.getElementById('cl');
if (cl) cl.onclick = () => {
  if (!confirm('Clear every reply you have written?')) return;
  boxes().forEach(t => { t.value = ''; localStorage.removeItem(K + t.dataset.eid); });
  refresh();
};
refresh();
setAuto(localStorage.getItem('_auto') === '1');
"""


#: Tools that live beside the log and are linked from its header when present. Each is a
#: page the USER drives rather than one the agent writes: the log is the agent talking, and
#: these are the two ways the user talks back — by labelling, or by drawing the answer.
#: Presence-detected rather than configured, so a tool appears the moment it is built and
#: never leaves a dead link behind.
_TOOLS = [
    ("gt/index.html",    "&#9998; draw ground truth", "--fg",
     "Hand-draw the correct answer over the real image"),
    ("label/index.html", "&#9635; labeller",          "--ok",
     "Label one object at a time, blind"),
]


def _tool_links() -> str:
    out = []
    for rel, label, var, tip in _TOOLS:
        if os.path.exists(os.path.join(ROOT, *rel.split("/"))):
            out.append(f'<a class="toggle" style="text-decoration:none;'
                       f'border-color:var({var});color:var({var})" href="{rel}" '
                       f'title="{html.escape(tip)}">{label}</a>')
    return "\n  ".join(out)


def render() -> str:
    entries = _read()
    pending = [e for e in entries if e.get("status") == "pending"]
    body = list(reversed(entries))                    # newest first
    bar = ""
    if pending:
        links = " · ".join(f'<a href="#{e["id"]}">{html.escape(e["title"])}</a>'
                           for e in pending)
        bar = (f'<div class="pending-bar"><b>{len(pending)} decision'
               f'{"s" if len(pending) > 1 else ""} waiting on you</b> — {links}'
               f'<br><span style="font-size:12.5px;color:var(--dim)">Each card has a '
               f'<b style="color:var(--dim)">your reply</b> box at the bottom. Write in it, '
               f'then hit <b style="color:var(--dim)">copy for Claude</b> at the bottom '
               f'right.</span></div>')
    cards = "".join(_entry(e, pinned=(e.get("status") == "pending")) for e in body)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(TITLE)}</title><style>{CSS}</style></head><body>
<header class="top"><div class="row">
  <h1>{html.escape(TITLE)}</h1>
  <span class="sub">reasoning &amp; evidence, streamed</span>
  <button class="toggle" id="tt">theme</button>
  <button class="toggle" id="rl">&#8635; reload</button>
  <button class="toggle" id="ar">auto off</button>
  {_tool_links()}
  <span class="live">{len(entries)} entries &middot; page built
    {time.strftime("%H:%M:%S")}</span>
</div></header>
<div class="wrap">{bar}{cards}</div>
<div id="fl"></div>
<div class="dock" id="dock">
  <span class="n"><b id="cnt">0</b> replies</span>
  <button class="pri" id="cp">copy for Claude</button>
  <button id="sv">save file</button>
  <button id="cl">clear</button>
</div>
<script>const AUTO_S={REFRESH_S};{JS}</script></body></html>"""
    os.makedirs(IMG, exist_ok=True)
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(page)
    return PAGE


# ───────────────────────── posterity: two exports ─────────────────────────
#
# `render()` writes a page that needs `img/` beside it.  These two do not: each is one
# file you can hand to somebody, attach to a paper, or commit without the originals.

def _data_uri(path: str, max_px: int = 1500, quality: int = 82) -> Optional[str]:
    """Re-encode an image small enough to inline.  Returns None if it cannot be read."""
    try:
        from PIL import Image
    except ImportError:
        try:
            with open(path, "rb") as f:
                return ("data:image/png;base64,"
                        + base64.b64encode(f.read()).decode("ascii"))
        except OSError:
            return None
    try:
        im = Image.open(path)
    except (OSError, ValueError):
        return None
    im = im.convert("RGB")
    if max(im.size) > max_px:
        s = max_px / float(max(im.size))
        im = im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def export_archive(path: Optional[str] = None, *, max_px: int = 1500,
                   quality: int = 82) -> str:
    """One self-contained HTML with every figure inlined and the reply UI stripped.

    This is the artefact for showing someone else why a decision was taken.  It is frozen:
    no reply boxes, no auto-reload, no relative image paths to break when it is moved.
    """
    out = path or os.path.join(ROOT, "evidence_archive.html")
    entries = _read()
    cache: Dict[str, Optional[str]] = {}
    n_img = n_missing = 0
    for e in entries:
        for im in e.get("images", []):
            src = im.get("src", "")
            if src not in cache:
                cache[src] = _data_uri(os.path.join(ROOT, src), max_px, quality)
            if cache[src]:
                im["src"] = cache[src]
                n_img += 1
            else:
                im["src"] = ""
                n_missing += 1
    cards = "".join(_entry(e) for e in reversed(entries))
    counts = {k: sum(1 for e in entries if e.get("kind") == k) for k in KINDS}
    st = {k: sum(1 for e in entries if e.get("status") == k) for k in STATUS}
    summary = (
        f'<div class="pending-bar" style="border-color:var(--line);background:var(--panel2)">'
        f'<b style="color:var(--fg)">{len(entries)} entries</b> &middot; '
        + " &middot; ".join(f"{KINDS[k][0].lower()} {v}" for k, v in counts.items() if v)
        + f' &middot; <b style="color:var(--ok)">{st.get("confirmed", 0)} confirmed</b>'
        f' &middot; <b style="color:var(--bad)">{st.get("rejected", 0)} rejected/retracted</b>'
        f'<br><span style="font-size:12.5px;color:var(--dim)">Frozen record exported '
        f'{time.strftime("%Y-%m-%d %H:%M")}. Newest first. Figures are embedded, so this '
        f'file is complete on its own. Entries are never edited after the fact &mdash; a '
        f'claim that was withdrawn is still here, marked REJECTED, with the reason.</span>'
        f'</div>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(TITLE)} — archived</title><style>{CSS}</style></head><body>
<header class="top"><div class="row">
  <h1>{html.escape(TITLE)}</h1>
  <span class="sub">archived evidence log &mdash; how each decision was reached</span>
  <button class="toggle" id="tt">theme</button>
</div></header>
<div class="wrap">{summary}{cards}</div>
<script>
const th=localStorage.getItem('_theme');
if(th)document.documentElement.setAttribute('data-theme',th);
const b=document.getElementById('tt');
if(b)b.onclick=()=>{{const r=document.documentElement;
  const c=r.getAttribute('data-theme')||
    (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  const n=c==='dark'?'light':'dark';
  r.setAttribute('data-theme',n);localStorage.setItem('_theme',n);}};
</script></body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"archived {len(entries)} entries, {n_img} figures inlined"
          + (f", {n_missing} MISSING" if n_missing else "")
          + f" -> {out} ({os.path.getsize(out) // 1024} KB)")
    return out


def export_markdown(path: Optional[str] = None) -> str:
    """The same record as markdown, with figures left as relative links.

    For committing next to the code, quoting in a plan document, or pasting into a paper's
    methods section.  Keeps every status marker so retractions stay visible.
    """
    out = path or os.path.join(ROOT, "evidence_log.md")
    entries = _read()
    L: List[str] = [f"# {TITLE} — evidence log", "",
                    f"Exported {time.strftime('%Y-%m-%d %H:%M')}. "
                    f"{len(entries)} entries, oldest first. Append-only: withdrawn claims "
                    f"are still here and marked, because the reason a decision changed is "
                    f"part of the record.", ""]
    for e in entries:
        kind = KINDS.get(e.get("kind", "did"), ("?", ""))[0]
        stat = STATUS.get(e.get("status", "info"), ("", ""))[0]
        L.append(f"## `{e['id']}` {e['title']}")
        L.append("")
        L.append(f"*{kind}{' — **' + stat + '**' if stat else ''} · {e['ts']}*")
        L.append("")
        if e.get("body"):
            L += [e["body"], ""]
        if e.get("why"):
            L += ["**Why:** " + e["why"].replace("\n", "\n> "), ""]
        for ev in e.get("evidence", []):
            t = ev.get("type")
            if ev.get("caption"):
                L += ["*" + ev["caption"] + "*", ""]
            if t == "table":
                cols = _cells(ev, "cols", "headers")
                rows = ev.get("rows", [])
                if cols or rows:
                    # A table logged with no header at all still has to reach the page;
                    # dropping it for a missing `cols` key loses the measurement itself.
                    if not cols:
                        cols = [""] * len(rows[0])
                    L.append("| " + " | ".join(map(str, cols)) + " |")
                    L.append("|" + "|".join(["---"] * len(cols)) + "|")
                    for row in rows:
                        L.append("| " + " | ".join(map(str, row)) + " |")
                    L.append("")
            elif t == "kv":
                L.append("| | value | note |")
                L.append("|---|---|---|")
                for row in _cells(ev, "rows", "items"):
                    r = list(row) + [""] * (3 - len(row))
                    L.append("| " + " | ".join(str(x).replace("\n", " ")
                                               for x in r[:3]) + " |")
                L.append("")
            elif t == "code":
                L += ["```", ev.get("text", ""), "```", ""]
            elif ev.get("text"):
                L += [ev["text"], ""]
        for im in e.get("images", []):
            L.append(f"![{im.get('caption', '')[:80]}]({im.get('src', '')})")
            if im.get("caption"):
                L += ["", "*" + im["caption"] + "*"]
            L.append("")
        if e.get("asks"):
            L += ["**What I need from you:** " + e["asks"], ""]
        if e.get("note"):
            L += ["> **Their note:** " + e["note"].replace("\n", "\n> "), ""]
        L += ["---", ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB, {len(entries)} entries)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="rebuild or export an evidence log")
    ap.add_argument("--root", default=None, help="log directory (default CodeLog/live)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--archive", action="store_true",
                    help="write one self-contained HTML with figures inlined")
    ap.add_argument("--markdown", action="store_true", help="write evidence_log.md")
    ap.add_argument("--max-px", type=int, default=1500)
    a = ap.parse_args()
    if a.root or a.title:
        configure(a.root or ROOT, title=a.title)
    if a.archive:
        export_archive(max_px=a.max_px)
    if a.markdown:
        export_markdown()
    if not (a.archive or a.markdown):
        print(render())
