"""Blind labelling sheet — turn the user's eye into ground truth.

WHY THIS EXISTS
    A threshold picked by an agent is a guess. Asking "where should the cut be?" gets a
    guess back. Asking the user to label 585 individual objects gets a *dataset*, and a
    dataset calibrates every candidate rule at once, out-of-sample, and becomes the target
    a synthetic fixture has to reproduce.

TWO RULES THAT MAKE OR BREAK IT
    1. BLIND. A tile shows the raw evidence and nothing else — no measurement, no score,
       no verdict from the agent. If the agent's prediction is visible, the labels anchor
       to it and cannot calibrate anything. Everything measured goes in the manifest
       instead and is joined back afterwards.
    2. STRATIFIED, not sorted. Ordering by the score being calibrated means the first
       hundred tiles are all the easy end and the interesting transition is hundreds of
       tiles away. Round-robin across contiguous bands of that score, so ANY prefix the
       user gets through is a balanced sample and partial work is still usable.

USAGE
    import sys; sys.path.insert(0, ".claude/skills/evidence-log")
    import labelsheet as LS

    # you render the tiles yourself -- only you know what the evidence looks like
    LS.build(
        out_dir="CodeLog/live/label",
        manifest=[{"uid": "M15_0001", "group": "M15", "solidity": 0.93, ...}, ...],
        question="How many granules are in each object?",
        vocab=[("1", "one granule", "#34d399"), ("2", "two", "#fbbf24"),
               ("3", "three or more", "#fb7185"),
               ("x", "not a granule", "#a78bfa"), ("?", "cannot tell", "#64748b")],
        strata_key="solidity",       # the score being calibrated; hidden from the user
        priority=["M15_0658", ...],  # objects already shown -- label these first
        note="Every tile is 400 um across; white bar = 100 um.",
    )
    # tiles are read from out_dir/img/<uid>.png

    rows, notes = LS.read("~/Downloads/labels.csv", manifest)   # join back
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

VOCAB_DEFAULT = [("1", "one", "#34d399"), ("2", "two", "#fbbf24"),
                 ("3", "three or more", "#fb7185"),
                 ("x", "not valid", "#a78bfa"), ("?", "cannot tell", "#64748b")]


def _order(manifest: List[Dict[str, Any]], strata_key: Optional[str],
           priority: Sequence[str], n_strata: int) -> Tuple[List[Dict], List[Dict]]:
    pri = list(priority or [])
    pset = set(pri)
    first = [r for r in manifest if r["uid"] in pset]
    first.sort(key=lambda r: pri.index(r["uid"]))
    rest = [r for r in manifest if r["uid"] not in pset]
    if strata_key:
        rest.sort(key=lambda r: -(r.get(strata_key)
                                  if r.get(strata_key) is not None else -1e18))
        n = len(rest)
        bands = [rest[n * i // n_strata:n * (i + 1) // n_strata]
                 for i in range(n_strata)]
        out, i = [], 0
        while any(bands):
            b = bands[i % n_strata]
            if b:
                out.append(b.pop(0))
            i += 1
        rest = out
    return first, rest


def build(out_dir: str, manifest: List[Dict[str, Any]], *, question: str,
          vocab: Sequence[Tuple[str, str, str]] = tuple(VOCAB_DEFAULT),
          strata_key: Optional[str] = None, priority: Sequence[str] = (),
          priority_note: str = "", note: str = "", group_key: str = "group",
          n_strata: int = 24, csv_name: str = "labels.csv",
          title: Optional[str] = None) -> str:
    """Write out_dir/index.html + out_dir/manifest.json.  Tiles must already be at
    out_dir/img/<uid>.png.  Returns the page path."""
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    missing = [r["uid"] for r in manifest
               if not os.path.exists(os.path.join(out_dir, "img", f"{r['uid']}.png"))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} tiles missing from {out_dir}/img, e.g. {missing[:4]}. "
            f"Render them before calling build().")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    first, rest = _order(manifest, strata_key, priority, n_strata)
    groups = sorted({str(r.get(group_key, "")) for r in manifest} - {""})
    keys = [k for k, _, _ in vocab]

    def tile(r: Dict[str, Any], mark: bool) -> str:
        g = str(r.get(group_key, ""))
        btns = "".join(
            f'<button data-s="{k}" title="{lb}">{k}</button>' for k, lb, _ in vocab)
        return (f'<div class="tile" data-uid="{r["uid"]}" data-grp="{g}">'
                f'<img src="img/{r["uid"]}.png" loading="lazy" alt="">'
                + ('<span class="seen">already shown</span>' if mark else '')
                + f'<div class="id"><span>{r["uid"]}</span><span>{g}</span></div>'
                f'<div class="bar">{btns}</div></div>')

    body = []
    if first:
        body.append(f'<div class="sec hot"><b>{len(first)} objects to do first.</b> '
                    f'{priority_note or "These are the ones already discussed."}</div>')
        body += [tile(r, True) for r in first]
        body.append(f'<div class="sec"><b>Everything else &mdash; {len(rest)} objects.</b> '
                    f'Deliberately shuffled so that however far you get, the sample spans '
                    f'the full range rather than all the easy ones first. Stop whenever '
                    f'you like; partial work is still usable.</div>')
    body += [tile(r, False) for r in rest]

    vocab_css = "\n".join(
        f'.tile[data-v="{k}"]{{border-color:{c}}}\n'
        f'.tile[data-v="{k}"] button[data-s="{k}"]{{background:{c};'
        f'color:#0b0f17;border-color:{c}}}'
        for k, _, c in vocab)
    keychips = "".join(
        f'<span class="k" style="color:{c};border-color:{c}"><b>{k}</b> {lb}</span>'
        for k, lb, c in vocab)
    gbtns = "".join(f'<button data-f="{g}">{g}</button>' for g in groups)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title or question}</title><style>{_CSS}{vocab_css}</style></head><body>
<header class="top">
  <h1>{question}</h1>
  <div class="sub">{note} <b>Nothing measured is shown to you on purpose</b> &mdash; no
    scores, no verdicts. If you could see them your labels would anchor to them and could
    not calibrate anything. Your labels become the ground truth.</div>
  <div class="keys">{keychips}
    <span class="k">arrows move &middot; <b>space</b> skips &middot;
      <b>backspace</b> clears &middot; a label auto-advances</span>
    <button class="k" id="tt" style="cursor:pointer">theme</button>
  </div>
  <div class="filt"><span style="color:var(--dim);font:11px var(--mono)">show</span>
    <button data-f="all" class="on">all {len(manifest)}</button>
    <button data-f="todo">unlabelled only</button>{gbtns}</div>
</header>
<div class="grid">{''.join(body)}</div>
<div id="fl"></div>
<div class="dock">
  <span class="tally" id="tl"></span>
  <span class="notes"><label for="nt">anything else I should know</label>
  <textarea id="nt" placeholder="free text — included in the export"></textarea></span>
  <span style="display:flex;gap:8px;align-items:flex-start">
    <button class="pri" id="cp">copy for Claude</button>
    <button id="sv">save csv</button><button id="cl">clear</button></span>
</div>
<script>const KEYS={json.dumps(keys)};const CSVNAME={json.dumps(csv_name)};
{_JS}</script></body></html>"""
    out = os.path.join(out_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


def read(csv_path: str, manifest: List[Dict[str, Any]]
         ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Join a saved/pasted label file back onto the manifest.

    Returns (rows, notes) where each row is the manifest record plus `label`.  Records
    with no label are omitted, so `len(rows)/len(manifest)` is the coverage.
    """
    by_uid = {r["uid"]: r for r in manifest}
    lab: Dict[str, str] = {}
    notes: List[str] = []
    in_notes = False
    with open(os.path.expanduser(csv_path), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("NOTES"):
                in_notes = True
                continue
            if in_notes:
                notes.append(s)
                continue
            if "," not in s or s.upper().startswith(("LABELS", "GRANULE LABELS", "UID,")):
                continue
            uid, v = s.rsplit(",", 1)
            uid, v = uid.strip(), v.strip()
            if uid in by_uid:
                lab[uid] = v
    rows = [dict(by_uid[u], label=v) for u, v in lab.items()]
    return rows, notes


_CSS = """
*{box-sizing:border-box}
:root{--bg:#0b0f17;--panel:#111827;--line:#1f2937;--fg:#e5e7eb;--dim:#9ca3af;
 --ok:#34d399;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:light){:root{--bg:#f8fafc;--panel:#fff;--line:#e2e8f0;
 --fg:#0f172a;--dim:#52627a}}
:root[data-theme=light]{--bg:#f8fafc;--panel:#fff;--line:#e2e8f0;--fg:#0f172a;--dim:#52627a}
:root[data-theme=dark]{--bg:#0b0f17;--panel:#111827;--line:#1f2937;--fg:#e5e7eb;--dim:#9ca3af}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
header.top{position:sticky;top:0;z-index:20;background:var(--bg);
 border-bottom:1px solid var(--line);padding:13px 20px}
h1{font-size:16px;margin:0 0 5px}
.sub{color:var(--dim);font-size:12.5px;max-width:1100px}
.keys{margin-top:9px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.k{font:11.5px var(--mono);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.k b{font-size:12.5px}
.filt{display:flex;gap:6px;align-items:center;margin-top:8px;flex-wrap:wrap}
.filt button{background:transparent;border:1px solid var(--line);color:var(--dim);
 border-radius:6px;padding:4px 9px;font:11px var(--mono);cursor:pointer}
.filt button.on{border-color:var(--ok);color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(184px,1fr));
 gap:11px;padding:18px 20px 150px}
.sec{grid-column:1/-1;margin:14px 0 2px;padding:9px 13px;border-radius:9px;
 border:1px solid var(--line);background:var(--panel);font:12.5px var(--mono);
 color:var(--dim)}
.sec b{color:var(--fg)}
.sec.hot{border-color:#38bdf8;color:#38bdf8}
.tile{background:var(--panel);border:2px solid var(--line);border-radius:10px;
 padding:7px;cursor:pointer;position:relative}
.tile.cur{border-color:#38bdf8;box-shadow:0 0 0 3px #38bdf844}
.tile img{width:100%;height:auto;display:block;border-radius:6px;background:#000}
.tile .seen{position:absolute;top:11px;left:11px;background:#38bdf8;color:#04121c;
 font:9.5px var(--mono);padding:2px 5px;border-radius:4px}
.tile .id{font:10.5px var(--mono);color:var(--dim);margin-top:5px;display:flex;
 justify-content:space-between}
.tile .bar{display:flex;gap:3px;margin-top:5px}
.tile .bar button{flex:1;background:transparent;border:1px solid var(--line);
 color:var(--dim);border-radius:5px;padding:3px 0;font:11px var(--mono);cursor:pointer}
.tile .bar button:hover{border-color:var(--fg);color:var(--fg)}
.hide{display:none!important}
.dock{position:fixed;left:0;right:0;bottom:0;z-index:30;background:var(--panel);
 border-top:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;
 font:12px var(--mono);flex-wrap:wrap;align-items:flex-start}
.dock .tally{display:flex;gap:11px;flex-wrap:wrap;align-items:center}
.dock .notes{flex:1 1 320px;min-width:240px}
.dock .notes label{display:block;font:10px var(--mono);letter-spacing:.09em;
 text-transform:uppercase;color:var(--dim);margin-bottom:3px}
#nt{width:100%;min-height:52px;background:var(--bg);color:var(--fg);
 border:1px solid var(--line);border-radius:8px;padding:8px 11px;font:13px/1.5 inherit;
 resize:vertical}
#nt:focus{outline:none;border-color:var(--ok)}
.dock button{background:transparent;border:1px solid var(--line);color:var(--fg);
 border-radius:7px;padding:6px 11px;font:11.5px var(--mono);cursor:pointer}
.dock button:hover{border-color:var(--ok);color:var(--ok)}
.dock button.pri{border-color:var(--ok);color:var(--ok)}
#fl{position:fixed;left:20px;bottom:96px;z-index:31;background:var(--ok);color:#04140d;
 border-radius:8px;padding:8px 13px;font:12px var(--mono);opacity:0;
 transition:opacity .25s;pointer-events:none;max-width:60vw}
"""

_JS = r"""
const K='_lbl_'+location.pathname+'_';
const tiles=[...document.querySelectorAll('.tile')];
let cur=0;
function vis(){return tiles.filter(t=>!t.classList.contains('hide'));}
function save(t,v){
  if(v){t.dataset.v=v;localStorage.setItem(K+t.dataset.uid,v);}
  else{delete t.dataset.v;localStorage.removeItem(K+t.dataset.uid);}
  tally();
}
function tally(){
  const c={};KEYS.forEach(k=>c[k]=0);
  tiles.forEach(t=>{if(t.dataset.v!==undefined)c[t.dataset.v]=(c[t.dataset.v]||0)+1;});
  const n=Object.values(c).reduce((a,b)=>a+b,0);
  document.getElementById('tl').innerHTML=`<span><b>${n}</b>/${tiles.length}</span>`+
    KEYS.map(k=>`<span>${k} &rarr; ${c[k]||0}</span>`).join('');
}
tiles.forEach((t,i)=>{
  const v=localStorage.getItem(K+t.dataset.uid);
  if(v)t.dataset.v=v;
  t.addEventListener('click',e=>{setCur(i);const b=e.target.closest('button');
    if(b)save(t,t.dataset.v===b.dataset.s?null:b.dataset.s);});
});
function setCur(i){
  const v=vis();if(!v.length)return;
  tiles.forEach(t=>t.classList.remove('cur'));
  let t=tiles[Math.max(0,Math.min(tiles.length-1,i))];
  if(t.classList.contains('hide'))t=v[0];
  cur=tiles.indexOf(t);t.classList.add('cur');
}
function look(){const r=tiles[cur].getBoundingClientRect();
  if(r.top<150||r.bottom>innerHeight-120)
    tiles[cur].scrollIntoView({block:'center',behavior:'smooth'});}
function advance(){
  const v=vis(),k=v.indexOf(tiles[cur]);
  for(let i=k+1;i<v.length;i++)if(v[i].dataset.v===undefined){
    setCur(tiles.indexOf(v[i]));return look();}
  if(k+1<v.length){setCur(tiles.indexOf(v[k+1]));look();}
}
addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(KEYS.includes(e.key)){save(tiles[cur],e.key);advance();e.preventDefault();return;}
  const cols=Math.max(1,Math.round(document.querySelector('.grid').clientWidth/195));
  const v=vis(),k=v.indexOf(tiles[cur]);
  const d={'ArrowRight':1,'ArrowLeft':-1,'ArrowDown':cols,'ArrowUp':-cols,
           ' ':1,'Backspace':-1}[e.key];
  if(d!==undefined){if(e.key==='Backspace')save(tiles[cur],null);
    setCur(tiles.indexOf(v[Math.max(0,Math.min(v.length-1,k+d))]));look();
    e.preventDefault();}
});
document.querySelectorAll('.filt button').forEach(b=>{
  b.onclick=()=>{const f=b.dataset.f;
    document.querySelectorAll('.filt button').forEach(x=>x.classList.toggle('on',x===b));
    tiles.forEach(t=>t.classList.toggle('hide',
      !(f==='all'||(f==='todo'&&t.dataset.v===undefined)||t.dataset.grp===f)));
    document.querySelectorAll('.sec').forEach(s=>s.classList.toggle('hide',f!=='all'));
    setCur(tiles.indexOf(vis()[0]||tiles[0]));};
});
const nt=document.getElementById('nt');
nt.value=localStorage.getItem(K+'__notes')||'';
nt.addEventListener('input',()=>localStorage.setItem(K+'__notes',nt.value));
function asText(){
  const out=tiles.filter(t=>t.dataset.v!==undefined)
                 .map(t=>`${t.dataset.uid},${t.dataset.v}`);
  if(!out.length&&!nt.value.trim())return '';
  let s='LABELS  uid,label\n'+out.join('\n');
  if(nt.value.trim())s+='\n\nNOTES\n'+nt.value.trim();
  return s;
}
function flash(m){const f=document.getElementById('fl');f.textContent=m;
  f.style.opacity=1;setTimeout(()=>{f.style.opacity=0;},3000);}
document.getElementById('cp').onclick=async()=>{
  const t=asText();if(!t){flash('nothing labelled yet');return;}
  try{await navigator.clipboard.writeText(t);}
  catch(e){const a=document.createElement('textarea');a.value=t;
    document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();}
  flash('copied - paste it into the chat');
};
document.getElementById('sv').onclick=()=>{
  const t=asText();if(!t){flash('nothing labelled yet');return;}
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([t],{type:'text/csv'}));
  a.download=CSVNAME;a.click();
  flash('saved as '+CSVNAME+' - tell me to read it');
};
document.getElementById('cl').onclick=()=>{
  if(!confirm('Clear every label?'))return;tiles.forEach(t=>save(t,null));};
const tb=document.getElementById('tt');
const th=localStorage.getItem('_theme');
if(th)document.documentElement.setAttribute('data-theme',th);
tb.onclick=()=>{const r=document.documentElement;
  const c=r.getAttribute('data-theme')||
    (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  const n=c==='dark'?'light':'dark';
  r.setAttribute('data-theme',n);localStorage.setItem('_theme',n);};
setCur(0);tally();
"""
