/* ND2Studios standalone viewer (2026-11-05) — vanilla JavaScript, no dependencies.
 *
 * The Viewer's controls and arithmetic, rebuilt for a browser: the additive composite
 * (nodelab_v2.viewer.composite_with_clim), the 1 / 99.5-percentile Auto (_auto_clim), the
 * log histogram LUT with its lo / hi handles and gamma dot (HistogramLUT), the M / T / Z
 * strips with a play button and an fps spinner each, the hover readout (_hover_text), the
 * 1-2-5 scale bar (movie_draw._scalebar) and the timestamp (placement.elapsed_text), and the
 * overlays — points, labels, tracks, a scalar wash, a field raster, vectors — in the looks
 * nodelab_v2.overlays declares. ND2V.core holds the pure functions so they can be checked
 * outside a browser; ND2V.mount / ND2V.mountFigure build the DOM.
 */
(function () {
"use strict";
const ND2V = window.ND2V = window.ND2V || {};
ND2V.version = "1";
const GOLDEN = 137.507764;
const BAR_STEPS_UM = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000];
const PRESET_FALLBACK = [["Red", [255, 40, 40]], ["Green", [60, 255, 80]], ["Blue", [70, 120, 255]],
  ["Cyan", [60, 230, 240]], ["Magenta", [255, 70, 220]], ["Yellow", [255, 220, 60]],
  ["Orange", [255, 150, 40]], ["Grey", [210, 214, 222]]];
const INK = "#c9d2de", MUTED = "#6f7c8d", ACCENT = "#6fe3ff", BODY = "#1e242c";

// ───────────────────────────── core: pure functions ─────────────────────────────
const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);
function flatWindow(lo, hi) { if (hi > lo) return [lo, hi]; return lo > 0 ? [0, lo] : [lo, lo + 1]; }
function percentile(sorted, p) {          // numpy's linear interpolation
  const n = sorted.length; if (!n) return NaN;
  const pos = (p / 100) * (n - 1), i = Math.floor(pos), f = pos - i;
  return i + 1 < n ? sorted[i] + f * (sorted[i + 1] - sorted[i]) : sorted[i];
}
function sampleValues(plane, target) {    // a[::step, ::step] in data units, finite only
  const { data, w, h } = plane, scale = plane.scale || 1, off = plane.offset || 0;
  const step = Math.max(1, Math.floor(Math.sqrt(data.length / (target || 200000))));
  const out = [];
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const v = data[y * w + x]; if (plane.nan0 && v === 0) continue; out.push(v * scale + off);
  }
  return Float64Array.from(out).sort();
}
function autoWindow(plane) {              // the Viewer's _auto_clim
  const s = sampleValues(plane, 200000);
  if (!s.length) return [0, 1];
  return flatWindow(percentile(s, 1.0), percentile(s, 99.5));
}
function histogram(plane, vmin, vmax, bins) {   // log-normalised bar heights in [0, 1]
  bins = bins || 256;
  const s = sampleValues(plane, 60000), counts = new Float32Array(bins);
  const span = vmax - vmin || 1;
  for (let i = 0; i < s.length; i++) {
    const k = Math.floor((s[i] - vmin) / span * bins);
    if (k >= 0 && k < bins) counts[k] += 1; else if (s[i] === vmax) counts[bins - 1] += 1;
  }
  let m = 0; for (let i = 0; i < bins; i++) { counts[i] = Math.log1p(counts[i]); if (counts[i] > m) m = counts[i]; }
  if (m > 0) for (let i = 0; i < bins; i++) counts[i] /= m;
  return counts;
}
function composite(out, w, h, layers) {   // composite_with_clim, additive, clipped
  const n = w * h, acc = new Float32Array(n * 3);
  for (const L of layers) {
    const lut = new Float32Array(65536), scale = L.scale || 1, off = L.offset || 0;
    let lo = L.lo, hi = L.hi; if (hi <= lo) hi = lo + 1;
    const g = L.gamma && Math.abs(L.gamma - 1) > 1e-3 ? Math.max(L.gamma, 1e-3) : 0;
    const top = Math.min(65535, Math.max(0, Math.ceil((hi - off) / scale)) + 1);
    for (let c = 0; c <= top; c++) { let t = clamp((c * scale + off - lo) / (hi - lo), 0, 1); if (g) t = Math.pow(t, g); lut[c] = t; }
    for (let c = top + 1; c < 65536; c++) lut[c] = 1;
    const r = L.rgb[0] / 255, gg = L.rgb[1] / 255, b = L.rgb[2] / 255, d = L.data;
    for (let i = 0, j = 0; i < n; i++, j += 3) { const t = lut[d[i]]; acc[j] += t * r; acc[j + 1] += t * gg; acc[j + 2] += t * b; }
  }
  for (let i = 0, j = 0, k = 0; i < n; i++, j += 3, k += 4) {
    out[k] = acc[j] * 255; out[k + 1] = acc[j + 1] * 255; out[k + 2] = acc[j + 2] * 255; out[k + 3] = 255;
  }
}
function elapsedText(seconds, span) {     // placement.elapsed_text
  let t = Number(seconds); span = Math.max(0, Number(span)); const sign = t < 0 ? "-" : ""; t = Math.abs(t);
  const p2 = v => String(v).padStart(2, "0");
  if (span < 90) return `${sign}${t.toFixed(1)} s`;
  if (span < 5400) return `${sign}${p2(Math.floor(t / 60))}:${p2(Math.round(t) % 60)}`;
  const s = Math.round(t);
  if (span < 86400) return `${sign}${p2(Math.floor(s / 3600))}:${p2(Math.floor((s % 3600) / 60))}:${p2(s % 60)}`;
  return `${sign}${Math.floor(s / 86400)}d ${p2(Math.floor((s % 86400) / 3600))}:${p2(Math.floor((s % 3600) / 60))}:${p2(s % 60)}`;
}
function fmtIntensity(v) { return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4))); }
function hsvRgb(hue, sat, val) {          // label_paint._hues_to_rgb
  const hh = ((hue / 60) % 6 + 6) % 6, i = Math.floor(hh), f = hh - i, v = val;
  const p = v * (1 - sat), q = v * (1 - sat * f), t = v * (1 - sat * (1 - f));
  const rgb = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i] || [v, p, q];
  return rgb.map(c => Math.round(clamp(c * 255, 0, 255)));
}
function slotColor(slot, sat255, val255) { return hsvRgb((slot * GOLDEN) % 360, (sat255 == null ? 205 : sat255) / 255, (val255 == null ? 255 : val255) / 255); }
function scalebar(widthPx, pxUm) {        // movie_draw._scalebar
  if (!pxUm || !(pxUm > 0)) return null;
  const target = 0.18 * widthPx * pxUm, fit = BAR_STEPS_UM.filter(s => s <= target);
  if (!fit.length) return null;
  const um = fit[fit.length - 1], q = um / pxUm, fl = Math.floor(q);
  const px = (q - fl === 0.5) ? (fl % 2 === 0 ? fl : fl + 1) : Math.round(q);   // Python's round: half to even
  if (px < 4 || px > widthPx * 0.9) return null;
  return { px, um, label: um < 1000 ? `${um} µm` : `${um / 1000} mm` };
}
function hexRgb(s) {
  const t = String(s || "").trim(); const m = /^#?([0-9a-f]{6})/i.exec(t);
  if (!m) return [255, 255, 255];
  const v = parseInt(m[1], 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
const rgbCss = (rgb, a) => a == null ? `rgb(${rgb[0]},${rgb[1]},${rgb[2]})` : `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
function Scale(kind, lo, hi, m0, m1) {    // figure.charts.Scale (mm ↔ data)
  const log = kind === "log";
  const a = log ? Math.log10(Math.max(lo, 1e-300)) : lo, b = log ? Math.log10(Math.max(hi, 1e-300)) : hi;
  return { fromMm: mm => { const f = (mm - m0) / ((m1 - m0) || 1); const v = a + f * (b - a); return log ? Math.pow(10, v) : v; } };
}
ND2V.core = { flatWindow, percentile, autoWindow, histogram, composite, elapsedText, fmtIntensity,
  hsvRgb, slotColor, scalebar, hexRgb, sampleValues, Scale };

// ───────────────────────────── plane store ─────────────────────────────
function decodeRecord(rec) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const w = img.naturalWidth, H = img.naturalHeight, u16 = rec.enc === "u16", h = u16 ? H / 2 : H;
        const cv = document.createElement("canvas"); cv.width = w; cv.height = H;
        const ctx = cv.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const d = ctx.getImageData(0, 0, w, H).data, n = w * h, out = new Uint16Array(n);
        // a 16-bit plane is two stacked grey halves: the high bytes above the low bytes
        if (u16) for (let i = 0, k = 0, lo = n * 4; i < n; i++, k += 4, lo += 4) out[i] = d[k] * 256 + d[lo];
        else for (let i = 0, k = 0; i < n; i++, k += 4) out[i] = d[k];
        resolve({ data: out, w, h, scale: rec.scale || 1, offset: rec.offset || 0, nan0: !!rec.nan0, extent: rec.extent, lattice: rec.lattice });
      } catch (e) { reject(e); }
    };
    img.onerror = () => reject(new Error("plane did not decode"));
    img.src = rec.png;
  });
}
class PlaneStore {
  constructor(budget) { this.budget = budget || 320e6; this.map = new Map(); this.bytes = 0; this.pending = new Map(); }
  get(key, rec) {
    const have = this.map.get(key);
    if (have) { this.map.delete(key); this.map.set(key, have); return Promise.resolve(have); }
    if (this.pending.has(key)) return this.pending.get(key);
    const p = decodeRecord(rec).then(pl => {
      this.pending.delete(key); this.map.set(key, pl); this.bytes += pl.data.byteLength;
      while (this.bytes > this.budget && this.map.size > 1) { const k = this.map.keys().next().value; this.bytes -= this.map.get(k).data.byteLength; this.map.delete(k); }
      return pl;
    }, e => { this.pending.delete(key); throw e; });
    this.pending.set(key, p); return p;
  }
}

// ───────────────────────────── DOM helpers ─────────────────────────────
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) { if (k === "class") e.className = attrs[k]; else if (k === "style") e.style.cssText = attrs[k]; else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]); else if (k === "text") e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]); }
  for (const kid of kids) if (kid != null) e.append(kid);
  return e;
}
function btn(label, onclick, cls) { return h("button", { class: "nd2v-btn" + (cls ? " " + cls : ""), onclick, text: label }); }
function numInput(value, step, onchange, cls) {
  const i = h("input", { class: "nd2v-num" + (cls ? " " + cls : ""), type: "number", step: step || "any", value });
  i.addEventListener("change", () => onchange(Number(i.value))); return i;
}
function deep(o) { return JSON.parse(JSON.stringify(o)); }

// ───────────────────────────── histogram LUT ─────────────────────────────
class HistLUT {
  constructor(onWindow, onGamma) {
    this.cv = h("canvas", { class: "nd2v-hist", title: "Drag the handles to set the LUT · drag the middle dot up/down for gamma · wheel to zoom · double-click to reset the zoom" });
    this.rmin = 0; this.rmax = 1; this.vmin = 0; this.vmax = 1; this.lo = 0; this.hi = 1; this.gamma = 1; this.tint = [120, 170, 255];
    this.hist = null; this.plane = null; this.drag = null; this.onWindow = onWindow; this.onGamma = onGamma;
    this.cv.addEventListener("pointerdown", e => this.down(e));
    this.cv.addEventListener("pointermove", e => { if (this.drag) this.apply(e); });
    this.cv.addEventListener("pointerup", () => { this.drag = null; });
    this.cv.addEventListener("dblclick", () => { this.vmin = this.rmin; this.vmax = this.rmax; this.rebin(); });
    this.cv.addEventListener("wheel", e => this.wheel(e), { passive: false });
    this.cv.addEventListener("contextmenu", e => e.preventDefault());
  }
  setRange(rmin, rmax) { rmax = rmax > rmin ? rmax : rmin + 1; if (rmin !== this.rmin || rmax !== this.rmax) { this.rmin = rmin; this.rmax = rmax; this.vmin = rmin; this.vmax = rmax; } this.paint(); }
  setWindow(lo, hi) { this.lo = lo; this.hi = hi; this.paint(); }
  setGamma(g) { this.gamma = Math.max(0.01, g); this.paint(); }
  setTint(rgb) { this.tint = rgb; this.paint(); }
  setValues(plane) { this.plane = plane; this.rebin(); }
  rebin() { this.hist = this.plane ? histogram(this.plane, this.vmin, this.vmax, 256) : null; this.paint(); }
  fitView() { const lo = Math.min(this.lo, this.hi), hi = Math.max(this.lo, this.hi); const pad = Math.max((hi - lo) * 0.15, (this.rmax - this.rmin) * 1e-3);
    this.vmin = Math.max(this.rmin, lo - pad); this.vmax = Math.min(this.rmax, hi + pad); if (this.vmax - this.vmin < 1e-9) { this.vmin = this.rmin; this.vmax = this.rmax; } this.rebin(); }
  v2x(v) { return (v - this.vmin) / (this.vmax - this.vmin) * this.cv.clientWidth; }
  x2v(x) { return this.vmin + clamp(x / Math.max(1, this.cv.clientWidth), 0, 1) * (this.vmax - this.vmin); }
  gammaDot() { const xlo = this.v2x(this.lo), xhi = this.v2x(this.hi), H = this.cv.clientHeight; return [xlo + 0.5 * (xhi - xlo), (H - 1) - Math.pow(0.5, this.gamma) * (H - 2)]; }
  down(e) {
    const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const xlo = this.v2x(this.lo), xhi = this.v2x(this.hi), [mx, my] = this.gammaDot();
    if ((x - mx) ** 2 + (y - my) ** 2 <= 64) this.drag = "gamma";
    else if (Math.abs(x - xlo) <= 6) this.drag = "lo"; else if (Math.abs(x - xhi) <= 6) this.drag = "hi";
    else if (xlo < x && x < xhi) this.drag = "band"; else this.drag = Math.abs(x - xlo) < Math.abs(x - xhi) ? "lo" : "hi";
    this.x0 = x; this.lo0 = this.lo; this.hi0 = this.hi; this.cv.setPointerCapture(e.pointerId); this.apply(e); e.preventDefault();
  }
  apply(e) {
    const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, eps = (this.vmax - this.vmin) * 1e-3;
    if (this.drag === "gamma") { const bot = this.cv.clientHeight - 1; const m = clamp((bot - y) / Math.max(bot, 1), 0.02, 0.98); this.gamma = clamp(Math.log(m) / Math.log(0.5), 0.1, 10); this.paint(); this.onGamma(this.gamma); return; }
    if (this.drag === "band") { const dv = this.x2v(x) - this.x2v(this.x0), span = this.hi0 - this.lo0; const lo = Math.min(Math.max(this.rmin, this.lo0 + dv), this.rmax - span); this.lo = lo; this.hi = lo + span; }
    else if (this.drag === "lo") this.lo = Math.min(this.x2v(x), this.hi - eps);
    else if (this.drag === "hi") this.hi = Math.max(this.x2v(x), this.lo + eps); else return;
    this.paint(); this.onWindow(this.lo, this.hi);
  }
  wheel(e) {
    e.preventDefault(); const r = this.cv.getBoundingClientRect(); const factor = e.deltaY < 0 ? 0.8 : 1.25; const center = this.x2v(e.clientX - r.left);
    const full = this.rmax - this.rmin; let span = clamp((this.vmax - this.vmin) * factor, full * 1e-3, full);
    const frac = (center - this.vmin) / Math.max(this.vmax - this.vmin, 1e-9); let vmin = center - frac * span, vmax = vmin + span;
    if (vmin < this.rmin) { vmin = this.rmin; vmax = vmin + span; } if (vmax > this.rmax) { vmax = this.rmax; vmin = vmax - span; }
    this.vmin = Math.max(this.rmin, vmin); this.vmax = Math.min(this.rmax, vmax); this.rebin();
  }
  paint() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 120, H = cv.clientHeight || 56;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = BODY; ctx.fillRect(0, 0, W, H);
    const [r, g, b] = this.tint;
    if (this.hist) { const n = this.hist.length, bw = W / n; ctx.fillStyle = rgbCss(this.tint, 130 / 255); for (let i = 0; i < n; i++) { const bh = this.hist[i] * (H - 2); ctx.fillRect(i * bw, H - bh, bw + 0.6, bh); } }
    const xlo = this.v2x(this.lo), xhi = this.v2x(this.hi);
    ctx.fillStyle = "rgba(255,255,255,0.086)"; ctx.fillRect(xlo, 0, Math.max(1, xhi - xlo), H);
    ctx.strokeStyle = rgbCss(this.tint); ctx.lineWidth = 1.6; ctx.beginPath();
    for (let k = 0; k <= 40; k++) { const tt = k / 40, out = Math.pow(tt, this.gamma); const x = xlo + tt * (xhi - xlo), y = (H - 1) - out * (H - 2); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(xlo, 0); ctx.lineTo(xlo, H); ctx.moveTo(xhi, 0); ctx.lineTo(xhi, H); ctx.stroke();
    ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(xlo, H - 3, 3, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(xhi, 3, 3, 0, 7); ctx.fill();
    const [mx, my] = this.gammaDot(); ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(mx, my, 4.5, 0, 7); ctx.fill(); ctx.stroke();
  }
}

// ───────────────────────────── a frame strip ─────────────────────────────
class Strip {
  constructor(axis, frames, viewer, compact) {
    this.axis = axis; this.frames = frames; this.viewer = viewer; this.playing = false; this.timer = null; this.fps = 10; this.ticks = []; this.shown = 0;
    this.cv = h("canvas", { class: "nd2v-strip", tabindex: "0" });
    this.play = btn("▶", () => this.toggle(), "play small");
    this.fpsIn = numInput(10, 1, v => { this.fps = Math.max(0.5, v || 10); if (this.playing) this.start(); });
    this.fpsIn.title = "target frames per second for ▶";
    this.read = h("span", { class: "nd2v-readout" });
    this.el = h("div", { class: "nd2v-axis" }, h("span", { class: "ax", text: axis.toUpperCase() }), this.play, this.cv, h("div", { class: "fps" }, this.fpsIn, h("span", { text: "fps" })), this.read);
    if (compact) { this.el.querySelector(".fps").style.display = "none"; }
    const scrub = e => { const r = this.cv.getBoundingClientRect(); const f = clamp((e.clientX - r.left) / Math.max(1, r.width), 0, 0.999999); this.set(Math.floor(f * frames.length)); };
    this.cv.addEventListener("pointerdown", e => { this.cv.setPointerCapture(e.pointerId); this.drag = true; scrub(e); e.preventDefault(); });
    this.cv.addEventListener("pointermove", e => { if (this.drag) scrub(e); else this.hoverTip(e); });
    this.cv.addEventListener("pointerup", () => { this.drag = false; });
    this.cv.addEventListener("wheel", e => { e.preventDefault(); this.step(e.deltaY > 0 ? 1 : -1); }, { passive: false });
    this.cv.addEventListener("keydown", e => { if (e.key === "ArrowRight") { this.step(1); e.preventDefault(); } if (e.key === "ArrowLeft") { this.step(-1); e.preventDefault(); } });
  }
  index() { return Math.max(0, this.frames.indexOf(this.viewer.get(this.axis))); }
  set(i) { i = clamp(i, 0, this.frames.length - 1); this.viewer.setAxis(this.axis, this.frames[i]); }
  step(d) { this.set(this.index() + d); }
  toggle() { if (this.playing) this.stop(); else { this.viewer.stopAll(); this.start(); } }
  start() { this.stop(); this.playing = true; this.play.textContent = "⏸"; this.play.classList.add("on"); this.ticks = []; this.timer = setInterval(() => { this.ticks.push(performance.now()); if (this.ticks.length > 20) this.ticks.shift(); this.set((this.index() + 1) % this.frames.length); this.paint(); }, Math.max(1, 1000 / this.fps)); }
  stop() { if (this.timer) clearInterval(this.timer); this.timer = null; this.playing = false; this.play.textContent = "▶"; this.play.classList.remove("on"); this.paint(); }
  hoverTip(e) {
    if (this.axis !== "t") { this.cv.title = ""; return; }
    const secs = this.viewer.doc.times && this.viewer.doc.times.seconds; if (!secs) { this.cv.title = ""; return; }
    const r = this.cv.getBoundingClientRect(); const i = clamp(Math.floor((e.clientX - r.left) / Math.max(1, r.width) * this.frames.length), 0, this.frames.length - 1); const t = this.frames[i];
    const span = secs[secs.length - 1] - secs[0]; let s = `t ${t + 1}: ${elapsedText(secs[t] - secs[0], span)} after frame 0`;
    if (t > 0) s += ` · ${(secs[t] - secs[t - 1]).toFixed(2)} s after the frame before`;
    if (secs.length > 1) s += ` · mean gap ${(span / (secs.length - 1)).toFixed(2)} s`;
    this.cv.title = s;
  }
  paint() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 100, H = cv.clientHeight || 22;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const n = this.frames.length, cur = this.index(); const bw = Math.min(16, (W - 2) / n); const boxes = bw >= 3;
    ctx.fillStyle = "#2b323c"; ctx.fillRect(0, 0, W, H);
    if (boxes) { for (let i = 0; i < n; i++) { ctx.fillStyle = i === cur ? ACCENT : "#39424f"; ctx.fillRect(1 + i * bw, 3, Math.max(1, bw - 1), H - 6); } }
    else { ctx.fillStyle = "#39424f"; ctx.fillRect(1, 3, W - 2, H - 6); ctx.fillStyle = ACCENT; ctx.fillRect(1 + cur / n * (W - 2), 2, Math.max(2, (W - 2) / n), H - 4); }
    let rate = ""; if (this.playing && this.ticks.length > 2) { const d = (this.ticks[this.ticks.length - 1] - this.ticks[0]) / (this.ticks.length - 1); rate = ` · ${(1000 / d).toFixed(1)} fps`; }
    this.read.textContent = `${this.axis} ${this.frames[cur] + 1}/${this.viewer.doc.axes[this.axis]}${rate}`;
  }
}

// ───────────────────────────── the viewer ─────────────────────────────
class Viewer {
  constructor(el, doc, opts) {
    this.el = el; this.doc = doc; this.opts = opts || {}; this.compact = !!this.opts.compact;
    this.store = this.opts.store || new PlaneStore(this.opts.cacheBytes || 320e6);
    this.m = doc.start.m; this.t = doc.start.t; this.z = String(doc.start.z);
    this.chans = doc.channels.map(c => Object.assign({}, c, { userWindow: !c.auto }));
    this.autoOn = this.chans.some(c => c.auto);
    this.split = false; this.look = deep(doc.overlays.look || {}); this.ov = doc.overlays || {};
    for (const tab of ["points", "labels", "tracks", "voxels", "field", "vectors"]) if (!this.ov[tab] && this.look[tab]) this.look[tab].enabled = false;
    this.zoom = 1; this.ox = 0; this.oy = 0; this.userView = false; this.gen = 0; this.planes = {}; this.labelCache = new Map(); this.rgbaCache = new Map();
    this.pointIndex = null; this.tracksByM = null; this.hoverPt = null; this.hists = {};
    this.build(); this.indexOverlays();
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this.stage);
    this.resize(); this.render();
  }
  // ── state ──
  get(axis) { return axis === "m" ? this.m : axis === "t" ? this.t : this.zIndex(); }
  zIndex() { const v = parseInt(this.z, 10); return Number.isFinite(v) ? v : 0; }
  setAxis(axis, value, silent) {
    if (axis === "m") this.m = value; else if (axis === "t") this.t = value; else this.z = String(value);
    for (const s of this.strips) s.paint();
    this.render();
    if (!silent && this.opts.onFrame) this.opts.onFrame(this, axis, value);
  }
  setFrame(f) { let changed = false; if (f.m != null && this.doc.shown.m.includes(f.m) && f.m !== this.m) { this.m = f.m; changed = true; } if (f.t != null && this.doc.shown.t.includes(f.t) && f.t !== this.t) { this.t = f.t; changed = true; } if (f.z != null && this.doc.shown.z.includes(String(f.z)) && String(f.z) !== this.z) { this.z = String(f.z); changed = true; } if (changed) { for (const s of this.strips) s.paint(); this.render(); } }
  stopAll() { for (const s of this.strips) s.stop(); }
  frameKey(c) { return `${this.m},${this.t},${this.z},${c}`; }
  ovKey() { return `${this.m},${this.t},${this.z}`; }
  // ── DOM ──
  build() {
    const doc = this.doc, el = this.el; el.classList.add("nd2v"); if (this.compact) el.classList.add("compact");
    this.canvas = h("canvas"); this.stage = h("div", { class: "nd2v-stage", tabindex: "0" }, this.canvas);
    this.hover = h("div", { class: "nd2v-hover hint", text: "hover the image to read a pixel" });
    this.status = h("div", { class: "nd2v-status" });
    this.main = h("div", { class: "nd2v-main" }, this.stage, this.hover, this.status);
    this.side = h("div", { class: "nd2v-side" });
    el.append(this.main, this.side);
    this.buf = document.createElement("canvas");
    // playback
    this.strips = [];
    const pb = h("div", { class: "nd2v-section" }, h("h4", { text: "Playback" }));
    for (const ax of ["m", "t", "z"]) { const frames = ax === "z" ? doc.shown.z.map(v => parseInt(v, 10)).filter(Number.isFinite) : doc.shown[ax]; if (frames.length > 1) { const s = new Strip(ax, frames, this, false); this.strips.push(s); pb.append(s.el); } }
    if (this.strips.length) this.side.append(pb);
    if (this.compact && this.strips.length) { this.mini = h("div", { class: "nd2v-ministrip" }); for (const s of this.strips) { const s2 = new Strip(s.axis, s.frames, this, true); this.strips.push(s2); this.mini.append(s2.el); } this.stage.append(this.mini); }
    // channels
    const ch = h("div", { class: "nd2v-section" }, h("h4", { text: "Channels" }));
    this.autoBtn = btn("Auto", () => { this.autoOn = !this.autoOn; this.autoBtn.classList.toggle("on", this.autoOn); if (this.autoOn) for (const c of this.chans) c.userWindow = false; this.render(); }, this.autoOn ? "on" : "");
    this.autoBtn.title = "Auto: recompute the 1 / 99.5-percentile window for every frame";
    this.fitBtn = btn("Fit", () => { for (const k in this.hists) this.hists[k].fitView(); }); this.fitBtn.title = "zoom every histogram to its window";
    this.splitBtn = btn("⊞ Split", () => { this.split = !this.split; this.splitBtn.classList.toggle("on", this.split); this.draw(); }); this.splitBtn.title = "split-channel view: the composite plus one pane per active channel, one shared zoom";
    this.ovBtn = btn("◈ Overlays", () => { this.ovBox.classList.toggle("open"); }); this.ovBtn.title = "overlay settings";
    ch.append(h("div", { class: "nd2v-chan-tools" }, this.autoBtn, this.fitBtn, this.splitBtn, this.ovBtn));
    this.cols = h("div", { class: "nd2v-cols" }); ch.append(this.cols);
    for (const c of this.chans) this.cols.append(this.channelColumn(c));
    this.side.append(ch);
    this.ovBox = h("div", { class: "nd2v-section nd2v-ov" }, h("h4", { text: "Overlays" })); this.buildOverlayBox(); this.side.append(this.ovBox);
    // decor + tools
    const tools = h("div", { class: "nd2v-section" }, h("h4", { text: "View" }));
    const row = h("div", { class: "nd2v-row" });
    row.append(btn("Fit", () => this.fit(true)), btn("1:1", () => this.one(), ""), btn("Save PNG", () => this.savePng()));
    const sb = h("input", { type: "checkbox" }); sb.checked = !!doc.scalebar.show; sb.addEventListener("change", () => { doc.scalebar.show = sb.checked; this.draw(); });
    row.append(h("label", {}, sb, " scale bar"));
    const ts = h("select", { class: "nd2v-num" }); for (const [v, l] of [["elapsed", "elapsed time"], ["clock", "clock"], ["frame", "frame"], ["elapsed_clock", "elapsed + clock"], ["off", "no timestamp"]]) ts.append(h("option", { value: v, text: l }));
    ts.value = doc.timestamp.mode || "off"; ts.addEventListener("change", () => { doc.timestamp.mode = ts.value; this.draw(); }); row.append(ts);
    tools.append(row, h("div", { class: "nd2v-note", text: "keys on the image: ← → T · ↑ ↓ Z · PgUp PgDn M · space play · Home fit · 0 one-to-one · S split · wheel zoom · drag pan" }));
    this.side.append(tools);
    if (this.compact) { this.gear = btn("⚙", () => el.classList.toggle("controls"), "nd2v-gear"); this.gear.title = "controls"; el.append(this.gear); this.figRead = h("div", { class: "nd2v-fig-readout" }); el.append(this.figRead); }
    // stage interaction
    const st = this.stage;
    st.addEventListener("wheel", e => { e.preventDefault(); const r = st.getBoundingClientRect(); this.zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15); }, { passive: false });
    st.addEventListener("pointerdown", e => { if (e.button !== 0) return; st.setPointerCapture(e.pointerId); this.dragging = { x: e.clientX, y: e.clientY, ox: this.ox, oy: this.oy, moved: false }; st.classList.add("dragging"); st.focus(); });
    st.addEventListener("pointermove", e => { const r = st.getBoundingClientRect(); if (this.dragging) { const dx = e.clientX - this.dragging.x, dy = e.clientY - this.dragging.y; if (Math.abs(dx) + Math.abs(dy) > 2) this.dragging.moved = true; this.ox = this.dragging.ox + dx; this.oy = this.dragging.oy + dy; this.userView = true; this.draw(); } this.onHover(e.clientX - r.left, e.clientY - r.top); });
    st.addEventListener("pointerup", () => { this.dragging = null; st.classList.remove("dragging"); });
    st.addEventListener("pointerleave", () => { this.hoverPt = null; this.hover.textContent = "hover the image to read a pixel"; this.hover.classList.add("hint"); if (this.figRead) this.figRead.textContent = ""; });
    st.addEventListener("dblclick", () => this.fit(true));
    st.addEventListener("keydown", e => this.onKey(e));
  }
  channelColumn(c) {
    const col = h("div", { class: "nd2v-col" });
    const b = h("button", { class: "nd2v-chbtn", text: c.name, title: c.emission_nm ? `${c.name} · emission ${c.emission_nm} nm · right-click for the colour` : `${c.name} · right-click for the colour` });
    const style = () => { b.style.background = rgbCss(c.rgb, 0.22); b.style.borderColor = rgbCss(c.rgb); b.style.color = rgbCss(c.rgb); b.classList.toggle("off", !c.on); };
    b.addEventListener("click", () => { c.on = !c.on; style(); this.render(); });
    b.addEventListener("contextmenu", e => { e.preventDefault(); this.colorMenu(c, e.clientX, e.clientY, style); });
    const hist = new HistLUT((lo, hi) => { c.lo = lo; c.hi = hi; c.userWindow = true; loIn.value = fmtIntensity(lo); hiIn.value = fmtIntensity(hi); this.render(); }, g => { c.gamma = g; this.render(); });
    hist.setRange(c.range[0], c.range[1]); hist.setWindow(c.lo, c.hi); hist.setGamma(c.gamma); hist.setTint(c.rgb); this.hists[c.c] = hist;
    const loIn = numInput(fmtIntensity(c.lo), "any", v => { c.lo = v; c.userWindow = true; hist.setWindow(c.lo, c.hi); this.render(); });
    const hiIn = numInput(fmtIntensity(c.hi), "any", v => { c.hi = v; c.userWindow = true; hist.setWindow(c.lo, c.hi); this.render(); });
    loIn.title = "black point (lo)"; hiIn.title = "white point (hi)"; c._inputs = [loIn, hiIn]; c._style = style; style();
    col.append(b, hist.cv, h("div", { class: "nd2v-lohi" }, loIn, hiIn));
    return col;
  }
  colorMenu(c, x, y, restyle) {
    document.querySelectorAll(".nd2v-menu").forEach(m => m.remove());
    const menu = h("div", { class: "nd2v-menu", style: `left:${x}px;top:${y}px` }, h("div", { class: "head", text: `${c.name} colour` }));
    const pick = rgb => { c.rgb = rgb; restyle(); this.hists[c.c].setTint(rgb); menu.remove(); this.render(); };
    for (const p of (this.doc.presets || PRESET_FALLBACK.map(([name, rgb]) => ({ name, rgb })))) menu.append(h("div", { onclick: () => pick(p.rgb) }, h("span", { class: "sw", style: `background:${rgbCss(p.rgb)}` }), p.name));
    const custom = h("input", { type: "color", value: `#${c.rgb.map(v => v.toString(16).padStart(2, "0")).join("")}` }); custom.addEventListener("input", () => pick(hexRgb(custom.value)));
    menu.append(h("div", {}, custom, "Custom colour…"));
    const emis = this.doc.channels.find(d => d.c === c.c); if (emis) menu.append(h("div", { onclick: () => pick(emis.rgb.slice()) }, h("span", { class: "sw", style: `background:${rgbCss(emis.rgb)}` }), "Reset to emission colour"));
    document.body.append(menu);
    const close = e => { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("pointerdown", close, true); } };
    setTimeout(() => document.addEventListener("pointerdown", close, true), 0);
  }
  buildOverlayBox() {
    const box = this.ovBox, ov = this.ov, tabs = h("div", { class: "nd2v-tabs" }), bodies = [];
    const avail = ["points", "labels", "tracks", "voxels", "field", "vectors"].filter(t => ov[t]);
    if (!avail.length) { box.append(h("div", { class: "nd2v-note", text: "this result carries no overlay" })); }
    const show = name => { for (const [n, b, t] of bodies) { b.classList.toggle("on", n === name); t.classList.toggle("on", n === name); } };
    for (const tab of avail) {
      const look = this.look[tab], body = h("div", { class: "nd2v-tabbody" }), tb = btn(tab[0].toUpperCase() + tab.slice(1), () => show(tab), "nd2v-tab small");
      tabs.append(tb); bodies.push([tab, body, tb]);
      const en = h("input", { type: "checkbox" }); en.checked = !!look.enabled; en.addEventListener("change", () => { look.enabled = en.checked; this.draw(); });
      body.append(h("div", { class: "nd2v-row" }, h("label", {}, en, ` draw ${tab}`), h("span", { class: "sp" }), h("label", { text: "opacity" }), this.slider(look, "opacity", 0, 100, 1)));
      const sel = (key, options) => { const s = h("select", { class: "nd2v-num" }); for (const o of options) s.append(h("option", { value: o, text: o })); s.value = look[key]; s.addEventListener("change", () => { look[key] = s.value; this.invalidate(tab); this.draw(); }); return s; };
      const row = (...kids) => body.append(h("div", { class: "nd2v-row" }, ...kids));
      if (tab === "points") { row(h("label", { text: "colour by" }), sel("color_mode", ["single", "per_point", "per_z", "per_layer"]), h("label", { text: "size" }), this.slider(look, "unit_px", 1, 8, 0.5)); const zp = h("input", { type: "checkbox" }); zp.checked = !!look.z_project; zp.addEventListener("change", () => { look.z_project = zp.checked; this.draw(); }); row(h("label", {}, zp, " show points from every Z"), h("label", { text: "off-plane opacity" }), this.slider(look, "off_opacity", 0, 100, 1)); if (ov.points.layers) body.append(h("div", { class: "nd2v-note", text: "layers: " + ov.points.layers.join(", ") })); }
      if (tab === "labels") { row(h("label", { text: "style" }), sel("style", ["outline", "fill", "both"]), h("label", { text: "width" }), this.slider(look, "width", 0.5, 6, 0.5)); const ids = h("input", { type: "checkbox" }); ids.checked = !!look.show_ids; ids.addEventListener("change", () => { look.show_ids = ids.checked; this.draw(); }); row(h("label", {}, ids, " ids"), h("label", { text: "fill opacity" }), this.slider(look, "fill_opacity", 0, 100, 1, () => this.invalidate("labels")), h("label", { text: "colour" }), sel("color_mode", ["per_label", "single"])); body.append(h("div", { class: "nd2v-note", text: `raster: ${ov.labels.layer || "?"}` })); }
      if (tab === "tracks") { row(h("label", { text: "trail" }), sel("trail", ["all", "past", "window"]), h("label", { text: "window" }), this.slider(look, "window", 1, 50, 1), h("label", { text: "width" }), this.slider(look, "width", 0.5, 6, 0.5)); const fd = h("input", { type: "checkbox" }); fd.checked = !!look.fade; fd.addEventListener("change", () => { look.fade = fd.checked; this.draw(); }); row(h("label", {}, fd, " fade with age"), h("label", { text: "colour" }), sel("color_mode", ["per_track", "single"])); }
      if (tab === "voxels") { row(h("label", { text: "colour map" }), sel("colormap", Object.keys(this.doc.cmaps || { viridis: 1 })), h("label", { text: "alpha" }), sel("alpha_mode", ["ramp", "flat", "gated"])); const au = h("input", { type: "checkbox" }); au.checked = !!look.clim_auto; const lo = numInput(look.clim_lo, "any", v => { look.clim_lo = v; look.clim_auto = false; au.checked = false; this.invalidate("voxels"); this.draw(); }); const hi = numInput(look.clim_hi, "any", v => { look.clim_hi = v; look.clim_auto = false; au.checked = false; this.invalidate("voxels"); this.draw(); }); au.addEventListener("change", () => { look.clim_auto = au.checked; this.invalidate("voxels"); this.draw(); }); row(h("label", {}, au, " limits from the data"), lo, hi); }
      if (tab === "field") { row(h("label", { text: "style" }), sel("style", ["smooth", "cells"]), h("label", { text: "colour map" }), sel("colormap", ["auto"].concat(Object.keys(this.doc.cmaps || {})))); const lim = sel("limits", ["series", "frame", "manual"]); const lo = numInput(look.clim_lo, "any", v => { look.clim_lo = v; look.limits = "manual"; lim.value = "manual"; this.invalidate("field"); this.draw(); }); const hi = numInput(look.clim_hi, "any", v => { look.clim_hi = v; look.limits = "manual"; lim.value = "manual"; this.invalidate("field"); this.draw(); }); row(h("label", { text: "limits" }), lim, lo, hi); const cb = h("input", { type: "checkbox" }); cb.checked = !!look.colorbar; cb.addEventListener("change", () => { look.colorbar = cb.checked; this.draw(); }); row(h("label", {}, cb, " colour bar")); body.append(h("div", { class: "nd2v-note", text: `${ov.field.layer} · ${ov.field.label}` })); }
      if (tab === "vectors") { const au = h("input", { type: "checkbox" }); au.checked = !!look.auto_scale; au.addEventListener("change", () => { look.auto_scale = au.checked; this.draw(); }); row(h("label", {}, au, " auto length"), h("label", { text: "× scale" }), this.slider(look, "scale", 0.1, 10, 0.1), h("label", { text: "every" }), this.slider(look, "every", 1, 20, 1)); row(h("label", { text: "colour" }), sel("color_mode", ["magnitude", "flat"]), h("label", { text: "width" }), this.slider(look, "width", 0.5, 5, 0.5)); }
      box.append(body);
    }
    if (avail.length) { box.insertBefore(tabs, box.children[1]); show(avail[0]); }
    for (const u of (ov.unsupported || [])) box.append(h("div", { class: "nd2v-note", text: `the ${u} overlay is not drawn in an HTML view` }));
    for (const n of (ov.notes || [])) box.append(h("div", { class: "nd2v-note", text: n }));
  }
  slider(look, key, lo, hi, step, after) { const r = h("input", { class: "nd2v-range", type: "range", min: lo, max: hi, step, value: look[key] }); r.title = `${key}: ${look[key]}`; r.addEventListener("input", () => { look[key] = Number(r.value); r.title = `${key}: ${look[key]}`; if (after) after(); this.draw(); }); return r; }
  invalidate(tab) { if (tab === "labels") this.labelCache.clear(); else this.rgbaCache.clear(); }
  // ── overlay indexes ──
  indexOverlays() {
    const pts = this.ov.points; if (pts) { this.pointIndex = new Map(); for (let i = 0; i < pts.m.length; i++) { const k = `${pts.m[i]},${pts.t[i]}`; let a = this.pointIndex.get(k); if (!a) { a = []; this.pointIndex.set(k, a); } a.push(i); } }
    const tr = this.ov.tracks; if (tr) { this.tracksByM = new Map(); for (const t of tr.tracks) { let a = this.tracksByM.get(t.m); if (!a) { a = []; this.tracksByM.set(t.m, a); } a.push(t); } }
  }
  // ── view transform ──
  resize() { const dpr = window.devicePixelRatio || 1, W = this.stage.clientWidth, H = this.stage.clientHeight; if (!W || !H) return; this.canvas.width = Math.round(W * dpr); this.canvas.height = Math.round(H * dpr); this.canvas.style.width = W + "px"; this.canvas.style.height = H + "px"; if (!this.userView) this.fit(false); else this.draw(); for (const k in this.hists) this.hists[k].paint(); for (const s of this.strips) s.paint(); }
  fit(user) { const W = this.stage.clientWidth, H = this.stage.clientHeight, pw = this.doc.plane.w, ph = this.doc.plane.h; if (!W || !H) return; this.zoom = Math.min(W / pw, H / ph); this.ox = (W - pw * this.zoom) / 2; this.oy = (H - ph * this.zoom) / 2; this.userView = false; if (user) this.userView = false; this.draw(); }
  one() { const W = this.stage.clientWidth, H = this.stage.clientHeight; this.zoomAt(W / 2, H / 2, 1 / this.zoom); }
  zoomAt(x, y, f) { const nz = clamp(this.zoom * f, 0.02, 400); f = nz / this.zoom; this.ox = x - (x - this.ox) * f; this.oy = y - (y - this.oy) * f; this.zoom = nz; this.userView = true; this.draw(); }
  imgPt(x, y) { const p = this.doc.plane; return [(x + 0.5) * p.sx - p.x0, (y + 0.5) * p.sy - p.y0]; }
  // ── rendering ──
  async render() {
    const gen = ++this.gen, doc = this.doc, on = this.chans.filter(c => c.on);
    const planes = {};
    try {
      for (const c of on) { const rec = doc.frames[this.frameKey(c.c)]; if (rec) planes[c.c] = await this.store.get(this.frameKey(c.c), rec); }
      const ok = this.ovKey(); const L = this.ov.labels && this.ov.labels.frames[ok]; if (L) this.labelPlane = await this.store.get("L" + ok, L); else this.labelPlane = null;
      const V = this.ov.voxels && this.ov.voxels.frames[ok]; if (V) this.voxPlane = await this.store.get("V" + ok, V); else this.voxPlane = null;
      const F = this.ov.field && this.ov.field.frames[ok]; if (F) this.fieldPlane = await this.store.get("F" + ok, F); else this.fieldPlane = null;
    } catch (e) { this.status.textContent = `decode failed: ${e.message}`; return; }
    if (gen !== this.gen) return;
    this.planes = planes;
    const layers = [];
    for (const c of on) { const pl = planes[c.c]; if (!pl) continue; if (this.autoOn && !c.userWindow) { [c.lo, c.hi] = autoWindow(pl); if (c._inputs) { c._inputs[0].value = fmtIntensity(c.lo); c._inputs[1].value = fmtIntensity(c.hi); } } const hs = this.hists[c.c]; if (hs) { hs.setWindow(c.lo, c.hi); hs.setGamma(c.gamma); hs.setValues(pl); } layers.push({ data: pl.data, scale: pl.scale, offset: pl.offset, lo: c.lo, hi: c.hi, gamma: c.gamma, rgb: c.rgb }); }
    const w = doc.plane.w, hgt = doc.plane.h; this.buf.width = w; this.buf.height = hgt;
    const ctx = this.buf.getContext("2d"); const img = ctx.createImageData(w, hgt); composite(img.data, w, hgt, layers); ctx.putImageData(img, 0, 0);
    this.paneBufs = {};
    if (this.split && on.length >= 2) for (const c of on) { const pl = planes[c.c]; if (!pl) continue; const cv = document.createElement("canvas"); cv.width = w; cv.height = hgt; const id = cv.getContext("2d").createImageData(w, hgt); composite(id.data, w, hgt, [{ data: pl.data, scale: pl.scale, offset: pl.offset, lo: c.lo, hi: c.hi, gamma: c.gamma, rgb: c.rgb }]); cv.getContext("2d").putImageData(id, 0, 0); this.paneBufs[c.c] = cv; }
    this.draw(); this.syncStatus(); if (this.hoverPt) this.onHover(this.hoverPt[0], this.hoverPt[1]);
  }
  syncStatus() { const d = this.doc; const parts = [d.title, `${d.axes.x} × ${d.axes.y} px`, `m ${this.m + 1}/${d.axes.m} · t ${this.t + 1}/${d.axes.t} · z ${this.z === "max" || this.z === "mean" ? this.z + " projection" : (this.zIndex() + 1) + "/" + d.axes.z}`]; if (d.plane_um) parts.push(`${d.plane_um.toPrecision(3)} µm/px shown`); this.status.textContent = parts.join("  ·  "); }
  draw() {
    const cv = this.canvas, dpr = window.devicePixelRatio || 1, W = this.stage.clientWidth, H = this.stage.clientHeight; if (!W || !H) return;
    const ctx = cv.getContext("2d"); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
    const on = this.chans.filter(c => c.on);
    const panes = (this.split && on.length >= 2) ? [{ label: "composite", buf: this.buf, rgb: null }].concat(on.map(c => ({ label: c.name, buf: this.paneBufs[c.c], rgb: c.rgb }))) : [{ label: "", buf: this.buf, rgb: null }];
    const n = panes.length, cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), pw = W / cols, ph = H / rows;
    panes.forEach((pane, i) => {
      if (!pane.buf) return;
      const px0 = (i % cols) * pw, py0 = Math.floor(i / cols) * ph, f = n > 1 ? Math.min(pw / W, ph / H) : 1;
      const z = this.zoom * f, ox = px0 + this.ox * f, oy = py0 + this.oy * f;
      ctx.save(); ctx.beginPath(); ctx.rect(px0 * dpr, py0 * dpr, pw * dpr, ph * dpr); ctx.clip();
      ctx.setTransform(z * dpr, 0, 0, z * dpr, ox * dpr, oy * dpr); ctx.imageSmoothingEnabled = z < 1; ctx.drawImage(pane.buf, 0, 0);
      if (i === 0) this.drawOverlays(ctx, z, dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (i === 0) this.drawHud(ctx, px0, py0, pw, ph, z, ox - px0, oy - py0);
      if (pane.label) { ctx.font = "bold 12px Segoe UI, sans-serif"; ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillText(pane.label, px0 + 7, py0 + 16); ctx.fillStyle = pane.rgb ? rgbCss(pane.rgb) : "#fff"; ctx.fillText(pane.label, px0 + 6, py0 + 15); }
      ctx.restore();
    });
  }
  cmap(name) { const c = this.doc.cmaps || {}; return c[name] || c.viridis || [[0, 0, 0], [255, 255, 255]]; }
  mapColor(lut, t) { const i = clamp(Math.round(t * (lut.length - 1)), 0, lut.length - 1); return lut[i]; }
  rgbaCanvas(key, plane, lut, lo, hi, opacity, alphaMode, nan0, threshold) {
    const ck = `${key}|${lo}|${hi}|${opacity}|${alphaMode}|${lut.length}`; const have = this.rgbaCache.get(ck); if (have && have.lutRef === lut) return have.cv;
    const cv = document.createElement("canvas"); cv.width = plane.w; cv.height = plane.h; const ctx = cv.getContext("2d"); const id = ctx.createImageData(plane.w, plane.h), d = id.data;
    const n = plane.w * plane.h, scale = plane.scale || 1, off = plane.offset || 0, span = (hi - lo) || 1, op = clamp(opacity / 100, 0, 1);
    for (let i = 0, k = 0; i < n; i++, k += 4) { const code = plane.data[i]; if (nan0 && code === 0) continue; const v = code * scale + off; const t = clamp((v - lo) / span, 0, 1); const c = this.mapColor(lut, t); d[k] = c[0]; d[k + 1] = c[1]; d[k + 2] = c[2]; let a = op; if (alphaMode === "ramp") a = op * t; else if (alphaMode === "gated") a = v > (threshold || 0) ? op : 0; d[k + 3] = Math.round(a * 255); }
    ctx.putImageData(id, 0, 0); if (this.rgbaCache.size > 24) this.rgbaCache.clear(); this.rgbaCache.set(ck, { cv, lutRef: lut }); return cv;
  }
  labelGeom(key, plane, look) {
    const ck = `${key}|${look.fill_opacity}|${look.color_mode}|${look.color}`; const have = this.labelCache.get(ck); if (have) return have;
    const { data, w, h: hh } = plane, keys = this.ov.labels.keys, paths = new Map(), sums = new Map();
    const colorOf = new Map(); const col = id => { let c = colorOf.get(id); if (!c) { c = look.color_mode === "single" ? hexRgb(look.color) : slotColor(keys && id < keys.length ? keys[id] : id, look.sat, look.val); colorOf.set(id, c); } return c; };
    const pathOf = id => { let p = paths.get(id); if (!p) { p = new Path2D(); paths.set(id, p); } return p; };
    const seg = (id, x0, y0, x1, y1) => { const p = pathOf(id); p.moveTo(x0, y0); p.lineTo(x1, y1); };
    let count = 0;
    for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
      const id = data[y * w + x]; const right = x + 1 < w ? data[y * w + x + 1] : 0, down = y + 1 < hh ? data[(y + 1) * w + x] : 0;
      if (id) { let s = sums.get(id); if (!s) { s = [0, 0, 0]; sums.set(id, s); } s[0] += x; s[1] += y; s[2] += 1; if (x === 0) seg(id, 0, y, 0, y + 1); if (y === 0) seg(id, x, 0, x + 1, 0); }
      if (right !== id) { if (id) { seg(id, x + 1, y, x + 1, y + 1); count++; } if (right) { seg(right, x + 1, y, x + 1, y + 1); count++; } }
      if (down !== id) { if (id) { seg(id, x, y + 1, x + 1, y + 1); count++; } if (down) { seg(down, x, y + 1, x + 1, y + 1); count++; } }
    }
    const fill = document.createElement("canvas"); fill.width = w; fill.height = hh; const id2 = fill.getContext("2d").createImageData(w, hh), d = id2.data, fa = Math.round(clamp(look.fill_opacity / 100, 0, 1) * 255);
    for (let i = 0, k = 0; i < w * hh; i++, k += 4) { const id = data[i]; if (!id) continue; const c = col(id); d[k] = c[0]; d[k + 1] = c[1]; d[k + 2] = c[2]; d[k + 3] = fa; }
    fill.getContext("2d").putImageData(id2, 0, 0);
    const cents = []; for (const [id, s] of sums) cents.push([id, s[0] / s[2] + 0.5, s[1] / s[2] + 0.5]);
    const geom = { paths, colorOf: col, fill, cents, count }; if (this.labelCache.size > 12) this.labelCache.clear(); this.labelCache.set(ck, geom); return geom;
  }
  drawOverlays(ctx, z, dpr) {
    const look = this.look, ov = this.ov, zi = this.zIndex(), proj = this.z === "max" || this.z === "mean";
    const screen = px => px / z;                             // a screen size in plane units
    // scalar wash, under everything
    if (ov.voxels && look.voxels.enabled && this.voxPlane) { const lk = look.voxels; const lims = lk.clim_auto ? ov.voxels.limits : [lk.clim_lo, lk.clim_hi]; const cv = this.rgbaCanvas("V" + this.ovKey(), this.voxPlane, this.cmap(lk.colormap), lims[0], lims[1], lk.opacity, lk.alpha_mode, false, 0); ctx.imageSmoothingEnabled = z < 1; ctx.drawImage(cv, 0, 0); }
    // a measured field's column
    if (ov.field && look.field.enabled && this.fieldPlane) { const lk = look.field, fp = this.fieldPlane; const lims = this.fieldLimits(); const cmap = lk.colormap === "auto" ? ov.field.cmap : lk.colormap; const cv = this.rgbaCanvas("F" + this.ovKey() + lk.colormap, fp, this.cmap(cmap), lims[0], lims[1], lk.opacity, "flat", true, 0); const [y0, y1, x0, x1] = fp.extent.map(v => v); const a = this.imgPt(x0, y0), b = this.imgPt(x1, y1); ctx.imageSmoothingEnabled = lk.style !== "cells"; ctx.drawImage(cv, a[0] - 0.5 * this.doc.plane.sx, a[1] - 0.5 * this.doc.plane.sy, (b[0] - a[0]), (b[1] - a[1])); }
    // vectors
    if (ov.vectors && look.vectors.enabled) { const rows = ov.vectors.frames[this.ovKey()]; if (rows) { const lk = look.vectors, k = (lk.auto_scale ? ov.vectors.auto : 1) * lk.scale, lut = this.cmap(lk.colormap), hi = ov.vectors.mag_hi || 1, flat = hexRgb(lk.color), every = Math.max(1, Math.round(lk.every)); ctx.lineWidth = screen(lk.width); ctx.lineCap = "round"; const head = screen(lk.head_px); for (let i = 0; i < rows.length; i += every) { const [y, x, dy, dx] = rows[i]; const mag = Math.hypot(dx, dy); if (mag < lk.gate) continue; const col = lk.color_mode === "flat" ? flat : this.mapColor(lut, clamp(mag / hi, 0, 1)); ctx.strokeStyle = rgbCss(col, lk.opacity / 100); const a = this.imgPt(x, y), b = this.imgPt(x + dx * k, y + dy * k); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); const ang = Math.atan2(b[1] - a[1], b[0] - a[0]); ctx.moveTo(b[0], b[1]); ctx.lineTo(b[0] - head * Math.cos(ang - 0.5), b[1] - head * Math.sin(ang - 0.5)); ctx.moveTo(b[0], b[1]); ctx.lineTo(b[0] - head * Math.cos(ang + 0.5), b[1] - head * Math.sin(ang + 0.5)); ctx.stroke(); } } }
    // labels
    if (ov.labels && look.labels.enabled && this.labelPlane) { const lk = look.labels, g = this.labelGeom("L" + this.ovKey(), this.labelPlane, lk); if (lk.style !== "outline") { ctx.imageSmoothingEnabled = false; ctx.drawImage(g.fill, 0, 0); } if (lk.style !== "fill") { ctx.lineWidth = screen(lk.width); ctx.lineJoin = "miter"; for (const [id, p] of g.paths) { ctx.strokeStyle = rgbCss(g.colorOf(id), lk.opacity / 100); ctx.stroke(p); } } if (lk.show_ids && g.cents.length <= 400) { ctx.font = `bold ${screen(lk.id_px)}px Segoe UI, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; for (const [id, cx, cy] of g.cents) { ctx.fillStyle = "rgba(0,0,0,.7)"; ctx.fillText(String(id), cx + screen(1), cy + screen(1)); ctx.fillStyle = "#fff"; ctx.fillText(String(id), cx, cy); } } }
    // points
    if (ov.points && look.points.enabled && this.pointIndex) { const pts = ov.points, lk = look.points, idx = this.pointIndex.get(`${this.m},${this.t}`) || []; const unit = screen(lk.unit_px), arm = unit * lk.spread; ctx.lineCap = "round"; for (const i of idx) { const zp = Math.round(pts.z[i]); const onPlane = proj || zp === zi; if (!onPlane && !lk.z_project) continue; const col = lk.color_mode === "single" ? hexRgb(lk.color) : lk.color_mode === "per_z" ? slotColor(zp, lk.sat, lk.val) : lk.color_mode === "per_layer" ? slotColor(pts.layer[i], lk.sat, lk.val) : slotColor(pts.slot[i], lk.sat, lk.val); const a = (onPlane ? lk.opacity : lk.off_opacity) / 100; const [x, y] = this.imgPt(pts.x[i], pts.y[i]); ctx.strokeStyle = rgbCss(col, a); ctx.lineWidth = screen(1.6); for (let s = 1; s <= 3; s++) { const r0 = arm * (s - 1) / 3, r1 = arm * s / 3; ctx.globalAlpha = 1 - (s - 1) * 0.3; ctx.beginPath(); ctx.moveTo(x - r1, y); ctx.lineTo(x - r0, y); ctx.moveTo(x + r0, y); ctx.lineTo(x + r1, y); ctx.moveTo(x, y - r1); ctx.lineTo(x, y - r0); ctx.moveTo(x, y + r0); ctx.lineTo(x, y + r1); ctx.stroke(); } ctx.globalAlpha = 1; ctx.fillStyle = rgbCss([255, 255, 255], a); ctx.fillRect(x - unit * 0.35, y - unit * 0.35, unit * 0.7, unit * 0.7); } }
    // tracks
    if (ov.tracks && look.tracks.enabled && this.tracksByM) { const lk = look.tracks, list = this.tracksByM.get(this.m) || [], T = this.t; ctx.lineWidth = screen(lk.width); ctx.lineJoin = "round"; ctx.lineCap = "round"; for (const tr of list) { const col = lk.color_mode === "single" ? hexRgb(lk.color) : slotColor(tr.slot, lk.sat, lk.val); const n = tr.t.length; let lo = 0, hi = n; if (lk.trail === "past") { hi = 0; while (hi < n && tr.t[hi] <= T) hi++; } else if (lk.trail === "window") { lo = 0; while (lo < n && tr.t[lo] < T - lk.window) lo++; hi = lo; while (hi < n && tr.t[hi] <= T) hi++; } if (hi - lo < 1) continue; if (lk.fade && lk.trail !== "all") { for (let i = lo + 1; i < hi; i++) { const age = T - tr.t[i]; ctx.strokeStyle = rgbCss(col, (lk.opacity / 100) * clamp(1 - age / Math.max(1, lk.window || 10), 0.15, 1)); const a = this.imgPt(tr.x[i - 1], tr.y[i - 1]), b = this.imgPt(tr.x[i], tr.y[i]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); } } else { ctx.strokeStyle = rgbCss(col, lk.opacity / 100); ctx.beginPath(); for (let i = lo; i < hi; i++) { const p = this.imgPt(tr.x[i], tr.y[i]); if (i === lo) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); } ctx.stroke(); } ctx.fillStyle = rgbCss(col, lk.opacity / 100); const vr = screen(lk.vertex_px) / 2; for (let i = lo; i < hi; i++) { const p = this.imgPt(tr.x[i], tr.y[i]); const r = tr.t[i] === T ? screen(lk.head_px) / 2 : vr; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.fill(); } } }
  }
  fieldLimits() { const lk = this.look.field, ov = this.ov.field; if (lk.limits === "manual") return [lk.clim_lo, lk.clim_hi]; if (lk.limits === "frame" && this.fieldPlane) { const s = sampleValues(this.fieldPlane, 50000); return s.length ? [percentile(s, 1), percentile(s, 99)] : ov.limits; } return ov.limits; }
  drawHud(ctx, px0, py0, pw, ph, z, ox, oy) {
    const doc = this.doc, W = pw, H = ph, imgX0 = px0 + ox, imgY0 = py0 + oy, imgW = doc.plane.w * z, imgH = doc.plane.h * z;
    const ax0 = Math.max(px0, imgX0), ay0 = Math.max(py0, imgY0), ax1 = Math.min(px0 + W, imgX0 + imgW), ay1 = Math.min(py0 + H, imgY0 + imgH);
    const shadow = (fn, x, y) => { ctx.fillStyle = "rgba(0,0,0,.66)"; fn(x + 1, y + 1); ctx.fillStyle = this.decorColor(); fn(x, y); };
    if (doc.scalebar.show && doc.plane_um && ax1 > ax0 && ay1 > ay0) {
      let lenPlane, label; if (doc.scalebar.um > 0) { lenPlane = doc.scalebar.um / doc.plane_um; label = doc.scalebar.um < 1000 ? `${doc.scalebar.um} µm` : `${doc.scalebar.um / 1000} mm`; } else { const sb = scalebar(doc.axes.x, doc.pixel_size_um); if (sb) { lenPlane = sb.um / doc.plane_um; label = sb.label; } }
      if (lenPlane) { let len = Math.min(lenPlane * z, (ax1 - ax0) * 0.9); if (len >= 4) { const inset = Math.max(8, Math.min(16, (ax1 - ax0) * 0.05)); const x1 = ax0 + inset, y1 = ay1 - inset; ctx.fillStyle = "rgba(0,0,0,.43)"; ctx.fillRect(x1 - 1.5, y1 - 4.5, len + 3, 6); ctx.fillStyle = this.decorColor(); ctx.fillRect(x1, y1 - 3, len, 3); ctx.font = "bold 12px Segoe UI, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; shadow((x, y) => ctx.fillText(label, x, y), x1 + len / 2, y1 - 7); ctx.textAlign = "left"; } }
    }
    const ts = this.timestampText(); if (ts) { ctx.font = "bold 13px Segoe UI, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top"; shadow((x, y) => ctx.fillText(ts, x, y), px0 + 8, py0 + 6); }
    if (this.ov.field && this.look.field.enabled && this.look.field.colorbar && this.fieldPlane) { const lims = this.fieldLimits(), lk = this.look.field, cmap = this.cmap(lk.colormap === "auto" ? this.ov.field.cmap : lk.colormap); const bx = px0 + W - 24, by = py0 + 10, bw = 10, bh = Math.min(120, H * 0.4); for (let i = 0; i < bh; i++) { const c = this.mapColor(cmap, 1 - i / bh); ctx.fillStyle = rgbCss(c); ctx.fillRect(bx, by + i, bw, 1.5); } ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.lineWidth = 1; ctx.strokeRect(bx - 0.5, by - 0.5, bw + 1, bh + 1); ctx.font = "11px Segoe UI, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "top"; shadow((x, y) => ctx.fillText(fmtIntensity(lims[1]), x, y), bx - 4, by - 2); ctx.textBaseline = "bottom"; shadow((x, y) => ctx.fillText(fmtIntensity(lims[0]), x, y), bx - 4, by + bh + 2); ctx.save(); ctx.translate(bx + bw + 11, by + bh / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "middle"; shadow((x, y) => ctx.fillText(this.ov.field.label, x, y), 0, 0); ctx.restore(); ctx.textAlign = "left"; }
  }
  decorColor() { const named = { white: "#ffffff", black: "#000000", yellow: "#ffd600", cyan: "#00e5ff" }; return named[this.doc.scalebar.color] || "#ffffff"; }
  timestampText() {
    const mode = this.doc.timestamp.mode; if (!mode || mode === "off") return ""; const tm = this.doc.times || {}, T = this.doc.axes.t, t = this.t;
    const clock = () => (tm.clock && tm.clock[t]) || "";
    const elapsed = () => { if (tm.seconds) return elapsedText(tm.seconds[t] - tm.seconds[0], tm.seconds[T - 1] - tm.seconds[0]); if (tm.dt_s) return elapsedText(tm.dt_s * t, tm.dt_s * (T - 1)); return `t ${t + 1}/${T}`; };
    if (mode === "clock") return clock(); if (mode === "frame") return `t ${t + 1}/${T}`; if (mode === "elapsed_clock") { const c = clock(); return c ? `${elapsed()}  [${c}]` : elapsed(); } return elapsed();
  }
  // ── hover ──
  screenToPlane(sx, sy) { const W = this.stage.clientWidth, H = this.stage.clientHeight, on = this.chans.filter(c => c.on); const n = (this.split && on.length >= 2) ? on.length + 1 : 1; const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), pw = W / cols, ph = H / rows; const i = Math.min(n - 1, Math.floor(sx / pw) + Math.floor(sy / ph) * cols); const px0 = (i % cols) * pw, py0 = Math.floor(i / cols) * ph, f = n > 1 ? Math.min(pw / W, ph / H) : 1; const z = this.zoom * f, ox = px0 + this.ox * f, oy = py0 + this.oy * f; return [(sx - ox) / z, (sy - oy) / z]; }
  onHover(sx, sy) {
    this.hoverPt = [sx, sy]; const [px, py] = this.screenToPlane(sx, sy); const doc = this.doc, p = doc.plane;
    if (px < 0 || py < 0 || px >= p.w || py >= p.h) { this.hover.textContent = "hover the image to read a pixel"; this.hover.classList.add("hint"); if (this.figRead) this.figRead.textContent = ""; return; }
    const xf = (px + p.x0) / p.sx - 0.5, yf = (py + p.y0) / p.sy - 0.5;   // the node's own pixel grid
    const parts = [`x ${Math.floor(xf + 0.5)}, y ${Math.floor(yf + 0.5)} px`];
    const vals = []; const ix = Math.floor(px), iy = Math.floor(py);
    for (const c of this.chans) { if (!c.on) continue; const pl = this.planes[c.c]; if (!pl) continue; const v = pl.data[iy * pl.w + ix] * (pl.scale || 1) + (pl.offset || 0); vals.push(`${c.name} ${fmtIntensity(v)}`); }
    if (doc.pixel_size_um) { parts.push(`${(xf * doc.pixel_size_um).toFixed(1)}, ${(yf * doc.pixel_size_um).toFixed(1)} µm`); const st = doc.stage_ok && doc.positions.stage_xy_um[this.m]; if (st) { const sxx = st[0] + (xf - (doc.axes.x - 1) / 2) * doc.pixel_size_um, syy = st[1] + (yf - (doc.axes.y - 1) / 2) * doc.pixel_size_um; parts.push(`stage ${sxx.toFixed(1)}, ${syy.toFixed(1)} µm`); } }
    if (vals.length) parts.push(vals.join(" · "));
    if (this.labelPlane) { const id = this.labelPlane.data[iy * this.labelPlane.w + ix]; if (id) parts.push(`${this.ov.labels.layer || "label"} ${id}`); }
    if (this.fieldPlane && this.look.field.enabled) { const fp = this.fieldPlane, [y0, y1, x0, x1] = fp.extent; const gx = (xf - x0) / ((x1 - x0) || 1) * fp.w, gy = (yf - y0) / ((y1 - y0) || 1) * fp.h; if (gx >= 0 && gy >= 0 && gx < fp.w && gy < fp.h) { const code = fp.data[Math.floor(gy) * fp.w + Math.floor(gx)]; if (code !== 0) parts.push(`${this.ov.field.column} ${fmtIntensity(code * fp.scale + fp.offset)}`); } }
    const text = parts.join("  ·  "); this.hover.textContent = text; this.hover.classList.remove("hint"); if (this.figRead) this.figRead.textContent = text;
  }
  onKey(e) {
    const strip = ax => this.strips.find(s => s.axis === ax); const k = e.key;
    if (k === "ArrowRight" || k === "ArrowLeft") { const s = strip("t"); if (s) s.step(k === "ArrowRight" ? 1 : -1); }
    else if (k === "ArrowUp" || k === "ArrowDown") { const s = strip("z"); if (s) s.step(k === "ArrowUp" ? 1 : -1); }
    else if (k === "PageUp" || k === "PageDown") { const s = strip("m"); if (s) s.step(k === "PageDown" ? 1 : -1); }
    else if (k === " ") { const s = strip("t"); if (s) s.toggle(); }
    else if (k === "Home") this.fit(true); else if (k === "0") this.one();
    else if (k === "s" || k === "S") this.splitBtn.click(); else return;
    e.preventDefault();
  }
  savePng() { const cv = this.canvas, a = document.createElement("a"); a.download = `${this.doc.title.replace(/[^\w.-]+/g, "_")}_m${this.m}_t${this.t}_z${this.z}.png`; a.href = cv.toDataURL("image/png"); a.click(); }
}

// ───────────────────────────── mounting ─────────────────────────────
ND2V.mount = function (el, doc, opts) { return new Viewer(el, doc, opts); };
ND2V.mountFigure = function (container, fig) {
  const W = fig.page.width_mm, H = fig.page.height_mm, store = new PlaneStore(480e6), viewers = {};
  const pct = (v, total) => `${(100 * v / total).toFixed(4)}%`;
  const groups = fig.links || [];
  const onFrame = (v, axis, value) => { for (const g of groups) if (g.includes(v.figId)) for (const other of g) if (other !== v.figId && viewers[other]) viewers[other].setFrame({ [axis]: value }); };
  for (const [id, view] of Object.entries(fig.views || {})) {
    const [x, y, w, hh] = view.rect_mm; const item = h("div", { class: "nd2v-fig-item", style: `left:${pct(x, W)};top:${pct(y, H)};width:${pct(w, W)};height:${pct(hh, H)}` });
    container.append(item); const v = new Viewer(item, view.doc, { compact: true, store, onFrame }); v.figId = id; viewers[id] = v;
    if (view.letter) item.append(h("div", { class: "nd2v-letter", text: view.letter }));
  }
  const letterPx = () => { const px = container.clientWidth * ((fig.page.font_pt || 8) * 1.25 / 72 * 25.4) / W; container.querySelectorAll(".nd2v-letter").forEach(e => { e.style.fontSize = `${Math.max(9, px)}px`; }); };
  new ResizeObserver(letterPx).observe(container); letterPx();
  const tip = h("div", { class: "nd2v-tip" }); document.body.append(tip);
  for (const [id, ch] of Object.entries(fig.charts || {})) {
    const [x, y, w, hh] = ch.rect_mm, g = ch.geometry; if (!g) continue;
    const el = h("div", { class: "nd2v-chart", style: `left:${pct(x, W)};top:${pct(y, H)};width:${pct(w, W)};height:${pct(hh, H)}` }); container.append(el);
    el.addEventListener("mousemove", e => {
      const r = el.getBoundingClientRect(); const xmm = x + (e.clientX - r.left) / r.width * w, ymm = y + (e.clientY - r.top) / r.height * hh;
      const [ax, ay, aw, ah] = g.axes_mm; const inside = xmm >= ax && xmm <= ax + aw && ymm >= ay && ymm <= ay + ah;
      const xs = Scale(g.x.scale === "category" ? "linear" : g.x.scale, g.x.lim[0], g.x.lim[1], ax, ax + aw), ys = Scale(g.y.scale === "category" ? "linear" : g.y.scale, g.y.lim[0], g.y.lim[1], ay + ah, ay);
      const fmt = (v, cats) => cats && cats.length ? (cats[Math.round(v)] != null ? cats[Math.round(v)] : "") : fmtIntensity(Number(v.toPrecision(4)));
      let best = null; for (const mk of g.marks || []) { if (mk.kind !== "markers" || !["series", "means"].includes(mk.role)) continue; (mk.pts_mm || []).forEach((p, k) => { const d = Math.hypot(p[0] - xmm, p[1] - ymm); if (d <= 1.5 && (!best || d < best.d)) best = { d, mk, k, p }; }); }
      let text = "";
      if (best) { const s = (ch.series || [])[best.mk.series]; const label = s && s.label ? `${s.label}: ` : ""; text = `${label}x = ${fmt(xs.fromMm(best.p[0]), g.x.categories)}, y = ${fmt(ys.fromMm(best.p[1]), g.y.categories)}`; }
      else if (inside) text = `x = ${fmt(xs.fromMm(xmm), g.x.categories)}, y = ${fmt(ys.fromMm(ymm), g.y.categories)}`;
      if (!text) { tip.style.display = "none"; return; }
      tip.textContent = text; tip.style.display = "block"; tip.style.left = `${e.pageX + 12}px`; tip.style.top = `${e.pageY + 12}px`;
    });
    el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  }
  return viewers;
};
})();
