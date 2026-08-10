"""GPU image surface for the Viewer — the "converter model" (napari / BigDataViewer).

A single textured quad is drawn per frame; each active channel's plane is uploaded to
a GPU texture **once** (cached by identity), and contrast (``lo/hi``), gamma, emission
colour and additive multi-channel compositing all happen in the **fragment shader** from
uniforms. Consequences that fix the "viewer updates way slower than the load" problem:

* Moving the T/Z cursor only swaps which textures are bound (or uploads the new plane
  once) — no per-frame ``np.percentile``, no numpy→QImage→QPixmap copies.
* Dragging a contrast/colour/gamma control is a uniform change + ``update()`` — **zero**
  decode, zero re-upload.

Planes are uploaded as single-channel 32-bit float (``R32F``): the uint16→float cast is
done **once at upload** (not per frame), which lets the shader use a plain ``sampler2D``
for every channel — far more portable than mixing ``usampler2D``/``sampler2D`` banks, and
the normalisation is exact.

Overlays (points / labels / tracks) stay on the CPU: :class:`GLImageView` calls back into
the panel with a :class:`QPainter` after the GL draw, and exposes :meth:`plane_to_widget`
so the panel maps plane-space geometry onto the (pan/zoom) view.

**Split-channel view** (:meth:`set_tiles`, NIS-Elements style): instead of one quad the
widget lays out a near-square grid of *panes*, each drawing its own subset of channels
(pane 0 is the composite, then one pane per channel) with the same shader and the same
uniforms. Zoom/pan is shared — a pane is a window onto the same plane region, so the
panes stay registered — and each pane is scissored so a zoomed image cannot bleed into
its neighbour. Overlays keep mapping through pane 0.

If the GL context or shader is unusable this widget never crashes — it emits
:data:`gl_failed` once and the panel swaps in the CPU :class:`~nodelab_v2.viewer._ImageView`.

Reparenting the Viewer (docking it into the mini-map and back) makes Qt destroy and
recreate the context, which invalidates every texture/buffer/program made in the old
one. :meth:`GLImageView._release_gl` frees them while that context is still current and
:meth:`GLImageView.initializeGL` rebuilds on the new one, replaying ``_last_planes`` so
the image comes straight back instead of going black.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QImage, QMatrix4x4, QPainter, QPen, QSurfaceFormat, QVector2D,
    QVector3D, QVector4D)
from PySide6.QtOpenGL import (
    QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from nodelab_v2.overlays import GL_MAX_CHANNELS as OV_GL_MAX_CHANNELS

#: raw GL enums we need (QOpenGLWidget hands us a GL-ES-style functions object)
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_TRIANGLE_STRIP = 0x0005
_GL_BLEND = 0x0BE2
_GL_FLOAT = 0x1406
_GL_TEXTURE_2D = 0x0DE1
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_MIN_FILTER = 0x2801
_GL_TEXTURE_MAG_FILTER = 0x2800
_GL_TEXTURE_WRAP_S = 0x2802
_GL_TEXTURE_WRAP_T = 0x2803
_GL_NEAREST = 0x2600
_GL_LINEAR = 0x2601
_GL_CLAMP_TO_EDGE = 0x812F
_GL_R32F = 0x822E
_GL_R16 = 0x822A
_GL_RED = 0x1903
_GL_RGBA = 0x1908
_GL_RGBA8 = 0x8058
_GL_UNSIGNED_BYTE = 0x1401
_GL_UNSIGNED_SHORT = 0x1403
_GL_UNPACK_ALIGNMENT = 0x0CF5
_GL_SCISSOR_TEST = 0x0C11

#: gap between the panes of the split-channel view, in widget pixels
_TILE_GAP = 2.0

#: a split pane: (label, the channels it composites, the label's (r,g,b))
Tile = Tuple[str, Tuple[int, ...], Tuple[int, int, int]]

#: sampler bank size (microscopy rarely exceeds this). Shared with the Viewer via
#: :data:`nodelab_v2.overlays.GL_MAX_CHANNELS` so the limit the status line reports and the
#: limit the shader enforces cannot drift.
_MAX_CH = OV_GL_MAX_CHANNELS

#: PixelUnpackBuffer enum (PySide6 exposes it on the class or the nested Type enum)
_PIXEL_UNPACK = getattr(QOpenGLBuffer, "PixelUnpackBuffer", None)
if _PIXEL_UNPACK is None:            # pragma: no cover — older bindings nest it under Type
    _PIXEL_UNPACK = QOpenGLBuffer.Type.PixelUnpackBuffer

_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;   // clip-space xy
layout(location = 1) in vec2 a_uv;    // texture uv
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

def _build_frag(n: int) -> str:
    # Sampler arrays MUST be indexed by a constant in GLSL 330 (dynamic indexing is
    # undefined pre-400 → links but samples black on many drivers), so unroll with a
    # guard per channel instead of a `for i` loop over u_tex[i]. Contrast is baked into
    # the 8-bit texture on the CPU (the float-texture upload path is broken in this
    # PySide6 build); the shader keeps colour + gamma as free uniforms.
    # The raw 16-bit value is packed into R (high byte) + G (low byte) of an RGBA8 texture
    # (only RGBA8/ubyte uploads reliably in this PySide6 build). The unpack + LUT window
    # are INLINED per channel with a constant sampler index — passing a sampler-array
    # element to a helper function returns a bad sampler on this driver. Windowing here in
    # the shader (u_vlo/u_vhi) makes contrast changes a free uniform update.
    # u_win[i] = vec2(vlo, vhi) — the LUT window is carried as a vec2 (set via QVector2D),
    # NOT two float uniforms: setUniformValue(loc, python_float) silently fails for a
    # float ARRAY element in this PySide6 build (vector overloads work), which was the
    # cause of the white frame. Windowing is inlined per channel with a constant index.
    # u_win[i] = vec3(vlo, vhi, gamma) — window + gamma carried as a vec3 (set via
    # QVector3D), NOT float array uniforms: setUniformValue(loc, python_float) silently
    # fails for a float ARRAY element in this PySide6 build (vector overloads work). The
    # transfer function is t = pow(clamp((v-vlo)/(vhi-vlo)), gamma) — gamma < 1 brightens
    # midtones, > 1 darkens. Windowing inlined per channel with a constant index
    # (passing a sampler-array element to a helper returns a bad sampler on this driver).
    # Per-channel BLEND (V2.19). `u_blend[i] = vec3(mode, opacity, checker_cells)` — a vec3
    # for the same reason `u_win` is one: setUniformValue silently fails for a float ARRAY
    # element in this PySide6 build, and the vector overloads are what work.
    #
    # Blending is per CHANNEL rather than global because that is exactly the granularity an
    # overlay needs: `view.overlay` contributes its source as extra channel indices ABOVE
    # the primary's own, so the primary's channels keep compositing additively among
    # themselves (the microscopy convention) and only the overlay carries the chosen mode.
    # The unroll runs in index order, so an overlay channel always blends against the
    # already-accumulated primary — which is what makes the non-commutative modes (over,
    # difference) mean what they say.
    lines = []
    for i in range(n):
        lines.append(
            f"    if (u_nchan > {i}) {{\n"
            f"        vec4 c{i} = texture(u_tex[{i}], v_uv);\n"
            # GL_R16 (`_upload`): the sampler already normalizes to u16/65535, so `.r`
            # IS the value the RGBA8 byte-packing used to reconstruct here.
            f"        float v{i} = c{i}.r;\n"
            f"        float t{i} = clamp((v{i} - u_win[{i}].x) / "
            f"max(u_win[{i}].y - u_win[{i}].x, 1e-6), 0.0, 1.0);\n"
            f"        t{i} = pow(t{i}, max(u_win[{i}].z, 1e-3));\n"
            f"        rgb = blend_one(rgb, t{i} * u_color[{i}], t{i}, u_blend[{i}], g_uv);\n"
            f"    }}")
    body = "\n".join(lines)
    return f"""
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex[{n}];
uniform int   u_nchan;
uniform vec3  u_color[{n}];
uniform vec3  u_win[{n}];
uniform vec3  u_blend[{n}];
// (u0, v0, du, dv): where this quad sits in the WHOLE image, in normalized image coords.
// (0,0,1,1) for the overview quad; the patch's own rect for a viewport detail quad. The
// spatial comparators (checkerboard, wipe) are defined on the image, so they must be
// computed from image uv — off the quad's own uv, a detail patch grew a full checkerboard
// inside its rect and the wipe divider jumped to the middle of wherever you had zoomed.
uniform vec4  u_rect;

// `src` is the channel's tinted colour, `t` its own windowed intensity, `spec` its
// (mode, opacity, checker_cells), `g_uv` its position in the whole image. Kept as ONE
// function taking a sampler-free argument list so the per-channel unroll above stays a
// single line — passing a sampler-array element to a helper returns a bad sampler on this
// driver, but plain vectors are fine.
vec3 blend_one(vec3 dst, vec3 src, float t, vec3 spec, vec2 g_uv) {{
    int mode = int(spec.x + 0.5);
    float op = clamp(spec.y, 0.0, 1.0);
    if (mode == 1) {{
        // OVER — the channel's own brightness is its alpha, so a dark background does not
        // occlude what is underneath it. That is the behaviour a fluorescence overlay
        // wants: a flat alpha would grey out the primary everywhere the secondary is empty.
        return mix(dst, src, clamp(t, 0.0, 1.0) * op);
    }} else if (mode == 2) {{
        // DIFFERENCE — the registration-QA look: aligned structure cancels toward black,
        // misalignment lights up as edge-doubling.
        return abs(dst - src * op);
    }} else if (mode == 3) {{
        // CHECKERBOARD — alternating squares of each source, in TEXTURE space so the
        // squares stay locked to the image while you pan and zoom rather than crawling
        // across it. spec.z is the number of cells on the long edge.
        float cells = max(spec.z, 1.0);
        float k = mod(floor(g_uv.x * cells) + floor(g_uv.y * cells), 2.0);
        return mix(dst, src, k * op);
    }} else if (mode == 4) {{
        // WIPE — one vertical divider at spec.z (0..1 across the image), the overlay to
        // its LEFT. Also in texture space, so dragging the divider moves it across the
        // specimen rather than across the window: the boundary you are judging stays on
        // the same feature while you pan.
        float k = step(g_uv.x, clamp(spec.z, 0.0, 1.0));
        return mix(dst, src, k * op);
    }}
    return dst + src * op;                       // 0 = ADD (the microscopy default)
}}

void main() {{
    vec3 rgb = vec3(0.0);
    vec2 g_uv = u_rect.xy + v_uv * u_rect.zw;
{body}
    frag = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}}
"""


_FRAG = _build_frag(_MAX_CH)

#: diagnostic: output the interpolated UV as R,G so we can tell geometry/UV apart from
#: texture upload (enabled with NODELAB_GL_UVTEST=1).
_FRAG_UVTEST = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(v_uv, 0.0, 1.0); }
"""

#: diagnostic: output channel-0 texture red directly (no gamma/colour/nchan) — isolates
#: the texture upload from the compositing/uniform logic (NODELAB_GL_TEXTEST=1).
_FRAG_TEXTEST = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex[%(N)d];
void main() { float v = texture(u_tex[0], v_uv).r; frag = vec4(v, v, v, 1.0); }
""" % {"N": _MAX_CH}


def pack_u16(plane: np.ndarray):
    """``(u16_plane, dmin, dmax)`` — a display plane packed into 16 bits plus the data
    range those texels map back to. Pure; module level so the gate can check it without a
    GL context (the offscreen probe runs the CPU surface, so nothing else here is
    reachable headlessly).

    A float plane is normalized by its OWN extremes, and that is deliberately NOT keyed to
    the payload's declared ``bit_depth``, though it looks like it should be. Every
    streaming provider computes in float64, so a Stitch or a Gaussian lands here while its
    values are still integer counts — but the range chosen here is handed to the shader as
    ``u_win`` and the LUT window is renormalized against it
    (``vlo = (lo - dmin) / span``), so ``dmin``/``dmax`` **cancel**: the rendered value is
    ``(v - clim_lo) / (clim_hi - clim_lo)`` whatever scale is used. Measured: a pixel of
    intensity 700 renders to 0.3913 on two frames with different extremes, per-plane and
    fixed-scale alike.

    Pinning it to ``0 .. 2**bit_depth-1`` was tried and reverted: it changes nothing
    on screen, quantizes more coarsely (a 4095-wide window into 16 bits instead of the
    plane's own ~1400-wide one), and CLIPS a node that legitimately overshoots full scale
    (a blur ringing, a feather sum) where the per-plane rule shows it. The contrast parity
    that DID need fixing was the LUT slider extent, which is
    ``ViewerPanel._display_range`` and reads the declaration properly."""
    a = np.asarray(plane)
    if a.dtype == np.uint16:
        return a, 0.0, 65535.0
    if a.dtype == np.uint8:
        return a.astype(np.uint16) * 257, 0.0, 255.0
    af = np.nan_to_num(a.astype(np.float32))
    dmin, dmax = float(af.min()), float(af.max())
    if dmax <= dmin:
        dmax = dmin + 1.0
    u16 = ((af - dmin) / (dmax - dmin) * 65535.0).astype(np.uint16)
    return u16, dmin, dmax


class GLImageView(QOpenGLWidget):
    """OpenGL textured-quad image surface with shader contrast/colour/compositing.

    Implements the same overlay/pick contract as the CPU
    :class:`~nodelab_v2.viewer._ImageView` — ``plane_to_widget`` / ``widget_to_plane`` /
    ``refresh`` / ``overlay_cb`` (:data:`nodelab_v2.viewer.SURFACE_CONTRACT`) — so one
    renderer and one picker serve both backends.
    """

    gl_failed = Signal()
    #: the context came up and reported what it can hold: ``GL_MAX_TEXTURE_SIZE`` in px. The
    #: Viewer forwards it to the runner, which uses it to decide full-resolution vs pyramid.
    limits_ready = Signal(int)
    #: the pan/zoom changed — the panel re-requests a viewport detail patch (debounced).
    view_changed = Signal()

    #: Declared at CLASS level (and re-bound per instance below) so the surface contract can
    #: be checked without constructing a GL context.
    overlay_cb = None

    def __init__(self) -> None:
        super().__init__()
        import os
        self._debug = os.environ.get("NODELAB_GL_DEBUG", "") in ("1", "true", "yes")
        self._dbg_once = False
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        # per-channel GPU state
        self._tex: Dict[int, QOpenGLTexture] = {}
        self._tex_key: Dict[int, int] = {}     # channel → id(plane) currently uploaded
        self._range: Dict[int, Tuple[float, float]] = {}  # channel → data range mapped to [0,1]
        self._clim: Dict[int, Tuple[float, float]] = {}   # channel → LUT window (data units)
        self._color: Dict[int, Tuple[float, float, float]] = {}
        self._gamma: Dict[int, float] = {}
        #: channel → (blend mode, opacity, checker cells). Absent means (0, 1, 8) — plain
        #: additive at full strength, i.e. exactly what every channel did before overlays.
        self._blend: Dict[int, Tuple[float, float, float]] = {}
        self._active: List[int] = []
        #: split-channel panes; empty == one pane compositing every active channel
        self._tiles: List[Tile] = []
        # planes waiting to be uploaded — ALL GL work is deferred to paintGL (the only
        # place the context is guaranteed current on the right thread); uploading from
        # set_planes via makeCurrent() before the first paint segfaults on some drivers.
        self._pending: Dict[int, np.ndarray] = {}
        # the last planes handed to us, kept so a context REBUILD (Qt destroys and
        # recreates the context when the widget is reparented — docking the Viewer into
        # the mini-map and back) can re-upload them instead of showing a black frame.
        self._last_planes: Dict[int, np.ndarray] = {}
        self._img_wh: Optional[Tuple[int, int]] = None   # (w, h) of the texture
        # ── viewport detail patch (a second, finer texture over the visible rect) ──
        self._dtex: Dict[int, QOpenGLTexture] = {}
        self._dtex_key: Dict[int, int] = {}
        self._drange: Dict[int, Tuple[float, float]] = {}
        self._dpending: Dict[int, np.ndarray] = {}
        self._dlast: Dict[int, np.ndarray] = {}
        #: normalized image rect the detail textures cover, or ``None`` for "no patch"
        self._drect: Optional[Tuple[float, float, float, float]] = None
        # view transform (plane-space → widget): fit-scale × user zoom, + pan (widget px)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_wh: Optional[Tuple[int, int]] = None
        self._panning: Optional[QPointF] = None
        # GL objects (built in initializeGL)
        self._prog: Optional[QOpenGLShaderProgram] = None
        self._vbo: Optional[QOpenGLBuffer] = None
        self._vao: Optional[QOpenGLVertexArrayObject] = None
        self._pbo: Optional[QOpenGLBuffer] = None
        self._ok = False
        self._failed = False
        #: the panel sets this to paint overlays after the GL draw: cb(painter)
        self.overlay_cb: Optional[Callable[[QPainter], None]] = None

    # ── GL lifecycle ───────────────────────────────────────────────────────────
    def initializeGL(self) -> None:
        import sys
        # Qt calls this again on a fresh context whenever the widget is reparented
        # (Viewer → mini-map → back). Every GL object built in the previous context is
        # dead by now, so start clean and re-queue the last planes for upload.
        self._tex.clear()
        self._tex_key.clear()
        self._dtex.clear()
        self._dtex_key.clear()
        self._prog = self._vbo = self._vao = None
        self._ok = False
        try:
            f = self.context().functions()
            f.glClearColor(0.02, 0.03, 0.05, 1.0)
            prog = QOpenGLShaderProgram(self)
            from PySide6.QtOpenGL import QOpenGLShader
            import os
            if os.environ.get("NODELAB_GL_UVTEST", "") in ("1", "true", "yes"):
                frag = _FRAG_UVTEST
            elif os.environ.get("NODELAB_GL_TEXTEST", "") in ("1", "true", "yes"):
                frag = _FRAG_TEXTEST
            else:
                frag = _FRAG
            if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERT):
                raise RuntimeError("vertex: " + prog.log())
            if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag):
                raise RuntimeError("fragment: " + prog.log())
            if not prog.link():
                raise RuntimeError("link: " + prog.log())
            self._prog = prog
            vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            vbo.create()
            vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)
            self._vbo = vbo
            # A VAO is MANDATORY in a 3.3 core profile — without one bound, glDrawArrays
            # draws nothing (silently, no error) → a black frame.
            vao = QOpenGLVertexArrayObject(self)
            vao.create()
            self._vao = vao
            self._ok = True
            # tear our objects down while this context is still alive (a reparent
            # destroys it) and bring the picture back on the new one
            self.context().aboutToBeDestroyed.connect(self._release_gl)
            if self._last_planes:
                self._pending.update(self._last_planes)
            if self._dlast and self._drect is not None:
                self._dpending.update(self._dlast)
            ver = self.context().format().version()
            # What this GPU will actually accept as one texture axis. It decides whether the
            # Viewer may show a big frame WHOLE at full resolution or has to work off the
            # pyramid — so it is published rather than assumed (the runner's conservative
            # default is 8192; this machine's driver reports 32768).
            self._max_tex = self._query_max_texture(f)
            self.limits_ready.emit(int(self._max_tex))
            print(f"[glview] GL ready — OpenGL {ver[0]}.{ver[1]}, "
                  f"max texture {self._max_tex} px", file=sys.stderr, flush=True)
        except Exception as e:                   # noqa: BLE001 — degrade, never crash
            print(f"[glview] GL init failed → CPU fallback: {e}", file=sys.stderr, flush=True)
            self._ok = False
            if not self._failed:
                self._failed = True
                self.gl_failed.emit()

    #: ``GL_MAX_TEXTURE_SIZE``. Not imported from PyOpenGL — this module deliberately depends
    #: only on Qt's own GL wrapper, so the enum is spelled out.
    _GL_MAX_TEXTURE_SIZE = 0x0D33

    @classmethod
    def _query_max_texture(cls, f: Any) -> int:
        """The context's ``GL_MAX_TEXTURE_SIZE``, or a conservative 8192.

        The return shape of ``glGetIntegerv`` differs across PySide6 builds (a scalar on some,
        a sequence on others), so both are handled and anything unrecognized falls back — the
        conservative direction to be wrong in, since too small only costs sharpness while too
        large is a driver-rejected upload and a black frame."""
        try:
            got = f.glGetIntegerv(cls._GL_MAX_TEXTURE_SIZE)
            if isinstance(got, (list, tuple)) and got:
                got = got[0]
            px = int(got)
            return px if px >= 1024 else 8192
        except Exception:  # noqa: BLE001 — a display hint, never a failure
            return 8192

    def max_texture_px(self) -> int:
        """What one texture axis may be on the live context (8192 until it comes up)."""
        return int(getattr(self, "_max_tex", 0) or 8192)

    def _release_gl(self) -> None:
        """Destroy this context's GL objects while it is still usable — Qt emits
        ``QOpenGLContext.aboutToBeDestroyed`` on a reparent (and at teardown). Nothing
        here may raise: the widget has to survive to be re-initialized on the next
        context, which :meth:`initializeGL` then rebuilds from ``_last_planes``."""
        try:
            self.makeCurrent()
            for tex in list(self._dtex.values()):
                try:
                    tex.destroy()
                except Exception:                # noqa: BLE001 — best-effort cleanup
                    pass
            self._dtex.clear()
            self._dtex_key.clear()
            for tex in self._tex.values():
                try:
                    tex.destroy()
                except Exception:                # noqa: BLE001 — best-effort cleanup
                    pass
            for obj in (self._vbo, self._vao, self._pbo):
                if obj is not None:
                    try:
                        obj.destroy()
                    except Exception:            # noqa: BLE001
                        pass
            if self._prog is not None:
                self._prog.setParent(None)       # drop it now, with the context current
        except Exception:                        # noqa: BLE001 — never break teardown
            pass
        finally:
            self._tex.clear()
            self._tex_key.clear()
            self._prog = self._vbo = self._vao = self._pbo = None
            self._ok = False
            try:
                self.doneCurrent()
            except Exception:                    # noqa: BLE001
                pass

    def _fail(self) -> None:
        self._ok = False
        if not self._failed:
            self._failed = True
            self.gl_failed.emit()

    # ── public API (mirrors _ImageView / used by the panel) ─────────────────────
    def set_planes(self, planes: Dict[int, np.ndarray]) -> None:
        """Queue each channel's plane for upload and repaint. No GL here — the actual
        texture upload happens in :meth:`paintGL` (context guaranteed current). Refit is
        pure math, so it's safe to do now."""
        if not planes:
            return
        self._active = list(planes.keys())
        self._last_planes = dict(planes)      # replayed if the context is rebuilt
        wh = None
        for ch, plane in planes.items():
            h, w = plane.shape[:2]
            wh = (w, h)
            if self._tex_key.get(ch) != id(plane):
                self._pending[ch] = plane     # (re)upload only when the plane changed
        if wh is not None and wh != self._img_wh:
            self.clear_detail()          # the patch belonged to the previous image
            self._img_wh = wh
            if self._last_wh != wh:                # new image size → refit (keep zoom on scrub)
                self._last_wh = wh
                self.fit()
        self.update()

    def set_detail(self, planes: Dict[int, np.ndarray],
                   rect01: Tuple[float, float, float, float]) -> None:
        """Queue a finer texture covering ``rect01`` (normalized image coords) to be drawn
        over the overview. Part of :data:`nodelab_v2.viewer.SURFACE_CONTRACT`.

        The overview is never replaced, so there is always a complete picture on screen and
        a patch that is late, partial or superseded can only ever make a sub-rect sharper.
        ``planes`` empty (or :meth:`clear_detail`) drops back to the overview alone."""
        if not planes:
            self.clear_detail()
            return
        self._drect = tuple(float(v) for v in rect01)      # type: ignore[assignment]
        self._dlast = dict(planes)
        for ch, plane in planes.items():
            if self._dtex_key.get(ch) != id(plane):
                self._dpending[ch] = plane
        self.update()

    def clear_detail(self) -> None:
        """Forget the detail patch (the frame/node/graph moved, or the view zoomed back
        out). Cheap and idempotent; the textures themselves are reused on the next patch."""
        if self._drect is None and not self._dpending:
            return
        self._drect = None
        self._dpending.clear()
        self._dlast.clear()
        self.update()

    def visible_rect01(self) -> Tuple[float, float, float, float]:
        """The image rect currently visible in pane 0, in normalized image coords, clamped
        to the image. ``(0,0,1,1)`` when the whole image is on screen — which is how the
        panel knows a detail patch would buy nothing."""
        w, h = self._img_wh or (1, 1)
        ox, oy, dw, dh = self._disp_rect(0)
        if dw <= 0 or dh <= 0:
            return (0.0, 0.0, 1.0, 1.0)
        rects = self._tile_rects()
        vx, vy, vw, vh = rects[0] if rects else (0.0, 0.0, self.width(), self.height())
        x0 = max(0.0, min(1.0, (vx - ox) / dw))
        y0 = max(0.0, min(1.0, (vy - oy) / dh))
        x1 = max(0.0, min(1.0, (vx + vw - ox) / dw))
        y1 = max(0.0, min(1.0, (vy + vh - oy) / dh))
        return (x0, y0, max(x1, x0), max(y1, y0))

    def _upload(self, ch: int, plane: np.ndarray,
                texmap: Optional[Dict[int, QOpenGLTexture]] = None,
                keymap: Optional[Dict[int, int]] = None,
                rangemap: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
        # Upload the RAW plane ONCE as a normalized 16-bit single-channel texture (R16).
        # The LUT window is applied in the shader (u_vlo/u_vhi) so contrast changes are
        # free — no re-upload, no re-decode. We record the data range that maps texel
        # [0,1] back to data units, so the viewer's clim (data units) → normalized window.
        u16, dmin, dmax = pack_u16(plane)
        u16 = np.ascontiguousarray(u16)
        h, w = u16.shape[:2]
        texmap = self._tex if texmap is None else texmap
        keymap = self._tex_key if keymap is None else keymap
        rangemap = self._range if rangemap is None else rangemap
        rangemap[ch] = (dmin, dmax)
        if self._debug:
            import sys
            # `plane`, not the `a` that only exists inside `pack_u16` — this line raised
            # NameError, and `paintGL`'s blanket except turned that into `_fail()`, a
            # one-way fall back to the CPU surface. So the one switch you reach for when
            # the GL path renders wrong was disabling the GL path.
            print(f"[glview] upload ch{ch}: shape={u16.shape} "
                  f"src_dtype={np.asarray(plane).dtype} "
                  f"range=({dmin:.4g},{dmax:.4g}) clim={self._clim.get(ch)}",
                  file=sys.stderr, flush=True)
        # A native GL_R16 texture: the sampler hands the shader ``u16/65535`` as ``.r``,
        # which is exactly the value the old code reconstructed. It used to pack the high
        # and low bytes into R and G of an RGBA8 texture (an ES2-era trick; this widget
        # requires a 3.3 core context, where R16 has been core since 3.0), and at
        # native-resolution frames that packing WAS the playback cost: building the 4-byte
        # copy of a 9217×6145 plane moved ~700 MB of CPU traffic per frame — measured
        # 0.36 s/frame, i.e. 3 fps with a fully warm plane cache (2026-08-10). R16 uploads
        # the plane's own bytes: same picture, same window math, a quarter of the traffic,
        # none of the repacking.
        f = self.context().functions()
        tex = texmap.get(ch)
        if tex is None:
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.create()
            texmap[ch] = tex
        f.glBindTexture(_GL_TEXTURE_2D, tex.textureId())
        # rows of odd width are 2-byte texels at arbitrary byte offsets; the default
        # 4-byte unpack alignment would shear them
        f.glPixelStorei(_GL_UNPACK_ALIGNMENT, 1)
        # NEAREST keeps this change invisible: it is what the packed texture required, so
        # sampling must not start interpolating the moment the format stops forbidding it.
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_NEAREST)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_NEAREST)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
        f.glTexImage2D(_GL_TEXTURE_2D, 0, _GL_R16, w, h, 0, _GL_RED,
                       _GL_UNSIGNED_SHORT, u16.tobytes())
        keymap[ch] = id(plane)

    def set_channel(self, ch: int, lo: float, hi: float,
                    color: Tuple[int, int, int], gamma: float = 1.0,
                    blend: int = 0, opacity: float = 1.0,
                    checker: float = 8.0) -> None:
        """Set a channel's LUT window (``lo``/``hi``, data units), emission colour, and how
        it composites. Every one of them is a free shader uniform — no re-upload, no
        re-decode — so dragging a contrast slider or an overlay's opacity is instantaneous.
        Call :meth:`refresh` to repaint.

        ``blend``: 0 add · 1 over · 2 difference · 3 checkerboard (see ``blend_one``).
        ``checker`` is the number of cells on the long edge, read only by mode 3."""
        self._clim[ch] = (float(lo), float(hi))
        self._color[ch] = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        self._gamma[ch] = float(gamma)
        self._blend[ch] = (float(int(blend)), float(opacity), float(checker))

    def set_tiles(self, tiles: Optional[Sequence[Tile]]) -> None:
        """Set the split-channel panes — ``[(label, channels, label_rgb), …]`` — or pass an
        empty sequence / ``None`` for the single composite pane. Pure layout state: no
        upload, no decode, so toggling the split is a repaint."""
        new = [(str(lb), tuple(int(c) for c in chans), tuple(int(v) for v in rgb))
               for lb, chans, rgb in (tiles or [])]
        if new == self._tiles:
            return
        self._tiles = new
        self.update()

    def refresh(self) -> None:
        self.update()

    def clear(self) -> None:
        self._active = []
        self.update()

    # ── view transform (pane-aware: every pane shares one zoom/pan) ──────────────
    def _ntiles(self) -> int:
        return max(1, len(self._tiles))

    def _tile_grid(self, W: float, H: float, n: int) -> Tuple[int, int]:
        """Columns × rows for ``n`` panes — the split whose cells show the image LARGEST.
        A square grid is the wrong default on the real shapes this panel takes: three
        panes over a wide viewer fit far better 3×1 than 2×2 (which also leaves a hole)."""
        w, h = self._img_wh or (1, 1)
        best, best_cols = -1.0, 1
        for cols in range(1, n + 1):
            rows = int(math.ceil(n / cols))
            tw = (W - _TILE_GAP * (cols - 1)) / cols
            th = (H - _TILE_GAP * (rows - 1)) / rows
            if tw <= 0 or th <= 0:
                continue
            s = min(tw / max(1, w), th / max(1, h))
            if s > best:
                best, best_cols = s, cols
        return best_cols, int(math.ceil(n / best_cols))

    def _tile_rects(self) -> List[Tuple[float, float, float, float]]:
        """The pane rectangles ``(x, y, w, h)`` in widget pixels — one equal-sized cell per
        pane (a single pane is the whole widget). Cells share their size, which is what
        lets one shared pan offset keep the panes registered."""
        W, H = float(max(1, self.width())), float(max(1, self.height()))
        n = self._ntiles()
        if n == 1:
            return [(0.0, 0.0, W, H)]
        cols, rows = self._tile_grid(W, H, n)
        tw = max(1.0, (W - _TILE_GAP * (cols - 1)) / cols)
        th = max(1.0, (H - _TILE_GAP * (rows - 1)) / rows)
        out = []
        for i in range(n):
            r, c = divmod(i, cols)
            out.append((c * (tw + _TILE_GAP), r * (th + _TILE_GAP), tw, th))
        return out

    def _disp_rect(self, tile: int = 0) -> Tuple[float, float, float, float]:
        """(origin_x, origin_y, disp_w, disp_h) of the image inside pane ``tile``, in
        widget pixels — fit-to-pane × user zoom, plus the shared pan."""
        rects = self._tile_rects()
        tx, ty, tw, th = rects[min(max(0, tile), len(rects) - 1)]
        w, h = self._img_wh or (1, 1)
        s = min(tw / max(1, w), th / max(1, h)) * self._zoom
        dw, dh = w * s, h * s
        ox = tx + (tw - dw) / 2.0 + self._pan.x()
        oy = ty + (th - dh) / 2.0 + self._pan.y()
        return ox, oy, dw, dh

    def plane_to_widget(self, px: float, py: float, tile: int = 0) -> QPointF:
        """Map a plane-space pixel (0..W, 0..H) to widget device-independent coords —
        the panel uses this to place overlay geometry on the pan/zoomed image. In the
        split view this addresses pane 0 (the composite), so overlays live there."""
        w, h = self._img_wh or (1, 1)
        ox, oy, dw, dh = self._disp_rect(tile)
        return QPointF(ox + (px / max(1, w)) * dw, oy + (py / max(1, h)) * dh)

    def widget_to_plane(self, pt: QPointF) -> Optional[Tuple[float, float]]:
        """The inverse of :meth:`plane_to_widget` — a widget point → displayed-plane
        ``(x, y)``, or ``None`` when there is no image or the point is outside it.

        Half of the overlay/pick contract (:meth:`refresh`, :attr:`overlay_cb` and
        :meth:`plane_to_widget` are the rest), which
        :class:`~nodelab_v2.viewer._ImageView` implements too so the picker is
        backend-agnostic. Omitting it here silently disabled every on-image gesture on the
        GPU path — the DEFAULT surface in a windowed session — because the panel probes for
        it with ``getattr`` and simply did nothing when it was absent. There is now a probe
        gate (``nodelab_v2.viewer.SURFACE_CONTRACT``) that fails loudly instead.

        Returns ``None`` outside the image rather than extrapolating: a click beside the
        picture is not a coordinate, and letting it through produced ROI shapes with
        negative vertices that rasterized to nothing."""
        at = self._img_at(pt)
        if at is None:
            return None
        _tile, px, py = at
        w, h = self._img_wh or (0, 0)
        if not (0 <= px <= w and 0 <= py <= h):
            return None
        return (px, py)

    def fit(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.clear_detail()          # the whole image fits again — nothing to refine
        self.update()
        self.view_changed.emit()

    def wheelEvent(self, e) -> None:
        f = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
        cursor = e.position()
        # keep the point under the cursor fixed while zooming — measured in the pane the
        # cursor is over, so zooming into a split pane holds that pane's point still.
        before = self._img_at(cursor)
        self._zoom = max(0.05, min(80.0, self._zoom * f))
        if before:
            tile, px, py = before
            self._pan += cursor - self.plane_to_widget(px, py, tile)
        self.update()
        self.view_changed.emit()

    def _tile_at(self, wpt: QPointF) -> int:
        """Index of the pane under a widget point (the nearest one if it lands in a gap)."""
        for i, (x, y, w, h) in enumerate(self._tile_rects()):
            if x <= wpt.x() <= x + w and y <= wpt.y() <= y + h:
                return i
        return 0

    def _img_at(self, wpt: QPointF):
        """``(tile, plane_x, plane_y)`` under a widget point, or ``None``."""
        w, h = self._img_wh or (0, 0)
        if not w or not h:
            return None
        tile = self._tile_at(wpt)
        ox, oy, dw, dh = self._disp_rect(tile)
        if dw <= 0 or dh <= 0:
            return None
        return (tile, (wpt.x() - ox) / dw * w, (wpt.y() - oy) / dh * h)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._panning = e.position()

    def mouseMoveEvent(self, e) -> None:
        if self._panning is not None:
            self._pan += e.position() - self._panning
            self._panning = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        was_panning = self._panning is not None
        self._panning = None
        if was_panning:
            # on RELEASE, not per mouse-move: a drag is dozens of moves and each would
            # supersede the previous request, so the patch could never land mid-drag
            self.view_changed.emit()

    def mouseDoubleClickEvent(self, e) -> None:
        self.fit()
        e.accept()

    # ── paint ────────────────────────────────────────────────────────────────────
    def paintGL(self) -> None:
        f = self.context().functions()
        f.glClear(_GL_COLOR_BUFFER_BIT)
        if self._ok and self._prog is not None:
            try:
                if self._pending:                # upload queued planes (context is current)
                    for ch, plane in list(self._pending.items()):
                        self._upload(ch, plane)
                    self._pending.clear()
                if self._dpending:               # ...and the viewport detail patch
                    for ch, plane in list(self._dpending.items()):
                        self._upload(ch, plane, self._dtex, self._dtex_key, self._drange)
                    self._dpending.clear()
                if self._active and self._img_wh:
                    rects = self._tile_rects()
                    if self._tiles:
                        dpr = self.devicePixelRatioF()
                        f.glEnable(_GL_SCISSOR_TEST)
                        for i, (_lb, chans, _rgb) in enumerate(self._tiles):
                            x, y, w, h = rects[i]
                            # clip to the pane: a zoomed image must not bleed into its
                            # neighbours (every pane draws the full quad, cropped here).
                            f.glScissor(int(x * dpr), int((self.height() - y - h) * dpr),
                                        max(1, int(w * dpr)), max(1, int(h * dpr)))
                            self._draw_image(f, i, [c for c in chans if c in self._tex])
                        f.glDisable(_GL_SCISSOR_TEST)
                    else:
                        self._draw_image(f, 0, self._active)
            except Exception as e:               # noqa: BLE001
                import sys
                print(f"[glview] paint/upload failed → CPU fallback: {e}",
                      file=sys.stderr, flush=True)
                self._fail()
        # pane frames + labels, then the overlays (CPU QPainter), sharing the pan/zoom
        if self.overlay_cb is not None or self._tiles:
            try:
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing, True)
                if self._tiles:
                    self._draw_tile_chrome(p)
                if self.overlay_cb is not None:
                    self.overlay_cb(p)
                p.end()
            except Exception:                    # noqa: BLE001 — overlays are non-fatal
                pass

    def _draw_tile_chrome(self, p: QPainter) -> None:
        """Name each split pane in its channel's colour, with a hairline frame — the pane
        legend is the whole point of the split view (which channel am I looking at)."""
        font = QFont(p.font())
        font.setPointSizeF(max(8.0, min(11.0, font.pointSizeF())))
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        for (label, _chans, rgb), (x, y, w, h) in zip(self._tiles, self._tile_rects()):
            col = QColor(*rgb)
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 70), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x + 0.5, y + 0.5, w - 1.0, h - 1.0))
            # a plate under the name rather than a text shadow: the label sits over live
            # pixels, and a second offset draw of the glyphs reads as a smear.
            tw = min(fm.horizontalAdvance(label) + 12.0, max(20.0, w - 8.0))
            plate = QRectF(x + 4, y + 4, tw, fm.height() + 4.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(6, 9, 14, 170))
            p.drawRoundedRect(plate, 4.0, 4.0)
            p.setPen(col)
            p.drawText(plate.adjusted(6, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, label)

    def _draw_image(self, f, tile: int, chans: Sequence[int]) -> None:
        """The overview quad, then — when one has arrived — the viewport DETAIL quad on
        top of it.

        Two quads rather than a bigger texture: the overview is the whole image at
        ``MAX_DISPLAY_DIM``, and the detail patch is the visible rect at that same budget,
        so together they are "always something to show, sharp where you are looking". The
        detail draws second and opaquely (blending is off), so it simply replaces the
        overview inside its own rect; nothing has to be blended or masked."""
        if not chans:
            return
        self._draw_quad(f, tile, chans, (0.0, 0.0, 1.0, 1.0), self._tex, self._range)
        if self._drect is not None:
            # `_dlast`, not `_dtex`: the texture map is kept across patches (uploads are
            # reused), so a channel the CURRENT patch does not carry would otherwise be drawn
            # from the previous patch's pixels at the new patch's rect. An overlay legitimately
            # drops out of a patch — the secondary covers only part of the field — and that is
            # a channel this quad must simply not draw.
            fine = [c for c in chans if c in self._dtex and c in self._dlast]
            if fine:
                self._draw_quad(f, tile, fine, self._drect, self._dtex, self._drange)

    def _draw_quad(self, f, tile: int, chans: Sequence[int],
                   rect01: Tuple[float, float, float, float],
                   texmap: Dict[int, QOpenGLTexture],
                   rangemap: Dict[int, Tuple[float, float]]) -> None:
        """Draw ``chans`` from ``texmap`` over the sub-rect ``rect01`` (normalized image
        coords) of pane ``tile``. ``rect01 == (0,0,1,1)`` is the whole image."""
        if not chans:
            return
        fx0, fy0, fx1, fy1 = rect01
        ox, oy, dw, dh = self._disp_rect(tile)
        ox, oy = ox + fx0 * dw, oy + fy0 * dh
        dw, dh = (fx1 - fx0) * dw, (fy1 - fy0) * dh
        W, H = max(1, self.width()), max(1, self.height())

        def clip(wx, wy):
            return (wx / W) * 2.0 - 1.0, 1.0 - (wy / H) * 2.0
        # triangle strip: TL, BL, TR, BR ; uv has v flipped (image row 0 at top)
        corners = [(ox, oy, 0.0, 0.0), (ox, oy + dh, 0.0, 1.0),
                   (ox + dw, oy, 1.0, 0.0), (ox + dw, oy + dh, 1.0, 1.0)]
        verts = []
        for wx, wy, u, v in corners:
            cx, cy = clip(wx, wy)
            verts += [cx, cy, u, v]
        data = np.asarray(verts, dtype=np.float32).tobytes()

        prog, vbo, vao = self._prog, self._vbo, self._vao
        prog.bind()
        vao.bind()
        vbo.bind()
        vbo.allocate(data, len(data))
        prog.enableAttributeArray(0)
        prog.setAttributeBuffer(0, _GL_FLOAT, 0, 2, 16)
        prog.enableAttributeArray(1)
        prog.setAttributeBuffer(1, _GL_FLOAT, 8, 2, 16)

        active = list(chans)[:_MAX_CH]
        # Uniforms MUST go through uniformLocation() + the (location:int, value) overload:
        # PySide6 has no setUniformValue(name:str, scalar) overload (only location-based
        # for a single int/float), so name-based scalar calls raise.
        prog.setUniformValue(prog.uniformLocation("u_nchan"), int(len(active)))
        # where this quad sits in the whole image — what makes the checkerboard and the wipe
        # land on the same specimen coordinates in a detail patch as in the overview
        prog.setUniformValue(prog.uniformLocation("u_rect"),
                             QVector4D(float(fx0), float(fy0),
                                       float(fx1 - fx0), float(fy1 - fy0)))
        for i, ch in enumerate(active):
            r, g, b = self._color.get(ch, (1.0, 1.0, 1.0))
            # the LUT window is in DATA units, so it must be renormalized against THIS
            # texture's own range — a detail patch of a float image has a different
            # min/max from the overview, and reusing the overview's would shift its
            # contrast against the picture underneath it
            dmin, dmax = rangemap.get(ch, (0.0, 1.0))
            span = max(dmax - dmin, 1e-9)
            lo, hi = self._clim.get(ch, (dmin, dmax))
            vlo = (float(lo) - dmin) / span      # LUT window → texel-normalized [0,1]
            vhi = (float(hi) - dmin) / span
            gm = self._gamma.get(ch, 1.0)
            bmode, bop, bchk = self._blend.get(ch, (0.0, 1.0, 8.0))
            prog.setUniformValue(prog.uniformLocation(f"u_color[{i}]"), QVector3D(r, g, b))
            prog.setUniformValue(prog.uniformLocation(f"u_win[{i}]"),
                                 QVector3D(float(vlo), float(vhi), float(gm)))
            prog.setUniformValue(prog.uniformLocation(f"u_blend[{i}]"),
                                 QVector3D(float(bmode), float(bop), float(bchk)))
            prog.setUniformValue(prog.uniformLocation(f"u_tex[{i}]"), int(i))
            tex = texmap.get(ch)
            if tex is not None:
                f.glActiveTexture(_GL_TEXTURE0 + i)
                f.glBindTexture(_GL_TEXTURE_2D, tex.textureId())

        f.glDisable(_GL_BLEND)
        f.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)

        f.glActiveTexture(_GL_TEXTURE0)
        prog.disableAttributeArray(0)
        prog.disableAttributeArray(1)
        vbo.release()
        vao.release()
        prog.release()


def default_surface_format() -> QSurfaceFormat:
    """A 3.3-core surface format — set as the app default BEFORE the QApplication so
    every QOpenGLWidget gets a modern context."""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setSwapInterval(0)               # don't vsync-cap playback fps
    return fmt


def probe_gl_available() -> bool:
    """Whether the Viewer should try the GPU backend. We deliberately do **not** create a
    probe GL context here: on the headless ``offscreen``/``minimal`` Qt platforms that
    call crashes the process (a C++ segfault, uncatchable), and those are exactly the
    platforms used by the selftest probe and CI. So gate on the platform name — a real
    windowed session gets GL, headless stays on the CPU path — and let
    :meth:`GLImageView.initializeGL` handle a genuine driver failure at runtime by
    emitting :data:`GLImageView.gl_failed` (the panel then swaps to the CPU view)."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        name = (app.platformName() if app is not None else "").lower()
        return name not in ("", "offscreen", "minimal", "vnc")
    except Exception:                            # noqa: BLE001
        return False


__all__ = ["GLImageView", "default_surface_format", "probe_gl_available"]
