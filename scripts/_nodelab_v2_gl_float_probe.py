"""Desktop-only probe (V4.00 step 4): a Viewer popped out of the window keeps its picture.

The GPU Viewer surface is a ``QOpenGLWidget``, and Qt gives it a NEW GL context whenever it
moves to another top-level window — popping its dock out, docking it back, the mini-map.
Everything built in the old context (textures, shaders, buffers) is gone by then;
:meth:`nodelab_v2.glview.GLImageView.initializeGL` rebuilds them and replays the last planes.
This probe makes Qt do that for real and checks the result:

* the surface was given a new context on each move (``_context_gen`` counts them), and
* the frame it paints afterwards is still the picture, not the clear colour.

It needs a real display and a real OpenGL driver, so it is NOT part of the offscreen gate
set (the GUI probe covers the CPU surface there, VW6). Run it on a desktop::

    PYTHONUTF8=1 .venv\\Scripts\\python.exe -u scripts/_nodelab_v2_gl_float_probe.py

It prints ``ALL GL FLOAT PROBES PASSED``, or ``SKIP`` with the reason when the machine
cannot run it (offscreen platform, ``NODELAB_GL=0``, no OpenGL).
"""
from __future__ import annotations

import os
import sys
import time

os.environ["NODELAB_LAYOUT"] = "0"               # never read or write the user's panel layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _skip(why: str) -> None:
    print(f"SKIP — {why}")
    sys.stdout.flush()
    os._exit(0)


def main() -> None:
    if os.environ.get("QT_QPA_PLATFORM", "") in ("offscreen", "minimal"):
        _skip("needs a real display (QT_QPA_PLATFORM is offscreen)")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv[:1])
    from nodelab_v2.window import MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    win.build_demo()
    done = []
    win.runner.finished.connect(lambda nid, *a: done.append(nid))
    win.pull_node("n3")
    t0 = time.time()
    while win.runner.run_id("n3") not in done and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert done, "the demo's n3 never finished"
    v = win.viewer
    gl = getattr(v, "_gl", None)
    if gl is None:
        _skip("the Viewer runs on the CPU surface here (NODELAB_GL=0, or no OpenGL)")

    def settle(cond, timeout=5.0) -> bool:
        t1 = time.time()
        while time.time() - t1 < timeout:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.02)
        return cond()

    def painted() -> bool:
        """A frame grabbed off the surface is not one flat colour (the clear colour)."""
        img = gl.grabFramebuffer()
        if img.isNull() or img.width() < 8 or img.height() < 8:
            return False
        first = img.pixel(img.width() // 2, img.height() // 2)
        step_x, step_y = max(1, img.width() // 24), max(1, img.height() // 24)
        return any(img.pixel(x, y) != first
                   for x in range(0, img.width(), step_x)
                   for y in range(0, img.height(), step_y))

    assert settle(lambda: gl._ok and painted()), "the docked Viewer never painted the image"
    gen0 = gl._context_gen
    dock = win._viewer_dock(v)
    print(f"[ok] docked: GL context #{gen0}, picture on screen")

    dock.setFloating(True)
    assert settle(lambda: gl._context_gen > gen0), \
        f"floating the dock did not rebuild the GL context (still #{gl._context_gen})"
    assert settle(lambda: gl._ok and painted()), "the floating Viewer lost its picture"
    print(f"[ok] floating: GL context #{gl._context_gen}, picture still on screen")

    gen1 = gl._context_gen
    dock.setFloating(False)
    assert settle(lambda: gl._context_gen > gen1), "docking back did not rebuild the context"
    assert settle(lambda: gl._ok and painted()), "the re-docked Viewer lost its picture"
    print(f"[ok] docked back: GL context #{gl._context_gen}, picture still on screen")

    print("\nALL GL FLOAT PROBES PASSED")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
