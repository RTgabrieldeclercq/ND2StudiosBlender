"""``scene`` — layer nodes of the composable 3-D HTML scene viewer (V4.10).

Each node here is a tap: it appends one *layer spec* to the stream's ``scene_layers``
metadata (or, for ``scene.place``, sets its ``scene_frame``) and hands the Dataset through
untouched. ``io.write_scene_viewer`` reads the specs of every wired stream and writes the
page. See ``CodeLog/ClaudesPlan/V4.10_scene_viewer.md``.
"""
