"""ND2 extended-metadata reader — the optics/calibration parse behind v2 ingest.

Vendored **verbatim** (2026-07-29) from the retired v1 backend's
``nd2studios/backend/nd2_loader.py`` when NodeLab v1 was removed. It is the one piece of v1
the v2 app layer depended on: :mod:`nodelab_v2.ingest` calls
:func:`read_nd2_metadata_extended` for a real microscope file's calibration + per-channel
display metadata (pixel/z size, channel names, emission/excitation λ, native colour,
exposure, objective, binning, stage positions, frame timestamps).

**Do not re-derive this.** The ``nd2`` SDK exposes these fields inconsistently across
versions and file vintages, so every read goes through :func:`_safe` and the structure is
probed defensively. It was proven against the lab's real files; a "cleaner" rewrite loses
that. Qt-free; ``nd2`` is imported lazily inside the function.

Only the two functions v2 actually calls came across — the v1 ``ND2Metadata`` dataclass,
the pixel loaders, the Z-projection helper, and the JSON sidecar cache stayed behind with
v1 (v2 reads pixels via :func:`nodelab_v2.ingest.read_nd2` / ``nd2.to_dask()`` and caches
through the engine memo instead).

**Additions (2026-07-31) — the placement vocabulary.** Overlaying two FILES needs each
one's absolute position on the microscope, and four fields the vendored parse left on the
floor turned out to be present in the lab's real ND2s all along. Nothing existing was
rewritten; these are extra reads beside it:

* ``stage_z_um`` — per-multipoint focal Z. It was already read, but only inside the
  ``frame_metadata`` FALLBACK branch, which never runs when ``f.experiment`` supplies the
  XY loop — so it came back ``[]`` on every file whose positions were planned. The
  ``XYPosLoop`` points carry ``z`` themselves, so it now costs nothing.
* ``frame_time_jd`` — per-timepoint ``absoluteJulianDayNumber``. ``frame_timestamps_s`` is
  ``relativeTimeMs``, whose ORIGIN is per-file: on the WellA3 pair the two relative clocks
  start 1207.3 s apart, so comparing them across files is meaningless. The Julian day is
  the only common clock, and it is what says GFP[k] was acquired 2415 s after 640[k].
* ``z_home_index`` / ``z_bottom_to_top`` — the ``ZStackLoop`` anchoring. ``stagePositionUm.z``
  is CONSTANT along a stack (it is the position's nominal focus, not the per-slice piezo
  reading), so without knowing which slice that nominal Z names, a stack cannot be placed in
  absolute Z at all. ``homeIndex=0, bottomToTop=True, stepUm=0.288`` puts slice *k* of the
  WellA3 640 stack at ``5971.96 + k*0.288`` µm, which is what reveals that the GFP plane at
  5999.74 µm falls INSIDE that stack, at slice ≈96 of 210.
* ``bit_depth`` — the significant sensor depth (12 on these files, not 16).
  :func:`nodelab_v2.ingest.read_calibration` already read it separately off
  ``f.attributes``; reporting it here too keeps the two readers from disagreeing.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def _safe(get, default=None):
    """Run a getter that may explode on missing fields and return `default`."""
    try:
        return get()
    except Exception:
        return default


def _point_names(f, n_m):
    """The acquisition's own name for each multipoint, or ``[]``.

    NIS keeps a POINT LIST behind a multipoint experiment, and each entry carries the label
    the user saw while setting it up. The default labels are a counter (``"#1"``, ``"#2"``,
    …) — and NIS restarts that counter for each point GROUP, so on a file acquired as six
    3x3 mosaics the list reads ``#1``..``#9`` six times over. That repetition is the
    microscope's own record of where one specimen ended and the next began, which is worth
    carrying even though nothing computes a coordinate from it
    (:func:`nodelab_v2.position_groups.names_agree` checks it against the geometry).

    ``[]`` unless the list covers EVERY multipoint — the rule the stage logs above already
    follow, because these are read by index and a short list names the wrong position.
    """
    names = []
    for loop in _safe(lambda: list(f.experiment), default=[]) or []:
        pts = _safe(lambda lp=loop: list(lp.parameters.points), default=None)
        if not pts:
            continue
        got = [_safe(lambda p=pt: str(p.name)) for pt in pts]
        if len(got) == n_m and all(v for v in got):
            names = got
            break
    return names


def read_nd2_metadata_extended(filepath):
    """Return a dict of all the metadata fields ND2Studios surfaces.

    Always-present keys (best-effort, may be empty):
        filepath, dim_order, sizes, dtype, height, width,
        n_timepoints, n_zslices, n_channels, n_multipoints,
        pixel_size_um, z_step_um, voxel_size_um,
        channel_names, channel_colors, channel_emission_nm,
        channel_excitation_nm, channel_exposure_ms,
        objective_name, objective_magnification, objective_na,
        objective_immersion, binning_x, binning_y,
        camera_name, microscope_name,
        acquisition_start, frame_timestamps_s, frame_time_jd,
        stage_xy_um, stage_z_um, stage_layout_source,
        z_home_index, z_bottom_to_top, bit_depth, loops

    ``stage_z_um`` / ``frame_time_jd`` / ``z_home_index`` / ``z_bottom_to_top`` /
    ``bit_depth`` are the placement vocabulary described in the module docstring. Every one
    of them is best-effort: a file that does not carry it gets an empty list or ``None``,
    never a fabricated value, because absence is what the consumers test for before they
    offer an absolute coordinate.
    """
    # The one edit to this vendored file: the bare `import nd2` goes through the shim door
    # (:mod:`nodelab_v2.nd2_compat`) so a file the SDK's own parser cannot open — a
    # zero-range ZStackLoop divides by zero in nd2 <= 0.11.3 — reaches this reader at all.
    # It must be HERE and not only in the caller: `read_calibration` calls this function
    # BEFORE its own nd2 import, so this is the first ND2File opened on a calibration read.
    # Not a re-derivation of the parse below, which stays verbatim.
    from nodelab_v2.nd2_compat import import_nd2
    nd2 = import_nd2()

    out = {"filepath": filepath}
    with nd2.ND2File(filepath) as f:
        sizes = dict(f.sizes)
        out["sizes"] = sizes
        out["dim_order"] = list(sizes.keys())
        out["dtype"] = str(f.dtype)
        out["height"] = sizes.get("Y", 0)
        out["width"] = sizes.get("X", 0)
        out["n_timepoints"] = sizes.get("T", 1)
        out["n_zslices"] = sizes.get("Z", 1)
        out["n_channels"] = sizes.get("C", 1)
        out["n_multipoints"] = sizes.get("P", sizes.get("M", 1))

        vox = _safe(f.voxel_size)
        if vox is not None:
            out["pixel_size_um"] = float(getattr(vox, "x", 1.0))
            out["z_step_um"] = float(getattr(vox, "z", 1.0))
            out["voxel_size_um"] = (
                float(getattr(vox, "z", 1.0)),
                float(getattr(vox, "y", out["pixel_size_um"])),
                float(getattr(vox, "x", out["pixel_size_um"])),
            )
        else:
            out["pixel_size_um"] = 1.0
            out["z_step_um"] = 1.0
            out["voxel_size_um"] = (1.0, 1.0, 1.0)

        meta = _safe(lambda: f.metadata)
        channels = _safe(lambda: list(meta.channels), default=[]) or []
        names = []
        colors = []
        emission = []
        excitation = []
        exposure_ms = []
        for ch in channels:
            ch_inner = getattr(ch, "channel", ch)
            names.append(str(getattr(ch_inner, "name", "Ch")))
            colors.append(_safe(lambda c=ch_inner: int(getattr(c, "colorRGB", None))))
            emission.append(_safe(lambda c=ch_inner: float(getattr(c, "emissionLambdaNm", None))))
            excitation.append(_safe(lambda c=ch_inner: float(getattr(c, "excitationLambdaNm", None))))
            exp = (
                _safe(lambda c=ch: float(getattr(c, "exposureTimeMs", None)))
                or _safe(lambda c=ch_inner: float(getattr(c, "exposureTimeMs", None)))
            )
            exposure_ms.append(exp)
        if not names:
            names = [f"Ch{i}" for i in range(out["n_channels"])]
        out["channel_names"] = names
        out["channel_colors"] = colors
        out["channel_emission_nm"] = emission
        out["channel_excitation_nm"] = excitation
        out["channel_exposure_ms"] = exposure_ms

        microscope = _safe(lambda: meta.channels[0].microscope) if channels else None
        if microscope is not None:
            out["objective_name"] = str(_safe(lambda: microscope.objectiveName) or "")
            out["objective_magnification"] = _safe(lambda: float(microscope.objectiveMagnification))
            out["objective_na"] = _safe(lambda: float(microscope.objectiveNumericalAperture))
            out["objective_immersion"] = str(_safe(lambda: microscope.immersionRefractiveIndex) or "")
        else:
            out["objective_name"] = ""
            out["objective_magnification"] = None
            out["objective_na"] = None
            out["objective_immersion"] = ""

        instrument = _safe(lambda: meta.channels[0].volume) if channels else None
        out["binning_x"] = _safe(lambda: int(instrument.cameraTransformationMatrix.binning))
        out["binning_y"] = out.get("binning_x")
        camera = _safe(lambda: meta.channels[0].instrument) if channels else None
        out["camera_name"] = str(_safe(lambda: camera.cameraName) or "")
        out["microscope_name"] = str(_safe(lambda: microscope.systemName) or "") if microscope else ""

        seq_dims = [d for d in out["dim_order"] if d not in ("Y", "X")]
        m_axis = "P" if "P" in seq_dims else ("M" if "M" in seq_dims else None)
        n_m = sizes.get(m_axis, 1) if m_axis else 1
        n_t = out["n_timepoints"]

        def _flat(coords: Dict[str, int]) -> int:
            idx = 0
            for d in seq_dims:
                idx = idx * sizes.get(d, 1) + int(coords.get(d, 0))
            return idx

        def _read_xy_from_experiment():
            """Planned stage positions from ``f.experiment`` (XYPosLoop) →
            ``(xy_pairs, z_values)``.

            The Z rides on the very same ``Position.stagePositionUm`` the XY comes from, so
            collecting it here is free — and it is the ONLY route on a file whose positions
            were planned, because the per-frame fallback below never runs then. ``z`` is
            gathered positionally: a point that has XY but no Z contributes ``None``, and
            the completeness check after the call drops a partial list rather than letting
            index *m* address the wrong position's focus.
            """
            pts: List[Tuple[float, float]] = []
            zs: List = []
            for loop in (_safe(lambda: f.experiment, default=[]) or []):
                ltype = str(_safe(lambda lp=loop: lp.type) or "")
                if "XYPos" not in ltype:
                    continue
                points = (
                    _safe(lambda lp=loop: list(lp.parameters.points), default=[]) or []
                )
                for pt in points:
                    sx = _safe(lambda p=pt: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pt: float(p.stagePositionUm.y))
                    sz = _safe(lambda p=pt: float(p.stagePositionUm.z))
                    if sx is not None and sy is not None:
                        pts.append((sx, sy))
                        zs.append(sz)
                if pts:
                    return pts, zs
            return [], []

        # 1) Per-M stage positions — try f.experiment XYPosLoop first, then
        #    fall back to frame_metadata() per-M (which requires correct flat-
        #    index arithmetic). The experiment loop is the authoritative planned
        #    positions; frame_metadata is the actual per-frame readback.
        stage_xy, stage_z = _read_xy_from_experiment()
        xy_source_method = "experiment" if stage_xy else "frame_metadata"

        if not stage_xy and m_axis is not None:
            for m in range(n_m):
                coords = {m_axis: m, "T": 0, "Z": 0, "C": 0}
                seq_idx = _flat(coords)
                fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
                if fm is None:
                    continue
                pos = (
                    _safe(lambda meta=fm: meta.channels[0].position) or
                    _safe(lambda meta=fm: meta.position)
                )
                if pos is None:
                    continue
                sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                sz = _safe(lambda p=pos: float(p.stagePositionUm.z))
                if sx is not None and sy is not None:
                    # z appended INSIDE the xy guard (possibly None), so index m addresses
                    # the same position in both lists — appending it separately let a
                    # position with z but no xy slide every later focus by one.
                    stage_xy.append((sx, sy))
                    stage_z.append(sz)
        elif not stage_xy:
            # Single-M file — record one nominal position from frame 0.
            fm0 = _safe(lambda: f.frame_metadata(0))
            if fm0 is not None:
                pos = (
                    _safe(lambda meta=fm0: meta.channels[0].position) or
                    _safe(lambda meta=fm0: meta.position)
                )
                if pos is not None:
                    sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                    sz = _safe(lambda p=pos: float(p.stagePositionUm.z))
                    if sx is not None and sy is not None:
                        stage_xy.append((sx, sy))
                        stage_z.append(sz)

        n_xy_from_stage = len(stage_xy)

        # A focus list that does not cover every multipoint, or that has a hole in it, is
        # DROPPED rather than carried: `stage_z_um[m]` is addressed by index, so a short or
        # gappy list silently reports some other position's focus. Same rule as
        # `nodegraph.nodes._stitch_stage_xy` applies to the XY log, and for the same reason
        # — a partial answer here looks exactly like a complete one.
        if len(stage_z) != n_m or any(v is None for v in stage_z):
            stage_z = []
        else:
            stage_z = [float(v) for v in stage_z]

        # 2) Per-T frame times — the per-file RELATIVE clock and the absolute one.
        #
        # `relativeTimeMs` is measured from each file's own origin, so two files' values are
        # NOT comparable: on the WellA3 pair the origins sit 1207.3 s apart, which makes a
        # naive cross-file "nearest timestamp" match land a whole acquisition cycle off.
        # `absoluteJulianDayNumber` is the shared wall clock and the only honest basis for
        # matching T across files; both are carried so a within-file consumer (dt_s) keeps
        # the cheap one.
        frame_ts: List[float] = []
        frame_jd: List[float] = []
        for t in range(n_t):
            coords = {"T": t, "Z": 0, "C": 0}
            if m_axis is not None:
                coords[m_axis] = 0
            seq_idx = _flat(coords)
            fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
            if fm is None:
                continue
            ts = (
                _safe(lambda meta=fm: float(meta.channels[0].time.relativeTimeMs)) or
                _safe(lambda meta=fm: float(meta.relativeTimeMs))
            )
            if ts is not None:
                frame_ts.append(ts / 1000.0)
            jd = _safe(
                lambda meta=fm: float(meta.channels[0].time.absoluteJulianDayNumber))
            if jd is None:
                jd = _safe(lambda meta=fm: float(meta.absoluteJulianDayNumber))
            if jd is not None:
                frame_jd.append(jd)
        if len(frame_jd) != n_t:
            frame_jd = []            # partial clock is no clock — same rule as stage_z

        if n_xy_from_stage == n_m and n_m > 0:
            stage_layout_source = f"stage_xy:{xy_source_method}"
        elif n_xy_from_stage > 0:
            stage_layout_source = f"partial:{n_xy_from_stage}/{n_m}:{xy_source_method}"
        else:
            stage_layout_source = "missing"

        out["frame_timestamps_s"] = frame_ts
        out["frame_time_jd"] = frame_jd
        # The same clock, readable: one ``YYYY-MM-DD HH:MM:SS.mmm`` per timepoint
        # (2026-10-02). Derived from the Julian day here, once, so every consumer — the
        # Viewer's timestamp, a table export, a Timeseries Builder's resolved order — shows
        # the identical text for a frame rather than each re-deriving it.
        from nodegraph.placement import jd_to_datetime_text
        out["frame_datetime"] = [jd_to_datetime_text(j) for j in frame_jd]
        out["stage_xy_um"] = stage_xy
        out["stage_z_um"] = stage_z
        out["stage_layout_source"] = stage_layout_source
        out["position_name"] = _point_names(f, n_m)

        # 3) Z-stack anchoring. `stagePositionUm.z` is CONSTANT down a stack — it is the
        # position's nominal focus, not a per-slice reading — so on its own it cannot say
        # where slice k sits. The ZStackLoop supplies the missing half: `homeIndex` is the
        # slice that nominal Z names, and `bottomToTop` its direction, giving
        #     z_um(k) = stage_z_um[m] + (k - home) * z_step_um     (bottomToTop)
        # Absent (a file with no Z loop, or an SDK that did not fill it in) both stay None
        # and a consumer must decline to place the stack rather than assume slice 0.
        z_home = None
        z_b2t = None
        for lp in (_safe(lambda: f.experiment, default=[]) or []):
            if "ZStack" not in str(_safe(lambda q=lp: q.type) or ""):
                continue
            z_home = _safe(lambda q=lp: int(q.parameters.homeIndex))
            z_b2t = _safe(lambda q=lp: bool(q.parameters.bottomToTop))
            break
        out["z_home_index"] = z_home
        out["z_bottom_to_top"] = z_b2t

        # 4) Significant sensor depth (12 on the lab's files, not the 16 bits it occupies).
        # `nodelab_v2.ingest.read_calibration` reads this off `f.attributes` itself; it is
        # reported here too so the two readers cannot disagree about the same file.
        attrs = _safe(lambda: f.attributes)
        bits = None
        if attrs is not None:
            for name in ("bitsPerComponentSignificant", "bitsPerComponentInMemory"):
                bits = _safe(lambda n=name: int(getattr(attrs, n, None)))
                if bits:
                    break
        out["bit_depth"] = bits or None

        # The human-readable acquisition date, straight from the text block. Second
        # resolution and free-form, so it labels a readout but never drives arithmetic —
        # `frame_time_jd` is the field to compute with.
        out["acquisition_start"] = str(
            _safe(lambda: (f.text_info or {}).get("date")) or "")
        out["loops"] = [
            {
                "type": str(_safe(lambda lp=lp: lp.type) or ""),
                "count": int(_safe(lambda lp=lp: lp.count) or 0),
            }
            for lp in (_safe(lambda: f.experiment, default=[]) or [])
        ]

    return out


__all__ = ["read_nd2_metadata_extended"]
