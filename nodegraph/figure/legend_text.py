"""The legend / methods sentence a figure's sources already know (V5.00 step 1).

The community checklists ask a figure legend to state the objective and its NA, the
pixel size, the z step, the frame interval, the channels' excitation and emission, the
scale bars, the projections and the display ranges. Every one of those is on the
envelope of the Page Output the image came from — the ND2 acquisition record
(``acquisition_info``, ``channel_acquisition``, both written by
``nodelab_v2.nd2_acquisition.acquisition_record``), ``pixel_size_um``, ``z_step_um``,
``dt_s``, ``channel_names``, ``channel_emission_nm`` — so the sentence is generated, and a
row is only written for something the file actually recorded.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.figure import schema as _S

#: legend field → the acquisition-record labels that carry it (matched case-insensitively,
#: as substrings, first hit wins; the record's own labels vary with the file's wording)
FIELD_LABELS: Dict[str, Tuple[str, ...]] = {
    "modality": ("modality",),
    "objective": ("objective",),
    "magnification": ("magnification",),
    "na": ("numerical aperture",),
    "ri": ("immersion refractive index", "refractive index"),
    "pinhole": ("pinhole",),
    "zoom": ("zoom",),
    "detector": ("camera / detector", "camera", "detector"),
    "z_stack": ("z stack",),
    "z_step": ("z step",),
    "time_loop": ("time loop",),
}

CHANNEL_LABELS: Dict[str, Tuple[str, ...]] = {
    "excitation": ("excitation",),
    "emission": ("emission",),
    "exposure": ("exposure",),
    "power": ("power", "laser"),
}


def _md_axes(env: Any) -> Tuple[Mapping[str, Any], Any]:
    """``(metadata, axes)`` of an envelope given as a MetaEnvelope, a Dataset, a
    ``(axes, metadata)`` pair or a bare metadata mapping."""
    if isinstance(env, Mapping):
        return env, None
    if isinstance(env, (tuple, list)) and len(env) == 2:
        return (env[1] or {}), env[0]
    md = getattr(env, "metadata", None)
    return (md or {}), getattr(env, "axes", None)


def _rows(section_rows: Any) -> List[Tuple[str, str]]:
    out = []
    for r in section_rows or []:
        if isinstance(r, (list, tuple)) and len(r) == 2:
            out.append((str(r[0]), str(r[1])))
    return out


def acquisition_fields(md: Mapping[str, Any]) -> Dict[str, str]:
    """The legend fields found in ``md["acquisition_info"]`` (``[[section, rows], …]``),
    keyed by :data:`FIELD_LABELS`' names. Missing fields are simply absent."""
    found: Dict[str, str] = {}
    rows: List[Tuple[str, str]] = []
    for sec in md.get("acquisition_info") or []:
        try:
            title, srows = sec
        except (TypeError, ValueError):
            continue
        rows.extend(_rows(srows))
        rows.append((str(title), ""))
    for key, labels in FIELD_LABELS.items():
        for label, value in rows:
            low = label.lower()
            if any(l in low for l in labels) and value:
                found[key] = value
                break
    return found


def channel_fields(md: Mapping[str, Any], c: int) -> Dict[str, str]:
    """The legend fields of channel ``c``: its name, emission (nm), and whatever the
    per-channel acquisition rows carry (excitation, emission, exposure, power)."""
    out: Dict[str, str] = {}
    names = list(md.get("channel_names") or [])
    if c < len(names) and names[c]:
        out["name"] = str(names[c])
    ems = list(md.get("channel_emission_nm") or [])
    if c < len(ems) and ems[c]:
        out["emission_nm"] = str(ems[c])
    per = list(md.get("channel_acquisition") or [])
    if c < len(per):
        rows = _rows(per[c])
        for key, labels in CHANNEL_LABELS.items():
            for label, value in rows:
                if any(l in label.lower() for l in labels) and value:
                    out[key] = value
                    break
    return out


def _num(v: Any, digits: int = 3) -> str:
    """``v`` with ``digits`` significant figures and no exponent in the range a legend
    meets (``2400``, ``0.25``, ``30``), scientific only past 1e6 or under 1e-3."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    if abs(f) >= 1e6 or abs(f) < 1e-3:
        return f"{f:.{digits}g}"
    import math as _m
    dec = max(0, digits - int(_m.floor(_m.log10(abs(f)))) - 1)
    s = f"{f:.{dec}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _join(parts: Sequence[str], sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)


def legend_sentence(doc: Mapping[str, Any], envelopes: Mapping[str, Any], *,
                    lut_windows: Optional[Mapping[str, Sequence[Tuple[int, float, float]]]]
                    = None) -> str:
    """The legend paragraph for ``doc`` (a normalized figure document): what the images
    show and where they came from, the acquisition, the channels, the projections, the
    scale bars, the display ranges (``lut_windows``: ``{item id: [(c, lo, hi), …]}`` for
    windows resolved at render time; manual channels need none), the overlays and the
    charts. Sentences are only written for what is known."""
    sources = {s["id"]: s for s in doc.get("sources") or []}
    items = list(doc.get("items") or [])
    sentences: List[str] = []

    # which sources the images show
    image_items = [it for it in items if it.get("kind") in ("image", "montage")]
    shown: List[str] = []
    for it in image_items:
        img = _S.effective_image(doc, it["id"])
        sid = img.get("source")
        if sid in sources:
            label = sources[sid]["label"]
            letter = it.get("letter") or ""
            shown.append(f"{letter}: {label}" if letter else label)
    if shown:
        sentences.append("Images show " + _join(shown, "; ") + ".")

    # acquisition, from the first image source that has a record
    acq_md: Optional[Mapping[str, Any]] = None
    acq_axes = None
    for it in image_items:
        sid = _S.effective_image(doc, it["id"]).get("source")
        if sid in envelopes:
            md, axes = _md_axes(envelopes[sid])
            if acq_md is None:
                acq_md, acq_axes = md, axes
            if md.get("acquisition_info"):
                acq_md, acq_axes = md, axes
                break
    if acq_md is not None:
        f = acquisition_fields(acq_md)
        bits: List[str] = []
        if f.get("objective"):
            obj = f["objective"]
            extra = _join([f"NA {f['na']}" if f.get("na") else "",
                           f"immersion RI {f['ri']}" if f.get("ri") else ""], "; ")
            bits.append(f"a {obj} objective" + (f" ({extra})" if extra else ""))
        elif f.get("magnification"):
            bits.append(f"{f['magnification']} magnification")
        if f.get("pinhole"):
            bits.append(f"pinhole {f['pinhole']}")
        if f.get("zoom"):
            bits.append(f"zoom {f['zoom']}")
        if f.get("detector"):
            bits.append(f"{f['detector']}")
        lead = "Acquired" + (f" by {f['modality']}" if f.get("modality") else "")
        if bits:
            sentences.append(f"{lead} with {_join(bits)}.")
        samp: List[str] = []
        if acq_md.get("pixel_size_um"):
            samp.append(f"pixel size {_num(acq_md['pixel_size_um'])} µm")
        nz = int(getattr(acq_axes, "z", 1) or 1) if acq_axes is not None else 1
        if acq_md.get("z_step_um") and nz > 1:
            samp.append(f"z step {_num(acq_md['z_step_um'])} µm ({nz} planes)")
        nt = int(getattr(acq_axes, "t", 1) or 1) if acq_axes is not None else 1
        if acq_md.get("dt_s") and nt > 1:
            samp.append(f"frames every {_num(acq_md['dt_s'])} s ({nt} frames)")
        elif nt > 1:
            samp.append(f"{nt} frames")
        if samp:
            sentences.append(_join(samp).capitalize() + ".")
        nc = int(getattr(acq_axes, "c", 1) or 1) if acq_axes is not None else \
            len(acq_md.get("channel_names") or [])
        chans: List[str] = []
        for c in range(nc):
            cf = channel_fields(acq_md, c)
            if not cf:
                continue
            name = cf.get("name", f"channel {c + 1}")
            spec = _join([f"excitation {cf['excitation']}" if cf.get("excitation") else "",
                          f"emission {cf['emission']}" if cf.get("emission")
                          else (f"emission {cf['emission_nm']} nm" if cf.get("emission_nm")
                                else "")], " / ")
            chans.append(f"{name}" + (f" ({spec})" if spec else ""))
        if chans:
            sentences.append("Channels: " + _join(chans, "; ") + ".")

    # projections and display
    proj: List[str] = []
    ranges: List[str] = []
    bars: List[str] = []
    overlays: List[str] = []
    for it in image_items:
        img = _S.effective_image(doc, it["id"])
        letter = it.get("letter") or it["id"]
        z = img.get("frame", {}).get("z")
        if z == "max":
            proj.append(f"{letter}: maximum intensity projection")
        elif z == "mean":
            proj.append(f"{letter}: mean projection")
        if img.get("z_mode") == "depth_coded":
            proj.append(f"{letter}: depth colour-coded")
        if img.get("t_mode") == "temporal_code":
            proj.append(f"{letter}: time colour-coded")
        sb = (img.get("decor") or {}).get("scale_bar_um")
        if sb:
            bars.append(f"{_num(sb)} µm")
        win = list((lut_windows or {}).get(it["id"]) or [])
        if not win:
            win = [(ch["c"], ch["lo"], ch["hi"]) for ch in img.get("channels") or []
                   if ch.get("mode") == "manual"]
        for c, lo, hi in win:
            ranges.append(f"{letter} ch{int(c) + 1} {_num(lo)}–{_num(hi)}")
        for L in img.get("layers") or []:
            kind = L.get("kind")
            if kind == "field":
                lim = L.get("limits") or {}
                col = L.get("column") or "value"
                if lim.get("mode") == "normalized":
                    overlays.append(f"{col} as a colour map, 0–1 of {_num(lim.get('vmax'))}")
                elif lim.get("mode") == "manual":
                    overlays.append(f"{col} as a colour map ({_num(lim.get('lo'))}–"
                                    f"{_num(lim.get('hi'))})")
                else:
                    overlays.append(f"{col} as a colour map")
            elif kind == "labels":
                look = L.get("look") or {}
                overlays.append("object outlines" if not look.get("fill") else "object fills")
            elif kind == "tracks":
                overlays.append("tracks")
            elif kind == "points":
                overlays.append("detections")
            elif kind == "vectors":
                overlays.append("displacement vectors")
            elif kind == "grid":
                overlays.append("the correlation grid")
            elif kind == "mesh":
                overlays.append("mesh sections")
            elif kind == "voxels":
                overlays.append(f"{L.get('layer') or 'a voxel layer'} as a wash")
    for it in items:
        if it.get("kind") == "scale_bar":
            bars.append(f"{_num(it.get('length_um'))} µm")
    if proj:
        sentences.append(_join(proj, "; ") + ".")
    if bars:
        uniq = list(dict.fromkeys(bars))
        sentences.append("Scale bars " + _join(uniq, ", ") + ".")
    if ranges:
        sentences.append("Display ranges: " + _join(ranges, "; ") + ".")
    if overlays:
        sentences.append("Overlays: " + _join(list(dict.fromkeys(overlays)), ", ") + ".")

    charts: List[str] = []
    for it in items:
        if it.get("kind") != "chart":
            continue
        ch = it.get("chart") or {}
        letter = it.get("letter") or it["id"]
        kind = ch.get("kind")
        what = kind
        if kind == "xy":
            what = f"{ch.get('y') or 'y'} against {ch.get('x') or 'x'}"
        elif kind == "timeseries":
            what = f"{ch.get('value') or 'value'} over time"
        elif kind == "distribution":
            what = f"{ch.get('sub_kind') or 'histogram'} of {ch.get('value') or 'value'}"
        elif kind == "bar":
            what = f"bars of {ch.get('value') or ch.get('y') or 'value'}"
        elif kind == "msd":
            what = "mean squared displacement"
        elif kind == "survival":
            what = "track survival"
        err = (ch.get("draw") or {}).get("error", "none")
        if err and err != "none":
            what += f", mean ± {err.upper() if err in ('sd', 'sem') else err}"
        gb = ch.get("group_by")
        if gb:
            what += f" per {gb}"
        test = (ch.get("stats") or {}).get("test", "none")
        if test and test != "none":
            what += f"; {test.replace('_', '–').replace('welch', 'Welch t')} test"
        charts.append(f"{letter}: {what}")
    if charts:
        sentences.append(_join(charts, "; ") + ".")
    return " ".join(sentences)
