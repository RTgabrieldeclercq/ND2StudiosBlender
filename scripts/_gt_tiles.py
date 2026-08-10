"""Ground-truth tiles — one colour composite per M position, Z midplane, one timepoint.

The image the user hand-draws over. Three requirements shape every choice here:

1. **The colour must MEAN something**, because the annotator defaults each polygon's type to
   the colour underneath it. So the composite is a direct channel map with no artistic
   licence: ``R = c1 R-B`` (F, the dot-carrying population), ``B = c2 Nile Blue`` (I, inert),
   ``G = c0 GFP`` (the cells on the granule surfaces). A red granule *is* an F granule. The
   type therefore defaults to a measurement rather than to an impression.

2. **An empty channel must render black, not as haze.** In the pure-F fields (M01/M08/M15)
   the Nile Blue channel carries no signal at all — median 52-57 DN against a 99.5th
   percentile of 112-134, i.e. a span of ~60 DN of pure read noise, against ~400-750 DN
   where the channel is real. A plain percentile stretch normalises that noise to full scale
   and floods a pure-F field with blue, which would then be handed to a classifier that
   defaults on colour. So the low point is ``median + 2.5 sigma_MAD`` — noise clips to black
   by construction — and the span carries an absolute floor as a second guard.

3. **It has to be legible enough to draw a boundary on.** A gamma lifts the mid-tones; the
   exact numbers are recorded in the manifest so any tile can be reproduced byte-for-byte.

Z midplane (z=3 of 7) is the user's choice and it holds up: foreground fraction there is
0.452 against 0.476 at z=1 on M08, so the attenuation that guts z=5-6 has barely started.

Usage
    PYTHONUTF8=1 python scripts/_gt_tiles.py            # render tiles + build the annotator
    PYTHONUTF8=1 python scripts/_gt_tiles.py --preview  # a montage to look at first
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, ".claude", "skills", "evidence-log"))

import _bed_nodegraph as B                                              # noqa: E402

OUT = os.path.join(_ROOT, "CodeLog", "live", "gt")

#: (m index, name, what the condition is). Six positions spanning the design: the three
#: pure-F controls at S / M / L, then three mixtures so BOTH populations appear and the
#: colour default has something to be right or wrong about. These are the same six
#: conditions the analysis harness reports on, so the ground truth lands on fields that
#: already have measurements attached rather than on a fresh set nothing else has seen.
FIELDS: List[Tuple[int, str, str]] = [
    (0,  "M01", "pure F, SMALL"),
    (7,  "M08", "pure F, MEDIUM"),
    (14, "M15", "pure F, LARGE"),
    (3,  "M04", "50% F small + inert medium"),
    (10, "M11", "50% F medium + inert medium"),
    (18, "M19", "75% F large + inert large"),
]

#: how the acquisition's channels map onto the screen. Order is (channel, rgb slot, label).
CHANNEL_MAP = [(B.C_F,   0, "R-B 571 - functional granules (F)"),
               (B.C_GFP, 1, "GFP 499 - cells on the granule surfaces"),
               (B.C_I,   2, "Nile Blue 649 - inert granules (I)")]

Z_MID = 3
T_ONE = 0
#: The black point, in DN, FIXED across every field and channel.
#:
#: Two failed adaptive estimators are the reason it is a constant. `median + k*sigma_MAD`
#: put the black point at 1146 DN on M15, where half the frame is granule, and clipped the
#: brightest field in the set to solid black. The histogram MODE, which is robust to
#: foreground fraction in principle, was unstable in the densest fields for the opposite
#: reason and blacked out M08 as well. Both failures were invisible in a table of numbers
#: and obvious in the rendered montage.
#:
#: A constant is also the CORRECT choice here rather than merely the safe one: this image is
#: hand-annotated, and a per-field adaptive stretch makes the same physical brightness look
#: different from field to field, which would bias where a boundary gets drawn between
#: conditions. 90 DN sits above the void (measured at 51-59 DN in every field and channel)
#: plus its noise, and below the 200 DN Nile-Blue granule cut calibrated on the pure-F
#: controls, so no real body is clipped away.
LO_DN = 90.0
#: the display span never falls below this, so a channel carrying nothing stays dark
#: instead of having its read noise normalised up to full scale.
SPAN_FLOOR_DN = 400.0
#: a channel is reported empty when almost no pixel carries signal — a statement about the
#: DESIGN (a pure-F field has no inert population), not about the display scaling.
SIGNAL_DN = 150.0
SIGNAL_FRAC = 0.001
GAMMA = 0.72


def _stretch(a: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """One channel to [0,1] on the fixed black point, with an absolute span floor."""
    raw_hi = float(np.percentile(a, 99.7))
    hi = max(raw_hi, LO_DN + SPAN_FLOOR_DN)
    frac = float((a > LO_DN + SIGNAL_DN).mean())
    out = np.clip((a - LO_DN) / (hi - LO_DN), 0.0, 1.0) ** GAMMA
    return out, {"lo_dn": LO_DN, "hi_dn": hi, "p99_7_dn": raw_hi, "gamma": GAMMA,
                 "signal_frac": frac, "empty": bool(frac < SIGNAL_FRAC)}


def composite(m: int, *, z: int = Z_MID, t: int = T_ONE
              ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """(Y,X,3) uint8 colour composite for one position, plus the exact stretch used."""
    rgb = np.zeros(B._dask().shape[-2:] + (3,), dtype=np.float32)
    prov: Dict[str, Dict[str, float]] = {}
    for c, slot, label in CHANNEL_MAP:
        chan, p = _stretch(B.raw_plane(m, t, z, c))
        rgb[..., slot] = chan
        p["channel"] = c
        p["label"] = label
        prov["rgb"[slot]] = p
    return (rgb * 255.0 + 0.5).astype(np.uint8), prov


def render(fields: Sequence[Tuple[int, str, str]] = tuple(FIELDS), *,
           z: int = Z_MID, t: int = T_ONE, out_dir: str = OUT) -> List[Dict]:
    """Write one PNG per field and return the tile manifest."""
    from PIL import Image
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    cal = B.calibration()
    px = float(cal.get("xy_um") or cal.get("pixel_um") or 1.7183)
    tiles = []
    for m, name, what in fields:
        rgb, prov = composite(m, z=z, t=t)
        path = os.path.join(img_dir, f"{name}.png")
        Image.fromarray(rgb).save(path, optimize=True)
        empty = [k for k, v in prov.items() if v["empty"]]
        tiles.append({
            "id": name, "file": f"img/{name}.png", "m": m, "t": t, "z": z,
            "what": what, "h": int(rgb.shape[0]), "w": int(rgb.shape[1]),
            "um_per_px": px, "stretch": prov,
            "empty_channels": empty,
        })
        print(f"[tile] {name}  m={m} z={z} t={t}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"{os.path.getsize(path) / 1e6:.2f} MB  "
              f"empty:{','.join(empty) if empty else '-'}")
    return tiles


def preview(out_png: str, fields: Sequence[Tuple[int, str, str]] = tuple(FIELDS),
            *, z: int = Z_MID) -> str:
    """A montage of the composites plus one zoomed crop — LOOK at this before shipping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.color": "#e8e8ea", "axes.labelcolor": "#e8e8ea",
                         "figure.facecolor": "#14161a", "savefig.facecolor": "#14161a",
                         "font.size": 8})
    n = len(fields)
    fig, AX = plt.subplots(2, n, figsize=(2.5 * n, 5.6),
                           gridspec_kw={"height_ratios": [1, 1]})
    AX = np.atleast_2d(AX)
    for j, (m, name, what) in enumerate(fields):
        rgb, prov = composite(m, z=z)
        AX[0, j].imshow(rgb, interpolation="nearest")
        empty = [k.upper() for k, v in prov.items() if v["empty"]]
        AX[0, j].set_title(f"{name} — {what}\n"
                           f"{'empty: ' + ','.join(empty) if empty else 'both bodies present'}",
                           fontsize=7.5)
        c = 200
        AX[1, j].imshow(rgb[c:c + 240, c:c + 240], interpolation="nearest")
        AX[1, j].set_title("240 px crop — is a boundary drawable here?", fontsize=7)
    for a in AX.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Ground-truth tiles: z={z} (midplane), t=0 — "
                 f"R = R-B (F granules) · G = GFP (cells) · B = Nile Blue (I granules)",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[preview] {out_png}")
    return out_png


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true",
                    help="render the montage only, do not build the annotator")
    ap.add_argument("--z", type=int, default=Z_MID)
    ap.add_argument("--t", type=int, default=T_ONE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.preview:
        preview(os.path.join(_ROOT, "CodeLog", "live", "img", "gt_tiles.png"), z=a.z)
        return 0
    tiles = render(z=a.z, t=a.t, out_dir=a.out)
    import groundtruth as GT
    page = GT.build(
        a.out, tiles,
        title="Hand-drawn granule ground truth",
        question="Outline each granule; circle each cell.",
        json_name="granule_ground_truth.json")
    print(f"\n[annotator] {page}  ({os.path.getsize(page) / 1e6:.1f} MB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
