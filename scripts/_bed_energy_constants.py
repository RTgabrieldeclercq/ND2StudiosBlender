"""Step 1 of V2.22 — derive the energy constants from the label-measured granule size.

Every constant in the Gibbs energy is supposed to come from a physical statement rather than a
knob, and the statements are all keyed to one length: the granule radius. That radius was wrong.

The spec (`CodeLog/ClaudesPlan/V2.22_packed_object_pose.md:90`) warns that the surface-energy
penalty is **inert on this data** because `w_min = 0.15*R_bar = 1.92 um = 1.12 px` sits at the
sampling limit — "a penalty calibrated to forbid exactly that forbids nothing". That used
`R_bar = 12.8 um`, an automated EDT-at-seed estimate later **retracted** after visual QC showed
the pipeline was measuring gap centres and debris rather than granules.

The replacement is the only scale the labels can supply and the only one that was ever checked
by eye: the median area of a *label-confirmed single granule*, per condition. This script
recomputes everything from it and reports whether the surface energy is live.

Nothing here is fitted. Every line is an identity from the spec plus one measured area.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _bed_nodegraph as B                                            # noqa: E402

#: the retracted estimate, kept only so the correction is visible rather than asserted
R_BAR_RETRACTED_UM = 12.8

#: max tolerated apparent overlap between two bodies, spec :246 / :1005
O_MAX = 0.15
#: protrusion radius forbidden, as a fraction of the body radius, spec :147
W_MIN_FRAC = 0.15
#: candidate-set cap per voxel, spec :170 — a jammed bed has contact number 6-12
M_CAND = 24
#: annealing stages and ratio, spec :154
N_STAGES, Q = 6, 0.7


def psf_sigma_um(px_um: float, emission_nm: float, na: float) -> dict:
    """Effective PSF sigma in um, and why the sampling term dominates here.

    Two contributions, added in quadrature because they are independent blurs:

      optical   FWHM = 0.51*lambda/NA, so sigma_opt = FWHM/2.355
      sampling  a square pixel of width p integrates the field, contributing p/sqrt(12)

    On this acquisition the pixel is 1.718 um against a Nyquist requirement of 0.34-0.44 um, so
    the grid is 3.9-5.1x undersampled and the SAMPLING term is the larger of the two. That is
    the same fact behind the standing rule that sub-voxel boundary claims on this data are
    meaningless, so the effective sigma is what the energy must use — a diffraction-only sigma
    would make `|s|_gain = 3*sigma_PSF` sub-pixel and quietly turn the surface term off.
    """
    fwhm_opt = 0.51 * emission_nm / na / 1000.0
    sig_opt = fwhm_opt / 2.3548
    sig_pix = px_um / np.sqrt(12.0)
    sig_eff = float(np.hypot(sig_opt, sig_pix))
    return {"sigma_optical_um": float(sig_opt), "sigma_pixel_um": float(sig_pix),
            "sigma_eff_um": sig_eff, "fwhm_optical_um": float(fwhm_opt),
            "rayleigh_um": float(0.61 * emission_nm / na / 1000.0),
            "sampling_dominates": bool(sig_pix > sig_opt)}


def constants(px_um: float, sig_psf: float, r_eq_um: float) -> dict:
    """The spec's identities, evaluated for one body radius."""
    w_min = W_MIN_FRAC * r_eq_um                    # :147  protrusion RADIUS, not width
    s_gain = 3.0 * sig_psf                          # :147
    gamma = w_min * s_gain / 2.0                    # :147  the Potts weight
    m0 = 1.5 * sig_psf                              # :981  background class energy
    ell = sig_psf                                   # :134  dimensionless intensity -> um
    cand_r = 2.0 * r_eq_um + 2.0 * sig_psf          # :170  kD-tree candidate radius
    return {
        "r_eq_um": r_eq_um,
        "w_min_um": w_min, "w_min_px": w_min / px_um,
        "w_min_diameter_px": 2.0 * w_min / px_um,
        "s_gain_um": s_gain, "s_gain_px": s_gain / px_um,
        "gamma_tilde_um2": gamma,
        "m0_um": m0, "ell_um": ell,
        "r_hard_um_equal_pair": 2.0 * r_eq_um * (1.0 - O_MAX),   # :1005
        "r0_um_equal_pair": 2.0 * r_eq_um,
        "r1_um_equal_pair": 1.3 * 2.0 * r_eq_um,
        "candidate_radius_um": cand_r,
        "candidate_radius_px": cand_r / px_um,
    }


def main() -> int:
    if not os.path.exists(B.ND2):
        print(f"SKIP: sample not found: {B.ND2}")
        return 0
    cal = B.calibration()
    px = float(cal["pixel_size_um"])
    emis = [e for e in (cal.get("channel_emission_nm") or []) if e]
    lam = float(emis[1]) if len(emis) > 1 else 571.0        # the R-B body channel
    na = float(cal.get("objective_na") or 0.45)

    print("=" * 78)
    print("V2.22 STEP 1 — energy constants from the label-measured granule size")
    print("=" * 78)
    print(f"pixel_size_um {px:.6f}   emission {lam:.0f} nm   NA {na:.2f}")

    p = psf_sigma_um(px, lam, na)
    print(f"\nPSF: optical sigma {p['sigma_optical_um']:.3f} um "
          f"(FWHM {p['fwhm_optical_um']:.3f}, Rayleigh {p['rayleigh_um']:.3f})")
    print(f"     pixel  sigma {p['sigma_pixel_um']:.3f} um   "
          f"sampling dominates: {p['sampling_dominates']}")
    print(f"     EFFECTIVE sigma_PSF = {p['sigma_eff_um']:.3f} um "
          f"= {p['sigma_eff_um']/px:.2f} px    <- used below")
    sig = p["sigma_eff_um"]

    _by, a_single = B.load_labels()
    rows = []
    print(f"\n{'cond':5s} {'A_single µm²':>13s} {'r_eq µm':>8s} {'w_min µm':>9s} "
          f"{'w_min px':>9s} {'jut width px':>13s} {'γ̃ µm²':>8s} {'|s|gain µm':>11s} "
          f"{'m₀ µm':>7s} {'cand r µm':>10s}")
    for cond in ("M01", "M08", "M15"):
        r_eq = float(np.sqrt(a_single[cond] / np.pi))
        c = constants(px, sig, r_eq)
        c["cond"] = cond
        c["a_single_um2"] = a_single[cond]
        rows.append(c)
        print(f"{cond:5s} {a_single[cond]:13.0f} {r_eq:8.2f} {c['w_min_um']:9.2f} "
              f"{c['w_min_px']:9.2f} {c['w_min_diameter_px']:13.2f} "
              f"{c['gamma_tilde_um2']:8.2f} {c['s_gain_um']:11.2f} {c['m0_um']:7.2f} "
              f"{c['candidate_radius_um']:10.1f}")

    # ── the verdict the whole step exists for ─────────────────────────────────
    ret = constants(px, sig, R_BAR_RETRACTED_UM)
    w_px = np.array([r["w_min_px"] for r in rows])
    live = bool(w_px.min() >= 2.0)
    print("\n" + "-" * 78)
    print("IS THE SURFACE ENERGY LIVE?")
    print("-" * 78)
    print(f"  spec's warning used the RETRACTED R̄ = {R_BAR_RETRACTED_UM} µm")
    print(f"    -> w_min {ret['w_min_um']:.2f} µm = {ret['w_min_px']:.2f} px "
          f"(forbidden jut {ret['w_min_diameter_px']:.2f} px across) — at the sampling limit, "
          f"so inert")
    print(f"  label-measured radii give w_min {w_px.min():.2f}-{w_px.max():.2f} px "
          f"(forbidden jut {2*w_px.min():.2f}-{2*w_px.max():.2f} px across)")
    print(f"    -> ALL conditions >= 2 px: {live}   "
          f"{'the term BITES' if live else 'STILL INERT — raise W_MIN_FRAC or set it in px'}")
    print(f"  ratio to the retracted value: {w_px.mean()/ret['w_min_px']:.2f}x")
    print(f"\n  smallest condition is the marginal one: M01 at {w_px.min():.2f} px. Worth")
    print(f"  watching — if the 'surface energy measurably matters' test fails anywhere it")
    print(f"  will fail there first, and the remedy is an absolute floor in px, not a smaller")
    print(f"  fraction.")

    print(f"\n  self-consistency: |s|_gain = 3·σ_PSF = {3*sig:.2f} µm = {3*sig/px:.2f} px, i.e.")
    print(f"  a protrusion must earn about one voxel of signed distance to pay for its surface.")

    # ── what is NOT derivable yet, stated rather than invented ────────────────
    print("\n" + "-" * 78)
    print("DEFERRED — these need a fit residual and cannot be set here")
    print("-" * 78)
    print("  σ_rough   robust std of the convex-polytope fit residual        -> step 4")
    print("  τ = 3σ_rough        the shape-slack hinge width (spec :150)     -> step 4")
    print("  T ≈ σ_PSF + σ_rough  the reported softness (spec :134)          -> step 4")
    print("  λ_I       calibrated so no intensity term exceeds one voxel of")
    print(f"            boundary motion, i.e. <= {px:.3f} µm (spec :558)      -> step 6")
    print("  A one-voxel budget is exact here because |∇s| = 1 for a true signed distance, so")
    print("  an energy imbalance of ΔE µm displaces the boundary by exactly ΔE µm.")

    out = {"pixel_size_um": px, "emission_nm": lam, "objective_na": na, "psf": p,
           "o_max": O_MAX, "w_min_frac": W_MIN_FRAC, "m_cand": M_CAND,
           "n_stages": N_STAGES, "q": Q,
           "per_condition": rows, "retracted": ret,
           "surface_energy_live": live,
           "deferred": ["sigma_rough", "tau", "T", "lambda_I"]}
    os.makedirs(B.CACHE, exist_ok=True)
    path = os.path.join(B.CACHE, "energy_constants.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(f"\n[wrote] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
