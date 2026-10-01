"""Suitability of Xe II lines for an ne/Te diagnostic: the numbers behind docs/crm/lines_final.md.

For every Xe II line in [--wmin, --wmax] (plus --extra wavelengths) it reports

* the measured line: Gaussian fit (amplitude, FWHM, S/B) in each sheet of the
  spectrum, saturation (ADC limit), lines of Xe I/II/III, Ba I/II, W I/II,
  Ca I/II, Al I/II (NIST) within +-1.5 FWHM;
* cross sections (Wang et al 2019 supplementary, CrossSectionsIon.xlsx): the
  initial states with data for the upper level, threshold, energy and value of
  the maximum of sigma(1 -> u);
* Te sensitivity: d ln k / d ln Te of the Maxwellian rate from the ground
  state between 1 and 2 eV (computed here from Wang's sigma);
* ne sensitivity: metastable lever sum_m k(m -> u) / k(1 -> u) at Te (Wang's
  sigma), and electron quenching of the upper level: k_q = sum over lower
  levels of the de-excitation rate (detailed balance from Wang's excitation
  sigma; collisions to higher levels are not in Wang and not included);
  n_crit = sum of the NIST A of the upper level / k_q (a lower bound, because
  the NIST A are incomplete);
* A-values: E1-allowed lower levels of the upper level in Wang's level table
  (even parity, below, |dJ| <= 1) and how many of them have an A in NIST.

    python examples/crm/line_criteria.py --wang CrossSectionsIon.xlsx --spectrum A25_second.xlsx \
        --wmin 541.647 --wmax 565 --extra 575.103,610.143,461.550,497.271 --out docs/crm/criteria.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pydbsr.rates import rate_from_sigma
from pydbsr.reference import read_xlsx

CM_PER_EV = 8065.544
SPECIES = ["XeI", "XeII", "XeIII", "BaI", "BaII", "WI", "WII", "CaI", "CaII", "AlI", "AlII"]


def _clean(s):
    return (s or "").replace('"', "").lstrip("=").strip()


def _num(s):
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", s or ""))
    except ValueError:
        return None


def _iv(s):
    m = re.match(r"\d+", s or "")
    return int(m.group()) if m else 0


def read_nist(path, sp):
    out = []
    for d in csv.DictReader(open(path, newline="")):
        d = {k: _clean(v) for k, v in d.items() if k}
        wl = _num(d.get("obs_wl_air(nm)")) or _num(d.get("ritz_wl_air(nm)")) or _num(d.get("obs_wl_vac(nm)")) \
            or _num(d.get("ritz_wl_vac(nm)"))
        if wl:
            out.append(dict(sp=sp, wl=wl, I=d.get("intens", ""), A=_num(d.get("Aki(s^-1)")),
                            Ei=_num(d.get("Ei(cm-1)")), Ek=_num(d.get("Ek(cm-1)"))))
    return out


def read_sheets(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        a = np.array([r[:2] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None], float)
        out[ws.title] = (a[:, 0], a[:, 1])
    return out


def fit_line(w, I, wl, shift=(-0.13, 0.02), sat=64000.0):
    """Gaussian + constant around the strongest point in [wl+shift]: (amp, centre, fwhm, S/B, saturated)."""
    m = (w > wl + shift[0]) & (w < wl + shift[1])
    if not m.any():
        return None
    k = np.flatnonzero(m)[np.argmax(I[m])]
    sel = slice(max(k - 4, 0), k + 5)
    x, y = w[sel], I[sel]
    bg = float(np.percentile(I[(w > w[k] - 1.5) & (w < w[k] + 1.5)], 25))
    try:
        p, _ = curve_fit(lambda x, a, c, s: a * np.exp(-0.5 * ((x - c) / s) ** 2) + bg, x, y,
                         p0=[I[k] - bg, w[k], 0.0104], maxfev=5000)
        amp, c, fw = p[0], p[1], 2.3548 * abs(p[2])
    except RuntimeError:
        amp, c, fw = I[k] - bg, w[k], float("nan")
    near = (w > w[k] - 1.0) & (w < w[k] + 1.0)
    d = np.diff(I[near])
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)      # robust point-to-point noise
    return dict(amp=amp, centre=c, fwhm=fw, snr=(I[k] - bg) / bg, sn=amp / noise if noise > 0 else float("nan"),
                saturated=bool(I[k] >= sat), peak=float(I[k]))


def fit_doublet(w, I, wl1, wl2, half=0.2):
    """Two Gaussians with a common width, all parameters free, + constant, around a
    blend of NIST lines wl1 < wl2 (the measured scale is shifted, so the window
    starts 0.15 nm below wl1).  Returns separation, FWHM, amplitudes with errors."""
    m = (w > wl1 - 0.15) & (w < wl2 + half)
    x, y = w[m], I[m]
    G = lambda x, a, c, s: a * np.exp(-0.5 * ((x - c) / s) ** 2)
    f = lambda x, a1, a2, c, d, s, b: G(x, a1, c, s) + G(x, a2, c + d, s) + b
    p0 = [y.max() - y.min(), (y.max() - y.min()) / 5, x[np.argmax(y)], wl2 - wl1, 0.0104, y.min()]
    p, cv = curve_fit(f, x, y, p0=p0, maxfev=20000)
    e = np.sqrt(np.diag(cv))
    res = y - f(x, *p)
    return dict(sep_pm=p[3] * 1e3, sep_err_pm=e[3] * 1e3, fwhm_pm=2.3548 * abs(p[4]) * 1e3,
                amp1=p[0], amp1_err=e[0], amp2=p[1], amp2_err=e[1], ratio=p[1] / p[0],
                max_resid_rel=float(np.max(np.abs(res)) / p[0]), n_points=len(x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wang", required=True)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--nist-dir", default="docs/crm")
    ap.add_argument("--wmin", type=float, default=541.647)
    ap.add_argument("--wmax", type=float, default=565.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--Te", type=float, default=1.1, help="Te for the metastable lever and quenching, eV")
    ap.add_argument("--fwhm", type=float, default=0.0245, help="instrumental FWHM, nm (blend window = 1.5 FWHM)")
    ap.add_argument("--ba", default="BaI:553.548,BaII:455.403,BaII:493.408,BaII:614.171,BaII:585.367,BaII:649.690",
                    help="Ba lines to check (visibility, blends): species:wavelength,...")
    ap.add_argument("--doublet", default="545.045,545.090", help="blend fitted with two Gaussians")
    ap.add_argument("--out", default="criteria.csv")
    a = ap.parse_args(argv)

    ref = read_xlsx(a.wang)
    lv = ref.levels
    elv = np.array([x.energy_cm for x in lv])

    def no(e):
        if e is None:
            return None
        k = int(np.argmin(abs(elv - e)))
        return k + 1 if abs(elv[k] - e) < 0.3 else None

    nist = {sp: read_nist(Path(a.nist_dir) / f"nist_{sp}.csv", sp) for sp in SPECIES
            if (Path(a.nist_dir) / f"nist_{sp}.csv").exists()}
    xe2 = list(nist["XeII"])
    for extra in ("nist_XeII_UV.csv", "nist_XeII_IR.csv"):
        if (Path(a.nist_dir) / extra).exists():
            xe2 += read_nist(Path(a.nist_dir) / extra, "XeII")
    for x in xe2:
        x["u"], x["l"] = no(x["Ek"]), no(x["Ei"])
    sheets = read_sheets(a.spectrum)
    extra = [float(v) for v in a.extra.split(",") if v.strip()]
    targets = [x for x in nist["XeII"] if a.wmin <= x["wl"] <= a.wmax or any(abs(x["wl"] - e) < 0.002 for e in extra)]

    def dE(i, j):
        return (elv[j - 1] - elv[i - 1]) / CM_PER_EV

    rows = []
    for x in sorted(targets, key=lambda x: x["wl"]):
        u, l = no(x["Ek"]), no(x["Ei"])
        r = dict(ion="XeII", wl_nm=round(x["wl"], 3), nist_intens=x["I"], A_nist=x["A"] or "", upper_no=u or "", lower_no=l or "",
                 upper=ref.label(u) if u else f"Ek={x['Ek']}", lower=ref.label(l) if l else f"Ei={x['Ei']}")
        for sh, (w, I) in sheets.items():
            f = fit_line(w, I, x["wl"])
            if f:
                r[f"{sh}_amp"] = round(f["amp"])
                r[f"{sh}_snr"] = round(f["snr"], 1)
                r[f"{sh}_S/N"] = round(f["sn"])
                r[f"{sh}_fwhm_pm"] = round(f["fwhm"] * 1e3, 1) if np.isfinite(f["fwhm"]) else ""
                r[f"{sh}_sat"] = "SAT" if f["saturated"] else ""
        blends = []
        for sp, L in nist.items():
            for y in L:
                if abs(y["wl"] - x["wl"]) <= 1.5 * a.fwhm and not (sp == "XeII" and abs(y["wl"] - x["wl"]) < 1e-4
                                                                   and y["Ek"] == x["Ek"]):
                    blends.append(f"{sp} {y['wl']:.3f} (I={y['I'] or '-'}{', A=%.1e' % y['A'] if y['A'] else ''})")
        r["nist_within_1.5fwhm"] = "; ".join(blends)
        if u and (1, u) in ref.sigma:
            E, s = ref.sigma[(1, u)]
            thr = dE(1, u)
            k1, k2 = rate_from_sigma(E, s, 1.0, thr), rate_from_sigma(E, s, 2.0, thr)
            kT = rate_from_sigma(E, s, a.Te, thr)
            inits = sorted(i for (i, j) in ref.sigma if j == u)
            lever = {i: rate_from_sigma(*ref.sigma[(i, u)], a.Te, dE(i, u)) / kT for i in inits if i > 2}
            g_u = lv[u - 1].two_j + 1
            kq = 0.0
            for i in inits:                       # de-excitation u -> i by detailed balance
                if i == u:
                    continue
                ki = rate_from_sigma(*ref.sigma[(i, u)], a.Te, dE(i, u))
                kq += ki * (lv[i - 1].two_j + 1) / g_u * np.exp(dE(i, u) / a.Te)
            # E1-allowed lower levels (even parity, below u, |dJ| <= 1) and NIST A for them
            allowed = [i for i, y in enumerate(lv, 1) if y.parity != lv[u - 1].parity and y.energy_cm < lv[u - 1].energy_cm
                       and abs(y.two_j - lv[u - 1].two_j) <= 2 and not (y.two_j == 0 and lv[u - 1].two_j == 0)]
            withA = {y["l"]: y["A"] for y in xe2 if y["u"] == u and y["A"]}
            sumA = sum(withA.values())
            r.update(threshold_eV=round(thr, 3), sigma_max_1e16=round(float(s.max()) / 1e-16, 4),
                     E_sigma_max_eV=round(float(E[np.argmax(s)]), 2), sigma_first_1e16=round(float(s[0]) / 1e-16, 4),
                     E_first_eV=round(float(E[0]), 3), k_gs_1eV=f"{k1:.3e}", k_gs_2eV=f"{k2:.3e}",
                     Te_slope_1_2eV=round(np.log(k2 / k1) / np.log(2), 2),
                     initial_states=" ".join(map(str, inits)),
                     meta_lever=f"{sum(lever.values()):.2e}",
                     top_meta=" ".join(f"{i}:{v:.1e}" for i, v in sorted(lever.items(), key=lambda kv: -kv[1])[:3]),
                     k_quench=f"{kq:.2e}", lower_levels_E1=len(allowed),
                     A_known_nist=f"{len(withA)}/{len(allowed)}", sumA_nist=f"{sumA:.2e}" if sumA else "",
                     n_crit_lower_bound=f"{sumA / kq:.1e}" if sumA else "")
        rows.append(r)
    for item in [v for v in a.ba.split(",") if v.strip()]:
        sp, wl = item.split(":")
        x = min(nist[sp], key=lambda y: abs(y["wl"] - float(wl)))
        r = dict(ion=sp, wl_nm=round(x["wl"], 3), nist_intens=x["I"], A_nist=x["A"] or "",
                 upper=f"Ek={x['Ek']}", lower=f"Ei={x['Ei']}")
        for sh, (w, I) in sheets.items():
            f = fit_line(w, I, x["wl"])
            if f:
                r[f"{sh}_amp"] = round(f["amp"])
                r[f"{sh}_snr"] = round(f["snr"], 1)
                r[f"{sh}_S/N"] = round(f["sn"])
                r[f"{sh}_fwhm_pm"] = round(f["fwhm"] * 1e3, 1) if np.isfinite(f["fwhm"]) else ""
                r[f"{sh}_sat"] = "SAT" if f["saturated"] else ""
        r["nist_within_1.5fwhm"] = "; ".join(
            f"{s2} {y['wl']:.3f} (I={y['I'] or '-'}{', A=%.1e' % y['A'] if y['A'] else ''})"
            for s2, L in nist.items() for y in L
            if abs(y["wl"] - x["wl"]) <= 1.5 * a.fwhm and not (s2 == sp and abs(y["wl"] - x["wl"]) < 1e-4))
        rows.append(r)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} lines -> {a.out}")
    if a.doublet:
        wl1, wl2 = (float(v) for v in a.doublet.split(","))
        out2 = Path(a.out).with_name(Path(a.out).stem + "_doublet.csv")
        with open(out2, "w", newline="") as f:
            wr = None
            for sh, (w, I) in sheets.items():
                if not (w.min() < wl1 < w.max()):
                    continue
                d = {"sheet": sh, "wl1": wl1, "wl2": wl2, **{k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in fit_doublet(w, I, wl1, wl2).items()}}
                if wr is None:
                    wr = csv.DictWriter(f, fieldnames=list(d))
                    wr.writeheader()
                wr.writerow(d)
        print(f"doublet {wl1}/{wl2} -> {out2}")


if __name__ == "__main__":
    main()
