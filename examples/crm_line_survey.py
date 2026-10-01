"""Xe II lines for a collisional-radiative model: what Wang et al (2019) covers.

For every NIST Xe II line in [--wmin, --wmax] whose upper level is a 6p level
with cross sections in the Wang et al (2019) supplementary spreadsheet:

* Maxwellian rate coefficient from the ground state k_gs(Te),
* its logarithmic Te slope d ln k_gs / d ln Te (Te-sensitivity of the line),
* the "metastable lever" sum_m k_m->up / k_gs: if the metastable fraction of
  Xe+ is f, stepwise excitation adds ~ f * lever to the upper-level
  population, so a large lever means a density-sensitive line,
* how many decay channels of the upper level have A-values in NIST,
* optionally, the peak in a measured spectrum (xlsx sheets: wavelength,
  intensity).

    python examples/crm_line_survey.py CrossSectionsIon.xlsx nist_XeII.csv \
        [--spectrum A25_second.xlsx --sheet data2] [--out xe2_lines.csv]

nist_XeII.csv: NIST ASD lines output, "CSV" format, with configurations,
terms, J, energies, A-values and intensities (lines1.pl).
"""
from __future__ import annotations

import argparse
import csv
import re

import numpy as np

from pydbsr.reference import read_xlsx

CM_PER_EV = 8065.544
ME, QE = 9.109e-31, 1.602e-19


def maxwell_rate(E, sig, Te, dE):
    """<sigma v> [cm^3/s] for a Maxwellian at Te [eV]; E [eV], sig [cm^2].

    Ion targets: sigma is finite at threshold, so it is held constant between
    the threshold dE and the first tabulated energy; above the last point it
    falls as 1/E.
    """
    Eg = np.linspace(dE, dE + 60 * Te, 6000)
    s = np.interp(Eg, E, sig, left=sig[0])
    hi = Eg > E[-1]
    s[hi] = sig[-1] * E[-1] / Eg[hi]
    v = np.sqrt(2 * Eg * QE / ME) * 100
    f = 2 * np.sqrt(Eg / np.pi) * Te ** -1.5 * np.exp(-Eg / Te)
    return np.trapezoid(s * v * f, Eg)


def _clean(s):
    return (s or "").replace('"', "").lstrip("=").strip()


def _num(s):
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", s))
    except ValueError:
        return None


def read_nist_lines(path):
    out = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            d = {k: _clean(v) for k, v in d.items() if k}
            wl = _num(d.get("obs_wl_air(nm)") or d.get("ritz_wl_air(nm)") or "")
            ei, ek = _num(d.get("Ei(cm-1)", "")), _num(d.get("Ek(cm-1)", ""))
            if wl is None or ei is None or ek is None:
                continue
            out.append(dict(wl=wl, intens=d.get("intens", ""), A=d.get("Aki(s^-1)", ""), Ei=ei, Ek=ek,
                            lower=f"{d.get('conf_i', '')} {d.get('term_i', '')} {d.get('J_i', '')}"))
    return out


def read_spectrum(path, sheet):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    a = np.array([r[:2] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None], float)
    return a[:, 0], a[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wang_xlsx")
    ap.add_argument("nist_csv")
    ap.add_argument("--wmin", type=float, default=300)
    ap.add_argument("--wmax", type=float, default=700)
    ap.add_argument("--Te", type=float, default=1.1, help="electron temperature for the table, eV")
    ap.add_argument("--spectrum")
    ap.add_argument("--sheet", default="data2")
    ap.add_argument("--shift", type=float, default=-0.11,
                    help="search window for the measured peak: [wl+shift, wl+0.01] nm")
    ap.add_argument("--saturation", type=float, default=64000)
    ap.add_argument("--out", default="xe2_lines_wang.csv")
    a = ap.parse_args(argv)

    ref = read_xlsx(a.wang_xlsx)
    elv = np.array([lv.energy_cm for lv in ref.levels])

    def level_no(e):
        k = int(np.argmin(abs(elv - e)))
        return k + 1 if abs(elv[k] - e) < 0.3 else None

    T2 = 2 * a.Te
    k = {}
    for (i, j), (E, s) in ref.sigma.items():
        dE = (elv[j - 1] - elv[i - 1]) / CM_PER_EV
        k[i, j] = (maxwell_rate(E, s, a.Te, dE), maxwell_rate(E, s, T2, dE))
    stepwise_up = sorted({j for (i, j) in k if i > 2})

    lines = read_nist_lines(a.nist_csv)
    for x in lines:
        x["lo"], x["up"] = level_no(x["Ei"]), level_no(x["Ek"])
    a_known = {j: (sum(1 for x in lines if x["up"] == j and x["A"]), sum(1 for x in lines if x["up"] == j))
               for j in stepwise_up}

    spec = read_spectrum(a.spectrum, a.sheet) if a.spectrum else None
    rows = []
    for x in sorted(lines, key=lambda x: x["wl"]):
        j = x["up"]
        if j not in stepwise_up or not a.wmin <= x["wl"] <= a.wmax:
            continue
        kg, kg2 = k[1, j]
        lever = sum(k[m, j][0] for m in range(3, len(elv) + 1) if (m, j) in k) / kg
        row = dict(wl_nm=f"{x['wl']:.3f}", nist_intens=x["intens"], A_nist=x["A"], upper_no=j,
                   lower_no=x["lo"] or "", upper=ref.label(j), lower=x["lower"].replace("5s2.5p4.", ""),
                   k_gs=f"{kg:.2e}", Te_slope=f"{np.log(kg2 / kg) / np.log(2):.1f}",
                   meta_lever=f"{lever:.1e}", A_known="%d/%d" % a_known[j])
        if spec is not None:
            w, I = spec
            m = (w > x["wl"] + a.shift) & (w < x["wl"] + 0.01)
            if m.any():
                bg = np.percentile(I[(w > x["wl"] - 1.5) & (w < x["wl"] + 1.5)], 25)
                pk = I[m].max()
                row.update(peak=f"{pk:.0f}", background=f"{bg:.0f}", saturated="yes" if pk >= a.saturation else "")
        rows.append(row)

    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in rows for c in r)))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} lines -> {a.out}  (Te = {a.Te} eV; Te_slope = d ln k_gs / d ln Te between Te and 2Te)")


if __name__ == "__main__":
    main()
