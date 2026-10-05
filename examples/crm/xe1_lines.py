"""Xe I lines in the measured spectra: which are seen, saturated or blended (an independent
ne/Te diagnostic from the neutral atom, whose atomic data are better known than those of Xe+).

For every Xe I line of NIST ASD (docs/crm/nist_XeI.csv) inside the spectrum it reports the
Gaussian fit in each sheet (amplitude, S/N, saturation; the wavelength error of the stitched
spectrum is measured locally on clean isolated Xe I/II lines), the Paschen labels of the levels
(5p5 6s = 1s5..1s2, 5p5 6p = 2p10..2p1 in order of decreasing energy) and lines of the other
species (Xe I/II/III, Ba I/II, W, Ca, Al) within +-1.5 FWHM.

    python examples/crm/xe1_lines.py --spectrum A2.xlsx --out docs/crm/xe1_lines_A2.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_criteria import SPECIES, fit_line, read_nist, read_sheets  # noqa: E402

FWHM = 0.0245            # nm, instrument width of the spectra (docs/crm/lines_final.md)


def _clean(s):
    return (s or "").replace('"', "").lstrip("=").strip()


def _num(s):
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", s or ""))
    except ValueError:
        return None


def xe1_lines(path):
    """NIST Xe I lines: wavelength, A, energies, level designations."""
    out = []
    for d in csv.DictReader(open(path, newline="")):
        d = {k: _clean(v) for k, v in d.items() if k}
        wl = _num(d.get("obs_wl_air(nm)")) or _num(d.get("ritz_wl_air(nm)"))
        if not wl:
            continue
        out.append(dict(wl=wl, A=_num(d.get("Aki(s^-1)")), acc=d.get("Acc", ""), Ei=_num(d.get("Ei(cm-1)")),
                        Ek=_num(d.get("Ek(cm-1)")), intens=d.get("intens", ""),
                        lower=f"{d.get('conf_i', '')} {d.get('term_i', '')} J={d.get('J_i', '')}",
                        upper=f"{d.get('conf_k', '')} {d.get('term_k', '')} J={d.get('J_k', '')}",
                        conf_i=d.get("conf_i", ""), conf_k=d.get("conf_k", "")))
    return out


def paschen(lines):
    """{energy: label} for the 5p5 6s (1s5..1s2) and 5p5 6p (2p10..2p1) levels."""
    lab = {}
    for shell, prefix, n in ((".6s", "1s", 5), (".6p", "2p", 10)):
        es = sorted({round(e, 2) for x in lines for e, c in ((x["Ei"], x["conf_i"]), (x["Ek"], x["conf_k"]))
                     if e is not None and c.endswith(shell)})
        for k, e in enumerate(es):
            lab[e] = f"{prefix}{n - k}"
    return lab


def calibration(w, I, refs, half=10.0):
    """Wavelength error of the stitched spectrum (nm) near a wavelength: the spectrum is made of
    ~25 nm pieces with their own scale errors (-20 ... -100 pm and more), and the joins cannot
    be found from the wavelength column of the full-range sheets (its step jitters by rounding).
    Measured on the reference lines ``refs`` (NIST wavelengths of lines with no other Xe/Ba line
    within 0.25 nm, so a wide search window cannot catch a neighbour): Gaussian fit in
    [wl - 0.22, wl + 0.03], kept if S/N > 200, not saturated, instrument width.  Returns
    wl -> median error (outliers dropped) of >= 3 references within +-``half`` nm, widened if needed."""
    pts = []
    for wl in refs:
        if not w[0] + 1 < wl < w[-1] - 1:
            continue
        f = fit_line(w, I, wl, shift=(-0.22, 0.03))
        if f and np.isfinite(f["sn"]) and f["sn"] > 200 and not f["saturated"] and 0.7 * FWHM < f["fwhm"] < 1.4 * FWHM:
            pts.append((wl, f["centre"] - wl))
    if not pts:
        return lambda lam: -0.06
    x, d = np.array(pts).T
    glob = float(np.median(d))

    def shift(lam):
        for h in (half, 2 * half, 4 * half):             # widen until >= 3 references
            near = np.abs(x - lam) < h
            if near.sum() >= 3:
                v = d[near]
                med = np.median(v)
                ok = np.abs(v - med) < max(0.02, 2.5 * 1.4826 * np.median(np.abs(v - med)))   # drop outliers
                return float(np.median(v[ok]))
        return glob
    return shift


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--nist-dir", default=str(Path(__file__).resolve().parents[2] / "docs" / "crm"))
    ap.add_argument("--min-sn", type=float, default=10.0, help="report lines with S/N above this in some sheet")
    ap.add_argument("--out", default="xe1_lines.csv")
    a = ap.parse_args(argv)

    nd = Path(a.nist_dir)
    xe1 = xe1_lines(nd / "nist_XeI.csv")
    lab = paschen(xe1)
    others = []
    for sp in SPECIES:
        f = nd / f"nist_{sp}.csv"
        if f.exists():
            others += read_nist(f, sp)
    sheets = read_sheets(a.spectrum)
    # reference lines for the wavelength scale: Xe I and Xe II lines with no other Xe or Ba line within
    # 0.3 nm (W, Ca, Al are not seen in these spectra); calibration() keeps the strong unsaturated ones
    seen = [o for o in others if o["sp"] in ("XeI", "XeII", "XeIII", "BaI", "BaII")]
    refs = [o["wl"] for o in seen if o["sp"] in ("XeI", "XeII")
            and not any(abs(p["wl"] - o["wl"]) < 0.25 and p is not o for p in seen)]
    calib = {name: calibration(w, I, refs) for name, (w, I) in sheets.items()}
    rows = []
    for x in sorted(xe1, key=lambda x: x["wl"]):
        res = {}
        for name, (w, I) in sheets.items():
            if w.min() + 0.2 < x["wl"] < w.max() - 0.2:
                d = calib[name](x["wl"])
                res[name] = fit_line(w, I, x["wl"], shift=(d - 0.04, d + 0.04))
        # only plausible fits: width near the instrument width, centre near the NIST wavelength
        res = {k: v for k, v in res.items() if v and v["amp"] > 0 and abs(v["centre"] - x["wl"]) < 0.25
               and (v["saturated"] or (np.isfinite(v["fwhm"]) and 0.4 * FWHM < v["fwhm"] < 3 * FWHM))}
        if not res:
            continue
        pool = [r for r in res.values() if not r["saturated"]] or list(res.values())   # S/N по ненасыщенным листам
        best = max(pool, key=lambda r: r["sn"] if np.isfinite(r["sn"]) else -1)
        if not (best["sn"] >= a.min_sn):
            continue
        near = [o for o in others if abs(o["wl"] - x["wl"]) < 1.5 * FWHM
                and not (o["sp"] == "XeI" and abs(o["wl"] - x["wl"]) < 1e-4)]
        lu, ll = (lab.get(round(x["Ek"], 2)), lab.get(round(x["Ei"], 2))) if x["Ek"] and x["Ei"] else (None, None)
        pl = f"{lu}-{ll}" if (lu and ll) else ""           # only when both levels have Paschen names
        rows.append(dict(wl_nm=x["wl"], paschen=pl, upper=x["upper"], lower=x["lower"],
                         E_upper_eV=round(x["Ek"] / 8065.544, 3) if x["Ek"] else None,
                         A_nist=x["A"], acc=x["acc"], best_sn=round(best["sn"]), best_amp=round(best["amp"]),
                         n_sheets=len(res), n_saturated=sum(r["saturated"] for r in res.values()),
                         blends="; ".join(f"{o['sp']} {o['wl']:.3f}" + (f" (I={o['I']})" if o["I"] else "")
                                          for o in near)))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} Xe I lines with S/N >= {a.min_sn:g} -> {a.out}")
    for r in rows:
        if r["paschen"].startswith("2p") and "1s" in r["paschen"]:
            print(f"{r['wl_nm']:9.3f}  {r['paschen']:9s}  A={r['A_nist'] or 0:9.2e}  S/N={r['best_sn']:6d}"
                  f"  насыщена в {r['n_saturated']}/{r['n_sheets']}  {r['blends']}")


if __name__ == "__main__":
    main()
