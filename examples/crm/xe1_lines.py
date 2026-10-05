"""Xe I lines in the measured spectra: which are seen, saturated or blended (an independent
ne/Te diagnostic from the neutral atom, whose atomic data are better known than those of Xe+).

For every Xe I line of NIST ASD (docs/crm/nist_XeI.csv) inside the spectrum it reports the
Gaussian fit in each sheet (amplitude, S/N, saturation; the wavelength error of the stitched
spectrum is measured locally on strong isolated Xe I/II lines), the Paschen labels of the levels
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


def calibration(w, I, refs, isolation=0.3, half=12.0):
    """Wavelength error of the stitched spectrum near each wavelength (nm): the spectrum is
    made of ~25 nm pieces, each with its own scale error (-30 ... -190 pm in these spectra).
    Measured on strong isolated lines ``refs`` (NIST wavelengths): the highest point within
    [wl - 0.25, wl + 0.05]; returns a function wl -> median error of the reference lines
    within +-``half`` nm (global median where there are none)."""
    refs = sorted(refs)
    pts = []
    for k, wl in enumerate(refs):
        if (k and wl - refs[k - 1] < isolation) or (k + 1 < len(refs) and refs[k + 1] - wl < isolation):
            continue
        m = (w > wl - 0.25) & (w < wl + 0.05)
        if m.sum() < 5:
            continue
        j = np.flatnonzero(m)[np.argmax(I[m])]
        bg = np.percentile(I[(w > wl - 1.0) & (w < wl + 1.0)], 25)
        if I[j] > 5 * bg:
            pts.append((wl, w[j] - wl))
    if not pts:
        return lambda wl: -0.06
    x, d = np.array(pts).T
    glob = float(np.median(d))

    def shift(wl):
        near = np.abs(x - wl) < half
        return float(np.median(d[near])) if near.sum() >= 2 else glob
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
    # reference lines for the wavelength scale: strong Xe I and Xe II lines with A in NIST
    xe_all = [o for o in others if o["sp"] in ("XeI", "XeII")]
    refs = [o["wl"] for o in xe_all if o["A"] and o["A"] > 5e6
            and not any(abs(p["wl"] - o["wl"]) < 0.3 and p is not o for p in others)]
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
        pl = f"{lab.get(round(x['Ek'], 2), '')}-{lab.get(round(x['Ei'], 2), '')}" if x["Ek"] and x["Ei"] else ""
        rows.append(dict(wl_nm=x["wl"], paschen=pl.strip("-"), upper=x["upper"], lower=x["lower"],
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
