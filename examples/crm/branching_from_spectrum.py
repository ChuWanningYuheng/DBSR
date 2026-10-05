"""Measured branching ratios of Xe II upper levels (spectrum + sensitivity curve, no CRM).

All lines from one upper level u share n_u, so the photon fluxes of its branches are
proportional to the A-values:  BR(line) = N(line) / sum_all N,  N = area / S(lambda).
Branches outside the calibrated range (UV, IR) or too weak are not seen; their share is
taken from DBSR (transitions_E1.csv), so two numbers are given per line:

  BR_vis  = N(line) / sum over the seen branches            (upper bound)
  BR_corr = BR_vis * (1 - sum of DBSR BR of the unseen ones) (best estimate)

Details that matter (found on A3):
* the spectrum is stitched from ~25 nm segments (2038 points each), each with its own
  wavelength error (-2 ... -100 pm, varying along the segment): a line is fitted at
  NIST lambda + the error of its segment at that lambda (straight line through clean
  lines with S/N > 50 of criteria_A3.csv, outliers > 15 pm dropped); the fit starts at the
  top of the nearest peak (climbing from there, at most 40 pm) and the centre may move by +-8 pm;
* 545.045 / 545.090 are 48 pm apart: two-Gaussian fit (line_criteria.fit_doublet);
* lines with several classifications in NIST ('*' in intensity) are left out (counted
  as unseen);
* below 458 nm the sensitivity is extrapolated (marked).

    python examples/crm/branching_from_spectrum.py --spectrum A3.xlsx --sheet data10 \
        --wang CrossSectionsIon_3.xlsx --transitions RUNS/xe2_wang_full/transitions_E1.csv \
        --out docs/crm/branching_A3_data10.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbtools as nb  # noqa: E402
from line_criteria import fit_doublet, fit_line, read_sheets  # noqa: E402

SAT = 64000.0
LINES = [(44, 545.045), (41, 561.667), (42, 557.219), (51, 545.090), (29, 553.107), (39, 543.896)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--sheet", default="data10")
    ap.add_argument("--wang", required=True, help="CrossSectionsIon.xlsx (level numbers)")
    ap.add_argument("--transitions", required=True, help="transitions_E1.csv of a run with all pairs")
    ap.add_argument("--nist-dir", default="docs/crm")
    ap.add_argument("--curve", default="docs/crm/sensitivity_A3_curve.csv")
    ap.add_argument("--min-sn", type=float, default=5.0)
    ap.add_argument("--lines", default=None,
                    help="upper:line pairs (Wang numbers, nm), e.g. 101:483.025 (default: the selected lines)")
    ap.add_argument("--out", default="branching.csv")
    a = ap.parse_args(argv)
    from pydbsr import reference
    ref = reference.read_xlsx(a.wang)
    w, I = read_sheets(a.spectrum)[a.sheet]
    nd = Path(a.nist_dir)

    # segments and their wavelength offsets
    from scipy.ndimage import median_filter
    dw = np.diff(w)
    loc = median_filter(dw, 101, mode="nearest")              # the step changes along the spectrum (13 -> 5 pm)
    starts = np.r_[w[0], w[1:][np.abs(dw - loc) > 0.45 * loc]]
    starts = starts[np.r_[True, np.diff(starts) > 5.0]]           # joins only, not rounding steps
    seg = lambda lam: int(np.searchsorted(starts, lam, "right")) - 1
    sh = {}
    for r in csv.DictReader(open(nd / "criteria_A3.csv")):
        wl = float(r["wl_nm"])
        if r["nist_within_1.5fwhm"] or not w[0] + 1 < wl < w[-1] - 1:
            continue
        f = fit_line(w, I, wl)                   # reference lines: the list of criteria_A3.csv, fitted here
        if f and f["sn"] > 50 and not f["saturated"] and 0.018 < f["fwhm"] < 0.032:
                sh.setdefault(seg(wl), []).append((wl, f["centre"] - wl))
    shift = {}                                   # per segment: offset(lambda) = c0 + c1 (lambda - start), robust
    for k, v in sh.items():
        x, y = np.array(v).T
        keep = np.abs(y - np.median(y)) < 0.03
        for _ in range(3):
            if keep.sum() >= 3:
                c = np.polyfit(x[keep], y[keep], 1)
            else:
                c = np.array([0.0, np.median(y[keep]) if keep.any() else np.median(y)])
            keep = np.abs(y - np.polyval(c, x)) < 0.015
        shift[k] = (c, int(keep.sum()))

    cur = np.array([[float(r["wl_nm"]), float(r["S_rel_550"])] for r in csv.DictReader(open(a.curve))])
    lin = np.polyfit(cur[:8, 0], np.log(cur[:8, 1]), 1)

    def S(lam):
        if lam < cur[0, 0]:
            return float(np.exp(np.polyval(lin, lam))), "S экстраполирована"
        if lam > cur[-1, 0]:
            return float("nan"), ""
        return float(np.interp(lam, cur[:, 0], cur[:, 1])), ""

    def area_at(lam):
        k = seg(lam)
        if k not in shift:
            return None
        c, n_ref = shift[k]
        if n_ref == 0:
            return None
        c0 = lam + float(np.polyval(c, lam))
        k = int(np.argmin(np.abs(w - c0)))      # the prediction is good to a few tens of pm: climb from it
        k0 = k                                  # to the top of the nearest peak (at most 40 pm away)
        while True:
            j = k + 1 if I[k + 1] >= I[k - 1] else k - 1
            if I[j] <= I[k] or abs(w[j] - w[k0]) > 0.04:
                break
            k = j
        c0 = float(w[k])
        m = (w > c0 - 0.06) & (w < c0 + 0.06)
        bg = np.percentile(I[(w > c0 - 1.5) & (w < c0 + 1.5)], 25)
        g = lambda x, amp, c, s: amp * np.exp(-0.5 * ((x - c) / s) ** 2) + bg
        try:
            p, _ = curve_fit(g, w[m], I[m], p0=[max(I[m].max() - bg, 1), c0, 0.0104],
                             bounds=([0, c0 - 0.008, 0.006], [1e7, c0 + 0.008, 0.016]))
        except Exception:
            return None
        if I[m].max() >= SAT:
            return "sat"
        d = np.diff(I[(w > c0 - 1) & (w < c0 + 1)])
        noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        return p[0] * 2.3548 * p[2], p[0] / noise

    elv = [lv.energy_cm for lv in ref.levels]
    nist = {}
    for f in ("nist_XeII.csv", "nist_XeII_UV.csv", "nist_XeII_IR.csv"):
        if (nd / f).exists():
            for x in nb.nist_lines(nd / f):
                u, low = nb.level_number(elv, x["Ek"]), nb.level_number(elv, x["Ei"])
                if u and low:
                    nist[(u, low)] = x
    trs = nb.read_csv(a.transitions)
    dbl = fit_doublet(w, I, 545.045, 545.090) if w[0] < 545 < w[-1] else None
    out = []
    lines = [(int(u), float(x)) for u, x in (t.split(":") for t in a.lines.split(","))] if a.lines else LINES
    for U, LINE in lines:
        rows = [r for r in trs if r["upper_no"] == str(U)]
        tot = sum(float(r["A_exp"]) for r in rows)
        seen, unseen, sat = [], 0.0, False
        for r in rows:
            low = int(r["lower_no"])
            br = float(r["A_exp"]) / tot
            x = nist.get((U, low))
            lam = x["wl"] if x else nb.fnum(r["wl_air_nm"])
            s, note = S(lam) if 400 < lam else (float("nan"), "")
            res = None
            if x and "*" in x["intens"]:
                why = "несколько классификаций в NIST (*)"
            elif not np.isfinite(s) or not w[0] < lam < w[-1]:
                why = "вне калиброванного диапазона"
            elif abs(lam - 545.045) < 0.002 or abs(lam - 545.090) < 0.002:
                if I[(w > 544.8) & (w < 545.2)].max() >= SAT:
                    sat = True
                    out.append(dict(upper=U, line=LINE, branch_nm=round(lam, 3), lower=low, BR_dbsr=br,
                                    note="насыщена"))
                    continue
                amp = dbl["amp1"] if lam < 545.07 else dbl["amp2"]
                res, why = (amp * dbl["fwhm_pm"] * 1e-3, 100.0), "фит дублета"
            else:
                res = area_at(lam)
                if res == "sat":
                    sat = True
                    res, why = None, "насыщена"
                else:
                    why = "" if res and res[1] >= a.min_sn else f"S/N < {a.min_sn:g}"
            if res is None or res[1] < a.min_sn:
                unseen += br
                out.append(dict(upper=U, line=LINE, branch_nm=round(lam, 3), lower=low, BR_dbsr=br,
                                photons=None, S=None, sn=None, note=why))
                continue
            seen.append(dict(upper=U, line=LINE, branch_nm=round(lam, 3), lower=low, BR_dbsr=br,
                             photons=res[0] / s, S=s, sn=res[1], note=" ".join(t for t in (note, why) if t)))
        T = sum(v["photons"] for v in seen) or float("nan")
        for v in seen:
            v["share_of_seen"] = v["photons"] / T
        out += seen
        me = [v for v in seen if abs(v["branch_nm"] - LINE) < 0.003]
        if sat:
            print(f"{LINE:8.3f} (уровень {U}): ветвь насыщена — ветвление по этому листу не определить")
            out.append(dict(upper=U, line=LINE, branch_nm="ИТОГ", note="насыщенная ветвь"))
            continue
        if me:
            b = me[0]["share_of_seen"]
            out.append(dict(upper=U, line=LINE, branch_nm="ИТОГ", BR_dbsr=me[0]["BR_dbsr"], share_of_seen=b,
                            BR_vis=b, BR_corr=b * (1 - unseen), unseen_dbsr=unseen))
            print(f"{LINE:8.3f} (уровень {U}): BR_vis = {b:.4f}, BR_corr = {b * (1 - unseen):.4f} "
                  f"(невидимые по DBSR {unseen:.2f}), BR_DBSR = {me[0]['BR_dbsr']:.4f}")
    keys = ["upper", "line", "branch_nm", "lower", "BR_dbsr", "photons", "S", "sn", "share_of_seen",
            "BR_vis", "BR_corr", "unseen_dbsr", "note"]
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)


if __name__ == "__main__":
    main()
