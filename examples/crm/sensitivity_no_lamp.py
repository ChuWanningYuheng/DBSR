"""Relative spectral sensitivity S(lambda) without a calibration lamp (branching-ratio method).

Lines from the same upper level: the photon ratio of two lines is A2/A1 (and
does not depend on the plasma), so the measured ratio N2/N1 gives

    S(lambda2) / S(lambda1) = (N2 / N1) / (A2 / A1).

The script takes every upper level that has at least two lines with an A-value
in NIST (Xe I, Xe II, Ba I, Ba II; optionally only some accuracy classes),
unsaturated and free of other NIST lines within 1.5 FWHM, measures the line
areas (Gaussian fits) in one sheet of the spectrum, and fits one smooth curve

    ln S(lambda) = sum_k c_k P_k(x),  x = (lambda - 550 nm) / 150 nm,  S(550) = 1

with a free normalisation for every upper level (least squares on ln N - ln A).

    python examples/crm/sensitivity_no_lamp.py --spectrum A3.xlsx --sheet data10 \
        --nist-dir docs/crm --out docs/crm/sensitivity_A3

Writes <out>_points.csv (every line used: lambda, upper level, measured area,
A, residual) and <out>_curve.csv (lambda, S, 1-sigma band).

Caveats: resonance lines (lower level = ground state, e.g. Ba II 455.4, 493.4)
can be self-absorbed; they are skipped unless --keep-resonance.  The accuracy
of the result is that of the NIST A-values used (class letters in the points
file).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_criteria import fit_line, read_sheets  # noqa: E402

SPECIES = ["XeI", "XeII", "XeIII", "BaI", "BaII", "WI", "WII", "CaI", "CaII", "AlI", "AlII"]
USE = ["XeI", "XeII", "BaI", "BaII"]


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
        wl = _num(d.get("obs_wl_air(nm)")) or _num(d.get("ritz_wl_air(nm)"))
        if wl:
            out.append(dict(sp=sp, wl=wl, I=d.get("intens", ""), A=_num(d.get("Aki(s^-1)")), acc=d.get("Acc", ""),
                            Ei=_num(d.get("Ei(cm-1)")), Ek=_num(d.get("Ek(cm-1)"))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--sheet", default="data10")
    ap.add_argument("--nist-dir", default="docs/crm")
    ap.add_argument("--wmin", type=float, default=400.0)
    ap.add_argument("--wmax", type=float, default=816.0)
    ap.add_argument("--fwhm", type=float, default=0.0245, help="instrumental FWHM, nm")
    ap.add_argument("--min-sn", type=float, default=30.0)
    ap.add_argument("--acc", default="", help="keep only these NIST accuracy classes, e.g. 'AA,A+,A,B+,B' (empty: all)")
    ap.add_argument("--exclude-acc", default="D+,D,E", help="drop NIST accuracy classes (D: > 50 %%)")
    ap.add_argument("--clip", type=float, default=3.0, help="drop points beyond clip*sigma (robust, MAD) once (0: off)")
    ap.add_argument("--keep-resonance", action="store_true")
    ap.add_argument("--order", type=int, default=3, help="polynomial order of ln S")
    ap.add_argument("--out", default="sensitivity")
    a = ap.parse_args(argv)

    nist = {sp: read_nist(Path(a.nist_dir) / f"nist_{sp}.csv", sp) for sp in SPECIES
            if (Path(a.nist_dir) / f"nist_{sp}.csv").exists()}
    w, I = read_sheets(a.spectrum)[a.sheet]
    acc = {x.strip() for x in a.acc.split(",") if x.strip()}
    pts, cand = [], []
    excl = {v.strip() for v in a.exclude_acc.split(",") if v.strip()}

    def reject(x, why):
        cand.append(dict(sp=x["sp"], wl=x["wl"], A=x["A"], acc=x["acc"], Ek=x["Ek"], status="отброшена", reason=why))

    for sp in USE:
        for x in nist.get(sp, []):
            # шаг 1: у линии есть A в NIST и она в диапазоне спектра
            if not (x["A"] and x["Ek"] is not None and a.wmin <= x["wl"] <= a.wmax):
                continue
            if acc and x["acc"] not in acc:
                reject(x, f"класс точности A {x['acc'] or '—'} не из списка {sorted(acc)}")
                continue
            if x["acc"] in excl:
                reject(x, f"класс точности A {x['acc']} (хуже 50 %)")
                continue
            if not a.keep_resonance and x["Ei"] == 0.0:
                reject(x, "резонансная (нижний уровень — основной): возможно самопоглощение")
                continue
            others = [y for s2, L in nist.items() for y in L if abs(y["wl"] - x["wl"]) <= 1.5 * a.fwhm
                      and not (s2 == sp and abs(y["wl"] - x["wl"]) < 1e-4)
                      and (_iv(y["I"]) >= 0.1 * max(1, _iv(x["I"])) or y["A"])]
            if others:
                o = others[0]
                reject(x, f"возможна бленда: {o['sp']} {o['wl']:.3f} в пределах 1.5 FWHM")
                continue
            f = fit_line(w, I, x["wl"])
            if not f:
                reject(x, "нет пика")
                continue
            if f["saturated"]:
                reject(x, "насыщена")
                continue
            if not np.isfinite(f["fwhm"]) or not 0.6 < f["fwhm"] / a.fwhm < 1.6:
                reject(x, f"ширина пика {f['fwhm'] * 1e3:.0f} пм не похожа на одиночную линию (~{a.fwhm * 1e3:.0f} пм)")
                continue
            if f["sn"] < a.min_sn or f["amp"] <= 0:
                reject(x, f"слабая: S/N = {f['sn']:.0f} < {a.min_sn:.0f}")
                continue
            pts.append(dict(sp=sp, wl=x["wl"], Ek=round(x["Ek"], 1), A=x["A"], acc=x["acc"],
                            amp=f["amp"], fwhm_pm=f["fwhm"] * 1e3, area=f["amp"] * f["fwhm"], sn=f["sn"]))
    groups = {}
    for p in pts:
        groups.setdefault((p["sp"], p["Ek"]), []).append(p)
    for k, v in groups.items():
        if len(v) < 2:
            p = v[0]
            cand.append(dict(sp=p["sp"], wl=p["wl"], A=p["A"], acc=p["acc"], Ek=p["Ek"], status="отброшена",
                             reason="других пригодных линий того же верхнего уровня нет — не с чем сравнить"))
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    used = [p for v in groups.values() for p in v]
    if not used:
        raise SystemExit("no upper level with two usable lines")
    x = lambda wl: (np.asarray(wl) - 550.0) / 150.0

    def solve(used):
        gkeys = sorted({(p["sp"], p["Ek"]) for p in used})
        M = np.array([[x(p["wl"]) ** k for k in range(1, a.order + 1)] +
                      [1.0 if j == gkeys.index((p["sp"], p["Ek"])) else 0.0 for j in range(len(gkeys))]
                      for p in used])
        y = np.array([np.log(p["area"] / p["A"]) for p in used])
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        res = y - M @ c
        s2 = float(res @ res) / max(len(y) - len(c), 1)
        return c, res, s2, s2 * np.linalg.pinv(M.T @ M)

    c, res, s2, cov = solve(used)
    if a.clip > 0:
        sig = 1.4826 * float(np.median(np.abs(res)))          # robust: outliers do not inflate it
        keep = [p for p, r in zip(used, res) if abs(r) <= a.clip * sig]
        dropped = [p for p, r in zip(used, res) if abs(r) > a.clip * sig]
        cnt = {}
        for p in keep:
            cnt[(p["sp"], p["Ek"])] = cnt.get((p["sp"], p["Ek"]), 0) + 1
        keep = [p for p in keep if cnt[(p["sp"], p["Ek"])] >= 2]
        if dropped:
            print("dropped (outliers):", ", ".join(f"{p['sp']} {p['wl']:.2f}" for p in dropped))
        for p in dropped:
            cand.append(dict(sp=p["sp"], wl=p["wl"], A=p["A"], acc=p["acc"], Ek=p["Ek"], status="отброшена",
                             reason=f"выброс: отклонение от кривой больше {a.clip:g} устойчивых σ"))
        for p in [p for p in keep if cnt.get((p["sp"], p["Ek"]), 0) < 2]:
            cand.append(dict(sp=p["sp"], wl=p["wl"], A=p["A"], acc=p["acc"], Ek=p["Ek"], status="отброшена",
                             reason="после удаления выброса у уровня осталась одна линия"))
        used = keep
        c, res, s2, cov = solve(used)
        groups = {k: v for k, v in groups.items() if any((p["sp"], p["Ek"]) == k for p in used)}
    for p, r in zip(used, res):
        p["resid_ln"] = round(float(r), 3)
    for p in used:
        cand.append(dict(sp=p["sp"], wl=p["wl"], A=p["A"], acc=p["acc"], Ek=p["Ek"], status="взята", reason=""))
    with open(a.out + "_candidates.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["sp", "wl", "A", "acc", "Ek", "status", "reason"])
        wr.writeheader()
        wr.writerows(sorted(cand, key=lambda c: c["wl"]))
    with open(a.out + "_points.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(used[0]))
        wr.writeheader()
        wr.writerows(sorted(used, key=lambda p: p["wl"]))
    grid = np.arange(max(a.wmin, min(p["wl"] for p in used)), min(a.wmax, max(p["wl"] for p in used)) + 1, 5.0)
    B = np.array([[x(g) ** k for k in range(1, a.order + 1)] for g in grid])
    lnS = B @ c[:a.order]
    err = np.sqrt(np.einsum("ij,jk,ik->i", B, cov[:a.order, :a.order], B))
    with open(a.out + "_curve.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["wl_nm", "S_rel_550", "S_low_1sigma", "S_high_1sigma"])
        for g, v, e in zip(grid, lnS, err):
            wr.writerow([round(float(g), 1), round(float(np.exp(v)), 4), round(float(np.exp(v - e)), 4),
                         round(float(np.exp(v + e)), 4)])
    print(f"{len(used)} lines from {len(groups)} upper levels ({', '.join(sorted({k[0] for k in groups}))}); "
          f"scatter of ln(S) = {np.sqrt(s2):.3f} (~{100 * (np.exp(np.sqrt(s2)) - 1):.0f} %)")
    for g, v, e in zip(grid[::10], lnS[::10], err[::10]):
        print(f"  {g:6.1f} nm  S/S(550) = {np.exp(v):.3f}  (1 sigma {np.exp(v - e):.3f} - {np.exp(v + e):.3f})")


if __name__ == "__main__":
    main()
