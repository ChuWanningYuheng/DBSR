"""Line list for the CRM: Xe II, Ba I, Ba II lines seen in a measured spectrum.

For every NIST line of Xe II / Ba I / Ba II in the spectral range of the
measurement the script finds the measured peak, checks for blends with other
lines (Xe I, Xe II, Xe III, Ba I, Ba II), and tells which atomic data exist:

  class A  upper level = 5p4 6p with cross sections from the ground state AND
           from the 6s/5d metastables in Wang et al (2019)
  class B  upper level with cross sections from the ground state only (Wang)
  class C  upper level not in Wang (7s, 6d, 8s, 4f, 7p, (1D2)6d ...):
           needs the extended DBSR model (notebook 03)
  class D  high-l Rydberg upper levels (5g, 6g, 6f, 7g): not calculated
  Ba       Ba I / Ba II lines (notebooks 04, 05)

For class A/B upper levels: Maxwellian rate from the ground state k_gs(Te),
its slope d ln k / d ln Te and the metastable lever sum_m k_m / k_gs.

    python examples/crm/line_list.py --wang CrossSectionsIon.xlsx \
        --spectrum A25_second.xlsx --sheet data2 --nist-dir docs/crm --out docs/crm/lines

writes lines.csv (all lines) and lines.md (formatted list).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from pydbsr.crm import rate_from_sigma
from pydbsr.reference import read_xlsx

CM_PER_EV = 8065.544
SPECTRA = {"XeI": "Xe I", "XeII": "Xe II", "XeIII": "Xe III", "BaI": "Ba I", "BaII": "Ba II"}

# preliminary roles (docs/crm_plan.md, section 4); final choice from the CRM sensitivity matrix
ROLES = {
    "Xe II": {
        561.667: "ne (41 vs 44)", 545.045: "ne (44 vs 41), бленда с 545.090", 680.574: "ne (41), A в NIST",
        575.103: "ne (44)", 610.143: "ne (44)",
        547.261: "ne (31 vs 29), общий нижний уровень", 553.107: "ne (29 vs 31), общий нижний уровень",
        487.650: "ne (52 vs 56)", 575.865: "ne (52)", 659.501: "ne (52); запрошена",
        466.849: "ne (56 vs 52)", 504.492: "ne (56)",
        594.553: "Te: (3P2)6p", 634.396: "Te: (3P2)6p",
        627.082: "Te: (1D2)6p, A в NIST", 476.905: "Te: (1D2)6p", 572.691: "Te: (1D2)6p",
        471.263: "Te: (1S0)6p", 501.283: "Te: (1S0)6p",
        484.433: "диплом; калибровка (уровень 31)", 529.221: "диплом; Te (3P2)6p", 537.239: "диплом (термометр)",
        541.914: "диплом; Te (3P2)6p", 543.896: "диплом (термометр)",
        546.039: "калибровка (уровень 31)", 699.088: "калибровка (уровень 31)",
        603.620: "калибровка (уровень 26)", 605.115: "калибровка (уровень 26)",
        452.421: "запрошена", 669.432: "запрошена; рядом Ba I 669.38",
        342.073: "Ванг, ваш список", 350.036: "Ванг, ваш список", 488.730: "ваш список", 492.148: "ваш список",
        519.137: "ваш список", 545.090: "новая; бленда с 545.045",
    },
    "Ba I": {553.548: "n(Ba); пересчёт = валидация (Fursa, Chen–Gallagher)"},
    "Ba II": {455.403: "n(Ba+); ne по 455.4/493.4", 493.408: "n(Ba+); ne по 455.4/493.4",
              614.171: "калибровка: тот же верхний уровень, что 455.4", 585.367: "калибровка (6p3/2)",
              649.690: "калибровка: тот же верхний уровень, что 493.4"},
}


def _clean(s):
    return (s or "").replace('"', "").lstrip("=").strip()


def _num(s):
    try:
        return float(re.sub(r"[^0-9.]", "", s or ""))
    except ValueError:
        return None


def read_nist(path, spectrum):
    out = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            d = {k: _clean(v) for k, v in d.items() if k}
            wl = _num(d.get("obs_wl_air(nm)")) or _num(d.get("ritz_wl_air(nm)"))
            if wl is None:
                continue
            out.append(dict(sp=spectrum, wl=wl, I=d.get("intens", ""), A=d.get("Aki(s^-1)", ""),
                            Ei=_num(d.get("Ei(cm-1)")), Ek=_num(d.get("Ek(cm-1)")),
                            lower=_label(d.get("conf_i", ""), d.get("term_i", ""), d.get("J_i", "")),
                            upper=_label(d.get("conf_k", ""), d.get("term_k", ""), d.get("J_k", ""))))
    return out


def _label(conf, term, j):
    """'5s2.5p4.(3P<1>).6p', '2[1]*', '3/2' -> '(3P1)6p 2[1]° 3/2'."""
    conf = re.sub(r"<(\d+)>", r"\1", conf.replace("5s2.5p4.", "")).replace(").", ")").replace(".", " ")
    return f"{conf} {term.replace('*', '°')} {j}".strip()


def _iv(s):
    m = re.match(r"\d+", s or "")
    return int(m.group()) if m else 0


def read_spectrum(path, sheet):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    a = np.array([r[:2] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None], float)
    return a[:, 0], a[:, 1]


def build(wang, nist_dir, spectrum=None, sheet="data2", Te=1.1, shift=(-0.11, 0.015), sat=64000.0,
          snr_min=1.0, wmin=300.0, wmax=710.0):
    ref = read_xlsx(wang)
    elv = np.array([lv.energy_cm for lv in ref.levels])

    def no(e):
        if e is None:
            return None
        k = int(np.argmin(abs(elv - e)))
        return k + 1 if abs(elv[k] - e) < 0.3 else None

    gs_up = {j for (i, j) in ref.sigma if i == 1}
    meta_up = {j for (i, j) in ref.sigma if i > 2}
    kgs, slope, lever = {}, {}, {}
    for j in gs_up:
        E, s = ref.sigma[(1, j)]
        dE = elv[j - 1] / CM_PER_EV
        k1 = rate_from_sigma(E, s, Te, dE)
        k2 = rate_from_sigma(E, s, 2 * Te, dE)
        kgs[j], slope[j] = k1, np.log(k2 / k1) / np.log(2)
        lev = 0.0
        for (i, jj), (Em, sm) in ref.sigma.items():
            if jj == j and i > 2:
                lev += rate_from_sigma(Em, sm, Te, (elv[j - 1] - elv[i - 1]) / CM_PER_EV)
        lever[j] = lev / k1

    allines = []
    for key, sp in SPECTRA.items():
        p = Path(nist_dir) / f"nist_{key}.csv"
        if p.exists():
            allines += read_nist(p, sp)
    w = I = None
    if spectrum:
        w, I = read_spectrum(spectrum, sheet)
    rows = []
    for x in allines:
        if x["sp"] not in ("Xe II", "Ba I", "Ba II") or not wmin <= x["wl"] <= wmax:
            continue
        role = next((v for k, v in ROLES.get(x["sp"], {}).items() if abs(k - x["wl"]) < 0.0025), "")
        row = dict(ion=x["sp"], wl_nm=round(x["wl"], 3), nist_intens=x["I"], A_nist=x["A"],
                   upper=x["upper"], lower=x["lower"], Ek_cm=x["Ek"], Ei_cm=x["Ei"], role=role)
        if x["sp"] == "Xe II":
            up, lo = no(x["Ek"]), no(x["Ei"])
            row.update(upper_no=up or "", lower_no=lo or "")
            u = x["upper"]
            if re.search(r"\)\d+[fgh] ", u) and not re.search(r"\)4f ", u):
                cls = "D"
            elif up in meta_up:
                cls = "A"
            elif up in gs_up:
                cls = "B"
            else:
                cls = "C"
            row["class"] = cls
            if up in kgs:
                row.update(k_gs=f"{kgs[up]:.2e}", Te_slope=f"{slope[up]:.1f}",
                           meta_lever=f"{lever[up]:.1e}" if lever[up] > 0 else "")
        else:
            row["class"] = "Ba"
        if w is not None:
            m = (w > x["wl"] + shift[0]) & (w < x["wl"] + shift[1])
            if not m.any():
                row["observed"] = "вне диапазона"
            else:
                k = np.argmax(np.where(m, I, -np.inf))
                bg = float(np.percentile(I[(w > w[k] - 1.5) & (w < w[k] + 1.5)], 25))
                snr = (I[k] - bg) / bg
                # other lines that can produce the same peak (inverse of the search window)
                comp = [y for y in allines if y is not x and w[k] - shift[1] < y["wl"] < w[k] - shift[0]
                        and not (y["sp"] == x["sp"] and abs(y["wl"] - x["wl"]) < 1e-4)]
                comp.sort(key=lambda y: -_iv(y["I"]))
                seen, uniq = set(), []
                for y in comp:
                    if (y["sp"], y["wl"]) not in seen:
                        seen.add((y["sp"], y["wl"]))
                        uniq.append(y)
                strong = [y for y in uniq if _iv(y["I"]) >= max(1, _iv(x["I"]) // 3) or (y["sp"] != x["sp"] and y["A"])]
                dominated = [y for y in uniq if y["sp"] in ("Xe I", "Xe II", "Xe III")
                             and (_iv(y["I"]) >= 3 * max(1, _iv(x["I"])) or x["sp"] != "Xe II")]
                obs = "да" if snr >= snr_min else ("слабо" if snr >= 0.3 else "нет")
                if dominated and obs != "нет":
                    obs = "бленда"
                row.update(peak=round(float(I[k])), bg=round(bg), snr=round(snr, 1),
                           saturated="да" if I[k] >= sat else "",
                           blend="; ".join(f"{y['sp']} {y['wl']:.3f} ({y['I'] or 'A=' + y['A']})" for y in strong[:3]),
                           observed=obs)
        rows.append(row)
    rows.sort(key=lambda r: (r["ion"], r["wl_nm"]))
    return rows


def write_csv(rows, path):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


CLASS_TEXT = {
    "A": "σ из основного и из метастабилей есть у Ванга; нужны A (ветвление) — ноутбук 01",
    "B": "σ только из основного (Ванг); из метастабилей — ноутбук 03",
    "C": "верхнего уровня нет у Ванга — расширенная модель, ноутбук 03",
    "D": "высокие l (5g, 6g, 6f, 7g) — не считаем",
}


def _md_row(r, full=True):
    obs = r.get("observed", "")
    if obs == "да":
        obs = f"S/B {r['snr']}" + (", **насыщ.**" if r.get("saturated") else "")
    elif obs in ("слабо", "нет", "бленда"):
        obs = f"{obs} (S/B {r['snr']})"
    blend = r.get("blend", "")
    no = f"{r.get('upper_no', '')}→{r.get('lower_no', '')}" if r["ion"] == "Xe II" else ""
    cells = [f"{r['wl_nm']:.3f}", r["ion"], f"{r['upper']} → {r['lower']}", no, r.get("A_nist") or "—",
             r.get("class", ""), obs or "—", blend or "", r.get("role", "")]
    if full:
        cells[4:4] = [r.get("Te_slope", ""), r.get("meta_lever", "")]
    return "| " + " | ".join(str(c) for c in cells) + " |"


def write_md(rows, path, source=""):
    head_full = ("| λ, нм | ион | переход (верхний → нижний) | № Ванга | d ln k/d ln Te | рычаг метаст. | A NIST | класс | "
                 "в спектре | возможная бленда | роль |\n|---|---|---|---|---|---|---|---|---|---|---|")
    head = ("| λ, нм | ион | переход (верхний → нижний) | № Ванга | A NIST | класс | в спектре | возможная бленда | роль |\n"
            "|---|---|---|---|---|---|---|---|---|")
    sel = [r for r in rows if r.get("role")]
    vis = [r for r in rows if r.get("observed") == "да" and not r.get("role")]
    out = ["# Список линий для CRM (Xe II, Ba I, Ba II)", "",
           "Сгенерировано `examples/crm/line_list.py`" + (f" по спектру {source}" if source else "") +
           ". Полная таблица — [lines.csv](lines.csv). Длины волн — NIST (воздух).", "",
           "Классы данных Xe II:", ""]
    out += [f"* **{k}** — {v}" for k, v in CLASS_TEXT.items()]
    out += ["", "«d ln k/d ln Te» — наклон константы возбуждения из основного состояния при Te = 1.1 эВ; "
            "«рычаг метаст.» — Σ k(метастабиль→верхний)/k(основное→верхний) (по сечениям Ванга). "
            "«S/B» — (пик − фон)/фон; **насыщ.** — пик упирается в предел АЦП. "
            "«Возможная бленда» — линии NIST в пределах ±0.06 нм (проверять по форме контура).", "",
            "## 1. Барий", "", head]
    out += [_md_row(r, False) for r in rows if r["ion"] != "Xe II" and r.get("role")]
    out += ["", "## 2. Xe II: выбранные линии (диагностика ne, Te, калибровка, запрошенные)", "", head_full]
    out += [_md_row(r) for r in sel if r["ion"] == "Xe II"]
    for cls in "ABCD":
        part = [r for r in vis if r["ion"] == "Xe II" and r.get("class") == cls]
        if not part:
            continue
        out += ["", f"## {3 + 'ABCD'.index(cls)}. Xe II, видимые в спектре, класс {cls}: {CLASS_TEXT[cls]}", "",
                f"{len(part)} линий.", "", head_full if cls in "AB" else head]
        out += [_md_row(r, cls in "AB") for r in part]
    Path(path).write_text("\n".join(out) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wang", required=True)
    ap.add_argument("--nist-dir", default="docs/crm")
    ap.add_argument("--spectrum")
    ap.add_argument("--sheet", default="data2")
    ap.add_argument("--Te", type=float, default=1.1)
    ap.add_argument("--out", default="lines", help="output prefix (.csv and .md are added)")
    a = ap.parse_args(argv)
    rows = build(a.wang, a.nist_dir, a.spectrum, a.sheet, a.Te)
    write_csv(rows, a.out + ".csv")
    write_md(rows, a.out + ".md", f"{Path(a.spectrum).name}, лист {a.sheet}" if a.spectrum else "")
    n = sum(r.get("observed") == "да" for r in rows)
    print(f"{len(rows)} lines, {n} seen in the spectrum -> {a.out}.csv, {a.out}.md")


if __name__ == "__main__":
    main()
