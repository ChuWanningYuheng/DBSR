"""Generate the CRM notebooks examples/crm/0*_*.ipynb (run: python tools/make_crm_notebooks.py)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "examples" / "crm"

PARAMS = '''
# ---------- параметры (поменяйте пути под себя)
from pathlib import Path
import sys
REPO    = Path("/data/pydbsr_calc/DBSR_src")          # клон репозитория pydbsr
CRM     = REPO / "examples" / "crm"                   # эта папка: nbtools.py, run_model.py, models.py
DATA    = Path("/data/pydbsr_calc/data")              # ваши файлы
XLSX    = DATA / "CrossSectionsIon_3.xlsx"            # Wang et al (2019), сечения e + Xe+
SPECTRUM, SHEET = DATA / "A25_second.xlsx", "data2"   # измеренный спектр
NIST    = REPO / "docs" / "crm"                       # выгрузки NIST (nist_*.csv) и список линий
RUNS    = Path("/data/pydbsr_calc/runs/crm")          # результаты расчётов (большой диск)
SCRATCH, SCRATCH_GB = Path.home() / "pydbsr_tmp", 5   # временные файлы dbsr_mat (быстрый диск)
CORES, MEM, HD_THREADS = 32, 100, 8                   # ядра, ГБ памяти, потоки диагонализации
sys.path.insert(0, str(CRM))
import numpy as np, matplotlib.pyplot as plt
import pydbsr as db
from pydbsr import crm, reference
import nbtools as nb
%matplotlib inline
'''


def notebook(name, cells):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python (pydbsr)", "language": "python",
                                                   "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, text) in enumerate(cells):
        c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n"), "id": f"c{i:02d}"}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUT / name)


def md(t):
    return ("markdown", t)


def code(t):
    return ("code", t)


# ====================================================================== 00 line list
notebook("00_line_list.ipynb", [
    md("""
# 00. Список линий Xe II, Ba I, Ba II по измеренному спектру

Отождествление пиков спектра с линиями NIST, проверка бленд и классификация по
наличию атомных данных:

* **A** — верхний уровень 5p⁴6p: σ из основного и из метастабилей есть у Ванга, нужны A (ноутбук 01);
* **B** — σ только из основного (Ванг);
* **C** — верхнего уровня нет у Ванга (6d, 7s, 8s, 4f, 7p, (¹D₂)6d) — расширенная модель (ноутбук 03);
* **D** — высокие l (5g, 6g, 6f, 7g) — не считаем;
* **Ba** — линии бария (ноутбуки 04, 05).

Результат: `lines.csv` и `lines.md` (оформленный список) в папке `NIST`.
"""),
    code(PARAMS),
    code('''
import line_list as ll
rows = ll.build(XLSX, NIST, SPECTRUM, SHEET, Te=1.1)
ll.write_csv(rows, NIST / "lines.csv")
ll.write_md(rows, NIST / "lines.md", f"{SPECTRUM.name}, лист {SHEET}")
import collections
print(collections.Counter((r["class"], r.get("observed")) for r in rows))
'''),
    md("## Спектр и отмеченные линии (выберите участок)"),
    code('''
w, I = ll.read_spectrum(SPECTRUM, SHEET)
LO, HI = 540, 565                                       # нм
m = (w > LO) & (w < HI)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(w[m], I[m], lw=0.6, color="k")
colors = {"A": "tab:blue", "B": "tab:cyan", "C": "tab:orange", "D": "tab:gray", "Ba": "tab:red"}
for r in rows:
    if LO < r["wl_nm"] < HI and r.get("observed") in ("да", "бленда"):
        ax.axvline(r["wl_nm"] - 0.045, color=colors[r["class"]], lw=0.6, alpha=0.7)
        if r.get("role") or r["class"] == "Ba":
            ax.text(r["wl_nm"] - 0.045, I[m].max() * 0.9, f"{r['ion']} {r['wl_nm']:.2f}", rotation=90,
                    fontsize=7, color=colors[r["class"]])
ax.set_yscale("log"); ax.set_xlabel("λ, нм (шкала прибора)"); ax.set_ylabel("отсчёты")
plt.show()
'''),
    md("## Запрошенные линии"),
    code('''
for wl in (452.421, 659.501, 669.432, 545.045, 545.090, 561.667, 553.548, 455.403, 493.408):
    r = min(rows, key=lambda r: abs(r["wl_nm"] - wl))
    print(f"{r['ion']:6s} {r['wl_nm']:8.3f}  {r['upper']} -> {r['lower']}  класс {r['class']}  "
          f"№ {r.get('upper_no', '')}->{r.get('lower_no', '')}  S/B {r.get('snr')}  бленда: {r.get('blend', '')}")
'''),
])

# ====================================================================== 01 Xe II A-values
notebook("01_xe2_A_values.ipynb", [
    md("""
# 01. Xe II: вероятности переходов (A) и ветвления для всех линий 6p → 6s, 5d

Без A нельзя получить сечение линии: σ_линии = σ(возбуждения верхнего уровня) · A_ul/ΣA.
В NIST у большинства 6p-уровней известны 0–4 ветви из 7–21. Здесь A считаются в DBSR
(`dbsr_mult3` + `dbsr_dmat3`) на той же мишени, что у Ванга, с номерами уровней Ванга.

Расчёт только структурный (без рассеяния): мишень (dbsr_hf + mchf) и E1/M1/E2 для всех
пар состояний. Идёт в фоне, результат — `transitions_E1.csv` и т. д.

Проверки: (1) энергии против NIST; (2) A против NIST (T8043 и др.); (3) отношение
скорость/длина (калибровки) — близко к 1 для надёжных линий; (4) времена жизни 6p
против измеренных (впишите свои значения из литературы).
"""),
    code(PARAMS),
    code('''
job = nb.Job(RUNS / "xe2_wang", "xe2_wang", levels=XLSX, stage="transitions", cores=CORES,
             multipoles="E1,E2")
job.start()
'''),
    code("job.status()          # ход расчёта (запускайте когда хотите)"),
    md("## Энергии мишени против NIST"),
    code('''
W = RUNS / "xe2_wang"
print((W / "target_table.txt").read_text())
'''),
    md("## A: DBSR против NIST"),
    code('''
ref = reference.read_xlsx(XLSX)
elv = [lv.energy_cm for lv in ref.levels]
tr = nb.read_csv(W / "transitions_E1.csv")
calc = {(int(r["upper_no"]), int(r["lower_no"])): r for r in tr if r["upper_no"] and r["lower_no"]}
pairs = []
for x in nb.nist_lines(NIST / "nist_XeII.csv"):
    u, l = nb.level_number(elv, x["Ek"]), nb.level_number(elv, x["Ei"])
    if x["A"] and (u, l) in calc:
        r = calc[(u, l)]
        pairs.append((x["wl"], x["A"], nb.fnum(r["A_exp"]), nb.fnum(r["gauge_ratio"]) or np.nan, u, l))
pairs = np.array(pairs)
print(f"{len(pairs)} линий с A в NIST")
for wl, an, ac, gr, u, l in pairs[np.argsort(pairs[:, 0])]:
    print(f"{wl:9.3f}  {int(u):3d}->{int(l):3d}  NIST {an:9.2e}  DBSR {ac:9.2e}  DBSR/NIST {ac/an:5.2f}  vel/len {gr:5.2f}")
plt.loglog(pairs[:, 1], pairs[:, 2], "o"); x = np.array([1e5, 3e8]); plt.plot(x, x, "k-", x, 2 * x, "k:", x, x / 2, "k:")
plt.xlabel("A NIST, с⁻¹"); plt.ylabel("A DBSR, с⁻¹"); plt.show()
'''),
    md("""
## Ветвления и времена жизни 6p

`MEASURED_TAU_NS` — впишите измеренные времена жизни 6p-уровней Xe II из литературы
(номер уровня Ванга → нс), они попадут в сравнение.
"""),
    code('''
MEASURED_TAU_NS = {}           # например {26: 7.5, ...}: № уровня Ванга -> τ (нс)
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")
for u in sorted(j for j in tot if 25 <= j <= 56 or j in (101, 105)):
    tau = 1e9 / tot[u]
    top = ", ".join(f"{l}:{b:.2f}" for l, b in sorted(br[u].items(), key=lambda kv: -kv[1])[:5])
    meas = f"  изм. {MEASURED_TAU_NS[u]} нс" if u in MEASURED_TAU_NS else ""
    print(f"{u:3d} {ref.label(u):28s} τ = {tau:6.2f} нс{meas}   ветвления: {top}")
'''),
    md("## Ветвления для линий из списка (lines.csv) → `xe2_branching.csv`"),
    code('''
import csv
lines = [r for r in nb.read_csv(NIST / "lines.csv") if r["ion"] == "Xe II" and r["class"] in ("A", "B")]
out = []
for r in lines:
    u, l = int(r["upper_no"]), (int(r["lower_no"]) if r["lower_no"] else None)
    b = br.get(u, {}).get(l) if l else None
    out.append(dict(wl_nm=r["wl_nm"], upper_no=u, lower_no=l, upper=r["upper"], lower=r["lower"],
                    A_dbsr=A.get(u, {}).get(l), A_nist=r["A_nist"], branching=b, tau_ns=1e9 / tot[u] if u in tot else None,
                    observed=r.get("observed", ""), role=r.get("role", "")))
with open(W / "xe2_branching.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
for o in out:
    if o["role"]:
        print(f"{o['wl_nm']:>8}  {o['upper_no']:3d}->{o['lower_no']}  BR = {o['branching'] or float('nan'):.3f}  "
              f"A = {o['A_dbsr'] or float('nan'):.2e} (NIST {o['A_nist'] or '—'})  {o['role']}")
'''),
    md("""
## E2: времена жизни метастабилей

5d-уровни с J ≥ 7/2 не распадаются E1 в 5p⁵; их сток — E2 в другие чётные уровни (и M1,
который здесь не считается: `dbsr_dmat3` падает на M1 для c-файлов; для CRM это
несущественно — такие распады на порядки медленнее пролёта).
"""),
    code('''
for kind in ("E2",):
    f = W / f"transitions_{kind}.csv"
    if f.exists():
        _, t, _ = nb.branching_from_csv(f)
        for u in sorted(t)[:12]:
            print(kind, u, ref.label(u), f"A_sum = {t[u]:.3e} с⁻¹  (τ = {1 / t[u]:.3e} с)")
'''),
])

# ====================================================================== 02 Xe II line cross sections + CRM
notebook("02_xe2_line_cross_sections.ipynb", [
    md("""
# 02. Xe II: сечения и константы скоростей линий, предварительная CRM

По сечениям Ванга (возбуждение верхних уровней) и ветвлениям из ноутбука 01:

* **σ линии** = BR · σ(возбуждения) из основного 5p⁵ ²P₃/₂, из ²P₁/₂ и из метастабилей 6s, 5d;
* **константы скорости линий** k_line(Te) для максвелловской ФРЭЭ и ФРЭЭ Бугровой;
* **предварительная CRM** (уровни 1–56 + (¹S₀)6p) с временем пролёта τ: интенсивности
  линий и их чувствительность к ne и Te → выбор линий-диагностик.

Каскадов с 6d/7s/4f и переходов метастабиль↔метастабиль у Ванга нет — их добавит
ноутбук 03 (тогда CRM здесь можно пересобрать с `crm_xe2_ext.json`).
"""),
    code(PARAMS),
    code('''
ref = reference.read_xlsx(XLSX)
W = RUNS / "xe2_wang"
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")        # из ноутбука 01
lines = [r for r in nb.read_csv(NIST / "lines.csv") if r["ion"] == "Xe II" and r["class"] in ("A", "B")]
print(len(lines), "линий класса A/B")
'''),
    md("## Сечения линий: запрошенные 452.4, 659.5, 669.4 и любые другие"),
    code('''
SHOW = [452.421, 659.501, 669.432, 561.667, 545.045, 545.090]
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for ax, wl in zip(axs.flat, SHOW):
    r = min(lines, key=lambda r: abs(float(r["wl_nm"]) - wl))
    u, l = int(r["upper_no"]), int(r["lower_no"])
    b = br[u][l]
    for i, lab in ((1, "из ²P₃/₂"), (2, "из ²P₁/₂")):
        if (i, u) in ref.sigma:
            E, s = nb.line_sigma(ref, u, b, i)
            ax.plot(E, s / 1e-16, lw=0.8, label=lab)
    ax.set_xscale("log"); ax.set_title(f"{r['wl_nm']} нм  ({u}→{l}), BR = {b:.3f}", fontsize=9)
    ax.set_xlabel("E, эВ"); ax.set_ylabel("σ линии, 10⁻¹⁶ см²")
axs.flat[0].legend(); fig.tight_layout(); plt.show()
'''),
    md("""
## Таблица: сечения всех линий класса A/B → `xe2_line_sigma.xlsx`

Лист на линию: энергия, σ линии из ²P₃/₂, из ²P₁/₂ и из каждого метастабиля (в 10⁻¹⁶ см²), как
у Ванга. Плюс лист `rates` с k_line(Te).
"""),
    code('''
import openpyxl
TE = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
wb = openpyxl.Workbook(); wr = wb.active; wr.title = "rates"
wr.append(["λ, нм", "верхний", "нижний", "BR"] + [f"k(²P3/2), Te={t} эВ" for t in TE] + ["Σk(метаст.)/k(осн.), 1.1 эВ"])
for r in lines:
    u, l = int(r["upper_no"]), (int(r["lower_no"]) if r["lower_no"] else None)
    b = br.get(u, {}).get(l)
    if not b:
        continue
    ws = wb.create_sheet(f"{float(r['wl_nm']):.3f}")
    init = sorted(i for (i, j) in ref.sigma if j == u)
    for i in init:
        E, s = ref.sigma[(i, u)]
        ws.append([f"{i}->{u}", ref.label(i)]); ws.append(["Energy(eV)", "Sigma_line(1E-16 cm^2)"])
        for e, x in zip(E, s * b / 1e-16):
            ws.append([float(e), float(x)])
        ws.append([])
    E, s = ref.sigma[(1, u)]
    thr = ref.level(u).energy_cm / nb.CM_PER_EV
    k = [b * crm.rate_from_sigma(E, s, t, thr) for t in TE]
    km = sum(crm.rate_from_sigma(*ref.sigma[(i, u)], 1.1, (ref.level(u).energy_cm - ref.level(i).energy_cm) / nb.CM_PER_EV)
             for i in init if i > 2)
    wr.append([float(r["wl_nm"]), r["upper"], r["lower"], b] + k + [km / crm.rate_from_sigma(E, s, 1.1, thr)])
wb.save(W / "xe2_line_sigma.xlsx"); print("->", W / "xe2_line_sigma.xlsx")
'''),
    md("""
## Предварительная CRM на данных Ванга

Уровни 1–56, 101, 105. Возбуждение: всё, что есть у Ванга (основное, ²P₁/₂ → всё;
метастабили → 6p); девозбуждение — детальный баланс. Радиация — A из ноутбука 01
(E1 + M1/E2 для метастабилей). Сток всех уровней — пролёт τ (ион пролетает зону
наблюдения за ~L/v).

**Ограничения**: нет переходов метастабиль↔метастабиль, каскадов с 6d/7s/4f и
ионизации. Числа предварительные; цель — увидеть, какие линии чувствительны к ne.
"""),
    code('''
TE_GRID = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
keep = list(range(1, 57)) + [101, 105]
lv = {i: ref.level(i) for i in keep}
data = crm.CRMData("Xe II", [str(i) for i in keep], [lv[i].two_j + 1 for i in keep],
                   [lv[i].energy_cm for i in keep], [ref.label(i) for i in keep], TE_GRID)
for (i, j), (E, s) in ref.sigma.items():
    if i in lv and j in lv and lv[i].energy_cm < lv[j].energy_cm:
        thr = (lv[j].energy_cm - lv[i].energy_cm) / nb.CM_PER_EV
        data.add_rate(str(i), str(j), [crm.rate_from_sigma(E, s, t, thr) for t in TE_GRID], "Wang2019")
for kind in ("E1", "M1", "E2"):
    f = W / f"transitions_{kind}.csv"
    if f.exists():
        for u, d in nb.branching_from_csv(f)[2].items():
            for l, a in d.items():
                if u in lv and l in lv:
                    data.add_A(str(u), str(l), a + data.A.get((data.index[str(u)], data.index[str(l)]), 0.0))
data.save(W / "crm_xe2_wang_prelim.json")
print(len(data.k), "констант скорости,", len(data.A), "A")
'''),
    code('''
TAU = 3e-6          # с: время пролёта иона Xe+ через зону наблюдения (L ~ 1 см, v ~ 3 км/с)
SEL = [(r["upper_no"], r["lower_no"], r["wl_nm"]) for r in lines if r.get("role") and r["lower_no"]]
pairs = [(u, l) for u, l, _ in SEL]
print(f"{'λ, нм':>8} {'верх':>4}   " + "   ".join(f"ne={ne:.0e}: dlnI/dlnne dlnI/dlnTe" for ne in (1e10, 1e11, 1e12)))
res = {ne: crm.sensitivity(data, pairs, ne, 1.1, tau=TAU) for ne in (1e10, 1e11, 1e12)}
for u, l, wl in SEL:
    print(f"{wl:>8} {u:>4}   " + "   ".join(f"{res[ne][(u, l)]['dlnI_dlnne']:24.2f} {res[ne][(u, l)]['dlnI_dlnTe']:11.2f}"
                                         for ne in res))
'''),
    md("## Отношения линий против ne и Te (карта для диагностики)"),
    code('''
RATIOS = [((\"41\", \"15\"), (\"44\", \"16\"), "561.7/545.0"), ((\"31\", \"7\"), (\"29\", \"7\"), "547.3/553.1"),
          ((\"52\", \"33\"), (\"56\", \"28\"), "659.5/504.5"), ((\"49\", \"28\"), (\"30\", \"9\"), "627.1/594.6")]
NE = np.logspace(9.5, 12.5, 25); TES = [0.8, 1.1, 1.5, 2.0, 3.0]
fig, axs = plt.subplots(1, len(RATIOS), figsize=(16, 3.8))
for ax, (a, b, lab) in zip(axs, RATIOS):
    for te in TES:
        y = []
        for ne in NE:
            p = crm.steady_state(data, ne, te, tau=TAU)
            y.append(crm.line_emission(data, p, *a) / crm.line_emission(data, p, *b))
        ax.loglog(NE, y, label=f"Te = {te} эВ")
    ax.set_title(lab); ax.set_xlabel("ne, см⁻³")
axs[0].set_ylabel("отношение (фотоны)"); axs[0].legend(fontsize=7); fig.tight_layout(); plt.show()
'''),
])

# ====================================================================== 03 Xe II extended scattering
notebook("03_xe2_extended_scattering.ipynb", [
    md("""
# 03. e + Xe⁺, расширенная модель: 6d, 7s, 8s, 4f, 7p и переходы между метастабилями

Нужна для:
* линий **класса C** (верхние уровни 6d, 7s, 4f, (¹D₂)6d: 405.7, 418.0, 420.8, 423.8, 439.3,
  482.3, 486.2, 508.1, 531.4 нм и др.);
* **каскадов** в 6p (6d, 7s → 6p) — вклад в сечения линий класса A;
* **переходов метастабиль ↔ метастабиль** и из метастабилей в 6s/5d (у Ванга их нет).

Модель `xe2_ext` (`models.py`): чётные 5s5p⁶, 5p⁴5d, 6s, 7s, 6d (+ корреляционная 7d),
нечётные 5p⁵, 5p⁴6p, 7p, 4f; состояния до 19.2 эВ. Это примерно вдвое больше
состояний, чем у Ванга: матрицы ~5×, время ~10× (начните с JMAX = 0 и стресс-теста).

Перед рассеянием: энергии мишени против NIST и A (переходы 6d/7s → 6p против NIST).
"""),
    code(PARAMS),
    code('''
JMAX = 0            # 0 (проверка), затем 10, 25, 50
job = nb.Job(RUNS / "xe2_ext", "xe2_ext", stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=40, de=0.0136, multipoles="E1")
job.start()
'''),
    code("job.status()"),
    code('''
W = RUNS / "xe2_ext"
print((W / "target_table.txt").read_text())
'''),
    md("## Стресс-тест после J = 0"),
    code('''
import subprocess
r = subprocess.run([sys.executable, str(REPO / "tools" / "stress_test.py"), "--target", str(W / "target"),
                    "--scat", str(W / "scat"), "--skip-matrices"], capture_output=True, text=True)
print(r.stdout[-5000:] or r.stderr[-3000:])
'''),
    md("## Сравнение с Ванга для общих переходов"),
    code('''
ref = reference.read_xlsx(XLSX)
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz")
tg = db.Target.load(W / "target")
st = {s.name: s for s in tg.states}
# Wang number of a state: by its NIST energy
elv = [lv.energy_cm for lv in ref.levels]
wang_no = {n: nb.level_number(elv, s.exp_energy_cm) for n, s in st.items() if s.exp_energy_cm is not None}
name_of = {v: k for k, v in wang_no.items() if v and k in cs.names}
trs = [(i, j) for (i, j) in ref.sigma if i in name_of and j in name_of][:9]
fig, axs = plt.subplots(3, 3, figsize=(13, 9))
for ax, (i, j) in zip(axs.flat, trs):
    E, s = ref.sigma[(i, j)]
    ax.plot(E, s / 1e-16, lw=0.8, label="Wang 2019")
    ax.plot(cs.incident_energy(name_of[i]), cs.sigma(name_of[i], name_of[j]) / 1e-16, lw=0.8, label=f"xe2_ext, J ≤ {JMAX}")
    ax.set_xscale("log"); ax.set_title(f"{i}→{j} {ref.label(j)}", fontsize=8)
axs.flat[0].legend(fontsize=7); fig.tight_layout(); plt.show()
'''),
    md("## Сечения линий класса C и каскады"),
    code('''
from pydbsr.transitions import TransitionTable
trE1 = nb.read_csv(W / "transitions_E1.csv")
Atab = {}
for r in trE1:
    Atab.setdefault(r["upper"], {})[r["lower"]] = Atab.get(r["upper"], {}).get(r["lower"], 0) + float(r["A_exp"])
by_e = {round(s.exp_energy_cm, 1): n for n, s in st.items() if s.exp_energy_cm is not None}
def state_at(e):
    k = min(by_e, key=lambda x: abs(x - e)); return by_e[k] if abs(k - e) < 0.5 else None
lines = [r for r in nb.read_csv(NIST / "lines.csv") if r["ion"] == "Xe II" and r["class"] == "C" and r.get("observed") == "да"]
g = cs.names.index(tg.ground.name) if tg.ground.name in cs.names else 0
rows = []
for r in lines:
    u, l = state_at(float(r["Ek_cm"])), state_at(float(r["Ei_cm"]))
    if not u or not l or u not in cs.names or u not in Atab:
        continue
    b = Atab[u].get(l, 0) / sum(Atab[u].values())
    k = [b * cs.rate(cs.names[g], u, t * 11604.5)[0] for t in (1.0, 2.0, 5.0)]
    rows.append((float(r["wl_nm"]), r["upper"], b, *k))
for x in sorted(rows):
    print(f"{x[0]:8.3f}  {x[1]:28s} BR = {x[2]:.3f}  k_line(1, 2, 5 эВ) = {x[3]:.2e} {x[4]:.2e} {x[5]:.2e}")
'''),
    md("Данные CRM (все пары, A): `crm_xe2_ext.json` — пишет стадия `crm` (ноутбук 06)."),
])

# ====================================================================== 04 Ba II
notebook("04_ba2.ipynb", [
    md("""
# 04. e + Ba⁺: мишень, A, сечения 6s → 6p, 5d и из метастабилей 5d

Линии: **455.403** (6p ²P₃/₂ → 6s), **493.408** (6p ²P₁/₂ → 6s); ветви 614.171, 585.367, 649.690
(→ 5d) — для калибровки чувствительности.

Модель `ba2`: один валентный электрон над [Xe]: 6s, 5d, 7s, 6d, 8s, 7d | 6p, 7p, 4f, 5f
(17 состояний). Поляризации остова нет — проверяем по энергиям и A.

**Валидация**:
1. энергии против NIST (пороги всё равно экспериментальные);
2. A(455.4) = 1.11·10⁸, A(493.4) = 9.53·10⁷, A(614.2) = 4.12·10⁷, A(585.4) = 6.0·10⁶,
   A(649.7) = 3.10·10⁷ с⁻¹ (NIST, класс B), ветвления 6p;
3. времена жизни 5d₃/₂, 5d₅/₂ (E2, десятки секунд) против измеренных;
4. связанные состояния (N+1) = уровни Ba I против NIST;
5. сечение 6s → 6p: асимптотика Бете с f из расчёта; сравнение с опубликованными данными
   (впишите их в `LIT`).
"""),
    code(PARAMS),
    code('''
JMAX = 20           # Ba+: дёшево, можно сразу 20-30 (дипольные 6s->6p сходятся медленно)
job = nb.Job(RUNS / "ba2", "ba2", stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=30, de=0.005, multipoles="E1,E2")
job.start()
'''),
    code("job.status()"),
    md("## 1. Энергии"),
    code('''
W = RUNS / "ba2"
print((W / "target_table.txt").read_text())
'''),
    md("## 2–3. A, ветвления, времена жизни"),
    code('''
NIST_A = {455.403: 1.11e8, 493.408: 9.53e7, 614.171: 4.12e7, 585.367: 6.00e6, 649.690: 3.10e7,
          452.493: 6.63e7, 489.993: 1.04e8, 413.065: 2.18e8, 389.178: 2.17e8}
rows = nb.read_csv(W / "transitions_E1.csv")
for r in rows:
    wl = nb.fnum(r["wl_air_nm"])
    hit = [k for k in NIST_A if wl and abs(k - wl) < 0.05]
    if hit:
        a = float(r["A_exp"])
        print(f"{hit[0]:9.3f}  {r['upper_nist']:>18s} -> {r['lower_nist']:<18s}  DBSR {a:.3e}  NIST {NIST_A[hit[0]]:.3e}"
              f"  ratio {a / NIST_A[hit[0]]:.2f}  vel/len {r['gauge_ratio']}")
for kind in ("E1", "M1", "E2"):
    f = W / f"transitions_{kind}.csv"
    if f.exists():
        _, tot, _ = nb.branching_from_csv(f)
        print(kind, {u: f"{1 / t:.3g} с" for u, t in sorted(tot.items()) if u <= 5})
'''),
    md("## 4. Связанные состояния e + Ba⁺ = уровни Ba I"),
    code('''
# отдельный короткий прогон (J <= 4): dbsr_hd3 itype=-1 на заново построенных матрицах
jb = nb.Job(RUNS / "ba2_bound", "ba2", stage="bound", jmax=4, cores=CORES, mem=MEM, hd_threads=HD_THREADS)
jb.start()
'''),
    code('''
jb.status()
lv1 = db.nist.fetch_levels("Ba I")
for b in sorted(nb.read_csv(RUNS / "ba2_bound" / "bound_states.csv"), key=lambda b: float(b["E_cm"]))[:20]:
    e = float(b["E_cm"])                      # над самым нижним связанным состоянием (Ba I 6s2)
    near = min(lv1, key=lambda l: abs(l.energy_cm - e))
    print(f"{b['label']:>14s}  E = {e:9.1f} см-1  (связь {float(b['E_bind']):8.4f})   "
          f"ближайший NIST: {near.config} {near.term} J={near.J:g}  {near.energy_cm:9.1f}  Δ = {e - near.energy_cm:+7.0f}")
'''),
    md("## 5. Сечения и константы скорости"),
    code('''
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz")
tg = db.Target.load(W / "target")
st = {s.name: s for s in tg.states}
lab = {n: f"{st[n].nist_label}" for n in cs.names}
gs = cs.names[0]
LIT = {}            # литературные σ(6s->6p): {"источник": (E_эВ, σ_см2)}
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for j in cs.names[1:5]:
    axs[0].plot(cs.incident_energy(gs), cs.sigma(gs, j) / 1e-16, lw=0.8, label=f"6s → {lab[j]}")
for k, (E, s) in LIT.items():
    axs[0].plot(E, s / 1e-16, "o", ms=3, label=k)
for i in cs.names[1:3]:
    for j in cs.names[3:5]:
        axs[1].plot(cs.incident_energy(i), cs.sigma(i, j) / 1e-16, lw=0.8, label=f"{lab[i]} → {lab[j]}")
for ax in axs:
    ax.set_xscale("log"); ax.set_xlabel("E, эВ"); ax.set_ylabel("σ, 10⁻¹⁶ см²"); ax.legend(fontsize=7)
plt.show()
TE = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
for i in cs.names[:3]:
    for j in cs.names[1:5]:
        if st[j].exp_energy_cm > st[i].exp_energy_cm:
            print(f"{lab[i]:>12s} -> {lab[j]:<12s}", " ".join(f"{x:.2e}" for x in cs.rate(i, j, TE * 11604.5)))
'''),
    md("""
## CRM Ba II: отношение 455.4 / 493.4 против ne

²P₁/₂ из 5d₅/₂ дипольно не возбуждается, ²P₃/₂ — возбуждается: отношение растёт с
заселённостью 5d₅/₂, то есть с ne·τ. τ — время пролёта атома/иона бария через зону
наблюдения (Ba при ~1500 K: v ≈ 430 м/с; ионы быстрее).
"""),
    code('''
data = crm.CRMData.load(W / "crm_ba2.json")
n32 = [n for n in data.names if "2P" in (data.labels[data.index[n]] or "") and data.g[data.index[n]] == 4][0]
n12 = [n for n in data.names if "2P" in (data.labels[data.index[n]] or "") and data.g[data.index[n]] == 2][0]
g = data.names[0]
NE = np.logspace(9, 13, 30)
for tau in (2e-6, 2e-5):
    for te in (0.8, 1.1, 2.0):
        y = [crm.line_emission(data, p := crm.steady_state(data, ne, te, tau=tau), n32, g) /
             crm.line_emission(data, p, n12, g) for ne in NE]
        plt.semilogx(NE, y, label=f"τ = {tau:g} с, Te = {te} эВ")
plt.xlabel("ne, см⁻³"); plt.ylabel("I(455.4) / I(493.4), фотоны"); plt.legend(fontsize=7); plt.show()
'''),
])

# ====================================================================== 05 Ba I
notebook("05_ba1.ipynb", [
    md("""
# 05. e + Ba: мишень, f(553.5), сечения против Fursa et al (1999), сечения из 6s5d

Линия **553.548 нм** (6s² ¹S₀ – 6s6p ¹P₁), A = 1.19·10⁸ с⁻¹, f = 1.64, ветвление ≈ 0.997.

Модель `ba1`: два валентных электрона над [Xe]; чётные 6s², 6s5d, 5d², 6p², 6s7s, 6s6d,
нечётные 6s6p, 5d6p, 6s7p, 6s4f; состояния до 4.1 эВ. Сильное КВ 6s6p ¹P – 5d6p ¹P
определяет f(553.5) — первая проверка мишени.

**Валидация** (самая сильная из трёх):
1. энергии и f(553.5) = 1.64 (A = 1.19·10⁸);
2. σ(¹P₁) прямое против CCC (Fursa, табл. II);
3. **σ_app = σ + каскады** против измерения Chen & Gallagher (±5 %) — проверяет и σ, и A;
4. остальные σ против рекомендованных Fursa (табл. III–VI, от 5 эВ).

Новое (нет нигде): σ из метастабилей 6s5d ³D, ¹D → 6s6p ¹P₁ и 5d6p; σ ниже 5 эВ.
"""),
    code(PARAMS),
    code('''
JMAX = 0            # 0 (проверка), затем 15-25
job = nb.Job(RUNS / "ba1", "ba1", stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=40, de=0.005, multipoles="E1,E2")
job.start()
'''),
    code("job.status()"),
    code('''
W = RUNS / "ba1"
print((W / "target_table.txt").read_text())
'''),
    md("## 1. f и A резонансной линии и других линий Ba I против NIST"),
    code('''
nistA = [x for x in nb.nist_lines(NIST / "nist_BaI.csv") if x["A"]]
rows = nb.read_csv(W / "transitions_E1.csv")
for r in rows:
    wl = nb.fnum(r["wl_air_nm"])
    hit = [x for x in nistA if wl and abs(x["wl"] - wl) < 0.02]
    if hit:
        x = hit[0]; a = float(r["A_exp"])
        print(f"{x['wl']:9.3f} {r['upper_nist']:>18s} -> {r['lower_nist']:<16s} DBSR {a:.3e}  NIST {x['A']:.3e}  "
              f"ratio {a / x['A']:.2f}  gf {float(r['gf']):.3f}  vel/len {r['gauge_ratio']}")
'''),
    md("## 2–3. σ(6s² → 6s6p ¹P₁): прямое против CCC, с каскадами против Chen & Gallagher"),
    code('''
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz")
tg = db.Target.load(W / "target")
st = {s.name: s for s in tg.states}
lab = {n: st[n].nist_label for n in cs.names}
gs = cs.names[0]
p1 = [n for n in cs.names if lab[n] == "6s.6p 1P* J=1"][0]
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")
num = {n: st[n].nist_no for n in cs.names}
# каскад: σ_app(1P1) = σ(1P1) + Σ_u σ(u) · P(u -> ... -> 1P1); P через многократные ветвления
def reach(u, target, depth=4):
    if u == target: return 1.0
    if depth == 0 or u not in br: return 0.0
    return sum(b * reach(l, target, depth - 1) for l, b in br[u].items())
E = cs.incident_energy(gs)
direct = np.nan_to_num(cs.sigma(gs, p1))
casc = sum(np.nan_to_num(cs.sigma(gs, u)) * reach(num[u], num[p1]) for u in cs.names
           if st[u].exp_energy_cm > st[p1].exp_energy_cm)
F = nb.FURSA_1P1
plt.plot(E, direct / 1e-16, label="DBSR прямое")
plt.plot(E, (direct + casc) / 1e-16, label="DBSR + каскады")
plt.plot(F["E"], F["Q"], "s", ms=4, label="Fursa CCC (рек.), прямое")
plt.plot(F["E"], F["Q_app"], "o", ms=4, label="Chen & Gallagher ×1.03 (apparent)")
plt.xscale("log"); plt.xlim(2, 1000); plt.xlabel("E, эВ"); plt.ylabel("σ, 10⁻¹⁶ см²"); plt.legend(); plt.show()
'''),
    md("## 4. Остальные переходы из основного против Fursa"),
    code('''
import re
def find_state(key):                       # "6s6p 3P1" -> state with NIST label "6s.6p 3P* J=1"
    conf, tj = key.split()
    c = ".".join(re.findall(r"\\d[a-z](?:\\d(?![a-z]))?", conf))
    want = {f"{c} {tj[:2]} J={tj[2:]}", f"{c} {tj[:2]}* J={tj[2:]}"}
    return [n for n in cs.names if lab[n] in want]
fig, axs = plt.subplots(3, 4, figsize=(15, 9))
for ax, key in zip(axs.flat, nb.FURSA_TABLES):
    Ef, sf = nb.fursa(key)
    ax.plot(Ef, sf / 1e-16, "s", ms=3, label="Fursa")
    for n in find_state(key)[:1]:
        ax.plot(cs.incident_energy(gs), cs.sigma(gs, n) / 1e-16, lw=0.8, label="DBSR")
    ax.set_xscale("log"); ax.set_title(key, fontsize=9)
axs.flat[0].legend(fontsize=7); fig.tight_layout(); plt.show()
'''),
    md("## Новое: из метастабилей 6s5d в 6s6p ¹P₁ и константы скорости"),
    code('''
TE = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 3.0])
meta = [n for n in cs.names if "6s.5d" in (lab[n] or "")]
for i in [gs] + meta:
    plt.plot(cs.incident_energy(i), cs.sigma(i, p1) / 1e-16, lw=0.8, label=f"{lab[i]} → ¹P₁")
    print(f"{lab[i]:>16s} -> 6s6p 1P1  k(Te) =", " ".join(f"{x:.2e}" for x in cs.rate(i, p1, TE * 11604.5)))
plt.xscale("log"); plt.xlabel("E, эВ"); plt.ylabel("σ, 10⁻¹⁶ см²"); plt.legend(fontsize=7); plt.show()
'''),
    md("""
## CRM Ba I: насколько 553.5 отличается от корональной модели

Отношение I(553.5) из CRM к корональной оценке n_e·k(6s²→¹P₁) при разных ne и времени
пролёта τ. Если оно далеко от 1 — уравнение 2.1 диплома нужно заменить CRM.
"""),
    code('''
data = crm.CRMData.load(W / "crm_ba1.json")
g0, P1 = data.names[0], p1
kP = lambda te: float(np.exp(np.interp(np.log(te), np.log(data.Te), np.log(data.k[(0, data.index[P1])]))))
NE = np.logspace(9, 13, 30)
for tau in (2e-6, 2e-5):
    for te in (0.8, 1.1, 2.0):
        y = [crm.line_emission(data, crm.steady_state(data, ne, te, tau=tau), P1, g0) / (ne * kP(te) * 0.997)
             for ne in NE]
        plt.semilogx(NE, y, label=f"τ = {tau:g} с, Te = {te} эВ")
plt.xlabel("ne, см⁻³"); plt.ylabel("I(553.5): CRM / корональная"); plt.legend(fontsize=7); plt.show()
'''),
])

# ====================================================================== 06 CRM assembly
notebook("06_crm_assembly.ipynb", [
    md("""
# 06. Сборка CRM: ne, Te по линиям Xe II, затем n(Ba), n(Ba⁺)

1. Данные: `crm_xe2_ext.json` (или предварительный `crm_xe2_wang_prelim.json` из 02),
   `crm_ba1.json`, `crm_ba2.json`.
2. Измеренные интенсивности линий (площади гауссов, **с поправкой на спектральную
   чувствительность**, в фотонах) — таблица `MEASURED`.
3. Подгонка ne, Te (и τ) по отношениям линий Xe II (χ²).
4. n(Ba)/n(Xe⁺) и n(Ba⁺)/n(Xe⁺) из 553.5 и 455.4/493.4 при найденных ne, Te, с учётом
   пленения (фактор выхода).
"""),
    code(PARAMS),
    code('''
def load(*names):
    for n in names:
        p = RUNS / n
        if p.exists():
            print("данные:", p); return crm.CRMData.load(p)
xe = load("xe2_ext/crm_xe2_ext.json", "xe2_wang/crm_xe2_wang_prelim.json")
ba1 = load("ba1/crm_ba1.json"); ba2 = load("ba2/crm_ba2.json")
'''),
    md("""
## Измеренные интенсивности

λ (нм) → интенсивность в фотонах (относительные единицы, одна шкала для всех линий).
Пример — впишите свои. Для Xe II нужны номера уровней (верхний, нижний) в нумерации
данных `xe` (для данных Ванга это номера Ванга, см. `lines.csv`).
"""),
    code('''
MEASURED = {          # (верхний, нижний): (λ, интенсивность, относит. погрешность)
    # ("41", "15"): (561.667, 1.0, 0.05),
    # ("44", "16"): (545.045, 0.8, 0.10),
}
TAU_XE = 3e-6
'''),
    code('''
from scipy.optimize import least_squares
keys = list(MEASURED)
def model(par):
    ne, te = 10 ** par[0], par[1]
    p = crm.steady_state(xe, ne, te, tau=TAU_XE)
    return np.array([crm.line_emission(xe, p, *k) for k in keys])
def resid(par):
    m = model(par); y = np.array([MEASURED[k][1] for k in keys]); s = np.array([MEASURED[k][2] for k in keys])
    scale = np.sum(y * m / s**2) / np.sum(m * m / s**2)            # общая нормировка (n(Xe+)·геометрия)
    return (y - scale * m) / (s * y)
if len(keys) >= 3:
    fit = least_squares(resid, x0=[11.0, 1.2], bounds=([9, 0.3], [13.5, 10]))
    J = fit.jac; cov = np.linalg.pinv(J.T @ J) * max(1, 2 * fit.cost / max(1, len(keys) - 2))
    print(f"ne = {10**fit.x[0]:.2e} см-3 (± {np.log(10) * np.sqrt(cov[0,0]) * 100:.0f} %),  Te = {fit.x[1]:.2f} ± {np.sqrt(cov[1,1]):.2f} эВ")
else:
    print("впишите хотя бы 3 линии в MEASURED")
'''),
    md("## Пленение резонансных линий бария (оценка)"),
    code('''
L_CM = 1.0                                   # длина пути вдоль луча зрения, см
for n_ba in (1e8, 1e9, 1e10, 1e11):
    t553 = crm.doppler_k0(553.548, 1, 3, 1.19e8, 1500, 137.33) * n_ba * L_CM
    print(f"n(Ba) = {n_ba:.0e}:  τ0(553.5) = {t553:.2g},  фактор выхода {crm.escape_factor_doppler(t553):.2f}")
'''),
])
