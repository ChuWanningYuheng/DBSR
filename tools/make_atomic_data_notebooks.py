"""Generate the atomic-data notebooks examples/crm/0*_*.ipynb.

    python tools/make_atomic_data_notebooks.py

The notebooks compute only input data for a collisional-radiative model:
A-values and branching ratios, line cross sections, rate coefficients.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "examples" / "crm"

PARAMS = '''
# ---------- параметры (поменяйте пути под себя)
from pathlib import Path
import sys
REPO    = Path("/data/pydbsr_calc/DBSR_src")          # клон репозитория pydbsr
CRM     = REPO / "examples" / "crm"                   # nbtools.py, run_model.py, models.py
NIST    = REPO / "docs" / "crm"                       # выгрузки NIST ASD (nist_*.csv)
DATA    = Path("/data/pydbsr_calc/data")              # ваши файлы
XLSX    = DATA / "CrossSectionsIon_3.xlsx"            # Wang et al (2019), сечения e + Xe+
RUNS    = Path("/data/pydbsr_calc/runs/crm")          # результаты расчётов (большой диск)
SCRATCH, SCRATCH_GB = Path.home() / "pydbsr_tmp", 5   # временные файлы dbsr_mat (быстрый диск)
CORES, MEM, HD_THREADS = 32, 100, 8                   # ядра, ГБ памяти, потоки диагонализации
TE = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]   # сетка Te (эВ) для констант скоростей
sys.path.insert(0, str(CRM))
import numpy as np, matplotlib.pyplot as plt
import pydbsr as db
from pydbsr import rates, reference
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


JOB_STATUS = "job.status()          # ход расчёта (запускайте когда хотите)"

# ====================================================================== 01 Xe II A-values
notebook("01_xe2_A_values.ipynb", [
    md("""
# 01. Xe II: A, силы осцилляторов и ветвления для верхних уровней выбранных линий

Линии и номера уровней (Wang et al 2019, лист «NIST Level Table») — `nb.XE2_LINES`
(обоснование выбора: `docs/crm/lines_final.md`).

У этих уровней в NIST известны A лишь для 0–3 ветвей из 14–27 разрешённых, поэтому все
ветви считаются в DBSR: мишень как у Ванга (`models.xe2_wang`), затем `dbsr_mult3` +
`dbsr_dmat3` для каждой пары «верхний уровень — нижний уровень» (E1). A пересчитываются
на экспериментальные энергии (NIST, через таблицу уровней Ванга): A ∝ S·ΔE³.

Ветви считаются для всех 19 уровней 5p⁴6p (секунды), чтобы сравнить с каждым A из NIST
(у выбранных уровней их всего 5). Расчёт только структурный (без рассеяния), идёт отдельным
процессом; замер на 4 ядрах: мишень ~5 мин, переходы ~10 с.

**Проверки**: энергии; все A против NIST (21 линия); отношение калибровок скорость/длина;
отношения A для линий с общим верхним уровнем против измеренных отношений интенсивностей
(спектр A3, `docs/crm/criteria_A3.csv`). Ветвления выдаются в двух вариантах: только DBSR
и «A из NIST там, где они есть, остальные ветви из DBSR».

**Внимание**: в тестовом прогоне ветви уровня 44 (линия 545.045) противоречили спектру,
поэтому 545.045 заменена на 557.219 (уровень 42). Для уровней 29, 39, 41, 42, 51 расчёт
со спектром и NIST согласуется (`docs/crm/lines_final.md`).

**Выход**: `xe2_wang/transitions_E1.csv` (все рассчитанные ветви) и
`xe2_wang/xe2_A_branching.xlsx` / `.csv` (ветви выбранных верхних уровней: λ, A, gf, S,
отношение калибровок скорость/длина, ветвление, A NIST там, где оно есть).
"""),
    code(PARAMS),
    code('''
UPPERS = sorted(set(nb.XE2_UPPERS) | set(nb.XE2_VALIDATION_UPPERS))   # выбранные + все 6p для проверки
job = nb.Job(RUNS / "xe2_wang", "xe2_wang", levels=XLSX, stage="transitions",
             uppers=",".join(map(str, UPPERS)), cores=CORES, multipoles="E1")
job.start()
'''),
    code(JOB_STATUS),
    md("## Энергии мишени против NIST (проверка мишени)"),
    code('''
W = RUNS / "xe2_wang"
print((W / "target_table.txt").read_text())
'''),
    md("""
## Ветви уровней 6p: DBSR и NIST

`A_exp` — A при экспериментальной энергии перехода; `vel/len` — отношение A в калибровках
скорости и длины (далеко от 1 — ненадёжная ветвь); `A_NIST` — из `docs/crm/nist_XeII*.csv`.
"""),
    code('''
import csv
ref = reference.read_xlsx(XLSX)
elv = [lv.energy_cm for lv in ref.levels]
nistA = {}
for f in ("nist_XeII.csv", "nist_XeII_UV.csv", "nist_XeII_IR.csv"):
    if (NIST / f).exists():
        for x in nb.nist_lines(NIST / f):
            u, l = nb.level_number(elv, x["Ek"]), nb.level_number(elv, x["Ei"])
            if u and l and x["A"]:
                nistA[(u, l)] = x["A"]
rows = []
for r in nb.read_csv(W / "transitions_E1.csv"):
    if not r["upper_no"] or not r["lower_no"]:
        continue
    u, l = int(r["upper_no"]), int(r["lower_no"])
    rows.append(dict(upper=u, upper_label=ref.label(u), lower=l, lower_label=ref.label(l),
                     wl_air_nm=nb.fnum(r["wl_air_nm"]), A_exp=float(r["A_exp"]), A_calc=float(r["A_calc"]),
                     gf=float(r["gf"]), S=float(r["S"]), vel_len=nb.fnum(r["gauge_ratio"]),
                     A_NIST=nistA.get((u, l))))
tot, tot_h = {}, {}
for r in rows:
    r["A_hybrid"] = r["A_NIST"] or r["A_exp"]            # NIST там, где есть, иначе DBSR
    tot[r["upper"]] = tot.get(r["upper"], 0) + r["A_exp"]
    tot_h[r["upper"]] = tot_h.get(r["upper"], 0) + r["A_hybrid"]
for r in rows:
    r["branching"] = r["A_exp"] / tot[r["upper"]]               # только DBSR
    r["branching_hybrid"] = r["A_hybrid"] / tot_h[r["upper"]]   # A NIST, где известны
    r["tau_ns"] = 1e9 / tot[r["upper"]]
rows.sort(key=lambda r: (r["upper"], -r["A_exp"]))
cmp = np.array([(r["A_exp"], r["A_NIST"]) for r in rows if r["A_NIST"]])
q = cmp[:, 0] / cmp[:, 1]
print(f"все A с NIST: {len(q)} линий, медиана DBSR/NIST = {np.median(q):.2f}, в пределах ×2: {np.sum((q > 0.5) & (q < 2))}")
plt.loglog(cmp[:, 1], cmp[:, 0], "o"); x = np.array([1e6, 3e8]); plt.plot(x, x, "k-", x, 2 * x, "k:", x, x / 2, "k:")
plt.xlabel("A NIST, с⁻¹"); plt.ylabel("A DBSR, с⁻¹"); plt.show()
for u in nb.XE2_UPPERS:
    print(f"\\n{u} {ref.label(u)}   ΣA = {tot[u]:.3e} с-1  (τ = {1e9 / tot[u]:.2f} нс)")
    for r in [r for r in rows if r["upper"] == u]:
        nist = f"NIST {r['A_NIST']:.2e} ({r['A_exp'] / r['A_NIST']:.2f})" if r["A_NIST"] else ""
        print(f"   -> {r['lower']:3d} {r['lower_label']:26s} {r['wl_air_nm'] or 0:9.3f} нм  A = {r['A_exp']:.3e}  "
              f"BR = {r['branching']:.4f}  gf = {r['gf']:.3e}  vel/len = {r['vel_len']}  {nist}")
'''),
    md("""
## Проверка по спектру: линии с общим верхним уровнем

Отношение числа фотонов двух линий одного верхнего уровня равно отношению их A (CRM не
нужна). Измеренные амплитуды — спектр A3 (`docs/crm/criteria_A3.csv`, лист `SHEET_CHECK`,
выдержка 0.4 с: насыщенных линий нет); для 545.045/545.090 — фит дублета
(`criteria_A3_doublet.csv`). Калибровки спектральной чувствительности нет, поэтому
сравнивайте прежде всего линии, близкие по длине волны.
"""),
    code('''
SHEET_CHECK = "data10"
crit = {round(float(r["wl_nm"]), 3): r for r in nb.read_csv(NIST / "criteria_A3.csv") if r["ion"] == "XeII"}
d2 = [d for d in nb.read_csv(NIST / "criteria_A3_doublet.csv") if d["sheet"] == SHEET_CHECK][0]
meas = {545.045: float(d2["amp1"]), 545.090: float(d2["amp2"])}
for wl, r in crit.items():
    if r.get(f"{SHEET_CHECK}_amp") and not r.get(f"{SHEET_CHECK}_sat") and wl not in meas:
        meas[wl] = float(r[f"{SHEET_CHECK}_amp"])
A_of, A_hyb = {}, {}
for wl, x in crit.items():
    if x.get("upper_no") and x.get("lower_no"):
        k = (int(x["upper_no"]), int(x["lower_no"]))
        a = [r for r in rows if (r["upper"], r["lower"]) == k]
        if a:
            A_of[wl], A_hyb[wl] = a[0]["A_exp"], a[0]["A_hybrid"]
print(f"{'пара':>18s}  {'измерено':>9s}  {'DBSR':>8s}  {'NIST/DBSR':>9s}")
for w1, w2 in nb.XE2_SAME_UPPER:
    if w1 in A_of and w2 in A_of:
        m = meas[w2] / meas[w1] if w1 in meas and w2 in meas else float("nan")
        print(f"{w2:8.3f}/{w1:8.3f}  {m:9.2f}  {A_of[w2] / A_of[w1]:8.2f}  {A_hyb[w2] / A_hyb[w1]:9.2f}")
'''),
    code('''
import openpyxl
out = W / "xe2_A_branching.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "branches"
ws.append(list(rows[0]))
for r in rows:
    ws.append(list(r.values()))
ws2 = wb.create_sheet("selected lines")
ws2.append(["wl_nm (NIST)", "upper", "lower", "role", "A_exp (DBSR)", "branching (DBSR)", "A_NIST",
            "branching (NIST A где есть, иначе DBSR)"])
for x in nb.XE2_LINES:
    r = [r for r in rows if r["upper"] == x["upper"] and r["lower"] == x["lower"]]
    ws2.append([x["wl"], x["upper"], x["lower"], x["role"], r[0]["A_exp"] if r else None,
                r[0]["branching"] if r else None, nistA.get((x["upper"], x["lower"])),
                r[0]["branching_hybrid"] if r else None])
wb.save(W / "xe2_A_branching.xlsx")
print("->", out, "и", W / "xe2_A_branching.xlsx")
'''),
])

# ====================================================================== 02 Xe II line cross sections and rates
notebook("02_xe2_line_cross_sections.ipynb", [
    md("""
# 02. Xe II: сечения выбранных линий и константы скоростей

* **σ возбуждения** верхнего уровня u из уровня i — Wang et al (2019), CrossSectionsIon.xlsx
  (листы gs->6p, 6s->6p, 5d->6p; i = 1, 2 — 5p⁵ ²P₃/₂, ²P₁/₂; i ≥ 4 — 6s, 5d);
* **ветвление** BR = A_line/ΣA — ноутбук 01 (`xe2_A_branching.csv`); для каждой линии
  выбран вариант «только DBSR» или «A из NIST, где есть» (`nb.XE2_LINES`, поле `br`);
* **σ линии** = BR · σ(i → u) для всех i, для которых у Ванга есть данные;
* **константы скоростей** ⟨σv⟩(Te) — максвелловское распределение, сетка `TE` (первая ячейка):
  для возбуждения уровня (k(i→u)) и для линии (BR·k(i→u)).

Каскады в верхние уровни здесь не учитываются (это часть модели CRM).

**Выход** (в папке `xe2_wang`): `xe2_line_sigma.xlsx` (формат Ванга: пары столбцов
Energy(eV) / Sigma(1E-16 cm^2), над столбцом σ — метка «i->u (λ)»), `xe2_k_excitation.xlsx`
и `xe2_k_line.xlsx` (+ `.csv`): строки — Te, столбцы — переходы.
"""),
    code(PARAMS),
    code('''
ref = reference.read_xlsx(XLSX)
W = RUNS / "xe2_wang"
tab = {(int(r["upper"]), int(r["lower"])): r for r in nb.read_csv(W / "xe2_A_branching.csv")}
col = {"dbsr": "branching", "hybrid": "branching_hybrid"}      # см. nb.XE2_LINES, поле br
lines = [dict(x, BR=float(tab[(x["upper"], x["lower"])][col[x["br"]]])) for x in nb.XE2_LINES]
for x in lines:
    init = sorted(i for (i, j) in ref.sigma if j == x["upper"])
    print(f"{x['wl']:8.3f}  {x['upper']}->{x['lower']}  BR = {x['BR']:.4f} ({x['br']})  σ у Ванга из уровней: {init}")
'''),
    code('''
sig, lab, sheet = {}, {}, {}
for x in lines:
    u = x["upper"]
    for (i, j), (E, s) in ref.sigma.items():
        if j == u:
            key = (i, u, x["wl"])
            sig[key] = (E, x["BR"] * s)
            lab[key] = f"{i}->{u} ({x['wl']:.3f})"
            sheet[key] = ref.sheet[(i, j)]
out = reference.write_xlsx(W / "xe2_line_sigma.xlsx", ref.levels, sig, sheet_of=lambda k: sheet[k], labels=lab)
print(len(sig), "сечений линий ->", out)
'''),
    md("## Сечения линий из основного состояния и из ²P₁/₂"),
    code('''
fig, axs = plt.subplots(1, len(lines), figsize=(4 * len(lines), 3.5), squeeze=False)
for ax, x in zip(axs[0], lines):
    for i in (1, 2):
        if (i, x["upper"], x["wl"]) in sig:
            E, s = sig[(i, x["upper"], x["wl"])]
            ax.plot(E, s / 1e-16, lw=0.8, label=f"из {i}: {ref.label(i)}")
    ax.set_xscale("log"); ax.set_title(f"{x['wl']} нм ({x['upper']}→{x['lower']}), BR = {x['BR']:.3f}", fontsize=9)
    ax.set_xlabel("E, эВ"); ax.set_ylabel("σ линии, 10⁻¹⁶ см²")
axs[0][0].legend(fontsize=7); fig.tight_layout(); plt.show()
'''),
    md("""
## Константы скоростей ⟨σv⟩(Te)

Ниже первой табличной точки σ держится равным первому значению (для иона σ конечно на
пороге), выше последней (109 эВ) — спадает как 1/E (`rates.rate_from_sigma`). При Te ≤ 20 эВ
вклад E > 109 эВ мал; проверьте по `tail="dipole"`, если нужно.
"""),
    code('''
TEa = np.array(TE)
k_exc, k_line, lab_e, lab_l = {}, {}, {}, {}
for x in lines:
    u = x["upper"]
    for (i, j), (E, s) in ref.sigma.items():
        if j != u:
            continue
        thr = (ref.level(u).energy_cm - ref.level(i).energy_cm) / nb.CM_PER_EV
        k = rates.rates_on_grid(E, s, TEa, thr)
        k_exc[(i, u)] = k
        lab_e[(i, u)] = f"{i}->{u}"
        k_line[(i, u, x["wl"])] = x["BR"] * k
        lab_l[(i, u, x["wl"])] = f"{i}->{u} ({x['wl']:.3f})"
reference.write_rates_xlsx(W / "xe2_k_excitation.xlsx", TEa, k_exc, lab_e,
                           note="<sigma v> (cm3/s), excitation i->u, sigma: Wang et al 2019 (CrossSectionsIon.xlsx), Maxwellian")
reference.write_rates_xlsx(W / "xe2_k_line.xlsx", TEa, k_line, lab_l,
                           note="BR * <sigma v> (cm3/s), line from u; BR: DBSR (notebook 01); sigma: Wang et al 2019")
print("->", W / "xe2_k_excitation.xlsx", W / "xe2_k_line.xlsx")
for x in lines:
    k = k_line[(1, x["upper"], x["wl"])]
    print(f"{x['wl']:8.3f} из 1: " + "  ".join(f"{t:g} эВ: {v:.2e}" for t, v in zip(TEa, k)))
'''),
])

# ====================================================================== 03 Ba II
notebook("03_ba2.ipynb", [
    md("""
# 03. e + Ba⁺: сечения и константы скоростей для 455.403 и 493.408 нм

* A и ветвления — NIST ASD (известны все ветви: 6p ²P₃/₂ → 6s, 5d₅/₂, 5d₃/₂; 6p ²P₁/₂ → 6s,
  5d₃/₂; класс точности B) — `nb.BA_LINES`;
* сечения возбуждения — **считаются здесь** (готовых данных нет в наших источниках):
  модель `ba2` (`models.py`): один валентный электрон над [Xe]; 6s, 5d, 7s, 6d, 8s, 7d |
  6p, 7p, 4f, 5f (17 состояний), пороги — NIST.

Проверки модели: энергии против NIST; A против NIST; времена жизни 5d (E2) против
измеренных (впишите из литературы); связанные (N+1)-состояния = уровни Ba I.

Время (замер на 4 ядрах, J ≤ 3, 8 волн): самая большая матрица 10730, диагонализация одной
волны до 24 мин. Полный расчёт (J ≤ 20) — оценка: часы на 32 ядрах.

**Выход** (папка `ba2`): `sigma_ba2.xlsx` и `rates_ba2.xlsx` (все возбуждения между 17
состояниями, формат Ванга, номера NIST), `ba2_line_sigma.xlsx`, `ba2_k_line.xlsx` (+ .csv).
"""),
    code(PARAMS),
    code('''
JMAX = 20           # сходимость по J проверьте: 15 / 20 / 25
job = nb.Job(RUNS / "ba2", "ba2", stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=30, de=0.005, te=",".join(map(str, TE)))
job.start()
'''),
    code(JOB_STATUS),
    md("## 1. Энергии мишени"),
    code('''
W = RUNS / "ba2"
print((W / "target_table.txt").read_text())
'''),
    md("## 2. A против NIST, времена жизни 5d"),
    code('''
NIST_A = {}
for x in nb.BA_LINES:
    if x["ion"] == "Ba II":
        NIST_A.update(x["A_branches"])
for r in nb.read_csv(W / "transitions_E1.csv"):
    wl = nb.fnum(r["wl_air_nm"])
    hit = [k for k in NIST_A if wl and abs(k - wl) < 0.05]
    if hit:
        a = float(r["A_exp"])
        print(f"{hit[0]:9.3f}  {r['upper_nist']:>14s} -> {r['lower_nist']:<14s}  DBSR {a:.3e}  NIST {NIST_A[hit[0]]:.3e}"
              f"  DBSR/NIST {a / NIST_A[hit[0]]:.2f}  vel/len {r['gauge_ratio']}")
MEASURED_TAU_S = {}        # впишите: {"5d 2D J=3/2": τ (с), "5d 2D J=5/2": τ (с)} со ссылкой
_, tot, _ = nb.branching_from_csv(W / "transitions_E2.csv")
for r in nb.read_csv(W / "transitions_E2.csv"):
    if r["upper_nist"].startswith("5d") and r["lower_nist"].startswith("6s"):
        u = int(r["upper_no"])
        print(f"E2 {r['upper_nist']}: τ = {1 / tot[u]:.1f} с  (изм.: {MEASURED_TAU_S.get(r['upper_nist'], '—')})")
'''),
    md("## 3. Связанные состояния e + Ba⁺ = уровни Ba I (отдельный короткий прогон, J ≤ 4)"),
    code('''
jb = nb.Job(RUNS / "ba2_bound", "ba2", stage="bound", jmax=4, cores=CORES, mem=MEM, hd_threads=HD_THREADS)
jb.start()
'''),
    code('''
jb.status()
lv1 = db.nist.fetch_levels("Ba I")
for b in sorted(nb.read_csv(RUNS / "ba2_bound" / "bound_states.csv"), key=lambda b: float(b["E_cm"]))[:20]:
    e = float(b["E_cm"])                      # над самым нижним связанным состоянием (Ba I 6s2)
    near = min(lv1, key=lambda l: abs(l.energy_cm - e))
    print(f"{b['label']:>14s}  E = {e:9.1f} см-1   ближайший NIST: {near.config} {near.term} J={near.J:g}"
          f"  {near.energy_cm:9.1f}  Δ = {e - near.energy_cm:+7.0f}")
'''),
    md("## 4. Сечения линий 455.4, 493.4 и константы скоростей"),
    code('''
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz")
tg = db.Target.load(W / "target")
st = {s.name: s for s in tg.states if s.name in cs.names}
name_of = {s.nist_label: n for n, s in st.items()}
TEa = np.array(TE)
sig, lab, k_line, lab_l = {}, {}, {}, {}
fig, ax = plt.subplots(figsize=(7, 4))
for x in [x for x in nb.BA_LINES if x["ion"] == "Ba II"]:
    u = name_of[x["upper"]]
    for i, si in st.items():
        if si.exp_energy_cm >= st[u].exp_energy_cm:
            continue
        E, s = cs.incident_energy(i), cs.sigma(i, u)
        m = np.isfinite(s) & (E > 0)
        key = (si.nist_no, st[u].nist_no, x["wl"])
        sig[key] = (E[m], x["branching"] * s[m])
        lab[key] = f"{si.nist_no}->{st[u].nist_no} ({x['wl']:.3f})"
        k_line[key] = x["branching"] * cs.rate(i, u, TEa * 11604.518)
        lab_l[key] = lab[key]
        if si.nist_no == 1:
            ax.plot(E[m], x["branching"] * s[m] / 1e-16, lw=0.8, label=f"{x['wl']} нм из 6s")
ax.set_xscale("log"); ax.set_xlabel("E, эВ"); ax.set_ylabel("σ линии, 10⁻¹⁶ см²"); ax.legend(); plt.show()
lv = [l for l in db.nist.fetch_levels("Ba II") if l.no in {s.nist_no for s in st.values()}]
reference.write_xlsx(W / "ba2_line_sigma.xlsx", lv, sig, sheet_of=lambda k: f"{k[2]:.3f}", labels=lab)
reference.write_rates_xlsx(W / "ba2_k_line.xlsx", TEa, k_line, lab_l,
                           note="BR * <sigma v> (cm3/s); sigma: pydbsr model ba2 (this run); BR: NIST ASD")
print("->", W / "ba2_line_sigma.xlsx", W / "ba2_k_line.xlsx")
'''),
])

# ====================================================================== 04 Ba I
notebook("04_ba1.ipynb", [
    md("""
# 04. e + Ba: сечение и константы скоростей для 553.548 нм

* A и ветвление — NIST ASD: 6s6p ¹P₁ → 6s² (553.548, 1.19·10⁸, A+), → 6s5d ¹D₂ (1499.985,
  2.5·10⁵, B), → ³D₂ (1130.303, 1.1·10⁵, C), → ³D₁ (1107.570, 3.1·10³, D+): BR = 0.997.
* σ из основного состояния — **готово**: Fursa et al, PRA 60, 4590 (1999), табл. II (прямое
  Q и «apparent» Q_app = Chen & Gallagher 1976 × 1.03), 2.5–897.6 эВ; ниже 5 эВ только
  точки 2.5, 3, 4 эВ.
* σ из метастабилей 6s5d ³D₁,₂,₃, ¹D₂ в ¹P₁ и подробная околопороговая область —
  **считаются здесь**: модель `ba1` (`models.py`): два валентных электрона над [Xe],
  состояния до 4.1 эВ.

Проверки: f(553.5) и A против NIST; σ(¹P₁) против Fursa (прямое) и Q_app (Chen & Gallagher)
с каскадами из этого же расчёта; остальные σ против Fursa, табл. III–VI.

Время: мишень и A — 2 мин 20 с (замер, 2 ядра; 39 из 49 состояний сопоставлены с NIST); рассеяние
не замерено, оценка — сутки на сервере.

**Выход** (папка `ba1`): `sigma_ba1.xlsx`, `rates_ba1.xlsx` (все возбуждения, формат Ванга),
`ba1_line_sigma.xlsx`, `ba1_k_line.xlsx` (+ .csv; для 6s² — и по Fursa, и по DBSR).
"""),
    code(PARAMS),
    md("## 0. Готовое: σ линии 553.5 из основного состояния по Fursa et al (1999), табл. II"),
    code('''
x553 = nb.BA_LINES[0]
F = nb.FURSA_1P1
TEa = np.array(TE)
k_fursa = x553["branching"] * rates.rates_on_grid(F["E"], F["Q"] * 1e-16, TEa, 18060.261 / nb.CM_PER_EV, ion=False)
print("σ ниже 2.5 эВ: линейно от порога 2.239 эВ до первой точки (ion=False)")
for t, k in zip(TEa, k_fursa):
    print(f"Te = {t:5.2f} эВ   BR·<σv> = {k:.3e} см3/с")
'''),
    code('''
JMAX = 0            # 0 (проверка), затем 15-25
job = nb.Job(RUNS / "ba1", "ba1", stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=40, de=0.005, te=",".join(map(str, TE)))
job.start()
'''),
    code(JOB_STATUS),
    code('''
W = RUNS / "ba1"
print((W / "target_table.txt").read_text())
'''),
    md("## 1. A и gf против NIST"),
    code('''
nistA = [x for x in nb.nist_lines(NIST / "nist_BaI.csv") + nb.nist_lines(NIST / "nist_BaI_IR.csv") if x["A"]]
for r in nb.read_csv(W / "transitions_E1.csv"):
    wl = nb.fnum(r["wl_air_nm"])
    hit = [x for x in nistA if wl and abs(x["wl"] - wl) < 0.02]
    if hit:
        x = hit[0]; a = float(r["A_exp"])
        print(f"{x['wl']:9.3f} {r['upper_nist']:>18s} -> {r['lower_nist']:<16s} DBSR {a:.3e}  NIST {x['A']:.3e}  "
              f"DBSR/NIST {a / x['A']:.2f}  gf {float(r['gf']):.3f}  vel/len {r['gauge_ratio']}")
'''),
    md("## 2. σ(6s² → 6s6p ¹P₁): прямое против Fursa, с каскадами против Chen & Gallagher"),
    code('''
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz")
tg = db.Target.load(W / "target")
st = {s.name: s for s in tg.states if s.name in cs.names}
lab = {n: s.nist_label for n, s in st.items()}
gs = min(st, key=lambda n: st[n].exp_energy_cm)
p1 = [n for n in st if lab[n] == "6s.6p 1P* J=1"][0]
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")
num = {n: st[n].nist_no for n in st}
def reach(u, target, depth=4):          # вероятность попасть из u в target через цепочку распадов
    if u == target: return 1.0
    if depth == 0 or u not in br: return 0.0
    return sum(b * reach(l, target, depth - 1) for l, b in br[u].items())
E = cs.incident_energy(gs)
direct = np.nan_to_num(cs.sigma(gs, p1))
casc = sum(np.nan_to_num(cs.sigma(gs, u)) * reach(num[u], num[p1]) for u in st
           if st[u].exp_energy_cm > st[p1].exp_energy_cm)
plt.plot(E, direct / 1e-16, label="DBSR прямое")
plt.plot(E, (direct + casc) / 1e-16, label="DBSR + каскады")
plt.plot(F["E"], F["Q"], "s", ms=4, label="Fursa 1999, табл. II, Q")
plt.plot(F["E"], F["Q_app"], "o", ms=4, label="Fursa 1999, табл. II, Q_app (Chen & Gallagher ×1.03)")
plt.xscale("log"); plt.xlim(2, 1000); plt.xlabel("E, эВ"); plt.ylabel("σ, 10⁻¹⁶ см²"); plt.legend(fontsize=7); plt.show()
'''),
    md("## 3. Остальные σ из основного против Fursa (табл. III, IV, VI)"),
    code('''
import re
def find_state(key):                       # "6s6p 3P1" -> состояние с меткой NIST "6s.6p 3P* J=1"
    conf, tj = key.split()
    c = ".".join(re.findall(r"\\d[a-z](?:\\d(?![a-z]))?", conf))
    want = {f"{c} {tj[:2]} J={tj[2:]}", f"{c} {tj[:2]}* J={tj[2:]}"}
    return [n for n in st if lab[n] in want]
fig, axs = plt.subplots(3, 4, figsize=(15, 9))
for ax, key in zip(axs.flat, nb.FURSA_TABLES):
    Ef, sf = nb.fursa(key)
    ax.plot(Ef, sf / 1e-16, "s", ms=3, label="Fursa")
    for n in find_state(key)[:1]:
        ax.plot(cs.incident_energy(gs), cs.sigma(gs, n) / 1e-16, lw=0.8, label="DBSR")
    ax.set_xscale("log"); ax.set_title(key, fontsize=9)
axs.flat[0].legend(fontsize=7); fig.tight_layout(); plt.show()
'''),
    md("## 4. Сечение линии 553.5 из 6s² и из 6s5d; константы скоростей"),
    code('''
sig, labs, k_line = {}, {}, {}
for i in [gs] + [n for n in st if (lab[n] or "").startswith("6s.5d")]:
    Ei, s = cs.incident_energy(i), cs.sigma(i, p1)
    m = np.isfinite(s) & (Ei > 0)
    key = (num[i], num[p1], 553.548)
    sig[key] = (Ei[m], x553["branching"] * s[m])
    labs[key] = f"{num[i]}->{num[p1]} (553.548)"
    k_line[key] = x553["branching"] * cs.rate(i, p1, TEa * 11604.518)
    plt.plot(Ei[m], x553["branching"] * s[m] / 1e-16, lw=0.8, label=f"из {lab[i]}")
plt.xscale("log"); plt.xlabel("E, эВ"); plt.ylabel("σ линии 553.5, 10⁻¹⁶ см²"); plt.legend(fontsize=7); plt.show()
k_line[("Fursa", 9, 553.548)] = k_fursa
labs[("Fursa", 9, 553.548)] = "1->9 (553.548) Fursa1999 tab.II"
lv = [l for l in db.nist.fetch_levels("Ba I") if l.no in set(num.values())]
reference.write_xlsx(W / "ba1_line_sigma.xlsx", lv, {k: v for k, v in sig.items()}, sheet_of=lambda k: "553.548",
                     labels=labs)
reference.write_rates_xlsx(W / "ba1_k_line.xlsx", TEa, k_line, labs,
                           note="BR * <sigma v> (cm3/s); sigma: pydbsr model ba1 (this run) and Fursa et al 1999 tab. II; BR = 0.997 (NIST)")
print("->", W / "ba1_line_sigma.xlsx", W / "ba1_k_line.xlsx")
'''),
])
