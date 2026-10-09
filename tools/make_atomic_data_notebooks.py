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
ROOT    = Path("/data/files_Otdel120/pydbsr_calc")    # папка установки (большой диск)
REPO    = ROOT / "pydbsr_src"                         # распакованный архив pydbsr (или клон репозитория)
CRM     = REPO / "examples" / "crm"                   # nbtools.py, run_model.py, models.py
NIST    = REPO / "docs" / "crm"                       # выгрузки NIST ASD (nist_*.csv)
DATA    = ROOT                                        # ваши файлы
XLSX    = DATA / "CrossSectionsIon_3.xlsx"            # Wang et al (2019), сечения e + Xe+
RUNS    = ROOT / "runs" / "crm"                       # результаты расчётов (большой диск)
XE2_RUN = ROOT / "runs" / "xe_plus_v2"                # готовый расчёт e + Xe+ (модель Ванга, J <= 25)
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
* **ветвление** BR = A_line/ΣA — ноутбук 01 (`xe2_A_branching.csv`) или измеренное по
  спектрам (`nb.XE2_BR_MEASURED`, `docs/crm/branching_measured.csv`); для каждой линии
  выбран вариант «только DBSR», «A из NIST, где есть» или «измерено» (`nb.XE2_LINES`, поле
  `br`; обоснование — `docs/crm/lines_final.md`, п. 6.1);
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
def br_of(x):
    if x["br"] == "measured":                                   # измерено по спектрам (nb.XE2_BR_MEASURED)
        return nb.XE2_BR_MEASURED[x["wl"]]["BR"]
    return float(tab[(x["upper"], x["lower"])][col[x["br"]]])
lines = [dict(x, BR=br_of(x)) for x in nb.XE2_LINES]
for x in lines:
    init = sorted(i for (i, j) in ref.sigma if j == x["upper"])
    r = tab.get((x["upper"], x["lower"]), {})
    other = "  ".join(f"{k}={float(r[c]):.4f}" for k, c in col.items() if r.get(c))
    m = nb.XE2_BR_MEASURED.get(x["wl"])
    meas = f"  измерено={m['BR']:.4f} ({m['range'][0]:.4f}-{m['range'][1]:.4f})" if m else ""
    print(f"{x['wl']:8.3f}  {x['upper']}->{x['lower']}  BR = {x['BR']:.4f} ({x['br']})   [{other}{meas}]"
          f"  σ у Ванга из уровней: {init}")
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
                           note="BR * <sigma v> (cm3/s), line from u; BR: see nb.XE2_LINES (DBSR / DBSR+NIST, notebook 01; or measured, docs/crm/branching_measured.csv); sigma: Wang et al 2019")
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
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz", partial_waves=False)
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
        # <sigma v> from sigma with a dipole tail above the computed grid (6s/5d -> 6p are E1-allowed);
        # cs.rate would integrate Upsilon only over the grid (too low at high Te)
        k_line[key] = x["branching"] * rates.rates_on_grid(E[m], s[m], TEa, tail="dipole")
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
    md("""
## 5. Поправка на силу линии 6s → 6p (поляризация остова)

Модель `ba2` без поляризации остова 5p⁶: сила линии S(6s→6p) получается больше
экспериментальной (тест: в 1.3 раза; S из A NIST: S = A·g_u·λ³[Å]/2.026·10¹⁸). Возбуждение
6s → 6p — дипольно разрешённое, при E ≫ порога σ ∝ S, и завышение S переходит в σ почти
целиком. Ниже — отношение S_NIST/S_DBSR и вариант σ и k линий из 6s, умноженных на него
(файлы `*_scaledS`). Это приближение (у порога недипольные вклады и резонансы так не
масштабируются); какой вариант брать — решите по сравнению с экспериментом
**Pace, Hooper, Phys. Rev. A 7, 2033 (1973)**, doi:10.1103/PhysRevA.7.2033 — «Absolute Experimental Cross Sections for the Excitation of Barium Ions by Electron Impact» (абсолютные σ возбуждения Ba⁺)
или другим расчётом. Различие вариантов — оценка неопределённости плотности Ba⁺.
"""),
    code('''
ratio = {}
for r in nb.read_csv(W / "transitions_E1.csv"):
    if r["lower_nist"].startswith("6s") and r["upper_nist"].startswith("6p"):
        wl = nb.fnum(r["wl_air_nm"])
        A_n = [v for k, v in NIST_A.items() if wl and abs(k - wl) < 0.05]
        if not A_n:
            continue
        g_u = int(r["upper_nist"].split("J=")[1].split("/")[0]) + 1
        S_n = A_n[0] * g_u * (10 * wl) ** 3 / 2.026e18
        ratio[r["upper_nist"]] = S_n / float(r["S"])
        print(f"{r['upper_nist']:>14s} -> 6s  S_DBSR = {float(r['S']):.2f}  S_NIST = {S_n:.2f}"
              f"  S_NIST/S_DBSR = {ratio[r['upper_nist']]:.3f}")
sig_s, k_s = {}, {}
for key in sig:
    x = [y for y in nb.BA_LINES if abs(y["wl"] - key[2]) < 1e-3][0]
    f = ratio.get(x["upper"], 1.0) if key[0] == 1 else 1.0      # только из основного 6s
    sig_s[key] = (sig[key][0], f * sig[key][1])
    k_s[key] = f * k_line[key]
reference.write_xlsx(W / "ba2_line_sigma_scaledS.xlsx", lv, sig_s, sheet_of=lambda k: f"{k[2]:.3f}", labels=lab)
reference.write_rates_xlsx(W / "ba2_k_line_scaledS.xlsx", TEa, k_s, lab_l,
                           note="as ba2_k_line; from 6s scaled by S_NIST/S_DBSR of 6s-6p (no core polarisation in the model)")
print("->", W / "ba2_line_sigma_scaledS.xlsx", W / "ba2_k_line_scaledS.xlsx")
'''),
    md("""
## 6. Сравнение с экспериментом Pace & Hooper (1973)

Pace, Hooper, Phys. Rev. A 7, 2033 (1973) измерили **сечения испускания** линий 455.4 и
493.4 нм при ударе электронов по Ba⁺ 6s (`nb.PACE_HOOPER`, табл. X и XI; погрешность
21–43 %). В сечение испускания входит всё, что даёт фотон линии:
σ_исп = BR · [σ(6s→6p_j) + Σ_h σ(6s→h)·P(h→6p_j)], где P(h→u) — доля распадов уровня h
(7s, 6d, …), проходящих через u (ветвления DBSR E1 этого расчёта), BR — NIST.
Сравниваем то же самое: прямое, с каскадами, с каскадами и поправкой S (раздел 5). Для отношений
расчёт сглаживается гауссианой с шириной `E_SPREAD` (разброс энергии электронов в эксперименте),
иначе у порога точка расчёта попадает на отдельный резонанс. Погрешность эксперимента 21–43 % —
без учёта абсолютной калибровки оптики (их табл. II).
"""),
    code('''
br_e1, _, _ = nb.branching_from_csv(W / "transitions_E1.csv")
P = nb.cascade_prob(br_e1)
by_no = {s.nist_no: n for n, s in st.items()}
i0 = by_no[1]
E0 = cs.incident_energy(i0)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, x in zip(axs, [x for x in nb.BA_LINES if x["ion"] == "Ba II"]):
    u = name_of[x["upper"]]; un = st[u].nist_no
    direct = np.nan_to_num(cs.sigma(i0, u))
    casc = sum(np.nan_to_num(cs.sigma(i0, by_no[h])) * p.get(un, 0.0)
               for h, p in P.items() if h in by_no and p.get(un, 0.0) > 0)
    f = ratio.get(x["upper"], 1.0)
    m = E0 > 0
    E_SPREAD = 0.5     # эВ, FWHM разброса энергии электронов в эксперименте (уточните по статье); сглаживание DBSR
    def smooth(y):
        g = np.exp(-0.5 * ((E0[m][:, None] - E0[m][None, :]) / (E_SPREAD / 2.3548)) ** 2)
        return (g @ y[m]) / g.sum(axis=1)
    ax.plot(E0[m], x["branching"] * direct[m] / 1e-16, lw=0.7, label="DBSR, прямое")
    ax.plot(E0[m], x["branching"] * (direct + casc)[m] / 1e-16, lw=0.9, label="DBSR + каскады")
    ax.plot(E0[m], x["branching"] * (f * direct + casc)[m] / 1e-16, lw=0.9, ls="--", label="DBSR + каскады, S → NIST")
    ph = nb.PACE_HOOPER[x["wl"]]
    ax.errorbar(ph["E"], ph["sigma"], yerr=ph["sigma"] * ph["err"] / 100, fmt="ko", ms=4, label="Pace, Hooper 1973")
    ax.set_xscale("log"); ax.set_title(f"{x['wl']} нм"); ax.set_xlabel("E, эВ"); ax.set_ylabel("σ испускания, 10⁻¹⁶ см²")
    print(f"\\n{x['wl']} нм:   E, эВ   эксп.   DBSR+касц/эксп.   (с поправкой S)/эксп.")
    for e, sp in zip(ph["E"], ph["sigma"]):
        k = np.argmin(abs(E0 - e))
        if E0[k] <= 0 or abs(E0[k] - e) > 0.5:
            continue
        km = int(np.argmin(abs(E0[m] - e)))                      # точка сглаженной кривой
        a = x["branching"] * smooth(direct + casc)[km] / 1e-16
        b = x["branching"] * smooth(f * direct + casc)[km] / 1e-16
        print(f"            {e:6.1f}  {sp:6.1f}   {a / sp:8.2f}            {b / sp:8.2f}")
axs[0].legend(fontsize=7); fig.tight_layout(); plt.show()
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
JMAX = 0.5          # e + Ba: 57 электронов, поэтому J полуцелые: 0.5 (проверка), затем 14.5 / 19.5 / 24.5
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
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz", partial_waves=False)
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
    k_line[key] = x553["branching"] * rates.rates_on_grid(Ei[m], s[m], TEa, tail="dipole", ion=False)  # neutral; E1 tail
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

# ====================================================================== 05 Xe II: Wang model, own run
COMMON_05_06 = '''
TEa = np.array(TE)
def tail_of(i, j):
    """Tail of sigma above the computed grid: ln E/E for E1-allowed i -> j (as run_model.py), else 1/E."""
    try:
        a_, b_ = STN[i], STN[j]
    except (NameError, KeyError):
        return "1/E"
    e1 = a_.parity != b_.parity and abs(a_.two_j - b_.two_j) <= 2 and a_.two_j + b_.two_j > 0
    return "dipole" if e1 else "1/E"
def k_of(E, s, thr_ev, tail="1/E"):
    """Maxwellian <sigma v> (cm3/s) on the TE grid."""
    return rates.rates_on_grid(E, s, TEa, thr_ev, tail=tail)
def show_k(title, k):
    print(f"{title:28s} " + "  ".join(f"{v:9.2e}" for v in k))
print(" " * 28 + "  ".join(f"Te={t:<6g}" for t in TEa))
'''

notebook("05_xe2_wang_check.ipynb", [
    md("""
# 05. Xe II: свой расчёт рассеяния в модели Ванга — проверка, метастабили, каскады

Три вещи, которых нет в ноутбуке 02 (там σ берутся готовыми у Ванга):

1. **Проверка σ Ванга.** Та же мишень (`models.xe2_wang`, 68 состояний, нумерация уровней —
   таблица Ванга), свой расчёт DBSR. Сравниваются σ и ⟨σv⟩ для **всех** пар, которые есть у
   Ванга (gs→6s/5d/6p, 6s→6p, 5d→6p). Это воспроизведение расчёта, а не другая физическая
   модель: совпадение проверяет, что Ванг и мы посчитали одно и то же (сетка, число
   парциальных волн, пороги). Зависимость от модели мишени — ноутбук 06.
2. **Переходы между метастабилями** (и с ²P₁/₂): у Ванга их нет. Метастабиль здесь — уровень
   ниже 6p, у которого время жизни по E1 (из этого же расчёта) длиннее `TAU_META` или
   E1-распада нет вовсе. M1 и E2 не считаются (dbsr_dmat3 падает на M1), поэтому настоящие
   времена жизни короче — проверьте по литературе, если важно.
3. **Каскады в верхние уровни выбранных линий.** Для каждого уровня h выше u — вероятность
   P(h→u), что h, распадаясь, пройдёт через u (все цепочки, ветвления DBSR E1). Константа
   каскадного заселения из уровня i: k_casc(i→u) = Σ_h k(i→h)·P(h→u). Отношение
   k_casc/k(i→u) — атомные данные (как σ_app у Fursa); как каскады войдут в заселённость, решает CRM.
   В модели Ванга выше 6p есть только 5d и 7s (до 18.5 эВ); 6d, 7p, 4f — ноутбук 06.

Все σ здесь — DBSR (этот расчёт); ветвления — DBSR E1 (этот расчёт).

**Время**: J = 0 (2 волны) на 4 ядрах — см. `docs/crm/lines_final.md`; полный расчёт
(J ≤ 25) — сервер. Сходимость по J проверьте (`JMAX` 15/20/25): у оптически разрешённых
переходов (gs→6s, 5d) большие J вносят заметный вклад.

**Выход** (папка готового расчёта `runs/xe_plus_v2`, если он есть, иначе `runs/crm/xe2_wang_full`): `sigma_xe2_wang.xlsx` (σ из начальных уровней Ванга,
формат Ванга), `rates_xe2_wang.xlsx` (k всех возбуждений), `check_vs_wang.csv`,
`xe2_levels_tau.csv`, `xe2_sigma_metastable.xlsx`, `xe2_k_metastable.xlsx`,
`xe2_cascade_P.csv`, `xe2_k_cascade.xlsx` (+ .csv).
"""),
    code(PARAMS),
    code('''
import shutil
JMAX, EMAX = 25, 60.0        # J: проверьте сходимость; EMAX (эВ): σ выше — хвост 1/E (как в 02)
ref = reference.read_xlsx(XLSX)
WANG_INITIAL = sorted({i for i, j in ref.sigma})         # 1, 2, 4, 5, ... 34
W = RUNS / "xe2_wang_full"
if list((XE2_RUN / "scat").glob("h.[0-9][0-9][0-9]")):
    # готовый расчёт xe_plus_v2 (examples/xe_plus_wang2019.ipynb) — та же модель, сетка энергий
    # 0.0136 эВ до 60 эВ: рассеяние не пересчитывается, считаются только A, σ и k(Te)
    W = XE2_RUN
    om = sorted((W / "compare").glob(f"omega_J{JMAX:g}_E*.npz"),           # наибольшая Emax
                key=lambda f: -float(f.name.split("_E")[1].split("_dE")[0]))
    if om and not (W / f"omega_J{JMAX:g}.npz").exists():
        (W / f"omega_J{JMAX:g}.npz").symlink_to(om[0])
    print("готовый расчёт:", W, "| парциальных волн:", len(list((W / "scat").glob("h.[0-9][0-9][0-9]"))),
          "| Ω:", (W / f"omega_J{JMAX:g}.npz").resolve())
elif (RUNS / "xe2_wang" / "target").exists() and not (W / "target").exists():
    shutil.copytree(RUNS / "xe2_wang" / "target", W / "target")      # мишень из ноутбука 01
job = nb.Job(W, "xe2_wang", levels=XLSX, stage="all", jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=EMAX, de=0.0136, multipoles="E1",
             sigma_initial=",".join(map(str, WANG_INITIAL)), te=",".join(map(str, TE)))
job.start()
'''),
    code(JOB_STATUS),
    code('''
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz", partial_waves=False)
tg = db.Target.load(W / "target")
name = {s.nist_no: s.name for s in tg.states if s.nist_no and s.name in cs.names}   # номер Ванга -> состояние
Ecm = {lv.no: lv.energy_cm for lv in ref.levels}
STN = {s.nist_no: s for s in tg.states if s.nist_no and s.name in cs.names}             # номер -> State (J, чётность)
def dbsr_sigma(i, j):
    E, s = cs.incident_energy(name[i]), cs.sigma(name[i], name[j])
    m = np.isfinite(s) & (E > 0)
    return E[m], s[m]
def thr(i, j):
    return (Ecm[j] - Ecm[i]) / nb.CM_PER_EV
'''
         + COMMON_05_06),
    md("""
## 1. σ Ванга против своего расчёта

Для каждой пары: k_DBSR/k_Ванг при каждом Te (обе константы — из σ одной и той же функцией
`rates.rates_on_grid`, хвост 1/E). Сводка — медиана и доля пар в пределах ×1.5 и ×2.
"""),
    code('''
import csv
check = []
for (i, j), (Ew, sw) in sorted(ref.sigma.items()):
    if i in name and j in name:
        kw = k_of(Ew, sw, thr(i, j), tail_of(i, j))
        kd = k_of(*dbsr_sigma(i, j), thr(i, j), tail_of(i, j))
        check.append(dict(i=i, j=j, label=f"{ref.label(i)} -> {ref.label(j)}",
                          **{f"ratio_Te{t:g}": d / w for t, d, w in zip(TEa, kd, kw)}))
for t in TEa:
    q = np.array([c[f"ratio_Te{t:g}"] for c in check])
    q = q[np.isfinite(q) & (q > 0)]
    print(f"Te = {t:5g} эВ: {len(q)} пар, медиана DBSR/Ванг = {np.median(q):.2f}, "
          f"в пределах ×1.5: {np.mean((q > 1 / 1.5) & (q < 1.5)):.0%}, ×2: {np.mean((q > 0.5) & (q < 2)):.0%}")
with open(W / "check_vs_wang.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(check[0])); w.writeheader(); w.writerows(check)
print("->", W / "check_vs_wang.csv")
print("\\nверхние уровни выбранных линий:")
for c in check:
    if c["j"] in nb.XE2_UPPERS:
        print(f"{c['i']:3d} -> {c['j']:3d}  " + "  ".join(f"{c[f'ratio_Te{t:g}']:5.2f}" for t in TEa))
'''),
    code('''
fig, axs = plt.subplots(2, len(nb.XE2_LINES), figsize=(4 * len(nb.XE2_LINES), 6.5), squeeze=False)
for col, x in enumerate(nb.XE2_LINES):
    u = x["upper"]
    for row, i in enumerate((1, 2)):
        ax = axs[row][col]
        if (i, u) in ref.sigma:
            E, s = ref.sigma[(i, u)]; ax.plot(E, s / 1e-16, "k", lw=0.8, label="Ванг 2019")
        E, s = dbsr_sigma(i, u); ax.plot(E, s / 1e-16, "r", lw=0.6, label=f"DBSR, J ≤ {JMAX}")
        ax.set_xscale("log"); ax.set_title(f"{i} → {u} ({x['wl']} нм)", fontsize=9)
axs[0][0].legend(fontsize=7); fig.supxlabel("E, эВ"); fig.supylabel("σ, 10⁻¹⁶ см²"); fig.tight_layout(); plt.show()
'''),
    md("""
## 2. Времена жизни (E1) и метастабили

`TAU_META` — порог: уровни ниже 6p с τ(E1) длиннее него (или без E1-распада) считаются
метастабилями. Уровень 2 (²P₁/₂) распадается только M1 — он всегда в списке.
"""),
    code('''
TAU_META = 1e-6            # с
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")
first6p = min(n for n in name if ")6p" in ref.label(n))
low = [n for n in sorted(name) if Ecm[n] < Ecm[first6p]]
META = [n for n in low if n > 1 and (n not in tot or 1 / tot[n] > TAU_META)]
rows = []
for n in sorted(name):
    tau = 1 / tot[n] if n in tot else float("inf")
    rows.append(dict(no=n, label=ref.label(n), E_eV=Ecm[n] / nb.CM_PER_EV, tau_E1_s=tau, metastable=n in META))
    if n in low:
        print(f"{n:3d} {ref.label(n):34s} {Ecm[n] / nb.CM_PER_EV:7.3f} эВ   τ(E1) = {tau:9.3g} с" + ("   метастабиль" if n in META else ""))
with open(W / "xe2_levels_tau.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("метастабили:", META, " (начальные уровни у Ванга:", WANG_INITIAL, ")")
'''),
    md("""
## 3. σ и ⟨σv⟩ между основным, ²P₁/₂ и метастабилями

Набор M — 1, 2, метастабили и начальные уровни Ванга ниже 6p (у Ванга это и 6s/5d с
τ ≈ 0.1–0.2 мкс, которые в плазме тоже заметно заселены). Только возбуждение i → j (E_i < E_j). Обратный процесс — детальное равновесие:
`rates.deexcitation(k, g_i, g_j, ΔE, Te)`, g = 2J+1 (столбец J в «NIST Level Table»).
Плюс k(i → u) из этих уровней в верхние уровни выбранных линий (у Ванга есть не все).
"""),
    code('''
M = sorted({1, 2} | set(META) | {i for i in WANG_INITIAL if Ecm[i] < Ecm[first6p]})   # + начальные уровни Ванга ниже 6p
sig_m, lab_m, k_m = {}, {}, {}
for a_, i in enumerate(M):
    for j in M[a_ + 1:] + [u for u in nb.XE2_UPPERS]:
        if Ecm[j] <= Ecm[i] or (i, j) in sig_m:
            continue
        E, s = dbsr_sigma(i, j)
        sig_m[(i, j)] = (E, s)
        lab_m[(i, j)] = f"{i}->{j}" + ("" if (i, j) in ref.sigma or j not in nb.XE2_UPPERS else " (нет у Ванга)")
        k_m[(i, j)] = k_of(E, s, thr(i, j), tail_of(i, j))
reference.write_xlsx(W / "xe2_sigma_metastable.xlsx", ref.levels, sig_m, sheet_of=lambda k: f"from {k[0]}", labels=lab_m)
reference.write_rates_xlsx(W / "xe2_k_metastable.xlsx", TEa, k_m, lab_m,
                           note=f"<sigma v> (cm3/s), excitation i->j, levels: Wang 2019 numbers; sigma: DBSR, model xe2_wang, J<={JMAX}")
for key in sorted(k_m):
    show_k(lab_m[key], k_m[key])
'''),
    md("""
## 3б. Проверка через силы линий (gf): какому набору σ из метастабилей верить

Возбуждение m → u (m — уровень 6s/5d, u — 6p) и излучение u → m — один и тот же переход.
Для разрешённого перехода σ(m→u) примерно пропорционально его силе линии (gf ∝ A):
при больших энергиях строго, у порога — приближённо. Наш расчёт даёт gf всех таких
переходов, NIST — для части из них. Если наш A(u→m) больше NIST в X раз, то и наше
σ(m→u), вероятно, завышено примерно в X раз.

Столбцы:
* `наш/NIST` — наш A против NIST (нет в NIST — пусто);
* `наш/изм.` — по вашим спектрам: (ветвление DBSR) / (измеренное ветвление), медиана по всем
  спектрам `docs/crm/branching/` (только верхние уровни ваших линий и видимые ветви). Это
  отношение A, только если время жизни уровня в расчёте верно: BR_ист/BR_наш = (A_ист/A_наш)·(τ_ист/τ_наш).
  Если знаете измеренные времена жизни уровней 6p, впишите их в `TAU_MEAS_NS` — поправка учтётся.
  Где есть и спектр, и NIST, верим спектру (для 680.57 нм спектр противоречит NIST);
* `vel/len` — согласие двух форм нашего расчёта (далеко от 1 — линии не верим);
* `k_наш/k_Ванг` — при `TE_CHECK` из раздела 1;
* `исправл./Ванг` — k_наш × (A_истинное/A_наш) / k_Ванг: если близко к 1, расхождение с Вангом
  объясняется силой линии в нашей модели, и правее Ванг; если далеко от 1 — дело не в A.
"""),
    code('''
TE_CHECK = 1.0                        # эВ
lines_nist = nb.nist_lines(NIST / "nist_XeII.csv")
nos = sorted(Ecm, key=Ecm.get)        # номера уровней Ванга по возрастанию энергии
E_sorted = [Ecm[n] for n in nos]
A_nist = {}
for x in lines_nist:
    if not x["A"] or x["Ei"] is None or x["Ek"] is None:
        continue
    lo = nb.level_number(E_sorted, x["Ei"], tol=1.0)
    up = nb.level_number(E_sorted, x["Ek"], tol=1.0)
    if lo and up:
        A_nist[(nos[up - 1], nos[lo - 1])] = x["A"]
A_our, gauge = {}, {}
for r in nb.read_csv(W / "transitions_E1.csv"):
    if r["upper_no"] and r["lower_no"]:
        key = (int(r["upper_no"]), int(r["lower_no"]))
        A_our[key] = nb.fnum(r["A_exp"])
        gauge[key] = nb.fnum(r["gauge_ratio"])
# измерено по спектрам (examples/crm/branching_from_spectrum.py): доля ветви среди видимых × (1 − доля
# невидимых по DBSR, строка «ИТОГ») = измеренное ветвление BR_ист; BR_ист / BR_наш = (A_ист/A_наш)·(τ_ист/τ_наш),
# поэтому A_ист/A_наш = (BR_ист/BR_наш)·(τ_наш/τ_ист). Без измеренного τ (TAU_MEAS_NS) берётся τ_ист = τ_наш.
TAU_MEAS_NS = {}            # {номер верхнего уровня: измеренное время жизни, нс} — впишите из литературы
meas = {}
for fcsv in sorted((NIST / "branching").glob("branching_*.csv")):
    rs = nb.read_csv(fcsv)
    unseen = {r["upper"]: nb.fnum(r["unseen_dbsr"]) for r in rs if r["branch_nm"] == "ИТОГ"}
    for r in rs:
        share = nb.fnum(r["share_of_seen"])
        if not (share and r["lower"].strip().isdigit() and unseen.get(r["upper"]) is not None):
            continue
        u_, l_ = int(r["upper"]), int(r["lower"])
        bd = br.get(u_, {}).get(l_)                     # ветвление DBSR этого расчёта (раздел 2)
        if bd:
            tau_fix = (1e9 / tot[u_]) / TAU_MEAS_NS[u_] if u_ in TAU_MEAS_NS else 1.0
            meas.setdefault((u_, l_), []).append(share * (1 - unseen[r["upper"]]) / bd * tau_fix)
meas = {k: float(np.median(v)) for k, v in meas.items()}
ck = {(c["i"], c["j"]): c for c in check}
rows = []
print(f"{'m->u':>8s}  {'λ, нм':>8s}  {'A_наш':>9s}  {'A_NIST':>9s}  {'наш/NIST':>8s}  {'наш/изм.':>8s}  {'vel/len':>7s}"
      f"  {'k_наш/k_Ванг':>12s}  {'исправл./Ванг':>13s}")
for u in nb.XE2_UPPERS:
    for m in sorted(M):
        if (u, m) not in A_our or Ecm[m] >= Ecm[u]:
            continue
        a, an = A_our[(u, m)], A_nist.get((u, m))
        mr = meas.get((u, m))                          # A_истинное / A_наш по спектру
        kr = ck.get((m, u), {}).get(f"ratio_Te{TE_CHECK:g}")
        true_over_our = mr if mr else (an / a if (an and a) else None)   # спектр важнее NIST
        fix = kr * true_over_our if (kr and true_over_our) else None
        wl = 1e7 / (Ecm[u] - Ecm[m]) / 1.000277
        rows.append(dict(m=m, u=u, wl_nm=wl, A_our=a, A_NIST=an, our_NIST=a / an if an else None,
                         our_measured=1 / mr if mr else None,
                         vel_len=gauge.get((u, m)), k_our_Wang=kr, corrected_Wang=fix))
        f = lambda v, fmt: format(v, fmt) if v is not None else "—"
        print(f"{m:3d}->{u:<3d}  {wl:8.3f}  {a:9.2e}  {f(an, '9.2e'):>9s}  {f(a / an if an else None, '8.2f'):>8s}"
              f"  {f(1 / mr if mr else None, '8.2f'):>8s}"
              f"  {f(gauge.get((u, m)), '7.2f'):>7s}  {f(kr, '12.2f'):>12s}  {f(fix, '13.2f'):>13s}")
with open(W / "xe2_gf_check.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("->", W / "xe2_gf_check.csv")
'''),
    md("""
## 4. Каскады в верхние уровни выбранных линий

P(h→u) — доля распадов h, проходящих через u (ветвления DBSR E1, все цепочки). Печатаются
уровни с P > `P_MIN`. k_casc(i→u) = Σ_h k(i→h)·P(h→u); отношение к прямому k(i→u).
"""),
    code('''
P_MIN = 1e-3
P = nb.cascade_prob(br)
prow, k_c, lab_c = [], {}, {}
kcache = {}
def k_pair(i, j):
    if (i, j) not in kcache:
        kcache[(i, j)] = k_of(*dbsr_sigma(i, j), thr(i, j), tail_of(i, j))
    return kcache[(i, j)]
for x in nb.XE2_LINES:
    u = x["upper"]
    feed = sorted(((h, P[h][u]) for h in P if P[h].get(u, 0) > P_MIN and h in name), key=lambda t: -t[1])
    print(f"\\n{x['wl']} нм, верхний уровень {u} {ref.label(u)}: питающих уровней {len(feed)}")
    for h, p in feed:
        prow.append(dict(upper=u, line_nm=x["wl"], h=h, h_label=ref.label(h), P=p))
        print(f"   {h:3d} {ref.label(h):34s} P = {p:.3f}")
    for i in M:
        if Ecm[i] >= Ecm[u]:
            continue
        kc = sum(k_pair(i, h) * p for h, p in feed if Ecm[h] > Ecm[i])
        kd = k_pair(i, u)
        k_c[(i, u, "casc")] = kc; lab_c[(i, u, "casc")] = f"{i}->{u} cascade"
        k_c[(i, u, "ratio")] = kc / kd; lab_c[(i, u, "ratio")] = f"{i}->{u} cascade/direct"
        if i in (1, 2):
            show_k(f"  из {i}: k_casc/k_прям", kc / kd)
with open(W / "xe2_cascade_P.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
reference.write_rates_xlsx(W / "xe2_k_cascade.xlsx", TEa, k_c, lab_c,
                           note="cascade: sum_h <sigma v>(i->h) P(h->u), cm3/s; ratio = cascade/direct; DBSR, model xe2_wang")
print("->", W / "xe2_cascade_P.csv", W / "xe2_k_cascade.xlsx")
'''),
    md("""
## 5. Набор k с поправкой по спектру (третий набор для CRM)

Для переходов m → u, у которых настоящая сила линии известна (ваш спектр — в приоритете,
иначе NIST, раздел 3б), наше k(m→u) умножается на A_истинное / A_наш. Это поправка
дипольной части сечения: точна при больших энергиях, у порога — приближённо (там вклад
обмена и резонансов не масштабируется). Для остальных пар k не меняются — берите их из
`rates_xe2_wang.xlsx`.

В CRM: три набора k(метастабиль → 6p) — Ванг, наш, наш с поправкой. Разброс ne по ним —
неопределённость от атомных данных; поправленный набор лучше всего согласован с вашим спектром.
"""),
    code('''
k_cor, lab_cor, cor_rows = {}, {}, []
for (u, m), a in sorted(A_our.items()):
    if m not in M or u not in nb.XE2_UPPERS or Ecm[m] >= Ecm[u] or not a:
        continue
    f_true = meas.get((u, m)) or (A_nist[(u, m)] / a if (u, m) in A_nist else None)
    if not f_true:
        continue
    src = "спектр" if (u, m) in meas else "NIST"
    k_cor[(m, u)] = f_true * k_pair(m, u)
    lab_cor[(m, u)] = f"{m}->{u}"
    cor_rows.append(dict(m=m, u=u, factor=f_true, source=src))
    show_k(f"{m}->{u} ×{f_true:.2f} ({src})", k_cor[(m, u)])
reference.write_rates_xlsx(W / "xe2_k_corrected.xlsx", TEa, k_cor, lab_cor,
                           note="<sigma v> (cm3/s), m->u: DBSR (xe_plus_v2) x A_true/A_DBSR (A_true: branching measured "
                                "from the spectra, else NIST); other pairs: rates_xe2_wang.xlsx")
with open(W / "xe2_k_corrected_factors.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(cor_rows[0])); wr.writeheader(); wr.writerows(cor_rows)
print("->", W / "xe2_k_corrected.xlsx", W / "xe2_k_corrected_factors.csv")
'''),
    md("""
## 6. Поиск ne-пар, устойчивых к выбору атомных данных

Скорость заселения верхнего уровня u: K(u) = k(1→u) + w·Σ_m k(m→u), где w — доля ионов на каждом
метастабиле m относительно основного (m из `LEVER_SET`: метастабили и долгоживущие уровни, у
которых σ есть и у Ванга, и у нас). Отношение двух линий R = (BR_a K_a)/(BR_b K_b); ветвления от
ne не зависят и в производные не входят.

* **чувствительность к ne** — d ln R / d ln w: насколько меняется отношение, если доля метастабилей
  (а она растёт с ne) меняется. Важно: при Te ≈ 1 эВ из метастабилей возбуждение в 10⁵–10⁷ раз
  эффективнее, чем из основного, поэтому уже при w ≳ 10⁻⁵ оба уровня заселяются почти только из
  метастабилей, и R перестаёт зависеть от w — пара теряет чувствительность к ne. Тогда ne влияет на
  R только через **соотношение заселённостей разных метастабилей** — это решает ваша CRM, здесь
  все метастабили взяты с одной долей w. Поэтому смотрите столбцы при разных w (`W_SCAN`) и берите
  w из вашей CRM;
* **чувствительность к Te** — d ln R / d ln Te при той же w;
* **устойчивость к атомным данным** — ошибка ln ne, которую даёт замена нашего набора на
  Ванговский: |ln R_наш − ln R_Ванг| / (d ln R/d ln w);
* `ошибка ne от 10% Te` — |d ln R/d ln Te| · 0.1 / (d ln R/d ln w).

Хорошая пара: заметная чувствительность к ne в обоих наборах, малые обе «ошибки ne».
"""),
    code('''
TE_PAIR = 1.0                  # эВ
SN_MIN = 50
W_REF = 1e-3                   # доля ионов на каждом метастабиле относительно основного (возьмите из CRM)
W_SCAN = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
LONG_TAU = 1e-7                # с: долгоживущие уровни тоже в наборе (уровень 4: τ ≈ 220 нс)
LEVER_SET = sorted(m for m in M if m in WANG_INITIAL and m > 1 and Ecm[m] < Ecm[first6p]
                   and (m in META or (m in tot and 1 / tot[m] > LONG_TAU)))
print("метастабили в расчёте:", LEVER_SET)
TEg = np.array([TE_PAIR, 1.25 * TE_PAIR])

def k_both(i, u):
    """(k_наш, k_Ванг) на TEg или None, если у Ванга нет σ."""
    if (i, u) not in ref.sigma:
        return None
    Ew, sw = ref.sigma[(i, u)]
    kd = rates.rates_on_grid(*dbsr_sigma(i, u), TEg, thr(i, u), tail=tail_of(i, u))
    kw = rates.rates_on_grid(Ew, sw, TEg, thr(i, u), tail=tail_of(i, u))
    return kd, kw

crit = [r for r in nb.read_csv(NIST / "criteria_A3.csv")
        if r["ion"] == "XeII" and r["upper_no"] and int(r["upper_no"]) in name and ")6p" in ref.label(int(r["upper_no"]))
        and not r["nist_within_1.5fwhm"]]
def best_sn(r):
    v = [nb.fnum(r[k]) for k in r if k.endswith("_S/N") and r[k]]
    return max(v) if v else 0.0
def n_unsat(r):
    return sum(1 for k in r if k.endswith("_sat") and r[k.replace("_sat", "_S/N")] and r[k] != "SAT")
crit = [r for r in crit if best_sn(r) >= SN_MIN and n_unsat(r) > 0]
KD, KW = {}, {}                  # u -> (k1 на TEg, Σ_m k(m->u) на TEg)
KDm, KWm = {}, {}                # u -> {m: k(m->u) на TEg}
for u in sorted({int(r["upper_no"]) for r in crit}):
    k1 = k_both(1, u)
    terms = {m: k_both(m, u) for m in LEVER_SET if Ecm[m] < Ecm[u]}
    if k1 is None or any(t is None for t in terms.values()):
        continue
    KD[u] = (k1[0], sum(t[0] for t in terms.values()))
    KW[u] = (k1[1], sum(t[1] for t in terms.values()))
    KDm[u] = {m: t[0] for m, t in terms.items()}
    KWm[u] = {m: t[1] for m, t in terms.items()}
lines_c = [r for r in crit if int(r["upper_no"]) in KD]
lever = {u: dict(LD=KD[u][1][0] / KD[u][0][0], LW=KW[u][1][0] / KW[u][0][0],
                 s=float(np.log(KD[u][0][1] / KD[u][0][0]) / np.log(1.25))) for u in KD}

def lnR(K, a, b, w, it=0):
    return np.log((K[a][0][it] + w * K[a][1][it]) / (K[b][0][it] + w * K[b][1][it]))
def sens(K, a, b, w):
    """d ln R / d ln w и d ln R / d ln Te при доле w."""
    sw = (lnR(K, a, b, 2 * w) - lnR(K, a, b, w / 2)) / np.log(4)
    st = (lnR(K, a, b, w, 1) - lnR(K, a, b, w, 0)) / np.log(1.25)
    return sw, st

print(f"линий-кандидатов: {len(lines_c)}, верхних уровней: {len(KD)}")
best = {}
for ia, a in enumerate(lines_c):
    for b in lines_c[ia + 1:]:
        ua, ub = int(a["upper_no"]), int(b["upper_no"])
        if ua == ub:
            continue
        key = frozenset((ua, ub))
        dwl = abs(float(a["wl_nm"]) - float(b["wl_nm"]))
        if key not in best or dwl < best[key]["dwl"]:          # одна пара линий на пару уровней: ближайшая по λ
            best[key] = dict(line_a=float(a["wl_nm"]), up_a=ua, line_b=float(b["wl_nm"]), up_b=ub, dwl=dwl)
pairs = []
for p in best.values():
    a, b = p["up_a"], p["up_b"]
    swD, stD = sens(KD, a, b, W_REF)
    swW, stW = sens(KW, a, b, W_REF)
    s_ne = min(abs(swD), abs(swW)) if swD * swW > 0 else 0.0
    data_err = abs(lnR(KD, a, b, W_REF) - lnR(KW, a, b, W_REF)) / s_ne if s_ne > 0 else np.inf
    te_err = 0.1 * max(abs(stD), abs(stW)) / s_ne if s_ne > 0 else np.inf
    p.update(sens_ne=s_ne, sens_ne_our=swD, sens_ne_wang=swW, sens_Te_our=stD, sens_Te_wang=stW,
             ne_err_data=data_err, ne_err_10pct_Te=te_err,
             **{f"sens_ne_w{w:g}": sens(KD, a, b, w)[0] for w in W_SCAN})
    pairs.append(p)
cur = [p for p in pairs if {p["up_a"], p["up_b"]} == {41, 44}]
# чувствительность к заселённости отдельного метастабиля m (остальные — с долей W_REF):
# d ln R / d ln w_m. Если доля всех метастабилей ~ ne, R от общего уровня не зависит (см. выше),
# и ne «видно» только через изменение соотношения метастабилей — его даёт ваша CRM.
def lnR_m(Km, K1, a, b, wm, it=0):
    Ka = K1[a][0][it] + sum(wm.get(m, W_REF) * Km[a][m][it] for m in Km[a])
    Kb = K1[b][0][it] + sum(wm.get(m, W_REF) * Km[b][m][it] for m in Km[b])
    return np.log(Ka / Kb)
for p in pairs:
    a, b = p["up_a"], p["up_b"]
    sm = {}
    for m in LEVER_SET:
        if m in KDm[a] or m in KDm[b]:
            sd = (lnR_m(KDm, KD, a, b, {m: 2 * W_REF}) - lnR_m(KDm, KD, a, b, {m: W_REF / 2})) / np.log(4)
            sw_ = (lnR_m(KWm, KW, a, b, {m: 2 * W_REF}) - lnR_m(KWm, KW, a, b, {m: W_REF / 2})) / np.log(4)
            sm[m] = (sd, sw_)
    mbest = max(sm, key=lambda m: min(abs(sm[m][0]), abs(sm[m][1])) if sm[m][0] * sm[m][1] > 0 else 0.0)
    p.update(meta_best=mbest, sens_meta_our=sm[mbest][0], sens_meta_wang=sm[mbest][1],
             **{f"sens_m{m}_our": v[0] for m, v in sm.items()}, **{f"sens_m{m}_wang": v[1] for m, v in sm.items()})
print("\\nчувствительность к отдельным метастабилям (лучший m для пары; d lnR / d ln w_m, наш / Ванг):")
pm = sorted((p for p in pairs if p["sens_meta_our"] * p["sens_meta_wang"] > 0),
            key=lambda p: -min(abs(p["sens_meta_our"]), abs(p["sens_meta_wang"])))
for p in pm[:15]:
    print(f"  {p['line_a']:9.3f} ({p['up_a']}) / {p['line_b']:9.3f} ({p['up_b']}):  метастабиль {p['meta_best']:3d}"
          f" {ref.label(p['meta_best']):28s} {p['sens_meta_our']:6.2f} / {p['sens_meta_wang']:6.2f}   Δλ = {p['dwl']:.1f}")
if cur:
    p = cur[0]
    print("  текущая 41/44: " + ", ".join(f"m={m}: {p.get(f'sens_m{m}_our', float('nan')):.2f}/{p.get(f'sens_m{m}_wang', float('nan')):.2f}"
                                       for m in LEVER_SET))
good = sorted((p for p in pairs if p["sens_ne"] > 0.05), key=lambda p: p["ne_err_data"] + p["ne_err_10pct_Te"])
print(f"\\nпар уровней: {len(pairs)}; с чувствительностью к ne > 0.05 в обоих наборах при w = {W_REF:g}: {len(good)}")
print(f"{'линия a':>9s} {'ур.':>4s} {'линия b':>9s} {'ур.':>4s} {'d lnR/d ln w':>12s} {'d lnR/d lnTe':>12s}"
      f" {'ош.ne данные':>12s} {'ош.ne 10%Te':>11s} {'Δλ':>6s}")
for p in good[:20]:
    print(f"{p['line_a']:9.3f} {p['up_a']:4d} {p['line_b']:9.3f} {p['up_b']:4d} {p['sens_ne']:12.3f}"
          f" {p['sens_Te_our']:12.2f} {np.exp(p['ne_err_data']):11.2f}× {np.exp(p['ne_err_10pct_Te']):10.2f}× {p['dwl']:6.1f}")
if cur:
    p = cur[0]
    print("\\nтекущая пара 41/44 (561.667/545.045): d lnR/d ln w (наш набор) при w = "
          + ", ".join(f"{w:g}: {p[f'sens_ne_w{w:g}']:.3f}" for w in W_SCAN))
    print("  при малой чувствительности на реальных w пара работает через соотношение метастабилей — это решает CRM")
with open(W / "xe2_ne_pairs.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(pairs[0])); wr.writeheader()
    wr.writerows(sorted(pairs, key=lambda p: -p["sens_ne"]))
print("->", W / "xe2_ne_pairs.csv")
'''),
    md("""
## 7. Таблица линий для общей подгонки

Все линии-кандидаты раздела 6 (видимые, без бленд, не насыщены хотя бы в одном листе A3) с
данными, нужными CRM: ветвление (DBSR; измеренное, где есть), согласие двух форм A (vel/len),
k(1→u) и рычаг метастабилей в обоих наборах, наклон. Подгонка по всем линиям сразу:
неизвестные ne, Te (и плотность ионов как общий множитель); минимизируется сумма
[ln(I_изм/I_CRM)]² по линиям с весами 1/σ². Линии, которые систематически не ложатся, —
признак плохих атомных данных для их верхнего уровня (смотрите `vel/len` и `BR_источник`).
"""),
    code('''
fit_rows = []
for r in sorted(lines_c, key=lambda r: float(r["wl_nm"])):
    u, lo = int(r["upper_no"]), int(r["lower_no"]) if r["lower_no"] else None
    brd = br.get(u, {}).get(lo) if lo else None                    # ветвление DBSR (раздел 2)
    ratio = meas.get((u, lo)) if lo else None                      # A_истинное / A_наш по спектру
    g = gauge.get((u, lo)) if lo else None
    fit_rows.append(dict(wl_nm=float(r["wl_nm"]), upper=u, upper_label=ref.label(u), lower=lo,
                         lower_label=ref.label(lo) if lo else "", BR_dbsr=brd,
                         BR_best=brd * ratio if (brd and ratio) else brd,
                         BR_source="спектр" if ratio else "DBSR", vel_len=g,
                         A_reliable=(g is not None and 0.7 < g < 1.4),
                         lever_our=lever[u]["LD"], lever_wang=lever[u]["LW"], Te_slope=lever[u]["s"],
                         best_SN=best_sn(r), n_unsaturated_sheets=n_unsat(r)))
with open(W / "xe2_lines_for_fit.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(fit_rows[0])); wr.writeheader(); wr.writerows(fit_rows)
print(f"{len(fit_rows)} линий ->", W / "xe2_lines_for_fit.csv")
print("из них с надёжной A (0.7 < vel/len < 1.4):", sum(r["A_reliable"] for r in fit_rows))
'''),
])

# ====================================================================== 06 Xe II: extended model
notebook("06_xe2_extended.ipynb", [
    md("""
# 06. Xe II: расширенная модель (+ 6d, 7p, 4f) — каскады и устойчивость σ

Модель `xe2_ext` (`models.py`): мишень Ванга + 5p⁴6d, 7p, 4f, корреляционная 7d, все уровни
до 19.2 эВ (до (¹D₂)6d). Уровни — NIST ASD (таблица Ванга кончается на 18.6 эВ); номера
переводятся в номера Ванга по энергии (`nb.level_number`, допуск 1 см⁻¹), у новых уровней
номера Ванга нет.

1. **Энергии и A** новых уровней против NIST.
2. **Устойчивость σ**: σ(1→u), σ(2→u) для выбранных u — расширенная модель против модели
   Ванга (ноутбук 05) и Ванга 2019. Разница — оценка неопределённости σ от выбора мишени.
3. **Каскады** с учётом 6d (и цепочек 7p → 7s/6d → 6p, 4f → 5d → 6p): P(h→u), k_casc, отношение
   к прямому k — как в 05; отдельно вклад уровней, которых нет в модели Ванга.

Все σ — DBSR (этот расчёт); ветвления — DBSR E1 (этот расчёт).

**Время**: состояний больше, чем в 05, — расчёт дольше (сервер). Сначала можно запустить
`stage="transitions"` (минуты): пункты 1 и P(h→u) из пункта 3 считаются без рассеяния.

**Выход** (папка `xe2_ext`): `sigma_xe2_ext.xlsx`, `rates_xe2_ext.xlsx` (номера NIST ASD),
`ext_vs_wang.csv`, `xe2_ext_cascade_P.csv`, `xe2_ext_k_cascade.xlsx` (+ .csv).
"""),
    code(PARAMS),
    code('''
JMAX, EMAX = 25, 60.0
STAGE = "all"                 # "transitions" — только структура (быстро)
W = RUNS / "xe2_ext"
job = nb.Job(W, "xe2_ext", stage=STAGE, jmax=JMAX, cores=CORES, mem=MEM, hd_threads=HD_THREADS,
             scratch=SCRATCH, scratch_gb=SCRATCH_GB, emax=EMAX, de=0.0136, multipoles="E1",
             sigma_initial="1,2", te=",".join(map(str, TE)))
job.start()
'''),
    code(JOB_STATUS),
    md("## 1. Энергии и номера Ванга"),
    code('''
print((W / "target_table.txt").read_text())
ref = reference.read_xlsx(XLSX)
elv = [lv.energy_cm for lv in ref.levels]
tg = db.Target.load(W / "target")
S = {s.nist_no: s for s in tg.states if s.nist_no}           # номер NIST ASD -> состояние
wang = {n: nb.level_number(elv, s.exp_energy_cm, tol=1.0) for n, s in S.items()}
Ecm = {n: s.exp_energy_cm for n, s in S.items()}
lab = {n: s.nist_label or s.config for n, s in S.items()}
of_wang = {w: n for n, w in wang.items() if w}
new = [n for n in sorted(S) if not wang[n]]
print(f"{len(S)} состояний; без номера Ванга (новые): {len(new)}")
for n in new:
    print(f"   ASD {n:3d}  {lab[n]:34s} {Ecm[n] / nb.CM_PER_EV:7.3f} эВ")
'''),
    md("## 2. A новых уровней против NIST"),
    code('''
nistA = [x for f in ("nist_XeII.csv", "nist_XeII_UV.csv", "nist_XeII_IR.csv") if (NIST / f).exists()
         for x in nb.nist_lines(NIST / f) if x["A"]]
q = []
for r in nb.read_csv(W / "transitions_E1.csv"):
    if not r["upper_no"] or not r["lower_no"]:
        continue
    u, l = int(r["upper_no"]), int(r["lower_no"])
    hit = [x for x in nistA if x["Ek"] and x["Ei"] and abs(x["Ek"] - Ecm[u]) < 1 and abs(x["Ei"] - Ecm[l]) < 1]
    if hit:
        a = float(r["A_exp"]); q.append(a / hit[0]["A"])
        if u in new:
            print(f"{hit[0]['wl']:9.3f} {lab[u]:30s} -> {lab[l]:30s} DBSR {a:.2e}  NIST {hit[0]['A']:.2e}  {a / hit[0]['A']:.2f}")
q = np.array(q)
print(f"все линии с A в NIST: {len(q)}, медиана DBSR/NIST = {np.median(q):.2f}, в пределах ×2: {np.mean((q > 0.5) & (q < 2)):.0%}")
'''),
    md("""
## 3. σ(1→u), σ(2→u): расширенная модель, модель Ванга (05), Ванг 2019

k_ext/k_Ванг и k_ext/k_05 по Te. Если 05 не досчитан, его столбцы пропускаются.
"""),
    code('''
import csv
cs = db.CollisionStrengths.load(W / f"omega_J{JMAX:g}.npz", partial_waves=False)
def ext_sigma(i, j):
    E, s = cs.incident_energy(S[i].name), cs.sigma(S[i].name, S[j].name)
    m = np.isfinite(s) & (E > 0)
    return E[m], s[m]
def thr(i, j):
    return (Ecm[j] - Ecm[i]) / nb.CM_PER_EV
W05 = XE2_RUN if (XE2_RUN / f"omega_J{JMAX:g}.npz").exists() else RUNS / "xe2_wang_full"   # результат ноутбука 05
f05 = W05 / f"omega_J{JMAX:g}.npz"
cs05 = db.CollisionStrengths.load(f05, partial_waves=False) if f05.exists() else None
if cs05:
    tg05 = db.Target.load(W05 / "target")
    n05 = {s.nist_no: s.name for s in tg05.states if s.nist_no and s.name in cs05.names}
'''
         + COMMON_05_06 + '''
rows = []
fig, axs = plt.subplots(2, len(nb.XE2_LINES), figsize=(4 * len(nb.XE2_LINES), 6.5), squeeze=False)
for col, x in enumerate(nb.XE2_LINES):
    u = of_wang[x["upper"]]
    for row, i in enumerate((1, 2)):
        ax = axs[row][col]
        E, s = ext_sigma(i, u); ke = k_of(E, s, thr(i, u)); ax.plot(E, s / 1e-16, "r", lw=0.6, label="xe2_ext")
        r = dict(i=i, upper_wang=x["upper"], line_nm=x["wl"])
        if (i, x["upper"]) in ref.sigma:
            Ew, sw = ref.sigma[(i, x["upper"])]; ax.plot(Ew, sw / 1e-16, "k", lw=0.8, label="Ванг 2019")
            r.update({f"ext/Wang_Te{t:g}": v for t, v in zip(TEa, ke / k_of(Ew, sw, thr(i, u)))})
        if cs05:
            E5, s5 = cs05.incident_energy(n05[i]), cs05.sigma(n05[i], n05[x["upper"]])
            m = np.isfinite(s5) & (E5 > 0); ax.plot(E5[m], s5[m] / 1e-16, "b", lw=0.6, label="05 (модель Ванга)")
            r.update({f"ext/05_Te{t:g}": v for t, v in zip(TEa, ke / k_of(E5[m], s5[m], thr(i, u)))})
        rows.append(r)
        ax.set_xscale("log"); ax.set_title(f"{i} → {x['upper']} ({x['wl']} нм)", fontsize=9)
axs[0][0].legend(fontsize=7); fig.supxlabel("E, эВ"); fig.supylabel("σ, 10⁻¹⁶ см²"); fig.tight_layout(); plt.show()
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("i", "upper_wang", "line_nm"), k))
with open(W / "ext_vs_wang.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["i"], "->", r["upper_wang"], "  ".join(f"{k}={v:.2f}" for k, v in r.items() if "Te" in k))
'''),
    md("""
## 4. Каскады с 6d, 7p, 4f

Как в 05, но по всем уровням расширенной модели. Колонка «новый» — уровня нет в модели
Ванга; `share_new` — доля каскада от таких уровней.
"""),
    code('''
P_MIN = 1e-3
br, tot, A = nb.branching_from_csv(W / "transitions_E1.csv")
P = nb.cascade_prob(br)
kcache = {}
def k_pair(i, j):
    if (i, j) not in kcache:
        kcache[(i, j)] = k_of(*ext_sigma(i, j), thr(i, j))
    return kcache[(i, j)]
prow, k_c, lab_c = [], {}, {}
for x in nb.XE2_LINES:
    u = of_wang[x["upper"]]
    feed = sorted(((h, P[h][u]) for h in P if P[h].get(u, 0) > P_MIN and h in S), key=lambda t: -t[1])
    print(f"\\n{x['wl']} нм, уровень Ванга {x['upper']} (ASD {u}): питают {len(feed)} уровней")
    for h, p in feed:
        prow.append(dict(upper_wang=x["upper"], upper_asd=u, line_nm=x["wl"], h_asd=h, h_wang=wang[h] or "",
                         h_label=lab[h], new=h in new, P=p))
        print(f"   ASD {h:3d} {lab[h]:34s} P = {p:.3f}" + ("   новый" if h in new else ""))
    for i in (1, 2):
        kc = sum(k_pair(i, h) * p for h, p in feed)
        kn = sum(k_pair(i, h) * p for h, p in feed if h in new)
        kd = k_pair(i, u)
        for tag, v in (("casc", kc), ("ratio", kc / kd), ("share_new", kn / np.maximum(kc, 1e-300))):
            k_c[(i, x["upper"], tag)] = v
            lab_c[(i, x["upper"], tag)] = f"{i}->{x['upper']} " + {"casc": "cascade", "ratio": "cascade/direct",
                                                                   "share_new": "share of new levels"}[tag]
        show_k(f"  из {i}: k_casc/k_прям", kc / kd)
        show_k(f"  из {i}: доля новых", kn / np.maximum(kc, 1e-300))
with open(W / "xe2_ext_cascade_P.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
reference.write_rates_xlsx(W / "xe2_ext_k_cascade.xlsx", TEa, k_c, lab_c,
                           note="cascade into Wang levels u: sum_h <sigma v>(i->h) P(h->u), cm3/s; DBSR, model xe2_ext")
print("->", W / "xe2_ext_cascade_P.csv", W / "xe2_ext_k_cascade.xlsx")
'''),
])
