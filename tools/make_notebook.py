"""Generate examples/xe_plus_wang2019.ipynb (run: python tools/make_notebook.py)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# e + Xe⁺: DBSR, 67 состояний, сравнение с Wang et al (2019)

Ядро Jupyter: **Python (pydbsr)**.

Расчёт идёт **отдельным фоновым процессом**: его не остановит закрытие браузера,
перезапуск ядра или выход с сервера. Ноутбук его запускает (шаг 2), показывает
ход (шаг 3) и строит графики по готовым результатам (шаг 5).

* временные файлы (матрицы) — в быстрой домашней папке `SCRATCH`;
* всё остальное (мишень, `h.nnn`, результаты, графики) — в `WORK` на большом диске;
* прерванный расчёт продолжается повторным запуском шага 2 (готовые волны пропускаются);
* чтобы добавить J: увеличьте `JMAX` и снова запустите шаг 2.
""")

code("""
# ---------- 1. параметры (поменяйте пути под себя)
from pathlib import Path
ROOT    = Path("/data/pydbsr_calc")                  # папка установки на большом диске (setup_server.sh)
WORK    = ROOT / "runs" / "xe_plus_v2"               # результаты -> большой диск; НОВАЯ папка для новой мишени
SCRATCH = Path.home() / "pydbsr_tmp"                 # временные файлы -> быстрая домашняя папка
SCRATCH_GB = 5                                       # сколько места в домашней папке можно занять
XLSX    = ROOT / "CrossSectionsIon_3.xlsx"           # таблица Wang et al (2019)
SCRIPT  = ROOT / "DBSR_src" / "examples" / "xe_plus_vs_wang2019.py"

JMAX  = 0          # сначала 0 (проверка, ~1 ч), потом 10, 25, 50
CORES = 64         # ядер можно занять
MEM   = 200        # ГБ оперативной памяти можно занять
HD_THREADS = 16    # потоков на одну диагонализацию
EMAX, DE = 60.0, 0.0136    # сетка энергий электрона, эВ (в статье шаг 0.001 Ry)

import pydbsr as db, shutil
print("pydbsr", db.__version__, "| программы DBSR:", db.bin_dir())
for p in (ROOT, SCRATCH.parent):
    u = shutil.disk_usage(p); print(f"{str(p):40s} свободно {u.free/1e9:8.1f} ГБ")
assert XLSX.exists(), f"нет файла {XLSX}"
assert SCRIPT.exists(), f"нет {SCRIPT}: запустите setup_server.sh ещё раз"
""")

code("""
# ---------- 2. запуск расчёта в фоне
import subprocess, sys, os
WORK.parent.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
LOG = Path(f"{WORK}.J{JMAX:g}.log")
PIDFILE = Path(f"{WORK}.pid")

def running():
    # pid of the running calculation or None (a finished child is reaped)
    try:
        pid = int(PIDFILE.read_text())
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except Exception:
        return None
    if state == "Z":                      # finished, not yet reaped by this kernel
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        return None
    return pid

if running():
    print("расчёт уже идёт, pid", running())
else:
    cmd = [sys.executable, "-u", str(SCRIPT), "--xlsx", str(XLSX), "--workdir", str(WORK),
           "--scratch", str(SCRATCH), "--scratch-gb", str(SCRATCH_GB),
           "--jmax", str(JMAX), "--cores", str(CORES), "--mem", str(MEM),
           "--hd-threads", str(HD_THREADS), "--emax", str(EMAX), "--de", str(DE)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)      # не зависит от ядра Jupyter
    PIDFILE.write_text(str(proc.pid))
    print("запущено, pid", proc.pid, "\\nлог:", LOG)
""")

code("""
# ---------- 3. ход расчёта (запускайте когда хотите)
import re
txt = LOG.read_text(errors="replace") if LOG.exists() else ""
done = [l for l in txt.splitlines() if "partial wave" in l]
print("идёт" if running() else "НЕ идёт (закончен или упал)")
print(f"парциальных волн готово: {sum('done in' in l for l in done)}, упало: {sum('FAILED' in l for l in done)}")
print("\\n".join(txt.splitlines()[-15:]))
""")

code("""
# ---------- (если нужно остановить расчёт)
import signal
pid = running()
if pid:
    os.killpg(pid, signal.SIGTERM); print("остановлен", pid)
""")

md("""
## 4. Проверка (после окончания J=0)
Стресс-тест на ваших данных: уравнение Дирака, сохранение тока, унитарность S,
пороги, согласованность мишени… Все строки должны быть `[PASS]`.
""")

code("""
r = subprocess.run([sys.executable, str(ROOT / "DBSR_src" / "tools" / "stress_test.py"),
                    "--target", str(WORK / "target"), "--scat", str(WORK / "scat"), "--skip-matrices",
                    "--log", str(WORK / "stress_log.json")], capture_output=True, text=True)
print(r.stdout[-6000:] or r.stderr[-3000:])
""")

md("## 5. Графики и скорости (по готовым результатам)")

code("""
import numpy as np, matplotlib.pyplot as plt
from pydbsr import reference
from pydbsr.outer import maxwell, bugrova
%matplotlib inline

ref = reference.read_xlsx(XLSX)
ofile = WORK / "compare" / f"omega_J{JMAX:g}_E{EMAX:g}_dE{DE:g}.npz"
cs = db.CollisionStrengths.load(ofile)
tg = db.Target.load(WORK / "target")
db.nist.assign(tg.states, ref.levels, verbose=False)
name_of = {s.nist_no: s.name for s in tg.states if s.nist_no is not None and s.name in cs.names}
print(ofile.name, "|", len(cs.names), "состояний,", len(cs.pw or []), "парциальных волн,",
      f"E до {cs.energies.max():.0f} эВ")
""")

code("""
# сечения: pydbsr против Wang 2019, по листам таблицы
for sheet in sorted(set(ref.sheet.values())):
    trs = [t for t in ref.transitions(sheet) if t[0] in name_of and t[1] in name_of][:9]
    if not trs:
        continue
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), squeeze=False)
    for ax, (i, j) in zip(axs.flat, trs):
        e_ref, s_ref = ref.sigma[(i, j)]
        ax.plot(e_ref, s_ref / 1e-16, lw=0.8, label="Wang 2019")
        ax.plot(cs.incident_energy(name_of[i]), cs.sigma(name_of[i], name_of[j]) / 1e-16, lw=0.8,
                label=f"pydbsr, J ≤ {JMAX:g}")
        ax.set_title(f"{i}→{j}: {ref.label(j)}", fontsize=9)
        ax.set_xscale("log"); ax.set_xlabel("E, эВ"); ax.set_ylabel("σ, $10^{-16}$ см²")
    for ax in list(axs.flat)[len(trs):]:
        ax.set_visible(False)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(sheet); fig.tight_layout(); plt.show()
""")

code("""
# сходимость по J: вклад парциальных волн в Ω(i→j) при нескольких энергиях
i, j = 1, 4                                  # номера уровней NIST
if cs.omega_pw is not None:
    a, b = cs.index(name_of[i]), cs.index(name_of[j])
    J = np.array([p[0] / 2 for p in cs.pw])
    for E in (15, 30, 50):
        k = np.argmin(abs(cs.energies - E))
        om = np.bincount(np.round(J * 2).astype(int), weights=cs.omega_pw[k, :, a, b])
        plt.semilogy(np.arange(len(om)) / 2, np.maximum(om, 1e-12), "o-", ms=3, label=f"{cs.energies[k]:.0f} эВ")
    plt.xlabel("J"); plt.ylabel(f"вклад в Ω({i}→{j})"); plt.legend(); plt.show()
""")

code("""
# скорости возбуждения (Максвелл и Бугрова), наши и Wang, в одном диапазоне энергий
rows = []
for (i, j) in ref.transitions():
    if i not in name_of or j not in name_of:
        continue
    e_ref, s_ref = ref.sigma[(i, j)]
    m = e_ref <= cs.energies.max()
    if m.sum() < 2:
        continue
    for Te in (2, 5, 10, 20):
        f = maxwell(Te)
        ours = cs.rate_eedf(name_of[i], name_of[j], f)
        theirs = np.trapezoid(s_ref[m] * 5.930969e7 * np.sqrt(e_ref[m]) * f(e_ref[m]), e_ref[m])
        rows.append((i, j, ref.sheet[(i, j)], Te, ours, theirs))
print(f"{'i':>3} {'j':>3}  {'лист':10s} {'Te':>4}  {'pydbsr':>10} {'Wang':>10}  отношение")
for i, j, sh, Te, a, b in rows[:60]:
    print(f"{i:3d} {j:3d}  {sh:10s} {Te:4d}  {a:10.3e} {b:10.3e}  {a / b if b > 0 else float('nan'):6.2f}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (pydbsr)", "language": "python",
                                                  "name": "pydbsr"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for k, c in enumerate(nb["cells"]):
    c["id"] = f"c{k:02d}"
out = Path(__file__).resolve().parents[1] / "examples" / "xe_plus_wang2019.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("written", out)
