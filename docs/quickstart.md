# Быстрый старт

## 1. Проверка за 2 минуты: 3 состояния Xe⁺, J ≤ 1

```python
import numpy as np
import pydbsr as db

ion = db.Ion("Xe", 1)                                   # Xe II: Z = 54, 53 электрона
tg = db.Target(ion, core="[Kr]4d10", workdir="test/target")
tg.add("5s2 5p5")                                       # опорная конфигурация
tg.add("5s 5p6")
tg.compute()
print(tg.table())                                       # 2P3/2, 2P1/2, 2S1/2

sc = db.Scattering(tg, "test/scat", jmax=1, jobs=2, exp_energies=False)
sc.run()                                                # prep → conf → breit → mat → hd
cs = sc.outer().collision_strengths(np.linspace(0.1, 30, 300))
print(cs.names)
print(cs.sigma(cs.names[0], cs.names[2]))               # σ(2P3/2 → 2S1/2), см²
```

Это то же, что делает тест `tests/test_pipeline.py`.

## 2. Полный расчёт e + Xe⁺ (67 состояний) из Jupyter

Ноутбук `examples/xe_plus_wang2019.ipynb` (при установке через `setup_server.sh`
он уже лежит в `runs/`). Ядро — **Python (pydbsr)**.

1. **Ячейка 1 — параметры.** Пути:
   * `ROOT` — папка установки на большом диске (`/data/pydbsr_calc`);
   * `WORK` — куда сохраняются результаты (большой диск). **Новая папка для каждой
     новой модели мишени**: мишень кэшируется в `WORK/target/pydbsr_target.json`;
   * `SCRATCH` — быстрая папка для временных матриц (например, домашняя);
     `SCRATCH_GB` — сколько места там можно занять;
   * `XLSX` — таблица Wang et al (2019) для сравнения.

   Ресурсы: `CORES`, `MEM` (ГБ), `HD_THREADS` (потоков на одну диагонализацию), `JMAX`.
2. **Ячейка 2 — запуск.** Расчёт стартует отдельным процессом (`start_new_session`):
   браузер можно закрыть, ядро перезапустить — расчёт продолжится. Лог: `WORK.J<jmax>.log`.
3. **Ячейка 3 — ход расчёта.** Запускайте, когда хотите: сколько волн готово, хвост лога.
4. **Ячейка «остановить»** — если нужно прервать.
5. **Проверка** — стресс-тест на ваших данных (после J = 0), все строки `[PASS]`.
6. **Графики**: σ(E) pydbsr против Wang по всем листам таблицы, сходимость по J,
   таблица скоростей для Максвелла.

Рекомендуемый порядок: `JMAX = 0` (≈1 ч на 64 ядрах) → проверка → `JMAX = 10`
→ `25` → `50`. При каждом увеличении JMAX считаются только новые волны.

## 3. То же из терминала

```bash
source /data/pydbsr_calc/env.sh
cd /data/pydbsr_calc
W=runs/xe_plus_v2
nohup python DBSR_src/examples/xe_plus_vs_wang2019.py --xlsx CrossSectionsIon_3.xlsx \
      --workdir $W --scratch ~/pydbsr_tmp --scratch-gb 5 \
      --jmax 0 --cores 64 --mem 200 --hd-threads 16 > $W.J0.log 2>&1 &
tail -f $W.J0.log
```
Все параметры: `python DBSR_src/examples/xe_plus_vs_wang2019.py --help`.

## 4. Свой ион: шаблон

```python
import numpy as np, pydbsr as db

ion = db.Ion("Xe", 2)                                    # Xe III
tg = db.Target(ion, core="[Kr]4d10", workdir="xe3/target",
               grid={"rmax": 40.0, "hmax": 0.5})          # радиус R-матрицы, шаг сплайнов
tg.add("5s2 5p4")                                        # опорная
tg.add(["5s 5p5", "5s2 5p3 5d", "5s2 5p3 6s"],           # чётность 1 — одним КВ
       correlation=["5s2 5p3 6d"])
tg.add(["5s2 5p4", "5s2 5p3 6p"])                        # чётность 2 — вместе с основной
tg.compute(jobs=4)
db.nist.assign(tg.states, ion)                           # уровни NIST (нужен интернет или CSV)
print(tg.table())

states = [s for s in tg.states if s.nist_no is not None]  # только отождествлённые
sc = db.Scattering(tg, "xe3/scat", states=states, jmax=10, exp_energies=True)
sc.prepare(); sc.run_prep(); sc.run_conf()
print(sc.wave_sizes()[-1])                               # размер самой большой волны
sc.run_streamed(cores=64, mem_gb=200, hd_threads=16, scratch="~/pydbsr_tmp", scratch_gb=5)

cs = sc.outer().collision_strengths(np.arange(0.01, 60, 0.0136), jobs=64, per_partial_wave=True)
cs.save("xe3/omega.npz")
```

Подробности каждого шага — в [target.md](target.md), [scattering.md](scattering.md), [outer.md](outer.md).
