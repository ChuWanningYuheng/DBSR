# pydbsr — DBSR как Python‑библиотека

`pydbsr` — Python‑обёртка над релятивистским R‑матричным комплексом **DBSR**
(Dirac B‑spline R‑matrix, O. Zatsarinny, <https://github.com/zatsaroi/DBSR3>).
Расчёт сечений возбуждения собирается из готовых блоков:

```
Ion ─► Target (dbsr_hf → jj‑состояния) ─► [NIST‑пороги] ─► Scattering (prep→conf→breit→mat→hd)
                                                              └─► OuterRegion ─► Ω, σ, Υ(T), k(T)
```

* все программы DBSR3 и утилиты собираются автоматически при `pip install`;
* мишень задаётся ионом и списком конфигураций, дальше всё делается само:
  2 шага `dbsr_hf` (LS → jj), разбиение `.j` на отдельные состояния (аналог `jcfile`,
  побайтово совпадает), `target_jj`, `dbsr_par`, `knot.dat`;
* уровни энергии из NIST ASD подставляются как экспериментальные пороги (`dbsr_hd iexp=1`);
* парциальные волны считаются параллельно (`jobs` процессов × `threads` потоков OpenMP/BLAS),
  при сборке с MPI используются `*_mpi` версии;
* внешняя область (в DBSR её нет — там нужен STGF/FARM) написана на Python:
  K‑матрицы, силы столкновений Ω, сечения σ, эффективные силы Υ(T) и константы скоростей;
* любую программу можно вызвать напрямую: `pydbsr.run("dbsr_mult3", [...], cwd)`.

## Установка

Нужны компилятор Fortran (gfortran ≥ 9 или ifort/ifx), CMake ≥ 3.18, BLAS/LAPACK, git.

```bash
# pip (сборка из исходников, ~1 мин)
pip install git+https://github.com/chuwanningyuheng/dbsr.git

# conda: окружение с компиляторами и openblas, затем pip
conda env create -f environment.yml && conda activate pydbsr
pip install .

# или собрать conda‑пакет
conda build conda/ && conda install --use-local pydbsr
```

Исходники Зацаринного (`DBSR3`, `LIBRARIES`, `UTILS`) скачиваются при сборке с GitHub
на зафиксированных коммитах. На кластере без интернета:

```bash
git clone https://github.com/zatsaroi/DBSR3 up/DBSR3
git clone https://github.com/zatsaroi/LIBRARIES up/LIBRARIES
git clone https://github.com/zatsaroi/UTILS up/UTILS
DBSR_UPSTREAM_DIR=$PWD/up pip install .
```

Параметры сборки (передаются как `pip install . -C cmake.define.DBSR_ENABLE_MPI=ON` и т.п.):

| опция | по умолчанию | смысл |
|---|---|---|
| `DBSR_ENABLE_MPI` | OFF | `dbsr_conf3_mpi`, `dbsr_breit3_mpi`, `dbsr_mat3_mpi` |
| `DBSR_ENABLE_SCALAPACK` | OFF | `dbsr_hd3_mpi` (ScaLAPACK) |
| `DBSR_LARGE_MEMORY` | ON | `-mcmodel=medium`: статические данные > 2 ГБ |
| `DBSR_DEFAULT_REAL8` | OFF | `-fdefault-real-8` (весь код DBSR и так `Real(8)`) |
| `DBSR_NATIVE` | OFF | `-march=native` |
| `DBSR_DEBUG_CHECKS` | OFF | `-fcheck=bounds -fbacktrace` |

Проверка: `pydbsr info` (список программ, путь к ним, какая реализация кулоновских функций).

### Исправления в исходниках DBSR

Применяются автоматически к копии исходников при сборке (`tools/patch_upstream.py`,
оригиналы не трогаются):

1. **Длина строк аргументов и параметров.** `Read_*arg`/`Read_*par` читали строку в
   `Character(80)` (некоторые — 180): длинные `conf=…`, `varied=…`, `core=…`, пути к файлам
   молча обрезались, а потом падали с `Bad integer for item N in list input` / `End of file`.
   Буфер увеличен до 2048.
2. **dbsr_hf:** `configuration`/`conf_LS`/`conf_AV` (160 символов), список варьируемых орбиталей
   `varied` (80) и строки в `Def_atom` (200) → 2048.
3. Длина имён файлов в модулях программ `ma = 80` → 256.
4. `ZCOM/recup_a.f90`: комментарий с `\` в конце строки. Если файл проходит через cpp
   (так делает генератор Ninja), следующий оператор исчезает, и `dbsr_conf3` падает с
   `ZRECUP: schemes Y1 and Y2 are incompatible`. Кроме патча, в CMake препроцессор отключён.

Размеры массивов (`max_ch`, `max_nc`, `max_wf` и т.п.) в DBSR3 динамические: считаются
в `dbsr_conf3` и записываются в `target_jj`. Жёстких лимитов там нет.

## Пример: e + Xe⁺

```python
import numpy as np
import pydbsr as db

ion = db.Ion("Xe", 1)                                  # Xe+: Z=54, N=53
tg = db.Target(ion, core="[Kr]4d10", workdir="xe_target",
               grid={"rmax": 50.0, "hmax": 0.5})       # B‑сплайны; rmax = радиус R‑матрицы
tg.add("5s2 5p5")          # опорная: оптимизируются все орбитали
tg.add("5s 5p6")           # орбитали 5s,5p берутся из опорной
tg.add("5s2 5p4 5d")       # оптимизируется только 5d
tg.add("5s2 5p4 6s")
tg.add("5s2 5p4 6p")
tg.compute(jobs=4)         # 60 jj‑состояний, каждое в своём name.c/name.bsw
print(tg.table())

db.nist.assign(tg.states, ion)           # уровни Xe II из NIST (кэшируются в ~/.cache/pydbsr)

sc = db.Scattering(tg, "xe_scat", states=tg.select(max_states=20),
                   jmax=5, jobs=8, threads=4)       # 2J = 0..10, обе чётности
sc.run()                                            # prep → conf → breit → mat → hd

cs = sc.outer(r_match=100.0).collision_strengths(np.linspace(0.01, 40, 4000), jobs=32)
g = cs.names[0]                                     # основное состояние
sigma = cs.sigma(g, cs.names[3])                    # см²; аргументы — имена, State или индексы
ups = cs.upsilon(g, cs.names[3], T=[5e3, 1e4, 2e4]) # Υ(T), T в K
rate = cs.rate(g, cs.names[3], T=1e4)               # см³/с
cs.save("omega.npz")
```

Полный скрипт с выводом таблиц и графиков: [`examples/xe_plus_excitation.py`](examples/xe_plus_excitation.py).

### Параллельность

* `Target.compute(jobs=n)` — конфигурации мишени считаются одновременно, после опорной.
* `Scattering(..., jobs=n, threads=m)` — `dbsr_breit3`, `dbsr_mat3`, `dbsr_hd3` запускаются по
  одной парциальной волне в отдельной песочнице, по `n` одновременно, у каждой `m`
  потоков OpenMP/BLAS. На одном узле с 32 ядрами разумно `jobs=8, threads=4`.
* `Scattering(..., mpi=64)` — `mpirun -np 64 dbsr_*_mpi` (нужна сборка с MPI).
* `collision_strengths(..., jobs=n)` — энергии раздаются по процессам.
* Шаги можно запускать по отдельности: `sc.run(steps=["prep", "conf"])`,
  `sc.run_mat()`, `sc.run_hd()`. Состояние сохраняется в `pydbsr_*.json`, поэтому
  `Scattering.load("xe_scat")` / `Target.load("xe_target")` продолжают работу в новой сессии.

### Уровни NIST без интернета

```bash
pydbsr nist "Xe II" -o xe2.tsv      # на машине с интернетом
```
```python
db.nist.assign(tg.states, db.nist.read_levels("xe2.tsv"))
```
Уровни сопоставляются состояниям по (открытые оболочки конфигурации, J, чётность), внутри
группы — по порядку энергий. Состояния без пары остаются с расчётной энергией, выдаётся
предупреждение. Если экспериментальная энергия нужна для всех порогов, а у части
состояний её нет, `Scattering` просит либо назначить её вручную
(`state.exp_energy_cm = ...`), либо включить `exp_energies=False`.

### Другие возможности DBSR

```python
sc.bound_states()                       # связанные состояния (N+1): dbsr_hd itype=-1 + dbound_tab
sc.call("dbsr_mult3", "cfg.001", "cfg.004", "E1")      # любая программа в рабочем каталоге
sc.call("dbsr_dmat3", "cfg.001", "cfg.004", "b", "b")
db.run("dbsr_ci3", ["name"], cwd="xe_target")          # или напрямую
tg.add("5s2 5p4 5d", ci=True)            # после dbsr_hf ещё dbsr_breit3 + dbsr_ci3 (КВ с Брейтом)
```

## Внешняя область: что именно считается

Уравнения — из руководства DBSR (разд. 2.2, ур. 2.15–2.36):

* R‑матрица на границе `a`: `R_ij = 1/(2a) Σ_k w_ik w_jk / (E_k − E)`, откуда
  лог‑производная `Y(a) = (R⁻¹ + b)/a`;
* при `r_match > a` лог‑производная распространяется методом Джонсона (4‑й порядок,
  устойчив при закрытых каналах) с дальнодействующими коэффициентами из `h.nnn`;
* сшивка с энергетически нормированными кулоновскими функциями (Barnett COULFG из ZCOM,
  с проверкой по mpmath) в открытых каналах и с затухающими в закрытых даёт K;
  `S = (1+iK)(1−iK)⁻¹`, `T = S − 1`;
* `Ω(i→j) = ½ Σ_J (2J+1) Σ |T|²`, `σ = πa₀² Ω / (g_i k_i²)`.

Ограничения: нет top‑up для высоких J (сходимость по J смотрите через
`collision_strengths(..., per_partial_wave=True)` → `cs.omega_pw`), дальнодействующий
потенциал после `r_match` не учитывается (в STGF это делает разложение Гайлитиса),
внешняя область нерелятивистская (как в STGF/FARM; для сильно ионизованных ионов этого
может не хватить).

Проверки (`tests/`): фаза кулоновского рассеяния восстанавливается с точностью 1e‑7,
пропагатор сверен с прямым интегрированием ОДУ, K не зависит от радиуса сшивки,
чтение `h.nnn` совпадает с `sum_hh_jj`, разбиение `.j` совпадает с `jcfile` побайтово,
полный прогон e + Xe⁺ (мишень → Ω → связанные состояния).

## Лицензии

Python‑часть и скрипты сборки — MIT (`LICENSE`). Fortran‑код DBSR, LIBRARIES и UTILS
принадлежит O. Zatsarinny и в этот репозиторий не входит: он скачивается при сборке
с <https://github.com/zatsaroi>. В тех репозиториях нет файла лицензии, поэтому
перед распространением собранных пакетов (колёс PyPI, conda‑пакетов с бинарниками)
уточните условия у автора. Публикации с расчётами должны ссылаться на DBSR:
O. Zatsarinny & K. Bartschat, *J. Phys. B* **46**, 112001 (2013);
O. Zatsarinny & C. Froese Fischer, *Comput. Phys. Commun.* **202**, 287 (2016).
