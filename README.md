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

📖 **Подробная документация — [docs/](docs/README.md)**: установка, быстрый старт
(в т.ч. из Jupyter — [examples/xe_plus_wang2019.ipynb](examples/xe_plus_wang2019.ipynb)),
мишень, рассеяние, внешняя область, [физический чек-лист](docs/physics.md),
[проверки](docs/validation.md), [ошибки](docs/troubleshooting.md), [API](docs/api.md).

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
5. **dbsr_mchf:** списки `levels=`/`weights=` (280 символов) → 2048; J‑блок без
   оптимизируемых уровней больше не роняет DSYEVX.
6. **dbsr_breit3 (и MPI‑версия):** имя расчёта (80), имя `name.int_new` (40) и команда
   `cat name.int_new >> name.int_res` (80) переполнялись при именах длиннее ~32 символов
   (`Fortran runtime error: End of record`) → 256/2048; то же для команды в `dbsr_mchf`.

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
# все состояния одной чётности — одним КВ‑расчётом: тогда они ортогональны и не
# взаимодействуют друг с другом (иначе dbsr_mat3 пишет «Target hamiltonian errors»,
# и эта паразитная связь попадает в уравнения сильной связи каналов).
# Корреляционная орбиталь 6d (dbsr_mchf по физическим состояниям) даёт
# термозависимость 5d; в результате остаются только физические состояния.
tg.add(["5s 5p6", "5s2 5p4 5d", "5s2 5p4 6s"], correlation=["5s2 5p4 6d"])
tg.add(["5s2 5p5", "5s2 5p4 6p"])   # нечётные: вместе с основным (опорные состояния заменяются)
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

### Линии NIST (какие уровни стоят за наблюдаемой линией)

```python
lines = db.nist.fetch_lines("Xe III", 474.5, 478.5)       # через ASDCache, если установлен (pip install "pydbsr[nist]")
levels = db.nist.fetch_levels("Xe III")
for line, up in db.nist.upper_levels(lines, levels):
    print(line, "->", up.no if up else None)
```
ASDCache (кэш ответов NIST на 2 недели) умеет только линии; уровни `pydbsr` берёт сам
из формы уровней ASD. Из терминала: `pydbsr lines "Xe III" 474 479 --csv lines.csv`,
`pydbsr nist "Xe III" --csv levels.csv`.

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
tg.refine("5s1__5p4_5d1")                # свои орбитали для каждого родительского терма:
                                         # только для структуры — состояния неортогональны,
                                         # dbsr_hd3 с ними не работает (используйте correlation=)
```

## Установка на сервер (большой диск, маленький домашний каталог)

```bash
df -h                                   # найти раздел на ~1 ТБ, например /data
curl -fsSLO https://raw.githubusercontent.com/ChuWanningYuheng/DBSR/claude/zealous-hawking-l7bm06/tools/setup_server.sh
bash setup_server.sh /data/pydbsr_calc  # Miniforge, окружение, кэши, tmp — всё туда
source /data/pydbsr_calc/env.sh         # в каждой новой сессии
```
Скрипт ставит conda (Miniforge) с gfortran, CMake и OpenBLAS/LAPACK в
`/data/pydbsr_calc`, собирает пакет и регистрирует ядро Jupyter «Python (pydbsr)».

## Большие расчёты: потоковый режим

Файлы `dbsr_mat.nnn` занимают примерно ½·N²·8 байт на волну (N — размер матрицы):
при N ≈ 50 000 это около 10 ГБ на волну, а при ~100 волнах — около 1 ТБ.
`run_streamed` обрабатывает волну целиком (breit → mat → hd), удаляет
`dbsr_mat.nnn` и `int_bnk.nnn` сразу после записи `h.nnn` и следит за общим
бюджетом ядер и памяти (`dbsr_hd3` держит в памяти ≈ 2·N²·8 байт):

```python
sc = db.Scattering(tg, "run", states=..., jmax=25)
sc.prepare(); sc.run_prep(); sc.run_conf()
print(sc.wave_sizes())                                   # каналы, N, память по волнам
sc.run_streamed(cores=64, mem_gb=200, hd_threads=16)     # повторный запуск продолжает с места обрыва
```

Долгие расчёты удобно запускать в фоне: `nohup python run.py > run.log 2>&1 &`
(или внутри `tmux`/`screen`).

## Сравнение с Wang et al (2019): e + Xe⁺, DBSR, 67 состояний

```bash
python examples/xe_plus_vs_wang2019.py --xlsx CrossSectionsIon.xlsx --jmax 10 --cores 64 --mem 200
python examples/xe_plus_vs_wang2019.py --xlsx CrossSectionsIon.xlsx --jmax 25 --cores 64 --mem 200  # добавит J = 11..25
```
Уровни и номера NIST берутся из листа «NIST Level Table» файла, пороги сдвигаются
на NIST (как в статье). Результат — `compare/*.png` (σ, наше и из статьи) и
`compare/rates.txt` (скорости для Максвелла и распределения Бугровой при
Tₑ = 2, 5, 10, 20 эВ и их отношения).

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

Сшивка на границе — с релятивистской кинематикой (по умолчанию): вне r = a большая
компонента уравнения Дирака подчиняется `P'' = [l(l+1)/r² − 2z(1+e/c²)/r − k²] P`,
`k² = 2e(1 + e/2c²)`. В STGF из пакета DBSR сшивка нерелятивистская; разница —
фаза ≈ k·a·e/4c² (3·10⁻³ рад при 60 эВ и a = 50 a₀): Σ|T|² меняется на ~0.2 %, слабые
переходы — до нескольких процентов. `OuterRegion(..., relativistic=False)` — как в STGF.

Ограничения: нет top‑up для высоких J (сходимость по J смотрите через
`collision_strengths(..., per_partial_wave=True)` → `cs.omega_pw`), дальнодействующий
потенциал после `r_match` не учитывается (в STGF это делает разложение Гайлитиса);
не учтены члены O(z²/c²r²) и множитель (1 + (e − V)/2c²) при переводе Q → P′ на
границе (~10⁻⁴). Шаг пропагатора при `r_match > a`: h·k ≤ 0.01 (ошибка K ~10⁻⁸).

Канал, у которого энергия совпадает с порогом (|k²| < 1e‑10 а.е.), считается только что
открывшимся: нормированные на энергию кулоновские функции имеют конечный предел при k → 0⁺.

Мультиполи: `dbsr_mat3` отбрасывает интегралы Rᵏ с k > `mk` (по умолчанию 7), т.е. часть
обмена для континуума с l ≥ 6. pydbsr для каждой парциальной волны передаёт
`mk = l_cont,max + l_bound,max` (все нужные мультиполи); ограничить: `Scattering(..., mk_max=...)`.

### Стресс‑тест

```bash
python tools/stress_test.py --target RUN/target --scat RUN/scat --log stress_log.json
```
Проверяет на настоящих выходных файлах DBSR: уравнение Дирака (водородоподобный Xe⁵³⁺ —
энергии и отношение Q/P с точной формулой), нормировку и кинетический баланс орбиталей
мишени, тонкую структуру 5p⁵ ²P, вронскиан кулоновских функций, полноту набора каналов,
симметрию K до симметризации (сохранение потока), унитарность S, |T|² ≤ 4, пороги
(точно на пороге и ±1e‑12), положительную определённость матрицы перекрытий,
согласованность состояний мишени, убийство процессов по таймауту. Все предупреждения
numpy/Python — ошибки.

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
