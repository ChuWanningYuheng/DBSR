# Установка

pydbsr — это Python-пакет плюс ~60 скомпилированных программ DBSR (Fortran).
Исходники DBSR скачиваются при сборке с GitHub O. Zatsarinny по закреплённым
коммитам, к ним автоматически применяются исправления (`tools/patch_upstream.py`,
список — в корневом README).

Нужно: Linux или macOS, Python ≥ 3.9, компилятор Fortran (gfortran ≥ 9), CMake ≥ 3.18,
BLAS/LAPACK. Всё это ставится через conda без прав администратора.

## 1. Сервер без root (рекомендуется): одна команда

Скрипт ставит Miniforge (conda), окружение с gfortran/CMake/OpenBLAS,
собирает pydbsr, клонирует примеры и регистрирует ядро Jupyter. Всё кладётся
в одну папку на **большом диске** (домашний каталог на кластерах обычно маленький).

```bash
df -h                                     # найти большой диск, например /data
cd /data
curl -fsSLO https://raw.githubusercontent.com/ChuWanningYuheng/DBSR/claude/zealous-hawking-l7bm06/tools/setup_server.sh
bash setup_server.sh /data/pydbsr_calc    # 20–40 минут
```

После установки в `/data/pydbsr_calc`:

| Путь | Что это |
|---|---|
| `env.sh` | включение окружения: `source /data/pydbsr_calc/env.sh` (в каждом новом терминале) |
| `env/` | conda-окружение (Python, gfortran, numpy, scipy, matplotlib, jupyterlab) |
| `DBSR_src/` | исходники: примеры `examples/`, стресс-тест `tools/stress_test.py`, документация |
| `runs/` | папка для расчётов; сюда скопирован ноутбук `xe_plus_wang2019.ipynb` |
| `install.log` | лог сборки (если что-то не собралось) |
| `cache/`, `tmp/` | кэши pip/conda/NIST и временные файлы |

**Обновление до новой версии** — тот же скрипт ещё раз (пересоберёт pydbsr и
обновит `DBSR_src`):
```bash
bash /data/pydbsr_calc/DBSR_src/tools/setup_server.sh /data/pydbsr_calc
```

## 2. Jupyter

Скрипт регистрирует ядро **«Python (pydbsr)»**. Если на сервере уже есть
JupyterHub / Jupyter — просто выберите это ядро. Если нет, запустите свой:

```bash
source /data/pydbsr_calc/env.sh
jupyter lab --no-browser --port 8890        # на сервере
# на своём компьютере — туннель, потом открыть http://localhost:8890 (токен из вывода jupyter)
ssh -N -L 8890:localhost:8890 user@server
```

Чтобы Jupyter не умер при выходе из ssh, запускайте его в `tmux`/`screen`
(`tmux new -s jl`, выйти — `Ctrl+B D`, вернуться — `tmux attach -t jl`).
Сам расчёт из ноутбука-примера и так идёт отдельным процессом (см. [quickstart.md](quickstart.md)).

## 3. Своя машина: pip

```bash
conda create -n pydbsr -c conda-forge python=3.11 fortran-compiler c-compiler cmake ninja \
      "libblas=*=*openblas" liblapack numpy scipy mpmath matplotlib openpyxl scikit-build-core
conda activate pydbsr
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
pip install --no-build-isolation -v "git+https://github.com/ChuWanningYuheng/DBSR.git"
```

Из клона репозитория: `pip install --no-build-isolation -v .`
Для разработки (правки Python без пересборки Fortran): `pip install -e . --no-build-isolation`.

Системные пакеты вместо conda (Ubuntu/Debian):
`sudo apt install gfortran cmake libopenblas-dev liblapack-dev`, затем `pip install .`.

### Параметры сборки (CMake)

Передаются так: `pip install . -C cmake.define.ИМЯ=ЗНАЧЕНИЕ`

| Опция | По умолчанию | Смысл |
|---|---|---|
| `DBSR_LARGE_MEMORY` | ON | `-mcmodel=medium`: статические массивы > 2 ГБ |
| `DBSR_DEFAULT_REAL8` | OFF | `-fdefault-real-8` (DBSR и так везде `Real(8)`) |
| `DBSR_NATIVE` | OFF | `-march=native` (быстрее, но бинарники не переносимы) |
| `DBSR_ENABLE_MPI` | OFF | MPI-версии (`dbsr_conf3_mpi`, `dbsr_breit3_mpi`, `dbsr_mat3_mpi`) |
| `DBSR_ENABLE_SCALAPACK` | OFF | `dbsr_hd3_mpi` (ScaLAPACK; для матриц, не влезающих в память одного узла) |
| `DBSR_BUILD_UTILS` | ON | утилиты Zatsarinny (`jcfile`, `sum_hh_jj`, `dbound_tab`, `bsw_rw`, …) |
| `DBSR_STATIC_RUNTIME` | OFF | статически прилинковать libgfortran (для колёс) |
| `DBSR_DEBUG_CHECKS` | OFF | `-fcheck=all -g` (медленно, для отладки) |
| `DBSR_UPSTREAM_DIR` | — | взять исходники DBSR из локальной папки вместо GitHub |

## 3a. Сервер без доступа к GitHub

Соберите архив со всеми исходниками там, где GitHub доступен:
```bash
bash tools/make_offline_bundle.sh pydbsr_offline.tar.gz      # pydbsr + DBSR3/LIBRARIES/UTILS
```
Перенесите его на сервер (scp, флешка) и соберите без интернета (conda-окружение
как в п. 3):
```bash
tar xzf pydbsr_offline.tar.gz
cd pydbsr_src && pip install --no-build-isolation --no-deps -v .
```
Исходники DBSR берутся из `pydbsr_src/upstream/`; примеры и ноутбук — в `pydbsr_src/examples/`.

## 4. Проверка установки

```bash
pydbsr info                 # версия, где лежат программы, список программ, какой бэкенд функций Кулона
cd /data/pydbsr_calc/DBSR_src && python -m pytest -q    # тесты (из клона); -m "not slow" — без сквозного расчёта
```
В `pydbsr info` должно быть `Coulomb functions: COULFG (Fortran)`. Если написано
`mpmath` — библиотека `libpydbsr_coulomb` не собралась; всё работает, но медленнее.

## 5. Google Colab

Работает для маленьких тестов (сборка ~15 минут, 2 ядра, 12 ГБ):
```python
!apt -qq install gfortran libopenblas-dev liblapack-dev > /dev/null
!pip -q install "git+https://github.com/ChuWanningYuheng/DBSR.git"
```
Для расчётов уровня статьи (десятки тысяч каналов·сплайнов) нужен сервер.

## Переменные окружения

| Переменная | Смысл |
|---|---|
| `DBSR_BIN` | папка с программами DBSR (если хотите использовать свою сборку) |
| `PYDBSR_COULOMB_LIB` | путь к `libpydbsr_coulomb.so` |
| `PYDBSR_CACHE` | кэш таблиц NIST (по умолчанию `~/.cache/pydbsr`) |
| `OMP_NUM_THREADS` | не выставляйте вручную для расчётов: pydbsr задаёт потоки каждой программе сам |
