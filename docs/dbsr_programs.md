# Программы DBSR и файлы

pydbsr вызывает программы DBSR3 (O. Zatsarinny) как отдельные процессы в
рабочей папке. Любую программу можно запустить напрямую:
```python
db.run("dbsr_mult3", ["cfg.001", "cfg.004", "E1"], cwd="scat")
sc.call("dbsr_dmat3", "cfg.001", "cfg.004", "b", "b")      # в папке рассеяния
```
```bash
pydbsr run dbsr_hf Xe          # то же из терминала; pydbsr bin — папка с программами
```
Справка по параметрам программы: `pydbsr run <программа> ?` (у большинства
программ) или исходник `read_arg.f90`.

## Цепочка

| Шаг | Программа | Вход | Выход | В pydbsr |
|---|---|---|---|---|
| Орбитали мишени | `dbsr_hf` | конфигурации, `knot.dat` | `name.c`, `name.bsw`, `name.j` | `Target.compute` |
| КВ / MCHF мишени | `dbsr_breit3`, `dbsr_ci3`, `dbsr_mchf` | `name.c`, `name.bsw` | `name.j`, новые `name.bsw` | `add(ci=True)`, `add(correlation=...)` |
| Разбиение на состояния | (`jcfile`) | `name.j` | `state.c`, `state.bsw` | `Target._split_states` (побайтово как jcfile) |
| Подготовка | `dbsr_prep3` | `target_jj`, состояния | `target.bsw`, `target_orb`, `targ_nnn.c` | `run_prep` |
| Каналы | `dbsr_conf3` | `target_jj`, `dbsr_par` | `cfg.nnn`, `pert_comp.nnn` | `run_conf` |
| Угловые коэффициенты | `dbsr_breit3` | `cfg.nnn` | `int_bnk.nnn` | `run_breit` / `run_streamed` |
| Гамильтониан | `dbsr_mat3` | `cfg.nnn`, `int_bnk.nnn` | `dbsr_mat.nnn`, `mat_log.nnn` | `run_mat` / `run_streamed` (с `mk=`) |
| Диагонализация | `dbsr_hd3` | `dbsr_mat.nnn`, `thresholds` | `h.nnn` (R-матрица) | `run_hd` / `run_streamed` |
| Внешняя область | (STGF в пакете BSR) | `h.nnn` | K, Ω | `OuterRegion` (Python) |
| Связанные (N+1) | `dbsr_hd3 itype=-1`, `dbound_tab` | | `dbound.nnn`, `dbound_tab` | `bound_states` |
| Переходы | `dbsr_mult3`, `dbsr_dmat3` | `cfg`, `bnk` | силы осцилляторов, A | `sc.call(...)` |
| Поляризуемость | `dbsr_pol3` | | | `db.run(...)` |

Утилиты (собираются при `DBSR_BUILD_UTILS=ON`): `jcfile`, `sum_hh_jj` (склейка
h-файлов), `dbound_tab`, `dbound_bsw`, `bsw_rw` (bsw → формат GRASP), `rw_bsw`,
`dbsr_merge`, `sec_*` (сечения из T-матриц программ STGF), `kma_*`, `oma_omb`,
`tma_tmb`, `f_values_jj`, `beb_ion` (ионизация по BEB), `adf04` (формат ADAS) и др.
Описание — `UTILS/DBSR_utils.pdf` в репозитории Zatsarinny.

## Нумерация

`klsp` — номер парциальной волны в `target_jj` (1, 2, …); файлы волны имеют
суффикс `.nnn` (`cfg.001`, `h.001`). Порядок волн — по J, внутри J — чётность +,
затем −. `Scattering.wave_sizes()` выдаёт klsp, 2J, чётность и размеры.

## Файлы рабочей папки рассеяния

| Файл | Что это |
|---|---|
| `target_jj` | список состояний, порогов, парциальных волн, каналов (после conf3) |
| `dbsr_par` | параметры для всех программ (`params=`) и условия ортогональности |
| `knot.dat` | сетка B-сплайнов |
| `thresholds` | экспериментальные энергии состояний (`exp_energies=True`) |
| `cfg.nnn` | конфигурации (N+1)-системы волны |
| `mat_log.nnn` | лог dbsr_mat3, в т.ч. ошибки мишени (`target_errors()`) |
| `h.nnn` | R-матрица волны: полюса, поверхностные амплитуды, коэффициенты дальнодействия |
| `pydbsr_scattering.json` | состояние объекта `Scattering` (`Scattering.load`) |
