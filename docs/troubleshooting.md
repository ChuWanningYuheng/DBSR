# Ошибки и что с ними делать

pydbsr ловит `STOP ...` и ошибки времени выполнения Fortran и бросает
`DBSRError(программа, папка, сообщение, хвост вывода)`. Полный вывод — в
`программа.out.nnn` / `name.hf1.out` и т.п. в рабочей папке.

| Сообщение | Причина | Что делать |
|---|---|---|
| `dbsr_hd3: DPOTRF info > 0 - Cholesky factorization failed` | матрица перекрытий (N+1)-базиса вырождена: (а) состояния мишени одного J неортогональны (из разных наборов орбиталей, `refine`); (б) континуум неортогонален корреляционной орбитали | (а) состояния одной чётности — в один КВ (`tg.add([...])`), не использовать `refine` для рассеяния; (б) `correlation=` в `tg.add` — pydbsr ставит условия сам; для ручных случаев `params`/`dbsr_par`: `< kd  \| 6d  >=0` |
| `dbsr_conf3: Can not find physical configuration` | в `target_jj`/файлах состояния нет физической конфигурации (обычно — неверные имена/порядок файлов состояний) | создавайте `target_jj` через `Scattering.prepare()` |
| `Bad integer for item N in list input`, `End of file`, `End of record` | строка длиннее буфера Fortran (длинные `conf=`, `varied=`, имена) | в pydbsr буферы увеличены патчем; если встретилось в другой программе — сообщите, добавим в `tools/patch_upstream.py` |
| `nsw <> ns` / `ksw <> ks` | орбитали посчитаны на другой сетке | одна сетка для всех расчётов: `Target(grid=...)`, не меняйте `knot.dat` вручную |
| `DSYEVX ... parameter number 10 had an illegal value` (dbsr_mchf) | J-блок без оптимизируемых уровней | исправлено патчем |
| `ZRECUP: schemes Y1 and Y2 are incompatible` | исходники прошли через cpp (Ninja) | исправлено патчем + препроцессор отключён в CMake |
| `CF2 HAS FAILED TO CONVERGE` (печатает COULFG) | |η| > 2·10⁴ | в pydbsr ≥ этой версии не возникает (mpmath) |
| `ValueError: invalid literal for int() ... 'Home"><'` | NIST вернул HTML | обновите pydbsr; или скачайте таблицу вручную (см. [nist.md](nist.md)) |
| `WARNING: target states i and j interact` | состояния мишени из разных расчётов взаимодействуют | объединить в один КВ (см. [physics.md](physics.md#1-согласованность-мишени)) |
| `no NIST level found for N state(s)` | не отождествлено | проверьте конфигурации; обычно такие состояния исключают |
| Уровни с p₁/₂ (6p-) на эВ выше NIST | варьировалась только p₃/₂ | в pydbsr `6p` → `6p-,6p` автоматически |
| `PYDBSR_DSYEVR: no ILP64 LAPACK ..., DSYEVR used` в выводе dbsr_hd3 | матрица > 32 000, нет LAPACK с 64-битными целыми | `pip install scipy-openblas64` (иначе диагонализация может быть в разы медленнее) |
| Память: процесс убит (OOM) | волна больше `mem_gb` | уменьшите `cores`/`hd_threads` или `mem_gb`; самые большие волны идут по одной |
| Диск переполнен | временные `dbsr_mat.nnn` | `cleanup=True` (по умолчанию), `scratch=` на большой диск, меньше волн одновременно |

## Перезапуск после сбоя

* Мишень: `tg.compute()` пропускает готовые конфигурации.
* Рассеяние: `sc.run_streamed(...)` пропускает волны с готовым `h.nnn`.
* Внешняя область: пример сохраняет `omega_J*_E*_dE*.npz` и переиспользует.

## Как сообщить об ошибке

Приложите: вывод `pydbsr info`, скрипт/ячейки, хвост `*.out.nnn` и, если
возможно, вывод `tools/stress_test.py`.
