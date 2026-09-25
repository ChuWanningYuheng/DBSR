# Справочник API

```python
import pydbsr as db
```

## `db.Ion(element, charge)`
`Ion("Xe", 1)`; свойства `z`, `nelc`, `symbol`, `spectrum` ("Xe II"), `name` ("Xe+").

## `db.Target(ion, core, workdir="target", grid=None, max_it=75, hf_args=None, echo=False)`
| Метод / атрибут | |
|---|---|
| `add(conf, name=None, varied=None, term="LS", jj_varied="none", ci=False, correlation=None, mchf_varied=None, mchf_max_it=100, **hf_args)` | конфигурация или список (один КВ); см. [target.md](target.md) |
| `compute(jobs=1, force=False, progress=print)` | dbsr_hf (+ mchf/ci), разбиение на состояния → `states` |
| `states` | список `State` |
| `ground`, `excitation_ev(state)` | основное состояние; энергия возбуждения |
| `select(max_states=None, emax_ev=None, configs=None, where=None)` | подмножество по энергии |
| `table(states=None)` | текстовая таблица |
| `superseded()` | расчёты, состояния которых заменены более полным КВ |
| `refine(spec, varied=None, key=None, ...)` | термозависимые орбитали по группам (только структура!) |
| `save()`, `Target.load(workdir)`, `Target.from_files(ion, workdir, names)` | сохранение / загрузка |

## `db.State`
`name, config, label, two_j, parity, energy (а.у.), source, exp_energy_cm, nist_label, nist_no, configs, correlation_orbitals`; `J`, `g`, `Jstr`.

## `db.Scattering(target, workdir, *, states=None, ion=None, jmax=None, jmin=None, partial_waves=None, params=None, prep_args=None, perturbers=(), exp_energies=True, jobs=1, threads=1, mpi=None, echo=False, progress=print, mk_max=None)`
| Метод | |
|---|---|
| `prepare()`, `run_prep()`, `run_conf()` | подготовка, каналы |
| `run_streamed(cores, mem_gb, hd_threads=8, cleanup=True, skip_done=True, itype=0, hd_args=None, scratch=None, scratch_gb=None)` | потоковый расчёт всех волн |
| `run(steps=...)`, `run_breit()`, `run_mat()`, `run_hd(itype=0, **kw)` | по программам |
| `wave_sizes()` | каналы, размер матриц, память по волнам |
| `multipole_max(klsp)`, `orth_conditions()` | что pydbsr передаёт DBSR |
| `target_errors(klsp=None)` | ⟨i\|H\|j⟩, ⟨i\|j⟩ мишени из `mat_log` |
| `outer(**kw)` | `OuterRegion` по `h.nnn` (с именами состояний) |
| `bound_states(msol=100)` | связанные (N+1)-состояния |
| `call(program, *args)` | любая программа в рабочей папке |
| `h_files()`, `h_target_names()`, `target_table()` | файлы и таблицы |
| `Scattering.load(workdir)` | восстановить объект |

## `db.OuterRegion(hdata, r_match=None, step=None, names=None, relativistic=True)`
`OuterRegion.from_files(paths, **kw)`; `collision_strengths(energies_ev, jobs=None, per_partial_wave=False)` → `CollisionStrengths`; `kmatrix(block, energy_ev)`; `thresholds_ev`.

## `db.CollisionStrengths`
Атрибуты: `energies` (эВ над основным), `thresholds`, `two_j`, `omega[E,i,j]`, `omega_pw[E,волна,i,j]`, `pw`, `names`, `relativistic`.
Методы: `index(state)`, `omega_ij(i,j)`, `sigma(i,j,units="cm2")`, `incident_energy(i)`, `upsilon(i,j,T)`, `rate(i,j,T)`, `rate_eedf(i,j,eedf)`, `save(path)`, `CollisionStrengths.load(path)`.
ФРЭЭ: `db.outer.maxwell(Te_eV)`, `db.outer.bugrova(Te_eV)`.

## `db.nist`
`fetch_levels(spectrum)`, `read_levels(path)`, `save_levels_csv(levels, path)`, `download_levels_csv(spectrum, path)`, `assign(states, levels)`, `fetch_lines(spectrum, wmin_nm, wmax_nm, backend="auto")`, `read_lines(path)`, `upper_levels(lines, levels)`; классы `Level`, `Line`.

## `pydbsr.reference`
`read_xlsx(path)` → `ReferenceData` (`levels`, `sigma[(i,j)] = (E_эВ, σ_см²)`, `sheet`, `transitions(sheet)`, `label(no)`).

## `pydbsr.io`
Разбор конфигураций (`parse_config`, `to_dbsr_conf`, `config_parity`), файлы DBSR (`CFile`, `read_j`, `write_state`, `TargetJJ`, `write_par`, `Grid`, `read_knot`, `modify_knot`), `partial_waves(nelc, jmax)`, `radial_functions(bsw, knot)` → `RadialFunction(n, kappa, energy, r, P, Q)`.

## `pydbsr.outer` (низкий уровень)
`read_h(paths)` → `HData` (блоки `HBlock`: `two_j, parity, nch, l, kappa, target, cf, poles, w`); `kmatrix(blk, hd, etot, r_match=None, step=None, symmetrize=True, relativistic=True)`; `t_matrix(K)`; `propagate_logderiv(...)`; `closed_logderiv(l, kappa2, z, r)`.

## `pydbsr.coulomb`
`coulomb_fg(l, eta, rho)` → F, G, F′, G′; `backend()`.

## Запуск программ
`db.run(program, args, cwd, threads=None, log=None, timeout=None, mpi=None)` → `RunResult`; `db.run_partial_waves(...)`; `db.bin_dir()`, `db.available_programs()`; исключение `db.DBSRError`.

## Командная строка
`pydbsr info | bin | run PROG ARGS | nist "Xe II" [--csv f] | lines "Xe III" wmin wmax [--csv f]`.
