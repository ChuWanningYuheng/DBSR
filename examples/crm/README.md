# Ноутбуки CRM: Xe II, Ba I, Ba II

Атомные данные для столкновительно-излучательной модели плазмы катода и сама модель.
План и обоснование — [docs/crm_plan.md](../../docs/crm_plan.md), список линий —
[docs/crm/lines.md](../../docs/crm/lines.md).

| Ноутбук | Что считает | Время (оценка) | Результат |
|---|---|---|---|
| [00_line_list](00_line_list.ipynb) | отождествление линий в спектре, классы данных | секунды | `lines.csv`, `lines.md` |
| [01_xe2_A_values](01_xe2_A_values.ipynb) | мишень Xe⁺ как у Ванга; A (E1, E2) для всех пар; ветвления линий; сравнение с NIST | часы (структура, без рассеяния) | `transitions_E1.csv`, `xe2_branching.csv` |
| [02_xe2_line_cross_sections](02_xe2_line_cross_sections.ipynb) | σ линий (Ванг × ветвление) для 452.4, 659.5, 669.4 и всех линий класса A/B; k(Te); предварительная CRM и чувствительность к ne, Te | минуты (после 01) | `xe2_line_sigma.xlsx`, `crm_xe2_wang_prelim.json` |
| [03_xe2_extended_scattering](03_xe2_extended_scattering.ipynb) | e + Xe⁺ с 6d, 7s, 4f, 7p: линии класса C, каскады, метастабиль↔метастабиль | дни (сервер) | `omega_J*.npz`, `crm_xe2_ext.json` |
| [04_ba2](04_ba2.ipynb) | e + Ba⁺: энергии, A, времена жизни 5d, уровни Ba I как (N+1)-состояния, σ, CRM 455.4/493.4 | часы | `crm_ba2.json` |
| [05_ba1](05_ba1.ipynb) | e + Ba: f(553.5), σ против Fursa и Chen–Gallagher, σ из 6s5d, CRM против корональной | сутки (сервер) | `crm_ba1.json` |
| [06_crm_assembly](06_crm_assembly.ipynb) | подгонка ne, Te по линиям Xe II, затем n(Ba), n(Ba⁺); пленение | минуты | — |

Порядок: 00 → 01 → 02 (это уже даёт сечения линий Xe II по данным Ванга) → 04, 05, 03
(можно параллельно на сервере) → 06.

## Как устроено

* Первая ячейка каждого ноутбука — пути и ресурсы (`REPO`, `DATA`, `RUNS`, `CORES`, `MEM`).
* Тяжёлые расчёты идут **отдельным процессом** `run_model.py` (`nb.Job(...).start()`): их не
  останавливает закрытие браузера или перезапуск ядра. `job.status()` показывает ход,
  повторный `start()` продолжает с места остановки (готовые стадии и волны пропускаются).
* Модели мишеней — [models.py](models.py) (`xe2_wang`, `xe2_ext`, `ba2`, `ba1`). Это стартовые
  модели без поляризации остова: проверяйте энергии и A до того, как доверять σ.
* Библиотека: `pydbsr.transitions` (A-коэффициенты для пар состояний мишени через `dbsr_mult3` +
  `dbsr_dmat3`), `pydbsr.crm` (k(Te) из σ(E), детальный баланс, данные CRM, стационарное решение,
  чувствительность, оптическая толщина и фактор выхода).

Из командной строки, без ноутбука:
```bash
python examples/crm/run_model.py --model ba2 --workdir RUNS/ba2 --stage all --jmax 20 --cores 32 --mem 100
python examples/crm/run_model.py --model xe2_wang --workdir RUNS/xe2_wang --levels CrossSectionsIon_3.xlsx --stage transitions
python examples/crm/line_list.py --wang CrossSectionsIon_3.xlsx --spectrum A25_second.xlsx --out docs/crm/lines
```

Ноутбуки генерируются из [tools/make_crm_notebooks.py](../../tools/make_crm_notebooks.py):
правьте генератор и запускайте `python tools/make_crm_notebooks.py`.
