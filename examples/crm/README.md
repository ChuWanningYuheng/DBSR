# Ноутбуки: атомные данные для CRM (Xe II, Ba I, Ba II)

Только входные данные для столкновительно-излучательной модели: A и ветвления, сечения
линий, константы скоростей. Сама модель, подгонка ne/Te и плотности бария сюда не входят.
Выбор линий и источники всех величин — [docs/crm/lines_final.md](../../docs/crm/lines_final.md).

| Ноутбук | Что считает | Время | Выход |
|---|---|---|---|
| [01_xe2_A_values](01_xe2_A_values.ipynb) | мишень Xe⁺ как у Ванга; A, gf, S всех E1-ветвей верхних уровней 29, 39, 41, 44, 51; ветвления; сравнение с NIST | см. lines_final.md, п. 4 | `xe2_wang/xe2_A_branching.xlsx/.csv`, `transitions_E1.csv` |
| [02_xe2_line_cross_sections](02_xe2_line_cross_sections.ipynb) | σ линий = σ(Ванг) × BR (из 01) для всех начальных уровней у Ванга; ⟨σv⟩(Te) | минуты | `xe2_line_sigma.xlsx`, `xe2_k_excitation.xlsx`, `xe2_k_line.xlsx` (+ .csv) |
| [03_ba2](03_ba2.ipynb) | e + Ba⁺: σ всех возбуждений (17 состояний), σ линий 455.4 и 493.4 (BR из NIST), ⟨σv⟩(Te); проверки мишени | часы (сервер) | `ba2/sigma_ba2.xlsx`, `rates_ba2.xlsx`, `ba2_line_sigma.xlsx`, `ba2_k_line.xlsx` |
| [04_ba1](04_ba1.ipynb) | e + Ba: σ линии 553.5 из 6s² (Fursa 1999 — готово; DBSR — проверка) и из 6s5d (DBSR), ⟨σv⟩(Te) | сутки (сервер, оценка) | `ba1/sigma_ba1.xlsx`, `rates_ba1.xlsx`, `ba1_line_sigma.xlsx`, `ba1_k_line.xlsx` |

Порядок: 01 → 02; 03 и 04 независимы.

* Выбранные линии заданы в одном месте — `nbtools.py` (`XE2_LINES`, `BA_LINES`).
* Сетка Te для констант скоростей — переменная `TE` в первой ячейке.
* Формат выходных xlsx — как у таблицы Ванга: лист «NIST Level Table» и листы с парами
  столбцов Energy(eV) / Sigma(1E-16 cm^2), над столбцом σ — метка `i->j` (номера уровней).
  Файлы читаются обратно `pydbsr.reference.read_xlsx`. Константы скоростей: строки — Te,
  столбцы — переходы (xlsx и csv).
* Тяжёлые расчёты идут **отдельным процессом** `run_model.py` (`nb.Job(...).start()`): их не
  прерывает закрытие браузера или ядра; повторный `start()` продолжает с места остановки.
* Модели мишеней — [models.py](models.py) (`xe2_wang`, `ba2`, `ba1`), без поляризации остова:
  проверяйте энергии и A (ячейки «проверка» в ноутбуках), прежде чем доверять σ.
* [line_criteria.py](line_criteria.py) — числа для выбора линий Xe II (видимость, бленды, порог,
  наклон k(Te), вклад метастабилей, тушение, полнота A в NIST).

Из командной строки:
```bash
python examples/crm/run_model.py --model xe2_wang --workdir RUNS/xe2_wang --levels CrossSectionsIon_3.xlsx \
       --stage transitions --uppers 29,39,41,44,51
python examples/crm/run_model.py --model ba2 --workdir RUNS/ba2 --stage all --jmax 20 --cores 32 --mem 100 \
       --te 0.3,0.5,1,2,5,10
python examples/crm/line_criteria.py --wang CrossSectionsIon_3.xlsx --spectrum A25_second.xlsx \
       --wmin 541.647 --wmax 565 --extra 575.103,610.143,461.550,497.271 --out criteria.csv
```

Ноутбуки генерируются из [tools/make_atomic_data_notebooks.py](../../tools/make_atomic_data_notebooks.py).
Если pydbsr установлен в режиме разработки (`pip install -e`), задайте `DBSR_BIN` (папка
`.../site-packages/pydbsr/bin` с собранными программами).
