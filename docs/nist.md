# NIST: уровни, линии, спектроскопия

## Уровни энергии

```python
levels = db.nist.fetch_levels("Xe II")          # или db.Ion("Xe", 1); кэшируется в PYDBSR_CACHE
for lv in levels[:5]:
    print(lv.no, lv.config, lv.term, lv.two_j, lv.energy_cm)
db.nist.save_levels_csv(levels, "xe2_levels.csv")
levels = db.nist.read_levels("xe2_levels.csv")  # без интернета (кластер)
```
Командная строка: `pydbsr nist "Xe II" --csv xe2_levels.csv`.

Если NIST вернул HTML вместо таблицы (профилактика, ограничения доступа),
скачайте таблицу вручную на
<https://physics.nist.gov/PhysRefData/ASD/levels_form.html> (формат *Tab-delimited*)
и прочитайте её `read_levels(path)`.

`lv.no` — номер уровня по возрастанию энергии (как «No.» в таблицах Wang 2019).

## Отождествление состояний

```python
db.nist.assign(tg.states, levels)       # проставляет exp_energy_cm, nist_label, nist_no
```
См. [target.md](target.md#отождествление-с-nist). Проверяйте таблицу `tg.table()`:
у каждого нужного состояния должна быть энергия NIST, разница с расчётом —
разумной (< 0.5–1 эВ).

## Линии и их верхние уровни (спектроскопия плазмы)

```python
lines = db.nist.fetch_lines("Xe III", 474, 479)         # ASDCache, если установлен, иначе напрямую
for ln in lines:
    print(ln.wavelength_nm, ln.aki, ln.conf_i, ln.term_i, ln.j_i, "->", ln.conf_k, ln.term_k, ln.j_k)
lv3 = db.nist.fetch_levels("Xe III")
for ln, up in db.nist.upper_levels(lines, lv3):
    print(ln.wavelength_nm, "upper level No.", up.no if up else None)
```
Командная строка: `pydbsr lines "Xe III" 474 479 --csv lines.csv`.

Дальше: включить верхние уровни этих линий (и их каскадных предшественников)
в модель мишени, посчитать σ(основное → верхний) и скорости для ФРЭЭ плазмы.

## Сравнение с опубликованными сечениями

`pydbsr.reference.read_xlsx` читает таблицу-приложение Wang et al (2019)
(лист «NIST Level Table» + листы с парами колонок энергия/сечение, метка
`i->j` над колонкой сечения):
```python
from pydbsr import reference
ref = reference.read_xlsx("CrossSectionsIon.xlsx")
e, s = ref.sigma[(1, 4)]          # эВ, см²
ref.transitions("gs->6s")         # переходы листа
```
