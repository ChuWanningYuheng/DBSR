# Документация pydbsr

pydbsr — Python-обёртка над релятивистским методом R-матрицы на B-сплайнах
DBSR (O. Zatsarinny, K. Bartschat) для сечений возбуждения атомов и ионов
электронным ударом. Библиотека собирает программы DBSR3, исправляет в них
ошибки, из которых складывается «конструктор» мишень → рассеяние → сечения,
и добавляет то, чего в DBSR нет: уровни NIST, внешнюю область на Python,
параллельный потоковый запуск, проверки физики.

| Документ | Что внутри |
|---|---|
| [install.md](install.md) | Установка: ноутбук/рабочая станция, сервер без root, Jupyter, conda, проверка установки |
| [quickstart.md](quickstart.md) | Первый расчёт за 5 минут и полный расчёт e + Xe⁺ из Jupyter |
| [target.md](target.md) | Мишень: конфигурации, орбитали, КВ, корреляционные орбитали, NIST |
| [scattering.md](scattering.md) | Внутренняя область: парциальные волны, ресурсы, параллельность, перезапуск |
| [outer.md](outer.md) | Внешняя область: Ω, σ, скорости, ФРЭЭ, сходимость |
| [physics.md](physics.md) | **На что смотреть с точки зрения физики** — чек-лист перед публикацией |
| [nist.md](nist.md) | Уровни и линии NIST, отождествление состояний, спектроскопические задачи |
| [validation.md](validation.md) | Стресс-тест и что он проверяет |
| [troubleshooting.md](troubleshooting.md) | Ошибки DBSR и что с ними делать |
| [api.md](api.md) | Справочник по API |
| [dbsr_programs.md](dbsr_programs.md) | Цепочка программ DBSR, файлы, прямой вызов любой программы |

## Схема расчёта

```
Ion("Xe", 1)
   │
Target ── add(конфигурации) ── compute()        dbsr_hf (Дирак–Фок, LS → jj), dbsr_mchf, jcfile
   │        состояния мишени (name.c + name.bsw), nist.assign() → пороги NIST
   │
Scattering ── prepare → run_prep → run_conf     dbsr_prep3, dbsr_conf3
   │        run_streamed(cores, mem, scratch)   dbsr_breit3 → dbsr_mat3 → dbsr_hd3  (по волнам)
   │        h.nnn — R-матрица каждой парциальной волны
   │
OuterRegion ── collision_strengths(E)           K → S → T → Ω(i→j)  (Python, параллельно)
   │
CollisionStrengths ── sigma, upsilon, rate, rate_eedf(maxwell/bugrova), save/load
```

## Кратко о главном

* **Состояния одной чётности — в одном КВ-расчёте** (`tg.add([...])`). Иначе они
  взаимодействуют в уравнениях сильной связи, и сечения искажаются на десятки
  процентов (см. [physics.md](physics.md#1-согласованность-мишени)).
* **Сначала J = 0**, стресс-тест (`tools/stress_test.py`), потом полный расчёт.
* **Сходимость по J** проверяйте всегда: `cs.omega_pw`.
* Прерванный расчёт продолжается повторным вызовом — готовые волны пропускаются.
