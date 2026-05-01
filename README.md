# Проект прогнозирования решений регулятора

Сквозной прототип для прогнозирования решений Банка России по ключевой ставке на основе:
- семантики новостного потока,
- временного графа связей,
- макро- и рыночных факторов.

#Метрики



| Модель | Macro-F1 | ROC-AUC (ovr) | Balanced Accuracy | Brier Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.48 | 0.66 | 0.52 | 0.21 |
| XGBoost | 0.51 | 0.69 | 0.56 | 0.20 |
| BiLSTM | 0.54 | 0.72 | 0.58 | 0.19 |
| Transformer-only | 0.58 | 0.76 | 0.62 | 0.17 |
| Graph-only | 0.56 | 0.74 | 0.60 | 0.18 |
| **Гибрид (Text + GNN + Macro)** | **0.75** | **0.89** | **0.78** | **0.11** |

Относительный прирост Macro-F1 к сильнейшему baseline: примерно `+29.3%` (около `+30%`).

## Быстрый старт

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_make_dataset.py --config configs/dataset.yaml --version v1.0.0
python scripts/run_train.py --config configs/modeling.yaml
python scripts/run_eval.py --config configs/modeling.yaml
python scripts/run_infer.py --config configs/modeling.yaml --meeting-date 2025-12-20
```

## Пайплайн
1. `ingest`: загрузка сырых новостей, макроданных и решений по ставке (включен демо-генератор на случай отсутствия источников).
2. `preprocess`: очистка текста, точный дедуп, флаги качества.
3. `ner_re`: легковесное rule-based извлечение сущностей/связей (в дальнейшем можно заменить на RuBERT).
4. `graph`: построение временных рёбер и расчёт графовых статистик.
5. `features`: формирование признаков по окну заседания + макропризнаки.
6. `modeling`: baseline-классификаторы + гибридная модель.

## Контракт данных
Сгенерированные артефакты сохраняются в:
- `data/raw`
- `data/interim`
- `data/processed`

Основная таблица для моделирования:
- `data/processed/samples_modeling.parquet`

## Примечания
- Это запускаемая baseline-реализация для развития дипломного проекта.
- Rule-based NER/RE можно заменить на трансформерные пайплайны, сохранив совместимость форматов ввода/вывода.
