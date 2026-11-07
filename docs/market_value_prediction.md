# Model 3 — Market Value Prediction

Исторический срез: 2022-09-26; 2886 CM, 2308 train / 578 test. 27 входов Feature Set v1.

## Модель и оценка

Random Forest: 160 деревьев, параметры {'min_samples_leaf': 2, 'max_features': 0.7}; target log1p стоимости. Imputation/encoding — только внутри fit folds. Train-only 3-fold CV; Ridge alpha=1.0.

Test RF: MAE=1,333,985, RMSE=3,203,745 исходных единиц; R²(raw)=0.811; MAE(log)=0.487; R²(log)=0.839; медианный multiplicative error=1.47×.

MAE(log): train median=1.274, Ridge=0.509, RF=0.487. Paired test bootstrap CI для Ridge minus RF: [-0.0087, 0.0517]. Условные интервалы не являются новой независимой проверкой.

По исходной шкале Ridge: MAE=1,315,547, RMSE=2,950,272; RF: MAE=1,333,985, RMSE=3,203,745. Преимущество по одной шкале не означает универсального превосходства. Интервал разности log-MAE включает ноль: убедительного преимущества RF над Ridge не установлено.

## Прогнозы и gap

Для всех 2886 игроков сохранены прогнозы вне собственного fit: 2308 nested OOF (5×3 CV) и 578 test от train-only модели. Test positive gaps: 282; train OOF: 1130.

Среднее пересечение test TOP-10 при 10 fit на 80% train: 69.0%. Средняя частота положительного знака для основного TOP-10: 90.0%. Это чувствительность, не прогнозный интервал.

ValueGap = PredictedValue − MarketValue. Большой положительный gap — исследовательский скаутинговый сигнал. Абсолютные и относительные gaps имеют разные перекосы; списки OOF и test ранжируются отдельно. Цена не добавлена в KNN/K-Means.

## Артефакты

Notebook: `notebooks/07_market_value_prediction.ipynb`; модуль: `src/valuation.py`; тесты: `tests/test_valuation.py`. Модель, per-player gaps, fold ID audit и manifest находятся в gitignored `data/processed/valuation/`. Агрегаты и пять PNG/SVG фигур: `reports/valuation/`.

## Пределы вывода

Прогноз не объективная справедливая цена. Неизвестны контракты, травмы, спрос и другие факторы. Историческая стоимость не цена трансфера. Валюта архива не подтверждена; значения в исходных единицах. Оценки могут отставать от среза на 365 дней. Log-inversion не даёт арифметическое условное среднее и может недооценивать дорогих игроков. Test уже изучался в EDA; новый сезон/независимая выборка нужны для подтверждающих выводов. Streamlit starter не интегрирован с этим модулем.
