# Research Problem / Исследовательская проблема

## Research problem (English)

Football clubs with limited financial resources need to identify players who can strengthen their squads without excessive transfer spending. Conventional selection methods relying primarily on overall ratings, popularity, and listed value may overlook less visible players with comparable playing characteristics.

This project investigates whether machine learning can identify **potentially cost-effective central midfielders** whose playing profiles resemble those of more expensive players, and whether the resulting shortlist provides measurable benefits over a simple rating-based baseline.

## Исследовательская проблема (русский)

Футбольные клубы с ограниченными финансовыми ресурсами испытывают трудности при поиске игроков, способных усилить состав без значительных трансферных затрат. Подходы, основанные преимущественно на общем рейтинге, известности и указанной стоимости футболистов, могут упускать менее известных игроков с сопоставимыми игровыми характеристиками.

Проект исследует, могут ли методы машинного обучения находить **потенциально экономически привлекательных центральных полузащитников**, схожих по игровому профилю с более дорогими игроками, и обеспечивают ли полученные рекомендации измеримые преимущества перед простым отбором по общему рейтингу.

## Main research question

> To what extent can machine learning methods identify potentially undervalued central midfielders with playing characteristics comparable to more expensive players, and how do these recommendations compare with traditional rating-based selection methods?

## Hypotheses

- **H0 (null):** The ML-based recommendation method does not outperform the predefined rating-based baseline on an independently justified evaluation criterion.
- **H1 (alternative):** The ML-based recommendation method outperforms that baseline on the same criterion.

**Note:** The evaluation criterion must be specified *before* testing. High similarity on the very features used for ranking is not sufficient evidence to accept H1.

## Working operational definition

A *potentially undervalued candidate* is a central midfielder whose recorded cost is lower than that of a chosen reference player while the measured playing profile is sufficiently similar. This operational definition identifies **candidates for further review**, not confirmed market mispricing.

## Intended users and output

- **Users:** scouting analysts and sporting directors at budget-constrained football clubs.
- **Decision:** which affordable central midfielders deserve further assessment for a possible transfer.
- **Output:** a ranked shortlist, transparent similarity rationale, cost/age filters, comparison with a baseline, and explicit warnings about data limitations.
