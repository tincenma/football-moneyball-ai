# Фактические столбцы предоставленных архивов

Схемы прочитаны из CSV. Полная статистика по каждому столбцу находится в `reports/data_profiling/*_columns.csv`.

## fifa_male: male_players (legacy).csv

Столбцов: **110**.

```text
player_id, player_url, fifa_version, fifa_update, fifa_update_date, short_name
long_name, player_positions, overall, potential, value_eur, wage_eur
age, dob, height_cm, weight_kg, league_id, league_name
league_level, club_team_id, club_name, club_position, club_jersey_number, club_loaned_from
club_joined_date, club_contract_valid_until_year, nationality_id, nationality_name, nation_team_id, nation_position
nation_jersey_number, preferred_foot, weak_foot, skill_moves, international_reputation, work_rate
body_type, real_face, release_clause_eur, player_tags, player_traits, pace
shooting, passing, dribbling, defending, physic, attacking_crossing
attacking_finishing, attacking_heading_accuracy, attacking_short_passing, attacking_volleys, skill_dribbling, skill_curve
skill_fk_accuracy, skill_long_passing, skill_ball_control, movement_acceleration, movement_sprint_speed, movement_agility
movement_reactions, movement_balance, power_shot_power, power_jumping, power_stamina, power_strength
power_long_shots, mentality_aggression, mentality_interceptions, mentality_positioning, mentality_vision, mentality_penalties
mentality_composure, defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle, goalkeeping_diving, goalkeeping_handling
goalkeeping_kicking, goalkeeping_positioning, goalkeeping_reflexes, goalkeeping_speed, ls, st
rs, lw, lf, cf, rf, rw
lam, cam, ram, lm, lcm, cm
rcm, rm, lwb, ldm, cdm, rdm
rwb, lb, lcb, cb, rcb, rb
gk, player_face_url
```

## fifa_male: male_teams.csv

Столбцов: **54**.

```text
team_id, team_url, fifa_version, fifa_update, fifa_update_date, team_name
league_id, league_name, league_level, nationality_id, nationality_name, overall
attack, midfield, defence, coach_id, home_stadium, rival_team
international_prestige, domestic_prestige, transfer_budget_eur, club_worth_eur, starting_xi_average_age, whole_team_average_age
captain, short_free_kick, long_free_kick, left_short_free_kick, right_short_free_kick, penalties
left_corner, right_corner, def_style, def_team_width, def_team_depth, def_defence_pressure
def_defence_aggression, def_defence_width, def_defence_defender_line, off_style, off_build_up_play, off_chance_creation
off_team_width, off_players_in_box, off_corners, off_free_kicks, build_up_play_speed, build_up_play_dribbling
build_up_play_passing, build_up_play_positioning, chance_creation_passing, chance_creation_crossing, chance_creation_shooting, chance_creation_positioning
```

## fifa_male: male_players.csv

Столбцов: **110**.

Схема полностью совпадает с **fifa_male: male_players (legacy).csv** выше.

## fifa_male: male_coaches.csv

Столбцов: **8**.

```text
coach_id, coach_url, short_name, long_name, dob, nationality_id
nationality_name, face_url
```

## fifa_female: female_coaches.csv

Столбцов: **8**.

Схема полностью совпадает с **fifa_male: male_coaches.csv** выше.

## fifa_female: female_players (legacy).csv

Столбцов: **110**.

Схема полностью совпадает с **fifa_male: male_players (legacy).csv** выше.

## fifa_female: female_players.csv

Столбцов: **110**.

Схема полностью совпадает с **fifa_male: male_players (legacy).csv** выше.

## fifa_female: female_teams.csv

Столбцов: **54**.

Схема полностью совпадает с **fifa_male: male_teams.csv** выше.

## transfermarkt: player_injuries.csv

Столбцов: **7**.

```text
player_id, season_name, injury_reason, from_date, end_date, days_missed
games_missed
```

## transfermarkt: player_latest_market_value.csv

Столбцов: **3**.

```text
player_id, date_unix, value
```

## transfermarkt: player_market_value.csv

Столбцов: **3**.

Схема полностью совпадает с **transfermarkt: player_latest_market_value.csv** выше.

## transfermarkt: player_national_performances.csv

Столбцов: **9**.

```text
player_id, team_id, matches, goals, shirt_number, debut
coach_id, debut_game_id, career_state
```

## transfermarkt: player_performances.csv

Столбцов: **20**.

```text
player_id, season_name, competition_id, competition_name, team_id, team_name
nb_in_group, nb_on_pitch, goals, assists, own_goals, subed_in
subed_out, yellow_cards, second_yellow_cards, direct_red_cards, penalty_goals, minutes_played
goals_conceded, clean_sheets
```

## transfermarkt: player_profiles.csv

Столбцов: **34**.

```text
player_id, player_slug, player_name, player_image_url, name_in_home_country, date_of_birth
place_of_birth, country_of_birth, height, citizenship, is_eu, position
main_position, foot, current_club_id, current_club_name, joined, contract_expires
outfitter, social_media_url, player_agent_id, player_agent_name, contract_option, date_of_last_contract_extension
on_loan_from_club_id, on_loan_from_club_name, contract_there_expires, second_club_url, second_club_name, third_club_url
third_club_name, fourth_club_url, fourth_club_name, date_of_death
```

## transfermarkt: player_teammates_played_with.csv

Столбцов: **6**.

```text
player_id, teammate_player_id, teammate_player_name, ppg_played_with, joint_goal_participation, minutes_played_with
```

## transfermarkt: team_children.csv

Столбцов: **5**.

```text
parent_team_id, parent_team_name, child_team_id, child_team_name, _last_modified_at
```

## transfermarkt: team_competitions_seasons.csv

Столбцов: **7**.

```text
club_id, team_name, season_id, competition_name, competition_id, club_division
_last_modified_at
```

## transfermarkt: team_details.csv

Столбцов: **12**.

```text
club_id, club_slug, club_name, logo_url, country_name, season_id
competition_id, competition_slug, competition_name, club_division, source_url, _last_modified_at
```

## transfermarkt: transfer_history.csv

Столбцов: **10**.

```text
player_id, season_name, transfer_date, from_team_id, from_team_name, to_team_id
to_team_name, transfer_type, value_at_transfer, transfer_fee
```

## Смысл и единица наблюдения

| Таблица | Одна строка и доступные данные |
| --- | --- |
| FIFA players | Игрок × версия × обновление; биография, клуб/лига, позиции, атрибуты FIFA, игровые цены и зарплата, позиционные рейтинги. |
| FIFA teams | Команда × версия × обновление; рейтинги, тактика, тренер, средний возраст, бюджет и стоимость клуба. |
| FIFA coaches | Тренер; ID, имя, дата рождения, гражданство, ссылки. |
| TM player_profiles | Текущий профиль; имена/алиасы, дата рождения, рост, гражданство, позиция, нога, клуб, контракт, агент. Историчность текущих полей не подтверждена. |
| TM player_market_value | Игрок × дата оценки; value. date_unix фактически содержит ISO-даты, а не Unix timestamp. |
| TM player_latest_market_value | Одна последняя оценка игрока; производная от истории, не отдельный исторический источник. |
| TM player_performances | Игрок × сезон × соревнование × команда; появления, голы, ассисты, минуты, карточки и вратарские показатели. Это не матчевые события. |
| TM player_injuries | Запись о травме/отсутствии; причина, начало, окончание, дни и матчи. |
| TM player_national_performances | Игрок × сборная; накопленные матчи/голы, номер и статус. Вся колонка debut пустая. |
| TM player_teammates_played_with | Пара игрок–партнёр; совместные минуты, участия в голах, очки. Временного окна нет. |
| TM transfer_history | Событие перехода/аренды/возврата; дата, сезон, команды, тип, оценка и сумма. Профиль есть не у каждого player_id. |
| TM team_competitions_seasons | Клуб × сезон × соревнование; история участия, но много точных повторов. |
| TM team_details | Одна строка на клуб, с одним season_id; это не полная панель клуб–сезон. |
| TM team_children | Связь основной/дочерней команды; встречаются ссылки клуба на самого себя. |

xG, xA, прогрессивных передач/продвижений, прессинга, точности передач и событийных отборов/перехватов на 90 минут в этих схемах нет.
