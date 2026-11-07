import numpy as np
import pandas as pd
import pytest

from src.evaluation import dated_value_outcomes, paired_list_outcomes


def players():
    return pd.DataFrame({"fifa_id":["a","b","c"],"tm_id":["1","2","3"],"as_of_date":"2022-09-26",
                         "market_value_date":"2022-09-01","tm_market_value":[100.,200.,300.]})


def history(rows):
    return pd.DataFrame(rows,columns=["player_id","date_unix","value"])


def test_boundaries_duplicates_and_no_future_peeking():
    h=history([("1","2023-05-29",110),("1","2023-05-29",110), # exactly 120 days before target
               ("1","2023-09-27",999), # after horizon
               ("2","2023-09-26",220),("3","2023-05-28",330)])
    p=players();saved=p.copy(deep=True);result=dated_value_outcomes(p,h).set_index("fifa_id")
    assert result.loc['a','future_value']==110 and result.loc['a','future_quote_lag_days']==120
    assert result.loc['b','future_value']==220
    assert not result.loc['c','outcome_observed'] and np.isnan(result.loc['c','value_change'])
    assert np.isclose(result.loc['a','log_value_ratio'],np.log(1.1))
    pd.testing.assert_frame_equal(p,saved)


def test_conflicts_zero_and_invalid_not_replaced_by_old_positive():
    h=history([("1","2023-06-01",120),("1","2023-06-01",130),
               ("2","2023-06-01",250),("2","2023-08-01",0),
               ("3","bad",400),("3","2023-08-01",-1)])
    result=dated_value_outcomes(players(),h).set_index('fifa_id')
    assert result.loc['a','outcome_status']=='conflicting_quotes_in_window'
    assert result.loc['b','outcome_status']=='latest_quote_zero'
    assert result.loc['c','outcome_status']=='no_valid_quote_in_window'
    assert result.value_change.isna().all()


def test_numeric_text_values_are_normalized_and_empty_history_is_missing():
    p=players();p['tm_market_value']=p.tm_market_value.astype(str)
    result=dated_value_outcomes(p,history([('1','2023-08-01','120')]))
    assert np.isclose(result.value_change_fraction.iloc[0],.2)
    empty=dated_value_outcomes(p,history([]))
    assert not empty.outcome_observed.any() and empty.log_value_ratio.isna().all()


@pytest.mark.parametrize('case',['future_base','duplicate_tm','wrong_cutoff','zero_base','overlapping_window'])
def test_invalid_historical_protocol(case):
    p=players();kw={}
    if case=='future_base': p.loc[0,'market_value_date']='2023-01-01'
    elif case=='duplicate_tm': p.loc[0,'tm_id']='2'
    elif case=='wrong_cutoff': p.loc[0,'as_of_date']='2023-09-26'
    elif case=='zero_base': p.loc[0,'tm_market_value']=0
    else: kw['max_lag_days']=365
    with pytest.raises(ValueError): dated_value_outcomes(p,history([]),**kw)


def test_paired_missing_outcome_not_zero_or_replaced():
    lists=pd.DataFrame({'reference_id':['r']*4,'method':['baseline','baseline','knn','knn'],'fifa_id':['a','b','b','c']})
    o=dated_value_outcomes(players(),history([('1','2023-08-01',120),('2','2023-08-01',220)]))
    result=paired_list_outcomes(lists,o,top_k=2)
    assert result.baseline_observed_slots.iloc[0]==2 and result.knn_observed_slots.iloc[0]==1
    assert not result.complete_both.iloc[0] and np.isnan(result.knn_minus_baseline_log_ratio.iloc[0])
    h=history([('1','2023-08-01',120),('2','2023-08-01',220),('3','2023-08-01',450)])
    complete=paired_list_outcomes(lists,dated_value_outcomes(players(),h),top_k=2)
    assert complete.complete_both.iloc[0]
    assert np.isclose(complete.knn_minus_baseline_log_ratio.iloc[0],(np.log(1.5)-np.log(1.2))/2)
    duplicate=pd.concat([lists,lists.iloc[:1]])
    with pytest.raises(ValueError): paired_list_outcomes(duplicate,o,top_k=2)
    with pytest.raises(ValueError): paired_list_outcomes(lists.iloc[:-1],o,top_k=2)
