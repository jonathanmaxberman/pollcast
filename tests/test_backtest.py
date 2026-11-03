import numpy as np
import pandas as pd
import pytest

from pollcast.backtest import ErrorSizes, estimate_error_sizes, priors_for, score
from pollcast.sources._538_format import is_special_senate


def test_special_elections_follow_the_class_rotation():
    assert not is_special_senate(2020, "Class II")  # 2020's regular class
    assert is_special_senate(2020, "Class III")  # Georgia and Arizona specials
    assert is_special_senate(2024, "Class II")  # Nebraska special
    assert not is_special_senate(2026, "")  # blank seat names count as regular


def test_score_rewards_confident_correct_calls():
    races = pd.DataFrame({"p": [0.9, 0.2], "actual": [1, 0], "margin": [5.0, -3.0], "mean": [4.0, -1.0]})
    s = score(races)
    assert s["log_loss"] == pytest.approx(-(np.log(0.9) + np.log(0.8)) / 2)
    assert s["brier"] == pytest.approx((0.01 + 0.04) / 2)
    assert s["mae"] == pytest.approx(1.5) and s["bias"] == pytest.approx(-0.5)
    assert s["correct"] == 1.0


def test_priors_use_the_same_seat_six_years_earlier():
    results = pd.DataFrame({
        "cycle": [2012, 2016, 2018], "state": ["Ohio"] * 3, "special": [False] * 3,
        "nonrep_pct": [50.7, 37.2, 53.4], "rep_pct": [44.7, 58.0, 46.6],
    })
    p = priors_for(results, 2018)
    assert p.iloc[0]["label"] == "Election 2012"
    assert p.iloc[0]["DEM"] - p.iloc[0]["REP"] == pytest.approx(6.0)


def test_error_sizes_recover_a_known_national_miss():
    rng = np.random.default_rng(0)
    rows = []
    for cycle, swing in ((2018, 4.0), (2020, -4.0), (2022, 4.0), (2024, -4.0)):
        for _ in range(40):
            mean = rng.normal(0, 10)
            rows.append({"cycle": cycle, "lead_days": 0, "basis": "polls", "mean": mean, "se": 0.0,
                         "effective_n": 1e9, "margin": mean + swing + rng.normal(0, 3)})
    sizes = estimate_error_sizes(pd.DataFrame(rows), ErrorSizes(1, 1, 1, 1))
    assert sizes.national_sd == pytest.approx(4.0, abs=0.6)
    assert sizes.state_sd == pytest.approx(3.0, abs=0.6)


def test_historical_leans_are_relative_to_peers_and_exclude_later_years():
    from pollcast.sources.fivethirtyeight_ratings import historical_leans

    hist = pd.DataFrame({
        "year": [2016, 2016, 2016, 2016, 2020], "race": ["a", "a", "b", "b", "c"],
        "pollster_rating_id": ["X", "Y", "X", "Y", "X"], "days_out": [5, 5, 5, 5, 5],
        # Every poll missed toward Democrats; X by 2 points more than Y.
        "bias": [6.0, 4.0, 6.0, 4.0, -20.0],
    })
    lean = historical_leans(hist, 2018, prior_polls=0.0)
    assert lean["X"] == pytest.approx(1.0) and lean["Y"] == pytest.approx(-1.0)  # the shared miss is removed
    assert historical_leans(hist, 2016) == {}  # nothing before 2016


def test_historical_bias_is_against_results_and_covers_sponsors():
    from pollcast.sources.fivethirtyeight_ratings import historical_bias

    hist = pd.DataFrame({
        "year": [2016, 2016, 2016], "race": ["a", "a", "a"], "pollster_rating_id": ["X", "Y", "Z"],
        "partisan": ["", "", "R"], "days_out": [5, 5, 5], "bias": [6.0, 4.0, -2.0],
    })
    b = historical_bias(hist, 2018, prior_polls=0.0)
    assert b["X"] == pytest.approx(6.0) and b["Y"] == pytest.approx(4.0)  # the shared miss is kept
    assert b["*|REP"] == pytest.approx(-2.0) and "*|DEM" not in b
