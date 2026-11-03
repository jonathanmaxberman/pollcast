import numpy as np
import pandas as pd
import pytest

from pollcast.simulate import SimulateConfig, simulate


def inputs(means, sds, challengers=None):
    return pd.DataFrame({"mean": means, "sd": sds, "challenger": challengers or ["DEM"] * len(means)})


def test_even_race_is_a_coin_flip_and_sure_things_are_sure():
    r = simulate(inputs([0.0, 40.0, -40.0], [5, 5, 5]), SimulateConfig(n_sims=40000))
    assert r["p_challenger"][0] == pytest.approx(0.5, abs=0.01)
    assert r["p_challenger"][1] > 0.99 and r["p_challenger"][2] < 0.01


def test_national_swing_makes_states_miss_together():
    sim = SimulateConfig(n_sims=40000, seats_not_up={"DEM": 0, "REP": 0}, rep_control_seats=2)
    tied = inputs([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with_swing = simulate(tied, sim)["rep_seats_hist"]
    no_swing = simulate(tied, SimulateConfig(**{**sim.__dict__, "national_sd": 0.0}))["rep_seats_hist"]
    # Independent coin flips sweep 3-0 a quarter of the time; correlated ones far more.
    sweep = lambda h: h.get(0, 0) + h.get(3, 0)
    assert sweep(no_swing) == pytest.approx(0.25, abs=0.02)
    assert sweep(with_swing) > 0.6


def test_independent_wins_count_as_neither_party():
    sim = SimulateConfig(n_sims=2000, seats_not_up={"DEM": 49, "REP": 50})
    r = simulate(inputs([30.0], [1.0], ["IND"]), sim)
    assert r["median_seats"] == {"REP": 50.0, "DEM": 49.0, "IND": 1.0}
    assert r["p_rep_control"] == pytest.approx(1.0)
