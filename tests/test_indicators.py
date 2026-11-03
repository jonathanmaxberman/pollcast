import numpy as np
import pandas as pd
import pytest

from pollcast.indicators import INDICATORS


def synthetic_races(k: float, swing_by_cycle=(3.0, -3.0, 2.0, -2.0), n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cycle, swing in zip((2018, 2020, 2022, 2024), swing_by_cycle):
        for _ in range(n):
            mean = rng.normal(0, 12)
            rows.append({"cycle": cycle, "lead_days": 0, "basis": "polls", "mean": mean,
                         "margin": k * mean + swing + rng.normal(0, 3)})
    return pd.DataFrame(rows)


def test_undecided_stretch_recovers_k_and_ignores_cycle_swings():
    ind = INDICATORS["undecided_stretch"]({"prior_sd": 1.0})
    params = ind.fit(synthetic_races(1.15))
    assert params["k"][0] == pytest.approx(1.15, abs=0.03)
    assert ind.effect(params) > 0


def test_shrinkage_pulls_a_weak_effect_toward_no_effect():
    races = synthetic_races(1.15, n=12)
    loose = INDICATORS["undecided_stretch"]({"prior_sd": 1.0}).fit(races)["k"][0]
    tight = INDICATORS["undecided_stretch"]({"prior_sd": 0.01}).fit(races)["k"][0]
    assert abs(tight - 1) < abs(loose - 1)


def test_apply_only_touches_poll_based_races_and_uses_nearest_lead():
    ind = INDICATORS["undecided_stretch"]()
    inputs = pd.DataFrame({"basis": ["polls", "prior"], "mean": [10.0, 10.0]})
    out = ind.apply(inputs, {"k": {0: 1.1, 90: 0.9}}, lead_days=5)
    assert out["mean"].tolist() == [pytest.approx(11.0), 10.0]
    assert inputs["mean"].tolist() == [10.0, 10.0]  # input left unchanged


def test_late_variant_fits_and_applies_only_inside_its_cutoff():
    races = pd.concat([synthetic_races(1.15).assign(lead_days=7),
                       synthetic_races(0.9, seed=1).assign(lead_days=90)])
    ind = INDICATORS["undecided_stretch"]({"prior_sd": 1.0, "max_lead_days": 30})
    params = ind.fit(races)
    assert set(params["k"]) == {7}  # the 90-day rows are left out of the fit
    inputs = pd.DataFrame({"basis": ["polls"], "mean": [10.0]})
    assert ind.apply(inputs, params, lead_days=60)["mean"].iloc[0] == 10.0
    assert ind.apply(inputs, params, lead_days=3)["mean"].iloc[0] > 10.0
    assert "last 30 days" in ind.label()


def test_each_group_defaults_to_its_best_passed_enabled_option(tmp_path):
    import json
    from datetime import date

    from pollcast.config import AverageConfig, Config
    from pollcast.simulate import SimulateConfig, adjustment_groups, scenario_keys

    (tmp_path / "backtests").mkdir()
    (tmp_path / "backtests" / "indicator_params.json").write_text(json.dumps({
        "a": {"accepted": False, "improvement": 0.01, "cycles_improved": "1/4", "direction_as_declared": False,
              "run_id": "r", "params": {"k": {"0": 1.1}}},
        "b": {"accepted": True, "improvement": 0.002, "cycles_improved": "3/4", "direction_as_declared": True,
              "run_id": "r", "params": {"k": {"0": 1.1}}},
        "c": {"accepted": True, "improvement": 0.003, "cycles_improved": "3/4", "direction_as_declared": True,
              "run_id": "r", "params": {"k": {"0": 1.1}}},
    }))
    config = Config(name="t", data_dir=tmp_path, sources={}, average=AverageConfig(), indicators={
        "a": {"module": "undecided_stretch", "enabled": True},
        "b": {"module": "undecided_stretch", "enabled": True},
        "c": {"module": "undecided_stretch", "enabled": False},  # passed better, but turned off
        "d": {"module": "undecided_stretch", "enabled": True},  # never backtested
    })
    [group], left_out = adjustment_groups(config, SimulateConfig(), date(2026, 11, 3))
    assert group["default"] == "b"
    # Only passed variants are choices; failed and untested ones are a record.
    assert [o["key"] for o in group["options"]] == ["off", "b", "c"]
    assert [(o["key"], o["status"]) for o in left_out] == [("a", "failed"), ("d", "untested")]
    assert {s["undecided_stretch"] for s in scenario_keys([group])} == {"off", "b", "c"}


def _rows(cycle, n, rng, lead=0):
    from datetime import date
    return pd.DataFrame({"cycle": cycle, "lead_days": lead, "basis": "polls", "as_of": date(cycle, 11, 1),
                         "state": [f"S{i}" for i in range(n)], "mean": rng.normal(0, 10, n)})


def test_rain_module_recovers_a_planted_effect_and_skips_far_leads():
    from pollcast.calendar import ELECTION_DAYS

    rng = np.random.default_rng(3)
    ind = INDICATORS["election_day_rain"]({"prior_sd": 10.0, "max_lead_days": 7})
    races = pd.concat([_rows(c, 50, rng) for c in (2018, 2020, 2022, 2024)], ignore_index=True)
    ind.rain = {(str(ELECTION_DAYS[c]), s): float(rng.uniform(0, 1)) for c, s in zip(races.cycle, races.state)}
    rain = ind.feature(races)
    races["margin"] = races["mean"] - 2.0 * rain + rng.normal(0, 0.5, len(races))
    beta = ind.fit(races)["beta"]
    assert beta == pytest.approx(-2.0, abs=0.3)
    far = races.assign(lead_days=30)
    assert ind.apply(far, {"beta": beta}, lead_days=30)["mean"].equals(far["mean"])


def test_economy_module_flips_sign_with_the_presidents_party():
    ind = INDICATORS["economy_vs_president"]({"series": "GASREGW"})
    ind.table = pd.DataFrame({"series": ["GASREGW"] * 2, "vintage_date": pd.to_datetime(["2018-11-01", "2022-11-01"]),
                              "change_12m": [0.5, 0.5]})
    rng = np.random.default_rng(0)
    rows = pd.concat([_rows(2018, 2, rng), _rows(2022, 2, rng)], ignore_index=True)
    x = ind.signed_feature(rows)
    assert x[rows.cycle == 2018].tolist() == [1.0, 1.0]  # Republican president: gas up helps the challenger
    assert x[rows.cycle == 2022].tolist() == [-1.0, -1.0]  # Democratic president: it helps the Republican


def test_news_module_is_registered_but_untestable():
    ind = INDICATORS["news_coverage"]({"keyword": "inflation"})
    assert not ind.testable() and "GDELT" in ind.untestable_reason()
    assert ind.label().endswith("(inflation)")


def test_fundamentals_blend_weights_polls_by_precision():
    from pollcast.indicators.fundamentals import SenateFundamentals, last_name

    assert last_name("ANGUS S. KING, JR.") == "KING" and last_name("Sherrod Brown") == "BROWN"
    ind = SenateFundamentals()
    ind.features = lambda rows: pd.DataFrame({"lean": 0.0, "env": 0.0, "inc": 0.0}, index=rows.index)
    params = {"coef": [10.0, 1.0, 1.0, 1.0], "sigma_f": 4.0}  # fundamentals say +10, sd 4
    inputs = pd.DataFrame({"basis": ["polls", "polls", "prior"], "mean": [0.0, 0.0, -20.0], "sd": [4.0, 1.0, 9.0]})
    out = ind.apply(inputs, params, lead_days=0)
    assert out["mean"].tolist()[0] == pytest.approx(5.0)  # equal precision: halfway
    assert out["mean"].tolist()[1] == pytest.approx(10 / 17)  # precise polls dominate
    assert out["sd"].tolist()[0] == pytest.approx(np.sqrt(8))
    assert out["mean"].tolist()[2] == 10.0 and out["sd"].tolist()[2] == 4.0  # no polls: fundamentals alone


def test_search_change_compares_the_last_two_fortnights():
    ind = INDICATORS["search_momentum"]()
    days = pd.date_range("2024-10-01", "2024-11-05")
    ind.series = {(2024, "Ohio", False): pd.DataFrame(
        {"challenger": [1] * 22 + [3] * 14, "republican": [1] * 36}, index=days)}
    x = ind._change(2024, "Ohio", False, date_(2024, 11, 5))
    assert x == pytest.approx(0.75 - 0.5)  # share 50% -> 75%
    assert ind._change(2024, "Iowa", False, date_(2024, 11, 5)) is None


def test_education_factor_moves_loaded_states_together():
    from pollcast.simulate import SimulateConfig, simulate

    inputs = pd.DataFrame({"mean": [0.0, 0.0], "sd": [0.1, 0.1], "challenger": ["DEM", "DEM"],
                           "loading_education": [5.0, 5.0]})
    sim = SimulateConfig(n_sims=4000, national_sd=0.1)
    hist = simulate(inputs, sim)["challenger_wins_hist"]
    assert hist.get(1, 0) < 0.05  # the shared factor makes split outcomes rare
    unloaded = simulate(inputs.drop(columns="loading_education"), sim)["challenger_wins_hist"]
    assert unloaded.get(1, 0) > 0.3


def date_(y, m, d):
    from datetime import date
    return date(y, m, d)


def test_education_shift_recovers_a_planted_bias():
    rng = np.random.default_rng(5)
    ind = INDICATORS["education_shift"]({"prior_sd": 5.0})
    states = [f"S{i}" for i in range(40)]
    zs = dict(zip(states, rng.normal(0, 1, 40)))
    ind.shares = {2021: zs}
    races = pd.concat([_rows(c, 40, rng) for c in (2018, 2020, 2022, 2024)], ignore_index=True)
    races["margin"] = races["mean"] - 2.0 * races["state"].map(zs) + rng.normal(0, 1, len(races))
    beta = ind.fit(races)["beta"]
    assert beta == pytest.approx(-2.0, abs=0.3) and ind.effect({"beta": beta}) < 0
    shifted = ind.apply(races.head(1), {"beta": -2.0}, 0)
    assert shifted["mean"].iloc[0] == pytest.approx(races["mean"].iloc[0] - 2.0 * zs["S0"])
