"""Error sizes calibrated from past misses, by lead time and race type.

The simulation's error sizes in the config are guesses. This fits them from
the training cycles' actual misses, separately for each lead time (polls
miss more months out) and each race type (governor and House polls miss more
than Senate ones), with the same method-of-moments estimator the backtest
uses (pollcast.backtest.estimate_error_sizes):

  national_sd   spread of each cycle's average miss
  state_sd, few_polls_sd
                leftover miss per race, regressed on 1 / effective polls

It changes no race's expected margin, only how uncertain it is, so it makes
no bet on which way polls miss. Declared before testing: improves held-out
log loss in both backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.indicators.base import Indicator


class CalibratedErrors(Indicator):
    name = "calibrated_errors"
    description = "Error sizes fit to past polling misses, by lead time and race type"
    expected = "fitted error sizes improve held-out log loss"
    expected_sign = 1

    def fit(self, races: pd.DataFrame) -> dict:
        from pollcast.backtest import ErrorSizes, estimate_error_sizes

        fallback = ErrorSizes(float(self.settings.get("national_sd", 3.0)), float(self.settings.get("state_sd", 4.0)),
                              float(self.settings.get("few_polls_sd", 6.0)), float(self.settings.get("prior_sd", 9.0)))
        types = races["race_type"] if "race_type" in races else pd.Series("senate", index=races.index)
        out = {}
        for lead, g in races.groupby("lead_days"):
            nat = estimate_error_sizes(g, fallback).national_sd
            by_type = {}
            for rt, gt in g.groupby(types[g.index]):
                s = estimate_error_sizes(gt, fallback)
                by_type[str(rt)] = {"state_sd": s.state_sd, "few_polls_sd": s.few_polls_sd, "prior_sd": s.prior_sd}
            out[str(int(lead))] = {"national_sd": nat, "types": by_type}
        return {"by_lead": out}

    def effect(self, params: dict) -> float:
        return 1.0  # nothing directional to check; judged on log loss alone

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        leads = [int(k) for k in params["by_lead"]]
        p = params["by_lead"][str(min(leads, key=lambda k: abs(k - lead_days)))]
        types = out["race_type"] if "race_type" in out else pd.Series("senate", index=out.index)
        sd = []
        for rt, basis, se, n in zip(types, out["basis"], out.get("se", 0.0), out.get("effective_n", 1.0)):
            t = p["types"].get(str(rt)) or next(iter(p["types"].values()))
            if basis == "polls":
                sd.append(np.sqrt(t["state_sd"] ** 2 + (se or 0.0) ** 2 + t["few_polls_sd"] ** 2 / max(n or 1.0, 1.0)))
            else:
                sd.append(t["prior_sd"])
        out["sd"] = sd
        out["sd_override"] = out["sd"]
        out["national_sd_override"] = p["national_sd"]
        return out
