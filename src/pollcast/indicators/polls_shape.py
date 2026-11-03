"""Adjustments built from the shape of each race's polls.

UndecidedAllocation
    Polls leave out undecided and third-party respondents. If a fraction
    theta of them ends up not voting for the Democrat or Republican and the
    rest split in proportion to decided support, the final margin is::

        adjusted margin = polling margin x (1 - theta x (1 - s)) / s

    where s is the polled Democratic + Republican share (0-1). theta = 1
    leaves margins as polled; theta = 0 hands every undecided to the two
    parties. Unlike the 30-day stretch's single multiplier, races with more
    undecideds stretch more. theta is fit per lead time by grid search on the
    training cycles' margins. Declared: improves held-out log loss in both
    backtests, beating the constant stretch.

LateMomentum
    Does movement in a race's average over the last two weeks continue to
    election day or snap back? adjusted margin = polling average +
    mu x (movement over the previous 14 days; 10 in the 1998-2022 file).
    Declared before testing: mu > 0 (momentum continues). mu is fit within
    each cycle and lead and shrunk toward zero.

MidtermPenalty
    The president's party loses ground in midterms; do polls already capture
    it? adjusted margin = polling average - gamma x s, where s = +1 when the
    president is a Democrat and -1 when a Republican, in midterm years only.
    Declared: gamma > 0. One number per cycle, so it is fit on the
    training cycles' average midterm miss and shrunk toward zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.calendar import PRESIDENT_PARTY
from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator


class UndecidedAllocation(Indicator):
    name = "undecided_allocation"
    description = "Undecided voters split like decided ones, race by race"
    expected = "theta < 1: part of the undecided vote goes to the two parties in proportion"
    expected_sign = 1

    @staticmethod
    def _adjust(mean, two_party, theta):
        s = (two_party / 100.0).clip(lower=0.5, upper=1.0)
        return mean * (1 - theta * (1 - s)) / s

    def fit(self, races: pd.DataFrame) -> dict:
        thetas = {}
        polled = races[(races["basis"] == "polls") & races["two_party"].notna()]
        grid = np.linspace(0, 1, 21)
        for lead, g in polled.groupby("lead_days"):
            sse = [((g["margin"] - self._adjust(g["mean"], g["two_party"], t)) ** 2).sum() for t in grid]
            thetas[str(int(lead))] = float(grid[int(np.argmin(sse))])
        return {"theta": thetas}

    def effect(self, params: dict) -> float:
        return float(np.mean([1 - t for t in params["theta"].values()])) if params["theta"] else 0.0

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        if "two_party" not in out or not params["theta"]:
            return out
        leads = [int(k) for k in params["theta"]]
        theta = params["theta"][str(min(leads, key=lambda k: abs(k - lead_days)))]
        hit = (out["basis"] == "polls") & out["two_party"].notna()
        out.loc[hit, "mean"] = self._adjust(out.loc[hit, "mean"], out.loc[hit, "two_party"], theta)
        return out


class LateMomentum(Indicator):
    name = "late_momentum"
    description = "Movement in the last two weeks of polling continues to election day"
    expected = "mu > 0: late movement continues"
    expected_sign = 1

    def fit(self, races: pd.DataFrame) -> dict:
        rows = races[(races["basis"] == "polls") & races["trend14"].notna()]
        keys = [rows["cycle"], rows["lead_days"]]
        x = rows["trend14"] - rows["trend14"].groupby(keys).transform("mean")
        miss = rows["margin"] - rows["mean"]
        y = miss - miss.groupby(keys).transform("mean")
        return {"mu": shrunk_slope(x, y, float(self.settings.get("prior_sd", 0.5)))}

    def effect(self, params: dict) -> float:
        return params["mu"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        if "trend14" not in out:
            return out
        hit = (out["basis"] == "polls") & out["trend14"].notna()
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["mu"] * out.loc[hit, "trend14"]
        return out


class MidtermPenalty(Indicator):
    name = "midterm_penalty"
    description = "Polls understate the president's party's midterm losses"
    expected = "gamma > 0: margins move against the president's party in midterms"
    expected_sign = 1

    @staticmethod
    def _sign(cycle) -> float:
        midterm = int(cycle) % 4 == 2
        return {"DEM": 1.0, "REP": -1.0}.get(PRESIDENT_PARTY.get(int(cycle)), 0.0) if midterm else 0.0

    def fit(self, races: pd.DataFrame) -> dict:
        eve = races[(races["basis"] == "polls") & (races["lead_days"] == races["lead_days"].min())]
        by = eve.assign(miss=eve["margin"] - eve["mean"]).groupby("cycle")["miss"].mean()
        s = pd.Series([self._sign(c) for c in by.index], index=by.index)
        m = s != 0
        # miss = -gamma x s  =>  regress -miss on s through the origin, shrunk.
        gamma = shrunk_slope(s[m], -by[m], float(self.settings.get("prior_sd", 1.0)),
                             noise_sd=float(self.settings.get("noise_sd", 3.0)))
        return {"gamma": gamma}

    def effect(self, params: dict) -> float:
        return params["gamma"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        s = pd.Series([self._sign(c) for c in out["cycle"]], index=out.index)
        hit = out["basis"] == "polls"
        out.loc[hit, "mean"] = out.loc[hit, "mean"] - params["gamma"] * s[hit]
        return out
