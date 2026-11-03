"""States with many non-college white voters miss together.

In 2016 and 2020 the polls' state misses lined up with the share of
non-college white adults (AAPOR's 2016 and 2020 reviews); the direction and
size vary by cycle. So this adds a shared error rather than a shift: each
simulation draws one education factor, and each state's margin moves by
factor x sd_edu x (its non-college white share, centered on the average state)::

    margin = mean + national + sd_edu x z_state x factor + state error

sd_edu is estimated from the training cycles: within each cycle, the
election-eve misses (net of the cycle's average miss) are regressed on z,
and sd_edu is the spread of those slopes across cycles after removing their
sampling noise.

Declared before testing: sd_edu > 0, and the seat-count score improves.
Because it changes how states miss together rather than each race's odds,
it is judged on the log probability given to each cycle's actual number of
challenger wins, not on race-by-race log loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.indicators.base import Indicator
from pollcast.sources import SOURCES
from pollcast.storage import latest_snapshot


class EducationCorrelation(Indicator):
    name = "education_correlation"
    description = "States with many non-college white voters miss together"
    expected = "sd_edu > 0, and the seat-count score improves"
    expected_sign = 1
    metric = "seat_log_score"

    def prepare(self, config) -> None:
        snap = latest_snapshot(config.data_dir, "census_acs")
        t = SOURCES["census_acs"](config).normalize(snap) if snap else pd.DataFrame()
        self.shares = {}
        for year, g in t.groupby("year"):
            g = g[g["state"] != "District of Columbia"]
            z = (g["noncollege_white_share"] - g["noncollege_white_share"].mean()) / g["noncollege_white_share"].std()
            self.shares[int(year)] = dict(zip(g["state"], z))

    def testable(self) -> bool:
        return bool(self.shares)

    def untestable_reason(self) -> str:
        return "no Census snapshot; run `pollcast fetch census_acs`"

    def z(self, rows: pd.DataFrame) -> pd.Series:
        years = sorted(self.shares)

        def pick(cycle):  # the latest vintage released before the election, else the earliest
            before = [y for y in years if y + 1 < cycle]
            return before[-1] if before else years[0]

        return pd.Series([self.shares[pick(int(c))].get(s, np.nan) for c, s in zip(rows["cycle"], rows["state"])],
                         index=rows.index, dtype=float)

    def fit(self, races: pd.DataFrame) -> dict:
        eve = races[(races["basis"] == "polls") & (races["lead_days"] == races["lead_days"].min())]
        eve = eve.assign(z=self.z(eve), miss=eve["margin"] - eve["mean"]).dropna(subset=["z"])
        slopes, noise = [], []
        for _, g in eve.groupby("cycle"):
            if len(g) < 8:
                continue
            x = g["z"] - g["z"].mean()
            y = g["miss"] - g["miss"].mean()
            b = float((x * y).sum() / (x**2).sum())
            resid = y - b * x
            slopes.append(b)
            noise.append(float(resid.var(ddof=2) / (x**2).sum()))
        if not slopes:
            return {"sd_edu": 0.0, "slopes": []}
        var = float(np.mean(np.square(slopes)) - np.mean(noise))
        return {"sd_edu": float(np.sqrt(max(var, 0.0))), "slopes": slopes}

    def effect(self, params: dict) -> float:
        return params["sd_edu"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        out["loading_education"] = (params["sd_edu"] * self.z(out)).fillna(0.0)
        return out


class EducationShift(EducationCorrelation):
    """Polls overstate the challenger where non-college white adults are many.

    Post hoc: proposed after fitting EducationCorrelation showed every cycle's
    slope was negative (2018-2024), so the training data already suggested it
    and a pass is weaker evidence than a first-try pass. The adjustment::

        adjusted margin = polling average + beta x z_state

    with z the state's standardized non-college white share. Declared:
    beta < 0. Fit like the other shifts: within each cycle and lead (the
    cycle's average miss removed), shrunk toward zero with prior sd
    `prior_sd` points per standard deviation. Judged on race-level log loss.
    """

    name = "education_shift"
    description = "Polls overstate the challenger where non-college white adults are many"
    expected = "beta < 0: the margin moves toward the Republican in high non-college-white states"
    expected_sign = -1
    metric = "race_log_loss"

    def fit(self, races: pd.DataFrame) -> dict:
        from pollcast.indicators._shrink import shrunk_slope

        rows = races[races["basis"] == "polls"].assign(z=lambda d: self.z(d)).dropna(subset=["z"])
        keys = [rows["cycle"], rows["lead_days"]]
        x = rows["z"] - rows["z"].groupby(keys).transform("mean")
        miss = rows["margin"] - rows["mean"]
        y = miss - miss.groupby(keys).transform("mean")
        return {"beta": shrunk_slope(x, y, float(self.settings.get("prior_sd", 2.0)))}

    def effect(self, params: dict) -> float:
        return params["beta"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        z = self.z(out)
        hit = (out["basis"] == "polls") & z.notna()
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["beta"] * z[hit]
        return out
