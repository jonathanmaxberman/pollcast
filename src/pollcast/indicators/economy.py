"""A souring economy hurts the president's party beyond what polls show.

The adjustment moves every poll-based race by the same amount::

    adjusted margin = polling average + beta x s x z

where z is how bad the economy looks as of the forecast date (from ALFRED
vintages, so nothing published later leaks in) and s is +1 when the
president is a Republican (bad news helps the challenger) and -1 when a
Democrat (bad news helps the Republican).

Series (`series` setting):
  UMCSENT  consumer sentiment; z = 12-month drop / 10 points
  GASREGW  regular gas price; z = 12-month rise / $0.50

Declared before testing: beta > 0 (bad news costs the president's party).

Because it moves every race together, it is really a prediction of the
cycle's national polling miss: one data point per cycle. beta is fit on the
election-eve national miss of each training cycle and shrunk toward zero
(prior sd `prior_sd` points per unit of z, noise sd `noise_sd`), so with
three training cycles it should stay small unless the pattern is strong.
"""

from __future__ import annotations

import pandas as pd

from pollcast.calendar import PRESIDENT_PARTY
from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator
from pollcast.sources import SOURCES
from pollcast.storage import latest_snapshot

SCALES = {"UMCSENT": -10.0, "GASREGW": 0.5}  # change in units that counts as z = +1 (bad)
NAMES = {"UMCSENT": "consumer sentiment", "GASREGW": "gas prices"}


class EconomyVsPresident(Indicator):
    name = "economy_vs_president"
    description = "A souring economy hurts the president's party beyond its polls"
    expected = "beta > 0: worse readings shift the margin toward the out-party"
    expected_sign = 1

    @property
    def series(self) -> str:
        return self.settings.get("series", "UMCSENT")

    def label(self) -> str:
        return f"{self.description} ({NAMES.get(self.series, self.series)})"

    def prepare(self, config) -> None:
        snap = latest_snapshot(config.data_dir, "alfred")
        table = SOURCES["alfred"](config).normalize(snap) if snap else pd.DataFrame()
        self.table = table[table["series"] == self.series].sort_values("vintage_date") if len(table) else table

    def z(self, as_of) -> float | None:
        if self.table is None or self.table.empty:
            return None
        known = self.table[self.table["vintage_date"] <= pd.Timestamp(as_of)]
        if known.empty or pd.isna(known.iloc[-1]["change_12m"]):
            return None
        return float(known.iloc[-1]["change_12m"] / SCALES[self.series])

    def signed_feature(self, rows: pd.DataFrame) -> pd.Series:
        vals = []
        for cycle, as_of in zip(rows["cycle"], rows["as_of"]):
            z = self.z(as_of)
            sign = {"REP": 1.0, "DEM": -1.0}.get(PRESIDENT_PARTY.get(int(cycle)), 0.0)
            vals.append(None if z is None else sign * z)
        return pd.Series(vals, index=rows.index, dtype=float)

    def fit(self, races: pd.DataFrame) -> dict:
        eve = races[(races["basis"] == "polls") & (races["lead_days"] == races["lead_days"].min())]
        by_cycle = eve.assign(miss=eve["margin"] - eve["mean"], x=self.signed_feature(eve)).groupby("cycle")
        x, y = by_cycle["x"].first(), by_cycle["miss"].mean()
        keep = x.notna()
        beta = shrunk_slope(x[keep], y[keep], float(self.settings.get("prior_sd", 1.0)),
                            noise_sd=float(self.settings.get("noise_sd", 4.0)))
        return {"beta": beta}

    def effect(self, params: dict) -> float:
        return params["beta"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        x = self.signed_feature(out)
        hit = (out["basis"] == "polls") & x.notna()
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["beta"] * x[hit]
        return out
