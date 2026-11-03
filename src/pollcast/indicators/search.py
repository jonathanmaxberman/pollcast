"""A late rise in searches for the challenger means they beat their polls.

From Google Trends (one query per race, challenger vs Republican, in-state):
the challenger's share of the two candidates' combined search interest in
the 14 days up to the forecast date, minus their share in the 14 days
before that. The adjustment::

    adjusted margin = polling average + beta x change in search share

Declared before testing: beta > 0. Expected to fail: search attention
mostly reflects news the polls already capture.

beta is fit within each cycle and lead (so a cycle-wide polling miss doesn't
leak in) and shrunk toward zero (prior sd `prior_sd` points per unit of
share change, so a 10-point share swing is worth at most about a point).
"""

from __future__ import annotations

import pandas as pd

from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator
from pollcast.sources import SOURCES
from pollcast.storage import latest_snapshot


class SearchMomentum(Indicator):
    name = "search_momentum"
    description = "A late rise in searches for the challenger means they beat their polls"
    expected = "beta > 0: a rising search share moves the margin toward the challenger"
    expected_sign = 1

    def prepare(self, config) -> None:
        snap = latest_snapshot(config.data_dir, "google_trends")
        table = SOURCES["google_trends"](config).normalize(snap) if snap else pd.DataFrame()
        self.series = {k: g.set_index("date")[["challenger", "republican"]].sort_index()
                       for k, g in table.groupby(["cycle", "state", "special"])} if len(table) else {}

    def testable(self) -> bool:
        return bool(self.series)

    def untestable_reason(self) -> str:
        return "no Google Trends snapshot; run `pollcast fetch google_trends`"

    def _change(self, cycle, state, special, as_of) -> float | None:
        s = self.series.get((int(cycle), state, bool(special)))
        if s is None:
            return None
        end = pd.Timestamp(as_of)
        last = s[(s.index > end - pd.Timedelta(days=14)) & (s.index <= end)].sum()
        prev = s[(s.index > end - pd.Timedelta(days=28)) & (s.index <= end - pd.Timedelta(days=14))].sum()
        if last.sum() == 0 or prev.sum() == 0:
            return None
        return float(last["challenger"] / last.sum() - prev["challenger"] / prev.sum())

    def feature(self, rows: pd.DataFrame) -> pd.Series:
        special = rows["special"] if "special" in rows else pd.Series(False, index=rows.index)
        return pd.Series([self._change(c, s, sp, a) for c, s, sp, a in
                          zip(rows["cycle"], rows["state"], special, rows["as_of"])], index=rows.index, dtype=float)

    def fit(self, races: pd.DataFrame) -> dict:
        rows = races[races["basis"] == "polls"].assign(x=lambda d: self.feature(d)).dropna(subset=["x"])
        keys = [rows["cycle"], rows["lead_days"]]
        x = rows["x"] - rows["x"].groupby(keys).transform("mean")
        miss = rows["margin"] - rows["mean"]
        y = miss - miss.groupby(keys).transform("mean")
        return {"beta": shrunk_slope(x, y, float(self.settings.get("prior_sd", 10.0)))}

    def effect(self, params: dict) -> float:
        return params["beta"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        x = self.feature(out)
        hit = (out["basis"] == "polls") & x.notna()
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["beta"] * x[hit]
        return out
