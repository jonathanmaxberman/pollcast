"""Rain on election day lowers turnout and helps Republicans.

Gomez, Hansford and Krause (2007) found that rain lowers turnout in US
presidential elections and modestly helps Republicans. The adjustment::

    adjusted margin = polling average + beta x election-day rain (inches, state mean)

Declared before testing: beta < 0 (rain moves the margin toward the
Republican; margins are challenger minus Republican).

beta is fit within each cycle and lead, so a cycle-wide polling miss doesn't
leak in, and shrunk toward zero with a normal prior of sd `prior_sd` points
per inch.

Weather forecasts only reach about a week out, so the adjustment is fit and
applied only within `max_lead_days` (default 7) of election day. In the
backtest, observed rain stands in for forecasts that were never archived,
which makes any gain there an upper bound. Early and mail voting, now most
votes in many states, should shrink the effect.
"""

from __future__ import annotations

import pandas as pd

from pollcast.calendar import ELECTION_DAYS
from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator
from pollcast.sources.noaa import weather_table
from pollcast.storage import latest_snapshot


class ElectionDayRain(Indicator):
    name = "election_day_rain"
    description = "Rain on election day lowers turnout and helps Republicans"
    expected = "beta < 0: rain moves the margin toward the Republican"
    expected_sign = -1

    def prepare(self, config) -> None:
        snap = latest_snapshot(config.data_dir, "noaa_weather")
        table = weather_table(snap) if snap else pd.DataFrame(columns=["date", "state", "prcp_in", "kind"])
        # Prefer observed rain where both exist (past elections); forecasts otherwise.
        table = table.assign(order=table["kind"].map({"observed": 0, "forecast": 1}))
        table = table.sort_values("order").drop_duplicates(["date", "state"])
        self.rain = {(str(r.date), r.state): float(r.prcp_in) for r in table.itertuples()}

    def label(self) -> str:
        return self.description

    def feature(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([self.rain.get((str(ELECTION_DAYS.get(int(c))), s)) for c, s in
                          zip(rows["cycle"], rows["state"])], index=rows.index, dtype=float)

    def _in_window(self, rows: pd.DataFrame) -> pd.Series:
        return (rows["basis"] == "polls") & (rows["lead_days"] <= int(self.settings.get("max_lead_days", 7)))

    def fit(self, races: pd.DataFrame) -> dict:
        rows = races[self._in_window(races)].assign(rain=lambda d: self.feature(d)).dropna(subset=["rain"])
        group = rows.groupby(["cycle", "lead_days"])
        x = rows["rain"] - group["rain"].transform("mean")
        miss = rows["margin"] - rows["mean"]
        y = miss - miss.groupby([rows["cycle"], rows["lead_days"]]).transform("mean")
        return {"beta": shrunk_slope(x, y, float(self.settings.get("prior_sd", 3.0)))}

    def effect(self, params: dict) -> float:
        return params["beta"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        if lead_days > int(self.settings.get("max_lead_days", 7)):
            return out
        rain = self.feature(out)
        hit = (out["basis"] == "polls") & rain.notna()
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["beta"] * rain[hit]
        return out
