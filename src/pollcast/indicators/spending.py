"""Outside spending as a window into private polling.

Parties and big groups commission their own polls and spend where those
polls say it matters. If spending runs ahead of what the public polls
justify, the private polls may know something.

SpendingCloseness
    Offense and defense both signal a closer race: Democrats spending big in
    a race public polls call safely Republican, or Republicans spending big to
    defend one. Excess spending e is the log of total outside spending (by the
    forecast date) minus what the race's polled closeness and the state's size
    predict, fit on the training cycles. The adjustment::

        adjusted margin = polling margin x (1 - kappa x e)

    Declared before testing: kappa > 0 (more excess spending, closer result).

SpendingTilt
    The Democratic side's share of outside spending, minus the share its
    polling position predicts (fit on training cycles), shifts the margin:
    adjusted margin = polling margin + beta x excess share. Declared: beta > 0.
    Weaker on theory, since defensive spending cuts the other way.

Both use only reports received by the forecast date, fit within each cycle
and lead (so a cycle-wide miss doesn't leak in), and shrink toward no
effect. FEC data starts in 2010.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator
from pollcast.storage import latest_snapshot


class _Spending(Indicator):
    def prepare(self, config) -> None:
        from pollcast.sources.fec import spending_rows

        snap = latest_snapshot(config.data_dir, "fec_ie")
        self.rows = spending_rows(snap) if snap else pd.DataFrame()
        self._cache = {}
        mit = latest_snapshot(config.data_dir, "mit_results")
        pres = pd.read_csv(mit.path / "president.csv", usecols=["year", "state_po", "totalvotes"]).drop_duplicates()
        from pollcast.sources._538_format import STATE_NAMES

        pres["state"] = pres["state_po"].map(STATE_NAMES)
        self.votes = pres.set_index(["year", "state"])["totalvotes"].to_dict()
        self.vote_years = sorted(pres["year"].unique())

    def testable(self) -> bool:
        return not self.rows.empty

    def untestable_reason(self) -> str:
        return "no FEC snapshot; run `pollcast fetch fec_ie`"

    def _spend(self, cycle, as_of) -> pd.DataFrame:
        from pollcast.sources.fec import spending_as_of

        key = (int(cycle), str(as_of))
        if key not in self._cache:
            self._cache[key] = spending_as_of(self.rows, int(cycle), as_of) if int(cycle) >= 2010 else None
        return self._cache[key]

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        total, share, size = [], [], []
        special = rows["special"] if "special" in rows else pd.Series(False, index=rows.index)
        # Senate outside spending only describes Senate races.
        other = rows["race_type"].ne("senate") if "race_type" in rows else pd.Series(False, index=rows.index)
        special = special | other
        for c, st, a, sp in zip(rows["cycle"], rows["state"], rows["as_of"], special):
            s = self._spend(c, a)
            years = [y for y in self.vote_years if y < int(c)]
            size.append(np.log(self.votes.get((years[-1], st), np.nan)) if years else np.nan)
            # FEC rows don't say which of two same-state races they're for; skip specials.
            if s is None or sp or st not in s.index:
                total.append(np.nan if s is None or sp else 0.0)
                share.append(np.nan)
                continue
            d, r = float(s.loc[st, "DEM"]), float(s.loc[st, "REP"])
            total.append(np.log1p((d + r) / 1e6))
            share.append(d / (d + r) if d + r > 0 else np.nan)
        return pd.DataFrame({"log_total": total, "d_share": share, "log_votes": size}, index=rows.index)


class SpendingCloseness(_Spending):
    name = "spending_closeness"
    description = "Outside spending beyond what the polls justify means a closer race"
    expected = "kappa > 0: excess spending pulls the result toward even"
    expected_sign = 1

    def fit(self, races: pd.DataFrame) -> dict:
        rows = races[races["basis"] == "polls"]
        X = self.features(rows).assign(absm=rows["mean"].abs())
        ok = X[["log_total", "log_votes"]].notna().all(axis=1)
        X, rows = X[ok], rows[ok]
        if len(rows) < 20:
            return {"coef": [0.0, 0.0, 0.0], "kappa": 0.0}
        A = np.column_stack([np.ones(len(X)), X["absm"], X["log_votes"]])
        coef = np.linalg.lstsq(A, X["log_total"].to_numpy(), rcond=None)[0]
        e = X["log_total"] - A @ coef
        keys = [rows["cycle"], rows["lead_days"]]
        reg = -e * rows["mean"]  # margin = mean (1 - kappa e)  =>  miss = kappa x (-e x mean)
        miss = rows["margin"] - rows["mean"]
        x = reg - reg.groupby(keys).transform("mean")
        y = miss - miss.groupby(keys).transform("mean")
        return {"coef": [float(v) for v in coef], "kappa": shrunk_slope(x, y, float(self.settings.get("prior_sd", 0.05)))}

    def effect(self, params: dict) -> float:
        return params["kappa"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        X = self.features(out)
        a, b, c = params["coef"]
        e = X["log_total"] - (a + b * out["mean"].abs() + c * X["log_votes"])
        hit = (out["basis"] == "polls") & e.notna()
        factor = (1 - params["kappa"] * e[hit]).clip(0.5, 1.5)
        out.loc[hit, "mean"] = out.loc[hit, "mean"] * factor
        return out


class SpendingTilt(_Spending):
    name = "spending_tilt"
    description = "A party outspending its polling position is doing better than the polls show"
    expected = "beta > 0: a higher Democratic-side spending share than the polls predict helps Democrats"
    expected_sign = 1

    def fit(self, races: pd.DataFrame) -> dict:
        rows = races[races["basis"] == "polls"]
        X = self.features(rows)
        ok = X["d_share"].notna()
        X, rows = X[ok], rows[ok]
        if len(rows) < 20:
            return {"coef": [0.5, 0.0], "beta": 0.0}
        A = np.column_stack([np.ones(len(X)), rows["mean"]])
        coef = np.linalg.lstsq(A, X["d_share"].to_numpy(), rcond=None)[0]
        resid = X["d_share"] - A @ coef
        keys = [rows["cycle"], rows["lead_days"]]
        miss = rows["margin"] - rows["mean"]
        x = resid - resid.groupby(keys).transform("mean")
        y = miss - miss.groupby(keys).transform("mean")
        return {"coef": [float(v) for v in coef], "beta": shrunk_slope(x, y, float(self.settings.get("prior_sd", 5.0)))}

    def effect(self, params: dict) -> float:
        return params["beta"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        X = self.features(out)
        a, b = params["coef"]
        resid = X["d_share"] - (a + b * out["mean"])
        hit = (out["basis"] == "polls") & resid.notna()
        # Margins are challenger minus Republican; the Democratic side includes independents.
        out.loc[hit, "mean"] = out.loc[hit, "mean"] + params["beta"] * resid[hit]
        return out
