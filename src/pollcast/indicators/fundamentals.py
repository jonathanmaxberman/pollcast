"""A fundamentals estimate for each Senate race, blended with the polls.

Research (Gelman and King 1993; the Economist's 2020 model) finds results are
more predictable from fundamentals than early polls suggest, and that polls
and fundamentals together beat either alone. The estimate::

    fundamentals margin = a + b_lean x lean + b_env x national + b_inc x incumbency

  lean        the state's last presidential margin minus the national margin
              (the most recent presidential election before the cycle)
  national    pollcast's generic-ballot average as of the forecast date
  incumbency  +1 if the seat's last winner is running for the challenger's
              side (DEM or IND), -1 if a Republican incumbent is running, 0 if
              open or a special election

Margins are challenger minus Republican. The coefficients are fit by least
squares on the training cycles' election results, one row per race. The
spread of its misses within a cycle, sigma_f, sets its weight: a poll-based
race gets the precision-weighted blend of polls and fundamentals (its sd
shrinks), and a race with no polls uses the fundamentals estimate in place of
its last Senate result.

Declared before testing: b_lean > 0, and blending improves held-out log loss.

Known gaps: candidate lists come from certified results in the backtest (in
late-primary states that is a little early 90 days out) and from poll
candidate names live; appointed incumbents in specials count as open seats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.average import poll_margins, time_weighted_average
from pollcast.calendar import ELECTION_DAYS
from pollcast.indicators.base import Indicator
from pollcast.storage import latest_snapshot

SUFFIXES = {"JR", "JR.", "SR", "SR.", "II", "III", "IV"}


def last_name(name: str) -> str:
    parts = [p for p in str(name).upper().replace(",", " ").split() if p not in SUFFIXES]
    return parts[-1] if parts else ""


class SenateFundamentals(Indicator):
    name = "senate_fundamentals"
    description = "Blend each race's polls with a fundamentals estimate (state lean, incumbency, national mood)"
    expected = "b_lean > 0, and blending improves held-out accuracy most where polls are thin"
    expected_sign = 1

    def prepare(self, config) -> None:
        from dataclasses import replace

        from pollcast.pipeline import build_polls
        from pollcast.sources.mit import president_margins, senate_candidates, senate_results

        snap = latest_snapshot(config.data_dir, "mit_results")
        if snap is None or not (snap.path / "president.csv").exists():
            raise RuntimeError("senate_fundamentals: run `pollcast fetch mit_results --force` first")
        pres = president_margins(snap.path / "president.csv")
        national = pres[pres["state"] == ""].set_index("year")["margin"]
        pres = pres[pres["state"] != ""]
        pres["lean"] = pres["margin"] - pres["year"].map(national)
        self.lean = pres.set_index(["year", "state"])["lean"].to_dict()
        self.pres_years = sorted(national.index)

        self.results = senate_results(snap.path / "senate.csv")
        self.candidates = senate_candidates(snap.path / "senate.csv")

        polls = build_polls(config)
        self.poll_names = (polls[(polls["race_type"] == "senate") & (polls["stage"].fillna("") == "general")]
                           .groupby(["cycle", "state"])["candidate"].apply(lambda s: {last_name(n) for n in s})
                           .to_dict())
        # Generic-ballot average by day for each cycle; each day uses only polls released by then.
        self.env = {}
        for cycle, day in ELECTION_DAYS.items():
            cfg = replace(config.average, race_type="generic", cycle=cycle)
            m = poll_margins(polls, cfg)
            if m.empty:
                continue
            avg = time_weighted_average(m, cfg, start=day - pd.Timedelta(days=120), end=day)
            self.env[cycle] = avg.set_index("date")["margin"]

    # ---- features
    def _lean(self, cycle: int, state: str) -> float:
        years = [y for y in self.pres_years if y < cycle]
        return float(self.lean.get((years[-1], state), np.nan)) if years else np.nan

    def _env(self, cycle: int, as_of) -> float:
        series = self.env.get(int(cycle))
        if series is None:
            return np.nan
        known = series[series.index <= pd.Timestamp(as_of)]
        return float(known.iloc[-1]) if len(known) else np.nan

    def _incumbency(self, cycle: int, state: str, special: bool) -> float:
        if special:
            return 0.0
        prev = self.candidates[(self.candidates["cycle"] == cycle - 6) & (self.candidates["state"] == state)
                               & ~self.candidates["special"] & self.candidates["won"]]
        if prev.empty:
            return 0.0
        winner = prev.iloc[0]
        now = self.candidates[(self.candidates["cycle"] == cycle) & (self.candidates["state"] == state)
                              & ~self.candidates["special"]]
        names = {last_name(n) for n in now["name"]} if len(now) else self.poll_names.get((cycle, state), set())
        if last_name(winner["name"]) not in names:
            return 0.0
        return -1.0 if winner["party"] == "REP" else 1.0

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        special = rows["special"] if "special" in rows else pd.Series(False, index=rows.index)
        # Rows may carry their own national mood (the 1998-2022 history replay does).
        env = rows["env"] if "env" in rows else pd.Series(
            [self._env(int(c), a) for c, a in zip(rows["cycle"], rows["as_of"])], index=rows.index)
        senate = rows["race_type"].eq("senate") if "race_type" in rows else pd.Series(True, index=rows.index)
        return pd.DataFrame({
            "lean": [self._lean(int(c), s) for c, s in zip(rows["cycle"], rows["state"])],
            "env": env.astype(float),
            # Incumbency comes from MIT's Senate results, so only Senate races get it.
            "inc": [self._incumbency(int(c), s, bool(sp)) if is_sen else 0.0
                    for c, s, sp, is_sen in zip(rows["cycle"], rows["state"], special, senate)],
        }, index=rows.index)

    # ---- model
    def fit(self, races: pd.DataFrame) -> dict:
        eve = races[races["lead_days"] == races["lead_days"].min()]
        X = self.features(eve)
        ok = X.notna().all(axis=1)
        X, y, cycles = X[ok], eve.loc[ok, "margin"], eve.loc[ok, "cycle"]
        A = np.column_stack([np.ones(len(X)), X["lean"], X["env"], X["inc"]])
        ridge = float(self.settings.get("ridge", 1.0)) * np.diag([0, 1, 1, 1])
        coef = np.linalg.solve(A.T @ A + ridge, A.T @ y.to_numpy())
        resid = pd.Series(y.to_numpy() - A @ coef, index=y.index)
        within = resid - resid.groupby(cycles).transform("mean")
        return {"coef": [float(c) for c in coef], "sigma_f": float(np.sqrt((within**2).mean()))}

    def effect(self, params: dict) -> float:
        return params["coef"][1]

    def estimate(self, rows: pd.DataFrame, params: dict) -> pd.Series:
        X = self.features(rows)
        a, b_lean, b_env, b_inc = params["coef"]
        return a + b_lean * X["lean"] + b_env * X["env"] + b_inc * X["inc"]

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        f = self.estimate(out, params)
        sf2 = params["sigma_f"] ** 2
        ok = f.notna()
        polled = ok & (out["basis"] == "polls")
        sp2 = out.loc[polled, "sd"] ** 2
        out.loc[polled, "mean"] = (out.loc[polled, "mean"] / sp2 + f[polled] / sf2) / (1 / sp2 + 1 / sf2)
        out.loc[polled, "sd"] = np.sqrt(1 / (1 / sp2 + 1 / sf2))
        unpolled = ok & (out["basis"] != "polls")
        out.loc[unpolled, "mean"] = f[unpolled]
        out.loc[unpolled, "sd"] = params["sigma_f"]
        out["sd_override"] = out["sd"]
        return out
