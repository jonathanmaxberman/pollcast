"""538's pollster-ratings data: every rated poll since 1998, with its actual error.

From github.com/fivethirtyeight/data (pollster-ratings/<vintage>/raw-polls.csv).
Each row is a poll of a general or primary election, the actual result, and
`bias` = poll margin minus actual margin (candidate 1 minus candidate 2; in
Democrat-vs-Republican races, positive means the poll leaned Democratic).

Used to give each pollster a historical lean before a cycle starts, from
elections strictly earlier than the one being forecast.
"""

from __future__ import annotations

import pandas as pd

from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/{vintage}/raw-polls.csv"


class FiveThirtyEightRatings(Source):
    name = "fivethirtyeight_ratings"
    table = "pollster_history"
    frozen = True
    attribution = "Pollster accuracy data: FiveThirtyEight (ABC News)."

    def fetch(self, writer: SnapshotWriter) -> None:
        vintage = self.settings.get("vintage", "2023")
        url = URL.format(vintage=vintage)
        r = self.get(url)
        if not r.content.startswith(b"poll_id"):
            raise RuntimeError(f"538 ratings: unexpected file at {url}: {r.content[:100]!r}")
        writer.write("raw-polls.csv", r.content, url=url, vintage=vintage)

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        raw = pd.read_csv(snapshot.path / "raw-polls.csv")
        raw = raw[(raw["cand1_party"] == "DEM") & (raw["cand2_party"] == "REP")
                  & ~raw["type_simple"].str.contains("-P")]
        days = (pd.to_datetime(raw["electiondate"], format="mixed")
                - pd.to_datetime(raw["polldate"], format="mixed")).dt.days
        return pd.DataFrame({
            "year": raw["year"].astype(int), "race": raw["race"],
            "pollster_rating_id": raw["pollster_rating_id"].astype("Int64").astype(str),
            "partisan": raw["partisan"].fillna(""), "days_out": days, "bias": raw["bias"].astype(float),
        })


def historical_leans(history: pd.DataFrame, before_year: int, *, max_days_out: int = 21,
                     prior_polls: float = 5.0) -> dict[str, float]:
    """Each pollster's past lean relative to other pollsters in the same races.

    Only elections before `before_year`, and polls within `max_days_out` of
    election day. A poll's relative lean is its bias minus the mean bias of
    every poll of that race, so a year when all polls missed together doesn't
    count against any one pollster. Each pollster's mean is shrunk toward zero
    as if it had `prior_polls` extra polls with no lean.
    """
    h = history[(history["year"] < before_year) & (history["days_out"] <= max_days_out)
                & (history["days_out"] >= 0)]
    if h.empty:
        return {}
    rel = h["bias"] - h.groupby("race")["bias"].transform("mean")
    g = rel.groupby(h["pollster_rating_id"])
    return (g.sum() / (g.count() + prior_polls)).to_dict()


def historical_bias(history: pd.DataFrame, before_year: int, *, max_days_out: int = 21,
                    prior_polls: float = 5.0, half_life_years: float = 8.0) -> dict[str, float]:
    """Each pollster's past lean against actual results (not against other pollsters).

    Only elections before `before_year`, polls within `max_days_out` of election
    day, recent elections weighted more (`half_life_years`). Positive means the
    pollster's polls ran more Democratic than the result. Each pollster's mean
    is shrunk toward zero as if it had `prior_polls` extra polls with no lean.
    Also returns "*|DEM" and "*|REP": the same for all polls sponsored by each
    party, a fallback for partisan polls from pollsters with no history.
    """
    h = history[(history["year"] < before_year) & (history["days_out"] <= max_days_out)
                & (history["days_out"] >= 0)]
    if h.empty:
        return {}
    w = 0.5 ** ((before_year - h["year"]) / half_life_years)
    wb = w * h["bias"]
    out = (wb.groupby(h["pollster_rating_id"]).sum()
           / (w.groupby(h["pollster_rating_id"]).sum() + prior_polls)).to_dict()
    for code, party in (("D", "DEM"), ("R", "REP")):
        m = h["partisan"] == code
        if m.any():
            out[f"*|{party}"] = float(wb[m].sum() / (w[m].sum() + prior_polls))
    return out


def historical_accuracy(history: pd.DataFrame, before_year: int, *, max_days_out: int = 21,
                        prior_polls: float = 10.0) -> dict[str, float]:
    """Each pollster's past error size relative to other polls of the same races.

    A poll's relative error is its absolute miss minus the mean absolute miss
    of every poll of that race, so hard races don't count against anyone.
    Positive = less accurate than peers. Shrunk toward zero as if each
    pollster had `prior_polls` extra average polls. Elections before
    `before_year` only.
    """
    h = history[(history["year"] < before_year) & (history["days_out"] <= max_days_out)
                & (history["days_out"] >= 0)]
    if h.empty:
        return {}
    err = h["bias"].abs()
    rel = err - err.groupby(h["race"]).transform("mean")
    g = rel.groupby(h["pollster_rating_id"])
    return (g.sum() / (g.count() + prior_polls)).to_dict()


def accuracy_multipliers(accuracy: dict[str, float], scale: float = 2.0,
                         low: float = 0.5, high: float = 2.0) -> dict[str, float]:
    """Poll weight multipliers: exp(-relative error / scale), clipped to [low, high]."""
    import math

    return {k: min(max(math.exp(-v / scale), low), high) for k, v in accuracy.items()}
