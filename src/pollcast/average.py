"""A simple time-weighted polling average.

This is the baseline that the Bayesian state-space model must beat. For each
day it uses only polls released by that day, so every point on the series is
what the average would have said at the time.

Weight of a poll on day d::

    0.5 ** (age / half_life)                      recency, age from field midpoint
  * min(sqrt(n / reference_n), max_sample_weight) sample size
  / k ** flooding_exponent                        k = that pollster's polls in window
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from pollcast.config import AverageConfig

PARTIES = ["DEM", "REP", "IND"]


def poll_margins(polls: pd.DataFrame, cfg: AverageConfig, challenger: str = "DEM") -> pd.DataFrame:
    """One row per poll: `challenger` minus Republican margin, in points.

    The challenger is DEM except in races where an independent is the main
    opponent of the Republican (e.g. Nebraska 2024 and 2026).

    Filters to the configured race, picks one population per poll by the
    preference order, and averages multiple questions within a poll (e.g.
    versions with and without third parties).
    """
    p = polls[(polls["race_type"] == cfg.race_type) & (polls["cycle"] == cfg.cycle)]
    if cfg.state is not None:
        p = p[p["state"] == cfg.state]
    if cfg.exclude_internal:
        p = p[~p["internal"].fillna(False)]
    if cfg.exclude_partisan:
        p = p[p["partisan"].fillna("") == ""]
    if cfg.stages:
        p = p[p["stage"].fillna("").isin([*cfg.stages, ""])]
    if cfg.source_priority and not p.empty:
        present = set(p["source"])
        best = next((s for s in cfg.source_priority if s in present), None)
        if best is not None:
            p = p[p["source"] == best]

    by_question = (
        p[p["party"].isin(PARTIES)]
        .pivot_table(index=["source", "poll_id", "question_id"], columns="party",
                     values="pct", aggfunc="max")
        .reindex(columns=PARTIES)
        .dropna(subset=[challenger, "REP"])
    )
    if by_question.empty:
        return pd.DataFrame(columns=["poll_id", "pollster", "state", "midpoint", "release_date",
                                     "sample_size", "population", "margin", "source"])
    by_question["margin"] = by_question[challenger] - by_question["REP"]
    meta = p.drop_duplicates(["source", "poll_id", "question_id"]).set_index(
        ["source", "poll_id", "question_id"]
    )[["pollster", "state", "start_date", "end_date", "release_date", "sample_size",
       "population", "partisan"]]
    q = by_question[["margin", *PARTIES]].join(meta).reset_index()

    rank = {pop: i for i, pop in enumerate(cfg.population_preference)}
    q["pop_rank"] = q["population"].map(rank).fillna(len(rank))
    best = q.groupby(["source", "poll_id"])["pop_rank"].transform("min")
    q = q[q["pop_rank"] == best]

    out = q.groupby(["source", "poll_id"], as_index=False).agg(
        pollster=("pollster", "first"),
        state=("state", "first"),
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
        release_date=("release_date", "min"),
        sample_size=("sample_size", "max"),
        population=("population", "first"),
        partisan=("partisan", "first"),
        margin=("margin", "mean"),
        **{party: (party, "mean") for party in PARTIES},
    )
    out["midpoint"] = out["start_date"] + (out["end_date"] - out["start_date"]) / 2
    return dedupe_across_sources(out)


def main_challenger(polls: pd.DataFrame, cfg: AverageConfig) -> tuple[str, pd.DataFrame]:
    """The Republican's main opponent in one race (DEM or IND), with its poll margins.

    Whichever has more support summed over polls (share x number of polls),
    judged on the same filtered, single-source polls the average uses. Summing
    keeps one odd poll from flipping a race polled twenty times as D vs R.
    """
    by_party = {c: poll_margins(polls, cfg, challenger=c) for c in ("DEM", "IND")}
    support = {c: (m[c].sum() if len(m) else 0.0) for c, m in by_party.items()}
    challenger = "IND" if support["IND"] > support["DEM"] else "DEM"
    return challenger, by_party[challenger]


def dedupe_across_sources(polls: pd.DataFrame) -> pd.DataFrame:
    """Drop the same poll reported by more than one source.

    Same pollster (case-insensitive), state and end date counts as a duplicate;
    the copy from the source listed first in the config wins, which in
    practice is the order rows arrive in.
    """
    key = (polls["pollster"].str.lower().str.strip() + "|" + polls["state"].fillna("") + "|"
           + polls["end_date"].dt.date.astype(str))
    return polls[~key.duplicated()].reset_index(drop=True)


def weights_on(day: pd.Timestamp, margins: pd.DataFrame, cfg: AverageConfig) -> pd.Series:
    age = (day - margins["midpoint"]).dt.days.astype(float)
    # The average for day d is as of the end of d: it includes polls released that day.
    released = margins["release_date"].dt.normalize() <= day
    live = released & (age >= 0) & (age <= cfg.max_age_days)
    n = margins["sample_size"].astype(float).fillna(cfg.reference_sample_size)
    w = 0.5 ** (age / cfg.half_life_days)
    w *= np.minimum(np.sqrt(n / cfg.reference_sample_size), cfg.max_sample_weight)
    k = margins.loc[live].groupby("pollster")["poll_id"].transform("count")
    w.loc[live] /= k ** cfg.pollster_flooding_exponent
    return w.where(live, 0.0)


def time_weighted_average(margins: pd.DataFrame, cfg: AverageConfig,
                          start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Daily average margin with a rough standard error.

    The standard error treats the weighted polls as independent draws around
    the average (weighted spread over sqrt of effective sample size). It
    ignores shared pollster error, so it is a floor, not a forecast interval.
    """
    if margins.empty:
        return pd.DataFrame(columns=["date", "margin", "se", "n_polls", "effective_n"])
    first = pd.Timestamp(start) if start else margins["release_date"].min().normalize()
    last = pd.Timestamp(end) if end else margins["release_date"].max().normalize()
    rows = []
    for day in pd.date_range(first, last, freq="D"):
        w = weights_on(day, margins, cfg)
        total = w.sum()
        if total <= 0:
            continue
        mean = float((w * margins["margin"]).sum() / total)
        n_eff = float(total**2 / (w**2).sum())
        var = float((w * (margins["margin"] - mean) ** 2).sum() / total)
        rows.append(
            {
                "date": day,
                "margin": mean,
                "se": np.sqrt(var / n_eff) if n_eff > 1 else np.nan,
                "n_polls": int((w > 0).sum()),
                "effective_n": n_eff,
            }
        )
    return pd.DataFrame(rows)
