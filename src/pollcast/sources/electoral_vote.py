"""Electoral-vote.com poll files: one poll per line, used as a cross-check.

Columns: Day (day of the election year at the poll's midpoint), Len (days in
the field), State, Dem, GOP, Ind, Date (end date, no year), Pollster (with a
"-<Len>" suffix). Rows from pollster "Election <year>" are prior results the
site seeds each state with, not polls, and are dropped.

The files carry no sample size, population or publication time.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from pollcast.schema import POLL_COLUMNS, conform
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

URL = "https://electoral-vote.com/evp{cycle}/{office}/{file}"
OFFICES = {"Senate": ("senate", "senate_polls.csv"), "Pres": ("president", "pres_polls.csv")}


class ElectoralVote(Source):
    name = "electoral_vote"
    table = "polls"
    attribution = "Poll data from Electoral-vote.com."

    def fetch(self, writer: SnapshotWriter) -> None:
        cycle = self.settings.get("cycle", self.config.average.cycle)
        for office in self.settings.get("offices", ["Senate"]):
            _, file = OFFICES[office]
            url = URL.format(cycle=cycle, office=office, file=file)
            writer.write(f"{cycle}_{office}.csv", self.get(url).content, url=url, cycle=cycle)

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        frames = []
        for p in snapshot.files():
            cycle_str, office = p.stem.split("_", 1)
            frames.append(parse_ev_csv(p, cycle=int(cycle_str), race_type=OFFICES[office][0]))
        return pd.concat(frames, ignore_index=True)


def parse_ev_csv(path, *, cycle: int, race_type: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = raw[~raw["Pollster"].str.startswith("Election")]
    day = pd.to_numeric(raw["Day"], errors="coerce")
    half = (pd.to_numeric(raw["Len"], errors="coerce").fillna(1) - 1) / 2
    jan1 = pd.Timestamp(f"{cycle}-01-01")
    start = jan1 + pd.to_timedelta(day - half - 1, unit="D")
    end = jan1 + pd.to_timedelta(day + half - 1, unit="D")
    pollster = raw["Pollster"].str.replace(r"-\d+$", "", regex=True)
    poll_id = [
        "ev-" + hashlib.sha1(f"{cycle}|{s}|{p}|{e.date()}".encode()).hexdigest()[:12]
        for s, p, e in zip(raw["State"], pollster, end)
    ]
    base = pd.DataFrame(
        {
            "source": "electoral_vote",
            "poll_id": poll_id,
            "question_id": poll_id,
            "pollster": pollster,
            "race_type": race_type,
            "cycle": cycle,
            "state": raw["State"],
            "start_date": start.dt.floor("D"),
            "end_date": end.dt.floor("D"),
            "release_date": end.dt.floor("D"),
            "release_date_imputed": True,
            "internal": False,
        }
    )
    parts = [
        base.assign(party=party, candidate=party, pct=pd.to_numeric(raw[column], errors="coerce"))
        for column, party in (("Dem", "DEM"), ("GOP", "REP"), ("Ind", "IND"))
    ]
    out = pd.concat(parts, ignore_index=True)
    return conform(out[out["pct"].notna()], POLL_COLUMNS)


def seed_results(path) -> pd.DataFrame:
    """The prior results the site seeds unpolled states with (e.g. "Election 2020").

    Columns: state, DEM, REP, IND, label. A stand-in prior until certified
    results data is loaded.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    seeds = raw[raw["Pollster"].str.startswith("Election")]
    return pd.DataFrame(
        {
            "state": seeds["State"].values,
            **{party: pd.to_numeric(seeds[col], errors="coerce").values
               for party, col in (("DEM", "Dem"), ("REP", "GOP"), ("IND", "Ind"))},
            "label": seeds["Pollster"].str.replace(r"-\d+$", "", regex=True).values,
        }
    )
