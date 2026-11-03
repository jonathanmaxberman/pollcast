"""Google search interest in each Senate race's two main candidates, by state.

One query per race: the challenger (top non-Republican) against the
Republican, in that state, over the 120 days before election day (or up to
today for the current cycle). Candidates come from certified results for
past cycles and from the NYT polls' candidate names for the current one.

Google Trends scales each query to its own peak, and its numbers are
sampled, so a repeated pull can differ slightly; every pull is kept as a
dated snapshot. Only ratios within one query are used, which the scaling
doesn't change. Uses the unofficial `pytrends` client.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

import pandas as pd

from pollcast.calendar import ELECTION_DAYS
from pollcast.sources._538_format import STATE_ABBR
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter, latest_snapshot


def search_term(name: str) -> str:
    """First and last name, title case: "ANGUS S. KING, JR." -> "Angus King"."""
    from pollcast.indicators.fundamentals import SUFFIXES

    parts = [p for p in str(name).replace(",", " ").split() if p.upper() not in SUFFIXES]
    parts = [p for p in parts if len(p.rstrip(".")) > 1]  # drop middle initials
    return " ".join([parts[0], parts[-1]]).title() if len(parts) >= 2 else str(name).title()


class GoogleTrends(Source):
    name = "google_trends"
    table = "search"
    attribution = "Search interest: Google Trends."

    def races(self, today) -> pd.DataFrame:
        from pollcast.pipeline import build_polls
        from pollcast.sources.mit import senate_candidates

        rows = []
        snap = latest_snapshot(self.config.data_dir, "mit_results")
        if snap:
            c = senate_candidates(snap.path / "senate.csv")
            c = c[(c["cycle"] >= min(self.settings.get("cycles", [2018]))) & (c["name"].str.upper() != "OTHER")]
            for (cycle, state, special), g in c.groupby(["cycle", "state", "special"]):
                rep = g[g["party"] == "REP"]
                non = g[g["party"] != "REP"]
                if rep.empty or non.empty:
                    continue
                top = lambda d: d.sort_values("votes", ascending=False).iloc[0]["name"]  # noqa: E731
                rows.append({"cycle": int(cycle), "state": state, "special": bool(special),
                             "challenger": search_term(top(non)), "republican": search_term(top(rep))})
        current = max(y for y, d in ELECTION_DAYS.items() if d - timedelta(days=120) <= today)
        if not any(r["cycle"] == current for r in rows):
            polls = build_polls(self.config)
            p = polls[(polls["race_type"] == "senate") & (polls["cycle"] == current)
                      & (polls["stage"].fillna("") == "general") & (polls["source"] == "nyt")
                      # Recent polls only, so early hypothetical matchups don't pick the candidate.
                      & (polls["end_date"] >= pd.Timestamp(today) - pd.Timedelta(days=60))]
            for state, g in p.groupby("state"):
                counts = g.groupby(["party", "candidate"])["poll_id"].nunique()
                if "REP" not in counts.index.get_level_values(0):
                    continue
                rep = counts["REP"].idxmax()
                non = counts.drop("REP", level=0, errors="ignore")
                non = non[non.index.get_level_values(0).isin(["DEM", "IND"])]
                if non.empty:
                    continue
                rows.append({"cycle": current, "state": state, "special": False,
                             "challenger": search_term(non.idxmax()[1]), "republican": search_term(rep)})
        return pd.DataFrame(rows)

    def fetch(self, writer: SnapshotWriter) -> None:
        from pytrends.request import TrendReq

        today = writer.now.date()
        trends = TrendReq(hl="en-US", tz=300, timeout=(10, 30))
        failures = []
        for r in self.races(today).itertuples():
            end = min(ELECTION_DAYS[r.cycle], today)
            start = end - timedelta(days=120)
            terms = [r.challenger, r.republican]
            geo = f"US-{STATE_ABBR[r.state]}"
            for attempt in range(4):
                try:
                    trends.build_payload(terms, timeframe=f"{start} {end}", geo=geo)
                    df = trends.interest_over_time()
                    break
                except Exception as e:  # Google answers bursts with 429s; back off and retry
                    err = e
                    time.sleep(30 * (attempt + 1))
            else:
                failures.append(f"{r.cycle} {r.state}: {err}")
                continue
            payload = {"cycle": r.cycle, "state": r.state, "special": r.special, "terms": terms, "geo": geo,
                       "data": [] if df.empty else
                       [[d.strftime("%Y-%m-%d"), int(row[terms[0]]), int(row[terms[1]])] for d, row in df.iterrows()]}
            name = f"{r.cycle}_{STATE_ABBR[r.state]}{'_special' if r.special else ''}.json"
            writer.write(name, json.dumps(payload).encode(), url=f"trends:{geo}:{start}:{end}", cycle=r.cycle)
            time.sleep(float(self.settings.get("request_pause_seconds", 6)))
        writer.failures += failures
        if not writer.entries:
            raise RuntimeError("google_trends: every query failed: " + "; ".join(failures[:3]))

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        rows = []
        for p in snapshot.files():
            d = json.loads(p.read_text())
            for day, c, r in d["data"]:
                rows.append({"cycle": d["cycle"], "state": d["state"], "special": d["special"],
                             "date": pd.Timestamp(day), "challenger": c, "republican": r})
        return pd.DataFrame(rows)
