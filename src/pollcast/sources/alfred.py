"""Economic series as they were known at the time, from ALFRED (St. Louis Fed).

ALFRED keeps every vintage of every FRED series, so a backtest forecasting
from October 2018 sees the consumer-sentiment figure published by then, not
a later revision. No API key is needed for the graph CSV endpoint used here.

Vintage dates: each election day minus each backtest lead time, plus today.
"""

from __future__ import annotations

import re
import time
from datetime import timedelta
from io import StringIO

import pandas as pd

from pollcast.calendar import ELECTION_DAYS, HISTORY_DAYS
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_SERIES = ["UMCSENT", "GASREGW"]  # Michigan consumer sentiment; regular gas, $/gallon
DEFAULT_LEADS = [0, 7, 30, 60, 90]


class ALFRED(Source):
    name = "alfred"
    table = "economy"
    attribution = "Economic data: ALFRED, Federal Reserve Bank of St. Louis."

    def vintage_dates(self, today) -> list:
        leads = self.settings.get("leads", DEFAULT_LEADS)
        dates = {d - timedelta(days=lead) for d in ELECTION_DAYS.values() for lead in leads}
        # The 1998-2022 history replay only forecasts from election eve.
        dates |= {d - timedelta(days=1) for d in HISTORY_DAYS.values()}
        return sorted(d for d in dates | {today} if d <= today)

    def fetch(self, writer: SnapshotWriter) -> None:
        import httpx

        for series in self.settings.get("series", DEFAULT_SERIES):
            current = None
            for vintage in self.vintage_dates(writer.now.date()):
                try:
                    r = httpx.get(URL, params={"id": series, "vintage_date": vintage.isoformat()}, timeout=60,
                                  follow_redirects=True)
                    r.raise_for_status()
                    content, how = r.content, "alfred_vintage"
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    # ALFRED keeps no vintage this old. Fall back on today's FRED series cut at
                    # the vintage date: fine for series that are essentially never revised
                    # (weekly gas prices); a small lookahead for others, recorded here.
                    if current is None:
                        current = self.get(FRED, params={"id": series}).text
                    lines = current.splitlines()
                    content = ("\n".join([lines[0]] + [ln for ln in lines[1:] if ln[:10] <= vintage.isoformat()])
                               + "\n").encode()
                    how = "fred_current_truncated"
                writer.write(f"{series}__{vintage}.csv", content, url=f"{URL}?id={series}&vintage_date={vintage}",
                             series=series, vintage_date=str(vintage), method=how)
                time.sleep(0.2)

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        rows = []
        for p in snapshot.files():
            m = re.match(r"(.+)__(\d{4}-\d{2}-\d{2})\.csv$", p.name)
            if not m:
                continue
            df = pd.read_csv(StringIO(p.read_text()))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            if df.empty:
                continue
            last = df.iloc[-1]
            year_ago = df[df["date"] <= last["date"] - pd.DateOffset(years=1)]
            rows.append({
                "series": m.group(1), "vintage_date": pd.Timestamp(m.group(2)),
                "latest_date": last["date"], "latest": float(last["value"]),
                "change_12m": float(last["value"] - year_ago.iloc[-1]["value"]) if len(year_ago) else None,
            })
        return pd.DataFrame(rows)
