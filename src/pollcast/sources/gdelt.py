"""GDELT DOC 2.0 news coverage, archived daily per keyword.

The DOC API only reliably serves about the last three months, so the archive
is the point: each run saves the full trailing window, and overlapping runs
let later analysis see each day's numbers as they were first published.

For each keyword two timelines are stored as raw JSON:

* ``timelinevolraw``: matching article count per day, plus the day's total
  monitored articles (``norm``), from which coverage share is computed
* ``timelinetone``: average tone of the matching articles per day
"""

from __future__ import annotations

import json
import re
import time
from datetime import timedelta

import pandas as pd
from gdeltdoc import Filters

from pollcast.schema import NEWS_COLUMNS, conform
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

API = "https://api.gdeltproject.org/api/v2/doc/doc"
MODES = ("timelinevolraw", "timelinetone")


def _slug(keyword: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", keyword.lower()).strip("-")


class GDELT(Source):
    name = "gdelt"
    table = "news"
    attribution = "News coverage data from the GDELT Project."

    def fetch(self, writer: SnapshotWriter) -> None:
        keywords: list[str] = self.settings.get("keywords") or []
        country = self.settings.get("country", "US")
        days = int(self.settings.get("lookback_days", 89))
        end = writer.now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        start = end - timedelta(days=days)
        failures = []
        for keyword in keywords:
            filters = Filters(start_date=start, end_date=end, keyword=keyword, country=country)
            for mode in MODES:
                url = f"{API}?query={filters.query_string}&mode={mode}&format=json"
                # GDELT asks clients to keep to about one request every 5 seconds,
                # and keeps throttling for a while after a 429, so back off hard.
                time.sleep(float(self.settings.get("request_pause_seconds", 6)))
                try:
                    r = self.get(url, retries=4, pause=30)
                    if "json" not in r.headers.get("content-type", ""):
                        raise RuntimeError(r.text.strip()[:200])
                except RuntimeError as e:
                    failures.append(f"{keyword!r} {mode}: {e}")
                    continue
                writer.write(f"{_slug(keyword)}__{mode}.json", r.content, url=url,
                             keyword=keyword, mode=mode, country=country)
        # Keep a partial archive rather than none: a missed day can't be recovered later.
        if not writer.entries:
            raise RuntimeError("gdelt: every request failed: " + "; ".join(failures))
        writer.failures += failures

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        manifest = snapshot.manifest()
        fetched_at = pd.Timestamp(manifest["fetched_at"])
        series: dict[tuple[str, str], list[pd.DataFrame]] = {}
        for entry in manifest["files"]:
            payload = json.loads((snapshot.path / entry["file"]).read_text() or "{}")
            timeline = (payload.get("timeline") or [{}])[0].get("data") or []
            df = pd.DataFrame(timeline)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ").dt.floor("D")
            if entry["mode"] == "timelinevolraw":
                df = df.rename(columns={"value": "articles", "norm": "total_articles"})
            else:
                df = df.rename(columns={"value": "tone"})
            series.setdefault((entry["keyword"], entry["country"]), []).append(df)
        frames = []
        for (keyword, country), parts in series.items():
            merged = parts[0]
            for part in parts[1:]:
                merged = merged.merge(part, on="date", how="outer")
            frames.append(merged.assign(source=self.name, keyword=keyword, country=country,
                                        fetched_at=fetched_at))
        if not frames:
            return conform(pd.DataFrame(), NEWS_COLUMNS)
        return conform(pd.concat(frames, ignore_index=True), NEWS_COLUMNS)
