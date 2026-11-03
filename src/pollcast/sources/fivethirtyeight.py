"""538's poll database, frozen when 538 shut down in March 2025.

538's own download URLs now redirect to ABC News, so the files come from the
Internet Archive's capture of the final published versions. Each "historical"
file holds cycles before 2024's election; the matching current file holds
polls since. Together they cover 2018-early 2025.
"""

from __future__ import annotations

import time

import pandas as pd

from pollcast.sources._538_format import parse_538_csv
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

BASE = "https://projects.fivethirtyeight.com/polls-page/data/"
# Last Wayback capture before shutdown. `id_` asks for the original bytes.
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"
DEFAULT_TIMESTAMP = "20250306"

FILES = {
    "generic_ballot_polls_historical": "generic",
    "generic_ballot_polls": "generic",
    "senate_polls_historical": "senate",
    "senate_polls": "senate",
    "governor_polls_historical": "governor",
    "governor_polls": "governor",
    "house_polls_historical": "house",
    "house_polls": "house",
    "president_polls_historical": "president",
    "president_polls": "president",
}


class FiveThirtyEightArchive(Source):
    name = "fivethirtyeight_archive"
    table = "polls"
    frozen = True
    attribution = "Poll data collected by FiveThirtyEight (ABC News), via the Internet Archive."

    def fetch(self, writer: SnapshotWriter) -> None:
        ts = str(self.settings.get("wayback_timestamp", DEFAULT_TIMESTAMP))
        wanted = self.settings.get("files") or list(FILES)
        for stem in wanted:
            url = WAYBACK.format(ts=ts, url=f"{BASE}{stem}.csv")
            r = self.get(url, pause=10)
            if not r.content.lstrip().startswith(b"poll_id"):
                raise RuntimeError(f"{url} did not return a 538 CSV (got {r.content[:80]!r})")
            writer.write(f"{stem}.csv", r.content, url=url)
            time.sleep(3)  # the Wayback Machine rate-limits bursts

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        frames = [
            parse_538_csv(p, source=self.name, default_race_type=FILES.get(p.stem, "other"))
            for p in snapshot.files()
            if p.suffix == ".csv"
        ]
        return pd.concat(frames, ignore_index=True)
