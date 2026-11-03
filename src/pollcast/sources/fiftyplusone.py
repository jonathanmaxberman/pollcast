"""FiftyPlusOne (fiftyplusone.news) poll CSVs, 2025 onward.

Downloads need an API key from a FiftyPlusOne subscription; set it in the
environment variable named by ``api_key_env`` (default FPO_API_KEY). The
files are served from ``/api/csv?api_key=...&filename=...``.

FiftyPlusOne's terms forbid redistributing its data, so pollcast never ships
these files: each user fetches their own copy under their own key.
"""

from __future__ import annotations

import os

import pandas as pd

from pollcast.sources._538_format import parse_538_csv
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

API = "https://fiftyplusone.news/api/csv"

FILES = {
    "generic": "generic",
    "senate_general": "senate",
    "governor_general": "governor",
    "house_general": "house",
    "president_general": "president",
    "presidential_approval": "approval",
    "pollsters": None,  # reference table, not polls
}


class FiftyPlusOne(Source):
    name = "fiftyplusone"
    table = "polls"
    attribution = "Poll data from FiftyPlusOne (fiftyplusone.news)."

    def fetch(self, writer: SnapshotWriter) -> None:
        env = self.settings.get("api_key_env", "FPO_API_KEY")
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"fiftyplusone: set ${env} to your FiftyPlusOne API key")
        for stem in self.settings.get("files") or list(FILES):
            r = self.get(API, params={"api_key": key, "filename": stem})
            # Record the URL without the key, so manifests are safe to share.
            writer.write(f"{stem}.csv", r.content, url=f"{API}?filename={stem}")

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        frames = [
            parse_538_csv(p, source=self.name, default_race_type=FILES[p.stem])
            for p in snapshot.files()
            if p.suffix == ".csv" and FILES.get(p.stem)
        ]
        return pd.concat(frames, ignore_index=True)
