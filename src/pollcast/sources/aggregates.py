"""Other forecasters' published polling averages, from a local CSV.

For when poll-level data isn't available: a file listing each aggregator's
current average (for example the generic-ballot table on Wikipedia's House
elections page). Expected columns::

    source, updated, dem_pct, rep_pct, margin_dem, in_average, url

Each fetch copies the file into a dated snapshot, so the forecast that used
it can be rebuilt later. Set ``path`` in the config.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter


class AggregateAverages(Source):
    name = "aggregate_averages"
    table = "averages"
    attribution = "Published polling averages, as compiled in the configured file."

    def fetch(self, writer: SnapshotWriter) -> None:
        path = Path(self.settings.get("path", "")).expanduser()
        if not path.is_file():
            raise RuntimeError(f"aggregate_averages: no file at {path!s}; set `path` in the config")
        writer.write(f"{self.settings.get('race_type', 'generic')}.csv", path.read_bytes(),
                     url=f"file://{path}")

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        frames = []
        for p in snapshot.files():
            df = pd.read_csv(p)
            df["race_type"] = p.stem
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        if "in_average" in out:
            out = out[out["in_average"].astype(str).str.lower() == "true"]
        out["updated"] = pd.to_datetime(out["updated"], errors="coerce")
        return out
