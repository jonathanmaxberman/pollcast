"""The New York Times poll database (successor to 538's).

The Times publishes its polls in close to 538's format, free with attribution.
It does not document stable download URLs, so the files are set in the
config, either as links or as files downloaded from the Times's poll pages::

    nyt:
      enabled: true
      urls:  {senate: https://...csv}
      files: {senate: data/inputs/nyt/senate.csv, house: data/inputs/nyt/house.csv}

Keys name the file (senate, house, governor, president, other); each row's
own office_type decides its race type. The House file also carries the
national generic ballot (state "US"). Each fetch copies the files into a
dated snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pollcast.sources._538_format import parse_538_csv
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter


class NYTimes(Source):
    name = "nyt"
    table = "polls"
    attribution = "Poll data from The New York Times."

    def fetch(self, writer: SnapshotWriter) -> None:
        urls: dict[str, str] = self.settings.get("urls") or {}
        files: dict[str, str] = self.settings.get("files") or {}
        if not urls and not files:
            raise RuntimeError("nyt: no `urls` or `files` configured; see pollcast/sources/nyt.py")
        for name, url in urls.items():
            writer.write(f"{name}.csv", self.get(url).content, url=url)
        for name, path in files.items():
            p = Path(path).expanduser()
            if not p.is_file():
                raise RuntimeError(f"nyt: no file at {p}")
            writer.write(f"{name}.csv", p.read_bytes(), url=f"file://{p.resolve()}")

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        frames = []
        for p in snapshot.files():
            header = p.open().readline()
            # Approval files (president.csv) have yes/no, not head-to-head shares.
            if p.suffix != ".csv" or '"pct"' not in header and "pct," not in header:
                continue
            frames.append(parse_538_csv(p, source=self.name, default_race_type=p.stem))
        return pd.concat(frames, ignore_index=True)
