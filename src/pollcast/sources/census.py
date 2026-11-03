"""State education by race, from the Census Bureau's ACS 5-year bulk tables.

The Census API now requires a key, so this reads the keyless "table-based
summary file": one pipe-delimited file per table covering every geography,
from which only state rows (GEO_ID 0400000US..) are kept.

Feature: white non-Hispanic adults (25+) without a bachelor's degree, as a
share of all adults 25+ (C15002H and B15002).

Vintages: the ACS 5-year release for 2021 (covering 2017-2021) and 2023.
The table-based files only start in 2021, so the 2018 and 2020 backtests see
demographics measured partly after those elections; for a share that moves
a point or two per decade, that lookahead is small but real.
"""

from __future__ import annotations

from io import StringIO

import httpx
import pandas as pd

from pollcast.sources.base import USER_AGENT, Source
from pollcast.storage import Snapshot, SnapshotWriter, latest_snapshot

URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/{year}/table-based-SF/"
       "data/5YRData/acsdt5y{year}-{table}.dat")
TABLES = ("c15002h", "b15002")


class CensusACS(Source):
    name = "census_acs"
    table = "demographics"
    frozen = True
    attribution = "Demographics: US Census Bureau, American Community Survey 5-year estimates."

    def fetch(self, writer: SnapshotWriter) -> None:
        for year in self.settings.get("vintages", [2021, 2023]):
            for table in TABLES:
                url = URL.format(year=year, table=table)
                keep = []
                # Stream: the files hold every geography (up to ~130 MB); keep the header and states.
                with httpx.stream("GET", url, timeout=300, headers={"User-Agent": USER_AGENT},
                                  follow_redirects=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line.startswith("GEO_ID") or line.startswith("0400000US"):
                            keep.append(line)
                if len(keep) < 50:
                    raise RuntimeError(f"census: {url} had {len(keep) - 1} state rows")
                writer.write(f"{table}_{year}.csv", ("\n".join(keep) + "\n").encode(), url=url, year=year)

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        from pollcast.sources._538_format import STATE_NAMES

        mit = latest_snapshot(self.config.data_dir, "mit_results")
        fips = (pd.read_csv(mit.path / "president.csv", usecols=["state_fips", "state_po"])
                .drop_duplicates().set_index("state_fips")["state_po"].map(STATE_NAMES).to_dict())
        rows = []
        for year in sorted({int(p.stem.split("_")[1]) for p in snapshot.files()}):
            h = pd.read_csv(StringIO((snapshot.path / f"c15002h_{year}.csv").read_text()), sep="|")
            b = pd.read_csv(StringIO((snapshot.path / f"b15002_{year}.csv").read_text()), sep="|")
            df = h.merge(b[["GEO_ID", "B15002_E001"]], on="GEO_ID")
            noncollege_white = df["C15002H_E001"] - df["C15002H_E006"] - df["C15002H_E011"]
            for geo, share in zip(df["GEO_ID"], noncollege_white / df["B15002_E001"]):
                state = fips.get(int(geo[-2:]))
                if state:
                    rows.append({"year": year, "state": state, "noncollege_white_share": float(share)})
        return pd.DataFrame(rows)
