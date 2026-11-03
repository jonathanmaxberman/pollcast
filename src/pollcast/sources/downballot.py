"""The Downballot's 2024 presidential results by congressional district.

Calculated on the district lines used in 2026, including maps redrawn in
2025, with each district's incumbent. Published as a Google Sheet; fetched
through the sheet's CSV export. Data © The Downballot; cite them when using it.
"""

from __future__ import annotations

import pandas as pd

from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

SHEET = "1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0"
GID = "620838163"


class DownballotDistricts(Source):
    name = "downballot_districts"
    table = "districts"
    attribution = "2024 presidential results by congressional district: The Downballot."

    def fetch(self, writer: SnapshotWriter) -> None:
        sheet = self.settings.get("sheet_id", SHEET)
        gid = self.settings.get("gid", GID)
        url = f"https://docs.google.com/spreadsheets/d/{sheet}/export?format=csv&gid={gid}"
        r = self.get(url)
        if b"District,Incumbent" not in r.content[:400]:
            raise RuntimeError(f"downballot: unexpected sheet layout: {r.content[:200]!r}")
        writer.write("districts_2026_lines.csv", r.content, url=url)

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        return parse_downballot_csv(snapshot.files()[0])


def parse_downballot_csv(path) -> pd.DataFrame:
    """One row per district: id, state, incumbent, incumbent_party, pres24 shares and margin."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    rows = raw[raw[0].str.match(r"^[A-Z]{2}-(\d{2}|AL)$")]
    return pd.DataFrame(
        {
            "district": rows[0].values,
            "state": rows[0].str[:2].values,
            "incumbent": rows[1].values,
            "incumbent_party": rows[2].str.strip("()").map({"D": "DEM", "R": "REP", "I": "IND"}).values,
            "pres24_dem": pd.to_numeric(rows[3], errors="coerce").values,
            "pres24_rep": pd.to_numeric(rows[4], errors="coerce").values,
            # Harris minus Trump, in points.
            "pres24_margin": pd.to_numeric(rows[5], errors="coerce").values,
        }
    )
