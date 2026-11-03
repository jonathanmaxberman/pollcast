"""Certified election results from the MIT Election Data + Science Lab.

Public domain (CC0), hosted on Harvard Dataverse. These are the backtests'
ground truth; they never feed a forecast except as prior results from
earlier cycles.

* Senate and President, 1976-2024: fetched automatically.
* House, 1976-2024: Dataverse asks for a one-time guestbook form before
  download, so fetch it by hand from https://doi.org/10.7910/DVN/IG0UN2 and
  set ``house_path`` in the config.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pollcast.sources._538_format import STATE_NAMES
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{id}?format=original"
SENATE_FILE_ID = 13887039  # 1976-2024-senate-state.tab, dataset doi:10.7910/DVN/PEJ5QU
PRESIDENT_FILE_ID = 13887042  # 1976-2024-president.csv, dataset doi:10.7910/DVN/42MVDX


class MITResults(Source):
    name = "mit_results"
    table = "results"
    frozen = True
    attribution = "Certified results: MIT Election Data and Science Lab (CC0)."

    def fetch(self, writer: SnapshotWriter) -> None:
        url = DATAVERSE.format(id=self.settings.get("senate_file_id", SENATE_FILE_ID))
        r = self.get(url)
        if not r.content.startswith(b"year,state"):
            raise RuntimeError(f"mit: unexpected Senate file: {r.content[:120]!r}")
        writer.write("senate.csv", r.content, url=url)
        url = DATAVERSE.format(id=self.settings.get("president_file_id", PRESIDENT_FILE_ID))
        r = self.get(url)
        if not r.content.startswith(b"year,state"):
            raise RuntimeError(f"mit: unexpected President file: {r.content[:120]!r}")
        writer.write("president.csv", r.content, url=url)
        house = self.settings.get("house_path")
        if house:
            p = Path(house).expanduser()
            if not p.is_file():
                raise RuntimeError(f"mit: no House file at {p}")
            writer.write("house.csv", p.read_bytes(), url=f"file://{p.resolve()}")

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        return senate_results(snapshot.path / "senate.csv")


def senate_results(path) -> pd.DataFrame:
    """One row per general-election Senate race.

    Columns: cycle, state, special, rep_pct, nonrep_pct (the best non-Republican
    candidate), nonrep_party (DEM or IND), margin (nonrep minus rep, points),
    nonrep_won. Races with no Republican or no non-Republican on the ballot
    are dropped, since there is nothing to forecast.
    """
    raw = pd.read_csv(path, dtype={"special": str, "writein": str})
    raw = raw[raw["stage"].str.lower() == "gen"]
    raw = raw[raw["writein"].str.lower() != "true"]
    raw["special"] = raw["special"].str.lower() == "true"
    raw["state_name"] = raw["state_po"].map(STATE_NAMES)
    raw["pct"] = 100 * raw["candidatevotes"] / raw["totalvotes"]
    # party_simplified is inconsistent (2022 has "DEMOCRATIC" candidates coded
    # OTHER), so read the detailed party too: DEMOCRATIC, DEMOCRATIC-FARMER-LABOR...
    detailed = raw["party_detailed"].fillna("").str.upper()
    simple = raw["party_simplified"].fillna("").str.upper()
    dem = (simple == "DEMOCRAT") | (detailed.str.startswith("DEMOCRAT") & (detailed != "DEMOCRATIC SOCIALIST"))
    rep = (simple == "REPUBLICAN") | detailed.str.startswith("REPUBLICAN")
    raw["party"] = "IND"
    raw.loc[dem, "party"] = "DEM"
    raw.loc[rep, "party"] = "REP"

    rows = []
    for (year, state, special), g in raw.groupby(["year", "state_name", "special"]):
        rep = g[g["party"] == "REP"].sort_values("pct", ascending=False)
        non = g[g["party"] != "REP"].sort_values("pct", ascending=False)
        if rep.empty or non.empty:
            continue
        best = non.iloc[0]
        rows.append(
            {
                "cycle": int(year),
                "state": state,
                "special": bool(special),
                "rep_pct": float(rep.iloc[0]["pct"]),
                "nonrep_pct": float(best["pct"]),
                "nonrep_party": best["party"],
                "nonrep_candidate": str(best["candidate"]).title(),
                "rep_candidate": str(rep.iloc[0]["candidate"]).title(),
            }
        )
    out = pd.DataFrame(rows)
    out["margin"] = out["nonrep_pct"] - out["rep_pct"]
    out["nonrep_won"] = out["margin"] > 0
    return out


def _parties(raw: pd.DataFrame) -> pd.Series:
    detailed = raw["party_detailed"].fillna("").str.upper()
    simple = raw["party_simplified"].fillna("").str.upper()
    dem = (simple == "DEMOCRAT") | (detailed.str.startswith("DEMOCRAT") & (detailed != "DEMOCRATIC SOCIALIST"))
    rep = (simple == "REPUBLICAN") | detailed.str.startswith("REPUBLICAN")
    return pd.Series("IND", index=raw.index).mask(dem, "DEM").mask(rep, "REP")


def president_margins(path) -> pd.DataFrame:
    """Democratic minus Republican presidential margin by state and nationally (state "").

    Columns: year, state, margin (points of the total vote).
    """
    raw = pd.read_csv(path, dtype={"writein": str})
    raw = raw[raw["writein"].str.lower() != "true"]
    raw["party"] = _parties(raw)
    raw["state_name"] = raw["state_po"].map(STATE_NAMES)
    votes = raw.groupby(["year", "state_name", "party"])["candidatevotes"].sum().unstack(fill_value=0)
    total = raw.groupby(["year", "state_name"])["totalvotes"].first()
    by_state = pd.DataFrame({"margin": 100 * (votes["DEM"] - votes["REP"]) / total}).reset_index()
    nat = votes.groupby("year").sum()
    nat_total = total.groupby("year").sum()
    national = pd.DataFrame({"year": nat.index, "state_name": "",
                             "margin": (100 * (nat["DEM"] - nat["REP"]) / nat_total).values})
    return pd.concat([by_state, national], ignore_index=True).rename(columns={"state_name": "state"})


def senate_candidates(path) -> pd.DataFrame:
    """Every general-election Senate candidate: cycle, state, special, name, party, votes, won."""
    raw = pd.read_csv(path, dtype={"special": str, "writein": str})
    raw = raw[(raw["stage"].str.lower() == "gen") & (raw["writein"].str.lower() != "true")]
    raw["special"] = raw["special"].str.lower() == "true"
    raw["party"] = _parties(raw)
    raw["state_name"] = raw["state_po"].map(STATE_NAMES)
    top = raw.groupby(["year", "state_name", "special"])["candidatevotes"].transform("max")
    return pd.DataFrame({"cycle": raw["year"], "state": raw["state_name"], "special": raw["special"],
                         "name": raw["candidate"].astype(str), "party": raw["party"],
                         "votes": raw["candidatevotes"], "won": raw["candidatevotes"] == top})
