"""Parser for 538-format poll CSVs.

538's archive, the NYT poll database and FiftyPlusOne all publish polls in
close variants of 538's layout, so they share this parser. Two shapes exist:

* wide: one row per question with ``dem`` / ``rep`` / ``ind`` columns
  (538's generic ballot files)
* long: one row per answer with ``party`` / ``answer`` / ``pct`` columns
  (538's candidate race files)
"""

from __future__ import annotations

import pandas as pd

from pollcast.schema import POLL_COLUMNS, conform

PARTY_MAP = {"DEM": "DEM", "REP": "REP", "IND": "IND", "D": "DEM", "R": "REP", "I": "IND"}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
STATE_ABBR = {name: abbr for abbr, name in STATE_NAMES.items()}
# States with a single at-large House seat under the 2020 apportionment.
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}


def state_name(raw: str) -> str:
    """Full state name. The NYT uses postal codes, and "US" for national polls."""
    raw = raw.strip()
    if raw in ("", "US", "National"):
        return ""
    return STATE_NAMES.get(raw, raw)


def district_id(state: str, seat_number: str) -> str:
    abbr = STATE_ABBR.get(state, "")
    if not abbr or not seat_number.strip().isdigit():
        return ""
    return f"{abbr}-AL" if abbr in AT_LARGE else f"{abbr}-{int(seat_number):02d}"


CLASS = {"class i": 1, "class ii": 2, "class iii": 3}


def is_special_senate(cycle: int, seat_name: str) -> bool:
    """Whether a Senate poll is for a special election, from 538's "Class N" seat names.

    Regular elections rotate through the classes: 2018 class I, 2020 II,
    2022 III, 2024 I, and so on. A blank seat name counts as the regular race.
    """
    cls = CLASS.get((seat_name or "").strip().lower())
    if cls is None or pd.isna(cycle):
        return False
    return cls != ((int(cycle) - 2018) // 2) % 3 + 1


def race_type_for(office_type: str | None, seat_name: str | None, default: str,
                  state: str = "") -> str:
    office = (office_type or "").lower()
    # 538 marks the generic ballot with seat_name "Generic"; the NYT with state "US".
    if "house" in office and ((seat_name or "").lower() == "generic" or state.strip() == "US"):
        return "generic"
    for key in ("senate", "governor", "house", "president"):
        if key in office:
            return key
    return default


def parse_538_csv(path, *, source: str, default_race_type: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if raw.empty:
        return conform(pd.DataFrame(), POLL_COLUMNS)
    # Ranked-choice polls report each round as its own question. Keep the
    # final round, which is the head-to-head the race is decided on.
    if "ranked_choice_final" in raw.columns:
        rc = raw["ranked_choice_round"].str.strip() != ""
        raw = raw[~rc | (raw["ranked_choice_final"].str.upper() == "TRUE")]

    def col(name: str) -> pd.Series:
        return raw[name] if name in raw.columns else pd.Series([""] * len(raw), index=raw.index)

    base = pd.DataFrame(
        {
            "source": source,
            "poll_id": col("poll_id"),
            "question_id": col("question_id"),
            "pollster": col("pollster"),
            "pollster_id": col("pollster_id"),
            "sponsors": col("sponsors"),
            "race_type": [
                race_type_for(o, s, default_race_type, st)
                for o, s, st in zip(col("office_type"), col("seat_name"), col("state"))
            ],
            "cycle": pd.to_numeric(col("cycle"), errors="coerce"),
            "state": col("state").map(state_name),
            "seat_name": col("seat_name"),
            "stage": col("stage").str.lower(),
            "seat_number": col("seat_number"),  # used for `district`, then dropped
            "start_date": col("start_date"),
            "end_date": col("end_date"),
            "release_date": col("created_at"),
            "sample_size": pd.to_numeric(col("sample_size"), errors="coerce"),
            "population": col("population").str.lower(),
            "mode": col("methodology") if "methodology" in raw.columns else col("mode"),
            "partisan": col("partisan"),
            "internal": col("internal").str.lower().isin(["true", "1", "yes"]),
        }
    )
    # Long files whose rows carry party answers take precedence over wide
    # columns: FiftyPlusOne election files carry both.
    if "pct" in raw.columns and "party" in raw.columns:
        out = base.assign(
            party=col("party").str.upper().map(PARTY_MAP).fillna("other"),
            candidate=col("candidate_name") if "candidate_name" in raw.columns else col("answer"),
            pct=pd.to_numeric(col("pct"), errors="coerce"),
        )
    else:
        parts = []
        for column, party in (("dem", "DEM"), ("rep", "REP"), ("ind", "IND")):
            if column in raw.columns:
                parts.append(
                    base.assign(party=party, candidate=party,
                                pct=pd.to_numeric(raw[column], errors="coerce"))
                )
        out = pd.concat(parts, ignore_index=True)
    out = out[out["pct"].notna()]
    house = out["race_type"] == "house"
    out["district"] = ""
    out.loc[house, "district"] = [district_id(s, n) for s, n in
                                  zip(out.loc[house, "state"], out.loc[house, "seat_number"])]
    # A missing created_at means we can't know when the poll was public:
    # fall back to end_date and flag it so backtests can apply a lag.
    out["release_date_imputed"] = out["release_date"].eq("")
    out.loc[out["release_date_imputed"], "release_date"] = out["end_date"]
    return conform(out, POLL_COLUMNS)
