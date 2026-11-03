"""Column contracts for the normalized tables.

The poll table is long: one row per (poll question, answer). A poll that asks
both a likely-voter and a registered-voter version of the same question has
two question_ids. Margins and averages are computed downstream.
"""

from __future__ import annotations

import pandas as pd

POLL_COLUMNS: dict[str, str] = {
    "source": "string",
    "poll_id": "string",
    "question_id": "string",
    "pollster": "string",
    "pollster_id": "string",
    "pollster_rating_id": "string",  # 538's ID, shared by its archive, the NYT and its ratings
    "sponsors": "string",
    "race_type": "string",  # generic | senate | governor | house | president | approval
    "cycle": "Int64",
    "state": "string",  # full state name; empty for national
    "seat_name": "string",
    "district": "string",  # House races: e.g. PA-07, AK-AL
    "stage": "string",  # general | primary | runoff ...; empty when the source doesn't say
    "start_date": "datetime64[ns]",
    "end_date": "datetime64[ns]",
    # When the poll became public. Backtests may only see polls released by
    # the as-of date, which is what makes them free of lookahead.
    "release_date": "datetime64[ns]",
    "release_date_imputed": "boolean",
    "sample_size": "Float64",
    "population": "string",  # lv | rv | v | a
    "mode": "string",
    "partisan": "string",  # sponsoring party for partisan polls, else empty
    "internal": "boolean",
    "party": "string",  # DEM | REP | IND | other
    "candidate": "string",
    "pct": "Float64",
}

NEWS_COLUMNS: dict[str, str] = {
    "source": "string",
    "keyword": "string",
    "country": "string",
    "date": "datetime64[ns]",
    "articles": "Float64",  # matching articles that day (timelinevolraw value)
    "total_articles": "Float64",  # all monitored articles that day (timelinevolraw norm)
    "tone": "Float64",  # average tone of matching articles (timelinetone)
    "fetched_at": "datetime64[ns, UTC]",
}


def conform(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add any missing columns, order them, and cast to the contract's dtypes."""
    out = df.copy()
    for col, dtype in columns.items():
        if col not in out.columns:
            out[col] = pd.NA
        if dtype.startswith("datetime64"):
            tz = "UTC" if "UTC" in dtype else None
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed", utc=tz is not None)
        else:
            out[col] = out[col].astype(dtype)
    return out[list(columns)]
