"""Federal election days, shared by the backtest, the weather source and the forecast."""

from datetime import date, timedelta


def election_day(year: int) -> date:
    """The Tuesday after the first Monday in November."""
    first = date(year, 11, 1)
    return first + timedelta(days=(7 - first.weekday()) % 7 + 1)


# Even years since 1998 (the 538 history replay), keyed by cycle.
HISTORY_DAYS = {y: election_day(y) for y in range(1998, 2018, 2)}

ELECTION_DAYS = {
    2018: date(2018, 11, 6),
    2020: date(2020, 11, 3),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
    2026: date(2026, 11, 3),
    2028: date(2028, 11, 7),
}

# The president's party in each cycle (for indicators that move the party in power).
PRESIDENT_PARTY = {1998: "DEM", 2000: "DEM", 2002: "REP", 2004: "REP", 2006: "REP", 2008: "REP",
                   2010: "DEM", 2012: "DEM", 2014: "DEM", 2016: "DEM",
                   2018: "REP", 2020: "REP", 2022: "DEM", 2024: "DEM", 2026: "REP"}


def days_to_election(cycle: int, as_of: date) -> int:
    return max((ELECTION_DAYS[cycle] - as_of).days, 0) if cycle in ELECTION_DAYS else 0
