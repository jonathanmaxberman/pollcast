"""Federal election days, shared by the backtest, the weather source and the forecast."""

from datetime import date

ELECTION_DAYS = {
    2018: date(2018, 11, 6),
    2020: date(2020, 11, 3),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
    2026: date(2026, 11, 3),
    2028: date(2028, 11, 7),
}

# The president's party in each cycle (for indicators that move the party in power).
PRESIDENT_PARTY = {2018: "REP", 2020: "REP", 2022: "DEM", 2024: "DEM", 2026: "REP"}


def days_to_election(cycle: int, as_of: date) -> int:
    return max((ELECTION_DAYS[cycle] - as_of).days, 0) if cycle in ELECTION_DAYS else 0
