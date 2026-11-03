"""Registry of indicator modules. Add a new indicator by importing it here."""

from pollcast.indicators.base import Indicator
from pollcast.indicators.economy import EconomyVsPresident
from pollcast.indicators.errors import CalibratedErrors
from pollcast.indicators.education import EducationCorrelation, EducationShift, EducationSwing
from pollcast.indicators.fundamentals import SenateFundamentals
from pollcast.indicators.news import NewsCoverage
from pollcast.indicators.polls_shape import LateMomentum, MidtermPenalty, UndecidedAllocation
from pollcast.indicators.search import SearchMomentum
from pollcast.indicators.undecided import UndecidedStretch
from pollcast.indicators.weather import ElectionDayRain

INDICATORS: dict[str, type[Indicator]] = {
    cls.name: cls for cls in (UndecidedStretch, ElectionDayRain, EconomyVsPresident, NewsCoverage,
                              SenateFundamentals, SearchMomentum, EducationCorrelation,
                              EducationShift, EducationSwing, CalibratedErrors,
                              UndecidedAllocation, LateMomentum, MidtermPenalty)
}

__all__ = ["INDICATORS", "Indicator"]
