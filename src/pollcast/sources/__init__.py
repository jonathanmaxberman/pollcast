"""Registry of data sources. Add a new source by importing it here."""

from pollcast.sources.aggregates import AggregateAverages
from pollcast.sources.alfred import ALFRED
from pollcast.sources.base import Source
from pollcast.sources.census import CensusACS
from pollcast.sources.downballot import DownballotDistricts
from pollcast.sources.electoral_vote import ElectoralVote
from pollcast.sources.fiftyplusone import FiftyPlusOne
from pollcast.sources.fivethirtyeight import FiveThirtyEightArchive
from pollcast.sources.fivethirtyeight_ratings import FiveThirtyEightRatings
from pollcast.sources.gdelt import GDELT
from pollcast.sources.google_trends import GoogleTrends
from pollcast.sources.mit import MITResults
from pollcast.sources.noaa import NOAAWeather
from pollcast.sources.nyt import NYTimes

SOURCES: dict[str, type[Source]] = {
    cls.name: cls for cls in (FiveThirtyEightArchive, FiftyPlusOne, NYTimes, ElectoralVote, GDELT,
                 DownballotDistricts, AggregateAverages, MITResults,
                 NOAAWeather, ALFRED, GoogleTrends, CensusACS, FiveThirtyEightRatings)
}

__all__ = ["SOURCES", "Source"]
