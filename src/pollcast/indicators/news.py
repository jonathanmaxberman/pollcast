"""News coverage of an issue moves voters beyond what polls show.

Coverage share for a keyword (articles mentioning it / all monitored
articles) from the GDELT archive pollcast keeps daily.

Not testable yet: GDELT's DOC API only serves about three months, and the
archive began on 2026-11-03, so no past cycle has coverage data. Backfilling
2018-2024 needs GDELT's raw event files or its BigQuery tables. Until then
this module is registered so it shows on the page, but it cannot be fit,
scored or applied.
"""

from __future__ import annotations

import pandas as pd

from pollcast.indicators.base import Indicator


class NewsCoverage(Indicator):
    name = "news_coverage"
    description = "Heavy news coverage of an issue moves voters beyond their polls"
    expected = "declared per keyword when the module is built"
    expected_sign = 1

    def label(self) -> str:
        keyword = self.settings.get("keyword")
        return self.description + (f" ({keyword})" if keyword else "")

    def testable(self) -> bool:
        return False

    def untestable_reason(self) -> str:
        return "no coverage history before 2026-11-03; backfill from GDELT raw files or BigQuery"

    def fit(self, races: pd.DataFrame) -> dict:
        raise NotImplementedError(self.untestable_reason())

    def effect(self, params: dict) -> float:
        return 0.0

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        return inputs.copy()
