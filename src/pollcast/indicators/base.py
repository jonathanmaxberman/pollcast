"""The indicator plug-in interface.

An indicator adjusts the baseline forecast: it predicts part of the
baseline's error. Each one declares the direction of its effect before it is
tested, is fit only on training cycles, and must improve held-out log loss
with its effect in the declared direction before it joins the headline
forecast. Until then it is exploratory: scored and shown, but left out.

To add one: subclass Indicator, implement fit / effect / apply, and register
it in pollcast/indicators/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd


class Indicator(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    # Declared before testing. `expected_sign` is the sign effect() must have.
    expected: ClassVar[str]
    expected_sign: ClassVar[int]

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or {}

    def prepare(self, config) -> None:
        """Load whatever data the indicator needs (called once before fit/apply).

        Race rows passed to fit and apply carry cycle, state, lead_days and
        as_of, so an indicator can look up its feature for each race.
        """

    def label(self) -> str:
        """Description shown on the forecast page; variants may add their settings."""
        return self.description

    def testable(self) -> bool:
        """False when the data for a backtest doesn't exist yet."""
        return True

    def untestable_reason(self) -> str:
        return ""

    @abstractmethod
    def fit(self, races: pd.DataFrame) -> dict:
        """Parameters from training races (backtest rows: forecast + result)."""

    @abstractmethod
    def effect(self, params: dict) -> float:
        """One signed number summarizing the fitted effect, compared with expected_sign."""

    @abstractmethod
    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        """A copy of the race inputs with the adjustment applied to `mean`."""
