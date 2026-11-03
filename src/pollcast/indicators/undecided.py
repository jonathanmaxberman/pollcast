"""Undecided voters break toward the poll leader.

Polling margins leave out undecided voters, and on election day they tend to
split roughly in proportion to decided support, which widens the leader's
margin. The adjustment stretches a race's polling margin by a factor k::

    adjusted margin = k x polling average

Declared before testing: k > 1 (leaders beat their polls).

k is fit separately for each lead time (more voters are undecided months
out) by regressing each race's miss on its polling margin. Both are centered
within each cycle and lead, so a cycle-wide polling miss, which the
simulation's national error already covers, doesn't leak into k. The
estimate is shrunk toward k = 1 with a normal prior of sd `prior_sd`, so the
adjustment has to earn its effect.

Applies only to races forecast from polls, not to prior results. With
`max_lead_days` set, it is fit and applied only that close to election day;
further out, k = 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pollcast.indicators.base import Indicator


class UndecidedStretch(Indicator):
    name = "undecided_stretch"
    description = "Undecided voters break toward the poll leader, widening margins"
    expected = "k > 1: the leader's margin grows on election day"
    expected_sign = 1

    def label(self) -> str:
        cutoff = self.settings.get("max_lead_days")
        return self.description + (f" (last {cutoff} days only)" if cutoff is not None else " (all lead times)")

    def fit(self, races: pd.DataFrame) -> dict:
        prior_sd = float(self.settings.get("prior_sd", 0.1))
        polled = races[races["basis"] == "polls"].copy()
        cutoff = self.settings.get("max_lead_days")
        if cutoff is not None:
            polled = polled[polled["lead_days"] <= cutoff]
        polled["miss"] = polled["margin"] - polled["mean"]
        ks = {}
        for lead, g in polled.groupby("lead_days"):
            group = g.groupby("cycle")
            x = g["mean"] - group["mean"].transform("mean")
            r = g["miss"] - group["miss"].transform("mean")
            if len(g) < 10:
                ks[int(lead)] = 1.0
                continue
            # Normal prior beta ~ N(0, prior_sd^2), residual variance from the data.
            sigma2 = float((r - r.mean()).pow(2).mean())
            beta = float((x * r).sum() / ((x**2).sum() + sigma2 / prior_sd**2))
            ks[int(lead)] = 1.0 + beta
        return {"k": ks}

    def effect(self, params: dict) -> float:
        return float(np.mean([k - 1 for k in params["k"].values()])) if params["k"] else 0.0

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        ks = {int(lead): k for lead, k in params["k"].items()}
        cutoff = self.settings.get("max_lead_days")
        if not ks or (cutoff is not None and lead_days > cutoff):
            return inputs.copy()
        nearest = min(ks, key=lambda lead: abs(lead - lead_days))
        out = inputs.copy()
        polled = out["basis"] == "polls"
        out.loc[polled, "mean"] = out.loc[polled, "mean"] * ks[nearest]
        return out
