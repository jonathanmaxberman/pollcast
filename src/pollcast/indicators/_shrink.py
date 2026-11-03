"""Shared estimator: a slope shrunk toward zero by a normal prior."""

from __future__ import annotations

import pandas as pd


def shrunk_slope(x: pd.Series, y: pd.Series, prior_sd: float, noise_sd: float | None = None) -> float:
    """Posterior mean of beta in y = beta * x + noise, with beta ~ N(0, prior_sd^2).

    noise_sd defaults to the spread of y itself.
    """
    if len(x) == 0 or (x**2).sum() == 0:
        return 0.0
    sigma2 = float(noise_sd**2) if noise_sd is not None else float((y - y.mean()).pow(2).mean())
    return float((x * y).sum() / ((x**2).sum() + sigma2 / prior_sd**2))
