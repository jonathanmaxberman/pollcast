# Changelog

## 0.1.0 (unreleased)

First public version, used for the 2026 midterm forecast published on
November 3, 2026, before results.

- **Data:** 15 sources fetched into dated, never-overwritten snapshots: NYT
  and 538 polls, Electoral-vote.com, FiftyPlusOne, MIT results, The
  Downballot, 538's pollster-ratings history, GDELT, NOAA, ALFRED, Census,
  FEC outside spending, Google Trends.
- **Model:** time-weighted polling averages; a Monte Carlo simulation with a
  shared national error and fat-tailed state errors; Senate, House (435
  districts on a hex map) and governor forecasts; a self-contained
  interactive page.
- **Backtests:** Senate 2018-2024 at five lead times, and election eve
  1998-2022 for about 1,600 Senate, governor, presidential and House races,
  both with no lookahead.
- **Adjustments:** a plug-in interface with out-of-sample scoring, declared
  directions, post-hoc flags, and a log of every test. About two dozen ideas
  tested so far;
  three in the reference forecast (undecided voters in the last 30 days, a
  lead-aware fundamentals blend, and early outside spending).
- **CLI:** `init`, `fetch`, `build`, `average`, `map`, `backtest`
  (`--history`).
