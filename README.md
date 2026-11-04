# pollcast

An open-source, reproducible toolkit for building your own US election
forecast: poll averages, a Monte Carlo simulation of the Senate, House and
governor races, backtests that replay past elections with no lookahead, and
plug-in adjustments that are scored out of sample before they can change
anything.

Everything is driven by a YAML config. Change sources, assumptions or
adjustments without touching model code, or write a module for a new idea
and let the backtests tell you whether it works.

**Live forecast** (the best model so far; updates as the model improves):
[jonathanmaxberman.github.io/pollcast](https://jonathanmaxberman.github.io/pollcast/)

**Election night** (enter results as races are called; every other race updates):
[jonathanmaxberman.github.io/pollcast/night](https://jonathanmaxberman.github.io/pollcast/night/) ·
`pollcast night` builds it

**Published record** (fixed; committed on election day, November 3, 2026, before results):
[interactive page](https://jonathanmaxberman.github.io/pollcast/predictions/2026-11-03/forecast.html) ·
[numbers for every race](predictions/2026-11-03/PREDICTION.md).

## Quick start

```bash
pip install git+https://github.com/jonathanmaxberman/pollcast   # or: uv sync, from a clone
pollcast init                   # writes configs/polls_only.yaml, polls_fundamentals.yaml, reference.yaml
pollcast fetch --config configs/polls_only.yaml
pollcast map --config configs/polls_only.yaml       # -> data/output/forecast_2026.html
```

The NYT poll files have to be downloaded by hand into `data/inputs/nyt/`
(the Times blocks automated downloads); see [DATA.md](DATA.md) for every
source, its terms, and which inputs are manual.

Other commands:

```bash
pollcast average --cycle 2022 --as-of 2022-11-07 --plot   # a generic-ballot average as it stood then
pollcast backtest                    # Senate 2018-2024 at 0-90 days out, every adjustment scored
pollcast backtest --history          # election eve 1998-2022, about 1,600 races
pollcast build                       # normalized Parquet tables, for DuckDB or pandas
```

`--as-of` anywhere rebuilds what was knowable on that date, so a past day's
forecast can be reproduced exactly.

## Extending it

- **New data or a new idea:** [docs/writing-modules.md](docs/writing-modules.md)
  covers data sources and adjustments, how they're scored, and the rules that
  keep backtests honest.
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md).
- **Data you can't fetch automatically:** [MANUAL_PROJECTS.md](MANUAL_PROJECTS.md).

## The forecast page

`pollcast map` simulates the Senate and the House 20,000 times each and
writes a self-contained HTML page with a tab per chamber: hover a state or
district for its odds, margins and recent polls.

**Senate:** each state's expected margin is its time-weighted polling
average. Each simulation draws one national polling
miss shared by every state plus a separate miss per state, both fat-tailed,
so states miss together. The error sizes live under `simulate:` in the
config and are uncalibrated until the backtest harness exists.

**House:** each district starts from its 2024 presidential margin on 2026
lines (The Downballot), relative to the national 2024 margin, shifted by the
national generic-ballot environment. That environment comes from pollcast's
own average when it has recent polls, otherwise from the published averages
in `data/inputs/generic_ballot_averages.csv` (see `aggregate_averages` in the
config). Districts sit on a hex tile map, one hex each; rebuild the layout
with `scripts/build_house_hexes.py` if apportionment changes.

## Backtests

```bash
uv run pollcast fetch mit_results     # certified results, the ground truth (once)
uv run pollcast backtest              # Senate 2018-2024, 0/7/30/60/90 days out
```

Each past cycle is replayed as of each lead time using only polls released
by then and prior results from earlier elections. The harness scores every
race (log loss, Brier score, margin error, bias, calls right) and draws a
calibration chart. It also estimates the simulation's error sizes from past
misses. Those estimates are scored leave-one-cycle-out: each cycle is
forecast with error sizes fit to the other cycles only.

Each run writes `data/backtests/<run>/` (races, scores, calibration, the
config hash) and adds a line to `data/backtests/log.csv`, so every attempt,
including failures, stays on record.

### Election-eve history, 1998-2022

```bash
uv run pollcast fetch fivethirtyeight_ratings
uv run pollcast backtest --history
```

538's pollster-ratings file has every poll it rated from the final 21 days
of Senate, governor, presidential and House races since 1998, with results.
That supports election-eve replays of about 1,600 Democrat-vs-Republican
races over 13 even-year cycles (no 30-90 day leads, no unpolled races, no
Census-based adjustments). Results from the run on 2026-11-03:

| | Senate | Governor | President (states) | House |
|---|---|---|---|---|
| Races | 375 | 279 | 294 | 657 |
| Log loss, config error sizes | 0.156 | 0.253 | 0.159 | 0.361 |
| Log loss, error sizes fit on other cycles | 0.150 | 0.253 | 0.150 | 0.361 |

| Adjustment (each cycle fit on the other 12) | Log loss | Cycles better |
|---|---|---|
| Undecided voters break to the leader | 0.2568 -> 0.2563 | 11 of 13, direction right every fold |
| Fundamentals blend | 0.2568 -> 0.2560 | 8 of 13 |
| Fundamentals blend, weighted by lead time | 0.2568 -> 0.2549 | 8 of 13 |
| Education shift (high non-college-white states) | 0.2568 -> 0.2587 | 7 of 13 (fails; margin miss still better, 4.97 -> 4.91) |
| Education correlation (seat-count score) | 3.504 -> 3.537 | 2 of 13 (fails) |
| Education correction scaled by the polled education swing | 0.2568 -> 0.2591 | 7 of 13 (fails; fitted scaling is about zero) |
| Election-day rain (relative to the cycle's average) | 0.2568 -> 0.2570 | 7 of 13 (fails; helped Republicans in 8 cycles but flipped in 2004-08 and every cycle since 2018) |
| Consumer sentiment against the president's party | 0.2568 -> 0.2602 | 4 of 13 (fails) |
| Gas prices against the president's party | 0.2568 -> 0.2577 | 4 of 13 (fails) |
| Error sizes calibrated by lead time and race type | 0.2568 -> 0.2553 | 8 of 13 (fails 2018-2024: 0 of 4) |
| Undecided voters split race by race (theta) | 0.2568 -> 0.2568 | 8 of 13 (fails; no better than the 30-day stretch) |
| Late momentum | 0.2568 -> 0.2568 | 8 of 13 (fails; late movement partly reverses, mu = -0.22) |
| Midterm penalty for the president's party | 0.2568 -> 0.2574 | 2 of 13 (fails; polls already capture it) |
| Pollster accuracy weights | 0.2568 -> 0.2563 | 7 of 13 (fails 2018-2024: 1 of 4) |
| Outside spending beyond what polls justify: closer race (FEC, 2010+) | 0.2568 -> 0.2569 | 2 of 7 spending cycles (fails at eve; on 2018-2024 it passed, 0.1927 -> 0.1870, 3 of 4, with the gain growing from +0.0008 at eve to +0.009 at 60 days: spending looks informative early and the public polls catch up) |
| Outside spending closeness, 30+ days out only (post hoc) | not testable (eve only) | 2018-2024: 0.1927 -> 0.1873, 4 of 4; with the live default 0.1403 -> 0.1380 (better in 2020 and 2024, worse in 2018 and 2022). **In the live default**; does nothing inside 30 days |
| Outside spending tilt toward one party | 0.2568 -> 0.2569 | 3 of 7 (fails; hurts at eve) |
| **Live default: undecided + lead-aware blend (+ early spending, which can't act at eve)** | **0.2568 -> 0.2545** | **10 of 13** |

Registered-vs-likely voter gap (measured within each cycle's polls) can only
be tested on 2018-2024: it passes alone (0.1927 -> 0.1908, 4 of 4) but adds
nothing on top of the live default (0.1405 vs 0.1403), so it is off.

An adjustment counts as passed only if it passes both the 2018-2024 replays
and, where its data reach back, this 1998-2022 test.
| House effects anchored to past results | 0.2572 -> 0.2638 | 5 of 13 (fails; its 2018-2024 pass didn't hold) |

### Indicators

Indicators are plug-ins under `src/pollcast/indicators/` that adjust the
baseline forecast. Each declares its expected direction before testing
(`expected`, `expected_sign`) and implements `fit` (on training cycles only),
`effect` and `apply`. List one under `indicators:` in the config and
`pollcast backtest` scores it leave-one-cycle-out against the baseline, with
the same random draws. It joins the headline forecast only when it is
enabled and passed: better held-out log loss, and the declared direction in
every fold. Every test goes into `data/backtests/indicator_log.csv`.

On the forecast page, each module with a passed variant is a group of radio
buttons: Off plus its passed variants (several config entries can share a
module via `module:`). A group starts on its passed, enabled variant with the
best held-out score, or Off. Failed and untestable ideas are never offered:
they appear in a "Tested and left out" record with their scores. The page
carries one precomputed simulation per combination of passed choices, so
switching is instant and only passes grow the number of combinations.

| Indicator | Declared | Result (Senate 2018-2024) |
|---|---|---|
| `undecided_stretch` | Leaders beat their polls (k > 1) | Not accepted: held-out log loss 0.1927 -> 0.1920, better in 2 of 4 cycles; helps 0-30 days out, hurts at 60-90 |
| `undecided_stretch_late` | Same, applied only in the last 30 days (cutoff chosen after the first try) | Passed: 0.1927 -> 0.1915, better in 3 of 4 cycles, k > 1 in every fold. Margin error grew in 3 of 4 cycles |
| `election_day_rain` | Rain helps Republicans (beta < 0); observed rain in the backtest, NWS forecast live | Not accepted: 0.1927 -> 0.1934, better in 1 of 4 cycles |
| `economy_sentiment` | Falling consumer sentiment costs the president's party | Not accepted: 0.1927 -> 0.1948, better in 1 of 4; overall sign opposite to declared |
| `economy_gas` | Rising gas prices cost the president's party | Not accepted: 0.1927 -> 0.1954, better in 1 of 4 |
| `senate_fundamentals` | Blending state lean, incumbency and the generic ballot with polls helps, most where polls are thin; b_lean > 0 | Passed: 0.1927 -> 0.1451, better in 3 of 4 cycles; margin error better in all 4; gain grows with lead time. Misses popular incumbents in hostile states (Manchin, Tester 2018) |
| `search_momentum` | A late rise in the challenger's Google search share means they beat their polls | Not accepted: 0.1927 -> 0.1925, better in 2 of 4; one fold's sign flipped |
| `education_correlation` | States with many non-college white adults miss together (judged on the seat-count score) | Passed: seat-count log loss 2.199 -> 2.191, better in 3 of 4; race-level log loss slightly worse in 3 of 4 |
| `education_shift` | Post hoc (after seeing education slopes): challengers run behind their polls in high non-college-white states | Passed: 0.1927 -> 0.1803, better in 4 of 4; mostly overlaps `senate_fundamentals` (both together 0.1439 vs 0.1451) |
| house effects, partisan polls included (`simulate.house_effects`, `partisan_weight: 1`) | Adjusting pollster and sponsor leans and keeping partisan polls helps | Failed: 0.1928 -> 0.1977, better in 1 of 4 (worst in 2020, 2022) |
| house effects, nonpartisan polls only (`partisan_weight: 0`) | Post hoc split of the above | Passed: 0.1928 -> 0.1912, better in 3 of 4 |
| house effects pooled across race types, with 538 history priors (`pooled`, `history`) | More data per pollster sharpens the leans and helps | Failed both ways: nonpartisan only 0.1928 -> 0.1934 (2 of 4); partisan included 0.1928 -> 0.2001 (0 of 4). Helps 2022, hurts 2020 and 2024, when Republican-leaning pollsters were closest |
| house effects anchored to past results (`anchor: results`), post hoc | Leans measured against actual results, not peers, so pollsters that were right aren't pulled toward a miss | Passed: nonpartisan 0.1928 -> 0.1871 (3 of 4, margin error better in all 4); partisan included 0.1914 (2 of 4, weak). Adds nothing on top of `senate_fundamentals` (0.1463 vs 0.1451) |
| `senate_fundamentals_by_lead` | Rescale the polls' side of the blend by how accurate polls were at each lead time | Passed: 0.1927 -> 0.1412 (fixed blend 0.1451), better in 4 of 4; eve 1998-2022 +0.0019 vs +0.0009 for the fixed blend, and positive whichever way the polls missed. Polls were 68-85% as noisy as assumed at every lead |
| `education_swing` | The education correction scales with how far polls show non-college-white states swinging (b < 0) | Failed: 2018-2024 0.1845 (weaker than the plain shift); 1998-2022 0.2591 (b about zero) |
| `news_inflation` | Declared when built | Not testable: coverage archive began 2026-11-03 |

The House backtest needs MIT's House results, which Dataverse serves only
after a one-time form: download `1976-2024-house.tab` from
https://doi.org/10.7910/DVN/IG0UN2 and set `house_path` under
`mit_results` in the config.

## Data

pollcast ships code that fetches data, not the data itself; each user pulls
every source under its own terms. [DATA.md](DATA.md) lists each source, its
terms, and whether it can go in the shared snapshot archive built by
`scripts/package_data.py`.

Raw files are kept as fetched under `data/raw/<source>/<date>/<time>/`, never
overwritten, with a manifest of URLs and hashes; everything downstream can
be rebuilt from them.

## Attribution

Poll data: The New York Times; FiveThirtyEight (ABC News); Electoral-vote.com;
FiftyPlusOne. Results: MIT Election Data and Science Lab; The Downballot.
Pollster history: FiveThirtyEight. Other data: GDELT, NOAA, the Federal
Reserve Bank of St. Louis (ALFRED), the US Census Bureau, the Federal
Election Commission, Google Trends.

## License

MIT.
