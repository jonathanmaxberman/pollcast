# pollcast

An open-source, reproducible toolkit for rolling your own US election
forecast. It pulls public polls on a schedule into dated, never-overwritten
snapshots, builds polling averages, and (in later milestones) backtests them
and scores extra indicators out of sample.

Everything is driven by a YAML config, so you can change sources,
assumptions and indicators without touching model code.

**Live forecast** (the best model so far; updates as the model improves):
[jonathanmaxberman.github.io/pollcast](https://jonathanmaxberman.github.io/pollcast/)

**Published record** (fixed; committed on election day, November 3, 2026, before results):
[interactive page](https://jonathanmaxberman.github.io/pollcast/predictions/2026-11-03/forecast.html) ·
[numbers for every race](predictions/2026-11-03/PREDICTION.md).

**Status: early.** Working today: snapshot storage, poll ingest from four
sources into one normalized table, a daily GDELT news archive, a simple
time-weighted average, and a Monte Carlo Senate forecast with an interactive
map. Not yet: the Bayesian model, results data, backtests, indicators.

## Quick start

```bash
uv sync
uv run pollcast fetch                 # every enabled source in configs/polls_only.yaml
uv run pollcast average --plot        # 2026 generic ballot
uv run pollcast average --cycle 2022 --as-of 2022-11-07 --start 2022-01-01 --plot
uv run pollcast build                 # write data/normalized/{polls,news}.parquet
uv run pollcast map                   # Senate + House simulations -> data/output/forecast_2026.html
```

`--as-of` rebuilds the average using only polls that had been released by
that date, so a past day's number can be reproduced exactly.

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
| `news_inflation` | Declared when built | Not testable: coverage archive began 2026-11-03 |

The House backtest needs MIT's House results, which Dataverse serves only
after a one-time form: download `1976-2024-house.tab` from
https://doi.org/10.7910/DVN/IG0UN2 and set `house_path` under
`mit_results` in the config.

## Data sources

pollcast ships fetch scripts, not data. Each user pulls data under its own
terms.

| Source | Config key | Covers | Access |
|---|---|---|---|
| 538 archive (via Internet Archive) | `fivethirtyeight_archive` | 2018 to March 2025, all race types | Free; fetched once (538 shut down) |
| FiftyPlusOne | `fiftyplusone` | 2025 onward | Subscription API key in `$FPO_API_KEY`; terms forbid redistribution |
| New York Times | `nyt` | 2025 onward: Senate, House (incl. generic ballot), governor | Free with attribution; download the CSVs into `data/inputs/nyt/` (or set links) |
| Electoral-vote.com | `electoral_vote` | Current-cycle Senate polls | Free; no sample sizes |
| MIT Election Lab | `mit_results` | Certified Senate (and House) results, 1976-2024 | Public domain (CC0); House file needs a one-time form |
| GDELT DOC 2.0 | `gdelt` | News volume and tone by keyword, last ~3 months | Free; archive daily |

## Storage

```
data/raw/<source>/<YYYY-MM-DD>/<HHMMSSZ>/<files>   raw bytes as downloaded
data/raw/<source>/<YYYY-MM-DD>/<HHMMSSZ>/_manifest.json   URLs, hashes, failures
data/normalized/*.parquet                           derived; rebuild any time
```

Query the normalized tables with DuckDB:

```bash
uv run python -c "import duckdb; print(duckdb.sql(\"select race_type, cycle, count(distinct poll_id) from 'data/normalized/polls.parquet' group by all order by all\"))"
```

## The simple average

For each day, each poll released by that day is weighted by recency
(half-life `half_life_days` from the field midpoint), by sample size
(`sqrt(n / reference_sample_size)`, capped), and down-weighted when one
pollster has many recent polls. Partisan and internal polls are excluded by
default. One population is used per poll, in `population_preference` order.
See `src/pollcast/average.py`.

## Adding a source

Subclass `pollcast.sources.base.Source`, implement `fetch(writer)` and
`normalize(snapshot)`, and register the class in `pollcast/sources/__init__.py`.
`normalize` must return rows matching `pollcast.schema.POLL_COLUMNS` (or
`NEWS_COLUMNS`), including a `release_date` so backtests stay free of
lookahead.

## Attribution

Poll data: FiveThirtyEight (ABC News); The New York Times; FiftyPlusOne;
Electoral-vote.com. News data: the GDELT Project.

## License

MIT.
