# Manual projects

Things pollcast can't do on its own, because they need an account, a form,
a purchase, or hand collection. Each says what it unlocks. Roughly in order
of payoff for the effort.

## Quick (under an hour)

- [ ] **Download MIT's House results.** Dataverse asks for a one-time
  guestbook form: https://doi.org/10.7910/DVN/IG0UN2, file
  `1976-2024-house.tab`. Save it as `data/inputs/mit/1976-2024-house.tab`
  and uncomment `house_path` under `mit_results` in the config.
  *Unlocks:* the House backtest (generic-ballot and district accuracy), so
  House adjustments can be scored like the Senate ones.

- [ ] **Get a free Census API key.** https://api.census.gov/data/key_signup.html,
  then `export CENSUS_API_KEY=...`.
  *Unlocks:* education-by-race shares for each past cycle (ACS 1-year from
  2005, decennial 2000), removing the lookahead in the 1998-2022 education
  tests, which now borrow 2021 shares for every year.

- [ ] **Refresh the NYT poll files** before any forecast you care about.
  The Times blocks automated downloads, so this stays manual: download the
  CSVs from the Times's poll pages into `data/inputs/nyt/` under the same
  names (`senate.csv`, `house.csv`, `governor.csv`, `president.csv`,
  `other.csv`), then run `uv run pollcast fetch nyt` and `uv run pollcast map`.

## Medium (an evening)

- [ ] **Score the 2026 record once results are certified.** MIT usually
  posts certified results months after the election; re-run
  `uv run pollcast fetch mit_results --force`, then compare
  `predictions/2026-11-03/forecast.json` with what happened. This is the
  first clean out-of-sample test of everything adopted on 2026-11-03,
  especially the post-hoc adjustments (30-day undecided cutoff, education shift).

- [ ] **List House retirements and open seats for 2026.** The House model
  gives incumbents a bonus even where they aren't running. A CSV of
  `district,incumbent_running` (from Ballotpedia or The Downballot's
  retirement tracker) fixes that.

## Big (a weekend or more)

- [ ] **Collect poll crosstabs by education and race.** For each major
  pollster's final polls in 2016-2024, the Democratic margin among
  non-college white voters, from the published crosstab PDFs (NYT/Siena,
  Marquette, Quinnipiac, etc.) or Roper iPoll. A CSV of
  `cycle,state,pollster,end_date,noncollege_white_dem,noncollege_white_rep,n`
  is enough.
  *Unlocks:* tying the education correction to how non-college whites
  actually polled, not just the state-level proxy pollcast uses now.

- [ ] **Backfill GDELT news coverage with BigQuery.** Free tier on a Google
  account; GDELT's `gdelt-bq.gdeltv2` tables go back to 2015.
  *Unlocks:* the news-coverage adjustment, which can't be tested until
  there's coverage history.

- [ ] **Request archived weather forecasts.** NOAA's National Digital
  Forecast Database archive (NCEI, via data request) has past forecasts.
  *Unlocks:* an honest weather backtest using what forecasters predicted,
  not what happened.

## Automatable (no account needed; ask Claude to build)

- **Cooperative Election Study (CES) cumulative file**, 2006-2025: Harvard
  Dataverse doi:10.7910/DVN/II2DB6, downloadable without a form. Validated
  vote by education and race for every state, so the actual non-college
  white shift by state and cycle can be measured and compared with how
  polls showed it.

## Optional purchases

- [ ] **FiftyPlusOne subscription** ($8/month or $80/year). Generic-ballot
  and approval crosstabs for 2025-26, a second current poll source, and an
  API that can be automated (set `FPO_API_KEY`, enable `fiftyplusone`).
