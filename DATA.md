# Data

pollcast ships code that fetches data, not the data itself, so each user
pulls every source under that source's own terms. Some sources can also be
shared as dated snapshots, which matters for data you can't collect after
the fact (news coverage, spending as it was reported, polls as they stood).
`scripts/package_data.py` builds that archive from your local snapshots,
including only the sources marked shareable below.

| Source (config key) | What | Terms | In the shared archive? |
|---|---|---|---|
| `fivethirtyeight_archive` | 538 polls, 2018 to March 2025 (via the Internet Archive) | CC BY 4.0 (fivethirtyeight/data) | Yes, with attribution |
| `fivethirtyeight_ratings` | 538 pollster-ratings history (every rated poll since 1998, with results) | CC BY 4.0 | Yes, with attribution |
| `nyt` | New York Times poll database, 2025 onward | Free to use with attribution to The Times | Yes, with attribution |
| `mit_results` | Certified Senate and presidential results, 1976-2024 | CC0 (public domain) | Yes |
| `gdelt` | News coverage volume and tone by keyword | Free and open, with citation | Yes |
| `noaa_weather` | Election-day precipitation (GHCN-Daily) and NWS forecasts | US government work, public domain | Yes |
| `census_acs` | Education by race by state (ACS 5-year) | US government work, public domain | Yes |
| `fec_ie` | Outside spending (independent expenditures) | Public FEC records | Yes |
| `alfred` | Gas prices (GASREGW) | Public domain | Yes (gas prices only) |
| `alfred` | Consumer sentiment (UMCSENT) | Copyright University of Michigan | No: fetch it yourself |
| `fiftyplusone` | FiftyPlusOne polls | Subscription; terms forbid redistribution | No |
| `electoral_vote` | Electoral-vote.com polls | No stated license | No: fetch it yourself |
| `downballot_districts` | 2024 presidential results by district on 2026 lines | No stated license | No: fetch it yourself |
| `google_trends` | Google search interest | Google's terms; fetched with an unofficial client | No: fetch it yourself |
| `aggregate_averages` | Your own compiled list of published averages | Yours | No |

Attribution for everything in the archive is in its `ATTRIBUTION.md`.
If you know a source's terms better than this table, please open an issue.

## Inputs you download by hand

Some sources need a file pollcast can't download for you; see
`MANUAL_PROJECTS.md` for each and what it unlocks:

- NYT poll CSVs into `data/inputs/nyt/` (the Times blocks automated downloads)
- MIT's House results (Dataverse asks for a one-time form)
- a FiftyPlusOne API key, if you subscribe

## Layout

```
data/raw/<source>/<YYYY-MM-DD>/<HHMMSSZ>/   raw files as fetched, plus _manifest.json
data/normalized/                            derived tables; rebuild any time
data/backtests/                             backtest runs and the log of every test
```
