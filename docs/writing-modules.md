# Writing a module

pollcast is built to be extended in two ways: **data sources**, which bring
new data in, and **adjustments** (indicators), which change the forecast.
Both plug in without touching the model, and an adjustment is scored out of
sample automatically before it can affect anything.

## How the pieces fit

```
fetch ──> dated raw snapshot ──> normalize ──> poll table ──> polling averages
                                                                   │
                     adjustments (fit on past cycles, applied) <───┤
                                                                   ▼
                              simulation (national + state errors) ──> forecast page
                                                                   │
                     backtests: replay past cycles with no lookahead, score
```

Three rules hold everywhere:

1. **Raw data is never overwritten.** Each fetch writes a new dated folder
   under `data/raw/<source>/`.
2. **Nothing may use information from after the forecast date.** Polls count
   from their release date; spending from when the FEC received it; economic
   data from vintages as published then.
3. **An adjustment counts only if it improves held-out accuracy.** It
   declares its expected direction before testing, is fit on training cycles
   only, and is scored on the cycles it didn't see.

## Adding a data source

Subclass `pollcast.sources.base.Source` and register it in
`pollcast/sources/__init__.py`:

```python
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

class MySource(Source):
    name = "my_source"          # config key and snapshot folder
    table = "polls"             # or another table name your adjustment reads
    attribution = "Data: Example Org."
    frozen = False              # True for archives that never change (fetched once)

    def fetch(self, writer: SnapshotWriter) -> None:
        r = self.get("https://example.org/data.csv")   # retries and backs off for you
        writer.write("data.csv", r.content, url="https://example.org/data.csv")

    def normalize(self, snapshot: Snapshot):
        ...  # return a DataFrame
```

- Settings come from the config: `self.settings` is the source's block under
  `sources:`.
- **Polls** must match `pollcast.schema.POLL_COLUMNS`. The important columns
  are `release_date` (when the poll became public: that's what makes
  backtests honest), `race_type`, `cycle`, `state`, `stage`, `partisan`,
  `party` and `pct`. If your source has no publication time, leave
  `release_date` empty and set `release_date_imputed`; pollcast then assumes
  the poll was public `release_lag_days` after fieldwork ended.
- Anything that varies over time (economic data, spending, coverage) should
  keep the date each value became known, so an adjustment can look up only
  what was known on the forecast date. See `sources/alfred.py` (vintages) and
  `sources/fec.py` (received dates) for two ways to do it.
- If a source's terms forbid redistribution, say so in its docstring and in
  `DATA.md`; pollcast never ships that data.

Then add it to a config and fetch:

```yaml
sources:
  my_source:
    enabled: true
```

```bash
uv run pollcast fetch my_source
```

## Adding an adjustment

Subclass `pollcast.indicators.base.Indicator` and register it in
`pollcast/indicators/__init__.py`. `indicators/undecided.py` is the shortest
complete example.

```python
import pandas as pd
from pollcast.indicators._shrink import shrunk_slope
from pollcast.indicators.base import Indicator

class MyAdjustment(Indicator):
    name = "my_adjustment"
    description = "What it claims, in a sentence"
    expected = "beta > 0: what you expect, declared before testing"
    expected_sign = 1                  # the sign effect() must have in every fold

    def prepare(self, config):         # optional: load your data once
        ...

    def fit(self, races: pd.DataFrame) -> dict:
        # Training cycles only. Rows are past races: the forecast as of then
        # (mean, sd, basis, n_polls, ...) and what happened (margin, actual).
        ...
        return {"beta": beta}

    def effect(self, params: dict) -> float:
        return params["beta"]           # compared with expected_sign

    def apply(self, inputs: pd.DataFrame, params: dict, lead_days: int) -> pd.DataFrame:
        out = inputs.copy()
        ...                              # change out["mean"] (and optionally sd)
        return out
```

**What a race row contains.** Each row is one race at one forecast date:
`cycle`, `state`, `special`, `race_type`, `as_of`, `lead_days`, `basis`
(`polls`, `prior` or `stale`), `mean` (expected margin, challenger minus
Republican), `sd`, `se`, `effective_n`, `n_polls`, `two_party` (polled
Democratic + Republican share), `trend14` (movement over the last 14 days),
and in training data also `margin` (actual) and `actual` (1 if the
challenger won).

**Patterns worth copying:**

- *Fit within each cycle.* A whole cycle's polls often miss together; that's
  the simulation's national error, not your effect. Subtract each cycle's
  (and lead's) average miss before fitting, as most modules here do.
- *Shrink toward no effect.* `shrunk_slope(x, y, prior_sd)` gives the
  posterior slope under a normal prior centered on zero, so an adjustment
  has to earn its size.
- *Change uncertainty, not just means.* Set `out["sd_override"]` to change
  a race's error size (`fundamentals.py` does), or add a column named
  `loading_<name>` for a new shared error factor that moves races together
  (`education.py`).
- *Say when you can't be tested.* Override `testable()` and
  `untestable_reason()` if the data doesn't exist yet (`news.py`).
- *Variants.* Several config entries can share one module with different
  settings via `module:`; each is scored and logged separately, and the
  page offers passed variants as alternatives in one choice.

## Testing it

Add it to the config, **switched off**:

```yaml
indicators:
  my_adjustment:
    enabled: false
    prior_sd: 1.0
```

Then run both backtests:

```bash
uv run pollcast backtest            # Senate 2018-2024, 0-90 days out
uv run pollcast backtest --history  # election eve 1998-2022, about 1,600 races
```

Every run appends to `data/backtests/indicator_log.csv`, passed or failed,
so the number of things tried stays visible.

**It passes** when held-out log loss improves and its effect has the declared
sign in every fold, in the 2018-2024 replays and, if its data reach back,
in the 1998-2022 history too. To use it in the forecast, set
`enabled: true`. Failed and untested adjustments are never offered on the
forecast page; they're listed under "Tested and left out".

**If you change the idea after seeing results** (a cutoff, a narrower
version), add it as a new entry with a `post_hoc:` note saying what you saw.
It's scored and shown like any other, flagged as weaker evidence.

## The forecast page

`uv run pollcast map` writes a self-contained HTML page with every passed
adjustment as a choice and every combination precomputed. Nothing else is
needed to publish it.
