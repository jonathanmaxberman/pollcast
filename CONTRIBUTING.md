# Contributing

Thanks for helping. The most useful contributions are new data sources and
new adjustment ideas, tested honestly.

## Setup

```bash
git clone https://github.com/jonathanmaxberman/pollcast && cd pollcast
uv sync
uv run pytest
```

## Adding an idea

1. Read [docs/writing-modules.md](docs/writing-modules.md).
2. Write the module, with its expected direction in `expected` /
   `expected_sign` **before** you run any backtest.
3. Add it to `configs/reference.yaml` with `enabled: false`.
4. Run `uv run pollcast backtest` and `uv run pollcast backtest --history`.
5. Open a pull request with the module, a test, and the results, **including
   failures**. A failed idea, logged, is a real contribution: it saves the
   next person from trying it.

If you change an idea after seeing its results, add the new version as a
separate config entry with a `post_hoc:` note rather than editing the old one.

## Pull requests

- Tests pass (`uv run pytest`); new code has a test.
- Nothing uses information from after the forecast date.
- No data files are committed; sources are fetched by code. See
  [DATA.md](DATA.md) for what may be shared.
- The record in `predictions/` is never edited.

## Reporting a problem

Open an issue with the command you ran, the config, and the output. If a
source's terms are wrong in DATA.md, an issue is very welcome.
