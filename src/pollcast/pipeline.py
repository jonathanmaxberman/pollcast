"""Fetch sources into snapshots and build normalized tables from them."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pollcast.config import Config
from pollcast.schema import NEWS_COLUMNS, POLL_COLUMNS, conform
from pollcast.sources import SOURCES
from pollcast.storage import Snapshot, SnapshotWriter, latest_snapshot, list_snapshots


@dataclass
class FetchResult:
    source: str
    snapshot: Snapshot | None
    skipped: str | None = None
    error: str | None = None


def fetch(config: Config, names: list[str] | None = None, force: bool = False) -> list[FetchResult]:
    results = []
    for name in names or config.enabled_sources():
        source = SOURCES[name](config)
        if source.frozen and not force and list_snapshots(config.data_dir, name):
            results.append(FetchResult(name, None, skipped="frozen source already archived"))
            continue
        writer = SnapshotWriter(config.data_dir, name)
        try:
            source.fetch(writer)
        except Exception as e:  # one failing source must not stop the others
            shutil.rmtree(writer.path)
            results.append(FetchResult(name, None, error=str(e)))
            continue
        results.append(FetchResult(name, writer.commit(attribution=source.attribution)))
    return results


def build_polls(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """The poll table as it could have been known on `as_of`.

    Uses each source's newest snapshot fetched by `as_of`, then keeps only
    polls released by `as_of`. With as_of=None, uses everything.
    """
    frames = []
    for name in config.enabled_sources():
        cls = SOURCES[name]
        if cls.table != "polls":
            continue
        # A frozen archive was captured after the fact, but every poll in it
        # carries its own release date, so its latest snapshot is fair to use.
        snap = latest_snapshot(config.data_dir, name, None if cls.frozen else as_of)
        if snap is not None:
            frames.append(cls(config).normalize(snap))
    if not frames:
        return conform(pd.DataFrame(), POLL_COLUMNS)
    polls = pd.concat(frames, ignore_index=True)
    lag = pd.Timedelta(days=config.average.release_lag_days)
    imputed = polls["release_date_imputed"].fillna(True)
    polls.loc[imputed, "release_date"] = polls.loc[imputed, "end_date"] + lag
    if as_of is not None:
        polls = polls[polls["release_date"].dt.date <= as_of]
    return polls.reset_index(drop=True)


def build_news(config: Config) -> pd.DataFrame:
    """All archived news rows, one per (keyword, day, snapshot)."""
    frames = []
    for name in config.enabled_sources():
        cls = SOURCES[name]
        if cls.table != "news":
            continue
        source = cls(config)
        frames += [source.normalize(s) for s in list_snapshots(config.data_dir, name)]
    if not frames:
        return conform(pd.DataFrame(), NEWS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_tables(config: Config) -> dict[str, int]:
    out = config.data_dir / "normalized"
    out.mkdir(parents=True, exist_ok=True)
    polls = build_polls(config)
    news = build_news(config)
    polls.to_parquet(out / "polls.parquet", index=False)
    news.to_parquet(out / "news.parquet", index=False)
    return {"polls": len(polls), "news": len(news)}
