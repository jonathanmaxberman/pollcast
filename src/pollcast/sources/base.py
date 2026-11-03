"""The source plug-in interface.

A source knows how to pull raw data into a dated snapshot and how to turn a
snapshot into rows of a normalized table. Adding a data source means writing
one subclass and registering it in ``pollcast.sources``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx
import pandas as pd

from pollcast import __version__
from pollcast.config import Config
from pollcast.storage import Snapshot, SnapshotWriter

USER_AGENT = f"pollcast/{__version__} (+https://github.com/pollcast)"


class Source(ABC):
    name: ClassVar[str]
    table: ClassVar[str]  # which normalized table this source feeds: "polls" or "news"
    attribution: ClassVar[str] = ""
    # Frozen sources (e.g. a shut-down archive) only need fetching once.
    frozen: ClassVar[bool] = False

    def __init__(self, config: Config):
        self.config = config
        self.settings: dict[str, Any] = config.source(self.name)

    @abstractmethod
    def fetch(self, writer: SnapshotWriter) -> None:
        """Download raw files into `writer`. Raise on failure; nothing is committed."""

    @abstractmethod
    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        """Parse one snapshot into rows conforming to the source's table schema."""

    # Shared HTTP helper, with polite retries for rate-limited hosts.
    def get(self, url: str, *, params: dict | None = None, retries: int = 3, pause: float = 5.0,
            headers: dict | None = None) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = httpx.get(url, params=params, timeout=120, follow_redirects=True,
                              headers={"User-Agent": USER_AGENT, **(headers or {})})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r
            except httpx.HTTPError as e:
                last = e
                time.sleep(pause * (attempt + 1))
        raise RuntimeError(f"{self.name}: failed to fetch {url}: {last}")
