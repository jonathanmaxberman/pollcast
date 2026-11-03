"""Load a forecast config from YAML.

The config is the only thing a user should need to edit to change sources,
assumptions or indicators. Model code reads settings from here, never from
constants of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AverageConfig:
    race_type: str = "generic"
    cycle: int = 2026
    state: str | None = None
    half_life_days: float = 14.0
    # Polls whose midpoint is older than this (relative to the as-of date) are dropped.
    max_age_days: int = 120
    population_preference: list[str] = field(default_factory=lambda: ["lv", "rv", "v", "a"])
    exclude_partisan: bool = True
    exclude_internal: bool = True
    reference_sample_size: int = 600
    max_sample_weight: float = 2.0
    # Each pollster's weights are divided by n_recent_polls ** this exponent,
    # so a pollster releasing daily trackers cannot dominate the average.
    pollster_flooding_exponent: float = 0.5
    # For sources with no publication timestamp, assume a poll became
    # public this many days after fieldwork ended.
    release_lag_days: int = 2
    # Only polls of these stages count. Sources without a stage always count.
    stages: list[str] = field(default_factory=lambda: ["general"])
    # When several sources poll the same race, only the first listed that has
    # polls is used, so a poll reported by two sources isn't counted twice.
    source_priority: list[str] = field(default_factory=list)


@dataclass
class Config:
    name: str
    data_dir: Path
    sources: dict[str, dict[str, Any]]
    average: AverageConfig
    indicators: dict[str, dict[str, Any]]
    simulate: dict[str, Any] = field(default_factory=dict)
    house: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def source(self, name: str) -> dict[str, Any]:
        return self.sources.get(name) or {}

    def enabled_sources(self) -> list[str]:
        return [n for n, s in self.sources.items() if (s or {}).get("enabled", False)]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    data_dir = Path(raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        # Relative data dirs resolve against the current directory, so the same
        # config works for anyone who runs pollcast from the repo root.
        data_dir = Path.cwd() / data_dir
    return Config(
        name=raw.get("name", path.stem),
        data_dir=data_dir,
        sources=raw.get("sources", {}) or {},
        average=AverageConfig(**(raw.get("average") or {})),
        indicators=raw.get("indicators", {}) or {},
        simulate=raw.get("simulate", {}) or {},
        house=raw.get("house", {}) or {},
        path=path,
    )
