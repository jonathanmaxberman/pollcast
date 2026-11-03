"""Append-only storage for raw snapshots.

Layout::

    <data_dir>/raw/<source>/<YYYY-MM-DD>/<HHMMSSZ>/<file>
    <data_dir>/raw/<source>/<YYYY-MM-DD>/<HHMMSSZ>/_manifest.json

A snapshot directory is written once and never modified. Normalized tables
under <data_dir>/normalized/ are derived and can always be rebuilt from raw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path


@dataclass
class Snapshot:
    source: str
    fetched_at: datetime
    path: Path

    def files(self) -> list[Path]:
        return sorted(p for p in self.path.iterdir() if not p.name.startswith("_"))

    def manifest(self) -> dict:
        return json.loads((self.path / "_manifest.json").read_text())


class SnapshotWriter:
    """Collects files for one fetch run, then commits them as a snapshot."""

    def __init__(self, data_dir: Path, source: str, now: datetime | None = None):
        self.now = now or datetime.now(UTC)
        self.source = source
        self.path = (
            data_dir / "raw" / source / self.now.strftime("%Y-%m-%d") / self.now.strftime("%H%M%SZ")
        )
        if self.path.exists():
            raise FileExistsError(f"snapshot already exists: {self.path}")
        self.path.mkdir(parents=True)
        self.entries: list[dict] = []
        # Parts of the fetch that failed but didn't sink the whole snapshot.
        self.failures: list[str] = []

    def write(self, name: str, content: bytes, url: str | None = None, **meta) -> Path:
        target = self.path / name
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(content)
        self.entries.append(
            {
                "file": name,
                "url": url,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                **meta,
            }
        )
        return target

    def commit(self, **meta) -> Snapshot:
        manifest = {
            "source": self.source,
            "fetched_at": self.now.isoformat(),
            "files": self.entries,
            "failures": self.failures,
            **meta,
        }
        (self.path / "_manifest.json").write_text(json.dumps(manifest, indent=2))
        return Snapshot(self.source, self.now, self.path)


def list_snapshots(data_dir: Path, source: str) -> list[Snapshot]:
    root = data_dir / "raw" / source
    if not root.exists():
        return []
    snaps = []
    for manifest in root.glob("*/*/_manifest.json"):
        fetched_at = datetime.fromisoformat(json.loads(manifest.read_text())["fetched_at"])
        snaps.append(Snapshot(source, fetched_at, manifest.parent))
    return sorted(snaps, key=lambda s: s.fetched_at)


def latest_snapshot(data_dir: Path, source: str, as_of: date | None = None) -> Snapshot | None:
    """The newest committed snapshot fetched on or before `as_of` (inclusive)."""
    snaps = list_snapshots(data_dir, source)
    if as_of is not None:
        snaps = [s for s in snaps if s.fetched_at.date() <= as_of]
    return snaps[-1] if snaps else None
