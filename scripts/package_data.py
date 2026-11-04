"""Bundle the shareable data snapshots into one archive (see DATA.md).

Usage: uv run python scripts/package_data.py [--out dist/] [--data-dir data]

Includes only sources whose terms allow sharing, every dated snapshot of
each (so time-sensitive history like news coverage is kept), an index of
what's inside, and ATTRIBUTION.md. Nothing is uploaded; attach the archive
to a GitHub release yourself.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

# Source -> attribution line. Sources not listed here are never packaged.
SHAREABLE = {
    "fivethirtyeight_archive": "Poll data: FiveThirtyEight (ABC News), CC BY 4.0, via the Internet Archive.",
    "fivethirtyeight_ratings": "Pollster-ratings data: FiveThirtyEight (ABC News), CC BY 4.0.",
    "nyt": "Poll data: The New York Times.",
    "mit_results": "Election results: MIT Election Data and Science Lab, CC0.",
    "gdelt": "News coverage: The GDELT Project.",
    "noaa_weather": "Weather: NOAA NCEI GHCN-Daily and the National Weather Service (public domain).",
    "census_acs": "Demographics: US Census Bureau, American Community Survey (public domain).",
    "fec_ie": "Outside spending: Federal Election Commission (public records).",
    "alfred": "Gas prices: US Energy Information Administration via FRED/ALFRED (public domain).",
}
# Files within a shareable source that must still be left out.
EXCLUDE = {"alfred": lambda name: name.startswith("UMCSENT")}  # University of Michigan copyright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args()

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"pollcast-data-{stamp}.tar.gz"
    index = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "sources": {}}
    with tarfile.open(out, "w:gz") as tar:
        for source, credit in SHAREABLE.items():
            root = args.data_dir / "raw" / source
            if not root.exists():
                continue
            skip = EXCLUDE.get(source, lambda name: False)
            files = [p for p in sorted(root.rglob("*")) if p.is_file() and not skip(p.name)]
            for p in files:
                tar.add(p, arcname=str(Path("data") / p.relative_to(args.data_dir)))
            index["sources"][source] = {"files": len(files), "snapshots": len(list(root.glob("*/*/_manifest.json"))),
                                        "attribution": credit}
        for name, text in (("INDEX.json", json.dumps(index, indent=2)),
                           ("ATTRIBUTION.md", "# Attribution\n\n" + "\n".join(f"- {c}" for c in SHAREABLE.values()) + "\n")):
            path = args.out / name
            path.write_text(text)
            tar.add(path, arcname=name)
            path.unlink()
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB): " + ", ".join(f"{k} {v['files']} files" for k, v in index["sources"].items()))


if __name__ == "__main__":
    main()
