"""Render forecasts as self-contained interactive HTML pages."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import httpx

# US state outlines, pre-projected to Albers USA (975x610), from the us-atlas package (ISC).
TOPOLOGY_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-albers-10m.json"


def load_topology(cache_dir: Path) -> dict:
    path = cache_dir / "us-atlas-3.0.1-states-albers-10m.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(TOPOLOGY_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return json.loads(path.read_text())


def _embed(obj) -> str:
    # Safe inside a <script> block: no "</script>" can appear in the JSON.
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def render_forecast_page(senate: dict, house: dict | None, topology: dict, *,
                         standalone: bool = True, governor: dict | None = None) -> str:
    """The Senate and House forecast page as HTML. Without `house`, the House tab is hidden.

    `standalone` adds the document wrapper a local file needs.
    """
    template = resources.files("pollcast.web").joinpath("forecast.html").read_text()
    body = (
        template.replace("/*__FORECAST__*/null", _embed(senate))
        .replace("/*__TOPOLOGY__*/null", _embed(topology))
        .replace("/*__HOUSE__*/null", _embed(house))
        .replace("/*__GOVERNOR__*/null", _embed(governor))
    )
    if not standalone:
        return body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


def render_night_page(payload: dict, *, standalone: bool = True) -> str:
    """The election-night page: results entered by hand condition the joint simulation."""
    body = resources.files("pollcast.web").joinpath("night.html").read_text().replace(
        "/*__NIGHT__*/null", _embed(payload))
    if not standalone:
        return body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
