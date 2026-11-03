import json
from importlib import resources
from pathlib import Path

from pollcast.sources.downballot import parse_downballot_csv

SHEET = """SPONSORED BY GRASSROOTS ANALYTICS,,,,,,,,
Calculated by The Downballot,,,,Subscribe to our newsletter,,,,
District,Incumbent,Party,2024,,,2020,,
,,,Harris,Trump,Margin,Biden,Trump,Margin
AK-AL,Nick Begich,(R),41,55,-13,43,53,-10
PA-07,Ryan Mackenzie,(R),49,50,-1,,,
"""


def test_parse_downballot_skips_headers(tmp_path: Path):
    f = tmp_path / "d.csv"
    f.write_text(SHEET)
    d = parse_downballot_csv(f)
    assert list(d["district"]) == ["AK-AL", "PA-07"]
    assert list(d["incumbent_party"]) == ["REP", "REP"]
    assert list(d["pres24_margin"]) == [-13, -1]


def test_hex_layout_has_one_free_cell_per_district():
    layout = json.loads(resources.files("pollcast.web").joinpath("house_hexes.json").read_text())
    cells = [tuple(v) for v in layout["districts"].values()]
    assert len(cells) == 435
    assert len(set(cells)) == 435  # no two districts share a hex
