"""Election-day weather by state, from NOAA.

Stations: the first-order airport stations in NOAA's GHCN-Daily list (IDs
starting USW, with a WMO number), about 475 across the 50 states.

* Observed: each past election day's precipitation at every station, from
  NCEI's daily-summaries service. Used by the backtest. Since forecasts
  issued before past elections were never archived, these observations
  stand in for them, which makes any backtest gain an upper bound: a real
  forecast can only do worse than knowing the weather.
* Forecast: when an election is at most 7 days away, the National Weather
  Service's gridded precipitation forecast for that day at up to
  `forecast_stations_per_state` stations per state.

A state's value is the mean over its stations, in inches.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

import pandas as pd

from pollcast.calendar import ELECTION_DAYS, HISTORY_DAYS
from pollcast.sources._538_format import STATE_NAMES
from pollcast.sources.base import Source
from pollcast.storage import Snapshot, SnapshotWriter

STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
DAILY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NWS = "https://api.weather.gov"


def first_order_stations(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        sid, state, wmo = line[0:11], line[38:40], line[80:85].strip()
        if sid.startswith("USW") and wmo and state in STATE_NAMES:
            rows.append({"station": sid, "lat": float(line[12:20]), "lon": float(line[21:30]),
                         "state": STATE_NAMES[state], "name": line[41:71].strip()})
    return pd.DataFrame(rows)


class NOAAWeather(Source):
    name = "noaa_weather"
    table = "weather"
    attribution = "Weather: NOAA NCEI GHCN-Daily and the National Weather Service."

    def fetch(self, writer: SnapshotWriter) -> None:
        text = self.get(STATIONS_URL).text
        stations = first_order_stations(text)
        writer.write("stations.csv", stations.to_csv(index=False).encode(), url=STATIONS_URL)
        today = writer.now.date()
        ids = stations["station"].tolist()
        days = dict(ELECTION_DAYS) | (HISTORY_DAYS if self.settings.get("history", True) else {})
        for cycle, day in sorted(days.items()):
            if day >= today:
                continue
            for i in range(0, len(ids), 100):
                params = {"dataset": "daily-summaries", "stations": ",".join(ids[i:i + 100]),
                          "startDate": day.isoformat(), "endDate": day.isoformat(),
                          "dataTypes": "PRCP", "units": "standard", "format": "csv"}
                r = self.get(DAILY_URL, params=params)
                writer.write(f"observed_{day}_{i // 100}.csv", r.content, url=DAILY_URL, date=str(day))
        upcoming = [d for d in ELECTION_DAYS.values() if today <= d <= today + timedelta(days=7)]
        if upcoming:
            self._forecast(writer, stations, upcoming[0])

    def _forecast(self, writer: SnapshotWriter, stations: pd.DataFrame, day: date) -> None:
        per_state = int(self.settings.get("forecast_stations_per_state", 2))
        # Prefer the big airports: "INTL" in the name sorts first.
        stations = stations.assign(big=~stations["name"].str.contains("INTL")).sort_values(["state", "big"])
        failures = []
        for _, s in stations.groupby("state").head(per_state).iterrows():
            try:
                point = self.get(f"{NWS}/points/{s.lat:.4f},{s.lon:.4f}", retries=2, pause=2).json()
                grid = self.get(point["properties"]["forecastGridData"], retries=2, pause=2).json()
            except Exception as e:  # NWS covers only the US; one bad point mustn't sink the rest
                failures.append(f"{s.station}: {e}")
                continue
            payload = {"station": s.station, "state": s.state, "day": str(day),
                       "time_zone": point["properties"].get("timeZone"),
                       "qpf": grid["properties"].get("quantitativePrecipitation", {})}
            writer.write(f"forecast_{day}_{s.station}.json", json.dumps(payload).encode(),
                         url=point["properties"]["forecastGridData"], date=str(day))
            time.sleep(0.3)
        writer.failures += failures

    def normalize(self, snapshot: Snapshot) -> pd.DataFrame:
        return weather_table(snapshot)


def _qpf_inches_on(payload: dict, day: date) -> float | None:
    """Sum the NWS precipitation forecast (mm per period) over the local election day."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(payload.get("time_zone") or "America/New_York")
    total, seen = 0.0, False
    for v in payload["qpf"].get("values", []):
        start_s, dur = v["validTime"].split("/")
        start = datetime.fromisoformat(start_s).astimezone(tz)
        hours = pd.Timedelta(dur).total_seconds() / 3600
        # Spread each period's amount evenly over its hours; keep the election-day share.
        for h in range(int(hours)):
            t = start + timedelta(hours=h)
            if t.date() == day and v["value"] is not None:
                total += v["value"] / hours
                seen = True
    return total / 25.4 if seen else None


def weather_table(snapshot: Snapshot) -> pd.DataFrame:
    """One row per (date, state): mean precipitation in inches, and its kind."""
    stations = pd.read_csv(snapshot.path / "stations.csv")
    rows = []
    for p in snapshot.files():
        if p.name.startswith("observed_"):
            df = pd.read_csv(p)
            if df.empty or "PRCP" not in df:
                continue
            df = df.merge(stations[["station", "state"]], left_on="STATION", right_on="station")
            for (d, state), g in df.groupby(["DATE", "state"]):
                vals = pd.to_numeric(g["PRCP"], errors="coerce").dropna()
                if len(vals):
                    rows.append({"date": d, "state": state, "prcp_in": float(vals.mean()),
                                 "stations": int(len(vals)), "kind": "observed"})
        elif p.name.startswith("forecast_"):
            payload = json.loads(p.read_text())
            inches = _qpf_inches_on(payload, date.fromisoformat(payload["day"]))
            if inches is not None:
                rows.append({"date": payload["day"], "state": payload["state"], "prcp_in": inches,
                             "stations": 1, "kind": "forecast"})
    out = pd.DataFrame(rows, columns=["date", "state", "prcp_in", "stations", "kind"])
    # Forecast rows are per station: average them per state.
    return (out.groupby(["date", "state", "kind"], as_index=False)
               .agg(prcp_in=("prcp_in", "mean"), stations=("stations", "sum")))
