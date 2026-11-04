"""Monte Carlo simulation of the House.

Each district's expected margin (Democrat minus Republican) is::

    district 2024 presidential margin - national 2024 presidential margin   (its lean)
  + national generic-ballot margin                                         (the environment)
  ± incumbency bonus toward the incumbent's party

Each simulated election draws one national error shared by all 435 districts
and an independent error per district, both Student-t. Control takes
`control_seats` seats.

Where a district has its own nonpartisan polls, their time-weighted
average is blended in with weight n / (n + district_poll_prior), n being
the effective number of polls. The incumbency bonus is applied whether or
not the incumbent is running again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from importlib import resources

import numpy as np
import pandas as pd

from pollcast.average import main_challenger, poll_margins, time_weighted_average
from pollcast.config import Config
from pollcast.pipeline import build_polls
from pollcast.simulate import _t_draws
from pollcast.sources import SOURCES
from pollcast.storage import latest_snapshot


@dataclass
class HouseConfig:
    cycle: int = 2026
    n_sims: int = 20000
    seed: int = 2026
    national_source: str = "auto"
    fixed_national_margin: float = 0.0
    lean_baseline_margin: float = -1.5
    incumbency_bonus: float = 1.5
    national_sd: float = 3.0
    district_sd: float = 6.0
    t_df: float = 5.0
    control_seats: int = 218
    # "auto" uses our own average only if it has a poll this recent.
    max_poll_age_days: int = 30
    # District polls count as this many polls' worth of the lean model.
    district_poll_prior: float = 3.0
    district_half_life_days: float = 21.0
    district_max_age_days: int = 365


def district_polls(config: Config, hc: HouseConfig, as_of: date) -> dict[str, dict]:
    """Time-weighted average of each district's own nonpartisan polls."""
    polls = build_polls(config, as_of=as_of)
    polls = polls[(polls["race_type"] == "house") & (polls["cycle"] == hc.cycle)
                  & (polls["district"].fillna("") != "")]
    cfg = replace(config.average, race_type="house", cycle=hc.cycle,
                  half_life_days=hc.district_half_life_days, max_age_days=hc.district_max_age_days)
    out = {}
    for district, rows in polls.groupby("district"):
        challenger, margins = main_challenger(rows, cfg)
        avg = time_weighted_average(margins, cfg, end=as_of)
        current = avg[avg["date"] == pd.Timestamp(as_of)]
        if current.empty:
            continue
        last = current.iloc[-1]
        out[district] = {
            "challenger": challenger,
            "margin": float(last["margin"]),
            "n_polls": int(last["n_polls"]),
            "effective_n": float(last["effective_n"]),
            "recent": [
                {"pollster": p.pollster, "end_date": p.end_date.strftime("%Y-%m-%d"),
                 "DEM": None if pd.isna(p.DEM) else float(p.DEM),
                 "IND": None if pd.isna(p.IND) else float(p.IND),
                 "REP": None if pd.isna(p.REP) else float(p.REP)}
                for p in margins.sort_values("end_date", ascending=False).head(4).itertuples()
            ],
        }
    return out


def national_environment(config: Config, hc: HouseConfig, as_of: date) -> dict:
    """The generic-ballot margin to center the House on, and where it came from."""
    if hc.national_source == "fixed":
        return {"margin": hc.fixed_national_margin, "source": "fixed",
                "label": "Fixed in the config (placeholder)", "details": []}

    if hc.national_source in ("auto", "polls"):
        cfg = replace(config.average, race_type="generic", cycle=hc.cycle)
        margins = poll_margins(build_polls(config, as_of=as_of), cfg)
        avg = time_weighted_average(margins, cfg, end=as_of)
        recent = margins[margins["release_date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=hc.max_poll_age_days)]
        if not avg.empty and (len(recent) or hc.national_source == "polls"):
            last = avg.iloc[-1]
            return {"margin": float(last["margin"]), "source": "polls",
                    "label": f"pollcast average of {int(last['n_polls'])} generic-ballot polls",
                    "details": []}
        if hc.national_source == "polls":
            raise RuntimeError("house: no generic-ballot polls available")

    snap = latest_snapshot(config.data_dir, "aggregate_averages", as_of)
    if snap is None:
        return {"margin": hc.fixed_national_margin, "source": "fixed",
                "label": "No polls or averages found; placeholder from the config", "details": []}
    rows = SOURCES["aggregate_averages"](config).normalize(snap)
    rows = rows[rows["race_type"] == "generic"]
    return {
        "margin": float(rows["margin_dem"].mean()),
        "source": "aggregates",
        "label": f"Mean of {len(rows)} published averages",
        "details": [
            {"source": r.source, "margin": float(r.margin_dem),
             "updated": r.updated.strftime("%Y-%m-%d") if pd.notna(r.updated) else None}
            for r in rows.itertuples()
        ],
    }


def house_inputs(config: Config, as_of: date) -> tuple[pd.DataFrame, "HouseConfig", dict]:
    """Each district's expected margin and challenger, the settings, and the national environment."""
    hc = HouseConfig(**config.house)
    snap = latest_snapshot(config.data_dir, "downballot_districts", as_of)
    if snap is None:
        raise RuntimeError("house: fetch downballot_districts first")
    d = SOURCES["downballot_districts"](config).normalize(snap).reset_index(drop=True)
    env = national_environment(config, hc, as_of)

    sign = d["incumbent_party"].map({"DEM": 1.0, "REP": -1.0}).fillna(0.0)
    d["lean"] = d["pres24_margin"] - hc.lean_baseline_margin
    d["model"] = d["lean"] + env["margin"] + hc.incumbency_bonus * sign
    dp = district_polls(config, hc, as_of)
    n_eff = d["district"].map(lambda k: dp.get(k, {}).get("effective_n", 0.0))
    d["poll_weight"] = n_eff / (n_eff + hc.district_poll_prior)
    poll_avg = d["district"].map(lambda k: dp.get(k, {}).get("margin", 0.0))
    d["expected"] = d["poll_weight"] * poll_avg + (1 - d["poll_weight"]) * d["model"]
    # Where an independent is the main opponent, margins are independent minus
    # Republican; the lean model stands in for the anti-Republican vote.
    d["challenger"] = d["district"].map(lambda k: dp.get(k, {}).get("challenger", "DEM"))
    d.attrs["district_polls"] = dp
    return d, hc, env


def house_forecast(config: Config, as_of: date | None = None) -> dict:
    as_of = as_of or datetime.now(UTC).date()
    d, hc, env = house_inputs(config, as_of)
    dp = d.attrs["district_polls"]

    rng = np.random.default_rng(hc.seed)
    national = _t_draws(rng, hc.t_df, hc.national_sd, (hc.n_sims, 1))
    local = _t_draws(rng, hc.t_df, hc.district_sd, (hc.n_sims, len(d)))
    sims = d["expected"].to_numpy() + national + local
    challenger_wins = sims > 0
    is_dem = (d["challenger"] == "DEM").to_numpy()
    dem_seats = (challenger_wins & is_dem).sum(axis=1)
    ind_seats = (challenger_wins & ~is_dem).sum(axis=1)
    seats, counts = np.unique(dem_seats, return_counts=True)
    lo, hi = np.percentile(sims, [10, 90], axis=0)

    hexes = json.loads(resources.files("pollcast.web").joinpath("house_hexes.json").read_text())
    districts = [
        {
            "id": r.district,
            "state": r.state,
            "incumbent": r.incumbent,
            "incumbent_party": r.incumbent_party,
            "challenger": r.challenger,
            "pres24": [r.pres24_dem, r.pres24_rep],
            "pres24_margin": r.pres24_margin,
            "expected": round(float(r.expected), 2),
            "model": round(float(r.model), 2),
            "polls": (dp[r.district] | {"weight": round(float(r.poll_weight), 3)}) if r.district in dp else None,
            # Chance the challenger (usually the Democrat) wins.
            "p_dem": round(float(challenger_wins[:, i].mean()), 4),
            "interval80": [round(float(lo[i]), 1), round(float(hi[i]), 1)],
            "hex": hexes["districts"].get(r.district),
        }
        for i, r in enumerate(d.itertuples())
    ]
    return {
        "race_type": "house",
        "cycle": hc.cycle,
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.name,
        "n_sims": hc.n_sims,
        "national": env,
        "assumptions": {
            "national_sd": hc.national_sd, "district_sd": hc.district_sd, "t_df": hc.t_df,
            "incumbency_bonus": hc.incumbency_bonus, "lean_baseline_margin": hc.lean_baseline_margin,
            "district_poll_prior": hc.district_poll_prior,
            "control_seats": hc.control_seats,
        },
        "control": {
            "p_dem_control": round(float((dem_seats >= hc.control_seats).mean()), 4),
            "dem_seats_hist": {int(s): round(c / hc.n_sims, 5) for s, c in zip(seats, counts)},
            "median_seats": {"DEM": float(np.median(dem_seats)),
                             "REP": float(np.median(len(d) - dem_seats - ind_seats)),
                             "IND": float(np.median(ind_seats))},
        },
        "grid": {"cols": hexes["cols"], "rows": hexes["rows"]},
        "districts": districts,
    }
