"""Election-eve backtests from 1998, from 538's pollster-ratings history.

538's raw-polls file holds every poll it rated: polls from the final 21 days
of Senate, governor, presidential and House races since 1998, each with the
certified result. That is enough to replay election eve for about 1,700
Democrat-vs-Republican races over 13 even-year cycles, against the 137
Senate races the main backtest covers. Cycles after the file (2026 on) are
rebuilt in the same shape from this project's own poll archive and the
results in `mit_results: extra_results:` (see `recent_raw`).

What it can't do: 30-90 day lead times (no earlier polls), unpolled races
(only polled races are in the file), or anything needing data from after
the cycle (the Census education shares start in 2021).

Each race's forecast is the time-weighted average of its nonpartisan polls
fielded by two days before the election, run through the same simulation:
one national error shared by every race in the cycle, plus state errors.
Release dates are not in the file, so a poll counts as public two days after
its median field date.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pollcast.average import time_weighted_average
from pollcast.config import Config
from pollcast.sources._538_format import STATE_NAMES
from pollcast.storage import latest_snapshot

TYPES = {"Sen": "senate", "Gov": "governor", "Pres": "president", "House": "house"}


from pollcast.calendar import election_day  # noqa: E402  (re-exported for callers)


def load_history(config: Config) -> pd.DataFrame:
    snap = latest_snapshot(config.data_dir, "fivethirtyeight_ratings")
    if snap is None:
        raise RuntimeError("history: run `pollcast fetch fivethirtyeight_ratings` first")
    raw = pd.read_csv(snap.path / "raw-polls.csv")
    raw = raw[~raw["type_simple"].str.contains("-P") & (raw["cand1_party"] == "DEM") & (raw["cand2_party"] == "REP")]
    raw["polldate"] = pd.to_datetime(raw["polldate"], format="mixed")
    raw["race_type"] = raw["type_simple"].str.split("-").str[0].map(TYPES)
    # type_detail is e.g. Sen-G, Sen-GS (special), Sen-GR (runoff), Sen-GSR.
    raw["special"] = raw["type_detail"].str.split("-").str[1].fillna("").str.contains("S")
    return pd.concat([raw, recent_raw(config, set(raw["year"]))], ignore_index=True)


def recent_raw(config: Config, have: set[int]) -> pd.DataFrame:
    """Cycles after 538's file, rebuilt in its shape from our own poll archive.

    Uses the results in `mit_results: extra_results:` and, like 538's file,
    only general-election Democrat-vs-Republican polls from the final 21 days
    (plus that window's national generic-ballot polls). Pollster ids are 538's
    where the source carries them, so house effects still line up.
    """
    from pollcast.backtest import extra_results
    from pollcast.pipeline import build_polls

    results = pd.concat([extra_results(config, rt) for rt in ("senate", "governor")], ignore_index=True)
    if results.empty or "dem_pct" not in results:
        return pd.DataFrame()
    results = results[~results["cycle"].isin(have) & results["dem_pct"].notna()]
    if results.empty:
        return pd.DataFrame()
    abbr = {v: k for k, v in STATE_NAMES.items()}
    polls = build_polls(config)
    keep = (polls["cycle"].isin(results["cycle"].unique()).to_numpy(dtype=bool, na_value=False)
            & (polls["stage"] == "general").to_numpy(dtype=bool, na_value=False)
            & polls["party"].isin(["DEM", "REP"]).to_numpy(dtype=bool, na_value=False))
    polls = polls[keep].copy()
    polls["state"] = polls["state"].fillna("").astype(str)
    polls = polls.assign(polldate=polls["start_date"] + (polls["end_date"] - polls["start_date"]) / 2)
    wide = (polls.groupby(["question_id", "party"], observed=True)["pct"].max().unstack("party")
            .dropna(subset=["DEM", "REP"]))
    meta = polls.drop_duplicates("question_id").set_index("question_id")
    q = meta.join(wide, how="inner").reset_index()
    rows = []
    for cycle in sorted(results["cycle"].unique()):
        eday = pd.Timestamp(election_day(int(cycle)))
        win = q[(q["cycle"] == cycle) & (q["polldate"] >= eday - pd.Timedelta(days=21)) & (q["polldate"] < eday)]
        targets = [(r.race_type, r.state, r.special, r.dem_pct - r.rep_pct)
                   for r in results[results["cycle"] == cycle].itertuples()]
        targets.append(("house", "US", False, np.nan))  # generic ballot, for the environment
        for rt, state, special, actual in targets:
            src = "generic" if rt == "house" else rt
            g = win[(win["race_type"] == src) & ((win["state"] == "") if state == "US" else (win["state"] == state))]
            if g.empty:
                continue
            loc = "US" if state == "US" else abbr.get(state, state)
            code = {"senate": "Sen", "governor": "Gov", "house": "House"}[rt]
            detail = f"{code}-G" + ("S" if special else "")
            rows.append(pd.DataFrame({
                "poll_id": g["poll_id"].astype(str), "question_id": g["question_id"].astype(str),
                "pollster": g["pollster"], "pollster_rating_id": pd.to_numeric(g["pollster_rating_id"], errors="coerce"),
                "partisan": g["partisan"].replace("", np.nan), "location": loc, "polldate": g["polldate"],
                "samplesize": g["sample_size"].astype(float), "cand1_party": "DEM", "cand2_party": "REP",
                "cand1_pct": g["DEM"].astype(float), "cand2_pct": g["REP"].astype(float),
                "margin_poll": (g["DEM"] - g["REP"]).astype(float), "margin_actual": actual,
                "year": int(cycle), "race": f"{cycle}_{detail}_{loc}", "type_simple": f"{code}-G",
                "type_detail": detail, "race_type": rt, "special": bool(special),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _two_party(polls: pd.DataFrame) -> pd.DataFrame:
    return _margins(polls).assign(margin=(polls["cand1_pct"] + polls["cand2_pct"]).astype(float).to_numpy())


def _margins(polls: pd.DataFrame) -> pd.DataFrame:
    """The poll_margins shape the averaging code expects."""
    return pd.DataFrame({
        "poll_id": polls["poll_id"].astype(str), "pollster": polls["pollster"],
        "pollster_rating_id": polls["pollster_rating_id"].astype("Int64").astype(str),
        "partisan": polls["partisan"].fillna(""), "state": polls["location"],
        "start_date": polls["polldate"], "end_date": polls["polldate"], "midpoint": polls["polldate"],
        "release_date": polls["polldate"] + pd.Timedelta(days=2),
        "sample_size": polls["samplesize"].astype(float), "population": "lv",
        "margin": polls["margin_poll"].astype(float),
    })


def eve_races(config: Config, *, house_effects: dict | None = None, accuracy: dict | None = None,
              race_types: tuple = ("senate", "governor", "president", "house")) -> pd.DataFrame:
    """One row per race: the election-eve forecast inputs and what happened.

    `house_effects` = {"bias": {pollster_rating_id: lean}} by cycle (see
    `results_anchored_effects`) subtracts each pollster's lean before averaging.
    """
    from dataclasses import replace

    from pollcast.simulate import SimulateConfig

    raw = load_history(config)
    sim = SimulateConfig(**{k: v for k, v in config.simulate.items() if k != "party_overrides"})
    cfg = replace(config.average, **sim.average)
    rows = []
    for (year, race), g in raw.groupby(["year", "race"]):
        rt = g["race_type"].iloc[0]
        loc = g["location"].iloc[0]
        # Skip odd years, the national polls, and presidential congressional districts (ME-02).
        if year % 2 or rt not in race_types or loc == "US" or (rt != "house" and "-" in loc):
            continue
        eday = election_day(int(year))
        as_of = eday - timedelta(days=1)
        m = _margins(g[g["partisan"].isna() | (g["partisan"] == "")])
        if accuracy:
            mult = accuracy.get(int(year), {})
            w = np.array([mult.get(k, 1.0) for k in m["pollster_rating_id"]], dtype=float)
            # The average weights polls by sqrt(sample size): scale n by the square.
            m["sample_size"] = m["sample_size"].fillna(cfg.reference_sample_size).to_numpy(dtype=float) * w**2
        if house_effects:
            bias = house_effects.get(int(year), {})
            lean = np.array([bias.get(k, 0.0) for k in m["pollster_rating_id"]], dtype=float)
            m["margin"] = m["margin"].to_numpy(dtype=float) - lean
        avg = time_weighted_average(m, cfg, start=as_of, end=as_of) if len(m) else m
        if avg.empty:
            continue
        last = avg.iloc[-1]
        n_eff, se = float(last["effective_n"]), (0.0 if np.isnan(last["se"]) else float(last["se"]))
        nonpartisan = g[g["partisan"].isna() | (g["partisan"] == "")]
        tp = time_weighted_average(_two_party(nonpartisan), cfg, start=as_of, end=as_of)
        # The file only has the final 21 days, so the trend looks back 10 days, not 14.
        early = as_of - timedelta(days=10)
        before = time_weighted_average(m, cfg, start=early, end=early) if len(m) else m
        state = STATE_NAMES.get(loc[:2], loc) if rt != "house" else loc
        actual_margin = float(g["margin_actual"].iloc[0])
        rows.append({
            "cycle": int(year), "race": race, "race_type": rt, "state": state, "special": bool(g["special"].iloc[0]),
            "as_of": as_of, "lead_days": 0, "basis": "polls", "challenger": "DEM",
            "mean": float(last["margin"]), "se": se, "effective_n": n_eff, "n_polls": int(last["n_polls"]),
            "two_party": float(tp["margin"].iloc[-1]) if len(tp) else np.nan,
            "trend14": float(last["margin"] - before["margin"].iloc[-1]) if len(before) else np.nan,
            "sd": float(np.sqrt(sim.state_sd**2 + se**2 + sim.few_polls_sd**2 / max(n_eff, 1))),
            "margin": actual_margin, "actual": int(actual_margin > 0),
        })
    races = pd.DataFrame(rows)
    races["env"] = races["cycle"].map(generic_ballot_eve(raw, cfg))
    return races


def generic_ballot_eve(raw: pd.DataFrame, cfg) -> dict[int, float]:
    """Each cycle's election-eve generic-ballot average (national House polls)."""
    out = {}
    gb = raw[(raw["race_type"] == "house") & (raw["location"] == "US")
             & (raw["partisan"].isna() | (raw["partisan"] == ""))]
    for year, g in gb.groupby("year"):
        as_of = election_day(int(year)) - timedelta(days=1)
        avg = time_weighted_average(_margins(g), cfg, start=as_of, end=as_of)
        if len(avg):
            out[int(year)] = float(avg["margin"].iloc[-1])
    return out


def results_anchored_effects(config: Config, cycles: list[int], **kw) -> dict[int, dict]:
    """{cycle: {pollster_rating_id: lean vs results}} from elections before each cycle."""
    from pollcast.sources import SOURCES
    from pollcast.sources.fivethirtyeight_ratings import historical_bias

    snap = latest_snapshot(config.data_dir, "fivethirtyeight_ratings")
    hist = SOURCES["fivethirtyeight_ratings"](config).normalize(snap)
    return {c: {k: v for k, v in historical_bias(hist, c, **kw).items() if not k.startswith("*")} for c in cycles}


def run_history_backtest(config: Config, n_sims: int = 5000) -> dict:
    """Baseline, fitted error sizes, and adjustments at election eve, every cycle on file."""
    import json
    from datetime import UTC, datetime

    from pollcast.backtest import (append_indicator_log, config_error_sizes, config_hash, leave_one_cycle_out,
                                   log_loss, resimulate, score_indicator, sim_for)
    from pollcast.indicators import INDICATORS

    races = eve_races(config)
    sizes, sim = config_error_sizes(config), sim_for(config, 0, n_sims=n_sims)
    races["p"] = resimulate(races, sizes, sim)
    races, fitted = leave_one_cycle_out(races, config)
    by_type = pd.DataFrame([{"race_type": t, "n": len(g), "log_loss": log_loss(g["p"], g["actual"]),
                             "log_loss_fitted_sizes": log_loss(g["p_loco"], g["actual"]),
                             "margin_miss": float((g["margin"] - g["mean"]).abs().mean())}
                            for t, g in races.groupby("race_type")])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_history_" + config.name
    out = config.data_dir / "backtests" / run_id
    out.mkdir(parents=True)
    meta = {"run_id": run_id, "race": "history_eve", "config": config.name, "config_hash": config_hash(config)}
    tested = {}
    import json as _json

    params_path = config.data_dir / "backtests" / "indicator_params.json"
    saved = _json.loads(params_path.read_text()) if params_path.exists() else {}
    for name, settings in (config.indicators or {}).items():
        settings = settings or {}
        module = settings.get("module", name)
        # Only modules whose data reach back before 2018 at election eve.
        if module not in ("undecided_stretch", "senate_fundamentals", "education_shift",
                          "education_correlation", "education_swing", "election_day_rain",
                          "economy_vs_president", "calibrated_errors", "undecided_allocation",
                          "late_momentum", "midterm_penalty", "spending_closeness", "spending_tilt"):
            continue
        # The history file only has election-eve polls: variants for earlier leads can't be tested.
        if settings.get("min_lead_days"):
            continue
        ind = INDICATORS[settings.get("module", name)](settings)
        ind.prepare(config)
        r = score_indicator(races, ind, config, name=name)
        r["table"].to_csv(out / f"indicator_{name}.csv", index=False)
        append_indicator_log(config, meta, r)
        tested[name] = r
        # The broader test can veto: an adjustment counts as passed only if it
        # passes here too (see simulate.adjustment_groups).
        saved.setdefault(name, {})["history"] = {
            "accepted": r["accepted"], "improvement": r["improvement"], "run_id": run_id,
            "cycles_improved": f"{r['cycles_improved']}/{r['cycles']}", "metric": r["metric"]}
    params_path.write_text(_json.dumps(saved, indent=2))
    effects = results_anchored_effects(config, sorted(races["cycle"].unique()))  # house effects, below
    he = eve_races(config, house_effects=effects).merge(races[["cycle", "race"]], on=["cycle", "race"])
    base = races.merge(he[["cycle", "race"]], on=["cycle", "race"])
    pb, ph = resimulate(base, sizes, sim), resimulate(he, sizes, sim)
    he_by_cycle = {int(c): log_loss(pb[base["cycle"] == c], base["actual"][base["cycle"] == c])
                   - log_loss(ph[he["cycle"] == c], he["actual"][he["cycle"] == c]) for c in sorted(base["cycle"].unique())}
    house = {"log_loss_base": log_loss(pb, base["actual"]), "log_loss_indicator": log_loss(ph, he["actual"]),
             "cycles_improved": sum(v > 0 for v in he_by_cycle.values()), "cycles": len(he_by_cycle), "by_cycle": he_by_cycle}
    races.to_parquet(out / "races.parquet", index=False)
    by_type.to_csv(out / "by_type.csv", index=False)
    (out / "run.json").write_text(json.dumps({**meta, "fitted_error_sizes": fitted, "house_effects_results": house},
                                             indent=2, default=str))
    return {"dir": out, "by_type": by_type, "fitted": fitted, "races": races, "indicators": tested, "house_effects": house}
