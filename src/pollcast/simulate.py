"""Monte Carlo simulation of a Senate cycle.

For each state with a race, the expected margin (challenger minus
Republican) is that state's time-weighted polling average, or a prior result
when there are no recent polls. Each simulated election draws:

* one national swing, shared by every state, so states miss together
* an independent state error per race, wider when a state has few polls

Both errors are Student-t, so big misses happen more often than a normal
distribution allows. Winning a race means a margin above zero; seats are
added to the holdovers to get chamber control.

These error sizes are assumptions in the config. They are not yet calibrated
against past cycles: the backtest harness is what will calibrate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from pollcast.average import main_challenger, poll_margins, time_weighted_average
from pollcast.config import Config
from pollcast.pipeline import build_polls
from pollcast.calendar import days_to_election
from pollcast.sources._538_format import is_special_senate
from pollcast.sources.electoral_vote import seed_results
from pollcast.storage import latest_snapshot


@dataclass
class SimulateConfig:
    race_type: str = "senate"
    cycle: int = 2026
    n_sims: int = 20000
    seed: int = 2026
    national_sd: float = 3.0
    state_sd: float = 4.0
    # Extra state error that shrinks as polls accumulate: few_polls_sd / sqrt(effective polls).
    few_polls_sd: float = 6.0
    # Error around a prior result used in place of polls.
    prior_sd: float = 9.0
    t_df: float = 5.0
    # Seats not up this cycle. Independents who caucus with a party count for it.
    seats_not_up: dict[str, int] = field(default_factory=lambda: {"DEM": 34, "REP": 31})
    # Republicans control with this many seats (50 when the vice president is a Republican).
    rep_control_seats: int = 50
    # Pollster and partisan-sponsor house effects (see HouseEffects).
    house_effects: dict = field(default_factory=dict)
    # Settings passed through to the per-state polling average.
    average: dict = field(default_factory=lambda: {"half_life_days": 21, "max_age_days": 365})
    # Fix a candidate's party where a source records it differently from how
    # they run, e.g. {"Nebraska": {"Dan Osborn": "IND"}}.
    party_overrides: dict = field(default_factory=dict)


@dataclass
class HouseEffects:
    """538-style house effects: estimate each pollster's lean, subtract it, keep the poll.

    A poll's residual is its margin minus its race's average of nonpartisan
    polls as of the forecast date. A pollster's effect is the recency-weighted
    mean residual over all its polls in the cycle, shrunk toward zero as if it
    had `prior_polls` extra polls with no lean. Partisan polls are grouped by
    pollster and sponsor party, and fall back on their sponsor party's pooled
    effect when a group has no residuals. Every adjusted poll, partisan ones
    included, then enters the average (partisan ones at `partisan_weight`).
    Uses only polls released by the forecast date.
    """

    enabled: bool = False
    prior_polls: float = 3.0
    half_life_days: float = 45.0
    partisan_weight: float = 1.0
    # Pooled: estimate leans from every race type polled this cycle, not just this one.
    pooled: bool = False
    race_types: list = field(default_factory=lambda: ["generic", "senate", "governor", "president", "house"])
    # History: shrink each pollster toward its lean in earlier elections (538's
    # raw-polls data), measured relative to other pollsters in the same races.
    history: bool = False
    history_prior_polls: float = 5.0
    history_days_out: int = 21


def _pollster_key(rating_id, pollster) -> str:
    rating_id = "" if rating_id is None or pd.isna(rating_id) else str(rating_id).strip()
    return rating_id if rating_id and rating_id != "<NA>" else f"name:{pollster}"


def _group(pollster_key, partisan) -> str:
    return f"{pollster_key}|{partisan or ''}"


def estimate_house_effects(race_margins: dict, averages: dict, he: HouseEffects, as_of: date,
                           prior: dict | None = None) -> dict:
    """{group: effect} from residuals against each race's nonpartisan average.

    `prior` maps a pollster key to a historical lean; each pollster's estimate
    is shrunk toward it (weight `prior_polls`) instead of toward zero.
    """
    prior = prior or {}
    rows = []
    for race, m in race_margins.items():
        if race not in averages or m.empty:
            continue
        age = (pd.Timestamp(as_of) - m["midpoint"]).dt.days.clip(lower=0)
        keys = [_pollster_key(i, p) for i, p in zip(m["pollster_rating_id"], m["pollster"])]
        rows.append(pd.DataFrame({"key": keys, "group": [_group(k, s) for k, s in zip(keys, m["partisan"].fillna(""))],
                                  "sponsor": m["partisan"].fillna(""),
                                  "resid": m["margin"] - averages[race],
                                  "w": 0.5 ** (age / he.half_life_days)}))
    effects = {f"{k}|": v for k, v in prior.items()}  # pollsters seen only in history
    if not rows:
        return effects
    r = pd.concat(rows)
    for g, d in r.groupby("group"):
        p = prior.get(d["key"].iloc[0], 0.0) if d["sponsor"].iloc[0] == "" else 0.0
        effects[g] = float(((d["w"] * d["resid"]).sum() + he.prior_polls * p) / (d["w"].sum() + he.prior_polls))
    # Sponsor-party fallback for partisan groups never seen against an average.
    for party, d in r[r["sponsor"] != ""].groupby("sponsor"):
        effects[f"*|{party}"] = float((d["w"] * d["resid"]).sum() / (d["w"].sum() + he.prior_polls))
    return effects


def _t_draws(rng: np.random.Generator, df: float, sd: float, size) -> np.ndarray:
    # Student-t rescaled so its standard deviation equals `sd`.
    return rng.standard_t(df, size=size) * sd * np.sqrt((df - 2) / df)


def state_inputs(config: Config, sim: SimulateConfig, as_of: date, *,
                 polls: pd.DataFrame | None = None, priors: pd.DataFrame | None = None,
                 trend_days: int = 180) -> pd.DataFrame:
    """One row per race: expected margin, its uncertainty, and what it's based on.

    `polls` defaults to everything released by `as_of`; pass a prebuilt table
    to avoid re-reading snapshots (the backtest does). It is filtered to polls
    released by `as_of` either way. `priors` (state, special, DEM, REP, IND,
    label) are fallback results for races without polls; they default to
    Electoral-vote.com's seed results. `trend_days` = 0 skips the daily trend.
    """
    if polls is None:
        polls = build_polls(config, as_of=as_of)
    cycle_polls = polls[(polls["cycle"] == sim.cycle) & (polls["release_date"].dt.normalize() <= pd.Timestamp(as_of))]
    polls = cycle_polls[cycle_polls["race_type"] == sim.race_type].copy()
    polls["special"] = [is_special_senate(sim.cycle, n) for n in polls["seat_name"].fillna("")]
    if priors is None:
        snap = latest_snapshot(config.data_dir, "electoral_vote", as_of)
        priors = (seed_results(next(p for p in snap.files() if "Senate" in p.name))
                  if snap else pd.DataFrame(columns=["state", "DEM", "REP", "IND", "label"]))
    if "special" not in priors:
        priors = priors.assign(special=False)
    for state, fixes in sim.party_overrides.items():
        for candidate, party in fixes.items():
            polls.loc[(polls["state"] == state) & (polls["candidate"] == candidate), "party"] = party
    base_avg = replace(config.average, race_type=sim.race_type, cycle=sim.cycle, **sim.average)

    races = set(zip(polls["state"], polls["special"])) | set(zip(priors["state"], priors["special"]))
    races = sorted(r for r in races if r[0])
    he = HouseEffects(**sim.house_effects)
    effects = {}
    if he.enabled:
        # First pass: each race's nonpartisan average, and all its polls, partisan included.
        race_margins, averages = {}, {}
        for state, special in races:
            race_polls = polls[(polls["state"] == state) & (polls["special"] == special)]
            cfg = replace(base_avg, state=state)
            challenger, nonpartisan = main_challenger(race_polls, cfg)
            avg = time_weighted_average(nonpartisan, cfg, start=as_of, end=as_of) if len(nonpartisan) else nonpartisan
            if len(avg):
                averages[(state, special)] = float(avg["margin"].iloc[-1])
            race_margins[(state, special)] = poll_margins(race_polls, replace(cfg, exclude_partisan=False),
                                                          challenger=challenger)
        if he.pooled:
            # Every other race polled this cycle: generic ballot, governor, president, House...
            other = cycle_polls[cycle_polls["race_type"].isin([t for t in he.race_types if t != sim.race_type])
                                & (cycle_polls["source"].isin(["nyt", "fivethirtyeight_archive", "fiftyplusone"]))]
            keys = ["race_type", "state", "district", "seat_name"]
            for key, race_polls in other.groupby([other[k].fillna("") for k in keys]):
                cfg = replace(base_avg, race_type=key[0], state=None)
                nonpartisan = poll_margins(race_polls, cfg)  # Democrat vs Republican
                if nonpartisan.empty:
                    continue
                avg = time_weighted_average(nonpartisan, cfg, start=as_of, end=as_of)
                if avg.empty:
                    continue
                averages[key] = float(avg["margin"].iloc[-1])
                race_margins[key] = poll_margins(race_polls, replace(cfg, exclude_partisan=False))
        prior = {}
        if he.history:
            snap = latest_snapshot(config.data_dir, "fivethirtyeight_ratings")
            if snap is not None:
                from pollcast.sources import SOURCES
                from pollcast.sources.fivethirtyeight_ratings import historical_leans

                hist = SOURCES["fivethirtyeight_ratings"](config).normalize(snap)
                prior = historical_leans(hist, sim.cycle, max_days_out=he.history_days_out,
                                         prior_polls=he.history_prior_polls)
        effects = estimate_house_effects(race_margins, averages, he, as_of, prior)

    rows = []
    for state, special in races:
        race_polls = polls[(polls["state"] == state) & (polls["special"] == special)]
        cfg = replace(base_avg, state=state)
        challenger, margins = main_challenger(race_polls, cfg)
        if he.enabled:
            margins = poll_margins(race_polls, replace(cfg, exclude_partisan=False), challenger=challenger)
            if len(margins):
                sponsor = margins["partisan"].fillna("")
                keys = [_pollster_key(i, p) for i, p in zip(margins["pollster_rating_id"], margins["pollster"])]
                shift = [effects.get(_group(k, s), effects.get(f"*|{s}", 0.0) if s else 0.0)
                         for k, s in zip(keys, sponsor)]
                margins = margins.assign(margin=margins["margin"] - np.array(shift),
                                         sample_size=np.where(sponsor != "", margins["sample_size"].fillna(
                                             cfg.reference_sample_size) * he.partisan_weight**2, margins["sample_size"]))
        start = pd.Timestamp(as_of) - pd.Timedelta(days=trend_days)
        avg = time_weighted_average(margins, cfg, start=start, end=as_of) if len(margins) else margins
        current = avg[avg["date"] == pd.Timestamp(as_of)] if len(avg) else avg
        prior = priors[(priors["state"] == state) & (priors["special"] == special)]
        row = {"state": state, "special": bool(special), "challenger": challenger}
        if len(current):
            last = current.iloc[-1]
            n_eff = float(last["effective_n"])
            se = 0.0 if np.isnan(last["se"]) else float(last["se"])
            row |= {
                "basis": "polls",
                "mean": float(last["margin"]),
                "se": se,
                "sd": float(np.sqrt(sim.state_sd**2 + se**2 + sim.few_polls_sd**2 / max(n_eff, 1))),
                "n_polls": int(last["n_polls"]),
                "effective_n": n_eff,
                "trend": avg[["date", "margin"]],
                "polls": margins.sort_values("end_date", ascending=False),
            }
        elif len(prior) or margins.empty:
            if prior.empty:
                continue
            p = prior.iloc[0]
            value = p.get(challenger)
            if pd.isna(value):  # e.g. an independent challenger with no prior result of their own
                value = p["DEM"]
            row |= {
                "basis": "prior",
                "prior_label": p["label"],
                "mean": float(value - p["REP"]),
                "se": 0.0,
                "sd": sim.prior_sd,
                "n_polls": 0,
                "effective_n": 0.0,
                "trend": pd.DataFrame(columns=["date", "margin"]),
                "polls": margins,
            }
        else:
            # Polls exist but all are too old to weight, and there is no prior
            # result: fall back on the latest poll, as uncertain as a prior.
            latest = margins.sort_values("end_date").iloc[-1]
            row |= {
                "basis": "stale",
                "prior_label": f"{latest['pollster']} poll ending {latest['end_date']:%b %-d, %Y}",
                "mean": float(latest["margin"]),
                "se": 0.0,
                "sd": sim.prior_sd,
                "n_polls": 0,
                "effective_n": 0.0,
                "trend": pd.DataFrame(columns=["date", "margin"]),
                "polls": margins.sort_values("end_date", ascending=False),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def simulate(inputs: pd.DataFrame, sim: SimulateConfig) -> dict:
    rng = np.random.default_rng(sim.seed)
    n, k = sim.n_sims, len(inputs)
    national = _t_draws(rng, sim.t_df, sim.national_sd, (n, 1))
    local = _t_draws(rng, sim.t_df, 1.0, (n, k)) * inputs["sd"].to_numpy()
    margins = inputs["mean"].to_numpy() + national + local
    # Extra shared error factors an indicator adds (columns named loading_*): one
    # draw per simulation, scaled per race, so loaded races miss together. Drawn
    # after the base errors, so a run without them is unchanged.
    for col in sorted(c for c in inputs.columns if str(c).startswith("loading_")):
        factor = _t_draws(rng, sim.t_df, 1.0, (n, 1))
        margins = margins + factor * inputs[col].fillna(0.0).to_numpy()
    challenger_wins = margins > 0
    wins, win_counts = np.unique(challenger_wins.sum(axis=1), return_counts=True)

    is_dem = (inputs["challenger"] == "DEM").to_numpy()
    rep_seats = sim.seats_not_up["REP"] + (~challenger_wins).sum(axis=1)
    dem_seats = sim.seats_not_up["DEM"] + (challenger_wins & is_dem).sum(axis=1)
    ind_seats = (challenger_wins & ~is_dem).sum(axis=1)
    seats, counts = np.unique(rep_seats, return_counts=True)
    return {
        "p_challenger": challenger_wins.mean(axis=0),
        "challenger_wins_hist": {int(w): c / n for w, c in zip(wins, win_counts)},
        "interval80": np.percentile(margins, [10, 90], axis=0).T,
        "p_rep_control": float((rep_seats >= sim.rep_control_seats).mean()),
        # Democrats need an outright majority; independents who don't caucus
        # with them can leave neither party in control.
        "p_dem_majority": float((dem_seats >= 51).mean()),
        "rep_seats_hist": {int(s): c / n for s, c in zip(seats, counts)},
        "median_seats": {
            "REP": float(np.median(rep_seats)),
            "DEM": float(np.median(dem_seats)),
            "IND": float(np.median(ind_seats)),
        },
    }


MAX_SCENARIOS = 512


def adjustment_groups(config: Config, sim: SimulateConfig, as_of: date) -> tuple[list[dict], list[dict]]:
    """Configured indicators, split into choices and a record of what didn't pass.

    Returns (groups, left_out). Each group is one radio-button choice: "off"
    plus every variant of that module that PASSED its backtest. Its default is
    the passed, enabled variant with the best held-out improvement, else "off".
    Modules with no passed variant get no group. Failed and untested variants
    go in `left_out`, shown as a record but never offered, so they can't be
    switched on and don't multiply the precomputed combinations.
    """
    import json

    from pollcast.indicators import INDICATORS

    path = config.data_dir / "backtests" / "indicator_params.json"
    tested = json.loads(path.read_text()) if path.exists() else {}
    groups: dict[str, dict] = {}
    for name, settings in (config.indicators or {}).items():
        settings = settings or {}
        module = settings.get("module", name)
        if module not in INDICATORS:
            continue
        ind = INDICATORS[module](settings)
        ind.prepare(config)
        group = groups.setdefault(module, {
            "module": module, "label": INDICATORS[module].description,
            "options": [{"key": "off", "label": "Off", "status": "baseline", "available": True}],
        })
        t = tested.get(name)
        option = {"key": name, "label": ind.label(), "expected": ind.expected,
                  "enabled": bool(settings.get("enabled", False)), "post_hoc": settings.get("post_hoc"),
                  "available": t is not None, "_indicator": ind, "_params": (t or {}).get("params")}
        if not ind.testable():
            option |= {"status": "untested", "available": False, "reason": ind.untestable_reason()}
        elif t is None:
            option["status"] = "untested"
        else:
            option |= {"improvement": t["improvement"], "cycles_improved": t["cycles_improved"],
                       "metric": t.get("metric", "race_log_loss"),
                       "direction_as_declared": t["direction_as_declared"], "run_id": t["run_id"],
                       "status": "passed" if t["accepted"] else "failed"}
        group["options"].append(option)
    left_out = []
    for group in groups.values():
        left_out += [{k: v for k, v in o.items() if not k.startswith("_")} | {"module": group["module"]}
                     for o in group["options"] if o["status"] in ("failed", "untested")]
        group["options"] = [o for o in group["options"] if o["status"] in ("baseline", "passed")]
        passed = [o for o in group["options"] if o["status"] == "passed" and o["enabled"]]
        group["default"] = max(passed, key=lambda o: o["improvement"])["key"] if passed else "off"
    return [g for g in groups.values() if len(g["options"]) > 1], left_out


def scenario_keys(groups: list[dict]) -> list[dict[str, str]]:
    """Every combination of choices, or, if that's too many, the default plus
    each single-group change from it."""
    import itertools

    choices = [[o["key"] for o in g["options"] if o["available"]] for g in groups]
    default = {g["module"]: g["default"] for g in groups}
    combos = list(itertools.product(*choices))
    if len(combos) <= MAX_SCENARIOS:
        return [dict(zip(default, c)) for c in combos]
    out = [default]
    for g, opts in zip(groups, choices):
        out += [default | {g["module"]: k} for k in opts if k != default[g["module"]]]
    return out


def scenario_id(choice: dict[str, str]) -> str:
    return "|".join(f"{m}={k}" for m, k in sorted(choice.items()))


def control_summary(result: dict) -> dict:
    return {
        "p_rep_control": round(result["p_rep_control"], 4),
        "p_dem_majority": round(result["p_dem_majority"], 4),
        "rep_seats_hist": {s: round(float(p), 5) for s, p in result["rep_seats_hist"].items()},
        "median_seats": result["median_seats"],
    }


def forecast(config: Config, as_of: date | None = None) -> dict:
    """Run the simulation and return a JSON-ready forecast.

    One simulation per combination of adjustment choices (`scenarios`), so the
    page can switch between them; the headline is each group's default.
    """
    sim = SimulateConfig(**config.simulate)
    as_of = as_of or datetime.now(UTC).date()
    lead = days_to_election(sim.cycle, as_of)
    inputs = state_inputs(config, sim, as_of).reset_index(drop=True).assign(
        cycle=sim.cycle, lead_days=lead, as_of=as_of)
    groups, left_out = adjustment_groups(config, sim, as_of)

    scenarios = {}
    for choice in scenario_keys(groups):
        adjusted = inputs
        for g in groups:
            option = next(o for o in g["options"] if o["key"] == choice[g["module"]])
            if option["key"] != "off":
                adjusted = option["_indicator"].apply(adjusted, option["_params"], lead)
        r = simulate(adjusted, sim)
        scenarios[scenario_id(choice)] = {
            "control": control_summary(r),
            # Per race, in `states` order: [p_challenger, mean, interval80 low, high]
            "states": [[round(float(r["p_challenger"][i]), 4), round(float(adjusted["mean"][i]), 2),
                        round(float(r["interval80"][i][0]), 1), round(float(r["interval80"][i][1]), 1)]
                       for i in range(len(adjusted))],
        }
    default_id = scenario_id({g["module"]: g["default"] for g in groups})
    headline = scenarios[default_id]
    for g in groups:
        for o in g["options"]:
            o.pop("_indicator", None)
            o.pop("_params", None)

    states = []
    for i, row in inputs.reset_index(drop=True).iterrows():
        polls = row["polls"]
        states.append(
            {
                "state": row["state"],
                "special": bool(row["special"]),
                "challenger": row["challenger"],
                "basis": row["basis"],
                "prior_label": row.get("prior_label") if row["basis"] != "polls" else None,
                "poll_mean": round(row["mean"], 2),  # before adjustments
                "mean": headline["states"][i][1],
                "sd": round(row["sd"], 2),
                "p_challenger": headline["states"][i][0],
                "interval80": headline["states"][i][2:],
                "n_polls": int(row["n_polls"]),
                "trend": [
                    [d.strftime("%Y-%m-%d"), round(float(m), 2)]
                    for d, m in zip(row["trend"]["date"], row["trend"]["margin"])
                ],
                "recent_polls": [
                    {
                        "pollster": p.pollster,
                        "end_date": p.end_date.strftime("%Y-%m-%d"),
                        **{party: (None if pd.isna(getattr(p, party)) else float(getattr(p, party)))
                           for party in ("DEM", "REP", "IND")},
                    }
                    for p in polls.head(6).itertuples()
                ],
            }
        )
    return {
        "race_type": sim.race_type,
        "cycle": sim.cycle,
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.name,
        "n_sims": sim.n_sims,
        "assumptions": {
            "national_sd": sim.national_sd, "state_sd": sim.state_sd, "t_df": sim.t_df,
            "few_polls_sd": sim.few_polls_sd, "prior_sd": sim.prior_sd,
            "seats_not_up": sim.seats_not_up, "rep_control_seats": sim.rep_control_seats,
        },
        "adjustments": groups,
        "left_out": left_out,
        "scenarios": scenarios,
        "default_scenario": default_id,
        "control": headline["control"],
        "states": states,
    }

