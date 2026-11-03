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
    # Shift registered-voter and adult polls by the cycle's measured gap from
    # likely-voter versions of the same polls: {enabled, prior_pairs}.
    population_gap: dict = field(default_factory=dict)
    # Weight polls by each pollster's past accuracy (538 history): {enabled, scale}.
    accuracy_weights: dict = field(default_factory=dict)
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
    # "consensus": leans vs other polls of the same race (538's approach).
    # "results": each pollster's past lean against actual results only, no
    # in-cycle residuals, so a pollster that was right isn't pulled toward a miss.
    anchor: str = "consensus"
    history_half_life_years: float = 8.0


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


def population_gap(polls: pd.DataFrame, prior_pairs: float = 10.0) -> float:
    """Mean (registered-voter minus likely-voter) D-R margin within the same polls.

    Uses every nonpartisan general-election poll released so far this cycle
    that asked both versions of a question, shrunk toward zero as if there
    were `prior_pairs` extra pairs with no gap.
    """
    p = polls[(polls["party"].isin(["DEM", "REP"])) & (polls["partisan"].fillna("") == "")
              & (polls["stage"].fillna("general").isin(["general", ""]))
              & (polls["population"].isin(["lv", "rv"]))]
    if p.empty:
        return 0.0
    q = p.pivot_table(index=["source", "poll_id", "question_id", "population"], columns="party",
                      values="pct", aggfunc="max").dropna()
    q["margin"] = q["DEM"] - q["REP"]
    by_pop = q.groupby(level=["source", "poll_id", "population"])["margin"].mean().unstack()
    if "lv" not in by_pop or "rv" not in by_pop:
        return 0.0
    diff = (by_pop["rv"] - by_pop["lv"]).dropna()
    return float(diff.sum() / (len(diff) + prior_pairs))


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
    if he.enabled and he.anchor == "results":
        snap = latest_snapshot(config.data_dir, "fivethirtyeight_ratings")
        if snap is not None:
            from pollcast.sources import SOURCES
            from pollcast.sources.fivethirtyeight_ratings import historical_bias

            bias = historical_bias(SOURCES["fivethirtyeight_ratings"](config).normalize(snap), sim.cycle,
                                   max_days_out=he.history_days_out, prior_polls=he.history_prior_polls,
                                   half_life_years=he.history_half_life_years)
            # The same pollster lean applies whoever sponsors the poll; sponsor-party
            # leans cover partisan polls from pollsters with no history.
            effects = {f"{k}|{sp}": v for k, v in bias.items() if not k.startswith("*")
                       for sp in ("", "DEM", "REP", "IND")}
            effects |= {k: v for k, v in bias.items() if k.startswith("*")}
    elif he.enabled:
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

    gap = None
    if (sim.population_gap or {}).get("enabled"):
        gap = population_gap(cycle_polls, float(sim.population_gap.get("prior_pairs", 10.0)))
    acc = sim.accuracy_weights or {}
    multipliers = {}
    if acc.get("enabled"):
        snap = latest_snapshot(config.data_dir, "fivethirtyeight_ratings")
        if snap is not None:
            from pollcast.sources import SOURCES
            from pollcast.sources.fivethirtyeight_ratings import accuracy_multipliers, historical_accuracy

            hist = SOURCES["fivethirtyeight_ratings"](config).normalize(snap)
            multipliers = accuracy_multipliers(historical_accuracy(hist, sim.cycle), scale=float(acc.get("scale", 2.0)))

    rows = []
    for state, special in races:
        race_polls = polls[(polls["state"] == state) & (polls["special"] == special)]
        cfg = replace(base_avg, state=state)
        challenger, margins = main_challenger(race_polls, cfg)
        if gap is not None and len(margins):
            # Margins are challenger minus Republican; the gap is measured D minus R.
            sign = 1.0 if challenger == "DEM" else 0.0
            rv = margins["population"].isin(["rv", "a", "v"]).to_numpy()
            margins = margins.assign(margin=margins["margin"] - sign * gap * rv)
        if multipliers and len(margins):
            w = np.array([multipliers.get(str(k), 1.0) for k in margins["pollster_rating_id"]], dtype=float)
            margins = margins.assign(sample_size=margins["sample_size"].fillna(cfg.reference_sample_size) * w**2)
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
            # Extra features for adjustments: the average Democratic + Republican share,
            # and how far the average moved over the previous 14 days.
            tp = time_weighted_average(margins.assign(margin=margins[challenger] + margins["REP"]),
                                       cfg, start=as_of, end=as_of)
            before = time_weighted_average(margins, cfg, start=pd.Timestamp(as_of) - pd.Timedelta(days=14),
                                           end=pd.Timestamp(as_of) - pd.Timedelta(days=14))
            row |= {
                "two_party": float(tp["margin"].iloc[-1]) if len(tp) else np.nan,
                "trend14": float(last["margin"] - before["margin"].iloc[-1]) if len(before) else np.nan,
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
    # An indicator can replace the national error size (e.g. one calibrated by lead time).
    national_sd = sim.national_sd
    if "national_sd_override" in inputs and inputs["national_sd_override"].notna().any():
        national_sd = float(inputs["national_sd_override"].dropna().iloc[0])
    national = _t_draws(rng, sim.t_df, national_sd, (n, 1))
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
        elif t is None or "accepted" not in t:  # no 2018-2024 verdict yet
            option["status"] = "untested"
        else:
            hist = t.get("history")
            option |= {"improvement": t["improvement"], "cycles_improved": t["cycles_improved"],
                       "metric": t.get("metric", "race_log_loss"),
                       "direction_as_declared": t["direction_as_declared"], "run_id": t["run_id"],
                       "history": hist,
                       # Passing needs the 2018-2024 test and, where it ran, the 1998-2022 one.
                       "status": "passed" if t["accepted"] and (hist is None or hist["accepted"]) else "failed"}
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
        "challenger_wins_hist": {int(k): round(float(v), 5) for k, v in result["challenger_wins_hist"].items()},
    }


# 2026 governor races; Hawaii and Wyoming have no public polls this cycle.
GOVERNOR_RACES_2026 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Iowa", "Kansas", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Nebraska", "Nevada", "New Hampshire", "New Mexico", "New York", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Vermont", "Wisconsin", "Wyoming",
]


def presidential_priors(config: Config, states: list[str], before_year: int) -> pd.DataFrame:
    """Fallback results for races without a prior of their own: the state's last presidential vote."""
    from pollcast.sources.mit import president_margins

    snap = latest_snapshot(config.data_dir, "mit_results")
    raw = pd.read_csv(snap.path / "president.csv", dtype={"writein": str})
    year = max(y for y in raw["year"].unique() if y < before_year)
    pres = president_margins(snap.path / "president.csv")
    m = pres[(pres["year"] == year)].set_index("state")["margin"]
    return pd.DataFrame([{"state": st, "special": False, "DEM": float(m[st]) / 2, "REP": -float(m[st]) / 2,
                          "IND": np.nan, "label": f"Presidential {year}"} for st in states if st in m.index])


def forecast(config: Config, as_of: date | None = None, race_type: str = "senate") -> dict:
    """Run the simulation and return a JSON-ready forecast.

    One simulation per combination of adjustment choices (`scenarios`), so the
    page can switch between them; the headline is each group's default.
    `race_type` "governor" forecasts the governor races with the same model
    and adjustments; races without polls fall back on the state's last
    presidential result (replaced by the fundamentals estimate when that
    adjustment is on).
    """
    settings = dict(config.simulate)
    priors = None
    if race_type == "governor":
        settings |= {"race_type": "governor", "party_overrides": {}}
        priors = presidential_priors(config, GOVERNOR_RACES_2026, settings.get("cycle", 2026))
    sim = SimulateConfig(**settings)
    as_of = as_of or datetime.now(UTC).date()
    lead = days_to_election(sim.cycle, as_of)
    inputs = state_inputs(config, sim, as_of, priors=priors).reset_index(drop=True)
    if race_type == "governor":
        inputs = inputs[inputs["state"].isin(GOVERNOR_RACES_2026)].reset_index(drop=True)
    inputs = inputs.assign(cycle=sim.cycle, lead_days=lead, as_of=as_of, race_type=sim.race_type)
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

