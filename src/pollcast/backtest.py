"""Backtests: replay past cycles with only the data available at the time.

For each cycle and lead time, the Senate forecast is rebuilt as of
(election day - lead) from polls released by then, with prior results only
from earlier elections. Each race's forecast is then scored against the
certified result.

Scores
  log loss   mean of -log(p assigned to what happened); lower is better
  Brier      mean squared error of the win probability; lower is better
  MAE        mean absolute error of the expected margin, in points
  bias       mean (actual - expected) margin; positive = the challenger
             (usually the Democrat) beat the forecast

Calibration estimates the two error sizes the simulation assumes, from
election-eve misses (method of moments):

  national_sd   spread of each cycle's average miss across polled races
  state_sd, few_polls_sd
                fit to the leftover miss in each race: miss^2 - se^2 is
                regressed on 1 / effective number of polls

With so few cycles, every number here comes with wide error bars. The
leave-one-cycle-out score is the honest one: each cycle is forecast with
error sizes estimated from the other cycles only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pollcast.config import Config
from pollcast.pipeline import build_polls
from pollcast.simulate import SimulateConfig, simulate, state_inputs
from pollcast.sources import SOURCES
from pollcast.storage import latest_snapshot

ELECTION_DAYS = {
    2018: date(2018, 11, 6),
    2020: date(2020, 11, 3),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
    2026: date(2026, 11, 3),
}
EPS = 1e-4  # probabilities are clipped to [EPS, 1 - EPS] before taking logs


def senate_results(config: Config) -> pd.DataFrame:
    snap = latest_snapshot(config.data_dir, "mit_results")
    if snap is None:
        raise RuntimeError("backtest: run `pollcast fetch mit_results` first")
    return SOURCES["mit_results"](config).normalize(snap)


def priors_for(results: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """Each race's fallback: the most recent earlier Senate result in that state.

    Prefers the same seat's last regular election (six years earlier).
    """
    rows = []
    for r in results[results["cycle"] == cycle].itertuples():
        earlier = results[(results["state"] == r.state) & (results["cycle"] < cycle)]
        same_seat = earlier[(earlier["cycle"] == cycle - 6) & ~earlier["special"]]
        pick = same_seat if len(same_seat) else earlier.sort_values("cycle").tail(1)
        if pick.empty:
            continue
        p = pick.iloc[-1]
        rows.append({"state": r.state, "special": r.special, "DEM": p.nonrep_pct,
                     "IND": p.nonrep_pct, "REP": p.rep_pct, "label": f"Election {p.cycle}"})
    return pd.DataFrame(rows, columns=["state", "special", "DEM", "IND", "REP", "label"])


def sim_for(config: Config, cycle: int, **overrides) -> SimulateConfig:
    base = {k: v for k, v in config.simulate.items() if k not in ("cycle", "party_overrides")}
    return SimulateConfig(**(base | {"cycle": cycle} | overrides))


def replay_senate(config: Config, cycles: list[int], leads: list[int],
                  n_sims: int = 5000) -> pd.DataFrame:
    """One row per (cycle, lead, race): the forecast as of then, and what happened."""
    results = senate_results(config)
    polls = build_polls(config)
    rows = []
    for cycle in cycles:
        actual = results[results["cycle"] == cycle]
        priors = priors_for(results, cycle)
        sim = sim_for(config, cycle, n_sims=n_sims)
        for lead in leads:
            as_of = ELECTION_DAYS[cycle] - timedelta(days=lead)
            inputs = state_inputs(config, sim, as_of, polls=polls, priors=priors, trend_days=0)
            inputs = inputs.merge(actual, on=["state", "special"], how="inner")
            if inputs.empty:
                continue
            inputs["p"] = simulate(inputs, sim)["p_challenger"]
            inputs["cycle"], inputs["lead_days"], inputs["as_of"] = cycle, lead, as_of
            rows.append(inputs.drop(columns=["trend", "polls"]))
    races = pd.concat(rows, ignore_index=True)
    races["actual"] = races["nonrep_won"].astype(int)
    return races


def score(races: pd.DataFrame, p_col: str = "p") -> dict:
    p = races[p_col].clip(EPS, 1 - EPS)
    y = races["actual"]
    miss = races["margin"] - races["mean"]
    return {
        "n": int(len(races)),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "mae": float(miss.abs().mean()),
        "bias": float(miss.mean()),
        "correct": float(((p > 0.5) == (y == 1)).mean()),
    }


def score_table(races: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    rows = []
    for (cycle, lead), g in races.groupby(["cycle", "lead_days"]):
        rows.append({"cycle": cycle, "lead_days": lead, **score(g, p_col)})
    for lead, g in races.groupby("lead_days"):
        rows.append({"cycle": "all", "lead_days": lead, **score(g, p_col)})
    return pd.DataFrame(rows)


def calibration_bins(races: pd.DataFrame, p_col: str = "p", bins: int = 10) -> pd.DataFrame:
    """Forecast probability vs how often the challenger actually won, by bin."""
    edges = np.linspace(0, 1, bins + 1)
    b = pd.cut(races[p_col], edges, include_lowest=True)
    out = races.groupby(b, observed=True).agg(n=("actual", "size"), forecast=(p_col, "mean"),
                                              observed=("actual", "mean")).reset_index()
    out["bin"] = out[p_col].astype(str)
    return out[["bin", "n", "forecast", "observed"]]


@dataclass
class ErrorSizes:
    national_sd: float
    state_sd: float
    few_polls_sd: float
    prior_sd: float


def estimate_error_sizes(races: pd.DataFrame, fallback: ErrorSizes) -> ErrorSizes:
    """Method-of-moments error sizes from election-eve misses (see module doc)."""
    eve = races[races["lead_days"] == races["lead_days"].min()]
    polled = eve[eve["basis"] == "polls"].copy()
    polled["miss"] = polled["margin"] - polled["mean"]
    cycle_miss = polled.groupby("cycle")["miss"].mean()
    national_sd = float(np.sqrt((cycle_miss**2).mean())) if len(cycle_miss) else fallback.national_sd

    within = polled["miss"] - polled["cycle"].map(cycle_miss)
    y = within**2 - polled["se"] ** 2
    x = 1 / polled["effective_n"].clip(lower=1)
    if len(polled) >= 10:
        slope, intercept = (np.polyfit(x, y, 1) if x.std() > 1e-3 else (0.0, y.mean()))
        if slope < 0:  # more polls can't mean more error: fit the constant alone
            slope, intercept = 0.0, y.mean()
        state_sd = float(np.sqrt(max(intercept, 1.0)))
        few_polls_sd = float(np.sqrt(max(slope, 0.0)))
    else:
        state_sd, few_polls_sd = fallback.state_sd, fallback.few_polls_sd

    unpolled = eve[eve["basis"] != "polls"]
    if len(unpolled) >= 8:
        prior_miss = unpolled["margin"] - unpolled["mean"] - unpolled["cycle"].map(cycle_miss).fillna(0)
        prior_sd = float(np.sqrt((prior_miss**2).mean()))
    else:
        prior_sd = fallback.prior_sd
    return ErrorSizes(national_sd, state_sd, few_polls_sd, prior_sd)


def resimulate(races: pd.DataFrame, sizes: ErrorSizes, sim: SimulateConfig,
               seat_scores: dict | None = None) -> pd.Series:
    """Win probabilities for already-replayed races under different error sizes.

    If `seat_scores` is a dict, it is filled with {(cycle, lead): log probability
    the simulation gave to the number of races the challenger actually won}.
    """
    out = pd.Series(index=races.index, dtype=float)
    sim = replace(sim, national_sd=sizes.national_sd)
    for _, g in races.groupby(["cycle", "lead_days"]):
        sd = np.where(
            g["basis"] == "polls",
            np.sqrt(sizes.state_sd**2 + g["se"] ** 2 + sizes.few_polls_sd**2 / g["effective_n"].clip(lower=1)),
            sizes.prior_sd,
        )
        # An indicator that changes a race's uncertainty (e.g. a blend) sets sd_override.
        if "sd_override" in g:
            sd = np.where(g["sd_override"].notna(), g["sd_override"], sd)
        r = simulate(g.assign(sd=sd), sim)
        out.loc[g.index] = r["p_challenger"]
        if seat_scores is not None:
            p_actual = r["challenger_wins_hist"].get(int(g["actual"].sum()), 0.0)
            seat_scores[(g["cycle"].iloc[0], g["lead_days"].iloc[0])] = float(np.log(max(p_actual, 1 / sim.n_sims)))
    return out


def leave_one_cycle_out(races: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict]:
    """Forecast each cycle with error sizes estimated from the other cycles only."""
    default = sim_for(config, 0)
    fallback = ErrorSizes(default.national_sd, default.state_sd, default.few_polls_sd, default.prior_sd)
    races = races.copy()
    races["p_loco"] = np.nan
    fitted = {}
    for cycle in sorted(races["cycle"].unique()):
        train = races[races["cycle"] != cycle]
        sizes = estimate_error_sizes(train, fallback)
        fitted[int(cycle)] = asdict(sizes)
        held = races["cycle"] == cycle
        races.loc[held, "p_loco"] = resimulate(races[held], sizes, default)
    fitted["all_cycles"] = asdict(estimate_error_sizes(races, fallback))
    return races, fitted


def config_error_sizes(config: Config) -> ErrorSizes:
    d = sim_for(config, 0)
    return ErrorSizes(d.national_sd, d.state_sd, d.few_polls_sd, d.prior_sd)


def log_loss(p: pd.Series, y: pd.Series) -> float:
    p = p.clip(EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score_indicator(races: pd.DataFrame, indicator, config: Config, name: str | None = None) -> dict:
    """Held-out test of one indicator against the baseline.

    For each cycle, the indicator is fit on the other cycles, applied to this
    one, and the races re-simulated with the config's error sizes. The
    baseline is re-simulated the same way with the same random draws, so the
    only difference is the indicator. Accepted when held-out log loss improves
    overall and every fold's fitted effect has the declared sign.
    """
    sizes, sim = config_error_sizes(config), sim_for(config, 0)
    seats_base, seats_ind = {}, {}
    base = resimulate(races, sizes, sim, seats_base)
    adjusted = races.copy()
    adjusted["sd_override"] = np.nan
    folds = {}
    for cycle in sorted(races["cycle"].unique()):
        params = indicator.fit(races[races["cycle"] != cycle])
        folds[int(cycle)] = {"params": params, "effect": indicator.effect(params)}
        for lead, g in races[races["cycle"] == cycle].groupby("lead_days"):
            applied = indicator.apply(g, params, int(lead))
            adjusted.loc[g.index, "mean"] = applied["mean"]
            if "sd_override" in applied:
                adjusted.loc[g.index, "sd_override"] = applied["sd_override"]
            for col in (c for c in applied.columns if str(c).startswith("loading_")):
                adjusted.loc[g.index, col] = applied[col]
    treated = resimulate(adjusted, sizes, sim, seats_ind)

    rows = []
    for (cycle, lead), g in races.groupby(["cycle", "lead_days"]):
        rows.append({"cycle": cycle, "lead_days": lead, "n": len(g),
                     "log_loss_base": log_loss(base[g.index], g["actual"]),
                     "log_loss_indicator": log_loss(treated[g.index], g["actual"]),
                     "mae_base": float((g["margin"] - g["mean"]).abs().mean()),
                     "mae_indicator": float((g["margin"] - adjusted.loc[g.index, "mean"]).abs().mean())})
    for cycle, g in races.groupby("cycle"):
        rows.append({"cycle": cycle, "lead_days": "all", "n": len(g),
                     "log_loss_base": log_loss(base[g.index], g["actual"]),
                     "log_loss_indicator": log_loss(treated[g.index], g["actual"]),
                     "mae_base": float((g["margin"] - g["mean"]).abs().mean()),
                     "mae_indicator": float((g["margin"] - adjusted.loc[g.index, "mean"]).abs().mean())})
    table = pd.DataFrame(rows)
    table["improvement"] = table["log_loss_base"] - table["log_loss_indicator"]

    # Seat-count score: mean log probability given to each (cycle, lead)'s actual
    # number of challenger wins. Higher is better; reported as a loss (negated).
    seat_base = -float(np.mean(list(seats_base.values())))
    seat_ind = -float(np.mean(list(seats_ind.values())))
    for col, scores in (("seat_loss_base", seats_base), ("seat_loss_indicator", seats_ind)):
        table[col] = [(-np.mean([v for (c, _), v in scores.items() if c == r.cycle]) if r.lead_days == "all"
                       else -scores.get((r.cycle, r.lead_days), np.nan)) for r in table.itertuples()]

    metric = getattr(indicator, "metric", "race_log_loss")
    if metric == "seat_log_score":
        overall_base, overall_ind = seat_base, seat_ind
        table["improvement"] = table["seat_loss_base"] - table["seat_loss_indicator"]
    else:
        overall_base = log_loss(base, races["actual"])
        overall_ind = log_loss(treated, races["actual"])
    signs_ok = all(f["effect"] * indicator.expected_sign > 0 for f in folds.values())
    by_cycle = table[table["lead_days"] == "all"]
    return {
        "name": name or indicator.name,
        "module": indicator.name,
        "metric": metric,
        "seat_loss_base": seat_base,
        "seat_loss_indicator": seat_ind,
        "expected": indicator.expected,
        "log_loss_base": overall_base,
        "log_loss_indicator": overall_ind,
        "improvement": overall_base - overall_ind,
        "cycles_improved": int((by_cycle["improvement"] > 0).sum()),
        "cycles": int(len(by_cycle)),
        "direction_as_declared": signs_ok,
        "accepted": bool(overall_ind < overall_base and signs_ok),
        "folds": folds,
        "params_all_cycles": indicator.fit(races),
        "table": table,
    }


def config_hash(config: Config) -> str:
    text = config.path.read_text() if config.path else json.dumps(config.simulate, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def run_senate_backtest(config: Config, cycles: list[int], leads: list[int],
                        n_sims: int = 5000) -> dict:
    """Replay, score, calibrate, and write everything to a dated run folder."""
    races = replay_senate(config, cycles, leads, n_sims)
    races, fitted = leave_one_cycle_out(races, config)
    scores = score_table(races).assign(error_sizes="config")
    loco = score_table(races, "p_loco").assign(error_sizes="leave-one-cycle-out")
    table = pd.concat([scores, loco], ignore_index=True)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_senate_" + config.name
    out = config.data_dir / "backtests" / run_id
    out.mkdir(parents=True)
    races.to_parquet(out / "races.parquet", index=False)
    table.to_csv(out / "scores.csv", index=False)
    eve = races[races["lead_days"] == min(leads)]
    calib = {"config": calibration_bins(eve, "p"), "loco": calibration_bins(eve, "p_loco")}
    for name, c in calib.items():
        c.to_csv(out / f"calibration_{name}.csv", index=False)
    meta = {
        "run_id": run_id, "race": "senate", "config": config.name, "config_hash": config_hash(config),
        "cycles": cycles, "leads": leads, "n_sims": n_sims, "fitted_error_sizes": fitted,
        "config_error_sizes": {k: config.simulate.get(k) for k in
                               ("national_sd", "state_sd", "few_polls_sd", "prior_sd", "t_df")},
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    plot_calibration(eve, out / "calibration.png")
    append_log(config, meta, table)

    from pollcast.indicators import INDICATORS

    tested = {}
    for name, settings in (config.indicators or {}).items():
        # An entry can reuse a module under its own name with different
        # settings (`module: undecided_stretch`), so each variant tried is
        # scored and logged as its own comparison.
        module = (settings or {}).get("module", name)
        if module not in INDICATORS:
            raise RuntimeError(f"backtest: unknown indicator {module!r}; known: {', '.join(INDICATORS)}")
        indicator = INDICATORS[module](settings)
        indicator.prepare(config)
        if not indicator.testable():
            print(f"{name}: not testable yet ({indicator.untestable_reason()}); skipped")
            continue
        result = score_indicator(races, indicator, config, name=name)
        result["table"].to_csv(out / f"indicator_{name}.csv", index=False)
        tested[name] = result
        append_indicator_log(config, meta, result)
    if tested:
        save_indicator_params(config, meta, tested)
    return {"dir": out, "scores": table, "fitted": fitted, "calibration": calib, "races": races,
            "indicators": tested}


def append_indicator_log(config: Config, meta: dict, result: dict) -> None:
    """Every indicator test is logged, passed or not, so the count of tries stays visible."""
    log = config.data_dir / "backtests" / "indicator_log.csv"
    row = pd.DataFrame([{
        "run_id": meta["run_id"], "race": meta["race"], "indicator": result["name"],
        "module": result["module"], "metric": result["metric"],
        "expected": result["expected"], "log_loss_base": result["log_loss_base"],
        "log_loss_indicator": result["log_loss_indicator"], "improvement": result["improvement"],
        "cycles_improved": f"{result['cycles_improved']}/{result['cycles']}",
        "direction_as_declared": result["direction_as_declared"], "accepted": result["accepted"],
        "settings": json.dumps((config.indicators or {}).get(result["name"]) or {}),
    }])
    # Rewrite rather than append, so rows stay aligned if columns are added later.
    old = pd.read_csv(log) if log.exists() else pd.DataFrame()
    pd.concat([old, row], ignore_index=True).to_csv(log, index=False)


def save_indicator_params(config: Config, meta: dict, tested: dict) -> None:
    """The latest test result and all-cycle fit per indicator, read by the live forecast."""
    path = config.data_dir / "backtests" / "indicator_params.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    for name, r in tested.items():
        saved[name] = {
            "accepted": r["accepted"], "run_id": meta["run_id"], "race": meta["race"], "metric": r["metric"],
            "improvement": r["improvement"], "cycles_improved": f"{r['cycles_improved']}/{r['cycles']}",
            "direction_as_declared": r["direction_as_declared"], "params": r["params_all_cycles"],
        }
    path.write_text(json.dumps(saved, indent=2))


def append_log(config: Config, meta: dict, table: pd.DataFrame) -> None:
    """Every run is logged, so the number of comparisons tried stays visible."""
    log = config.data_dir / "backtests" / "log.csv"
    rows = table[table["cycle"] == "all"].assign(
        run_id=meta["run_id"], race=meta["race"], config=meta["config"],
        config_hash=meta["config_hash"], indicators=json.dumps(config.indicators or {}))
    rows.to_csv(log, mode="a", header=not log.exists(), index=False)


def plot_calibration(eve: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    ax.plot([0, 1], [0, 1], color="#999", linewidth=1, linestyle="--", label="Perfect calibration")
    for col, color, label in (("p", "#3b6fb6", "Config error sizes"),
                              ("p_loco", "#c4302f", "Leave-one-cycle-out")):
        c = calibration_bins(eve, col, bins=5)
        ax.plot(c["forecast"], c["observed"], marker="o", color=color, label=label)
        for r in c.itertuples():
            ax.annotate(str(r.n), (r.forecast, r.observed), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=color)
    ax.set_xlabel("Forecast chance the challenger wins")
    ax.set_ylabel("Share of races the challenger won")
    ax.set_title("Senate election-eve calibration, 2018-2024")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
