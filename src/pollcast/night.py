"""Election night: results condition one shared national polling error.

Every race shares one national polling error. So a called race says
something about that error, and through it about every race still out:
Ossoff winning Georgia rules out the worlds where the polls overstated
Democrats by ~9 points, which nudges every other race.

Given the national error, races are independent (their own errors are
separate draws). So the page works on a grid of national-error values
instead of a finite set of simulated elections:

  prior        the model's Student-t national error, at each grid point
  a call       multiplies each point by P(that side wins | national error)
  a margin     multiplies it by the density of that margin (the race's own
               error widened by `margin_sd` for reporting noise)

Every other race, Senate control (exact seat-count distribution at each
point), House control and the implied polling miss are then averaged over
the updated grid. Nothing runs out: dozens of calls, or results far in a
tail, still leave a smooth answer, where reweighting 20,000 simulations
left only a few hundred that matched.
"""

from __future__ import annotations

import base64
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import yaml

from pollcast.config import Config
from pollcast.house import house_inputs
from pollcast.simulate import forecast

GRID = np.linspace(-6.0, 6.0, 481)  # national error in standard deviations


def _t_unit(df: float):
    """pdf and cdf of the Student t rescaled to standard deviation 1."""
    s = math.sqrt((df - 2) / df)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    xs = np.linspace(-80.0, 80.0, 160001)
    raw = c * (1 + (xs / s) ** 2 / df) ** (-(df + 1) / 2) / s
    cdf = np.concatenate([[0.0], np.cumsum((raw[1:] + raw[:-1]) / 2 * np.diff(xs))])
    cdf /= cdf[-1]
    return (lambda x: c * (1 + (np.asarray(x) / s) ** 2 / df) ** (-(df + 1) / 2) / s,
            lambda x: np.interp(x, xs, cdf))


def _pack_u16(p: np.ndarray) -> str:
    return base64.b64encode(np.round(np.clip(p, 0, 1) * 65535).astype("<u2").tobytes()).decode()


def _national_sd(inputs, sim) -> float:
    if "national_sd_override" in inputs and inputs["national_sd_override"].notna().any():
        return float(inputs["national_sd_override"].dropna().iloc[0])
    return sim.national_sd


def reported_calls(path: Path) -> list[dict]:
    """Calls to ship with the page, from a YAML list of {race, winner: DEM|REP|IND, margin?, source}.

    `race` is a page race id such as "senate:Georgia" or "governor:Illinois".
    """
    if not path.exists():
        return []
    calls = []
    for c in yaml.safe_load(path.read_text()) or []:
        calls.append({"id": c["race"], "winner": "rep" if c["winner"] == "REP" else "challenger",
                      "margin": c.get("margin"), "source": c.get("source", "")})
    return calls


def night_payload(config: Config, as_of: date | None = None) -> dict:
    as_of = as_of or datetime.now(UTC).date()
    sen_result, sen, sen_sim = forecast(config, as_of, return_inputs=True)
    gov_result, gov, gov_sim = forecast(config, as_of, race_type="governor", return_inputs=True)
    districts, hc, env = house_inputs(config, as_of)
    pdf, cdf = _t_unit(sen_sim.t_df)
    prior = pdf(GRID)
    prior /= prior.sum()

    races, p_win = [], []
    for kind, inputs, sim in (("senate", sen, sen_sim), ("governor", gov, gov_sim)):
        nsd = _national_sd(inputs, sim)
        for r in inputs.itertuples():
            p = 1 - cdf(-(r.mean + GRID * nsd) / r.sd)  # P(challenger wins | national error)
            p_win.append(p)
            races.append({
                "id": f"{kind}:{r.state}" + (":special" if getattr(r, "special", False) else ""),
                "kind": kind, "state": r.state, "special": bool(getattr(r, "special", False)),
                "challenger": r.challenger, "expected": round(float(r.mean), 2), "sd": round(float(r.sd), 3),
                "national_sd": nsd, "n_polls": int(r.n_polls), "p_prior": round(float(prior @ p), 4),
            })

    # House: exact distribution of Democratic seats at each grid point (districts independent given it).
    _, hcdf = _t_unit(hc.t_df)
    dem = (districts["challenger"] == "DEM").to_numpy()
    pd_win = 1 - hcdf(-(districts["expected"].to_numpy()[None, :] + GRID[:, None] * hc.national_sd) / hc.district_sd)
    seats = np.zeros((len(GRID), len(districts) + 1))
    seats[:, 0] = 1.0
    for j in np.flatnonzero(dem):
        q = pd_win[:, j][:, None]
        seats[:, 1:] = seats[:, 1:] * (1 - q) + seats[:, :-1] * q
        seats[:, 0] *= 1 - q[:, 0]
    house_control = seats[:, hc.control_seats:].sum(axis=1)

    return {
        "as_of": as_of.isoformat(), "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.name,
        "grid": [round(float(g), 4) for g in GRID], "prior": [float(f"{x:.6g}") for x in prior],
        "t_df": sen_sim.t_df, "national_sd": sen_sim.national_sd,
        # P(challenger wins | grid point) per race, uint16 /65535, race-major.
        "p_win": _pack_u16(np.concatenate(p_win)),
        "house_control": _pack_u16(house_control),
        "races": races,
        "senate": {"seats_not_up": sen_sim.seats_not_up, "rep_control_seats": sen_sim.rep_control_seats,
                   "p_rep_control_prior": sen_result["control"]["p_rep_control"]},
        "house": {"control_seats": hc.control_seats, "national_margin": env["margin"],
                  "p_control_prior": round(float(prior @ house_control), 4)},
        "calls": reported_calls(config.data_dir / "inputs" / "night_calls.yaml"),
        "margin_sd": 2.0,
    }
