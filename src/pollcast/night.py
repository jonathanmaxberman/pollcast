"""Election night: one joint simulation that results can condition.

Every race in a simulated election shares one national polling error. So a
called race says something about that error, and through it about every race
still out: Ossoff winning Georgia rules out the simulations where the polls
overstated Democrats by ~9 points, which nudges every other race.

This draws one standardized national error per simulation and scales it for
the Senate, governor and House models, then stores each simulation's margin
in every Senate and governor race and its Democratic House seat count. The
election-night page reweights those simulations as results are entered:

  a called race       keeps simulations where that side won (all but a tiny
                      weight on the rest, so one surprise can't empty the set)
  a reported margin   weights each simulation by how close its margin is
                      (normal, sd `margin_sd` points)

Everything else is then read from the reweighted simulations.
"""

from __future__ import annotations

import base64
from pathlib import Path
from datetime import UTC, date, datetime

import numpy as np
import yaml

from pollcast.config import Config
from pollcast.house import house_inputs
from pollcast.simulate import _t_draws, forecast


def _pack_int16(a: np.ndarray, scale: float) -> str:
    return base64.b64encode(np.clip(np.round(a * scale), -32767, 32767).astype("<i2").tobytes()).decode()


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


def night_payload(config: Config, as_of: date | None = None, n_sims: int = 20000, seed: int = 2026) -> dict:
    as_of = as_of or datetime.now(UTC).date()
    sen_result, sen, sen_sim = forecast(config, as_of, return_inputs=True)
    gov_result, gov, gov_sim = forecast(config, as_of, race_type="governor", return_inputs=True)
    districts, hc, env = house_inputs(config, as_of)

    rng = np.random.default_rng(seed)
    z = _t_draws(rng, sen_sim.t_df, 1.0, (n_sims, 1))  # one national error per simulation, sd 1

    def race_margins(inputs, sim):
        national_sd = sim.national_sd
        if "national_sd_override" in inputs and inputs["national_sd_override"].notna().any():
            national_sd = float(inputs["national_sd_override"].dropna().iloc[0])
        local = _t_draws(rng, sim.t_df, 1.0, (n_sims, len(inputs))) * inputs["sd"].to_numpy()
        m = inputs["mean"].to_numpy() + z * national_sd + local
        for col in sorted(c for c in inputs.columns if str(c).startswith("loading_")):
            m = m + _t_draws(rng, sim.t_df, 1.0, (n_sims, 1)) * inputs[col].fillna(0.0).to_numpy()
        return m

    m_sen = race_margins(sen, sen_sim)
    m_gov = race_margins(gov, gov_sim)
    local = _t_draws(rng, hc.t_df, hc.district_sd, (n_sims, len(districts)))
    m_house = districts["expected"].to_numpy() + z * hc.national_sd + local
    is_dem_house = (districts["challenger"] == "DEM").to_numpy()
    house_dem = ((m_house > 0) & is_dem_house).sum(axis=1)

    races = []
    for kind, inputs, m in (("senate", sen, m_sen), ("governor", gov, m_gov)):
        for i, r in enumerate(inputs.itertuples()):
            races.append({
                "id": f"{kind}:{r.state}" + (":special" if getattr(r, "special", False) else ""),
                "kind": kind, "state": r.state, "special": bool(getattr(r, "special", False)),
                "challenger": r.challenger, "expected": round(float(r.mean), 2),
                "n_polls": int(r.n_polls), "p_prior": round(float((m[:, i] > 0).mean()), 4),
            })
    return {
        "as_of": as_of.isoformat(), "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.name, "n_sims": n_sims,
        # Race margins (challenger minus Republican) x 10, int16, race-major: race r, sim s at r*n + s.
        "margins": _pack_int16(np.concatenate([m_sen, m_gov], axis=1).T, 10),
        "house_dem_seats": base64.b64encode(house_dem.astype("<u2").tobytes()).decode(),
        "national": _pack_int16(z[:, 0] * sen_sim.national_sd, 100),
        "races": races,
        "senate": {"seats_not_up": sen_sim.seats_not_up, "rep_control_seats": sen_sim.rep_control_seats,
                   "p_rep_control_prior": sen_result["control"]["p_rep_control"]},
        "house": {"control_seats": hc.control_seats, "national_margin": env["margin"]},
        "calls": reported_calls(config.data_dir / "inputs" / "night_calls.yaml"),
        "margin_sd": 2.0,
        "called_weight": 0.003,
    }
