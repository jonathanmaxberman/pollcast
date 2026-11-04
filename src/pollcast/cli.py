"""Command-line entry point: ``pollcast <command> --config configs/x.yaml``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from pollcast.average import poll_margins, time_weighted_average
from pollcast.config import load_config
from pollcast.pipeline import build_polls, fetch, write_tables
from pollcast.sources import SOURCES


def _date(s: str) -> date:
    return date.fromisoformat(s)


def init_configs(target: Path, force: bool = False) -> int:
    """Copy the packaged starter configs into `target`."""
    from importlib import resources

    target.mkdir(parents=True, exist_ok=True)
    for name in ("polls_only.yaml", "polls_fundamentals.yaml", "reference.yaml"):
        dest = target / name
        if dest.exists() and not force:
            print(f"kept {dest} (exists; --force to overwrite)")
            continue
        dest.write_text(resources.files("pollcast.examples").joinpath(name).read_text())
        print(f"wrote {dest}")
    print("Next: `pollcast fetch --config configs/polls_only.yaml`, then `pollcast map --config ...`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pollcast")
    parser.add_argument("--config", "-c", default="configs/reference.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write starter configs into ./configs")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")

    p_fetch = sub.add_parser("fetch", help="pull raw data into dated snapshots")
    p_fetch.add_argument("sources", nargs="*", help=f"default: enabled sources ({', '.join(SOURCES)})")
    p_fetch.add_argument("--force", action="store_true", help="re-fetch frozen archives too")

    sub.add_parser("build", help="rebuild normalized Parquet tables from all snapshots")

    p_avg = sub.add_parser("average", help="compute the time-weighted average")
    p_avg.add_argument("--as-of", type=_date, help="use only data available on this date")
    p_avg.add_argument("--cycle", type=int, help="override average.cycle from the config")
    p_avg.add_argument("--start", type=_date)
    p_avg.add_argument("--out", type=Path, help="CSV path (default: <data_dir>/output/...)")
    p_avg.add_argument("--plot", action="store_true", help="also save a PNG next to the CSV")

    p_map = sub.add_parser("map", help="run the Senate and House simulations and write the forecast page")
    p_map.add_argument("--as-of", type=_date, help="use only data available on this date")
    p_map.add_argument("--out", type=Path, help="HTML path (default: <data_dir>/output/...)")
    p_map.add_argument("--fragment", action="store_true",
                       help="omit the <html> wrapper (for hosts that add their own)")

    p_night = sub.add_parser("night", help="write the election-night page (enter results, odds update)")
    p_night.add_argument("--as-of", type=_date)
    p_night.add_argument("--out", type=Path)
    p_night.add_argument("--fragment", action="store_true")

    p_bt = sub.add_parser("backtest", help="replay past Senate cycles and score the forecasts")
    p_bt.add_argument("--cycles", type=int, nargs="+", default=[2018, 2020, 2022, 2024])
    p_bt.add_argument("--leads", type=int, nargs="+", default=[0, 7, 30, 60, 90],
                      help="days before election day to forecast from")
    p_bt.add_argument("--n-sims", type=int, default=5000)
    p_bt.add_argument("--history", action="store_true",
                      help="election-eve backtest over 1998-2022 from 538's pollster-ratings history")

    args = parser.parse_args(argv)
    if args.command == "init":
        return init_configs(Path("configs"), args.force)
    if not Path(args.config).exists():
        print(f"pollcast: no config at {args.config}. Run `pollcast init` to write starter configs, "
              "or pass --config.", file=sys.stderr)
        return 2
    config = load_config(args.config)

    if args.command == "fetch":
        failed = False
        for r in fetch(config, args.sources or None, force=args.force):
            if r.snapshot:
                print(f"{r.source}: {len(r.snapshot.files())} files -> {r.snapshot.path}")
                for failure in r.snapshot.manifest().get("failures", []):
                    failed = True
                    print(f"{r.source}: partial: {failure}", file=sys.stderr)
            elif r.skipped:
                print(f"{r.source}: skipped ({r.skipped})")
            else:
                failed = True
                print(f"{r.source}: FAILED: {r.error}", file=sys.stderr)
        return 1 if failed else 0

    if args.command == "build":
        for table, n in write_tables(config).items():
            print(f"{table}: {n} rows")
        return 0

    if args.command == "average":
        cfg = replace(config.average, cycle=args.cycle) if args.cycle else config.average
        margins = poll_margins(build_polls(config, as_of=args.as_of), cfg)
        avg = time_weighted_average(margins, cfg, start=args.start, end=args.as_of)
        if avg.empty:
            print(f"no {cfg.race_type} polls for {cfg.cycle}", file=sys.stderr)
            return 1
        label = f"{cfg.race_type}_{cfg.cycle}" + (f"_asof_{args.as_of}" if args.as_of else "")
        out = args.out or config.data_dir / "output" / f"average_{label}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        avg.to_csv(out, index=False)
        last = avg.iloc[-1]
        print(f"{len(margins)} polls; {last['date'].date()}: D{last['margin']:+.1f} "
              f"(se {last['se']:.1f}, {int(last['n_polls'])} polls weighted) -> {out}")
        if args.plot:
            from pollcast.plot import plot_average

            title = f"{cfg.race_type.title()} ballot {cfg.cycle}: time-weighted average"
            print(plot_average(margins, avg, title, out.with_suffix(".png")))
        return 0
    if args.command == "map":
        import json

        from pollcast.house import house_forecast
        from pollcast.simulate import forecast
        from pollcast.web import load_topology, render_forecast_page

        result = forecast(config, as_of=args.as_of)
        governor = forecast(config, as_of=args.as_of, race_type="governor")
        try:
            house = house_forecast(config, as_of=args.as_of)
        except RuntimeError as e:
            house = None
            print(f"house: skipped ({e})", file=sys.stderr)
        out = args.out or config.data_dir / "output" / f"forecast_{result['cycle']}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps({"senate": result, "house": house, "governor": governor},
                                                       indent=1))
        html = render_forecast_page(result, house, load_topology(config.data_dir / "cache"),
                                    standalone=not args.fragment, governor=governor)
        out.write_text(html)
        c = result["control"]
        print(f"senate: {len(result['states'])} races; P(R control) {c['p_rep_control']:.1%}, "
              f"P(D majority) {c['p_dem_majority']:.1%}")
        if house:
            print(f"house: {len(house['districts'])} districts, national {house['national']['margin']:+.1f} "
                  f"({house['national']['source']}); P(D control) {house['control']['p_dem_control']:.1%}")
        gw = governor["control"]["challenger_wins_hist"]
        median = next(k for k, c in zip(sorted(gw), __import__("itertools").accumulate(gw[k] for k in sorted(gw))) if c >= 0.5)
        print(f"governor: {len(governor['states'])} races; challengers win {median} in the median simulation")
        print(f"-> {out}")
        return 0
    if args.command == "night":
        from pollcast.night import night_payload
        from pollcast.web import render_night_page

        payload = night_payload(config, as_of=args.as_of)
        out = args.out or config.data_dir / "output" / f"night_{payload['as_of'][:4]}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_night_page(payload, standalone=not args.fragment))
        print(f"{len(payload['races'])} races, House control {payload['house']['p_control_prior']:.1%} -> {out}")
        return 0

    if args.command == "backtest" and args.history:
        from pollcast.history import run_history_backtest

        r = run_history_backtest(config, args.n_sims)
        print("== election eve, 1998-2022, by race type")
        print(r["by_type"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("fitted error sizes (all cycles):", {k: round(v, 2) for k, v in r["fitted"]["all_cycles"].items()})
        for name, ind in r["indicators"].items():
            print(f"{name}: {ind['log_loss_base']:.4f} -> {ind['log_loss_indicator']:.4f} "
                  f"({ind['improvement']:+.4f}); better in {ind['cycles_improved']}/{ind['cycles']}; "
                  f"direction as declared in every fold: {ind['direction_as_declared']}")
        h = r["house_effects"]
        print(f"house effects vs results: {h['log_loss_base']:.4f} -> {h['log_loss_indicator']:.4f}; "
              f"better in {h['cycles_improved']}/{h['cycles']}")
        print(f"-> {r['dir']}")
        return 0

    if args.command == "backtest":
        from pollcast.backtest import run_senate_backtest

        r = run_senate_backtest(config, args.cycles, args.leads, args.n_sims)
        t = r["scores"]
        cols = ["cycle", "lead_days", "n", "log_loss", "brier", "mae", "bias", "correct"]
        for label, g in t.groupby("error_sizes", sort=False):
            print(f"\n== error sizes: {label}")
            print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n== error sizes fitted from past misses (points)")
        print(pd.DataFrame(r["fitted"]).T.to_string(float_format=lambda v: f"{v:.2f}"))
        for name, ind in r["indicators"].items():
            verdict = "PASSED: joins the headline forecast if enabled" if ind["accepted"] else "not accepted: stays exploratory"
            print(f"\n== indicator {name} ({ind['expected']})")
            t = ind["table"]
            print(t[t["lead_days"] == "all"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            metric = "seat-count log loss" if ind["metric"] == "seat_log_score" else "log loss"
            print(f"held-out {metric} {ind['log_loss_base']:.4f} -> {ind['log_loss_indicator']:.4f} "
                  f"({ind['improvement']:+.4f}); better in {ind['cycles_improved']}/{ind['cycles']} cycles; "
                  f"direction as declared in every fold: {ind['direction_as_declared']}")
            print("fitted effect per fold:", {c: round(f["effect"], 3) for c, f in ind["folds"].items()},
                  "| all cycles:", ind["params_all_cycles"])
            print(verdict)
        print(f"\n-> {r['dir']}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
