from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from pollcast.average import poll_margins, time_weighted_average
from pollcast.config import AverageConfig, Config
from pollcast.pipeline import build_polls
from pollcast.sources._538_format import parse_538_csv
from pollcast.sources.electoral_vote import parse_ev_csv
from pollcast.storage import SnapshotWriter, latest_snapshot

GENERIC_538 = """poll_id,pollster_id,pollster,sponsors,start_date,end_date,question_id,sample_size,population,created_at,internal,partisan,cycle,office_type,seat_name,dem,rep,ind
1,10,Alpha,,10/1/22,10/3/22,100,1000,lv,10/4/22 09:00,false,,2022,U.S. House,Generic,47,45,
1,10,Alpha,,10/1/22,10/3/22,101,1200,rv,10/4/22 09:00,false,,2022,U.S. House,Generic,48,44,
2,20,Beta,,10/20/22,10/22/22,200,800,lv,10/30/22 12:00,false,,2022,U.S. House,Generic,44,48,
3,30,Gamma,,10/5/22,10/6/22,300,600,lv,10/7/22 12:00,false,REP,2022,U.S. House,Generic,40,52,
"""

SENATE_538 = """poll_id,pollster,start_date,end_date,question_id,sample_size,population,created_at,cycle,office_type,state,party,answer,candidate_name,pct
5,Delta,9/1/24,9/3/24,500,700,lv,9/5/24,2024,U.S. Senate,Ohio,DEM,Brown,Sherrod Brown,48
5,Delta,9/1/24,9/3/24,500,700,lv,9/5/24,2024,U.S. Senate,Ohio,REP,Moreno,Bernie Moreno,46
"""

EV = """Day,Len,State,EV,Dem,GOP,Ind,Date,,,,,,,,Pollster
303.0,5,Alaska,,50,45,03,Nov 01,,,,,,,,AtlasIntel-5
1.0,1,Alaska,,40,55,,Jan 01,,,,,,,,Election 2020-1
"""


def test_parse_wide_538(tmp_path: Path):
    f = tmp_path / "g.csv"
    f.write_text(GENERIC_538)
    df = parse_538_csv(f, source="t", default_race_type="generic")
    assert set(df["race_type"]) == {"generic"}
    assert len(df) == 8  # 4 questions x DEM/REP (ind blank)
    assert df["release_date"].iloc[0] == pd.Timestamp("2022-10-04 09:00")
    assert not df["release_date_imputed"].any()


def test_parse_long_538(tmp_path: Path):
    f = tmp_path / "s.csv"
    f.write_text(SENATE_538)
    df = parse_538_csv(f, source="t", default_race_type="senate")
    assert list(df["party"]) == ["DEM", "REP"]
    assert list(df["candidate"]) == ["Sherrod Brown", "Bernie Moreno"]
    assert set(df["state"]) == {"Ohio"}


def test_parse_ev_dates_and_drops_seed_results(tmp_path: Path):
    f = tmp_path / "ev.csv"
    f.write_text(EV)
    df = parse_ev_csv(f, cycle=2026, race_type="senate")
    assert set(df["pollster"]) == {"AtlasIntel"}
    assert df["start_date"].iloc[0] == pd.Timestamp("2026-10-28")
    assert df["end_date"].iloc[0] == pd.Timestamp("2026-11-01")  # matches the Date column
    assert set(df["party"]) == {"DEM", "REP", "IND"}


def test_margins_prefer_lv_and_drop_partisan(tmp_path: Path):
    f = tmp_path / "g.csv"
    f.write_text(GENERIC_538)
    polls = parse_538_csv(f, source="t", default_race_type="generic")
    m = poll_margins(polls, AverageConfig(cycle=2022))
    assert sorted(m["poll_id"]) == ["1", "2"]  # Gamma is a partisan poll
    assert m.set_index("poll_id").loc["1", "margin"] == 2  # LV question, not RV


def test_average_has_no_lookahead(tmp_path: Path):
    f = tmp_path / "g.csv"
    f.write_text(GENERIC_538)
    m = poll_margins(parse_538_csv(f, source="t", default_race_type="generic"),
                     AverageConfig(cycle=2022))
    avg = time_weighted_average(m, AverageConfig(cycle=2022), start=date(2022, 10, 25),
                                end=date(2022, 10, 31)).set_index("date")
    # Beta (D-4) fielded Oct 20-22 but was only released Oct 30.
    assert avg.loc["2022-10-29", "margin"] == pytest.approx(2.0)
    assert avg.loc["2022-10-30", "margin"] < 2.0


def test_snapshots_are_append_only_and_selected_by_date(tmp_path: Path):
    early = SnapshotWriter(tmp_path, "s", now=datetime(2026, 1, 1, tzinfo=UTC))
    early.write("a.csv", b"x")
    early.commit()
    with pytest.raises(FileExistsError):
        SnapshotWriter(tmp_path, "s", now=datetime(2026, 1, 1, tzinfo=UTC))
    late = SnapshotWriter(tmp_path, "s", now=datetime(2026, 2, 1, tzinfo=UTC))
    late.write("a.csv", b"y")
    late.commit()
    assert latest_snapshot(tmp_path, "s", as_of=date(2026, 1, 15)).path == early.path
    assert latest_snapshot(tmp_path, "s").path == late.path


def test_build_polls_respects_as_of(tmp_path: Path):
    w = SnapshotWriter(tmp_path, "fivethirtyeight_archive", now=datetime(2026, 1, 1, tzinfo=UTC))
    w.write("generic_ballot_polls_historical.csv", GENERIC_538.encode())
    w.commit()
    cfg = Config(name="t", data_dir=tmp_path, sources={"fivethirtyeight_archive": {"enabled": True}},
                 average=AverageConfig(cycle=2022), indicators={})
    assert build_polls(cfg, as_of=date(2022, 10, 10))["poll_id"].nunique() == 2
    assert build_polls(cfg)["poll_id"].nunique() == 3


NYT_HOUSE = """poll_id,pollster,start_date,end_date,question_id,sample_size,population,created_at,cycle,office_type,state,seat_number,stage,partisan,party,candidate_name,pct,ranked_choice_round,ranked_choice_final
7,Siena,10/1/26,10/3/26,70,500,lv,10/5/26,2026,U.S. House,AK,1,general,,REP,Nick Begich,52,,
7,Siena,10/1/26,10/3/26,70,500,lv,10/5/26,2026,U.S. House,AK,1,general,,IND,Bill Hill,46,,
7,Siena,10/1/26,10/3/26,70,500,lv,10/5/26,2026,U.S. House,AK,1,general,,DEM,Eric Hafner,2,,
8,Emerson,10/1/26,10/2/26,80,900,lv,10/4/26,2026,U.S. House,US,,general,,DEM,Democrat,50,,
8,Emerson,10/1/26,10/2/26,80,900,lv,10/4/26,2026,U.S. House,US,,general,,REP,Republican,44,,
9,Siena,10/1/26,10/3/26,90,500,lv,10/5/26,2026,U.S. House,AK,1,general,,REP,Nick Begich,45,1,FALSE
9,Siena,10/1/26,10/3/26,90,500,lv,10/5/26,2026,U.S. House,AK,1,general,,IND,Bill Hill,40,1,FALSE
"""


def test_nyt_house_file_gives_generic_ballot_districts_and_final_rounds(tmp_path: Path):
    f = tmp_path / "house.csv"
    f.write_text(NYT_HOUSE)
    df = parse_538_csv(f, source="nyt", default_race_type="house")
    assert set(df.loc[df["race_type"] == "generic", "state"]) == {""}
    district = df[df["race_type"] == "house"]
    assert set(district["district"]) == {"AK-AL"} and set(district["state"]) == {"Alaska"}
    assert set(district["question_id"]) == {"70"}  # round 1 of a ranked-choice poll dropped


def test_main_challenger_picks_the_independent_when_he_is_the_real_opponent(tmp_path: Path):
    from pollcast.average import main_challenger

    f = tmp_path / "house.csv"
    f.write_text(NYT_HOUSE)
    df = parse_538_csv(f, source="nyt", default_race_type="house")
    challenger, margins = main_challenger(df[df["district"] == "AK-AL"],
                                          AverageConfig(race_type="house", cycle=2026))
    assert challenger == "IND"
    assert margins["margin"].tolist() == [-6]
