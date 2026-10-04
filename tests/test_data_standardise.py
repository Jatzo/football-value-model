from pathlib import Path

import pandas as pd
import pytest

from valuemodel.data import (
    MATCH_COLUMNS,
    ODDS_COLUMNS,
    decode,
    parse_dates,
    read_raw,
    standardise,
)

SEASONS = ("0506", "1516", "2526")


def load_fixture(fixtures_dir: Path, season: str) -> pd.DataFrame:
    return standardise(read_raw(fixtures_dir / f"E0_{season}.csv"), "E0", season)


@pytest.mark.parametrize("season", SEASONS)
def test_every_era_maps_to_the_same_schema(fixtures_dir: Path, season: str) -> None:
    frame = load_fixture(fixtures_dir, season)
    assert tuple(frame.columns) == MATCH_COLUMNS + ODDS_COLUMNS
    assert len(frame) == 20
    assert frame["date"].notna().all()
    assert frame["home_goals"].dtype == "Int64"


def test_byte_order_mark_does_not_leak_into_column_names(fixtures_dir: Path) -> None:
    assert (fixtures_dir / "E0_2526.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_raw(fixtures_dir / "E0_2526.csv").columns[0] == "Div"


def test_decode_falls_back_to_latin1() -> None:
    assert decode("Bayern München".encode("latin-1")) == "Bayern München"


def test_two_digit_year_era(fixtures_dir: Path) -> None:
    first = load_fixture(fixtures_dir, "0506").iloc[0]
    assert first["date"] == pd.Timestamp(2005, 8, 13)
    assert (first["home_team"], first["away_team"]) == ("Aston Villa", "Bolton")
    assert (first["home_goals"], first["away_goals"], first["result"]) == (2, 2, "D")
    assert pd.isna(first["kickoff"])


def test_betbrain_columns_map_to_max_and_average(fixtures_dir: Path) -> None:
    first = load_fixture(fixtures_dir, "0506").iloc[0]
    assert first["b365_home"] == 2.3
    assert first["max_home"] == 2.4
    assert first["avg_away"] == 3.05
    assert first["max_over25"] == 2.2
    assert first["avg_under25"] == 1.7
    assert pd.isna(first["pinnacle_home"])
    assert pd.isna(first["pinnacle_close_home"])


def test_pinnacle_closing_era(fixtures_dir: Path) -> None:
    first = load_fixture(fixtures_dir, "1516").iloc[0]
    assert first["date"] == pd.Timestamp(2015, 8, 8)
    assert first["pinnacle_home"] == 1.95
    assert first["pinnacle_draw"] == 3.65
    assert first["pinnacle_close_home"] == 1.82
    assert first["pinnacle_close_away"] == 4.7
    assert first["avg_over25"] == 2.02
    assert pd.isna(first["b365_close_home"])


def test_current_era_with_closing_odds_for_every_source(fixtures_dir: Path) -> None:
    first = load_fixture(fixtures_dir, "2526").iloc[0]
    assert first["date"] == pd.Timestamp(2025, 8, 15)
    assert first["kickoff"] == "20:00"
    assert first["b365_home"] == 1.3
    assert first["pinnacle_home"] == 1.28
    assert first["pinnacle_close_home"] == 1.29
    assert first["max_home"] == 1.34
    assert first["avg_close_away"] == 8.68
    assert first["pinnacle_over25"] == 1.37
    assert first["pinnacle_close_under25"] == 2.95
    assert first["b365_close_over25"] == 1.36


def test_missing_pinnacle_odds_become_nan(fixtures_dir: Path) -> None:
    frame = load_fixture(fixtures_dir, "2526")
    from_gap = frame[frame["date"] >= pd.Timestamp(2026, 1, 17)]
    assert len(from_gap) == 5
    assert from_gap["pinnacle_home"].isna().all()
    assert from_gap["pinnacle_close_home"].isna().all()
    assert from_gap["b365_home"].notna().all()


def test_parse_dates_handles_both_year_lengths() -> None:
    parsed = parse_dates(pd.Series(["13/08/05", "08/08/2015", " 01/02/26 "]))
    assert list(parsed) == [
        pd.Timestamp(2005, 8, 13),
        pd.Timestamp(2015, 8, 8),
        pd.Timestamp(2026, 2, 1),
    ]


def test_parse_dates_rejects_unknown_formats() -> None:
    with pytest.raises(ValueError, match="2015-08-08"):
        parse_dates(pd.Series(["2015-08-08"]))


def test_invalid_odds_become_nan() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["08/08/2015"] * 3,
            "HomeTeam": ["Chelsea"] * 3,
            "AwayTeam": ["Swansea"] * 3,
            "FTHG": ["2"] * 3,
            "FTAG": ["2"] * 3,
            "FTR": ["D"] * 3,
            "B365H": ["1.36", "1", "n/a"],
        }
    )
    odds = standardise(raw, "E0", "1516")["b365_home"]
    assert odds.iloc[0] == 1.36
    assert odds.iloc[1:].isna().all()


def test_alternative_result_column_names_are_accepted() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["08/08/2015"],
            "HomeTeam": ["Chelsea"],
            "AwayTeam": ["Swansea"],
            "HG": ["2"],
            "AG": ["1"],
            "Res": ["H"],
        }
    )
    row = standardise(raw, "E0", "1516").iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"]) == (2, 1, "H")


def test_missing_required_column_is_reported() -> None:
    raw = pd.DataFrame({"Date": ["08/08/2015"], "HomeTeam": ["Chelsea"]})
    with pytest.raises(ValueError, match="missing required columns"):
        standardise(raw, "E0", "1516")
