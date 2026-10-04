import logging
from pathlib import Path

import pandas as pd
import pytest

from valuemodel.data import find_problems, load_season, read_raw, standardise


def matches(rows: list[tuple[str, str, str, str, str, str]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    return standardise(raw, "E0", "2425")


def test_clean_fixture_has_no_problems(fixtures_dir: Path) -> None:
    frame = standardise(read_raw(fixtures_dir / "E0_2526.csv"), "E0", "2526")
    assert find_problems(frame).empty


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (("16/08/2024", "Man United", "Fulham", "1", "0", "A"), "result does not match the score"),
        (("16/08/2024", "Man United", "Fulham", "1", "0", "X"), "result is not H, D or A"),
        (("16/08/2024", "Man United", "Fulham", "", "0", "H"), "missing results data"),
        (("16/08/2024", "Fulham", "Fulham", "1", "1", "D"), "team plays itself"),
    ],
)
def test_bad_rows_are_flagged(row: tuple[str, str, str, str, str, str], reason: str) -> None:
    good = ("17/08/2024", "Ipswich", "Liverpool", "0", "2", "A")
    problems = find_problems(matches([good, row]))
    assert list(problems.index) == [1]
    assert reason in problems.iloc[0]["problem"]


def test_duplicate_fixture_flags_both_rows() -> None:
    row = ("16/08/2024", "Man United", "Fulham", "1", "0", "H")
    assert list(find_problems(matches([row, row])).index) == [0, 1]


def test_several_reasons_are_joined() -> None:
    problems = find_problems(matches([("16/08/2024", "Fulham", "Fulham", "1", "0", "D")]))
    assert problems.iloc[0]["problem"] == "result does not match the score; team plays itself"


def test_load_season_drops_and_logs_bad_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,17/08/2024,Ipswich,Liverpool,0,2,A\n"
        "E0,16/08/2024,Man United,Fulham,1,0,A\n"
        "E0,16/08/2024,Arsenal,Wolves,2,0,H\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        frame = load_season(path, "E0", "2425")
    assert list(frame["home_team"]) == ["Arsenal", "Ipswich"]
    assert "Man United v Fulham" in caplog.text
