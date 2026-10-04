from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from conftest import FakeSource
from simulation import simulated_seasons

from valuemodel import cli


@pytest.fixture
def fake_site(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> FakeSource:
    clean_environment.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    clean_environment.setenv("VALUEMODEL_REQUEST_DELAY", "0")
    clean_environment.setattr(cli, "make_client", source.client)
    return source


def test_download_reports_each_season(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["download", "--seasons", "1516", "2526"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "E0 1516  downloaded   20 matches  Pinnacle closing odds for 20",
        "E0 2526  downloaded   20 matches  Pinnacle closing odds for 15",
    ]


def test_second_run_uses_the_cache(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["download", "--seasons", "2526"])
    cli.main(["download", "--seasons", "2526"])
    assert len(fake_site.requests) == 1
    assert "cached" in capsys.readouterr().out.splitlines()[-1]


def test_download_failure_returns_an_error_code(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["download", "--seasons", "9900"]) == 1
    assert "Could not download" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["download", "--seasons", "2426"], "does not cover two consecutive years"),
        (["download", "--leagues", "XX"], "Unknown league 'XX'"),
    ],
)
def test_bad_arguments_are_rejected(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.main(args)
    assert message in capsys.readouterr().err


@pytest.fixture
def simulated_cache(clean_environment: pytest.MonkeyPatch) -> pd.DataFrame:
    """Replace the cached files with a simulated league covering history and tuning seasons."""
    seasons = [("2021", "2020-08-01"), ("2122", "2021-08-01"), ("2223", "2022-08-01")]
    league = simulated_seasons(5, 10, 2, seasons=seasons, with_odds=False)
    clean_environment.setattr(cli, "load_matches", lambda *_: league)
    clean_environment.setattr(cli, "load_available", lambda *_: league)
    return league


def test_tune_xi_prints_a_row_per_value(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["tune-xi", "--model", "poisson", "--xi", "0.001", "0.003"]) == 0
    out = capsys.readouterr().out
    assert "0.0010" in out
    assert "0.0030" in out
    assert "Lowest ranked probability score at xi =" in out


def test_predict_prints_fair_odds(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["predict", "--home", "Team 00", "--away", " Team 01 "]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Team 00 v Team 01\nShots-adjusted, fitted on E0")
    for label in ("Home win", "Draw", "Away win", "Over 2.5", "Under 2.5"):
        assert label in out


def test_predict_refuses_teams_with_little_history(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    second_match_day = sorted(simulated_cache["date"].unique())[1]
    as_of = pd.Timestamp(second_match_day).strftime("%Y-%m-%d")
    assert cli.main(["predict", "--home", "Team 00", "--away", "Team 01", "--as-of", as_of]) == 1
    assert "fewer than the 10 needed" in capsys.readouterr().err


def test_predict_unknown_team(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["predict", "--home", "Team 00", "--away", "Nobody"]) == 1
    assert "Nobody has no matches" in capsys.readouterr().err


def test_predict_suggests_close_team_names(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["predict", "--home", "Teem 03", "--away", "Team 01"]) == 1
    assert "Did you mean Team 03" in capsys.readouterr().err


def test_missing_cache_is_reported(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_environment.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    assert cli.main(["predict", "--home", "Arsenal", "--away", "Chelsea"]) == 1
    assert "valuemodel download" in capsys.readouterr().err


def test_predict_with_odds_shows_edges_and_a_paper_bet(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["predict", "--home", "Team 00", "--away", "Team 01"]
    assert cli.main([*args, "--odds", "50", "50", "50", "--totals-odds", "1.01", "1.01"]) == 0
    out = capsys.readouterr().out
    assert "Book chance" in out
    assert "Bookmaker margin, match result" in out
    assert "Bookmaker margin, over/under 2.5" in out
    assert out.count("value") == 1
    assert "Paper bet:" in out
    assert "of a 1000 unit bankroll (quarter Kelly)" in out


def test_predict_with_poor_odds_finds_no_value(
    simulated_cache: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["predict", "--home", "Team 00", "--away", "Team 01", "--odds", "1.01", "1.01", "1.01"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "No outcome reaches the 3% edge threshold" in out
    assert "Paper bet" not in out


def test_flat_staking_is_described(
    simulated_cache: pd.DataFrame,
    clean_environment: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_environment.setenv("VALUEMODEL_STAKING", "flat")
    args = ["predict", "--home", "Team 00", "--away", "Team 01", "--odds", "50", "50", "50"]
    assert cli.main(args) == 0
    assert (
        "stake 10.00 of a 1000 unit bankroll (flat stakes of 1% of the starting bankroll)"
        in capsys.readouterr().out
    )


def test_odds_must_be_above_one(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["predict", "--home", "A", "--away", "B", "--odds", "1.0", "3", "4"])
    assert "decimal odds must be greater than 1" in capsys.readouterr().err


def test_bad_setting_is_reported(
    simulated_cache: pd.DataFrame,
    clean_environment: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_environment.setenv("VALUEMODEL_MARGIN_METHOD", "shin")
    assert cli.main(["predict", "--home", "Team 00", "--away", "Team 01"]) == 1
    assert "VALUEMODEL_MARGIN_METHOD must be one of" in capsys.readouterr().err


@pytest.fixture
def simulated_cache_with_odds(clean_environment: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    league = simulated_seasons(6, 8, 2)
    clean_environment.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    clean_environment.setattr(cli, "load_matches", lambda *_: league)
    clean_environment.setattr(cli, "load_available", lambda *_: league)
    return tmp_path


def test_backtest_prints_the_report_and_saves_the_run(
    simulated_cache_with_odds: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["backtest", "--seasons", "2324"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Backtest of E0, seasons 2023/24")
    assert "Closing line value" in out
    assert "Saved as run 1 in" in out
    assert (simulated_cache_with_odds / "valuemodel.sqlite").exists()


def test_backtest_without_saving(
    simulated_cache_with_odds: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["backtest", "--seasons", "2324", "--no-save"]) == 0
    assert "Saved as run" not in capsys.readouterr().out
    assert not (simulated_cache_with_odds / "valuemodel.sqlite").exists()


def test_fixtures_downloads_the_file_and_refreshes_this_season(
    fake_site: FakeSource, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "current_season", lambda: "2526")
    assert cli.run_fixtures(["E0"], today=date(2026, 10, 4)) == 0
    out = capsys.readouterr().out
    assert "Fixtures file has 12 matches" in out
    assert "E0 2025/26: 20 results, latest 2026-01-17" in out
    assert "E0 schedule: 20 games to play, next on 2026-10-10" in out
    assert fake_site.requests[0].endswith("/fixtures.csv")
    assert fake_site.requests[1].endswith("/2526/E0.csv")
    assert fake_site.requests[2].endswith("/2025-26/en.1.json")


def test_leagues_without_a_schedule_source_are_named() -> None:
    assert cli._schedule_line("SP1", None, date(2026, 10, 4)) == (
        "SP1 schedule: no source for this league"
    )


def test_fixtures_refreshes_both_english_leagues_by_default() -> None:
    assert cli.build_parser().parse_args(["fixtures"]).leagues == ["E0", "E1"]
    assert cli.build_parser().parse_args(["download"]).leagues == ["E0"]


def test_fixtures_reports_a_failed_download(
    fake_site: FakeSource, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "current_season", lambda: "9900")
    assert cli.main(["fixtures"]) == 1
    assert "Could not download" in capsys.readouterr().err


def test_backtest_names_the_seasons_it_needs(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_environment.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    assert cli.main(["backtest", "--seasons", "1819"]) == 1
    assert "--seasons 1516 1617 1718 1819" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "leagues"), [([], "E0, E1"), (["--single-league"], "E0")])
def test_backtest_names_the_leagues_it_fitted_on(
    simulated_cache_with_odds: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    leagues: str,
) -> None:
    assert cli.main(["backtest", "--seasons", "2324", "--no-save", *flags]) == 0
    assert f"Models fitted on results from {leagues}." in capsys.readouterr().out
