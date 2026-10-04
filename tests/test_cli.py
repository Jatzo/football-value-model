from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FakeSource
from simulation import simulate_league, true_model

from valuemodel import cli


@pytest.fixture
def fake_site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource) -> FakeSource:
    monkeypatch.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VALUEMODEL_REQUEST_DELAY", "0")
    monkeypatch.setattr(cli, "make_client", source.client)
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
def simulated_cache(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Replace the cached files with a simulated league covering history and tuning seasons."""
    rng = np.random.default_rng(5)
    model = true_model(10, -0.1, rng)
    seasons = [("2021", "2020-08-01"), ("2122", "2021-08-01"), ("2223", "2022-08-01")]
    league = pd.concat(
        [simulate_league(model, 2, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    monkeypatch.setattr(cli, "load_matches", lambda *_: league)
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
    assert out.startswith("Team 00 v Team 01\n")
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


def test_missing_cache_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    assert cli.main(["predict", "--home", "Arsenal", "--away", "Chelsea"]) == 1
    assert "valuemodel download" in capsys.readouterr().err
