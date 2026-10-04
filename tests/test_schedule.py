import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from conftest import FakeSource
from simulation import simulated_seasons

from valuemodel.config import Settings
from valuemodel.data import DownloadError
from valuemodel.markets import OUTCOMES
from valuemodel.schedule import (
    SCHEDULE_FILES,
    download_schedule,
    load_schedule,
    predict_games,
    schedule_path,
    schedule_url,
    upcoming_games,
)
from valuemodel.teams import SCHEDULE_NAMES, schedule_team


def test_schedule_url_uses_the_season_folder() -> None:
    assert schedule_url("E1", "2627") == (
        "https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/en.2.json"
    )


def test_leagues_without_a_schedule_are_rejected() -> None:
    with pytest.raises(ValueError, match="No season schedule source for SP1"):
        schedule_url("SP1", "2627")


def test_download_saves_the_schedule(source: FakeSource, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with source.client() as client:
        path = download_schedule(client, "E0", "2627", settings)
    assert path == schedule_path(settings.raw_dir, "E0", "2627")
    assert json.loads(path.read_text(encoding="utf-8"))["name"].startswith("English Premier")


@pytest.mark.parametrize("body", [b"<html>Not found</html>", b'{"name": "x"}', b'{"matches": 1}'])
def test_download_rejects_anything_but_a_schedule(
    source: FakeSource, tmp_path: Path, body: bytes
) -> None:
    source.body = body
    settings = Settings(data_dir=tmp_path)
    with source.client() as client, pytest.raises(DownloadError, match="season schedule"):
        download_schedule(client, "E0", "2627", settings)
    assert not schedule_path(settings.raw_dir, "E0", "2627").exists()


def test_download_failure_is_reported(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    with client, pytest.raises(DownloadError, match="Could not download"):
        download_schedule(client, "E0", "2627", Settings(data_dir=tmp_path))


def test_load_keeps_only_unplayed_games_with_data_source_names(fixtures_dir: Path) -> None:
    schedule = load_schedule(fixtures_dir / "schedule_E0_2627.json", "E0")
    # Played games carry a score as a dict or, in some rounds, a bare list.
    assert len(schedule) == 20
    assert set(schedule["round"]) == {"Matchday 6", "Matchday 7"}
    first = schedule.iloc[0]
    assert (first["date"], first["kickoff"]) == (pd.Timestamp("2026-10-10"), "12:30")
    assert (first["home_team"], first["away_team"]) == ("Arsenal", "Leeds")
    assert set(schedule["league"]) == {"E0"}


@pytest.mark.parametrize("league", list(SCHEDULE_FILES))
def test_every_team_in_the_saved_schedules_has_a_data_source_name(
    fixtures_dir: Path, league: str
) -> None:
    raw = json.loads((fixtures_dir / f"schedule_{league}_2627.json").read_text(encoding="utf-8"))
    teams = {match[side] for match in raw["matches"] for side in ("team1", "team2")}
    assert len(teams) == {"E0": 20, "E1": 24}[league]
    assert teams <= set(SCHEDULE_NAMES)


def test_names_match_the_results_data(fixtures_dir: Path) -> None:
    results = pd.read_csv(fixtures_dir / "E0_2526.csv", encoding="latin-1")
    names = set(results["HomeTeam"]) | set(results["AwayTeam"])
    for team in ("Arsenal", "Man United", "Nott'm Forest", "Tottenham"):
        assert team in names
        assert team in SCHEDULE_NAMES.values()


def test_unknown_schedule_team_is_named() -> None:
    with pytest.raises(ValueError, match="'Gotham City FC' has no football-data name"):
        schedule_team("Gotham  City FC")


def _schedule(rounds: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": "E0",
            "date": [pd.Timestamp(day) for day, _ in rounds],
            "kickoff": "15:00",
            "round": [name for _, name in rounds],
            "home_team": "Team 00",
            "away_team": "Team 01",
        }
    )


def test_upcoming_games_drop_past_dates_and_keep_the_next_rounds() -> None:
    schedule = _schedule(
        [
            ("2026-09-20", "Matchday 5"),
            ("2026-10-10", "Matchday 6"),
            ("2026-10-11", "Matchday 6"),
            ("2026-10-17", "Matchday 7"),
            ("2026-10-24", "Matchday 8"),
        ]
    )
    today = date(2026, 10, 4)
    assert list(upcoming_games(schedule, today, rounds=2)["round"]) == [
        "Matchday 6",
        "Matchday 6",
        "Matchday 7",
    ]
    assert len(upcoming_games(schedule, today, rounds=None)) == 4
    assert upcoming_games(schedule, date(2026, 11, 1), rounds=3).empty


def test_predictions_cover_every_outcome_and_skip_teams_with_little_history() -> None:
    history = simulated_seasons(5, 10, 2, with_odds=False)
    games = _schedule([("2024-08-10", "Matchday 1"), ("2024-08-17", "Matchday 2")])
    games.loc[1, "away_team"] = "Newly Promoted"
    priced = predict_games(history, games, xi=0.006)

    known = priced.iloc[0]
    assert known["reliable"]
    assert known["home"] + known["draw"] + known["away"] == pytest.approx(1, abs=1e-6)
    assert known["over25"] + known["under25"] == pytest.approx(1, abs=1e-6)
    assert known["home_goals"] > 0 and known["away_goals"] > 0

    unknown = priced.iloc[1]
    assert not unknown["reliable"]
    assert np.isnan([unknown[o] for o in OUTCOMES]).all()
    assert np.isnan(unknown["home_goals"])
