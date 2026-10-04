"""Season schedules from openfootball, so games beyond the next round can be predicted.

football-data.co.uk only lists matches a few days ahead, once bookmakers have
priced them. openfootball publishes each season's full fixture list as public
domain JSON, without odds, so these games get the model's view alone.
"""

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from valuemodel.backtest import MAIN_MODEL, MODELS
from valuemodel.config import MIN_TEAM_MATCHES, Settings, season_start_year
from valuemodel.data import DownloadError, save_file
from valuemodel.markets import price_matches
from valuemodel.teams import schedule_team

SCHEDULE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
SCHEDULE_FILES: dict[str, str] = {"E0": "en.1", "E1": "en.2"}
SCHEDULE_COLUMNS = ["league", "date", "kickoff", "round", "home_team", "away_team"]


def schedule_url(league: str, season: str) -> str:
    if league not in SCHEDULE_FILES:
        raise ValueError(f"No season schedule source for {league}")
    folder = f"{season_start_year(season)}-{season[2:]}"
    return f"{SCHEDULE_URL}/{folder}/{SCHEDULE_FILES[league]}.json"


def schedule_path(raw_dir: Path, league: str, season: str) -> Path:
    return raw_dir / f"schedule_{league}_{season}.json"


def download_schedule(client: httpx.Client, league: str, season: str, settings: Settings) -> Path:
    """Fetch a season schedule. Dates move and scores are added, so it is always fetched again."""
    url = schedule_url(league, season)
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise DownloadError(f"Could not download {url}: {error}") from error
    try:
        matches = response.json()["matches"]
    except (ValueError, KeyError, TypeError) as error:
        raise DownloadError(f"{url} did not return a season schedule") from error
    if not isinstance(matches, list):
        raise DownloadError(f"{url} did not return a season schedule")
    path = schedule_path(settings.raw_dir, league, season)
    save_file(path, response.content)
    return path


def load_schedule(path: Path, league: str) -> pd.DataFrame:
    """Every game in a saved schedule that has no score yet, with football-data team names."""
    matches = json.loads(path.read_text(encoding="utf-8"))["matches"]
    rows = [
        {
            "league": league,
            "date": pd.Timestamp(match["date"]),
            "kickoff": match.get("time", ""),
            "round": match.get("round", ""),
            "home_team": schedule_team(match["team1"]),
            "away_team": schedule_team(match["team2"]),
        }
        for match in matches
        if match.get("score") is None
    ]
    frame = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    return frame.sort_values(["date", "kickoff", "home_team"], ignore_index=True)


def upcoming_games(schedule: pd.DataFrame, today: date, rounds: int | None) -> pd.DataFrame:
    """Unplayed games from today on, limited to the next few rounds when rounds is given.

    Games dated before today are dropped: they have been played, or postponed
    without a new date, and the schedule simply has no score for them yet.
    """
    ahead = schedule[schedule["date"] >= pd.Timestamp(today)]
    if rounds is None:
        return ahead
    next_rounds = list(dict.fromkeys(ahead["round"]))[:rounds]
    return ahead[ahead["round"].isin(next_rounds)]


def predict_games(history: pd.DataFrame, games: pd.DataFrame, xi: float) -> pd.DataFrame:
    """The main model's chances and expected goals for each game, fitted once on all results.

    Every game is priced from the same fit, so games further ahead use ratings
    that will have moved by the time they are played.
    """
    model = MODELS[MAIN_MODEL](history, games["date"].min(), xi)
    priced = games.join(price_matches(model, games, MIN_TEAM_MATCHES))
    goals = [
        model.expected_goals(home, away) if reliable else (float("nan"), float("nan"))
        for home, away, reliable in zip(
            priced["home_team"], priced["away_team"], priced["reliable"], strict=True
        )
    ]
    priced["home_goals"] = [home for home, _ in goals]
    priced["away_goals"] = [away for _, away in goals]
    return priced
