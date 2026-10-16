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
from valuemodel.markets import price_both_teams_to_score, price_matches
from valuemodel.teams import schedule_team

SCHEDULE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
SCHEDULE_FILES: dict[str, str] = {"E0": "en.1", "E1": "en.2"}
SCHEDULE_COLUMNS = ["league", "date", "kickoff", "round", "home_team", "away_team"]

# A round's games fall from Friday to Monday at most, so each lies within a
# few days of the round's middle date. One further away has been moved.
_ROUND_SPAN = pd.Timedelta(days=3)


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

    Rearranged games keep their original round, so a round label alone is a
    poor guide. Only rounds with at least half as many games left as the
    largest one count, in order of their middle date, which stops a lone
    rearranged game standing in for a whole round. The next rounds then run to
    the last of their games played near that middle date, so a game moved
    weeks later does not pull everything before it in, while any other game
    dated before the cutoff, such as a rearranged one, is still shown.
    """
    ahead = schedule[schedule["date"] >= pd.Timestamp(today)]
    if rounds is None or ahead.empty:
        return ahead
    dates = ahead.groupby("round")["date"]
    sizes, middles = dates.size(), dates.median()
    chosen = middles[sizes * 2 >= sizes.max()].sort_values().index[:rounds]
    in_chosen = ahead["round"].isin(chosen)
    near_middle = (ahead["date"] - ahead["round"].map(middles)).abs() <= _ROUND_SPAN
    cutoff = ahead.loc[in_chosen & near_middle, "date"].max()
    return ahead[ahead["date"] <= cutoff]


def predict_games(history: pd.DataFrame, games: pd.DataFrame, xi: float) -> pd.DataFrame:
    """The main model's chances, including both teams to score, and expected goals per game.

    Every game is priced from the same fit, so games further ahead use ratings
    that will have moved by the time they are played.
    """
    model = MODELS[MAIN_MODEL](history, games["date"].min(), xi)
    priced = games.join(price_matches(model, games, MIN_TEAM_MATCHES)).join(
        price_both_teams_to_score(model, games, MIN_TEAM_MATCHES)
    )
    goals = [
        model.expected_goals(home, away) if reliable else (float("nan"), float("nan"))
        for home, away, reliable in zip(
            priced["home_team"], priced["away_team"], priced["reliable"], strict=True
        )
    ]
    priced["home_goals"] = [home for home, _ in goals]
    priced["away_goals"] = [away for _, away in goals]
    return priced
