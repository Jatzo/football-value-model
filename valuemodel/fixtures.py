"""Upcoming fixtures from football-data.co.uk, priced by the model.

The data source publishes one fixtures file covering many leagues, with
pre-match odds from several bookmakers but not Pinnacle. Only leagues with
cached results can be priced.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, Settings, current_season
from valuemodel.data import (
    MATCH_COLUMNS,
    ODDS_COLUMNS,
    SHOT_COLUMNS,
    download_csv,
    odds_columns,
    read_raw,
    standardise,
)
from valuemodel.markets import OUTCOMES, price_matches
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.odds import MARKETS, find_value

FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"


def fixtures_path(settings: Settings) -> Path:
    return settings.raw_dir / "fixtures.csv"


def download_fixtures(client: httpx.Client, settings: Settings) -> Path:
    """Fetch the latest fixtures file. It changes often, so it is always downloaded again."""
    path = fixtures_path(settings)
    download_csv(client, FIXTURES_URL, path)
    return path


def fetched_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def load_fixtures(path: Path) -> pd.DataFrame:
    """Read the fixtures file into the standard columns, one league at a time."""
    raw = read_raw(path)
    if raw.empty:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, *SHOT_COLUMNS, *ODDS_COLUMNS])
    season = current_season()
    frames = [
        standardise(group, str(league), season, has_results=False)
        for league, group in raw.groupby("Div", sort=False)
    ]
    return pd.concat(frames).sort_values(["date", "kickoff", "league", "home_team"])


@dataclass
class PricedFixtures:
    fixtures: pd.DataFrame
    priced_leagues: list[str]
    unpriced_leagues: dict[str, str]


def price_fixtures(
    matches: pd.DataFrame, fixtures: pd.DataFrame, settings: Settings, xi: float
) -> PricedFixtures:
    """Price every fixture in a league that has cached results.

    Each league is fitted once, on all results before its earliest fixture.
    Model probabilities sit in columns such as `home`, the bookmaker's odds in
    `odds_home`, the edge in `edge_home`, and any value bet in `value_1x2` and
    `value_totals`. Leagues that cannot be priced are returned with the reason.
    """
    known = set(matches["league"])
    frames, priced, unpriced = [], [], {}
    for league in dict.fromkeys(fixtures["league"]):
        if league not in known:
            unpriced[league] = "no cached results"
            continue
        upcoming = fixtures[fixtures["league"] == league]
        history = matches[matches["league"] == league]
        try:
            model = fit_dixon_coles(history, upcoming["date"].min(), xi)
        except (ValueError, RuntimeError) as error:
            unpriced[league] = str(error)
            continue
        frames.append(
            _with_odds(upcoming, price_matches(model, upcoming, MIN_TEAM_MATCHES), settings)
        )
        priced.append(league)
    result = pd.concat(frames) if frames else fixtures.iloc[0:0]
    return PricedFixtures(fixtures=result, priced_leagues=priced, unpriced_leagues=unpriced)


def _with_odds(upcoming: pd.DataFrame, forecasts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add the bookmaker's odds, the edge on every outcome and any value bets."""
    odds = upcoming[odds_columns(settings.bookmaker)]
    odds.columns = list(OUTCOMES)
    priced = upcoming.join(forecasts)
    for outcome in OUTCOMES:
        priced[f"odds_{outcome}"] = odds[outcome]
        priced[f"edge_{outcome}"] = forecasts[outcome] * odds[outcome] - 1
    reliable = forecasts["reliable"]
    bets = find_value(
        forecasts.loc[reliable, list(OUTCOMES)], odds.loc[reliable], settings.edge_threshold
    )
    for market in MARKETS:
        chosen = bets[bets["market"] == market]["outcome"]
        priced[f"value_{market}"] = chosen.reindex(priced.index)
    return priced
