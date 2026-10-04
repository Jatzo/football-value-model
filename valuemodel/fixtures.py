"""Upcoming fixtures from football-data.co.uk, priced by the model.

The data source publishes one fixtures file covering many leagues, with
pre-match odds from several bookmakers but not Pinnacle. Only leagues with
cached results can be priced.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, Settings, current_season
from valuemodel.data import MATCH_COLUMNS, ODDS_COLUMNS, download_csv, read_raw, standardise
from valuemodel.markets import predict
from valuemodel.models.common import FittedModel
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.odds import find_value
from valuemodel.walkforward import FORECAST_COLUMNS

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
        return pd.DataFrame(columns=[*MATCH_COLUMNS, *ODDS_COLUMNS])
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
    unpriced_leagues: list[str]


def price_fixtures(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    settings: Settings,
    xi: float,
    min_matches: int = MIN_TEAM_MATCHES,
) -> PricedFixtures:
    """Price every fixture in a league that has cached results.

    Each league is fitted once, on all results before its earliest fixture.
    Model probabilities sit in columns such as `home`, the bookmaker's odds in
    `odds_home`, the edge in `edge_home`, and any value bet in `value_1x2` and
    `value_totals`.
    """
    leagues = list(dict.fromkeys(fixtures["league"]))
    known = set(matches["league"])
    priced = [league for league in leagues if league in known]
    frames = []
    for league in priced:
        upcoming = fixtures[fixtures["league"] == league]
        model = fit_dixon_coles(matches[matches["league"] == league], upcoming["date"].min(), xi)
        frames.append(_price_league(model, upcoming, settings, min_matches))
    result = pd.concat(frames) if frames else fixtures.iloc[0:0]
    return PricedFixtures(
        fixtures=result,
        priced_leagues=priced,
        unpriced_leagues=[league for league in leagues if league not in known],
    )


def _price_league(
    model: FittedModel, upcoming: pd.DataFrame, settings: Settings, min_matches: int
) -> pd.DataFrame:
    rows = []
    for fixture in upcoming.itertuples():
        reliable = model.is_reliable(fixture.home_team, min_matches) and model.is_reliable(
            fixture.away_team, min_matches
        )
        prices = predict(model, fixture.home_team, fixture.away_team).__dict__ if reliable else {}
        rows.append({"reliable": reliable, **{c: prices.get(c, np.nan) for c in FORECAST_COLUMNS}})
    forecasts = pd.DataFrame(rows, index=upcoming.index)
    odds = upcoming[[f"{settings.bookmaker}_{outcome}" for outcome in FORECAST_COLUMNS]]
    odds.columns = list(FORECAST_COLUMNS)

    priced = upcoming.join(forecasts)
    for outcome in FORECAST_COLUMNS:
        priced[f"odds_{outcome}"] = odds[outcome]
        priced[f"edge_{outcome}"] = forecasts[outcome] * odds[outcome] - 1
    bets = find_value(
        forecasts.loc[forecasts["reliable"], list(FORECAST_COLUMNS)],
        odds.loc[forecasts["reliable"]],
        settings.edge_threshold,
    )
    for market in ("1x2", "totals"):
        chosen = bets[bets["market"] == market]["outcome"]
        priced[f"value_{market}"] = chosen.reindex(priced.index)
    return priced
