"""Walk-forward forecasting: refit on past results only, then price what comes next."""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, TRAINING_WINDOW_DAYS
from valuemodel.markets import predict
from valuemodel.models.common import FittedModel

FitFunction = Callable[[pd.DataFrame, pd.Timestamp, float, int], FittedModel]
AsOfRule = Callable[[pd.Series], pd.Series]

FORECAST_COLUMNS: tuple[str, ...] = ("home", "draw", "away", "over25", "under25")
MATCH_COLUMNS: tuple[str, ...] = (
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)

# Days back from each weekday to the afternoon its pre-match odds were collected:
# Friday for games from Friday to Monday, Tuesday for games from Tuesday to Thursday.
_DAYS_SINCE_CAPTURE: dict[int, int] = {0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}


def week_start(dates: pd.Series) -> pd.Series:
    """The Monday of each match's week."""
    return dates.dt.to_period("W").dt.start_time


def odds_capture_date(dates: pd.Series) -> pd.Series:
    """The day the data source collected each match's pre-match odds.

    Fitting on results before this date, rather than before the match itself,
    stops a Sunday forecast from using Saturday's results that the Friday price
    could not have known about.
    """
    offsets = dates.dt.dayofweek.map(_DAYS_SINCE_CAPTURE)
    return dates.dt.normalize() - pd.to_timedelta(offsets, unit="D")


def walk_forward_forecasts(
    matches: pd.DataFrame,
    seasons: Iterable[str],
    fit: FitFunction,
    xi: float,
    as_of_rule: AsOfRule = week_start,
    min_matches: int = MIN_TEAM_MATCHES,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Forecast every match in the given seasons from results before its as_of date.

    The model is refitted once for each distinct as_of date and prices every
    match that shares it. Matches involving a team with too little history are
    kept but marked unreliable and left unpriced. The result is indexed like
    `matches`, so odds and results can be joined back on.
    """
    targets = matches[matches["season"].isin(list(seasons))]
    frames = []
    for league, league_targets in targets.groupby("league"):
        history = matches[matches["league"] == league]
        for as_of, group in league_targets.groupby(as_of_rule(league_targets["date"])):
            model = fit(history, as_of, xi, window_days)
            frames.append(_price_group(model, group, as_of, min_matches))
    if not frames:
        return pd.DataFrame(columns=["as_of", "reliable", *FORECAST_COLUMNS])
    return pd.concat(frames).sort_index()


def _price_group(
    model: FittedModel, group: pd.DataFrame, as_of: pd.Timestamp, min_matches: int
) -> pd.DataFrame:
    rows = []
    for match in group.itertuples():
        reliable = model.is_reliable(match.home_team, min_matches) and model.is_reliable(
            match.away_team, min_matches
        )
        prices = predict(model, match.home_team, match.away_team).__dict__ if reliable else {}
        rows.append(
            {
                "as_of": as_of,
                "reliable": reliable,
                **{column: prices.get(column, np.nan) for column in FORECAST_COLUMNS},
            }
        )
    forecasts = pd.DataFrame(rows, index=group.index)
    return group[[column for column in MATCH_COLUMNS if column in group]].join(forecasts)
