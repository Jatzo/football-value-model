"""Walk-forward forecasting: refit on past results only, then price what comes next."""

from collections.abc import Callable, Iterable

import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, TRAINING_WINDOW_DAYS
from valuemodel.data import MATCH_COLUMNS
from valuemodel.markets import OUTCOMES, price_matches
from valuemodel.models.common import FittedModel

FitFunction = Callable[[pd.DataFrame, pd.Timestamp, float, int], FittedModel]

# Days back from each weekday to the afternoon its pre-match odds were collected:
# Friday for games from Friday to Monday, Tuesday for games from Tuesday to Thursday.
_DAYS_SINCE_CAPTURE: dict[int, int] = {0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}


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
    min_matches: int = MIN_TEAM_MATCHES,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Forecast every match in the given seasons from results before its odds were taken.

    The model is refitted once for each odds capture date and prices every
    match that shares it. Matches involving a team with too little history are
    kept but marked unreliable and left unpriced. The result is indexed like
    `matches`, so odds and results can be joined back on.
    """
    targets = matches[matches["season"].isin(list(seasons))]
    frames = []
    for league, league_targets in targets.groupby("league"):
        history = matches[matches["league"] == league]
        for as_of, group in league_targets.groupby(odds_capture_date(league_targets["date"])):
            model = fit(history, as_of, xi, window_days)
            prices = price_matches(model, group, min_matches).assign(as_of=as_of)
            frames.append(group[[c for c in MATCH_COLUMNS if c in group]].join(prices))
    if not frames:
        return pd.DataFrame(columns=["as_of", "reliable", *OUTCOMES])
    return pd.concat(frames).sort_index()
