"""Choose the time decay rate xi by walk-forward validation."""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, TRAINING_WINDOW_DAYS
from valuemodel.markets import predict
from valuemodel.models.common import FittedModel
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.scoring import (
    brier_score,
    log_loss,
    outcome_indices,
    ranked_probability_score,
)

FitFunction = Callable[[pd.DataFrame, pd.Timestamp, float, int], FittedModel]

XI_GRID: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005)


def walk_forward_forecasts(
    matches: pd.DataFrame,
    seasons: Iterable[str],
    fit: FitFunction,
    xi: float,
    min_matches: int = MIN_TEAM_MATCHES,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Forecast every match in the given seasons using only results from earlier weeks.

    The model is refitted at the start of each calendar week, then prices all of
    that week's matches. Matches involving a team with too little history are
    kept but marked unreliable, so they can be left out of the scores.
    """
    targets = matches[matches["season"].isin(list(seasons))]
    rows = []
    for league, league_targets in targets.groupby("league"):
        history = matches[matches["league"] == league]
        week_starts = league_targets["date"].dt.to_period("W").dt.start_time
        for as_of, week in league_targets.groupby(week_starts):
            model = fit(history, as_of, xi, window_days)
            for match in week.itertuples():
                reliable = model.is_reliable(match.home_team, min_matches) and model.is_reliable(
                    match.away_team, min_matches
                )
                probabilities = (
                    predict(model, match.home_team, match.away_team) if reliable else None
                )
                rows.append(
                    {
                        "league": league,
                        "season": match.season,
                        "date": match.date,
                        "as_of": as_of,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "result": match.result,
                        "reliable": reliable,
                        "home": probabilities.home if probabilities else np.nan,
                        "draw": probabilities.draw if probabilities else np.nan,
                        "away": probabilities.away if probabilities else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def score_forecasts(forecasts: pd.DataFrame) -> dict[str, float]:
    scored = forecasts[forecasts["reliable"]]
    probabilities = scored[["home", "draw", "away"]].to_numpy()
    outcomes = outcome_indices(scored["result"])
    return {
        "matches": len(scored),
        "log_loss": log_loss(probabilities, outcomes),
        "rps": ranked_probability_score(probabilities, outcomes),
        "brier": brier_score(probabilities, outcomes),
    }


def evaluate_xi(
    matches: pd.DataFrame,
    seasons: Iterable[str],
    xi_values: Iterable[float] = XI_GRID,
    fit: FitFunction = fit_dixon_coles,
    min_matches: int = MIN_TEAM_MATCHES,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Score walk-forward forecasts for each candidate xi."""
    seasons = list(seasons)
    rows = []
    for xi in xi_values:
        forecasts = walk_forward_forecasts(matches, seasons, fit, xi, min_matches, window_days)
        rows.append({"xi": xi, **score_forecasts(forecasts)})
    return pd.DataFrame(rows)
