"""Choose model settings by walk-forward validation on the tuning seasons.

Forecasts are refitted on the same schedule as the backtest, at each odds
capture date, so each setting is chosen under the conditions it is later judged by.
"""

from collections.abc import Iterable
from functools import partial

import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES, SHOTS_XI
from valuemodel.models.shots_adjusted import fit_shots_adjusted
from valuemodel.scoring import (
    brier_score,
    log_loss,
    outcome_indices,
    ranked_probability_score,
)
from valuemodel.walkforward import FitFunction, walk_forward_forecasts

XI_GRID: tuple[float, ...] = (
    0.0,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.01,
)
SHOT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)


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
    fit: FitFunction = fit_shots_adjusted,
    min_matches: int = MIN_TEAM_MATCHES,
) -> pd.DataFrame:
    """Score walk-forward forecasts for each candidate xi."""
    seasons = list(seasons)
    rows = []
    for xi in xi_values:
        forecasts = walk_forward_forecasts(matches, seasons, fit, xi, min_matches)
        rows.append({"xi": xi, **score_forecasts(forecasts)})
    return pd.DataFrame(rows)


def evaluate_shot_weights(
    matches: pd.DataFrame,
    seasons: Iterable[str],
    weights: Iterable[float] = SHOT_WEIGHT_GRID,
    xi: float = SHOTS_XI,
    min_matches: int = MIN_TEAM_MATCHES,
) -> pd.DataFrame:
    """Score the shots-adjusted model for each share of expected goals in the blend.

    A weight of 0 fits actual goals only and 1 fits shot-based expected goals only.
    """
    seasons = list(seasons)
    rows = []
    for weight in weights:
        fit = partial(fit_shots_adjusted, weight=weight)
        forecasts = walk_forward_forecasts(matches, seasons, fit, xi, min_matches)
        rows.append({"weight": weight, **score_forecasts(forecasts)})
    return pd.DataFrame(rows)
