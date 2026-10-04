"""Choose the time decay rate xi by walk-forward validation.

Forecasts are refitted on the same schedule as the backtest, at each odds
capture date, so xi is chosen under the conditions it is later judged by.
"""

from collections.abc import Iterable

import pandas as pd

from valuemodel.config import MIN_TEAM_MATCHES
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.scoring import (
    brier_score,
    log_loss,
    outcome_indices,
    ranked_probability_score,
)
from valuemodel.walkforward import FitFunction, walk_forward_forecasts

XI_GRID: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005)


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
) -> pd.DataFrame:
    """Score walk-forward forecasts for each candidate xi."""
    seasons = list(seasons)
    rows = []
    for xi in xi_values:
        forecasts = walk_forward_forecasts(matches, seasons, fit, xi, min_matches)
        rows.append({"xi": xi, **score_forecasts(forecasts)})
    return pd.DataFrame(rows)
