"""Baseline model: home and away goals as independent Poisson variables."""

import numpy as np
import pandas as pd

from valuemodel.config import TRAINING_WINDOW_DAYS
from valuemodel.models.common import (
    FittedModel,
    TrainingData,
    log_rates,
    minimise,
    prepare,
    rate_gradient,
    starting_params,
    unpack,
)


def negative_log_likelihood(params: np.ndarray, data: TrainingData) -> tuple[float, np.ndarray]:
    """Weighted Poisson negative log likelihood and its gradient.

    The log factorial terms do not depend on the parameters, so they are left
    out. Dividing by the total weight keeps the scale steady as xi changes.
    """
    log_home, log_away = log_rates(params, data)
    home_rate, away_rate = np.exp(log_home), np.exp(log_away)
    w = data.weights / data.weights.sum()
    value = -np.sum(
        w * (data.home_goals * log_home - home_rate + data.away_goals * log_away - away_rate)
    )
    grad = rate_gradient(
        -w * (data.home_goals - home_rate), -w * (data.away_goals - away_rate), data
    )
    return float(value), grad


def fit_poisson(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> FittedModel:
    """Fit on matches played before as_of, weighting recent ones more heavily."""
    data = prepare(matches, as_of, xi, window_days)
    params = minimise(lambda p: negative_log_likelihood(p, data), starting_params(data))
    attack, defence, home_advantage = unpack(params, data.n_teams)
    return FittedModel(
        teams=data.teams,
        attack=attack,
        defence=defence,
        home_advantage=float(home_advantage),
        rho=0.0,
        xi=xi,
        as_of=as_of,
        match_counts=data.match_counts,
    )
