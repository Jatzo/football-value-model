"""Dixon-Coles (1997) model with time decay.

Independent Poisson goals get 0-0 and 1-1 slightly wrong in real football, and
1-0 and 0-1 wrong the other way. Dixon and Coles multiply those four scores by
a factor tau controlled by one extra parameter, rho.
"""

import numpy as np
import pandas as pd

from valuemodel.config import TRAINING_WINDOW_DAYS
from valuemodel.models import poisson
from valuemodel.models.common import (
    FittedModel,
    TrainingData,
    log_rates,
    minimise,
    prepare,
    rate_gradient,
    starting_params,
    tau,
    unpack,
)

# Keeps tau positive for any realistic pair of goal rates. Published estimates
# for top leagues sit well inside this range.
RHO_BOUNDS = (-0.3, 0.3)

# Floor for tau inside the log, so the optimiser never sees log(0) if it
# probes an extreme rho on its way to the answer.
_TAU_FLOOR = 1e-10


def _tau_derivatives(
    data: TrainingData, home_rate: np.ndarray, away_rate: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives of tau with respect to the log home rate, log away rate and rho."""
    x, y = data.home_goals, data.away_goals
    nil_nil = (x == 0) & (y == 0)
    nil_one = (x == 0) & (y == 1)
    one_nil = (x == 1) & (y == 0)
    one_one = (x == 1) & (y == 1)
    both = home_rate * away_rate
    d_log_home = np.where(nil_nil, -both * rho, np.where(nil_one, home_rate * rho, 0.0))
    d_log_away = np.where(nil_nil, -both * rho, np.where(one_nil, away_rate * rho, 0.0))
    d_rho = np.select(
        [nil_nil, nil_one, one_nil, one_one], [-both, home_rate, away_rate, -1.0], 0.0
    )
    return d_log_home, d_log_away, d_rho


def negative_log_likelihood(params: np.ndarray, data: TrainingData) -> tuple[float, np.ndarray]:
    """Poisson likelihood plus the log tau term, with the gradient for both."""
    team_params, rho = params[:-1], params[-1]
    value, grad = poisson.negative_log_likelihood(team_params, data)

    log_home, log_away = log_rates(team_params, data)
    home_rate, away_rate = np.exp(log_home), np.exp(log_away)
    factor = np.maximum(
        tau(data.home_goals, data.away_goals, home_rate, away_rate, rho), _TAU_FLOOR
    )
    w = data.weights / data.weights.sum()
    value -= np.sum(w * np.log(factor))

    d_log_home, d_log_away, d_rho = _tau_derivatives(data, home_rate, away_rate, rho)
    grad = grad + rate_gradient(-w * d_log_home / factor, -w * d_log_away / factor, data)
    grad_rho = -np.sum(w * d_rho / factor)
    return float(value), np.append(grad, grad_rho)


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> FittedModel:
    """Fit on matches played before as_of, weighting recent ones more heavily."""
    data = prepare(matches, as_of, xi, window_days)
    start = np.append(starting_params(data), 0.0)
    bounds = [(None, None)] * (len(start) - 1) + [RHO_BOUNDS]
    params = minimise(lambda p: negative_log_likelihood(p, data), start, bounds)
    attack, defence, home_advantage = unpack(params[:-1], data.n_teams)
    return FittedModel(
        teams=data.teams,
        attack=attack,
        defence=defence,
        home_advantage=float(home_advantage),
        rho=float(params[-1]),
        as_of=as_of,
        match_counts=data.match_counts,
    )
