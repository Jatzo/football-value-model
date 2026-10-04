"""Scores for judging 1X2 probability forecasts. Lower is better for all three.

Forecasts are arrays of shape (n, 3) holding home, draw and away probabilities.
Outcomes are 0 for a home win, 1 for a draw and 2 for an away win.
"""

import numpy as np
import pandas as pd

OUTCOME_INDEX: dict[str, int] = {"H": 0, "D": 1, "A": 2}

# Stops a forecast of exactly zero from making the log loss infinite.
_PROBABILITY_FLOOR = 1e-15


def outcome_indices(results: pd.Series) -> np.ndarray:
    return results.map(OUTCOME_INDEX).to_numpy(dtype=int)


def _one_hot(outcomes: np.ndarray) -> np.ndarray:
    return np.eye(3)[outcomes]


def log_loss(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log of the probability given to what actually happened."""
    chosen = forecasts[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log(np.maximum(chosen, _PROBABILITY_FLOOR))))


def brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error across all three outcomes, from 0 (perfect) to 2."""
    return float(np.mean(np.sum((forecasts - _one_hot(outcomes)) ** 2, axis=1)))


def ranked_probability_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score on cumulative probabilities, so it respects the order H, D, A.

    Calling a draw when the home side wins is penalised less than calling an
    away win, which suits football results better than the plain Brier score.
    """
    cumulative_forecast = np.cumsum(forecasts, axis=1)[:, :-1]
    cumulative_outcome = np.cumsum(_one_hot(outcomes), axis=1)[:, :-1]
    return float(np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1) / 2))
