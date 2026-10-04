"""Poisson model fitted to a blend of actual goals and shot-based expected goals.

Goals are noisy: a side can dominate and lose. Shots carry extra evidence about
how strong each team is, so fitting to a mix of goals and expected goals can
estimate team strengths better than either alone. The Dixon-Coles adjustment
only applies to whole-number scores, so it has nothing to act on here and the
model is fitted as Poisson.
"""

import pandas as pd

from valuemodel.config import SHOT_WEIGHT, TRAINING_WINDOW_DAYS
from valuemodel.expected_goals import blend_goals, fit_shot_values
from valuemodel.models.common import FittedModel, training_window
from valuemodel.models.poisson import fit_poisson


def fit_shots_adjusted(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float,
    window_days: int = TRAINING_WINDOW_DAYS,
    weight: float = SHOT_WEIGHT,
) -> FittedModel:
    """Fit on matches before as_of, with shot values learned from those same matches."""
    window = training_window(matches, as_of, window_days)
    blended = blend_goals(window, fit_shot_values(window), weight)
    return fit_poisson(blended, as_of, xi, window_days)
