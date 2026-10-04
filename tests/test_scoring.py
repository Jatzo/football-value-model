import numpy as np
import pandas as pd
import pytest

from valuemodel.scoring import brier_score, log_loss, outcome_indices, ranked_probability_score

UNIFORM = np.full((1, 3), 1 / 3)
HOME, DRAW, AWAY = 0, 1, 2


def test_outcome_indices() -> None:
    assert list(outcome_indices(pd.Series(["H", "D", "A", "H"]))) == [0, 1, 2, 0]


def test_perfect_forecast_scores_zero() -> None:
    forecasts = np.eye(3)
    outcomes = np.array([HOME, DRAW, AWAY])
    assert log_loss(forecasts, outcomes) == pytest.approx(0.0)
    assert brier_score(forecasts, outcomes) == 0.0
    assert ranked_probability_score(forecasts, outcomes) == 0.0


def test_uniform_forecast_matches_hand_calculation() -> None:
    home = np.array([HOME])
    assert log_loss(UNIFORM, home) == pytest.approx(np.log(3))
    assert brier_score(UNIFORM, home) == pytest.approx((2 / 3) ** 2 + 2 * (1 / 3) ** 2)
    assert ranked_probability_score(UNIFORM, home) == pytest.approx(
        ((2 / 3) ** 2 + (1 / 3) ** 2) / 2
    )
    assert ranked_probability_score(UNIFORM, np.array([DRAW])) == pytest.approx(
        ((1 / 3) ** 2 + (1 / 3) ** 2) / 2
    )


def test_ranked_probability_score_respects_order() -> None:
    near_miss = np.array([[0.0, 1.0, 0.0]])
    far_miss = np.array([[0.0, 0.0, 1.0]])
    home = np.array([HOME])
    assert brier_score(near_miss, home) == brier_score(far_miss, home)
    assert ranked_probability_score(near_miss, home) == pytest.approx(0.5)
    assert ranked_probability_score(far_miss, home) == pytest.approx(1.0)


def test_scores_average_over_matches() -> None:
    forecasts = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    outcomes = np.array([HOME, HOME])
    assert log_loss(forecasts, outcomes) == pytest.approx(-(np.log(0.5) + np.log(0.2)) / 2)


def test_zero_probability_does_not_give_infinite_log_loss() -> None:
    assert np.isfinite(log_loss(np.array([[0.0, 0.0, 1.0]]), np.array([HOME])))
