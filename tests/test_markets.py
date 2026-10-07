import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson
from simulation import true_model

from valuemodel.markets import (
    BOTH_TEAMS_TO_SCORE,
    OUTCOMES,
    both_teams_to_score,
    market_probabilities,
    predict,
    price_both_teams_to_score,
    price_matches,
)
from valuemodel.models.common import FittedModel


def test_markets_from_a_small_matrix() -> None:
    matrix = np.array(
        [
            [0.10, 0.08, 0.02],
            [0.15, 0.12, 0.03],
            [0.20, 0.18, 0.12],
        ]
    )
    markets = market_probabilities(matrix)
    assert markets.home == pytest.approx(0.15 + 0.20 + 0.18)
    assert markets.draw == pytest.approx(0.10 + 0.12 + 0.12)
    assert markets.away == pytest.approx(0.08 + 0.02 + 0.03)
    assert markets.over25 == pytest.approx(0.03 + 0.18 + 0.12)
    assert markets.under25 == pytest.approx(0.10 + 0.08 + 0.02 + 0.15 + 0.12 + 0.20)


def test_over_25_matches_the_poisson_total_goals_distribution() -> None:
    goals = np.arange(11)
    matrix = np.outer(poisson.pmf(goals, 1.6), poisson.pmf(goals, 1.1))
    matrix /= matrix.sum()
    expected_under = poisson.cdf(2, 1.6 + 1.1)
    assert market_probabilities(matrix).under25 == pytest.approx(expected_under, abs=1e-6)


@pytest.mark.parametrize("rho", [-0.12, 0.0, 0.08])
def test_model_markets_sum_to_one(rho: float) -> None:
    model = true_model(20, rho, np.random.default_rng(3))
    for home, away in [("Team 00", "Team 01"), ("Team 07", "Team 13"), ("Team 19", "Team 02")]:
        markets = predict(model, home, away)
        assert markets.home + markets.draw + markets.away == pytest.approx(1.0, abs=1e-9)
        assert markets.over25 + markets.under25 == pytest.approx(1.0, abs=1e-9)


def test_price_matches_leaves_unreliable_matches_unpriced() -> None:
    model = true_model(4, -0.1, np.random.default_rng(3))
    model = FittedModel(**{**model.__dict__, "match_counts": {"Team 00": 12, "Team 01": 12}})
    matches = pd.DataFrame(
        {"home_team": ["Team 00", "Team 02"], "away_team": ["Team 01", "Team 00"]}, index=[7, 9]
    )
    prices = price_matches(model, matches, min_matches=10)
    assert list(prices.index) == [7, 9]
    assert list(prices.columns) == ["reliable", *OUTCOMES]
    assert list(prices["reliable"]) == [True, False]
    assert prices.loc[7, ["home", "draw", "away"]].sum() == pytest.approx(1.0)
    assert prices.loc[9, list(OUTCOMES)].isna().all()


def test_both_teams_to_score_from_a_small_matrix() -> None:
    matrix = np.array(
        [
            [0.10, 0.08, 0.02],
            [0.15, 0.12, 0.03],
            [0.20, 0.18, 0.12],
        ]
    )
    assert both_teams_to_score(matrix) == pytest.approx(0.12 + 0.03 + 0.18 + 0.12)


def test_both_teams_to_score_matches_independent_poisson_goals() -> None:
    home_rate, away_rate = 1.6, 1.1
    goals = np.arange(11)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    expected = (1 - np.exp(-home_rate)) * (1 - np.exp(-away_rate))
    assert both_teams_to_score(matrix) == pytest.approx(expected, abs=1e-6)


def test_both_teams_to_score_for_each_match() -> None:
    model = true_model(6, -0.1, np.random.default_rng(3))
    model = FittedModel(**{**model.__dict__, "match_counts": {"Team 00": 12, "Team 01": 12}})
    matches = pd.DataFrame(
        {"home_team": ["Team 00", "Team 02"], "away_team": ["Team 01", "Team 00"]},
        index=[10, 11],
    )
    priced = price_both_teams_to_score(model, matches, min_matches=10)
    assert list(priced.columns) == list(BOTH_TEAMS_TO_SCORE)
    assert list(priced.index) == [10, 11]
    first = priced.loc[10]
    assert first["btts_yes"] == pytest.approx(
        both_teams_to_score(model.score_matrix("Team 00", "Team 01"))
    )
    assert first["btts_yes"] + first["btts_no"] == pytest.approx(1.0)
    assert priced.loc[11].isna().all()
