import numpy as np
import pytest
from scipy.stats import poisson
from simulation import true_model

from valuemodel.markets import market_probabilities, predict


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


def test_fair_odds_are_reciprocals() -> None:
    model = true_model(20, -0.1, np.random.default_rng(3))
    markets = predict(model, "Team 00", "Team 01")
    odds = markets.fair_odds()
    assert odds["home"] == pytest.approx(1 / markets.home)
    assert odds["under25"] == pytest.approx(1 / markets.under25)
    assert set(odds) == {"home", "draw", "away", "over25", "under25"}
