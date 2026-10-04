from collections.abc import Callable
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from simulation import simulate_league, true_model

from valuemodel.models import dixon_coles, poisson
from valuemodel.models.common import (
    FittedModel,
    UnknownTeamError,
    prepare,
    tau,
    time_weights,
    training_window,
)
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.models.poisson import fit_poisson

TRUE_RHO = -0.1


@pytest.fixture(scope="module")
def truth() -> FittedModel:
    return true_model(20, TRUE_RHO, np.random.default_rng(7))


@pytest.fixture(scope="module")
def league(truth: FittedModel) -> pd.DataFrame:
    return simulate_league(truth, rounds=10, rng=np.random.default_rng(8))


@pytest.fixture(scope="module")
def as_of(league: pd.DataFrame) -> pd.Timestamp:
    return league["date"].max() + pd.Timedelta(days=1)


def aligned(fitted: FittedModel, truth: FittedModel, values: np.ndarray) -> np.ndarray:
    return values[[fitted.teams.index(team) for team in truth.teams]]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_dixon_coles_recovers_true_parameters(
    league: pd.DataFrame, truth: FittedModel, as_of: pd.Timestamp
) -> None:
    fitted = fit_dixon_coles(league, as_of, xi=0.0)
    assert rms(aligned(fitted, truth, fitted.attack), truth.attack) < 0.08
    assert rms(aligned(fitted, truth, fitted.defence), truth.defence) < 0.08
    assert fitted.home_advantage == pytest.approx(truth.home_advantage, abs=0.06)
    assert fitted.rho == pytest.approx(TRUE_RHO, abs=0.06)


def test_poisson_recovers_team_strengths(
    league: pd.DataFrame, truth: FittedModel, as_of: pd.Timestamp
) -> None:
    fitted = fit_poisson(league, as_of, xi=0.0)
    assert rms(aligned(fitted, truth, fitted.attack), truth.attack) < 0.08
    assert rms(aligned(fitted, truth, fitted.defence), truth.defence) < 0.08
    assert fitted.home_advantage == pytest.approx(truth.home_advantage, abs=0.06)
    assert fitted.rho == 0.0


@pytest.mark.parametrize("fit", [fit_poisson, fit_dixon_coles])
def test_attack_sums_to_zero(
    fit: Callable[..., FittedModel], league: pd.DataFrame, as_of: pd.Timestamp
) -> None:
    assert fit(league, as_of, xi=0.001).attack.sum() == pytest.approx(0.0, abs=1e-12)


def test_tau_matches_hand_calculation() -> None:
    home_rate, away_rate, rho = 1.5, 1.2, -0.1
    expected = {
        (0, 0): 1 - 1.5 * 1.2 * -0.1,
        (0, 1): 1 + 1.5 * -0.1,
        (1, 0): 1 + 1.2 * -0.1,
        (1, 1): 1 + 0.1,
        (2, 0): 1.0,
        (0, 2): 1.0,
        (2, 2): 1.0,
        (3, 1): 1.0,
    }
    for (home_goals, away_goals), value in expected.items():
        assert tau(home_goals, away_goals, home_rate, away_rate, rho) == pytest.approx(value)
    assert expected[(0, 0)] == pytest.approx(1.18)
    assert expected[(0, 1)] == pytest.approx(0.85)
    assert expected[(1, 0)] == pytest.approx(0.88)


@pytest.mark.parametrize("rho", [-0.15, 0.0, 0.1])
def test_score_matrix_sums_to_one(truth: FittedModel, rho: float) -> None:
    model = FittedModel(**{**truth.__dict__, "rho": rho})
    for home, away in [("Team 00", "Team 01"), ("Team 05", "Team 19")]:
        matrix = model.score_matrix(home, away)
        assert matrix.shape == (11, 11)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-9)
        assert (matrix >= 0).all()


def test_score_matrix_applies_the_low_score_adjustment(truth: FittedModel) -> None:
    independent = FittedModel(**{**truth.__dict__, "rho": 0.0})
    adjusted = FittedModel(**{**truth.__dict__, "rho": -0.1})
    home_rate, away_rate = adjusted.expected_goals("Team 00", "Team 01")
    ratio = adjusted.score_matrix("Team 00", "Team 01") / independent.score_matrix(
        "Team 00", "Team 01"
    )
    assert ratio[0, 0] == pytest.approx(1 + home_rate * away_rate * 0.1, rel=1e-6)
    assert ratio[1, 1] == pytest.approx(1.1, rel=1e-6)
    assert ratio[2, 3] == pytest.approx(1.0, rel=1e-6)


def test_dixon_coles_with_zero_rho_is_the_poisson_model(
    league: pd.DataFrame, as_of: pd.Timestamp
) -> None:
    data = prepare(league, as_of, xi=0.001)
    params = np.random.default_rng(1).normal(0, 0.2, 2 * data.n_teams)
    poisson_value, poisson_grad = poisson.negative_log_likelihood(params, data)
    dc_value, dc_grad = dixon_coles.negative_log_likelihood(np.append(params, 0.0), data)
    assert dc_value == pytest.approx(poisson_value)
    np.testing.assert_allclose(dc_grad[:-1], poisson_grad)


@pytest.mark.parametrize(
    ("module", "extra"), [(poisson, 0), (dixon_coles, 1)], ids=["poisson", "dixon_coles"]
)
def test_analytic_gradient_matches_finite_differences(
    module: ModuleType, extra: int, league: pd.DataFrame, as_of: pd.Timestamp
) -> None:
    data = prepare(league, as_of, xi=0.002)
    rng = np.random.default_rng(2)
    params = rng.normal(0, 0.2, 2 * data.n_teams + extra)
    if extra:
        params[-1] = -0.08
    error = check_grad(
        lambda p: module.negative_log_likelihood(p, data)[0],
        lambda p: module.negative_log_likelihood(p, data)[1],
        params,
    )
    assert error < 1e-5


def test_time_weights_decay_exponentially() -> None:
    as_of = pd.Timestamp("2024-05-01")
    xi = 0.002
    half_life = round(np.log(2) / xi)
    dates = pd.Series([as_of, as_of - pd.Timedelta(days=half_life)])
    weights = time_weights(dates, as_of, xi)
    assert weights[0] == 1.0
    assert weights[1] == pytest.approx(0.5, abs=1e-3)


def test_training_window_excludes_the_prediction_date_and_old_matches() -> None:
    dates = pd.to_datetime(["2020-01-01", "2023-05-01", "2024-04-30", "2024-05-01", "2024-06-01"])
    matches = pd.DataFrame({"date": dates})
    window = training_window(matches, pd.Timestamp("2024-05-01"), window_days=1095)
    assert list(window["date"]) == list(dates[1:3])


@pytest.mark.parametrize("fit", [fit_poisson, fit_dixon_coles])
def test_future_results_do_not_change_the_fit(
    fit: Callable[..., FittedModel], league: pd.DataFrame, as_of: pd.Timestamp
) -> None:
    cutoff = league["date"].iloc[len(league) // 2]
    future = league[league["date"] >= cutoff].copy()
    future["home_goals"] = pd.array([9] * len(future), dtype="Int64")
    future["away_goals"] = pd.array([0] * len(future), dtype="Int64")
    past_only = league[league["date"] < cutoff]
    tampered = pd.concat([past_only, future])

    expected = fit(past_only, cutoff, xi=0.002)
    actual = fit(tampered, cutoff, xi=0.002)
    np.testing.assert_array_equal(actual.attack, expected.attack)
    np.testing.assert_array_equal(actual.defence, expected.defence)
    assert actual.rho == expected.rho


def test_teams_with_few_matches_are_unreliable(league: pd.DataFrame, as_of: pd.Timestamp) -> None:
    newcomer = pd.DataFrame(
        {
            "date": [as_of - pd.Timedelta(days=d) for d in (3, 2)],
            "home_team": ["Newcomers", "Team 00"],
            "away_team": ["Team 01", "Newcomers"],
            "home_goals": pd.array([1, 2], dtype="Int64"),
            "away_goals": pd.array([1, 0], dtype="Int64"),
            "result": ["D", "H"],
            "league": "E0",
        }
    )
    fitted = fit_dixon_coles(pd.concat([league, newcomer]), as_of, xi=0.0)
    assert fitted.match_counts["Newcomers"] == 2
    assert not fitted.is_reliable("Newcomers", min_matches=10)
    assert fitted.is_reliable("Team 00", min_matches=10)
    assert not fitted.is_reliable("Never Seen", min_matches=10)


def test_unknown_team_raises(league: pd.DataFrame, as_of: pd.Timestamp) -> None:
    fitted = fit_poisson(league, as_of, xi=0.0)
    with pytest.raises(UnknownTeamError, match="Never Seen"):
        fitted.score_matrix("Never Seen", "Team 00")


def test_one_league_per_fit(league: pd.DataFrame, as_of: pd.Timestamp) -> None:
    mixed = pd.concat([league, league.assign(league="E1")])
    with pytest.raises(ValueError, match="one league"):
        fit_poisson(mixed, as_of, xi=0.0)


def test_fit_needs_matches_before_the_date(league: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No matches"):
        fit_poisson(league, league["date"].min(), xi=0.0)
