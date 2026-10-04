import numpy as np
import pandas as pd
import pytest
from simulation import simulate_league, true_model

from valuemodel.models.common import FittedModel
from valuemodel.models.poisson import fit_poisson
from valuemodel.tuning import evaluate_xi, walk_forward_forecasts


def drifting_league(seed: int) -> pd.DataFrame:
    """Two seasons in which every team's strength is swapped with another's."""
    rng = np.random.default_rng(seed)
    before = true_model(12, -0.1, rng)
    after = FittedModel(
        **{**before.__dict__, "attack": before.attack[::-1], "defence": before.defence[::-1]}
    )
    return pd.concat(
        [
            simulate_league(before, 4, rng, start="2019-08-01", season="1920"),
            simulate_league(after, 4, rng, start="2020-08-01", season="2021"),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    return drifting_league(seed=0)


def test_forecasts_cover_only_the_target_seasons(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2021"], fit_poisson, xi=0.002, min_matches=0)
    assert len(forecasts) == (league["season"] == "2021").sum()
    assert set(forecasts["season"]) == {"2021"}


def test_every_forecast_is_made_before_its_match(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2021"], fit_poisson, xi=0.002, min_matches=0)
    assert (forecasts["as_of"] <= forecasts["date"]).all()
    assert (forecasts["date"] - forecasts["as_of"] < pd.Timedelta(days=7)).all()
    assert (forecasts["as_of"].dt.dayofweek == 0).all()


def test_forecast_probabilities_sum_to_one(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2021"], fit_poisson, xi=0.002, min_matches=0)
    totals = forecasts[["home", "draw", "away"]].sum(axis=1)
    np.testing.assert_allclose(totals, 1.0, atol=1e-9)


def test_unreliable_matches_are_kept_but_not_priced(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2021"], fit_poisson, xi=0.002, min_matches=10_000)
    assert not forecasts["reliable"].any()
    assert forecasts["home"].isna().all()


def test_evaluate_xi_scores_the_same_matches_for_every_value(league: pd.DataFrame) -> None:
    results = evaluate_xi(league, ["2021"], [0.0, 0.005], fit=fit_poisson, min_matches=0)
    assert list(results["xi"]) == [0.0, 0.005]
    assert results["matches"].nunique() == 1
    assert set(results.columns) == {"xi", "matches", "log_loss", "rps", "brier"}


def test_time_decay_wins_when_team_strengths_change(league: pd.DataFrame) -> None:
    results = evaluate_xi(league, ["2021"], [0.0, 0.01], min_matches=0).set_index("xi")
    assert results.loc[0.01, "rps"] < results.loc[0.0, "rps"]
    assert results.loc[0.01, "log_loss"] < results.loc[0.0, "log_loss"]
