import numpy as np
import pandas as pd
import pytest
from simulation import simulate_league, true_model

from valuemodel.models.common import FittedModel
from valuemodel.models.poisson import fit_poisson
from valuemodel.tuning import evaluate_xi


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


def test_evaluate_xi_scores_the_same_matches_for_every_value(league: pd.DataFrame) -> None:
    results = evaluate_xi(league, ["2021"], [0.0, 0.005], fit=fit_poisson, min_matches=0)
    assert list(results["xi"]) == [0.0, 0.005]
    assert results["matches"].nunique() == 1
    assert set(results.columns) == {"xi", "matches", "log_loss", "rps", "brier"}


def test_time_decay_wins_when_team_strengths_change(league: pd.DataFrame) -> None:
    results = evaluate_xi(league, ["2021"], [0.0, 0.01], min_matches=0).set_index("xi")
    assert results.loc[0.01, "rps"] < results.loc[0.0, "rps"]
    assert results.loc[0.01, "log_loss"] < results.loc[0.0, "log_loss"]
