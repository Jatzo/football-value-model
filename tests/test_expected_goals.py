import numpy as np
import pandas as pd
import pytest
from simulation import simulate_league, true_model

from valuemodel.expected_goals import (
    ShotValues,
    blend_goals,
    expected_goals,
    fit_shot_values,
    has_shots,
)
from valuemodel.models.poisson import fit_poisson
from valuemodel.models.shots_adjusted import fit_shots_adjusted


def with_shots(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Give every match shot counts that track its goals, as real ones do."""
    frame = frame.copy()
    for side in ("home", "away"):
        goals = frame[f"{side}_goals"].astype(int).to_numpy()
        on_target = goals + rng.poisson(2.5, len(frame))
        frame[f"{side}_shots_on_target"] = pd.array(on_target, dtype="Int64")
        frame[f"{side}_shots"] = pd.array(on_target + rng.poisson(6, len(frame)), dtype="Int64")
    return frame


def shot_table(rows: list[tuple[int, int, int, int, int, int]]) -> pd.DataFrame:
    columns = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
    ]
    return pd.DataFrame(rows, columns=columns).astype("Int64")


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    return with_shots(simulate_league(true_model(10, -0.1, rng), 3, rng), rng)


def test_shot_values_match_hand_calculation() -> None:
    # Every goal comes from a shot on target and three in ten shots on target score.
    matches = shot_table([(3, 0, 15, 4, 10, 0), (0, 3, 2, 12, 0, 10), (6, 3, 25, 14, 20, 10)])
    values = fit_shot_values(matches)
    assert values.on_target == pytest.approx(0.3)
    assert values.off_target == pytest.approx(0.0, abs=1e-12)


def test_shot_values_are_never_negative() -> None:
    # More shots off target here goes with fewer goals, which a free fit would
    # turn into a negative value per shot.
    matches = shot_table([(2, 0, 4, 20, 4, 0), (0, 2, 20, 4, 0, 4), (1, 1, 12, 12, 2, 2)])
    values = fit_shot_values(matches)
    assert values.on_target >= 0
    assert values.off_target >= 0


def test_no_shot_counts_gives_no_values() -> None:
    matches = shot_table([(1, 0, 10, 5, 4, 2)]).astype(float)
    matches.loc[:, "home_shots"] = np.nan
    assert not has_shots(matches).any()
    assert fit_shot_values(matches) is None
    assert fit_shot_values(matches.drop(columns="home_shots")) is None


def test_expected_goals_arithmetic() -> None:
    matches = shot_table([(1, 0, 12, 7, 5, 2)])
    home, away = expected_goals(matches, ShotValues(on_target=0.3, off_target=0.05))
    assert home.iloc[0] == pytest.approx(0.3 * 5 + 0.05 * 7)
    assert away.iloc[0] == pytest.approx(0.3 * 2 + 0.05 * 5)


def test_blend_weights() -> None:
    matches = shot_table([(2, 0, 10, 4, 5, 1)])
    values = ShotValues(on_target=0.3, off_target=0.0)
    assert blend_goals(matches, values, 0.0)["home_goals"].iloc[0] == 2.0
    assert blend_goals(matches, values, 1.0)["home_goals"].iloc[0] == pytest.approx(1.5)
    assert blend_goals(matches, values, 0.5)["home_goals"].iloc[0] == pytest.approx(1.75)
    assert blend_goals(matches, None, 0.5)["home_goals"].iloc[0] == 2.0


def test_matches_without_shots_keep_actual_goals() -> None:
    matches = shot_table([(2, 1, 10, 4, 5, 1), (3, 0, 0, 0, 0, 0)]).astype(float)
    matches.loc[1, ["home_shots", "away_shots"]] = np.nan
    blended = blend_goals(matches, ShotValues(on_target=0.3, off_target=0.0), 1.0)
    assert blended["home_goals"].iloc[0] == pytest.approx(1.5)
    assert (blended.loc[1, "home_goals"], blended.loc[1, "away_goals"]) == (3.0, 0.0)


def test_zero_weight_is_the_poisson_model(league: pd.DataFrame) -> None:
    as_of = league["date"].max() + pd.Timedelta(days=1)
    adjusted = fit_shots_adjusted(league, as_of, 0.003, weight=0.0)
    plain = fit_poisson(league, as_of, 0.003)
    np.testing.assert_allclose(adjusted.attack, plain.attack)
    np.testing.assert_allclose(adjusted.defence, plain.defence)


def test_shots_change_the_fit(league: pd.DataFrame) -> None:
    as_of = league["date"].max() + pd.Timedelta(days=1)
    adjusted = fit_shots_adjusted(league, as_of, 0.003, weight=0.5)
    plain = fit_poisson(league, as_of, 0.003)
    assert np.abs(adjusted.attack - plain.attack).max() > 0.01


def test_later_shot_counts_do_not_reach_the_fit(league: pd.DataFrame) -> None:
    cutoff = league["date"].iloc[len(league) // 2]
    tampered = league.copy()
    later = tampered["date"] >= cutoff
    tampered.loc[later, ["home_shots", "home_shots_on_target"]] = 40
    honest = fit_shots_adjusted(league, cutoff, 0.003)
    leaked = fit_shots_adjusted(tampered, cutoff, 0.003)
    np.testing.assert_array_equal(honest.attack, leaked.attack)
