import numpy as np
import pandas as pd
import pytest
from simulation import add_odds, simulate_league, true_model

from valuemodel.config import Settings
from valuemodel.fixtures import price_fixtures
from valuemodel.picks import PICK_COLUMNS, price_to_beat, value_bets


@pytest.fixture(scope="module")
def priced() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    model = true_model(8, -0.1, rng)
    history = simulate_league(model, 3, rng, start="2025-08-01", season="2526")
    upcoming = add_odds(
        simulate_league(model, 1, rng, start="2026-08-01", season="2627"), model, rng
    ).iloc[:8]
    # Long prices on two home wins guarantee value, the second one bigger.
    upcoming.loc[upcoming.index[0], "b365_home"] = 20.0
    upcoming.loc[upcoming.index[1], "b365_home"] = 40.0
    newcomer = upcoming.iloc[[2]].assign(home_team="Newcomers", b365_home=60.0)
    fixtures = pd.concat([upcoming, newcomer], ignore_index=True)
    return price_fixtures(history, fixtures, Settings(), xi=0.003).fixtures


def test_picks_are_value_bets_ranked_by_edge(priced: pd.DataFrame) -> None:
    picks = value_bets(priced, Settings())
    assert list(picks.columns) == PICK_COLUMNS
    assert len(picks) >= 2
    assert (picks["edge"] >= Settings().edge_threshold).all()
    assert picks["edge"].is_monotonic_decreasing
    top = picks.iloc[0]
    assert (top["outcome"], top["odds"]) == ("home", 40.0)
    assert top["edge"] == pytest.approx(top["probability"] * top["odds"] - 1)
    assert top["fair_odds"] == pytest.approx(1 / top["probability"])


def test_one_pick_per_market_per_match(priced: pd.DataFrame) -> None:
    picks = value_bets(priced, Settings())
    assert not picks.duplicated(["home_team", "away_team", "date", "market"]).any()


def test_teams_with_little_history_are_never_picked(priced: pd.DataFrame) -> None:
    picks = value_bets(priced, Settings())
    assert "Newcomers" not in set(picks["home_team"])


def test_stakes_follow_the_staking_settings(priced: pd.DataFrame) -> None:
    kelly = value_bets(priced, Settings())
    assert (kelly["stake"] > 0).all()
    assert (kelly["stake"] <= 0.02 * 1000 + 1e-9).all()
    flat = value_bets(priced, Settings(staking="flat"))
    assert (flat["stake"] == 10.0).all()


def test_no_value_gives_an_empty_table(priced: pd.DataFrame) -> None:
    picks = value_bets(priced.assign(value_1x2=np.nan, value_totals=np.nan), Settings())
    assert list(picks.columns) == PICK_COLUMNS
    assert picks.empty
    assert value_bets(priced.iloc[0:0], Settings()).empty


@pytest.mark.parametrize("probability", [0.05, 0.25, 0.5, 0.9])
def test_price_to_beat_gives_exactly_the_threshold_edge(probability: float) -> None:
    odds = price_to_beat(probability, 0.03)
    assert probability * odds - 1 == pytest.approx(0.03)


def test_price_to_beat_rejects_impossible_chances() -> None:
    with pytest.raises(ValueError, match="probability"):
        price_to_beat(0.0, 0.03)
