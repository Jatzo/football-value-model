import numpy as np
import pandas as pd
import pytest

from valuemodel.odds import (
    check_quotes,
    edge,
    find_value,
    implied_probabilities,
    overround,
    remove_margin,
)

ODDS = np.array([2.0, 3.5, 4.0])


def test_implied_probabilities_and_overround() -> None:
    np.testing.assert_allclose(implied_probabilities(ODDS), [0.5, 1 / 3.5, 0.25])
    assert overround(ODDS) == pytest.approx(0.5 + 1 / 3.5 + 0.25 - 1)
    assert overround(ODDS) == pytest.approx(0.035714, abs=1e-6)


def test_odds_of_one_or_less_are_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        implied_probabilities(np.array([1.0, 3.0, 4.0]))


def test_proportional_method_matches_hand_calculation() -> None:
    total = 0.5 + 1 / 3.5 + 0.25
    expected = [0.5 / total, (1 / 3.5) / total, 0.25 / total]
    np.testing.assert_allclose(remove_margin(ODDS, "proportional"), expected)
    np.testing.assert_allclose(
        remove_margin(ODDS, "proportional"), [0.482759, 0.275862, 0.241379], atol=1e-6
    )


def test_power_method_uses_one_exponent_greater_than_one() -> None:
    fair = remove_margin(ODDS, "power")
    assert fair.sum() == pytest.approx(1.0, abs=1e-12)
    exponents = np.log(fair) / np.log(implied_probabilities(ODDS))
    np.testing.assert_allclose(exponents, exponents[0])
    assert exponents[0] > 1


@pytest.mark.parametrize("method", ["power", "proportional"])
def test_fair_odds_are_unchanged(method: str) -> None:
    fair_odds = np.array([2.0, 4.0, 4.0])
    np.testing.assert_allclose(remove_margin(fair_odds, method), [0.5, 0.25, 0.25])


@pytest.mark.parametrize("method", ["power", "proportional"])
def test_many_markets_at_once(method: str) -> None:
    odds = np.array([[2.0, 3.5, 4.0], [1.25, 6.0, 12.0], [1.9, 1.95, np.nan]])
    fair = remove_margin(odds[:, :3], method)
    np.testing.assert_allclose(fair[:2].sum(axis=1), 1.0, atol=1e-12)
    assert np.isnan(fair[2]).all()


def test_two_way_market() -> None:
    fair = remove_margin(np.array([1.9, 1.9]), "power")
    np.testing.assert_allclose(fair, [0.5, 0.5])


def test_power_takes_more_off_the_longshot() -> None:
    odds = np.array([1.25, 6.0, 12.0])
    power = remove_margin(odds, "power")
    proportional = remove_margin(odds, "proportional")
    assert power[2] < proportional[2]
    assert power[0] > proportional[0]


@pytest.mark.parametrize("odds", [[1.004, 1.004], [1.01, 1.01, 1.01], [1.001, 30.0, 60.0]])
def test_power_method_handles_extreme_prices(odds: list[float]) -> None:
    fair = remove_margin(np.array(odds), "power")
    assert fair.sum() == pytest.approx(1.0, abs=1e-12)


def test_negative_margin_is_handled() -> None:
    fair = remove_margin(np.array([2.2, 3.8, 4.4]), "power")
    assert fair.sum() == pytest.approx(1.0, abs=1e-12)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown margin method"):
        remove_margin(ODDS, "shin")


def test_edge() -> None:
    assert edge(0.55, 2.0) == pytest.approx(0.10)
    assert edge(0.4, 2.0) == pytest.approx(-0.2)


def test_find_value_picks_the_biggest_edge_in_each_market() -> None:
    probabilities = pd.DataFrame(
        {
            "home": [0.50, 0.30],
            "draw": [0.30, 0.30],
            "away": [0.20, 0.40],
            "over25": [0.50, 0.50],
            "under25": [0.50, 0.50],
        },
        index=[10, 11],
    )
    odds = pd.DataFrame(
        {
            "home": [2.2, 3.0],
            "draw": [3.6, 3.2],
            "away": [4.0, 2.7],
            "over25": [1.80, 1.95],
            "under25": [2.10, 1.95],
        },
        index=[10, 11],
    )
    bets = find_value(probabilities, odds, threshold=0.03)

    assert list(bets.index) == [10, 11, 10]
    assert list(bets["market"]) == ["1x2", "1x2", "totals"]
    assert list(bets["outcome"]) == ["home", "away", "under25"]
    assert bets.iloc[0]["edge"] == pytest.approx(0.10)
    assert bets.iloc[1]["edge"] == pytest.approx(0.08)
    assert bets.iloc[2]["odds"] == 2.10
    assert bets.iloc[2]["edge"] == pytest.approx(0.05)


def test_edge_exactly_at_the_threshold_counts() -> None:
    probabilities = pd.DataFrame({"home": [0.515], "draw": [0.25], "away": [0.235]})
    odds = pd.DataFrame({"home": [2.0], "draw": [3.0], "away": [3.0]})
    bets = find_value(probabilities, odds, threshold=0.03)
    assert list(bets["outcome"]) == ["home"]


def test_no_value_and_missing_odds_give_no_bets() -> None:
    probabilities = pd.DataFrame({"home": [0.45, 0.6], "draw": [0.3, 0.2], "away": [0.25, 0.2]})
    odds = pd.DataFrame({"home": [2.0, np.nan], "draw": [3.2, np.nan], "away": [3.8, np.nan]})
    bets = find_value(probabilities, odds, threshold=0.03)
    assert bets.empty
    assert list(bets.columns) == ["market", "outcome", "probability", "odds", "edge"]


def test_check_quotes() -> None:
    probabilities = {"home": 0.55, "draw": 0.25, "away": 0.20, "over25": 0.5, "under25": 0.5}
    quoted = {"home": 2.0, "draw": 3.5, "away": 4.0}
    checks, margins = check_quotes(probabilities, quoted, "proportional", 0.03)

    assert [check.outcome for check in checks] == ["home", "draw", "away"]
    assert margins == {"1x2": pytest.approx(0.5 + 1 / 3.5 + 0.25 - 1)}
    home = checks[0]
    assert home.edge == pytest.approx(0.10)
    assert home.book_probability == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    assert home.value
    assert not any(check.value for check in checks[1:])


def test_check_quotes_skips_partly_quoted_markets() -> None:
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2, "over25": 0.6, "under25": 0.4}
    checks, margins = check_quotes(probabilities, {"over25": 1.9}, "power", 0.03)
    assert checks == []
    assert margins == {}
