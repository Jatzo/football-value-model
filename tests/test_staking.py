import pytest

from valuemodel.config import Settings
from valuemodel.staking import flat_stake, kelly_fraction, kelly_stake, stake


def test_kelly_fraction_matches_hand_calculation() -> None:
    assert kelly_fraction(0.4, 2.7) == pytest.approx(0.08 / 1.7)
    assert kelly_fraction(0.5, 2.2) == pytest.approx(0.1 / 1.2)


@pytest.mark.parametrize(("probability", "odds"), [(0.5, 2.0), (0.4, 2.0), (0.1, 5.0)])
def test_kelly_is_zero_without_a_positive_edge(probability: float, odds: float) -> None:
    assert kelly_fraction(probability, odds) == 0.0
    assert kelly_stake(probability, odds, bankroll=1000) == 0.0


def test_quarter_kelly_below_the_cap() -> None:
    assert kelly_stake(0.4, 2.7, bankroll=1000) == pytest.approx(1000 * 0.25 * 0.08 / 1.7)
    assert kelly_stake(0.4, 2.7, bankroll=1000) == pytest.approx(11.76, abs=0.01)


def test_quarter_kelly_is_capped() -> None:
    uncapped = 1000 * 0.25 * 0.1 / 1.2
    assert uncapped == pytest.approx(20.83, abs=0.01)
    assert kelly_stake(0.5, 2.2, bankroll=1000) == pytest.approx(20.0)


def test_kelly_stake_scales_with_bankroll() -> None:
    assert kelly_stake(0.4, 2.7, bankroll=500) == pytest.approx(kelly_stake(0.4, 2.7, 1000) / 2)


@pytest.mark.parametrize("probability", [0.3, 0.5, 0.7, 0.9, 0.99])
def test_kelly_never_exceeds_the_cap(probability: float) -> None:
    assert kelly_stake(probability, 3.0, bankroll=1000, fraction=1.0) <= 20.0


@pytest.mark.parametrize("bankroll", [0.0, -50.0])
def test_no_stake_without_a_bankroll(bankroll: float) -> None:
    assert kelly_stake(0.6, 2.0, bankroll) == 0.0
    assert flat_stake(bankroll, unit=10) == 0.0


def test_flat_stake_is_constant_until_the_cap_bites() -> None:
    assert flat_stake(1000, unit=10) == 10
    assert flat_stake(2000, unit=10) == 10
    assert flat_stake(400, unit=10) == pytest.approx(8.0)


def test_stake_uses_the_configured_method() -> None:
    kelly = Settings(staking="kelly")
    flat = Settings(staking="flat", flat_stake=0.01, starting_bankroll=1000)
    assert stake(0.4, 2.7, 1000, kelly) == pytest.approx(11.76, abs=0.01)
    assert stake(0.4, 2.7, 1000, flat) == pytest.approx(10.0)


@pytest.mark.parametrize("staking", ["kelly", "flat"])
def test_stake_is_zero_without_an_edge(staking: str) -> None:
    assert stake(0.4, 2.5, 1000, Settings(staking=staking)) == 0.0
