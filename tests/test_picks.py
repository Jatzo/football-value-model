import itertools
import math

import numpy as np
import pandas as pd
import pytest
from simulation import add_odds, simulate_league, true_model

from valuemodel.config import Settings
from valuemodel.fixtures import price_fixtures
from valuemodel.picks import (
    BET_TYPE_LABELS,
    BET_TYPES,
    PICK_COLUMNS,
    Leg,
    Selection,
    accumulator,
    best_slips,
    likely_slips,
    price_to_beat,
    top_combinations,
    value_bets,
)
from valuemodel.staking import stake


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


ARSENAL = Leg("2026-10-10 Arsenal v Leeds", "home", 1.9, 0.582)
UNITED = Leg("2026-10-10 Man United v Tottenham", "home", 2.0, 0.6)
LIVERPOOL = Leg("2026-10-11 Liverpool v Man City", "away", 2.6, 0.43)


def test_accumulator_multiplies_odds_and_chances() -> None:
    acca = accumulator([ARSENAL, UNITED, LIVERPOOL])
    assert acca.odds == pytest.approx(1.9 * 2.0 * 2.6)
    assert acca.odds == pytest.approx(9.88)
    assert acca.probability == pytest.approx(0.582 * 0.6 * 0.43)
    assert acca.fair_odds == pytest.approx(1 / (0.582 * 0.6 * 0.43))
    assert acca.edge == pytest.approx(0.582 * 0.6 * 0.43 * 9.88 - 1)
    assert acca.returns(10) == pytest.approx(98.8)


def test_a_single_leg_is_a_single_bet() -> None:
    single = accumulator([ARSENAL])
    assert (single.odds, single.probability) == (1.9, 0.582)
    assert single.edge == pytest.approx(0.582 * 1.9 - 1)


def test_two_legs_from_the_same_match_are_refused() -> None:
    over = Leg(ARSENAL.match, "over25", 2.0, 0.376)
    with pytest.raises(ValueError, match="Arsenal v Leeds is already in the accumulator"):
        accumulator([ARSENAL, over])


@pytest.mark.parametrize(
    ("legs", "message"),
    [
        ([], "at least one leg"),
        ([Leg("a", "home", 1.0, 0.5)], "greater than 1"),
        ([Leg("a", "home", 2.0, 0.0)], "probability"),
    ],
)
def test_invalid_accumulators_are_refused(legs: list[Leg], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        accumulator(legs)


def pick_table(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    """Value picks as value_bets returns them, from (home, away, outcome, odds, chance)."""
    return pd.DataFrame(
        [
            {
                "league": "E0",
                "date": pd.Timestamp("2026-10-10"),
                "kickoff": "15:00",
                "home_team": home,
                "away_team": away,
                "market": "totals" if outcome.startswith(("over", "under")) else "1x2",
                "outcome": outcome,
                "probability": chance,
                "fair_odds": 1 / chance,
                "odds": odds,
                "edge": chance * odds - 1,
                "stake": 10.0,
            }
            for home, away, outcome, odds, chance in rows
        ],
        columns=PICK_COLUMNS,
    )


PICKS = pick_table(
    [
        ("Arsenal", "Leeds", "home", 2.0, 0.6),
        ("Arsenal", "Leeds", "over25", 2.2, 0.6),
        ("Fulham", "Hull", "home", 2.1, 0.5),
        ("Everton", "Wolves", "away", 3.1, 0.33),
    ]
)


def test_best_slips_give_three_options_of_the_chosen_size() -> None:
    slips = best_slips(PICKS, Settings(), legs=2)
    assert [slip.name for slip in slips] == [
        "Double, option 1",
        "Double, option 2",
        "Double, option 3",
    ]
    legs = [[(leg.match[11:], leg.outcome) for leg in slip.accumulator.legs] for slip in slips]
    assert legs[0] == [("Arsenal v Leeds", "over25"), ("Fulham v Hull", "home")]
    edges = [slip.accumulator.edge for slip in slips]
    assert edges == sorted(edges, reverse=True)
    assert len({tuple(option) for option in legs}) == 3


def test_best_slips_never_repeat_a_match() -> None:
    for size in (1, 2, 3):
        for slip in best_slips(PICKS, Settings(), legs=size):
            matches = [leg.match for leg in slip.accumulator.legs]
            assert len(matches) == len(set(matches)) == size


def test_best_slips_need_enough_matches() -> None:
    assert best_slips(PICKS, Settings(), legs=4) == []
    assert best_slips(PICKS.iloc[0:0], Settings(), legs=1) == []
    treble = best_slips(PICKS, Settings(), legs=3)
    assert len(treble) == 2  # three matches, two ways to pick Arsenal v Leeds
    assert treble[0].accumulator.odds == pytest.approx(2.2 * 2.1 * 3.1)


def test_suggested_stakes_treat_the_slip_as_one_bet() -> None:
    for slip in best_slips(PICKS, Settings(), legs=2):
        acca = slip.accumulator
        assert slip.stake == pytest.approx(
            stake(acca.probability, acca.odds, Settings().starting_bankroll, Settings())
        )
        assert 0 < slip.stake <= 20


SELECTIONS = [
    Selection("2026-10-10 Arsenal v Leeds", "home", 0.58),
    Selection("2026-10-10 Arsenal v Leeds", "under25", 0.62),
    Selection("2026-10-10 Man United v Tottenham", "home", 0.60),
    Selection("2026-10-11 Liverpool v Man City", "away", 0.43),
    Selection("2026-10-09 West Ham v QPR", "home", 0.59),
]


def test_likely_slips_give_three_options_likeliest_first() -> None:
    slips = likely_slips(SELECTIONS, legs=3)
    assert [slip.name for slip in slips] == [
        "Treble, option 1",
        "Treble, option 2",
        "Treble, option 3",
    ]
    first, second, third = ([(s.match[11:], s.outcome) for s in slip.selections] for slip in slips)
    assert first == [
        ("Arsenal v Leeds", "under25"),
        ("Man United v Tottenham", "home"),
        ("West Ham v QPR", "home"),
    ]
    assert second == [
        ("Arsenal v Leeds", "home"),
        ("Man United v Tottenham", "home"),
        ("West Ham v QPR", "home"),
    ]
    assert third == [
        ("Arsenal v Leeds", "under25"),
        ("Man United v Tottenham", "home"),
        ("Liverpool v Man City", "away"),
    ]
    assert slips[0].probability == pytest.approx(0.62 * 0.60 * 0.59)
    assert slips[0].probability * slips[0].price_to_beat(0.03) - 1 == pytest.approx(0.03)


def test_likely_slips_with_few_matches() -> None:
    assert likely_slips(SELECTIONS[:2], legs=2) == []
    singles = likely_slips(SELECTIONS[:2], legs=1)
    assert [slip.selections[0].outcome for slip in singles] == ["under25", "home"]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_top_combinations_match_trying_every_combination(seed: int, size: int) -> None:
    rng = np.random.default_rng(seed)
    candidates = [
        Selection(f"match {rng.integers(9)}", f"outcome {index}", float(rng.uniform(0.1, 0.9)))
        for index in range(25)
    ]
    exhaustive = sorted(
        (
            combination
            for combination in itertools.combinations(candidates, size)
            if len({c.match for c in combination}) == size
        ),
        key=lambda combination: math.prod(c.probability for c in combination),
        reverse=True,
    )
    fast = top_combinations(candidates, size, lambda c: c.probability)
    scores = [math.prod(c.probability for c in combination) for combination in fast]
    expected = [math.prod(c.probability for c in combination) for combination in exhaustive[:3]]
    assert scores == pytest.approx(expected)


def test_likely_slips_keep_to_one_bet_type() -> None:
    selections = [*SELECTIONS, Selection("2026-10-10 Arsenal v Leeds", "btts_no", 0.66)]
    results = likely_slips(selections, legs=2, bet_type="result")
    assert all(s.outcome in BET_TYPES["result"] for slip in results for s in slip.selections)
    assert [s.outcome for s in results[0].selections] == ["home", "home"]
    both = likely_slips(selections, legs=1, bet_type="btts")
    assert [slip.selections[0].outcome for slip in both] == ["btts_no"]
    anything = likely_slips(selections, legs=1)
    assert anything[0].selections[0].outcome == "btts_no"


def test_best_slips_keep_to_one_bet_type() -> None:
    goals = best_slips(PICKS, Settings(), legs=1, bet_type="goals")
    assert [slip.accumulator.legs[0].outcome for slip in goals] == ["over25"]
    assert best_slips(PICKS, Settings(), legs=1, bet_type="btts") == []


def test_every_bet_type_has_a_label() -> None:
    assert set(BET_TYPE_LABELS) == set(BET_TYPES)
    assert set(BET_TYPES["any"]) == set(
        BET_TYPES["result"] + BET_TYPES["goals"] + BET_TYPES["btts"]
    )
