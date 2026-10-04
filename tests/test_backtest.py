from typing import Any

import numpy as np
import pandas as pd
import pytest
from simulation import add_odds, simulate_league, true_model

from valuemodel.backtest import (
    BET_COLUMNS,
    betting_summary,
    calibration_table,
    closing_line_value,
    longest_losing_run,
    market_forecasts,
    max_drawdown,
    model_scores,
    place_bets,
    run_backtest,
    settle,
)
from valuemodel.config import Settings
from valuemodel.staking import kelly_stake

OUTCOMES = ["home", "draw", "away", "over25", "under25"]


def match_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Matches with every odds column, filled from each row's `odds` entry."""
    records = []
    for row in rows:
        record = {
            "league": "E0",
            "season": "2324",
            "kickoff": pd.NA,
            "date": pd.Timestamp(row["date"]),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_goals": row["score"][0],
            "away_goals": row["score"][1],
            "result": "HDA"[1 - int(np.sign(row["score"][0] - row["score"][1]))],
        }
        for source in ("b365", "pinnacle", "pinnacle_close"):
            for outcome, price in zip(OUTCOMES, row.get(source, row["odds"]), strict=True):
                record[f"{source}_{outcome}"] = price
        records.append(record)
    return pd.DataFrame(records)


def forecast_table(
    probabilities: list[list[float]], reliable: list[bool] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=OUTCOMES)
    frame["reliable"] = reliable if reliable is not None else True
    return frame


@pytest.mark.parametrize(
    ("outcome", "score", "won"),
    [
        ("home", (2, 1), True),
        ("home", (1, 1), False),
        ("draw", (1, 1), True),
        ("away", (0, 3), True),
        ("over25", (2, 1), True),
        ("over25", (1, 1), False),
        ("under25", (2, 0), True),
        ("under25", (3, 0), False),
    ],
)
def test_settle(outcome: str, score: tuple[int, int], won: bool) -> None:
    result = settle(pd.Series([outcome]), pd.Series([score[0]]), pd.Series([score[1]]))
    assert bool(result.iloc[0]) is won


EVEN_ODDS = [2.0, 3.4, 4.0, 1.9, 1.9]


@pytest.fixture
def two_days() -> pd.DataFrame:
    return match_table(
        [
            {
                "date": "2023-08-12",
                "home_team": "A",
                "away_team": "B",
                "score": (2, 0),
                "odds": EVEN_ODDS,
            },
            {
                "date": "2023-08-12",
                "home_team": "C",
                "away_team": "D",
                "score": (0, 1),
                "odds": EVEN_ODDS,
            },
            {
                "date": "2023-08-19",
                "home_team": "B",
                "away_team": "C",
                "score": (1, 0),
                "odds": EVEN_ODDS,
            },
        ]
    )


VALUE_ON_HOME = [0.56, 0.24, 0.20, 0.5, 0.5]


def test_stakes_on_one_day_share_the_opening_bankroll(two_days: pd.DataFrame) -> None:
    forecasts = forecast_table([VALUE_ON_HOME] * 3)
    bets = place_bets(forecasts, two_days, Settings())

    opening = kelly_stake(0.56, 2.0, 1000)
    assert list(bets["home_team"]) == ["A", "C", "B"]
    assert bets["stake"].iloc[0] == pytest.approx(opening)
    assert bets["stake"].iloc[1] == pytest.approx(opening)
    day_one_bankroll = 1000 + opening - opening
    assert bets["bankroll"].iloc[1] == pytest.approx(day_one_bankroll)
    assert bets["stake"].iloc[2] == pytest.approx(kelly_stake(0.56, 2.0, day_one_bankroll))
    assert bets["bankroll"].iloc[-1] == pytest.approx(1000 + bets["profit"].sum())
    assert list(bets.columns) == list(BET_COLUMNS)


def test_profit_follows_the_result(two_days: pd.DataFrame) -> None:
    bets = place_bets(forecast_table([VALUE_ON_HOME] * 3), two_days, Settings())
    assert list(bets["won"]) == [True, False, True]
    assert bets["profit"].iloc[0] == pytest.approx(bets["stake"].iloc[0] * 1.0)
    assert bets["profit"].iloc[1] == pytest.approx(-bets["stake"].iloc[1])


def test_bets_use_the_configured_bookmakers_prices(two_days: pd.DataFrame) -> None:
    two_days["pinnacle_home"] = 2.3
    bets = place_bets(forecast_table([VALUE_ON_HOME] * 3), two_days, Settings(bookmaker="b365"))
    assert (bets["odds"] == 2.0).all()
    bets = place_bets(forecast_table([VALUE_ON_HOME] * 3), two_days, Settings(bookmaker="pinnacle"))
    assert (bets["odds"] == 2.3).all()


def test_no_bets_on_unreliable_matches_or_missing_odds(two_days: pd.DataFrame) -> None:
    two_days.loc[2, "b365_home"] = np.nan
    forecasts = forecast_table([VALUE_ON_HOME] * 3, reliable=[False, True, True])
    bets = place_bets(forecasts, two_days, Settings())
    assert list(bets["match_id"]) == [1]


def test_one_bet_per_market_per_match(two_days: pd.DataFrame) -> None:
    forecasts = forecast_table([[0.56, 0.30, 0.14, 0.6, 0.4]] * 3)
    bets = place_bets(forecasts, two_days, Settings())
    assert len(bets) == 6
    assert not bets.duplicated(["match_id", "market"]).any()
    assert bets.index.is_unique


def test_max_drawdown() -> None:
    units, share = max_drawdown(np.array([1000, 1100, 880, 950, 1200, 1100]))
    assert units == pytest.approx(220)
    assert share == pytest.approx(0.2)
    assert max_drawdown(np.array([1000, 1010, 1020])) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("won", "expected"),
    [([True, False, False, True, False, False, False, True], 3), ([True, True], 0), ([], 0)],
)
def test_longest_losing_run(won: list[bool], expected: int) -> None:
    assert longest_losing_run(won) == expected


def test_betting_summary_adds_up(two_days: pd.DataFrame) -> None:
    bets = place_bets(forecast_table([VALUE_ON_HOME] * 3), two_days, Settings())
    summary = betting_summary(bets, 1000)
    assert summary["bets"] == 3
    assert summary["final_bankroll"] == pytest.approx(1000 + summary["profit"])
    assert summary["roi"] == pytest.approx(summary["profit"] / summary["staked"])
    assert summary["level_roi"] == pytest.approx((1.0 - 1.0 + 1.0) / 3)
    assert summary["level_roi_low"] <= summary["level_roi"] <= summary["level_roi_high"]
    assert summary["longest_losing_run"] == 1


def test_betting_summary_without_bets() -> None:
    assert betting_summary(pd.DataFrame(columns=list(BET_COLUMNS)), 1000)["bets"] == 0


def test_closing_line_value_matches_hand_calculation() -> None:
    matches = match_table(
        [
            {
                "date": "2023-08-12",
                "home_team": "A",
                "away_team": "B",
                "score": (1, 0),
                "odds": [2.2, 3.4, 4.0, 1.9, 1.9],
                "pinnacle_close": [2.0, 3.5, 4.0, 1.9, 2.0],
            },
            {
                "date": "2023-08-12",
                "home_team": "C",
                "away_team": "D",
                "score": (1, 0),
                "odds": [2.2, 3.4, 4.0, 1.9, 1.9],
                "pinnacle_close": [np.nan] * 5,
            },
        ]
    )
    bets = pd.DataFrame({"match_id": [0, 1], "outcome": ["home", "home"], "odds": [2.2, 2.2]})
    clv = closing_line_value(bets, matches, "proportional")
    fair_home = 0.5 / (0.5 + 1 / 3.5 + 0.25)
    assert clv["fair_closing_odds"].iloc[0] == pytest.approx(1 / fair_home)
    assert clv["clv"].iloc[0] == pytest.approx(2.2 * fair_home - 1)
    assert clv["clv"].iloc[0] == pytest.approx(0.062069, abs=1e-6)
    assert np.isnan(clv["clv"].iloc[1])


def test_market_forecasts_remove_the_margin(two_days: pd.DataFrame) -> None:
    two_days.loc[1, "pinnacle_draw"] = np.nan
    forecasts = market_forecasts(two_days, "pinnacle", "power")
    np.testing.assert_allclose(forecasts.loc[[0, 2], ["home", "draw", "away"]].sum(axis=1), 1.0)
    assert list(forecasts["reliable"]) == [True, False, True]


def test_model_scores_use_only_matches_every_forecaster_priced(two_days: pd.DataFrame) -> None:
    complete = forecast_table([VALUE_ON_HOME] * 3)
    patchy = forecast_table([VALUE_ON_HOME] * 3)
    patchy.loc[1, ["home", "draw", "away"]] = np.nan
    scores = model_scores({"complete": complete, "patchy": patchy}, two_days)
    assert list(scores["matches"]) == [2, 2]
    assert scores["log_loss"].iloc[0] == pytest.approx(scores["log_loss"].iloc[1])


def test_calibration_table_pools_all_three_outcomes(two_days: pd.DataFrame) -> None:
    forecasts = forecast_table([[0.65, 0.27, 0.08, 0.5, 0.5], [0.15, 0.23, 0.62, 0.5, 0.5]])
    table = calibration_table(forecasts, two_days.loc[[0, 1]])
    assert list(table["count"]) == [1, 1, 2, 2]
    assert table["count"].sum() == 6
    top = table.iloc[-1]
    assert (top["bin_low"], top["mean_forecast"], top["observed"]) == pytest.approx(
        (0.6, 0.635, 1.0)
    )
    assert table.iloc[2]["observed"] == 0.0


@pytest.fixture(scope="module")
def simulated_league() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    model = true_model(10, -0.1, rng)
    seasons = [("2223", "2022-08-01"), ("2324", "2023-08-01")]
    league = pd.concat(
        [simulate_league(model, 2, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    return add_odds(league, model, rng)


def test_run_backtest_end_to_end(simulated_league: pd.DataFrame) -> None:
    result = run_backtest(simulated_league, "E0", ["2324"], Settings(), min_matches=0)
    assert list(result.summary["strategy"]) == ["dixon-coles", "poisson", "market"]
    assert list(result.scores["forecaster"]) == [
        "dixon-coles",
        "poisson",
        "bet365 pre-match",
        "pinnacle closing",
    ]
    assert set(result.calibration) == {"dixon-coles", "poisson"}
    for name, bets in result.bets.items():
        if len(bets):
            row = result.summary.set_index("strategy").loc[name]
            assert row["final_bankroll"] == pytest.approx(1000 + bets["profit"].sum())
            assert set(bets["season"]) == {"2324"}


def test_closing_odds_never_change_the_bets(simulated_league: pd.DataFrame) -> None:
    moved = simulated_league.copy()
    closing = [column for column in moved if column.startswith("pinnacle_close_")]
    moved[closing] = moved[closing] * 1.3

    original = run_backtest(simulated_league, "E0", ["2324"], Settings(), min_matches=0)
    shifted = run_backtest(moved, "E0", ["2324"], Settings(), min_matches=0)

    betting = ["match_id", "outcome", "odds", "stake", "profit"]
    for name in original.bets:
        pd.testing.assert_frame_equal(original.bets[name][betting], shifted.bets[name][betting])
    dixon_coles = original.bets["dixon-coles"]
    assert len(dixon_coles) > 0
    assert not np.allclose(dixon_coles["clv"], shifted.bets["dixon-coles"]["clv"])


def test_pinnacle_as_bookmaker_is_scored_and_skips_the_market_strategy(
    simulated_league: pd.DataFrame,
) -> None:
    settings = Settings(bookmaker="pinnacle")
    result = run_backtest(simulated_league, "E0", ["2324"], settings, min_matches=0)
    assert list(result.summary["strategy"]) == ["dixon-coles", "poisson"]
    assert "pinnacle pre-match" in list(result.scores["forecaster"])


def test_flat_stakes_end_to_end(simulated_league: pd.DataFrame) -> None:
    settings = Settings(staking="flat", flat_stake=0.01, starting_bankroll=1000)
    bets = run_backtest(simulated_league, "E0", ["2324"], settings, min_matches=0).bets
    stakes = bets["dixon-coles"]["stake"]
    assert len(stakes) > 0
    assert (stakes <= 10.0 + 1e-9).all()
    assert (stakes == 10.0).mean() > 0.5
