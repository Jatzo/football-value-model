import numpy as np
import pandas as pd
import pytest
from simulation import simulate_league, true_model

from valuemodel.markets import OUTCOMES
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.models.poisson import fit_poisson
from valuemodel.walkforward import (
    odds_capture_date,
    walk_forward_forecasts,
)


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    model = true_model(10, -0.1, rng)
    return pd.concat(
        [
            simulate_league(model, 2, rng, start="2022-08-01", season="2223"),
            simulate_league(model, 2, rng, start="2023-08-01", season="2324"),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize(
    ("match_day", "capture_day"),
    [
        ("2024-03-01", "2024-03-01"),
        ("2024-03-02", "2024-03-01"),
        ("2024-03-03", "2024-03-01"),
        ("2024-03-04", "2024-03-01"),
        ("2024-03-05", "2024-03-05"),
        ("2024-03-06", "2024-03-05"),
        ("2024-03-07", "2024-03-05"),
    ],
    ids=["friday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday"],
)
def test_odds_capture_date(match_day: str, capture_day: str) -> None:
    captured = odds_capture_date(pd.Series([pd.Timestamp(match_day)]))
    assert captured.iloc[0] == pd.Timestamp(capture_day)


def test_forecasts_keep_the_match_index_and_cover_both_markets(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2324"], fit_poisson, 0.003, min_matches=0)
    target = league[league["season"] == "2324"]
    assert list(forecasts.index) == list(target.index)
    assert (forecasts["home_team"] == target["home_team"]).all()
    np.testing.assert_allclose(forecasts[["home", "draw", "away"]].sum(axis=1), 1.0)
    np.testing.assert_allclose(forecasts[["over25", "under25"]].sum(axis=1), 1.0)
    assert (forecasts["as_of"] <= forecasts["date"]).all()
    assert (forecasts["date"] - forecasts["as_of"] < pd.Timedelta(days=4)).all()


def test_unreliable_matches_have_no_prices(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2324"], fit_poisson, 0.003, min_matches=10_000)
    assert not forecasts["reliable"].any()
    assert forecasts[list(OUTCOMES)].isna().all().all()


def test_future_results_cannot_leak_into_forecasts(league: pd.DataFrame) -> None:
    """Rewrite every result from a cutoff onwards and check earlier forecasts do not move.

    A forecast is made on its as_of date, so it may only use matches played
    before that date. If any fit could see a match on or after its as_of date,
    the rewritten scores would change some forecast made at or before the cutoff.
    """
    target = league[league["season"] == "2324"]
    cutoff = target["date"].iloc[len(target) // 2]
    tampered = league.copy()
    future = tampered["date"] >= cutoff
    tampered.loc[future, "home_goals"] = 7
    tampered.loc[future, "away_goals"] = 0
    tampered.loc[future, "result"] = "H"

    honest = walk_forward_forecasts(league, ["2324"], fit_dixon_coles, 0.003)
    leaked = walk_forward_forecasts(tampered, ["2324"], fit_dixon_coles, 0.003)

    made_by_cutoff = honest["as_of"] <= cutoff
    assert made_by_cutoff.sum() > 50
    pd.testing.assert_frame_equal(
        honest.loc[made_by_cutoff, list(OUTCOMES)],
        leaked.loc[made_by_cutoff, list(OUTCOMES)],
    )
    later = honest.loc[~made_by_cutoff, "home"] - leaked.loc[~made_by_cutoff, "home"]
    assert later.abs().max() > 0.01


def test_forecasts_are_fitted_at_the_odds_capture_date(league: pd.DataFrame) -> None:
    forecasts = walk_forward_forecasts(league, ["2324"], fit_dixon_coles, 0.003)
    pd.testing.assert_series_equal(
        forecasts["as_of"], odds_capture_date(forecasts["date"]), check_names=False
    )


@pytest.fixture(scope="module")
def divisions() -> pd.DataFrame:
    """Two divisions over two seasons, with one team promoted and one relegated between them."""
    rng = np.random.default_rng(13)
    model = true_model(12, -0.1, rng)
    top, bottom = list(model.teams[:6]), list(model.teams[6:])
    seasons = [
        ("2022-08-01", "2223", top, bottom),
        ("2023-08-01", "2324", [*top[:-1], bottom[0]], [top[-1], *bottom[1:]]),
    ]
    return pd.concat(
        [
            simulate_league(model, 4, rng, start=start, league=league, season=season, teams=teams)
            for start, season, upper, lower in seasons
            for league, teams in (("E0", upper), ("E1", lower))
        ],
        ignore_index=True,
    )


def test_linked_forecasts_cover_only_the_target_league(divisions: pd.DataFrame) -> None:
    linked = {"E0": ["E0", "E1"]}
    forecasts = walk_forward_forecasts(
        divisions, ["2324"], fit_poisson, 0.003, history_leagues=linked
    )
    target = divisions[(divisions["league"] == "E0") & (divisions["season"] == "2324")]
    assert list(forecasts.index) == list(target.index)


def test_promoted_teams_are_priced_from_their_lower_division_results(
    divisions: pd.DataFrame,
) -> None:
    promoted = "Team 06"
    e0 = divisions[divisions["league"] == "E0"]
    alone = walk_forward_forecasts(e0, ["2324"], fit_poisson, 0.003)
    linked = walk_forward_forecasts(
        divisions, ["2324"], fit_poisson, 0.003, history_leagues={"E0": ["E0", "E1"]}
    )
    plays = (alone["home_team"] == promoted) | (alone["away_team"] == promoted)
    assert not alone.loc[plays, "reliable"].iloc[:5].any()
    assert linked.loc[plays, "reliable"].all()


def test_linked_results_cannot_leak_into_forecasts(divisions: pd.DataFrame) -> None:
    """Rewrite the lower division's results from a cutoff and check earlier forecasts hold."""
    linked = {"E0": ["E0", "E1"]}
    target = divisions[(divisions["league"] == "E0") & (divisions["season"] == "2324")]
    cutoff = target["date"].iloc[len(target) // 2]
    tampered = divisions.copy()
    future = (tampered["league"] == "E1") & (tampered["date"] >= cutoff)
    tampered.loc[future, ["home_goals", "away_goals", "result"]] = [8, 0, "H"]

    honest = walk_forward_forecasts(divisions, ["2324"], fit_poisson, 0.003, history_leagues=linked)
    leaked = walk_forward_forecasts(tampered, ["2324"], fit_poisson, 0.003, history_leagues=linked)
    made_by_cutoff = honest["as_of"] <= cutoff
    assert made_by_cutoff.sum() > 20
    pd.testing.assert_frame_equal(
        honest.loc[made_by_cutoff, list(OUTCOMES)], leaked.loc[made_by_cutoff, list(OUTCOMES)]
    )
    later = honest.loc[~made_by_cutoff, "home"] - leaked.loc[~made_by_cutoff, "home"]
    assert later.abs().max() > 0.01


def test_saturday_results_do_not_reach_sunday_forecasts(league: pd.DataFrame) -> None:
    """Odds for a Sunday match are taken on Friday, so Saturday's games must not count."""
    target = league[league["season"] == "2324"]
    sunday = target.loc[target["date"].dt.dayofweek == 6, "date"].iloc[2]
    saturday = sunday - pd.Timedelta(days=1)
    assert (league["date"] == saturday).any()

    tampered = league.copy()
    on_saturday = tampered["date"] == saturday
    tampered.loc[on_saturday, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]

    honest = walk_forward_forecasts(league, ["2324"], fit_dixon_coles, 0.003)
    leaked = walk_forward_forecasts(tampered, ["2324"], fit_dixon_coles, 0.003)
    on_sunday = honest["date"] == sunday
    pd.testing.assert_frame_equal(
        honest.loc[on_sunday, list(OUTCOMES)], leaked.loc[on_sunday, list(OUTCOMES)]
    )
    next_week = honest["date"] > sunday + pd.Timedelta(days=4)
    assert (honest.loc[next_week, "home"] - leaked.loc[next_week, "home"]).abs().max() > 0.001
