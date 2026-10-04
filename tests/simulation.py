"""Simulated leagues with known parameters, for checking that fits recover them."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from valuemodel.markets import predict
from valuemodel.models.common import FittedModel


def true_model(n_teams: int, rho: float, rng: np.random.Generator) -> FittedModel:
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defence = rng.normal(np.log(1.2), 0.25, n_teams)
    return FittedModel(
        teams=tuple(f"Team {i:02d}" for i in range(n_teams)),
        attack=attack,
        defence=defence,
        home_advantage=0.25,
        rho=rho,
        as_of=pd.Timestamp("2000-01-01"),
        match_counts={},
    )


def simulate_league(
    model: FittedModel,
    rounds: int,
    rng: np.random.Generator,
    start: str = "2020-08-01",
    league: str = "E0",
    season: str = "2021",
    teams: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Play every pairing home and away `rounds` times, sampling each score from the model.

    `teams` limits the league to some of the model's teams, so several divisions
    can be simulated from one set of true strengths.
    """
    teams = list(teams or model.teams)
    rows = []
    date = pd.Timestamp(start)
    for _ in range(rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                matrix = model.score_matrix(home, away)
                cell = rng.choice(matrix.size, p=matrix.ravel())
                home_goals, away_goals = divmod(cell, matrix.shape[1])
                rows.append((date, home, away, home_goals, away_goals))
            date += pd.Timedelta(days=1)
    frame = pd.DataFrame(
        rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    frame["league"] = league
    frame["season"] = season
    frame["home_goals"] = frame["home_goals"].astype("Int64")
    frame["away_goals"] = frame["away_goals"].astype("Int64")
    difference = frame["home_goals"] - frame["away_goals"]
    frame["result"] = np.select([difference > 0, difference == 0], ["H", "D"], "A")
    return frame


def add_odds(frame: pd.DataFrame, model: FittedModel, rng: np.random.Generator) -> pd.DataFrame:
    """Give each match Bet365, Pinnacle and Pinnacle closing odds around its true prices.

    Each source adds a margin and some noise, with the closing line the sharpest,
    which is roughly how real prices behave.
    """
    sources = {"b365": (0.05, 0.05), "pinnacle": (0.03, 0.02), "pinnacle_close": (0.025, 0.01)}
    true = pd.DataFrame(
        [
            predict(model, home, away).__dict__
            for home, away in zip(frame["home_team"], frame["away_team"], strict=True)
        ],
        index=frame.index,
    )
    frame = frame.copy()
    frame["kickoff"] = pd.NA
    for source, (margin, noise) in sources.items():
        for outcome in true.columns:
            wobble = rng.lognormal(0, noise, len(frame))
            frame[f"{source}_{outcome}"] = np.maximum(1.01, wobble / (true[outcome] * (1 + margin)))
    return frame


TWO_SEASONS: tuple[tuple[str, str], ...] = (("2223", "2022-08-01"), ("2324", "2023-08-01"))


def simulated_seasons(
    seed: int,
    n_teams: int,
    rounds: int,
    seasons: Sequence[tuple[str, str]] = TWO_SEASONS,
    with_odds: bool = True,
) -> pd.DataFrame:
    """Consecutive simulated seasons, given as (code, start date), from one true model."""
    rng = np.random.default_rng(seed)
    model = true_model(n_teams, -0.1, rng)
    league = pd.concat(
        [simulate_league(model, rounds, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    return add_odds(league, model, rng) if with_odds else league
