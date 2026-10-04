"""Pieces shared by the Poisson and Dixon-Coles models.

Both models predict log goal rates the same way:

    log home rate = home advantage + attack[home] + defence[away]
    log away rate = attack[away] + defence[home]

A higher defence value means a team concedes more. The attack values are
constrained to sum to zero so the model is identifiable. Without that, adding a
constant to every attack and subtracting it from every defence would give the
same predictions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from valuemodel.config import TRAINING_WINDOW_DAYS

MAX_GOALS = 10


class UnknownTeamError(ValueError):
    """Raised when a team has no matches in the model's training window."""


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles adjustment to the independent Poisson probability of a score.

    Only 0-0, 1-0, 0-1 and 1-1 are changed. Every other score gets a factor of 1.
    """
    home_goals, away_goals = np.asarray(home_goals), np.asarray(away_goals)
    home_rate, away_rate = np.asarray(home_rate), np.asarray(away_rate)
    factor = np.ones(np.broadcast(home_goals, away_goals, home_rate, away_rate).shape)
    factor = np.where(
        (home_goals == 0) & (away_goals == 0), 1 - home_rate * away_rate * rho, factor
    )
    factor = np.where((home_goals == 0) & (away_goals == 1), 1 + home_rate * rho, factor)
    factor = np.where((home_goals == 1) & (away_goals == 0), 1 + away_rate * rho, factor)
    return np.where((home_goals == 1) & (away_goals == 1), 1 - rho, factor)


def time_weights(dates: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    """Weight each match by exp(-xi * days before as_of)."""
    days_ago = (as_of - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * days_ago)


def training_window(
    matches: pd.DataFrame, as_of: pd.Timestamp, window_days: int = TRAINING_WINDOW_DAYS
) -> pd.DataFrame:
    """Matches played strictly before as_of and within the window.

    This is the only place a fit sees results, so it is where lookahead is
    prevented: nothing on or after the prediction date can get through.
    """
    start = as_of - pd.Timedelta(days=window_days)
    return matches[(matches["date"] < as_of) & (matches["date"] >= start)]


@dataclass(frozen=True)
class TrainingData:
    teams: tuple[str, ...]
    home: np.ndarray
    away: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    weights: np.ndarray
    match_counts: dict[str, int]

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def prepare(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> TrainingData:
    """Turn the matches before as_of into arrays ready for fitting."""
    if matches["league"].nunique() > 1:
        raise ValueError("Fit one league at a time, since each has its own home advantage")
    window = training_window(matches, as_of, window_days)
    if window.empty:
        raise ValueError(f"No matches to fit before {as_of.date()}")

    teams = tuple(sorted(set(window["home_team"]) | set(window["away_team"])))
    position = {team: i for i, team in enumerate(teams)}
    counts = pd.concat([window["home_team"], window["away_team"]]).value_counts()
    return TrainingData(
        teams=teams,
        home=window["home_team"].map(position).to_numpy(),
        away=window["away_team"].map(position).to_numpy(),
        home_goals=window["home_goals"].to_numpy(dtype=float),
        away_goals=window["away_goals"].to_numpy(dtype=float),
        weights=time_weights(window["date"], as_of, xi),
        match_counts={team: int(counts[team]) for team in teams},
    )


def unpack(params: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a parameter vector into attack, defence and home advantage.

    Only n - 1 attack values are free. The last is minus the sum of the others,
    which enforces the sum-to-zero constraint exactly.
    """
    free_attack = params[: n_teams - 1]
    attack = np.append(free_attack, -free_attack.sum())
    defence = params[n_teams - 1 : 2 * n_teams - 1]
    home_advantage = params[2 * n_teams - 1]
    return attack, defence, home_advantage


def log_rates(params: np.ndarray, data: TrainingData) -> tuple[np.ndarray, np.ndarray]:
    attack, defence, home_advantage = unpack(params, data.n_teams)
    log_home = home_advantage + attack[data.home] + defence[data.away]
    log_away = attack[data.away] + defence[data.home]
    return log_home, log_away


def rate_gradient(
    grad_log_home: np.ndarray, grad_log_away: np.ndarray, data: TrainingData
) -> np.ndarray:
    """Chain per-match gradients of the log rates back to the team parameters."""
    n = data.n_teams
    grad_attack = np.bincount(data.home, grad_log_home, n) + np.bincount(
        data.away, grad_log_away, n
    )
    grad_defence = np.bincount(data.away, grad_log_home, n) + np.bincount(
        data.home, grad_log_away, n
    )
    grad_free_attack = grad_attack[:-1] - grad_attack[-1]
    return np.concatenate([grad_free_attack, grad_defence, [grad_log_home.sum()]])


def starting_params(data: TrainingData) -> np.ndarray:
    """Equal teams, with average goal rates and home advantage from the data."""
    mean_home = np.average(data.home_goals, weights=data.weights)
    mean_away = np.average(data.away_goals, weights=data.weights)
    return np.concatenate(
        [
            np.zeros(data.n_teams - 1),
            np.full(data.n_teams, np.log(mean_away)),
            [np.log(mean_home / mean_away)],
        ]
    )


def minimise(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    start: np.ndarray,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> np.ndarray:
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Model fit did not converge: {result.message}")
    return result.x


@dataclass(frozen=True)
class FittedModel:
    """Team strengths from one fit, able to price any match between known teams."""

    teams: tuple[str, ...]
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    rho: float
    as_of: pd.Timestamp
    match_counts: dict[str, int]

    def _position(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError:
            raise UnknownTeamError(
                f"{team} has no matches in the training window before {self.as_of.date()}"
            ) from None

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        h, a = self._position(home), self._position(away)
        home_rate = np.exp(self.home_advantage + self.attack[h] + self.defence[a])
        away_rate = np.exp(self.attack[a] + self.defence[h])
        return float(home_rate), float(away_rate)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Probability of each score, with home goals on rows and away goals on columns.

        The Dixon-Coles adjustment moves probability between the four low scores
        without changing the total, so the only reason to renormalise is the tiny
        amount of probability beyond MAX_GOALS that the matrix leaves out.
        """
        home_rate, away_rate = self.expected_goals(home, away)
        goals = np.arange(MAX_GOALS + 1)
        matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
        matrix *= tau(goals[:, None], goals[None, :], home_rate, away_rate, self.rho)
        return matrix / matrix.sum()

    def is_reliable(self, team: str, min_matches: int) -> bool:
        return self.match_counts.get(team, 0) >= min_matches
