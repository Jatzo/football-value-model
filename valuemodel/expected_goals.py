"""Shot-based expected goals.

football-data.co.uk records shots and shots on target but not where or how each
shot was taken, so these are not true expected goals. Each kind of shot is worth
the average number of goals it led to in past matches, which smooths out the
luck in actual scores while keeping each team's attacking output.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass(frozen=True)
class ShotValues:
    """Goals expected from one shot on target and from one other shot."""

    on_target: float
    off_target: float


SHOT_FIELDS = ("home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target")


def has_shots(matches: pd.DataFrame) -> pd.Series:
    """Which matches have all four shot counts. False throughout if a column is absent."""
    if not set(SHOT_FIELDS) <= set(matches.columns):
        return pd.Series(False, index=matches.index)
    return matches[list(SHOT_FIELDS)].notna().all(axis=1)


def fit_shot_values(matches: pd.DataFrame) -> ShotValues | None:
    """Learn how many goals each kind of shot is worth, or None without shot data.

    The values are constrained to be non-negative. Left free, a fit can give
    shots off target a slightly negative value, and a team with shots but none
    on target would then be expected to score fewer than zero goals.
    """
    played = matches[has_shots(matches)]
    if played.empty:
        return None
    on_target = np.r_[played["home_shots_on_target"], played["away_shots_on_target"]]
    shots = np.r_[played["home_shots"], played["away_shots"]]
    goals = np.r_[played["home_goals"], played["away_goals"]]
    features = np.column_stack([on_target, shots - on_target]).astype(float)
    (value_on, value_off), _ = nnls(features, goals.astype(float))
    return ShotValues(on_target=float(value_on), off_target=float(value_off))


def expected_goals(matches: pd.DataFrame, values: ShotValues) -> tuple[pd.Series, pd.Series]:
    """Each side's shot-based expected goals. Missing where shot counts are missing."""

    def side(prefix: str) -> pd.Series:
        on_target = matches[f"{prefix}_shots_on_target"].astype(float)
        other = matches[f"{prefix}_shots"].astype(float) - on_target
        return values.on_target * on_target + values.off_target * other

    return side("home"), side("away")


def blend_goals(matches: pd.DataFrame, values: ShotValues | None, weight: float) -> pd.DataFrame:
    """Replace each score with a weighted mix of expected and actual goals.

    Matches without shot counts, or every match when no shot values could be
    learned, keep their actual goals.
    """
    blended = matches.copy()
    for side in ("home", "away"):
        blended[f"{side}_goals"] = matches[f"{side}_goals"].astype(float)
    if values is None or weight == 0:
        return blended
    home_xg, away_xg = expected_goals(matches, values)
    shot_rows = has_shots(matches)
    for side, xg in (("home", home_xg), ("away", away_xg)):
        actual = blended[f"{side}_goals"]
        blended[f"{side}_goals"] = actual.where(~shot_rows, weight * xg + (1 - weight) * actual)
    return blended
