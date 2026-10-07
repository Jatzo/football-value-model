"""Turn a scoreline probability matrix into market probabilities."""

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from valuemodel.models.common import FittedModel


@dataclass(frozen=True)
class MarketProbabilities:
    home: float
    draw: float
    away: float
    over25: float
    under25: float


OUTCOMES: tuple[str, ...] = tuple(field.name for field in fields(MarketProbabilities))


def market_probabilities(matrix: np.ndarray) -> MarketProbabilities:
    """Sum a matrix with home goals on rows and away goals on columns into markets."""
    goals = np.arange(matrix.shape[0])
    total_goals = goals[:, None] + goals[None, :]
    over25 = float(matrix[total_goals >= 3].sum())
    return MarketProbabilities(
        home=float(np.tril(matrix, -1).sum()),
        draw=float(np.trace(matrix)),
        away=float(np.triu(matrix, 1).sum()),
        over25=over25,
        under25=float(matrix.sum()) - over25,
    )


def predict(model: FittedModel, home: str, away: str) -> MarketProbabilities:
    return market_probabilities(model.score_matrix(home, away))


def price_matches(model: FittedModel, matches: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Probabilities for every outcome of each match, indexed like `matches`.

    A match involving a team with fewer than min_matches results in the fit is
    marked unreliable and left unpriced rather than given an extreme price.
    """
    rows = []
    for home, away in zip(matches["home_team"], matches["away_team"], strict=True):
        reliable = model.is_reliable(home, min_matches) and model.is_reliable(away, min_matches)
        prices = asdict(predict(model, home, away)) if reliable else {}
        rows.append({"reliable": reliable, **{o: prices.get(o, np.nan) for o in OUTCOMES}})
    return pd.DataFrame(rows, index=matches.index, columns=["reliable", *OUTCOMES])


# Priced from the same scoreline matrix, but kept apart from OUTCOMES because the
# data source has no odds for it, so it can never be backtested or value-checked.
BOTH_TEAMS_TO_SCORE: tuple[str, ...] = ("btts_yes", "btts_no")


def both_teams_to_score(matrix: np.ndarray) -> float:
    """Chance that both sides score at least once."""
    return float(matrix[1:, 1:].sum())


def price_both_teams_to_score(
    model: FittedModel, matches: pd.DataFrame, min_matches: int
) -> pd.DataFrame:
    """Both teams to score, yes and no, for each match, indexed like `matches`.

    Matches the model cannot price reliably are left empty, as in price_matches.
    """
    rows = []
    for home, away in zip(matches["home_team"], matches["away_team"], strict=True):
        reliable = model.is_reliable(home, min_matches) and model.is_reliable(away, min_matches)
        yes = both_teams_to_score(model.score_matrix(home, away)) if reliable else np.nan
        rows.append({"btts_yes": yes, "btts_no": 1 - yes})
    return pd.DataFrame(rows, index=matches.index, columns=list(BOTH_TEAMS_TO_SCORE))
