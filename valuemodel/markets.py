"""Turn a scoreline probability matrix into market probabilities."""

from dataclasses import dataclass

import numpy as np

from valuemodel.models.common import FittedModel


@dataclass(frozen=True)
class MarketProbabilities:
    home: float
    draw: float
    away: float
    over25: float
    under25: float

    def fair_odds(self) -> dict[str, float]:
        """Decimal odds with no margin, the price at which a bet breaks even."""
        return {name: 1 / value for name, value in self.__dict__.items()}


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
