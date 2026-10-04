"""Implied probabilities, margin removal and value detection.

Odds arrays have one row per market and one column per outcome, for example
home, draw and away. A one-dimensional array is treated as a single market.
"""

import numpy as np
import pandas as pd

MARKETS: dict[str, tuple[str, ...]] = {
    "1x2": ("home", "draw", "away"),
    "totals": ("over25", "under25"),
}

# An edge that is meant to equal the threshold, such as 0.515 at 2.0 against 3%,
# can land a hair below it in floating point. It should still count.
_EDGE_TOLERANCE = 1e-9

_POWER_ITERATIONS = 100


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """1 / odds. Missing prices stay missing, and prices of 1 or less are rejected."""
    odds = np.asarray(odds, dtype=float)
    if np.any(odds[~np.isnan(odds)] <= 1.0):
        raise ValueError("Decimal odds must be greater than 1")
    return 1.0 / odds


def overround(odds: np.ndarray) -> np.ndarray:
    """The bookmaker's margin: how far the implied probabilities sum above 1."""
    return implied_probabilities(odds).sum(axis=-1) - 1.0


def _power_exponent(implied: np.ndarray) -> np.ndarray:
    """Find k for each row so that the implied probabilities raised to k sum to 1.

    The sum falls steadily as k rises, so bisection always finds the answer.
    Starting from [0, 100] covers any market with a margin below several
    hundred percent, far beyond anything a bookmaker offers.
    """
    low = np.zeros(implied.shape[0])
    high = np.full(implied.shape[0], 100.0)
    for _ in range(_POWER_ITERATIONS):
        middle = (low + high) / 2
        too_big = (implied ** middle[:, None]).sum(axis=1) > 1.0
        low = np.where(too_big, middle, low)
        high = np.where(too_big, high, middle)
    return (low + high) / 2


def remove_margin(odds: np.ndarray, method: str = "power") -> np.ndarray:
    """Turn bookmaker odds into probabilities that sum to 1.

    The proportional method scales every outcome down by the same factor. The
    power method raises each implied probability to the same exponent, which
    takes relatively more off longshots. That matches the well known tendency
    for bookmakers to load more of their margin onto unlikely outcomes.
    """
    odds = np.asarray(odds, dtype=float)
    implied = np.atleast_2d(implied_probabilities(odds))
    if method == "proportional":
        fair = implied / implied.sum(axis=1, keepdims=True)
    elif method == "power":
        fair = implied ** _power_exponent(implied)[:, None]
    else:
        raise ValueError(f"Unknown margin method {method!r}, expected power or proportional")
    complete = ~np.isnan(implied).any(axis=1)
    fair[~complete] = np.nan
    return fair.reshape(odds.shape)


def edge(probability: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Expected profit per unit staked if the probability is right."""
    return np.asarray(probability) * np.asarray(odds) - 1.0


def find_value(probabilities: pd.DataFrame, odds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pick at most one bet per market for each row, the outcome with the biggest edge.

    Both frames share an index and have a column per outcome. The result has one
    row per bet with the market, outcome, probability, odds and edge, indexed
    like the inputs so bets can be joined back to their matches.
    """
    bets = []
    for market, outcomes in MARKETS.items():
        columns = [outcome for outcome in outcomes if outcome in probabilities]
        if not columns:
            continue
        edges = pd.DataFrame(
            edge(probabilities[columns].to_numpy(), odds[columns].to_numpy()),
            index=probabilities.index,
            columns=columns,
        )
        priced = edges.notna().any(axis=1)
        best = edges[priced].fillna(-np.inf).idxmax(axis=1)
        best_edge = edges[priced].max(axis=1)
        chosen = best[best_edge >= threshold - _EDGE_TOLERANCE]
        bets.append(
            pd.DataFrame(
                {
                    "market": market,
                    "outcome": chosen,
                    "probability": [probabilities.at[i, o] for i, o in chosen.items()],
                    "odds": [odds.at[i, o] for i, o in chosen.items()],
                    "edge": [edges.at[i, o] for i, o in chosen.items()],
                },
                index=chosen.index,
            )
        )
    columns = ["market", "outcome", "probability", "odds", "edge"]
    return pd.concat(bets) if bets else pd.DataFrame(columns=columns)
