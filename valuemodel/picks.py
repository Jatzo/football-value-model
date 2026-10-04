"""Paper bets on upcoming fixtures, ranked by edge, and the odds each outcome needs.

A pick follows the backtest's rules exactly: the edge against the bookmaker's
odds must reach the threshold, only one outcome per market per match, and no
games involving a team with too little history. The stake is a paper stake
from the configured staking method on the starting bankroll.
"""

import pandas as pd

from valuemodel.config import Settings
from valuemodel.odds import MARKETS
from valuemodel.staking import stake

PICK_COLUMNS = [
    "league",
    "date",
    "kickoff",
    "home_team",
    "away_team",
    "market",
    "outcome",
    "probability",
    "fair_odds",
    "odds",
    "edge",
    "stake",
]


def price_to_beat(probability: float, edge_threshold: float) -> float:
    """The lowest decimal odds at which an outcome reaches the edge threshold.

    A bet has value when probability * odds - 1 is at least the threshold, so
    the odds needed are (1 + threshold) / probability.
    """
    if not 0 < probability <= 1:
        raise ValueError(f"probability must be in (0, 1], got {probability}")
    return (1 + edge_threshold) / probability


def value_bets(priced: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Every value bet among priced fixtures, the biggest edge first.

    `priced` is the output of fixtures.price_fixtures, which has already chosen
    at most one value outcome per market in `value_1x2` and `value_totals`.
    """
    rows = []
    for fixture in priced.to_dict("records"):
        for market in MARKETS:
            outcome = fixture.get(f"value_{market}")
            if not isinstance(outcome, str):
                continue
            probability, odds = fixture[outcome], fixture[f"odds_{outcome}"]
            rows.append(
                {
                    **{name: fixture[name] for name in PICK_COLUMNS[:5]},
                    "market": market,
                    "outcome": outcome,
                    "probability": probability,
                    "fair_odds": 1 / probability,
                    "odds": odds,
                    "edge": fixture[f"edge_{outcome}"],
                    "stake": stake(probability, odds, settings.starting_bankroll, settings),
                }
            )
    picks = pd.DataFrame(rows, columns=PICK_COLUMNS)
    return picks.sort_values(["edge", "date"], ascending=[False, True], ignore_index=True)
