"""Paper bets on upcoming fixtures, ranked by edge, the odds each outcome needs, and accumulators.

A pick follows the backtest's rules exactly: the edge against the bookmaker's
odds must reach the threshold, only one outcome per market per match, and no
games involving a team with too little history. The stake is a paper stake
from the configured staking method on the starting bankroll.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Leg:
    """One selection in an accumulator: an outcome of one match at the odds taken."""

    match: str
    outcome: str
    odds: float
    probability: float


@dataclass(frozen=True)
class Accumulator:
    legs: tuple[Leg, ...]
    odds: float
    probability: float

    @property
    def fair_odds(self) -> float:
        return 1 / self.probability

    @property
    def edge(self) -> float:
        return self.probability * self.odds - 1

    def returns(self, stake: float) -> float:
        """What a winning bet pays back, stake included."""
        return stake * self.odds


def accumulator(legs: Sequence[Leg]) -> Accumulator:
    """Combine legs into one bet that wins only if every leg wins.

    The odds multiply, and so do the model's chances, which is only fair when
    the legs are independent. The model prices each match on its own, so legs
    must come from different matches: two outcomes of the same match, such as a
    home win and over 2.5 goals, are linked and their chances cannot simply be
    multiplied.
    """
    if not legs:
        raise ValueError("An accumulator needs at least one leg")
    seen: set[str] = set()
    for leg in legs:
        if leg.match in seen:
            raise ValueError(f"{leg.match} is already in the accumulator")
        if not leg.odds > 1:
            raise ValueError(f"Decimal odds must be greater than 1, got {leg.odds}")
        if not 0 < leg.probability <= 1:
            raise ValueError(f"probability must be in (0, 1], got {leg.probability}")
        seen.add(leg.match)
    return Accumulator(
        legs=tuple(legs),
        odds=math.prod(leg.odds for leg in legs),
        probability=math.prod(leg.probability for leg in legs),
    )


# Beyond three legs the chance that every leg wins falls fast while the
# model's errors compound, so suggested slips stop at trebles.
MAX_LEGS = 3
SLIP_NAMES: dict[int, str] = {1: "Best single", 2: "Best double", 3: "Best treble"}


def match_key(day: pd.Timestamp, home: str, away: str) -> str:
    """Identifies a match, so a slip never holds two legs from the same one."""
    return f"{day:%Y-%m-%d} {home} v {away}"


@dataclass(frozen=True)
class SuggestedSlip:
    name: str
    accumulator: Accumulator
    stake: float


def best_slips(
    picks: pd.DataFrame, settings: Settings, max_legs: int = MAX_LEGS
) -> list[SuggestedSlip]:
    """The best single, double and treble that can be built from the value picks.

    For each size, every combination of picks from different matches is tried
    and the one with the highest combined edge is kept, the likelier one on a
    tie. Every leg is already a value bet, so adding legs only raises the edge,
    which is why each size is suggested separately rather than ranked against
    the others. The paper stake treats the slip as one bet at its combined odds.
    """
    legs = [
        Leg(
            match=match_key(pick["date"], pick["home_team"], pick["away_team"]),
            outcome=pick["outcome"],
            odds=float(pick["odds"]),
            probability=float(pick["probability"]),
        )
        for pick in picks.to_dict("records")
    ]
    slips = []
    for size in range(1, max_legs + 1):
        candidates = [
            accumulator(combination)
            for combination in itertools.combinations(legs, size)
            if len({leg.match for leg in combination}) == size
        ]
        if not candidates:
            break
        best = max(candidates, key=lambda acca: (acca.edge, acca.probability))
        amount = stake(best.probability, best.odds, settings.starting_bankroll, settings)
        slips.append(SuggestedSlip(SLIP_NAMES.get(size, f"Best {size}-fold"), best, amount))
    return slips
