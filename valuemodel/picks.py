"""Paper bets on upcoming fixtures, ranked by edge, the odds each outcome needs, and accumulators.

A pick follows the backtest's rules exactly: the edge against the bookmaker's
odds must reach the threshold, only one outcome per market per match, and no
games involving a team with too little history. The stake is a paper stake
from the configured staking method on the starting bankroll.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

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


# Suggested slips run from singles to six-folds. Beyond that the chance that
# every leg wins is tiny while the model's errors keep compounding.
MAX_LEGS = 6
DEFAULT_LEGS = 3
SLIP_OPTIONS = 3
SIZE_NAMES: dict[int, str] = {
    1: "Single",
    2: "Double",
    3: "Treble",
    4: "Four-fold",
    5: "Five-fold",
    6: "Six-fold",
}


BET_TYPES: dict[str, tuple[str, ...]] = {
    "any": ("home", "draw", "away", "over25", "under25", "btts_yes", "btts_no"),
    "result": ("home", "draw", "away"),
    "goals": ("over25", "under25"),
    "btts": ("btts_yes", "btts_no"),
}
BET_TYPE_LABELS: dict[str, str] = {
    "any": "Any bet",
    "result": "Match result",
    "goals": "Over/under 2.5 goals",
    "btts": "Both teams to score",
}
DEFAULT_BET_TYPE = "any"


def match_key(day: pd.Timestamp, home: str, away: str) -> str:
    """Identifies a match, so a slip never holds two legs from the same one."""
    return f"{day:%Y-%m-%d} {home} v {away}"


def slip_name(size: int, option: int) -> str:
    return f"{SIZE_NAMES.get(size, f'{size}-fold')}, option {option}"


class _Candidate(Protocol):
    @property
    def match(self) -> str: ...


def top_combinations[T: _Candidate](
    candidates: Sequence[T], size: int, score: Callable[[T], float], options: int = SLIP_OPTIONS
) -> list[tuple[T, ...]]:
    """The highest scoring combinations of `size` candidates from different matches.

    A combination scores the product of its candidates' scores, so it only
    needs a small search. Only a match's best `options` candidates can appear in
    a top combination, since any other could be swapped for that many better
    ones. For the same reason only the best `size + options - 1` matches, ranked
    by their best candidate, can appear: a combination using any other match
    leaves at least `options` better matches unused.
    """
    by_match: dict[str, list[T]] = {}
    for candidate in candidates:
        by_match.setdefault(candidate.match, []).append(candidate)
    if len(by_match) < size:
        return []
    shortlists = [sorted(group, key=score, reverse=True)[:options] for group in by_match.values()]
    shortlists.sort(key=lambda group: score(group[0]), reverse=True)
    pool = shortlists[: size + options - 1]
    combinations = [
        legs
        for matches in itertools.combinations(pool, size)
        for legs in itertools.product(*matches)
    ]
    combinations.sort(key=lambda legs: math.prod(score(leg) for leg in legs), reverse=True)
    return combinations[:options]


@dataclass(frozen=True)
class SuggestedSlip:
    name: str
    accumulator: Accumulator
    stake: float


def best_slips(
    picks: pd.DataFrame,
    settings: Settings,
    legs: int = DEFAULT_LEGS,
    options: int = SLIP_OPTIONS,
    bet_type: str = DEFAULT_BET_TYPE,
) -> list[SuggestedSlip]:
    """The best few slips of a given size that can be built from the value picks.

    Combinations of picks from different matches are ranked by combined edge.
    Every leg is already a value bet, so adding legs only raises the edge,
    which is why each size is suggested on its own rather than ranked against
    the others. The paper stake treats the slip as one bet at its combined odds.
    """
    candidates = [
        Leg(
            match=match_key(pick["date"], pick["home_team"], pick["away_team"]),
            outcome=pick["outcome"],
            odds=float(pick["odds"]),
            probability=float(pick["probability"]),
        )
        for pick in picks.to_dict("records")
        if pick["outcome"] in BET_TYPES[bet_type]
    ]
    slips = []
    for option, combination in enumerate(
        top_combinations(candidates, legs, lambda leg: leg.odds * leg.probability, options), 1
    ):
        acca = accumulator(combination)
        amount = stake(acca.probability, acca.odds, settings.starting_bankroll, settings)
        slips.append(SuggestedSlip(slip_name(legs, option), acca, amount))
    return slips


@dataclass(frozen=True)
class Selection:
    """An outcome the model rates, before any bookmaker's odds are known."""

    match: str
    outcome: str
    probability: float


@dataclass(frozen=True)
class LikelySlip:
    name: str
    selections: tuple[Selection, ...]

    @property
    def probability(self) -> float:
        return math.prod(selection.probability for selection in self.selections)

    @property
    def fair_odds(self) -> float:
        return 1 / self.probability

    def price_to_beat(self, edge_threshold: float) -> float:
        """The lowest combined odds at which the slip would be a value bet."""
        return price_to_beat(self.probability, edge_threshold)


def likely_slips(
    selections: Sequence[Selection],
    legs: int = DEFAULT_LEGS,
    options: int = SLIP_OPTIONS,
    bet_type: str = DEFAULT_BET_TYPE,
) -> list[LikelySlip]:
    """The few slips of a given size the model thinks most likely to win, one leg per match.

    Without odds there is no edge, so these say nothing about value: they are
    the model's view of what will probably happen, with the price that would
    make each slip worth backing.
    """
    return [
        LikelySlip(slip_name(legs, option), combination)
        for option, combination in enumerate(
            top_combinations(
                [s for s in selections if s.outcome in BET_TYPES[bet_type]],
                legs,
                lambda selection: selection.probability,
                options,
            ),
            1,
        )
    ]
