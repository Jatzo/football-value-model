"""Display names and number formats shared by the command line report and the dashboard."""

import math

OUTCOME_LABELS: dict[str, str] = {
    "home": "Home win",
    "draw": "Draw",
    "away": "Away win",
    "over25": "Over 2.5",
    "under25": "Under 2.5",
}

MARKET_LABELS: dict[str, str] = {"1x2": "Match result", "totals": "Over/under 2.5"}

# Every bet the slips can hold: the backtested outcomes plus both teams to score.
BET_LABELS: dict[str, str] = {
    **OUTCOME_LABELS,
    "btts_yes": "Both teams score: yes",
    "btts_no": "Both teams score: no",
}

STRATEGY_LABELS: dict[str, str] = {
    "dixon-coles": "Dixon-Coles",
    "poisson": "Poisson",
    "shots-adjusted": "Shots-adjusted",
    "market": "Follow the market",
}

FORECASTER_LABELS: dict[str, str] = {
    "dixon-coles": "Dixon-Coles",
    "poisson": "Poisson",
    "shots-adjusted": "Shots-adjusted",
    "bet365 pre-match": "Bet365 pre-match, margin removed",
    "pinnacle pre-match": "Pinnacle pre-match, margin removed",
    "pinnacle closing": "Pinnacle closing, margin removed",
}


NO_COMMON_MATCHES = (
    "No match had a price from every forecaster, so the forecasts could not be compared. "
    "This happens when Pinnacle's closing odds are missing for the seasons tested."
)


def label(names: dict[str, str], key: str) -> str:
    """The display name for a key, or the key itself when it has none."""
    return names.get(key, key)


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def percent(value: float | None, signed: bool = False) -> str:
    if is_missing(value):
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def share(value: float) -> str:
    """A setting such as a threshold or cap as a percentage, without rounding it away.

    0.03 gives "3%" and 0.035 gives "3.5%", where a fixed format would show 4%.
    """
    return f"{round(value * 100, 6):g}%"


def units(value: float | None) -> str:
    return "n/a" if is_missing(value) else f"{value:,.2f}"


def decimal_odds(value: float | None) -> str:
    return "" if is_missing(value) else f"{value:.2f}"


def tone(value: float | None) -> str:
    """CSS class for a figure that is good or bad news, or blank when neutral."""
    if is_missing(value) or value == 0:
        return ""
    return "positive" if value > 0 else "negative"
