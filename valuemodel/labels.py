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

STRATEGY_LABELS: dict[str, str] = {
    "dixon-coles": "Dixon-Coles",
    "poisson": "Poisson",
    "market": "Follow the market",
}

FORECASTER_LABELS: dict[str, str] = {
    "dixon-coles": "Dixon-Coles",
    "poisson": "Poisson",
    "bet365 pre-match": "Bet365 pre-match, margin removed",
    "pinnacle closing": "Pinnacle closing, margin removed",
}


def label(names: dict[str, str], key: str) -> str:
    """The display name for a key, or the key itself when it has none."""
    return names.get(key, key)


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def percent(value: float | None, signed: bool = False) -> str:
    if is_missing(value):
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def units(value: float | None) -> str:
    return "n/a" if is_missing(value) else f"{value:,.2f}"


def decimal_odds(value: float | None) -> str:
    return "" if is_missing(value) else f"{value:.2f}"


def tone(value: float | None) -> str:
    """CSS class for a figure that is good or bad news, or blank when neutral."""
    if is_missing(value) or value == 0:
        return ""
    return "positive" if value > 0 else "negative"
