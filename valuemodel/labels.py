"""Display names shared by the command line report and the dashboard."""

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
