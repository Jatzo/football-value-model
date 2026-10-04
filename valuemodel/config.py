"""Settings, leagues and seasons."""

import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Leagues fitted together, so a team keeps its ratings when it moves division.
# The teams that move each season put the divisions on one scale. They share one
# home advantage: from 2019/20 to 2025/26 it was 0.19 (log of home over away
# goals) in E0, 0.21 in E1 and 0.22 in E2, while it ranged from 0.06 to 0.29
# between seasons of the same league.
LINKED_LEAGUES: dict[str, tuple[str, ...]] = {
    "E0": ("E0", "E1"),
    "E1": ("E0", "E1", "E2"),
}

# Leagues downloaded only to rate the teams moving into a linked league. They
# have never been tuned or backtested, so their own matches are not priced.
HISTORY_ONLY_LEAGUES: tuple[str, ...] = ("E2",)

DEFAULT_LEAGUES: tuple[str, ...] = ("E0",)
# The leagues with downloaded history, so the fixtures page can price them.
FIXTURE_LEAGUES: tuple[str, ...] = ("E0", "E1")

# Seasons are split so that nothing used to choose settings is later reported
# as a backtest result. The first two only ever serve as training history.
HISTORY_SEASONS: tuple[str, ...] = ("1920", "2021")
TUNING_SEASONS: tuple[str, ...] = ("2122", "2223")
BACKTEST_SEASONS: tuple[str, ...] = ("2324", "2425", "2526")
DEFAULT_SEASONS: tuple[str, ...] = HISTORY_SEASONS + TUNING_SEASONS + BACKTEST_SEASONS

# Time decay per day for the models fitted to goals. Chosen with `valuemodel
# tune-xi` on walk-forward forecasts for E0 2021/22 and 2022/23, refitted on the
# backtest's schedule, where 0.003 gave the lowest ranked probability score and
# log loss for Dixon-Coles and Poisson. The README records the full tables.
DEFAULT_XI = 0.003

# The shots-adjusted model's time decay and the share of each score it takes
# from shot-based expected goals, chosen together on the same tuning seasons.
# Shots are less noisy than goals, so recent matches can safely count for more.
SHOTS_XI = 0.006
SHOT_WEIGHT = 0.6

MODEL_XI: dict[str, float] = {
    "shots-adjusted": SHOTS_XI,
    "dixon-coles": DEFAULT_XI,
    "poisson": DEFAULT_XI,
}

# Matches older than this are left out of a fit. Time decay already gives them
# little weight, and dropping them keeps each refit fast.
TRAINING_WINDOW_DAYS = 1095

# Teams with fewer matches than this in the training window get unreliable
# estimates, so their games are flagged and not bet on.
MIN_TEAM_MATCHES = 10

MARGIN_METHODS: tuple[str, ...] = ("power", "proportional")
STAKING_METHODS: tuple[str, ...] = ("kelly", "flat")

# Bookmakers whose pre-match prices a paper bet can be taken at. Market maximum
# odds are left out because nobody can reliably get the best price everywhere.
BOOKMAKERS: dict[str, str] = {"b365": "Bet365", "pinnacle": "Pinnacle"}

USER_AGENT = "football-value-model (+https://github.com/Jatzo/football-value-model)"

_SEASON_PATTERN = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    request_delay: float = 2.0
    margin_method: str = "power"
    edge_threshold: float = 0.03
    staking: str = "kelly"
    kelly_fraction: float = 0.25
    max_stake: float = 0.02
    flat_stake_share: float = 0.01
    starting_bankroll: float = 1000.0
    bookmaker: str = "b365"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def _number(name: str, default: float, low: float, high: float) -> float:
    """Read a number from the environment and check it lies in [low, high]."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _choice(name: str, default: str, options: Iterable[str]) -> str:
    value = os.environ.get(name, default)
    options = tuple(options)
    if value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}, got {value!r}")
    return value


def load_settings() -> Settings:
    """Read settings from the environment, falling back to a local .env file."""
    load_dotenv()
    return Settings(
        data_dir=Path(os.environ.get("VALUEMODEL_DATA_DIR", "data")),
        request_delay=_number("VALUEMODEL_REQUEST_DELAY", 2.0, 0.0, 60.0),
        margin_method=_choice("VALUEMODEL_MARGIN_METHOD", "power", MARGIN_METHODS),
        edge_threshold=_number("VALUEMODEL_EDGE_THRESHOLD", 0.03, 0.0, 1.0),
        staking=_choice("VALUEMODEL_STAKING", "kelly", STAKING_METHODS),
        kelly_fraction=_number("VALUEMODEL_KELLY_FRACTION", 0.25, 0.0, 1.0),
        max_stake=_number("VALUEMODEL_MAX_STAKE", 0.02, 0.0, 1.0),
        flat_stake_share=_number("VALUEMODEL_FLAT_STAKE", 0.01, 0.0, 1.0),
        starting_bankroll=_number("VALUEMODEL_STARTING_BANKROLL", 1000.0, 1.0, 1e12),
        bookmaker=_choice("VALUEMODEL_BOOKMAKER", "b365", BOOKMAKERS),
    )


def season_code(start_year: int) -> str:
    """Return the data source's code for a season, for example 2024 gives "2425"."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> str:
    """Code of the season in progress. A new season is taken to start on 1 July."""
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return season_code(start_year)


def linked_leagues(league: str, linked: bool = True) -> tuple[str, ...]:
    """The leagues whose results a model for this league is fitted on."""
    return LINKED_LEAGUES.get(league, (league,)) if linked else (league,)


def can_fit_together(leagues: Iterable[str]) -> bool:
    leagues = set(leagues)
    return len(leagues) <= 1 or any(leagues <= set(group) for group in LINKED_LEAGUES.values())


def season_start_year(code: str) -> int:
    """The calendar year a season code starts in, for example "2425" gives 2024."""
    short = int(code[:2])
    return short + (1900 if short >= 50 else 2000)


def season_label(code: str) -> str:
    """Readable form of a season code, for example "2425" gives "2024/25"."""
    return f"{season_start_year(code)}/{code[2:]}"


def history_seasons(seasons: Iterable[str]) -> list[str]:
    """The given seasons plus enough earlier ones to fill the training window."""
    first = min(season_start_year(code) for code in seasons)
    years_back = math.ceil(TRAINING_WINDOW_DAYS / 365)
    earlier = [season_code(year) for year in range(first - years_back, first)]
    return list(dict.fromkeys([*earlier, *sorted(seasons, key=season_start_year)]))


def validate_season(code: str) -> str:
    """Check a season code has two consecutive two-digit years and return it."""
    if not _SEASON_PATTERN.match(code):
        raise ValueError(f"Season must be four digits such as 2425, got {code!r}")
    first, second = int(code[:2]), int(code[2:])
    if (first + 1) % 100 != second:
        raise ValueError(f"Season {code!r} does not cover two consecutive years")
    return code


def validate_league(code: str) -> str:
    """Check a league code is one the project knows about and return it."""
    if code not in LEAGUES:
        known = ", ".join(LEAGUES)
        raise ValueError(f"Unknown league {code!r}. Known leagues: {known}")
    return code
