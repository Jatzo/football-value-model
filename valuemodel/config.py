"""Settings, leagues and seasons."""

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
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

DEFAULT_LEAGUES: tuple[str, ...] = ("E0",)

# Seasons are split so that nothing used to choose settings is later reported
# as a backtest result. The first two only ever serve as training history.
HISTORY_SEASONS: tuple[str, ...] = ("1920", "2021")
TUNING_SEASONS: tuple[str, ...] = ("2122", "2223")
BACKTEST_SEASONS: tuple[str, ...] = ("2324", "2425", "2526")
DEFAULT_SEASONS: tuple[str, ...] = HISTORY_SEASONS + TUNING_SEASONS + BACKTEST_SEASONS

# Time decay per day. Chosen with `valuemodel tune-xi` on walk-forward forecasts
# for E0 2021/22 and 2022/23, where 0.003 gave the lowest ranked probability
# score and log loss for both models. The README records the full table.
DEFAULT_XI = 0.003

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

USER_AGENT = "football-value-model (+https://github.com/Jatzo/footballbetfinder)"

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
    flat_stake: float = 0.01
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
        flat_stake=_number("VALUEMODEL_FLAT_STAKE", 0.01, 0.0, 1.0),
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


def season_label(code: str) -> str:
    """Readable form of a season code, for example "2425" gives "2024/25"."""
    century = 1900 if int(code[:2]) >= 50 else 2000
    return f"{century + int(code[:2])}/{code[2:]}"


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
