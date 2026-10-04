"""Settings, leagues and seasons."""

import os
import re
from dataclasses import dataclass
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

DEFAULT_SEASONS: tuple[str, ...] = ("2122", "2223", "2324", "2425", "2526")

USER_AGENT = "football-value-model (+https://github.com/Jatzo/footballbetfinder)"

_SEASON_PATTERN = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    request_delay: float

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def load_settings() -> Settings:
    """Read settings from the environment, falling back to a local .env file."""
    load_dotenv()
    return Settings(
        data_dir=Path(os.environ.get("VALUEMODEL_DATA_DIR", "data")),
        request_delay=float(os.environ.get("VALUEMODEL_REQUEST_DELAY", "2")),
    )


def season_code(start_year: int) -> str:
    """Return the data source's code for a season, for example 2024 gives "2425"."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


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
