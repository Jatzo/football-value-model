"""Download, cache, clean and standardise football-data.co.uk results and odds.

Column names change between seasons. The functions here map every era onto one
schema so the rest of the project never sees the raw names.
"""

import io
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx
import pandas as pd

from valuemodel.config import USER_AGENT, Settings
from valuemodel.teams import normalise_team

logger = logging.getLogger(__name__)

# The www host answers with a redirect, so go straight to the bare domain.
BASE_URL = "https://football-data.co.uk/mmz4281"


class DownloadError(RuntimeError):
    """Raised when the data source does not return a usable CSV file."""


OUTCOMES_1X2: dict[str, str] = {"H": "home", "D": "draw", "A": "away"}
OUTCOMES_TOTALS: dict[str, str] = {">2.5": "over25", "<2.5": "under25"}

# Each source lists its raw prefixes for 1X2 and for over/under 2.5, newest
# naming first. BetBrain (Bb) supplied the max and average odds until 2018/19.
# The notes file gives PH as an older name for Pinnacle's PSH.
ODDS_SOURCES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "b365": (("B365",), ("B365",)),
    "pinnacle": (("PS", "P"), ("P",)),
    "max": (("Max", "BbMx"), ("Max", "BbMx")),
    "avg": (("Avg", "BbAv"), ("Avg", "BbAv")),
}

# Columns every file has, then the results columns that only played matches
# have, with the alternative names the notes file lists.
MATCH_SOURCES: dict[str, tuple[str, ...]] = {
    "date": ("Date",),
    "home_team": ("HomeTeam",),
    "away_team": ("AwayTeam",),
}
RESULT_SOURCES: dict[str, tuple[str, ...]] = {
    "home_goals": ("FTHG", "HG"),
    "away_goals": ("FTAG", "AG"),
    "result": ("FTR", "Res"),
}


def _odds_candidates() -> dict[str, tuple[str, ...]]:
    """Map each standard odds column to the raw names it may appear under."""
    candidates: dict[str, tuple[str, ...]] = {}
    for source, (prefixes_1x2, prefixes_totals) in ODDS_SOURCES.items():
        for closing in (False, True):
            marker = "C" if closing else ""
            stem = f"{source}_close" if closing else source
            for raw, outcome in OUTCOMES_1X2.items():
                candidates[f"{stem}_{outcome}"] = tuple(f"{p}{marker}{raw}" for p in prefixes_1x2)
            for raw, outcome in OUTCOMES_TOTALS.items():
                candidates[f"{stem}_{outcome}"] = tuple(
                    f"{p}{marker}{raw}" for p in prefixes_totals
                )
    return candidates


ODDS_CANDIDATES = _odds_candidates()
ODDS_COLUMNS: tuple[str, ...] = tuple(ODDS_CANDIDATES)
MATCH_COLUMNS: tuple[str, ...] = (
    "league",
    "season",
    "date",
    "kickoff",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)


def decode(raw: bytes) -> str:
    """Decode a downloaded file, allowing for a UTF-8 byte order mark.

    Recent files are UTF-8 with a BOM and older ones are plain ASCII. Latin-1 is
    the fallback because it accepts any byte, so a stray accented name in an old
    file cannot stop a season loading.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_raw(path: Path) -> pd.DataFrame:
    """Read a raw CSV as strings, dropping the blank rows and columns some files carry."""
    frame = pd.read_csv(io.StringIO(decode(path.read_bytes())), dtype=str)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    return frame.dropna(how="all").reset_index(drop=True)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse day-first dates written with either a two or four digit year."""
    text = values.str.strip()
    short = pd.to_datetime(text.where(text.str.len() == 8), format="%d/%m/%y")
    long = pd.to_datetime(text.where(text.str.len() == 10), format="%d/%m/%Y")
    parsed = long.fillna(short)
    unparsed = text[parsed.isna() & text.notna()]
    if not unparsed.empty:
        raise ValueError(f"Unrecognised date format: {unparsed.iloc[0]!r}")
    return parsed


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def _odds(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Convert raw odds to floats. Anything that is not a valid decimal price becomes NaN."""
    if values is None:
        return pd.Series(float("nan"), index=index)
    odds = pd.to_numeric(values, errors="coerce")
    return odds.where(odds > 1.0)


def standardise(
    raw: pd.DataFrame, league: str, season: str, has_results: bool = True
) -> pd.DataFrame:
    """Map a raw file onto the project's standard columns.

    Season files have results. The upcoming fixtures file does not, so with
    has_results False the goals and result columns are left empty.
    """
    required = {**MATCH_SOURCES, **(RESULT_SOURCES if has_results else {})}
    core = {name: _first_present(raw, names) for name, names in required.items()}
    missing = [name for name, values in core.items() if values is None]
    if missing:
        raise ValueError(f"{league} {season} is missing required columns: {missing}")
    empty = pd.Series(pd.NA, index=raw.index, dtype="string")

    frame = pd.DataFrame(index=raw.index)
    frame["league"] = league
    frame["season"] = season
    frame["date"] = parse_dates(core["date"])
    frame["kickoff"] = raw["Time"].str.strip() if "Time" in raw else pd.NA
    for side in ("home_team", "away_team"):
        frame[side] = core[side].map(normalise_team, na_action="ignore")
    for side in ("home_goals", "away_goals"):
        frame[side] = pd.to_numeric(core.get(side, empty), errors="coerce").astype("Int64")
    frame["result"] = core.get("result", empty).str.strip()
    for column, names in ODDS_CANDIDATES.items():
        frame[column] = _odds(_first_present(raw, names), raw.index)
    return frame


def _score_result(frame: pd.DataFrame) -> pd.Series:
    """The H, D or A that the goals imply, or NA when a score is missing."""
    difference = (frame["home_goals"] - frame["away_goals"]).astype("Float64")
    sign = difference.map(lambda d: (d > 0) - (d < 0), na_action="ignore")
    return sign.map({1: "H", 0: "D", -1: "A"})


def find_problems(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that cannot be trusted, with the reasons in a problem column."""
    core = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    valid_result = frame["result"].isin(["H", "D", "A"])
    implied = _score_result(frame)
    checks = pd.DataFrame(
        {
            "missing results data": frame[core].isna().any(axis=1),
            "result is not H, D or A": frame["result"].notna() & ~valid_result,
            "result does not match the score": valid_result
            & implied.notna()
            & (frame["result"] != implied),
            "team plays itself": frame["home_team"] == frame["away_team"],
            "fixture appears twice": frame.duplicated(
                ["league", "season", "home_team", "away_team"], keep=False
            ),
        }
    )
    checks = checks.fillna(False).astype(bool)
    failed = checks.any(axis=1)
    problems = frame[failed].copy()
    problems["problem"] = ["; ".join(checks.columns[row]) for row in checks[failed].to_numpy()]
    return problems


def load_season(path: Path, league: str, season: str) -> pd.DataFrame:
    """Load one cached season file as clean, date-ordered matches.

    Rows that fail validation are logged and left out rather than guessed at,
    because a wrong result would quietly distort every model fitted on it.
    """
    frame = standardise(read_raw(path), league, season)
    problems = find_problems(frame)
    for _, row in problems.iterrows():
        logger.warning(
            "%s %s: dropping %s v %s on %s (%s)",
            league,
            season,
            row["home_team"],
            row["away_team"],
            row["date"],
            row["problem"],
        )
    clean = frame.drop(index=problems.index)
    return clean.sort_values(["date", "kickoff", "home_team"]).reset_index(drop=True)


def season_url(league: str, season: str) -> str:
    return f"{BASE_URL}/{season}/{league}.csv"


def cache_path(raw_dir: Path, league: str, season: str) -> Path:
    return raw_dir / f"{league}_{season}.csv"


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)


def download_season(
    client: httpx.Client, league: str, season: str, raw_dir: Path, force: bool = False
) -> bool:
    """Save one season to the cache. Returns False when a cached copy was used instead."""
    path = cache_path(raw_dir, league, season)
    if path.exists() and not force:
        return False
    download_csv(client, season_url(league, season), path)
    return True


def download_csv(client: httpx.Client, url: str, path: Path) -> None:
    """Fetch a CSV file and save it, replacing any earlier copy only once it is complete."""
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise DownloadError(f"Could not download {url}: {error}") from error

    # A missing season can come back as an empty body or an HTML page rather
    # than an error status, and caching either would break every later load.
    content = response.content
    if not content.strip() or content.lstrip().startswith(b"<"):
        raise DownloadError(f"{url} did not return a CSV file")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(content)
    partial.replace(path)


def download_seasons(
    client: httpx.Client,
    leagues: Iterable[str],
    seasons: Iterable[str],
    settings: Settings,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> list[tuple[str, str, bool]]:
    """Download several seasons, pausing between requests to be polite to the source."""
    results = []
    fetched_any = False
    for league in leagues:
        for season in seasons:
            if fetched_any and (force or not cache_path(settings.raw_dir, league, season).exists()):
                sleep(settings.request_delay)
            downloaded = download_season(client, league, season, settings.raw_dir, force)
            fetched_any = fetched_any or downloaded
            results.append((league, season, downloaded))
    return results


def load_matches(
    leagues: Iterable[str], seasons: Iterable[str], settings: Settings
) -> pd.DataFrame:
    """Load cached seasons into one date-ordered frame. Nothing is downloaded here."""
    leagues, seasons = list(leagues), list(seasons)
    for league in leagues:
        for season in seasons:
            if not cache_path(settings.raw_dir, league, season).exists():
                raise FileNotFoundError(
                    f"No cached data for {league} {season}. Run: valuemodel download"
                )
    return load_available(leagues, seasons, settings)


def load_available(
    leagues: Iterable[str], seasons: Iterable[str], settings: Settings
) -> pd.DataFrame:
    """Load whichever of these seasons are cached, skipping the rest.

    Used where the season in progress may or may not have been fetched yet.
    The result is empty when nothing is cached. A season listed twice is loaded
    once, since duplicate matches would double their weight in every fit.
    """
    seasons = list(dict.fromkeys(seasons))
    frames = [
        load_season(path, league, season)
        for league in leagues
        for season in seasons
        if (path := cache_path(settings.raw_dir, league, season)).exists()
    ]
    if not frames:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, *ODDS_COLUMNS])
    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values(["date", "kickoff", "league", "home_team"]).reset_index(drop=True)
