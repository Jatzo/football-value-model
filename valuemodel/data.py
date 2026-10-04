"""Read, clean and standardise football-data.co.uk results and odds files.

Column names change between seasons. The functions here map every era onto one
schema so the rest of the project never sees the raw names.
"""

import io
from pathlib import Path

import pandas as pd

from valuemodel.teams import normalise_team

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

# Results columns, with the alternative names the notes file lists.
CORE_SOURCES: dict[str, tuple[str, ...]] = {
    "date": ("Date",),
    "home_team": ("HomeTeam",),
    "away_team": ("AwayTeam",),
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


def standardise(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Map a raw season file onto the project's standard columns."""
    core = {name: _first_present(raw, names) for name, names in CORE_SOURCES.items()}
    missing = [name for name, values in core.items() if values is None]
    if missing:
        raise ValueError(f"{league} {season} is missing required columns: {missing}")

    frame = pd.DataFrame(index=raw.index)
    frame["league"] = league
    frame["season"] = season
    frame["date"] = parse_dates(core["date"])
    frame["kickoff"] = raw["Time"].str.strip() if "Time" in raw else pd.NA
    for side in ("home_team", "away_team"):
        frame[side] = core[side].map(normalise_team, na_action="ignore")
    for side in ("home_goals", "away_goals"):
        frame[side] = pd.to_numeric(core[side], errors="coerce").astype("Int64")
    frame["result"] = core["result"].str.strip()
    for column, names in ODDS_CANDIDATES.items():
        frame[column] = _odds(_first_present(raw, names), raw.index)
    return frame
