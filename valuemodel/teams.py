"""Team name normalisation.

football-data.co.uk is the single source of names, so its spellings are the
canonical ones. Aliases are only added for variants actually seen in the data.
A scan of every Premier League season the project uses, 2019/20 to
2026/27, found none, so the table is empty. Other leagues may need entries.
"""

from collections.abc import Mapping

TEAM_ALIASES: dict[str, str] = {}


def normalise_team(name: str, aliases: Mapping[str, str] = TEAM_ALIASES) -> str:
    """Return the canonical spelling of a team name.

    Whitespace is collapsed first because stray spaces, including non-breaking
    ones, are the commonest difference between otherwise identical names.
    """
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("Team name is empty")
    return aliases.get(cleaned, cleaned)
