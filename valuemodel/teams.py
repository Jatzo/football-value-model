"""Team name normalisation.

football-data.co.uk supplies the results the models are fitted on, so its
spellings are the canonical ones. Aliases are only added for variants actually
seen in the data. A scan of every Premier League season the project uses,
2019/20 to 2026/27, found none, so the table is empty.

The season schedules come from openfootball, which uses full club names.
SCHEDULE_NAMES maps each one seen in its 2026/27 Premier League and
Championship files to the football-data spelling. A promoted club will need
an entry when it first appears.
"""

from collections.abc import Mapping

TEAM_ALIASES: dict[str, str] = {}

SCHEDULE_NAMES: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Birmingham City FC": "Birmingham",
    "Blackburn Rovers FC": "Blackburn",
    "Bolton Wanderers FC": "Bolton",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Bristol City FC": "Bristol City",
    "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff",
    "Charlton Athletic FC": "Charlton",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Derby County FC": "Derby",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Lincoln City FC": "Lincoln",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Middlesbrough FC": "Middlesbrough",
    "Millwall FC": "Millwall",
    "Newcastle United FC": "Newcastle",
    "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest",
    "Portsmouth FC": "Portsmouth",
    "Preston North End FC": "Preston",
    "Queens Park Rangers FC": "QPR",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke",
    "Sunderland AFC": "Sunderland",
    "Swansea City AFC": "Swansea",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Brom",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Wrexham AFC": "Wrexham",
}


def normalise_team(name: str, aliases: Mapping[str, str] = TEAM_ALIASES) -> str:
    """Return the canonical spelling of a team name.

    Whitespace is collapsed first because stray spaces, including non-breaking
    ones, are the commonest difference between otherwise identical names.
    """
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("Team name is empty")
    return aliases.get(cleaned, cleaned)


def schedule_team(name: str) -> str:
    """The football-data spelling of a team named in a season schedule."""
    cleaned = " ".join(name.split())
    if cleaned not in SCHEDULE_NAMES:
        raise ValueError(
            f"Schedule team {cleaned!r} has no football-data name. Add it to SCHEDULE_NAMES."
        )
    return SCHEDULE_NAMES[cleaned]
