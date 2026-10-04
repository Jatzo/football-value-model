"""Command line entry point."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

import pandas as pd

from valuemodel.config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    DEFAULT_XI,
    HISTORY_SEASONS,
    MIN_TEAM_MATCHES,
    TUNING_SEASONS,
    load_settings,
    validate_league,
    validate_season,
)
from valuemodel.data import (
    DownloadError,
    cache_path,
    download_seasons,
    load_matches,
    load_season,
    make_client,
)
from valuemodel.markets import predict
from valuemodel.models.common import FittedModel, UnknownTeamError
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.models.poisson import fit_poisson
from valuemodel.teams import normalise_team
from valuemodel.tuning import XI_GRID, evaluate_xi

MODELS: dict[str, Callable[..., FittedModel]] = {
    "dixon-coles": fit_dixon_coles,
    "poisson": fit_poisson,
}

MARKET_LABELS: dict[str, str] = {
    "home": "Home win",
    "draw": "Draw",
    "away": "Away win",
    "over25": "Over 2.5",
    "under25": "Under 2.5",
}


def _argument(check: Callable[[str], str]) -> Callable[[str], str]:
    """Let argparse show the checker's own message instead of a generic one."""

    def parse(value: str) -> str:
        try:
            return check(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return parse


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--league",
        default=DEFAULT_LEAGUES[0],
        type=_argument(validate_league),
        metavar="CODE",
        help="league code (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="dixon-coles",
        help="model to fit (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="valuemodel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download", help="download and cache results and odds from football-data.co.uk"
    )
    download.add_argument(
        "--leagues",
        nargs="+",
        default=list(DEFAULT_LEAGUES),
        type=_argument(validate_league),
        metavar="CODE",
        help="league codes such as E0 (default: %(default)s)",
    )
    download.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        type=_argument(validate_season),
        metavar="CODE",
        help="season codes such as 2425 (default: %(default)s)",
    )
    download.add_argument(
        "--refresh",
        action="store_true",
        help="download again even when a cached copy exists",
    )

    tune = commands.add_parser(
        "tune-xi", help="score walk-forward forecasts for a range of time decay rates"
    )
    _add_model_arguments(tune)
    tune.add_argument(
        "--xi",
        nargs="+",
        type=float,
        default=list(XI_GRID),
        help="decay rates per day to try (default: %(default)s)",
    )

    forecast = commands.add_parser("predict", help="price one match with the fitted model")
    _add_model_arguments(forecast)
    forecast.add_argument("--home", required=True, help="home team, as spelt in the data")
    forecast.add_argument("--away", required=True, help="away team, as spelt in the data")
    forecast.add_argument(
        "--as-of",
        type=pd.Timestamp,
        help="fit on matches before this date (default: the day after the last cached match)",
    )
    forecast.add_argument(
        "--xi", type=float, default=DEFAULT_XI, help="decay rate per day (default: %(default)s)"
    )
    return parser


def run_download(leagues: Sequence[str], seasons: Sequence[str], refresh: bool) -> int:
    settings = load_settings()
    try:
        with make_client() as client:
            results = download_seasons(client, leagues, seasons, settings, force=refresh)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for league, season, downloaded in results:
        matches = load_season(cache_path(settings.raw_dir, league, season), league, season)
        closing = int(matches["pinnacle_close_home"].notna().sum())
        status = "downloaded" if downloaded else "cached"
        print(
            f"{league} {season}  {status:<10}  {len(matches):>3} matches  "
            f"Pinnacle closing odds for {closing}"
        )
    return 0


def run_tune_xi(league: str, model: str, xi_values: Sequence[float]) -> int:
    matches = load_matches([league], HISTORY_SEASONS + TUNING_SEASONS, load_settings())
    results = evaluate_xi(matches, TUNING_SEASONS, xi_values, fit=MODELS[model])
    print(f"{model} on {league}, forecasting seasons {', '.join(TUNING_SEASONS)}")
    print(f"{'xi':>8}  {'matches':>7}  {'log loss':>8}  {'RPS':>7}  {'Brier':>7}")
    for row in results.itertuples():
        print(
            f"{row.xi:>8.4f}  {row.matches:>7}  {row.log_loss:>8.5f}  "
            f"{row.rps:>7.5f}  {row.brier:>7.5f}"
        )
    best = results.loc[results["rps"].idxmin(), "xi"]
    print(f"Lowest ranked probability score at xi = {best}")
    return 0


def run_predict(
    league: str, model: str, home: str, away: str, as_of: pd.Timestamp | None, xi: float
) -> int:
    matches = load_matches([league], DEFAULT_SEASONS, load_settings())
    if as_of is None:
        as_of = matches["date"].max() + pd.Timedelta(days=1)
    home, away = normalise_team(home), normalise_team(away)
    fitted = MODELS[model](matches, as_of, xi)
    try:
        probabilities = predict(fitted, home, away)
        home_goals, away_goals = fitted.expected_goals(home, away)
    except UnknownTeamError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # With only a handful of results a team's estimates can run to extremes, for
    # example a single clean sheet implies a defence that never concedes.
    for team in (home, away):
        if not fitted.is_reliable(team, MIN_TEAM_MATCHES):
            count = fitted.match_counts[team]
            print(
                f"error: {team} has {count} {'match' if count == 1 else 'matches'} in the "
                f"training window, fewer than the {MIN_TEAM_MATCHES} needed for a reliable price",
                file=sys.stderr,
            )
            return 1

    print(f"{home} v {away}")
    print(f"{model}, fitted on {league} matches before {as_of.date()}, xi {xi}")
    print(f"Expected goals: {home} {home_goals:.2f}, {away} {away_goals:.2f}")
    print(f"{'Market':<10}  {'Chance':>7}  {'Fair odds':>9}")
    fair_odds = probabilities.fair_odds()
    for name, label in MARKET_LABELS.items():
        print(f"{label:<10}  {getattr(probabilities, name):>7.1%}  {fair_odds[name]:>9.2f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            return run_download(args.leagues, args.seasons, args.refresh)
        if args.command == "tune-xi":
            return run_tune_xi(args.league, args.model, args.xi)
        if args.command == "predict":
            return run_predict(args.league, args.model, args.home, args.away, args.as_of, args.xi)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
