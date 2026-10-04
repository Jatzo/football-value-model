"""Command line entry point."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

import pandas as pd

from valuemodel.backtest import MODELS, run_backtest
from valuemodel.config import (
    BACKTEST_SEASONS,
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    DEFAULT_XI,
    HISTORY_SEASONS,
    MIN_TEAM_MATCHES,
    TUNING_SEASONS,
    Settings,
    current_season,
    load_settings,
    season_label,
    validate_league,
    validate_season,
)
from valuemodel.data import (
    DownloadError,
    cache_path,
    download_seasons,
    load_available,
    load_matches,
    load_season,
    make_client,
)
from valuemodel.fixtures import download_fixtures, load_fixtures
from valuemodel.labels import MARKET_LABELS, OUTCOME_LABELS
from valuemodel.markets import predict
from valuemodel.models.common import UnknownTeamError
from valuemodel.odds import MARKETS, check_quotes
from valuemodel.report import format_report, staking_description
from valuemodel.staking import stake
from valuemodel.store import connect, database_path, save_run
from valuemodel.teams import normalise_team
from valuemodel.tuning import XI_GRID, evaluate_xi


def _argument[T](check: Callable[[str], T]) -> Callable[[str], T]:
    """Let argparse show the checker's own message instead of a generic one."""

    def parse(value: str) -> T:
        try:
            return check(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return parse


def _decimal_odds(value: str) -> float:
    odds = float(value)
    if not odds > 1.0:
        raise ValueError(f"decimal odds must be greater than 1, got {value}")
    return odds


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

    fixtures = commands.add_parser(
        "fixtures",
        help="download upcoming fixtures and refresh this season's results for the dashboard",
    )
    fixtures.add_argument(
        "--leagues",
        nargs="+",
        default=list(DEFAULT_LEAGUES),
        type=_argument(validate_league),
        metavar="CODE",
        help="leagues whose current season results to refresh (default: %(default)s)",
    )

    backtest = commands.add_parser(
        "backtest", help="walk forward through past seasons with paper bets and print a report"
    )
    backtest.add_argument(
        "--league",
        default=DEFAULT_LEAGUES[0],
        type=_argument(validate_league),
        metavar="CODE",
        help="league code (default: %(default)s)",
    )
    backtest.add_argument(
        "--seasons",
        nargs="+",
        default=list(BACKTEST_SEASONS),
        type=_argument(validate_season),
        metavar="CODE",
        help="seasons to bet on (default: %(default)s)",
    )
    backtest.add_argument(
        "--no-save", action="store_true", help="print the report without saving the run"
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
    forecast.add_argument(
        "--odds",
        nargs=3,
        type=_argument(_decimal_odds),
        metavar=("HOME", "DRAW", "AWAY"),
        help="bookmaker decimal odds for the match result, to check for value",
    )
    forecast.add_argument(
        "--totals-odds",
        nargs=2,
        type=_argument(_decimal_odds),
        metavar=("OVER", "UNDER"),
        help="bookmaker decimal odds for over and under 2.5 goals, to check for value",
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


def _print_prices(
    probabilities: dict[str, float], quoted: dict[str, float], settings: Settings
) -> None:
    """Show model prices and, for any quoted market, the edge against the bookmaker."""
    checks, margins = check_quotes(
        probabilities, quoted, settings.margin_method, settings.edge_threshold
    )
    by_outcome = {check.outcome: check for check in checks}

    header = f"{'Market':<10}  {'Chance':>7}  {'Fair odds':>9}"
    if checks:
        header += f"  {'Odds':>6}  {'Book chance':>11}  {'Edge':>7}"
    print(header)
    for outcome, label in OUTCOME_LABELS.items():
        chance = probabilities[outcome]
        line = f"{label:<10}  {chance:>7.1%}  {1 / chance:>9.2f}"
        if check := by_outcome.get(outcome):
            flag = "  value" if check.value else ""
            line += (
                f"  {check.odds:>6.2f}  {check.book_probability:>11.1%}  {check.edge:>+7.1%}{flag}"
            )
        print(line)
    if not checks:
        return

    for market, margin in margins.items():
        print(f"Bookmaker margin, {MARKET_LABELS[market].lower()}: {margin:.1%}")
    bets = [check for check in checks if check.value]
    if not bets:
        print(f"No outcome reaches the {settings.edge_threshold:.0%} edge threshold")
    for bet in bets:
        amount = stake(bet.probability, bet.odds, settings.starting_bankroll, settings)
        print(
            f"Paper bet: {OUTCOME_LABELS[bet.outcome]} at {bet.odds:.2f}, stake {amount:.2f} "
            f"of a {settings.starting_bankroll:g} unit bankroll ({staking_description(settings)})"
        )


def run_predict(
    league: str,
    model: str,
    home: str,
    away: str,
    as_of: pd.Timestamp | None,
    xi: float,
    quoted: dict[str, float],
) -> int:
    settings = load_settings()
    matches = load_available([league], [*DEFAULT_SEASONS, current_season()], settings)
    if matches.empty:
        raise FileNotFoundError(f"No cached data for {league}. Run: valuemodel download")
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
    _print_prices(probabilities.__dict__, quoted, settings)
    return 0


def run_fixtures(leagues: Sequence[str]) -> int:
    settings = load_settings()
    season = current_season()
    try:
        with make_client() as client:
            path = download_fixtures(client, settings)
            download_seasons(client, leagues, [season], settings, force=True)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    fixtures = load_fixtures(path)
    counts = fixtures["league"].value_counts(sort=False)
    listed = ", ".join(f"{league} {count}" for league, count in counts.items()) or "none"
    print(f"Fixtures file has {len(fixtures)} matches by league: {listed}")
    for league in leagues:
        played = load_season(cache_path(settings.raw_dir, league, season), league, season)
        latest = played["date"].max().date() if len(played) else "no matches yet"
        print(f"{league} {season_label(season)}: {len(played)} results, latest {latest}")
    return 0


def run_backtest_command(league: str, seasons: Sequence[str], save: bool) -> int:
    settings = load_settings()
    needed = list(dict.fromkeys([*DEFAULT_SEASONS, *seasons]))
    matches = load_matches([league], needed, settings)
    result = run_backtest(matches, league, seasons, settings)
    print(format_report(result))
    if save:
        path = database_path(settings)
        connection = connect(path)
        try:
            run_id = save_run(connection, result)
        finally:
            connection.close()
        print()
        print(f"Saved as run {run_id} in {path}")
    return 0


def _quoted_odds(args: argparse.Namespace) -> dict[str, float]:
    quoted: dict[str, float] = {}
    if args.odds:
        quoted.update(zip(MARKETS["1x2"], args.odds, strict=True))
    if args.totals_odds:
        quoted.update(zip(MARKETS["totals"], args.totals_odds, strict=True))
    return quoted


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            return run_download(args.leagues, args.seasons, args.refresh)
        if args.command == "tune-xi":
            return run_tune_xi(args.league, args.model, args.xi)
        if args.command == "fixtures":
            return run_fixtures(args.leagues)
        if args.command == "backtest":
            return run_backtest_command(args.league, args.seasons, not args.no_save)
        if args.command == "predict":
            return run_predict(
                args.league,
                args.model,
                args.home,
                args.away,
                args.as_of,
                args.xi,
                _quoted_odds(args),
            )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
