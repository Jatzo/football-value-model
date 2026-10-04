"""Price football matches, look for value against bookmaker odds and backtest the results."""

import argparse
import logging
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

from valuemodel.backtest import MAIN_MODEL, MODELS, run_backtest
from valuemodel.config import (
    BACKTEST_SEASONS,
    BOOKMAKERS,
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    FIXTURE_LEAGUES,
    HISTORY_SEASONS,
    LEAGUES,
    MIN_TEAM_MATCHES,
    MODEL_XI,
    TUNING_SEASONS,
    Settings,
    current_season,
    history_seasons,
    linked_leagues,
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
from valuemodel.labels import MARKET_LABELS, OUTCOME_LABELS, STRATEGY_LABELS, label, share
from valuemodel.markets import predict
from valuemodel.models.common import UnknownTeamError
from valuemodel.odds import MARKETS, check_quotes
from valuemodel.picks import DEFAULT_LEGS, MAX_LEGS, SIZE_NAMES
from valuemodel.report import format_report, staking_description, table
from valuemodel.schedule import SCHEDULE_FILES, download_schedule, load_schedule, upcoming_games
from valuemodel.staking import stake
from valuemodel.store import connect, database_path, save_run
from valuemodel.teams import normalise_team
from valuemodel.tuning import SHOT_WEIGHT_GRID, XI_GRID, evaluate_shot_weights, evaluate_xi
from valuemodel.web import views


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


def _add_league(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--league",
        default=DEFAULT_LEAGUES[0],
        type=_argument(validate_league),
        metavar="CODE",
        help="league code (default: %(default)s)",
    )


def _add_leagues(
    parser: argparse.ArgumentParser, purpose: str, default: Sequence[str] = DEFAULT_LEAGUES
) -> None:
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(default),
        type=_argument(validate_league),
        metavar="CODE",
        help=f"{purpose} (default: %(default)s)",
    )


def _add_single_league(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--single-league",
        action="store_true",
        help="fit on the league's own results only, without the divisions linked to it",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    _add_league(parser)
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default=MAIN_MODEL,
        help="model to fit (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="valuemodel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download", help="download and cache results and odds from football-data.co.uk"
    )
    _add_leagues(download, "league codes such as E0", linked_leagues(DEFAULT_LEAGUES[0]))
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
    _add_single_league(tune)
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
    _add_leagues(fixtures, "leagues whose current season results to refresh", FIXTURE_LEAGUES)

    commands.add_parser(
        "picks",
        help="list value bets on listed fixtures and the odds the next round's games need",
    )

    slips = commands.add_parser(
        "slips",
        help="suggest the best slips from the value bets, or the likeliest ones before odds",
    )
    slips.add_argument(
        "--legs",
        type=int,
        choices=range(1, MAX_LEGS + 1),
        default=DEFAULT_LEGS,
        metavar="N",
        help=f"legs per slip, 1 to {MAX_LEGS} (default: %(default)s)",
    )

    shots = commands.add_parser(
        "tune-shots",
        help="score the shots-adjusted model for a range of expected goals weights",
    )
    _add_league(shots)
    _add_single_league(shots)
    shots.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(SHOT_WEIGHT_GRID),
        help="shares of expected goals in the blend to try (default: %(default)s)",
    )

    backtest = commands.add_parser(
        "backtest", help="walk forward through past seasons with paper bets and print a report"
    )
    _add_league(backtest)
    _add_single_league(backtest)
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
        "--xi",
        type=float,
        help="decay rate per day (default: the tuned value for the chosen model)",
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


def run_tune_xi(league: str, model: str, xi_values: Sequence[float], linked: bool) -> int:
    leagues = linked_leagues(league, linked)
    matches = load_matches(leagues, HISTORY_SEASONS + TUNING_SEASONS, load_settings())
    results = evaluate_xi(
        matches, TUNING_SEASONS, xi_values, fit=MODELS[model], history_leagues={league: leagues}
    )
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
    for outcome, outcome_label in OUTCOME_LABELS.items():
        chance = probabilities[outcome]
        line = f"{outcome_label:<10}  {chance:>7.1%}  {1 / chance:>9.2f}"
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
        print(f"No outcome reaches the {share(settings.edge_threshold)} edge threshold")
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
    xi: float | None,
    quoted: dict[str, float],
) -> int:
    settings = load_settings()
    seasons = [*DEFAULT_SEASONS, current_season()]
    matches = load_available(linked_leagues(league), seasons, settings)
    if matches.empty:
        raise FileNotFoundError(f"No cached data for {league}. Run: valuemodel download")
    if as_of is None:
        as_of = matches["date"].max() + pd.Timedelta(days=1)
    if xi is None:
        xi = MODEL_XI[model]
    home, away = normalise_team(home), normalise_team(away)
    fitted = MODELS[model](matches, as_of, xi)
    try:
        probabilities = predict(fitted, home, away)
        home_goals, away_goals = fitted.expected_goals(home, away)
    except UnknownTeamError as error:
        unknown = home if home not in fitted.teams else away
        suggestions = get_close_matches(unknown, fitted.teams, n=3)
        hint = f" Did you mean {' or '.join(suggestions)}?" if suggestions else ""
        print(f"error: {error}.{hint}", file=sys.stderr)
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

    model_name = label(STRATEGY_LABELS, model)
    print(f"{home} v {away}")
    print(f"{model_name}, fitted on {league} matches before {as_of.date()}, xi {xi}")
    print(f"Expected goals: {home} {home_goals:.2f}, {away} {away_goals:.2f}")
    _print_prices(probabilities.__dict__, quoted, settings)
    return 0


def run_fixtures(leagues: Sequence[str], today: date | None = None) -> int:
    settings = load_settings()
    today = today or date.today()
    season = current_season()
    scheduled = [league for league in leagues if league in SCHEDULE_FILES]
    try:
        with make_client() as client:
            path = download_fixtures(client, settings)
            time.sleep(settings.request_delay)
            download_seasons(client, leagues, [season], settings, force=True)
            schedules = {}
            for league in scheduled:
                time.sleep(settings.request_delay)
                schedules[league] = download_schedule(client, league, season, settings)
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
    for league in leagues:
        print(_schedule_line(league, schedules.get(league), today))
    return 0


def _schedule_line(league: str, path: Path | None, today: date) -> str:
    if path is None:
        return f"{league} schedule: no source for this league"
    games = upcoming_games(load_schedule(path, league), today, rounds=None)
    if games.empty:
        return f"{league} schedule: no games left to play"
    return f"{league} schedule: {len(games)} games to play, next on {games['date'].min().date()}"


PICK_HEADERS = ["Match", "Bet", "Chance", "Fair odds", "Odds", "Edge", "Stake"]
SCHEDULE_HEADERS = ["Match", "Home", "Draw", "Away", "Over 2.5", "Under 2.5"]


def run_picks(today: date | None = None) -> int:
    settings = load_settings()
    today = today or date.today()
    bookmaker = BOOKMAKERS[settings.bookmaker]
    fixtures = views.fixtures_view(settings)
    if fixtures.status == "missing":
        raise FileNotFoundError("No fixtures downloaded yet. Run: valuemodel fixtures")

    print(f"Paper bets at {bookmaker} odds, biggest edge first")
    if fixtures.picks:
        print(table(PICK_HEADERS, [_pick_row(pick) for pick in fixtures.picks]))
        print(
            f"Stakes are {staking_description(settings)} on a "
            f"{settings.starting_bankroll:g} unit bankroll."
        )
    else:
        print(f"No listed game has a value bet at {bookmaker}'s current odds.")

    schedule = views.schedule_view(settings, rounds=1, today=today)
    if schedule.missing:
        print("\nNo season schedule downloaded yet. Run: valuemodel fixtures")
    for league, games in schedule.rows.items():
        print(f"\n{LEAGUES[league]}: lowest {bookmaker} odds that would be a value bet")
        if not games:
            print("No games left to play.")
            continue
        print(table(SCHEDULE_HEADERS, [_price_to_beat_row(game) for game in games]))
    for league, reason in schedule.problems.items():
        print(f"\n{LEAGUES[league]}: schedule not shown ({reason})")
    print(
        f"\nValue means an edge of at least {share(settings.edge_threshold)}. In the backtests, "
        "bets chosen this way\nlost money, so treat these as a test of the model, not as tips."
    )
    return 0


def run_slips(today: date | None = None, legs: int = DEFAULT_LEGS) -> int:
    settings = load_settings()
    today = today or date.today()
    bookmaker = BOOKMAKERS[settings.bookmaker]
    fixtures = views.fixtures_view(settings, legs)
    if fixtures.status == "missing":
        raise FileNotFoundError("No fixtures downloaded yet. Run: valuemodel fixtures")
    if fixtures.slips:
        print(f"Value slips at {bookmaker} odds, one leg per match\n")
        for slip in fixtures.slips:
            print(
                f"{slip['name']}: odds {slip['odds']:.2f}, model chance {slip['probability']:.1%}, "
                f"edge {slip['edge']:+.1%}, paper stake {slip['stake']:.2f}, "
                f"returns {slip['returns']:.2f}"
            )
            for leg in slip["legs"]:
                print(
                    f"  {leg['match']}: {leg['bet']} at {leg['odds']:.2f}, "
                    f"model chance {leg['chance']:.1%}"
                )
            print()
        print(
            f"Stakes are {staking_description(settings)} on a "
            f"{settings.starting_bankroll:g} unit bankroll.\n"
        )
    elif fixtures.picks:
        size = SIZE_NAMES[legs].lower()
        print(f"No value {size}: too few matches have a value bet for {legs} legs.\n")
    else:
        print(f"No value slips: no listed game has a value bet at {bookmaker}'s current odds.\n")

    schedule = views.schedule_view(settings, rounds=1, today=today)
    likely = views.likely_slip_cards(schedule, settings.edge_threshold, legs)
    if likely:
        print("Likeliest slips for the next round, which need no odds\n")
    for slip in likely:
        print(
            f"{slip['name']}: model chance {slip['probability']:.1%}, "
            f"fair odds {slip['fair_odds']:.2f}, value from {slip['price_to_beat']:.2f}"
        )
        for leg in slip["legs"]:
            print(
                f"  {leg['match']}: {leg['bet']}, model chance {leg['chance']:.1%}, "
                f"value from {leg['price_to_beat']:.2f}"
            )
        print()
    print(
        f"Value from is the lowest {bookmaker} odds that would make a slip a value bet.\n"
        "Each extra leg multiplies the model's errors, and in the backtests single bets chosen\n"
        "by edge lost money, so treat these as a test of the model."
    )
    return 0


def _match(day: pd.Timestamp, time_of_day: object, home: str, away: str) -> str:
    kickoff = f"{day:%a %d %b} {time_of_day or ''}".strip()
    return f"{kickoff}  {home} v {away}"


def _pick_row(pick: dict[str, Any]) -> list[str]:
    return [
        _match(pick["date"], pick["kickoff"], pick["home_team"], pick["away_team"]),
        OUTCOME_LABELS[pick["outcome"]],
        f"{pick['probability']:.1%}",
        f"{pick['fair_odds']:.2f}",
        f"{pick['odds']:.2f}",
        f"{pick['edge']:+.1%}",
        f"{pick['stake']:.2f}",
    ]


def _price_to_beat_row(game: views.ScheduleRow) -> list[str]:
    prices = [f"{cell.price_to_beat:.2f}" for cell in game.cells]
    return [
        _match(game.date, game.kickoff, game.home_team, game.away_team),
        *(prices or ["not priced", "", "", "", ""]),
    ]


def run_tune_shots(league: str, weights: Sequence[float], linked: bool) -> int:
    leagues = linked_leagues(league, linked)
    matches = load_matches(leagues, HISTORY_SEASONS + TUNING_SEASONS, load_settings())
    results = evaluate_shot_weights(
        matches, TUNING_SEASONS, weights, history_leagues={league: leagues}
    )
    print(f"Shots-adjusted on {league}, forecasting seasons {', '.join(TUNING_SEASONS)}")
    print(f"{'weight':>8}  {'matches':>7}  {'log loss':>8}  {'RPS':>7}  {'Brier':>7}")
    for row in results.itertuples():
        print(
            f"{row.weight:>8.2f}  {row.matches:>7}  {row.log_loss:>8.5f}  "
            f"{row.rps:>7.5f}  {row.brier:>7.5f}"
        )
    best = results.loc[results["rps"].idxmin(), "weight"]
    print(f"Lowest ranked probability score at weight = {best}")
    return 0


def run_backtest_command(league: str, seasons: Sequence[str], save: bool, linked: bool) -> int:
    settings = load_settings()
    leagues = linked_leagues(league, linked)
    matches = load_matches(leagues, history_seasons(seasons), settings)
    result = run_backtest(matches, league, seasons, settings, linked=linked)
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
    commands: dict[str, Callable[[], int]] = {
        "download": lambda: run_download(args.leagues, args.seasons, args.refresh),
        "tune-xi": lambda: run_tune_xi(args.league, args.model, args.xi, not args.single_league),
        "fixtures": lambda: run_fixtures(args.leagues),
        "picks": lambda: run_picks(),
        "slips": lambda: run_slips(legs=args.legs),
        "tune-shots": lambda: run_tune_shots(args.league, args.weights, not args.single_league),
        "backtest": lambda: run_backtest_command(
            args.league, args.seasons, not args.no_save, not args.single_league
        ),
        "predict": lambda: run_predict(
            args.league, args.model, args.home, args.away, args.as_of, args.xi, _quoted_odds(args)
        ),
    }
    try:
        return commands[args.command]()
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
