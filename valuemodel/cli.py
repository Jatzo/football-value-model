"""Command line entry point."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from valuemodel.config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    load_settings,
    validate_league,
    validate_season,
)
from valuemodel.data import (
    DownloadError,
    cache_path,
    download_seasons,
    load_season,
    make_client,
)


def _argument(check: Callable[[str], str]) -> Callable[[str], str]:
    """Let argparse show the checker's own message instead of a generic one."""

    def parse(value: str) -> str:
        try:
            return check(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return parse


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


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "download":
        return run_download(args.leagues, args.seasons, args.refresh)
    return 2


if __name__ == "__main__":
    sys.exit(main())
