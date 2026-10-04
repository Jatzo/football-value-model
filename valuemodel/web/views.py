"""Turn stored backtest runs and cached fixtures into what each page shows.

Nothing here knows about Flask, so the page logic can be tested directly.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from valuemodel.backtest import (
    MAIN_MODEL,
    MODELS,
    BacktestResult,
    main_strategy,
    probability_bands,
)
from valuemodel.config import (
    DEFAULT_SEASONS,
    LEAGUES,
    MODEL_XI,
    Settings,
    current_season,
    linked_leagues,
)
from valuemodel.data import load_available
from valuemodel.fixtures import fetched_at, fixtures_path, load_fixtures, price_fixtures
from valuemodel.labels import (
    MARKET_LABELS,
    NO_COMMON_MATCHES,
    STRATEGY_LABELS,
    is_missing,
    label,
    percent,
    share,
    tone,
)
from valuemodel.odds import MARKETS
from valuemodel.schedule import (
    SCHEDULE_FILES,
    load_schedule,
    predict_games,
    schedule_path,
    upcoming_games,
)

PER_PAGE = 50

MODEL_FORECASTERS = tuple(MODELS)


@dataclass
class Card:
    label: str
    value: str
    tone: str = ""
    note: str = ""


def headline_cards(result: BacktestResult) -> list[Card]:
    """The main model's most important figures, closing line value first."""
    row = result.summary.set_index("strategy").loc[main_strategy(result)]
    start = result.settings.starting_bankroll
    if not row["bets"]:
        threshold = share(result.settings.edge_threshold)
        return [
            Card(
                "Bets placed",
                "0",
                note=f"no outcome reached the {threshold} edge threshold",
            )
        ]
    return [
        Card(
            "Mean closing line value",
            percent(row["mean_clv"], signed=True),
            tone(row["mean_clv"]),
            f"on {int(row['clv_bets']):,} bets with Pinnacle closing odds",
        ),
        Card(
            "Beat the closing line",
            percent(row["beat_close_share"]),
            note="share of bets struck at a better price than the close",
        ),
        Card(
            "ROI",
            percent(row["roi"], signed=True),
            tone(row["roi"]),
            f"on {int(row['bets']):,} bets",
        ),
        Card(
            "Level-stakes ROI",
            percent(row["level_roi"], signed=True),
            tone(row["level_roi"]),
            f"95% interval {percent(row['level_roi_low'], signed=True)} to "
            f"{percent(row['level_roi_high'], signed=True)}",
        ),
        Card(
            "Final bankroll",
            f"{row['final_bankroll']:,.0f}",
            tone(row["final_bankroll"] - start),
            f"from {start:,.0f} units",
        ),
        Card(
            "Maximum drawdown",
            percent(row["max_drawdown_share"]),
            note=f"{row['max_drawdown']:,.0f} units, longest losing run "
            f"{int(row['longest_losing_run'])}",
        ),
    ]


def clv_explanation(result: BacktestResult) -> str:
    """Explain closing line value in terms of what this run actually shows."""
    mean_clv = result.summary.set_index("strategy").loc[main_strategy(result), "mean_clv"]
    intro = (
        "Closing line value compares each price taken with Pinnacle's closing price after its "
        "margin is removed. It is the most reliable sign of a real edge, because it is far less "
        "noisy than profit."
    )
    if is_missing(mean_clv):
        return f"{intro} No bets in this run had a closing price to compare with."
    if mean_clv < 0:
        return (
            f"{intro} Here it is negative: the market moved against the model's selections more "
            "often than not. The model finds prices where it disagrees with the market, and the "
            "market tends to be right."
        )
    return (
        f"{intro} Here it is positive: on average the prices taken beat the close, which would "
        "be evidence of an edge if it holds over enough bets."
    )


def bankroll_series(result: BacktestResult) -> list[dict[str, object]]:
    """End-of-day bankroll for each strategy, from the opening bankroll to the last bet.

    A list keeps the strategies in order once serialised. Each line is carried
    flat to the final betting day so strategies that stopped early still span
    the whole chart. Points are millisecond timestamps for the chart's axis.
    """
    placed = {name: bets for name, bets in result.bets.items() if not bets.empty}
    if not placed:
        return []
    first = min(bets["date"].min() for bets in placed.values()) - pd.Timedelta(days=1)
    last = max(bets["date"].max() for bets in placed.values())
    series = []
    for strategy, bets in placed.items():
        daily = bets.groupby("date")["bankroll"].last()
        if daily.index[-1] < last:
            daily[last] = daily.iloc[-1]
        points = [{"x": _milliseconds(first), "y": result.settings.starting_bankroll}]
        points += [{"x": _milliseconds(day), "y": round(value, 2)} for day, value in daily.items()]
        series.append({"name": label(STRATEGY_LABELS, strategy), "points": points})
    return series


def _milliseconds(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def calibration_series(result: BacktestResult) -> list[dict[str, object]]:
    return [
        {
            "name": label(STRATEGY_LABELS, model),
            "points": [
                {"x": row.mean_forecast, "y": row.observed, "count": int(row.count)}
                for row in table.itertuples()
            ],
        }
        for model, table in result.calibration.items()
    ]


def scores_verdict(scores: pd.DataFrame) -> str:
    """One sentence saying whether the models or the market forecast better on RPS."""
    if scores.empty or not scores["matches"].iloc[0] or scores["rps"].isna().all():
        return NO_COMMON_MATCHES
    is_model = scores["forecaster"].isin(MODEL_FORECASTERS)
    model_best = scores.loc[is_model, "rps"].min()
    market_best = scores.loc[~is_model, "rps"].min()
    if model_best < market_best:
        return (
            "The best model forecasts better than the market on ranked probability score, "
            "which would be unusual and worth checking carefully."
        )
    return (
        "The bookmakers' prices, with their margins removed, are better forecasts than either "
        "model. A model built only from past scores knows nothing about injuries, suspensions "
        "or team news, all of which the market prices in."
    )


@dataclass
class BetFilters:
    strategy: str = ""
    league: str = ""
    season: str = ""
    market: str = ""
    result: str = ""
    sort: str = "newest"
    page: int = 1

    @classmethod
    def from_args(cls, args: dict[str, str]) -> "BetFilters":
        try:
            page = max(1, int(args.get("page", "1")))
        except ValueError:
            page = 1
        return cls(
            strategy=args.get("strategy", ""),
            league=args.get("league", ""),
            season=args.get("season", ""),
            market=args.get("market", ""),
            result=args.get("result", ""),
            sort="probability" if args.get("sort") == "probability" else "newest",
            page=page,
        )


@dataclass
class BetPage:
    rows: pd.DataFrame
    page: int
    pages: int
    totals: dict[str, float]
    options: dict[str, list[str]] = field(default_factory=dict)


def filter_bets(bets: pd.DataFrame, filters: BetFilters) -> pd.DataFrame:
    keep = pd.Series(True, index=bets.index)
    for column in ("league", "season", "market"):
        wanted = getattr(filters, column)
        if wanted:
            keep &= bets[column] == wanted
    if filters.result in ("won", "lost"):
        keep &= bets["won"] == (filters.result == "won")
    return bets[keep]


def bet_page(result: BacktestResult, filters: BetFilters) -> BetPage:
    """One page of the bet log for the chosen filters, newest bets first."""
    if filters.strategy not in result.bets:
        filters.strategy = main_strategy(result)
    bets = result.bets.get(filters.strategy, pd.DataFrame())
    chosen = filter_bets(bets, filters) if len(bets) else bets
    staked = float(chosen["stake"].sum()) if len(chosen) else 0.0
    profit = float(chosen["profit"].sum()) if len(chosen) else 0.0
    totals = {
        "bets": len(chosen),
        "won": int(chosen["won"].sum()) if len(chosen) else 0,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else math.nan,
        "mean_clv": float(chosen["clv"].mean()) if len(chosen) else math.nan,
    }
    pages = max(1, math.ceil(len(chosen) / PER_PAGE))
    page = min(filters.page, pages)
    if filters.sort == "probability" and len(chosen):
        ordered = chosen.sort_values("probability", ascending=False, kind="stable")
    else:
        ordered = chosen.iloc[::-1] if len(chosen) else chosen
    rows = ordered.iloc[(page - 1) * PER_PAGE : page * PER_PAGE]
    options = {
        "strategy": list(result.bets),
        "league": sorted(bets["league"].unique()) if len(bets) else [],
        "season": sorted(bets["season"].unique()) if len(bets) else [],
        "market": list(MARKET_LABELS),
    }
    return BetPage(rows=rows, page=page, pages=pages, totals=totals, options=options)


@dataclass
class PriceCell:
    fair_odds: float
    offered: float
    edge: float
    value: bool
    likely: bool = False


@dataclass
class FixtureRow:
    date: pd.Timestamp
    kickoff: str
    home_team: str
    away_team: str
    reliable: bool
    cells: list[PriceCell]


def fixture_rows(priced: pd.DataFrame, league: str) -> list[FixtureRow]:
    """One row per fixture, with a cell for each outcome the model priced."""
    rows = []
    for fixture in priced[priced["league"] == league].to_dict("records"):
        cells = []
        if fixture["reliable"]:
            likeliest = max(MARKETS["1x2"], key=lambda outcome: fixture[outcome])
            for market, outcomes in MARKETS.items():
                for outcome in outcomes:
                    cells.append(
                        PriceCell(
                            fair_odds=1 / fixture[outcome],
                            offered=fixture[f"odds_{outcome}"],
                            edge=fixture[f"edge_{outcome}"],
                            value=fixture[f"value_{market}"] == outcome,
                            likely=outcome == likeliest,
                        )
                    )
        kickoff = fixture["kickoff"]
        rows.append(
            FixtureRow(
                date=fixture["date"],
                kickoff="" if pd.isna(kickoff) else str(kickoff),
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                reliable=bool(fixture["reliable"]),
                cells=cells,
            )
        )
    return rows


@dataclass
class LikelyOutcome:
    date: pd.Timestamp
    league: str
    home_team: str
    away_team: str
    outcome: str
    probability: float
    offered: float
    edge: float


def likely_outcomes(priced: pd.DataFrame) -> list[LikelyOutcome]:
    """Each priced fixture's most likely match result, the likeliest first.

    This is the model's view of what will probably happen, not a list of bets:
    a likely outcome is usually a short price, and only pays over time when the
    odds offered beat the probability.
    """
    likely = []
    for fixture in priced[priced["reliable"]].to_dict("records"):
        outcome = max(MARKETS["1x2"], key=lambda name: fixture[name])
        likely.append(
            LikelyOutcome(
                date=fixture["date"],
                league=fixture["league"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                outcome=outcome,
                probability=fixture[outcome],
                offered=fixture[f"odds_{outcome}"],
                edge=fixture[f"edge_{outcome}"],
            )
        )
    return sorted(likely, key=lambda item: item.probability, reverse=True)


def probability_band_rows(result: BacktestResult) -> list[dict[str, object]]:
    """How each model's bets fared, grouped by its chance of them winning."""
    rows = []
    for strategy in MODELS:
        bets = result.bets.get(strategy)
        if bets is None or bets.empty:
            continue
        for band in probability_bands(bets).to_dict("records"):
            rows.append({"strategy": strategy, **band})
    return rows


@dataclass
class FixturesView:
    status: str
    fetched: datetime | None = None
    rows: dict[str, list[FixtureRow]] = field(default_factory=dict)
    priced_leagues: list[str] = field(default_factory=list)
    unpriced_leagues: dict[str, str] = field(default_factory=dict)
    has_odds: bool = True
    likely: list[LikelyOutcome] = field(default_factory=list)
    latest_result: dict[str, str] = field(default_factory=dict)


def fixtures_view(settings: Settings) -> FixturesView:
    path = fixtures_path(settings)
    if not path.exists():
        return FixturesView(status="missing")
    fixtures = load_fixtures(path)
    if fixtures.empty:
        return FixturesView(status="empty", fetched=fetched_at(path))

    leagues = [league for league in dict.fromkeys(fixtures["league"]) if league in LEAGUES]
    sources = dict.fromkeys(source for league in leagues for source in linked_leagues(league))
    history = load_available(sources, [*DEFAULT_SEASONS, current_season()], settings)
    if history.empty:
        return FixturesView(
            status="no_history",
            fetched=fetched_at(path),
            unpriced_leagues=dict.fromkeys(fixtures["league"], "no cached results"),
        )
    priced = price_fixtures(history, fixtures, settings, MODEL_XI[MAIN_MODEL])
    latest = {
        league: str(group["date"].max().date()) for league, group in history.groupby("league")
    }
    return FixturesView(
        status="ok" if priced.priced_leagues else "no_history",
        fetched=fetched_at(path),
        rows={league: fixture_rows(priced.fixtures, league) for league in priced.priced_leagues},
        priced_leagues=priced.priced_leagues,
        unpriced_leagues=priced.unpriced_leagues,
        has_odds=bool(priced.fixtures.filter(like="odds_").notna().any().any()),
        likely=likely_outcomes(priced.fixtures),
        latest_result=latest,
    )


ROUND_CHOICES: tuple[int, ...] = (1, 3, 6, 0)
ALL_ROUNDS = 0
DEFAULT_ROUNDS = 3


def parse_rounds(value: str | None) -> int:
    """How many rounds of the schedule to show, falling back to the default."""
    try:
        rounds = int(value or DEFAULT_ROUNDS)
    except ValueError:
        return DEFAULT_ROUNDS
    return rounds if rounds in ROUND_CHOICES else DEFAULT_ROUNDS


@dataclass
class ChanceCell:
    chance: float
    fair_odds: float
    likely: bool


@dataclass
class ScheduleRow:
    date: pd.Timestamp
    kickoff: str
    round: str
    home_team: str
    away_team: str
    reliable: bool
    cells: list[ChanceCell]
    home_goals: float
    away_goals: float


def schedule_rows(priced: pd.DataFrame) -> list[ScheduleRow]:
    """One row per scheduled game, with the model's chance of each outcome."""
    rows = []
    for game in priced.to_dict("records"):
        cells = []
        if game["reliable"]:
            likeliest = max(MARKETS["1x2"], key=lambda outcome: game[outcome])
            cells = [
                ChanceCell(
                    chance=game[outcome], fair_odds=1 / game[outcome], likely=outcome == likeliest
                )
                for outcomes in MARKETS.values()
                for outcome in outcomes
            ]
        rows.append(
            ScheduleRow(
                date=game["date"],
                kickoff=game["kickoff"],
                round=game["round"],
                home_team=game["home_team"],
                away_team=game["away_team"],
                reliable=bool(game["reliable"]),
                cells=cells,
                home_goals=game["home_goals"],
                away_goals=game["away_goals"],
            )
        )
    return rows


@dataclass
class ScheduleView:
    rounds: int
    rows: dict[str, list[ScheduleRow]] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)

    @property
    def missing(self) -> bool:
        return not self.rows and not self.problems


def schedule_view(settings: Settings, rounds: int, today: date) -> ScheduleView:
    """The main model's view of each league's coming rounds, from the saved schedules."""
    season = current_season(today)
    view = ScheduleView(rounds=rounds)
    for league in SCHEDULE_FILES:
        path = schedule_path(settings.raw_dir, league, season)
        if not path.exists():
            continue
        try:
            schedule = load_schedule(path, league)
            games = upcoming_games(schedule, today, rounds if rounds != ALL_ROUNDS else None)
            if games.empty:
                view.rows[league] = []
                continue
            history = load_available(linked_leagues(league), [*DEFAULT_SEASONS, season], settings)
            if history.empty:
                view.problems[league] = "no cached results"
                continue
            priced = predict_games(history, games, MODEL_XI[MAIN_MODEL])
        except (ValueError, RuntimeError) as error:
            view.problems[league] = str(error)
            continue
        view.rows[league] = schedule_rows(priced)
    return view
