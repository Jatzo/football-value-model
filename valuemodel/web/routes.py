"""Dashboard pages."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

from flask import Blueprint, Flask, abort, current_app, render_template, request

from valuemodel.backtest import MAIN_MODEL, BacktestResult
from valuemodel.config import BOOKMAKERS, LEAGUES, MIN_TEAM_MATCHES, Settings, season_label
from valuemodel.labels import (
    FORECASTER_LABELS,
    MARKET_LABELS,
    OUTCOME_LABELS,
    STRATEGY_LABELS,
    decimal_odds,
    label,
    percent,
    share,
    tone,
    units,
)
from valuemodel.report import staking_description
from valuemodel.store import connect, database_path, latest_run_id, list_runs, load_run
from valuemodel.web import views

pages = Blueprint("pages", __name__)


def register_filters(app: Flask) -> None:
    app.jinja_env.filters.update(
        percent=percent,
        signed=lambda value: percent(value, signed=True),
        units=units,
        share=share,
        odds=decimal_odds,
        tone=tone,
        season=season_label,
        strategy=lambda name: label(STRATEGY_LABELS, name),
        outcome=lambda name: label(OUTCOME_LABELS, name),
        market=lambda name: label(MARKET_LABELS, name),
        forecaster=lambda name: label(FORECASTER_LABELS, name),
        league=lambda code: label(LEAGUES, code),
    )


def _settings() -> Settings:
    return current_app.config["SETTINGS"]


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    connection = connect(database_path(_settings()))
    try:
        yield connection
    finally:
        connection.close()


def _selected_run() -> tuple[BacktestResult | None, int | None, list[dict[str, object]]]:
    """The run chosen with ?run=, or the latest one, plus every run for the picker."""
    with _database() as connection:
        runs = list_runs(connection).to_dict("records")
        run_id = request.args.get("run", type=int) or latest_run_id(connection)
        if run_id is None:
            return None, None, runs
        try:
            return load_run(connection, run_id), run_id, runs
        except KeyError:
            abort(404)


def _render_run_page(template: str, **context: object) -> str:
    """Render a page for the selected run. Context values that are functions get the run."""
    result, run_id, runs = _selected_run()
    if result is None:
        return render_template("no_runs.html", runs=runs)
    settings = result.settings
    description = (
        f"Bets at {BOOKMAKERS[settings.bookmaker]} pre-match odds, "
        f"{staking_description(settings)} capped at {share(settings.max_stake)}, "
        f"edge threshold {share(settings.edge_threshold)}"
    )
    return render_template(
        template,
        result=result,
        run_id=run_id,
        runs=runs,
        description=description,
        **{name: build(result) if callable(build) else build for name, build in context.items()},
    )


@pages.route("/")
def summary() -> str:
    return _render_run_page(
        "summary.html",
        cards=views.headline_cards,
        explanation=views.clv_explanation,
        bands=views.probability_band_rows,
        bankroll=views.bankroll_series,
    )


@pages.route("/bets")
def bets() -> str:
    filters = views.BetFilters.from_args(request.args.to_dict())
    return _render_run_page(
        "bets.html",
        filters=filters,
        bet_page=lambda result: views.bet_page(result, filters),
    )


@pages.route("/models")
def models() -> str:
    return _render_run_page(
        "models.html",
        calibration=views.calibration_series,
        verdict=lambda result: views.scores_verdict(result.scores),
    )


@pages.route("/fixtures")
def fixtures() -> str:
    settings = _settings()
    rounds = views.parse_rounds(request.args.get("rounds"))
    return render_template(
        "fixtures.html",
        view=views.fixtures_view(settings),
        schedule=views.schedule_view(settings, rounds, date.today()),
        round_choices=views.ROUND_CHOICES,
        settings=settings,
        bookmaker=BOOKMAKERS[settings.bookmaker],
        min_matches=MIN_TEAM_MATCHES,
        model_name=label(STRATEGY_LABELS, MAIN_MODEL),
        run_id=request.args.get("run", type=int),
    )
