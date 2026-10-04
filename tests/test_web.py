import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flask.testing import FlaskClient
from simulation import simulate_league, simulated_seasons, true_model

from valuemodel.backtest import BacktestResult, run_backtest
from valuemodel.config import Settings
from valuemodel.labels import NO_COMMON_MATCHES
from valuemodel.report import format_report
from valuemodel.store import connect, database_path, save_run
from valuemodel.web import create_app, views


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    league = simulated_seasons(17, 10, 3)
    return run_backtest(league, "E0", ["2324"], Settings(), min_matches=0)


@pytest.fixture(scope="module")
def result_without_bets() -> BacktestResult:
    league = simulated_seasons(17, 8, 1)
    settings = Settings(edge_threshold=100.0)
    return run_backtest(league, "E0", ["2324"], settings, min_matches=0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


@pytest.fixture
def client(settings: Settings, result: BacktestResult) -> FlaskClient:
    connection = connect(database_path(settings))
    save_run(connection, result)
    connection.close()
    return create_app(settings).test_client()


@pytest.fixture
def empty_client(settings: Settings) -> FlaskClient:
    return create_app(settings).test_client()


def embedded_json(html: str, element_id: str) -> object:
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S
    )
    assert match, f"no {element_id} on the page"
    return json.loads(match.group(1))


def test_summary_page(client: FlaskClient) -> None:
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert html.index("Mean closing line value") < html.index("ROI")
    assert "Bets at Bet365 pre-match odds, quarter Kelly capped at 2%" in html
    assert "BeGambleAware" in html
    series = embedded_json(html, "bankroll-data")
    assert [line["name"] for line in series][:2] == ["Dixon-Coles", "Poisson"]


def test_models_page(client: FlaskClient) -> None:
    response = client.get("/models")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    for name in ("Dixon-Coles", "Poisson", "Shots-adjusted", "Bet365 pre-match, margin removed"):
        assert name in html
    calibration = embedded_json(html, "calibration-data")
    assert [line["name"] for line in calibration] == ["Dixon-Coles", "Poisson", "Shots-adjusted"]
    assert all("count" in point for point in calibration[0]["points"])


def test_bets_page_with_filters(client: FlaskClient, result: BacktestResult) -> None:
    response = client.get("/bets?strategy=dixon-coles&market=totals&result=won")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    bets = result.bets["dixon-coles"]
    expected = int(((bets["market"] == "totals") & bets["won"]).sum())
    assert f"<strong>{expected:,}</strong> bets" in html
    assert "Home win" not in html.split("<tbody>")[1]


def test_bets_page_is_paginated(client: FlaskClient, result: BacktestResult) -> None:
    total = len(result.bets["dixon-coles"])
    assert total > views.PER_PAGE
    first = client.get("/bets").get_data(as_text=True)
    assert "Page 1 of" in first
    assert "Older" in first
    last = client.get("/bets?page=999").get_data(as_text=True)
    assert "Newer" in last
    assert client.get("/bets?page=abc").status_code == 200


def test_run_picker(client: FlaskClient) -> None:
    assert client.get("/?run=1").status_code == 200
    assert client.get("/?run=99").status_code == 404
    assert 'option value="1"' in client.get("/").get_data(as_text=True)


@pytest.mark.parametrize("path", ["/", "/bets", "/models"])
def test_pages_without_a_run(empty_client: FlaskClient, path: str) -> None:
    response = empty_client.get(path)
    assert response.status_code == 200
    assert "No backtest has been run yet" in response.get_data(as_text=True)


def test_fixtures_page_before_any_download(empty_client: FlaskClient, settings: Settings) -> None:
    html = empty_client.get("/fixtures").get_data(as_text=True)
    assert not database_path(settings).exists()
    assert "No fixtures downloaded yet" in html
    assert "valuemodel fixtures" in html


def test_fixtures_page_without_history(
    empty_client: FlaskClient, settings: Settings, fixtures_dir: Path
) -> None:
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "fixtures.csv").write_bytes((fixtures_dir / "upcoming.csv").read_bytes())
    html = empty_client.get("/fixtures").get_data(as_text=True)
    assert "No fixtures for the leagues with cached results" in html
    for league in ("E2", "E3", "EC"):
        assert league in html


def raw_season(frame: pd.DataFrame) -> str:
    """Write simulated matches in the data source's own CSV layout."""
    raw = pd.DataFrame(
        {
            "Div": frame["league"],
            "Date": frame["date"].dt.strftime("%d/%m/%Y"),
            "HomeTeam": frame["home_team"],
            "AwayTeam": frame["away_team"],
            "FTHG": frame["home_goals"],
            "FTAG": frame["away_goals"],
            "FTR": frame["result"],
        }
    )
    return raw.to_csv(index=False)


def test_fixtures_page_prices_known_leagues(empty_client: FlaskClient, settings: Settings) -> None:
    rng = np.random.default_rng(3)
    model = true_model(8, -0.1, rng)
    history = simulate_league(model, 1, rng, start="2025-08-01", season="2526")
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "E0_2526.csv").write_text(raw_season(history), encoding="utf-8")
    fixtures = (
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,B365>2.5,B365<2.5\n"
        "E0,10/10/2026,15:00,Team 00,Team 01,40.0,3.5,1.9,1.9,1.9\n"
        "E0,10/10/2026,17:30,Newcomers,Team 02,2.0,3.5,3.5,1.9,1.9\n"
        "E2,10/10/2026,15:00,Burton,Reading,2.0,3.4,3.6,1.9,1.9\n"
    )
    (settings.raw_dir / "fixtures.csv").write_text(fixtures, encoding="utf-8")

    html = empty_client.get("/fixtures").get_data(as_text=True)
    assert "Premier League" in html
    assert "Team 00 v Team 01" in html
    assert html.count('class="num price value"') >= 1
    assert "Not priced: a team has fewer than 10 matches" in html
    assert "E2 (no cached results)" in html


def test_headline_cards_lead_with_closing_line_value(result: BacktestResult) -> None:
    cards = views.headline_cards(result)
    assert cards[0].label == "Mean closing line value"
    mean_clv = result.summary.set_index("strategy").loc["dixon-coles", "mean_clv"]
    assert cards[0].value == f"{mean_clv:+.1%}"


def test_run_without_closing_odds(settings: Settings) -> None:
    league = simulated_seasons(23, 8, 2)
    closing = [column for column in league if column.startswith("pinnacle_close_")]
    league[closing] = float("nan")
    result = run_backtest(league, "E0", ["2324"], Settings(), min_matches=0)
    assert result.scores["matches"].eq(0).all()

    cards = views.headline_cards(result)
    assert cards[0].value == "n/a"
    assert not any("nan" in card.value + card.note for card in cards)
    assert views.scores_verdict(result.scores) == NO_COMMON_MATCHES
    assert NO_COMMON_MATCHES in format_report(result)

    connection = connect(database_path(settings))
    save_run(connection, result)
    connection.close()
    html = create_app(settings).test_client().get("/models").get_data(as_text=True)
    assert "could not be compared" in html
    assert "nan" not in html.split("<main")[1].split("calibration-data")[0]


def test_bankroll_lines_start_at_the_bankroll_and_end_together(result: BacktestResult) -> None:
    series = views.bankroll_series(result)
    assert all(line["points"][0]["y"] == 1000 for line in series)
    assert len({line["points"][-1]["x"] for line in series}) == 1


def test_filter_bets(result: BacktestResult) -> None:
    bets = result.bets["dixon-coles"]
    lost = views.filter_bets(bets, views.BetFilters(result="lost", market="1x2"))
    assert len(lost) == int(((bets["market"] == "1x2") & ~bets["won"]).sum())
    assert not lost["won"].any()


def test_bet_page_totals(result: BacktestResult) -> None:
    page = views.bet_page(result, views.BetFilters())
    bets = result.bets["dixon-coles"]
    assert page.totals["bets"] == len(bets)
    assert page.totals["profit"] == pytest.approx(bets["profit"].sum())
    assert len(page.rows) == views.PER_PAGE
    assert page.rows.iloc[0]["date"] == bets["date"].max()


@pytest.mark.parametrize(
    ("model_rps", "says"), [(0.21, "better forecasts than either model"), (0.18, "unusual")]
)
def test_scores_verdict(model_rps: float, says: str) -> None:
    scores = pd.DataFrame(
        {
            "forecaster": ["dixon-coles", "poisson", "pinnacle closing"],
            "matches": [100, 100, 100],
            "rps": [model_rps, 0.22, 0.19],
        }
    )
    assert says in views.scores_verdict(scores)


@pytest.mark.parametrize("path", ["/", "/bets", "/models"])
def test_pages_for_a_run_without_bets(
    settings: Settings, result_without_bets: BacktestResult, path: str
) -> None:
    connection = connect(database_path(settings))
    save_run(connection, result_without_bets)
    connection.close()
    response = create_app(settings).test_client().get(path)
    assert response.status_code == 200
    if path == "/":
        html = response.get_data(as_text=True)
        assert "Bets placed" in html
        assert "No bets were placed" in html
        assert embedded_json(html, "bankroll-data") == []


def test_clv_explanation_follows_the_sign(result: BacktestResult) -> None:
    negative = result.summary.assign(mean_clv=-0.05)
    positive = result.summary.assign(mean_clv=0.02)
    assert "negative" in views.clv_explanation(
        BacktestResult(**{**result.__dict__, "summary": negative})
    )
    assert "positive" in views.clv_explanation(
        BacktestResult(**{**result.__dict__, "summary": positive})
    )


def test_bets_can_be_ordered_by_model_chance(client: FlaskClient, result: BacktestResult) -> None:
    page = views.bet_page(result, views.BetFilters(sort="probability"))
    probabilities = list(page.rows["probability"])
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] == result.bets["dixon-coles"]["probability"].max()
    html = client.get("/bets?sort=probability").get_data(as_text=True)
    assert 'value="probability" selected' in html
    assert views.BetFilters.from_args({"sort": "nonsense"}).sort == "newest"


def test_summary_shows_bets_by_model_chance(client: FlaskClient, result: BacktestResult) -> None:
    rows = views.probability_band_rows(result)
    assert {row["strategy"] for row in rows} <= set(result.bets)
    assert sum(row["bets"] for row in rows if row["strategy"] == "dixon-coles") == len(
        result.bets["dixon-coles"]
    )
    assert "Bets by the model's chance of winning" in client.get("/").get_data(as_text=True)


def test_most_likely_outcomes_are_sorted_and_marked() -> None:
    priced = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-10"] * 3),
            "kickoff": ["15:00"] * 3,
            "league": ["E0"] * 3,
            "home_team": ["A", "C", "E"],
            "away_team": ["B", "D", "F"],
            "reliable": [True, True, False],
            "home": [0.30, 0.70, np.nan],
            "draw": [0.25, 0.20, np.nan],
            "away": [0.45, 0.10, np.nan],
            "over25": [0.5, 0.6, np.nan],
            "under25": [0.5, 0.4, np.nan],
            **{f"odds_{o}": [2.0] * 3 for o in ("home", "draw", "away", "over25", "under25")},
            **{f"edge_{o}": [0.0] * 3 for o in ("home", "draw", "away", "over25", "under25")},
            "value_1x2": [None] * 3,
            "value_totals": [None] * 3,
        }
    )
    likely = views.likely_outcomes(priced)
    assert [(item.home_team, item.outcome) for item in likely] == [("C", "home"), ("A", "away")]
    first_row = views.fixture_rows(priced, "E0")[0]
    assert [cell.likely for cell in first_row.cells] == [False, False, True, False, False]
