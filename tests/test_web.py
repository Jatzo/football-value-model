import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flask.testing import FlaskClient
from simulation import add_odds, simulate_league, true_model

from valuemodel.backtest import BacktestResult, run_backtest
from valuemodel.config import Settings
from valuemodel.store import connect, database_path, save_run
from valuemodel.web import create_app, views


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    rng = np.random.default_rng(17)
    model = true_model(10, -0.1, rng)
    seasons = [("2223", "2022-08-01"), ("2324", "2023-08-01")]
    league = pd.concat(
        [simulate_league(model, 3, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    return run_backtest(add_odds(league, model, rng), "E0", ["2324"], Settings(), min_matches=0)


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
    for name in ("Dixon-Coles", "Poisson", "Bet365 pre-match, margin removed"):
        assert name in html
    calibration = embedded_json(html, "calibration-data")
    assert [line["name"] for line in calibration] == ["Dixon-Coles", "Poisson"]
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


def test_fixtures_page_before_any_download(empty_client: FlaskClient) -> None:
    html = empty_client.get("/fixtures").get_data(as_text=True)
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
    assert "not priced, because there are no cached results for them: E2" in html


def test_headline_cards_lead_with_closing_line_value(result: BacktestResult) -> None:
    cards = views.headline_cards(result)
    assert cards[0].label == "Mean closing line value"
    assert cards[0].value.endswith("%")


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
            "rps": [model_rps, 0.22, 0.19],
        }
    )
    assert says in views.scores_verdict(scores)
