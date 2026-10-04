import numpy as np
import pandas as pd
import pytest
from simulation import add_odds, simulate_league, true_model

from valuemodel.backtest import BacktestResult, run_backtest
from valuemodel.config import Settings
from valuemodel.report import format_report, percent, staking_description, table


def test_table_aligns_columns() -> None:
    text = table(["Name", "Value"], [["a", 1], ["longer", 22]])
    assert text.splitlines() == ["Name    Value", "a           1", "longer     22"]


def test_percent() -> None:
    assert percent(0.0642) == "6.4%"
    assert percent(-0.0642, signed=True) == "-6.4%"
    assert percent(0.05, signed=True) == "+5.0%"
    assert percent(float("nan")) == "n/a"


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (Settings(), "quarter Kelly"),
        (Settings(kelly_fraction=0.5), "0.5 Kelly"),
        (Settings(staking="flat"), "flat stakes of 1% of the starting bankroll"),
    ],
)
def test_staking_description(settings: Settings, expected: str) -> None:
    assert staking_description(settings) == expected


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    rng = np.random.default_rng(9)
    model = true_model(8, -0.1, rng)
    seasons = [("2223", "2022-08-01"), ("2324", "2023-08-01")]
    league = pd.concat(
        [simulate_league(model, 2, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    return run_backtest(add_odds(league, model, rng), "E0", ["2324"], Settings(), min_matches=0)


def test_report_puts_closing_line_value_first(result: BacktestResult) -> None:
    report = format_report(result)
    headings = [
        "Backtest of E0, seasons 2023/24",
        "Closing line value",
        "Betting results",
        "By season",
        "Model quality",
        "Calibration of Dixon-Coles",
    ]
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Bets taken at Bet365 pre-match odds with quarter Kelly" in report


def test_report_for_a_run_without_bets() -> None:
    rng = np.random.default_rng(9)
    model = true_model(8, -0.1, rng)
    seasons = [("2223", "2022-08-01"), ("2324", "2023-08-01")]
    league = pd.concat(
        [simulate_league(model, 1, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    settings = Settings(edge_threshold=100.0)
    empty = run_backtest(add_odds(league, model, rng), "E0", ["2324"], settings, min_matches=0)
    assert (empty.summary["bets"] == 0).all()
    report = format_report(empty)
    assert "Follow the market" in report
    assert "n/a" in report
