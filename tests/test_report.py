import pytest
from simulation import simulated_seasons

from valuemodel.backtest import BacktestResult, run_backtest
from valuemodel.config import Settings
from valuemodel.labels import percent, share, tone, units
from valuemodel.report import format_report, staking_description, table


def test_table_aligns_columns() -> None:
    text = table(["Name", "Value"], [["a", 1], ["longer", 22]])
    assert text.splitlines() == ["Name    Value", "a           1", "longer     22"]


def test_percent() -> None:
    assert percent(0.0642) == "6.4%"
    assert percent(-0.0642, signed=True) == "-6.4%"
    assert percent(0.05, signed=True) == "+5.0%"
    assert percent(float("nan")) == "n/a"
    assert percent(None) == "n/a"


def test_units_and_tone() -> None:
    assert units(-876.7137) == "-876.71"
    assert units(9166.3757) == "9,166.38"
    assert (tone(0.02), tone(-0.02), tone(0.0), tone(float("nan"))) == (
        "positive",
        "negative",
        "",
        "",
    )


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
    league = simulated_seasons(9, 8, 2)
    return run_backtest(league, "E0", ["2324"], Settings(), min_matches=0)


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
    league = simulated_seasons(9, 8, 1)
    settings = Settings(edge_threshold=100.0)
    empty = run_backtest(league, "E0", ["2324"], settings, min_matches=0)
    assert (empty.summary["bets"] == 0).all()
    report = format_report(empty)
    assert "Follow the market" in report
    assert "n/a" in report


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.03, "3%"), (0.035, "3.5%"), (0.02, "2%"), (0.025, "2.5%"), (0.07, "7%")],
)
def test_share_keeps_meaningful_decimals(value: float, expected: str) -> None:
    assert share(value) == expected


def test_report_shows_a_fractional_threshold(result: BacktestResult) -> None:
    fractional = BacktestResult(**{**result.__dict__, "settings": Settings(edge_threshold=0.035)})
    assert "Edge threshold 3.5%" in format_report(fractional)
