"""Plain text report of a backtest, closing line value first."""

from collections.abc import Sequence

import pandas as pd

from valuemodel.backtest import MAIN_MODEL, MODELS, BacktestResult, probability_bands
from valuemodel.config import BOOKMAKERS, Settings, season_label
from valuemodel.labels import (
    FORECASTER_LABELS,
    NO_COMMON_MATCHES,
    STRATEGY_LABELS,
    label,
    percent,
    share,
    units,
)

Cell = str | int | float


def table(headers: Sequence[str], rows: Sequence[Sequence[Cell]]) -> str:
    """Align columns: the first to the left, the rest to the right."""
    text = [[str(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in text) for i in range(len(headers))]
    lines = []
    for row in text:
        cells = [row[0].ljust(widths[0])] + [
            cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def staking_description(settings: Settings) -> str:
    if settings.staking == "flat":
        return f"flat stakes of {share(settings.flat_stake_share)} of the starting bankroll"
    fraction = "quarter" if settings.kelly_fraction == 0.25 else f"{settings.kelly_fraction:g}"
    return f"{fraction} Kelly"


def _header(result: BacktestResult) -> str:
    settings = result.settings
    seasons = ", ".join(season_label(season) for season in result.seasons)
    cap, threshold = share(settings.max_stake), share(settings.edge_threshold)
    return (
        f"Backtest of {result.league}, seasons {seasons}\n"
        f"Bets taken at {BOOKMAKERS[settings.bookmaker]} pre-match odds with "
        f"{staking_description(settings)}, capped at {cap} of the bankroll.\n"
        f"Edge threshold {threshold}, {settings.margin_method} margin removal, "
        f"xi {result.xi}, starting bankroll {settings.starting_bankroll:g} units."
    )


def _clv_section(summary: pd.DataFrame) -> str:
    rows = [
        [
            label(STRATEGY_LABELS, row.strategy),
            int(row.bets),
            f"{int(row.clv_bets)} ({percent(row.clv_coverage)})" if row.bets else "0",
            percent(row.mean_clv, signed=True),
            percent(row.beat_close_share),
        ]
        for row in summary.itertuples()
    ]
    return (
        "Closing line value against Pinnacle's closing odds, margin removed\n"
        + table(["Strategy", "Bets", "With closing odds", "Mean CLV", "Beat the close"], rows)
        + "\nPositive CLV means the prices taken were better than where the market closed. "
        "It is the most\nreliable sign of a real edge, because it is far less noisy than profit."
    )


def _betting_section(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples():
        if not row.bets:
            rows.append([label(STRATEGY_LABELS, row.strategy), 0, "", "", "", "", "", "", ""])
            continue
        interval = (
            f"{percent(row.level_roi, True)} "
            f"({percent(row.level_roi_low, True)} to {percent(row.level_roi_high, True)})"
        )
        rows.append(
            [
                label(STRATEGY_LABELS, row.strategy),
                int(row.bets),
                units(row.staked),
                units(row.profit),
                percent(row.roi, signed=True),
                units(row.final_bankroll),
                f"{units(row.max_drawdown)} ({percent(row.max_drawdown_share)})",
                int(row.longest_losing_run),
                interval,
            ]
        )
    headers = [
        "Strategy",
        "Bets",
        "Staked",
        "Profit",
        "ROI",
        "Final bankroll",
        "Max drawdown",
        "Losing run",
        "Level-stakes ROI (95% interval)",
    ]
    return "Betting results\n" + table(headers, rows)


def _confidence_section(result: BacktestResult) -> str:
    rows = []
    for strategy in MODELS:
        bets = result.bets.get(strategy)
        if bets is None or bets.empty:
            continue
        for band in probability_bands(bets).itertuples():
            rows.append(
                [
                    label(STRATEGY_LABELS, strategy),
                    f"{band.band_low:.0%} to {band.band_high:.0%}",
                    band.bets,
                    percent(band.mean_probability),
                    percent(band.win_rate),
                    f"{band.mean_odds:.2f}",
                    percent(band.level_roi, signed=True),
                ]
            )
    headers = ["Strategy", "Model chance", "Bets", "Average", "Won", "Mean odds", "Level ROI"]
    note = (
        "Bets the model was surest about win most often, but at short odds. A band only makes\n"
        "money if its bets win more often than the odds imply, so compare Won with Average."
    )
    return "\n".join(["Bets by the model's chance of winning", table(headers, rows), note])


def _season_section(season_summary: pd.DataFrame) -> str:
    rows = [
        [
            label(STRATEGY_LABELS, row.strategy),
            season_label(row.season),
            int(row.bets),
            units(row.staked),
            units(row.profit),
            percent(row.mean_clv, signed=True),
        ]
        for row in season_summary.itertuples()
    ]
    return "By season\n" + table(
        ["Strategy", "Season", "Bets", "Staked", "Profit", "Mean CLV"], rows
    )


def _scores_section(scores: pd.DataFrame) -> str:
    matches = int(scores["matches"].iloc[0]) if len(scores) else 0
    if not matches:
        return "\n".join(["Model quality", NO_COMMON_MATCHES])
    rows = [
        [
            label(FORECASTER_LABELS, row.forecaster),
            f"{row.log_loss:.4f}",
            f"{row.rps:.4f}",
            f"{row.brier:.4f}",
        ]
        for row in scores.itertuples()
    ]
    return (
        f"Model quality on the {matches} matches every forecaster priced (lower is better)\n"
        + table(["Forecaster", "Log loss", "RPS", "Brier"], rows)
    )


def _calibration_section(model: str, calibration: pd.DataFrame) -> str:
    rows = [
        [
            f"{row.bin_low:.0%} to {row.bin_high:.0%}",
            percent(row.mean_forecast),
            percent(row.observed),
            int(row.count),
        ]
        for row in calibration.itertuples()
    ]
    return (
        f"Calibration of {label(STRATEGY_LABELS, model)}, home, draw and away forecasts pooled\n"
        + table(["Forecast range", "Mean forecast", "Happened", "Count"], rows)
    )


def format_report(result: BacktestResult) -> str:
    sections = [
        _header(result),
        _clv_section(result.summary),
        _betting_section(result.summary),
        _confidence_section(result),
        _season_section(result.season_summary),
        _scores_section(result.scores),
    ]
    if MAIN_MODEL in result.calibration:
        sections.append(_calibration_section(MAIN_MODEL, result.calibration[MAIN_MODEL]))
    return "\n\n".join(sections)
