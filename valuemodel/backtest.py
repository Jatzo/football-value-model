"""Walk-forward backtest with paper stakes, closing line value and model scores.

Bets are struck at the configured bookmaker's pre-match odds. Pinnacle's
closing odds are only ever used as a benchmark: they measure whether the prices
taken were better than where the market finished, and they score the market as
a forecaster. They never decide a bet or its stake.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from valuemodel.config import DEFAULT_XI, MIN_TEAM_MATCHES, TRAINING_WINDOW_DAYS, Settings
from valuemodel.markets import OUTCOMES
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.models.poisson import fit_poisson
from valuemodel.odds import MARKETS, find_value, remove_margin
from valuemodel.scoring import brier_score, log_loss, outcome_indices, ranked_probability_score
from valuemodel.staking import stake
from valuemodel.walkforward import (
    FitFunction,
    odds_capture_date,
    walk_forward_forecasts,
)

MODELS: dict[str, FitFunction] = {"dixon-coles": fit_dixon_coles, "poisson": fit_poisson}

# Treating Pinnacle's margin-free pre-match prices as the forecast gives the
# "simply follow the market" strategy: bet wherever the bookmaker is more
# generous than the sharpest price available at the same time.
MARKET_STRATEGY = "market"
MARKET_SOURCE = "pinnacle"
CLOSING_SOURCE = "pinnacle_close"

BET_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "market",
    "outcome",
    "probability",
    "odds",
    "edge",
    "stake",
    "won",
    "profit",
    "bankroll",
)

_BOOTSTRAP_SAMPLES = 2000


def odds_columns(source: str) -> list[str]:
    return [f"{source}_{outcome}" for outcome in OUTCOMES]


def market_forecasts(matches: pd.DataFrame, source: str, method: str) -> pd.DataFrame:
    """Margin-free probabilities from one source's odds, shaped like model forecasts."""
    forecasts = pd.DataFrame(index=matches.index, columns=list(OUTCOMES), dtype=float)
    for outcomes in MARKETS.values():
        odds = matches[[f"{source}_{outcome}" for outcome in outcomes]].to_numpy()
        forecasts[list(outcomes)] = remove_margin(odds, method)
    forecasts["reliable"] = forecasts[list(MARKETS["1x2"])].notna().all(axis=1)
    return forecasts


def settle(outcome: pd.Series, home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """Whether each bet won. Over 2.5 needs at least three goals in total."""
    total = home_goals + away_goals
    wins = {
        "home": home_goals > away_goals,
        "draw": home_goals == away_goals,
        "away": home_goals < away_goals,
        "over25": total >= 3,
        "under25": total <= 2,
    }
    won = pd.Series(False, index=outcome.index)
    for name, mask in wins.items():
        won |= (outcome == name) & mask.astype(bool)
    return won


def place_bets(forecasts: pd.DataFrame, matches: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Find value at the configured bookmaker's prices and stake it day by day.

    Every bet on a given day is sized from the bankroll at the start of that day,
    since in reality none of them would be settled before the others are placed.
    """
    priced = forecasts[forecasts["reliable"]]
    odds = matches.loc[priced.index, odds_columns(settings.bookmaker)]
    odds.columns = list(OUTCOMES)
    candidates = find_value(priced[list(OUTCOMES)], odds, settings.edge_threshold)
    if candidates.empty:
        return pd.DataFrame(columns=list(BET_COLUMNS))

    match_info = ["date", "kickoff", "league", "season", "home_team", "away_team"]
    bets = candidates.join(matches[[*match_info, "home_goals", "away_goals"]])
    bets["won"] = settle(bets["outcome"], bets["home_goals"], bets["away_goals"])
    bets = bets.rename_axis("match_id").reset_index()
    bets = bets.sort_values(["date", "kickoff", "home_team", "market"], na_position="first")
    bets = bets.reset_index(drop=True)

    bankroll = settings.starting_bankroll
    stakes, bankrolls = [], []
    for _, day in bets.groupby("date", sort=True):
        opening = bankroll
        for bet in day.itertuples():
            amount = stake(bet.probability, bet.odds, opening, settings)
            bankroll += amount * (bet.odds - 1) if bet.won else -amount
            stakes.append(amount)
            bankrolls.append(bankroll)
    bets["stake"] = stakes
    bets["profit"] = np.where(bets["won"], bets["stake"] * (bets["odds"] - 1), -bets["stake"])
    bets["bankroll"] = bankrolls
    bets = bets[bets["stake"] > 0]
    return bets[list(BET_COLUMNS)]


def max_drawdown(bankroll: np.ndarray) -> tuple[float, float]:
    """Largest fall from a peak, in units and as a share of that peak."""
    if len(bankroll) == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(bankroll)
    falls = peaks - bankroll
    worst = int(np.argmax(falls))
    share = falls[worst] / peaks[worst] if peaks[worst] else 0.0
    return float(falls[worst]), float(share)


def longest_losing_run(won: Iterable[bool]) -> int:
    longest = current = 0
    for result in won:
        current = 0 if result else current + 1
        longest = max(longest, current)
    return longest


def betting_summary(bets: pd.DataFrame, starting_bankroll: float) -> dict[str, float | int]:
    """Headline betting figures, plus level-stakes ROI with a bootstrap interval.

    Level stakes put one unit on every bet, so the ROI does not depend on the
    order of results the way a compounding Kelly bankroll does. The interval
    shows how much of the result could be luck.
    """
    if bets.empty:
        return {
            "bets": 0,
            "win_rate": np.nan,
            "mean_odds": np.nan,
            "staked": 0.0,
            "profit": 0.0,
            "roi": np.nan,
            "final_bankroll": starting_bankroll,
            "max_drawdown": 0.0,
            "max_drawdown_share": 0.0,
            "longest_losing_run": 0,
            "level_roi": np.nan,
            "level_roi_low": np.nan,
            "level_roi_high": np.nan,
        }
    curve = np.concatenate([[starting_bankroll], bets["bankroll"].to_numpy()])
    drawdown, drawdown_share = max_drawdown(curve)
    level_returns = np.where(bets["won"], bets["odds"] - 1, -1.0)
    resampled = np.random.default_rng(0).choice(
        level_returns, size=(_BOOTSTRAP_SAMPLES, len(level_returns))
    )
    low, high = np.percentile(resampled.mean(axis=1), [2.5, 97.5])
    staked = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    return {
        "bets": len(bets),
        "win_rate": float(bets["won"].mean()),
        "mean_odds": float(bets["odds"].mean()),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else np.nan,
        "final_bankroll": float(curve[-1]),
        "max_drawdown": drawdown,
        "max_drawdown_share": drawdown_share,
        "longest_losing_run": longest_losing_run(bets["won"]),
        "level_roi": float(level_returns.mean()),
        "level_roi_low": float(low),
        "level_roi_high": float(high),
    }


def closing_line_value(bets: pd.DataFrame, matches: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compare each price taken with Pinnacle's closing price after its margin is removed.

    CLV is odds taken divided by fair closing odds, minus 1. Beating the close
    on average is the clearest sign of a real edge, because results are noisy
    but closing prices are the market's best estimate.
    """
    closing = market_forecasts(matches.loc[bets["match_id"].unique()], CLOSING_SOURCE, method)
    fair_probability = [
        closing.at[match_id, outcome]
        for match_id, outcome in zip(bets["match_id"], bets["outcome"], strict=True)
    ]
    fair_odds = 1 / np.asarray(fair_probability, dtype=float)
    return pd.DataFrame(
        {"fair_closing_odds": fair_odds, "clv": bets["odds"].to_numpy() / fair_odds - 1},
        index=bets.index,
    )


def clv_summary(clv: pd.Series, total_bets: int) -> dict[str, float]:
    measured = clv.dropna()
    return {
        "clv_bets": len(measured),
        "clv_coverage": len(measured) / total_bets if total_bets else np.nan,
        "mean_clv": float(measured.mean()) if len(measured) else np.nan,
        "beat_close_share": float((measured > 0).mean()) if len(measured) else np.nan,
    }


def model_scores(forecasts: dict[str, pd.DataFrame], matches: pd.DataFrame) -> pd.DataFrame:
    """Score every forecaster on the matches that all of them priced."""
    outcomes_1x2 = list(MARKETS["1x2"])
    common = matches.index
    for frame in forecasts.values():
        common = common.intersection(frame.index[frame[outcomes_1x2].notna().all(axis=1)])
    outcomes = outcome_indices(matches.loc[common, "result"])
    rows = []
    for name, frame in forecasts.items():
        probabilities = frame.loc[common, outcomes_1x2].to_numpy(dtype=float)
        rows.append(
            {
                "forecaster": name,
                "matches": len(common),
                "log_loss": log_loss(probabilities, outcomes),
                "rps": ranked_probability_score(probabilities, outcomes),
                "brier": brier_score(probabilities, outcomes),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    forecasts: pd.DataFrame, matches: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
    """How often outcomes forecast at each probability actually happened.

    Home, draw and away forecasts are pooled, so a well calibrated model has
    each bin's observed frequency close to its mean forecast.
    """
    priced = forecasts[forecasts[list(MARKETS["1x2"])].notna().all(axis=1)]
    results = matches.loc[priced.index, "result"]
    probabilities = priced[list(MARKETS["1x2"])].to_numpy(dtype=float).ravel()
    happened = np.column_stack([results == code for code in ("H", "D", "A")]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            rows.append(
                {
                    "bin_low": edges[b],
                    "bin_high": edges[b + 1],
                    "mean_forecast": float(probabilities[in_bin].mean()),
                    "observed": float(happened[in_bin].mean()),
                    "count": int(in_bin.sum()),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class BacktestResult:
    league: str
    seasons: tuple[str, ...]
    xi: float
    settings: Settings
    bets: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    season_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: dict[str, pd.DataFrame] = field(default_factory=dict)


def _summarise(bets: pd.DataFrame, starting_bankroll: float) -> dict[str, float]:
    return {
        **betting_summary(bets, starting_bankroll),
        **clv_summary(bets["clv"] if "clv" in bets else pd.Series(dtype=float), len(bets)),
    }


def run_backtest(
    matches: pd.DataFrame,
    league: str,
    seasons: Iterable[str],
    settings: Settings,
    xi: float = DEFAULT_XI,
    min_matches: int = MIN_TEAM_MATCHES,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> BacktestResult:
    """Forecast, bet and score every match of the given seasons in one league."""
    seasons = tuple(seasons)
    history = matches[matches["league"] == league]
    targets = history[history["season"].isin(seasons)]
    result = BacktestResult(league=league, seasons=seasons, xi=xi, settings=settings)

    forecasts = {
        name: walk_forward_forecasts(
            history, seasons, fit, xi, odds_capture_date, min_matches, window_days
        )
        for name, fit in MODELS.items()
    }
    forecasts[MARKET_STRATEGY] = market_forecasts(targets, MARKET_SOURCE, settings.margin_method)

    summaries, season_rows = [], []
    for name, frame in forecasts.items():
        bets = place_bets(frame, targets, settings)
        if not bets.empty:
            bets = bets.join(closing_line_value(bets, targets, settings.margin_method))
        result.bets[name] = bets
        summaries.append({"strategy": name, **_summarise(bets, settings.starting_bankroll)})
        for season in seasons:
            in_season = bets[bets["season"] == season] if not bets.empty else bets
            season_rows.append(
                {
                    "strategy": name,
                    "season": season,
                    "bets": len(in_season),
                    "staked": float(in_season["stake"].sum()) if len(in_season) else 0.0,
                    "profit": float(in_season["profit"].sum()) if len(in_season) else 0.0,
                    "mean_clv": float(in_season["clv"].mean()) if len(in_season) else np.nan,
                }
            )
    result.summary = pd.DataFrame(summaries)
    result.season_summary = pd.DataFrame(season_rows)

    scored = {
        "dixon-coles": forecasts["dixon-coles"],
        "poisson": forecasts["poisson"],
        "bet365 pre-match": market_forecasts(targets, "b365", settings.margin_method),
        "pinnacle closing": market_forecasts(targets, CLOSING_SOURCE, settings.margin_method),
    }
    result.scores = model_scores(scored, targets)
    result.calibration = {name: calibration_table(forecasts[name], targets) for name in MODELS}
    return result
