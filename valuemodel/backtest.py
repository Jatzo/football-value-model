"""Walk-forward backtest with paper stakes, closing line value and model scores.

Bets are struck at the configured bookmaker's pre-match odds. Pinnacle's
closing odds are only ever used as a benchmark: they measure whether the prices
taken were better than where the market finished, and they score the market as
a forecaster. They never decide a bet or its stake.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from valuemodel.config import (
    BOOKMAKERS,
    MIN_TEAM_MATCHES,
    MODEL_XI,
    Settings,
    linked_leagues,
)
from valuemodel.data import odds_columns
from valuemodel.markets import OUTCOMES
from valuemodel.models.dixon_coles import fit_dixon_coles
from valuemodel.models.poisson import fit_poisson
from valuemodel.models.shots_adjusted import fit_shots_adjusted
from valuemodel.odds import MARKETS, find_value, remove_margin
from valuemodel.scoring import brier_score, log_loss, outcome_indices, ranked_probability_score
from valuemodel.staking import stake
from valuemodel.walkforward import (
    FitFunction,
    walk_forward_forecasts,
)

MODELS: dict[str, FitFunction] = {
    "shots-adjusted": fit_shots_adjusted,
    "dixon-coles": fit_dixon_coles,
    "poisson": fit_poisson,
}

# Treating Pinnacle's margin-free pre-match prices as the forecast gives the
# "simply follow the market" strategy: bet wherever the bookmaker is more
# generous than the sharpest price available at the same time.
MARKET_STRATEGY = "market"

# The model the report, dashboard and fixtures lead with. It forecast best on
# the tuning seasons. Dixon-Coles and the Poisson baseline are kept for comparison.
MAIN_MODEL = "shots-adjusted"

CALIBRATION_BINS = 10

# Edges of the bands used to group bets by the model's chance of them winning.
PROBABILITY_BANDS: tuple[float, ...] = (0.0, 0.3, 0.45, 0.6, 1.0)
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


def probability_bands(bets: pd.DataFrame) -> pd.DataFrame:
    """How bets fared grouped by the model's chance of them winning.

    This answers whether the bets the model was surest about did better. Value
    bets are chosen where the model disagrees with the market, so the model's
    confidence on them is not the same as its confidence on matches in general.
    """
    rows = []
    for low, high in pairwise(PROBABILITY_BANDS):
        in_band = bets[(bets["probability"] > low) & (bets["probability"] <= high)]
        if in_band.empty:
            continue
        level_returns = np.where(in_band["won"], in_band["odds"] - 1, -1.0)
        rows.append(
            {
                "band_low": low,
                "band_high": high,
                "bets": len(in_band),
                "mean_probability": float(in_band["probability"].mean()),
                "win_rate": float(in_band["won"].mean()),
                "mean_odds": float(in_band["odds"].mean()),
                "level_roi": float(level_returns.mean()),
            }
        )
    return pd.DataFrame(rows)


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
        scored = len(common) > 0
        rows.append(
            {
                "forecaster": name,
                "matches": len(common),
                "log_loss": log_loss(probabilities, outcomes) if scored else np.nan,
                "rps": ranked_probability_score(probabilities, outcomes) if scored else np.nan,
                "brier": brier_score(probabilities, outcomes) if scored else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(forecasts: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """How often outcomes forecast at each probability actually happened.

    Home, draw and away forecasts are pooled, so a well calibrated model has
    each bin's observed frequency close to its mean forecast.
    """
    priced = forecasts[forecasts[list(MARKETS["1x2"])].notna().all(axis=1)]
    results = matches.loc[priced.index, "result"]
    probabilities = priced[list(MARKETS["1x2"])].to_numpy(dtype=float).ravel()
    happened = np.column_stack([results == code for code in ("H", "D", "A")]).ravel()
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    which = np.clip(np.digitize(probabilities, edges) - 1, 0, CALIBRATION_BINS - 1)
    rows = []
    for b in range(CALIBRATION_BINS):
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
    # Not stored with saved runs, so empty for any run loaded from the database.
    history_leagues: tuple[str, ...] = ()


def _summarise(bets: pd.DataFrame, starting_bankroll: float) -> dict[str, float]:
    return {
        **betting_summary(bets, starting_bankroll),
        **clv_summary(bets["clv"] if "clv" in bets else pd.Series(dtype=float), len(bets)),
    }


def _season_rows(
    strategy: str, bets: pd.DataFrame, seasons: Iterable[str]
) -> list[dict[str, object]]:
    rows = []
    for season in seasons:
        in_season = bets[bets["season"] == season] if len(bets) else bets
        rows.append(
            {
                "strategy": strategy,
                "season": season,
                "bets": len(in_season),
                "staked": float(in_season["stake"].sum()) if len(in_season) else 0.0,
                "profit": float(in_season["profit"].sum()) if len(in_season) else 0.0,
                "mean_clv": float(in_season["clv"].mean()) if len(in_season) else np.nan,
            }
        )
    return rows


def main_strategy(result: "BacktestResult") -> str:
    """The main model, or the first strategy for runs saved before it existed."""
    if MAIN_MODEL in result.bets or result.bets == {}:
        return MAIN_MODEL
    return next(iter(result.bets))


def run_backtest(
    matches: pd.DataFrame,
    league: str,
    seasons: Iterable[str],
    settings: Settings,
    xi: float | None = None,
    min_matches: int = MIN_TEAM_MATCHES,
    linked: bool = True,
) -> BacktestResult:
    """Forecast, bet and score every match of the given seasons in one league.

    Each model uses its own tuned time decay unless xi is given for all of them.
    Models are fitted on the league's linked divisions too, unless linked is False.
    """
    seasons = tuple(seasons)
    sources = linked_leagues(league, linked)
    in_league = matches["league"] == league
    targets = matches[in_league & matches["season"].isin(seasons)]
    model_xi = {name: MODEL_XI[name] if xi is None else xi for name in MODELS}
    result = BacktestResult(
        league=league,
        seasons=seasons,
        xi=model_xi[MAIN_MODEL],
        settings=settings,
        history_leagues=sources,
    )

    forecasts = {
        name: walk_forward_forecasts(
            matches, seasons, fit, model_xi[name], min_matches, history_leagues={league: sources}
        )
        for name, fit in MODELS.items()
    }
    # Following the market means betting where the bookmaker beats Pinnacle, which
    # is meaningless when the bookmaker is Pinnacle itself.
    if settings.bookmaker != MARKET_SOURCE:
        forecasts[MARKET_STRATEGY] = market_forecasts(
            targets, MARKET_SOURCE, settings.margin_method
        )

    summaries, season_rows = [], []
    for name, frame in forecasts.items():
        bets = place_bets(frame, targets, settings)
        if not bets.empty:
            bets = bets.join(closing_line_value(bets, targets, settings.margin_method))
        result.bets[name] = bets
        summaries.append({"strategy": name, **_summarise(bets, settings.starting_bankroll)})
        season_rows += _season_rows(name, bets, seasons)
    result.summary = pd.DataFrame(summaries)
    result.season_summary = pd.DataFrame(season_rows)

    pre_match = f"{BOOKMAKERS[settings.bookmaker].lower()} pre-match"
    scored = {
        **{name: forecasts[name] for name in MODELS},
        pre_match: market_forecasts(targets, settings.bookmaker, settings.margin_method),
        "pinnacle closing": market_forecasts(targets, CLOSING_SOURCE, settings.margin_method),
    }
    result.scores = model_scores(scored, targets)
    result.calibration = {name: calibration_table(forecasts[name], targets) for name in MODELS}
    return result
