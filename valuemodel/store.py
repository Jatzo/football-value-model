"""Save backtest runs to SQLite and load them back for the report and dashboard."""

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from valuemodel.backtest import BacktestResult
from valuemodel.config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    league TEXT NOT NULL,
    seasons TEXT NOT NULL,
    xi REAL NOT NULL,
    settings TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bets (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    match_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    league TEXT NOT NULL,
    season TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome TEXT NOT NULL,
    probability REAL NOT NULL,
    odds REAL NOT NULL,
    edge REAL NOT NULL,
    stake REAL NOT NULL,
    won INTEGER NOT NULL,
    profit REAL NOT NULL,
    bankroll REAL NOT NULL,
    fair_closing_odds REAL,
    clv REAL
);
CREATE INDEX IF NOT EXISTS bets_by_run ON bets (run_id, strategy);
CREATE TABLE IF NOT EXISTS summaries (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL
);
CREATE TABLE IF NOT EXISTS season_summaries (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    season TEXT NOT NULL,
    bets INTEGER NOT NULL,
    staked REAL NOT NULL,
    profit REAL NOT NULL,
    mean_clv REAL
);
CREATE TABLE IF NOT EXISTS scores (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    forecaster TEXT NOT NULL,
    matches INTEGER NOT NULL,
    log_loss REAL NOT NULL,
    rps REAL NOT NULL,
    brier REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    bin_low REAL NOT NULL,
    bin_high REAL NOT NULL,
    mean_forecast REAL NOT NULL,
    observed REAL NOT NULL,
    count INTEGER NOT NULL
);
"""

BET_TABLE_COLUMNS: tuple[str, ...] = (
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
    "fair_closing_odds",
    "clv",
)


def database_path(settings: Settings) -> Path:
    return settings.data_dir / "valuemodel.sqlite"


def connect(path: Path) -> sqlite3.Connection:
    """Open the database, creating it and its tables if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def _settings_json(settings: Settings) -> str:
    return json.dumps({**asdict(settings), "data_dir": str(settings.data_dir)})


def _append(connection: sqlite3.Connection, table: str, frame: pd.DataFrame, run_id: int) -> None:
    if not frame.empty:
        frame.assign(run_id=run_id).to_sql(table, connection, if_exists="append", index=False)


def save_run(connection: sqlite3.Connection, result: BacktestResult) -> int:
    """Store a whole backtest in one transaction and return its run id."""
    with connection:
        cursor = connection.execute(
            "INSERT INTO runs (created_at, league, seasons, xi, settings) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                result.league,
                ",".join(result.seasons),
                result.xi,
                _settings_json(result.settings),
            ),
        )
        run_id = int(cursor.lastrowid)
        for strategy, bets in result.bets.items():
            stored = bets.reindex(columns=list(BET_TABLE_COLUMNS)).assign(
                strategy=strategy,
                date=bets["date"].dt.strftime("%Y-%m-%d") if len(bets) else bets["date"],
                won=bets["won"].astype(int),
            )
            _append(connection, "bets", stored, run_id)
        summaries = result.summary.melt(id_vars="strategy", var_name="metric")
        _append(connection, "summaries", summaries, run_id)
        _append(connection, "season_summaries", result.season_summary, run_id)
        _append(connection, "scores", result.scores, run_id)
        for model, table in result.calibration.items():
            _append(connection, "calibration", table.assign(model=model), run_id)
    return run_id


def latest_run_id(connection: sqlite3.Connection) -> int | None:
    row = connection.execute("SELECT MAX(id) FROM runs").fetchone()
    return row[0]


def _query(connection: sqlite3.Connection, sql: str, run_id: int) -> pd.DataFrame:
    return pd.read_sql_query(sql, connection, params=(run_id,))


def load_run(connection: sqlite3.Connection, run_id: int) -> BacktestResult:
    """Rebuild a stored run in the same shape run_backtest returns."""
    row = connection.execute(
        "SELECT league, seasons, xi, settings FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"No backtest run with id {run_id}")
    league, seasons, xi, settings_json = row
    stored_settings = json.loads(settings_json)
    settings = Settings(**{**stored_settings, "data_dir": Path(stored_settings["data_dir"])})
    result = BacktestResult(
        league=league, seasons=tuple(seasons.split(",")), xi=xi, settings=settings
    )

    summaries = _query(
        connection, "SELECT strategy, metric, value FROM summaries WHERE run_id = ?", run_id
    )
    strategies = list(dict.fromkeys(summaries["strategy"]))
    metrics = list(dict.fromkeys(summaries["metric"]))
    result.summary = (
        summaries.pivot(index="strategy", columns="metric", values="value")
        .reindex(index=strategies, columns=metrics)
        .reset_index()
        .rename_axis(columns=None)
    )

    bets = _query(connection, "SELECT * FROM bets WHERE run_id = ? ORDER BY rowid", run_id)
    bets["date"] = pd.to_datetime(bets["date"])
    bets["won"] = bets["won"].astype(bool)
    for strategy in strategies:
        chosen = bets[bets["strategy"] == strategy]
        result.bets[strategy] = chosen[list(BET_TABLE_COLUMNS)].reset_index(drop=True)

    result.season_summary = _query(
        connection,
        "SELECT strategy, season, bets, staked, profit, mean_clv "
        "FROM season_summaries WHERE run_id = ? ORDER BY rowid",
        run_id,
    )
    result.scores = _query(
        connection,
        "SELECT forecaster, matches, log_loss, rps, brier FROM scores WHERE run_id = ? "
        "ORDER BY rowid",
        run_id,
    )
    calibration = _query(
        connection,
        "SELECT model, bin_low, bin_high, mean_forecast, observed, count "
        "FROM calibration WHERE run_id = ? ORDER BY rowid",
        run_id,
    )
    for model, table in calibration.groupby("model", sort=False):
        result.calibration[model] = table.drop(columns="model").reset_index(drop=True)
    return result
