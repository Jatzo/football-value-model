from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from simulation import add_odds, simulate_league, true_model

from valuemodel.backtest import BacktestResult, run_backtest
from valuemodel.config import Settings
from valuemodel.store import (
    connect,
    database_path,
    latest_run_id,
    list_runs,
    load_run,
    save_run,
)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    rng = np.random.default_rng(4)
    model = true_model(8, -0.1, rng)
    seasons = [("2223", "2022-08-01"), ("2324", "2023-08-01")]
    league = pd.concat(
        [simulate_league(model, 2, rng, start=start, season=code) for code, start in seasons],
        ignore_index=True,
    )
    settings = Settings(data_dir=Path("somewhere"), edge_threshold=0.02)
    return run_backtest(add_odds(league, model, rng), "E0", ["2324"], settings, min_matches=0)


def test_database_lives_in_the_data_directory(tmp_path: Path) -> None:
    assert database_path(Settings(data_dir=tmp_path)) == tmp_path / "valuemodel.sqlite"


def test_round_trip(tmp_path: Path, result: BacktestResult) -> None:
    connection = connect(tmp_path / "runs.sqlite")
    run_id = save_run(connection, result)
    loaded = load_run(connection, run_id)

    assert (loaded.league, loaded.seasons, loaded.xi) == (result.league, result.seasons, result.xi)
    assert loaded.settings == result.settings
    pd.testing.assert_frame_equal(loaded.summary, result.summary, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.season_summary, result.season_summary, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.scores, result.scores, check_dtype=False)
    assert set(loaded.calibration) == set(result.calibration)
    for model, table in result.calibration.items():
        pd.testing.assert_frame_equal(loaded.calibration[model], table, check_dtype=False)
    assert set(loaded.bets) == set(result.bets)
    assert len(result.bets["dixon-coles"]) > 0
    for strategy, bets in result.bets.items():
        if len(bets):
            pd.testing.assert_frame_equal(
                loaded.bets[strategy], bets.reset_index(drop=True), check_dtype=False
            )


def test_latest_run_id(tmp_path: Path, result: BacktestResult) -> None:
    connection = connect(tmp_path / "runs.sqlite")
    assert latest_run_id(connection) is None
    first = save_run(connection, result)
    second = save_run(connection, result)
    assert second > first
    assert latest_run_id(connection) == second
    runs = list_runs(connection)
    assert list(runs["id"]) == [second, first]
    assert list(runs["seasons"]) == ["2324", "2324"]


def test_unknown_run_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="No backtest run"):
        load_run(connect(tmp_path / "runs.sqlite"), 99)
