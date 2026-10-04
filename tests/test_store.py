from pathlib import Path

import pandas as pd
import pytest
from simulation import simulated_seasons

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
    league = simulated_seasons(4, 8, 2)
    settings = Settings(data_dir=Path("somewhere"), edge_threshold=0.02)
    return run_backtest(league, "E0", ["2324"], settings, min_matches=0)


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


def test_round_trip_of_a_run_without_bets(tmp_path: Path, result: BacktestResult) -> None:
    empty = BacktestResult(
        **{**result.__dict__, "bets": {name: bets.iloc[0:0] for name, bets in result.bets.items()}}
    )
    connection = connect(tmp_path / "runs.sqlite")
    loaded = load_run(connection, save_run(connection, empty))
    assert all(len(bets) == 0 for bets in loaded.bets.values())


def test_old_runs_load_after_settings_change(tmp_path: Path, result: BacktestResult) -> None:
    connection = connect(tmp_path / "runs.sqlite")
    run_id = save_run(connection, result)
    with connection:
        connection.execute(
            "UPDATE runs SET settings = json_set(settings, '$.retired_option', 1) WHERE id = ?",
            (run_id,),
        )
    assert load_run(connection, run_id).settings == result.settings
