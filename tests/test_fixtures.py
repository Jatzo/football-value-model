from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FakeSource
from simulation import add_odds, simulate_league, true_model

from valuemodel.config import Settings
from valuemodel.data import DownloadError
from valuemodel.fixtures import (
    FIXTURES_URL,
    download_fixtures,
    fixtures_path,
    load_fixtures,
    price_fixtures,
)


def test_real_fixtures_file_loads_without_results(fixtures_dir: Path) -> None:
    fixtures = load_fixtures(fixtures_dir / "upcoming.csv")
    assert len(fixtures) == 12
    assert set(fixtures["league"]) == {"E2", "E3", "EC"}
    assert fixtures["date"].dt.date.astype(str).unique().tolist() == ["2026-10-03"]
    assert fixtures["home_goals"].isna().all()
    assert fixtures["result"].isna().all()
    burton = fixtures[fixtures["home_team"] == "Burton"].iloc[0]
    assert (burton["away_team"], burton["kickoff"], burton["b365_home"]) == (
        "Huddersfield",
        "15:00",
        3.9,
    )
    assert fixtures["pinnacle_home"].isna().all()


def test_fixtures_are_ordered_by_kickoff(fixtures_dir: Path) -> None:
    fixtures = load_fixtures(fixtures_dir / "upcoming.csv")
    assert fixtures.iloc[0]["home_team"] == "Chesterfield"
    assert fixtures.iloc[0]["kickoff"] == "12:30"


def test_header_only_file_gives_no_fixtures(tmp_path: Path, fixtures_dir: Path) -> None:
    header = (fixtures_dir / "upcoming.csv").read_bytes().splitlines(keepends=True)[0]
    path = tmp_path / "fixtures.csv"
    path.write_bytes(header)
    assert load_fixtures(path).empty


def test_download_fixtures(tmp_path: Path, source: FakeSource, fixtures_dir: Path) -> None:
    source.body = (fixtures_dir / "upcoming.csv").read_bytes()
    settings = Settings(data_dir=tmp_path)
    path = download_fixtures(source.client(), settings)
    assert path == fixtures_path(settings)
    assert path.read_bytes() == source.body
    assert source.requests == [FIXTURES_URL]


def test_download_fixtures_rejects_html(tmp_path: Path, source: FakeSource) -> None:
    source.body = b"<html></html>"
    with pytest.raises(DownloadError):
        download_fixtures(source.client(), Settings(data_dir=tmp_path))


@pytest.fixture(scope="module")
def history_and_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(13)
    model = true_model(8, -0.1, rng)
    history = simulate_league(model, 3, rng, start="2025-08-01", season="2526")
    upcoming = add_odds(
        simulate_league(model, 1, rng, start="2026-08-01", season="2627"), model, rng
    )
    upcoming = upcoming.iloc[:6].copy()
    upcoming.loc[upcoming.index[0], "b365_home"] = 50.0
    newcomer = upcoming.iloc[[1]].assign(home_team="Newcomers")
    other_league = upcoming.iloc[[2]].assign(league="E2")
    fixtures = pd.concat([upcoming, newcomer, other_league], ignore_index=True)
    return history, fixtures


def test_price_fixtures(history_and_fixtures: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    history, fixtures = history_and_fixtures
    result = price_fixtures(history, fixtures, Settings(), xi=0.003)

    assert result.priced_leagues == ["E0"]
    assert result.unpriced_leagues == ["E2"]
    priced = result.fixtures
    assert len(priced) == 7
    reliable = priced[priced["reliable"]]
    np.testing.assert_allclose(reliable[["home", "draw", "away"]].sum(axis=1), 1.0)
    np.testing.assert_allclose(reliable["edge_home"], reliable["home"] * reliable["odds_home"] - 1)
    assert priced.iloc[0]["value_1x2"] == "home"


def test_unreliable_fixtures_are_not_priced(
    history_and_fixtures: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    history, fixtures = history_and_fixtures
    priced = price_fixtures(history, fixtures, Settings(), xi=0.003).fixtures
    newcomer = priced[priced["home_team"] == "Newcomers"].iloc[0]
    assert not newcomer["reliable"]
    assert np.isnan(newcomer["home"])
    assert pd.isna(newcomer["value_1x2"])


def test_no_known_leagues(history_and_fixtures: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    history, fixtures = history_and_fixtures
    result = price_fixtures(history, fixtures.assign(league="SC1"), Settings(), xi=0.003)
    assert result.fixtures.empty
    assert result.unpriced_leagues == ["SC1"]
