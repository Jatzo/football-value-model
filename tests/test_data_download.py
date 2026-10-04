from pathlib import Path

import pytest
from conftest import FakeSource

from valuemodel.config import Settings
from valuemodel.data import (
    DownloadError,
    cache_path,
    download_season,
    download_seasons,
    load_available,
    load_matches,
    season_url,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, request_delay=1.5)


def test_season_url_uses_the_bare_domain() -> None:
    assert season_url("E0", "2425") == "https://football-data.co.uk/mmz4281/2425/E0.csv"


def test_download_saves_exact_bytes(
    source: FakeSource, settings: Settings, fixtures_dir: Path
) -> None:
    assert download_season(source.client(), "E0", "2526", settings.raw_dir)
    saved = cache_path(settings.raw_dir, "E0", "2526").read_bytes()
    assert saved == (fixtures_dir / "E0_2526.csv").read_bytes()


def test_cached_season_is_not_downloaded_again(source: FakeSource, settings: Settings) -> None:
    client = source.client()
    assert download_season(client, "E0", "2526", settings.raw_dir)
    assert not download_season(client, "E0", "2526", settings.raw_dir)
    assert len(source.requests) == 1


def test_force_downloads_again(source: FakeSource, settings: Settings) -> None:
    client = source.client()
    download_season(client, "E0", "2526", settings.raw_dir)
    assert download_season(client, "E0", "2526", settings.raw_dir, force=True)
    assert len(source.requests) == 2


@pytest.mark.parametrize(
    "body", [b"", b"   \r\n", b"<!DOCTYPE html><html></html>", b"\r\n<html></html>"]
)
def test_non_csv_response_is_rejected_and_not_cached(
    source: FakeSource, settings: Settings, body: bytes
) -> None:
    source.body = body
    with pytest.raises(DownloadError, match="did not return a CSV"):
        download_season(source.client(), "E0", "2526", settings.raw_dir)
    assert not cache_path(settings.raw_dir, "E0", "2526").exists()


def test_http_error_is_wrapped(source: FakeSource, settings: Settings) -> None:
    with pytest.raises(DownloadError, match="Could not download"):
        download_season(source.client(), "E0", "9900", settings.raw_dir)


def test_pauses_only_between_real_requests(source: FakeSource, settings: Settings) -> None:
    client = source.client()
    download_season(client, "E0", "0506", settings.raw_dir)
    pauses: list[float] = []

    results = download_seasons(
        client, ["E0"], ["0506", "1516", "2526"], settings, sleep=pauses.append
    )

    assert results == [("E0", "0506", False), ("E0", "1516", True), ("E0", "2526", True)]
    assert pauses == [1.5]


def test_load_matches_combines_seasons_in_date_order(
    source: FakeSource, settings: Settings
) -> None:
    download_seasons(source.client(), ["E0"], ["2526", "0506"], settings, sleep=lambda _: None)
    matches = load_matches(["E0"], ["2526", "0506"], settings)
    assert len(matches) == 40
    assert matches["date"].is_monotonic_increasing
    assert list(matches["season"].unique()) == ["0506", "2526"]


def test_load_matches_needs_a_cached_file(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="--seasons 2425 2526"):
        load_matches(["E0"], ["2425", "2526"], settings)


def test_load_available_skips_missing_seasons(source: FakeSource, settings: Settings) -> None:
    download_seasons(source.client(), ["E0"], ["2526"], settings, sleep=lambda _: None)
    matches = load_available(["E0"], ["2425", "2526"], settings)
    assert len(matches) == 20
    assert set(matches["season"]) == {"2526"}
    assert load_available(["E1"], ["2526"], settings).empty


def test_a_season_listed_twice_is_loaded_once(source: FakeSource, settings: Settings) -> None:
    download_seasons(source.client(), ["E0"], ["2526"], settings, sleep=lambda _: None)
    assert len(load_available(["E0"], ["2526", "2526"], settings)) == 20
    assert len(load_matches(["E0"], ["2526", "2526"], settings)) == 20
