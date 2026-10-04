from pathlib import Path

import httpx
import pytest

from valuemodel.config import Settings
from valuemodel.data import (
    DownloadError,
    cache_path,
    download_season,
    download_seasons,
    load_matches,
    season_url,
)


class FakeSource:
    """Serves fixture files in place of the real site and records every request."""

    def __init__(self, fixtures_dir: Path, body: bytes | None = None, status: int = 200) -> None:
        self.fixtures_dir = fixtures_dir
        self.body = body
        self.status = status
        self.requests: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if self.body is not None:
            return httpx.Response(self.status, content=self.body)
        season, league = request.url.path.split("/")[-2:]
        fixture = self.fixtures_dir / f"{league.removesuffix('.csv')}_{season}.csv"
        if not fixture.exists():
            return httpx.Response(404)
        return httpx.Response(self.status, content=fixture.read_bytes())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, request_delay=1.5)


def test_season_url_uses_the_bare_domain() -> None:
    assert season_url("E0", "2425") == "https://football-data.co.uk/mmz4281/2425/E0.csv"


def test_download_saves_exact_bytes(fixtures_dir: Path, settings: Settings) -> None:
    source = FakeSource(fixtures_dir)
    assert download_season(source.client(), "E0", "2526", settings.raw_dir)
    saved = cache_path(settings.raw_dir, "E0", "2526").read_bytes()
    assert saved == (fixtures_dir / "E0_2526.csv").read_bytes()


def test_cached_season_is_not_downloaded_again(fixtures_dir: Path, settings: Settings) -> None:
    source = FakeSource(fixtures_dir)
    client = source.client()
    assert download_season(client, "E0", "2526", settings.raw_dir)
    assert not download_season(client, "E0", "2526", settings.raw_dir)
    assert len(source.requests) == 1


def test_force_downloads_again(fixtures_dir: Path, settings: Settings) -> None:
    source = FakeSource(fixtures_dir)
    client = source.client()
    download_season(client, "E0", "2526", settings.raw_dir)
    assert download_season(client, "E0", "2526", settings.raw_dir, force=True)
    assert len(source.requests) == 2


@pytest.mark.parametrize(
    "body", [b"", b"   \r\n", b"<!DOCTYPE html><html></html>", b"\r\n<html></html>"]
)
def test_non_csv_response_is_rejected_and_not_cached(
    fixtures_dir: Path, settings: Settings, body: bytes
) -> None:
    source = FakeSource(fixtures_dir, body=body)
    with pytest.raises(DownloadError, match="did not return a CSV"):
        download_season(source.client(), "E0", "2526", settings.raw_dir)
    assert not cache_path(settings.raw_dir, "E0", "2526").exists()


def test_http_error_is_wrapped(fixtures_dir: Path, settings: Settings) -> None:
    source = FakeSource(fixtures_dir)
    with pytest.raises(DownloadError, match="Could not download"):
        download_season(source.client(), "E0", "9900", settings.raw_dir)


def test_pauses_only_between_real_requests(fixtures_dir: Path, settings: Settings) -> None:
    source = FakeSource(fixtures_dir)
    client = source.client()
    download_season(client, "E0", "0506", settings.raw_dir)
    pauses: list[float] = []

    results = download_seasons(
        client, ["E0"], ["0506", "1516", "2526"], settings, sleep=pauses.append
    )

    assert results == [("E0", "0506", False), ("E0", "1516", True), ("E0", "2526", True)]
    assert pauses == [1.5]


def test_load_matches_combines_seasons_in_date_order(
    fixtures_dir: Path, settings: Settings
) -> None:
    download_seasons(
        FakeSource(fixtures_dir).client(),
        ["E0"],
        ["2526", "0506"],
        settings,
        sleep=lambda _: None,
    )
    matches = load_matches(["E0"], ["2526", "0506"], settings)
    assert len(matches) == 40
    assert matches["date"].is_monotonic_increasing
    assert list(matches["season"].unique()) == ["0506", "2526"]


def test_load_matches_needs_a_cached_file(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="valuemodel download"):
        load_matches(["E0"], ["2526"], settings)
