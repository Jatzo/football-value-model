import os
from pathlib import Path

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeSource:
    """Serves fixture files in place of the real site and records every request."""

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir
        self.body: bytes | None = None
        self.requests: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if self.body is not None:
            return httpx.Response(200, content=self.body)
        season, league = request.url.path.split("/")[-2:]
        fixture = self.fixtures_dir / f"{league.removesuffix('.csv')}_{season}.csv"
        if not fixture.exists():
            return httpx.Response(404)
        return httpx.Response(200, content=fixture.read_bytes())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def source() -> FakeSource:
    return FakeSource(FIXTURES)


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove any settings from the real environment and stop .env being read."""
    for name in [n for n in os.environ if n.startswith("VALUEMODEL_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr("valuemodel.config.load_dotenv", lambda: None)
    return monkeypatch
