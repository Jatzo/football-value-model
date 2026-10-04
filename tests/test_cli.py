from pathlib import Path

import pytest
from conftest import FakeSource

from valuemodel import cli


@pytest.fixture
def fake_site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource) -> FakeSource:
    monkeypatch.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VALUEMODEL_REQUEST_DELAY", "0")
    monkeypatch.setattr(cli, "make_client", source.client)
    return source


def test_download_reports_each_season(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["download", "--seasons", "1516", "2526"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "E0 1516  downloaded   20 matches  Pinnacle closing odds for 20",
        "E0 2526  downloaded   20 matches  Pinnacle closing odds for 15",
    ]


def test_second_run_uses_the_cache(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["download", "--seasons", "2526"])
    cli.main(["download", "--seasons", "2526"])
    assert len(fake_site.requests) == 1
    assert "cached" in capsys.readouterr().out.splitlines()[-1]


def test_download_failure_returns_an_error_code(
    fake_site: FakeSource, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["download", "--seasons", "9900"]) == 1
    assert "Could not download" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["download", "--seasons", "2426"], "does not cover two consecutive years"),
        (["download", "--leagues", "XX"], "Unknown league 'XX'"),
    ],
)
def test_bad_arguments_are_rejected(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.main(args)
    assert message in capsys.readouterr().err
