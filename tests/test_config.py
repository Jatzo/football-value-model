from pathlib import Path

import pytest

from valuemodel.config import load_settings, season_code, validate_league, validate_season


@pytest.mark.parametrize(
    ("start_year", "expected"),
    [(2024, "2425"), (2005, "0506"), (1999, "9900"), (2009, "0910")],
)
def test_season_code(start_year: int, expected: str) -> None:
    assert season_code(start_year) == expected


@pytest.mark.parametrize("code", ["2425", "0506", "9900"])
def test_validate_season_accepts_consecutive_years(code: str) -> None:
    assert validate_season(code) == code


@pytest.mark.parametrize("code", ["2426", "24", "24/25", "abcd", "2524"])
def test_validate_season_rejects_bad_codes(code: str) -> None:
    with pytest.raises(ValueError):
        validate_season(code)


def test_validate_league() -> None:
    assert validate_league("E0") == "E0"
    with pytest.raises(ValueError, match="Unknown league"):
        validate_league("XX")


def test_load_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VALUEMODEL_REQUEST_DELAY", "0.5")
    settings = load_settings()
    assert settings.data_dir == tmp_path
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.request_delay == 0.5
