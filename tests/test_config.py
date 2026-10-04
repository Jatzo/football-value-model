from pathlib import Path

import pytest

from valuemodel.config import (
    Settings,
    load_settings,
    season_code,
    validate_league,
    validate_season,
)


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


def test_load_settings_reads_environment(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_environment.setenv("VALUEMODEL_DATA_DIR", str(tmp_path))
    clean_environment.setenv("VALUEMODEL_REQUEST_DELAY", "0.5")
    settings = load_settings()
    assert settings.data_dir == tmp_path
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.request_delay == 0.5


def test_defaults_when_nothing_is_set(clean_environment: pytest.MonkeyPatch) -> None:
    assert load_settings() == Settings()
    assert Settings().margin_method == "power"
    assert Settings().edge_threshold == 0.03
    assert Settings().kelly_fraction == 0.25
    assert Settings().max_stake == 0.02


def test_staking_settings_are_read(clean_environment: pytest.MonkeyPatch) -> None:
    clean_environment.setenv("VALUEMODEL_MARGIN_METHOD", "proportional")
    clean_environment.setenv("VALUEMODEL_STAKING", "flat")
    clean_environment.setenv("VALUEMODEL_EDGE_THRESHOLD", "0.05")
    clean_environment.setenv("VALUEMODEL_BOOKMAKER", "pinnacle")
    settings = load_settings()
    assert settings.margin_method == "proportional"
    assert settings.staking == "flat"
    assert settings.edge_threshold == 0.05
    assert settings.bookmaker == "pinnacle"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("VALUEMODEL_MARGIN_METHOD", "shin", "must be one of power, proportional"),
        ("VALUEMODEL_STAKING", "martingale", "must be one of kelly, flat"),
        ("VALUEMODEL_BOOKMAKER", "max", "must be one of b365, pinnacle"),
        ("VALUEMODEL_EDGE_THRESHOLD", "-0.01", "between"),
        ("VALUEMODEL_MAX_STAKE", "1.5", "between"),
        ("VALUEMODEL_KELLY_FRACTION", "quarter", "must be a number"),
    ],
)
def test_bad_settings_are_rejected(
    clean_environment: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    clean_environment.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        load_settings()
