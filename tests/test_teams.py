import pytest

from valuemodel.teams import normalise_team


def test_canonical_name_is_unchanged() -> None:
    assert normalise_team("Nott'm Forest") == "Nott'm Forest"


@pytest.mark.parametrize(
    "raw", ["  Man United", "Man United ", "Man  United", "Man" + chr(0xA0) + "United"]
)
def test_whitespace_is_collapsed(raw: str) -> None:
    assert normalise_team(raw) == "Man United"


def test_alias_maps_to_canonical_name() -> None:
    aliases = {"Manchester United": "Man United"}
    assert normalise_team(" Manchester  United ", aliases) == "Man United"


def test_unknown_name_passes_through() -> None:
    assert normalise_team("Ipswich", {"Manchester United": "Man United"}) == "Ipswich"


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_name_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalise_team(raw)
