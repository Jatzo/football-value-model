"""Paper stake sizing: flat stakes and fractional Kelly, both capped per bet."""

from valuemodel.config import Settings


def kelly_fraction(probability: float, odds: float) -> float:
    """Share of the bankroll that full Kelly would stake. Zero without a positive edge."""
    edge = probability * odds - 1.0
    if edge <= 0:
        return 0.0
    return edge / (odds - 1.0)


def kelly_stake(
    probability: float,
    odds: float,
    bankroll: float,
    fraction: float = Settings.kelly_fraction,
    max_share: float = Settings.max_stake,
) -> float:
    """Fractional Kelly stake, never more than max_share of the bankroll.

    Full Kelly assumes the model's probabilities are exactly right. They are
    not, so a fraction of it trades a little growth for much smaller swings.
    """
    if bankroll <= 0:
        return 0.0
    return min(fraction * kelly_fraction(probability, odds), max_share) * bankroll


def flat_stake(bankroll: float, unit: float, max_share: float = Settings.max_stake) -> float:
    """The same amount on every bet, still within the per-bet cap."""
    if bankroll <= 0:
        return 0.0
    return min(unit, max_share * bankroll)


def stake(probability: float, odds: float, bankroll: float, settings: Settings) -> float:
    """Stake for one bet using the configured method. Zero when there is no edge."""
    if probability * odds - 1.0 <= 0:
        return 0.0
    if settings.staking == "flat":
        unit = settings.flat_stake_share * settings.starting_bankroll
        return flat_stake(bankroll, unit, settings.max_stake)
    return kelly_stake(probability, odds, bankroll, settings.kelly_fraction, settings.max_stake)
