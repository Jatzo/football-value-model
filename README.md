# Football Value Model

A Dixon-Coles model that prices football matches, compares its prices with bookmaker odds and tests the result with an honest walk-forward backtest. Paper trading only.

![Dashboard screenshot](docs/screenshot.png)

## Status

Work in progress. The data pipeline, both models, paper staking, the backtest and the dashboard are in place.

The short version of the results: the model does not beat the market. Over three Premier League seasons its bets were struck at prices worse than where the market closed, and they lost money. The details are below.

## Quick start

You need Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
valuemodel download
pytest
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

`valuemodel download` fetches the Premier League from 2019/20 to 2025/26 and prints how many matches each season holds and how many have Pinnacle closing odds. Files are cached in `data/raw/`, so running it again does not touch the network. Use `--leagues`, `--seasons` and `--refresh` to change what is fetched, for example `valuemodel download --leagues E0 E1 --seasons 2425`.

Once the data is cached you can price a match. By default the model is fitted on every result before the day after the last cached match:

```bash
valuemodel predict --home Arsenal --away Chelsea
valuemodel predict --home Arsenal --away Chelsea --as-of 2025-02-01 --model poisson
```

Team names are spelt as football-data.co.uk spells them, for example `Man United` and `Nott'm Forest`.

Add a bookmaker's decimal odds to see the edge on each outcome and the paper stake the model would put on any value bet:

```bash
valuemodel predict --home Arsenal --away Chelsea --odds 1.70 3.90 5.25 --totals-odds 1.95 1.95
```

To run the backtest and print the report below:

```bash
valuemodel backtest
```

Each run is also saved to `data/valuemodel.sqlite`. Add `--no-save` to skip that.

## Dashboard

```bash
valuemodel backtest
valuemodel fixtures
flask --app valuemodel.web run
```

Then open http://127.0.0.1:5000. The dashboard has four pages:

- **Summary**: headline figures with closing line value first, a bankroll chart for each strategy, and results by season.
- **Bets**: the full bet log, filtered by strategy, league, season, market and result.
- **Models**: the model comparison table and a calibration chart.
- **Fixtures**: upcoming matches priced by the model next to Bet365's odds, with value highlighted.

Use the run picker in the header to look at earlier backtests. The dashboard only reads the local database and cached files and never contacts the data source itself, so `valuemodel fixtures` is what fetches the latest fixtures file and refreshes this season's results. The fixtures file covers many leagues and changes through the week. The model can only price leagues it has results for, and the page says so when the file has no matches for them. Charts are drawn with Chart.js, loaded from a CDN, so they need an internet connection.

## How the model works

Each team gets two numbers: an attack strength and a defence strength. A team's expected goals in a match come from its own attack, the opponent's defence and, for the home side, a home advantage term shared by the whole league. Goals are then treated as Poisson counts, which gives a probability for every scoreline up to 10 goals each. Adding up the right scorelines gives the chances of a home win, a draw, an away win, and over or under 2.5 goals.

That is the baseline Poisson model. Dixon and Coles (1997) noticed that real football has slightly different numbers of 0-0, 1-0, 0-1 and 1-1 results than independent Poisson counts predict, and added one parameter, rho, to correct those four scores. That is the main model here. On recent Premier League data the fitted rho is close to zero, so in practice the two models give almost the same prices. The comparison is still worth keeping, because the effect is clearer in other leagues and eras.

Both models are fitted by maximum likelihood, and the attack strengths are constrained to sum to zero so the numbers have a single answer. Recent matches count for more than old ones: each match is weighted by `exp(-xi * days_ago)`, and matches more than three years old are left out. A fit of three seasons takes a few hundredths of a second, which matters because the backtest refits before every matchday.

### Choosing the time decay

The decay rate `xi` was chosen by validation rather than guesswork. The seasons are split so that nothing used to choose a setting is later reported as a backtest result:

| Seasons | Use |
| --- | --- |
| 2019/20 and 2020/21 | Training history only |
| 2021/22 and 2022/23 | Choosing `xi` |
| 2023/24 to 2025/26 | Backtest |

For each candidate `xi`, the model was refitted at the start of every week of 2021/22 and 2022/23 using only earlier results, then used to forecast that week's matches. `valuemodel tune-xi` reproduces this. Lower scores are better:

| `xi` per day | Log loss | Ranked probability score |
| --- | --- | --- |
| 0 | 0.98408 | 0.20511 |
| 0.001 | 0.97970 | 0.20345 |
| 0.002 | 0.97714 | 0.20241 |
| 0.003 | 0.97638 | 0.20196 |
| 0.004 | 0.97710 | 0.20201 |
| 0.005 | 0.97903 | 0.20246 |

Both scores are lowest at 0.003, which means a match from about 230 days ago counts half as much as one played today. The Poisson baseline gives the same answer. The gaps between neighbouring values are small, so anything from about 0.0025 to 0.004 would give very similar forecasts.

Of the 760 matches in those two seasons, 740 were scored. The other 20 involved a team with fewer than 10 matches in the training window, which in practice means a newly promoted side early in the season.

## Finding value and staking

A bookmaker's odds imply a probability of `1 / odds` for each outcome. Those probabilities add up to more than 1, and the excess is the bookmaker's margin. To compare the model with the market fairly, the margin is removed in one of two ways. The proportional method scales every outcome down by the same factor. The power method raises every implied probability to the same exponent, which takes more off longshots than favourites. Bookmakers tend to load their margin onto unlikely outcomes, so the power method is the default.

A bet has value when `model probability * odds - 1` is at least the edge threshold, 3% by default. This uses the odds actually on offer, margin included, because that is the price a bet would be struck at. At most one bet is taken per market per match: the outcome with the biggest edge in the match result, and the same in over/under 2.5 goals.

Stakes are paper only. The default is quarter Kelly: a quarter of the stake the Kelly criterion recommends, because full Kelly assumes the model's probabilities are exactly right. Flat staking is the alternative. Either way, no single bet can exceed 2% of the current bankroll, and nothing is staked without a positive edge.

## Backtest

### Method

The backtest walks through 2023/24, 2024/25 and 2025/26 in date order. Before each round of matches the model is refitted using only results that were already known when the bookmaker's odds were collected. The data source collects pre-match odds on Friday afternoon for games from Friday to Monday, and on Tuesday afternoon for games from Tuesday to Thursday. So a Sunday match is priced from results up to the Thursday before it, never from Saturday's games. A test rewrites every result after a cutoff date and checks that no forecast made before the cutoff changes.

Bets are struck at Bet365's pre-match price whenever the edge reaches 3%, at most one bet per market per match, with quarter Kelly stakes capped at 2% of the bankroll. All stakes on one day are sized from that morning's bankroll. Games involving a team with fewer than 10 matches in the training window are skipped. None of these seasons was used to choose `xi`.

Three strategies are compared: Dixon-Coles, the Poisson baseline, and simply following the market. The market strategy treats Pinnacle's pre-match prices, with the margin removed, as its forecast, and bets whenever Bet365 offers at least 3% more.

### Closing line value

Closing line value (CLV) compares the price taken with Pinnacle's closing price after its margin is removed. Pinnacle's closing line is widely treated as the most accurate price available, so a bettor with a real edge should usually beat it. CLV is far less noisy than profit, which makes it the most honest single measure here. It is a benchmark only: closing odds never decide a bet or its stake.

| Strategy | Bets | Bets with closing odds | Mean CLV | Beat the close |
| --- | --- | --- | --- | --- |
| Dixon-Coles | 1,413 | 1,205 | -6.4% | 19.8% |
| Poisson | 1,428 | 1,212 | -6.5% | 19.6% |
| Market | 7 | 7 | -4.4% | 57.1% |

For context, backing every Bet365 price in these seasons without any model at all gives a mean CLV of between -4.9% and -8.6%, depending on the outcome. The model's selections are no better than that. It is finding prices where it disagrees with the market, and the market turns out to be right more often than not.

Pinnacle's odds are missing from 17 January 2026 onwards, which is why about 15% of bets have no closing price to compare with.

### Betting results

| Strategy | Bets | Staked | Profit | ROI | Max drawdown | Level-stakes ROI (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| Dixon-Coles | 1,413 | 9,166 | -877 | -9.6% | 91% | -8.3% (-15.4% to -1.0%) |
| Poisson | 1,428 | 9,086 | -877 | -9.6% | 92% | -10.1% (-17.1% to -3.0%) |
| Market | 7 | 21 | +17 | +79.3% | 1% | +125% (-43% to +350%) |

Starting from 1,000 units, both models finished with about 123. Level-stakes ROI puts one unit on every bet, which removes the effect of the order in which results arrived. The interval comes from resampling the bets, and for both models it sits entirely below zero. The market strategy found only seven bets, too few to mean anything, which is itself a sign of how rarely Bet365 is 3% more generous than Pinnacle.

### Model quality

Scored on the 940 matches that every forecaster priced. Lower is better for all three scores.

| Forecaster | Log loss | Ranked probability score | Brier |
| --- | --- | --- | --- |
| Dixon-Coles | 0.9617 | 0.1966 | 0.5708 |
| Poisson | 0.9622 | 0.1967 | 0.5711 |
| Bet365 pre-match, margin removed | 0.9480 | 0.1923 | 0.5623 |
| Pinnacle closing, margin removed | 0.9429 | 0.1907 | 0.5582 |

The models are well calibrated: when Dixon-Coles gives an outcome a 25% chance, it happens about 26% of the time. But the market's forecasts are sharper. A model built only from past scores knows nothing about injuries, suspensions, managerial changes or team news, all of which the market prices in. That gap is the most likely reason the model loses.

## Configuration

Settings come from environment variables. Copy `.env.example` to `.env` to change them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VALUEMODEL_DATA_DIR` | `data` | Where downloaded files are cached |
| `VALUEMODEL_REQUEST_DELAY` | `2` | Seconds to wait between downloads |
| `VALUEMODEL_MARGIN_METHOD` | `power` | How the bookmaker's margin is removed: `power` or `proportional` |
| `VALUEMODEL_EDGE_THRESHOLD` | `0.03` | Smallest edge that counts as value |
| `VALUEMODEL_STAKING` | `kelly` | `kelly` or `flat` |
| `VALUEMODEL_KELLY_FRACTION` | `0.25` | Share of the full Kelly stake to use |
| `VALUEMODEL_MAX_STAKE` | `0.02` | Largest stake on one bet, as a share of the current bankroll |
| `VALUEMODEL_FLAT_STAKE` | `0.01` | Flat stake, as a share of the starting bankroll |
| `VALUEMODEL_STARTING_BANKROLL` | `1000` | Starting bankroll in units |
| `VALUEMODEL_BOOKMAKER` | `b365` | Whose pre-match prices bets are taken at: `b365` or `pinnacle` |

## The data

Results and odds come from [football-data.co.uk](https://www.football-data.co.uk/). Its column names, date formats and file encodings have changed over the years, so the loader maps every era onto one schema and the tests run against small extracts of real files from 2005/06, 2015/16 and 2025/26.

A few things about the data shape what the model can honestly claim:

- The pre-match odds are collected on Friday afternoons for weekend games and Tuesday afternoons for midweek games, so they are roughly a day before kick-off rather than the price available at any chosen moment.
- Pinnacle closing odds, the benchmark for judging whether the model has an edge, are available from 2012/13. In 2025/26 Pinnacle odds stop on 17 January 2026, so only 210 of that season's 380 matches can be measured against them.
- Rows that fail basic checks, such as a result that does not match the score, are logged and left out rather than corrected by guesswork.
- Most of 2020/21 and the end of 2019/20 were played without crowds, and home advantage almost disappeared. Those seasons are only used as training history, and time decay gives them little weight by the time any forecast is scored.

## Limitations

A backtest is not real betting. Prices in the data are a snapshot from one moment and may not have been available for the stake suggested. Bookmakers limit accounts that win, and the gap between a backtest and real results is usually unfavourable. Here the backtest already loses, so this mostly matters as a warning against reading too much into the market strategy's seven bets.

Newly promoted teams arrive with no recent Premier League results, so their strengths cannot be estimated well. A team with fewer than 10 matches in the training window is flagged, its games are not priced, and the backtest will not bet on them. For a side with no Premier League matches in the last three years, that means its first 10 games of the season are skipped.

## Not betting advice

This is a statistics project. It does not place bets and nothing in it is a recommendation to gamble. If gambling is causing you problems, [BeGambleAware](https://www.begambleaware.org/) offers free, confidential support.

## Credits

Data from [football-data.co.uk](https://www.football-data.co.uk/). The model follows Dixon, M. J. and Coles, S. G. (1997), "Modelling association football scores and inefficiencies in the football betting market", *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265 to 280.

## Licence

MIT. See [LICENSE](LICENSE).
