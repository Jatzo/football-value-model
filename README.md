# Football Value Model

A football model that prices matches from goals and shots, compares its prices with bookmaker odds and tests the result with an honest walk-forward backtest. Paper trading only.

![Dashboard summary page](docs/screenshot.png)

## Results in brief

The model does not beat the market. The main model learns team strengths from both goals and shot-based expected goals. Over three Premier League seasons, 2023/24 to 2025/26, it placed 1,369 paper bets at Bet365's pre-match prices. On average those prices were 6.8% worse than Pinnacle's closing line, fewer than one bet in five beat the close, and the bankroll fell from 1,000 to 270 units. It forecasts slightly better than a classic Dixon-Coles model fitted to goals alone and is well calibrated, but the bookmakers' own prices are better forecasts still. The same holds in the Championship, where shots help more but the model still has no edge. The full method and figures are in [Backtest](#backtest).

## How the model works

Each team gets two numbers: an attack strength and a defence strength. A team's expected goals in a match come from its own attack, the opponent's defence and, for the home side, a home advantage term shared by the whole league. Goals are treated as Poisson counts, which gives a probability for every scoreline up to 10 goals each. Adding up the right scorelines gives the chances of a home win, a draw, an away win, and over or under 2.5 goals.

That is the baseline Poisson model. Dixon and Coles (1997) noticed that real football has slightly different numbers of 0-0, 1-0, 0-1 and 1-1 results than independent Poisson counts predict, and added one parameter, rho, to correct those four scores. That is the classic model for football scores, kept here for comparison. On recent Premier League data the fitted rho is small and unstable: across the backtest it drifts between about -0.11 and +0.08, changing sign along the way. In practice the two models give almost the same prices. The main model goes a step further and also learns from shots, as described below.

The models are fitted by maximum likelihood with `scipy.optimize` and an analytic gradient. The attack strengths are constrained to sum to zero so the fit has a single answer. Recent matches count for more than old ones: each match is weighted by `exp(-xi * days_ago)`, and matches more than three years old are left out. Fitting three seasons takes a few hundredths of a second, which matters because the backtest refits before every round of matches.

### Choosing the time decay

The decay rate `xi` was chosen by validation, not guesswork. The seasons are split so that nothing used to choose a setting is later reported as a backtest result:

| Seasons | Use |
| --- | --- |
| 2019/20 and 2020/21 | Training history only |
| 2021/22 and 2022/23 | Choosing settings |
| 2023/24 to 2025/26 | Backtest |

For each candidate `xi`, Dixon-Coles walked through 2021/22 and 2022/23 on exactly the backtest's schedule, described below, refitting before each round of matches using only earlier results. Lower scores are better:

| `xi` per day | Log loss | Ranked probability score |
| --- | --- | --- |
| 0 | 0.98381 | 0.20504 |
| 0.001 | 0.97930 | 0.20335 |
| 0.002 | 0.97668 | 0.20230 |
| 0.003 | 0.97589 | 0.20187 |
| 0.004 | 0.97659 | 0.20193 |
| 0.005 | 0.97848 | 0.20238 |

Both scores are lowest at 0.003, so a match from about 230 days ago counts half as much as one played today. The Poisson baseline gives the same answer, and anything from about 0.0025 to 0.004 would give very similar forecasts. `valuemodel tune-xi --model dixon-coles` reproduces the table. The main model's time decay is chosen together with its use of shots, below.

### Shot-based expected goals

Goals are noisy: a side can dominate a match and lose it. Shots carry extra evidence about how strong each team is. football-data.co.uk records shots and shots on target, but not where or how each shot was taken, so these are not true expected goals. Instead, each shot on target and each other shot is worth the average number of goals that kind of shot led to in the training window. In recent Premier League seasons that is about 0.3 goals per shot on target and nothing for other shots. The values are kept non-negative, since otherwise a side with shots but none on target could be expected to score fewer than zero goals.

The main model, called shots-adjusted, fits each team's strengths to a blend of actual goals and these expected goals. Its time decay and the share of expected goals in the blend affect each other, so they were chosen together on the same tuning seasons:

| `xi` per day | Share of expected goals | Log loss | Ranked probability score |
| --- | --- | --- | --- |
| 0.005 | 0.5 | 0.96828 | 0.19975 |
| 0.005 | 0.6 | 0.96822 | 0.19971 |
| 0.005 | 0.7 | 0.96872 | 0.19985 |
| 0.006 | 0.5 | 0.96826 | 0.19975 |
| 0.006 | 0.6 | 0.96794 | 0.19963 |
| 0.006 | 0.7 | 0.96826 | 0.19972 |
| 0.007 | 0.5 | 0.96871 | 0.19991 |
| 0.007 | 0.6 | 0.96817 | 0.19973 |
| 0.007 | 0.7 | 0.96831 | 0.19976 |

Wider searches found nothing better, and fitting goals only or expected goals only was clearly worse (ranked probability scores of 0.2023 and 0.2014 at an `xi` of 0.005). A 60% share of expected goals and an `xi` of 0.006 forecast best, so a match about 115 days old counts half as much as one played today. Shots are less noisy than goals, so recent matches can safely count for more than in the goal-only models. `valuemodel tune-xi` and `valuemodel tune-shots` reproduce the search one setting at a time. The Dixon-Coles correction only applies to whole-number scores, so it has nothing to act on with a blend, and this model is fitted as Poisson.

### Finding value and staking

A bookmaker's odds imply a probability of `1 / odds` for each outcome. Those probabilities add up to more than 1, and the excess is the bookmaker's margin. When the market is used as a forecast, the margin is removed with either the proportional method, which scales every outcome down by the same factor, or the power method, which raises every implied probability to the same exponent and so takes more off longshots. Bookmakers tend to load their margin onto unlikely outcomes, so the power method is the default.

A bet has value when `model probability * odds - 1` is at least the edge threshold, 3% by default. This uses the odds actually on offer, margin included, because that is the price a bet would be struck at. At most one bet is taken per market per match: the outcome with the biggest edge in the match result, and the same in over/under 2.5 goals.

Stakes are paper only. The default is quarter Kelly, a quarter of the stake the Kelly criterion recommends, because full Kelly assumes the model's probabilities are exactly right. Flat staking is the alternative. Either way no single bet can exceed 2% of the current bankroll, and nothing is staked without a positive edge.

## Quick start

You need Python 3.12 or later.

```bash
git clone https://github.com/Jatzo/football-value-model.git
cd football-value-model
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
valuemodel download
pytest
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

`valuemodel download` fetches the Premier League from 2019/20 to 2025/26, waiting a couple of seconds between files, and caches them in `data/raw/` so later runs do not touch the network. Use `--leagues`, `--seasons` and `--refresh` to change what is fetched.

Run the backtest, which prints the report below and saves the run to `data/valuemodel.sqlite`:

```bash
valuemodel backtest
```

Price a single match, optionally against a bookmaker's decimal odds to see the edge and the paper stake. Team names are spelt as the data source spells them, for example `Man United` and `Nott'm Forest`:

```bash
valuemodel predict --home Arsenal --away Chelsea
valuemodel predict --home Arsenal --away Chelsea --odds 1.70 3.90 5.25 --totals-odds 1.95 1.95
```

### Dashboard

```bash
valuemodel fixtures
flask --app valuemodel.web run
```

Then open http://127.0.0.1:5000. The summary page leads with closing line value and shows a bankroll chart for each strategy. The bets page has the full bet log with filters for strategy, league, season, market and result. The models page compares the forecasters and shows a calibration chart. The fixtures page prices upcoming matches next to Bet365's odds, highlights value, and lists each match's most likely result in order of the model's confidence. The bets page can also be ordered by the model's chance of each bet winning.

The dashboard only reads the local database and cached files. `valuemodel fixtures` is what fetches the latest fixtures file and refreshes this season's results, for the Premier League and the Championship unless `--leagues` says otherwise. That file covers many leagues and changes through the week, and the model can only price leagues it has results for. Charts use Chart.js from a CDN, so they need an internet connection.

### Development

```bash
pytest
ruff check .
ruff format --check .
```

GitHub Actions runs the same checks on every push and pull request. The code is organised as follows:

| Module | Purpose |
| --- | --- |
| `cli.py`, `config.py` | The `valuemodel` command, settings, leagues and seasons |
| `data.py`, `teams.py`, `fixtures.py` | Downloading, caching, cleaning and standardising results, odds and upcoming fixtures |
| `models/`, `expected_goals.py` | Poisson, Dixon-Coles and shots-adjusted models with time decay, and shot-based expected goals |
| `markets.py`, `odds.py`, `staking.py` | Market probabilities, margin removal, value detection and stakes |
| `walkforward.py`, `tuning.py`, `backtest.py` | Walk-forward forecasting, choosing settings, and the backtest metrics |
| `store.py`, `report.py`, `labels.py`, `web/` | SQLite storage, the text report, shared display formats and the dashboard |

## Configuration

Settings come from environment variables. Copy `.env.example` to `.env` to change them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VALUEMODEL_DATA_DIR` | `data` | Where downloaded files and the database are kept |
| `VALUEMODEL_REQUEST_DELAY` | `2` | Seconds to wait between downloads |
| `VALUEMODEL_MARGIN_METHOD` | `power` | How the margin is removed: `power` or `proportional` |
| `VALUEMODEL_EDGE_THRESHOLD` | `0.03` | Smallest edge that counts as value |
| `VALUEMODEL_STAKING` | `kelly` | `kelly` or `flat` |
| `VALUEMODEL_KELLY_FRACTION` | `0.25` | Share of the full Kelly stake to use |
| `VALUEMODEL_MAX_STAKE` | `0.02` | Largest stake on one bet, as a share of the current bankroll |
| `VALUEMODEL_FLAT_STAKE` | `0.01` | Flat stake, as a share of the starting bankroll |
| `VALUEMODEL_STARTING_BANKROLL` | `1000` | Starting bankroll in units |
| `VALUEMODEL_BOOKMAKER` | `b365` | Whose pre-match prices bets are taken at: `b365` or `pinnacle`. With `pinnacle` the follow-the-market strategy is skipped |

## Backtest

### Method

The backtest walks through 2023/24, 2024/25 and 2025/26 in date order. Before each round of matches the model is refitted using only results that were known when the bookmaker's odds were collected. The data source collects pre-match odds on Friday afternoon for games from Friday to Monday, and on Tuesday afternoon for games from Tuesday to Thursday, so a Sunday match is priced from results up to the Thursday before it, never from Saturday's games. A test rewrites every result after a cutoff date and checks that no forecast made before the cutoff changes.

Bets are struck at Bet365's pre-match price whenever the edge reaches 3%, with quarter Kelly stakes capped at 2% of the bankroll. All stakes on one day are sized from that morning's bankroll. Games involving a team with fewer than 10 matches in the training window are skipped.

Four strategies are compared: the main shots-adjusted model, Dixon-Coles, the Poisson baseline, and following the market. The last treats Pinnacle's pre-match prices, with the margin removed, as its forecast, and bets whenever Bet365 offers at least 3% more.

### Closing line value

Closing line value (CLV) compares the price taken with Pinnacle's closing price after its margin is removed. Pinnacle's closing line is widely treated as the most accurate price available, so a bettor with a real edge should usually beat it. CLV is far less noisy than profit, which makes it the most honest single measure here. It is a benchmark only: closing odds never decide a bet or its stake.

| Strategy | Bets | Bets with closing odds | Mean CLV | Beat the close |
| --- | --- | --- | --- | --- |
| Shots-adjusted | 1,369 | 1,138 | -6.8% | 19.0% |
| Dixon-Coles | 1,413 | 1,205 | -6.4% | 19.8% |
| Poisson | 1,428 | 1,212 | -6.5% | 19.6% |
| Follow the market | 7 | 7 | -4.4% | 57.1% |

For context, backing every Bet365 price in these seasons without any model gives a mean CLV of between -4.9% and -8.6%, depending on the outcome. The model's selections are no better than that. It finds prices where it disagrees with the market, and the market turns out to be right more often than not. Pinnacle's odds are missing from 17 January 2026 onwards, which is why about 15% of bets have no closing price to compare with.

### Betting results

| Strategy | Bets | Staked | Profit | ROI | Max drawdown | Level-stakes ROI (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| Shots-adjusted | 1,369 | 10,840 | -730 | -6.7% | 81% | -7.1% (-15.4% to +1.2%) |
| Dixon-Coles | 1,413 | 9,166 | -877 | -9.6% | 91% | -8.3% (-15.4% to -1.0%) |
| Poisson | 1,428 | 9,086 | -877 | -9.6% | 92% | -10.1% (-17.1% to -3.0%) |
| Follow the market | 7 | 21 | +17 | +79.3% | 1% | +125% (-43% to +350%) |

Starting from 1,000 units, the main model finished with 270, and Dixon-Coles and Poisson with about 123. Level-stakes ROI puts one unit on every bet, which removes the effect of the order in which results arrived. The interval comes from resampling the bets. For Dixon-Coles and Poisson it sits entirely below zero. For the main model it just reaches above zero, but its closing line value is no better, so the smaller loss is most likely luck rather than a real edge. The market strategy found only seven bets, too few to mean anything, which itself shows how rarely Bet365 is 3% more generous than Pinnacle.

### Model quality

Scored on the 940 matches that every forecaster priced. Lower is better for all three scores.

| Forecaster | Log loss | Ranked probability score | Brier |
| --- | --- | --- | --- |
| Shots-adjusted | 0.9617 | 0.1962 | 0.5703 |
| Dixon-Coles | 0.9617 | 0.1966 | 0.5708 |
| Poisson | 0.9622 | 0.1967 | 0.5711 |
| Bet365 pre-match, margin removed | 0.9480 | 0.1923 | 0.5623 |
| Pinnacle closing, margin removed | 0.9429 | 0.1907 | 0.5582 |

The models are well calibrated: when the main model gives an outcome a 25% chance, it happens about a quarter of the time. But the market's forecasts are sharper. A model built only from past scores knows nothing about injuries, suspensions, managerial changes or team news, all of which the market prices in. That gap is the most likely reason the model loses. Adding shots narrows it only slightly: on seasons it was never tuned on, the main model beats Dixon-Coles on ranked probability and Brier scores and ties it on log loss. Its tuned settings did a little worse here than the even blend first tried (a ranked probability score of 0.1962 against 0.1959). That is a normal cost of tuning on limited data, and the tuned settings are kept, since switching after seeing these results would mean tuning on the test seasons.

### Bets by the model's chance of winning

The bets each model was surest about win most often, but at short odds. For the main model:

| Model's chance | Bets | Average chance | Won | Level-stakes ROI |
| --- | --- | --- | --- | --- |
| Under 30% | 407 | 21.0% | 14.7% | -16.4% |
| 30% to 45% | 399 | 37.5% | 33.1% | +3.4% |
| 45% to 60% | 415 | 52.2% | 43.6% | -7.0% |
| Over 60% | 148 | 66.9% | 54.1% | -10.3% |

In every band the bets won less often than the model expected, and the bets it was most confident about did no better than the rest. That is a selection effect: value bets are chosen where the model disagrees with the market, and on exactly those matches the model is overconfident. The 30% to 45% band happened to make money, but picking out one profitable band after the event is how backtests mislead, so it is not treated as a finding.

### Championship

The same backtest was run on the Championship, with every setting left exactly as tuned on the Premier League. The model never saw Championship data while its settings were chosen, which makes this a clean out-of-sample test. To reproduce it, run `valuemodel download --leagues E1` and then `valuemodel backtest --league E1`.

![Dashboard summary page for the Championship backtest](docs/screenshot-championship.png)

| Strategy | Bets | Mean CLV | Beat the close | ROI | Final bankroll | Level-stakes ROI (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| Shots-adjusted | 1,779 | -5.6% | 19.3% | -4.9% | 154 | -3.5% (-10.0% to +3.0%) |
| Dixon-Coles | 1,940 | -6.0% | 17.6% | -11.0% | 43 | -8.7% (-14.1% to -2.8%) |
| Poisson | 1,944 | -6.1% | 17.2% | -10.7% | 46 | -8.5% (-14.2% to -2.5%) |
| Follow the market | 23 | +2.1% | 65.2% | +20.2% | 1,023 | +46.6% (-42.4% to +156.1%) |

| Forecaster | Log loss | Ranked probability score | Brier |
| --- | --- | --- | --- |
| Shots-adjusted | 1.0349 | 0.2155 | 0.6227 |
| Dixon-Coles | 1.0419 | 0.2178 | 0.6275 |
| Poisson | 1.0416 | 0.2178 | 0.6273 |
| Bet365 pre-match, margin removed | 1.0296 | 0.2139 | 0.6184 |
| Pinnacle closing, margin removed | 1.0256 | 0.2128 | 0.6157 |

Shots help more here than in the Premier League. The main model beats Dixon-Coles on all three scores, by 0.0023 in ranked probability score against 0.0004 in the Premier League, and its forecasts come closer to Bet365's. The Championship market is probably priced less sharply than the Premier League. The verdict is still the same, though: a mean closing line value of -5.6% means the prices taken were worse than where the market closed, so the model has no edge. The market strategy's positive closing line value comes from only 23 bets, too few to read anything into.

## Limitations

**Data quality.** Results and odds come from one free source whose column names, date formats and encodings have changed over the years. The loader maps every era onto one schema and is tested against real extracts from 2005/06, 2015/16 and 2025/26. Rows that fail basic checks, such as a result that does not match the score, are logged and left out rather than corrected by guesswork. Most of 2020/21 was played without crowds and home advantage almost disappeared. Those matches are only used as training history, and time decay gives them little weight by the time any forecast is scored.

**When the odds were captured.** The pre-match odds are a single snapshot taken on Friday or Tuesday afternoon, roughly a day before kick-off. They are not the price available at any chosen moment, and the backtest assumes every stake could have been placed at that snapshot.

**Promoted teams.** Newly promoted sides arrive with no recent Premier League results, so their strengths cannot be estimated well. A team with fewer than 10 matches in the training window is flagged and its games are not priced or bet on. For a side with no Premier League matches in the last three years, that means its first 10 games are skipped.

**Backtests are not real betting.** Bookmakers limit accounts that win, prices move once money arrives, and the gap between a backtest and real results is almost always unfavourable. Here the backtest already loses, so this mostly matters as a warning against reading anything into the market strategy's seven bets.

## Not betting advice

This is a statistics project. It does not place bets and nothing in it is a recommendation to gamble. If gambling is causing you problems, [BeGambleAware](https://www.begambleaware.org/) offers free, confidential support.

## Credits

Data from [football-data.co.uk](https://www.football-data.co.uk/). The model follows Dixon, M. J. and Coles, S. G. (1997), "Modelling association football scores and inefficiencies in the football betting market", *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265 to 280.

## Licence

MIT. See [LICENSE](LICENSE).

## Technical skills

| Area | Skills used in this project |
| --- | --- |
| Python | Python 3.12, type hints throughout, dataclasses, generics, packaging with `pyproject.toml` and a console script |
| Statistical modelling | Poisson regression, the Dixon-Coles model, maximum likelihood estimation with analytic gradients, L-BFGS-B optimisation with SciPy, identifiability constraints, exponential time decay, shot-based expected goals with non-negative least squares |
| Model evaluation | Walk-forward validation, choosing a hyperparameter on held-out seasons, log loss, Brier score, ranked probability score, calibration analysis, bootstrap confidence intervals |
| Betting maths | Implied probabilities, margin removal by the proportional and power methods, expected value, fractional Kelly staking with a cap, closing line value, drawdown and losing run analysis |
| Backtesting | Point-in-time refitting with no lookahead, separate tuning and test periods, realistic odds timing, comparison against simple baselines |
| Data engineering | pandas and NumPy vectorisation, cleaning CSV files whose columns, encodings and date formats changed over 20 years, validation of every row, a polite HTTP client with httpx, caching and atomic file writes |
| Storage | SQLite through the standard library, schema design, transactions, foreign keys, older saved runs that still load after the settings change |
| Web | Flask with an app factory and blueprint, Jinja templates and filters, Chart.js, plain JavaScript, responsive CSS with light and dark modes, subresource integrity |
| Command line | argparse subcommands, input validation with clear error messages, close-match suggestions for misspelt names |
| Testing | pytest with fixtures and parametrisation, simulation tests that recover known parameters, gradient checks, a test proven to catch lookahead, mocked HTTP, Flask's test client, fixtures cut from real data files |
| Engineering practice | ruff for linting and formatting, GitHub Actions CI, configuration through environment variables, small focused commits |
