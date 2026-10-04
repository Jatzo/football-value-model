# Football Value Model

A Dixon-Coles model that prices football matches, compares its prices with bookmaker odds and tests the result with an honest walk-forward backtest. Paper trading only.

![Dashboard screenshot](docs/screenshot.png)

## Status

Work in progress. The data pipeline is in place: it downloads results and odds, caches them locally and cleans them into one consistent table. The models, backtest and dashboard come next, and this README will report the backtest results, good or bad, once they exist.

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

`valuemodel download` fetches the Premier League for the last five complete seasons (2021/22 to 2025/26) and prints how many matches each season holds and how many have Pinnacle closing odds. Files are cached in `data/raw/`, so running it again does not touch the network. Use `--leagues`, `--seasons` and `--refresh` to change what is fetched, for example `valuemodel download --leagues E0 E1 --seasons 2425`.

## Configuration

Settings come from environment variables. Copy `.env.example` to `.env` to change them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VALUEMODEL_DATA_DIR` | `data` | Where downloaded files are cached |
| `VALUEMODEL_REQUEST_DELAY` | `2` | Seconds to wait between downloads |

## The data

Results and odds come from [football-data.co.uk](https://www.football-data.co.uk/). Its column names, date formats and file encodings have changed over the years, so the loader maps every era onto one schema and the tests run against small extracts of real files from 2005/06, 2015/16 and 2025/26.

A few things about the data shape what the model can honestly claim:

- The pre-match odds are collected on Friday afternoons for weekend games and Tuesday afternoons for midweek games, so they are roughly a day before kick-off rather than the price available at any chosen moment.
- Pinnacle closing odds, the benchmark for judging whether the model has an edge, are available from 2012/13. In 2025/26 Pinnacle odds stop on 17 January 2026, so only 210 of that season's 380 matches can be measured against them.
- Rows that fail basic checks, such as a result that does not match the score, are logged and left out rather than corrected by guesswork.

## Not betting advice

This is a statistics project. It does not place bets and nothing in it is a recommendation to gamble. If gambling is causing you problems, [BeGambleAware](https://www.begambleaware.org/) offers free, confidential support.

## Credits

Data from [football-data.co.uk](https://www.football-data.co.uk/). The model follows Dixon, M. J. and Coles, S. G. (1997), "Modelling association football scores and inefficiencies in the football betting market", *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265 to 280.

## Licence

MIT. See [LICENSE](LICENSE).
