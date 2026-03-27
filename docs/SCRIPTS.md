# Scripts Reference

<!-- AUTO-GENERATED from source files -->

## Available Runners

| Script | Purpose |
|--------|---------|
| `batch_runner.py` | Unified portfolio batch backtest on multiple symbols |
| `strategy_finder_runner.py` | Automated profitable strategy discovery |
| `run_backtest.py` | Simple streak breakout backtest runner |

---

## batch_runner.py

Unified portfolio mode - runs backtest on multiple symbols as a single unified portfolio with shared capital.

```bash
# Single symbol
py batch_runner.py --symbols BTCUSDT --strategy streak

# Multiple symbols
py batch_runner.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --strategy streak

# All available symbols
py batch_runner.py --all --strategy streak

# Custom TP/SL
py batch_runner.py --symbols BTCUSDT --strategy streak --tp 0.03 --sl 0.015

# Custom capital and positions
py batch_runner.py --all --strategy streak --capital 100000 --max-positions 20 --position-size 0.05

# Save results
py batch_runner.py --symbols BTCUSDT --output results.csv

# Quiet mode (faster - skips validation, reduces logging)
py batch_runner.py --all --strategy streak --quiet
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbols` | BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT | Symbols to test |
| `--all` | False | Run on all available symbols |
| `--strategy` | rsi | Strategy name (rsi, ma, macd, streak) |
| `--data-dir` | C:\Personals\Code\backtest-with-bon | Data directory |
| `--tp` | 0.02 | Take profit % |
| `--sl` | 0.01 | Stop loss % |
| `--leverage` | 1.0 | Leverage multiplier |
| `--capital` | 100000 | Initial capital for unified portfolio |
| `--max-positions` | 10 | Maximum simultaneous positions |
| `--position-size` | 0.1 | Position size as % of available capital |
| `--output` | None | Output CSV file |
| `--quiet` | False | Suppress output |

---

## strategy_finder_runner.py

Automated strategy discovery and optimization. Tests multiple strategies across symbols, filters by statistical significance, and ranks by composite score.

```bash
# Find best strategies across all symbols
python strategy_finder_runner.py --all

# Run on specific symbols
python strategy_finder_runner.py --symbols BTCUSDT ETHUSDT

# Run specific strategies only
python strategy_finder_runner.py --all --strategies RSI MACD

# Custom filtering
python strategy_finder_runner.py --all --min-sharpe 1.5 --min-trades 50

# Output options
python strategy_finder_runner.py --all --output results.csv --top-k 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbols` | BTCUSDT | Symbols to test |
| `--all` | False | Run on all available symbols |
| `--strategies` | All | Strategies to test (RSI, MA_CROSSOVER, MACD, BOLLINGER_BANDS, STREAK_BREAKOUT, STOCHASTIC, CONSECUTIVE_CANDLE) |
| `--data-dir` | C:\Personals\Code\backtest-with-bon | Data directory |
| `--output` | strategy_finder_results.csv | Output CSV file |
| `--summary` | strategy_finder_summary.txt | Summary text file |
| `--top-k` | 5 | Top K strategies per symbol |
| `--min-trades` | 30 | Minimum trades for filtering |
| `--min-sharpe` | 1.0 | Minimum Sharpe ratio |
| `--min-pf` | 1.2 | Minimum profit factor |
| `--max-dd` | 30.0 | Maximum drawdown % |
| `--max-combos` | 500 | Max parameter combinations per symbol |
| `--parallel` | 4 | Parallel workers |
| `--quiet` | False | Reduce logging |

---

## run_backtest.py

Simple streak breakout strategy runner with hardcoded parameters for quick testing.

```bash
python run_backtest.py
```

Note: This script has hardcoded parameters (TP=6%, SL=6%, leverage=10, etc.) for quick runs. For flexible testing, use `batch_runner.py` instead.

<!-- /AUTO-GENERATED -->
