# Strategy Finder

Automated strategy discovery and optimization module for finding profitable trading strategies.

## Overview

The strategy finder package provides tools to:
- Scan available symbols with valid 1h and 5m data
- Test multiple strategy patterns per symbol
- Optimize parameters via grid search
- Validate with statistical filters
- Rank strategies by composite score
- Output ranked strategies to CSV and summary files

## Architecture

```
backtest/strategy_finder/
├── __init__.py              # Main exports
├── strategies.py            # Strategy templates (RSI, MA_CROSSOVER, MACD, etc.)
├── genetic_optimizer.py     # Genetic algorithm optimizer
├── statistical_filter.py    # Statistical significance filters
├── strategy_ranker.py       # Strategy ranking and output formatting
├── consecutive_candle_strategy.py  # Consecutive candle strategy template
```

## Available Strategy Templates

| Template | Description | Parameters |
|----------|-------------|------------|
| `RSI_TEMPLATE` | RSI mean reversion | `rsi_period`, `oversold`, `overbought` |
| `MA_CROSSOVER_TEMPLATE` | Moving average crossover | `fast_period`, `slow_period` |
| `MACD_TEMPLATE` | MACD momentum | `fast_period`, `slow_period`, `signal_period` |
| `BOLLINGER_BANDS_TEMPLATE` | Bollinger Bands breakout | `bb_period`, `bb_std` |
| `STREAK_BREAKOUT_TEMPLATE` | Consecutive candle breakout | `consecutive_candles`, `atr_window` |
| `STOCHASTIC_TEMPLATE` | Stochastic oscillator | `k_period`, `d_period` |
| `CONSECUTIVE_CANDLE_TEMPLATE` | N consecutive candles | `consecutive_count`, `candle_type` |

## Usage

### Python API

```python
from backtest.strategy_finder import (
    SymbolScanConfig,
    get_available_symbols,
    get_all_templates,
    get_templates_by_names,
    GeneticOptimizer,
    StatisticalFilter,
    StrategyRanker,
)

# Configure symbol scan
scan_config = SymbolScanConfig(data_dir=r"C:\Personals\Code\backtest-with-bon")
symbols = get_available_symbols(scan_config)

# Get strategy templates
templates = get_all_templates()  # All strategies
templates = get_templates_by_names(['RSI', 'MACD'])  # Specific strategies

# Configure statistical filter
stat_filter = StatisticalFilter(
    StatisticalFilterConfig(
        min_trades=30,
        min_sharpe=1.0,
        min_profit_factor=1.2,
        max_drawdown=30.0,
    )
)

# Apply filter
filter_result = stat_filter.apply_all_filters({
    'total_trades': 50,
    'sharpe_ratio': 1.5,
    'profit_factor': 1.5,
    'max_drawdown': 20.0,
    'win_rate': 55.0,
})

# Rank strategies
ranker = StrategyRanker(
    min_trades=30,
    min_sharpe=1.0,
    min_profit_factor=1.2,
    max_drawdown=30.0,
)

ranked = ranker.composite_rank(results)
```

### CLI

```bash
# Find best strategies across all symbols
python strategy_finder_runner.py --all

# Run on specific symbols with filtering
python strategy_finder_runner.py --symbols BTCUSDT ETHUSDT \
    --min-sharpe 1.5 --min-trades 50 --top-k 10
```

## StrategyResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | str | Trading symbol |
| `strategy_name` | str | Name of strategy template |
| `params` | dict | Strategy parameters |
| `total_trades` | int | Total number of trades |
| `winning_trades` | int | Number of winning trades |
| `losing_trades` | int | Number of losing trades |
| `win_rate` | float | Win rate percentage |
| `total_pnl` | float | Total profit/loss in dollars |
| `return_pct` | float | Return percentage |
| `max_drawdown` | float | Maximum drawdown percentage |
| `sharpe_ratio` | float | Sharpe ratio |
| `sortino_ratio` | float | Sortino ratio |
| `profit_factor` | float | Profit factor (gross profit / gross loss) |
| `expectancy` | float | Average trade expectancy |
| `long_trades` | int | Number of long trades |
| `short_trades` | int | Number of short trades |
| `statistical_passed` | bool | Whether strategy passed statistical filters |
