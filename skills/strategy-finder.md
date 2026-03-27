# Strategy Finder - Automated Profitable Strategy Discovery

## Overview

The Strategy Finder is an automated system that scans trading symbols, tests multiple strategy patterns with parameter optimization, validates robustness via walk-forward analysis, filters by statistical significance, and outputs ranked profitable strategies per symbol.

## System Architecture

```
strategy_finder_runner.py (CLI Entry Point)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  backtest/strategy_finder/                                  │
│  ├── __init__.py          # Symbol scanner, exports        │
│  ├── strategies.py         # 6 strategy templates          │
│  ├── genetic_optimizer.py  # GA for parameter tuning       │
│  ├── statistical_filter.py # t-test, CI, walk-forward       │
│  └── strategy_ranker.py    # Ranking & output              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```bash
# Test on single symbol with RSI strategy
py strategy_finder_runner.py --symbols BTCUSDT --strategies RSI

# Multiple symbols
py strategy_finder_runner.py --symbols BTCUSDT ETHUSDT SOLUSDT --strategies RSI MACD

# All available symbols (583)
py strategy_finder_runner.py --all
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbols` | list | `['BTCUSDT']` | Symbols to test |
| `--all` | flag | False | Run on all 583 available symbols |
| `--strategies` | list | all | Strategies to test: RSI, MA_CROSSOVER, MACD, BOLLINGER_BANDS, STREAK_BREAKOUT, STOCHASTIC |
| `--data-dir` | str | project dir | Data directory path |
| `--output` | str | results.csv | Output CSV file |
| `--summary` | str | summary.txt | Summary text file |
| `--top-k` | int | 5 | Top K strategies per symbol |
| `--min-trades` | int | 30 | Minimum trades for filtering |
| `--min-sharpe` | float | 1.0 | Minimum Sharpe ratio |
| `--min-pf` | float | 1.2 | Minimum profit factor |
| `--max-dd` | float | 30.0 | Maximum drawdown % |
| `--max-combos` | int | 500 | Max parameter combinations per symbol |
| `--parallel` | int | 4 | Parallel workers |
| `--quiet` | flag | False | Reduce logging |

### Example Commands

```bash
# Conservative scan (fast)
py strategy_finder_runner.py --symbols BTCUSDT --strategies RSI --max-combos 20

# Aggressive scan (thorough)
py strategy_finder_runner.py --all --strategies RSI MACD MA_CROSSOVER --max-combos 200 --min-trades 50

# High threshold (only best strategies)
py strategy_finder_runner.py --all --min-sharpe 2.0 --min-pf 1.5 --min-trades 100
```

## Available Strategies

### 1. RSI (Relative Strength Index)
Mean reversion strategy based on overbought/oversold levels.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `rsi_period` | [7, 10, 14, 21, 28] | 14 |
| `oversold` | [20, 25, 30, 35, 40] | 30 |
| `overbought` | [60, 65, 70, 75, 80] | 70 |

**Logic:**
- LONG: RSI crosses above oversold level
- SHORT: RSI crosses below overbought level

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies RSI
```

### 2. MA_CROSSOVER (Moving Average Crossover)
Trend-following strategy based on fast/slow MA crossovers.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `fast_period` | [5, 10, 15, 20, 25, 30] | 20 |
| `slow_period` | [40, 50, 60, 70, 80, 100] | 50 |
| `direction` | ['both', 'long', 'short'] | 'both' |

**Logic:**
- LONG: Fast MA crosses above Slow MA
- SHORT: Fast MA crosses below Slow MA

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies MA_CROSSOVER
```

### 3. MACD (Moving Average Convergence Divergence)
Momentum strategy based on MACD line crossovers.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `fast_period` | [8, 12, 16, 20] | 12 |
| `slow_period` | [20, 26, 32, 40] | 26 |
| `signal_period` | [6, 9, 12, 15] | 9 |

**Logic:**
- LONG: MACD crosses above signal line
- SHORT: MACD crosses below signal line

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies MACD
```

### 4. BOLLINGER_BANDS
Bollinger Bands breakout or mean reversion strategy.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `bb_period` | [10, 15, 20, 25, 30] | 20 |
| `std_dev` | [1.5, 2.0, 2.5, 3.0] | 2.0 |
| `mode` | ['mean_reversion', 'breakout'] | 'mean_reversion' |

**Logic (Mean Reversion):**
- LONG: Price crosses above lower band (from below)
- SHORT: Price crosses below upper band (from above)

**Logic (Breakout):**
- LONG: Price crosses above upper band
- SHORT: Price crosses below lower band

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies BOLLINGER_BANDS
```

### 5. STREAK_BREAKOUT
Catches reversals after extended consecutive moves.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `consecutive_candles` | [3, 4, 5, 6] | 4 |
| `atr_window_min` | [0.5, 1.0, 1.5] | 1.0 |
| `atr_window_max` | [4.0, 6.0, 8.0] | 6.0 |
| `risk_reward_ratio_sl` | [0.3, 0.5, 0.7, 1.0] | 0.5 |
| `risk_reward_ratio_tp` | [1.0, 1.5, 2.0, 2.5] | 1.5 |

**Logic:**
- LONG: After N consecutive green candles with ATR filter
- SHORT: After N consecutive red candles with ATR filter

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies STREAK_BREAKOUT
```

### 6. STOCHASTIC
Stochastic Oscillator mean reversion strategy.

**Parameters:**
| Param | Range | Default |
|-------|-------|---------|
| `k_period` | [10, 14, 20] | 14 |
| `d_period` | [3, 5] | 3 |
| `oversold` | [15, 20, 25] | 20 |
| `overbought` | [75, 80, 85] | 80 |

**Logic:**
- LONG: %K crosses above %D from oversold
- SHORT: %K crosses below %D from overbought

**Example:**
```bash
py strategy_finder_runner.py --symbols BTCUSDT --strategies STOCHASTIC
```

## Output Files

### CSV Output (`--output`)
Contains all strategies passing filters with columns:
- `symbol` - Trading symbol
- `strategy` - Strategy name
- `params` - JSON string of optimized parameters
- `total_trades` - Number of trades
- `winning_trades` / `losing_trades`
- `win_rate` - Win rate percentage
- `total_pnl` - Profit/loss in dollars
- `return_pct` - Return percentage
- `sharpe_ratio` - Sharpe ratio
- `profit_factor` - Profit factor
- `max_drawdown` - Maximum drawdown %
- `sortino_ratio` - Sortino ratio
- `consistency_score` - Walk-forward consistency (0-1)
- `statistical_passed` - Whether statistical filters passed

### Summary Text (`--summary`)
Human-readable summary with:
- Top N strategies overall
- Top strategy per symbol
- Key metrics comparison

## Statistical Filters

The system applies these filters by default:

| Filter | Default Threshold | Description |
|--------|-------------------|-------------|
| `min_trades` | 30 | Minimum trades for statistical significance |
| `min_sharpe` | 1.0 | Minimum Sharpe ratio |
| `min_profit_factor` | 1.2 | Gross profit / gross loss |
| `max_drawdown` | 30.0% | Maximum drawdown allowed |

### Adjusting Filters

```bash
# Lenient (more results)
py strategy_finder_runner.py --all --min-trades 10 --min-sharpe 0.5 --min-pf 0.8

# Strict (only best)
py strategy_finder_runner.py --all --min-trades 100 --min-sharpe 3.0 --min-pf 2.0
```

## Parameter Optimization

### Grid Search
The system exhaustively tests parameter combinations from predefined ranges. For example, RSI has:
- `rsi_period`: 5 values
- `oversold`: 5 values
- `overbought`: 5 values
- **Total combinations: 5 × 5 × 5 = 125**

### Sampling
When `--max-combos` is set, the system randomly samples combinations if the grid is too large:
```bash
# Only test 50 combinations
py strategy_finder_runner.py --all --max-combos 50
```

### Genetic Algorithm (Advanced)
For fine-tuning, use the `GeneticOptimizer` class directly:

```python
from backtest.strategy_finder import GeneticOptimizer
from backtest.strategy_finder.strategies import RSI_TEMPLATE

# Define fitness function
def fitness(params):
    # Run backtest with params
    result = run_backtest(RSI_TEMPLATE.strategy_func, params)
    return result['sharpe_ratio']

# Run GA
optimizer = GeneticOptimizer(
    param_ranges=RSI_TEMPLATE.param_ranges,
    fitness_func=fitness,
    population_size=50,
    generations=20,
)
best_params = optimizer.run()
```

## Walk-Forward Validation

The system supports walk-forward validation for robustness testing:

```python
from backtest.strategy_finder.statistical_filter import WalkForwardValidator

validator = WalkForwardValidator(
    n_folds=5,           # 5 rolling windows
    train_ratio=0.6,     # 60% train, 40% test
    min_positive_periods=4  # Require 4/5 positive test periods
)
```

## API Usage

### Python API

```python
from backtest.strategy_finder import (
    get_all_templates,
    get_templates_by_names,
    StrategyRanker,
    StatisticalFilter,
    StatisticalFilterConfig,
)
from backtest.unified_portfolio import UnifiedPortfolioConfig, UnifiedPortfolioBacktest

# Get templates
templates = get_templates_by_names(['RSI', 'MACD'])
print(f"Testing {len(templates)} strategies")

# Configure
config = UnifiedPortfolioConfig(
    initial_capital=10000,
    max_positions=5,
    position_size_pct=0.1,
    tp_pct=0.02,
    sl_pct=0.01,
    verbose=False,
)

# Run backtest
for template in templates:
    engine = UnifiedPortfolioBacktest(
        config=config,
        strategy=template.strategy_func,
        data_dir=r'C:\Personals\Code\backtest-with-bon',
    )
    engine.load_data(['BTCUSDT'])
    result = engine.run_backtest()
    print(f"{template.name}: Sharpe={result['metrics']['sharpe_ratio']:.2f}")

# Filter results
stat_filter = StatisticalFilter(StatisticalFilterConfig(
    min_trades=30,
    min_sharpe=1.0,
    min_profit_factor=1.2,
))
filter_result = stat_filter.apply_all_filters({
    'total_trades': 100,
    'sharpe_ratio': 2.5,
    'profit_factor': 1.8,
    'max_drawdown': 10.0,
    'win_rate': 55.0,
})
print(f"Passed: {filter_result['passed']}")
```

### Single Symbol Test

```python
from backtest.strategy_finder.strategies import RSI_TEMPLATE
from backtest.unified_portfolio import UnifiedPortfolioConfig, UnifiedPortfolioBacktest

# Create strategy with specific params
params = {'rsi_period': 14, 'oversold': 30, 'overbought': 70}

def wrapped_strategy(data):
    return RSI_TEMPLATE.strategy_func(data, **params)

engine = UnifiedPortfolioBacktest(
    config=UnifiedPortfolioConfig(verbose=False),
    strategy=wrapped_strategy,
    data_dir=r'C:\Personals\Code\backtest-with-bon',
)
engine.load_data(['BTCUSDT'])
result = engine.run_backtest()

print(result['metrics'])
```

## Data Requirements

### Directory Structure
```
data_dir/
├── 1h/
│   ├── BTCUSDT.csv
│   ├── ETHUSDT.csv
│   └── ...
└── 5m/
    ├── BTCUSDT.csv
    ├── ETHUSDT.csv
    └── ...
```

### CSV Format
Must have columns: `opentime, open, high, low, close, volume`
- `opentime` - Timestamp in milliseconds
- Prices - Numeric

### Symbol Discovery
The system automatically finds symbols with both 1h and 5m data:
```python
from backtest.strategy_finder import SymbolScanConfig, get_available_symbols

config = SymbolScanConfig(data_dir=r'C:\Personals\Code\backtest-with-bon')
symbols = get_available_symbols(config)
print(f"Found {len(symbols)} symbols")
```

## Performance Tips

### Speed Optimization
1. **Use `--quiet`** - Reduces logging overhead
2. **Limit `--max-combos`** - Fewer combinations = faster
3. **Fewer symbols** - Start with 1-5, scale up
4. **Parallel workers** - Increase `--parallel` on multi-core machines

```bash
# Fast scan
py strategy_finder_runner.py --symbols BTCUSDT --strategies RSI --max-combos 20 --quiet

# Balanced
py strategy_finder_runner.py --symbols BTCUSDT ETHUSDT --strategies RSI MACD --max-combos 100

# Thorough (slow)
py strategy_finder_runner.py --all --strategies RSI MACD MA_CROSSOVER --max-combos 300 --parallel 8
```

### Memory Management
For full scan with 583 symbols, consider:
- Running in batches
- Using `--quiet` to reduce memory for logs
- Processing symbols sequentially if memory constrained

## Troubleshooting

### No results found
- Lower thresholds (`--min-trades 10 --min-sharpe 0.5 --min-pf 0.8`)
- Check data files exist in correct directories
- Verify data format is correct

### Too slow
- Reduce `--max-combos`
- Fewer symbols
- Use `--quiet`
- Reduce parallel workers if memory constrained

### Data validation errors
- Ensure both 1h and 5m data exist for each symbol
- Check `opentime` column is in milliseconds
- Verify overlapping time ranges

### Import errors
```bash
# Ensure you're in project root
cd C:\Personals\Code\backtest-with-bon

# Test imports
py -c "from backtest.strategy_finder import get_all_templates; print(get_all_templates().keys())"
```

## Best Practices

1. **Start Small** - Test on 1-3 symbols first
2. **Iterate** - Start with lenient filters, tighten after seeing results
3. **Validate** - Use walk-forward analysis for robustness
4. **Diversify** - Test multiple strategies, not just one
5. **Document** - Save parameters and results for reproducibility

## Example Workflow

```bash
# Step 1: Quick scan to understand what works
py strategy_finder_runner.py --symbols BTCUSDT ETHUSDT SOLUSDT --strategies RSI MACD MA_CROSSOVER --max-combos 50 --quiet

# Step 2: Found promising strategies? Deep dive
py strategy_finder_runner.py --symbols BTCUSDT --strategies RSI --max-combos 200 --min-sharpe 1.0 --output btc_rsi_deep.csv

# Step 3: Full scan with best settings
py strategy_finder_runner.py --all --strategies RSI MACD --min-sharpe 1.5 --min-trades 50 --output best_strategies.csv

# Step 4: Analyze results
# Check profitable_strategies.csv for details
# Check strategy_finder_summary.txt for top picks
```
