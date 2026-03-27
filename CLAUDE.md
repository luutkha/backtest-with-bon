# Backtesting Framework

A production-grade backtesting framework for futures trading strategies using Python + vectorbt.

## Architecture

```
backtest/
├── __init__.py                    # Package exports
├── backtest_engine.py             # Main orchestrator (BacktestEngine + factory functions)
├── config/
│   └── __init__.py               # BacktestConfig, Timeframe enum
├── data/
│   └── data_loader.py            # DataLoader, DataConfig
├── signals/
│   ├── __init__.py               # Signal exports
│   ├── signal_generator.py       # SignalGenerator, Signal, SignalType + strategies + indicators
│   └── streak_breakout_strategy.py  # StreakConfig, streak_breakout_strategy, ma_streak_strategy
├── execution/
│   └── execution_engine.py       # ExecutionEngine, IntradaySimulator, Trade, enums
├── portfolio/
│   └── portfolio_tracker.py      # PortfolioTracker, PortfolioConfig, PositionSizeModel
├── analytics/
│   └── metrics_calculator.py     # MetricsCalculator (vectorbt)
├── reporting/
│   └── report_generator.py       # ReportGenerator
├── unified_portfolio.py          # UnifiedPortfolioBacktest, UnifiedPortfolioConfig
├── optimization/
│   ├── grid_search.py           # ParameterGridSearch
│   ├── walk_forward.py           # WalkForwardAnalysis
│   └── monte_carlo.py            # MonteCarloSimulator
└── strategy_finder/
    ├── strategies.py             # Strategy templates (RSI, MA_CROSSOVER, MACD, etc.)
    ├── genetic_optimizer.py     # Genetic algorithm optimizer
    ├── statistical_filter.py     # Statistical significance filters
    └── strategy_ranker.py        # Strategy ranking and output formatting
```

## Quick Start

### Run Batch Test (Unified Portfolio Mode)

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

### Run Strategy Finder (Automated Discovery)

```bash
# Find best strategies across all symbols
python strategy_finder_runner.py --all

# Run on specific symbols
python strategy_finder_runner.py --symbols BTCUSDT ETHUSDT

# Custom filtering
python strategy_finder_runner.py --all --min-sharpe 1.5 --min-trades 50

# Output top 10 strategies
python strategy_finder_runner.py --all --output results.csv --top-k 10
```

### Run Single Test

```python
from backtest import BacktestEngine, BacktestConfig
from backtest.signals import rsi_strategy

config = BacktestConfig(
    data_dir=r"C:\Personals\Code\backtest-with-bon",
    symbol="BTCUSDT",
    initial_capital=10000,
    tp_pct=0.02,
    sl_pct=0.01,
    leverage=1,
)

engine = BacktestEngine(config=config, strategy=rsi_strategy)
results = engine.run_backtest()

# Print summary
engine.print_summary()

# Access metrics
print(results['metrics']['sharpe_ratio'])
print(results['metrics']['max_drawdown'])
print(results['metrics']['profit_factor'])
```

### Factory Functions

```python
from backtest import create_rsi_backtest, create_ma_crossover_backtest, create_streak_breakout_backtest

# Quick RSI backtest
engine = create_rsi_backtest(
    data_dir=r"C:\Personals\Code\backtest-with-bon",
    symbol="BTCUSDT",
    rsi_period=14,
    oversold=30,
    overbought=70,
)

# Quick MA Crossover backtest
engine = create_ma_crossover_backtest(
    data_dir=r"C:\Personals\Code\backtest-with-bon",
    symbol="BTCUSDT",
    fast_period=20,
    slow_period=50,
)

# Quick Streak Breakout backtest
engine = create_streak_breakout_backtest(
    data_dir=r"C:\Personals\Code\backtest-with-bon",
    symbol="BTCUSDT",
    atr_window_min=1.0,
    atr_window_max=6.0,
    consecutive_candles=4,
    risk_reward_ratio_sl=0.5,
    risk_reward_ratio_tp=1.5,
)
```

## Configuration

### BacktestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | `""` | Data directory path |
| `symbol` | str | `"BTCUSDT"` | Trading symbol |
| `h1_timeframe` | str | `"1h"` | Higher timeframe for signals |
| `intrabar_timeframe` | str | `"5m"` | Intraday timeframe for simulation |
| `start_time` | int | `None` | Start time in milliseconds |
| `end_time` | int | `None` | End time in milliseconds |
| `initial_capital` | float | `10000.0` | Starting capital |
| `fee_rate` | float | `0.0004` | Fee per side (0.04%) |
| `slippage` | float | `0.0001` | Slippage % |
| `exit_priority` | ExitPriority | `CONSERVATIVE` | TP/SL priority |
| `tp_pct` | float | `0.02` | Take profit % |
| `sl_pct` | float | `0.01` | Stop loss % |
| `trailing_stop_enabled` | bool | `False` | Enable trailing stop |
| `trailing_stop_pct` | float | `0.0` | Trailing stop percentage |
| `leverage` | float | `1.0` | Leverage multiplier |
| `position_size_model` | PositionSizeModel | `FIXED_PERCENT` | Position sizing model |
| `position_size_pct` | float | `0.95` | Position size % of capital |
| `risk_per_trade` | float | `0.02` | Risk per trade % |
| `verbose` | bool | `True` | Enable logging |
| `skip_validation` | bool | `False` | Skip data validation for speed |

### UnifiedPortfolioConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | `100000.0` | Single capital pool for all symbols |
| `max_positions` | int | `10` | Max simultaneous trades |
| `position_size_pct` | float | `0.1` | 10% per position |
| `fee_rate` | float | `0.0004` | Fee per side |
| `slippage` | float | `0.0001` | Slippage % |
| `tp_pct` | float | `0.02` | Take profit % |
| `sl_pct` | float | `0.01` | Stop loss % |
| `leverage` | float | `1.0` | Leverage multiplier |
| `exit_priority` | ExitPriority | `CONSERVATIVE` | TP/SL priority |
| `verbose` | bool | `True` | Enable logging |

### Command Line Arguments (unified_portfolio_runner.py)

```bash
--symbols SYMBOLS    # Symbols to test (default: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT)
--all                 # Run on all available symbols
--strategy STRATEGY  # streak (currently only active strategy)
--tp TP              # Take profit % (default: 0.02)
--sl SL              # Stop loss % (default: 0.01)
--leverage LEV       # Leverage (default: 1.0)
--capital CAP        # Initial capital (default: 100000)
--max-positions N    # Max simultaneous positions (default: 10)
--position-size PCT  # Position size % (default: 0.1 = 10%)
--output FILE        # Output CSV file
--quiet              # Suppress output
--data-dir PATH      # Data directory (default: C:\Personals\Code\backtest-with-bon)
```

## Strategies

### Built-in Strategies

1. **RSI Strategy** (`rsi_strategy`)
   - Long: RSI crosses above oversold (default 30)
   - Short: RSI crosses below overbought (default 70)
   - Parameters: `rsi_period`, `oversold`, `overbought`

2. **MA Crossover** (`moving_average_crossover_strategy`)
   - Long: Fast MA crosses above Slow MA
   - Short: Fast MA crosses below Slow MA
   - Parameters: `fast_period`, `slow_period`

3. **MACD** (`macd_strategy`)
   - Long: MACD crosses above signal line
   - Short: MACD crosses below signal line
   - Parameters: `fast_period`, `slow_period`, `signal_period`

4. **Streak Breakout** (`streak_breakout_strategy`)
   - Long: After N consecutive green candles (default 4)
   - Short: After N consecutive red candles
   - Includes ATR filter for volatility confirmation
   - Optional volume and ADX filters
   - Parameters: `atr_window_min`, `atr_window_max`, `consecutive_candles`, `risk_reward_ratio_sl`, `risk_reward_ratio_tp`

5. **MA Streak** (`ma_streak_strategy`)
   - Combines streak breakout with MA filter
   - Only trades in direction of long-term trend

### Custom Strategy

```python
from backtest.signals.signal_generator import Signal, SignalType, rsi, sma

def my_strategy(h1_data):
    """
    Your custom strategy.

    Args:
        h1_data: DataFrame with columns [opentime, open, high, low, close, volume]

    Returns:
        List of Signal objects
    """
    df = h1_data.copy()

    # Calculate indicators
    df['rsi'] = rsi(df['close'], 14)
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Your signal logic
        if previous['rsi'] <= 30 and current['rsi'] > 30:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={'rsi': float(current['rsi'])}
            ))

        elif previous['rsi'] >= 70 and current['rsi'] < 70:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={'rsi': float(current['rsi'])}
            ))

    return signals
```

### Available Indicator Functions

```python
from backtest.signals.signal_generator import (
    sma, ema, rsi, macd, bollinger_bands,
    atr, stochastic, adx, cross_above, cross_below,
    volume_weighted_price
)
```

### Signal Types

```python
from backtest.signals.signal_generator import SignalType

SignalType.LONG_ENTRY    # Enter long position
SignalType.SHORT_ENTRY   # Enter short position
SignalType.LONG_EXIT     # Exit long position
SignalType.SHORT_EXIT    # Exit short position
SignalType.FLAT          # Close all positions
```

### Enums

```python
from backtest.config import Timeframe
Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1

from backtest.execution import PositionSide, ExitReason, ExitPriority, SlippageModel
PositionSide.LONG, PositionSide.SHORT
ExitReason.TAKE_PROFIT, ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP, ExitReason.MANUAL, ExitReason.ENDT_OF_DATA, ExitReason.SIGNAL_EXIT
ExitPriority.CONSERVATIVE (SL first), ExitPriority.AGGRESSIVE (TP first)
SlippageModel.FIXED, SlippageModel.VOLUME_BASED, SlippageModel.RANDOM

from backtest.portfolio import PositionSizeModel
PositionSizeModel.FIXED, PositionSizeModel.FIXED_PERCENT, PositionSizeModel.RISK_BASED
```

## Data Format

### Expected CSV Format

CSV files should have these columns:
- `opentime` - Timestamp in milliseconds
- `open` - Open price
- `high` - High price
- `low` - Low price
- `close` - Close price
- `volume` - Volume

### Data Directory Structure

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

## Metrics

The framework calculates these metrics:

| Metric | Description |
|--------|-------------|
| `total_trades` | Total number of trades |
| `winning_trades` | Number of winning trades |
| `losing_trades` | Number of losing trades |
| `win_rate` | Win rate % |
| `total_pnl` | Total PnL in dollars |
| `return_pct` | Return % |
| `max_drawdown` | Maximum drawdown % |
| `max_drawdown_dollar` | Maximum drawdown in dollars |
| `sharpe_ratio` | Sharpe ratio |
| `sortino_ratio` | Sortino ratio |
| `calmar_ratio` | Calmar ratio |
| `profit_factor` | Gross profit / gross loss |
| `expectancy` | Average trade expectancy |
| `avg_win` | Average win amount |
| `avg_loss` | Average loss amount |
| `avg_holding_bars` | Average holding time in bars |
| `total_fees` | Total fees paid |
| `long_trades` | Number of long trades |
| `long_win_rate` | Long trade win rate % |
| `short_trades` | Number of short trades |
| `short_win_rate` | Short trade win rate % |
| `max_consecutive_wins` | Maximum consecutive wins |
| `max_consecutive_losses` | Maximum consecutive losses |
| `volatility` | Annualized volatility % |
| `avg_trade_duration` | Average trade duration in hours |
| `exit_reasons` | Dict of exit reasons and counts |
| `avg_leverage` | Average leverage used |

### Exit Reasons

| Reason | Description |
|--------|-------------|
| `TAKE_PROFIT` | Take profit triggered |
| `STOP_LOSS` | Stop loss triggered |
| `TRAILING_STOP` | Trailing stop triggered |
| `SIGNAL_EXIT` | Exit signal generated |
| `ENDT_OF_DATA` | Position closed at end of data |
| `MANUAL` | Manual exit |

## Common Errors

### Error: Missing columns

```
ValueError: Missing required columns: {'opentime'}
```

**Solution**: Ensure CSV has lowercase column names: `opentime, open, high, low, close, volume`

### Error: File not found

```
FileNotFoundError: Data file not found: C:\...\1h\BTCUSDT.csv
```

**Solution**: Check that data files exist in `data_dir/1h/` and `data_dir/5m/`

### Error: No overlapping time range

```
ValueError: No overlapping time range between 1h and 5m data
```

**Solution**: Ensure both 1h and 5m data cover the same time period

### Error: No trades executed

- Check signal generation is working
- Verify TP/SL levels are reasonable
- Ensure data has price movement

### Memory issues with large datasets

- Process symbols in smaller batches
- Use `--quiet` flag to reduce memory overhead

## Performance Tips

1. **Quiet Mode**: Use `--quiet` flag for batch runs - skips validation and reduces logging overhead
2. **Data Range**: Use `start_time` and `end_time` in config to limit data range
3. **Position Size**: Lower `position_size_pct` to reduce capital requirements
4. **Leverage**: Use leverage for smaller capital requirements (higher risk!)

The framework uses vectorized numpy operations for intrabar simulation (~10x faster than iterative approaches).

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
```

Install:
```bash
pip install pandas numpy
```

## Files

- `backtest/__init__.py` - Package exports
- `backtest/backtest_engine.py` - Main backtest engine with factory functions
- `backtest/unified_portfolio.py` - Unified portfolio backtest implementation
- `batch_runner.py` - CLI runner for unified portfolio mode
- `strategy_finder_runner.py` - Automated strategy discovery runner
- `example.py` - Example usage
- `sample_strategy.py` - Sample custom strategy (old architecture)
- `backtest_engine.py` - Legacy single-file backtest engine (old architecture)

## Unified Portfolio Mode

The framework runs in unified portfolio mode where all symbols share a single capital pool.

### Running Unified Portfolio

```bash
# Example: 20 max positions, 5% per position
py batch_runner.py --all --strategy streak \
    --capital 100000 --max-positions 20 --position-size 0.05
```

**Key Features:**

| Feature | Value |
|---------|-------|
| Capital | $100k shared (default) |
| Positions | Max 20 (default: 10) |
| Position Size | 10% per trade (default) |
| Risk | Correlated across symbols |

**Unified Portfolio API:**

```python
from backtest import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from backtest.signals import rsi_strategy

config = UnifiedPortfolioConfig(
    initial_capital=100000,    # Single pool for all
    max_positions=20,         # Max simultaneous trades
    position_size_pct=0.05,    # 5% of capital per trade
    tp_pct=0.02,
    sl_pct=0.01,
)

engine = UnifiedPortfolioBacktest(
    config=config,
    strategy=rsi_strategy,
    data_dir=r"C:\Personals\Code\backtest-with-bon"
)

engine.load_data(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
results = engine.run_backtest()
engine.print_summary()
```

## Results Structure

```python
{
    'config': {...},           # Configuration dict
    'metrics': {...},          # All calculated metrics
    'trades': [...],           # List of Trade objects
    'trades_df': DataFrame,    # Trades as DataFrame
    'equity_df': DataFrame,    # Equity curve
    'final_capital': float
}
```
