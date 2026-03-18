# Backtesting Framework

A production-grade backtesting framework for futures trading strategies using Python + vectorbt.

## Architecture

```
backtest/
├── __init__.py              # Package exports
├── backtest_engine.py       # Main orchestrator
├── config/
│   └── __init__.py         # BacktestConfig
├── data/
│   └── data_loader.py       # Data Layer: load, validate, align
├── signals/
│   └── signal_generator.py  # Signal Layer: strategies + indicators
├── execution/
│   └── execution_engine.py  # Execution Layer: intrabar simulation
├── portfolio/
│   └── portfolio_tracker.py # Portfolio Layer: capital tracking
├── analytics/
│   └── metrics_calculator.py # Analytics Layer (vectorbt)
└── reporting/
    └── report_generator.py  # Reporting Layer
```

## Quick Start

### Run Batch Test

```bash
# Single symbol
py batch_runner.py --symbols BTCUSDT --strategy rsi

# Multiple symbols
py batch_runner.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --strategy rsi

# All available symbols
py batch_runner.py --all --strategy rsi

# Custom TP/SL
py batch_runner.py --symbols BTCUSDT --strategy rsi --tp 0.03 --sl 0.015

# Custom leverage
py batch_runner.py --symbols BTCUSDT --strategy rsi --leverage 3

# Save results
py batch_runner.py --symbols BTCUSDT --output results.csv
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
from backtest.backtest_engine import create_rsi_backtest, create_ma_crossover_backtest

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
```

## Configuration

### BacktestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | "" | Data directory path |
| `symbol` | str | "BTCUSDT" | Trading symbol |
| `h1_timeframe` | str | "1h" | Higher timeframe for signals |
| `intrabar_timeframe` | str | "5m" | Intraday timeframe for simulation |
| `start_time` | int | None | Start time in milliseconds |
| `end_time` | int | None | End time in milliseconds |
| `initial_capital` | float | 10000 | Starting capital |
| `fee_rate` | float | 0.0004 | Fee per side (0.04%) |
| `slippage` | float | 0.0001 | Slippage % |
| `tp_pct` | float | 0.02 | Take profit % |
| `sl_pct` | float | 0.01 | Stop loss % |
| `leverage` | float | 1.0 | Leverage multiplier |
| `position_size_pct` | float | 0.95 | Position size % of capital |
| `trailing_stop_enabled` | bool | False | Enable trailing stop |
| `trailing_stop_pct` | float | 0.0 | Trailing stop percentage |
| `position_size_model` | PositionSizeModel | FIXED_PERCENT | Position sizing model |
| `risk_per_trade` | float | 0.02 | Risk per trade % |
| `exit_priority` | ExitPriority | CONSERVATIVE | TP/SL priority |

### Command Line Arguments

```bash
--symbols SYMBOLS    # Symbols to test
--all                # Run on all available symbols
--strategy STRATEGY  # rsi, ma, macd
--tp TP              # Take profit %
--sl SL              # Stop loss %
--leverage LEV       # Leverage
--capital CAP        # Initial capital
--output FILE        # Output CSV file
--parallel           # Run in parallel (default: True)
--workers N          # Max parallel workers (default: 4)
--quiet              # Suppress output
--data-dir PATH      # Data directory (default: C:\Personals\Code\backtest-with-bon)
```

## Strategies

### Built-in Strategies

1. **RSI Strategy** (`rsi`)
   - Long: RSI crosses above oversold (30)
   - Short: RSI crosses below overbought (70)
   - Parameters: `rsi_period`, `oversold`, `overbought`

2. **MA Crossover** (`ma`)
   - Long: Fast MA crosses above Slow MA
   - Short: Fast MA crosses below Slow MA
   - Parameters: `fast_period`, `slow_period`

3. **MACD** (`macd`)
   - Long: MACD crosses above signal line
   - Short: MACD crosses below signal line
   - Parameters: `fast_period`, `slow_period`, `signal_period`

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
SignalType.FLAT           # Close all positions
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

### Exit Reasons

| Reason | Description |
|--------|-------------|
| `tp` | Take profit |
| `sl` | Stop loss |
| `trailing_stop` | Trailing stop triggered |
| `signal_exit` | Exit signal generated |
| `end_of_data` | Position closed at end of data |
| `manual` | Manual exit |

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

- Reduce `--workers` to 1 for sequential processing
- Process symbols in smaller batches

## Performance Tips

1. **Parallel Processing**: Use `--parallel --workers 4` for faster batch processing
2. **Data Range**: Use `start_time` and `end_time` in config to limit data range
3. **Position Size**: Lower `position_size_pct` to reduce capital requirements
4. **Leverage**: Use leverage for smaller capital requirements (higher risk!)

## Requirements

```
pandas
numpy
vectorbt
```

Install:
```bash
pip install vectorbt pandas numpy
```

## Files

- `backtest_engine.py` - Main backtest engine with factory functions
- `batch_runner.py` - Batch runner script
- `example.py` - Example usage
- `sample_strategy.py` - Sample custom strategy
- `batch_results.csv` - Latest batch results
