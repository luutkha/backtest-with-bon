"""
Example: Using the Backtesting Framework

This example demonstrates how to use the clean architecture backtesting framework.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import BacktestEngine, BacktestConfig
from backtest.signals.signal_generator import (
    Signal, SignalType,
    rsi_strategy,
    moving_average_crossover_strategy,
    macd_strategy,
    sma, ema, rsi, macd, bollinger_bands
)
from backtest.execution import ExitPriority


# Example: Custom strategy function
def custom_bollinger_bounce_strategy(h1_data, bb_period=20, bb_std=2, rsi_period=14):
    """
    Bollinger Bands Bounce Strategy with RSI filter.
    - Long: Price touches lower BB and RSI crosses above 30
    - Short: Price touches upper BB and RSI crosses below 70
    """
    df = h1_data.copy()

    # Calculate indicators
    bb = bollinger_bands(df['close'], bb_period, bb_std)
    df['bb_upper'] = bb['upper']
    df['bb_middle'] = bb['middle']
    df['bb_lower'] = bb['lower']
    df['rsi'] = rsi(df['close'], rsi_period)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long: RSI crosses above 30 and price near lower BB
        if (previous['rsi'] <= 30 and current['rsi'] > 30 and
            current['close'] <= current['bb_lower'] * 1.02):
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={
                    'rsi': float(current['rsi']),
                    'bb_lower': float(current['bb_lower']),
                }
            ))

        # Short: RSI crosses below 70 and price near upper BB
        elif (previous['rsi'] >= 70 and current['rsi'] < 70 and
              current['close'] >= current['bb_upper'] * 0.98):
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={
                    'rsi': float(current['rsi']),
                    'bb_upper': float(current['bb_upper']),
                }
            ))

    print(f"Custom strategy generated {len(signals)} signals")
    return signals


def main():
    # Configuration
    DATA_DIR = r"C:\Personals\Code\backtest-with-bon"
    SYMBOL = "BTCUSDT"

    # Create backtest config
    config = BacktestConfig(
        data_dir=DATA_DIR,
        symbol=SYMBOL,
        initial_capital=10000.0,
        fee_rate=0.0004,           # 0.04% Binance taker fee
        slippage=0.0001,           # 0.01% slippage
        exit_priority=ExitPriority.CONSERVATIVE,
        tp_pct=0.02,               # 2% take profit
        sl_pct=0.01,               # 1% stop loss
        leverage=1.0,
        position_size_pct=0.95,
    )

    # Use built-in RSI strategy
    print("=" * 60)
    print("Running RSI Strategy Backtest")
    print("=" * 60)

    engine = BacktestEngine(config=config, strategy=rsi_strategy)

    # Run backtest with custom parameters
    results = engine.run_backtest()

    # Print summary
    engine.print_summary()

    # Save results
    output_dir = os.path.join(DATA_DIR, "output")
    engine.save_results(output_dir, prefix=f"{SYMBOL}_rsi")

    print("\n" + "=" * 60)
    print("Running MA Crossover Strategy Backtest")
    print("=" * 60)

    # Use MA Crossover strategy
    config2 = BacktestConfig(
        data_dir=DATA_DIR,
        symbol=SYMBOL,
        initial_capital=10000.0,
        fee_rate=0.0004,
        slippage=0.0001,
        exit_priority=ExitPriority.CONSERVATIVE,
        tp_pct=0.03,
        sl_pct=0.015,
        leverage=1.0,
        position_size_pct=0.95,
    )

    engine2 = BacktestEngine(config=config2, strategy=moving_average_crossover_strategy)
    results2 = engine2.run_backtest(params={'fast_period': 20, 'slow_period': 50})
    engine2.print_summary()

    # Save results
    engine2.save_results(output_dir, prefix=f"{SYMBOL}_ma_crossover")

    # Example: Custom strategy
    print("\n" + "=" * 60)
    print("Running Custom Bollinger Bounce Strategy")
    print("=" * 60)

    config3 = BacktestConfig(
        data_dir=DATA_DIR,
        symbol=SYMBOL,
        initial_capital=10000.0,
        fee_rate=0.0004,
        slippage=0.0001,
        exit_priority=ExitPriority.CONSERVATIVE,
        tp_pct=0.025,
        sl_pct=0.01,
        leverage=1.0,
        position_size_pct=0.95,
    )

    engine3 = BacktestEngine(config=config3, strategy=custom_bollinger_bounce_strategy)
    results3 = engine3.run_backtest(params={'bb_period': 20, 'bb_std': 2, 'rsi_period': 14})
    engine3.print_summary()

    engine3.save_results(output_dir, prefix=f"{SYMBOL}_bb_rsi")

    print("\nDone! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
