"""
Sample Strategy: RSI + Moving Average Crossover
- Entry signals generated from 1h candle data
- Uses RSI and SMA for entry signals
"""

import pandas as pd
import numpy as np
from typing import List
from backtest_engine import Signal


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def sample_strategy(h1_data: pd.DataFrame) -> List[Signal]:
    """
    Sample strategy using RSI + SMA crossover

    Long entry: RSI crosses above 30 (oversold recovery)
    Short entry: RSI crosses below 70 (overbought decline)
    """
    df = h1_data.copy()

    # Calculate indicators
    df['sma_20'] = sma(df['close'], 20)
    df['rsi'] = rsi(df['close'], 14)

    # Generate signals
    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long signal: RSI crosses above 30 (oversold recovery)
        if (previous['rsi'] <= 30 and current['rsi'] > 30):
            signal = Signal(
                time=int(current['openTime']),
                action="long",
                price=float(current['close']),
                metadata={
                    'rsi': float(current['rsi']),
                }
            )
            signals.append(signal)

        # Short signal: RSI crosses below 70 (overbought decline)
        elif (previous['rsi'] >= 70 and current['rsi'] < 70):
            signal = Signal(
                time=int(current['openTime']),
                action="short",
                price=float(current['close']),
                metadata={
                    'rsi': float(current['rsi']),
                }
            )
            signals.append(signal)

    print(f"Generated {len(signals)} signals")
    return signals


if __name__ == "__main__":
    from backtest_engine import BacktestEngine, BacktestConfig, ExitPriority

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.0004,
        slippage=0.0001,
        exit_priority=ExitPriority.CONSERVATIVE,
        trailing_stop_enabled=False,
        tp_pct=0.02,
        sl_pct=0.01,
        position_size_pct=0.95,
        leverage=1.0,
    )

    # Create and run backtest
    engine = BacktestEngine(
        config=config,
        data_dir="C:\\Personals\\Code\\backtest-with-bon",
        symbol="BTCUSDT",
        strategy=sample_strategy
    )

    engine.run()

    # Get results as dictionary
    results = engine.get_results()
    print("\n=== RESULTS DICT ===")
    for key, value in results.items():
        if key != 'trades':
            print(f"{key}: {value}")
